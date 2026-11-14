"""Run the three-strategy ingestion and retrieval benchmark."""
import argparse
import csv
import importlib.metadata
import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, gold_evidence_options, summarize
from .ingestion import load_pdf
from .models import Chunk, DocumentPage
from .store import PgVectorStore

MIN_CONTENT_PAGES = 30
MIN_QUESTIONS = 20
MIN_DEV_QUESTIONS = 15
MIN_TEST_QUESTIONS = 5
FILTER_RESERVED_FIELDS = {
    "document_id", "source", "page", "has_text", "content_page",
    "start_char", "end_char", "content_pages",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSONL file not found: {path}")
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL line must be an object")
            records.append(record)
    return records


def _read_benchmark(data_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    jsonl_path = data_dir / "benchmark.jsonl"
    json_path = data_dir / "benchmark.json"
    if jsonl_path.exists() and json_path.exists():
        raise ValueError(
            "Found both benchmark.json and benchmark.jsonl; keep one benchmark input"
        )
    if jsonl_path.exists():
        return {}, _read_jsonl(jsonl_path)
    if json_path.exists():
        try:
            benchmark = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{json_path}: invalid JSON: {exc.msg}") from exc
        if not isinstance(benchmark, dict) or not isinstance(
            benchmark.get("questions"), list
        ):
            raise ValueError(
                f"{json_path}: expected an object containing a questions array"
            )
        if any(not isinstance(question, dict) for question in benchmark["questions"]):
            raise ValueError(f"{json_path}: each question must be an object")
        return benchmark, benchmark["questions"]
    raise FileNotFoundError(
        f"Required benchmark file not found: {jsonl_path} or {json_path}"
    )


def _load_corpus(
    data_dir: Path, benchmark_source: dict[str, Any] | None = None
) -> list[DocumentPage]:
    manifest_path = data_dir / "manifest.jsonl"
    documents = _read_jsonl(manifest_path) if manifest_path.exists() else []
    if not documents:
        pdfs = sorted(data_dir.rglob("*.pdf"))
        source_id = (
            benchmark_source.get("source_id")
            if benchmark_source and len(pdfs) == 1
            else None
        )
        source_url = (
            benchmark_source.get("citation_url")
            if benchmark_source and len(pdfs) == 1
            else None
        )
        documents = [
            {
                "document_id": (
                    source_id
                    if isinstance(source_id, str) and source_id.strip()
                    else pdf.relative_to(data_dir).with_suffix("").as_posix()
                ),
                "path": pdf.relative_to(data_dir).as_posix(),
                "source": (
                    source_url
                    if isinstance(source_url, str) and source_url.strip()
                    else pdf.relative_to(data_dir).as_posix()
                ),
            }
            for pdf in pdfs
        ]
    if not documents:
        raise FileNotFoundError(
            f"No PDFs found in {data_dir}; add PDFs or create manifest.jsonl"
        )

    pages = []
    seen_ids = set()
    for row in documents:
        document_id = row.get("document_id")
        relative_path = row.get("path")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("each manifest entry needs a non-empty document_id")
        if document_id in seen_ids:
            raise ValueError(f"duplicate document_id in manifest: {document_id}")
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError(f"manifest entry {document_id} needs a PDF path")
        if not isinstance(row.get("metadata", {}), dict):
            raise ValueError(f"manifest metadata for {document_id} must be an object")

        pdf_path = Path(relative_path).expanduser()
        if not pdf_path.is_absolute():
            pdf_path = data_dir / pdf_path
        source = row.get("source", relative_path.replace("\\", "/"))
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"manifest source for {document_id} must be a non-empty string")
        content_pages = row.get("content_pages")
        if content_pages is not None and (
            not isinstance(content_pages, list)
            or any(type(page) is not int or page < 1 for page in content_pages)
        ):
            raise ValueError(f"content_pages for {document_id} must be a list of page numbers")
        if content_pages is not None and len(set(content_pages)) != len(content_pages):
            raise ValueError(f"content_pages for {document_id} contains duplicate pages")

        extracted = load_pdf(pdf_path, document_id, source)
        allowed_pages = set(content_pages) if content_pages is not None else None
        if allowed_pages and max(allowed_pages) > len(extracted):
            raise ValueError(
                f"content_pages for {document_id} includes a page beyond the "
                f"PDF's {len(extracted)} physical pages"
            )
        document_metadata = dict(row.get("metadata", {}))
        for page in extracted:
            page.metadata = {
                **document_metadata,
                **page.metadata,
                "content_page": (
                    bool(page.text) and
                    (allowed_pages is None or page.page in allowed_pages)
                ),
            }
        pages.extend(extracted)
        seen_ids.add(document_id)
    return pages


def _apply_declared_filter_metadata(
    pages: list[DocumentPage], benchmark: dict[str, Any]
) -> dict[str, Any] | None:
    specification = benchmark.get("filter_metadata")
    if specification is None:
        return None
    if not isinstance(specification, dict):
        raise ValueError("benchmark filter_metadata must be an object")
    field = specification.get("field")
    field_type = specification.get("type")
    derivation = specification.get("derivation", "")
    if not isinstance(derivation, str):
        raise ValueError("benchmark filter_metadata.derivation must be a string")
    derivation = derivation.casefold()
    if field == "chapter" and field_type == "integer":
        if "heading" not in derivation or "number" not in derivation:
            raise ValueError(
                "chapter metadata derivation must use explicit numbered headings"
            )
        heading_pattern = re.compile(r"^\s*(\d{1,2})\.\d{1,2}\s+[A-Z]")
    elif field == "section" and field_type == "string":
        if "heading" not in derivation or "section" not in derivation:
            raise ValueError(
                "section metadata derivation must use explicit numbered section headings"
            )
        heading_pattern = re.compile(r"^\s*(\d{1,2}\.\d{1,2})\s+[A-Z]")
    else:
        raise ValueError(
            "filter_metadata must declare supported chapter/integer or section/string metadata"
        )

    active_document = None
    current_value = None
    for page in pages:
        if page.document_id != active_document:
            active_document = page.document_id
            current_value = None
        if not page.metadata.get("content_page", True):
            continue
        for line in page.text.splitlines():
            match = heading_pattern.match(line)
            if match:
                current_value = (
                    int(match.group(1)) if field == "chapter" else match.group(1)
                )
                break
        if current_value is not None:
            page.metadata[field] = current_value

    values = sorted(
        {page.metadata[field] for page in pages if field in page.metadata}
    )
    return {
        "field": field,
        "pages_with_value": sum(field in page.metadata for page in pages),
        "distinct_values": values,
    }


def _has_content_page_manifest(data_dir: Path) -> bool:
    manifest_path = data_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        return False
    records = _read_jsonl(manifest_path)
    return bool(records) and all(
        isinstance(record.get("content_pages"), list)
        and bool(record["content_pages"])
        for record in records
    )


def _normalized(value: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFC", value).casefold(),
    ).strip()


def _validate_questions(questions: list[dict[str, Any]],
                        pages: list[DocumentPage]) -> None:
    if len(questions) < MIN_QUESTIONS:
        raise ValueError(f"benchmark needs at least {MIN_QUESTIONS} questions")
    question_ids = set()
    split_counts = {"dev": 0, "test": 0}
    filtered_test_count = 0
    filtered_count = 0
    page_lookup = {(page.document_id, page.page): page for page in pages}

    for question in questions:
        question_id = question.get("id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("every benchmark question needs a non-empty id")
        if question_id in question_ids:
            raise ValueError(f"duplicate benchmark question id: {question_id}")
        question_ids.add(question_id)
        if question.get("split") not in split_counts:
            raise ValueError(f"question {question_id} split must be dev or test")
        split_counts[question["split"]] += 1
        if not isinstance(question.get("question"), str) or not question["question"].strip():
            raise ValueError(f"question {question_id} needs question text")
        if not isinstance(question.get("expected_answer"), str) or not question["expected_answer"].strip():
            raise ValueError(f"question {question_id} needs expected_answer")

        options = gold_evidence_options(question)
        for option in options:
            page = page_lookup.get((option["source_id"], option["page"]))
            if page is None:
                raise ValueError(
                    f"question {question_id} gold page not found: "
                    f"{option['source_id']} page {option['page']}"
                )
            if _normalized(option["quote"]) not in _normalized(page.text):
                raise ValueError(
                    f"question {question_id} evidence quote is not present on its gold page"
                )

        filters = question.get("filter")
        if filters is None:
            continue
        if not isinstance(filters, dict) or not filters:
            raise ValueError(f"question {question_id} filter must be a non-empty object")
        filtered_count += 1
        if question["split"] == "test":
            filtered_test_count += 1
        for key in filters:
            if not isinstance(key, str) or key in FILTER_RESERVED_FIELDS:
                raise ValueError(
                    f"question {question_id} filter must use a real custom metadata field"
                )
        if not any(
            page.metadata.get("content_page")
            and all(page.metadata.get(key) == value for key, value in filters.items())
            for page in pages
        ):
            raise ValueError(
                f"question {question_id} filter has no matching eligible page "
                "in the corpus metadata"
            )

    if split_counts["dev"] < MIN_DEV_QUESTIONS:
        raise ValueError(f"benchmark needs at least {MIN_DEV_QUESTIONS} dev questions")
    if split_counts["test"] < MIN_TEST_QUESTIONS:
        raise ValueError(f"benchmark needs at least {MIN_TEST_QUESTIONS} test questions")
    if filtered_count < 3 or filtered_test_count < 1:
        raise ValueError("benchmark needs at least 3 filtered questions, including 1 test question")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _print_step(step: str, **details: Any) -> None:
    print(
        json.dumps(
            {"step": step, "status": "complete", **details},
            ensure_ascii=False,
        ),
        flush=True,
    )


def _write_ingestion_errors(results_dir: Path, pages: list[DocumentPage]) -> None:
    path = results_dir / "ingestion_errors.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        fields = ["document_id", "source", "page", "issue"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for page in pages:
            if not page.text:
                writer.writerow(
                    {
                        "document_id": page.document_id,
                        "source": page.source,
                        "page": page.page,
                        "issue": "No extractable text (blank page, scan, or extraction limitation)",
                    }
                )


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _chunk_statistics(chunks: list[Chunk]) -> dict[str, Any]:
    lengths = sorted(len(chunk.text) for chunk in chunks)
    if not lengths:
        return {
            "chunks": 0,
            "average_chunk_characters": 0,
            "minimum_chunk_characters": 0,
            "median_chunk_characters": 0,
            "p95_chunk_characters": 0,
            "maximum_chunk_characters": 0,
            "total_chunk_characters": 0,
        }
    return {
        "chunks": len(lengths),
        "average_chunk_characters": sum(lengths) / len(lengths),
        "minimum_chunk_characters": lengths[0],
        "median_chunk_characters": statistics.median(lengths),
        "p95_chunk_characters": lengths[math.ceil(0.95 * len(lengths)) - 1],
        "maximum_chunk_characters": lengths[-1],
        "total_chunk_characters": sum(lengths),
    }


def _configured_cost_rate() -> float | None:
    value = os.getenv("EMBEDDING_COST_USD_PER_1M_TOKENS", "").strip()
    if not value:
        return None
    try:
        rate = float(value)
    except ValueError as exc:
        raise ValueError(
            "EMBEDDING_COST_USD_PER_1M_TOKENS must be a non-negative number"
        ) from exc
    if not math.isfinite(rate) or rate < 0:
        raise ValueError(
            "EMBEDDING_COST_USD_PER_1M_TOKENS must be a non-negative number"
        )
    return rate


def _write_manual_audits(results_dir: Path, traces: list[dict[str, Any]],
                         chunks_by_strategy: dict[str, list[Chunk]],
                         pages: list[DocumentPage]) -> None:
    grounding_path = results_dir / "manual_grounding_metadata.csv"
    with grounding_path.open("w", encoding="utf-8-sig", newline="") as file:
        fields = [
            "strategy", "question_id", "question", "rank", "chunk_id",
            "document_id", "source", "page", "metadata_json", "filter_applied",
            "filter_satisfied", "score", "metadata_correct_0_1",
            "grounding_score_0_2", "rationale",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        test_counts: dict[str, int] = {}
        for trace in traces:
            if trace["split"] != "test":
                continue
            strategy = trace["strategy"]
            if test_counts.get(strategy, 0) >= 5:
                continue
            test_counts[strategy] = test_counts.get(strategy, 0) + 1
            hits = trace["hits"] or [None]
            for rank, hit in enumerate(hits, start=1):
                writer.writerow(
                    {
                        "strategy": strategy,
                        "question_id": trace["question_id"],
                        "question": trace["question"],
                        "rank": rank if hit else "",
                        "chunk_id": hit["chunk_id"] if hit else "",
                        "document_id": hit["document_id"] if hit else "",
                        "source": hit["source"] if hit else "",
                        "page": hit["page"] if hit else "",
                        "metadata_json": json.dumps(
                            hit["metadata"], ensure_ascii=False
                        ) if hit else "",
                        "filter_applied": json.dumps(
                            trace["filter"], ensure_ascii=False
                        ) if strategy == "C_paragraph_filter" and trace["filter"] else "",
                        "filter_satisfied": (
                            int(all(
                                hit["metadata"].get(key) == value
                                for key, value in trace["filter"].items()
                            ))
                            if hit and strategy == "C_paragraph_filter" and trace["filter"]
                            else ""
                        ),
                        "score": hit["score"] if hit else "",
                    }
                )

    coherence_path = results_dir / "manual_coherence.csv"
    sample_pages = sorted(
        (page for page in pages if page.metadata.get("content_page")),
        key=lambda page: (page.document_id, page.page),
    )[:10]
    with coherence_path.open("w", encoding="utf-8-sig", newline="") as file:
        fields = [
            "strategy", "document_id", "page", "sample_start_char",
            "chunk_id", "chunk_text", "coherence_score_0_2", "rationale",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for strategy, chunks in chunks_by_strategy.items():
            by_page: dict[tuple[str, int], list[Chunk]] = {}
            for chunk in chunks:
                by_page.setdefault((chunk.document_id, chunk.page), []).append(chunk)
            for page in sample_pages:
                candidates = [
                    chunk for chunk in by_page.get((page.document_id, page.page), [])
                    if chunk.metadata["start_char"] <= 0 < chunk.metadata["end_char"]
                ]
                chunk = candidates[0] if candidates else None
                writer.writerow(
                    {
                        "strategy": strategy,
                        "document_id": page.document_id,
                        "page": page.page,
                        "sample_start_char": 0,
                        "chunk_id": chunk.chunk_id if chunk else "",
                        "chunk_text": chunk.text if chunk else "",
                    }
                )


def main() -> None:
    starter_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate inputs and chunking without reading .env, calling APIs, or writing to the database",
    )
    args = parser.parse_args()
    if args.validate_only:
        cost_per_million_tokens = None
    else:
        load_dotenv(starter_dir / ".env")
        cost_per_million_tokens = _configured_cost_rate()

    data_dir = starter_dir / "data"
    results_dir = starter_dir / "results"
    if not args.validate_only:
        results_dir.mkdir(parents=True, exist_ok=True)
    run_started_at = datetime.now(timezone.utc).isoformat()
    benchmark, questions = _read_benchmark(data_dir)
    content_page_selection_declared = _has_content_page_manifest(data_dir)
    ingestion_started = time.perf_counter()
    pages = _load_corpus(data_dir, benchmark.get("source"))
    filter_metadata = _apply_declared_filter_metadata(pages, benchmark)
    if not args.validate_only:
        _write_ingestion_errors(results_dir, pages)
    content_pages = [
        page for page in pages
        if page.text and page.metadata.get("content_page", False)
    ]
    read_details: dict[str, Any] = {
        "physical_pages": len(pages),
        "content_pages": len(content_pages),
        "content_page_selection_declared": content_page_selection_declared,
        "pages_without_extractable_text": sum(not page.text for page in pages),
        "total_content_characters": sum(len(page.text) for page in content_pages),
        "filter_metadata": filter_metadata,
    }
    if not args.validate_only:
        read_details["errors_report"] = str(results_dir / "ingestion_errors.csv")
    _print_step("read_and_normalize", **read_details)
    if len(content_pages) < MIN_CONTENT_PAGES:
        raise ValueError(
            f"Corpus has {len(content_pages)} eligible content pages; "
            f"at least {MIN_CONTENT_PAGES} are required"
        )
    if not content_page_selection_declared:
        raise ValueError(
            "Submission runs require starter/data/manifest.jsonl with an explicit "
            "content_pages list for every document. Review the physical PDF pages "
            "and exclude covers, blank pages, and references before running embeddings."
        )

    _validate_questions(questions, content_pages)
    _print_step(
        "validate_benchmark",
        questions=len(questions),
        dev=sum(question["split"] == "dev" for question in questions),
        test=sum(question["split"] == "test" for question in questions),
        filtered_questions=sum(bool(question.get("filter")) for question in questions),
    )

    chunks_by_strategy = {
        "A_fixed": [
            chunk
            for page in content_pages
            for chunk in fixed_size(page, size=500, overlap=50)
        ],
        "B_paragraph": [
            chunk
            for page in content_pages
            for chunk in by_paragraph(page, max_size=700)
        ],
    }
    chunks_by_strategy["C_paragraph_filter"] = chunks_by_strategy["B_paragraph"]
    if not chunks_by_strategy["A_fixed"] or not chunks_by_strategy["B_paragraph"]:
        raise ValueError("Chunking produced no chunks; check that the corpus contains text")
    _print_step(
        "chunking",
        strategies={
            name: _chunk_statistics(chunks)
            for name, chunks in chunks_by_strategy.items()
        },
    )
    if args.validate_only:
        _print_step(
            "preflight_complete",
            api_calls=0,
            database_writes=0,
            result_files_written=0,
        )
        return

    model_name = os.getenv("EMBEDDING_MODEL", "").strip() or "text-embedding-3-small"
    embedder = MyEmbedder(model_name)
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise ValueError("DATABASE_URL is missing; configure starter/.env")
    store_info = PgVectorStore.prepare_database(database_url)
    _print_step(
        "vector_store_ready",
        store={"engine": "PostgreSQL + pgvector", **store_info},
    )

    unique_texts = list(
        dict.fromkeys(
            chunk.text
            for strategy in ("A_fixed", "B_paragraph")
            for chunk in chunks_by_strategy[strategy]
        )
    )
    embedding_started = time.perf_counter()
    unique_vectors = embedder.embed_documents(unique_texts)
    embedding_seconds = time.perf_counter() - embedding_started
    _print_step(
        "document_embeddings",
        provider=embedder.provider,
        model=embedder.model_name,
        dimensions=embedder.dimensions,
        unique_texts=len(unique_texts),
        vectors=len(unique_vectors),
        elapsed_seconds=embedding_seconds,
    )
    vectors_by_text = dict(zip(unique_texts, unique_vectors))
    vectors_by_strategy = {
        "A_fixed": [
            vectors_by_text[chunk.text] for chunk in chunks_by_strategy["A_fixed"]
        ],
        "B_paragraph": [
            vectors_by_text[chunk.text] for chunk in chunks_by_strategy["B_paragraph"]
        ],
    }
    vectors_by_strategy["C_paragraph_filter"] = vectors_by_strategy["B_paragraph"]

    store = PgVectorStore(database_url, embedder.dimensions)
    store_write_started = time.perf_counter()
    for strategy, chunks in chunks_by_strategy.items():
        store.upsert(strategy, chunks, vectors_by_strategy[strategy])
    store_write_seconds = time.perf_counter() - store_write_started
    ingestion_seconds = time.perf_counter() - ingestion_started
    _print_step(
        "vector_store_upsert",
        store={"engine": "PostgreSQL + pgvector", **store_info},
        rows_by_strategy={
            strategy: len(chunks) for strategy, chunks in chunks_by_strategy.items()
        },
        elapsed_seconds=store_write_seconds,
    )

    traces = []
    retrieval_seconds = 0.0
    query_embedding_seconds = 0.0
    query_vectors = {}
    for question in questions:
        query = question["question"]
        if query not in query_vectors:
            query_started = time.perf_counter()
            query_vectors[query] = embedder.embed_query(query)
            query_embedding_seconds += time.perf_counter() - query_started
    _print_step(
        "query_embeddings",
        unique_queries=len(query_vectors),
        elapsed_seconds=query_embedding_seconds,
    )
    for strategy in chunks_by_strategy:
        for question in questions:
            filters = (
                question.get("filter") or {}
                if strategy == "C_paragraph_filter"
                else {}
            )
            started = time.perf_counter()
            hits = store.search(
                strategy,
                query_vectors[question["question"]],
                top_k=3,
                filters=filters,
            )
            retrieval_seconds += time.perf_counter() - started
            trace = evaluate_query(question, hits)
            traces.append({"strategy": strategy, **trace})

    _print_step(
        "retrieval",
        traces=len(traces),
        elapsed_seconds=retrieval_seconds,
        metrics=summarize(traces),
    )
    _write_jsonl(results_dir / "traces.jsonl", traces)
    strategy_trace_paths = []
    for strategy in chunks_by_strategy:
        strategy_path = results_dir / f"traces_{strategy}.jsonl"
        strategy_traces = [
            trace for trace in traces if trace["strategy"] == strategy
        ]
        _write_jsonl(strategy_path, strategy_traces)
        strategy_trace_paths.append(str(strategy_path))
    metrics = summarize(traces)
    with (results_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    strategy_stats = {
        strategy: _chunk_statistics(chunks)
        for strategy, chunks in chunks_by_strategy.items()
    }
    embedding_tokens = (
        embedder.total_input_tokens if embedder.token_usage_complete else None
    )
    cost_usd = (
        0.0
        if cost_per_million_tokens == 0
        else (
            cost_per_million_tokens * embedding_tokens / 1_000_000
            if cost_per_million_tokens is not None and embedding_tokens is not None
            else None
        )
    )
    metadata_fields = sorted(
        {
            key
            for page in content_pages
            for key in page.metadata
            if key not in FILTER_RESERVED_FIELDS
        }
    )
    metadata = {
        "run_started_at": run_started_at,
        "python_version": sys.version.split()[0],
        "pypdf_version": _package_version("pypdf"),
        "psycopg_version": _package_version("psycopg"),
        "embedding_provider": embedder.provider,
        "embedding_model": embedder.model_name,
        "embedding_dimensions": embedder.dimensions,
        "document_embedding_task": (
            "RETRIEVAL_DOCUMENT (Gemini) or same-model provider default"
        ),
        "query_embedding_task": (
            "RETRIEVAL_QUERY (Gemini) or same-model provider default"
        ),
        "embedding_tokens": embedding_tokens,
        "embedding_token_usage_complete": embedder.token_usage_complete,
        "content_pages": len(content_pages),
        "content_page_selection_declared": content_page_selection_declared,
        "total_content_characters": sum(len(page.text) for page in content_pages),
        "metadata_fields": metadata_fields,
        "questions": len(questions),
        "unique_document_texts_embedded": len(unique_texts),
        "embedding_seconds": embedding_seconds,
        "query_embedding_seconds": query_embedding_seconds,
        "store_write_seconds": store_write_seconds,
        "ingestion_seconds": ingestion_seconds,
        "retrieval_seconds": retrieval_seconds,
        "cost_usd": cost_usd,
        "cost_per_1m_input_tokens_usd": cost_per_million_tokens,
        "cost_note": (
            "Cost is computed only when a provider price is configured and token usage is reported."
        ),
        "vector_store": {
            "engine": "PostgreSQL + pgvector",
            **store_info,
            "metric": "cosine",
            "search": "exact",
        },
        "strategies": strategy_stats,
    }
    with (results_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    _write_manual_audits(results_dir, traces, chunks_by_strategy, content_pages)
    _print_step(
        "write_results",
        files=[
            str(results_dir / "traces.jsonl"),
            *strategy_trace_paths,
            str(results_dir / "summary.json"),
            str(results_dir / "run_metadata.json"),
            str(results_dir / "ingestion_errors.csv"),
            str(results_dir / "manual_grounding_metadata.csv"),
            str(results_dir / "manual_coherence.csv"),
        ],
    )
    print(json.dumps({"metrics": metrics, "run_metadata": metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
