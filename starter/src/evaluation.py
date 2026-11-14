"""Evaluate source-and-evidence retrieval and aggregate benchmark results."""
import re
import unicodedata
from typing import Any


def _normalized(value: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFC", value).casefold(),
    ).strip()


def gold_evidence_options(question: dict[str, Any]) -> list[dict[str, Any]]:
    options = [
        {
            "source_id": question.get("gold_source_id"),
            "page": question.get("gold_page"),
            "quote": question.get("evidence_quote"),
        }
    ]
    alternatives = question.get(
        "alternative_gold_evidence", question.get("gold_evidence", [])
    )
    if alternatives is None:
        alternatives = []
    if not isinstance(alternatives, list):
        raise ValueError("alternative_gold_evidence must be a list of evidence objects")
    for item in alternatives:
        if not isinstance(item, dict):
            raise ValueError("each alternative gold evidence item must be an object")
        options.append(
            {
                "source_id": item.get("gold_source_id", item.get("source_id")),
                "page": item.get("gold_page", item.get("page")),
                "quote": item.get("evidence_quote", item.get("quote")),
            }
        )
    for option in options:
        if (
            not isinstance(option["source_id"], str)
            or not option["source_id"].strip()
            or not isinstance(option["page"], int)
            or not isinstance(option["quote"], str)
            or not option["quote"].strip()
        ):
            raise ValueError(
                "each gold evidence option needs source ID, integer page, and evidence quote"
            )
    return options


def _matches(hit: dict[str, Any], options: list[dict[str, Any]]) -> bool:
    metadata = hit.get("metadata") or {}
    source_id = hit.get("document_id", metadata.get("document_id"))
    page = hit.get("page", metadata.get("page"))
    text = _normalized(str(hit.get("text", "")))
    for option in options:
        if source_id != option["source_id"] or page != option["page"]:
            continue
        if _normalized(option["quote"]) in text:
            return True
    return False


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check page/source and exact gold evidence without inventing human labels."""
    options = gold_evidence_options(question)
    top_hits = hits[:3]
    matched = [_matches(hit, options) for hit in top_hits]
    return {
        "question_id": question["id"],
        "split": question["split"],
        "question": question["question"],
        "filter": question.get("filter"),
        "gold_evidence": options,
        "hits": top_hits,
        "hit_at_1": bool(matched and matched[0]),
        "evidence_hit_at_3": any(matched),
        "matched_chunk_ids": [
            hit["chunk_id"] for hit, is_match in zip(top_hits, matched) if is_match
        ],
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 counts and rates by strategy and split."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for trace in traces:
        strategy = trace["strategy"]
        split = trace["split"]
        grouped.setdefault(strategy, {}).setdefault(split, []).append(trace)

    summary = {}
    for strategy, splits in sorted(grouped.items()):
        summary[strategy] = {}
        for split, rows in sorted(splits.items()):
            count = len(rows)
            hit1 = sum(bool(row["hit_at_1"]) for row in rows)
            hit3 = sum(bool(row["evidence_hit_at_3"]) for row in rows)
            summary[strategy][split] = {
                "questions": count,
                "hit_at_1": hit1,
                "hit_at_1_rate": hit1 / count,
                "evidence_hit_at_3": hit3,
                "evidence_hit_at_3_rate": hit3 / count,
            }
    return summary
