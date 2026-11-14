"""Fixed-size and paragraph-aware chunking with source traceability."""
import hashlib
import re

from .models import Chunk, DocumentPage


def _chunk(page: DocumentPage, text: str, strategy: str, ordinal: int,
           start: int, end: int) -> Chunk:
    identity = "\0".join(
        (strategy, page.document_id, str(page.page), str(ordinal), str(start), str(end), text)
    )
    chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    metadata = dict(page.metadata)
    metadata.update({"start_char": start, "end_char": end})
    return Chunk(
        chunk_id=chunk_id,
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=metadata,
    )


def _windows(page: DocumentPage, text: str, size: int, overlap: int,
             strategy: str, start_offset: int = 0) -> list[Chunk]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        value = text[start:end].strip()
        if value:
            chunks.append(
                _chunk(page, value, strategy, len(chunks),
                       start_offset + start, start_offset + end)
            )
        if end == len(text):
            break
        start = end - overlap
    return chunks


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split text into fixed character windows with a configurable overlap."""
    if size <= 0:
        raise ValueError("size must be greater than zero")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")
    return _windows(page, page.text, size, overlap, "fixed")


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Pack whole paragraphs up to max_size and window oversized paragraphs."""
    if max_size <= 0:
        raise ValueError("max_size must be greater than zero")

    paragraphs = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", page.text, re.DOTALL):
        paragraphs.append((match.start(), match.end(), match.group().strip()))

    chunks = []
    pending = []
    pending_start = 0
    pending_end = 0

    def flush() -> None:
        nonlocal pending
        if pending:
            text = "\n\n".join(item[2] for item in pending)
            chunks.append(_chunk(page, text, "paragraph", len(chunks),
                                 pending_start, pending_end))
            pending = []

    for start, end, paragraph in paragraphs:
        if len(paragraph) > max_size:
            flush()
            oversized = _windows(page, paragraph, max_size, 0, "paragraph", start)
            for item in oversized:
                item.chunk_id = hashlib.sha256(
                    f"{item.chunk_id}\0{len(chunks)}".encode("utf-8")
                ).hexdigest()
                chunks.append(item)
            continue

        proposed = len(paragraph) if not pending else (
            sum(len(item[2]) for item in pending) + 2 * len(pending) + len(paragraph) + 2
        )
        if pending and proposed > max_size:
            flush()
        if not pending:
            pending_start = start
        pending.append((start, end, paragraph))
        pending_end = end

    flush()
    return chunks
