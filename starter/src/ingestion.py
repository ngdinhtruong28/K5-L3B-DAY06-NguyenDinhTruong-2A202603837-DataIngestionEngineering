"""Read and normalize source data."""
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract every physical PDF page and mark pages without extractable text."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    if not document_id.strip() or not source.strip():
        raise ValueError("document_id and source must not be empty")

    reader = PdfReader(str(path))
    pages = []
    for page_number, pdf_page in enumerate(reader.pages, start=1):
        text = normalize_text(pdf_page.extract_text() or "")
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata={"has_text": bool(text)},
            )
        )
    return pages


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraph boundaries."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.split("\n")]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()
