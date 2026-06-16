"""LLM OCR flow (spec section 15).

Fast path: extract embedded text from PDFs/DOCX/TXT. When a page is image-only
or text is too sparse (scanned report), render it with PyMuPDF and send the
image to a vision-capable OpenRouter model. Returns the OCR output schema.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import llm

# Below this many characters on a page, treat it as scanned and use vision OCR.
MIN_TEXT_CHARS_PER_PAGE = 40
MAX_VISION_PAGES = 8

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff")

_OCR_SYSTEM = (
    "You are an OCR and document transcription engine for clinical documents. "
    "Transcribe ALL legible text from the provided page image(s) faithfully, "
    "preserving numbers, units, dates, and table structure as plain text. Do not "
    "summarize, interpret, or invent content."
)


def _pdf_text_pages(path: Path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [(page.extract_text() or "").strip() for page in reader.pages]


def _docx_text(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    return "\n\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())


def _render_pdf_pages_to_data_urls(path: Path, max_pages: int) -> list[str]:
    import fitz  # PyMuPDF

    urls: list[str] = []
    doc = fitz.open(str(path))
    try:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(dpi=150)
            png = pix.tobytes("png")
            b64 = base64.b64encode(png).decode("ascii")
            urls.append(f"data:image/png;base64,{b64}")
    finally:
        doc.close()
    return urls


def _image_to_data_url(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    return f"data:image/{mime};base64,{b64}"


def run_ocr(path: Path, force_vision: bool = False) -> dict:
    """Extract text from a file.

    Returns ``{extracted_text, page_count, method, confidence, token_usage}``.
    Structured clinical-entity extraction is done separately by
    ``clinical_extraction.extract_entities`` over ``extracted_text``.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    token_usage: list[dict] = []

    # --- Images: always vision OCR. ---
    if suffix in IMAGE_SUFFIXES:
        text, usage = llm.invoke_vision(
            step="llm_ocr",
            system=_OCR_SYSTEM,
            user_text="Transcribe all text from this clinical document image.",
            image_data_urls=[_image_to_data_url(path)],
        )
        token_usage.append(usage)
        return {
            "extracted_text": text.strip(),
            "page_count": 1,
            "method": "vision",
            "confidence": 0.85 if text.strip() else 0.0,
            "token_usage": token_usage,
        }

    # --- DOCX / TXT: text only. ---
    if suffix in (".docx", ".doc"):
        text = _docx_text(path)
        return {
            "extracted_text": text,
            "page_count": 1,
            "method": "text",
            "confidence": 0.95 if text.strip() else 0.0,
            "token_usage": token_usage,
        }
    if suffix in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        return {
            "extracted_text": text,
            "page_count": 1,
            "method": "text",
            "confidence": 0.99 if text.strip() else 0.0,
            "token_usage": token_usage,
        }

    # --- PDF: text-first, vision fallback per page. ---
    if suffix == ".pdf":
        pages = _pdf_text_pages(path)
        sparse = force_vision or any(
            len(p) < MIN_TEXT_CHARS_PER_PAGE for p in pages
        )
        if not sparse and any(pages):
            return {
                "extracted_text": "\n\n".join(pages).strip(),
                "page_count": len(pages),
                "method": "text",
                "confidence": 0.95,
                "token_usage": token_usage,
            }
        # Vision OCR for scanned/sparse PDFs.
        urls = _render_pdf_pages_to_data_urls(path, MAX_VISION_PAGES)
        if not urls:
            return {
                "extracted_text": "\n\n".join(pages).strip(),
                "page_count": len(pages),
                "method": "text",
                "confidence": 0.4,
                "token_usage": token_usage,
            }
        text, usage = llm.invoke_vision(
            step="llm_ocr",
            system=_OCR_SYSTEM,
            user_text=(
                "Transcribe all text from these clinical document pages, in order."
            ),
            image_data_urls=urls,
        )
        token_usage.append(usage)
        return {
            "extracted_text": text.strip(),
            "page_count": len(pages) or len(urls),
            "method": "vision",
            "confidence": 0.85 if text.strip() else 0.0,
            "token_usage": token_usage,
        }

    raise ValueError(f"Unsupported file type: {path.suffix}")
