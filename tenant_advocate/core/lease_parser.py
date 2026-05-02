"""
tenant_advocate/core/lease_parser.py
-----------------------------------------
Parses an uploaded NSW lease PDF into structured plain text.

Privacy-by-design: parsed text is held in FastAPI request scope only.
It is not written to disk, never logged, and never indexed into ChromaDB.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

import pdfplumber
from loguru import logger

_MAX_CHARS = 20_000     # ~2 500 tokens — leaves room for law context + system prompt


@dataclass
class ParsedLease:
    filename: str
    full_text: str
    page_count: int
    char_count: int
    extraction_success: bool
    warning: str | None = None
    pages: list[str] = field(default_factory=list)

    def get_context_text(self) -> str:
        """Return lease text truncated to fit the LLM context window."""
        if len(self.full_text) <= _MAX_CHARS:
            return self.full_text
        return (
            self.full_text[:_MAX_CHARS]
            + f"\n\n[... lease truncated at {_MAX_CHARS:,} chars for context window ...]"
        )

    @property
    def is_ready(self) -> bool:
        return self.extraction_success and bool(self.full_text.strip())


def parse_lease_bytes(pdf_bytes: bytes, filename: str = "lease.pdf") -> ParsedLease:
    """
    Extract text from raw PDF bytes.
    Used by the FastAPI endpoint which receives the file as bytes.
    Never raises — errors surface in ParsedLease.warning.
    """
    page_texts: list[str] = []
    total_pages = 0

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        with pdfplumber.open(tmp_path) as pdf:
            total_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    page_texts.append(f"[Page {i + 1}]\n{text.strip()}")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"pdfplumber failed for '{filename}': {exc}")
        return ParsedLease(
            filename=filename, full_text="", page_count=0,
            char_count=0, extraction_success=False,
            warning=f"Could not parse this PDF: {exc}. Please use a text-based PDF.",
        )
    finally:
        os.unlink(tmp_path)

    full_text = "\n\n".join(page_texts)

    if not full_text.strip():
        return ParsedLease(
            filename=filename, full_text="", page_count=total_pages,
            char_count=0, extraction_success=False,
            warning="No text extracted — this may be a scanned image PDF.",
        )

    logger.info(f"Lease parsed: '{filename}' | {total_pages}pp | {len(full_text):,} chars")
    return ParsedLease(
        filename=filename, full_text=full_text,
        page_count=total_pages, char_count=len(full_text),
        extraction_success=True, pages=page_texts,
    )
