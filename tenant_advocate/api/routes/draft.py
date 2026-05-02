"""
tenant_advocate/api/routes/draft.py
----------------------------------------
POST /draft — Mode 3: Communication Drafting Assistant (streaming SSE)

Accepts a situation description + optional lease text and streams a
grounded draft communication with a mandatory pre-send checklist.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from tenant_advocate.core.lease_parser import parse_lease_bytes
from tenant_advocate.core.rag_engine import generate_draft

router = APIRouter()


def _sse_stream(generator):
    try:
        for token in generator:
            safe = token.replace("\n", "\\n")
            yield f"data: {safe}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        yield f"data: [ERROR] {exc}\n\n"


@router.post("")
async def communication_draft(
    situation: str = Form(..., min_length=10, max_length=2000),
    tenant_name: str = Form(default="[YOUR NAME]", max_length=100),
    landlord_name: str = Form(default="[LANDLORD/AGENT NAME]", max_length=100),
    file: UploadFile | None = File(default=None),
) -> StreamingResponse:
    """
    Generate a grounded draft communication for the tenant to review and personalise.

    Form fields:
      situation     — Plain-English description of the dispute/situation (required)
      tenant_name   — Used as a placeholder in the draft closing (optional)
      landlord_name — Used as a placeholder in the draft salutation (optional)
      file          — Lease PDF for clause cross-referencing (optional)
    """
    if not situation.strip():
        raise HTTPException(status_code=422, detail="Situation cannot be empty.")

    # Parse optional lease
    lease_text: str | None = None
    if file is not None and file.filename:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=422, detail="Only PDF files are accepted.")
        pdf_bytes = await file.read()
        parsed = parse_lease_bytes(pdf_bytes, filename=file.filename)
        if parsed.is_ready:
            lease_text = parsed.get_context_text()

    return StreamingResponse(
        _sse_stream(generate_draft(
            situation=situation,
            lease_text=lease_text,
            tenant_name=tenant_name or "[YOUR NAME]",
            landlord_name=landlord_name or "[LANDLORD/AGENT NAME]",
        )),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
