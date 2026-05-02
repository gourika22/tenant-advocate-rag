"""
tenant_advocate/api/routes/audit.py
----------------------------------------
POST /audit — Mode 2: Proactive Lease Audit (Novel Feature 1, streaming SSE)

Accepts a PDF file upload, parses it, and streams the Tenant Risk Report.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from tenant_advocate.core.lease_parser import parse_lease_bytes
from tenant_advocate.core.rag_engine import run_lease_audit

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
async def audit_lease(file: UploadFile = File(...)) -> StreamingResponse:
    """
    Upload a NSW lease PDF -> stream back the structured Tenant Risk Report.

    The lease is parsed in memory.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()

    if len(pdf_bytes) > 10 * 1024 * 1024:   # 10 MB limit
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10 MB.")

    parsed = parse_lease_bytes(pdf_bytes, filename=file.filename)

    if not parsed.is_ready:
        raise HTTPException(
            status_code=422,
            detail=parsed.warning or "Could not extract text from the uploaded PDF.",
        )

    return StreamingResponse(
        _sse_stream(run_lease_audit(parsed.full_text)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
