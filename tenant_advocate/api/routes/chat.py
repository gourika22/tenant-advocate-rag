"""
tenant_advocate/api/routes/chat.py
---------------------------------------
POST /chat — Mode 1: Reactive Q&A (streaming SSE)

The Streamlit frontend calls this endpoint and iterates over
the Server-Sent Event stream to display tokens as they arrive.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from tenant_advocate.core.rag_engine import stream_chat_answer

router = APIRouter()

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    lease_text: str | None = Field(default=None, max_length=2000000)
    chat_history: list[list[str]] = Field(default_factory=list)


def _sse_stream(generator):
    try:
        for token in generator:
            safe = token.replace("\n", "\\n")
            yield f"data: {safe}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        yield f"data: [ERROR] {exc}\n\n"


@router.post("")
def chat(request: ChatRequest) -> StreamingResponse:
    """
    Stream a grounded NSW tenancy law answer token-by-token.

    Response: text/event-stream
    Each event: data: <token>\\n\\n
    Final event: data: [DONE]\\n\\n
    """
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    generator = stream_chat_answer(
        question=request.question,
        lease_text=request.lease_text,
        chat_history=request.chat_history,
    )

    return StreamingResponse(
        _sse_stream(generator),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
