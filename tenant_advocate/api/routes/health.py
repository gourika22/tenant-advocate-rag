"""
tenant_advocate/api/routes/health.py
-----------------------------------------
GET /health — liveness + knowledge base readiness check.
Called by Render for health checks and by the Streamlit sidebar.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from tenant_advocate.core.rag_engine import get_engine_status

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    api_configured: bool
    knowledge_base_ready: bool
    knowledge_base_count: int
    knowledge_base_message: str
    model: str
    embedding_model: str
    jurisdiction: str


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Returns the system readiness status.
    HTTP 200 always — degraded state is communicated in the response body.
    """
    s = get_engine_status()
    return HealthResponse(
        status="ok" if s["api_configured"] and s["knowledge_base_ready"] else "degraded",
        **s,
    )
