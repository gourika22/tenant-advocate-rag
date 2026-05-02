"""
tenant_advocate/api/main.py
--------------------------------
FastAPI application entry point.

Endpoints:
  GET  /health              — liveness + knowledge base status
  POST /chat                — streaming Q&A  (Mode 1)
  POST /audit               — lease risk audit  (Mode 2)
  POST /draft               — communication draft  (Mode 3)

All endpoints stream responses using Server-Sent Events (SSE) so the
Streamlit frontend can display tokens as they arrive.

CORS is configured to accept requests from the Streamlit Cloud origin.
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tenant_advocate.api.routes import audit, chat, draft, health
from tenant_advocate.config import settings

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Tenant Rights Advocate API",
    description=(
        "Agentic RAG backend for NSW tenant rights information. "
        "Provides grounded, citation-enforced answers based on the "
        "Residential Tenancies Act 2010 (NSW)."
    ),
    version="1.0.0",
    docs_url="/docs",      
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(chat.router,   prefix="/chat",  tags=["Q&A"])
app.include_router(audit.router,  prefix="/audit", tags=["Lease Audit"])
app.include_router(draft.router,  prefix="/draft", tags=["Communication Draft"])


# ── Dev server entry point ────────────────────────────────────────────────────

def serve() -> None:
    """Called by `poetry run serve` for local development."""
    uvicorn.run(
        "tenant_advocate.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    serve()
