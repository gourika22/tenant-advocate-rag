"""
tenant_advocate/core/rag_engine.py
---------------------------------------
Agentic RAG Orchestration Engine

Public API (called by FastAPI routers):
  stream_chat_answer()  — Mode 1: Reactive Q&A
  run_lease_audit()     — Mode 2: Proactive lease risk scan 
  generate_draft()      — Mode 3: Communication drafting assistant
  get_engine_status()   — Health check

All three modes follow the same pattern:
  1. Call search_laws() to retrieve relevant NSW law chunks 
  2. Build a grounded prompt using the retrieved context
  3. Stream GPT-4o response tokens back to the caller
"""

from __future__ import annotations

from typing import Generator, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from tenant_advocate.config import settings
from tenant_advocate.core.prompts import (
    AUDIT_SYSTEM_PROMPT,
    CHAT_SYSTEM_PROMPT,
    DRAFT_SYSTEM_PROMPT,
    build_audit_user_message,
    build_chat_user_message,
    build_draft_user_message,
)
from tenant_advocate.ingestion.interfaces import (
    RetrievedChunk,
    StoreStatus,
    get_store_status,
    search_laws,
)


def _get_llm(*, streaming: bool = True) -> ChatOpenAI:
    """Return a configured GPT-4o instance."""
    return ChatOpenAI(
        model=settings.chat_model,
        temperature=0.1,        # Low temperature: factual legal answers
        streaming=streaming,
        openai_api_key=settings.openai_api_key,
    )

# ── Mode 1: Reactive Q&A ──────────────────────────────────────────────────────

def stream_chat_answer(
    question: str,
    lease_text: str | None = None,
    chat_history: list[tuple[str, str]] | None = None,
) -> Generator[str, None, None]:
    """
    Retrieve NSW law context, build a grounded prompt, stream GPT-4o tokens.

    Args:
        question:     User's plain-English question.
        lease_text:   Extracted lease text (from ParsedLease.get_context_text()).
        chat_history: Prior (question, answer) tuples for conversational context.

    Yields:
        String tokens from OpenAI streaming response.
    """
    law_chunks: list[RetrievedChunk] = search_laws(question, top_k=settings.top_k_results)
    logger.info(f"[CHAT] {len(law_chunks)} NSW law chunk(s) retrieved | '{question[:60]}'")

    messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT)]

    # Last 4 conversation turns for continuity
    for entry in (chat_history or [])[-4:]:
        if len(entry) == 2:
            prior_q, prior_a = entry[0], entry[1]
        else:
            continue
        messages.append(HumanMessage(content=prior_q))
        messages.append(SystemMessage(content=prior_a))
    
    # Truncate lease to 6000 chars for chat context
    # Enough for most relevant clauses without hitting token limits
    lease_context = lease_text[:6000] if lease_text else None

    messages.append(HumanMessage(content=build_chat_user_message(
        question=question,
        law_chunks=law_chunks,
        lease_text=lease_context,
    )))

    for chunk in _get_llm(streaming=True).stream(messages):
        if chunk.content:
            yield chunk.content


# ── Mode 2: Proactive Lease Audit ───────────────────────────
def run_lease_audit(lease_text: str) -> Generator[str, None, None]:
    """
    Proactively scan every clause in an uploaded NSW lease and classify it.

    This is the core novel contribution:
      - Existing tools (ChatGPT) only answer questions reactively.
      - This mode scans the full lease WITHOUT the tenant needing to know what to ask.
      - Output: structured Tenant Risk Report with ILLEGAL/UNFAIR/STANDARD/FAVOURABLE
        labels and Act citations for every flagged clause.

    Multi-query retrieval: auditing requires broad law coverage, so we run
    three distinct queries covering the main areas of NSW tenancy law and
    deduplicate the results before passing them to the LLM.

    Args:
        lease_text: Full text extracted from the uploaded lease PDF.

    Yields:
        Streaming Markdown tokens of the Tenant Risk Report.
    """
    # Multi-query retrieval for broad NSW law coverage
    audit_queries = [
        "bond entry notice inspections rent increase termination eviction",
        "tenant obligations subletting pets alterations damage",
        "landlord responsibilities urgent repairs maintenance habitability",
        "rent payment method frequency receipt landlord bank account",
        "end of tenancy vacating cleaning bond return condition report",
        "locks security keys access privacy safety smoke alarms",
        "water usage charges utilities electricity gas bills",
        "domestic violence termination safety protection provisions",
        "discrimination protected grounds tenancy application refusal",
        "strata body corporate rules tenant common property",
    ]

    law_chunks = []
    seen = set()
    for q in audit_queries:
        for chunk in search_laws(q, top_k=3):   # reduced from 5 to 3 to save tokens
            if chunk.content not in seen:
                law_chunks.append(chunk)
                seen.add(chunk.content)

    logger.info(f"[AUDIT] {len(law_chunks)} law chunks retrieved")

    # Truncate lease to 8000 chars — enough for a standard residential lease
    # while keeping total prompt within safe token limits
    truncated_lease = lease_text[:8000]
    if len(lease_text) > 8000:
        truncated_lease += "\n\n[... remainder of lease not shown — audit based on clauses above ...]"

    messages = [
        SystemMessage(content=AUDIT_SYSTEM_PROMPT),
        HumanMessage(content=build_audit_user_message(truncated_lease, law_chunks)),
    ]

    for chunk in _get_llm(streaming=True).stream(messages):
        if chunk.content:
            yield chunk.content

# ── Mode 3: Communication Drafting Assistant ────────────────

def generate_draft(
    situation: str,
    lease_text: str | None = None,
    tenant_name: str = "[YOUR NAME]",
    landlord_name: str = "[LANDLORD/AGENT NAME]",
) -> Generator[str, None, None]:
    """
    Generate a draft communication grounded in retrieved NSW tenancy law.

    Safety design (ethical review + Tenants' Union of NSW):
      - Framed as a DRAFT, not a finished legal document
      - Output includes a mandatory pre-send checklist
      - Every Act citation sourced from retrieved context, not LLM memory
      - NCAT Procedural Direction 7 warning included in every output

    Args:
        situation:      Plain-English description of the tenant's situation.
        lease_text:     Optional lease text for specific clause cross-referencing.
        tenant_name:    Placeholder used in draft closing.
        landlord_name:  Placeholder used in draft salutation.

    Yields:
        Streaming Markdown tokens of the draft communication + checklist.
    """
    law_chunks = search_laws(situation, top_k=settings.top_k_results)
    logger.info(f"[DRAFT] {len(law_chunks)} NSW law chunks retrieved | '{situation[:60]}'")

    messages = [
        SystemMessage(content=DRAFT_SYSTEM_PROMPT),
        HumanMessage(content=build_draft_user_message(
            situation=situation,
            law_chunks=law_chunks,
            lease_text=lease_text,
            tenant_name=tenant_name,
            landlord_name=landlord_name,
        )),
    ]

    for chunk in _get_llm(streaming=True).stream(messages):
        if chunk.content:
            yield chunk.content


# ── Engine status (for API health endpoint) ───────────────────────────────────

def get_engine_status() -> dict:
    """Return a health/readiness dict. Always safe — never raises."""
    kb: StoreStatus = get_store_status()
    api_ok = bool(settings.openai_api_key and settings.openai_api_key.startswith("sk-"))

    return {
        "api_configured":         api_ok,
        "knowledge_base_ready":   kb.ready,
        "knowledge_base_count":   kb.vector_count,
        "knowledge_base_message": kb.message,
        "model":                  settings.chat_model,
        "embedding_model":        settings.embedding_model,
        "jurisdiction":           settings.jurisdiction,
    }
