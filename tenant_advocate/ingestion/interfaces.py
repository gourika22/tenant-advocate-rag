"""
tenant_advocate/ingestion/interfaces.py
----------------------------------------
Integration layer between RAG engine and Vector DB API.

Vector DB API: https://tenant-advocate-vdb.onrender.com
Docs:          https://tenant-advocate-vdb.onrender.com/docs
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from loguru import logger

# ── Vector DB API config ──────────────────────────────────────────────────────

VDB_BASE_URL = "https://tenant-advocate-vdb.onrender.com"
VDB_TIMEOUT  = 30   # seconds — Render free tier can be slow on cold start

# ── Shared data types ───────────

@dataclass
class RetrievedChunk:
    """A single NSW law chunk returned by the vector search."""
    content:     str
    source_file: str
    page:        int | str
    score:       float
    doc_type:    str = "legislation"
    section:     str = ""


@dataclass
class StoreStatus:
    """Knowledge base readiness — displayed in the Streamlit sidebar."""
    ready:        bool
    vector_count: int
    message:      str


# ── API — search function ────────────────────────────────────────────

def search_laws(query: str, top_k: int = 6) -> list[RetrievedChunk]:
    """
    Embed the query and retrieve the top-k most relevant NSW law chunks
    from the vector database.

    Calls: POST https://tenant-advocate-vdb.onrender.com/query

    Args:
        query:  The user's question or audit topic.
        top_k:  Number of results to retrieve.

    Returns:
        List of RetrievedChunk, sorted by relevance (best first).
        Returns empty list if the API is unreachable — never raises.
    """
    try:
        response = httpx.post(
            f"{VDB_BASE_URL}/query",
            json={
                "query": query,
                "top_k": top_k,
            },
            timeout=VDB_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        chunks: list[RetrievedChunk] = []

        for result in data.get("results", []):
            if not result.get("document", "").strip():
                continue 
            meta = result.get("metadata", {})

            # Distance → similarity score
            # This API returns cosine distance (lower = more similar)
            # Convert to similarity score (higher = more similar) for consistency
            distance = result.get("distance", 1.0)
            score    = round(1 - distance, 4)

            chunks.append(RetrievedChunk(
                content=result.get("document", ""),
                source_file=meta.get("source_file", "unknown"),
                page=meta.get("page", "?"),
                score=score,
                doc_type=meta.get("doc_type", "legislation"),
                section=str(meta.get("section", "")),
            ))

        logger.info(
            f"[VDB] Retrieved {len(chunks)} chunks for query: '{query[:60]}'"
        )
        return chunks

    except httpx.TimeoutException:
        logger.warning(
            "[VDB] Request timed out. The vector DB may be cold-starting on Render. "
            "Returning empty results — answer will use general knowledge fallback."
        )
        return []

    except httpx.HTTPStatusError as exc:
        logger.error(f"[VDB] HTTP error {exc.response.status_code}: {exc}")
        return []

    except Exception as exc:  # noqa: BLE001
        logger.error(f"[VDB] Unexpected error: {exc}")
        return []


def search_laws_filtered(
    query: str,
    top_k: int = 5,
    filters: dict | None = None,
) -> list[RetrievedChunk]:
    """
    Search with optional metadata filters.
    Uses POST /query/where for simple key=value filters.

    Example — search only within legislation documents:
        search_laws_filtered("bond amount", filters={"doc_type": "legislation"})

    Example — search within a specific part:
        search_laws_filtered("repairs", filters={"part": "Part 3"})
    """
    if not filters:
        return search_laws(query, top_k)

    try:
        response = httpx.post(
            f"{VDB_BASE_URL}/query/where",
            json={
                "query":   query,
                "top_k":   top_k,
                "filters": filters,
            },
            timeout=VDB_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        chunks: list[RetrievedChunk] = []
        for result in data.get("results", []):
            if not result.get("document", "").strip():
                continue 
            meta     = result.get("metadata", {})
            # Cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 = identical, -1 = opposite
            # We clamp to 0 minimum since negative similarity isn't useful
            distance = result.get("distance", 1.0)
            score    = round(max(0.0, 1 - distance), 4)

            chunks.append(RetrievedChunk(
                content=result.get("document", ""),
                source_file=meta.get("source_file", "unknown"),
                page=meta.get("page", "?"),
                score=score,
                doc_type=meta.get("doc_type", "legislation"),
                section=str(meta.get("section", "")),
            ))

        logger.info(
            f"[VDB] Filtered search: {len(chunks)} chunks | filters={filters}"
        )
        return chunks

    except Exception as exc:  # noqa: BLE001
        logger.error(f"[VDB] Filtered search error: {exc}")
        return search_laws(query, top_k)   # fall back to unfiltered


# ── Knowledge base status ─────────────────────────────────────────────────────

def get_store_status() -> StoreStatus:
    """
    Check if the vector DB is live and return document count.
    Calls: GET https://tenant-advocate-vdb.onrender.com/health
    """
    try:
        response = httpx.get(
            f"{VDB_BASE_URL}/health",
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        count = data.get("document_count", 0)

        return StoreStatus(
            ready=True,
            vector_count=count,
            message=f"{count:,} NSW law clauses indexed",
        )

    except httpx.TimeoutException:
        return StoreStatus(
            ready=False,
            vector_count=0,
            message="Vector DB is starting up — please wait 30 seconds and refresh",
        )

    except Exception as exc:  # noqa: BLE001
        logger.error(f"[VDB] Health check failed: {exc}")
        return StoreStatus(
            ready=False,
            vector_count=0,
            message=f"Vector DB unavailable: {exc}",
        )