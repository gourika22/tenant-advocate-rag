# tenant-advocate-rag

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

An end-to-end Agentic RAG application that empowers renters to understand their legal rights. The system allows users to upload their lease agreements, ask questions in plain English, and receive legally grounded summaries cross-referenced against state tenancy laws.

**Agentic RAG Backend — NSW Tenant Rights Advocate**

This repository contains the FastAPI backend and RAG orchestration engine for the NSW Tenant Rights Advocate application. It is one of three repositories that make up the full system:

| Repository | Purpose |
|---|---|
| `tenant-advocate-vdb` | Vector database API — document ingestion, embedding, ChromaDB storage and retrieval  |
| **`tenant-advocate-rag`** | **RAG orchestration backend — prompt engineering, LLM orchestration, FastAPI endpoints** |
| `tenant-advocate-streamlit` | Streamlit frontend — user interface |

This README covers **only the RAG backend**. See the other repositories for the vector database and frontend documentation.

---

## What this repository does

This service sits between the vector database and the frontend. It:

1. Receives a user question (or lease PDF) via HTTP from the Streamlit frontend
2. Queries the vector database API (`tenant-advocate-vdb`) to retrieve the most relevant NSW law chunks
3. Combines retrieved law context with the user's uploaded lease text
4. Constructs a strictly grounded prompt with 7 anti-hallucination rules enforced
5. Streams GPT-4o responses token by token back to the frontend via Server-Sent Events (SSE)

### Three application modes

| Mode | Endpoint | Description |
|---|---|---|
| **Q&A** | `POST /chat` | User asks a question — system retrieves relevant NSW law + cross-references their lease |
| **Lease Audit** | `POST /audit` | User uploads a lease — system proactively scans every clause and classifies it as Illegal / Unfair / Standard / Favourable |
| **Communication Draft** | `POST /draft` | User describes a dispute — system generates a grounded draft communication with Act citations and a mandatory pre-send safety checklist |

---

## Architecture

```
[Streamlit Frontend]
        │
        │  HTTP POST (JSON or multipart/form-data)
        ▼
[FastAPI — this repo]
        │
   ┌────┴────┐
   │         │
   ▼         ▼
[Vector DB   [OpenAI GPT-4o]
 API]              │
   │               │
   │  top-5 NSW    │  streamed tokens
   │  law chunks   │
   └────┬──────────┘
        │
        ▼
  SSE stream back to frontend
  (token by token)
```

### Retrieval strategy by mode

```
Chat mode:
  1 semantic query -> Vector DB -> top-5 chunks + lease text -> GPT-4o

Audit mode:
  10 targeted queries -> Vector DB -> deduplicated -> up to 30 unique chunks -> GPT-4o
  Queries cover: bond, entry, repairs, rent, termination, security,
                 water, domestic violence, discrimination, strata

Draft mode:
  1 situation query -> Vector DB -> top-5 chunks + lease text -> GPT-4o
```

---

## Repository structure

```
tenant-advocate-rag/
│
├── pyproject.toml                        # Poetry config, dependencies, scripts
├── .env.example                          # Environment variable template
├── render.yaml                           # Render deployment config
├── Procfile                              # Process file for deployment
│
├── tenant_advocate/                      # Main Python package
│   ├── config.py                         # Pydantic Settings — all config in one place
│   │
│   ├── api/                              # FastAPI layer
│   │   ├── main.py                       # App entry point, CORS config, router registration
│   │   └── routes/
│   │       ├── health.py                 # GET /health — liveness + KB status
│   │       ├── chat.py                   # POST /chat — streaming Q&A
│   │       ├── audit.py                  # POST /audit — lease risk audit
│   │       └── draft.py                  # POST /draft — communication draft
│   │
│   ├── core/                             # RAG orchestration
│   │   ├── prompts.py                    # All system prompts — 3 modes, 7 rules
│   │   ├── rag_engine.py                 # Orchestration logic for all 3 modes
│   │   └── lease_parser.py               # PDF -> plain text extraction
│   │
│   └── ingestion/
│       └── interfaces.py                 # Integration contract with tenant-advocate-vdb
│
└── tests/
    ├── unit/
    │   └── test_api_routes.py            # FastAPI endpoint tests (mocked LLM)
    └── integration/
```

---

## Endpoints

### `GET /health`

Returns system readiness status. Called by the Streamlit sidebar on every page load.

**Response:**
```json
{
  "status": "ok",
  "api_configured": true,
  "knowledge_base_ready": true,
  "knowledge_base_count": 2085,
  "knowledge_base_message": "2,085 NSW law clauses indexed",
  "model": "gpt-4o",
  "embedding_model": "text-embedding-3-small",
  "jurisdiction": "New South Wales, Australia"
}
```

---

### `POST /chat`

Streams a grounded NSW tenancy law answer.

**Request body (JSON):**
```json
{
  "question": "Can my landlord enter my home without notice?",
  "lease_text": "optional extracted lease text string or null",
  "chat_history": [
    ["Prior question?", "Prior answer."]
  ]
}
```

**Response:** `text/event-stream` — tokens streamed as SSE events.

```
data: No\n\n
data:  —\n\n
data:  your landlord cannot enter without notice.\n\n
data: [DONE]\n\n
```

---

### `POST /audit`

Accepts a lease PDF and streams back a structured Tenant Risk Report.

**Request:** `multipart/form-data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `file` | PDF | Yes | Max 10MB, text-based PDF |

**Response:** `text/event-stream` — streams a Markdown report with Illegal / Unfair / Standard / Favourable classification per clause.

---

### `POST /draft`

Generates a grounded draft communication for the tenant to review.

**Request:** `multipart/form-data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `situation` | string | Yes | Plain English description, 10–2000 chars |
| `tenant_name` | string | No | Defaults to `[YOUR NAME]` |
| `landlord_name` | string | No | Defaults to `[LANDLORD/AGENT NAME]` |
| `file` | PDF | No | Lease PDF for cross-referencing |

**Response:** `text/event-stream` — streams a draft letter with Act citations and mandatory pre-send checklist.

---

## Setup

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/)
- OpenAI API key
- `tenant-advocate-vdb` running (locally or deployed)

### 1. Clone the repository

```bash
git clone https://github.com/your-org/tenant-advocate-rag.git
cd tenant-advocate-rag
```

### 2. Install dependencies

```bash
poetry install --no-root
```

### 3. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set the required values:

```env
# Required
OPENAI_API_KEY=sk-...your-key...

# Vector DB — point to tenant-advocate-vdb
# Local dev: http://localhost:8001
# Production: https://tenant-advocate-vdb.onrender.com
# (configured in tenant_advocate/ingestion/interfaces.py)

# App
CHAT_MODEL=gpt-4o
EMBEDDING_MODEL=microsoft/harrier
JURISDICTION=New South Wales, Australia
TOP_K_RESULTS=5
RELEVANCE_THRESHOLD=0.30
LOG_LEVEL=INFO

# CORS — set to your Streamlit app URL
ALLOWED_ORIGINS=http://localhost:8501
```

### 4. Start the server

```bash
poetry run python -m uvicorn tenant_advocate.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be live at `http://localhost:8000`.
Swagger UI (interactive docs): `http://localhost:8000/docs`

### 5. Test it

```bash
# Health check
curl http://localhost:8000/health

# Chat question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the maximum bond in NSW?", "chat_history": []}' \
  --no-buffer
```

---

## Integration with the Vector Database

The `tenant_advocate/ingestion/interfaces.py` file is the **only file** that connects to `tenant-advocate-vdb`. It exposes two functions to the rest of the codebase:

```python
search_laws(query: str, top_k: int = 5) -> list[RetrievedChunk]
get_store_status() -> StoreStatus
```

The vector database is called at:
```
POST https://tenant-advocate-vdb.onrender.com/query
GET  https://tenant-advocate-vdb.onrender.com/health
```

If the vector DB URL changes, only `interfaces.py` needs updating — nothing else in the codebase changes.

### Expected response shape from the vector DB

```json
{
  "results": [
    {
      "document": "Section 52 — A landlord must not enter...",
      "distance": 0.5349,
      "metadata": {
        "source_file": "residential_tenancies_act_2010",
        "page": "48",
        "section": "52",
        "part": "Part 3 Rights and obligations of landlords and tenants",
        "doc_type": "legislation",
        "jurisdiction": "New South Wales, Australia",
        "last_updated": "2026-04"
      }
    }
  ]
}
```

Distance is cosine distance (lower = more similar). The interface converts this to a similarity score: `score = 1 - distance`.

---

## Prompt Engineering

All prompts live in `tenant_advocate/core/prompts.py`. Three system prompts are defined — one per mode.

### 7 Anti-Hallucination Rules (enforced in all prompts)

These rules were designed in response to documented failures of generic AI tools on NSW tenancy law, as reported by the RTA Queensland (2025) and the Tenants' Union of NSW (2026):

| Rule | Description |
|---|---|
| 1 | Answer **exclusively** from retrieved context — never from training knowledge |
| 2 | **Mandatory citation** after every legal claim `[Act, s.XX]` |
| 3 | **Conflict detection** — flag when lease clause contradicts the Act |
| 4 | **No outcome predictions** — always escalate to Fair Trading / NCAT |
| 5 | **NSW-only** — explicit refusal for other states |
| 6 | **NCAT Procedural Direction 7 (2025)** warning on every output |
| 7 | **Explicit uncertainty** — refuse and redirect when context is insufficient |

### GPT-4o parameters

| Parameter | Value | Reason |
|---|---|---|
| `temperature` | `0.1` | Minimises randomness — factual legal answers must be consistent |
| `streaming` | `True` | Tokens sent as they arrive — improves perceived response time |
| `model` | `gpt-4o` | Best accuracy for complex legal reasoning |

---

## Deployment on Render

The `render.yaml` file in the repo root configures deployment automatically.

### Manual setup steps

1. Go to [render.com](https://render.com) -> **New Web Service** -> connect this GitHub repo
2. Render detects `render.yaml` — set the following **secret** environment variable manually in the Render dashboard (do not put this in `render.yaml`):
   - `OPENAI_API_KEY` -> your real key
3. Click **Deploy**

Build command (from `render.yaml`):
```bash
pip install poetry && poetry install --no-root
```

Start command:
```bash
uvicorn tenant_advocate.api.main:app --host 0.0.0.0 --port $PORT
```

Health check path: `/health`


