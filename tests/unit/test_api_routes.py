"""
tests/unit/test_api_routes.py
------------------------------
Tests for FastAPI endpoints using TestClient.
All LLM calls are mocked — no API key or real OpenAI needed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tenant_advocate.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_shape(self):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "knowledge_base_ready" in data
        assert "jurisdiction" in data

    def test_jurisdiction_is_nsw(self):
        resp = client.get("/health")
        assert "New South Wales" in resp.json()["jurisdiction"]


class TestChatEndpoint:
    def _mock_stream(self, tokens: list[str]):
        """Helper: mock stream_chat_answer to yield fixed tokens."""
        def _gen(*args, **kwargs):
            yield from tokens
        return _gen

    @patch("tenant_advocate.api.routes.chat.stream_chat_answer")
    def test_returns_streaming_response(self, mock_fn):
        mock_fn.side_effect = self._mock_stream(["Bond ", "must not ", "exceed 4 weeks."])
        resp = client.post("/chat", json={"question": "What is the maximum bond?"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    @patch("tenant_advocate.api.routes.chat.stream_chat_answer")
    def test_sse_contains_done_marker(self, mock_fn):
        mock_fn.side_effect = self._mock_stream(["Answer"])
        resp = client.post("/chat", json={"question": "Notice period?"})
        assert "[DONE]" in resp.text

    def test_empty_question_returns_422(self):
        resp = client.post("/chat", json={"question": ""})
        assert resp.status_code == 422

    def test_missing_question_returns_422(self):
        resp = client.post("/chat", json={})
        assert resp.status_code == 422

    @patch("tenant_advocate.api.routes.chat.stream_chat_answer")
    def test_accepts_lease_text(self, mock_fn):
        mock_fn.side_effect = self._mock_stream(["Answer with lease context."])
        resp = client.post("/chat", json={
            "question": "Can my landlord enter?",
            "lease_text": "Clause 5: Landlord may enter with 24hrs notice.",
        })
        assert resp.status_code == 200

    @patch("tenant_advocate.api.routes.chat.stream_chat_answer")
    def test_accepts_chat_history(self, mock_fn):
        mock_fn.side_effect = self._mock_stream(["Follow-up answer."])
        resp = client.post("/chat", json={
            "question": "What about rent increases?",
            "chat_history": [["Prior question?", "Prior answer."]],
        })
        assert resp.status_code == 200


class TestAuditEndpoint:
    def test_no_file_returns_422(self):
        resp = client.post("/audit")
        assert resp.status_code == 422

    def test_non_pdf_returns_422(self):
        resp = client.post(
            "/audit",
            files={"file": ("lease.txt", b"some text", "text/plain")},
        )
        assert resp.status_code == 422

    @patch("tenant_advocate.api.routes.audit.run_lease_audit")
    @patch("tenant_advocate.api.routes.audit.parse_lease_bytes")
    def test_pdf_triggers_audit(self, mock_parse, mock_audit):
        from tenant_advocate.core.lease_parser import ParsedLease
        mock_parse.return_value = ParsedLease(
            filename="lease.pdf", full_text="Tenant pays $2000/month.",
            page_count=1, char_count=30, extraction_success=True,
        )
        mock_audit.return_value = iter(["## NSW Lease Audit Report\n", "All good."])

        resp = client.post(
            "/audit",
            files={"file": ("lease.pdf", b"%PDF-fake", "application/pdf")},
        )
        assert resp.status_code == 200
        assert "[DONE]" in resp.text


class TestDraftEndpoint:
    def test_empty_situation_returns_422(self):
        resp = client.post("/draft", data={"situation": ""})
        assert resp.status_code == 422

    @patch("tenant_advocate.api.routes.draft.generate_draft")
    def test_valid_situation_streams(self, mock_draft):
        mock_draft.return_value = iter(["**DRAFT**\n", "Dear Landlord,\n"])
        resp = client.post(
            "/draft",
            data={
                "situation": "My landlord has not fixed the broken hot water for 3 weeks.",
                "tenant_name": "Jane Smith",
                "landlord_name": "ABC Realty",
            },
        )
        assert resp.status_code == 200
        assert "[DONE]" in resp.text
