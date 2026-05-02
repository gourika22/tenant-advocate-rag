"""
tenant_advocate/config.py
------------------------------
Centralised, validated settings via pydantic-settings.
NSW jurisdiction is used — this system is intentionally
single-jurisdiction. Changing state will require re-ingesting different PDFs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── NSW constants ───────────────────────────────────────
NSW_JURISDICTION = "New South Wales, Australia"
NSW_ACT          = "Residential Tenancies Act 2010 (NSW)"
NSW_CONTACTS = {
    "fair_trading":    ("NSW Fair Trading",          "13 32 20",       "https://www.nsw.gov.au/departments-and-agencies/fair-trading/contact"),
    "ncat":            ("NSW Civil & Admin Tribunal", "1300 006 228",   "https://ncat.nsw.gov.au/about-ncat/contact-us.html"),
    "tenants_union":   ("Tenants' Union of NSW",      "02 8117 3700",   "https://www.tenants.org.au/contact-us"),
    "clc":             ("Community Legal Centres NSW", "1300 888 529",  "https://www.clcnsw.org.au/contact"),
    "legislation":     ("NSW Legislation",             "",               "https://legislation.nsw.gov.au/"),
    "legal_advice":    ("Legal Aid NSW",              "1300 888 529",   "https://www.legalaid.nsw.gov.au/my-problem-is-about/my-housing"),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI
    openai_api_key:  str  = Field(...,  description="OpenAI API key — required")
    embedding_model: str  = Field(default="microsoft/harrier-oss-v1-0.6b")
    chat_model:      str  = Field(default="gpt-4o")

    # ChromaDB
    chroma_persist_dir:      Path = Field(default=Path("./data/chroma_db"))
    chroma_collection_name:  str  = Field(default="tenancy_laws_nsw")

    # Retrieval
    top_k_results:        int   = Field(default=6,    ge=1, le=20)
    relevance_threshold:  float = Field(default=0.30, ge=0.0, le=1.0)

    # Chunking (Yukthi tunes these)
    chunk_size:    int = Field(default=1000)
    chunk_overlap: int = Field(default=150)

    # API
    api_host:        str  = Field(default="0.0.0.0")
    api_port:        int  = Field(default=8000)
    allowed_origins: str  = Field(default="http://localhost:8501")

    # App — jurisdiction fixed to NSW
    jurisdiction: str = Field(default=NSW_JURISDICTION)
    log_level:    str = Field(default="INFO")

    @field_validator("chroma_persist_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        return Path(v).resolve()

    @property
    def origins_list(self) -> list[str]:
        """Parse comma-separated ALLOWED_ORIGINS into a list."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
