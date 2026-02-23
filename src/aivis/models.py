from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class ToolEntry(BaseModel):
    rank: int
    name_raw: str
    name_norm: str
    why: str
    citation_domains: list[str] = Field(default_factory=list)


class VisibilityObj(BaseModel):
    # --- Core identity (16) ---
    visibility_id: str
    client_id: str
    client_brand_name: str
    category: str
    prompt_id: str
    prompt_text: str
    prompt_version: str
    prompt_family: str
    expected_list_min: int
    model_provider: str
    model_name: str
    model_version_hint: str | None = None
    temperature: float
    max_tokens: int
    run_index: int
    executed_at_utc: datetime

    # --- Raw artifact storage (4) ---
    request_payload: dict
    raw_response_text: str
    raw_response_json: dict | None = None
    response_hash: str

    # --- Parsed outcome (8) ---
    tool_list: list[ToolEntry]
    brand_mentioned: bool
    brand_rank: int | None = None
    brand_cited: bool
    brand_citation_domains: list[str] = Field(default_factory=list)

    # --- Quality signals (6) ---
    parse_success: bool
    parse_errors: list[str] = Field(default_factory=list)
    list_length: int
    has_duplicates: bool
    output_contract_violations: list[str] = Field(default_factory=list)
    parse_mode: str = "unknown"

    # --- Scoring primitives (4) ---
    mention_score: float
    rank_score: float
    citation_score: float
    stability_anchor_key: str

    # --- Audit flags (4) ---
    high_variance_flag: bool
    low_confidence_cap: float
    cap_reasons: list[str] = Field(default_factory=list)
    notes: str | None = None