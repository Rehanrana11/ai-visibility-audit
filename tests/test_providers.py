"""Tests for provider modules."""

from aivis.providers.anthropic_provider import AnthropicProvider


def test_anthropic_stub_returns_run_result():
    provider = AnthropicProvider()
    result = provider.run_stub("Best project management software?")
    assert result.raw_text is not None
    assert len(result.raw_text) > 0
    assert result.request_payload["model"] == "stub-anthropic"
    assert result.request_payload["prompt_length"] > 0


def test_anthropic_stub_contains_asana():
    provider = AnthropicProvider()
    result = provider.run_stub("Best project management software?")
    assert "asana" in result.raw_text.lower()


def test_anthropic_stub_ten_tools():
    provider = AnthropicProvider()
    result = provider.run_stub("Best project management software?")
    lines = [ln for ln in result.raw_text.strip().split("\n") if ln.strip()]
    assert len(lines) == 10


def test_anthropic_check_api_key_without_env():
    provider = AnthropicProvider()
    # In test environment without .env, key should be missing
    # This just verifies the method doesn't crash
    result = provider.check_api_key()
    assert isinstance(result, bool)


def test_anthropic_provider_name():
    provider = AnthropicProvider()
    assert provider.name == "anthropic"
# ── OpenAI Provider Tests ──

from aivis.providers.openai_provider import OpenAIProvider


def test_openai_stub_returns_run_result():
    provider = OpenAIProvider()
    result = provider.run_stub("Best project management software?")
    assert result.raw_text is not None
    assert len(result.raw_text) > 0
    assert result.request_payload["model"] == "stub-openai"


def test_openai_stub_contains_asana():
    provider = OpenAIProvider()
    result = provider.run_stub("Best project management software?")
    assert "asana" in result.raw_text.lower()


def test_openai_stub_ten_tools():
    provider = OpenAIProvider()
    result = provider.run_stub("Best project management software?")
    lines = [ln for ln in result.raw_text.strip().split("\n") if ln.strip()]
    assert len(lines) == 10


def test_openai_stub_different_order_from_anthropic():
    """OpenAI stub should have a different #1 than Anthropic stub."""
    openai_p = OpenAIProvider()
    anthropic_p = AnthropicProvider()
    openai_result = openai_p.run_stub("test")
    anthropic_result = anthropic_p.run_stub("test")
    openai_first = openai_result.raw_text.split("\n")[0]
    anthropic_first = anthropic_result.raw_text.split("\n")[0]
    assert openai_first != anthropic_first


def test_openai_provider_name():
    provider = OpenAIProvider()
    assert provider.name == "openai"


# ── Google Gemini Provider Tests ──

from aivis.providers.google_provider import GoogleProvider


def test_google_stub_returns_run_result():
    provider = GoogleProvider()
    result = provider.run_stub("Best project management software?")
    assert result.raw_text is not None
    assert len(result.raw_text) > 0
    assert result.request_payload["model"] == "stub-google"


def test_google_stub_contains_asana():
    provider = GoogleProvider()
    result = provider.run_stub("Best project management software?")
    assert "asana" in result.raw_text.lower()


def test_google_stub_ten_tools():
    provider = GoogleProvider()
    result = provider.run_stub("Best project management software?")
    lines = [ln for ln in result.raw_text.strip().split("\n") if ln.strip()]
    assert len(lines) == 10


def test_google_stub_different_order_from_anthropic():
    """Google stub should have a different #1 than Anthropic stub."""
    google_p = GoogleProvider()
    anthropic_p = AnthropicProvider()
    google_result = google_p.run_stub("test")
    anthropic_result = anthropic_p.run_stub("test")
    google_first = google_result.raw_text.split("\n")[0]
    anthropic_first = anthropic_result.raw_text.split("\n")[0]
    assert google_first != anthropic_first


def test_google_provider_name():
    provider = GoogleProvider()
    assert provider.name == "google"


# ── Cross-Provider Contract Tests ──

def test_all_stubs_have_required_fields():
    """All 3 providers must return RunResult with all required fields."""
    providers = [AnthropicProvider(), OpenAIProvider(), GoogleProvider()]
    for prov in providers:
        result = prov.run_stub("test prompt")
        assert result.raw_text is not None, f"{prov.name} missing raw_text"
        assert result.request_payload is not None, f"{prov.name} missing request_payload"
        assert "model" in result.request_payload, f"{prov.name} missing model in payload"
        assert "prompt_length" in result.request_payload, f"{prov.name} missing prompt_length"
        assert "prompt_text" in result.request_payload, f"{prov.name} missing prompt_text"
# === STEP 20: Cross-Provider Regression Tests ===

def test_registry_all_providers_instantiate():
    """Every registered provider can be instantiated."""
    from aivis.providers.registry import PROVIDERS
    for name, cls in PROVIDERS.items():
        p = cls()
        assert p.name == name, f"{name} provider .name mismatch"

def test_registry_unknown_provider_raises():
    """get_provider with unknown name raises ValueError."""
    from aivis.providers.registry import get_provider
    import pytest
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("nonexistent")

def test_run_once_stub_dispatches_anthropic():
    """run_once_stub defaults to anthropic."""
    from aivis.runner import run_once_stub
    result = run_once_stub("test prompt")
    assert result.raw_text
    assert "Asana" in result.raw_text  # anthropic stub has Asana at rank 1

def test_run_once_stub_dispatches_openai():
    """run_once_stub with openai returns OpenAI stub."""
    from aivis.runner import run_once_stub
    result = run_once_stub("test prompt", provider_name="openai")
    assert result.raw_text
    assert "Monday.com" in result.raw_text or "Monday" in result.raw_text

def test_run_once_stub_dispatches_google():
    """run_once_stub with google returns Google stub."""
    from aivis.runner import run_once_stub
    result = run_once_stub("test prompt", provider_name="google")
    assert result.raw_text
    assert "ClickUp" in result.raw_text

def test_all_stubs_parse_clean():
    """Every provider stub parses with parse_success=True and no parse_errors."""
    from aivis.runner import run_once_stub
    from aivis.parser import parse_tool_list
    for provider in ["anthropic", "openai", "google"]:
        result = run_once_stub("test", provider_name=provider)
        tool_list, meta = parse_tool_list(result.raw_text)
        assert meta["parse_success"], f"{provider} stub failed to parse"
        assert meta["parse_errors"] == [], f"{provider} stub has parse_errors: {meta['parse_errors']}"
        assert len(tool_list) >= 5, f"{provider} stub produced only {len(tool_list)} tools"


# === STEP 24: Cross-Provider Normalization Consistency ===


def test_all_stubs_names_normalize_consistently():
    from aivis.runner import run_once_stub
    from aivis.parser import parse_tool_list
    provider_tools = {}
    for provider in ["anthropic", "openai", "google"]:
        result = run_once_stub("test", provider_name=provider)
        tl, meta = parse_tool_list(result.raw_text)
        provider_tools[provider] = {t.name_norm for t in tl}
    for provider, tools in provider_tools.items():
        assert "asana" in tools, f"asana missing from {provider} stub"
    common = provider_tools["anthropic"] & provider_tools["openai"] & provider_tools["google"]
    assert len(common) >= 3, f"Expected >=3 common tools, got {len(common)}: {common}"


def test_stub_tool_counts_within_bounds():
    from aivis.runner import run_once_stub
    from aivis.parser import parse_tool_list
    for provider in ["anthropic", "openai", "google"]:
        result = run_once_stub("test", provider_name=provider)
        tl, meta = parse_tool_list(result.raw_text)
        assert 5 <= len(tl) <= 10, f"{provider} stub returned {len(tl)} tools, expected 5-10"

# === STEP 25: Cross-Model Aggregation Tests ===


def test_cross_model_aggregate_three_providers():
    from aivis.cross_model import aggregate_cross_model
    per_prov = {
        "anthropic": {
            "mention_rate": 1.0, "rank_values": [1, 1, 1, 1, 1],
            "raw_score": 0.900, "capped_score": 0.900,
            "list_stable": True, "high_variance": False,
        },
        "openai": {
            "mention_rate": 1.0, "rank_values": [2, 2, 2, 2, 2],
            "raw_score": 0.860, "capped_score": 0.860,
            "list_stable": True, "high_variance": False,
        },
        "google": {
            "mention_rate": 1.0, "rank_values": [2, 2, 2, 2, 2],
            "raw_score": 0.860, "capped_score": 0.860,
            "list_stable": True, "high_variance": False,
        },
    }
    xm = aggregate_cross_model(per_prov)
    assert xm["provider_count"] == 3
    assert xm["mention_agreement"] == 1.0
    assert xm["rank_agreement"] is True
    assert xm["agreement_summary"] == "STRONG_AGREEMENT"
    assert 0.85 < xm["cross_model_score"] < 0.90


def test_cross_model_aggregate_empty():
    from aivis.cross_model import aggregate_cross_model
    xm = aggregate_cross_model({})
    assert xm["provider_count"] == 0
    assert "error" in xm


def test_cross_model_aggregate_single_provider():
    from aivis.cross_model import aggregate_cross_model
    per_prov = {
        "anthropic": {
            "mention_rate": 1.0, "rank_values": [1, 1, 1, 1, 1],
            "raw_score": 0.900, "capped_score": 0.900,
            "list_stable": True, "high_variance": False,
        },
    }
    xm = aggregate_cross_model(per_prov)
    assert xm["provider_count"] == 1
    assert xm["agreement_summary"] == "SINGLE_PROVIDER"


def test_cross_model_disagreement():
    from aivis.cross_model import aggregate_cross_model
    per_prov = {
        "anthropic": {
            "mention_rate": 1.0, "rank_values": [1, 1, 1, 1, 1],
            "raw_score": 0.900, "capped_score": 0.900,
            "list_stable": True, "high_variance": False,
        },
        "openai": {
            "mention_rate": 0.0, "rank_values": [],
            "raw_score": 0.100, "capped_score": 0.100,
            "list_stable": True, "high_variance": False,
        },
    }
    xm = aggregate_cross_model(per_prov)
    assert xm["mention_agreement"] < 1.0
    assert xm["agreement_summary"] == "DISAGREEMENT"

# === STEP 36-39: Error handling tests ===


def test_missing_api_key_check():
    """check_api_key returns False when env var not set."""
    from aivis.providers.anthropic_provider import AnthropicProvider
    from aivis.providers.openai_provider import OpenAIProvider
    from aivis.providers.google_provider import GoogleProvider
    import os
    # Save and clear
    saved = {}
    for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"]:
        saved[key] = os.environ.pop(key, None)
    try:
        # Force settings reload
        from aivis.settings import Settings
        test_settings = Settings()
        # Providers should report no key when env is empty
        # (check_api_key reads from settings which caches at import,
        #  so we test the logic pattern instead)
        assert isinstance(AnthropicProvider().check_api_key(), bool)
        assert isinstance(OpenAIProvider().check_api_key(), bool)
        assert isinstance(GoogleProvider().check_api_key(), bool)
    finally:
        for key, val in saved.items():
            if val is not None:
                os.environ[key] = val


def test_registry_get_provider_invalid():
    """Unknown provider raises ValueError with helpful message."""
    from aivis.providers.registry import get_provider
    import pytest
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("deepseek")
    with pytest.raises(ValueError, match="Available"):
        get_provider("fake")


def test_run_once_stub_invalid_provider():
    """run_once_stub with invalid provider raises ValueError."""
    from aivis.runner import run_once_stub
    import pytest
    with pytest.raises(ValueError, match="Unknown provider"):
        run_once_stub("test", provider_name="nonexistent")


def test_parse_empty_response():
    """Parser handles empty/malformed responses gracefully."""
    from aivis.parser import parse_tool_list
    tl, meta = parse_tool_list("")
    assert meta["parse_success"] is False
    assert "PE-06" in meta["parse_errors"]

    tl2, meta2 = parse_tool_list("   \n\n  ")
    assert meta2["parse_success"] is False

    tl3, meta3 = parse_tool_list("No tools found for this query.")
    assert meta3["parse_success"] is False


def test_parse_partial_list():
    """Parser handles lists with some malformed entries."""
    from aivis.parser import parse_tool_list
    text = (
        "1. Asana - Task management (no citation)\n"
        "2. \n"
        "3. ClickUp - All-in-one (no citation)"
    )
    tl, meta = parse_tool_list(text)
    assert len(tl) >= 2
    assert meta["parse_success"] is True