from datetime import datetime, timezone

from aivis.models import ToolEntry, VisibilityObj
from aivis.variance import jaccard, summarize_anchor

SCORING_CFG = {
    "list_stability_threshold": 0.70,
    "rank_spread_max": 2,
    "weights": {
        "mention": 0.50,
        "rank": 0.40,
        "citation": 0.10,
    },
    "confidence_cap": {
        "mention_unstable_cap_to_mention_rate": True,
        "rank_spread_over_max_cap": 0.70,
        "list_stability_below_threshold_cap": 0.60,
        "list_stability_below_0_50_cap": 0.50,
        "any_parse_error_cap": 0.60,
        "list_length_below_expected_min_cap": 0.70,
    },
}


def _make_obj(
    brand_rank=1,
    tools=None,
    parse_success=True,
    list_length=5,
):
    if tools is None:
        tools = ["asana", "jira", "clickup", "monday.com", "trello"]
    tool_list = [
        ToolEntry(
            rank=i + 1,
            name_raw=t,
            name_norm=t,
            why="x",
            citation_domains=[],
        )
        for i, t in enumerate(tools)
    ]
    return VisibilityObj(
        visibility_id="x",
        client_id="c",
        client_brand_name="asana",
        category="PM",
        prompt_id="PM-D01",
        prompt_text="t",
        prompt_version="v1.0",
        prompt_family="category",
        expected_list_min=5,
        model_provider="anthropic",
        model_name="m",
        temperature=0.0,
        max_tokens=100,
        run_index=1,
        executed_at_utc=datetime.now(timezone.utc),
        request_payload={},
        raw_response_text="r",
        response_hash="h",
        tool_list=tool_list,
        brand_mentioned=brand_rank is not None,
        brand_rank=brand_rank,
        brand_cited=False,
        brand_citation_domains=[],
        parse_success=parse_success,
        parse_errors=[],
        list_length=list_length,
        has_duplicates=False,
        output_contract_violations=[],
        parse_mode="list",
        mention_score=1.0 if brand_rank else 0.0,
        rank_score=(11 - brand_rank) / 10 if brand_rank else 0.0,
        citation_score=0.0,
        stability_anchor_key="k",
        high_variance_flag=False,
        low_confidence_cap=1.0,
        cap_reasons=[],
    )


def test_jaccard_identical():
    assert jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0


def test_jaccard_disjoint():
    assert jaccard({"a", "b"}, {"c", "d"}) == 0.0


def test_jaccard_partial():
    assert jaccard({"a", "b", "c"}, {"b", "c", "d"}) == 0.5


def test_stable_runs():
    objs = [_make_obj(brand_rank=1) for _ in range(5)]
    s = summarize_anchor(objs, SCORING_CFG)
    assert s["run_count"] == 5
    assert s["mention_rate"] == 1.0
    assert s["mention_stable"] is True
    assert s["rank_spread"] == 0
    assert s["high_variance"] is False
    assert s["confidence_cap"] == 1.0
    assert s["cap_reasons"] == []


def test_mention_instability():
    objs = (
        [_make_obj(brand_rank=1)] * 3
        + [_make_obj(brand_rank=None)] * 2
    )
    s = summarize_anchor(objs, SCORING_CFG)
    assert s["mention_rate"] == 0.6
    assert s["mention_stable"] is False
    assert s["high_variance"] is True
    assert s["confidence_cap"] == 0.6


def test_rank_spread_cap():
    objs = (
        [_make_obj(brand_rank=1)] * 3
        + [_make_obj(brand_rank=5)] * 2
    )
    s = summarize_anchor(objs, SCORING_CFG)
    assert s["rank_spread"] == 4
    assert s["rank_stable"] is False
    assert s["confidence_cap"] <= 0.70


def test_list_instability():
    objs = [
        _make_obj(tools=["asana", "jira", "clickup"]),
        _make_obj(tools=["notion", "linear", "basecamp"]),
        _make_obj(tools=["asana", "jira", "clickup"]),
        _make_obj(tools=["notion", "linear", "basecamp"]),
        _make_obj(tools=["wrike", "smartsheet", "trello"]),
    ]
    s = summarize_anchor(objs, SCORING_CFG)
    assert s["list_stability_score"] < 0.7
    assert s["list_stable"] is False
    assert s["high_variance"] is True