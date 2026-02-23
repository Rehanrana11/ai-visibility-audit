from __future__ import annotations

import itertools
from statistics import mean

from .models import VisibilityObj


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_list_stability(objs: list[VisibilityObj]) -> float:
    sets = [{t.name_norm for t in o.tool_list} for o in objs]
    if len(sets) < 2:
        return 1.0
    sims = [
        jaccard(sets[i], sets[j])
        for i, j in itertools.combinations(range(len(sets)), 2)
    ]
    return float(mean(sims)) if sims else 1.0


def summarize_anchor(objs: list[VisibilityObj], scoring_cfg: dict) -> dict:
    runs = len(objs)
    if runs == 0:
        return {"run_count": 0, "error": "no objects"}

    mentioned = [o.brand_mentioned for o in objs]
    mention_rate = sum(1 for x in mentioned if x) / runs
    mention_stable = mention_rate in (0.0, 1.0)

    ranks = [
        o.brand_rank for o in objs
        if o.brand_mentioned and o.brand_rank is not None
    ]
    rank_spread = (max(ranks) - min(ranks)) if len(ranks) >= 2 else 0
    rank_stable = rank_spread <= int(scoring_cfg["rank_spread_max"])

    list_stability = compute_list_stability(objs)
    list_stable = list_stability >= float(
        scoring_cfg["list_stability_threshold"]
    )

    mentioned_count = sum(1 for x in mentioned if x)
    citation_rate = (
        sum(1 for o in objs if o.brand_mentioned and o.brand_cited)
        / mentioned_count
        if mentioned_count > 0
        else None
    )

    any_parse_error = any(
        len(o.parse_errors) > 0 or not o.parse_success for o in objs
    )
    below_min = any(
        o.list_length < o.expected_list_min for o in objs
    )

    high_variance = (
        (not mention_stable)
        or (not rank_stable)
        or (not list_stable)
    )

    cap = 1.0
    reasons: list[str] = []
    caps = scoring_cfg["confidence_cap"]

    if caps.get("mention_unstable_cap_to_mention_rate", True):
        if not mention_stable:
            cap = min(cap, mention_rate)
            reasons.append(
                f"MENTION_UNSTABLE({mention_rate:.2f})"
            )

    if not rank_stable:
        cap = min(cap, float(caps["rank_spread_over_max_cap"]))
        reasons.append(f"RANK_SPREAD({rank_spread})")

    if not list_stable:
        val = float(caps["list_stability_below_threshold_cap"])
        cap = min(cap, val)
        reasons.append(f"LIST_STABILITY({list_stability:.2f})")

    if list_stability < 0.5:
        val = float(caps["list_stability_below_0_50_cap"])
        cap = min(cap, val)
        reasons.append(
            f"LIST_STABILITY_HARD({list_stability:.2f})"
        )

    if any_parse_error:
        cap = min(cap, float(caps["any_parse_error_cap"]))
        reasons.append("PARSE_ERROR")

    if below_min:
        val = float(caps["list_length_below_expected_min_cap"])
        cap = min(cap, val)
        reasons.append("BELOW_MIN_LIST")

    weights = scoring_cfg.get(
        "weights",
        {"mention": 0.50, "rank": 0.40, "citation": 0.10},
    )
    raw_score = (
        mean([o.mention_score for o in objs]) * weights["mention"]
        + mean([o.rank_score for o in objs]) * weights["rank"]
        + mean([o.citation_score for o in objs]) * weights["citation"]
    )
    capped_score = raw_score * cap

    return {
        "run_count": runs,
        "mention_rate": mention_rate,
        "mention_stable": mention_stable,
        "rank_values": ranks,
        "rank_spread": rank_spread,
        "rank_stable": rank_stable,
        "citation_rate": citation_rate,
        "list_stability_score": list_stability,
        "list_stable": list_stable,
        "high_variance": high_variance,
        "confidence_cap": cap,
        "cap_reasons": reasons,
        "raw_score": raw_score,
        "capped_score": capped_score,
    }