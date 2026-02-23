"""Cross-model aggregation metrics.

Compares visibility results across multiple AI providers to determine
whether models agree on brand visibility, rank, and citation.
"""

from __future__ import annotations

from statistics import mean


def aggregate_cross_model(
    per_provider: dict[str, dict],
) -> dict:
    """
    Aggregate per-provider variance summaries into a cross-model report.

    Args:
        per_provider: {provider_name: summarize_anchor output dict}

    Returns:
        Cross-model summary dict with agreement metrics.
    """
    providers = sorted(per_provider.keys())
    n_providers = len(providers)

    if n_providers == 0:
        return {"provider_count": 0, "error": "no providers"}

    if n_providers == 1:
        name = providers[0]
        s = per_provider[name]
        return {
            "provider_count": 1,
            "providers": providers,
            "mention_agreement": 1.0,
            "mention_rates": {name: s["mention_rate"]},
            "rank_values": {name: s["rank_values"]},
            "rank_means": {name: _safe_mean(s["rank_values"])},
            "rank_agreement": True,
            "raw_scores": {name: s["raw_score"]},
            "capped_scores": {name: s["capped_score"]},
            "score_spread": 0.0,
            "cross_model_score": s["capped_score"],
            "cross_model_stable": s["list_stable"] and not s["high_variance"],
            "agreement_summary": "SINGLE_PROVIDER",
        }

    # Mention agreement: do all providers agree on whether brand is mentioned?
    mention_rates = {p: per_provider[p]["mention_rate"] for p in providers}
    mention_bools = [r > 0.0 for r in mention_rates.values()]
    mention_agreement = 1.0 if len(set(mention_bools)) == 1 else (
        sum(mention_bools) / n_providers
    )

    # Rank comparison
    rank_values = {p: per_provider[p]["rank_values"] for p in providers}
    rank_means = {p: _safe_mean(v) for p, v in rank_values.items()}
    valid_means = [m for m in rank_means.values() if m is not None]
    rank_spread = (max(valid_means) - min(valid_means)) if len(valid_means) >= 2 else 0.0
    rank_agreement = rank_spread <= 2.0

    # Score comparison
    raw_scores = {p: per_provider[p]["raw_score"] for p in providers}
    capped_scores = {p: per_provider[p]["capped_score"] for p in providers}
    score_spread = max(capped_scores.values()) - min(capped_scores.values())

    # Cross-model composite score (average of capped scores)
    cross_model_score = mean(capped_scores.values())

    # Stability: all providers must be individually stable
    all_stable = all(
        per_provider[p]["list_stable"] and not per_provider[p]["high_variance"]
        for p in providers
    )

    # High variance flags
    variance_flags = {p: per_provider[p]["high_variance"] for p in providers}

    # Agreement classification
    if mention_agreement == 1.0 and rank_agreement and score_spread < 0.10:
        agreement_summary = "STRONG_AGREEMENT"
    elif mention_agreement >= 0.5 and score_spread < 0.25:
        agreement_summary = "MODERATE_AGREEMENT"
    else:
        agreement_summary = "DISAGREEMENT"

    return {
        "provider_count": n_providers,
        "providers": providers,
        "mention_agreement": mention_agreement,
        "mention_rates": mention_rates,
        "rank_values": rank_values,
        "rank_means": rank_means,
        "rank_spread": rank_spread,
        "rank_agreement": rank_agreement,
        "raw_scores": raw_scores,
        "capped_scores": capped_scores,
        "score_spread": round(score_spread, 4),
        "cross_model_score": round(cross_model_score, 4),
        "cross_model_stable": all_stable,
        "variance_flags": variance_flags,
        "agreement_summary": agreement_summary,
    }


def _safe_mean(values: list) -> float | None:
    """Mean of a list, or None if empty."""
    if not values:
        return None
    return round(mean(values), 2)