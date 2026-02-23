from __future__ import annotations


def rank_score_from_rank(rank: int | None, rank_map: dict) -> float:
    if rank is None:
        return 0.0
    return float(rank_map.get(str(rank), rank_map.get("default", 0.0)))


def compute_scores(
    brand_mentioned: bool,
    brand_rank: int | None,
    brand_cited: bool,
    rank_map: dict,
) -> tuple[float, float, float]:
    mention_score = 1.0 if brand_mentioned else 0.0
    rank_score = rank_score_from_rank(brand_rank, rank_map) if brand_mentioned else 0.0
    citation_score = 1.0 if (brand_mentioned and brand_cited) else 0.0
    return mention_score, rank_score, citation_score