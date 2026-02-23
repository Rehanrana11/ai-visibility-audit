from aivis.scorer import compute_scores, rank_score_from_rank

RANK_MAP = {
    "1": 1.0, "2": 0.9, "3": 0.8, "4": 0.7, "5": 0.6,
    "6": 0.5, "7": 0.4, "8": 0.3, "9": 0.2, "10": 0.1,
    "default": 0.0,
}


def test_rank_score_rank_1():
    assert rank_score_from_rank(1, RANK_MAP) == 1.0


def test_rank_score_rank_10():
    assert rank_score_from_rank(10, RANK_MAP) == 0.1


def test_rank_score_none():
    assert rank_score_from_rank(None, RANK_MAP) == 0.0


def test_scores_mentioned_and_cited():
    m, r, c = compute_scores(True, 1, True, RANK_MAP)
    assert m == 1.0
    assert r == 1.0
    assert c == 1.0


def test_scores_mentioned_not_cited():
    m, r, c = compute_scores(True, 3, False, RANK_MAP)
    assert m == 1.0
    assert r == 0.8
    assert c == 0.0


def test_scores_not_mentioned():
    m, r, c = compute_scores(False, None, False, RANK_MAP)
    assert m == 0.0
    assert r == 0.0
    assert c == 0.0