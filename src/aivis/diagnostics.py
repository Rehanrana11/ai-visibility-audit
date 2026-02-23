"""
Diagnostic Engine v1.0 - Five Advanced Diagnostic Metrics
=========================================================
All computed from benchmark JSONL data. No external dependencies.

1. PFCI - Prompt-Family Collapse Index
2. ESA  - Engine Suppression Asymmetry
3. RVRM - Rank Volatility Root-Cause Map
4. CDM  - Competitive Displacement Matrix
5. IGDS - Intent Gradient Decay Score
"""
import json
import math
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict

DIAG_VERSION = "1.0"

FAMILY_DEPTH = {"category": 1, "comparison": 2, "problem": 3, "buyer_intent": 4, "long_tail": 5}

CDM_STOPLIST = {
    "reason", "because", "pros", "cons", "summary", "conclusion",
    "disclaimer", "note", "however", "overall", "alternative",
    "option", "recommendation", "consideration", "mattress",
    "brand", "product", "price", "budget", "quality", "comfort",
    "support", "review", "rating", "comparison", "verdict",
}

CDM_STOPLIST = {
    "reason", "because", "pros", "cons", "summary", "conclusion",
    "disclaimer", "note", "however", "overall", "alternative",
    "option", "recommendation", "consideration", "mattress",
    "brand", "product", "price", "budget", "quality", "comfort",
    "support", "review", "rating", "comparison", "verdict",
}


def _stdev(vals):
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def load_benchmark(path, brand=None):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if brand and r.get("client_brand_name", "").strip().lower() != brand.strip().lower():
                continue
            rows.append(r)
    return rows


def compute_pfci(rows, brand, delta_p=0.15, delta_t=0.15, delta_r=1.5, w1=0.4, w2=0.35, w3=0.25):
    brand_norm = brand.strip().lower()
    brand_rows = [r for r in rows if r.get("client_brand_name", "").strip().lower() == brand_norm]
    if not brand_rows:
        return {"brand": brand, "collapses": [], "error": "no data"}

    brand_presence = _mean([1 if r.get("brand_mentioned") else 0 for r in brand_rows])
    brand_top3 = _mean([1 if r.get("brand_rank") and r["brand_rank"] <= 3 else 0 for r in brand_rows])
    ranks_all = [r["brand_rank"] for r in brand_rows if r.get("brand_rank") is not None]
    brand_avg_rank = _mean(ranks_all) if ranks_all else 10.0

    families = defaultdict(list)
    for r in brand_rows:
        families[r.get("prompt_family", "unknown")].append(r)

    collapses = []
    for fam, fam_rows in families.items():
        fp = _mean([1 if r.get("brand_mentioned") else 0 for r in fam_rows])
        ft3 = _mean([1 if r.get("brand_rank") and r["brand_rank"] <= 3 else 0 for r in fam_rows])
        franks = [r["brand_rank"] for r in fam_rows if r.get("brand_rank") is not None]
        far = _mean(franks) if franks else 10.0

        collapsed = (fp < brand_presence - delta_p) or (ft3 < brand_top3 - delta_t) or (far > brand_avg_rank + delta_r)
        severity = w1 * max(0, brand_presence - fp) + w2 * max(0, brand_top3 - ft3) + w3 * max(0, far - brand_avg_rank)

        collapses.append({
            "family": fam,
            "presence": round(fp, 3),
            "top3_rate": round(ft3, 3),
            "avg_rank": round(far, 2),
            "collapsed": collapsed,
            "severity": round(severity, 3),
            "runs": len(fam_rows),
        })

    collapses.sort(key=lambda x: -x["severity"])
    return {
        "brand": brand,
        "brand_baseline": {"presence": round(brand_presence, 3), "top3": round(brand_top3, 3), "avg_rank": round(brand_avg_rank, 2)},
        "collapses": collapses,
    }


def compute_esa(rows, brand, tau=0.3, kappa=0.4):
    brand_norm = brand.strip().lower()
    brand_rows = [r for r in rows if r.get("client_brand_name", "").strip().lower() == brand_norm]

    engine_presence = defaultdict(list)
    for r in brand_rows:
        engine_presence[r["model_provider"]].append(1 if r.get("brand_mentioned") else 0)

    engine_rates = {e: round(_mean(v), 3) for e, v in engine_presence.items()}
    rates = list(engine_rates.values())
    esa = round(max(rates) - min(rates), 3) if rates else 0

    labels = {}
    for e, rate in engine_rates.items():
        if rate < tau:
            labels[e] = "SUPPRESSED"
        elif esa > kappa:
            labels[e] = "FRAGMENTED"
        else:
            labels[e] = "NORMAL"

    family_esa = {}
    fam_engine = defaultdict(lambda: defaultdict(list))
    for r in brand_rows:
        fam_engine[r.get("prompt_family", "unknown")][r["model_provider"]].append(1 if r.get("brand_mentioned") else 0)

    for fam, engines in fam_engine.items():
        fam_rates = {e: round(_mean(v), 3) for e, v in engines.items()}
        fr = list(fam_rates.values())
        family_esa[fam] = {"rates": fam_rates, "esa": round(max(fr) - min(fr), 3) if fr else 0}

    return {
        "brand": brand,
        "engine_presence": engine_rates,
        "esa_global": esa,
        "fragmented": esa > kappa,
        "engine_labels": labels,
        "family_esa": family_esa,
    }


def compute_rvrm(rows, brand):
    brand_norm = brand.strip().lower()
    brand_rows = [r for r in rows if r.get("client_brand_name", "").strip().lower() == brand_norm]

    heatmap = defaultdict(lambda: defaultdict(list))
    for r in brand_rows:
        if r.get("brand_rank") is not None:
            key = (r["model_provider"], r.get("prompt_family", "unknown"))
            heatmap[key[0]][key[1]].append(r["brand_rank"])

    cells = []
    for engine, families in heatmap.items():
        for fam, ranks in families.items():
            std = round(_stdev(ranks), 3) if len(ranks) >= 2 else 0
            cells.append({
                "engine": engine,
                "family": fam,
                "instability": std,
                "mean_rank": round(_mean(ranks), 2),
                "n_runs": len(ranks),
                "contribution": round(std * len(ranks), 2),
            })

    cells.sort(key=lambda x: -x["contribution"])
    return {"brand": brand, "heatmap": cells}


def compute_cdm(rows, brand):
    brand_norm = brand.strip().lower()
    all_rows = rows

    displacement = defaultdict(lambda: Counter())
    for r in all_rows:
        if r.get("client_brand_name", "").strip().lower() != brand_norm:
            continue
        if r.get("brand_mentioned", False):
            continue
        tools = r.get("tool_list", [])
        for t in tools[:5]:
            tn = t.get("name_norm", "")
            if tn and brand_norm not in tn and len(tn) > 3 and tn not in CDM_STOPLIST and not any(s in tn for s in CDM_STOPLIST):
                displacement[r["model_provider"]][tn] += 1

    result = {}
    for engine, counts in displacement.items():
        total = sum(counts.values())
        top = [(name, cnt, round(cnt / total, 3)) for name, cnt in counts.most_common(5)]
        result[engine] = {"displacers": top, "absent_runs": total}

    return {"brand": brand, "displacement_by_engine": result}


def compute_igds(rows, brand, alpha=0.4, beta=0.35, gamma=0.25):
    brand_norm = brand.strip().lower()
    brand_rows = [r for r in rows if r.get("client_brand_name", "").strip().lower() == brand_norm]

    fam_engine = defaultdict(list)
    for r in brand_rows:
        fam = r.get("prompt_family", "unknown")
        fam_engine[fam].append(r)

    family_perf = {}
    for fam, fr in fam_engine.items():
        p = _mean([1 if r.get("brand_mentioned") else 0 for r in fr])
        t3 = _mean([1 if r.get("brand_rank") and r["brand_rank"] <= 3 else 0 for r in fr])
        ranks = [r["brand_rank"] for r in fr if r.get("brand_rank") is not None]
        ar = _mean(ranks) if ranks else 10.0
        ar_norm = min(1.0, ar / 10.0)
        perf = alpha * p + beta * t3 - gamma * ar_norm
        depth = FAMILY_DEPTH.get(fam, 3)
        family_perf[fam] = {"depth": depth, "perf": round(perf, 3), "presence": round(p, 3), "top3": round(t3, 3), "avg_rank": round(ar, 2)}

    points = [(v["depth"], v["perf"]) for v in family_perf.values() if v["depth"] > 0]
    if len(points) >= 2:
        n = len(points)
        sx = sum(x for x, y in points)
        sy = sum(y for x, y in points)
        sxy = sum(x * y for x, y in points)
        sxx = sum(x * x for x, y in points)
        denom = n * sxx - sx * sx
        slope = round((n * sxy - sx * sy) / denom, 4) if denom != 0 else 0
    else:
        slope = 0

    return {
        "brand": brand,
        "family_performance": family_perf,
        "igds_slope": slope,
        "interpretation": "steep_decay" if slope < -0.05 else "moderate_decay" if slope < -0.02 else "stable" if slope > -0.02 else "unknown",
    }


def run_full_diagnostics(benchmark_path, brand, category=""):
    rows = load_benchmark(benchmark_path)
    return {
        "version": DIAG_VERSION,
        "brand": brand,
        "category": category,
        "pfci": compute_pfci(rows, brand),
        "esa": compute_esa(rows, brand),
        "rvrm": compute_rvrm(rows, brand),
        "cdm": compute_cdm(rows, brand),
        "igds": compute_igds(rows, brand),
    }