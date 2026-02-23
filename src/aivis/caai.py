"""
CAAI v1.0 - AI Category Authority Index
Frozen formula. Version-controlled.
CAAI = (PS * 0.25) + (DS * 0.25) + (SS * 0.20) + ((1-SRS) * 0.20) + ((1-FI) * 0.10)
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

CAAI_VERSION = "1.0"
WEIGHTS = {"version": CAAI_VERSION, "PS": 0.25, "DS": 0.25, "SS": 0.20, "SRS": 0.20, "FI": 0.10}
ENGINE_WEIGHTS_DEFAULT = {"openai": 0.35, "google": 0.30, "anthropic": 0.20, "grok": 0.15}

@dataclass
class CAAIDimensions:
    presence_score: float = 0.0
    dominance_score: float = 0.0
    stability_score: float = 0.0
    suppression_risk: float = 0.0
    fragmentation_index: float = 0.0

@dataclass
class CAAIResult:
    brand: str
    category: str
    caai_version: str = CAAI_VERSION
    caai_raw: float = 0.0
    caai_tier: str = ""
    dimensions: CAAIDimensions = field(default_factory=CAAIDimensions)
    per_engine: dict = field(default_factory=dict)
    prompt_count: int = 0
    run_count: int = 0
    engine_count: int = 0
    weights_used: dict = field(default_factory=lambda: dict(WEIGHTS))
    engine_weights_used: dict = field(default_factory=dict)

def compute_caai_from_records(brand, category, records, engine_weights=None):
    providers = sorted(set(r["model_provider"] for r in records))
    if not providers:
        raise ValueError("No records")
    if engine_weights is None:
        present = {p: ENGINE_WEIGHTS_DEFAULT.get(p, 0.15) for p in providers}
        tw = sum(present.values())
        engine_weights = {k: v / tw for k, v in present.items()}
    prompt_ids = set()
    total_runs = 0
    engine_data = {}
    for prov in providers:
        pr = [r for r in records if r["model_provider"] == prov]
        total = len(pr)
        total_runs += total
        for r in pr:
            prompt_ids.add(r.get("prompt_id", ""))
        mentions = sum(1 for r in pr if r.get("brand_mentioned", False))
        ranks = [r["brand_rank"] for r in pr if r.get("brand_rank") is not None]
        top3 = sum(1 for r in ranks if r <= 3)
        mr = mentions / total if total > 0 else 0
        t3r = top3 / total if total > 0 else 0
        if len(ranks) >= 2:
            mn = sum(ranks) / len(ranks)
            var = sum((r - mn) ** 2 for r in ranks) / len(ranks)
            stab = max(0, 1.0 - var / 25.0)
        elif len(ranks) == 1:
            stab = 1.0
        else:
            stab = 0.0
        ar = sum(ranks) / len(ranks) if ranks else None
        engine_data[prov] = {"mention_rate": round(mr, 4), "top3_rate": round(t3r, 4), "stability": round(stab, 4), "invisibility_rate": round(1.0 - mr, 4), "avg_rank": round(ar, 2) if ar else None, "runs": total}
    def ew(p):
        return engine_weights.get(p, 1.0 / len(providers))
    ps = sum(engine_data[e]["mention_rate"] * ew(e) for e in providers) * 100
    ds = sum(engine_data[e]["top3_rate"] * ew(e) for e in providers) * 100
    ss = sum(engine_data[e]["stability"] * ew(e) for e in providers) * 100
    srs = sum(engine_data[e]["invisibility_rate"] * ew(e) for e in providers) * 100
    avg_ranks = [engine_data[e]["avg_rank"] for e in providers if engine_data[e]["avg_rank"] is not None]
    if len(avg_ranks) >= 2:
        fi_rank = min(1.0, (max(avg_ranks) - min(avg_ranks)) / 10.0)
    else:
        fi_rank = 0.5
    mrates = [engine_data[e]["mention_rate"] for e in providers]
    mspread = max(mrates) - min(mrates)
    fi = ((fi_rank * 0.7) + (mspread * 0.3)) * 100
    caai_raw = round((ps * 0.25) + (ds * 0.25) + (ss * 0.20) + ((100 - srs) * 0.20) + ((100 - fi) * 0.10), 2)
    if caai_raw >= 80: tier = "DOMINANT"
    elif caai_raw >= 65: tier = "STRONG"
    elif caai_raw >= 45: tier = "MODERATE"
    elif caai_raw >= 25: tier = "WEAK"
    else: tier = "CRITICAL"
    dims = CAAIDimensions(presence_score=round(ps, 1), dominance_score=round(ds, 1), stability_score=round(ss, 1), suppression_risk=round(srs, 1), fragmentation_index=round(fi, 1))
    return CAAIResult(brand=brand, category=category, caai_raw=caai_raw, caai_tier=tier, dimensions=dims, per_engine=engine_data, prompt_count=len(prompt_ids), run_count=total_runs, engine_count=len(providers), engine_weights_used=engine_weights)

def compute_caai_from_benchmark(brand, category, benchmark_path, engine_weights=None):
    bn = brand.strip().lower()
    p = Path(benchmark_path)
    recs = [json.loads(l) for l in p.open(encoding="utf-8") if json.loads(l).get("client_brand_name", "").strip().lower() == bn]
    if not recs:
        raise ValueError(f"No records for '{brand}'")
    return compute_caai_from_records(brand, category, recs, engine_weights)

def caai_summary(result):
    lines = []
    lines.append("=" * 60)
    lines.append(f"CAAI v{result.caai_version} -- {result.brand}")
    lines.append(f"Category: {result.category}")
    lines.append("=" * 60)
    lines.append(f"  CAAI Score:  {result.caai_raw:.1f} / 100  [{result.caai_tier}]")
    lines.append(f"  --- DIMENSIONS ---")
    d = result.dimensions
    lines.append(f"  Presence (PS):       {d.presence_score:6.1f}")
    lines.append(f"  Dominance (DS):      {d.dominance_score:6.1f}")
    lines.append(f"  Stability (SS):      {d.stability_score:6.1f}")
    lines.append(f"  Suppression (SRS):   {d.suppression_risk:6.1f}")
    lines.append(f"  Fragmentation (FI):  {d.fragmentation_index:6.1f}")
    lines.append(f"  --- PER ENGINE ---")
    for eng, data in sorted(result.per_engine.items()):
        avg = f"#{data['avg_rank']:.1f}" if data["avg_rank"] else "INVISIBLE"
        lines.append(f"  {eng:12s}: mention={data['mention_rate']*100:.0f}% top3={data['top3_rate']*100:.0f}% stab={data['stability']:.3f} avg={avg}")
    lines.append(f"  Prompts: {result.prompt_count}  Runs: {result.run_count}  Engines: {result.engine_count}")
    return chr(10).join(lines)

def to_dict(result):
    return asdict(result)