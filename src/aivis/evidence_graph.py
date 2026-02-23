"""Evidence Graph — structured evidence model for AI visibility diagnostics.

Builds a queryable graph from JSONL run data, computes derived metrics,
and generates rule-based diagnostic hypotheses.
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any


# ── Node Types ──────────────────────────────────────────────

@dataclass
class PromptNode:
    id: str
    text: str
    version: str
    family: str
    category: str
    prompt_hash: str

@dataclass
class ProviderNode:
    id: str
    model: str
    temperature: float
    model_version_hint: str = ""

@dataclass
class RunNode:
    run_id: str
    prompt_id: str
    provider_id: str
    run_index: int
    timestamp: str
    response_hash: str
    parse_success: bool
    tool_count: int

@dataclass
class RankedToolNode:
    name_norm: str
    name_raw: str
    rank: int
    run_id: str
    provider_id: str
    citation_domains: list = field(default_factory=list)

@dataclass
class BrandEntityNode:
    canonical_name: str
    aliases: list = field(default_factory=list)
    ambiguity_score: float = 0.0

@dataclass
class CompetitorNode:
    canonical_name: str
    avg_rank: float = 0.0
    stability: float = 0.0
    rank1_pct: float = 0.0

# ── Edge Types ──────────────────────────────────────────────

@dataclass
class Edge:
    source: str
    target: str
    edge_type: str
    properties: dict = field(default_factory=dict)


# ── Derived Metrics ─────────────────────────────────────────

@dataclass
class DiagnosticMetrics:
    stability_scores: dict = field(default_factory=dict)       # provider -> 0-1
    cross_model_agreement: float = 0.0                          # 0-1
    rank_elasticity: dict = field(default_factory=dict)         # provider -> spread
    competitive_adjacency: dict = field(default_factory=dict)   # competitor -> freq above brand
    co_mention_frequency: dict = field(default_factory=dict)    # tool -> count
    share_of_voice_top3: dict = field(default_factory=dict)     # tool -> pct in top 3
    citation_concentration: float = 0.0                         # HHI (0-1)
    dominant_competitor: str = ""
    dominant_competitor_rank1_pct: float = 0.0


# ── Hypothesis ──────────────────────────────────────────────

@dataclass
class Hypothesis:
    id: str
    description: str
    probability_band: str  # HIGH / MEDIUM / LOW / UNKNOWN
    supporting_signals: list = field(default_factory=list)
    conflicting_signals: list = field(default_factory=list)
    missing_data: list = field(default_factory=list)
    recommended_next_test: str = ""


# ── Evidence Graph Builder ──────────────────────────────────

class EvidenceGraph:
    """Builds structured evidence graph from compare JSONL files."""

    def __init__(self, brand: str):
        self.brand = brand
        self.brand_norm = brand.strip().lower()
        self.prompts: dict[str, PromptNode] = {}
        self.providers: dict[str, ProviderNode] = {}
        self.runs: list[RunNode] = []
        self.tools: list[RankedToolNode] = []
        self.edges: list[Edge] = []
        self.competitors: dict[str, CompetitorNode] = {}
        self.brand_entity: BrandEntityNode | None = None
        self.metrics: DiagnosticMetrics = DiagnosticMetrics()
        self.hypotheses: list[Hypothesis] = []
        self._raw_records: list[dict] = []

    def load_compare_files(self, audit_dir: str, providers: list[str] | None = None):
        """Load JSONL files from compare command output."""
        audit_path = Path(audit_dir)
        if providers is None:
            providers = [f.stem.replace("compare_", "")
                        for f in audit_path.glob("compare_*.jsonl")]

        for prov in providers:
            fpath = audit_path / f"compare_{prov}.jsonl"
            if not fpath.exists():
                continue
            for line in fpath.open():
                record = json.loads(line.strip())
                self._raw_records.append(record)

    def build(self):
        """Build the full evidence graph from loaded records."""
        self._build_nodes()
        self._build_edges()
        self._compute_metrics()
        self._generate_hypotheses()

    def _build_nodes(self):
        """Extract nodes from raw records."""
        brand_aliases = set()

        for rec in self._raw_records:
            # Prompt node
            pid = rec["prompt_id"]
            if pid not in self.prompts:
                self.prompts[pid] = PromptNode(
                    id=pid,
                    text=rec.get("prompt_text", ""),
                    version=rec.get("prompt_version", ""),
                    family=rec.get("prompt_family", ""),
                    category=rec.get("category", ""),
                    prompt_hash=hashlib.sha256(
                        rec.get("prompt_text", "").encode()
                    ).hexdigest()[:16],
                )

            # Provider node
            prov = rec["model_provider"]
            if prov not in self.providers:
                self.providers[prov] = ProviderNode(
                    id=prov,
                    model=rec.get("model_name", ""),
                    temperature=rec.get("temperature", 0.0),
                    model_version_hint=rec.get("model_version_hint", ""),
                )

            # Run node
            run_id = f"{pid}-{prov}-{rec['run_index']}"
            self.runs.append(RunNode(
                run_id=run_id,
                prompt_id=pid,
                provider_id=prov,
                run_index=rec["run_index"],
                timestamp=rec.get("executed_at_utc", ""),
                response_hash=rec.get("response_hash", ""),
                parse_success=rec.get("parse_success", False),
                tool_count=rec.get("list_length", 0),
            ))

            # Ranked tool nodes
            for tool in rec.get("tool_list", []):
                self.tools.append(RankedToolNode(
                    name_norm=tool["name_norm"],
                    name_raw=tool.get("name_raw", ""),
                    rank=tool["rank"],
                    run_id=run_id,
                    provider_id=prov,
                    citation_domains=tool.get("citation_domains", []),
                ))

                # Track brand aliases
                if self.brand_norm in tool["name_norm"] or tool["name_norm"] in self.brand_norm:
                    brand_aliases.add(tool["name_raw"])

        # Brand entity
        self.brand_entity = BrandEntityNode(
            canonical_name=self.brand_norm,
            aliases=sorted(brand_aliases),
            ambiguity_score=len(brand_aliases) / max(len(self._raw_records), 1),
        )

    def _build_edges(self):
        """Build relationship edges."""
        for tool in self.tools:
            # Tool appears in run
            self.edges.append(Edge(
                source=tool.run_id,
                target=tool.name_norm,
                edge_type="RUN_CONTAINS_TOOL",
                properties={"rank": tool.rank},
            ))

            # Brand relationships
            is_brand = (self.brand_norm in tool.name_norm
                       or tool.name_norm in self.brand_norm)

            if is_brand:
                self.edges.append(Edge(
                    source=tool.name_norm,
                    target="BRAND_TARGET",
                    edge_type="TOOL_IS_BRAND",
                    properties={"rank": tool.rank, "provider": tool.provider_id},
                ))

        # Competitive ranking edges
        brand_ranks = defaultdict(list)  # provider -> [ranks]
        competitor_ranks = defaultdict(lambda: defaultdict(list))  # competitor -> provider -> [ranks]

        for tool in self.tools:
            is_brand = (self.brand_norm in tool.name_norm
                       or tool.name_norm in self.brand_norm)
            if is_brand:
                brand_ranks[tool.provider_id].append(tool.rank)
            else:
                competitor_ranks[tool.name_norm][tool.provider_id].append(tool.rank)

        # Build competitive adjacency
        for comp_name, prov_ranks in competitor_ranks.items():
            all_ranks = [r for ranks in prov_ranks.values() for r in ranks]
            if not all_ranks:
                continue

            avg = sum(all_ranks) / len(all_ranks)
            rank_set = set(all_ranks)
            stability = 1.0 - (len(rank_set) - 1) / max(len(all_ranks), 1)

            # Count how often this competitor is #1
            rank1_count = sum(1 for r in all_ranks if r == 1)
            rank1_pct = rank1_count / len(all_ranks)

            self.competitors[comp_name] = CompetitorNode(
                canonical_name=comp_name,
                avg_rank=round(avg, 2),
                stability=round(stability, 3),
                rank1_pct=round(rank1_pct, 3),
            )

            # Rank-above/below edges vs brand
            brand_all = [r for ranks in brand_ranks.values() for r in ranks]
            brand_avg = sum(brand_all) / len(brand_all) if brand_all else 99
            if avg < brand_avg:
                self.edges.append(Edge(
                    source=comp_name,
                    target=self.brand_norm,
                    edge_type="RANKS_ABOVE",
                    properties={"avg_rank_diff": round(brand_avg - avg, 2)},
                ))

    def _compute_metrics(self):
        """Compute derived diagnostic metrics."""
        m = self.metrics

        # A) Stability scores per provider
        brand_ranks_by_prov = defaultdict(list)
        for tool in self.tools:
            is_brand = (self.brand_norm in tool.name_norm
                       or tool.name_norm in self.brand_norm)
            if is_brand:
                brand_ranks_by_prov[tool.provider_id].append(tool.rank)

        for prov, ranks in brand_ranks_by_prov.items():
            if not ranks:
                m.stability_scores[prov] = 0.0
                continue
            unique = len(set(ranks))
            m.stability_scores[prov] = round(1.0 - (unique - 1) / max(len(ranks), 1), 3)
            m.rank_elasticity[prov] = max(ranks) - min(ranks) if ranks else 0

        # B) Cross-model agreement
        means = [sum(r)/len(r) for r in brand_ranks_by_prov.values() if r]
        if len(means) >= 2:
            spread = max(means) - min(means)
            m.cross_model_agreement = round(max(0, 1.0 - spread / 10), 3)

        # C) Co-mention frequency
        tool_counts = Counter()
        for tool in self.tools:
            tool_counts[tool.name_norm] += 1
        m.co_mention_frequency = dict(tool_counts.most_common(15))

        # D) Share of voice top-3
        top3_counts = Counter()
        total_runs = len(self.runs)
        for tool in self.tools:
            if tool.rank <= 3:
                top3_counts[tool.name_norm] += 1
        if total_runs > 0:
            m.share_of_voice_top3 = {
                name: round(count / total_runs, 3)
                for name, count in top3_counts.most_common(10)
            }

        # E) Citation concentration (HHI)
        all_citations = []
        for tool in self.tools:
            all_citations.extend(tool.citation_domains)
        if all_citations:
            total = len(all_citations)
            domain_counts = Counter(all_citations)
            hhi = sum((c / total) ** 2 for c in domain_counts.values())
            m.citation_concentration = round(hhi, 3)
        else:
            m.citation_concentration = -1  # Not measured

        # F) Dominant competitor
        if self.competitors:
            best = max(self.competitors.values(), key=lambda c: c.rank1_pct)
            m.dominant_competitor = best.canonical_name
            m.dominant_competitor_rank1_pct = best.rank1_pct

    def _generate_hypotheses(self):
        """Generate rule-based diagnostic hypotheses."""
        m = self.metrics
        h_id = 0

        # H1: Cross-model authority inconsistency
        unstable_providers = [p for p, s in m.stability_scores.items() if s < 0.9]
        if m.cross_model_agreement < 0.7 and len(unstable_providers) >= 1:
            h_id += 1
            signals = [
                f"Cross-model agreement: {m.cross_model_agreement:.3f} (below 0.7 threshold)",
                f"Unstable providers: {', '.join(unstable_providers)}",
            ]
            for p in unstable_providers:
                signals.append(
                    f"{p} rank elasticity: {m.rank_elasticity.get(p, 'N/A')}"
                )
            self.hypotheses.append(Hypothesis(
                id=f"H{h_id}",
                description=(
                    "Cross-model authority inconsistency: AI engines disagree on "
                    f"{self.brand}'s positioning, and {len(unstable_providers)} "
                    f"provider(s) show intra-model instability."
                ),
                probability_band="HIGH" if len(unstable_providers) >= 2 else "MEDIUM",
                supporting_signals=signals,
                missing_data=["Citation diversity not measured"]
                    if m.citation_concentration < 0 else [],
                recommended_next_test="Run citation concentration analysis",
            ))

        # H2: Competitor dominance
        if m.dominant_competitor_rank1_pct >= 0.5:
            h_id += 1
            comp = self.competitors.get(m.dominant_competitor)
            self.hypotheses.append(Hypothesis(
                id=f"H{h_id}",
                description=(
                    f"Competitor dominance: '{m.dominant_competitor}' holds #1 position "
                    f"in {m.dominant_competitor_rank1_pct*100:.0f}% of all runs across "
                    f"all providers, indicating strong default-answer bias."
                ),
                probability_band="HIGH",
                supporting_signals=[
                    f"'{m.dominant_competitor}' #1 in {m.dominant_competitor_rank1_pct*100:.0f}% of runs",
                    f"Competitor avg rank: {comp.avg_rank:.1f}" if comp else "N/A",
                    f"Competitor stability: {comp.stability:.3f}" if comp else "N/A",
                    f"{self.brand} never reaches #1",
                ],
            ))

        # H3: Platform-specific weakness
        for prov, stab in m.stability_scores.items():
            if stab < 0.8:
                h_id += 1
                ranks = [t.rank for t in self.tools
                        if t.provider_id == prov
                        and (self.brand_norm in t.name_norm
                             or t.name_norm in self.brand_norm)]
                self.hypotheses.append(Hypothesis(
                    id=f"H{h_id}",
                    description=(
                        f"Platform-specific instability on {prov}: "
                        f"rank oscillates between #{min(ranks)} and #{max(ranks)} "
                        f"across identical queries."
                    ),
                    probability_band="MEDIUM",
                    supporting_signals=[
                        f"Stability score: {stab:.3f}",
                        f"Rank values: {ranks}",
                        f"Elasticity: {m.rank_elasticity.get(prov, 0)}",
                    ],
                    recommended_next_test=(
                        f"Run expanded prompt set on {prov} to determine if "
                        f"instability is prompt-specific or systematic."
                    ),
                ))

        # H4: Top-3 share-of-voice deficit
        brand_sov = 0.0
        for name, sov in m.share_of_voice_top3.items():
            if self.brand_norm in name or name in self.brand_norm:
                brand_sov = sov
                break

        if brand_sov < 0.5 and m.dominant_competitor:
            dominant_sov = m.share_of_voice_top3.get(m.dominant_competitor, 0)
            if dominant_sov > brand_sov:
                h_id += 1
                self.hypotheses.append(Hypothesis(
                    id=f"H{h_id}",
                    description=(
                        f"Top-3 share-of-voice deficit: {self.brand} appears in "
                        f"top-3 in {brand_sov*100:.0f}% of runs vs "
                        f"'{m.dominant_competitor}' at {dominant_sov*100:.0f}%."
                    ),
                    probability_band="MEDIUM",
                    supporting_signals=[
                        f"Brand top-3 SoV: {brand_sov*100:.0f}%",
                        f"Dominant competitor top-3 SoV: {dominant_sov*100:.0f}%",
                        f"SoV gap: {(dominant_sov - brand_sov)*100:.0f} percentage points",
                    ],
                ))

        # H5: Citation data gap
        if m.citation_concentration < 0:
            h_id += 1
            self.hypotheses.append(Hypothesis(
                id=f"H{h_id}",
                description=(
                    "Insufficient citation data: AI models did not provide source "
                    "citations in their responses. Citation concentration and "
                    "fragility metrics cannot be computed."
                ),
                probability_band="UNKNOWN",
                missing_data=[
                    "Citation domains not present in model responses",
                    "Cannot compute Herfindahl-Hirschman Index",
                ],
                recommended_next_test=(
                    "Use citation-eliciting prompts or models with web search "
                    "to capture citation domain data."
                ),
            ))

    def to_dict(self) -> dict:
        """Export full graph as dictionary."""
        return {
            "brand": self.brand,
            "brand_entity": asdict(self.brand_entity) if self.brand_entity else {},
            "prompts": {k: asdict(v) for k, v in self.prompts.items()},
            "providers": {k: asdict(v) for k, v in self.providers.items()},
            "runs": [asdict(r) for r in self.runs],
            "tools_count": len(self.tools),
            "competitors": {k: asdict(v) for k, v in self.competitors.items()},
            "edges_count": len(self.edges),
            "metrics": asdict(self.metrics),
            "hypotheses": [asdict(h) for h in self.hypotheses],
        }

    def to_json(self, path: str, indent: int = 2):
        """Write graph to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent, default=str)

    def summary(self) -> str:
        """Print human-readable diagnostic summary."""
        lines = []
        lines.append(f"=== EVIDENCE GRAPH: {self.brand} ===")
        lines.append(f"Prompts: {len(self.prompts)}")
        lines.append(f"Providers: {list(self.providers.keys())}")
        lines.append(f"Runs: {len(self.runs)}")
        lines.append(f"Ranked tools: {len(self.tools)}")
        lines.append(f"Competitors: {len(self.competitors)}")
        lines.append(f"Edges: {len(self.edges)}")

        lines.append(f"\n--- METRICS ---")
        m = self.metrics
        lines.append(f"Stability scores: {m.stability_scores}")
        lines.append(f"Cross-model agreement: {m.cross_model_agreement}")
        lines.append(f"Rank elasticity: {m.rank_elasticity}")
        lines.append(f"Dominant competitor: {m.dominant_competitor} ({m.dominant_competitor_rank1_pct*100:.0f}% #1)")
        lines.append(f"Citation concentration: {'NOT MEASURED' if m.citation_concentration < 0 else m.citation_concentration}")

        lines.append(f"\nTop-3 Share of Voice:")
        for name, sov in list(m.share_of_voice_top3.items())[:7]:
            marker = " <-- BRAND" if (self.brand_norm in name or name in self.brand_norm) else ""
            lines.append(f"  {name}: {sov*100:.0f}%{marker}")

        lines.append(f"\n--- HYPOTHESES ---")
        for h in self.hypotheses:
            lines.append(f"\n[{h.id}] {h.probability_band}: {h.description}")
            for s in h.supporting_signals:
                lines.append(f"  + {s}")
            for m_data in h.missing_data:
                lines.append(f"  ? MISSING: {m_data}")
            if h.recommended_next_test:
                lines.append(f"  >> NEXT: {h.recommended_next_test}")

        return "\n".join(lines)