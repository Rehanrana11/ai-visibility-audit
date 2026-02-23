from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint

from .models import VisibilityObj
from .parser import parse_tool_list
from .reporter import write_simple_pdf, write_cross_model_pdf
from .runner import run_once, run_once_stub
from .scorer import compute_scores
from .storage import read_jsonl, write_jsonl
from .variance import summarize_anchor

app = typer.Typer(no_args_is_help=True)


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _anchor_key(
    client_id: str,
    prompt_id: str,
    provider: str,
    model: str,
    temp: float,
    pv: str,
) -> str:
    return f"{client_id}:{prompt_id}:{provider}:{model}:{temp}:{pv}"


@app.command()
def run(
    client_id: Annotated[str, typer.Option()] = "demo",
    client_brand: Annotated[str, typer.Option()] = "Asana",
    category: Annotated[str, typer.Option()] = "Project Management Software",
    prompt_id: Annotated[str, typer.Option()] = "PM-D01",
    prompt_version: Annotated[str, typer.Option()] = "v1.0",
    model_provider: Annotated[str, typer.Option()] = "anthropic",
    model_name: Annotated[str, typer.Option()] = "",
    temperature: Annotated[float, typer.Option()] = 0.0,
    max_tokens: Annotated[int, typer.Option()] = 2048,
    run_index: Annotated[int, typer.Option()] = 1,
    live: Annotated[bool, typer.Option("--live")] = False,
    out: Annotated[str, typer.Option()] = "data/audits/visibility_runs.jsonl",
):
    """Execute a single prompt run. Use --live for real API."""
    out_path = Path(out)
    prompts = _load_json(Path("config/prompts_v1.json"))
    p = next((x for x in prompts if x["id"] == prompt_id), None)
    if not p:
        raise typer.BadParameter(f"Unknown prompt_id: {prompt_id}")
    if not model_name:
        models_cfg = _load_json(Path("config/models.json"))
        model_name = models_cfg["providers"][model_provider]["model_name"]
    prompt_text = p["text"]
    expected_list_min = int(p["expected_list_min"])
    prompt_family = p["family"]

    if live:
        rr = run_once(
            prompt_text,
            provider_name=model_provider,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        rr = run_once_stub(prompt_text, provider_name=model_provider)
    raw = rr.raw_text

    tool_list, meta = parse_tool_list(raw)

    brand_norm = client_brand.strip().lower()
    brand_rank = None
    brand_cited = False
    brand_domains: list[str] = []
    for t in tool_list:
        if brand_norm in t.name_norm or t.name_norm in brand_norm:
            brand_rank = t.rank
            brand_domains = t.citation_domains
            brand_cited = len(brand_domains) > 0
            break
    brand_mentioned = brand_rank is not None

    scoring_cfg = _load_json(Path("config/scoring_v1.json"))
    ms, rs, cs = compute_scores(
        brand_mentioned,
        brand_rank,
        brand_cited,
        scoring_cfg["rank_map"],
    )

    vo = VisibilityObj(
        visibility_id=str(uuid.uuid4()),
        client_id=client_id,
        client_brand_name=client_brand,
        category=category,
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        prompt_version=prompt_version,
        prompt_family=prompt_family,
        expected_list_min=expected_list_min,
        model_provider=model_provider,
        model_name=model_name,
        model_version_hint=getattr(rr, "model_version_hint", None),
        temperature=temperature,
        max_tokens=max_tokens,
        run_index=run_index,
        executed_at_utc=datetime.now(timezone.utc),
        request_payload=rr.request_payload,
        raw_response_text=raw,
        raw_response_json=rr.raw_json,
        response_hash=_sha256(raw),
        tool_list=tool_list,
        brand_mentioned=brand_mentioned,
        brand_rank=brand_rank,
        brand_cited=brand_cited,
        brand_citation_domains=brand_domains,
        parse_success=bool(meta["parse_success"]),
        parse_errors=meta["parse_errors"],
        list_length=len(tool_list),
        has_duplicates=bool(meta.get("has_duplicates", False)),
        output_contract_violations=meta["violations"],
        parse_mode=meta["parse_mode"],
        mention_score=ms,
        rank_score=rs,
        citation_score=cs,
        stability_anchor_key=_anchor_key(
            client_id,
            prompt_id,
            model_provider,
            model_name,
            temperature,
            prompt_version,
        ),
        high_variance_flag=False,
        low_confidence_cap=1.0,
        cap_reasons=[],
    )

    write_jsonl(out_path, [vo])
    rprint(
        f"[green]OK[/green] {prompt_id} run={run_index} "
        f"brand={'YES' if brand_mentioned else 'NO'} "
        f"rank={brand_rank} parse={vo.parse_success} "
        f"tools={vo.list_length}"
    )


@app.command()
def smoke(
    prompt_id: Annotated[str, typer.Option()] = "PM-D01",
    runs: Annotated[int, typer.Option()] = 5,
    client_id: Annotated[str, typer.Option()] = "demo",
    client_brand: Annotated[str, typer.Option()] = "Asana",
    live: Annotated[bool, typer.Option("--live")] = False,
    model_provider: Annotated[str, typer.Option()] = "anthropic",
    out: Annotated[str, typer.Option()] = "data/audits/smoke_runs.jsonl",
    aggregate_out: Annotated[str, typer.Option()] = "data/aggregates/smoke_aggregate.json",
    pdf_out: Annotated[str, typer.Option()] = "data/reports/smoke_report.pdf",
):
    """Run prompt N times, compute variance, generate PDF."""
    out_path = Path(out)
    agg_path = Path(aggregate_out)
    pdf_path = Path(pdf_out)

    if out_path.exists():
        out_path.unlink()

    mode = "LIVE" if live else "STUB"
    rprint(f"[cyan]Running {prompt_id} x {runs} ({mode})...[/cyan]")
    for i in range(1, runs + 1):
        run(
            client_id=client_id,
            client_brand=client_brand,
            prompt_id=prompt_id,
            run_index=i,
            live=live,
            model_provider=model_provider,
            out=out,
        )

    rows = read_jsonl(out_path)
    if not rows:
        raise RuntimeError("No rows produced")

    objs = [VisibilityObj.model_validate(r) for r in rows]
    scoring_cfg = _load_json(Path("config/scoring_v1.json"))
    summ = summarize_anchor(objs, scoring_cfg)

    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg_path.write_text(
        json.dumps(summ, indent=2, default=str),
        encoding="utf-8",
    )

    cap_str = ", ".join(summ["cap_reasons"]) or "none"
    lines = [
        f"Prompt: {prompt_id}",
        f"Brand: {client_brand}",
        f"Runs: {summ['run_count']}",
        "",
        "=== MENTION ===",
        f"  Rate: {summ['mention_rate']:.0%}"
        f" (stable={summ['mention_stable']})",
        "",
        "=== RANK ===",
        f"  Values: {summ['rank_values']}",
        f"  Spread: {summ['rank_spread']}"
        f" (stable={summ['rank_stable']})",
        "",
        "=== LIST STABILITY ===",
        f"  Jaccard: {summ['list_stability_score']:.2f}"
        f" (stable={summ['list_stable']})",
        "",
        "=== SCORING ===",
        f"  Raw: {summ['raw_score']:.3f}",
        f"  Cap: {summ['confidence_cap']:.2f}",
        f"  Capped: {summ['capped_score']:.3f}",
        f"  Variance: {summ['high_variance']}",
        f"  Reasons: {cap_str}",
    ]

    write_simple_pdf(pdf_path, "AI Visibility Smoke Report", lines)

    rprint(f"\n[bold]Results:[/bold]")
    for ln in lines:
        rprint(f"  {ln}")
    rprint(f"\n[magenta]Aggregate -> {agg_path}[/magenta]")
    rprint(f"[magenta]PDF -> {pdf_path}[/magenta]")

@app.command()
def compare(
    prompt_id: Annotated[str, typer.Option()] = "PM-D01",
    runs: Annotated[int, typer.Option()] = 5,
    client_id: Annotated[str, typer.Option()] = "demo",
    client_brand: Annotated[str, typer.Option()] = "Asana",
    live: Annotated[bool, typer.Option("--live")] = False,
    providers: Annotated[str, typer.Option()] = "anthropic,openai,google",
    out_dir: Annotated[str, typer.Option()] = "data",
):
    """Run prompt across multiple providers, compute cross-model agreement."""
    from .cross_model import aggregate_cross_model

    provider_list = [p.strip() for p in providers.split(",")]
    mode = "LIVE" if live else "STUB"
    rprint(f"[cyan]Compare {prompt_id} x {runs} runs x {len(provider_list)} providers ({mode})[/cyan]")

    per_provider = {}
    for prov in provider_list:
        rprint(f"\n[bold]--- {prov.upper()} ---[/bold]")
        smoke_out = f"{out_dir}/audits/compare_{prov}.jsonl"
        smoke_out_path = Path(smoke_out)
        if smoke_out_path.exists():
            smoke_out_path.unlink()

        for i in range(1, runs + 1):
            run(
                client_id=client_id,
                client_brand=client_brand,
                prompt_id=prompt_id,
                run_index=i,
                live=live,
                model_provider=prov,
                out=smoke_out,
            )

        rows = read_jsonl(smoke_out_path)
        if not rows:
            rprint(f"[red]No rows for {prov}[/red]")
            continue

        objs = [VisibilityObj.model_validate(r) for r in rows]
        scoring_cfg = _load_json(Path("config/scoring_v1.json"))
        summ = summarize_anchor(objs, scoring_cfg)
        per_provider[prov] = summ

        rprint(
            f"  mention={summ['mention_rate']:.0%} "
            f"ranks={summ['rank_values']} "
            f"raw={summ['raw_score']:.3f} "
            f"capped={summ['capped_score']:.3f}"
        )

    if len(per_provider) < 2:
        rprint("[yellow]Need >= 2 providers for cross-model comparison[/yellow]")
        return

    xm = aggregate_cross_model(per_provider)

    # Save cross-model aggregate
    agg_path = Path(f"{out_dir}/aggregates/cross_model_aggregate.json")
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg_path.write_text(
        json.dumps(xm, indent=2, default=str),
        encoding="utf-8",
    )

    # Generate cross-model PDF
    pdf_path = Path(f"{out_dir}/reports/cross_model_report.pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_lines = [
        f"Prompt: {prompt_id}",
        f"Brand: {client_brand}",
        f"Providers: {', '.join(provider_list)}",
        f"Runs per provider: {runs}",
        "",
        "=== CROSS-MODEL SUMMARY ===",
        f"  Agreement: {xm['agreement_summary']}",
        f"  Mention agreement: {xm['mention_agreement']:.0%}",
        f"  Rank agreement: {xm['rank_agreement']}",
        f"  Rank spread: {xm['rank_spread']}",
        f"  Score spread: {xm['score_spread']}",
        f"  Cross-model score: {xm['cross_model_score']:.4f}",
        f"  All stable: {xm['cross_model_stable']}",
        "",
        "=== PER-PROVIDER ===",
    ]
    for prov in provider_list:
        if prov in per_provider:
            s = per_provider[prov]
            pdf_lines.extend([
                f"  {prov.upper()}:",
                f"    Mention rate: {s['mention_rate']:.0%}",
                f"    Ranks: {s['rank_values']}",
                f"    Raw: {s['raw_score']:.3f}  Capped: {s['capped_score']:.3f}",
            ])

    report_meta = {
        "client_brand": client_brand,
        "category": "Project Management Software",
        "prompt_id": prompt_id,
        "runs": runs,
        "generated_at": str(datetime.now(timezone.utc).isoformat()),
    }
    write_cross_model_pdf(pdf_path, xm, per_provider, report_meta)

    rprint(f"\n[bold]===  CROSS-MODEL RESULTS ===[/bold]")
    for ln in pdf_lines:
        rprint(f"  {ln}")
    rprint(f"\n[magenta]Aggregate -> {agg_path}[/magenta]")
    rprint(f"[magenta]PDF -> {pdf_path}[/magenta]")


@app.command()
def audit(
    client_id: Annotated[str, typer.Option()] = "demo",
    client_brand: Annotated[str, typer.Option()] = "Asana",
    category: Annotated[str, typer.Option()] = "Project Management Software",
    runs: Annotated[int, typer.Option()] = 5,
    live: Annotated[bool, typer.Option("--live")] = False,
    providers: Annotated[str, typer.Option()] = "anthropic,openai,google",
    out_dir: Annotated[str, typer.Option()] = "data",
):
    """Full audit: all prompts x all providers x N runs. Produces complete report."""
    from .cross_model import aggregate_cross_model

    provider_list = [p.strip() for p in providers.split(",")]
    prompts = _load_json(Path("config/prompts_v1.json"))
    scoring_cfg = _load_json(Path("config/scoring_v1.json"))
    mode = "LIVE" if live else "STUB"
    total_calls = len(prompts) * len(provider_list) * runs

    rprint(f"[bold cyan]PCOS AI Visibility Audit[/bold cyan]")
    rprint(f"  Brand: {client_brand}")
    rprint(f"  Category: {category}")
    rprint(f"  Prompts: {len(prompts)}")
    rprint(f"  Providers: {', '.join(provider_list)}")
    rprint(f"  Runs per prompt: {runs}")
    rprint(f"  Total API calls: {total_calls} ({mode})")

    if live:
        cost_low = total_calls * 0.01
        cost_high = total_calls * 0.05
        rprint(f"  [yellow]Estimated cost: ${cost_low:.2f} - ${cost_high:.2f}[/yellow]")
        from .providers.registry import get_provider
        missing = [p for p in provider_list if not get_provider(p).check_api_key()]
        if missing:
            rprint(f"  [red]Missing API keys for: {', '.join(missing)}[/red]")
            rprint(f"  [red]Set keys in .env before running --live[/red]")
            raise typer.Exit(code=1)
        
        
   

    rprint("")

    # Storage paths
    audit_out = Path(f"{out_dir}/audits/full_audit.jsonl")
    agg_dir = Path(f"{out_dir}/aggregates")
    report_dir = Path(f"{out_dir}/reports")
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    agg_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if audit_out.exists():
        audit_out.unlink()

    call_count = 0
    for prov in provider_list:
        rprint(f"[bold]--- {prov.upper()} ---[/bold]")
        for p in prompts:
            pid = p["id"]
            for i in range(1, runs + 1):
                call_count += 1
                run(
                    client_id=client_id,
                    client_brand=client_brand,
                    category=category,
                    prompt_id=pid,
                    run_index=i,
                    live=live,
                    model_provider=prov,
                    out=str(audit_out),
                )
            if call_count % 30 == 0:
                rprint(f"  [dim]{call_count}/{total_calls} calls complete[/dim]")

    rprint(f"\n[green]All {call_count} calls complete.[/green]")

    # Read all results
    rows = read_jsonl(audit_out)
    all_objs = [VisibilityObj.model_validate(r) for r in rows]
    rprint(f"  Total records: {len(all_objs)}")

    # Per-provider, per-prompt variance summaries
    from collections import defaultdict
    grouped = defaultdict(list)
    for o in all_objs:
        key = (o.model_provider, o.prompt_id)
        grouped[key].append(o)

    per_prompt_summaries = {}
    for (prov, pid), objs in sorted(grouped.items()):
        summ = summarize_anchor(objs, scoring_cfg)
        per_prompt_summaries[(prov, pid)] = summ

    # Per-provider aggregate (all prompts combined)
    provider_groups = defaultdict(list)
    for o in all_objs:
        provider_groups[o.model_provider].append(o)

    per_provider_agg = {}
    for prov, objs in sorted(provider_groups.items()):
        summ = summarize_anchor(objs, scoring_cfg)
        per_provider_agg[prov] = summ

    # Cross-model aggregate
    xm = aggregate_cross_model(per_provider_agg)

    # Save aggregates
    full_agg = {
        "meta": {
            "client_id": client_id,
            "client_brand": client_brand,
            "category": category,
            "providers": provider_list,
            "prompts": len(prompts),
            "runs_per_prompt": runs,
            "total_calls": total_calls,
            "mode": mode,
        },
        "cross_model": xm,
        "per_provider": per_provider_agg,
        "per_prompt": {
            f"{prov}:{pid}": s
            for (prov, pid), s in per_prompt_summaries.items()
        },
    }
    agg_path = agg_dir / "full_audit_aggregate.json"
    agg_path.write_text(
        json.dumps(full_agg, indent=2, default=str),
        encoding="utf-8",
    )

    # Generate PDF
    pdf_path = report_dir / "full_audit_report.pdf"
    report_meta = {
        "client_brand": client_brand,
        "category": category,
        "prompt_id": f"ALL ({len(prompts)} prompts)",
        "runs": runs,
        "generated_at": str(datetime.now(timezone.utc).isoformat()),
    }
    write_cross_model_pdf(pdf_path, xm, per_provider_agg, report_meta)

    # Print summary
    rprint(f"\n[bold]=== AUDIT COMPLETE ===[/bold]")
    rprint(f"  Cross-model score: {xm.get('cross_model_score', 0):.4f}")
    rprint(f"  Agreement: {xm.get('agreement_summary', 'N/A')}")
    rprint(f"  Score spread: {xm.get('score_spread', 0)}")
    for prov in provider_list:
        if prov in per_provider_agg:
            s = per_provider_agg[prov]
            rprint(
                f"  {prov.upper()}: mention={s['mention_rate']:.0%} "
                f"raw={s['raw_score']:.3f} capped={s['capped_score']:.3f}"
            )

    # Per-family breakdown
    family_groups = defaultdict(list)
    for (prov, pid), s in per_prompt_summaries.items():
        fam = next((p["family"] for p in prompts if p["id"] == pid), "unknown")
        family_groups[fam].append(s)

    rprint(f"\n[bold]Per-Family Scores:[/bold]")
    for fam in sorted(family_groups.keys()):
        scores = [s["capped_score"] for s in family_groups[fam]]
        from statistics import mean as _mean
        avg = _mean(scores) if scores else 0
        rprint(f"  {fam}: avg_capped={avg:.3f} ({len(scores)} prompt-provider combos)")

    rprint(f"\n[magenta]JSONL -> {audit_out}[/magenta]")
    rprint(f"[magenta]Aggregate -> {agg_path}[/magenta]")
    rprint(f"[magenta]PDF -> {pdf_path}[/magenta]")