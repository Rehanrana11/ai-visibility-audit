from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def write_simple_pdf(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    y = height - 72

    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, y, title)
    y -= 30

    c.setFont("Helvetica", 9)
    c.drawString(72, y, "PCOS Visibility Engine - Audit Report")
    y -= 20
    c.line(72, y, width - 72, y)
    y -= 20

    c.setFont("Helvetica", 10)
    for ln in lines:
        if y < 72:
            c.showPage()
            y = height - 72
            c.setFont("Helvetica", 10)
        c.drawString(72, y, ln[:120])
        y -= 14

    c.save()

def write_cross_model_pdf(
    path: Path,
    cross_model: dict,
    per_provider: dict[str, dict],
    meta: dict,
) -> None:
    """Generate a structured cross-model PDF report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    y = height - 72

    def check_page(needed: int = 40) -> int:
        nonlocal y
        if y < needed:
            c.showPage()
            y = height - 72
        return y

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, y, "AI Visibility Cross-Model Report")
    y -= 24
    c.setFont("Helvetica", 9)
    c.drawString(72, y, "PCOS Visibility Engine - Multi-Provider Audit")
    y -= 16
    c.setFont("Helvetica", 9)
    c.drawString(72, y, f"Generated: {meta.get('generated_at', 'N/A')}")
    y -= 20
    c.line(72, y, width - 72, y)
    y -= 24

    # Section 1: Executive Summary
    c.setFont("Helvetica-Bold", 13)
    c.drawString(72, y, "1. Executive Summary")
    y -= 20
    c.setFont("Helvetica", 10)
    summary_lines = [
        f"Client Brand: {meta.get('client_brand', 'N/A')}",
        f"Category: {meta.get('category', 'N/A')}",
        f"Prompt: {meta.get('prompt_id', 'N/A')}",
        f"Providers: {', '.join(cross_model.get('providers', []))}",
        f"Runs per provider: {meta.get('runs', 'N/A')}",
        "",
        f"Cross-Model Score: {cross_model.get('cross_model_score', 0):.4f}",
        f"Agreement: {cross_model.get('agreement_summary', 'N/A')}",
        f"Mention Agreement: {cross_model.get('mention_agreement', 0):.0%}",
        f"Rank Agreement: {cross_model.get('rank_agreement', 'N/A')}",
        f"Score Spread: {cross_model.get('score_spread', 0)}",
        f"All Stable: {cross_model.get('cross_model_stable', 'N/A')}",
    ]
    for ln in summary_lines:
        y = check_page()
        c.drawString(90, y, ln[:110])
        y -= 14
    y -= 10

    # Section 2: Per-Provider Results
    y = check_page(60)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(72, y, "2. Per-Provider Results")
    y -= 20

    providers = cross_model.get("providers", [])
    for prov in providers:
        y = check_page(80)
        summ = per_provider.get(prov, {})
        c.setFont("Helvetica-Bold", 11)
        c.drawString(90, y, prov.upper())
        y -= 16
        c.setFont("Helvetica", 10)
        prov_lines = [
            f"Mention Rate: {summ.get('mention_rate', 0):.0%}  (stable: {summ.get('mention_stable', 'N/A')})",
            f"Rank Values: {summ.get('rank_values', [])}  (spread: {summ.get('rank_spread', 'N/A')})",
            f"List Stability: {summ.get('list_stability_score', 0):.2f}  (stable: {summ.get('list_stable', 'N/A')})",
            f"Raw Score: {summ.get('raw_score', 0):.3f}",
            f"Confidence Cap: {summ.get('confidence_cap', 1.0):.2f}  Reasons: {', '.join(summ.get('cap_reasons', [])) or 'none'}",
            f"Capped Score: {summ.get('capped_score', 0):.3f}",
            f"High Variance: {summ.get('high_variance', False)}",
        ]
        for ln in prov_lines:
            y = check_page()
            c.drawString(108, y, ln[:100])
            y -= 14
        y -= 8

    # Section 3: Cross-Model Comparison Table
    y = check_page(80)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(72, y, "3. Cross-Model Comparison")
    y -= 20

    # Table header
    c.setFont("Helvetica-Bold", 9)
    cols = [90, 200, 280, 340, 410, 480]
    headers = ["Provider", "Mention", "Avg Rank", "Raw", "Capped", "Variance"]
    for i, h in enumerate(headers):
        c.drawString(cols[i], y, h)
    y -= 4
    c.line(90, y, width - 72, y)
    y -= 14

    # Table rows
    c.setFont("Helvetica", 9)
    rank_means = cross_model.get("rank_means", {})
    raw_scores = cross_model.get("raw_scores", {})
    capped_scores = cross_model.get("capped_scores", {})
    mention_rates = cross_model.get("mention_rates", {})
    variance_flags = cross_model.get("variance_flags", {})

    for prov in providers:
        y = check_page()
        c.drawString(cols[0], y, prov.upper())
        c.drawString(cols[1], y, f"{mention_rates.get(prov, 0):.0%}")
        rm = rank_means.get(prov)
        c.drawString(cols[2], y, f"{rm}" if rm is not None else "N/A")
        c.drawString(cols[3], y, f"{raw_scores.get(prov, 0):.3f}")
        c.drawString(cols[4], y, f"{capped_scores.get(prov, 0):.3f}")
        c.drawString(cols[5], y, str(variance_flags.get(prov, False)))
        y -= 14

    # Footer
    y -= 20
    y = check_page()
    c.line(72, y, width - 72, y)
    y -= 16
    c.setFont("Helvetica", 8)
    c.drawString(72, y, "PCOS Visibility Engine | Confidential | Do not distribute without authorization")

    c.save()