# AI Visibility Audit (aivis) v2.0.0

**PCOS Visibility Engine** — Audit brand visibility across AI models (Claude, ChatGPT, Gemini).

## What It Does

Sends 30 frozen prompts to AI models asking recommendation questions (e.g., "What's the best project management tool?"), runs each prompt 5 times per model, parses ranked tool lists from responses, checks if your brand appears and at what rank, computes variance/stability metrics, applies confidence caps, and generates PDF reports.

## Quick Start
```
# Install
poetry install

# Copy and fill in API keys
cp .env.example .env

# Stub smoke test (no API keys needed)
poetry run aivis smoke --prompt-id PM-D01 --runs 5

# Stub smoke per provider
poetry run aivis smoke --prompt-id PM-D01 --runs 5 --model-provider openai

# Cross-model comparison (stub)
poetry run aivis compare --prompt-id PM-D01 --runs 5

# Full audit - 30 prompts x 3 models x 5 runs = 450 calls (stub)
poetry run aivis audit --runs 5

# Full audit - LIVE (requires API keys in .env)
poetry run aivis audit --runs 5 --live
```

## Commands

| Command | Description |
|---------|-------------|
| `aivis run` | Single prompt, single run |
| `aivis smoke` | Single prompt x N runs, variance report |
| `aivis compare` | Single prompt x N runs x M providers, cross-model report |
| `aivis audit` | All 30 prompts x all providers x N runs, full report |

## Key Options

- `--model-provider` anthropic/openai/google (default: anthropic)
- `--providers` comma-separated list (default: anthropic,openai,google)
- `--live` use real API calls (default: stub mode)
- `--client-brand` brand to track (default: Asana)
- `--runs` runs per prompt (default: 5)

## Architecture

- 30 frozen prompts across 5 clusters: category, comparison, problem, buyer_intent, long_tail
- 3 AI providers: Anthropic Claude, OpenAI ChatGPT, Google Gemini
- Per-run scoring: mention (50%), rank (40%), citation (10%)
- 6 confidence cap rules for variance control
- Cross-model agreement classification: STRONG_AGREEMENT / MODERATE_AGREEMENT / DISAGREEMENT

## Output

- `data/audits/` — Raw JSONL records (one per API call)
- `data/aggregates/` — JSON variance summaries and cross-model aggregates
- `data/reports/` — PDF reports with executive summary, per-provider detail, comparison table

## Tech Stack

Python 3.13, Poetry, Pydantic v2, httpx, Typer, Rich, ReportLab. JSONL storage. No database.

## Tests
```
poetry run pytest -q    # 63 tests
```

## File Structure
```
config/prompts_v1.json          30 frozen prompts
config/scoring_v1.json          Scoring weights + 6 cap rules
config/models.json              3 provider configs
src/aivis/cli.py                CLI commands (run, smoke, compare, audit)
src/aivis/runner.py             Thin dispatcher to providers
src/aivis/parser.py             Response parser + name normalization
src/aivis/scorer.py             Mention/rank/citation scoring
src/aivis/variance.py           Jaccard stability + confidence caps
src/aivis/cross_model.py        Cross-model aggregation
src/aivis/reporter.py           PDF report generation
src/aivis/storage.py            JSONL read/write
src/aivis/models.py             VisibilityObj (36 fields)
src/aivis/providers/            Provider implementations
  anthropic_provider.py         Claude API + stub
  openai_provider.py            ChatGPT API + stub
  google_provider.py            Gemini API + stub
  registry.py                   Provider name -> class mapping
  base.py                       AIProvider Protocol
```