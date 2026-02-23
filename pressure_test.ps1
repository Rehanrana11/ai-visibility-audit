# ================================
# AIVIS v2.0.0 — LOCAL PRESSURE TEST HARNESS
# Adapted from HIREINSTEIN FAANG pressure test
# Run from repo root: .\pressure_test.ps1
# ================================

$ErrorActionPreference = "Stop"
$ProjectRoot = (Get-Location).Path
$OutDir = Join-Path $ProjectRoot "out\pressure_test"
$Tag = "pt_" + (Get-Date -Format "yyyyMMdd_HHmmss")

Write-Host "
========================================="
Write-Host "AIVIS v2.0.0 PRESSURE TEST"
Write-Host "========================================="
Write-Host "[0] Root: $ProjectRoot"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# --- 1) Environment ---
Write-Host "
[1] Environment..."
python --version
poetry run python -c "import aivis; print('aivis import OK')"

# --- 2) Static Quality Gates ---
Write-Host "
[2] Lint + Tests..."
poetry run pytest -q

# --- 3) CLI Sanity ---
Write-Host "
[3] CLI sanity..."
poetry run aivis --help

# --- 4) Fingerprint ---
Write-Host "
[4] Fingerprinting..."
$FP = Join-Path $OutDir "fingerprint_$Tag.txt"
"==== FINGERPRINT ($Tag) ====" | Out-File -Encoding utf8 $FP
"Date: $(Get-Date -Format o)" | Out-File -Append -Encoding utf8 $FP
"Python: $(python --version 2>&1)" | Out-File -Append -Encoding utf8 $FP
try { "Git: $(git rev-parse HEAD)" | Out-File -Append -Encoding utf8 $FP } catch { "Git: NO_GIT" | Out-File -Append -Encoding utf8 $FP }

$configs = @("config\prompts_v1.json", "config\scoring_v1.json", "config\models.json")
foreach ($c in $configs) {
    if (Test-Path $c) {
        $h = (Get-FileHash $c -Algorithm SHA256).Hash
        "$h  $c" | Out-File -Append -Encoding utf8 $FP
    }
}
Write-Host "[4] Saved: $FP"

# --- 5) Smoke Run 1 (stub, single provider) ---
Write-Host "
[5] Smoke Run 1 (stub, anthropic, 1 prompt x 5 runs)..."
poetry run aivis smoke --prompt-id PM-D01 --runs 5 --model-provider anthropic

# --- 6) Smoke Run 2 (stub, identical) ---
Write-Host "
[6] Smoke Run 2 (stub, identical for determinism check)..."
poetry run aivis smoke --prompt-id PM-D01 --runs 5 --model-provider anthropic

# --- 7) Cross-Provider Compare (stub) ---
Write-Host "
[7] Cross-provider compare (stub, 3 models)..."
poetry run aivis compare --prompt-id PM-D01 --runs 5

# --- 8) Determinism Check ---
Write-Host "
[8] Determinism check (hash JSONL)..."
$jsonl = "data\audits\smoke_audit.jsonl"
if (Test-Path $jsonl) {
    $h = (Get-FileHash $jsonl -Algorithm SHA256).Hash
    Write-Host "JSONL SHA256: $h"
    $lines = (Get-Content $jsonl).Count
    Write-Host "JSONL records: $lines"
} else {
    Write-Host "WARN: No smoke JSONL found"
}

# --- 9) Schema Validation ---
Write-Host "
[9] Schema validation..."
poetry run python -c "import json; rows=[json.loads(l) for l in open('data/audits/smoke_audit.jsonl')]; print(len(rows),'records'); print(len(rows[0].keys()),'fields in first record'); assert len(rows[0].keys()) >= 30, 'Schema too small'"

# --- 10) Mini Full Audit (stub, 30 prompts x 1 provider x 2 runs = 60 calls) ---
Write-Host "
[10] Mini full audit (stub, 60 calls)..."
$sw = [System.Diagnostics.Stopwatch]::StartNew()
poetry run aivis audit --runs 2 --providers anthropic
$sw.Stop()
Write-Host ("Elapsed: {0:n2}s" -f $sw.Elapsed.TotalSeconds)

# --- 11) Output Validation ---
Write-Host "
[11] Output validation..."
$files = @(
    "data\audits\full_audit.jsonl",
    "data\aggregates\full_audit_aggregate.json",
    "data\reports\full_audit_report.pdf"
)
foreach ($f in $files) {
    if (Test-Path $f) {
        $sz = (Get-Item $f).Length
        Write-Host "OK: $f ($sz bytes)"
    } else {
        Write-Host "FAIL: $f MISSING"
    }
}

# --- 12) Record Count Validation ---
Write-Host "
[12] Record count validation..."
poetry run python -c "lines=sum(1 for _ in open('data/audits/full_audit.jsonl')); print(lines,'records'); assert lines==60, f'Expected 60, got {lines}'"

# --- 13) LIVE Single Call (if key present) ---
Write-Host "
[13] Live API test (1 call)..."
try {
    poetry run aivis run --prompt-id PM-D01 --model-provider anthropic --live
    Write-Host "LIVE CALL: OK"
} catch {
    Write-Host "LIVE CALL: FAILED (check .env)"
}

Write-Host "
========================================="
Write-Host "PRESSURE TEST COMPLETE"
Write-Host "========================================="