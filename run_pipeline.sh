#!/usr/bin/env bash
#
# Pipeline (the Cosmos3 server must already be running; this script does not start it):
#   1. For every resources/real_frames/sample_<x>.jpeg -> prompt-improvement.py -> resources/real_batch_<x>/
#   2. For every resources/real_batch_<x>/ -> main.py -> outputs/real_batch_<x>/
#
# Every step writes its own log into logs/<run timestamp>/.
# A failing step never aborts the script.
#
# Usage (from the project root, no sudo needed):
#   ./run_pipeline.sh
#   PYTHON=/path/to/venv/bin/python ./run_pipeline.sh

# No `set -e` on purpose: single failures must not kill the run.
set -uo pipefail

# Config
PYTHON="${PYTHON:-python3}"
PROMPT_IMPROVEMENT="${PROMPT_IMPROVEMENT:-prompt-improvement.py}"
MAIN_PY="${MAIN_PY:-main.py}"
SERVER_URL="${SERVER_URL:-http://localhost:8000}"

FRAMES_DIR="resources/real_frames"
BASE_PROMPTS="resources/prompts_longer.json"
BATCH_ROOT="resources"
OUTPUTS_ROOT="outputs"

cd "$(dirname "$(readlink -f "$0")")" || exit 1

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/${RUN_ID}"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY"
}

# run_step <logfile> <cmd...>  -> streams to console + logfile, returns cmd's exit code
run_step() {
    local logfile="$1"; shift
    "$@" 2>&1 | tee "$logfile"
    return "${PIPESTATUS[0]}"
}

PI_OK=(); PI_FAIL=()
GEN_OK=(); GEN_FAIL=(); GEN_SKIP=()

log "Run $RUN_ID started. Logs: $LOG_DIR"

# ---------------------------------------------------------------------------
# Phase 1: prompt improvement per frame
# ---------------------------------------------------------------------------
log "===== PHASE 1: prompt improvement ====="
shopt -s nullglob
frames=("$FRAMES_DIR"/sample_*.jpeg "$FRAMES_DIR"/sample_*.jpg)
shopt -u nullglob

if [[ ${#frames[@]} -eq 0 ]]; then
    log "WARNING: no sample_*.jpeg found in $FRAMES_DIR"
fi

for frame in "${frames[@]}"; do
    base="$(basename "$frame")"
    stem="${base%.*}"
    x="${stem#sample_}"
    out_dir="$BATCH_ROOT/real_batch_${x}"
    step_log="$LOG_DIR/prompt-improvement_sample_${x}.log"

    log "[sample_${x}] prompt improvement -> $out_dir"
    run_step "$step_log" "$PYTHON" "$PROMPT_IMPROVEMENT" \
        --prompts "$BASE_PROMPTS" \
        --frame "$frame" \
        --out-dir "$out_dir"
    rc=$?

    # exit code 0 alone is not enough: the script also "succeeds" if every template was skipped/failed
    if [[ $rc -eq 0 ]] && grep -qE "Wrote [1-9][0-9]* redacted prompts" "$step_log"; then
        log "[sample_${x}] SUCCESS ($(grep -E 'Wrote [0-9]+ redacted' "$step_log" | tail -n1))"
        PI_OK+=("sample_${x}")
    else
        log "[sample_${x}] FAILED (exit code $rc, see $step_log)"
        PI_FAIL+=("sample_${x}")
    fi
done

# ---------------------------------------------------------------------------
# Phase 2: generate clips for every real_batch_x
# ---------------------------------------------------------------------------
log "===== PHASE 2: video generation ====="
if ! curl -sf "$SERVER_URL/v1/models" >/dev/null 2>&1; then
    log "WARNING: Cosmos3 server not reachable at $SERVER_URL - generation will likely fail"
fi

shopt -s nullglob
batches=("$BATCH_ROOT"/real_batch_*/)
shopt -u nullglob

for batch_path in "${batches[@]}"; do
    batch_path="${batch_path%/}"
    batch="$(basename "$batch_path")"
    prompts_file="$batch_path/prompts_real.json"
    frames_prefix="$batch_path/cabin_conditioned/"
    out_dir="$OUTPUTS_ROOT/$batch"
    step_log="$LOG_DIR/main_${batch}.log"

    if [[ ! -f "$prompts_file" ]]; then
        log "[$batch] no prompts_real.json, skipping"
        GEN_SKIP+=("$batch")
        continue
    fi

    log "[$batch] generating -> $out_dir"
    # main.py reads these three settings from the environment
    PROMPTS="$prompts_file" \
    CABIN_IMAGE_PREFIX="$frames_prefix" \
    OUTPUT_DIR="$out_dir" \
        run_step "$step_log" "$PYTHON" "$MAIN_PY"
    rc=$?

    if [[ $rc -eq 0 ]] && ! grep -qE "[0-9]+ succeeded, [1-9][0-9]* failed" "$step_log"; then
        log "[$batch] SUCCESS"
        GEN_OK+=("$batch")
    elif [[ $rc -eq 0 ]]; then
        log "[$batch] PARTIAL: some clips failed ($(grep -E 'batch complete' "$step_log" | tail -n1))"
        GEN_FAIL+=("$batch (partial)")
    else
        log "[$batch] FAILED (exit code $rc, see $step_log)"
        GEN_FAIL+=("$batch")
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "===== SUMMARY ====="
log "Prompt improvement OK    (${#PI_OK[@]}): ${PI_OK[*]:-none}"
log "Prompt improvement FAIL  (${#PI_FAIL[@]}): ${PI_FAIL[*]:-none}"
log "Generation OK            (${#GEN_OK[@]}): ${GEN_OK[*]:-none}"
log "Generation FAIL/PARTIAL  (${#GEN_FAIL[@]}): ${GEN_FAIL[*]:-none}"
log "Generation skipped       (${#GEN_SKIP[@]}): ${GEN_SKIP[*]:-none}"
log "Done."
exit 0