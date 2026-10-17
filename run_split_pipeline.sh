#!/usr/bin/env bash
#
# Chained pipeline for the split clips (N_PARTS x 5 s per group, default 6).
# The Cosmos3 server must already be running; this script does not start it.
#
# Flow, one ROUND per part p = 1..N_PARTS (all groups advance together):
#   round 1:  resources/real_frames/sample_<x>.jpeg  ->  <x>.1 input frame
#   round p:  prompt-improvement.py  (only the <x>.p templates, each with its OWN input frame)
#             main.py                (one call per group -> outputs/.../group_<x>/<x>.p_rf.mp4)
#             extract_last_frame.py  (last frame of <x>.p -> input frame of <x>.(p+1))
#   after the last round: concat_videos.py in every group folder -> group_<x>_full.mp4
#
# Prompt improvement is therefore called once per round, and only ever sees the frame that
# belongs to that exact part (the last frame of the previous clip), never frames from other parts.
#
# Resumable: finished clips are kept and skipped, so re-running continues where it stopped.
# A group whose clip fails is dropped from later rounds (the chain cannot continue without
# its last frame) and is not concatenated; the other groups carry on.
#
# Every step writes its own log into logs/<run timestamp>/.
# A failing step never aborts the script.
#
# Usage (from the project root, no sudo needed):
#   ./run_pipeline_split.sh
#   PYTHON=/path/to/venv/bin/python ./run_pipeline_split.sh
#   ONLY_GROUPS="1 3" N_PARTS=2 ./run_pipeline_split.sh      # quick partial run
#   DROP_FIRST_FRAME=0 ./run_pipeline_split.sh               # keep the duplicated seam frame
#
# Needs: bash >= 4.4, ffmpeg on PATH, and the sibling scripts prompt-improvement.py, main.py,
# extract_last_frame.py, concat_videos.py.

# No `set -e` on purpose: single failures must not kill the run.
set -uo pipefail

if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
    echo "bash >= 4.4 is required (found $BASH_VERSION)" >&2
    exit 1
fi

# Config
PYTHON="${PYTHON:-python3}"
PROMPT_IMPROVEMENT="${PROMPT_IMPROVEMENT:-prompt-improvement.py}"
MAIN_PY="${MAIN_PY:-main.py}"
EXTRACT_PY="${EXTRACT_PY:-extract_last_frame.py}"
CONCAT_PY="${CONCAT_PY:-concat_videos.py}"
SERVER_URL="${SERVER_URL:-http://localhost:8000}"

FRAMES_DIR="${FRAMES_DIR:-resources/real_frames}"            # sample_<x>.jpeg = first image of group x
BASE_PROMPTS="${BASE_PROMPTS:-resources/prompts_split.json}"
BATCH_ROOT="${BATCH_ROOT:-resources/split_batch}"            # per-round prompts + conditioning frames
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs/split_batch}"          # per-group clip folders
N_PARTS="${N_PARTS:-6}"
ID_SUFFIX="${ID_SUFFIX:-rf}"                                 # must match prompt-improvement --id-suffix
DROP_FIRST_FRAME="${DROP_FIRST_FRAME:-1}"                    # 1 = drop duplicated seam frame when joining
ONLY_GROUPS="${ONLY_GROUPS:-}"                               # space-separated subset, e.g. "1 3"
BATCH_ROOT="${BATCH_ROOT:-resources/split_batch${OUT_SUFFIX}}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs${OUT_SUFFIX}/split_batch}"

BACKEND="${BACKEND:-cosmos}"                                 # cosmos | ltx
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)   BACKEND="${2:-}"; shift 2 ;;
        --backend=*) BACKEND="${1#*=}"; shift ;;
        *) echo "unknown argument: $1 (usage: $0 [--backend cosmos|ltx])" >&2; exit 1 ;;
    esac
done

case "$BACKEND" in
    cosmos) MAIN_BACKEND="cosmos";       OUT_SUFFIX="" ;;
    ltx)    MAIN_BACKEND="ltx-full-one"; OUT_SUFFIX="_ltx" ;;
    *) echo "unknown backend '$BACKEND' (use cosmos or ltx)" >&2; exit 1 ;;
esac

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

# ---------------------------------------------------------------------------
# JSON helpers (stdlib python, no jq dependency)
# ---------------------------------------------------------------------------

# list_groups <prompts.json> <n_parts>  -> groups that have every part 1..n_parts
list_groups() {
    "$PYTHON" -I -c '
import json, sys, collections
parts = collections.defaultdict(set)
for r in json.load(open(sys.argv[1], encoding="utf-8")):
    parts[str(r["group"])].add(int(r["part"]))
need = set(range(1, int(sys.argv[2]) + 1))
print(" ".join(sorted((g for g, s in parts.items() if need <= s), key=lambda g: (len(g), g))))
' "$1" "$2"
}

# filter_prompts <src.json> <dst.json> <part> <group...>  -> rows of that part for those groups; prints row count
filter_prompts() {
    local src="$1" dst="$2" part="$3"; shift 3
    "$PYTHON" -I -c '
import json, sys
src, dst, part, groups = sys.argv[1], sys.argv[2], int(sys.argv[3]), set(sys.argv[4:])
rows = [r for r in json.load(open(src, encoding="utf-8"))
        if int(r["part"]) == part and str(r["group"]) in groups]
json.dump(rows, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(len(rows))
' "$src" "$dst" "$part" "$@"
}

# split_by_group <prompts_real.json> <dst_dir>  -> prompts_real_<g>.json per group; prints the groups found
split_by_group() {
    "$PYTHON" -I -c '
import json, sys, collections
src, dst = sys.argv[1], sys.argv[2]
by = collections.defaultdict(list)
for r in json.load(open(src, encoding="utf-8")):
    by[str(r["group"])].append(r)
for g, rows in by.items():
    json.dump(rows, open(f"{dst}/prompts_real_{g}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(" ".join(sorted(by)))
' "$1" "$2"
}

# stage_frame <src image> <dst .jpeg>  -> main.py loads <id>.jpeg, so everything is normalised to .jpeg
stage_frame() {
    local src="$1" dst="$2" ext
    mkdir -p "$(dirname "$dst")"
    ext="${src##*.}"; ext="${ext,,}"
    if [[ "$ext" == "jpeg" ]]; then
        cp -f "$src" "$dst"
    else
        ffmpeg -y -loglevel error -i "$src" -q:v 1 "$dst"
    fi
}

clip_path() {  # <group> <part>
    echo "$OUTPUTS_ROOT/group_${1}/${1}.${2}_${ID_SUFFIX}.mp4"
}

# ---------------------------------------------------------------------------
# Setup: which groups run
# ---------------------------------------------------------------------------
log "Run $RUN_ID started. Logs: $LOG_DIR"
log "Config: parts=$N_PARTS prompts=$BASE_PROMPTS frames=$FRAMES_DIR outputs=$OUTPUTS_ROOT drop_first_frame=$DROP_FIRST_FRAME"

if [[ ! -f "$BASE_PROMPTS" ]]; then
    log "ERROR: $BASE_PROMPTS not found"
    exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
    log "ERROR: ffmpeg not found on PATH"
    exit 1
fi
if ! curl -sf "$SERVER_URL/v1/models" >/dev/null 2>&1; then
    log "WARNING: Cosmos3 server not reachable at $SERVER_URL - generation will likely fail"
fi

read -ra COMPLETE_GROUPS <<< "$(list_groups "$BASE_PROMPTS" "$N_PARTS")"

declare -A SAMPLE_FOR=()
shopt -s nullglob
for f in "$FRAMES_DIR"/sample_*.jpeg "$FRAMES_DIR"/sample_*.jpg "$FRAMES_DIR"/sample_*.png; do
    b="$(basename "$f")"; stem="${b%.*}"
    SAMPLE_FOR["${stem#sample_}"]="$f"
done
shopt -u nullglob

declare -A FAILED=()      # group -> reason
ALL_GROUPS=()             # groups that started
ALIVE=()                  # groups still progressing

for x in "${COMPLETE_GROUPS[@]}"; do
    if [[ -n "$ONLY_GROUPS" && " $ONLY_GROUPS " != *" $x "* ]]; then
        continue
    fi
    if [[ -z "${SAMPLE_FOR[$x]:-}" ]]; then
        log "[group_${x}] no sample_${x}.* in $FRAMES_DIR, skipping group"
        continue
    fi
    if stage_frame "${SAMPLE_FOR[$x]}" "$BATCH_ROOT/round_1/frames/${x}.1.jpeg"; then
        ALL_GROUPS+=("$x"); ALIVE+=("$x")
    else
        log "[group_${x}] could not stage ${SAMPLE_FOR[$x]}, skipping group"
    fi
done

if [[ ${#ALIVE[@]} -eq 0 ]]; then
    log "ERROR: no runnable groups (need sample_<x>.jpeg for groups with parts 1..$N_PARTS in $BASE_PROMPTS)"
    exit 1
fi
log "Groups: ${ALIVE[*]}"

# ---------------------------------------------------------------------------
# Rounds
# ---------------------------------------------------------------------------
for (( p = 1; p <= N_PARTS; p++ )); do
    if [[ ${#ALIVE[@]} -eq 0 ]]; then
        log "No groups left alive, stopping before round $p"
        break
    fi

    round_dir="$BATCH_ROOT/round_${p}"
    frames_in="$round_dir/frames"
    mkdir -p "$round_dir"
    log "===== ROUND $p/$N_PARTS: groups ${ALIVE[*]} ====="

    # 1. which groups still need this part?
    todo=()
    for x in "${ALIVE[@]}"; do
        if [[ -s "$(clip_path "$x" "$p")" ]]; then
            log "[${x}.${p}] clip already exists, skipping"
        elif [[ ! -f "$frames_in/${x}.${p}.jpeg" ]]; then
            log "[${x}.${p}] input frame missing, group dropped"
            FAILED["$x"]="round $p: no input frame"
        else
            todo+=("$x")
        fi
    done

    if [[ ${#todo[@]} -gt 0 ]]; then
        # 2. prompt improvement: only the <x>.p templates of the groups that need them
        rm -f "$round_dir/prompts_real.json" "$round_dir"/prompts_real_*.json
        round_prompts="$round_dir/prompts_input.json"
        n_rows="$(filter_prompts "$BASE_PROMPTS" "$round_prompts" "$p" "${todo[@]}")"
        log "[round $p] prompt improvement for ${n_rows:-0} template(s): ${todo[*]/%/.$p}"

        run_step "$LOG_DIR/prompt-improvement_round${p}.log" "$PYTHON" "$PROMPT_IMPROVEMENT" \
            --prompts "$round_prompts" \
            --frames-dir "$frames_in" \
            --out-dir "$round_dir" \
            --id-suffix "$ID_SUFFIX"
        rc=$?

        improved=""
        if [[ -s "$round_dir/prompts_real.json" ]]; then
            improved="$(split_by_group "$round_dir/prompts_real.json" "$round_dir")"
        fi
        if [[ $rc -ne 0 ]]; then
            log "[round $p] prompt improvement exited with code $rc (see $LOG_DIR/prompt-improvement_round${p}.log)"
        fi

        # 3. generate per group straight into its own folder
        for x in "${todo[@]}"; do
            if [[ " $improved " != *" $x "* ]]; then
                log "[${x}.${p}] prompt improvement produced nothing, group dropped"
                FAILED["$x"]="round $p: prompt improvement failed"
                continue
            fi

            log "[${x}.${p}] generating -> $OUTPUTS_ROOT/group_${x}"
            PROMPTS="$round_dir/prompts_real_${x}.json" \
            CABIN_IMAGE_PREFIX="$round_dir/cabin_conditioned/" \
            OUTPUT_DIR="$OUTPUTS_ROOT/group_${x}" \
                run_step "$LOG_DIR/main_round${p}_group${x}.log" "$PYTHON" "$MAIN_PY" --backend "$MAIN_BACKEND"

            # main.py exits 0 even when a clip failed, so the clip file is the source of truth
            if [[ -s "$(clip_path "$x" "$p")" ]]; then
                log "[${x}.${p}] SUCCESS"
            else
                log "[${x}.${p}] FAILED (exit code $rc, see $LOG_DIR/main_round${p}_group${x}.log), group dropped"
                FAILED["$x"]="round $p: generation failed"
            fi
        done
    fi

    # 4. survivors = groups with a clip for this part; extract the frame that starts the next part
    survivors=()
    for x in "${ALIVE[@]}"; do
        clip="$(clip_path "$x" "$p")"
        [[ -s "$clip" ]] || continue
        if (( p < N_PARTS )); then
            next_frame="$BATCH_ROOT/round_$((p + 1))/frames/${x}.$((p + 1)).jpeg"
            if run_step "$LOG_DIR/extract_${x}.${p}.log" "$PYTHON" "$EXTRACT_PY" "$clip" "$next_frame" \
                && [[ -s "$next_frame" ]]; then
                log "[${x}.${p}] last frame -> $next_frame"
            else
                log "[${x}.${p}] last-frame extraction FAILED, group dropped"
                FAILED["$x"]="round $p: frame extraction failed"
                continue
            fi
        fi
        survivors+=("$x")
    done
    ALIVE=("${survivors[@]}")
done

# ---------------------------------------------------------------------------
# Concat: one joined clip per finished group, inside its own folder
# ---------------------------------------------------------------------------
log "===== CONCAT ====="
CONCAT_OK=(); CONCAT_FAIL=()
concat_flags=()
[[ "$DROP_FIRST_FRAME" == "1" ]] && concat_flags+=(--drop-first-frame)

for x in "${ALL_GROUPS[@]}"; do
    if [[ -n "${FAILED[$x]:-}" ]]; then
        log "[group_${x}] not concatenated (${FAILED[$x]})"
        continue
    fi

    clips=()
    for (( p = 1; p <= N_PARTS; p++ )); do
        clips+=("$(clip_path "$x" "$p")")
    done
    missing=0
    for c in "${clips[@]}"; do [[ -s "$c" ]] || missing=1; done
    if (( missing )); then
        log "[group_${x}] not concatenated (a part is missing)"
        FAILED["$x"]="missing part(s) at concat time"
        continue
    fi

    final="$OUTPUTS_ROOT/group_${x}/group_${x}_full.mp4"
    log "[group_${x}] concatenating ${#clips[@]} clips -> $final"
    run_step "$LOG_DIR/concat_group${x}.log" "$PYTHON" "$CONCAT_PY" \
        -o "$final" --keep-order "${concat_flags[@]}" "${clips[@]}"
    rc=$?
    if [[ $rc -eq 0 && -s "$final" ]]; then
        log "[group_${x}] SUCCESS"
        CONCAT_OK+=("group_${x}")
    else
        log "[group_${x}] CONCAT FAILED (exit code $rc, see $LOG_DIR/concat_group${x}.log)"
        CONCAT_FAIL+=("group_${x}")
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
dropped=()
for x in "${ALL_GROUPS[@]}"; do
    [[ -n "${FAILED[$x]:-}" ]] && dropped+=("group_${x} (${FAILED[$x]})")
done

log "===== SUMMARY ====="
log "Joined clips   (${#CONCAT_OK[@]}): ${CONCAT_OK[*]:-none}"
log "Concat failed  (${#CONCAT_FAIL[@]}): ${CONCAT_FAIL[*]:-none}"
log "Groups dropped (${#dropped[@]}): ${dropped[*]:-none}"
log "Done."
exit 0