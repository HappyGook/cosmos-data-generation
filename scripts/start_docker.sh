#!/usr/bin/env bash
#
# Starts the Cosmos3-Nano vLLM-Omni server, sharded across 2 GPUs via
# tensor parallelism, and waits until it's actually ready to accept
# requests before returning.
#
# Prerequisites (one-time):
#   - CUDA 13! For CUDA 12.x - check `start_native_server.sh`
#   - NVIDIA driver + nvidia-container-toolkit installed (`docker run --gpus all` must work)
#   - Docker logged into a HF-authenticated cache dir, or run:
#       uvx hf@latest auth login
#     once on the host so ~/.cache/huggingface has your token
#
# Usage:
#   ./start_cosmos3_server.sh
#   ./start_cosmos3_server.sh --gpus 0,1 --port 8000
#
set -euo pipefail

# Config
CONTAINER_NAME="${CONTAINER_NAME:-cosmos3-nano-server}"
IMAGE="${IMAGE:-vllm/vllm-omni:cosmos3}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-all}"                # e.g. "all" or "0,1"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
INIT_TIMEOUT="${INIT_TIMEOUT:-1800}"
VIDEO_SYNC_TIMEOUT="${VIDEO_SYNC_TIMEOUT:-3600}"  # VLLM_OMNI_VIDEO_SYNC_TIMEOUT: /v1/videos/sync generation timeout (s)
CREATE_START_TIMEOUT="${CREATE_START_TIMEOUT:-120}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(pwd)}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/home/iis/.cache/huggingface}"
DISABLE_GUARDRAILS_SERVERWIDE="${DISABLE_GUARDRAILS_SERVERWIDE:-false}"
LOG_DIR="${LOG_DIR:-./server_logs}"
READY_POLL_INTERVAL_S=10

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_cosmos3_server.log"

# Small logging helper -> stdout AND logfile
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | tee -a "$LOG_FILE"
}

dump_container_state() {
    local name="$1"
    log "---- docker inspect state for '$name' ----"
    if docker inspect "$name" \
        --format 'Status: {{.State.Status}}
        ExitCode: {{.State.ExitCode}}
        Error: {{.State.Error}}
        OOMKilled: {{.State.OOMKilled}}
        StartedAt: {{.State.StartedAt}}
        FinishedAt: {{.State.FinishedAt}}' 2>&1 | tee -a "$LOG_FILE"; then
        :
    else
        log "  (docker inspect itself failed - container may not exist)"
    fi
    log "---- last 50 lines of container logs (may be empty if it never started) ----"
    docker logs --tail 50 "$name" 2>&1 | tee -a "$LOG_FILE" || log "  (no logs available)"
    log "-------------------------------------------------------------------------"
}

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --tp) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --no-guardrails) DISABLE_GUARDRAILS_SERVERWIDE="true"; shift ;;
        *) log "unknown flag: $1"; exit 1 ;;
    esac
done

# Sanity checks
log "checking for nvidia-smi / GPU visibility..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "ERROR: nvidia-smi not found. Install the NVIDIA driver first."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv | tee -a "$LOG_FILE"

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
log "detected $GPU_COUNT GPU(s), requesting tensor-parallel-size=$TENSOR_PARALLEL_SIZE"
if [[ "$GPU_COUNT" -lt "$TENSOR_PARALLEL_SIZE" ]]; then
    log "ERROR: tensor-parallel-size ($TENSOR_PARALLEL_SIZE) exceeds visible GPU count ($GPU_COUNT)"
    exit 1
fi

if [[ ! -d "$HF_CACHE_DIR" ]]; then
    log "WARNING: $HF_CACHE_DIR not found. If model download fails with an auth error, run:"
    log "  uvx hf@latest auth login"
fi

# Clean up any previous instance
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log "removing existing container '$CONTAINER_NAME' (any prior state)"
    docker rm -f "$CONTAINER_NAME" 2>&1 | tee -a "$LOG_FILE" || true
fi


# Optional server-wide guardrail disable
DEPLOY_CONFIG_MOUNT=()
if [[ "$DISABLE_GUARDRAILS_SERVERWIDE" == "true" ]]; then
    CONFIG_PATH="$WORKSPACE_DIR/no_guardrails.yaml"
    log "writing server-wide guardrail-disable config to $CONFIG_PATH"
    cat > "$CONFIG_PATH" <<'EOF'
async_chunk: false
stages:
  - stage_id: 0
    max_num_seqs: 1
    enforce_eager: true
    trust_remote_code: true
    model_class_name: Cosmos3OmniDiffusersPipeline
    model_config:
      guardrails: false
      offload_guardrail_models: false
EOF
    DEPLOY_CONFIG_ARGS=(--deploy-config /workspace/no_guardrails.yaml)
else
    DEPLOY_CONFIG_ARGS=()
fi


# Launch
log "starting container '$CONTAINER_NAME' (image=$IMAGE, gpus=$GPUS, tp=$TENSOR_PARALLEL_SIZE, port=$PORT)"

RUN_CMD=(docker run -d
    --name "$CONTAINER_NAME"
    --gpus "$GPUS"
    -e HF_HUB_DISABLE_XET=1
    -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT="$VIDEO_SYNC_TIMEOUT"
    -v "$HF_CACHE_DIR:/root/.cache/huggingface"
    -v "$WORKSPACE_DIR:/workspace"
    -p "${PORT}:8000"
    --ipc=host
    --restart unless-stopped
    "$IMAGE"
    vllm serve nvidia/Cosmos3-Nano
    --omni
    --model-class-name Cosmos3OmniDiffusersPipeline
    --gpu-memory-utilization 0.85
    --allowed-local-media-path /workspace
    --port 8000
    --init-timeout "$INIT_TIMEOUT"
    "${DEPLOY_CONFIG_ARGS[@]}"
)
log "exact docker run command: ${RUN_CMD[*]}"

set +e
timeout "$CREATE_START_TIMEOUT" "${RUN_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
RUN_STATUS=${PIPESTATUS[0]}
set -e

if [[ "$RUN_STATUS" -eq 124 ]]; then
    log "ERROR: 'docker run' did not return within ${CREATE_START_TIMEOUT}s (create/start phase hung)."
    dump_container_state "$CONTAINER_NAME"
    exit 1
elif [[ "$RUN_STATUS" -ne 0 ]]; then
    log "ERROR: 'docker run' exited with status $RUN_STATUS."
    dump_container_state "$CONTAINER_NAME"
    exit 1
fi
log "'docker run' returned successfully (exit 0), container object created."

# confirm the container reached 'running'
CONTAINER_STATUS=$(docker inspect "$CONTAINER_NAME" --format '{{.State.Status}}' 2>&1) || CONTAINER_STATUS="unknown"
log "container status immediately after 'docker run': $CONTAINER_STATUS"
if [[ "$CONTAINER_STATUS" != "running" ]]; then
    log "ERROR: container did not enter 'running' state (status: $CONTAINER_STATUS)."
    dump_container_state "$CONTAINER_NAME"
    exit 1
fi

log "container started, tailing logs to $LOG_FILE while waiting for readiness..."
docker logs -f "$CONTAINER_NAME" >>"$LOG_FILE" 2>&1 &
TAIL_PID=$!

log "polling http://localhost:${PORT}/v1/models every ${READY_POLL_INTERVAL_S}s (timeout ${INIT_TIMEOUT}s)..."
elapsed=0
while (( elapsed < INIT_TIMEOUT )); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        kill "$TAIL_PID" 2>/dev/null || true
        log "server is READY at http://localhost:${PORT}"
        log "full startup log: $LOG_FILE"
        exit 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        kill "$TAIL_PID" 2>/dev/null || true
        log "ERROR: container is no longer running (exited during startup)."
        dump_container_state "$CONTAINER_NAME"
        exit 1
    fi
    sleep "$READY_POLL_INTERVAL_S"
    elapsed=$((elapsed + READY_POLL_INTERVAL_S))
    log "still waiting... (${elapsed}s elapsed)"
done

kill "$TAIL_PID" 2>/dev/null || true
log "ERROR: server did not become ready within ${INIT_TIMEOUT}s"
dump_container_state "$CONTAINER_NAME"
exit 1