#!/usr/bin/env bash
set -euo pipefail

EXECUTION="${1:?execution: oxygen or isolated}"
K="${2:?steps per frame}"
GPU="${3:?gpu index}"
PORT="${4:?server port}"
OUTPUT_ROOT="${5:?output root}"

REPO_ROOT="/home/lixiangyu/oxygen_ws/OxyGen"
EXP_ROOT="/home/lixiangyu/oxygen_ws/libero_exp"
CHECKPOINT="$EXP_ROOT/checkpoints/pi05_libero_modelscope"
ADAPTER="$EXP_ROOT/language_adapter_runs/v6_suffix_lora_20260812/training/confirm2_lr3e6/adapter_step_250.npz"
NORM_STATS="$CHECKPOINT/assets/physical-intelligence/libero/norm_stats.json"
CLIENT_PYTHON="$EXP_ROOT/client_venv/bin/python"
CLIENT_PATH="$REPO_ROOT/packages/openpi-client/src:$EXP_ROOT/openpi_subtask_generation/third_party/libero"
RUN_ROOT="$OUTPUT_ROOT/$EXECUTION/k$K"
LOG_ROOT="$OUTPUT_ROOT/logs/$EXECUTION/k$K"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
uv run python -m experiments.libero_language_adapter.serve_adapter \
  --checkpoint "$CHECKPOINT" --norm-stats "$NORM_STATS" \
  --rank 16 --adapter "$ADAPTER" --port "$PORT" \
  --request-mode new_each_call --steps-per-frame "$K" \
  --max-decoding-steps 20 --temperature 0.1 --execution "$EXECUTION" \
  >"$LOG_ROOT/server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 240); do
  if curl --fail --silent "http://127.0.0.1:$PORT/healthz" >/dev/null; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server failed: $LOG_ROOT/server.log" >&2
    exit 1
  fi
  sleep 2
done
curl --fail --silent "http://127.0.0.1:$PORT/healthz" >/dev/null

run_rollout() {
  local task_id="$1"
  local episode="$2"
  local output="$3"
  PYTHONPATH="$CLIENT_PATH" LIBERO_CONFIG_PATH="$EXP_ROOT/libero_config" \
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$GPU" PYOPENGL_PLATFORM=egl \
  "$CLIENT_PYTHON" experiments/libero_language_adapter/rollout_review.py \
    --host 127.0.0.1 --port "$PORT" --output-root "$output" \
    --suites libero_10 --task-id "$task_id" --episodes "$episode" \
    --replan-steps 5 --video-fps 100 --source-control-hz 20 \
    --wall-clock-timeline
}

# Compile all shapes encountered by this natural EOS workload. This output is
# retained for audit but excluded from the comparison pages and statistics.
run_rollout 3 1 "$RUN_ROOT/warmup" >"$LOG_ROOT/warmup.log" 2>&1
run_rollout 3 0 "$RUN_ROOT/task3" >"$LOG_ROOT/task3.log" 2>&1
run_rollout 8 0 "$RUN_ROOT/task8" >"$LOG_ROOT/task8.log" 2>&1
