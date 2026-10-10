#!/usr/bin/env bash
set -euo pipefail

lock_file=/mnt/lixiangyu/oxygen_ws/locks/qwen35_formal_gpu.lock
repo=/mnt/lixiangyu/repos/starVLA
work=/mnt/lixiangyu/oxygen_ws
runner="$work/remote_edit/starVLA_qwen35_tp_benchmark.py"
helper="$work/remote_edit/starVLA_qwen25vl_tp_benchmark.py"
analyzer="$work/remote_edit/analyze_qwen35_scaling_tp.py"
venv=/tmp/oxygen_qwen35_fastpath
out="$work/results/qwen35_scaling_tp_formal_20260728"
log_dir="$work/logs/qwen35_scaling_tp_formal_20260728"

export PYTHONPATH="${repo}:${work}/remote_edit:/mnt/lixiangyu/repos/Xiaomi-Robotics-0/scripts"
export TRITON_CACHE_DIR=/tmp/triton_qwen35_fastpath
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_qwen35_fastpath
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

mkdir -p "$out" "$log_dir" "$(dirname "$lock_file")"

declare -a passed_labels=()
declare -a passed_outputs=()

model_path() {
    case "$1" in
        0p8b) echo /mnt/lixiangyu/checkpoints/Qwen3.5-0.8B ;;
        2b) echo /mnt/lixiangyu/checkpoints/Qwen3.5-2B ;;
        4b) echo /mnt/lixiangyu/checkpoints/Qwen3.5-4B ;;
        9b) echo /mnt/lixiangyu/checkpoints/Qwen3.5-9B ;;
        *) return 2 ;;
    esac
}

display_label() {
    case "$1" in
        0p8b_tp1) echo 0.8B-TP1 ;;
        0p8b_tp2) echo 0.8B-TP2 ;;
        2b_tp1) echo 2B-TP1 ;;
        2b_tp2) echo 2B-TP2 ;;
        4b_tp1) echo 4B-TP1 ;;
        4b_tp2) echo 4B-TP2 ;;
        9b_tp1) echo 9B-TP1 ;;
        9b_tp2) echo 9B-TP2 ;;
        *) return 2 ;;
    esac
}

run_benchmark() {
    local label=$1
    local devices=$2
    local processes=$3
    local model=$4
    local phase=$5
    local output=$6
    shift 6
    local log="${log_dir}/${phase}_${label}.log"
    local exit_file="${log_dir}/${phase}_${label}.exit"

    echo "START ${phase} ${label} $(date --iso-8601=seconds)" | tee -a "$out/launcher.log"
    {
        printf 'CUDA_VISIBLE_DEVICES=%q %q -m torch.distributed.run --standalone --nproc_per_node=%q %q --model %q --require-fast-path --output %q' \
            "$devices" "$venv/bin/python" "$processes" "$runner" "$model" "$output"
        printf ' %q' "$@"
        printf '\n'
    } >>"$out/commands.txt"
    set +e
    CUDA_VISIBLE_DEVICES="$devices" "$venv/bin/python" -m torch.distributed.run \
        --standalone --nproc_per_node="$processes" "$runner" \
        --model "$model" --require-fast-path --output "$output" "$@" \
        >"$log" 2>&1
    local rc=$?
    set -e
    echo "$rc" >"$exit_file"
    echo "END ${phase} ${label} rc=${rc} $(date --iso-8601=seconds)" | tee -a "$out/launcher.log"
    return "$rc"
}

validate_capacity() {
    "$venv/bin/python" - "$1" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
metadata = payload["metadata"]
assert metadata["fast_path_required"] and metadata["fast_path_available"]
assert metadata["compile"] is False and metadata["cuda_graph"] is False
assert metadata["denoise_steps"] == 4
rows = [row for row in payload["records"] if row["mode"] == "continuous_batching"]
assert len(rows) == 1
row = rows[0]
assert row["decode_steps"] == 30 and row["steps_per_frame"] == 1
assert row["avg_batch_size"] == 30
assert len(row["frame_times_ms"]) == 3
assert row["completed_requests"] == 3
assert row["all_completed_lengths_correct"]
assert row["all_completed_text_nonempty"]
assert row["action_finite"] and row["action_shape"] == [1, 16, 7]
print("capacity gate passed", sys.argv[1])
PY
}

capacity_gate() {
    local label=$1
    local devices=$2
    local processes=$3
    local size=${label%%_tp*}
    local model
    model=$(model_path "$size")
    local output="$out/capacity_${label}_n30_k1.json"

    run_benchmark "$label" "$devices" "$processes" "$model" capacity "$output" \
        --decode-steps 30 --steps-per-frame 1 --denoise-steps 4 \
        --warmup 1 --repeats 1 --baseline-measured-frames 2 \
        --continuous-measured-frames 3 --profile-repeats 0
    validate_capacity "$output"
}

formal_run() {
    local label=$1
    local devices=$2
    local processes=$3
    local size=${label%%_tp*}
    local model
    model=$(model_path "$size")
    local output="$out/formal_${label}.json"

    run_benchmark "$label" "$devices" "$processes" "$model" formal "$output" \
        --decode-steps 10,20,30 --steps-per-frame 1,5,10 --denoise-steps 4 \
        --warmup 2 --repeats 3 --baseline-measured-frames 10 \
        --continuous-measured-frames 50 --profile-repeats 1
    passed_labels+=("$(display_label "$label")")
    passed_outputs+=("$output")
}

run_all() {
    cd "$repo"
    : >"$out/launcher.log"
    : >"$out/commands.txt"
    date --iso-8601=seconds >"$out/started_at.txt"
    hostname >"$out/hostname.txt"
    nvidia-smi --query-gpu=index,name,uuid,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader >"$out/pre_run_gpu_state.csv"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader \
        >"$out/pre_run_gpu_processes.csv" || true

    local active_processes
    active_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
    if [[ -n "$active_processes" ]]; then
        echo "Refusing formal run: non-task GPU processes are active: $active_processes" >&2
        return 42
    fi

    sha256sum "$runner" "$helper" "$analyzer" >"$out/script_sha256.txt"
    cp "$runner" "$out/benchmark_script.py"
    cp "$helper" "$out/benchmark_helper.py"
    cp "$analyzer" "$out/analysis_script.py"
    git rev-parse HEAD >"$out/starvla_commit.txt"
    git status --short >"$out/starvla_status.txt"
    git -C /mnt/lixiangyu/repos/Xiaomi-Robotics-0 rev-parse HEAD >"$out/xiaomi_commit.txt"
    "$venv/bin/python" - <<'PY' >"$out/software_versions.txt"
import importlib.metadata
import platform
import torch
import transformers

print("python", platform.python_version())
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda", torch.version.cuda)
print("cudnn", torch.backends.cudnn.version())
for package in ("flash-linear-attention", "causal-conv1d"):
    print(package, importlib.metadata.version(package))
PY

    "$venv/bin/pip" freeze >"$out/pip_freeze.txt"
    cp "$work/remote_edit/run_qwen35_scaling_tp_formal.sh" "$out/launcher_script.sh"

    # Maximum formal occupancy is N=30,k=1 (continuous language batch 30).
    # Every capacity gate performs one full warmup and three measured frames,
    # including prefix, action, manager, and language decode stages.
    capacity_gate 0p8b_tp1 0 1
    capacity_gate 0p8b_tp2 0,1 2
    capacity_gate 2b_tp1 0 1
    capacity_gate 2b_tp2 0,1 2
    capacity_gate 4b_tp1 0 1
    capacity_gate 4b_tp2 0,1 2
    capacity_gate 9b_tp2 0,1 2

    # 9B TP1 is optional on a 24 GiB device. Preserve its OOM/error log and
    # continue with the capacity-qualified configurations if it cannot fit.
    set +e
    capacity_gate 9b_tp1 0 1
    local nine_b_tp1_rc=$?
    set -e
    echo "$nine_b_tp1_rc" >"$out/capacity_9b_tp1_status.txt"

    formal_run 0p8b_tp1 0 1
    formal_run 0p8b_tp2 0,1 2
    formal_run 2b_tp1 0 1
    formal_run 2b_tp2 0,1 2
    formal_run 4b_tp1 0 1
    formal_run 4b_tp2 0,1 2
    formal_run 9b_tp2 0,1 2
    if [[ "$nine_b_tp1_rc" -eq 0 ]]; then
        formal_run 9b_tp1 0 1
    fi

    local analyze_args=()
    local index
    for index in "${!passed_labels[@]}"; do
        analyze_args+=(--run "${passed_labels[$index]}=${passed_outputs[$index]}")
    done
    "$venv/bin/python" "$analyzer" "${analyze_args[@]}" --output-dir "$out"

    date --iso-8601=seconds >"$out/finished_at.txt"
    nvidia-smi --query-gpu=index,temperature.gpu,power.draw,memory.used,utilization.gpu \
        --format=csv,noheader >"$out/post_run_gpu_state.csv"
}

export -f run_all run_benchmark validate_capacity capacity_gate formal_run model_path display_label
export lock_file repo work runner helper analyzer venv out log_dir
export PYTHONPATH TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR TOKENIZERS_PARALLELISM PYTHONHASHSEED
flock -x "$lock_file" bash -c 'set -euo pipefail; declare -a passed_labels=(); declare -a passed_outputs=(); run_all'
