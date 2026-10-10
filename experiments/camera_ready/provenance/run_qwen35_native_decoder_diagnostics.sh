#!/usr/bin/env bash
set -euo pipefail

lock_file=/mnt/lixiangyu/oxygen_ws/locks/qwen35_formal_gpu.lock
work=/mnt/lixiangyu/oxygen_ws
runner="$work/remote_edit/qwen35_native_decoder_diagnostic.py"
venv=/tmp/oxygen_qwen35_fastpath
out="$work/results/qwen35_scaling_tp_formal_20260728/diagnostics"
log_dir="$out/logs"

export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

run_one() {
    local label=$1
    local devices=$2
    local processes=$3
    local model=$4
    local output="$out/native_${label}.json"
    local log="$log_dir/native_${label}.log"

    echo "START ${label} $(date --iso-8601=seconds)" | tee -a "$out/launcher.log"
    printf 'CUDA_VISIBLE_DEVICES=%q %q -m torch.distributed.run --standalone --nproc_per_node=%q %q --model %q --output %q\n' \
        "$devices" "$venv/bin/python" "$processes" "$runner" "$model" "$output" \
        >>"$out/commands.txt"
    set +e
    CUDA_VISIBLE_DEVICES="$devices" "$venv/bin/python" -m torch.distributed.run \
        --standalone --nproc_per_node="$processes" "$runner" \
        --model "$model" --output "$output" >"$log" 2>&1
    local rc=$?
    set -e
    echo "$rc" >"$log_dir/native_${label}.exit"
    echo "END ${label} rc=${rc} $(date --iso-8601=seconds)" | tee -a "$out/launcher.log"
    return "$rc"
}

run_all() {
    mkdir -p "$out" "$log_dir"
    : >"$out/launcher.log"
    : >"$out/commands.txt"
    date --iso-8601=seconds >"$out/started_at.txt"
    hostname >"$out/hostname.txt"

    local active_processes
    active_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
    if [[ -n "$active_processes" ]]; then
        echo "Refusing diagnostics: GPU processes are active: $active_processes" >&2
        return 42
    fi

    cp "$runner" "$out/diagnostic_script.py"
    sha256sum "$runner" >"$out/script_sha256.txt"
    nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader \
        >"$out/gpu_inventory.csv"

    run_one 4b_tp1 0 1 /mnt/lixiangyu/checkpoints/Qwen3.5-4B
    run_one 9b_tp1 0 1 /mnt/lixiangyu/checkpoints/Qwen3.5-9B
    run_one 4b_tp2 0,1 2 /mnt/lixiangyu/checkpoints/Qwen3.5-4B
    run_one 9b_tp2 0,1 2 /mnt/lixiangyu/checkpoints/Qwen3.5-9B

    date --iso-8601=seconds >"$out/finished_at.txt"
}

export -f run_all run_one
export lock_file work runner venv out log_dir
export TOKENIZERS_PARALLELISM PYTHONHASHSEED
flock -x "$lock_file" bash -c 'set -euo pipefail; run_all'
