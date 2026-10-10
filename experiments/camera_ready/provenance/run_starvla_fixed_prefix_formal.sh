#!/usr/bin/env bash
set -euo pipefail

task_result_dir=/mnt/lixiangyu/results/starvla_4b_multi_expert_fixed_prefix_20260727
task_repo=/mnt/lixiangyu/repos/starVLA
task_lock=/tmp/oxygen_gd8_formal_timing.lock

mkdir -p "${task_result_dir}"
exec 9>"${task_lock}"
flock -x 9

date -Iseconds >"${task_result_dir}/lock_acquired_at.txt"
nvidia-smi \
  --query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader >"${task_result_dir}/gpu_before.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader >"${task_result_dir}/gpu_processes_before.csv" || true

cd "${task_repo}"
git rev-parse HEAD >"${task_result_dir}/starvla_commit.txt"
git status --short >"${task_result_dir}/starvla_status.txt"
cp scripts/oxygen_k_expert_formal_fixed_prefix.py "${task_result_dir}/benchmark_script.py"
cp scripts/oxygen_real_experts.py "${task_result_dir}/oxygen_real_experts.py"
cp scripts/oxygen_heterogeneous_sweep.py "${task_result_dir}/oxygen_heterogeneous_sweep.py"

set +e
CUDA_VISIBLE_DEVICES=0 .venv-system/bin/python \
  scripts/oxygen_k_expert_formal_fixed_prefix.py \
  --base-vlm /mnt/lixiangyu/checkpoints/Qwen3-VL-4B-Instruct \
  --pi-checkpoint /mnt/lixiangyu/checkpoints/Qwen3VL-PI_v3-Bridge-RT_1/checkpoints/steps_50000_pytorch_model.pt \
  --groot-checkpoint /mnt/lixiangyu/checkpoints/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt \
  --oft-checkpoint /mnt/lixiangyu/checkpoints/Qwen3VL-OFT-Bridge-RT-1/checkpoints/steps_5000_pytorch_model.pt \
  --expert-counts 2,3,4 \
  --language-tokens 5,10,15,20,30 \
  --denoise-steps 4 \
  --steps-per-frame 1,5,10 \
  --warmup 1 \
  --repeats 3 \
  --single-measured-frames 10 \
  --single-adaptive-frames 30 \
  --single-cv-threshold 0.02 \
  --single-trend-threshold 0.02 \
  --measured-frames 50 \
  --continuous-warmup-frames 5 \
  --continuous-runtime persistent \
  --output "${task_result_dir}/raw_results.json" \
  >"${task_result_dir}/benchmark.log" 2>&1
task_exit_code=$?
set -e

date -Iseconds >"${task_result_dir}/finished_at.txt"
nvidia-smi \
  --query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader >"${task_result_dir}/gpu_after.csv"
printf '%s\n' "${task_exit_code}" >"${task_result_dir}/exit_code.txt"
exit "${task_exit_code}"
