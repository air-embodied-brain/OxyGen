#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT=/mnt/lixiangyu/results/xiaomi_protocol_aligned_rerun_20260727
REPO=/mnt/lixiangyu/repos/Xiaomi-Robotics-0
LOCK=/tmp/oxygen_gd8_formal_timing.lock
mkdir -p "$RESULT_ROOT"

flock -x "$LOCK" bash -lc '
  set -euo pipefail
  cd /mnt/lixiangyu/repos/Xiaomi-Robotics-0
  export CUDA_VISIBLE_DEVICES=0
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export PYTHONHASHSEED=0
  result_root=/mnt/lixiangyu/results/xiaomi_protocol_aligned_rerun_20260727
  log="$result_root/formal_run.log"
  exec > >(tee -a "$log") 2>&1
  echo "formal_start=$(date -Iseconds)"
  echo "hostname=$(hostname)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_status=$(git status --short)"
  sha256sum scripts/oxygen_paper_sweep_formal_20260727.py \
    /mnt/lixiangyu/results/xiaomi_protocol_aligned_rerun_20260727/inputs/base.png \
    /mnt/lixiangyu/results/xiaomi_protocol_aligned_rerun_20260727/inputs/wrist.png \
    /mnt/lixiangyu/results/xiaomi_protocol_aligned_rerun_20260727/inputs/state.json
  nvidia-smi --query-gpu=index,name,driver_version,temperature.gpu,power.draw,memory.total,memory.used --format=csv
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
  .venv/bin/python - <<"PY"
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("flash_sdp", torch.backends.cuda.flash_sdp_enabled())
PY
  command=(.venv/bin/python scripts/oxygen_paper_sweep_formal_20260727.py
    --checkpoint /mnt/lixiangyu/checkpoints/Xiaomi-Robotics-0-LIBERO
    --common-prefix-mode libero_common
    --libero-task "Pick up the black bowl between the plate and the ramekin and place it on the plate."
    --base-image "$result_root/inputs/base.png"
    --wrist-image "$result_root/inputs/wrist.png"
    --state-json "$result_root/inputs/state.json"
    --denoise-steps 5
    --max-decoding-steps 5,10,15,20,30
    --steps-per-frame 1,5,10
    --instruction-repeats 1
    --warmup-runs 1
    --measured-repeats 3
    --single-measured-frames 10
    --single-extended-frames 30
    --single-cv-threshold 0.02
    --measured-frames 50
    --skip-invalid-continuous-combos
    --output "$result_root/xiaomi_libero_eager_s5_protocol_aligned.json")
  printf "command="; printf "%q " "${command[@]}"; echo
  "${command[@]}"
  echo "formal_end=$(date -Iseconds)"
' 
