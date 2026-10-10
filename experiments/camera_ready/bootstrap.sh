#!/usr/bin/env bash
# Usage: bash experiments/camera_ready/bootstrap.sh /absolute/path/to/envs
set -euo pipefail
bundle=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
envs=${1:?Supply an absolute environment directory}
[[ "$envs" = /* ]] || { echo 'Use an absolute environment directory' >&2; exit 2; }
command -v uv >/dev/null
uv python install 3.11.15 3.12.13
# Original pi package + lock, independent of the main branch runtime.
UV_PROJECT_ENVIRONMENT="$envs/pi" uv sync --project "$bundle/vendor/oxygen" --frozen --python 3.11.15
uv pip install --python "$envs/pi/bin/python" 'matplotlib==3.10.3'
for name in xiaomi starvla qwen35; do
    if [[ ! -x "$envs/$name/bin/python" ]]; then uv venv --python 3.12.13 "$envs/$name"; fi
    uv pip install --python "$envs/$name/bin/python" 'torch==2.8.0' 'torchvision==0.23.0' --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python "$envs/$name/bin/python" -r "$bundle/environments/$name.lock.txt"
done
mkdir -p "$envs/wheels"
fetch_wheel() {
    local repo=$1 version=$2 filename=$3 target=$4
    if [[ ! -f "$envs/wheels/$filename" ]]; then
        curl --fail --location --retry 3 --connect-timeout 20 --max-time 1800 \
          "https://github.com/Dao-AILab/$repo/releases/download/$version/$filename" \
          --output "$envs/wheels/$filename.partial"
        mv "$envs/wheels/$filename.partial" "$envs/wheels/$filename"
    fi
    python3 - "$bundle/environments/wheels.sha256.json" "$envs/wheels/$filename" <<'PYHASH'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[2])
if hashlib.sha256(p.read_bytes()).hexdigest()!=json.load(open(sys.argv[1]))[p.name]:
    raise SystemExit('Wheel checksum mismatch: '+str(p))
PYHASH
    uv pip install --python "$envs/$target/bin/python" "$envs/wheels/$filename"
}
fetch_wheel flash-attention v2.8.3 flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl xiaomi
fetch_wheel causal-conv1d v1.6.2.post1 causal_conv1d-1.6.2.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl qwen35
for name in pi xiaomi starvla qwen35; do
    uv pip freeze --python "$envs/$name/bin/python" > "$envs/$name.freeze.txt"
done
printf 'Environments ready. Set config python paths under %s. Install Liberation Serif for plots if absent.\n' "$envs"
