#!/usr/bin/env bash
# Usage: bash model_extensions/setup.sh xiaomi|starvla|qwen35|all [environment-directory]
set -euo pipefail
bundle=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
group=${1:?Choose xiaomi, starvla, qwen35, or all}
case "$group" in
    xiaomi|starvla|qwen35) groups=("$group") ;;
    all) groups=(xiaomi starvla qwen35) ;;
    *) printf 'Unknown model group: %s\n' "$group" >&2; exit 2 ;;
esac
envs=${2:-"$bundle/.envs"}
mkdir -p "$envs"
envs=$(cd -- "$envs" && pwd)
command -v uv >/dev/null
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}"
uv python install 3.12.13
for name in "${groups[@]}"; do
    if [[ ! -x "$envs/$name/bin/python" ]]; then uv venv --python 3.12.13 "$envs/$name"; fi
    uv pip install --python "$envs/$name/bin/python" -r "$bundle/environments/$name.lock.txt"
    uv pip install --python "$envs/$name/bin/python" --no-deps -e "$bundle"
    if [[ "$name" != xiaomi ]]; then
        git -C "$bundle/.." submodule update --init -- model_extensions/third_party/starvla
        uv pip install --python "$envs/$name/bin/python" --no-deps -e "$bundle/third_party/starvla"
    fi
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
for name in "${groups[@]}"; do
    case "$name" in
      xiaomi) fetch_wheel flash-attention v2.8.3 flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl xiaomi ;;
      qwen35) fetch_wheel causal-conv1d v1.6.2.post1 causal_conv1d-1.6.2.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl qwen35 ;;
    esac
    uv pip freeze --python "$envs/$name/bin/python" > "$envs/$name.freeze.txt"
done
printf 'Environments ready: %s\n' "$envs"
