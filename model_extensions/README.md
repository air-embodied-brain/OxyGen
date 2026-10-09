# OxyGen model extensions

These examples adapt OxyGen's prefix reuse and continuous language batching to
Xiaomi-Robotics-0 and models built with StarVLA. They show how to connect action
experts to a shared vision-language backbone, keep each request's decoding state,
and batch unfinished language requests across control frames.

| Example | What to explore | Implementation |
| --- | --- | --- |
| Xiaomi-Robotics-0 | One language expert and a flow-matching action expert | [Cache and action state](src/oxygen_models/common/cache.py), [example](src/oxygen_models/xiaomi/benchmark.py) |
| StarVLA / Qwen3VL-PI_v3 | Add GR00T and OFT action experts to the same Qwen3-VL backbone | [Expert adapters](src/oxygen_models/starvla/experts.py), [scheduler](src/oxygen_models/starvla/runtime.py) |
| Qwen3.5 + PI_v3 | Scale the backbone from 0.8B to 9B; batch both attention KV and recurrent state | [Hybrid cache](src/oxygen_models/qwen35/runtime.py), [action adapter](src/oxygen_models/qwen35/helpers.py) |

The examples use synthetic inputs and random parameters by default, so they need
model definitions and tokenizers but no trained weights. They exercise inference
and measure performance. Use trained, task-compatible models for meaningful robot
actions. The Qwen3.5 example always creates a matching PI_v3 action head with random
parameters, including when the language backbone loads pretrained weights.

For pi0.5 inference, see the [OxyGen guide](../README.md); for training and
serving the textual-memory adapter, see the [LIBERO example](../language_adaptation/libero/README.md).
Each extension uses a separate environment for its framework dependencies.

## Quick start

Use Linux x86-64, `uv`, and an NVIDIA GPU with a driver supporting CUDA 12.8.
A 24 GB GPU is recommended. Each example has its own Python 3.12 environment
because the model implementations require different Transformers versions.
Run the following commands from the repository root. The installer initializes
the StarVLA submodule when needed; its source remains unmodified. Python packages
use the Tsinghua PyPI mirror by default. Set `UV_DEFAULT_INDEX` to select another
index. CUDA extension wheels come from their official releases and are verified
against fixed SHA256 checksums.

Start with StarVLA:

```bash
bash model_extensions/setup.sh starvla
model_extensions/.envs/starvla/bin/python model_extensions/download_models.py starvla
python3 model_extensions/run.py starvla --gpu 0 --smoke \
  --output model_extensions/outputs/starvla-smoke.json
```

This runs language + PI_v3, then adds GR00T and OFT. Expert counts include the
language expert: 2, 3, and 4. All configurations use the same observation and
OFT-compatible action-query tokens, so the comparison isolates the added experts.

Remove `--smoke` for longer measurements. Options after `--` are forwarded to the
model's example, such as `--expert-counts 2,4` or `--language-tokens 10,30`.
Use a new output filename for each run. `--dry-run` prints the command without
loading a model.

## Xiaomi-Robotics-0

```bash
bash model_extensions/setup.sh xiaomi
model_extensions/.envs/xiaomi/bin/python model_extensions/download_models.py xiaomi
python3 model_extensions/run.py xiaomi --gpu 0 --check \
  --output model_extensions/outputs/xiaomi-check.json
python3 model_extensions/run.py xiaomi --gpu 0 --smoke \
  --output model_extensions/outputs/xiaomi-smoke.json
```

The runtime exposes separate prefill, action denoising, and incremental language
steps. The default example uses two synthetic camera views and Xiaomi's LIBERO
prompt format. Supply `--base-image`, `--wrist-image`, and `--state-json` after
`--` to use your own observation. The state JSON contains a `state` vector.

`--check` exercises staggered requests, cache stacking and splitting, request
completion, and isolation when a slot is reused. FlashAttention 2 is required;
the setup script installs a version matched to PyTorch 2.8.

## Backbone scaling with Qwen3.5

```bash
bash model_extensions/setup.sh qwen35
model_extensions/.envs/qwen35/bin/python model_extensions/download_models.py qwen35 --size 0.8B
python3 model_extensions/run.py qwen35 --size 0.8B --gpu 0 --smoke \
  --output model_extensions/outputs/qwen35-0.8b-smoke.json
```

Choose `--size 2B`, `4B`, or `9B` in both commands to try another backbone. Omit
`--size` from the download command to prepare all four sizes. Each backbone gets
a PI_v3 head configured for its hidden dimensions and layers.

Qwen3.5 combines full attention with linear attention. Its runtime batches the
full-attention KV cache together with convolution and recurrent state, preserving
each request's age and resetting completed slots before reuse. The example checks
these operations before timing. It uses the FLA and causal-conv1d kernels installed
by `setup.sh`.

The supplied launcher runs on one GPU. The lower-level example also supports
Hugging Face tensor parallelism with a pretrained backbone; config-only random
initialization does not apply a tensor-parallel plan.

## Understand and extend the runtime

Each control frame prefills the current observation, runs the selected action
experts, and advances active language requests by `k` tokens. Requests finish
after `N` tokens in these examples. With one arrival per frame and `N` divisible
by `k`, the steady batch has `N/k` requests.

The prefix computation is reused, while each expert or language request has its
own mutable state. Start with `PrefixState`, `ActionState`, and `LanguageState` in
[`common/cache.py`](src/oxygen_models/common/cache.py). StarVLA's
[`experts.py`](src/oxygen_models/starvla/experts.py) shows how different action heads
consume backbone outputs; [`runtime.py`](src/oxygen_models/starvla/runtime.py)
handles persistent language slots. Qwen3.5's
[`runtime.py`](src/oxygen_models/qwen35/runtime.py) extends slot management to a
hybrid cache.

When adapting another model, identify the prefix outputs each expert needs, its
incremental decoding state, and the operations needed to batch and reset that
state. Check independent execution against shared-prefix execution, then test
requests at different ages and slot replacement. These interfaces are model
specific; the examples keep those differences visible.

## Configuration and outputs

The three modes are `baseline` (independent execution), `shared_kv` (OxyGen without
batching), and `continuous_batching`. Qwen3.5 calls independent execution
`separate` in its JSON output. Each example writes frame timings, summaries, and
runtime checks as JSON. These outputs are useful for comparing settings on your
own device; short smoke runs check functionality rather than stable performance.

Run `python3 model_extensions/run.py MODEL --output /tmp/example.json -- --help`
for model-specific arguments. Use `--envs` and `--assets` to change the environment
and model directories. The downloader fetches only the files listed in
[`models.json`](models.json), at fixed HF revisions, and verifies their contents.
For an existing metadata directory, use `download_models.py GROUP --verify-only
--output /path/to/assets`.

The environment versions are pinned in [`environments/`](environments/):

| Example | PyTorch | Transformers | Attention |
| --- | --- | --- | --- |
| Xiaomi | 2.8.0 | 4.57.1 | FlashAttention 2.8.3 |
| StarVLA | 2.8.0 | 4.57.0 | PyTorch SDPA |
| Qwen3.5 | 2.8.0 | 5.2.0 | SDPA, FLA 0.5.1, causal-conv1d 1.6.2.post1 |

## Development checks

After installing the StarVLA environment, run the initialization tests with:

```bash
model_extensions/.envs/starvla/bin/python -m unittest discover -s model_extensions/tests -v
```

The tests check parameter initialization and restoration of the upstream loading
factory. The GPU examples exercise the cache and scheduler paths.

## Dependencies and license

OxyGen extensions use Apache-2.0. The StarVLA dependency and its licenses are
pinned as an official submodule under [`third_party/starvla`](third_party/starvla); see
[`THIRD_PARTY.md`](THIRD_PARTY.md) for upstream sources and local adaptations.
Model code and assets retain their upstream licenses. Xiaomi's model definition
is downloaded from its pinned HF revision and uses `trust_remote_code=True`.
