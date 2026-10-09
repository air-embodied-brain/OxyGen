# Third-party components

## StarVLA

Source: https://github.com/starVLA/starVLA

Base revision: `c00b8e18c1b5ba6228fd68a8f38acca8df4e938a`.
The official repository is pinned as a submodule in `third_party/starvla`,
with its original [MIT license](third_party/starvla/LICENSE) and nested licenses.
No upstream source files are modified.

OxyGen's config-only initialization adapter lives in
`src/oxygen_models/starvla/loading.py`. It supplies a local Qwen3-VL factory during
expert construction and restores the upstream factory afterwards. Expert execution,
shared-prefix reuse, and continuous batching live in the other OxyGen modules.

## Xiaomi-Robotics-0

Source: https://github.com/XiaomiRobotics/Xiaomi-Robotics-0

The model implementation is obtained from
https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-0-LIBERO at the revision in
`models.json`. Its [Apache-2.0 license](licenses/LICENSE_XIAOMI) is retained here.
The extension calls the original prefill and denoising primitives and adds
incremental language state and batching.

## Qwen and attention kernels

Qwen3-VL and Qwen3.5 configurations and tokenizers come from the official Qwen
Hugging Face repositories listed in `models.json`. Their model cards contain
the applicable model licenses.

Transformers, PyTorch, FlashAttention, Flash Linear Attention, and causal-conv1d
are installed as dependencies and retain their upstream licenses.
