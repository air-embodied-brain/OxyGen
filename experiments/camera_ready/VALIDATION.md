# Validation performed on 2026-10-05

## Passed locally

- Installed all four isolated environments from the bootstrap recipe on Ubuntu x86-64:
  pi (Python 3.11.15), Xiaomi/StarVLA/Qwen3.5 (Python 3.12.13).
- Verified pinned package versions, CUDA availability, JAX GPU visibility, FlashAttention import,
  StarVLA's complete framework registry, and Qwen3.5 native fused fast-path availability.
- Parsed **all 24 exact commands**, including optional TP=2, inside their assigned environments
  using the real archived argument parsers, without loading model weights.
- Nine CPU regression tests cover complete task inventory, portable arguments, physical GPU
  selection, warmup exclusion, frozen-runtime import paths, missing results, missing repeats,
  preservation of failed Xiaomi attempts, and detection of changed checkpoint contents.
- Revalidated all 23 available historical supplement task artifacts against the new validators.
  For this archive-only check, used the historical short-N and manager point sets (the new
  suite deliberately adds N=10/15/20/30 and manager k=10). The new paired pi reference task
  uses the frozen driver; its new full grid has not been measured locally.
- Recomputed cross-model, scaling, stage-profile, memory, and long-run table numbers from raw
  archived records. Examples: Xiaomi N=30 speedups 5.50x/3.14x, StarVLA E=2 N=30
  6.18x/3.41x, manager k=5/1 fractions 9.64%/35.65%, JAX fluctuating first/last
  328.8/328.6 ms and PyTorch 389.9/388.0 ms.
- Generated all three PDF/PNG figures from the rebuilt aggregates and visually inspected them.
  Fonts, panel arrangement, legends and colors follow the reviewed figures; ranges/annotations
  respond to data. Existing paper figure files were not overwritten.
- Compiled the seven replacement paper tables at the NeurIPS 5.5-inch text width; no overfull boxes.
- Checked Python syntax, shell syntax and that every source-lock file is included in Git,
  including inputs and JSON/TXT manifests otherwise ignored by the parent repository.

## Still requires the destination host

No formal performance rerun or full checkpoint-backed model load has been performed in this
preparation pass. All eight GPUs currently have an unrelated compute job, and the historical
model directories are not present at their archived paths on this machine.

The destination run must pass the code/asset/environment preflight, each runner's model and
correctness checks, complete-grid validation, and final figure inspection. Timing equality with
old hardware is not a validation requirement. Checkpoint identity requires hashing the original
source model directories and comparing the transferred copies; model names alone are insufficient.

The environment locks describe the clean rebuilt environments. Most historical supplementary
runs inherited a broad system environment with no complete standalone lock (Qwen's full old
freeze is retained as evidence). Core measured libraries are pinned to historical versions;
Python patch and some supporting dependency versions may differ. This is documented rather
than claiming a bit-identical historical OS environment.

## Destination preparation on dodo, 2026-10-07

The original preparation's random-init change had not been exercised through model
construction. Destination audit found and fixed these issues in the local copy:

- Pi runners accepted but ignored `--random-init`. They now explicitly select the
  frozen implementation's missing-local-path random-init branch and reject an
  existing sentinel path.
- StarVLA config-only loading still required a weight file and loaded the VLM via
  `from_pretrained`. It now accepts a metadata directory and constructs the VLM
  from config. CPU meta-device construction of all three frameworks passes while
  model `from_pretrained` is patched to fail if called.
- Native Qwen diagnostics still tried to open safetensors for parameter counting
  in random-init mode. Checkpoint counts are now null in this mode; actual model
  parameter counts are still recorded.
- Empty argument filtering removed the fluctuating run's legitimate empty arrival
  pattern. Exact parser checking reproduced the error; the entry point now only
  omits the unused random-init flag.
- Random-init asset contents are hashed for resume identity. Random parameters
  use seed 0. JAX cache location can be directed to HDD via
  `JAX_COMPILATION_CACHE_DIR`.
- Random-init TP=2 previously constructed full replicas without applying a TP
  plan. It is rejected explicitly; the requested 18-task single-GPU protocol is
  unaffected.

All four pinned environments installed; `uv pip check` passes in each. All 18
single-GPU command parsers pass. Eleven CPU regression tests pass. StarVLA and
Qwen model constructors were checked on the meta device without allocating GPU
weights. Qwen's four input token sequences, shapes, layer types, and action-head
parameter counts match the historical records; 4B/9B model parameter counts match
historical native-decoder records. Xiaomi's metadata and processor construct in
a separate CPU architecture-only diagnostic using eager attention; this is not a
FlashAttention benchmark validation or a change to the formal attention path.

HF metadata was recovered at immutable revisions with file hashes under the
host's assets directory. Historical metadata directories were unavailable at the
recorded source paths, so byte-identical historical asset provenance is not claimed.

## Exclusive destination validation, 2026-10-07 evening

The robot run recorded `homed_and_disabled` and operator-confirmed unloading.
No controllers remained. After the user's conditional authorization took effect,
two idle inference services were stopped with their exact restoration information
saved privately on HDD. Robot controllers will not be restarted by this suite.

Small GPU kernel checks passed in all four environments. Initial full-model smoke
checks passed for the PyTorch manager but failed for JAX (process faults), StarVLA
(interpreter failure), and Xiaomi (strict independent/batched token mismatch).
These are diagnostic runs, not formal performance results.

Removing inherited dynamic-loader paths allowed four JAX small-N checks to pass,
including unrestricted CPU affinity and the previously implicated CPU core. This
does not establish a hardware fault or definitively prove the cause. The launcher
now constructs child environments from an explicit allowlist and disables Python
user site packages. ROS/conda paths, LD_LIBRARY_PATH/LD_PRELOAD, distributed launch
state, and unrelated CUDA settings are not inherited. Twelve CPU regression tests
pass, including a deliberately polluted parent-environment check.

A new sequential full-model validation queue uses this isolated environment.
Xiaomi strict token equivalence remains a blocker under investigation; acceptance
criteria have not been relaxed. No formal destination result is available yet.
All caches, assets, logs and outputs stay on HDD. No remote results or paper files
have been updated.

### Subsequent stability blocker

The isolated queue still failed: JAX manager aborted with `free(): invalid pointer`,
PyTorch manager segfaulted during standard-library metadata parsing, and Xiaomi
also segfaulted. StarVLA and Qwen 0.8B smoke checks exited successfully, but no
result is promoted to formal evidence on this unstable host.

A standard-library-only test under `env`-equivalent minimal variables and Python
`-I`, pinned with taskset, reproduced incorrect JSON roundtrips on logical CPU 8
using BOTH the Ubuntu system Python and the separate experiment Python. The
system interpreter decoded one input value 23 as 0; later repeats produced
MemoryError. CPU 0, 9 and 16 passed the same short checks. CPU 8 and 9 are sibling
threads. This points beyond the ML environments, without establishing whether
the ultimate cause is CPU, firmware, memory, or another platform component.

The model dispatcher has been stopped. User guidance was requested before
changing the CPU conditions for further model diagnostics or arranging host
remediation. No BIOS, global CPU setting, or robot controller was changed.

### CPU-affinity diagnostics and precision correction

The user authorized diagnostic-only runs excluding logical CPUs 8/9. Eleven model
checks completed without interpreter/process faults: nine passed; Xiaomi strict
token equivalence and Qwen3.5 4B mixed-age cache audit failed. The corresponding
Qwen 0.8B/2B/9B checks and 4B/9B native seven-point grids passed. Xiaomi's
teacher-forced batch-size control found exactly matching static/dynamic logits at
batch 1 and small BF16 differences at batch 2/6. Qwen4B's mixed-age versus same-age
batch controls match exactly; its singleton-versus-batch controls do not. These
findings do not waive the strict validators.

Supplemental short-run validation uncovered JAX fluctuating OOM at frame 25,
recorded in summary.json despite exit status 0. A dtype audit found all
3,353,433,872 random JAX parameter elements were FP32, whereas checkpoint loading
restores BF16. The random branch now initializes/casts on CPU, then transfers BF16
parameters to the selected GPU. This avoids simultaneous FP32/BF16 GPU copies.
PyTorch random init now also applies the checkpoint branch's selected-parameter
precision restoration (including its FP32 vision/norm exceptions). Inference
algorithms and formal workload grids are unchanged. Prior pi diagnostics are
superseded; fresh checks use separate v4 result directories.

All results remain diagnostic-only because CPU affinity differs from the intended
formal host conditions. Formal runs and historical performance conclusions are
still pending host stability and unresolved strict correctness checks.

### Completed diagnostic-only pass

All 18 selected task entry points have now been exercised. Content validation
passes for 16; Xiaomi token equivalence and Qwen4B cache equivalence remain failed.
After the pi precision fix, all nine pi entry points pass shortened checks. Both
backends complete steady (30 warmup + 30 measured) and fluctuating (100 warmup +
30 measured) without OOM; measured batch sizes reach 32 and 40 respectively.
JAX steady live memory is approximately 7.17–7.19 GB; fluctuating is 7.13–7.42 GB.
The native Qwen4B/9B full seven-point diagnostic grids pass their validators.
These are NOT the full performance grids or the required 3,000 measured frames.
No performance discrepancy is declared resolved from these short diagnostics.

The exclusive diagnostic window has ended. The two original inference services
were restored and their ports checked; no robot controllers were started. The
formal launch helper remains blocked on host stability and the two strict
correctness failures. Pretrained-weight correctness controls await user approval.

### Performance-only validation approved and rechecked, 2026-10-08

The user approved implementation checks without cross-batch bit/token equality,
and instructed us not to download trained weights. Qwen's singleton-versus-batch
argmax condition is now diagnostic only; matched-batch cache, rollout, slot
recycling and input/progress checks remain. Nonfinite cache tensors now fail
explicitly instead of being hidden by a scalar max reduction.

Xiaomi adds fixed, distinct request histories at ages 0/1/3/5/8, 30 steps,
matched-batch logits/cache comparisons, replacement isolation, stack/split checks
and a deliberately swapped-row negative control. CPU regression checks pass (13).
The new Xiaomi GPU audit passes all 150 rows (actual max logits/cache errors 0),
including the negative control; the five greedy requests still produce 30 tokens
each. Qwen4B also passes the revised audit and smoke workload. All 18 entry points
now pass diagnostic content validation, using the precision-aligned v4 pi results
and v5 checks where applicable. No trained model weights were downloaded.

The user additionally authorized residual robot service cleanup. Both original
inference services are stopped, ports 5583/18093 closed, no robot controllers
found, and automatic restoration is no longer planned. Formal CPU conditions
still require a decision: fix platform stability or explicitly accept exclusion
of logical CPUs 8/9 with disclosure in historical comparisons.
