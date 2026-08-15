# Visual-memory suffix LoRA

This run trains the language suffix to describe task progress visible before
the next action, rather than predicting the next subtask. The shared prefix is
unchanged: current observation, robot state, and task instruction. Generated
history is not fed back to either expert.

## Labels

The `visual_memory` target contains only goal predicates that made a stable
false-to-true transition and remain stably true at the current observation.
It excludes grasp and pickup events, uses no future trajectory information,
and does not latch reversible states. Recompilation covered all 338,575 frames
in 2,000 demonstrations across the four official suites with no QA issue.

## Training and held-out evaluation

Rank-16 suffix LoRA was trained through the deployed incremental path for 1,500
steps. A sweep over learning rates `1e-4`, `3e-5`, and `1e-5` selected `1e-4`.
The selected checkpoint reached 98.15% token accuracy on all 5,082 held-out
sampled frames. On a 400-sample task/state-balanced greedy set, exact match was
87.25% and word F1 was 90.01%.

With fixed diffusion noise, the shared root KV and action tensors were exactly
unchanged over 20 held-out observations. For the median 11-token suffix, LoRA
added 8.16 ms, compared with a 47.66 ms prefix forward removed by sharing.

## Rollout review

Three continuous-batching rollouts all completed successfully. Every action
replan created a new memory request at 4 Hz; unfinished requests resumed in
subsequent batches, reaching batch size 5. The two multi-stage LIBERO-10 tasks
showed the intended progress transitions. The single-stage drawer task was an
important failure case: the action succeeded, but language continued to report
no visible progress after the drawer opened.

The terminal observation is frozen for six additional language replans in the
review only, allowing the final request to finish. These frames do not execute
actions or change rollout success.

Large annotations, adapters, predictions, and videos are release assets and
remain outside Git. The selected adapter hash and packaging requirements are
listed in [`../../../RESULTS.md`](../../../RESULTS.md).
