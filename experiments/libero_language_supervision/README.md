# LIBERO language supervision

This experiment builds language supervision without changing the original LIBERO
rollouts. It first evaluates simulator predicates at every recorded state and
writes a structured trajectory. Natural-language labels are a separate
post-processing step, so their style and granularity can change without running
the simulator again.

## Decisions

- Start with a suffix-only language adapter. The action path and its inputs stay
  frozen; language tokens are appended after the shared context and are hidden
  from the action suffix.
- A usable adapter must cost less at inference time than the duplicate prefix
  forward that shared KV removes. Measure end-to-end frame latency, language
  throughput, and adapter-only latency before scaling training.
- Use standard LIBERO demonstrations for the first data pass. Preserve BDDL
  predicate names and arguments as the source of truth; do not bake generated
  prose into the replay pipeline.
- Keep action quality as a hard non-regression target. First check frozen-path
  tensor equivalence, then evaluate the existing 10-seed LIBERO protocol.

## Why raw LIBERO HDF5 is required

LIBERO includes a BDDL predicate system. Each environment exposes its parsed
`goal_state`, evaluates each condition with `_eval_predicate`, and defines task
success as their conjunction. Official HDF5 demonstrations contain full MuJoCo
states, actions, the environment XML, and BDDL metadata. This is sufficient for
exact per-state annotation and for validating action replay.

The commonly used LeRobot conversion contains images, an 8D robot state, and
actions, but not the full simulator state or environment XML. It cannot recover
object and fixture state by itself. Action-only replay is possible only when the
exact initial simulator state and matching environment assets are available.

The current official LIBERO downloader supports the raw files hosted at
`yifengzhu-hf/LIBERO-datasets`. The older Box links in some LIBERO checkouts are
no longer reliable.

## Annotation schema

Every JSONL row represents one simulator state and contains:

- the original BDDL goal predicates and their Boolean values;
- auxiliary built-in `up(object)` predicates for movable goal objects;
- newly satisfied and newly unsatisfied predicate IDs;
- completed-goal count, goal progress, and task success;
- replay divergence when annotation is generated from actions.

Goal predicates and auxiliary predicates are kept separate. Only goal
predicates determine success.

## Usage

The simulator tools run in the Python environment that contains LIBERO,
robosuite, and `h5py`. If LIBERO is not vendored under this repository, pass its
checkout explicitly or use the automatic workspace fallback.

Inspect a raw file without starting MuJoCo:

```bash
python inspect_dataset.py /path/to/task_demo.hdf5
```

Annotate exact recorded states:

```bash
MUJOCO_GL=egl python replay_and_annotate.py \
  /path/to/task_demo.hdf5 \
  --demo demo_0 \
  --mode recorded-states \
  --output annotations/demo_0.jsonl
```

Validate that the same labels can be recovered from actions:

```bash
MUJOCO_GL=egl python replay_and_annotate.py \
  /path/to/task_demo.hdf5 \
  --demo demo_0 \
  --mode action-replay \
  --use-stored-model-xml \
  --output annotations/demo_0_action_replay.jsonl
```

Render a first-pass natural-language view without running MuJoCo again:

```bash
python render_text_labels.py annotations/demo_0.jsonl \
  --output annotations/demo_0.with_text.jsonl
```

The renderer is deterministic and rule-based. It is a reproducible baseline and
can later be replaced by paraphrase generation while retaining the predicate
record as provenance.

Run the environment-independent tests with:

```bash
PYTHONPATH=. pytest -q tests
```

Generated datasets, checkpoints, annotations, and videos must remain outside
Git.
