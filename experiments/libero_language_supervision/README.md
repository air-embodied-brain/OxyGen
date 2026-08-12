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
- auxiliary `grasped(object)` predicates for movable goal objects, evaluated
  from gripper-object contact in the recorded MuJoCo state;
- newly satisfied and newly unsatisfied predicate IDs;
- completed-goal count, goal progress, and task success;
- replay divergence when annotation is generated from actions.
- three language fields derived from a stable progress view:
  - `completed`: stable goal or prerequisite milestones completed by the
    current state;
  - `remaining`: demonstrated milestones that remain incomplete;
  - `next`: the next demonstrated predicate transition, phrased as a concise
    pi0.5-style subtask.
- predicate IDs and target frames used to produce each language field.

Goal predicates and auxiliary predicates are kept separate. Only goal
predicates determine success. Raw Boolean values and raw transitions are never
smoothed or overwritten. A separate three-frame stability rule suppresses
contact flicker for text, and terminal goal states are latched after the
demonstration first reaches stable success. This also handles goals such as
`Close(drawer)`, which may be true initially, become false while the drawer is
used, and become true again only when the task is complete.

The renderer uses generic rules for `on`, `in`, `grasped`, `open`, `close`,
`turnon`, and `turnoff`. A stable grasp transition produces `Pick up ...`, and
the later placement predicate produces `Place ...`; spatial pushing tasks keep
their single `Move ...` milestone. Task-local naming distinguishes repeated
objects and left/right targets. Small semantic overrides cover cases where the
BDDL relation is intentionally coarser than the instruction, such as the cream
cheese "in the bowl" task represented by `On`.

Each demonstration is interpreted with its own stored MuJoCo XML before its
recorded states are loaded. This is required for tasks with repeated instances:
reusing one task-level model can swap identifiers such as `moka_pot_1` and
`moka_pot_2`. Each task also runs in a separate subprocess so MuJoCo/EGL
resources are released before the next task starts.

All three text components are stored independently. Training can select any
ordered subset without replaying the simulator or regenerating labels. For
example, a training configuration can specify:

```yaml
language_targets:
  components: [completed, remaining, next]
  separator: "\n"
```

Changing `components` to `[next]` or `[completed, next]` changes only the
training target composition. `compose_language()` in `render_text_labels.py`
implements this operation.

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

Annotate all downloaded demonstrations in the four standard suites. The driver
writes one atomic JSONL file per demonstration and skips completed files when
resumed:

```bash
MUJOCO_GL=egl python annotate_all.py \
  --dataset-root /path/to/raw_libero \
  --output-root /path/to/annotations
```

Run the dataset-wide temporal audit:

```bash
python qa_annotations.py /path/to/annotations \
  --output /path/to/annotations/qa.json
```

If only the language rules change, rebuild the three derived text fields from
the saved raw predicates without replaying MuJoCo:

```bash
python recompile_annotations.py /path/to/annotations
```

Render one seeded random episode from every task and build the review site:

```bash
python build_review_site.py \
  --dataset-root /path/to/raw_libero \
  --annotation-root /path/to/annotations \
  --output-root /path/to/review_site
```

Videos use the stored observation frames, play at 10 fps (0.5x the simulator's
20 Hz rate), and are rejected if they exceed 5 MB. The page keeps video URLs
out of the DOM until each item approaches the viewport, so a 40-task review
does not download every video at startup.

Run the environment-independent tests with:

```bash
PYTHONPATH=. pytest -q tests
```

Generated datasets, checkpoints, annotations, and videos must remain outside
Git.
