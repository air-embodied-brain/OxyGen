# LIBERO language supervision

This pipeline derives language targets from the simulator state stored in the
official LIBERO demonstrations. Simulator predicates are extracted first and
saved as structured JSONL. A separate deterministic pass converts predicate
trajectories into text, so wording can change without replaying MuJoCo.

## Why raw HDF5 is required

LIBERO defines success through BDDL predicates. The official HDF5 files retain
the MuJoCo state, task metadata, and model XML needed to restore each recorded
state and evaluate those predicates. Image/action conversions such as LeRobot
do not contain enough simulator state to recover object relations exactly.

The pipeline reads recorded states by default. Action replay is available as a
diagnostic, but it is not the annotation source because small simulator drift
can move predicate transition times.

## Output

Each frame record contains:

- raw goal-predicate values and task success;
- auxiliary `grasped` and `picked_up` values for movable goal objects;
- raw predicate transitions and stable completion times;
- four independent language fields: `completed`, `remaining`, `next`, and
  `textual_memory`.

`textual_memory` is the observation-grounded target used by the suffix adapter.
It includes only goal predicates that changed from false to true, remained
stable for three frames, and are still true in the current observation. It does
not use future states and excludes grasp/pickup events. The other fields remain
available for later experiments and can be composed without rerunning the
simulator.

Raw predicate values are never smoothed or overwritten. Stability affects only
the derived language view. Each demonstration is loaded with its own stored
MuJoCo XML; this is necessary for repeated instances such as the two moka pots,
whose object IDs are not interchangeable across saved models.

## Environment

Run the simulator tools in a Python environment with LIBERO, robosuite, MuJoCo,
and `h5py`. Place LIBERO at `third_party/libero`, or pass its checkout explicitly
with `--libero-checkout`.

## Annotate all suites

```bash
MUJOCO_GL=egl python -m language_adaptation.libero.supervision.annotate_all \
  --dataset-root /path/to/raw_libero \
  --output-root /path/to/annotations \
  --libero-checkout /path/to/LIBERO
```

The driver covers `libero_spatial`, `libero_object`, `libero_goal`, and
`libero_10`. It writes one JSONL file per demonstration using an atomic rename
and skips complete outputs when resumed. Tasks run in separate processes so
MuJoCo and EGL resources are released between environments.

To inspect one demonstration or compare action replay with recorded-state
annotation:

```bash
MUJOCO_GL=egl python -m language_adaptation.libero.supervision.replay_and_annotate \
  /path/to/task_demo.hdf5 \
  --demo demo_0 \
  --mode recorded-states \
  --libero-checkout /path/to/LIBERO \
  --output /path/to/demo_0.jsonl
```

## Rebuild text without MuJoCo

`annotate_all` already writes the text fields. When the wording rules change,
rerun only the deterministic compiler on a raw predicate trajectory:

```bash
python -m language_adaptation.libero.supervision.render_text_labels \
  /path/to/annotations/libero_10/task_name/demo_0.jsonl \
  --output /path/to/recompiled/demo_0.jsonl
```

`compose_language()` can combine any ordered subset of `completed`,
`remaining`, `next`, and `textual_memory` at data-loading time.

## Audit

```bash
uv run python -m language_adaptation.libero.supervision.qa_annotations \
  /path/to/annotations \
  --dataset-root /path/to/raw_libero \
  --output /path/to/annotations/qa.json
```

The audit checks dataset coverage, frame alignment, monotonic stable progress,
language completeness, and consistency between predicate IDs and text state.
It exits nonzero on any issue.

The validated dataset used for the adapter covered 40 tasks, 2,000
demonstrations, and 338,575 frames with no audit issue. The generated JSONL and
raw HDF5 data are not tracked in this repository.

Environment-independent rule tests run with:

```bash
uv run pytest --import-mode=importlib -q language_adaptation/libero/supervision/tests
```
