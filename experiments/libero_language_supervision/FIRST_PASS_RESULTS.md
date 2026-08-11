# First-pass annotation result

Date: 2026-08-11

## Input

- Suite: LIBERO-Spatial
- Task: `pick up the black bowl between the plate and the ramekin and place it on the plate`
- Demonstration: `demo_0` (98 states and 98 actions)
- Raw file: `yifengzhu-hf/LIBERO-datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5`
- File SHA-256: `ff6f26121653c77280eb40a38773a74141c11a8509f3466058cb56dd2cc60ead`

The downloaded HDF5 contains 50 demonstrations. It is stored outside Git at:

```text
/home/lixiangyu/oxygen_ws/libero_exp/datasets/libero_spatial/
```

## Exact-state annotation

The source BDDL goal is:

```text
on(akita_black_bowl_1, plate_1)
```

The annotator also evaluates LIBERO's built-in `up(akita_black_bowl_1)` as an
auxiliary progress signal. Across the 98 recorded states:

- the bowl first becomes `up` at frame 63;
- the goal first becomes true at frame 79;
- the final state satisfies the goal;
- seven frames contain a predicate transition.

The contact-based `on` predicate briefly toggles while the bowl settles after
placement (frames 79-87). This is retained in the structured source rather than
silently smoothed. A text-training export may add an explicit temporal filter,
but the raw values remain available for auditing.

Generated artifacts are stored outside Git at:

```text
/home/lixiangyu/oxygen_ws/libero_exp/annotations/libero_spatial_between_plate_ramekin/
```

`demo_0.recorded_states.jsonl` is the authoritative structured annotation and
`demo_0.with_text.jsonl` is the deterministic natural-language rendering.

Example labels:

```text
frame 0:  The black bowl is not yet on the plate.
frame 63: The black bowl is lifted; the black bowl is not yet on the plate.
frame 79: The black bowl is on the plate; the task is complete.
frame 97: The black bowl is on the plate; the task is complete.
```

## Action replay check

Starting from the stored initial state and using the stored model XML with local
asset-path rewriting recovers:

- the same final success;
- the same first success frame (79);
- the goal predicate on 94/98 frames;
- the auxiliary `up` predicate on 95/98 frames.

The remaining differences occur around contact and object settling. Simulator
integration is sensitive enough that action replay should not replace exact
recorded-state annotation when full states are available. It is still a viable
fallback if each trajectory preserves its exact initial simulator state, model
configuration, controller settings, and compatible simulator version. Actions
plus an 8D robot state are not sufficient.

## Survey findings

1. LIBERO already provides the required symbolic layer. BDDL goals are parsed
   into `goal_state`; task success evaluates each entry with `_eval_predicate`.
   The built-in predicates are `in`, `on`, `up`, `open`, `close`, `turnon`, and
   `turnoff`. LIBERO does not export per-frame predicate labels itself.
2. Official raw HDF5 is the most reliable source because it includes actions,
   full MuJoCo states, model XML, BDDL metadata, and observations. The public
   Physical Intelligence LeRobot conversion and the current LIBERO-Plus LeRobot
   release omit full object/fixture simulator state.
3. SPRVLA is an open progress-aware LIBERO implementation based on spatial
   subgoals. Its evaluation code and checkpoints are public, but its repository
   currently marks training code and datasets as coming later; it does not
   provide a reusable simulator-predicate annotation pipeline.
4. Recent work such as IVLR constructs subgoal supervision by temporal
   segmentation followed by VLM captioning. This is complementary to the
   predicate-first approach: predicates provide auditable state transitions,
   while a captioner can enrich the text after segmentation.

The practical data path is therefore: replay official raw demonstrations once,
retain predicate provenance, optionally filter short contact flicker, and then
generate task-specific text variants as a separate stage.
