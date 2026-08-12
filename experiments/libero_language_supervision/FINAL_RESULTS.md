# LIBERO language supervision: final results

## Coverage

The final `libero_all_v5` annotation set covers the complete training split of
the four standard LIBERO suites:

| Suite | Tasks | Demonstrations |
| --- | ---: | ---: |
| LIBERO-Spatial | 10 | 500 |
| LIBERO-Object | 10 | 500 |
| LIBERO-Goal | 10 | 500 |
| LIBERO-10 | 10 | 500 |
| **Total** | **40** | **2,000** |

The 2,000 JSONL trajectories contain 338,575 frame records, exactly matching
the raw HDF5 files. Each row retains the raw goal and auxiliary predicate
values and adds three independent language components: `completed`,
`remaining`, and `next`. `compose_language()` selects any ordered subset from
training configuration without regenerating the annotations.

## Validation

The final audit reports:

- 40/40 HDF5 files, 2,000/2,000 demonstrations, and 338,575/338,575 frames;
- no missing or unexpected demonstrations and no frame-count mismatch;
- 0 files with structural, temporal, or language-component issues;
- 2,000/2,000 demonstrations reach simulator success;
- 14 demonstrations end on a false raw success value after previously reaching
  success because of tail contact flicker. Stable textual completion remains
  latched and does not regress.

The raw annotations and audit are stored outside Git at:

```text
/home/lixiangyu/oxygen_ws/libero_exp/annotations/libero_all_v5
/home/lixiangyu/oxygen_ws/libero_exp/annotations/libero_all_v5/qa.json
```

## Corner cases

- Every demonstration loads its own stored MuJoCo XML before applying its
  states. This prevents repeated instances such as the two moka pots from being
  swapped. The affected Scene 8 task now reaches simulator success in 50/50
  demonstrations.
- Raw predicates are never smoothed or changed. A separate three-frame stable
  progress view suppresses text flicker and latches terminal completion.
- Pickup milestones use gripper-object contact rather than LIBERO's geometric
  height predicate. A three-frame stable grasp transition produces `Pick up`,
  followed by `Place` when the destination predicate becomes stable. Across
  2,100 placement goals, all 2,100 have a preceding grasp milestone.
- The 100 spatial movement goals remain a single `Move` milestone and contain
  no spurious pickup label.
- Task-local naming resolves repeated moka pots, left/right plates, drawer
  levels, cabinet and microwave regions, the caddy compartment, and spatial
  target regions.
- The cream-cheese-in-bowl task is rendered as “in” even though its BDDL goal
  uses the coarser `On` relation.

## Human review set

The review set contains one seeded random demonstration per task (40 videos).
The 20 Hz source is shown at 10 fps for 0.5x real-time playback. All videos are
below 5 MB; the largest is 0.572 MB and the complete set is 8.7 MB. The page
loads video sources only when their cards approach the viewport.

```text
/home/lixiangyu/oxygen_ws/libero_exp/review/libero_language_v5
```
