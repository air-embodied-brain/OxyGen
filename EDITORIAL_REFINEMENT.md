# Desktop editorial refinement — 2026-10-10

Baseline: `website-editorial` at `f2ed456`, with camera-ready data aligned.
Working branch: `website-editorial-refinement`.

## Direction

Warm paper, homepage typography, fine rules, 1080px editorial grid. Promotional
research page, with technical depth in the blog. Performance and model transfer
lead; success is validation. The robot photo is a setup reference, not a demo.
Mobile-specific design and dark-theme refinement remain deferred.

## Current section decisions

The author rejected the first two refinement passes as visually worse. This
pass uses actual section screenshots and independent visual review, removing
the extra structures that weakened the original composition.

- Hero: retain the original masthead; four aligned performance columns show
  π0.5 speedup, concurrent action rate and Xiaomi/StarVLA transfer. One common
  protocol line. Integer frequency consistently rounds 70.9 to 71 Hz.
- Demo: video and task selection stay together in the original two-column
  arrangement; no detached full-width selectors or duplicate giant readout.
- Method: two concise panel titles, aligned captions, no oversized transition
  paragraph; compact measured-benefit notes alongside the latency diagram.
- Results: each figure has one heading. Matched-workload points and direct
  ablation labels guide reading. Models, experts and Qwen sizes share one
  table, with three short interpretations and scoped protocol notes.
- Robot: setup photograph and caption align; hardware settings occupy one
  short line. Measured inference and illustrative overlap remain distinct,
  with a compact timeline starting the execution window at dispatch.
- Application: concise introduction, aligned memory log and two metrics;
  zeroed-adapter validation stays secondary.
- Resources: equal heading baselines, restrained link typography, visible
  BibTeX and copy button. No explanatory paragraph or empty collapsed column.

Independent agent reviewed each section visually; a separate agent implemented
and screenshot-reviewed the generality table. Primary agent also inspected the
section images and integrated requested alignment/spacing corrections.

Desktop validation: 1280/1440/1920px, completed count-up and chart entry
animations, lazy image loaded. Check video/configuration switching, citation
copy, build and whitespace. These checks supplement visual inspection.
