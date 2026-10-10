# Editorial refinement — 2026-10-10

Baseline: `website-editorial` at `f2ed456` (camera-ready content aligned).
Working branch: `website-editorial-refinement`.

## Agreed direction

Preserve the warm paper background, homepage font family, fine rules, restrained
teal/amber data colors and 1080px editorial grid. The page promotes the work;
the blog carries extended technical explanation. Avoid card grids, dashboard
styling and fixed-height sections. Prioritize mainstream desktop light mode.
Mobile-specific layout and dark-theme refinement follow later.

Performance is the headline; cross-model transfer provides supporting evidence.
Success rate is validation, not a hero statistic. The robot photo documents the
setup; the on-board section centers measured latency, not a visual demo claim.
Keep arXiv and its BibTeX until proceedings are available.

## Implemented

- Hero: direct description of competing action/language workloads; headline
  hierarchy of π0.5 speedup, concurrent output rate and cross-model speedups.
- Demo: shorten adjacent explanation; retain all three logged comparisons.
- Method: bridge shared-prefix design to the measured latency illustration.
- Results: four named evidence blocks, with compact two-column introductions.
- Generality: separate models/experts table from the model-size visualization;
  paired historical tensor-parallel results live in an expandable note.
- On-board: small setup photo, wider latency figure and measurement facts.
- Application: two prominent adapter metrics, supporting details in a caption.
- Resources: emphasize Paper/Code through typography, retain visible citation.

No measurement data or demo video was changed. Desktop verification uses
1280, 1440 and 1920px viewports; scroll through sections before capturing to
load the robot image and finish chart entry animations. Early count-up frames
are not final experiment values (70 Hz is the current rounded headline).

## Deferred by scope

- Dedicated mobile diagrams and chart typography.
- Dark-theme and video theme refinements.
- Real-robot footage (existing setup photo is not a demo).
