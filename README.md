# OxyGen project website

This orphan branch contains the static project website for [OxyGen](https://github.com/air-embodied-brain/OxyGen). The site is built with Astro and has no runtime backend. It is the promotional page; technical walkthroughs live in the [blog post](https://xxxxyu.github.io/blog/oxygen-mot-vla-inference/).

## Local development

```bash
npm install
npm run dev
```

Run `npm run build` before submitting changes. The GitHub Actions workflow builds the site with the `/OxyGen` base path and deploys `dist/` to GitHub Pages.

## Data provenance

Every measured number on the page lives in `src/data/` and is traced to its paper location and raw evidence in [DATA_SOURCES.md](DATA_SOURCES.md), which also explains how to replace results after a re-run.

## Demo videos

`public/videos/*.mp4` are rendered from logged LIBERO-10 rollouts (OxyGen with `k=1` and the blocking isolated baseline, same seed and initial state):

1. `scripts/demo/replay_hires.py` replays each logged action trajectory at 768×768 in the LIBERO client environment and checks every step against the logged goal predicates.
2. `scripts/demo/render_comparison.py` builds a 1× wall-clock timeline from the logged server inference latency and the 20 Hz control rate, then renders the side-by-side video (`--vertical` gives the stacked phone version).

The source runs are `visual_memory_v1_20260813/demo_campaign_v2/oxygen/k1/task{0,6,8}` and `blocking_baseline_eos_v2/task{0,6,8}` in the experiment workspace; the selected episodes and task times are listed in `src/data/site.ts`.
