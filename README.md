# OxyGen project website

This orphan branch contains the static project website for [OxyGen](https://github.com/air-embodied-brain/OxyGen). The site is built with Astro and has no runtime backend. It is the promotional page; technical walkthroughs live in the [blog post](https://xxxxyu.github.io/blog/oxygen-mot-vla-inference/).

## Local development

```bash
npm install
npm run dev
```

Use Node.js 22+ and [uv](https://docs.astral.sh/uv/). Run `npm run build && npm test` before submitting changes. The GitHub Actions workflow runs both checks, builds the site with the `/OxyGen` base path, and deploys `dist/` to GitHub Pages.

## Generated fonts and loading order

`npm run build` and `npm run dev` automatically prepare the fonts. The preparation
step renders Astro to `.font-render/`, collects the actual text in regions marked
`data-font-priority="critical"` (masthead, performance highlights, and Demo), and
includes strings from the Demo scripts and recorded replay data. The Python
generator uses locked FontTools/WOFF2 dependencies through uv; it never relies on
a manually maintained character list.

WOFF2 files and their manifest live in `public/generated-fonts/`. This directory,
the intermediate render, and the Python environment are ignored by Git. The
generator checks character sets, source-font hashes, its own code, dependency
version, and output hashes before reusing cached files. The final build verifies
coverage against the final rendered HTML. Source fonts retain their original
license metadata; generated font names contain content hashes.

Critical glyphs load first. Remaining glyphs use disjoint Unicode ranges, so
loading them cannot replace an already visible critical glyph. Before each tier
is ready, text is masked with theme-aware rounded placeholders; fallback-font
glyphs are never displayed during normal loading. The original fonts are revealed
without an opacity transition. Restart `npm run dev` after edits that introduce
new characters; `npm run build` always recomputes coverage from the current page.

Video requests start only after both font tiers and still images (including the
offscreen robot photo and all three task posters) finish loading or the images
report an error. The selected task downloads first, followed by the other tasks
one at a time at low fetch priority. Switching tasks interrupts an unfinished
background transfer. Completed files are reused as in-memory Blob URLs, avoiding
the initial duplicate media request and making seeking independent of additional
network requests. This first version downloads a selected file completely before
playback starts; it favors page loading order and smooth seeking over progressive
video startup.

## Favicon source

The site favicon uses outline paths derived from the bundled Libre Baskerville Bold 700 font. It has no runtime font dependency; the font is distributed under the SIL Open Font License 1.1. The favicon SVG includes the same attribution comment.

## Social preview and metadata

`src/data/site.ts` owns the shared project title, description, conference, and
tagline. `npm run social` renders `public/social-card.png` at 1200×630 using the
site's palette, bundled font outlines, and current headline measurements from
`src/data/paper-figures.ts`. It also derives a 32px PNG favicon and a 180px Apple
touch icon from the existing SVG mark. Font outlines keep generation independent
of browser installations and system fonts. Source fonts retain their SIL OFL
license notices. `npm run build` and `npm run dev` regenerate these assets first.

Open Graph and X/Twitter tags share the title, description, image, and alternative
text. The image includes its MIME type and pixel dimensions, following the
[Open Graph structured-image properties](https://ogp.me/#structured). Sharing
image and icon URLs carry content hashes so a changed asset gets a new URL;
platforms may still cache previews of previously shared page URLs.

The five-second acceptance delay was a temporary local proxy, not application
code. Normal builds have no artificial font or video delay.

## Data provenance

Every measured number on the page lives in `src/data/` and is traced to its paper location and raw evidence in [DATA_SOURCES.md](DATA_SOURCES.md), which also explains how to replace results after a re-run.

## Demo videos

`public/videos/*.mp4` are rendered from logged LIBERO-10 rollouts (OxyGen with `k=1` and the blocking isolated baseline, same seed and initial state):

1. `scripts/demo/replay_hires.py` replays each logged action trajectory at 768×768 in the LIBERO client environment and checks every step against the logged goal predicates.
2. `scripts/demo/render_comparison.py` builds a 1× wall-clock timeline from the logged server inference latency and the 20 Hz control rate, then renders the side-by-side video (`--vertical` gives the stacked phone version).

The source runs are `visual_memory_v1_20260813/demo_campaign_v2/oxygen/k1/task{0,6,8}` and `blocking_baseline_eos_v2/task{0,6,8}` in the experiment workspace; the selected episodes and task times are listed in `src/data/site.ts`.

### Interactive replay

The web player uses a clean crop of the same synchronized films and a JSON
export of their original timeline. Regenerate from the website root:

```sh
python3 scripts/demo/export_timelines.py --runs-root ../libero_exp/language_adapter_runs/visual_memory_v1_20260813 --verify-videos
python3 scripts/demo/crop_interactive.py
```

The latter requires `ffmpeg` and writes `*-interactive.mp4` and matching posters.
It removes the baked-in labels, retaining both simulator views and their gutter;
frequent keyframes support scrubbing. The player uses a shared wall clock for
both runs, stops at task completion, and supports dragging and 50 ms arrow-key
steps. The full annotated source films remain available as the data-load fallback.
Subphase reconstruction and source hashes are documented in `DATA_SOURCES.md`.
