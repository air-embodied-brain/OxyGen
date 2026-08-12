"""Build task-specific OxyGen versus isolated-baseline comparison pages."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import statistics


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", action="append", required=True, help="page slug=task title")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    task_pages = []
    for task_spec in args.task:
        slug, title = task_spec.split("=", 1)
        rows = []
        for k in range(1, 6):
            cells = []
            for execution in ("oxygen", "isolated"):
                run_root = args.input_root / execution / f"k{k}" / slug
                manifest = _load(run_root / "manifest.json")[0]
                rollout = json.loads((run_root / "rollouts.jsonl").read_text().splitlines()[0])
                events = [
                    json.loads(line)
                    for line in (run_root / "inference_events.jsonl").read_text().splitlines()
                    if line.strip()
                ]
                median_call_ms = statistics.median(event["client_round_trip_ms"] for event in events)
                video = (run_root / manifest["video"]).resolve()
                poster = (run_root / manifest["poster"]).resolve()
                asset_root = args.output_root / "assets" / slug / execution / f"k{k}"
                asset_root.mkdir(parents=True, exist_ok=True)
                video_link = asset_root / video.name
                poster_link = asset_root / poster.name
                for link, target in ((video_link, video), (poster_link, poster)):
                    if link.exists() or link.is_symlink():
                        link.unlink()
                    link.symlink_to(target)
                video_url = os.path.relpath(video_link, args.output_root)
                poster_url = os.path.relpath(poster_link, args.output_root)
                cells.append(
                    f'<article><h3>{execution.title()}</h3><video controls preload="metadata" '
                    f'poster="{html.escape(poster_url)}" src="{html.escape(video_url)}"></video>'
                    f'<p>{rollout["inference_calls"]} replans · '
                    f'{median_call_ms:.0f} ms median model call · {rollout["inference_round_trip_s"]:.2f}s total model wait · '
                    f'{rollout["simulator_video_s"]:.2f}s simulator · '
                    f'{"success" if rollout["success"] else "failure"}</p></article>'
                )
            rows.append(f'<section><h2>k = {k}</h2><div class="pair">{"".join(cells)}</div></section>')
        page = args.output_root / f"{slug}.html"
        page.write_text(
            f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
:root{{--bg:#f4f6f8;--surface:#fff;--line:#d8dde3;--text:#1b232c;--muted:#64707d}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid var(--line);padding:14px 22px}}
h1{{font-size:18px;margin:0 0 4px}}header p,p{{margin:0;color:var(--muted)}}main{{max-width:1420px;margin:auto;padding:18px}}
section{{margin-bottom:22px}}h2{{font-size:16px;margin:0 0 8px}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
article{{background:var(--surface);border:1px solid var(--line);padding:10px}}h3{{font-size:14px;margin:0 0 8px}}
video{{display:block;width:100%;max-height:74vh;background:#111}}article p{{padding-top:8px;font-size:12px}}
@media(max-width:800px){{.pair{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>{html.escape(title)}</h1><p>Left: OxyGen · Right: isolated baseline · real simulator and measured inference time</p></header>
<main>{"".join(rows)}</main><script>
document.querySelectorAll('.pair').forEach(pair=>{{const vs=[...pair.querySelectorAll('video')];vs.forEach(v=>v.addEventListener('play',()=>vs.filter(x=>x!==v).forEach(x=>{{x.currentTime=v.currentTime;x.play()}})))}});
</script></body></html>''',
            encoding="utf-8",
        )
        task_pages.append((slug, title))

    links = "".join(f'<li><a href="{slug}.html">{html.escape(title)}</a></li>' for slug, title in task_pages)
    (args.output_root / "index.html").write_text(
        f'<!doctype html><meta charset="utf-8"><title>LIBERO comparison</title><h1>LIBERO rollout comparison</h1><ul>{links}</ul>',
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
