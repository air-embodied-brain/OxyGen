#!/usr/bin/env python3
"""Render one random demonstration per task and build a lazy review page."""

from __future__ import annotations

# ruff: noqa: E501, UP006, UP007, UP035
import argparse
import html
import json
from pathlib import Path
import random
import subprocess
import textwrap
from typing import Any, Dict, List, Sequence

import h5py
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

SUITE_LABELS = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "LIBERO-10",
}
COLORS = {"completed": "#26734d", "remaining": "#a24614", "next": "#155da4"}


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def _draw_wrapped(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int, font: Any, fill: str) -> int:
    average = max(12, int(width / (font.size * 0.57)))
    lines = textwrap.wrap(text, width=average, break_long_words=False) or [""]
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + 5
    return y


def render_video(dataset_path: Path, annotation_path: Path, output_path: Path, fps: int) -> Dict[str, Any]:
    rows = _read_rows(annotation_path)
    demo = rows[0]["demo"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x800", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "26", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    title_font = _font(20)
    body_font = _font(17)
    with h5py.File(dataset_path, "r") as dataset:
        images = dataset[f"data/{demo}/obs/agentview_rgb"]
        if len(images) != len(rows):
            raise ValueError(f"Image/annotation length mismatch for {annotation_path}")
        poster_path = output_path.parent.parent / "posters" / (output_path.stem + ".jpg")
        poster_path.parent.mkdir(parents=True, exist_ok=True)
        for frame, (image_array, row) in enumerate(zip(images, rows)):  # noqa: B905
            image = Image.fromarray(image_array[::-1]).resize((512, 512), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (640, 800), "#f7f8fa")
            canvas.paste(image, (64, 0))
            draw = ImageDraw.Draw(canvas)
            y = 524
            for key, label in (("completed", "Completed"), ("remaining", "Remaining"), ("next", "Next")):
                draw.text((16, y), label, font=title_font, fill=COLORS[key])
                y = _draw_wrapped(draw, row["language"][key], 132, y, 492, body_font, "#15181c") + 5
            if frame == len(rows) // 2:
                canvas.save(poster_path, quality=82, optimize=True)
            process.stdin.write(canvas.tobytes())
    process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    size = output_path.stat().st_size
    if size > 5_000_000:
        raise ValueError(f"Rendered video exceeds 5 MB: {output_path} ({size} bytes)")
    return {"frames": len(rows), "bytes": size, "demo": demo, "poster": "posters/" + poster_path.name}


def build_html(items: Sequence[Dict[str, Any]], output_path: Path, *, fps: int, source_hz: int = 20) -> None:
    cards = []
    for item in items:
        cards.append(f'''<article class="video-item" data-suite="{html.escape(item['suite'])}">
  <div class="video-shell"><video controls muted playsinline preload="none" data-src="{html.escape(item['video'])}" data-poster="{html.escape(item['poster'])}"></video></div>
  <h2>{html.escape(item['instruction'])}</h2>
  <p>{html.escape(SUITE_LABELS[item['suite']])} · {html.escape(item['demo'])} · {item['frames']} frames · {item['bytes'] / 1_000_000:.2f} MB</p>
</article>''')
    buttons = ['<button class="active" data-filter="all">All</button>'] + [
        f'<button data-filter="{suite}">{label}</button>' for suite, label in SUITE_LABELS.items()
    ]
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LIBERO language annotation review</title>
<style>
:root{{--bg:#f3f5f7;--surface:#fff;--text:#16191d;--muted:#68717c;--line:#d9dee4;--accent:#1769aa}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:2;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line);padding:14px 24px}}
.bar{{max-width:1480px;margin:auto;display:flex;align-items:center;gap:18px;flex-wrap:wrap}} h1{{font-size:18px;margin:0}}
.filters{{display:flex;gap:6px;flex-wrap:wrap}} button{{border:1px solid var(--line);background:#fff;padding:7px 11px;border-radius:5px;cursor:pointer}}
button.active{{background:var(--accent);border-color:var(--accent);color:#fff}}
main{{max-width:1480px;margin:20px auto;padding:0 20px;display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:18px}}
.video-item{{background:var(--surface);border:1px solid var(--line);border-radius:6px;overflow:hidden}} .video-item[hidden]{{display:none}}
.video-shell{{aspect-ratio:4/5;background:#101214}} video{{display:block;width:100%;height:100%;object-fit:contain}}
h2{{font-size:14px;margin:12px 14px 5px;font-weight:600}} p{{margin:0 14px 13px;color:var(--muted);font-size:12px}}
@media(max-width:520px){{header{{padding:12px}}main{{padding:0 10px;grid-template-columns:1fr}}}}
</style></head><body><header><div class="bar"><h1>LIBERO annotation review</h1><span>{fps / source_hz:.1f}x simulator real time ({fps} FPS / {source_hz} Hz)</span><div class="filters">{''.join(buttons)}</div></div></header>
<main>{''.join(cards)}</main>
<script>
const videos=[...document.querySelectorAll('video[data-src]')];
const load=v=>{{if(!v.src){{v.poster=v.dataset.poster;v.src=v.dataset.src;v.load()}}}};
const observer=new IntersectionObserver(es=>es.forEach(e=>{{if(e.isIntersecting)load(e.target)}}),{{rootMargin:'240px'}});
videos.forEach(v=>observer.observe(v));
document.querySelectorAll('button[data-filter]').forEach(b=>b.addEventListener('click',()=>{{
 document.querySelectorAll('button').forEach(x=>x.classList.remove('active'));b.classList.add('active');
 document.querySelectorAll('.video-item').forEach(x=>x.hidden=b.dataset.filter!=='all'&&x.dataset.suite!==b.dataset.filter);
}}));
</script></body></html>'''
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=int, default=10, help="Playback FPS; source control frequency is 20 Hz")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    items = []
    for task_root in sorted(path for path in args.annotation_root.glob("*/*") if path.is_dir()):
        candidates = sorted(task_root.glob("demo_*.jsonl"))
        if not candidates:
            continue
        annotation_path = rng.choice(candidates)
        rows = _read_rows(annotation_path)
        suite, task = task_root.parent.name, task_root.name
        dataset_path = args.dataset_root / suite / f"{task}_demo.hdf5"
        video_name = f"{suite}__{task}__{rows[0]['demo']}.mp4"
        metadata = render_video(dataset_path, annotation_path, args.output_root / "videos" / video_name, args.fps)
        items.append({
            "suite": suite, "task": task, "instruction": rows[0]["task_instruction"],
            "video": "videos/" + video_name, **metadata,
        })
        print(json.dumps(items[-1]), flush=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(items, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    build_html(items, args.output_root / "index.html", fps=args.fps)
    print(json.dumps({"videos": len(items), "output": str(args.output_root)}, indent=2))


if __name__ == "__main__":
    main()
