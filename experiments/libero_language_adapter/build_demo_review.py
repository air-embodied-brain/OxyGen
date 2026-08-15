"""Render publication-style OxyGen versus blocking-baseline rollout demos."""

# ruff: noqa: PLW0603, RUF001

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from openpi.models import tokenizer as tokenizer_lib

SELECTIONS = {0: (3, 0), 6: (1, 3), 8: (2, 2)}
SOURCE_SIZE = 256
SIDE_WIDTH = 960
HEIGHT = 720
HEADER_HEIGHT = 106
BODY_HEIGHT = 470
FOOTER_HEIGHT = HEIGHT - HEADER_HEIGHT - BODY_HEIGHT
VIEW_WIDTH = 470
QUEUE_X = 492
QUEUE_WIDTH = SIDE_WIDTH - QUEUE_X - 22
OUTPUT_FPS = 25

BACKGROUND = "#0b1017"
PANEL = "#111a25"
PANEL_ALT = "#182433"
TEXT = "#f2f5f9"
MUTED = "#9aa7b6"
LINE = "#334155"
INFERENCE = "#f4b13d"
SIMULATION = "#35c99a"
OXYGEN = "#65a9ff"
BASELINE = "#c2cad5"
COMPLETE = "#67d9a5"
FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    if bold:
        bold_font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold_font.exists():
            return ImageFont.truetype(str(bold_font), size)
    for font in FONT_CANDIDATES:
        if font.exists():
            return ImageFont.truetype(str(font), size)
    return ImageFont.load_default(size=size)


def _configure_layout(layout: str) -> None:
    global SIDE_WIDTH, QUEUE_WIDTH
    SIDE_WIDTH = 1080 if layout == "vertical" else 960
    QUEUE_WIDTH = SIDE_WIDTH - QUEUE_X - 22


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rollout(run_root: Path, episode: int) -> dict:
    return next(record for record in _load_jsonl(run_root / "rollouts.jsonl") if record["episode_idx"] == episode)


def _episode_records(path: Path, episode: int) -> list[dict]:
    return [record for record in _load_jsonl(path) if record["episode_idx"] == episode]


def _prepare_segments(run_root: Path, episode: int) -> list[dict]:
    raw_segments = _episode_records(run_root / "timeline_segments.jsonl", episode)
    inference_events = {
        int(event["step"]): event for event in _episode_records(run_root / "inference_events.jsonl", episode)
    }
    segments = []
    inference_frames = 0
    simulation_frames = 0
    display_cursor = 0
    for raw_segment in raw_segments:
        source_frames = raw_segment["end_frame"] - raw_segment["start_frame"]
        segment = {
            **raw_segment,
            "source_start_frame": raw_segment["start_frame"],
            "source_end_frame": raw_segment["end_frame"],
            "start_frame": display_cursor,
        }
        segment["inference_before"] = inference_frames
        segment["simulation_before"] = simulation_frames
        if segment["phase"] == "model_inference":
            inference_ms = float(inference_events[int(segment["step"])]["server_timing"]["infer_ms"])
            display_frames = max(1, round(inference_ms / 1000 * 100))
            segment["inference_ms"] = inference_ms
            inference_frames += display_frames
        else:
            display_frames = source_frames
            simulation_frames += display_frames
        display_cursor += display_frames
        segment["end_frame"] = display_cursor
        segments.append(segment)
    return segments


def _segment_at(segments: list[dict], source_frame: float) -> dict:
    for segment in segments:
        if segment["start_frame"] <= source_frame < segment["end_frame"]:
            return segment
    return segments[-1]


def _visible_requests(requests: list[dict], *, completed_slots: int) -> list[dict]:
    unfinished = sorted(
        (request for request in requests if not request["is_finished"]),
        key=lambda request: request["start_step"],
        reverse=True,
    )
    completed = sorted(
        (request for request in requests if request["is_finished"]),
        key=lambda request: request["start_step"],
        reverse=True,
    )[:completed_slots]
    return unfinished + completed


def _apply_updates(requests: list[dict], updates: list[dict], *, step: int) -> list[dict]:
    updated = {request["request_id"]: dict(request) for request in requests}
    for update in updates:
        request_id = update["request_id"]
        previous = updated.get(request_id, {})
        updated[request_id] = {
            "request_id": request_id,
            "start_step": previous.get("start_step", step),
            "is_finished": bool(update["is_finished"]),
            "tokens": list(update["tokens_full"]),
            "new_tokens": list(update["tokens_this_frame"]),
        }
    return list(updated.values())


def _request_transitions(run_root: Path, episode: int) -> dict[int, dict]:
    requests: list[dict] = []
    transitions = {}
    request_ids: dict[str, str] = {}
    for event in _episode_records(run_root / "inference_events.jsonl", episode):
        updates = []
        for update in event["language_updates"]:
            original_id = update["request_id"]
            if original_id not in request_ids:
                request_ids[original_id] = f"mem_{len(request_ids)}"
            updates.append({**update, "request_id": request_ids[original_id]})
        before = [dict(request) for request in requests]
        requests = _apply_updates(requests, updates, step=int(event["step"]))
        transitions[int(event["step"])] = {
            "before": before,
            "after": [dict(request) for request in requests],
            "updates": updates,
        }
    return transitions


def _queue_layout(transitions: dict[int, dict]) -> tuple[int, int]:
    max_unfinished = max(
        1,
        *(
            sum(not request["is_finished"] for request in state)
            for item in transitions.values()
            for state in (item["before"], item["after"])
        ),
    )
    completed_slots = 1 if max_unfinished >= 8 else 3
    return max_unfinished + completed_slots, completed_slots


def _transition_at(transitions: dict[int, dict], step: int) -> dict | None:
    eligible = [event_step for event_step in transitions if event_step <= step]
    return transitions[max(eligible)] if eligible else None


def _decode(tokenizer: tokenizer_lib.PaligemmaTokenizer, tokens: list[int]) -> str:
    if not tokens:
        return ""
    return tokenizer.detokenize(np.asarray(tokens, dtype=np.int32)).replace("<eos>", "").strip()


def _token_labels(tokenizer: tokenizer_lib.PaligemmaTokenizer, tokens: list[int]) -> list[str]:
    labels = []
    previous = ""
    for index in range(len(tokens)):
        current = _decode(tokenizer, tokens[: index + 1])
        delta = current[len(previous) :].strip() if current.startswith(previous) else current
        labels.append(delta or "·")
        previous = current
    return labels


def _animated_requests(
    transition: dict | None,
    progress: float,
    tokenizer: tokenizer_lib.PaligemmaTokenizer,
    *,
    completed_slots: int,
) -> list[dict]:
    if transition is None:
        return []
    before_by_id = {request["request_id"]: request for request in transition["before"]}
    before_visible = _visible_requests(transition["before"], completed_slots=completed_slots)
    after_visible = _visible_requests(transition["after"], completed_slots=completed_slots)
    before_index = {request["request_id"]: index for index, request in enumerate(before_visible)}
    after_index = {request["request_id"]: index for index, request in enumerate(after_visible)}
    animated = []
    for request in after_visible:
        request_id = request["request_id"]
        before = before_by_id.get(request_id)
        old_tokens = before.get("tokens", []) if before else []
        new_tokens = request.get("tokens", [])[len(old_tokens) :]
        reveal = (
            len(new_tokens) if progress >= 1 else min(len(new_tokens), math.floor(progress * (len(new_tokens) + 1)))
        )
        tokens = old_tokens + new_tokens[:reveal]
        is_new = request_id not in before_index
        start_y = before_index.get(request_id, after_index[request_id])
        end_y = after_index[request_id]
        smooth = progress * progress * (3 - 2 * progress)
        animated.append(
            {
                **request,
                "tokens": tokens,
                "text": _decode(tokenizer, tokens),
                "token_labels": _token_labels(tokenizer, tokens),
                "row": start_y + (end_y - start_y) * smooth,
                "scale": max(0.08, smooth) if is_new else 1.0,
                "exiting": False,
                "is_finished": request["is_finished"] and reveal == len(new_tokens),
            }
        )
    for request in before_visible:
        request_id = request["request_id"]
        if request_id in after_index:
            continue
        tokens = request.get("tokens", [])
        animated.append(
            {
                **request,
                "text": _decode(tokenizer, tokens),
                "token_labels": _token_labels(tokenizer, tokens),
                "row": float(before_index[request_id]),
                "scale": max(0.0, 1.0 - smooth),
                "exiting": True,
            }
        )
    return animated


def _stable_requests(
    transition: dict | None,
    tokenizer: tokenizer_lib.PaligemmaTokenizer,
    *,
    completed_slots: int,
) -> list[dict]:
    if transition is None:
        return []
    requests = []
    for row, request in enumerate(_visible_requests(transition["after"], completed_slots=completed_slots)):
        tokens = request.get("tokens", [])
        requests.append(
            {
                **request,
                "text": _decode(tokenizer, tokens),
                "token_labels": _token_labels(tokenizer, tokens),
                "row": float(row),
                "scale": 1.0,
                "exiting": False,
            }
        )
    return requests


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> str:
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text.rstrip() + "…"


def _draw_token_stream(
    draw: ImageDraw.ImageDraw,
    token_count: int,
    box: tuple[int, int, int, int],
    *,
    complete: bool,
) -> None:
    left, top, right, bottom = box
    visible_count = min(24, token_count)
    for index in range(visible_count):
        x = left + index * 15
        if x + 10 > right:
            break
        newest = index == visible_count - 1
        fill = COMPLETE if complete else (OXYGEN if newest else "#52708f")
        draw.rounded_rectangle((x, top, x + 10, bottom), radius=3, fill=fill)


def _render_side(
    image: Image.Image,
    *,
    task: str,
    label: str,
    phase: str,
    phase_progress: float,
    phase_ms: float,
    requests: list[dict],
    max_rows: int,
    inference_s: float,
    simulation_s: float,
    step: int,
) -> Image.Image:
    canvas = Image.new("RGB", (SIDE_WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    phase_color = INFERENCE if phase == "model_inference" else SIMULATION
    label_color = OXYGEN if "OxyGen" in label else BASELINE
    draw.rectangle((0, 0, SIDE_WIDTH, 7), fill=phase_color)
    draw.text((24, 22), label, font=_font(28, bold=True), fill=label_color)
    phase_label = "MODEL INFERENCE" if phase == "model_inference" else "SIMULATOR"
    draw.text(
        (SIDE_WIDTH - 24, 24),
        phase_label,
        font=_font(16, bold=True),
        fill=phase_color,
        anchor="ra",
    )
    detail = f"environment paused · {phase_ms:.0f} ms" if phase == "model_inference" else "20 Hz control · model idle"
    draw.text((SIDE_WIDTH - 24, 53), detail, font=_font(12), fill=MUTED, anchor="ra")
    draw.text((24, 72), "TASK  " + task, font=_font(13, bold=True), fill=TEXT)

    view = image.resize((VIEW_WIDTH, BODY_HEIGHT), Image.Resampling.LANCZOS)
    canvas.paste(view, (0, HEADER_HEIGHT))
    draw.rectangle((0, HEADER_HEIGHT, VIEW_WIDTH, HEADER_HEIGHT + BODY_HEIGHT), outline=LINE, width=2)
    draw.rounded_rectangle(
        (QUEUE_X, HEADER_HEIGHT, SIDE_WIDTH - 22, HEADER_HEIGHT + BODY_HEIGHT),
        radius=8,
        fill=PANEL,
        outline=LINE,
        width=2,
    )
    draw.text(
        (QUEUE_X + 16, HEADER_HEIGHT + 15),
        "TEXTUAL MEMORY",
        font=_font(14, bold=True),
        fill=TEXT,
    )
    draw.text((SIDE_WIDTH - 38, HEADER_HEIGHT + 15), "newest first", font=_font(12), fill=MUTED, anchor="ra")
    draw.line((QUEUE_X + 16, HEADER_HEIGHT + 42, SIDE_WIDTH - 38, HEADER_HEIGHT + 42), fill=LINE, width=1)

    queue_top = HEADER_HEIGHT + 51
    queue_bottom = HEADER_HEIGHT + BODY_HEIGHT - 14
    row_height = min(72, (queue_bottom - queue_top) / max_rows)
    row_gap = 5
    # Exiting cards are drawn first so active cards sliding downward remain on top.
    ordered_requests = sorted(requests, key=lambda request: (not request.get("exiting", False), request["row"]))
    for request in ordered_requests:
        y = queue_top + request["row"] * row_height
        height = row_height - row_gap
        scale = float(request.get("scale", 1.0))
        if scale <= 0:
            continue
        center_y = y + height / 2
        scaled_height = max(2.0, height * scale)
        y = center_y - scaled_height / 2
        height = scaled_height
        if y + height < queue_top or y > queue_bottom:
            continue
        complete = request["is_finished"]
        fill = "#172b26" if complete else PANEL_ALT
        outline = COMPLETE if complete else OXYGEN
        draw.rounded_rectangle(
            (QUEUE_X + 12, y, SIDE_WIDTH - 34, y + height), radius=7, fill=fill, outline=outline, width=1
        )
        if scale < 0.95:
            continue
        status = "COMPLETE" if complete else "GENERATING"
        draw.text(
            (QUEUE_X + 24, y + 7),
            status,
            font=_font(10, bold=True),
            fill=COMPLETE if complete else OXYGEN,
        )
        draw.text((SIDE_WIDTH - 46, y + 7), request["request_id"], font=_font(10), fill=MUTED, anchor="ra")
        if height >= 60:
            text = request["text"] or "waiting for first token"
            draw.text(
                (QUEUE_X + 24, y + 27), _fit_text(draw, text, _font(13), QUEUE_WIDTH - 60), font=_font(13), fill=TEXT
            )
            _draw_token_stream(
                draw,
                len(request["token_labels"]),
                (QUEUE_X + 24, int(y + height - 11), SIDE_WIDTH - 46, int(y + height - 6)),
                complete=complete,
            )
        else:
            text = request["text"] or "waiting"
            draw.text(
                (QUEUE_X + 24, y + 22), _fit_text(draw, text, _font(11), QUEUE_WIDTH - 60), font=_font(11), fill=TEXT
            )
            _draw_token_stream(
                draw,
                len(request["token_labels"]),
                (QUEUE_X + 125, int(y + 10), SIDE_WIDTH - 115, int(y + 14)),
                complete=complete,
            )

    footer_top = HEADER_HEIGHT + BODY_HEIGHT
    draw.rectangle((0, footer_top, SIDE_WIDTH, HEIGHT), fill="#111821")
    draw.rectangle((0, footer_top, SIDE_WIDTH, footer_top + 5), fill=phase_color)
    metrics = [
        (24, "SERVER INFERENCE", f"{inference_s:.2f} s", INFERENCE),
        (int(SIDE_WIDTH * 0.292), "SIMULATOR", f"{simulation_s:.2f} s", SIMULATION),
        (int(SIDE_WIDTH * 0.5375), "ELAPSED", f"{inference_s + simulation_s:.2f} s", TEXT),
        (int(SIDE_WIDTH * 0.792), "SIM FRAME", str(step), MUTED),
    ]
    for x, metric, value, color in metrics:
        draw.text((x, footer_top + 23), metric, font=_font(11, bold=True), fill=MUTED)
        draw.text((x, footer_top + 52), value, font=_font(24, bold=True), fill=color)
    bar_left, bar_right, bar_y = 24, SIDE_WIDTH - 24, footer_top + 99
    total = max(inference_s + simulation_s, 1e-6)
    split = bar_left + int((bar_right - bar_left) * inference_s / total)
    draw.rounded_rectangle((bar_left, bar_y, bar_right, bar_y + 8), radius=4, fill="#263241")
    draw.rounded_rectangle((bar_left, bar_y, max(bar_left + 4, split), bar_y + 8), radius=4, fill=INFERENCE)
    draw.rectangle((split, bar_y, bar_right, bar_y + 8), fill=SIMULATION)
    return canvas


def _decoder(path: Path) -> subprocess.Popen:
    return subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )


def _read_frame(process: subprocess.Popen) -> bytes:
    assert process.stdout is not None
    size = SOURCE_SIZE * SOURCE_SIZE * 3
    chunks, remaining = [], size
    while remaining:
        chunk = process.stdout.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _render_pair(
    oxygen_root: Path,
    baseline_root: Path,
    oxygen: dict,
    baseline: dict,
    output: Path,
    tokenizer: tokenizer_lib.PaligemmaTokenizer,
    layout: str,
) -> None:
    roots = (oxygen_root, baseline_root)
    records = (oxygen, baseline)
    labels = ("Ours (OxyGen)", "Baseline (openpi)")
    segments = [_prepare_segments(root, record["episode_idx"]) for root, record in zip(roots, records, strict=True)]
    transitions = [
        _request_transitions(root, record["episode_idx"]) for root, record in zip(roots, records, strict=True)
    ]
    layouts = [_queue_layout(item) for item in transitions]
    decoders = [_decoder(root / record["video"]) for root, record in zip(roots, records, strict=True)]
    output_width = SIDE_WIDTH * 2 if layout == "horizontal" else SIDE_WIDTH
    output_height = HEIGHT if layout == "horizontal" else HEIGHT * 2
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{output_width}x{output_height}",
        "-r",
        str(OUTPUT_FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "21",
        "-tune",
        "animation",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert encoder.stdin is not None
    last_images: list[Image.Image | None] = [None, None]
    decoder_positions = [-1, -1]
    total_display_frames = max(items[-1]["end_frame"] for items in segments)
    output_frames = math.ceil(total_display_frames / 100 * OUTPUT_FPS)
    for output_index in range(output_frames):
        display_frame = output_index * 100 / OUTPUT_FPS
        sides = []
        for index, (record, label) in enumerate(zip(records, labels, strict=True)):
            display_index = min(int(display_frame), segments[index][-1]["end_frame"] - 1)
            segment = _segment_at(segments[index], display_index)
            frames = segment["end_frame"] - segment["start_frame"]
            progress = min(1.0, max(0.0, (display_index - segment["start_frame"] + 1) / frames))
            source_frames = segment["source_end_frame"] - segment["source_start_frame"]
            source_index = min(
                segment["source_end_frame"] - 1,
                segment["source_start_frame"] + int(progress * source_frames),
            )
            raw = b""
            if source_index > decoder_positions[index]:
                for _ in range(source_index - decoder_positions[index]):
                    candidate = _read_frame(decoders[index])
                    if candidate:
                        raw = candidate
                decoder_positions[index] = source_index
            if raw:
                last_images[index] = Image.frombytes("RGB", (SOURCE_SIZE, SOURCE_SIZE), raw)
            image = last_images[index]
            assert image is not None
            transition = _transition_at(transitions[index], int(segment["step"]))
            max_rows, completed_slots = layouts[index]
            if segment["phase"] == "model_inference":
                requests = _animated_requests(
                    transition,
                    progress,
                    tokenizer,
                    completed_slots=completed_slots,
                )
                inference_frames = segment["inference_before"] + progress * frames
                simulation_frames = segment["simulation_before"]
            else:
                requests = _stable_requests(transition, tokenizer, completed_slots=completed_slots)
                inference_frames = segment["inference_before"]
                simulation_frames = segment["simulation_before"] + progress * frames
            sides.append(
                _render_side(
                    image,
                    task=record["task_description"],
                    label=label,
                    phase=segment["phase"],
                    phase_progress=progress,
                    phase_ms=float(segment.get("inference_ms", 50.0)),
                    requests=requests,
                    max_rows=max_rows,
                    inference_s=inference_frames / 100,
                    simulation_s=simulation_frames / 100,
                    step=int(segment["step"]),
                )
            )
        pair = Image.new("RGB", (output_width, output_height), BACKGROUND)
        pair.paste(sides[0], (0, 0))
        pair.paste(sides[1], (SIDE_WIDTH, 0) if layout == "horizontal" else (0, HEIGHT))
        encoder.stdin.write(pair.tobytes())
    encoder.stdin.close()
    if encoder.wait():
        raise subprocess.CalledProcessError(encoder.returncode, command)
    for decoder in decoders:
        if decoder.stdout is not None:
            decoder.stdout.close()
        decoder.terminate()
        decoder.wait()


def _write_site(output_root: Path, selections: list[dict]) -> None:
    cards = [
        f"""<article><header><div><span>LIBERO-10 · Task {selection["task_id"]}</span><h2>{html.escape(selection["task"])}</h2></div>
<strong>{selection["speedup"]:.2f}× rollout</strong></header><video controls preload="none" src="{html.escape(selection["video"])}"></video>
<footer><b>k={selection["k"]}</b><span>OxyGen {selection["oxygen_timeline_s"]:.1f}s</span><span>Baseline {selection["baseline_timeline_s"]:.1f}s</span><span>success / success</span></footer></article>"""
        for selection in selections
    ]
    document = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OxyGen demo review</title><style>
:root{{--bg:#090d13;--panel:#111821;--line:#303b4a;--text:#eef2f7;--muted:#9da8b5;--blue:#3f9cff;--green:#22c58b}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}}.top{{padding:20px 26px;border-bottom:1px solid var(--line);background:#0e141d;position:sticky;top:0;z-index:2}}h1{{font-size:21px;margin:0 0 4px}}.top p{{margin:0;color:var(--muted)}}main{{max-width:1500px;margin:auto;padding:24px}}article{{background:var(--panel);border:1px solid var(--line);border-radius:9px;margin-bottom:24px;overflow:hidden}}article header{{padding:14px 18px;display:flex;justify-content:space-between;gap:16px;align-items:center}}article header span{{color:var(--muted);font-size:12px}}h2{{font-size:15px;margin:3px 0 0}}strong{{color:var(--green);font-size:18px;white-space:nowrap}}video{{display:block;width:100%;background:#000}}footer{{display:flex;gap:28px;padding:12px 18px;color:var(--muted);border-top:1px solid var(--line)}}footer b{{color:var(--blue)}}@media(max-width:720px){{main{{padding:10px}}article header{{align-items:flex-start;flex-direction:column}}footer{{gap:12px;flex-wrap:wrap}}}}</style></head><body><div class="top"><h1>OxyGen × LIBERO visual-memory demos</h1><p>Measured server inference plus 1× simulator playback; transport and client serialization are excluded. Orange: model inference with the environment paused. Green: 20 Hz simulator execution.</p></div><main>{"".join(cards)}</main></body></html>"""
    (output_root / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oxygen-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--layout", choices=("horizontal", "vertical"), default="horizontal")
    parser.add_argument("--task-id", type=int, action="append")
    args = parser.parse_args()
    _configure_layout(args.layout)
    args.output_root.mkdir(parents=True, exist_ok=True)
    tokenizer = tokenizer_lib.PaligemmaTokenizer(max_len=48)
    selections = []
    for task_id, (k, episode) in SELECTIONS.items():
        if args.task_id and task_id not in args.task_id:
            continue
        oxygen_root = args.oxygen_root / f"k{k}" / f"task{task_id}"
        baseline_root = args.baseline_root / f"task{task_id}"
        oxygen = _rollout(oxygen_root, episode)
        baseline = _rollout(baseline_root, episode)
        output = args.output_root / "videos" / f"task{task_id:02d}_k{k}_episode{episode}.mp4"
        _render_pair(oxygen_root, baseline_root, oxygen, baseline, output, tokenizer, args.layout)
        oxygen_s = _prepare_segments(oxygen_root, episode)[-1]["end_frame"] / 100
        baseline_s = _prepare_segments(baseline_root, episode)[-1]["end_frame"] / 100
        selection = {
            "task_id": task_id,
            "task": oxygen["task_description"],
            "k": k,
            "episode": episode,
            "oxygen_timeline_s": oxygen_s,
            "baseline_timeline_s": baseline_s,
            "speedup": baseline_s / oxygen_s,
            "video": str(output.relative_to(args.output_root)),
            "oxygen_source": str(oxygen_root),
            "baseline_source": str(baseline_root),
        }
        selections.append(selection)
        print(json.dumps(selection, sort_keys=True), flush=True)
    (args.output_root / "selections.json").write_text(json.dumps(selections, indent=2) + "\n", encoding="utf-8")
    _write_site(args.output_root, selections)


if __name__ == "__main__":
    main()
