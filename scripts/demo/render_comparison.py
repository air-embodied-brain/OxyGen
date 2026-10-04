"""Render a wall-clock OxyGen vs. isolated-baseline comparison video.

Inputs are the logged rollout records (inference_events.jsonl, rollouts.jsonl)
and high-resolution frames replayed from the logged actions by replay_hires.py.
Model time comes from the logged server inference latency; simulator time runs
at the 20 Hz control rate, so the output plays at 1x wall-clock speed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080
FPS = 30
VIEW_ASPECT = 3 / 4  # simulator view height / width
CONTROL_DT = 0.05
HOLD_S = 2.0

BG = (7, 9, 13)
CARD = (13, 17, 23)
CARD_LINE = (30, 37, 48)
TEXT = (236, 240, 245)
MUTED = (132, 142, 156)
FAINT = (64, 73, 86)
ACTION = (63, 224, 255)
LANGUAGE = (255, 181, 71)
COMPUTE = (120, 132, 150)
OXYGEN = ACTION
BASELINE = (176, 184, 196)

FONT_DIR = Path(__file__).resolve().parent / "fonts"


def font(name: str, size: int, weight: int | None = None) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(FONT_DIR / name), size)
    if weight is not None:
        f.set_variation_by_axes([min(max(size, 14), 32), weight] if name == "Inter.ttf" else [weight])
    return f


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@dataclass
class Segment:
    start: float
    end: float
    kind: str  # "action", "language", "move"
    step: int


@dataclass
class Side:
    label: str
    sublabel: str
    color: tuple
    frames: np.ndarray
    steps: list[int]
    segments: list[Segment] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)  # (t_start, t_split, t_end, event)
    duration: float = 0.0
    speedup: float = 0.0


def build_side(run: Path, episode: int, npz: Path, *, label: str, sublabel: str, color: tuple) -> Side:
    events = [e for e in load_jsonl(run / "inference_events.jsonl") if e["episode_idx"] == episode]
    events.sort(key=lambda e: e["step"])
    data = np.load(npz)
    side = Side(label, sublabel, color, data["frames"], [int(s) for s in data["steps"]])
    last_step = side.steps[-1]
    t = 0.0
    for i, event in enumerate(events):
        infer_s = event["server_timing"]["infer_ms"] / 1000
        timing = event["policy_timing"]
        if "action_ms" in timing:  # blocking baseline
            action_ms, language_ms = timing["action_ms"], timing["language_ms"]
        else:  # OxyGen: shared prefill + action, then batched decoding and bookkeeping
            action_ms = timing["prefill_actions_ms"]
            language_ms = timing["text_gen_ms"] + timing["post_proc_ms"]
        split = t + infer_s * action_ms / (action_ms + language_ms)
        side.segments.append(Segment(t, split, "action", event["step"]))
        side.segments.append(Segment(split, t + infer_s, "language", event["step"]))
        side.calls.append({"t0": t, "t1": split, "t2": t + infer_s, "event": event})
        t += infer_s
        next_step = events[i + 1]["step"] if i + 1 < len(events) else last_step
        move = (next_step - event["step"]) * CONTROL_DT
        side.segments.append(Segment(t, t + move, "move", event["step"]))
        t += move
    side.duration = t
    return side


def frame_at(side: Side, t: float) -> np.ndarray:
    for seg in side.segments:
        if seg.start <= t < seg.end:
            if seg.kind != "move":
                step = seg.step
            else:
                step = seg.step + int((t - seg.start) / CONTROL_DT) + 1
            break
    else:
        step = side.steps[-1]
    index = min(max(step - side.steps[0], 0), len(side.steps) - 1)
    return side.frames[index]


def phase_at(side: Side, t: float) -> Segment | None:
    for seg in side.segments:
        if seg.start <= t < seg.end:
            return seg
    return None


def requests_at(side: Side, t: float) -> dict[str, dict]:
    """Replay language updates; tokens of a call appear during its language phase."""
    state: dict[str, dict] = {}
    for call in side.calls:
        if t < call["t1"]:
            break
        updates = call["event"]["language_updates"]
        progress = 1.0 if t >= call["t2"] else (t - call["t1"]) / max(call["t2"] - call["t1"], 1e-6)
        for update in updates:
            rid = update["request_id"]
            prev = state.get(rid, {"tokens": 0, "text": "", "finished": False, "born": call["t1"]})
            full = len(update["tokens_full"])
            new = len(update["tokens_this_frame"])
            shown = full - new + int(round(new * progress))
            state[rid] = {
                "tokens": shown,
                "text": update["text"] if shown == full else prev["text"],
                "finished": bool(update["is_finished"]) and shown == full,
                "born": prev["born"],
                "done_at": call["t2"] if update["is_finished"] else None,
            }
    return state


class Painter:
    def __init__(self) -> None:
        self.title = font("InterTight.ttf", 40, 640)
        self.sub = font("Inter.ttf", 20, 450)
        self.clock = font("JBM.ttf", 44, 500)
        self.clock_unit = font("JBM.ttf", 22, 400)
        self.label = font("JBM.ttf", 15, 600)
        self.memory = font("Inter.ttf", 27, 500)
        self.pill = font("JBM.ttf", 16, 600)
        self.badge = font("InterTight.ttf", 30, 640)
        self.legend = font("Inter.ttf", 17, 450)

    def render(self, sides: list[Side], t: float, total: float) -> Image.Image:
        canvas = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(canvas)
        margin, gap = (48, 32) if W > H else (36, 28)
        for i, side in enumerate(sides):
            if W > H:
                pane_w = (W - 2 * margin - gap) // 2
                self.pane(canvas, draw, side, margin + i * (pane_w + gap), 40, pane_w, H - 80, t, total)
            else:
                pane_h = (H - 2 * margin - gap) // 2
                self.pane(canvas, draw, side, margin, margin + i * (pane_h + gap), W - 2 * margin, pane_h, t, total)
        return canvas

    def pane(self, canvas, draw, side: Side, x, y, w, h, t, total) -> None:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=28, fill=CARD, outline=CARD_LINE, width=2)
        pad = 32
        local_t = min(t, side.duration)
        finished = t >= side.duration
        # Header
        draw.ellipse((x + pad, y + 38, x + pad + 14, y + 52), fill=side.color)
        draw.text((x + pad + 28, y + 24), side.label, font=self.title, fill=TEXT)
        draw.text((x + pad + 28, y + 74), side.sublabel, font=self.sub, fill=MUTED)
        clock = f"{local_t:5.1f}"
        cw = draw.textlength(clock, font=self.clock)
        uw = draw.textlength(" s", font=self.clock_unit)
        cx = x + w - pad - cw - uw
        draw.text((cx, y + 28), clock, font=self.clock, fill=side.color if finished else TEXT)
        draw.text((cx + cw, y + 46), " s", font=self.clock_unit, fill=MUTED)
        # Simulator view, cropped to 4:3
        view_y = y + 124
        view_w = w - 2 * pad
        aspect = VIEW_ASPECT
        view_h = int(view_w * aspect)
        frame = frame_at(side, local_t)
        src = frame.shape[0]
        crop_h = int(src * aspect)
        top = int((src - crop_h) * 0.75)
        img = Image.fromarray(frame[top : top + crop_h]).resize((view_w, view_h), Image.LANCZOS)
        seg = phase_at(side, local_t) if not finished else None
        waiting = seg is not None and seg.kind != "move"
        if waiting:
            img = Image.blend(img, Image.new("RGB", img.size, (0, 0, 0)), 0.16)
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, view_w, view_h), radius=18, fill=255)
        canvas.paste(img, (x + pad, view_y), mask)
        # Status pill
        if finished:
            text, color = "TASK COMPLETE", side.color
        elif waiting:
            if seg.kind == "action":
                text, color = "WAITING · ACTION INFERENCE", COMPUTE
            elif side.color == OXYGEN:
                text, color = "WAITING · BATCHED LANGUAGE", LANGUAGE
            else:
                text, color = "WAITING · BLOCKING LANGUAGE", LANGUAGE
        else:
            text, color = "ROBOT MOVING", ACTION
        tw = draw.textlength(text, font=self.pill)
        px, py = x + pad + 20, view_y + view_h - 56
        draw.rounded_rectangle((px, py, px + tw + 52, py + 36), radius=18, fill=(8, 10, 14))
        draw.ellipse((px + 16, py + 12, px + 28, py + 24), fill=color)
        draw.text((px + 38, py + 8), text, font=self.pill, fill=color)
        if finished:
            badge = f"Done in {side.duration:.1f} s"
            if side.speedup and t >= total:
                badge += f" · {side.speedup:.2f}× sooner"
            bw = draw.textlength(badge, font=self.badge)
            bx, by = x + pad + view_w - bw - 60, view_y + 24
            draw.rounded_rectangle((bx, by, bx + bw + 40, by + 58), radius=29, fill=side.color)
            draw.text((bx + 20, by + 10), badge, font=self.badge, fill=BG)
        # Timeline on a shared scale
        tl_y = view_y + view_h + 34
        draw.text((x + pad, tl_y), "WALL-CLOCK TIMELINE", font=self.label, fill=MUTED)
        idle = sum(max(0.0, min(s.end, local_t) - s.start) for s in side.segments if s.kind != "move")
        idle_text = f"ROBOT IDLE {idle:4.1f} s"
        iw = draw.textlength(idle_text, font=self.label)
        draw.text((x + pad + view_w - iw, tl_y), idle_text, font=self.label, fill=TEXT)
        bar_y = tl_y + 28
        bar_h = 22
        bar_w = view_w
        draw.rounded_rectangle((x + pad, bar_y, x + pad + bar_w, bar_y + bar_h), radius=6, fill=(20, 25, 33))
        scale = bar_w / total
        colors = {"move": ACTION, "action": COMPUTE, "language": LANGUAGE}
        for s in side.segments:
            if s.start >= local_t:
                break
            x0 = x + pad + s.start * scale
            x1 = x + pad + min(s.end, local_t) * scale
            if x1 - x0 >= 0.5:
                draw.rectangle((x0, bar_y, x1, bar_y + bar_h), fill=colors[s.kind])
        if not finished:
            head = x + pad + local_t * scale
            draw.rectangle((head - 1, bar_y - 6, head + 1, bar_y + bar_h + 6), fill=TEXT)
        # Legend
        lx = x + pad
        ly = bar_y + bar_h + 14
        for name, color in (("robot moving", ACTION), ("action inference", COMPUTE), ("language inference", LANGUAGE)):
            draw.rounded_rectangle((lx, ly + 5, lx + 14, ly + 17), radius=3, fill=color)
            draw.text((lx + 22, ly), name, font=self.legend, fill=MUTED)
            lx += 22 + draw.textlength(name, font=self.legend) + 30
        # Memory
        mem_y = ly + 54
        reqs = requests_at(side, local_t)
        active = [] if finished else [r for r in reqs.values() if not r["finished"]]
        done = [r for r in reqs.values() if r["finished"]]
        draw.text((x + pad, mem_y), "TEXTUAL MEMORY", font=self.label, fill=MUTED)
        info = f"{len(active)} request{'s' if len(active) != 1 else ''} decoding"
        iw = draw.textlength(info, font=self.label)
        draw.text((x + pad + view_w - iw, mem_y), info, font=self.label, fill=LANGUAGE if active else FAINT)
        # in-flight requests as token bars
        bx = x + pad + view_w - iw - 24
        for r in sorted(active, key=lambda r: r["born"])[:14]:
            hgt = min(18, 3 + r["tokens"] * 1.6)
            draw.rounded_rectangle((bx - 6, mem_y + 18 - hgt, bx, mem_y + 18), radius=2, fill=LANGUAGE)
            bx -= 11
        latest = max(done, key=lambda r: r["done_at"] or 0)["text"] if done else "—"
        draw.text((x + pad, mem_y + 32), f"“{latest}”" if done else latest, font=self.memory, fill=TEXT if done else FAINT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oxygen-run", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--oxygen-frames", type=Path, required=True)
    parser.add_argument("--baseline-frames", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--poster", type=Path)
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--preview", type=float, help="render a single frame at this time")
    parser.add_argument("--vertical", action="store_true", help="stack the two runs for portrait screens")
    args = parser.parse_args()
    global W, H, VIEW_ASPECT
    if args.vertical:
        W, H, VIEW_ASPECT = 1080, 1920, 9 / 16

    sides = [
        build_side(args.oxygen_run, args.episode, args.oxygen_frames, label="OxyGen",
                   sublabel=f"shared KV cache · cross-frame batching · k={args.k}", color=OXYGEN),
        build_side(args.baseline_run, args.episode, args.baseline_frames, label="Isolated baseline",
                   sublabel="separate prefill · language blocks actions", color=BASELINE),
    ]
    total = max(s.duration for s in sides)
    sides[0].speedup = sides[1].duration / sides[0].duration
    painter = Painter()
    if args.preview is not None:
        painter.render(sides, args.preview, total).save(args.out)
        return
    n = int((total + HOLD_S) * FPS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "22", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-tune", "animation", str(args.out)],
        stdin=subprocess.PIPE,
    )
    for i in range(n):
        frame = painter.render(sides, i / FPS, total)
        proc.stdin.write(frame.tobytes())
        if args.poster and i == int(min(sides[0].duration * 0.45, total) * FPS):
            frame.save(args.poster, quality=86)
    proc.stdin.close()
    proc.wait()
    if args.meta:
        args.meta.write_text(json.dumps({
            "oxygen_s": round(sides[0].duration, 2),
            "baseline_s": round(sides[1].duration, 2),
            "k": args.k,
            "episode": args.episode,
        }, indent=2))


if __name__ == "__main__":
    main()
