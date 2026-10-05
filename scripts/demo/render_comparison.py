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
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FPS = 30
VIEW_ASPECT = 3 / 4  # simulator view height / width
CONTROL_DT = 0.05
HOLD_S = 2.0

# Colors follow the project page (src/styles/global.css). --theme picks one.
THEMES = {
    "light": {
        "paper": (250, 248, 243), "deep": (241, 237, 228), "ink": (23, 22, 15),
        "soft": (61, 59, 52), "muted": (110, 106, 96), "faint": (162, 157, 145),
        "hair": (226, 221, 210), "action": (11, 111, 134), "compute": (142, 137, 125),
        "language": (176, 106, 16), "base": (142, 137, 125),
    },
}
T = THEMES["light"]

FONT_DIR = Path(__file__).resolve().parent / "fonts"


def font(name: str, size: int, **axes: float) -> ImageFont.FreeTypeFont:
    """Load a (variable) font and set named axes, e.g. weight=600, opsz=24."""
    f = ImageFont.truetype(str(FONT_DIR / name), size)
    if axes:
        names = {"Weight": "weight", "Optical Size": "opsz", "Optical size": "opsz"}
        values = []
        for axis in f.get_variation_axes():
            key = names.get(axis["name"].decode(), "")
            values.append(axes.get(key, axis["default"]))
        f.set_variation_by_axes(values)
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
    color: str  # "ours" or "base"
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
    """Editorial layout: hairline rules and type, no boxes, matching the project page."""

    def __init__(self) -> None:
        self.name = font("SourceSans3.woff2", 17, weight=700)
        self.sub = font("SourceSans3.woff2", 19, weight=450)
        self.clock = font("LibreBaskerville.woff2", 46, weight=400)
        self.unit = font("SourceSans3.woff2", 20, weight=500)
        self.label = font("SourceSans3.woff2", 14, weight=650)
        self.status = font("SourceSans3.woff2", 17, weight=650)
        self.done = font("LibreBaskerville.woff2", 30, weight=400)
        self.legend = font("SourceSans3.woff2", 16, weight=450)
        self.memory = font("SourceSerif4-Italic.woff2", 29, weight=400, opsz=28)

    @staticmethod
    def tracked(draw, xy, text, fnt, fill, tracking=1.6):
        """Small caps style: letter-spaced uppercase label."""
        x, y = xy
        for ch in text.upper():
            draw.text((x, y), ch, font=fnt, fill=fill)
            x += draw.textlength(ch, font=fnt) + tracking
        return x

    def render(self, sides: list[Side], t: float, total: float) -> Image.Image:
        canvas = Image.new("RGB", (W, H), T["paper"])
        draw = ImageDraw.Draw(canvas)
        if W > H:
            margin, gap = 64, 56
            pane_w = (W - 2 * margin - gap) // 2
            for i, side in enumerate(sides):
                self.pane(canvas, draw, side, margin + i * (pane_w + gap), 52, pane_w, t, total)
        else:
            margin, gap = 48, 56
            pane_h = (H - 2 * margin - gap) // 2
            for i, side in enumerate(sides):
                self.pane(canvas, draw, side, margin, margin + i * (pane_h + gap), W - 2 * margin, t, total)
        return canvas

    def pane(self, canvas, draw, side: Side, x, y, w, t, total) -> None:
        ours = side.color == "ours"
        accent = T["action"] if ours else T["base"]
        local_t = min(t, side.duration)
        finished = t >= side.duration

        # Header: a strong rule, name in tracked caps, configuration, wall clock.
        draw.rectangle((x, y, x + w, y + 1), fill=T["ink"])
        draw.rectangle((x, y + 24, x + 10, y + 34), fill=accent)
        self.tracked(draw, (x + 20, y + 19), side.label, self.name, T["ink"])
        draw.text((x, y + 48), side.sublabel, font=self.sub, fill=T["muted"])
        clock = f"{local_t:4.1f}"
        cw = draw.textlength(clock, font=self.clock)
        uw = draw.textlength(" s", font=self.unit)
        draw.text((x + w - cw - uw, y + 14), clock, font=self.clock, fill=T["action"] if finished and ours else T["ink"])
        draw.text((x + w - uw, y + 34), " s", font=self.unit, fill=T["muted"])

        # Simulator view.
        view_y = y + 92
        view_h = int(w * VIEW_ASPECT)
        frame = frame_at(side, local_t)
        src = frame.shape[0]
        crop_h = int(src * VIEW_ASPECT)
        top = int((src - crop_h) * 0.75)
        img = Image.fromarray(frame[top : top + crop_h]).resize((w, view_h), Image.LANCZOS)
        seg = phase_at(side, local_t) if not finished else None
        waiting = seg is not None and seg.kind != "move"
        canvas.paste(img, (x, view_y))

        # Status line under the view.
        sy = view_y + view_h + 14
        if finished:
            text = f"Done in {side.duration:.1f} s"
            if side.speedup and t >= total:
                text += f", {side.speedup:.2f}× sooner"
            draw.text((x, sy - 4), text, font=self.done, fill=T["action"] if ours else T["ink"])
        else:
            if not waiting:
                text, color = "Robot moving", T["action"]
            elif seg.kind == "action":
                text, color = "Waiting for action inference", T["compute"]
            elif ours:
                text, color = "Waiting for batched language", T["language"]
            else:
                text, color = "Waiting for blocking language", T["language"]
            draw.rectangle((x, sy + 6, x + 10, sy + 16), fill=color)
            self.tracked(draw, (x + 20, sy + 2), text, self.status, color, 1.2)
        idle = sum(max(0.0, min(s.end, local_t) - s.start) for s in side.segments if s.kind != "move")
        idle_text = f"robot idle {idle:4.1f} s"
        iw = draw.textlength(idle_text, font=self.sub)
        draw.text((x + w - iw, sy + 1), idle_text, font=self.sub, fill=T["soft"])

        # Wall-clock timeline on a shared scale.
        bar_y = sy + 48
        bar_h = 12
        draw.rectangle((x, bar_y, x + w, bar_y + bar_h), fill=T["deep"])
        scale = w / total
        colors = {"move": T["action"], "action": T["compute"], "language": T["language"]}
        for s in side.segments:
            if s.start >= local_t:
                break
            x0 = x + s.start * scale
            x1 = x + min(s.end, local_t) * scale
            if x1 - x0 >= 0.5:
                draw.rectangle((x0, bar_y, x1, bar_y + bar_h), fill=colors[s.kind])
        if not finished:
            head = x + local_t * scale
            draw.rectangle((head, bar_y - 5, head + 1, bar_y + bar_h + 5), fill=T["ink"])
        lx, ly = x, bar_y + bar_h + 10
        for name, color in (("robot moving", T["action"]), ("action inference", T["compute"]), ("language inference", T["language"])):
            draw.rectangle((lx, ly + 6, lx + 12, ly + 15), fill=color)
            draw.text((lx + 18, ly), name, font=self.legend, fill=T["muted"])
            lx += 18 + draw.textlength(name, font=self.legend) + 26

        # Textual memory: latest finished line, in-flight requests as ticks.
        my = ly + 44
        draw.rectangle((x, my, x + w, my), fill=T["hair"])
        reqs = requests_at(side, local_t)
        active = [] if finished else [r for r in reqs.values() if not r["finished"]]
        done = [r for r in reqs.values() if r["finished"]]
        self.tracked(draw, (x, my + 12), "Textual memory", self.label, T["muted"])
        info = f"{len(active)} request{'s' if len(active) != 1 else ''} decoding"
        iw = draw.textlength(info, font=self.legend)
        draw.text((x + w - iw, my + 9), info, font=self.legend, fill=T["language"] if active else T["faint"])
        bx = x + w - iw - 16
        for r in sorted(active, key=lambda r: r["born"])[:16]:
            hgt = min(16, 3 + r["tokens"] * 1.4)
            draw.rectangle((bx - 3, my + 28 - hgt, bx, my + 28), fill=T["language"])
            bx -= 8
        latest = max(done, key=lambda r: r["done_at"] or 0)["text"] if done else "—"
        draw.text((x, my + 40), f"“{latest}”" if done else latest, font=self.memory, fill=T["ink"] if done else T["faint"])


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
                   sublabel=f"shared KV cache, cross-frame batching, k = {args.k}", color="ours"),
        build_side(args.baseline_run, args.episode, args.baseline_frames, label="Isolated baseline",
                   sublabel="separate prefill, language blocks actions", color="base"),
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
