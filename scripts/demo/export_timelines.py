"""Export the exact reconstructed clock used by render_comparison.py.

Usage (from the website root):
    python3 scripts/demo/export_timelines.py --runs-root ../libero_exp/language_adapter_runs/visual_memory_v1_20260813

Only Python's standard library is needed. Original run logs are deliberately
external to this repository. --verify-videos checks the existing 30 fps MP4s.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

TASKS = (
    ("white-mug-and-chocolate-pudding", 6, 3),
    ("alphabet-soup-and-tomato-sauce", 0, 2),
    ("moka-pots", 8, 3),
)
CONTROL_DT = 0.05
FPS = 30
HOLD_S = 2.0


def records(path: Path, episode: int) -> list[dict]:
    return [item for line in path.read_text().splitlines() if line.strip()
            and (item := json.loads(line))["episode_idx"] == episode]


def export_side(run: Path, episode: int) -> dict:
    events = sorted(records(run / "inference_events.jsonl", episode), key=lambda event: event["step"])
    actions = records(run / "action_trajectory.jsonl", episode)
    # replay_hires.py saves frames at each logged action step, plus one final
    # frame at last action step + 1. This is build_side()'s last_step.
    first_step, last_step = actions[0]["step"], actions[-1]["step"] + 1
    assert [a["step"] for a in actions] == list(range(first_step, last_step))
    segments, memory, calls = [], [], []
    t = 0.0
    for i, event in enumerate(events):
        infer_s = event["server_timing"]["infer_ms"] / 1000
        timing = event["policy_timing"]
        if "action_ms" in timing:
            action_ms, language_ms = timing["action_ms"], timing["language_ms"]
        else:
            action_ms = timing["prefill_actions_ms"]
            language_ms = timing["text_gen_ms"] + timing["post_proc_ms"]
        split = t + infer_s * action_ms / (action_ms + language_ms)
        end = t + infer_s
        calls.append({"start": t, "actionEnd": split, "end": end, "updates": [{"requestId": u["request_id"], "text": u["text"], "full": len(u["tokens_full"]), "new": len(u["tokens_this_frame"]), "finished": bool(u["is_finished"])} for u in event["language_updates"]]})
        segments.extend([
            {"start": t, "end": split, "kind": "action", "step": event["step"]},
            {"start": split, "end": end, "kind": "language", "step": event["step"]},
        ])
        for update in event["language_updates"]:
            if update["is_finished"]:
                memory.append({"time": end, "text": update["text"]})
        t = end
        next_step = events[i + 1]["step"] if i + 1 < len(events) else last_step
        end = t + (next_step - event["step"]) * CONTROL_DT
        segments.append({"start": t, "end": end, "kind": "move", "step": event["step"]})
        t = end
    assert all(s["end"] >= s["start"] for s in segments)
    assert all(a["end"] == b["start"] for a, b in zip(segments, segments[1:]))
    return {"duration": t, "firstStep": first_step, "lastStep": last_step,
            "segments": segments, "calls": calls, "memory": memory}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("public/data/demo-timelines.json"))
    parser.add_argument("--verify-videos", action="store_true")
    args = parser.parse_args()
    result, evidence = {}, {}
    for slug, task, episode in TASKS:
        item = {"taskId": task, "episode": episode, "controlDt": CONTROL_DT}
        evidence[slug] = {}
        for side, folder in (("oxygen", "demo_campaign_v2/oxygen/k1"), ("baseline", "blocking_baseline_eos_v2")):
            relative = Path(folder) / f"task{task}"
            run = args.runs_root / relative
            item[side] = export_side(run, episode)
            evidence[slug][side] = {
                "run": str(relative),
                "sha256": {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                           for name in ("inference_events.jsonl", "action_trajectory.jsonl")},
            }
        item["total"] = max(item["oxygen"]["duration"], item["baseline"]["duration"])
        if args.verify_videos:
            path = Path("public/videos") / f"{slug}.mp4"
            duration = float(subprocess.check_output([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1", str(path)], text=True))
            expected = int((item["total"] + HOLD_S) * FPS) / FPS
            assert math.isclose(duration, expected, abs_tol=0.000001), (slug, duration, expected)
            print(f"{slug}: MP4 duration verified; OxyGen {item['oxygen']['duration']:.5f}s, baseline {item['baseline']['duration']:.5f}s")
        result[slug] = item
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    args.out.with_name("demo-timelines.provenance.json").write_text(json.dumps({
        "runsRoot": args.runs_root.name,
        "timingSource": "render_comparison.py build_side; server inference durations split by policy timing ratios; simulated movement at 20 Hz",
        "evidence": evidence,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
