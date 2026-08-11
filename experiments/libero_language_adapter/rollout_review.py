#!/usr/bin/env python3
"""Run real LIBERO rollouts and render model text into a lazy review site."""

from __future__ import annotations

import argparse
import collections
import datetime
import html
import json
import logging
import math
from pathlib import Path
import subprocess
import textwrap
import time

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
SUITE_LABELS = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "LIBERO-10",
}


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()  # noqa: UP017 - Python 3.8 client env


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def _wrapped_lines(text: str, *, width: int, font_size: int, max_lines: int) -> list[str]:
    characters = max(12, int(width / (font_size * 0.56)))
    lines = textwrap.wrap(text, width=characters, break_long_words=False) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .") + "..."
    return lines


def _render_frame(image: np.ndarray, caption: dict, *, success: bool | None) -> Image.Image:
    canvas = Image.new("RGB", (640, 704), "#f5f7f9")
    view = Image.fromarray(image).resize((512, 512), Image.Resampling.LANCZOS)
    canvas.paste(view, (64, 0))
    draw = ImageDraw.Draw(canvas)
    label_font = _font(18)
    body_font = _font(17)
    small_font = _font(14)

    draw.text((18, 528), "Task", font=label_font, fill="#5a6470")
    y = 527
    for line in _wrapped_lines(caption["task"], width=500, font_size=17, max_lines=2):
        draw.text((112, y), line, font=body_font, fill="#15191e")
        y += 22

    y = max(y + 8, 582)
    draw.text((18, y), "Model", font=label_font, fill="#1769aa")
    model_text = caption["text"].strip() or "(generating...)"
    for line in _wrapped_lines(model_text, width=500, font_size=17, max_lines=2):
        draw.text((112, y), line, font=body_font, fill="#15191e")
        y += 22

    status = "complete" if caption["is_finished"] else "partial"
    outcome = "running" if success is None else ("success" if success else "failure")
    footer = (
        f"step {caption['step']}  |  request {caption['request_id'] or '-'}  |  text {status}  |  rollout {outcome}"
    )
    draw.text((18, 676), footer, font=small_font, fill="#68717c")
    return canvas


def _render_video(
    frames: list[np.ndarray],
    captions: list[dict],
    output_path: Path,
    *,
    fps: int,
    success: bool,
) -> tuple[int, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    poster_path = output_path.parent.parent / "posters" / f"{output_path.stem}.jpg"
    poster_path.parent.mkdir(parents=True, exist_ok=True)
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
        "640x704",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "29",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    if len(frames) != len(captions):
        raise ValueError(f"Frame/caption count mismatch: {len(frames)} vs {len(captions)}")
    for index, (frame, caption) in enumerate(zip(frames, captions)):  # noqa: B905 - Python 3.8 client env
        rendered = _render_frame(frame, caption, success=success if index == len(frames) - 1 else None)
        if index == len(frames) // 2:
            rendered.save(poster_path, quality=82, optimize=True)
        process.stdin.write(rendered.tobytes())
    process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    size = output_path.stat().st_size
    if size > 5_000_000:
        raise ValueError(f"Rendered video exceeds 5 MB: {output_path} ({size} bytes)")
    return size, f"posters/{poster_path.name}"


def _get_env(task, *, seed: int):
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(seed)
    return env


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(quat[3]) / denominator


def _observation(obs: dict, task_description: str, *, resize_size: int) -> tuple[dict, np.ndarray]:
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    model_image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))
    model_wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_image, resize_size, resize_size))
    element = {
        "observation/image": model_image,
        "observation/wrist_image": model_wrist,
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": task_description,
    }
    return element, image


def _close_client(client) -> None:
    websocket = getattr(client, "_ws", None)
    if websocket is not None:
        websocket.close()


def _run_episode(
    *,
    host: str,
    port: int,
    suite_name: str,
    task_id: int,
    episode_idx: int,
    seed: int,
    resize_size: int,
    replan_steps: int,
    num_steps_wait: int,
) -> tuple[dict, list[np.ndarray], list[dict], list[dict]]:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    task_description = str(task.language)
    initial_states = suite.get_task_init_states(task_id)
    if episode_idx >= len(initial_states):
        raise IndexError(f"Episode {episode_idx} is unavailable for {suite_name} task {task_id}")

    env = _get_env(task, seed=seed)
    client = websocket_client_policy.WebsocketClientPolicy(host, port)
    metadata = client.get_server_metadata()
    frames: list[np.ndarray] = []
    captions: list[dict] = []
    inference_events: list[dict] = []
    action_plan: collections.deque = collections.deque()
    latest = {"text": "", "request_id": None, "is_finished": False}
    done = False
    exception = None
    started = time.monotonic()
    env.reset()
    obs = env.set_init_state(initial_states[episode_idx])
    step = 0
    try:
        while step < MAX_STEPS[suite_name] + num_steps_wait:
            if step < num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                step += 1
                continue

            element, review_image = _observation(obs, task_description, resize_size=resize_size)
            if not action_plan:
                response = client.infer(element)
                actions = response["actions"]
                if len(actions) < replan_steps:
                    raise ValueError(f"Policy returned {len(actions)} actions; need {replan_steps}")
                action_plan.extend(actions[:replan_steps])
                latest = {
                    "text": str(response.get("text", "")),
                    "request_id": response.get("request_id"),
                    "is_finished": bool(response.get("is_finished", False)),
                }
                inference_events.append(
                    {
                        "step": step,
                        **latest,
                        "tokens_full": np.asarray(response.get("tokens_full", [])).astype(int).tolist(),
                        "server_timing": response.get("server_timing"),
                        "policy_timing": response.get("policy_timing"),
                    }
                )

            frames.append(review_image)
            captions.append({"task": task_description, "step": step, **latest})
            action = action_plan.popleft()
            obs, _, done, _ = env.step(np.asarray(action).tolist())
            if done:
                break
            step += 1
    except Exception as error:
        logging.exception("Rollout failed")
        exception = f"{type(error).__name__}: {error}"
    finally:
        _close_client(client)
        env.close()

    record = {
        "event": "rollout",
        "timestamp_utc": _utc_now(),
        "task_suite": suite_name,
        "task_id": task_id,
        "task_description": task_description,
        "episode_idx": episode_idx,
        "initial_state_idx": episode_idx,
        "success": bool(done),
        "exception": exception,
        "env_steps": step,
        "elapsed_s": time.monotonic() - started,
        "seed": seed,
        "policy_seed": metadata.get("policy_seed"),
        "server_metadata": metadata,
        "max_steps": MAX_STEPS[suite_name],
        "num_steps_wait": num_steps_wait,
        "replan_steps": replan_steps,
        "inference_calls": len(inference_events),
    }
    return record, frames, captions, inference_events


def _build_html(items: list[dict], output_path: Path) -> None:
    buttons = ['<button class="active" data-filter="all">All</button>'] + [
        f'<button data-filter="{suite}">{label}</button>' for suite, label in SUITE_LABELS.items()
    ]
    cards = []
    for item in items:
        outcome = "Success" if item["success"] else "Failure"
        cards.append(
            f"""<article class="item" data-suite="{html.escape(item["suite"])}">
  <div class="video-shell"><video controls muted playsinline preload="none" data-src="{html.escape(item["video"])}" data-poster="{html.escape(item["poster"])}"></video></div>
  <div class="body"><div class="outcome {"ok" if item["success"] else "bad"}">{outcome}</div>
  <h2>{html.escape(item["task"])}</h2>
  <p>{html.escape(SUITE_LABELS[item["suite"]])} · task {item["task_id"]} · initial state {item["episode_idx"]} · {item["frames"]} frames · {item["bytes"] / 1_000_000:.2f} MB</p></div>
</article>"""
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LIBERO suffix-LoRA rollout review</title><style>
:root{{--bg:#f3f5f7;--surface:#fff;--text:#171a1e;--muted:#69727d;--line:#d9dee4;--accent:#1769aa;--ok:#217346;--bad:#a23d35}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:2;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line);padding:14px 24px}}
.bar{{max-width:1480px;margin:auto;display:flex;align-items:center;gap:18px;flex-wrap:wrap}}h1{{font-size:18px;margin:0}}
.filters{{display:flex;gap:6px;flex-wrap:wrap}}button{{border:1px solid var(--line);background:#fff;padding:7px 11px;border-radius:5px;cursor:pointer}}button.active{{background:var(--accent);border-color:var(--accent);color:#fff}}
main{{max-width:1480px;margin:20px auto;padding:0 20px;display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:18px}}
.item{{background:var(--surface);border:1px solid var(--line);border-radius:6px;overflow:hidden}}.item[hidden]{{display:none}}
.video-shell{{aspect-ratio:640/704;background:#111}}video{{display:block;width:100%;height:100%;object-fit:contain}}
.body{{padding:12px 14px 14px}}h2{{font-size:14px;margin:5px 0;font-weight:600}}p{{margin:0;color:var(--muted);font-size:12px}}
.outcome{{font-size:12px;font-weight:700}}.ok{{color:var(--ok)}}.bad{{color:var(--bad)}}
@media(max-width:520px){{header{{padding:12px}}main{{padding:0 10px;grid-template-columns:1fr}}}}
</style></head><body><header><div class="bar"><h1>LIBERO suffix-LoRA rollouts</h1><div class="filters">{"".join(buttons)}</div></div></header>
<main>{"".join(cards)}</main><script>
const videos=[...document.querySelectorAll('video[data-src]')];
const load=v=>{{if(!v.src){{v.poster=v.dataset.poster;v.src=v.dataset.src;v.load()}}}};
const observer=new IntersectionObserver(es=>es.forEach(e=>{{if(e.isIntersecting)load(e.target)}}),{{rootMargin:'240px'}});videos.forEach(v=>observer.observe(v));
document.querySelectorAll('button[data-filter]').forEach(b=>b.addEventListener('click',()=>{{document.querySelectorAll('button').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.querySelectorAll('.item').forEach(x=>x.hidden=b.dataset.filter!=='all'&&x.dataset.suite!==b.dataset.filter)}}));
</script></body></html>"""
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--suites", default=",".join(SUITE_LABELS))
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", default="0,1,2,3,4")
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--video-fps", type=int, default=30)
    args = parser.parse_args()

    suites = [value.strip() for value in args.suites.split(",") if value.strip()]
    episodes = [int(value) for value in args.episodes.split(",") if value.strip()]
    unknown = sorted(set(suites) - set(SUITE_LABELS))
    if unknown:
        raise ValueError(f"Unknown suites: {unknown}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rollout_path = args.output_root / "rollouts.jsonl"
    event_path = args.output_root / "inference_events.jsonl"
    if rollout_path.exists() or event_path.exists():
        raise FileExistsError(f"Use an empty output directory: {args.output_root}")

    items = []
    for suite in suites:
        for episode_idx in episodes:
            record, frames, captions, events = _run_episode(
                host=args.host,
                port=args.port,
                suite_name=suite,
                task_id=args.task_id,
                episode_idx=episode_idx,
                seed=args.seed,
                resize_size=args.resize_size,
                replan_steps=args.replan_steps,
                num_steps_wait=args.num_steps_wait,
            )
            stem = f"{suite}__task{args.task_id:02d}__episode{episode_idx:02d}"
            video_path = args.output_root / "videos" / f"{stem}.mp4"
            size, poster = _render_video(
                frames,
                captions,
                video_path,
                fps=args.video_fps,
                success=record["success"],
            )
            record.update({"video": f"videos/{video_path.name}", "video_bytes": size})
            _append_jsonl(rollout_path, record)
            for event in events:
                _append_jsonl(
                    event_path,
                    {
                        "task_suite": suite,
                        "task_id": args.task_id,
                        "episode_idx": episode_idx,
                        **event,
                    },
                )
            items.append(
                {
                    "suite": suite,
                    "task_id": args.task_id,
                    "episode_idx": episode_idx,
                    "task": record["task_description"],
                    "success": record["success"],
                    "frames": len(frames),
                    "bytes": size,
                    "video": record["video"],
                    "poster": poster,
                }
            )
            print(json.dumps(record, sort_keys=True), flush=True)

    (args.output_root / "manifest.json").write_text(
        json.dumps(items, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _build_html(items, args.output_root / "index.html")
    print(
        json.dumps(
            {
                "episodes": len(items),
                "successes": sum(item["success"] for item in items),
                "review_page": str(args.output_root / "index.html"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
