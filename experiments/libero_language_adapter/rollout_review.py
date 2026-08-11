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
import shutil
import subprocess
import textwrap
import time

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
LIBERO_CONTROL_HZ = 20
REVIEW_WIDTH = 640
REVIEW_HEIGHT = 860
REVIEW_ASSET_VERSION = "request-buffer-v3"
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


def _bold_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)


def _wrapped_lines(text: str, *, width: int, font_size: int, max_lines: int) -> list[str]:
    characters = max(12, int(width / (font_size * 0.56)))
    lines = textwrap.wrap(text, width=characters, break_long_words=False) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .") + "..."
    return lines


def _fit_text(draw: ImageDraw.ImageDraw, text: str, *, font: ImageFont.FreeTypeFont, width: int) -> str:
    if draw.textlength(text, font=font) <= width:
        return text
    suffix = "..."
    while text and draw.textlength(text + suffix, font=font) > width:
        text = text[:-1]
    return text.rstrip() + suffix


def _render_request_buffer(
    canvas: Image.Image,
    caption: dict,
    *,
    success: bool | None,
    playback_fps: int,
    source_control_hz: int,
) -> Image.Image:
    draw = ImageDraw.Draw(canvas)
    label_font = _bold_font(15)
    body_font = _font(15)
    small_font = _font(13)

    draw.text((18, 526), "Task", font=label_font, fill="#5a6470")
    y = 526
    for line in _wrapped_lines(caption["task"], width=525, font_size=15, max_lines=2):
        draw.text((88, y), line, font=body_font, fill="#15191e")
        y += 20

    draw.line((18, 580, 622, 580), fill="#d5dbe1", width=1)
    draw.text((18, 592), "Language request buffer", font=_bold_font(14), fill="#1769aa")
    draw.text((477, 592), "newest first", font=small_font, fill="#68717c")

    requests = caption.get("requests", [])
    row_labels = ["t", "t-1", "t-2"]
    for index in range(3):
        row_y = 618 + index * 61
        request = requests[index] if index < len(requests) else None
        is_current = index == 0 and request is not None
        fill = "#e8f2fa" if is_current else "#edf0f3"
        outline = "#8db8d8" if is_current else "#d5dbe1"
        draw.rounded_rectangle((18, row_y, 622, row_y + 52), radius=4, fill=fill, outline=outline, width=1)
        draw.text((28, row_y + 16), row_labels[index], font=_bold_font(15), fill="#1769aa" if is_current else "#68717c")
        if request is None:
            draw.text((78, row_y + 16), "-", font=body_font, fill="#9aa2aa")
            continue

        complete = bool(request["is_finished"])
        status = "COMPLETE" if complete else "GENERATING"
        status_color = "#217346" if complete else "#1769aa"
        draw.text((78, row_y + 7), status, font=_bold_font(11), fill=status_color)
        metadata = (
            f"{request['request_id']}  |  start frame {request['start_step']}  |  {request['token_count']} tokens"
        )
        draw.text((180, row_y + 7), metadata, font=small_font, fill="#68717c")
        text = request["text"].strip() or "(waiting for first tokens)"
        draw.text(
            (78, row_y + 27),
            _fit_text(draw, text, font=body_font, width=526),
            font=body_font,
            fill="#15191e",
        )

    speed = playback_fps / source_control_hz
    outcome = "running" if success is None else ("success" if success else "failure")
    footer = (
        f"sim frame {caption['step']}  |  playback {speed:.1f}x real time "
        f"({playback_fps} FPS / {source_control_hz} Hz)  |  rollout {outcome}"
    )
    draw.text((18, 836), footer, font=small_font, fill="#68717c")
    return canvas


def _render_frame(
    image: np.ndarray,
    caption: dict,
    *,
    success: bool | None,
    playback_fps: int,
    source_control_hz: int,
) -> Image.Image:
    canvas = Image.new("RGB", (REVIEW_WIDTH, REVIEW_HEIGHT), "#f5f7f9")
    view = Image.fromarray(image).resize((512, 512), Image.Resampling.LANCZOS)
    canvas.paste(view, (64, 0))
    return _render_request_buffer(
        canvas,
        caption,
        success=success,
        playback_fps=playback_fps,
        source_control_hz=source_control_hz,
    )


def _render_video(
    frames: list[np.ndarray],
    captions: list[dict],
    output_path: Path,
    *,
    fps: int,
    source_control_hz: int,
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
        f"{REVIEW_WIDTH}x{REVIEW_HEIGHT}",
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
        rendered = _render_frame(
            frame,
            caption,
            success=success if index == len(frames) - 1 else None,
            playback_fps=fps,
            source_control_hz=source_control_hz,
        )
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
    from libero.libero import get_libero_path  # noqa: PLC0415 - optional rollout dependency
    from libero.libero.envs import OffScreenRenderEnv  # noqa: PLC0415 - optional rollout dependency

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
    from openpi_client import image_tools  # noqa: PLC0415 - optional rollout dependency

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


def _language_updates(response: dict) -> list[dict]:
    updates = response.get("language_updates")
    if updates is None:
        updates = [
            {
                "request_id": response.get("request_id"),
                "text": response.get("text", ""),
                "is_finished": response.get("is_finished", False),
                "tokens_full": response.get("tokens_full", []),
                "tokens_this_frame": response.get("tokens_this_frame", []),
                "created_this_call": False,
            }
        ]
    return [
        {
            "request_id": update.get("request_id"),
            "text": str(update.get("text", "")),
            "is_finished": bool(update.get("is_finished", False)),
            "tokens_full": np.asarray(update.get("tokens_full", [])).astype(int).tolist(),
            "tokens_this_frame": np.asarray(update.get("tokens_this_frame", [])).astype(int).tolist(),
            "created_this_call": bool(update.get("created_this_call", False)),
        }
        for update in updates
    ]


def _apply_request_updates(request_buffer: list[dict], updates: list[dict], *, step: int) -> None:
    index_by_id = {request["request_id"]: index for index, request in enumerate(request_buffer)}
    for update in updates:
        request_id = update["request_id"]
        current = {
            "request_id": request_id,
            "text": update["text"],
            "is_finished": update["is_finished"],
            "token_count": len(update["tokens_full"]),
            "start_step": step,
        }
        if request_id in index_by_id:
            index = index_by_id[request_id]
            current["start_step"] = request_buffer[index]["start_step"]
            request_buffer[index] = current
        else:
            request_buffer.append(current)
        index_by_id = {request["request_id"]: index for index, request in enumerate(request_buffer)}
    request_buffer.sort(key=lambda request: request["start_step"], reverse=True)


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
    from libero.libero import benchmark  # noqa: PLC0415 - optional rollout dependency
    from openpi_client import websocket_client_policy  # noqa: PLC0415 - optional rollout dependency

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
    request_buffer: list[dict] = []
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
                updates = _language_updates(response)
                _apply_request_updates(request_buffer, updates, step=step)
                inference_events.append(
                    {
                        "step": step,
                        "request_id": response.get("request_id"),
                        "text": str(response.get("text", "")),
                        "is_finished": bool(response.get("is_finished", False)),
                        "tokens_full": np.asarray(response.get("tokens_full", [])).astype(int).tolist(),
                        "language_updates": updates,
                        "active_language_requests": response.get("active_language_requests"),
                        "server_timing": response.get("server_timing"),
                        "policy_timing": response.get("policy_timing"),
                    }
                )

            frames.append(review_image)
            captions.append(
                {
                    "task": task_description,
                    "step": step,
                    "requests": [dict(request) for request in request_buffer[:3]],
                }
            )
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


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _buffer_captions(record: dict, events: list[dict], *, frame_count: int) -> list[dict]:
    events = sorted(events, key=lambda event: event["step"])
    request_buffer: list[dict] = []
    captions = []
    event_index = 0
    first_step = int(record["num_steps_wait"])
    for step in range(first_step, first_step + frame_count):
        while event_index < len(events) and int(events[event_index]["step"]) <= step:
            event = events[event_index]
            updates = _language_updates(event)
            _apply_request_updates(request_buffer, updates, step=int(event["step"]))
            event_index += 1
        captions.append(
            {
                "task": record["task_description"],
                "step": step,
                "requests": [dict(request) for request in request_buffer[:3]],
            }
        )
    return captions


def _read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _rerender_video(
    source_path: Path,
    captions: list[dict],
    output_path: Path,
    *,
    fps: int,
    source_control_hz: int,
    success: bool,
) -> tuple[int, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    poster_path = output_path.parent.parent / "posters" / f"{output_path.stem}.jpg"
    poster_path.parent.mkdir(parents=True, exist_ok=True)
    decoder_command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-vf",
        "crop=640:512:0:0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    encoder_command = [
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
        f"{REVIEW_WIDTH}x{REVIEW_HEIGHT}",
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
    decoder = subprocess.Popen(decoder_command, stdout=subprocess.PIPE)
    encoder = subprocess.Popen(encoder_command, stdin=subprocess.PIPE)
    assert decoder.stdout is not None
    assert encoder.stdin is not None
    frame_bytes = REVIEW_WIDTH * 512 * 3
    for index, caption in enumerate(captions):
        frame = _read_exact(decoder.stdout, frame_bytes)
        if len(frame) != frame_bytes:
            raise ValueError(f"Source video ended at frame {index}: {source_path}")
        source_view = Image.frombytes("RGB", (REVIEW_WIDTH, 512), frame)
        canvas = Image.new("RGB", (REVIEW_WIDTH, REVIEW_HEIGHT), "#f5f7f9")
        canvas.paste(source_view, (0, 0))
        rendered = _render_request_buffer(
            canvas,
            caption,
            success=success if index == len(captions) - 1 else None,
            playback_fps=fps,
            source_control_hz=source_control_hz,
        )
        if index == len(captions) // 2:
            rendered.save(poster_path, quality=82, optimize=True)
        encoder.stdin.write(rendered.tobytes())

    trailing = decoder.stdout.read()
    decoder_return_code = decoder.wait()
    encoder.stdin.close()
    encoder_return_code = encoder.wait()
    if decoder_return_code:
        raise subprocess.CalledProcessError(decoder_return_code, decoder_command)
    if encoder_return_code:
        raise subprocess.CalledProcessError(encoder_return_code, encoder_command)
    if trailing:
        extra_frames = len(trailing) / frame_bytes
        raise ValueError(f"Source video has {extra_frames:.2f} unexpected trailing frames: {source_path}")
    size = output_path.stat().st_size
    if size > 5_000_000:
        raise ValueError(f"Rendered video exceeds 5 MB: {output_path} ({size} bytes)")
    return size, f"posters/{poster_path.name}"


def _rerender_existing(
    source_root: Path,
    output_root: Path,
    *,
    fps: int,
    source_control_hz: int,
) -> None:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if source_root == output_root:
        raise ValueError("Rerender into a separate output directory")
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output_root}")

    manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    records = _load_jsonl(source_root / "rollouts.jsonl")
    events = _load_jsonl(source_root / "inference_events.jsonl")
    record_by_key = {
        (record["task_suite"], int(record["task_id"]), int(record["episode_idx"])): record for record in records
    }
    events_by_key: dict[tuple[str, int, int], list[dict]] = collections.defaultdict(list)
    for event in events:
        key = (event["task_suite"], int(event["task_id"]), int(event["episode_idx"]))
        events_by_key[key].append(event)

    output_items = []
    output_records = []
    for item in manifest:
        key = (item["suite"], int(item["task_id"]), int(item["episode_idx"]))
        record = dict(record_by_key[key])
        captions = _buffer_captions(record, events_by_key[key], frame_count=int(item["frames"]))
        video_path = output_root / item["video"]
        size, poster = _rerender_video(
            source_root / item["video"],
            captions,
            video_path,
            fps=fps,
            source_control_hz=source_control_hz,
            success=bool(item["success"]),
        )
        output_item = {
            **item,
            "bytes": size,
            "poster": poster,
            "playback_fps": fps,
            "source_control_hz": source_control_hz,
            "playback_speed": fps / source_control_hz,
            "request_buffer_rows": 3,
        }
        output_items.append(output_item)
        record.update(
            {
                "video_bytes": size,
                "render_playback_fps": fps,
                "source_control_hz": source_control_hz,
                "render_playback_speed": fps / source_control_hz,
                "request_buffer_rows": 3,
            }
        )
        output_records.append(record)
        print(json.dumps({"video": item["video"], "bytes": size}, sort_keys=True), flush=True)

    with (output_root / "rollouts.jsonl").open("w", encoding="utf-8") as stream:
        for record in output_records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    shutil.copy2(source_root / "inference_events.jsonl", output_root / "inference_events.jsonl")
    (output_root / "manifest.json").write_text(
        json.dumps(output_items, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _build_html(output_items, output_root / "index.html")


def _build_html(items: list[dict], output_path: Path) -> None:
    playback_speed = float(items[0].get("playback_speed", 1.0)) if items else 1.0
    buttons = ['<button class="active" data-filter="all">All</button>'] + [
        f'<button data-filter="{suite}">{label}</button>' for suite, label in SUITE_LABELS.items()
    ]
    cards = []
    for item in items:
        outcome = "Success" if item["success"] else "Failure"
        video_url = f"{item['video']}?v={REVIEW_ASSET_VERSION}"
        poster_url = f"{item['poster']}?v={REVIEW_ASSET_VERSION}"
        cards.append(
            f"""<article class="item" data-suite="{html.escape(item["suite"])}">
  <div class="video-shell"><video controls muted playsinline preload="none" data-src="{html.escape(video_url)}" data-poster="{html.escape(poster_url)}"></video></div>
  <div class="body"><div class="outcome {"ok" if item["success"] else "bad"}">{outcome}</div>
  <h2>{html.escape(item["task"])}</h2>
  <p>{html.escape(SUITE_LABELS[item["suite"]])} · task {item["task_id"]} · initial state {item["episode_idx"]} · {item["frames"]} frames · {item.get("playback_speed", 1.0):.1f}x real time · {item["bytes"] / 1_000_000:.2f} MB</p></div>
</article>"""
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LIBERO suffix-LoRA rollout review</title><style>
:root{{--bg:#f3f5f7;--surface:#fff;--text:#171a1e;--muted:#69727d;--line:#d9dee4;--accent:#1769aa;--ok:#217346;--bad:#a23d35}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:2;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line);padding:14px 24px}}
.bar{{max-width:1480px;margin:auto;display:flex;align-items:center;gap:18px;flex-wrap:wrap}}h1{{font-size:18px;margin:0}}.playback{{font-size:12px;color:var(--muted)}}
.filters{{display:flex;gap:6px;flex-wrap:wrap}}button{{border:1px solid var(--line);background:#fff;padding:7px 11px;border-radius:5px;cursor:pointer}}button.active{{background:var(--accent);border-color:var(--accent);color:#fff}}
main{{max-width:1480px;margin:20px auto;padding:0 20px;display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:18px}}
.item{{background:var(--surface);border:1px solid var(--line);border-radius:6px;overflow:hidden}}.item[hidden]{{display:none}}
.video-shell{{aspect-ratio:640/860;background:#111}}video{{display:block;width:100%;height:100%;object-fit:contain}}
.body{{padding:12px 14px 14px}}h2{{font-size:14px;margin:5px 0;font-weight:600}}p{{margin:0;color:var(--muted);font-size:12px}}
.outcome{{font-size:12px;font-weight:700}}.ok{{color:var(--ok)}}.bad{{color:var(--bad)}}
@media(max-width:520px){{header{{padding:12px}}main{{padding:0 10px;grid-template-columns:1fr}}}}
</style></head><body><header><div class="bar"><h1>LIBERO suffix-LoRA rollouts</h1><div class="playback">{playback_speed:.1f}x simulator real time · newest request first</div><div class="filters">{"".join(buttons)}</div></div></header>
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
    parser.add_argument("--rerender-from", type=Path)
    parser.add_argument("--suites", default=",".join(SUITE_LABELS))
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", default="0,1,2,3,4")
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--source-control-hz", type=int, default=LIBERO_CONTROL_HZ)
    args = parser.parse_args()

    if args.video_fps <= 0 or args.source_control_hz <= 0:
        raise ValueError("Video FPS and source control frequency must be positive")
    if args.rerender_from is not None:
        _rerender_existing(
            args.rerender_from,
            args.output_root,
            fps=args.video_fps,
            source_control_hz=args.source_control_hz,
        )
        print(
            json.dumps(
                {
                    "review_page": str(args.output_root / "index.html"),
                    "playback_speed": args.video_fps / args.source_control_hz,
                },
                indent=2,
            )
        )
        return

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
                source_control_hz=args.source_control_hz,
                success=record["success"],
            )
            record.update(
                {
                    "video": f"videos/{video_path.name}",
                    "video_bytes": size,
                    "render_playback_fps": args.video_fps,
                    "source_control_hz": args.source_control_hz,
                    "render_playback_speed": args.video_fps / args.source_control_hz,
                    "request_buffer_rows": 3,
                }
            )
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
                    "playback_fps": args.video_fps,
                    "source_control_hz": args.source_control_hz,
                    "playback_speed": args.video_fps / args.source_control_hz,
                    "request_buffer_rows": 3,
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
