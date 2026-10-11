import { $all, select, initTabs } from "./dom";
import { observeVisibility } from "./visibility";

export function initDemo() {
  const root = document.querySelector<HTMLElement>("[data-demo]");
  if (!root) return;
  type Phase = "action" | "language" | "move";
  type Segment = { start: number; end: number; kind: Phase; step: number };
  type Side = { duration: number; segments: Segment[]; calls: { start: number; actionEnd: number; end: number; updates: { text: string; full: number; new: number; finished: boolean }[] }[]; memory: { time: number; text: string }[]; firstStep: number; lastStep: number };
  type Replay = { total: number; controlDt: number; oxygen: Side; baseline: Side };
  const video = root.querySelector<HTMLVideoElement>("[data-demo-video]")!;
  const tabs = $all<HTMLButtonElement>("[role=tab]", root);
  const play = root.querySelector<HTMLButtonElement>("[data-demo-play]")!;
  const seek = root.querySelector<HTMLInputElement>("[data-demo-seek]")!;
  const tracks = root.querySelector<HTMLElement>("[data-demo-tracks]")!;
  const head = root.querySelector<HTMLElement>("[data-demo-playhead]")!;
  const clock = root.querySelector<HTMLOutputElement>("[data-demo-clock]")!;
  const error = root.querySelector<HTMLElement>("[data-demo-error]")!;
  const overlay = root.querySelector<HTMLButtonElement>("[data-demo-overlay]")!;
  const sides = ["oxygen", "baseline"] as const;
  const labels = { action: "Action inference", language: "Language inference", move: "Executing actions" };
  let replay: Replay | undefined;
  let taskId = tabs[0].dataset.id!;
  let animation = 0;
  let wantsPlay = true;
  let visible = false;
  let dragging = false;
  const data = fetch(`${root.dataset.base}data/demo-timelines.json`).then(response => {
    if (!response.ok) throw new Error("Replay data unavailable");
    return response.json() as Promise<Record<string, Replay>>;
  });
  const render = (time: number) => {
    if (!replay) return;
    // Media timestamps are quantized to microseconds; preserve the exact end state.
    const t = time >= replay.total - 0.00001 ? replay.total : Math.max(0, time);
    seek.value = String(t);
    seek.setAttribute("aria-valuetext", `${t.toFixed(2)} seconds`);
    clock.value = t.toFixed(2);
    head.style.left = `${t / replay.total * 100}%`;
    if (video.paused) updateButton();
    for (const side of sides) {
      const run = replay[side];
      const phase = root.querySelector<HTMLElement>(`[data-demo-phase="${side}"]`)!;
      const progress = root.querySelector<HTMLElement>(`[data-demo-step="${side}"]`)!;
      const segment = run.segments.find(s => s.start <= t && t < s.end);
      const step = segment ? Math.min(run.lastStep, segment.step + (segment.kind === "move" ? Math.floor((t - segment.start) / replay.controlDt) + 1 : 0)) : run.lastStep;
      phase.textContent = segment ? labels[segment.kind] : "Task complete";
      phase.dataset.kind = segment?.kind ?? "done";
      progress.textContent = `Step ${step} / ${run.lastStep}`;
      const output = root.querySelector<HTMLButtonElement>(`[data-demo-output="${side}"]`)!;
      const latest = run.memory.filter(item => item.time <= t).at(-1);
      const activeCall = run.calls.filter(call => call.start <= t).at(-1) ?? run.calls[0];
      const inferenceElapsed = run.segments.reduce((sum, item) => {
        if (item.kind === "move" || item.start >= t) return sum;
        return sum + Math.max(0, Math.min(item.end, t) - item.start);
      }, 0);
      root.querySelector<HTMLElement>(`[data-demo-inference-elapsed="${side}"]`)!.textContent = `Inference ${inferenceElapsed.toFixed(2)} s`;
      const previousCall = run.calls.filter(call => call.end <= t).at(-1);
      const currentText = activeCall?.updates.at(-1)?.text ?? previousCall?.updates.at(-1)?.text ?? run.calls[0]?.updates[0]?.text ?? "Waiting for output";
      root.querySelector<HTMLElement>(`[data-demo-output-text="${side}"]`)!.textContent = currentText;
      output.disabled = side !== "oxygen";
      output.dataset.active = side === "oxygen" ? "true" : "false";
      const popover = root.querySelector<HTMLElement>(`[data-demo-output-popover="${side}"]`)!;
      if (activeCall) {
        popover.innerHTML = activeCall.updates.map((item, index) => `<div><span>Request ${index === 0 ? "t" : `t-${index}`}</span><b>${item.text || "…"}</b></div>`).join("");
      } else popover.innerHTML = latest ? `<div><span>Latest request</span><b>${latest.text}</b></div>` : "<div><span>Output</span><b>—</b></div>";
    }
  };
  const updateButton = () => {
    const playing = !video.paused && !video.ended;
    play.innerHTML = playing ? '<span class="icon-pause"></span>' : '<span class="icon-play"></span>';
    overlay.classList.toggle("is-paused", !playing);
    overlay.setAttribute("aria-label", playing ? "Pause demo" : "Play demo");
    play.setAttribute("aria-label", playing ? "Pause demo" : "Play demo");
  };
  const tick = () => {
    if (dragging) return;
    render(video.currentTime);
    if (replay && video.currentTime >= replay.total) {
      video.pause();
      wantsPlay = false;
      render(replay.total);
      updateButton();
      return;
    }
    if (!video.paused) animation = requestAnimationFrame(tick);
  };
  const start = () => {
    if (!visible || !replay || video.readyState < 1) return;
    if (video.currentTime >= replay.total - 0.001) video.currentTime = 0;
    void video.play().catch(() => { wantsPlay = false; updateButton(); });
  };
  const load = async (tab: HTMLButtonElement) => {
    const id = tab.dataset.id!;
    taskId = id;
    video.pause();
    replay = undefined;
    play.disabled = true;
    seek.disabled = true;
    error.hidden = true;
    video.poster = `${root.dataset.base}posters/${id}-interactive.webp`;
    video.querySelector("source")!.src = `${root.dataset.base}videos/${id}-interactive.mp4`;
    video.setAttribute("aria-label", `${tab.dataset.label}: synchronized OxyGen and isolated baseline replay`);
    root.querySelector("[role=tabpanel]")!.setAttribute("aria-labelledby", tab.id);

    root.querySelector<HTMLAnchorElement>("[data-demo-fallback]")!.href = `${root.dataset.base}videos/${id}.mp4`;
    for (const side of sides) {
      root.querySelector(`[data-demo-duration="${side}"]`)!.textContent = `${Number(tab.dataset[side]).toFixed(1)} s`;
      root.querySelector(`[data-demo-phase="${side}"]`)!.textContent = "Loading replay…";
      root.querySelector(`[data-demo-step="${side}"]`)!.textContent = "";
      root.querySelector<HTMLElement>(`[data-demo-output-text="${side}"]`)!.textContent = "—";
      root.querySelector<HTMLElement>(`[data-demo-output-popover="${side}"]`)!.hidePopover();
      root.querySelector(`[data-demo-track="${side}"]`)!.replaceChildren();
    }
    clock.value = "0.00";
    head.style.left = "0%";
    video.load();
    try {
      const all = await data;
      if (taskId !== id) return;
      replay = all[id];
      if (!replay) throw new Error("Missing task data");
      seek.max = String(replay.total);
      root.querySelector("[data-demo-total]")!.textContent = replay.total.toFixed(2);
      for (const side of sides) {
        const inference = replay[side].segments.filter(segment => segment.kind !== "move").reduce((sum, segment) => sum + segment.end - segment.start, 0);
        root.querySelector(`[data-demo-inference="${side}"]`)!.textContent = `${inference.toFixed(1)} s`;
        if (side === "oxygen") {
          const baselineInference = replay.baseline.segments.filter(segment => segment.kind !== "move").reduce((sum, segment) => sum + segment.end - segment.start, 0);
          root.querySelector(`[data-demo-inference-speedup="oxygen"]`)!.textContent = `(${(baselineInference / inference).toFixed(2)}×)`;
          root.querySelector(`[data-demo-e2e-speedup="oxygen"]`)!.textContent = `(${(replay.total / replay.oxygen.duration).toFixed(2)}×)`;
        }
      }
      for (const side of sides) {
        const track = root.querySelector(`[data-demo-track="${side}"]`)!;
        for (const segment of replay[side].segments) {
          const span = document.createElement("span");
          span.dataset.kind = segment.kind;
          span.style.left = `${segment.start / replay.total * 100}%`;
          span.style.width = `${(segment.end - segment.start) / replay.total * 100}%`;
          track.append(span);
        }
      }
      render(0);
      updateButton();
      play.disabled = video.readyState < 1;
      seek.disabled = video.readyState < 1;
      if (wantsPlay) start();
    } catch {
      if (taskId === id) error.hidden = false;
    }
  };
  video.addEventListener("loadedmetadata", () => {
    if (replay) { play.disabled = false; seek.disabled = false; }
    if (wantsPlay) start();
  });
  const mediaError = () => { error.hidden = false; play.disabled = true; seek.disabled = true; };
  video.addEventListener("error", mediaError);
  video.querySelector("source")!.addEventListener("error", mediaError);
  video.addEventListener("play", () => { updateButton(); cancelAnimationFrame(animation); tick(); });
  video.addEventListener("pause", () => { cancelAnimationFrame(animation); updateButton(); });
  video.addEventListener("seeked", () => { if (!dragging) render(video.currentTime); });
  video.addEventListener("timeupdate", () => { if (!dragging) render(video.currentTime); });
  const togglePlayback = () => {
    wantsPlay = video.paused;
    if (wantsPlay) start(); else video.pause();
  };
  play.addEventListener("click", () => {
    wantsPlay = video.paused;
    if (wantsPlay) start(); else video.pause();
  });
  overlay.addEventListener("click", event => { event.stopPropagation(); togglePlayback(); });
  video.addEventListener("click", togglePlayback);
  const scrub = (value: number) => {
    if (!replay) return;
    wantsPlay = false;
    video.pause();
    const t = Math.max(0, Math.min(replay.total, value));
    video.currentTime = t;
    render(t);
  };
  seek.addEventListener("input", () => scrub(Number(seek.value)));
  seek.addEventListener("keydown", event => {
    if (!replay) return;
    const direction = event.key === "ArrowRight" || event.key === "ArrowUp" ? 1 : event.key === "ArrowLeft" || event.key === "ArrowDown" ? -1 : 0;
    if (direction) { event.preventDefault(); scrub(Number(seek.value) + direction * replay.controlDt); }
  });
  const point = (event: PointerEvent) => {
    const bounds = tracks.getBoundingClientRect();
    scrub((event.clientX - bounds.left) / bounds.width * (replay?.total ?? 0));
  };
  seek.addEventListener("pointerdown", event => {
    if (seek.disabled) return;
    event.preventDefault(); dragging = true; seek.focus(); seek.setPointerCapture(event.pointerId); point(event);
  });
  seek.addEventListener("pointermove", event => { if (dragging) point(event); });
  seek.addEventListener("pointerup", event => { if (dragging) point(event); dragging = false; });
  seek.addEventListener("pointercancel", () => { dragging = false; });
  for (const tab of tabs) {
    tab.addEventListener("click", () => {
      if (tab.dataset.id === taskId) return;
      wantsPlay = !video.paused;
      select(tabs, tab);
      for (const item of tabs) item.tabIndex = item === tab ? 0 : -1;
      void load(tab);
    });
  }
  initTabs(tabs);
  void load(tabs[0]);
  observeVisibility(video, next => {
    visible = next;
    if (visible && wantsPlay) start();
    else if (!visible) video.pause();
  });
}

