const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function $all<T extends Element>(selector: string, root: ParentNode = document) {
  return Array.from(root.querySelectorAll<T>(selector));
}

function select(buttons: HTMLElement[], active: HTMLElement, attr = "aria-selected") {
  for (const button of buttons) button.setAttribute(attr, String(button === active));
}

/** Highlights the current section in the running head. */
function initRunningHead() {
  const links = $all<HTMLAnchorElement>(".running-head nav a");
  const current = document.querySelector<HTMLElement>("[data-current]");
  const sections = $all<HTMLElement>("section[data-title]");
  const update = () => {
    const line = window.innerHeight * 0.3;
    let active: HTMLElement | undefined;
    for (const section of sections) {
      if (section.getBoundingClientRect().top <= line) active = section;
    }
    if (current) current.textContent = active ? `§ ${sections.indexOf(active) + 1} · ${active.dataset.title}` : "";
    for (const link of links) {
      link.classList.toggle("active", Boolean(active) && link.hash === `#${active!.id}`);
    }
  };
  update();
  window.addEventListener("scroll", update, { passive: true });
}

function initTheme() {
  document.querySelector("#theme-toggle")?.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("oxygen-theme", next);
  });
}

/**
 * Counts the headline numbers up from zero once, on first load. The display
 * face has proportional digits, so each number gets a box as wide as the widest
 * value it will show; the unit after it stays still while the digits change,
 * then the box eases to the final width.
 */
function initCountUp() {
  const targets = $all<HTMLElement>("[data-count-up]");
  if (!targets.length) return;
  const duration = 1400;
  const steps = 90;
  const ease = (t: number) => 1 - Math.pow(1 - t, 3);
  const items = targets.map((el) => {
    const text = el.textContent ?? "0";
    const value = Number(text);
    const decimals = (text.split(".")[1] ?? "").length;
    const probe = el.cloneNode() as HTMLElement;
    probe.removeAttribute("data-count-up");
    probe.style.cssText = "position:absolute;visibility:hidden;white-space:nowrap";
    el.after(probe);
    let widest = 0;
    for (let i = 0; i <= steps; i += 1) {
      probe.textContent = (value * ease(i / steps)).toFixed(decimals);
      widest = Math.max(widest, probe.getBoundingClientRect().width);
    }
    probe.textContent = text;
    const final = probe.getBoundingClientRect().width;
    probe.remove();
    el.style.width = `${Math.ceil(widest)}px`;
    return { el, text, value, decimals, final };
  });
  const start = performance.now();
  const frame = (now: number) => {
    const t = Math.min(1, (now - start) / duration);
    for (const item of items) {
      item.el.textContent = t < 1 ? (item.value * ease(t)).toFixed(item.decimals) : item.text;
    }
    if (t < 1) {
      requestAnimationFrame(frame);
      return;
    }
    for (const item of items) {
      item.el.style.transition = "width 0.35s ease";
      item.el.style.width = `${Math.ceil(item.final)}px`;
    }
  };
  requestAnimationFrame(frame);
}

/** Reveal the latency breakdown once, in execution order. */
function initStageAnimation() {
  const root = document.querySelector<HTMLElement>("[data-stage-animation]");
  if (!root || reducedMotion) return;
  const rows = $all<HTMLElement>(".lane", root);
  root.classList.add("stage-ready");
  for (const count of $all<HTMLElement>("[data-stage-count]", root)) count.textContent = "0";
  const observer = new IntersectionObserver(entries => {
    if (!entries.some(entry => entry.isIntersecting)) return;
    observer.disconnect();
    void document.fonts.ready.then(() => {
      const start = performance.now();
      const duration = 420;
      const stageEnd = Math.max(...rows.map(row => row.querySelectorAll(".block").length)) * duration;
      const ease = (x: number) => 1 - Math.pow(1 - Math.max(0, Math.min(1, x)), 3);
      const animate = (now: number) => {
        const elapsed = now - start;
        for (const row of rows) {
          $all<HTMLElement>(".block", row).forEach((block, index) => {
            block.style.transform = `scaleX(${ease((elapsed - index * duration) / duration)})`;
          });
          const count = row.querySelector<HTMLElement>("[data-stage-count]")!;
          count.textContent = String(Math.round(Number(count.dataset.stageCount) * ease(elapsed / stageEnd)));
        }
        $all<HTMLElement>(".batch-layers i", root).forEach((layer, index) => {
          layer.style.opacity = String((.7 - index * .2) * ease((elapsed - stageEnd - index * 120) / 180));
        });
        const textProgress = ease((elapsed - stageEnd - 450) / 250);
        for (const label of $all<HTMLElement>(".stage-block-label", root)) label.style.opacity = String(textProgress);
        if (elapsed < stageEnd + 700) requestAnimationFrame(animate);
      };
      requestAnimationFrame(animate);
    });
  }, { threshold: .45 });
  observer.observe(root);
}

/**
 * Fig. 2: the left panel walks through one frame (observe, prefill, shared KV,
 * fan-out); each loop the right panel's window slides to the next frame and
 * shows which requests decode together. Both share one clock.
 */
function initDesign() {
  const root = document.querySelector<HTMLElement>("[data-design]");
  if (!root) return;
  const frames = Number(root.dataset.frames);
  const batch = Number(root.dataset.batch);
  const cycle = 2600;
  root.style.setProperty("--cycle", `${cycle}ms`);
  const label = root.querySelector<SVGTextElement>("[data-frame-label]");
  const slider = root.querySelector<SVGGElement>("[data-slider]");
  const batchLabel = root.querySelector<SVGTextElement>("[data-batch-label]");
  const cells = $all<SVGElement>(".dg-cell", root);
  const done = $all<SVGElement>("[data-completed-frame]", root);
  const col = Number(slider?.dataset.col ?? 0);
  let frame = 0;
  const show = (f: number) => {
    if (label) label.textContent = f === 0 ? "frame t" : `frame t+${f}`;
    slider?.style.setProperty("--x", `${f * col}px`);
    if (batchLabel) batchLabel.textContent = `batch of ${Math.min(f + 1, batch)}`;
    for (const cell of cells) cell.classList.toggle("future", Number(cell.dataset.f) > f);
    for (const mark of done) {
      const finishedFrame = Number(mark.dataset.completedFrame);
      mark.classList.toggle("completed", finishedFrame < f);
      mark.classList.toggle("finishing", finishedFrame === f);
    }
  };
  const clock = root.querySelector<SVGElement>(".ring-1");
  let playing = false;
  const step = () => {
    frame += 1;
    if (frame >= frames) {
      // Fade the window out, rewind while hidden, then fade back in.
      slider?.classList.add("reset");
      window.setTimeout(() => {
        frame = 0;
        show(frame);
        slider?.classList.remove("reset");
      }, 400);
      return;
    }
    show(frame);
  };
  // The left panel's CSS animation is the clock: each loop advances one frame.
  clock?.addEventListener("animationiteration", step);
  const observer = new IntersectionObserver((entries) => {
    const visible = entries.some((e) => e.isIntersecting);
    if (visible && !playing) {
      playing = true;
      root.classList.add("playing");
      show(frame);
    } else if (!visible && playing) {
      playing = false;
      root.classList.remove("playing");
    }
  });
  observer.observe(root);
}

function initDemo() {
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
  let wantsPlay = false;
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
      const activeLanguage = segment?.kind === "language";
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
    if (!replay || video.readyState < 1) return;
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
      root.querySelector<HTMLElement>(`[data-demo-output-popover="${side}"]`)!.hidden = true;
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
  for (const side of sides) {
    const button = root.querySelector<HTMLButtonElement>(`[data-demo-output="${side}"]`)!;
    const popover = root.querySelector<HTMLElement>(`[data-demo-output-popover="${side}"]`)!;
    button.addEventListener("click", () => { if (!button.disabled) popover.hidden = !popover.hidden; });
  }
  const detailsButton = root.querySelector<HTMLButtonElement>("[data-demo-details]")!;
  const detailsPopover = root.querySelector<HTMLElement>("[data-demo-details-popover]")!;
  detailsButton.addEventListener("click", () => { detailsPopover.hidden = !detailsPopover.hidden; detailsButton.setAttribute("aria-expanded", String(!detailsPopover.hidden)); });
  document.addEventListener("click", event => {
    if (!(event.target instanceof Node)) return;
    const insideRoot = root.contains(event.target);
    for (const side of sides) {
      const button = root.querySelector<HTMLButtonElement>(`[data-demo-output="${side}"]`)!;
      const popover = root.querySelector<HTMLElement>(`[data-demo-output-popover="${side}"]`)!;
      if (!insideRoot || (event.target !== button && !popover.contains(event.target))) popover.hidden = true;
    }
    if (!insideRoot || (event.target !== detailsButton && !detailsPopover.contains(event.target))) {
      detailsPopover.hidden = true;
      detailsButton.setAttribute("aria-expanded", "false");
    }
  });
  for (const [index, tab] of tabs.entries()) {
    tab.addEventListener("click", () => {
      if (tab.dataset.id === taskId) return;
      wantsPlay = !video.paused;
      select(tabs, tab);
      for (const item of tabs) item.tabIndex = item === tab ? 0 : -1;
      void load(tab);
    });
    tab.addEventListener("keydown", event => {
      let next = index;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      else if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = tabs.length - 1;
      else return;
      event.preventDefault(); tabs[next].focus(); tabs[next].click();
    });
  }
  void load(tabs[0]);
  if (!reducedMotion) {
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) {
        wantsPlay = true; start(); observer.disconnect();
      }
    }, { threshold: 0.6 });
    observer.observe(video);
  }
}

/** Draws a chart panel's lines once. */
function draw(panel: Element) {
  if (reducedMotion) return;
  panel.classList.remove("drawn");
  void (panel as HTMLElement).getBoundingClientRect();
  panel.classList.add("drawn");
}

function drawOnFirstView(root: HTMLElement, targets: () => Element[]) {
  const observer = new IntersectionObserver(
    (entries) => {
      if (!entries.some((entry) => entry.isIntersecting)) return;
      for (const target of targets()) draw(target);
      observer.disconnect();
    },
    { threshold: 0.35 },
  );
  observer.observe(root);
}

function initTradeoff() {
  const root = document.querySelector<HTMLElement>("[data-tradeoff]");
  if (!root) return;
  const buttons = $all<HTMLButtonElement>("[data-device]", root);
  const panels = () => $all<HTMLElement>("[data-panel]:not(.off)", root);
  for (const button of buttons) {
    button.addEventListener("click", () => {
      select(buttons, button);
      root.querySelector("[data-chart-gain]")!.textContent = button.dataset.gain!;
      for (const panel of $all<HTMLElement>("[data-panel]", root)) {
        panel.classList.toggle("off", panel.dataset.panel !== button.dataset.device);
      }
      for (const panel of panels()) draw(panel);
    });
  }
  drawOnFirstView(root, panels);
}

function initAblation() {
  const root = document.querySelector<HTMLElement>("[data-ablation]");
  if (!root) return;
  const buttons = $all<HTMLButtonElement>("[data-ablation-device]", root);
  const panels = () => $all<HTMLElement>("[data-ablation-panel]:not(.off)", root);
  for (const button of buttons) button.addEventListener("click", () => {
    select(buttons, button);
    root.querySelector("[data-chart-gain]")!.textContent = button.dataset.gain!;
    for (const panel of $all<HTMLElement>("[data-ablation-panel]", root)) panel.classList.toggle("off", panel.dataset.ablationPanel !== button.dataset.ablationDevice);
    for (const panel of panels()) draw(panel);
  });
  drawOnFirstView(root, panels);
}

function initTooltip() {
  const tip = document.querySelector<HTMLElement>("#tooltip");
  if (!tip) return;
  document.addEventListener("pointerover", (event) => {
    const target = (event.target as Element).closest<SVGElement>("[data-tip]");
    if (!target) return;
    tip.textContent = target.dataset.tip ?? "";
    const box = target.getBoundingClientRect();
    tip.style.left = `${box.left + box.width / 2}px`;
    tip.style.top = `${box.top}px`;
    tip.classList.add("show");
  });
  document.addEventListener("pointerout", (event) => {
    if ((event.target as Element).closest("[data-tip]")) tip.classList.remove("show");
  });
  window.addEventListener("scroll", () => tip.classList.remove("show"), { passive: true });
}

function initCitation() {
  const button = document.querySelector<HTMLButtonElement>("#copy-citation");
  const text = document.querySelector<HTMLElement>("#citation-text");
  button?.addEventListener("click", async () => {
    if (!text) return;
    await navigator.clipboard.writeText(text.textContent ?? "");
    const label = button.querySelector("span");
    if (!label) return;
    label.textContent = "Copied";
    button.classList.add("done");
    window.setTimeout(() => {
      label.textContent = "Copy BibTeX";
      button.classList.remove("done");
    }, 1600);
  });
}

/** Keeps SVG chart text at fixed pixel sizes however wide the chart is drawn. */
function initChartText() {
  const svgs = $all<SVGSVGElement>(".chart svg");
  const observer = new ResizeObserver(() => {
    for (const svg of svgs) {
      const width = svg.getBoundingClientRect().width;
      // Schematics keep their proportions: their labels sit inside drawn boxes.
      const scale = svg.viewBox.baseVal.width / width;
      if (width > 0) svg.style.setProperty("--s", String(svg.closest(".design") ? Math.min(scale, 1) : scale));
    }
  });
  for (const svg of svgs) observer.observe(svg);
}

export function initPage() {
  initChartText();
  initRunningHead();
  initTheme();
  void document.fonts.ready.then(initCountUp);
  initStageAnimation();
  initDesign();
  initDemo();
  initTradeoff();
  initAblation();
  initTooltip();
  initCitation();
}
