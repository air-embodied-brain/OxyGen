import { $all } from "./dom";
import { observeAnimation, observeVisibility } from "./visibility";
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export function initCountUp() {
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
    return { el, text, value, decimals, final, widest };
  });
  let raf = 0;
  observeAnimation(targets[0].closest<HTMLElement>(".performance-groups")!, () => {
    cancelAnimationFrame(raf);
    for (const item of items) {
      item.el.style.transition = "none";
      item.el.style.width = `${Math.ceil(item.widest)}px`;
    }
    const start = performance.now();
    const frame = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      for (const item of items) {
        item.el.textContent = t < 1 ? (item.value * ease(t)).toFixed(item.decimals) : item.text;
      }
      if (t < 1) {
        raf = requestAnimationFrame(frame);
        return;
      }
      for (const item of items) {
        item.el.style.transition = "width 0.35s ease";
        item.el.style.width = `${Math.ceil(item.final)}px`;
      }
    };
    raf = requestAnimationFrame(frame);
  });
}

/** Reveal the latency breakdown on entry, in execution order. */

export function initStageAnimation() {
  const root = document.querySelector<HTMLElement>("[data-stage-animation]");
  if (!root) return;
  const rows = $all<HTMLElement>(".lane", root);
  root.classList.add("stage-ready");
  for (const count of $all<HTMLElement>("[data-stage-count]", root)) count.textContent = "0";
  let raf = 0;
  observeAnimation(root, () => {
    cancelAnimationFrame(raf);
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
      if (elapsed < stageEnd + 700) raf = requestAnimationFrame(animate);
    };
    raf = requestAnimationFrame(animate);

  });
}

/**
 * The method illustration: the left panel walks through one frame (observe, prefill, shared KV,
 * fan-out); each loop the right panel's window slides to the next frame and
 * shows which requests decode together. Both share one clock.
 */

export function initDesign() {
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
  observeVisibility(root, visible => {
    root.classList.add("playing");
    root.classList.toggle("motion-paused", !visible);
    show(frame);
  });
}

export function initSchedule() {
  const root = document.querySelector<HTMLElement>(".schedule-demo");
  if (root) observeVisibility(root, visible => root.classList.toggle("motion-paused", !visible));
}

export function initBarReveal() {
  for (const root of $all<HTMLElement>("[data-bar-reveal]")) {
    root.classList.add("reveal-ready");
    const counts = $all<HTMLElement>("[data-reveal-count]", root);
    let frame = 0;
    observeAnimation(root, () => {
      root.classList.add("reveal-ready");
      cancelAnimationFrame(frame);
      root.classList.remove("reveal-playing");
      void root.getBoundingClientRect();
      root.classList.add("reveal-playing");
      const start = performance.now();
      const tick = (now: number) => {
        const progress = Math.min((now - start) / (reducedMotion ? 1500 : 2700), 1);
        for (const el of counts) el.textContent = String(Math.round(Number(el.dataset.revealCount) * (1 - (1 - progress) ** 3)));
        if (progress < 1) frame = requestAnimationFrame(tick);
      };
      frame = requestAnimationFrame(tick);
    });
  }
}
