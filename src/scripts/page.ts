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
  const col = Number(slider?.dataset.col ?? 0);
  let frame = 0;
  const show = (f: number) => {
    if (label) label.textContent = f === 0 ? "frame t" : `frame t+${f}`;
    slider?.style.setProperty("--x", `${f * col}px`);
    if (batchLabel) batchLabel.textContent = `batch of ${Math.min(f + 1, batch)}`;
    for (const cell of cells) cell.classList.toggle("future", Number(cell.dataset.f) > f);
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
  const video = root.querySelector<HTMLVideoElement>("[data-demo-video]")!;
  const tabs = $all<HTMLButtonElement>("[role=tab]", root);
  const portrait = window.matchMedia("(max-width: 680px)");
  const load = (tab: HTMLButtonElement) => {
    const name = `${tab.dataset.id}${portrait.matches ? "-vertical" : ""}`;
    video.poster = `${root.dataset.base}posters/${name}.webp`;
    video.querySelector("source")!.src = `${root.dataset.base}videos/${name}.mp4`;
    video.setAttribute("aria-label", `${tab.dataset.label}: OxyGen and the isolated baseline side by side`);
    video.load();

  };
  const current = () => tabs.find((tab) => tab.getAttribute("aria-selected") === "true") ?? tabs[0];
  if (portrait.matches) load(current());
  portrait.addEventListener("change", () => load(current()));
  for (const tab of tabs) {
    tab.addEventListener("click", () => {
      if (tab.getAttribute("aria-selected") === "true") return;
      select(tabs, tab);
      load(tab);
      void video.play().catch(() => undefined);
    });
  }
  if (reducedMotion) return;
  const observer = new IntersectionObserver(
    (entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        void video.play().catch(() => undefined);
        observer.disconnect();
      }
    },
    { threshold: 0.6 },
  );
  observer.observe(video);
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
  const buttons = $all<HTMLButtonElement>("[data-config]", root);
  const panels = () => $all<HTMLElement>("[data-panel]:not(.off)", root);
  for (const button of buttons) {
    button.addEventListener("click", () => {
      select(buttons, button);
      for (const panel of $all<HTMLElement>("[data-panel]", root)) {
        panel.classList.toggle("off", panel.dataset.panel !== button.dataset.config);
      }
      for (const panel of panels()) draw(panel);
    });
  }
  drawOnFirstView(root, panels);
}

function initAblation() {
  const root = document.querySelector<HTMLElement>("[data-ablation]");
  if (root) drawOnFirstView(root, () => $all(".chart svg", root));
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
  initCountUp();
  initDesign();
  initDemo();
  initTradeoff();
  initAblation();
  initTooltip();
  initCitation();
}
