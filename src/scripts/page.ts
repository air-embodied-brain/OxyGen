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

/** Sweeps a playhead across one second of measured actions and tokens, slowed down. */
function initSecond() {
  const root = document.querySelector<HTMLElement>("[data-second]");
  if (!root || reducedMotion) return;
  const period = Number(root.dataset.slowdown ?? 10) * 1000;
  const clock = root.querySelector<HTMLElement>("[data-clock]");
  const runs = $all<HTMLElement>(".run", root).map((run) => ({
    run,
    hz: Number(run.dataset.hz),
    tps: Number(run.dataset.tps),
    hzOut: run.querySelector<HTMLElement>('[data-count="hz"]'),
    tpsOut: run.querySelector<HTMLElement>('[data-count="tps"]'),
  }));
  const hold = 0.12; // fraction of each loop spent showing the full second
  const frame = (now: number) => {
    const t = (now % period) / period;
    const p = Math.min(1, t / (1 - hold));
    root.style.setProperty("--p", `${(p * 100).toFixed(2)}%`);
    if (clock) clock.textContent = p.toFixed(2);
    for (const r of runs) {
      if (r.hzOut) r.hzOut.textContent = String(Math.floor(r.hz * p + 1e-6));
      if (r.tpsOut) r.tpsOut.textContent = String(Math.floor(r.tps * p + 1e-6));
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
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
  if (!root) return;
  const buttons = $all<HTMLButtonElement>("[data-device]", root);
  for (const button of buttons) {
    button.addEventListener("click", () => {
      select(buttons, button);
      for (const panel of $all<HTMLElement>("[data-panel]", root)) {
        const on = panel.dataset.panel === button.dataset.device;
        panel.classList.toggle("off", !on);
        if (on) draw(panel);
      }
    });
  }
  drawOnFirstView(root, () => $all("[data-panel]:not(.off)", root));
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
    await navigator.clipboard.writeText(text.innerText);
    button.textContent = "Copied";
    window.setTimeout(() => (button.textContent = "Copy BibTeX"), 1600);
  });
}

/** Keeps SVG chart text at fixed pixel sizes however wide the chart is drawn. */
function initChartText() {
  const svgs = $all<SVGSVGElement>(".chart svg");
  const observer = new ResizeObserver(() => {
    for (const svg of svgs) {
      const width = svg.getBoundingClientRect().width;
      if (width > 0) svg.style.setProperty("--s", String(svg.viewBox.baseVal.width / width));
    }
  });
  for (const svg of svgs) observer.observe(svg);
}

export function initPage() {
  initChartText();
  initRunningHead();
  initTheme();
  initSecond();
  initDemo();
  initTradeoff();
  initAblation();
  initTooltip();
  initCitation();
}
