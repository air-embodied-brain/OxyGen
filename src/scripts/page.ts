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

/** Counts the headline numbers up from zero once, on first load. */
function initCountUp() {
  const targets = $all<HTMLElement>("[data-count-up]");
  if (reducedMotion || !targets.length) return;
  const items = targets.map((el) => {
    const text = el.textContent ?? "0";
    return { el, text, value: Number(text), decimals: (text.split(".")[1] ?? "").length };
  });
  const duration = 1400;
  const start = performance.now();
  const frame = (now: number) => {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    for (const item of items) {
      item.el.textContent = t < 1 ? (item.value * eased).toFixed(item.decimals) : item.text;
    }
    if (t < 1) requestAnimationFrame(frame);
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
  initDemo();
  initTradeoff();
  initAblation();
  initTooltip();
  initCitation();
}
