// Memory outputs from the demo rollouts, typed into the masthead band.
const MEMORIES = [
  "No relevant task progress is visible yet.",
  "The alphabet soup is in the basket.",
  "The white mug is on the plate.",
  "The first moka pot is on the stove.",
];

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

/** Types memory words at each row's measured token rate, slowed down for reading. */
function initRates() {
  const root = document.querySelector<HTMLElement>("[data-rates]");
  if (!root) return;
  const slowdown = Number(root.dataset.slowdown ?? 10);
  const words = MEMORIES.join(" ").split(" ");
  for (const [index, row] of $all<HTMLElement>(".rate-row", root).entries()) {
    const target = row.querySelector<HTMLElement>("[data-words]");
    if (!target) continue;
    if (reducedMotion) {
      target.textContent = MEMORIES[index];
      continue;
    }
    let cursor = 0;
    let text = "";
    window.setInterval(() => {
      text = `${text} ${words[cursor % words.length]}`.slice(-110);
      target.textContent = text.trim();
      cursor += 1;
    }, (1000 * slowdown) / Number(row.dataset.tps));
  }
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

function drawOnFirstView(root: HTMLElement, panel: () => Element | null) {
  const observer = new IntersectionObserver(
    (entries) => {
      if (!entries.some((entry) => entry.isIntersecting)) return;
      const target = panel();
      if (target) draw(target);
      observer.disconnect();
    },
    { threshold: 0.35 },
  );
  observer.observe(root);
}

function initTradeoff() {
  const root = document.querySelector<HTMLElement>("[data-tradeoff]");
  if (!root) return;
  const state = { config: "libero", device: "rtx4090" };
  const configButtons = $all<HTMLButtonElement>("[data-config]", root);
  const deviceButtons = $all<HTMLButtonElement>("[data-device]", root);
  const active = () => root.querySelector<HTMLElement>(`[data-panel="${state.config}-${state.device}"]`);
  const update = () => {
    for (const panel of $all<HTMLElement>("[data-panel]", root)) {
      panel.classList.toggle("off", panel !== active());
    }
    const panel = active();
    if (!panel) return;
    for (const out of $all<HTMLElement>("[data-out]", root)) {
      const key = out.dataset.out!.replace(/-([a-z])/g, (_, c: string) => c.toUpperCase());
      out.textContent = panel.dataset[key] ?? "";
    }
    draw(panel);
  };
  for (const button of configButtons) {
    button.addEventListener("click", () => {
      state.config = button.dataset.config!;
      select(configButtons, button);
      update();
    });
  }
  for (const button of deviceButtons) {
    button.addEventListener("click", () => {
      state.device = button.dataset.device!;
      select(deviceButtons, button);
      update();
    });
  }
  drawOnFirstView(root, active);
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
  drawOnFirstView(root, () => root.querySelector("[data-panel]:not(.off)"));
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

export function initPage() {
  initRunningHead();
  initTheme();
  initRates();
  initDemo();
  initTradeoff();
  initAblation();
  initTooltip();
  initCitation();
}
