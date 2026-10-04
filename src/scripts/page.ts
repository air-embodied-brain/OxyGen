const MEMORIES = [
  "No relevant task progress is visible yet.",
  "The alphabet soup is in the basket.",
  "The white mug is on the plate.",
  "The first moka pot is on the stove.",
];

function $all<T extends Element>(selector: string, root: ParentNode = document) {
  return Array.from(root.querySelectorAll<T>(selector));
}

function setSelected(buttons: HTMLElement[], active: HTMLElement, attr = "aria-selected") {
  for (const button of buttons) button.setAttribute(attr, String(button === active));
}

function initHeader() {
  const header = document.querySelector<HTMLElement>("[data-header]");
  const onScroll = () => header?.classList.toggle("scrolled", window.scrollY > 12);
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });

  const links = $all<HTMLAnchorElement>(".site-nav a");
  const sections = links
    .map((link) => document.querySelector<HTMLElement>(link.getAttribute("href")!))
    .filter((section): section is HTMLElement => Boolean(section));
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        for (const link of links) link.classList.toggle("active", link.getAttribute("href") === `#${entry.target.id}`);
      }
    },
    { rootMargin: "-45% 0px -50% 0px" },
  );
  for (const section of sections) observer.observe(section);
}

function initTheme() {
  document.querySelector("#theme-toggle")?.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("oxygen-theme", next);
  });
}

function initReveal() {
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        entry.target.classList.add("in-view");
        observer.unobserve(entry.target);
      }
    },
    { rootMargin: "0px 0px -12% 0px", threshold: 0.05 },
  );
  for (const element of $all(".reveal")) observer.observe(element);
}

/** Types memory text into each hero stream at its measured token rate, slowed for legibility. */
function initClocks() {
  const root = document.querySelector<HTMLElement>("[data-clocks]");
  if (!root) return;
  const slowdown = Number(root.dataset.slowdown ?? 10);
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  for (const [index, clock] of $all<HTMLElement>("[data-clock]", root).entries()) {
    const target = clock.querySelector<HTMLElement>("[data-stream]");
    if (!target) continue;
    const words = MEMORIES.join(" ").split(" ");
    if (reduced) {
      target.textContent = MEMORIES[index];
      continue;
    }
    const interval = (1000 * slowdown) / Number(clock.dataset.tps);
    let cursor = 0;
    let text = "";
    window.setInterval(() => {
      text = `${text} ${words[cursor % words.length]}`.slice(-90);
      target.textContent = text.trim();
      cursor += 1;
    }, interval);
  }
}

function initDemo() {
  const root = document.querySelector<HTMLElement>("[data-demo]");
  if (!root) return;
  const video = root.querySelector<HTMLVideoElement>("[data-demo-video]")!;
  const tabs = $all<HTMLButtonElement>(".demo-tab", root);
  const portrait = window.matchMedia("(max-width: 720px)");
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
      setSelected(tabs, tab);
      load(tab);
      void video.play().catch(() => undefined);
    });
  }
  // Start muted playback once the demo scrolls into view.
  const observer = new IntersectionObserver(
    (entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        void video.play().catch(() => undefined);
        observer.disconnect();
      }
    },
    { threshold: 0.5 },
  );
  if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) observer.observe(video);
}

function replay(panel: Element) {
  panel.classList.remove("draw");
  void (panel as HTMLElement).getBoundingClientRect();
  panel.classList.add("draw");
}

function initTradeoff() {
  const root = document.querySelector<HTMLElement>("[data-tradeoff]");
  if (!root) return;
  const state = { config: "libero", device: "rtx4090" };
  const configButtons = $all<HTMLButtonElement>("[data-config]", root);
  const deviceButtons = $all<HTMLButtonElement>("[data-device]", root);
  const update = () => {
    let active: HTMLElement | undefined;
    for (const panel of $all<HTMLElement>("[data-panel]", root)) {
      const on = panel.dataset.panel === `${state.config}-${state.device}`;
      panel.classList.toggle("off", !on);
      if (on) active = panel;
    }
    if (!active) return;
    for (const out of $all<HTMLElement>("[data-out]", root)) {
      const key = out.dataset.out!.replace(/-([a-z])/g, (_, c: string) => c.toUpperCase());
      out.textContent = active.dataset[key] ?? "";
    }
    replay(active);
  };
  for (const button of configButtons) {
    button.addEventListener("click", () => {
      state.config = button.dataset.config!;
      setSelected(configButtons, button);
      update();
    });
  }
  for (const button of deviceButtons) {
    button.addEventListener("click", () => {
      state.device = button.dataset.device!;
      setSelected(deviceButtons, button);
      update();
    });
  }
  const observer = new IntersectionObserver((entries) => {
    if (entries.some((entry) => entry.isIntersecting)) {
      update();
      observer.disconnect();
    }
  }, { threshold: 0.3 });
  observer.observe(root);
}

function initAblation() {
  const root = document.querySelector<HTMLElement>("[data-ablation]");
  if (!root) return;
  const buttons = $all<HTMLButtonElement>("[data-device]", root);
  for (const button of buttons) {
    button.addEventListener("click", () => {
      setSelected(buttons, button);
      for (const panel of $all<HTMLElement>("[data-panel]", root)) {
        const on = panel.dataset.panel === button.dataset.device;
        panel.classList.toggle("off", !on);
        if (on) replay(panel);
      }
    });
  }
}

function initGenerality() {
  const root = document.querySelector<HTMLElement>("[data-generality]");
  if (!root) return;
  const tabButtons = $all<HTMLButtonElement>("[data-tab]", root);
  const kButtons = $all<HTMLButtonElement>("[data-k]", root);
  let k = "k5";
  const applyK = () => {
    for (const bar of $all<HTMLElement>(".bar-fill", root)) {
      bar.style.setProperty("--v", bar.dataset[k]!);
      bar.textContent = bar.dataset[`${k}Label`] ?? "";
    }
  };
  for (const button of tabButtons) {
    button.addEventListener("click", () => {
      setSelected(tabButtons, button);
      for (const panel of $all<HTMLElement>("[data-tab-panel]", root)) {
        const on = panel.dataset.tabPanel === button.dataset.tab;
        panel.hidden = !on;
        if (on) {
          panel.classList.remove("in-view");
          void panel.offsetWidth;
          panel.classList.add("in-view");
        }
      }
    });
  }
  for (const button of kButtons) {
    button.addEventListener("click", () => {
      k = button.dataset.k!;
      setSelected(kButtons, button, "aria-pressed");
      applyK();
    });
  }
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
  initHeader();
  initTheme();
  initReveal();
  initClocks();
  initDemo();
  initTradeoff();
  initAblation();
  initGenerality();
  initTooltip();
  initCitation();
}
