import { $all } from "./dom";
import { initCountUp, initStageAnimation, initDesign, initSchedule, initBarReveal } from "./motion";
import { initDemo } from "./demo";
import { initDeviceCharts, initTooltip, initChartText } from "./charts";
import { initPopovers } from "./popovers";

export function initRunningHead() {
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

export function initTheme() {
  document.querySelector("#theme-toggle")?.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("oxygen-theme", next);
  });
}

export function initCitation() {
  const button = document.querySelector<HTMLButtonElement>("#copy-citation");
  const text = document.querySelector<HTMLElement>("#citation-text");
  button?.addEventListener("click", async () => {
    if (!text) return;
    await navigator.clipboard.writeText(text.dataset.citation ?? text.textContent ?? "");
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

export function initPage() {
  initChartText();
  initRunningHead();
  initTheme();
  void document.fonts.ready.then(initCountUp);
  initStageAnimation();
  initSchedule();
  initDesign();
  initDemo();
  initDeviceCharts();
  initPopovers();
  initBarReveal();
  initTooltip();
  initCitation();
}
