import { $all, select, initTabs } from "./dom";
import { observeAnimation } from "./visibility";

export function draw(panel: Element) {
  panel.classList.remove("drawn");
  void (panel as HTMLElement).getBoundingClientRect();
  panel.classList.add("drawn");
}

export function replayOnView(root: HTMLElement, targets: () => Element[]) {
  const replay = () => { for (const target of targets()) draw(target); };
  observeAnimation(root, replay);
  return replay;
}

export function initDeviceCharts() {
  for (const root of $all<HTMLElement>("[data-device-chart]")) {
    const buttons = $all<HTMLButtonElement>("[data-device]", root);
    const panels = () => $all<HTMLElement>("[data-chart-panel]:not(.off)", root);
    const replay = replayOnView(root, panels);
    for (const button of buttons) button.addEventListener("click", () => {
      select(buttons, button);
      root.querySelector("[data-chart-gain]")!.textContent = button.dataset.gain!;
      for (const panel of $all<HTMLElement>("[data-chart-panel]", root)) {
        panel.classList.toggle("off", panel.dataset.chartPanel !== button.dataset.device);
      }
      replay();
    });
    initTabs(buttons);
  }
}

export function initTooltip() {
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

export function initChartText() {
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
