import { $all } from "./dom";

export function initPopovers() {
  for (const panel of $all<HTMLElement>("[popover]")) {
    const trigger = document.querySelector<HTMLButtonElement>(`[popovertarget="${panel.id}"]`);
    if (!trigger) continue;
    const position = () => {
      if (!panel.hasAttribute("data-anchor-popover")) return;
      const rect = trigger.closest(".demo-output-box")!.getBoundingClientRect();
      const width = Math.min(Math.max(rect.width, 300), innerWidth - 24);
      const above = rect.top > innerHeight - rect.bottom;
      const space = Math.max(80, (above ? rect.top : innerHeight - rect.bottom) - 24);
      Object.assign(panel.style, {
        width: `${width}px`, maxHeight: `${space}px`,
        left: `${Math.max(12, Math.min(rect.left, innerWidth - width - 12))}px`,
        top: above ? "auto" : `${rect.bottom + 8}px`,
        bottom: above ? `${innerHeight - rect.top + 8}px` : "auto",
      });
    };
    panel.addEventListener("beforetoggle", event => {
      const open = (event as ToggleEvent).newState === "open";
      trigger.setAttribute("aria-expanded", String(open));
      if (open) position();
    });
    if (panel.hasAttribute("data-anchor-popover")) {
      window.addEventListener("resize", position);
      window.addEventListener("scroll", () => { if (panel.matches(":popover-open")) panel.hidePopover(); }, { passive: true });
    }
  }
}
