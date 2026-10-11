/** Visibility uses reachable pixel thresholds, including tall figures on short screens. */

export function observeVisibility(root: HTMLElement, update: (visible: boolean) => void) {
  let active = false;
  const check = () => {
    const rect = root.getBoundingClientRect();
    const pixels = Math.min(rect.bottom, innerHeight) - Math.max(rect.top, 0);
    const next = !document.hidden && (active ? pixels > 0 : pixels >= Math.min(rect.height * .6, innerHeight * .35));
    if (next !== active) { active = next; update(active); }
  };
  update(false);
  window.addEventListener("scroll", check, { passive: true });
  window.addEventListener("resize", check, { passive: true });
  document.addEventListener("visibilitychange", check);
  new ResizeObserver(check).observe(root);
  check();
}

export function observeAnimation(root: HTMLElement, play: () => void) {
  observeVisibility(root, visible => { if (visible) play(); });
}
