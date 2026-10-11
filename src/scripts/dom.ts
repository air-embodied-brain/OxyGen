export function $all<T extends Element>(selector: string, root: ParentNode = document) {
  return Array.from(root.querySelectorAll<T>(selector));
}

export function select(buttons: HTMLElement[], active: HTMLElement, attr = "aria-selected") {
  for (const button of buttons) {
    button.setAttribute(attr, String(button === active));
    if (button.getAttribute("role") === "tab") button.tabIndex = button === active ? 0 : -1;
  }
}

/** Shared roving focus for task and device tabs. Click handlers own their content. */
export function initTabs(buttons: HTMLButtonElement[]) {
  for (const [index, button] of buttons.entries()) button.addEventListener("keydown", event => {
    let next = index;
    if (event.key === "ArrowRight") next = (index + 1) % buttons.length;
    else if (event.key === "ArrowLeft") next = (index + buttons.length - 1) % buttons.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = buttons.length - 1;
    else return;
    event.preventDefault(); buttons[next].focus(); buttons[next].click();
  });
}
