// Inline at the end of the document: placeholders do not wait for a JS request.
(() => {
  const root = document.documentElement;
  const config = JSON.parse(document.getElementById("font-config").textContent);
  const placeholders = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = node.parentElement;
      return node.textContent.trim() && parent && !parent.closest("script,style,noscript,svg,textarea,option")
        ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
    },
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const node of nodes) {
    const critical = Boolean(node.parentElement.closest('[data-font-priority="critical"]'));
    const span = document.createElement("span");
    span.className = "font-placeholder";
    node.replaceWith(span);
    span.append(node);
    placeholders.push({ span, critical });
  }
  const svgPlaceholders = [];
  for (const text of document.querySelectorAll("svg text")) {
    text.setAttribute("data-svg-font-pending", "");
    const box = text.getBBox();
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    for (const [key, value] of Object.entries({ x: box.x, y: box.y, width: box.width, height: box.height, rx: 2 })) rect.setAttribute(key, String(value));
    rect.classList.add("svg-font-placeholder");
    rect.setAttribute("aria-hidden", "true");
    text.before(rect);
    svgPlaceholders.push({ text, rect, critical: Boolean(text.closest('[data-font-priority="critical"]')) });
  }
  root.dataset.fonts = "skeleton";
  const reveal = critical => {
    for (const item of placeholders) {
      if (item.critical === critical && item.span.isConnected) item.span.replaceWith(...item.span.childNodes);
    }
    for (const item of svgPlaceholders) {
      if (item.critical === critical) { item.text.removeAttribute("data-svg-font-pending"); item.rect.remove(); }
    }
  };
  const loadFonts = fonts => Promise.all(fonts.map(font => document.fonts.load(
    `${font.style} 400 16px ${JSON.stringify(font.family)}`, font.text,
  ).then(faces => {
    if (!faces.length) throw new Error(`Font face unavailable: ${font.family}`);
  })));
  const criticalReady = loadFonts(config.critical).then(() => {
    reveal(true);
    root.dataset.fonts = "critical";
  });
  window.oxygenCriticalFontsReady = criticalReady;
  const fontsReady = criticalReady.then(async () => {
    const style = document.createElement("style");
    style.textContent = config.deferredCss;
    document.head.append(style);
    await loadFonts(config.page);
    reveal(false);
    root.dataset.fonts = "ready";
  });
  window.oxygenFontsReady = fontsReady;

  // Explicitly include offscreen/lazy images and every task's poster. Otherwise
  // video would race the robot photo when the reader scrolls down to it.
  const images = [...document.images];
  const posters = new Set([...document.querySelectorAll("video[poster]")].map(video => video.poster));
  for (const demo of document.querySelectorAll("[data-demo]")) {
    for (const tab of demo.querySelectorAll("[data-id]")) posters.add(new URL(`${demo.dataset.base}posters/${tab.dataset.id}-interactive.webp`, location.href).href);
  }
  for (const poster of posters) { const image = new Image(); image.src = poster; images.push(image); }
  const imageReady = image => new Promise(resolve => {
    const done = () => resolve();
    image.addEventListener("load", done, { once: true });
    image.addEventListener("error", done, { once: true });
    image.loading = "eager";
    if (image.complete) done();
  }).then(() => image.decode().catch(() => {}));
  const imagesReady = Promise.all(images.map(imageReady));
  window.oxygenAssetsReady = Promise.all([fontsReady, imagesReady]).then(() => {
    root.dataset.assets = "ready";
    performance.mark("oxygen-assets-ready");
  });
  // Keep failed font text masked, rather than flashing a substitute font.
  fontsReady.catch(() => { root.dataset.fonts = "error"; });
  window.oxygenAssetsReady.catch(() => {});
})();
