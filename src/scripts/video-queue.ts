/** One low-priority transfer at a time, after fonts and still images settle.
 * Blob URLs reuse the exact downloaded bytes for playback and scrubbing. A tab
 * change interrupts a background transfer and gives the selected task priority.
 */
export function createVideoQueue(urls: string[], ready: Promise<void>) {
  const cache = new Map<string, string>();
  const failed = new Set<string>();
  const waiting = new Map<string, { resolve: (value: string) => void; reject: (reason: unknown) => void }[]>();
  let selected = urls[0];
  let allowed = false;
  let disposed = false;
  let active: { url: string; controller: AbortController } | undefined;

  const pump = () => {
    if (!allowed || disposed || active) return;
    const url = [selected, ...urls].find(url => !cache.has(url) && !failed.has(url));
    if (!url) return;
    const controller = new AbortController();
    active = { url, controller };
    void fetch(url, { signal: controller.signal, priority: "low" }).then(async response => {
      if (!response.ok) throw new Error(`Video unavailable: ${response.status}`);
      const blob = await response.blob();
      if (disposed || controller.signal.aborted) return;
      const objectUrl = URL.createObjectURL(blob);
      cache.set(url, objectUrl);
      for (const waiter of waiting.get(url) ?? []) waiter.resolve(objectUrl);
      waiting.delete(url);
    }).catch((error: unknown) => {
      if (controller.signal.aborted) return;
      failed.add(url);
      for (const waiter of waiting.get(url) ?? []) waiter.reject(error);
      waiting.delete(url);
    }).finally(() => {
      active = undefined;
      // Let the selected player attach its source before starting background IO.
      setTimeout(pump, 0);
    });
  };
  void ready.then(() => { allowed = true; pump(); }).catch((error: unknown) => {
    for (const waiters of waiting.values()) for (const waiter of waiters) waiter.reject(error);
    waiting.clear();
  });

  window.addEventListener("pagehide", event => {
    if (event.persisted) return;
    disposed = true;
    active?.controller.abort();
    for (const objectUrl of cache.values()) URL.revokeObjectURL(objectUrl);
  });
  return {
    select(url: string): Promise<string> {
      selected = url;
      const cached = cache.get(url);
      if (cached) return Promise.resolve(cached);
      failed.delete(url); // A deliberate re-selection retries a failed transfer.
      if (active && active.url !== url) active.controller.abort();
      const promise = new Promise<string>((resolve, reject) => {
        const waiters = waiting.get(url) ?? [];
        waiters.push({ resolve, reject });
        waiting.set(url, waiters);
      });
      pump();
      return promise;
    },
  };
}
