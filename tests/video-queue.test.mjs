import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import ts from 'typescript';

const { outputText } = ts.transpileModule(readFileSync(new URL('../src/scripts/video-queue.ts', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
});
const { createVideoQueue } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);
const turn = () => new Promise(resolve => setTimeout(resolve, 5));
async function until(check) {
  for (let i=0;i<100;i++) { if (check()) return; await turn(); }
  throw new Error('Video queue did not reach the expected state');
}
function harness(t, urls) {
  const previousFetch = globalThis.fetch;
  const previousWindow = globalThis.window;
  const window = new EventTarget();
  globalThis.window = window;
  const requests = [];
  globalThis.fetch = (url, options) => new Promise((resolve,reject) => {
    options.signal.addEventListener('abort', () => reject(new DOMException('Aborted','AbortError')), { once:true });
    requests.push({url, options, respond: (status=200) => resolve(new Response(new Uint8Array([1,2,3]), {status,headers:{'content-type':'video/mp4'}}))});
  });
  let release;
  const ready = new Promise(resolve => {release=resolve});
  const queue = createVideoQueue(urls, ready);
  t.after(() => {
    window.dispatchEvent(new Event('pagehide'));
    globalThis.fetch = previousFetch;
    if (previousWindow === undefined) delete globalThis.window;
    else globalThis.window = previousWindow;
  });
  return {queue, requests, release};
}

test('media waits for fonts and images, then selected task precedes background tasks', async t => {
  const h=harness(t,['other','selected']);
  const selected=h.queue.select('selected');
  await turn(); assert.equal(h.requests.length,0);
  h.release(); await until(()=>h.requests.length===1);
  assert.equal(h.requests[0].url,'selected');
  assert.equal(h.requests[0].options.priority,'low');
  h.requests[0].respond(); assert.match(await selected,/^blob:/);
  await until(()=>h.requests.length===2);
  assert.equal(h.requests[1].url,'other');
});

test('task switch interrupts background transfer and completed video is reused', async t => {
  const h=harness(t,['first','background','new-selection']);
  const first=h.queue.select('first');h.release();
  await until(()=>h.requests.length===1);h.requests[0].respond();
  const original=await first;
  await until(()=>h.requests.length===2);
  const switched=h.queue.select('new-selection');
  assert.equal(h.requests[1].options.signal.aborted,true);
  await until(()=>h.requests.length===3);
  assert.equal(h.requests[2].url,'new-selection');
  h.requests[2].respond();await switched;
  assert.equal(await h.queue.select('first'),original);
  assert.equal(h.requests.filter(r=>r.url==='first').length,1);
});

test('failed selected transfer reports error and deliberate re-selection retries', async t => {
  const h=harness(t,['first']);const first=h.queue.select('first');h.release();
  const failure=assert.rejects(first,/Video unavailable: 503/);
  await until(()=>h.requests.length===1);h.requests[0].respond(503);await failure;
  const retry=h.queue.select('first');await until(()=>h.requests.length===2);
  h.requests[1].respond();assert.match(await retry,/^blob:/);
});
