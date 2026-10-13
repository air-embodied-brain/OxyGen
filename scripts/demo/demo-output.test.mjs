import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { replayOutput } from '../../src/scripts/demo-output.ts';

const update = text => ({ text, full: 1, new: 1, finished: false });
const run = {
  calls: [
    { start: 0, actionEnd: 0.4, end: 1, updates: [update('partial')] },
    { start: 2, actionEnd: 2.4, end: 3, updates: [update('complete')] },
    { start: 4, actionEnd: 4.4, end: 5, updates: [update('future')] },
  ],
  memory: [{ time: 3, text: 'complete' }],
};
assert.deepEqual(replayOutput(run, 0), { text: 'Waiting for output', batch: undefined });
assert.equal(replayOutput(run, 0.999).batch, undefined);
assert.equal(replayOutput(run, 1).batch.updates[0].text, 'partial');
assert.equal(replayOutput(run, 1).text, 'Waiting for output');
assert.equal(replayOutput(run, 2.999).batch.updates[0].text, 'partial');
assert.equal(replayOutput(run, 3).text, 'complete');
assert.equal(replayOutput(run, 4.999).batch.updates[0].text, 'complete');
assert.equal(replayOutput(run, 5).batch.updates[0].text, 'future');
assert.equal(replayOutput(run, 0).batch, undefined); // backward seek

const data = JSON.parse(readFileSync(new URL('../../public/data/demo-timelines.json', import.meta.url)));
let boundaries = 0;
for (const replay of Object.values(data)) for (const side of ['oxygen', 'baseline']) {
  const run = replay[side];
  for (const call of run.calls) for (const time of [call.start, call.end - 1e-7, call.end, call.end + 1e-7]) {
    const result = replayOutput(run, time);
    assert.ok(!result.batch || result.batch.end <= time);
    assert.equal(result.batch, run.calls.findLast(c => c.end <= time));
    assert.equal(result.text, run.memory.findLast(m => m.time <= time)?.text ?? 'Waiting for output');
    boundaries++;
  }
}
console.log(`Passed fixture and ${boundaries} recorded call-boundary checks across all three tasks.`);
