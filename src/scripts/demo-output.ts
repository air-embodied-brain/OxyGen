export type LanguageCall = {
  start: number;
  actionEnd: number;
  end: number;
  updates: { text: string; full: number; new: number; finished: boolean }[];
};

export function replayOutput(run: {
  calls: LanguageCall[];
  memory: { time: number; text: string }[];
}, time: number) {
  // Updates are recorded at call completion, not at call start. There are no
  // token timestamps with which to reconstruct progress inside a call.
  const batch = run.calls.filter(call => call.end <= time).at(-1);
  const latest = run.memory.filter(item => item.time <= time).at(-1);
  return { text: latest?.text ?? "Waiting for output", batch };
}
