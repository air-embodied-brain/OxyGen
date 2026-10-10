// Camera-ready paper and adopted dodo first-run tables (2026-10-08).
// Rounded to paper precision; see DATA_SOURCES.md for evidence and hardware.
// Speedups are OxyGen frame-latency speedups over isolated execution.

export type SpeedupRow = { N: number; baselineMs: number; k1: number; k5: number; k10: number | null };

/** Appendix, tab:cross-model-results; full sweep from dodo xiaomi_full.csv. RTX 4090, S=5. */
export const xiaomi: SpeedupRow[] = [
  { N: 5, baselineMs: 171.8, k1: 2.05, k5: 1.09, k10: null },
  { N: 10, baselineMs: 249.9, k1: 2.91, k5: 1.49, k10: 0.96 },
  { N: 15, baselineMs: 327.9, k1: 3.7, k5: 1.94, k10: null },
  { N: 20, baselineMs: 406.3, k1: 4.47, k5: 2.39, k10: 1.47 },
  { N: 30, baselineMs: 562.8, k1: 5.83, k5: 3.23, k10: 2.03 },
];

/** Appendix, tab:cross-model-results; full sweep from dodo starvla_full.csv. E counts all experts; S=4. */
export const starvla: Record<2 | 3 | 4, SpeedupRow[]> = {
  2: [
    { N: 5, baselineMs: 144.8, k1: 2.22, k5: 1.23, k10: null },
    { N: 10, baselineMs: 211.8, k1: 3.2, k5: 1.71, k10: 1.13 },
    { N: 15, baselineMs: 278.6, k1: 4.11, k5: 2.22, k10: null },
    { N: 20, baselineMs: 344.9, k1: 5.0, k5: 2.73, k10: 1.73 },
    { N: 30, baselineMs: 478.0, k1: 6.64, k5: 3.6, k10: 2.35 },
  ],
  3: [
    { N: 5, baselineMs: 184.1, k1: 2.34, k5: 1.4, k10: null },
    { N: 10, baselineMs: 250.7, k1: 3.13, k5: 1.82, k10: 1.25 },
    { N: 15, baselineMs: 317.0, k1: 3.9, k5: 2.28, k10: null },
    { N: 20, baselineMs: 383.6, k1: 4.64, k5: 2.74, k10: 1.8 },
    { N: 30, baselineMs: 515.8, k1: 6.05, k5: 3.53, k10: 2.38 },
  ],
  4: [
    { N: 5, baselineMs: 209.4, k1: 2.66, k5: 1.59, k10: null },
    { N: 10, baselineMs: 276.3, k1: 3.44, k5: 2.01, k10: 1.37 },
    { N: 15, baselineMs: 342.4, k1: 4.2, k5: 2.45, k10: null },
    { N: 20, baselineMs: 409.1, k1: 4.91, k5: 2.91, k10: 1.92 },
    { N: 30, baselineMs: 541.2, k1: 6.3, k5: 3.69, k10: 2.49 },
  ],
};

export type ScalingRow = SpeedupRow & { backbone: string; tp: number };

/** Appendix, tab:model-scaling. Dodo single-GPU performance runs with random parameters. */
export const scaling: ScalingRow[] = [
  { backbone: "0.8B", tp: 1, N: 10, baselineMs: 196.4, k1: 2.91, k5: 1.65, k10: 1.12 },
  { backbone: "0.8B", tp: 1, N: 20, baselineMs: 313.6, k1: 4.62, k5: 2.62, k10: 1.7 },
  { backbone: "0.8B", tp: 1, N: 30, baselineMs: 429.0, k1: 6.3, k5: 3.58, k10: 2.32 },
  { backbone: "2B", tp: 1, N: 10, baselineMs: 211.3, k1: 2.99, k5: 1.71, k10: 1.14 },
  { backbone: "2B", tp: 1, N: 20, baselineMs: 331.2, k1: 4.65, k5: 2.65, k10: 1.73 },
  { backbone: "2B", tp: 1, N: 30, baselineMs: 453.3, k1: 6.31, k5: 3.62, k10: 2.34 },
  { backbone: "4B", tp: 1, N: 10, baselineMs: 274.0, k1: 3.03, k5: 1.7, k10: 1.12 },
  { backbone: "4B", tp: 1, N: 20, baselineMs: 437.9, k1: 4.82, k5: 2.72, k10: 1.76 },
  { backbone: "4B", tp: 1, N: 30, baselineMs: 602.9, k1: 6.55, k5: 3.71, k10: 2.39 },
  { backbone: "9B", tp: 1, N: 10, baselineMs: 316.7, k1: 3.22, k5: 1.68, k10: 1.13 },
  { backbone: "9B", tp: 1, N: 20, baselineMs: 523.6, k1: 5.23, k5: 2.73, k10: 1.72 },
  { backbone: "9B", tp: 1, N: 30, baselineMs: 731.9, k1: 7.18, k5: 3.79, k10: 2.39 },
];

/** Appendix, Tensor parallelism paragraph. Historical paired runs on a separate
 * eight-RTX-4090 server (PCIe, no NVLink); never compare against dodo TP=1. */
export const tensorParallel = [
  { tp: 1, N: 30, baselineMs: 1725.9, oxygenMs: 493.4, k1: 5.96, k5: 3.50 },
  { tp: 2, N: 30, baselineMs: 1983.5, oxygenMs: 535.3, k1: 6.52, k5: 3.71 },
];

/** Appendix, tab:quality. Success rate (%), OxyGen mean and 95% CI over 10 seeds. */
export const quality = [
  { suite: "Spatial", openpi: 98.8, mean: 98.42, ci: [98.12, 98.72] },
  { suite: "Object", openpi: 98.2, mean: 98.52, ci: [97.95, 99.09] },
  { suite: "Goal", openpi: 98.0, mean: 97.24, ci: [96.66, 97.82] },
  { suite: "LIBERO-10", openpi: 92.4, mean: 92.86, ci: [92.4, 93.32] },
  { suite: "Overall", openpi: 96.85, mean: 96.76, ci: [96.55, 96.97] },
] as const;

/** Appendix, tab:overhead. RTX 4090. */
export const overhead = [
  { setting: "Baseline", memGiB: 6.43, powerW: 293.5, energyJ: 117.4 },
  { setting: "Parallel", memGiB: 12.49, powerW: 324.4, energyJ: 120.9 },
  { setting: "Ours w/o batching", memGiB: 6.43, powerW: 287.9, energyJ: 97.7 },
  { setting: "Ours (batch 2)", memGiB: 7.35, powerW: 173.3, energyJ: 39.1 },
  { setting: "Ours (batch 4)", memGiB: 7.41, powerW: 154.4, energyJ: 25.8 },
];

/** Main text, fig:realrobot. Jetson AGX Thor on the humanoid, ms. */
export const robot = {
  baseline: { action: 207.5, language: 822.3 },
  oxygen: { action: 198.0, language: 195.4 },
  executionWindowMs: 333,
};

/** Appendix, tab:small-n-profile. Synchronized stage profile, π0.5-LIBERO, S=10, RTX 4090 (ms, rounded mean over N=1–5). */
export const stageProfile = { prefixMs: 44.0, denoiseMs: 20.4 };

/** Main text §4.5 and Appendix, tab:quality: evaluation protocol for the LIBERO success rates. */
export const qualityProtocol = { seeds: 10, rollouts: 20000 };

/**
 * Textual-memory adapter (post-paper application). Source: OxyGen repository
 * language_adaptation/libero/README.md ("Reference results") and the Hugging Face
 * model card xxxxyu/oxygen-pi05-textual-memory-lora. RTX 4090.
 */
export const memoryAdapter = {
  rank: 16,
  exactMatchPct: 87.25,
  adapterMs: 8.16,
  prefixForwardMs: 47.66,
  actionDiff: 0,
  annotatedFrames: 338575,
  demonstrations: 2000,
};
