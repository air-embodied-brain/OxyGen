// Values copied from the camera-ready paper (NeurIPS-2026---OxyGen, tex/).
// Speedups are OxyGen frame-latency speedups over isolated execution.

export type SpeedupRow = { N: number; baselineMs: number; k1: number; k5: number; k10: number | null };

/** Appendix, tab:xiaomi-full. One RTX 4090, S=5. */
export const xiaomi: SpeedupRow[] = [
  { N: 5, baselineMs: 469.4, k1: 1.89, k5: 1.03, k10: null },
  { N: 10, baselineMs: 681.1, k1: 2.61, k5: 1.4, k10: 0.92 },
  { N: 15, baselineMs: 911.3, k1: 3.52, k5: 1.87, k10: null },
  { N: 20, baselineMs: 1118.6, k1: 4.16, k5: 2.31, k10: 1.46 },
  { N: 30, baselineMs: 1540.9, k1: 5.5, k5: 3.14, k10: 1.98 },
];

/** Appendix, tab:starvla-full. Shared Qwen3-VL-4B prefix; E counts all experts. */
export const starvla: Record<2 | 3 | 4, SpeedupRow[]> = {
  2: [
    { N: 5, baselineMs: 422.4, k1: 1.87, k5: 1.07, k10: null },
    { N: 10, baselineMs: 603.8, k1: 2.75, k5: 1.51, k10: 1.0 },
    { N: 15, baselineMs: 790.7, k1: 3.71, k5: 1.99, k10: null },
    { N: 20, baselineMs: 962.3, k1: 4.44, k5: 2.47, k10: 1.57 },
    { N: 30, baselineMs: 1351.5, k1: 6.18, k5: 3.41, k10: 2.18 },
  ],
  3: [
    { N: 5, baselineMs: 523.1, k1: 2.0, k5: 1.21, k10: null },
    { N: 10, baselineMs: 724.9, k1: 2.79, k5: 1.67, k10: 1.12 },
    { N: 15, baselineMs: 906.7, k1: 3.38, k5: 2.05, k10: null },
    { N: 20, baselineMs: 1092.1, k1: 4.2, k5: 2.48, k10: 1.67 },
    { N: 30, baselineMs: 1449.0, k1: 5.53, k5: 3.32, k10: 2.19 },
  ],
  4: [
    { N: 5, baselineMs: 599.2, k1: 2.24, k5: 1.36, k10: null },
    { N: 10, baselineMs: 776.4, k1: 2.94, k5: 1.74, k10: 1.17 },
    { N: 15, baselineMs: 985.7, k1: 3.66, k5: 2.21, k10: null },
    { N: 20, baselineMs: 1144.5, k1: 4.27, k5: 2.6, k10: 1.76 },
    { N: 30, baselineMs: 1541.3, k1: 5.83, k5: 3.48, k10: 2.34 },
  ],
};

export type ScalingRow = { backbone: string; tp: number; N: number; baselineMs: number; k1: number; k5: number; k10: number };

/** Appendix, tab:model-scaling. Qwen3.5 backbones via StarVLA; randomly initialized action heads. */
export const scaling: ScalingRow[] = [
  { backbone: "0.8B", tp: 1, N: 10, baselineMs: 563.4, k1: 2.71, k5: 1.59, k10: 1.1 },
  { backbone: "0.8B", tp: 1, N: 20, baselineMs: 901.9, k1: 4.24, k5: 2.53, k10: 1.67 },
  { backbone: "0.8B", tp: 1, N: 30, baselineMs: 1257.5, k1: 5.94, k5: 3.53, k10: 2.34 },
  { backbone: "2B", tp: 1, N: 10, baselineMs: 605.9, k1: 2.76, k5: 1.65, k10: 1.14 },
  { backbone: "2B", tp: 1, N: 20, baselineMs: 936.1, k1: 4.24, k5: 2.51, k10: 1.71 },
  { backbone: "2B", tp: 1, N: 30, baselineMs: 1267.5, k1: 5.73, k5: 3.44, k10: 2.29 },
  { backbone: "4B", tp: 1, N: 10, baselineMs: 779.8, k1: 2.74, k5: 1.61, k10: 1.1 },
  { backbone: "4B", tp: 1, N: 20, baselineMs: 1246.3, k1: 4.32, k5: 2.64, k10: 1.73 },
  { backbone: "4B", tp: 1, N: 30, baselineMs: 1728.0, k1: 6.07, k5: 3.6, k10: 2.4 },
  { backbone: "9B", tp: 1, N: 10, baselineMs: 798.1, k1: 2.82, k5: 1.63, k10: 1.12 },
  { backbone: "9B", tp: 1, N: 20, baselineMs: 1261.4, k1: 4.33, k5: 2.58, k10: 1.72 },
  { backbone: "9B", tp: 1, N: 30, baselineMs: 1725.9, k1: 5.96, k5: 3.5, k10: 2.3 },
  { backbone: "9B", tp: 2, N: 10, baselineMs: 905.7, k1: 2.93, k5: 1.69, k10: 1.12 },
  { backbone: "9B", tp: 2, N: 20, baselineMs: 1435.0, k1: 4.72, k5: 2.68, k10: 1.74 },
  { backbone: "9B", tp: 2, N: 30, baselineMs: 1983.5, k1: 6.52, k5: 3.71, k10: 2.38 },
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
export const stageProfile = { prefixMs: 45.1, denoiseMs: 20.4 };

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
