export const links = {
  paper: "https://arxiv.org/abs/2603.14371",
  code: "https://github.com/air-embodied-brain/OxyGen",
  blog: "https://xxxxyu.github.io/blog/oxygen-mot-vla-inference/",
  model: "https://huggingface.co/xxxxyu/oxygen-pi05-textual-memory-lora",
  dataset: "https://huggingface.co/datasets/xxxxyu/libero-textual-memory-annotations",
};

// Kept as the arXiv entry until the NeurIPS proceedings are published.
export const citation = `@article{li2026oxygen,
  title={OxyGen: Unified KV Cache Management for VLA Inference under Multi-Task Parallelism},
  author={Li, Xiangyu and Tang, Huaizhi and Ding, Xin and Wang, Weijun and Cao, Ting and Liu, Yunxin},
  journal={arXiv preprint arXiv:2603.14371},
  year={2026}
}`;

/**
 * Side-by-side LIBERO-10 rollouts rendered by scripts/demo/render_comparison.py.
 * Durations are wall-clock task completion times from the logged runs: measured
 * model inference on one RTX 4090 plus simulation at the 20 Hz control rate.
 */
export const demos = [
  {
    id: "white-mug-and-chocolate-pudding",
    title: "White mug & chocolate pudding",
    task: "Put the white mug on the plate and put the chocolate pudding to the right of the plate.",
    oxygenS: 18.66,
    baselineS: 25.08,
  },
  {
    id: "alphabet-soup-and-tomato-sauce",
    title: "Alphabet soup & tomato sauce",
    task: "Put both the alphabet soup and the tomato sauce in the basket.",
    oxygenS: 23.46,
    baselineS: 29.99,
  },
  {
    id: "moka-pots",
    title: "Two moka pots",
    task: "Put both moka pots on the stove.",
    oxygenS: 36.87,
    baselineS: 47.61,
  },
];
