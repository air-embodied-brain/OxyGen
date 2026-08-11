# LIBERO language adapter：第一轮三组对照

日期：2026-08-11

## 结论

suffix-only LoRA 方案可行，但需要让训练路径与真实增量解码严格一致。标准的
2,000-step suffix-LoRA 已能在 teacher forcing 下学会标注；经过 500-step 增量路径
continuation 后，held-out token accuracy 为 98.47%，80 个跨 40 任务样本的真实逐
token greedy exact match 为 83.75%，Word F1 为 0.916。作为对照，原 LIBERO
checkpoint 的文本几乎不可读，token accuracy 只有 0.75%。

这套 adapter 不改变公共 prefix 或动作路径。20 个 held-out observations 上，root KV
和固定噪声的完整 10-step action tensor 与零 adapter 均逐 bit 相同。实际 OxyGen
continuous-batching 请求也能同时返回 `(10, 32)` action chunk 和正确的增量文本。

运行时也有足够余量：一个典型请求的 adapter 增量开销为 5.61 ms，而复用 root KV
避免的重复 prefix forward 为 50.15 ms，后者约为前者的 8.94 倍。换言之，在当前
请求长度下，adapter 不会抵消 prefix sharing 的收益。

## 实验设置

三组实验使用同一个 `pi05_libero` checkpoint、数据划分、rank 和训练预算：

| 配置 | adapter 位置 | 可训练参数 | 对动作 root 的影响 |
|---|---|---:|---|
| Suffix LoRA | 18 层语言 attention Q/K/V/O 与 FFN gating/up/down，仅 suffix 启用 | 27,869,184 | 无 |
| Full-sequence LoRA | 相同 LoRA 模块，prefix 与 suffix 均启用 | 27,869,184 | 有 |
| Final MLP | 最后一层语言 hidden state 后的 residual bottleneck | 65,536 | 无 |

LoRA rank 和 alpha 均为 16。标准训练为 batch size 2、2,000 steps、100 warmup、
AdamW、cosine decay、gradient clipping 1.0，分别测试 `1e-4` 和 `3e-4`。数据覆盖四个
标准 LIBERO suite 的全部 40 个任务；每个任务 45 个训练 episode、5 个验证 episode，
共 45,935 / 5,075 个采样帧。公共 prefix 使用官方 LIBERO observation/state/task
prompt 处理；监督 suffix 为 `Subtask: <predicate-derived next label> EOS`。

## 标准三组对照

下表使用每组较好的学习率。Token accuracy 在全部 5,075 个 held-out frames、42,281
个 target tokens 上计算；greedy 指标使用相同的 80 个跨任务样本，并通过真实
block-plus-token 增量路径生成。

| 配置 | LR | Token acc. | Greedy exact | Word F1 | Root / action exact | Median adapter overhead |
|---|---:|---:|---:|---:|---|---:|
| Suffix LoRA | `3e-4` | 98.57% | 51.25% | 0.814 | yes / yes | 5.98 ms |
| Full-sequence LoRA | `3e-4` | 98.33% | 40.00% | 0.654 | no / no | 6.14 ms |
| Final MLP | `3e-4` | 66.80% | 3.75% | 0.283 | yes / yes | below timing noise |
| Base LIBERO checkpoint | - | 0.75% | 0.00% | 0.000 | - | - |

这组结果给出两个直接判断。第一，外挂 final MLP 很便宜，但容量明显不足。第二，
full-sequence LoRA 提供了更大的修改自由度，却没有在固定预算下优于 suffix LoRA，
并且会改变 action 所读取的 root KV。它是容量对照，不是性能一定更高的“oracle”。

## 增量训推一致性

标准 suffix-LoRA 若一次性送入完整 suffix，80-sample exact match 可达 93.75%；改成
部署时的 seed block 加逐 token append 后只有 51.25%。这不是 LoRA target 数量不足，
而是当前自定义 BF16 Gemma 对不同执行 shape 存在不可忽略的数值差异，误差会在 greedy
解码中累积。

因此我们从标准 `suffix_lora, 3e-4, step 2000` checkpoint 出发，保留全部 18 层
attention/FFN LoRA，仅用真实增量路径继续训练 500 steps。`1e-4` continuation 的
step 400 最好：

| 指标 | 标准 suffix-LoRA | + incremental continuation |
|---|---:|---:|
| Full held-out token accuracy | 98.57% | 98.47% |
| Greedy exact（80 samples） | 51.25% | **83.75%** |
| Word F1（80 samples） | 0.814 | **0.916** |

continuation 没有牺牲 teacher-forced 准确率，并显著缩小了真实生成差距。83.75% exact
仍不是最终部署质量：剩余错误主要是相近物体或相邻阶段混淆，后续应增加视觉/进度
监督，而不是先削减 LoRA 层数。

## 动作质量与 serving

Suffix LoRA 的 root prefill 强制使用 base 权重，语言 seed 和后续 token 只写入 private
append cache。20 个 held-out observations 的 root KV 与零 adapter bit-exact；使用固定
diffusion noise 时，10-step denoising 的 action tensors 也全部 bit-exact，最大绝对差
为 0。最终 Policy 路径的单请求检查同时返回正确文本 `Turn on the stove.` 和
`(10, 32)` actions，并在文本结束后正确释放 request state。

Full-sequence LoRA 会改变 root KV（最大绝对差 69.5）和动作（最大绝对差 0.436）。
因此额外做了小规模 simulator 回归：四个 suite 的第一个任务、每个任务前 5 个固定
initial states、seed 7、`replan_steps=5`、10-step denoising。原 checkpoint 在这 20 个
状态上为 19/20，full-LoRA 为 20/20。该结果只说明这批样本未观察到明显动作退化；
样本太小，不能声称 full-LoRA 提升了整体成功率。

最终 suffix-LoRA 也使用相同 20 个状态通过 action+language continuous-batching 路径
复测，结果为 19/20，与 base reference 完全一致；四个 suite 分别为 5/5、5/5、4/5、
5/5，唯一失败也是 LIBERO-Goal 的同一个 initial state。由于样本量较小，动作逐 bit
检查仍是“adapter 不改变动作路径”的直接证据，rollout 结果作为实际 serving 路径的
回归验证。

## 运行时

以下为单张 RTX 4090、warmup 后的中位数。典型 held-out suffix 有 10 个输入 token、
其中 7 个是监督 target token。

| 路径 | Median latency |
|---|---:|
| 可复用的 observation/task prefix forward | 50.15 ms |
| Adapter-disabled incremental suffix request | 95.79 ms |
| Adapter-enabled incremental suffix request | 101.40 ms |
| Adapter 新增开销 | **5.61 ms** |

prefix saving / adapter overhead 为 8.94 倍。这个比较已经使用与部署一致的 fixed-shape
private cache、seed block 和 token append，而不是只估算 LoRA GEMM。首次 JIT 编译
不计入以上数字。

## 结果出处

- [机器可读汇总](results/2026-08-11/three_way_accuracy/aggregate_summary.json)
- [最佳 suffix-LoRA 正式评测](results/2026-08-11/three_way_accuracy/evaluation_incremental_continuation/best_suffix_lora_lr1e4_step400/summary.json)
- [最佳模型 80 条生成明细](results/2026-08-11/three_way_accuracy/evaluation_incremental_continuation/best_suffix_lora_lr1e4_step400/predictions.jsonl)
- [标准三组增量评测](results/2026-08-11/three_way_accuracy/evaluation_incremental/)
- [标准训练配置与日志](results/2026-08-11/three_way_accuracy/training/)
- [增量 continuation 配置与日志](results/2026-08-11/three_way_accuracy/training_incremental_continuation/)
- [Full-LoRA rollout](results/2026-08-11/three_way_accuracy/rollout/full_lora_lr3e4.jsonl)
- [Suffix-LoRA rollout](results/2026-08-11/three_way_accuracy/rollout/suffix_lora_incremental_step400.jsonl)
- [最终 serving 检查](results/2026-08-11/three_way_accuracy/serving_verification/best_suffix_lora_sample0.json)

早期错误归一化、旧 final-only adapter 和错误 stop-gradient 的输出已移到 workspace 的
`archive/libero_language_adapter_20260811/`，不再属于正式结果。
