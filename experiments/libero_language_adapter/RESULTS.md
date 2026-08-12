# LIBERO language adapter：Suffix LoRA 可行性实验

日期：2026-08-11

## Final v6 suffix-LoRA rerun (2026-08-12)

The final run uses the reviewed v6 predicate annotations: 40 tasks, 2,000
demonstrations, and 338,575 annotated frames. Five episodes per task are held
out, producing 47,773 stride-8 training samples and 5,277 validation samples.
The adapter is a rank-16 LoRA on Q/K/V/O and gated/up/down FFN projections in
all 18 language layers. Training and evaluation both use the deployed
incremental suffix path; the observation, state, and task prompt use the frozen
base model and remain the shared root prefix.

Checkpoint selection used a fixed 400-sample task/target-balanced greedy set.
The first 2,000-step sweep had not converged: the best exact match was still the
last point (89.25% at `3e-4`). Low-rate continuation and two confirmation points
then reached a 90.25% plateau. The selected confirmation checkpoint ties the
best exact score and has the highest Word F1 (0.932).

| Metric | Result |
|---|---:|
| Full held-out incremental token accuracy (5,277 samples) | 98.10% |
| Greedy normalized exact (400 samples) | 90.25% |
| Greedy Word F1 | 0.932 |
| Spatial / Object / Goal / LIBERO-10 exact | 90% / 91% / 92% / 88% |
| Root KV max difference from zero-adapter path | 0 |
| Fixed-noise 10-step action max difference (20 observations) | 0 |

On one RTX 4090, a median 10-token suffix request takes 92.59 ms with the
adapter disabled and 98.80 ms with it enabled, so the full-layer LoRA adds
6.21 ms. The reusable root prefix forward takes 47.79 ms; the saved duplicate
prefix computation is therefore 7.69x the measured adapter overhead.

The real simulator regression uses four standard suites, task 0 in each suite,
initial states 0--4, 10-step action denoising, and `replan_steps=5`. Every action
replan creates a new language request, and all unfinished requests are advanced
by five tokens together. All 20 episodes succeeded without exceptions. Across
665 replans, every call created one request; the first call of each episode had
batch size 1 and the remaining 645 calls had batch size 2. All 665 requests
produced non-empty model text, 647 completed before episode termination, and 18
were released when their successful episode ended. These rollouts are a serving
and qualitative regression, not a full LIBERO success-rate estimate.

Local raw outputs are under
`/home/lixiangyu/oxygen_ws/libero_exp/language_adapter_runs/v6_suffix_lora_20260812`.

## 第二轮：balanced incremental 训练

第一轮确认了 suffix-only LoRA 可以在不改变 action path 的前提下恢复文本
能力，但标准 full-suffix teacher forcing 与真实逐 token 解码之间有明显差距。
本轮从第一轮 `suffix_lora, 3e-4, step 2000` 开始，保留 18 层 language
attention Q/K/V/O 和 FFN gating/up/down 上的全部 rank-16 LoRA，使用部署时完全
相同的 block-plus-token 增量路径训练 2,000 steps。训练改为先均匀抽取
40 个任务，再在该任务内均匀抽取 target/predicate，避免长 episode 和高频
`Task complete` 标签主导梯度。

三个学习率使用相同的初始 adapter、batch size 2、100-step warmup、AdamW、
cosine decay 和 gradient clipping 1.0；每组都是一次连续的 2,000-step 训练，
中途没有重置 optimizer。一次训练约需 39 分钟，单张 RTX 4090 即可完成。

### Checkpoint 选择

我们使用固定的 400 条验证集选点：40 个任务每个 10 条，覆盖验证集全部
42 种 target。指标是真实增量 greedy generation，而非仅看 teacher-forced token
accuracy。

| LR / step | 500 | 1,000 | 1,500 | 2,000 |
|---|---:|---:|---:|---:|
| `1e-4` | 72.75% | 84.50% | **86.50%** | 85.50% |
| `3e-5` | 68.50% | 77.50% | 80.25% | - |
| `1e-5` | 63.00% | 68.25% | 70.50% | - |

表中为 exact match。`1e-4, step 1500` 达到最高 exact，Word F1 也最高
（0.9047），因此选为正式 checkpoint。虽然固定 monitor 的 teacher-forced token
accuracy 在 step 2,000 继续升高，greedy exact 已从 86.50% 回落到 85.50%；
这说明真实增量生成指标比单独的 loss 或 token accuracy 更适合选点。

### 正式评测

| 项目 | 结果 |
|---|---:|
| 全部 5,075 个 held-out frames，增量 teacher-forced token accuracy | **98.13%** |
| 全部 42,281 个 target tokens，perplexity | **1.053** |
| 400 条分层样本，增量 greedy exact | **86.50%** |
| 400 条分层样本，Word F1 | **0.9047** |
| 20 个 observations，root KV / 10-step action tensor | **bit-exact / bit-exact** |

Suffix LoRA 只在 language seed 和后续 token 上启用；observation/state/task prompt
构成的 root KV 始终由 frozen base 计算。因此 action expert 读到的 root KV
与零 adapter 完全一致，在固定 diffusion noise 下，20 个 observation 的完整
10-step action tensor 也逐 bit 一致，最大绝对差为 0。

分 suite 的 greedy exact 为 LIBERO-Spatial 81%、Object 92%、Goal 89%、LIBERO-10
84%。54 条错误中，13 条是将 `Place the black bowl on the plate.` 提前判为
`Task complete.`；其余主要是相邻阶段或外观相近物体的混淆。这些错误表明当前
adapter 已经学会了标注语言和大部分任务阶段，但阶段边界仍是后续数据和
训练的主要改进点。

在 RTX 4090 上，本轮样本的中位 suffix 输入长度为 12 tokens（其中 target
中位数为 9）。启用 adapter 前后的增量 suffix request 中位延迟分别为
118.83 ms 和 127.56 ms，adapter 引入 **8.73 ms** 开销。可复用的 prefix forward
中位延迟为 **47.56 ms**，是 adapter 开销的 **5.45 倍**。因此，在当前文本
长度和完整全层 LoRA 配置下，adapter 没有抵消 prefix sharing 的收益。

### 真实 LIBERO rollout

我们通过真实 OxyGen continuous-batching serving 路径运行了 20 个可视化 episode：
四个标准 suite 各选 task 0，每个任务使用 initial state 0--4，`replan_steps=5`，
10-step action denoising，seed 7。每次 action replan 都基于当前 observation 新建一个
language request，同时将所有未完成 request 各推进 5 tokens；动作只为最新 observation
生成。LIBERO 以 20 Hz 运行，因此 action replan 与 language request arrival 均为 4 Hz。

20 个 episode 均成功且无异常。636 次 replan 全部创建了一个新 request，其中 611 次
还恢复了上一 request，即实际 batch size 为 2；其余 25 次 batch size 为 1，出现在
episode 的第一次 replan，或上一 request 已在 5 tokens 内遇到 EOS 时。621 个 request
在 episode 内输出了完整非空文本；15 个尚未完成的末尾 request 在 episode 成功后随
连接关闭而清理。模型的句子通常在 10 tokens 内遇到 EOS，因此本次自然文本 rollout
没有形成大于 2 的 active batch。视频字幕直接来自对应 simulator frame 的 batched
policy response，并显示当前及前两个 request 的增量状态；所有视频均以 0.5x 实时速度
渲染，大小为 0.091--0.281 MB。

这个 20/20 是覆盖四个 suite 的 serving 回归和定性检查，不是完整 LIBERO
benchmark success rate；adapter 不改变动作的直接证据仍是上述 bit-exact 检查。

## 第一轮：三组对照

### 结论

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
| Full-sequence LoRA | `3e-4` | 98.66% | 51.25% | 0.774 | no / no | 6.41 ms |
| Final MLP | `3e-4` | 66.80% | 3.75% | 0.283 | yes / yes | below timing noise |
| Base LIBERO checkpoint | - | 0.75% | 0.00% | 0.000 | - | - |

这组结果给出两个直接判断。第一，外挂 final MLP 很便宜，但容量明显不足。第二，
full-sequence 和 suffix-only LoRA 的文本准确率相当；前者的 token accuracy 略高，
两者的 sequence exact 相同，但前者会改变 action 所读取的 root KV。因此它是表达能力
更大的容量对照，不是验证准确率必然更高的“oracle”。

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

Full-sequence LoRA step 2000 会改变 root KV（最大绝对差 71.0）和动作（最大绝对差
0.418）。另用 step 1500 checkpoint 做了小规模 simulator 回归：四个 suite 的第一个
任务、每个任务前 5 个固定 initial states、seed 7、`replan_steps=5`、10-step
denoising。原 checkpoint 在这 20 个状态上为 19/20，full-LoRA 为 20/20。该结果只说明
这批样本未观察到明显动作退化；样本太小，不能声称 full-LoRA 提升了整体成功率。

最终 suffix-LoRA 也使用相同 20 个状态通过 action+language continuous-batching 路径
复测，结果为 19/20，与 base reference 完全一致；四个 suite 分别为 5/5、5/5、4/5、
5/5，唯一失败也是 LIBERO-Goal 的同一个 initial state。由于样本量较小，动作逐 bit
检查仍是“adapter 不改变动作路径”的直接证据，rollout 结果作为实际 serving 路径的
回归验证。该历史运行与第二轮的 20/20 使用不同 checkpoint 和 request/EOS
生命周期，不是可配对比较的 success-rate 实验。

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

## 第一轮的收敛判断

当前 2,000 steps 还不能视为充分收敛。Batch size 为 2，训练按 45,935 个帧有放回
采样，因此 2,000 steps 只产生 4,000 次样本抽取，期望覆盖约 3,830 个不同帧（8.3%）。
从 step 1500 到 2000，suffix-LoRA 的完整 held-out token accuracy 仍由 98.16% 升到
98.57%，greedy exact 由 47.50% 升到 51.25%；full-LoRA 也由 98.33% / 40.00% 升到
98.66% / 51.25%。继续训练仍有增长空间。

训练时的旧 quick validation 每次只抽 40 帧且随 step 更换样本，适合发现发散，不适合
选择 checkpoint；这也是原先误选 full-LoRA step 1500 的原因。本轮已将其替换为
固定的 task/target-balanced monitor，并以 400 条真实增量 greedy generation 作为
主要选点依据。结果显示，token accuracy 继续上升不代表 sequence exact 也会
继续上升；后续训练应继续同时监控完整验证 loss 和固定集上的真实增量生成。

## 结果出处

- [第二轮机器可读汇总](results/2026-08-11/balanced_incremental_v2/aggregate_summary.json)
- [第二轮正式评测](results/2026-08-11/balanced_incremental_v2/evaluation/summary.json)
- [400 条增量生成明细](results/2026-08-11/balanced_incremental_v2/evaluation/predictions.jsonl)
- [Checkpoint 扫描结果](results/2026-08-11/balanced_incremental_v2/checkpoint_scan.json)
- [20 个 rollout 汇总](results/2026-08-11/balanced_incremental_v2/rollout/manifest.json)
- [Rollout 原始记录](results/2026-08-11/balanced_incremental_v2/rollout/rollouts.jsonl)
- [多请求 continuous-batching 汇总](results/2026-08-12/multirequest_rollout/summary.json)
- [本地多请求视频审查页](outputs/balanced_incremental_multirequest_review_v2/index.html)
- [机器可读汇总](results/2026-08-11/three_way_accuracy/aggregate_summary.json)
- [最佳 suffix-LoRA 正式评测](results/2026-08-11/three_way_accuracy/evaluation_incremental_continuation/best_suffix_lora_lr1e4_step400/summary.json)
- [最佳模型 80 条生成明细](results/2026-08-11/three_way_accuracy/evaluation_incremental_continuation/best_suffix_lora_lr1e4_step400/predictions.jsonl)
- [标准三组增量评测](results/2026-08-11/three_way_accuracy/evaluation_incremental/)
- [标准训练配置与日志](results/2026-08-11/three_way_accuracy/training/)
- [增量 continuation 配置与日志](results/2026-08-11/three_way_accuracy/training_incremental_continuation/)
- [Full-LoRA step-1500 rollout](results/2026-08-11/three_way_accuracy/rollout/full_lora_lr3e4_step1500.jsonl)
- [Suffix-LoRA rollout](results/2026-08-11/three_way_accuracy/rollout/suffix_lora_incremental_step400.jsonl)
- [最终 serving 检查](results/2026-08-11/three_way_accuracy/serving_verification/best_suffix_lora_sample0.json)

早期错误归一化、旧 final-only adapter 和错误 stop-gradient 的输出已移到 workspace 的
`archive/libero_language_adapter_20260811/`，不再属于正式结果。
