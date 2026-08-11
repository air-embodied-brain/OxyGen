# LIBERO suffix-only language adapter：首轮可行性实验

日期：2026-08-11

## 结论

这轮实验支持 suffix-only language adapter 的基本可行性。我们冻结
`pi05_libero` checkpoint 的视觉编码器、语言主干、action expert 和动作投影，只训练
语言 suffix 上的低秩残差 adapter。最终选择的 rank-8 adapter 只有 589,824 个参数，
在 episode 隔离的验证集上达到 97.79% token accuracy；80 个跨任务样本的自由生成
exact match 为 88.75%，word F1 为 90.82%。未训练的零 adapter 在相同输入上 exact
match 和 word F1 都为 0，且生成不可读 token。

动作路径没有受到影响。20 个 held-out observation 使用各自固定的 diffusion noise，
比较训练 adapter 与零 adapter 后，完整 10-step denoising action tensor 全部逐 bit
一致，最大绝对差为 0。小规模 LIBERO rollout 中，原 checkpoint 和 adapter model
均为 4/20 success，分 suite 的结果也相同。

性能上，rank-8 adapter 对典型 7-token 请求增加约 2.08 ms 计算，而一次可复用的
image/prompt prefix forward 为 50.31 ms；adapter 开销约为可节省 prefix 计算的
4.1%，留有 24.1 倍余量。这个结果说明 adapter 本身不会抵消 prefix sharing 的收益，
但还不是完整的 OxyGen 端到端加速结果：当前评估器每个 token 会重新执行短 suffix，
尚未接入 OxyGen 持久化的增量 append state。

## 实验设置

本地 checkpoint 是 ModelScope 上的 `pi05_libero` snapshot，commit 为
`c27c8531f6748ee8814db1dc55cd62c6b9e862ea`。语言主干的 18 层各加入一个低秩
残差 adapter；adapter 只在 causal language suffix 上启用，在公共 prefix 和所有
action denoising step 中静态关闭。root KV 在训练时停止梯度，只有名称匹配
`suffix_lora` 的参数参与优化。

输入严格对齐仓库的官方 `pi05_libero` 配置：

- `discrete_state_input=False`。state 仍保留在 observation 中并按 checkpoint 统计量
  归一化，但不离散化到文本 prefix。
- raw HDF5 相机图像旋转 180 度，与官方 RLDS 转换和 rollout client 一致。与一条
  官方 RLDS 样本直接比较时，该方向的像素 MAE 为 3.50，其他候选方向为
  42.59–70.34。
- 公共 prefix 是图像和原始任务 instruction；语言 suffix 为
  `Subtask: <predicate-derived next label> EOS`。
- suffix 固定长度为 20；数据集中最长 target 为 18 tokens，没有截断。

校对中发现第一轮 smoke/训练曾残留 `discrete_state_input=True`。这批结果已经作废并
移出正式结果目录；下表和后续所有数字均来自修正后的重新训练。正式结果的
`config.json` 也显式记录了输入协议，便于后续审计。

## 数据与划分

数据来自四个标准 LIBERO suite 的全部 40 个任务。每个任务固定抽取 5 个 episode
作为验证集，其余 45 个用于训练；划分 seed 为 7，验证 episode 不会贡献任何训练帧。

| Split | Episodes | 采样帧数 |
|---|---:|---:|
| Train | 1,800 | 45,935 |
| Validation | 200 | 5,075 |

每条轨迹保留首帧、末帧、每 8 帧以及 `next` label 转换前后的两帧。这既控制了本地
训练成本，也保留了最容易混淆的阶段边界。

## 训练与选模

四组实验在 RTX 4090 GPU 0–3 上并行训练。统一使用 batch size 2、2,000 steps、
100 warmup steps、cosine decay（末端为 peak LR 的 0.1）、AdamW
（`b1=0.9`、`b2=0.95`、zero weight decay）和 gradient clipping 1.0。冻结权重为
BF16，adapter 与 optimizer state 为 FP32。四组均在 10.4–10.7 分钟内完成，稳定
step time 为 0.140–0.144 s，没有 OOM、NaN 或发散。

训练过程的 validation loss/token accuracy 使用相同的 40 个采样验证帧；自由生成
使用相同的 80 个样本，每个任务覆盖两个不同 target state。

| Rank | LR | 参数量 | Val loss | Token acc. | Greedy exact | Word F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | `1e-4` | 589,824 | 0.0862 | 97.06% | 76.25% | 85.28% |
| **8** | **`3e-4`** | **589,824** | **0.0298** | **99.12%** | **88.75%** | **90.82%** |
| 16 | `1e-4` | 1,179,648 | 0.0529 | 98.24% | 87.50% | 92.23% |
| 16 | `3e-4` | 1,179,648 | 0.0296 | 99.12% | 85.00% | 90.01% |

rank 8 / `3e-4` 的 exact match 最高，而且参数量和额外计算仅为 rank 16 的一半，
因此作为首轮配置。rank 16 / `1e-4` 的 word F1 略高，但 exact 少一个样本，不足以
抵消更大的运行时成本。

## 语言质量

选中的 checkpoint 在全部 5,075 个 held-out frames、42,281 个 target tokens 上得到：

| 指标 | 结果 |
|---|---:|
| Mean per-sample loss | 0.07618 |
| Perplexity | 1.0792 |
| Token accuracy | 97.79% |
| Greedy normalized exact match（80 samples） | 88.75% |
| Greedy mean word F1（80 samples） | 90.82% |

正确输出包括 `Turn on the stove.`、`Pick up the moka pot.` 和
`Put the yellow and white mug in the microwave.`。9 个 exact-match error 都是相邻
阶段判断错误，例如在 `Pick up the black bowl.` 与
`Place the black bowl on the plate.` 之间提前或滞后；没有乱码或不完整句子。这说明
主要瓶颈已经不是语言格式，而是从单帧 observation 判断阶段边界。

## 动作隔离与 rollout

adapter 分支在 prefix 和 action denoising 中静态关闭。20 个 held-out observation
分别使用固定 diffusion noise，训练 adapter 与零 adapter 的完整 10-step action
tensor 全部逐 bit 相同，最大绝对差为 0。

rollout 使用四个 suite 的第一个任务，每个任务 5 个初始状态，`replan_steps=5`、
10 denoising steps、seed 7。结果如下：

| Model | Spatial | Object | Goal | LIBERO-10 | Total |
|---|---:|---:|---:|---:|---:|
| 原 `pi05_libero` checkpoint | 4/5 | 0/5 | 0/5 | 0/5 | 4/20 |
| + suffix adapter | 4/5 | 0/5 | 0/5 | 0/5 | 4/20 |

这个 rollout 只用于确认新模型能够通过真实 simulator/action API 运行且未观察到回归。
样本量太小，不能估计 checkpoint 的整体成功率，也不能据此声称动作质量提升；动作
逐 bit 隔离测试才是“不影响 action path”的主要证据。

## 运行时成本

以下均为单张 RTX 4090、warmup 后的中位数。adapter microbenchmark 执行 18 层中
实际新增的 down projection、GELU、up projection 和 residual，并按 autoregressive
生成逐 token 计费。

| 项目 | Median latency |
|---|---:|
| 可复用的 image/prompt prefix forward | 50.31 ms |
| rank-8 adapter，每个 token | 0.298 ms |
| rank-8 adapter，典型 7-token 请求 | 2.08 ms |

因此 adapter 计算只占可省去 prefix forward 的约 4.1%，prefix saving / adapter
overhead 为 24.1 倍。adapter 的 FP32 参数只占约 2.25 MiB。

需要注意，现有自由生成评估器为了实现简单，每步从 root cache 重新执行当前短
suffix；它没有复用 OxyGen 的持久化增量 language state。因此这里验证的是新增 adapter
计算量不会吃掉 prefix-sharing 收益，而不是最终系统的端到端 latency。下一步应把
adapter-enabled suffix 接到 OxyGen 的固定 shape append state，再测共享 prefix 下的
完整 action+language 请求。

## 原始结果

- [四组训练日志与配置](results/2026-08-11/training/)
- [统一 80-sample 选模结果](results/2026-08-11/selection/)
- [最终完整验证与动作隔离](results/2026-08-11/evaluation/final_r8_lr3e4/summary.json)
- [最终自由生成明细](results/2026-08-11/evaluation/final_r8_lr3e4/predictions.jsonl)
- [零 adapter 对照](results/2026-08-11/evaluation/zero_rank8/summary.json)
- [原 checkpoint rollout](results/2026-08-11/rollout/baseline.jsonl)
- [adapter rollout](results/2026-08-11/rollout/adapter.jsonl)
- [选中的 rank-8 adapter](checkpoints/r8_lr3e4_step2000.npz)

## 下一步

优先级最高的是完成持久化增量 KV 集成和端到端 latency 测量。训练侧随后加入
`completed + next` 的可配置混合监督；本轮错误几乎全部来自相邻阶段，显式的进度
监督比继续增大 adapter 更可能有效。完成这两步后，再扩大 rollout 规模评估是否维持
动作成功率。
