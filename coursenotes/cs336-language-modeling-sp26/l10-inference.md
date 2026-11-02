---
title: "第 10 讲 · 推理"
description: "推理是模型训练完成后反复发生的成本，在智能体（agent）时代，生成 token 的数量几乎等同于投入的计算量。与训练可并行处理整个序列不同，自回归推理必须逐 token 生成，因此 generation 阶段是内存受限…"
---

# 第 10 讲 · 推理

> **Stanford CS336 语言模型从零构建 · L10** ｜ 讲者 Stanford Online ｜ 85 分钟 ｜ [原视频](https://www.youtube.com/watch?v=EfM546A79aM)

## 核心结论

推理是模型训练完成后反复发生的成本，在智能体（agent）时代，生成 token 的数量几乎等同于投入的计算量。与训练可并行处理整个序列不同，自回归推理必须逐 token 生成，因此 **generation 阶段是内存受限的**，而 attention 的 KV cache 是核心瓶颈。课程先建立算术强度、延迟、吞吐和 TTFT 的分析框架，再介绍减少 KV cache 的架构方法（GQA、MLA、跨层共享、滑动窗口、线性注意力）、量化与剪枝、投机解码，以及连续批处理和 PagedAttention 等系统优化。贯穿始终的原则是：**降低 KV cache 与内存流量，同时尽量不损失精度**。

---

## 1. 推理为什么重要 [00:05 - 05:02]

推理问题的定义很简单：已经训练好一个模型，给定 prompt，希望尽可能准确、尽可能快地生成 response。

推理出现在很多场景中：

- 聊天机器人与 AI 助手；
- 代码补全；
- agent 工作流；
- 批量数据处理；
- 需要生成结果的模型评估；
- 强化学习训练中的 rollout 生成与打分。

从成本角度看，训练是一次性成本，虽然非常昂贵，但结束后就结束了；推理是重复性成本，每天都会发生。转写中提到，OpenAI 每天大约处理 8.6 万亿 token，而一个前沿模型的训练 token 量约为 32 万亿，这意味着不到 4 天，推理产生的 token 量就可与一次大规模训练相当。

更重要的是，推理的重要性在过去一年明显上升。传统 chatbot 中，大部分生成 token 是给人读的，而人类阅读速度有限，因此推理速度超过某个阈值后收益有限。但在 agent 场景中，query 进入后，模型会思考、推理、调用工具、自省，最后才输出少量给人看的结果。**大部分 token 不是给人读的，而是计算本身**；只要问题足够有野心，token 生成量就没有上限。

商业上，闭源 API 提供商、开源权重模型服务商都在大量投入推理。开源社区常见工具包括 vLLM、SGLang、NVIDIA TensorRT 和 CPU 上的 llama.cpp 等。

![推理工作流示意图：模型与提示词输入推理模块，输出响应](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0001_00-00-18.jpg)

![推理效率重要性及聊天机器人 token 消耗瓶颈的代码页面](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0002_00-02-51.jpg)

![推理效率、token 消耗及推理服务公司分类的代码页面](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0003_00-04-16.jpg)

---

## 2. 三个“快”的指标与推理的根本难点 [05:02 - 08:15]

“快”在不同场景下含义不同，主要有三个指标：

1. **Time-to-first-token（TTFT）**：用户从提交 query 到看到第一个 token 的等待时间。交互式应用非常在意这个延迟，因为等待期间用户什么都做不了；一旦 token 开始流出，人类阅读速度本身有限，后续速度不一定要极快。
2. **Latency**：单个 query 的 token 生成速度，即 token 之间有多快出现。对交互式应用同样重要。
3. **Throughput**：单位时间内整个系统为多个 query 生成的 token 总数，度量 tokens/second。批量数据处理更关心吞吐，而不是单个请求多快返回。

Latency 和 throughput 通常相关，很多优化会同时改善二者，但通过 batch size 调节时会出现明确权衡。

推理与训练的根本区别在于：训练时可以看到所有 token，序列维度可以并行，Transformer 中的 attention 和 MLP 都是对一个大张量做矩阵乘法；推理由于自回归性质，必须一个 token 一个 token 地生成，**不能并行化序列维度**。这导致推理更难获得高算术强度，也更难充分利用加速器算力。

![推理性能指标与效率影响因素幻灯片，高亮TTFT定义](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0004_00-05-42.jpg)

---

## 3. 记号与 Transformer 结构回顾 [08:15 - 14:22]

课程使用类似 einsum 的维度记号：

- $B$：batch 维度，也是序列数量；
- $T$：序列维度，也是 token 数；
- $D$：模型维度；
- $F$：MLP 的 up-projection 维度，通常 $F=4D$；
- $N$：query head 数；
- $H$：每个 head 的维度；
- $K$：KV head 数或 KV 组数；
- $G$：每个 KV 组对应的 query head 数；
- $S$：输入 token 数（conditioning tokens）；
- $T$：输出 token 数。

约定：

$$F = 4D,\qquad D = N H,\qquad N = K \cdot G.$$

训练时 $S=T$；推理 generation 时通常 $T=1$，$S$ 是已有上下文长度。

在 einsum 图中：

- 红色维度是 contracting 维度，出现在两个操作数中并在结果中消失；
- 黑色维度是 regular 维度，只出现在一个操作数中并保留；
- 蓝色维度是 batching 维度，出现在两个操作数中但保留，不参与收缩。

例如：

$$BTD \times DH \rightarrow BTH.$$

Attention 中 Q、K、V 投影后做 attention；GQA 中 K/V 的头数可以少于 query 头数。MLP 则是 up/gate/down 投影，形式简单但计算量大。

问答中修正了 GQA 的记号：$K$ 是 KV 组数，$G$ 是每个 KV 组对应的 query head 数，因此每组有 $N/K$ 个 query heads。

![Transformer 维度记号（类 einsum）与收缩/批处理维度示例](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0006_00-09-58.jpg)

![注意力机制的计算流程图及张量维度符号说明表](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0007_00-11-24.jpg)

![MLP层的计算流程图及模型维度约定说明](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0008_00-12-49.jpg)

---

## 4. 算术强度回顾：从矩阵乘法到 H100 [14:22 - 18:45]

以矩阵乘法 $X(B \times D) \cdot W(D \times F)$ 为例：

- FLOPs：$2BDF$；
- 读取 $X$：$2BD$ 字节；
- 读取 $W$：$2DF$ 字节；
- 写回 $Y$：$2BF$ 字节；
- 总字节数：$2BD + 2DF + 2BF$。

算术强度为：

$$\text{intensity} = \frac{2BDF}{2BD + 2DF + 2BF}.$$

当 $B \ll D,F$ 时，分母主要由 $2DF$ 决定，强度近似为 $B$。

H100 的加速器强度为：

$$\frac{989\ \text{TFLOPS}}{3.35\ \text{TB/s}} \approx 295.$$

若计算强度大于 295，则为 compute-bound，是好的；若小于 295，则为 memory-bound，是不好的。对于这个矩阵乘法，$B > 295$ 才能 compute-bound。

极端情况 $B=1$ 时，算术强度为 1：需要读取整个 $D \times F$ 矩阵，却只做 $2DF$ 次 FLOPs，严重 memory-bound。这正是推理 generation 中常见的“很瘦的矩阵乘”形态。

![矩阵乘法算术强度的计算代码，包含读取X和W、计算Y以及写回Y的FLOPs和内存传输字节数统计](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0010_00-15-40.jpg)

![推导算术强度等于批次大小B的代码，并给出H100加速器的算术强度计算](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0011_00-17-06.jpg)

![总结算术强度与加速器强度的比较结论，并分析B=1时的极端情况](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0012_00-18-31.jpg)

---

## 5. 朴素推理与 KV Cache [20:29 - 25:08]

最朴素的推理方式是：把 prompt 喂入 Transformer，得到 logits，采样一个 token；把该 token 拼到 prompt 后，再把整个序列喂入 Transformer，重复直到生成结束。

这种方法正确但非常低效。生成第 $t$ 个 token 时 attention 需要 $O(t^2)$ 时间，因此生成 $T$ 个 token 总共需要 $O(T^3)$ FLOPs。

关键观察是：对于 causal Transformer，追加新 token 不会改变前面 token 的激活和 key/value。因此不同前缀之间可以共享大量计算。

解决方案是把 key/value 存到 HBM 中，即 **KV cache**。这样在生成新 token 时，不需要重新计算之前所有 token 的 K/V。

KV cache 的大小为：对每个 sequence（$B$）、每个 token（$S$）、每层（$L$）、每个 KV head（$K$），存储一个 $H$ 维向量；K 和 V 各一份，BF16 每元素 2 字节：

$$\text{KV cache bytes} = B \cdot S \cdot L \cdot K \cdot H \cdot 2 \cdot 2.$$

推理分为两个阶段：

1. **Prefill**：给定 prompt，编码为向量并填充 KV cache。由于一次看到整个 prompt，可以像训练一样并行计算，通常是 compute-bound。
2. **Generation**：逐个生成 response token。每步只计算新 token 的 K/V，并复用已有 KV cache，但必须顺序执行，通常是 memory-bound。

![朴素采样（Naive Sampling）生成序列的流程图，展示Transformer如何逐个生成token](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0013_00-20-38.jpg)

![朴素采样流程图及复杂度分析，并引出KV Cache解决方案](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0014_00-22-48.jpg)

![KV Cache 在 Prefill 与 Generate 阶段的显存与计算流程示意图](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0015_00-24-13.jpg)

![KV Cache 定义及推理两阶段（Prefill/Generation）的文字说明页](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0016_00-25-39.jpg)

---

## 6. Prefill 与 Generation 的算术强度分析 [25:08 - 34:44]

只考虑 matmul，因为它们是主要计算量。

### MLP 层

MLP 的 FLOPs 约为：

$$6 B T D F,$$

字节传输约为：

$$4 B T D + 4 B T F + 6 D F.$$

当 $B T \ll D,F$ 时，算术强度近似为：

$$\text{intensity}_{\text{MLP}} \approx B T.$$

- Prefill 时 $T=S$，强度约为 $B S$。只要 batch 大、序列长，就能 compute-bound。
- Generation 时 $T=1$，强度约为 $B$。这里的 $B$ 是并发请求数。如果并发请求足够多，MLP 的 generation 仍可接受；但并发数取决于线上负载，可能波动。

### Attention 层

Attention 的算术强度为：

$$\text{intensity}_{\text{attn}} = \frac{ST}{S+T}.$$

- Prefill 时 $T=S$，强度为 $S/2$。长序列下仍然不错。
- Generation 时 $T=1$，强度为：

$$\frac{S}{S+1} < 1.$$

这远低于 H100 的 295，是推理的根本瓶颈。

为什么 attention 不能像 MLP 一样靠增大 $B$ 提升强度？因为 MLP 的权重跨序列共享，一次加载权重可以服务整个 batch；而 attention 中每个序列有自己独立的 KV cache，增大 $B$ 只是增加彼此独立的 matmul，不能摊薄权重加载成本。Attention 中的 batch 维度是 batching 维度，不会像 contracting 维度那样带来高算术强度。

总结：

- Prefill 是 compute-bound；
- Generation 是 memory-bound；
- Prefill MLP 强度 $B S$，很好；
- Prefill attention 强度 $S/2$，可用；
- Generation MLP 强度 $B$，取决于并发请求；
- Generation attention 强度 $<1$，是 transformer 推理的根本瓶颈。

![MLP 层 FLOPs 与显存传输量的计算代码及断言验证](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0017_00-27-04.jpg)

![Prefill 与 Generation 两阶段的算术强度对比及结论](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0018_00-31-21.jpg)

![对比 Prefill 与 Generation 两阶段计算强度的公式及 MLP 与 Attention 批处理差异的幻灯片](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0019_00-32-46.jpg)

![总结 Prefill 计算受限与 Generation 内存受限，并列出各阶段计算强度公式的幻灯片](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0020_00-34-12.jpg)

---

## 7. 实例：Llama 2 13B 的延迟与吞吐 [34:44 - 45:36]

以 Llama 2 13B 在 H100 上为例。配置大致为：

- 序列长度 $S=1024$；
- 模型维度 $D=5120$；
- MLP 维度 $F=13824$；
- query heads $N=40$；
- KV heads $K=40$（无 GQA）；
- head dimension $H=128$；
- 层数 $L=40$；
- 词表大小 $V=32000$；
- HBM 带宽约 $3.35\ \text{TB/s}$。

参数量计算得到约 13,015,449,600，符合 13B 的标称。

内存包括参数和 KV cache：

$$\text{memory} \approx 838{,}860{,}800 \cdot B + 26{,}030{,}899{,}200\ \text{bytes}.$$

第一项是随 batch 增长的 KV cache，第二项是 BF16 参数。

延迟由内存流量决定：

$$\text{latency} = \frac{\text{memory}}{\text{memory bandwidth}}.$$

吞吐为：

$$\text{throughput} = \frac{B}{\text{latency}}.$$

代入后：

$$\text{latency} \approx 0.000250406 \cdot B + 0.0077704\ \text{s},$$

$$\text{throughput} \approx \frac{127792.36 \cdot B}{32B + 993}\ \text{tokens/s}.$$

具体数值：

| Batch size | Latency (s/token) | Throughput (tokens/s) | 备注 |
|---:|---:|---:|---|
| 1 | 0.0080 | 124.7 | 延迟最好，吞吐最低 |
| 64 | 0.0238 | 2689.8 | 吞吐显著提升 |
| 256 | 0.0719 | 3561.8 | 内存约 240 GB，超过 H100 容量 |

可以看到：

- Latency 随 $B$ 线性增长，因为 KV cache 随 $B$ 增长，需要搬运更多内存。
- Throughput 随 $B$ 提升，但会渐近饱和，并且受限于显存容量，无法无限增大 batch。
- 小 batch 改善延迟但降低吞吐；大 batch 改善吞吐但恶化延迟。

此外，如果启动 $M$ 个模型副本，延迟不变，吞吐提升 $M$ 倍。更复杂的模型/KV cache 切分可参考 scaling book 的 inference 章节。

TTFT 基本等于 prefill 时间。要更快 TTFT，prefill 阶段应使用较小 batch；要提高生成吞吐，generation 阶段应使用较大 batch。

![计算 Transformer 参数量、KV 缓存大小及总内存占用的代码实现](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0021_00-38-28.jpg)

![延迟与吞吐量计算公式及代入 config 参数后的符号化简结果](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0022_00-39-54.jpg)

![展示Transformer模型内存、延迟和吞吐量的计算公式及代入配置参数后的符号推导过程](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0023_00-41-19.jpg)

![对比不同批次大小（1、64、256）下的模型参数量、内存占用、延迟和吞吐量计算结果](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0024_00-42-45.jpg)

![关于延迟与吞吐量权衡的幻灯片，包含批处理大小对TTFT和生成吞吐量的影响](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0025_00-45-36.jpg)

---

## 8. 减少 KV Cache 的架构方法 [45:36 - 63:59]

由于推理内存受限，KV cache 又占用大量内存，最直接的优化方向就是缩小 KV cache。关键是不要损失太多精度。

### 8.1 分组查询注意力（GQA）[47:01 - 50:52]

Multi-head attention（MHA）中，每个 token 的每个 head 都有独立的 K 和 V。GQA 保留 $N$ 个 query heads，但只计算 $K$ 组 K/V，每组服务 $N/K$ 个 query heads。Multi-query attention（MQA）是 $K=1$ 的极端情况，精度较差，实际很少单独使用。

GQA 把 KV cache 缩小为原来的 $N/K$ 倍。由于推理是 memory-bound，减少内存流量会同时改善延迟和吞吐——这里的改善不是通过增大 batch 换来的，因此二者不冲突。

以 Llama 13B 为例：

- 原始 $K=40$、batch 64：latency 约 0.0238 s，throughput 约 2689 tokens/s；
- 改为 $K=8$、batch 64：latency 约 0.0100 s，throughput 约 6416 tokens/s；
- 而且 KV cache 变小后，可以把 batch 增大到 256，进一步提高吞吐。

精度方面，GQA 原论文显示在多项任务上精度损失很小；但后续 DeepSeek 的论文显示 GQA 在困难 benchmark 上可能明显伤害精度。因此对非纯数学的精度结论要谨慎。

![介绍分组查询注意力（GQA）的幻灯片，包含减少KV缓存大小的动机和三种注意力机制对比图](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0026_00-47-01.jpg)

![展示MHA、GQA、MQA在不同GQA组数下每样本时间（延迟）对比的折线图](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0027_00-48-27.jpg)

![GQA 与 MHA 在参数量、显存、延迟和吞吐量上的性能对比代码及结果](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0028_00-49-52.jpg)

![不同注意力机制模型（MHA vs GQA）在多项基准测试上的准确率对比表](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0029_00-51-18.jpg)

![不同注意力机制（MHA、GQA、MQA）在7B稠密模型上的性能对比表](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0030_00-54-09.jpg)

### 8.2 多头潜在注意力（MLA）[51:32 - 54:47]

DeepSeek 提出的 MLA 采用另一种思路：不减少 K/V 的头数，而是压缩 K/V 的表示。

正常 attention 中，激活乘以矩阵得到 K 和 V，维度通常是 $N \times H$，即模型维度，较大。MLA 先把激活投影到一个低维潜在向量 $C$，例如 DeepSeek V2 从约 16,000 维压缩到 512 维，然后从 $C$ 计算 K 和 V。这样只需要存储 $C$，需要时再 materialize K/V。

MLA 与 RoPE 不兼容，因为 RoPE 直接作用在 K/V 上；DeepSeek 额外增加一些维度来处理 RoPE，但整体仍有大幅压缩。

KV cache 变小后，延迟和吞吐按近似线性的关系改善。精度上，DeepSeek 报告 MLA 与 MHA 相当甚至略好，这与 GQA 论文的结论存在张力。

问答中提到：MLA 与直接降低模型维度不同。降低模型维度是不加区分地压缩所有部分，可能更差；MLA 的思路是找到模型中可压缩的位置，这需要实验探索。

### 8.3 跨层注意力、局部注意力与线性注意力 [55:30 - 63:39]

**跨层注意力（cross-layer attention）**：正常每层都有自己的 K/V。跨层注意力只计算一部分层的 KV，其他层复用之前层的 KV。类似 GQA 跨 head 共享，这里跨 layer 共享，可改善精度-效率的 Pareto 前沿。

**局部/滑动窗口注意力**：每个 token 只关注最近 $K$ 个 token，而不是整个历史。这样 KV cache 与序列长度无关，对长上下文特别有利。多层堆叠后，有效上下文长度可以超过单层窗口，因为信息可以逐层传播。也可以加入全局 token，形成“全局 + 滑动窗口”的混合结构。

代价是降低表达力，仍可能伤害精度。常见做法是局部注意力层与全局注意力层交错，在精度和速度之间平衡。

**线性注意力与状态空间模型**：不存储完整 KV cache，而是把历史压缩成固定大小的状态。最朴素的线性注意力把 K/V 求和成一个向量；GateNet、DeltaNet、Mamba 等用更复杂的方式压缩，但尽量不遗忘。它们比滑动窗口更强大，有更多表示空间，甚至可以表示滑动窗口行为；但在长上下文 needle-in-a-haystack 任务中，压缩整个历史必然丢失信息，没有免费午餐。实际中可以组合全注意力、滑动窗口注意力和线性注意力，分别捕获局部高分辨率信息和全局摘要。

**DeepSeek 的压缩稀疏注意力**：

- CSA（Compressed Sparse Attention）：每 $m$ 个 token 压缩成 1 个；
- DSA（DeepSeek Sparse Attention）：用轻量 query/key 计算索引分数，选出 top-$k$ 保留；
- HCA（Heavily Compressed Attention）：进一步加重压缩。

![DeepSeek v4注意力机制架构图及局部与全局注意力交替方案](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0031_01-00-57.jpg)

![DeepSeek稀疏注意力机制架构图，展示KV Token与Query Token经Token-Level Compressor压缩为Compressed KV Entries和Compressed Indexer Keys的过程](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0032_01-02-42.jpg)

---

## 9. 量化与剪枝 [63:59 - 71:43]

### 量化

量化通过降低数值精度来减少内存占用和内存流量，从而改善延迟和吞吐。可选精度从 BF16 一直到 INT4 等。关键仍是精度。

两种主要路线：

- **Quantization-aware training（QAT）**：训练时在前向传播中量化再反量化，模拟量化误差，让权重适应量化。效果通常更好，但需要大规模训练，成本高。
- **Post-training quantization（PTQ）**：训练后量化，便宜得多。朴素做法是按 tensor 或 layer 确定 scale 和 zero point，但效果一般。GPTQ 使用 Hessian 信息逐层量化，并把误差传播到未量化权重中进行校正。激活感知量化观察到某些激活通道幅度远大于其他通道，对应权重更重要，因此给这些通道保留更高精度（如 FP16），其余通道用低精度（如 INT3）。

### 剪枝

剪枝直接移除模型的一部分，再“修复”模型。流程是：

1. 估计不同部分的重要性；
2. 保留最重要的隐藏单元/层，移除其余部分；
3. 在相关数据或任务上 post-train，使模型恢复。

NVIDIA 的工作把 15B 模型剪到 8B，精度损失不大，所需训练量远小于从头训练。重要性判断通常用 calibration set 看激活幅度：接近零的死单元可以剪掉；高均值低方差的通道也许可以吸收为 bias。

整体上，这类方法是在减少参数量或 KV cache，可以定义更快的架构并训练，也可以从原模型初始化，再通过蒸馏修复。

---

## 10. 投机解码（Speculative Decoding）[71:43 - 76:55]

前面的方法大多是有损的。投机解码提供了一种无损加速思路。

核心不对称性：**验证一个序列比逐 token 生成快得多**。Prefill 可以并行处理所有 token 并给出概率，是 compute-bound 的；generation 只能一个 token 一个 token 来。

方法：

1. 用一个便宜的小 draft model $P$ 自回归生成 $K$ 个候选 token；
2. 用真正想要的 target model $Q$ 并行计算这 $K$ 个 token 的 logits；
3. 逐个决定接受或拒绝。接受概率为：

$$\min\left(1, \frac{Q}{P}\right).$$

如果拒绝，则从残差分布中采样一个 token 并停止本轮，然后重新开始。这个流程类似拒绝采样，但保证最终样本精确服从 target model 的分布，因此是无损的。

$K$ 的选择有 trade-off：太小不能充分利用 target model 的并行验证能力；太大则 draft token 被拒绝的概率高。经验上甜点约在 3 到 4 个 token。

Draft model 越小越好，但最好与 target model 接近，因此可以用蒸馏。前面通过减少 KV cache 得到的小模型，如果精度满意就直接服务；如果不满意，至少可以作为 draft model，用主模型修正。

---

## 11. 动态工作负载与系统优化 [76:55 - 83:46]

线上推理服务与训练很不一样：请求在不同时间到达，前缀可能共享，长度各不相同，batch 是动态且不规则的。

### 连续批处理（Continuous Batching）

Orca 系统提出连续批处理：每一步为当前 batch 中所有序列各解码一个 token；某个序列生成结束后就退出 batch；新请求到达时加入 batch。这样 batch 是动态更新的，而不是等一整批全部完成。

### 选择性批处理（Selective Batching）

不同请求长度不同，attention 计算依赖序列长度，难以把不同长度的 attention 直接合并。但 MLP 等非 attention 部分与序列长度无关，可以把所有序列拼接成一个 mega-sequence 统一处理，从而提高效率。

### PagedAttention

vLLM 引入 PagedAttention 管理 KV cache。朴素分配会带来两类碎片：

- 内部碎片：不知道序列何时结束，只能按最大 token 数预留内存，造成浪费；
- 外部碎片：不同请求之间的空闲空间太小，难以有效利用。

PagedAttention 借用操作系统分页思想，把 KV cache 划分成固定大小的非连续块，用索引表记录块的位置。块不需要连续存放，只要按块对齐即可。

这还带来前缀共享能力：

- 相同 system prompt 的 KV cache 只需计算一次，所有请求复用；
- 同一 prompt 要生成多个 response 时，可以共享 prompt 的 KV cache；
- 使用 copy-on-write 语义：多个样本采样到相同 token 时继续共享块；采样到不同 token 时分裂块，各自继续。

这些系统优化把操作系统中的分页、写时复制等思想带到了推理服务中。

---

## 12. 总结与展望 [83:46 - 85:21]

推理非常重要，且与训练的工作负载非常不同：同一个模型，训练时并行处理序列，推理时逐 token 自回归，导致 generation 严重 memory-bound；线上服务还是动态、不规则的。

课程介绍的技术包括：

- 减少 KV cache 的架构：GQA、MLA、跨层注意力、滑动窗口/局部注意力、线性注意力；
- 量化、剪枝与蒸馏；
- 投机解码；
- 连续批处理、选择性批处理、PagedAttention 等系统优化。

所有这些方法背后的共同原则是：**减少 KV cache 和内存流量，但不要过度伤害精度**。

最后值得强调的是新架构的潜力。Transformer 的 attention 和 KV cache 机制本质上对推理不友好；状态空间模型、线性注意力、扩散模型等非自回归或压缩历史的方法，可能为推理带来更大的改进空间。下节课将回到 scaling laws 第二部分。

---

## 要点回顾

- 推理是重复性成本；agent 时代生成 token 数近似等于计算量，且没有明显上限。
- 三个关键指标：TTFT（首 token 等待）、latency（单请求 token 速度）、throughput（系统总 token 速度）。
- 训练可并行序列，推理必须逐 token 自回归；这是推理算术强度低的根本原因。
- 矩阵乘算术强度在 $B \ll D,F$ 时近似为 $B$；H100 加速器强度约 295，低于它即为 memory-bound。
- 朴素推理生成 $T$ 个 token 需要 $O(T^3)$；KV cache 通过复用前缀 K/V 把每步计算降为只处理新 token。
- Prefill 并行编码 prompt，通常 compute-bound；generation 逐 token 生成，通常 memory-bound。
- MLP generation 强度约 $B$，可靠并发请求提升；attention generation 强度为 $S/(S+1)<1$，是根本瓶颈，增大 $B$ 也无法改善。
- Llama 2 13B 实例：latency 随 batch 线性增长，throughput 渐近饱和并受显存限制；小 batch 利于延迟，大 batch 利于吞吐。
- GQA 通过减少 KV 组数把 KV cache 缩小 $N/K$ 倍，可同时改善延迟和吞吐；但精度影响需按任务评估。
- MLA 把 K/V 压缩到低维潜在向量再 materialize，压缩率很高，但需处理与 RoPE 的兼容问题。
- 跨层共享 KV、滑动窗口、线性注意力/SSM 都是减少 KV cache 或固定状态大小的方向，均存在表达力与精度的权衡。
- 量化（QAT/PTQ、GPTQ、激活感知量化）和剪枝通过降低精度或移除参数减少内存流量，需要 post-train 或校准来保精度。
- 投机解码用 draft model 猜多个 token，target model 并行验证，按 $\min(1,Q/P)$ 接受，保证无损；$K$ 的经验甜点约 3–4。
- 连续批处理动态维护 batch；选择性批处理只合并 MLP 等非 attention 部分；PagedAttention 用分页块管理 KV cache，支持前缀共享和 copy-on-write。
- 新架构（SSM、线性注意力、扩散）可能比 Transformer 更适合推理，是未来重要方向。
