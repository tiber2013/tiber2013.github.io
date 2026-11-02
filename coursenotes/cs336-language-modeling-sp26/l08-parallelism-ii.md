---
title: "第 8 讲 · 并行化 II：数据、流水线、张量与专家并行"
description: "大规模语言模型训练必须同时解决计算和显存两个瓶颈，因此并行策略不是单一选择，而是数据并行、ZeRO/FSDP、流水线并行、张量并行、序列并行、专家并行和上下文并行的组合。"
---

# 第 8 讲 · 并行化 II：数据、流水线、张量与专家并行

> **Stanford CS336 语言模型从零构建 · L08** ｜ 讲者 Stanford Online ｜ 80 分钟 ｜ [原视频](https://www.youtube.com/watch?v=6-cXp-aOmdg)

## 核心结论

大规模语言模型训练必须同时解决**计算**和**显存**两个瓶颈，因此并行策略不是单一选择，而是数据并行、ZeRO/FSDP、流水线并行、张量并行、序列并行、专家并行和上下文并行的组合。  
数据并行最容易扩展计算，但会消耗全局 batch size 且不节省激活内存；ZeRO/FSDP 通过切分优化器状态、梯度和参数，以很小的通信代价换取显著显存收益。  
模型并行通过切分层、矩阵或专家来进一步扩展显存，但不同策略的通信模式差异很大：张量/专家并行通信量大、适合高速互联；流水线并行通信量小、适合跨节点慢链路。  
实际训练中的经验法则很清楚：尽量最大化数据并行；张量/专家并行限制在单机高速互联内；用流水线或 FSDP 让模型装进显存；剩余 GPU 全部用于数据并行。

---

## 问题背景与通信原语 [00:05 - 04:20]

训练大模型需要把单芯片扩展到多机多卡，主要动机有两个：

- **计算瓶颈**：单 GPU 算力有限，而顶级超算可以达到 exaflops 级别。
- **显存瓶颈**：模型参数、梯度和优化器状态无法全部放进一张 GPU，必须切分。

并行设计中的一个核心区分是：

- **节点内并行**：连接很快，可以承受通信昂贵的操作。
- **节点间并行**：连接较慢，必须使用通信更节省的策略。

课程在算法层面讨论并行，不深入到发包和硬件驱动，而是使用集合通信原语来记账，例如 all-reduce、all-gather、reduce-scatter。一个关键恒等式是：

$$
\text{all-reduce} \approx \text{reduce-scatter} + \text{all-gather}
$$

这个等价关系后面会直接解释为什么 ZeRO Stage 1/2 的显存节省几乎是“免费”的。![GPU计算扩展极限：单芯片推理性能10年提升超1000倍及预测性能发展曲线图](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0002_00-01-20.jpg)

![集合通信基础：All reduce、Broadcast、Reduce、All Gather 和 Reduce Scatter 的数据流向示意图](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0003_00-02-40.jpg)

课程把“新的计算单元”定义为整个数据中心，而不是单张 GPU；目标是在多机扩展时同时获得近似线性的显存扩展和计算扩展。![第一部分回顾：数据中心作为新计算单元及多机扩展的线性内存/计算目标](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0007_00-10-41.jpg)

---

## 硬件与网络拓扑：TPU 与 GPU 的设计哲学 [04:20 - 10:33]

TPU 和 GPU 的网络拓扑反映了不同的设计哲学：

- **TPU**：典型的是 **toroidal mesh**，芯片主要与邻居通信，拓扑规则、可简单扩展；每个节点的邻居数量不随规模增长。
- **GPU**：更接近 **all-to-all / fat tree**，低层 GPU 高速互联，再通过 pod、spine switch 逐层扩展；规模越大，通信拓扑越复杂。

这导致不同工作负载的适配性不同：

- 如果通信模式是规则的、主要访问邻居，TPU 的 mesh 很合适，可以用较低成本把单条链路做得更宽。
- 如果通信模式更随机，例如 MoE 中 token 被路由到不同专家，GPU 的 all-to-all 拓扑更灵活。

讲师也提到 TPU 新代际的变化：TPU8i/t 的 scale-out 网络“Virgo”更接近交换式、树状/全互联风格，可能反映了 MoE 推理和训练对 all-to-all 带宽的需求。![TPU 与 GPU 在通信层面的设计差异对比，包含拓扑结构图及 DGX SuperPOD 性能数据表](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0004_00-04-01.jpg)

![TPU8i/t 网络拓扑架构图，展示从单机到全球 WAN 的层级扩展](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0005_00-08-01.jpg)

另一个有趣的硬件权衡是华为 Ascend 910C Cloud Matrix 384：单芯片性能弱于 H200，但通过大量光纤/交换资源把 384 张卡连成一个大域，以“暴力通信”弥补单芯片劣势；代价是功耗非常高，约为等效 Nvidia 系统的四倍。这说明硬件设计可以在“能效/制造效率”和“暴力扩展”之间做不同取舍。![华为 Ascend 910C Cloud Matrix 384 与 Nvidia GB200 NVL72 的关键能力对比表](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0006_00-09-21.jpg)

---

## 数据并行与显存账本 [12:01 - 15:04]

最直观的并行方式是**数据并行**。以朴素 SGD 为例，全局 batch 为 $B$，把它切到 $M$ 张 GPU 上，每张卡计算 $B/M$ 个样本的梯度，然后同步梯度。

数据并行的特点：

- **计算扩展**：只要每张卡分到足够样本，计算可以近似线性扩展。
- **通信开销**：每个 batch 需要同步梯度，通信量约为 $2 \times \#\text{params}$。
- **显存扩展**：没有。每张 GPU 都保存完整模型、完整梯度和完整优化器状态。

显存问题比直觉更严重。每条参数通常需要保存：

- FP/BF16 模型参数：2 字节
- FP/BF16 梯度：2 字节
- FP32 master weights：4 字节
- Adam 一阶矩：4 或 2 字节
- Adam 二阶矩：4 或 2 字节

规则大约是**每个参数 16 字节、约 5 份权重副本**，其中优化器状态占大头。![朴素数据并行原理：SGD公式及B/M样本切分与通信开销分析](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0008_00-12-02.jpg)

![朴素数据并行架构示意图：数据集分发至多GPU并复制模型副本](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0009_00-13-22.jpg)

![朴素数据并行内存开销分析：每参数需5份副本共16字节](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0010_00-14-42.jpg)

---

## ZeRO 与 FSDP：切分优化器状态、梯度和参数 [16:02 - 25:50]

ZeRO 的核心思想是：显存中大部分开销来自优化器状态，因此可以把这些昂贵状态切分到不同 GPU 上。随着阶段推进，切分范围从优化器状态扩展到梯度，最后扩展到参数。

### ZeRO Stage 1：只切优化器状态

每张 GPU 仍然保存完整参数和完整梯度，但只保存一部分优化器状态，并负责更新对应的一片参数。

流程：

1. 每张 GPU 在自己的数据子集上计算完整梯度。
2. 对梯度做 **reduce-scatter**，每张 GPU 只收到自己负责更新那部分参数所需的梯度。
3. 每张 GPU 用本地优化器状态更新自己负责的参数切片。
4. 对更新后的参数做 **all-gather**，让所有 GPU 重新拿到完整参数。

通信量：一次 reduce-scatter 加一次 all-gather，等价于一次 all-reduce，因此与朴素数据并行相同，都是 $2 \times \#\text{params}$。显存收益基本免费。![ZeRO各阶段内存消耗对比图：从Baseline到P_os+g+p的显存节省效果](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0011_00-16-02.jpg)

![ZeRO Stage 1工作流程图：四步通信与参数更新过程](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0012_00-17-22.jpg)

![ZeRO stage 1与朴素数据并行在通信原语、通信开销及内存占用上的对比表格](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0013_00-18-42.jpg)

### ZeRO Stage 2：进一步切分梯度

Stage 2 连梯度也切分。难点是不能等整个梯度向量物化后再切，因为每张卡已经存不下完整梯度。

做法是利用反向传播的计算图：

- 反向传播逐层计算梯度；
- 每算完一层的梯度，就立即 reduce 到负责该片参数的 worker；
- 一旦某个梯度在后续反向图中不再需要，就立即释放。

这样可以在不增加通信量的情况下切分梯度。![ZeRO stage 2工作流程图解，展示多Rank间的梯度计算、释放及参数更新步骤](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0014_00-20-03.jpg)

### ZeRO Stage 3 / FSDP：参数、梯度、优化器状态全部切分

Stage 3 也就是 PyTorch 中的 **FSDP**，把参数、梯度和优化器状态全部切分。每张 GPU 在任何时刻只持有模型的一部分。

工作方式：

- 前向传播时，按层 **all-gather** 参数，算完该层后立即释放完整权重；
- 反向传播时，再次按层 all-gather 参数，计算梯度后立即 **reduce-scatter** 梯度，再释放权重；
- 最后更新本地负责的参数切片。

通信量：两次 all-gather 加一次 reduce-scatter，约为 $3 \times \#\text{params}$，是朴素数据并行的 1.5 倍。但 FSDP 通过**通信与计算重叠**把开销隐藏掉：

- 在计算当前层前向时，预取下一层参数；
- 在反向传播时，通信流与计算流并行；
- 如果计算足够大、网络足够快，FSDP 的 GPU 利用率可以接近单卡训练。

![ZeRO stage 3 (FSDP) 简化版工作流程图，展示前向、反向传播中的权重聚合与释放逻辑](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0015_00-21-23.jpg)

![FSDP/ZeRO stage 3 前向与反向过程中计算流与通信流的时序重叠示意图](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0016_00-22-43.jpg)

![ZeRO 各阶段（1/2/3）相对于数据并行的通信开销对比分析幻灯片](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0017_00-25-23.jpg)

一个常见误解是：FSDP 中每张 GPU 仍然会走完整个模型，只是不会同时持有完整参数；它不是流水线并行。通信操作数量随层数增加，但每次通信的数据量更小，总账仍然是对应参数大小。![ZeRO各阶段在8xA100 80G上可训练的最大参数量与每参数字节数公式对比表](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0018_00-27-57.jpg)

---

## 为什么还需要模型并行 [28:53 - 31:42]

数据并行有两个硬限制：

1. **它消耗 batch size**。数据并行的并行度不能超过全局 batch size；而 batch size 不能无限增大，因为存在 **critical batch size**：超过某个点后，增加 batch 元素的收益低于用同样计算量多走一步 SGD。
2. **它不解决激活内存**。ZeRO Stage 1/2 不切参数；Stage 3 切参数、梯度和优化器状态，但激活内存仍然可能非常大。对于大模型和长序列，激活值会超过参数内存。

因此需要更细粒度的模型切分。模型并行与 FSDP 的关键区别是：

- FSDP 主要让**参数**在 GPU 之间流动；
- 模型并行主要让**激活**在 GPU 之间流动。

课程覆盖三类模型并行：

1. 流水线并行：切分层；
2. 张量并行：切分矩阵；
3. 专家并行：切分 MoE 专家。![模型并行概述幻灯片，列出流水线、张量、专家三种并行类型](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0020_00-30-44.jpg)

---

## 流水线并行：切分层与气泡 [31:48 - 39:20]

流水线并行把不同层放到不同 GPU 上，前向传播传递激活，反向传播传递部分梯度。最朴素的调度会让同一时刻只有一张 GPU 工作，产生巨大的 **pipeline bubble**，GPU 利用率极差。![层间并行示意图，展示Layer 0-3分配到GPU 0-3及前后向传播路径](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0021_00-32-04.jpg)

解决办法是把一个 batch 切成多个 **micro-batch**，让不同 micro-batch 在不同阶段重叠执行。气泡占比大致为：

$$
\frac{n_{\text{stages}} - 1}{n_{\text{micro}}}
$$

因此需要较大的 batch size 才能把气泡压到接近零。![流水线并行甘特图，展示micro-batch的前向F和反向B计算及气泡时间](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0022_00-33-24.jpg)

流水线并行的优点：

- 相比 DDP 可以节省显存；
- 通信量只依赖激活，约为 $b \times s \times h$；
- 通信是点对点的，不是 all-to-all；
- 适合放在最慢的网络链路上，例如跨节点或多 pod。

因此实际系统通常把流水线并行用于节点间或更慢的互联。![流水线并行优势总结：相比DDP省显存，相比FSDP点对点通信且仅依赖激活值](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0023_00-34-45.jpg)

为了减少气泡，还有更复杂的调度：

- 给每张设备分配多个 stage，交错执行前向和反向；
- 使用更聪明的 micro-batch 调度；
- **Zero Bubble Pipelining**：把反向传播拆成两部分：
  - $B$：把激活的偏导继续往回传，位于关键路径上；
  - $W$：计算当前层权重的梯度，可以延后执行。

先尽快完成 $B$，再利用空隙填充 $W$，可以几乎填满流水线。代价是实现非常复杂。![流水线并行调度示意图：通过向每个设备分配多个阶段（Assign multiple stages）以用带宽换利用率](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0024_00-36-36.jpg)

![零气泡流水线（Zero bubble pipelining）原理图，展示MLP计算图及1F1B与ZB-H1/H2调度对比](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0025_00-37-25.jpg)

---

## 张量并行与序列并行 [39:27 - 51:53]

### 张量并行

张量并行沿“宽度”切分模型：把矩阵乘法拆成多个子矩阵乘法，再对部分和做 all-reduce。它本质上和矩阵 tiling 是同一个原语。

以一个 MLP 为例：

$$
Y = \mathrm{GeLU}(XA), \quad Z = \mathrm{Dropout}(YB)
$$

把 $A$ 按列切成 $[A_1, A_2]$，把 $B$ 按行切成 $[B_1; B_2]$，分到不同 GPU 上计算。

前向和反向存在对偶关系：

- 前向：输入复制，$f$ 是 identity；输出处 $g$ 是 all-reduce。
- 反向：$g$ 变成 identity；$f$ 处需要对部分导数做 all-reduce。

在 Transformer 块中：

- **列切分**：QKV 投影、MLP 的 up-projection；
- **行切分**：attention 输出投影、MLP 的 down-projection；
- **复制**：layer norm、非线性和 MoE router 等小模块通常完全复制，不值得切分。

![张量并行（Tensor parallel）中GPU子矩阵分配示意图，展示MLP层的列切分与行切分](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0026_00-40-05.jpg)

![Transformer块中张量并行切分策略图，对比MLP与自注意力层的行/列切分方式及吞吐量缩放](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0027_00-42-46.jpg)

张量并行非常“吃通信”：每个 matmul 都可能伴随 activation 大小的 all-reduce，而且非常频繁。因此 GPU 上通常只在节点内使用，最多约 8 卡，依赖 NVLink 等高速互联。  
在 TPU 上，由于 toroidal mesh 在规则通信模式下带宽很高，可以使用比 GPU 大得多的张量并行度；这会改变 TPU 与 GPU 在张量并行/流水线并行之间的选择。

与流水线并行相比：

- 张量并行没有 pipeline bubble，概念简单；
- 但通信量大得多：流水线每个 micro-batch 约 $b s h$ 点对点通信；张量并行每层约为 $8 b s h \cdot (n_{\text{devices}}-1)/n_{\text{devices}}$ 的 all-reduce 通信。![张量并行与流水线并行的优缺点对比幻灯片，含通信量公式](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0028_00-44-06.jpg)

### 激活内存与序列并行

实际训练中，显存不只是参数和优化器状态。用 PyTorch profiler 可以看到，激活值形成巨大的动态内存峰值；最大显存通常出现在反向传播开始之后，因为此时既要保存大量激活，又要存梯度。![训练过程中内存随时间变化的堆叠面积图，展示激活值等动态内存占用](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0029_00-45-26.jpg)

如果不做任何优化，每层激活内存约为：

$$
s b h \left(34 + \frac{5 a s}{h}\right)
$$

其中：

- $s$：序列长度
- $b$：micro-batch 大小
- $h$：hidden dimension
- $a$：attention head 数

$s b h$ 依赖是根本性的：每个序列位置、每个 batch 元素、每个 hidden 维度都需要存东西。第二项来自 attention 的二次项和 dropout，可以通过 FlashAttention 式的重计算去掉。![每层激活值内存计算公式及变量定义表](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0030_00-46-46.jpg)

张量并行可以切掉 MLP 和 attention head 部分：

$$
s b h \left(10 + \frac{24}{t} + \frac{5 a s}{h t}\right)
$$

其中 $t$ 是张量并行度。这样 MLP 的 24 份和 attention 的二次项都能随 $t$ 缩小，但 layer norm、dropout、残差输入等 10 份点式操作仍然完全复制，不会被张量并行切分。![张量并行下每层激活内存的计算公式及参数定义表](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0031_00-48-06.jpg)

**序列并行**用来处理剩下的 10 份开销：把这些轻量、计算量小的操作沿序列轴切分，而不是沿 hidden 轴切分。需要时通过 all-gather/reduce-scatter 物化激活，概念上很像 FSDP：平时以切分形式保存，用到时再收集。

这样可以让激活内存对并行度呈完全线性依赖。如果再对 attention 做重计算，可以去掉 $5as/ht$ 项，得到常见下界：

$$
\frac{34 s b h}{t}
$$

这是估算“模型能否装进 GPU”时非常有用的公式；总显存还需要加上参数、梯度和优化器状态。![序列并行架构流程图，展示Transformer层内各模块的并行切分策略](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0032_00-49-26.jpg)

---

## 专家并行 [53:21 - 60:58]

MoE 模型把 FFN 拆成多个专家，专家并行就是把不同专家放到不同设备上。它与张量并行类似：都是切分 MLP、都需要高带宽通信、都能减少激活内存；但专家并行通过路由 token 而不是切分矩阵来实现。

对于 MoE，通常优先使用专家并行而不是张量并行：

- 张量并行把矩阵切得太细会降低 matmul 尺寸和 GPU 利用率；
- MoE 中路由稀疏 token 比路由稠密的大矩阵激活更容易；
- 可以直接把 token 送到它需要的专家所在设备；
- 如果已经是 MoE，本来就需要专家切分。

专家并行的难点在于 **all-to-all dispatch**：每个 MoE 层都要把 token 路由到不同设备，而且对延迟非常敏感，因为计算在等待 token 到达。DeepSeek 的 DeepEP 和 NVIDIA 的 Hybrid EP 都是针对这一问题的低层优化库；DeepSeek 甚至使用未公开的 PTX 指令来加速网络通信。

组合数据并行和专家并行时也有约束。朴素做法是让 DP 和 EP 使用同一组 replica：先按 DP 切分 GPU，再在 DP 组内切分专家。这会限制 EP 的最大并行度，并约束 DP 与 TP 的交互。

另一个复杂性是：MoE 只改变 MLP，不改变 attention。因此 EP 只并行化 MLP，不并行化 attention。于是出现矛盾：

- attention 希望较高的张量并行度来切分矩阵；
- MLP 希望较低的张量并行度，避免和 EP 一起把矩阵切得太碎。

现代系统通过**解耦 attention 与 MoE 层的张量并行**来解决：attention 层和 MoE 层使用不同的 TP 配置，从而更灵活地组合 EP、TP 和 DP。

---

## 上下文并行 [61:00 - 61:51]

上下文并行（context parallel）或 ring attention 把长序列的激活切分到不同加速器上，并以 ring 的方式传递所需的 KV。它在 TPU mesh 上表现很好，主要用于长上下文扩展阶段和模型推理服务。课程没有展开细节，因为其概念与前面讨论的激活切分有重叠。

---

## 组合策略与 4D 并行 [61:52 - 68:06]

没有任何一种并行策略严格占优，每种都有代价：

- FSDP 显存收益好，但不解决激活内存，并且消耗全局 batch size；
- 张量并行能切激活，但需要高速网络；
- 流水线并行通信省，但有气泡且依赖大 batch；
- 专家并行适合 MoE，但路由复杂、延迟敏感。

因此需要组合使用。可以用“每层计算量 vs 每层通信量”的 roofline 分析来判断：

- 如果计算时间大于通信时间，通信可以被隐藏；
- batch size 较大时，FSDP 可能已经足够；
- batch size 变小时，FSDP 会变成通信瓶颈，需要加入张量并行把曲线往外推；
- 继续加入流水线、专家、上下文并行，就是所谓的 3D/4D 并行。

实际经验法则：

1. 先让模型装进显存：在高速互联内使用张量并行或专家并行，通常不超过单机 8 卡；
2. 如果还不够，使用流水线并行或 FSDP/ZeRO-3 继续切分；
3. 模型装下后，剩余 GPU 全部用于数据并行；
4. 如果数据并行导致 batch size 太小，使用梯度累积提高利用率。

Megatron 的实践指南也一致：

- 尽量最大化数据并行；
- TP/EP 保持在 NVLink/单机范围内；
- 跨多节点使用流水线并行；
- MoE 优先专家并行；
- 长序列使用上下文并行。

---

## 大规模训练案例 [68:07 - 78:27]

课程最后用多个公开训练案例说明这些策略如何落地。

- **Olmo / Dolma 7B**：完全使用 FSDP 训练。对于 7B 级别模型，FSDP 可以扩展到很多 GPU，是简单有效的方案。
- **DeepSeek V1**：数据并行 ZeRO Stage 1，加上张量并行、序列并行和流水线并行，是经典组合。
- **DeepSeek V3**：MoE 模型，使用流水线并行和专家并行替代张量并行；专家并行达到 64 路，通过把 8 台机器组成一个 EP 域，并复用流水线调度技巧来避免低利用率。
- **Yi**：ZeRO Stage 1 + 张量并行 + 流水线并行，属于经典的数据/张量/流水线组合；MoE 版本则用专家并行替代张量并行。
- **Llama 3 405B**：密集模型，预训练阶段使用 TP=8、CP=1、PP=16、DP=128；长上下文扩展阶段提高 context parallel、降低 data parallel。训练中 GPU 故障 148 次，说明大规模训练还需要冗余和容错。
- **Gemma 2**：使用 FSDP + 张量并行 + 序列并行，不使用流水线并行；这体现了 Google 在 TPU 大 toroidal mesh 上做大张量并行的思路。
- **Mixtral 8x22B**：EP=8、PP=4，并对 attention 层额外使用 TP=4。
- **Nemotron 3 Super**：采用较多专家并行，并在长上下文场景加入上下文并行。
- **Qwen 3**：接近 DeepSeek 配方，EP=32、PP=8、TP=2，用较小 TP 切分 attention 矩阵。

NVIDIA 的 Megatron Bridge 仓库公开了许多模型的推荐并行配置和基准测试；即使同在张量并行内部，不同子配置也会显著影响性能，说明系统集成仍然很重要。

---

## 总结与经验法则 [78:49 - 80:10]

这堂课的最高层结论是：现代语言模型训练需要**多 GPU、多节点甚至多数据中心**的并行思维。要同时利用快链路、慢链路、batch size 等资源，把多种并行策略组合起来，才能获得接近满负荷的硬件利用率。

从公开训练案例中可以看到一些共同规律：

- 尽可能最大化数据并行域；
- 张量并行通常保持在 8 以内；
- 专家并行在 MoE 时代可以变得很大，部分原因是 DeepSeek V3 等系统工作的推动；
- 流水线并行用于跨节点扩展；
- 长上下文训练提高上下文并行。

---

## 要点回顾

- 并行训练的两大动机：单芯片算力不足、模型显存不足。
- 节点内链路快，适合通信密集的张量/专家并行；节点间链路慢，适合通信较省的流水线并行。
- 数据并行计算扩展好，但通信量为 $2 \times \#\text{params}$，且不节省显存。
- 模型显存不只是参数：优化器状态、梯度、激活值都可能成为瓶颈；经验上每参数约需 16 字节。
- ZeRO Stage 1 切优化器状态，通信量与 DDP 相同；Stage 2 进一步切梯度；Stage 3/FSDP 切参数、梯度和优化器状态，通信量约 1.5 倍，但可通过通信计算重叠隐藏。
- 流水线并行切分层，通信量小但会产生 bubble；micro-batch 和 zero-bubble 调度可以缓解。
- 张量并行切矩阵，通信量大，适合节点内高速互联；在 TPU mesh 上可以使用更大 TP 度。
- 激活内存约为 $s b h(34 + 5as/h)$；张量并行后为 $s b h(10 + 24/t + 5as/(ht))$；序列并行可让激活内存近似线性扩展；重计算后下界约为 $34sbh/t$。
- 专家并行适合 MoE，优先于张量并行；难点是低延迟 all-to-all 路由，且需要处理 DP/EP/TP 的组合约束。
- 实际配方：先用 TP/EP 和 PP/FSDP 让模型装进显存，再把剩余 GPU 全部用于数据并行；batch 不够时用梯度累积。
- 公开案例普遍最大化 DP、TP 通常不超过 8、MoE 模型大量使用 EP、长上下文使用 CP。
