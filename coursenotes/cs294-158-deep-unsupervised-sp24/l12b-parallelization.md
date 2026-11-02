---
title: "第 12b 讲 · 并行化（客座讲师 Wilson Yan）"
description: "大 Transformer 无法塞进单张 GPU，训练必须把参数、激活和 batch 切分到多设备。Transformer 的计算几乎全是矩阵乘法，因此并行的核心是矩阵乘分片，以及随之而来的 all-gather、all…"
---

# 第 12b 讲 · 并行化（客座讲师 Wilson Yan）

> **CS294-158 深度无监督学习 · L12b** ｜ 讲者 Pieter Abbeel ｜ 40 分钟 ｜ [原视频](https://www.youtube.com/watch?v=Hr2FWHBuNXs)

## 核心结论

大 Transformer 无法塞进单张 GPU，训练必须把参数、激活和 batch 切分到多设备。Transformer 的计算几乎全是矩阵乘法，因此并行的核心是矩阵乘分片，以及随之而来的 all-gather、all-reduce 等通信。常见路线包括数据并行、FSDP 和张量并行：FSDP 通过逐层 all-gather 权重节省内存并可重叠通信，TP 则切权重/激活以支持小 batch。工程上通常优先最大化 FSDP，TP 用于单卡放不下或长序列场景；PyTorch 需显式写通信，JAX 可自动推导。

## 课程定位与硬件背景 [00:02 - 02:58]

这是课程最后一部分，主题是 Transformer 的并行化：像 GPT-3 这样 175B 参数的模型无法放在单卡上，需要几十、几百甚至上千张 GPU/TPU 一起训练。讲者先对比 TPU 与 GPU 的几个关键参数：FLOPS、HBM 大小与带宽、设备间互连带宽。A100 的 HBM 为 40GB 或 80GB，TPU 视版本约为 16–32GB；互连带宽尤其重要，因为多设备训练必须通信，带宽会直接决定模型能开多大、数据能跑多少。

![讲座大纲幻灯片，列出双向多模态模型、编码器-解码器模型、仅解码器模型、多模态生成及Transformer并行化五个主题](/coursenotes/cs294-158-deep-unsupervised-sp24/l12b-parallelization/kf_0001_00-00-00.jpg)

Transformer 之所以适合这些硬件，是因为其计算约 90%–99% 都是矩阵乘法。GPU 由很多较小的计算单元组成（如大量 8×8×8 ALU），TPU 则用脉动阵列以更大块的方式处理矩阵乘（如 8×128×128）。这形成软硬件循环：Transformer 以矩阵乘为主、容易优化，硬件便加速矩阵乘，模型也更倾向使用矩阵乘；反之，矩阵乘较少的模型在纯硬件层面可能更慢。

![TPU与GPU对比幻灯片，展示FLOPS、HBM大小与带宽、互连等参数及芯片架构示意图](/coursenotes/cs294-158-deep-unsupervised-sp24/l12b-parallelization/kf_0002_00-00-40.jpg)

![GEMM幻灯片，对比GPU的多个8x8x8 ALU与TPU的8x128x128脉动阵列结构](/coursenotes/cs294-158-deep-unsupervised-sp24/l12b-parallelization/kf_0003_00-01-59.jpg)

## 矩阵乘法分片：并行化的基本构件 [03:19 - 10:43]

设矩阵乘法为：

$$
Y = XA
$$

把矩阵切到多设备上有两种基本模式。

**切分求和维，得到部分和。** 例如把 $X$ 按列切、$A$ 按行切：

$$
X=[X_1,X_2],\quad
A=\begin{bmatrix}A_1\\A_2\end{bmatrix}
$$

则：

$$
Y = X_1A_1 + X_2A_2
$$

每台设备只计算一个部分和，最后需要对各设备结果做 all-reduce 求和，才能得到完整 $Y$。all-reduce 会把同一张量在所有设备上求和，并把结果保留在每台设备上。

**切分输出维，得到输出分块。** 例如只把 $A$ 按列切：

$$
A=[A_1,A_2],\quad
Y=[XA_1,XA_2]
$$

每台设备直接算出输出的一部分，最后通过 all-gather 把各设备的分块收集起来，拼成完整输出。通信带宽在这里很关键：通信快，等待少；通信慢，就会成为瓶颈。

反向传播与前向通信大致互为镜像：前向做 split，反向可能需要 gather；前向做 all-gather，反向可能需要切分梯度并做相应归约。分片（sharding）与并行化在这里基本是同一件事：把一个大矩阵乘拆到多设备，既可能加速，也分摊参数内存。

## MLP 分片与 f/g 通信算子 [10:43 - 14:08]

MLP 的结构可以写成：

$$
Y=\mathrm{GELU}(XA),\quad
Z=\mathrm{Dropout}(YB)
$$

其中 $A$ 把维度从 $D$ 映射到 $4D$，$B$ 再从 $4D$ 映射回 $D$。

![MLP分片（MLP sharding）的两种切分策略：按列切分X并按行切分A，以及按列切分A](/coursenotes/cs294-158-deep-unsupervised-sp24/l12b-parallelization/kf_0004_00-11-12.jpg)

幻灯片给出两种切分策略：

- **方法一：** 按列切 $X$、按行切 $A$。每个设备得到部分和，但 GELU 是非线性的，必须在 GELU 之前先同步/求和，否则不能直接对部分和做 GELU。
- **方法二：** 只按列切 $A$，不切 $X$ 的收缩维。这样每台设备可以直接计算自己那部分输出的 GELU，不需要在 GELU 前同步。

视频强调实践中更常用的是减少中间通信的做法：$X$ 复制，$A$ 按列切为 $A=[A_1,A_2]$，每台设备计算：

$$
Y_i=\mathrm{GELU}(XA_i)
$$

然后第二层权重 $B$ 按行切为 $B=\begin{bmatrix}B_1\\B_2\end{bmatrix}$，由于 $Y_1,Y_2$ 已经按对应维度分片，可以直接计算：

$$
Z_i=Y_iB_i
$$

最后做一次 all-reduce：

$$
Z=Z_1+Z_2
$$

这样中间不需要额外 all-gather，只在 MLP 末尾通信一次。

![MLP分片前向与反向传播中f和g算子的示意图，展示GELU与Dropout的分片计算流程](/coursenotes/cs294-158-deep-unsupervised-sp24/l12b-parallelization/kf_0005_00-11-51.jpg)

图中的 $f$ 与 $g$ 表示通信算子：$f$ 在前向是 identity、反向是 all-reduce；$g$ 在前向是 all-reduce、反向是 identity。权重矩阵在初始化时就可以预分片并常驻各设备，不需要为了“共享”而反复通信。

## 多头注意力的分片 [15:09 - 18:55]

注意力分片与 MLP 结构非常相似。输入 $X$ 复制，QKV 投影矩阵按注意力头切分：若注意力有 4 个头、使用 2 台设备，则每台设备存储并计算 2 个头。不同头在注意力计算阶段彼此独立，因此各设备可以独立完成：

- 计算 $QK^\top$
- softmax
- dropout
- 乘以 $V$

得到各自的 $Y_1,Y_2$。

![Attention sharding 幻灯片，对比缩放点积注意力与多头注意力的计算流程图](/coursenotes/cs294-158-deep-unsupervised-sp24/l12b-parallelization/kf_0006_00-15-09.jpg)

注意力的输出投影会混合不同头的信息，因此输出 dense 层也需要分片：每台设备处理自己那几个头对应的输出部分，最后通过 all-reduce 汇总。整体流程与 MLP 的“列切权重—本地激活—行切输出—all-reduce”几乎一致。

## 数据并行与全分片数据并行 [19:00 - 29:39]

### 数据并行（DP）

数据并行把 batch 维切到多设备，权重在每个设备上复制。若张量形状为 `batch × E`，权重为 `E × E`，则每台设备拿到不同的 batch 行，但都持有完整的 `E × E` 权重。

例如全局 batch 为 4、使用 2 台设备，则每台设备 batch 为 2。这样可以减小每台设备的 batch 内存，并可能缩短单步时间；但每台设备仍存储完整模型参数，参数内存没有分摊。

### 全分片数据并行（FSDP）

FSDP 仍然是数据并行：每台设备看到自己的 batch；但权重不再复制，而是分片存储。前向计算某一层时，先对该层权重做 all-gather，让每台设备临时拿到完整权重，完成矩阵乘后再把权重切分回去（reshard）。关键优势是：每台设备只需要在某一时刻持有一层的完整权重，而不是整个模型的完整权重，因此对很深的网络能显著节省内存。

FSDP 还能把通信与计算重叠：在计算当前层时，可以同时 all-gather 下一层权重。只要当前层计算时间足够长，下一层权重在需要时已经准备好，就不会产生额外等待。

视频用算术强度说明所需 batch 大小。对 `(batch × E) × (E × E)` 的矩阵乘：

$$
\text{FLOPs}=2\cdot B\cdot E^2
$$

其中 $B$ 是每芯片 batch。all-gather 需要通信整个权重矩阵，半精度下约为：

$$
\text{通信字节}\approx 2E^2
$$

因此算术强度为：

$$
\frac{\text{FLOPs}}{\text{通信字节}}\approx B
$$

也就是说，每芯片 batch 越大，计算相对通信越多，通信越容易被掩盖。视频给出的经验是每芯片 batch 需要达到约千级，才能让 FSDP 的通信基本不成为瓶颈。若使用 layer norm（Transformer 常见），不需要跨激活维额外通信；若使用 batch norm，则需要同步均值和标准差，会增加开销。

## 张量并行及其与 FSDP 的组合 [29:47 - 32:44]

张量并行（TP）不切 batch，而是切模型权重和/或激活。它有两种常见做法：

1. **只切权重：** 不切 batch，切 QKV 投影和输出投影权重。这与前面 MLP/注意力分片的方式一致。
2. **同时切激活：** 除了切权重，还切激活维，产生部分和；然后用 reduce-scatter 汇总并重新分片。reduce-scatter 可以理解为“先 all-reduce 得到完整结果，再 scatter/reshard，使每台设备只保留自己那部分”。

TP 与 FSDP 不冲突，可以组合成二维设备网格。例如 16 台设备可以排成 `FSDP 4 × TP 4`：全局 batch 为 4 时，每个 FSDP rank 分到 batch 1，再沿 TP 维把这 1 个样本切到 4 台设备上计算。设备数越多，组合方式越复杂，但原理仍是沿不同维度分片。

## 工程实践、选型与工具 [32:44 - 39:31]

目前如何选择 sharding 方式仍主要靠手动配置，未来可能优化，但还不是自动完成的事情。

**PyTorch 与 JAX 的差异：**

- PyTorch 需要显式编写通信；FSDP 的实现很复杂，尤其要让通信和计算重叠时。
- JAX 会把多设备视为一个逻辑设备，用户定义矩阵的 sharding 约束，编译器在编译图时自动插入所需 collective。这样写起来简单，但自动插入的 collective 可能并非最优，甚至插入不想要的通信，调试也更困难。

**模型实现：** 对 HuggingFace 模型，通常希望已有现成并行实现；Llama 等流行模型有第三方实现支持 tensor parallelism，否则往往需要自己实现。

**collective 的同步语义：** 各设备运行相同的函数调用序列，必须在大致相同的点调用 all-reduce 等 collective；较快的设备会阻塞等待较慢的设备。

**FSDP 与 TP 的选型对比：**

- FSDP 仍要求每台设备至少有 batch 1，因此设备数很多时，全局 batch 也必须很大。
- TP 可以让多台设备共享同一个 batch，甚至 batch size 为 1，适合单条长序列放不进单卡的情况。
- TP 中的 all-reduce 往往是阻塞的，难以与计算重叠；并且某些激活会被复制，内存效率不如 FSDP。
- 一般规则是尽量最大化 FSDP，只有在单卡放不下、需要更小 batch 或长序列场景时才使用 TP。

**与其他并行方式共存：** 注意力的 head 分片与 ring attention 的序列长度分片并不冲突，可以同时使用。

## 要点回顾

- 大模型训练必须多设备并行；Transformer 以矩阵乘法为主，因此并行核心是矩阵乘分片与通信。
- 矩阵乘分片有两种基本结果：切求和维得到部分和，需要 all-reduce；切输出维得到输出分块，需要 all-gather。
- MLP 可采用“复制 $X$、列切 $A$、本地 GELU、行切 $B$、末尾 all-reduce”的方式，减少中间同步。
- 多头注意力按 head 切分 QKV 投影；各头独立计算，输出投影混合 heads，最后 all-reduce。
- 数据并行切 batch、复制权重；FSDP 进一步把权重分片，逐层 all-gather 并在计算后 reshard。
- FSDP 的通信可与计算重叠；所需 batch 大小由算术强度决定，视频给出的经验是每芯片 batch 约千级。
- 张量并行切权重/激活而不切 batch，可用 `FSDP × TP` 二维网格组合；TP 支持 batch size 1 跨多设备。
- 选型上通常尽量最大化 FSDP；TP 用于单卡放不下或长序列场景。PyTorch 需显式通信，JAX 可自动推导 collective。
