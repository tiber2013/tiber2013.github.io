---
title: "第 5 讲 · GPU 与 TPU：硬件与性能模型"
description: "这讲把 GPU 视为一台为吞吐量而设计的并行机器：大量 SM、warp 和线程执行相同指令，而计算能力的增长远快于内存带宽，因此性能优化的核心是减少并复用全局内存访问。课程先建立 SM、SIMT、warp 与内存层级的心…"
---

# 第 5 讲 · GPU 与 TPU：硬件与性能模型

> **Stanford CS336 语言模型从零构建 · L05** ｜ 讲者 Stanford Online ｜ 79 分钟 ｜ [原视频](https://www.youtube.com/watch?v=izZba4UA7iY)

## 核心结论

这讲把 GPU 视为一台为吞吐量而设计的并行机器：大量 SM、warp 和线程执行相同指令，而计算能力的增长远快于内存带宽，因此性能优化的核心是减少并复用全局内存访问。课程先建立 SM、SIMT、warp 与内存层级的心智模型，再给出六个手段：控制发散、低精度、算子融合、重计算、合并访存、tiling。最后用这些原理解释矩阵乘法吞吐曲线中的 roofline、对齐与 wave quantization，并拆解 FlashAttention 如何用 tiling、online softmax 和重计算实现高效注意力。

## 课程定位与三条主线 [00:05 - 04:10]

课程从神经网络部分转入系统部分，后续会讨论 GPU、并行化和推理。讲者强调系统部分的可推理性和逻辑性，但初次接触 GPU 时会觉得它像“魔法”。本讲目标是让 CUDA/GPU 不再神秘，并能理解 GPU 何时变慢、如何写出快速算法。

![课程大纲幻灯片，分为GPU原理、性能理解和FlashAttention拆解三个部分](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0005_00-03-56.jpg)

课程分为三部分：

1. **GPU 深入**：硬件模型与编程模型；
2. **理解 GPU 性能**：让 ML workload 变快的若干技巧；
3. **综合案例**：拆解 FlashAttention。

讲者推荐了 Horace He 的博客、CUDA MODE 群组，以及 TPU/GPU book 作为延伸资源。

## 从串行频率到并行吞吐：为什么需要 GPU [04:14 - 07:27]

传统 CPU 性能提升依赖时钟频率和 Dennard scaling，但这条路在 2000 年代后逐渐失效：晶体管继续变小，并不自动意味着时钟更快。于是计算转向并行扩展：不是让单个任务跑得更快，而是让更多任务同时执行。

GPU 的浮点算力在过去多年呈超指数增长。除了制程和并行度，数值表示、专用指令、稀疏性等也贡献了巨大增益；其中 V100 时代引入的 Tensor Core 是关键硬件创新。

![1996-2023年硬件FLOPS、DRAM带宽及互连带宽的扩展趋势对比图](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0015_00-26-37.jpg)

但更重要的长期趋势是：**计算增长快于内存带宽和互连带宽**。因此现代 GPU 优化很大程度上是内存优化，越往后越难把计算单元“喂饱”。

## GPU 硬件模型：SM、执行单元与内存层级 [07:31 - 13:20]

CPU 优化少量快速线程，控制单元大、缓存复杂，适合分支和低延迟串行执行；GPU 优化大量线程，拥有许多轻量级 ALU，单任务延迟可能很高，但聚合吞吐量巨大。

![CPU与GPU架构对比图，展示CPU侧重控制单元与缓存而GPU侧重大量ALU阵列的结构差异](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0007_00-07-37.jpg)

GPU 的基本计算单元是 **SM（streaming multiprocessor）**。每个 SM 类似一个独立核心，内部有多个 streaming processor，可并行执行线程；GPU 由大量 SM 组成，例如课程图示中 GA100 有 128 个 SM。

![GPU执行单元解剖图，展示SM内部SP结构及GA100芯片的128个SM布局](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0008_00-09-11.jpg)

内存层级是理解 GPU 性能的关键：

- **寄存器**：最快、最局部；
- **共享内存 / L1 cache**：距离 SM 近，延迟低；
- **L2 cache**：更远、更慢；
- **全局内存 / HBM**：容量大但延迟高，通常说的“GPU 有多少 GB 显存”指的就是它。

课程给出的 A100 延迟表中，全局内存约 290 cycles，L2 约 200 cycles，L1 约 33 cycles，共享内存约 23/19 cycles。共享内存和 L1 都是 SRAM，但共享内存是可编程的，L1 cache 由硬件自动管理。SRAM 更快但昂贵、耗电，因此实际加速器不会全部用 SRAM，而是采用内存层级并依赖程序有效利用。

![GPU内存层次结构图，包含不同内存类型的延迟对比表及GA100芯片物理布局](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0009_00-10-20.jpg)

核心结论：**一旦访问超出共享内存，速度就会明显变慢；减少全局内存读取是贯穿本讲的主线。**

## GPU 执行与内存模型：线程、Block、Warp、SIMT [13:26 - 17:13]

GPU 执行模型有三个关键对象：

- **Thread**：执行工作的轻量线程；同一 warp 内线程遵循 SIMT，执行相同指令但输入不同。
- **Block**：一组线程，保证运行在同一个 SM 上，因此可以使用该 SM 的共享内存。
- **Warp**：32 个连续编号线程组成的调度单位；调度器决定下一个执行哪个 warp。

![GPU执行模型示意图，解释线程(Thread)、块(Block)和Warp的层级关系及SIMT执行机制](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0011_00-14-25.jpg)

SIMT 简化了编程，但也带来效率约束：如果代码里有分支，warp 可能要把两个分支都走一遍，不活跃线程被 mask 掉，形成控制发散。

内存模型方面，device code 可以读写寄存器、线程局部内存、block 共享内存、grid 全局内存和只读常量内存；host 与 device 之间也可以传输数据。

![GPU内存层级结构图，展示寄存器、共享内存、全局内存和常量内存的访问权限](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0013_00-15-51.jpg)

再次强调：**共享内存之外的数据移动很慢；把 block 组织好、减少全局内存读写，是后续优化的核心。**

## TPU 与 GPU：趋同演化 [17:18 - 22:36]

TPU 与 GPU 在高层次上非常相似：都有矩阵乘法单元、并行向量单元、控制逻辑，以及“慢速 HBM + 快速本地共享内存”的层级。这是面向 ML 加速器的趋同演化。

主要区别在尺寸与灵活性：

- GPU 有很多较小的 SM 和矩阵乘单元，编程更灵活；
- TPU 用更少、更大的矩阵乘单元，更偏向大矩阵乘法；
- 讲者举例：某代 TPU 只有两个大计算单元，而 GPU 有很多 SM；TPU 的矩阵乘单元更大更少，batch size 太小时甚至无法充分利用。

一个容易混淆的命名：TPU 把它的 streaming multiprocessor 叫 **tensor core**，而 GPU 把矩阵乘单元叫 **tensor core**。需要根据上下文区分。二者底层都常用 systolic array 做矩阵乘。

讲者指出，TPU 与 GPU 更大的差异其实在网络互连，而不是单芯片内部。由于核心概念高度对应，本讲关于 GPU 的优化思想大多也能迁移到 TPU。

## GPU 的优势与性能瓶颈 [22:41 - 30:07]

GPU 成功的原因包括：容易通过增加 SM 扩展吞吐；编程模型看似底层但本质是 SIMT/SIMD 式；线程非常轻量，可被调度器快速切换，从而在部分任务停顿时仍保持吞吐。

Tensor Core 出现后，矩阵乘法成为 ML 中的“特权操作”：matmul 的吞吐可比其他浮点操作高 10 倍以上。因此，任何希望随算力扩展的 ML 架构，通常都会包含矩阵乘法。

![不同GPU型号上矩阵乘法与非矩阵运算的TFLOPS性能对比折线图](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0014_00-24-54.jpg)

但随着计算与内存差距扩大，内存和通信瓶颈越来越突出。课程回顾：GPU 大规模并行、计算（尤其 matmul）增长快于内存、必须尊重内存层级。

![GPU工作原理回顾幻灯片，包含GPU架构示意图、计算与内存扩展对比图及内存访问延迟表](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0016_00-27-32.jpg)

## 性能目标：roofline 与矩阵乘法吞吐曲线 [30:07 - 33:00]

Part 2 的目标是解释一张“方阵矩阵乘法吞吐 vs 矩阵维度”的曲线。总体上矩阵越大、工作量越多，吞吐越高；但曲线上有奇怪的下凹和周期模式。

roofline 模型给出基本形状：

- 低算术强度区域是**内存受限**，吞吐随工作量增加而上升；
- 足够高的算术强度后进入**计算受限**平台区，继续增加工作量不再提升吞吐。

算术强度可写为：

$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes}}
$$

优化目标是提高算术强度，让程序位于平台区。随后课程列出六个技巧：

1. 控制发散；
2. 低精度计算；
3. 算子融合；
4. 重计算；
5. 合并访存；
6. tiling。

![GPU加速的六大策略列表，包括控制发散、低精度计算、算子融合等](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0019_00-32-46.jpg)

其中第一个不是内存瓶颈，其余大多围绕减少内存移动。

## 技巧一：控制发散 [33:00 - 34:43]

GPU 的 warp 内线程执行相同指令。遇到 `if` 时，CPU 通常只走一个分支；GPU 则可能让 warp 依次执行两个分支，不属于当前分支的线程被 mask、空转。这就是**控制发散**。

```c
if (threadIdx.x < 4) {
  A;
  B;
} else {
  X;
  Y;
}
Z;
```

![GPU SIMT模型架构图及条件分支导致的线程发散（Control Divergence）时序示意图](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0020_00-34-26.jpg)

因此 GPU 代码常避免分支：例如 ReLU 不用 `if` 更新一半元素，而用掩码或乘法把负值置零。分支会带来计算空洞，降低吞吐。

## 技巧二：低精度计算 [34:53 - 47:42]

降低数值精度可以直接减少需要移动的字节数。例如 FP32 读写一个数需要 4+4=8 字节，而一次只做少量 FLOP，算术强度很低；降到 16 位或 8 位后，每 FLOP 的字节数减半或更少。

![低精度计算收益分解幻灯片，列出数值表示、复杂指令、工艺、稀疏性带来的加速倍数及单芯片推理性能十年增长曲线](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0021_00-35-09.jpg)

现代 Tensor Core 的低精度矩阵乘并不是所有步骤都低精度：输入先 downcast，乘积用全精度，部分和用 FP32 累加，输出可为 FP32。难点在于判断哪些操作必须保持高精度：matmul 的权重和激活可以低精度，softmax、指数等可能仍需 FP32 或 BF16。低精度训练是多年 empirical 工作积累的结果。

FP8 之后没有统一格式：常见有 E4M3（更多 mantissa）和 E5M2（更多 exponent），适用于不同场景。

![低精度前沿幻灯片，对比FP8不同权衡与MXFP8多缩放因子结构及前向反向传播流程](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0023_00-38-01.jpg)

更激进的是 **MXFP8**：不再每个矩阵只有一个缩放因子，而是每 32 个元素一个缩放因子；缩放因子本身是 FP8/E8M0，只有 exponent，因此是 2 的幂。好处是能适应矩阵内部不同区域的数值范围。代价是**转置变得不平凡**：转置后缩放因子的分组模式改变，需要重新量化。实际实现中会为每个量化矩阵同时保存原矩阵和转置版本，以避免运行时重排。

FP4 如 MXFP4 更极端：可表示的值只有 -6 到 6，每 16 个元素一个缩放因子，缩放因子用 E4M3。讲者认为 FP4 训练尚未在大模型中广泛成功，但可能是下一代方向。

低精度的收益并非线性：FP8 矩阵乘可能带来 20%–30% 或更多节省，但由于量化/反量化开销，不会直接变成 2 倍加速。首层和末层通常更难量化，末层因为直接决定大量 loss，容易不稳定。

## 技巧三：算子融合 [47:42 - 50:11]

把 GPU 比作工厂：全局内存是仓库，SM 是工厂，内存带宽是传送带。若每个小操作都单独读写全局内存，传送带会成为瓶颈。算子融合就是把多个操作合成一个 kernel：从全局内存读一次，在 SM 内完成全部计算，再写回一次。

例如 `sin(x)^2 + cos(x)^2` 的朴素计算图会为 sin、cos、平方、加法分别产生 kernel，每次都读写全局内存；融合后只需一个 kernel。简单的融合可由编译器自动完成，例如 `torch.compile` 和 JAX 的编译器等会把这类基本操作 fuse 成单个 CUDA 调用；更复杂的融合可能需要手动干预。

## 技巧四：重计算 [50:12 - 52:50]

反向传播通常需要保存前向激活。但从系统角度看，如果计算很便宜而内存访问很贵，可以丢弃部分激活，在反向时重新计算，用计算换内存。

以 `x -> sigmoid -> sigmoid -> sigmoid -> out` 为例：保存所有激活时，前向和反向合计约 8 次内存读写；丢弃中间激活、反向时重新前向计算，则降为约 5 次内存访问。计算量增加，但全局内存流量减少。

这就是 activation checkpointing / recomputation 的核心思想。

## 技巧五：合并访存（Coalescing） [52:52 - 57:51]

DRAM 按 burst 读取：访问某个地址时，同一 burst 内的连续数据可能被一起取回，例如一次可取约 128 字节。若一个 warp 的线程访问落在同一 burst 内，就是 **coalesced**，能充分利用内存带宽。

行主序矩阵中，若线程跨行读取同一列，会落到不同 burst，读一个 tile 可能触发多倍内存流量；让线程沿连续地址读取，或通过 padding 对齐 burst 与 tile 边界，可以显著改善。

因此矩阵的行/列访问顺序、对齐和 padding 都会影响实际性能。

## 技巧六：Tiling [57:52 - 65:09]

Tiling 是本讲最重要的优化思想：把矩阵切成小块，把 tile 加载到共享内存，然后在共享内存中反复读写和计算，最后写回全局内存。

朴素 $n \times n$ 矩阵乘中，每个输入元素可能被从全局内存读取 $n$ 次。若 tile 大小为 $T$，每个输入从全局内存读取约 $n/T$ 次，再在 tile 内被访问 $T$ 次：

$$
\text{global reads per input} \approx \frac{n}{T}
$$

当 $T=n$ 时，每个输入只需从全局内存读一次，之后都在快速共享内存中复用。这能带来约 $T$ 倍的全局内存访问减少。

![GPU上矩阵乘法性能曲线图，标注了分块、计算强度和波量化现象](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0018_00-30-09.jpg)

但 tile 大小并不简单：矩阵尺寸、共享内存容量、coalescing、burst 对齐都会影响最优 tile。PyTorch 的 `max-autotune` 会在编译时尝试不同 tile 配置并 benchmark。矩阵尺寸若不能整除 tile，会产生大量空 tile；若 tile 读取跨 burst 边界，则需要额外读取，通常用 padding 对齐。

## 回到神秘曲线：对齐与 wave quantization [65:10 - 69:20]

现在可以解释矩阵乘法吞吐曲线中的多个现象：

- **roofline 对角线**：算术强度不足导致内存受限；
- **按整除性着色的多条线**：能被 16、32 等较大数整除的矩阵更容易获得 coalesced、对齐的 tile 读取；只能被 1 或 2 整除的矩阵吞吐明显差；
- **周期性下凹**：矩阵维度变化改变 tile 数量和对齐方式。

一个典型例子是维度从 1792 增到 1793：使用 $256 \times 128$ 的 tile 时，tile 数从 98 变为 120。98 个 tile 可以一次装进 GPU 的 SM 中并行处理；120 个则超过可用 SM 数，需要多跑一轮，第二轮只有少量 SM 工作，大量 SM 空闲，吞吐骤降。这称为 **wave quantization**。

![GPU上矩阵乘法性能曲线图，标注了分块、计算强度和波量化现象](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0018_00-30-09.jpg)

因此，矩阵维度、tile 大小、SM 数量和内存对齐共同造成了看似“神秘”的性能模式。实践上通常希望矩阵尺寸是 2 的幂，并尽量能被 32 整除。

## 小结：一切围绕内存 [69:22 - 70:05]

让 ML workload 在 GPU 上变快的核心是：

- 合并访存，减少内存事务；
- 算子融合，减少中间读写；
- tiling，把数据搬到共享内存并复用；
- 低精度与重计算，用其他资源换内存流量。

本质上都是：**内存是瓶颈，减少和复用内存访问。**

## 案例：用这些原理解剖 FlashAttention [71:50 - 77:14]

FlashAttention 是系统优化的典型案例：从 PyTorch 的朴素 attention 实现，变成一个高度融合的 kernel，显著降低延迟和 HBM 流量，使更大上下文的 attention 可行。

![课程大纲幻灯片，包含GPU性能瓶颈分析图与FlashAttention算法结构及对比图](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0003_00-01-09.jpg)

attention 主要包含 $QK^\top$、softmax 和与 $V$ 的矩阵乘。矩阵乘可以 tiling，但 softmax 是全局操作，似乎把不同 tile 绑在一起，不能直接分块。

关键技巧是 **online softmax**：按 block 逐个计算 softmax，维护当前最大值和指数 running sum；遇到更大的最大值时，更新最大值并对已有累加结果做修正。这样不必先看到全部元素，也能逐 tile 得到正确的 softmax 归一化。

于是 FlashAttention 可以：

1. 对 $QK^\top$ 做 tiled matmul，结果留在 SRAM；
2. 在每个 tile 上计算指数和 online softmax 的 running max / sum；
3. 继续处理下一个 tile，保留少量统计量；
4. 最后与 $V$ 做 tiled matmul，并在结束时归一化。

反向传播时，不保存 $n^2$ 大小的 attention 激活，而是按 tile 重新计算。因此 FlashAttention 同时使用了 **tiling、online softmax 和重计算**。

## 总结 [77:16 - 78:27]

- 理解硬件，尤其 GPU 的并行与内存层级，是理解性能决策的基础；
- 当前算力增长使 matmul 成为核心操作，而计算与内存的差距要求仔细设计数据移动；
- FlashAttention 这类算法说明：硬件感知的算法与架构设计对现代系统性能至关重要。

## 要点回顾

- GPU 优化吞吐而非单线程延迟：大量 SM、warp、轻量线程并行执行。
- 执行模型：thread 执行工作，block 绑定一个 SM 并共享共享内存，warp 是 32 线程的调度单位；warp 内遵循 SIMT。
- 内存层级决定性能：寄存器/共享内存快，L2 较慢，全局 HBM 最慢；优化核心是减少全局内存访问。
- TPU 与 GPU 高度趋同：都有矩阵乘单元和内存层级；TPU 用更少更大的单元，GPU 更灵活；最大差异在网络。
- 六个加速手段：控制发散、低精度、算子融合、重计算、coalescing、tiling。
- 低精度减少字节移动，但需处理缩放因子、转置、量化开销和数值稳定性；FP8/MXFP8/FP4 越来越依赖硬件与精细工程。
- 算子融合把多个全局内存往返合并为一个 kernel；重计算用额外计算换更少激活存储。
- Coalescing 利用 DRAM burst；行主序下跨行读列会破坏合并访存，padding 可对齐边界。
- Tiling 把 tile 搬入共享内存复用，将全局读取从约 $n$ 次降为 $n/T$ 次；tile 大小、对齐和矩阵整除性都会影响性能。
- Wave quantization：tile 数超过 SM 数时多出一轮，大量 SM 空闲，造成吞吐骤降。
- FlashAttention = tiled matmul + online softmax + 反向重计算，大幅降低 HBM 流量并避免保存 $n^2$ 激活。
