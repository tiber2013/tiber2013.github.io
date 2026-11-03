---
title: "第 11 讲 · 大规模分布式训练：从一块 GPU 到万卡集群"
description: "CS231n Spring 2025 L11 视频笔记；对照 2026 春季课件补充新增内容。"
---

# 第 11 讲 · 大规模分布式训练：从一块 GPU 到万卡集群

> **Stanford CS231n 计算机视觉 · L11** ｜ 讲者：Stanford CS231n 课程团队 ｜ 73 分钟 ｜ [原视频](https://www.youtube.com/watch?v=9MvD-XsowsE)

> 课件版本：2026 春季 ｜ 视频版本：2025 春季。2026 课件新增或调整的内容集中在文末「2026 课件更新」一节。

## 核心结论

这一讲回答两个问题：训练大模型的那块芯片里到底有什么，以及怎样把上万块芯片当成一台计算机来训练同一个网络。前半段从 H100 的存储层次、Tensor Core 和集群带宽的逐级衰减讲起：单设备算力十年涨了约千倍，而带宽只涨了两倍多，于是所有并行算法的核心都是“把通信藏进计算”。后半段把并行的对象拆成四条轴——数据（DP）、序列（CP）、层（PP）、通道（TP），再叠加 FSDP、HSDP 与 activation checkpointing 两类显存技巧，最后收束到唯一该优化的指标 **MFU**。对多数人来说，真正会用到的只是 DP → FSDP → checkpointing → HSDP 这条路线。

## 为什么拿 Llama 3 405B 当例子 [00:05 - 02:50]

讲者开场先给尺度感：十年前这门课刚开始时，几乎所有模型都在一块 GPU 上训练，多卡是少数派；十年后，几十、几百、几千甚至上万块设备同时训练一个模型才是常规。规模量级变了，算法和思维方式也必须跟着变。

贯穿全讲的例子是 Llama 3 405B。它不是最强模型，但它是少数把训练细节公开的、接近 SOTA 的模型。对比之下，GPT-4 的技术报告写得很直白：考虑到竞争格局和大规模模型的安全影响，报告不包含任何关于架构、模型规模、硬件、训练算力、数据构造与训练方法的细节。讲者调侃说，从那以后“能告诉你它是个 Transformer 就算客气了”。Llama 3 是 Meta 在 2024 年 4 月开源的模型，论文较详细地给出了系统与基础设施；视频录制时 Llama 4 刚于 2025 年 4 月发布，还没有论文。

![以Llama3-405B为例说明GPT4后模型细节不公开的趋势及Llama3/4的发布情况](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0001_00-02-01.jpg)

## 拆开一块 GPU：H100 的存储层次与算力 [02:50 - 11:53]

GPU 原本是为图形学造的协处理器——渲染要同时处理大量像素和几何图元，天然适合并行——后来被证明是很通用的并行处理器。今天训练的主力仍是 NVIDIA 的 H100（视频录制时下一代刚发布，还不太容易拿到）。

从芯片照片往里看，中间是计算核心，四周环绕 80 GB 的 **HBM**（high bandwidth memory）。显存与计算核心物理分离，数据要经过总线搬运，H100 这条通路的带宽是 3352 GB/s，约合每秒 3 TB。再往里是一块只有 50 MB 的 **L2 cache**，容量远小于 HBM，但离计算单元近得多。最核心的是 132 个 **SM**（streaming multiprocessor）。有意思的是照片上能数出 144 个：芯片制造采用 **binning**，那么多晶体管不可能全部完好，厂商标称“至少 132 个可用”，就能把更多芯片卖出去。

![NVIDIA H100 GPU 芯片实物图，标注出计算核心区域](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0002_00-05-08.jpg)

单个 SM 内部还有两级更小的存储：256 KB 的 L1 cache / 共享内存，以及同样 256 KB 的寄存器堆；此外有 128 个 FP32 core 与 4 个 Tensor Core。FP32 core 每个时钟周期做一次标量 $a\cdot x+b$，即 2 FLOP，128 个加起来是每 SM 每周期 256 FLOP。Tensor Core 的名字有点误导，它其实是“矩阵核”：一次完成 $[16\times4]\times[4\times8]+[16\times8]$ 的块矩阵乘加，$16\times4\times8\times2=1024$ FLOP，4 个就是每 SM 每周期 4096 FLOP，是 FP32 路径的 16 倍。所以想榨干 GPU，就必须让计算走 Tensor Core。

Tensor Core 还工作在**混合精度**下：输入通常是 16 bit，乘法在低精度完成，累加在 32 bit 完成。这在 PyTorch 层面非常具体：如果忘了把模型转成 16 bit（例如不用 autocast），计算会退回 FP32 core，比预期慢约 20 倍。

![H100 流式多处理器（SM）内部架构图，展示寄存器、L1缓存与 Tensor Core 布局](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0003_00-08-32.jpg)

回头看，这段硬件介绍本质上是在复习**存储层次**：越靠近计算单元的内存越小越快（256 KB → 50 MB → 80 GB），写高性能 kernel 的大量工作就是把数据在这一层层之间搬得恰到好处。

## 十年千倍：算力增长曲线 [11:53 - 15:14]

讲者用一代代 GPU 的峰值算力解释“为什么这十年 AI 变强了”。K40（2013）整卡只有 5 TFLOPS FP32；P100 是 10.6 TFLOPS；V100（2017 前后）第一次引入 Tensor Core，混合精度算力 125 TFLOPS；A100 是 312 TFLOPS；H100 是 989 TFLOPS；B200 理论上到 5000 TFLOPS。同期 FP32 路径只从 5 涨到 83.3 TFLOPS。

![展示从2013年K40到2024年B200 GPU算力（TFLOPS）增长趋势的折线图，突出Tensor Core带来的1000倍加速](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0005_00-14-06.jpg)

也就是说，12 年间单设备算力涨了约 1000 倍，而芯片的物理尺寸、重量几乎没变。讲者强调，任何东西在十年里变快 1000 倍都值得停下来认真对待，这大概是深度学习进步最大的单一驱动力。他也顺手纠正一个常见误读：B200 的“5000”是 Tensor Core 上的 TFLOPS，不是 5000 个 Tensor Core。

![在GPU算力增长折线图上叠加了“We can also train with > 1 GPU!”的文字说明](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0006_00-14-46.jpg)

## 从一块卡到一个集群：带宽逐级衰减 [15:14 - 22:10]

单卡之内是存储层次，出了芯片，同样的规律在数据中心里重演：能搬多少 bit、搬多快，随距离迅速下降。以 Meta 训练 Llama 3 的集群为例：

- **GPU 内部**：3352 GB/s；
- **服务器**（8 卡一机）：任意两张卡之间约 900 GB/s；
- **机架**：2 台服务器 = 16 卡，约 6 英尺高；
- **Pod**：192 个机架 = 3072 卡，任意两卡之间约 50 GB/s；
- **集群**：8 个 Pod = 24576 卡，跨 Pod 不到 50 GB/s。

![NVIDIA H100 GPU 芯片结构图，标注 GPU 内部带宽 3352 GB/sec](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0007_00-15-14.jpg)

![Meta Llama3 集群层级架构图：从 8 卡服务器到 24,576 卡集群的扩展关系及各级带宽](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0008_00-18-29.jpg)

这组数字是全讲后面所有算法的约束条件：机内通信比机架快十几倍，机架又比跨 Pod 快。集群规模也不是终点，世界上已有 5 万卡、10 万卡级别的集群，扩展方式就是把更多 Pod 接到一起，或者再加一层 super pod。训练时长上，最大的模型通常按“几个月”计——这更多是人的限制而不是技术的限制，因为计划、协作与调试都很难跨越更长的时间。

机架之所以是标准单位，是因为数据中心几十年来就按这个尺寸建，GPU 服务器虽然更重、更耗电，也不能把机房推倒重来。除了计算机架，集群里还有专门的网络机架和存储机架，散热则是另一个量级的工程问题。

![GPU 集群整体统计：24,576 张 GPU、1.875 PB 显存、24.3 EFLOP/sec 算力](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0009_00-22-16.jpg)

## 把集群当成一台计算机 [22:10 - 23:10]

换个视角：不要想单卡，而把这个集群看成一台大计算机。它有 24576 张 GPU、1.875 PB 显存、4.15 亿个 FP32 core、1300 万个 Tensor Core，合计 24.3 EFLOP/s。目标是让这一整块机器像一台超级计算机那样，连续几个月训练一个巨大的网络。

## 不止 NVIDIA：TPU 与其他加速器 [23:10 - 26:20]

GPU 是最主流的训练硬件，但 Google 的 **TPU** 是真正的竞争者：自研芯片，已经迭代六代。视频里给出的 v5p 单芯片 459 TFLOP/s（BF16）、95 GB 显存，以最多 8960 颗芯片组成 pod；现场展示的还是一台只有 256 颗芯片的 V2 pod，四排机架并排，可以想象 9000 颗规模有多大。TPU 的特点是你买不到，只能去 Google 工作或在云上租。讲者判断 Gemini 几乎肯定是在 TPU 上训练的。其他厂商也在追赶：AMD 的 MI325X 纸面参数接近 H100，AWS 的 Trainium 被 Anthropic 用于部分训练；但今天真正站稳的仍是 NVIDIA GPU 与 TPU 两家。

![Google TPU v5p 芯片规格与机架部署图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0010_00-23-24.jpg)

## 四条并行轴：先想清楚在切什么 [26:20 - 29:00]

第二部分的问题是：怎么写出能用上几万块 GPU 的算法。讲者的框架很干脆——计算机只做两件事，**计算**与**通信**；并行训练的全部技巧，就是切分计算让每块设备都有活干，同时把通信重叠进计算里。

Transformer 是 $L$ 层堆叠，每层的激活是一个三维张量（batch, sequence, dim）。于是有四个可切的轴：切 batch 是**数据并行**（DP），切 sequence 是**上下文并行**（CP），切层是**流水线并行**（PP），切 channel 是**张量并行**（TP）。

![本节主题页：GPU 硬件简介与多 GPU 训练方法](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0011_00-26-20.jpg)

## 数据并行：最朴素也最常用 [29:00 - 39:31]

训练时损失是 mini-batch 上各样本损失的平均，而每个样本的前向、反向互不依赖。如果一张卡能放下 $N$ 个样本、手上有 $M$ 张卡，就用 $M\times N$ 的宏 batch，每张卡各拿 $N$ 个。数学上成立是因为梯度是线性的：

$$
L = \frac{1}{MN}\sum_{i=1}^{M}\sum_{j=1}^{N}\ell(x_{i,j},W), \qquad
\frac{\partial L}{\partial W} = \frac{1}{M}\sum_{i=1}^{M}\left(\frac{1}{N}\sum_{j=1}^{N}\frac{\partial}{\partial W}\ell(x_{i,j},W)\right)
$$

括号里是单卡对 $N$ 个样本的普通反向传播，外层只是把 $M$ 张卡的梯度做平均。这不是近似，只是把求和顺序换了一下，结果与在一张更大的卡上训练完全一致。

流程是：每张卡各有一份模型、优化器状态和梯度；各自加载**不同**的 batch；各自前向算 loss、反向算梯度；然后做一次 **all-reduce**，每张卡都得到全局平均梯度；最后各自更新权重。这里最容易犯的 bug 是让所有卡加载了同一批数据——那样等于什么都没做。

关键在于反向传播与 all-reduce 可以重叠：反向往回走时，最后一层的梯度算完就开始通信，同时计算倒数第二层的梯度。通信按层“切片”进行，等反向传播结束时 all-reduce 也基本完成，权重更新不必等待。

![斯坦福CS231n课程中关于数据并行（Data Parallelism）的幻灯片，回顾了小批量样本损失平均的概念](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0012_00-29-05.jpg)

至于哪一步是瓶颈，讲者的回答是“取决于你的情况”——设备速度、模型大小、batch 大小、互连带宽都会改变答案，只能实测。有同学问能否每张卡各走几步再平均，这就是历史上的**异步 SGD**，Google 在 TPU pod 之前用过，但更不稳定、更难复现和调试，同步更新仍是主流。重叠通信与计算必须由软件显式安排（PyTorch 的 `DistributedDataParallel` 就替你做了）；有意思的是在单设备内 CUDA 会由硬件自动完成大量异步搬运，到了集群层面反而要靠软件编排。

## FSDP：把权重也切开 [39:31 - 48:46]

数据并行很快撞上显存墙。每个权重需要维护四五个数字：权重本身、梯度、Adam 的一阶与二阶矩，有时还有权重的指数滑动平均；按 16 bit 存，每个数 2 字节，约合 8 字节/参数。10 亿参数就要 8 GB，80 GB 的 H100 最多装下约 100 亿参数——而大模型远不止这个规模。

修法是把模型权重也分片，这就是 **fully sharded data parallelism（FSDP）**。每个权重（按整层划分，而不是单个标量）有一个 owner GPU，由它同时保管该权重的全局梯度与优化器状态。

前向时，第 $i$ 层的 owner 把权重广播给所有卡，大家各算各的激活，算完非 owner 立刻删掉本地副本以省显存；同时可以预取第 $i+1$ 层的权重，把通信藏在计算后面。反向时反过来：owner 再广播一次第 $i$ 层权重，各卡算自己的局部梯度 $dL/dW_i$，送回 owner 求和得到宏 batch 的完整梯度并更新。更新后的权重不必立刻广播——下一次前向还会再发一遍。

于是稳态里同时有三件事在跑：算第 $L$ 层的反向、聚合与更新第 $L+1$ 层的梯度、预取第 $L-1$ 层的权重。**2026 课件补充**：实际实现中每个权重是切分到所有 GPU，前向用 all-gather、反向用 reduce-scatter，比 broadcast/reduce 更高效（详见文末）。收益很直接：1000 亿参数的模型需要 800 GB，摊到 80 张卡上每张只需 10 GB。

## HSDP：按拓扑安排两种并行 [48:51 - 52:59]

FSDP 的通信量比 DP 大：一轮前向 + 反向要搬三份权重量级的数据（前向权重、反向权重、梯度），而普通 DP 组间只需同步一次梯度。既然如此，就该把通信密集的那种放进带宽高的地方。

**Hybrid sharded data parallelism（HSDP）** 把 $N=M\times K$ 张卡排成二维网格：每个 $K$ 卡组内部做 FSDP（权重切在这 $K$ 张卡上），$M$ 个组之间做普通 DP（每组持有完整的一份权重副本）。典型排法是 $K$ 取一台服务器内的 8 张卡、$M$ 跨服务器——组内 900 GB/s，组间慢一些也没关系。这也是“多维并行”的第一个例子：同时使用两种策略，并按集群拓扑摆放。

## Activation checkpointing：拿计算换显存 [53:03 - 55:54]

FSDP 解决了权重，激活又开始爆显存。Llama 3 405B 有 126 层、隐层维度 16384、序列长度 4096，仅前向的隐藏状态就非常可观。

把每层看成两个函数：前向 $A_{i+1}=F_i^{\rightarrow}(A_i)$，反向 $G_i=F_i^{\leftarrow}(A_i,G_{i+1})$。常规训练保存所有激活，$N$ 层需要 $O(N)$ 计算与 $O(N)$ 显存。**Activation checkpointing** 的思路是反向时重算：最激进的做法是前向算完就扔，反向需要时从边界重算，计算变成 $O(N^2)$、显存降到 $O(1)$——$N^2$ 太贵。折中方案是每隔 $C$ 层存一个 checkpoint，只在块内重算：

$$
\text{compute} = O\!\left(\frac{N^2}{C}\right), \qquad \text{memory} = O(C)
$$

取 $C=\sqrt{N}$，得到 $O(N\sqrt{N})$ 计算与 $O(\sqrt{N})$ 显存。代价是训练变慢，收益是能放下更大的模型与更长的序列。

## 扩展配方 [55:54 - 57:52]

把上面的东西串起来，就是一条从单卡到万卡的路线：

1. 约 128 卡、10 亿参数以内，普通 **DP** 就够；
2. 每卡 batch 尽量开满显存，这几乎总是对的；
3. 模型超过 10 亿参数，换 **FSDP**；
4. 激活放不下，加 **activation checkpointing**；
5. 卡数到二三百甚至五百以上，FSDP 通信开始不划算，换 **HSDP**；
6. 上千卡、500 亿参数以上、序列长度上万，才轮到 **CP / PP / TP**。

## MFU：迷路时唯一该看的指标 [57:52 - 63:02]

到这一步旋钮已经很多：global batch、local batch、HSDP 维度、FSDP 维度、重算多少层……讲者给的答案是只优化一个指标——**model FLOPs utilization（MFU）**。

先看它的上界 **hardware FLOPs utilization（HFU）**：H100 的理论峰值是 989.4 TFLOP/s（16 bit 矩阵乘），写几行 PyTorch 只跑大矩阵乘，8000×8000 时能到约 80%。但 HFU 只衡量“纯 matmul”，不包含激活重算、数据加载与增强、优化器这些真实训练里的开销。

MFU 换了个问法：GPU 的理论峰值里，有多大比例真正用在了模型的前向 + 反向上？做法是算出一轮 forward + backward 的矩阵乘 FLOP（反向约等于前向的 2 倍，忽略非线性、归一化和残差这类跑在 FP32 core 上的算子），除以设备峰值得到理论时间，再用实测的一整轮迭代时间去除：

$$
\text{MFU} = \frac{t_{\text{theoretical}}}{t_{\text{actual}}}, \qquad t_{\text{theoretical}} = \frac{\text{FLOP}_{\text{theoretical}}}{\text{FLOP/s}_{\text{theoretical}}}
$$

```python
L, D, N = 8, 8192, 8192
flop_fwd = N * L * 2 * D * D                     # 前向矩阵乘 FLOP
t_theoretical = 3 * flop_fwd / (989.4 * 10**12)  # 反向约 2x 前向
# ... 计时一轮 data load + forward + backward + optimizer.step ...
mfu = t_theoretical / t_actual
```

一个宽 MLP、大 batch 的例子在 H100 上能到约 50% MFU。经验值是：**30% 以上算不错，40% 以上算优秀**。Llama 3 405B 在 8192 到 16384 张卡上训练，MFU 大致落在 38%–43%。反常的是新卡有时 MFU 更低：A100 上能超过 50%，因为从 A100 到 H100，算力涨了约 3 倍，显存带宽只涨了约 2 倍，通信越来越跟不上计算。

## 上下文并行：切序列 [63:27 - 66:00]

序列很长时一张卡放不下一整条序列，就沿 sequence 维切开，每张卡负责一段。layernorm、残差、FFN 这些本身逐 token 独立，切起来几乎免费；FFN 有权重，各卡各存一份、反向时像 DP 一样同步梯度即可。真正麻烦的是 **attention**：它要算序列内所有 token 的两两交互。

两种做法：**Ring Attention** 把注意力矩阵切成块，让各卡按顺序轮转处理；**Ulysses** 则把并行维度换成 head——先算出 QKV，再把 (batch, seq/N, heads, head_dim) 重排成 (batch, seq, heads/N, head_dim)，让每张卡负责一部分 head。后者实现简单，但 head 数必须能被卡数整除。Llama 3 的训练分两阶段：第一阶段序列 8192、完全不用 CP；第二阶段把序列拉到 131072，用 16 路 CP，即每张卡处理 8192 个 token。

## 流水线并行：切层与 bubble [66:02 - 69:16]

把层分给不同 GPU 很直观，但层之间有顺序依赖：第 2 段要等第 1 段算出激活，反向又要等上游传回梯度。朴素做法下大部分卡在空等，这段空白被称为 **bubble**。$N$ 路流水线的理论 MFU 上限只有 $1/N$——8 路就是 12% 左右。

解法是同时跑多个 **micro-batch**：第 1 段算完蓝色 micro-batch 就把激活交给第 2 段，自己接着算黄色 micro-batch。4 路流水线配 4 个 micro-batch，非空白的比例从 25% 提到约 57%。micro-batch 越多 bubble 越小，但激活也越多，于是又要接上 activation checkpointing——旋钮之间互相牵制，还是回到 MFU 来调。

## 张量并行：切权重矩阵 [69:16 - 71:15]

最后一条轴切的是 channel 维，也就是每个线性层的权重矩阵本身。在 $XW=Y$ 中把 $W$ 按列切成 $W_1,\dots,W_4$，第 $i$ 张卡算 $Y_i=XW_i$，每张卡只算输出的一块。问题是下一层需要完整的 $Y$，得先拼接，而拼接无法与计算重叠。

巧妙之处在连续两层：第一层按列切，第二层按行切，于是

$$
Z = YU = \sum_{i=1}^{4} Y_i U_i
$$

两层之间完全不需要通信，各卡算自己那一项，最后对 $Z$ 做一次 all-reduce 即可。Transformer 的 FFN 正好是两层 MLP，所以这个技巧在 FFN 上用得非常普遍。

## 四维并行：Llama 3 的实际配置 [71:15 - 72:53]

哪种并行最好？讲者的答案是“全都要”。Llama 3 405B 最大的一次训练用 16384 张卡：8 路 TP × 16 路 CP × 16 路 PP × 8 路 DP。排列这些轴的原则仍然是拓扑——通信最密集的 TP 放在机内，通信较少的 DP 放到跨机架、跨 Pod。

回顾整讲：一块 GPU 是一个通用并行处理器；一个集群是拥有上万块 GPU 的巨型并行机，我们想把它编程成一台机器；并行的手段是把计算沿 batch、sequence、layer、channel 四个轴切开，外加 activation checkpointing 这个省显存的技巧；而所有选择最终都服从同一个指标——MFU。

## 2026 课件更新

以上正文以 2025 春季视频为准。对照 2026 春季课件，主要有这些差异：

- **行政信息置顶**：2026 课件开场列出 5/7 Assignment 2 截止、5/12 课堂期中；2025 视频直接从主题讲起。
- **Llama 4 的状态改了**：视频里是“2025 年 4 月刚发布、暂无论文”，2026 课件改为“引发争议，技术报告始终没有发布”。
- **新增「通信原语」一节**：课件系统介绍了 **all-reduce**、**reduce-scatter**、**all-gather**、**all-to-all** 四种 collective 的语义与用途，并注明 all-reduce 用于 DDP 的反向、all-to-all 用于 Ulysses 式 CP 的重排。视频只在 DP/FSDP 里顺带提到 all-reduce。
- **FSDP 实现补充**：课件脚注说明真实系统是把每个权重切分到所有 GPU，前向 all-gather 权重、反向 reduce-scatter 梯度，比 owner + broadcast/reduce 更高效，并标注 **ZeRO**（Rajbhandari et al., 2019）为出处；视频只描述了 owner/broadcast 的理想化版本。
- **新增 Expert Parallelism（EP）**：MoE 模型把不同专家切到不同 GPU。视频口述“一共有五种并行”却只展开了四种，2026 课件补上了 EP 这一维度。
- **硬件参数更新**：TPU v5p（459 TFLOP/s、95 GB、pod 8960）换成 **TPU 8t**（3151 TFLOP/s BF16、216 GB、pod 9600 芯片）；AMD MI325X 换成 **MI355X**（2500 TFLOP/s BF16、288 GB）；AWS Trainium 换成 **Trainium3**（2500 TFLOP/s FP8、144 GB、144 芯片的 UltraServer）。
- **激活显存给出算式**：Llama3-405B 仅 FFN 隐藏激活就需要 $2\times126\times(4\times16384)\times4096$ 字节 ≈ 636 GB，视频只定性说“很快就爆显存”。
- **配方阈值微调**：2026 课件写“>256 卡考虑 HSDP”“>1000 卡、>50B 参数或序列长度 >16K 才用 CP/PP/TP”；视频口述为“约 256–512 卡”“序列一万左右”。
- **MFU 数据表格化**：新增 GPT-3 21.3%（V100）、Gopher 32.5%（4096 TPU v3）、Megatron-Turing NLG 530B 30.2%（2240 A100）、PaLM 540B 46.2%（6144 TPU v4）的对照表；Llama3 三档为 43%/41%/38%；A100→H100 明确写成 3.1× FLOPs、2.1× 显存带宽（视频口述约 3×/2×）。
- **ND 并行给出完整配置表**：8192 卡（TP 8 / CP 1 / PP 16 / DP 64）、16384 卡（8/1/16/128）、16384 卡（8/16/16/8，序列 131072），每 DP 组 batch 32/16/16，tokens/batch 均为 16M。
- **结尾不变**：下一讲仍是 Self-Supervised Learning，与视频一致。

## 要点回顾

- **规模是常态**：从单卡到万卡，算法必须为通信而设计；Llama 3 405B 是少数公开系统细节的接近 SOTA 的模型。
- **GPU 是存储层次的机器**：H100 有 256 KB L1、50 MB L2、80 GB HBM，带宽 3352 GB/s，132 个 SM（物理 144，靠 binning 出货）。
- **Tensor Core 是吞吐主力**：一个 SM 每周期 4096 FLOP 对 FP32 的 256 FLOP；必须用 16 bit 混合精度，否则慢约 20 倍。
- **十年千倍**：单卡算力从 K40 的 5 TFLOPS 到 B200 的 5000 TFLOPS（Tensor Core），这是深度学习进步的主要驱动力。
- **集群带宽逐级衰减**：GPU 3352 → 机内 900 → Pod 50 → 跨 Pod 不到 50 GB/s，并行算法要顺着拓扑设计。
- **DP**：宏 batch 切给 $M$ 张卡，各自前向反向，all-reduce 平均梯度；数学上与单卡等价，反向可与通信重叠。
- **FSDP**：权重、梯度、优化器状态分片，owner 广播权重并聚合梯度，把 100B 模型压到每卡 10 GB。
- **HSDP**：组内 FSDP、组间 DP，把通信密集的部分放进高带宽的机内。
- **Activation checkpointing**：每 $C$ 层存一个 checkpoint，用 $O(N^2/C)$ 计算换 $O(C)$ 显存，$C=\sqrt N$ 是常用折中。
- **扩展路线**：DP → FSDP → checkpointing → HSDP，只有上千卡、超大模型与超长序列才需要 CP/PP/TP。
- **MFU** 是分布式训练的唯一北极星：>30% 不错、>40% 优秀；A100→H100 算力涨 3 倍而带宽只涨 2 倍，新卡 MFU 反而可能更低。
- **PP 的 bubble** 靠 micro-batch 填补，**TP** 靠连续两层的列/行切分省掉中间通信，**4D 并行**把它们全部叠起来。

## 视频来源

https://www.youtube.com/watch?v=9MvD-XsowsE
