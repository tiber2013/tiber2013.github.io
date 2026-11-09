---
title: "第 11 讲 · 大规模分布式训练：从一块 GPU 到万卡集群"
description: "Stanford CS231n Spring 2025 L11 中文视频笔记（2026-11 重写版）"
---

# CS231n Spring 2025 · Lecture 11 大规模分布式训练：从一块 GPU 到一台「数据中心计算机」

**核心结论**：本讲把「大规模分布式训练」拆成两半——先弄清 GPU 硬件，再弄清如何用成千上万块 GPU 协作训练一个模型。以 Meta 的 Llama3-405B 为贯穿案例（GPT-4 之后业界普遍不再公开架构与训练细节，而 Llama3 论文公开了大量系统与训练信息），讲者解剖 H100 的三级存储层次、Tensor Core 混合精度与 SM 结构，指出 2013 年 K40 到 B200 单卡算力约涨 1000 倍，而 Llama3 集群把 24,576 块 GPU、24.3 EFLOP/s 组织成一台「计算机」。算法侧把 Transformer 的 $(Batch, Sequence, Dim)$ 张量与 $L$ 层拆成四条并行轴：数据并行（DP）、全分片数据并行（FSDP）、混合分片（HSDP）、上下文/流水线/张量并行（CP/PP/TP），并用激活检查点以计算换显存。贯穿所有调参的「指路明灯」是 MFU（模型 FLOPs 利用率，>30% 良好、>40% 优秀），Llama3 在 8k–16k GPU 上用 8 路 TP × 16 路 CP × 16 路 PP × 8 路 DP 的四维并行把 MFU 做到 high-30s / low-40s。

## 1. 从单卡到万卡：为什么需要分布式训练，以及 Llama3-405B 案例 [00:05 - 02:58]

十年前，几乎所有模型都在单块 GPU、单台设备上训练，多设备训练相当少见；如今在数十、数百、数千乃至数万块设备上**并发训练同一个模型**已成新常态，这需要新算法与新思维方式。本讲分两部分：GPU 硬件到底是什么；以及如何写算法用好大量 GPU。

贯穿全讲的案例是 **Llama3-405B**——不是因为它最强或最有趣，而是因为它接近 SOTA 且**公开了大量实现细节**。GPT-4 论文（2023）开启了一个行业转折，原文称「考虑到竞争环境以及 GPT-4 这类大模型的安全影响，本报告不包含关于架构（含模型规模）、硬件、训练算力、数据集构造、训练方法等的任何进一步细节」；此后三年 Google、OpenAI、Anthropic 等的最强模型几乎什么都不公开，能告诉你「是个 Transformer」就算幸运。Llama3 是 Meta 于 2024 年 4 月开源的 LLM，论文公开了很多模型与训练细节（数据集不多，系统基础设施很多），让我们得以一窥大规模 LLM 究竟怎么训；Llama4 已于 2025 年 4 月发布初始模型，但尚无论文。

![Llama3-405B运行示例及GPT4/Llama系列模型细节公开情况对比](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0001_00-01-04.jpg)

## 2. GPU 硬件解剖：H100、存储层次与 Tensor Core [02:58 - 11:53]

### 2.1 从图形协处理器到通用并行处理器 [02:58 - 05:45]

GPU 即 graphics processing unit，最初是为计算机图形设计的专用协处理器：要生成屏幕上的大量像素，就要并行处理大量细小的几何图元，天然适合大规模并行。2000 年代初研究者把显卡「扭」成通用并行编程工具；到 2000 年代末、2010 年代，Nvidia 把它当通用并行处理器来设计、营销，并在深度学习刚起飞时就早早重金投入，使其十多年来一直是训练大规模深度学习模型的主要硬件。讲者调侃：在本课的 Huang Auditorium（Jensen Huang 是 Nvidia 创始人兼 CEO）讲这一讲很应景。

### 2.2 H100 芯片概览与 binning [05:45 - 07:27]

当前训练主力是 **NVIDIA H100**（下一代刚发布但还不易获得）。中间是 compute cores，周围是 **80 GB HBM（高带宽显存）**；显存与计算核分离，二者通过总线交换数据，带宽约 **3 TB/s**（课件标注 3352 GB/sec）。再往里是约 **50 MB L2 cache**——比 80 GB 小得多，但离计算单元极近、访问快得多。真正的核心是 **132 个 Streaming Multiprocessor（SM）**，相当于独立的并行核：并行能力比典型 CPU 核强，但时钟更低、分支预测更弱，不宜与 CPU 核直接类比。

![NVIDIA H100 GPU内部架构概览图，展示计算核心、L2缓存及流式多处理器分布](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0002_00-06-04.jpg)

画面上其实能数出 144 个 SM，讲者特意解释了差异：芯片制造存在**良率问题（binning）**，晶体管太多，无论投入多少都不可能每颗都完美；于是设计按 144 个做，只承诺其中 132 个可用，从而能卖出更大比例的产出。

### 2.3 SM 内部：FP32 Core、Tensor Core 与混合精度 [07:27 - 11:53]

深入单个 SM（132 个之一），能看清完整的**三级存储层次**：256 KB L1 cache + 256 KB 寄存器文件（课件标 Register File 16,384×32-bit、第 4 代 Tensor Core），往上是 50 MB L2，再往外是 80 GB HBM——越靠近计算单元，容量越小但速度越快；编写高性能 GPU kernel 的大量工作，就是在优化数据在这几层之间的搬运。讲者笑称「你以为在学深度学习，其实在学计算机体系结构」。

![H100流式多处理器(SM)内部详细结构图，包含Tensor Core、L1缓存及寄存器文件](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0003_00-08-32.jpg)

计算单元有两类。**128 个 FP32 core**：每个能在一个时钟周期内算一次 $ax+b$（一次乘法加一次加法），故整个 SM 每周期约 256 次浮点运算。真正的主力是红色的 **4 个 Tensor Core**（讲者认为叫 matrix core 更贴切）：专用电路，只做矩阵乘。H100 的每个 Tensor Core 每周期可完成一小块 $AX+B$——输入 $A$ 为 $16\times4$、$B$ 为 $4\times8$、偏置为 $16\times8$，按每次乘加各计一次约 **1024 FLOP**；4 个核合计每 SM 每周期约 **4096 FLOP**，是 FP32 路径（256）的十几倍。所以想让 GPU 跑满，代码必须最大限度使用 Tensor Core。

另一个关键是**混合精度**：Tensor Core 用 16-bit 输入（bf16/fp16 等）做乘法，用 32-bit 高精度做累加。讲者给出一个很实际的警告——如果在 PyTorch 层忘了把模型 cast 成 16-bit，运算就会落到 FP32 core 上，**比预期慢约 20 倍**。

## 3. 算力 1000 倍与 GPU 集群：把数据中心当作一台计算机 [11:53 - 22:58]

### 3.1 单卡算力的十年演进 [11:53 - 14:46]

讲者读博时用的 K40（2013）整卡只有 **5 TFLOP/s FP32**。以时间为横轴（2013→现在）、单卡峰值 TFLOP/s 为纵轴，曲线陡升；关键转折是 **V100（约 2016/2017）首次引入 Tensor Core**，此后 Tensor Core 更多、更大、占芯片面积更多，吞吐量暴涨。最新 **B200** 理论值约 **83.3 TFLOP/s FP32、5000 TFLOP/s Tensor Core 混合精度**。课件给出的完整序列：K40 5 FP32；P100 10.6 FP32；V100 125 TC / 15.7 FP32；A100 312 TC / 19.5 FP32；H100 989 TC / 67 FP32；B200 5000 TC / 83.3 FP32（务必区分 Tensor Core 算力与 FP32 算力，5000 不是 5000 个核，而是 TFLOP 数）。12 年间单卡约 **1000×** 提升，讲者视其为深度学习十年进步的主要驱动力；而更疯狂的是我们并不只用一块 GPU。

![GPU性能随时间演进折线图，展示从K40到B200的FP32与Tensor Core算力增长趋势](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0004_00-14-46.jpg)

### 3.2 Llama3 集群层级与带宽悬崖 [14:46 - 18:29]

把 GPU 放进数据中心，本质是存储/带宽层次的继续延伸，而**越往外带宽越慢**。Llama3 技术报告给出了具体数字（不同集群略有差异）：单块 H100 片内 **3352 GB/s**；**Server = 8×GPU**，机内任意两块 GPU 间约 **900 GB/s**（约 3× 下降）；**Rack = 2 Servers = 16 GPU**（约一人高的标准机柜，物理约束使 rack 成为数据中心几十年来的标准单元）；**Pod = 192 Racks = 3072 GPU**，跨 rack 的任意两块 GPU 约 **50 GB/s**（再降约 20×）；**Cluster = 8 Pods = 24,576 GPU**，跨 pod 通信 < 50 GB/s。

![Meta Llama3集群架构示意图，展示Server Rack、CPU Server与GPU的层级关系及带宽参数](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0006_00-17-07.jpg)

![Meta Llama3 GPU集群层级结构及带宽参数表](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0007_00-18-29.jpg)

这还不是世界最大集群，5 万、10 万 GPU 的集群确实存在，只要把更多 pod 堆起来（或再加一层 super-pod）。集群里除计算 racks 外，还有专门放网络硬件、存储硬件的机柜；机柜要排成行留出走道，占地很大。散热同样是硬约束——单张 4090 打游戏就能让房间明显变热，成千上万块堆在一个数据中心里需要严肃的冷却方案。设计系统的秘诀是：小单元内部仍保持高带宽，能取快通信时尽量取，规模放大后优雅地回退到慢通信。

### 3.3 总算力、训练时长与「一台大计算机」 [18:29 - 22:58]

面对这种规模，心态要转变：**不再盯着单块设备或单台服务器，而把整个数据中心当成一台计算机**。总账是：24,576 块 GPU、**1.875 PB** GPU 显存、**415 M** 个 FP32 core、**13 M** 个 Tensor Core、**24.3 EFLOP/s**（$24.3\times10^{18}$）。目标就是让这 2.4 万块 GPU 像一台巨型超级计算机，连续数月训练一个能吸收海量数据的巨型网络。最长训练通常以「月」计，讲者认为这更多是人的计划与进度约束而非技术限制（最大的 GPT-4.5 / GPT-5 级别模型也许已接近一年）。

![GPU集群整体算力与内存统计及训练目标](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0008_00-22-48.jpg)

## 4. GPU 之外的训练硬件：Google TPU、AMD 与 AWS Trainium [22:58 - 26:10]

Nvidia 虽最主流，但并非唯一。**Google TPU（tensor processing unit）**是当下最大的竞争者，已迭代六代，规格与 H100 同量级但有不少不同的设计决策（讲者说没时间展开）。**TPU v5p**：每芯片 **459 TFLOP/s BF16**、**95 GB** 显存，可组成最多 **8960 芯片**的 pod；课件图片是一个只有 256 芯片的 V2 pod，四个比人略高的机柜并排。Google 的 Gemini 等大模型几乎肯定跑在 TPU 上；与 Nvidia 不同，TPU 买不到，只能在 Google 工作或在 Google Cloud 上租，因此普及度仍略低。其他公司在追赶：AMD 的 **MI325X**（纸面参数与 H100 相当，但影响远不及）、AWS 自研 **Trainium**（Anthropic 部分训练在用）。讲者的判断是：目前 Nvidia GPU 与 Google TPU 在易用性、性能、市场份额上明显领先。

![Google TPU v5p硬件规格及机柜实物照片](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0009_00-23-21.jpg)

## 5. 并行化的四条轴：计算、通信与张量切分 [26:10 - 28:58]

计算机在高层次上只做两件事：**computation**（把输入 bit 算成新的输出 bit）与 **communication**（把 bit 从一处内存搬到另一处）。整个诀窍就是利用集群里多尺度的存储层次，把通信与计算**重叠**，并把计算切分并行，让成千上万块 GPU、数百万个计算单元都始终有有用的工作，再让它们彼此通信以共同训练一个巨型网络。

这些切分大多针对 Transformer（大规模训练的主流架构）：模型是 $L$ 层的堆叠，每层作用在三维张量 $(Batch, Sequence, Dim)$ 上——mini-batch 里有若干序列，每个 token 是一个 $Dim$ 维向量。于是有四条可切的轴，讲者先给出四种按维度切分的并行策略（实践中还能叠加成多维并行）：

- **Data Parallelism (DP)**：切 Batch 维；
- **Context Parallelism (CP)**：切 Sequence 维；
- **Pipeline Parallelism (PP)**：切 $L$（层）维；
- **Tensor Parallelism (TP)**：切 Dim 维。

![多GPU训练策略总览：DP/CP/PP/TP四种并行方式及其切分维度](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0010_00-27-56.jpg)

名字各不相同，本质都是沿这四个维度把计算拆开。后面会逐一展开其中的细节与权衡。

## 6. 数据并行 DP 与全分片 FSDP [28:58 - 48:51]

### 6.1 DP 的数学等价、all-reduce 与通信重叠 [28:58 - 39:31]

训练总在 mini-batch 上进行：loss 是各样本 loss 的平均，梯度是各样本梯度的平均，而「对每个样本算 loss、算梯度」彼此独立，天然可并行。若单卡装得下 $n$ 个样本、共有 $m$ 块 GPU，就用一个 $m\times n$ 的大 mini-batch，每卡分到 $n$ 个。数学上之所以成立，是因为**梯度是线性的**：

$$L=\frac{1}{mn}\sum_{j=1}^{m}\sum_{i=1}^{n}\ell(x_{ij},W),\qquad \nabla_W L=\frac{1}{m}\sum_{j=1}^{m}\underbrace{\left(\frac{1}{n}\sum_{i=1}^{n}\nabla_W \ell(x_{ij},W)\right)}_{\text{每卡本地反向传播，可并行}}$$

内层（蓝色项）是一块 GPU 上对 $n$ 个元素的正常反向传播，可并行计算；外层再对 $m$ 个设备的梯度求平均。这不是近似，与在单块更大的 GPU 上训练**完全等价**，只是巧妙地换掉了求和、求梯度、求平均的次序。

GPU 视角（图中 $m=3$，实际远大于 3）：每块 GPU 各自保存一份完整的**模型权重、优化器状态与梯度**；各卡必须加载**不同**的 mini-batch（讲者特别提醒，学生代码里常见的 bug 是所有卡读入同一批数据，那样毫无收益）；各自独立做 forward 得到本地 loss、再独立做 backward 得到本地梯度，全程无需通信。之后才需要通信：做 **all-reduce**——每卡把自己的梯度广播给所有卡、同时收集所有卡的梯度，通常按设备数呈对数时间完成；结束后每卡都持有一份相同的、跨整个 macro-batch 平均后的梯度，于是各自在本地权重上做同样的更新（假定算术确定性，权重保持一致）。

关键优化是**把 backward 与通信重叠**：算完最后一层的本地梯度就开始对它的 all-reduce，同时计算单元继续算倒数第二层的 backward，如此逐层「chunk along」，理想情况下 backward 结束时梯度已全部归约，无需等待即可一次性更新权重。因为通信相对慢，隐藏通信成本是核心技巧。它是否成为瓶颈完全取决于设备速度、模型/批次大小与互联带宽，必须针对自己的场景 benchmark。讲者还提到曾经的**异步 SGD**（Google 在 TPU pod 之前、2010 年代初用过）：各副本独立走多步再偶尔平均，更不稳定、难调试难复现、效果略差，但更契合分布式；他猜测未来几年可能回潮。

软件层面补一句：硬件不会自动替你调度通信，必须在软件里编排（单卡 CUDA 编程时硬件会自动做异步传输，集群层则不会，这是一个有意思的不对称）；好在 PyTorch 的 **DistributedDataParallel（DDP）** 把常见情形做得相对透明。整个系统是异构的：底层 GPU kernel 用 CUDA（Nvidia 的类 C 语言）写，外面由 C++ / Python 包装成易用的 PyTorch。

### 6.2 显存瓶颈：每个参数要存 4–5 个数 [39:31 - 41:01]

DP 是最早的并行方式，但很快撞上模型规模的墙：每块 GPU 都要存一份完整权重。对每个参数，通常需要跟踪 **4–5 个数**——参数本身、它的梯度、Adam 的 $\beta_1$、$\beta_2$，有时还有参数的指数移动平均（EMA）。以 16-bit 训练（每个数 2 bytes）作下界，约 **8 bytes/参数**：10 亿参数就要约 **8 GB**，而 H100 整卡只有 80 GB，于是 DP 能训的模型上限约 **100 亿参数**——远不够，必须摆脱「GPU 显存决定模型能多大」的暴政。

### 6.3 FSDP：按层分片、广播与重叠 [41:01 - 48:51]

解法很直接：除了切数据，也**切模型权重**，这就是 **Fully Sharded Data Parallelism（FSDP）**。每个权重 $W_i$（通常按层，即整层的权重矩阵，而非单个标量）指派给唯一一个 **owner GPU**，该 GPU 同时负责该权重的全局梯度与优化器状态。它仍是数据并行：每卡加载自己独立的 batch、做完整 forward/backward、归约梯度、更新。以 4 层网络分到 2 卡（$W_1,W_2$ 归 GPU1，$W_3,W_4$ 归 GPU2）为例：

- **前向**：算第 $i$ 层前，拥有 $W_i$ 的 GPU 把权重**广播**给所有卡；所有卡跑完该层前向、算出激活；不拥有 $W_i$ 的卡随即**删除本地副本**以省显存，权重回到分片状态。同时在用 $W_i$ 计算时可**预取** $W_{i+1}$，把通信藏进计算。最后一层通常保留权重不删，因为反向马上要用。
- **反向**：三件事并行——广播权重、各卡算本地梯度、把梯度**汇总到唯一 owner GPU**（不再是 DP 那样 all-reduce 到所有卡，而是 reduce 到 owner）；owner 拿到完整的 $\partial L/\partial W$ 后，只有它能更新该权重。更新后的权重不必立刻广播，下一轮前向会重新通信。稳态下在三个连续层上分别进行「算第 $L$ 层 backward」「聚合第 $L+1$ 层梯度并更新」「预取第 $L-1$ 层权重」。

![FSPD全分片数据并行机制示意图，展示GPU间权重广播与重叠计算过程](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0011_00-43-44.jpg)

CPU 上的 data loader 也会异步备好下一批数据，让 Tensor Core 尽量满负荷。这些系统本质上是「并行化机器」：GPU 内与 GPU 间的大量工作都要尽可能重叠。

## 7. HSDP、激活检查点与扩展配方 [48:51 - 58:06]

### 7.1 HSDP：二维并行的通信量设计 [48:51 - 52:43]

再进一步是 **Hybrid Sharded Data Parallelism（HSDP）**：把 $N=M\times K$ 块 GPU 组织成二维网格，分成 $M$ 组、每组 $K$ 块（$K$ 可达上百）。**组内做 FSDP**（权重分片、来回传权重与梯度），**组间做普通 DP**（各有一份完整权重，反向后跨组 all-reduce 梯度）。这就是多维并行（multidimensional parallelism）——同时使用两种并行策略。

![混合分片数据并行（HSDP）的GPU网格结构与通信模式示意图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0012_00-51-01.jpg)

好处来自通信量的差异：一次 forward+backward 中，FSDP 约需通信 **3 倍**网络权重（前向的 $W$、反向的 $W$ 与 $\partial L/\partial W$），而 DP 只需 all-reduce 梯度（约 **1 倍**）。于是把要求高带宽的 FSDP 组放在同一节点/pod 内（如单机 8 卡，900 GB/s），把只需慢通信的 DP 放在服务器之间——这是第一个「按已知网络拓扑设计算法」的例子，尽量利用快链路、把慢链路留给通信量小的轴。

### 7.2 显存账本与激活检查点 [52:43 - 55:54]

有了分片，模型可以很大：100B 参数 $\times$（4 个数 $\times$ 2 bytes）= **800 GB**，切到 80 块 GPU 上每卡仅 **10 GB**。但**激活值**会重新填满显存。以 Llama3-405B 为例：Transformer 有 **126 层**、$D=16{,}384$、序列长 **4096**，仅 FFN 的 hidden activations 就需要

$$2\times126\times(4\times16{,}384)\times4096\ \text{bytes}=63\ \text{GB}$$

再加上其他激活，GPU 很快 OOM。解法是 **activation checkpointing**：不再把所有激活都存在显存里，反向时**重算**。若每层前向算完立即丢弃，反向时逐层重算，计算量是 $n^2$、显存 $O(1)$（$n+(n-1)+\cdots+1$），对深网络太糟。折中做法是每 $C$ 层存一个 checkpoint，只在块内重算：计算量 $n^2/C$、显存 $O(C)$；常取 $C=\sqrt{n}$，得到 $n\sqrt n$ 计算、$O(\sqrt n)$ 显存——用计算换显存，从而训得动更大的模型。

![数据并行（DP/FSPD/HSDP）原理幻灯片，含双GPU权重分片示意图及100B模型显存计算](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0013_00-52-45.jpg)

![数据并行幻灯片更新版，新增Llama3-405B激活值显存占用问题分析](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0014_00-53-27.jpg)

### 7.3 一套能走很远的 scaling recipe [55:54 - 58:06]

把 DP + FSDP + HSDP + activation checkpointing 组合起来，有一套能走很远的配置经验：

1. 先用**原始 DP**，大约 ≤128 GPU、≤10 亿参数都够用；且几乎总应把每卡 **local batch size 设到打满显存**。
2. 模型超过约 10 亿参数，权重开始占满显存，转 **FSDP**。
3. 激活显存成为瓶颈时打开 **activation checkpointing**（它会明显变慢，但能训更大模型），可撑到数百 GPU。
4. 到几百块设备（视拓扑约 256 / 512）时 FSDP 太贵，切到 **HSDP**，可到约千卡、数百亿参数、较长序列。
5. 超过约 1000 GPU、>500 亿参数、序列长 >10,000 时，才需要 CP / PP / TP 这些进阶策略。

面对 global batch size、local batch size、HSDP/FSDP 维度、重算多少等一堆旋钮，答案是统一优化一个指标：**MFU**。

## 8. 指引指标：硬件 FLOPs 利用率与模型 FLOPs 利用率 [58:06 - 63:02]

### 8.1 从 HFU 到 MFU [58:06 - 59:53]

H100 的 989 TFLOP/s 是**理论**值。先看 **Hardware FLOPs Utilization（HFU）**：实际跑 kernel 时达到理论峰值的比例。讲者用几行 PyTorch 在 H100 上循环做稠密矩阵乘（矩阵规模从 512 到 32,000），在约 $8000\times8000$ 的大矩阵乘上得到约 **80% HFU**——相当不错。但 HFU 没算训练循环里的其他开销：数据加载、数据增强、激活重算、通信、旁路模型等。因此真正要盯的是 **Model FLOPs Utilization（MFU）**：GPU 理论峰值 FLOPs 中，究竟有多少用在「有用的」模型前向/反向计算上。它是调分布式训练时唯一的「指路明灯」。

![Model FLOPs Utilization (MFU) 概念定义幻灯片，提出核心问题：GPU理论峰值FLOPS中有多少用于有效模型计算](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0015_00-59-53.jpg)

### 8.2 五步算 MFU、经验阈值与代际差异 [59:53 - 63:02]

课件给出 MFU 的五个计算步骤：

```text
1. FLOP_theoretical     = forward + backward 中所有矩阵乘的 FLOP 总数
2. FLOP/sec_theoretical = 设备理论峰值吞吐（H100: 989 TFLOP/sec）
3. t_theoretical        = FLOP_theoretical / FLOP/sec_theoretical
4. t_actual             = 实测一次完整 iteration 的时间
5. MFU                  = t_theoretical / t_actual
```

$$MFU=\frac{t_{\text{theoretical}}}{t_{\text{actual}}}=\frac{FLOP_{\text{theoretical}}/FLOP/sec_{\text{theoretical}}}{t_{\text{actual}}}$$

先按模型架构（层数、层宽）算出一次完整 forward+backward 的理论 FLOP 数，除以设备峰值得到「理论最快耗时」，再实测一次 iteration 的真实耗时，二者相除即得 0–1 之间的 MFU。示例：一个带 ReLU 的宽 MLP、超宽层、超大 batch，在单块 H100 上约 **49–50% MFU**。

![MFU 计算的五个具体步骤及 PyTorch 实现代码示例，展示宽 MLP 在 H100 上达到约 49% MFU](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0016_01-01-11.jpg)

经验阈值：**>30% 算好**，远低于 30% 说明某处有巨大瓶颈，**>40% 优秀**（接近 SOTA）。Llama3-405B 论文最终训练阶段在 **8,000–16,000 GPU** 上取得 **high-30s / low-40s** 的 MFU，在 H100 上已很难更高。反直觉的是更新设备反而 MFU 更低：A100 上有时能 >50%，因为 A100→H100 理论算力约 **3×**，但内存带宽只约 **2×**——GPU 变快的速度超过通信变快的速度，这个不断拉大的鸿沟使新代设备有时更难喂饱。

![MFU 性能评估标准：大于 30% 为良好，大于 40% 为优秀](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0017_01-01-47.jpg)

## 9. 上下文并行、流水线并行、张量并行与 ND 并行 [63:02 - 72:52]

### 9.1 上下文并行 CP [63:20 - 66:02]

CP 切 **Sequence 维**：让不同 GPU 处理同一条长序列的不同片段。LayerNorm、FFN/MLP、残差连接本来就沿序列独立，切起来直接；MLP 里有权重，仍需像 DP 那样对梯度做 reduce。真正麻烦的是 **attention**：要计算序列内所有 token pair 的交互，QKV 投影容易并行，核心 attention 矩阵难并行。早期方案是 **ring attention**——把完整 attention 矩阵切块，让各 GPU 按正确顺序分块并行；概念上更简单的是 **Ulysses attention**——多头注意力本就并行算多个 attention 矩阵，于是按 **head** 并行核心 attention 算子，其余部分仍按序列维切。Llama3 预训练分两阶段：第一阶段序列长 **8,000**、完全不用 CP；第二阶段把序列拉到 **130,000**，此时启用 **16-way CP**，即 16 块 GPU 共同处理一条序列，相当于 batch size 只有 1/16。

### 9.2 流水线并行 PP 与气泡 [66:02 - 69:16]

PP 切 **$L$（层）维**：把网络各层分给不同 GPU，在 GPU 边界拷贝激活值。直觉上很自然，但存在顺序依赖——每块 GPU 要等上一块的激活才能继续前向，反向也依赖上游梯度。时间线图（纵轴 GPU1–4、横轴时间）里，GPU1 forward → GPU2 → GPU3 → GPU4，GPU4 一口气做完 forward+backward 再把梯度逐层回传；结果大部分 GPU 空转，$n$ 块卡只有约 $1/n$ 的时间在做有用功，8-way PP 的最大 MFU 只有约 12%（4-way 时 $1/4=25\%$）。这段空闲有个可爱的名字 **bubble**。

![流水线并行（PP）示意图，展示模型层在4个GPU间的切分及激活值传输路径](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0019_01-06-06.jpg)

缩小 bubble 的办法是 **microbatches**：同时让多个 microbatch 在流水线里流动。4-way PP + 4 个 microbatch（蓝/黄/绿/红）时，GPU1 依次做四个 batch 的 forward，GPU2 在 GPU1 做黄色时就能做蓝色的 forward，反向同理交错级联；最大理论 MFU 从 $1/4=25\%$ 提升到 $16/28\approx57.1\%$（Huang et al., "Gpipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism", arXiv 2018）。microbatch 越多 MFU 越好，但要存下更多激活，于是又要 activation checkpointing——PP 度数、microbatch 数、重算力度、是否再叠 DP……这么多旋钮，答案仍然是「最大化 MFU」。

![流水线并行（Pipeline Parallelism）的微批次调度示意图，展示了4个GPU上的前向与反向传播时间线及气泡问题。](/coursenotes/cs231n-deep-learning-for-cv-sp25/l11-distributed-training/kf_0020_01-08-29.jpg)

### 9.3 张量并行 TP [69:16 - 71:15]

TP 切 **Dim 维**：把**单个**权重矩阵切到多块 GPU 上（与 FSDP 不同——FSDP 按层把不同矩阵分给不同 GPU，TP 是切同一个矩阵）。模型里反复在做 $XW=Y$；把 $W$ 切成 $W_1,W_2,W_3,W_4$，每卡用完整输入算一个分块矩阵乘、得到输出的一个切片，随后需要 gather 激活才能进入下一层。有个巧妙技巧：若有**两个连续的层**，把第一个矩阵按**列**切、第二个按**行**切，利用分块矩阵乘的代数，最终输出可以写成这些分块结果的某种内积结构，从而**每两层才通信一次**、不必每层 gather。这正好契合 Transformer FFN 里的两层 MLP，所以大型 Transformer 常在 MLP 上采用这种两层 TP。

### 9.4 ND 并行与全讲总结 [71:15 - 72:52]

哪种并行最好？答案是**全都要（ND parallelism）**。HSDP 已经是二维并行的例子，而当前 SOTA 是四维：Llama3 最大规模训练（16,000 GPU）同时使用 **8-way TP × 16-way CP × 16-way PP × 8-way DP**。不同并行机制的通信需求不同，只要把各并行轴沿集群拓扑精心排布，就能匹配不同层级链路的速度差异。全讲收束为三句话：单块 GPU 是可泛化的并行计算机；GPU 集群是把成千上万块 GPU 编程成一台巨型并行机；再加上多种并行切分、activation checkpointing 省显存，以及设计这些流水线时唯一该优化的指标 **MFU**。

## 要点回顾

- 大规模训练已是新常态（单卡→数万卡并发）；Llama3-405B 因公开系统与训练细节成为贯穿案例，而 GPT-4 之后业界不再公开架构、硬件与训练方法。
- H100：80 GB HBM（片内约 3 TB/s / 3352 GB/s）、50 MB L2、132 个 SM（按 144 设计、因 binning 只启用 132）；单个 SM 有 256 KB L1 + 256 KB 寄存器 + 128 FP32 core + 4 Tensor Core。
- Tensor Core 做 16-bit 输入 / 32-bit 累加的混合精度矩阵乘，吞吐约 4096 FLOP/SM/cycle，远高于 FP32 的 256；PyTorch 里忘转 16-bit 会慢约 20 倍。
- 单卡算力 2013 K40 5 TFLOP/s → B200 5000 TFLOP/s TC，约 1000×（V100 首引 Tensor Core）；带宽悬崖 3352 → 900 → 50 → <50 GB/s，Llama3 集群 24,576 GPU / 1.875 PB / 24.3 EFLOP/s，训练常以月计。
- 其他硬件：Google TPU v5p（459 TFLOP/s BF16、95 GB/chip、8960-chip pod，仅可云租）、AMD MI325X、AWS Trainium。
- 四条切分轴：DP 切 Batch、CP 切 Sequence、PP 切 $L$、TP 切 Dim；核心是把通信与计算重叠，让所有计算单元都有活干。
- DP 靠梯度线性与单卡大 batch 严格等价，backward 后 all-reduce 平均梯度并与 backward 重叠（PyTorch DDP）；异步 SGD 不稳、难复现，但可能回潮。
- 显存账：每参数 4–5 个数 × 2 bytes ≈ 8 bytes → 10 亿参数 8 GB，DP 上限约 100 亿参数；FSDP 按层分片 + 广播/删除/预取，100B 参数切 80 卡仅 10 GB/卡；HSDP 用二维网格把组内 3× 通信与组间 1× 通信按拓扑排布。
- 激活检查点用计算换显存：Llama3-405B（126 层、$D=16384$、seq 4096）仅 FFN 激活就 63 GB；全重算 $n^2$ 计算 / $O(1)$ 显存，每 $C$ 层 checkpoint 得 $n^2/C$、$O(C)$，取 $C=\sqrt n$ 得 $n\sqrt n$、$O(\sqrt n)$。
- MFU 是唯一指路明灯（>30% 好、>40% 优秀，宽 MLP 约 49%，Llama3 在 8k–16k GPU 达 high-30s/low-40s；H100 算力 3× 而带宽仅 2×，MFU 反可能低于 A100）；进阶 CP（ring/Ulysses attention、Llama3 130k 序列 16-way CP）、PP（bubble + microbatches 把 25% 提到 57.1%）、TP（列切+行切每两层通信一次，契合 FFN 两层 MLP），SOTA 是 8 TP×16 CP×16 PP×8 DP 的四维并行。

## 视频来源

https://www.youtube.com/watch?v=9MvD-XsowsE
