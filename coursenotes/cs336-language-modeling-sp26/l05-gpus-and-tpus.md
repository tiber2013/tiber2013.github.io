---
title: "第 5 讲 · GPU 与 TPU：硬件与性能模型"
description: "这讲把 GPU 视为一台为吞吐量而设计的并行机器：大量 SM、warp 和线程执行相同指令，而计算能力的增长远快于内存带宽，因此性能优化的核心是减少并复用全局内存访问。课程先建立 SM、SIMT、warp 与内存层级的心…"
---

# 第 5 讲 · GPU 与 TPU：硬件与性能模型

> **Stanford CS336 语言模型从零构建 · L05** ｜ 讲者 Stanford Online ｜ 79 分钟 ｜ [原视频](https://www.youtube.com/watch?v=izZba4UA7iY)

> **v2 详写版** · 按 20 分钟分段详写，配图覆盖全片；文末附「对照参考笔记补遗（Spring 2025 版）」，记录与 2025 版讲义笔记的核对结果。

## 核心结论

本讲把视角从模型本身转向底层硬件，目标是拆掉 CUDA 与 GPU 的魔法感：先建立硬件与编程的心智模型，再解释性能为何起伏，最后用这些原理重新发明 FlashAttention。核心判断是 GPU 是一台为吞吐而生的并行机器——CPU 优化少量线程的低延迟，GPU 用大量轻量 SM、warp 与 SIMT 锁步执行换取聚合吞吐；而算力的增长长期快于内存与互连带宽，这种剪刀差使内存而非算力成为瓶颈，也让有专用 Tensor Core 支持的矩阵乘成为远快于其它浮点运算的特权操作。于是 roofline 模型把优化目标写成让算术强度足够高、把程序推到计算受限的平台段，课程据此给出六个手段：控制发散、低精度、算子融合、重计算、合并访存与最重要的 tiling，其中除控制发散外都围绕减少、复用或置换全局内存访问。攒齐这些零件后，方阵矩阵乘吞吐曲线上原本神秘的对角上升、按整除性分色的差距与周期性下凹都能拆开，分别对应 roofline、burst 对齐，以及 tile 数跨过 SM 数而多跑一波的 wave quantization。终章 FlashAttention 把分块矩阵乘、online softmax、算子融合与反向重计算拼成单个融合 kernel，靠压低 HBM 读写而非增加 FLOPs 换来加速，印证了全讲落点：理解硬件到低层细节、认真对待矩阵乘与数据移动，才是硬件感知架构设计的前提。

## 从神经网络转向系统：本讲定位、目标与可用资源 [00:00 - 04:14]

讲者在 [00:05] 宣布课程进入 systems（系统）部分：前两讲一直在做 neural network（神经网络）的内容，从现在起转向 GPUs、parallelization（并行化）与 inference（推理）。他对系统部分给出一个很鲜明的定位——这是整个技术栈里"最讲得通"（most reasonable）的一块，因为你可以把每个部件都拆开来推理，顺着一条条逻辑步骤走到结论，这种可推导性和确定性"很不一样，也很令人满足"。但这个开场同时埋了一个反差：尽管系统可推理，人第一次和 GPU 打交道时，感受却往往是"既像魔法，又像在跟一台非常奇怪的设备交互"。讲者现场做了个小调查，问有多少人上过任何形式的 GPU programming class，举手者大约四分之一、可能还略少（a fourth, maybe a little bit less），他评价"这已经相当不错了"，并顺势点出本讲的动机：deep neural networks 让人觉得像魔法，GPUs 也一样，而把这份魔法感拆掉正是本讲要做的事。画面里还留着上一讲推导的白板，能辨认出 $P^{\top}b\,u$、$\min C(u)+P^{\top}b\,u$ 一类式子，以及 $c_i$、$u_i$、$M_i$、`softmax`、$u^0=0$ 和形如 $\text{if } P^{\top}b_i > c_i \Rightarrow u_i^{*}=M_i$ 的分段条件分支——这些是神经网络与优化部分的残留，正好构成"从 NN 转向 systems"的视觉注脚，也提醒我们本讲并不接上一讲的具体公式往下推。

紧接着讲者抛出一张会贯穿全讲的"神秘图"：矩阵乘（matrix multiply）的 throughput（吞吐）随矩阵维度变化的曲线，x 轴是 dimension（维度），y 轴是吞吐。直觉上人们会以为"矩阵越大、可做的功越多，吞吐自然越高"，但图上出现一堆 crazy patterns：某些尺寸的 matmul 明显比相邻尺寸慢得多。讲者把这句话当作给全班的承诺（my promise to you）：到本讲结束时，你会看懂这张图为什么长成这样。要兑现这个承诺，需要两件事：一是理解 GPU 的心智模型以便构建 fast algorithms，二是理解硬件执行模型本身。他特别强调，即便你不是 systems person——他估计班里多数人甚至大多数人都不是——只要你在思考 architectures（架构）、在一般性地推理这些模型，就必须理解模型将要在什么样的硬件上执行，否则做不出 effective（有效率）的模型；这直接呼应 Percy 的第一讲：scaling（规模化）很大程度就是确保资源被有效使用，而不懂系统就永远不可能高效地使用资源。换句话说，本讲不是系统工程师的选修课，而是架构设计的必修前提。

![Outline and goals 幻灯片：左侧是矩阵乘吞吐随维度变化的曲线与 Tiling、Compute Intensity、Wave Quantization 手写批注，右侧是 FlashAttention 分块示意图与 GPT-2 注意力运行时间对比](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0002_00-01-50.jpg)

这张 "Outline and goals" 幻灯片把目标写得很直白，主标题是 `Make CUDA and GPUs less magic`，下面并列两个子目标 `Understand when GPUs get slow` 与 `Understand how to make fast algorithms`。左边那张吞吐曲线正是讲者承诺要解释的图：横轴刻度依次是 512、1024、1536、2048、2560、3072、3584、4096，纵轴标着 TF/s、范围到约 250；图上还有手写的绿色批注 `Tiling`、`Compute Intensity`、`Wave Quantization`，用箭头和圆圈圈出曲线抖动与下凹的区域，横轴附近还写了个 `128`——这些术语在本段还没展开，却是后半讲解释曲线形状的全部钥匙，属于"图上出现但讲者此刻没念到"的典型增量信息。右边则是 `Understand how to make fast algorithms` 的落点：一张 FlashAttention 的分块示意图，标着 `Outer Loop`、`Inner Loop`、`SRAM`、`HBM`，Q、K、V 三个矩阵块与注意力计算被拆成两层循环；旁边是 "Attention on GPT-2" 的柱状对比，纵轴 Runtime (ms)，PyTorch 一侧把 Matmul、Dropout、Softmax、Mask 逐个拆成算子，FlashAttention 一侧则合成一个 `Fused Kernel` 加少量 Matmul，图下注明 `Dao et al, FlashAttention`。把注意力写成公式就是 $\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d}}\right)V$，而本讲最终要回答的是：为什么把它融合、分块之后能从那根柱子里省下大量时间。

讲者随后把本讲拆成三部分（[03:19] 起）：第一部分是 broad introductory material on GPUs，只讲 hardware model（硬件模型）和 GPU 编程大致怎么工作，重点是让你从哲学上理解它为什么和 CPU 不同，因为 GPU 是 "a foreign device"（一台外来设备）；第二部分是"大概六个"重要 tricks，即让 GPU 跑快的一组 building blocks，教你怎么摆弄代码才能在 GPU 上高效执行；第三部分把前两部分合起来，walk through flash attention，他称之为本讲的 "victory lap"（胜利一圈）——到那时应当已经理解到足以 "reinvent our own flash attention"（重新发明一遍 FlashAttention）并说清每个组件。这个结构在幻灯片上被压缩成一句 `Make CUDA and GPUs less magic`，但三部分之间的依赖很清楚：没有第一部分的心智模型，第二部分的技巧只是口诀；没有前两部分，第三部分的综合案例就无法拆解。

在正式开始之前，讲者用一整页致谢式清单推荐了延伸资源（[02:24]–[03:12]）：Horace He 的博客（幻灯片卡片上写着 `Thonk From First Principles`，副标题 `ML Systems from first principles. Aims to be better than a ChatGPT summary`），一个叫 CUDA MODE 的爱好者社区（现场有学生当场纠正 "They're GPU mode now. They changed their name"，讲者从善如流改口 "GPU mode"），以及 Google 那批人写的 TPU book——他特意补充"现在它也是一本 GPU book 了"，认为这是 a really incredible resource，还配了 exercises，并预告作业里会有"看起来有点像这些练习"的题目，等于提前给后面的 assignment 打了预防针。把这张 "Before we start.." 幻灯片上的信息照录下来，比转述更能说明本段的资源密度：

```text
Before we start..
Substantial credit goes to a few sources that I'd like to highlight..
Thonk From First Principles - ML Systems from first principles.
                              Aims to be better than a ChatGPT summary.
CUDA MODE group               TPU (and now GPU) book (!!)
How to Scale Your Model - A systems view of LLMs on TPUs
```

这里有几个容易踩的坑。第一，别把"矩阵越大吞吐越高"当作规律——讲者专门用这张曲线反驳的正是这个直觉，而幻灯片上手写的 `Wave Quantization` 已经提前剧透了答案的一部分。第二，别误以为只有 systems person 才需要懂硬件；讲者反复说，做架构、做模型设计的人同样要理解执行模型。第三，社区名称以现场更正为准是 `GPU mode`，幻灯片上的 `CUDA MODE` 是旧称，两者指同一个群组。第四，本段末尾讲者还没开始讲任何具体机制，所有 `Tiling` / `Compute Intensity` / `Wave Quantization` 都只是伏笔，此时不要替他下结论；白板上的公式属于上一讲，也不该被误读成本讲内容。下一章才会从"为什么需要 GPU"这个更根本的问题切入，说明计算规模是怎么一路把我们从串行频率推到并行吞吐的。

## 从频率到并行：Dennard scaling 的终结与并行算力曲线 [04:14 - 07:31]

本段开头先把"算力"确立为整门课的通用货币。讲者说，虽然还没正式讲 scaling，但从 Percy 的导论课或一般性地跟进 language modeling 就能知道：手上有更多 compute（算力），就能得到更好的 language models，这是"我们赖以运转的货币"（the currency by which we operate）。因此更快的硬件、更好的 utilization（利用率）、更多的芯片、改进的 parallelization（并行化），全都能推动进步；过去几年的游戏一直是"如何把更多算力投入到训练里"，而最近一年左右，问题又多了"如何把更多算力投入到这些模型的推理里"。配套幻灯片的标题是 `Setting the stage: compute leads to predictable perf`，副标题 `Often times, compute leads to predictable performance gains for language models`：图里画的是 Kaplan 等人的 Neural Scaling Laws 曲线，横轴 Compute (PetaFLOP/s-days) 取对数刻度，纵轴 Validation Loss 从约 1.5 到 6，右侧色条是 Parameters（参数量，约 $10^5$ 到 $10^{11}$），图中还标注了拟合式

$$
L = 2.57 \cdot C^{-0.048}
$$

底部一句 `Faster hardware, better utilization, improved parallelization alone can drive progress (for now..)` 尤其值得留意：括号里的 "for now.." 是讲者自己留的余地，说明"算力提升带来可预测收益"是当下的经验规律，不是永恒定律。这一页把上一章结尾"系统是架构必修前提"的说法量化了——既然损失随算力按幂律下降，那么谁能把硬件用得更满，谁就能沿这条曲线往前走。

![Setting the stage 幻灯片：Kaplan 神经缩放定律曲线，验证损失随计算量按幂律下降，右侧色条为参数量](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0004_00-04-18.jpg)

接着讲者把时间拨回上世纪九十年代：那时若讨论"怎么让计算机变快"，焦点一定是 clock speed（时钟频率）——CPU 串行地执行指令，我们让这些指令执行得越来越快。晶体管变小、时钟变快这种扩展叫 Dennard scaling（登纳德缩放），它"相当长一段时间里都很成立"，真正见顶是在 2000 年代。讲者特意加了一个 side note：晶体管数量当然还在继续上涨，但把晶体管做小并不必然让晶体管跑得更快，它们不会以更高的时钟频率切换，到了某个点之后，存在 fundamental physical reasons（根本的物理原因）使你无法继续从越做越小的晶体管里拿到好处。于是"把时钟做快、让指令执行更快"这条老路走不通了，问题变成：在一个无法靠频率继续提速的世界里，怎么让 LLM 继续 scale？幻灯片上那张著名的 "42 Years of Processor Data" 图（约 1970 到 2020 年，纵轴 $10^0$ 到 $10^7$ 的对数刻度）把这个过程画得很清楚：图例里有 Single Thread Performance (SpecINT × IPC)、Frequency (MHz)、Typical Power (Watts)、Number of Logical Cores、Transistors (thousands)，曲线上还标着 Moore's Law、"The Free Lunch Is Over" 一类注记——频率在 2000 年代中期撞墙、功耗上升、逻辑核数开始增加，而晶体管数仍单调上升。页面底部写着 `But the traditional form of scaling (Dennard scaling) from 1980-2000s has tapped out. .. How do we feed LLMs' insatiable appetite for compute?`

答案就是 GPU 所代表的 parallel scaling（并行扩展）。讲者说 Percy 下一讲会讲更高级的主题，再往后就是 parallelism，而并行与 GPU 其实是同一个 paradigm（范式）的两面：不再让单条指令流串行地越跑越快，而是 scale horizontally（横向扩展），让更多执行单元同时处理你的指令。他用一张图概括并行扩展的现状：早年 K20s、M40s 时代的 flops 按今天标准看是 "respectable, but somewhat pitiful"（尚可，但有点寒酸），而到了 P100s、V100s 附近曲线突然起飞，出现 super exponential（超指数）的浮点算力逐年增长。他把驱动因素逐个点名：V100 时代、也就是 2017 年，开始出现 Tensor Cores，这是重大硬件创新之一；随后几年 structured sparsity（结构化稀疏）、转写记作 "FPA" 的低位 number formats 以及各类 lower number formats，继续把 flops 往前推，并预告"稍后会详细看它们意味着什么"。

![Parallel scaling continues 幻灯片：收益分解清单与「十年千倍」单芯片推理性能曲线，底部强调没有 GPU 扩展就没有 LLM 扩展](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0006_00-06-45.jpg)

这张幻灯片的信息量远大于讲者口述。左侧 `Gains from` 把"千倍增益"拆成四层，右侧是 Bill Dally 在 HotChips keynote 里的 `Single inference performance | 1000X in 10 years` 曲线，图上依次标出 P100、V100、A100、H100 以及 `2.5x Sparsity`、Winograd、Tensor Cores 等注记；底部结论是 `Parallel scaling with GPUs has scaled > 1000x in 10 years. There is no LLM scaling without GPU scaling`。照录这些条目，可以清楚看到增益并非来自单一因素：

```text
Gains from
  Number Representation   FP32, FP16, int8 (TF32, BF16)   16x
  Complex Instructions    DP4A, HMMA, IMMA                12.5x
  Process                 28nm, 16nm, 7nm, 5nm            2.5x
  Sparsity                                                ~2x
  Model efficiency has also improved - overall gain > 1000x
Parallel scaling with GPUs has scaled > 1000x in 10 years.
There is no LLM scaling without GPU scaling
```

把这几项写成连乘，就得到整页的核心数量关系：$\text{总体增益} > 16\times(\text{数值表示}) \times 12.5\times(\text{复杂指令}) \times 2.5\times(\text{工艺}) \times 2\times(\text{稀疏性}) \approx 1000\times$，而且幻灯片还额外提醒 `Model efficiency has also improved`，即模型效率本身的提升没有被算进这四项里。这里有几个易错点：其一，别把 1000x 全归功于制程，Process 一项只有 2.5x，真正的大头是 number representation（16x）与 complex instructions（12.5x）；其二，别把"晶体管继续变多"等同于"单线程继续变快"，这正是 Dennard scaling 见顶后最反直觉的地方；其三，Tensor Cores 是 2017 年 V100 时代才引入的，FP8、structured sparsity 一类低精度与稀疏手段是随后几年才叠加进来的，不能倒推到 K20/M40 时代；其四，`for now..` 与 `Model efficiency has also improved` 都提示这条曲线是经验叠加，不是可无限外推的物理定律。与前后内容的关系上，本章把上一章的"资源要用得有效"落到了具体的历史曲线上：正因为串行频率这条路断了，GPU 才会被设计成下一章要讲的那种"为吞吐而生"的机器；而本章点名的低精度、稀疏、专用指令，又正是后面六个 tricks 里"低精度计算"的伏笔。下一章将正式回答"GPU 到底和 CPU 有什么哲学上的不同"。

## CPU 与 GPU 的哲学差异：延迟优化与吞吐优化 [07:31 - 09:02]

讲者用一句 `How is a GPU different from a CPU?` 开启本段，并先安抚已有 GPU 经验的同学"我们马上就讲更进阶的东西"，但对其他人强调，理解二者的 model、乃至 philosophy（哲学），是最重要的一步。CPU 是为 quick serial execution（快速串行执行）设计的：它要应付 complex branching logic（复杂分支逻辑）、complex conditionals（复杂条件判断）、complex control flows（复杂控制流）。正因为要伺候这些复杂性，CPU 一般配 really big control units（很大的控制单元），只带 few ALUs（少量算术逻辑单元）来做数学或其它运算，但一切都要跑得极快、追求 low latency（低延迟）——从拿到一条指令到把它执行完，时间应当非常非常短，这就是 CPU 的整体设计哲学。相对地，GPU 关心的是 throughput（吞吐）：你派发一个任务，它可能要很久才结束；可能处理一会儿、停下来去干另一个任务、再回来接着干，单条任务并不快，但所有任务加起来能得到很大的 aggregate throughput（聚合吞吐）。

实现这种吞吐靠的是 tons and tons of lightweight cores（成百上千个轻量核心）：一块芯片里塞进 hundreds and hundreds of compute units，全部可以并行执行。讲者说这是"一种和 CPU 非常不同的编程与思考模式"。幻灯片把这句话写成 `CPUs optimize for a few, fast threads while GPUs optimize for many many threads`，并用两幅框图对照：CPU 一侧是一个大 `Control` 块、四个小小的 `ALU` 块、一大块 `Cache` 和最底下的 `DRAM`；GPU 一侧则是一整片绿色 ALU 方格阵列，控制部分只剩左边一条极窄的列，底下同样接 `DRAM`。图下注脚是 `Many tiny compute units (ALUs). Much less support for branching (control, cache)`，右下角两句结论分别是 `CPUs optimize for latency (each thread finishes quickly)` 与 `GPUs optimize for throughput (total processed data)`，右侧还有一幅用 `Computation Thread`、`Processing`、`Waiting for data`、`Data to be processed` 标注的时序小图，画出"单条线程漫长等待、但很多线程重叠填满时间"的情形，图下注明来源为 NVIDIA 的 CUDA refresher 博文。把幻灯片原文照录如下：

```text
CPUs optimize for a few, fast threads while GPUs optimize for many many threads
Many tiny compute units (ALUS).
Much less support for branching (control, cache)
CPUs optimize for latency (each thread finishes quickly)
GPUs optimize for throughput (total processed data)
```

![How is a GPU different from a CPU? 幻灯片：CPU 侧为大 Control 配 4 个小 ALU 与 Cache，GPU 侧为整片绿色 ALU 阵列，图下注明延迟优化与吞吐优化](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0007_00-07-59.jpg)

如果把这两句话改写成目标函数，差异就一目了然：

$$
\text{CPU: } \min \; L_{\text{per-thread}}, \qquad
\text{GPU: } \max \; \frac{N_{\text{tasks}} \times \text{work per task}}{T_{\text{total}}}
$$

也就是说，CPU 优化的是分母意义上的"单任务完成时间"，GPU 优化的是单位时间内处理掉的总数据量。这与上一章的历史叙事正好接上：既然频率无法再降延迟，那么继续提升性能的唯一出路就是让 $N_{\text{tasks}}$ 变大、让大量任务重叠，用吞吐补延迟。图中 CPU 只画了 4 个 ALU 方块，而 GPU 是一整片阵列，讲者口述则是"hundreds and hundreds"个计算单元——这些具体数量级说明，两种芯片不是"同一种东西快慢不同"，而是把晶体管预算花在了完全不同的地方：CPU 花在 control 与 cache 上，GPU 花在 ALU 阵列上。

这里有几个容易混淆的地方。第一，别用"某个任务多久跑完"去衡量 GPU 好坏：单任务延迟高是设计取舍，不是缺陷；反过来，若用"总共处理了多少数据"去衡量 CPU，也会低估它对分支与低延迟的价值。第二，GPU 的 "lightweight core" 不能按 CPU 的完整核心去理解，幻灯片明说它 `Much less support for branching (control, cache)`，这意味着复杂控制流在 GPU 上是弱项，也为后面要讲的"控制发散"埋下伏笔。第三，这两幅框图是 pictorial illustration（示意性图示），绿色方格数量不代表真实晶体管比例，不要拿它去推算 ALU 的确切个数。第四，讲者说"马上讲更进阶的东西"，但本段到此（[09:01] 的 `Okay?`）为止只建立了哲学对比，还没给出任何硬件细节；下一章才会把"大量轻量核心"具体拆成 SM、并把 GPU 的另一半——内存——正式引入，因为现代性能优化真正的瓶颈并不在算力这一侧。

## GPU 硬件解剖：SM、执行单元与内存层级 [09:02 - 13:26]

讲者说要 demystify（去神秘化）GPU：不希望学生把它想象成 NVIDIA 官网上的一段文字，而要能在脑里 visualize（可视化）出硬件长什么样。他先给出基本单元 SM，即 streaming multiprocessor（流式多处理器）：它"有点像一个 core"，是自带子组件、能加速计算、并能连到某些内存的独立计算单元。沿着上一章的并行思路，SM 内部并不是单一计算部件，而是含许多 streaming processor，可以并行执行不同 threads；但 SM 本身才是那个 discrete compute unit（离散的计算单元），讲者特意说"后面讲到编程模型时你们会明白为什么这么划分"。数量上他给了明确例子：一块 A100 有 128 个 SMs，每个都能被独立编程、独立执行不同 jobs，并且都能访问外面的 global memory。配套幻灯片 "Anatomy of a GPU (execution units)" 左边是放大的 SM 内部图（成列绿色块、以及 L1/Shared Memory、Warp Scheduler、Register File 一类标注），右边是 `GA100 Full GPU with 128 SMs` 的整芯片布局：绿色小格是 SM，中间横贯一条蓝色 L2 带；图下两行说明分别是 `Each SM further contains many SPs (streaming processor) that can execute 'threads' in parallel` 与 `GPUs have many SM (streaming multiprocessors) that independently execute 'blocks' (jobs)`。注意这里已经悄悄跨越了本章与下一章的边界：SP 对应 thread、SM 对应 block 的映射，要等讲完执行模型才能完全对上号。

![Anatomy of a GPU (execution units) 幻灯片：左侧为 SM 内部的 SP 阵列，右侧为 GA100 整芯片 128 个 SM 与 L2 的布局](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0008_00-09-13.jpg)

讲者随即把重心一转：compute（计算）只是故事的一半，而且"贯穿本讲会不断看到，真正非常非常重要的另一半是 memory（内存）"——他甚至说 modern hardware and LM optimization is really defined by the memory（现代硬件与语言模型优化其实是由内存定义的）。于是必须搞清楚有哪几种内存、它们住在哪。他先按速度口述分层：有不同 levels of caches，有更有限也更快的 registers（寄存器），再往下是 global or high bandwidth memory（全局内存/高带宽内存）。接着引用一张被他称作"一些人对 A100 延迟做的 benchmark"的表：L1 和 shared memory 很快，约 20 到 30 cycles 就能读到数据；命中 L2 cache 就慢得多；而 global memory 比 L1 慢大约 10 倍。他特别提醒一个常见误解：高层程序员平时说的"GPU 显存"就是 global memory，例如一张 H200 告诉你"我有 144 gigs of memory"，指的是全局内存，绝不是 shared 或 L1 cache。幻灯片把这张表定量化了：

```text
TABLE IV  THE MEMORY ACCESS LATENCIES
Memory type             CPI (cycles)
Global memory           290
L2 cache                200
L1 cache                33
Shared Memory (ld/st)   (23/19)
```

![Anatomy of a GPU (memory) 幻灯片：A100 内存访问延迟表、GA100 芯片物理布局与板卡照片，底部注明 SRAM 成本高 100 倍但快约 8 倍](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0009_00-10-26.jpg)

把表写成不等式，层级关系就是 $290\ \text{cycles (Global)} > 200\ \text{(L2)} > 33\ \text{(L1)} > 23/19\ \text{(Shared, ld/st)}$，且 $\frac{290}{33}\approx 8.8$，正好对应讲者口述的"约 10 倍"；shared memory 的 `(23/19)` 则对应 load 与 store 两种操作的细微差别。幻灯片副标题进一步给出物理位置：`The closer the memory to the SM, the faster it is – L1 and shared memory is inside the SM, L2 cache is on die, and global memory are the memory chips next to the GPU`——L1 与 shared memory 在 SM 内部，L2 在芯片上（on die），而 global memory 是紧挨着 GPU 的独立内存芯片。右侧 GA100 的 die 图把这点画成实物：绿色块是 SM，蓝紫色块是 `L2 Cache Partition`，上下边缘是 `HBM2(0) PHY` 与 `16 512-bit Memory Control`；旁边的规格清单一口气给了大量具体参数：

```text
Nvidia GA100, 7nm TSMC
8 GPC, 64 TPC, 128 SM
8192 FP32 Units / 512 Tensor Units
48MB L2 Cache / 6144-Bit HBM2(e)
Die size w/ scribe lines: 836 mm2
Die size w/o scribe lines: ~ 826mm2
```

左下角还有一张 GPU 板卡照片，用红字标出 `Interconnection interface`、`Graphics Processing Unit (GPU)`、`Video Memory (VRAM)`、`Network Interface`、`Memory Interface`、`Voltage Regulators` 等部件，把"全局内存在芯片之外"这件事落到实物上。页面底部一句结论尤其关键：`SRAM (shared/cache memory) is much more expensive (100x) but ~ 8x faster than DRAM (Global memory)`——注意讲者口述说 shared memory "贵上百倍（hundreds of times）"，幻灯片给的是 `100x`，两者量级一致，取幻灯片数值为准；他还补了一句它 `much more power hungry`（更耗电）。于是一个自然的追问被提了出来：既然 shared memory 这么好，为什么不拿它把整块芯片填满？答案正是成本与功耗，所以实用加速器只能做 memory hierarchy（内存层级），并要求程序去有效利用它。讲者顺带提到一个"以巨量 SRAM 为特色、据称最近被 Nvidia 收购"的设计（口述作 Grok），说这类设计对 inference 这类负载很有帮助，但多数加速器仍然必须尊重层级才能跑快。

[12:54] 有学生追问 shared memory 与 L1 cache 到底差在哪，这个问答正好点破本章最易混的一处：讲者说二者"运作方式不同"——cache 就是个 cache，自动存放 recently accessed data elements（最近访问过的数据元素）；而 shared memory 是 programmable element（可编程部件），你要显式地把东西放进去、取出来。也就是说，L1 与 shared memory 虽然延迟都在 20–30 cycles 量级、物理上都紧贴 SM，但一个是硬件自动管理、一个是程序员可控，后者才是后面 tiling 能成立的前提。本章的易错点还有几处：其一，"GPU 有多少 GB 显存"永远指 global memory（H200 的 144 GB），别把 L2 的 48MB 或 shared memory 混进来；其二，290/200/33/23/19 是 A100 的一次 benchmark 结果，不是所有 GPU 的通用常数；其三，`100x` 成本与 `8x` 速度是量级对比，说明 SRAM 不是"越多越好"，Grok 那种全 SRAM 方案是特例而非通则；其四，别只盯着 8192 个 FP32 单元这类算力数字，讲者反复说优化由内存定义。与前后内容的关系上：上一章"大量轻量核心"在这里被具体化为 128 个 SM 与其中的 SP；而"block 跑在单个 SM 上才能用它的 shared memory"这一约束，正是下一章执行模型的核心，也是后面所有减少 global memory reads 技巧的出发点。

## 执行模型与内存模型：Thread、Block、Warp、SIMT 与可访问范围 [13:26 - 17:18]

讲者说硬件心智模型已经就位，现在开始讲 GPU 的 software and programming model（软件与编程模型）。他指出 execution model（执行模型）里有三个重要 players：`thread`、`block`、`warp`。thread（线程）是并行做事的轻量单位，任何并行编程的人都熟悉；GPU 里特别之处在于线程遵循 SIMT 模型（Single Instruction Multiple Threads，单指令多线程）——所有线程执行完全相同的指令，但输入不同，因此各线程仍能做各自有用的工作。讲者强调这"对你的代码效率和编程模型都有重要影响"：如果线程能各干各的，编程会变得极难，所以 GPU 本质是 programmability（可编程性）与 efficiency（效率）之间的一笔 trade-off。block（块）是一组线程，它之所以是关键对象，是因为一个 block 保证跑在单个 SM 上；而上一章刚说过每个 SM 有自己的 shared memory，于是同一 block 内的线程可以共享一个内存池——讲者明说这会在后面谈 tiling 时变得极其重要，因为 block 会反复复用同一批内存。warp（线程束）则是 GPU 内部的调度单位：线程以 32 个连续编号（32 consecutively numbered threads）为一组执行，这降低了调度器决定"下一步跑哪些线程"的开销；他补充说以后自己若讲"不同的 warps 在执行"，指的就是这个 32 线程一起执行的基本单位。幻灯片 "Execution model of a GPU" 把三条定义写成原文，图中的 Grid 由 `Block (0,0)`、`Block (1,0)` 等构成，SM 一侧标着 L1 与 Shared Memory，右侧绿色列标着 `Warp` 与 `Thread (0,0)`、`Thread (1,0)`，并注明 `Single Instruction Multiple Threads`：

```text
There are 3 important players in the execution model
Threads: Threads 'do the work' in parallel - all threads execute the same
         instructions but with different inputs (SIMT).
Blocks: Blocks are groups of threads. Each block runs on a SM w/ its own
        shared memory.
Warp: Threads always execute in a 'warp' of 32 consecutively numbered
      threads each.
```

把这三者的约束写成式子，就是 $\text{warp} = 32\ \text{个连续编号线程}$、$\text{block} \mapsto 1\ \text{SM}$，以及 SIMT 的核心条件 $\forall t \in \text{warp}:\ I_t = I,\ \text{但 } x_t \ne x_{t'}$——指令 $I$ 相同、输入 $x$ 不同。[15:22] 有学生问"所有线程执行相同指令"到底指 block 内全部线程还是 warp 内全部线程，讲者明确答是 warp 内；[15:34] 又有人追问调度器是不是决定"下一条指令发给哪个 warp"，他纠正为调度器决定"下一个执行的是哪个 warp"（which warp is going to be executing next），属于 warp assignment。这两个问答把粒度钉死了：SIMT 的锁步是 warp 级，不是整个 block 或整个 grid 级。

[15:49] 起讲者转向 memory model（内存模型），逐层说明 device code（设备端代码）能读写什么：registers 是最快最局部的，可用来存"数组从哪开始、到哪结束"这类内存地址；local memory 局限于单个 SM 内；shared memory 能在 block 内访问，凡是要跨线程传递、或要被多个线程复用的信息都放这里；再往外可以读 global memory，但那是远处 DRAM，必须付出等待数据到来的 latency hit；此外还有 constant memory，讲者说自己在各种编程里"没怎么见人用过"。host memory（主机内存，即机器上 CPU 的内存）则支持双向传输，当你需要 offload 到超出 GPU 容量的数据时用得上。幻灯片 "Memory model of a GPU" 把这些权限列得比口述更整齐，图中 Grid 下有 `Block (0, 0)`、`Block (0, 1)`，每块内含 `Shared Memory`、`Registers`、`Thread (0, 0)`、`Thread (1, 0)`，下方是 `Global Memory` 与 `Constant Memory`，左侧 `Host` 通过双向箭头连到这两层：

```text
Device code can:
- R/W per-thread registers
- R/W per-thread local memory
- R/W per-block shared memory
- R/W per-grid global memory
- Read only per-grid constant memory
Host code can:
- Transfer data to/from per grid global and constant memories
```

![Memory model of a GPU 幻灯片：Device code 与 Host code 的读写权限清单，以及 Grid、Block、Thread、Global Memory、Constant Memory 的层级框图](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0014_00-16-35.jpg)

页面底部两行字把本章与全讲的主线拴在一起：`Each thread can access its own register, and shared memory within the block.` 以及加粗的 `Information that goes across blocks need to be read/written to global memory (slow)`。也就是说，跨 block 的信息交换必须绕道慢速全局内存，可写成 $\text{跨 block 通信} \Rightarrow \text{经 global memory (slow)}$。讲者在 [17:00] 用一句话收束：整个内存模型里最关键的一点是"一旦走出 shared memory，事情就会变慢"，因此把 block 组织好、减少 global memory reads，将是贯穿本讲的 "name of the game"（制胜关键）。

本章的易错点相当集中。第一，SIMT 的"相同指令"是 warp 内 32 个线程，不是 block 内全部线程——这正是学生提问要澄清的；把它误当成 grid 级约束，会错判后面控制发散的代价。第二，`Each block runs on a SM` 说的是绑定关系，别反过来以为一个 SM 永远只跑一个 block，也别以为 block 数可以随便超过 SM 数而不付代价。第三，shared memory 只在 block 内共享，跨 block 必须走 global memory，幻灯片特意把 slow 加粗；constant memory 是 `Read only`，不要当成可写暂存区。第四，32 是讲者针对当前 GPU 给出的 warp 大小，本段没有说其它架构或 TPU 也用 32，不宜外推。第五，讲者说 constant memory "没怎么见人用过"，属于经验之谈而非硬性限制。与前后内容的关系上：本章把上一章的 SM / shared memory 落成了可编程对象——正因为 block 绑在单个 SM 上、warp 是 32 线程锁步，后面的六个 tricks 才有抓手（tiling 复用 shared memory、控制发散来自 warp 锁步、减少全局读取来自跨 block 的 slow 路径）；同时"no warps"这一点，将在下一章成为 TPU 与 GPU 最直观的对照项。

## TPU 与 GPU 的趋同演化：结构相似、尺度与网络不同 [17:18 - 19:39]

讲者用 "I'm going to talk for two slides about TPUs" 说明这只是个 side thread（支线）：本讲主体仍是 GPU，因为他判断多数人出去写代码、建模型时用的都是 GPU；但谈 TPU 既因为它 "cool and becoming more popular"，更因为它是 GPU 的 alternative evolution（另一种演化路径），对照着看能清楚分辨"什么相同、什么不同"。他给出的核心判断是 convergent evolution（趋同演化）：TPU 与 GPU 在高层次上非常相似，只要你想造一台 energy-efficient（节能的）机器学习加速器，最后往往会落到同一个地方。心智上要记住的差别是 TPU "in some sense simpler"——更针对 ML workload 优化，control units 更轻量（lighter weight control units），而 matrix multiply units 大得多（much bigger matrix multiply units）。除此之外它和 GPU 有着看起来一样的骨架：一块专门乘矩阵的电路、能做并行向量操作的部件、某种 control system，以及很相似的 memory structure——慢的那层叫 high bandwidth memory（HBM，高带宽内存），更快的本地内存讲者口述作 "S-man or shared memory"（转写含糊，幻灯片图上标的是 `S mem` 与 `Vector Memory (Vmem)`）。所以差别主要在这些部件的 size（尺寸）与操作的 flexibility（灵活性），而不是"有没有这些部件"。

![Side thread – What about TPUs? 幻灯片：TPU TensorCore 的抽象布局，含 Scalar Unit、VPU、Vmem、MXU 与 HBM，右侧标注 GPU 有更多 SM 而 TPU 有更少 TC](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0015_00-17-49.jpg)

这张幻灯片标题是 `Side thread – What about TPUs?`，第一句就是 `GPUs, TPUs, and many other accelerators are at a high level, similar`；中间那幅 `Abstract layout of a TPU TensorCore.` 把结构拆成 `Scalar Unit`、`S mem`、`Vector Unit (VPU)`、`Vector Memory (Vmem)`、`Matrix Multiply Unit (MXU)` 与右侧一整条 `High Bandwidth Memory (HBM)`。左侧三条注释把数据流讲得很完整：`The Scalar Unit sort of acts like a CPU 'dispatching' instructions to the VPU and MXU`（标量单元像 CPU 一样给 VPU 和 MXU 派发指令）、`The VPU performs elementwise operations (e.g. activations), loads data into the MXU`（VPU 做逐元素操作如激活，并把数据装进 MXU）、`The MXU performs matrix multiplications - and is therefore our driver of chip FLOP/s`（MXU 做矩阵乘，是芯片 FLOP/s 的主要来源）。右侧注释则说明 `HBM stores the weights, activations, optimiser states, new batch data etc`，而 `HBM bandwidth: determines how fast data goes to and from the computational elements`——这恰好把前面几章反复强调的"内存带宽决定数据进出计算单元的速度"换了个名字又讲了一遍。旁边一句对照尤其关键：`A GPU has more SMs / TPUs has fewer TCs (but similar matmul perf)`，与上一章 GA100 的 128 个 SM 放在一起看，就是"GPU 用很多较小的 SM、TPU 用更少但更大的 TC，矩阵乘吞吐却相近"。页面底部把共性压成一句 `Core structure – lightweight control, fast (big) matmul unit, fast memory.`，可形式化为

$$
\text{Core structure} = \text{lightweight control} + \text{fast (big) matmul unit} + \text{fast memory}
$$

再往下是差异清单，照录如下：

```text
Core structure - lightweight control, fast (big) matmul unit, fast memory.
Differences - how the accelerators are networked (in the parallelism lecture)
  - no warps (just blocks - tradeoffs in matmul vs non-matmul)
A GPU has more SMs
TPUs has fewer TCs (but similar matmul perf)
```

`no warps (just blocks – tradeoffs in matmul vs non-matmul)` 直接接头上一章的 32 线程 warp：TPU 没有 warp 这一层，只有 block，于是在 matmul 与非 matmul 操作之间有不同的取舍。讲者在 [19:02] 明确说，今天不展开、但真正最大的差别是 networking（互连网络）：如果问 TPU 与 GPU 最大的区别是什么，答案其实来自网络而非单颗芯片，因为"这些芯片活着就是为了乘矩阵"（they live to just multiply matrices），互连方式才在 parallelism lecture（并行化那一讲）里讲。他随后引到 Jax book（称其为 "a really lovely resource"），说书里有两张表、其中一节讲怎么把 GPU 概念映射到 TPU 概念，并强调"今天讲的左边这套 GPU 的东西，都能直接映射到 TPU"——这正是本讲后续所有 GPU 优化思想可以迁移到 TPU 的依据。本段在 [19:37] 讲到 `But the important thing, you know, at this point in the lecture to notice is` 时戛然而止，句子没说完；按"转写含糊处宁可略过"的原则，这里不替讲者补结论。

本章的易错点主要有四处。第一，别以为 TPU 与 GPU 是两种根本不同的机器：讲者的主结论恰恰是 high level 上 similar、属于趋同演化，差别在尺寸与灵活性；把 `A GPU has more SMs / TPUs has fewer TCs (but similar matmul perf)` 读成"TPU 算力更弱"就错了，`but similar matmul perf` 明说矩阵乘吞吐相近。第二，别把最大差异误放在芯片内部——讲者明说最大的差异来自 networking，而且本段明确不讲，任何关于互连拓扑的细节都属于编造。第三，术语要分清：幻灯片把 TPU 的这套结构叫 `a TPU TensorCore`，其中真正做矩阵乘的是 `MXU`，做逐元素操作的是 `VPU`，本地快速内存标作 `S mem` / `Vmem`；讲者口述的 "S-man" 转写含糊，不宜当成确定术语。第四，`no warps` 是幻灯片上的一条 bullet，讲者本段口述并未展开其后果，不宜替他推导调度细节。与前后内容的关系上：本章把前五章建立的 SM、warp、内存层级全部拿来做了对照——`lightweight control` 呼应第三章 CPU/GPU 的控制单元差异，`fast (big) matmul unit` 呼应第二章 Tensor Cores 与专用矩阵乘指令，`fast memory` 与 `HBM bandwidth` 呼应第四章的内存层级，`no warps` 呼应第五章的 32 线程锁步；同时它把话题交棒给后续的并行化讲座与六个 tricks，说明本讲关于 GPU 的优化思想大多可以原样迁移到 TPU。

## TPU 与 GPU 的概念映射、命名陷阱与数量对比 [19:39 - 22:39]

本段开头承接上一段已经建立好的 GPU 硬件心智模型，讲者把 TPU 拉进来做对照，出发点是一句很强的断言：GPU 里的每一个概念，在 TPU 里都有一个对应的概念，而且这个映射 "actually pretty precise"（相当精确）。最典型的一对是 GPU 的 tensor core（张量核心，即矩阵乘单元，matrix multiply unit）对应 TPU 的 MXU（matrix multiply unit，矩阵乘单元）；如果再往 "needy-greedy details" 里钻，两者底层用的是同一类电路——systolic array（脉动阵列），它的工作方式是 "streams data in and out to do a matrix multiply"（把数据流水式地送入送出以完成矩阵乘），所以架构非常相似。这里要留意讲者的诚实边界：当场有学生追问 "typical systolic array size on a GPU versus TPU"（GPU 与 TPU 上脉动阵列的典型尺寸各是多少），讲者明确回答 "I do not know that off off my head. I'm sorry. I'll have to look that up."，也就是没有给出具体数字。因此本段不能编造脉动阵列的尺寸，只能记下"两者都基于 systolic array"这一定性结论，这是后面所有推论的前提而非可量化的参数。

紧接着讲者专门插入一段 "one very, very confusing thing"——命名陷阱：TPU 把它的 streaming multiprocessor（流式多处理器）叫做 tensor core，而 GPU 把它的 matrix multiply unit（矩阵乘单元）叫做 tensor core，两者 "named exactly the same thing"（名字完全一样）。消歧只能靠上下文：有人说 "tensor core for a TPU"，指的是一个 processor（处理器／计算单元）；有人说 "tensor core for a GPU"，指的是矩阵乘单元。这个坑在读论文、看硬件白皮书时极易踩中，因为同一个词在两种语境里分别对应"整颗粒"和"颗粒里的一个部件"。幻灯片右侧的对照表把这种一一对应列得很清楚：Streaming Multiprocessor (SM) ↔ Tensor Core，标注为 "Core 'cell' that contains other units"（包含其它单元的核心 cell）；Warp Scheduler ↔ VPU，标为 "SIMD vector arithmetic unit"；CUDA Core ↔ VPU ALU，标为 "SIMD ALU"；SMEM (L1 Cache) ↔ VMEM，标为 "Fast on-chip cache memory"；Tensor Core ↔ MXU，标为 "Matrix multiplication unit"；HBM (aka GMEM) ↔ HBM，标为 "High bandwidth high capacity memory"。幻灯片右下角给出的出处是 `https://jax-ml.github.io/scaling-book/gpus/`。

幻灯片左侧还画了一张 TPU TensorCore 的抽象布局（Abstract layout of a TPU TensorCore），并逐块标注职责：Scalar Unit "sort of acts like a CPU (dispatching instructions to the VPU and MXU)"（有点像 CPU，负责把指令派发给 VPU 和 MXU）；Vector Unit (VPU) "performs elementwise operations (e.g. activations), loads data into the MXU"（做逐元素运算如激活函数，并把数据装进 MXU）；Matrix Multiply Unit (MXU) "performs matrix multiplications – and is therefore our driver of chip FLOPs"（做矩阵乘，因而是芯片 FLOPs 的主要来源）；Vector Memory (VMEM) 是片上快速存储；HBM "stores the weights, activations, optimizer states, new input data etc."（存权重、激活、优化器状态、新输入数据等），且 "HBM bandwidth determines how fast data goes to and from the computational elements"（HBM 带宽决定数据进出计算单元的快慢）。整页底部的总结句是 "Core structure – lightweight control, fast (big) matmul unit, fast memory."（轻量控制、快速且大的矩阵乘单元、快速内存）。把这张图与上一段的 GPU 内存层级对照，就能看出"轻量控制 + 大 matmul 单元 + 快速片上内存"是两类加速器共有的骨架，这也解释了为什么讲者说本讲的基本概念可以整体迁移。

数字对比是本页最有信息量的部分。下方表格列出 H100 与 TPU v5p 的具体数量：

$$
\begin{array}{l c c}
\text{GPU} & \text{TPU} & \text{H100 \# / TPU v5p \#}\\
\hline
\text{SM (streaming multiprocessor)} & \text{Tensor Core} & 132\ /\ 2\\
\text{Warp Scheduler} & \text{VPU slots} & 528\ /\ 8\\
\text{SMEM (L1 cache)} & \text{VMEM} & 32\text{MB}\ /\ 128\text{MB}\\
\text{Registers} & \text{Vector Registers (VRegs)} & 32\text{MB}\ /\ 256\text{kB}\\
\text{Tensor Core} & \text{MXU} & 528\ /\ 8
\end{array}
$$

转写里讲者口头给的是同一组数：一块典型 GPU 如 H100 有 "something like 132 streaming multiprocessors"，而 TPU "only have two units"，只有两个独立并行单元；只看矩阵乘单元，GPU 有 528 个，TPU 只有 8 个。他特别提醒这些方块 "roughly scaled to be comparable looking objects"（大致按可比较的观感缩放），所以不要把面积比例当成真实算力比例。真正的差别是设计哲学：TPU 依赖 "much bigger, smaller numbers of matrix multiply units"（更少但更大的矩阵乘单元），GPU 依赖 "smaller, but much more numerous"（更小但数量多得多）的单元。数量多带来灵活性（"you can program them to do this and that"），而 TPU "locked in to big, big matrix multiplies"（被锁死在大矩阵乘上）。与之配套的一个具体经验是讲者自己论文里做 batch size sweep（批大小扫描），曲线 "goes all the way down to 64 and it stops at 64"，原因是 "the tensor core refuses to accept anything smaller than a 64-dimensional input"——把这条硬约束写下来就是 $\dim(\text{input}) \ge 64$，小于 64 维的输入根本喂不进 TPU 的矩阵乘单元。这是"大单元"取舍最直接的副作用，也是把训练任务移植到 TPU 时最容易忽略的易错点。

收尾处讲者把两者归结为 "convergent evolution"（趋同演化）：GPU 与 TPU 都依赖 fast memory / slow memory 的内存层级，都有矩阵乘单元，所以 "the core concepts that I'm teaching you today is going to transfer over between GPU and TPU"，本讲的基本概念可以互相迁移。理由很朴素——"there are only so many ways to sort of cost-effectively allocate memory, and there's only so many ways to multiply matrix very fast"（高性价比地组织内存的方式有限，把矩阵乘做快的方式也有限），于是设计决策虽有不同，终点却落在同一个地方。这一段与后面的关系很直接：正因为硬件骨架趋同，后面讨论的 GPU 优化技巧（尊重内存层级、提高算术强度、减少全局内存搬运）在 TPU 上同样成立；而"数量少但更大"的差异，则为下一章讲 GPU 靠增加 SM 扩展吞吐、以及推理阶段按计算／内存特性拆分芯片埋下伏笔。局限也要说清楚：这里的所有数字都是某一代产品（H100 与 TPU v5p）的快照，不能外推到其它代际，讲者也拒绝对脉动阵列尺寸做定量承诺。

![TPU 与 GPU 概念对照表及 H100 与 TPU v5p 单元数量对比](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0017_00-20-16.jpg)

## GPU 模型为何成功，以及用图形硬件「黑」出矩阵乘 [22:39 - 24:40]

讲者用 "enormously successful" 总结 GPU 作为硬件模型为什么成功，并给出三条理由，幻灯片标题为 "Strengths of the GPU model"，bullet 依次是 "Easily scales up hard workloads (by adding more SMs)"、"Easy (?) to program due to the SIMT model"、"Threads are 'lightweight' and can be stopped and started"。第一条讲可扩展性：想提高 throughput（吞吐），只要不断加 SM 即可——"If you want more throughput, you just keep adding SMs"，前提是 "as long as your memory bandwidth can handle the SMs"（内存带宽喂得饱这些 SM）。这条因果链很重要：加算力不是无条件的，它把内存带宽设成了扩展的天花板，正好呼应上一段 "HBM 带宽决定数据进出计算单元快慢" 的标注，也预告了后面"计算增长快于内存"的失衡。第二条讲编程模型："the programming is something we don't think about, but it's deceptively easy"（编程这件事我们不太去想，但其实出奇地容易）。哪怕去写 PTX 这种 "lower-level machine code-looking stuff"，也 "actually doable because it's all SIMD"——因为本质上都是 SIMD（单指令多数据），你不是在为每一个线程单独写程序，而是在说：

```text
here are my instructions, here are my inputs,
now run them on all of my inputs
```

讲者把这种心智模型类比到函数式编程里的 apply / map："you're saying, 'Okay, here are my instructions, here are my inputs, now run them on all of my inputs, right?'"，熟悉 functional programming 的人会觉得眼熟。第三条是线程极轻量：GPU 里的 threads "always very lightweight. They can be stopped, started at any time"，调度器可以随时判断 "I have another warp that's better to run, so I'm going to stop you and start this one"，并且 "very quick to sort of swap in these jobs"（换入换出非常快）。好处是当某些 job 停顿或吞吐下降时，调度器立刻用别的 warp 把空隙填上，"allows for much higher throughput"。幻灯片底部的对照图把这点画得很直观："GPU – High Throughput Processor" 里 $T_1,T_2,T_3,T_4$ 四条线程的时间条彼此交错，绿色 "Processing"、白色 "Waiting for data"、橙色 "Ready to be processed" 交替出现；而 "CPU core – Low Latency Processor" 里 $T_1,T_2,T_3,T_4$ 是顺序排开的一段段。也就是说 GPU 用"让大量线程轮流填满彼此的等待空隙"换总吞吐，CPU 用"让单个线程尽快跑完"换单任务延迟。图中间的 SIMT 示意还标了 "Instruction Decoder and Warp Scheduler" 把同一条指令广播给多个绿色 thread 框，每个 thread 下面挂自己的 registers，并注明 "1 instruction – multiple threads"。

第三条理由之后讲者把时间往回拨，讲 Tensor Cores 出现之前的 "hacker work"：即使在早期，人们就意识到 "commodity graphics hardware, this massive parallelism" 对科学计算非常有效。屏幕上放的是 "one of the original papers on using graphics hardware for fast matrix multiplies"——《Fast Matrix Multiplies using Graphics Hardware》，作者 E. Scott Larsen 与 David McAllister，均来自 Department of Computer Science, University of North Carolina at Chapel Hill，邮箱分别写着 `larsen@cs.unc.edu` 与 `davemc@cs.unc.edu`；图注是 "Early days of NVIDIA GPUs – programmable shaders. Researchers hacked this to do matmuls"。当时的思路是：显卡里有现成的 shaders（可编程着色器），但可以 "program specific shaders that will actually give us sort of matrix multiplies"（专门写着色器来变相完成矩阵乘），而且 "different rendering settings actually give us faster matrix multiplies"——连渲染设置都会改变 matmul 的速度，讲者称之为 "really, really cool sort of hacker work"。

论文截图右侧 "Implementation" 栏里有一组非常具体的经验参数，转写里没有逐条念，属于图上独有的信息，值得完整转述：Refresh Rate——"setting the refresh rate on the monitor as low as possible made marginal improvements (about 10%)"，把显示器刷新率压到最低只带来约 $10\%$ 的边际提升；RGBA——"4 numbers can be packed into a single pixel, by setting the red, green, blue, and alpha channels to different values"，即把 4 个数塞进一个像素的 RGBA 四通道；Texture Format——把纹理创建与回读格式从 `RGBA` 改成 `ARGB_EXT`（in OpenGL）"gave about 40% improvement on our hardware"，约 $40\%$ 提升，原因是 "the hardware driver avoids reformatting the data from the application format to the card format"（驱动省去了应用格式到显卡格式的重新格式化），并且 "a number of options here, with near equal performance for each option except the one used natively on the specific hardware"，原生格式应当给出显著提升；Full Screen——"Running full screen instead of in a window provides improved performance"；最后 "Various other optimizations yielded minor (<1%) improvements"，其余优化都不到 $1\%$。把这些落成一个可核对的式子：

$$
\text{RGBA}: 4\ \text{values/pixel},\quad
\text{Refresh Rate}\approx 10\%,\quad
\text{ARGB\_EXT}\approx 40\%,\quad
\text{others}<1\%
$$

这些数字把"渲染设置影响 matmul 速度"从一句感慨落到了具体量级上。

与前后内容的关系与易错点：这一页是"手动 hack"时代的收尾，讲者紧接着就说 "nowadays, you do not have to do this. You do not have to manually program your GPUs to do matrix multiplies, because as of the V100, Nvidia decided, you know, here is a piece of hardware that will do your matrix multiplies for you"，也就是说从 V100 起专用硬件接管了 matmul，手工写着色器做矩阵乘变成历史。因此不要把 $10\%$ / $40\%$ / $<1\%$ 当成今天仍可照搬的优化清单——它们是特定驱动、特定 OpenGL 实现上的经验观察；也不要把"4 个数打包进一个像素"误当成现代 Tensor Core 的低精度打包，两者只是形似。真正被继承下来的是那条主线：GPU 的并行度可以很容易地"喂"给矩阵乘这类规则运算，而专用电路一旦出现，matmul 就会被抬成"特权操作"，这正是下一章的内容。

![GPU 模型的三大优势与高吞吐低延迟线程调度对照图](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0019_00-22-44.jpg)

![Larsen 与 McAllister 用图形硬件做快速矩阵乘的论文截图及实现观察](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0020_00-23-57.jpg)

## V100 之后 matmul 成为特权操作，以及各部件扩展速率的分化 [24:40 - 27:30]

讲者把 V100 当作分水岭："as of the V100, Nvidia decided, you know, here is a piece of hardware that will do your matrix multiplies for you"——从 V100 开始，芯片上有了专门做矩阵乘的电路，于是 "once that hardware sort of existed, matmuls became the sort of one privileged operation in machine learning"（matmul 成了机器学习里唯一的特权操作）。所谓"特权"，指的是吞吐量上出现了一个 "gigantic gap"：一边是 "parallelizable but non matrix multiply operations"（可并行但非矩阵乘的浮点运算），另一边是矩阵乘能拿到的算力，二者不再同一量级。用幻灯片标题说就是 "New matmul hardware means matmuls are fast and special"，图名 "Matmul vs. non-matmul FLOPS across GPUs"，纵轴是 `TFLOP/S`（对数刻度 $10^1,10^2,10^3$），横轴依次是 `K80, M80, P100, V100, A100, H100`，蓝色线是 `non-matmul`，橙色线是 `matmul`。两条线在 P100 之前几乎重合在 $10^1$ 附近，从 V100 起橙色线陡升，到 H100 已接近 $10^3$，蓝线只到几十；图下的结论写得很直白：

$$
\text{matmul throughput} > 10\times\ \text{non-matmul throughput}
$$

即 "Tensor cores (introduced in V, T series) are specialized matrix multiplication circuits. Matmuls are >10x faster than other floating point ops!"（V/T 系列引入的 Tensor Core 是专用矩阵乘电路，matmul 比其它浮点运算快 10 倍以上）。转写里的措辞是 "More than 10 times faster than any other floating point operation that you can do in a GPU"。

由此推出一个架构层面的判断：讲者说 "any near future machine learning architecture that we see that scales with compute is going to have a matrix multiply in it"——任何想随算力扩展的 ML 架构里几乎都必须含一个矩阵乘，因为 "this is the one way that you can really effectively get a lot of compute throughput"。这就把硬件特性反向约束到了模型设计上：不是先设计算子再想办法加速，而是先把 matmul 这块"最便宜的算力"用足。问答环节还补了一刀确认：有学生问 "before the V100s, there was no matrix multiply unit like"，讲者答 "Yeah, yeah, there was no tensor core that like specialized to a matmul. Yeah, the tensor core is a V100 addition"，并补充在专用电路之前 "you have so many SMs and so many like ALUs that you can still do quite good throughput"，但 "a specialized circuit for matrix multiply really changes the game"。也就是说通用 ALU 仍能凑出不错的吞吐，只是量级被专用电路彻底拉开。

本段最后一张硬件幻灯片（"the last slide of this section"）转向时间维度：不同组件 "are scaling at different rates over time"。屏幕上标题是 "Compute scaling is faster than memory scaling"，图名 "Scaling of Peak Hardware FLOPS, and Memory/Interconnect Bandwidth"，图里三条线分别标着灰色的整体算力、绿色的内存带宽、蓝色的互连带宽，并直接给出 20 年的倍率与每两年倍率：

$$
\begin{array}{l c c}
\text{HW FLOPS} & 60000\times / 20\ \text{yrs} & (3.0\times/2\text{yrs})\\
\text{DRAM BW} & 100\times / 20\ \text{yrs} & (1.6\times/2\text{yrs})\\
\text{Interconnect BW} & 30\times / 20\ \text{yrs} & (1.4\times/2\text{yrs})
\end{array}
$$

讲者口头的对应关系是：灰线 "overall compute throughput is getting faster and faster"、"pretty fast"；绿色是 "memory bandwidth. This is how fast we can transfer data. That's growing comparatively slowly"；蓝色是 "parallelism. This is how fast we can connect different GPU devices. That's also going very slowly"。把两组倍率相除就能看出失衡的程度：20 年里算力涨了 $60000/100 = 600$ 倍于 DRAM 带宽、$60000/30 = 2000$ 倍于互连带宽；换算成每两年，算力约 $3.0\times$ 而 DRAM 只有 $1.6\times$、互连只有 $1.4\times$。图下的结论句是 "FLOPs scale faster than memory -> hard to keep our compute units fed with data!"（算力比内存涨得快，越来越难把计算单元喂饱）。

讲者随后给出历史解释与全段小结：早期做 GPU 编程 "probably you weren't thinking much about memory because your compute wasn't that much faster than your memory transfer"，但 "as we go more and more to the right of this plot, we're going to see a bigger and bigger gap between compute and memory"，于是 "a lot of the optimization I'm talking about today are memory optimizations"，越往右越会撞上内存与通信瓶颈，"it's going to take more and more work to fully utilize the hardware that we have"。他给出的心智模型是三句话：GPU 是 "massively, massively parallel"，同时执行大量指令；"compute scales really fast, really, really fast, and even faster than memory, which means memory is really important"；因此 "everything we do has to respect the memory hierarchy"，"as much stuff that we do goes into shared memory and not global memory"。他还自谦说 20 分钟不可能讲透 GPU 硬件，"hopefully you have a flavor"，有了这个 flavor 就足以内化下一部分。

与前后内容的关系和易错点：这一章把上一段"TPU/GPU 趋同演化"里的内存层级，正式升级成全讲的因果主线——正因为算力与带宽的增速差是 $3.0\times$ 对 $1.6\times$ / $1.4\times$ 每两年，后面所有技巧（控制发散、低精度、算子融合、重计算、合并访存、tiling）才都围绕"减少内存搬运"展开。易错点有两个：其一，">10x" 是跨代对比里 matmul 与"其它浮点运算"的比值，不是同一张卡上任意两个 kernel 的比值，也不是 V100 之前就成立的数字；其二，$60000\times$ 这类倍率是 20 年累计的归一化趋势，不能拿来估算某一代 H100 的绝对 TFLOPS，讲者也没有在转写里给出绝对数值。另外要注意蓝线被称作 "parallelism / connect different GPU devices"，即它衡量的是设备间互连而非单卡内部并行度，别把两者混为一谈。

![跨 GPU 型号的矩阵乘与非矩阵乘 TFLOPS 对比，V100 之后 matmul 快 10 倍以上](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0021_00-25-11.jpg)

![算力、DRAM 带宽与互连带宽 20 年扩展速率对比](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0022_00-26-25.jpg)

## 问答延伸：L1 与共享内存、SRAM 以及推理端的芯片拆分 [27:30 - 30:10]

讲完硬件概览后讲者专门留出提问时间（"I can pause for a moment here in case people have hardware questions"），这一小段虽然没有新幻灯片，却把前面几处容易含混的概念钉死了。第一问是 "shared memory 与 L1 cache 的区别"，回答极简："the cache just operates on its own. Like, you don't get to control the L1 cache."——L1 cache 由硬件自动管理，程序员无法直接控制；而共享内存是可编程的、由代码显式读写。这与上一章 "as much stuff that we do goes into shared memory and not global memory" 形成呼应：正因为它可控，shared memory 才是优化者真正能下手的地方，L1 只能被动配合。第二问是 "L2 比 L1 慢，只是因为物理距离，还是有别的原因"，讲者答 "physical distance is a big part of it"，并给出一个关键的材料事实："they are, if I believe if I remember right, they're both SRAM"（两者都是 SRAM），"that's one of the reasons why global memory is much slower"；更细的差别来自 "the physical distance and the interconnects that are used"。这里要保留讲者自己的不确定语气（"if I remember right"），不要把"都是 SRAM"写成绝对定论。

紧接着他补了一个跨硬件的对照："this is not inherent, by the way"——L2 慢并不是物理定律；"the equivalent to L2 cache in TPUs is much faster because they make different design tradeoffs on silicon to enable that. That's one of their selling points."（TPU 上相当于 L2 的那级缓存快得多，因为它们在硅片上做了不同的设计取舍，这是卖点之一）。把这句话放回第 1 章的映射表就很清楚：VMEM 对 TPU v5p 是 $128\text{MB}$，而 H100 的 SMEM (L1 cache) 是 $32\text{MB}$，$256\text{kB}$ 的 Vector Registers 又与 $32\text{MB}$ 的 Registers 处在不同量级——同样是"快速片上内存"，两类芯片给的预算差别很大。易错点也就在这里：不能把 GPU 上"L2 一定比 L1 慢很多"的经验直接搬到 TPU，这是设计取舍而非必然。画面此时切到讲台，白板上还残留着上一讲关于带 $\ell_1$ 型正则项的优化板书：

$$
u^* = \min_u H + \sum_{i} c_i |u_i| + P^\top b\,u
     = \begin{cases}
       (c_i + P^\top b_i)\,u_i, & u_i \ge 0,\\
       (-c_i + P^\top b_i)\,u_i, & u_i < 0,
       \end{cases}
$$

右侧还按 $P^\top b_i > c_i$、$0 < P^\top b_i < c_i$、$-c_i < P^\top b_i < 0$、$P^\top b_i < -c_i$ 分情况写了解的形式。这段板书属于前面课程的推导，与本段 GPU 硬件讨论无关，转写里也完全没有念到，这里只作为画面事实记录，不据此展开任何结论。

第三个问题把话题从训练推到推理："If this is compute-memory and the keeps growing, do you think that affects the way future chips will be laid out? Like"——即算力与内存差距持续拉大，会不会改变未来芯片的版图划分／预算分配（"future chip kind of budget splits"）。讲者先承认 "one of the things that I really didn't want to talk about, but now you forced me to"，然后说 "the inference landscape for hardware is actually even more crazy"。他给出的具体做法是 prefill / decode disaggregation（预填充与解码解耦）："the prefill part, which is very heavy matrix multiplies, is one chip, and then the decode part for your inference, which is more memory bandwidth limited, is another chip"。把它写成结构示意就是：

```text
prefill  -> chip A   : very heavy matrix multiplies (compute-bound)
decode   -> chip B   : more memory bandwidth limited (memory-bound)
```

也就是说，把一次推理里"算力密集"与"带宽密集"的两段拆到不同芯片上，各自按自己的瓶颈配预算。他还提到更激进的按层拆分："some of the models... even has a thing where they like try to disaggregate different layers. So an attention will go to one chip optimized for one thing, or one accelerator, and then MLPs will go to a different accelerator"——attention 与 MLP 分别放到针对不同特性优化过的加速器上。转写里模型名含糊（"I think this was uh step one three, uh which is a Chinese open source model"），讲者自己也不确定，因此这里不强行指认具体模型，只保留"某中国开源模型、按层拆分"这层信息，这符合"转写含糊处宁可略过，不要猜"的要求。

讲者给出的动因与上一章的三条扩展曲线完全咬合："if this gap grows, any sort of clever tricks to utilize your very precious fast memory becomes very valuable, and you start to see this very heavily in inference, which is even more memory bound than training"——差距越大，能省下快速内存的技巧越值钱，而推理比训练更 memory bound，所以这类拆分最先在推理侧爆发。最后一个问题 "How many warps are in a block?" 讲者答 "I don't think I know that off the top of my head. I think it's also hardware dependent."（记不清，而且依赖具体硬件）。这与本章开头"systolic array 尺寸不知道"是同一种诚实边界：凡转写里没给、讲者明确说不确定的数字，就不该在笔记里补一个"典型值"。

与前后内容的关系：这一段是硬件部分到性能部分的过渡。它把"算力 $3.0\times$ / DRAM $1.6\times$ / 互连 $1.4\times$ 每两年"的失衡，从"优化时要尊重内存层级"进一步推到"硬件本身要按瓶颈拆芯片"，然后讲者在 [30:07] 用 "Okay, good. Um Okay, so the second part of this lecture" 正式切到 "all of the basic tricks to make workloads fast on a GPU"。易错点：prefill/decode disaggregation 是推理阶段的工程做法，不是训练技巧，也不是本章后面要讲的六个 GPU 优化手段之一；把它和后面的算子融合、重计算混为一类就错了。另外 L1／共享内存的区别是"是否可编程控制"，不是"谁离 SM 更近"，讲者并没有在这一问里给出任何延迟数字。

## Part 2 的目标：解释吞吐曲线并用 roofline 模型定位瓶颈 [30:10 - 32:57]

讲者宣布进入本讲第二部分："now you all have a general sense of what a GPU is and how it works. Now we are going to talk about all of the basic tricks to make workloads fast on a GPU"，这些技巧是 "basic components that you can kind of put together to make things fast"。为了给整段一个抓手，他设了一个目标："one of my goals by the end of this section is to explain this plot to you"。屏幕上（kf_0025）标题是 "Part 2: Making ML workloads fast on a GPU"，图名 "FLOPs achieved for square matmuls"，纵轴 `TF/s` 刻度 $0,50,100,150,200,250$，横轴刻度 $0,512,1024,1536,2048,2560,3072,3584,4096$，右下角还标了个 $128$。图下写着 "Performance on a GPU can be complex, even for something as simple as a square matmul"。讲者解释坐标含义："the size of the matrix, a square matrix that I'm multiplying on multiplying two square matrices"，Y 轴是 "how many items I'm processing per second"，也就是 throughput（吞吐）。总体规律是 "as I'm going to the right, in general, throughput is going up because I have bigger matrices, I have more work to do"，工作越多越能把硬件喂满；但 "there's all sorts of strange patterns here"，曲线上有一串奇怪的下凹和分层，本节要回答的正是 "why are there strange patterns here? Why and how can we be at the very top of that curve rather than the bottom"。讲者在投影上手写了三处批注：紫色 "Compute Intensity"（计算强度）指向左下的对角上升段，绿色 "Tiling" 与箭头指向上方几条曲线，绿色 "Wave Quantization"（波量化）指向周期性下凹——这三个词正好预告了后面要靠 tiling 与算术强度把曲线抬到顶、并解释周期性掉速。这批注转写里没有逐字念，属于图上独有的信息。

接着讲者给出解释这幅图的理论工具："Percy already mentioned this. This is the roofline model"（屋顶线模型），并说前面那张图 "looks a little bit like this classic roofline model... there's a diagonal part and then a flat part"。屏幕上（kf_0026）标题 "What makes ML workloads fast?"，副标题 "The roofline model"，双对数坐标：纵轴 `Throughput (Gflops)` 刻度 $1,10,100,1000$，横轴 `Operational Intensity (flops/byte)` 刻度 $0.01,0.1,1,10,100,1000$。图里画了多条斜线与平台线，分别标着 `GPU registers`、`GPU shared memory`、`GPU main memory`、`CPU main memory`，以及两条水平上限 `GPU ALU throughput` 与 `CPU ALU throughput`；右上角的图例给出两个数据点：`Dense matrix multiply`（蓝色菱形）与 `Sparse matrix multiply`（蓝色圆点）。页面底部一句把整个第二部分的动机挑明："Key to this section: how do we avoid being memory bound?"（本节的关键：怎样才不被内存带宽绑住？）

roofline 的直觉可以分段说清。讲者说 "up until a certain point, we will be memory limited, right? We're memory limited, so no matter how much compute you give me, my throughput is not going to go up"——在对角段，瓶颈是搬数据，算力再大也没用；"at some point you have enough work per unit of memory movement that I can keep my compute units completely saturated"，过了那个拐点就进入 "compute bound region which is flat"，"no matter how much extra work you give me, that doesn't help because I've fully saturated my matrix multiply units"。把这两段写成一个式子（讲者用语言描述、未板书公式）就是：

$$
\text{Throughput} = \min\big(\text{Peak FLOPs},\ \ \text{Memory BW} \times \underbrace{\text{Operational Intensity}}_{\text{FLOPs}/\text{Bytes}}\big)
$$

其中横轴那个量叫 operational intensity（运算强度），也就是每读一个字节能做多少 FLOP；讲者的说法是 "we want to increase the operational intensity. The amount of computation we do per memory read needs to be high enough"，本讲后面沿用旧笔记里的写法即 arithmetic intensity（算术强度）：

$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes}}
$$

于是优化目标被明确成一句话："if you want to make sure that your code runs efficiently on a GPU, really what you need to do is you need to be on this flat part"——必须待在平台段，"as soon as we have this flat part, there's nothing else to do, we're at maximum throughput"；反过来 "we want to prevent ourselves from being in this diagonal part"。讲者还给了 roofline 的一个动态读法："as we sort of have faster memory, we can kind of push this upwards"，内存变快会把斜线整体抬高，但 "memory bound region which is diagonal, and then a compute bound region which is flat" 这个形状不变。随后他报出全段的路线图："there's six different tricks that we will talk about"，并提前定了性质："some of these are not sort of memory tricks, the first one only"，"The first one's a little bit different in flavor, but then the rest of them will have this flavor of trying to minimize memory movement as much as possible"——六个技巧里只有第一个不是内存技巧，其余五个都围绕"尽量少搬内存"。旧笔记把这六项记为控制发散、低精度计算、算子融合、重计算、合并访存（coalescing）、tiling，本段转写里只明确讲到前两项，后几项留到本段之后，但先把清单摆出来有助于理解每个技巧的定位。

与前后内容的关系与易错点：roofline 把上一章"算力增速 $3.0\times$ / DRAM $1.6\times$ 每两年"的宏观失衡，翻译成"某个具体 kernel 落在斜线段还是平台段"的微观判据；下一章的控制发散之所以被特意标成 "not a memory issue"，正是因为它不在这个 min 式子的两项里，而是执行模型带来的空转。易错点有三：其一，斜线段"给更多算力没用"不等于"给更多工作没用"——讲者说往右走（矩阵更大、工作更多）吞吐确实上升，只是上升的原因是把内存带宽吃满，别把 "memory limited" 误读成"增加工作量无效"；其二，$\text{FLOPs}/\text{Bytes}$ 里的 Bytes 指实际发生的内存流量，不是矩阵的理论字节数，所以后面低精度、融合、tiling 都是在改这个分母；其三，屏幕上的 `Dense matrix multiply` 与 `Sparse matrix multiply` 两个点分别落在高、低算术强度处，说明同一个 roofline 框架里不同算子的位置不同，不能拿一个 kernel 的结论套所有 kernel。

![经典 roofline 模型：吞吐对运算强度的双对数图及各内存层级上限](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0026_00-31-36.jpg)

## 技巧一：控制发散，一个不是内存问题的 GPU 特例 [32:57 - 34:50]

讲者说 "the first one is kind of a GPU specific thing and it relates to one of the things I said before, the SIMD model"。屏幕标题直接点明性质："Control divergence (not a memory issue)"，副标题 "GPUs operate in a SIMT model – every thread in a warp is executing the same instruction"（warp 里每个线程都在执行同一条指令）。问题从一句提问切入："in a GPU, every thread is executing the same thing. Now, what happens if you write a if statement on a piece of code that executes on a GPU?"。CPU 上的行为很清楚："you will either pick one branch or the other, you will execute that branch, and you will move on with your life"；GPU 上却 "quite different and slightly counterintuitive"——"every thread has to execute the same instruction. So, actually every thread will kind of execute both the if and the else, but if you're not part of the if branch, you will just mask out that computation, you'll just sit idle"。也就是不在当前分支的线程被 mask（屏蔽）掉、空转等待，这叫 control divergence（控制发散）。

屏幕左下角给出了逐字可抄的示例代码：

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

对照右侧时序图就能看懂执行顺序：绿色方块里画着一排排代表线程的波浪箭头，中间一条深色竖条标着 `diverge`；分叉后先走下分支的 `A;` 与 `B;`（对应下排方块），再走上分支的 `X;` 与 `Y;`（对应上排方块，旁边还标了个 `*`），最后汇合执行 `Z;`，中间用 `...` 表示还有后续，底部一根箭头标 `Time`。讲者的口头描述是 "part of your threads will diverge based on this if statement. Some of them will go to the bottom branch, some of them will go to the top branch. And the bottom branch ones will execute first, top branch ones will wait, and then the top branch ones will execute and the bottom branch ones will wait"，所以 "in order to execute an if statement, you're actually sort of clock walking through both of these"（要把两个分支按时钟各走一遍）。屏幕上那句概括是 "Conditionals are fine, but lead to significant overhead from the execution model"（条件语句本身没问题，但会因执行模型产生显著开销）。

具体代价与量化：讲者给的量级是 "a if statement might actually have to wait two clock cycles or something like this, two operations"——一次分支可能要多等约两个时钟周期、相当于两次操作；而乘法 "is going to all kind of go at once"（所有元素一次性同时算完）。他因此给出一条很实用的编码结论："this is one of the reasons why you really do not want to write if statements"，并举了最典型的例子："if you look at GPU code, like if you have a ReLU, you'll often find you're like multiplying with zeros or multiplying with masks instead of doing like a if statement where you're updating half"。把 ReLU 写成 $x \mapsto \max(0,x)$ 的分支版本会让 warp 内一半线程空转，而写成掩码乘法 $x \cdot \mathbf{1}[x>0]$ 或乘零这类逐元素运算，则所有 lane 同时执行同一条指令。幻灯片中间的 SIMT 示意图也重申了机制："Instruction Decoder and Warp Scheduler" 把一条指令广播给多个绿色 thread 框，每个 thread 下挂 registers，标注 "1 instruction – multiple threads"，右侧单个绿色框被引线标为 `thread`——硬件层面根本没有"每个线程各自取指"这回事。

讲者对这一技巧本身有个两分评价："this is called control divergence. Um and this is nice in some ways because it simplifies programming. It's not nice in that you now have these big compute gaps where your GPU's doing nothing"——好处是编程模型简单（不用为每个线程写独立控制流），坏处是出现大段算力空洞。他也明确把它从主叙事里摘出来："that's a sort of separate story from all this memory stuff, which is kind of the main narrative or the main story of this"，即这是唯一一个与内存搬运无关的技巧；紧接着 [34:53] 就回到 "the first one, which you don't really have to do too much for, but I think is one of the most important ones"，开始讲低精度。

与前后内容的关系与易错点：上一章 roofline 的 $\min(\text{Peak FLOPs}, \text{BW}\times \text{AI})$ 只解释了"算力不够或带宽不够"，控制发散解释的是第三种浪费——算力明明有，却因为分支串行化而空转，所以它确实 "not a memory issue"。易错点：不要以为只有 `if/else` 才会发散，任何让同一 warp 内线程走不同路径的写法（提前 `return`、按 `threadIdx` 分派、循环次数不同）都会触发同样的 mask／等待；也不要把示例代码里的 `threadIdx.x < 4` 当成通用阈值——它只是幻灯片为了画图选的小数字，真正决定粒度的是 warp（上一讲定义为 32 个连续线程），而"一个 block 里有多少 warp"这一问讲者在 [29:54] 明确答 "also hardware dependent"、不肯给数。最后，掩码乘法只是把分支换成算术，并没有减少内存读写，所以它提高的是指令发射效率而非算术强度，别指望靠它把 roofline 的斜线段抬成平台段。

![SIMT 条件分支导致控制发散的示例代码与线程时序图](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0027_00-33-00.jpg)

## 技巧二：低精度计算，从算术强度到 FP8 与 MXFP8 [34:50 - 39:19]

讲者把 low precision computation（低精度计算）称为 "one of the most important ones and hardware-wise I think where Nvidia's investing the most effort"，并回到前面那张 Bill Dally 的 "super exponential growth" 图问 "where does this really come from?"，答案是 "a pretty non-trivial part of this is number representations. You start at FP32, you go to BF16, and then you go to INT8. You just keep decreasing the precision of your numbers"。核心因果极短："if you have half the bits, you have half the bits to move. And so, you have your memory movement, right? And so, you've solved the memory bottleneck for at least a little bit"——位数减半，要搬的字节就减半。屏幕标题是 "Low precision improves arithmetic intensity"，给的例子是 "elementwise ReLU ($x = \max(0,x)$) on a vector of size $n$"，把这件事算成了可核对的账：

$$
\begin{aligned}
\text{(Float 32)}\quad & \text{Memory access: } 1\ \text{read}(x) + 1\ \text{write},\ \text{float32}=4\ \text{bytes per op}\\
& \text{Operations: } 1\ \text{comparison op},\ 1\ \text{FLOP}\\
& \text{Intensity: } 8\ \text{bytes}/\text{FLOP}\\[4pt]
\text{(Float 16)}\quad & \text{Memory access: } 1\ \text{read}(x) + 1\ \text{write},\ \text{float16}=2\ \text{bytes per op}\\
& \text{Operations: } 1\ \text{comparison op},\ 1\ \text{FLOP}\\
& \text{Intensity: } 4\ \text{bytes}/\text{FLOP}
\end{aligned}
$$

转写里的口头版是："let's say I have a value on a vector of size N. If I'm on float 32, I need to do one read... one write after I've computed my value. That's going to be 4 bytes per operation, total of 8 bytes for my memory access. I only have one flop worth of operation. So, that's 8 bytes per flop"，降到半精度后 "halved the number of bytes that I need to move per flop"。

这里有一个讲者当场自我纠正、必须记牢的易错点：他刚说完就改口 "Sorry, this is the inverse of arithmetic intensity. Uh arithmetic intensity should be reverse of this"。也就是说幻灯片上写的 `Intensity: 8 bytes / FLOP` 和 `4 bytes / FLOP` 其实是"每 FLOP 多少字节"（bytes per FLOP），而上一章 roofline 横轴的 $\text{Arithmetic Intensity} = \text{FLOPs}/\text{Bytes}$ 是它的倒数；FP32 对应 $1/8\ \text{FLOP/byte}$，FP16 对应 $1/4\ \text{FLOP/byte}$，即低精度把算术强度提高了 $2$ 倍。别把这两个方向写反，否则会得出"低精度让算术强度减半"的错误结论。这也解释了为什么这个例子选的是 elementwise ReLU：它每个元素只有 $1$ FLOP 却要读写 $8$ 字节，是典型的"内存受限"角落，正好用来示范 roofline 斜线段。

现代 GPU 里的低精度远不止"把数存小一点"。屏幕 "Low precision drives faster matrix multiplies" 写着 "Lots of operations in modern GPUs are accelerated via low / mixed precision operations"，并画了 Tensor cores 的框图：两个 `16-bit input` 进一个绿色乘法圆圈，标注 `Full precision product`；乘积再进绿色加法圆圈，上方 `more products` 多个箭头汇入，下方标 `Sum with FP32 accumulator`（用 FP32 累加器求和），最终输出 `FP32`。讲者的对应描述是："you're actually not going to keep everything in low precision. You're going to downcast before you take your product. And when you sum up your partial summons in your matrix multiply, you're going to do this in full precision. And then you might emit your outputs or your results in FP32"（先降精度再相乘，部分和用全精度累加，输出可以是 FP32）。所以一次"低精度 matmul"其实是混合精度：输入 $16$-bit、乘积全精度、累加 FP32、输出 FP32。右侧还列了两类操作，属于转写没逐条念、图上有而必须补的信息：

- "Operations that can use 16-bit storage (FP16/BF16)"：`Matrix multiplications`、`Most pointwise operations (e.g. relu, tanh, add, sub, mul)`；
- "Operations that need more precision (FP32/FP16)"：`Adding small values to large sums can lead to rounding errors`、`Reduction operations (e.g. sum, softmax, normalization)`、`Pointwise operations where |f(x)| ≫ |x| (e.g. exp, log, pow)`、`Loss functions`。

页脚出处是 `https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/files/dusan_stosic-training-neural-networks-with-tensor-cores.pdf`。讲者由此点出低精度之所以是 "black art"："not because you can just like decrease precision. Anyone can do that. You have to figure out which of your operations has to be in which number format"——matmul 的权重与激活可以低精度，softmax 或 exponential "maybe I need to do that in FP32... or maybe I can get away with BF16"；"it took several years really to get low precision training to be the state that it is today"，靠的是 "slow, incremental, empirical work" 去判断哪些能 downcast、怎么 downcast、以及如何在尽量低精度的同时 "preserve stability of training"。

前沿部分（"Frontiers in low precision"）先讲 FP8。讲者的动机很实在："once you think about how hard it is to pay for fast memory, you realize if only I can cut my precision by another factor of two... people really, really want to do that"，于是 "FP8 was kind of the next frontier"。但 "once you get to FP8, there's no longer a standard canonical format that everyone uses"——16 位时代 "you're probably using BF16"，8 位时代则分裂成 `E4M3`（four bits of exponent and three of mantissa，4 位指数、3 位尾数）与 `E5M2`（more exponent, fewer mantissa，更多指数、更少尾数），"because these are used in different places, there's no one-size-fits-all once you start to get to so few bits"。屏幕左侧把四种格式的位分配画成了格子：表头 `sign / exponent / mantissa`，FP16 是 $1+5+10$ 位，行末标 $=0.395264$；BF16 是 $1+8+7$ 位，$=\;0.394531$；`FP8 E4M3` 是 $1+4+3$ 位，$=\;0.40625$；`FP8 E5M2` 是 $1+5+2$ 位，$=\;0.375$。四行其实在用不同格式逼近同一个约 $0.39$ 的数，可以直观看到位数越少、量化误差越大（$0.375$ 与 $0.40625$ 分别偏低、偏高）。这也是"没有统一格式"的原因：指数位多则动态范围大、尾数位多则精度高，鱼与熊掌在 8 位里不可兼得。

最后讲 MXFP8（Blackwell）。屏幕右半标题 "Multiple scaling factors MXFP8 (Blackwell)"，用两张网格对比：普通 FP8 是一整片橙色 `data` 配一个单独的小方块 `Scaling factor`（标 `FP32`）；MXFP8 则把 `data` 分成多色块，旁边竖排一列小方块也是 `Scaling factor`，每格标 `E8M0`。左下角三条 bullet 是转写没念全、图上独有的："Uses E4M3 (more mantissa) due to more scale factors"、"Scale factors themselves are FP8 (E8M0), 1 per 32"、"Transposes are now nontrivial!"。讲者的口头解释是传统 FP8 里 "all of your, let's say, activations are in eight-bits... and then you're going to have a scaling factor in FP32 that's going to allow you to scale this up and down"，为什么需要缩放因子——"because you only have four bits of exponent, let's say, right? That's so few bits of exponent, you're going to quickly over-..."（转写在 [39:15] 处截断，讲者话没说完，此处不替他补结论）。多缩放因子的好处是能适应矩阵内部不同区域的数值范围，代价是转置不再平凡：转置后每 32 个元素一组的分组模式变了，需要重新量化。屏幕下方的框图还给出训练流程：`Forward pass` 里 `High precision` 输入经 `Cast` 变成 `rowwise` 低精度送进 `Matrix multiply (fwd)`，`Weights` 以 `columnwise` 形式进入；`Backward pass` 里并列 `Matrix multiply (dgrad)` 与 `Matrix multiply (wgrad)`，末端再经 `Cast` 回到 `High precision`。易错点：低精度的收益不是线性的——缩放因子本身仍占 FP32 或 E8M0、量化／反量化与 `Cast` 都有开销，把位宽减半不会自动变成两倍加速；而且本节到 [39:19] 只讲到 MXFP8 的开始，转写在此截断，关于 FP4 / MXFP4 等更极端格式本段没有出现，不应提前补写。

![FP8 各类位分配、MXFP8 多缩放因子与 Blackwell 前反向传播框图](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0031_00-38-37.jpg)

## 从单一缩放因子到 MXFP8：多缩放因子、E8M0 与转置难题 [39:19 - 41:39]

本段开头承接前面关于低精度量化的讨论：要把浮点数压进 FP8 这么窄的表示，必须靠 scaling factor（缩放因子）把数值整体搬到一个既不会 overflow（上溢）、也不会 underflow（下溢）的区间里，才能「keep some information」，也就是保留住有效信息。讲者随即指出，单靠一个缩放因子并不 optimal（最优），因为「a single matrix might have very different magnitudes」——同一个矩阵里，序列的某一部分激活值可能比另一部分大得多。于是 MXFP8 以及这类更先进的 multi-scaling factor（多缩放因子）格式不再给整个矩阵配一个缩放因子，而是配许多个：每个缩放因子对应矩阵里一块带颜色的子矩阵（relevant colored submatrix），「the first four elements in this matrix corresponds to this first scaling factor, and so on and so forth」，即最前面几个元素归第一个缩放因子管，后面的依次类推。直觉上，这相当于把「全局一张量程表」换成「分片量程表」，让大值区域和小值区域各自拥有合适的动态范围，代价是缩放因子本身也要占位数、也要参与运算。

幻灯片「Frontiers in low precision」把这件事讲得更具体。左半边标题是「Very low precision (FP8) with different tradeoffs」，把四种位模式并排列出：FP16、BF16、FP8 E4M3、FP8 E5M2，每行都按 `sign | exponent | mantissa` 三段涂色，并给出对应的十进制近似值 0.392264、0.394531、0.40625、0.375。E4M3 是 1 位符号 + 4 位指数 + 3 位尾数，E5M2 是 1 + 5 + 2，二者总位数都是 8 位：

```text
FP8 E4M3  : [s][e e e e][m m m]    1 + 4 + 3 = 8 bit
FP8 E5M2  : [s][e e e e e][m m]   1 + 5 + 2 = 8 bit
E8M0 scale: [e e e e e e e e]     只有 exponent bits，取值为 2 的幂
```

右半边「Multiple scaling factors MXFP8 (Blackwell)」把普通 FP8（整块橙色 data 配单个小方块 scaling factor）与 MXFP8（data 被切成青、绿、黄、紫多块，旁边一列是对应颜色的 scaling factor）并置；下方还画了 Forward pass / Backward pass 的数据流：高精度输入经 `Cast` 进入 `Matrix multiply (fwd)`，反向则是 `Matrix multiply (input)` 与 `Matrix multiply (weight)` 两条路径，末尾再 `Cast` 回高精度。左下的三条 bullet 是本段最关键的「图上独有」信息：MXFP8 uses E4M3 (more mantissa) due to more scale factors；Scale factors themselves are FP8 (E8M0), 1 per 32；**Transposes are now nontrivial!** 也就是说缩放因子本身也是 8 位、且是 E8M0（只有指数位），所以缩放系数的取值全是 2 的幂：

$$x_i^{(q)} = s_{\lfloor i/32 \rfloor}\cdot x_i,\qquad s_k = 2^{e_k},\ e_k \in \text{E8M0}$$

每 32 个元素共享一个 $s_k$。2 的幂约束意味着缩放因子只能取离散的 $2^{e_k}$，而不是任意实数比例，这是精度与硬件友好度之间的 tradeoff，也呼应讲者那句「So, you have to make these tradeoffs」。

![FP8 位模式对比与 MXFP8 多缩放因子结构](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0032_00-39-56.jpg)

讲者在这里故意停顿：「don't read the third thing。Now, think for a moment, what is the problem with this design?」——让听众自己想这个设计的毛病，答案正是第三条 bullet：转置。原来矩阵转置是「免费」的，$A^T$ 只是换个索引，不需要重新量化；但 MXFP8 的分组是按行方向每 32 个元素切块的，转置之后缩放因子的分组模式（scaling pattern）完全变了，于是必须把整个矩阵重新量化，去凑那个「one out of 32 pattern」。讲者说实际用 MXFP8 训练时，库会为每个量化矩阵**同时保存两份副本**：一份原矩阵、一份转置矩阵，「if you ever want to transpose, you have the transposed version waiting for you」，他形容这件事「crazy, but also kind of cool」。这与后面 42:23 的「MXFP8 training in practice」图直接呼应，那里明确写着 transposes also separately quantized。易错点在于别把 FP8/FP4 当成「只是把 float 换成更小的类型」——讲者在 41:44 强调，一旦进入 FP8、FP4 这类格式，「things start getting pretty exotic」，因为 exponent 与 mantissa 的损失非常 extreme，转置、累加、缩放都要重新设计；而且 E8M0 缩放因子粒度比浮点缩放因子粗，遇到矩阵内部量级突变虽然比单缩放因子好，仍不是无损。这一段承接前面「低精度减少字节移动」的主线，同时把话题从数值表示引向系统实现代价，为下一章讲训练实践与真实收益做铺垫。

## MXFP8 训练实践：只量化安全层、20–30% 收益与 MXFP4 的全部取值 [41:39 - 44:20]

讲者把话题收束到「真正用 FP8 训练模型」：他认为不少 frontier labs（前沿实验室）为了 compute benefits 正在这么做，但「you have to be very careful」。幻灯片「MXFP8 training in practice」用一个 Transformer block 图说明这一点：图例把白色标成 BF16、绿色标成 MXFP8，block 内部的 `Layer Norm`、`BMM 1`、`Softmax`、`BMM 2`、`Add`、`Act Func` 保持白色，只有 `Projection`、`FC1`、`FC2` 以及 Layer Norm 之后的一个绿色小块被涂成绿色。也就是说**并非所有权重都在 MXFP8**（Notice – not all weights in MXFP8），只挑「认为安全」的层做量化。讲者在转写里也强调这是「another trial and error thing that you have to figure out」：哪些层可以量化靠反复试验，不是一条能推导出来的规则。这与上一章的转置难题合起来，构成低精度训练的两大工程开销：只量化部分层，加上为每个量化矩阵准备原版与转置版两份副本。

下半张图把一次前向/反向拆成三条矩阵乘路径，标识符按幻灯片原样保留：

```text
Weight (BF16)      → quantize            → MXFP8 → FPROP → To next layer → BF16
Activation (BF16)  → quantize            → MXFP8 → FPROP
Activation (BF16)  → transpose & quantize→ MXFP8 → DGRAD
DGRAD (BF16) / MXFP8                              → WGRAD
WGRAD              → MXFP8 → transpose & quantize → Gradient
FP32               → To optimizer for master-weights update
```

`FPROP`（forward prop）、`DGRAD`（data gradient）、`WGRAD`（weight gradient）三个方框分别对应前向、输入梯度和权重梯度的矩阵乘；每条路径都要先 `quantize`，需要转置的地方用 `transpose & quantize`，梯度侧还要再量化一次才输出 `Gradient`。图底一行字点题：「Notice – not all weights in MXFP8, transposes also separately quantized」，右下角给出来源 `https://arxiv.org/html/2506.08027v2`。这里有一个容易忽略的细节：优化器更新用的是 **FP32 master weights**（FP32 to optimizer for master-weights update），低精度只作用在矩阵乘的输入上，权重主副本仍是高精度，这正是「低精度是计算与搬运的优化，而不是把整个训练都降精度」的直观体现。

![MXFP8 training in practice 的 Transformer block 与 FPROP/DGRAD/WGRAD 数据流](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0034_00-42-23.jpg)

收益的量化来自转写：把这些代价都付完，做 matmul 时「you're probably going to get 20 to 30% savings, maybe more, depending on the size of your matrix」。但讲者立刻泼冷水——不会是 2× 加速，因为所有的 quantize / dequantize 操作本身就是 overhead，「There's a big overhead doing this, so it's not free」：

$$\text{speedup}_{\text{real}} \approx 20\%\sim30\%\ (\text{或更多}),\qquad \text{speedup}_{\text{real}} \ll 2\times$$

紧接着 42:48 有一段关于首层/末层的提问：同学提到「the first layer and the last layer are hard to quantize」，讲者承认「That's right」，并给出直觉——末层之所以难量化，是因为它是直接决定 loss 的一阶因素（「it is kind of a first-order factor. It derives a ton of the loss」），量化它会同时带来 instability 和 loss 的大幅上升；而首层他坦言「I don't know if I have intuitions for the first layer」，这种含糊处就不该替他补理由。随后 43:13 他展示 MXFP4 的幻灯片，说这张图荒谬得近乎可笑：整个 MXFP4 能表示的值一屏就能列完，「It's negative six to six」。表格按 `sign | exponent | mantissa` 三段列出 4 位模式，正负两侧各 8 行，取值依次为 $\pm 0.0,\ \pm 0.5,\ \pm 1.0,\ \pm 1.5,\ \pm 2.0,\ \pm 3.0,\ \pm 4.0,\ \pm 6.0$：

$$\{-6,\,-4,\,-3,\,-2,\,-1.5,\,-1,\,-0.5,\,-0,\,+0,\,+0.5,\,+1,\,+1.5,\,+2,\,+3,\,+4,\,+6\}$$

![MXFP4 的 4 位位模式与全部可表示值](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0035_00-43-37.jpg)

屏幕底部那行字是「This is all the values you can represent! 1 per 16 scaling, E4M3 scaling factors.」——即每 **16** 个元素配一个缩放因子（比 MXFP8 的 32 更密），且缩放因子改用 **E4M3**（不再是 E8M0）。讲者说已有论文用 FP4 训练过，但「I don't think I've heard of anyone successfully training in the wild... real big serious models with FP4」，并预测「the next generation of models will probably be FP4」。易错点：MXFP4 的 16 个取值并不是等间距的（0.5 到 6 之间跳变很大），它不是「低精度版的 FP8」，而是一套完全不同的离散码本；把 MXFP8 的 32 元素分组、E8M0 缩放因子直接套到 MXFP4 上是错的，分组粒度与缩放因子格式都变了。这一章把低精度从位模式推进到训练流水线，也把话题从收益数字引到「哪些对象值得量化」，正好接上 44:20 之后那一连串问答。

## 低精度的适用边界：量化对象、带宽与算力、缩放因子的确定方式 [44:20 - 47:42]

讲者用「Okay. So, that was low precision」收束低精度主线后，进入一段密集问答。第一个问题 44:23 问：这套方法是否只适用于矩阵、不适用于单个标量值？回答是「you can quantize anything」，比如 ReLU 之后的 activations 也能量化，「but there just isn't as much of a need to」：matmul 一旦量化就能拿到很大的 throughput improvements，所以 MXFP8 训练里被 FP8 量化的全是 matmul；对于标量或逐元素操作，量化/反量化的 overhead「is probably not worth it for most of the time」。这里的关键是**收益究竟来自哪里**：低精度本身会减少 memory movement（任何低精度数据都少搬字节），但只有 matmul 这种算术强度高、被 Tensor Core 特权招待的算子，才能把字节节省真正兑换成吞吐。

第二个问题 44:55 追问这到底是硬件限制，还是带宽/算力问题。讲者澄清：这不是硬件限制（「It's not a limitation」），但硬件里确实「baked in a bunch of this」来加速；对 compute 来说，「you get basically linear improvements for just multiplying the quantized numbers」，而 quantize / dequantize 让收益被稀释：

$$\text{compute gain} \propto \text{precision reduction}\ (\text{近似线性}),\qquad \text{net gain} < \text{linear due to de/quant overhead}$$

第三个问题 45:43 问缩放因子要不要训练，回答是「the scaling factor is an implementation-dependent choice」——不同库有不同 quantize / dequantize 方式，「training is a strong word」：

```text
quantize(x):  s = max(|x|) or (max, min)   # 扫描一遍取极值
quantize(x):  s = running_statistics(x)   # 用历史 running statistics
```

讲者明确说训练时**没有梯度**（「So there's no gradients.」→「At training time you would not」），只在 inference 时「you might fit them more」。也就是说缩放因子是统计量或启发式，不是可学习参数——这是最容易被误解的一点，也是与上一段「1 per 32 / E8M0」参数配置相配套的实现细节。

第四个问题 46:13 问 structured sparsity（结构化稀疏）。讲者说这方面做过的很多，MoE（mixture of experts，混合专家）是「one successful example of a structured sparse operation that people use」，还提到 Chris Ray 在「lots of different kinds of structured matrices」上做过大量工作；但他给出的判断是收益「in terms of the compute benefits versus the representation losses... has kind of washed out」，即算力收益被表示能力损失抵消，empirically「hasn't, I think, necessarily panned out」，「I don't think it's a bad idea」但经验上没兑现。第五个问题 46:49 问 state of the art：是不是 train in full net 或 half net、再 distill 成量化模型？回答是「you probably want to train a bigger model and then quantize down」，并提到有各种 quantization scaling laws 的论文在「precisely characterizing the trade-offs here」；他认为 SOTA 是「combining both like quantization aware training to some extent and like model scaling with post-training quantization of various kinds」，即 QAT（量化感知训练）+ 更大模型 + PTQ（训练后量化）的组合，而且「there's a lot that I think is not fully well understood」，连他聊过的工业界团队都还在做「science of quantization」。局限与易错点要一并记住：不要把「能量化任何东西」理解成「量化任何东西都划算」；不要把线性算力收益当成端到端收益；不要把缩放因子当成可训练参数；结构化稀疏与 quantization scaling laws 都还是经验性结论而非定论。这段问答既补完了上一章留下的「哪些层、哪些对象可量化」的疑问，也用 47:42 的「the other trick」把话题从数值精度彻底转向编程/系统手段——下一章的 operator fusion（算子融合）不再动数值，而是动计算图的执行方式。

## 算子融合：把多次全局内存往返压成一个 kernel [47:42 - 50:12]

讲者把 operator fusion（算子融合）称为「much more of like a conceptual programming thing」，并说「you'll be surprised at like how much this doesn't already happen」。他用工厂比喻建立直觉：GPU 里全局内存是仓库（memory warehouse），SM 是小小的 compute factory，两者之间是一条来回传送的 conveyor belt（传送带）。如果只提升算力，瓶颈就落在这条传送带上；而操作越多，原材料在仓库与工厂之间来回运输的次数越多，付出的双向（duplex bidirectional）内存带宽代价越大。「Wouldn't it be better if I just had one giant factory that would take all the raw materials and ship back my finished products?」——这样内存带宽只需付两次。这就是 fusion：把一串操作合成单个 kernel，从 global memory 读一次，在 SM 内部做完，再写回一次。

举的例子是 $\sin^2 x + \cos^2 x$（其结果恒为 1，但这里关注的是计算图本身）：

$$\sin^2 x + \cos^2 x = 1,\qquad y = \text{add}\big(\text{pow}(\sin x,\,2),\ \text{pow}(\cos x,\,2)\big)$$

```python
y = torch.sin(x) ** 2 + torch.cos(x) ** 2   # 朴素计算图
```

PyTorch 的计算图是 $X \to \sin$、$X \to \cos$、两侧各自平方、再相加得到输出；naively 每一个节点都是一座「工厂」，每次调用都要读写一次全局内存（「You will read and write from global memory each one of each time one of these is called」）。幻灯片「Operator fusion to minimize memory access」用手绘图把这一点画得很直白：左边「Naïve (non-fused)」在 Memory 与 Compute 两条竖线之间画了多条横带，每条带子都有来回箭头，配文「What if we have to do many operations? Shipping back and forth is somewhat silly」；右边「Fused kernel」只保留底部一条横带，Compute 侧把各个小算子用箭头串成一条链。融合后的 kernel 写成：

```text
fused_kernel(x):  read x once → sin / cos / pow / add inside SM → write y once
```

于是全局内存往返从「每个算子一次」降到「整个子图一次」，也就是讲者说的「I only have to pay for memory bandwidth twice」：

$$\text{HBM round trips: } N_{\text{ops}} \longrightarrow 2\ (\text{1 read} + \text{1 write})$$

「Fusion example」幻灯片给出了更具体的编译器信息：左侧「Before operator fusion」是一个被红框圈住的计算图，里面是一串 PyTorch ATen 节点；右侧「TorchInductor operator fusion」把整张图压成一个橙色 kernel 节点，上下各接一个小节点。图下注明：「All 5 pointwise operations can be fused into a single CUDA kernel call. 'Easy' fusions like this can be done automatically by compilers (torch.compile)」——是 **5 个** pointwise 操作（sin、cos、两次平方、add）被 fuse 成一次 CUDA kernel call。

![TorchInductor 把 5 个 pointwise 算子融合成单个 CUDA kernel](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0040_00-49-46.jpg)

转写里点名两个编译器：Torch Compile（「a compiler for PyTorch」）和 JAX Compile（「more deeply integrated into Jax」），二者都会把这类基本操作「fuse these kinds of basic things into a single CUDA call」。易错点有两层：一是融合并不减少 FLOPs，只减少中间结果的全局内存读写，所以只有在算术强度低、逐元素操作密集的子图上收益才明显；二是「the advanced version sometimes requires sort of manual intervention」——编译器只能自动处理容易的融合，复杂的融合仍需手写 kernel。这一章与上一章的低精度形成对照：低精度让每次搬运的字节更少，融合让搬运的次数更少，两者都服务于同一个目标——减少全局内存流量；而后面重计算、合并访存、tiling 三章会继续沿着这条主线，分别用「计算换存储」「对齐换效率」「共享内存换带宽」推进。

## 重计算：用便宜的算力换昂贵的激活存储 [50:12 - 52:52]

讲者把 re-computation（重计算）称为「the third idea」，并刻意换视角：不再从数学上问「怎么算梯度」，而是从系统上问「怎么最小化内存」（「we're thinking about this from the perspective of systems」）。标准 backpropagation（反向传播，他提到这是 Percy 的 221 课内容）是沿计算树向上做前向、把每一层的 activations（激活）存下来，反向时再取出这些激活去计算向下的梯度信号；这些「小小的激活数字」是要占全局内存、要读写全局内存的。Sigmoid 这种逐元素算子算术强度极低，堆叠一堆 sigmoid 时，存激活的代价远超计算本身的代价。于是出现一个反直觉的做法：把中间激活全部丢掉（「throw away all of the activations」），在反向需要时**重新做一次前向**，on the fly 地把激活算出来。前提是世界里「computation is super super cheap. You just have abundant computation. You're not using all of your units, but your memory is very expensive」。

例子是 $X \to \text{sigmoid} \to \text{sigmoid} \to \text{sigmoid} \to \text{out}$：

```text
X → sigmoid → sigmoid → sigmoid → out
fwd 存: S2, S1, out        bwd 用: S2, S1, dout → dx
```

幻灯片「Storing (and retrieving) activations can be expensive!」把旧方案画成两个图：Old Fwd pass 里 $X$ 经过三个 sigmoid，分别产生 $S_2$、$S_1$ 和 $out$，标注「1 mem read / 3 mem writes」；Old Bwd pass 里「Backward graph」吃进 $s_2$、$s_1$、$dout$ 并输出 $dx$，标注「3 mem reads / 1 mem writes」。合计：

$$1 + 3 + 3 + 1 = 8\ \text{mem read/writes},\qquad \text{very low arithmetic intensity}$$

![保存激活的旧方案：前向 3 次写、反向 3 次读，合计 8 次访存](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0041_00-50-59.jpg)

幻灯片底部的原话是「This is really terrible for perf – 8 mem read/writes, very low arithmetic intensity.」（图下链接指向 PyTorch dev-discuss 上关于 min-cut optimal recomputation / activation checkpointing with AotAutograd 的帖子）。新方案见「Throw away the activations, re-compute them!」：New Fwd pass 只读 $X$、只写 $out$，「1 mem read / 1 mem write」；New Bwd pass 在拿到 $dout$ 和 $X$ 之后，现场再跑一遍三个 sigmoid（图中用蓝框标出的重算部分），把结果喂给「Original Backward graph」，「2 mem reads / 1 mem write」。合计变成 **5** 次：

$$\text{New: } 2 + 3 = 5\ \text{accesses},\qquad \frac{5}{8}\ \text{of the memory accesses}$$

图底写着「Throwing away computation can actually be optimal, w/ 5/8th the memory accesses!」。注意 $out$ 仍然必须写回，因为它是整张计算图的输出——重计算省掉的是中间激活 $S_1, S_2$ 的存取，不是输出本身。

![丢弃激活、反向时重新前向：访存降到 5/8](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0042_00-52-13.jpg)

讲者强调「Same computation, but I've spent a little bit of extra compute and saved a little bit of extra memory」，并说「We don't normally think about this because usually we're concerned with like just computing the gradient, but once you have a system's mindset, you start to think about recomputation」。这正是 activation checkpointing（激活检查点）的核心：用额外一次前向的算力，换掉激活的全局内存读写。与上一章的融合相比，融合减少的是「算子之间的往返次数」，重计算减少的是「为反向保存的中间结果」，两者常常叠加在同一个 kernel 里。易错点与局限很明确：这个交易只在算力相对便宜、内存访问昂贵、且计算单元没被占满时才划算；如果算子本身是 matmul 这类高算术强度运算，多跑一次前向的代价可能远超省下的内存流量；另外重计算会让反向路径变长，实现上需要框架在计算图里自动切分（幻灯片引用的正是 min-cut optimal recomputation 的思路），不是随手删掉保存逻辑就完事。这一章把「用一种资源换另一种资源」的思路讲透，下一章的合并访存虽然也谈内存，但换的不是算力，而是**访问顺序与地址布局**。

## 合并访存：DRAM burst、coalesced 定义与行主序矩阵 [52:52 - 57:52]

讲者称第四个 trick 是「another sort of memory trivia thing」，但「surprisingly becomes important」，也是「why sometimes it's important to understand the hardware model of what you are interacting with」的理由。global memory（全局内存）是芯片外的 DRAM，而 DRAM 因为物理结构「actually comes in bursts」（按突发方式读取）：访问第一个位置时，并不是只拿到第一个值，而是能把同一段里前四个值一起「免费」读出来。原因是 memory cells 排成网格、成块排列，激活一个 sense amplifier（读出放大器）后，同一行/同一列的多个单元可以几乎零成本一起读出。这被称作 **burst section**（突发段）：一个 section 被访问时，其中所有位置「can also get delivered for free」。于是「if you have your reads arranged contiguously where they're all together, it's going to go much faster」——这引出 coalesced（合并访存）的定义。

幻灯片「Trick (?) 4: Memory coalescing and DRAM」给出的基本例子是一个 **16-byte** 地址空间，切成 **4-byte** 的 burst sections：0 1 2 3（黄）、4 5 6 7（红）、8 9 10 11（蓝）、12 13 14 15（绿），标题下写着「DRAM (global memory) is read in 'burst mode' – each read gives you many bytes!」。四条 bullet 原样是：「Each address space is partitioned into burst sections」；「Whenever a location is accessed, all other locations in the same section are also delivered to the processor」；「Basic example: a 16-byte address space, 4-byte burst sections」；以及「In practice, we have at least 4GB address space, burst section sizes of 128-bytes or more」。底部箭头指向一句「Burst mode comes from the slow per-row copy to the sense amplifier」。转写里对应的数字是「you might have something like 128 bytes worth of a burst. That means a single read will return potentially 128 bytes worth of data as long as it lives in the same contiguous block」：

```text
16-byte 地址空间 / 4-byte burst sections:
 burst 0: [ 0][ 1][ 2][ 3]   黄
 burst 1: [ 4][ 5][ 6][ 7]   红
 burst 2: [ 8][ 9][10][11]   蓝
 burst 3: [12][13][14][15]   绿
```

![DRAM 按 burst section 读取：16 字节地址空间、128 字节突发段](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0043_00-53-27.jpg)

coalesced 的严格定义来自「Memory coalescing」幻灯片：「Memory accesses are coalesced if all the threads (in a warp) fall within the same burst」；灰字补充「When all threads of a warp execute a load instruction, if all accessed locations fall into the same burst section, only one DRAM request will be made and the access is fully coalesced」：

$$\text{coalesced} \iff \forall\, t \in \text{warp},\ \text{addr}(t) \in \text{同一 burst section} \Rightarrow 1\ \text{DRAM request}$$

图里画了两组「Coalesced Loads」（$T_0\ T_1\ T_2\ T_3$ 四个线程箭头）：一组落在黄色 burst（0–3），另一组落在蓝色 burst（8–11），各自只触发一次 DRAM 请求。右下角的「Reminder」是必须记住的参数：「a warp is a set of 32 consecutively numbered threads that execute together in a block. Memory accesses happen together」——warp（线程束）是 **32** 个连续编号、在 block 里一起执行的线程，它们的内存访问是同时发生的；讲者在转写里也反复纠正自己的口误：「three threads and all three of my thread or sorry, four threads. Wait, four threads. And my four threads live inside of a single burst section」。

讲者自问「why are you telling me DRAM trivia」，答案是矩阵：matmul 要按大块读矩阵，「certain kinds of memory accesses on matmuls are sort of privileged」，利用得好能拿到极快的读取。他让大家想象一个 4×4 的 row major（行主序）矩阵：内存里黄色块排在最前、红色块次之，依次线性排开。若线程每个 clock cycle 读一列、然后向右移一列，第一次迭代会命中 0、4、8、12 这四个元素，「Notice that each one of these limit a different burst window, and I read the entire vector... even though all I wanted was a single column」；反过来若把矩阵翻转、让线程沿连续地址读，每个线程一次读取就能拿到数据，「coalescing can give you significant improvements in sort of reads from large contiguous structures like matrices」。55:00 之后有一段追问解释物理成因：memory cells 排成 grid，放大器「operate at a column-wise level」，给整列加电压「often a thing that takes a significant amount of time」，所以一旦选中某列，该列内多个元素的读出「comparatively cheaper」；讲者还纠正了一个误解——「You don't get a key for free」，免费的不是任意键值，而是**同一 section 内的其余位置**。「Coalescing for matrix multiplication」幻灯片把两种情形并排画成 (A)「not coalesced」（标 $d_M$、$WIDTH$ 以及 Thread 1 / Thread 2 的水平线）与 (B)「coalesced」（标 $d_N$、$WIDTH$），右侧还有「Access direction in Kernel code」小图与 $M_{0,0}, M_{0,1}, \dots$ 的「Load Iteration 0/1」时序，底部粗体字说明行主序矩阵里线程沿行跨步访问不合并。易错点：coalesced 判据是「落在同一 burst section」而不是「地址相邻即可」，地址相邻但跨过 burst section 边界仍会触发额外的 DRAM 请求；行/列主序、padding（填充对齐）与 tile 边界都会改变实际访存次数；另外带宽收益与上一章问答里的算力收益不是一回事。这一章把硬件物理特性直接接到数据布局上，正好为下一章的 tiling 提供约束：tile 既要装进共享内存，也要尽量保持 coalesced。

## Tiling：把内存层级用到极致的最重要一招 [57:52 - 58:59]

讲者把最后一个、也是「the big one」「maybe the most important idea」留到最后——tiling（分块），并说「I think this one has probably the big impact on performance」。幻灯片「Trick 5 (the big one): tiling」给出的定义是：「Tiling is the idea of grouping and ordering threads to minimize global memory access.」，副标题「Let's go back to matrix multiplication..」。他的说法是：tiling 比别的技巧更体现「trying to do as much as you can with the memory hierarchy」（尽可能利用内存层级）——把该分组的内存访问分到一组，该在 shared memory（共享内存）上做的访问就一起放到 shared memory 上做。之所以叫 tiling，是因为要把需要的数据「cut up the things that I need, I'm going to tile my space」，每个 tile 被加载进 shared memory，「once that tile is loaded, I'm going to repeatedly read from that tile and do as much as I can」。这直接呼应前几章：融合减少往返次数、重计算减少保存量、合并访存减少单次请求数，而 tiling 是把**同一份数据被全局内存读取的次数**压到最低——靠复用，而不是靠省略。

看矩阵乘的情形：左边矩阵、右边矩阵相乘，右下是输出。幻灯片用 $M_{0,0}\ M_{1,0}\ M_{2,0}\ M_{3,0}$ / $M_{0,1}\dots$ 标出左侧块，$N_{0,0}\ N_{1,0}\ N_{2,0}\ N_{3,0}$ 标出上方块，$P_{0,0}\ P_{1,0}$ 标出输出块；右侧「Access order」表格按 thread 逐格列出每一步要读的乘积项：

$$P_{i,j} = \sum_{k} M_{i,k}\,N_{k,j}$$

```text
thread 计算 P_{i,j} 的访问顺序（依图中 M_{i,k}*N_{k,j} 项）：
 P_{0,0}: M_{0,0}*N_{0,0} + M_{0,1}*N_{1,0} + M_{0,2}*N_{2,0} + M_{0,3}*N_{3,0}
 P_{1,0}: M_{1,0}*N_{0,0} + M_{1,1}*N_{1,0} + M_{1,2}*N_{2,0} + M_{1,3}*N_{3,0}
```

讲者指出，按线程这样读下去，「I'm going to be reading... the same entry will be read multiple times」，并且幻灯片底部明确写着「Note that memory access is not coalesced, and repeated (M0,0 and N1,0)」——同一个 $M_{0,0}$、$N_{1,0}$ 会被不同线程反复从全局内存读取，而且这些读取并不 coalesced。也就是说，朴素实现同时犯了两个错：重复读 + 不合并读。tiling 的做法是把 $M$ 与 $N$ 各切出一块 tile 塞进 shared memory，让一个 tile 内的所有线程都从这块快速内存里取数，把原本散落在全局内存上的多次请求收敛成「每 tile 一次全局读入 + 多次共享内存复用」。

![Trick 5 tiling 的矩阵乘访问顺序与重复读取 M0,0、N1,0](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0047_00-58-22.jpg)

这一章是前六章的汇总点：低精度让每次搬的字节更少，融合让搬运次数更少，重计算让需要保存的中间量更少，合并访存让每次请求更有效，而 tiling 让「同一数据被全局内存读取的次数」最少，并且天然要求前几章的条件同时成立——tile 要装得进 shared memory（受容量限制）、tile 内访存要 coalesced（受上一章的 burst 对齐约束）、tile 边界要对齐（受 128 bytes 与 padding 影响）。讲者在本段末尾才开始说「And to be precise, each...」，定量分析已经超出 58:59 这一段的转写范围，因此不在这里替他补公式。易错点：不要把 tiling 理解成「把矩阵切小就算完」——切得太大装不进共享内存，切得太小则复用次数不够、全局内存读取次数又回升；而且如果 tile 的读取跨 burst 边界，合并访存的好处会被抵消。本段只建立了 tiling 的动机与「重复读取」这一观察，具体的 tile 尺寸取舍与性能曲线属于后续内容。

## 分块（tiling）的动机、分阶段流程与内存复用 [58:59 - 61:19]

本段从上一节「合并访存（coalescing）」自然过渡到六大加速手段里的最后一项、也是最重要的一项——分块（tiling）。讲者先给出一个朴素事实：在一个 $n \times n$ 的方阵矩阵乘（matmul）里，输入矩阵的每一个元素会被读取 $n$ 次。原因是输出有 $n^2$ 个元素，每个输出都是一行与一列的内积，同一份输入元素会被反复用到；如果每次都去全局内存（global memory，也就是 HBM）取数，代价极高。既然同一个元素要被读很多次，直觉上就应该把这些内存访问「归组」：不要一次次敲全局内存，而是一次把一小块搬进离计算单元更近、速度快得多的共享内存（shared memory，幻灯片缩写为 SHM），然后在共享内存里反复读写。矩阵之所以特别适合这么做，是因为它有一种非常自然的切法——把大矩阵切成子矩阵（submatrix，也就是 tile），逐块加载、逐块运算、最后写回。讲者用「2 by 2 submatrices」举例：先取左上角 $M_{00}$ tile 和上方矩阵的 $N_{00}$ tile，付一次从全局内存慢慢搬运的成本，之后就在这两个 tile 之间高速读写。

这一流程在幻灯片里被写得很具体。标题是 "Tiling – store and reuse information in shared memory"，副标题 "Cut up the matrix into smaller 'tiles', and load this into shared memory"。右侧把它写成按阶段（phases）推进的步骤：1. Load $M_{0,0}$ and $N_{0,0}$ tiles into SHM；2. Compute partial sums for $P$（Done with one tile）；3. Load the $M_{0,0}$ and $N_{2,0}$ tile into SHM；4. $\ldots$。左侧彩色方块图里，一个输入矩阵的 tile 标成 $M_{0,0}\ldots M_{1,3}$，另一个矩阵标成 $N_{0,0}\ldots N_{3,1}$，输出块标成 $P_{0,0}\ldots P_{3,3}$：两个子矩阵在共享内存里相乘，把部分和（partial sum）累加进同样暂驻共享内存的输出子矩阵，全部算完再写回全局内存。底部用粗体给出结论 "Advantages: repeated reads now access shared, not global memory and memory access can be coalesced"——重复读落在共享而非全局内存，而且访存还能保持合并。这里把「复用」和「合并」两件事同时点出来，正是前几章伏笔的汇合处。

![分块把 tile 搬入共享内存、按阶段累加部分和的流程图](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0048_00-59-41.jpg)

接着讲者把收益写成公式。"Tiling math" 幻灯片给出 Matrix A、Matrix B、Matrix C 三幅网格图，上方括号分别标着 tile size $T$ 与 matrix size $N$，图例依次是 Outer loop over tiles、Current tile in outer loop、Inner loop over elements、Current element in inner loop、Temporary result tile——也就是外层循环扫 tile、内层循环扫元素、中间结果暂存在橙色 tile 里。底部两行粗体字原文是 "Non-tiled matrix multiply: each input is read $N$ times from global memory" 与 "Tiled matrix multiply: each input is read $\frac{N}{T}$ times from global memory, and $T$ times within each tile. This is a factor of $T$ reduction in global memory access"。写成式子即

$$
\text{global reads per input}=\frac{N}{T},\qquad \text{shared-memory reads per input}=T
$$

当 $T=N$ 时取到极端：每个输入只从全局内存读 $1$ 次，其余 $N$ 次访问全部发生在共享内存里，相当于把全局内存访问削减了 $T$ 倍。讲者说这在 matmul 这类算子上「dramatically reduce your global memory access」，是 tiling 收益巨大的根本原因。注意口述里混用了小写 $n$，而幻灯片统一用大写 $N$ 表示矩阵尺寸、$T$ 表示 tile 尺寸，两者指的是同一件事。

![Tiling math 幻灯片：N 次、N/T 次与 T 次读取的对比及网格图例](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0049_01-01-05.jpg)

随后有学生追问：这么做是不是因为无法一次性更新整个矩阵？讲者确认「Yes. Yes. Yes. Yes. That's right」，并补了一句很关键的边界条件：如果换成像 Rock 那样几乎全用 SRAM 的另类加速器，你完全可以把整个矩阵放进去、全程高速，但「very expensive, very very expensive」。也就是说 tiling 不是数学上的必需，而是「共享内存太小、全局内存太慢」这一硬件现实下的折中；它承接前文的合并访存（都在减少 HBM 流量），也为后面两章埋下伏笔——只要 tile 切得不合适，收益立刻被抵消。另一个易错点是：把数据搬进共享内存本身要付费，只有当同一 tile 内的复用次数 $T$ 足够大、省下的全局内存读超过搬运开销时，tiling 才真正划算；而 $T$ 能取多大又受限于共享内存容量，这个约束下一章就会展开。

## 自动 tiling、tile 尺寸的复杂度与 max-autotune [61:19 - 63:58]

讲者先把 tiling 放进更大的并行化图景里。学生问「这种切分是不是也发生在加速器层面」，讲者答 "That's right"，并说 data parallel（数据并行）可以想成「if I have data on this axis and weights here if I cut it on one of the axes」，而 tensor parallel（张量并行）则是「another kind of parallelism」，最后总结 "All these parallelism look a little bit like tiling in some ways"。接着有人问 PyTorch 是否会自动做，回答很干脆："Tiling yes yes yes yeah. So like any like matmul kernel that you call will do tiling underneath"，即任何 matmul kernel 底层都会 tiling。但紧接着就留了转折："even PyTorch or something is not perfect and so you know we will talk about some of the imperfections that will happen"，以及 "If you're doing some sort of exotic operation you might want to think about tiling like things happening right? You want to be able to locally reuse your reads"。换言之 tiling 作为底层系统细节通常被库隐藏，使用者不必手写 kernel；但正因为它是自动的，一旦输入形状与自动选中的 tile 不匹配，性能就会出现难以解释的塌陷。

于是讲者用一组具体数字展示「tile sizing is actually pretty complex」。设 tile size 为 $128\times128$、矩阵为 $256\times256$，此时「life is good. I've got four tiles right? Perfectly cut」，即 $\frac{256}{128}=2$、$2\times2=4$ 个整齐的 tile。现在把矩阵维度只加 $1$（口述说 "increase my coordinate size by one"），就会「generated two very skinny tiles with basically nothing inside of them」，大量算力花在空转上。于是 tile 尺寸变成一个需要权衡的优化量：一边是内存带宽（tile 越大、复用越多、全局读越少），另一边是那片「gigantic empty space where I'm doing nothing」。讲者列出三个共同决定最优 tile 的因素：coalesced memory accesses（合并访存）、the size of your shared memory on your accelerator（加速器共享内存大小）、the sizes of your matrices（矩阵尺寸）。沿用上一章的公式 $\text{global reads per input}=\frac{N}{T}$，就能看出矛盾：$T$ 增大压低全局读，但 $T$ 超过矩阵尺寸后多出来的部分是纯浪费。

这一时段的 "Complexities with tiling" 幻灯片（63:54 的清晰版与 62:29 的投影版内容相同）把上述现象写成了可核对的文字。粗体副标题是 "Tile sizes may not divide the matrix size and lead to low utilization"。Figure 6 的图注逐字为 "Example of tiling with 128x128 thread block tiles. (a) Best case - matrix dimensions are divisible by tile dimensions (b) Worse case - tile quantization results in six thread blocks being launched, two of which waste most of their work."。图 (a) 是 $256\times256$ 的方块并标出 $128$ 的分割线，图 (b) 标 $257$，右侧与上侧用虚线画出被浪费的那一圈；把图里的算术写出来就是 $\lceil 257/128\rceil=3$，于是一侧变成 $3\times2=6$ 个 thread block，其中两个几乎全空——这正是「six thread blocks being launched, two of which waste most of their work」的数字来源。下方 "Factors affecting tile sizes" 列了三条：Coalesced memory access、Shared memory size、Divisibility of the matrix dim；页脚还给出 NVIDIA 文档链接（docs.nvidia.com 的 dl-performance-matrix-multiplication 页，锚点 #tile-quant）。这些标题、bullet 与数字都是画面独有、口述没有逐字念到的内容。

最后是工程上的解法：PyTorch 有个编译器选项叫 `max-autotune`（讲者念作 max auto tune），「you turn on that flag, for the next 15 minutes PyTorch is going to like run a endless series of benchmarking tests」，本质是在你的矩阵上穷举各种 tile 尺寸并 benchmark，「trying all sorts of different tile sizes on your matrices and trying to figure out which one is actually faster」，能拿到 "non-trivial benefits"。数字是 $15$ 分钟的编译期搜索换运行时收益。局限也很清楚：这是编译期搜索，代价是等待时间，且结论绑定「给定形状」，形状一变就可能失效；对 exotic operation 库不会替你 tiling，得自己考虑局部复用读。本章把 tiling 从「一个漂亮想法」拉回「一个有工程复杂度的调参对象」，下一章顺着 coalescing 与对齐继续深挖为什么形状差 $1$ 会致命。

## 合并访存错位、padding 与矩阵维度整除性 [63:58 - 66:15]

承接上一章的三个因素，讲者专门展开 coalescing 与 tile 边界的相互作用。DRAM 按 burst section 成段读取：讲者用「蓝色这一行」举例，假设 burst section 的宽度正好等于 tile 宽度，那么读一个 tile 只要 $4$ 次读取就能把整个子矩阵取回，"I'm good to go because because of the burst sections, I read this entire submatrix, right? So, good memory read from coalescing"。现在把矩阵尺寸只挪动 $1$：第一行被顶过去，蓝色方块不再落在 burst 边界上，「there's no way for me to coalesce my reads to read a single tile in one go」，所有东西变得 "ragged and misaligned"，于是「I need to do at least two tiles worth of reads to read out this one tile」。也就是形状差 $1$，读一个 tile 的访存量至少翻倍，而计算量几乎没变——这正是上一章那两个「瘦 tile」在访存层面的对应代价。

解决办法是 padding（填充）：「what you have to do is padding in order to line up essentially the boundaries of the burst sections with the tile sizes that you have」。把矩阵尺寸向上补到 tile 或 burst 的整数倍，写成

$$
N_{\text{pad}}=\left\lceil \frac{N}{T}\right\rceil \times T
$$

让行边界重新与 burst section 对齐，tile 读取便恢复成合并的。代价是补出来的行列是「useless dimensions」，会做无用计算，但换来的是访存路径的整齐。这与前面「tile 尺寸要整除矩阵维度」其实是同一件事的两面：前者从 tile 数量与占用率角度看，后者从 burst 对齐角度看。讲者提醒，一旦理解到这一层，「Padding can start to shift the rows and columns so that you actually get very efficient reads, whereas at other times you do not」，很多看似神秘的现象就不神秘了。

紧接着就是他「one of my favorite weird ones」的例子：Andrej Karpathy 做 nanoGPT speed run 时的推文。幻灯片标题 "Putting it together: understanding a matrix mystery"，推文截图里 @karpathy 写道 "The most dramatic optimization to nanoGPT so far (~25% speedup) is to simply increase vocab size from 50257 to 50304 (nearest multiple of 64). This calculates added useless dimensions but goes down a different kernel path with much higher occupancy. Careful with your Powers of 2."，发布时间 10:36 AM · Feb 3, 2023，1.2M Views；下方提问 "Why is it *faster* to have bigger matrices?"，右下角注明 "This section is from https://www.thonking.ai/p/what-shapes-do-matrix-multiplications"。数字可以自己核对：$50304=64\times786$，而 $50257/64\approx785.3$，故 $\left\lceil 50257/64\right\rceil\times64=786\times64=50304$。明明多算了无用维度，却因为 padding 改变了行/列对齐、走上另一条 occupancy 更高的 kernel path，拿到约 $25\%$ 加速。讲者点明「it's kind of weird to have to pad in order to get a speed up」，但一旦理解了 burst 边界就顺理成章。口述里把这两个数念得含糊，幻灯片推文原文才给出 $50257\to50304$ 与 "nearest multiple of 64"，应以画面为准。

![Karpathy 关于把 vocab 从 50257 增大到 50304 获得约 25% 加速的推文](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0052_01-05-18.jpg)

随后两轮问答把经验法则讲清楚。学生问「有没有该采用的 tile 尺寸」，讲者纠正：不要把 tile 尺寸当作使用者该操心的东西，"tiles you should normally think of as like a lower-level systems thing, unless you're writing your own kernels"，真正该关心的是 the size of your matrices。学生接着问「是否要让矩阵尺寸能被 tile 整除」，讲者答 "generally powers of two, ideally divisible also by like 32"。所以经验法则是取二的幂、并尽量能被 $32$ 整除；但推文结尾的 "Careful with your Powers of 2" 提醒：盲目迷信「二的幂」并不足够，关键是能否填满 burst window。本章把「形状差 1」的代价量化到至少 $2\times$ 访存与约 $25\%$ 吞吐反转，下一章就用 roofline、整除性着色和 SM 数量这三件零件去拆那条神秘曲线。

## 矩阵之谜：roofline、对齐着色与 wave quantization [66:15 - 69:20]

讲者宣布「we have essentially all of the pieces that we need to understand this」，开始逐条拆解方阵 matmul 的吞吐曲线，也就是幻灯片 "Putting it together: understanding a matrix mystery" 里那个 question 要回答的东西。第一件零件是前面讲过的 compute intensity 与 roofline plot：必须让每次内存读对应足够多的计算，才能把计算单元喂饱，这解释了曲线整体向上倾斜的对角线。写成

$$
\text{Arithmetic Intensity}=\frac{\text{FLOPs}}{\text{Bytes}}
$$

低算术强度时是内存受限（memory-bound），吞吐随工作量增加而上升；强度足够后进入计算受限（compute-bound）平台区，继续加工作量不再提升吞吐。讲者说「that one is done」，意思是这条对角线已经由 roofline 解释完毕，剩下的是另外两个更反直觉的现象。

第二件是按整除性着色。把每条线按矩阵维度的可整除性上色：蓝色只被 $1$ 整除（"They're not even even"），橙色被 $2$ 整除，绿色被 $8$ 整除，红色是 $16$，紫色是 $32$。幻灯片 "Part 1: tiling" 里那张图标题为 "FLOPs achieved for square matmuls (color coded by whether a shape is divisible by K)"，图例依次是 $K=2,\,8,\,16,\,32,\,128$，纵轴 TF/s（$0$ 到 $250$），横轴 "N: NxN @ NxN matmul"（刻度 $0,512,1024,1536,2048,2560,3072,3584,4096$）。结论是：只能被 $1$ 或 $2$ 整除的吞吐很差，而到 $16$、$32$ 就「perform equally well」，不再有额外惩罚。讲者特意澄清「it's not because of like somehow powers of two are magic or something like that」——不是二的幂有魔力，而是 $16$ 与 $32$ 已经大到「has the right sort of burst window property」，能把整个 burst window 装进去，从而对整个 tile 做 coalesced reads，「has a major impact on alignment」。幻灯片右侧手绘对比 "Aligned Layout" 下写着 "One Nice Tile :)"，"Unaligned Layout" 下写着 "Two Bad Tiles :("，底部结论句是 "Tiling has a major impact through alignment."。这些图例数字、手绘标注与结论句都是画面独有信息。

第三件是周期性下凹。剩下那条橙线会出现「weird periodic behavior」，"Why does adding a single extra dimension suddenly kill my throughput?"。讲者说 Percy 管这类现象叫 "GPU trivia"，「I think he hates but I love this one」。落差发生在 $1792\to1793$。用 tile size $256\times128$ 计算，幻灯片给出完整算式：

$$
\frac{1792}{256}\times\frac{1792}{128}=7\times14=98
$$

个 tiles；把维度加到 $1793$ 后「I've increased both tile sizes by one」，变成

$$
8\times15=120
$$

个 tiles。$98\to120$ 只增加约 $22\%$，但曲线跌幅是「gigantic」。真正原因在硬件："An A100 has 108 SMs, so it cannot execute all 120"——$98$ 个 tile 能一次装进 $108$ 个 SM，$120$ 个不能。跑完 $108$ 个 SM 后还剩 $12$ 个 tile，必须「wait a whole 'nother cycle」，而那一轮里「most of your GPU sitting idle」。写成波数 $\lceil 120/108\rceil=2$ 波，第二波占用率极低。讲者把这个现象命名为 wave quantization（波量化），并说「by bumping up your tile size by one, suddenly you need to wait a whole 'nother cycle」。幻灯片左侧还配了局部曲线，横轴标着 $1536$ 与 $2048$，能看到橙线锯齿状跌落，右上角问句 "What's with the periodic behavior?"。

![wave quantization 幻灯片：1792→1793、256×128 tile、98 与 120 tiles、A100 的 108 SMs](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0054_01-08-06.jpg)

三件零件凑齐后，曲线就完全可解释：roofline 决定整体斜率，整除性/对齐决定不同颜色线之间的差距，SM 数量与 tile 数量的整除关系决定周期性下凹。易错点在于把 $1792\to1793$ 的暴跌归因于「多算了 $1\times$ 计算」或「二的幂迷信」——真正触发条件是 tile 数从 $98$ 跨过 $108$ 这条 SM 边界，属于「刚刚好多出一波」的离散效应；同理 $16$ 与 $32$ 表现一样好，也说明继续追求更大的整除数（例如图例里的 $128$）在这里不会带来额外收益。本章的 $98$、$120$、$108$、$256\times128$ 这些参数，连同「内存是瓶颈」这一主线，正是下一章 Part 2 收束的直接依据。

## Part 2 收束：一切围绕内存与硬件问答 [69:20 - 71:50]

「That was all for for part two.」讲者用四个 memory 概括整部分："a lot of the story is really just about memory memory memory memory"。逻辑链是：先减少必须做的内存访问——「Let's coalesce our memory reads together. Let's fuse our operations together so we don't have to read as much」；对那些无论如何都要读的数据，就复用它——「shared memory uh move things to shared memory and reuse it as much as possible, which is tiling」；最后还可以用别的资源去换内存，「that's quantization and recomputation」。幻灯片 "Recap of part 2: making ML workloads go fast" 把这条链写成三组：Reduce memory accesses → Coalescing、Fusion；Move memory to shared memory → Tiling；Trade memory for compute/accuracy → Quantization、Recomputation。讲者收口说「all these ideas um pretty much relate to this idea of like memory is our bottleneck, how do we deal with that」。注意六项手段里 control divergence（控制发散）是唯一不靠省内存的，其余五项都落在「减少/复用/置换内存访问」上，这是回顾时最容易混淆的地方。

![Part 2 收束幻灯片：减少访存、搬入共享内存、用计算或精度换内存](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0055_01-09-38.jpg)

这张幻灯片右侧还叠了三幅配图，含不少口述没念到的细节。最上面是 "Coalesced Loads" 示意：warp 的 $T_0,T_1,T_2,T_3$ 同时发出 load，下面 $0\sim15$ 的格子按颜色分成四段 Burst section，图注 "When all threads of a warp execute a load instruction, if all accessed locations fall into the same burst section, only one DRAM request will be made and the access is fully coalesced."；中间是那张 Tiling math 网格图（Matrix A、B、C，tile size $T$、matrix size $N$，图例 Outer loop over tiles / Inner loop over elements / Temporary result tile）；最下面是重计算示意：左栏 $x\to\text{sigmoid}\to\text{sigmoid}\to\text{sigmoid}\to out$，标 "New Fwd pass"，写 "1 mem read / 1 mem write"；右栏是同样三条 sigmoid 加一个 "Original Backward graph" 框（输入 $dx$、$dout$），标 "New Bwd pass"，写 "2 mem reads / 1 mem write"。数字说明重计算把一次前向压到 $1$ 读 $1$ 写，代价是反向多付一次读。把贯穿全部分的两个判据再列一次：

$$
\text{global reads per input}=\frac{N}{T},\qquad \text{Arithmetic Intensity}=\frac{\text{FLOPs}}{\text{Bytes}}
$$

前者管「读几次」，后者管「每次读做多少活」，二者共同决定程序落在 roofline 的哪一段。

随后进入问答，先谈 SRAM 为什么不能无限做大。学生问「bump up the SRAM is very expensive」，讲者给了三条理由：第一 "it's just expensive to build SRAM"；第二 "very difficult cuz it has to be physically close, like, you know, signal propagation is hard"，物理上必须贴近计算单元、信号传播困难；第三 "SRAM and DRAM are just kind of different storage mechanisms. Like SRAM needs to be powered the whole time to hold value, very energy hungry"，SRAM 需持续供电才能保持比特，能耗高，而 DRAM 是另一套存储机制。因此从 total cost of ownership 与 energy consumption 看，「maybe you don't want a giant SRAM chip. You really want to be using it where you need it」。这也回头解释了第一章里那个「全 SRAM 加速器」为何只是昂贵的思想实验。

其余问答同样在给边界条件。学生问「整除到 32 之后再往上（例如 64）还有没有好处」，讲者答「No」，因为这里本质上「as long as it divides by your birth size then you're good to go, right? It's a divisor property」——只要能被 burst size 整除就够了，别处或许有更大除数带来收益的现象，"but not not here"。问「为什么是 108」，讲者笑答 "That's just the number of SMs. You will have to ask Jensen for that. I do not have an answer for you"，即 $108$ 是具体芯片的 SM 数、不是可推导的常数。最后有人问 GPU 与 single / wafer scale engine 的对比，讲者说 wafer scale 的编译极难，「you have like wave interference effects and all sorts of horrible things that happen」，并坦言那是自己不熟悉的领域、比这里复杂得多。本章把 Part 2 收束成一句「内存是瓶颈」，下一章进入讲者口中的 "the finale"，用这些零件拼出 FlashAttention。

## 终章案例：FlashAttention 的 tiling、online softmax 与重计算 [71:50 - 77:16]

「in some sense the finale」——把前面所有 GPU 知识串起来，去理解 flash attention。讲者称它是「one of the big improvements in attention capabilities in a while and it's all systems」：从 PyTorch 的朴素 attention 实现，变成一个 "single very cleverly fused kernel"，原论文里 serving latency 有 "dramatic improvement"；两讲前也讲过它是 memory efficient 的算法，能做更大的 attention，收益主要来自「improvements in the amount of memory that is transferred from HBM」。"Part 3: Using what we know to understand Flash Attention" 幻灯片的副标题是 "Flash attention [Dao et al] dramatically accelerates attention.. But how?"：左侧 "Attention on GPT-2" 柱状图纵轴 Time (ms) $0\sim15$，PyTorch 一侧被拆成 Matmul、Dropout、Softmax、Mask、Matmul 五段，FlashAttention 一侧只剩一个 Fused Kernel；中间表格给出 Standard Attention 与 FlashAttention 三行数字：GFLOPs $66.6$ / $75.2$、HBM R/W (GB) $40.3$ / $4.4$、Runtime (ms) $41.7$ / $7.3$；右侧 "Effect of Block Size" 图横轴 Block Size $64,128,256,512$，纵轴 HBM Accesses (GB) 与 Runtime (ms)，画有 HBM、Runtime、Kernel 三条线；底部 "Technique from paper: We apply two established techniques (tiling, recomputation) to overcome the technical challenge of computing exact attention in sub-quadratic HBM accesses. We describe this in Algorithm 1."。这组数字很关键：FlashAttention 的 GFLOPs 反而从 $66.6$ 升到 $75.2$（重计算多算了），但 HBM 读写从 $40.3$ GB 降到 $4.4$ GB、runtime 从 $41.7$ ms 降到 $7.3$ ms——再次印证收益来自内存流量而非算力。

讲者先复述 attention 的算子构成：有 $K,Q,V$，先用 $K$ 与 $Q$ 做矩阵乘，中间一个 softmax，最后与 $V$ 再做一次矩阵乘，「Three matrix multiplies and one softmax」。论文 Figure 1 就只是朴素的 tiled matrix multiply：把 $K$、$Q$ 切块、逐 chunk 在 tile 上相乘、算完写出，"Nothing fancy or special here"。难点只有一个：中间那个 "big global softmax on the whole thing"，它是全局操作，「It connects all of the different tiles together」，所以「It might not seem obvious how to tile uh attention」，讲者还说「you'll see this in the assignment」。这与前面几章形成鲜明对照：matmul 天然可分块，softmax 不行，除非换一种算法。

关键技巧（"once you see this, the rest is very simple"）是把 softmax 改成在线（online）计算。"Tiling part 2: incremental computation of the softmax" 幻灯片标注 "From Mikailov and Gimelshein 2018,"，左栏 "Normal softmax" 给的是带数值稳定性的安全版本，即式 (2)：

$$
y_i=\frac{e^{x_i-\max_{k=1}^{V}x_k}}{\sum_{j=1}^{V}e^{x_j-\max_{k=1}^{V}x_k}}
$$

下面注明 "All major DL frameworks are using this safe version for the Softmax computation: TensorFlow"，并给出 Algorithm 2 Safe softmax 原文：

```text
1: m_0 ← -∞
2: for k ← 1, V do
3:   m_k ← max(m_{k-1}, x_k)
4: end for
5: d_0 ← 0
6: for j ← 1, V do
7:   d_j ← d_{j-1} + e^{x_j - m_V}
8: end for
9: for i ← 1, V do
10:  y_i ← e^{x_i - m_V} / d_V
11: end for
```

它需要三遍扫描：先求全局最大值 $m_V$，再累加分母 $d_V$，最后逐个归一化。右栏 "Online softmax" 的 Algorithm 3 Safe softmax with online normalizer calculation 把三遍压成一遍：

```text
1: m_0 ← -∞
2: d_0 ← 0
3: for j ← 1, V do
4:   m_j ← max(m_{j-1}, x_j)
5:   d_j ← d_{j-1} × e^{m_{j-1} - m_j} + e^{x_j - m_j}
6: end for
7: for i ← 1, V do
8:   y_i ← e^{x_i - m_V} / d_V
9: end for
```

核心是第 5 行：每当遇到更大的数，最大值从 $m_{j-1}$ 换成 $m_j$，同时用 $e^{m_{j-1}-m_j}$ 把此前累积的 $d_{j-1}$ 整体缩放回来。这正是幻灯片底部 "To keep track of the max, incrementally update the max, and set up a telescoping sum" 与加粗结论 "This lets you compute the softmax tile-by-tile" 的含义。讲者的口述版本是：一路维护 normalized softmax，遇到更大的数就「swap out the maximum」并「correct for it every time we see a new maximum」，其余部分只是「a online running sum of an exponential」，最后「divide everything by the accumulator that you've computed online」。

![Normal softmax 与 Online softmax 两段算法原文及 telescoping sum 说明](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0058_01-14-33.jpg)

因为 online softmax 是 online、block by block 的，「we can compute this tile by tile」——不必先看到后面的 tile 就能算出当前 tile 的 softmax，把部分结果存下（也许写回全局内存）再「continue on with my day」。这就把原本互相牵连的算子拆开了。配合讲者喜欢的 FlashAttention 2 可视化（"I think this was from flash attention two or maybe three. I think it's two."）：虚线框表示在 SRAM 里 tiled 完成，蓝色框表示在 HBM（global memory）里；$KQ$ 内积是 tiled matmul 且结果留在 SRAM，接着在 SRAM 里算 exponential 与 running softmax，移到下一个 tile 时只把 partial running sums 保留在 shared memory、registers 或别处——「This is a very small number of things to keep track of」。全部指数项与 normalizer 算完后，再以 tiled 形式与 $V$ 相乘，最后做一次除法。

"Putting it all together – the forward pass of flash attention" 幻灯片把这套流程写成了公式。蓝虚线框表示 Stored in HBM、橙虚线框表示 Computed in SRAM (not materialized in HBM)：$(K^{(1)})^T$、$(K^{(2)})^T$、$Q$、$V^{(1)}$、$V^{(2)}$ 存在 HBM；$S^{(1)}=Q(K^{(1)})^T$、$S^{(2)}=Q(K^{(2)})^T$、$A^{(1)}=\exp(S^{(1)})$、$A^{(2)}=\exp(S^{(2)})$ 在 SRAM 里算且不落回 HBM。下方红色公式给出 running normalizer：

$$
\ell^{(1)}=\sum_i \exp(S^{(1)})_i,\qquad \ell^{(2)}=\ell^{(1)}+\sum_i \exp(S^{(2)})_i
$$

Output 框内两行（箭头标注 "Rescaling to correct denominator"）是：

$$
O^{(1)}=\frac{A^{(1)}}{\ell^{(1)}}\cdot V^{(1)}
$$

$$
O^{(2)}=\frac{\ell^{(1)}}{\ell^{(2)}}O^{(1)}+\frac{A^{(2)}}{\ell^{(2)}}\cdot V^{(2)}
$$

第二行的 $\frac{\ell^{(1)}}{\ell^{(2)}}$ 就是把上一 tile 的输出按新分母重新缩放，正是 online softmax 里 $e^{m_{j-1}-m_j}$ 那一步在块级别的体现。下方 "From Dao 2023, we see" 列了三条：Tile-wise computation of the inner products, ($S$)；Fusion of the exponential operator；Tile-wise computation of the softmax via the online, telescoping sum trick；并补一句 "(We won't cover the backward pass – but they recompute tile-by-tile..)"。

![FlashAttention 前向：HBM/SRAM 分块、running normalizer 与输出重缩放公式](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0059_01-16-11.jpg)

最后一块是 backward pass：如果保存激活，仍要留下一个 $n^2$ 大小的 attention 激活；FlashAttention 的做法是「throw it all away and recompute everything tile by tile on the backward pass」。讲者明确说「I'm not going to cover this or talk about it」，只点出这就是 recomputation 那一块。于是 FlashAttention = tiled matmul + online softmax + fusion + 反向 recomputation，几乎把前面六项技巧全用上了，「That gives you flash attention」「Now, we've seen all of the different tricks in action」。易错点也在这里：attention 里的 softmax 不是逐元素独立算子，不能像 ReLU 那样直接 fuse，必须靠 online / telescoping sum 维护全局最大值与分母，且 max 的更新会连锁缩放此前所有累积量；此外反复强调收益来自 HBM 流量而非 FLOPs（表格里 GFLOPs 甚至上升），这与「算力增长快于带宽」的主线一致。本章是整讲的 finale，下一章只是把结论收拢到硬件、matmul 与数据移动三句话。

## 全课收束：理解硬件、matmul 与数据移动 [77:16 - 78:38]

讲者以「Now, we've seen all of the different tricks in action」「and that's essentially all there is」开场，说 "A lot of the tricks that you see for efficient implementation and how to use GPUs well boil down to those kinds of ideas"，然后 "To put it all back together"。第一句是 Important to understand the hardware：硬件（这里即 GPU）正在 powering scale and compute，必须一直理解到 low-level details 才能明白「why we make certain decisions」。他特别不希望学生变成「cargo cult, you know, 32 multipliers for your matrices」的人——也就是盲目照抄「矩阵尺寸取 32 的倍数」这类口诀却不懂原因；要「really understand, you know, why we do those things and that goes all the way down in my opinion to the hardware level」。这句话正好回扣前面 $50257\to50304$、$1792\to1793$ 两个例子：口诀本身没有魔力，性能来自 burst 对齐与 SM 数量这些低层事实。

第二句关于 matmul 与数据移动："the current way that compute is growing means that we really want to think about matmuls. Those are the core operations that are very arithmetically dense"，而「because of the gap between compute and memory, we need to think very carefully about data movement」。第三句把视野推到架构设计：flash attention 这类工作很酷，"There's also like architecture work that's like very hardware aware"，"Thinking carefully about the hardware when we do things like architecture design is, you know, critical to good performance for future systems"，最后以 "Thanks." 结束全讲。"Recap for the whole lecture" 幻灯片的三条 bullet 与之逐条对应："Hardware powers scale, and low-level details determine what scales or doesnt"（原文即写作 doesnt，无撇号）、"Current GPU based compute strongly encourages thinking about matmul + data movement"、"Thinking carefully about the GPU (coalescing, tiling, fusion) leads us to good performance"。

![全课收束幻灯片：三条总结 bullet 与右侧 matmul、带宽、内存层级图](/coursenotes/cs336-language-modeling-sp26/l05-gpus-and-tpus/kf_0060_01-18-13.jpg)

幻灯片右侧三幅图补上了口述没念到的数字。最上方 "Matmul vs. non-matmul FLOPS across GPUs"：纵轴 TFLOPS 用对数刻度 $10^1,10^2,10^3$，横轴 GPU 依次 K80、M80、P100、V100、A100、H100，图例 non-matmul（蓝）与 matmul（橙）；橙线自 V100 起陡升，到 H100 已接近 $10^3$，蓝线只到几十——这就是「matmul 是特权操作、算术密度最高」的量化证据。中间 "Scaling of Peak hardware FLOPS, and Memory/Interconnect Bandwidth"：横轴 YEAR $1996\sim2023$，三行标注逐字为 "HW FLOPS: 60000x / 20 yrs (3.0x/2yrs)"、"DRAM BW: 100x / 20 yrs (1.6x/2yrs)"、"Interconnect BW: 30x / 20 yrs (1.4x/2yrs)"，即 20 年间算力涨 $60000$ 倍（每两年 $3.0\times$），DRAM 带宽只涨 $100$ 倍（每两年 $1.6\times$），互连带宽只涨 $30$ 倍（每两年 $1.4\times$）；散点旁还标着 GDDR3、GDDR4、GDDR5、HBM、HBM2、HBM2E、PCIe 1.0a/2.0/3.0/5.0、NVLink 1.0/4.0、GTX 580、K40、TPU v3、TPU v4、A100、H100 等。把剪刀差写成比值：

$$
\frac{60000}{100}=600,\qquad \frac{60000}{30}=2000
$$

即算力相对内存、互连的差距分别拉到约 $600$ 倍与 $2000$ 倍，这正是全讲「内存是瓶颈」的根源，也解释了为什么 coalescing、tiling、fusion 会成为主角。

最下方是 "Memory Hierarchy with Bandwidth & Memory Size" 三角形加 FlashAttention 流程图。三角形三层逐字为 "SRAM: 19 TB/s (20 MB)"（GPU SRAM）、"HBM: 1.5 TB/s (40 GB)"（GPU HBM）、"DRAM: 12.8 GB/s (>1 TB)"（Main Memory / CPU DRAM）；相邻层级带宽差一个数量级以上，容量却反过来差三个数量级，所以才需要 tiling 把小块搬进 SRAM。右侧 FlashAttention 流程标着 $K^T: d\times N$、$Q: N\times d$、$V: N\times d$、$QK^T: N\times N$、$sm(QK^T)V: N\times d$，绿色长条表示矩阵，橙色小块写 "Copy Block to SRAM" 与 "Copy"，虚线框内写 "Compute Block on SRAM" 与 "Output to HBM"，红色箭头为 Outer Loop、蓝色箭头为 Inner Loop——外循环扫 $K/V$ 块、内循环扫 $Q$ 块，正是上一章分块策略的图示。易错点：这张 recap 把 coalescing、tiling、fusion 并列为「thinking carefully about the GPU」的三件事，但不要误以为它们能替代对硬件的理解；讲者反复强调 32 的倍数、二的幂这类口诀本身没有魔力，性能来自对齐、burst window、SM 数量这些低层事实。而 $19\,\text{TB/s}$ 与 $12.8\,\text{GB/s}$ 的巨大落差也说明，未来做架构设计时把数据移动放在与 FLOPs 同等重要的位置，才是「good performance for future systems」的前提。

## 要点回顾

### 本讲定位与延迟/吞吐的哲学分野
- 课程从神经网络部分转入 systems，主目标是 `Make CUDA and GPUs less magic`，并兑现"看懂矩阵乘吞吐随维度变化那条神秘曲线"的承诺；全程分三部分：GPU 硬件与编程心智模型、让负载跑快的若干 tricks、最后 walk through flash attention 的 "victory lap"。
- 讲者强调系统部分可拆成一条条逻辑步骤推理、最 "reasonable"，但初见 GPU 却"既像魔法又像跟奇怪设备交互"；现场约四分之一（可能略少）学生上过 GPU programming class。
- CPU 为 quick serial execution 设计，配大控制单元、少量 ALU、擅长复杂分支，追求低延迟；GPU 用成百上千轻量核心追求聚合吞吐，幻灯片写作 `CPUs optimize for latency` 对 `GPUs optimize for throughput`。
- 写成目标函数即 CPU 取 $\min L_{\text{per-thread}}$，GPU 取 $\max \frac{N_{\text{tasks}}\times \text{work per task}}{T_{\text{total}}}$；两种芯片把晶体管预算分别花在 control/cache 与 ALU 阵列上。

### 算力货币、扩展失速与内存瓶颈
- 算力是整门课的通用货币，Kaplan 缩放律拟合为 $L = 2.57\cdot C^{-0.048}$，底部一句 `compute leads to predictable performance gains... for now..` 留了余地。
- Dennard scaling 在 2000 年代见顶（频率撞墙、功耗上升、逻辑核数增加而晶体管仍单调上升），于是转向 parallel scaling；"十年千倍"拆成 Number Representation 16x、Complex Instructions 12.5x、Process 2.5x、Sparsity ~2x 连乘约 1000x，另有 `Model efficiency has also improved`。
- `Compute scaling is faster than memory scaling`：HW FLOPS 60000×/20yrs（3.0×/2yrs）、DRAM BW 100×（1.6×/2yrs）、Interconnect BW 30×（1.4×/2yrs），相对差距达 600 倍与 2000 倍，结论是 `hard to keep our compute units fed with data`。
- V100 起 Tensor Cores 让 matmul 成为特权操作，`matmul throughput > 10× non-matmul throughput`（H100 橙线近 $10^3$ TFLOP/S、蓝线只到几十），故随算力扩展的架构几乎必含矩阵乘；内存三角形为 SRAM 19 TB/s (20 MB)、HBM 1.5 TB/s (40 GB)、DRAM 12.8 GB/s (>1 TB)。

### GPU 硬件解剖与执行/内存模型
- A100/GA100 有 128 个 SM（规格含 8 GPC、64 TPC、8192 FP32 Units、512 Tensor Units、48MB L2、6144-Bit HBM2e、836 mm²），SM 内含多个 SP，每个 SM 独立编程并访问 global memory。
- A100 延迟表：Global memory 290、L2 200、L1 33、Shared Memory ld/st 23/19 cycles，$290/33\approx 8.8$ 对应口述"约 10 倍"；SRAM 贵 100x 却快约 8x，shared memory 可编程而 L1 由硬件自动管理，H200 的 144 GB 指 global memory。
- 执行模型三 players：thread 遵循 SIMT（warp 内 32 个连续编号线程执行相同指令但输入不同，锁步是 warp 级）、block 保证跑在单个 SM 并共享 shared memory、warp 是调度单位；`warp = 32 个连续编号线程`、`block ↦ 1 SM`。
- 内存模型里 device code 可读写 per-thread registers、local、per-block shared、per-grid global 与只读 constant，host 与 global/constant 双向传输；跨 block 信息必须经 global memory (slow)，"一旦走出 shared memory 就变慢"。

### TPU 与 GPU 的趋同演化
- 二者是 convergent evolution，共性压成 `Core structure = lightweight control + fast (big) matmul unit + fast memory`；TPU control 更轻、MXU 更大，VPU 做逐元素操作并装数据，HBM bandwidth 决定数据进出计算单元的快慢。
- 命名陷阱：TPU 把 SM（处理器）叫 tensor core，GPU 把 matrix multiply unit 叫 tensor core，只能靠上下文消歧；两者底层都是 systolic array（讲者明确说不知典型尺寸，未给数）。
- H100 对 TPU v5p 的数量快照：SM 132 / Tensor Core 2、Warp Scheduler 528 / VPU slots 8、SMEM 32MB / VMEM 128MB、Registers 32MB / VRegs 256kB、Tensor Core 528 / MXU 8；TPU 用更少更大的单元，锁死在大矩阵乘。
- TPU `no warps (just blocks)`，大单元要求 $\dim(\text{input}) \ge 64$（batch size sweep 到 64 就停）；讲者点明真正最大的差异来自 networking，本讲不展开。

### 六个手段之一：roofline、控制发散与低精度
- roofline 给出 $\text{Throughput} = \min(\text{Peak FLOPs},\ \text{Memory BW}\times\text{Operational Intensity})$，$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes}}$，目标是待在 flat 的 compute-bound 段；六 tricks 为控制发散、低精度、算子融合、重计算、合并访存、tiling，仅第一个不是内存技巧。
- 控制发散（not a memory issue）：warp 内 `if/else` 两分支都要按时钟各走一遍、非当前分支被 mask 空转，一次分支约多等两个 clock cycles；ReLU 宜用 $x\cdot\mathbf{1}[x>0]$ 掩码乘法代替 $\max(0,x)$ 分支。
- 低精度把字节减半：elementwise ReLU 在 FP32 是 8 bytes/FLOP、FP16 是 4 bytes/FLOP（讲者当场纠正这是算术强度的倒数，降精度实际把算术强度提高 2 倍）；混合精度 matmul 为 16-bit 输入、全精度乘积、`Sum with FP32 accumulator`、输出 FP32。
- FP8 无统一格式，分裂为 E4M3 (1+4+3) 与 E5M2 (1+5+2)；MXFP8（Blackwell）每 32 元素配一个 E8M0 缩放因子（$s_k = 2^{e_k}$），代价是 `Transposes are now nontrivial!`，训练时需为量化矩阵同时保存原版与转置版。

### 六个手段之二：融合、重计算、合并访存与 tiling
- 算子融合把 `sin²x + cos²x` 的 5 个 pointwise 算子由 TorchInductor/torch.compile 合成单个 CUDA kernel，HBM 往返从 $N_{\text{ops}}$ 降到 2（1 read + 1 write），不减 FLOPs 只减中间读写。
- 重计算以 `x→sigmoid→sigmoid→sigmoid→out` 为例：旧方案前向 3 写、反向 3 读合计 8 次访存，丢弃激活反向重算降到 5 次（$\frac{5}{8}$），即用便宜算力换激活存储（activation checkpointing）。
- 合并访存源于 DRAM burst：16-byte 地址空间切 4-byte burst sections（实战 128 bytes+），coalesced 当且仅当 warp 内所有线程地址落在同一 burst 才只发 1 次 DRAM request；行主序跨行读列不合并，用 $N_{\text{pad}}=\lceil N/T\rceil\times T$ 补齐。
- tiling 是 "the big one"：按 $P_{i,j}=\sum_k M_{i,k}N_{k,j}$ 把 tile 搬进 shared memory 反复复用，$\text{global reads per input}=\frac{N}{T}$、tile 内读 $T$ 次，实现 $T$ 倍削减；受 shared memory 容量、coalescing、矩阵整除性共同约束。
- 工程上 `max-autotune` 用约 15 分钟编译期穷举 tile 尺寸换 nontrivial benefits；Karpathy 把 vocab 从 50257 补到 50304（$\lceil 50257/64\rceil\times64$、64 的倍数）反拿到约 25% 加速，经验法则是取二的幂并尽量能被 32 整除。

### 曲线之谜与 FlashAttention 终章
- 三零件解释吞吐曲线：roofline 定对角斜率；按整除性着色（图例 K=2,8,16,32,128，能被 16、32 整除表现一样好）；wave quantization 解释周期性下凹，与 `Tiling`、`Compute Intensity` 手写批注对应。
- wave quantization：tile $256\times128$ 下 $\frac{1792}{256}\times\frac{1792}{128}=7\times14=98$ 个 tiles，加到 1793 变 $8\times15=120$；A100 只有 108 SMs，120 需 $\lceil 120/108\rceil=2$ 波，第二波大量 SM 空转。
- Part 2 收束成 `memory memory memory memory`：Reduce memory accesses → Coalescing、Fusion；Move memory to shared memory → Tiling；Trade memory for compute/accuracy → Quantization、Recomputation。
- FlashAttention 表格里 GFLOPs 66.6→75.2 反升，HBM R/W 40.3→4.4 GB、Runtime 41.7→7.3 ms，印证收益来自内存流量；难点是全局 softmax 连接所有 tiles，靠 online softmax 把 safe softmax 三遍扫描压成一遍（$d_j = d_{j-1}\times e^{m_{j-1}-m_j}+e^{x_j-m_j}$）逐块计算。
- FlashAttention = tiled matmul + online softmax + fusion + 反向 tile-by-tile recomputation，输出重缩放 $O^{(2)}=\frac{\ell^{(1)}}{\ell^{(2)}}O^{(1)}+\frac{A^{(2)}}{\ell^{(2)}}\cdot V^{(2)}$；全课收束为 `Hardware powers scale`、思考 matmul + data movement、以及 coalescing/tiling/fusion 三件事。

## 视频来源

- 原始链接：https://www.youtube.com/watch?v=izZba4UA7iY
- 课程：Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 5

---

## 对照参考笔记补遗（Spring 2025 版）

> 编者注：本节依据 2025 版参考笔记整理，其中的「更正」小节记录的是核对时发现的问题，对应正文已在本版订正；补遗内容不代表 2026 课程视频原文。


本节内容来自 Spring 2025 版参考笔记，仅用于与 2026 版笔记对照补充，不代表 2026 课程视频原文；两版年份与讲授编排不同，以下按参考表述整理，凡与正文不一致处请以 2026 版正文为准。

### 从 kernel 下发到 warp 调度：执行模型的完整链条

2025 参考在讲解执行模型时，除了给出 thread、block、warp 三个 players 的定义，还把三者串成了一条动态的下发过程：`A CUDA program launches many blocks`，每个 block 被指派（assigned）到一个 SM 上；进入 SM 之后，这个 block 又被划分（divided）成若干个 warp；每个 warp 由 `Warp Scheduler` 调度，发射到该 SM 的 streaming processors（SP，流处理器）上运行，且 warp 内所有线程对各自的数据执行同一条指令。用式子可以把它写成两级映射：$\text{blocks}\longrightarrow\text{SM}$，以及 $\text{block}\longrightarrow \lceil N_{\text{threads}}/32\rceil\ \text{warps}$，其中 $warp=32$ 个连续编号线程。我们正文已经把 $block\mapsto 1\,SM$、$warp=32$、SIMT 锁步讲清楚了，补上这条链条的价值在于把静态定义变成一次 kernel 发射的真实时序：先有 grid 级的 block 分配，才有 SM 内的 warp 切分与指令广播。它与前文「block 绑在单个 SM 上才能用它的 shared memory」直接咬合——正因为分配发生在 block 这一级，同一 block 的线程才共享同一个快速内存池。

参考还补了两处容易被「GPU 控制单元很少」这句话掩盖的事实。其一是 SM 自身 `has control logic and can decide what to execute (e.g., branching)`，即 SM 仍带控制逻辑、能决定执行什么甚至处理 branching，SP 才是在它之下把同一条指令施加到多份数据上的部件（Single Instruction, Multiple Threads，SIMT）。其二是那张 detailed die 图相对前一页 cartoon 图的差别：绿色 SM 方块里除了控制单元，还画出了 `FP32`、`INT32`、`Tensor Cores` 几类专用处理单元，芯片边缘的黄色 `HBM connectors` 则连到 off-chip DRAM。把这两点合起来，才能把 CPU 与 GPU 的差异读成「控制与缓存预算的多少」而非「有无控制」：CPU 配 really big control units 与少量 ALU，GPU 把晶体管预算挪给 ALU 阵列，但每个 SM 仍留着够用的控制逻辑。

这里最值得记牢的易错点是层级不要压平。一个 block 会被切成多个 warp，所以「一个 block 就是一个 warp」是错的；SIMT 的锁步发生在 warp 级，而不是整个 block 或整个 grid 级。由此也能理解为什么跨 block 的信息交换必须绕道 global memory (slow)：block 是 SM 级绑定、warp 是 SM 内调度级，二者之间没有共享的快速通道，跨越这个边界只能走慢速 DRAM。把这个链条立起来之后，后面 wave quantization 里「tile 数跨过 SM 数就要多跑一波」的离散效应就有了着落——可分配的单元始终是 SM，而真正被发射的是 warp。

### Dennard scaling 的物理机制与并行范式的转向

关于算力为何从串行频率转向并行吞吐，2025 参考给出了比「频率撞墙」更完整的一条因果链。在 Dennard Scaling（登纳德缩放，1980s–2000s）时期，CPU 变快靠的是 Moore's Law 式的 `packing more transistors onto a chip`：晶体管做小之后，`could run at higher clock speeds with lower power`，也就是既抬高时钟频率又降低单个晶体管的功耗，二者共同抬升 single-thread performance。把这件事写成正比关系，大致是 $\text{single-thread perf}\ \propto\ f_{\text{clock}}\times \text{IPC}$；于是早期「让计算机变快」几乎等价于「让一串指令执行得越来越快」，延迟 $L_{\text{per-thread}}$ 是唯一的优化对象。

拐点出现在 2000 年代：`single-thread performance started to taper off`，图上那组 blue dots 开始走平，而晶体管数量仍按 Moore's Law 单调上升，却不再同比转成单线程吞吐。参考把由此产生的范式转变定义得很干净——从 `single-thread scaling (doing computation faster in absolute terms)` 转向 `parallel scaling (performing many workloads simultaneously)`，也就是不再追求单次计算在绝对意义上更快，而是让大量工作负载同时进行。Bill Dally 在 HotChips keynote 里那张 `1000X in 10 years` 曲线，2025 参考把它标作 `integer operations per second (INT8 TOPS)` 的 super-exponential（超指数）增长；2026 版口述则按 flops 讲，这属于同一张图的两版口径差异，引用时须注明版本，不宜混用。

这条因果链向前接住了 CPU 优化 latency、GPU 优化 throughput 的哲学分野，向后解释了那「十年千倍」的来源。参考把它拆成四项连乘：Number Representation 由 FP32 到 FP16、BF16、INT8 贡献 $16\times$，Complex Instructions（DP4A、HMMA、IMMA 一类专用指令）贡献 $12.5\times$，Process 从 28nm 到 5nm 只贡献 $2.5\times$，Sparsity 贡献约 $2\times$，再叠加 Model Efficiency 得到 overall gain $>1000\times$。易错点有两处：一是别把「晶体管继续变多」等同于「单线程继续变快」，这正是频率见顶后最反直觉的地方；二是别把 $1000\times$ 归因于制程，Process 一项只有 $2.5\times$，真正的大头是数值表示与专用指令这两项——它们也正是后面六个手段里「低精度计算」的历史伏笔。

### 低精度收益的两种口径与混合精度的输出去向

2025 参考把低精度的收益写成一句很直白的因果：`fewer bits mean fewer bits to move. This effectively increases memory bandwidth for free`，并在 elementwise ReLU 的例子后总结 `Halving memory access effectively doubles memory bandwidth`。例子的记账是：对长度为 $n$ 的向量做 $x=\max(0,x)$，FP32 下每个数 4 bytes、一次 read 加一次 write 共 8 bytes，却只有 1 个 comparison 与 1 FLOP，幻灯片记作 `Intensity: 8 bytes/FLOP`；换成 FP16 后每数 2 bytes、共 4 bytes，记作 `4 bytes/FLOP`。参考还给出 `16x for FP32 to INT8` 这一跨格式增益口径，与增益清单里的 Number Representation $16\times$ 相呼应。这套说法的落点是「等效带宽翻倍」，即从带宽一侧描述同一件事。

必须小心的是方向。roofline 横轴的 operational intensity 是 $\text{Arithmetic Intensity}=\text{FLOPs}/\text{Bytes}$，而幻灯片字面上的 bytes/FLOP 恰好是它的倒数：FP32 的 $8\ \text{bytes/FLOP}$ 对应 $1/8\ \text{FLOP/byte}$，FP16 的 $4\ \text{bytes/FLOP}$ 对应 $1/4\ \text{FLOP/byte}$，所以降精度是把算术强度提高 $2$ 倍，而不是减半。2026 版里讲者当场自我纠正了这一点（`Sorry, this is the inverse of arithmetic intensity`）；若把两种口径混用，会得出「低精度让算术强度减半、反而更 memory bound」的错误结论。这是全讲最易写反方向的一处，引用 2025 的「等效翻倍内存带宽」与 2026 的「算术强度提高 2 倍」时应理解为同一事实的两种表述，一个从带宽侧、一个从 roofline 横轴侧切入，始终以 $\text{FLOPs}/\text{Bytes}$ 的方向为准。

参考还补了一句我们正文没有写死的去向：混合精度 matmul 里 `inputs can be 16-bit, intermediate accumulations can be 32-bit for precision, and the final result can be downcast to 16-bit`，即 16-bit 输入、32-bit 部分和累加、输出可以再降回 16-bit（2026 版口述则说输出可以是 FP32，属讲法差异，不构成矛盾）。并不是所有环节都能降：`sums, reductions` 这类归约、`exponential functions` 这类高动态范围运算，以及 `loss functions` 仍需 FP32 或 BF16，否则会出现 rounding errors 或训练不稳定。与前后知识点的关系上，这正解释了为什么低精度被称作 black art——收益来自减少字节搬运，但只有 matmul 这种算术强度高、被 Tensor Core 特权招待的算子，才能把字节节省真正兑换成吞吐；它也和算子融合、tiling 并列，服务于同一个目标：压低 $\text{FLOPs}/\text{Bytes}$ 里的分母 Bytes。

### 重计算与检查点：目标在提速而非仅省容量

2025 参考在重计算一节末尾给出一句我们正文没有点明的限定：`Similar to gradient checkpointing, but here the goal is execution speed rather than just memory savings`。通常把 activation checkpointing（激活检查点）理解为「省显存容量、让大模型塞得进卡」，而本讲把它放在「让 workload 跑快」的六个手段里，落点是少做慢速全局内存读写、缩短 runtime，并非扩容。它成立的前提被写得很清楚：`compute is abundant and memory bandwidth is scarce`，也就是算力便宜到可以挥霍、内存访问昂贵到必须省，且计算单元没有被占满。丢掉的是中间激活 $S_1,S_2$，输出 $out$ 仍必须写回，因为它是整张计算图的输出。

参考的逐项访存记账与 2026 版完全一致，可用来核对同一笔账：旧方案 Normal Forward 为 `1 memory read (x), 3 memory writes (S1, S2, out)`，Normal Backward 为 `3 memory reads (S1, S2, dout), 1 memory write (dx)`，合计 $8$ 次；新方案 Recomputed Forward 只 `1 read (x), 1 write (out)`，$S_1$ 与 $S_2$ 不再保存，Recomputed Backward 用 `2 reads (x, dout)` 在片上重算三层 sigmoid，再 `1 write (dx)`，合计 $5$ 次，即 $\frac{5}{8}$ 的访存。数字吻合说明两版讲的是同一笔「用一次额外前向换掉激活存取」的交易，2025 参考额外贡献的是对交易目的的定位：它换出去的是便宜算力，买回来的是运行时间，而不只是显存空间。

把它与同组手段的分工分清，是这一节最容易混的地方：算子融合减少的是「算子之间往返全局内存的次数」，tiling 减少的是「同一数据被全局内存读取的次数」，重计算减少的是「为反向专门保存的中间结果」，三者都在改内存流量或 $\text{Arithmetic Intensity}=\text{FLOPs}/\text{Bytes}$ 的分母，但置换出去的资源不同。边界条件也很硬：它只对 sigmoid 这类算术强度极低的逐元素算子划算，若换成 matmul 这类高算术强度运算，多跑一次前向的代价会远超省下的那点访存；反向路径因此变长，框架还需要在计算图里自动决定哪些节点该重算、哪些该留。易错点归结为一句：别把重计算当成纯省显存技巧，也别以为实现上只是删掉几行保存逻辑——它是 systems 视角下「用一种资源换另一种资源」的典型，与前面所有围绕内存瓶颈展开的手段同属一条主线。
