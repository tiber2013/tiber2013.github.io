---
title: "第 8 讲 · 并行化 II：数据、流水线、张量与专家并行"
description: "大规模语言模型训练必须同时解决计算和显存两个瓶颈，因此并行策略不是单一选择，而是数据并行、ZeRO/FSDP、流水线并行、张量并行、序列并行、专家并行和上下文并行的组合。"
---

# 第 8 讲 · 并行化 II：数据、流水线、张量与专家并行

> **Stanford CS336 语言模型从零构建 · L08** ｜ 讲者 Stanford Online ｜ 80 分钟 ｜ [原视频](https://www.youtube.com/watch?v=6-cXp-aOmdg)

> **v2 详写版** · 按 20 分钟分段详写，配图覆盖全片；文末附「对照参考笔记补遗（Spring 2025 版）」，记录与 2025 版讲义笔记的核对结果。

## 核心结论

这一讲承接上一讲的并行底层机制，讲现代语言模型训练为何要叠加多种并行范式：算力与显存两大瓶颈、又无严格占优的单一方案，目标是让显存与算力随设备数线性扩展，并在节点内快链路、节点间慢链路和全局 batch size 之间分工。全讲以集合通信原语记账，用 all-reduce ≡ reduce-scatter + all-gather 说明 ZeRO 切分优化器状态与梯度几乎不增通信，FSDP 虽多一次 all-gather 仍可借通信计算重叠隐藏开销；但它不省激活显存、又消耗全局 batch size，越过临界批量收益递减，于是转向改传激活的模型并行。其中流水线只传 b×s×h 的点对点消息、适合慢链路却要用 micro-batch 压气泡，张量并行沿宽度切矩阵、通信密集只宜放在节点内，序列并行配合重计算把每层激活压到 34sbh/t 的下界，MoE 时代专家并行优先于张量并行、又需解耦 attention 与 MLP 的并行维度。按先让模型装进显存、再把剩余 GPU 全给数据并行、batch 不足用梯度累积的顺序组合就是 3D／4D 并行，roofline 用 B/N 与计算通信时间比划定高效区，真实案例与硬件故障数据说明，利用率取决于按链路速度、负载形态与系统集成做的权衡。

## 课程定位与并行训练的两大瓶颈 [00:00 - 03:05]

讲师开场先交代承接关系：上一讲由 Percy 讲了并行的底层机制（underlying mechanics of parallelism），本讲不再重复这些机制，而是集中讲「现代语言模型训练里各种可以知道的细节与冷知识」（knowledge details and trivia）。课程目标写在一张 *Outline and goals* 幻灯片上，三条原文分别是 Understand the systems complexities of training huge models（理解训练超大模型带来的系统复杂性）、Different parallelization paradigms and why people use multiple at once（不同的并行范式，以及为什么人们要同时用好几种）、What large scale training runs often look like（大规模训练运行实际长什么样）。这里最关键的预告是：到课程末尾会出现一张写着 4D parallelism（四维并行）的幻灯片，因为「有四件不同的事可以同时做」，而且讲师补了一句 actually even more——实际可叠加的维度比四个还多。他强调，要在最大规模上做并行，就得把其中全部或大部分策略一起用上，这也是后续所有章节的总纲；作业则是给定特定的网络拓扑（network topology）和特定的模型，找出最优的并行化策略，最后还会用已公开发表细节的近期大规模训练运行来做实例讲解。

为什么必须叠加？讲师把动机归结为两个瓶颈：算力（compute）与显存（memory）。算力一侧的说法很直白：单张芯片能提供的算力远远不够，必须把很多机器连起来；世界上最快的超算有 exaflops（百亿亿次浮点运算每秒）级别的算力，而单张 GPU 离这个数字差得很远。*Limits to GPU-based scaling – compute* 那张幻灯片用两张图把这件事量化了：左图是 Single-Chip Inference Performance 在十年里约 1000 倍的增益分解，右图 *Projected Performance Development* 的纵轴从 100 MFlop/s 一直标到 10 EFlop/s，横轴覆盖 1990 到 2025 年，绿色、橙色、蓝色三条曲线分别向上攀升；左图红色曲线沿途标注了 K20X、M60、P40、V100、T4、A100、H100 等代际，末段标着 Tensor Cores、Structured Sparsity、Transformer Trg 等节点。左列 Gains from 把增益拆成四项：Number Representation（数值表示）从 FP32、FP16 到 INT8，括号里还有 TF32、BF16，约 ~16x；Complex Instructions（复杂指令）列出 DP4A、HMMA、IMMA，约 ~12.5x；Process（制程）从 28nm、16nm、7nm 到 5nm，约 ~2.5x；Sparsity（稀疏性）约 ~2x。幻灯片还补了一句 Model efficiency has also improved – overall gain > 1000x（模型效率本身也在改善，总增益超过 1000 倍）。把四个因子相乘正好得到这个数量级：

$$ 16 \times 12.5 \times 2.5 \times 2 = 1000 \quad (\text{overall gain} > 1000\times) $$

幻灯片底部那句结论保留了原样拼写：There are limits to single-GPU scaling. The word's fastest supercomputers have *exaflops* of compute。显存一侧的动机则是模型太大、一张 GPU 装不下，所以必须用某种方式把模型切成更小的碎片（shard）。这两条动机共同决定了本讲后面所有算法要同时满足的两个目标，也决定了「多机多卡」必然带来跨机器、跨 GPU 的复杂性。紧接着讲师引入一个贯穿全讲的概念区分：intra-node parallelism（节点内并行）与 inter-node communication（节点间通信）。节点内连接非常快，因此可以承受通信代价很高的操作；节点间链路较慢，就必须采用更尊重信道通信约束的做法。这个「快链路 / 慢链路」的二分后面会直接决定张量并行、专家并行适合放在节点内，而流水线并行适合跨节点。

在记账方式上，讲师明确划定了抽象层次：今天不会真的去发包（sending packets），也不会下到驱动或硬件层，而是停留在 collective communication primitives（集合通信原语）这一层。每当需要核算通信量，说法会是「这里要做一次 all reduce（全规约）」或「这里要做一次 all gather（全收集）」来 implement some of these operations。他同时提醒一个易错点：要把这些原语真正高效地实现出来，最终还是得 go all the way to the hardware level；本讲选择 algorithmic（算法层面）的讨论，是因为一旦理解了这个层面的概念，就能把它们迁移到任何硬件基底上。这也解释了为什么本讲先花时间回顾原语、再谈拓扑：原语是跨硬件通用的记账单位，而拓扑只是影响这些原语实际成本的实现细节。

因此这一段只是「很快过一遍」的基础回顾（The very first part I'll go through somewhat quickly），真正的算法内容要到 11:07 之后才开始。把这一段当作术语与动机的铺垫，比当作结论更准确：00:05 到 01:18 给的是目标与 4D 并行的预告，01:18 到 02:04 给的是两大瓶颈，02:06 到 02:36 给的是节点内 / 节点间的快慢之分，02:38 到 03:10 给的是「以集合通信为单位做算法记账」的方法论。这四层合起来，构成后面每一个并行策略都要回答的评价框架。

![单卡算力扩展极限：十年千倍增益的四项来源分解与 1990–2025 性能发展曲线](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0002_00-01-52.jpg)

## 集合通信原语与 all-reduce 等价关系 [03:05 - 04:20]

基础回顾里最要紧的一条，是讲师特意重新拎出来的恒等式：all reduce（全规约）与 reduce scatter（规约散射）加 all gather（全收集）在通信代价上等价。原话是 there's going to be one algorithm where on one hand you do a all reduce, and on the other hand you do two of these different steps；正因为两者「在某种意义上有相同的代价」，所以可以把右边那个两步算法「免费」（for free）地做出来。讲师说明这只是 a quick refresher，用来确认大家还记得 Percy 上一讲讲过的内容。之所以在本段就反复强调，是因为 18:23 之后推导 ZeRO Stage 1 与朴素数据并行通信量相同时，整个论证几乎完全押在这条等价关系上。

*But first.. Some basics about collective communication* 这张幻灯片把五个原语画成 rank 0 到 rank 3 四个进程之间的数据块流向，并给每个原语配了一行下标公式。蓝、红、绿、黄四种颜色分别代表 in0、in1、in2、in3 四份输入。All reduce 的图示是四个 rank 各持一块输入，操作后四个 rank 都拿到同一个 out，标注为 `out[i] = sum(inX[i])`；Broadcast 以 rank 2 为 root（图上明确写着 rank 2 (root)），把 in 复制到所有 rank，标注 `out[i] = in[i]`；Reduce 同样以 rank 2 为 root，四个 rank 的 in0..in3 汇总到根节点，标注 `out[i] = sum(inX[i])`。All Gather 的图示反过来：每个 rank 起初只持有一小片（rank 0 持蓝色 in0，rank 1 持红色 in1，rank 2 持绿色 in2，rank 3 持黄色 in3），操作后每个 rank 都叠齐四色，标注 `out[Y*count+i] = inY[i]`。Reduce Scatter 则是每个 rank 起初都持有完整的四色输入，操作后 rank 0 只留 out0、rank 1 只留 out1、rank 2 只留 out2、rank 3 只留 out3，标注 `out[Y][i] = sum(inX[Y*count+i])`。把这些原式原样抄在一起，就是本段反复使用的记账依据：

```text
All reduce:      out[i] = sum(inX[i])
Broadcast:       out[i] = in[i]
Reduce:          out[i] = sum(inX[i])
All Gather:      out[Y*count+i] = inY[i]
Reduce Scatter:  out[Y][i] = sum(inX[Y*count+i])
```

$$ \text{all-reduce} \;\equiv\; \text{reduce-scatter} + \text{all-gather} $$

从数据布局上看，这五个原语的差别比公式看起来更大，也是最容易混淆的地方。All reduce 与 Reduce 都做求和，但前者把结果发给所有 rank，后者只发给一个 root；All Gather 与 Reduce Scatter 是一对互逆的动作，前者把每人身上一片拼成每人都有的全量，后者把每人都有的全量按片切开、每人只留一片。正因为 reduce-scatter 的输出是「按 rank 分片」的，它恰好对应「每个 worker 只负责更新参数空间里属于自己的那一片」——这正是 ZeRO Stage 1 需要的语义。相比之下，单纯做 all reduce 会让每张卡都拿到完整梯度，于是每张卡也就不得不存下完整梯度，显存省不下来。

课堂上有学生当场追问：为什么偏偏要讲「梯度做 reduce scatter」这一种分解，而不是把梯度直接 all reduce？讲师的回答很坦率，也点出了一个常被忽略的局限：there are other decompositions（还存在别的分解方式），这一条之所以被特别强调，仅仅是因为 This one in particular will be useful for AI algorithm（这一条对后面要讲的 AI 算法特别有用）。也就是说，恒等式给出的是代价上的等价，而不是「唯一正确的分解」；把它当成通用最优解是误读。另有一层局限来自抽象层次本身：本段所有通信量核算都停留在集合通信层面，讲师在 03:00 附近明确说，要真正高效实现这些原语还得下到硬件层。因此后面凡出现「通信量相同」的结论，都应理解为在算法记账意义上成立，实际耗时还会受拓扑、带宽与实现影响。

这一段还顺带交代了课程的方法论边界：即将展开的算法是 hardware and substrate agnostic（与硬件、基底无关）的，一旦理解基本概念，就能套用到任何硬件上；本讲主体（most of our discussion today）也是如此。但讲师紧接着又说，还是想先花一点篇幅讲 low-level hardware stack，因为它「有意思」，而且能解释不同模型训练公司为什么会走向不同的并行化策略。这句话正好构成本段与下一段的接缝：抽象原语讲完，马上用 TPU 与 GPU 的真实网络拓扑来给这些原语填上具体的成本数字。

![集合通信五原语：All reduce、Broadcast、Reduce、All Gather、Reduce Scatter 的数据块流向与下标公式](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0003_00-03-07.jpg)

## 硬件底座：TPU 环形网格与 GPU 胖树 [04:20 - 08:43]

进入硬件部分前，讲师先把范围说清楚：本讲主体是 hardware agnostic 的，但 low-level hardware stack 值得单独讲，理由是它 leads to different sort of parallelization strategies across different model training companies（导致不同训练公司走向不同的并行策略）。第一个对比对象是 TPU——Google 的加速器，讲师说它「有点像轻量版 GPU」（a TPU is kind of like a lightweight GPU），真正不同的地方是 networking。TPU 网络叫 toroidal mesh（环形网格 / 环面网格）：可以想成一张网格，再允许网格的两端彼此通信，因此所有芯片主要与邻居相连，且邻居在边界处 wrap around（回绕）。讲师提到真实结构更复杂、是 3D 的，并让 Marcel 把一个 3D 可视化的网页链接发到课程频道，但随即说心智模型用二维网格这张图就够了（You can forget the more complex one）。

这种拓扑最要紧的性质是可无限扩展而每个节点的邻居数不随规模增长。用度数来写，就是

$$ \deg(\text{chip}) = O(1)\ \text{，与集群芯片总数}\ N\ \text{无关} $$

讲师的原话是 the number of neighbors you have is kind of the ... no matter how large your network gets，因此拓扑可以 indefinitely scaled up very simply（一直简单地扩下去）。GPU 的联网哲学几乎相反：它是 all-to-all（全互联）取向，结构像一棵树，常被称为 fat tree（胖树）。最底层的 GPU 之间连得极快，其上把成组 GPU 连成 pod，pod 之间再靠 spine switches（脊交换机）互通。代价也写在同一页：as the number of nodes grows, the tree gets bigger and bigger，通信成本会上升、拓扑会更复杂，因为你在逐层把越来越多的东西全互联起来。*TPUs vs GPUs – design differences at the comm level* 这张幻灯片把两种哲学并列画出：左侧是 4×4 的 Chip 网格加红色回绕箭头，右侧两张小图分别是 DGX A100 256 SuperPOD（标着 IB HDR spine switches、IB HDR leaf switches、… 32 nodes (256 GPUs) …）与 DGX H100 256 SuperPOD（标着 NVS、NVLink Switch、Fully NVLink-connected、Massive bisection bandwidth、… 32 nodes (256 GPUs) …），GPU 一侧的 bullet 直接写 All-to-all (up to 256)。

绿色表格给出了可核对的具体数字，是这一页最容易被转述漏掉的信息。表头分三组：A100 SuperPod、H100 SuperPod 与 Speedup，每组下的列是 Dense PFLOP/s、Bisection [GB/s]、Reduce [GB/s]，Speedup 组下再分 Bisection 与 Reduce。逐行读下来是：

```text
1 DGX / 8 GPUs      A100: 2.5, 2,400, 150 | H100: 16, 3,600, 450 | Speedup: 1.5x, 3x
32 DGXs / 256 GPUs  A100: 80, 6,400, 100 | H100: 512, 57,600, 450 | Speedup: 9x, 4.5x
```

也就是说，从 8 卡单机扩到 256 卡（32 台 DGX）时，A100 的 bisection 带宽从 2,400 涨到 6,400 GB/s，而 H100 从 3,600 涨到 57,600 GB/s；H100 SuperPod 相对 A100 在 256 卡规模的 bisection 上快 9x、在 reduce 上快 4.5x，Dense 算力则从 2.5 / 16 PFLOP/s 涨到 80 / 512 PFLOP/s。这些数字把「节点内极快、跨层逐级变贵」的定性说法落到了实测量级上，也解释了为什么并行策略要按链路速度分层摆放。

由拓扑差异可以推出工作负载的适配性，这也是讲师说「Bill Dally 和 Jeff Dean 讨论过各自擅长什么」时想表达的。若通信只发生在邻居之间，TPU 很划算（cost-effective），在同样功耗下可以把单条链路做得更 beefy；若连接方式是随机、stochastic、不可预测的，GPU 的全互联更灵活。具体到模型：GPUs are good for things like mixture of experts（GPU 适合混合专家模型），因为 token 要被路由到不同专家，通信模式不清晰；TPUs are great if you have this dense model that you're doing very predictable partitions of（TPU 适合稠密模型上那种高度可预测的切分），讲师点名这就是后面要讲的 tensor parallelism（张量并行）。06:53 那一帧还叠了一层淡色标注，能辨认出的有 Why mesh? Why tree? 与 Pro mesh – Cheaper，正好概括了「可预测通信用便宜的 mesh」这一取舍。这里的易错点是把拓扑差异说成「TPU 一定比 GPU 快」：实际上它只在规则、局部的通信模式下占优，一旦换成稀疏路由就未必。

本段最有戏剧性的转折是讲师临时改了讲稿：他说原本打算讲到这里就收住（where I was going to leave it for the lecture），结果当天早上 Google 公布了新一代 TPU，他看完发现 that's a tree topology、they've switched to a much more all-to-all connection。*But then things change.. TPU8i/t* 这张幻灯片把新结构画得很细：左侧写 Fully-connected 4 TPUs on each board (rectangle)、Fully connected 8 boards in each group (rack)、Fully-connected 36 groups in each pod (total 1,152 chips)；右侧是 Virgo Network 与 Jupiter Network 的层级框图，标有 2-layer switching、Optical Circuit Switch (Apollo)、Aggregation blocks、Accelerator racks、Data center building，最外层指向 Distributed Global WAN，三条编号说明分别是 2 layer switching, fully non-blocking、Resilient fabric with independent planes、Expandable to multi-datacenter sites。底部粗体结论是 TPU8i networking – closer to tree-style topologies (maybe for MoEs?) 与 TPU8t scale-out networking – 'Virgo' with switched networks。讲师的解释是：现代语言模型多是 MoE，推理时把 token 路由到不同专家会产生大量乱飞的带宽，此时通信是真实瓶颈，因此更想要 all-to-all；连训练芯片 TPU 8T 的跨机架（cross-rack）连接也开始更像 GPU，靠 Virgo 这一更高层网络栈提供 all-to-all 式交换。他把这总结为 TPUs 与 GPUs 之间的一场 convergence / evolution（趋同演化）：不是硬件决定负载，而是 workloads are defining the network that we need to have。这为下一章「用暴力互联弥补单芯片劣势」埋下对照，也与 03:05 那段「算法与硬件无关」形成有张力的补充：概念通用，但最优策略仍被拓扑与负载共同塑造。

![TPU 环形网格与 GPU 全互联胖树拓扑对比，及 A100/H100 SuperPOD 的算力与带宽数值表](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0004_00-04-23.jpg)

![TPU8i/t 的 Virgo 与 Jupiter 网络层级：1152 芯片 pod 与全互联交换结构](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0007_00-08-08.jpg)

## 暴力互联的代价与数据中心即计算单元 [08:43 - 11:05]

硬件部分的收尾落在「为什么不干脆把所有东西都连起来」这个问题上，讲师先拿上一讲的一个提问做类比：既然 SRAM 那么好，为什么不整颗芯片全用 SRAM？答案是 Groq，而且确实做得很好；同理，既然 all-to-all connectivity 这么好，为什么不干脆用极快的光纤芯片把一切连起来？他给出的现成答案是华为 Ascend 910（幻灯片写作 Huawei Ascend 910C Cloud Matrix 384）。讲师强调这张规格表「很有意思」：单看每颗芯片，它比 H200 差不少——做 matmul 更慢、很多方面都慢——但如果愿意 dedicate a giant rack of fiber optic switches，把远多于此数量的芯片连成一个大域，或许就能靠「连得更多」把单芯片的劣势补回来，也就是 in their case 384 in a big rack。

*Why not connect everything?* 这张幻灯片用一张两层规格表把这笔账算得很实，标题下的小标题是 Domain sizes – why not connect everything?，表格名是 Key Capabilities – Huawei Ascend 910C Cloud Matrix 384 vs Nvidia GB200 NVL72。Chip and Package Level（芯片与封装级）逐行对照 GB200、Ascend 910C 与 Huawei vs Nvidia 的比值：BF16 dense TFLOPS 为 2,500 对 780（0.3x），HBM capacity 为 192 GB 对 128 GB（0.7x），HBM bandwidth 为 8.0 对 3.2 TB/s（0.4x），Scale Up Bandwidth 为 7,200 对 2,800 Gb/s uni-di（0.4x），Scale Out Bandwidth 为 400 对 400 Gb/s uni-di（1.0x）。到 System Level（系统级）结论整个翻转：Nvidia GB200 NVL72 对 Cloud Matrix CM384 的 BF16 dense PFLOPS 是 180 对 300（1.7x），HBM capacity 13.8 对 49.2 TB（3.6x），HBM bandwidth 576 对 1,229 TB/s（2.1x），Scale Up Bandwidth 518,400 对 1,075,200 Gb/s uni-di（2.1x），Scale Up Domain Size 72 对 384 GPUs（5.3x），Scale Out Bandwidth 28,800 对 153,600 Gb/s uni-di（5.3x）。单芯片全面落后、系统级全面反超，正是「用规模换单芯片」这条思路的数字证据。

表格里最触目的一行是功耗，也正好对应讲师口中的「四倍」。All-in System Power 为 145,000 对 599,821 W（4.1x）；All-in Power per BF16 dense FLOP 为 0.81 对 2.00 W/TFLOP（2.5x）；All-in Power per memory bandwidth 为 251.7 对 488.1 W per TB/s（1.9x）；All-in Power per memory capacity 为 10.5 对 12.2 kW/TB（1.2x）。脚注写明 1. All-in System Power is total cluster power including scale-out networking, storage, etc.（系统总功耗包含 scale-out 网络、存储等）。把这两个数相除就是讲师那句 it's four times that of the equivalent Nvidia system 的来源：

$$ \frac{599{,}821\ \text{W}}{145{,}000\ \text{W}} \approx 4.1\times $$

于是权衡非常清楚：if you're willing to pay the power cost, you can solve a lot of kind of communication problems brute force and you can scale out much more aggressively to 300 chips，代价是功耗。讲师把它与 SRAM 的故事并列——想设计在功耗和制造上都高效的硬件，会落到一个位置；想 brute force，会落到完全不同的另一个位置。这页底部还给了出处 https://newsletter.semianalysis.com/p/huawei-ai-cloudmatrix-384-chinas-answer-to-nvidia-gb200-nvl72，右侧配了一张机柜照片。这里要提醒一个易错点：4.1x 是整系统口径而非单芯片口径，单芯片层面 Ascend 910C 反而是 0.3x；离开「系统级」这个前提谈「华为比英伟达费电四倍」或「华为算力更强」都会失真。

把硬件这几页合起来，讲师在 10:33 给出整个第一部分的收束：the new unit of compute is not the GPU, it's the entire data center（新的计算单元不是 GPU，而是整座数据中心）。*Part 1 recap* 幻灯片把目标写成三条：New unit of compute – the datacenter；What we want from multi-machine scaling 之下是 Linear memory scaling (max model params scales with num gpus) 与 Linear compute scaling (model flops scale linearly with num gpus)；以及 Simple collective comms primitives。讲师口头上把这两条线性目标说成：要能控制用多少显存、能控制能调动多少算力，并且希望这一切 lossless（无损），也就是 use all of the resources that we have。这正是 00:00 那两个瓶颈的镜像表述：显存要随 GPU 数线性增长、算力也要随 GPU 数线性增长，而实现手段只能是后面要讲的、建立在这些简单集合通信原语之上的算法（the algorithms that will enable this through all these like communication primitives components）。

这一章与前后的关系值得单独点明。向前看，它把 02:11 引入的 intra-node / inter-node 快慢之分具体化成了 toroidal mesh、fat tree、Virgo 与 384 卡光纤域这几种真实拓扑，也解释了为什么 03:05 那条 all-reduce 恒等式需要在不同链路上分别核算成本；向后看，它给出了 11:07 之后全部并行算法的评价函数——任何策略都要回答「显存是否线性扩展、算力是否线性扩展、通信是否落在可承受的链路上」。局限同样明显：这些数字来自厂商公开规格与第三方分析，讲的是硬件能力，不等于实际训练能拿到的利用率；讲师自己也说，本讲主体仍是硬件无关的算法，硬件这一节只是「有意思」且能解释公司之间的策略分化，不能把它当成选择并行策略的充分依据。

![华为 Ascend 910C Cloud Matrix 384 与 Nvidia GB200 NVL72 的芯片级/系统级规格对比表及整机功耗](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0008_00-09-23.jpg)

## 朴素数据并行与每参数 16 字节的显存账本 [11:05 - 15:04]

11:07 起进入本讲主体，讲师说 the second part of this lecture is going to be the bulk of the content，即用来并行化语言模型的算法。*Part 2 – Standard LLM parallelization primitives* 幻灯片把问题写成 How do we parallelize LLMs? 3 important ideas，三类分别是 Data parallelism（数据并行，子项 Naïve data parallel 与 ZeRO levels 1-3）、Model parallelism（模型并行，子项 Pipeline parallel 与 Tensor parallel）、Activation parallelism（激活并行，子项 Sequence parallel）。讲师口头上的分法略有不同但更贴近直觉：data parallel 是把要训练的数据分到各 GPU 上，动的是数据（the data is the ones that move around）；model parallelism 是把模型的不同部分切开来分，动的是模型。他特意提醒这条边界 a little leaky（有点漏），因为其中某个算法其实也会切参数——指的就是后面 ZeRO 把参数也切开的情形。之所以仍这样分箱，是因为概念上要同时 scale out compute 与 memory，两类技巧解决的是不同侧面。

最先讲的是最直观的 data parallelism（数据并行）。为了先把机制讲干净，讲师让听众「忘掉 Adam」，暂时用最朴素的 SGD。设全局 batch 大小为 $B$，每个 update step 取 $B$ 个元素、把梯度加起来再更新，幻灯片把起点写成 Starting point – imagine we are doing naïve SGD，并给出公式（注意幻灯片里就是对梯度求和，没有除以 $B$ 的归一化因子，与讲师「I'm going to sum up the gradients, and I'll take my update」一致）：

$$ \theta_{t+1} = \theta_t - \eta \sum_{i=1}^{B} \nabla f(x_i) $$

最朴素的并行做法写在下面一行：Naive parallelism: split the elements of B sized batch across M machines. Exchange gradients to synchronize。也就是把 $B$ 切成 $M$ 份，若整除则每台机器拿 $B/M$ 个样本，各自算梯度，再同步（synchronize）梯度求和。幻灯片右下角那个浅蓝框 How does this do? 把三项评估写得很干脆：Compute scaling – each GPU gets B/M examples.（只要每卡分到足够样本，计算就能近线性扩展）；Communication overhead – transmits 2x # params every batch. OK if batches are big（每个 batch 要传 2 倍参数量，batch 足够大时才划算）；Memory scaling – none. Every GPU needs # params at least（完全没有显存扩展，每张卡至少要装下全部参数）。讲师口头上把通信量的来历说成「要来回传梯度」，也就是一次把梯度汇总、一次把结果拿回来，合计 $2 \times \#\text{params}$；显存侧则是 Every GPU has the same copies of the models、需要同样大的 activations，因此 you don't save any memory at all。

于是问题就很清楚了：compute 解决了一部分，memory 一点没解决。讲师用一句带点自嘲的话过渡——「起初你可能觉得显存状况很糟（bad），算完账会发现其实是 terrible」。*What's wrong with naïve data parallelism? – Memory* 这张幻灯片第一句是加粗的 Our memory situation is actually terrible.，接着 Depending on our precision..，然后用一个浅蓝框给出结论：We need 5 copies of weights and 16 bytes per param!（需要 5 份权重副本、每个参数 16 字节）。下面五条逐项把 16 字节拆开：2 bytes for FP/BF 16 model parameters（FP/BF16 模型参数）、2 bytes for FP/BF 16 gradients（FP/BF16 梯度）、4 bytes for FP32 master weights (the thing you accumulate into in SGD)（FP32 主权重，即 SGD 里累加进去的那份）、4 (or 2) bytes for FP32/BF16 Adam first moment estimates（Adam 一阶矩估计）、4 (or 2) bytes for FP32/BF16 Adam second moment estimates（Adam 二阶矩估计）。右侧一个竖括号把这些「往里累加的量」括起来标成 "Optimizer state"（优化器状态）。写成算式就是：

$$ 2 + 2 + 4 + 4 + 4 = 16\ \text{bytes / param},\qquad \text{共 } 5\ \text{份副本} $$

为什么「比参数还多」是这里最需要转述清楚的直觉。讲师解释，参数本身只是一份；梯度是写回参数之前的累加器，还得再占一份；为了数值稳定，有时还需要更高精度的累加器（可能是临时的）；而一旦用 Adam，坏事才真正开始——要跨时间追踪梯度的一阶矩和二阶矩，也就是两份随时间累积的量，而且受稳定性（stability characteristics）影响往往得用高精度保存，相当贵。他把这类东西统称 optimizer state（优化器状态），并点明「看账本的话，这才是 SGD/Adam 一次更新里显存开销的大头」。这也解释了 15:23 之后那张彩色条形图为什么把绿色（Optimizer States）画得最厚：参数与梯度大小相同（都是 2 字节量级），状态却明显更大。一个常见易错点是只按「参数 2 字节」估显存，从而低估 8 倍；另一个是把 16 bytes/param 当成普适常数——幻灯片开头就写了 Depending on our precision，两条 Adam 矩都标成 4 (or 2) bytes，说明它随精度选择浮动。

这一章的落点是把「显存没法靠朴素数据并行省下来」量化到可核对的程度，为下一章 ZeRO 的切分提供动机。讲师在 15:04 前后已经预告了思路：既然 optimizer state 占大头，那就用一种结构化的方式把这部分开销切开、分到不同加速器上，而且 doing so in this structured way that gets you these gains for basically free。值得注意的是，朴素数据并行「显存复制」的坏处是线性发散的：讲师在 15:47 会说 naive data parallel 下 memory consumption is linear with your number of accelerators——不是「多卡反而更省」，而是每张卡都复制一份、总占用随卡数线性增加。因此本章的 $2\times\#\text{params}$ 通信量与 16 bytes/param 显存账，正是后面所有 Stage 1/2/3 公式要对照的 baseline（基线）；同时「OK if batches are big」也埋了一个伏笔：通信能否被摊薄取决于 batch 大小，这一点会在后续讨论 batch size 限制时再次出现。

![朴素数据并行的显存账本：每参数需 5 份副本共 16 字节，优化器状态占大头](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0012_00-14-24.jpg)

## ZeRO 的切分路径：从 120GB 到 1.9GB [15:04 - 16:45]

有了上一章的账本，讲师把解法概括成一句 Core idea: split up the expensive parts (state) and use the reduce-scatter equivalence.（把最贵的部分——也就是状态——切开，并利用 reduce-scatter 的等价关系）。*ZeRO – solving the memory overhead issue of DP* 这张幻灯片用四行堆叠条形图把收益画了出来，横轴是 gpu_0、gpu_i、gpu_{N-1} 三张卡（中间用省略号），图例三色分别是蓝色 Parameters（参数）、橙色 Gradients（梯度）、绿色 Optimizer States（优化器状态）。Baseline 那一行三张卡都顶着同样厚的绿条，正是朴素数据并行「每张卡复制全量状态」的样子；往下依次是 $P_{os}$（只切优化器状态）、$P_{os+g}$（再切梯度）、$P_{os+g+p}$（连参数一起切），绿条与橙条逐层被削成只剩一小截竖条，最后一行每张卡只剩下一条极窄的彩色条。讲师在口头上把颜色对应关系说得很清楚：optimizer state 是绿色、占大头，gradients 是橙色、parameters 是蓝色，且 parameters 与 gradients 大小相同、state 更大。

右侧 Memory Consumed 一列给出了逐行的精确公式，右上角还标了算例参数 $K=12$、$\Psi=7.5B$、$N_d=64$（$\Psi$ 是参数量，$N_d$ 是数据并行度，$K$ 是与优化器状态字节数相关的系数）。四行依次是：

$$ (2 + 2 + K)\times\Psi = 120\text{GB} $$
$$ 2\Psi + 2\Psi + \frac{K\cdot\Psi}{N_d} = 31.4\text{GB} $$
$$ 2\Psi + \frac{(2+K)\cdot\Psi}{N_d} = 16.6\text{GB} $$
$$ \frac{(2 + 2 + K)\cdot\Psi}{N_d} = 1.9\text{GB} $$

把 $K=12$、$\Psi=7.5B$、$N_d=64$ 代进去可以核对：$(2+2+12)\times 7.5 = 120$，$120/64 = 1.875 \approx 1.9$。这正是讲师在 16:17 说的 all the way down from 120 to 1.9——显存从 120GB 降到 1.9GB，约 63 倍的降幅。三项切分各自对应一条公式：$\Psi$ 前的系数 2 是 2 字节的参数或梯度，$K$ 是优化器状态的字节倍数；随着切分范围从 state 扩到 gradients 再扩到 parameters，被 $N_d$ 除掉的项越来越多，剩下的未切分项越来越少。

直觉上这四行还回答了一个「为什么切 state 最划算」的问题。Baseline 的 16 字节里，优化器状态对应 $K=12$ 字节、占了四分之三；先只切它（$P_{os}$），显存就从 120GB 掉到 31.4GB，一步拿走绝大部分收益；再切梯度只降到 16.6GB，再切参数才到 1.9GB。也就是说收益不是均匀的，第一刀最值钱。讲师在 15:53 到 16:17 用口语把这条路径说了一遍：perhaps we can shard the optimizer state onto different GPUs（这会是 big savings），再 shard 梯度、乃至 in the extreme shard everything，最后 the implied memory gains on the very right would be quite dramatic。把「切得越多省得越多」从定性说法变成 120 → 31.4 → 16.6 → 1.9 这条数列，是这页相对转写最重要的增量；它也把 11:42 那句「数据并行与模型并行的边界有点漏」落到了具体数字上——被切开的确实是参数与梯度这些「模型的东西」。

紧接着讲师把话题转到代价：the main thing that we will sort of study now is what is the cost of doing this。他先替听众说出预期——You might imagine there's no free lunch，baseline 最快，越往下走越要为省下的显存付一大笔通信代价（pay a big communication cost）；然后立刻给出反转：I think the remarkable thing is going to be that a lot of this will be free（值得注意的是，其中很大一部分会是免费的）。这里的「免费」不是指没有通信，而是指切分之后所需的通信操作，恰好能塞进原本就不得不做的那次 all-reduce 里——这正是 03:14 那条恒等式被提前反复强调的原因。于是本章在结构上是一个悬念：显存收益已经用公式坐实，通信代价要到 16:45 之后逐 Stage 拆开算才知道。

这一章的易错点有两处。其一，$K=12$、$\Psi=7.5B$、$N_d=64$ 是这张幻灯片给定的一组算例，不是普适常数；把 120GB / 1.9GB 直接套到别的模型或别的并行度上是错的，正确做法是保留公式本身再代入自己的 $\Psi$ 与 $N_d$。其二，公式里的 $2\Psi$、$K\Psi$ 都隐含「每参数字节数」的精度假设，与上一章 16 bytes/param 那条 Depending on our precision 的提醒是同一个前提；精度一变，系数就变。另外要区分「显存省了」与「训练变快了」：本章只核算显存占用，通信量与利用率要等 Stage 1/2 的推导，别把 1.9GB 直接读成吞吐提升；同理，四行里的 $N_d$ 是数据并行度，而不是 GPU 总数，把它和流水线或张量并行的度数混为一谈会让整套公式失去意义。

![ZeRO 四阶段显存公式与数值：Baseline 120GB 到 P_os+g+p 的 1.9GB，含 K=12、Ψ=7.5B、N_d=64](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0013_00-15-39.jpg)

## ZeRO Stage 1 的免费午餐与 Stage 2 的梯度切分 [16:45 - 20:02]

从 16:45 起讲师按 Stage 逐个拆通信代价，并说明 there's going to be three stages corresponding to each of these steps（三个阶段分别对应前面那三步切分）。ZeRO Stage 1 最简单：只切 optimizer state，把它 shard across the GPUs；每张卡仍保有完整的 parameters 与 gradients。分工方式可以想成 each worker is going to be responsible for updating one slice of the parameters——若我是 GPU 0，就只负责更新最前面那一片参数，其余更新一概不管。*ZeRO stage 1, how it works* 这张幻灯片把流程写成四步：Step 1. Everyone computes a full gradient on their subset of the batch；Step 2. ReduceScatter the gradients – incur #params communication cost；Step 3. Each machine updates their param using their gradient + state.；Step 4. All Gather the parameters – incur #params communication cost。四步里第一步对应「每人数据不同、各自算完整梯度」，第二步对应「把梯度按片发给该负责的人」，第三步对应「用本地参数、收到的梯度与本地状态做更新」，第四步对应「把更新好的参数片发回所有人」。

四步之间的数据形状变化，幻灯片用小图配下标公式画了出来。Step 2 的 reduce-scatter 图里，rank 0 到 rank 3 起初各持 in0（蓝）、in1（红）、in2（绿）、in3（黄），操作后 rank 0 只拿 out0、rank 1 只拿 out1、rank 2 只拿 out2、rank 3 只拿 out3，标注 `out[Y][i] = sum(inX[Y*count+i])`；Step 4 的 all-gather 图反过来，每个 rank 起初只持一小片，操作后每个 rank 都叠齐四色，标注 `out[Y*count+i] = inY[i]`。这两条公式正是 03:07 那页五个原语里 Reduce Scatter 与 All Gather 的原式，可见整段推导是建立在同一个记账体系上的。讲师在口头上把 Step 2 的必要性解释得很细：每台机器都算出了完整梯度，但别人并不需要全部梯度、只需要与自己那片参数相关的那部分（they don't need all the gradients, they only need the gradients associated with their update），所以「把我的梯度切开、按人发过去」正好用 reduce-scatter，通信量是 $\#\text{params}$ 级别；Step 4 则是把自己更新好的那一片参数（比如四分之一，I was responsible for let's say like 1/4 of the parameters）发回给所有人，用 all-gather，同样是 $\#\text{params}$ 级别。

关键的免费结论就在这一步合拢。朴素数据并行要做一次 all reduce，通信量 $2\times\#\text{params}$；Stage 1 做的是 reduce scatter + all gather 两次操作，各自 $\#\text{params}$，合计仍是 $2\times\#\text{params}$。而这正是 03:14 那条恒等式：

$$ \text{reduce-scatter} + \text{all-gather} \;\equiv\; \text{all-reduce} \;\Longrightarrow\; \#\text{params} + \#\text{params} = 2\times\#\text{params} $$

讲师在 18:54 到 19:03 把话说死：So, zero stage one is going to have the exact same sort of communication characteristics as naive DDP... this was free. Free memory savings. 显存这边则把除优化器状态以外的部分按 GPU 数均分，用他的说法是 taken the parameters and anything that's not optimizer state, I've divided by NGPUs，对应上一章公式里 $2\Psi + 2\Psi + K\Psi/N_d = 31.4\text{GB}$ 那一行。这里要留意的易错点：等价是通信「量」与「代价特征」上的等价，不是操作序列完全相同；朴素 DDP 是所有人拿到完整梯度后各自更新，Stage 1 是每人只拿一片梯度、只更新一片参数再收集参数，语义并不一样，只是账单相同。另一个易错点是别把「免费」理解成零通信——两次操作仍各有 $\#\text{params}$ 的数据要传，只是没有比原本多付。

Stage 2 从 19:15 开始，目标是把 gradients 也切开。讲师先说这会 trickier：Stage 1 依赖的是「我能算出完整梯度」，而 Stage 2 里连完整梯度都不能 materialize（物化）出来。*ZeRO stage 2. the simple extension to gradient sharding* 这张幻灯片上方直接引用了上一章那一行的片段——$P_{os+g}$ 对应的 $2\Psi + \frac{(2+K)\cdot\Psi}{N_d}$ 与 16.6GB——中间写着 Emboldened by our success, let's shard even more stuff。High level idea 两条是 Also keep the gradients (pink slices) sharded across the machines. 与 Use the same (rough) tricks as stage 1.；Complexity 一条点破难点：we can never instantiate a full gradient vector, but each worker must compute a full gradient (since we're data parallel!)——不能一次性物化完整梯度向量，但因为仍是数据并行，每个 worker 又必须算出完整梯度。

讲师给出的解法被他自己称作 a systems trick（一个系统层面的技巧）：不再等整个梯度向量算完再切，而是 as I sweep backwards through my network（沿网络反向回扫）时逐层处理——每算完一层的梯度，就立即 reduce 给负责该片的 worker；一旦某个梯度在后续反向图里不再需要（once gradients aren't needed in the backwards graph），就立即 free 掉。于是「完整梯度」从头到尾没有作为一个整体存在过，只以分片形式短暂出现并按层释放，显存因此进一步降到 16.6GB 这一行，而通信量仍由 reduce-scatter 与 all-gather 的等价关系兜住。这也解释了 11:42 那句「边界有点漏」的伏笔：ZeRO 被归在 data parallelism 名下，却实实在在切了参数与梯度。本段在 20:02 处收尾时讲师这句话还没说完，Stage 2 的后续推导与三个 Stage 的通信量对照都留在本段之后，因此不要把 16.6GB 当成 ZeRO 的终点；另一个局限是这套逐层 reduce 与立即释放的做法对实现要求很高，讲师把它归为 systems trick 而非算法上的新结论，说明收益来自对反向计算图生命周期的精细管理，而不是改变了数学本身。

![ZeRO Stage 1 四步流程：完整梯度、ReduceScatter、本地更新、All Gather 及两条下标公式](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0015_00-18-09.jpg)

## 从 ZeRO Stage 2 的收尾到 ZeRO Stage 3：把参数也切分 [20:02 - 22:17]

本段开场承接上一段 ZeRO Stage 2 的收尾。讲师指出，反向传播时可以「一边增量地计算梯度、一边增量地把梯度发出去」（kind of now incrementally computing and sending gradients as you go），把梯度分批发还是一次性发完在通信账上是同一回事（doing it incrementally or doing it all at once, same thing），因此「分不分批」并不改变总字节数。等这一步做完，最后再对参数做一次 all-gather，于是梯度在某种意义上已经被切分（sharded）到各张 GPU 上，却没有带来任何额外通信成本（without incurring any additional costs）。这正是 Stage 2 的「免费」结论：它靠的是 all-reduce 与 reduce-scatter + all-gather 的恒等关系，把原本必须做的梯度同步重排成「每算完一片就散出去、算完再收回来」的形式。这里的易错点在于，免费指的是通信字节数不变，而不是没有同步；每张卡仍然要完整走一遍前向与反向，只是不再同时持有完整的梯度向量。这个「免费」的甜头，正是接下来要把事情推到极致的动机。

紧接着讲师说，最后、也最「毛茸茸」（the last and the most hairy thing）的一步，是把参数也沿着所有状态一路切到底——「we've gotten something for free so far, let's keep pushing it... shard the parameters as well all the way across all the states」。幻灯片标题是 `ZeRO stage 3 (aka FSDP) shard everything`，副标题写着 "We've gotten almost everything for free so far.. lets try to solve all our memory issues"，并抛出问题 "Is it possible to do this with low overhead?"。图上方画了一条被切成许多细柱的显存条，标注 $P_{os+g+p}$，右侧给出每卡显存公式

$$P_{os+g+p} = \frac{(2 + 2 + K)\,\Psi}{N_d}$$

其中 $\Psi$ 是参数量、$N_d$ 是数据并行度、$K$ 是优化器状态相关系数，右侧还标了一个具体数值 `1.9GB`；图例用蓝、橙、绿三色分别对应 `Parameters`、`Gradients`、`Optimizer States`。`High level idea` 下有三条 bullet：`Shard everything – incl parameters!`、`Use the same 'incremental communication / computation' ideas`、`Send and request parameters on demand while stepping through the compute graph.`。也就是说，Stage 3 沿用 Stage 2 的增量思路，但把「按需收发」的对象从梯度换成参数：在遍历计算图（stepping through the compute graph）的过程中按需 send / receive parameters，于是每张 GPU 在任意时刻只同时看到参数、梯度、优化器状态三者各自的一个切片（each GPU only sees a slice of both the parameters, gradients, and optimizer states at any one time）。这一阶段在 PyTorch 里就叫 FSDP（Fully Sharded Data Parallel），讲师补了一句：只要用 torch 做过并行，大概率已经和它打过交道（if you've ever paralyzed anything in a torch library, you've probably dealt with this before）。

![ZeRO Stage 3（FSDP）shard everything：P_os+g+p 每卡显存公式、Parameters/Gradients/Optimizer States 三色切片与 High level idea 三条 bullet](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0017_00-20-40.jpg)

底层做法是在一次完整更新里做「两次 all-gather 加一次 reduce-scatter」（two all gathers and one reduce scatter as you go）。按时间从左到右：先 load model，上下两张 GPU 各自只持有分片；对第 0 层先 all-gather 出这一层的完整权重，做 forward，然后立刻 free full weights——因为这一层的前向已经算完，完整参数不再需要（I can free my weights cuz I've already done before）；进入 backward 时，被保留下来的是 activations，参数则要再 all-gather 一次按需取回（which I can all gather kind of on demand），算完 backward 立刻 reduce-scatter 把梯度散回负责该分片的卡（I got the gradients, I reduce scatter the gradients out），再 free full weights，如此逐层重复，最后各自 update weights。幻灯片 `ZeRO stage 3 (aka FSDP) how it works (baby version)` 把这条链画成上下两行流程图，方块依次是 `LOAD-MODEL SHARD` → `ALL-GATHER` → `FORWARD (LOCAL)` → `FREE FULL WEIGHTS` → `ALL-GATHER` → `BACKWARD (LOCAL)` → `REDUCE-SCATTER` → `FREE FULL WEIGHTS` → … → `UPDATE WEIGHTS (LOCAL)`；虚线箭头旁标注 `GATHER WEIGHTS (ALL_GATHER)`、`SYNC GRADS (REDUCE_SCATTER)`、`Load shard From CPU if CPU Offloaded`、`Offload grads to CPU if CPU offload is enabled`，行下方写着 `FSDP instance 1: N layers`、`FSDP instance N: M layers`，左侧还有一个 `Data` 方框 feeding 两条时间线。底部一行结论是 `Communication cost – 2 all gather (#param), 1 reduce-scatter (#param).`，并给出 PyTorch 官方教程链接 `https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html`。用伪代码概括这个 wrapper 就是：

```python
for layer in model.layers:            # 逐层遍历计算图，前向
    W = all_gather(shard[layer])      # 按需取回完整权重
    act[layer] = forward(W, act[layer - 1])
    free(W)                           # 前向算完立刻释放完整权重
# backward 反向再来一遍
for layer in reversed(model.layers):
    W = all_gather(shard[layer])      # 反向时再次按需取回
    g = backward(W, act[layer])
    reduce_scatter(g)                 # 梯度一拿到就散出去
    free(W)
optimizer_step(local_shard)           # 只更新本地负责的那一片
```

![ZeRO Stage 3 工作流程 baby version：LOAD MODEL SHARD → ALL-GATHER → FORWARD(LOCAL) → FREE FULL WEIGHTS → ALL-GATHER → BACKWARD(LOCAL) → REDUCE-SCATTER → FREE FULL WEIGHTS → UPDATE WEIGHTS(LOCAL)](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0018_00-21-55.jpg)

与前后内容的关系要理清：Stage 2 只把梯度与优化器状态切碎，参数仍完整；Stage 3 连参数也切，所以前向与反向都必须在「用到之前」把完整权重临时物化出来，用完再释放。易错点有两个：一是 activations 并没有被切分，它们在显存里一直保留到反向，所以 Stage 3 省的是参数、梯度、优化器状态那份账，不是全部显存；二是 `free` 必须发生在确认后续计算不再需要该层完整权重之后，否则会重复 all-gather，通信量会翻倍。讲师也埋下伏笔：这个「每层都在通信」的做法看起来疯狂（this seems a little crazy in terms of the overhead because we're doing communication all the time, every layer we're doing this communication），下一章正是要解释它为什么实际上几乎不花钱。

## 通信账本：3×#param 的代价与通信／计算重叠 [22:17 - 25:50]

先算通信账。讲师自问 "what's the communication cost, right? How many all gathers did we do?"，答案是两次 all-gather 加一次 reduce-scatter，共 `3 × #param`，相比朴素 DDP 的 `2 × #param` 多了一次 all-gather（one extra all gather, which seems bad）。更糟的是频率：不是每个 step 通信一次，而是每一层都在通信（every layer we're doing this communication），看起来延迟与开销都极其昂贵。写成公式：

$$\text{comm}_{\text{ZeRO-3}} = 2\,\text{all-gather}(\Psi) + 1\,\text{reduce-scatter}(\Psi) = 3\Psi = 1.5 \times \text{comm}_{\text{DDP}}, \qquad \text{comm}_{\text{DDP}} = 2\Psi$$

其中 $\Psi$ 是参数量。这个 `1.5x` 正是幻灯片 `What's the point?` 里的原话：`Distributed data parallel costs 2*# param communication`，而 `Zero stage 1 is 2*# param – it's free! – you might as well always do it`、`Zero stage 2 is 2*# param – it's (almost) free (ignoring overhead)`、`Zero stage 3 is 3*# param – 1.5x comm cost, but that's not bad! (ignoring latency..)`。注意括号里的 `ignoring overhead` 与 `ignoring latency..` 正是易错点：这个等式只计字节数，不计小消息带来的延迟与 kernel launch 开销，现实里层数越多、消息越碎，越可能被 latency 咬到。

但 Stage 3 用两个重要想法把这份开销几乎抹平（two really important ideas to make this essentially overhead free）。第一个前面见过：沿着计算图扫过去（sweep through compute graph），做必要的通信，然后立刻释放显存。第二个是本段新引入的：把通信与计算的时间重叠起来（overlap the communication and computation cost of this process）。讲师为此引用了一张来自解释 PyTorch FSDP 架构论文的图（arxiv `https://arxiv.org/pdf/2304.11277.pdf`），幻灯片标题 `Actual picture of how FSDP / ZeRO stage 3 works`，副标题 `Let's walk through a FSDP example to see some important ideas`。两条粗体小标题分别是 `Incremental computation / communication: Parameters / gradients are requested / sent and then immediately freed`，以及 `Overlapping communication and computation $(W_1 W_0 + W_2 W_0)x = y$`，下面一条 bullet：`The all-gathers happen all at once while forward happens, masking the comm cost.`。时序图分三行：最上面 `CPU` 一排编号小方块（左半标 `Forward`、右半标 `Backward`），中间 `GPU Comp. Stream` 有 `FWD0`、`FWD1 FWD0 FWD2`、`BWD2 2BWD0`、`BWD1 1BWD0`，最下面 `GPU Comm. Stream` 有 `AG0 AG1 AG2 AG2`、`RS2 AG1 RS1 RS0`。图例用五色区分 `All-Gather (AG)`、`Reduce-Scatter (RS)`、`Forward Comp. (FWD)`、`Backward Comp. (BWD)`、`Parameter Free`，还有 `FSDP Unit i` 标记每个单元。

![FSDP/ZeRO Stage 3 实际时序：CPU、GPU Comp. Stream、GPU Comm. Stream 三行，AG/RS/FWD/BWD/Parameter Free 图例与 (W1W0+W2W0)x=y](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0019_00-23-10.jpg)

逐帧看这张图：FSDP 必须先拿到参数才能计算，所以先 all-gather 第 0 层（I have to first get my parameters before I can do my computation. So, I'm going to all gather zero layer zero），即 `AG0`；一旦 `AG0` 完成就能做 `forward 0`，而在做 `FWD0` 的同时已经在请求下一层参数——「I'm all gathering one while I'm doing this computation」。等 `AG1` 到位就做 `forward 1`，同时开始 `AG2`。公式 $(W_1 W_0 + W_2 W_0)x = y$ 描述权重被复用的情形：讲师举例 `if you reuse that W naught, I'm reusing this forward zero sort of twice`，于是同一份参数可以在通信流里少取一次，把 `Parameter Free`（释放参数）与额外一次 `FWD0` 挤进计算间隙里。结论是：只要计算量足够大、网络足够快（if you have enough computation and your network's fast enough），计算耗时就会长于通信耗时（the computation can take longer than the communication），通信被完全盖住，于是得到「巨大的显存改善却几乎不付代价」（this really great improvement in memory usage without paying for almost any amount of memory use）。图上仍能看到少量 `bubble`（You do see some bubbles. These are overheads），这是无法被重叠掉的空隙。

把三个阶段横向放在一起：DDP（朴素数据并行）成本 `2 × #param`、单次 all-reduce；ZeRO Stage 1/2 在字面上是免费的（free in the literal sense），因为它们只是把一次 all-reduce 拆成 reduce-scatter + all-gather 这两个等价操作（the identity of all-reduce being equivalent to two operations），通信字节数完全不变；只有 Stage 3 多一次 all-gather。讲师强调，实践中跑 FSDP 会看到 GPU 利用率「非常接近单卡性能」（very close to, you know, just the single GPU performance），并称这一点 quite remarkable；它同时「概念上非常简单」（also very conceptually simple）——课程作业就要求自己写一个 FSDP 实现（you will actually have to write an FSDP implementation as part of your assignment），做法不过是写一个能包住任意 module 的 wrapper（write a wrapper that will wrap any module and turn it into its FSDP version），内部逻辑就是 `do a bunch of all-gathers, compute, free, and then repeat on the backwards pass`。幻灯片 `What's the point?` 底部那句 `This is also conceptually very simple – write a FSDP block wrapper.` 正是对此的总结。易错点与局限：重叠成立的前提是「计算 ≫ 通信」，一旦模型很小、层数很碎或跨慢链路，bubble 与 latency 会重新暴露出来；另外 `1.5x` 是字节数之比，不是端到端时间之比，不能据此推断训练变慢 50%。

## 课堂问答：FSDP 不是流水线，以及 ZeRO 实际能装多大 [25:50 - 28:53]

进入问答，这一段把上一章的抽象机制落到了具体数字，也为下一章的显存动机做铺垫。第一位同学把 FSDP 理解成「从下一张 GPU 拿梯度来算上一张 GPU 的梯度，一张接一张顺序传递」（taking gradients from the next GPU to calculate the gradients for the previous）。讲师立刻纠正：这更接近 pipelining（流水线并行）——即一层放在 GPU 0、另一层放在 GPU 1 那种切法（one layer was on one GPU and another layer was on another GPU）；而 FSDP 不是这样。在所有这些 ZeRO/DDP 情形里，每张 GPU 都要从头到尾走完整个模型（every GPU kind of goes through the entire, uh, model, right? From the start to finish），差别只在于没有任何一张卡会同时持有完整参数（no GPU is going to hold the entire parameters at the same time）。如果某张卡手上的网络分片里没有 layer 0，它会先按需把 layer 0 的参数取回来（demand the layer zero parameters first），做完这一层的完整计算再往下走。所以「计算和朴素数据并行的单卡计算完全一样」（the same single GPU computation as naive data parallel），只是在每个计算操作之间插入了 `request parameters` / `free parameters` 两步（request parameters, free parameters in between every computational operation）。

第二位同学追问「是不是每张 GPU 都持有每个层的一部分，优化器就是干这个的？」（every GPU has a part of every layer, and that's what the optimizer is for?），讲师答 "Yes, that's all"：parameters、gradients 和 optimizer states 三者都跨 GPU 切分（are all sharded across different GPUs）。第三问很有价值：为什么通信成本不随层数相乘而爆炸？（Why is the communication cost not multiplied by the number of layers as small?）答案是：通信操作的次数确实随层数线性增长（the number of operations multiplies by layer, that's So, that's correct），但每一次通信的数据量都很小，因为只是「一个很小的 MLP」（this is just one tiny MLP or something）；对比的基准是对整个网络做一次 all-reduce 的开销（I'm comparing that to the cost of all-reducing a whole network），把每一层的小通信加起来（it all adds up in some sense），总量仍与参数规模同阶。这是理解 FSDP 的关键记账方式：总字节数不变，变的是消息粒度与次数。

转到实际收益。讲师说，如果看单张 A100 GPU 能装下的最大模型，baseline 连 7B 都装不下（you can not even fit a 7B model），而用上 ZeRO Stage 3 之后可以装到 50 billion parameter 级别（you can now fit, you know, 50 uh billion parameter models and so on），显存改善相当显著（The improvements that you get in terms of the memory improvement is quite significant）。另一位同学要求回到上一张幻灯片，问 `8X A100` 上的数字换到 H100 或 H200 会怎样（on 8X E100, what would be the limitation on H100 or P200?，转述里口误为 E100/P200，讲师按 H100/H200 作答）。讲师坦承不做心算（I can't do the mental math of like multiply by 141 and divide by 80），但指出 H200 对应的就是把显存按比例换一下——`141`（GB，H200 的 HBM 容量）除以 `80`（GB，A100 80G 的容量）——并且强调这种换算「是线性的」（it would be linear），从一种加速器换到另一种并不会改变数学结构（wouldn't really change the math very much to go from one accelerator to another）。

这张 `ZeRO in practice: will it fit?` 幻灯片给出了精确账本。开头两句是前提：`Pure BF16 training (with Kahan summation), is viable and optimizer states are less beefy`，以及 `Let's say BF16 for everything but the master weights – 12 bytes per param`——即除了 master weights 保留高精度外全部用 BF16，并用 Kahan summation 补偿精度，于是每参数从常见经验值 16 字节降到 `12 bytes per param`。条件是 `On a 8X A100 80G..`。表格两列为 `Max size (params)` 与 `Formula for B/param`，四行分别是：`Baseline` → `6.66.. B` / `12`；`Zero stage 1` → `16 B` / `5`；`Zero stage 2` → `24.62 B` / `2 (param) + (10 (grad+state))/8`；`Zero stage 3` → `53.33 B` / `12/8`。把公式摊开：

$$\text{Baseline}: 12 \;\to\; \text{Stage 1}: 5 \;\to\; \text{Stage 2}: 2 + \frac{10}{8} = 3.25 \;\to\; \text{Stage 3}: \frac{12}{8} = 1.5$$

其中 `12/8` 里的 `8` 正是 `8X A100` 的卡数，说明 Stage 3 把全部 12 字节的状态平均分到 8 张卡上；`10 (grad+state)` 是梯度与优化器状态那部分，只有参数本身 `2` 字节没被切，这与前面「Stage 2 切梯度与优化器状态、不切参数」的定性描述完全对上。用 `8 × 80 GB = 640 GB` 除以每参数字节数即可得到那些最大规模，例如 $640/12 \approx 53.33\,\text{B}$，也解释了口述里的「7B → 50B」。易错点：这套数字依赖「BF16 + Kahan summation + FP32 master weights」的特定假设，换成 FP32 Adam 或不做 Kahan 补偿时，16 字节/参数的老经验值会重新生效；另外表格算的是「能否装下」，不含 activations，长序列下激活内存可能先于参数把显存撑爆——这正是下一章要展开的动机。

![ZeRO in practice: will it fit?：8X A100 80G 上 Baseline/ZeRO Stage 1/2/3 的最大模型规模与每参数字节数公式表](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0023_00-28-11.jpg)

## 数据并行的两个硬限制与模型并行的登场 [28:53 - 31:48]

讲师先给 FSDP 一个很高的评价——`FSDP, I think is very clean. It's very elegant. I like it a lot`，随后话锋一转：`I wish I could end the lecture here, but unfortunately, we will have to proceed into uglier and more hairy things`。必须继续往下走有两个原因，而且第一个是并行化里的重要观念：数据并行在消耗一种重要资源，那就是 batch size（data parallel is consuming an important resource... the batch size, so to speak）。若全局 batch size 为 8，那么加速器数量永远不可能超过 8（If you have a batch size of eight, you can never have more than a eight accelerators）。你也许会说「那就把 batch size 一直加大」，但这是不行的：到了某个点会出现 critical batch size（临界批量），此时「多加一个 batch 元素带来的收益，已经低于拿同样的算力在这个单样本上多走一步 SGD」（the gain that you get from the additional batch element is less than if you had taken another SGD step on that single element）。小 batch 时增加元素完全等价于多走 SGD 步（at small batch sizes, adding extra elements is, you know, perfectly helpful. It's the same as taking more SGD steps），但越过临界点后边际收益递减（diminishing return hits in），无穷大的 batch 并不比无穷多步更优（an infinitely large batch size is not infinitely better than infinite steps）。于是出现一个硬性权衡：要小 batch、让 GPU 更空闲，还是要大 batch、承受优化上的损失（Do we want small batch sizes and let our GPUs be more idle? Do we want big batch sizes and sort of take the hit from optimization?）。

幻灯片 `Issues remain with data parallel – compute scaling` 把这件事画成曲线。上方两行粗体：`With data parallel, #machines < batch size (and near this, comm overhead is high)` 与 `And there's diminishing returns to batch sizes`。图标题 `Predicted Training Speed`，纵轴是 $\varepsilon_{opt}(B)/\varepsilon_{max}$（从 $10^{-2}$ 到 $10^{0}$ 的对数刻度），横轴是 `Batch Size / Noise Scale $(B/\mathcal{B})$`（从 $10^{-2}$ 到 $10^{2}$ 的对数刻度）。曲线在左下沿直线上升的区域标着 `Perfect scaling`，越过 $B/\mathcal{B} = 10^0$ 处那条竖直虚线后趋于平台，右上标着 `Ineffective scaling`。也就是说，把 batch 增大到噪声尺度 $\mathcal{B}$ 之前，训练速度近似线性提升；之后再增大 batch，有效收益迅速饱和。写成约束就是：

$$\#\text{machines} \le \text{batch size}, \qquad \frac{\varepsilon_{opt}(B)}{\varepsilon_{max}} \to 1 \quad \text{as} \quad \frac{B}{\mathcal{B}} \gg 1$$

这里的 $\mathcal{B}$（noise scale）是图上独有的符号，转述时不要把它和 micro-batch 的 $b$、全局 batch 的 $B$ 混淆。

![数据并行的计算扩展瓶颈：Predicted Training Speed 曲线、Perfect scaling 与 Ineffective scaling 分区](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0024_00-29-26.jpg)

第二个原因是显存仍然有问题：ZeRO Stage 1/2 根本不让显存随卡数扩展（zero stages one and two don't really let you scale memory at all），Stage 2 能切分参数显存（zero stage two lets you cut up the parameter memory），但 activation memory 以及其它类型的显存占用完全没有被切分、也没有被减少（there's activation memory and other kinds of memory use that this does not split up, or this does not reduce at all）。因此需要更细粒度地切开模型（we need better ways, more fine-grained ways of cutting up the model in order to really start pushing down the memory use beyond these points）。由此引出 model parallelism（模型并行）：它和 FSDP 一样把参数跨 GPU 切分，但**关键概念差异**在于「现在要通信的是 activations」。讲师说得很直白：FSDP 里切的是参数，但某种意义上它只是个 wrapper，计算照常做，只是做之前把参数来回传（we were still doing the normal computation. We were just sending parameters back and forth before doing it, right? So parameters were the things flying around）；现在反过来，飞来飞去的是激活（now what we're going to do instead is we're going to communicate activations back and forth）。举例：layer 0 住在 GPU 0、layer 1 住在 GPU 1，那么两者之间要传的就是中间层的 activation（What I have to communicate is the activation of the layer in between）。

幻灯片 `Beyond data parallel – model parallelism` 的副标题点明目标：`Scaling up in memory (without changing batch size) with model parallelism`——即在不改 batch size 的前提下扩显存，正好绕开上一节的 critical batch size 困境。蓝色方框里 `What model parallelism is..` 两条 bullet 是 `It splits up the parameters across GPUs (like zero3)..` 与 `But communicate activations (while zero3 sends params).`——括号里的对比把 FSDP 与模型并行的分界线写得很清楚。下方 `We cover three different types of parallelism` 列出三类：`1. Pipeline parallel`、`2. Tensor parallel (+ Sequence parallel)`、`3. Expert parallel`，并收尾 `These correspond to three different ways of cutting up the model.`。对应讲师口述：pipeline parallel 是切分层（cutting up layers），tensor parallel 是切分矩阵（cutting up matrices），expert parallel 是把 experts 分片放到不同地方（sharding experts into different places）；他还提到 Percy 之前已讲过 tensor parallel，所以听众已有大致心智模型（Percy's talked about, you know, tensor parallel already）。与前后内容的关系：这一章是整堂课从「数据并行」转向「模型并行」的枢纽，后面三章分别展开 pipeline，以及由 depth/width 类比引出的 tensor parallel。易错点：模型并行并不天然比 FSDP 省通信，它只是把通信对象从「参数矩阵」换成「激活张量」，只有在 $b \times s \times h$ 小于参数矩阵时才更划算。

## 流水线并行：按层切分、气泡与 micro-batch [31:48 - 34:13]

先讲 pipeline parallel，也叫 layer-wise parallel（按层并行）。讲师说这在概念上简单到有点天真——早年学并行时的第一反应就是「把层切开、放到不同 GPU 上」（Oh yes, it's simple. Just cut up the layers and put them on different GPUs）。做法是：前向传播里把 activations 往下一张卡传（pass activations uh forward in the forward pass），反向传播里把 partial gradients 往回传（pass like partial gradients backwards in the backward pass）。幻灯片 `Layer-wise parallel` 画了四个竖条 `Layer 0`–`Layer 3`，下方对应四个 GPU 图标 `GPU 0`–`GPU 3`；上方一排红箭头标 `forward` 从左往右，下方一排蓝箭头标 `backward` 从右往左。幻灯片上还压着一行淡淡的字 `Utilization of layer-wise parallelism is terrible`（利用率极差），底部 caption 是 `Layer-wise parallel cuts up layers, assigns some subset to GPUS` 与 `Activations and partial gradients are passed back and forth`。

如果就这么朴素地做，会得到一张「非常难看、非常 depressing」的图（You will get a terrible-looking picture like this. Um very very depressing）。讲师设定 4 张 accelerator，每张负责模型的 1/4（I have four um accelerators, um each handling a quarter of the model），时间从左往右扫（time is sweeping from left to right），不同行是不同 accelerator。第 0 张卡（负责第一层）先算一会儿，然后就停下来（it will do some computation, and then it will stop），把计算交给第二张，之后几乎全程空闲，直到最后反向回来时才又活动一下（it will do nothing for the rest of time um until the very end）。任何时刻只有一张 GPU 在干活（one GPU will be active at a time），反向时同样一次只有一张卡在算（coming backwards, one GPU will be active handling the backward pass at a time）。结果绝大多数时间 GPU 都是 idle，利用率 truly truly terrible。这段空闲就叫 **bubble**——`It's called a bubble. It's the part of the pipeline where you're not doing anything`。这里的关系要接上一章：正因为 FSDP 让每张卡走完整模型、没有这种层间等待，才显得「elegant」；模型并行一旦按层切开，就必须为 bubble 付出代价。

解法本质上是 batching / pipelining：不再一个样本接一个样本地串行，而是把 batch 切成多个 micro-batch，一旦某个 micro-batch 在某一 stage 算完就立刻把它交给下一 stage、并马上开始下一个 micro-batch（as soon as you process one, you start on the next element, and then you pass off the one that's already done to your next layer）。幻灯片 `A solution: pipeline parallel` 的甘特图用 $F_{i,j}$ / $B_{i,j}$ 标出四个 micro-batch（图中 `in this case, 4`）在四个 stage 上的前向与反向方块：`F_{0,0} F_{0,1} F_{0,2} F_{0,3}`、`F_{1,0} F_{1,1} F_{1,2} F_{1,3}`、`F_{2,0} F_{2,1} F_{2,2} F_{2,3}`、`F_{3,0} F_{3,1} F_{3,2} F_{3,3}`，右侧对应 $B_{3,0} \dots B_{0,0}$ 一排 $B$ 方块，最右一列是四个 `Update`。中间那块大空白被明确标成 `Bubble`。蓝色方框写着 `Solution: Pipeline-parallel. Process 'micro-batches' (in this case, 4). Send off the first microbatch and start computing the second.`。底部给出定量结论 `The ratio of bubble time to useful compute is ..`

$$\text{bubble ratio} = \frac{n_{\text{stages}} - 1}{n_{\text{micro}}}$$

后面紧跟 `so we need a big batch size!`。也就是气泡占比与 stage 数成正比、与 micro-batch 数成反比，要让气泡趋近于零，就必须把 $n_{\text{micro}}$ 做得很大（if we want our bubble to be zero, it's going to go down as one over the micro batch size）。讲师顺势回扣上一章：`This is kind of why I said um batch sizes are useful resource because now we can spend it in a different way, which is the pipeline more to reduce the amount of uh idle time in pipeline parallel`——batch size 不再只用于数据并行，还能「花」在加深 pipeline 上以减少空闲。

![流水线并行 micro-batch 甘特图：F/B 方块、Bubble 与气泡占比公式 (n_stages−1)/n_micro](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0027_00-33-38.jpg)

易错点与局限：公式 $\frac{n_{\text{stages}}-1}{n_{\text{micro}}}$ 只在每个 micro-batch 前向/反向耗时相近、且 stage 划分均匀时才成立；一旦某层特别重，或反向约为前向两倍耗时，气泡形状会变形。更要命的是，把 batch 切成很多 micro-batch 会与上一章的 critical batch size 冲突——为了压气泡需要大 batch，但大 batch 又会带来优化上的边际收益递减。讲师还引了一句行内玩笑：`parallelization code looks reasonable and people can understand it until you implement pipeline parallel`，暗示调度实现复杂度陡增。这正是下一章要回答「既然这么糟，为什么还要用」，以及再下一章 zero bubble 调度要收拾的烂摊子。

## 为什么还要用流水线：通信性质与最慢链路 [34:13 - 36:04]

讲师先自嘲 `Okay. So pipelines are terrible`，然后正面回答「为什么还做」。其一，相比 DDP 能省显存，因为层被切开了（pipelines can save memory compared to DDP cuz we've cut up layers），而且还能与 data parallel 组合使用（You also compose this with data parallel, so you can do both）；其二，也是更重要的一点，在若干拓扑里「你总会有一条 pipeline」（you will always have a pipeline in certain kinds of topologies is that it has very good communication properties），因为它的通信性质非常好（compared to both FSDP and compared to you know any other kinds of things you can do）。幻灯片 `Why pipeline parallel?` 开头是 `Pipelines seem terrible. Why do we do it?`，蓝色方框里两条编号理由：`1. Pipelines save memory (compared to DDP)`；`2. Pipelines can have good communication properties (compared to FSDP) – it depends only on activations (b×s×h) and is point to point`。底部一句 `Generally, we will use pipelines on slower network links (i.e. inter-node) as a way to get better memory-wise scaling.`。

![Why pipeline parallel?：两条理由（相比 DDP 省显存、仅依赖 b×s×h 激活且 point to point）与跨节点慢链路说明](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0028_00-35-04.jpg)

把通信量写清楚。流水线每个 stage 之间只传激活张量，大小是

$$\text{comm}_{\text{PP}} = b \times s \times h$$

其中 $b$ 是 micro-batch 大小、$s$ 是 sequence length、$h$ 是 hidden dimension。它「几乎总是小于试图传整个参数矩阵」（almost always a smaller amount of data than like attempting to communicate a whole parameter matrix）——这是紧接着的课堂问答里同学追问「为什么比 FSDP 通信省」时讲师给出的机制：`it's batch times sort of sequence length times hidden`，而 FSDP 要 all-gather 的是 $\Psi$ 量级的参数。更关键的是拓扑性质：它是 point-to-point（one layer is going to communicate to another. It's not all-to-all），只有相邻 stage 之间一对一来回，不需要全员参与的 all-reduce / all-gather。于是「depending on how you sequence these things, they can be a very efficient way to utilize your network」。这与前面硬件章节的节点内/节点间区分一脉相承：张量并行、专家并行吃带宽，留在节点内 NVLink；流水线通信量小、点对点，放到跨节点慢链路。

由此得到部署经验：pipelines are going to live on the slowest networking links。如果有多个 data center 或多个 pod，彼此通信较慢（if you have multiple data centers or multiple pods that are slower to talk to），就用 pipeline parallel 跨它们切（you're going to parallelize across those two using pipeline parallel because that's kind of the most communication efficient uh parallelism that you can kind of bring to bear here）。除此之外，流水线省显存这件事还可以与 data parallel 叠加——先按数据并行复制若干份、再在每份内部按层切开（You also compose this with data parallel, so you can do both），于是「省显存」与「扩算力」不必二选一，这也解释了为什么明知气泡难看，工业界仍保留这条路径。

讲师还提醒 pipeline 性能强烈依赖 batch size（pipeline performance is really dependent on batch size），并点名本段反复引用的论文是 NVIDIA 的 Megatron paper（This is um uh the Megatron paper from Nvidia. They have a bunch of really lovely sort of parameter sweeps of what happens to utilization as a function of different sort of configurations you can do）。这类 parameter sweep 的意义在于：利用率不是流水线并行的固有属性，而是 batch size、pipeline size、micro-batch 数等配置共同决定的函数，所以「pipeline 好不好」必须放到具体配置里回答。结论是：`if you have large batch sizes and you pick a large pipeline size, you can get pretty good utilization close to, you know, not doing pipeline parallel at all`，但 `Otherwise, pipeline parallel is going to rapidly degrade your utilization and performance`。讲到这里他特意停了一下：`Actually, I'm going to pause for a moment here to ask questions cuz the next slide is uh is complicated`，因为接下来要进入更复杂的调度；行内那句玩笑 `parallelization code looks reasonable and people can understand it until you implement pipeline parallel` 也恰好说明，一旦要把这些配置与调度真正写进代码，复杂度会陡增。

易错点：$b \times s \times h$ 里的 $b$ 是 micro-batch 而非全局 batch，切得越碎、单次点对点消息越小，越容易被延迟主导；长序列 $s$ 很大或 hidden $h$ 很大时，这个值会迅速逼近参数矩阵大小，此时「流水线一定更省通信」的判断就不再成立。另一个局限是气泡与省显存之间的拉扯：省显存来自切层，压气泡却需要大 batch，而大 batch 又受 critical batch size 约束。这些矛盾正是下一章 zero bubble 调度要处理的对象。

## 零气泡流水线与沿宽度切分：张量并行的引出 [36:04 - 40:05]

问答先补上了通信机制：同学问「上一张说 pipeline 比 FSDP 通信性质好，具体机制是什么？」（What is the specific mechanism for why that is?），讲师答 `it's a smaller amount of stuff that's being communicated. So, it's batch times sort of sequence length times hidden`，即 $b \times s \times h$，并强调 `that's almost always a smaller amount of data than like attempting to communicate a whole parameter matrix`。随后进入更聪明的调度：人们试图用更精巧的切法减少 pipeline size，比如把不同阶段分给不同层、不同 micro-batch（cut up sort of the different stages of what you're doing for different layers or different microbatches），把某些 forward pass 元素插到不同 backward pass 元素之间（sequence different forward pass elements in between different backwards pass elements），通过 clever scheduling 进一步压小气泡。讲师点明 `This was taken from the DeepSeek paper`——即 DeepSeek 的工作通过精妙操纵 forward / backward 的计算时机（by really cleverly manipulating when you're doing forward and backward computation），把 bubble size 再降一点。

但「更聪明」的是 zero bubble pipelining（零气泡流水线）。讲师强调它「其实不是调度问题，而是仔细思考 backward pass 的结构」（is not really about scheduling per se, but it's actually about kind of thinking carefully about the structure of your backward pass）。反向传播在计算图里其实要做两件事：一是把 partial derivatives 继续往回传（you're going to propagate your sort of partial derivatives further down your computation graph），二是计算当前权重的梯度（you also want to compute the gradients for the current weight）。关键区别在于：回传偏导非常要紧，因为下一 stage 在收到这个信号之前没法干活（your next stage can't do any work until you've propagated that signal back）；而 $\partial L/\partial W$ 是计算图上的「叶子节点」（a leaf node, so to speak, on this graph），什么时候算都行（I can do it whenever）。幻灯片 `'Zero bubble' pipelining` 左侧写着 `Split up backwards into two parts`：`1. Backpropagating activations (z,x)`、`2. Computing weight gradients (W)`，下面一句 `The second part can be done whenever`；左下角是 `Figure 1: Computation Graph for MLP` 的计算图，横轴分三段标 `F`、`B`、`W`，前向是 $z = W x$、$y = \sigma(z)$，反向里 $B$ 段算 $\nabla_z L$ 往回传，$W$ 段算 $W^\top \nabla_z L$ 与 $\nabla_W L$。于是策略是：把 $B$（图中标 `B` 的方块）尽快排完以解锁下游（you kind of compute the B's as quickly as you can, right? So, those come first），把 $W$（标 `W`）延后（you can defer the W's until later），塞进计算的空隙里（do it when you have a gap in your computation），这样 `you can basically almost all completely fill up the pipeline`。

![Zero bubble pipelining：把 backward 拆成 B（回传激活）与 W（权重梯度）两部分，及 1F1B 与 ZB-H1/ZB-H2 调度对比](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0030_00-37-56.jpg)

幻灯片右侧给了两张调度对比图。`Figure 2: 1F1B pipeline schedule.` 是四行 `Device 1`–`Device 4` 的经典 1F1B 时序，图例为 `Forward`（蓝）、`Backward`（橙）、`Optimizer step`（浅黄），能看到大量白色空格。`Figure 3: Handcrafted pipeline schedules, top: ZB-H1; bottom: ZB-H2` 则把 backward 拆成三色：`F`（蓝）、`B`（浅绿）、`W`（深绿）、`Optimizer step`（浅黄），格子里填满编号，白色空格几乎消失。讲师评价 `this zero bubble pipelining thing is very complicated, or much more complicated than you would normally like to deal with`，但 `it allows you to almost deal with these like pipelining issues almost entirely, depending on, you know, essentially the workload involved in the W versus the B and so on and so forth`。也就是说，效果取决于 $W$ 与 $B$ 的工作量比例：只有当 $W$ 的耗时足以填满 $B$ 留下的空隙时，才能真的把气泡吃干净；若 $W$ 太轻，仍会剩空档。他忍不住赞叹 `Oh, this is this is so clever. Such clever things that you can do um with systems.`

收尾转向新的类比：`pipeline parallel is like depth. So you have your depth, you cut up your network along the depth dimension`，但我们有两个可扩展的轴——depth 与 width（we got a depth axis, and we have a width axis），于是自然要问「沿 width 切会怎样」（What about width?）。这就是 tensor parallel（张量并行），Percy 之前已讲过。核心想法简单：`matrix multiplies can kind of be cut up into smaller matrix multiplies, which you then add the partial sums back together`，而且 `This is the really the same idea as tiling, as well. So, this core primitive just appears many, many times`。幻灯片 `Model parallel along the width axes` 上方两行是 `Are there model parallelism schemes with better utilization?` 与 `We can think of pipeline parallel as cutting up along depth. What about width?`；中间画了 $X \cdot A = Y$ 的矩阵切分：$X$ 切成 $X_1$、$X_2$，$A$ 切成 $A_1$、$A_2$，得到 $X_1 \cdot A_1 = Y_1$、$X_2 \cdot A_2 = Y_2$，再加号相连还原 $Y$；底部一句 `Simple matrix multiplication observation: decompose into submatrices, add partial sums`。写成公式即：

$$Y = X A,\qquad X = \begin{bmatrix}X_1 \\ X_2\end{bmatrix},\ A = \begin{bmatrix}A_1 & A_2\end{bmatrix} \ \Longrightarrow\ Y = X_1 A_1 + X_2 A_2$$

也就是把 $A$ 沿列切、$X$ 沿行切成两片，各自算出部分积再求和。与前后内容的关系：这一句是本段的落点，也是下一段张量并行／序列并行的开场——沿 width 切没有 pipeline bubble，概念更简单，但代价是每个 matmul 都要伴随 activation 大小的 all-reduce，通信量远大于流水线的 $b s h$ 点对点。易错点：图中 $X_1$、$X_2$ 是沿「行」切 $X$，而 $A_1$、$A_2$ 是沿「列」切 $A$，两者切分维度必须对齐才能把部分和相加；把它和 pipeline 的「沿 depth 切层」混为一谈，是初学者最常见的混淆。

## 张量并行的子矩阵切分与前向、反向对偶 [40:05 - 41:31]

本段从最具体的 MLP 计算图切入，解释张量并行（tensor parallel）究竟把什么切开了。未做任何并行时，一个 MLP 的串行计算是：输入 $X$ 从幻灯片最右侧进入，先与矩阵 $A$ 相乘，施加逐元素非线性 GELU，再与 $B$ 相乘做 down projection（降维投影），最后输出 $Z$。幻灯片把这两步写成 $Y = \mathrm{GeLU}(XA)$ 与 $Z = \mathrm{Dropout}(YB)$，这是理解后续所有切分的基础恒等式。张量并行的做法不是把整层搬到另一张卡，而是把 $A$ 按列切成两块、把 $B$ 按行切成两块，写成 $A = [A_1, A_2]$ 与 $B = \begin{bmatrix}B_1 \\ B_2\end{bmatrix}$，再把这些子矩阵分配到不同 GPU 上，让计算并行发生，最后把结果"收拢"在一起。幻灯片下方的说明原文是 "Assign columns (A1, A2) and rows (B1, B2) to separate GPUs."

之所以能这样切，关键在于 GELU 是逐元素（elementwise）非线性：把 $XA$ 按列拆成 $XA_1$ 与 $XA_2$ 之后，$\mathrm{GeLU}$ 可以各自独立作用，即 $\mathrm{GeLU}(XA) = [\mathrm{GeLU}(XA_1),\ \mathrm{GeLU}(XA_2)]$，这一步完全不需要通信。而第二个矩阵乘按行切后，每个 GPU 只算出部分和 $Y_1B_1$ 与 $Y_2B_2$，必须把它们相加才能得到完整的 $Z = Y_1B_1 + Y_2B_2$，这正是通信发生的位置。图里用 $f$ 标在输入分叉处、用 $g$ 标在两段计算的汇合处：左侧 $X \to f \to$ 两份复制的 $X \to XA_1$/GeLU 与 $XA_2$/GeLU $\to g$；右侧 $Y_1B_1$、$Y_2B_2 \to g \to$ Dropout $\to Z$。把这段逻辑直译成示意代码，可以看得更清楚：

```python
# 两张 GPU 上的张量并行 MLP，前向传播示意
# GPU0 持有 A1（列切）与 B1（行切），GPU1 持有 A2 与 B2
Y1 = torch.nn.functional.gelu(X @ A1)        # GPU0，无通信
Y2 = torch.nn.functional.gelu(X @ A2)        # GPU1，无通信
Z_partial_0 = Y1 @ B1                        # 部分和
Z_partial_1 = Y2 @ B2                        # 部分和
Z = all_reduce(Z_partial_0 + Z_partial_1)    # g: all-reduce 求和
Z = dropout(Z)
```

真正需要记住的是前向与反向之间一种"对偶翻转"的关系，幻灯片用两条 bullet 原文给出："In the forward pass, f is the identity, and g is an all-reduce." 以及 "In the backward pass, f is an all-reduce, g is the identity." 也就是说，前向时输入 $X$ 只是被复制两份（$f$ 是恒等映射 identity），在并行组件末端用 $g$ 做一次 all-reduce（全归约）把两份结果加起来；反向传播时，偏导数从后往前流，$g$ 变成 identity，而 $f$ 处反而要把各 GPU 上累加出来的偏导数求和，因此 $f$ 变成 all-reduce。讲师强调 "this duality is kind of important if you're going to write tensor parallel"，意思是自己实现张量并行时，如果照抄前向的通信布局去写反向，梯度就会缺一次求和，模型会静默地算错。

这一章与前文的流水线并行（pipeline parallel）形成对照：流水线并行切的是"层与层的边界"，张量并行切的是"单个矩阵乘的内部"，因此两者可以叠加使用。它也为下一章埋下问题——既然有列切和行切两种切法，一个完整的 Transformer 块里到底哪些层该用哪种切法？容易踩的坑有两个：一是误以为任何非线性都能无通信地按列切，实际上若换成依赖整行统计量的操作（例如跨被切维度的 softmax 或 layer norm），切分后就必须补通信；二是把 $f$、$g$ 在前向里的角色直接搬到反向，忽略对偶翻转。

![张量并行把矩阵 A 按列切为 A1、A2，把 B 按行切为 B1、B2 分到不同 GPU 的 MLP 计算图，标注前向中 f 为 identity、g 为 all-reduce](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0032_00-40-42.jpg)

## Transformer 块内的列切、行切与全复制 [41:31 - 42:22]

把上一章的切法推广到整个 Transformer 块，就得到 "Row vs Column tensor parallel" 这张幻灯片，它并列画了 (a) MLP 与 (b) Self-Attention 两套图。讲师给出的答案是三条原文 bullet："Columnwise – QKV, up-projection"、"Rowwise – Attention output, down-projection"、"Replicated – norms, routers, etc"。换句话说，在每个 transformer block 里，MLP 的输入侧投影（up-projection）和 attention 的 QKV 投影按列切（column-wise cuts），因为它们处在"第一阶段"；而 MLP 的 down projection 和 attention 的输出投影按行切（row-wise cuts），处在"对应的第二阶段"。这样一进一出恰好能拼回完整形状，残差连接才不会对不上维度。

Self-Attention 那一半图给出了更细的独有信息：模块写成 $Y = \mathrm{Self\text{-}Attention}(X)$ 与 $Z = \mathrm{Dropout}(YB)$，并明确标注 "split attention heads"，把三组投影按 head 拆成 $Q = [Q_1, Q_2]$、$K = [K_1, K_2]$、$V = [V_1, V_2]$。第一张 GPU 拿 $Q_1, K_1, V_1$ 各自经过 Softmax 与 Dropout 得到 $Y_1$，第二张 GPU 同理得到 $Y_2$；随后分别乘 $Y_1B_1$、$Y_2B_2$，在 $g$ 处求和并过 Dropout 输出 $Z$。这与上一章 MLP 的结构完全同构，只是把"按列切矩阵 $A$"换成了"按 head 切 $Q/K/V$"——直觉上，attention head 本来就是彼此独立计算的，把它们整体分配给不同设备是最自然的切分粒度，而且不需要在 head 内部再做通信。

第三类层则要完全复制。讲师点名 layer norm、非线性（non-linearity）以及 MoE 里的 routers 这类"小层"（small layer），原话是 "those are all fully replicated across the machines. You don't really want to bother with the overhead of cutting those guys." 原因是这些操作的计算量极小、但都依赖完整的隐藏维度或完整的序列统计量；一旦切开，为同步统计量而引入的通信开销远大于省下的那点计算与显存，得不偿失。幻灯片把它简写成 "Replicated – norms, routers, etc"。

从整段结构看，这一章把上一章抽象的 $f/g$ 对偶落到了具体的层类型上，同时也直接解释了后面显存章节里那个顽固的 $10\,sbh$ 项：因为 layer norm、dropout 和残差输入是被复制的，它们的激活显存不会被张量并行切小。一个常见易错点是把列切与行切随意配对——列切的输出维度是原维度除以并行度，必须紧接一个按行切的第二矩阵乘才能相加还原；若两个连续层都用列切却没有 all-gather，形状就会错。另一个易错点是误以为"复制"意味着这些层不占显存，实际上它们占的是激活显存，且会被完整复制 $t$ 份。

![Row vs Column tensor parallel 幻灯片，对比 MLP 与 Self-Attention 的列切、行切与复制策略](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0033_00-41-57.jpg)

## 何时使用张量并行：节点内 8 卡边界、TPU mesh 与流水线并行的取舍 [42:22 - 44:56]

明确了"怎么切"，接下来要回答"什么时候值得切"。讲师给出的判断是：张量并行极其"吃通信"（communication hungry），因为每遇到一次 matmul 就要伴随一次通信——前向在 $g$ 处一次 all-reduce，反向在 $f$ 处又一次 all-reduce，而且通信量是 activation 大小、发生得"非常频繁"。因此结论是只在节点内（within a node）使用：GPU 场景里 8 张卡通过高速互联紧密地装在同一个机箱里（"eight GPUs are going to be networked tightly together in a single box"），所以 "up to eight, you might be willing to do some tensor parallel"；一旦超过 8 就要跨节点（cross-node），那些 interconnect 慢得多，性能会在跨过单机边界的那一刻出现明显跳水（"a big drop in performance"）。记住的口诀是：张量并行要放在手头最快的互联上。

"When do we tensor parallel?" 这张幻灯片用一张柱状图把这个边界量化了。副标题原文是 "On GPUs, tensor parallel within a node (up to 8 GPUs) due to high speed interconnects."，图名 "Throughput Scaling with TP (3B Model)"，纵轴是 Tokens/sec/GPU（刻度 0、5k、10k），横轴是 Tensor Parallelism (TP)，取值为 2、4、8、16、32。柱子从约 13k 逐级下降，粉色标注给出相对跌幅：TP=4 为 -10.8%，TP=8 为 -12.2%，TP=16 为 -42.7%，TP=32 为 -65.6%。也就是说 8 以内每 GPU 吞吐只是缓慢下滑，一旦到 16、32 就崩塌式下跌——这正是"单机 8 卡"经验边界的实证依据。

讲师随后回扣课程前面关于 TPU 的讨论：TPU 没有"快的 8 张卡 + 慢的跨机"这种区分，"You just have this big mesh"，一整块大 mesh 在非常规则的通信模式下能提供很高带宽，因此 "you can tensor parallel very large numbers compared to the GPU world"。这带来一个系统层面的结论：在 TPU 与 GPU 之间做并行方案移植时，张量并行与流水线并行各自占的比例会相当不同，不能照搬 GPU 上 TP≤8 的配方。

紧接着是与流水线并行的正面比较幻灯片 "Tensor parallel – pros and cons vs pipeline parallel"。Pros 一侧原文为 "no bubble. If your network is fast enough, there's no waiting for others."，并列出 "low complexity – simple to 'wrap' models without major infra changes" 与 "doesn't need large batch sizes to work well"：张量并行没有流水线气泡，网络够快就能满利用率，实现上只是"切开一些 matmul"、包裹模型时不需要大规模改造基础设施，也不像流水线那样依赖大 batch 才能压住气泡。Cons 一侧是 "much larger communication than pipeline parallel"，并给出两个通信量公式做对照：流水线是每个 microbatch 的 $bsh$ 点对点（point-to-point）通信；张量并行是每层 $8\,bsh\left(\frac{n_{\text{devices}}-1}{n_{\text{devices}}}\right)$ 的 all-reduce 通信。写成便于比较的形式：

$$
C_{\text{pipeline}} = b\,s\,h \quad \text{(point-to-point, per microbatch)}
$$
$$
C_{\text{tensor}} = 8\,b\,s\,h \cdot \frac{n_{\text{devices}}-1}{n_{\text{devices}}} \quad \text{(all-reduce, per layer)}
$$

转述里的说法是：原来是 $B \times S \times H$ 的点对点通信，现在变成大约 $8 \times B \times S \times H$ 的一次 all-reduce，而且 "this is not even just point-to-point. This is all-to-all communication that needs to happen"。幻灯片底部结论句为 "Use tensor parallel whenever we have low-latency, high-bandwidth interconnects"，其余情况（"Every other time"）大概都应该用流水线并行。易错点在于：把 "no bubble" 误读成"零通信"，忽略 $\frac{n_{\text{devices}}-1}{n_{\text{devices}}}$ 虽然随设备数趋近 1、但通信次数随层数线性累积；或者把 TP 度盲目设到 16、32 还期待线性扩展。这一章承上启下：前面几章都在讲计算与通信，从下一章起视角转向显存账本。

![When do we tensor parallel 幻灯片中的 3B 模型吞吐量随 TP 度变化的柱状图，TP 取 2、4、8、16、32 并标注相对下降百分比](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0034_00-43-13.jpg)

## 动态显存剖析与每层激活内存公式 [44:56 - 47:51]

这一章把话题从通信切到显存，讲师先拆解一个很普遍的误解：深度学习里最朴素的显存观是"显存就是参数"，也就是图里那块绿色的 $PARAMETER$；稍微进阶一点会加上此前讲过的 optimizer state（优化器状态，黄色块）。但 "if you look at this plot"——这是一张用 PyTorch profiler 实际采集的显存曲线——"this is not really the whole story"。标题原文 "A final complexity – memory is dynamic!"，副标题 "Memory isn't just the static bits, but also activations! This can be big"。

图的图例一共八项：$PARAMETER$（绿）、$OPTIMIZER\_STATE$（黄）、$INPUT$（黑）、$TEMPORARY$（浅紫）、$ACTIVATION$（红）、$GRADIENT$（蓝）、$AUTOGRAD\_DETAIL$（浅蓝）、$Unknown$（灰）。纵轴是 Memory (GB)，横轴是 Time (ms)，刻度延伸到约 250 ms；图上方标注 "Max memory allocated: 0.53 GB" 与 "Max memory reserved: 0.59 GB"。红色激活区形成一个个明显的鼓包（"these big red bumps"、"this big red hump"），随前向传播逐层累积；蓝色梯度区在反向扫描（backward sweep）时叠加进来。讲师特别指出峰值时刻："maximum memory usage happens a little bit after the maximum activation point"，也就是刚开始反向、梯度开始生成、但大量激活还没来得及释放的那一段——此时既要存激活又要存梯度，才是真正的最坏情况。因此想省显存，"maybe we need to deal with this big red hump, this activation"。

再看更大尺度的趋势：在中等偏长的序列长度下，随着模型参数变大，baseline 列里的 activations 会 "dwarf the parameter memory"（远远盖过参数显存）。讲师明确说 "any sort of memory saving strategy has to sort of reason about activations in order to be fully effective"，只处理参数显存是不够的。这就把 ZeRO/FSDP 一类切参数、切优化器状态的方案与本段要讲的切激活方案区分开了。

于是给出一个"如果什么都存"的起点公式，幻灯片 "What's the activation memory per layer?" 的标题下写着 "Starting point: activation memory needed if storing everything"：

$$
\text{Activations memory per layer} = s\,b\,h\left(34 + 5\frac{a s}{h}\right)
$$

变量定义表逐项给出：$a$ = number of attention heads（注意力头数）、$b$ = microbatch size、$h$ = hidden dimension size（隐藏维度）、$L$ = number of transformer layers、$p$ = pipeline parallel size、$s$ = sequence length、$t$ = tensor parallel size、$v$ = vocabulary size。两条 bullet 原文解释了那个奇怪的第二项："The $5\frac{as}{h}$ terms come from the quadratic attention terms incl dropout" 与 "As with flash attention, we can drop this term via recomputation"。也就是说 $5\frac{as}{h}$ 来自 attention 的二次项（含 dropout），可以通过重计算（recomputation）像 FlashAttention 那样丢掉——注意 $s\,b\,h \cdot 5\frac{as}{h} = 5\,a\,b\,s^2$，它其实是 $s^2$ 级别的，序列一长就会主导，别把它误当成随 $s$ 线性增长的普通项。

讲师强调 $SBH$ 这个乘积是"根本性的"（"kind of fundamental"）：每个序列位置要存东西、每个 batch 元素要存东西、每个 hidden 维度也要存东西，因此后面所有削减激活显存的公式里都会反复出现 $SBH$ 项；$34$ 是每层的经验常数，$24$ 来自 MLP、$10$ 来自点式操作，这个拆分要到下一章的张量并行公式才显出来。用代码把这个估算固化下来，便于手算模型能否装进一张 GPU：

```python
# 每层激活显存粗估（字节），bf16 下每元素 2 字节
def activation_bytes_per_layer(s, b, h, a, bytes_per_elem=2):
    return s * b * h * (34 + 5 * a * s / h) * bytes_per_elem
```

与前后内容的关系很直接：前三章讨论的都是"计算怎么切、通信多贵"，从这里开始讨论"切了之后显存省多少"。易错点除了上面提到的 $s^2$ 项，还有把显存峰值位置想成前向结束时刻（实际在反向早期），以及以为 optimizer state 永远是大头——在大模型长序列下真正的大头是激活。

![PyTorch profiler 截取的训练显存随时间变化堆叠面积图，含参数、优化器状态、激活、梯度等图例与峰值标注](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0036_00-45-43.jpg)

![What's the activation memory per layer 幻灯片，给出每层激活内存公式 sbh 乘 34 加 5as 除以 h，并附变量定义表](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0037_00-46-58.jpg)

## 张量并行下的激活内存、序列并行与完全线性下界 [47:51 - 53:21]

现在把上一章的 $34 + 5\frac{as}{h}$ 放到张量并行里逐项压缩，兑现前面"模型并行能减少激活"的承诺。"Activation under tensor parallel" 幻灯片给出：

$$
\text{Activations memory per layer} = s\,b\,h\left(10 + \frac{24}{t} + 5\frac{a s}{h t}\right)
$$

幻灯片正文写着 "Tensor parallel splits out the matrix multiplies in attention + MLP"，并明确拆开那个残留的 $10$："The remaining 10 term is for the LayerNorm (4sbh), Dropout (2sbh), and inputs to the attention and MLP (4sbh). These terms alone will continue to grow with size"。对照上一章就能看懂系数来源：$34 = 24 + 10$，$24$ 是 MLP 部分，被 $t$ 除成 $\frac{24}{t}$；attention heads 的二次项 $5\frac{as}{h}$ 被除成 $5\frac{as}{ht}$；而 LayerNorm 的 $4sbh$、Dropout 的 $2sbh$、以及 attention 与 MLP 输入（作为残差要留给反向）的 $4sbh$ 合计 $10sbh$，因为张量并行不切 layer norm、也不切这些点式操作，所以完全不随 $t$ 缩小。讲师用一句带玩笑的话点出后果：就算 "a thousand different GPUs in tensor parallel cuz you're Google"，"you're still going to suffer a 10x SBH penalty"——这个固定惩罚会一直存在。

处理这 $10\,SBH$ 的就是序列并行（sequence parallel）。讲师先纠正命名："This is an extremely misleading name"，真正更配得上这个名字的是 context parallel；序列并行实际做的是：剩下的这些项都是"很轻量、计算量小"的点式操作（"pointwise ops over the sequence"），于是把它们沿 sequence 轴（序列轴）而不是 hidden 轴切分，分到各加速器上，并在每次需要它们的前后引入 all-gather 与 reduce-scatter 来按需物化。两张 "Making memory truly linear – sequence parallel" 幻灯片画出了完整数据流：左右两端是绿色的 All-Gather 与 Reduce-Scatter 竖条，中间依次是 LayerNorm → Self Attention → Linear → GeLU → Linear → Dropout，虚线框交替标注 Sequence Parallel 与 Tensor Parallel 区域，$g$ 与 $\bar{g}$ 分别标在通信点上，残差用 $\oplus$ 表示。底部 "Observation: all the 10sbh terms are pointwise ops over the sequence"、"… so split up the layer norm/dropout layers along the sequence axis."，以及两条对偶 bullet："In the forward pass, 'g' is an all gather, 'ḡ' is reduce-scatter" 与 "In the backward pass, the two are reversed."

讲师指出这与 FSDP 在概念上非常像：前向需要激活、反向扫梯度时也需要激活，"But we don't need them now"，所以平时以切分（sharded）形式存着、用到时才 gather 出来，用完即弃——"conceptually very similar to the FSDP style idea"。区别在于 FSDP 切的是参数，序列并行切的是激活，且切分轴是序列而非参数维度。这又是一次前文对偶性的复现：前向里 $g$ 是 all-gather、$\bar{g}$ 是 reduce-scatter，反向里两者互换。

把各种配置组合起来，就得到 "Making activation memory fully scale with more machines" 幻灯片里那张关键表格（副标题 "Putting it together to get full linear scaling for memory."），表头为 Configuration 与 Activations Memory Per Transformer Layer，五行依次是：

| Configuration | Activations Memory Per Transformer Layer |
|---|---|
| no parallelism | $s b h\left(34 + 5\frac{a s}{h}\right)$ |
| tensor parallel (baseline) | $s b h\left(10 + \frac{24}{t} + 5\frac{a s}{h t}\right)$ |
| tensor + sequence parallel | $s b h\left(\frac{34}{t} + 5\frac{a s}{h t}\right)$ |
| tensor parallel + selective activation recomputation | $s b h\left(10 + \frac{24}{t}\right)$ |
| tensor parallel + sequence parallel + selective activation recomputation | $s b h\left(\frac{34}{t}\right)$ |

可见序列并行把那个不随 $t$ 缩小的 $10$ 也除成了 $\frac{10}{t}$，于是 $10 + \frac{24}{t}$ 合并成 $\frac{34}{t}$，实现"对并行度完全线性"（fully linear dependence）；再对 attention 做选择性重计算（selective activation recomputation）丢掉 $5\frac{as}{ht}$ 项（因为那只是 softmax 的存储），就得到常见下界 $s b h\left(\frac{34}{t}\right)$，即 $\dfrac{34\,s b h}{t}$。讲师说这是 "kind of the lower bound of what you can achieve reasonably speaking for normal training for activation memory"，手算模型能否装进 GPU 时至少要留这么多激活显存，再加上 optimizer state、parameters 与 gradient 的份额，就有了总显存的粗略上界感。

段末还插入了一段课堂问答，信息很实用：有学生问 $24$ 那一项能否也靠重计算消掉，讲师回答 "24 over key is MLP part of this"、"MLP part cannot be optimized through the recomputation"，但补充说技术上 "You can recompute the MLP as well. Like you can do a lot more recomputation than what is listed here."，只是 "recomputation beyond what is listed here is very computationally expensive"，因为要在反向里把整个 MLP 再跑一遍，"which you probably do not want to do"；相比之下 "Recomputation for attention is generally cheaper"，一来可以 tile-wise（分块）地做，二来 "you don't want to pay the quadratic cost"。易错点集中在这几处：以为张量并行能把激活线性压到接近零（$10\,SBH$ 压不掉）、把 sequence parallel 与 context parallel 混为一谈、以及把重计算当成免费午餐。这一章把显存公式从"起点"一路推到"下界"，是整段最密集的定量部分；下一章开始转向 MoE 场景的专家并行。

![Activation under tensor parallel 幻灯片，给出 TP 下每层激活内存公式 sbh 乘 10 加 24 除以 t 加 5as 除以 ht](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0038_00-48-13.jpg)

## 专家并行：为何优先于张量并行与低延迟 all-to-all dispatch [53:21 - 57:58]

讲师在这里停了一下，说明流水线并行与张量并行都已讲完，"before I move on to expert parallel"，接着引入"标准并行工具箱的最后一个 ingredient"——专家并行（expert parallelism, EP）。动机是 MoE 如今已是标配（"most big models are MOEs these days"），而 MoE 带来的一个额外好处正是 EP。讲师把 EP 类比成张量并行：都是把 MLP / FFN 组件以各种方式切开、都要付出通信代价，"It's in the same way that you're maybe splitting up the matrices, except here you're splitting up the whole MLP across different devices"；系统行为也"roughly like tensor parallel"，都是高带宽的并行原语、都能减少激活。但有一条很强的实践结论：做 MoE 时 "you almost always prefer using an expert parallel sharding strategy over tensor parallel"。

"Why EP?" 幻灯片把依据摆得很完整。副标题是 "EP is roughly like TP in behavior for MLPs – high bandwidth, reduces activation"，中间嵌入一张来自 NVIDIA Megatron 并行指南的黑底截图，标题 "Guideline 4: Prefer EP over TP for Expert Layers"，以 EP Advantages / Details 两列表格列出四条：Better GEMM efficiency → "Larger local matrix sizes improve GPU utilization"；Lower communication → "EP has less communication overhead than TP for MoE layers"；Simpler computation graph → "Easier to overlap communication with computation"；Token permutation → "When EP = num_experts, local token permutation is eliminated"。表里还给出一个具体例子："Example: For Mixtral 8x7B, EP8×TP1 outperforms EP4×TP2." 幻灯片底部一句 "But, splitting matmuls can reduce efficiency vs routing activations" 则点出张量并行侧的短板。讲师的口头解释与之一致：把矩阵切得太细，"the matrices are going to start to get small, and your GPU utilization will suffer"，而 matmul 越大越好；MoE 层里路由稀疏的 token 激活（sparse token activations），比路由稠密的大张量并行 matmul 激活容易得多；还可以把 token 精确送到它需要的设备、跳过某些开销；既然本来就是 MoE，"you might as well have the number of experts sort of split up over all the devices"。

但"EP 很好、大家都该用"是错的印象。讲师明确说 "expert parallel is still really complicated and still very difficult"，并坦言 FSDP 之后讲的所有东西——张量并行、流水线并行——"systems-wise is quite non-trivial to get working really well"。他没有展开库的内部细节，而是点名两个库来说明这件事的工程量：DeepSeek 的 DPP（讲师称 "one of my favorite things from the Deep Seek V3 era"，是他偏爱的 V3 时代产物），是自家做 EP routing 与 dispatching 的库，"looking at really low-level GPU networking primitives"，目的是把不同操作合并、或让硬件直接承担某些操作，让 MoE 路由尽可能高效；NVIDIA 的 Hybrid EP 则是"类似的、非常高效的低层硬件实现"的 EP dispatching 努力。

配套的 "Complexity – doing 'EP right' is hard" 幻灯片把难点具象化了：大标题下两行 "EP requires somewhat frequent all-to-all dispatches" 与 "Systems work to optimize this is hard!"。左侧黑底是 DeepEP 的说明全文，值得逐条转述："DeepEP is a communication library tailored for Mixture-of-Experts (MoE) and expert parallelism (EP). It provides high-throughput and low-latency all-to-all GPU kernels, which are also known as MoE dispatch and combine. The library also supports low-precision operations, including FP8."；接着 "To align with the group-limited gating algorithm proposed in the DeepSeek-V3 paper, DeepEP offers a set of kernels optimized for asymmetric-domain bandwidth forwarding, such as forwarding data from NVLink domain to RDMA domain. These kernels deliver high throughput, making them suitable for both training and inference prefilling tasks. Additionally, they support SM (Streaming Multiprocessors) number control."；最后 "For latency-sensitive inference decoding, DeepEP includes a set of low-latency kernels with pure RDMA to minimize delays. The library also introduces a hook-based communication-computation overlapping method that does not occupy any SM resource."。右侧流程图里能读到 Router、Local routing map、Global routing map、Allgather、Metadata Preprocessing、Attention、dispatch (Permute)、Expert MLP、combine、combine W、dispatch 等标签，并划出 Transformer Layer 与 Forward Pass / Backward Pass 的分界；下方两张 stream 时序图对比 "Traditional overlapping with communication SMs"（Stream 0: Attention 0 → Dispatch 0 → MoE 0 → Combine 0 → Attention 0；Stream 1 同理）与 "Overlapping without communication SMs → Faster computation with more SMs"（把 Dispatch 藏进 Attention 里、"with background RDMA"、把 MoE 0/MoE 1 连续排布），直观展示省下通信 SM 资源后算力更满。

讲师给出的根因是：MoE 的通信模式是大量 all-to-all dispatch，每个 MLP 都得把 token 路由到各种不同地方，"and you need to do this in a very latency-sensitive way because your computation is waiting for your tokens to arrive"，所以降低 dispatch 延迟 "extremely extremely important"。他还补了一则趣闻：DeepSeek 团队为了榨干性能，找出并使用了一批未公开的 PTX（"like GPU machine code"）指令来进一步加速网络通信，"So really this is like the level of stuff that you kind of need in order to get to the frontier of parallelism and efficiency"。易错点在于把 EP 当成 TP 的即插即用替代、低估 all-to-all 的延迟敏感性，或把 DPP / Hybrid EP 当成普通集合通信封装——它们其实是贴着 NVLink、RDMA、SM 占用、FP8 低精度和 PTX 指令做的系统级优化。这一章承接前面"切激活"的主线，把它平移到 MoE；下一章则讨论这些原语彼此组合时的额外约束。

![Why EP 幻灯片，含 Megatron 的 Guideline 4 截图，列出专家并行优于张量并行的四条理由与 Mixtral 8x7B 配置示例](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0043_00-54-29.jpg)

## 并行原语的组合约束与 attention / MLP 解耦 [57:58 - 60:07]

到目前为止，所有并行原语都是"孤立地"讲的，隐含假设是它们能像乐高积木一样随意拼装（"you can just kind of like put them together like LEGO blocks"）。讲师说这对绝大多数策略确实成立，比如 data parallel 与 tensor parallel 很容易组合；但专家并行 "has sort of additional constraints that you want to be somewhat respectful of"。第一种约束来自朴素组合方式：许多老库里 "the sort of replicas for DP and EP are the same"，举例说若 data parallelism = 8，就把 8 个 expert 切到这 8 个 replica 上，数据也按这 8 个 replica 切分再路由——"you split your GPUs according to DP and then your EP is sort of a subset of that split"。这种做法因为要路由 token 而显得自然，但它给 EP 的并行度设了上限，也约束了 DP 与 TP 之间的交互（"there's a maximum bound to how far you can parallelize with EP and it constrains sort of how DP and TP interact"）。

"Complexity – combining EP and others" 幻灯片把这句话写得更精确："Naively – we can compose EP and all other strategies.."、"But there are important notes – DP usually shares replicas with EP splits (so EP<DP)"、"DP and TP can interact badly to lower utilization"。下方 Fig. 8 用四个子图并列画出多种 MoE 并行组合：(a) Data + Expert Parallelism，含 GPU1/GPU2、输入 $X_1$/$X_2$、Self-Attention、Gate、Encode、All-to-All Dispatch、FFN1/FFN2、Decode、All-to-All Combine、Add + Normalize、输出 $Y_1$/$Y_2$；(b) Data + Expert + Tensor Parallelism，画出 FFN1–FFN4、Gate、Self-Attention 与 GPU1–GPU4；(c) Data + Expert + Pipeline Parallelism，画出 FFN11、FFN12、FFN21、FFN22、SA1、SA2 与 GPU1–GPU4；(d) Expert + Tensor Parallelism，画出 FFN1–FFN4、Gate、Self-Attention 与 GPU1/GPU2。图注原文为 "Fig. 8. Schematic depiction of diverse parallel strategies for MoE. For clarity and conciseness, this illustration omits some All-to-All, All-Reduce, Point-to-Point communication within parallelism, and Normalization, Encode, Decode, Gate in subfigures (b), (c), and (d)."——也就是说图里省略的通信与子模块恰恰是真实系统里最麻烦的部分。

第二种约束更根本，来自 MoE 的结构本身。"Complexity: Decoupling attention / expert parallelism" 幻灯片开头 "Recall: MoEs apply to the MLPs, not the attention (with some exotic exceptions)"，紧接着 "This can lead to some imbalance between attention and MLPs for parallelism"，并用两条 bullet 把矛盾摊开："High TP – useful for attention parallelization (since we cant use EP)" 与 "Low TP – useful for MLP parallelization (we'd rather EP than TP)"。讲师的口头推导是：MoE 只改 MLP、完全不改 attention，所以 EP 对模型的作用是"不均匀的"（"applies unevenly to the model"）——在 expert 层并行时，MLP 被并行化了，attention 却没有；想并行化 attention，"maybe I want to also use tensor parallel"，需要较高的 TP 度来切开 attention 的矩阵，但高 TP 又会把 MLP 的矩阵切得过碎、与 EP 冲突。转写在 60:03 处的 "if I have high…" 就此截断，属于本段边界，不作猜测。

幻灯片给出的对策是更灵活的并行策略，"What to do? More flexible parallelism strategies (i.e. in Megatron)"。框内原文："MoE Parallel Folding is Megatron Core's solution that decouples attention and MoE parallelism:"，下面是一张两列表：Parallelism Group 的 Dimensions 一行，Attention Layers 为 $TP \times CP \times DP \times PP$，MoE Layers 为 $ETP \times EP \times EDP \times PP$；结论句 "Separates out TP/CP/DP (for attention) and (ETP/EP/EDP) for MLPs"。也就是说 attention 层与 MoE 层各自拥有独立的并行维度组合，不再被迫共用一个 TP 度，从而把前面那个"高 TP 想要 / 低 TP 想要"的死结解开；页脚还给出了 Megatron-Core 文档链接 https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html 供查阅。

![Complexity: Decoupling attention / expert parallelism 幻灯片，含 MoE Parallel Folding 的 attention 层与 MoE 层并行维度表](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0047_00-59-30.jpg)

这一章是整段模型并行工具箱的收束。易错点有三：一是把 DP、TP、EP 当成完全正交、可任意相乘的维度，忽略 EP<DP 这类隐藏上界；二是忽略 attention 与 MLP 需要不同并行度这一结构性不均衡，直接给整个网络设一个统一 TP；三是把 MoE Parallel Folding 当成通用默认配置，而它其实是 Megatron Core 里针对 attention / MoE 解耦的特定方案。它与前文的关系是：第 1–5 章给出了张量并行、序列并行如何把激活显存压到 $\frac{34\,sbh}{t}$ 的下界，第 6 章说明 MoE 场景下 EP 通常优于 TP，本章则说明这些原语组合起来时还必须在 DP/EP/TP 的维度分配上做额外权衡。

## 解耦 attention 与专家并行：MoE Parallel Folding 与上下文并行 [60:07 - 61:59]

本段从上一段「专家并行」的结尾直接接过来。讲师在 [60:09] 用一句话点出矛盾的核心：当把 tensor parallel（TP，张量并行）和 expert parallel（EP，专家并行）都开得很高时，"I've cut up my matri- matrices into really tiny pieces, and my utilization is very bad"——矩阵被切成极小的碎片，GPU 的矩阵乘法单元利用率会急剧下降。屏幕上这张标题为 "Complexity: Decoupling attention / expert parallelism" 的幻灯片把根因写得很清楚：**Recall: MoEs apply to the MLPs, not the attention (with some exotic exceptions)**，即 MoE / EP 只作用在 MLP 层，几乎不碰 attention 层。于是同一个模型里出现了两种诉求完全相反的并行需求：attention 层因为没有 EP 可用，只能靠 High TP 来切分 QKV / 输出投影矩阵；而 MLP 层既然已经有 EP 可以把专家分到不同设备，就只想要 Low TP，甚至 "we'd rather EP than TP"，因为再叠一层 TP 只会把本已稀疏的专家矩阵切得更碎。这种 "This can lead to some imbalance between attention and MLPs for parallelism" 不是一个调参问题，而是结构性的冲突，讲师在 [60:17] 把它概括成 "you want high tensor parallel for the attention, but you want low tensor parallel for the MLPs ... there's this conflict"。

幻灯片给出的解法叫 **MoE Parallel Folding**，画面上写作 "MoE Parallel Folding is Megatron Core's solution that decouples attention and MoE parallelism"。它不再给整个网络强加一套统一的并行维度，而是为两类层各自定义一套 Parallelism Group，其 Dimensions 分别为：

$$
\underbrace{TP \times CP \times DP \times PP}_{\text{Attention Layers}}
\qquad\longleftrightarrow\qquad
\underbrace{ETP \times EP \times EDP \times PP}_{\text{MoE Layers}}
$$

也就是把用于 attention 的 $TP/CP/DP$ 与用于 MLP（MoE）的 $ETP/EP/EDP$ 彻底分开，只有 $PP$（pipeline parallel，流水线并行）作为跨阶段的公共维度保留在两边。幻灯片底部那句 "Separates out TP/CP/DP (for attention) and (ETP/EP/EDP) for MLPs" 正是对这个乘积式的文字注解；右下角还给出了出处链接 `https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html`。转写在 [60:26]–[60:56] 用口语复述了同一件事：近几年人们做出的 "more complicated system solutions" 把 attention 的 TP 和 MoE 层的 TP 解耦，"the attention layers get one kind of tensor parallel, and the MOE layers get another kind of tensor parallel"，从而能 "fully decouple the parallelism"，得到 EP、TP、DP 之间 "more complicated, but more effective combinations"。这里的 $ETP$（expert tensor parallel）与 $EDP$（expert data parallel）是画面上独有、转写里没有念出的标识符，属于本次细看幻灯片才能拿到的增量信息；注意 $CP$ 出现在 attention 一侧而 $EP/ETP/EDP$ 只出现在 MoE 一侧，两组维度并非简单换位。

这套解耦思路是前面「专家并行」一章的自然延伸，也直接预告了本段后面 [62:16] 那张 "Recap: LLM parallelism table" 大表里为什么要给 attention 与 MoE 分别记通信账。它同时引出了 [61:00] 开始的收尾话题：context parallel（上下文并行），也叫 ring attention。讲师说它的本质是 "splitting activations in a very long sequence across different accelerators"，把超长序列的激活切开分到不同加速器上，并按 TPU 的 mesh topology 以 "kind of this ring-like way" 把所需的 KV 传给下一个设备；ring attention "which I think was the original paper that did this ... showed that this worked really well on TPUs"。它既用于 long context extension 阶段，也用于 model serving（推理服务），"Both of these are used in long context extension stages and in sort of model serving"，但因为概念与前面讲过的激活切分高度重叠，讲师明确说 "I'm not going to talk about it mainly because I think it overlaps in concept to a lot of what we've talked about already"。

这里有两个容易踩的坑。其一是想当然地给全网设一个全局 TP 度——对 MoE 而言这会同时损害 attention 的切分需求和 MLP 的 matmul 尺寸，正确做法是像 MoE Parallel Folding 那样按层族拆维度，让 "What to do? More flexible parallelism strategies (i.e. in Megatron)" 落地。其二是把 context parallel 当成一个独立的、可以替代 TP/EP 的「第五种并行」；实际上转写与画面都只把它定位成长序列场景下的补充手段，后面 [77:54] 讲 Nemotron 3 Super、[75:28] 讲 Llama 3 长上下文扩展时才真正把它调大。此外幻灯片特意写了 "with some exotic exceptions"，提醒 "MoE 只作用于 MLP" 这条 Recall 本身也有例外，不能当成绝对定律；而 [60:12] 的 "utilization is very bad" 也说明解耦的动机不是显存不够，而是切得太碎导致的算力浪费，这与后面纯显存驱动的 FSDP / PP 决策是不同性质的约束。

![解耦 attention 与专家并行：MoE Parallel Folding 为 attention 与 MoE 层各设一套并行维度](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0048_01-00-50.jpg)

## 并行策略大表与「没有银弹」 [62:00 - 63:59]

[61:52] 讲师宣布 "this is the end of the second part"，并用一张 "Recap: LLM parallelism table" 把整个第二部分的并行策略汇总。他特意说明 "I've colored in red what I subjectively feel like are the drawbacks of the different methods"，用红色标出每种方法主观上最痛的缺点，目的就是传递 "there is no one strictly dominant parallelization strategies. It's all like a whole bunch of trade-offs that you somehow have to manage gracefully in order to get a good outcome"。表格共七列：Method、Comm/Sync、Param memory per rank、Activation / KV memory per rank、Main bandwidth cost、Scales global batch?、Easy to use?，行则覆盖 DDP / ZeRO-1、FSDP / ZeRO-3、Pipeline parallel、Tensor parallel、Sequence / Context parallel、Expert parallel (MoE) 六种；转写 [61:57] 补充说明 "data parallel, which are the top two rows, and the various model parallelism strategies below them"，即头两行是数据并行族，下面是各类模型并行。

把表里的显存缩放与通信量写成可对照的形式，就是：

$$
\text{Param memory per rank}:\ \text{DDP}=1,\ \text{FSDP}\sim\tfrac{1}{DP},\ \text{PP}\sim\tfrac{1}{PP},\ \text{TP}\sim\tfrac{1}{TP},\ \text{EP}\sim\tfrac{1}{EP}
$$

$$
\text{Gradient traffic}\sim O(\text{params}),\qquad \text{Param traffic}\sim O(\text{params})\ (\text{higher than DDP, overlap})
$$

逐行读，DDP / ZeRO-1 的 Comm/Sync 是 "gradient all-reduce (ZeRO-1 only opt state) per step"，红色缺点写着 **No param scaling (ZeRO-1 only opt state)** 与激活一栏的 **None**，带宽是 "Gradient traffic ~ O(params)"，Scales global batch? 是红字 **Yes, linear in DP**，易用性 Very。FSDP / ZeRO-3 是 "Gradient all reduce – can be overlapped"，参数/梯度/优化器状态 $\sim 1/DP$，激活仍是红字 **None**，带宽 "Param traffic ~ O(params), higher than DDP, overlap"，同样是红字 **Yes, linear in DP**，易用性 Moderate。Pipeline parallel 的通信是 "Activation between layers, **pipeline bubbles**"，参数 $\sim 1/PP$，激活 "Depends on pipeline buffers"，带宽 "**Activation traffic between stages**"，Scales global batch? 为 "No, but needs microbatches"，易用性红字 **Hard**。Tensor parallel 的通信是红字 **Blocking activation communication**，参数 $\sim 1/TP$ for TP-sharded weights，激活 $\sim 1/TP$ for relevant matmul activations w/ SP，带宽 "**Activation-sized collectives every block**"，易用性 Hard。Sequence / Context parallel 是 "per-layer sequence-shard exchange"，参数红字 **None**，激活 $\sim 1/SP$ or $1/CP$ on sequence-side activations / KV，带宽 "Activation / KV communication"。Expert parallel (MoE) 是 "**token dispatch - all-to-all per MoE**"，参数 $\sim 1/EP$ for expert weights only，激活 **None**，带宽 "**Token-routing all-to-all**"，且 "No, but needs enough tokens per expert"，易用性 Hard。这些单元格文字与红色标注是转写没有逐格念出、只有看图才能补齐的信息。

[62:27] 起讲师拿 FSDP 举例把红字串起来：FSDP "is great. It's a wonderful strategy"，但 "does not help you with activation"（对应表里 Activation = None），并且 "it sort of consumes your global batch size in order to do parallelization"（对应 Scales global batch? = Yes, linear in DP 反而成为限制），"So, it has limits"。要补这些短板，就得组合其它手段："maybe you want to do some tensor parallel to cut down on activation memory"，TP "also doesn't touch global batch size"，但代价是 "now you need fast networking, and you need sort of high bandwidth"；剩下的慢链路再 "maybe you want to use some of the other parallelization strategies" 去吃。这正是上一章解耦思路和下一章 roofline 分析的动机：不同策略各自占掉一种资源（batch size、激活显存、快链路带宽），必须搭配着用，才使得 "in many large-scale architectures use a large combination of these"。

还有一处值得单独说明：整张表的红字是讲师 "I've colored in red what I subjectively feel like are the drawbacks" 的主观标注，不是客观测量值，因此同一格在不同硬件、不同模型规模下可能翻性。例如 "Scales global batch? = Yes, linear in DP" 对 DDP / FSDP 反而被标红，是因为它把 global batch size 当成并行度的燃料，而 PP 的 "No, but needs microbatches"、EP 的 "No, but needs enough tokens per expert" 同样是红字，说明「不随 batch 扩展」也未必是优点，只是把约束换成了 micro-batch 数或每专家 token 数。读表时必须把 "Scales global batch?" 与 "Main bandwidth cost" 两列交叉着看：前者决定你能不能靠加卡来消化 batch，后者决定加卡之后到底是快链路还是慢链路在扛通信，两者合起来才解释了 [62:46] 那句 "maybe you want to compose them with some of these other tricks that you have at hand"。这也埋下了下一章把定性表格变成 $B/N$ roofline 的伏笔。

[63:20] 有学生问 pipeline 与 FSDP 是否仍适用于 MoE，讲师在 [63:29] 明确回答 "Yes, yes, they're absolutely applicable"，并补充标准建议里 "expert parallel is kind of like tensor parallel, so you want to go up to like eight or like, you know, eight fast connections"，但又说 "People don't really follow that anymore"——EP 到 8 只是早期经验，现代大模型未必遵守；最后强调 "FSDP and or data parallel in general, and pipeline and tensor are used extensively in all the frontier models"。易错点在于把这张表当成「打分表」去选一个赢家：红色标注恰恰说明每列的 Yes/No、$1/TP$、$1/DP$ 都有反面，例如 TP 虽不耗 batch size 却需要节点内高带宽、EP 虽省激活却要求每个专家分到足够 token、FSDP 虽易用却把 global batch size 当成燃料；单独看某一列会误判，必须连着 "Main bandwidth cost" 和 "Scales global batch?" 一起读。另一个易错处是把 "Easy to use? = Very" 当成推荐度——DDP / ZeRO-1 之所以 Very，恰恰因为它什么显存都没省（红字 No param scaling、None），这正是后面必须用 FSDP 或模型并行去补的原因。

![并行策略汇总表：六种方法在通信、显存、带宽、batch 扩展与易用性上的红字缺点对照](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0049_01-02-16.jpg)

## roofline 定量分析：计算与通信的比值 [64:00 - 66:19]

[63:58] 起讲师说并行化 "one of the things that's cool about parallelism ... is that you can do some math"：可以逐层计算不同 sharding strategy 的 compute、逐层计算不同通信策略的 communication，再看 "if I have different devices allocated to different communication strategies, how do these things scale"。把这些量画成图，横轴是 batch size，纵轴本质是 "the ratio of compute time to communications time"。核心判据在 [64:48] 讲得很直白："as long as my compute time is longer than communications time, in principle, you can just hide communications underneath your computation"。屏幕上标题为 "Model vs Tensor parallel (TPU book)" 的幻灯片把这条判据画成 roofline：左侧大字 "Key quantity: Global batch size (divided by GPU)"，右侧图名是 "Batch-size scaling behavior of parallelism strategies on a 4x4x4 mesh"。这条判据把上一章那张定性大表变成了可计算的账本——只要比值大于 1，通信就能藏在计算底下，GPU 就是满负荷的。

幻灯片左下角给出逐层 compute 与 comms（bytes，forward + backward pass）的对照表，表头注明 compute 一列 "(ignoring gating einsum)"，原文可整理为：

$$
\begin{aligned}
\text{DP}:&\quad \tfrac{4BDF}{X} + \tfrac{8BDF}{X}, && \text{comms}:\ 0 + 8DF\\
\text{FSDP}:&\quad \tfrac{4BDF}{X} + \tfrac{8BDF}{X}, && \text{comms}:\ 4DF + 8DF\\
\text{MP}:&\quad \tfrac{4BDF}{Y} + \tfrac{8BDF}{Y}, && \text{comms}:\ 4BD + 4BD\\
\text{FSDP}+\text{MP}:&\quad \tfrac{4BDF}{XY} + \tfrac{8BDF}{XY}, && \text{comms}:\ \left(\tfrac{4BD}{X} + \tfrac{4DF}{Y}\right) + \left(\tfrac{8BD}{X} + \tfrac{8DF}{Y}\right)
\end{aligned}
$$

可以看出 DP 与 FSDP 的每层计算量相同，差别全在通信项（FSDP 多出 $4DF$ 的参数 all-gather）；而加入 MP 后计算量被 $Y$ 摊薄、通信项变成与 $BD$ 相关的激活。右图纵轴是 $\text{FLOPS time}/\text{Comms time}$（对数刻度 $10^{-2}, 10^{-1}, 10^{0}$），横轴 $B/N$，即 "batch size divided by total number of chips"，范围 0 到 2000；$10^{0}$ 处的水平虚线标为 "Computation bound"，其下的橙色水平线标为 "Communication bound"，讲师在 [64:58] 描述为 "this dash line at the very top, that's kind of the boundary of where you're efficient. Anything above that is sort of fully utilizing your computation. Anything below that is on the bad part of your roof line ... You're waiting for communication"。三条曲线分别是 FSDP Only（蓝）、MP Only（橙）、FSDP + MP（绿）。图上两条竖直虚线落在约 400 与 850，并配三段注释："No scheme works when $B < 400$"、"Only mixed FSDP + MP works when $B < 850$"、"Both mixed FSDP + MP and pure FSDP work when $B > 850$"。这些阈值、曲线标签与公式表是转写里没有逐字念出、只有看图才能得到的量化信息。

[65:14] 起讲师用口语把曲线讲了一遍："If your batch size is big enough, FSDP only is good"，并给出具体例子 "if you have a batch size of 2,000 ... per chip"，此时 FSDP "is doing extremely well. You're fully compute bound"。关键在于 "as the batch size goes down, FSDP only will hit a point where you're now communication bound"；一旦掉到 roofline 下面，就 "You're waiting for communication to do your computation"。补救办法是把 MP（这里具体是 tensor parallel）掺进来，"in order to keep pushing efficiencies, you have to incorporate MP, in this case is tensor parallel, to the mix ... you can sort of push that curve out a little bit more and be compute bound even into smaller batch size regimes"，然后 "keep adding strategies to push this curve out"。讲师在 [66:05] 给这种叠加起了名字："this is what people call 3D or 4D ... I don't think there's 5D ... parallelism, which is when you put all these strategies together to keep your compute units fully utilized under different communication topologies"。换句话说，3D/4D 并行不是新算法，而是为了把 roofline 的高效区往小 batch 方向延展而做的策略堆叠。

这一章与前后内容咬合得很紧：大表说 FSDP "Activation = None"、TP "需要高带宽"，这里用 $B/N$ 和 compute/comms 比值解释为什么 batch 一大 FSDP 就够、batch 一小必须加 TP；下一章则把「叠加」落成可执行顺序。易错点有两处。一是把 "batch size" 当成绝对值，而幻灯片的关键量明确是 $B/N$（除以总芯片数后的全局 batch），设备越多单个芯片分到的 batch 越小，越容易跌破通信瓶颈，[65:24] 特意补了 "per chip" 正是此意。二是误以为叠加策略越多越好——图上 FSDP + MP 曲线只在中段胜出，超过 850 后纯 FSDP 同样落在 computation bound 之上，盲目上 4D 并行只会增加实现复杂度而不带来收益；此外 $B < 400$ 时 "No scheme works"，说明再怎么组合也救不了过小的 batch，此时该动的是 gradient accumulation 而不是并行维度。

![TPU book roofline 图：三种并行策略的计算/通信时间比随 B/N 变化的曲线与逐层公式表](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0051_01-05-08.jpg)

## 3D/4D 并行的简单配方与 Megatron 实践指南 [66:20 - 68:06]

[66:19] 讲师说前面讲了一堆 "generalities" 可能让人觉得复杂，但 "the strategy for how you parallelize is very simple, actually"。屏幕上 "'3D (4D) parallelism' – putting it all together" 这张幻灯片把它写成 "Simple rules of thumb from the literature" 的两步法：第一步 "Until your model fits in memory.."，做法是 "Tensor / expert parallel up to GPUs / machine"、"Pipeline parallel across machines"、"（Or use Zero-3, depending on BW）"；第二步 "Then until you run out of GPUs"，做法是 "Scale the rest of the way with data parallel"。末尾还有一句补充："If your batch size is small.. gradient accumulate to trade higher batch sizes for better communication efficiency." 右侧的 GPU 网格图把 Data Parallel Rank 0 / Data Parallel Rank 1 各画成三个 Pipeline Stage（Network Layers 0-7、8-15、16-23），内部用不同颜色块标出 TP、EP、CP、DP、PP 等维度如何叠在同一批卡上，是对「3D/4D」这个名字最直观的图解。

[66:31] 起的口述与幻灯片一一对应，但给出了更具体的数字：先 "until your model fits into memory, you're going to cut up your model by whatever means necessary"，切分手段是给高速互联用的 tensor 或 expert parallel，"if you have eight GPUs per machine, you're going to use a tensor or expert parallel of eight"，即单机 8 卡时 $TP=8$ 或 $EP=8$；然后 "you're going to pipeline parallel or zero three, you know, which is FSDP, the rest of the way to make your model fit"；"once your model fits, then you're going to just data parallel the rest of the way ... They used up the rest of your GPUs with data parallel"；"if your batch size ends up being too small, you can do gradient accumulation to get better GPU utilization"。讲师用 "Very simple strategy" 收尾，并在 [63:58]（紧接本段之前）预告 "you will do this or something like this as part of your assignment"，说明这是课程作业里要动手实现的东西。这套顺序把上一章 roofline 里「保持 compute bound」的目标翻译成了一条按显存→GPU 数逐步推进的贪心流程。

[67:08] 起讲师说这套思路 "is born out in the actual kind of like large-scale parallel training libraries that you might use"，并给出 "Megatron's like guide to sort of paralyzing MOEs"，强调 "this idea actually applies to the dense models as well"。画面上 "Example from Megatron's current recommendations" 一页把它拆成 The "basics" 三条与 Plus more advanced "extras" 两条，底部同样附 `https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html`。对应的命令行配置项可原样记为：

```bash
--use-distributed-optimizer              # Guideline 1: 在 DP ranks 间切分优化器状态
--num-layers-per-virtual-pipeline-stage  # Guideline 3: 控制 VPP 大小
--context-parallel-size                  # Guideline 5: 把序列切分到多 GPU
```

Guideline 1 是 "Minimize Model Parallelism, Maximize Data Parallelism"，Goal 写 "Keep TP/EP/PP as small as possible while avoiding OOM"，Why 是 "Model parallelism introduces communication overhead that hurts performance"，How 是 "Use distributed optimizer (--use-distributed-optimizer) to shard optimizer states across DP ranks, freeing memory for larger DP size"。Guideline 2 "Keep EP and TP Communication Within NVLink Domain"，Goal 是 "Ensure EP+TP fits within a single node (typically 8 GPUs)"，Why 是 "EP and TP are communication-intensive, NVLinks provides much higher bandwidth than cross-node interconnects"，Scaling 建议 "When scaling beyond one node, prefer PP over expanding TP/EP across nodes"，Note 特别点名 "For very large MoE models (like DeepSeek-V3), the EP communication may exceed the NVLink bandwidth. In this case, consider using 1F1B A2A Overlap to overlap the EP communication"。Guideline 3 "Use Pipeline Parallelism (PP) for Multi-Node Scale"，VPP 一栏写 "Enable Virtual Pipeline Parallelism to reduce pipeline bubbles when PP ≥ 2"，Config 即 `--num-layers-per-virtual-pipeline-stage`。右侧 Guideline 4 "Prefer EP over TP for Expert Layers" 列出 EP 的四点优势（Better GEMM efficiency / Larger local matrix sizes improve GPU utilization、Lower communication、Simpler computation graph、Token permutation），并给出具体例子 "For Mixtral 8x7B, EP=8+TP1 outperforms EP=4+TP2"，以及 "When EP = num_expert, local token permutation is eliminated"。Guideline 5 "Enable Context Parallelism (CP) for Long Sequences"，When to use 是 "Sequence lengths ≥ 8K tokens"，Key factor 是 "CP efficiency depends on overlapping communication with computation"，Config 即 `--context-parallel-size`。

把这几条 guideline 与 [67:25] 的口述对照，会发现顺序正好相反：讲师说 "you see exactly the thing that I said, but kind of in reverse order"，文档第一条就是 "Minimize Model Parallelism, Maximize Data Parallelism, right? so use most of your GPUs to shard data if you can"，随后才是 "stay within NVLink. So, that's one machine or so one one box"、"if you're going to go multi-node, use pipeline parallelism"、"if you're MOE, then prefer expert parallel"、"if you're doing long sequences, use context parallel, right? Ring attention"。也就是说文档从「尽量少用模型并行、先把 DP 最大化」写起，课堂则从「先切到模型装得下」讲起，两者是同一条贪心链的正反两种叙述：一个从结果往回倒推，一个从约束往前推进。讲师用 "in kind of practitioner form that you see here" 收束，正是要把前面抽象的 trade-off 落到工程文档上。

这一章把前一章 roofline 的「叠加策略」落成可执行顺序，也预告了 68:07 之后的两个课堂问答与后面所有真实案例。易错点有三处。其一，幻灯片第二步说 "data parallel the rest of the way"，但 DP 的并行度不能超过全局 batch size（第 2 章大表里 "Yes, linear in DP" 是双刃剑），所以才需要 gradient accumulation 兜底，不能以为 DP 可以无限加。其二，[68:27] 的问答里讲师补充 "sequence parallel is like often just like you know, you do it with tensor parallel in order to reduce activations. It's kind of more like an add-on ... It's not its own like standalone thing in many cases"——序列并行通常只是 TP 的附赠优化，而非可与 TP/PP/DP 并列的一个独立维度，这与「4D 并行」里把 CP/SP 当一维的直觉不同。其三，Guideline 2 的 "single node (typically 8 GPUs)" 只是当下硬件的经验值，转写紧接着就用 DeepSeek-V3 的 64-way EP、后面 [78:12] 用 Qwen 3 的 EP=32 说明这条规则正在被 MoE 时代打破；此外 Guideline 1 的 "avoiding OOM" 提醒我们这几条是按「先装进显存」排序的，若模型本就装得下，最省事的反而是纯 FSDP，正如 [72:51] 的 OLMo 案例所示。

![Megatron 当前并行推荐指南：最小化模型并行、NVLink 域、跨节点 PP、优先 EP 与长序列 CP](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0053_01-08-00.jpg)

## 课堂问答、定量扩展证据与激活重计算 [68:07 - 72:30]

[68:07] 讲师 "maybe I can pause here in case anyone's curious about like 4D parallelization stuff" 引出一串问答。第一条在 [68:27] 讨论 sequence parallel，已如上一章所述被定性为 "an add-on ... not its own like standalone thing in many cases"，即它常与 TP 绑定、只为削激活而存在。第二条在 [68:43] 有学生问 "loop transformer"（循环 / 权重共享的 Transformer）会不会让并行更容易或更难，讲师先表态 "I don't know how much I buy the loop transformer mythos rumors. Maybe it is, it'll be interesting"，再从系统角度认真回答："it might mess with things like FSDP. Like you can't really discard the weights. It does have the advantage though that like you don't have to keep getting weights. Like it is much more parameter efficient and so much maybe much of the model parallelism stuff wouldn't be as important"。关键在 [69:25] 的那句拆解：FSDP 的整个策略 "is really predicated on like get weights, discard weights, get weights, discard weights and in a loop you're not discarding anything"——FSDP（ZeRO-3）依赖「按层 all-gather 取权重、算完即释放」的循环，而 loop transformer 反复复用同一组权重，既不丢弃也不重复获取，于是 FSDP 的通信隐藏前提被釜底抽薪，很多模型并行手段反而变得不那么必要。这个问答的价值在于把 FSDP 的隐含假设显性化：它不是无条件省显存，而是建立在「权重可以反复取了又扔」这一结构之上。

[69:36] 讲师转入 "an old paper but it is actually a you know probably one of the best resources um and also it's not like the networking fundamentals changed that much. So I think you know the the lessons from here are very relevant today"。这是 Nvidia 的论文，作者含 "formerly unfortunately Stanford folks Matei and Deepak"，做法是 "did a bunch of runs of large-scale parallel training across many different configurations to basically tell you you know as you're scaling up how should your parallelization strategies change across different scales"。结论在 [70:14] 描述得很具体："you have data parallel sort of maxed out and then your tensor parallel kind of increases until it hits eight and then you stop because you don't want to go any higher and then from that point on your pipeline parallel increases and increases and increases"，即 DP 先拉满、TP 增到 8 封顶、之后一路加 PP；"with sufficiently large scale data parallel kind of decreases and at the very end you have a DP of six and this is because you just need this much tensor and pipeline parallel to fit the stuff right and you're using the rest of your budget on data parallel"。[70:46] 补充："even as you go to ludicrously large numbers of GPUs ... your utilization stays very flat and very good"，这也是 "why I think, you know, you see these gigantic data center build outs ... even cross data center training, you can get really good utilization" 的原因。这段把上一章的两步配方用真实 sweep 数据坐实了一遍。

屏幕上 "Careful '3D' parallelism gives linear gains" 一页给出了定量证据。图例是四条曲线：ZeRO-3, 175B（蓝虚线圆点）、ZeRO-3, 530B（蓝实线菱形）、PTD-P, 175B（橙虚线三角）、PTD-P, 530B（橙实线方块）。纵轴 "Achieved teraFLOP/s per GPU"，刻度 0 / 50 / 100 / 150 / 200；横轴 "Number of GPUs"，刻度 768、1152、1536、1920。橙色的 PTD-P 曲线几乎水平：530B 稳定在约 $173$，175B 从约 $150$ 缓降到 $143$；蓝色的 ZeRO-3 曲线则随 GPU 数明显下滑，530B 从约 $140 \to 100 \to 50$，175B 从约 $145 \to 88 \to 45$。图注 "Figure 10: Throughput per GPU of PTD-P and ZeRO-3 for two different GPT models (the 175B GPT-3 model is shown with dotted lines, and the 530B model is shown with solid lines). Global batch sizes are fixed and ZeRO-3 is used without any model parallelism." 下方一行结论是 "More GPUs, same, flat utilization!"。用一行式子概括这个对照，就是：

$$
\text{PTD-P} = PP \times TP \times DP:\ \text{throughput/GPU}\approx\text{const},\qquad \text{ZeRO-3 only}:\ \text{throughput/GPU}\downarrow\ \text{as GPUs}\uparrow
$$

讲师在 [71:19] 又补了两点：论文给出 "quantitative evidence showing you ... for different batch sizes, for tensor parallel ... what is the right place to stop and it's clearly, you know, tensor parallel size of eight. As you go beyond that, you're going to get into trouble"；以及 "In pipeline parallel, you also need big batch sizes in order to be able to utilize the machines effectively. That's kind of why there's a blue to orange gap ... the larger the pipeline parallel, the larger the gap between orange and blue"，即 PP 越大越依赖大 batch 才能压住气泡，这正是橙蓝曲线间距的来源。

[71:50] 起讲师抛出 "another thing that maybe you have already internalized from flash attention, but maybe not" 的反直觉观察："maybe you should do a bunch of activation recomputation because if you do activation recomputation, you know, very cleverly, then you can get a bigger batch size and the bigger batch size then allows you to get better utilization"。他把因果链讲得很完整："you should do more computation via recomputation in order to end up getting better utilization because it allows you to save memory and memory can be turned into batch size"，可写作 $\text{recompute} \Rightarrow \text{save memory} \Rightarrow \text{larger } B \Rightarrow \text{compute-bound}$。这与第 3 章 roofline 完全呼应：重计算不是目的，把省下的显存换成更大的 $B/N$、让曲线回到 computation bound 之上才是目的，[72:04] 也点明这是 "initially I think very counterintuitive observation"。易错点在于把「多算」直接等同于「变慢」——在这个显存受限的场景里，多花一点算力重算激活反而能把通信隐藏得更好；同时也别把 Figure 10 的平坦曲线当成普遍规律，它的前提是 "Global batch sizes are fixed" 且 "ZeRO-3 is used without any model parallelism"，一旦 batch 不固定、或给 ZeRO-3 也配上 MP，结论就要重新按 $B/N$ 评估，这也是为什么讲师反复强调 "careful 3D parallelism" 而非「随便堆并行」。

![Figure 10：PTD-P 与 ZeRO-3 在 175B/530B GPT 模型上每 GPU 吞吐随 GPU 数变化的对比](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0055_01-10-59.jpg)

## 现代训练案例（一）：OLMo、DeepSeek 与 Llama 3 405B [72:31 - 76:23]

[72:33] 讲师说 "the last thing I want to end with is just to look at just a very few training runs"，想通过真实训练配置看 "how modern parallelization is used in practice and how it's evolving"。第一个例子是 AI2 的 OLMo：转写里先说 "Dolma by AI2 was a 7B open source model" 随即更正 "sorry, Olmo, which was trained on the Dolma data set. And this is in the Dolma paper for the details ... was a 7B model, trained fully with FSDP across a whole bunch of accelerators. I forget exactly how many"。屏幕 "Recent LMs – what do they do?" 一页引用了论文 3.1 节 Distributed Training Framework 的原文，底部一行结论是 "Dolma – 7B model, FDSP (probably fits intra-node)"（画面上 FDSP 显系 FSDP 的笔误）。这一笔误本身也提醒读者：幻灯片上的字符串要原样辨认，但理解时要按 FSDP 读。

这段引文信息密度很高，且全是转写里没有逐字念出的画面独有内容：训练用 "the ZeRO optimizer strategy (Rajbhandari et al., 2019) via PyTorch's FSDP framework (Zhao et al., 2023), which reduces memory consumption by sharding the model weights and their corresponding optimizer state across GPUs"；"At the 7B scale, this enables training with a micro-batch size of 4096 tokens per GPU on our hardware"；"For OLMo-1B and -7B models, we use a constant global batch size of approximately 4M tokens (2048 instances, each with a sequence length of 2048 tokens)"，即 $2048 \times 2048 \approx 4\text{M}$；"For OLMo-65B model (currently training), we use a batch size warmup that starts at approximately 2M tokens (1024 instances), then doubles every 100B tokens until reaching approximately 16M tokens (8192 instances)"。精度方面："we employ mixed-precision training (Mickievicius et al., 2017) through FSDP's built-in settings and PyTorch's amp module"，softmax 等特定操作 "always run in full precision to improve stability, while all other operations run in half-precision with the bf16 format"，"the shared model weights and optimizer state local to each GPU are kept in full precision"，权重只在 "the full-sized parameters are materialized on each GPU during the forward and backward passes" 时才 cast 成 bf16，"Gradients are reduced across GPUs in full precision"。讲师据此在 [73:05] 总结："FSDP scales surprisingly well, even to like many, many GPUs, as long as you're training a small model ... I think many 7B-ish models are trained purely with FSDP"，这与第 4 章「模型装得下就别上模型并行」的配方完全一致。

[73:23] 起讲师说 DeepSeek "because of their systems, you know, skills, does do some pretty exotic and useful stuff"。V1 "was trained just with data parallel zero stage one, with tensor sequence and pipeline parallel, as I kind of talked you before"；屏幕 "DeepSeek" 一页底部把这行写成 "DeepSeek – ZeRO stage 1 with Tensor, Sequence, and Pipeline parallel"，并引用 2.4 Infrastructures：框架名 "HAI-LLM (High-flyer, 2023)"，"Data parallelism, tensor parallelism, sequence parallelism, and 1F1B pipeline parallelism are integrated into this framework as done in Megatron (Korthikanti et al., 2023; Narayanan et al., 2021; Shoeybi et al., 2019)"，还用了 "flash attention (Dao, 2023; Dao et al., 2022)"、"ZeRO-1 (Rajbhandari et al., 2020) to partition optimizer states over data parallel ranks"、把 LayerNorm / GEMM / Adam 做算子融合、"train the model in bf16 precision but accumulate gradients in fp32 precision"、"In-place cross-entropy is performed to reduce GPU memory consumption"。V3 是 MoE，"they have not only pipeline parallel, but they also have expert parallel, instead of tensor parallel"，且 "their expert parallel actually is a little bit exotic, because they have 64-way parallelism. So, they group eight different machines together, and that's their expert parallel sort of domain"，并 "use the same tricks that they have from their pipeline parallel pipelining ... to try to make sure that expert parallel doesn't sort of like have low utilization periods"。幻灯片把 V3 压缩成 "V3 – PP (16), EP (64-way, 8 nodes), ZeRO stage 1" 与 "EP uses 1F1B A2A Overlap"。[74:22] 的 Yi 则是 "zero stage one and tensor and pipeline parallel ... the classic data parallel, tensor parallel, pipeline parallel combo"，并且 "once you go to MoEs, you replace the tensor parallelism with expert parallelism ... expert parallelism is just a little bit more efficient"。

把 2.4 Infrastructures 这段引文再往下读，还能看到一批转写里完全没念、只存在于幻灯片上的工程细节。为压低等待开销，"Efforts are also made to overlap computation and communication to minimize additional waiting overhead, including the backward procedure of the last micro-batch and reduce-scatter operation in ZeRO-1, and GEMM computation and all-gather/reduce-scatter in sequence parallel"，即把最后一个 micro-batch 的反向与 ZeRO-1 的 reduce-scatter、把 GEMM 与序列并行的 all-gather / reduce-scatter 尽量重叠；"Some layers/operators are fused to speed up training, including LayerNorm, GEMM whenever possible, and Adam updates" 讲的是算子融合；"In-place cross-entropy is performed to reduce GPU memory consumption, i.e.: we convert bf16 logits to fp32 precision on the fly in the cross-entropy CUDA kernel (instead of converting it beforehand in HBM), calculate the corresponding bf16 gradient, and overwrite logits with its gradient" 则是把 logits 就地覆写成梯度来省显存，避免先在 HBM 里物化一份 fp32 logits。这些细节说明：所谓「64-way EP」不是把专家数调大就完事，而是靠 1F1B A2A Overlap、通信计算重叠、算子融合与就地交叉熵一整套系统工作撑起来的，也正好呼应了 [74:01] 讲师说的 "they basically use the same tricks that they have, um, from their uh, pipeline parallel pipelining, in order to try to make sure that expert parallel doesn't sort of like have, uh, low utilization periods"。换句话说，EP 的难点从来不是切分本身，而是切分之后别让设备空转，这一点与流水线并行里用 micro-batch 压气泡是同一种思路。

[74:49] 起讲师说 "on the other extreme of a really big dense model, you have Llama 3"，并称赞 "the Llama 3 report I like quite a bit, because they're one of the few reports that have like literally a full breakdown of their parallelism strategy, not just like in the abstract, but across all the different phases"。表格有 pre-pre-training 暖机行（"You can ignore that first row"）、主预训练行和长上下文扩展行；主预训练配置是 "Tensor parallel of eight, one sort of context parallel, 16 pipeline parallel, and 128 data parallel"，即

$$
TP=8,\quad CP=1,\quad PP=16,\quad DP=128
$$

长上下文阶段则 "crank up the context parallel, lower the data parallel"，因为这一阶段 "very memory-hungry"。讲师特别提醒 "we're in tensor parallel, not expert parallel land, because Llama 3 405B is a gigantic dense model. It's not a MoE model"。[75:52] 起他插入 infra 视角的旁注："if you, for example, become an infra engineer at one of these companies ... you'll notice is, you know, GPUs fail all the time"，画面 "Llama 3 405B / Side note – Lots of GPU failures at this scale!" 给出 Table 5：Component / Category / Interruption Count / % of Interruptions，其中 Faulty GPU 148 次占 30.1%、GPU HBM3 Memory 72 次占 17.2%、Software Bug 54 次占 12.9%、Network Switch/Cable 35 次占 8.4%、Host Maintenance 32 次占 7.6%，其余 GPU SRAM Memory 19（4.5%）、GPU System Processor 17（4.1%）、NIC 7（1.7%）、NCCL Watchdog Timeouts 7（1.7%）、Silent Data Corruption 6（1.4%）、GPU Thermal Interface + Sensor 6（1.4%）、SSD 3（0.7%）、Power Supply 3（0.7%），以及 Server Chassis / IO Expansion Board / Dependency / CPU / System Memory 各 2（0.5%）；表注写明 "Root-cause categorization of unexpected interruptions during a 54-day period of Llama 3 405B pre-training. About 78% of unexpected interruptions were attributed to confirmed or suspected hardware issues"。转写 [76:06] 正对应 "during Llama 3 405B training, GPUs failed 148 times"，并由此引出 "you need lots of not just like fast parallelism, you need redundancy in order to be able to deal with all these ... So, there's a distributed systems challenge as well"。易错点：不要把 Llama 3 与 DeepSeek-V3 的配方混为一谈，前者是 dense 模型所以用 TP=8，后者是 MoE 所以用 EP=64；也不要把 148 这个数字理解成总中断数，它只是表格里 "Faulty GPU" 一行的计数，全部中断次数远多于此，但表注并未给出中断总数，只说明 54 天内约 78% 的意外中断被归因于确认或疑似硬件问题，读表时不要自行把各行相加去反推这个数；更别忽略「并行策略再好也扛不住硬件故障」这一层——它把讨论从算法扩展拉回到分布式系统容错。

![DeepSeek 幻灯片的 2.4 Infrastructures 引文与 V3 的 PP/EP/ZeRO 配置摘要](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0057_01-14-19.jpg)

![Llama 3 405B 预训练 54 天内意外中断的根因分类表，含 148 次 GPU 故障](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0058_01-15-59.jpg)

## 现代训练案例（二）、Megatron Bridge 与全课总结 [76:24 - 80:10]

[76:23] 起讲师补完案例清单，"Gemma 2, which is a Google open source model, has FSDP plus tensor parallel and sequence parallel, and uses these two strategies only"。他特意点出举这个例子的理由："as far as this is basically a realization of the Google claim that, you know, for TPUs, really you don't need to do pipelines, you just take a really big toroidal mesh, and you do tensor tensor parallel over that big mesh network that you have"——TPU 的大规模 toroidal mesh 让 TP 可以铺得很大，从而完全不需要流水线并行。但他也诚实标注了局限："Unclear whether this can like scale out forever. Um that has always been a little bit less clear to me, but at least at, you know, the Gemma scales, this is definitely the case." 这与第 3 章 TPU book 的 roofline 图（"on a 4x4x4 mesh"）、以及第 4 章 Guideline 2 的「NVLink 域内 TP」形成鲜明对照：GPU 侧受单机 8 卡限制，TPU 侧靠 mesh 把同一维度放大，于是同一套配方在不同硬件上会收敛到不同的组合。

[77:04] 起讲师 "took a look at some of the models that were trained this year, like Mixtral 8x22B"，并介绍 "Nvidia has a repository called uh Megatron Bridge, where they sort of release a lot of recommended training configurations for a lot of different model sizes and model settings"，可以查到 "what they end up saying is the right model configuration for like a 8x24 22B uh MoE or DeepSeek v3 model"。屏幕 "Mixtral 8x22B" 一页写着 "From megatron docs –" 与 "TP / PP / CP / EP (4/4/1/8) – DP likely 2 (to add up to 256GPUs)"，即

$$
TP=4,\quad PP=4,\quad CP=1,\quad EP=8,\quad DP\approx2\ \Rightarrow\ 256\ \text{GPUs}
$$

下方黑底表格 "Mixture-of-Experts Models" 列出三行：Mixtral 8x7B、64 GPUs、TP=1、PP=4、CP=1、EP=8，备注 "EP=8 for 8 experts"；Mixtral 8x22B、256 GPUs、TP=4、PP=4、CP=1、EP=8，备注 "TP+PP+EP for large MoE"；DeepSeek-V3、671B、1024 GPUs、TP=2、PP=16、CP=1、EP=64，备注 "Massive MoE with 256 experts"。转写 [77:39] 把 Mixtral 的配置讲成 "expert parallel of eight, pipeline parallel four, and an additional tensor parallel of four. This one for the attention layers"，并指出 "you kind of see, you know, this follows the prescription of keep your expert parallel roughly around eight"——TP=4 专门留给 attention 层、EP=8 留给专家，正是第 1 章 MoE Parallel Folding 解耦思路在真实配置里的落地；而 DeepSeek-V3 那一行 EP=64 又说明「around eight」只是常规建议，超大 MoE 会把它推到几十路。

[77:53] 的 "NeMo Tron 3 Super um follows kind of the DeepSeek v3 model, has a bunch of expert parallel, as well as context parallel, because they were doing extensions, uh at least in this article"；[78:03] 的 Qwen 3 "follows the DeepSeek recipe ... fairly large amount of expert parallel, 32, um and they have uh eight pipeline parallel and two tensor parallel to split up kind of things like the attention matrices"，即 $EP=32,\ PP=8,\ TP=2$，同样是用很小的 TP 只切 attention 矩阵。[78:22] 讲师还提到 Nvidia 在 Bridge repo 里做了 "different kinds of benchmarking that Nvidia folks have done for, you know, parallelizing different models"，"There's all sorts of different like sub configurations that you can pick even within tensor parallel that can significantly affect the performance of these. So, the the systems integration is quite significant"，并指明 "This is in the um Nvidia Bridge repo, if you want to take a look"。[78:49] 起他 "decided maybe I would build overview"，给出三条共性："as much as you can, all the models use data parallel. They maximize the data parallel domains to the extent that they can"、"Tensor parallel almost always remains below eight"、"And expert parallel can sometimes be big now, and I think that's partially because of Deep Seek V3 and a lot of the infra that they built for large-scale expert parallel training"。这三条正好把第 4 章「最大化 DP / TP 封顶 8 / EP 可放大」的配方用真实数据坐实。

把本章与上一章的十余个案例并排看，规律比任何单个数字都更有说服力：dense 模型（OLMo 7B、Yi、Llama 3 405B、Gemma 2）几乎都落在 FSDP / TP / PP / DP 这条经典链上，MoE 模型（DeepSeek-V3、Mixtral 8x22B、Nemotron 3 Super、Qwen 3）则一律把 TP 换成 EP、或把 TP 压到只够切 attention 的小数值，同时把 EP 顶上去，而长上下文阶段又统一调高 CP。就连 [77:13] 那句 "if you're interested in like model training and parallelism configurations in the wild" 所引出的 Megatron Bridge，其价值也不在于给出某个「最优配置」，而在于把 "a lot of, you know, recommended training configurations for a lot of different model sizes and model settings" 摊开供人对照；[78:38] 更提醒 "There's all sorts of different like sub configurations that you can pick even within tensor parallel that can significantly affect the performance of these"，同一 TP 度下不同子配置就能显著改变性能，所以 [78:47] 才会说 "the the systems integration is quite significant"。这也解释了为什么讲师在 [78:33] 明确 "I won't go into details of what this is"——案例里的并行度只是骨架，通信重叠、算子融合、容错冗余这些系统集成才决定最终能不能跑满，而骨架正是前面几章那条简单规则在不同硬件上的投影。

[79:20] 讲师 "to put everything together" 给出 "The highest-level point, right, of this lecture is we need to think about multi-GPU, multi-node, maybe even multi-data center parallelism"，并强调要 "bring to bear all the approaches that we have because you've got fast links, you've got slow links, you've got techniques that make use of batch sizes and all these different resources that you all want to consume as much as possible"，但最终 "given all these, it turns out that there's fairly simple rules of thumb for combining all these different forms of parallelism to get, you know, in the end what is effective, effectively like full utilization of your computer hardware"。屏幕 "Recap for the whole lecture" 一页用三条 bullet 收束全课："Scaling beyond a certain point requires multi-gpu, multi-node parallelism"、"No single solution to the parallelism problem (probably want all 3 approaches)"、"Simple, interpretable rules of thumb for combining different forms of parallelism"。最后 [80:01] 一句 "And next week I think we're talking about scaling loss" 把话头交给下一讲。易错点在于把总结里的 "all 3 approaches" 读成某三种固定策略的清单——结合第 2 章那张红字大表与第 3 章 roofline，它指的其实是按资源（快链路、慢链路、batch size）分工组合的思路，而不是 DP/TP/PP 的死板公式；同理，"Tensor parallel almost always remains below eight" 是硬件经验而非数学上限，DeepSeek-V3 的 EP=64、Qwen 3 的 EP=32 已经说明在 MoE 时代被突破的是 EP 而非 TP；最后也不要把本章的「案例罗列」当成新的推荐配置，讲师在 [78:33] 明确 "I won't go into details of what this is"，这些数字只是用来印证前面那条简单规则，真正的选型仍要回到自己硬件的 $B/N$、链路带宽与模型是否装得下。

![全课总结：扩展需要多机多卡并行、没有单一方案、组合并行有简单可解释的经验法则](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/kf_0060_01-19-45.jpg)

## 要点回顾

### 动机、硬件拓扑与集合通信记账
- 承接上一讲的并行底层机制；三条目标 Understand the systems complexities of training huge models、Different parallelization paradigms and why people use multiple at once、What large scale training runs often look like，预告 4D parallelism 与 actually even more：最大规模须把大部分策略一起用上。
- 两大瓶颈是算力与显存：最快超算有 exaflops 级算力、单张 GPU 差得很远，模型大到一张卡装不下必须 shard；由此有 intra-node parallelism 极快、inter-node communication 较慢的二分，决定张量／专家并行放节点内、流水线并行跨节点。
- 记账停留在 collective communication primitives 层；五原语下标原式 All reduce `out[i] = sum(inX[i])`、All Gather `out[Y*count+i] = inY[i]`、Reduce Scatter `out[Y][i] = sum(inX[Y*count+i])`，核心恒等式 all-reduce ≡ reduce-scatter + all-gather 让两步算法 for free 做出。
- 硬件上 TPU 的 toroidal mesh 满足 deg(chip)=O(1) 与芯片总数 N 无关，GPU 是 all-to-all 的 fat tree；H100 SuperPod 256 卡 bisection 57,600 GB/s、相对 A100 快 9x（reduce 4.5x），TPU8i/t 的 Virgo 因 MoE 趋向树状全互联，Ascend 910C CM384 系统级功耗 599,821/145,000 W ≈ 4.1x（单芯片 0.3x）。收束为 the new unit of compute is not the GPU, it's the entire data center，要求 Linear memory scaling 与 Linear compute scaling。

### 朴素数据并行与 ZeRO 三阶段显存切分
- 忘掉 Adam 的朴素 SGD 起点 θ_{t+1} = θ_t − η Σ_{i=1}^{B} ∇f(x_i)，把 B 切给 M 台机器各算 B/M 再同步；三项评估 Compute scaling（each GPU gets B/M examples）、Communication overhead（transmits 2x # params every batch，OK if batches are big）、Memory scaling = none（Every GPU needs # params at least）。
- 显存账本 Our memory situation is actually terrible：We need 5 copies of weights and 16 bytes per param，即 2 字节 FP/BF16 参数 + 2 字节梯度 + 4 字节 FP32 master weights + 4(or 2) 字节 Adam 一阶矩 + 4(or 2) 字节二阶矩，2+2+4+4+4=16，optimizer state 占大头，注明 Depending on our precision。
- ZeRO 四行 Memory Consumed：(2+2+K)Ψ=120GB、2Ψ+2Ψ+KΨ/N_d=31.4GB、2Ψ+(2+K)Ψ/N_d=16.6GB、(2+2+K)Ψ/N_d=1.9GB；算例 K=12、Ψ=7.5B、N_d=64，120→1.9 约 63 倍，先切优化器状态那一刀最值钱。
- Stage 1 四步 Everyone computes a full gradient on their subset of the batch → ReduceScatter the gradients（#params）→ Each machine updates their param using their gradient + state → All Gather the parameters（#params）；reduce-scatter + all-gather ≡ all-reduce 使合计仍是 2×#params、与 naive DDP 相同，故 zero stage one ... this was free. Free memory savings，对应 31.4GB。
- Stage 2 的 Complexity 是 we can never instantiate a full gradient vector, but each worker must compute a full gradient；解法 a systems trick：反向回扫时每算完一层梯度立即 reduce 给负责该片的 worker、后续反向图不再需要就 free，降到 16.6GB。

### FSDP／ZeRO-3 的通信账、计算重叠与实践规模
- Stage 3 写作 ZeRO stage 3 (aka FSDP) shard everything，P_{os+g+p} = (2+2+K)Ψ/N_d = 1.9GB；三条 High level idea 是 Shard everything – incl parameters!、Use the same 'incremental communication / computation' ideas、Send and request parameters on demand while stepping through the compute graph，PyTorch 里即 FSDP。
- baby version 流程链 LOAD-MODEL SHARD → ALL-GATHER → FORWARD (LOCAL) → FREE FULL WEIGHTS → ALL-GATHER → BACKWARD (LOCAL) → REDUCE-SCATTER → FREE FULL WEIGHTS → … → UPDATE WEIGHTS (LOCAL)；伪代码 for layer in model.layers、W = all_gather(shard[layer])、reduce_scatter(g)、optimizer_step(local_shard)，每卡任意时刻只持三者各一片。
- 一次更新 two all gathers and one reduce scatter：comm_{ZeRO-3} = 2 all-gather(Ψ) + 1 reduce-scatter(Ψ) = 3Ψ = 1.5 × comm_{DDP}(2Ψ)；What's the point? 里 Stage 1 是 2*# param – it's free!、Stage 2 是 it's (almost) free、Stage 3 是 1.5x comm cost, but that's not bad!，注明 ignoring overhead / ignoring latency。
- 抹平开销靠 two really important ideas：沿 compute graph 扫后立即 free、overlap the communication and computation；AG0 完成就做 FWD0 同时预取 AG1，公式 (W1 W0 + W2 W0)x = y 表示权重复用，图分 CPU / GPU Comp. Stream / GPU Comm. Stream 三行、仍留少量 bubble，但 GPU 利用率 very close to single GPU performance，作业要写 wrap any module 的 FSDP wrapper。
- 问答澄清 FSDP 不是 pipeline：every GPU goes through the entire model、no GPU holds the entire parameters，通信次数随层数线性增长但每次只是 one tiny MLP、总量仍与 #params 同阶，activations 并未被切。ZeRO in practice: will it fit? 在 8X A100 80G、BF16 + Kahan summation、12 bytes per param 下给出 Baseline 6.66B/12 → Stage 1 16B/5 → Stage 2 24.62B/(2+10/8) → Stage 3 53.33B/(12/8)，即 640/12≈53.33B、口述 7B → 50B。

### 数据并行的硬限制与流水线、零气泡调度
- 数据并行消耗 batch size：#machines ≤ batch size，越过 critical batch size 后 the gain that you get from the additional batch element is less than if you had taken another SGD step；图上 Perfect scaling 越过 B/ℬ=10^0 转入 Ineffective scaling（纵轴 ε_opt(B)/ε_max，ℬ 是 noise scale，别与 b、B 混淆）。另一硬限制是 Stage 1/2 不让显存随卡数扩展、activation memory 完全没被切。
- 引出 model parallelism，副标题 Scaling up in memory (without changing batch size) with model parallelism；与 FSDP 的关键差异是 now what we're going to do instead is we're going to communicate activations back and forth（FSDP 传 parameters），三类是 1. Pipeline parallel（cutting up layers）、2. Tensor parallel (+ Sequence parallel)（cutting up matrices）、3. Expert parallel（sharding experts）。
- 流水线把 Layer 0–3 分到 GPU 0–3，前向传 activations、反向传 partial gradients，朴素时 one GPU will be active at a time 而生 bubble；解法是切成 micro-batch（图中 4 个）用 F_{i,j} / B_{i,j} 甘特图交错，bubble ratio = (n_stages − 1)/n_micro，紧跟 so we need a big batch size!。
- 仍保留流水线因 comm_PP = b × s × h 的 point-to-point 通信 almost always a smaller amount of data than attempting to communicate a whole parameter matrix，且 pipelines are going to live on the slowest networking links（跨节点、多 pod），还能 You also compose this with data parallel；Megatron paper 的 parameter sweeps 由 batch size、pipeline size、micro-batch 数决定利用率。
- Zero bubble pipelining 不是调度而是拆解 backward：Split up backwards into two parts，1. Backpropagating activations (z,x)（B，关键路径，下一 stage 等信号才能干活）、2. Computing weight gradients (W)（a leaf node，The second part can be done whenever），先排完 B 解锁下游、把 W 塞进空隙；Figure 2 的 1F1B 与 Figure 3 的 ZB-H1/ZB-H2 白色空格几乎消失，效果取决于 W 与 B 的工作量比例，DeepSeek paper 的 clever scheduling 也压气泡，段末以 pipeline parallel is like depth 引出 Y = X1 A1 + X2 A2。

### 张量并行、序列并行与激活线性下界
- 串行 MLP 是 Y = GeLU(XA)、Z = Dropout(YB)；A 按列切 A=[A1, A2]、B 按行切 B=[B1; B2] 分到不同 GPU（Assign columns (A1, A2) and rows (B1, B2) to separate GPUs.），因 GELU 逐元素有 GeLU(XA) = [GeLU(XA1), GeLU(XA2)] 无需通信，只在 Z = Y1B1 + Y2B2 的 g 处 all-reduce，代码 Y1 = torch.nn.functional.gelu(X @ A1)、Z = all_reduce(Z_partial_0 + Z_partial_1)。前向 In the forward pass, f is the identity, and g is an all-reduce.，反向 In the backward pass, f is an all-reduce, g is the identity.。
- Row vs Column tensor parallel 三条：Columnwise – QKV, up-projection、Rowwise – Attention output, down-projection、Replicated – norms, routers, etc；Self-Attention 一侧 split attention heads（Q=[Q1,Q2]、K=[K1,K2]、V=[V1,V2]），layer norm、非线性与 router 等 small layer 因依赖完整隐藏维度而 fully replicated across the machines，其激活显存被完整复制 t 份。
- 张量并行 communication hungry，只在 within a node：eight GPUs are going to be networked tightly together in a single box、up to eight；3B 模型柱状图 TP=2/4/8/16/32 跌幅 -10.8%/-12.2%/-42.7%/-65.6%，跨单机即 a big drop in performance，TPU 的 big mesh 则 you can tensor parallel very large numbers。对照 C_pipeline = b s h（point-to-point, per microbatch）与 C_tensor = 8 b s h·(n_devices−1)/n_devices（all-reduce, per layer），Pros 是 no bubble / low complexity / doesn't need large batch sizes，Cons 是 much larger communication。
- PyTorch profiler 显示 Memory isn't just the static bits, but also activations!，红色激活鼓包随前向累积，maximum memory usage happens a little bit after the maximum activation point（反向早期既存激活又存梯度才是峰值），长序列下 activations 会 dwarf the parameter memory。起点公式 s b h(34 + 5 as/h)，34=24(MLP)+10(点式)、5as/h 是 s² 级 quadratic attention terms incl dropout，可 As with flash attention 经 recomputation 丢掉；代码 activation_bytes_per_layer(s, b, h, a, bytes_per_elem=2)。
- TP 下变为 s b h(10 + 24/t + 5 as/(h t))，残留 10 拆成 LayerNorm (4sbh)、Dropout (2sbh)、attention 与 MLP 输入 (4sbh)，a thousand different GPUs 仍 suffer a 10x SBH penalty。序列并行处理这 10 SBH：This is an extremely misleading name（真正配得上的是 context parallel），把 observation: all the 10sbh terms are pointwise ops over the sequence 沿序列轴切分，前向 g 是 all gather、ḡ 是 reduce-scatter、反向互换，conceptually very similar to the FSDP style idea 但切激活。组合表格五行到 tensor parallel + sequence parallel + selective activation recomputation ⇒ s b h(34/t) 即 34sbh/t，是 kind of the lower bound；24 is MLP part、MLP recomputation 很贵，Recomputation for attention is generally cheaper（可 tile-wise、不付 quadratic cost）。

### 专家并行、组合约束与 roofline 配方
- 专家并行是工具箱最后一个 ingredient，行为 roughly like tensor parallel（high bandwidth, reduces activation），但结论是 you almost always prefer using an expert parallel sharding strategy over tensor parallel；Megatron 的 Guideline 4: Prefer EP over TP for Expert Layers 列 Better GEMM efficiency / Lower communication / Simpler computation graph / Token permutation，并给 Example: For Mixtral 8x7B, EP8×TP1 outperforms EP4×TP2、When EP = num_experts, local token permutation is eliminated。
- 难点是 EP requires somewhat frequent all-to-all dispatches 且 latency-sensitive（computation is waiting for your tokens to arrive）：DeepSeek 的 DPP / DeepEP 提供 high-throughput and low-latency all-to-all GPU kernels、FP8、NVLink 到 RDMA 域转发、pure RDMA 低延迟 kernel 与 hook-based communication-computation overlapping that does not occupy any SM resource，NVIDIA 有 Hybrid EP，DeepSeek 还用未公开 PTX 指令。组合约束是 DP usually shares replicas with EP splits (so EP<DP)、DP and TP can interact badly to lower utilization。
- 解耦源于 Recall: MoEs apply to the MLPs, not the attention (with some exotic exceptions)：attention 要 High TP、MLP 要 Low TP 形成结构性冲突；MoE Parallel Folding 给两类层各设 Parallelism Group，Attention Layers 为 TP × CP × DP × PP、MoE Layers 为 ETP × EP × EDP × PP（仅 PP 公共），Separates out TP/CP/DP (for attention) and (ETP/EP/EDP) for MLPs。context parallel / ring attention 把长序列激活切开、按 TPU mesh 以 ring-like way 传 KV，用于 long context extension 与 model serving。
- Recap: LLM parallelism table 用红字标六种方法（DDP/ZeRO-1、FSDP/ZeRO-3、Pipeline、Tensor、Sequence/Context、Expert）的主观缺点，Param memory per rank 约 1、1/DP、1/PP、1/TP、1/EP，结论是 there is no one strictly dominant parallelization strategies；roofline 的 Key quantity 是 Global batch size (divided by GPU) 即 B/N、纵轴 FLOPS time/Comms time，逐层 compute 如 DP: 4BDF/X + 8BDF/X、comms 0 + 8DF，FSDP 多 4DF，三段注释 No scheme works when B < 400、Only mixed FSDP + MP works when B < 850、Both mixed FSDP + MP and pure FSDP work when B > 850，讲师称此为 3D or 4D parallelism。
- 简单配方 Simple rules of thumb 两步法：Until your model fits in memory 用 Tensor / expert parallel up to GPUs / machine（单机 8 卡即 TP=8 或 EP=8）、Pipeline parallel across machines、Or use Zero-3，Then until you run out of GPUs 则 Scale the rest of the way with data parallel，batch 小则 gradient accumulate。Megatron 五条 Guideline 顺序相反（Minimize Model Parallelism, Maximize Data Parallelism、Keep EP and TP Communication Within NVLink Domain、Use Pipeline Parallelism (PP) for Multi-Node Scale、Prefer EP over TP、Enable Context Parallelism (CP) for Long Sequences 用 --context-parallel-size 与 Sequence lengths ≥ 8K tokens），配置项还有 --use-distributed-optimizer、--num-layers-per-virtual-pipeline-stage；问答补充 sequence parallel 是 an add-on、loop transformer 破坏 FSDP 的 get weights, discard weights 前提、recompute ⇒ save memory ⇒ larger B ⇒ compute-bound。

### 真实训练案例、定量证据与全课总结
- 扩展证据 Figure 10：PTD-P = PP × TP × DP 曲线 More GPUs, same, flat utilization（530B 约 173、175B 约 150→143 teraFLOP/s per GPU，横轴 768–1920 GPUs），纯 ZeRO-3 的 530B 从约 140→100→50、175B 从约 145→88→45 下滑，前提是 Global batch sizes are fixed、ZeRO-3 is used without any model parallelism；讲师给出 tensor parallel size of eight 的停止点。
- Dense 侧：OLMo / Dolma 7B 是 Dolma – 7B model, FDSP (probably fits intra-node)（FDSP 系 FSDP 笔误），micro-batch 4096 tokens per GPU、全局约 4M tokens（2048 × 2048）；DeepSeek V1 是 ZeRO stage 1 with Tensor, Sequence, and Pipeline parallel（HAI-LLM、1F1B、flash attention、bf16 训练 fp32 累加、in-place cross-entropy、算子融合），Yi 是经典 ZeRO-1 + TP + PP；Gemma 2 用 FSDP + TP + SP、不用 PP，印证 TPU 大 toroidal mesh 不需 pipeline。
- Llama 3 405B 主预训练 TP=8, CP=1, PP=16, DP=128，长上下文阶段 crank up the context parallel, lower the data parallel，因是 gigantic dense model 仍在 tensor parallel, not expert parallel land；Table 5 记录 54 天里 Faulty GPU 148 次占 30.1%、GPU HBM3 Memory 72 次占 17.2%，约 78% 意外中断归因硬件。
- MoE 侧：DeepSeek V3 是 PP (16), EP (64-way, 8 nodes), ZeRO stage 1 且 EP uses 1F1B A2A Overlap；Megatron Bridge 里 Mixtral 8x22B 为 TP/PP/CP/EP (4/4/1/8)、DP likely 2 凑 256 GPUs，DeepSeek-V3 671B/1024 GPUs 为 TP=2/PP=16/CP=1/EP=64（Massive MoE with 256 experts），Mixtral 8x7B 是 64 GPUs/TP=1/PP=4/CP=1/EP=8，Qwen 3 为 EP=32/PP=8/TP=2，Nemotron 3 Super 叠加 context parallel。
- 三条共性：as much as you can, all the models use data parallel（maximize the data parallel domains）、Tensor parallel almost always remains below eight、expert parallel can sometimes be big now；同一 TP 度下不同子配置显著影响性能、the systems integration is quite significant。全课以 Recap for the whole lecture 三条收束：Scaling beyond a certain point requires multi-gpu, multi-node parallelism、No single solution to the parallelism problem (probably want all 3 approaches)、Simple, interpretable rules of thumb for combining different forms of parallelism，末句把话头交给下一讲 scaling loss。

## 视频来源

- 原始链接：https://www.youtube.com/watch?v=6-cXp-aOmdg
- 课程：Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 8

---

## 对照参考笔记补遗（Spring 2025 版）

> 编者注：本节依据 2025 版参考笔记整理，其中的「更正」小节记录的是核对时发现的问题，对应正文已在本版订正；补遗内容不代表 2026 课程视频原文。


本节内容来自 2025 版参考笔记（Qihong Ruan 整理的 Lecture 08: Parallelism 2，由 MinerU 从 PDF 解析），仅用于与 2026 版详写笔记对照补充，不代表 2026 课程视频原文；凡与本讲视频表述不一致之处，一律以视频为准，下列内容只作背景与参照。

### 集合通信原语的完整图景：六类操作、world size 与 rank

2026 版笔记以五个下标公式（all reduce、broadcast、reduce、all gather、reduce scatter）和 all-reduce ≡ reduce-scatter + all-gather 这条恒等式作为全部记账基础，重点落在 ZeRO 与 FSDP 的推导上；2025 参考则把集合通信（collective operations）本身铺得更完整，把它追溯到 1980 年代并行编程文献中「比自己管理 point-to-point 通信更好、更快的抽象」。它先给出两个基本术语：world size 是进程组里的设备总数（例如 4），rank 是每个设备的唯一编号（0、1、2、3）；随后用张量的初态、终态逐一定义六类原语，而不只给公式。以 world size = 4 为例，broadcast 是 rank 0 持有 $[t_0]$、其余为空，操作后所有 rank 都拿到 $[t_0]$；scatter 是 rank 0 持有 $[t_0,t_1,t_2,t_3]$，操作后 rank $i$ 各得 $[t_i]$；gather 是 scatter 之逆，把各 rank 的 $[t_i]$ 汇总回单一 rank 得到 $[t_0,t_1,t_2,t_3]$；reduce 与 gather 类似但不做拼接，而是对收集来的数据做可结合、可交换的运算（sum、min、max、average），终态是根 rank 持有 $T=t_0+t_1+t_2+t_3$；all-gather 把收集结果发给所有 rank；reduce-scatter 则让每个 rank $i$ 先贡献自己的 $in_i$，得到全局规约 $\mathrm{SUM}(in_0,\dots,in_N)$ 后，再把其中一段 $out_i$ 散回各 rank。

最值得补的是 scatter 与 gather 这两个 2026 版未单列的原语，以及一条记忆法：reduce 指某种可结合、可交换运算，broadcast 与 scatter 是 gather 的逆，「all」前缀表示目的地是所有设备而非单一 root。把这条记忆法与 2026 版那张五原语幻灯片并排看，正好解释了为什么 all-gather 与 reduce-scatter 在数据布局上互逆、而 all-reduce 恰好等于二者串联——all-gather 让「每人一片」变成「每人全量」，reduce-scatter 让「每人全量」变成「每人一片并带上求和」，二者串起来正是「每人都得到全局和」，也正对应 Stage 1 里「每个 worker 只更新参数空间中属于自己的那一片」。需要注意两处易错点：root 的选取只是举例，参考用 rank 0、2026 幻灯片用 rank 2，并不改变原语语义；参考把 broadcast / scatter 并列为 gather 之逆，与 2026 版「all-gather 与 reduce-scatter 互逆」是两组不同配对，不要混为一谈。

### 从原语到硬件：NCCL、torch.distributed 与通信层次带宽

2026 版本讲明确把讨论层次限定在 collective communication primitives，并说「不会真的去发包，也不会下到驱动或硬件层」，只在回顾 TPU/GPU 拓扑时给出集群级带宽；2025 参考则专门补上这层被跳过的软件栈与硬件规格。硬件层次从快到慢、从小到大依次是：SM 内极小的 L1 cache / shared memory、单 GPU 上的 HBM、同节点多 GPU 之间的 NVLink、以及跨节点的交换网络，并据此强调 arithmetic intensity（算术强度）与「数据离 SM 越远、访问越慢」这一贯穿全讲的主题。经典家庭或普通服务器配置里，同节点 GPU 走 PCIe bus（v7.0、16 lanes，约 $242\ \mathrm{GB/s}$），跨节点走 Ethernet（约 $200\ \mathrm{MB/s}$），因数据拷贝、内核介入与慢传输而不适合深度学习；现代数据中心里，节点内用 NVLink 直连绕过 CPU，跨节点用高速交换网络绕过慢速以太网，例如每颗 H100 有 18 条 NVLink 4.0 链路、合计约 $900\ \mathrm{GB/s}$，而 HBM 带宽约 $3.9\ \mathrm{TB/s}$，仍明显高于 NVLink。参考还给出一条可直接使用的排查命令 `nvidia-smi topo -m`，用来查看 GPU 之间经 NVLink 与 NIC 的实际连接方式。

软件栈这一层同样是 2026 版略过的增量。NCCL（NVIDIA Collective Communication Library）是底层库，负责探测硬件拓扑（节点数、交换机、NVLink/PCIe）、优化 GPU 之间的传输路径、并启动 CUDA kernel 完成收发，从而把高级集合操作翻译成硬件优化过的包传输；`torch.distributed` 则在其上提供 Pythonic 接口（如 `all_gather_into_tensor`），通过可替换的 backend 兼顾可移植性——Gloo 用于 CPU、NCCL 用于 GPU，同一份代码能跑在不同硬件上只是性能不同，并且它还内置了 FullyShardedDataParallel（FSDP）这类更高层的算法。参考给出的 `setup(rank, world_size)` 也值得补：各进程通过 `MASTER_ADDR="localhost"`、`MASTER_PORT="15623"` 这类 rendezvous 点互相发现（这里不传实际数据，数据仍走 NCCL），再按是否可用 CUDA 选择 `dist.init_process_group("nccl", ...)` 或 `("gloo", ...)`；`dist.barrier()` 则让进程组内所有 rank 同步到同一点，便于把打印输出分组或满足特定同步需求。把这些与 2026 版「原语是跨硬件通用的记账单位、拓扑只是影响成本的实现细节」合起来读，正好补上「通用记账如何落到具体库与链路」这一环，也与本讲后面 FSDP、通信计算重叠这些建立在原语之上的算法相衔接。

需要提醒的是，2025 参考把 NVSwitch 描述成「跨节点直连、绕过 CPU 与以太网」，又在同一份笔记里说经典配置的跨节点通信走 Ethernet，两处口径并不自洽；这类由 PDF 解析带来的表述只宜当作背景，不宜照搬成结论。同理，$242\ \mathrm{GB/s}$ 与 $200\ \mathrm{MB/s}$ 单位不同（GB 对 MB），转述时极易写错量级；2026 版 A100/H100 SuperPod 表里的 $3{,}600$、$57{,}600\ \mathrm{GB/s}$ 说的是 256 卡聚合 bisection 带宽，与这里的单卡 NVLink $900\ \mathrm{GB/s}$、HBM $3.9\ \mathrm{TB/s}$ 属于不同口径，不能直接相比，更不能互相替换。

### 集合通信的实测：benchmarking 方法与 277 / 75 GB/s

2026 版笔记给出的都是理论或规格层面的带宽，并没有演示如何测一次真实的 NCCL 传输；2025 参考补上了 benchmarking（基准测量）的完整方法与一组实测数字，正好可作为「记账之后如何验账」的对照。测量流程三步：先 warmup 跑一次操作，确保 CUDA kernel 已加载、惰性计算已完成；再用 `torch.cuda.synchronize()` 配合 `dist.barrier()` 保证所有进程与 CUDA kernel 都结束，然后才启动计时；最后用 `time.time()` 量出耗时。带宽换算口径写得很明确——对 all-reduce，sent bytes 取 `tensor.element_size() * tensor.numel() * 2`，乘 2 是因为每个 rank 既要发送自己的输入、又要接收聚合后的输出；总时长按 `world_size * duration` 计（假设各 rank 贡献相同），于是 $\text{bandwidth} = \text{sent\_bytes}/\text{total\_duration}$。

参考给出的算例，是在 4 张 GPU 上对约 $10^8$（100 million，代码里 `num_elements=104857600`）个元素做 all-reduce，实测约 $277\ \mathrm{GB/s}$，明显低于 H100 的理论 $900\ \mathrm{GB/s}$；同样规模的 reduce-scatter 只有约 $75\ \mathrm{GB/s}$，其 sent bytes 用的是 `* 1` 而非 `* 2`，因为每个 rank 只发送输入、却只收回规约结果中属于自己的那一小片。参考把 $277$ 与 $75\ \mathrm{GB/s}$ 的差距部分归因于 NVIDIA 网络对 all-reduce 可能存在的硬件加速，并由此得出一条与 2026 版「通信能否被计算重叠掩盖」相呼应的结论：真实性能取决于张量大小、设备数量与具体实现，光看规格表不够，必须实测。这里的易错点有两处：一是 `* 2` 与 `* 1` 的因子只在这套记账口径下成立，与 2026 版用 $\Psi$（参数量）核算 $2\Psi$、$3\Psi$ 的抽象单位不是一回事，不能把 $277\ \mathrm{GB/s}$ 直接套进 ZeRO-3 的通信公式；二是终端输出里 `took 0.00ms` 显然不是真实耗时，带宽是用另一套字节数与时长算出来的，照抄时长会得出错误结论。参考的终端日志还附了 all-reduce 前后各 rank 的张量（$[0.,1.,2.,3.]$ 到 $[6.,10.,14.,18.]$）、reduce-scatter 各 rank 只留一片（$6./10./14./18.$）、以及 all-gather 再拼回全量的完整过程，可作为本讲 all-reduce 恒等式的一次数值化印证；其中个别行把结果误印为 $[0.,10.,14.,18.]$，与同段其它 rank 矛盾，应以多数行互证为准。

### 从零实现三种并行：DDP、张量并行、流水线并行的代码参数

2026 版对三种并行的处理是概念、通信公式与真实集群配置，2025 参考则保留了一段「从零写起」的教学实现，用深层 MLP 作为代表架构（理由是 Transformer 的计算瓶颈往往与 MLP 相似），并给出可直接核对的具体参数，这是笔记里完全缺的一环。三者统一用 $\text{batch\_size}=128$、$\text{num\_dim}=1024$，参数矩阵为 $1024\times1024$，优化器 `torch.optim.AdamW`，前向是 `x = x @ param` 接 `F.gelu(x)`，损失取 `loss = x.square().mean()`。数据并行（DDP）沿 batch 维切数据：每卡取 $\text{local\_batch\_size}=\text{batch\_size}/\text{world\_size}$，各自算本地 loss 与梯度，再对每个参数调用 `dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG)` 把梯度平均到一致，最后 `optimizer.step()`；参考点明各 rank 的 loss 不同、梯度经 all-reduce 后相同、因而参数始终保持相同，等价于 $\text{world\_size}$ 份 SGD 被同步成同一个状态。

张量并行沿 hidden 维切模型：$\text{local\_num\_dim}=\text{num\_dim}/\text{world\_size}=256$，每层参数变成 $\text{num\_dim}\times\text{local\_num\_dim}$，前向每算完一层的局部激活（$128\times256$）就 `dist.all_gather_into_tensor(...)` 把各卡分片收集齐，再 `torch.cat(activations, dim=1)` 拼回 $128\times1024$ 的完整激活供下一层使用；参考强调「数据在所有 rank 上相同、每层都要传激活」，因此通信密集、依赖高带宽互联，这与本讲「张量并行只宜放在节点内最快链路」的判断一致。流水线并行沿层切模型：$\text{local\_num\_layers}=\text{num\_layers}/\text{world\_size}=2$，每卡只持有自己那两层的参数，把 batch 拆成 $\text{micro\_batch\_size}=32$ 的 microbatch，用 `dist.recv(tensor=x, src=rank-1)` 与 `dist.send(tensor=x, dst=rank+1)` 在相邻 rank 间传激活。参考明确列出这份朴素实现缺的两件事——用 `isend` / `irecv` 做异步通信以重叠计算、掩盖 pipeline bubble，以及需要交错调度 forward / backward 的反向传播，并把它留作作业练习。把这段与本讲的气泡占比 $(n_{\text{stages}}-1)/n_{\text{micro}}$、zero bubble 把 backward 拆成 B 与 W 两部分对照，能从「代码怎么写」这一侧补上「公式为什么长这样」的来由。

有一处版本性差异要标明而非判错：2025 参考的 DDP 用 `op=dist.ReduceOp.AVG` 对梯度取平均，而 2026 版幻灯片明确写的是「对梯度求和、没有除以 $B$ 的归一化因子」；参考自己的集合通信示例又用 `dist.ReduceOp.SUM`，可见这是两年编排与代码取例的不同，不是哪一版写错。此外参考这几段代码经 PDF 解析后带有噪声，补录时以语义互证为准、不要照抄字符串；这也再次说明抽象记账与具体库实现之间还隔着一层次。

### 另一种范式：Jax/TPU 的编译器驱动分片

2026 版在硬件章节详细讲了 TPU 的 toroidal mesh（节点度数 $\deg(\text{chip})=O(1)$、与芯片总数无关）、GPU 的 all-to-all fat tree，以及 TPU8i/t 因 MoE 转向 Virgo 交换网络的「趋同演化」，并用 Gemma 2 印证 TPU 大 mesh 上可以不做流水线；但它始终把并行实现放在 PyTorch「从原语搭起」的框架里。2025 参考则给出一条对照性范式：Jax/TPU 让用户只定义模型与分片策略（sharding strategy），由 Jax 编译器把高层分片规格自动翻译成底层通信原语，概念上比 PyTorch 自底向上构建更简洁。参考举了两个例子：在 Levanter 项目里，用户给出模型与优化器状态、再指定参数沿哪些轴分片（例如 `embed` 用于嵌入、`data` 用于 batch），编译器便自动处理切分与通信，于是 FSDP 约十行代码即可实现；张量并行同理，只要指定张量并行的分片轴（如 attention 的 `head`、MLP 的 `model`），其余交给编译器。

这条内容与 2026 版「算法与硬件基底无关（hardware and substrate agnostic）」「理解原语后可迁移到任何硬件」的方法论正好构成一体两面：一边是 PyTorch 暴露原语、由工程师手工编排通信与计算重叠（FSDP wrapper、zero bubble 调度、DeepEP 的低延迟 all-to-all），另一边是 Jax 把同样的决策交给编译器，代价是失去对底层时序的精细控制。参考也点明课程本身仍选 PyTorch 来演示「如何从原语搭起」，因此两种范式不是对错之分，而是抽象层次与可控性的取舍。把它与 2026 版结尾「案例里的并行度只是骨架、系统集成才决定能否跑满」合读，能更清楚地看到：编译器驱动与手工编排，本质上都在回答同一组关于显存、带宽与 batch size 的约束，只是把决策放在了软件栈的不同高度。参考在结尾留下的开放问题（GPU 非确定性与精确复现、`isend`/`irecv`、Grock/Cerebras 一类大片上内存专用硬件、CPU 时代控制流编程与深度学习数据流负载的根本差异、静态计算图如何支撑更聪明的优化与布局）也可作为这组对照的延伸，但因其较零散、且与本讲主体关系较远，这里只点到为止，不另展开。
