---
title: "第 8 讲 · 并行化 II：数据、流水线、张量与专家并行"
description: "大规模语言模型训练必须同时解决计算和显存两个瓶颈，因此并行策略不是单一选择，而是数据并行、ZeRO/FSDP、流水线并行、张量并行、序列并行、专家并行和上下文并行的组合。"
---

# 第 8 讲 · 并行化 II：数据、流水线、张量与专家并行

> **Stanford CS336 语言模型从零构建 · L08** ｜ 讲者 Stanford Online ｜ 80 分钟 ｜ [原视频](https://www.youtube.com/watch?v=6-cXp-aOmdg)

> **内容驱动配图版** · 配图数量与位置由「章节内容 ↔ 画面语义」相关度决定，不设固定张数；文末附「对照参考笔记补遗（Spring 2025 版）」，记录与 2025 版讲义笔记的核对结果。

## 核心结论

这一讲承接上一讲的并行底层机制，讲现代语言模型训练为何要叠加多种并行范式：算力与显存两大瓶颈决定了必须走向多机多卡，而节点内快链路、节点间慢链路与全局 batch size 又要求不同策略按链路分层摆放。全讲以集合通信原语记账，用 all reduce ≡ reduce scatter + all gather 说明 ZeRO 切分优化器状态与梯度几乎不增通信，FSDP 虽多一次 all-gather 仍可借通信与计算重叠隐藏开销。但数据并行不省激活显存、又消耗 batch size，越过临界批量收益递减，于是转向改传激活的模型并行。流水线只传 $b\times s\times h$ 的点对点消息、适合慢链路却要用 micro-batch 压气泡，张量并行沿宽度切矩阵、通信密集只宜放在节点内，二者由 depth/width 的类比串起，为专家并行与 3D/4D 组合埋下伏笔。

## 一、课程定位与并行训练的两大瓶颈 [00:00 - 03:05]

本讲承接上一讲 Percy 讲的并行底层机制（underlying mechanics of parallelism），集中讲现代语言模型训练里的 knowledge details and trivia。课程目标写在 *Outline and goals* 页：Understand the systems complexities of training huge models（理解超大模型训练的系统复杂性）、Different parallelization paradigms and why people use multiple at once（不同并行范式及为何并用）。最要紧的是末尾 4D parallelism（四维并行）预告——「四件事可同时做」，讲师还补了 actually even more；要在最大规模并行就得把大部分策略一起用，作业正是给定网络拓扑（network topology）与模型找最优策略。

![课程标题页：CS336 Lecture 8 Parallelism Basics](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0002_00-00-16.jpg)
图注：课程标题页，对应本讲主题导入。

![课程大纲与目标：理解大模型训练复杂性及并行范式](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0003_00-00-32.jpg)
图注：大纲与目标，对应多范式并用。

动机归结为两个瓶颈：算力（compute）与显存（memory）。单芯片算力远远不够，最快超算已有 exaflops（百亿亿次浮点运算每秒）级算力，单张 GPU 差得很远；*Limits to GPU-based scaling – compute* 把十年约 1000 倍增益拆成 Number Representation ~16x、Complex Instructions ~12.5x、Process ~2.5x、Sparsity ~2x，即 $16 \times 12.5 \times 2.5 \times 2 = 1000$（overall gain > 1000x）。显存一侧则是模型太大、一张 GPU 装不下，必须切成碎片（shard）。

![GPU计算扩展限制：单芯片推理性能10年提升超1000倍](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0006_00-01-20.jpg)
图注：单卡算力极限，对应算力瓶颈。

![多GPU多机并行架构：节点内NVLink与节点间InfiniBand互联拓扑](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0009_00-02-08.jpg)
图注：多机互联，对应节点内外快慢链路。

由此引出贯穿全讲的区分：intra-node parallelism（节点内并行）连接极快，可承受通信代价高的操作；inter-node communication（节点间通信）较慢，须更尊重信道约束。这条快慢二分后面决定张量、专家并行留在节点内，流水线并行跨节点。记账上今天不真的发包（sending packets），而停留在 collective communication primitives（集合通信原语）层，核算时只说「做一次 all reduce（全规约）」或「一次 all gather（全收集）」；但讲师提醒，高效实现还得 go all the way to the hardware level，本讲取 algorithmic（算法层面）讨论，因为概念可迁移到任何硬件基底。

![集合通信基础：All reduce、Broadcast、Reduce、All Gather及Reduce Scatter示意图](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0011_00-02-40.jpg)
图注：集合通信五原语，对应记账方法。

这段只是 quickly 过一遍的基础回顾，真正算法从 11:07 才开始；4D 预告、两大瓶颈、节点内外之分与集合通信记账四层合起来，正是后面每个策略的评价框架。

## 二、集合通信原语与 all-reduce 等价关系 [03:05 - 04:20]

基础回顾里最要紧的一条，是讲师特意重新拎出的恒等式：all reduce（全规约）与 reduce scatter（规约散射）加 all gather（全收集）在通信代价上等价——一边做一次 all reduce，另一边做两步不同操作，正因两者「在某种意义上有相同代价」，右边算法便可 for free（免费）地做出来。他说明这只是 a quick refresher；之所以此时反复强调，是因为 18:23 后推导 ZeRO Stage 1 与朴素数据并行通信量相同时，整个论证几乎全押在这条等价上：

$$\text{all-reduce} \equiv \text{reduce-scatter} + \text{all-gather}$$

![All Reduce 分解为 Reduce-Scatter 加 All-gather 的 GPU 数据流示意图](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0014_00-03-28.jpg)
图注：All Reduce 分解，对应核心等价关系。

*Some basics about collective communication* 一页把五个原语的下标公式写全：All reduce 的 `out[i] = sum(inX[i])`、Broadcast 的 `out[i] = in[i]`、Reduce 的 `out[i] = sum(inX[i])`、All Gather 的 `out[Y*count+i] = inY[i]`、Reduce Scatter 的 `out[Y][i] = sum(inX[Y*count+i])`。数据布局的差别比公式更大、也最易混淆：all reduce 与 reduce 都求和，但前者把结果发给所有 rank、后者只发一个 root；all gather 与 reduce scatter 恰是一对互逆动作。正因 reduce-scatter 的输出按 rank 分片，它才刚好对应「每个 worker 只负责更新属于自己的那一片参数」——这正是 ZeRO Stage 1 需要的语义；单纯 all reduce 会让每张卡都拿到完整梯度，于是显存省不下来。

学生追问为何偏讲「梯度做 reduce scatter」这一种分解，讲师坦承 there are other decompositions（还有别的分解），这一条只因 will be useful for AI algorithm 才被特别强调。也就是说恒等式给的是代价上的等价，而非唯一正确分解，当成通用最优解便是误读。方法论上，即将展开的算法是 hardware and substrate agnostic（与硬件、基底无关）的，但讲师仍想先讲 low-level hardware stack，因为它能解释不同训练公司为何走向不同并行策略——这句正好构成与下一段硬件拓扑的接缝。

## 三、硬件底座：TPU 环形网格与 GPU 胖树 [04:20 - 08:43]

进入硬件部分前，讲师先划清范围：本讲主体是 hardware agnostic 的，但 low-level hardware stack 仍值得单独讲，理由是它 leads to different parallelization strategies across different model training companies（导致不同公司走向不同并行策略）。第一个对象是 TPU——Google 的加速器，讲师说它「有点像轻量版 GPU」（a TPU is kind of like a lightweight GPU），真正不同在 networking。TPU 网络叫 toroidal mesh（环形网格）：想成一张网格再允许两端彼此通信，所有芯片主要与邻居相连、邻居在边界 wrap around（回绕）；真实结构是 3D 且更复杂，但心智模型用二维这张图就够。它最要紧的性质是邻居数不随规模增长，即 $\deg(\text{chip}) = O(1)$ 与集群芯片总数 $N$ 无关，因此可以 indefinitely scaled up very simply。

GPU 的联网哲学几乎相反：all-to-all（全互联）取向，结构像一棵树，常称 fat tree（胖树）。最底层 GPU 之间连得极快，其上把成组 GPU 连成 pod，pod 间再靠 spine switches（脊交换机）互通；代价是 as the number of nodes grows，树越大、通信成本越高。*TPUs vs GPUs* 那页的绿色表格给出可核对数字：1 DGX／8 GPUs 时 A100 为 2.5 Dense PFLOP/s、2,400 GB/s bisection、150 reduce，H100 为 16、3,600、450；扩到 32 DGXs／256 GPUs 时 A100 为 80、6,400、100，H100 为 512、57,600、450，即 H100 SuperPod 在 256 卡规模 bisection 快 9x、reduce 快 4.5x。由拓扑差异可推出适配性：通信只发生在邻居之间时 TPU cost-effective，随机、stochastic 的路由则 GPU 全互联更灵活——故 GPUs are good for mixture of experts（token 要路由到不同专家），TPUs are great for dense model on predictable partitions，点名即后面的 tensor parallelism。易错点是把拓扑差异说成「TPU 一定更快」，它只在规则、局部通信下占优。

最戏剧性的转折是讲师当天临时改稿：Google 早晨公布新一代 TPU，他看完发现 that's a tree topology、switched to a much more all-to-all connection。新结构 Fully-connected 4 TPUs on each board、8 boards in each group、36 groups in each pod（共 1,152 chips），靠 Virgo Network 与 Jupiter Network 提供 2-layer switching、Optical Circuit Switch (Apollo)，连训练芯片 TPU 8T 的 cross-rack 连接也开始更像 GPU。解释是现代语言模型多是 MoE，推理时路由 token 会产生大量乱飞的带宽，通信成为真实瓶颈，因此更想要 all-to-all。他把这总结为 TPUs 与 GPUs 之间一场 convergence / evolution（趋同演化）：workloads are defining the network that we need to have。这既与「算法与硬件无关」形成张力，也为下一章「用暴力互联补单芯片劣势」埋下对照。

## 四、暴力互联的代价与数据中心即计算单元 [08:43 - 11:05]

硬件收尾落在「为何不干脆把一切连起来」。讲师以上一讲提问类比：既然 SRAM 那么好，为何不整颗芯片全用？答案是 Groq，且确实做得很好；同理，既然 all-to-all connectivity 好，为何不用极快的光纤芯片把一切连起来？现成答案是华为 Ascend 910C Cloud Matrix 384。单看每颗芯片比 H200 差不少——做 matmul 更慢——但若愿意 dedicate a giant rack of fiber optic switches 把远多于此的芯片连成大域（in their case 384 in a big rack），或许就能靠「连得更多」补回单芯片劣势。

*Why not connect everything?* 用两层表算账。Chip and Package Level 对照 GB200 与 Ascend 910C：BF16 dense TFLOPS 2,500 对 780（0.3x）、HBM capacity 192 对 128 GB（0.7x）、HBM bandwidth 8.0 对 3.2 TB/s（0.4x）。System Level 结论整个翻转：GB200 NVL72 对 CM384 的 BF16 dense PFLOPS 180 对 300（1.7x）、HBM capacity 13.8 对 49.2 TB（3.6x）、Scale Up Domain Size 72 对 384 GPUs（5.3x）。单芯片全面落后、系统级全面反超，正是「用规模换单芯片」的数字证据。最触目的是功耗：All-in System Power 145,000 对 599,821 W，$\frac{599{,}821}{145{,}000} \approx 4.1\times$，即 it's four times that of the equivalent Nvidia system 的来源。易错点是 4.1x 属整系统口径（含 scale-out 网络、存储），单芯片层面反而是 0.3x；离开「系统级」前提谈「费电四倍」会失真。

权衡于是清楚：if you're willing to pay the power cost，就能 brute force 掉通信问题、scale out much more aggressively to 300 chips。把硬件几页合起来，10:33 给出收束：the new unit of compute is not the GPU, it's the entire data center（新计算单元不是 GPU 而是整座数据中心）。*Part 1 recap* 把目标写成 Linear memory scaling、Linear compute scaling 与 Simple collective comms primitives，口头说成要控制显存与算力、并希望一切 lossless，即 use all of the resources that we have。

![Part 1 回顾：数据中心作为新计算单元及多机扩展的线性内存/计算目标](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0041_00-10-41.jpg)
图注：Part 1 回顾，对应数据中心与线性扩展。

这正是 00:00 两大瓶颈的镜像，也是之后全部算法的评价函数：显存、算力是否线性扩展，通信是否落在可承受链路上。局限是数字来自厂商规格与第三方分析，讲硬件能力而非实际利用率，不能当作选并行策略的充分依据。

## 五、朴素数据并行与每参数 16 字节的显存账本 [11:05 - 15:04]

11:07 起进入主体（the bulk of the content）。*Part 2* 把问题写成 How do we parallelize LLMs? 3 important ideas：Data parallelism（数据并行，含 Naïve data parallel 与 ZeRO levels 1-3）、Model parallelism（模型并行，含 Pipeline 与 Tensor parallel）、Activation parallelism（激活并行，含 Sequence parallel）。口头分法更贴直觉：data parallel 把数据分到各 GPU、动的是数据，model parallelism 把模型切开、动的是模型；讲师提醒这条边界 a little leaky（有点漏），因为某个算法其实也会切参数——正指后面 ZeRO 连参数一起切。

![Part 2 大纲：LLM 并行化的三大类原语（数据/模型/激活并行）](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0043_00-11-13.jpg)
图注：Part 2 大纲，对应三类并行原语。

最先讲最直观的 data parallelism。为讲干净机制，让听众「忘掉 Adam」、暂用朴素 SGD：设全局 batch 为 $B$，每步取 $B$ 个元素把梯度加起来再更新，$\theta_{t+1} = \theta_t - \eta \sum_{i=1}^{B} \nabla f(x_i)$（幻灯片对梯度求和，没有除以 $B$ 的归一化）。最朴素做法是把 $B$ 切成 $M$ 份、整除则每台拿 $B/M$ 个样本，各自算梯度再 synchronize 求和。三项评估很干脆：Compute scaling – each GPU gets B/M examples；Communication overhead – transmits 2x # params every batch（来回传梯度，合计 $2\times\#\text{params}$）；Memory scaling – none（每张卡仍要装下全部参数与同样大的 activations）。

![朴素数据并行：SGD 更新公式及计算/通信/内存扩展分析](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0046_00-12-02.jpg)
图注：SGD 公式与三项扩展分析。

于是 compute 解决一部分、memory 一点没解决。讲师自嘲起初觉得显存 bad，算完账其实是 terrible：*What's wrong with naïve data parallelism?* 给出结论 We need 5 copies of weights and 16 bytes per param!，逐项拆为 2 bytes FP/BF16 model parameters、2 bytes FP/BF16 gradients、4 bytes FP32 master weights、4 (or 2) bytes Adam first moment、4 (or 2) bytes Adam second moment，即 $2+2+4+4+4 = 16$ 字节／参数、共 5 份副本。Adam 要跨时间追踪一阶矩与二阶矩、受 stability characteristics 影响往往得用高精度，这些统称 optimizer state（优化器状态），看账本才是开销大头。

![朴素数据并行示意图：每个 GPU 持有完整模型副本并共享数据集](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0051_00-13-22.jpg)
图注：每 GPU 完整副本，对应显存瓶颈。

![朴素数据并行内存开销：每参数需 5 份权重共 16 字节的精度分解](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0052_00-13-38.jpg)
图注：每参数 16 字节的精度分解。

易错点是只按「参数 2 字节」估显存会低估 8 倍，或把 16 bytes/param 当普适常数——页首写 Depending on our precision，两条 Adam 矩都标 4 (or 2) bytes，说明它随精度浮动；另一处是 naive DDP 下 memory consumption is linear with your number of accelerators，不是多卡反而更省。本章的 $2\times\#\text{params}$ 通信量与 16 bytes/param 账，正是后面各 Stage 公式要对照的 baseline。

## 六、ZeRO 的切分路径：从 120GB 到 1.9GB [15:04 - 16:45]

有了上一章账本，解法可概括成一句 Core idea: split up the expensive parts (state) and use the reduce-scatter equivalence（把最贵的状态切开并利用 reduce-scatter 等价）。*ZeRO – solving the memory overhead issue of DP* 用四行堆叠条形图画出收益：蓝 Parameters、橙 Gradients、绿 Optimizer States，Baseline 一行每张卡都顶着同样厚的绿条，往下依次是 $P_{os}$（只切优化器状态）、$P_{os+g}$（再切梯度）、$P_{os+g+p}$（连参数一起切），绿条橙条逐层被削成窄竖条。讲师把颜色说得很清楚：optimizer state 是绿色占大头，参数与梯度大小相同、state 更大。

右侧 Memory Consumed 一列给出逐行公式，右上角标了算例 $K=12$、$\Psi=7.5B$、$N_d=64$（$\Psi$ 参数量、$N_d$ 数据并行度、$K$ 优化器状态字节系数）：

$$(2 + 2 + K)\times\Psi = 120\text{GB}$$
$$2\Psi + 2\Psi + \frac{K\cdot\Psi}{N_d} = 31.4\text{GB},\quad 2\Psi + \frac{(2+K)\cdot\Psi}{N_d} = 16.6\text{GB},\quad \frac{(2 + 2 + K)\cdot\Psi}{N_d} = 1.9\text{GB}$$

代入可核对 $(2+2+12)\times 7.5 = 120$、$120/64 = 1.875 \approx 1.9$，正是讲师说的 all the way down from 120 to 1.9，约 63 倍降幅。

![ZeRO 各阶段（P_os / P_os+g / P_os+g+p）内存占用对比图与公式](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0058_00-15-14.jpg)
图注：各阶段内存公式，对应 120GB 到 1.9GB。

这四行还回答「为什么切 state 最划算」：Baseline 16 字节里优化器状态占 $K=12$ 字节、四分之三，先只切它显存就从 120GB 掉到 31.4GB，一步拿走绝大部分收益，再切梯度只到 16.6GB、切参数才到 1.9GB——收益并不均匀，第一刀最值钱。被 $N_d$ 除掉的项随切分范围扩大而增多，也把 11:42「边界有点漏」落到了具体数字上。紧接着话题转向代价：the main thing that we will study now is what is the cost of doing this；讲师先替听众说出 You might imagine there's no free lunch，随即反转 I think the remarkable thing is going to be that a lot of this will be free——「免费」不是没有通信，而是切分所需通信恰好能塞进原本就不得不做的那次 all-reduce，这正是 03:14 恒等式被提前强调的原因。

易错点有三：$K=12$、$\Psi=7.5B$、$N_d=64$ 是给定算例而非普适常数，套到别的模型须保留公式再代入自己的 $\Psi$ 与 $N_d$；$N_d$ 是数据并行度而非 GPU 总数；本章只核算显存占用，别把 1.9GB 读成吞吐提升，通信量与利用率要等 Stage 1/2 推导。

## 七、ZeRO Stage 1 的免费午餐与 Stage 2 的梯度切分 [16:45 - 20:02]

从 16:45 起讲师按 Stage 逐个拆通信代价（three stages corresponding to each of these steps）。Stage 1 最简单：只切 optimizer state 并 shard across the GPUs，每张卡仍保有完整 parameters 与 gradients，分工是 each worker is responsible for updating one slice of the parameters——若我是 GPU 0，就只更新最前面那一片。显存对应 $2\Psi + 2\Psi + K\Psi/N_d = 31.4\text{GB}$ 那行。

![ZeRO stage 1优化器状态分片示意图及内存消耗对比表](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0064_00-16-50.jpg)
图注：Stage 1 切分优化器状态。

*ZeRO stage 1, how it works* 把流程写成四步：Step 1 Everyone computes a full gradient；Step 2 ReduceScatter the gradients（#params）；Step 3 Each machine updates their param using gradient + state；Step 4 All Gather the parameters（#params）。别人并不需要全部梯度、只需与自己那片参数相关的部分，所以「把梯度切开按人发过去」正好用 reduce-scatter；Step 4 则把自己更新好的那一片（如 1/4）发回所有人用 all-gather。

![ZeRO stage 1工作流程四步图解：梯度计算、ReduceScatter、参数更新、AllGather](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0066_00-17-22.jpg)
图注：Stage 1 四步工作流程。

免费结论在这一步合拢：朴素 DDP 做一次 all reduce、通信 $2\times\#\text{params}$，Stage 1 做 reduce scatter + all gather 两次操作、各 $\#\text{params}$，合计仍是 $2\times\#\text{params}$，即 $\text{reduce-scatter} + \text{all-gather} \equiv \text{all-reduce} \Rightarrow \#\text{params}+\#\text{params}=2\times\#\text{params}$。讲师把话说死：zero stage one has the exact same communication characteristics as naive DDP... this was free. Free memory savings.

![ZeRO stage 1与朴素数据并行(Naive DDP)在通信原语、开销及内存上的对比表](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0071_00-18-42.jpg)
图注：Stage 1 与 DDP 通信对比。

易错点是等价只在通信量与代价特征上成立，语义并不相同：DDP 是所有人拿完整梯度各自更新，Stage 1 是每人只拿一片、只更新一片再收集参数；「免费」也不是零通信，只是没有多付。Stage 2 从 19:15 开始要切 gradients，讲师说这会 trickier：Stage 1 依赖「能算出完整梯度」，Stage 2 连完整梯度都不能 materialize，对应 $2\Psi + \frac{(2+K)\Psi}{N_d} = 16.6\text{GB}$。

![ZeRO stage 2梯度分片示意图及高层概念说明](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0074_00-19-31.jpg)
图注：Stage 2 梯度分片与 16.6GB。

解法被他称作 a systems trick：不再等整个梯度向量算完再切，而沿网络反向回扫（as I sweep backwards through my network），每算完一层梯度就立即 reduce 给负责该片的 worker，一旦该梯度在后续反向图里不再需要就立即 free。于是「完整梯度」从没作为整体存在过，只以分片短暂出现并按层释放。

![ZeRO stage 2 工作原理流程图，展示梯度增量计算与释放及参数更新步骤](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0076_00-20-03.jpg)
图注：Stage 2 逐层发送并释放梯度。

这也让「边界有点漏」的伏笔落地：ZeRO 归在 data parallelism 名下却切了参数与梯度。别把 16.6GB 当成终点；逐层 reduce 与立即释放属 systems trick 而非算法新结论，收益来自对反向计算图生命周期的精细管理。

## 八、从 ZeRO Stage 2 的收尾到 ZeRO Stage 3：把参数也切分 [20:02 - 22:17]

承接 Stage 2 收尾：反向时可增量算梯度、增量发出去（incrementally computing and sending gradients as you go），分批发与一次性发完在通信账上同一回事，做完再 all-gather 参数，于是梯度已切分却 without incurring any additional costs。这正是 Stage 2 的「免费」：靠恒等关系把梯度同步重排成「每算完一片就散出去、算完再收回来」。易错点是免费指字节数不变、不是没有同步，每张卡仍完整走一遍前向反向，只是不再同时持有完整梯度向量。

紧接着讲师说最后、也最 hairy 的一步是把参数也沿所有状态切到底（shard the parameters as well），幻灯片标题 `ZeRO stage 3 (aka FSDP) shard everything`，每卡显存 $P_{os+g+p} = \frac{(2+2+K)\Psi}{N_d}$，右侧标 `1.9GB`。High level idea 三条是 Shard everything – incl parameters!、沿用 incremental communication / computation、Send and request parameters on demand while stepping through the compute graph，于是每张 GPU 任意时刻只同时看到参数、梯度、优化器状态各自的一个切片。这一阶段在 PyTorch 里叫 FSDP（Fully Sharded Data Parallel），讲师补一句用 torch 做过并行的人大概率和它打过交道。

![ZeRO stage 3 (FSDP) 内存占用示意图，展示参数、梯度和优化器状态的切分](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0078_00-20-35.jpg)
图注：Stage 3 切分参数，对应 1.9GB。

底层做法是一次完整更新里做「两次 all gather 加一次 reduce scatter」（two all gathers and one reduce scatter as you go）：先 load model，对第 0 层 all-gather 出完整权重、做 forward 后立刻 free full weights；进入 backward 时 activations 被保留、参数要再 all-gather 一次按需取回，算完 backward 立刻 reduce-scatter 把梯度散回负责的卡，再 free full weights，如此逐层重复最后 update weights。流程图依次是 `LOAD-MODEL SHARD → ALL-GATHER → FORWARD (LOCAL) → FREE FULL WEIGHTS → ALL-GATHER → BACKWARD (LOCAL) → REDUCE-SCATTER → FREE FULL WEIGHTS → UPDATE WEIGHTS (LOCAL)`，底部结论 `Communication cost – 2 all gather (#param), 1 reduce-scatter (#param).`

![FSDP (ZeRO stage 3) 简化版工作流程图，展示前向、反向传播中的通信原语调用顺序](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0081_00-21-23.jpg)
图注：FSDP 通信原语调用顺序。

与前后关系要理清：Stage 2 参数仍完整，Stage 3 连参数也切，故前向反向都须在用到之前把完整权重临时物化、用完再释放。易错点有二：activations 并未被切分、会一直留到反向，所以 Stage 3 省的是参数／梯度／优化器状态那份账而非全部显存；free 必须发生在确认后续计算不再需要该层完整权重之后，否则重复 all-gather 会把通信量翻倍。讲师也埋下伏笔：这个「每层都在通信」看起来疯狂（every layer we're doing this communication），下一章正要解释它为何几乎不花钱。

## 九、通信账本：3×#param 的代价与通信／计算重叠 [22:17 - 25:50]

先算通信账。两次 all-gather 加一次 reduce-scatter 共 $3\Psi = 1.5\times\text{comm}_{\text{DDP}}$（$\text{comm}_{\text{DDP}}=2\Psi$），即 $\text{comm}_{\text{ZeRO-3}} = 2\,\text{all-gather}(\Psi) + 1\,\text{reduce-scatter}(\Psi) = 3\Psi$，相比朴素 DDP 多一次 all-gather。更糟的是频率：不是每 step 通信一次而是每一层都在通信（every layer we're doing this communication），看起来延迟与开销都极贵。*What's the point?* 页写 Distributed data parallel costs 2*# param、Zero stage 1 is 2*# param – it's free!、Zero stage 3 is 3*# param – 1.5x comm cost。括号里的 ignoring overhead 与 ignoring latency 正是易错点：等式只计字节数，不计小消息的延迟与 kernel launch 开销，层数越多越容易被 latency 咬到。

但 Stage 3 用 two really important ideas 把开销几乎抹平：一是沿计算图 sweep through compute graph、做必要通信后立刻释放显存，二是本段新引入的 overlap the communication and computation。讲师为此引了一张解释 PyTorch FSDP 架构论文的时序图，分 CPU、GPU Comp. Stream、GPU Comm. Stream 三行，用 All-Gather(AG)、Reduce-Scatter(RS)、FWD、BWD、Parameter Free 五色标注。逐帧看：FSDP 必须先拿到参数才能算，所以先 all-gather 第 0 层（AG0），完成即做 FWD0，而做 FWD0 时已在请求下一层——I'm all gathering one while I'm doing this computation；AG1 到位做 FWD1 同时开始 AG2。公式 $(W_1 W_0 + W_2 W_0)x = y$ 描述权重复用：if you reuse that W naught，同一份参数可少取一次，把 Parameter Free 与额外一次 FWD0 挤进计算间隙。结论是只要计算量足够大、网络足够快，通信被完全盖住；图上仍有少量无法重叠掉的 bubble。

![FSDP/ZeRO stage 3 实际执行时序图，展示通信与计算的重叠（Overlap）机制](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0086_00-22-43.jpg)
图注：执行时序，对应通信计算重叠。

横向对比：DDP 是 $2\times\#\text{param}$、单次 all-reduce；Stage 1/2 free in the literal sense，只是把一次 all-reduce 拆成 reduce-scatter + all-gather 两个等价操作，字节数完全不变；只有 Stage 3 多一次 all-gather。讲师强调实践中 FSDP 的 GPU 利用率 very close to just the single GPU performance，且 also very conceptually simple——作业就要求自己写 FSDP 实现，做法不过是写一个能包住任意 module 的 wrapper，内部逻辑是 do a bunch of all-gathers, compute, free, and then repeat on the backwards pass。

![ZeRO 各阶段通信开销对比总结，指出 Stage 3 相比数据并行增加 1.5 倍通信成本](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0095_00-25-07.jpg)
图注：各阶段通信开销对比。

易错点与局限：重叠成立前提是「计算 ≫ 通信」，模型很小、层数很碎或跨慢链路时 bubble 与 latency 会重新暴露；1.5x 是字节数之比而非端到端时间之比，不能据此推断训练变慢 50%。

## 十、课堂问答：FSDP 不是流水线，以及 ZeRO 实际能装多大 [25:50 - 28:53]

进入问答，把抽象机制落到具体数字。第一位同学把 FSDP 理解成「顺序传递梯度」，讲师立刻纠正：这更接近 pipelining（流水线并行）——一层放 GPU 0、另一层放 GPU 1 的切法。在所有 ZeRO/DDP 情形里每张 GPU 都要走完整个模型，差别只在于 no GPU is going to hold the entire parameters at the same time；若分片里没有 layer 0，会先按需取回参数再往下走，因此计算与朴素数据并行的单卡计算一样，只是每个操作之间插入了 request parameters / free parameters 两步。

![FSDP实例的前向传播流程图，包含加载、聚合、前向计算和释放权重步骤](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0105_00-27-48.jpg)
图注：FSDP 前向流程，对应请求与释放参数。

第二问「是不是每张 GPU 都持有每层的一部分」，讲师答 Yes, that's all：parameters、gradients、optimizer states 三者都跨 GPU 切分。第三问更有价值：为何通信成本不随层数相乘而爆炸？答案是操作次数确实随层数线性增长，但每次只是「一个很小的 MLP」的数据量，对比基准是对整个网络做一次 all-reduce，把每层小通信加起来总量仍与参数规模同阶——关键记账是总字节数不变，变的是消息粒度与次数。转到实际收益，讲师说单张 A100 上 baseline 连 7B 都装不下，用上 Stage 3 后可装到 50 billion parameter 级别；换到 H200 就是按 141 GB 除以 80 GB 换算，且 it would be linear。

*ZeRO in practice: will it fit?* 页给出精确账本，前提是 Pure BF16 training (with Kahan summation) 与 12 bytes per param，即除 master weights 外全用 BF16、借 Kahan summation 补精度，每参数从 16 字节降到 12。条件是 On a 8X A100 80G，四行是 Baseline `6.66.. B / 12`、Stage 1 `16 B / 5`、Stage 2 `24.62 B / 2 + 10/8 = 3.25`、Stage 3 `53.33 B / 12/8 = 1.5`。用 $8 \times 80\text{GB} = 640\text{GB}$ 除以每参数字节数即得这些最大规模，$640/12 \approx 53.33B$，正解释口述的「7B → 50B」；`12/8` 里的 8 是卡数，`10 (grad+state)` 是梯度与优化器状态、只有参数本身 2 字节没被切，与「Stage 2 不切参数」完全对上。

![ZeRO各阶段在8xA100上可训练的最大模型参数量及每参数字节数公式对比表](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0106_00-28-04.jpg)
图注：8X A100 最大参数量对比表。

易错点：这套数字依赖「BF16 + Kahan summation + FP32 master weights」的特定假设，换成 FP32 Adam 时 16 字节／参数的老经验值会重新生效；表格算的是「能否装下」、不含 activations，长序列下激活内存可能先于参数撑爆显存——正是下一章的动机。

## 十一、数据并行的两个硬限制与模型并行的登场 [28:53 - 31:48]

FSDP 虽 elegant，仍需 proceed into uglier and more hairy things，原因有二。第一个是重要观念：数据并行在消耗一种重要资源，那就是 batch size（data parallel is consuming an important resource... the batch size）。若全局 batch size 为 8，加速器数量永远不可能超过 8；想「一直加大 batch」也不行，因为到了 critical batch size（临界批量），多加一个 batch 元素的收益已低于拿同样算力在这个单样本上多走一步 SGD。小 batch 时增加元素等价于多走 SGD 步，越过临界点后 diminishing return。于是硬性权衡：要小 batch 让 GPU 空闲，还是要大 batch 承受优化损失。

*Issues remain with data parallel* 把这件事画成曲线：纵轴 $\varepsilon_{opt}(B)/\varepsilon_{max}$、横轴 Batch Size / Noise Scale $(B/\mathcal{B})$，左侧沿直线上升标 Perfect scaling，越过 $B/\mathcal{B}=10^0$ 的竖虚线后趋于平台、右上标 Ineffective scaling，即 $\#\text{machines} \le \text{batch size}$。这里的 $\mathcal{B}$（noise scale）是图上独有符号，勿与 micro-batch 的 $b$、全局 batch 的 $B$ 混淆。

![数据并行下预测训练速度随批大小/噪声比变化的曲线图，展示完美扩展与无效扩展区域](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0110_00-29-08.jpg)
图注：训练速度曲线，对应临界批量。

第二个原因是显存仍有问题：Stage 1/2 根本不让显存随卡数扩展，Stage 2 能切参数显存，但 activation memory 完全没被切分或削减，因此需要 more fine-grained ways of cutting up the model。由此引出 model parallelism（模型并行）：它和 FSDP 一样把参数跨 GPU 切分，关键概念差异在于「现在要通信的是 activations」。讲师说得很直白：FSDP 里切的是参数、某种意义上只是个 wrapper，计算照常做、只是把参数来回传（parameters were the things flying around）；现在反过来飞来飞去的是激活。若 layer 0 住 GPU 0、layer 1 住 GPU 1，两者之间要传的就是中间层的 activation。副标题 Scaling up in memory (without changing batch size) with model parallelism 正好绕开 critical batch size 困境。

![模型并行概念介绍页，列出流水线、张量（含序列）和专家三种并行类型](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0116_00-30-44.jpg)
图注：模型并行概念，对应通信激活。

三类是 Pipeline parallel（切层 cutting up layers）、Tensor parallel（切矩阵 cutting up matrices，Percy 已讲过）、Expert parallel（experts 分片放到不同地方）。*Layer-wise parallel* 画出 Layer 0–3 分配到 GPU 0–3、红箭头 forward、蓝箭头 backward 的路径。

![层间并行（Layer-wise parallel）示意图，展示4个Layer分配到4个GPU及前向/反向传播路径](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0120_00-31-48.jpg)
图注：层间并行，对应 4 层分 4 GPU。

这一章是从数据并行转向模型并行的枢纽，后面展开 pipeline 与由 depth/width 类比引出的 tensor parallel。易错点：模型并行不天然比 FSDP 省通信，只是把通信对象从参数矩阵换成激活张量，只有 $b \times s \times h$ 小于参数矩阵时才划算。

## 十二、流水线并行：按层切分、气泡与 micro-batch [31:48 - 34:13]

先讲 pipeline parallel，也叫 layer-wise parallel（按层并行）。讲师说这在概念上简单到有点天真——早年学并行时的第一反应就是「把层切开、放到不同 GPU 上」：前向把 activations 往下一张卡传，反向把 partial gradients 往回传。*Layer-wise parallel* 画了 Layer 0–3 对 GPU 0–3、forward 红箭头与 backward 蓝箭头，页上还压着一行淡字 Utilization of layer-wise parallelism is terrible。

若就这么朴素地做，会得到一张「非常难看、非常 depressing」的图。讲师设定 4 张 accelerator 各负责模型 1/4，时间从左往右扫、不同行是不同卡：第 0 张卡先算一会儿就停下，把计算交给第二张后几乎全程空闲；任何时刻只有一张 GPU 在干活（one GPU will be active at a time），反向同样一次只一张卡算，绝大多数时间 GPU 都 idle，利用率 truly truly terrible。这段空闲就叫 bubble——It's the part of the pipeline where you're not doing anything。接上一章：正因为 FSDP 让每张卡走完整模型、没有这种层间等待才显得 elegant，模型并行一旦按层切开就必须为 bubble 付代价。

![层间并行利用率问题示意图，显示n个GPU中每个仅活跃1/n时间，存在大量空闲等待](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0122_00-32-20.jpg)
图注：利用率问题，对应空闲气泡。

解法本质是 batching / pipelining：不再一个样本接一个样本串行，而把 batch 切成多个 micro-batch，一旦某个 micro-batch 在某 stage 算完就立刻交给下一 stage、并马上开始下一个。*A solution: pipeline parallel* 的甘特图用 $F_{i,j}$ / $B_{i,j}$ 标出 4 个 micro-batch（in this case, 4）在 4 个 stage 上的前向反向方块，最右一列是四个 `Update`，中间那块大空白明确标成 `Bubble`。底部给出定量结论

$$\text{bubble ratio} = \frac{n_{\text{stages}} - 1}{n_{\text{micro}}}$$

紧跟 so we need a big batch size!——气泡占比与 stage 数成正比、与 micro-batch 数成反比，要趋近于零就得把 $n_{\text{micro}}$ 做得很大。

![流水线并行（Pipeline parallel）解决方案示意图，展示4个micro-batch在4个stage上的调度及Bubble时间](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0125_00-33-08.jpg)
图注：micro-batch 调度与 Bubble。

讲师顺势回扣上一章：batch sizes are useful resource，现在能 spend it in a different way，花在加深 pipeline 以减少 idle time。易错点与局限：公式只在每个 micro-batch 前反向耗时相近、stage 划分均匀时成立；更要命的是切很多 micro-batch 会与 critical batch size 冲突——压气泡需要大 batch，大 batch 又带来优化边际收益递减。行内玩笑 parallelization code looks reasonable... until you implement pipeline parallel 也暗示调度复杂度陡增，这正是下一章与 zero bubble 调度要收拾的烂摊子。

## 十三、为什么还要用流水线：通信性质与最慢链路 [34:13 - 36:04]

讲师先自嘲 Okay. So pipelines are terrible，再正面回答「为什么还做」。其一，相比 DDP 能省显存，因为层被切开了（pipelines can save memory compared to DDP cuz we've cut up layers），且还能与 data parallel 组合；其二更重要，在若干拓扑里「你总会有一条 pipeline」（you will always have a pipeline in certain kinds of topologies），因为通信性质非常好。*Why pipeline parallel?* 页开头 Pipelines seem terrible. Why do we do it?，两条编号理由是 1. Pipelines save memory (compared to DDP)、2. Pipelines can have good communication properties (compared to FSDP) – it depends only on activations (b×s×h) and is point to point，底部一句 Generally, we will use pipelines on slower network links (i.e. inter-node)。

![为什么使用流水线并行的两点理由：节省内存和通信特性好（点对点）](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0129_00-34-12.jpg)
图注：两点理由，对应省显存与点对点。

把通信量写清楚：流水线每个 stage 之间只传激活张量 $\text{comm}_{\text{PP}} = b \times s \times h$，其中 $b$ 是 micro-batch 大小、$s$ sequence length、$h$ hidden dimension，it's almost always a smaller amount of data than attempting to communicate a whole parameter matrix——这是紧接的问答里讲师给出的机制（it's batch times sort of sequence length times hidden），而 FSDP 要 all-gather 的是 $\Psi$ 量级参数。更关键的是拓扑性质：它是 point-to-point（one layer is going to communicate to another. It's not all-to-all），只有相邻 stage 一对一来回、不需全员参与的 all-reduce / all-gather。这与硬件章节的节点内／节点间区分一脉相承：张量、专家并行吃带宽留在节点内 NVLink，流水线通信量小、点对点，放到跨节点慢链路。

由此得部署经验 pipelines are going to live on the slowest networking links：若有多个 data center 或多个 pod 彼此通信较慢，就用 pipeline parallel 跨它们切，因为 that's the most communication efficient parallelism you can bring to bear here。省显存还可与 data parallel 叠加——先按数据并行复制若干份、再在每份内部按层切开，于是「省显存」与「扩算力」不必二选一。

讲师还提醒 pipeline performance is really dependent on batch size，并点名反复引用的论文是 NVIDIA 的 Megatron paper，他们做了很多 parameter sweeps of what happens to utilization as a function of different configurations。意义在于利用率不是流水线的固有属性，而是 batch size、pipeline size、micro-batch 数共同决定的函数：if you have large batch sizes and you pick a large pipeline size, you can get pretty good utilization close to not doing pipeline parallel at all，但 Otherwise, pipeline parallel is going to rapidly degrade your utilization and performance。

![流水线并行性能与batch size关系折线图，对比batch size=8和128时不同pipeline-parallel size下的TFLOP/s](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0134_00-35-33.jpg)
图注：性能与 batch size 关系折线。

易错点：$b \times s \times h$ 里的 $b$ 是 micro-batch 而非全局 batch，切得越碎单次消息越小、越容易被延迟主导；长序列 $s$ 或大 $h$ 时这个值会迅速逼近参数矩阵大小，此时「流水线一定更省通信」不再成立。另一局限是气泡与省显存的拉扯：省显存来自切层，压气泡却需大 batch，大 batch 又受 critical batch size 约束。

## 十四、零气泡流水线与沿宽度切分：张量并行的引出 [36:04 - 40:05]

问答先补上通信机制：同学问「为何 pipeline 比 FSDP 通信性质好」，讲师答 it's a smaller amount of stuff being communicated，即 batch times sequence length times hidden（$b \times s \times h$），almost always smaller than communicating a whole parameter matrix。随后是更聪明的调度：把不同阶段分给不同层、不同 micro-batch（cut up the different stages for different layers or different microbatches），把某些 forward 元素插到 backward 之间，靠 clever scheduling 压小气泡，This was taken from the DeepSeek paper。

![通过为每个设备分配多个阶段来用通信带宽换取利用率的流水线调度对比图](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0138_00-36-37.jpg)
图注：多阶段调度，对应压小气泡。

但「更聪明」的是 zero bubble pipelining（零气泡流水线）。讲师强调它 is not really about scheduling per se，而是 thinking carefully about the structure of your backward pass：反向传播要做两件事，一是把 partial derivatives 继续往回传，二是计算当前权重的梯度。关键区别是回传偏导非常要紧、因为下一 stage 在收到信号前没法干活（your next stage can't do any work until you've propagated that signal back），而 $\partial L/\partial W$ 是计算图上的叶子节点（a leaf node, so to speak）、I can do it whenever。*'Zero bubble' pipelining* 左侧写 Split up backwards into two parts：1. Backpropagating activations (z,x)、2. Computing weight gradients (W)，下面一句 The second part can be done whenever；计算图里前向 $z = Wx$、$y = \sigma(z)$，反向 B 段算 $\nabla_z L$ 往回传、W 段算 $W^\top\nabla_z L$ 与 $\nabla_W L$。策略是尽快排完 B 解锁下游、把 W 延后塞进计算空隙，于是 almost all completely fill up the pipeline。

![零气泡流水线的原理：将反向传播拆分为激活值反传和权重梯度计算两部分](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0141_00-37-25.jpg)
图注：零气泡原理，对应反向拆两部分。

右侧 `1F1B` 调度有大量白色空格，`ZB-H1 / ZB-H2` 把 backward 拆成 F／B／W 三色、白色空格几乎消失。讲师评价 very complicated，但效果取决于 W 与 B 的工作量比例——只有 W 足够重才能填满 B 留下的空隙。

收尾转向新类比：pipeline parallel is like depth（沿 depth 维切网络），但我们还有 width axis，What about width? 这就是 tensor parallel（张量并行），Percy 之前已讲过。核心想法简单：matrix multiplies can be cut up into smaller matrix multiplies, which you then add the partial sums back together，且 This is really the same idea as tiling。*Model parallel along the width axes* 把 $X \cdot A = Y$ 切成 $X_1$、$X_2$ 与 $A_1$、$A_2$，得：

$$Y = X A,\quad X = \begin{bmatrix}X_1 \\ X_2\end{bmatrix},\ A = \begin{bmatrix}A_1 & A_2\end{bmatrix} \Longrightarrow Y = X_1 A_1 + X_2 A_2$$

![沿宽度轴进行模型并行的矩阵乘法分解示意图](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0148_00-39-17.jpg)
图注：宽度轴矩阵分解，对应 width 切分。

落到 MLP 就是 $Y = \text{GeLU}(XA)$、$Z = \text{Dropout}(YB)$，把 $A = [A_1, A_2]$ 按列、$B = [B_1; B_2]$ 按行分给不同 GPU 各算子矩阵，再把部分和相加。

![张量并行中GPU持有子矩阵的MLP前向传播结构图](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0151_00-40-05.jpg)
图注：子矩阵 MLP 前向结构。

与前后关系：这一句是本段落点、也是下一段张量／序列并行的开场——沿 width 切没有 pipeline bubble、概念更简单，但代价是每个 matmul 都要伴随 activation 大小的 all-reduce，通信量远大于流水线的 $bsh$ 点对点。易错点：$X_1$、$X_2$ 是沿行切 $X$，$A_1$、$A_2$ 是沿列切 $A$，维度必须对齐才能把部分和相加；把它和 pipeline「沿 depth 切层」混为一谈是初学者最常见的混淆。

## 十五、张量并行的子矩阵切分与前向、反向对偶 [40:05 - 41:31]

本段从最具体的 MLP 计算图切入，说明张量并行（tensor parallel）究竟把什么切开。未做任何并行时，一个 MLP 的串行计算是：输入 $X$ 乘矩阵 $A$，施加逐元素非线性 GELU，再乘 $B$ 做 down projection（降维投影），输出 $Z$，写成 $Y=\mathrm{GeLU}(XA)$ 与 $Z=\mathrm{Dropout}(YB)$。张量并行不是把整层搬到另一张卡，而是把 $A$ 按列切成 $A=[A_1,A_2]$、把 $B$ 按行切成 $B=\begin{bmatrix}B_1\\ B_2\end{bmatrix}$，把这些子矩阵分到不同 GPU 上并行计算、末尾收拢，幻灯片原句是 "Assign columns (A1, A2) and rows (B1, B2) to separate GPUs."。这是理解后续一切切分的基础恒等式。

之所以能这样切，关键在于 GELU 是逐元素（elementwise）非线性：拆成 $XA_1$ 与 $XA_2$ 后有 $\mathrm{GeLU}(XA)=[\mathrm{GeLU}(XA_1),\ \mathrm{GeLU}(XA_2)]$，这一步无需通信；而第二个矩阵乘按行切后每张卡只得到部分和 $Y_1B_1$、$Y_2B_2$，必须相加才得到 $Z=Y_1B_1+Y_2B_2$，这正是通信发生的位置 $g$。真正需要记住的是前向与反向之间的"对偶翻转"："In the forward pass, f is the identity, and g is an all-reduce." 与 "In the backward pass, f is an all-reduce, g is the identity."。前向时 $X$ 只被复制两份（$f$ 是恒等映射 identity），末端 $g$ 用一次 all-reduce（全归约）求和；反向时偏导数从后往前流，$g$ 变成 identity，而 $f$ 处反而要把各卡偏导数累加成 all-reduce。讲师强调 "this duality is kind of important if you're going to write tensor parallel"——若照抄前向的通信布局去写反向，梯度会缺一次求和，模型会静默地算错。

与前文流水线并行（pipeline parallel）对照：流水线切的是"层与层的边界"，张量并行切的是"单个矩阵乘的内部"，因此两者可以叠加。易错点有两个：一是误以为任何非线性都能无通信地按列切，若换成依赖整行统计量的操作（例如跨被切维度的 softmax 或 layer norm），切分后必须补通信；二是把 $f$、$g$ 在前向里的角色直接搬到反向，忽略对偶翻转。本章也埋下下一章的问题：既然有列切、行切两种切法，一个完整 Transformer 块里到底哪些层该用哪种？

## 十六、Transformer 块内的列切、行切与全复制 [41:31 - 42:22]

把上一章的切法推广到整个 Transformer 块，就得到 "Row vs Column tensor parallel" 这张并列画了 (a) MLP 与 (b) Self-Attention 的幻灯片。讲师给出的答案是三条原文 bullet："Columnwise – QKV, up-projection"、"Rowwise – Attention output, down-projection"、"Replicated – norms, routers, etc"。也就是说，每个 transformer block 里，MLP 输入侧的 up-projection 和 attention 的 QKV 投影处在"第一阶段"，按列切（column-wise cuts）；MLP 的 down projection 与 attention 的输出投影处在"对应的第二阶段"，按行切（row-wise cuts）。一进一出恰好拼回完整形状，残差连接才不会对不上维度。Self-Attention 那一半还给出独有信息：模块写成 $Y=\mathrm{Self\text{-}Attention}(X)$ 与 $Z=\mathrm{Dropout}(YB)$，并标注 "split attention heads"，把三组投影按 head 拆成 $Q=[Q_1,Q_2]$、$K=[K_1,K_2]$、$V=[V_1,V_2]$——attention head 本就彼此独立计算，整体分配给不同设备是最自然的切分粒度，且 head 内部无需再通信。

![Transformer块中MLP和Self-Attention层的行/列张量并行拆分对比图](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0157_00-41-41.jpg)
图注：Row vs Column 幻灯片，对比 MLP 与 Self-Attention 的列切、行切与按 head 拆分，对应本段两阶段切法。

第三类层则要完全复制。讲师点名 layer norm、非线性（non-linearity）以及 MoE 里的 routers 这类"小层"（small layer），原话是 "those are all fully replicated across the machines. You don't really want to bother with the overhead of cutting those guys."。原因是这些操作计算量极小、却都依赖完整的隐藏维度或完整的序列统计量；一旦切开，为同步统计量而引入的通信开销远大于省下的那点计算与显存。这把上一章抽象的 $f/g$ 对偶落到了具体层类型上，也直接解释了后面显存章节里那个顽固的 $10\,sbh$ 项：因为 layer norm、dropout 与残差输入是被复制的，它们的激活显存不会被张量并行切小。

易错点集中在两处：一是把列切与行切随意配对——列切的输出维度是原维度除以并行度，必须紧接一个按行切的第二矩阵乘才能相加还原，若两个连续层都用列切却没有 all-gather，形状就会错；二是误以为"复制"意味着这些层不占显存，实际上它们占的是激活显存，且会被完整复制 $t$ 份。这一章把"怎么切"讲清楚了，下一章转向"什么时候值得切"。

## 十七、何时使用张量并行：节点内 8 卡边界、TPU mesh 与流水线并行的取舍 [42:22 - 44:56]

明确了"怎么切"，接下来回答"什么时候值得切"。张量并行极其"吃通信"（communication hungry）：每次 matmul 都伴随通信——前向在 $g$ 处一次 all-reduce，反向在 $f$ 处又一次，且通信量是 activation 大小、发生得"非常频繁"。因此只在节点内（within a node）使用：GPU 场景 8 张卡高速互联装在同一机箱（"eight GPUs ... tightly together in a single box"），"up to eight" 尚可；超过 8 就跨节点（cross-node），interconnect 慢得多，跨过单机边界即 "a big drop in performance"。口诀是：张量并行要放在手头最快的互联上。

![3B模型在不同张量并行度(2/4/8/16/32)下的吞吐量扩展柱状图，显示超过8卡后性能急剧下降](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0160_00-42-30.jpg)
图注：3B 模型随 TP 度扩展的吞吐柱状图，跌幅 -10.8% 至 -65.6%，对应本段节点内 8 卡边界。

"When do we tensor parallel?" 把边界量化了：图名 "Throughput Scaling with TP (3B Model)"，TP=2/4/8/16/32 的相对跌幅为 -10.8%/-12.2%/-42.7%/-65.6%，8 以内缓慢下滑、16 与 32 崩塌，正是"单机 8 卡"经验边界的实证。讲师回扣 TPU：TPU 没有"快的 8 卡 + 慢的跨机"之分，"You just have this big mesh"，规则的通信模式下带宽很高，因此 "you can tensor parallel very large numbers compared to the GPU world"；于是 TPU 与 GPU 之间移植方案时，TP 与 PP 各占比例会相当不同，不能照搬 TP≤8 的配方。

紧接着是与流水线并行的正面比较。Pros 是 "no bubble"、网络够快即满利用率、"low complexity"（只是切开一些 matmul）、不需大 batch；Cons 是 "much larger communication"。两个通信量公式对照：流水线每 microbatch 的 $C_{\text{pipeline}}=b\,s\,h$ 点对点（point-to-point）通信，张量并行每层 $C_{\text{tensor}}=8\,b\,s\,h\cdot\frac{n_{\text{devices}}-1}{n_{\text{devices}}}$ 的 all-reduce，即由 $B\times S\times H$ 变成约 $8\times B\times S\times H$ 的一次 all-reduce，且 "this is not even just point-to-point. This is all-to-all communication"。结论 "Use tensor parallel whenever we have low-latency, high-bandwidth interconnects"，其余大概都用流水线并行。

![张量并行与流水线并行的优缺点对比表，含通信量公式](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0166_00-44-06.jpg)
图注：TP 与 PP 优缺点对比表，含 $bsh$ 点对点与 $8bsh(n-1)/n$ all-reduce 公式，对应本段通信量取舍。

易错点在于：把 "no bubble" 误读成"零通信"，忽略 $\frac{n-1}{n}$ 虽趋近 1、但通信次数随层数线性累积；或把 TP 度盲目设到 16、32 还期待线性扩展。这一章承上启下：前面讲计算与通信，从下一章起转向显存账本。

## 十八、动态显存剖析与每层激活内存公式 [44:56 - 47:51]

本章把话题从通信切到显存。最朴素的显存观是"显存就是参数"（绿色块），进阶会加上 optimizer state（优化器状态，黄色块）；但 "if you look at this plot"——PyTorch profiler 采集的显存曲线——"this is not really the whole story"，标题 "A final complexity – memory is dynamic!"、副标题 "Memory isn't just the static bits, but also activations!"。红色激活区形成明显鼓包（"these big red bumps"）随前向逐层累积，蓝色梯度区在反向扫描时叠加。峰值时刻 "maximum memory usage happens a little bit after the maximum activation point"，即刚进入反向、既存激活又存梯度的那段才是真正最坏情况，所以 "maybe we need to deal with this big red hump, this activation"。

![训练过程中内存随时间变化的堆叠面积图，展示参数、优化器状态、激活值等动态内存占用](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0169_00-44-54.jpg)
图注：PyTorch profiler 显存堆叠面积图与 0.53/0.59 GB 峰值，对应本段"显存动态、红色激活鼓包"。

更大尺度上，中等偏长序列下随着参数变大，baseline 列里 activations 会 "dwarf the parameter memory"，"any sort of memory saving strategy has to sort of reason about activations"，只处理参数显存不够——这把 ZeRO/FSDP 一类切参数方案与切激活方案区分开了。

![不同模型规模(22B/175B/530B/1T)下基线与本文方法的内存对比柱状图，区分参数和优化器状态与激活值内存](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0174_00-46-14.jpg)
图注：22B/175B/530B/1T 规模下参数与激活显存对比柱状图，对应本段激活远超参数的论述。

于是给出"什么都存"的起点公式 $\text{Activations memory per layer}=s\,b\,h\left(34+5\frac{as}{h}\right)$。变量表给出 $a$ 注意力头数、$b$ microbatch size、$h$ hidden dimension、$s$ sequence length、$t$ tensor parallel size。两条 bullet 解释奇怪的第二项："The $5\frac{as}{h}$ terms come from the quadratic attention terms incl dropout" 与 "As with flash attention, we can drop this term via recomputation"——注意 $s\,b\,h\cdot 5\frac{as}{h}=5\,a\,b\,s^2$ 是 $s^2$ 级别，序列一长就主导，别误当随 $s$ 线性增长。讲师强调 $SBH$ "kind of fundamental"：每个序列位置、batch 元素、hidden 维度都要存东西，后面所有公式都会反复出现它；$34$ 是每层经验常数，$24$ 来自 MLP、$10$ 来自点式操作，拆分到下一章才显出来。

![每层激活值内存计算公式及变量定义表](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0176_00-46-46.jpg)
图注：每层激活内存公式 $sbh(34+5as/h)$ 及变量定义表，对应本段起点公式推导。

与前后关系很直接：前三章讲"计算怎么切、通信多贵"，这里转向"切了之后显存省多少"。易错点除了 $s^2$ 项，还有把显存峰值想成前向结束时刻（实际在反向早期），以及以为 optimizer state 永远是大头——长序列下真正的大头是激活。

## 十九、张量并行下的激活内存、序列并行与完全线性下界 [47:51 - 53:21]

把上一章的 $34+5\frac{as}{h}$ 放进张量并行逐项压缩。"Activation under tensor parallel" 给出 $\text{Activations memory per layer}=s\,b\,h\left(10+\frac{24}{t}+5\frac{as}{h t}\right)$。系数来源是 $34=24+10$：$24$ 是 MLP 部分被 $t$ 除成 $\frac{24}{t}$，attention 二次项被除成 $5\frac{as}{ht}$；而 LayerNorm 的 $4sbh$、Dropout 的 $2sbh$、attention 与 MLP 输入（残差）的 $4sbh$ 合计 $10sbh$，张量并行不切这些点式操作，故完全不随 $t$ 缩小。讲师笑称：就算 "a thousand different GPUs in tensor parallel cuz you're Google"，"you're still going to suffer a 10x SBH penalty"，这个固定惩罚一直存在。

![张量并行下的每层激活内存公式及剩余项分解](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0181_00-48-06.jpg)
图注：TP 下每层激活内存公式 $sbh(10+24/t+5as/ht)$，对应本段压不掉的 $10$ 的分解。

处理这 $10\,SBH$ 的就是序列并行（sequence parallel）。讲师先纠正命名："This is an extremely misleading name"，真正配得上的是 context parallel；它把"很轻量、计算量小"的点式操作（"pointwise ops over the sequence"）沿 sequence 轴而非 hidden 轴切分，并在用到它们的前后引入 all-gather 与 reduce-scatter 按需物化。这与 FSDP 概念上很像——平时以切分（sharded）形式存着、用到才 gather、用完即弃，"conceptually very similar to the FSDP style idea"；区别是 FSDP 切参数、序列并行切激活。这也是前文对偶性的复现：前向 $g$ 为 all-gather、$\bar{g}$ 为 reduce-scatter，反向互换。

![序列并行实现图：Transformer层内各模块的张量/序列并行划分](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0186_00-49-26.jpg)
图注：序列并行实现图，标注 "all the 10sbh terms are pointwise ops over the sequence"，对应本段沿序列轴切分。

把配置组合起来得到关键表格：无并行 $s b h(34+5\frac{as}{h})$、纯 TP $s b h(10+\frac{24}{t}+5\frac{as}{ht})$、TP+序列并行 $s b h(\frac{34}{t}+5\frac{as}{ht})$，再加选择性激活重计算（selective activation recomputation）丢掉 $5\frac{as}{ht}$ 得下界 $s b h(\frac{34}{t})$，即 $\dfrac{34\,s b h}{t}$。序列并行把不随 $t$ 缩小的 $10$ 也除成 $\frac{10}{t}$，与 $\frac{24}{t}$ 合并成 $\frac{34}{t}$，实现"对并行度完全线性"；讲师说这是 "kind of the lower bound ... for activation memory"，手算至少留这么多，再加优化器状态、参数与梯度份额即有总显存上界感。

![不同并行配置下每层激活内存的对比表格](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0192_00-51-03.jpg)
图注：不同并行配置下每层激活内存对比表，展示到 $\frac{34\,sbh}{t}$ 的完全线性下界。

段末问答很实用：有学生问 $24$ 能否靠重计算消掉，讲师答 "24 over key is MLP part of this"、"MLP part cannot be optimized through the recomputation"，技术上 "You can recompute the MLP as well"，但要在反向把整个 MLP 再跑一遍、"very computationally expensive"；相比之下 "Recomputation for attention is generally cheaper"，可 tile-wise 做、且 "you don't want to pay the quadratic cost"。易错点：以为 TP 能把激活线性压到近零（$10\,SBH$ 压不掉）、混淆 sequence parallel 与 context parallel、把重计算当免费午餐。本章把公式从"起点"推到"下界"，是整段最密集的定量部分；下一章转向 MoE 的专家并行。

## 二十、专家并行：为何优先于张量并行与低延迟 all-to-all dispatch [53:21 - 57:58]

讲师说明流水线与张量并行都已讲完，"before I move on to expert parallel"，引入"标准工具箱最后一个 ingredient"——专家并行（expert parallelism, EP）。动机是 MoE 已成标配（"most big models are MOEs these days"）。他把 EP 类比 TP：都把 MLP / FFN 切开、都付通信代价，"except here you're splitting up the whole MLP across different devices"；系统行为 "roughly like tensor parallel"，都是高带宽原语、都减少激活。但结论很强：做 MoE 时 "you almost always prefer using an expert parallel sharding strategy over tensor parallel"——不拆矩阵，而是拆专家并路由激活值。

![专家并行（Expert Parallelism）架构：MoE Transformer编码器的设备放置方案](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0201_00-53-27.jpg)
图注：专家并行架构图，标注 "split up the experts and route activations"，对应本段 EP 拆专家的核心概念。

"Why EP?" 嵌入 NVIDIA Megatron 指南截图 "Guideline 4: Prefer EP over TP for Expert Layers"，列四条优势：Better GEMM efficiency（更大本地矩阵提升 GPU 利用率）、Lower communication、Simpler computation graph、Token permutation（EP=num_experts 时消除本地置换），并给例子 "For Mixtral 8x7B, EP8×TP1 outperforms EP4×TP2"。讲师解释一致：把矩阵切太细 "the matrices are going to start to get small, and your GPU utilization will suffer"，matmul 越大越好；MoE 层路由稀疏的 token 激活（sparse token activations）比路由稠密的大 TP matmul 激活容易，且 "you might as well have the number of experts sort of split up over all the devices"。

![Why EP?幻灯片，含Guideline 4表格对比EP相对TP在专家层的优势及Mixtral 8x7B示例](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0204_00-54-15.jpg)
图注：Megatron Guideline 4 表格，列 EP 相对 TP 的四条优势与 Mixtral 8x7B 示例，对应本段"为何优先 EP"。

但"EP 很好、大家都该用"是错的，讲师说 "expert parallel is still really complicated and still very difficult"。他点名 DeepSeek 的 DeepEP："DeepEP is a communication library tailored for Mixture-of-Experts (MoE) and expert parallelism (EP). It provides high-throughput and low-latency all-to-all GPU kernels, which are also known as MoE dispatch and combine."，还支持 FP8、NVLink 到 RDMA 域的非对称转发、纯 RDMA 低延迟 kernel，以及 "hook-based communication-computation overlapping method that does not occupy any SM resource"——把 dispatch 与 combine 藏进计算流水、不占 SM。根因是 MoE 有大量 all-to-all dispatch，每个 MLP 都要路由 token，"you need to do this in a very latency-sensitive way because your computation is waiting for your tokens to arrive"，故降低 dispatch 延迟 "extremely extremely important"。讲师还补趣闻：DeepSeek 用未公开的 PTX（"like GPU machine code"）指令进一步加速网络通信。

![DeepEP通信库介绍及MoE前向/反向传播中dispatch/combine的流水线重叠示意图](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0210_00-55-51.jpg)
图注：DeepEP 通信库与 MoE 的 dispatch/combine 流水线重叠示意，对应本段低延迟 all-to-all dispatch。

易错点：把 EP 当 TP 的即插即用替代、低估 all-to-all 延迟敏感性，或把 DeepEP 当普通集合通信封装——它其实贴着 NVLink、RDMA、SM 占用与 FP8 做系统级优化。讲师还提到 NVIDIA 的 Hybrid EP 是类似努力，并坦言 FSDP 之后讲的张量、流水线、专家并行在 systems-wise 都 "quite non-trivial to get working really well"，别低估工程量。本章承接"切激活"主线并平移到 MoE；下一章讨论组合约束。

## 二十一、并行原语的组合约束与 attention / MLP 解耦 [57:58 - 60:07]

此前所有并行原语都是"孤立地"讲的，隐含假设是能像乐高一样随意拼装（"you can just kind of like put them together like LEGO blocks"）。这对多数策略成立，比如 data parallel 与 tensor parallel 容易组合；但专家并行 "has sort of additional constraints"。第一种来自朴素组合：老库里 "the sort of replicas for DP and EP are the same"，例如 DP=8 就把 8 个 expert 切到 8 个 replica、数据也按这 8 个 replica 切分再路由——"you split your GPUs according to DP and then your EP is sort of a subset of that split"。它因要路由 token 而自然，却给 EP 并行度设了上限，也约束 DP 与 TP 的交互，幻灯片写成 "DP usually shares replicas with EP splits (so EP<DP)"、"DP and TP can interact badly to lower utilization"。Fig. 8 用四个子图并列画出 Data+EP、Data+EP+TP、Data+EP+PP、EP+TP 四种组合，图注说明为简洁省略了 All-to-All、All-Reduce 等通信与 Gate、Encode、Decode 等子模块——这些恰恰是真实系统里最麻烦的部分。

![EP与其他并行策略组合的四种示意图：Data+EP、Data+EP+TP、Data+EP+PP、EP+TP](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0218_00-58-00.jpg)
图注：EP 与 DP/TP/PP 组合的四种示意图，标注组合约束，对应本段 "EP<DP" 的讨论。

第二种更根本，来自 MoE 结构："Recall: MoEs apply to the MLPs, not the attention (with some exotic exceptions)"——MoE 只改 MLP，所以 EP 作用"不均匀"（"applies unevenly to the model"）：MLP 被并行化了，attention 却没有。想并行化 attention 就得 "maybe I want to also use tensor parallel"、用较高 TP 切矩阵，但高 TP 又把 MLP 切得过碎、与 EP 冲突。两条 bullet 摊开矛盾："High TP – useful for attention parallelization (since we cant use EP)" 与 "Low TP – useful for MLP parallelization (we'd rather EP than TP)"。转写在 60:03 的 "if I have high…" 截断，属本段边界，不作猜测。

![解耦attention与expert并行的复杂度讨论，含Megatron MoE Parallel Folding维度表](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0223_00-59-20.jpg)
图注：解耦 attention 与专家并行页及 MoE Parallel Folding 维度表，对应本段 High/Low TP 冲突。

对策是 MoE Parallel Folding：Megatron Core 为两类层各设一套 Parallelism Group，Attention Layers 为 $TP\times CP\times DP\times PP$，MoE Layers 为 $ETP\times EP\times EDP\times PP$，只有 $PP$（pipeline parallel）作公共维度，即 "Separates out TP/CP/DP (for attention) and (ETP/EP/EDP) for MLPs"，解开死结。本章是模型并行工具箱的收束。易错点：把 DP、TP、EP 当完全正交可任意相乘（忽略 EP<DP）；忽略 attention 与 MLP 需不同并行度而设统一 TP；把 MoE Parallel Folding 当通用默认配置，其实它只是 Megatron Core 的解耦方案。它与前文关系是：前几章给出如何把激活压到 $\frac{34\,sbh}{t}$ 下界，本章说明组合时还须在维度分配上额外权衡。

## 二十二、解耦 attention 与专家并行：MoE Parallel Folding 与上下文并行 [60:07 - 61:59]

本段从上一段结尾接过来。[60:09] 点出矛盾核心：TP 与 EP 都开很高时 "I've cut up my matri- matrices into really tiny pieces, and my utilization is very bad"。根因仍是 "Recall: MoEs apply to the MLPs, not the attention (with some exotic exceptions)"：EP 只作用 MLP，于是 attention 层没有 EP 只能靠 High TP 切 QKV / 输出投影，MLP 层已有 EP 就只想要 Low TP 甚至 "we'd rather EP than TP"。讲师在 [60:17] 概括为 "you want high tensor parallel for the attention, but you want low tensor parallel for the MLPs ... there's this conflict"。这不是调参问题，而是结构性的 "imbalance between attention and MLPs for parallelism"，动机是切得太碎导致算力浪费、而非显存不够，与后面纯显存驱动的 FSDP / PP 决策是不同性质的约束。

解法是 MoE Parallel Folding，写作 "MoE Parallel Folding is Megatron Core's solution that decouples attention and MoE parallelism"。它不给全网强加统一维度，而为两类层各定义一套：Attention Layers 为 $TP\times CP\times DP\times PP$，MoE Layers 为 $ETP\times EP\times EDP\times PP$，即 $\underbrace{TP\times CP\times DP\times PP}_{\text{Attention}}\ \longleftrightarrow\ \underbrace{ETP\times EP\times EDP\times PP}_{\text{MoE}}$，只有 $PP$ 公共，底部注解 "Separates out TP/CP/DP (for attention) and (ETP/EP/EDP) for MLPs"。$ETP$（expert tensor parallel）、$EDP$（expert data parallel）是画面独有、转写未念出的标识符；注意 $CP$ 在 attention 一侧而 $EP/ETP/EDP$ 只在 MoE 一侧，并非简单换位。转写 [60:26]–[60:56] 复述：让 "the attention layers get one kind of tensor parallel, and the MOE layers get another kind"，从而 "fully decouple the parallelism"，得到 "more complicated, but more effective combinations"。

这套解耦是「专家并行」一章的延伸，也引出 [61:00] 的收尾话题 context parallel（上下文并行），也叫 ring attention：本质是 "splitting activations in a very long sequence across different accelerators"，按 TPU 的 mesh topology 以 "kind of this ring-like way" 把 KV 环形传给下一设备；ring attention "was the original paper that did this ... showed that this worked really well on TPUs"。它用于 long context extension 与 model serving，但因与前面激活切分高度重叠，讲师 "I'm not going to talk about it mainly because I think it overlaps in concept"。

![Context parallel / Ring attention示意图：长序列激活在多GPU间按query block与Key Value Block环形传递](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0230_01-01-12.jpg)
图注：Context parallel / Ring attention 示意图，按 query block 与 Key Value Block 环形传递，对应本段上下文并行讲解。

易错点有两个：想当然给全网设全局 TP 度，正确做法是按层族拆维度；以及把 context parallel 当可替代 TP/EP 的独立"第五种并行"，其实它只是长序列补充手段，后面讲 Llama 3 长上下文扩展、Nemotron 3 Super 时才调大。幻灯片特意写 "with some exotic exceptions"，提醒"MoE 只作用于 MLP"也有例外。下一章用一张大表收束整个第二部分。

## 二十三、并行策略大表与「没有银弹」 [62:00 - 63:59]

[61:52] 讲师宣布 "this is the end of the second part"，用 "Recap: LLM parallelism table" 汇总第二部分。他说明 "I've colored in red what I subjectively feel like are the drawbacks of the different methods"，目的是传递 "there is no one strictly dominant parallelization strategies. It's all like a whole bunch of trade-offs"。表格列有 Method、Comm/Sync、Param memory per rank、Activation / KV memory per rank、Main bandwidth cost、Scales global batch?、Easy to use?，行覆盖 DDP / ZeRO-1、FSDP / ZeRO-3、Pipeline、Tensor、Sequence / Context、Expert (MoE) 六种；"the top two rows" 是数据并行族、下面是模型并行。显存缩放写成 $\text{DDP}=1,\ \text{FSDP}\sim\tfrac{1}{DP},\ \text{PP}\sim\tfrac{1}{PP},\ \text{TP}\sim\tfrac{1}{TP},\ \text{EP}\sim\tfrac{1}{EP}$。逐行读，DDP / ZeRO-1 的 Comm/Sync 是每步 "gradient all-reduce"，FSDP / ZeRO-3 是 "Gradient all reduce – can be overlapped"，Pipeline 是层间激活并带红字 "pipeline bubbles"，Tensor 是红字 "Blocking activation communication"，Expert (MoE) 是 "token dispatch - all-to-all per MoE"。

![LLM并行方法对比表，列出DDP/ZeRO、FSDP、Pipeline、Tensor、Sequence、Expert并行的通信同步、显存占用与带宽成本](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0233_01-02-00.jpg)
图注：LLM 并行方法对比大表，列六种策略的通信、显存与带宽成本，对应本段红字缺点解读。

[62:27] 起讲师拿 FSDP 举例把红字串起来：FSDP "is great. It's a wonderful strategy"，但 "does not help you with activation"（Activation = None），且 "it sort of consumes your global batch size"（Yes, linear in DP 反成限制），"So, it has limits"。补短板就得组合："maybe you want to do some tensor parallel to cut down on activation memory"，TP "also doesn't touch global batch size"，但 "now you need fast networking, and you need sort of high bandwidth"；慢链路再交给其它策略。不同策略各占掉一种资源（batch size、激活显存、快链路带宽），必须搭配用，才使得 "in many large-scale architectures use a large combination of these"。例如 TP 虽不耗 batch size 却需要节点内高带宽，EP 虽省激活却要求每专家分到足够 token，FSDP 虽易用却把全局 batch 当燃料，单看一列必误判。

红字是主观标注、不是测量值。例如 "Scales global batch? = Yes, linear in DP" 对 DDP / FSDP 反被标红，因它把 global batch size 当燃料；PP 的 "No, but needs microbatches"、EP 的 "No, but needs enough tokens per expert" 也是红字，说明「不随 batch 扩展」只是把约束换成 micro-batch 数或每专家 token 数。读表须把 "Scales global batch?" 与 "Main bandwidth cost" 交叉看，这埋下下一章把定性表格变成 $B/N$ roofline 的伏笔。[63:20] 学生问 pipeline 与 FSDP 是否仍适用 MoE，讲师答 "Yes, yes, they're absolutely applicable"，说标准建议 "expert parallel ... you want to go up to like eight ... fast connections"，但又说 "People don't really follow that anymore"，并强调 "FSDP and or data parallel in general, and pipeline and tensor are used extensively in all the frontier models"。易错点：把表当「打分表」选赢家，红色恰说明每列都有反面；把 "Easy to use? = Very" 当推荐度——DDP/ZeRO-1 之所以 Very，正因它什么显存都没省，这才要用 FSDP 或模型并行去补。

## 二十四、roofline 定量分析：计算与通信的比值 [64:00 - 66:19]

[63:58] 讲师说并行化 "one of the things that's cool ... is that you can do some math"：逐层算不同 sharding strategy 的 compute 与 communication，再看 "how do these things scale"。横轴 batch size、纵轴本质是 "the ratio of compute time to communications time"。核心判据 "as long as my compute time is longer than communications time ... you can just hide communications underneath your computation"。标题 "Model vs Tensor parallel (TPU book)" 的幻灯片把它画成 roofline：左侧 "Key quantity: Global batch size (divided by GPU)"，右侧 "Batch-size scaling behavior of parallelism strategies on a 4x4x4 mesh"——只要比值大于 1，通信就能藏在计算底下。这把上一章的定性大表变成可计算账本。

![模型并行与张量并行的批大小扩展行为曲线图，含各策略每层计算量与通信量公式表](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0241_01-04-08.jpg)
图注：TPU book roofline 图，含逐层计算/通信公式表与三种策略随批大小扩展曲线，对应本段定量分析。

左下角逐层 compute 与 comms 对照表：DP 与 FSDP 计算量相同（$\tfrac{4BDF}{X}+\tfrac{8BDF}{X}$），差别在通信项（FSDP 多 $4DF$ 参数 all-gather）；加入 MP 后计算被 $Y$ 摊薄、通信变成与 $BD$ 相关的激活，如 FSDP+MP 的 comms 为 $\left(\tfrac{4BD}{X}+\tfrac{4DF}{Y}\right)+\left(\tfrac{8BD}{X}+\tfrac{8DF}{Y}\right)$。右图纵轴 $\text{FLOPS time}/\text{Comms time}$，$10^{0}$ 处虚线标 "Computation bound"、其下橙线标 "Communication bound"；FSDP Only、MP Only、FSDP + MP 三条曲线配三段注释："No scheme works when $B<400$"、"Only mixed FSDP + MP works when $B<850$"、"Both ... work when $B>850$"。这些阈值与公式是看图才补齐的量化信息。右图横轴是 $B/N$（batch size 除以总芯片数），两条竖直虚线约在 400 与 850 把图分成上述三段；曲线之上即 "Anything above that is sort of fully utilizing your computation"，之下则 "You're waiting for communication to do your computation"，讲师用这句点明 roofline 两侧的含义。

[65:14] 起口语版："If your batch size is big enough, FSDP only is good"，例如 "batch size of 2,000 ... per chip" 时 "You're fully compute bound"；但 "as the batch size goes down, FSDP only will hit a point where you're now communication bound"，掉到 roofline 下就得 "incorporate MP, in this case is tensor parallel, to the mix ... push that curve out ... be compute bound even into smaller batch size regimes"，再 "keep adding strategies"。[66:05] 起名："this is what people call 3D or 4D ... I don't think there's 5D ... parallelism"，即 "put all these strategies together to keep your compute units fully utilized"。3D/4D 不是新算法，而是把 roofline 高效区往小 batch 延展的策略堆叠。

![3D/4D并行组合示意图，展示数据并行rank内pipeline stage与张量并行的层级结构及经验法则](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0249_01-06-17.jpg)
图注：3D/4D 并行组合示意图及经验法则，展示 DP rank 内 pipeline stage 与 TP 层级，对应本段组合策略。

这章与前后咬合：大表说 FSDP "Activation = None"、TP "需高带宽"，这里用 $B/N$ 解释为何 batch 大 FSDP 就够、batch 小须加 TP；下一章把「叠加」落成可执行顺序。易错点：把 batch 当绝对值，关键量是 $B/N$（除以总芯片数），[65:24] 补 "per chip" 正是此意；以及误以为叠加越多越好——FSDP+MP 只在中段胜出，超 850 后纯 FSDP 同样 compute bound，且 $B<400$ 时 "No scheme works"，此时该动 gradient accumulation 而非并行维度。

## 二十五、3D/4D 并行的简单配方与 Megatron 实践指南 [66:20 - 68:06]

[66:19] 讲师说前面 "generalities" 让人觉得复杂，但 "the strategy for how you parallelize is very simple, actually"。"Simple rules of thumb from the literature" 两步法：第一步 "Until your model fits in memory.." 用 "Tensor / expert parallel up to GPUs / machine"、"Pipeline parallel across machines"、"（Or use Zero-3, depending on BW）"；第二步 "Then until you run out of GPUs"，"Scale the rest of the way with data parallel"；补 "If your batch size is small.. gradient accumulate"。口述给具体数字："until your model fits into memory ... cut up your model by whatever means necessary"，"if you have eight GPUs per machine, you're going to use a tensor or expert parallel of eight"（$TP=8$ 或 $EP=8$），然后 "pipeline parallel or zero three, you know, which is FSDP, the rest of the way to make your model fit"，"once your model fits ... data parallel the rest of the way"；"if your batch size ends up being too small, you can do gradient accumulation"。讲师用 "Very simple strategy" 收尾。这套顺序把 roofline 里「保持 compute bound」的目标翻译成按显存→GPU 数推进的贪心流程。右侧 GPU 网格图把 Data Parallel Rank 0 / 1 各画成三个 Pipeline Stage（Network Layers 0-7、8-15、16-23），用不同颜色块标出 TP、EP、CP、DP、PP 叠在同一批卡上，是「3D/4D」最直观的图解。

[67:08] 起讲师说这套思路 "is born out in the actual ... large-scale parallel training libraries"，给出 "Megatron's like guide to sort of paralyzing MOEs"，强调 "this idea actually applies to the dense models as well"。画面 "Example from Megatron's current recommendations" 拆成 basics 与 advanced extras 共五条：Guideline 1 "Minimize Model Parallelism, Maximize Data Parallelism"（"Keep TP/EP/PP as small as possible while avoiding OOM"、`--use-distributed-optimizer`）；Guideline 2 "Keep EP and TP Communication Within NVLink Domain"（EP+TP 留单节点、通常 8 卡，跨节点优先 PP）；Guideline 3 "Use Pipeline Parallelism (PP) for Multi-Node Scale"（`--num-layers-per-virtual-pipeline-stage` 控 VPP 减气泡）；Guideline 4 "Prefer EP over TP for Expert Layers"；Guideline 5 "Enable Context Parallelism (CP) for Long Sequences"（`--context-parallel-size`、Sequence lengths ≥ 8K）。Guideline 4 还给出 "For Mixtral 8x7B, EP=8+TP1 outperforms EP=4+TP2"、"When EP = num_expert, local token permutation is eliminated"；Guideline 5 点明 "CP efficiency depends on overlapping communication with computation"，几个配置项可原样记为 `--use-distributed-optimizer`、`--num-layers-per-virtual-pipeline-stage`、`--context-parallel-size`。

![Megatron当前推荐的并行配置指南，分基础与进阶两部分列出五条准则及具体参数设置](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0253_01-07-21.jpg)
图注：Megatron 当前并行推荐指南，列五条准则与参数设置，对应本段"最小化模型并行最大化 DP"。

把几条与 [67:25] 口述对照，顺序正好相反："you see exactly the thing that I said, but kind of in reverse order"——文档从 "Minimize Model Parallelism, Maximize Data Parallelism ... use most of your GPUs to shard data" 写起，随后才是 "stay within NVLink ... one box"、"if you're going to go multi-node, use pipeline parallelism"、"if you're MOE, then prefer expert parallel"、"if you're doing long sequences, use context parallel, right? Ring attention"；课堂则从「先切到模型装得下」讲起，两者是同一贪心链的正反叙述。易错点：DP 并行度不能超过全局 batch size（大表里 "Yes, linear in DP" 是双刃剑），需 gradient accumulation 兜底；[68:27] 问答补 "sequence parallel is ... more like an add-on ... It's not its own like standalone thing in many cases"，它常只是 TP 的附赠优化；Guideline 2 的 "single node (typically 8 GPUs)" 只是经验值，被 DeepSeek-V3 的 64-way EP、Qwen 3 的 EP=32 打破。此外 "avoiding OOM" 提醒这是按「先装进显存」排序，若模型本就装得下最省事的反是纯 FSDP，正如后面的 OLMo 案例。这套配方也预告了后面两个课堂问答与全部真实案例。

## 二十六、课堂问答、定量扩展证据与激活重计算 [68:07 - 72:30]

[68:07] 讲师 "maybe I can pause here in case anyone's curious about like 4D parallelization stuff" 引出问答。第一条把 sequence parallel 定性为 "an add-on ... not its own like standalone thing in many cases"，常与 TP 绑定、只为削激活。第二条学生问 "loop transformer" 会不会让并行更易或更难，讲师先说 "I don't know how much I buy the loop transformer mythos rumors"，再从系统角度答："it might mess with things like FSDP. Like you can't really discard the weights ... it is much more parameter efficient and so much maybe much of the model parallelism stuff wouldn't be as important"。关键在 [69:25]：FSDP "is really predicated on like get weights, discard weights, get weights, discard weights and in a loop you're not discarding anything"——FSDP（ZeRO-3）依赖「按层 all-gather 取权重、算完即释放」，loop transformer 反复复用同一组权重，通信隐藏前提被釜底抽薪。问答把 FSDP 隐含假设显性化：它不是无条件省显存。

[69:36] 转入 "an old paper but ... one of the best resources"，作者含 "formerly unfortunately Stanford folks Matei and Deepak"，做法是 "did a bunch of runs of large-scale parallel training across many different configurations"。结论 [70:14] 很具体："you have data parallel sort of maxed out and then your tensor parallel kind of increases until it hits eight and then you stop ... from that point on your pipeline parallel increases and increases"，"at the very end you have a DP of six"；[70:46] 补 "even as you go to ludicrously large numbers of GPUs ... your utilization stays very flat and very good"，即 $PTD\text{-}P=PP\times TP\times DP$ 的 "More GPUs, same, flat utilization"，而纯 ZeRO-3 在固定全局 batch 下每 GPU 吞吐随卡数下滑。论文即 Megatron-LM 的 "Careful '3D' parallelism gives linear gains"：图例四条曲线为 ZeRO-3 175B/530B 与 PTD-P 175B/530B，橙色 PTD-P 几乎水平（530B 约 173、175B 约 150→143 teraFLOP/s per GPU），蓝色 ZeRO-3 随 GPU 数下滑（530B 约 140→100→50、175B 约 145→88→45）。

![Narayanan 2021论文中Megatron-LM不同规模模型的并行策略配置表](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0256_01-08-09.jpg)
图注：Narayanan 2021 并行策略配置表，标注 "Tensor parallel first up to 8, then caps out at 8"，对应本段缩放规则。

[71:19] 补两点：论文给出 "tensor parallel size of eight. As you go beyond that, you're going to get into trouble"；以及 "In pipeline parallel, you also need big batch sizes ... That's kind of why there's a blue to orange gap ... the larger the pipeline parallel, the larger the gap"，即 PP 越大越依赖大 batch 压气泡。

![不同流水线-张量并行配置组合下的每GPU吞吐量折线图](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0268_01-11-21.jpg)
图注：不同流水线-张量并行配置的每 GPU 吞吐折线图（Figure 13，"Tensor parallel = 8 is often optimal"），对应本段 TP=8 最优停止点。

[71:50] 起抛出反直觉观察："maybe you should do a bunch of activation recomputation because if you do activation recomputation very cleverly, then you can get a bigger batch size and the bigger batch size then allows you to get better utilization"，因果链 "more computation via recomputation ... allows you to save memory and memory can be turned into batch size"，即 $\text{recompute}\Rightarrow\text{save memory}\Rightarrow\text{larger }B\Rightarrow\text{compute-bound}$，[72:04] 点明是 "initially I think very counterintuitive observation"。这与 roofline 呼应：重计算不是目的，把显存换成更大 $B/N$、让曲线回到 computation bound 之上才是。Figure 10 的结论句正是 "More GPUs, same, flat utilization!"，重计算让这条平坦曲线在小 batch 下也能维持。

![激活重计算与不重计算的吞吐量随批大小变化对比折线图](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0270_01-11-53.jpg)
图注：激活重计算与否的吞吐随批大小对比折线图，"Activation recomputation can pay for itself (via memory)"，对应本段讨论。

易错点：把「多算」直接等同「变慢」——显存受限时多花算力重算反而把通信藏得更好；也别把平坦曲线当普遍规律，它前提是 "Global batch sizes are fixed" 且 "ZeRO-3 is used without any model parallelism"，一旦 batch 不固定或给 ZeRO-3 配 MP，结论要按 $B/N$ 重估，这正是讲师强调 "careful 3D parallelism" 而非随便堆并行的原因。

## 二十七、现代训练案例（一）：OLMo、DeepSeek 与 Llama 3 405B [72:31 - 76:23]

[72:33] 讲师想通过真实配置看 "how modern parallelization is used in practice and how it's evolving"。第一个例子是 AI2 的 OLMo：转写先说 "Dolma by AI2 was a 7B open source model" 随即更正 "sorry, Olmo, which was trained on the Dolma data set"，是 "a 7B model, trained fully with FSDP across a whole bunch of accelerators"。屏幕引用论文 3.1 节 Distributed Training Framework：用 "the ZeRO optimizer strategy ... via PyTorch's FSDP framework"，7B 规模下 "a micro-batch size of 4096 tokens per GPU"、全局约 4M tokens（$2048\times 2048$），混合精度 bf16 计算、fp32 累加。讲师据此总结 "FSDP scales surprisingly well, even to like many, many GPUs, as long as you're training a small model ... many 7B-ish models are trained purely with FSDP"，与「模型装得下就别上模型并行」一致。画面底部结论 "Dolma – 7B model, FDSP (probably fits intra-node)" 里 FDSP 显系 FSDP 笔误，提醒幻灯片字符串要原样辨认、按 FSDP 理解。

![Dolma 7B模型使用FSDP分布式训练框架的论文段落截图](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0272_01-12-25.jpg)
图注：Dolma 7B 论文 3.1 节，标注 ZeRO optimizer strategy 与 PyTorch's FSDP framework，对应本段 OLMo 案例。

[73:23] 起 DeepSeek "does do some pretty exotic and useful stuff"。V1 "was trained just with data parallel zero stage one, with tensor sequence and pipeline parallel"；屏幕 2.4 Infrastructures 写框架 "HAI-LLM"，"Data parallelism, tensor parallelism, sequence parallelism, and 1F1B pipeline parallelism are integrated"，还用 flash attention、ZeRO-1、算子融合、"train the model in bf16 precision but accumulate gradients in fp32 precision" 与 in-place cross-entropy。V3 是 MoE，"they have not only pipeline parallel, but they also have expert parallel, instead of tensor parallel"，且 "64-way parallelism ... group eight different machines together"，并 "use the same tricks ... from their pipeline parallel pipelining ... to try to make sure that expert parallel doesn't ... have low utilization periods"——靠 1F1B A2A Overlap、通信计算重叠、算子融合与就地交叉熵让 64-way EP 不空转。[74:22] 的 Yi 则是 "zero stage one and tensor and pipeline parallel ... the classic ... combo"，且 "once you go to MoEs, you replace the tensor parallelism with expert parallelism ... a little bit more efficient"。V3 幻灯片压缩成 "V3 – PP (16), EP (64-way, 8 nodes), ZeRO stage 1" 与 "EP uses 1F1B A2A Overlap"。

![DeepSeek使用HAI-LLM框架及多种并行策略的基础设施论文段落截图](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0276_01-13-30.jpg)
图注：DeepSeek 2.4 Infrastructures 引文，标注 HAI-LLM 与数据/张量/序列/1F1B 流水线并行，对应 V1 配置。

![Yi模型性能与成本效率的论文段落及并行策略总结](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0280_01-14-34.jpg)
图注：Yi 模型段落，标注 ZeRO-1、tensor parallel combined with pipeline parallel 与 kernel fusion，对应本段组合策略。

[74:49] 起 "on the other extreme ... you have Llama 3"，报告有 "a full breakdown of their parallelism strategy ... across all the different phases"。主预训练 "Tensor parallel of eight, one sort of context parallel, 16 pipeline parallel, and 128 data parallel"，即 $TP=8,\ CP=1,\ PP=16,\ DP=128$；长上下文阶段 "crank up the context parallel, lower the data parallel"，因该阶段 "very memory-hungry"。讲师提醒 "we're in tensor parallel, not expert parallel land, because Llama 3 405B is a gigantic dense model. It's not a MoE model"。[75:52] 起插入 infra 旁注："GPUs fail all the time"，画面给 54 天预训练中断根因表：Faulty GPU 148 次占 30.1%、GPU HBM3 Memory 72 次占 17.2%、Software Bug 54 次占 12.9%，约 78% 归因硬件；转写 [76:06] 正对应 "GPUs failed 148 times"，引出 "you need lots of not just like fast parallelism, you need redundancy ... So, there's a distributed systems challenge as well"。表格还有 pre-pre-training 暖机行（"You can ignore that first row"）与主预训练、长上下文扩展三行，并另列 GPU SRAM、NIC、NCCL Watchdog Timeouts、Silent Data Corruption 等细项。

![Llama3 405B不同训练阶段的GPU并行配置与MFU数据表](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0281_01-14-50.jpg)
图注：Llama 3 405B 各阶段 GPU 并行配置与 MFU 表（8,192 GPUs、TP=8、PP=16、DP=128），对应本段配置。

![Llama3 405B预训练期间GPU故障根因分类统计表](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0285_01-15-54.jpg)
图注：Llama 3 405B 预训练 GPU 故障根因表（Faulty GPU 148 次、30.1%），对应本段 148 次故障旁注。

易错点：别把 Llama 3 与 DeepSeek-V3 配方混同，前者 dense 用 TP=8、后者 MoE 用 EP=64；别把 148 当总中断数，它只是 "Faulty GPU" 一行，表注只说约 78% 归因硬件、未给总数，不要自行相加反推；更别忽略「并行策略再好也扛不住硬件故障」这一层，它把讨论从算法扩展拉回分布式容错。

## 二十八、现代训练案例（二）、Megatron Bridge 与全课总结 [76:24 - 80:10]

[76:23] 起补完案例清单，"Gemma 2 ... has FSDP plus tensor parallel and sequence parallel, and uses these two strategies only"；理由是 "a realization of the Google claim that ... for TPUs, really you don't need to do pipelines, you just take a really big toroidal mesh, and you do tensor tensor parallel over that big mesh"。TPU 的大 toroidal mesh 让 TP 铺得很大、不需 PP，但他也标注 "Unclear whether this can like scale out forever ... but at least at the Gemma scales, this is definitely the case."。这与前面 TPU book roofline（"4x4x4 mesh"）、Guideline 2 的「NVLink 域内 TP」形成对照：同一配方在不同硬件收敛到不同组合。

![Gemma 2 模型训练的计算基础设施与并行策略说明页](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0287_01-16-26.jpg)
图注：Gemma 2 基础设施页，标注 2/9/27B 用 ZeRO-3、MP(=TP+SP)、DP 及 512/4096/6144 chips，对应本段 TPU mesh 无需 PP。

[77:04] 起介绍 "Nvidia has a repository called uh Megatron Bridge, where they sort of release a lot of recommended training configurations"。Mixtral 8x22B 写 "TP / PP / CP / EP (4/4/1/8) – DP likely 2 (to add up to 256 GPUs)"；转写 [77:39] 讲成 "expert parallel of eight, pipeline parallel four, and an additional tensor parallel of four. This one for the attention layers"，并指出 "this follows the prescription of keep your expert parallel roughly around eight"——TP=4 专留给 attention、EP=8 留给专家，正是 MoE Parallel Folding 解耦思路的落地。Megatron Bridge 的价值不在给某个「最优配置」，而在把不同规模、不同设置的推荐配置摊开对照；DeepSeek-V3 一行是 671B/1024 GPUs、TP=2/PP=16/CP=1/EP=64（"Massive MoE with 256 experts"），Mixtral 8x7B 则是 64 GPUs、TP=1/PP=4/CP=1/EP=8。

![Mixtral 8x22B 及 MoE 模型的 GPU 并行配置表](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0290_01-17-14.jpg)
图注：Mixtral 8x22B 等 MoE 的 GPU 并行配置表，标注 TP/PP/CP/EP (4/4/1/8)，对应本段 Bridge 配置。

[77:53] 的 "NeMo Tron 3 Super ... has a bunch of expert parallel, as well as context parallel, because they were doing extensions"；[78:03] 的 Qwen 3 "follows the DeepSeek recipe ... fairly large amount of expert parallel, 32 ... eight pipeline parallel and two tensor parallel to split up kind of things like the attention matrices"，即 $EP=32,\ PP=8,\ TP=2$。讲师还说 Bridge repo 里 "There's all sorts of different like sub configurations ... even within tensor parallel that can significantly affect the performance ... the systems integration is quite significant"，故 [78:33] 明确 "I won't go into details of what this is"。[78:49] 起 "decided maybe I would build overview"，给三条共性："all the models use data parallel. They maximize the data parallel domains"、"Tensor parallel almost always remains below eight"、"expert parallel can sometimes be big now ... partially because of Deep Seek V3 and a lot of the infra"，把「最大化 DP / TP 封顶 8 / EP 可放大」用真实数据坐实。

![Nemotron 3 Super 120B-A12B 长上下文扩展阶段的训练细节](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0293_01-18-02.jpg)
图注：Nemotron 3 Super 长上下文扩展细节，标注 64-way context/tensor/expert parallelism，对应本段叠加上下文并行。

![Qwen 3 系列模型的并行配置表及硬件带宽对照表](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0294_01-18-18.jpg)
图注：Qwen 3 并行配置表，标注 TP=2 PP=8 EP=32 Total GPUs=512，对应本段 EP=32/PP=8/TP=2 配置。

![多个大模型并行策略（DP/TP/EP/PP/CP）总览对比表](/coursenotes/cs336-language-modeling-sp26/l08-parallelism-ii/cand_0296_01-18-50.jpg)
图注：多模型 DP/TP/EP/PP/CP 总览对比表，标注 "TP generally <= 8. EP can be big (but hard!)"，对应本段规律总结。

把十余案例并排看，规律比单个数字更有说服力：dense 模型（OLMo 7B、Yi、Llama 3 405B、Gemma 2）几乎都落在 FSDP / TP / PP / DP 经典链上，MoE 模型（DeepSeek-V3、Mixtral 8x22B、Nemotron 3 Super、Qwen 3）一律把 TP 换成 EP、或压到只够切 attention 的小数值、同时把 EP 顶上去，长上下文阶段统一调高 CP；并行度只是骨架，通信重叠、算子融合、容错冗余才决定能否跑满。[79:20] 讲师 "to put everything together" 给出 "The highest-level point ... is we need to think about multi-GPU, multi-node, maybe even multi-data center parallelism"，要 "bring to bear all the approaches that we have because you've got fast links, you've got slow links ... all these different resources"，但最终 "there's fairly simple rules of thumb for combining all these different forms of parallelism to get ... full utilization of your computer hardware"。屏幕 "Recap for the whole lecture" 三条收束全课："Scaling beyond a certain point requires multi-gpu, multi-node parallelism"、"No single solution to the parallelism problem (probably want all 3 approaches)"、"Simple, interpretable rules of thumb for combining different forms of parallelism"；[80:01] 末句 "And next week I think we're talking about scaling loss" 把话头交给下一讲。易错点：把 "all 3 approaches" 读成某三种固定策略清单，它其实指按资源（快链路、慢链路、batch size）分工组合；"TP below eight" 是硬件经验而非数学上限，被突破的是 EP 而非 TP；也别把案例罗列当新的推荐配置，选型仍要回到自己硬件的 $B/N$、链路带宽与模型是否装得下。

## 要点回顾

- **课程目标与扩展动机**：理解训练超大模型的系统复杂性、多种并行范式为何要叠加、真实大规模训练长什么样；最大规模须多机多卡乃至多数据中心，最终 the new unit of compute is not the GPU, it's the entire data center。
- **硬件拓扑与两大瓶颈**：算力与显存逼迫模型 shard，intra-node 极快、inter-node 较慢；TPU 是 toroidal mesh（deg(chip)=O(1)）、GPU 是 fat tree，H100 SuperPod 256 卡 bisection 57,600 GB/s。
- **集合通信记账**：all-reduce、all-gather、reduce-scatter 等原语，恒等式 all-reduce ≡ reduce-scatter + all-gather 让两步算法免费做出。
- **朴素数据并行**：把 $B$ 切成 $B/M$ 分给 $M$ 台机器，每卡得 $B/M$ 样本、每步传 $2\times$#params、Memory scaling = none。
- **显存账本与 ZeRO 三阶段**：Adam 下每参数约 16 字节（2+2+4+4+4）；从 $(2+2+K)\Psi=120\text{GB}$ 到 $(2+2+K)\Psi/N_d=1.9\text{GB}$（$K=12,\Psi=7.5\text{B},N_d=64$，约 63 倍），先切优化器状态最值钱。
- **Stage 1/2/3**：Stage 1 的 ReduceScatter + All Gather 仍为 $2\times$#params 故 "this was free"；Stage 2 反向逐层 reduce 并 free；FSDP／ZeRO-3 shard everything，一次更新 $3\Psi=1.5\times\text{comm}_{\text{DDP}}$，靠立即 free 与 overlap 逼近单卡性能，但 activations 并未被切。
- **数据并行硬限制与三类模型并行**：#machines ≤ batch size、越过 critical batch size 收益递减，引出 pipeline（切层）、tensor + sequence（切矩阵）、expert（切专家），从传 parameters 改为传 activations。
- **流水线并行**：micro-batch 的 $F_{i,j}/B_{i,j}$ 交错压气泡，bubble ratio $=(n_{\text{stages}}-1)/n_{\text{micro}}$、需大 batch；$\text{comm}_{PP}=b\times s\times h$ 点对点放在最慢跨节点链路；Zero bubble 把 backward 拆成关键路径 B 与可塞空隙的 W。
- **张量并行子矩阵切分与 f/g 对偶**：$Y=\mathrm{GeLU}(XA)$、$Z=\mathrm{Dropout}(YB)$，$A$ 按列切、$B$ 按行切到不同 GPU，GELU 逐元素无需通信、只在 $g$ 处 all-reduce 求 $Z=Y_1B_1+Y_2B_2$；前向 "f is the identity, and g is an all-reduce"，反向 "f is an all-reduce, g is the identity"。
- **Transformer 块切分与全复制**：Columnwise 用于 QKV、up-projection，Rowwise 用于 Attention output、down-projection，layer norm、非线性、routers 等小层 fully replicated，其激活显存被复制 $t$ 份、正是后面 $10\,sbh$ 的来源。
- **TP 的 8 卡边界与 TP/PP 取舍**：communication hungry 只在节点内，3B 吞吐在 TP=2/4/8/16/32 跌幅 -10.8%/-12.2%/-42.7%/-65.6%；$C_{\text{pipeline}}=bsh$ 对 $C_{\text{tensor}}=8bsh\cdot(n-1)/n$，TPU 大 mesh 可把 TP 放得很大。
- **动态显存与起点公式**：PyTorch profiler 显示红色激活鼓包、峰值在反向早期；大模型长序列下 activations dwarf parameter memory，起点 $\text{Activations memory per layer}=sbh(34+5\frac{as}{h})$，$5as/h$ 是 $s^2$ 级 quadratic attention terms、可经 recomputation 丢掉，$SBH$ 根本性。
- **TP、序列并行与线性下界**：TP 下 $sbh(10+\frac{24}{t}+5\frac{as}{ht})$，残留 $10\,SBH$（LayerNorm 4 + Dropout 2 + 输入 4）压不掉；sequence parallel（真正配得上名字的是 context parallel）沿序列轴切点式操作、概念似 FSDP 但切激活，前向 g 为 all-gather、$\bar{g}$ 为 reduce-scatter、反向互换；组合表格到 $sbh(\frac{34}{t})$ 即 $\frac{34\,sbh}{t}$ 下界，MLP 重计算很贵、attention 重计算更便宜。
- **专家并行与低延迟 dispatch**：EP 类比 TP（high bandwidth, reduces activation）但 MoE 下 "prefer EP over TP"，Guideline 4 列 Better GEMM efficiency / Lower communication / Simpler computation graph / Token permutation，例 "Mixtral 8x7B, EP8×TP1 outperforms EP4×TP2"；DeepEP 提供 all-to-all dispatch/combine、FP8、NVLink→RDMA、不占 SM 的 hook 重叠，根因是 latency-sensitive 的通信，DeepSeek 还用未公开 PTX 指令。
- **组合约束与 attention/MLP 解耦**：naive 做法 "DP usually shares replicas with EP splits (so EP<DP)"、"DP and TP can interact badly"；因 "MoEs apply to the MLPs, not the attention" 产生 High TP（attention）与 Low TP（MLP）冲突，MoE Parallel Folding 给 Attention 层 $TP\times CP\times DP\times PP$、MoE 层 $ETP\times EP\times EDP\times PP$、仅 $PP$ 公共；context parallel / ring attention 沿 TPU mesh 环形传 KV，用于 long context extension 与 model serving。
- **并行大表「没有银弹」**：六种策略红字缺点，Param memory per rank 约 1、1/DP、1/PP、1/TP、1/EP；"there is no one strictly dominant parallelization strategies"，FSDP 不帮 activation 又耗 global batch size，需搭配 TP 与慢链路策略；红字是主观标注，须把 "Scales global batch?" 与 "Main bandwidth cost" 交叉看。
- **roofline 定量分析**：关键量是 $B/N$（Global batch size divided by GPU）、纵轴 FLOPS time/Comms time；DP 与 FSDP 计算量同为 $\tfrac{4BDF}{X}+\tfrac{8BDF}{X}$、FSDP 多 $4DF$；阈值 "No scheme works when $B<400$"、"Only mixed FSDP + MP works when $B<850$"、"Both ... work when $B>850$"，"as long as my compute time is longer than communications time ... hide communications underneath"，叠加即 3D/4D parallelism。
- **简单配方与 Megatron 五准则**：Until your model fits in memory 用 TP/EP up to GPUs/machine、PP across machines、或 Zero-3；Then scale the rest with DP，batch 小则 gradient accumulate。Megatron 顺序相反：Minimize MP / Maximize DP、Keep EP+TP within NVLink、Use PP for multi-node、Prefer EP over TP、Enable CP for long sequences（≥8K、--context-parallel-size），配置含 --use-distributed-optimizer 与 --num-layers-per-virtual-pipeline-stage。
- **问答与定量扩展证据**：sequence parallel 是 add-on 而非独立维度；loop transformer 破坏 FSDP 的 "get weights, discard weights" 前提；Narayanan 2021 给出 DP 拉满、TP 到 8 封顶、之后加 PP、末端 DP=6，$PTD\text{-}P=PP\times TP\times DP$ 的 "More GPUs, same, flat utilization" 对纯 ZeRO-3 随卡数下滑；"tensor parallel size of eight" 是最优停止点，PP 越大越需大 batch。
- **激活重计算反直觉收益**：recompute ⇒ save memory ⇒ larger $B$ ⇒ compute-bound，"memory can be turned into batch size"；前提是 "Global batch sizes are fixed" 且 "ZeRO-3 is used without any model parallelism"，要按 $B/N$ 重新评估，别把「多算」直接等同于变慢。
- **真实案例（dense）**：OLMo/Dolma 7B 纯 FSDP（micro-batch 4096 tokens、全局约 4M）；DeepSeek V1 是 ZeRO-1 + Tensor/Sequence/Pipeline（HAI-LLM、1F1B、flash attention、bf16/fp32、in-place cross-entropy）；Yi 是经典 ZeRO-1 + TP + PP；Gemma 2 用 FSDP + TP + SP、不用 PP，印证 TPU 大 toroidal mesh 不需 pipeline。
- **Llama 3 405B 与容错**：主预训练 $TP=8,\ CP=1,\ PP=16,\ DP=128$，长上下文阶段 crank up CP、lower DP，因是 gigantic dense model 仍在 tensor parallel land；54 天里 Faulty GPU 148 次（30.1%）、GPU HBM3 Memory 72 次（17.2%），约 78% 意外中断归因硬件，"GPUs fail all the time" 引出冗余与分布式系统挑战。
- **真实案例（MoE）与 Megatron Bridge**：DeepSeek V3 是 PP(16), EP(64-way, 8 nodes), ZeRO-1 且 EP uses 1F1B A2A Overlap；Mixtral 8x22B 为 TP/PP/CP/EP (4/4/1/8)、DP≈2 凑 256 GPUs；Qwen 3 为 EP=32/PP=8/TP=2；Nemotron 3 Super 叠加 context parallel；同一 TP 度下不同子配置显著影响性能，"the systems integration is quite significant"。
- **三条共性与全课总结**："all the models use data parallel"（maximize DP domains）、"Tensor parallel almost always remains below eight"、"expert parallel can sometimes be big now"；全课以 "Scaling beyond a certain point requires multi-gpu, multi-node parallelism"、"No single solution to the parallelism problem (probably want all 3 approaches)"、"Simple, interpretable rules of thumb for combining different forms of parallelism" 收束，最终目标是 full utilization of your computer hardware，末句把话头交给下一讲 scaling loss。

## 视频来源

https://www.youtube.com/watch?v=6-cXp-aOmdg

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
