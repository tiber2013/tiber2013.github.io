---
title: "第 7 讲 · 并行化 I：多卡互连与集合通信"
description: "这节课把视角从单张 GPU 内部扩展到多卡、多节点集群，核心问题仍然是数据搬运：算力单元离数据很远，多卡时数据可能在另一张卡甚至另一个机架上。课程先建立集合通信原语（broadcast、scatter、gather、re…"
---

# 第 7 讲 · 并行化 I：多卡互连与集合通信

> **Stanford CS336 语言模型从零构建 · L07** ｜ 讲者 Stanford Online ｜ 81 分钟 ｜ [原视频](https://www.youtube.com/watch?v=SzpOcwdIL0Y)

> **v2 详写版** · 按 20 分钟分段详写，配图覆盖全片；文末附「对照参考笔记补遗（Spring 2025 版）」，记录与 2025 版讲义笔记的核对结果。

## 核心结论

这一讲把视角从单张 GPU 内部扩展到多卡、多节点集群，核心矛盾却始终是同一个：算力单元离数据很远，单卡时数据在 HBM、多卡时可能在另一张 GPU 上，因此用一大堆 GPU 容易、把它们用得有效很难，全讲都在编排计算以避免数据搬运成为瓶颈。课程先建立源自 1980 年代、以 rank 与 world size 描述的集合通信原语，其中 all-gather、reduce-scatter、all-reduce 是训练主力，恒等式 all-reduce = reduce-scatter + all-gather 既解释梯度同步，也是 FSDP/ZeRO 拆分存储的前提。硬件一侧给出从 L1/shared memory、HBM、NVLink/NVSwitch 到 InfiniBand、Ethernet 越远越慢的层级，RDMA 让 GPU 绕过 CPU 直读写远端显存，NCCL 负责探测拓扑、选路并发射通信 kernel，torch.distributed 把它们落成 all_reduce 等代码，有效带宽基准实测约 400 GB/s。第二部分在深层 MLP 上实现三种切法：数据并行沿 batch 切、每卡持全参并靠一次 all-reduce 平均梯度，张量并行沿 width 切、每层用 all-gather 拼回激活，流水线并行沿 depth 切、用 micro-batch 与点对点 send/recv 减少气泡。三者通信量与互连要求各异，选择哪种并行本质是在计算冗余、显存占用与通信开销之间权衡，而模型放不下时的 FSDP/ZeRO 留待后续。

## 从单 GPU 到多 GPU：同一个数据搬运瓶颈 [00:00 - 03:17]

本讲开门见山地把视角从「单张 GPU 内部」切换到「多张 GPU 之间」。讲者先回顾上一周（last week）的主题：通过手写 kernel 深入 GPU 内部，研究怎样让一张卡跑得更快，那一周反复拆解的硬件单元包括高带宽显存 HBM（High Bandwidth Memory）、L2 cache、L1 cache、寄存器（registers）以及一组流式多处理器 SM（streaming multiprocessors）。而本周（this week）要回答的问题是：怎样利用多张 GPU 让代码跑得还要更快。画面上那张架构图最能说明变化——上周只盯着其中一个方框，本周把它横向复制成 4 张、甚至 1000 张 GPU，并用互连把它们接起来；至于这些 GPU 具体怎么连，讲者明确留到后面才展开。这里要建立的核心直觉只有一个：无论单卡还是多卡，算术逻辑单元（arithmetic logic units）和 tensor core 这些「算力」距离「数据」都很远；单卡语境下「远」指的是数据在 HBM 里，多卡语境下「远」可能意味着数据躺在另一张 GPU 上，你得想办法把它「铲」（shovel）过来。因此两种情况在放大之后是同一个游戏——编排（orchestrate）计算，尽量避免数据传输成为瓶颈。讲者用一句话点破难点：用一大堆 GPU 很容易，但把它们用得有效（effectively）很难。

![四张 GPU 经 NVLink 接入 NVSwitch、再下连 InfiniBand/Ethernet 的架构图，展示单卡内部 SM/L2/HBM 层级](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0001_00-00-37.jpg)

顺着这个直觉，幻灯片给出了一份「广义层级」（Generalized hierarchy），把单卡的存储层次和多卡的通信层次统一成一条从快到慢的谱系。值得注意的是，上周还被嫌弃「太慢」的 HBM，到了本周的多卡视角里反而被算作「快」的一层，因为现在它比跨卡、跨节点的链路近得多。下面这段按幻灯片 bullet 原样转述，可以当成后面所有带宽决策的地图：最靠近 SM 的一层是单节点单 GPU 内部的 L1 cache / shared memory（fastest，最快），其下是单节点单 GPU 的 HBM，再往下是单节点多 GPU 之间通过 NVLink/NVSwitch 相连，最外层是多节点多 GPU 只能借助 InfiniBand/Ethernet（slowest，最慢）。

```text
Generalized hierarchy:  (fastest -> slowest)
  Single node, single GPU: L1 cache / shared memory   (fastest)
  Single node, single GPU: HBM
  Single node, multi-GPU:  NVLink / NVSwitch
  Multi-node, multi-GPU:   InfiniBand / Ethernet       (slowest)
```

这份层级不是凭空列出来的，它直接对应前后内容的衔接。最上面两层正是上周用融合（fusion）与分块（tiling）去优化的对象，做法是「读进 shared memory、尽量在里面把能做的都算完、再写回 HBM」；而下面两层是本周新增的战场，对应幻灯片里高亮的那句「This week: reduce communication across GPUs/nodes via replication/sharding（复制/分片）」。换句话说，优化手段从「减少访存」升级为「减少跨 GPU、跨节点的通信」，但目标始终没变：让算力尽量别等数据。把层级写成上面这种自顶向下的清单还有一个好处——它和上一章的存储墙、下一章的带宽代价是同一套物理事实，只是换了一个尺度来描述。

![幻灯片列出广义层级四个层级，并高亮「本周通过复制/分片减少跨 GPU 通信」](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0003_00-03-09.jpg)

这里有几个容易踩的坑。第一，别把多卡并行当成一个全新问题：它只是把「算力离数据远」这条老规律在空间尺度上放大了，上周理解的存储墙（memory wall）本周变成通信墙（communication wall），所以「In both cases, compute is far from inputs/outputs (data)」里那个 both cases 才是全讲的总纲。第二，HBM 的「快/慢」是相对的，取决于你把边界画在哪一层；一旦引入 NVLink/NVSwitch 与 InfiniBand/Ethernet，越往下带宽越窄、延迟越高，这个物理约束会一直延续到后面选择数据并行、张量并行还是流水线并行。第三，讲者特别强调「很容易铺开、很难用有效」，意思是单纯把 GPU 数量堆到 4 张或 1000 张并不自动带来加速，若编排不当，通信开销会把多出来的算力吃光；这句提醒也正是下一章讨论「为什么要用多 GPU」以及「何时反而要付带宽代价」的伏笔。最后要留意一个细节：讲者说这张图是「taking a little bit of liberty（稍作自由发挥）」画出来的，也就是说层级是概念模型而非严格的硬件规格，不能把「4 张 GPU」「1000 张 GPU」当成某个具体机型的固定配置。

## 为什么需要多 GPU，以及本课的两部分结构 [03:17 - 05:45]

把图铺开之后，讲者马上追问「So why do you do multi-GPUs」，并给出两个比「为了扩展（scale）」更精确的理由。第一个理由是「放不下」：参数（parameters）、激活（activations）、梯度（gradients）和优化器状态（optimizer state）加起来可能超出单张 GPU 的 HBM，幻灯片把它写成「Your parameters (optimizer state + gradients + activations) don't fit on a single GPU」。这里有一个非常具体的数字锚点：B200 的 HBM 是 192 GB，而如果你要训练一个 1 trillion（一万亿）参数的模型，光是参数本身就远超这张卡的容量，更别提优化器状态、梯度与激活这些额外开销。第二个理由是「想更快」：即便模型能塞进单卡，你也可能想把工作拆开、利用更多 GPU 的 FLOPs 来缩短训练时间，也就是「You want to use more GPUs (more FLOPs) to train faster」。讲者随即点出两者之间要做权衡——如果全部塞在一张 GPU 上，你拥有的核心（cores）更少；一旦铺开，你获得了更多算力，却必须为通信带宽（communication bandwidth）付账，所以「that's some calculation you're going to have to do to figure out how to parallelize」，即到底怎么切才划算是需要算账的。

这一段的幻灯片还夹着一段很实用的代码运行说明，属于画面独有、口头只略略带过的信息，必须原样转述。讲者提醒：到目前为止这门课的惯例都是「Python 写完直接执行、把结果全部展示出来」，但本讲如果直接运行，会走 multiprocessing（多进程）；而在课堂上逐行 trace（追踪）时，他切到了某种「special single process mode（单进程模式）」，所以此时「we're not actually doing multiprocessing」。如果想看到本讲真实的标准输出，需要按多进程方式重新运行，幻灯片上留了一个可点击的 `[stdout for this lecture]` 链接。源文件里的注释把两条路径写得很清楚：

```python
# When you execute this lecture directly (python lecture_07.py),
#   it uses multiprocessing, which produces output for each process (below).
# However, when you trace this lecture (python -m edtrace.execute -m lecture_07),
#   we turn off multiprocessing.
# [stdout for this lecture]
```

这里的易错点相当典型：很多同学照着课堂里逐行打印的输出去理解分布式程序，会误以为那就是多进程并发的真实行为，其实课堂追踪时特意关掉了分布式，打印顺序和真实多进程环境并不一致；想知道每个 rank 各自 print 了什么，得老老实实用 `python lecture_07.py` 跑一遍。这也和上一章的层级形成呼应——单进程模式只是为了方便讲解，真正的多卡通信仍然要落到后面讲的 NCCL 与 torch.distributed 上。

接着讲者把本讲切成两部分（two parts）。第一部分是「building blocks of distributed communication/computation」，也就是分布式通信与计算的基础构件：先讲编程模型（programming model，即集合操作 collective operations），再讲一点硬件（hardware，GPU 究竟怎么连），然后在 torch 里动手实现（对应作业二 assignment 2 会做的事），最后还要测量真实带宽。第二部分才是「actual training」，看三种并行策略如何以不同方式切分模型。幻灯片用一组带注释的函数名把大纲列得一目了然，这里原样抄录：

```python
# Part 1: building blocks of distributed communication/computation
collective_operations()   # Programming model
hardware()                # Hardware: how GPUs are connected
torch_distributed()       # How this is implemented in NCCL/PyTorch
benchmarking()            # Measure actual NCCL bandwidth

# Part 2: distributed training
#   Walk through bare-bones implementations of each strategy on deep MLPs.
#   Recall that MLPs are the compute bottleneck in Transformers,
#   so this is representative.
data_parallelism()        # Cut up along the batch dimension
tensor_parallelism()      # Cut up along the width dimension
pipeline_parallelism()    # Cut up along the depth dimension
```

![课程大纲：Part 1 四个基础构件函数与 Part 2 三种并行策略的切分维度注释](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0005_00-05-41.jpg)

三种并行方式分别「沿 batch 维度切」「沿 width 维度切」「沿 depth 维度切」，这个约定会在后续章节反复出现，务必记牢。还有一个容易误解的地方：讲者强调用 MLP（多层感知机）而不是完整 Transformer 来演示，但原因是「MLPs are the compute bottleneck in Transformers, so this is representative」——MLP 是 Transformer 里的主要计算瓶颈，所以在它上面展示核心计算是有代表性的，并不意味着这些并行方法只适用于 MLP。此外 Part 1 只是打地基，真正的训练要到 Part 2 才出现；幻灯片末尾还留了一句「What's missing?」，暗示本讲覆盖的并非全部并行方式。这段大纲与前后内容的关系是：它把上一章抽象的「广义层级」落实成可执行的四个函数，又把下一章要讲的「集合通信原语」定位成整个 Part 1 的第一块砖，后面所有 all-reduce、all-gather 的开销，最终都要回到「为什么当初要付这份带宽」的账上来算。

## 集合通信原语：编程模型、rank 与 world size [05:45 - 08:16]

正式进入第一个函数 `collective_operations()`。讲者先把集合通信（collective operations）的来历讲清楚：它们是分布式编程里的「primitives（原语）」，可以追溯到 1980 年代（go back to the '80s），并行编程本身「is very old」，并不是为语言模型训练才发明的，然而「it's still the case that these primitives are the ones that we use today」——我们今天做大模型训练用的仍是这一套。接着给出「collective（集合）」这个词的准确含义：你指定的是一个「general communication pattern or a template across multiple devices（跨多设备的通用通信模式或模板）」，而不是自己逐条管理「这张 GPU 怎么和那张 GPU 点对点（point-to-point）通信」。好处是写起来简单得多，而且「the system can do a lot more work for you」，系统（底层库）能替你做大量优化，因此这是一个「very tried and true interface（久经考验的接口）」。幻灯片把这几条 bullet 和函数签名写在一起：

```python
def collective_operations():
    # Collective operations are the conceptual primitives used for
    # distributed programming [article]
    # - These are classic in the parallel programming literature from the 1980s.
    # - Collective means that you specify a general communication pattern
    #   across many devices.
    # - This can be better/faster than managing point-to-point
    #   communication yourself.
```

![集合操作定义：1980 年代经典原语、跨多设备通信模板，以及 Rank 0–Rank 3 的 Setup 示意](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0006_00-06-57.jpg)

概念之后是术语。讲者坦言这套命名「a little bit strange（有点怪）」，但它是并行编程里的标准（standard），必须接受。两个最基本的词是 rank 与 world size：rank 对应「a particular device（某一个设备）」，在本课语境下通常就是一张 GPU，不过幻灯片特意补了一句「(e.g., 0, 1, 2, 3)」并说明「could be a TPU」，也就是说 rank 并不被 GPU 绑死；world size 则是「total number of devices（设备总数）」。当 Setup 图里画出 Rank 0、Rank 1、Rank 2、Rank 3 四个方框时，world size 就是 4。用记号写就是 $W = 4$，而 rank 取自集合 $\\{0, 1, 2, 3\\}$——注意这里 rank 是从 0 开始编号的，和后面张量下标、数组分片的习惯一致。把「设备」抽象成编号 rank 的意义在于：后面所有操作都不再描述「哪张物理卡」，而是描述「编号为 r 的 rank 持有什么、要发给谁」，这样才能把通信模式写成与硬件无关的模板。

接下来讲者一口气列出要讲的 8 个操作：broadcast、scatter、gather、reduce、all gather、reduce scatter、all reduce、all to all，并强调「each of these operations is going to specify how this set of ranks or devices is going to transfer some amount of data / compute with it to some other set of devices（每个操作都规定一组 rank 如何把一定量的数据搬运、并与数据一起做计算，送到另一组设备）」。幻灯片把这些操作按用途分了三组，这个分类直接决定了本段的讲解节奏：

```text
Operations:
  Broadcast, scatter, gather, reduce        (foundations)   # 基础/热身
  All-gather, reduce-scatter, all-reduce    (workhorse)     # 训练主力
  All-to-all                                (For MoEs)      # 用于混合专家
```

讲者把前四个（broadcast、scatter、gather、reduce）称为「warm-ups（热身）」，说它们「allow you to get a sense of how these collective operations work, but they're not really going to be the ones that are driving most of training」——它们帮你建立直觉，却不直接驱动训练主路径；真正「show up again and again for distributed training of language models（在语言模型分布式训练里反复出现）」的是 all-gather、reduce-scatter、all-reduce 这三个 workhorse；最后的 all-to-all 只对 MoE（Mixture of Experts）重要，本讲不花太多时间（幻灯片里这条被高亮成「All-to-all (For MoEs)」）。

这一章有几个必须澄清的易错点。第一，不要把 rank 想成「必须是 GPU」：定义里写的是 device，TPU 也算，这与前面广义层级里「GPU/节点」的具体硬件是刻意解耦的。第二，collective 是「模板」而非「点到点」，你描述的是整体模式，底层如何选路、如何分块由系统决定，这正是它「better/faster」的原因，也意味着你不该在业务代码里手写 send/recv 去替代它。第三，别因为前四个是 warm-up 就跳过——讲者明确说 scatter 是理解 reduce-scatter 的 stepping stone、gather 是理解 all-gather 的 stepping stone、reduce 是理解 all-reduce 的 stepping stone；跳过基础，后面三个 workhorse 的「all」「scatter」「reduce」组合就会变成死记硬背。第四，这一章建立的正是 Part 1 的 programming model，与上一章「广义层级」互为表里：层级讲物理带宽，集合操作讲逻辑通信模式，后面测带宽、实现并行都要同时用到这两者。

## 四个基础操作：broadcast、scatter、gather、reduce [08:16 - 12:32]

从最简单的 broadcast（广播）开始。讲者说「you have a rank zero. It could be any rank, but let's just say for sake of picking one, rank zero」——源可以是任意 rank，只是为了方便固定选 rank 0。它持有某个张量，幻灯片上写的是 `tensor([0., 1, 2, 3])`（口头转写里那串含糊的「0 1 0 1 2 3」显然应以此为准），broadcast 之后「each of the ranks has the same tensor on it（每个 rank 都拿到完全相同的张量）」。这个操作「doesn't really show up in the core path of training」，典型用途是初始化：rank 0 加载初始 checkpoint，然后广播给所有 rank，「something that's done like once（基本上只做一次）」。对应代码与幻灯片上的小字注释如下：

```python
# Broadcast: copy from rank 0 to all ranks
rank0 = tensor([0., 1, 2, 3])          # Input
# Output
rank0 = tensor([0., 1, 2, 3])
rank1 = tensor([0., 1, 2, 3])
rank2 = tensor([0., 1, 2, 3])
rank3 = tensor([0., 1, 2, 3])
# Minor use case: rank 0 loads initial checkpoint and broadcasts to all ranks
```

第二个操作 scatter（散射）与 broadcast 形似而神不同。它说的是「I have a tensor at rank zero, whose is split up into the world size（rank 0 上有一个张量，按 world size 切成若干份）」，然后「scatter my tensor onto the other ranks」：rank 0 拿到第 0 个分量、rank 1 拿到第 1 个、rank 2 拿到第 2 个、rank 3 拿到第 3 个。换句话说 broadcast 是「一份复制成 W 份」，scatter 是「一份拆成 W 块、每卡拿一块」，输出张量的维度从 4 降到 1：

```python
# Scatter tensor on rank 0 to all ranks
rank0 = tensor([0., 1, 2, 3])          # Input
# Output
rank0 = tensor([0.])
rank1 = tensor([1.])
rank2 = tensor([2.])
rank3 = tensor([3.])
# Note: stepping stone to understanding reduce-scatter
```

讲者顺带解释了它的价值：「you want all the GPUs you're scattering to to do some local computation on the different parts（你希望被散射到的各张 GPU 能对不同分片各自做局部计算）」，这正是后面数据并行、张量并行的雏形。幻灯片也点明 scatter 本身不被直接使用，只是「a important stepping stone to understand reduce-scatter」。紧接着的 gather（收集）被讲者称为「the inverse of scatter」，输入是散落在各 rank 上的若干块，gather 到指定 rank（这里仍是 rank 0）后「just concatenate all the pieces together（把各块拼接起来）」，于是又从 4 个 1 维张量变回 `tensor([0., 1, 2, 3])`。reduce（归约）则「和函数式编程里的 reduce 完全一样」：起点与 gather 相同，但不是拼接，而是对所有分片施加某个归约运算，把结果放到 rank 0。用 sum 时，0 + 1 + 2 + 3 得到 6；讲者还给了一个很妙的统一视角——「you can think about gather as a reduction where the operation is concatenation if you will（可以把 gather 看成运算为『拼接』的 reduce）」。这两段代码如下：

```python
# Gather pieces from all ranks to rank 0 (opposite of scatter)
rank0 = tensor([0.]); rank1 = tensor([1.]); rank2 = tensor([2.]); rank3 = tensor([3.])
# Output
rank0 = tensor([0., 1, 2, 3])
# Note: stepping stone to understanding all-gather

# Reduce pieces from all ranks to rank 0, applying some operation (e.g., sum, min, max)
rank0 = tensor([0.]); rank1 = tensor([1.]); rank2 = tensor([2.]); rank3 = tensor([3.])
# Output
rank0 = tensor([6.])   # Sum of all ranks (0 + 1 + 2 + 3)
# Note: stepping stone to understanding all-reduce
```

![Gather 把各 rank 的单元素张量拼回完整张量，并高亮 Reduce 的 sum/min/max 归约定义](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0009_00-10-45.jpg)

中间还插入了一段问答（12:05–12:32）：有人问这是否和 NumPy 里的 broadcasting 有关。讲者的回答是「conceptually the same idea where you have one thing that goes to many things」——概念上都是「一个东西变成很多个」，比如 NumPy 里把标量（scalar）广播成张量；「but the instantiation, this is for collective communication, so it's a bit different（但具体实例化不同，这里指的是集合通信）」。这个澄清值得记牢：中文都叫「广播」，但 NumPy 的 broadcasting 是免复制的按形状对齐语义，而这里的 broadcast 是一次真实的跨设备数据复制，二者不能混为一谈。

这一章其余的易错点还包括：reduce 的运算必须是「some associative commutative operation（某种满足结合律、交换律的运算）」，幻灯片举例 sum、min、max，因为只有这样才能把归约拆成树形或环形分步执行；gather / reduce 的目标 rank「不必预先固定」，调用时指定即可，这一点在后续问答里也被确认；四个操作都是垫脚石而非训练主路径，若误以为训练就是反复 broadcast 会完全走偏。最后把它们和前后连起来看：broadcast/scatter 是「一到多」，gather/reduce 是「多到一」，二者互为逆；把「多到一」的 reduce 与「一到多」的 scatter 按维度组合起来，就是下一章 reduce-scatter，而把 gather 推广到所有 rank 就是 all-gather——整个 workhorse 家族其实只是这四个基础操作的组合与「all」化。

## 训练主力：all-gather、reduce-scatter、all-reduce [12:32 - 16:48]

讲者用「let's move on to something more interesting」引出 all-gather：它就是「perform gather to all ranks, not just rank 0（对所有 rank 都做 gather，而不只对 rank 0）」。那句「All means do it for all the output to all the ranks」把前缀 all 的含义钉死了——输出目的地是所有设备。它为什么重要？讲者提前剧透：「later we'll see that each rank hold part of the parameters and then what you need to do is all gather the parameters to get the full parameters for the full forward pass（之后会看到每个 rank 只持有一部分参数，做完整前向之前必须 all-gather 出完整参数）」，并预告训练过程会「see a lot of this gather to do something and then scatter and then gather and scatter again（大量地 gather→做点什么→scatter→再 gather）」。用上一章 gather 的输入，all-gather 的输出是四个 rank 都拿到完整张量：

```python
# All-gather: perform gather to all ranks, not just rank 0
rank0 = tensor([0.]); rank1 = tensor([1.]); rank2 = tensor([2.]); rank3 = tensor([3.])
# Output
rank0 = tensor([0., 1, 2, 3])
rank1 = tensor([0., 1, 2, 3])
rank2 = tensor([0., 1, 2, 3])
rank3 = tensor([0., 1, 2, 3])
# Use case: each rank holds parameter shard, gather to get full parameters for forward pass
```

第二个 workhorse 是 reduce-scatter（归约-散射），定义是「performing reduce on each dimension and then scattering the results（对每个维度分别做 reduce，再把结果散射出去）」。这里的输入不再是 4 个单元素张量，而是 4 个长度为 4 的向量，构成一个 4×4 的矩阵：rank0 持有 `[0., 1, 2, 3]`、rank1 持有 `[1., 2, 3, 4]`、rank2 持有 `[2., 3, 4, 5]`、rank3 持有 `[3., 4, 5, 6]`。所谓「对每个维度」就是按列归约：第 0 维 0+1+2+3=6、第 1 维 1+2+3+4=10、第 2 维 2+3+4+5=14、第 3 维 3+4+5+6=18，然后这 4 个结果分别落到 rank0～rank3。用公式写，若 $x_r[d]$ 表示 rank $r$ 上第 $d$ 个分量，则

$$s_d = \sum_{r=0}^{W-1} x_r[d], \qquad (s_0, s_1, s_2, s_3) = (6,\ 10,\ 14,\ 18)$$

归约结果 $s_d$ 被放到 rank $d$，这就是「scatter the results」。幻灯片把这段代码和注释写得很完整：

```python
# Reduce-scatter: perform reduce on each dimension, scatter results
rank0 = tensor([0., 1, 2, 3])
rank1 = tensor([1., 2, 3, 4])
rank2 = tensor([2., 3, 4, 5])
rank3 = tensor([3., 4, 5, 6])
# Output
rank0 = tensor([6.])    # Sum along dim 0 (0 + 1 + 2 + 3)
rank1 = tensor([10.])   # Sum along dim 1 (1 + 2 + 3 + 4)
rank2 = tensor([14.])   # Sum along dim 2 (2 + 3 + 4 + 5)
rank3 = tensor([18.])   # Sum along dim 3 (3 + 4 + 5 + 6)
# Use case: after backward pass, sum gradients from different data shards, but distribute storage
```

![Reduce-scatter 对四个 4 维向量按维度求和得到 6/10/14/18，并高亮「反向传播后汇总梯度但分散存储」](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0012_00-14-33.jpg)

讲者给出的使用场景是反向传播之后：每个 GPU 处理不同数据，需要把来自不同分片（data shards）的梯度求和，同时还要「redistribute this storage（把存储重新分散开）」——注意「distribute storage」正是 reduce-scatter 与 reduce 的关键差别，reduce 把所有结果堆到 rank 0，而 reduce-scatter 把结果分散存到各 rank，显存占用因此降下来。

最后是 all-reduce，讲者说「if you understand all reduce scatter and all gather, it's basically you do one and then you do the other」，并把它称为「in some sense the easiest to understand」：有一堆张量，做 reduce（这里是 sum），然后「replicate them on all the nodes（复制到所有节点）」。用同样的 4×4 输入，先 reduce-scatter 得到 `[6., 10., 14., 18.]` 分散在四个 rank，再 all-gather 把这四项复制到每个 rank，于是每个 rank 都持有完整的 `[6., 10, 14, 18]`。这个恒等式是整段最重要的一条：

$$\text{all-reduce} = \text{reduce-scatter} + \text{all-gather}$$

```python
# All-reduce = reduce-scatter + all-gather
rank0 = tensor([0., 1, 2, 3])
rank1 = tensor([1., 2, 3, 4])
rank2 = tensor([2., 3, 4, 5])
rank3 = tensor([3., 4, 5, 6])
# Output
rank0 = tensor([6., 10, 14, 18])
rank1 = tensor([6., 10, 14, 18])
rank2 = tensor([6., 10, 14, 18])
rank3 = tensor([6., 10, 14, 18])
# Use case: after backward pass, sum gradients from different data shards, but replicate full parameters
# Breaking all-reduce into reduce-scatter + all-gather allows for flexibility (e.g., ZeRO/FSDP)
```

![All-reduce 的代码示例，输入与 6/10/14/18 输出，并注明「拆成 reduce-scatter + all-gather 可支持 ZeRO/FSDP」](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0013_00-15-49.jpg)

讲者特意让大家「focus your attention on all reduce」，因为数据并行（data parallel）第一个用到的就是它：反向传播后「sum the gradients, and then replicate the full parameters（求和梯度、再让每张卡持有完整参数）」。但他紧接着埋下后半段的伏笔：「to get to fancier things like zero or FSDP, we need to break the all reduce into reduce scatter and all gather. Because then you can intervene and you can manage things a bit more（要做 ZeRO / FSDP 这类更花哨的东西，必须把 all-reduce 拆成两步，才能在中间介入、更精细地管理存储）；but for the basic version, all reduce is fine」。幻灯片把这一点写成「Breaking all-reduce into reduce-scatter + all-gather allows for flexibility (e.g., ZeRO/FSDP)」。

本章的易错点集中在三处。其一，all-gather 的目标是「所有 rank」而不是 rank 0，少一个 all 语义完全不同；其二，reduce-scatter 是「按每个维度分别归约」而非把整个张量压成一个标量，输出是 4 个分别位于不同 rank 的单元素张量 `[6.]`、`[10.]`、`[14.]`、`[18.]`，不是 `[6., 10, 14, 18]` 堆在一处；其三，all-reduce 与 reduce-scatter 的差别在于结果「复制」（replicate）还是「分散」（distribute storage），这正是 FSDP/ZeRO 省显存的突破口。把三者与前后连起来：all-gather 对应前向取完整参数、reduce-scatter 对应反向汇总梯度且省显存、all-reduce 对应最朴素的数据并行，而下一章的 all-to-all 则是这 8 个操作里最通用、却主要用于 MoE 的那一个。

## all-to-all 与 MoE 的动态路由 [16:48 - 19:43]

最后一个集合操作是 all-to-all，讲者称它「in some ways, the most general（在某种意义上最通用）」：你「basically specify how each rank sends a particular message to another rank（基本上要指定每个 rank 给其他每个 rank 各发哪一段消息）」。例子沿用 4 个 rank、每个 rank 持有 4 个数的设定，但语义变了——张量里的「位置」表示这条数据的最终目的地。rank0 的 `[0., 1, 2, 3]` 里，第 0 个位置的值 0 发给自己（rank 0，keep it myself）、1 发给 rank 1、2 发给 rank 2、3 发给 rank 3；rank1 的 `[4., 5, 6, 7]` 同理，4 发给 rank 0、5 发给 rank 1、6 发给 rank 2、7 发给 rank 3；rank2 与 rank3 依次类推。幻灯片用行内注释把这张「发送表」写得非常直白：

```python
# All-to-all: each rank sends each other rank some tensor (most general)
# Input
rank0 = tensor([0., 1, 2, 3])     # send 0 to rank 0, 1 to rank 1, 2 to rank 2, 3 to rank 3
rank1 = tensor([4., 5, 6, 7])     # send 4 to rank 0, 5 to rank 1, 6 to rank 2, 7 to rank 3
rank2 = tensor([8., 9, 10, 11])   # send 8 to rank 0, 9 to rank 1, 10 to rank 2, 11 to rank 3
rank3 = tensor([12., 13, 14, 15]) # send 12 to rank 0, 13 to rank 1, 14 to rank 2, 15 to rank 3
# Output
rank0 = tensor([0, 4, 8, 12])
rank1 = tensor([1, 5, 9, 13])
rank2 = tensor([2, 6, 10, 14])
rank3 = tensor([3, 7, 11, 15])
```

![All-to-all 的输入/输出张量与 MoE 笔记：均衡切分等价于转置、也支持不均衡切分](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0015_00-18-21.jpg)

输出该怎么读？讲者解释：「the first rank is going to receive everyone who sent everything in column zero（rank 0 会收到所有 rank 发往第 0 列的那些值）」，因为所有 rank 都把「位置 0」的元素发给了 rank 0，所以 rank0 收到 0、4、8、12；同理 rank1 收到第 1 列的 1、5、9、13，rank2 收到 2、6、10、14，rank3 收到 3、7、11、15。把输入看成 4×4 矩阵 $X$、输出看成 $Y$，就得到 $Y[r][c] = X[c][r]$，也就是均衡情形下的转置：

$$Y = X^{\top}$$

讲者原话是「if everything were balanced, meaning that every rank sent the same number of bytes to every other rank, then all-to-all, you can think about it as essentially a transpose（如果每个 rank 发给其他每个 rank 的字节数都相同，all-to-all 本质上就是一次矩阵转置）」。但他立刻补上一般性：「in general, all-to-all also handles unbalanced splits（一般也能处理不均衡切分）」，「I'm not showing this here, but you can configure it to send any number of bytes to any other rank（可以配置成给任意 rank 发任意字节数）」，只是「in general you want the splits to be as balanced as possible（总体还是希望切分尽量均衡）」。幻灯片把这三条 Notes 原样列了出来：「Useful for MoEs: each rank has split of data and subset of experts; need to route data to experts」「For balanced splits, all-to-all looks like transpose」「Also handles unbalanced splits (but want splits to be as balanced as possible)」。

为什么 all-to-all 值得单列？因为它是 MoE（Mixture of Experts，混合专家）的通信骨架。讲者的直觉解释是：每个 rank「has both a split of the data and also a subset of experts（既持有一份数据分片，又持有一部分专家）」，而 MoE 的关键在于「it's sort of dynamic routing（某种动态路由）」——「You have to look at your data to figure out which experts you need to route those activations to（必须先看数据，才能决定把这些激活路由到哪些专家）」。由于路由结果依赖数据、事先无法确定，每个 rank 都得把自己那部分激活按目的地分发给所有其他 rank，这「ends up being a all-to-all communication（最终就是一次 all-to-all 通信）」。讲者还把它和之前 「Tatsuzou lecture」里的负载均衡（load balancing）联系起来：那次课想让各专家负载尽量均衡，这里则希望 all-to-all 的收发量尽量均衡，二者「morally the ideal goal（在精神上是同一个目标）」——理想情况下 all-to-all 看上去就该像一次干净的转置。

本章的易错点有三。第一，张量里的「位置」在 all-to-all 中代表「目的地 rank」，而不是数据的来源或维度编号，按前一章 reduce-scatter 的「按维度归约」去理解会完全读反。第二，「等价于转置」只在 balanced splits 下成立；一般实现支持不均衡切分，但不均衡会拖慢最慢的那条链路，所以要靠 load balancing 去逼近均衡。第三，别因为 all-to-all 最通用就以为训练里最常用——讲者明说它主要用于 MoE，本讲「not going to actually spend too much time on this」，真正反复出现的是上一章的三个 workhorse。与前后关系上，这一章补齐了 8 个操作里的最后一块，也把「all」这个前缀从 all-gather、all-reduce 延伸到 all-to-all；紧接着讲者用一段术语梳理给这一大串操作收尾，帮助大家记住谁是谁的逆、谁负责归约。

## 术语记忆法：reduce、scatter、gather 与 all [19:43 - 20:15]

一口气讲完 8 个操作后，讲者说「just to summarize maybe a few helpful tips to remember the terminology because I just went through like quite a few different operations」，用极短的一段给出记忆法。第一条是关于 reduce：「reduce is well, it's reduced. It performs on some sort of associative commutative operation. Could be some, could be max, could be min（reduce 就是归约，它执行某种满足结合律、交换律的运算，可以是 sum、max、min）」。幻灯片底部那行灰字把它写成「Reduce: performs some associative/commutative operation (sum, min, max)」。第二条是 scatter 与 gather 的关系：「scatter is the inverse of gather. Scatter distributes, gather centralizes（scatter 是 gather 的逆；scatter 负责分发，gather 负责集中）」。第三条是前缀 all：「And all just means that the destination is all you know devices（all 只表示目的地是所有设备）」。把三条合起来就是这份速记：

```text
Ways to remember the terminology:
  reduce  -> performs some associative/commutative operation (sum, min, max)
  scatter -> inverse of gather   (scatter distributes, gather centralizes)
  all     -> destination is all devices
```

为什么 reduce 一定要强调「associative commutative（结合律、交换律）」？这不只是数学措辞，而是分布式实现的硬约束：只有满足这两条性质，归约才能被拆成任意顺序的两两合并——无论底层用树形（tree）还是环形（ring）把各 rank 的部分和累加起来，结果都一致。前面算过的 $0+1+2+3=6$ 是 sum 的例子，而 max、min 同样合法；反过来，像「拼接（concatenation）」虽然满足结合律却不是交换律，所以讲者只是打个比方说「gather 像运算为拼接的 reduce」，并没有把它当成真正的 reduce 原语。这一点也解释了为什么前面四个基础操作里 reduce 和 gather 要分开讲：gather 不改数据、只拼接，reduce 会用运算把多份数据压成一份。

把这套命名法套回 8 个操作，能看出它们其实是一条很整齐的谱系，下面这张对照表只作补充，帮助把前面几章的零散定义串起来：

| 操作 | 来源 | 目的地 | 是否带运算 |
|---|---|---|---|
| broadcast | 单个 rank | 所有 rank | 否，复制 |
| scatter | 单个 rank 的大张量 | 各 rank 各一块 | 否，切分 |
| gather | 各 rank 各一块 | 指定 rank | 否，拼接 |
| reduce | 各 rank 各一块 | 指定 rank | 是，sum/min/max |
| all-gather | 各 rank 各一块 | 所有 rank | 否，拼接 |
| reduce-scatter | 各 rank 的向量 | 各 rank 各一块 | 是，按维度归约 |
| all-reduce | 各 rank 的向量 | 所有 rank | 是，归约后复制 |
| all-to-all | 每个 rank | 每个 rank | 否，按目的地分发 |

最后几个易错点值得单独点出。其一，「all」修饰的是「目的地（destination）」而不是「来源」或「运算」：all-reduce 是「归约后送到所有设备」，all-gather 是「收集后送到所有设备」，all-to-all 是「每个 rank 都发给所有 rank」，三者里的 all 含义一致，别误解成「所有数据都参与运算」。其二，scatter 与 gather 互逆，但 reduce 与 gather 并不互逆——reduce 带运算且通常把多份压成一份，gather 只是拼接，二者的逆关系不成立。其三，broadcast 里的「到所有 rank」是复制语义，all-reduce 里的「到所有设备」是归约后复制，机制不同，不要因为都带 all/to-all 的字样就混为一谈。讲者把这段密集的术语梳理停在这里，是因为它恰好是进入代码与硬件之前的最后一块拼图：后面无论 `torch.distributed` 里的 `dist.all_reduce`、数据并行里对梯度的 `ReduceOp.SUM`，还是 FSDP/ZeRO 把 all-reduce 拆成 reduce-scatter 加 all-gather，调用的都仍是这一章反复咀嚼过的这几个词根；而本讲 Part 2 的 data_parallelism()、tensor_parallelism()、pipeline_parallelism() 三种切法，本质上也就是在决定「什么时候用 all-reduce、什么时候用 all-gather 加 reduce-scatter」。

## 集合通信的收尾问答、术语记忆与转向硬件 [20:15 - 22:29]

本段以一段关于集合通信（collective operations）的问答开场。20:17 讲师刚讲完 all-reduce 与 all-gather，随即暂停接受提问。第一个问题针对 gather / reduce 这类「把较小的 rank 结果汇聚到 rank zero」的操作：rank zero 是每次都固定的某张 GPU，还是可以变化？讲师的回答是，不必在很早之前就把目标定死，但必须在真正执行调用（execute the call）的时候确定；后面看代码会看到，你基本上是指定 GPU ID 或 rank（"you basically specify the GPU ID or the rank"），输出就写到那里去。这一点对理解集合通信很要紧：所谓「目标 rank」是本次调用的参数，而不是硬件拓扑里写死的常量。紧接着第二个问题追问这些箭头图是不是只是概念上的积木（conceptual building blocks）、还是真实代码，讲师明确说目前展示的是概念积木（"just conceptual building blocks"），但「我们很快就会看到它们在代码里怎么实现」（we'll very quickly see how these are implemented in code）。这句承诺预告了本段后半段会真正进入 PyTorch 代码，也把这一小节定位成从「集合通信语义」到「可运行实现」的过渡。

画面上的讲义页还给出一套本讲自创的「术语记忆法」（Way to remember the terminology），这部分转写没有逐字念出，是幻灯片独有的信息，值得完整转述。三条分别是：`Reduce: performs some associative/commutative operation (sum, min, max)`——归约就是执行某种满足结合律、交换律的运算，典型如求和、取最小、取最大；`Scatter is inverse of gather`——scatter 是 gather 的逆操作，一个分发、一个收集；`All: means destination is all devices`——前缀 All 表示目的地是全部设备，于是 all-reduce 就是「归约到所有设备」，all-gather 就是「收集到所有设备」。同页还留有上一段关于 MoE 与 all-to-all 的收尾 bullet：`Useful for MoEs: each rank has split of data and subset of experts, need to route data to experts`（每个 rank 持有一部分数据和一部分专家，需要把数据路由到对应专家），`For balanced splits, all-to-all looks like transpose`（切分均衡时 all-to-all 看起来就像矩阵转置），以及 `Also handles unbalanced splits (but want splits to be as balanced as possible)`（也能处理不均衡切分，但希望尽量均衡）。这些文字把上一章 all-to-all 与 MoE 负载均衡的关系收束成几条可背诵的规则，原文可原样保留为：

```text
Way to remember the terminology:
  Reduce: performs some associative/commutative operation (sum, min, max)
  Scatter is inverse of gather
  All: means destination is all devices
For balanced splits, all-to-all looks like transpose
Also handles unbalanced splits (but want splits to be as balanced as possible)
```

紧接着这页讲义出现一行带语法高亮的 Python 定义 `def hardware():`，它标记了整节课从「编程模型」切换到「硬件」。与之配套的是同页下方标注为 `Classic (in the home):` 的架构草图：一台服务器里有 CPU、RAM，下面挂着 GPU，底部是 PCIe 总线，两台机器之间通过 Ethernet 相连，图上可见两个 CPU 塔、RAM 条、每侧两块 GPU，以及 PCIe 与 Ethernet 的标注。讲师在 21:53 明确交代了转场动机：在进代码之前，要先谈一点硬件，特别是「GPU 之间怎么连」（how GPUs are connected），因为上一周已经讲过单张 GPU 里面是什么，这一周要补上 networking（网络互连）。这也构成本段的核心逻辑链条：集合通信定义了「要传什么」，硬件决定了「传得多快」，两者必须合起来看，缺任何一半都无法解释后面为什么某些并行策略只适用于某些互连。

本小节虽然主要是问答，却埋下了后面所有数字与代码的前提。此处要记住的参数化事实是：rank 的取值范围是 $0$ 到 $\text{world\_size}-1$，而 world size 是进程（设备）总数——转写里要到 38:12 才正式定义，但此刻「gather/reduce 的目标 rank 可指定」已经隐含了这个编号体系，用集合写就是 $\text{rank} \in \{0,1,\ldots,W-1\}$，其中 $W=\text{world\_size}$。一个常见易错点是把 rank zero 当成物理上固定的「主卡」：它其实只是本次调用里被选中的那个编号，换一次调用就可以换目标。另一个易错点是把概念积木当成可执行的 API：这一小节里的 broadcast / scatter / gather / reduce 箭头只是语义示意，真正的张量、后端、同步都要到 `torch.distributed` 才落地。还要注意术语记忆里的 `associative/commutative` 不是随便加的修饰——只有满足结合律与交换律的运算，才能被底层库拆成 ring 或 tree 上的多步归约而保持结果正确，这直接关联到后面 all-reduce 的分解与带宽公式。局限上，讲师坦言此时不展开代码，因此这节只建立直觉、不给可运行示例；讲义里 `def hardware():` 这个空壳函数本身也说明，硬件内容要靠后面几章的架构图和数字去填实。

![讲义页上的术语记忆法、MoE 的 all-to-all 收尾说明，以及 def hardware() 与经典家用机架构草图](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0018_00-22-09.jpg)

## 从家用机到数据中心：两张拓扑图与带宽标尺 [22:29 - 24:41]

讲师先展示「经典的、看起来很老的图」（a very classic picture / very old-looking image），描述通用计算机的一般工作方式：一台 server 里有若干 CPU，下面是一条 PCIe bus，早年用来接鼠标、键盘这类外设，如今 GPU 也挂在这条总线上，旁边还有 RAM；这台计算机再通过 Ethernet 连到另一台计算机，如此一台一台延伸出去。转写里带了一句玩笑：如果你买张游戏 GPU、和朋友连起来说「我要训个大模型」（"I'm going to train some big model"），那你大概就得这么做。结论被概括为一句拓扑判断：同一节点（same node）内的 GPU 走 PCIe 通信，不同节点（different node）的 GPU 得一路绕到 Ethernet。这里的 topology（拓扑）是本段关键词——它指设备之间物理连线的结构，而结构直接决定通信代价；把「哪条线连到哪」画清楚，是后面所有带宽数字的前提。

真正严肃训练时的画面被标注为 `Modern (in the data center):`，这是幻灯片上一张结构清晰的架构图，转写只提到「有 NVLink、NVLink switch 和 InfiniBand」，图里的分层细节需要转述。图上并排画了 4 张 GPU，每张 GPU 内部自上而下是：若干 SM（每个 SM 里标着 `Reg` 寄存器与 `L1 cache` 一级缓存/共享内存块）、一整块 `L2` 缓存、写着 `GPU` 的紫色层、以及底部的 `HBM`；4 张 GPU 各自通过一条标为 `NVLink` 的竖线连到下方横贯整幅的粉色 `NVSwitch`，NVSwitch 再向下引出一根线标为 `Infiniband/Ethernet`。也就是说，节点内是「GPU → NVLink → NVSwitch」，跨节点才落到 InfiniBand/Ethernet。这与上一节的 Classic 图形成一一对照：PCIe 被 NVLink 取代，直连被交换芯片取代，而单卡内部的 SM / L2 / HBM 层级被原样保留下来，说明这一周只是把上周那张图的下边界往下延伸了一层。

讲师在 23:45 特意给数字做校准（"just for kind of calibration"）：典型是每节点 8 张 GPU（"eight is typical"），而幻灯片上出现的 `256` 是「编出来的」（"this 256 is kind of made up"），只是示意规模，不是规格。带宽方面，若用 NVLink 5（NVLink five），总带宽约 $1.8\ \text{TB/s}$（转写原话先说成 terabits、随即自我纠正为 terabytes per second，并强调是 total bandwidth）；而 B200 的 HBM 带宽是 $8\ \text{TB/s}$。于是得到一个关键比值：

$$
\frac{\text{NVLink 5.0}}{\text{HBM (B200)}} = \frac{1.8\ \text{TB/s}}{8\ \text{TB/s}} \approx \frac{1}{4}
$$

即节点内跨卡互连大约比片上高带宽内存慢 4 倍（"about four x slower"）。讲师紧接着把这条放进完整存储层级里定位：NVLink 仍然很快（"this is still pretty fast"），但显然不如 HBM，而 HBM 又远慢于 shared memory 或 L1 cache。可以把它写成一条单调递减的带宽链：

$$
\text{L1/shared memory} \;>\; \text{HBM}\ (8\ \text{TB/s}) \;>\; \text{NVLink 5.0}\ (1.8\ \text{TB/s}) \;>\; \text{InfiniBand} \;>\; \text{Ethernet}
$$

参照系一变，判断就变：上周还嫌 HBM 慢，把视角拉到多卡之后，HBM 反而成了快的那一层。这种「同一套层级、不同尺度」的类比，是后面理解三级组网的关键。

从编程视角看，NVLink + NVSwitch 带来一个极重要的抽象：你可以认为任意 GPU 直接连到任意其他 GPU（"you can think about GPUs as connected to any other GPU"），硬件负责把数据送到交换机、再由交换机路由（the switch routes it）。这正是集合通信能写成「all-reduce 到所有设备」而不必关心谁和谁相邻的物理基础，也是第 1 章那些概念箭头得以成立的硬件保证。易错点有两个：一是把 $1.8\ \text{TB/s}$ 当成单条链路速率，图与转写强调它是 total bandwidth（整个域的总带宽），不是每对 GPU 之间的峰值；二是把 NVLink 与 HBM 的 4 倍差距理解成「可以忽略」——4 倍在访存密集的训练里已足以让通信成为瓶颈，本讲后面的带宽基准测试会把它量化。局限上，NVSwitch 域不可能无限扩张，讲师在 26:13 会点明「你不可能让一个 NV switch 处理 100,000 张 GPU」，所以「任意互连」的直觉只在单个 NVLink 域内成立，出了域就必须降级到别的线缆；这也直接引出下一章把 `Infiniband/Ethernet` 那根线展开成三级结构。

![现代数据中心架构图：4 张 GPU 各自的 SM、L2、HBM 经 NVLink 汇入 NVSwitch，再下接 InfiniBand/Ethernet](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0019_00-23-25.jpg)

## 典型集群的三级组网与「越远越慢」 [24:41 - 26:18]

幻灯片上 `Typical setup:` 标题下给出三条 bullet，转写只念了其中一部分，图上文字更精确，逐条转述如下——这是本段最重要的「图上独有」数字。第一条 `8 GPUs per node, connected by NVLink to an NVSwitch (B200s' NVLink 5.0 gets 1.8 TB/s; HBM was 8 TB/s)`，把上一章的 8 卡与 $1.8/8\ \text{TB/s}$ 再确认一次；第二条 `256 nodes per pod, connected by Infiniband (via PCIe -> HCA / Infiniband NIC -> Infiniband cable) (~0.05 TB/s)`，给出 pod 规模与完整数据路径，并标出约 $0.05\ \text{TB/s}$ 的带宽；第三条 `N pods per cluster / datacenter, connected by Ethernet (via PCIe -> CPU)`，说明最外层用 N 个 pod 经 Ethernet 相连、路径要穿过 CPU。原文可原样保留为：

```text
Typical setup:
  8 GPUs per node, connected by NVLink to an NVSwitch (B200s' NVLink 5.0 gets 1.8 TB/s; HBM was 8 TB/s)
  256 nodes per pod, connected by Infiniband (via PCIe -> HCA / Infiniband NIC -> Infiniband cable) (~0.05 TB/s)
  N pods per cluster / datacenter, connected by Ethernet (via PCIe -> CPU)
```

转写把第二、三条 bullet 展开成因果解释。随着集群里 GPU 数量增长，你不可能一直待在 NV switch / NVLink 域里（"you can't have NV switch and NV link"），于是必须把节点装进 pod（put these nodes into pods），pod 之间改用 InfiniBand。此时 GPU 不再直连另一张 GPU，而是先经过 PCIe，再走「特殊的 InfiniBand 线缆」（special InfiniBand cable），速度「低得多、低得多」（much much lower）。若连 InfiniBand 域也用尽、pod 变得巨大，就退到 Ethernet；而 Ethernet 这条路径要经过 PCIe 并且真的穿过 CPU（"goes through your CPU"），讲师强调这是「更慢」（even slower）。把三级带宽并排写就是：

$$
\underbrace{1.8\ \text{TB/s}}_{\text{NVLink，节点内}} \;\gg\; \underbrace{0.05\ \text{TB/s}}_{\text{InfiniBand，pod 内}} \;>\; \underbrace{\text{Ethernet，经 PCIe}\to\text{CPU}}_{\text{集群 / 数据中心间}}
$$

注意 InfiniBand 相对 NVLink 已经掉了约 36 倍（$1.8/0.05$），这个落差不是小数，它直接决定了后面数据并行、张量并行、流水线并行各自适合放在哪一层互连上。

讲师在 26:02 给出一句总结：这「有点像内存那套情形」（analogous to the kind of memory situation）——节点越多就越慢，和你不可能让一个 NV switch 处理 100,000 张 GPU 是同一个道理。这种同构性贯穿全讲：单卡内部有 L1 → L2 → HBM 的层级，多卡则有 NVLink → InfiniBand → Ethernet 的层级，二者都是「规模越大、带宽越低、离算力越远」。从前后关系看，这一章把第 2 章那张 `Modern (in the data center)` 架构图里的 `Infiniband/Ethernet` 一根线，展开成带具体数字与路径的三级结构；同时它也是第 4 章「绕过 CPU」的直接前提——正因为最外层 Ethernet 必须经 `PCIe → CPU`，才会引出 RDMA 为什么要把 CPU 从数据路径上摘掉。换句话说，本章给的是「带宽随距离衰减」，下一章补的是「除了带宽还有一次 CPU 中转的延迟」。

几个易错点值得单列。第一，`256 nodes per pod` 里的 256 是讲师明说的 "made up" 示意值，不要当成 NVIDIA 的硬性规格，真正稳定的是「每节点 8 卡」这个 typical 值；第二，`~0.05 TB/s` 是 pod 级 InfiniBand 的量级估计，不是单条链路的精确标称；第三，最容易混淆的是 `HCA / Infiniband NIC` 与 `CPU` 的角色——InfiniBand 路径里数据经 PCIe 到网卡 / HCA 后直接上线缆，而 Ethernet 路径里数据要「经由 CPU」（via PCIe -> CPU），两者都走 PCIe，但后者多了一次 CPU 中转，这正是下一章的痛点。局限方面，这套三级划分是概念模型，真实数据中心还有 spine/leaf 交换、多轨网卡等细节，课程不展开；讲师也提醒这些数字只用于 calibration（校准量级），不是让你据此精确算训练时间，精确值要靠后面的 benchmarking 实测。

![典型组网三条配置：节点内 8 卡 NVLink 接 NVSwitch、pod 内 256 节点走 InfiniBand、集群间 Ethernet 经 CPU](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0021_00-25-57.jpg)

## 绕过 CPU：RDMA、NVL72 与 RoCE [26:18 - 29:58]

幻灯片小节标题是 `Bypassing the CPU:`。传统 Ethernet 的问题被写成一条完整的数据拷贝链：`Ethernet requires passing through the CPU (copying data to kernel socket buffer, build TCP packets, copy to NIC ring buffer)`。转写把它讲得更细：GPU 得先和 CPU 打交道把数据复制出来，CPU 有一个 kernel socket buffer，然后构建一些 network packets（TCP 包），再复制到网络接口（NIC ring buffer），最后才发出去，「这通常引入大量延迟」（introduces a lot of latency）。讲师特别澄清一个术语歧义：这里的 "kernel" 指的是 CPU 操作系统传统意义上的内核，**不是** GPU kernel（"not the GPU kernel, but the CPU traditional notion of a kernel"）——同一个词在前后文里指两样东西，是这节课最容易听岔的地方。用通路公式写：

$$
\text{Ethernet: GPU} \xrightarrow{\text{PCIe}} \text{CPU: kernel socket buffer} \to \text{TCP packets} \to \text{NIC ring buffer} \to \text{wire}
$$

$$\text{RDMA: GPU}_A \;\xrightarrow{\text{direct read/write}}\; \text{HBM of GPU}_B \quad (\text{no CPU})$$

幻灯片上被黄色高亮的那条 bullet 是本节定义：`Remote Direct Memory Access (RDMA): allows one GPU to directly read/write another GPU's memory without involving the CPU`——RDMA 允许一张 GPU 直接读 / 写另一张 GPU 的显存，完全不经过 CPU。转写补充了支持范围：如果你在 NVLink / NV Switch 域里，天然就有 RDMA；InfiniBand 也支持 RDMA（"InfiniBand also supports RDMA"），所以走 InfiniBand 的两张 GPU 可以直接互连、不牵涉 CPU；但幻灯片原文 `InfiniBand supports RDMA, but standard Ethernet does not`——标准 Ethernet 不支持。这解释了一个看似矛盾的现象：第 3 章说 InfiniBand 比 NVLink 慢约 36 倍，但至少它还能保持「GPU 直连 GPU」的语义，而标准 Ethernet 连这个语义都丢了，要额外付一次 CPU 中转的延迟；慢和「要不要经过 CPU」是两笔不同的账。

`Advancements:` 小节有两条 bullet，都是转写里念到、但幻灯片文字更精确的内容。第一条 `GB200/GB300 NVL72: 8 GPUs per tray, 9 trays per rack -> 72 GPUs in one NVLink domain`：NVIDIA 面向 B200 / B300 推出的 NVL72，一个 tray 放 8 张 GPU，一个 rack 叠 9 个 tray，于是 72 张 GPU 全部处于同一个 NVLink 域。写成

$$
8\ \frac{\text{GPUs}}{\text{tray}} \times 9\ \frac{\text{trays}}{\text{rack}} = 72\ \text{GPUs in one NVLink domain}
$$

讲师把这称为「如果你很有钱」（if you have a lot of money）就能买到的高级硬件，把快速互连的边界从普通人的 8 卡推到 72 卡，因为 NVLink 的速度「非常快」。第二条黄色高亮 bullet：`RDMA over Converged Ethernet (RoCE): Ethernet bypasses CPU, similar but cheaper/weaker than InfiniBand, used by Meta`——融合以太网上的 RDMA 让 Ethernet 也能绕过 CPU，与 InfiniBand 相似但更便宜 / 更弱，Meta 在用。转写里把它叫 "RoCE, RDMA over a converged Ethernet"，并说这是「对 InfiniBand 的回应」（their answer to InfiniBand），因为 InfiniBand 一般非常贵（"very expensive, as is a lot of NVIDIA products"）；还提到 Meta 有论文在探索，Llama "may or may not" 是用融合以太网训的——讲师用了明显不确定的措辞，笔记不应把这句坐实成事实。

这一章在结构上是对第 3 章「越远越慢」的修正与补充：慢的不只是线缆带宽，还有「是否经过 CPU」这次语义断裂；RDMA 与 RoCE 都是在尽量不换物理线缆的前提下，把 CPU 从数据路径上摘掉。与后面内容的关系也很直接——第 5 章的 NCCL 之所以能「启动 GPU kernel 去收发数据」，前提正是底层有 RDMA 这类不经过 CPU 的能力。易错点有三：别把 RDMA 当成某根线缆，它是操作层面的能力（第 6 章问答会专门澄清）；别把 `standard Ethernet does not` 记成「所有以太网都不支持」，RoCE 恰恰是以太网侧的补足；别把 NVL72 的 72 当成单机 GPU 数，它是 NVLink 域规模，与 "8 GPUs per node" 不是一回事。局限上，讲师明确划定 RDMA 在 NVLink/NVSwitch 与 InfiniBand 上可用、标准 Ethernet 不可用，而 RoCE 只是 "cheaper/weaker" 的折中；并且 29:42 之后他自己收束道这只是硬件的 "brief overview"：GPU 经 NVLink 接到 NV switch（域可能是 8、也可能是 72），再往外是 InfiniBand——接下来该谈「怎么给这堆硬件编程」了。

![讲义上的 Bypassing the CPU 与 RDMA 定义：GPU 可不经 CPU 直接读写另一张 GPU 显存](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0022_00-27-13.jpg)

![讲义 Advancements 两条：GB200/GB300 NVL72 的 72 卡单 NVLink 域，与 Meta 使用的 RoCE](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0024_00-29-45.jpg)

## 从硬件到编程：NCCL 与课程大纲 [29:58 - 31:08]

29:58 讲师用 "how do you program this?" 收束硬件部分。最底层是 `NVIDIA Collective Communication Library (NCCL)`，转写里被念成 "NCCL or NCCL"、后面又戏称为 "nickel"。幻灯片标题就是 "NVIDIA Collective Communication Library (NCCL)"，下面一句定义：`NCCL translates collective operations into low-level packets that are sent between GPUs. [talk]`——把 all-reduce、reduce、broadcast 这些集合操作翻译成 GPU 之间实际发送的低层数据包。三条 bullet 分别是：`Detects topology of hardware (e.g., number of nodes, switches, NVLink/PCIe)`（探测硬件拓扑，例如节点数、交换机、NVLink/PCIe）、`Optimizes the path between GPUs`（优化 GPU 之间的路径）、`Launches GPU kernels to send/receive data`（启动 GPU kernel 来收发数据）。转写把它讲成一句话：用 NCCL 时你基本上只要说「我要做一次 all-reduce」（"I want to all reduce"），NCCL 自己去搞清硬件拓扑是什么、找出不同 GPU 之间的路径，然后启动 kernel 完成发送与接收。这正好把第 3 章那三级拓扑变成库内部要自动处理的信息。

讲师补了一句贯穿全课的统一视角：归根结底，凡是跑在 GPU 上的东西都是 kernel（"everything that runs on the GPU is a kernel"），因此也存在「通信 kernel」（communication kernels）专门负责与其它 GPU 通信。这把第 1 章的「概念积木」落到了可执行单元上：

$$
\text{collective op} \xrightarrow{\text{NCCL}} \underbrace{\text{detect topology} \to \text{optimize path} \to \text{launch comm. kernels}}_{\text{全部由 NCCL 代劳}}
$$

换句话说，第 1 章画的那些箭头，物理上就是 NCCL 依据拓扑挑好路径后发射的一组 CUDA kernel。讲师随后说「我们不会太深入 nickel，只要知道它存在」（we're not going to look too too much more into nickel, but just know that it exists），这正是从零构建课程的取舍：知道接口之下的存在，但不替 NVIDIA 讲内部实现。

31:06 那页幻灯片给出了整讲的目录结构，转写完全没有逐字念，属于必须转述的图上信息。开头是两段关于运行方式的注释：`# When you execute this lecture directly (python lecture_07.py), it uses multiprocessing, which produces [stdout for each] process (below).` 与 `# However, when you trace this lecture (python -m edtrace.execute -m lecture_07), we turn off multiprocessing [and use] stdout for this lecture`——直接跑 `python lecture_07.py` 会启用多进程、每个进程各打一份 stdout；而用 `python -m edtrace.execute -m lecture_07` 追踪讲解时会关掉多进程，只保留本讲用的 stdout。随后是 `Part 1: building blocks of distributed communication/computation`，以四个函数列出：

```python
collective_operations()   # Programming model
hardware()                # Hardware: how GPUs are connected
torch_distributed()       # How this is implemented in NCCL/PyTorch
benchmarking()            # Measure actual NCCL bandwidth
```

再往下是 `Part 2: distributed training`，说明 `Walk through bare-bones implementations of each strategy on deep MLPs.`，并给出理由 `Recall that MLPs are the compute bottleneck in Transformers, so this is representative.`（MLP 是 Transformer 的计算瓶颈，所以在深层 MLP 上演示具有代表性），列出的第一个策略是 `data_parallelism()  # Cut up along the batch dimension`（沿 batch 维度切分）。

这张大纲把本段精确定位在 Part 1 的 `hardware()` 与 `torch_distributed()` 之间：第 2–4 章讲的是 "how GPUs are connected"，本章的 NCCL 是 "how this is implemented" 的最底层，下一章的问答与第 7 章的 `torch.distributed` 才是上层接口。数字与参数方面，本章新数字不多，但 `world_size=4` 这个后面代码要用的默认值，以及大纲里 `python lecture_07.py`、`python -m edtrace.execute -m lecture_07` 这两条真实命令，都是可核对的具体参数。易错点：一是把 NCCL 当成 PyTorch 的一部分，其实它是更底层的 NVIDIA 库，PyTorch 只是调用它；二是把 "kernel" 一律理解成计算 kernel，这里明确有通信 kernel；三是把多进程 stdout 与单进程追踪输出搞混——讲师特意写了关闭分布式的分支来讲课，若直接按多进程跑，print 顺序会乱，这正是第 7 章引入 `dist.barrier()` 的动机。局限上，讲师坦言不深入 NCCL 内部（不展开它如何选 ring 还是 tree），也明确本课 "from scratch" 不用 FSDP 这类高层封装——这两处都划定了本段的知识边界。

![课程大纲：Part 1 的 collective_operations / hardware / torch_distributed / benchmarking 与 Part 2 分布式训练说明](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0025_00-31-06.jpg)

## 硬件问答：机架、RDMA 层次、非整齐配置与 TPU [31:08 - 36:20]

31:03 讲师说「在去 PyTorch 之前，关于硬件还有没有问题」，于是进入一段约五分钟的问答，画面是讲师在讲台前比划解释。第一个问题要求物理上描述 rack 与 tray。讲师先自谦「我不是硬件专家」（"I'm not a hardware expert"），然后给出 NVL72 的构成：rack 就是数据中心里那种机柜，每个 tray 是一个可插拔单元；`G` 代表 Grace（"the G stands for a grace"），每个 tray 上有 **2 颗 CPU**，每颗 CPU 连接 **4 张 GPU**，所以每个 tray 有 **8 张 GPU**，tray 堆叠起来全部接到 NV switch。写成式子：

$$
2\ \frac{\text{CPUs}}{\text{tray}} \times 4\ \frac{\text{GPUs}}{\text{CPU}} = 8\ \frac{\text{GPUs}}{\text{tray}}, \qquad 9\ \text{trays} \to 72\ \text{GPUs}
$$

这正好把第 4 章 `8 GPUs per tray, 9 trays per rack -> 72 GPUs in one NVLink domain` 那条 bullet 拆成了可理解的物理来源：72 不是凭空来的，而是 2×4×9 乘出来的。

32:15 的问题「RDMA 和这些有什么区别」，讲师给了一个很有价值的层次划分：RDMA 更像是一个「数字层面的路由器」（a more of a digital router），意思是一张 GPU 能读写另一张 GPU 的显存；实现 RDMA 有多种方式——可以用 NVLink 与 NV Switch，也可以用 InfiniBand。而 InfiniBand、NV Switch、NVLink 更偏「硬件」：是「有哪些线缆、有哪些交换机」（what cables and switches are there）；RDMA 则更偏「操作层面」（operationally）：通信发生时到底做了什么。33:15 又补一句：RDMA over converged Ethernet 是「另一种做 RDMA 的方式」（another way to do RDMA）。用一句话概括这个区分：

$$
\underbrace{\text{NVLink / NVSwitch / InfiniBand / Ethernet}}_{\text{硬件：线缆与交换机}} \quad\text{vs}\quad \underbrace{\text{RDMA}}_{\text{操作能力：远端直接读写显存}}
$$

这纠正了第 4 章容易产生的误解：RDMA 不是与 InfiniBand 并列的一根线，而是横跨多种硬件之上的能力；同一套 RDMA 语义可以由不同物理介质承载。

33:50 问 NCCL 是否针对多节点集群优化。讲师没有给内部细节（"I don't know the details of how they have or have not optimized it"），只给了一个基于动机的判断：NVIDIA 一直在为大模型推理与训练优化整个栈（"optimizing the entire stack"），因为他们的主要客户就是那些大型语言模型提供商，所以「如果他们没考虑过这类工作负载的优化，我会很惊讶」。这种「不确定就明说」的态度值得保留，不应把推测写成官方结论。34:44 的问题更实用：有 9 张 GPU 时怎么分配负载？讲师回答取决于这第 9 张落在拓扑的哪里：常见情况是每节点 8 卡，于是第 9 张必然在另一个节点上；如果它们之间没有 NVLink，那会「非常糟」（really bad）——因为多出来的那个节点既提供不了多少算力，通信又极其昂贵；但如果全部由 NVLink switch 连起来，就「合理得多」（much more reasonable）。可以用

$$
9 = \underbrace{8}_{\text{node A}} + \underbrace{1}_{\text{node B，无 NVLink}}
$$

来记住这个不整齐（ragged）配置的尴尬：它把第 3 章「越远越慢」的结论变成了一个具体的调度反例，也说明 world size 不是想设成多少就设成多少，得看物理拓扑能否整除、能否处在同一个快速域里。

35:37 最后一个问题是「这和 TPU 有什么不同」。讲师的回答很克制：TPU 总体上「更简单一些」（much simpler objects），但「我不太熟悉每个组件对应什么」（I'm not too familiar with the details of how each of these components corresponds to），提议线下再聊（"maybe we can talk about it offline"）。按「转写含糊处宁可略过，不要猜」的要求，这里不补任何 TPU 内部细节。整章与前后的关系是：它不引入新机制，而是把第 2–5 章的 NVLink / NVSwitch / InfiniBand / RDMA / NCCL 这些名词在问答里重新校准了一遍边界，随后 36:13 讲师用 "let's actually get down to some code to take advantage of this hardware" 正式切入代码。易错点：把 RDMA 与 InfiniBand 当成同层概念、把 Grace 当成 GPU 型号（它其实是 CPU 代号）、把 9 卡问题当成「平均分配就行」（真正的约束是拓扑与通信代价）、以及把讲师关于 NCCL 优化的措辞当成 NVIDIA 官方结论。局限上，本章多处答案都带 "I'm not a hardware expert"、"I don't know the details"、"maybe offline"，说明这些是课堂即兴回答而非可依赖的规格说明，只能作为直觉补充。

## 用 torch.distributed 写集合通信：spawn、setup、barrier 与 all-reduce [36:20 - 40:31]

36:20 讲师正式进入代码。幻灯片 `def torch_distributed():` 下写：`PyTorch distributed library (torch.distributed) [documentation]`，两条 bullet 是 `Provides clean interface for collective operations (e.g., all_gather_into_tensor)` 与黄色高亮的 `Supports multiple backends for different hardware: gloo (CPU), nccl (GPU)`，还有一条淡色 bullet `Also supports higher-level algorithms (e.g., FullyShardedDataParallel) [not used in this course]`。转写对应：PyTorch conveniently 提供了 `torch.distributed`，让你不必显式去想 NCCL（"you don't have to explicitly think about nickel"）；库支持不同硬件的不同后端——GPU 用 `nccl` 后端，CPU 用 `gloo`（转写口误念成 "glue"），后者说明并行处理在 GPU 之前很久就存在、CPU 上也能做集合操作；库还提供 FSDP 等更高层的模型与算法，但本课「从零构建」（because we're building things from scratch）不用它们。这一层正好坐在第 5 章 NCCL 之上：你写的是 `torch.distributed` 的 API，底层由 `nccl` 后端翻译成数据包与通信 kernel。

37:12 开始走例子。讲义写着 `Let's walk through some examples.` 与 `spawn(collective_operations_main, world_size=4)`。讲师说这个 `spawn` 接收一个函数，并「复制运行四次」（run this replicated four times），其中 4 就是 world size。他特别说明这是他自己写的一个 wrapper，用来绕开课堂里不能做多进程的限制：正常你会调 `torch.multiprocessing.spawn` 再调函数，而讲课追踪时走一个关闭分布式（disables distributed）的分支，直接在单进程里跑，这正对应第 5 章大纲里 `python -m edtrace.execute -m lecture_07` 与 `python lecture_07.py` 的差别。随后是函数签名：

```python
def collective_operations_main(rank: int, world_size: int):
    """This function is running asynchronously for each process (rank = 0, ..., world_size - 1)."""
    setup(rank, world_size)
```

转写把术语定义补全：world size 是进程数（"world size is the number of processes"），rank 取 $0,1,2,\ldots,\text{world\_size}-1$（"the rank is either zero or one or two all the way up to world size minus one"），并且有 world size 个这样的函数各自在一个进程上同时、异步地（at the same time / asynchronously）运行。用集合写就是 $\text{rank} \in \{0,1,\ldots,W-1\}$，其中 $W=\text{world\_size}=4$。这也把第 1 章「gather/reduce 的目标 rank 可指定」里隐含的编号体系正式定了下来。

38:20 起讲 `setup(rank, world_size)`：里面配置 master address 与 port，但讲师立刻强调「注意这并不是 GPU 真正通信的方式」（"this is not actually how the GPUs are going to communicate"）——它只用于通用元数据与进程协调（general metadata and coordination），真正的数据走 NCCL，「否则会非常非常慢」（otherwise, it'll be very very slow）；有 CUDA 就用 `nccl` 后端，讲师在笔记本上则用 `gloo`。接着是 `dist.barrier()`：`### All-reduce (dist = torch.distributed)` 段落下第一行就是它，注释写着 `# Waits for all processes to get to this point (in this case, for print statements)`。讲师解释：所有进程异步运行，你控制不了谁先谁后，某个进程可能完全跑完、print 也可能交错（interleaving）；若想保证某段代码先于另一段执行，就插入同步屏障，让所有进程都到达这一点才继续；代价是「可能不必要地等待」（waiting potentially unnecessarily）。最后是 all-reduce 例子，讲义上的代码原文（kf_0031）可整理为：

```python
dist.barrier()  # Waits for all processes to get to this point (in this case, for print statements)
data = tensor([0., 1, 2, 3], device=cuda_if_available(rank)) + rank  # Both input and output
print(f"Rank {rank} [before all-reduce]: {data}", flush=True)
dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)  # Modifies tensor in place
print(f"Rank {rank} [after all-reduce]: {data}", flush=True)
```

讲师在 40:01 说「我们来试一次 all-reduce」，创建张量 `0 1 2 3`，并「为了更有趣」（to make it more interesting）让每个 rank 持有不同的张量（即 `+ rank`），然后打印 all-reduce 之前的值；转写到 40:29 停在 "rank zero before all reduce has 0 1 2 …"，也就是 rank 0 的初始张量 $[0,1,2,3]$ 被读出来的那一刻，本段到此结束。

把这段代码的参数收拢：`world_size=4`、张量 `[0., 1, 2, 3]`（第一个元素写成 `0.` 表明是浮点张量）、设备由 `cuda_if_available(rank)` 决定、归约操作 `dist.ReduceOp.SUM`、`async_op=False`（同步，且注释 `# Modifies tensor in place` 说明原地写回），`data` 既是输入也是输出（`# Both input and output`）。由 `data = [0,1,2,3] + rank` 与 `SUM` 可直接算出四个 rank 在每个分量上的和：分量 0 为 $0+1+2+3=6$，分量 1 为 $1+2+3+4=10$，依次得 $[6,10,14,18]$——这与 all-reduce「每个 rank 都拿到完整归约结果」的定义一致，但本段画面与转写并未打印出 after 的值，故这里只是按代码做的数学推演，不应当成屏幕上出现的输出。与前后内容的关系：本章把第 1 章「概念积木很快会在代码里实现」的承诺兑现了，`spawn` / `setup` / `barrier` / `all_reduce` 正好对应 NCCL 三步（探测拓扑、选路径、发射 kernel）之上的 PyTorch 接口层；而 `barrier` 又直接回应了第 5 章大纲里「单进程追踪 vs 多进程 stdout 顺序」的问题。易错点：把 master address/port 当成数据通道（其实只是元数据协调）、把 `gloo` 当成 GPU 后端（它是 CPU 后端，"glue" 是口误）、忘记 `all_reduce` 是原地修改而以为要接收返回值、把 `async_op=False` 与 barrier 的作用混为一谈（前者管这一次通信是否异步返回，后者管多进程是否对齐）、以及误以为 print 顺序可控。局限：关闭分布式的单进程分支只是讲课用的 hack，真实训练必须走 `torch.multiprocessing.spawn`；`async_op=True` 的通信 / 计算重叠、`wait()` 等本段还没展开，留到后续基准测试与作业。

![NCCL 三条职责与 torch.distributed 讲义：gloo(CPU)/nccl(GPU) 后端、FSDP 高层算法不使用](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0029_00-36-54.jpg)

![collective_operations_main 代码：docstring、setup、dist.barrier 与 all-reduce 前后的 print](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0031_00-39-47.jpg)

## all_reduce 与 reduce-scatter 的代码实操：原地写回、独立输出与打印顺序 [40:31 - 43:45]

本段承接上一部分已经建立好的集合通信示例程序。屏幕上幻灯片标题是 `Let's walk through some examples.`，`spawn(collective_operations_main, world_size=4)` 启动 4 个进程，覆盖的小卡片显示当前讲解停在 `rank = 0`、`world_size = 4`。函数文档字符串写明 `"""This function is running asynchronously for each process (rank = 0, ..., world_size - 1)."""`，也就是这四份代码彼此异步、各跑各的。讲课人先提醒：rank 1 的初始张量是 `[0., 1, 2, 3] + rank`，即 `[1, 2, 3, 4]`，而 `print` 的输出顺序「由硬件随意决定」（whatever order the hardware feels like），因为进程在异步运行，但数据一条不少。这正是后面所有 `dist.barrier()` 的直觉起点：分布式程序没有天然的全局先后，只有显式插入的同步点。屏幕上关于 `torch.distributed` 的 bullet 还点明两层抽象——`Supports multiple backends for different hardware: gloo (CPU), nccl (GPU)`，以及 `Also supports higher-level algorithms (e.g., FullyShardedDataParallel) [not used in this course]`；本课刻意不用 FSDP 这类高层算法，坚持从零写集合操作。

第一个原语 `### All-reduce (dist = torch.distributed)` 的代码原文如下（第 249–260 行附近）：

```python
def collective_operations_main(rank: int, world_size: int):
    setup(rank, world_size)
    ### All-reduce (dist = torch.distributed)
    dist.barrier()  # Waits for all processes to get to this point (in this case, for print statements)
    data = tensor([0., 1, 2, 3], device=cuda_if_available(rank)) + rank  # Both input and output
    print(f"Rank {rank} [before all-reduce]: {data}", flush=True)
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)  # Modifies tensor in place
    print(f"Rank {rank} [after all-reduce]: {data}", flush=True)
```

`dist.all_reduce` 是 PyTorch 的函数：传进去一个张量、一个归约操作（这里是 `dist.ReduceOp.SUM`），并且特意设 `async_op=False`（讲课人说 "don't do it async"）。底层会调用 gloo 或 nccl，由后者启动 CUDA kernel、完成通信、把结果**原地写回** `data`——屏幕黄色高亮正是这一行，注释 `# Modifies tensor in place`。all-reduce（全归约）的语义是「每一列求和，再把结果复制到所有 rank」：四个 rank 的输入分别为 `[0,1,2,3]`、`[1,2,3,4]`、`[2,3,4,5]`、`[3,4,5,6]`，按列求和得到 $[6, 10, 14, 18]$，结束后每个 rank 都持有同一份结果。现场有个问答：学生问 rank 是不是 GPU，讲课人确认「在这门课里 rank 就是 GPU」——rank 是进程编号，本课每个进程绑定一张 GPU，二者一一对应；这个约定在后面看日志里的 `device='cuda:0'` 到 `device='cuda:3'` 时会反复用到。

接着切到 `### Reduce-scatter`，先 `dist.barrier()` 再构造两个张量（第 262–270 行）：

```python
### Reduce-scatter
dist.barrier()
input = torch.arange(world_size, dtype=torch.float32, device=cuda_if_available(rank)) + rank  # Input
output = torch.empty(1, device=cuda_if_available(rank))  # Allocate output
print(f"Rank {rank} [before reduce-scatter]: input = {input}, output = {output}", flush=True)
dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
print(f"Rank {rank} [after reduce-scatter]: input = {input}, output = {output}", flush=True)
```

与 all-reduce 的关键差别是**输入与输出分离**：`input` 不被改动，`output` 单独用 `torch.empty(1)` 分配，它碰巧初值为零，但讲课人强调「可以是任意值，别在意」。`torch.arange(world_size, dtype=torch.float32)` 造出 `0..world_size-1` 再加 rank，于是四个 rank 的输入仍是上面那四行；按列求和 $[6, 10, 14, 18]$ 后，四个分量分别写进 rank 0/1/2/3 的 `output`。屏幕上的运行日志印证了这一点：rank 2 打印 `input = tensor([2., 3., 4., 5.], device='cuda:2'), output = tensor([14.], device='cuda:2')`，`input` 一字未改。讲课人把 `reduce_scatter_tensor` 概括成「每个分量的归约结果写进各自的 rank」，这正是 reduce-scatter（归约散射）名字的由来。

这一章把上一段停留在概念图层面的 all-reduce / reduce-scatter 落成了可运行的 `torch.distributed` 代码，也为 46:30 之后的基准测试、以及 55:07 之后的数据并行提供了同一套 API（`dist.all_reduce`、`dist.ReduceOp.SUM`）。易错点有三处：其一，`async_op=True` 虽然更「花哨」，却会打乱这些 `print` 的先后，所以讲课人说自己「比平常多放了几个 barrier」；其二，all-reduce 原地覆盖输入，而 `reduce_scatter_tensor` 用独立 `output`，把两者混用会写错张量；其三，`torch.empty` 不初始化，`before` 打印里的旧值不是计算结果，只能以操作后的打印为准。此外，本例把 `world_size` 固定为 4、张量长度也恰为 4，分片数与张量维度相等只是为了让例子对齐，并不代表一般情形。

![all_reduce 示例代码：dist.all_reduce 原地写回，覆盖层显示 rank=0、world_size=4](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0032_00-41-09.jpg)

## 异步通信与 all_gather：用代码证明 all-reduce = reduce-scatter + all-gather [43:45 - 46:30]

43:50 有学生问「all-reduce 异步到底怎么工作」。讲课人的解释是：这是一次「整体式（monolithic）」操作——你说一句 go do the all reduce，底层就启动 kernel、开始通信；CUDA 本来相对于进程就是异步的，现在又叠加了「多个进程彼此异步」这一层，所以这行 Python 会立刻返回。`async_op=True` 的价值不在于让通信本身变快，而在于**重叠计算与通信（overlapping computation and communication）**：调用返回后，可以去加载下一步所需的、与本次通信无关的数据（"load some other data for the next step"），等真正要用结果时再 `wait()` 或放一个 `barrier()`。讲课人明说这块「这门课不展开」，但它是后面流水线并行、以及把 all-reduce 与反向传播重叠的伏笔。上一章之所以坚持 `async_op=False` 并插入额外 `dist.barrier()`，正是为了让四份 `print` 能按可预期的顺序展示数据，而不是为了性能。

45:03 进入最后一个原语 `### All-gather`，屏幕第 273–282 行的代码原文如下：

```python
### All-gather
dist.barrier()
input = output  # Input is the output of reduce-scatter
output = torch.empty(world_size, device=cuda_if_available(rank))  # Allocate output
print(f"Rank {rank} [before all-gather]: input = {input}, output = {output}", flush=True)
dist.all_gather_into_tensor(output_tensor=output, input_tensor=input, async_op=False)
print(f"Rank {rank} [after all-gather]: input = {input}, output = {output}", flush=True)
```

关键在于 `input = output` 这一行：它把上一步 reduce-scatter 的单标量输出**直接当作 all-gather 的输入**，于是 rank 0/1/2/3 分别持有 `[6.]`、`[10.]`、`[14.]`、`[18.]`。`torch.empty(world_size)` 分配长度为 4 的输出（同样「碰巧有值，别担心」）。`all_gather_into_tensor`（全收集）把各 rank 的这一块收集起来、拼成完整张量、再复制到所有 rank。日志里的对照很清楚：`before all-gather` 时 rank 0 的 `output` 还是 `tensor([ 0., 10., 14., 18.], device='cuda:0')`、rank 1 是 `tensor([1., 3., 4., 5.], device='cuda:1')`，各不相同的旧值；`after all-gather` 之后四个 rank 全部变成 `tensor([ 6., 10., 14., 18.], device='cuda:0')` 到 `device='cuda:3'`。

屏幕第 282 行用黄色高亮写下结论：`Indeed, all-reduce = reduce-scatter + all-gather!`，讲课人称这是 "proof via example"（用例子来证明）。用具体数字核对：reduce-scatter 阶段按列求和得到向量 $[6, 10, 14, 18]$ 并把它切成四片分存到四个 rank，all-gather 阶段再把这四个标量拼回 $[6, 10, 14, 18]$ 并复制到每个 rank，与本章前面 all-reduce 的输出逐元素一致。写成式子，令 $W=4$、各 rank 输入为 $\mathbf{x}_0,\dots,\mathbf{x}_{W-1}$：

$$
\text{all-reduce}(\mathbf{x}_0,\dots,\mathbf{x}_{W-1}) = \text{all-gather}\big(\text{reduce-scatter}(\mathbf{x}_0,\dots,\mathbf{x}_{W-1})\big)
$$

这个恒等式不只是数学趣闻：屏幕下方 `Notes` 与后续内容都预告了 `Next time: FSDP/ZeRO: use all-gather and reduce-scatter to avoid holding all parameters in memory`，也就是把一次 all-reduce 拆成两步，才能在中间介入、只保留一片参数或优化器状态，这正是下一节课 FSDP/ZeRO 的动机。

讲完三个原语后，代码以 `cleanup()` 收尾（讲课人说「有始有终，清理是好习惯」），并宣告「这就是你们的第一个 torch distributed 程序」。同一屏下方紧接着出现 `def benchmarking():`、`How fast does communication happen?` 和 `spawn(all_reduce, world_size=4, num_elements=100 * 1024**2)`，把话题引向下一章的基准测试。局限也要点明：这里的恒等式是在 `world_size=4`、张量长度恰为 4、且 `op=SUM` 的同步（`async_op=False`）条件下验证的；`all_gather_into_tensor` 要求各 rank 输入形状一致、`output_tensor` 预先按 `world_size` 分配好，若实际分片不均就要另做处理（那是 all-to-all 一类操作要解决的问题）。另外，异步路径在本章只是口头描述、没有在示例里真正开启，因此「立即返回」「wait/barrier」这些结论都还是概念层面的说明。

![all_gather 代码与黄色高亮结论「all-reduce = reduce-scatter + all-gather」](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0036_00-46-13.jpg)

## 通信基准测试：warmup、双重同步与计时协议 [46:30 - 49:10]

第一部分（集合操作、硬件、NCCL/PyTorch 实现）到此收束，讲课人用「很快，因为我想赶紧进入 part two」引出基准测试，问题是 `How fast does communication happen?`。屏幕上方还留着 `[How to reason about collective operations]` 与 `[Sample benchmarking code]` 两行小标题。做法与此前计算 MFU（Model FLOPs Utilization，模型算力利用率）时一致：先 warm up（预热），再计时。这里用 1 亿个元素做一次 all-reduce，`spawn(all_reduce, world_size=4, num_elements=100 * 1024**2)`，运行日志里把它打印为 `num_elements=104857600`。之所以要预热，是因为首次调用会触发 NCCL 建连、kernel 加载等一次性开销；不预热就会把这些准备时间算进「通信时间」，测出来的带宽偏低。

`def all_reduce(rank: int, world_size: int, num_elements: int):` 的函数体（第 301–315 行附近）原文如下：

```python
def all_reduce(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)
    # Create tensor
    data = torch.randn(num_elements, device=cuda_if_available(rank))
    # Warmup
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # Wait for CUDA kernels to finish
    dist.barrier()            # Wait for all the processes to get here
    # Perform all-reduce
    start_time = time.time()
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # Wait for CUDA kernels to finish
    dist.barrier()
    end_time = time.time()
    duration = end_time - start_time
    print(f"[all_reduce] Rank {rank}: all_reduce(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)
```

`data = torch.randn(num_elements, device=cuda_if_available(rank))` 造出随机张量。重点是 `start_time = time.time()` 之前必须同时做 `torch.cuda.synchronize()`（等 CUDA kernel 跑完）和 `dist.barrier()`（等所有进程都到达这里）。讲课人强调存在**两种异步**：CUDA kernel 相对于 host 进程异步、以及不同进程之间彼此异步，所以两种同步缺一不可；屏幕黄色高亮正是 `torch.cuda.synchronize() # Wait for CUDA kernels to finish`。计时结束后同样要再同步一次再取 `end_time = time.time()`，否则测到的只是「把命令提交出去」的时间，而不是「通信真正完成」的时间。`dist.all_reduce` 即便设了 `async_op=False`，也只保证集合调用在 host 侧返回，GPU 上 kernel 是否结束仍要靠 `torch.cuda.synchronize()` 兜底。

这段代码在每一个 rank 上都要跑一遍，因此输出是四条独立记录，而且日志里出现的顺序是 `Rank 0 2 1 3` 这样的乱序——与本章开头讲的「打印顺序由硬件决定」完全一致。实测 `all_reduce(world_size=4, num_elements=104857600)` 的耗时分别为 `1.60ms`（rank 0）、`1.38ms`（rank 1）、`1.50ms`（rank 3）等，讲课人据此说「这次大约 1.6 毫秒」。因为它们来自不同进程，各报各的测量值，彼此有差异是正常的；若只想对外报一个数，可以取平均（take the average），这与前面算 MFU 时把各 rank 结果汇总成单一指标的做法一致。这里的 `duration = end_time - start_time` 用的是 `time.time()` 的 wall-clock 时间，也是下一章公式里的 $T$。

这个计时协议直接服务于下一章的「有效带宽」：只有先把 `duration` 测准，$\text{bytes}/T$ 才有意义。易错点恰恰在同步的顺序上（54:18 的问答会再回到这里）：如果先 `barrier()` 再 `synchronize()`，各进程可能只是各自到达屏障、而 CUDA kernel 还没跑完，屏障实际上没有起到全局对齐的作用；正确写法就是代码里这样的「先 synchronize、后 barrier」。另一个容易忽略的细节是，计时区间里包含了第二次 `dist.all_reduce` 与两次同步的开销，因此 `duration` 不是纯粹的「链路上比特飞行时间」，而是该进程观察到的端到端墙钟耗时；这正是后面把分母乘 `world_size`（当作所有 rank 一共等待的时长）来处理的原因。

![all_reduce 基准测试函数：torch.randn 建张量、warmup、cuda.synchronize 与 barrier 双重同步](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0037_00-47-29.jpg)

## 有效带宽：从发送字节数到两条「无关」性质 [49:10 - 51:30]

光有一个 `1.6ms` 无法判断快慢，于是类比 MFU 引入**有效带宽（effective bandwidth）**：先算「这次计算理应发送多少字节」，再除以总耗时。屏幕第 322–330 行在 `dist.barrier()` 之后给出原文：

```python
# Measure the effective bandwidth
dist.barrier()
size_bytes = data.element_size() * data.numel()
sent_bytes = size_bytes * 2 * (world_size - 1)  # 2x because send + receive, world_size-1 steps in all-reduce
total_duration = world_size * duration
bandwidth = sent_bytes / total_duration
print(f"[all_reduce] Rank {rank}: all_reduce measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)
```

`size_bytes = data.element_size() * data.numel()` 就是「每个元素的字节数 × 元素个数」，对应讲课人说的「这个数据张量一共多少字节」。真正发送的量需要拆开：以 $W=4$ 为例，`rank0 + rank1 + rank2 + rank3` 需要做 $W-1=3$ 次加法（屏幕注释写 `world_size-1 steps in all-reduce`）；再乘 2，因为「既要 send 也要 reduce/receive」（注释 `# 2x because send + receive`）；于是 $\text{sent\_bytes} = \text{size\_bytes}\times 2\times(W-1)$。分母不是单进程看到的 `duration`，而是 `total_duration = world_size * duration`——讲课人的说法是「乘上 world size，相当于所有 rank 一共等待的时长」。最后 `round(bandwidth / 1024**3)` 把字节每秒换算成 $\text{GB/s}$（这里除以 $1024^3$）。

整理成公式，其中 $W$ 是 world size、$T$ 是单 rank 测得的 wall-clock 时长：

$$
\text{bandwidth} = \frac{\text{size\_bytes}\times 2\times (W-1)}{W \times T},\qquad \frac{W-1}{W}\xrightarrow[W\to\infty]{}1,\qquad \text{bandwidth}\approx \frac{2\,\text{size\_bytes}}{T}
$$

屏幕 `# Notes:` 下的三条 bullet 原样为：`# - Effective bandwidth ~ 2 * size_bytes / total_duration`、`# - Independent of world_size`、`# - Independent of topology (ring or tree)`；第一条是便于记忆的近似写法，精确形式仍以代码里的 `sent_bytes / total_duration` 为准。代入实测，讲课人报出 "about 400 GB per second"，日志里逐 rank 为 `390 GB/s`、`366 GB/s`、`425 GB/s`，量级一致。这个指标衡量的更接近链路与实现的有效吞吐，而不是随 GPU 数量线性增长的「总流量」。

「与 world size 无关」是件好事：把 GPU 从 4 张扩到更多张，按这个定义算出的带宽不会凭空变差，通信时间主要由数据量与链路速率决定。「与拓扑无关」则来自 NCCL：它自己决定用 ring 还是 tree 来传递消息（讲课人原话 "whether you're going to pass the messages in a kind of a ring or a tree topology"），上层按同一个公式计算即可，不必关心底层走了哪条路径。这也呼应了第一部分讲的 NCCL 三项职责——探测硬件拓扑、选择 GPU 间路径、启动通信 kernel；屏幕上 `Independent of topology (ring or tree)` 把这件事压缩成了一行注释。

这一章的公式是下一章对比 reduce-scatter 的基准：只要把系数 2 去掉，就能解释「为什么 all-reduce 搬两倍数据却得到相同带宽」。局限有两点必须点明：其一，$\frac{2\cdot\text{size\_bytes}}{T}$ 只是 $W$ 较大时的近似，$W=4$ 时 $\frac{W-1}{W}=0.75$，精确值要用 $\frac{2(W-1)}{W}$ 修正，不能把「无关」误读成「任何 $W$ 下数值完全相等」；其二，这个带宽是**按算法语义推算的字节数除以耗时**，不是硬件规格书上的峰值带宽，实测受张量大小、warmup 是否充分、以及多进程抖动影响，日志里 366 与 425 GB/s 之间的差距就是这种随机性（讲课人后面也说 "sometimes there's some stochasticity"）。因此它适合做同一套代码、不同策略之间的相对比较，而不宜当作硬件的绝对标称值。

![有效带宽计算：sent_bytes = size_bytes * 2 * (world_size - 1) 与黄色高亮注释](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0039_00-50-00.jpg)

## reduce-scatter 带宽、实测数据与「两倍数据、两倍时间」 [51:30 - 53:30]

51:37 之后，讲课人把同一套基准测试搬到 reduce-scatter 上，说「非常类似（very similar）」：同样是创建输入与输出、warm up、执行操作、计时，同样可以测有效带宽。屏幕第 335 行附近能看到新函数的开头：

```python
cleanup()
def reduce_scatter(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)
    # Create input and outputs
```

差别在于发送字节数：reduce-scatter 的输入字节数照算，但「这里没有那个 2x（here there is no 2x here）」。按上一章同一套记法，去掉系数 2 后可写成

$$
\text{bandwidth}_{\text{RS}} = \frac{\text{size\_bytes}\times (W-1)}{W \times T}
$$

也就是说，分子（语义上应发送的字节）与分母（总时长）都不带 all-reduce 的 send+receive 那一倍。讲课人给出的结论是：带宽应该「非常相似」，偶尔有些随机波动，但都在 400 多的量级（"in the kind of 400s"）。这里与上一章形成直接呼应——公式结构完全平行，唯一改动就是把 2 拿掉。

真正有说服力的是屏幕上的运行日志（52:32 那一帧），它同时给出了上一章 all-reduce 与本例 reduce-scatter 的实测值。`all_reduce(world_size=4, num_elements=104857600)` 的耗时是 `1.60ms`、`1.38ms`、`1.50ms`，测得带宽为 `390 GB/s`、`366 GB/s`、`425 GB/s`；`reduce_scatter(world_size=4, num_elements=104857600)` 的耗时为 `2.47ms`、`2.39ms`、`2.39ms`，测得带宽为 `475 GB/s`、`450 GB/s`、`490 GB/s`。日志末尾还带出了下一部分数据并行的首步输出，例如 `[data_parallelism] Rank 2: step = 0, loss = 0.011991873383522034, params = ['1024x1024[-0.0299...]', ...]`、`[data_parallelism] Rank 0: step = 0, loss = 0.01151061337441206`，说明这份 `stdout` 是把多个示例连着跑出来的。这些数字把「约 400 GB/s」「in the kind of 400s」「sometimes there's some stochasticity」三句口头判断落成了可核对的具体值；同一 `world_size=4`、同一 `num_elements=104857600` 也保证了两者可比。

讲课人随后给出解释：`A all reduce is, remember as we stated, reduce scatter plus an all gather`。因此 all-reduce 天然要搬**两倍的数据**——reduce-scatter 有一段开销，all-gather 有一段开销，all-reduce 等于把两份工作都做了；相应地它也花**两倍的时间**。但「两倍」出现在分子与分母两边，恰好相互抵消，所以两者算出的带宽是同一个量级。用式子概括就是

$$
\text{all-reduce} = \text{reduce-scatter} + \text{all-gather},\quad \text{bytes}_{AR}\approx 2\,\text{bytes}_{RS},\quad T_{AR}\approx 2\,T_{RS}\ \Longrightarrow\ \frac{\text{bytes}_{AR}}{T_{AR}}\approx\frac{\text{bytes}_{RS}}{T_{RS}}
$$

这既是上一章恒等式在性能维度上的推论，也解释了为什么 FSDP/ZeRO 把 all-reduce 拆成两步后，通信总量并不会凭空翻倍——拆开是为了在中间管理存储，而不是为了少传数据。

与前后内容的关系上，本章是「第一部分：building blocks of distributed communication/computation」的收尾：程序、计时协议、带宽公式、两种操作的对照都齐了，紧接着 53:19 讲课人宣布 "that is the end of part one"。局限方面要诚实：屏幕日志里 reduce-scatter 的耗时（`2.39ms` 上下）明显比 all-reduce（`1.38ms` 上下）更长，与口头说的「all-reduce 花两倍时间」在字面上并不吻合，讲课人也只用了 "should be very similar" 这类留有余地的措辞并提到 stochasticity，因此不应把「严格两倍」当成普适定律；此外这些数字来自单次 `world_size=4` 的运行，换互连、换张量大小、换 NCCL 版本都会变，能稳固保留的只是「同一量级、随 world size 与拓扑基本不变」这一结构性结论。

![基准测试运行日志：all_reduce 与 reduce_scatter 的耗时及 390/366/425、475/450/490 GB/s 带宽](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0041_00-52-32.jpg)

## 同步顺序的问答与从第一部分过渡到分布式训练 [53:30 - 55:07]

第一部分结束后进入问答，两个问题都围绕上一章反复出现的同步动作。第一个问题是「为什么还要为 CUDA kernel 做 synchronize」。讲课人的回答很直接：说到底我们仍然在做 CUDA 操作（"at the end of the day, we are still doing CUDA operations"），只是现在有多个进程、每个进程带一张 GPU、各自在跑 CUDA 操作；而 CUDA 操作默认就是异步的，所以当 Python 执行到下一行时，那个操作可能还没结束。因此**总是要等到它完成**，办法就是 `torch.cuda.synchronize()`。这把前面两章里那两行同步代码的必要性讲透了：`dist.all_reduce` 这类集合调用只是把 NCCL 的通信 kernel 排进 GPU 队列，真正的完成时刻由 GPU 决定，host 侧的 `async_op=False` 并不替代 GPU 同步。

第二个问题是「要不要先 barrier 再 synchronize」，讲课人先说 "not sure"，随后给出一种会出问题的情形：如果先 `barrier()`，而 CUDA 还没跑完，你就立刻冲到屏障那里；于是各进程其实还是在**各自独立地同步各自的 CUDA kernel**，"which means that you're not really synchronized"。更糟的是，如果那些操作只是「返回」了（barrier 本身不关心 GPU 队列），"the barrier doesn't really do any thing"。换句话说，屏障对齐的是「进程到达某一点」，而不是「GPU 工作已完成」；的正确用法仍是基准测试代码里那样——先 `torch.cuda.synchronize()` 再 `dist.barrier()`，让每个进程先排空自己的 GPU 队列，再在屏障处汇合。可以用一行伪代码记住这个顺序：

```python
dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
torch.cuda.synchronize()  # Wait for CUDA kernels to finish
dist.barrier()            # Wait for all the processes to get here
end_time = time.time()
```

这是本节唯一需要「照抄」的代码片段，它同时也是 54:18 那段问答的结论：顺序不是风格问题，而是会把计时测歪的正确性问题。

55:07 讲课人说 "let me move on"，正式开始想「怎么训练模型」。屏幕第 41–55 行是一张承上启下的大纲页，先交代运行方式：`# When you execute this lecture directly (python lecture_07.py), it uses multiprocessing, which produces output each process (below).`、`# However, when you trace this lecture (python -m execute -m lecture_07), we turn off multiprocessing.`、以及链接文字 `[stdout for this lecture]`。随后列出 `Part 1: building blocks of distributed communication/computation` 的四个函数——`collective_operations()  # Programming model`、`hardware()  # Hardware: how GPUs are connected`、`torch_distributed()  # How this is implemented in NCCL/PyTorch`、`benchmarking()  # Measure actual NCCL bandwidth`——这正好把前面几章逐一归档。紧接其下用黄色高亮标出 `Part 2: distributed training`，并写明 `Walk through bare-bones implementations of each strategy on deep MLPs.`、`Recall that MLPs are the compute bottleneck in Transformers, so this is representative.`，以及三种切法：`data_parallelism()  # Cut up along the batch dimension`、`tensor_parallelism()  # Cut up along the width dimension`、`pipeline_parallelism()  # Cut up along the depth dimension`。

这页大纲既是第一部分的目录，也是下一章数据并行的入口：batch / width / depth 三个维度，决定了后面几种并行策略各自切什么。局限与易错点也在这里埋下：讲课人反复强调这是 "bare-bones"（骨架式）实现，MLP 之所以有代表性只是因为它是 Transformer 的计算瓶颈，并不意味着真实 Transformer 的前向结构、attention 或归一化都被覆盖；另外 `python lecture_07.py` 与 `python -m execute -m lecture_07` 两种运行方式的输出形态不同，直接读多进程 `stdout` 时会看到各 rank 交错的打印（前面日志里 `Rank 0 2 1 3` 的乱序就是这么来的），而本课逐行讲解时关掉了 multiprocessing，因此读者自己复现时看到的顺序可能与课堂截图不一致。

![课程大纲：Part 1 四个函数与黄色高亮的 Part 2 分布式训练三种切分维度](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0042_00-53-48.jpg)

## 数据并行的切分示意、参数表与 MLP 准备 [55:07 - 58:00]

讲课人先给出「脑子里该有的那张图」：data parallelism（数据并行）、tensor parallelism（张量并行）、pipeline parallelism（流水线并行）三种切法，并提醒「这只是一个 schematic，别想得太深」，用途是把「怎么切数据和参数」概念化。数据并行的定义是：把数据切成若干片，每张 GPU 负责其中一片，同时**保留全部参数**、照常做模型训练，最后再做一次同步。屏幕上对应的示意图里，绿色的 `layer 0`、`layer 1` 两个方框保持完整（参数不切），灰色 `Data` 方框被一条橙色横线切开；第 377 行黄色高亮写着 `Sharding strategy: each rank gets a slice of the data`。同一屏的 `Notes` 三条把因果链写得很清楚：`Losses are different across ranks (computed on local data)`、`Gradients are all-reduced to be the same across ranks`、`Therefore, parameters remain the same across ranks`；末尾还预告 `Next time: FSDP/ZeRO: use all-gather and reduce-scatter to avoid holding all parameters in memory`，把上一章的恒等式重新拉回视野。

示例的入口是 `data = generate_sample_data()` 与 `spawn(data_parallelism_main, world_size=4, data=data, num_layers=4, num_steps=1)`：样本数据是 `batch_size = 128`、`num_dim = 1024` 的矩阵，即一个 $128\times 1024$ 的「batch size by num dim」数据矩阵，层数 `num_layers=4`，只跑 `num_steps=1` 步。函数签名是

```python
def data_parallelism_main(rank: int, world_size: int, data: tensor, num_layers: int, num_steps: int):
    # Get the slice of data for this rank (in practice, each rank should load only its own data)
    # --- B0 ---
    # --- B1 ---
    # --- B2 ---
    # --- B3 ---
    batch_size = data.size(0)
    num_dim = data.size(1)
    local_batch_size = int_divide(batch_size, world_size)
    start_index = rank * local_batch_size
    end_index = start_index + local_batch_size
    data = data[start_index:end_index].to(cuda_if_available(rank))
```

屏幕覆盖的参数表给出具体值：`batch_size = 128`、`num_dim = 1024`、`local_batch_size = 32`、`start_index = 0`、`end_index = 32`（这是 rank 0 的情形）。把数据矩阵的**行**按 `world_size` 切成 4 份，注释 `# --- B0 ---` 到 `# --- B3 ---` 正好对应四片，每个 rank 拿一片；`local_batch_size = int_divide(batch_size, world_size)` 即 $128/4 = 32$，所以每张 GPU 只看 32 个数据点。切片靠 `start_index = rank * local_batch_size`、`end_index = start_index + local_batch_size` 算出，再用 `data[start_index:end_index].to(cuda_if_available(rank))` 搬到本 rank 的设备上（这一行被黄色高亮）。至此每个 GPU 都持有一个彼此不同的 `data` 张量。

模型与优化器紧随其后，屏幕第 412–414 行给出：

```python
# Create MLP parameters params[0], ..., params[num_layers - 1] (each rank has all parameters)
params = [get_init_params(num_dim, num_dim, rank) for layer in range(num_layers)]
optimizer = torch.optim.AdamW(params, lr=1e-3)  # Each rank has own optimizer state
```

即 MLP 有 `num_layers` 层、每层是一个 `num_dim × num_dim` 的 $1024\times 1024$ 矩阵，用 `get_init_params(num_dim, num_dim, rank)` 随机初始化；注释明确写 `# Each rank has all parameters`（每个 rank 都持有全部参数），优化器是 `torch.optim.AdamW(params, lr=1e-3)`，且 `# Each rank has own optimizer state`（每个 rank 各自维护优化器状态）。日志里打印的 `params = ['1024x1024[-0.0299...]', '1024x1024[-0.0279...]', ...]` 正是这四个方阵的摘要。这些是屏幕上有、而转写只含糊说「随机初始化一组参数、交给优化器」的细节：`AdamW`、`lr=1e-3`、`num_dim × num_dim`、以及参数与优化器状态在每个 rank 上都有完整副本。

与前后内容的关系：本章把上一页大纲里的 `data_parallelism() # Cut up along the batch dimension` 落实成具体张量切片，并为下一章的训练循环准备好 `params`、`optimizer` 与局部 `data`。易错点有两处。其一，讲课人特别说明「实践中每个 rank 应该自己加载自己的数据，而不是像这里先由一处生成再分发」——示例里的 `generate_sample_data()` 是一次性造好整块数据再切，会形成一个本可避免的瓶颈，只为讲解方便。其二，`local_batch_size = 32` 依赖 `batch_size` 能被 `world_size` 整除；这里用了 `int_divide`，若不能整除就得自己处理余数或 padding，否则各 rank 拿到的行数不一致，`all_gather`/`all_reduce` 的形状对不上。另外，参数与优化器状态在每个 rank 上完整复制，正是 DDP 的显存局限，也是 `Notes` 里预告 FSDP/ZeRO 要用 reduce-scatter 把状态分片的原因。

![数据并行切分示意图：layer 0/1 完整、Data 被切开，附 Notes 与 FSDP/ZeRO 预告](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0044_00-56-20.jpg)

## 训练循环、梯度同步与 DDP 的优雅之处 [58:00 - 60:46]

训练循环是数据并行真正落地的地方，屏幕第 414–434 行给出了完整代码：

```python
optimizer = torch.optim.AdamW(params, lr=1e-3)  # Each rank has own optimizer state
for step in range(num_steps):
    # Forward pass
    x = data
    for param in params:
        x = x @ param
        x = F.gelu(x)
    loss = x.square().mean()  # Loss function is average squared magnitude
    # Backward pass
    loss.backward()
    # Sync gradients across workers (ONLY difference between standard training and DDP)
    for param in params:
        dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)
    # Update parameters
    optimizer.step()
    print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = {[summarize_tensor(params[layer]) for layer in range(num_layers)]}", flush=True)
```

前向传播里 `x = data` 用的**不是全部数据**：讲课人强调「如果我是 rank 2，我只拿到 B2 那一片」。随后 `for param in params:` 逐层做 `x = x @ param` 与 `x = F.gelu(x)`（黄色高亮在 `x = x @ param` 上），损失函数是 `loss = x.square().mean()`，屏幕注释写 `# Loss function is average squared magnitude`（平均平方幅度）。`loss.backward()` 之后进入关键一步：注释 `# Sync gradients across workers (ONLY difference between standard training and DDP)` 明说，这是标准训练与 DDP（Distributed Data Parallel，分布式数据并行）**唯一的差别**——对每个参数做一次 `dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)`，然后 `optimizer.step()` 更新参数，最后打印 `step`、`{loss.item()}` 和用 `summarize_tensor` 概括的各层参数。

这里有一个转写没明说、但屏幕上写得很清楚的细节：同步梯度用的是 `op=dist.ReduceOp.AVG`，也就是由集合操作**直接完成平均**，而不是先 `ReduceOp.SUM` 再手动除以 `world_size`。讲课人口头说的是「I'm going to average」，落到代码里就是这一个枚举值。为什么必须平均？因为每个 rank 只在自己的 32 行局部数据上算梯度，四个 rank 的 `param.grad` 天然不同；把四份梯度取平均，等价于一整块 $128\times 1024$ 数据算出来的梯度。用式子表示，令 $g_r$ 为 rank $r$ 的局部梯度：

$$
\bar{g} = \frac{1}{W}\sum_{r=0}^{W-1} g_r = \text{all-reduce}_{\text{AVG}}(g_0,\dots,g_{W-1}),\qquad W = 4
$$

all-reduce 结束后，每个 rank 都持有完全相同的 $\bar{g}$，于是各自 `optimizer.step()` 之后参数也保持一致——这与 `Notes` 里 `Gradients are all-reduced to be the same across ranks` / `Therefore, parameters remain the same across ranks` 完全对应。屏幕上首步日志里各 rank 的 loss 并不相同（rank 2 是 `0.011991873383522034`、rank 0 是 `0.01151061337441206`、rank 3 是 `0.012755745090544224`），正好印证 `Losses are different across ranks (computed on local data)`；但参数摘要 `1024x1024[-0.0299...]` 在同步后必须收敛到一致。

讲课人反复用 "elegant"（优雅）形容这个设计：它基本就是标准训练，只不过把标准训练套在各自的 local batch 上，再在反向传播之后插一句「用 all-reduce 把所有梯度平均一下」，是「一行代码的改动（a one-line code change）」。改动之后，每个 rank 更新参数时表现得「好像全部数据都在自己卡上（as if it were have all the data on it）」，而实际上它只处理了 128 行里的 32 行。这一章因此把前面所有零件串了起来：`dist.all_reduce` 来自 40:31 的第一个示例，`ReduceOp` 家族来自基准测试，参数与优化器来自上一章的 MLP 准备，而梯度同步之所以可能，正是因为 all-reduce 会把结果复制到所有 rank。

与前后内容的关系上，60:36 讲课人收束为「that's basically uh DDP or the first type of data parallel」并问有没有问题；屏幕 `Notes` 里预告的下节课 FSDP/ZeRO 则指向另一种数据并行——不再让每个 rank 都持有全部参数。局限必须点明：DDP 里每个 rank 都保存完整的参数、梯度与优化器状态（本章代码里 `params` 是四个 $1024\times 1024$ 矩阵、`AdamW` 状态各自一份），模型一旦放不下单卡就无能为力；梯度同步发生在 `loss.backward()` 之后、`optimizer.step()` 之前，是一次阻塞式（`async_op=False`）的全局通信，本课没有做「某层梯度一就绪就发送」这类计算通信重叠，讲课人在第二部分末尾把这留给了作业；此外示例用 `num_steps=1`、只跑一步，目的是展示机制而非收敛效果，`loss = x.square().mean()` 也只是个便于演示的玩具损失，不是语言模型的训练目标。

![DDP 完整训练循环：forward/backward 后以 dist.ReduceOp.AVG 同步 param.grad，再 optimizer.step()](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0047_01-00-08.jpg)

## DDP 的收尾问答、切片约束与三条小结 [60:46 - 63:00]

本段是从上一节 DDP（Distributed Data Parallel，分布式数据并行）的现场问答切入的。先把 DDP 的直觉再钉一遍：每个 rank（一张 GPU，或者说一个进程）都持有完整的模型参数，只沿 batch 维度把数据切开，各自在本地数据上做前向与反向，最后靠一次 all-reduce 把梯度同步成一致。学生问的第一个问题是「是不是只有 batch size 大于 1 才能这么做」，回答是肯定的，而且 batch size 至少要不小于 world size（设备总数）才有意义，通常还应当「明显更大一些」（quite a bit larger）；紧接着第二个问题是「batch size 是否应该是 world size 的倍数」，回答是「那样当然最好」，如果不整除也可以用补零（pad it with zeros）之类的办法对齐，只是「如果整除，对所有人都更省事」。这背后的直觉并不深奥：每个 rank 至少要分到一行数据，否则有的卡空转；而整除能让 local batch size 是个整齐的整数，省掉掩码、补零和额外的记账。

61:44 有人把问题推向 Transformer：这套做法放到 Transformer 上会变成什么样？讲者的回答是「基本一样」（basically the same）。理由在于 DDP 有一个非常 modular（模块化）的好处：你只管做自己的 forward pass，DDP 只是把各 rank 的梯度平均掉，它「不关心你的前向长什么样」。换句话说，改动被限制在训练循环里、模型被视为一个黑盒模块，这正是数据并行最优雅的地方——切的是数据，模型代码几乎不动。这里也顺带纠正一个很容易记混的点：各 rank 上的 loss 并不相同，因为每个 rank 是在自己的局部数据上算的；真正被同步的是梯度，而不是 loss 本身。

62:10 起讲者用三句话给 DDP 做了小结，幻灯片把这三条写成了 Notes。画面上「Sharding strategy: each rank gets a slice of the data」（切片策略：每个 rank 拿到数据的一片）之下是启动代码：

```python
data = generate_sample_data()
spawn(data_parallelism_main, world_size=4, data=data, num_layers=4, num_steps=1)
```

这里的具体参数是 world_size=4、num_layers=4、num_steps=1，画面底部露出的 `def generate_sample_data():` 里还能读到 `batch_size = 128`。Notes 三条依次是：Losses are different across ranks (computed on local data)——各 rank 的 loss 不同，因为是在局部数据上算的；Gradients are all-reduced to be the same across ranks——梯度经 all-reduce 后在各 rank 上一致；Therefore, parameters remain the same across ranks——因此参数更新后在各 rank 上始终保持一致。讲者口述的版本是「loss 不同、梯度一开始也不同，但都被 reduce 成一样，所以参数保持一样」。

![数据并行切片策略与三条 Notes，以及底部黄色高亮的 FSDP/ZeRO 预告](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0049_01-02-56.jpg)

这张幻灯片底部还有一条黄色高亮注解，是本段承上启下的关键：Next time: FSDP/ZeRO: use all-gather and reduce-scatter to avoid holding all parameters in memory（下节课：FSDP/ZeRO，用 all-gather 与 reduce-scatter 来避免把所有参数都攥在显存里）。它正好点破 DDP 的局限：62:37 讲者说这里用的 all-reduce 是一个「很简单的、整块的操作」（a very simple monolithic operation），但代价是必须把全部模型参数放进内存；一旦模型参数放不下，就得「更聪明一点」，也就是把 all-reduce 拆成 reduce-scatter + all-gather 两步，好在中途介入、管理存储——这正是下节课由 Tatsu 讲的 FSDP（Fully Sharded Data Parallel）与 ZeRO。易错点也就在这里：不要把 FSDP 记成另一种「切数据」的方法，它切的是参数与优化器状态本身；也不要因为「梯度被 all-reduce 成一致」就顺推出 loss 也被同步。

## 张量并行：不切数据而切每一层 [63:00 - 65:00]

63:00 讲者用「切法」把话题转到 tensor parallelism（张量并行）：前面数据并行是沿 batch 方向切，现在「不切数据，而是切每一层」（we're not going to cut the data, we're going to cut essentially each layer），让每个 rank 只拿到每层参数的一部分（each rank is going to get part of each layer）。他同时预告了一个代价：这种切法「一般要传输多得多的数据」，后面会具体讨论。为了让推导清爽，这里先做一个简化假设——每个 rank 都拿到全部数据（Every rank has all the data, just for simplicity），数据形状仍然是 batch size × num dim。这个假设只是本讲为了少一层记账而设的，不要误以为张量并行天然会复制数据；它只是不沿 batch 维切而已。

沿宽度切之后需要一个新的量：local num dim，定义为「对本 rank 而言，我只负责 num dim 个维度里的一个子集」。代码里写成 `local_num_dim = int_divide(num_dim, world_size)`，注释直接标了 `# Shard \`num_dim\``。于是每个模型依然拥有全部层（the model still has all the layers），但每层的参数矩阵不再是 $num\_dim \times num\_dim$，而是

$$
\text{params}_\ell \in \mathbb{R}^{\,num\_dim \times local\_num\_dim}, \qquad local\_num\_dim = \frac{num\_dim}{world\_size}
$$

讲者说这相当于「沿列往下劈」（splitting down the columns），所以叫 column tensor parallel（列张量并行）；他还特意补了一句「也可以按行切（by rows），但现在先不讲」。这个命名很重要，因为后面 all-gather 拼接时用的是 `dim=1`，正是列方向。

![tensor_parallelism_main 中 local_num_dim 与按 rank 切分参数的初始化代码](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0050_01-04-23.jpg)

64:23 这张幻灯片（第 443–459 行）给出了完整的主函数骨架，把上面的公式落成了代码：

```python
data = generate_sample_data()
spawn(tensor_parallelism_main, world_size=4, data=data, num_layers=4)

def tensor_parallelism_main(rank: int, world_size: int, data: tensor, num_layers: int):
    setup(rank, world_size)
    data = data.to(cuda_if_available(rank))  # All ranks get the data (batch_size x num_dim)
    batch_size = data.size(0)
    num_dim = data.size(1)
    local_num_dim = int_divide(num_dim, world_size)  # Shard `num_dim`

    # Create model (each rank gets 1/world_size of the parameters)
    #  |  |  |  |
    # W0 W1 W2 W3
    params = [get_init_params(num_dim, local_num_dim, rank) for layer in range(num_layers)]
```

具体数字与参数都很清楚：world_size=4、num_layers=4；那段 `#  |  |  |  |` / `# W0 W1 W2 W3` 的示意小图表示 4 个 rank 各自认领一列分片，所以注释写「each rank gets 1/world_size of the parameters」；`get_init_params(num_dim, local_num_dim, rank)` 把 rank 传进去，说明初始化本身就是按 rank 挑自己那一片，而不是先建完整矩阵再切。与上一章 DDP 的对照极其鲜明：DDP 里每个 rank 都构造完整的 $num\_dim \times num\_dim$ 矩阵，这里每个 rank 只构造 $num\_dim \times local\_num\_dim$，显存占用直接降到 $1/world\_size$。

画面下方已经露出 `# Forward pass`、`x = data`、`for layer in range(num_layers):`、`# Compute activations (batch_size x local_num_dim)`，为下一章的前向计算埋好了伏笔。易错点有两处：一是 `int_divide` 隐含 num_dim 必须能被 world_size 整除，否则各列分片宽窄不一——这与上一章「batch size 最好整除 world size」是同一类约束，只是这次落在维度上而不是样本上；二是别把「每个 rank 拿全部数据」当成张量并行的定义特征，它只是简化假设，真正的定义特征是「每个 rank 拿每层参数的一个列切片」。

## 列并行前向传播：all-gather 激活与拼接回完整维度 [65:00 - 67:30]

前向要做的事是「逐层计算激活」（compute activations）。从 X（也就是 data）出发，访问第 layer 层的参数；关键在于 `params[layer]` 只是「一片」（only a slice of the parameters），如果当前是 rank one，就只拿到矩阵里属于自己的那部分。但这并不妨碍继续算：一次矩阵乘法本来就可以拆成一堆更小的矩阵乘法，分派到不同 rank 上分别做，再把结果 gather 回来。而且中间的非线性 `F.gelu(x)` 是逐元素（element-wise）的，可以直接作用在局部激活上，完全不需要跨卡通信。讲者 65:16 的原话是「我照样能往下走，因为这个非线性本来就是逐元素的」。

65:50 这张幻灯片把前向的代码铺开了：

```python
# Forward pass
x = data
for layer in range(num_layers):
    # Compute activations (batch_size x local_num_dim)
    x = x @ params[layer]  # Note: this is only on a slice of the parameters
    x = F.gelu(x)

    # Allocate memory for activations (world_size x batch_size x local_num_dim)
    activations = [torch.empty(batch_size, local_num_dim,
                               device=cuda_if_available(rank))
                   for _ in range(world_size)]
```

具体形状参数值得逐一看：`batch_size x local_num_dim` 是局部激活的形状；`activations` 是一个长度为 world_size 的列表，注释写成 `world_size x batch_size x local_num_dim`，也就是给每个 rank 预留一份空位。这里用 `torch.empty` 只分配显存、不做初始化，正好对应讲者 66:19–66:28 反复澄清的那句「这是在给激活分配内存，X 才是真正的那部分激活」（this is the allocating memory for the activations. So, X is the actual part of the activation）。把「缓冲区」和「数据」混为一谈，是读这段代码时最常见的误读。

![列并行前向计算、为激活分配 world_size 份显存的代码](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0051_01-05-50.jpg)

接下来就是本模块的核心通信。讲者 65:45 说「我需要把全部激活放到所有 rank 上」，而前面介绍过的 all gather 这个集合通信原语（collective primitive）正好能干这件事：

```python
    # Send activations via all gather
    dist.all_gather(tensor_list=activations, tensor=x, async_op=False)

    # Concatenate them to get batch_size x num_dim
    x = torch.cat(activations, dim=1)
```

`dist.all_gather(tensor_list=activations, tensor=x, async_op=False)` 的语义是：每个 rank 都持有自己的 `x`，all-gather 结束后 `x` 被拷贝到 `activations` 列表中各自对应的位置，于是每个 rank 都拿到了全部 world_size 份分片；`async_op=False` 表示同步调用、等到通信完成才返回。随后 `torch.cat(activations, dim=1)` 沿第 1 维把分片拼回完整的 `batch_size x num_dim` 的 X，这样下一层才有完整输入可用。67:17 这张幻灯片末尾还留了两行：一行是 `print(f"[tensor_parallelism] Rank {rank}: forward pass produced activations {summarize_tensor(x)}", flush=True)`，用来打印各 rank 前向算出的激活摘要；另一行是注释 `# Backward pass: homework exercise`——反向传播被明确留作作业，不在课堂上实现。

![通过 dist.all_gather 发送激活、再用 torch.cat 拼回完整维度的代码](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0052_01-07-17.jpg)

这一步「对每一层都要做」（This is done for every layer），所以通信量随层数线性增长，正是 63:23 预告的「要传多得多的数据」的根源，也决定了张量并行后面只能待在高带宽域里。易错点：`dim=1` 必须是列方向，因为切的是列，若误写成 `dim=0` 就会沿 batch 维把样本拼起来；另外别把 `activations` 的长度记成 num_layers——它是 world_size（每个 rank 一份分片），与层数无关。67:09–67:18 讲者停下来问了一句「any questions about column tensor parallel?」，作为这一模块的收束，也把话题引向下一章「代价与反向」。

## 张量并行的侵入性代价、反向对偶与 autograd 的边界 [67:30 - 69:38]

67:20 起讲者把张量并行和数据并行做了一次结构性对比，结论是：数据并行很优雅（elegant），因为它按数据切，模型被当成一个整体模块（module）对待；而现在「我们不得不去折腾模型本身」（now we have to kind of muck around with the model）。张量并行强依赖一个线性代数事实：如果想做一次矩阵乘法，可以把它拆成一组更小的矩阵乘法，放到不同 rank 上算，再把结果 gather 回来。也就是说，改动从「在训练循环外加一行 all-reduce」升级成「重写每一层的前向与通信」，模型代码不再是黑盒。这个对比给三种并行排了个「侵入性」的序：DDP 几乎不动模型，张量并行必须改模型，下一章的流水线并行介于两者之间（只改层的归属）。

68:08 有人问反向传播（back prop）怎么办。回答是：反向时你手上有激活，必须把梯度「reduce scatter 到各不相同的梯度上」（reduce scatter to all the different gradients）。讲者随即点出 all-gather 与 reduce-scatter 之间的对偶性（duality）：「in forward, if you're all gathering, in the backward, you're reduce scattering」。把形状写清楚就是

$$
\text{forward: } \operatorname{all\_gather}\!\left(x_{b \times local\_num\_dim}\right) \to x_{b \times num\_dim}
\quad\Longleftrightarrow\quad
\text{backward: } \operatorname{reduce\_scatter}\!\left(\nabla_{b \times num\_dim}\right) \to \nabla_{b \times local\_num\_dim}
$$

这与本讲反复强调的恒等式 $\text{all-reduce} = \text{reduce-scatter} + \text{all-gather}$ 是同一套逻辑的两面：前向需要「把分散的结果拼成完整」，反向就需要「把完整的梯度归约并散回分散」。理解了这一层，下节课 FSDP/ZeRO 把 all-reduce 拆开管理存储的做法就不再突兀。

68:36 的追问把这章推到最实用的一个坑：「这些是不是 autograd 自动帮你做了？」讲者明确否定：如果只是调用 `loss.backward()`，它不会做，因为在那张计算图里根本没有「并行」这回事（there's no parallelism in that）；不过他也承认「PyTorch 有一堆东西是自动替你做好的」（PyTorch has all these things that are done automatically for you）。69:12–69:27 他解释了本课的取舍：「我们这里管理得相当显式」（we're managing things fairly explicitly），意思是本课根本不写反向传播，你必须自己管理、自己调用 reduce-scatter；而这「是设计使然，因为这是 336、是从零构建语言模型」（that's baked in by design because this is 336, building language models from scratch），并补了一句「实践中你大概不必这么做」（In practice, you probably wouldn't have to do that）。最该警惕的错误，就是想当然地以为 autograd 会像处理普通算子一样，自动把跨 rank 的集合通信插进反向图里。

把这一章放回整段的脉络：它解释了为什么张量并行「通信量大、代码侵入」，也为 76:19 的硬件选型埋下伏笔。数字上仍是 world_size=4、num_layers=4 这套 4 卡设定，数据沿用生成函数里的 batch_size=128；通信发生在每一层——每层一次 all-gather，反向对应一次 reduce-scatter，共 num_layers 次。局限也很直白：显式管理意味着代码量、出错面和调试成本都显著变大，每层的参数布局、缓冲区分配、拼接维度都得自己对齐，这正是「from scratch」要付的学费，也是作业与下节课要继续处理的问题。

## 流水线并行：按层切分、micro-batch 与点对点 send/recv [69:38 - 72:23]

69:38 讲者说「快速讲一下」pipeline parallelism（流水线并行）。切法又换了一个轴：沿「深度」切，每个 rank 拿到网络的一个层子集（a subset of the layers now），但在每一层内部保留全部维度（within each layer, it's going to get all the dimensions）；其中一个 rank 拿到全部数据，而数据最终会「以某种形式」流经所有 rank（every rank is going to see all the data in some form）。数据形状依旧是 batch size × num dim。这个「一个 rank 拿数据、其余 rank 接龙」的画面，和前面两种并行都不一样：它既不像 DDP 那样把数据同时摊给所有卡，也不像张量并行那样每层都做集合通信。

参数上的变化是：定义 local num layers 为「某个 rank 要处理的层数」，于是 local params 只有 local num layers 个，但每层内部依旧是 $num\_dim \times num\_dim$ 的完整方阵（而不是张量并行那种 $num\_dim \times local\_num\_dim$）。写成集合就是

$$
local\_params = \{\, W_\ell \in \mathbb{R}^{\,num\_dim \times num\_dim} \mid \ell \in \text{本 rank 的 } local\_num\_layers \text{ 层} \,\}
$$

把它和上一章的 $num\_dim \times local\_num\_dim$ 并排看，三种并行的切法一目了然：DDP 切的是数据的 batch 行，张量并行切的是每层权重矩阵的列（宽度），流水线并行切的是层的个数（深度）。这也是幻灯片上 `data_parallelism()` 标注「Cut up along the batch dimension」、`tensor_parallelism()` 标注「Cut up along the width dimension」、`pipeline_parallelism()` 标注「Cut up along the depth dimension」的原因。

70:52 起引入 micro batches（微批次）。讲者先声明「Tatsu 周三大概会细讲，我这里先讲为什么这么安排」：除了切层，还要把 batch 切成 num micro batches 个更小的批次；rank zero 拿到数据后把它 chunk 成微批次，对每个微批次依次做三件事——从前一个 rank 接收（receive it from the previous rank）、只在分配给本 rank 的那几层上跑 feed forward pass、再发送给下一个 rank。把转写里描述的步骤按原样标识符整理成代码骨架：

```python
for micro_batch in micro_batches:
    if rank == 0:
        x = micro_batch
    else:
        x = recv(from=rank - 1)          # point-wise 接收
    x = forward(x, local_params[rank])   # 只跑本 rank 负责的层
    if rank != world_size - 1:
        send(x, to=rank + 1)             # point-wise 发送
```

讲者 71:51 特意点出，这里用的是 `receive` 和 `send`，属于 point-wise（点对点）操作，「之前没讲过，但一看就懂」（fairly explanatory）：一句是「我是 rank，我从 rank minus one 接收这个张量」，另一句是「我把张量 X 发给 rank plus one」（72:01–72:19）。这是本段唯一不用集合通信（collective）的并行实现，与前两种形成鲜明对照。

![讲解 micro-batch 与 send/recv 流水线时的课堂画面](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0055_01-11-45.jpg)

与张量并行对比，流水线并行只在 stage 边界上做一次 send/recv，不必每层都 all-gather 大激活，通信量小得多，因此能容忍慢得多的互连——这一点在 76:46 之后会被展开成硬件选型的依据。易错点有两处：其一，`send/recv` 是阻塞的点对点调用，若不做微批次切分、也不做异步，后面的 GPU 会长时间空等，这正是下一章要处理的「气泡」；其二，别把「rank 0 拿全部数据」误读成数据并行——数据并没有按 batch 切开分发给各 rank，而是沿流水线依次流过，每个 rank 最终「以某种形式」看到全部数据。

## 流水线气泡、通信与计算重叠、以及作业二的伏笔 [72:23 - 75:09]

72:23 讲者回头解释「为什么要用 micro batches」。在流水线里，一个 rank 拿到数据、处理若干层后发给下一张 GPU，下一张 GPU 处理若干层再往下发，这是划分深网络很自然的方式；但麻烦在于会出现 pipeline bubbles（流水线气泡）——当某张卡「不是在算，而是在等别的张量算完」时，它就在空转（waiting around for other tensors to process），「结果相当低效」。把 batch 拆成更小的 micro batch 后，数据可以更细粒度地流动，「快速处理完就马上往下发」，让各个 stage 更早被填满，从而减少气泡的数量（reduce the number of pipeline bubbles）。注意转写用词是 reduce，不是 eliminate：微批次只是缓解，不是根除。

73:32 讲者补上第二个、也是「对这个朴素版本（very naive version）没有处理」的要点：overlapping communication and computation（通信与计算重叠），并强调它「对流水线并行其实非常重要」。给出的做法是——「如果给这些前面加个 I，它就变成异步（async）了」，即把 `send`/`recv` 换成 `isend`/`irecv`：

```python
# 转写 73:47-73:52："if you put a I before these, then it becomes kind of async"
req = dist.irecv(x, src=rank - 1)
dist.isend(x, dst=rank + 1)
# 还要加更多代码来管理这些异步请求（"you have to add more things to kind of manage the code"）
```

直觉是「当你在计算这里的时候，你可以同时在收数据或发数据」（while you're computing here, you can be receiving data or sending data），让 computation 和 communication 重叠起来，减少真正花在等待上的时间。用时间模型把它写清楚，朴素串行版本与重叠版本的差别是

$$
T_{\text{naive}} = T_{\text{compute}} + T_{\text{comm}}, \qquad
T_{\text{overlap}} \approx \max(T_{\text{compute}},\, T_{\text{comm}})
$$

其中 $T_{\text{comm}}$ 是 send/recv 的传输时间，$T_{\text{compute}}$ 是本地若干层前向的时间；重叠把原来加在总时长上的通信项藏进了计算项里。气泡的多少则与 stage 数（world_size 个流水线级）和 num micro batches 的个数直接相关：微批次越多、每级越早被喂满，空转比例越低。

74:24 讲者把同一个思想回扣到数据并行，说明这不是流水线独有的：DDP 里「我只是先做完整个反向传播，最后才做那一堆 all-reduce」，但其实「如果你聪明一点，反向传播中某个梯度一算完就可以开始发送」（as soon as the gradients are done, you can start sending that），不必等整段 backward 结束；这同样是通信与计算重叠，而且「这是你们在 assignment two（作业二）里会探索的内容」。75:09 有人再问 general models（更一般的模型）怎么办，回答是 MLP 已经给了「理解基础所需的大部分东西」（mostly of most of kind of what you need for understanding the basics），更大的模型只是 bookkeeping（记账）多得多，反而「更难看清核心算法」（harder to kind of see the core algorithms）——这为下一章列举未覆盖的并行方式做了铺垫。

易错点必须说清：别以为加了 `isend`/`irecv` 就万事大吉，讲者明说「还得加更多东西来管理代码」，也就是要管理异步 request 的完成与依赖，否则很容易出现张量被提前覆盖、recv 尚未就绪就开算之类的竞态；也别把「micro-batch 减少气泡」记成「消除气泡」。还有一个容易混淆的地方：流水线里的重叠是 send/recv 与 forward 重叠，数据并行里的重叠是 all-reduce 与 backward 重叠，二者的通信对象不同（点对点 vs. 集合操作）、所处的传播阶段也不同，但背后的「别让算力等数据」是同一个思想。

## 其它并行方式、硬件选型与编译器路线 [75:09 - 78:02]

75:36 起讲者列举本课没有覆盖的并行类型。sequence parallelism（序列并行）把一整条 sequence 切成若干段（chops it up into pieces），从而并行化 attention 计算；expert parallelism（专家并行）把 MoE 的 experts 切开，让不同 rank 负责不同专家，而这正是前面提过的 all-to-all 的用武之地（this is where the all-to-all that I mentioned comes in）；再就是「不同并行技巧的不同组合」（different combinations of different parallelization techniques），这些「也会出现在作业里」。幻灯片「What's missing?」清单与此一一对应：Communication/computation overlap、More general models (with attention, etc.)、Other forms of parallelism (e.g., sequence parallelism, expert parallelism, combinations)。讲者还重申了上一章的 MLP 判断：MLP 是 Transformer 的计算瓶颈，所以用深 MLP 做 bare-bones 实现是有代表性的（representative）。

76:08 一句总纲把话题从算法拉到系统：「你选哪种并行技术，会强烈依赖于硬件」（strongly dependent on the hardware）。张量并行通信量大，因为「每一层都要把这些相当大的激活发出去」（for every layer you need to send all these activations which are fairly big），所以它一般发生在节点内、走 NVLink 这种高带宽互连；而「你不会在没有 NVLink 的域里做张量并行」。流水线并行则能容忍慢得多的互连（slower interconnects），一些 decentralized training（去中心化训练）里 GPU 甚至「分布在半个地球之外」，用的就是流水线并行，但你绝不会在那种场景做张量并行。常见的组合可以写成

$$
\underbrace{\text{节点内: TP (NVLink / NVSwitch)}}_{\text{高带宽}}
\;+\;
\underbrace{\text{跨节点: DP / FSDP}}_{\text{中带宽}}
\;+\;
\underbrace{\text{必要时: PP}}_{\text{低带宽}}
$$

这正是 77:18–77:32 的口语版：「tensor parallel within a node，然后 data parallel 或 FSDP，然后 pipeline parallelism if you need it」。把上一章的 $1/world\_size$ 参数切分也放进来对照：world_size=4 时张量并行让每个 rank 只持有四分之一参数（幻灯片原话 each rank gets 1/world_size of the parameters），代价是每层一次大激活的 all-gather；流水线并行不省参数、但只传边界激活，所以能跨慢链路。

77:35 讲者又补了一个效应：数据并行可以做得很猛，但会撞上 critical batch size（临界批量）——一旦 batch size 继续增大「其实帮不了你」（it doesn't actually help you），那就「只是在浪费算力」，此时不如转向张量并行。转写只给了定性结论，没有给出任何阈值数字，因此这里不要臆造具体批量大小。78:02 起是本段最后一块内容，回到 TPU 与 PyTorch 两条路线：讲者说本课「是刻意地」（on purpose）使用 PyTorch，而且是「非常原始地」使用集合操作（using the collective operations in a very primitive way），为的是让你「机械地看到每一步在做什么」（so you can see mechanically what's happening）；另一条路是 JAX/TPU 阵营——只需定义模型和 sharding strategy（分片策略），由编译器决定需要哪些通信操作，「你只要说这块数据得出现在这里、这里和这里，编译器就做点魔法把它算出来」。幻灯片上写着 `Jax/TPUs: just define the model, the sharding strategy, and the Jax compiler handles the rest [levanter]`，紧接着是 `But we're doing PyTorch so you can see how one builds up from the primitives`；讲者最后的评论是「这很吸引人，但显然会拿走从零构建的很多乐趣」（it would take a lot of the joy out of actually building things from scratch）。

![Part 2: distributed training 幻灯片，列出三种并行切法与 What's missing 清单](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0058_01-16-49.jpg)

这一章的落点是：切法决定通信量，通信量决定能放在哪一层互连上，于是「并行策略」从来不是纯算法选择，而是算法与拓扑的联合设计。易错点有两个：别把「张量并行需要 NVLink」读成物理上的绝对禁令，它是带宽经济性上的建议，脱离高带宽域只是代价高到不划算；也别把编译器路线理解成「不需要通信」——编译器只是替你推导出该插哪些集合操作、替你决定分片，通信本身依然存在。此外 `[levanter]` 只是幻灯片里的一个引用标注，转写没有展开，不要替它编内容。

## 片尾总结：并行切法、互连要求与分层结构 [78:02 - 81:02]

79:08 起讲者做最后的口述总结，第一句是「there's many ways to parallelize」：可以 cut by data（按数据切）、cut by tensor 或 expert（按张量/专家切）、cut by pipeline（按流水线切）、cut by sequence（按序列切）。这正好对应幻灯片 Summary 第一条 `Many ways to parallelize: data (batch), tensor/expert (width), pipeline (depth), sequence (length)`（kf_0060 第 68 行）。第二条 `Data parallelism: DDP (all-reduce), FSDP/ZeRO (all-gather + reduce-scatter)`（第 68 行附近）把两种数据并行并列：本讲只做了 DDP，FSDP 与 ZeRO 留到「next time」——这与 62:31 的预告、以及 62:56 那张幻灯片底部黄色高亮的「Next time: FSDP/ZeRO」形成首尾呼应，说明整段其实是被「all-reduce 太整块、模型放不下怎么办」这条线索串起来的。顺带把三条切法再固化一次：`data_parallelism()` 切 batch dimension、`tensor_parallelism()` 切 width dimension、`pipeline_parallelism()` 切 depth dimension。

![What's missing 中 JAX/TPU 编译器与 PyTorch 原语路线的对比](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0059_01-18-30.jpg)

结尾屏幕上还保留了完整的「What's missing?」清单（第 60–65 行）：Communication/computation overlap、More general models (with attention, etc.)、Other forms of parallelism (e.g., sequence parallelism, expert parallelism, combinations)、`Jax/TPUs: just define the model, the sharding strategy, and the Jax compiler handles the rest [levanter]`，以及被黄色高亮的第 64 行 `But we're doing PyTorch so you can see how one builds up from the primitives`。这一条把上一章的编译器对比收进了官方总结：本课的价值恰恰在于不隐藏原语，让你看见 `dist.all_gather`、`dist.all_reduce`、`send`/`recv` 这些集合操作与点对点操作是怎么被一步步搭成三种并行的。把这些「没讲完的」列出来，也解释了为什么 74:24 说重叠要留到作业二、75:40 说序列/专家并行会出现在作业里、80:50 说下周三由 Tatsuo 做 deep dive。

关于互连与气泡，Summary 第三、四条是 `Tensor parallelism: requires very fast interconnects (e.g., NVLink)` 和 `Pipeline parallelism: can work with slow interconnects, but need to work to reduce pipeline bubbles`；79:31–79:39 的口述版是「Tensor parallelism requires very fast interconnects; Pipeline, less so. But you need to really work hard to reduce these pipeline bubbles」。写成一行对照：

$$
\text{TP}: \text{带宽敏感、每层一次大激活通信} \qquad \text{PP}: \text{带宽宽容、但必须努力消除 bubble}
$$

再往后 79:44，讲者提炼出一个「反复出现的高层模式」（this kind of pattern come up a lot）：你既可以 recompute（重算），也可以 store in memory（存在内存里），而本讲可以看成它的扩展——store on a different GPU（存到另一张 GPU 上）。幻灯片第 71 行把它写成 `Can re-compute or store in memory or store in another GPUs memory and communicate`，其中 re-compute、memory、another GPUs memory、communicate 都被加粗。例子也很具体：activation checkpointing（激活重算）是「重算」那一端；而数据并行里每个 rank 都在冗余地更新并保存全部参数（doing redundant work），之所以这么做，是为了「不必把优化器状态搬来搬去」（you don't have to move the optimizer state across）。最后一条黄色高亮是第 72 行 `Hardware is getting faster, but will always want bigger models, so will have this hierarchical structure`——硬件会越来越快，但人们总想要更大的模型，所以这种分层结构会一直存在。画面底部还露出了下一讲的开头，可作为本段与后续内容的接口：

```text
def collective_operations():
Collective operations are the conceptual primitives used for distributed programming [article]
These are classic in the parallel programming literature from the 1980s.
```

![片尾 Summary 五条小结，含重算/存储/跨 GPU 通信与分层结构](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0060_01-20-37.jpg)

80:47 讲者宣布「that's it for today」，并说下周三 Tatsuo 会对更多并行技术做 deep dive，本段到 81:02 结束。易错点需要收个尾：别把「分层结构」（hierarchical structure）只理解成前面那张 L1 / HBM / NVLink / InfiniBand 的存储层级表——这里指的是更一般的「重算 / 本地存储 / 远端存储 + 通信」三选一权衡，存储层级只是它的一个实例；也别把数据并行的冗余简单斥为浪费，那是用冗余计算换取「不搬运优化器状态」的主动设计，与张量并行「用通信换显存」、流水线并行「用气泡换慢链路」处在同一个权衡坐标系里。整段从 DDP 的收尾问答出发，依次走完张量并行的列切与 all-gather、流水线并行的按层切分与微批次、硬件选型与编译器路线，最后落回这个贯穿全课的权衡，时间上一直覆盖到片尾总结。

## 要点回顾

### 从单卡到多卡：同一个数据搬运瓶颈

- 核心矛盾是算力（ALU、tensor core）离数据很远：单卡时「远」指 HBM，多卡时「远」可能是另一张 GPU 的显存，所以用很多 GPU 容易、用得有效很难，目标是编排计算、避免通信成为瓶颈。
- 广义层级从快到慢为 L1 cache / shared memory > HBM > NVLink / NVSwitch > InfiniBand / Ethernet；参照系一变，上周还嫌慢的 HBM 在多卡视角里反而算「快」的一层。
- 用多 GPU 的两个理由：放不下（B200 的 HBM 为 192 GB，1 trillion 参数模型连同优化器状态、梯度、激活远超单卡）与想更快（用更多 GPU 的 FLOPs 缩短训练），而铺开获得更多核心必须为 communication bandwidth 付账。
- 课程分 Part 1 的 collective_operations()、hardware()、torch_distributed()、benchmarking() 四个构件，与 Part 2 在 MLP（Transformer 的计算瓶颈）上的三种并行；直接跑 python lecture_07.py 走 multiprocessing，trace 时 python -m edtrace.execute -m lecture_07 关闭多进程。

### 集合通信原语与 all-reduce 分解

- 原语可追溯到 1980 年代，collective 指跨多设备的通用通信模板而非点对点；rank 是一个 device（本课通常一张 GPU，也可 TPU），world size 是设备总数，W = 4、rank ∈ {0,1,2,3} 从 0 编号。
- 八个操作分三组：broadcast、scatter、gather、reduce 是 warm-up，all-gather、reduce-scatter、all-reduce 是训练主力，all-to-all 主要用于 MoE；术语上 reduce 执行 sum/min/max 等结合律交换律运算，scatter 是 gather 的逆，all 表示目的地是所有设备。
- 基础语义：broadcast 从 rank 0 复制到所有 rank（多用于初始化 checkpoint），scatter 把张量切成 W 块，gather 拼接回完整张量，reduce 对分片归约得 0+1+2+3=6；可把 gather 看成「运算为拼接」的 reduce。
- reduce-scatter 按维度分别归约再散射，4×4 输入按列求和 s_d = Σ_r x_r[d] 得 (6, 10, 14, 18)，结果分散存到各 rank 以省显存；all-reduce = reduce-scatter + all-gather，每个 rank 拿到完整 [6, 10, 14, 18]，把这一步拆开正是 ZeRO/FSDP 的突破口。
- all-to-all 最通用，张量的「位置」代表目的地 rank；balanced splits 下等价于转置 Y = X^T，也能处理不均衡切分但应尽量均衡，这正是 MoE 把激活动态路由到各专家的通信骨架。

### 互连硬件、带宽标尺与绕过 CPU

- 经典家用机是 CPU 经 PCIe 挂 GPU、跨节点走 Ethernet；现代数据中心为每张 GPU 内部的 SM/L2/HBM 经 NVLink → NVSwitch，再下接 InfiniBand/Ethernet，单卡层级被原样保留并向下延伸一层。
- 典型三级组网：8 GPUs/node 经 NVLink 接 NVSwitch（NVLink 5.0 约 1.8 TB/s，B200 的 HBM 为 8 TB/s，约 1/4）；256 nodes/pod 经 InfiniBand（PCIe → HCA/NIC → 线缆，约 0.05 TB/s，比 NVLink 慢约 36 倍）；N pods 经 Ethernet（PCIe → CPU），其中 256 是讲师明说 "made up" 的示意值。
- 单调递减的带宽链为 L1/shared memory > HBM (8 TB/s) > NVLink 5.0 (1.8 TB/s) > InfiniBand > Ethernet，与单卡存储层级同构，都是规模越大、离算力越远、带宽越低。
- 传统 Ethernet 需经 CPU（kernel socket buffer → TCP 包 → NIC ring buffer，此处 kernel 是 CPU 操作系统内核而非 GPU kernel）；RDMA 让一张 GPU 直接读写另一张 GPU 显存、不经过 CPU，NVLink/NVSwitch 与 InfiniBand 支持，标准 Ethernet 不支持。
- GB200/GB300 NVL72 以 8 GPUs/tray × 9 trays/rack = 72 GPUs 处于同一 NVLink 域（每 tray 2 CPU × 4 GPU = 8）；RoCE 让 Ethernet 也绕过 CPU，比 InfiniBand 更便宜/更弱、Meta 在用；RDMA 是操作层能力，与 NVLink/InfiniBand 等硬件线缆分属不同层次。

### NCCL、torch.distributed 与有效带宽

- NCCL 把集合操作翻译成 GPU 间低层数据包，三项职责是探测拓扑、优化 GPU 间路径、启动通信 kernel（凡跑在 GPU 上的都是 kernel）；torch.distributed 提供干净接口，后端为 gloo(CPU)/nccl(GPU)，FSDP 等高层算法本课从零构建不用。
- spawn(collective_operations_main, world_size=4) 复制运行四次，world size 是进程数、rank ∈ {0,…,W-1}；setup(rank, world_size) 的 master address/port 只用于元数据与进程协调，真正数据走 NCCL；dist.barrier() 让所有进程对齐，进程异步运行使 print 顺序由硬件决定。
- 代码实证：dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False) 对 data = [0., 1, 2, 3] + rank 原地写回；dist.reduce_scatter_tensor(output, input, op=SUM) 用 torch.empty(1) 独立输出；dist.all_gather_into_tensor 以 reduce-scatter 的输出为输入，用例子证明 all-reduce = reduce-scatter + all-gather。
- 基准用 spawn(all_reduce, world_size=4, num_elements=100 * 1024**2 = 104857600)，先 warmup 再计时；计时前后须先 torch.cuda.synchronize() 再 dist.barrier()，因为并存 CUDA kernel 相对 host、进程之间两种异步，async_op=False 只保证 host 侧返回。
- 有效带宽 bandwidth = sent_bytes / total_duration，其中 sent_bytes = size_bytes * 2 * (W-1)（2x 因 send+receive，W-1 步）、total_duration = W * duration，故 bandwidth = 2·size_bytes·(W-1)/(W·T) ≈ 2·size_bytes/T；实测约 400 GB/s（390/366/425 GB/s），与 world size 和拓扑（ring/tree）无关，reduce-scatter 去掉系数 2 后同一量级（475/450/490 GB/s）。

### 三种并行策略的骨架实现

- 数据并行沿 batch 维度切：generate_sample_data() 给 batch_size = 128、num_dim = 1024，local_batch_size = int_divide(batch_size, world_size) = 32，靠 start_index = rank * local_batch_size 切片；每卡持全部 params（num_layers = 4 个 1024×1024 矩阵）与 torch.optim.AdamW(params, lr=1e-3) 状态。
- DDP 循环只在 loss.backward() 与 optimizer.step() 之间多一句 dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)，直接平均得 ḡ = (1/W)Σ g_r；各 rank 的 loss 因局部数据而不同，但梯度、参数保持一致，被形容为「一行代码改动」，且对 Transformer 因模块化同样适用。
- 张量并行沿 width 切、不切数据：local_num_dim = int_divide(num_dim, world_size)，params_ℓ ∈ R^(num_dim × local_num_dim)，每卡只持 1/world_size 参数；前向 x = x @ params[layer] 与逐元素 F.gelu(x) 后，用 dist.all_gather(tensor_list=activations, tensor=x) 再 torch.cat(activations, dim=1) 拼回 batch_size × num_dim，每层通信一次。
- 张量并行前向 all-gather、反向即 reduce-scatter 呈对偶；autograd 不会自动插入跨 rank 通信（计算图里没有并行），反向留作 homework。数据并行几乎不动模型，张量并行必须重写每层前向，侵入性更强。
- 流水线并行沿 depth 切：每卡拿一个层子集、层内仍是 num_dim × num_dim 完整方阵，数据经 micro_batches 依次流过，循环里用 recv(from=rank-1) / send(x, to=rank+1) 点对点通信，是唯一不用集合操作的并行；micro-batch 减少（而非消除）pipeline bubbles，isend/irecv 让通信与计算重叠，T_overlap ≈ max(T_compute, T_comm)。

### 选型、未覆盖内容与片尾权衡

- 选型强依赖硬件：张量并行每层都发大激活、通信量大，一般放在节点内 NVLink/NVSwitch 域；流水线并行只在 stage 边界传一次、能容忍慢互连。常见组合是节点内 TP + 跨节点 DP/FSDP + 必要时 PP，数据并行还会撞上 critical batch size（继续增大 batch 收益递减）。
- What's missing 列出：Communication/computation overlap、More general models (with attention, etc.)、sequence parallelism（切 sequence 并行化 attention）、expert parallelism（切 MoE experts，正是 all-to-all 的用武之地）与各种组合，多留给作业二；DDP 的 all-reduce 也可与 backward 重叠。
- 两条路线对比：本课刻意用 PyTorch 的集合操作原语（dist.all_gather、dist.all_reduce、send/recv）让人机械看到每一步；JAX/TPU 只需定义模型与 sharding strategy、由编译器推导通信，但会拿走从零构建的乐趣。
- 片尾 Summary：Many ways to parallelize 为 data (batch)、tensor/expert (width)、pipeline (depth)、sequence (length)；DDP 用 all-reduce，FSDP/ZeRO 用 all-gather + reduce-scatter 并留待下节课 Tatsu 深讲，张量并行需极快互连、流水线并行可用慢互连但需努力消除气泡。
- 反复出现的模式是 Can re-compute or store in memory or store in another GPUs memory and communicate：activation checkpointing 是重算，DDP 冗余保存全部参数与优化器状态是换「不搬运优化器状态」；片尾以 Hardware is getting faster, but will always want bigger models, so will have this hierarchical structure 收束。

## 视频来源

- 原始链接：https://www.youtube.com/watch?v=SzpOcwdIL0Y
- 课程：Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 7

---

## 对照参考笔记补遗（Spring 2025 版）

> 编者注：本节依据 2025 版参考笔记整理，其中的「更正」小节记录的是核对时发现的问题，对应正文已在本版订正；补遗内容不代表 2026 课程视频原文。


本节内容来自 Spring 2025 版参考笔记，用于对照补充本讲主题，不代表 2026 视频原文；其中公式与数字请以「2025 版解析口径」理解，凡 MinerU 解析可能有歧义处已随文标注。

### 一、ZeRO 与 FSDP：拆开 all-reduce 之后的显存账

本讲正文已经用代码和张量例子证明了 $\text{all-reduce}=\text{reduce-scatter}+\text{all-gather}$，并反复说「要做 FSDP/ZeRO 就得把 all-reduce 拆开，才能在中间介入管理存储」。2025 版恰好把这一步走到底，给出了拆开之后能省多少显存的完整账本，正好把正文的恒等式闭环。起点是 naive Data Parallelism 的显存不扩展：在混合精度下，一个参数大约要占 **16 字节、约等于五份权重副本**，具体是 2 字节 BF16/FP16 参数、2 字节 BF16 梯度、4 字节 FP32 master weight（SGD 累加所用的主权重），再加上 Adam 一阶矩与二阶矩各 4（或 2）字节。把它写成显存式子，朴素 DDP 与 ZeRO Stage 1 的对照是

$$(4+K)\cdot\Psi \quad\longrightarrow\quad \Big(4+\frac{K}{N_{gpu}}\Big)\cdot\Psi$$

其中 $\Psi$ 是参数总量、$K$ 代表优化器状态系数、$N_{gpu}$ 为设备数，而通信量只有一次 all-reduce 的 $2\cdot\#params$；瓶颈几乎全在显存，优化器状态又是其中最大头。

ZeRO（Zero Redundancy Optimizer）的核心思想就是「把昂贵的状态切片，再用 reduce-scatter 等价性」。它分三级：**Stage 1（$P_{os}$）** 只切分优化器状态，每个 worker 仍持有完整参数与梯度，只负责更新自己那一片参数，显存降到 $(4+K/N_{gpu})\cdot\Psi$；**Stage 2（$P_{os+g}$）** 再把梯度也分片；**Stage 3（$P_{os+g+p}$，即 FSDP / Fully Sharded Data Parallel）** 连参数一起分片，按层在前向时 all-gather 需要的权重、算完立刻释放。2025 版给的可核对数字是：一个 7.5B 参数模型放在 64 张加速卡上，朴素 DDP 基线约 120 GB，Stage 1 降到约 31.4 GB，Stage 2 约 16.6 GB，Stage 3 约 1.9 GB。对应到 8×A100 80GB 单节点，可容纳的模型上限依次约为 6.66B、16B、24.62B、53.33B（按每参数字节数 12、5、$2+10/8$、$12/8$ 估算）。要注意参考里各级显存公式存在 OCR 自相矛盾：Stage 1 一处写 $(2+2+K)\frac{\Psi}{N_d}$、另一处表格写 $(4+K/N_{gpu})\cdot\#params$，Stage 2 与 Stage 3 甚至共用同一公式却给出不同 GB 值，因此以表格与上述 GB 数值为准、把公式下标理解为「带解析歧义」更稳妥。

最值得记的易错点是通信成本：DDP 与 Stage 1 都是 $2\cdot\#params$（前者一次 all-reduce，后者 reduce-scatter 梯度 + all-gather 参数），所以 **「ZeRO Stage 1 is free in the bandwidth-limited regime」**——通信一分没多、显存却省下大半，几乎总该做；Stage 2 仍是 $2\cdot\#params$，称为 almost free；Stage 3 因为前向、反向各要一次 all-gather 权重、再加一次 reduce-scatter 梯度，升到 $3\cdot\#params$，即 1.5 倍通信，但靠 overlap 把延迟藏起来后并不算贵。Stage 2/3 能成立的关键是 incremental computation/communication：某层梯度一算完就立刻 reduce-scatter 到拥有该分片的 GPU 并释放，权重则在真正用到之前预先排队请求，让 all-gather 与前向计算重叠。这正好呼应正文「拆开不是为少传数据、而是为在中间管理存储」的结论，也解释了正文片尾为什么把 FSDP/ZeRO 留到后续讲——它本就是恒等式的下一章。

### 二、通信成本标尺与 Model Parallelism 的边界

正文按语义逐个讲过八个集合原语，并用代码算过 `sent_bytes`；2025 版补的是一把更便于心算的「成本标尺」：`all_reduce` 约为 $2\times\text{size\_of\_data}$（各 rank 输入求和后复制到所有 rank），`broadcast` 约为 $1\times$（一个 root 复制到所有），`reduce` 约为 $1\times$（只送回 root），`all_gather` 是拼接后复制到所有 rank，`reduce_scatter` 是各 rank 拿走一段归约结果。与之配套的是一句纲领：把整个数据中心（datacenter）而非单张 GPU 当作新的计算单元，追求 linear memory scaling 与 linear compute scaling，于是性能分析常常可以归约为「数一数一共做了几次通信操作」。这为正文第一部分那些箭头补上了量纲。

由此还能得到正文恒等式的一层理论注脚：在 bandwidth-limited regime（带宽受限区间）里，`reduce_scatter + all_gather` 两步与直接 `all_reduce` 的通信成本相同，而且往往是能达到的最优做法。换句话说，正文用例子证明的「拆开后带宽仍是同一量级」，在 2025 版被提升为一条性能上界判断；两者说的是同一件事，只是粒度不同。真正要区分清楚的是 Model Parallelism 的边界：它 splits parameters across GPUs（像 ZeRO3 那样把参数切开），但它 communicates **activations**（传递激活），而 ZeRO3 传的是 **parameters**（传递参数）——一个动数据、一个动权重，这是把 FSDP 与 PP/TP 分开的关键。2025 版把它归为两类（Pipeline Parallelism 与 Tensor Parallelism），并用一张 Recap 表收束：DDP/ZeRO1 是 Per-batch 同步、显存 No scaling、带宽 $2\cdot\#param$、Batch Size 影响为 Linear、Very 易用；FSDP(ZeRO3) 是 Per-batch、Linear、$3\cdot\#param$、Linear、Very；Pipeline 是 Per-pipeline、Linear、按 Activations 计、No Impact、NO；Tensor+Seq 是 Per-transformer block、Linear、$8\cdot$activations/layer + all_reduce、No Impact、YES。一句话总纲是：必须同时权衡 memory、bandwidth、batch size 这三种有限资源，没有任何单一策略能通吃。

### 三、流水线气泡的定量刻画与 Zero-Bubble

正文已经讲过 micro-batch 能减少（reduce 而非 eliminate）流水线气泡，并用 `isend/irecv` 把通信与计算重叠、写成 $T_{overlap}\approx\max(T_{compute},T_{comm})$。2025 版给的是可直接背诵的比例：朴素的 layer-wise parallelism 把第 0 层放 GPU0、第 1 层放 GPU1……每个 GPU 只有 $1/N$ 的时间在算，气泡来自前向要沿流水线逐级传播、反向再逐级传回的空转。把 batch 切成 $N_{microbatches}$ 个微批次后，气泡时间占有效计算的比例是

$$\frac{N_{stages}-1}{N_{microbatches}}$$

微批次越多（意味着 batch 越大）气泡越能被藏住，因此 **batch size 是一种可以花费的资源**，而不只是超参数。实测观察也支持这一点：小 batch（如 8）时，流水线并行度越大每 GPU 吞吐掉得越快；大 batch（如 128）时，吞吐随并行度增大基本持平。

这还能把正文「流水线能容忍慢互连」的定性判断量化：PP 每个微批次只在 stage 边界传一次 $\text{batch\_size}\times\text{sequence\_length}\times\text{hidden\_dim}$（记作 $bsh$）的激活，而且是 point-to-point（点对点），所以适合 inter-node、跨 rack 甚至跨数据中心的慢链路；相对地，张量并行没有 bubble、复杂度低（可直接 wrap 模型）、不需要大 batch，代价却是通信量大得多。进阶的 **Zero-Bubble（Dualpipe）** 试图把气泡彻底挤掉：把反向拆成「回传激活梯度 $\partial L/\partial z$、$\partial L/\partial x$」与「计算权重梯度 $\partial L/\partial W$」两半，后者没有跨层串行依赖，可以被重新调度进空闲气泡里；代价是要深度介入 autodiff 系统、调度极其复杂。这里的易错点仍是正文强调过的那条——微批次是 reduce 而非 eliminate 气泡；此外 PP 也省显存，因为每张卡只需存自己那几层的参数与激活，这与它「省显存 + 点对点通信」的双重好处是正文中「用气泡换慢链路」的定量版本。

### 四、张量并行的通信代价、激活显存与序列并行

正文在深 MLP 上做的是「列并行 + all-gather 拼接」这一种切法，2025 版则展示了同一思想的另一半：矩阵乘可拆，对 $X\times A=Y$，按列拆 $A=[A_1,A_2]$、按行拆 $X=[X_1;X_2]$ 得到 $Y=X_1A_1+X_2A_2$（需要一次 `all_reduce` 累加部分和），或按行拆 $A$、按列拆 $X$ 得到 $Y=XA_1+XA_2$；Transformer 层里把 A 的列与 B 的行分给不同 GPU，输入复制到各卡，前向与反向各做一次 `all_reduce`。通信量的对照很具体：Pipeline 每微批次是 $bsh$ 的点对点，Tensor 则是每层 $8\,bsh\,\frac{N_{devices}-1}{N_{devices}}$ 再加 all_reduce，量级远大于 PP。HuggingFace 的实测是 TP=2/4 时每 GPU 吞吐仅降约 10–12%，但 TP=16 降 42.7%、TP=32 降 65.6%，所以经验上把 TP 封顶在单节点 8 卡，只在有低延迟高带宽互连时使用——这正是正文「TP 需 NVLink」的数字来源。

更难的一层是激活显存（activation memory）：它不像参数与优化器状态那样静态，而是在前向增长、反向释放，梯度在反向累积，峰值常落在反向中段；即便把参数与优化器状态激进分片，激活仍随模型与序列长度增长，LayerNorm、Dropout、MLP 输入这些 pointwise 操作不随 TP/PP 线性缩小。若把中间量全部存下，单层激活显存约为 $\text{sbh}\big(34+\dfrac{5as}{h}\big)$，其中 $s$ 为序列长、$b$ 为 microbatch 大小、$h$ 为 hidden dimension、$a$ 为 attention 头数；$5as/h$ 来自含 dropout 的二次 attention 项，FlashAttention 可用重算把它去掉，$34$ 来自 MLP 与 pointwise 操作。施加张量并行（拆 attention 与 MLP 里的矩阵乘）后降为 $\text{sbh}\big(10+\dfrac{24}{t}+\dfrac{5}{ht}\big)$（$t$ 为 TP 设备数），其中剩下的 $10$ 项 = LayerNorm $4sbh$ + Dropout $2sbh$ + attention/MLP 输入 $4sbh$，**不被 $t$ 除**，这正是「TP 不能线性降全部激活」的根源。要突破它，需要 **Sequence Parallelism（SP，序列并行）**：沿序列轴切分这些 pointwise 项，前向用 `all_gather` 合并、反向用 `reduce_scatter` 归并——与正文「TP 前向 all-gather、反向 reduce-scatter 呈对偶」是同一套逻辑；再把 TP、SP 与选择性激活重算（FlashAttention）合起来，单层激活可压到 $\text{sbh}\cdot\dfrac{34}{t}$，实现真正的线性缩放。整条压缩链可写成

$$\text{sbh}\Big(34+\frac{5as}{h}\Big)\ \xrightarrow{\ \text{TP}\ }\ \text{sbh}\Big(10+\frac{24}{t}+\frac{5}{ht}\Big)\ \xrightarrow{\ \text{TP+SP+重算}\ }\ \text{sbh}\cdot\frac{34}{t}$$

这里的易错点有两处：一是别误以为 TP 能把所有激活按 $t$ 除尽，pointwise 的 $10sbh$ 是硬残留；二是参考里这组公式有 OCR 歧义（$5as/h$ 被拆成 $as/h+5/h$），应理解为 2025 版解析口径、下标可能有偏差，不要当成 2026 视频逐字结论。

### 五、3D 并行经验法则与规模化代价

把前面的原语、显存账与通信量合起来，就得到 2025 版反复强调的经验法则：在模型能塞进显存之前，先用 Tensor Parallel 用到单机卡数（如 8），再用 Pipeline Parallel 跨机（或按带宽改用 ZeRO-3），剩余 GPU 交给 Data Parallel 扩张，batch 偏小时用 gradient accumulation（梯度累积）拿 batch 换通信效率。全局 batch size 是关键量：**小于约 400 时没有单一并行策略高效、整体处于 communication bound**；随 batch 增大，最优解依次是小 batch 用纯 FSDP、中 batch 用 FSDP + TP、大 batch 两者或纯 FSDP 都行、超大 batch 纯 FSDP 就够。这把正文片尾那句「data parallel 会撞上 critical batch size、继续增大 batch 收益递减」从定性补成了可操作的分区。

经典证据是 **Narayanan 2021（Megatron-LM）**：用 TP/PP/DP 组合训练 1.7B 到 1T 参数，TP 从 1 升到 8 后封顶、PP 在模型放不下时递增、DP 初始尽量大（如 32）再随 PP 增大递减，最终达到理论峰值 **40–52% FLOPs**，且 TP=8 常是每 GPU 吞吐最优点；精心设计的 3D 并行给出平坦的 achieved TFLOPs/GPU，也就是总 FLOPs 随 GPU 数线性增长。还有一个反直觉结论：**activation recomputation（激活重算）能自己回本**——它虽然多花 FLOPs，却省下激活显存、允许更大 batch，进而通过隐藏气泡或摊薄通信提高吞吐，这与正文片尾「re-compute / store in memory / store in another GPU's memory and communicate」的三选一权衡完全同构。真实系统配置可作佐证：Dolma 7B 主要用 FSDP；DeepSeek-V3 用 ZeRO1 + TP + SP + PP（16-way PP、64-way Expert Parallel、8-way TP）；Yi 34B 用 ZeRO1 + TP + PP，MoE 场景以「E-lightning」用专家并行替代 TP；Llama 3 405B 是 TP=8、CP=1（Context Parallel，长上下文）、PP=16、DP=128；Gemma 2 在 TPUv4/v5p 上用 ZeRO3 + MP + DP、512-way data replication。最后别忘了规模化的代价：Llama 3 405B 报告中训练中断的原因为故障 GPU 30.1%、故障 HBM3 17.2%、软件 bug 12.9%、网络交换/线缆 8.4%、计划外主机维护 7.6%，于是容错与稳健 checkpointing 成为必需，而 silent data corruption（静默数据损坏）比显式报错更可怕。最大的易错点是版本归属：这些 2025 版的论文名、模型配置与故障统计是「对照补充」，不能写成 2026 本讲视频里讲过的内容；正文里真正对应的是那句「硬件越来越快、人总想要更大模型，于是分层结构会一直存在」。
