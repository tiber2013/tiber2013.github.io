---
title: "第 6 讲 · Kernel、Triton 与 XLA"
description: "本讲从 GPU 硬件约束出发，说明为什么“能写对”和“跑得快”是两件事：编程模型只描述线程、线程块和网格，而性能由 warp、寄存器占用、共享内存 bank 冲突、内存合并和 block 占用率等硬件细节决定。作者强调先…"
---

# 第 6 讲 · Kernel、Triton 与 XLA

> **Stanford CS336 语言模型从零构建 · L06** ｜ 讲者 Stanford Online ｜ 87 分钟 ｜ [原视频](https://www.youtube.com/watch?v=xnDHaNUvHBg)

> **v2 详写版** · 按 20 分钟分段详写，配图覆盖全片；文末附「对照参考笔记补遗（Spring 2025 版）」，记录与 2025 版讲义笔记的核对结果。

## 核心结论

本讲把“能写对”与“跑得快”拆成两件事：编程模型只给出线程、线程块与网格三层抽象，真正决定速度的却是 warp 锁步、寄存器占用率、共享内存 bank 冲突、HBM 合并访问与线程块占用率这些硬件细节，因此必须先 benchmark 与 profile 找出瓶颈，再动手写 kernel。作者用 GELU 作样本：朴素 PyTorch 把计算图里每个算子都变成一次 HBM 往返，内置融合 kernel 与 torch.compile 生成的 Triton kernel 则每个元素只读一次、写一次，在 dim=16384 上把耗时从 3.7583 降到 0.6670 与 0.9388。随后以线程块为单位，用 Triton 从 element-wise GELU、行内 softmax（一行一块、mask 填 -inf）、一行放不进块时的 tile 循环 row sum，一路写到 matmul + ReLU：靠二维 tiling 把算术强度从 O(1) 抬到 O(tile_size)，并在写回 HBM 前顺手融合激活。整条主线都是“把数据读入片上、计算、再写回 HBM”，Triton 的价值在于省去显式线程同步与 shared memory 管理；但极限性能仍受 SM 数、bank 数、寄存器容量等有限硬件约束，可下探 PTX 或换用 ThunderKittens、CUTE 等 DSL，而本讲至此只覆盖单 GPU。

## 课程定位与 GPU 存储层级回顾 [00:00 - 03:18]

开场先做定位。讲者欢迎回来，并提醒周一 Tatsu 已经从高层很好地讲过 GPU、如何思考性能、以及 GPU 自带的各种 quirks（怪癖）；本讲是那次内容的延续，要更深入地钻进代码里：编写 Triton kernel（内核），以及做 benchmarking（基准测试）与 profiling（性能剖析）。为了让后面的讨论有共同的心智模型，讲者先"刷新"一张简化图：左边是实际的 GPU 芯片，右边是独立的内存。虽然本次聚焦 NVIDIA GPU——转写里也顺带提到 Tatsu 上次讲过 TPU，以及 AMD 等——但真正要记住的是存储与带宽的层级结构。从 A100、H100 到 B200 的多代产品里，GPU 都由若干 streaming multiprocessor（SM，流式多处理器）构成，SM 数量大约在 100 到 200 之间，讲者强调"这个数字其实没怎么变"；幻灯片规格表给出的确切值是 A100 为 108、H100 为 132、B200 为 148 个 SM。

再往里看层级。每个 SM 内有一组 register（寄存器），B200 有 65,000 个寄存器、合计每 SM 256 KB，这个量级同样"没怎么变"；此外还有 L1 cache 与 shared memory（共享内存），讲者特别指出它们其实是同一块物理内存，区别只在于 shared memory 可由程序员控制、L1 不能，而且都是 per SM（每个 SM 一份），容量处于同一数量级。再往外是 L2 cache，它不属于单个 SM、而是整块芯片共享，容量更大一些。最外层是 high-bandwidth memory（HBM，高带宽内存），容量最大，而且讲者说"这个数字实际上涨得相当多"，是逐代变化最明显的一项。除了容量还要看 bandwidth（带宽），二者大致成反比：寄存器最快，L1 略慢，L2 更慢，HBM 最慢——不过 B200 的 8 TB/s 放在宏观尺度上"也并不算慢"。用一句话概括整个环境：大而慢的存储在远处、但容量大；快而小的存储（寄存器与 L1）位于 SM 上、是局部的、快、但小。

![GPU 存储层级示意图：紫色 GPU 芯片内排布多个含 Reg 与 L1+shmem 的蓝色 SM 方框，下方为 L2，右侧经绿色总线连接大块 HBM](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0001_00-00-40.jpg)

![A100、H100、B200 三代加速器规格表：SM 数、寄存器、L1+shared memory、L2、HBM 容量与带宽逐代对比](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0002_00-02-01.jpg)

幻灯片规格表的原文（黄色高亮行是 `L1 cache + shared memory (per SM)`，底部还有一行关于 TMEM 的注释）逐行转写如下：

```text
Accelerator                        | A100      | H100      | B200
# SMs                              | 108       | 132       | 148
Register size (per SM)             | 256 KB    | 256 KB    | 256 KB
L1 cache + shared memory (per SM)  | 192 KB    | 256 KB    | 256 KB
L2 cache size                      | 40 MB     | 50 MB     | 96-126 MB
HBM size                           | 80 GB     | 80 GB     | 192 GB
Register bandwidth                 | ~116 TB/s | ~401 TB/s | ~447 TB/s
L1 cache + shared memory bandwidth | ~19 TB/s  | ~33 TB/s  |
L2 cache bandwidth                 | ~5-8 TB/s | ~12 TB/s  | ~9 TB/s
HBM bandwidth                      | 2 TB/s    | 3.35 TB/s | 8 TB/s
(B200s also have tensor memory (TMEM) for tensor cores (between registers and shared memory), invisible to programmer.)
```

把带宽关系写成不等式就是 $$\text{BW}_{\text{register}} \;>\; \text{BW}_{L1/\text{shared}} \;>\; \text{BW}_{L2} \;>\; \text{BW}_{HBM}, \qquad \text{而容量大小恰好反过来}$$。这一"远慢近快、近快却小"的直觉是后面所有优化的根基：等到讲 kernel 时反复出现的"从 HBM 读入片上、算完写回"，正是为了绕开 HBM 的相对昂贵；可以说本段后面每一个硬件细节，都是在解释为什么不能让数据频繁往返 HBM。

这里有几个易错点。第一，不要把 L1 与 shared memory 当成两块独立的存储，幻灯片把它们写在同一行 `L1 cache + shared memory (per SM)`，就是在强调物理上是一块、只是控制权不同。第二，别被 8 TB/s 这个"大数字"迷惑：与寄存器带宽 ~447 TB/s 相比，HBM 仍慢了约两个数量级，所以"慢"始终是相对片上存储而言。第三，讲者说"没怎么变"的是 SM 数量与寄存器文件（256 KB 三代数保持一致），而 HBM 容量从 80 GB 涨到 192 GB、带宽从 2 TB/s 涨到 8 TB/s 才是逐代大涨的列；表格里 B200 的 L2 写成 96–126 MB 是一个区间而非定值。第四，幻灯片底部专门注明 B200 还新增了 tensor memory（TMEM，张量内存），位于寄存器与 shared memory 之间、服务于 tensor cores，且"对程序员不可见"——这类"硬件里有、模型里没有"的机制，正好预告了本段后半部分"编程模型与硬件并不一一对应"的主题。

## 编程模型：线程、线程块与 Grid [03:18 - 05:05]

讲者接着问"如何给 GPU 编程"，给出 programming model（编程模型）的三层概念。thread（线程）在数据的一小部分上执行一段代码，图中每一根箭头都可以看作一个线程；这些线程被组织成 thread block（线程块），也叫 concurrent thread array（CTA，并发线程数组），即一组线程；thread block 的集合再构成 grid（网格）。启动一个 kernel，本质上就是启动一个由线程块和线程组成的 grid，让它们全部并行地同时做计算。幻灯片 `Programming model` 用一块绿色大方框标成 `Grid`，里面排成 2 行 4 列共 8 个白色 CTA 小方框，每个 CTA 里画了约 9 条绿色向下箭头代表线程——这就是"grid 装 CTA、CTA 装 thread"的三层包含关系，用集合可写作 $$\text{Grid} = \{\text{CTA}_1, \text{CTA}_2, \ldots, \text{CTA}_8\}, \qquad \text{CTA} = \{t_1, t_2, \ldots, t_n\}$$。

![GPU 编程模型示意图：绿色 Grid 大方框内排布 2 行 4 列共 8 个 CTA，每个 CTA 内含若干线程箭头](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0003_00-03-23.jpg)

幻灯片下方的三条定义原文逐字如下，它们把三层概念的职责说得很清楚：

```text
Thread: executes code on a small part of the data
Thread block or concurrent thread array (CTA): a group of threads
Grid: collection of thread blocks
```

讲者随即说明这幅图仍是"简化"的：真实硬件里还发生着别的事。H100 与 B200 还有 thread block clusters（线程块簇），即若干线程块组成的簇，能支持一定程度的 distributed shared memory（分布式共享内存）；B200 又新增了 tensor memory（TMEM），位置介于寄存器与 shared memory 之间、服务于 tensor cores。幻灯片上的原句是 `(H100s and B200s also have thread block clusters that enable distributed shared memory.)`，以及 `(B200s also have tensor memory (TMEM) for tensor cores (between registers and shared memory), invisible to programmer.)`。讲者点明这些机制有的对程序员不可见，但确实存在于硬件里，"不过这门导论课先不为它们操心"——这与上一章结尾 TMEM 的注释互相呼应，都在提醒：优雅的三层抽象并没有覆盖全部硬件。

紧接着幻灯片抛出下一节的问题 `Why thread blocks?`（为什么要线程块？），并先给出半个答案：对 elementwise（逐元素）操作，例如激活函数 GELU，线程模型非常自然，每个线程处理一个元素，对应幻灯片上的 `f(i) for i = 0, ..., N-1`。这也解释了上一章的三层抽象为什么在最简单情形下够用——只要线程之间不需要打交道，一个"由 N 个线程组成的 grid"就能把整个向量算完，写成代码就是下面这行原文所表达的形式：

```text
For elementwise operations (e.g., GELU), threads are most natural: each thread processes one element.
f(i) for i = 0, ..., N-1
```

但"为什么还需要线程块"这个提问本身，已经说明三层抽象不是故事的全部。与上一章衔接起来看：上一章给出的是硬件能提供的存储资源，这一章给出的是程序员能用的抽象；而下一章要论证的正是"抽象里最不起眼的线程块这一层，恰恰是唯一对应片上 shared memory 的一层"。易错点有三处：不要把 CTA 与 grid 混为一谈（CTA 是线程的集合，grid 是 CTA 的集合）；不要误以为 L1/shared memory 是全芯片共享的，上一章规格表写明 `L1 cache + shared memory (per SM)` 是 per SM，而 cluster 的 distributed shared memory 正是这一限制在 H100/B200 上的例外，本讲刻意忽略；也不要把 8 个 CTA 方框当成硬件里 SM 的个数——SM 数是 108/132/148，图里的 2×4=8 只是示意。数字上请记住 100–200 个 SM、B200 的 148、每 SM 256 KB 寄存器文件，以及图中 2 行 4 列共 8 个 CTA、每个 CTA 约 9 条线程箭头这些具体画面信息。

## 为什么要有线程块：逐元素操作与线程间通信 [05:05 - 07:13]

讲者正面回答上一章留下的提问。如果做的全是 element-wise 操作，"一个 grid 的线程、每个线程取一小块数据各算各的"完全够用；但 softmax、matrix multiplication（矩阵乘法）这类操作需要线程之间通信（communicate）。理论上仍然可以只用线程：让每个线程对应输出矩阵的一个 cell（元素），自己去 HBM 把所需数据读进来、算完再写回 HBM。转写里明确说"如果你愿意付出读写 HBM 的代价，这也够用"，但正如第一遍强调过的，HBM 很慢，所以这不是一个好策略。换句话说，"能写对"和"跑得快"在这里第一次分岔：朴素做法的正确性没问题，代价全在带宽上。

正确做法是利用 SM 本地的 shared memory。线程块恰好提供了这个语义：它是"一组都会访问同一块 shared memory 的线程"，幻灯片把这一行高亮成 `Thread block: a collection of threads that access the same shared memory`。于是顺理成章地 `Consequently, a thread block is scheduled on one SM`（一个线程块被调度到一个 SM 上）。一个线程块完整的工作循环是：从 HBM 读入一批数据 → 在片上处理，处理过程中线程之间借 shared memory 通信 → 把结果写回 HBM。把这条数据流写成公式，就是本段后面所有 Triton kernel 的骨架：

$$\text{HBM} \xrightarrow{\text{load}} \underbrace{\text{shared memory}}_{\text{one SM / one CTA}} \xrightarrow{\text{线程间通信}} \text{threads} \xrightarrow{\text{compute + store}} \text{HBM}$$

![Why thread blocks 幻灯片：解释逐元素操作用线程、非逐元素操作需 shared memory 通信，以及线程块被调度到单个 SM](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0005_00-06-05.jpg)

幻灯片上这一整段论证的原文（含讲者提到的 `In Triton, think natively in terms of thread blocks (later)`）逐行转写如下，其中第 71 行是黄色高亮的定义句：

```text
Why thread blocks?
For elementwise operations (e.g., GELU), threads are most natural: each thread processes one element.
  f(i) for i = 0, ..., N-1
However, for non-elementwise operations like softmax or matrix multiplication, threads need to communicate.
Reading/writing from HBM is slow, so use shared memory (local to SM).
Thread block: a collection of threads that access the same shared memory.
Consequently, a thread block is scheduled on one SM.
In Triton, think natively in terms of thread blocks (later).
(H100s and B200s also have thread block clusters that enable distributed shared memory.)
```

讲者在这里埋了一句很重的话：`Later, we'll talk about tiling and that's the whole kind of game here`（后面要讲的 tiling/分块，就是整个游戏的全部）。Triton 作为如今写 kernel 的主要方式，要求人原生地以线程块为单位思考；讲者说一旦习惯了以线程块思考，"这会让事情合理得多，也让你的日子好过很多"。这句话把上一章"三层抽象"与后面整门课的 kernel 写法连了起来：编程模型里真正要动脑的不是单个线程，而是线程块这一层。

数字上把理由坐实：shared memory 与 L1 是同一块，每 SM 为 192 KB（A100）/ 256 KB（H100、B200），而 HBM 是 80 GB / 80 GB / 192 GB，带宽只有 2 / 3.35 / 8 TB/s，对比寄存器带宽 ~447 TB/s——数量级差距正是"必须把数据搬上片再算"的原因。易错点也集中在这条边界上：不要以为把线程数开到最大就能解决通信问题，线程间通信只能发生在同一 SM 的 shared memory 上，因此线程块必须"整块"待在一个 SM 里，不能摊开；这正是本段末尾答疑里"一个块能不能跨 SM"问题的伏笔。另一处易错是 shared memory 由程序员控制、L1 不能，二者物理上却是同一块，别把它们当成两种可分别分配的存储。此外，只关心正确性时确实可以忽略这一切——但本讲的目的恰恰是榨性能。幻灯片底部已经灰显出下一节标题 `Interaction between programming model and hardware`（编程模型与硬件的交互），以及它的第一行 `Programming model provides an abstraction of the hardware`，预告讲者马上要指出：这套优雅模型并不足以解释速度。

## 编程模型与硬件的张力：warp 的引入 [07:13 - 10:15]

讲者转入本段主线。他说编程模型"其实非常简洁"，它对硬件做了抽象：写 kernel 时你只需知道有一堆线程块、定义它们、再定义块内每个线程或线程块要做什么计算，"就像写 Python 一样"，这部分并不难；而且如果只关心正确性，知道这些就够了。但紧接着是一记转折：`In practice, performance is very sensitive to the hardware`（实践中性能对硬件极其敏感），要获得高性能就必须深入理解硬件。讲者甚至把动机挑明：整门课之所以要谈 GPU 和 kernel，就是为了 squeeze out performance（把性能榨出来）。于是形成两个层次——计算本身属于编程模型，而"跑多快"强依赖硬件；只写对代码的人停在第一层，追求速度的人必须下到第二层。幻灯片 `Interaction between programming model and hardware` 的三行原文是：

```text
Programming model provides an abstraction of the hardware.
In principle, don't need to think about anything else (for correctness).
In practice, performance is very sensitive to the hardware, so need to understand it to obtain high performance.
Let's go over some considerations.
```

讲者口述接下来大约有 `around five different examples`（五个例子），只为给出"需要考虑哪些因素"的味道，其中一部分是 Tatsu 上次讲过的复习。这五个例子依次是 warp、warp occupancy、bank conflicts、memory coalescing 与 block occupancy，构成本段后半部分的结构。

第一个例子是 warp（线程束）。讲者强调 warp"并不真的属于"线程 / 线程块 / 网格这幅干净图景；技术上你也可以显式操纵 warps，但普通编程时并不必要。定义是：一个线程块里的线程被分组为 warp，每个 warp 有 32 个线程（`32 threads per warp`）。幻灯片给的例子是 `Example: thread block has 64 threads => it has 2 warps`，并用一行 ASCII 画出两段各 32 个 T 的线程：

```text
| TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT | TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT |
```

也就是 $64 = 2 \times 32$，前 32 个线程是第一个 warp、后 32 个是第二个 warp。关键约束随之而来：`All threads within a warp must execute same instructions in lockstep on an SM`，同一 warp 内所有线程必须在 SM 上以 lockstep（锁步）执行同一条指令，每一个周期都执行完全相同的指令。

由此引出 control divergence（控制分歧）：同一 warp 内不同线程需要执行不同指令时，例如出现分支 `if A else B`，就只能先让需要 A 的线程执行、再让需要 B 的线程执行，两条路径被 sequentialized（串行化）。幻灯片用两行 ASCII 直观展示了这种浪费，第一遍只有 8 个 A 线程活跃、其余用 `.` 空转，第二遍反过来：

```text
| AAAAAAAA........................ |
| ........BBBBBBBBBBBBBBBBBBBBBBBB |
```

讲者的结论很直接：`This is bad and inefficient, and that's why branching is something you want to generally avoid`——分支通常应当避免。

![Interaction between programming model and hardware 幻灯片：Warps 小节给出 32 threads per warp、lockstep、control divergence 等要点](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0007_00-08-48.jpg)

数字上这一章要记住 32（每 warp 线程数）、$2 \times 32 = 64$（两个 warp 的例子）、以及"约 5 个例子"这一整体结构；幻灯片里 ASCII 那行 A 恰好 8 个、B 恰好 24 个，也是分歧导致部分线程空转的具体画面。易错点有三：其一，warp 是硬件的执行单位、不是编程模型里的抽象层，所以"写正确的代码"时看不见它，但"写快的代码"时绕不开；其二，不要以为 64 线程的块能同时跑两条不同指令流，它只是两个各自锁步的 warp；其三，别把 control divergence 与后面要讲的 occupancy（占用率）混为一谈——前者是同一 warp 内指令不一致导致串行，后者是能同时驻留多少 warp，成因完全不同。与前后内容的关系也很清楚：这一章把上一章结尾灰显的"模型与硬件交互"具体化成第一个硬件细节，后面四个例子都沿同一逻辑（模型看不见、但性能绕不开）展开；而上一章"线程块必须整块待在一个 SM"的结论，正是 warp 与 SM 绑定关系的前提。

## 控制分歧、warp 调度与占用率的定量计算 [10:15 - 14:21]

warp 还有一个被讲者称为"really cool"的性质：一个 SM 实际上同时运行多个 warp，由 warp scheduler（warp 调度器）管理一批 resident warps（驻留 warp），每个 warp 的线程各有自己的寄存器。SM 可以在它们之间以 zero cost（零成本）切换——讲者特别指出这在 CPU 上一般不成立，GPU 这样设计的目的就是 hide latency（隐藏延迟）。典型场景是：某个 warp 正在读 HBM，一次访问可能要 ~100 个 cycle（周期），与其让 SM 空等，不如立刻切到另一个 warp 去做 tensor core 运算，等数据就绪再切回来。幻灯片原句是 `SM runs multiple warps and switches between them (e.g., when one warp is blocked on HBM reads/writes) with zero cost`。这与上一章的 lockstep 并不矛盾：锁步约束的是同一个 warp 内部，而切换发生在 warp 之间；正因为切换零成本，多驻留一些 warp 才成为掩盖 HBM 延迟的手段。

第二个话题是 occupancy，更准确说是 (Warp) occupancy（warp 占用率）。硬件约束是 `Each thread can use between 0 and 255 registers`，即每个线程最多使用 255 个寄存器。SM 的寄存器总数固定，所以 `The more registers threads use, the fewer threads can be scheduled on an SM (low occupancy)`：单线程用的寄存器越多，能同时驻留的线程就越少，占用率随之下降。但讲者立刻提醒 `Low occupancy isn't necessarily bad if each thread is doing more work`——占用率是可以测量的量，却不是"越大越好"，因为还有别的权衡。对应的设计手段叫 thread coarsening（线程粗化）：与其让一个线程只处理一个元素（线程很"轻"、数量很多），不如让每个线程处理多个元素，例如常数 8 个，这样线程数变少、调度更省，而每个线程做更多工作；讲者的原话是"if your threads are very light, maybe you want to fatten them up a little bit"（线程太轻就把它们养胖一点）。

接着给出可复算的定量例子，转写口述与幻灯片代码框完全一致。设定：一个线程块有 128 个线程、每线程使用 160 个寄存器。幻灯片上的 Python 代码逐字为：

```python
# What we want to run
num_threads_per_block = 128
num_registers_per_thread = 160

# What hardware offers
max_registers = 65536   # Registers allowed per SM
max_warps = 64          # Concurrent warps allowed per SM

# What we can run at once
assert num_registers_per_thread <= 255
num_registers_per_block = num_threads_per_block * num_registers_per_thread
num_blocks = max_registers // num_registers_per_block   # Limited by registers
num_warps = num_blocks * num_threads_per_block / 32
occupancy = num_warps / max_warps
```

运行结果以浮层形式直接给出，四个值分别是：

```text
num_registers_per_block = 20480
num_blocks              = 3
num_warps               = 12
occupancy               = 0.1875
```

![寄存器占用率计算幻灯片：绿色浮层给出 num_registers_per_block=20480、num_blocks=3、num_warps=12、occupancy=0.1875，下方是完整 Python 计算代码](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0011_00-14-13.jpg)

逐步心算一遍：$128 \times 160 = 20480$ 个寄存器/块；$65536 \mathbin{//} 20480 = 3$，即一个 SM 上最多同时跑 3 个块，代码注释 `# Limited by registers` 明确说瓶颈在寄存器；$3 \times 128 / 32 = 12$ 个 warp（除以 32 是因为一个 warp 恰好 32 线程）；最后 $12 / 64 = 0.1875$，也就是约 18%，讲者口述为 `you have a occupancy of 18%`。第一行断言 `assert num_registers_per_thread <= 255` 先确认 160 没有越过单线程的 255 上限。讲者的评语是这个例子说明"内存（寄存器文件）反过来在约束你能做多少 compute"：因为每线程寄存器用得多，占用率被压到 18%。

易错点同样不少。第一，口述把 B200 每 SM 寄存器说成 "65,000"，幻灯片代码写的是 `max_registers = 65536`，这是同一个量的近似值与精确值，别当成两个不同参数；同理 18% 与 0.1875 也是同一结果的两种写法。第二，255 是"单线程"寄存器上限，65536 是"每 SM"寄存器总量，单位不同，最容易被混读；`num_warps` 里的 `/ 32` 则来自上一章的每 warp 32 线程。第三，低占用率不必然是坏事——若每个线程借 thread coarsening 做更多工作，较少线程反而可能更高效，所以别把 occupancy 当成唯一优化目标。第四，别忘了 `max_warps = 64` 这一并发 warp 上限是与寄存器并列的另一条硬件约束，本例中寄存器先撞墙，warp 上限并未用满。与前后内容的关系：这一章把上一章的定性概念变成了可测量、可复算的数字，也呼应讲者说的"profiling 时可以看 occupancy、看 bank conflicts"；下一章转向同样位于片上、但按 bank 组织的 shared memory，那里的冲突是另一类串行化。

## 共享内存 bank 冲突与 HBM 内存合并 [14:21 - 18:15]

第三个例子是 bank conflicts（存储体冲突），作用于 shared memory。先回顾位置：shared memory 像 L1 一样位于 SM 上。硬件把它切成 32 个 bank（存储体），每个 bank 4 字节宽，幻灯片原句是 `Shared memory is divided into 32 banks, each 4 bytes wide`，并用 `B00 B01 B02 B03 ... B27` 标出各 bank，下面用点阵画出矩阵按行铺开、每行横跨全部 bank 的布局。约束是：每个时钟周期，每个 bank 最多只能被一个线程访问（前提是访问的不是完全相同的地址）；因此当多个线程访问同一 bank 的不同位置时，这些访问必须被 serialized（串行化），这就是 bank conflict。幻灯片把相关条目写成：

```text
Each cycle, each bank can only be accessed by one thread (if not the same exact location).
If multiple threads access the same bank, accesses serialized (bank conflict).
Worst case example: matrix where each row spans all banks; 32 threads accessing first column results in 32-way bank conflict!
Unavoidable: when doing matmul A @ B, access rows of A and columns of B
Solution: swizzling rearranges shared memory (e.g., row xor col) to avoid bank conflicts
```

最坏情况是：一个每行横跨所有 32 个 bank 的矩阵，若 32 个线程同时去访问它的第一列，它们会全部落在同一个 bank 上"排队"，形成 32-way bank conflict（32 路冲突）——讲者说这是"你能遇到的最坏设置"。他随即替听众反问"那按行访问不就好了吗"，答案是 unavoidable（无法避免）：做矩阵乘法 $A @ B$ 时，既要按行读 $A$，又要按列读 $B$，有时还要做 transpose（转置），你无法为每个矩阵都规定 row-major（行优先）还是 column-major（列优先）；只有 element-wise 操作无所谓顺序，可以按任意顺序走。幻灯片给出的解法叫 swizzling（重排）：重新安排 shared memory 的布局，使遍历时避开冲突，括号里还给了具体实现线索 `e.g., row xor col`（例如用行号与列号按位异或）：

$$\text{swizzling}: \; \text{bank} \leftarrow \text{row} \oplus \text{col}, \qquad 32\ \text{banks} \times 4\ \text{bytes} = 128\ \text{bytes/行}$$

讲者明说 `There's some solutions here, which I won't get into`，所以本讲只点出名字与思路，不展开实现。顺带一提，转写里提到做 profiling 时可以查看 bank conflicts 与 occupancy，把这一章和上一章的占用率计算连到同一套性能诊断流程里。

![Bank conflicts 与 Memory coalescing 幻灯片：上方为 32 个 B00–B27 bank 与 M00–M32 矩阵行示意，下方给出 128 字节 cache line 与 full coalescing 公式](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0013_00-16-55.jpg)

第四个例子是 memory coalescing（内存合并访问），作用于 HBM。当一个 warp 的 32 个线程访问 HBM 时，内存访问会被合并成 128 字节的 transaction（事务），也就是 cache line（缓存行），一次性全部取回。幻灯片把内存按 32 宽画出两行矩阵元素：

```text
B00 B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 ...
M00 M01 M02 M03 M04 M05 M06 M07 M08 M09 M10 M11 M12 M13 M14 M15 M16 M17 M18 M19 M20 M21 M22 M23 M24 M25 M26 M27 M28 M29 M30 M31 M32
M32 M33 M34 M35 M36 M37 M38 M39 M40 M41 M42 M43 M44 M45 M46 M47 M48 M49 M50 M51 M52 M53 M54 M55 M56 M57 M58 M59 ...
When the 32 threads in a warp access HBM, memory accesses combined into transactions of 128 bytes (cache lines).
Best case: full coalescing, all threads access the same cache line (32 threads x 4 bytes = 128 bytes).
```

最佳情况叫 full coalescing（完全合并）：thread 1 取 `M00`、thread 2 取 `M01`……32 个线程全部落在同一条 cache line 上，一次取回整条，对应幻灯片给出的算式 $32\ \text{threads} \times 4\ \text{bytes} = 128\ \text{bytes}$。反之若沿列往下走，每访问一个元素都要取回一整条大多用不上的 cache line，第二行、第三行依此类推，白白浪费带宽。讲者特别提醒两者"感觉相似，但是非常不同的约束"，并明确划界：`this is about shared memory, and this is about HBM`——bank conflict 是片上 shared memory 里 32 个 4 字节 bank 的每周期单线程限制，coalescing 是片外 HBM 的 128 字节 cache line 事务；把两者混为一谈是最常见的错误。数字上要分清 32 个 bank、每 bank 4 字节、最坏 32 路冲突、128 字节 cache line、32 线程 × 4 字节。与前后内容的关系：这是五个例子中的第四、第五个（按转写顺序 bank conflicts 与 coalescing 连讲），都延续"硬件细节决定性能"的主线；局限是讲者对 swizzling 只给名字不展开，且 matmul 中行列访问顺序不可自由选择，意味着这类冲突无法靠改写遍历顺序彻底消除。下一章的 block occupancy 是最后一个例子，也是从"片上"回到"整块芯片调度"的一层。

## 线程块占用率尾波、层级总结与答疑 [18:15 - 21:40]

第五个例子是 block occupancy（线程块占用率）。线程块被调度到 SM 上；逻辑上你可以定义任意多个线程块，但物理上芯片只有有限个 SM。幻灯片写得很具体：`B200 has 148 SMs, if we launch 160 thread blocks, first wave has 148 blocks, second wave has 12 blocks`。若启动 160 个块，第一波（wave 0）铺满 148 个 SM，剩下 12 个要等第一波结束才作为第二波（wave 1）调度，此时只有 12 个 SM 在干活、其余大量 SM 空闲。这被幻灯片命名为 Wave quantization problem（波量化问题）：`last wave has fewer thread blocks, leaving some SMs idle (low block occupancy)`。解法只有一行：`Solution: make number of thread blocks divide # SMs`，也就是让线程块总数能被 SM 数整除：

$$160 = 1 \times 148 + 12, \qquad \text{理想情形: } \; N_{\text{blocks}} \bmod 148 = 0$$

幻灯片顶部的波形图以 SM 为纵轴、`time` 为横轴，画出 `wave 0` 满高的绿条和标着 `wave 1 (tail)` 的明显更矮的绿条，正是这个"尾巴"的可视化。讲者的口述版本是：若启动 160 个块，先调度 148 个，等它们做完再调度剩下的 12 个；"当最后这一波的线程块少于总数时，就有一堆 SM 什么都不干"，这叫低占用率。

![Block occupancy 与总结幻灯片：给出 B200 148 SMs / 160 blocks 两波调度、Wave quantization problem、以及 grid→thread block→thread 总结链](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0015_00-19-38.jpg)

讲者随后收束全段。幻灯片黄色高亮的总结行是 `Programming model: grid (HBM) -> thread block (shared memory) -> thread (registers)`：HBM 对所有线程全局可见，shared memory 属于线程块，寄存器属于线程；而下一行 `Details of hardware (warps, bank conflicts, memory coalescing, occupancy) determine performance` 则把前面四章的硬件细节一一点名，说明它们才真正决定速度。转写里讲者进一步坦承这些细节"很难搞清楚"：profiling 会告诉你一堆信息，但你仍得知道究竟有多少个 SM、每一级容量究竟多大，而且"有时候调度器会做一些你根本无法控制的事"，所以现实比编程模型 messy（凌乱）得多。幻灯片底部已经灰显出下一节的开头，是画面上存在、但本段口述尚未展开的内容：

```text
Summary:
Programming model: grid (HBM) -> thread block (shared memory) -> thread (registers)
Details of hardware (warps, bank conflicts, memory coalescing, occupancy) determine performance

def benchmarking_and_profiling():
Recipe for success:
1. Benchmark and profile your code
2. Make changes
3. Benchmark and profile your code again
benchmarking()  # How long does it take?
```

20:20 之后进入答疑。学生问：既然有 148 个 SM、却要跑 165 个块（学生口中的数字比讲者例子的 160 更不整齐），有没有办法让一个 block 共享（摊到）一个 SM 上？讲者的回答偏向否定：如果一个块已经把 SM 上的 tensor cores 用得差不多，再塞一个块上去并不会真的加速；根本困难是一种 `jagged / unevenness`（参差不齐、不均匀）的问题，因为 `the blocks have to stay together`（块必须保持在一起），不能把一个块拆开摊到别处去。可行的做法是改变 block size（块大小）、也就是改变块的数量，从而不产生这种 tail（尾巴）。这与上一章"线程块必须整块待在一个 SM"的结论完全一致，也解释了为什么解法是"改块数"而不是"拆块"。

数字再核对一遍：148（SM 数）、160（启动块数）、148+12（两波）、165（学生提问里的块数）、12（尾波里实际工作的 SM 数）、以及前面算出的 $0.1875 \approx 18\%$ warp occupancy。这里最容易混淆的是两个 occupancy：block occupancy 指"块数没对齐 SM 数而产生的尾波空闲"，warp occupancy 指"寄存器用量限制了一个 SM 上能同时驻留多少 warp"，名字相近、层次与成因都不同，务必区分。局限方面，讲者坦承调度器行为不完全可控、硬件参数必须精确已知，因此"能写对"与"跑得快"之间仍隔着一层需要 benchmark 与 profiling 才能填平的鸿沟——这正好回答了开头为什么本讲要"深入代码"。整段从第一遍的存储层级出发，经三层编程模型、五个硬件例子，最后回到 `grid (HBM) -> thread block (shared memory) -> thread (registers)` 这条总结链，并把"如何理解性能"交给下一节的"如何动手测量"。

## 先测量再动手：benchmarking 与 profiling 的成功配方 [21:40 - 23:20]

在前面把 GPU 的存储层级、warp、bank conflict、memory coalescing 与 occupancy 讲完之后，讲者用一句 "hopefully you guys are getting more comfortable with GPUs now" 收束，正式把话题从硬件转向 benchmarking（基准测试）与 profiling（性能剖析）。他特意说明，这一小段"在内容上不会讲太多"，真正想强调的是背后的 philosophy（方法论）——屏幕上把它写成 `def benchmarking_and_profiling():` 下面的一句 "Recipe for success"（成功配方）。这个配方只有三步，且第三步被高亮：第一步 Benchmark and profile your code，第二步 Make changes，第三步 Benchmark and profile your code again。紧随其后的两行注释把两个工具的分工点得很清楚：`benchmarking()  # How long does it take?`（要花多长时间）与 `profiling()  # Where time is being spent?`（时间花在哪里）。换句话说，测量不是一次性的开场动作，而是一个"测—改—再测"的闭环；只改不测，或者只在开头测一次，都不算走完这个配方。

为什么把 benchmark / profile 放在教 Triton 写 kernel 之前？讲者给出的理由是："you should always just measure what's going on in your code and figure out what the bottlenecks are before you start you know, writing kernels"——总要先量一量代码在干什么、把瓶颈找出来，再动笔写 kernel。这句话是本段全篇的纲：后面 GELU 的例子、profiler 里那些长名字、以及最终引出 Triton，都是在演示"先测量、再定位、最后才优化"这条路径。画面顶部还残留着上一段的 Summary（小结）两行，正好构成承接关系：`Programming model: grid (HBM) -> thread block (shared memory) -> thread (registers)` 与 `Details of hardware (warps, bank conflicts, memory coalescing, occupancy) determine performance`。前者说编程模型只描述三级结构，后者说真正决定性能的是那些硬件细节——正因为二者之间存在张力，才必须用测量去揭示实际发生了什么，而不能凭模型推断。同一张幻灯片更靠上的几行还记着 wave quantization（波量化）的例子：B200 有 148 个 SM，若启动 160 个 thread blocks，第一波装满 148 个，第二波只剩 12 个，于是部分 SM 空闲（low block occupancy，即低块占用率）；对应的 Solution 是 "make number of thread blocks divide # SMs"。这些数字属于上一段的收尾，但被一并留在画面里，提醒读者 benchmark 要度量的正是这类"模型看不见"的效应。

```python
def benchmarking_and_profiling():
    # Recipe for success:
    # 1. Benchmark and profile your code
    # 2. Make changes
    # 3. Benchmark and profile your code again
    benchmarking()   # How long does it take?
    profiling()      # Where time is being spent?
```

接着讲者给 benchmarking 下定义：它测的是 "the wall-clock time of performing some operation"（执行某个操作的墙钟时间），并且 "It only gives you end-to-end time, not where time is spent (profiling)"——只给一个端到端总时长，不告诉你时间花在哪个环节，后者是 profiling 的职责。尽管信息量被压缩成一个数，它仍然有用，幻灯片列了两条：comparing different implementations (which is faster?)（比较不同实现谁更快），以及 understanding how performance scales (e.g., with dimension)（理解性能如何随维度扩展）。讲者在口述里补了一句直觉：正因为把所有东西蒸馏成一个数字，才能拿它去看 scaling（扩展性），比如随 dimension（维度）变化的趋势。工具层面，他提到可以用 `torch.utils.benchmark`，但因为这门课叫 language models from scratch，所以 "We will roll our own to make benchmarking more transparent"——自己从零写一个，目的就是把几个 gotchas（坑）暴露出来。

被测量的第一个算子是矩阵乘法：画面里写着 `matmul = run_operation2(dim=1024, operation=lambda a, b: a @ b)` 与 `result = benchmark(matmul)`，其中 `dim=1024`。`run_operation2` 是一个 wrapper（包装器）：它先实例化两个 `dim × dim` 的随机方阵（random square matrices），再返回一个只负责执行该运算的函数；讲者强调随机矩阵是预先生成好的，调用 `matmul` 时"就只是把两个矩阵相乘"，所以生成数据的开销不会被算进计时。同一页更淡的几行还给出了一次随维度扫描的实验：`results = {}`，然后 `for dim in [256, 512, 1024, 2048, 4096, 8192]:` 逐个执行 `results[dim] = benchmark(run_operation2(dim=dim, operation=lambda a, b: a @ b))`，旁边一句注释 "Note: time is roughly constant when dimension is small, then cubic scaling." 这个 `[256, 512, 1024, 2048, 4096, 8192]` 的维度列表和"小维度近似恒定、随后立方增长"的结论都是画面上独有、口述里没逐字念出来的，下一章会展开它的成因。

易错点与局限也在这里埋下伏笔。第一，benchmark 天然只回答"多久"，不回答"哪里慢"，把端到端时间当成瓶颈定位是范畴错误；必须配合 profiling 才能形成闭环。第二，配方的第三步常被省略：改完代码不重新测，等于没有验证。第三，wall-clock 时间会被两类噪声污染——首次运行的惰性编译（lazily compiled）开销，以及 CPU 侧把 kernel 发射出去的开销；讲者说因为多数场景下你会反复运行同一段代码，所以初始条件并不重要，真正该测的是稳态。这些坑正是下一章要逐条处理的：warm up（预热）、多次取样取平均、用 CUDA events 而不是 CPU 时钟、以及 `synchronize()` 同步屏障。局限性上，单一平均值会掩盖方差，讲者后面也承认若特别讲究应该去看整个分布的 P95。

![成功配方三步、benchmarking()/profiling() 分工注释，以及上一段留下的编程模型与硬件细节小结](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0017_00-22-20.jpg)

## 从零写 benchmark：warm up、CUDA events 与 synchronize [23:20 - 26:23]

定义讲清之后，问题变成 "how do you benchmark?"。讲者先摆出最朴素的写法："start time, run it and then stop time"——记一个开始时间、跑一次、再记一个结束时间。紧接着他说 "there's a few things"，逐条拆坑。第一条是 warm up（预热）：他在第二讲就提过，这里值得再强调，因为 "some things are lazily compiled"（有些东西是惰性编译的），第一次调用会把编译时间算进去；而 "you're going to run it over and over again"，真正关心的是反复运行时的速度，"the initial conditions don't really matter"。代码里把这两层意思写成了注释：`# Warmup: first times might be slower (e.g., compilation), etc.` 与 `# Since we will run the kernel multiple times, the timing that matters is steady state.`，对应 `for _ in range(num_warmups): run()`。函数签名本身也给了默认值：`def benchmark(run: Callable, num_warmups: int = 1, num_trials: int = 3) -> float:`，即默认预热 1 次、正式计时 3 次，docstring 写明 `"""Benchmark \`run\` by running it \`num_trials\`. Return the average time."""`。

第二条是同步。GPU 上的一切都是异步（async）执行的，CPU 把 kernel 发射出去后立即返回，若此时直接读表，量到的只是发射动作而非 kernel 真正跑完的时间。因此代码在预热循环之后插入 `torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)`，注释里那个 `(important!)` 就是讲者口述的 "synchronization barrier"（同步屏障）。第三条是用对的计时源："the proper way to time things is to use these CUDA events"——一个 `start_event`、一个 `end_event`，在计算前 `start_event.record()`，执行 `run()`，再 `end_event.record()`，同步之后用 `start_event.elapsed_time(end_event)` 取差值。代码注释把理由写得很直白：`# Use CUDA events for accurate GPU timing (avoid capturing CPU overhead)`，也就是避免把 CPU 侧开销混进来。

```python
def benchmark(run: Callable, num_warmups: int = 1, num_trials: int = 3) -> float:
    """Benchmark `run` by running it `num_trials`. Return the average time."""
    # Warmup: first times might be slower (e.g., compilation), etc.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run()

    torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Time it for real now!
    times: list[float] = []
    for trial in range(num_trials):  # Do it multiple times to capture variance
        # Use CUDA events for accurate GPU timing (avoid capturing CPU overhead)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()  # Start timing
        run()  # Actually perform computation
        end_event.record()  # End timing

        torch.cuda.synchronize()  # Wait for CUDA threads to finish
        times.append(start_event.elapsed_time(end_event))
```

第四条是重复取样。画面左上角的变量查看框显示 `times` 此刻还是空列表 `[]`，随后 `for trial in range(num_trials):  # Do it multiple times to capture variance` 每次把 `elapsed_time` 的结果 append 进去，最后按 docstring 所说取平均（讲者口述 "here we're just taking the average"）。他同时给了局限："if you are being very particular, you want probably want to maybe look at the whole distribution the P95 or or whatever, but we'll just do the average here"——严谨时应看完整分布、看 P95 这类分位数，平均值对长尾不敏感，但本讲为了简单只用均值。于是 `num_warmups=1`、`num_trials=3`、`enable_timing=True`、两次 `synchronize()`、`times: list[float]` 这些参数与写法共同构成了后面所有计时的公共基础设施：下一章的 `profile(...)`、以及 GELU 一节的 `naive_time = benchmark(run_operation1(dim=16384, operation=naive_gelu))` 等，都是把不同的 `run` 塞进同一个 `benchmark`。

把这套工具用在随维度扩展上，就得到本段最直观的一条曲线（画面注释 "Note: time is roughly constant when dimension is small, then cubic scaling."）。讲者说矩阵乘法"as expected should grow you know, cubically"，即计算量随维度按 $O(d^3)$ 增长；但实测时间在小维度并不跟着涨，而是存在一段 floor（平台）："up until you get up to like almost 2000 dimensional matrices things are basically kind of a constant"。他给出的解释接回前一段的硬件直觉：GPU 就是为 "fairly large matrix multiplications" 设计的，若拿一个 2×2 的矩阵去乘，"it's going to be very kind of inefficient"——规模太小，SM 和内存带宽都填不满，启动与调度开销反而主导了总时间。所以 $O(d^3)$ 只是算法复杂度的趋势，而实测曲线在 2000 维以下被固定开销压平，这正是 benchmark "把一切蒸馏成一个数、从而暴露 scaling" 的价值：它让你看见理论复杂度与真实硬件行为之间的落差。

易错点可以归纳成一句话：这段 benchmark 的正确性依赖于四个缺一不可的动作——预热剔除惰性编译、用 CUDA events 而非 CPU 时钟、每次记录后 `synchronize()` 等待 GPU、多次取样再取平均。漏掉任何一个，量到的都不是 kernel 的稳态耗时。局限同样明确：平均值掩盖方差（严格要看 P95），端到端时间不区分 CPU 发射开销与 GPU 计算，更不告诉你是哪个 kernel 慢；而"2000 维以下近似恒定"是这台机器、这个算子上的经验观察，不能当成普遍常数推广。正因为这些局限，讲者在 [26:25] 立刻转向 profiling："profiling tells you where time is actually spent"。

![benchmark 函数原文：num_warmups=1、num_trials=3、torch.cuda.Event 计时、两次 synchronize 与 times 列表](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0019_00-25-03.jpg)

## Profiling：看见底层真正被启动的 CUDA kernel [26:23 - 30:14]

讲者用 "really quickly on profiling" 开场，给出与 benchmarking 对偶的定义："profiling tells you where time is actually spent"（时间实际花在哪里）。他补了一个不那么显然的用途："even if you don't care about the time helps you figure out what's actually happening under the hood"——即便不关心性能，profiling 也能帮你看清高层代码底下到底发生了什么；用高级语言写完代码、一跑就有结果，有时"it can be just like good to understand what's actually going on"。工具上他提到 PyTorch 自带 profiler（作业里用的那个会给出更多细节），但为了赶时间（"in the interest of time"）课上直接用简版。代码层面，profiling 与 benchmark 一样先 warm up，再把 `run` 放进 profiling context 里执行；画面 26:24 那一页正好把两行分工并列高亮：`benchmarking()  # How long does it take?` 与 `profiling()  # Where time is being spent?`，下面还有一句 "Benchmark and profile your code!"。

```python
# 先 warm up，再进入 profiling context 执行 run
add_profile = profile(run_operation2(dim=2048, operation=lambda a, b: a + b))
```

第一个实验是 `A + B`：讲者说平时写 PyTorch 大概不会多想，"these two tensors just get added"，但 profiler 显示底层调用的是一长串名字，`void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add, std::array<...`——也就是一个 `CUDAFunctor_add` kernel，专门把两个张量逐元素相加。表格里这一行 Self CPU % 是 `0.00%`，而 `Activity Buffer Request` 占 `97.35%`、`cudaLaunchKernel` 占 `2.02%`、`cudaDeviceSynchronize` 占 `0.63%`，底部写着 `Self CPU time total: 3.391ms` 与 `Self CUDA time total: 4.960us`。讲者提醒此时时间数字 "aren't going to be that interesting because I'm only adding"，因为只做一次加法，kernel 会占掉 100% 的时间；真正有价值的结论是 "underneath the hood there is this thing called add"——你以为的 `+` 其实是一个具名的 CUDA kernel。

第二个实验把 `+` 换成 `@`，即在 PyTorch 里做 `A @ B`（matmul）。profiler 里出现的是另一条更长、信息量更大的名字：`cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_32x32x16_1x1x1_3_nnn_align1_bias_f32_relu`，对应表格 `0.00%` Self CPU；旁边是 `cuLaunchKernelEx` `0.78%`、`Activity Buffer Request` `98.83%`、`cudaDeviceSynchronize` `0.40%`，底部 `Self CPU time total: 4.134ms`、`Self CUDA time total: 4.544us`。讲者顺势给出 Observations（观察）：第一，"You can see which CUDA kernels are actually being called (the long names)"——长名字的那些行就是真正被调用的 CUDA kernel；第二，"different CUDA kernels are invoked depending on the tensor dimensions"——换张量维度就会换 kernel，他口述里先念到一个 `F32 F32 64 64 16`，又说把矩阵改成 128×128 时会得到不同的一个，"this is 64 64 16. This is 32 32 16"，画面里 29:07 这一页显示的正是带 `32x32x16` 的那条。这说明高层看似一个 `@`，底层却可能根据形状分派到完全不同的实现。

```text
void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add, std::array<...
cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_32x32x16_1x1x1_3_nnn_align1_bias_f32_relu
```

更有意思的是"名字本身会泄露实现细节"。讲者逐段拆这个长名：`CUTLASS` 是 NVIDIA 的 CUDA 线性代数库（"the NVIDIA CUDA library for linear algebra"）；`SM100` 对应 Blackwell 架构（转写作 Blackwall），也就是说这个 kernel 是 "specifically designed for Blackwall"；`F32` / `FP 32` 指单精度浮点；而 `64 64 16`（画面里是 `32x32x16`）是 the shape of the tile（tile 的形状，即分块尺寸），讲者明确说这个 tiling 的话题 "we're going to talk about later when we talk about tiling in the context of matmuls"，留到后面讲矩阵乘法的 tiling 时再展开。于是 `sm100`、`sgemm`、`f32`、`32x32x16`、`align1`、`bias_f32_relu` 这些片段，把库、架构、精度、分块尺寸乃至是否融合 bias 与 relu 都编码进了符号名。这与前一段硬件课里讲的 tile、occupancy 形成呼应：profiler 是连接"PyTorch 一行代码"和"硬件上真正跑的 kernel"之间的那台显微镜。

收尾时讲者只留一句："last benchmarking profiling, just remember to do it ... we make you do it on the assignment, so you have no choice"，呼应开场的成功配方。易错点有两层：一是别把 Self CPU % / Self CUDA % 的百分比直接读成 kernel 的算力分布——表里占比最高的往往是 `Activity Buffer Request`、`cudaLaunchKernel`、`cudaDeviceSynchronize` 这类条目，真正的 kernel 行反而显示 `0.00%`；二是别以为同一个高层算子总对应同一个 kernel，维度一变实现就变。局限上，这一节用的是简版 profiler，只列名字与 CPU 侧统计，作业里那个工具才会给更多细节；而"看名字猜实现"终究是推断，要确认还得靠后续对 Triton / PTX 的深入。

![matmul 的 PyTorch profiler 输出：cutlass3x_sm100_simt_sgemm 长名 kernel 与 98.83% 的 Activity Buffer Request](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0022_00-29-07.jpg)

## GELU 案例：公式、三种实现与性能数字 [30:14 - 33:30]

"let's apply this to another example here on the GELU"——讲者把刚建立的 benchmark + profile 组合拳用到 GELU（Gaussian Error Linear Unit，高斯误差线性单元）上，它是 Transformer 里常用的非线性（"a typical non-linearity that's used"）。画面是 PyTorch v2.11 文档里的 `torch.nn.GELU` 页，左侧导航依次列着 CELU、GELU（当前高亮）、Sigmoid、SiLU、Mish、Softplus、Softshrink、Softsign、Tanh、Tanhshrink、Threshold。类签名是 `class torch.nn.GELU(approximate='none')`，正文一句 "Applies the Gaussian Error Linear Units function."，紧接着给出精确定义 $\mathrm{GELU}(x) = x \cdot \Phi(x)$，并说明 "where $\Phi(x)$ is the Cumulative Distribution Function for Gaussian Distribution"（$\Phi(x)$ 是高斯分布的累积分布函数）。因为精确式里的 $\Phi$ 计算较贵，常用一个 "more compute-friendly" 的 tanh 近似代替；文档原话是 "When the approximate argument is 'tanh', Gelu is estimated with:"，公式为：

$$
\mathrm{GELU}(x) = 0.5 \cdot x \cdot \left(1 + \mathrm{Tanh}\!\left(\sqrt{2/\pi}\,\left(x + 0.044715\,x^3\right)\right)\right)
$$

这里的 $0.5$、$\sqrt{2/\pi}$、$0.044715$、$x^3$ 都是画面上明确写出的常数，后面 Triton kernel 的注释里还会以 `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` 的形式原样复现，是本章与最后一章之间的直接纽带。讲者说朴素实现（naive implementation）就是把这条方程直接搬进 PyTorch："you take a tensor, you just take this equation and you just put it into PyTorch"，画面里第一条注释把它标成 `# 1. Implementation: naive, translated directly ... (non-fused)`（非融合）；PyTorch 也内置了 `nn.functional.gelu`（代码里叫 `builtin_gelu`），调用它得到第二种版本。第三匹马来自一个"people probably some have discovered"的重要技巧：对任意 PyTorch 函数调用 `torch.compile`，会生成另一个函数，"and it does the same thing"——`compiled_gelu = torch.compile(naive_gelu)`。于是 "we have three horses in this race"：naive（手写）、built-in（内置）、compiled（编译）。

```python
x = torch.tensor([1.])

# 2. Built-in PyTorch implementation (fused)
y2 = builtin_gelu(x)
check_equal_id(naive_gelu, builtin_gelu)  # Check it works

# 3. Use PyTorch compiler on the naive implementation
compiled_gelu = torch.compile(naive_gelu)
y3 = compiled_gelu(x)
check_equal_id(naive_gelu, compiled_gelu)  # Check it works (compilation shouldn't change semantics)

# Benchmarking
naive_time = benchmark(run_operation1(dim=16384, operation=naive_gelu))
builtin_time = benchmark(run_operation1(dim=16384, operation=builtin_gelu))
compiled_time = benchmark(run_operation1(dim=16384, operation=compiled_gelu))
```

三种实现先要确认语义一致：代码里用 `check_equal_id(naive_gelu, builtin_gelu)` 和 `check_equal_id(naive_gelu, compiled_gelu)` 做校验，后者的注释特别写着 "compilation shouldn't change semantics"（编译不应改变语义）；输入取 `x = torch.tensor([1.])`，弹出的变量查看框显示 `x = 1`，而 `y1 = y2 = y3 = 0.8412`——三个版本对同一个输入给出完全相同的 `0.8412`。接着用上一章的 `benchmark` 计时，注意这里换成了 `run_operation1`，维度统一取 `dim=16384`：变量框里 `naive_time = 3.7583`。讲者口述 "Naive takes ... 3.75. Built-in is much faster and compiled is is much faster as well, but not maybe as quite as fast as built-in"，也就是三者数值相同却 "have wildly different performance characteristics"，快慢差距极大。这个 `dim=16384`、`3.7583`、`0.8412` 都是画面独有的具体数字。

为什么会差这么多？讲者立刻把 profiler 拉出来："here we can pull up the profiler and see what's actually going on underneath the hood"，并先切到一个"something's wrong with this view"的视图再看正确那个。画面 33:16 的终端（标题 `petrushka:~/cs336`，tmux 状态栏显示 `[cs336] 0:bash- 1:* 2:bash 3:bash 4:>"petrushka:~/cs336/lec" 15:34 15-Apr-26`）里，naive GELU 的 profiler 一次列出好几条 kernel：`at::native::BinaryFunctor<float, float, float, ...>`、`at::native::AUnaryFunctor<float, float, float, ...>`、`at::native::CUDAFunctor_add<float>`、`at::native::CUDAFunctorOnSelf_add<float>`、`at::native::tanh_kernel_cuda(at::TensorIterator...`，Self CPU % 全是 `0.00%`；其余是 `Activity Buffer Request` `44.07%`、`cudaLaunchKernel` `7.56%`、`cudaDeviceSynchronize` `48.37%`，底部 `Self CPU time total:` 的数字被画面裁掉了一部分。讲者的解释是：在 PyTorch 里写下这个表达式，就会得到一张 computation graph（计算图），"each primitive in the computation graph is actually realizing a kernel"——图里每个 primitive（原子操作）都会落实成一个独立的 kernel；binary functor、unary、add、tanh 正是把公式里的乘法、加法、$\tanh$ 等逐项拆开的结果。这与前两章的观察一脉相承：高层一行代码，底层是一串 kernel。

易错点在于：别以为"公式写对了、数值对上了"就等于实现没问题——三个版本都输出 `0.8412`，但 `3.7583` 与另两个数字之间的差距来自 kernel 数量而非算术本身。另一个易错点是近似式的选择：`approximate='none'` 是精确 $\mathrm{GELU}(x)=x\Phi(x)$，`'tanh'` 才是上面那条含 $0.044715$ 的近似，二者数值并不严格相等，混用时校验要写清楚。局限上，这一节只给了 `naive_time = 3.7583`，`builtin_time` 与 `compiled_time` 的具体数值要到下一张（34:49）的变量框才出现；而且 `dim=16384`、单次 `num_trials=3` 的均值只是这台机器上的一次测量，讲者后面也强调结果 "very hardware-dependent"。

![PyTorch v2.11 文档中的 GELU 定义页：x·Φ(x) 与 tanh 近似公式 0.5x(1+Tanh(√(2/π)(x+0.044715x³)))](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0023_00-30-28.jpg)

![GELU 三种实现对比代码与变量查看框：x=1、y1/y2/y3=0.8412、naive_time=3.7583、dim=16384](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0024_00-31-49.jpg)

## 从 profiler 看 kernel fusion：内置 kernel 与 torch.compile [33:30 - 36:51]

naive GELU 慢的根因，讲者用一段很有画面感的描述点破："when you launch a kernel, the kernel has to read from HBM, pull it all the way over to your SM, do the computation, and write it back. And then the next kernel picks it up from HBM ... and then writes it back and so on and so forth." 也就是每个 kernel 都要把数据从 HBM（高带宽内存）搬到 SM、算完再写回 HBM，下一个 kernel 再从 HBM 读一遍。于是 "you're doing a lot of reads and writes back and forth because between kernel invocations things have to go back to HBM"。把这句话和上一章 profiler 里那一串 `BinaryFunctor` / `AUnaryFunctor` / `CUDAFunctor_add` / `tanh_kernel_cuda` 对应起来：GELU 公式里每多一个 primitive，就多一次 HBM 往返，而 HBM 正是前几讲反复强调的"大而慢、在远处"的存储。结论是 "There's no ... kernel fusion here. This is slow"——没有 kernel fusion（算子融合），所以慢。

对照之下，内置版本 "actually not that interesting"：profiler 里只有一条 `GELU CUDA kernel implementation`，"a single kernel that just implements the GELU"。讲者顺带回答了"为什么会有它"："because you know, people use GELU, so someone wrote a kernel for it and put it in the standard library ... there's no nothing magical"——有人用了、就有人手写了一个 CUDA kernel 塞进标准库，没什么魔法；当被问到它是不是也直接来自 CUDA 时，他看着名字说 "I guess this says CUDA kernel implementation, so I imagine someone wrote it in CUDA"。第三种，也就是 `torch.compile`，才是他口中 "really interesting" / "a fascinating topic" 的那个：naive 实现本来有一张 computation graph，编译器 "look at the computation graph and essentially write that kernel in Triton"，把整张图写成**一个** kernel，而且 "this is actually a Triton kernel"。于是三种实现的差别被压缩成一句话：naive 是 multiple kernels、多次读写 HBM、无 fusion；built-in 与 compiled 都是 one kernel，"all the operation in the GELU have been fused together into one kernel. So, you read from HBM once, you write to HBM once per element"。

```python
builtin_time = benchmark(run_operation1(dim=16384, operation=builtin_gelu))
compiled_time = benchmark(run_operation1(dim=16384, operation=compiled_gelu))
compiled_gelu_profile = profile(run_operation1(dim=16384, operation=compiled_gelu))
```

```text
triton_poi_fused_add_mul_tanh_0
```

34:49 那一页把数字补齐了。变量查看框里依次是 `x = 1`、`y1 = 0.8412`、`y2 = 0.8412`、`y3 = 0.8412`，以及 `naive_time = 3.7583`、`builtin_time = 0.6670`、`compiled_time = 0.9388`。按这三个数字算，在 `dim=16384` 上内置比 naive 快约 5.6 倍，编译版快约 4 倍，且 "compiled is much faster as well, but not maybe as quite as fast as built-in"——编译版略慢于内置版。下方 `compiled_gelu` 的 profiler 表列名是 `Name`、`Self CPU %`、`Self CPU`、`CPU total %`、`CPU total`、`CPU time avg`、`Self ...`，其中唯一一条计算 kernel 是 `triton_poi_fused_add_mul_tanh_0`（`0.00%`、`0.000us`），名字里的 `fused_add_mul_tanh` 正好把被融合掉的加法、乘法与 $\tanh$ 都点了出来；表格上方还残留着 `cudaLaunchKernel 21.88%`、`cudaDeviceSynchronize 0.78%`，以及 `Self CPU time total: 1.296ms`、`Self CUDA time total: 305.09us`。与上一章 naive 表里那一长串 kernel 相比，"一排 kernel" 变成 "一条 kernel"，就是 fusion 最直接的证据。

紧接着是一段值得记下来的现场问答。有听众问 "Why is Triton kernel faster than this one?"，讲者先纠正前提："a Triton kernel is actually not faster in this case ... The compiled kernel is one Triton kernel and this is slower than the built-in"——编译出来的那个 Triton kernel 恰恰比内置 CUDA kernel **慢**，Triton 并不天然更快。他又补了一句很重要的限定："last year when I did this, it was actually closer ... these things change and it's very hardware-dependent ... none of this is like terribly optimized. This is just giving you a the general idea here." 去年两者更接近，结果会随硬件与库版本变化，且这些实现都没做深度优化，演示只为传达思路。画面 36:22 切到讲师站在白板前的镜头（白板上是别的课留下的手写公式，与本讲内容无关），正对应这段问答。这一问一答也自然引出下一章："that maybe is a good segue to talk about what this Triton thing is all about"。

易错点有三：第一，把 `torch.compile` 当成"自动变快"的开关，而忽略它快的原因是 fusion 减少了 HBM 往返——若计算图本身只有一个 primitive，编译收益就有限；第二，把 `built_in`/`compiled` 的快慢当成语言优劣（Triton vs CUDA），讲者明确否认，强调结果 hardware-dependent；第三，读 profiler 时把 `Self CPU %` 那几列当成 kernel 耗时，`Activity Buffer Request` 之类的条目会占掉绝大部分。局限上，这里只看了一个 `dim=16384` 的逐元素算子、一次 `num_trials=3` 的平均，既没覆盖不同形状，也没看 P95；而 "read from HBM once, write once per element" 是对 element-wise 融合的理想描述，到后面 matmul / softmax 这类需要线程间通信的算子就不再成立。

![compiled_gelu 的 profiler 报告：triton_poi_fused_add_mul_tanh_0 单 kernel，以及 naive/builtin/compiled 三种耗时](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0026_00-34-49.jpg)

## Triton 的抽象层级：以 thread block 为中心 [36:51 - 39:44]

"let's write some Triton kernels" 之前，讲者先回到 programming model（编程模型）把坐标系摆正：一堆 thread 组织成 thread group / thread block，thread block 再组成 grid of thread blocks。屏幕上 37:55 与 39:28 两页都画着同一张绿色示意图：一个大绿框标着 `Grid`，里面整齐排着 2 行 4 列共 8 个方框，每个方框标 `CTA`（即 thread block），框里画着约 8 根向下的绿色箭头代表线程。这张图把上一讲的"grid → thread block → thread"三级结构具象化，也正好对应 kf_0017 小结里那行 `Programming model: grid (HBM) -> thread block (shared memory) -> thread (registers)`，可写成：

$$
\text{grid (HBM)} \;\to\; \text{thread block (shared memory)} \;\to\; \text{thread (registers)}
$$

两种语言的分野在于"你以谁为单位思考"。幻灯片原文（以 39:28 高亮到最后一行为准）：

```text
In CUDA (developed by NVIDIA), specify what each thread does.
    Pros: fine-grained control
    Cons: need to manage more things (e.g., shared memory)
In Triton (developed by OpenAI), specify what each thread block does.
    Generally powerful enough (especially when getting started)
    Conceptual framework: load data into shared memory, operate on it, write back to global memory
```

CUDA 由 NVIDIA 开发，多年来就是写 kernel 的默认选择，mental model 是 "what does each thread do?"：写一段带 thread ID 的代码，用这个 ID 标识当前线程，然后每个线程各自执行同一段代码。好处是 "very closely related to what is actually happening underneath the hood"、"fine-grained gives you fine-grained control"（贴近硬件、控制精细）；代价是 "need to manage more things"——同一个 thread block 里的线程若要通信（例如一起从 HBM 读数据），就必须自己 synchronize、做 bookkeeping：一起读、同步、计算、再写回。讲者的判断很务实：如果全是 element-wise 操作，"CUDA is just fine. It doesn't really matter"；但 "as you get more complex operations, then Triton provides some value of the abstraction"。

Triton 由 OpenAI 开发，"by now it's been pretty standard"，思路变成 "specify what each thread block does"。它的 conceptual framework（概念框架）被高亮成一句："A block is going to load data into shared memory, operate it on, and write it back to you know global memory"——一个块把数据 load 到 shared memory、在片上计算、再 store 回 global memory。讲者给它的定位是"中间层"："these blocks are intermediate point between thinking about what individual elements are doing, as well as thinking about the general operation"。一端是 PyTorch：你定义巨大的矩阵、说"把它们乘起来"，`operation=lambda a, b: a @ b` 就是那种 atomic operation，写 PyTorch 时 "a lot of what you're thinking about is how do I get things into big matmuls"（怎么把计算塞进大矩阵乘）；另一端是逐元素的线程。Triton 处在两者之间，是 "sort of hybrid between that and you know, the individual elements"。这与上一章 GELU 的结论接得上：`torch.compile` 生成的 `triton_poi_fused_add_mul_tanh_0` 已经是"一个块把整段公式算完"的形态，现在要亲手写出这种 kernel。

能力边界讲者也没回避："Generally powerful enough (especially when getting started)"——对本课程和入门足够；但 "if you really go and want to exploit every single new feature of the latest hardware, you know, it might not give you the full flexibility"，要榨干最新硬件的每个特性时，Triton 未必给得了 CUDA 那样的完全自由度，"but let's not worry about that"。这既是与前面硬件课（warp、bank conflict、occupancy 都要手工调优）的呼应，也是给后文留的伏笔：Triton 把"线程间同步、shared memory 管理、数据放寄存器还是 shared memory"这些决策交给编译器，换来了写法的简洁。

易错点：别把 Triton 的 "block" 和后面 matmul 里的 "tile" 混为一谈，也别以为"以块为单位"就意味着一个块能装下任意大的张量——下一章马上会讲，张量太大装不进一个 SM，必须先切成 blocks。另一个易错点是把 shared memory 当成程序员可随意摆布的资源：在 Triton 里它是概念框架的一部分，但具体分配与同步由编译器代劳。局限上，这种抽象牺牲了对最新硬件特性的完全控制，且不改变一个事实：无论 CUDA 还是 Triton，性能仍由前几讲的 warp（32 线程束）、bank conflict（共享内存 bank 冲突）、memory coalescing（HBM 合并访问）、occupancy（占用率）决定——抽象层级变的是"你写什么"，不是"硬件怎么跑"。

![Grid 中 8 个 CTA 的示意图，以及 CUDA 与 Triton 抽象层级的 Pros/Cons 与 conceptual framework 原文](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0029_00-39-28.jpg)

## 第一个 Triton kernel：指针、grid 与 mask [39:44 - 43:20]

"let's start with the value example here"——讲者用 GELU 的逐元素例子落地 Triton（"Triton is basically going to be you know, you write Python"，先问了在场多少人写过 Triton，举手不多，于是"you won't be bored"）。开头几行刻意 "this is normal PyTorch, right? There's no Triton here, but I'm just preparing"：定义一个向量、分配输出。代码原文（41:01 页）：

```python
def triton_gelu(x: torch.Tensor):
    # Check input
    assert x.is_cuda
    assert x.is_contiguous()

    # Allocate output tensor
    y = torch.empty_like(x)

    # Determine grid (elements divided into blocks)
    # | T T T T T T T T | T T T T T T T T | T T T T T T T T T | T T T T T T T T |
    # |     Block 0     |     Block 1     |     Block 2     |     Block 3      |
    num_elements = x.numel()
```

两个 `assert` 是边界约束（张量必须在 CUDA 上、必须 contiguous 连续），`y = torch.empty_like(x)` 则是观念上的转折点：讲者说 "in Triton we're not thinking functionally anymore ... we have to explicitly read and write. So, there's no like returning value"——不再函数式地返回结果，而是显式读、显式写，所以必须先分配好输出张量让 kernel 往里写。

接下来是规模与切块。"this tensor is can be arbitrarily big ... I can't generally fit this all into one SM because it's just too big. Um so, I need to break it up into blocks"：张量可能任意大，一个 SM 装不下，必须切成 blocks。讲者口头说定义一个 "8,000 dimensional vector"、"the total number of elements is 8,000"，把 `BLOCK_SIZE` 先设成 `1024`，"and then I have eight blocks"；但画面 41:01 的变量查看框里显示的是 `num_elements = 8192`，而 $8192 = 8 \times 1024$ 正好对应 1024 的 block size 与 8 个 block。两者略有出入，读代码时应以实际的 `num_elements = x.numel()` 结果为准——这本身就是一个值得记下的"口述取整 vs 实际 2 的幂"的小坑。示意图用一串 `T` 把数组画成若干段，标上 `Block 0`、`Block 1`、`Block 2`、`Block 3`（示意只画了 4 段，实际是 8 段）。随后讲者说要 "use this sort of weird syntax to call the kernel"：kernel 名后面跟一对方括号，"this basically in brackets tells me essentially the shape of the grid"，即方括号里写 grid 有多少个 block（`num_blocks`）；"for every one of these blocks invoke this function"，并把 `X`、`Y`、`num_elements`、`BLOCK_SIZE` 传进去。

内核本体在 42:33 页展开，"this is probably the kind of the simplest kernel you can imagine"（讲者还说 "unfortunately I'm not going to be able to trace through this, so I'll just show you that what the code looks like"）。代码原文：

```python
@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # Input starts at `x_ptr`
    # Output starts at `y_ptr`

    # | T T T T T T T | T T T T T T T T | T T T T T T T T | T T T T T T T T |
    # |     Block 0   |     Block 1     |     Block 2     |     Block 3      |

    pid = tl.program_id(axis=0)        # Identifies the block
    start = pid * BLOCK_SIZE           # Starting index of this block

    # Indices where this thread block should operate
    offsets = start + tl.arange(0, BLOCK_SIZE)

    # Don't read/write past the end of the tensor
    mask = offsets < num_elements

    # Read
    x = tl.load(x_ptr + offsets, mask=mask)

    # Approx gelu is 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

和外面 PyTorch 里的 `x`、`y` 不同，这里的 `x_ptr`、`y_ptr` "are pointers, right? You can think about these are just like integers, they're addresses"——是指针/地址，"you have to get comfortable with that"。`num_elements` 与 `BLOCK_SIZE` 则是从调用处传进来的，其中 `BLOCK_SIZE: tl.constexpr` 标明它是编译期常量。

逐行读这个最简 kernel：`@triton.jit` 把函数交给 Triton 编译；`pid = tl.program_id(axis=0)` 取当前 block 的编号（注释 "Identifies the block"）；`start = pid * BLOCK_SIZE` 算出本块负责的起始下标（"Starting index of this block"）；`offsets = start + tl.arange(0, BLOCK_SIZE)` 生成块内这一整段索引；`mask = offsets < num_elements` 用来 "Don't read/write past the end of the tensor"，即处理最后一块不满 `BLOCK_SIZE` 的越界问题；`x = tl.load(x_ptr + offsets, mask=mask)` 才是真正从 HBM 读入，注意这里用的是指针算术 `x_ptr + offsets`，而不是 Python 下标。最后那行注释把 GELU tanh 近似原样写进代码：`0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`，与前面 30:28 那张 PyTorch 文档页上的公式完全对应。讲者特别强调 "for every block we're going to have this function being called"——同一份 kernel 代码会被每个 block 各执行一次，靠 `pid` 区分彼此，这正是上一章 "specify what each thread block does" 的具体化。

与前后内容的关系要交代清楚：本段在 [43:20] 处被截断，讲者正说到 "and what happens is the first thing I'm ..."，画面也停在 `tl.load` 之后的注释上——GELU 的实际计算与 `tl.store` 写回还没出现，那是下一段（继续讲 kernel、再到 PTX）的内容，不能提前补写。画面 41:01 上方还有几行灰色未激活的注释，已经预告了后续：可以看到 Triton 生成的 PTX 代码（`triton_gelu_ptx.txt`）、`%ctaid.x is block index, %tid.x is thread index`、`%f* are floating point registers, %r* are integer registers`，以及 "One thread processes 8 elements at the same time (thread coarsening)"，即后面会讲的 thread coarsening（线程加粗）。易错点也很集中：忘记 `mask` 会在最后一块越界读写；把 `x_ptr` 当张量用会错（它只是地址）；`BLOCK_SIZE` 必须是 `tl.constexpr` 这类编译期量；且 `num_blocks` 与 SM 数量的匹配（前面 wave quantization 讲过的 148 / 160 例子）依旧会影响 occupancy。局限上，这个 kernel 只是 element-wise 的向量化写法，尚不涉及线程间通信、shared memory 归约或 tiling——那些复杂度留到后面的 softmax 与 matmul 才登场。

![第一个 Triton kernel 的源码：@triton.jit、指针参数、tl.program_id、tl.arange、mask 与 tl.load](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0031_00-42-33.jpg)

## 第一个 Triton kernel 的脚手架：从 program id 到 mask [43:20 - 46:20]

上一段已经把 GELU 的启动配置搭好：在 Python 侧分配输出张量 `y`，把输入按 `BLOCK_SIZE` 切块，再用 `triton_gelu_kernel[(num_blocks,)](...)` 启动一个一维的 grid。这一段作者回到 kernel 内部，逐行回答一个看似朴素却决定一切的问题：一个线程块被唤醒之后，它怎么知道「我是谁、我该处理哪一段数据」。这里的 program ID（PID）就是线程块的身份标识，幻灯片上用一行 `@triton.jit` 装饰的 `def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):` 开场，紧接着两条注释 `# Input starts at 'x_ptr'` 与 `# Output starts at 'y_ptr'` 说明两个参数都是指针，也就是指向张量起始地址的内存位置，而不是张量本身。代码行号显示这段定义落在幻灯片里的第 361–386 行附近，前一行还有 `return y`（第 358 行），说明它紧接在上一段的 Python 包装函数之后。

幻灯片里画了一张很直观的 ASCII 示意图，用一排 `T` 表示元素或线程，再用竖线把它们切成四段：`# | T T T T T T T T | T T T T T T T T | T T T T T T T T | T T T T T T T T |`，下一行标注 `# |    Block 0    |    Block 1    |    Block 2    |    Block 3    |`。作者口述的对应关系是：PID 为 0 对应第一个块、为 1 对应第二个块、为 2 对应第三个、为 3 对应第四个，也就是 `pid = tl.program_id(axis=0)` 这条语句，注释写作 `# Identifies the block`。有了身份之后就要决定处理的数据区间，`start = pid * BLOCK_SIZE` 给出该块的起始下标，注释是 `# Starting index of this block`：PID 为 0 时 `start` 是 0，PID 为 1 时是 `BLOCK_SIZE`，PID 为 2 时是 `2 * BLOCK_SIZE`，依此类推。这一步本质上是把一维的块编号线性映射回全局数组的下标，也是所有 Triton kernel 共同的第一步。

接下来 `# Indices where this thread block should operate` 引出 `offsets = start + tl.arange(0, BLOCK_SIZE)`。`tl.arange(0, BLOCK_SIZE)` 概念上生成整数 $0, 1, \ldots, \text{BLOCK\_SIZE}-1$，加上 `start` 之后就得到该块要操作的全部全局下标。作者举了 block 1 的例子：其 `offsets` 依次是 `BLOCK_SIZE`、`BLOCK_SIZE + 1`，一直到 $2\times\text{BLOCK\_SIZE}-1$。这里 Triton 与 CUDA 的差别已经显现：CUDA 里通常是一个线程对应一个元素，而 Triton 的 `offsets` 是一个向量，一次描述整块数据的下标，因此可以把它想成「向量化版本的 CUDA」。但现实中元素总数未必恰好被 `BLOCK_SIZE` 整除，最后一个块往往只覆盖到张量末尾之前的一小段，于是幻灯片在 `# Don't read/write past the end of the tensor` 之下给出 `mask = offsets < num_elements`。这个 mask 在有效范围内为 true、越界之后为 false；对非末尾的块来说它全为 1（all ones），只有末尾块才出现 0，用来阻止越界读写。作者特别提醒，在 Triton 代码里几乎总能看到这种 masking，因为它处理的是「张量只到这里」的一般情形。

有了下标与掩码，剩下的就是读写与计算。幻灯片在 `# Read` 下给出 `x = tl.load(x_ptr + offsets, mask=mask)`，并在 `# Approx gelu is 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` 里写出 GELU 的 tanh 近似。由于 Triton 没有 `tl.tanh`，注释 `# Compute (tl.tanh doesn't exist, use tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)` 指明要用指数函数自行拼装，于是代码是 `a = 0.79788456 * (x + 0.044715 * x * x * x)`、`exp = tl.exp(2 * a)`、`tanh = (exp - 1) / (exp + 1)`、`y = 0.5 * x * (1 + tanh)`。其中 $0.79788456 \approx \sqrt{2/\pi}$，$0.044715$ 是近似公式里的三次项系数，这两个数值必须原样保留；把 `tanh` 改写成 $\frac{e^{2a}-1}{e^{2a}+1}$ 是因为 $\tanh(a)=\frac{e^{2a}-1}{e^{2a}+1}$。作者强调 `x_ptr` 是一个整数、指定了 X 所在的内存地址，加上 `offsets` 就是指针算术（pointer arithmetic），得到要读的前 `BLOCK_SIZE` 个元素的地址；被 mask 屏蔽的位置干脆不读。读进来的 `x` 可以当作一个向量来做常规计算。

算完得到的 `y` 与 `x` 同尺寸，最后用 `tl.store(y_ptr + offsets, y, mask=mask)` 写回，作者概括为「这个 load 从 HBM 读、做一些事、再写回 HBM」。这一章的易错点集中在三处：`BLOCK_SIZE` 被声明为 `tl.constexpr`，是编译期常量而不是运行时变量；`mask` 必须同时传给 `tl.load` 和 `tl.store`，只读侧加掩码而写侧不加会越界写；`tl.tanh` 并不存在，必须自己用 `tl.exp` 构造。它承接上一段的启动配置，同时把「输入指针、输出指针、唤醒、算下标、读、算、写」这套模板固定下来，后面讲 PTX、softmax、matmul 时都会反复复用这个骨架。

![Triton GELU kernel 前段代码：@triton.jit 定义、Block 0 到 Block 3 的 ASCII 分块示意图、pid 定位块、start 起始下标、offsets 与越界 mask](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0032_00-44-00.jpg)

![Triton GELU kernel 后段代码：tl.load 带掩码读取、GELU tanh 近似公式注释、用 tl.exp 拼装 tanh 的 a、exp、tanh、y 四行计算](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0033_00-45-22.jpg)

## 课堂问答：Triton 与 CUDA、tensor units 与内存层级 [46:20 - 50:39]

讲完第一个 kernel，作者停下来接受提问，这一段几乎全部由问答构成，却澄清了初学者最容易产生的几类误解。第一个问题是 Triton 和 CUDA 到底差在哪。作者的回答很坦率：对于 element-wise（逐元素）操作，两者看起来几乎一样，CUDA 甚至更简单，因为 CUDA 真的是「唤醒、找到线程、直接对那一个元素操作」，而 Triton 是它的向量化版本——你拿到的是一个 block（块），对这个块整体操作。真正的分水岭在后面：一旦要做的不只是逐元素运算，比如需要块内通信、归约或者矩阵乘法，CUDA 就会「a lot more annoying to work with」，因为程序员得手工管理 shared memory 与线程同步；Triton 把这一层抽象掉，代价是牺牲了对底层的直接控制。这也解释了为什么幻灯片上把 `In Triton (developed by OpenAI), specify what each thread block does.`（在 OpenAI 开发的 Triton 里，你指定每个线程块做什么）作为纲领，并用两条 bullet 补充：`Generally powerful enough (especially when getting started)`（一般足够强，尤其适合上手）与 `Conceptual framework: load data into shared memory, operate on it, write back to global memory`（概念框架是把数据加载进 shared memory、在其上运算、再写回 global memory）；而 CUDA 一侧则记着 `Cons: Need to manage more things (e.g., shared memory)` 这样的代价。

第二个问题关于 tensor units（张量计算单元）该如何使用。作者给了短答并把话题推迟：`you don't control that. Like the hardware figures out where to put things.`——也就是说程序员并不直接控制数据是否进入张量核心，由硬件决定把东西放在哪里、怎么调度；他说稍后展示代码究竟编译成什么时再回到这个话题。这个「不控制」很关键：Triton 之所以能比 CUDA 简洁，正是因为它把寄存器分配、shared memory 布局、是否用张量单元这类决策交给编译器与硬件。第三个问题更长，观众希望一步步走一遍从 HBM 到 shared memory 再到 register 的数据流。作者先给了一个让人意外的定性：从某种意义上说，刚才那段 Triton 代码「is in some sense a lie」——GPU 并不是真的在 GPU 上调用 Triton 库、逐条执行你看到的这些 Python 语句；这段代码是写给人看的，用来 specify（描述）计算。真正发生的是编译器把它改写成 PTX，再由 PTX 去做实际的工作，这一点会在下一章展开。

不过作者还是给出概念层面的图像：`X pointer` 是 HBM（高带宽内存）里的一个内存位置，`tl.load` 指定一段内存范围并取出与之对应的数据，放进一个局部变量 `x`；而这个 `x` 在实践中「generally a register or shared memory」（一般是寄存器或 shared memory），具体落在哪一层 Triton 会自己决定，幻灯片与口述都用 `Triton kind of figures out what to do there` 来描述这种自动决策。更微妙的是执行时机：作者坦承在线程层面「it's not really clear」，究竟是事先就把数据排布到 shared memory 和寄存器、还是到此刻才发生，这属于「way late」的细节；他还提到所有线程会「sitting idle to get memory to flush in」，也就是空等着数据从 HBM 灌进来，并强调不应「twiddling our thumbs just waiting for data to come in from HBM」（干等 HBM 数据）。当观众追问「when does this actually get executed? Doesn't this block?」时，作者没有当场回答，而是说等展示 PTX 时再回来，这个悬念正是第 4 章 warp 调度要补上的。

这一章还夹着两段可以直接引用的图上信息。其一是 `def triton_gelu_example():` 包装函数的实际内容：输入用 `x = torch.randn(8192, device=cuda_if_available())` 生成，长度是 $8192$ 而不是旧笔记里的 $8000$；调用 `y = triton_gelu(x)`；再用 `check_equal_1d(triton_gelu, naive_gelu)  # Check for correctness` 校验 Triton 版与朴素版一致，说明「先保证正确性、再谈性能」是这一讲一贯的做法。其二是作者对「所有 kernel 长什么样」的归纳（约在 50:15 之后）：一般都有 inputs、outputs，然后 wake up（唤醒）、figure out which index you're going to look at（确定看哪个下标）、read（读）、do some stuff（做一些计算）、write to HBM（写回 HBM）。这六步刚好是上一章 `pid → start → offsets → mask → tl.load → 计算 → tl.store` 的抽象版。

这里的易错点尤其值得记牢：把 Triton 源码当成 GPU 上逐条执行的指令是根本性误解，它只是计算的描述；不要假定某个中间变量一定在 register 或一定在 shared memory，那是编译器的自由；也不要指望手工指定 tensor units。就全篇结构而言，这一章既回收了上一章的 kernel 模板，又用「编译器会生成 PTX」把话题交给下一章，还用「线程空等 HBM」为后面 warp scheduler 隐藏延迟埋下伏笔，是从「会写 kernel」过渡到「理解 kernel 到底怎么跑」的转折点。

## 从 Triton 到 PTX：线程真正执行的指令 [50:39 - 55:08]

作者用一个链接把话题落到实处以 `https://github.com/stanford-cs336/lectures/blob/main/var/triton_gelu-ptx.txt` 给出编译产物，幻灯片上被黄色高亮的那句话是核心结论：`Triton compiles down to PTX (parallel thread execution), an assembly language for GPUs.`（Triton 会编译成 PTX，即 parallel thread execution，一种面向 GPU 的汇编语言），紧接着 `We can see the PTX code generated by Triton.`。PTX 是 NVIDIA GPU 的中间汇编语言（intermediate assembly language），它描述的不再是线程块，而是「一个线程实际在做什么」——作者特别说明 `not a thread block because that's been kind of compiled away`（线程块这一层已经被编译掉了）。幻灯片在 `Observations:` 下列出几条观察：`ld.global.* and st.global.* reads and writes from global memory`，以及 `%ctaid.x is block index, %tid.x is thread index`。口述里也对应提到 `LD global` 是从 HBM 装入寄存器，寄存器以 `R` 表示整数寄存器、`FR` 表示浮点寄存器，后面还有 `move zero into R5`、`move zero into R6` 之类的 `mov`，以及把寄存器乘以某个常数的乘法，最后在底部看到 global store 写回 HBM。这一段 PTX 才是真正在某个线程上执行的代码，Triton 只是它上面的一层。

打开的 GitHub 文件 `lectures / var / triton_gelu-ptx.txt` 里能看到更具体的指令（行号约 56–77）。最显眼的是两条带谓词的向量加载：`@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];` 与 `@%p2 ld.global.v4.b32 { %r5, %r6, %r7, %r8 }, [ %rd2 + 0 ];`，`v4.b32` 表示一次取回 4 个 32 位字，`@%p1`、`@%p2` 是谓词寄存器，对应前面 mask 生成的条件，因此被 mask 屏蔽时这条加载不会生效。它们前后夹着 `// end inline asm` 与 `// begin inline asm` 注释，还有 `mov.u32 %r5, 0x0;`、`mov.u32 %r6, 0x0;`、`mov.u32 %r7, 0x0;`、`mov.u32 %r8, 0x0;` 四条把零搬进寄存器的指令——正好对应口述里的「move zero into R5、R6」。后面 `.loc 1 383 37  // lecture_06.py:383:37` 这样的行把机器指令映射回 Python 源码位置，紧接着是八条形如 `mul.f32 %r25, %r1, 0f3D372713;` 的浮点乘法（目标寄存器从 `%r25` 排到 `%r32`），其中 `0f3D372713` 是一个 float32 十六进制立即数，数值上就是 GELU 公式里的 $0.044715$；随后 `.loc 1 383 41  // lecture_06.py:383:41` 之后又有 `mul.f32 %r33, %r25, %r1;` 这类把两个寄存器相乘的指令。这些指令把上一章 `a = 0.79788456 * (x + 0.044715 * x * x * x)` 里的常数与乘法一一落实到了寄存器层面。

最值得琢磨的是 thread coarsening（线程加粗）。幻灯片里那些一块一块的重复指令对应作者的解释：`this is one thread, but rather than processing a single element, it's actually processing eight elements`——一个线程并不只处理一个元素，而是处理 $8$ 个元素；编译器的判断是这个线程「pretty lightweight, it doesn't do that much」，于是「let's just try to thicken it up a little bit」（把它加粗一点）。前面两条 `ld.global.v4.b32` 各取 4 个 32 位值，合计正好 8 个寄存器 `%r1` 到 `%r8`，就是这一决策在 PTX 里的证据。另一个常见疑问是「这段 PTX 是不是每个线程各生成一份」，答案是否定的：它 `compiled once`，是同一段代码被每个线程各跑一遍，线程靠传入的身份来区分自己，口述里说这是 thread ID，幻灯片与 PTX 里则是 `%ctaid.x` 给出块索引（block index）、`%tid.x` 给出块内线程索引（thread index），分别回答「我在哪个块」「我是块里的哪个线程」。看 PTX 的意义正在于获得这种「what's going on underneath the hood」（底层到底在发生什么）的直观。

关于 PTX 的责任边界，作者说得清楚：它是编译器生成的，不是人平常会去写的东西；确实有人手写 PTX，通常是「think you're better than the compiler」，但 NVIDIA 的编译器「generally pretty mature」，一般不必这么做，反倒是一些不够成熟的加速器（less developed accelerators）有时「you just have to reach in and actually hand-hold a bit more」。更重要的限制是：PTX 里仍然有大量事情没有被指定，比如「which SMs things are operating on」（在哪些 SM 上运行）、warp 如何组织等等，这些「a lot of those are kind of hardware controlled」，你在 PTX 里根本看不到。幻灯片上那张 `def main():` 大纲（约第 13 行起）也把这一讲的骨架摆了出来：`Last lecture: high-level overview of GPUs and performance`、`This lecture: benchmarking/profiling + writing kernels`，以及 `review_of_gpus()`、`benchmarking_and_profiling()  # Where are the bottlenecks?`、`naive_vs_builtin_vs_compiled_gelu()  # Apply it to the GeLU example`；在 `# Write Triton kernels` 之下依次是 `triton_introduction()`、`triton_gelu_example()  # Elementwise operation`、`triton_softmax_example()  # Reduction (row fits in a block)`、`triton_row_sum_example()  # Reduction (row doesn't fit in block)`、`triton_matmul_relu_example()  # Tiling: use shared memory`。底部 `Summary:` 还列出：用编程模型（PyTorch、Triton、PTX）保证正确性；理解硬件（SMs、warps、occupancy、bank conflicts 等）来优化性能；用 Benchmark 理解 scaling；用 Profile 看「what's being executed for how long」；以及 `Triton: think in terms of thread blocks (read to shared memory, do stuff (fusion), write back HBM)`。

这一章的易错点有三类。第一，PTX 里的寄存器名（如 `%r25`、`0f3D372713`）与 Triton 源码里的变量名并非一一对应，不要把机器码里的常数当成新的算法参数，它只是 $0.044715$ 的浮点编码。第二，thread coarsening 的粒度（这里是 $8$ 个元素）由编译器自行决定，程序员既没有在 kernel 里写、也不应假设它固定不变。第三，PTX 并不描述 warp 调度与 SM 分配，因此「这段代码何时真正执行、遇到 load 会不会阻塞」无法从 PTX 本身读出，而这正是下一章要回答的。就前后关系看，本章兑现了上一章「编译器会生成 PTX」的承诺，也把第 2 章埋下的两个悬念（tensor units 由硬件决定、线程空等 HBM）推进到了指令层面，同时为随后讨论 warp 调度与 softmax 的归约做好铺垫。

![Triton 编译到 PTX 的幻灯片：黄色高亮句 Triton compiles down to PTX、GitHub 生成的 PTX 链接、ld.global 与 st.global 观察、%ctaid.x 与 %tid.x 说明，以及 triton_gelu_example 校验代码](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0037_00-50-47.jpg)

![GitHub 上 triton_gelu-ptx.txt 的 PTX 汇编：ld.global.v4.b32 向量加载、mov.u32 置零、.loc 源码行映射、mul.f32 与 0f3D372713 浮点常数乘法](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0038_00-52-08.jpg)

## warp 调度隐藏 HBM 延迟与后续示例路线图 [55:08 - 58:06]

这一段先由一位观众用 CPU 作类比，把第 2 章「线程空等 HBM」的悬念彻底补上。观众的说法是：当一个 warp 被调度到某个 SM 上，执行到 `tl.load`（他口误说成 TF.load）时，几乎就像一次 CPU trap call，接下来只是在等某件事发生、自己在「twiddling my thumbs」（搓手指干等）；这时另一个 warp 会被调度上来覆盖它、去做运算；等 `tl.load` 完成，再把原来那个 warp 重新调度回来继续往下走。作者明确回应 `that's right`，并复述了一遍机制：Triton 代码里像 `load` 这样的语句会 `block for some number of cycles`（阻塞若干周期），它运行在某个 thread、某个 warp、某个 SM 上，而「remember the SM is running multiple warps at the same time」（SM 同时驻留并运行多个 warp），所以到达阻塞点时 SM 可以「find another warp to run」，等数据就绪后「the warp scheduler comes back and takes over」。这正是 GPU 用零成本上下文切换隐藏内存延迟的核心：HBM 访问很慢，但只要 SM 上有足够多可运行的 warp，等待时间就能被别的 warp 的计算填满，而不是让整个 SM 闲置。它同时回答了第 2 章那个「Doesn't this like block?」的追问——会阻塞，但阻塞的是 warp，不是 SM。

紧接着有个诚实的留白：观众问为什么每个 SM 上会有四个 warp、为什么要四个 warp scheduler，作者直接说 `I don't know exactly the reason behind that`（我不太清楚确切原因）。这类问题在讲课时没有被给出权威解释，做笔记时也不该替他编一个「因为吞吐最优」之类的理由，记住「作者未作答」本身就够了。这也提醒我们，PTX 里看不到的 warp 组织与 SM 分配，恰恰属于硬件实现细节，连讲课人也不一定给出确定答案；程序员能依赖的只是「SM 上有多个 warp、load 会阻塞、scheduler 会切换」这一层抽象。从全篇看，本章是 element-wise GELU 与底层 PTX 两条线索的收束：先解释清楚「代码怎么跑」，再转向「接下来还能写什么 kernel」。

于是作者给出后续示例的路线图，作为从 GELU 过渡到复杂算子的桥梁。他说「now let's go through some other examples, maybe just as a kind of a quick preview」，还要做三个例子。第一个是 GELU，虽然计算本身「has a lot of messiness」（指 tanh 近似那一串公式），但因为只是 element-wise，在本讲的语境里「actually very simple」，概念上最简单，也就是刚讲完的这个。第二个是 softmax，从这里开始「you're going to have to do reduction」（必须做归约），并且要分两种情形讨论：先考虑「row fits on a block」（一行能放进一个 block），再考虑「row doesn't fit on a block」（一行放不进一个 block）。第三个一路上升到 matmul（矩阵乘法）。作者点明动机：`hopefully by that point, you'll have all the ingredients that you need to do the assignment and implement flash attention`（到那时你应该就具备了完成作业、实现 flash attention 所需的全部要素）。他还把当前位置定性为「so far we looked at element-wise operations」，而接下来要处理的是「operations that aggregate over multiple values」——即需要把多个值聚合起来的运算，这正是从逐元素走向归约与 tiling 的分界线。

幻灯片上那张 `def main():` 大纲此时把高亮行从 `triton_gelu_example()  # Elementwise operation` 下移到了 `triton_softmax_example()  # Reduction (row fits in a block)`，用一屏把整个进阶路径写得很清楚：`benchmarking_and_profiling()  # Where are the bottlenecks?` 负责定位瓶颈，`naive_vs_builtin_vs_compiled_gelu()  # Apply it to the GeLU example` 把方法用到 GELU 上；`# Write Triton kernels` 之下四个示例依次标注 `Elementwise operation`、`Reduction (row fits in a block)`、`Reduction (row doesn't fit in block)`、`Tiling: use shared memory`。这四行注释其实是一份难度梯度表：从不需要块内通信，到需要行内归约但一行装得下，再到一行装不下需要循环 tile，最后到需要 shared memory 与 `tl.dot` 的矩阵乘法。它与作者口述的「GELU → softmax → matmul → flash attention」完全对应，也只是本段的「预览」，真正的实现从下一章 softmax 才开始。

本章的易错点在于几个容易想当然的地方。其一，不要把 `tl.load` 理解成立即返回的普通 Python 表达式，它在硬件上是一条会阻塞若干周期的全局内存指令；其二，不要以为一个 SM 一次只跑一个 warp，正是「多 warp 驻留 + scheduler 切换」才让 HBM 延迟得以隐藏，这与第 2 章「所有线程 sitting idle 等内存 flush」并不矛盾——那是单一 warp 视角，这是 SM 调度视角；其三，关于「为什么是四个 warp scheduler」这类硬件细节，讲义没有给出答案，不要自行补全因果。就承上启下而言，本章把第 3 章「PTX 未规定 warp 与 SM」的空白填成了可理解的调度图像，也用路线图把读者的注意力从 GELU 引向后面两章的 softmax。

![lecture_06.py 的 def main() 大纲，黄色高亮 triton_softmax_example() # Reduction (row fits in a block)，列出 GELU、softmax、row sum、matmul 四个 Triton 示例与 Summary 要点](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0042_00-57-33.jpg)

## softmax 的朴素实现与 Triton 启动配置 [58:06 - 62:18]

路线图之后进入第一个非逐元素算子。作者先复习 softmax 的定义：把它当作一个矩阵来看，「you exponentiate and normalize each row of a matrix」，即对每一行取指数再归一化，它既用在 attention 里，也用来生成输出概率（generating output probabilities），是「generally a good thing to do」的基础操作。写成公式，对一行 $x$ 的第 $i$ 个元素有
$$
\text{softmax}(x)_i = \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)},\qquad m = \max_j x_j ,
$$
其中先减去该行最大值 $m$ 纯粹是为了数值稳定性（numerical stability），避免 $\exp$ 溢出；它不改变结果，因为分子分母会同比例缩放。作者强调「everyone should be familiar with what a softmax is doing」，所以重点不在数学，而在「这段朴素 PyTorch 到底产生了多少次 HBM 读写」。

幻灯片在 `def naive_softmax(x: torch.Tensor):` 下给出了完整实现，而且每行都带读写计数注释，这些注释是画面独有、口述没有念到的关键增量。代码是：`# M: number of rows, N: number of columns` 之后 `M, N = x.shape`（这一行被黄色高亮）；`# Compute the max of each row (MN reads, M writes)` 对应 `x_max = x.max(dim=1)[0]`；`# Subtract off the max (MN + M reads, MN writes)` 对应 `x = x - x_max[:, None]`；`# Exponentiate (MN reads, MN writes)` 对应 `numerator = torch.exp(x)`；`# Compute normalization constant (MN reads, M writes)` 对应 `denominator = numerator.sum(dim=1)`；`# Normalize (MN reads, MN writes)` 对应 `y = numerator / denominator[:, None]`；最后 `return y`（这一行在另一张截图里被黄色高亮）。把注释累加，幻灯片给出精确结论：`# Total: 5MN + M reads, 3MN + 2M writes`，并对比 `# In principle, should have MN reads, MN writes (speedup of 4x!)`。也就是说，对一个 $M\times N$ 矩阵，朴素实现读了 $5MN+M$ 次、写了 $3MN+2M$ 次，而理论下界只是读 $MN$ 次、写 $MN$ 次，理论上能有约 $4\times$ 的加速空间。口述里把数字说成「five MN reads, three MN writes」，画面上的 $5MN+M$ 与 $3MN+2M$ 才是把归约量 $M$ 也算进去的精确版本。

为什么朴素版这么浪费？作者解释：这是 `plain PyTorch`，每一个操作（求 max、减 max、`torch.exp`、求和、相除）都是一个不同的 kernel、一个不同的算子，除非你调用 `torch.compile`，否则「each operation is going to read and write, read and write from HBM」——每个算子都要从 HBM 读一遍、算完再写回一遍，中间结果反复往返。这正是前面 GELU 案例里 kernel fusion（算子融合）动机的重演，只不过 softmax 多了归约。幻灯片上半部分还保留了包装函数与校验：`one program instance (pid = row index)` 的图示（一个 input matrix 里标出 row 3）、`y2 = triton_softmax(x)`，以及 `# Check our implementations are correct` 之下的 `check_equal_2d(pytorch_softmax, naive_softmax)` 和 `check_equal_2d(pytorch_softmax, triton_softmax)`，说明 Triton 版会同时和 PyTorch 内置版、朴素版比对正确性。这里 `pid = row index` 一句话已经预告了块划分方式：一个 program instance（线程块）对应一行。

于是 Triton 版沿用第 1 章的模板，但把「块」重新定义成「行」。幻灯片在 `def triton_softmax(x: torch.Tensor):` 下给出启动侧代码：`# Allocate output tensor` 对应 `y = torch.empty_like(x)`；`# Determine grid` 之下 `M, N = x.shape  # Number of rows x number of columns`；被黄色高亮的是 `block_size = triton.next_power_of_2(N)  # Each block contains all the columns`，即把列数取到下一个 2 的幂，作者口述时半开玩笑说「go to the next power of two for good luck」，好处是让一个 block 恰好装下整行的所有列；再往下 `num_blocks = M  # Each block is a row`，然后 `# Launch kernel` 调用 `triton_softmax_kernel[(M,)](x_ptr=x, y_ptr=y, x_row_stride=x.stride(0), y_row_stride=y.stride(0), num_cols=N, BLOCK_SIZE=block_size)`。注意 grid 是一维的 `(M,)`，块数就是行数 $M$，作者现场问「how many blocks are there?」并自答「M, right? One for each row」。传给 kernel 的除了输入指针 `x_ptr`、输出指针 `y_ptr`，还有 `x_row_stride` 与 `y_row_stride` 这两个 stride（步幅），用来描述「how far to move down」（向下移动一行要走多远），row stride 基本就等于列数；`num_cols=N` 与 `BLOCK_SIZE=block_size` 则把形状信息交给 kernel。

为什么敢让「一行一个块」？作者的论证是：softmax 不是 element-wise，而是「sort of row-wise」，每一行内部要 normalize 和 sum，但不同行之间「the blocks don't interact」，而且「There's no shared memory across blocks」（线程块之间没有共享内存），所以行与行天然独立、无需跨块通信，一个块包办一整行正好。这也解释了为什么这一版叫「row fits in a block」：它默认一行能塞进 `BLOCK_SIZE`，下一章的 kernel 里甚至会写 `assert num_cols <= BLOCK_SIZE` 来把前提显式化。本章的易错点包括：朴素实现的读写计数必须把归约量 $M$ 计入（$5MN+M$ 与 $3MN+2M$），只记 $5MN$、$3MN$ 就与画面不符；`triton.next_power_of_2(N)` 得到的 `block_size` 与真实的 `num_cols = N` 并不相等，越界列要靠 mask 处理；stride 若填错（例如误用列 stride）会读到错误的行，而 `x.stride(0)` 必须来自实际张量而非硬编码。就前后关系而言，本章把第 1 章的「输入/输出指针 + 唤醒 + 下标 + 读 + 算 + 写」模板从逐元素推广到行级归约，用精确的读写计数说明了融合的收益，并用 `pid = row index` 和启动参数把舞台交给下一章的 kernel 本体。

![naive_softmax 完整代码与逐行读写计数注释：5MN + M reads、3MN + 2M writes，对比理想的 MN reads、MN writes 与 4x 加速](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0044_01-00-16.jpg)

![triton_softmax 启动配置：torch.empty_like 分配输出、triton.next_power_of_2(N) 求 block_size、num_blocks = M、以及带 x_row_stride 与 num_cols 的 kernel 调用](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0045_01-01-37.jpg)

## softmax kernel 本体：行内归约与边界问答 [62:18 - 65:00]

最后这一章落到 kernel 内部。屏幕上的代码以 `@triton.jit` 开始，签名是 `def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):`，第一句就是 `assert num_cols <= BLOCK_SIZE`。这条断言把上一章「row fits in a block」的假设写成了硬约束：本版本只在列数不超过块大小时成立，超出就必须换用后面要讲的 tile 循环版本。紧接着 `# Process each row independently`（每行独立处理）之下是 `row_idx = tl.program_id(0)` 与 `col_offsets = tl.arange(0, BLOCK_SIZE)`：前者让唤醒的块知道自己在哪一行，后者给出 $0, 1, \ldots, \text{BLOCK\_SIZE}-1$ 的全部列偏移，作者口述时说这会给出「all the columns from zero to the number of block size」，并假设它覆盖了所有列。与第 1 章 GELU 的 `pid = tl.program_id(axis=0)` 对照，骨架完全同构，只是这里的 `pid` 语义是行号而非元素块号。

`# Read from global memory` 之下是地址计算三连：`x_start_ptr = x_ptr + row_idx * x_row_stride`、`x_ptrs = x_start_ptr + col_offsets`、`x_row = tl.load(x_ptrs, mask=col_offsets < num_cols, other=float("-inf"))`。作者解释得细致：`row_idx * x_row_stride` 用 row stride 定位到该行的起点（`the start of my data plus the which row and the row stride`），`x_ptrs` 就是「the addresses of all the data I need to load」，即需要加载的全部地址。这里最重要的细节是 `other=float("-inf")`：被 mask 屏蔽的列不读，而是填入负无穷。为什么是 $-\infty$ 而不是 0？因为接下来要对整行取 $\exp$ 再归一化，$\exp(-\infty)=0$，这些位置对分子分母都贡献 0，等价于「a zero for the softmax operation」（softmax 里的零贡献）。若误填成 0，$\exp(0)=1$ 反而会污染归一化常数，这是实现 softmax 时最典型的一个坑。

`# Compute` 之下的四行与朴素版一一对应，作者说「this part is essentially the same as the naive softmax」：`x_row = x_row - tl.max(x_row, axis=0)` 减去行内最大值（数值稳定性，且 $-\infty$ 减去有限最大值仍是 $-\infty$，不会影响 `tl.max` 的结果）、`numerator = tl.exp(x_row)` 逐元素取指数、`denominator = tl.sum(numerator, axis=0)` 求归一化常数、`y_row = numerator / denominator` 相除得到概率。注意 `tl.max(..., axis=0)` 与 `tl.sum(..., axis=0)` 都是对块内这一维做归约，这正是从 element-wise 迈向 reduction 的实质一步：GELU 的每一步都是逐元素，softmax 却必须把一个向量压成标量再广播回去。之后 `# Write back to global memory` 之下是 `y_start_ptr = y_ptr + row_idx * y_row_stride`，用输出侧自己的 `y_row_stride` 定位写回位置，再配 `mask` 把 `y_row` 存回去。作者把整体概括成「compute the max, subtract it off, exponentiate, sum, and divide, and then I write it back」，与第 1 章「load 从 HBM 读、算、写回 HBM」是同一句话。

讲完代码后是一组边界问答，把这一版的适用条件说清楚。有观众确认「this is the version where each block can just span the entire row」，作者肯定，并感叹「Triton makes this very easy, right? Because this is as if you were just writing normal PyTorch code」——只要一行能放进一个 block，写 Triton kernel 几乎就像写普通 PyTorch，这也是 Triton 抽象的价值所在。另一位观众问「what if my number of columns and number of rows are bigger than block size?」，作者没有在本段展开，只说「we'll get to that. Now, if the number of columns, rows, in general, it's going to be much larger than that block size, so we'll come back to that」，把超长行留给了后续的 `triton_row_sum_example()` 一类 tile 循环版本。最后有人问能否「softmax by column」（按列做 softmax），作者的初步判断是「I think that should be fine because here we're tracking these pointers, right? So the pointers can be anything」，即因为地址完全由指针算术控制，指针可以指向任意布局，按列归约原则可行；这句话在 65:00 处尚未说完，本段到此结束，做笔记时不宜替他补出结论。

本章的易错点与局限值得逐条记下：`other=-float("inf")` 必须在 `tl.max` 之前填入且只能用于指数之前的输入，填 0 会得到错误结果；`assert num_cols <= BLOCK_SIZE` 意味着本 kernel 不能处理 $N > \text{BLOCK\_SIZE}$ 的长行；`BLOCK_SIZE` 取自 `triton.next_power_of_2(N)`，是编译期 `tl.constexpr`，因此 `col_offsets < num_cols` 的 mask 必须覆盖多出来的那些 2 的幂列；`tl.max` 与 `tl.sum` 的 `axis=0` 是块内归约维度，写错维度会把多维张量压错方向；输出侧要用 `y_row_stride` 而不是想当然复用输入 stride。就全段结构而言，这一章把第 1 章的 kernel 模板、第 5 章的启动配置与读写计数全部收拢成一份可运行的 kernel，完成了「从逐元素到行内归约」的跨越；它与前面 GELU 案例共享同一套「指针 + 唤醒 + 偏移 + 掩码 + load + 计算 + store」骨架，差别只在计算段引入了归约，而作者留下的两个开放问题（超长行、按列 softmax）正好指向本讲后续的 tile 循环、matmul 与 flash attention。

![triton_softmax_kernel 完整实现：assert num_cols <= BLOCK_SIZE、row_idx 与 col_offsets、带 other=float(-inf) 的 tl.load、tl.max/tl.exp/tl.sum 行内归约与 y_start_ptr 写回](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0046_01-02-58.jpg)

## 从列访问收尾到「一行放不进 block」的问题 [65:00 - 66:47]

本段开头承接上一段行内 softmax 的讨论。65:04 讲者在回答「如果想沿列方向做 softmax 该改哪里」时说："I've um basically I think all you would have to do is change the stride uh here. Um to basically access the columns. Actually, it would be here. The column offsets, you would just make um like the multiply this by like the row stride, I think." 直觉是：Triton kernel 里 `tl.arange(0, BLOCK_SIZE)` 生成的是一串偏移，若这串偏移代表列号，那么沿列遍历（按列 softmax）时就要把「列偏移」乘上**行** stride 才能跨行取到同一列；反之按行遍历时乘的是列 stride。这背后是一个贯穿整讲的事实：tensor（张量）在 HBM 里被 linearized（线性化）成一维数组，任何多维索引最终都要折算成一个线性地址。画面上方还残留着上一讲 softmax kernel 的尾部代码，可原样抄下作为对照：

```python
numerator = tl.exp(x_row)
denominator = tl.sum(numerator, axis=0)
y_row = numerator / denominator

# Write back to global memory
y_start_ptr = y_ptr + row_idx * y_row_stride
y_ptrs = y_start_ptr + col_offsets
tl.store(y_ptrs, y_row, mask=col_offsets < num_cols)
```

65:23 一句 "Okay, let's let's move on" 之后，讲者抛出一个更接近真实规模的新问题："suppose that your row doesn't fit into a block. Okay, so what do you do in this case?" 他给的具体数字是「一行 4000 列」（口述先说 400、随即改口 4,000），而 block size 只有 1024，"So, you can't cons uh you know, you have to do something here." 幻灯片上写的是更规整的版本 `Example: 4096 columns, but block size is 1024...`，即 4096 列、block size 1024，恰好 $4096/1024 = 4$，切成四个 tile（瓦片／分块）。讲者给出的策略是 "We're going to break up the row into tiles. So, in this case, there's going to be four tiles and each thread is going to iterate over the tiles and accumulate a sum. And then finally at the end, um we're going to do the the kind of the uh reduction." 画面上的 Strategy 三条把它写得更精确：`Break up row into tiles (4 in the example above)`、`Each thread iterates over tiles and accumulates a sum`、`Do final reduction (sum) over accumulators of each thread (shared memory or warp shuffles)`；最后一行还提示 `Consider the simpler example (row sum instead of softmax)`。

![幻灯片定格在 triton_row_sum_example：4096 列对 1024 的 block size，以及切成四个 tile、逐 tile 累加、最后跨线程归约的 Strategy 三条](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0048_01-05-46.jpg)

于是 66:20 讲者明确 "I'm sort of switching from softmax to row sum um because it's just easier to think about"，把例子的主角换成 row sum（行求和）。它的定义很朴素：给一个矩阵，逐行把元素加起来，输出一个长度等于行数的向量，写成公式就是

$$y_i = \sum_{j=0}^{N-1} X_{i,j}$$

其中 $N$ 是一行的元素个数。与 softmax 的关键差别在于：softmax 要先求 max、减 max、取 exp、再求和、再相除，是「多阶段归约」；row sum 只有一次加法归约，累加器（accumulator）的语义最干净。换句话说，本段真正要引入的复杂度不是某个非线性函数，而是「线程块内部出现 for 循环」这一结构性变化。66:32–66:47 讲者顺带确认内置 row sum "does uh what you would expect"——"the row sum operation just basically takes um a a matrix and then computes uh the sum of each row"，没有 surprises。这句看似闲笔，其实是在给后面自写 kernel 定一个正确性基准：只要自写的 `row_sum_kernel` 与内置结果一致，就说明 tile 拆分、mask、累加、最终归约这条链路没有写错。

与前后内容的关系：上一段（58:06–65:21）的前提是「整行能放进一个 block」，归约在一个 block 内由 Triton 自动完成，因此写法几乎像 PyTorch；本段起这个前提被打破，后面 66:47–71:53 全部围绕「如何在 block 内循环 tile」展开，而 71:53 之后的 matmul 会把一维 tile 循环升级为二维 tiling。易错点有两类。其一是 tile 与 block 的区别，讲者要到 71:06 才正式澄清，但此刻就该埋下伏笔：这里的 4 个 tile 不是 4 个线程块，而是同一个线程块依次处理的 4 段数据，线程块总数仍等于矩阵行数。其二是数字口径：转写口误先说 400 又说 4000，幻灯片写 4096，而下一节的示意图用的是 `n_cols = 10`、`BLOCK_SIZE = 4` 的玩具例子，三者分别对应「真实规模」「整块规模」「教学示意」，不能混用。此外按列 softmax 的 stride 改法讲者带着 "I think" 的犹豫，本笔记只保留「把列偏移乘行 stride」这一层直觉，不去臆测未显示的具体代码。

## tile 循环的数值直觉：每个线程一个累加器 [66:47 - 68:47]

66:47 起讲者给出概念框架："each block is still in charge of one row. So, that part hasn't changed." 也就是「一行 ↔ 一个 block」的映射与 softmax 完全相同，改变的只是 block 内部的动作。他让自己扮演 "block one, row one"：这一行按列切成若干 tile，tile zero 是第 0–3 列，tile one 是第 4–7 列，tile two 是第 8–11 列。每个线程手里握着一个 accumulator，先把第一个 tile 的元素加进去，再移动到下一个 tile，把当前元素继续加到同一个累加器上。若第 $t$ 个 tile 覆盖列 $[tB,\,(t+1)B)$，$B$ 为 BLOCK_SIZE，则线程 $c$ 的累加器满足

$$acc_c \;\leftarrow\; acc_c + \sum_{t} X_{i,\,tB+c}$$

最后再对所有线程的 $acc_c$ 做一次归约得到该行的和。这个「先分块逐 tile 累加、再跨线程归约」的两阶段结构，是本章与下一章全部代码的骨架。讲者的口述是："each thread keeps a kind of accumulator and processes the first tile and it's going to move on to the next tile and it's going to add the current element to the accumulator."

转写里的口算例子与示意图完全对应，可以把图上独有的数字全部钉死。图左侧标注 `input matrix n_cols = 10`，并把 `row 0 → block 0`、`row 1 → block 1`、`row 2 → block 2`、`row 3 → block 3` 一一对应；虚线框内是 `block 1 (pid = 1, processes row 1)`，框里画了 4 个线程 t0–t3，即 $B=4$。`tile 0 (cols 0–3)` 的四个值是 $x=3$、$x[1]=1$、$x[2]=4$、$x[3]=1$，于是寄存器（registers）里的累加器变成 `acc=3`、`acc=1`、`acc=4`、`acc=1`；`loop: start += BLOCK_SIZE` 之后进入 `tile 1 (cols 4–7)`，值为 5、9、2、6，累加器变成 `acc=8`（3+5）、`acc=10`（1+9）、`acc=6`（4+2）、`acc=7`（1+6）；再进入 `tile 2 (cols 8–11, mask cols 10–11)`，只有 $x[8]=5$、$x[9]=3$ 有效，t2、t3 被标成 `masked`。最后 `tl.sum(acc) → tree reduce across threads` 给出 `out[1] = 39`，图上还把算式原样写成 $3+1+4+1+5+9+2+6+5+3$。

![block 1 处理 row 1 的 tile 累加示意图：n_cols=10、四个线程 t0–t3、三个 tile、masked 列与 out[1]=39 的树形归约](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0049_01-07-19.jpg)

把这段示意图翻成可核对的代码，就是下面这个纯 Python 复现（数值与图一致）：

```python
# kf_0049: n_cols = 10, BLOCK_SIZE = 4, 三个 tile = cols 0-3 / 4-7 / 8-11
tile0 = [3, 1, 4, 1]
tile1 = [5, 9, 2, 6]
tile2 = [5, 3, 0, 0]          # cols 10-11 被 mask，等价于补 0
acc = [a + b + c for a, b, c in zip(tile0, tile1, tile2)]   # [13, 13, 6, 7]
out_1 = sum(acc)                                            # 39
```

注意 masked 位置不是「跳过整条线程」，而是让该位置贡献 0，这与上一讲 softmax 里把 mask 位置填 `-float("inf")` 形成漂亮对照：softmax 求的是 $\max$ 与 $\exp$ 的和，被掩掉的元素应当贡献零概率，所以填 $-\infty$；row sum 求的是加法，被掩掉的元素应当贡献加法单位元 0，所以后面代码里写 `other=0.0`。同一个 `mask` 机制，因归约的代数结构不同而选择不同填充值，这是最容易写错的地方。另一个细节：讲者口述 "I'm putting 3 1 4 1 and then the second loop iteration, I'm going to add five to this, I get eight. I add nine to this, I get 10, two to four, um and six to one" 中的 "two to four" 与 "six to one" 是口语略写，按图应理解为「把 2 加到当前 acc=4 上得 6」「把 6 加到当前 acc=1 上得 7」；到 68:00 他说 "for tile two, um I add five and three to their respective accumulators"，正好对应 tile 2 只有两个有效元素。

68:08 之后 "I have this um you know, a vector of accumulators and then that is actually uh you know, that I can just like sum up"——即先得到一个长度为 BLOCK_SIZE 的向量，再压成一个标量。这个「向量 → 标量」的压缩，就是后面代码里 `tl.sum(acc, axis=0)` 的位置。与前文的联系是：softmax 版本里 `tl.max(x, axis=0)`、`tl.sum(num, axis=0)` 已经演示过 block 内归约，本章只是把「一次归约」放大成「先循环累加、再归约」；与后文的联系是 68:21 讲者说 "let's see what the code looks like here. Um so I'm just going to call the the row sum kernel"，于是下一章进入 `row_sum_kernel` 的逐行实现。局限方面也要说清：这个玩具例子把 $N=10$、$B=4$、3 个 tile 画得很直观，却掩盖了两件事——一是真实情况下累加器究竟落在寄存器还是 shared memory 由编译器决定，二是 `mask` 只影响最后一个不满 tile、但循环次数仍由 `range(0, N, BLOCK_SIZE)` 决定。图中 `tree reduce across threads` 还提示最终归约是树形跨线程操作，并非某个线程串行求和，Triton 会自动选择实现，程序员不显式写同步。

## row_sum_kernel 代码、累加器驻留与「tile ≠ block」 [68:47 - 71:53]

68:42 起讲者正式读代码。他先定义两个量：$N$ 是「这一行有多少个元素」，BLOCK_SIZE 是「一个 tile 的大小」，并强调 "remember block size is the number of threads. I'm processing more data than larger than the block size. I just have to do it iteratively now."——BLOCK_SIZE 同时是线程数、也是每次循环能并行加载的元素数。于是 `start` 依次取 $0,\ B,\ 2B,\ \dots$（"start is going to go from zero to block size, to two times block size, and so on and so forth"），每一步算出该 tile 的 offsets，`tl.load` 从 HBM 取数并加到累加器，循环结束后对累加器求和得到标量再写出。代码画面上的注释还给出一组更明确的参数：`One row: T2 T3 T4 | T1 T2 T3 T4 | T1 T2 T3 T4 (N = 12, BLOCK_SIZE = 4)`，即 12 个元素、4 个线程、3 个 tile。下面这段是画面上的 kernel 原文：

```python
@triton.jit
def row_sum_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)  # Which row are we processing?

    # Accumulator for each thread
    # One row: T2 T3 T4 | T1 T2 T3 T4 | T1 T2 T3 T4 (N = 12, BLOCK_SIZE = 4)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Loop over tiles
    for start in range(0, N, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)
        acc += x

    # Final reduction from BLOCK_SIZE (all threads) to a scalar
    result = tl.sum(acc, axis=0)

    tl.store(out_ptr + row, result)
```

![Jupyter 中的 row_sum_kernel 完整源码：tl.program_id、tl.zeros 累加器、range(0, N, BLOCK_SIZE) 循环、tl.sum 与 tl.store](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0050_01-08-52.jpg)

逐行看：`row = tl.program_id(0)` 表明一个 program（即一个线程块）对应一行，与 softmax 一致；`acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)` 建的是长度为 BLOCK_SIZE 的**向量**累加器，注释写 "Accumulator for each thread"，每个线程位置一个槽；`for start in range(0, N, BLOCK_SIZE)` 是本段新增的循环；`cols = start + tl.arange(0, BLOCK_SIZE)` 生成当前 tile 的全局列号，`mask = cols < N` 处理最后一个不满块，`other=0.0` 把越界元素置零；`acc += x` 逐 tile 累加；循环外 `result = tl.sum(acc, axis=0)` 完成「向量 → 标量」的最终归约（注释 "Final reduction from BLOCK_SIZE (all threads) to a scalar"），最后 `tl.store(out_ptr + row, result)` 每行只写一个标量。讲者特别点出 "this is a little bit more complicated than before because now we have a for loop within a thread and this is, uh, necessary when your data doesn't fit within a block"——for 循环出现在单个 program 内部，是数据放不下 block 时的必然代价，也是 Triton 写法开始不像 PyTorch 的起点。

70:30 的问答把累加器的物理位置挑明了。学生问能否控制 accumulator 放在哪里，讲者答："At least in this Triton program, you don't explicitly say and this is up to the, uh, Triton compiler to, uh, figure out where to put it. But in general, if the block size is large enough, it has to go in shared, uh, shared memory." 也就是说 Triton 不让你像 CUDA 那样显式声明 `__shared__`，由编译器根据 BLOCK_SIZE 决定放 registers 还是 shared memory（共享内存）；BLOCK_SIZE 足够大时必须是 shared memory。这与 69:40 的口述 "this accumulation is going to be, um, either in registers or, uh, shared memory" 呼应。易错点：不要把 `acc = tl.zeros(...)` 理解成「每个线程一个标量」或「全局内存里的数组」——它是块内的向量值，Triton 语言层面只描述值、不描述存储层级；这既是 Triton 比 CUDA 省事之处，也是它比 PyTorch 更难精确控制之处。

70:57–71:40 讲者专门做了一次概念澄清，因为学生容易把本段与前面的 GELU 混起来："Remember in uh, gelu, we also split a row into a bunch of, um, pieces, but those were blocks, right? And so, each of those pieces was a block that was processed independently. These are not blocks. These are tiles. The block corresponds to this whole, uh, row. And and has to basically process all the tiles." 对照很干脆：GELU 里一行被切成很多 piece，每个 piece 是一个独立 block，由不同 program 并行处理；这里整行仍是**一个** block，tile 只是这个 block 内 for 循环的一轮数据。因此 "this is where it starts to not look like PyTorch because, uh, you're not able to process all your data in one nice kind of not everything fits into shared memory"——数据无法一次性全部摊进 shared memory，PyTorch 那种「把整个张量当成一个大算子」的心智模型失效了。71:44 学生追问、讲者确认 "let's say uh, the cumulator is stored in shared memory"，把上一节的推断落到具体假设上。

结构上，本章是上一段「一行放不进 block 的策略」与下一段「matmul 正式 tiling」之间的桥梁：这里出现的是讲者在 83:59 总结里称之为 "baby tiling"（kf_0059 画面上写作 `row sum (baby tiling)`）的一维 tile 循环，只沿列方向切；到 matmul 会升级成同时沿 $M$、$N$、$K$ 三个方向切块的正式 tiling。局限同样清楚：row sum 只做一次加法归约，累加满足结合律，逐 tile 累加与一次性求和在数学上等价（浮点舍入误差另论）；一旦换成 softmax 这类需要 max 的非线性归约，就不能把 `acc += x` 简单搬过去，而要维护 running max 与缩放因子。此外 `x_ptr + row * N + cols` 把行 stride 直接写成 $N$，隐含「矩阵行连续、行长恰为 $N$」的假设；若输入是转置或非连续张量，就必须像 79:03 之后那样显式传入 stride 参数，这正是 matmul kernel 要处理的问题。

## matmul 的朴素做法与算术强度瓶颈 [71:53 - 75:14]

71:53 讲者宣布进入 "our finale, which is matmul"，并说 "matmul of of course is the bread and butter of deep learning. It's been optimized to death"——矩阵乘法是深度学习最基础、也被优化得最彻底的操作。他加了一个小花样："I'm going to do a matmul followed by a relu just because just for kicks"，理由并不牵强："if you have, uh, you know, one linear layer, it's a matmul and then you apply, uh, a relu activation. So, this is not like completely out of nowhere." 一个 linear layer（线性层）本来就是 matmul 后接 ReLU，所以 $C = \mathrm{ReLU}(A@B)$ 是真实存在的小计算图；"but I'll show you later why I I I did this" 则预告了后面 kernel fusion 的伏笔。画面上的入口代码原文是：

```python
def triton_matmul_relu_example():
    # Matrix multiplication is the bread and butter of deep learning.
    a = torch.randn(1024, 1024, device=cuda_if_available())
    b = torch.randn(1024, 1024, device=cuda_if_available())
    c = naive_matmul_relu(a, b)
```

两个输入都是 $1024 \times 1024$ 的随机矩阵（`torch.randn(1024, 1024, device=cuda_if_available())` 出现两次），这是本段非常具体的规模数字。尺寸记法是标准形式：$A$ 为 $M \times K$，$B$ 为 $K \times N$，输出 $C$ 为 $M \times N$，即

$$C_{m,n} = \sum_{k=0}^{K-1} A_{m,k}\,B_{k,n}$$

讲者用画面上那个 3×3 玩具矩阵把下标讲活了：$A$ 的元素按行写成 $A1\,A2\,A3 / A4\,A5\,A6 / A7\,A8\,A9$，$B$ 是 $B1\ldots B9$，$C$ 是 $C1\ldots C9$，左侧标 $m$（行）、上方标 $k$ 与 $n$（列）。他先说 "fix an, uh, a a one of these elements. Let's say M equals 1, N equals 2"，随即改成 "let's do M equals 1, uh, N equals 1. So, I'm doing C5"，即固定输出矩阵的某个元素（如 $C5$），对每一个 $k$ 读 $A$ 的对应行元素与 $B$ 的对应列元素，相乘、累加，最后写出这一个元素。幻灯片把 Naive approach 逐条写得很清楚：`Fix any (m, n).` / `For each k:` / `Read A[m, k] and B[k, n] from HBM.` / `Multiply and accumulate.` / `Write result to C[m, n] in HBM.`

![朴素 matmul 幻灯片：3×3 的 A1–A9 / B1–B9 / C1–C9 矩阵示意、Naive approach 四条、Bottleneck: M K N reads, M N writes 与 Arithmetic intensity: O(1)](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0054_01-15-04.jpg)

把它写成可读的伪代码（与画面 bullet 一一对应）：

```python
# Naive approach: 固定任意 (m, n)
for m in range(M):
    for n in range(N):
        acc = 0.0
        for k in range(K):
            acc += A[m, k] * B[k, n]   # 每次都从 HBM 读
        C[m, n] = acc                  # 写回 HBM
```

讲者强调 "So, that is a valid matmul kernel. Okay?"——它**正确**，问题不在对错而在代价。73:57 他数读写次数："if you look at how many reads and writes it's doing, uh this is not good, right? So, um, basically for every M and N and K, I have to read from HBM. So, it's on the order of M times K times N reads"，画面上对应 `Bottleneck: M K N reads, M N writes`。接着回到第二讲的定义：arithmetic intensity（算术强度）= 运算次数 ÷ 传输字节数，希望它越高越好；这里运算量是 $O(MKN)$，读数也是 $O(MKN)$，于是

$$\text{AI}_{\text{naive}} = \frac{O(MKN)}{O(MKN)} = O(1)$$

画面直接写 `Arithmetic intensity: O(1)`。常数级算术强度意味着算力被 HBM 带宽死死卡住，这是朴素 kernel 的根本瓶颈。

冗余读的直觉由 74:43 起用玩具矩阵点破，也正是画面高亮那一行：`Computing C4 and C5 both need A4, A5, A6.`——计算同一行的 $C4$ 与 $C5$ 时，$A$ 的 $A4,A5,A6$ 被读了两次；下一行写着 `Can we read A4, A5, A6 from HBM once to compute both?`，再往下淡出的是 `Answer: using shared memory`。讲者的口述是 "if you can just read those once, then you really save on reads and that's that's great. So, let's try to use, uh, shared memory to do that." 与前后内容的关系：前一章的 row sum 已经演示了「把 HBM 数据搬进片上、在 shared memory 里累加」，这里把同一个动机从一维归约推广到二维乘加；下一章则先给出「把 $A$、$B$ 全搬进 shared memory」的理想版本，再因放不下而落到 tiling。

易错点有三。其一，别以为「写次数也不少」——讲者明说 "number of writes, I mean, it doesn't matter"：写是 $O(MN)$，比读低一个量级，真正的瓶颈是 $O(MKN)$ 次读。其二，别把 arithmetic intensity 与总运算量混淆：$O(MKN)$ 的运算量本身不小，但相对于同样量级的字节传输，比值仍是常数，增大矩阵规模并不会自动改善它，必须靠数据复用（reuse）而非堆规模。其三，玩具矩阵用 $A1\ldots A9$ 这样的扁平编号，容易让人误以为下标从 1 开始；公式里 $k$ 仍应从 $0$ 到 $K-1$，$C4,C5,A4,A5,A6$ 只是元素标签而非数值。另外讲者念尺寸时先说 $M=1,N=2$ 又改口 $M=1,N=1$，属即兴口误，本笔记以「固定一个输出元素 $C_{m,n}$」为准。

## 理想做法、tiling 图示与 O(tile_size) 算术强度 [75:14 - 78:19]

75:14 讲者给出 "the idealized approach"：`I'm going to load all of A and B into shared memory and then I'm going to just compute C.` 若真能做到，读次数就从立方降到平方（$A$ 读 $MK$ 个、$B$ 读 $KN$ 个、输出写 $MN$ 个），算术强度变成 $O(N)$ 量级——"which means I get the arithmetic intensity of order N, um, which in the second lecture I sort of, um, you know, said was a I kind of ideal thing you could hope for"。把三种情形并排写，差距一目了然：

$$\text{AI}_{\text{naive}} = O(1),\qquad \text{AI}_{\text{tiling}} = O(T),\qquad \text{AI}_{\text{ideal}} = O(N)$$

其中 $T$ 是 tile size（分块边长）。图上方两行 bullet 正是这个推理的收尾：`This yields the idealized O(N) arithmetic intensity from before.` 与 `However, A and B are usually too large to fit in shared memory.`——理想很美，但 $A$、$B$ 通常远大于 shared memory，讲者在 76:02–76:09 又口述了一遍 "the problem is that A and B are usually too large to fit into shared memory"。于是 76:10 引出 "the idea here is this, you know, very classic idea of tiling. And the idea basically is to, well, fit as much you can to shared memory as you can"，并给出一句极好的总纲："it's going to be look like the naive ... globally look like the naive approach, but locally look like the idealized approach"。整体看仍是「逐块算、块间独立」的朴素框架，局部看每个块内部又享有「数据已在 shared memory、无冗余读」的理想待遇。76:37 讲者还提了一句 "I think Tatsuo showed this as well"，随后进入三张彩色网格图。

![tiling 三张彩色网格图：Matrix A 的横向行 tile、Matrix B 的纵向列 tile、Matrix C 的橙色 Temporary result tile，以及五项颜色图例与 tile size T / matrix size N 标注](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0055_01-16-44.jpg)

这张图值得逐项转述，因为转写几乎没念图例。三张网格依次是 Matrix $A$、Matrix $B$、Matrix $C$：$A$ 上有一条横向浅紫色带（行 tile，宽度跨满、高度若干行），$B$ 上有一条纵向浅紫色带（列 tile，高度跨满、宽度若干列），两者相乘得到 $C$ 上一个橙色小方块（Temporary result tile，临时结果块）。$A$ 上方标注 `tile size T`，$C$ 上方标注 `matrix size N`，直观说明 $T \ll N$。图例共五项：浅紫 `Outer loop over tiles`（外层循环遍历 tile）、深紫 `Current tile in outer loop`（外层循环当前 tile）、浅绿 `Inner loop over elements`（内层循环遍历元素）、深绿 `Current element in inner loop`（内层循环当前元素）、橙色 `Temporary result tile`。深紫小块里那个深绿小格，正是「在 tile 内部还要再遍历每个元素」的示意；这与后面代码里 `tl.dot` 一次算完整个 tile 形成对照——图示是概念分解，真实 kernel 用一次矩阵乘把内层循环整体消掉了。

高亮句 `Key idea: divide the matrix C into output tiles (thread blocks).` 点明切的是**输出** $C$，且每个 output tile 就是一个 thread block（线程块）。同一张幻灯片往下滚的部分把算法逐条补全，转写里讲者是在 76:48–78:15 用口语念的：`Fix an output tile in C.` / `For each pair of (row tile of A, column tile of B):` / `Load the corresponding A tile and B tile from HBM into shared memory.` / `Perform matrix multiplication on the tiles.` / `Accumulate into the partial sum (in shared memory).` / `Write output tile to HBM.`；画面另给出一行高亮 `Arithmetic intensity: O(tile_size).`。写成 Triton 风格的代码骨架：

```python
# Tiling: 固定 C 的一个 output tile (thread block)，沿 K 扫描
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)   # partial sum, shared memory
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs)     # A 的行 tile: HBM -> shared memory
    b = tl.load(b_ptrs)     # B 的列 tile
    acc += tl.dot(a, b)     # Perform matrix multiplication on the tiles
tl.store(c_ptrs, acc)       # Write output tile to HBM
```

讲者的口述与此一致：把 $C$ 切成 tile 后 "each of these tiles is going to be a thread block. I'm going to have a bunch of threads that is responsible to for computing this. Now, a different tile is going to be computed, you know, completely separately by another, uh, thread block"；在 Triton 里 "I wake up, I'm I'm looking at this thread block"，沿 $A$ 的行 tile 扫、沿 $B$ 的列 tile 扫，把对应小块搬进 shared memory，相乘后 "accumulate that um, in the partial sum and this is all sitting in shared memory"，扫完 "then I can finally write this output tile to HBM"。

78:15–78:33 给出代价与收益的定论：`Arithmetic intensity: O(tile_size).` 讲者补充 "you can generally not reach order N because that would require you to fit everything into shared memory, but, you know, if your tiles are big, then that's, you know, still not too bad"。也就是说 $O(T)$ 是 $O(1)$ 与 $O(N)$ 之间的折中：$T$ 越大越接近理想，但 $T$ 受 shared memory 容量上限约束。与前后内容的关系：上一章证明朴素做法算术强度是常数，本章证明 tiling 把它抬到 tile 边长量级，下一章则把这个二维 tiling 落成 Triton 代码并顺手融合 ReLU。局限与易错点：其一，`O(tile_size)` 里的 tile size 是边长而非元素总数，读次数与运算量之比随边长线性增长，不要误读成随面积增长；其二，图例里的 `Outer loop over tiles / Inner loop over elements` 是概念性的两层循环，真实 kernel 里内层被 `tl.dot` 张量核指令替代，程序员并不写逐元素循环；其三，不同 output tile 完全独立、不共享 shared memory，因此 tiling 无法跨块复用数据，块边界上的 $A$/$B$ 小块仍会被不同块重复读入 HBM——这正是它达不到 $O(N)$ 的原因。

## stride、matmul kernel 代码与 kernel fusion [78:19 - 82:06]

78:36 讲者先用 "just as kind of a bonus" 引出 kernel fusion（算子融合）：既然已经在为 matmul 写 kernel，"sometimes if you want to apply an element-wise activation function, um, it's very easy to just like put it on at the end. And this is, you know, kernel fusion." 画面下方 Bonus 三条写得很清楚：`Often, you want to apply an elementwise activation function.` / `Example: GeLU(A @ B)` / `Solution: kernel fusion!`。也就是说，与其让 matmul 写一个 kernel、GeLU（本例是 ReLU）再起一个 kernel 反复往返 HBM，不如在写回 HBM 之前顺手把逐元素非线性做掉，几乎零额外成本；这也解释了 72:16 那句 "just for kicks" 的真正用意。

![tiling 幻灯片续滚：Key idea 与逐 tile 算法六条、高亮 Arithmetic intensity: O(tile_size)，以及 Bonus 三条 GeLU(A @ B) 与 kernel fusion](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0056_01-18-33.jpg)

79:03 插入一段 stride（步长）复习，因为 matmul kernel 里全是 stride。定义：tensor 是多维数组，但在内存里被 linearized（线性化）成一维；stride 告诉你如何把多维索引（如 `(row, column)`）映射成实际地址，做法是「行索引乘行 stride、加列索引乘列 stride」，即

$$\text{offset}(m,n) = m\cdot s_{\text{row}} + n\cdot s_{\text{col}}$$

转写给的例子是："every time you go advance to the next row, you go four positions in your memory. And every time you advance a column, you go one"——即 $s_{\text{row}}=4$、$s_{\text{col}}=1$（一个 4 列的行主序矩阵）；并补一句 "In the if it were a transpose, it would be the flipped"，转置后两个 stride 对调。易错点：stride 只是地址换算规则，与前面讲的 HBM memory coalescing（内存合并）、shared memory bank conflict（bank 冲突）处在完全不同层次；把 stride 写反（尤其 $B$ 按列读时）会直接导致非合并访问，而且这种错误不影响数值正确性、只影响速度，最容易被忽略。

79:46 起读 kernel。讲者说 "the launch is not interesting"，重点是 "you wake up and you are on tile M and N. So, it's like, 'Okay, I'm responsible for computing um the C uh matrix, but for the M / uh comma N tile.'"；"there's a bunch of like index manipulation, which um you can kind of I'll just gloss through, but it's it's straightforward, but a little bit"，随后交代这三组索引分别给出「$A$ 的哪些行」「$B$ 的哪些列」以及 "the numbers one through one through K"。画面把这些原文保留下来了：

```python
# kf_0057 画面中可见的 kernel 前段（行 639-659）
    stride_am, stride_ak,   # How to navigate a
    stride_bk, stride_bn,   # How to navigate b
    stride_cm, stride_cn,   # How to navigate c
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # We are working on the (m, n)-th tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Indices
    indices_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # Row indices of a [BLOCK_M]
    indices_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # Column indices of b [BLOCK_N]
    indices_k = tl.arange(0, BLOCK_K)                    # Row indices of a = column indices of b [BLOCK_K]

    # Initial matrix of pointers of a and b
    a_ptrs = a_ptr + indices_m[:, None] * stride_am + indices_k[None, :] * stride_ak  # [BLOCK_M, BLOCK_K]
    b_ptrs = b_ptr + indices_k[:, None] * stride_bk + indices_n[None, :] * stride_bn  # [BLOCK_K, BLOCK_N]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
```

![Triton matmul kernel 源码前段：六个 stride 参数、BLOCK_M/N/K、pid_m/pid_n、indices_m/n/k 与带 None 广播的 a_ptrs / b_ptrs 指针矩阵](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0057_01-20-21.jpg)

注意 `indices_m[:, None]`、`indices_k[None, :]` 里的 `None` 在画面上被蓝色高亮，它是 NumPy/Triton 风格「插入新轴做 broadcast（广播）」的写法，把一维索引撑成二维指针矩阵；注释里明确标出形状 `[BLOCK_M, BLOCK_K]` 与 `[BLOCK_K, BLOCK_N]`，累加器则是 `[BLOCK_M, BLOCK_N]`、dtype 为 `tl.float32`。这里一共出现 6 个 stride 参数：`stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn`，分别对应「如何 navigate a / b / c」，比 row sum kernel 里直接用 $N$ 当行 stride 严密得多。

80:45–81:51 讲者把循环与融合口述完：`acc` 是 "a accumulator matrix. This is going to be in shared memory. Um this is M by N"，并且 "look kind of like the uh row reduction"——与 row sum 的累加器同构，只是从向量升级成矩阵，循环从「只沿一个方向扫 tile」变成 "instead of just going across the the tiles, I'm now going across the row tiles and also simultaneously down the column tiles of B"。每个 $k$ 步 `tl.load` 一个 $A$ 小矩阵和一个 $B$ 小 tile，然后 "I perform this dot"，并给出那句关键心智模型："remember, whenever things are in shared memory, things look like PyTorch, and I can just say say 'Matmul it.' And you know, it'll do the the thing."——数据一旦在片上，就可以直接用 `tl.dot(a, b)` 这种张量级指令，而不必写逐元素循环。随后 "I advance to the next row tile of A and the next column tile of B"；循环结束、写回 HBM 之前，"if I wanted to apply element-wise non-linearity, I might as well, you know, do that here. Uh before I write it out to HBM, I can do some any sort of operation on it"，即融合 ReLU（`acc = tl.maximum(acc, 0.0)`），最后 `tl.store` 写出。81:52 他收尾："there's some sort of indices that you have to pay attention to, but hopefully the um form of the algorithm is is clear."

与前文的联系有两层：一是 `acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)` 与 kf_0050 的 `acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)` 几乎是同一行代码从一维推广到二维；二是 79:03 的 stride 复习正好补上 row sum kernel 里 `x_ptr + row * N + cols` 隐含「行 stride = $N$」的假设，这里改为显式传入 6 个 stride 参数，才能处理任意布局。局限方面：讲者明确把 index manipulation 略过（"I'll just gloss through"），所以本笔记不复原画面未显示的 `for k in range(...)` 完整循环体与 `tl.store` 地址表达式，只保留画面与口述都能确认的部分；另外 `tl.dot` 依赖输入 tile 已在 shared memory 且形状满足张量核要求，`BLOCK_M/BLOCK_N/BLOCK_K` 的取值受 shared memory 容量与寄存器约束，讲者并未在本段给出具体数值，不能凭空补。融合的边界也要注意：只有逐元素（element-wise）操作适合挂在写回前，跨元素归约或需要全局信息的操作不能顺手塞进来。

## 全讲总结、Triton 的替代方案与片尾问答 [82:06 - 86:40]

82:06 一句 "Okay."、82:07 "Any questions about that?" 之后，讲者于 82:12 说 "maybe I'll just summarize, and you can ask me later"，进入收尾。他先把整讲主线重新立起来：左边是 programming model（编程模型）——"today we talked about the there's a programming model, which is you know, you're talking about um either PyTorch or Triton or PTX. Even this is um the you know, what the kind of programmer can control"，即 PyTorch、Triton、PTX 是程序员可以主动选择、主动特化的层次，"even PTX, you can write PTX, and you can control every, you know, specialize however you want"；但这 "is not the full picture"，因为 "your code has to run on hardware, and there's only a finite number of SMs, a finite number of banks, and and you know, sizing of memory and registers all are finite"。带着大模型、大矩阵和 transformer 进来，"it has to kind of fit in with the constraints of um the the hardware"。因此 "benchmarking and profiling are really important to understand how the messiness of the hardware translates to, you know, performance"——benchmark（基准测试）看整体随规模怎么变，profile（性能剖析）看时间具体花在哪个 kernel 上，二者合起来才能把硬件的「脏活」翻译成性能数字。

83:15 起是对 Triton 的定位："we talked about Triton, which is I think a pretty nice neat language to think about uh thread blocks"，并给出本讲最实用的一句心智模型——"things are easier to think about thread blocks than individual threads, because you don't have to think about explicitly synchronizing threads or doing shared memory"。具体做法是："you figure out you have your computation, you break it down into these thread blocks, where you just need to read from shared memory, do some stuff, and write it back into HBM"。接着他把本讲例子按难度递增排了一遍："Element-wise is the easiest, and then reduction over a row, um reduction where the it doesn't uh fit into a row, and that's sort of we introduced kind of the baby tiling, and then Matmul is the kind of the the canonical example, where you actually do tiling"。84:11 收束："that's all I'm going to say about how to program a single GPU. Next time, we're going to go to more GPU and talk about um multi-GPU programming."

画面把这份总结以「伪代码目录 + Summary」的形式钉在屏幕上，是本段图上信息最密集的一处，转写里几乎没有念：

```python
# This lecture: benchmarking/profiling + writing kernels
review_of_gpus()
benchmarking_and_profiling()          # Where are the bottlenecks?
naive_vs_builtin_vs_compiled_gelu()   # Apply it to the GeLU example

# Write Triton kernels
triton_introduction()
triton_gelu_example()                 # Elementwise operation
triton_softmax_example()              # Reduction (row fits in a block)
triton_row_sum_example()              # Reduction (row doesn't fit in block)
triton_matmul_relu_example()          # Tiling: use shared memory
```

下面 Summary 六条依次是：`Know the programming model (PyTorch, Triton, PTX) to give you correctness`（编程模型保证正确性）、`Understand the hardware (SMs, warps, occupancy, bank conflicts, etc.) to optimize performance`（懂硬件才能优化性能，点名 warps、occupancy、bank conflicts）、`Benchmark to understand scaling`、`Profile to see what's being executed for how long`、`Triton: think in terms of thread blocks (read to shared memory, do stuff (fusion), write back HBM)`、`Examples: GeLU (elementwise), softmax (row-wise), row sum (baby tiling), matmul (tiling)`；屏幕最后一行是 `Next time: more than one GPU`。这份目录正好把本段（row sum → matmul）与整讲前 65 分钟的接口对齐了：本段覆盖 65:00–86:40，约 21 分 40 秒，其中片尾问答从 84:17 一直持续到 86:33。

![全讲总结幻灯片：benchmarking/profiling 与各 triton 示例的伪代码目录、六条 Summary bullet、以及 Next time: more than one GPU](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0059_01-23-58.jpg)

84:17 起是片尾问答。第一个问题："if I'm able to write my own kernels, what are the alternatives that I have to Triton, and what are the range of my choices that I have? Like how close can I get to the optimal performance?" 讲者的回答落在 inductive bias（归纳偏置）上："you know, there's a trade-off between every language has sort of inductive bias, right? That makes certain things easier and certain things harder." 具体到 Triton："most of what we do, Triton was built, you know, by people who, you know, train transformers. So, anything involving uh transformers, I think it's going to be relatively um you know, easy there"；"of course, in the extreme, you can always go to PTX and uh write that, but you know, wouldn't advise that as a first step"；此外 "There are a bunch of other uh language uh sort of libraries. There's you know, ThunderKittens, there's uh you know, cute you know, various DSLs that allow you to give you you know, they're not necessarily comparable uh either up or down the stack, they just give you different um you know, characteristics"。也就是说 PTX 最底层、Triton 面向 transformer 负载、ThunderKittens 与 CUTE（cute）等 DSL（domain-specific language，领域专用语言）各有取向，并不是简单的「谁比谁高一级」，也没有给出「离最优性能多近」的量化答案。

85:30 的第二个问题关于高维张量："When you have a high-dimensional tensor, uh you know, multiplications or a high-dimensional tensor processing, uh what is the best approach? Is there is there like what I'm thinking is can I load unload the whole tensor both of the tensors on you know, bunch of threads or thread blocks on my GPU at the same time and compute them, or would it be better to sort of, you know, take each element, just what we did here, right? For the particular particular component of of my tensor, then process it, and then write it back into HBM?" 讲者 86:12 收尾："the briefly the question, and we should probably wrap up, is um is it better to kind of read all at once or maybe process individual elements at um a time. Um I think it's hard to answer this in the abstract. It kind of depends on the nature of the computation. Um uh maybe I we can talk offline about that." 随后 86:33 "Okay. All right, see you next time." 这正是本段 86:40 的片尾；片尾画面里讲者双手合十站在写满无关公式的白板前做收束，白板内容并非本讲课件，不应误当成知识点。

把三档算术强度并排写，就是整讲从「能写对」到「跑得快」的浓缩：

$$\text{AI}_{\text{naive}} = O(1)\ \xrightarrow{\text{tiling}}\ \text{AI}_{\text{tiling}} = O(T)\ \xrightarrow{\text{全放进 shared memory}}\ \text{AI}_{\text{ideal}} = O(N)$$

其中 $T$ 是 tile size。与前文的关系：本段的 row sum 是 "baby tiling"（一维 tile 循环），matmul 是正式 tiling（二维切块 + shared memory 累加 + kernel fusion），两者共用「读入片上、计算、写回 HBM」这一个模板，正好对应 Summary 里 `Triton: think in terms of thread blocks (read to shared memory, do stuff (fusion), write back HBM)`。局限必须讲清：讲者对「一次性全读还是逐元素处理」明确拒绝给抽象答案，只说取决于计算性质并建议线下讨论，本笔记也不替他下结论；对「离最优性能有多近」同样没有量化回答，只指出每种语言有归纳偏置、PTX 是理论下限但不建议首选。此外本讲到此只覆盖单 GPU，多 GPU 编程被明确推迟到下一讲；SM 数量、bank 数量、寄存器与内存容量在本段都只被描述为 "finite"（有限），讲者没有给出具体数字，不能编造。

## 要点回顾

### GPU 存储层级与编程模型
- 容量与带宽大致反向：$\text{BW}_{\text{register}} > \text{BW}_{L1/\text{shared}} > \text{BW}_{L2} > \text{BW}_{HBM}$；SM 数为 A100 108、H100 132、B200 148，每 SM 寄存器 256 KB 三代不变，HBM 从 80 GB / 2 TB/s 涨到 192 GB / 8 TB/s。B200 的 8 TB/s 相对寄存器带宽 ~447 TB/s 仍慢约两个数量级，所以“慢”始终是相对片上存储而言。
- 编程模型三层是 thread → thread block / CTA → grid，可写成 $\text{Grid} = \{\text{CTA}_1, \ldots, \text{CTA}_8\}$；thread 处理一小块数据，CTA 是一组共享同一块 shared memory 的线程，grid 是 CTA 的集合。
- elementwise（如 GELU）用 `f(i) for i = 0, ..., N-1` 足够；softmax、matmul 需要线程间通信，于是 `a thread block is scheduled on one SM`，数据流为 HBM → shared memory → threads → HBM。
- 总结链 `grid (HBM) -> thread block (shared memory) -> thread (registers)`；L1 与 shared memory 物理上同一块，区别只在 shared memory 可由程序员控制、L1 不能，且二者都是 per SM。

### 决定性能的五类硬件细节
- warp 每束 32 线程，64 线程块即 2 个 warp；同 warp 必须 lockstep 执行同一指令，分支 `if A else B` 被 sequentialized 成 control divergence，通常应避免。它与 occupancy 成因不同：一个是同 warp 内指令不一致导致串行，一个是能同时驻留多少 warp。
- warp occupancy：每线程最多 255 个寄存器、每 SM `max_registers = 65536`；128 线程 × 160 寄存器得 `num_registers_per_block = 20480`、`num_blocks = 3`、`num_warps = 12`、`occupancy = 0.1875`（约 18%），瓶颈注释是 `# Limited by registers`，thread coarsening 可让每线程做更多工作。
- bank conflicts 作用于 shared memory：32 个 bank、每 bank 4 字节，每周期每 bank 只许一个线程访问，最坏形成 32 路冲突；解法 swizzling 用 `row xor col` 重排布局。
- memory coalescing 作用于 HBM：warp 的 32 线程访问合并成 128 字节 cache line，full coalescing 即 `32 threads × 4 bytes = 128 bytes`；讲者强调两者“感觉相似但非常不同”，一个是片上 shared memory、一个是片外 HBM。
- block occupancy 的 Wave quantization：B200 有 148 SMs，启动 160 个块时 $160 = 1 \times 148 + 12$，尾波只剩 12 个 SM 工作，解法是 `make number of thread blocks divide # SMs`；warp scheduler 以零成本在多 warp 间切换，隐藏 HBM 约 100 cycle 的延迟。

### Benchmarking 与 Profiling 方法
- `def benchmarking_and_profiling():` 的成功配方三步：Benchmark and profile your code → Make changes → 再 Benchmark and profile；`benchmarking() # How long does it take?` 与 `profiling() # Where time is being spent?` 分工明确。讲者强调总要先量一量代码在干什么、找出瓶颈，再动笔写 kernel。
- `def benchmark(run: Callable, num_warmups: int = 1, num_trials: int = 3) -> float:` 先 warm up 剔除惰性编译，再用 `torch.cuda.Event(enable_timing=True)` 的 `start_event` / `end_event`，配两次 `torch.cuda.synchronize()` 与 `times` 列表取平均，严格时应看 P95 分布。
- benchmark 只测 wall-clock 端到端时间，能比较实现快慢与看 scaling；对 `[256, 512, 1024, 2048, 4096, 8192]` 的 matmul，小维度近似恒定、接近 2000 维后才按 $O(d^3)$ 立方增长。
- profiling 揭示底层：`A + B` 调用 `CUDAFunctor_add`，`A @ B` 调用 `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_32x32x16_1x1x1_3_nnn_align1_bias_f32_relu`；CUTLASS 是 NVIDIA 线性代数库、SM100 对应 Blackwell、`32x32x16` 是 tile 形状，换张量维度就会换 kernel。

### GELU 案例与 kernel fusion
- 精确式 $\mathrm{GELU}(x) = x \cdot \Phi(x)$，tanh 近似为 $0.5 \cdot x \cdot (1 + \mathrm{Tanh}(\sqrt{2/\pi}(x + 0.044715 x^3)))$；常数在 Triton 注释里原样复现为 `a = 0.79788456 * (x + 0.044715 * x * x * x)`。`approximate='none'` 是精确式，`'tanh'` 才是上面这条近似，二者数值并不严格相等。
- 三匹马是 `naive_gelu`（non-fused）、`builtin_gelu`、`compiled_gelu = torch.compile(naive_gelu)`；`check_equal_id` 校验三者对 `x = torch.tensor([1.])` 都输出 `0.8412`，语义一致。
- 在 `dim=16384` 上 `naive_time = 3.7583`、`builtin_time = 0.6670`、`compiled_time = 0.9388`，内置约快 5.6 倍、编译版约快 4 倍；但讲者强调 Triton 并不天然更快，结果 very hardware-dependent。
- profiler 显示 naive 把计算图拆成 `BinaryFunctor`、`AUnaryFunctor`、`CUDAFunctor_add`、`tanh_kernel_cuda` 等多个 kernel，每个都 HBM 往返；内置是单条 GELU CUDA kernel，`torch.compile` 融合成 `triton_poi_fused_add_mul_tanh_0`，做到每个元素 read once / write once。

### Triton kernel 通用模板与 PTX
- 通用骨架是 `pid = tl.program_id(axis=0)` → `start = pid * BLOCK_SIZE` → `offsets = start + tl.arange(0, BLOCK_SIZE)` → `mask = offsets < num_elements` → `tl.load(x_ptr + offsets, mask=mask)` → 计算 → `tl.store(y_ptr + offsets, y, mask=mask)`，对应 wake up、定下标、read、do stuff、write to HBM 六步。
- `BLOCK_SIZE: tl.constexpr` 是编译期常量，`x_ptr` / `y_ptr` 是指针而非张量，`tl.tanh` 不存在须用 $\tanh(a) = (e^{2a}-1)/(e^{2a}+1)$ 自行拼装；mask 必须同时传给 `tl.load` 与 `tl.store`。
- 启动侧 `y = torch.empty_like(x)`、`num_elements = 8192`、`BLOCK_SIZE = 1024`、`triton_gelu_kernel[(num_blocks,)](...)`，并用 `check_equal_1d(triton_gelu, naive_gelu)` 先保证正确性。口述说的 8000 维与画面上的 `num_elements = 8192` 略有出入，应以 `x.numel()` 为准。
- `Triton compiles down to PTX`：`@%p1 ld.global.v4.b32` 向量加载、`%ctaid.x` 为块索引、`%tid.x` 为线程索引，`0f3D372713` 是 `0.044715` 的 float32 编码；一个线程处理 8 个元素体现 thread coarsening，PTX 不规定 warp 调度与 SM 分配。
- CUDA（NVIDIA）指定每个 thread 做什么，fine-grained control 但要管 shared memory；Triton（OpenAI）指定每个 thread block 做什么——抽象层级变的是“你写什么”，不是“硬件怎么跑”。

### Softmax 与行归约：从一块到 tile 循环
- $\text{softmax}(x)_i = \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)}$、$m = \max_j x_j$ 先减最大值只为数值稳定；朴素 PyTorch 读 `5MN + M` 次、写 `3MN + 2M` 次，理想只需 `MN reads, MN writes`（约 4× 加速空间）。
- 启动配置 `block_size = triton.next_power_of_2(N)`、`num_blocks = M`、`triton_softmax_kernel[(M,)](...)` 并传入 `x_row_stride = x.stride(0)`；一个 program instance 对应 `pid = row index`，行与行不交互故无需跨块 shared memory。
- kernel 以 `assert num_cols <= BLOCK_SIZE` 为前提，`x_row = tl.load(x_ptrs, mask=col_offsets < num_cols, other=float("-inf"))` 后用 `tl.max(x_row, axis=0)`、`tl.exp(x_row)`、`tl.sum(numerator, axis=0)` 做行内归约；填 `-inf` 是因 $\exp(-\infty)=0$，填 0 会污染归一化常数。
- 一行放不进块时（4096 列 / block 1024 → 4 个 tile）改用 row sum：每个线程握 `acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)`，`for start in range(0, N, BLOCK_SIZE)` 逐 tile `acc += x`，最后 `result = tl.sum(acc, axis=0)`；玩具例子 `n_cols=10`、`BLOCK_SIZE=4` 得 `out[1] = 39`。
- mask 填充值因归约代数而异：row sum 用 `other=0.0`、softmax 用 `other=float("-inf")`；tile ≠ block，整行仍是一个 block，tile 只是块内循环的一轮（讲者称 “baby tiling”），累加器落 registers 还是 shared memory 由编译器决定。

### Matmul 的 tiling、算术强度与全讲主线
- $C_{m,n} = \sum_{k=0}^{K-1} A_{m,k} B_{k,n}$，输入是两个 `torch.randn(1024, 1024)`；朴素做法 `Bottleneck: M K N reads, M N writes`，$\text{AI}_{\text{naive}} = O(1)$，写是 $O(MN)$ 并非瓶颈，`Computing C4 and C5 both need A4, A5, A6` 点出冗余读。
- 三档算术强度 $\text{AI}_{\text{naive}} = O(1) \to \text{AI}_{\text{tiling}} = O(T) \to \text{AI}_{\text{ideal}} = O(N)$；理想是把 A、B 全放进 shared memory，但通常放不下，故 `Key idea: divide the matrix C into output tiles (thread blocks)`。
- kernel 用 6 个 stride（`stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn`）与 $\text{offset}(m,n) = m\cdot s_{\text{row}} + n\cdot s_{\text{col}}$ 做地址换算；`indices_m[:, None]` 的 `None` 插入新轴 broadcast 出 `[BLOCK_M, BLOCK_K]` 指针矩阵，`acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)`。
- 沿 K 扫描 `for k in range(0, K, BLOCK_K)` 累加 `acc += tl.dot(a, b)`，写回前 `acc = tl.maximum(acc, 0.0)` 融合 ReLU 即 kernel fusion，几乎零额外成本；`Arithmetic intensity: O(tile_size)` 里的 T 是 tile 边长而非面积。
- 全讲 Summary：用编程模型（PyTorch、Triton、PTX）保证正确性，用硬件（SMs、warps、occupancy、bank conflicts）优化性能，Benchmark 看 scaling、Profile 看 what's being executed for how long；难度梯度为 GeLU → softmax → row sum（baby tiling）→ matmul（tiling），心智模型是 `Triton: think in terms of thread blocks (read to shared memory, do stuff (fusion), write back HBM)`。Triton 因面向 Transformer 负载而上手容易，极限仍可下探 PTX 或换用 ThunderKittens、CUTE 等 DSL，但讲者对“离最优性能多近”没给量化答案；下一讲转向 `more than one GPU`。

## 视频来源

- 原始链接：https://www.youtube.com/watch?v=xnDHaNUvHBg
- 课程：Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 6

---

## 对照参考笔记补遗（Spring 2025 版）

> 编者注：本节依据 2025 版参考笔记整理，其中的「更正」小节记录的是核对时发现的问题，对应正文已在本版订正；补遗内容不代表 2026 课程视频原文。


本节内容来自 Spring 2025 版参考笔记、仅用于对照补充，不代表 2026 课程视频原文；凡与正文数字或讲法冲突之处，一律以 2026 视频笔记为准。

### 一、算术强度：为什么大多数算子受限于内存

正文在矩阵乘法一章把算术强度写成三档 $\text{AI}_{\text{naive}}=O(1) \to \text{AI}_{\text{tiling}}=O(T) \to \text{AI}_{\text{ideal}}=O(N)$，却没把它立成贯穿全讲的判据。参考笔记在 GPU 复习阶段就给出定义：arithmetic intensity（算术强度）是浮点运算数与内存传输字节数之比 $\mathrm{AI}=\#\mathrm{FLOPs}/\#\mathrm{bytes}$，比值高即 compute-bound（compute-bound，计算受限，good），比值低即 memory-bound（memory-bound，内存受限，bad）。它给出的宏观理由是 "Compute scaling is much faster than memory scaling"——算力的代际增速远快于内存带宽的增速，于是越来越多的算子不是算不完，而是 GPU 在等数据，"the GPU spends more time waiting for data than computing"。由此得到一条总纲：`Matrix multiplication (if done cleverly) is compute-bound. Everything else is often memory-bound. We aim to reduce memory-bound operations.`

这条判据与正文的存储层级不等式 $\text{BW}_{\text{register}} > \text{BW}_{L1/\text{shared}} > \text{BW}_{L2} > \text{BW}_{HBM}$ 是同一件事的两种说法：越靠外的存储越慢，数据每往返一次 HBM 就把 AI 往下拉一截。它也为正文的三档公式提供了落点——tiling 提升的不是运算量，而是每个字节被复用的次数（data reuse，数据复用），这正是后面 softmax 逐行归约与 matmul 二维分块共同的出发点。易错点在于把 AI 当成「总运算量」：$O(MKN)$ 的乘加本身不小，但相对同量级的字节传输，比值仍是常数，单纯把矩阵做大并不会自动改善它；真正起作用的是复用，而不是堆规模。

### 二、warp 的硬件动因与线程块数经验法则

正文把 warp（线程束）定义为 32 线程锁步执行的硬件单位，并区分了 control divergence（控制分歧）与 occupancy（占用率）；参考笔记补上了「为什么要有 warp」这一层：`Warps exist to reduce the amount of control machinery needed, as a single control unit can manage 32 threads executing the same instruction. This allows for more compute units with simpler controls compared to CPUs.` 也就是说，用一个控制单元同时驱动 32 个执行同一条指令的线程，省下的控制逻辑可以换成更多计算单元，这正是 GPU 相对 CPU 能用更简控制、堆更多算力的取舍。它与正文「warp 不属于编程模型三层抽象、却决定性能」的结论正好接得上：抽象里看不见它，是因为它本就是硬件为省控制逻辑而引入的执行粒度，写正确的代码时感觉不到，写快的代码时绕不开。

参考笔记另给一条正文没有的经验法则：除了 `Wave quantization: make number of thread blocks divide # SMs`（让线程块数整除 SM 数，消掉正文 B200 148 个 SM、启动 160 个块、$160=1\times148+12$ 的尾波），还有 `Rule of Thumb: The number of thread blocks should be >= 4x # SMs`。前者消尾波，后者保证每个 SM 上有足够多的块可调度，让 warp scheduler 在某个 warp 等 HBM（正文记约 100 个 cycle）时切到别的 warp。若沿用正文的 B200 148 个 SM 推算，这条法则大约对应不少于 592 个块。易错点有两处：一是同一份参考另有一处把该值写成 `>24x # SMs`，与 $\ge 4\times$ 冲突，属解析讹误，不予采纳；二是这毕竟是 2025 版讲法，2026 视频只重申了「整除」而未重申 $4\times$，补录时须标明版本，不能当成 2026 的结论。

### 三、2025 版的基准实验：MLP、A100 具体毫秒与线性扩展

正文的 benchmark（基准测试）演示以矩阵乘法为主：用 `run_operation2` 扫 `[256, 512, 1024, 2048, 4096, 8192]`，结论是「小维度近似恒定、接近 2000 维后才按 $O(d^3)$ 立方增长」。参考笔记用了不同的载体：一个由 `Linear -> GELU` 反复堆叠的 MLP（Multi-Layer Perceptron，多层感知机）跑 forward/backward，参数为 `dim=128, num_layers=16, batch_size=128, num_steps=5`；随后又用 `dim=256, num_layers=4, batch_size=256, num_steps=2` 做缩放实验。它把「先测量再动手」落在一个真实训练循环上，比单一算子更贴近正文反复强调的 wall-clock（墙钟时间）语义，也与正文「benchmark 只给一个端到端总时长」的分工一致。

具体数字值得单独记下。在 A100 上，方阵维度 `dims = (1024, 2048, 4096, 8192, 16384)` 对应约 `0.70 / 0.83 / 3.14 / 21.68 / 162.93 ms`（另一处变量框记为 `0.87 / 0.84 / 3.15 / 21.69 / 162.94`），1024 与 2048 几乎不涨，被 kernel launch 与 CPU-GPU 传输的固定开销压平，之后才显出超线性。MLP 侧 `mlp_base = 6.293 ms`；把 num_steps 按 2、3、4、5 放大得 `11.326 / 16.761 / 22.028 / 27.310 ms`，约每步 5 ms；把 num_layers 同样放大得 `9.783 / 13.274 / 16.856 / 20.847 ms`，约每层 4 ms，二者都呈线性。易错点：这些是 2025 版 A100 的绝对值，与正文 `dim=16384` 上 GELU 的 `3.7583 / 0.6670 / 0.9388` 既不同机器、也不同算子，不能拿来做交叉校验；它们只补足正文「有定性趋势、无绝对毫秒与随步数层数线性这第二个 scaling 维度」的空白。

### 四、CPU-GPU 异步执行、时间线与「仓库—工厂」类比

正文在写 `benchmark` 时强调 GPU 一切异步、CPU 发射完立即返回，因此必须 `torch.cuda.synchronize()` 作同步屏障，并用 CUDA events 避开 CPU 发射开销。参考笔记用 NVIDIA Nsight Systems 把这个抽象画成了时间线：上半是 `CUDA HW` 的硬件活动、下半是 CPU threads，CPU 不必等 GPU 完成即可继续派发，`the CPU can run ahead`，甚至出现 `CPU might be processing layer_9 while the GPU is still working on layer_1`。更实用的一个 gotcha（易踩的坑）是：CPU 侧一条需要 GPU 值的 `print`（例如打印反向后的 loss）会隐式触发 `torch.cuda.synchronize()`，让 CPU 时间线在此暂停、可能引入 stall——这正是正文「改完代码要重新测、计时前后都要同步」的底层原因。易错点还在于：profiler 里显示的 active CPU time 并不等于 GPU 利用率，别把 CPU 侧百分比读成算力分布。

同一条逻辑在参考笔记里还有一个更好复述的类比：kernel fusion（算子融合）一节把 DRAM 比作 warehouse（仓库）、把 SMs 与 SRAM 比作 factory（工厂），两者之间的数据搬运产生 `bandwidth cost`（带宽成本）。朴素做法每做一个小算子就把数据从仓库搬进工厂、算完再搬回，往返次数正比于计算图里的 primitive（原子操作）数；融合后只读一次、在片上把全部计算做完、写回一次，代价骤降。这与正文 `read from HBM once, you write to HBM once per element` 是同一句话的两种讲法，也把 GELU 案例中 `triton_poi_fused_add_mul_tanh_0` 单 kernel 的收益讲透。参考笔记另用 `torch.cdist` 演示反面：一个 `aten::euclidean_dist` 被拆成 `aten::matmul`（78% GPU）、`aten::pow`（6%）、`aten::sum`（3%）、`aten::max`、`aten::sqrt` 等多个 CUDA kernel；而 `time.sleep` 因纯 CPU、GPU 空闲，profiler 里 `100%` 落在 `cudadeviceSynchronize`。易错点仍是老问题：占比最高的往往是被分解后的具体 kernel，而非调用入口，读表要顺着长名字往下追；也只有逐元素算子适合挂在写回前，归约类操作不能顺手融合。

### 五、手写 CUDA C++ GELU kernel 与 Triton 的自动管理边界

2026 正文完全跳过了手写 CUDA kernel，直接从 PyTorch 三实现走到 Triton；参考笔记则用一整段演示了 CUDA C++（CUDA C++，NVIDIA 的 C++ 扩展）版本，恰好可以和正文的 Triton 骨架逐行对照。核函数是 `__global__ void gelu_kernel(float* in, float* out, int num_elements)`，每个线程先算全局下标 `int i = blockIdx.x * blockDim.x + threadIdx.x`，用 `if (i < num_elements)` 挡住越界，再写 `out[i] = 0.5 * in[i] * (1.0 + tanh(0.79788456 * (in[i] + 0.044715 * in[i] * in[i] * in[i])))`（float 字面量带 `f` 后缀）。把它与正文 `pid = tl.program_id(axis=0)`、`offsets = start + tl.arange(0, BLOCK_SIZE)`、`mask = offsets < num_elements` 并排看，`blockIdx.x*blockDim.x+threadIdx.x` 就是那个向量下标的标量版：CUDA 让程序员自己算每一个线程的下标，Triton 让编译器一次生成一整块下标。常数 $0.79788456\approx\sqrt{2/\pi}$ 与 $0.044715$ 与正文 GELU 公式完全一致，可见两版讲的是同一个算子。

包装侧同样具体：`TORCH_CHECK(x.device().is_cuda())` 与 `TORCH_CHECK(x.is_contiguous())` 两项断言（后者因线性索引只在连续张量上成立，正是最易错的一步），`torch::empty_like(x)` 分配输出，`block_size = 1024`，`num_blocks = cdiv(num_elements, block_size)` 做向上取整，再以 `gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), num_elements)` 启动并跟一句 `C10_CUDA_KERNEL_LAUNCH_CHECK()`；编译绑定用 `torch.utils.cpp_extension.load_inline(cuda_sources=[...], cpp_sources=[...], functions=["gelu"], extra_cflags=["-02"], name="inline_gelu", build_directory="var/cuda_gelu")`（参考中 `-02` 疑为 `-O2` 的解析讹误），调试时设 `CUDA_LAUNCH_BLOCKING=1` 让错误及时报出。参考笔记随后把 Triton 的取舍列成清单：自动管理 `Memory coalescing`、`Shared memory management`、`Scheduling within SMs`，仍由程序员手动决定的是 `Scheduling across SMs`（launch 多少 block），此外强调 compiler power（编译器常能超过 PyTorch 原生实现）与 debuggability（Python 可单步调试）。易错点：正因为跨 SM 的调度仍归人管，正文 wave quantization 里「块数对齐 SM 数」依旧影响性能，把 Triton 理解成「全自动」是误解；它与正文「Triton 省去显式同步与 shared memory 管理、但极限仍可下探 PTX」是同一层意思的正反两面。

### 六、2025 版五种实现与 softmax 的横向性能对照

正文 GELU 一章是三匹马：`naive_gelu`、`builtin_gelu`、`compiled_gelu = torch.compile(naive_gelu)`，在 `dim=16384` 上得 `3.7583 / 0.6670 / 0.9388`，并明确「compiled 略慢于内置」「a Triton kernel is actually not faster in this case」。参考笔记的 2025 版多出两条手写实现——`manual_gelu`（朴素 Python）、`pytorch_gelu`（内置）、`cuda_gelu`（手写 CUDA C++）、`triton_gelu`（手写 Triton）、`compiled_gelu`，数字是 `8.162 / 1.173 / 1.845 / 1.727 / 1.47 ms`（另一处为 `8.099 / 1.189 / 1.816 / 1.727 / 1.47`），结论写作 `torch.compile (1.47 ms) performs better than our custom CUDA (1.816 ms) and Triton (1.727 ms)`、`modern JIT compilers are very good at automatically performing optimization`。这组对照正好解释了正文记下的那句讲者原话 "last year when I did this, it was actually closer ... these things change and it's very hardware-dependent"：同一套思路隔一年、换个硬件与库版本，绝对数值与快慢关系都会变，因此正文强调结果 hardware-dependent（强依赖硬件）而非语言优劣。

softmax 也类似。正文只给 kernel 实现与精确读写计数（`5MN + M reads`、`3MN + 2M writes`，理想 `MN reads, MN writes`，约 4 倍空间），参考笔记另给 `dim=16384` 的 benchmark：manual `3.713`、compiled `1.54`、pytorch `1.59`、triton `1.917 ms`，并说 `torch.compile (1.54 ms) is the fastest, even outperforming PyTorch's native softmax (1.59 ms)`；编译版融合 kernel 名为 `triton_red_fused_div_exp_max_sub_sum_0`，其中 `red` 指 reduction（归约），恰与 GELU 的 `triton_poi_fused_add_mul_tanh_0` 里 `poi`（pointwise，逐元素）形成对照——名字本身就泄露了被融合的是逐元素链还是归约链，与正文「profiler 长名字会泄露实现细节」呼应。易错点仍是版本：以上全部是 2025 版 A100 数据，2026 视频的 GELU 三实现数值普遍更小、且「内置快于编译」的关系与 2025 不同，补录时只作横向参考，不可与正文数字互相代入，更不能据此改写正文任何结论。
