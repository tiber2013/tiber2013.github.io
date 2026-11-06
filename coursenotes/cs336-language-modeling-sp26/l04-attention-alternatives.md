---
title: "第 4 讲 · 注意力替代方案与混合专家"
description: "本讲讨论两类“在标准 Transformer 之上”的进阶架构改造：一是替换注意力块，用线性时间注意力及其混合变体把长上下文的成本从二次降为近似线性；二是替换 MLP 块，用混合专家（MoE）在不增加前向 FLOPs 的…"
---

# 第 4 讲 · 注意力替代方案与混合专家

> **Stanford CS336 语言模型从零构建 · L04** ｜ 讲者 Stanford Online ｜ 86 分钟 ｜ [原视频](https://www.youtube.com/watch?v=cKSwj_qZ8Jg)

> **v2 详写版** · 按 20 分钟分段详写，配图覆盖全片；文末附「对照参考笔记补遗（Spring 2025 版）」，记录与 2025 版讲义笔记的核对结果。

## 核心结论

本讲在标准 Transformer 之上叠加两条更进阶的架构改造：一条动注意力块，一条动 MLP 块。前者的起点是矩阵乘法结合律——把 $\rho(QK^\top)V$ 里的 softmax 归一化去掉后重排成 $Q(K^\top V)$，注意力成本便从 $O(n^2d_k)$ 降为 $2nd_vd_k$，把对序列长度的二次依赖挪到只有几千到几万的隐藏维度上；这一步是有损近似，却同时换来「训练用稠密矩阵乘并行、推理用固定状态 $S_t=S_{t-1}+k_tv_t^\top$ 串行」的 RNN 对偶，Mamba-2 的 $\gamma_t=f(x_t)$ 衰减门与 Gated DeltaNet 的 $(I-\beta_tk_tk_t^\top)$ 擦除项，都是在门控只依赖输入、不依赖状态的前提下增强表达力。正因纯线性时间注意力从未在规模上被验证，产业界跑通的都是混合式：MiniMax M1 用 7:1、Nemotron 3 约 3:1、Qwen 3.5 用 3:1 的递推层配少量全注意力层，从 1:1 到 24:1 再到 pure 的扫描显示低比例几乎无损、纯 RNN 明显退化；另有 DeepSeek V3.2 的 DSA 作为替代路线，用可跑 FP8 的轻量闪电索引器打分取 top-k，只在小子集上做全注意力，索引器仍是二次但常数极小，且能在长上下文扩展阶段才 post-hoc 接入。后者即混合专家（MoE），把大 FFN 换成多个同尺寸专家加选择器，参数量成倍增加而每 token 的前向计算量基本不变，代价是专家并行的 All-to-All 通信；其落地关键是 token 级内积 top-k 路由加负载均衡辅助损失 $\alpha N\sum f_iP_i$——路由离散不可微、训练时也看不到未被选中的专家，但这类简单启发式通过给热门专家施加更强的负梯度抑制住专家坍缩，DeepSeek 再叠加逐设备均衡、共享专家与细粒度专家，到 V3 演进为 aux-loss-free 的 bias 方案。全讲反复申明，成功的语言模型训练不只是深度学习，还必须真正尊重系统约束：FlashAttention 的常数因子、结构化稀疏矩阵乘、float32 路由器与 z-loss 这些工程细节，与复杂度阶数同等重要。

## 课程定位：两条架构改造主线 [00:00 - 01:32]

本讲以一段简短的事务性说明开场：讲师先为「本讲幻灯片当天早些时候还没上传」致歉，随后说明现在已可以通过课程网站取用，若不方便也一定能在课程 GitHub 上找到，因此建议听众把讲义放在手边对照阅读。画面上的标题页写着 "Lecture 4"，副标题为 "Attention Alternatives and Mixtures of Experts"，课程编号 CS336，署名 Tatsu H。这个标题本身就点出了本讲的两条主线：一条针对注意力，一条针对专家混合。讲师随即把它与上一讲区分开来——上一讲讲的是基础 Transformer（basic transformer），以及如何微调其中各个部件逐步得到现代语言模型；而本讲要谈的是叠加在常规 Transformer 之上、明显更进阶也更复杂（much more advanced, much more complex）的发展。

第一条主线是注意力替代方案（attention alternatives）。它的目标是通过架构层面的改动支持长得多的上下文，并且「大体上」让计算量对序列长度呈线性依赖，而不是二次依赖。用复杂度的记号来写，就是把注意力的成本从
$$
\mathrm{Cost}_{\text{attn}} = O(n^2 d_k) \;\longrightarrow\; O(n\,d_v d_k),
$$
其中 $n$ 是序列长度，$d_k$、$d_v$ 分别是 key 与 value 的维度。这里最重要的限定词是「大体上」：讲师在后面会反复说明，真正在产业规模上跑通的方案都是混合式的，纯粹的线性时间注意力迄今没有被验证过，因此 $O(n^2)$ 的项并不会被彻底消灭，只是被压缩到少数层里。第二条主线是混合专家（mixture of experts, MoE），它改造的是 MLP 部分，目的是在硬件利用率上取得明显改善——用讲师的话说，是「相对于我们必须花费的 compute，拥有更多的参数」。若把参数量与每 token 的前向计算量写成比例，MoE 想抬高的是
$$
\frac{\#\text{parameters}}{\text{FLOPs per token}}\ \uparrow\quad \text{而}\quad \mathrm{FLOPs\ per\ token}\approx\text{不变},
$$
这与注意力替代方案「改变计算量随 $n$ 的增长阶数」是完全不同的两种杠杆：一个动的是增长阶数，一个动的是单位算力能撬动的参数量。

两条主线在结构上的分工非常干净：前者修改 attention block，后者修改 MLP block。这也解释了本讲节奏与本段的边界——这一段（00:00–21:35）几乎全部用于第一条主线，从注意力成本的动机出发，一路推到线性注意力、RNN 对偶形式、Mamba-2 与 Gated DeltaNet，最后用混合比例的经验研究收尾；MoE 只在此处作为「今天要讲的第二个想法」被登记，其路由、负载均衡与专家并行等细节留到后续段落。理解这个分工有助于避免一个常见误读：不要把本段反复出现的 hybrid（混合）理解成「注意力与 MoE 的混合」，它指的是全 softmax 注意力层与线性／递推注意力层的混合；架构图里那些 MoE／FFN 块是模型整体骨架的一部分，不是本节讨论的变量。

从后续内容看，本段所有推导的公共起点可以浓缩成标准注意力的一个公式，本讲后面每一次改造都是对它动刀：
$$
\mathrm{Attn}(Q,K,V)=\rho\!\left(QK^\top\right)V,
$$
其中 $\rho$ 表示对 $QK^\top$ 做逐行归一化（softmax）。第 4 章要做的第一件事正是把 $\rho$ 去掉，第 6、7 章则会在这个基础上给递推状态加上输入依赖的门控。这里需要提前标记两个易错点：其一，「线性时间」是渐进复杂度的说法，不等于实际墙钟时间一定更短，本段第 3 章会用 FlashAttention 的实测数据说明常数因子往往比 big-O 更决定体验；其二，去掉 softmax 这一步是有损近似，它换来复杂度的下降，代价是表达能力的损失，这也是为什么后面所有生产模型都必须保留少量全注意力层。带着这两点，就能理解讲师为什么把本讲称为「进阶架构」而不是对基础 Transformer 的简单改良。

## 长上下文竞赛与注意力的成本结构 [01:32 - 02:49]

讲师首先为「为什么需要长上下文」给出直觉：人们希望把大量东西塞进 context，让模型拥有更多知识；也可能是 agent 需要在很多对象上持续操作，工作负载因此变复杂。画面左侧的散点图标题是 "Evolution of LLM Context Window Sizes (2018–2025)"，横轴 Release Date 从 2018 延伸到 2025，纵轴 Context Window Size (tokens) 采用对数刻度，刻度依次为 1,000、10,000、100,000、1,000,000、10,000,000。图例按 Developer 区分四家：Anthropic、Google、Meta AI、OpenAI。图上用红色小字标出两个里程碑 "First 128K Context Window" 与 "First 1M Context Window"，散点依次覆盖 GPT-1、GPT-2、GPT-3、GPT-3.5、Claude 2、GPT-4 Turbo、Gemini 1.5、Llama 4 等模型。讲师对这张图的解读是：各家头部 LM 厂商在竞相提供越来越大的上下文窗口（a clear rush by a lot of the top LM vendors），以支撑这些更复杂的工作负载。对数刻度本身是关键——从千级到千万级是四个数量级的跃迁，视觉上被压缩了，但计算代价并不会被压缩。

右侧的堆叠面积图给出了成本的量化对比：横轴 Sequence length 标到 0、5000、10000、15000，纵轴 TFLOPS 标到 600，橙色区域是 Attention，蓝色区域是 Feed-forward。可以看到蓝色（前馈）从原点起近似直线上升，在 15000 处约到 170；橙色（注意力）初期与蓝色贴合，随后以明显的上凸曲线甩开，在 15000 处逼近 600。把两类成本写成随 $n$ 增长的阶数，就是
$$
\mathrm{Cost}_{\text{FFN}} = \Theta\!\left(n\, d_{\text{model}}^2\right),\qquad \mathrm{Cost}_{\text{attn}} = \Theta\!\left(n^2 d_k\right),
$$
前者对序列长度线性，后者因为注意力是全部位置之间的 all-to-all 连接而对 $n$ 二次。转写里讲师把这一点说得直白：feed forward「grows linearly」，而 attention「is an all-to-all connection between all the different positions, quadratic, so quickly outpaces feed forward as the sequence length grows」。幻灯片右下角还给出了这张图的出处链接（meibel.ai 上一篇关于上下文窗口增长影响的分析），说明这些曲线并非示意，而是有具体测算依据的。

把两张图合起来，就得到本段最核心的动机链条：上下文从数千涨到数百万乃至千万 token 已是产业现实，而在这个区间里二次项会彻底压倒线性项。讲师特意补了一句历史视角，提醒听众不要把这个结论当成永恒真理——「it used to be that for big models at like smallish sequence lengths, feed forward was the dominant cost」，在大模型、短序列的旧设定下，前馈块才是主要开销；只有在序列变长之后，注意力才「越来越成为一个问题」。这是一个容易被忽略的易错点：注意力并非天生是瓶颈，它是被长上下文推上瓶颈位置的。也正因为如此，下一章讨论的两类基础手段（局部／全局混合、系统常数因子优化）都只是「控制」成本而非「消除」二次项；当目标推进到 500 万到 1000 万 token（five, 10 million tokens）时，讲师明确说这些技巧都不够用，必须要有更根本的、对序列长度呈线性依赖的方案——这正是第 4 章起整条推导链的出发点。

![2018–2025 上下文窗口演进散点图与注意力/前馈 TFLOPs 随序列长度增长对比](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0002_00-02-01.jpg)

## 控制成本的基础工具箱：局部/全局混合与系统工程 [02:49 - 05:08]

在走向线性时间注意力之前，讲师先清点了一遍「basic toolkit」——也就是不改动复杂度阶数、却能实际控制成本的两类常规手段。第一类是架构层面的混合：把局部注意力（local attention）与全局注意力以各种 hybrid 方式组合起来，用来压制全局注意力的高昂代价。幻灯片左上方并排放了三张注意力掩码示意图，分别标注 (a) Transformer、(b) Sparse-Transformer (strided)、(c) Sparse-Transformer (fixed)，用蓝色方块展示从完整下三角到带状、分块稀疏的不同模式；下方的小架构图里绿色块标着 "Local Attention" 连续堆叠，粉色块标着 "Global Attention"，旁边还有 "alternately x2" 一类的重复次数标注，直观表达「少量全局层 + 大量局部层」的堆叠方式。转写给出的具体例子是：如果每 8 层才做一次全局注意力，其余层都用代价极低的局部注意力，成本就被大幅控制住了。把全局层占比写成一个比例式，即
$$
\frac{\#\text{global layers}}{\#\text{layers}} = \frac{1}{8},
$$
也就是 8 层里只有 1 层要付 $O(n^2)$，另外 7 层只付局部窗口内的代价。幻灯片在左栏下方把这招概括为 "Combine local + global attention"。

第二类是系统工程（systems engineering），讲师称之为一个「被严重低估的事实」：constant factors really, really matter（常数因子极其重要）。他特意点出这是本课程反复强调的主题之一——必须关注细节（pay attention to the details），而受过偏理论的计算机科学传统训练的人，很容易脱口而出「重要的是 big O，线性还是二次」。FlashAttention 就是反例：它并没有改变注意力的二次复杂度，只是把注意力运算重新排列成对系统更友好的形式，以最小化显存搬运开销（minimize memory transfer overhead），不再把庞大的注意力矩阵 materialize 出来，却带来了「truly dramatic」的性能提升。右侧柱状图标题为 "Attention forward + backward speed (A100 80GB SXM4)"，纵轴 Speed (TFLOPs/s) 标到 200，横轴 Sequence length 依次是 512、1k、2k、4k、8k、16k，图例列出五种实现：Pytorch（蓝）、FlashAttention（橙）、xformers（绿）、FlashAttention Triton（红）、FlashAttention-2（紫）。柱顶数值很有说服力：基础 Pytorch 在短序列上只有约 30–40 TFLOPs/s，1k 约 40、2k 约 45、4k 约 45、8k 约 46，到 16k 直接标成 OOM（放不进显存）；而 FlashAttention-2 从 1k 的 132 一路升到 2k 的 162、4k 的 171、8k 的 175、16k 的 176 TFLOPs/s，FlashAttention（橙）约 91–110，xformers 约 68–100，FlashAttention Triton 约 75–98。讲师口述的「factors of two improvements」正对应这种成倍差距；他还补充说，在某些尺寸上朴素实现已经装不进显存，而 FlashAttention 因为不物化大矩阵仍能继续跑， albeit somewhat slowly（ albeit 会变慢）。

这一章的易错点非常集中，需要逐条厘清。首先，FlashAttention 是纯粹的常数因子改善，它解决的是显存搬运（memory transfer）而非算术复杂度，因此绝不能把它当成线性时间注意力；把它与第 4 章的重排混淆，是理解本段最常见的错误。其次，图中 16k 处 Pytorch 的 OOM 标记说明「能不能跑」与「跑多快」是两个问题，FlashAttention 让超长序列从「不可行」变成「可行但偏慢」，这一点与后面线性注意力让复杂度本身下降是不同性质的收益。第三，即便有这些手段，讲师在 04:48 明确划了界限：如果要处理 500 万到 1000 万 token，这些 tricks 就不够了，需要「much more radical, much larger gains」。幻灯片底部那句 "But what if we want more radical and potentially large gains?" 正是全段的转折，把讨论从「控制成本」推向「改变复杂度的阶数」，也就是下一章要讲的矩阵乘法结合律。

![基础工具箱：Transformer 与两种稀疏掩码示意、局部+全局注意力组合，以及 A100 80GB SXM4 上各注意力实现的前反向速度柱状图](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0003_00-03-22.jpg)

## 线性注意力的起点：矩阵乘法结合律 [05:08 - 08:04]

进入正题前，讲师先给了一段研究史的判断：多年来很多人尝试过各种线性时间注意力，其中有不少 false starts（走不通的尝试）；但大约最近两年，涌现出一批「fairly effective」且已经在规模上、生产环境里被 battle-tested 的配方，这才使得 2026 年春季这一讲成为他第一次正式讲授线性时间注意力的年份——因为「it's clear now that these things really work at scale and in production」。而要理解后面所有这些方法，其实只需要先吃透一个核心想法：矩阵乘法的结合律（associativity of multiplication），其余内容都是在这个想法上叠加与细化。

幻灯片把标准注意力写成紧凑形式，并明确给出三个矩阵的维度：
$$
Q \in \mathbb{R}^{n\times d_k},\quad K \in \mathbb{R}^{n\times d_k},\quad V \in \mathbb{R}^{n\times d_v},
$$
$$
\mathrm{Attn}(Q,K,V)=\rho(QK^\top)V.
$$
其中 $Q$ 与 $K$ 先做一次 all-to-all 的矩阵乘法得到 $n\times n$ 的 $QK^\top$，经 $\rho$（对每一行做 softmax 归一化）后再乘 $V$ 得到输出。图上的说明文字直接点出二次项的来源：This is quadratic due to $QK^\top$ ($n^2 d_k$). Can we do better (when $\rho$ is the identity)? 也就是说，代价 $n^2 d_k$ 完全来自先算 $QK^\top$ 这个 $n\times n$ 矩阵；而讲师口述时把 $\rho$ 这一归一化称作「这一行的 softmax」，紧接着说「for the moment, let's forget that attention has a softmax and then I'm just going to drop the row」——即暂时把 $\rho$ 当作恒等映射丢掉。

一旦 $\rho$ 是恒等映射，括号就可以重排：
$$
(QK^\top)V = Q(K^\top V).
$$
幻灯片对这一步的评价是 "This is very silly, but surprisingly important.. We get from $n^2 d_k + n^2 d_v$ to $2 n d_v d_k$"。也就是说，原本要先算 $n^2 d_k$ 的 $QK^\top$、再算 $n^2 d_v$ 的乘积，现在改成先算 $K^\top V$（一个 $d_k\times d_v$ 的小矩阵，代价 $n d_v d_k$），再左乘 $Q$（又一个 $n d_v d_k$），合计 $2 n d_v d_k$。依赖对象因此从「$n^2$」变成「$n\cdot d_v d_k$」：序列长度 $n$ 可以大到百万量级（these Ns are very big, they're millions），而 key／value 的隐藏维度通常只有几千到几万（on the order of like thousands, tens of thousands），讲师强调「No one has like a million coordinates in their hidden dimensions」，所以把二次依赖挪到隐藏维度上是一次非常划算的交换。这正是他在 07:56 收束的那句话：associativity of multiplication allows us to change kind of the dependence on the sequence length。

这一步与前后内容的关系必须说清楚：它是本段所有后续方法的公共起点，第 5 章的 RNN 递推形式、第 6 章的 Mamba-2 门控、第 7 章的 Gated DeltaNet 擦除项，全部建立在「先去掉 $\rho$、再重排成 $Q(K^\top V)$」之上。但同样必须在此时就标记最关键的易错点——**这一步是有损的**。把 softmax 归一化 $\rho$ 换成恒等映射并不保持输出不变，它改变了注意力的函数形式，也损失了 softmax 带来的选择性聚焦能力，所以讲师在 09:36 才会补一句「this is linear, which is not very good」（线性注意力本身表达能力并不好）。与之相对，后面从线性形式改写成递推形式则是精确等价的，这个区分会在本段结尾的问答里被讲师再次强调。幻灯片底部还有一行转写完全没有念到的文献注记：「[Shen et al 2018, Katharopoulos 2020 for kernel version. Also connections to things like fast weight programmers]」——它提示线性注意力还有 kernel 版本，并与 fast weight programmers 一类思路存在联系；这条线索在第 7 章会以 fast weight programming / test-time training 的形式重新出现，构成本段前后呼应的一处细节。

![线性注意力推导幻灯片：给出 Attn(Q,K,V)=ρ(QKᵀ)V 的维度与 (QKᵀ)V=Q(KᵀV) 重排，并标注复杂度从 n²d_k+n²d_v 变为 2nd_vd_k](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0005_00-06-04.jpg)

## RNN 对偶：训练并行、推理串行 [08:04 - 09:52]

上一章把注意力重排成 $Q(K^\top V)$，本章要说明这个形式还有一层更重要的价值。讲师指出，右侧这个「先算 $K^\top V$ 再左乘 $Q$」的矩阵乘，在自注意力情形下看起来就像一个 RNN：与其写成上面那种稠密的大矩阵乘，不如沿着上下文从左往右扫，增量地把 $k_t$ 与 $v_t$ 的外积累加进一个状态里，再用当前的 $q_t$ 读出输出。幻灯片 "Recurrent form of linear attention" 先重述了重排式 $(QK^\top)V = Q(K^\top V)$，紧接着写道 "This is linear time (great) but the even nicer thing is that this looks like a RNN."，并给出递推形式
$$
S_t = S_{t-1} + k_t v_t^\top \quad\text{and}\quad y_t = q_t^\top S_t .
$$
这里 $S_t$ 是一个 $d_k\times d_v$ 的矩阵状态，每一步只做一次 $k_t v_t^\top$ 的外积累加；$y_t$ 用 $q_t^\top S_t$ 读出。转写里讲师把这个过程描述为「incrementally multiplying and accumulating my KVs」，并明确说稠密形式与这个 RNN 形式「are kind of equivalent」——也就是说，从第 4 章的线性形式走到递推形式，没有引入任何近似。

这一等价带来了本段最实用的结论：duality（对偶性）。幻灯片原文是 "This 'duality' allows us to train efficiently (using the parallel, quadratic form) and inference efficiently (using the serial, linear form)"。展开来说，RNN 一直有个众所周知的两面性：讲师说 RNN「quite nice because at inference time... it's fixed in size and I'm just carrying that forward」——推理时状态 $S$ 大小固定，逐 token 往前搬即可，每步代价 $O(d_v d_k)$ 与上下文长度 $n$ 无关，因此解码极省；但 RNN「not nice for training reasons」，因为串行递推无法并行，训练效率低。线性注意力的巧妙之处在于，它同时拥有两种写法：$Q(K^\top V)$ 这种稠密矩阵乘是并行的，适合训练；$S_t=S_{t-1}+k_tv_t^\top$ 这种串行递推状态固定，适合推理。讲师由此总结「So you can get the best of both worlds」，即「you can have it in this dense form, which will be great for training, it's parallel, or you can have it in this sort of serial form like an RNN, which is great for inference」。

从参数与量级上看，这个状态 $S$ 的维度是 $d_k\times d_v$，落在第 4 章说的「几千到几万」这一隐藏维度区间，与可能达到百万量级的 $n$ 完全脱钩；整个训练的计算量是 $O(n d_v d_k)$，解码时则退化为常数级的逐步更新加一次读出。它与前后内容的关系也很清楚：向上承接第 4 章去掉 $\rho$ 后的重排，向下则是第 6、7 章所有门控变体的骨架——Mamba-2 与 Gated DeltaNet 做的都是在这个 $S_t$ 递推上「加东西」，而没有推翻这套对偶性。

幻灯片底部还有一行转写完全没有念到的注记：「(Note: if one weights $S_{t-1}$ by $\gamma$, you get a RetNet)」——只要给上一时刻的状态乘上一个衰减因子 $\gamma$，这个递推就变成了 RetNet。这行注记看似闲笔，其实预告了下一章 Mamba-2 的全部动机：门控状态本身就是一条通往更强表达力的现成路径。本章的易错点有两个：一是别把「稠密形式适合训练」误记成「稠密形式就是 full softmax attention」，幻灯片里 "parallel, quadratic form" 的字面措辞容易让人误会，而转写强调的 dense form 指的是 $Q(K^\top V)$ 这个可并行计算；二是别把后面生产模型出现的性能退化归咎于这个 RNN 递推，讲师在 09:36 说的是「this is linear, which is not very good」——问题出在第 4 章丢掉 softmax 那一步的表达能力损失，而不是稠密形式与递推形式之间的转换。这一区分会在本段最后的问答里被再次确认。

![线性注意力的循环形式幻灯片：给出 (QKᵀ)V=Q(KᵀV) 重排、RNN 式状态更新 S_t=S_{t-1}+k_t v_tᵀ 与 y_t=q_tᵀ S_t，并注明 RetNet 注记](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0007_00-08-46.jpg)

## 从混合到生产：MiniMax M1 与 Mamba-2 门控 [09:52 - 14:15]

在给出对偶性之后，讲师先用 MiniMax M1 证明「最朴素的线性注意力在规模上就够用」，再逐步加上门控。MiniMax M1 被转写描述为一个规模大、性能高的中文开源模型（a fairly large-scale, fairly high-performance Chinese, open model），采用 7:1 混合：7 层线性注意力加 1 层完整 softmax 注意力。幻灯片底部的原文是 "Minimax M1 (and minimax-text-01) use a 7-to-1 hybrid (7 linear, 1 full) linear attention."，并评价 "Performance (generally) seems strong, and they show linear scaling in context length"。图上的证据分三块：左侧柱状图按 "Closed-weight Models"（OpenAI o3、Claude-4 Opus、Gemini-2.5 Pro、Seed-Thinking v1.5）与 "Open-weight Models"（DeepSeek-R1、Qwen3-235B-A22B、DeepSeek-R1-0528、Minimax M1）分组，纵轴 Accuracy (%) 0–100，横轴是 AIME 2024、LiveCodeBench、SWE-bench Verified、TAU-bench、MRCR (4-needle) 等评测，柱顶标有 86.0、60.0、56.0、62.3、73.4 等分数；中间的表格对比 Arch. 为 Softmax 与 Hybrid-lightning 两行在 BBH↑、DROP↑、MMLU↑、MATH↑、GSM8K↑、ARC-C↑、WG↑ 等列上的得分，Softmax 行约 28.2 / 27.4 / 49.3 / 4.6 / 18.8 / 46.4 / 65.6，Hybrid-lightning 行约 32.2 / 29.0 / 49.5 / 6.8 / 18.5 / 47.4 / 67.5；右侧 FLOPs 曲线的横轴 Generation Length 取 0、32K、64K、96K、128K，纵轴 FLOPs ($\times 10^{14}$) 标到 7，DeepSeek-R1 在 128K 冲到约 7、Qwen3-235B-A22B 约 4.5，而 MiniMax-M1 只到约 1.7，斜率明显更缓，这正是 "linear scaling in context length" 的可视化。右侧架构图还画出两类 block：标注 1x 的块含 Softmax Attention、MoE 与 FFN 1…FFN N、Router（$k = 2$），标注 Mx 的块含 Lightning Attention、MLP、Linear、RMSNorm、SiLU、Conv 等，直观对应 7 个轻量块配 1 个全注意力块。

讲师由此给出一个必须记住的边界：对上下文长度的依赖「is not fully linear cuz you've got these softmaxes, but the dependence is much, much milder」。他进一步断言，「As long as you've got some sort of full, quadratic connections involved」，这套最朴素的线性注意力就能在规模上奏效；而「No one has thus far really proven out, uh, fully linear time attention mechanisms at scale」，接下来要讲的全部是 hybrid。这句话把本章与前后内容钉死了：7:1 中的那个 1 不是可有可无的装饰，它是保留 $O(n^2)$ 连接、弥补第 4 章表达能力损失的必要成分；也正因为纯线性方案没被验证过，后面才需要 Mamba-2、Gated DeltaNet 这些「更强的递推」来尽量提升轻量层的表达力，同时始终把全注意力层的比例压得很低。

接着是 Mamba-2。讲师半开玩笑地说，若嫌线性注意力「too naive and simple」，想要更「serious, complex」的神经网络机制，那就看它的 RNN 形式再加一点细化。他介绍 Mamba、Mamba-2（乃至 Mamba-3）是 Albert Gu、Tri Dao 等人提出的一族 state space models（状态空间模型），原本从 state space theory 推导而来；但只要看实际机制，就会发现它其实是线性注意力机制的一个非常简单的发展。核心直觉来自 LSTM 时代：线性注意力的问题在于「always passing my state forward」——状态总被原封不动地往前传，而老经验告诉我们，模型需要知道何时传递信息、何时遗忘（forget things and send them to zero）。于是引入一个门 $\gamma_t$ 来调制「多少状态被带到未来」。幻灯片 "From linear attention to Mamba-2" 把两行公式并列写出：
$$
S_t = S_{t-1} + k_t v_t^\top \quad\text{and}\quad y_t = q_t^\top S_t \qquad (\text{Linear attention})
$$
$$
S_t = \gamma_t S_{t-1} + k_t v_t^\top \quad\text{and}\quad y_t = q_t^\top S_t + v_t^\top D \ \text{ with }\ \gamma_t = f(x_t) \qquad (\text{Mamba-2})
$$
页面上方写着 "Let's generalize linear attention a little bit and add per-position weights.."，下方则坦白 "There is a lot more words to justify this (go read the mamba 2 paper) but the mechanics is that we can make linear attention more expressive via gating (gating is good!)"，并强调 "This continues to have duality properties (compute $\gamma$ in parallel, apply duality)"。关键在于 $\gamma_t$ 只依赖当前输入 $x_t$、不依赖状态（Gamma of T is not stateful），因此所有 state-dependent 项仍然只有 $S$，$\gamma$ 可以并行算出；于是 Mamba-2 项既能写成大稠密矩阵乘用于训练，也能在推理时用递推形式获得收益。图底部并排画了 Sequential Mamba Block 与 Parallel Mamba Block 两种结构，图例区分 Linear projection、Sequence transformation 与 Nonlinearity (activation, normalization, multiplication)，框内可见 SSM、Conv 等模块，正是对偶性在系统实现上的体现。

生产验证落在 Nemotron 3 上。转写说它把 Mamba-2 当作「lightweight layer」，与偶尔出现的大 softmax attention 层以各种方式交替堆叠，用来管理 inference cost 与 expressiveness 的取舍；性能与 Qwen 3、GPT-OSS 等相比「pretty good」，且因为大量 Mamba-2 层，在相当大的上下文长度下仍有「pretty good throughput」，属于开源的小型前沿模型（small frontier models）。幻灯片 "Nemotron 3" 补充了转写没念到的细节：底部写着 "Mamba attention hybrid (3-1 ish) – comparable (or better) pref to other similar models"，即约 3:1 的混合；架构图标题为 "Nemotron-3-Nano-30B"，模块由 Mamba-2、MoE、Attention 组成并标有 x5、x3、x1、x4 的重复次数，Figure 1 说明写着 "Nemotron 3 models (e.g. Nemotron Nano 3) leverage a hybrid Mamba-Transformer MoE architecture consisting predominantly of interleaved Mamba-2 and MoE layers, with a select few self attention layers"；Figure 2 的柱状图给出 Accuracy (%) 与 Throughput，图例含 Nemotron-3-Nano-30B、Qwen3-30B-A3B-Thinking-2507、GPT-OSS-20B，横轴覆盖 Arena-Hard-v2 (Chat)、AIME25 (Math)、BFCLv3、SWE-Bench (Coding)、RULER @ 1M (Long Context) 以及 Throughput @ 8k/16k Tokens，图注称该混合架构能在推理基准与超长上下文任务上达到 state-of-the-art accuracy，同时相对同规模 Transformer MoE 提供吞吐提升。本章的易错点是：7:1 与 3:1 都是「线性／递推层 : 全注意力层」的比例，不要读反；$\gamma_t=f(x_t)$ 的「输入依赖、非状态依赖」是对偶性得以保持的前提，一旦门控依赖状态，并行训练与串行推理的等价就不再成立；另外 $v_t^\top D$ 这一项的来历要到下一章讲师对比 Gated DeltaNet 时才会补说，此处只需知道它是 Mamba-2 真实形式的一部分。

![MiniMax M1 幻灯片：各模型准确率柱状图、Softmax 与 Hybrid-lightning 得分表、FLOPs 随生成长度曲线，以及 7-to-1 混合的架构块示意](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0008_00-10-07.jpg)

![从线性注意力到 Mamba-2 幻灯片：并列写出线性注意力与带 γ_t 门控及 v_tᵀD 项的状态更新公式，附 Sequential/Parallel Mamba Block 结构图](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0010_00-12-48.jpg)

## Gated DeltaNet：输入门与 key 方向擦除 [14:15 - 18:49]

本章先把上一章的门控思路提炼成一条经验法则，再推到当下用得最广的变体。讲师说，若想把递推「推得更远、越来越复杂」，可操作的规则大致是：只要递推里各项的门控只用 input-dependent terms（只依赖输入的项）来 gating，也就是「no state dependence」，那么就仍然保有「并行运算用于训练、串行运算用于推理」这套漂亮的 duality。他紧接着点名 Gated DeltaNet 大概是当下用得最广泛的 state space model，已在不少论文里被检验，并在 Qwen 3.5 里有很好的大规模 scale-up。幻灯片 "Gated delta net (and friends)" 的引导句是 "Let's generalize things further – gate the input and selectively erase the state."，随后把 Mamba-2 与新公式并列：
$$
S_t = \gamma_t S_{t-1} + k_t v_t^\top \quad\text{and}\quad y_t = q_t^\top S_t + v_t^\top D \ \text{ with }\ \gamma_t = f(x_t) \qquad (\text{Mamba-2})
$$
$$
S_t = \gamma_t\left(I - \beta_t k_t k_t^\top\right) S_{t-1} + \beta_t k_t v_t^\top \quad\text{and}\quad y_t = q_t^\top S_t \ \text{ with }\ \gamma_t = f(x_t),\ \beta_t = f(x_t) \qquad (\text{Gated Delta Net})
$$
相对 Mamba-2，变化有两处：状态前多了 $(I-\beta_t k_t k_t^\top)$，写入项从 $k_t v_t^\top$ 变成 $\beta_t k_t v_t^\top$，而 $\gamma_t$ 与 $\beta_t$ 都写成 $f(x_t)$，仍是纯输入依赖。图上的解释文字是 "The gated delta net adds a 'no input operation' gate ($\beta = 0$)" 与 "And erases anything in the direction of the current key $(I - \beta_t k_t k_t^\top)$"，末行还写着 "Close relationships to various fast weight programming / test time training ideas."。

第二个门 $\beta_t$ 的含义需要单独拆开。讲师把它称作「no input operation gate」：$\beta_t=0$ 就意味着「don't take any of my current information, don't add it into my state」，即完全不把当前时间步的信息写进状态。他提醒听众大多熟悉 LSTM，这套机制与 LSTM 里控制「是否遗忘、是否放入当前时间步信息」的门非常像——$\gamma_t$ 对应遗忘门，$\beta_t$ 对应输入门——只不过二者最初是从很不一样的原理推导出来的。更有意思的是「更新方向」这一改动，它来自 DeltaNet 而非 Gated DeltaNet 的门控：讲师说，本来完全可以直接用 $1-\beta_t$ 之类的方式去 gating，但 DeltaNet 选择的是，在写入 $\beta_t$、把当前 key $k_t$ 的信息放进状态时，同时「project out my current key direction」——把之前沿同一个 key 方向存进去的旧信息擦掉。这就是蓝色项 $I-\beta_t k_t k_t^\top$ 的来历，它「essentially acting as a projector that is projecting out things in the $k_t$ dimension」。讲师立刻加了一个重要的限定：「That's not exactly right because you're not doing things like unit normalization」——因为并没有做单位归一化，所以它只是直觉上近似一个投影算子，不是严格的正交投影。这个精细的保留，正是本章最容易出错的地方。

补说上一章留下的 $v_t^\top D$。讲师在 15:33 道歉说忘了解释它：Mamba-2 还给 $y_t$ 的更新加了「basically a residual layer」，让当前步的 value 直接透传到输出；它「not really core to sort of the state update」，不是状态更新的核心，也不影响「为什么这是个好主意」的论证，写出来只是为了给的是真正的 Mamba-2 更新式而非删节版，因此可以先忽略。与此呼应，讲师还指出这个 projector 在不同领域被反复重新发现：如果去解某些 meta-learning least squares 问题，同样的更新式会出现；在 fast weight programming 或 test time training 里，研究者从非常不同的设计原则出发，最终也「ended up with the exact same kinds of solutions」。这与第 4 章幻灯片底部那行关于 fast weight programmers 的文献注记正好首尾相接，是本段埋得最深的伏笔。

生产侧落在 Qwen 3.5 与作为其前身的 Qwen Next 上。讲师称它们是「some of the best open source models available today」，用的正是这套架构：3:1 的 Gated DeltaNet : attention 混合，性能「very good, very reasonable」，推理特性很强——右侧面板显示 Qwen Next 随上下文长度不断增大，解码吞吐相对 Qwen 3 明显更高；中间面板把性能与各种闭源模型及上一代模型对比，结论是这套 hybrid 架构「doesn't seem to hurt performance very much at all」。把这些数字与上一章并列，就得到一条清晰的谱系：MiniMax M1 用 7:1、Nemotron 3 约 3:1、Qwen 3.5 / Qwen Next 用 3:1，比例都在压低全注意力层的占比，但谁也没有把它降到零。本章的易错点因而有三条：不要把 $(I-\beta_t k_t k_t^\top)$ 当成严格投影；不要忘记 $\beta_t$ 与 $\gamma_t$ 的「只依赖 $x_t$」是对偶性存续的前提，一旦引入状态依赖就破坏并行／串行等价；也不要把「no input operation」理解成整个时间步无操作，它只是不写入当前输入，状态仍会经 $\gamma_t$ 衰减后传递。下一章要回答的正是：这个比例到底能压到多低。

![Gated delta net (and friends) 幻灯片：并列 Mamba-2 与带 (I−β_t k_t k_tᵀ) 擦除项和 β_t 输入门的 Gated Delta Net 状态更新公式](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0012_00-15-30.jpg)

## 混合比例的经验研究与等价性问答 [18:49 - 21:35]

最后一部分转向一个更经验性的问题：混合比例究竟能推到多高。讲师先坦言，关于混合架构表现的「great controlled studies」并不多，比较系统的一份来自 ByteDance Seed 与 UC Santa Cruz 的研究者，他们把 Mamba-2、Gated DeltaNet 等架构作为「非全注意力层」，按混合层数的多少做扫描；他也预告结果「kind of messy」。幻灯片 "Hybrid performance" 给出了这份工作的完整出处：两侧是 ByteDance Seed 与 UC SANTA CRUZ 的标识，论文题为 "A Systematic Analysis of Hybrid Linear Attention"，作者列有 Dustin Wang*、Rui-Jia Zhu*、Steven Abreu、Yong Shan、Taylor Kergan、Yuyang Pan、Yuhong Chou、Zheng Li、Ge Zhang、Wenhao Huang、Jason Eshraghian，机构覆盖 UC Santa Cruz、ByteDance Seed、University of Groningen、CASIA、PolyU、M-A-P。左下角是四个 RULER 子任务的折线图（Single-key、Multi-key、Multi-value、Multi-query），横轴为 hybrid ratio、纵轴为 Average recall (HIGHER)；右下角的 "Recall vs. ratio" 图最关键，纵轴 Average recall (RULER) 取 0.0–0.5，横轴 Ratio (linear : full) 依次是 1:1、6:1、12:1、24:1、pure，一条带箭头的黑色虚线标出 full attention 的性能（约 0.42），黄、橙、蓝、红等曲线对应 gated delta nets 及若干变体。

这张图要读出的结论与讲师口述完全对得上：虚线是全注意力基线；从左往右是「non-full attention layers」也就是 RNN 层比例不断上升，整体可见性能退化；但对表现最好的几种架构（黄色、橙色、蓝色）来说，「at the low ratios, there's basically no hit」——低混合比例下几乎没有损失；「once you pass a certain point」之后长上下文性能才开始明显退化；到最右端 pure（纯 RNN）时，所有架构都有「very noticeable performance degradation」，其中红色曲线已跌到约 0.15。幻灯片底部的总结写得很克制："Not many controlled ablations, but some evidence of low losses at small hybrid ratios"。讲师补充说，在其他评测上也是同样的故事：single key retrieval 这类任务「all of these long context architectures explicitly optimize for」，参考价值偏弱；而看 QA 性能，随 hybrid ratio 上升「we see decreases in performance fairly steadily and fairly clearly」，一路降到 pure 点。把 1:1、6:1、12:1、24:1 与前面 7:1、3:1 的数字放在一起，就能理解产业界为何把比例集中在这个区间：低比例几乎无损，高比例则稳定变差。

右上角还有一张转写完全没念到、但信息量很大的表格，按 Model / Update rule ($S_t$) / Read-out ($o_t$) 三列把各类递推模型分成三族，值得完整转述。第一族是 "Vector-valued hidden state (classical / gated RNNs)"，含 HGRN、Bank (BGK/GRU) 等，读出为 $o_t = h_t \odot q_t$；第二族是 "Matrix-valued state via outer products"，含 RetNet/Lightning、GLA、Mamba、RWKV、HGRN-2/MetaLA，读出为 $o_t = S_t q_t$；第三族是 "Delta-rule / controlled-forgetting family"，列出 DeltaNet 的 $S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$ 与 Gated DeltaNet 的 $S_t = \alpha_t S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$，读出同样是 $o_t = S_t q_t$。这张表把本段第 4 到第 7 章的推导收进了一个统一坐标系：从「向量值隐藏状态」到「外积矩阵状态」再到「带 delta-rule 的可控遗忘」，正是状态从向量升级为矩阵、再升级为带擦除投影的演化路径，也解释了为什么 Gated DeltaNet 被归为「controlled-forgetting family」。

本段以一段问答收束，恰好回扣第 4 章标记的核心易错点。有学生问：开头讲的 parallel form 与 linear form，是不是「approximately equivalent to slightly degradation in performance」（近似等价、只是略有性能损失）。讲师澄清，他说的 recurrent 指的是 RNN in general，即「any recurrent formulation」，Gated DeltaNet 也算一种 recurrence；而这种递推形式与最上面那个 full softmax attention（带 $\rho$ 的）「is not equivalent」。真正的分界是：所有这类方法的第一步都是「drop the $\rho$（转写记作 row）然后变成 linear」，「this part is going to be lossy」；在此之后，从 linear form 到 recurrent form 的转换，「this equivalence, that is exact」。用一条链写就是
$$
\underbrace{\rho(QK^\top)V}_{\text{full softmax attention}}
\ \xrightarrow[\text{lossy}]{\ \rho\ \to\ I\ }\ \underbrace{Q(K^\top V)}_{\text{linear, dense, 可并行}}
\ \xrightarrow[\text{exact}]{}\ \underbrace{S_t=S_{t-1}+k_tv_t^\top,\ \ y_t=q_t^\top S_t}_{\text{recurrent / RNN, 固定状态}} .
$$
于是本章的易错点也就明确了：不要把「稠密形式与递推形式精确等价」误当成「与 full attention 等价」，也不要把纯 RNN 端的性能退化归因于递推写法本身——损失早在丢掉 softmax 那一步就发生了，而混合比例的实验只是在度量「用多少全注意力层能把这份损失补回来」。至此，从第 2 章的成本动机、第 3 章的常数因子工具箱，到第 4 章的结合律、第 5 章的对偶性、第 6/7 章的门控谱系，最后落到比例与等价性的边界，整段形成一条闭合的论证。

![混合架构性能幻灯片：ByteDance Seed 与 UC Santa Cruz 的 A Systematic Analysis of Hybrid Linear Attention，含更新规则分类表、RULER 子任务折线与 Recall vs. ratio 曲线](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0015_00-19-33.jpg)

## 线性注意力的收尾：当前步 value 的透传项与门控 D [21:35 - 22:50]

本段开头承接上一段的 linear attention（线性注意力）对偶形式。讲师先回应台下关于 "attention term below"（下面那个注意力项）的追问，重新点明：在 dual form（对偶形式）里，我们做的事本质上是 "taking these KVs and accumulating them with Qs"——把 KV 累积成状态、再用 Q 去读。这正是把 $(QK^\top)V$ 借结合律重排成 $Q(K^\top V)$ 之后得到的递推写法，写成固定大小状态逐步累加外积就是

$$S_t = S_{t-1} + k_t v_t^\top,\qquad y_t = q_t^\top S_t$$

其中 $S_t\in\mathbb{R}^{d_k\times d_v}$ 是与序列长度 $n$ 无关的常驻状态，$k_t\in\mathbb{R}^{d_k}$、$v_t\in\mathbb{R}^{d_v}$、$q_t\in\mathbb{R}^{d_k}$ 分别是当前 token 的 key、value、query。复杂度从原来 $QK^\top$ 带来的 $O(n^2 d_k)$ 降到约 $2nd_vd_k$ 次乘加的 $O(n d_k d_v)$：依赖对象由"序列长度 $n$（可达百万量级）"变成"隐藏维度 $d_k,d_v$（通常几千到几万）"。这是上一段"去掉 softmax + 换括号"的直接后果，也是本段后面讨论稀疏注意力、状态空间模型时反复对照的基准。

但纯粹的递推读出 $y_t=q_t^\top S_t$ 有一个直觉上不太够用的地方：状态里只装了历史 $kv^\top$ 的累积，当前时间步自身的 value 反而没有被直接送到输出。讲师指出，"you might actually want in your outputs to also have some value from your current time step"，因为 "the value for my current time step for my token is actually useful for various things downstream"。于是 Mamba-2 一类模型的真实形式会在读出端补一项：

$$S_t=\gamma_t S_{t-1}+k_t v_t^\top,\qquad y_t = q_t^\top S_t + v_t^\top D,\qquad \gamma_t=f(x_t)$$

这里 $D$ 是 another modulated gate（另一个调制门），用来控制 pass through（直通透传）的量。它 "acts a little bit like a residual connection"（有点像残差连接）：当前输入的 value 绕过状态，直接加到注意力的 outputs 上。要特别注意 $D$ 不是状态更新的核心，它只作用于读出；而 $\gamma_t$ 只依赖当前输入 $x_t$、不依赖状态，所以"稠密矩阵乘 / 递推"两种等价写法依旧成立，训练可并行、推理可串行的对偶性没被破坏。参数上，$D$ 与 $\gamma_t$、以及后面会提到的 $\beta_t$ 都属于 input-dependent gate（只依赖输入的门）。

讲师随即给这整条线性注意力路线收尾："So, that's linear time attention." 他给出的经验判断是：真正被大规模测过、跑通的方法，绝大多数都有 "this nice simple linear attention style recurrent form"（简洁的线性注意力式递推形式）；"There's a few exceptions, but for the most part, a lot of this has converged to almost very LSTM-like objects in the end"——有少数例外，但大体上最终都收敛到非常像 LSTM 的对象。这句话呼应前面讲过的 Mamba-2 的 $\gamma_t$（对应遗忘门）与 Gated DeltaNet 的 $\beta_t$（对应输入门），也为 [32:39] 那段"状态空间模型与 LSTM 相似"的问答埋下伏笔；转写里 [22:50] 之后还有一句 "Um What do you want to say? Uh right, okay." 属于现场停顿，不含技术信息，略过不表。

与前后内容的关系上，这一小段既是上一段线性注意力推导的尾巴，也是 [22:50] 起转向"另一条完全不同的路线"的过渡：讲师明说线性注意力带有 "complexity theoretic flavor"（复杂度理论色彩），下面要讲的 sparse attention 不追求把二次变线性，而是走"少看一些 token"的系统路线。易错点有两处：一是别把 $v_t^\top D$ 当成状态更新的一部分，它只是读出端的残差旁路，加上它只是为了完整呈现 Mamba-2 的真实形式；二是别把"去掉 softmax"和"线性形式转递推形式"混为一谈——前者有损、后者精确等价，本段 [29:54] 的问答还会再确认一次：去掉 softmax 并不会带来已知的 training stability（训练稳定性）问题。另一个容易踩的坑是：一旦某个门控开始依赖状态 $S_{t-1}$，上面这套对偶性就会失效，这正是后面所有门控设计都要守住的约束。

## DSA 的引入：闪电索引器与 top-k 细粒度选择 [22:50 - 24:30]

讲师在 [22:54] 明确换题：前面讲的 linear attention 带有 "complexity theoretic"（复杂度理论）的味道，现在要讲 "another attention optimization"——它没在前面提过，但同属 efficient attention（高效注意力）家族里 "kind of important" 的一支。出处是 DeepSeek 在 DeepSeek 3.2（讲师原话带 "I think"；幻灯片写作 V3.2 / v3.2）里用的 DSA。这里有个现场语音双关：他念到 DSA 时顺口带出 "not do sparse attention, that's DeepSeek attention"，随即补一句 "Oh, that was not intentional pun"——缩写既可以读作 DeepSeek（Sparse）Attention，又恰好谐音 "do sparse attention"。核心思路一句话：[23:29] "instead of computing attention over all the tokens, what you're going to do is you're going to first have a lightweight indexer that's going to subset a bunch of tokens"——先用一个 lightweight indexer（轻量索引器）从 very long context 里筛出一小撮 token，规模 "much smaller than the full sequence"，之后只在这个小得多的子集上做 full attention（完整注意力）。

![DSA 原型幻灯片：闪电索引器打分公式(1)与细粒度 top-k 选择公式(2)](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0018_00-23-36.jpg)

幻灯片把这套机制写成两个组件："a lightning indexer"（闪电索引器）和 "a fine-grained token selection mechanism"（细粒度 token 选择机制）。索引器在 query token $h_t\in\mathbb{R}^{d}$ 与 preceding token（前序 token）$h_s\in\mathbb{R}^{d}$ 之间计算 index score（索引得分）$I_{t,s}$，"determining which tokens to be selected by the query token"：

$$I_{t,s}=\sum_{j=1}^{H^I} w^I_{t,j}\cdot \mathrm{ReLU}\!\left(q^I_{t,j}\cdot k^I_s\right) \tag{1}$$

$$u_t=\mathrm{Attn}\!\left(h_t,\ \{c_s\mid I_{t,s}\in \mathrm{Top\text{-}k}(I_{t,:})\}\right) \tag{2}$$

其中 $H^I$ denotes the number of indexer heads（索引器头数），$q^I_{t,j}\in\mathbb{R}^{d^I}$ 与标量 $w^I_{t,j}\in\mathbb{R}$ 都 derived from the query token $h_t$，而 $k^I_s\in\mathbb{R}^{d^I}$ derived from the preceding token $h_s$。公式 (1) 就是转写里 "it takes the QK inner products, it takes a ReLU, and then it's going to have these weights that are derived from the preceding tokens, and then this is going to give you the activations for each of the positions" 的精确形式：先做低维内积 $q^I_{t,j}\cdot k^I_s$，过 ReLU，再按头加权求和，得到每个位置的激活值。

幻灯片正文还给了转写没念到的工程细节，这是本次详写最该补进来的增量：选 ReLU 是 "for throughput consideration"（出于吞吐考虑）；并且 "Given that the lightning indexer has a small number of heads and can be implemented in FP8, its computational efficiency is remarkable"——头数少、可用 FP8 实现，因此效率极为可观。公式 (2) 对应 fine-grained token selection：拿到每个 query 的 $\{I_{t,s}\}$ 后，"retrieves only the key-value entries $\{c_s\}$ corresponding to the top-k index scores"，再在 $h_t$ 与这些 sparsely selected key-value entries 之间套用常规注意力得到 $u_t$。转写里讲师说 forward pass 的 mechanics "pretty simple"：普通 Q、K 过索引器 → QK 内积 → ReLU → 由前序 token 导出的权重 → 各位置激活 → "pick this through a top K" 挑出若干位置 → 用 usual attention computation "that emits a key value"。可调参数就三个：$k$ 可控、$H^I$ 很小、$d^I$ 是专为索引准备的低维投影维度，它们共同决定索引器的常数开销。

与前后内容的关系：这条路线是上一段 "linear time attention" 的替代视角，也和更前面的 hybrid（混合架构）不同——幻灯片标题直接写 "Alternative to hybrids: sparse adaptation"，副标题 "Instead of attending to every token, do sparse attention (DSA)"，底部两行大字是 "The indexer can be very lightweight, giving significant gains" 与 "Can be 'post hoc' adapted after dense short context pretraining"。也就是说它不是改状态递推，而是靠"少算几个 token"。易错点：DSA 不是线性时间——索引器仍要对全体 token 算 QK 内积，是 all-to-all、quadratic（二次）的，讲师在 [26:46] 会专门 "I'll go back one slide" 回去强调；把 DSA 误当线性注意力，就会错判它的成本结构。另一个易错点是公式 (2) 里的 $\mathrm{Top\text{-}k}$ 是 non-differentiable（不可微）的离散选择，并不能直接反传，这正是 [24:43] 说"带着索引器从头训练很烦"的根因。

## post-hoc 适配、实证效果与「非线性时间」的定位 [24:30 - 27:29]

幻灯片底部那句 "Can be 'post hoc' adapted after dense short context pretraining" 在转写里被展开成一整套训练策略。讲师说 DSA 和 GLM 都显示 "you don't actually have to train the model with this"，因为 "training with this like indexer thing is maybe pretty annoying and very complex"。做法是：先 "just train a normal transformer"（正常训练一个稠密 Transformer），等到做 long context extension（长上下文扩展）时 "you actually then drop in this lightning indexer"（把闪电索引器插进去），再在一个 "extension stage that's separate from pre-training"（独立于预训练的扩展阶段）里训练模型去适应它。因此 "actually the computational cost of doing this is itself pretty light"——额外开销很轻。用式子概括这个替换：预训练阶段覆盖全部 $n$ 个位置的标准 $\mathrm{Attn}(Q,K,V)$，在扩展阶段被限定为

$$\mathrm{Attn}\big(h_t,\ \{c_s\mid I_{t,s}\in\mathrm{Top\text{-}k}(I_{t,:})\}\big),\qquad k\ll n,$$

即把参与注意力的候选集从 $n$ 换成受控的 $k$。

![同一张 DSA 幻灯片在 24:57 再次出现，底部强调索引器极轻且可在稠密短上下文预训练后 post hoc 接入](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0019_00-24-57.jpg)

效果方面，讲师在 [25:14] 特意强调 DeepSeek V3.2 提出 DSA 后 "this model is quite good"，V3.2 "matches all of the other frontier models at the time"，点名 Claude 4.5 Sonnet 与 Gemini 3。幻灯片把对比量化了：上半部柱状图的图例是 DeepSeek-V3.2-Speciale、DeepSeek-V3.2-Thinking、GPT-5-High、Claude-4.5-Sonnet、Gemini-3.0-Pro，横轴分 Reasoning Capabilities（AIME 2025、HMMT 2025、HLE、Codeforces）与 Agentic Capabilities（SWE Verified、Terminal Bench 2.0、τ²-Bench、Tool Decathlon），纵轴 Accuracy / Percent (%)；AIME 2025、HMMT 2025 的柱子都在 90 以上，HLE 一档约 30 上下，Codeforces 用右侧 0–3000 刻度能读到 2701、2386 这类数值。右侧八张小图是 GLM-5 与 DeepSeek-V3.2、Claude Opus 4.5、Gemini 3 Pro、GPT-5.2 (high) 的分组对比，可见 50.4、77.8、73.3、56.2、75.9、89.7 等分数。

![DeepSeek-V3.2 与 GLM5 的多基准柱状图、Prefilling/Decoding 成本曲线及 RULER 长上下文检索表](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0020_00-26-18.jpg)

更贴题的是左下两张成本曲线：横轴 Token Position 从 0K 到 128K，纵轴 Cost Per Million Tokens；(a) Prefilling（预填充，即 "attending to the input"）里 DeepSeek-V3.1-Terminus 从约 0.15 线性涨到 0.75，而 DeepSeek-V3.2 几乎压平在 0.2 附近；(b) Decoding（解码，"where you're actually generating"）里前者从 0.45 一路涨到 2.05，后者仍近乎水平。讲师的解释很朴素："If you're only paying attention to a very small subset, you're going to dramatically be reducing your costs of doing attention." 右下角 Table 6 是 "RULER benchmark results for the GLM-4.7-Flash with DSA"，列为 4K / 8K / 16K / 32K / 64K / 128K，说明文字写明 "The warmup-only variant trains only the indexer while keeping the base model frozen; the full DSA variant jointly trains both for 150B tokens"；三行分别是 GLM-4.7-Flash、GLM-4.7-Flash + DSA warmup、GLM-4.7-Flash + DSA，数值大体从 4K 的 97 分落到 128K 的 71–79 分，说明完整 DSA 训练相对全注意力损失很小。讲师在 [25:56] 也点名 GLM 5 是 "one of the best open models that are available, like period"，其消融比较了 "DSA with some warm-up"、"just do DSA"、"don't do DSA at all"，结论是 "if they do full DSA training, you don't lose very much in performance relative to full attention, even at long context retrieval tasks"——而这类长上下文检索任务 "are fairly difficult to do with RNN-style architectures"。

与前后关系：这一章把上章的公式落到了真实模型与真实数字上，同时立刻自我修正定位。讲师在 [26:43] 说 "this is a totally alternative view of how to reduce attention costs"，紧接着 "Notice, I'll go back one slide, this is not linear time, right? The indexer still has to operate on full all-to-all attention, cuz you got these QK inner products that are not going to be linear in cost"；但可以 "making this lower dimensional or making this W very small" 把索引器变得 much cheaper。易错点正是把 "sparse" 等同于 "linear"：收益来自常数因子和候选集规模，不是渐近复杂度的改变；而且 V3.2 与 GLM 5 的证据是"性能几乎不掉 + 成本显著降"，不是"完全无损失"，转写用的是 "don't lose very much" 这种留有余地的措辞。伏笔：[27:12] "it turns out that this idea of top K selection is going to be core to the next part of this lecture"，[27:26] 讲师停下接受 efficient attention 提问、再转向 MOEs——top-k 这个原语会在混合专家里以另一种形式回来。

## 高效注意力现场问答：二次复杂度、常数因子与长上下文扩展 [27:29 - 30:23]

第一个提问直指要害：索引器的 time complexity 是不是 quadratic？讲师答 "Quadratic, yeah... Cuz it has to in order to know what to select, it does have to look at everything"，并补一句 "there's no clever like state transition stuff that happens here. It is really brute force inner products"——没有状态转移的巧劲，就是暴力内积。紧接着追问：既然第一阶段遍历全体是二次，"the step afterwards is also quadratic as well"，收益从哪来？讲师把它定性为 "It's kind of like a systems-y trick almost"（几乎是个系统层面的技巧），给出两处可压的地方：一是把索引器做得 "much more lightweight"，手段包括 "much lower precision, which they talk about" 以及 "ways that are lower dimensional by projecting the Qs and Ks further just for the indexing"（仅为索引把 Q/K 再投影到更低维）；二是第二阶段虽然也是二次，但 "it's quadratic on a shorter context length, cuz it's top K, we can control K"。用式子写就是

$$\underbrace{O(n^2\, d^I)\,H^I}_{\text{索引器：二次但常数极小}} \;+\; \underbrace{O(k^2\, d)}_{\text{注意力：二次但 } k\ll n},\qquad k=\text{人为设定的可控上界}.$$

讲师的原话总结是 "this is expensive, but small, right?"（贵，但规模小）。

由此引出本讲反复出现的工程教训："sometimes don't get too stuck up on the quadratic quadratic versus not. Sometimes the constant factors are really really important."（别太纠结二次还是线性，常数因子有时极其重要。）这与本段开头 linear attention 的 "complexity theoretic flavor" 正好成对照：一条路改渐近阶，一条路改常数与有效长度。具体参数上，$H^I$ 头数少、$d^I$ 远小于 $d$、索引器可跑 FP8，而 $k$ 是人为设定、与输入总长 $n$ 解耦——这几个量就是全部可调旋钮；对照上一章 kf_0020 的成本曲线，横轴一路到 128K token，正是 "shorter context length" 所相对的那个长上下文尺度。转写没有给出 $k$ 的具体数值，但下一问补上了它的取值原则。

关于何时插入索引器，[28:44] 的问答把它定位成 "Or it's part of like a continued pre-training style thing"（继续预训练式的一环）。理由是 "all the models, you know, you don't train long context from scratch for compute and various other reasons, you train a shortish context model first, and then you've got these long context extension stages"。于是 "we're going to do the second phase anyway, why don't we bolt the long context, you know, cost savings on at the same stage, right?"——反正要做第二阶段，顺手把省成本的改造一起挂上去。[29:28] 给出通用配方："short context pre-training, long context extension, then post-training"（短上下文预训练 → 长上下文扩展 → 后训练）。讲师也坦承这事 "it's kind of surprising that it works, you know, honestly"，因为你要 bolt on 的是一个 "frankly scary-looking top K non-differentiable object"（看着挺吓人的、不可微的 top-k 对象），并预告 "as you'll see in the next part, um maybe you shouldn't be so scared of top K selection"。

[29:39] 的提问是 $k$ 是否与短上下文长度同量级，讲师确认 "Yeah, that's right. Like you would pick values that are like much closer to short context performance and you would bound it, right? So, regardless of the input size, they would be bound."——即 $k$ 取接近短上下文长度的值并设为上界，无论输入多长都封顶。最后 [29:54] 有人问去掉 softmax 是否对 context length 有 "Training stability issues"，讲师答 "I don't think so. I'm not aware of like, you know, documented linear uh soft max causing lots of attentions. If anything, soft maxes are usually more of the problems for various reasons. Um, so I think this would actually improve the stability properties of the architecture."（含糊处按原意理解为：没见过相关记录，反而 softmax 才是麻烦来源，去掉它可能改善稳定性。）易错点：别把"top-k 不可微"和"训练不稳"划等号，讲师明确表示没有见过相关记录；也别以为 $k$ 会随上下文变长而放大，它是被 bound（封顶）的常量。与前后关系：这一章既回收了上一章的公式，也为下章 FP4、状态空间模型等问答定下"系统 + 经验"而非"理论最优"的讨论基调。

## 更多问答：注意力的未来、FP4 低精度与状态空间模型的表达力 [30:23 - 34:22]

有同学追问"下一代更好的注意力机制"会是什么样（提问里提到 "low community cost and the low latency"，转写含糊，不臆测其确切含义）。讲师先自谦 "I don't know if I have good predictions for what the future attention architecture looks like"，给出 "my somewhat cop-out answer"：大概率是 "we throw all of these tricks in"——把各种成功配方全堆进去，"In the same ways that architectures have gotten very complex by taking all of these successful recipes and throwing them in"。他还提出更高一层的判断："there's a even broader layer or like a higher level layer of basically using post-training to have the model manage its own context"（用后训练让模型自己管理上下文），点名 compaction（压缩）、retrieval（检索）这类 "an added layer on top"（叠加在模型之上的层）。他预期未来的工作会是 "integration of these different layers"，并重复前面的收敛判断：至少 "the linear time attention stuff has converged a lot into LSTM-like or linear attention-like architectures and I don't really see that changing too much in the near future"。这与 [22:43] 那句 "converged to almost very LSTM-like objects" 首尾呼应。

[31:41] 的提问是 "you speak about like lower precision attention, but I think is FP4 attention possible?"。讲师答 "I think the indexer is partially motivated by things like this. Like FP4 attention definitely possible"——低精度注意力可行，而且索引器的设计动机之一正来源于此。但他立刻指出 drawback："the loss of precision for things like soft maxes where like small underflow, overflow, these kinds of things can really start to matter can sometimes be pretty problematic"（下溢、上溢会开始变得要命）。于是合理的折中是 "let's do selection with low precision and then do a full precision computation afterwards cuz then the value vectors can be like more finely added together and weighted together"（value 向量可以更精细地被加权求和）。可用一行式子概括这个两段式：

$$\text{select: } I_{t,s}\ \text{in FP8 / FP4}\quad\longrightarrow\quad \text{compute: } u_t=\mathrm{Attn}(\cdot)\ \text{in full precision}.$$

讲师也没把话说满："I don't know if there's, you know, really good ways of um of doing full attention in in FP4 or even lower precision of some kind." 参数上的锚点是幻灯片里索引器可用的 FP8 与提问中的 FP4，以及 softmax 的 underflow / overflow 边界。

[32:39] 最后一个大问题：compared to the transformers，state space models（状态空间模型，SSM）的 downside 是什么、训练是否更复杂？讲师先说 "there has to be some downside or otherwise everybody..."，然后给出答案：expressive power（表达能力）。"the all-to-all connection in soft max attention is incredibly powerful. It's also very easy to train, right?" 他回顾历史：相比 LSTM，attention 过去很大的优势是 hardware efficiency（硬件效率，"you could train in parallel"）；SSM 之所以现在能流行，"despite their similarity to LSTMs and similar drawbacks in terms of representational power"，是因为线性注意力带来的 "well-understood duality"（对偶性）——"you can go from, you know, this RNN form to the more dense matrix multiply form. And the matrix multiply form allows for computational efficiency"。于是硬件效率这一项 "that trade-off has now been sort of checked off"（被划掉了），剩下的取舍就是有限状态必须把整个上下文压进 $S$：

$$\text{finite state } S\in\mathbb{R}^{d_k\times d_v}\ \Rightarrow\ |S|=d_k d_v \ll n\,d,\qquad \text{若 } d_k d_v \sim n d \text{ 则「good to go」但要付 very large costs}.$$

用讲师的话说，"if you have a finite state and you have to carry everything through, you know, you're going to be losing some information relative to just carrying everything"（相比直接把全部历史留着，必然丢信息）。

[33:50] 追问上下文长度本身怎么算，讲师答：在状态空间里 "you have to basically kind of think about the context length relative to the size of your state"——"if you have the state the size of your context, then you're good to go, but then you're paying these like very large costs, right? So, really the point is the free lunch is if you want a really tiny state, it's really hard to compress all the information in a big context"。他留了余地："It might be possible one day to to not have any trade-offs, but I think thus far that's where the trade-offs are showing up." 易错点：别把"SSM 训练更难"当成讲师给出的答案，他明确把 downside 归到 expressive power；也别把 linear attention 的 RNN 形式与 SSM 完全等同——对偶性解决的是可并行训练，不是信息压缩。与前后关系：这一章是高效注意力板块的收束，[34:22] 一句 "Okay, so now I'm going to talk about mixture of experts" 直接切到本段后半的 MoE 主题；前面反复出现的 top-k、finite state、low precision 都会在 MoE 的路由与专家并行里以另一种形式回来。

## 混合专家 MoE：概念、动机与把 FFN 换成稀疏专家 [34:22 - 37:45]

[34:22] 转入 mixture of experts（混合专家，MoE）。讲师先定调："conceptually they don't really change the game"——概念上并不改变游戏规则，一种理解是 "they are just a more efficient MLP"（就是一个更高效的 MLP）：你把 MLP 换掉、有人白送你一个更高效的 MLP，差不多就是 MoE。为什么要专门讲？他给了两条理由：一是 "everyone's doing MOEs these days. Everyone's shipping MOEs these days"（如今人人都在做、都在发 MoE 模型），想成为 "people who deeply understand language models and how they work" 就得懂它；二是 MoE 有 "really interesting um sort of mechanical components"——"if you look at how MOEs are built and what primitives they use, you will kind of quickly see MOE-like primitives in many places"，从而意识到 "wow, I can do more interesting things with neural networks"。这第二条正是把前半段的 top-k 与后半段连起来的线索。幻灯片用一屏新闻与论文卡片佐证"人人都在做"：左上角那张参数散点图标着 GPT-MoE-1.8T、MT-NLG 530B、GPT-3 175B、Chinchilla、PaLM、BLOOM、Microsoft T-NLG、Megatron-NLG、WuDao 2.0、BERT Large 等，下面写着 "GPT4 (?)"；此外还有 Mistral AI 那条宣布 8x22B 模型的 magnet 链接推文、黑底的 Grok、DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models、DeepSeek-V3 Technical Report、"Llama 4: Leading Multimodal Intelligence"（含 Llama 4 Behemoth / Maverick / Scout）、以及 OLMoE: Open Mixture-of-Experts Language Models（列出 Niklas Muennighoff、Luca Soldaini、Dirk Groeneveld 等作者与 Allen Institute for AI、Contextual AI、University of Washington、Princeton University 等机构）。

![Mixture of experts 幻灯片：GPT4(?)、Mistral 8x22B、Grok、DeepSeekMoE、Llama 4、OLMoE 等模型与论文卡片](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0026_00-34-41.jpg)

"What's a MoE?" 那张图把 Dense Model 与 Sparse Model 并排画出来。左边稠密块自下而上是：输入 $x$ → Self-Attention → Add + Normalize → FFN Layer → Add + Normalize → 输出 $Y$；放大框里对 $x_1$ "The"、$x_2$ "Dog" 两个 token，Self-Attention 之后各过一个 FFN，再 Add + Normalize 得到 $y_1,y_2$。右边稀疏块把 FFN Layer 换成 Sparse FFN Layer，放大框里出现 FFN 1、FFN 2、FFN 3、FFN 4 四个专家，两个 token 各自只连到被选中的少数专家。图下两行字是 "Replace big feedforward with (many) big feedforward networks and a selector layer" 与 "You can increase the # experts without affecting FLOPs"，右下角标注 [Fedus et al 2022]。讲师在 [35:33] 的描述一致：usual MLP 里 "You've got your attention, you've got some normalization, and then you've got a feed forward"，而 FFN "This is kind of where a lot of the dense, big information processing happens"（大量稠密信息处理发生的地方）。

![What's a MoE 幻灯片：Dense Model 与 Sparse Model 的 FFN / Sparse FFN Layer 架构对比](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0027_00-36-13.jpg)

构造上的关键数字是 4。讲师先说 "let's say I take my big FFN and I'm going to cut them up into smaller FFNs, right? So, I have now have four feed forward networks"，随即又特意纠正："actually let's say I don't cut them up into smaller pieces. Let's say they're the same size as my original FFN"（不是切小，而是四个和原来一样大的 FFN）。加上 "by some magic" 的选择系统，"I have a system that tells me which FFN to pick for every input"，结果就是：

$$N=4:\qquad \text{Params}\approx 4\times P_{\mathrm{FFN}},\qquad \text{FLOPs}_{\text{fwd/bwd}}\approx 1\times P_{\mathrm{FFN}}.$$

"I have 4x the parameters or at least for my FFN, right? I have four FFNs worth of parameters, but on any forward or backward pass, I'm only going to pay one FFN worth of cost. That's great." 参数量翻 4 倍，前反向却只付一个 FFN 的钱。讲师把这称为原始论文里 "very um parameter-centric view"（以参数为中心的视角）：出发点是 "Well, let's say you just wanted more parameters because you believe more parameters are good"，既要更多参数、又不愿付更多算力成本，"you need something like something on the right"。于是 mental model 一句话："You want to increase parameters without affecting your flops"（在不影响 FLOPs 的前提下增加参数）。

[36:54] 转向"为什么流行"：Hugging Face 上 past a certain size，"everything that you get seems to be an MOE"；经验规律是 "if you keep the total compute the same, but you just increase the number of sparse parameters, somehow the models are generally getting better"，这被看作 "more parameters are in fact generally good even if only a subset of them are active at a time" 的证据。与前后关系：这一章把前半段的 "top-k selection" 从 token 选择平移成了"专家选择"，也是下一章 Switch Transformer 曲线的铺垫。易错点有三：一是别把 MoE 理解成"把大 FFN 切成小块"，讲师明确说专家是 same size as my original FFN，切片只是他先抛出又否掉的假想；二是 selector layer 的路由是稀疏、离散的，图里那句 "without affecting FLOPs" 只在计算量上成立，参数量与显存都实打实变成 $N$ 倍；三是别把 MoE 当成注意力改造，它替换的是 FFN / MLP 块，与前面所有 attention alternatives 是两条主线。

## 稀疏参数的收益曲线、激活参数对比与新的并行轴 [37:45 - 40:42]

讲师说整场讲座会 "refer to some of the original Google papers extensively"，这里点名 "Fedus et al. in 2022 where they were doing I think the switch transformer for this one"。幻灯片标题 "Why are MoEs getting popular?"，副标题 "Same FLOP, more param does better"。左图横轴 Sparse Model Parameters（对数刻度，标了 $10^9$、$10^{10}$），纵轴 Test Loss（4.8–6.0），一条下降曲线上依次标着专家数 1e、2e、4e、8e、16e、32e、64e、128e、256e：在 "active parameter stays the same"（激活参数固定）的前提下，专家数从 1 增到 256，语言建模 test loss 从约 6.0 一路降到约 4.8，"is just decreasing, decreasing, decreasing"。转写里的评价是 "That's lovely if all you care about is your forward pass flops"。用式子概括这张图的变量关系：

$$N_{\text{active}}=\text{const},\quad E:1\to256,\quad N_{\text{total}}\approx E\cdot N_{\text{active}},\quad \mathcal{L}_{\text{test}}:6.0\to4.8.$$

右图把同一结论放到训练算力维度：横轴 Training Step（0 到 $4\times10^5$，刻度 0、1、2、3、4 配 1e5），纵轴 Neg Log Perplexity（−2.0 到 −1.2），图例是 Switch-Base: 128e（蓝）、Switch-Base: 64e（红）、Switch-Base: 32e（绿）、Switch-Base: 16e（橙）以及 T5-Base（紫）。五条曲线都先陡升后趋缓，128e 最高（约 −1.30）、64e 次之、32e、16e（约 −1.40）依次排开，稠密基线 T5-Base 明显最低（约 −1.56）。讲师的读法是 "as you increase uh your training compute on the x-axis, if you have more experts for the same amount of training compute, you just get better performance, right? So, for both training and inference, um MOEs just give you a free win"。数字上 16e / 32e / 64e / 128e 是专家数，$4\times10^5$ 是训练步数量级，−1.2 ~ −2.0 是负对数困惑度区间，右下角引用 [Fedus et al 2022]，这些都是幻灯片上独有的信息。

![Why are MoEs getting popular 幻灯片：专家数 1e–256e 的 Test Loss 曲线与 Switch-Base 训练步数曲线](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0028_00-37-46.jpg)

[37:56] 补充独立复现：AIQ 团队（AI2）的 Olmo 论文——"their like open source mixture of experts uh analysis and sort of training paper"——"they see the same thing as well"：无论 training loss、validation loss 还是 downstream benchmark performance，"in training time, right? Um, these models are doing much better as MOEs um compared to their dense counterparts. So, something like uh two times faster or something like that um training an MOE relative to a dense model"，即约 2 倍量级的训练加速。[38:50] 又说 "I think the proof is in the pudding"：已发布的 MoE 以 activated parameters（激活参数）衡量明显强于稠密模型，而这才是关心 inference flops、乃至 "the cost to train and serve these systems, right? Like the the all-in cost" 时真正该看的量。散点图把这个对比画得很直观：横轴 Activated Parameters (Billions) 0–100，纵轴 Performance (MMLU) 55–80，红色五角星是 DeepSeek-V2（约 21B 激活、MMLU 约 78.5），周围散着 Mixtral 8x22B、LLaMA 3 70B、Qwen1.5 72B、Command R+、DBRX、Qwen1.5 32B、Mixtral 8x7B、LLaMA 3 8B、DeepSeek 67B、Grok-1、Command R、Mistral 7B、LLaMA 2 70B、LLaMA 2 34B、LLaMA 1 65B、LLaMA 2 13B、LLaMA 1 33B 等点，图例按 LLaMA 1 Family、LLaMA 2 Family、LLaMA 3 Family、Mixtral Family、Command R Family、Qwen1.5 Family 分族连线，底部一句 "Highly competitive vs dense equivalents"。讲师对应地说 "when DeepSeek V2 and like the earlier DeepSeek MOEs came out, you know, it was really kind of a big shift"：用 "much fewer active parameters" 拿到 "just as good if not better MMLU performance uh compared to everyone else"。

![Why are MoEs getting popular 幻灯片：激活参数量对 MMLU 性能的散点图，DeepSeek-V2 以红色五角星标出](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0029_00-39-18.jpg)

[39:42] 收尾时补上"为什么必须学 MoE"的最后一条系统理由："MOEs give you another axis of parallelization"（多一个并行维度）。讲师预告 parallelization "in systems in three three more lectures, I think"，但先点明 "LLMs are very big. They don't fit in single devices either to train or inference. So, you want many different ways of cutting up your model to efficiently serve or train them"。MoE 的好处是 "your experts within each FFN naturally come in different, you know, chunks, right? Each expert is a natural chunk. You can put them on different devices, and then you can route the activations to different devices"——每个专家天然是一块，可放到不同设备再把激活路由过去，这就是后面会展开的 expert parallel（专家并行），"which gives you an additional systems thing that you can optimize"。易错点与局限：这两张图的结论都建立在 "same FLOP / same active params" 这个受控前提上，脱离前提不能直接说"参数越多越好"；"free win" 只是讲师在计算量意义上的措辞，参数量、显存与路由开销并没有免费；而 "two times faster" 是转写里的口语估计（"or something like that"），不是精确测值。与前后关系：这一章把上章的参数量公式换成了实测曲线，也为下一章的产业格局与通信瓶颈埋下"专家要分布到不同设备"这一前提。

## 产业格局、专家并行通信与训练稀疏性的伏笔 [40:42 - 43:10]

[40:43] 讲师给出地域与产业观察："in the west, you know, I think there's been several open-source releases that are big strong MOEs"，大模型侧是 Llama 4 和 OpenAI 的 GPT-OSS，"both top-tier models in their own right"；但 "Unfortunately, I guess open-source model releases in the west have kind of stalled for the most part. So, there aren't quite as many being released now"。相反 "a lot of the action for both MOE research and training has happened in China"——Qwen、DeepSeek 以及 "some others like MiniCPM did some of the earliest work kind of training and popularizing MOEs. They continue to do really great work on MOEs"。早期证据是 Qwen 1.5 MOE："they showed that they're like small 2.7 billion parameter active model, you know, was doing better than many of the 7B models at the time"。讲师认为正是 DeepSeek 与 Qwen 这批 "early proof of work" "really convinced everyone else in the open-source community that this was the right way to go"。数字锚点：2.7B 激活参数对标 7B 稠密模型。

![Some MoE results – from the west 幻灯片：Llama 4 Maverick 等模型的基准分数、每百万 token 成本与 Humanity's Last Exam 柱状图](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0030_00-40-51.jpg)

这张幻灯片把这批模型量化成表格与图。标题 "Some MoE results – from the west"；左侧表格列是 Llama 4 Maverick、Gemini 2.0 Flash、Deepseek v3.1、GPT-4o，行名与数值包括：Cost per 1M input & output tokens (3:1 blended) 分别约 $0.19–$0.49、$0.17、$0.48、$4.38；Image Reasoning 的 MMMU 73.4 / 71.7 / (no multimodal support) / 69.1，MathVista 73.7 / 73.1 / — / 63.8，ChartQA 90.0 / 88.3 / — / 85.7，DocVQA_human 94.4 / — / — / 92.8；Coding 的 LiveCodeBench 43.4 / 34.5 / 45.8/49.2* / 32.3*；Reasoning & Knowledge 的 MMLU Pro 80.5 / 77.6 / 81.2 / —，GPQA Diamond 69.8 / 60.1 / 68.4 / 53.6。右侧黑底柱状图是 "Humanity's Last Exam — Expert-level questions across subjects"，纵轴 Accuracy (%)，柱顶能读到 10、14.7、17.3、10.8、31.5、27.1、15.4 一类数值。底部结论句 "MoEs are most of the highest-performance open models, and are quite quick" 同时点了性能与速度，正好对应上章的 all-in cost 视角。

表里最值得对照的是成本那一列：$0.17、$0.48 与 $4.38 之间相差一个数量级以上，而 MMLU Pro 80.5 / 77.6 / 81.2、GPQA Diamond 69.8 / 60.1 / 68.4 这些分数并没有因为"便宜"而明显塌陷。这正好把上一章散点图的结论从"激活参数"推进到"每百万 token 的实际价格"：以推理成本衡量，稀疏模型在同等质量下更省，转写里 [39:00] "I think the proof is in the pudding" 和 [39:14] "which is what matters if you care about inference flops" 说的就是这件事。需要说明的是，这些逐项数字讲师在转写里并没有念出来，它们只出现在画面上，属于本次详写要补进正文的独有信息；带星号的 45.8/49.2* 与 32.3* 在表格里本身带注脚标记，此处按原样保留、不臆测其含义。另一处易错点是别把 "no multimodal support" 读成模型缺陷对比——它只是说明 Deepseek v3.1 在图像推理那几行没有可填的数，不能据此推断多模态能力高下。

[41:57] 的提问回到 expert parallel："isn't there like a communication bottleneck that's built up because these experts are communicating with each other?" 讲师先肯定 "Yes. So, in some sense, you pay the communication cost of like shipping an activation over"，并说 "I will not explain, but I will show you one trick of like how you reduce that"（先不展开，后面会给一个降通信的技巧）。随后把权衡写成一句可对照的关系：

$$\underbrace{\text{aggregate FLOPs}\uparrow,\ \text{memory use}\downarrow}_{\text{收益}} \;\Longleftrightarrow\; \underbrace{\text{All-to-All comms}\uparrow}_{\text{代价}},$$

"You're getting more aggregate flops. You're reducing your memory use. In exchange, you're going to pay for comms, right? So, you know, it's highly dependent on your topology and all these other things whether this is going to be a net win."（是否为净收益强依赖网络拓扑。）这与 [39:53] 新增并行轴的说法是一体两面：专家能分到不同设备，代价就是要把激活搬过去。

最后 [42:33] 有同学从训练角度追问，讲师打断强调 "this is the important part. During training, they are also sparse, right? This is really the key thing that makes MOEs hard"，并自嘲在 "spoilering sort of the key idea from a few slides down"。他的推理是：如果训练时 "we activated all the experts during training"，那会很省事——"then we get to see, oh, which expert is good for my input. So, you you get to learn how to route"；难就难在 "during training, you only have, you know, one or K experts active. So, you don't know what happened with the rest of them. You only know the ones that activated. Despite this, you must somehow learn to route." 若把当前 token 的激活专家集合记作 $\mathcal{A}(x)$，训练时实际成立的是

$$|\mathcal{A}(x)|\in\{1,K\},\quad K\ll N,\qquad \nabla_{\theta_i}\mathcal{L}=0\quad (i\notin\mathcal{A}(x)),$$

即只有被选中的 1 或 $K$ 个专家参与前向、也只有它们拿到梯度，其余 $N-K$ 个专家既没被算、也没被更新，却仍要"学会路由"。易错点：别把"训练时稀疏"当成可选优化——它是 MoE 效率成立的前提，也因此才是 "the key thing that makes MOEs hard"；路由里的 top-k 是离散、不可微的，和前面 DSA 的 "frankly scary-looking top K non-differentiable object" 是同一类麻烦，讲师在 [29:15] 已预告 "maybe you shouldn't be so scared of top K selection"。与前后关系：本段在 [43:05] 处被截断（"Right? So, it's..."），负载均衡辅助损失等解法要到后面才展开；这一句伏笔把整个 MoE 板块从"为什么好"翻到"为什么难"。

## 路由粒度、并行上限与 MoE 为何普及缓慢 [43:10 - 47:23]

本段从上一段留下的悬念切入：讲师说路由问题「kind of got this RL bandit flavor to the problem」，也就是带点强化学习里 bandit（多臂老虎机）的味道，但立刻否定用 RL 或 bandit 来解——「we're not going to solve it with either RL or bandit」，而是要靠「the power of heuristics and deep learning magic」（启发式的力量加上深度学习的魔法）。这句话几乎就是本段后半部分的提纲：一直到 64:45 为止，真正落地的方法既不是 RL 也不是严谨的探索策略，而是一组启发式（heuristics）损失。紧接着是现场问答。第一位同学问「切换专家的粒度」即 granularity of switching between experts 是什么，讲师明确回答是 **token level（词元级）**：「every token gets an expert」，路由的尺度就是单个 token，而不是整句话、整个文档或整个 batch。为了让 mental model（心智模型）正确，他强调 routers（路由器）「super naive」——只是输入与某个权重之间的一次矩阵乘（a single matrix multiply）。写成式子，就是每个 token 的隐藏向量 $u$ 与每个专家对应的方向向量做内积：

$$s_i = u^\top w_i,\qquad i=1,\ldots,N$$

这里 $N$ 是专家数，$s_i$ 是第 $i$ 个专家的得分。讲师特意泼冷水：不要把它想象成语义分工，「You don't know it's like a medical question or not」，路由器不会判断「这是不是医学问题」，更可能只是「这个 token 看起来像日语，就把它送到 expert seven（专家七）」。这是本讲反复出现的易错点：MoE 里的 expert 不是人类可解释的领域专家，而是由一次线性打分切出来的参数块；所谓「专家七」只是一个编号。

第二个问答关于并行上限。同学问「is there any upper limit to parallelizing these MOE models in training or inference?」，答案是有：随着把模型 shard（分片）到越来越多设备上，「the communication cost explodes」，通信代价会爆炸式增长。这与前面讲过的专家并行（expert parallel）是同一枚硬币的两面：专家天然是可切分的块，但把激活搬到对应设备需要跨设备通信，设备越多越贵。讲师顺势预告作业（assignment）：要么让你「figure out the networking topology and things like this that will make sharding make sense」，即设计网络拓扑让分片有意义；要么反过来，「Given a networking topology, you'll have to figure out how to shard models」，在给定拓扑下决定怎么切模型。这说明 MoE 的设计从来不只是深度学习问题，还受系统约束支配；这一点后面还会以「共享专家拿不到并行收益」「设备级均衡损失」的形式回来。

44:30 起话题转向 DeepSeek。讲师说虽然它现在以 DeepSeek R1 闻名，但这门课最初的版本「was a, you know, let's just go through a DeepSeek paper lecture」，因为他认为 DeepSeek 长期在做扎实的 architecture and LLM science（架构与语言模型科学）工作，是较早系统论证「why MOEs are good」的团队之一，早期 DeepSeek MoE 论文里有大量 ablation（消融）：「what happens if we use a dense layer? What if we use hash routing? What if we use a switch router?」——稠密层、哈希路由、switch 路由的对照，他建议对架构设计感兴趣的人去读这些早期论文。进展并未停止：DeepSeek V3、V3.2 以及 GLM 都做出了「wonderful very strong MOE models」，因此「It's quite clear that this is kind of where the future of these big models are going to be for the foreseeable future」，至少未来几年（at least the next couple years）大模型都会是 MoE。45:11 的幻灯片（标题 "Larger MoE results – DeepSeek v3"）给出了直观证据：六个基准 MMLU-Pro (EM)、GPQA-Diamond (Pass@1)、MATH 500 (EM)、AIME 2024 (Pass@1)、Codeforces (Percentile)、SWE-bench Verified (Resolved)，纵轴为 Accuracy / Percentile (%)，对比 DeepSeek-V3、DeepSeek-V2.5、Qwen2.5-72B-Inst、Llama-3.1-405B-Inst、GPT-4o-0513、Claude-3.5-Sonnet-1022 六个模型；DeepSeek-V3 的六根斜纹柱分别标着 75.9、59.1、90.2、39.2、51.6、42.0，其中 MATH 500 上的 90.2 与 AIME 2024 上的 39.2 明显领先同组其它模型。

既然能省算力，为什么 MoE 直到 2024 之后才真正流行？讲师给的时间线是：Google 早在 2022 年就在研究和力推 MoE，但「it's only in like 2024 onwards that MOEs really started to catch on」；做 LLM 研究的人大多仍在用 dense（稠密）模型。46:32 的幻灯片把原因拆成两条。

![MoE 普及缓慢的两大原因：多节点基础设施复杂，以及训练目标偏启发式且不稳定](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0034_00-46-32.jpg)

第一条是 "Infrastructure is complex / advantages on multi node"，框内那段 [Fedus et al 2022] 的原文解释得很具体：稀疏带来的额外参数需要有「many accelerators (e.g. GPU/TPU) to host all the additional parameters」；通常模型用 data-parallelism（数据并行）训练，不同机器拿到训练/推理数据的不同切片，而这些原本只处理数据切片的机器「can now be used to host many more model parameters」，于是稀疏模型在「training with data parallelism and/or have high throughput serving」时才划算。换句话说，MoE 的优势是条件性的：机器多、吞吐高才成立；单机场景下参数量大反而「hard to fit them on a single device」，专家也「hard to parallelize... in ways that are like efficient in utilization」。第二条是 "Training objectives are somewhat heuristic (and sometimes unstable)"，框里嵌了一张 [Zoph et al 2022] 的图，图注写着「Sparse models often suffer from training instabilities (Figure 1) worse than those observed in standard densely-activated Transformers.」——左图的 Training Loss 在约 15000 Step 处突然竖直飙升，右图则是平滑下降的对照，直观展示「the MOEs can really blow up on you」。讲师补了一句「there's good rules of thumb now, but it's not easy」：现在有经验规则了，但 MoE 仍不是好对付的对象。

46:46 起他用一页带过另一种做法：也有人把 mixture of experts 用在 transformer 的 attention block（注意力块）上，把 attention heads 拆成专家，但「much less common」，而且「haven't been quite as easy to tame or to get to work compared to just replacing the FFN or the MLP layer」。因此本讲只讲替换 FFN/MLP 这一侧——「This is basically what all the big models do」，主流大模型都不会去切分注意力头。到这里（47:23）「the broad industrial overview」结束。易错点提醒：把「MoE」默认理解为「把 FFN 换成稀疏专家」是安全的，把注意力头也专家化属于少数派探索且更难训稳；另外「router 是 token 级、一次矩阵乘」这条结论会在后面被写成完整方程，届时不要误以为门控里藏着复杂网络。

## 设计空间三轴与 token choice top-k 的胜出 [47:23 - 49:50]

从「how you design an MOE, how you think about an MOE」开始，讲师给出三条 axes of variation（变化轴）。47:53 的幻灯片标题 "MoE – what varies?"，三个蓝色菱形 bullet 依次是 "Routing function"（路由函数）、"Expert sizes"（专家规模）、"Training objectives"（训练目标）。口述与幻灯片有细微差别：口述第三条只说 "training"，并解释「once you start to realize how MOEs work, you realize, wow, it is not easy to train something that routes tokens to different places」；幻灯片写的是 "Training objectives"，把「难」具体落到目标函数上——这与第 6 章要加的辅助损失正好呼应。第一条轴是路由函数，因为 MoE 的核心动作就是「a router has to send tokens to experts」；第二条轴是专家大小，在给定预算下「you might have more experts with fewer parameters or fewer experts with more parameters」，即专家数与单专家参数量的权衡；第三条轴是训练，也是本段后半（58:26 起）的重头戏。这三条轴构成了后续几章的骨架：先讲路由函数，再讲专家规模（细粒度化与共享专家），最后讲训练目标。

48:15 讲师先立一条通用前提：「In all cases, we don't want to activate all the experts」，任何设计都只激活一部分专家，因此「we are going to always be choosing some top K」。把这件事写成式子，就是先对 $N$ 个专家打分 $s_j$，再取前 $K$ 个（$K\ll N$）组成被激活集合，输出是各专家 FFN 的加权和再加残差：

$$\mathcal{S}_k(u)=\mathrm{TopK}\big(\{s_1(u),\ldots,s_N(u)\},K\big),\qquad K\ll N$$

$$h(u)=u+\sum_{i\in\mathcal{S}_k(u)} g_i(u)\,\mathrm{FFN}_i(u)$$

这里的 $u$ 是输入兼残差项，$g_i$ 是门控（gate）权重。这个「打分 → TopK → 门控加权 → 加残差」的模式极其重要：后面会指出它与本讲前面讲过的 DSA（DeepSeek Sparse Attention）以及 H-Nets 等论文里的结构「looks a lot like DSA」，是值得一眼认出来的通用 pattern。48:32 给出三种指派方式：token choose the expert（token 选专家，每个 token 挑自己最喜欢的 $K$ 个专家）；expert choose the token（专家选 token，「each expert picks their favorite tokens」）；以及「somehow have this very complicated router that like globally decides」，由路由器全局决定每个 token 该去哪个专家，试图全局最优地优化指派。讲师随即给出结论：「almost all the MOEs do token choice top K」，即由 token 当选择方，$K$ 个专家被选中进入 pool。这条结论是工业事实而非数学必然，也是本章最需要记住的判断。

49:09 起引用 OLMo 论文里 token choice 与 expert choice 的对照，49:14 的幻灯片副标题写着「Almost all the MoEs do a standard 'token choice topk' routing. Some recent ablations」，下面并排四张曲线图：Training loss、Validation loss (C4)、HellaSwag、MMLU Var，横轴统一是 Tokens (B)，刻度 10、50、100、150、200（即训练 token 数，单位十亿），纵轴标 Performance；图例 TC（洋红，token choice）与 EC（青，expert choice）。

![OLMo 的路由类型消融：TC 与 EC 在训练损失、C4 验证损失、HellaSwag、MMLU Var 上的四条曲线](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0036_00-49-14.jpg)

在 Validation loss (C4) 上 TC 曲线稳定低于 EC（约 2.8 对 3.0 附近）；HellaSwag 上 TC 升到约 58–59、EC 约 54；MMLU Var 上两条曲线在 26–32 区间剧烈抖动、互相穿插，差距不明显；Training loss 图里两条线都降到 2.3–2.6 附近，但横轴约 170B 处有一条竖直的洋红细线，提示某次运行在此处出现中断或发散——这与上一章 [Zoph et al 2022] 提到的训练不稳定相呼应。讲师的解读是 token choice「lower validation losses, higher downstream benchmark scores」，而且「has generally been much easier to get working」，是当今所有主流模型的标准；expert choice「also trains fine」「this also trains fine」，并非不能用，只是更难调通。

别把「TC 更好」读成理论上的绝对优势：讲师明确说 expert choice 也能训，只是 token choice 更容易调；他还顺口提到「I think one of the llama four models was maybe expert choice, but I don't think that's necessarily a strong vote of confidence」，并补了一句「I think that was one of the unreleased llama four models」——一个未发布的 Llama 4 变体算不上强有力的背书。MMLU Var 那种高噪声曲线也提醒我们，消融里「看似领先」未必稳健。另外，本章只确定了「由谁选」和「选几个」，还没回答「怎么打分」；这正是 49:50 起要展开的路由函数全景，而「$K$ 到底取多少」会一直延续到 57:20 那张按模型列出 K 值的表格。

## 路由函数全景：内积、哈希、RL 与线性指派 [49:50 - 52:58]

49:50 起「there's many different ways that you can do routing」，第一种也是绝对主流：一个很小的线性投影，「a small sort of ML like a linear projection」。讲师的描述是「My router is an inner product. Each expert has sort of a vector direction and the sort of the closest in inner product space will get selected」——每个专家对应一个方向向量 $e_i$，输入 $u$ 与之内积最大者入选。写成式子：

$$s_i=\langle u,e_i\rangle = u^\top e_i,\qquad i=1,\ldots,N;\qquad \mathcal{S}(u)=\mathrm{TopK}(s,K)$$

「This is by far the most common router」。50:35 的幻灯片 "Common routing variants in detail" 把这套画成了 "Top-2 Routing" 示意图：两个输入 token "The" 与 "Dog" 各自经过 Router 的柱状打分，只有两个 FFN 被选中（左路标出 p = 0.65 与 p = 0.3，右路标出 p = 0.8 与 p = 0.15），输出经 "Add + Normalize" 回到 "Output Hidden"。右侧文字写着 "Used in most MoEs"，并逐条列出各模型的 $K$：Switch Transformer (k=1)、Gshard (k=2)、Grok (2)、Mixtral (2)、Qwen (4)、DBRX (4)、DeepSeek (7)。

![常见路由变体详解：Top-2 内积路由图与各模型 K 值清单，以及哈希路由基线](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0037_00-50-35.jpg)

口述里点名的使用者是「the classic switch transformer, the GShard, those two are the early Google papers」以及 Grok、Mistral、DBRX、DeepSeek 等，并强调「the number of K that you choose varies depending on, you know, which paper you're looking at」，但「for the most part, you're just taking inner products」。注意这里的 $K$ 从 1 到 7 不等，是后面 57:20 表格里 "Active" 列的伏笔。

50:35 讲师提到一件「has always been a little mysterious to me」的事：很多论文证明「you don't need to do any sort of like learned routing」，直接「take your X's, you can hash them, and send them to different feedforward networks」也行，而且「this sometimes or this gives gains. Not as much as top K」，收益不如 top-k 但确实存在。同一张幻灯片的下半部分 "Hash Routing" 图里没有 Router 柱状打分，取而代之的是两个 "Hash Function" 方块，把 "The"/"Dog" 直接映射到各自的 FFN 1–4，右侧标注 "Common baseline"，页脚 [Fedus et al 2022]。讲师补充它「Not used in deployment for for real reasons for many reasons」——只作论文基线，不进生产。易错点：哈希路由有收益这一点常被误读成「路由不重要」，但收益明显小于可学习的内积 top-k，且无法按输入内容自适应，部署上不可行。

51:04 起第三种：把内积路由器当成 policy（策略），用 reinforcement learning 学路由。讲师替「classic learning theory type oriented person」把逻辑说透：「It's kind of like a bandit problem. I'm selecting one out of K or K out of N. And I don't observe all of them. This is a bandit problem. I will use a bandit or an RL algorithm」——只观测到被选中专家的反馈，看不到反事实，正是多臂老虎机结构。51:56 的幻灯片 "Other routing methods" 上半部分画的就是 "Reinforcement Learning"：Router 旁标注类似「Loss = α · r + β」的奖励项，右侧文字 "Used in some of the earliest work Bengio 2013, not common now"。口述也点名 Bengio 2013 是最早的工作之一。但它不流行，原因是「the approaches needed to you do RL introduce a lot of overhead in terms of both the RL algorithm and the stochasticity」——RL 算法本身的开销与随机性都太重。更关键的是「what people have figured out is there's a good set of heuristics like these recipes that you can apply to this very simple classic top K routing scheme that basically just has it working」，既然简单 top-k 加一组配方就能工作，就没理由上复杂方案。这句话把 43:10 的「heuristics and deep learning magic」和 59:45 的「number three」串成了一条线。

52:19 讲师讲了一个他个人很喜欢、「make sense」的思路：解一个 linear assignment problem（线性指派问题）。做法是「globally compute kind of the score」——先算出所有 (token, expert) 两两配对的得分，再「use linear assignment to exactly solve for the optimal assignment from experts to tokens」，精确求出全局最优指派。51:56 那张幻灯片的下半部分对应 "Solve a matching problem"：图中 "BASE Routing" 中央是绿色 "Solve Linear Assignment" 模块，右侧文字 "Linear assignment for routing. Used in various papers like Clark '22"，页脚同样是 [Fedus et al 2022]。口述评价是「It's been used. It's been shown to do good things in some cases」，但「Hasn't been seen at scale at all」，原因是「extremely expensive relative to the others to get working」。局限很明确：全局最优在理论上漂亮，代价却是相对其它方案高得多的计算量，因此从未在规模上被采用。

四条路线（内积 top-k、哈希、RL/bandit、线性指派）里，只有第一条成为共识；哈希是基线，RL 与线性指派分别卡在随机性开销与计算开销上。还有一个容易被忽略的细节：这两张路由分类幻灯片都标 [Fedus et al 2022]，说明它们是同一篇综述性工作里的分类图，而不是某个模型的内部实现。52:58 起，讲师会把「共识机制」top-k 路由写成完整方程，并指出它与 DSA 的结构相似性；本章反复出现的「$K$ 个专家」「$N$ 个专家」届时会被写成 $s_{i,t}=\mathrm{Softmax}_i(u_t^{l\top}e_i^l)$ 与 $g_{i,t}$ 的分段定义。

## top-k 路由的方程形式、共享专家与细粒度分割 [52:58 - 58:26]

52:58「top K routing, as I said, is basically the consensus routing mechanism」，讲师说它「very very simple」，而且「if you were paying attention in the DSA slide, this looks a lot like DSA」，读 H-Nets 等论文也会看到同样结构，「this is a good kind of pattern to be able to recognize」。53:17 的幻灯片 "Top-K routing in detail." 副标题「Most papers do the old and classic top-k routing. How does this work?」给出了方程形式：

$$\mathbf{h}_t^l=\sum_{i=1}^{N}\left(g_{i,t}\,\mathrm{FFN}_i\left(\mathbf{u}_t^l\right)\right)+\mathbf{u}_t^l$$

$$g_{i,t}=\begin{cases}s_{i,t}, & s_{i,t}\in\mathrm{Topk}(\{s_{j,t}\mid 1\le j\le N\},K),\\ 0,& \text{otherwise},\end{cases}$$

$$s_{i,t}=\mathrm{Softmax}_i\left(\mathbf{u}_t^{l\,T}\mathbf{e}_i^l\right),$$

其中 $\mathbf{u}_t^l$ 是第 $l$ 层、第 $t$ 个 token 的输入兼残差项（residual term），$\mathbf{e}_i^l$ 是第 $i$ 个专家的权重向量，$N$ 为专家总数。

![Top-K 路由的完整方程形式：门控分段定义、softmax 打分与残差连接](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0039_00-53-17.jpg)

图中蓝色箭头把 $g_{i,t}$ 标为 "Gating"，把 $s_{i,t}$ 那行标为 "Gates selected by a logistic regressor"——门控就是「每个专家一个权重向量与输入做内积」，「Very very lightweight」，没有任何复杂结构。右侧还写了一条容易漏掉的实现差异：「This is the DeepSeek (V1-2) router (Grok, Qwen do this too)」而「Mixtral, DBRX, DeepSeek v3 softmaxes after the TopK」——DeepSeek V1/V2、Grok、Qwen 先 softmax 再取 TopK，Mixtral、DBRX、DeepSeek v3 则先取 TopK 再 softmax；同一族 top-k 路由，softmax 的位置并不统一。页脚引用 [Dai et al 2024]。

54:19 起进入 DeepSeek 的创新：「even within this simple thing, there's been some innovation and variation from DeepSeek and friends. DeepSeek MoE pioneered this idea, which is now very widely used, called shared experts」。经典 top-k 里有 $N$ 个专家、每次从中选 $K$ 个；但「there's some processing some very common processing that I want to apply to all tokens」，不希望所有专家每次都不同，而是「some experts to always be applied. And then some other experts to be applied conditionally」。DeepSeek MoE 的做法是两步：「They cut up the experts into smaller finer-grained chunks」把专家切得更小更细，「some subset of those fine-grained experts were set as shared experts that are always on」，这些专家「bypass the router」，始终处理输入并加回输出。动机是经典设计里「you were just kind of reusing a lot of these weights to do common modeling」，大量权重在重复做公共建模，不如把这部分卸载到共享专家，让其余专家「specialize even more」。

![从常规 Top-2 路由到细粒度专家分割，再到共享专家隔离的三阶段架构演进](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0040_00-54-38.jpg)

54:38 的幻灯片 "Recent variations from DeepSeek and other Chinese LMs" 用三幅图把演进画得很清楚：(a) Conventional Top-2 Routing，$K=2$，专家 1、2、…、$N$；(b) + Fine-grained Expert Segmentation，$K=4$，专家变成 1、2、3、4、…、$2N\!-\!1$、$2N$；(c) + Shared Expert Isolation (DeepSeekMoE)，$K=3$，最左侧专家 1 被涂成绿色的 "Shared Expert"，其余 2、3、4、…、$2N$ 仍是 "Routed Expert"。图例区分淡紫的 Routed Expert 与绿色的 Shared Expert，标题下写着「Smaller, larger number of experts + a few shared experts that are always on.」，右下角注明「(Used in DeepSeek / Qwen, originally from DeepSpeed MoE)」。

55:41 起讲师称赞 DeepSeek「do careful ablations」，并展示 55:59 的幻灯片 "Various ablations from the DeepSeek paper" 的柱状图：纵轴 Normalized Performance（0.5–1.2），横轴六个 Metrics——HellaSwag、PIQA、ARC-easy、ARC-challenge、TriviaQA、NaturalQuestions；四条图例分别是「0 shared expert + 2 out of 16 routed experts (GShard)」（蓝）、「1 shared expert + 1 out of 15 routed experts (+ shared expert isolation)」（橙）、「1 shared expert + 3 out of 31 routed experts (+ fine-grained expert segmentation)」（绿）、「1 shared expert + 7 out of 63 routed experts (+ finer expert segmentation)」（红）。在 TriviaQA 上蓝柱约 0.62、橙约 0.85、绿约 0.94、红达 1.0；NaturalQuestions 上蓝约 0.56、橙约 0.79、绿约 0.88、红约 1.0——讲师口中的「This blue to yellow gap, right, is shared experts or not」正指这里，TriviaQA、NaturalQuestions 这类任务上共享专家提升尤其显著。图 3 的图注强调所有对比模型「have the same number of parameters and activated parameters」，性能按最佳值归一化以便展示，结论是「fine-grained expert segmentation and shared expert isolation both contribute to stronger overall performance」；幻灯片底部大字总结「More experts, shared experts all seem to generally help」。口述里也复述了老 GShard 设计是「zero shared experts, and then you have these big experts, 16 routed experts」。易错点：这些数值是归一化后的相对性能，不是绝对分数，且刻意控制了总参数量与激活参数量相同，不能拿来和其它论文里的绝对分数直接比较。

56:24 起讲师补上反方证据：OLMo（口误说成 "Elmo"）是「the nice western sort of carefully controlled MoE study」，也做了「do shared experts help?」这类消融，结论是「shared experts don't help very much」，但「doing fine-grained many experts is helpful」——两家在共享专家上「disagree a little bit」，在细粒度多专家上一致。随后给出产业谱系：最早三个 MoE 都来自 Google，Mistral、DBRX、Grok 是西方早期尝试，DeepSeek V1 同时提出细粒度与共享专家设计，「and then that sort of catches on and everyone else sort of follows from DeepSeek」，地位类似稠密 Transformer 里的 Llama 设计；DeepSeek MoE 与 DeepSeek V3 成了被复制的标准模板，Qwen 3.5、GLM 等现代模型里「both shared experts and fine-grained experts」都在广泛使用，是「very battle-tested」的设计。57:20 的表格 "Expert routing setups for recent MoEs" 把这套谱系量化了，列为 Model、Routed、Active、Shared、Fine-grained ratio：GShard 2048 / 2 / 0 / —；Switch Transformer 64 / 1 / 0 / —；ST-MOE 64 / 2 / 0 / —；Mixtral 8 / 2 / 0 / —；DBRX 16 / 4 / 0 / —；Grok 8 / 2 / 0 / —；DeepSeek v1 64 / 6 / 2 / 1/4；Qwen 1.5 60 / 4 / 4 / 1/8；DeepSeek v3 256 / 8 / 1 / 1/14；OLMoE 64 / 8 / 0 / 1/8；MiniMax 32 / 2 / 0 / ~1/4；Llama 4 (maverick) 128 / 1 / 1 / 1/2。可以看到早期模型 Shared 列基本为 0，而从 DeepSeek v1 开始 Shared 列非零、Fine-grained ratio 出现 1/4 到 1/14 这样的分数：比率越小，专家被切得越碎、数量越多。

58:00 的问答把共享专家拉回系统层面。同学问「if there is a shared expert uh or there are like few shared experts, how would parallelization work in that case?」讲师回答得很直接：「in the shared expert case, you wouldn't get any parallelization savings. Like every activation has to route through those」——每个激活都要过共享专家，因此拿不到专家并行带来的节省；可行的折中是「you can copy the shared expert um to basically eat memory savings to reduce comms cost」，即复制若干份共享专家，用显存换通信。这条局限把路由函数这一轴与系统约束重新连起来：共享专家是「always on」的稠密路径，天然不稀疏。至此设计空间的前两条轴讲完，58:26 起转入最难的第三条轴——训练目标。

## 训练 MoE 的根本难题与随机近似 [58:26 - 62:25]

58:26「that's kind of the sort of design space of routers. Now I want to talk about training MoEs」，讲师预告这部分「gets into really surprising and I would say very deep learning things」，并自述初学时的判断：「I thought there's no way that we can train these things well or like reasonably」，结果「there are lots of, you know, tricks that when combined together like just really work well um and robustly for some reason」——注意「for some reason」这个措辞，坦承这套配方并没有干净的理论解释。58:41 的幻灯片 "How do we train MoEs?" 把矛盾写成两行加粗字：「Major challenge: we need sparsity for training-time efficiency… But sparse gating decisions are not differentiable!」口述展开为两点：为了训练效率必须稀疏，否则「we're going to pay the full flops cost of all of our experts. And we have a lot of experts」，等于付全部专家的 FLOPs；但一旦稀疏，「gating decisions are no longer differentiable」，而且「we also don't see all the different counterfactual experts we could have picked」——未被选中的专家既没有梯度、也看不到反事实结果。这两点正是 43:10 所说「RL bandit flavor」的根源。

幻灯片的 "Solutions?" 下列了三条：1. Reinforcement learning to optimize gating policies；2. Stochastic perturbations；3. Heuristic 'balancing' losses. 末尾还留了一句互动式的「Guess which one people use in practice?」讲师的口述答案在 59:45：「It's number three, right? It's a collection of really interesting heuristics that end up working」，并预告「I'll talk about uh the other two first... they're not widely used... but I think it's useful to know」。这里的结构很关键：59:55–63:07 讲的 RL 与随机扰动都是「陪衬」，真正落地的启发式均衡损失从 63:13 才开始。易错点：不要把「RL 是理论上最自然的解法」误当成「实践中使用的解法」；本章与下一章的内容恰好是「被否定的两条」和「被采用的一条」。

60:02 的幻灯片 "RL for MoEs" 标题下写着「RL via REINFORCE does work, but not so much better that it's a clear win」，下面四张散点图分别为 S-BASE、RL-R、Hash、Comparisons，横轴 Expert Count 取 1、2、4、8、32、128、512，纵轴 Validation Loss 约 2.0–3.2，图上按模型规模标了 15M、25M、55M、130M、370M、1.3B 等标签，图注「(REINFORCE baseline approach, Clark et al 2020)」；底部结论「RL is the 'right solution' but gradient variances and complexity means it's not widely used」。口述对应 60:02–60:32：讲师说「This is the green line over here. Um that's one of the baseline approaches. It works fine, but because of the gradient variance and complexity, it's not, you know, actually the optimal thing to do」，甚至「even sort of some of the approaches that are proposed in this paper, you know, handily beat using reinforce gradients for routing decisions」——Clark 2020 这篇论文自己提出的方法就明显胜过用 REINFORCE 梯度学路由。也就是说，梯度方差（gradient variance）与实现复杂度这两项开销，让理论上「正确」的 RL 在实测里赢不了。

60:32 起讲师转向「very close to the idea of RL, um but sort of starts to move more towards this like heuristic world of like, you know, uh write down objectives and try things」——即 stochastic approximation（随机近似），来自 Noam Shazeer 等人最早的 MoE 论文。核心是训练时「you don't really make a hard decision during training. What you do is you make stochastic decisions」：仍保留标准路由内积 $H$，但注入「some noise that's kind of, you know, dependent on the scale of the input」，再照常做 TopK 与 softmax。61:22 的幻灯片 "Stochastic approximations" 给出完整定义：

$$G(x)=\mathrm{Softmax}\big(\mathrm{KeepTopK}(H(x),k)\big)$$

$$H(x)_i=(x\cdot W_g)_i+\mathrm{StandardNormal}()\cdot\mathrm{Softplus}\big((x\cdot W_{noise})_i\big)$$

$$\mathrm{KeepTopK}(v,k)_i=\begin{cases}v_i,& \text{if }v_i\text{ is in the top }k\text{ elements of }v,\\ -\infty,&\text{otherwise.}\end{cases}$$

其中 $W_g$ 是路由门控权重，$W_{noise}$ 产生与输入相关的噪声尺度，$\mathrm{StandardNormal}()$ 采样标准高斯噪声，$\mathrm{Softplus}$ 保证尺度非负；$\mathrm{KeepTopK}$ 把非 top-k 元素置为 $-\infty$，这样后面的 softmax 会把它们的概率压成 0。

![Shazeer 2017 的随机近似：带高斯扰动的路由打分与 KeepTopK 定义](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0045_01-01-22.jpg)

幻灯片注明「From Shazeer et al 2017 – routing decisions are stochastic with gaussian perturbations.」，并列出两条结论：1.「This naturally leads to experts that are a bit more robust.」2.「The softmax means that the model learns how to rank K experts」。口述补充了直觉：若两个专家得分「closely tied」，随机扰动会让你这次选其中一个、下次选另一个，起到 tiebreak（打破平局）与 explore（探索）的作用；反向传播时「the experts that are helpful will get high weights」，有用的专家被逐步强化、权重越来越高；softmax 则让模型学会给这 $K$ 个专家「rank」而不是硬选，等价于按概率加权（「you're kind of waiting by these elements」）。局限也在这里：它只是「one reasonable thing that you can do」，属于探索与稳健化技巧，讲师明确说它「not widely used」，而且下一章会给出它被移除的证据。

本章把前面 top-k 的方程往前推了一步：53:17 那页里 $g_{i,t}$ 是确定性的分段门控，61:22 这页里 $G(x)$ 变成了带高斯扰动的随机门控，二者差别只在 $H(x)$ 是否加噪声、以及 $\mathrm{KeepTopK}$ 后用 softmax 排序。而 51:15 的「I'm selecting one out of K or K out of N」、59:31 的「if you're familiar with like, you know, explore-exploit bandit style stuff, you could do some of that」在这张幻灯片上落成了具体公式。62:26 会讲 Fedus et al. 2022 的另一种乘性扰动（uniform multiplicative perturbation），并说明它被 Zoph et al. 2022 移除；随后 63:13 才进入真正被采用的第三条路线——启发式均衡损失。

## 随机抖动的退场与启发式均衡损失 [62:25 - 64:45]

62:26「there's another sort of scheme that's very closely related that's done in Fit Fedus et al. in 2022, which does kind of a uniform multiplicative perturbation」，目的是「hardening the experts, like making them robust」。62:43 的幻灯片（标题同为 "Stochastic approximations"）直接贴出了实现代码：

```python
if is_training:
    # Add noise for exploration across experts.
    router_logits += mtf.random_uniform(shape=router_logits.shape, minval=1-eps, maxval=1+eps)
# Convert input to softmax operation from bfloat16 to float32 for stability.
router_logits = mtf.to_float32(router_logits)
# Probabilities for each token of what expert it should be sent to.
router_probs = mtf.softmax(router_logits, axis=-1)
```

第一个注释「Add noise for exploration across experts」点明动机；乘性体现在 `mtf.random_uniform(..., minval=1-eps, maxval=1+eps)` 是对 `router_logits` 做 $1\pm\varepsilon$ 附近的均匀乘性扰动，而不是 Shazeer 2017 那种加性高斯噪声；后两行还顺带暴露了一个工程细节：softmax 前先把 `router_logits` 从 `bfloat16` 转成 `float32`「for stability」，再在 `axis=-1` 上算 `router_probs`。幻灯片正文写着「Stochastic jitter in Fedus et al 2022. This does a uniform multiplicative perturbation for the same goal of getting less brittle experts. This was later removed in Zoph et al 2022」。

![Fedus 2022 的随机抖动实现代码，以及稳定性与质量对照小表](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0046_01-02-43.jpg)

代码下方的小表给出对照，三列为 Method、Fraction Stable、Quality (↑)：Baseline 为 4/6 与 $-1.755\pm0.02$；Input jitter ($10^{-2}$) 为 3/3 与 $-1.777\pm0.03$；Dropout (0.1) 为 3/3 与 $-1.822\pm0.11$。也就是说，在「稳定次数占比」上 Baseline 只有 4/6，加输入抖动或 dropout 能到 3/3；但在标注「↑」（越大越好）的 Quality 一列，Baseline 的 $-1.755$ 反而优于 Input jitter 的 $-1.777$ 与 Dropout 的 $-1.822$。讲师口述的结论更干脆：「this was later removed in later Google papers that do sort of routing. And it's not really clear that this is, you know, necessary at all」，并说后续消融显示「not doing any any of these like stochastic sort of robustness tricks, um actually helps um with uh both stability and overall quality of the final trained MOE」。这里存在不同实验设置下的张力，讲师自己也用「not really clear」「you don't need necessarily these stochastic exploration terms」留了余地，因此不宜把任何一行数值绝对化。可以确定的收紧结论是：Shazeer 2017 的加性高斯扰动与 Fedus 2022 的均匀乘性抖动都属于「让专家不那么脆」的探索技巧，而后续 Google 工作把它们移除了，理由是稳定性与最终质量都不需要它们。

63:13「So, this takes us to what you actually do, um which is a series of sort of heuristics that, you know, in the end is going to give us a working uh MOE. And a lot of these have to do with essentially balancing」。63:24 起讲师描述如果「just do normal gradient descent ignoring any of these like exploration exploitation concerns」会发生什么：先路由 top K 个专家，然后「the strongest of those experts will get sort of more signal」，反向传播会判定「that expert was good, you know, increase the weight of that expert」，于是「rich gets richer」——被选中的专家权重越来越强，权重强又意味着更常被选中，最终「they kind of run away taking on everything」。这就是 expert collapse（专家坍缩）或 expert starvation（专家饥饿）：「is a very very real problem」，大量参数白费，也正是「the core issue that you have to solve with sort of heuristic training uh of MOEs」。解法是「add a sort of heuristic loss um to the total modeling loss」，用来「balance out the tokens that go to different experts」。注意这与前两条被否定的路线形成完整闭环：不靠 RL 的策略梯度，也不靠随机探索，而是直接往目标函数里加一项启发式惩罚。

64:25 起讲师开始讲「one approach, which is done in the Switch Transformer over here」，并说明设定：「you have different experts going from uh one to N, and you have a a batch of tokens」，目标是「make sure that these two inner products are kind of small」；到 64:45 本段截断处，他刚说到「F is the fraction of tokens that are dispatched to any single expert」。64:04 的幻灯片 "Heuristic balancing losses" 已经把完整定义摆了出来，副标题「Another key issue – systems efficiency requires that we use experts evenly..」先点明动机是系统效率（专家要被均匀使用，否则某些设备空转）。

![Switch Transformer 的启发式均衡损失：f_i、P_i 定义与「越常用越强下调」的梯度解释](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0047_01-04-04.jpg)

框内 [Fedus et al 2022] 的原文是：「For each Switch layer, this auxiliary loss is added to the total model loss during training. Given N experts indexed by i = 1 to N and a batch $\mathcal{B}$ with T tokens, the auxiliary loss is computed as the scaled dot-product between vectors $f$ and $P$」，编号为式 (4)：

$$\mathrm{loss}=\alpha\cdot N\cdot\sum_{i=1}^{N} f_i\cdot P_i$$

$$f_i=\frac{1}{T}\sum_{x\in\mathcal{B}}\mathbb{1}\{\mathrm{argmax}\ p(x)=i\}$$

$$P_i=\frac{1}{T}\sum_{x\in\mathcal{B}} p_i(x).$$

式 (5) 定义 $f_i$ 为分派给专家 $i$ 的 token 比例（$\mathbb{1}\{\mathrm{argmax}\ p(x)=i\}$ 是「该 token 的 argmax 是否落在专家 $i$」的指示函数），式 (6) 定义 $P_i$ 为路由器分配给专家 $i$ 的概率质量在 batch 上的均值，$\alpha$ 是缩放系数、$N$ 是专家数、$T$ 是 batch 内 token 数。幻灯片底部给出了最关键的梯度解释：「The derivative with respect to $p_i(x)$ is $\frac{\alpha N}{T^2}\sum 1_{\mathrm{argmax}\ p(x)=i}$, so more frequent use = stronger downweighting」——对 $p_i(x)$ 求导得到与「该专家被选中次数」成正比的项，越热门的专家受到的负梯度压制越强。这解释了为什么一个看起来不易从第一原理推出的点积式子能工作：它不是在直接优化均衡，而是在给热门专家施加更强的下调压力。

几个容易踩的坑：其一，$\alpha\cdot N$ 的缩放与 $1/T^2$ 的导数系数说明损失对 batch 大小 $T$ 和专家数 $N$ 都敏感，调参时不能只看 $\alpha$；其二，$f_i$ 由不可微的 argmax 决定，因此这一项本身也带启发式色彩，不能当成严格可微的均衡约束；其三，副标题说的「systems efficiency」提醒我们，均衡损失的第一动机是让各设备上的专家都被用起来，而非纯粹为了模型质量。本段在讲师刚开始口头解释 $F$ 的地方（64:45）结束，式 (4)–(6) 与梯度结论已经完整给出，后续关于「去掉均衡损失会怎样」「DeepSeek 的设备级均衡与 aux-loss-free bias 方案」的展开属于下一段的内容。回看 43:10 那句「we're going to solve it with the power of heuristics and deep learning magic」：到这里它终于兑现——一个含 top-k 不可微选择的系统，最终靠的既不是 RL 也不是随机扰动，而是加在总损失上的这一项启发式均衡惩罚。

## 启发式均衡损失：从目标式到梯度解释 [64:45 - 66:00]

本段开头延续对 Switch Transformer 负载均衡辅助损失（load balancing auxiliary loss）的讨论，重点是把式子里两个因子的含义讲清楚。幻灯片 "Heuristic balancing losses" 的副标题直接点明动机："Another key issue – systems efficiency requires that we use experts evenly.."，也就是说均衡不是为了让模型更聪明，而是为了让专家并行的设备都能跑满。图中方框里写道："For each Switch layer, this auxiliary loss is added to the total model loss during training. Given $N$ experts indexed by $i = 1$ to $N$ and a batch $\mathcal{B}$ with $T$ tokens, the auxiliary loss is computed as the scaled dot-product between vectors $f$ and $P$"，即辅助损失是向量 $f$ 与 $P$ 的缩放点积，会在训练时加到总损失上。讲师特别区分了两者：$f_i$ 是"是否把 token 分派给某个专家"的硬计数，而 $P_i$ 是路由器分配给该专家的**总概率质量**（total probability mass），是"更软的版本"（softer version）——它不只看 argmax 的胜负，还看 softmax 之后的概率数值。

![Switch Transformer 的启发式均衡损失：式(4)(5)(6)与对 p_i(x) 的梯度解释](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0048_01-05-31.jpg)

幻灯片给出的三条式子（编号 4、5、6）可以原样写为

$$\text{loss} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i \tag{4}$$

$$f_i = \frac{1}{T}\sum_{x\in\mathcal{B}} \mathbf{1}\{\text{argmax } p(x) = i\} \tag{5}$$

$$P_i = \frac{1}{T}\sum_{x\in\mathcal{B}} p_i(x) \tag{6}$$

来源标注为 "From the Switch Transformer [Fedus et al 2022]"。这里的 $\alpha$ 是损失权重系数，$N$ 是专家数，$T$ 是 batch $\mathcal{B}$ 里的 token 数，$p_i(x)$ 是路由器对 token $x$ 在专家 $i$ 上的输出概率。$f_i$ 含 $\mathbf{1}\{\text{argmax }p(x)=i\}$ 指示函数与 argmax，本身不可微；真正把梯度送回路由器的只有 $P_i$ 这一项。这也解释了为什么这个目标"不是从第一原理推出来的"（not a thing where you would like derive it from first principles）——它是一条启发式规则，理解它最好的办法不是盯着目标函数，而是去看它的梯度。

对 $P_i$ 求偏导，得到的正是被分派到该专家的 token 比例（fraction of tokens allocated）。幻灯片底部给出了更精确的一行："The derivative with respect to $p_i(x)$ is $\frac{\alpha N}{T^2}\sum \mathbf{1}_{\text{argmax }p(x)=i}$, so more frequent use = stronger downweighting"。用讲师的话说，这是一个"梯度空间里的惩罚"（penalty in the gradient space）：某个专家拿到的 token 越多，它受到的负梯度就越强，从而把路由器分配给热门专家的概率质量按 $f_i$ 的比例往下压。换句话说，均衡损失的作用不是奖励冷门专家，而是**按受欢迎程度成比例地抑制热门专家**，这正是 $f_i P_i$ 相乘结构的效果，即 $\partial L_{\text{aux}}/\partial P_i \propto f_i$。一旦从梯度的角度去看，"equation four 里那个一开始并不显然的目标"就变得清楚了：它要做的就是 push down popular experts。

与前后内容的关系很清楚：这一节把上一章留下的"为什么偏偏是 $f_iP_i$"补成了可计算的梯度解释，而下一节会说明 DeepSeek V1/V2 几乎原封不动地复用了这套逐专家损失，只是又叠了一层按设备的版本。易错点有两个：其一，$\alpha N$ 与 $1/T$ 的缩放让损失对 batch 大小、专家数敏感，$\alpha$ 调得太大会把专家压成"完全均匀"，反而损害训练动态（这一点在 71:10 后的问答里会被再次提到）；其二，$f_i$ 的不可微性常被误认为整个损失都不可微，实际梯度只经 $P_i$ 回传，只需把不可微的路由当作"直通"处理即可，这也是后面"直接把梯度泵过系统"这一做法能成立的前提。

## DeepSeek 的逐专家与逐设备双层均衡 [66:00 - 67:51]

讲师随后用"重构"的口吻把 DeepSeek V1/V2 的 MoE 训练方案拼出来：做法其实相当朴素——正常损失照算，直接让梯度"直穿"（backprop straight through）专家，刻意忽略 top-k 选择的不可微性与探索（exploration）问题；唯一的额外动作是加上与 Switch Transformer 完全相同的逐专家均衡（per-expert balancing），也就是把 $\alpha$、$f$、$P$ 那一套搬过来。幻灯片 "Example from deepseek (v1-2)" 的第一部分标题即为 "Per-expert balancing – same as the switch transformer"，并给出编号 12、13、14 的三条式子：

$$\mathcal{L}_{\text{ExpBal}} = \alpha_1 \sum_{i=1}^{N'} f_i P_i \tag{12}$$

$$f_i = \frac{N'}{K' T}\sum_{t=1}^{T}\mathbf{1}(\text{Token } t \text{ selects Expert } i) \tag{13}$$

$$P_i = \frac{1}{T}\sum_{t=1}^{T} s_{i,t} \tag{14}$$

与式(4)(5)对照，DeepSeek 用了 $N'$ 个细粒度专家、每个 token 选 $K'$ 个，因此 $f_i$ 的归一化系数变成 $\frac{N'}{K'T}$ 而不是 $\frac{1}{T}$：这是为了让均匀分派时 $f_i$ 的期望落在 $K'/N'$ 附近，$\alpha_1$ 是这一项的权重，$s_{i,t}$ 是 token $t$ 对专家 $i$ 的路由得分。式(14)把 Switch 里的 $p_i(x)$ 换成了 $s_{i,t}$，本质上仍是"总概率质量"的 batch 平均。

真正体现 DeepSeek 系统工程取向的是第二层。讲师说他们"不只均衡专家，还要按设备均衡"：假设一台机器放 4 个专家、另一台机器放另外 4 个专家，光让 8 个专家各自均匀还不够，还得让两台机器都跑在满利用率（full utilization）上，否则设备之间会出现空转与等待。幻灯片第二部分 "Per-device balancing – the objective above, but aggregated by device." 给出编号 15、16、17：

$$\mathcal{L}_{\text{DevBal}} = \alpha_2 \sum_{i=1}^{D} f_i' P_i' \tag{15}$$

$$f_i' = \frac{1}{|\mathcal{E}_i|}\sum_{j\in\mathcal{E}_i} f_j \tag{16}$$

$$P_i' = \sum_{j\in\mathcal{E}_i} P_j \tag{17}$$

其中 $D$ 是设备数，$\mathcal{E}_i$ 是第 $i$ 台设备上的专家集合，$|\mathcal{E}_i|$ 是该设备上的专家数，$f_i'$ 取设备内各专家 $f_j$ 的平均值，$P_i'$ 则把设备内各专家的 $P_j$ 直接求和，$\alpha_2$ 是这一"次要目标"的权重。它作用方式与逐专家损失完全相同，只是把操作粒度从专家换成设备，纯粹为了利用率（utilization reasons）。

![DeepSeek V1/V2 的逐专家均衡式(12)-(14)与逐设备均衡式(15)-(17)](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0049_01-07-04.jpg)

紧接着讲师把话题推到 DeepSeek V3：V3 引入了逐专家 bias 项（per-expert bias term），配合一种在线学习（online learning）技巧来均衡专家，从而"去掉一部分这些辅助损失"（getting rid of some of these auxiliary losses），也就是后来常被称作 aux-loss-free 的方向。但他马上补了一句重要的限定：到头来他们仍不得不保留一些辅助损失，以确保极端失衡（extreme imbalances）不发生；"目前还没有哪种方案能完全把这些（较丑陋的）辅助损失去掉"（there's no kind of solution that fully gets rid of them so far）。这与上一节的梯度解释形成呼应：辅助损失虽是启发式，但它承担的"抑制富者愈富"的功能目前仍无可替代。

从全段结构看，这一节是"启发式均衡"从单模型（Switch）到工业级系统（DeepSeek）的桥：V1/V2 用 $\alpha_1$、$\alpha_2$ 两级损失把算法目标与系统拓扑绑在一起，V3 则试图用 bias 项把它内生化。易错点在于别把 $f_i'$ 与 $P_i'$ 的聚合方式记混——一个是平均、一个是求和，这源于 $f_j$ 本身是比例而 $P_j$ 是概率质量；另外，$\alpha_2$ 引入的是"额外"损失而非替代 $\alpha_1$，把 $\alpha_1$ 单独调到很大去强行达到完全均匀，会像后面问答所指出的那样破坏训练动态，因此工程上宁可多挂一项设备损失。

## 去掉均衡损失的消融、训练收束与两则现场问答 [67:51 - 71:41]

负载均衡损失既然是"一堆复杂启发式"，一个自然的问题是能不能干脆不要它。讲师给出的答案是"效果相当灾难性"（pretty catastrophic），证据来自 OLMo 论文里做得很干净的消融（ablation）。幻灯片 "What happens when removing load balancing losses?" 上半部是四张并排曲线：Training loss、Load balancing loss、Validation loss (C4)、Validation loss (Pile)，横轴都是 Tokens (B)，从 1 到 10（十亿 token 量级），纵轴标 Performance，图例是粉色的 LBL（有均衡损失）与青色的 No LBL（去掉均衡损失）。图注 Figure 9 写明 "Impact of applying a load balancing loss (LBL). The training loss plot excludes the load balancing loss for both models."，也就是说训练损失曲线里已经把辅助损失本身扣掉，比较是公平的。结果是有 LBL 时粉色曲线是"很漂亮的正常曲线"（very nice normal curves），去掉后损失明显抬高、训练损失也不再好看；单独那张 Load balancing loss 图里，LBL 大致压在 0.1 附近平坦，No LBL 则从约 0.4 缓慢降到 0.3 左右；两张验证损失（C4 与 Pile）都从约 4.5 降到 3.7 附近，但 No LBL 始终更差。

![OLMo 去掉负载均衡损失的消融：Figure 9 四条损失曲线与 Figure 10 专家分派占比](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0050_01-08-36.jpg)

更能说明问题的是下半部两张专家分派图，图注 Figure 10 为 "Expert assignment during training when using or not using a load balancing loss for the first MoE layer."。纵轴是 "% of tokens in batch assigned to expert"（0 到 100），横轴 Tokens (B)，图例列出 Expert 0 到 Expert 7 共 8 个专家。左图 "No load balancing" 里几乎全部 token 涌向两个专家——黄色那一条和偏粉色那一条，最高冲到约 100%，其余专家基本贴着 0；右图 "Load balancing" 里 8 条线都稳定在相近的低位，呈现"漂亮的均匀利用"（nice even utilization）。讲师由此下的结论很直白：没有均衡损失，等于把一大堆参数扔掉了（thrown away a ton of parameters），那些专家在几乎整个训练过程中什么都没干。把这套机制写成梯度形式仍是上一节的 $\partial L_{\text{aux}}/\partial P_i = \alpha N f_i$，而路由层输出本身是 $y=\sum_{i\in\mathrm{TopK}(s)} s_i\,\mathrm{FFN}_i(x)+x$；均衡项只是给这个含不可微 top-k 的系统加了一个反向的力。

由此可以收束 MoE 训练的整体图景，这也是讲师本人觉得"挺令人惊讶"的地方：这是一个含 top-k 选择、本质上非可微、而且相当复杂的对象，可你真正要做的只是加一个均衡损失把专家抹平，其余部分就当成能正常把梯度泵过去（pump gradients through the system）来训，模型就能训得很顺（trains very nicely）。背后的动力学是两股力相互抵消：一个专家如果有用，就会被选中、获得更多梯度、变得更强、更常被选中，形成正反馈（positive reinforcement cycle）；均衡损失则把这种强化"抹平"（evening things out），两种动态恰好抵消。他还指出同一个技巧不止用于 MoE：DSA 里已经见过的 top-k 选择、以及试图去掉 tokenizer（tokenizers）的 H-Net，都靠"top-k 选择 + 负载均衡或其它辅助损失"来驯服不可微性，并预言这会成为未来架构设计的常见配料（ingredient of future architecture design）。

这一段末尾还嵌了两则现场问答，恰好补上两处理解盲区。第一问是"MoE 和系统优化怎么联系、这些专家真的是专家吗"：确有一些论文可视化了哪些 token 被哪些专家激活，但因为路由器极其简单（routers are so simple），它们并不是"聪明的专家"——不是医学专家、法律专家之类；你能看到标点（punctuations）可能被送到某个专家、其它符号（symbols）到另一个、非英语字符集（non-English language character sets）再到另一个，但绝不能指着说"这是华尔街日报专家"，一句话概括就是"没有语义"（There's no semantics）。第二问是"为什么同时要逐专家损失和逐设备损失"：讲师的解释是，逐专家损失如果被完美满足、且专家在设备间均匀切分，理论上**也会**带来逐设备均衡；但你并不想把逐专家损失的系数调到能产生真正完全均匀（true full uniformity）的程度，因为那会损害训练动态（deleterious effects on training dynamics）；而设备级均衡又足够重要，所以宁可额外挂一小项损失去鼓励它。这两问也顺带点出局限：均衡损失是经验性的、没有第一原理保证，且"专家"并不承担可解释的语义分工。

## 系统侧：并行轴、结构化稀疏矩阵乘与通信下投影 [71:41 - 76:20]

从 71:37 起讲师转向"系统侧"（the systems side），先把标准并行方式数了一遍：数据并行（data parallelism）把一个数据集拆到许多机器上，模型并行（model parallelism）把模型按层切开分到不同机器，两者还可以组合。但他强调每种并行都有上限：数据并行一旦把 batch size 吃满（maxed out your batch size）就没法再靠数据切分了；模型并行受限于模型里"天然的切分点"（natural cut points），切完也就到头了。专家并行（expert parallelism）恰好补上第三个轴。幻灯片 "Training MoEs – the systems side" 的标题下写着 "MoEs parallelize nicely – Each FFN can fit in a device" 与 "Enables additional kinds of parallelism"，并分别画出 "How the model weights are split over cores" 和 "How the data is split over cores" 两排示意，各列出 Data Parallelism、Model Parallelism、Model and Data Parallelism、Expert and Data Parallelism、Expert, Model and Data Parallelism 共 5 种切法。左侧的 "MoE Transformer Encoder with device placement" 结构图里标了 Feed Forward、ADD & Norm、Multi-Head Attention、Router (per-token)、All-to-All Dispatch / All-to-All Combine、FFN1…FFNe、Device 1 / Device 2 / Device e，以及 Input embeddings + Positional Encoding——专家并行的关键动作就是把激活经 All-to-All 发到对应设备再收回来。

第二个系统红利是稀疏矩阵乘。如果一块 GPU 上放多个专家（比如 Expert-0/1/2 共 3 个），最朴素的想法是做"多次小矩阵乘"（multiple small matrix multiplies），但讲师说这并不好：硬件更喜欢**大的**矩阵乘，这样才能复用缓存（reuse caches）并做各种优化。更好的写法是利用稀疏性——块对角（block diagonal）只是最基本的形式，还可以用硬件原生支持（natively supported in hardware）的更复杂的结构化稀疏（structured sparsity），把"专家 × 输入"的乘法表达得非常干净、非常快。幻灯片 "MoE routing allows for parallelism, but also some complexities" 把三条路线并排画出来：(A) Batched Matrix Multiplication，"Compute a set of independent matrix multiplications of the same size in parallel."，轴上标 hidden_size、ffn_hidden_size、expert_capacity；(B) Block Diagonal Matrix Multiplication，"Expert computation can equivalently be computed using block diagonal matrix products with equal sized blocks along the diagonal."，把 $T$ 个 token 堆起来一次性算；(C) Block Sparse Matrix Multiplication，"We can enable load imbalanced routing and variable sized experts by expressing expert computation as block sparse matrix multiplication."，专门应付负载不均衡、专家大小不一的情形。底部一行写着 "Modern libraries like MegaBlocks (used in many open MoEs) use smarter sparse MMs"。讲师把这一契合称为"硬件架构协同设计"（hardware architecture co-design）：MoE 的计算模式几乎天然对应那些硬件友好、容易高效支持的结构化矩阵乘。

![三种专家计算表达：批量矩阵乘、块对角矩阵乘与块稀疏矩阵乘](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0053_01-13-14.jpg)

第三个话题是通信开销，来自很新的 Nemotron 3 工作。专家并行时必须把激活从一个设备搬到另一个设备（"you belong in that expert, I need to ship you over"），All-to-All 集合通信会带来显著开销（significant communications overhead）。于是有人反过来设计维度：共享专家（shared expert）不参与通信，可以留在很大的维度上；而需要通信的专家，希望激活是更小、更低维的向量。做法是在发送之前先把残差流（residual stream）**下投影**一次，

$$z = W_{\downarrow} x \in \mathbb{R}^{d_c},\quad d_c \ll d,\qquad \hat{x} = W_{\uparrow} z,$$

然后对 $z$ 发起集合通信（collective communication call），到目标设备后再上投影回去，传输量按 $d_c/d$ 的比例下降。这样既省了通信量，又不必承受"隐藏维度被永久缩小"的代价。幻灯片 "MoE parallelism and architecture modifications" 引的是 "2.2. LatentMoE: Hardware-Aware Expert Design for Improved Accuracy per Byte"，并排展示 (a) Standard MoE architecture 与 (b) LatentMoE architecture：后者在 Router 前多出 "Latent down-proj"、在 All-to-All combine 后多出 "Latent up-proj"，专家从 E1–E4 扩到 E1–E8 并保留 SE（shared expert），底部一句话概括 "New ideas from Nemotron 3 – down-projecting the activations to reduce communication"。

本段最后还留了一个"有趣的旧毛病"：随机性（stochasticity）。如果专家在输入分布上没有被完美均衡，某个特别热门的专家（比如 expert 0）的 token 队列会不断堆积（building up and building up），长到一定程度系统就不得不开始丢 token（dropping tokens）。早期一代 MoE 推理框架的做法是**静默丢弃**：把这个专家直接跳过、回传一个 0，然后"当作什么都没发生"继续算。由此产生很反直觉的行为——只要别的用户也在往你用的那几个专家发请求，就可能把你挤出专家队列（bump you out of the expert queue），于是你的结果反而变差。局限也就在这里：这种随机性不是模型层面的，而是服务层面的、取决于同批里其它请求的，因而很难在单机实验里复现；它和"共享专家无法获得并行收益、可复制以省通信"是同一类系统权衡。下一节会讲 dropless 架构如何把这个问题基本解决。

## dropless、路由 softmax 稳定性与微调过拟合 [76:20 - 79:39]

先收掉上一节丢 token 的尾巴：讲师说这些"如今已有 dropless 架构把这类问题去掉了一大半"，MegaBlocks 以及其它常见开源 MoE 框架不再有这个毛病，"所以这已经不算一个问题了"。紧接着 76:35 起进入本讲最后一个 MoE 设计议题——稳定性（stability）。他回扣上一讲架构设计时的口头禅："exponentials are bad, divisions are bad"，因此 softmax 属于稳定性问题的"危险区"（danger zone）；而 MoE 偏偏在路由里又引入了一个 softmax：先 top-k、再 softmax，用它来给专家加权。早年在 Google 做 MoE 的 Barret Zoph 等人就有一整篇论文专门讨论 MoE 稳定性，softmax 正是他们点名的问题之一。

![路由稳定性：bfloat16 舍入误差示例、float32 路由器与 z-loss 式(5)](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0055_01-16-26.jpg)

幻灯片 "Issues with MoEs – stability" 给了一个很具体的数值例子："Exponential functions have the property that a small input perturbation can lead to a large difference in the output. As an example, consider inputting 10 logits to a softmax function with values of 128 and one logit with a value 128.5." 由于 softmax 为数值稳定会先减去最大值，bfloat16 里 0.5 的舍入误差会把 128.5 变成 128，于是 softmax 输出被改动约 36%：

$$\frac{\exp(0)}{\exp(0)+10\exp(-0.5)} \approx 0.142 \;\longrightarrow\; \frac{\exp(0)}{\exp(0)+10\exp(0)} \approx 0.091.$$

原文补充 "This example was in bfloat16, but analogous situations occur in OLMo v2 with larger logit values"，右侧标注来源 [Zoph 2022]。图里的路由流程还标出 (1) Routing、(2) Permutation、(3) Computation、(4) Un-permutation，Router 输出 Expert Indices 与 Probabilities，并出现 capacity_factor=1、capacity_factor=2 与 "Up-penalize the results and scale each by its Expert Probability" 等字样；底部一句 "Token dropping from routing happens at a batch level – this means that other people's queries can drop your token!" 正好把上一节的诡异现象归因到 batch 级丢弃。

对应的解法在幻灯片上写得很直白："Solution: Use float 32 just for the expert router (sometimes with an aux z-loss)"——只为专家路由器单独保留 float32 精度，其余部分仍可用低精度；另一个手段是 z-loss，本讲前文讨论 softmax 稳定性时已经见过，这里把它加在路由器上。幻灯片给出的式(5)为

$$L_z(x) = \frac{1}{B}\sum_{i=1}^{B}\left(\log\sum_{j=1}^{N} e^{x_j^{(i)}}\right)^2,$$

即对 log-sum-exp（配分函数的对数）取平方再对 batch 平均，惩罚 logits 整体幅值变大。讲师说 z-loss "quite popular"，早期做 MoE router stability 就在用；OLMo 也做过加/减 z-loss 的消融，"从那些非常尖刺的训练损失曲线（very spiky training loss curves）可以相当清楚地看出，路由器上的 z-loss 挺有帮助"。易错点是：float32 与 z-loss 都是**缓解**而非根治，且只对路由器这一处生效，专家内部的低精度计算不受保护；把 z-loss 系数设太大同样会干扰路由分布。

最后一个稳定性之外的麻烦是微调（fine-tuning）。MoE 参数量太大，若直接微调专家，会出现相当严重的过拟合（overfitting）。幻灯片 "Issues with MoEs – fine-tuning" 标题下写 "Sparse MoEs can overfit on smaller fine-tuning data"：上方 SuperGLUE CB Task 曲线纵轴 Metric 从 80.0 到 100.0、横轴 Step，四条线分别是 Sparse train_eval、Sparse validation_eval、Dense train_eval、Dense validation_eval——稠密模型的 train 与 val 靠得很近，稀疏模型的 train/val gap 却"极其大"，讲师提到这是 GLUE 里的某个下游任务。左下 "Zoph et al solution – finetune non-MoE MLPs" 的柱状图纵轴 SuperGLUE Score 81–87，横轴 "Parameters Being Updated" 为 All、Non MoE、MoE、Attention、FFN：只更新 MoE 那一栏明显最低（约 82.7 且误差棒很大），更新 Attention 约 85.7、FFN 约 86.5、All 与 Non MoE 约 86.2。于是常见的三种干预是：只微调非 MoE 的前馈层、只微调注意力（近期 MoE 工作里尤其常见）、以及只要模型里有非 MoE 层就干脆只调那些层——Zoph 等人主张的正是这类做法。右下 "DeepSeek solution – use lots of data 1.4M SFT" 则给出"苦涩的教训"（bitter lesson）版本：引文说 "we conduct supervised fine-tuning (SFT) on our in-house curated data, comprising 1.4M training examples"，覆盖 math、code、writing、question answering、reasoning、summarization 等，且以中英双语为主；用 140 万条样本基本可以在微调阶段把整个 MoE 重训一遍，泛化差距就不那么明显了。局限也很清楚：前三种干预等于放弃了 expert 参数的适配能力，而"海量数据"路线对多数下游任务并不现实。

## Upcycling：把稠密模型复制成 MoE [79:39 - 82:11]

最后一个"MoE 专属技巧"是 upcycling（升级再造，即从稠密模型初始化 MoE）。讲师先给了个时间判断：这在过去一年"变得很不流行"，"今年我好像一个 upcycled model 都没见到"，但因为想法漂亮、早些年也确有几个好模型用了它，所以仍值得讲。定义很直接：想要 MoE，不必从零训，可以拿一个已训好的稠密模型（dense model），在它之上实例化一个 MoE——把包括 MLP 在内的所有权重都复制（copy everything）过来，把 MLP 复制成很多份作为专家，路由器（router）随机初始化（random initialization），然后继续训练。由于"哪些输入落到哪个 MLP"带有随机性（stochasticity of which inputs go to where），这些原本相同的副本会逐渐分化、专门化（specialize），最终就得到一个真正能用的 MoE。这个初始化过程可以写成一段伪代码：

```python
def upcycle(dense_ckpt, E, top_k):
    moe = copy_all_weights(dense_ckpt)                                # attention/norm/embedding 全部照搬
    moe.experts = [copy.deepcopy(dense_ckpt.mlp) for _ in range(E)]  # MLP 复制 E 份
    moe.router = random_init()                                        # 路由器随机初始化
    moe.top_k = top_k
    train(moe)                                                        # 靠路由随机性让专家分化
    return moe
```

幻灯片 "Other training methods - upcycling" 把结构画得很清楚：上方 "Original Dense Block" 是 Layer Norm → Attention → Layer Norm → MLP 加两处残差相加，三处 "copy weights" 指向 "Base E MLP copies"，下方 "Upcycled MoE Block" 里 MLP 被换成 Router 加一个 MoE 模块，内含 MLP 1、MLP 2、…、MLP E，末端接 Weighted Sum 再做残差相加；底部抛出问题 "Can we use a pre-trained LM to initialize a MoE?"。右侧曲线纵轴是 "C4 Validation Token Accuracy"（68% 到 75%），横轴是 "Extra Pretraining Time (TPU-core-days)"，对数刻度从 $10^1$ 到 $10^3$；图例 method 为蓝色 Dense 与橙色 Upcycling，variant 用不同大小的点表示 Base、Large、XL，三条虚线分别标在约 68%（Base）、72%（Large）、74%（XL）。橙色 upcycling 曲线随额外预训练时间一路爬到约 75%，明显超过同算力下蓝色的 dense 延续训练。早期提出 upcycling 的论文正是用这类图说明：相比"继续训练同一个稠密模型"，upcycle 能拿到好得多的语言建模验证准确率。这与前面几章的关系是：upcycling 绕开了从零训 MoE 时最麻烦的路由冷启动问题，代价是继承了一个按稠密目标优化过的 MLP。

规模上的验证给了两个中国团队的例子。MiniCPM 是讲师很欣赏的模型，因为他们做大量受控消融：他们把 MiniCPM 2.4B 稠密模型 upcycle 成 13.4 billion parameter 的 MoE，并从中拿到"几乎是免费的收益"（almost free wins）。Qwen 在做 QwenMOE 时则从 1.8B 的 Qwen 模型初始化，upcycle 成 Qwen1.5-MoE-A2.7B，成为"最早的大规模 upcycling 成功案例之一"，也是当时性能很高的模型之一。

![Qwen MoE 的 upcycling 配置与各基准对比表：top-k=4、60 experts w/ 4 shared](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0058_01-21-50.jpg)

"Upcycling example – Qwen MoE" 这张表把这些数字坐实了。副标题是 "Qwen MoE – Initialized from the Qwen 1.8B model top-k=4, 60 experts w/ 4 shared."：从 1.8B 出发，top-k=4、共 60 个专家、其中 4 个是共享专家。表格列为 Model、#Parameters、#(Activated) Parameters、MMLU、GSM8K、HumanEval、Multilingual、MT-Bench：Mistral-7B 为 7.2 / 7.2 / 64.1 / 47.5 / 27.4 / 40.0 / 7.60；Qwen1.5-7B 为 7.7 / 7.7 / 64.6 / 50.9 / 32.3 / – / –；Gemma-7B 为 8.5 / 7.8 / 61.0 / 62.5 / 36.0 / 45.2 / 7.60；DeepSeekMoE 16B 为 16.4 / 2.8 / 45.0 / 18.8 / 26.8 / – / 6.93；Qwen1.5-MoE-A2.7B 为 14.3 / 2.7 / 62.5 / 61.5 / 34.2 / 40.8 / 7.17。可以看到它用 2.7B 激活参数就逼近 7B 级稠密模型的 MMLU，底部一句 "Similar architecture / setup to DeepSeekMoE, but one of the first (confirmed) upcycling successes" 点明架构与 DeepSeekMoE 相近、但这是最早被确认的 upcycling 成功之一。

不过讲师立刻给这个技巧判了"历史地位"：如今基本没人在 upcycle 了，因为你不会先训一个稠密模型再转换它——"不如一开始就直接用 MoE 训你的 hero run（主跑）"。易错点也有两处：一是路由器随机初始化意味着训练初期分派近乎随机，专家分化的收益要等额外预训练时间堆上去才显现（右图横轴正是对数刻度的 TPU-core-days）；二是复制出来的 $E$ 份 MLP 初始完全相同，若没有足够的路由随机性或均衡约束，它们可能长期保持同质、退化成"一个被切碎的大 MLP"，这也是它必须与 Switch/DeepSeek 那套辅助损失配合使用的原因。

## DeepSeek V1 → V2 → V3 的演进、MLA 与 MTP [82:11 - 86:20]

最后一段用 DeepSeek 三代模型的演化把整讲的线收起来。讲师先推荐读 DeepSeek 论文，理由是"写得很漂亮、有很多值得挖的细节"。V1（DeepSeek MoE）已经是大量现代 MoE 的原型（prototype）：共享专家 + 细粒度专家结构（shared and fine-grained expert structure）、标准 top-k 路由加辅助损失均衡，他甚至称之为 MoE 的"柏拉图式理想"（Platonic ideal）。V2 基本就是"把它放大"：2 个共享专家、更多细粒度专家，并加入设备级路由（device routing）与通信均衡（communication balancing）两个组件——讲师特意点明，这两项"本质上都是为优化系统而加的辅助损失"，并借此说出一句全场反复出现的判断："successful language model training is not just about deep learning, it's also about really respecting your system"，成功的语言模型训练不只是深度学习，还必须真正尊重系统约束。V3 仍保留共享 + 细粒度专家，但换了均衡方式：采用 aux-loss-free（无辅助损失）的 bias 方案，并把专家加权从纯 softmax 改成 sigmoid plus softmax 的组合，"但大体相似"（mostly similar）。

![DeepSeek MoE V3：671B/37 active、共享1+细粒度258专家、Sigmoid+Softmax topK 与 aux-loss-free 公式](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0059_01-23-38.jpg)

幻灯片 "DeepSeek MoE v3" 给出的具体规格是 "V2 (671B – 37 active)"（画面标注如此），以及 "Shared (1) + Fine-grained (258) experts, 8 active"——1 个共享专家、258 个细粒度路由专家、每次激活 8 个；示意图里标了 Transformer Block × L、Feed-Forward Network、RMS Norm、Input Hidden $u_t$、Output Hidden $h_t$、Router Top-$K_r$、Routed Expert、Shared Expert。"New things" 下面左边一栏 "Sigmoid+Softmax topK + topM" 给出完整前向式：

$$\mathbf{h}'_t = \mathbf{u}_t + \sum_{i=1}^{N_s}\mathrm{FFN}^{(s)}_i(\mathbf{u}_t) + \sum_{j=1}^{N_r} g_{j,t}\,\mathrm{FFN}^{(r)}_j(\mathbf{u}_t),$$

$$g_{j,t} = \frac{g'_{j,t}}{\sum_{j=1}^{N_r} g'_{j,t}},\qquad g'_{j,t}=\begin{cases}s_{j,t}, & s_{j,t}\in\mathrm{Topk}\left(\{s_{j,t}\mid 1\le j\le N_r\}, K_r\right)\\ 0, & \text{otherwise}\end{cases},\qquad s_{i,t}=\mathrm{Sigmoid}\!\left(\mathbf{u}_t'^{T}\mathbf{e}_i\right).$$

右边 "Aux-loss-free + seq-wise aux" 则把偏置 $b_i$ 写进 top-k 判据：

$$g'_{i,t}=\begin{cases}s_{i,t}, & s_{i,t}+b_i\in\mathrm{Topk}\left(\{s_{j,t}+b_j\mid 1\le j\le N_r\}, K_r\right)\\ 0, & \text{otherwise}.\end{cases}$$

也就是说 V3 不再靠 $\alpha_1\sum f_iP_i$ 那样的乘性辅助损失，而是给每个专家维护一个可在线调整的 bias $b_i$，用它把选中集合调平，同时仍保留一个序列级（seq-wise）的辅助项兜底——这正对应 67:10 之后讲师说的"V3 想去掉辅助损失、但极端失衡仍需兜底"。易错点：sigmoid 打分后仍要做一次归一化（分母 $\sum_j g'_{j,t}$），别把 $s_{i,t}$ 直接当成门控权重；另外画面上 "V2 (671B – 37 active)" 的标注与标题 "DeepSeek MoE v3" 不一致，转写里也未念到，按"不猜"的原则照录即可。

既然讲到 V3，讲师顺带把其余两个亮点过了一遍。其一是 MLA（multi-head latent attention，多头潜在注意力）：不再由隐藏输入直接产生 Q、K、V，而是先产生一个更低维的 latent $c$，再由 $c$ 生成 Q、K、V；推理做 KV cache 时"不必缓存所有 K 和 V，只需存这些 $c$"，而 $c$ 维度更低，于是显著省显存——这也是 MLA 里 "latent activation" 的由来，$c$ 才是真正需要保存的 KV cache 量。唯一要小心的是它与 RoPE（旋转位置编码）冲突：做 KV caching 时得仔细处理"哪些维度被旋转"，口诀是用**非 latent 的维度**（non-latent dimensions）来编码位置，讲师表示不在这里展开。其二是 MTP（multi-token prediction，多 token 预测）：一次预测多个未来 token 而不是一个，统计上有"也许能更好地预测未来"的论据，系统上更妙——相当于内置了一个投机解码器（speculative decoder），Percy 后面讲推理时会展开，本质是加速解码的技巧；不过他补了一句这个想法"并没有很流行"（hasn't caught on very much）。

![MoE 总结三条：稀疏性、离散路由与 top-k 启发式、以及大量经验证据](/coursenotes/cs336-language-modeling-sp26/l04-attention-alternatives/kf_0060_01-25-55.jpg)

片尾 85:41 起讲师收束全讲："MOEs are this kind of very clever idea to take advantage of sparsity so that you can have more parameters than you're sort of paying for"——用稀疏性换来"参数量远超你实际付的算力"，不为全部参数付计算成本，却拿到参数带来的收益；"You might initially think that this kind of routing problem is very hard, but it turns out that very simple things work well even at scale"，离散路由看似很难，结果即便在规模上，非常简单的做法也够用；"at this point, it's kind of clear that MOEs are here to stay"，所以应当理解它们如何工作，随后以致谢结束。屏幕上 "MoE summary" 三条正是这份收束："MoEs take advantage of sparsity – not all inputs need the full model"、"Discrete routing is hard, but top-k heuristics seem to work"、"Lots of empirical evidence now that MoEs work, and are cost-effective"。从 64:45 的 $f_iP_i$ 梯度解释，到 66:00 的双层均衡、71:41 的系统协同、79:39 的 upcycling，再到这里的三代谱系，整段其实都在为这三句话提供证据。局限也一并留在了这里：MLA 与 RoPE 的兼容需要额外的非 latent 维度来编码位置，MTP 尚未被广泛采用，而 aux-loss-free 也并非真的"零辅助损失"——V3 仍留了 seq-wise aux 防止极端失衡；这些未收干净的尾巴，恰恰说明 MoE 的工程细节远没到可以一键套用的程度。

## 要点回顾

### 成本结构：注意力为何成为瓶颈
- 上下文窗口从 2018 到 2025 由千级跃迁到千万级 token（对数刻度），各家厂商竞相扩大窗口以支撑更复杂负载。
- 前馈成本 $\Theta(n\,d_{\text{model}}^2)$ 线性，注意力 $\Theta(n^2 d_k)$ 因 all-to-all 连接而二次；旧的大模型短序列设定下 FFN 才是主导，序列变长后注意力迅速反超。
- 基础工具箱只「控制」不「消除」二次项：每 8 层做 1 次全局注意力的局部/全局混合（$\#\text{global}/\#\text{layers}=1/8$），以及不改复杂度、只减少显存搬运的 FlashAttention。
- A100 80GB SXM4 上基础 Pytorch 在 16k 处 OOM，FlashAttention-2 从 1k 的 132 升到 16k 的 176 TFLOPs/s；但 500 万到 1000 万 token 时这些 tricks 都不够用。

### 线性注意力与 RNN 对偶
- 起点 $\mathrm{Attn}(Q,K,V)=\rho(QK^\top)V$，二次项来自 $n\times n$ 的 $QK^\top$；去掉 $\rho$ 后由结合律得 $(QK^\top)V=Q(K^\top V)$，复杂度从 $n^2d_k+n^2d_v$ 变为 $2nd_vd_k$。
- $n$ 可达百万量级，而 $d_k,d_v$ 通常几千到几万，把二次依赖挪到隐藏维度是一次划算的交换；但 $\rho\to I$ 改变了函数形式，是**有损**近似。
- 递推形式 $S_t=S_{t-1}+k_tv_t^\top$、$y_t=q_t^\top S_t$，状态 $S\in\mathbb{R}^{d_k\times d_v}$ 大小固定，每步 $O(d_vd_k)$ 与 $n$ 无关；稠密形式并行适合训练，串行形式适合推理。
- 等价链：$\rho(QK^\top)V\xrightarrow{\text{lossy}}Q(K^\top V)\xrightarrow{\text{exact}}S_t$ 递推；性能损失出在丢掉 softmax 那一步，而非稠密转递推。 RetNet 注记对应给 $S_{t-1}$ 乘衰减因子 $\gamma$。

### 门控递推与混合比例
- Mamba-2：$S_t=\gamma_t S_{t-1}+k_tv_t^\top$、$y_t=q_t^\top S_t+v_t^\top D$，$\gamma_t=f(x_t)$ 只依赖输入；$v_t^\top D$ 是让当前 value 直通透传的读出端残差旁路，不属于状态更新。
- Gated DeltaNet：$S_t=\gamma_t(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top$，$\beta_t$ 是「无输入操作」门、$\gamma_t$ 对应遗忘门；$(I-\beta_tk_tk_t^\top)$ 只是近似投影，因未做单位归一化而非严格正交投影。
- 门控只用 input-dependent terms、保持 no state dependence，是并行训练/串行推理对偶性存续的前提；讲师判断规模化方法多收敛到「very LSTM-like objects」。
- 产业比例：MiniMax M1 用 7:1、Nemotron 3 约 3:1、Qwen 3.5 / Qwen Next 用 3:1；ByteDance Seed 与 UC Santa Cruz 的 RULER 扫描 1:1、6:1、12:1、24:1、pure，full attention 基线约 0.42，pure 端红色曲线跌到约 0.15。

### 稀疏注意力 DSA
- 闪电索引器打分 $I_{t,s}=\sum_{j=1}^{H^I}w^I_{t,j}\cdot\mathrm{ReLU}(q^I_{t,j}\cdot k^I_s)$，细粒度选择 $u_t=\mathrm{Attn}(h_t,\{c_s\mid I_{t,s}\in\mathrm{Top\text{-}k}(I_{t,:})\})$。
- 索引器头数少、可跑 FP8，选 ReLU 出于吞吐考虑；成本为 $O(n^2d^I)H^I+O(k^2d)$，$k\ll n$ 是人为设定、与输入总长解耦的封顶常量。
- 索引器仍要算全体 QK 内积，是 all-to-all 二次的「brute force inner products」，故 DSA **不是线性时间**，收益来自常数因子与候选集规模。
- post-hoc 适配配方：短上下文预训练 → 长上下文扩展 → 后训练；DeepSeek-V3.2 性能与同期前沿模型相当，GLM-4.7-Flash + DSA（warmup-only 只训索引器、full DSA 联合训练 150B tokens）在长上下文检索上损失很小。

### MoE 的动机、收益与路由
- 把大 FFN 换成 $N=4$ 个同尺寸专家加 selector，$\text{Params}\approx 4\times P_{\mathrm{FFN}}$ 而 $\text{FLOPs}\approx 1\times P_{\mathrm{FFN}}$，即增加参数不影响每 token 计算量；专家不是语义分工，只是参数块。
- Switch Transformer 曲线：激活参数固定、专家数 $E:1\to256$，test loss 从约 6.0 降到约 4.8；OLMo 观察到相对稠密约 2 倍训练加速，DeepSeek-V2 以少得多的激活参数拿到不逊的 MMLU。
- 设计三轴为 Routing function、Expert sizes、Training objectives；token choice top-k 是共识（OLMo 中 TC 验证 loss 更低、下游分更高），内积路由 $s_i=u^\top e_i=\langle u,e_i\rangle$ 绝对主流。
- 其余路由：哈希作基线（收益不如 top-k、不进部署）、RL/bandit（Bengio 2013、Clark 2020，因梯度方差与复杂度不常用）、线性指派（Clark '22，太贵未见规模应用）；各模型 $K$ 为 Switch 1、GShard/Mixtral/Grok 2、Qwen/DBRX 4、DeepSeek 7。
- top-k 方程 $\mathbf{h}_t^l=\sum_{i=1}^{N}(g_{i,t}\mathrm{FFN}_i(\mathbf{u}_t^l))+\mathbf{u}_t^l$，$s_{i,t}=\mathrm{Softmax}_i(\mathbf{u}_t^{l\top}\mathbf{e}_i^l)$；DeepSeek 的细粒度专家分割与绕过 router 的共享专家（Shared Expert Isolation）已成标准模板。

### 训练 MoE：启发式均衡损失
- 训练必须稀疏却带来门控不可微、看不到未选中专家的梯度；三条路线中 RL 与随机扰动均未被采用，落地的是启发式均衡损失，用以抑制「富者愈富」的专家坍缩。
- Switch 辅助损失 $\text{loss}=\alpha\cdot N\cdot\sum_{i=1}^{N}f_i\cdot P_i$，$f_i=\frac{1}{T}\sum_{x\in\mathcal{B}}\mathbf{1}\{\mathrm{argmax}\,p(x)=i\}$ 为硬计数、$P_i=\frac{1}{T}\sum_{x\in\mathcal{B}}p_i(x)$ 为总概率质量；梯度 $\frac{\alpha N}{T^2}\sum\mathbf{1}_{\mathrm{argmax}\,p(x)=i}$ 使越热门的专家受越强下调。
- Shazeer 2017 随机近似 $G(x)=\mathrm{Softmax}(\mathrm{KeepTopK}(H(x),k))$ 加高斯扰动，Fedus 2022 代码 `router_logits += mtf.random_uniform(..., minval=1-eps, maxval=1+eps)` 并 `mtf.to_float32(...) for stability`，后续 Zoph 2022 将其移除。
- DeepSeek V1/V2 叠加双层：$\mathcal{L}_{\text{ExpBal}}=\alpha_1\sum f_iP_i$（式 12–14）与 $\mathcal{L}_{\text{DevBal}}=\alpha_2\sum f_i'P_i'$（式 15–17，$f'$ 取平均、$P'$ 取求和）；V3 改用逐专家 bias $b_i$ 的 aux-loss-free 方案，仍留 seq-wise aux 兜底极端失衡。
- OLMo 消融（Figure 9/10）：去掉 LBL 后训练/验证 loss 明显变差、几乎全部 token 涌向两个专家；结论是只需加均衡损失、其余当作正常反传，即可让含 top-k 非可微系统的训练稳定。

### 系统、稳定性与 DeepSeek 谱系
- 专家并行补上第三个并行轴，代价是 All-to-All 通信；可用块对角/结构化稀疏矩阵乘（MegaBlocks）、LatentMoE 下投影 $z=W_\downarrow x\in\mathbb{R}^{d_c},\ d_c\ll d$ 降传输量，dropless 架构基本解决早期静默丢 token。
- 路由 softmax 是新稳定性风险点：bfloat16 例子里输出从约 0.142 变到约 0.091（改动约 36%）；解法是路由器单独用 float32 与 z-loss $L_z(x)=\frac{1}{B}\sum_{i=1}^{B}(\log\sum_{j=1}^{N}e^{x_j^{(i)}})^2$。
- MoE 微调易过拟合（train/val gap 极大）：Zoph 主张只更新非 MoE 的 FFN 或 Attention，DeepSeek 则用 1.4M 条 SFT 数据基本整体重训；upcycling 由 `copy.deepcopy(dense_ckpt.mlp)` 复制 E 份 + `random_init()` 路由器，MiniCPM 2.4B→13.4B、Qwen1.5-MoE-A2.7B（top-k=4、60 experts w/ 4 shared）曾验证，现已不流行。
- DeepSeek 谱系：V1 共享+细粒度原型 → V2 设备级路由与通信均衡 → V3 规格为 1 共享 + 258 细粒度专家、8 active、Sigmoid+Softmax 加权与 aux-loss-free；另有 MLA（低维 latent $c$ 省 KV cache，需用非 latent 维度处理 RoPE）与 MTP（多 token 预测，相当于内置投机解码器）。
- 全讲收束：MoE 用稀疏性换取「参数量超过所付算力」，离散路由虽难但 top-k 启发式在规模上够用，已有大量经验证据表明其有效且划算；成功训练不只是深度学习，还必须尊重系统约束。

## 视频来源

- 原始链接：https://www.youtube.com/watch?v=cKSwj_qZ8Jg
- 课程：Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 4

---

## 对照参考笔记补遗（Spring 2025 版）

> 编者注：本节依据 2025 版参考笔记整理，其中的「更正」小节记录的是核对时发现的问题，对应正文已在本版订正；补遗内容不代表 2026 课程视频原文。


本节内容全部来自 Spring 2025 版参考笔记，仅用于与本讲（Spring 2026 视频）对照补充，不代表 2026 视频原文或讲师口径；其中数字、公式与讲法若与本讲不一致，一律以本讲正文为准。

### 一、稀疏收益的量化倍数与细粒度专家的维度设计

本讲以「Same FLOP, more param does better」立论，用 Switch Transformer 曲线（激活参数固定、专家数 $1\to256$、test loss 由约 $6.0$ 降到 $4.8$）与 OLMo 口语「约 2 倍」佐证。2025 参考把这条收益链补成了可引用的量级：Fedus et al. 2022 报告 MoE 达到与稠密模型相同 perplexity 只需约七分之一训练时间，即 **7× speedup**；AI2 OLMo 的受控复现给出「相似性能下 **3× less FLOPs** 或 **2× faster training time**」。两个实验固定的量并不相同——前者固定质量比算力，后者与「same FLOP」互补。把这些倍数与 $6.0\to4.8$ 的曲线并置，才能判断「加速」是渐近趋势还是具体数值。易错点：7× 与 2× 出自不同论文、不同设定，不能相乘或互相替换，本讲转写里的「or something like that」本就是口语估计。

本讲把细粒度专家描述为「切得更小更细」，2025 参考给出了可计算的维度关系。标准 FFN 把隐维放大 4 倍，记作 $H\to 4H$；若改为 $H\to 2H$，单专家的参数与 FLOPs 减半，专家数便可翻倍而激活参数不变，幻灯片写作「2W instead of 4W, N experts instead of N/2」。一般地取 $H\to H/X$，即得 $X$ 倍数量的专家，这就是 **fine-grained expert segmentation**（细粒度专家分割）。于是配置表里的 fine-grained ratio $1/4,1/8,1/14$ 可读成「单专家相对标准 FFN 的尺寸比」：ratio 越小，专家越碎、数目越多。它与本讲的「共享专家隔离」是同一页幻灯片上相邻的两步——先碎化，再挑出少数专家常开。

2025 参考还记录了 OLMoE 的对照消融：细粒度专家从 8 增到 64，训练损失、验证损失与 MMLU Var 都稳步改善；但 **shared experts**（共享专家）在 OLMo 设定下未显出显著收益，OLMo 最终选择不用共享专家。这与本讲「DeepSeek 与 OLMo 在共享专家上略有分歧、在细粒度多专家上一致」互为佐证。易错点有二：ratio 是尺寸比例而非专家个数；DeepSeek 那组柱状图是**归一化后的相对性能**、且刻意控制了总参数量与激活参数量相同，不能与其它论文的绝对分数直接比较。

### 二、三种指派范式与 Top-K 路由的实现细节

本讲把 **token choice top-k** 定为工业共识，并用 OLMo 的 TC/EC 曲线说明 token choice 验证损失更低。2025 参考把另外两条被略过的路线讲全了。**Expert Choice**（专家选 token）反过来让每个专家按偏好分挑 top-K 个 token，因为每专家处理固定数量 token，**load balancing**（负载均衡）是内建性质、无需额外辅助损失；代价是某个 token 可能因最偏爱的专家「已满」而进不去，相关性被牺牲。**Global Routing via Optimization**（全局优化路由，即 BASE Routing、Clark '22）把指派写成 **linear assignment / optimal transport**（线性指派／最优传输）问题，理论上同时最优地权衡 token 相关性与专家利用率，却因求解代价过高而「Hasn't been seen at scale at all」。三者构成一条清晰的取舍轴：均衡是内建、后补，还是全局最优。

本讲把 **Hash Routing**（哈希路由）列为「有收益、不如 top-k、只作基线不进部署」，2025 参考补上了「为什么随机指派也能涨点」的直觉：即便用无语义的 hash 把 token 直接映射到 FFN，每个专家仍会稳定见到一份数据子集；自然语言近似 **Zipfian distribution**（齐普夫分布），高频 token 与模式反复落到同一专家，于是专家会**隐式地专门化**（implicit specialization）到常见模式上。这把哈希路由与本讲「expert 并非语义专家」的判断接上了：专门化来自数据分布而非语义理解。也正因它无法按输入内容自适应，增益才明显小于可学习的内积 top-k，部署上不可行。

本讲已写出 $s_{i,t}=\mathrm{Softmax}_i(u_t^{l\top}e_i^l)$ 与 $g_{i,t}$ 的分段门控，2025 参考的课堂问答补齐四处实现细节。其一，早期论文主张 $K>1$（常用 **K=2**）是为**鼓励探索**（exploration），避免像多臂老虎机那样永远只选当前最强臂而错过更优专家；$K>1$ 时多专家输出按门控分数加权求和。其二，**先 softmax 再 Top-K** 时，非 top-K 置零后 $g_{i,t}$ 不再满足 $\sum_i g_{i,t}=1$，这可接受，因为后续 **LayerNorm** 会重新调整尺度。其三，打分用的 $e_i$ 是 router 自学的 **expert embedding**（专家嵌入向量），与 $\mathrm{FFN}_i$ 参数**不共享（not tied）**，表征「该专家在寻找什么样的输入」。其四，Top-K 不可微，**未被选中专家的梯度恒为 0**——这正是本讲「训练时看不到反事实专家」那道难题的数学根源。

### 三、DeepSeek 三代架构的规格与系统级均衡

本讲的 DeepSeek 谱系收束在 V3，V1/V2 只留「共享+细粒度原型」「放大并加设备级路由与通信均衡」的定性描述。2025 参考给出可对照的规格：**V1（DeepSeek MoE）16B 总参 / 2.8B 激活**，2 个共享专家 + 64 个细粒度路由专家（各为标准 FFN 的 $1/4$ 大小），标准 top-k 加专家级与设备级辅助损失；**V2 236B / 21B 激活**，10 共享 + 160 细粒度、每次激活 6 个；**V3 671B / 37B 激活**，改用 Sigmoid + Softmax + TopM 与 aux-loss-free。本讲 V3 页记下的 671B/37B 与 2025 参考完全一致，是全讲少数可跨版本确认的数字；差异集中在 V1/V2 与 shared 计数上，宜按版本差异并置，不要互相覆盖。

V2 的两项「为系统而加」的创新在 2025 参考里有机制说明。**Top-M device routing**（设备级 top-M 路由）是两阶段过程：router 先选出 top-M 个**设备**，再仅在这些设备内部挑选专家，目的是把 token 流量收敛到少数设备、压低细粒度专家带来的跨设备通信；配套 **communication balancing loss**（通信均衡损失）同时平衡每个设备的**输入与输出**通信量，避免只算一半。这与本讲「逐设备均衡损失按设备聚合 $f_i',P_i'$」是同一套系统思路，只是粒度从「专家」提到了「设备」。易错点：Top-M 里的 M 是**设备数**、K 才是专家数，二者不是同一个旋钮。

把三节连起来看，V1 立住算法（共享+细粒度+辅助损失），V2 补系统（Top-M 与通信均衡），V3 改训练目标（bias 与序列级辅助项），正好对应本讲反复申明的「successful language model training is not just about deep learning」——架构演进越来越由通信拓扑而非建模能力驱动。2025 参考对 V3 的 shared 专家数给出 8（正文）与 2（表格）两个互斥值，本讲画面又记作 1 共享 + 258 细粒度，三说并存；这类数字冲突应标为版本差异或参考解析存疑，而不是据此回改 2026 正文。

### 四、逐专家偏置、序列级辅助损失与「无辅助损失」的真相

本讲已写出 V3 把 $b_i$ 写进 top-k 判据的精确式，2025 参考则把它的学习规则讲成了可操作的一步。DeepSeek V3 用 **per-expert biases**（逐专家偏置）替代乘性辅助损失，参考记作 $g'_{i,t}=s_{i,t}+b_i$；$b_i$ 用**在线梯度**（online gradient scheme）更新：某个专家分到的 token 偏少就把 $b_i$ **调高**、偏多就**调低**，于是不再需要显式 $\alpha_1\sum f_iP_i$ 那一项，故称 **auxiliary-loss-free balancing**（无辅助损失均衡）。这与本讲「V3 想去掉辅助损失、但极端失衡仍需兜底」是同一句话的两种写法，差别只在参考把 bias 直接加到分数上、2026 版把它只用于 top-k 选择而门控值仍取 $s_{i,t}$。

为什么去掉了又加回？2025 参考点明 batch 级均衡约束不了**单条序列内部**的失衡，推理遇到 **out-of-distribution sequences**（分布外序列）时仍可能倾斜，因此 V3 又引入 **complementary sequence-wise auxiliary loss**（互补的序列级辅助损失），在**序列级**而非仅 batch 级兜底。结论值得记牢：所谓「aux-loss-free」并非真的零辅助损失，而是「用 bias 换掉主损失、再留一条序列级尾巴」。易错点：不要把 bias 项与 aux 项当成二选一的开关，二者作用域不同——一个调选中集合、一个约束序列内分布；本讲讨论 V3 均衡时那句「目前还没有方案能完全去掉这些较丑陋的辅助损失」正是对它的注脚。

### 五、MLA、MTP 与微调阶段的架构级对策

本讲已说明 **MLA**（multi-head latent attention，多头潜在注意力）把隐藏状态先压成低维 latent $c$、KV cache 只存 $c$ 以省显存，并提醒与 **RoPE** 冲突时要用非 latent 维度编码位置。2025 参考补上系统侧的收尾：从 $c$ 上投影回 K/V 需要额外矩阵乘 $W^{UK}$，但 $W^{UK}$ **可与已有的 Q 投影矩阵合并**，因此这次投影并不带来净 FLOPs 增加；与 RoPE 兼容则需专门处理旋转矩阵作用在哪些维度上。这把「省显存」与「不额外付算力」分清楚了——MLA 的代价是矩阵重排与位置编码的工程复杂度，而不是计算量。

本讲把 **MTP**（multi-token prediction，多 token 预测）概括为「一次预测多个未来 token、相当于内置投机解码器、并未很流行」。2025 参考给出具体形态：除主 transformer 预测下一个 token 外，V3 用一个**轻量单层 transformer（MTP Module 2）**去预测「下下个」token，让模型在生成时向前看一步。这解释了它为何天然贴合 **speculative decoding**（投机解码）：额外那一步预测本身就是草稿模型。易错点：MTP 是训练目标与解码加速的双重技巧，不能只按「提升连贯性」理解；两版都注明它 hasn't caught on very much，不宜当成 V3 的核心收益。

本讲给出微调过拟合的三种干预（只调非 MoE 前馈层、只调注意力、只调非 MoE 层），2025 参考把第一种上升为架构设计：**Hybrid Architectures**（混合架构）并非每层都用 MoE，而是**交替堆叠 dense 层与 MoE 层**，微调时只更新行为更稳的 dense 层；另一路是 DeepSeek 用 **1.4M 条 SFT 样本**在微调阶段基本整体重训。此外两版在 upcycling 上有一处讲法差异值得留意：2025 参考的机制含「轻微扰动复制的 FFN 权重」以促使专家分化，本讲伪代码则不扰动权重、靠路由随机性（stochasticity of routing）让相同副本逐渐专门化。二者是同一技巧的不同讲授侧重，并存即可，不必据此改动本讲实现。
