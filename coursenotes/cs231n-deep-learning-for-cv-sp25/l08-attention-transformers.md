---
title: "第 8 讲 · 注意力机制与 Transformer"
description: "CS231n Spring 2025 L08 视频笔记；对照 2026 春季课件补充新增内容。"
---

# 第 8 讲 · 注意力机制与 Transformer

> **Stanford CS231n 计算机视觉 · L08** ｜ 讲者：Stanford CS231n 课程团队 ｜ 67 分钟 ｜ [原视频](https://www.youtube.com/watch?v=RQowiOF_FvQ)

> 课件版本：2026 春季 ｜ 视频版本：2025 春季。2026 课件新增或调整的内容集中在文末「2026 课件更新」一节。

## 核心结论

这一讲回答一个问题：怎样让序列模型不必把所有输入压进一个定长向量。讲者从 RNN 的 seq2seq 翻译讲起，指出 **context vector** 是输入与输出之间唯一的通道，也是瓶颈；然后用 attention 让解码器每一步都回看整个输入。接着把 attention 从 RNN 中剥离，连续做三次泛化——点积相似度、多 query 并行、key/value 分离——得到一个只由矩阵乘法组成的 **self-attention** 层。Transformer 用 self-attention 负责向量之间的通信，用 MLP 负责逐向量的计算，再配上残差与 LayerNorm。代价是 $O(N^2)$ 的计算与显存，回报是极强的并行性和一层之内的长程依赖；今天几乎所有大规模模型都是 transformer。

## 回顾：从 RNN 到 seq2seq [00:04 - 07:15]

上一讲把神经网络从“一张图进、一个标签出”的一对一问题扩展到序列：一对多（图像描述）、多对一（视频分类）、多对多（等长或不等长的翻译）。这一讲在此基础上引入两样东西：**attention** 这个作用于向量集合的新原语，以及以 self-attention 为核心的 **transformer** 架构。讲者先给了一个判断：今天在图像分类、图像生成、文本生成、语音等方向上，所有大规模、在大量数据上训练过的模型，基本都是 transformer。

机器翻译是贯穿全讲的例子。输入英文词序列，输出意大利文词序列，两种语言的词数和词序都没有对应关系，正是序列到序列的典型场景。编码器是一个 RNN，按 $h_t = f_W(x_t, h_{t-1})$ 逐词推进，把整句话编码进隐状态；解码器是另一个参数不同的 RNN，按

$$
s_t = g_U(y_{t-1}, s_{t-1}, c)
$$

逐词生成。$c$ 是 **context vector**，通常取编码器最后一个隐状态 $h_T$，因为它经由循环结构已经“看过”整句话；$s_0$ 通常也由编码器最后一个隐状态得到。视频里的句子是 “we see the sky”，输出 “vediamo il cielo [STOP]”。

![循环神经网络五种输入输出结构示意图（一对一、一对多、多对一、多对多）](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0001_00-00-49.jpg)

![完整的RNN编码器-解码器架构图，展示编码器生成上下文向量c并传递给解码器初始状态s0的过程](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0003_00-05-58.jpg)

## 瓶颈：所有信息压进一个定长向量 [07:15 - 08:25]

$c$ 的长度在设定网络规模时就固定了，比如 128 或 1024 个浮点数。四个词的短句压进去也许没问题，但如果要翻译一整段、一整本书，甚至整个语料库，要求网络把全部信息塞进同一个向量就不合理了。问题不在编码器不够强，而在于输入与输出之间只有这一条通道。视频用一句话点出症结：input sequence bottlenecks through fixed sized $c$，并追问如果 $T=1000$ 会怎样。

![完整的RNN编码器-解码器架构图，展示解码器逐步生成完整输出序列，并标注了固定大小上下文向量的瓶颈问题](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0004_00-07-23.jpg)

## 注意力：让解码器每一步回看输入 [08:25 - 15:19]

解决思路很直接：不再让输入序列经过单个向量，而是在解码器每一个时间步都允许它回看整个输入序列，并当场生成一个新的 context vector。

第一步，计算 **alignment score**。用一个线性层 $f_{att}$ 把上一个解码器状态 $s_{t-1}$ 与第 $i$ 个编码器状态 $h_i$ 拼接后压成标量：

$$
e_{t,i} = f_{att}(s_{t-1}, h_i)
$$

它衡量“当前要生成的东西”与“输入第 $i$ 个词”有多匹配。$f_{att}$ 就是普通的线性层，和其他参数一起用梯度下降学习。

第二步，对这些无界标量做 **softmax**，得到注意力权重 $a_{t,i}$，满足 $0<a_{t,i}<1$ 且 $\sum_i a_{t,i}=1$。它相当于“给定当前解码状态，在输入词上的一个概率分布”。

第三步，用这组权重对编码器隐状态做加权求和：

$$
c_t=\sum_i a_{t,i}h_i
$$

再把它送进解码器，$s_t = g_U(y_{t-1}, s_{t-1}, c_t)$。解码器结构本身没有变，变的只是 $c$ 从固定向量换成了每个时间步现算的 $c_t$。

最关键的性质是：整条链路都可微，注意力权重没有任何直接监督。没有人告诉网络“vediamo 对应 we see”，训练信号仍然只是输出序列上的 softmax 与交叉熵；网络在学会预测下一个词的过程中，顺带学会了该把注意力放到哪里。视频给的直觉是：生成 “vediamo” 时，权重可能集中在 “we” 和 “see”（例如各 0.45），而 “the” 和 “sky” 只有 0.05。

![基于RNN的序列到序列模型架构图，展示了编码器、解码器以及上下文向量的传递过程](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0005_00-08-25.jpg)

![带有注意力机制的Seq2Seq模型架构图，展示了对齐分数的计算与归一化过程](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0006_00-11-05.jpg)

![带注意力机制的RNN序列到序列模型结构图，展示对齐分数计算、softmax归一化与上下文向量生成](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0007_00-14-47.jpg)

## 逐步迭代、初始化与注意力可视化 [15:19 - 25:53]

第二步及以后完全重复同一套流程：用 $s_1$ 重新计算对齐分数 $e_{2,i}$ 与权重 $a_{2,i}$，得到 $c_2$，再生成下一个词。同一个 $f_{att}$ 在每个时间步复用，不同之处只在 query 换成了最新的解码状态。这样就彻底避开了固定长度瓶颈：输入不再被压成一个向量，而是在每个输出步被重新摘要一次。

课堂上有两个值得记下的问题。其一是“解码器怎么初始化”，讲者提醒这个词有歧义：一是指网络权重，按惯例随机初始化再训练；二是指处理序列时的初始隐状态 $s_0$，可以取编码器最后隐状态、用一个学出来的线性投影，或者直接置零——三种都行，前提是训练时始终使用同一种约定。其二是“语法是谁教的”，答案是没有谁教：我们只提供了“输出可以回看输入”的机制，语法结构是网络在大量平行语料上自己学出来的。

把注意力权重画成矩阵就能看到这件事。横轴是英文输入，纵轴是法文输出，每一行是生成该词时在整个输入上的分布。前半段和后半段出现清晰的对角线，说明那些词一一对应且顺序一致；中间 “European Economic Area” 附近出现反向对角线和 2×2 小块，说明那里语序被打乱，并且存在两个词对两个词的合并。最初把这套机制用于机器翻译的是 Bahdanau 等人的 *Neural Machine Translation by Jointly Learning to Align and Translate*（ICLR 2015），讲者提到它刚在 ICLR 2025 获得 test of time 的 runner-up 奖。

![注意力机制解码过程示意图，展示如何用前一步状态计算新的对齐分数与上下文向量](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0008_00-18-01.jpg)

![英法翻译示例及注意力权重热力图可视化](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0009_00-22-10.jpg)

## 把 attention 从 RNN 里剥出来 [25:53 - 31:37]

到这里 attention 还长在 RNN 上，但它其实是一个更一般的算子。把角色重新命名：编码器隐状态是 **data vectors**，即“要被总结的数据”；解码器隐状态是 **query vectors**，即“要为它产出结果的问题”；算子的输出是 context vector。对每一个 query，都回到全部 data vectors 上，按相似度做一次定制化的摘要，得到一个输出向量。

“query 从哪来”对 attention 层并不重要：在 RNN 里它来自上一个隐状态，但算子本身只要求输入一组 query 和一组 data。这正是把 attention 单独拿出来当积木用的起点。

![带注意力机制的Seq2Seq模型完整架构图，包含编码器、解码器及注意力计算流程](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0010_00-25-58.jpg)

![注意力机制中Query、Data、Output向量的抽象关系示意图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0011_00-27-07.jpg)

## Attention Layer 的形式化 [31:37 - 34:36]

写成层的形式：输入一个 query $q\in\mathbb{R}^{D_Q}$ 和 $N_X$ 个 data vectors $X\in\mathbb{R}^{N_X\times D_X}$，计算

$$
e_i=f_{att}(q,X_i),\qquad a=\mathrm{softmax}(e),\qquad y=\sum_i a_i X_i
$$

输出维度是 $D_X$。

第一处泛化是把 $f_{att}$ 换成最简单的相似度：**点积** $e_i=q\cdot X_i$。代价是两个向量维度必须一致，也就是此后 $D_Q=D_X$（后面的 key/value 投影会解除这个限制）。第二处是缩放：点积的量级随维度增长。若 $q$ 与 $X_i$ 都是全 1 向量，模长是 $\sqrt{D}$，内积就是 $D$；维度一大，softmax 的输入被拉得很开，输出接近 one-hot，梯度趋近于零。所以实际使用的是 **scaled dot-product**：

$$
e_i=\frac{q\cdot X_i}{\sqrt{D_X}}
$$

这不是数学上的必需，而是为了让不同维度的模型都有正常的梯度流；随着模型变大、向量维度变高，这一点会越来越重要。

![注意力层（Attention Layer）的计算流程图，展示查询向量与数据向量生成注意力权重并加权求和得到输出](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0012_00-31-47.jpg)

![注意力层流程图，重点标注相似度计算改为点积 e_i = q · X_i](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0013_00-32-13.jpg)

## 从单个 query 到矩阵、再到 key/value [34:36 - 44:47]

实际中我们总是有多个 query。把 $q$ 换成矩阵 $Q\in\mathbb{R}^{N_Q\times D_Q}$，一次算出所有 query 与所有 data 的相似度：

$$
E=\frac{QX^{\top}}{\sqrt{D_X}}\in\mathbb{R}^{N_Q\times N_X},\qquad
A=\mathrm{softmax}(E,\dim=1),\qquad
Y=AX
$$

softmax 只在 data 这一维上做，所以 $A$ 的每一行是一个 query 在全部 data 上的分布。两次矩阵乘法就完成了整批计算——矩阵乘法的两种读法正好对应“算所有点积”和“按权重做线性组合”。

再往下走一步：$X$ 在这套计算里出现了两次，一次用来和 query 比相似度，一次用来被加权求和。这两种用途其实不一样，于是引入两组可学习投影，把每个 data vector 变成 **key** 和 **value**：

$$
K=XW_K,\qquad V=XW_V
$$

key 用来和 query 算相似度（$W_K$ 把 $D_X$ 投到 $D_Q$），value 用来做加权和（$W_V$ 把 $D_X$ 投到 $D_V$，$D_V$ 可以和 $D_Q$ 不同）。最终

$$
E=\frac{QK^{\top}}{\sqrt{D_Q}},\qquad
A=\mathrm{softmax}(E,\dim=1),\qquad
Y=AV
$$

讲者用搜索引擎打比方：你输入的 query 是“世界上最好的学校”，后端拿它去匹配网页的 key，而你真正想拿回的 value 是“Stanford”——要找的东西和要得到的东西本来就是两回事。key/value 的投影矩阵也不由人指定，同样由梯度下降学出来；可以把它们理解成对原始 data vector 的两种“过滤”。

## Cross-attention 与 self-attention [44:47 - 50:40]

至此 attention 已经和 RNN 无关，是一个有两组输入、两组权重的独立层，通常叫 **cross-attention**：query 与 data 来自不同来源，数量也可以不同。翻译（源语言与目标语言）和图像描述（图像块与生成词）都属于这种情形。

更常见的是 **self-attention**：只有一组输入向量，每个向量同时投影成 query、key、value 三个角色，

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

其余计算完全一样。$D_{in}$、$D_{out}$ 是层的超参数，原则上可以不同，实践中几乎总是相等。三个投影还可以拼成一个矩阵一次算完，因为硬件上“少数大矩阵乘”比“多个小矩阵乘”更高效。论文里的这种形式对应 decoder-only attention，也就是今天最常用的那种；它已经无法再放回开头那个 RNN 解码器里了。判断用哪种，只看问题里有没有“两类东西”：有就 cross，没有就让一组向量自己和自己比。

## 置换等变、位置编码、mask 与多头 [50:50 - 58:21]

self-attention 有一个容易被忽略的性质。把输入向量的顺序打乱，$Q$、$K$、$V$ 只是跟着打乱，相似度、softmax、加权和也都只是跟着打乱，输出集合不变。这就是 **permutation equivariance**：

$$
F(\sigma(X))=\sigma(F(X))
$$

换句话说，self-attention 本质上作用在**集合**上，而不是序列上；它把向量打包进矩阵时用到了顺序，但计算本身不依赖顺序。

如果任务需要顺序，就往每个输入向量上拼一段 **positional embedding**，告诉网络“这是第 1 个、第 2 个……”。（2026 课件补充了更现代的 RoPE，见文末。）

第二个变体是 **masked self-attention**：在相似度矩阵上把不允许看的格子改成 $-\infty$，softmax 之后这些位置正好是 0，输出就不会包含对应 value 的信息。语言建模必须这样做——预测 “attention is very” 的下一个词时，不能让它偷看 “cool”。

第三个是 **multi-head self-attention**：并行跑 $H$ 份独立的 self-attention，每份有自己的权重（初始化不同，架构完全相同），输出拼起来再过一层线性投影融合。多头不改变计算形式，只是给模型更多容量和更多种“看的角度”。实现上不需要 for 循环，用 **batched matmul** 一次算完；整个 self-attention 说到底就是四次矩阵乘法（QKV 投影、QK 相似度、V 加权、输出投影），这正是它能被大规模分布式训练的原因。

## 三种处理序列的原语 [58:21 - 62:40]

课程到这里已经给出三种基本的序列处理方式，可以放在一起比较：

- **RNN** 作用于有序一维序列，计算与显存都是 $O(N)$，但隐状态必须逐个计算，无法在序列方向上并行，很难做大。
- **卷积** 作用于 $N$ 维网格，局部混合信息，每个位置可以并行；但要让一个输出看到很远的地方，只能堆很多层或使用很大的卷积核。
- **self-attention** 作用于向量集合，一层之内每个向量都能看到所有向量，天然适合长序列，而且只是四次矩阵乘；代价是 $O(N^2)$ 的计算与显存。

并行性为什么重要？因为单核频率已经接近物理极限，过去二十年算力的增长主要来自“把更多处理器一起用”。一个算法如果能轻松摊到成百上千个处理器上，就能随硬件规模一起变大。至于 $O(N^2)$，讲者的态度是“不一定坏”：更多计算也意味着网络在输入上做了更多处理，只要肯买更多 GPU 就可以换。

## Transformer 与 ViT [62:40 - 66:25]

**Transformer** 就是把 self-attention 放在核心的架构。一个 **transformer block** 的顺序是：输入一组向量 → multi-head self-attention（向量之间通信）→ 残差 → LayerNorm → MLP/FFN（对每个向量独立作用的两层网络）→ 残差 → LayerNorm。self-attention 负责“互相看”，MLP 负责“各自想”。整个 transformer 就是把同样的 block 堆很多层。

这套结构从 2017 年提出后基本没变，只是规模变了：原始版本约 12 个 block、2 亿参数，今天则是数百个 block、万亿参数。它既能做语言，也能做图像：把图片切成 patch，每个 patch 投影成一个向量送进 transformer，每个 patch 得到一个输出向量；做分类时对所有输出做 pooling，再接一层线性分类器，这就是 **ViT** 的骨架。讲者最后留了一句：2017 年之后还有一些小改动，时间不够，留作课外阅读——这些改动在 2026 课件里已经补上。

## 2026 课件更新

以上正文以 2025 春季视频为准。对照 2026 春季课件，主要有这些差异：

- **行政信息前置**：2026 课件以行政页开场——project proposal 当天截止、Assignment 2 发布（5/7 截止）、次日 4/24 有 PyTorch review session。2025 视频直接从上一讲回顾开始。
- **位置编码从一句话变成一页**：视频只说“拼一个 positional embedding”；2026 课件单独讲 **RoPE**：把位置映射为角度并旋转 query 和 key，使内积只依赖相对位置，$(R(\theta_i)q_i)^{\top}(R(\phi_j)k_j)=q_i^{\top}R(\phi_j-\theta_i)k_j$，引用 RoFormer（2021）。
- **缩放点积给出完整推导**：2026 课件用 $a\cdot b=|a||b|\cos\theta$ 与常量向量模长 $\sqrt{D}$ 说明“相似度量级随维度增长 → softmax 饱和 → 梯度消失”，与视频的口头解释一致，但写成了公式。
- **复杂度单独成节**：2026 课件明确 self-attention 是四次矩阵乘，计算 $O(N^2)$、显存 $O(N^2)$；当 $N=100K$、$H=64$ 时注意力矩阵需要 1.192 TB，**FlashAttention** 通过不落地完整注意力矩阵把显存降到 $O(N)$。视频只提到 $O(N^2)$，没有展开显存优化。
- **多头写法形式化**：2026 课件给出 $W_Q,W_K,W_V:[D\times HD_H]$、$W_O:[HD_H\times D]$、$D_H=D/H$，Q/K/V 形状为 $[H\times N\times D_H]$，softmax 在 $\dim=2$ 上做，并说明 H 个头用 batched matmul 并行计算。
- **新增 LLM 一节**：2026 课件用三页讲 “Transformers for Language Modeling”：词表 $V$、模型维度 $D$ 的 embedding 查表 $[V\times D]$；每个 block 内用 masked attention；末端 $[D\times V]$ 投影到词表分数，用 softmax + 交叉熵预测下一个 token。视频只在 mask 部分提了一句语言建模。
- **ViT 细节补齐**：2026 课件写明 224×224×3 图像切成 16×16×3 的 patch、flatten 后经 768→$D$ 线性变换，并指出这等价于 16×16、stride 16、输出通道 $D$ 的卷积；用二维位置编码，不做 mask，每个 patch 可看所有 patch；最后 average pool 成 1×$D$，再经 $D\to C$ 线性层输出类别分数。
- **规模数字具体化**：2026 课件列出原始 transformer（12 blocks、D=1024、H=16、N=512、213M 参数）、GPT-2（48 blocks、D=1600、H=25、N=1024、1.5B）、GPT-3（96 blocks、D=12288、H=96、N=2048、175B）。视频只说“数百 block、万亿参数”。
- **“小改动”整理成章**：视频说时间不够、留作课外阅读，2026 课件补成 “Tweaking Transformers” 一节：**Pre-Norm**（LayerNorm 移到残差内部，训练更稳）、**QK-Norm**（对 Q/K 做 RMSNorm，抑制梯度尖峰）、**SwiGLU**（$Y=(\sigma(XW_1)\odot XW_2)W_3$，$H=8D/3$ 保持参数量）、**MoE**（每 block 学 $E$ 组 MLP 权重，每 token 只路由到 $A<E$ 个专家，参数增 $E$ 倍、计算只增 $A$ 倍；例子是 Gemma4 26B-A4B（4/2/2026），1 个 shared expert + 128 个 routed experts，每 token 选 8/128，总参数 26B、激活 4B）。
- **LayerNorm 与 block 计算量**：2026 课件插入 “Recall Layer Normalization”，给出 $\mu_i$、$\sigma_i$、$z_i=(h_i-\mu_i)/\sigma_i$、$y_i=\gamma z_i+\beta$（Ba et al. 2016）；并注明一个 block 的计算主要是 6 个 matmul（self-attention 4 个 + MLP 2 个），MLP 经典设置是 $D\to 4D\to D$。

## 要点回顾

- **seq2seq**：RNN 编码器把输入压成 context vector $c$（常取 $h_T$），解码器按 $s_t=g_U(y_{t-1},s_{t-1},c)$ 逐词生成。
- 固定长度的 $c$ 是瓶颈，序列越长，信息越挤。
- **attention** 在每个解码步计算 $e_{t,i}=f_{att}(s_{t-1},h_i)$，softmax 得权重，$c_t=\sum_i a_{t,i}h_i$；全程可微，对齐关系无需监督。
- 注意力矩阵的对角线表示顺序对应，反向对角线与 2×2 小块表示语序变化和多对多对齐；机制源自 Bahdanau et al. 2015。
- attention 可抽象为 query/data → output 的算子，与 RNN 解耦。
- 点积是最简单的相似度，除以 $\sqrt{D}$ 是为了防止 softmax 饱和与梯度消失。
- 多 query 用两次矩阵乘一次算完；把 data 投影为 **key** 与 **value**，分离“匹配”和“取值”。
- **self-attention** 中 Q/K/V 都来自同一组输入，**cross-attention** 有两组来源；前者常用于单模态，后者用于跨模态。
- self-attention 置换等变，作用在集合上，需要位置编码补顺序；mask 阻止偷看未来；多头增加容量。
- 三种原语：RNN 顺序不可并行、卷积局部可并行、self-attention 全局但 $O(N^2)$。
- **transformer block** = self-attention + 残差 + LayerNorm + MLP，堆叠成 transformer；ViT 把图像 patch 当作 token。
- 2026 课件补充了 RoPE、FlashAttention、LLM 与 ViT 细节，以及 Pre-Norm / QK-Norm / SwiGLU / MoE 等常见改动。

## 视频来源

https://www.youtube.com/watch?v=RQowiOF_FvQ
