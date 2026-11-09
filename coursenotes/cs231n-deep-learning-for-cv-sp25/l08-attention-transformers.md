---
title: "第 8 讲 · 注意力机制与 Transformer"
description: "Stanford CS231n Spring 2025 L08 中文视频笔记（2026-11 重写版）"
---

# CS231n Spring 2025 · Lecture 8 注意力与 Transformer：从 RNN 瓶颈到「Attention is All You Need」

**核心结论**：本讲从 RNN 的 Seq2Seq 机器翻译讲起，指出把整个输入序列压进单个固定长度的 context vector 会成为信息瓶颈；注意力机制让解码器在每一步都能回看全部输入、按相似度做加权汇总，且全程可微、无需对齐标注、端到端可学。把这套机制从 RNN 中剥离出来，就得到一个以 Query、Key、Value 为核心、本质只是几次矩阵乘的通用算子——它操作的是无序向量集合（置换等变），需靠位置编码补充顺序，也可用掩码和多头加以约束与增强。以 self-attention 为核心，配上残差连接、Layer Norm 与逐向量 MLP 便组成 Transformer Block；相比 RNN 与卷积，它高度并行、天然适配长序列，代价是 $O(N^2)$ 计算。如今 Transformer 已是几乎所有大规模深度学习应用的默认架构（Vaswani et al., NeurIPS 2017），本讲正是沿着「修 RNN 的瓶颈」这条历史线索把它一步步推演出来。

## 1. 开篇定位：从 RNN 到注意力与 Transformer [00:04 - 02:55]

上一讲的循环神经网络（RNN）是一类专为**处理序列**而生的架构，它让网络跳出以往「one-to-one」的范式（输入一张图、输出一个分类），得以处理 one-to-many（如图像描述：输入图像、输出一串词）、many-to-one（输入若干帧、输出一个分类）等新问题。本讲要在此基础上再引入两样东西：其一是 **attention（注意力）**，一种 fundamentally 操作**向量集合（sets of vectors）**的全新神经网络原语；其二是 **transformer**，一种把 self-attention 放在核心位置的神经网络架构。

![注意力机制与Transformer架构的对比及定义幻灯片](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0001_00-02-13.jpg)
这一页开宗明义：Attention 是「作用于向量集合的新原语」，Transformer 是「处处使用注意力的架构」，并提示「Transformers are used everywhere today」，但「they developed as an offshoot of RNNs」，所以要从 RNN 讲起。

讲者先给出结论：transformer 如今几乎是所有问题的架构——无论是图像分类、图像生成、文本生成、文本分类还是音频，凡是规模大、达到 SOTA、用大量数据训练、由大公司部署的模型，今天几乎都是 transformer。但它并非一夜之间降临的剧变：self-attention、attention 这些想法在领域里已存在数年，而且恰恰是从 RNN 里长出来的分支。因此本讲刻意**复现历史发展顺序**，从上一讲的 RNN 出发来引出注意力。

## 2. Seq2Seq：RNN 编码器-解码器与固定瓶颈 [02:55 - 09:23]

作为动机问题，考虑**序列到序列（Seq2Seq）机器翻译**：输入一串英文词，输出另一串意大利语词。这里不能假设两种语言的词有任何对应——词数可能不同、语序可能完全不同，正是序列处理架构的用武之地。用 RNN 做 Seq2Seq 可追溯到 2014 年（Sutskever et al., NeurIPS 2014），而用 RNN 处理序列的历史还要再早十几年。

基本架构是「编码器 + 解码器」两个 RNN。编码器把输入序列递归地吞进去：

![RNN编码器结构图，展示输入序列到隐藏状态的映射及机器翻译示例](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0002_00-02-57.jpg)
图中给出编码器的递推式 $h_t = f_W(x_t, h_{t-1})$：每个时间步把当前输入 $x_t$ 与上一隐藏状态 $h_{t-1}$ 送进同一个 RNN 单元，产出新隐藏状态 $h_t$，沿时间展开即可处理任意长度序列。

以短句「we see the sky」为例，每个词占一个 tick，处理完全部词后，用最后一个隐藏状态把整句内容概括成一个向量——**context vector（上下文向量）$c$**。由于循环结构，这个末状态被认为「编码」了整个输入序列的信息。随后用架构相同、但权重 $U$ 不同的第二个 RNN 作解码器，逐词生成输出：

![Seq2Seq完整模型图，包含编码器、上下文向量及解码器生成过程](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0003_00-08-28.jpg)
图中给出解码器递推式 $s_t = g_U(y_{t-1}, s_{t-1}, c)$：每一步接收三个输入——上一时刻的输出 token $y_{t-1}$、解码器上一隐藏状态 $s_{t-1}$、以及概括全句的上下文向量 $c$，再输出新隐藏状态与一个新词（课件上的意大利语词，讲者未拼读）。

关键在于一个**通信瓶颈**：输入序列与输出序列之间唯一的通道就是那个 context vector $c$。它是**固定长度**的（比如 128 或 1024 个浮点数），其大小在我们设定 RNN 规模时就定死了，不随输入／输出序列变长而增长。翻译「we see the sky」这种短句，把所需信息塞进 1024 维向量还算合理；可一旦要翻译一整段、一整本书、乃至整个语料，再要求网络把全部输入压进单一固定向量就不合情理了。

于是解决方案的直觉很直接：**不再把网络逼进单一固定长度向量的瓶颈**，而是在解码的每一步都让模型回看整个输入序列，即时地为这一步重新生成一个上下文向量。这样没有瓶颈、能扩展到长得多的序列——讲者说，一个版本的故事里，注意力与 transformer 这一整套东西，正是从「试图解决 RNN 的这个瓶颈」开始的。

## 3. 带注意力的 Seq2Seq：对齐分数、Softmax 与端到端学习 [09:23 - 17:57]

编码器保持不变，解码器仍需要一个初始状态 $s_0$。改变发生在「怎么算 context」：拿到 $s_0$ 后，先让它回看输入序列，为每个输入位置计算一个**对齐分数（alignment score）**，衡量「这个解码状态与该输入词有多匹配」。输入有四个词，就算出四个标量。最简单的实现是一个线性层 $f_{att}$：把解码隐藏状态与某个编码器隐藏状态**拼接**成一个向量，再用线性变换压成一个标量——它只是个线性算子，可以放进计算图、随其它参数一起用梯度下降学出来。

![RNN与注意力机制的Seq2Seq模型架构图，展示对齐分数计算、Softmax归一化及上下文向量生成过程](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0004_00-14-25.jpg)
这一页把带注意力的 Seq2Seq 写成三条式子：$e_{t,i} = f_{att}(s_{t-1}, h_i)$（对齐分数）、$c_t = \sum_i a_{t,i}h_i$（上下文向量）、$s_t = g_U(y_{t-1}, s_{t-1}, c_t)$（解码），并标注出处为 Bahdanau et al., ICLR 2015。

这些对齐分数是 $(-\infty, +\infty)$ 的无界实数，为避免数值爆炸，要过一次 **softmax**：把任意实数向量变成「每项在 0~1 之间、且总和为 1」的离散概率分布。于是可以理解为「给定当前解码状态，在输入各词上的一个分布」。再用这些注意力权重 $a_{t,i}$ 对编码器隐藏状态 $h_i$ 做**线性组合**，得到新的上下文向量 $c_t = \sum_i a_{t,i}h_i$。此后解码器结构完全没变，只是 $c$ 换了一种算法：把 $c_t$ 与输出序列首 token $y_0$ 一起送进 RNN 单元，得到下一隐藏状态与第一个输出词。

![RNN与注意力机制的Seq2Seq架构示意图，展示编码器状态、Softmax权重计算及解码器上下文向量生成过程](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0006_00-19-57.jpg)
画面用一组具体权重演示这种分布：例如 $a_{21}=a_{22}=0.05$、$a_{24}=0.1$、$a_{23}=0.8$，即网络把绝大部分注意力压在某一个输入词上。直觉上，要产出对应「we see」的意大利语词时，网络应给这两个词高权重、而几乎不管「sky」，因为后者对当前要预测的词无关紧要。

![注意力机制架构图的更新版本，右下角增加了关于全微分性和反向传播的关键说明](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0005_00-17-09.jpg)
更新版架构图右下角特意加了一句关键说明：「All differentiable! No supervision on attention weights. Backprop through everything」——全流程可微。

这一点最为要紧：我们**不监督**注意力权重，不必告诉网络「哪个输入词对应哪个输出词」。整个系统是一张由可微运算拼成的计算图，只在输出端施加「预测输出序列 token」的交叉熵 softmax 损失，网络在努力预测正确词的过程中，会自己学会该去看输入的哪些部分。若反过来要求人工标注对齐关系，这类训练数据几乎无法获得。讲者还顺带澄清「初始化」一词的两个含义：一是解码器**权重**通常随机初始化再靠梯度下降优化；二是处理序列时要设的**初始隐藏状态** $s_0$，可用编码器末隐藏状态、一个学到的线性投影、或干脆全零，只要训练时让网络适应即可。学生问到否定与异或（negations / xors）会不会带来问题，讲者承认这是难题，需要大量数据与算力才有望解开。

## 4. 迭代解码与注意力权重的可视化解释 [17:57 - 25:53]

上面只走了解码器的一个 tick，实际每个时间步都要**把整套流程重做一遍**：用新得到的 $s_1$ 再去和所有编码器隐藏状态算相似度（用同一个 $f_{att}$ 线性投影）、过 softmax 得到这一时刻的新分布、加权得到新的上下文向量 $c_2$、再跑一步解码器得到 $s_2$ 与下一个词（课件上是「il」），如此循环。至此瓶颈被真正解决：解码每一步都回看全部输入、即时重算上下文，这个「在输出的每一刻看向输入的不同部分」的机制就叫做 **attention**。

注意力权重还附带一个好处：它能让我们**反推网络在看哪里**。把解码每一步对输入序列预测的概率分布画成二维网格，就能看到一张对齐热力图。

![英法机器翻译实例及注意力权重热力图可视化，展示输入输出序列与对齐关系](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0007_00-22-10.jpg)
例子是英译法，输入为「The agreement on the European Economic Area was signed in August 1992.」，顶部是输入的英文词、逐行是输出的法语词，格子里是注意力权重 $a_{t,i}$。

观察这张矩阵能读出不少结构：第一行（预测第一个法语词）几乎把全部概率质量压在英文「the」上，第二行压在「agreement」上，说明网络确实自己找到了输入词与输出词的对齐。**对角结构**意味着输入与输出词按顺序一一对应——句首前四个词、句尾的「August 1992」都呈对角。中段则出现**反向对角／非对角**结构，对应英语「European Economic Area」与法语词序不同；还有一处 2×2 小格，对应「两个法语词对应两个英语词、没完全一一拆解」的情形。

值得强调的是：讲者从未教过网络任何语法，只是用大量「英文输入／法文输出」配对、加上一个设计者认为可能有用的机制，让网络通过梯度下降自己决定怎么用它。学生问「它怎么学会语法」，讲者笑称「那是深度学习的奥秘」。这也是注意力在机器学习中的最初用途——来自 Bahdanau et al. 2015 的《Neural Machine Translation by Jointly Learning to Align and Translate》，讲者提到这篇论文刚获得 **ICLR 2025 的 runner-up test of time award**，可见影响之深远。

## 5. 把注意力从 RNN 中抽离成独立算子 [25:53 - 31:36]

但这里其实藏着一个比「修 RNN」更一般的想法：我们为修补 RNN 而发明的这套机制，本身就是一个通用且强大的算子，值得把它与 RNN **剥离（divorce attention from RNN）**，成为可独立插进任意架构的计算原语。为此先把里面流动的东西重新命名、抽象成三类向量：

- **data vectors（数据向量）**：希望被总结的数据，即编码器隐藏状态序列；
- **query vectors（查询向量）**：为产生某个输出而服务的向量，即解码器隐藏状态；
- **output / context vectors（上下文向量）**：注意力算子的输出，原先喂给 RNN 下一步。

![Seq2Seq完整流程图解，区分Query向量（解码器状态）与Data向量（编码器状态），展示上下文向量的生成](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0008_00-27-43.jpg)
这张流程图刻意把 Query 向量（解码器状态，绿色）与 Data 向量（编码器状态）分开，强调每个 query 都会去关注所有 data vectors、并产出一个 output vector。

![注意力层基础结构图，展示查询向量与数据向量的交互流程](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0009_00-29-56.jpg)
注意力层的基础结构图标出输入为「Query vector: $q$ $[D_Q]$」，展示从输入算相似度、softmax、加权到输出 $c_1$ 的完整路径。

站在算子本身的角度：先对某个 query 与所有 data vectors 计算相似度，再用 softmax 把相似度压成针对该 query 的注意力分布，最后用这组权重对 data vectors 做线性组合得到输出向量；下一时刻换一个新 query（它从哪来，算子并不关心），重复同样三步即可。这就是 attention 算子的全部核心。相似度函数 $f_{att}$ 原则上可以是任意「输入两向量、输出一个标量」的函数，但接下来反而要把它**简化**。

## 6. 缩放点积注意力与矩阵化计算 [31:36 - 38:15]

第一个泛化是把相似度函数简化到极简：输入两向量、输出标量，最简单的函数就是**点积（dot product）**，而实践证明点积已足够好用。但点积与 softmax 之间有个微妙的相互作用：当向量维度变大时（讲者用「10 维全 1 向量」对比「100 维全 1 向量」），点积之和随之变大，softmax 里被更大的数值相除，概率分布被压得极尖，进而导致**梯度消失**、使整个网络学不动。

![引入缩放点积注意力的推导过程及公式定义](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0010_00-34-22.jpg)
公式为 $e_i = q \cdot X_i / \sqrt{D_Q}$，画面解释了为何要除以 $\sqrt{D_Q}$：防止 softmax 饱和、避免梯度消失。

解决办法是把点积按向量维度的平方根缩小，即 **scaled dot-product（缩放点积）** $e_i = q \cdot X_i / \sqrt{D_Q}$。这既让梯度在不同维度下都能顺畅回传，也让架构能随向量维度放大而扩展——因为网络越做越大时我们总想要更高维的向量（更多算力、更大容量），架构必须能随之 scale，这一点非常关键。

第二个泛化是**一次处理一组 query vectors**，而不必一个一个来。设 $Q$ 为 $N_Q \times D_Q$（$N_Q$ 个查询、每个 $D_Q$ 维），data 矩阵 $X$ 为 $N_X \times D_Q$（点积要求维度匹配，这一限制后面会被解除）。要算「所有 query 与所有 data 之间」的全部成对相似度，最自然的方式就是一次矩阵乘——因为矩阵乘的每个元素恰好是一行与一列的内积：

![多头注意力的矩阵化计算公式及维度变化说明](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0011_00-35-01.jpg)
矩阵化公式为 $E = QX^T / \sqrt{D_Q}$，相似度矩阵形状为 $[N_Q \times N_X]$，把单向量计算推广到批量。

得到 $N_Q \times N_X$ 的相似度矩阵 $E$ 后，沿其中一个轴（dim=1）逐行做 softmax，得到每个 query 对 data 的注意力权重矩阵 $A$；输出则再用一次矩阵乘 $Y = AV$ 完成——这里用到矩阵乘的第二种解读：「以另一矩阵的值为权重，对若干列做线性组合」。于是原本要反复循环的计算，被压缩成几次矩阵乘就全部并行算完。

## 7. 分离 Key/Value：Cross-Attention 与 Self-Attention 层 [38:15 - 50:41]

第三个泛化注意到：data 矩阵 $X$ 在计算里出现了两次——一次用来与 query 算相似度、一次用来做线性组合算输出。同一份数据被复用在两种语境下有点别扭，于是把这两种用途**拆开**，让网络自己学出两种投影。为每个 data 向量投影出两个向量：

$$K = XW_K,\quad V = XW_V$$

- **key（键）**：$W_K$ 把 $D_X$ 维投影到 $D_Q$ 维，专门用来与 query 比对、算对齐分数（维度必须与 query 一致）；
- **value（值）**：$W_V$ 把 $D_X$ 投影到 $D_V$ 维，专门用来做线性组合、生成层输出（$D_V$ 原则上可与 $D_Q$ 不同）。

$W_K$、$W_V$ 都是可学习参数，与其余部分一起由梯度下降学出，我们并不告诉它该怎么拆；可以把它理解为对数据向量做两种不同的「过滤」。讲者用**搜索引擎**打比方：你在 Google / ChatGPT 输入「什么是世界上最好的学校」，这是 query，需要去和后端大量字符串（keys）匹配；而你真正想取回的 value 是「Stanford」——你要找的东西与拿回的东西并不相同，正对应 key 与 value 的分离。

![Attention Layer 结构图，展示输入、计算步骤及缩放点积注意力的变化](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0012_00-39-13.jpg)
这一页汇总了 Attention Layer 的全部改动（缩放点积、多个 query、分离 key/value），并给出完整公式 $K = XW_K,\ V = XW_V,\ E = QK^T/\sqrt{D_Q},\ A = \text{softmax}(E,\ \text{dim}=1),\ Y = AV$。

到这里 RNN 被彻底扔掉：这是一个能独立站立的神经网络层，接收两组输入（query vectors 与 data vectors）、带两组可学习权重、输出一组向量，可插进架构里的任意位置。因为两组输入可能来自不同来源、数量也不同，它叫 **cross-attention layer（交叉注意力）**——在两组不同的东西之间「交叉」读取信息。

![Cross-Attention Layer 结构图，展示 Query、Key、Value 矩阵运算生成输出的过程](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0013_00-45-18.jpg)
Cross-Attention Layer 结构图标出 Softmax（沿列）与 Product / Sum（沿行）的维度方向，以及「每个 query 产出一个 output」的机制。

而更常用的变体是 **self-attention layer（自注意力）**：只有一组输入向量 $X$，输出同数量的 $Y$，不再区分 data 与 query，而是把每个输入向量分别投影成 query、key、value 三份：

![Self-Attention Layer 完整计算流程与矩阵维度定义图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0014_00-45-49.jpg)
Self-Attention Layer 完整计算图给出 $Q = XW_Q,\ K = XW_K,\ V = XW_V,\ E = QK^T/\sqrt{D_Q},\ A = \text{softmax}(E,\ \text{dim}=1),\ Y = AV$，数据流维度为 $[N \times D_{in}] \to [N \times D_{out}]$。

三个可学习权重矩阵 $W_Q, W_K, W_V$ 各司其职，后续计算与 cross-attention 完全相同。$D_{in}$、$D_{out}$ 是这一层的架构超参数（如同普通线性层的输入／输出维度），理论上可以不同，但讲者说实践中几乎总设为相同。工程上还有个常见 trick：硬件做「少而大的矩阵乘」比「多而小的」更高效，因此常把三个权重矩阵沿维度拼接，用**一次大矩阵乘**同时算出所有输入的 Q/K/V；这也就是当下最常使用的 decoder-only attention。至于用哪种：机器翻译、图像描述这类天然存在「两组不同东西要比较」的问题用 cross-attention；图像分类只有一幅图、需要让图像各部分互相比，则用 self-attention。把上述公式写成代码，形态大致是：

```python
def attention(Q, X, W_K, W_V, d_q):
    K = X @ W_K                       # 投影出 key
    V = X @ W_V                       # 投影出 value
    E = Q @ K.T / (d_q ** 0.5)        # 缩放点积相似度
    A = softmax(E, dim=1)             # 每个 query 上的分布
    Y = A @ V                         # 加权求和得到输出
    return Y
```

## 8. Self-Attention 的三个重要性质：置换等变、掩码与多头 [50:41 - 58:21]

**置换等变性（permutation equivariance）**。设想把输入顺序打乱（如 $X_3, X_1, X_2$）：Q/K/V 都是输入的线性投影，因此还是那批向量、只是被同样重排；点积相似度、softmax（它本就不关心输入次序）、线性组合也都只相应重排；最终输出 $Y$ 仍是同一批向量、按同样方式重排。

![演示输入序列置换（Permuting inputs）对 Self-Attention 结构影响的示意图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0015_00-50-56.jpg)
图中把底部输入改成 $X_3, X_1, X_2$ 来演示这种置换不变／等变的结构。

前几讲卷积具有平移等变性，这里 self-attention 则具有置换等变性。它的深层含义是：**这一层根本不关心输入顺序**，本质上操作的是**无序的向量集合（set）**而非序列——向量只是被打包进一个有序矩阵而已。若任务确实需要顺序信息，补救办法是给每个输入向量再拼接一段 **positional embedding（位置编码）**，告诉网络「这是第 1 个、第 2 个、第 3 个……」，实现方式有很多种。讲者特别澄清：这里讨论的是固定权重、只看前向计算的性质，并非训练收敛后的结果。

**masked self-attention（掩码自注意力）**。默认每个输入可以看全部输入；但有时想规定「谁能看谁」。做法是在对齐分数 $E$ 的屏蔽位置填入 $-\infty$，经 softmax 后这些位置变成 0，对应输出便不再依赖那些位置的 value。典型用途是**语言建模**：输入「attention is very」、输出「is very cool」，必须让第 1 个输出只依赖第 1 个词、第 2 个只依赖前两个词，防止网络「偷看」后面的词作弊。有了它，就能彻底不用 RNN、纯靠 self-attention block 完成上一讲同款的语言建模任务。

**multi-head self-attention（多头自注意力）**。并行跑 $H$ 个独立的 self-attention 副本：架构与计算完全相同，唯一区别是权重不同、且随机初始化不同，于是各头会学到略有不同的处理方式；把各头的输出 $Y$ 沿维度堆叠，再接一个线性投影把它们融合。这带来更多参数、更多 FLOPs、更大容量，深度学习里「越大越好」——实践中见到的 self-attention 几乎都是这个多头版本，而且用 batched matrix multiplies 就能把 $H$ 个头全部并行算完，无需 for 循环。由此也可看出，整个 self-attention 算子看似公式繁多，其实只是**四次大矩阵乘**：① 把输入投影成 Q/K/V，② 算 $QK^T$ 相似度，③ 用 softmax 权重对 V 加权求和，④ 跨头做输出投影。矩阵乘高度可并行、易在 GPU 与分布式环境扩展，正是它 scalable 的根本原因。

## 9. 三种序列处理方式对比与 Transformer Block [58:21 - 66:25]

到这里，本课已经见过三种处理序列的原语，讲者把它们并排比较（出自 Vaswani et al., NeurIPS 2017《Attention is All You Need》）：

![序列处理的三种方式对比：RNN、卷积与自注意力机制的优缺点分析](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0016_01-01-02.jpg)
这一页「Three Ways of Processing Sequences」对比 RNN、Convolution、Self-Attention：RNN「(-) Not parallelizable, need to compute hidden states sequentially」；Self-Attention「(+) Highly parallel, it's just 4 matmuls」「(-) Expensive: $O(N^2)$ compute, $O(N)$ memory for sequence of length N」。

- **RNN**：操作一维有序序列，功能强、曾被长期青睐，但因每个隐藏状态依赖前一个，本质是顺序算法、无法沿序列并行，因而很难做大、很难 scale。
- **Convolution（卷积）**：操作多维网格（图像是 2D，也可 1D/3D/4D），靠在网格上滑动卷积核来局部混合信息，各位置可并行计算、并行度很高；但要建立大感受野，要么用很大的核、要么堆很多层，这又重新引入了顺序性。
- **Self-Attention**：操作向量集合，天然适配长序列、没有 RNN 那样的瓶颈，而且**仅一层**之内每个向量就能看遍所有向量，高度并行（4 次矩阵乘）。唯一代价是对长度 $N$ 的序列需要 $O(N^2)$ 计算、$O(N)$ 内存，当 $N$ 到十万、百万、千万量级时非常昂贵——讲者调侃「解法就是多买 GPU」。

为什么并行性如此值钱？因为让**单个处理器变快**已逼近硬件极限、越来越难，但拿到**大量处理器**却很容易；过去二十年算力的增长，靠的正是那些能同时调动 10 个、100 个、1000 个乃至百万个处理器协同工作的算法。讲者还给出反直觉的一点：$O(N^2)$ 未必是坏事——对神经网络而言，更多计算意味着网络「思考」得更多、处理能力更强，只是更贵而已。结论是「Attention is all you need」：三者中仅用 attention 就能走得很远。

![Transformer Block介绍页：定义输入为向量集合x](/coursenotes/cs231n-deep-learning-for-cv-sp25/l08-attention-transformers/kf_0017_01-02-44.jpg)
「The Transformer / Transformer Block」页把输入定义为「Set of vectors x」，同样引 Vaswani et al., 2017。

**Transformer Block** 就是把 self-attention 放在核心、再包上几样标准件：输入是向量集合 $x$ → 过 self-attention（让所有向量互相交流、比较）→ 套**残差连接**（理由与 ResNet 相同）→ 过 **Layer Normalization**（与 CNN 中一样让训练更稳定）→ 接一个两层的 **MLP / FFN**，它对每个向量**独立**做计算 → 再套残差 + Layer Norm，框起来即一个 block。self-attention 负责向量之间、FFN 负责向量之内，二者互为补充；**Transformer 就是一串这样的 block**。

规模上，2017 年原始 transformer 约 12 个 block、2 亿参数；如今人们训练出上百个 block、万亿参数的模型，而架构八年来几乎没变。它既能做语言建模，也能直接用于图像：把图像切成若干 **patches**、每个 patch 单独投影成一个向量送入 transformer，每个 patch 得到一个输出向量；若要分类，就对全部输出向量做 pooling、再接一个线性层预测类别分数。讲者最后收束到开场的两个承诺：attention 是操作向量集合、高度并行／可扩展／灵活的新原语；transformer 以 self-attention 为主计算原语，是当今深度学习几乎每个应用都在用的架构，短期内看不到它退场。下一讲将转向检测、分割与可视化等新任务。

## 要点回顾

- 本讲沿历史线索展开：attention 与 transformer 并非凭空出现，而是从 RNN 的 Seq2Seq 机器翻译里为解决瓶颈而逐步长出来的分支。
- 经典 Seq2Seq（Sutskever et al., NeurIPS 2014）用编码器 $h_t = f_W(x_t, h_{t-1})$ 把整句压成单个固定长度 context vector，解码器 $s_t = g_U(y_{t-1}, s_{t-1}, c)$ 逐词生成；固定向量（128/1024 floats）无法承载长段落／整本书，是核心瓶颈。
- 注意力（Bahdanau et al., ICLR 2015，获 ICLR 2025 runner-up test of time award）让解码每步回看全部输入：对齐分数 $e_{t,i}=f_{att}(s_{t-1},h_i)$ → softmax 得分布 → $c_t=\sum_i a_{t,i}h_i$ → 解码 $s_t=g_U(y_{t-1},s_{t-1},c_t)$；全程可微、无需对齐监督、端到端由交叉熵损失驱动自学。
- 注意力权重可画成二维热力图（英法翻译例子），对角结构=词序一一对应，非对角／2×2 小块=语序差异或非一一对应；网络未被教过语法却自己发现了对齐。
- 把注意力从 RNN 剥离后得到独立算子：data vectors 被 query vectors 按相似度加权汇总成 output/context vectors；$f_{att}$ 可任意，实践中简化为点积。
- 缩放点积 $e_i = q\cdot X_i/\sqrt{D_Q}$ 用 $\sqrt{D_Q}$ 防止高维下 softmax 饱和与梯度消失；多查询向量用 $E=QX^T/\sqrt{D_Q}\ [N_Q\times N_X]$ 一次矩阵乘算全部相似度，softmax 后 $Y=AV$ 再乘一次得输出。
- 分离出 $K=XW_K$、$V=XW_V$（搜索引擎直觉：query「最好的学校」↔ value「Stanford」），得到输入两组向量的 cross-attention；把每个输入投影成 $Q=XW_Q,K=XW_K,V=XW_V$ 则是 self-attention，二者公式同为 $E=QK^T/\sqrt{D_Q},\ A=\text{softmax}(E,\text{dim}=1),\ Y=AV$。
- Self-Attention 具有置换等变性，本质是操作无序向量集合，需靠 positional embedding 补顺序；masked self-attention 用 $-\infty$ 屏蔽实现因果语言建模；multi-head 并行 $H$ 个不同权重的头再线性融合。整个算子只有四次 batched 矩阵乘，高度并行可扩展。
- 三种序列原语对比：RNN 顺序不可并行、卷积可并行但难建大感受野、self-attention 一层看遍全序列且高度并行，代价是 $O(N^2)$ 计算／$O(N)$ 内存；「Attention is All You Need」(Vaswani et al., NeurIPS 2017)。
- Transformer Block = self-attention + 残差 + Layer Norm + 逐向量 MLP/FFN + 残差 + Layer Norm；Transformer 是其堆叠。规模从 2017 年 12 blocks／2 亿参数到如今数百 blocks／万亿参数，架构基本未变；图像上切 patches 投影后送入，分类时 pooling + 线性层。

## 视频来源

https://www.youtube.com/watch?v=RQowiOF_FvQ
