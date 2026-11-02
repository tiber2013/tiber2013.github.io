---
title: "第 8 讲 · 大语言模型（客座讲师 Hao Liu）"
description: "语言模型本质上是对 token 序列的概率分布建模，现代 LLM 普遍采用自回归分解与最大似然训练，并分为大规模预训练、任务微调、人类反馈对齐三个阶段，其中预训练决定模型能力上限。讲座用缩放定律说明算力应如何在模型参数量…"
---

# 第 8 讲 · 大语言模型（客座讲师 Hao Liu）

> **CS294-158 深度无监督学习 · L08** ｜ 讲者 Pieter Abbeel ｜ 135 分钟 ｜ [原视频](https://www.youtube.com/watch?v=tCgX48cvuw4)

## 核心结论

语言模型本质上是对 token 序列的概率分布建模，现代 LLM 普遍采用自回归分解与最大似然训练，并分为大规模预训练、任务微调、人类反馈对齐三个阶段，其中预训练决定模型能力上限。讲座用缩放定律说明算力应如何在模型参数量与训练 token 数之间分配：早期工作偏向更大模型，Chinchilla 则表明参数与数据应大致同步增长，实践中常用“token 数约为参数量的 20 倍、最好加倍到 40 倍”的经验规则。损失是预测下游性能的关键指标：在相同数据分布下，相同 loss 的大模型与小模型下游表现接近。长上下文的核心瓶颈是注意力 \(O(s^2)\) 的内存开销，Blockwise Parallel Transformer 与 RingAttention 通过重排计算图和环形通信，把序列长度扩展到百万级。FlashAttention 减少显存层级间的数据搬运，检索增强弥补模型参数中的事实知识不足，而滑动窗口注意力和状态空间模型则是降低注意力成本的重要替代方向。

## 课程安排与讲座定位 [00:00 - 03:03]

- 课程行政：项目提案正在逐份反馈；4 月有里程碑，5 月交最终项目报告与视频展示；HW3 已截止，HW4 已发布，内容为扩散模型。
- 本次为嘉宾讲座，讲者 Hao Liu 是 Berkeley 最后一年博士生，研究覆盖强化学习、语言建模、多模态学习、视频建模等。
- 讲座目标：前半讲语言模型的基础与能力，后半讲研究与实践中的可操作准则。

## 语言模型的定义与三阶段训练 [03:03 - 07:33]

语言模型是定义在 token 序列上的概率分布：

$$
p_\theta(y)=\prod_{t=1}^{T}p_\theta(y_t\mid y_{<t})
$$

其中 \(y=(y_1,\dots,y_T)\) 是 token 序列，\(\theta\) 是模型参数。模型通常用 Transformer 参数化，训练目标是最大化观测序列的对数似然：

$$
\max_\theta \frac{1}{|\mathcal{D}|}\sum_{y\in\mathcal{D}}\log p_\theta(y)
$$

![语言模型的形式化定义页：语言模型是序列上的概率分布，包含似然分布、token 序列与参数三要素](/coursenotes/cs294-158-deep-unsupervised-sp24/l08-large-language-models/kf_0001_00-03-04.jpg)

选择自回归分解的主要理由：

- 每个 token 都成为预测目标，监督信号更密集，有利于规模扩展。
- 预测下一个 token 的形式天然适合对话式 AI，模型可以逐步生成语言。
- 目标本质上是“预测未来”，与最大似然目标一致。

典型训练分为三个阶段：

1. **大规模预训练**：在极大规模序列上最大化似然，学习数据分布。以 LLaMA 为例，语料规模可达约 1.5 万亿 token，参数量可达数十亿到数百亿。
2. **微调**：预训练学到的是互联网数据分布，未必对应特定任务，因此需要在目标任务上做少量微调。
3. **从反馈中学习**：利用人类反馈让模型遵循用户意图，即对齐阶段。

讲座重点放在预训练，因为它基本决定模型能力；微调与反馈学习只作简要讨论。

## 为什么自回归无监督学习能扩展 [07:33 - 11:45]

以 LLaMA 预训练曲线为例，横轴是训练见过的 token 数，纵轴是平均负对数似然损失。不同规模模型的损失随 token 数持续下降，在万亿 token 量级仍未饱和。

![不同参数量LLaMA模型在无监督学习下的训练损失随token数量变化曲线图](/coursenotes/cs294-158-deep-unsupervised-sp24/l08-large-language-models/kf_0002_00-09-00.jpg)

选择无监督学习的关键原因：

- 互联网上有海量无标注文本，数据量远大于标注数据。
- 训练 token 越多，模型越能捕捉数据分布，泛化通常优于纯监督学习。
- 这呼应 Rich Sutton 的“苦涩教训”：利用算力的方法往往取得更大成功。

这里的“算力”指前向与反向传播的计算，主要是矩阵乘法，用 FLOPs 度量。增加算力有三条路径：

- 增加模型参数量；
- 增加训练 token 数；
- 增大上下文窗口。

## Tokenization 与从 LSTM 到 Transformer [11:45 - 21:30]

### Tokenization

- **Byte 级**：最通用，但序列很长，训练昂贵。
- **字符级**：一个词被拆成多个字符，序列仍然偏长。
- **词级**：序列短，但词表巨大，且模型难以共享相似词的语义。
- **Byte Pair Encoding（BPE）**：折中方案。把词拆成子词，反复统计最频繁的相邻 pair，合并为新 token，直到达到指定词表大小。

### 早期 LSTM 的情感神经元

2017 年前后，有工作用 LSTM 在大量 Amazon 评论上做自回归 token 预测。结果发现某个神经元的值可以控制生成评论的正负情感：调节该神经元即可生成正面或负面评论。在 IMDb 评论上逐段检查该神经元，也能看到它随文本情感由负到正而变化。

![LSTM训练后情感神经元数值分布直方图，正负评论明显分离](/coursenotes/cs294-158-deep-unsupervised-sp24/l08-large-language-models/kf_0003_00-15-44.jpg)

这说明自回归语言建模可以让模型学到人类情感等抽象属性。但当时 LSTM 很难继续扩展，研究者尝试更多数据、更大模型和替代架构，效果有限。

### Transformer 的缩放优势

注意力机制允许每个 token 直接关注任意过去 token，而不像 RNN 那样必须把历史压缩进固定隐状态。Transformer 还可以通过增大 MLP 轻松增加参数量，而 MLP 主要是大规模矩阵乘法，和 GPU/TPU 的算力增长非常匹配。

![Transformer架构图，包含缩放点积注意力、多头注意力及完整编码器-解码器结构](/coursenotes/cs294-158-deep-unsupervised-sp24/l08-large-language-models/kf_0004_00-20-14.jpg)

实验对比了 LSTM 与 Transformer 的 per-token test loss：横轴是序列中的 token 位置，纵轴是该位置上的预测损失。随着模型变大，Transformer 始终优于 LSTM；上下文越长，差距越大。把模型平均性能对参数量作图，Transformer 呈现漂亮的线性缩放，而 LSTM 很早就饱和。

## 训练目标与架构选择：Masked、Prefix、Autoregressive [21:30 - 28:15]

常见目标函数有三类：

- **Masked Language Modeling**：如 BERT，随机遮住一部分 token，让模型预测被遮住的部分。适合学习句子级 embedding，因为双向注意力能更好捕捉完整句子的语义；任务也相对容易，小模型即可训练。BERT 时代最大模型不到 10 亿参数，但今天自回归模型 10 亿参数往往太小。
- **Prefix Language Modeling**：前缀部分使用双向注意力，后续部分使用自回归注意力。可以把它理解为 encoder-decoder 的一种统一形式：双向部分相当于 encoder，自回归部分相当于 decoder。实际使用不如全自回归广泛。
- **Autoregressive / Causal Language Modeling**：GPT 式，每个 token 只能关注之前的 token，并预测下一个 token。

计算成本上，这些目标消耗的 FLOPs 几乎相同，但因果解码器被证明最可扩展，因此成为主流 LLM 的选择。

讲座还提醒：研究新架构时，应把预训练困惑度对训练 FLOPs 作图，检查其缩放是否优于 vanilla Transformer。很多替代架构在特定场景有用，但整体缩放不如标准 Transformer。

## 计算成本估算与缩放定律 [28:15 - 47:55]

### 计算成本公式

训练一个 Transformer 的 FLOPs 可近似为：

$$
C=6ND\left(1+\frac{s}{6d}\right)
$$

其中 \(N\) 是参数量，\(D\) 是训练 token 数，\(s\) 是序列长度，\(d\) 是隐藏维度。MLP 的中间维度通常是 \(4d\)，因此参数量大致随 \(d^2\) 增长。

对于 LLaMA 这类模型，序列长度 \(s\) 通常为 2K–4K，而 \(6d\) 约为 24K–48K，因此 \(s\ll 6d\)，公式可简化为：

$$
C\approx 6ND
$$

例如 70 亿参数模型训练 2 万亿 token：

$$
C\approx 6\times 7\times 10^9\times 2\times 10^{12}
=8.4\times 10^{22}\ \text{FLOPs}
$$

![计算成本公式及LLaMA模型算力消耗示例，以及最优分配问题的数学定义](/coursenotes/cs294-158-deep-unsupervised-sp24/l08-large-language-models/kf_0005_00-29-14.jpg)

### 缩放定律的基本形式

固定算力 \(C\)，需要选择模型参数量 \(N\) 与训练 token 数 \(D\)：

$$
N_{\text{opt}}(C),D_{\text{opt}}(C)
=\arg\min_{N,D:\ \text{FLOPs}(N,D)=C}L(N,D)
$$

实验发现，测试损失与计算量、模型大小、数据量之间在 log-log 坐标下呈幂律关系：

$$
N_{\text{opt}}\propto C^a,\qquad D_{\text{opt}}\propto C^b
$$

并且 \(a+b=1\)。

![最优参数量与数据量随计算量变化的幂律关系图及拟合公式](/coursenotes/cs294-158-deep-unsupervised-sp24/l08-large-language-models/kf_0006_00-33-44.jpg)

### 从“更大模型”到 Chinchilla

早期 OpenAI 的缩放定律认为应把更多算力分配给模型参数，即训练更大的模型。后来的 Chinchilla 工作做了大量不同数据规模的实验，发现参数量与数据量应大致等比例增长，拟合结果约为 \(a=0.49,\ b=0.51\)，即数据甚至略更重要。

![Chinchilla缩放定律下模型参数量与计算量的关系图](/coursenotes/cs294-158-deep-unsupervised-sp24/l08-large-language-models/kf_0007_00-40-29.jpg)

Chinchilla 还指出，当时许多已发布模型（如 GPT-3、Megatron、Gopher）把过多算力花在增大参数上，训练 token 不足，因而没有达到算力最优。图中这些模型对应的点明显高于 Chinchilla 最优线，说明存在算力浪费。

讲者强调：这些常数高度依赖数据集。公开论文使用的数据往往不可得，因此做研究时最好自己训练一批小模型，拟合缩放常数，而不是直接照搬论文数值。

### 实用经验规则

Chinchilla 表格给出不同参数量下的最优 token 数，大致满足：

$$
D\approx 20N
$$

![不同参数量模型的最优训练Token数量估算表，包含参数量、FLOPs及对应Token数](/coursenotes/cs294-158-deep-unsupervised-sp24/l08-large-language-models/kf_0008_00-49-28.jpg)

实践中建议把这个估计加倍：

$$
D\approx 40N
$$

例如训练 10 亿参数模型，可考虑 400 亿 token。这样在论文或工程中更容易说服读者/评审。

如果目标是**推理最优**，可以进一步超出 Chinchilla 最优：训练时多花算力把模型训得更充分，推理时只支付固定模型大小的成本。例如 LLaMA 7B 训练了约 1 万亿 token，远多于 20–40 倍参数量的建议，目的是得到一个推理成本低但能力强的模型。

### 损失决定下游性能

在相同数据分布下，损失是预测下游性能的关键指标。实验比较不同规模 LLaMA 模型：大模型用较少 token 达到某个 loss，小模型用更多 token 达到相同 loss，二者在 TriviaQA、HellaSwag 等下游任务上的表现几乎一致。

![训练损失与下游任务性能的关系图，证明相同损失下大模型和小模型性能一致](/coursenotes/cs294-158-deep-unsupervised-sp24/l08-large-language-models/kf_0009_00-53-58.jpg)

因此实践中可以：用较小模型和较低成本试错，只要最终 loss 更低，通常下游性能更好；若两个模型 loss 相同，则性能大致相当。

数据采样方面：

- 训练数据通常随机采样，不做“先低质量后高质量”的课程学习。
- 不应把低质量数据混入预训练。
- 可用分类器筛选接近 Wikipedia 的高质量数据，因为互联网 token 几乎无限，难点在于筛选。

## 长上下文：Blockwise Parallel Transformer 与 RingAttention [59:52 - 72:28]

### 问题：注意力内存是二次的

Transformer 注意力需要计算 \(N\times N\) 的注意力权重矩阵，内存开销随序列长度平方增长。视频、长文档、代码库、智能体轨迹等场景可能需要数百万甚至数十亿 token 的序列，无法直接放入单张 GPU/TPU。

简单把序列切块分到多设备也不行：每个 chunk 仍需与其他 chunk 做注意力交互，设备最终还是要持有完整序列。

### Blockwise Parallel Transformer（BPT）

BPT 的核心是重排注意力和前馈网络的计算顺序：

- 外层循环遍历 query block；
- 内层循环遍历 key/value 序列；
- 每算完一个 query block 的注意力，立即计算该 block 的 FFN，而不必等整个序列的注意力完成。

![Blockwise Parallel Transformers的架构图，展示注意力与前馈网络的分块并行计算流程](/coursenotes/cs294-158-deep-unsupervised-sp24/l08-large-language-models/kf_0010_01-01-03.jpg)

这样峰值激活内存显著降低。分析显示，标准注意力加标准 FFN 的峰值约为 \(O(s^2)\)；BPT 可把峰值激活内存降低约 4 倍，从而训练更长序列。

![对比标准注意力、内存高效注意力与BPT三种方案的内存峰值开销分析](/coursenotes/cs294-158-deep-unsupervised-sp24/l08-large-language-models/kf_0011_01-05-13.jpg)

BPT 使用精确注意力，不是近似方法，也可以与近似注意力技术组合。实验表明，在 Gopher 等模型上可获得约 16 倍更低内存、约 6 倍更长序列。

### RingAttention：把长序列分布到多设备

BPT 仍受单设备内存限制。RingAttention 进一步把 query 的外层循环分布到多台设备上，key/value 在设备环中传递：

- 每台设备负责一部分 query；
- 当前设备从上一台设备接收 KV block，计算完再把 KV 发给下一台设备；
- 通信与计算重叠，通信量较小，因为只需传递相邻的 KV block，而不是所有序列。

![RingAttention 扩展架构图，展示在环形设备间循环传递 Key-Value 并计算局部 Query 的注意力与 FFN 流程](/coursenotes/cs294-158-deep-unsupervised-sp24/l08-large-language-models/kf_0012_01-07-28.jpg)

关键条件是块大小要足够大，使计算时间不小于通信时间，即满足算术强度要求：

$$
\text{compute\_time}(\text{blockwise attention}+\text{blockwise FFN})
\ge \text{communication\_time}
$$

![算术强度分析幻灯片，给出块大小需满足计算时间大于等于通信时间的条件，并列出 A100/TPU 各型号的 FLOPS、HBM、互联带宽与最小块大小/序列长度表格](/coursenotes/cs294-158-deep-unsupervised-sp24/l08-large-language-models/kf_0013_01-09-42.jpg)

这样多 GPU 训练百万 token 序列不会产生额外通信开销，序列长度只受设备数量限制。端到端训练可使用数百倍于以往的长序列。

## 多模态与百万级上下文能力 [72:28 - 80:18]

有了长序列建模能力，就可以把文本、图像、视频、音频统一为 token 序列进行自回归建模。

### 百万上下文检索

Large World Model（LWM）在 100 万 token 上下文下做“大海捞针”检索任务：横轴是上下文长度，纵轴是事实放置深度，绿色表示正确检索。结果显示几乎全绿，说明模型在 1M 上下文内能有效检索事实，没有明显“lost in the middle”。

![1M上下文模型在“大海捞针”任务中的检索准确率热力图，全绿表示近乎完美](/coursenotes/cs294-158-deep-unsupervised-sp24/l08-large-language-models/kf_0014_01-14-12.jpg)

### 长视频问答

LWM 可以处理 1 小时 YouTube 视频（超过 500 个片段）。示例问题包括“穿霸王龙服装的人骑的是什么交通工具”和“蹦床上女孩的夹克是什么颜色”。GPT-4V、Gemini Pro Vision、Video-LLaVA 均回答错误，LWM 正确回答为摩托车和蓝色夹克。

![Large World Model 长视频问答对比：在1小时YouTube视频上，LWM正确回答关于霸王龙服装和蹦床女孩的问题，而GPT-4V、Gemini Pro Vision和Video-LLaVA均回答错误](/coursenotes/cs294-158-deep-unsupervised-sp24/l08-large-language-models/kf_0015_01-16-05.jpg)

### Gemini 1.5 的长上下文表现

Gemini 1.5 把上下文扩展到 100 万 token。实验显示：

- 损失随序列位置呈幂律下降，上下文越长，预测越好；
- 代码任务上，条件上下文越长，代码预测损失持续下降，在 1000 万位置甚至优于拟合曲线；
- 1M 上下文可直接分析整个代码仓库、超过 1000 页的文档；
- 音频转录可处理从 12 分钟到 11 小时，最长约 22 小时，端到端单模型优于 Whisper + GPT-4 Turbo 的组合；
- 低资源语言翻译中，把语法书和全部可用语料放入上下文，模型可超过人类专家。

![Gemini 1.5 Pro与Whisper+GPT-4 Turbo在长音频转录任务上的性能对比热力图](/coursenotes/cs294-158-deep-unsupervised-sp24/l08-large-language-models/kf_0016_01-18-42.jpg)

## FlashAttention 与检索增强 [80:18 - 87:42]

### FlashAttention：减少数据搬运

GPU 内存有层级：HBM 容量大但慢，SRAM 容量小但快。标准注意力实现会把注意力矩阵频繁写入/读出 HBM，成为瓶颈。

FlashAttention 通过 tiling 和 CUDA 实现，把立即需要计算的块保留在 SRAM，减少 HBM 读写。原则是“下一步马上要用的数据留在 SRAM”。FlashAttention-2 进一步把 query 放在外循环、KV 放在内循环，因为不同 query 之间独立，可减少通信并提升吞吐。

![FlashAttention与标准注意力机制在不同序列长度和头维度下的吞吐量对比柱状图](/coursenotes/cs294-158-deep-unsupervised-sp24/l08-large-language-models/kf_0017_01-23-12.jpg)

### 检索增强：RETRO

语言模型参数无法压缩所有事实知识，也无法访问最新信息。工具使用与检索是重要方向。RETRO 的流程：

- 用 BERT 编码输入序列和外部数据集；
- 用 kNN 检索相似邻居；
- 通过 chunked cross attention 把检索到的邻居注入 Transformer；
- 数据集的 embedding 可预先缓存，检索器可冻结。

![RETRO模型架构图，展示了通过检索增强语言模型及分块交叉注意力（CCA）机制](/coursenotes/cs294-158-deep-unsupervised-sp24/l08-large-language-models/kf_0018_01-25-27.jpg)

在相同参数量下，RETRO 在多项任务上取得更好的缩放；提升最明显的是 GitHub 代码和书籍等长文档数据集，因为这些场景难以把所有上下文放进注意力窗口，检索能补充必要信息。

## 注意力的替代与补充：滑动窗口与状态空间模型 [87:42 - 92:12]

### 滑动窗口注意力

每个 token 只关注固定窗口内的过去 token，计算成本从二次降为线性。随着层数加深，token 的有效感受野逐步扩大，因此仍能访问较远的过去。训练较快，但效果通常不如全注意力。

### 状态空间模型（SSM）

SSM 用连续状态空间描述序列：

$$
x'=Ax+Bu,\qquad y=Cx+Du
$$

离散化后得到：

$$
x'=\bar A x+\bar B u,\qquad y=\bar C x+\bar D u
$$

并可用卷积形式并行训练：

$$
y=\bar K * u
$$

![状态空间模型（SSM）的连续状态空间、长程依赖与离散表示三分支结构图](/coursenotes/cs294-158-deep-unsupervised-sp24/l08-large-language-models/kf_0019_01-29-57.jpg)

SSM 维护对遥远过去的隐藏状态，推理时只需常数级状态，因此推理快。Mamba 是代表性模型，在 10 亿、30 亿参数的小规模上缩放可匹配 Transformer，但随着算力增大，曲线略高于 Transformer。

![不同架构（Attention、SSM、Convolution）在关联召回任务上的准确率与循环状态大小对比散点图](/coursenotes/cs294-158-deep-unsupervised-sp24/l08-large-language-models/kf_0020_01-32-12.jpg)

关联召回任务比较了 Attention、SSM、Convolution 等架构在循环状态大小与召回准确率之间的权衡。讲座最后提到可以探索 SSM 与滑动窗口等机制的组合，但转写在此处截断，未展开具体方案。

## 要点回顾

- 语言模型是 token 序列上的概率分布，主流做法是自回归分解加最大似然训练。
- LLM 训练分三阶段：大规模预训练、任务微调、从人类反馈中学习对齐；预训练决定能力上限。
- 自回归无监督学习可利用海量无标注数据，训练损失随 token 数持续下降。
- BPE 通过反复合并高频相邻子词，在字节级、字符级、词级之间取得折中。
- Transformer 的注意力可直接访问过去 token，大 MLP 易扩展，缩放性显著优于 LSTM。
- 训练算力近似 \(C\approx 6ND\)；LLaMA 规模下 \(s\ll 6d\)，公式可简化。
- 缩放定律表明损失与算力、参数量、数据量呈幂律关系；Chinchilla 建议参数与数据大致同步增长。
- 实用规则：训练 token 数约为参数量的 20 倍，实践中可加倍到 40 倍；推理最优模型可训练更久。
- 相同数据分布下，loss 可预测下游性能；相同 loss 的大模型与小模型表现接近。
- Blockwise Parallel Transformer 通过 query 外循环、KV 内循环和逐块 FFN 降低峰值内存。
- RingAttention 把 query 分布到设备环，KV 环形传递，通信与计算重叠，可训练百万 token 序列。
- FlashAttention 通过 tiling 减少 HBM 与 SRAM 间数据搬运；FlashAttention-2 优化循环顺序提升吞吐。
- RETRO 用 BERT embedding + kNN 检索 + chunked cross attention 注入外部知识，长文档任务收益明显。
- 滑动窗口注意力降低计算成本；SSM/Mamba 以隐藏状态实现快速推理，小规模可匹配 Transformer，仍在探索组合方案。
