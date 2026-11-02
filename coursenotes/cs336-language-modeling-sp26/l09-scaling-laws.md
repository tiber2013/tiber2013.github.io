---
title: "第 9 讲 · 缩放定律"
description: "Scaling law 的核心是用 log-log 图上的幂律关系描述“资源投入—损失下降”的规律，从而用小规模实验预测大规模训练结果。数据缩放是最自然、最干净的起点：损失随数据量多项式下降，但神经网络的指数远慢于均值估…"
---

# 第 9 讲 · 缩放定律

> **Stanford CS336 语言模型从零构建 · L09** ｜ 讲者 Stanford Online ｜ 78 分钟 ｜ [原视频](https://www.youtube.com/watch?v=Q15rhEWZPQ4)

## 核心结论

Scaling law 的核心是用 log-log 图上的幂律关系描述“资源投入—损失下降”的规律，从而用小规模实验预测大规模训练结果。数据缩放是最自然、最干净的起点：损失随数据量多项式下降，但神经网络的指数远慢于均值估计，大致像高维非参回归。架构、优化器、模型形状、MoE、批大小和学习率等也常表现出可预测的 scaling，但这些规律是“工程化”的，依赖正确的参数计数、超参设置和实验区间。固定 compute 时，数据量与模型大小需要联合优化；Kaplan 与 Chinchilla 的差异主要来自实现细节，而不是某一方做了完全不同的魔法。生产模型由于推理成本占主导，往往主动选择“过训练”的小模型，而不是严格采用 Chinchilla 的 20:1 比例。

---

## 课程定位：为什么要认真对待 scaling [00:05 - 03:59]

本讲暂时离开系统主题，回到深度学习与 scaling law 的基础内容。后续还会有一讲更高级的 scaling law，涉及现代开放模型技术报告、参数初始化、优化器等；下一讲则是 inference，因此课程会在 scaling 与 inference 之间来回切换。

讲者用一个具体场景引入：假设你有一位富有的朋友，给你一万块 B200 使用一个月，要求你构建一个优秀的开源语言模型。你已经组建了基础设施团队、准备好了预训练数据集，接下来要决定“训练一个怎样的大模型”。但大模型训练涉及大量选择：架构、超参数、模型规模、数据量等，而一次大 run 可能花费数百万美元，不能靠盲目试错。

![CS336 课程第9讲标题页：Scaling Laws - Basics](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0001_00-00-31.jpg)

Scaling law 的价值在于：把大部分优化放在小规模上完成，再用简单、稳健的规则外推到大规模。若小规模与大规模之间存在稳定的正则性，就可以对小规模实验结果外推，从而对大 run 的结果有定量预期。这本质上是一种工程视角：不是单纯复制文献中的最佳实践，而是用可预测的规律指导大规模决策。

![以“一万块B200训练开源大模型”为引子，列出基础设施、数据集、模型规模三个待决策问题](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0002_00-01-18.jpg)

---

## 历史与理论渊源：从泛化界到数据缩放 [04:03 - 11:13]

Scaling law 并不是纯神经网络时代的新事物，它与经典机器学习理论中的“经验样本复杂度”和泛化界密切相关。理论中常见的泛化界会给出模型误差随样本量 $m$ 变化的界，例如有限假设集上的形式：

$$
\epsilon(\hat h)\le \min_{h\in H}\epsilon(h)+2\sqrt{\frac{1}{m}\log\frac{2k}{\delta}}
$$

这类结果说明：误差上界会随样本量增加而下降，因此“数据量—误差”关系本来就是理论学习关心的问题。

![样本复杂度与收敛速率相关的数学公式，包括有限假设集和生成建模的理论界](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0005_00-05-12.jpg)

最早的数据 scaling law 可以追溯到 1993 年 AT&T Bell Labs 的学习曲线工作。其思路是：在完整大数据集上训练分类器很昂贵，因此可以先用小样本训练，拟合误差随训练集大小衰减的曲线，再外推完整数据集的性能。幻灯片给出的模型形式包括：

$$
\epsilon_{\text{test}}=a+\frac{b}{l^\alpha},\qquad
\epsilon_{\text{train}}=a-\frac{c}{l^\beta}
$$

这几乎就是早期的数据 scaling law。

![1993年学习曲线论文的清晰版幻灯片，展示了测试误差和训练误差随训练集大小变化的幂律模型公式及图表](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0007_00-06-32.jpg)

早期 NLP 也有类似探索。幻灯片展示了不同函数族（Exp3、Pow3、Pow4 等）对 BLEU 分数随训练样本量变化的拟合比较，并发现幂律形式很有预测力；其中 pow3、pow4 与今天常用的多项式形式非常接近。

![早期数据缩放定律历史幻灯片，展示了使用递归最小二乘法拟合的BLEU分数曲线图以及不同模型（Exp3, Exp4等）的公式表格](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0008_00-07-48.jpg)

讲者特别强调 2017 年一项跨领域的数据 scaling 研究（幻灯片标注为 Hestness II）：它在语音识别、机器翻译、语言模型等多个领域展示了数据量与性能之间的幂律趋势，并且已经讨论了后来广为人知的现象，例如：

- **Emergence**：准确率比损失更不连续，因此模型能力可能突然显现；
- **Scaling by compute**：如果系统能随数据可预测地改进，compute 就会变得极其重要；
- **Speed = accuracy**：既然靠 compute 扩展，系统和速度优化最终会转化为精度。

![Hestness II 关于涌现、计算规模与速度-精度权衡的幻灯片](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0009_00-09-06.jpg)

关于这些函数形式从何而来，讲者说明：很多是纯曲线拟合；理论可以提供候选函数形式，物理学家也擅长通过极限行为推导 scaling 形式，但不存在“唯一正确”的黄金规则。

---

## 数据缩放定律：幂律形式与指数直觉 [11:22 - 20:24]

数据 scaling law 是最简单、最自然的对象。设定是：固定训练过程，通常让模型显著大于数据集，然后系统性地增加数据量 $D$，观察测试损失如何下降。若超参调得好，损失应单调下降，最终趋于任务的不可约误差，因此整体形状类似 sigmoid；但在中间区间，log-log 图上会呈现清晰的线性趋势。

![数据缩放定律定义及预期表现：从数据量到误差的映射，呈现单调的逻辑斯谛曲线](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0014_00-14-17.jpg)

log-log 图上的直线意味着幂律关系：

$$
L(D)\propto D^{-\alpha}
\quad\Longleftrightarrow\quad
\log L=-\alpha \log D+c
$$

这也通常意味着模型还远离渐近线；一旦接近不可约误差，曲线会弯折而不是继续直线下降。

为什么幂律很自然？讲者用均值估计举例：若 $x_1,\dots,x_n\sim\mathcal N(\mu,\sigma^2)$，用样本均值估计 $\mu$，则

$$
\mathbb E[(\hat\mu-\mu)^2]=\frac{\sigma^2}{n}
$$

在 log-log 图上斜率是 $-1$。更一般地，只要误差形如“$1/n^\alpha$ 加常数”，就会出现 scaling law。

![均值估计的玩具示例，展示估计误差与样本量的缩放关系](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0016_00-16-53.jpg)

但神经网络的指数并不是 $-1$。经验拟合中常见指数大约在 $-0.1$ 到 $-0.3$ 量级。这比均值估计或线性回归慢得多，提示模型行为更像高维非参回归。若在 $d$ 维空间估计任意光滑函数，非参估计误差大致为

$$
\text{Error}\sim n^{-1/d}
$$

于是 log-log 斜率是 $-1/d$。神经网络的经验指数大致相当于 $d\approx 10$ 维的非参回归。有人据此论证神经网络在“像非参平滑器一样学习”，讲者认为这个直觉有趣，但证据依赖内在维度估计，不能完全当真。

![非参数学习的缩放定律推导，展示维度 d 对误差缩放的影响](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0018_00-19-29.jpg)

经验上，计算量、数据量、参数量分别与测试损失在 log-log 图上都能线性化，只是指数不同。例如幻灯片给出的拟合包括：

$$
L\propto C^{-0.050},\qquad
L\propto D^{-0.095},\qquad
L\propto N^{-0.076}
$$

![多因素缩放定律展示：计算量、数据集大小、参数量与测试损失的关系，以及非标准设置下的表现](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0013_00-12-59.jpg)

关于“模型大于数据”的含义：只要模型相对数据足够大、处于功率律区间即可，并不要求严格的一一比例；通常大 10 倍左右即可。若模型相对数据太小，会进入不可约误差区，此时增加数据不再有帮助；或者也可以显式拟合渐近常数再修正。数据 scaling law 本身主要用于预测“模型学得多快”，但不足以直接回答数据混合、重复等工程问题。

![机器翻译、语音和语言模型三个领域的缩放定律经验曲线图](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0017_00-18-11.jpg)

---

## 数据组成、重复与过滤 [20:25 - 29:00]

### 数据混合影响截距，而不一定影响斜率

数据组成对 scaling law 的影响通常体现在截距上，而斜率更多由模型类决定。一个线性回归玩具模型可以说明：若只有单一数据源，误差可能一直较高；混合多种数据反而能降低截距，因此数据多样性很重要。

实际做法是：在很小模型、很少数据上训练不同混合比例，拟合“混合比例—损失”的函数，找小规模最优点，再外推到大规模。但现实噪声很大；很多团队的实际做法是直接在小模型上选最佳数据混合，然后放大。DataDecide 等大规模实证研究发现，这种“小规模选最佳混合直接放大”通常有效，这与“斜率不变、只有截距变”的假设一致。

![对比数据混合选择两种方法的幻灯片：左侧为基于缩放定律的预测流程，右侧为DataDecide小规模实验评估图表](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0021_00-24-41.jpg)

分布偏移下的数据 scaling 也支持这一点：不同数据组成会改变误差截距，而斜率保持相似。因此收集更多样化的数据往往是有益的。

![展示分布偏移下的数据缩放定律，包含不同数据比例下的误差曲线及预期误差截距图](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0020_00-23-23.jpg)

### 数据重复

随着 compute 增长，唯一数据量不会同步增长，因此数据重复越来越重要。早期研究表明：在标准训练配方下，大约 4 个 epoch 以内基本不受伤；超过后，实际 scaling 会明显差于使用新鲜数据的投影曲线。可以写出修正后的函数形式来描述重复数据下的有效数据量：

$$
D'=U_D+U_D R_D^*\left(1-e^{-R_D/R_D^*}\right)
$$

其中 $U_D$ 是唯一 token 数，$R_D$ 是重复次数。

![关于数据重复下缩放定律的幻灯片，包含计算回报曲线、计算分配图以及有效数据量公式](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0022_00-25-59.jpg)

在“compute 无界”的极端设定下，重复数据和增大模型都会出现收益递减，最终需要借助 ensemble 等手段继续榨取数据价值。重要观察是：很多干预（正则化、ensemble 等）改变的是截距，斜率仍然相似。

![计算无界设置下的缩放定律幻灯片，包含三个图表展示不同参数对损失的影响](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0023_00-27-17.jpg)

### 数据过滤应随 scale 变化

数据过滤不是静态决策。compute 少时，应激进过滤，只保留最高质量数据；compute 多时，则需要更多数据，过滤器会变松，甚至开始使用质量较低的数据，以避免过度重复高质量数据。因此最优过滤策略随规模变化。

![数据选择缩放与有限性核算：数据池质量权衡及估计缩放曲线图](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0027_00-29-07.jpg)

另外，若实验只覆盖很窄的 compute 范围，很难区分多项式 scaling 和指数 scaling，因为局部 Taylor 展开下两者都可能看起来近似线性。 fitting scaling law 时要对区间和函数形式保持怀疑。

---

## 模型工程缩放：架构、优化器与模型形状 [29:04 - 38:54]

现在进入模型工程问题：如何高效设计大语言模型？例如 LSTM 是否不如 Transformer？SGD 是否不如 Adam？有限资源应花在更大模型、更长训练还是更多数据上？Scaling law 提供定量比较手段。

![模型工程缩放定律：如何高效设计大型语言模型及资源分配问题](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0028_00-30-17.jpg)

### 架构比较

要判断 Transformer 是否优于 LSTM，暴力方法是训练一个 GPT-3 规模的 LSTM；scaling law 方法是训练一系列小模型，在不同 compute 范围比较曲线。若 LSTM 的截距更差、甚至斜率更差，就不应选择它。现代架构论文（Mamba、Gated DeltaNet 等）通常都会给出这种 scaling 曲线；若新模型斜率更差，大规模后会被反超。

![Transformer与LSTM在不同参数量下的测试损失对比曲线图，Transformer曲线整体更低](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0030_00-32-29.jpg)

Google 的 T5 架构 scaling 研究覆盖了多种架构，并捕捉到了今天前沿模型仍在使用的趋势：Gated Linear Unit 表现更好；Performer 这类高效注意力 scaling 不佳；Switch Transformer 趋势较好；Mixture of Softmax 可能有效但后来不再常用。因此很多人把“是否在 scaling 曲线上有优势”作为判断干预是否值得的标准。

![展示多种模型架构性能对比的幻灯片，左侧为负对数困惑度与FLOPs的散点图，右侧为11种不同架构的缩放曲线小图。](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0031_00-33-34.jpg)

### 优化器比较

SGD 与 Adam 的 scaling 分析显示：截距不同，但斜率常常非常接近。即使从 SGD 换到 Adam 这样看似巨大的干预，斜率也可能不变。这是 scaling law 分析中反复出现的现象。

### 模型形状与参数计数

层数不是 scale-invariant 量：模型越大通常需要更多层；但 aspect ratio（如每层 $d_{\text{model}}$）可能近似 scale-invariant。经验上最优约每层 100 维 $d_{\text{model}}$，更深模型会略向更小 aspect ratio 偏移。单层模型 scaling 极差；超过一层后，更多层在每个 compute 水平上通常更好。

参数计数会显著影响 scaling law 形状。Kaplan 等人发现计入 embedding 参数时曲线很怪，于是排除了 embedding 参数，同时也排除了最后一层 softmax 参数。这个选择后来被证明对结果有实质影响。讲者强调：scaling law 不是自动发生的魔法，而是需要选对 x 轴、设对超参后才能得到跨多个数量级的可预测性。

### MoE 的参数与激活参数

MoE 将总参数与激活参数解耦。Apple/MIT 的分析显示：模型越大，最优稀疏度往往越高；在相同激活参数下增加总参数（即更多“非激活”参数）仍能降低损失。也就是说，MoE 中不激活的参数仍然有帮助，相关权衡也有可预测的 scaling 规律。

---

## 批大小与 critical batch size [41:02 - 48:22]

批大小和学习率是大 run 前必须重新调整的两个关键超参。批大小越大越利于数据并行，但太大会低效。Critical batch size 描述“继续增大 batch 何时开始收益递减”。

- **噪声限制区**：batch 较小时，增加 batch 会降低梯度噪声，带来接近完美的 scaling；
- **偏差限制区**：超过临界点后，限制因素不再是方差，而是局部梯度方向与全局最优之间的偏差，增加 batch 收益递减。

估计 critical batch size 的机械步骤是：选一个目标损失，扫描不同 batch size，记录达到目标损失所需的步数 $S$ 和样本数 $E$。二者通过 batch size 联系：$E=B\cdot S$。OpenAI 相关工作认为 steps 与 examples 之间存在权衡，平衡两者可得到

$$
B_{\text{crit}}\approx \frac{E_{\min}}{S_{\min}}
$$

即用最少样本数除以最少步数。这样会略多于最优步数、略多于最优样本数，但平衡了收敛速度与 batch 大小。

Critical batch size 随目标损失降低而幂律增大：想要更好的模型，就可以使用更大的 batch。这也符合直觉：越接近最小值，越需要降低方差来分辨细微差异。

---

## 学习率缩放与 μP [48:22 - 50:55]

学习率也会随规模移动。对宽度 scaling 的标准 MLP，模型越大，学习率通常应越小；一个常见经验规则是按 $1/\text{width}$ 缩小学习率。

另一条路线是重参数化：改变初始化、优化器步长或不同部分的缩放，使最优学习率在不同规模下保持不变，这类方法以 μP 为代表。于是有两种哲学：

1. **预测最优学习率如何随规模变化**，然后在大 run 上使用预测值；
2. **重参数化模型**，让最优学习率跨规模不变，再直接选最佳学习率。

两种方法都在大规模训练中成功使用过；讲者观察到更多人似乎倾向于 scaling law 预测路线，但两者都可行，后续高级讲座会详述。

---

## 上游/下游迁移与实验实践 [50:56 - 56:37]

Perplexity 的 scaling 非常干净、可预测，但下游性能不一定与 perplexity 强相关。T5 架构论文中，NL-12 的 perplexity 更好，但下游最佳模型却是 perplexity 更差的 NL-32 XL。这说明从预训练损失到下游任务的迁移存在不确定性。

因此，比较稳妥的做法是：先在低方差、干净的指标（如 perplexity）上建立 scaling 规律，再依赖或验证其向下游的迁移。不要只看预训练 perplexity 就认为下游一定更好。

实验细节方面：

- Perplexity 的方差通常很小，因此 scaling 图上的每个点常是单次运行；重新运行的差异可能在第二位小数。
- 学习率、critical batch size 等量的 scaling 实验方差可能很大，需要更多重复或方差缩减。
- 大多数预训练是一次 pass，因此训练损失与验证损失非常接近，很多图甚至混用 train/test loss；有些预训练代码库根本不记录 validation loss。

---

## 计算最优分配：数据还是模型 [56:37 - 61:26]

固定 compute 时，compute 大致与“数据量 × 参数量”成正比，即 $C\sim N D$。小模型灌入大量数据会浪费 compute；需要联合考虑模型大小和数据量。

联合 scaling law 的典型形式是两项误差相加：

$$
L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}
$$

Rosenfeld 的形式基本就是两个 inverse 项之和；Kaplan 的形式稍复杂但思想类似。检查极限很直观：若 $D\to\infty$，只剩模型项，退化为模型 scaling law；若 $N\to\infty$，只剩数据项，退化为数据 scaling law。

这类联合律可以在低 compute、低数据区域拟合，再外推到高 compute、高数据区域；随后在固定 flops 约束下最小化 $L(N,D)$，得到最优的模型大小与数据量分配。

---

## Kaplan 与 Chinchilla：差异来自哪里 [61:26 - 74:21]

Kaplan 的处方使参数量随 compute 增长更快，tokens per parameter 随 compute 下降，因此倾向于训练越来越大的模型。GPT-3 时代大量百亿、万亿参数模型的部分动机就来自这类预测。

Chinchilla 则指出应训练更小的模型、用更多数据，给出约 20 tokens/parameter 的经验比例。Chinchilla 提出三种拟合方法：

1. **Lower envelope**：取每条训练曲线下包络，找到每个 flop 预算下的最佳损失，再拟合模型大小；
2. **Isoflops**：固定多个 flop 预算，在每个预算下扫描参数/数据权衡，取每个预算的最小损失，再拟合；
3. **联合函数拟合**：假设 $L(N,D)$ 的函数形式，直接对大量训练 run 做曲线拟合。

方法 1 和方法 2 给出的数据/参数指数大约都是 0.5/0.5；方法 3 给出约 0.46/0.54，二者不完全一致。Kaplan 的指数差异则大得多。

为什么 Kaplan 与 Chinchilla 差异这么大？后续研究逐步复现并定位原因：

- **参数计数**：Kaplan 排除了 embedding 参数，也排除了最后一层 softmax 参数；是否计入这些参数会显著改变 scaling 形状。
- **学习率 warm-up**：Kaplan 的许多模型很小，warm-up 结束时模型还没收敛，学习率设置不优。
- **批大小**：Kaplan 固定使用较大的 batch，这对小模型并不最优；改为随规模调整 batch 后，结果更接近 Chinchilla。
- Pearson 与 Song 的模拟进一步指出：Kaplan 处于更低 compute 区间，对细节更敏感；排除非 embedding 参数引入的非线性也会造成偏差。
- Chinchilla 方法 3 与 1/2 的不一致，后来被 Epoch AI 通过从图中提取数据并重新拟合发现：原论文的方法 3 可能 underfit；重拟合后结果又回到约 20:1 的比例。

讲者的结论是：scaling law 在某种意义上像“下界”——它告诉你“如果继续按这个 recipe 放大，会得到什么”。如果 recipe 本身有问题（warm-up 不合理、batch 设置不合理、参数计数不对），就会得到误导性的 scaling law。因此必须让实验尽量接近真正的大 run 配方。

---

## 生产取舍、isoflops 与总结 [74:21 - 77:56]

Chinchilla 的 20:1 比例并不是生产模型的终极答案。生产环境中，训练 compute 往往不是主要成本；大部分 compute 花在研发和推理服务上。因此，模型越小、能力越强越好，即使训练时多花 compute 也值得。这就是所谓的“过训练”（overtraining）——讲者强调，对生产模型而言，过训练往往才是正确的训练量。

- GPT-3 约 3 tokens/parameter，明显 undertrained；
- Chinchilla 将比例推到约 20:1；
- 之后随着推理服务规模化，模型重新走向 overtraining，并结合 MoE 优化推理成本。

因此 Chinchilla 的重要性更多在于方法论：如何严谨地拟合 scaling law、如何避免参数计数和超参陷阱，而不是 20:1 这个数字本身。

Isoflops 是持久有用的研究工具：固定 flop 预算，扫描其他自由度，观察损失曲面。它容易执行，通常能可靠读出不同参数如何随规模变化；扩散模型、MoE 等研究都采用过 isoflops 设计。遇到“如何权衡多个资源”的问题时，isoflops 是很好的默认方法。

最后总结：资源投入与性能之间存在 log-linear 正则性，适用于数据、模型参数、compute、MoE 稀疏度等多种对象；这使得我们可以在不实际运行大模型的情况下进行“大规模工程”。数据 scaling 是最自然的基础对象，模型 scaling 支持架构与优化器决策，联合 scaling 指导 compute 分配，而 scaling law 本身是稳健但需要细心构造的预测工具。

---

## 要点回顾

- Scaling law 用 log-log 线性/幂律关系把资源投入映射到损失，使小规模实验可外推大规模训练。
- 历史根源包括泛化界、1993 年学习曲线、早期 NLP 函数形式比较和 2017 年跨领域数据 scaling 研究。
- 数据 scaling 最干净：损失随数据量多项式下降；神经网络的指数约 $-0.1$ 到 $-0.3$，类似高维非参回归。
- 数据组成通常改变截距而非斜率；小规模选最佳混合再放大在实践中常有效。
- 数据重复在约 4 epoch 内影响小，之后明显差于新鲜数据；可用修正函数形式描述有效数据量。
- 数据过滤应随 scale 自适应：compute 越少过滤越激进，compute 越多过滤越松。
- 架构、优化器、aspect ratio、MoE 等都有可预测 scaling；但参数计数和超参设置会显著改变结论。
- Critical batch size 标志从噪声限制的完美 scaling 进入偏差限制的收益递减区；它随目标损失降低而幂律增大。
- 学习率可随规模预测变化，也可用 μP 类重参数化保持最优学习率不变；两种路线都成功。
- Perplexity scaling 很干净，但向下游迁移不一定可靠；应先在低方差指标建立规律，再验证迁移。
- 固定 compute 时需联合优化 $N$ 与 $D$；Rosenfeld/Kaplan 的联合律可从小规模外推并做约束优化。
- Kaplan 与 Chinchilla 的差异主要来自参数计数、warm-up、batch size 等实现细节；Chinchilla 方法更稳健。
- 生产模型因推理成本常主动“过训练”小模型；isoflops 是研究资源权衡的默认工具。
