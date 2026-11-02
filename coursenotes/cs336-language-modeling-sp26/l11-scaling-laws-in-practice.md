---
title: "第 11 讲 · 缩放定律进阶：μP、WSD 与外推"
description: "经典 Chinchilla 之后，前沿 scaling 研究大致分成两条路线：一条用 μP 等参数化把最优学习率等超参数“稳定”到不随规模变化；另一条直接在小规模上网格搜索并拟合学习率、批大小的幂律，再外推到大规模。Mi…"
---

# 第 11 讲 · 缩放定律进阶：μP、WSD 与外推

> **Stanford CS336 语言模型从零构建 · L11** ｜ 讲者 Stanford Online ｜ 77 分钟 ｜ [原视频](https://www.youtube.com/watch?v=vTfEyOyzV9E)

## 核心结论

经典 Chinchilla 之后，前沿 scaling 研究大致分成两条路线：一条用 μP 等参数化把最优学习率等超参数“稳定”到不随规模变化；另一条直接在小规模上网格搜索并拟合学习率、批大小的幂律，再外推到大规模。MiniCPM 展示了 μP 的稳定效果，并提出 WSD 学习率，使 Chinchilla 式数据/模型权衡实验可以复用长训练、只重做衰减阶段。DeepSeek 则展示了直接拟合 $B_{\text{opt}}$、$\eta_{\text{opt}}$ 的 scaling law，并用 WSD 复现 Chinchilla。后续 Qwen、Kimi K2、Llama 3、Hunyuan、MiniMax 等把 MoE 稀疏度、架构选择也纳入 scaling 分析；但超参数和优化器仍有很强的 scale 依赖，scaling law 的外推在实践中远没有看起来那么“科学”。

## 课程动机：从 Chinchilla 到前沿实践 [00:05 - 03:47]

课程先回顾经典 scaling law  canon：Kaplan、Hestness、Chinchilla，大约到 2022 年。此后由真正训练大模型的人发表的 scaling 论文变少，很多来自开源社区；课程希望讨论这些论文中更偏实践的细节：Chinchilla 式分析在真实开源模型规模上是否有效、能否节省训练和拟合成本、是否应选择特定架构或参数化来让模型“scale nicely”。

![关于语言模型缩放与超参数调优最佳实践的动机幻灯片，列出三个核心研究问题](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0001_00-00-26.jpg)

一个反复出现的主题是：初始化、优化器和各种超参数（学习率、批大小）都可能具有 scale 敏感性。好的初始化可以让某些超参数不再随规模剧烈漂移。

![初始化与优化器对缩放敏感性的多图表分析，包含学习率、批大小随模型规模变化及不同优化器损失对比](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0002_00-02-34.jpg)

课程重点选了两篇论文：MiniCPM（2024，小型高性能语言模型）和原始 DeepSeek LLM 论文。两者对“敏感超参数”采取不同策略：MiniCPM 用 μP 试图稳定它们；DeepSeek 直接拟合它们的 scaling law。

![MiniCPM论文介绍幻灯片，包含论文标题、作者列表及核心结论总结](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0003_00-03-51.jpg)

## MiniCPM：用 μP 稳定学习率 [03:47 - 09:48]

MiniCPM 是 2024 年来自清华等团队的高性能小模型，在 1–2B 参数区间按当时标准很有竞争力。它做了大量细致的 scaling 计算，并用 μP 来稳定和简化缩放。

μP 的目标是让最优学习率不随模型规模变化。MiniCPM 论文给出的具体操作包括：

- 缩放 embedding 输出；
- 按 $\sqrt{\text{num\_layers}}$ 缩放残差连接；
- 按 fan-in/fan-out 比例缩放二维矩阵张量的初始化；
- 对张量使用逐参数学习率；
- 缩放 LM head。

![muP稳定缩放技术的操作列表，包含嵌入输出缩放、残差连接缩放、张量初始化、学习率缩放和LM头缩放的具体公式与参数。](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0004_00-05-08.jpg)

实验显示，采用这套参数化后，不同模型大小的学习率-损失曲线的最小值基本都落在 $10^{-2}$ 附近，最优学习率随规模的偏移很小。这是 μP 的一个成功案例：如果最优学习率固定，就不需要为每个规模重新调它。

![不同模型大小下学习率与损失的关系折线图，显示应用Tensor Program后最优学习率偏移极小。](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0005_00-07-42.jpg)

但批大小并不会因此固定。MiniCPM 发现最优批大小仍随数据量和模型大小变化；把最优批大小对目标损失作图，会看到类似 Kaplan 临界批大小的幂律结构：目标损失越低，最优批越大。

![三种模型大小（9m, 30m, 170m）下，数据量、批次大小与损失关系的三维散点图及最优批次大小拟合线。](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0006_00-08-59.jpg)

## WSD 学习率与低成本的 Chinchilla 分析 [09:48 - 15:33]

做 Chinchilla 分析需要训练很多不同数据量、不同模型大小的模型。如果使用 cosine 学习率，每个训练长度都需要知道总预算，增加数据量就必须从头重训；这让拟合 scaling law 的成本近似从 $n$ 变成 $n^2$。

![模型规模与数据权衡的六联图，展示不同余弦周期长度下学习率、训练损失和C4损失的对比曲线](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0007_00-10-16.jpg)

MiniCPM 推广了后来常被称为 warm-up–stable–decay（WSD）的学习率：先用固定步数 warm-up，然后长时间保持恒定学习率，最后在训练末端用较短阶段快速衰减。衰减阶段通常约占总训练量的 10–20%。

![miniCPM中WSD学习率调度方案，对比余弦调度与WSD调度的学习率曲线，并附有方法说明文字](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0008_00-11-33.jpg)

WSD 的关键好处是可复用：如果想延长训练，可以回滚到稳定阶段末尾的 checkpoint，继续以稳定学习率训练，然后重新做一次衰减。这样只需重做约 10% 的衰减成本，而不必重跑整个预训练。性能上，WSD 在稳定阶段看起来损失较高，但进入衰减后会迅速追回，通常能匹配甚至偶尔超过调好的 cosine；实践中 cosine 有时略好，但 WSD 是非常通用的默认选择。

![WSD学习率在miniCPM上的C4损失曲线，对比不同WSD配置与余弦调度在稳定期和衰减期的表现](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0009_00-12-51.jpg)

有了 WSD，Chinchilla 式分析变得容易：先跑一条长训练，再回滚 checkpoint 反复衰减，就能沿数据维度做 sweep，同时也可沿模型维度做 sweep。MiniCPM 复现了 Chinchilla 的 method 1 和 method 3，得到看起来合理的 lower envelope 和平滑拟合；但其联合拟合的指数与原始 Chinchilla 不同，论文据此声称需要更多 token。这个结论是否真实并不完全清楚，可能与其拟合方式有关。

![展示模型损失随数据量变化的拟合曲线图，包含拟合公式及最优模型与数据规模比例的计算公式。](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0010_00-14-08.jpg)

![Chinchilla方法3的联合拟合结果，展示了计算量、参数量与损失之间的等高线图及拟合公式。](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0011_00-15-25.jpg)

## DeepSeek：直接拟合超参数 scaling law [15:58 - 22:29]

DeepSeek 的原始 LLM 论文采取另一条路线：不使用 μP 做参数化稳定，而是直接拟合最优批大小和最优学习率随规模变化的规律。具体做法是在多个规模上做大量网格搜索：固定计算量，扫描学习率和批大小，找终端损失最低的点。

![不同计算预算下批量大小与学习率的网格搜索结果热力图，显示泛化误差在较宽参数空间内保持稳定。](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0012_00-16-42.jpg)

跨规模收集最优点后，可以拟合出幂律：

$$\eta_{\text{opt}} = 0.3118 \cdot C^{-0.1250}$$

$$B_{\text{opt}} = 0.2920 \cdot C^{0.3271}$$

其中 $C$ 是非 embedding 训练 FLOPs。批大小的拟合看起来相当合理；学习率的拟合则不那么完美，部分原因是网格位置和量化误差会影响最优点估计。

![学习率与批大小的缩放分析幻灯片，包含拟合公式及两张缩放曲线图](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0013_00-17-59.jpg)

DeepSeek 也用 WSD 风格的学习率来做 Chinchilla 复现，其调度可以描述为 80% + 10% + 10% 的多步/两段衰减。虽然这个两段衰减的变体后来没有广泛流行，但说明 WSD 是有效的。借助 isoflops 扫描，DeepSeek 得到比 MiniCPM 更干净的计算量-损失曲线和 Chinchilla tradeoff，并最终对实际训练的大模型给出不错的损失预测。

![DeepSeek 使用的 WSD 风格学习率调度策略说明及对比图表](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0014_00-20-04.jpg)

![DeepSeek LLM 的性能缩放曲线图，展示验证集损失随训练计算量增加而下降的趋势](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0015_00-21-50.jpg)

两种路线的对比可以概括为：MiniCPM 试图通过 μP 让敏感超参数“不随规模变”；DeepSeek 承认它们会变，但直接拟合它们如何变。

## 近期开源模型的 scaling 实践 [22:34 - 30:25]

此后很多开源模型论文不再详细展示核心 scaling 流程，因为 Chinchilla 式分析和学习率 scaling 已逐渐成为默认技能。

- **Qwen 2.5 / Qwen 3**：对批大小和学习率做 scaling 拟合，Qwen 3 基本沿用 Qwen 2.5 的做法。

![关于 Qwen 2.5 和 Qwen 3 模型在批次大小和学习率等超参数上的缩放定律拟合说明](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0016_00-23-07.jpg)

- **Kimi K2 等 MoE 模型**：做稀疏度 scaling law，观察不同稀疏度下验证损失随 FLOPs 的变化。更稀疏的网络在固定 FLOPs 下损失更好，但存在收益递减；Kimi K2 据此选择约 48 的稀疏度。

![Kimi K2 的稀疏度缩放定律图表，展示不同稀疏度下的验证集损失变化](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0017_00-24-24.jpg)

- **Hunyuan**：同样在切换到 MoE 后做 MoE scaling law，固定稀疏度，最终采用约 96 data points per active parameter 的比例。

- **Llama 3**：做 isoflops 式 scaling，得到略不同但相近的 token/model 比例；还把 log loss 与下游 accuracy 用 sigmoid 关系联系起来，说明损失下降通常会带来下游精度提升。

![LLaMA 3 (2024) 的缩放定律图表，包含 IsoFLOPs 曲线和计算量到下游性能的缩放关系](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0018_00-25-41.jpg)

- **MiniMax-01**：用 scaling law 比较 lightning attention、softmax full attention 和 hybrid 架构，发现它们在缩放行为上大多可比、所需模型大小相似，从而支持在实际系统中采用 hybrid 架构。

![MiniMax-01 (2025) 的架构缩放定律图表，展示损失、模型大小和 token 数与计算量的关系](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0019_00-26-58.jpg)

![总结DeepSeek、miniCPM、Qwen等近期大模型缩放定律配方的幻灯片](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0020_00-29-21.jpg)

问答部分提到一个开放问题：post-training 与预训练 scaling 如何整合仍没有好答案。post-training 可能改变“应该做什么预训练”；目前只有一些关于预训练覆盖度/多样性来预测 post-training 的初步工作，仍很不成熟。[29:26 - 30:25]

## 超参数 scaling 的实证研究：学习率与批大小 [30:29 - 39:21]

课程引用 StepFun 的一项较新研究来讨论学习率和批大小的 scaling。不同论文给出的公式连输入变量都不一致：OpenAI 的临界批大小是终端损失的函数，DeepSeek 用计算量，StepFun 则同时用模型大小和数据量。这些公式不应被当作绝对真理，但其中的实验现象很有学习价值。

![对比不同缩放定律研究中学习率和批次大小公式的表格](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0022_00-32-06.jpg)

实验设计与 DeepSeek 类似：训练大量模型，对学习率和批大小做高分辨率网格搜索，跨模型大小和数据量寻找最优点。高分辨率网格显示，固定数据量和模型时，损失对批大小、学习率近似凸且平滑，说明网格搜索超参数空间是可行的。

![展示预训练损失随Batch Size和Learning Rate变化的凸性曲线及3D损失景观图](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0023_00-34-40.jpg)

关键观察有两个：

1. **最优批大小似乎只依赖总数据量**。不同模型大小的点大致落在同一条趋势线上，在 log-log 坐标下呈幂律，约为 $B_{\text{opt}} \propto D^{0.571}$。
2. **最优学习率同时依赖模型大小和数据量**：模型越大，最优学习率越低；数据越多，最优学习率越高。拟合形式约为 $\eta_{\text{opt}} \propto N^{-0.713}D^{0.307}$。其中“数据越多学习率越高”有些反直觉，也可能比较脆弱，已有其他工作给出相反的数据依赖。

![展示不同参数量下数据集大小与学习率、Batch Size的缩放趋势图](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0024_00-35-58.jpg)

学习率整体上比较鲁棒，这些 scaling law 在一定程度上也能迁移到 MoE（只要控制 active 参数）；但换训练数据后，最优学习率和批大小会偏移，说明具体数字可能与数据配方有关。

![展示MoE模型在不同稀疏度及不同数据集下的验证损失景观等高线图](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0025_00-37-15.jpg)

如果按 Chinchilla 方式缩放，$N$ 和 $D$ 都由计算量驱动，那么学习率会随计算量下降、批大小会随计算量上升，形式类似 DeepSeek 的规律，只是指数不同。问答中有人问应直接套用 scaling law 还是自己做网格搜索；回答是取决于计算预算和需要的精度。在相近计算区间，这些 law 是不错的默认值，但架构、weight decay 等差异可能要求重新验证。scaling law 仍有“vibes”成分：你无法确知别人的实验设置是否与你的足够相似。

## 优化器与 scale：Muon 的故事 [39:21 - 54:44]

NanoGPT speedrun 上，Muon 相比 Adam 取得了显著加速，且单步时间没有明显变慢。但优化器是强 scale 依赖的，小规模上的大增益必须拿到大规模检验。

![不同优化器在NanoGPT上的验证损失随时间变化曲线，以及不同模型规模下的加速比对比图](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0026_00-39-49.jpg)

一项大规模优化器对比研究指出，比较优化器时有很多陷阱：不同优化器需要不同的超参数，甚至可能有不同的 scaling 指数；Adam 如果学习率没调好会显得很差，调好后差距可能消失。weight decay 也需要单独调，否则会严重影响结论。

![展示不同优化器在130M模型上的损失曲线及权重衰减对Lion和AdamW的影响对比图](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0028_00-43-40.jpg)

研究 scaling 时要警惕两个轴：

- **计算量/模型大小轴**：固定 model:data 比例时，模型变大等价于计算量变大。
- **Chinchilla 比例轴**：数据量与模型参数量的比例。有些算法在过参数化（模型大、数据少）时表现好，有些在大 Chinchilla 比例（数据多）时表现好。很多 scaling 研究只扫模型大小而忽略这个比例，这是重要混淆因素。在该优化器对比中，不同 Chinchilla 比例下各优化器的相对差距基本一致，但这并非总是成立。

![展示优化器加速比随模型规模变化及不同优化器在520M模型上的损失缩放对比图](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0029_00-44-57.jpg)

即使算法看起来在多数量级上遵循漂亮幂律，也可能突然失效。例如 Cautious AdamC 加平方根批大小学习率缩放，在很长一段计算量范围内拟合很好，继续放大后却明显变差甚至发散。修复方法包括更仔细的 μP 式参数化和更换优化器。这说明建立可靠的 scaling 外推并不容易。

![展示IsoFLOP抛物线及缩放定律验证图，说明建立缩放定律的困难性](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0030_00-47-31.jpg)

Muon 的核心思想是区分矩阵参数和向量参数。标准 momentum 更新对所有参数一视同仁；Muon 对矩阵参数的动量更新 $B_t$ 做 Newton–Schulz5 近似正交化：

$$B_t = USV^T \longrightarrow UV^T$$

即把所有奇异值拉成 1，再用于参数更新：

$$\theta_t \leftarrow \theta_{t-1} - \eta O_t,\quad O_t = \text{NewtonSchulz5}(B_t)$$

直觉上，Adam 是按坐标大小归一化，使每个坐标步长大致相同；Muon 是在谱范数下让每个方向步长单位化。因此它只适用于矩阵参数；向量参数（如 RMSNorm 参数）仍可用 AdamW。Newton–Schulz 只用矩阵乘法迭代近似正交化，不需要在 GPU 上做 SVD，因而系统上可行。

![Muon优化器算法伪代码及与Adam等优化器的验证损失对比曲线图](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0031_00-50-05.jpg)

Muon 在 NanoGPT speedrun 上很强，但 scaling 研究曾显示其相对 AdamW 的增益随模型变大而减小。不过 Kimi K2 整个模型都用 Muon 训练，并加入了一些防止不稳定的技巧，最终模型表现很好。这说明 Muon 在大规模上是可行、可用的；但 Kimi K2 没有做同规模 ablation，因此还不能断定它在那个规模上一定优于 Adam。

![Muon在不同模型规模下的扩展性研究及Kimi K2训练损失曲线](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0032_00-53-56.jpg)

问答补充：SVD 在 GPU 上不快，但 Muon 用的是 Newton–Schulz 矩阵乘法近似；有人讨论“每层甚至每个参数用不同优化器”的极限方向；实际调参时通常只对最敏感的学习率/批大小做网格，weight decay 等再做单变量局部扫描。

## μP：不变量、推导与经验边界 [54:45 - 75:52]

μP 是一个有些神秘的对象，不同论文和实现对其数学细节并不完全一致，但核心目标清楚：当模型宽度增大时，让最优学习率保持不变。可调的旋钮包括逐层初始化、逐参数学习率，以及残差连接的缩放。

Cerebras GPT 在 0.1–13B 模型上使用 μP 变体，发现 scaling law 拟合更稳定，外推预测更接近实际模型。若想读较易理解的 μP 材料，课程推荐 Jeremy Bernstein 的综述性论文；Greg Yang 的 tensor programs 系列则较难读。

μP 的推导基于两个“物理学家式”的不变量：

- **A1：初始化时激活不随宽度爆炸或消失**。单个激活大致 $O(1)$，因此一个含 $n$ 个隐藏单元的层，其激活向量范数约为 $\sqrt{n}$。
- **A2：一步梯度后激活变化为 $O(1)$**，即网络确实在 feature learning。神经正切核（NTK）情形下这个变化会随宽度消失，不是 μP 想要的。

在深层线性网络、SGD、单样本的简化设定下，权重更新是 rank-one 的，激活变化可拆成三项。希望每一项都与 $\sqrt{n_L}$ 同阶，并假设损失更新为 $O(1)$，通过 Taylor 展开可解出逐层学习率：

$$\eta \propto \frac{n_L}{n_{L-1}} \quad\text{（即 fan-out / fan-in）}$$

对 Adam，则会得到类似 $1/n_{L-1}$（fan-in 倒数）的逐层学习率。

最终 μP 的初始化形式可概括为：

$$\text{init} \sim \frac{1}{\sqrt{n_{L-1}}}\cdot \frac{1}{\sqrt{\text{fan-out}/\text{fan-in}}}$$

当 fan-out/fan-in 接近 1 时，它退化为标准参数化。学习率在标准做法中通常是常数，μP 则设为 fan-out/fan-in；Adam 下用 $1/\text{fan-in}$，于是 fan-in 大的层学习率更小，形成层自适应学习率。

![muP稳定缩放技术的操作列表，包含嵌入输出缩放、残差连接缩放、张量初始化、学习率缩放和LM头缩放的具体公式与参数。](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0004_00-05-08.jpg)

这套推导的高层意义不只是得到一组公式，而是展示了一种方法论：取网络宽度极限，断言不变量，再附加假设，反推出超参数应如何随规模缩放。这与常见的 CS/ML 调参思路很不同。

经验上，独立研究者的压力测试表明：baseline μP 以及 attention projection bias 的 μP 变体在语言模型上迁移良好；Swiglu、RMSNorm 的某些初始化变体、部分优化器改动在理论上不满足 μP，但多数仍能工作。明显会破坏 μP 的因素包括：学习 RMSNorm 的 gain（不过 gain 往往可以去掉而不太伤性能）、使用 Lion 这类依赖梯度符号的优化器、以及较大的 decoupled weight decay。

μP 有用，是控制超参数随规模漂移的工具之一，但不是“银弹”；直接拟合 scaling law 同样是有效路线。课程最后总结：scaling in the wild 很棘手，scaling law 的初始叙事听起来像精确科学，实际外推充满不确定性；人们用它选架构、选优化器、选超参数，但必须尽量采取合理措施提高成功概率，目前还没有一劳永逸的解决方案。[75:53 - 76:48]

## 要点回顾

- 经典 Chinchilla 之后，前沿 scaling 实践分成“用 μP 稳定超参数”和“直接拟合超参数 scaling law”两条路线。
- MiniCPM 用 μP 让最优学习率基本不随模型规模变化；但最优批大小仍随数据和模型规模变化，并呈幂律。
- WSD 学习率由 warm-up、长稳定期和末端快速衰减组成，可复用稳定阶段 checkpoint，大幅降低 Chinchilla 式扫描成本。
- DeepSeek 通过跨规模网格搜索直接拟合 $B_{\text{opt}}(C)$ 和 $\eta_{\text{opt}}(C)$，并用 WSD 复现 Chinchilla，最终对实际大模型给出不错预测。
- 近期开源模型把 scaling 扩展到 MoE 稀疏度（Kimi K2、Hunyuan）、架构选择（MiniMax-01）和 loss→下游精度映射（Llama 3）；核心 Chinchilla 流程已逐渐被视为默认技能。
- StepFun 的高分辨率超参数网格显示损失对批大小/学习率近似凸且平滑；最优批大小主要依赖数据量，最优学习率随模型变大而下降、随数据增多而上升，但具体数字可能依赖数据配方。
- 优化器比较必须同时警惕模型大小轴和 Chinchilla 比例轴；小规模上的优化器增益不一定能外推到大规模。
- Muon 对矩阵参数做 Newton–Schulz 近似正交化，使谱方向步长单位化；它在 NanoGPT speedrun 上增益显著，scaling 研究显示增益随规模减小，但 Kimi K2 已全量使用，证明其在大规模可行。
- μP 通过初始化时激活范数和一步更新后激活变化两个不变量，推导出逐层初始化与学习率缩放；它能控制超参数漂移，但会被 RMSNorm gain、Lion、大 weight decay 等破坏。
- scaling law 的外推在实践中仍有很多“vibes”成分，没有银弹；需要结合 μP、超参数搜索和对实验设置差异的谨慎验证。
