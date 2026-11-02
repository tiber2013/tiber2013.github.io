---
title: "第 6 讲 · 生成式策略：扩散模型与流模型"
description: "本讲解决的是行为克隆最根本的一个失配——从观测到动作的映射不是一个函数而是一个分布（7 自由度机械臂到达同一目标有无穷方式、专家演示本身不一致），因此用 MSE 训练的确定性策略必然被数学推向\"所有模式的平均\"，在雪地里…"
---

# 第 6 讲 · 生成式策略：扩散模型与流模型

> **ETH Zürich 机器人学习 · L06** ｜ 讲者 Oier Mees ｜ 51 分钟 ｜ [原视频](https://www.youtube.com/watch?v=qd6Ldsuu46I)

## 核心结论

本讲解决的是行为克隆最根本的一个失配——从观测到动作的映射**不是一个函数而是一个分布**（7 自由度机械臂到达同一目标有无穷方式、专家演示本身不一致），因此用 MSE 训练的确定性策略必然被数学推向"所有模式的平均"，在雪地里就是直直撞树而不是左绕或右绕。要学出完整分布，就必须回答"怎么表示一个高维多模态分布"，讲师给出三条互补路线：**VAE 一族回答"如何编码分布"**（隐变量、重参数化、VQ-VAE 的离散码本），**扩散模型一族回答"如何从分布中采样"**（前向加噪、预测噪声而非预测图像、DDPM/DDIM 采样、无分类器引导），**流模型一族回答"如何沿分布运输"**（学一个速度场把噪声沿直线运成数据，rectified flow 再把路径拉直）。三条路线都不是纯理论：VQ-VAE 支撑了 GAIA-1、Genie、LaPa、VQ-BeT 的 token 化与潜动作，扩散直接催生 Diffusion Policy 及其后续 Octo 与 adaLN 条件注入，流匹配则被 π0 用作第一个 flow-matching VLA、如今已是主流 VLA 的默认选择。

## 为什么机器人策略需要生成模型 [00:00 - 03:04]

![课程标题页：机器人学习：从基础到基础模型，第六讲生成模型](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0001_00-00-00.jpg)

本讲由 Oier Mees（ETH Zürich / Microsoft，2026-03-23）主讲。开场先把动机收回到第三讲的结论：机器人策略要学的 $p_\theta(x)$，在机器人里就是 $\pi_\theta(a_t \mid s_t)$，而**观测到动作的映射本质上不是一个函数**。原因有两条：

- **随机性（冗余）**：一台 7 自由度机械臂要到达同一个目标状态，存在无穷多种关节配置与轨迹。
- **专家不一致**：人采集演示时本身就有多个模式——今天动作快一点、明天慢一点，这次侧抓（side grasp）、下次顶抓（top-down grasp）。同一个任务、同一个目标，有很多种合法做法。

结论是：策略应当是**一个动作分布**，而生成模型正是用来建模这个分布的工具。

![回顾页：多模态行为，确定性 MSE 行为克隆因模式平均而失败，目标为学习完整分布，配雪地双路径示意图](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0002_00-01-45.jpg)

回顾页把失败模式画得很直白：左图是雪地里绕开同一棵树的两条车辙（两个模式），右图是确定性策略的结局——直直冲进树里。幻灯片原话是 "Deterministic MSE BC policy will fail due to **"averaging"** of modes!"，目标是 "learn the **full distribution**!"。用 MSE 训练确定性策略时，数学上会让预测收敛到所有演示动作的均值，于是策略在两条合法路径的正中间输出一个"平均动作"，正好撞上障碍物。

于是本讲的问题被定义为：**如何表示这个高维、多模态的分布？** 幻灯片给出三大家族，每家从一个不同角度回答：

| 问题 | 方法 |
|---|---|
| How to Encode a Distribution（如何编码分布） | VAEs |
| How to Sample from a Distribution（如何从分布采样） | Diffusion Models |
| How to Transport through a Distribution（如何沿分布运输） | Flow Models |

![生成模型三大流派分类图：VAEs、扩散模型与流模型](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0003_00-02-33.jpg)

## 生成模型的统一定义与潜变量策略 [03:04 - 04:36]

在进入具体架构之前，先统一"生成建模"这个词的含义——不管后面讲的 VAE、扩散、流还是 GAN，底层概念是同一个：

> **生成模型是一个学出来的变换，把简单、可处理的初始分布 $p(z)$ 映射到复杂的数据分布 $p_\text{data}(x)$。**

起点通常取标准高斯噪声，终点是我们真正想建模的分布：幻灯片上画的是"噪声图 → Generative Model → 猫图"，在机器人里对应的则是"噪声 → 机器人动作轨迹"。

![生成模型将初始噪声分布转换为数据分布的示意图](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0004_00-03-24.jpg)

处理多模态的第一种思路，是上一讲模仿学习里已经预演过的**潜变量模型（Latent Variable Models）**：策略输出仍然是高斯，但额外接收一个隐变量输入，因此原则上"可以表示任意分布"。

$$\pi(\mathbf{a} \mid \mathbf{o}, z) = f_\theta(\mathbf{o}, z)$$

其中 $f_\theta$ 被称为**条件解码器（Conditional Decoder）**。隐变量 $z$ 的作用相当于"模式索引"：幻灯片在雪地双路径示意图下方标出 $z_\text{left} \sim \mathcal{N}(0, \mathbf{I})$ 与 $z_\text{right} \sim \mathcal{N}(0, \mathbf{I})$，分别对应"左绕"和"右绕"，从而把"该走哪条"这件事从随机性里显式拿出来。

![潜变量模型与条件VAE的解码器公式及雪景示意图](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0005_00-04-15.jpg)

但上一讲回避了一个问题：**怎么学会把专家动作压缩进这个 $z$**？这正是自编码器要解决的。

## 自编码器：压缩、重建与三个缺陷 [04:36 - 06:40]

![自编码器结构图：输入x经编码器压缩为隐变量z，再由解码器重建为x'，并给出重建损失公式](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0006_00-05-06.jpg)

自编码器的结构非常朴素，两个网络：

- 编码器 $z = f_\phi(x)$：把输入 $x$ 压成紧凑的隐表示 $z$；
- 解码器 $x' = g_\theta(z)$：拿隐码去重建原始输入。

损失就是 L2 重建误差 $\mathcal{L} = \left\| x - g_\theta\left(f_\phi(x)\right) \right\|^2$。不需要任何额外标签，因此是自监督的；中间那个瓶颈迫使编码器只保留数据分布里最重要的特征——因为压低重建误差的唯一途径就是把关键信息挤过瓶颈。

![自编码器隐空间问题示意图：不同类别点簇分散、中间存在无法采样的空白区域](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0007_00-05-57.jpg)

幻灯片 "Issues with Autoencoders" 列出三条限制，第三条对生成任务最致命：

1. **隐空间没有结构**：编码器可以想怎么摆簇就怎么摆，损失里没有任何东西强迫它组织自己。
2. **唯一的正则项就是瓶颈尺寸本身**：$z$ 的维度只限制"能过多少信息"，不规定"信息该怎么组织"。
3. **$z$ 是确定性的，无法采样新数据**：每个输入映射到唯一一点，根本不存在分布，测试时自然也没法生成。

示意图里 $z_1$–$z_2$ 平面上 class A（紫）、class C（绿）、class B（橙）三簇彼此孤立，中间大片区域被标成 "? empty ? can't sample here"——落在空白处就什么都解不出来。

## 变分自编码器与 ELBO [06:40 - 08:47]

![变分自编码器结构图：编码器输出均值μ和方差σ²，采样后经解码器重建，下方给出ELBO损失公式](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0008_00-06-48.jpg)

VAE 一次性解决上面三个问题，关键想法是：**不编码到一个点，而是编码到 $z$ 上的一个分布**。具体做法（也是最流行的形式）是编码器 $q_\phi(z \mid x)$ 输出一个高斯的参数——均值 $\mu$ 与方差 $\sigma^2$——再从该高斯中采样 $z$ 送进解码器 $p_\theta(x \mid z)$。为了让隐空间有结构，再加一个正则项，强迫近似后验贴近高斯先验 $p(z) = \mathcal{N}(0, I)$；幻灯片强调这条约束"迫使隐空间连续、有结构、可采样"。

数学上目标是最大化数据对数似然 $\log p_\theta(x)$，但如第三讲所说它通常不可解，于是改为最大化**证据下界（ELBO）**：

$$\log p_\theta(x) \;\ge\; -\mathrm{KL}\left(q_\phi(z \mid x) \,\middle\|\, p(z)\right) + \mathbb{E}_{q_\phi(z \mid x)}\left[\log p_\theta(x \mid z)\right]$$

- **正则化项**（KL）：隐码 $z$ 离先验 $p(z)$ 有多近？它惩罚偏离先验分布的程度。
- **重建项**（期望）：形式上和自编码器的重建损失一样，只是 $z$ 不再确定——变成"在我可能采出的所有 $z$ 上取平均，解码器还原 $x$ 还原得多好"。

关键收益：现在测试时原则上可以采**任意** $z \sim p(z)$ 并解码出有意义的输出，这才真正成为一个生成模型。

## 重参数化技巧：让采样可微 [08:47 - 13:22]

![重参数化技巧对比图：朴素采样梯度阻断与重参数化后梯度可回传的结构](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0009_00-09-22.jpg)

但随机采样这一步不可微（上一讲策略梯度时也遇到过，只是没展开）。幻灯片 "Reparameterization Trick" 把标题下的问题写成 "Problem: Sampling is non-differentiable"，并用上下两张计算图对比：

- 上图（红色，✗ Naive sampling — gradients cannot flow）：`Encoder outputs μ, σ²` → `Sample z, z ~ N(μ, σ²)` → `Decoder outputs x'` → `Loss`，虚线箭头标出**梯度在 Sample 节点被阻断**。
- 下图（绿色，✓ Reparameterised — gradients flow through μ and σ）：编码器输出 $\mu$ 与 $\sigma^2$，$\sigma$ 由 $\sigma = \sqrt{\sigma^2}$ 得到，与外部噪声 $\epsilon$ 经 $\odot$ 相乘后再与 $\mu$ 相加，随机性被挪到一边。$\epsilon$ 的方框是虚线并标注 "~ N(0,I) / **not learned**"——它没有任何可学习参数。

做法即把 $z$ 写成"参数的确定性函数 + 外部噪声"：

$$z = \mu + \sigma \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

从梯度的视角看，$\mu$ 和 $\sigma$ 就只是普通确定性节点，梯度可以流过去。

![VAE 梯度推导对比：REINFORCE 得分函数估计与重参数化技巧的公式推导](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0012_00-12-46.jpg)

**为什么这样就真的可微了？** 推导页（"Derivation"）指出核心困难：损失里带的是**一个自身依赖 $\phi$ 的分布下的期望**，

$$\nabla_\phi \mathcal{L} = \nabla_\phi \int q_\phi(z \mid x)\, f(z)\, dz$$

处理它有两条路。

**选项 1：得分函数估计器（Score Function Estimator），在强化学习里叫 REINFORCE。** 用上一讲的 log-derivative trick 把梯度搬进积分：

$$\nabla_\phi \mathcal{L} = \mathbb{E}_{z \sim \mathbf{q}_\phi(z \mid x)}\left[f(z)\, \nabla_\phi \log q_\phi(z \mid x)\right]$$

幻灯片右侧的批注是要害：$\nabla_\phi$ **只作用在 $\log q_\phi$ 上**，完全忽略了 $z$ 自身如何随 $\phi$ 变化，因此 $\frac{\partial z}{\partial \phi} = 0$，**方差极高**。直觉上优化器只能"猜"：它知道某个样本好还是坏，却不知道该怎么挪 $\mu$ 和 $\sigma$ 让它变好。

**选项 2：重参数化。** 把 $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$ 代回去，此时 $p(\epsilon)$ 与 $\phi$ 完全无关（幻灯片写作 "Reparam. $z \to \epsilon$, $p(\epsilon)$ independent of $\phi$"），于是可以像上一讲那样用 Leibniz 法则/线性性把梯度搬进积分号，再换回对 $\epsilon$ 的期望：

$$\nabla_\phi \mathcal{L} = \nabla_\phi \int p(\epsilon)\, f\left(\mu_\phi(x) + \sigma_\phi(x) \odot \epsilon\right) d\epsilon
= \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}\left[\nabla_\phi f\left(\mu_\phi(x) + \sigma_\phi(x) \odot \epsilon\right)\right]$$

此时 $\nabla_\phi$ **直接作用在 $f$ 上**，$\frac{\partial z}{\partial \phi} \ne 0$，方差低。按链式法则展开：

$$\nabla_\phi \mathcal{L} \approx \frac{\partial f}{\partial z} \cdot \frac{\partial z}{\partial \phi}, \qquad \frac{\partial z}{\partial \mu_\phi} = 1, \quad \frac{\partial z}{\partial \sigma_\phi} = \epsilon$$

实践含义很直接：**采一个 $\epsilon$、算梯度、结束**。$\frac{\partial f}{\partial z}$ 来自解码器，它告诉我们该把 $z$ 往哪个方向挪，因此也就直接告诉我们编码器的参数会如何影响隐码。

## VAE 换来了什么，又留下什么问题 [13:22 - 15:43]

![变分自编码器（VAE）潜空间特性示意图：连续结构化、可生成、平滑](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0013_00-13-37.jpg)

绕完数学，对比一下隐空间的实际形态：与自编码器那张散乱、有洞的图相比，VAE 的隐空间**连续且有结构**；class A/B/C 三簇被 $\mathcal{N}(0, I)$ 的虚线圆（先验 $p(z)$）包在一起，圆心附近标出一个橙色点 "sample $z \sim p(z)$"。幻灯片给的三条性质是：

- 隐空间连续且有结构；
- **有生成能力**：任意 $z \sim p(z)$ 都能有意义地解码；
- **平滑**：邻近的 $z$ 产生相似的输出。

而且测试时生成**根本不需要编码器**——直接从先验采 $z$，再过解码器即可。

![变分自编码器的两大问题：后验坍缩与先验不匹配，配雪山树木示意图](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0014_00-14-28.jpg)

不过 VAE 有两类常见失效模式：

**后验坍缩（Posterior Collapse）**：解码器容量足够大时，它完全可以自己直接建模 $p(x)$，把你传给它的那个 $z$ 忽略掉。此时模型又退化回一个（无条件的、确定性的）模型，前面所有努力白费。

**先验不匹配（Prior Mismatch）**：更微妙，也更危险。训练时两个模式的隐码来自各自的后验，

$$z_\text{left} \sim q_\phi(z \mid x_\text{left}) = \mathcal{N}\left(\mu_\text{left}, \sigma^2_\text{left}\right), \qquad z_\text{right} \sim q_\phi\left(z \mid x_\text{right}\right) = \mathcal{N}\left(\mu_\text{right}, \sigma^2_\text{right}\right)$$

而测试时是从 $\mathcal{N}(0, \mathbf{I})$ 采样，幻灯片明确列出三种落点：$z_\text{left}$（✓）、$z_\text{right}$（✓）**或两者之间的 gap（✗）**。若落在中间空白处，解码器从未在该区域被训练过（因为没有对应的演示），哪怕数学上隐空间被强制连续，解码器也会**幻觉**出一个完全错误的输出。讲者特别强调这在机器人上是严重问题——你不希望幻觉出不安全的动作。

## VQ-VAE：离散码本与直通估计器 [15:43 - 19:00]

![VQ-VAE 整体架构图，展示编码器、离散码本、解码器及梯度直通路径](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0015_00-16-10.jpg)

缓解先验不匹配的一条路，是把连续高斯隐变量换成**由 $K$ 个可学习向量组成的离散码本（codebook）**，即向量量化 VAE（VQ-VAE）。幻灯片两条要点：用离散码本替换连续高斯 $z$；强制编码器"承诺（commit）"到一个离散码。

工作流程（架构图，取自原论文）：图像经 CNN 编码器得到连续隐表示 $z_e(x)$，**不直接**送进解码器，而是做一次最近邻查表，在 Embedding Space 里找码本中离它最近的向量 $e_k$，把那个离散码 $z_q(x)$ 交给解码器；图中还画出 $q(z|x)$ 的索引网格（1、3、2、53）以及右侧散点图上 $z_e(x)$ 被"吸附"到 $z_q(x)$ 的过程。因为 $z$ 变成了硬离散选择，**既没有 KL 散度要最小化，解码器也没有办法忽略 $z$**，后验坍缩被消掉。出处为 *Neural Discrete Representation Learning*，van den Oord、Vinyals 等（2017）（幻灯片把作者姓印成 "van den Ord"）。

![VQ-VAE 中最近邻码本向量查找公式及直通估计器（STE）的梯度推导](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0016_00-17-01.jpg)

量化在数学上是一次 argmin 查表：

$$z_q(x) = e_k, \qquad k = \arg\min_j \left\| z_e(x) - e_j \right\|_2, \qquad \mathcal{C} = \{e_1, e_2, \ldots, e_K\}$$

老问题又来了：**它不可导**。argmin 是分段常数函数，把 $z_e(x)$ 稍微往左或往右挪一点，最近邻仍是同一个，所以"更新 $z_e(x)$ 后它照样指向同一个邻居"，梯度几乎处处为零。

**直通估计器（Straight-Through Estimator, STE）** 的做法是：反向传播时**假装量化那一步没发生**，把解码器输入端的梯度原样抄给编码器输出端，绕过 argmin：

$$\frac{\partial z_q}{\partial z_e} \approx \mathbf{I} \quad \Longrightarrow \quad \nabla_{z_e} \mathcal{L} \approx \nabla_{z_q} \mathcal{L}$$

实现上写成

$$z_q = z_e + \mathrm{sg}\left[z_q - z_e\right] \quad \Longrightarrow \quad \frac{\partial z}{\partial z_e} = \frac{\partial z_e}{\partial z_e} + \frac{\partial\, \mathrm{sg}\left[z_q - z_e\right]}{\partial z_e} = 1 + 0 = 1$$

其中 $\mathrm{sg}[\cdot]$ 是 stop-gradient：前向是恒等，反向梯度为零。直觉是：$z_q$ 与 $z_e$ 活在同一个嵌入空间里，按定义彼此非常接近，因此"告诉解码器该怎么改 $z_q$ 才能降损失"的那份梯度，对"编码器当初该怎么改 $z_e$"是一个相当好的代理。

## VQ-VAE 的训练损失与两阶段生成 [19:00 - 21:34]

![VQ-VAE 训练阶段的总损失函数，包含重建损失、码本损失和承诺损失](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0017_00-19-34.jpg)

训练目标由三项组成（幻灯片 "VQ-VAE: Training"，Full Loss）：

$$\mathcal{L} = \underbrace{\left\| x - D(z_q) \right\|_2^2}_{\text{Reconstruction}} + \underbrace{\left\| \mathrm{sg}(z_e) - e_k \right\|_2^2}_{\text{Codebook}} + \underbrace{\beta \left\| z_e - \mathrm{sg}(e_k) \right\|_2^2}_{\text{Commitment}}$$

- **重建项**：解码器有没有从量化后的码里恢复出 $x$？通过 STE 联合训练编码器与解码器。
- **码本项**：把码本向量 $e_k$ 拉向编码器输出，学出一份有代表性的"数据词汇表"。
- **承诺项**（权重 $\beta$）：逼编码器留在它所选中码字的附近。这一项是 STE 成立的前提——没有它，$z_e$ 与 $z_q$ 可以差得很远，"代理梯度"就失效了，隐空间也会任意膨胀。

幻灯片末尾抛出下一节的问题：训练只给我们码本 $\mathcal{C}$ 和解码器 $D$，**测试时怎么采出新数据？**

![VQ-VAE生成新数据的四个步骤：训练编码、收集索引、学习自回归先验及推理采样](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0018_00-20-25.jpg)

答案是加**第二个训练阶段**。幻灯片 "VQ-VAE: How to Generate New Data?" 把流程写成四步：

1. 训练阶段：$x \xrightarrow{\text{encoder}} z_e \xrightarrow{\arg\min} e_k \in \mathcal{C} = \{e_1, e_2, \ldots, e_K\}$
2. 对每个训练样本收集**码索引**：$\mathcal{D} = \{k^{(1)}, k^{(2)}, \ldots, k^{(N)}\}$ —— 相当于把整个数据集翻译成一段段离散 token 序列
3. 在码序列上学一个**类别型先验**（例如自回归 Transformer）：
$$p(k_1, \ldots, k_n) = \prod_{i=1}^{n} p(k_i \mid k_1, \ldots, k_{i-1})$$
4. 推理时：I) 采样索引 $k_i \sim p(k_i \mid k_1, \ldots, k_{i-1})$；II) 查表取向量 $e_{k_i} = \mathrm{Lookup}(\mathcal{C}, k_i)$；III) 用第一阶段的解码器生成干净样本 $\hat{x} = D(e_{k_1}, \ldots, e_{k_n})$。

生成的可以是图像，在本讲语境下也可以是机器人动作。

## 三种隐空间对比与 VQ-VAE 的机器人应用 [21:34 - 24:05]

![自编码器、VAE与VQ-VAE的潜在空间结构对比及优缺点总结](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0019_00-22-08.jpg)

幻灯片把三种形式并排对比（三张 $z_1$–$z_2$ 散点图）：

- **Autoencoder**：scattered · gaps · no structure，注为 "no structure → gaps"；
- **VAE**：compact · continuous · sampleable，三簇被 $\mathcal{N}(0, I)$ 圆包住、中间有 sample 点，注为 "structured but continuous"；
- **VQ-VAE**：discrete · explicit · no gaps，每簇被吸附到一个菱形码字 $e_1, e_2, e_3, e_4$，注为 "discrete → no gaps possible"。

下方对勾/叉清单是本节最实用的结论：

- ✓ 修复了后验坍缩；✓ 没有先验不匹配；✓ 离散、可控、压缩效果好。
- ✗ **码本坍塌（codebook collapse）**：编码器收敛到只用一小撮码字，大量码全程未被使用——这也是如今有人改用最有限标量量化 FSQ（Finite Scalar Quantization）这类更稳健替代方案的原因（本讲不展开）。
- ✗ 必须**事先决定码本大小 $K$**。
- ✗ 重建仍用 MSE，**输出层面的 mean-seeking 依旧存在**（图像上就还是模糊脸）。

![VQ-VAE在机器人学习中的应用：图像/视频标记化、潜在动作学习及动作标记化](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0020_00-22-59.jpg)

一句话总结：**VQ-VAE 把连续信号变成离散 token，让 Transformer 能在统一的 token 空间里同时推理观测、动作与行为。** 幻灯片 "Applications of VQ-VAEs in Robot Learning" 按三栏列出代表作（含图上标注的完整引用信息）：

- **图像与视频 token 化**：GAIA-1: A Generative World Model for Autonomous Driving（Hu et al., 2023）、Cosmos World Foundation Model Platform for Physical AI（Agarwal et al., 2025）。
- **潜动作学习（Latent Action Learning）**：Latent Action Pretraining from Videos（Ye et al., 2024，即 LaPa）、Genie: Generative Interactive Environments（Bruce et al., 2024），以及讲者自己的工作 What Matters in Language Conditioned Robotic Imitation Learning over Unstructured Data（Mees et al., 2022，把"从玩耍中学到的"潜规划量化成码）。这一路的共同套路是用 VQ-VAE 量化游戏画面或第一视角人类视频中帧间的运动，再把这些潜码迁移到机器人上。
- **动作 token 化**：VQ-BeT: Behavior Generation with Latent Actions（Lee et al., 2024），把连续的机器人动作离散化，供 VLA 一类模型使用。

## 扩散模型：前向加噪与反向去噪 [24:05 - 26:54]

![扩散模型前向加噪与反向生成过程的示意图，包含从清晰猫图到纯噪声的演变](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0021_00-24-05.jpg)

扩散模型换了一个提问方式：与其学一个带瓶颈的压缩表示再解码，**能不能完全不设瓶颈、直接对连续变量 $x$ 的完整分布建模？** 幻灯片同时点明动机——VQ-VAE 因为用 MSE，输出仍有 mean-seeking。

扩散有两个过程：

- **前向过程（Forward Process）**：从干净样本 $x_0$（图里那张猫的照片）出发，逐步迭代加高斯噪声，直到所有信息都被抹掉，得到 $x_T$。
- **生成式反向过程（Generative Backward Process）**：训练一个网络逐步去噪，从噪声里把干净图像找回来。

这样拆分的意义在于：不再要求模型"一步猜出整张图"，而是让它做**许多次小的增量修正**——这比一次性解决整个问题好学得多。

![扩散模型前向过程公式推导，标注了信号衰减、噪声缩放及重参数化技巧](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0022_00-26-23.jpg)

前向过程的公式把下一步写成当前状态与随机噪声的加权和：

$$x_{i+1} = \underbrace{\sqrt{1 - \beta_{i+1}}}_{\text{Signal Decay}}\, x_i + \underbrace{\sqrt{\beta_{i+1}}}_{\text{Noise Scale}}\, \epsilon_i, \qquad \epsilon_i \sim \mathcal{N}(0, \mathbf{I})$$

- 第一项是**信号衰减**：每一步都把当前样本轻微收缩一点，往零拉近一些；
- 第二项是**噪声缩放**，其中 $\beta_i$ 称为**噪声调度（noise schedule）或方差调度**，是一个超参数，精确规定每一步注入多少噪声；
- 箭头标注 "Reparametrization Trick!"：噪声样本又被写成"当前状态的确定性函数 + 一个固定的噪声变量"，它的好处再过几张幻灯片就会兑现。

## 反向过程与训练目标：从 ELBO 到简化损失 [26:54 - 30:22]

![扩散模型反向过程原理，说明通过预测添加的噪声来近似还原图像](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0023_00-27-14.jpg)

反向过程原则上要学 $p(x_{i-1} \mid x_i)$，但这是**不可解**的：能产生某张噪声图的干净图有无穷多张，无从判断哪一张才是"真的"。

让扩散模型真正可训练的关键洞见是：**不要预测上一张图像 $x_{i-1}$**（那是个病态问题），**去预测被加进去的那份噪声**：

$$x_{i-1} \approx x_i - \epsilon_\theta(x_i, i)$$

网络接收噪声图像和时间步 $i$，输出该步加入的噪声。为什么这是个好得多的学习问题？因为噪声本身就是标准高斯的采样，是一个**干净、容易的回归目标**；而且前向过程里加了多少噪声我们完全知道，于是可以直接用预测噪声与真实噪声之间的 MSE 来训练。每一步去噪都只是一个小规模修正，不必一次到位。

![扩散模型训练目标：ELBO分解为三项及简化损失函数](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0024_00-28-30.jpg)

训练目标（幻灯片 "Diffusion: Objective"）与 VAE 是**同一个 ELBO**，区别只在于把一步换成 $T$ 个去噪步，并且 **$q$ 是固定的、不需要学习**：

$$\log p_\theta(x_0) \ge \underbrace{-D_{\mathrm{KL}}\left(q(x_T \mid x_0) \,\|\, p(x_T)\right)}_{\text{Prior Regularization}} + \underbrace{\sum_{i=2}^{T} -D_{\mathrm{KL}}\left(q(x_{i-1} \mid x_i, x_0) \,\|\, p_\theta(x_{i-1} \mid x_i)\right)}_{\text{Denoising Matching}} + \underbrace{\log p_\theta(x_0 \mid x_1)}_{\text{Reconstruction}}$$

- **先验正则项**：保证 $T$ 步加噪之后的结果看起来像高斯先验——这一条在前向过程中**由构造自动满足**，实践里不需要学。
- **去噪匹配项**（中间的大求和）：每一步上，学出来的反向过程与真实反向过程匹配得多好。
- **重建项**：能否恢复出原始样本。

由于前向与反向分布都是**方差相同的高斯**，它们之间的 KL 散度化简为两个均值之间的 L2 距离，有闭式解：

$$D_{\mathrm{KL}} \propto \left\| \tilde{\mu}_i - \mu_\theta(x_i, i) \right\|^2 \propto \left\| \epsilon - \epsilon_\theta(x_i, i) \right\|^2$$

于是 *Denoising Diffusion Probabilistic Models*（Ho et al., 2020）把 ELBO 里那些复杂的加权系数全部丢掉，得到极简目标：

$$\mathcal{L}_\text{simple} = \mathbb{E}_{i \sim \mathcal{U}(1,T),\, \epsilon \sim \mathcal{N}(0, \mathbf{I})}\left[\left\| \epsilon - \epsilon_\theta(x_i, i) \right\|^2\right]$$

## 高效训练：一步"瞬移"到任意噪声水平 [30:22 - 32:14]

![扩散模型高效训练：前向过程代数展开与单步跳跃采样推导](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0025_00-30-38.jpg)

$\mathcal{L}_\text{simple}$ 里有个隐藏麻烦：公式要求我们**直接拿到任意噪声水平 $x_i$**。如果老老实实按前向过程一步步走，从 $x_0$ 走到第 500 步就是 O(T) 的开销。幻灯片 "Diffusion: Efficient Training" 给出完整的代数展开：

1. 为方便起见定义**信号保持（signal retention）** $\alpha_i = 1 - \beta_i$（纯粹为了代数方便）。
2. 展开前两步：$x_1 = \sqrt{\alpha_1} x_0 + \sqrt{1 - \alpha_1}\,\epsilon_0$，$x_2 = \sqrt{\alpha_2} x_1 + \sqrt{1 - \alpha_2}\,\epsilon_1$。
3. 把 $x_1$ 代入 $x_2$：
$$x_2 = \sqrt{\alpha_2}\left(\sqrt{\alpha_1} x_0 + \sqrt{1 - \alpha_1}\,\epsilon_0\right) + \sqrt{1 - \alpha_2}\,\epsilon_1 = \sqrt{\alpha_1 \alpha_2}\, x_0 + \sqrt{\alpha_2(1 - \alpha_1)}\,\epsilon_0 + \sqrt{1 - \alpha_2}\,\epsilon_1$$
4. 关键性质：两个噪声项来自**相互独立**的高斯，而两个高斯之和仍是高斯，$\mathcal{N}(0, \sigma_a^2 I) + \mathcal{N}(0, \sigma_b^2 I) = \mathcal{N}(0, (\sigma_a^2 + \sigma_b^2) I)$；配合 $\alpha_2(1-\alpha_1) + (1-\alpha_2) = 1 - \alpha_1 \alpha_2$，两个噪声可以塌缩成单个噪声样本 $\bar{\epsilon}$：
$$x_2 = \sqrt{\alpha_1 \alpha_2}\, x_0 + \sqrt{1 - \alpha_1 \alpha_2}\, \bar{\epsilon}$$
5. 推广到 $i$ 步，信号系数变成所有 $\alpha$ 的乘积，记作 $\bar{\alpha}_i$：
$$\text{Signal Coefficient} = \sqrt{\alpha_i \alpha_{i-1} \cdots \alpha_1} = \sqrt{\textstyle\prod_{j=1}^{i} \alpha_j} = \sqrt{\bar{\alpha}_i}$$

于是得到最终形式（幻灯片用方框框出）：

$$x_i = \sqrt{\bar{\alpha}_i}\, x_0 + \sqrt{1 - \bar{\alpha}_i}\, \bar{\epsilon}$$

它允许我们**从 $x_0$ 一步"瞬移"到任意噪声水平 $x_i$，复杂度 O(1)**——训练时只需随机抽一个 $i$ 即可。

## DDPM 采样：预测均值 + Langevin 噪声 [32:14 - 34:00]

![DDPM采样算法伪代码及去噪步骤公式分解](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0026_00-32-20.jpg)

训练完之后怎么生成？幻灯片 "Diffusion: DDPM Sampling" 给出算法：

1. 采样高斯噪声 $x_T \sim \mathcal{N}(0, I)$
2. 对 $i = T, \ldots, 1$（旁注：**Very slow! 图像生成里 T = 1000!**）
   - I. 采样 $z \sim \mathcal{N}(0, I)$，若 $i > 1$；否则 $z = 0$
   - II. 去噪
$$x_{i-1} = \underbrace{\frac{1}{\sqrt{\alpha_i}}\left(x_i - \frac{1 - \alpha_i}{\sqrt{1 - \bar{\alpha}_i}}\, \epsilon_\theta(x_i, i)\right)}_{\text{Predicted Mean}} + \underbrace{\sigma_i z}_{\text{Stochastic (Langevin) Term}}, \qquad \sigma_i = \sqrt{\beta_i}\ \text{（典型取值）}$$
3. 返回 $x_0$

第一项是**确定性去噪**：减去按调度缩放过的预测噪声再重新归一化，朝"我们认为 $x_0$ 所在的方向"移动。第二项看起来自相矛盾——明明在去噪，为什么又往里加噪声？这一点微妙但重要：如果整个过程纯确定性，那么从不同 $x_T$ 出发的样本都会**朝同一个均值坍缩**；注入噪声是为了**维持样本多样性**，保证我们探索的是整个分布，而不是永远生成同一个平均输出。

幻灯片还补了两条理解框架：DDPM 是**随机微分方程（SDE）的离散版本**；DDPM 的去噪 = 朝高密度方向走的 score step（$\nabla \log p$）+ Langevin 噪声。代价是慢——上千步去噪对图像可以接受，对讲究实时控制的机器人就很吃力了。

## DDIM：确定性采样与少步数推理 [34:00 - 35:52]

![DDIM 采样算法步骤及去噪与重加噪公式推导](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0027_00-34-53.jpg)

DDIM（*Denoising Diffusion Implicit Models*，Song et al., 2020）通过**解耦训练与推理的去噪迭代次数**解决速度问题。算法（幻灯片 "Diffusion: DDIM Sampling"）：

1. 采样高斯噪声 $x_T \sim \mathcal{N}(0, I)$
2. 定义时间步的**子集** $\tau = \{t_1, t_2, \ldots, t_K\}$（例如只用 20 步，而不是 1000 步）
3. 对 $i = K, K-1, \ldots, 1$（约定 $x_i$、$\overline{\alpha_i}$ 分别表示 $x_{t_i}$ 与 $\overline{\alpha_{t_i}}$）：
   - I. **Denoise**：用上一步"瞬移"公式的逆，从当前含噪状态估出干净样本
$$\widehat{x_0} = \frac{x_i - \sqrt{1 - \overline{\alpha_i}}\, \epsilon_\theta(x_i, i)}{\sqrt{\overline{\alpha_i}}}$$
   - II. **Deterministic Re-Noise**：把它重新加噪到子集里的下一个时间步
$$x_{i-1} = \sqrt{\overline{\alpha_{i-1}}}\, \widehat{x_0} + \sqrt{1 - \overline{\alpha_{i-1}}}\, \epsilon_\theta(x_i, i)$$
4. 返回 $x_0$

两条箭头批注点明要害：$\alpha_{i-1}$ 取自 $\tau$，**不必是相邻的时间步**，这正是"能跳步"的原因；而第二个式子里**没有噪声项**——与 DDPM 每步都回灌噪声不同，DDIM 是确定性的。于是推理时可以用少得多的去噪步数产出干净图像或机器人动作。

幻灯片下方的三条总结：DDIM 解耦了训练与推理的迭代次数；DDPM 是马尔可夫的，而 DDIM 定义了一个**非马尔可夫的前向过程，其边缘分布与 DDPM 完全相同**；DDIM 离散化的是**概率流 ODE**（probability flow ODE）。

## 条件生成与无分类器引导 CFG [35:52 - 38:44]

![从无条件生成过渡到条件生成的示意图](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0028_00-36-36.jpg)

到目前为止讲的全是**无条件生成**：从数据的边缘分布 $p(x)$ 里采样，对最终出来什么毫无控制。实践中这几乎从来不是我们要的——图像生成要按文本提示词出图，机器人则要按任务行动。修法听起来很简单：把条件信息 $\mathbf{c}$ 作为噪声预测网络的额外输入即可，训练流程完全不变：

$$\epsilon_\theta(x_t, t, \mathbf{c})$$

幻灯片用的例子正是那句经典的 "A photorealistic image of an astronaut riding a horse"：噪声图 + 这段文本一起进入生成模型，输出宇航员骑马的图像。

![无分类器引导（CFG）的原理与核心公式](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0029_00-37-27.jpg)

但条件注入有个隐疾（幻灯片 "Classifier-Free Guidance (CFG)"）：**模型可能干脆忽略条件**，照样做无条件去噪——和 VAE 里解码器忽略 $z$ 是同一类问题，因为无条件往往是更容易的优化路径。文献里最早的解法是 **classifier guidance**：额外训练一个分类器 $p(c \mid x_t)$，而且要在**每一个噪声级别**上都能用，代价相当高。

如今更流行的是**无分类器引导（CFG，Ho & Salimans, 2022 *Classifier-Free Diffusion Guidance*）**，用一个更优雅的技巧同时解决：

- **训练时**：以某个概率**随机丢弃条件信息**（把条件替换成 null token 之类），迫使同一个网络同时学会条件生成与无条件生成。
- **推理时**：把两份预测混合起来——

$$\hat{\epsilon} = \underbrace{\epsilon_\theta(x_t, t, \emptyset)}_{\text{Unconditional}} + w \underbrace{\left(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset)\right)}_{\text{Guidance} = \text{Conditional} - \text{Unconditional}}$$

括号里那一项是**条件把样本在噪声空间里推的方向**；$w$ 是**引导尺度（guidance scale）**，控制沿该方向走多狠，它对应的是"条件保真度 vs 多样性"的权衡。图像生成里的经验是：$w$ 调大能得到更锐利、更贴合提示词的图像，但代价是多样性和变化减少。

## 案例研究：Diffusion Policy [38:44 - 41:52]

![Diffusion Policy案例研究幻灯片，包含去噪动作序列的核心思想、DDPM训练与DDIM推理公式，以及机械臂操作红色T型物体的演示图](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0030_00-39-09.jpg)

Diffusion Policy（Cheng Chi 等，2023；讲者顺带预告 Chi 正是当日课程的客座讲者）把上面这套数学第一次完整地搬到真实机器人系统上。核心洞见其实很朴素：**为图像生成推导的一切，原封不动适用于动作生成**——把像素换成机器人动作，去噪的对象不再是图像，而是动作序列 $A_t$。

幻灯片给出的两条公式对应"训练用 DDPM、推理用 DDIM"：

$$\mathcal{L} = \mathrm{MSE}\left(\epsilon^k,\, \epsilon_\theta\left(O_t, A_t^0 + \epsilon^k, k\right)\right), \qquad A_t^{k-1} = \alpha\left(A_t^k - \gamma\, \epsilon_\theta\left(O_t, A_t^k, k\right) + \mathcal{N}(0, \sigma^2 I)\right)$$

关键在于**观测被直接注入噪声预测网络作为条件**：当前相机图像、机器人状态等都是 $O_t$ 的一部分。与先前工作相比，最漂亮的一点是它**完全不需要预测未来状态**（不用去预测未来的图像），因此效率高得多——策略直接输出以当前观测为条件的动作序列。配图是机械臂推动桌面上红色 T 形积木的演示。

![Diffusion Policy架构图，展示图像观测序列输入、动作序列输出、扩散策略一般公式，以及基于CNN和Transformer的两种网络结构](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0031_00-40-51.jpg)

架构页（"Case Study: Diffusion Policy"）补充三条工程要点：

- **$O_t$ 只在 $K$ 次去噪之前计算一次**，不必每步重算——这对效率至关重要。
- 噪声预测网络用 **1D temporal U-Net**，把动作序列当作一维时间信号处理；观测条件通过 **FiLM（Feature-wise Linear Modulation，特征线性调制）** 注入，即根据观测嵌入对 U-Net 各层的激活做缩放与平移（图中 $a \cdot x + b$）。
- 论文同时给出 **Transformer 版本**，用 **cross-attention** 让动作去噪过程注意观测 token。

左侧的输入/输出示意还标出观测窗口 $O_{t-2}, O_{t-1}, O_t, O_{t+4}$、机器人位姿（Robot Pose）、动作序列 $A_t$ 与预测时域 $T_p$，以及从 $A_t^K$（噪声）经 $A_t^3$ 到 $A_t^0$（干净动作）的去噪链条。幻灯片最后一条结论是：**它能很好地处理演示中的多模态**。

![同一标题的幻灯片上叠加了文本框，列出四个要点：Specialist Models、Single Task Policies、Not Scalable、Sensitive to Hyperparams。](/coursenotes/eth-robot-learning-2026/l06-generative-policies/kf_0032_00-41-42.jpg)

总体评价是：真实机器人系统上效果确实好、进展巨大，但它有一个结构性挑战——**它们是专家型模型（specialist models）**。幻灯片 "What's Missing?" 用四条列出：Specialist Models、Single Task Policies、Not Scalable、Sensitive to Hyperparams。也就是说，单任务上表现优异，但要扩成通才或多任务策略就更难；尤其是原始 Diffusion Policy 论文里的 Transformer 版本，相当脆弱、对超参数敏感。

## 扩散策略的后续改进：Octo 与 adaLN 条件注入 [41:52 - 43:41]

针对上面的局限，讲者举了两个后续方向（此段无关键帧，内容依据转写）：

- **Octo**：在 **800,000 条机器人轨迹**上训练一个高容量 Transformer 主干，并引入 **readout tokens** 来压缩观测与任务信息，再接上一个**条件扩散解码头**，该头去注意这些汇总了前序信息的 readout token。这样既能用**自然语言**条件，也能用**目标图像**条件。（讲者说明后面讲通才策略时还会展开。）
- **改用 adaLN 注入条件的 block 式策略**（转写文本此处名称不清，讲者读作 "IT block policies"）：架构上的关键创新是**把用于条件注入的 cross-attention 换成自适应层归一化 adaLN-Zero**（adaptive layer normalization），这一设计借自图像生成里的 **Diffusion Transformer** 文献。具体做法是不再让观测 token 与动作 token 做完整交叉注意力，而是把条件信息**直接调制每个 Transformer block 内部 layer norm 的缩放与平移参数**。结果是训练更稳定、更能随规模扩展，并且能够处理**真正长程的灵巧操作任务**，例如切寿司。

## 流匹配：学习速度场而非噪声 [43:41 - 46:16]

第三个、也是最后一个框架：**流匹配（Flow Matching）**（本节及其后两节均无关键帧覆盖，内容依据转写）。

前面两族模型一直在用复杂的噪声调度"摧毁"数据。流匹配反问：**为什么不在噪声与数据之间直接画一条直线？** 于是要学的不再是"预测噪声"，而是一个**速度场（velocity field）**，它把样本沿连续路径从噪声**运输**到数据。

- **路径选择**：直线（linear/straight-line）插值。约定 $t = 0$ 处是纯噪声、$t = 1$ 处是干净数据，即 $x_t = (1 - t)\,\epsilon + t\,x_1$，其中 $\epsilon \sim \mathcal{N}(0, I)$、$x_1 \sim p_\text{data}$。好处是**没有任何复杂噪声调度需要调参**，就是一条确定的直线路径。
- **回归目标**：真值速度就是那个从噪声直接指向数据的向量 $u_t = x_1 - \epsilon$（沿直线为常量）；网络 $v_\theta(x_t, t)$ 在任意时刻 $t$ 上拟合它，损失形如
$$\mathcal{L}_\text{FM}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0,1),\, \epsilon \sim \mathcal{N}(0, I),\, x_1 \sim p_\text{data}}\left[\left\| v_\theta(x_t, t) - (x_1 - \epsilon) \right\|^2\right]$$
- **测试时**：不再以概率意义"去噪"，而是**积分**——从噪声出发，用一个 ODE 求解器（例如欧拉步）沿学到的速度场把样本推向数据：$x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t)$。

**与扩散的关系**：如果取线性调度，流匹配与扩散其实是**等价的**——扩散预测噪声、流匹配预测速度，两者之间只差一个简单的重参数化。但一般而言，流匹配的直线路径**更容易学、需要的积分步数更少**，也没有复杂噪声调度这类超参数，这正是它逐步取代扩散、成为图像生成与机器人学习默认框架的原因。

## 整流流与机器人落地：π0 [46:16 - 48:55]

流匹配给出的路径已经比扩散直得多，但仍有一个微妙问题：**当我们随机地把噪声样本与数据样本配对时，连接它们的直线可能彼此交叉**。轨迹一旦交叉，同一点上速度场就必须同时表示多个冲突的方向，于是学到的场被迫**弯曲**；场一变弯，积分就又慢又需要更多步。

**Rectified Flow** 给出的解法是一个迭代过程：

1. 先正常训练一个流匹配模型；
2. 用它来**生成耦合的噪声–数据对**：采一个噪声，跑一遍学到的流，得到与之对应的数据点——这样 $x_0$（噪声）与 $x_1$（数据）就被这个流向量天然绑定成一对；
3. **在这些配对上重新训练**。由于配对是由流本身决定的，轨迹互相解开、交叉被消除，路径明显更直（讲者展示的可视化对比了原始流匹配与整流流的路径）。

机器人侧的落地是 **π0（Pi-Zero，Physical Intelligence）**：据讲者所知，这是**第一个采用流匹配的模型**——它把一个 VLM 微调成通过流匹配输出动作，因此是**第一个 flow-matching VLA**；这一选择如今已成为绝大多数 VLA 的默认标准。讲者特别点出一个不错的技术细节：训练时**不**均匀采样流的时间 $t$，而是从一个 **shifted beta 分布**采样，把采样权重集中到**更噪、更难的时段**——那里正是动作预测最困难、模型需要预测更多信息的地方。VLA 与生成式策略留到第 9 讲再细谈。

## 本讲总结：三条路线各自回答了什么问题 [48:55 - 51:03]

收束时回到那张三栏分类图。三族生成模型，各自回答"如何表示复杂数据分布"的一个不同侧面：

- **潜变量模型 / 自编码器**：靠重建拿到压缩表示；用**重参数化**把它升级为真正概率化的隐空间；再用 **VQ-VAE** 把隐空间离散成一份学出来的码本。
- **扩散模型**：**DDPM** 教我们迭代地往数据里加高斯噪声、再逐步去噪；**DDIM** 通过离散化一个概率流 ODE 把它变快、变确定；**无分类器引导**通过混合条件与无条件预测来操纵生成方向。
- **流匹配**：学一个速度场，沿**直线路径**把噪声运输成数据；**rectified flow** 再把路径进一步拉直，从而用更少的积分步完成生成。

对机器人而言的关键要点（key takeaway）是：我们从"行为与动作预测本质上是多模态的"这个简单观察出发，已经看到用 MSE 回归的确定性策略必然失败；而今天讲的从 VAE 到扩散再到流匹配这一整套东西，正是我们最新、最强那批机器人策略底层实际使用的工具。

## 要点回顾

- **动机**：观测→动作不是函数而是分布（7 自由度冗余 + 专家不一致）；确定性 MSE 行为克隆会被推向模式平均，雪地示例里就是撞树。目标是学出**完整分布**。
- **统一视角**：生成模型 = 学一个变换，把易处理的 $p(z)$（通常是标准高斯）映到 $p_\text{data}(x)$；在机器人里对应 $\pi_\theta(a_t \mid s_t)$。三族分别对应"编码分布 / 从分布采样 / 沿分布运输"。
- **潜变量策略**：$\pi(\mathbf{a} \mid \mathbf{o}, z) = f_\theta(\mathbf{o}, z)$，用 $z_\text{left}$、$z_\text{right}$ 索引不同模式；但"怎么学会把动作压进 $z$"要靠自编码器。
- **自编码器**：$z = f_\phi(x)$、$x' = g_\theta(z)$、$\mathcal{L} = \|x - g_\theta(f_\phi(x))\|^2$；三个缺陷是无结构、只有瓶颈尺寸做正则、$z$ 确定因而无法采样。
- **VAE**：编码器输出 $\mu, \sigma^2$，最大化 ELBO $\log p_\theta(x) \ge -\mathrm{KL}(q_\phi(z|x)\|p(z)) + \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$；隐空间连续、结构化、平滑，测试时只需先验采样 + 解码器。
- **重参数化**：$z = \mu + \sigma \odot \epsilon$，$\epsilon \sim \mathcal{N}(0,I)$ 无可学习参数；相比 REINFORCE（$\nabla_\phi$ 只作用于 $\log q_\phi$、$\partial z/\partial\phi = 0$、高方差），重参数化让 $\nabla_\phi$ 直接作用于 $f$，$\partial z/\partial\mu_\phi = 1$、$\partial z/\partial\sigma_\phi = \epsilon$，低方差。
- **VAE 的两大失效**：后验坍缩（大解码器绕过 $z$ 直接建模 $p(x)$）；先验不匹配（测试时从 $\mathcal{N}(0,I)$ 采到训练从未覆盖的 gap → 幻觉，机器人场景下意味着不安全动作）。
- **VQ-VAE**：用 $K$ 个可学习码字组成的码本 $\mathcal{C}$ 替换连续 $z$，$z_q(x) = e_k,\ k = \arg\min_j \|z_e(x) - e_j\|_2$；argmin 不可导 → 直通估计器 $z_q = z_e + \mathrm{sg}[z_q - z_e]$，前向恒等、反向把解码器梯度抄给编码器。
- **VQ-VAE 损失**：重建 $\|x - D(z_q)\|_2^2$ + 码本 $\|\mathrm{sg}(z_e) - e_k\|_2^2$ + 承诺 $\beta\|z_e - \mathrm{sg}(e_k)\|_2^2$；承诺项保证 $z_e$ 与 $z_q$ 足够近，STE 的代理梯度才站得住。
- **VQ-VAE 生成**：两阶段——收集码索引序列，再学一个自回归先验 $p(k_1..k_n) = \prod_i p(k_i \mid k_{<i})$，推理时"采样索引 → 查表 → 解码"。
- **VQ-VAE 的得与失**：✓修复后验坍缩、✓无先验不匹配、✓离散可控；✗码本坍塌（→ 更稳健的 FSQ）、✗要预先定 $K$、✗MSE 导致输出仍 mean-seeking。应用：GAIA-1、Cosmos（视频 token 化）、LaPa、Genie、Mees et al. 2022（潜动作）、VQ-BeT（动作 token 化）。
- **扩散**：前向 $x_{i+1} = \sqrt{1-\beta_{i+1}}x_i + \sqrt{\beta_{i+1}}\epsilon_i$（信号衰减 + 噪声缩放，$\beta_i$ 为噪声调度）；反向不预测图像而**预测噪声** $x_{i-1} \approx x_i - \epsilon_\theta(x_i,i)$，因为噪声是标准高斯、是干净的回归目标。
- **目标函数**：与 VAE 同一个 ELBO 但展开到 $T$ 步且 $q$ 固定；同方差高斯的 KL 化简为均值差 L2，丢掉权重即得 $\mathcal{L}_\text{simple} = \mathbb{E}_{i,\epsilon}\| \epsilon - \epsilon_\theta(x_i,i)\|^2$（Ho et al., 2020）。
- **高效训练**：令 $\alpha_i = 1-\beta_i$、$\bar{\alpha}_i = \prod_{j\le i}\alpha_j$，利用"独立高斯之和仍为高斯"得到 $x_i = \sqrt{\bar{\alpha}_i}x_0 + \sqrt{1-\bar{\alpha}_i}\bar{\epsilon}$，一步瞬移到任意噪声水平，O(1)。
- **DDPM**：$x_{i-1} = \frac{1}{\sqrt{\alpha_i}}(x_i - \frac{1-\alpha_i}{\sqrt{1-\bar\alpha_i}}\epsilon_\theta(x_i,i)) + \sigma_i z$，$\sigma_i = \sqrt{\beta_i}$；预测均值 + 随机 Langevin 项，加噪是为了防止样本坍到均值、维持多样性；DDPM 是 SDE 的离散版，慢（图像 $T=1000$）。
- **DDIM**：在时间步子集 $\tau$ 上先估 $\widehat{x_0}$（瞬移公式求逆）再确定性地重加噪到 $\overline{\alpha_{i-1}}$，**无噪声项**、可跳步；非马尔可夫前向但边缘分布与 DDPM 相同，离散化的是 probability flow ODE（Song et al., 2020）。
- **条件与 CFG**：把 $c$ 作为额外输入 $\epsilon_\theta(x_t,t,\mathbf{c})$，但模型可能忽略条件；classifier guidance 要额外训 $p(c|x_t)$ 且每个噪声级别都要，昂贵；CFG 训练时随机丢弃条件、推理时 $\hat{\epsilon} = \epsilon_\theta(x_t,t,\emptyset) + w(\epsilon_\theta(x_t,t,c) - \epsilon_\theta(x_t,t,\emptyset))$，$w$ 是"保真度 vs 多样性"的旋钮。
- **Diffusion Policy**：把像素换成动作序列去噪，DDPM 训练 + DDIM 推理，观测 $O_t$ 作为条件注入且只算一次；网络为 1D temporal U-Net + FiLM，或 Transformer + cross-attention；能处理演示多模态、无需预测未来状态；但属于专家模型、单任务、不易扩展、超参敏感。后续：Octo（80 万轨迹 + readout token + 条件扩散头）、以 adaLN-Zero 取代 cross-attention 的 block 式策略（更稳、更能 scale，可做切寿司这类长程灵巧任务）。
- **流匹配**：不预测噪声而预测**速度场**，沿 $t=0$ 噪声到 $t=1$ 数据的**直线**路径运输；测试时用 ODE 求解器（如欧拉步）积分；线性调度下与扩散等价（只差一个重参数化）；直线路径更好学、步数更少、无需噪声调度超参。
- **Rectified Flow 与 π0**：随机配对会让直线交叉、迫使速度场弯曲 → 先训一个流、用它生成耦合的噪声–数据对、再在这些配对上重训，轨迹解缠、路径更直。π0（Physical Intelligence）把 VLM 微调成用流匹配出动作，是第一个 flow-matching VLA，并改用 **shifted beta 分布**采样时间 $t$ 以侧重更难的噪声时段。
- **一句话**：从 VAE 到扩散到流匹配，这套生成模型工具正是当前最强机器人策略的底座。
