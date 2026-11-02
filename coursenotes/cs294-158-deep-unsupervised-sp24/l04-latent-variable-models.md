---
title: "第 4 讲 · 潜变量模型与变分自编码器"
description: "潜变量模型引入低维隐变量 z，用 pZzpx|z 解释观测 x，目标是最大化边缘似然 px=z pZzpx|z。该求和/积分通常不可精确计算，因此需要重要性采样或变分后验 qz|x 来构造下界。摊销推断用编码器网络从 x…"
---

# 第 4 讲 · 潜变量模型与变分自编码器

> **CS294-158 深度无监督学习 · L04** ｜ 讲者 Pieter Abbeel ｜ 129 分钟 ｜ [原视频](https://www.youtube.com/watch?v=NlIqjtbjjRE)

## 核心结论

潜变量模型引入低维隐变量 \(z\)，用 \(p_Z(z)p_\theta(x|z)\) 解释观测 \(x\)，目标是最大化边缘似然 \(p_\theta(x)=\sum_z p_Z(z)p_\theta(x|z)\)。该求和/积分通常不可精确计算，因此需要重要性采样或变分后验 \(q_\phi(z|x)\) 来构造下界。摊销推断用编码器网络从 \(x\) 直接预测 \(q\) 的参数，避免对每个数据点单独优化。梯度估计可用似然比梯度或重参数化技巧，后者是 VAE 的常用做法。最终 VAE 目标可读作“重构损失减去 KL 正则”。

---

## 一、从全观测模型到潜变量模型 [02:03 - 08:34]

自回归模型和流模型都在直接建模观测变量 \(x\) 的分布，所有随机变量都是可观测的；潜变量模型则假设存在一个更低维、更紧凑的 \(z\) 来编码 \(x\) 中的高层信息。这样做的潜在好处包括：压缩数据、在表示空间插值、学习更适合下游分类器的特征，以及更快采样——先采样 \(z\)，再一次性生成整张 \(x\)，不必像自回归模型那样逐像素等待。

以柯基图片为例：像素级描述 vs. “左边红白柯基、中间另一只、右边三色柯基、背景木凳和公园”的高层描述。理想情况下，生成模型可以把图像投影到这种高层解释上，再反投影回图像。给定 \(z\) 后，图像各部分可能条件独立地生成，因此采样可以并行。

潜变量模型也可以人工设计解码器，例如把 \(z\) 看作图形引擎的场景描述，解码器是硬编码的渲染过程；课程后段会涉及 NeRF 类方法。但本讲不假设这种结构，而是希望从数据中学习。

![潜变量模型（LVM）与自回归模型/流的对比及概率图模型示意图](/coursenotes/cs294-158-deep-unsupervised-sp24/l04-latent-variable-models/kf_0001_00-02-09.jpg)

---

## 二、最简单的 LVM 与四个核心操作 [08:34 - 12:32]

最简单的潜变量模型是：

\[
z \sim p_Z(z),\qquad x \sim p_\theta(x|z).
\]

其中 \(p_Z(z)\) 通常选成容易采样和评估的分布，例如伯努利或高斯；\(p_\theta(x|z)\) 通常由深度神经网络把 \(z\) 映射为 \(x\) 的分布参数。\(z\) 可以是多个二值变量，表示“有没有车”“有没有狗”等高层属性。

围绕该模型有四个核心操作：

- **采样**：先采样 \(z\)，再采样 \(x\)。
- **似然评估**：\(p_\theta(x)=\sum_z p_Z(z)p_\theta(x|z)\)。
- **训练**：最大化数据对数似然  
  \[
  \max_\theta \sum_i \log p_\theta(x^{(i)})
  =\sum_i \log \sum_z p_Z(z)p_\theta(x^{(i)}|z).
  \]
- **表示**：由 \(x\) 推断 \(z\)，原则上可用贝叶斯公式  
  \[
  p_\theta(z|x)=\frac{p_\theta(x|z)p_Z(z)}{p_\theta(x)}.
  \]

注意这里与流模型不同：流模型中 \(z\) 与 \(x\) 同维且一一映射；潜变量模型中 \(z\) 可以更低维，且多个 \(z\) 都可能生成同一个 \(x\)。

![潜在变量模型示例：包含先验分布、生成模型公式及概率图模型](/coursenotes/cs294-158-deep-unsupervised-sp24/l04-latent-variable-models/kf_0002_00-08-34.jpg)

![潜在变量模型的四个核心操作：采样、似然评估、训练与表示](/coursenotes/cs294-158-deep-unsupervised-sp24/l04-latent-variable-models/kf_0003_00-10-43.jpg)

---

## 三、训练场景一：可枚举的 \(z\) 与混合高斯示例 [12:32 - 18:46]

最基础的情形是 \(z\) 只能取少量值，此时边缘似然的求和可以精确枚举，目标函数直接可优化。

先验 \(p_Z(z)\) 的选择是一个权衡：

- 固定一个简单先验（如高斯或伯努利），可以迫使数据分布映射到简单空间，使表示空间中的邻近性更有意义。
- 但高斯先验会强制 \(z\) 各维独立，可能无法捕捉数据中的相关性，例如“有狗时更可能有另一只狗”这类依赖。
- 有时也会学习先验 \(p_Z\)，课程后续会涉及。

一个具体例子是二维数据上的混合高斯模型：\(z\) 取三个值，\(x|z\) 为高斯分布，固定混合系数，只学习每个高斯分量的均值和方差。用梯度下降训练后，三个高斯分量会从随机初始化逐步移动到数据的三个众数。这个例子说明精确目标可行时，LVM 可以像普通最大似然模型一样训练。

![潜变量模型训练目标页，含 z→x 概率图及精确/近似两种场景的讨论](/coursenotes/cs294-158-deep-unsupervised-sp24/l04-latent-variable-models/kf_0004_00-15-00.jpg)

---

## 四、训练场景二：不可枚举时用先验采样 [18:46 - 25:02]

当 \(z\) 连续或离散组合极多（例如 100 个二值变量有 \(2^{100}\) 种取值）时，无法枚举求和。由于求和本身是 \(p_Z(z)\) 下的期望，可以用采样近似：

\[
\sum_i \log \sum_z p_Z(z)p_\theta(x^{(i)}|z)
\approx
\sum_i \log \frac{1}{K}\sum_{k=1}^K p_\theta(x^{(i)}|z_k^{(i)}),
\quad z_k^{(i)}\sim p_Z(z).
\]

\(K\) 是超参数，类似 batch size：\(K\) 越大估计越准，但一次梯度更新的计算越贵。有限 \(K\) 的估计还会受到外层 \(\log\) 非线性的影响；当 \(K\to\infty\) 时目标收敛到真实值。

先验采样的关键问题是：在高维 \(z\) 下，随机采到的 \(z\) 很难与给定 \(x^{(i)}\) 匹配。例如 \(z\) 有 100 个高层属性，正确组合概率可能只有 \(0.5^{100}\)，绝大多数样本对目标没有贡献，计算被浪费。

![先验采样（Prior Sampling）的近似目标函数公式及SGD优化说明](/coursenotes/cs294-158-deep-unsupervised-sp24/l04-latent-variable-models/kf_0005_00-18-46.jpg)

![先验采样的示例与挑战：高维数据下均匀采样难以匹配数据点](/coursenotes/cs294-158-deep-unsupervised-sp24/l04-latent-variable-models/kf_0006_00-23-35.jpg)

---

## 五、重要性采样：把样本引向有用的 \(z\) [25:02 - 30:00]

重要性采样用于计算一个分布下的期望，但样本从另一个更容易采样、且更集中的提议分布 \(q(z)\) 中抽取：

\[
\mathbb{E}_{z\sim p_Z}[f(z)]
=
\mathbb{E}_{z\sim q}\left[\frac{p_Z(z)}{q(z)}f(z)\right].
\]

其中 \(p_Z(z)/q(z)\) 是校正权重。若 \(q\) 在 \(p_Z\) 概率低但 \(f(z)\) 大的区域放样本，权重会把这些样本降权，但至少能获得有信号的样本。

在 LVM 中，理想提议分布是给定 \(x^{(i)}\) 的后验：

\[
q(z)=p_\theta(z|x^{(i)})
=\frac{p_\theta(x^{(i)}|z)p_Z(z)}{p_\theta(x^{(i)})}.
\]

它会把样本集中在能解释当前 \(x^{(i)}\) 的 \(z\) 上。但该分布本身不容易直接采样，因为分母 \(p_\theta(x^{(i)})\) 正是难以计算的边缘似然。因此需要用参数化分布去近似它。

![重要性采样算法推导：通过引入提议分布 q(z) 将期望重写为可采样形式](/coursenotes/cs294-158-deep-unsupervised-sp24/l04-latent-variable-models/kf_0007_00-27-52.jpg)

![将重要性采样应用于隐变量模型训练目标，并讨论提议分布 q(z) 的选择](/coursenotes/cs294-158-deep-unsupervised-sp24/l04-latent-variable-models/kf_0008_00-30-00.jpg)

---

## 六、变分推断与摊销：用 \(q_\phi(z|x)\) 近似后验 [30:00 - 40:43]

变分方法不直接得到真实后验，而是提出一个容易采样的参数化分布 \(q_\phi(z|x)\)，并最小化它与真实后验的 KL：

\[
\min_\phi\ D_{\mathrm{KL}}\big(q_\phi(z|x^{(i)})\|p_\theta(z|x^{(i)})\big).
\]

展开 KL 后，含 \(z\) 的可优化项包括 \(\log q_\phi(z|x)\)、\(\log p_Z(z)\)、\(\log p_\theta(x|z)\)，这些都由设计保证容易计算；不含 \(z\) 的 \(\log p_\theta(x^{(i)})\) 对 \(q\) 是常数，不影响优化。

逐数据点求解 \(q\) 太慢，因此采用**摊销推断**：用一个神经网络 \(f_\phi\) 从 \(x\) 直接输出 \(q_\phi(z|x)\) 的参数，通常是均值 \(\mu_\phi(x)\) 和方差 \(\sigma_\phi^2(x)\)（实际中常输出 \(\log\sigma^2\) 保证正性）。这样把“对每个 \(x^{(i)}\) 解一个优化问题”变成“训练一个共享网络”。

由于 \(\theta\) 更新后后验也会变化，实践中交替优化 \(\theta\) 和 \(\phi\)。从 \(q_\phi\) 采样时可写成：

\[
z=\mu_\phi(x)+\varepsilon\cdot\sigma_\phi(x),
\quad \varepsilon\sim\mathcal{N}(0,I).
\]

摊销的优点是更快、有正则化效果；缺点是网络近似不一定精确。

![摊销推断（Amortized Inference）的核心思想：用神经网络参数化推断问题以加速求解](/coursenotes/cs294-158-deep-unsupervised-sp24/l04-latent-variable-models/kf_0009_00-36-26.jpg)

---

## 七、IWAE：重要性加权自编码器目标 [40:43 - 45:49]

把重要性采样用于 LVM 训练，得到重要性加权自编码器（IWAE）目标：

\[
\sum_i \log\left(
\frac{1}{K}\sum_{k=1}^K
\frac{p_Z(z_k^{(i)})}{q_\phi(z_k^{(i)}|x^{(i)})}
p_\theta(x^{(i)}|z_k^{(i)})
\right),
\quad z_k^{(i)}\sim q_\phi(z_k^{(i)}|x^{(i)}).
\]

同时还要最小化 \(q_\phi(z|x^{(i)})\) 与真实后验的 KL。两类项可以合并为“最大化 term1 减去 term2”。理论结果表明，\(K\) 越大，下界越紧；但每个梯度步的计算也越贵，因此 \(K\) 是计算与精度的权衡。

![重要性加权自编码器（IWAE）的目标函数及与标准VAE损失的关系](/coursenotes/cs294-158-deep-unsupervised-sp24/l04-latent-variable-models/kf_0010_00-40-43.jpg)

---

## 八、ELBO 的另外两种推导 [45:49 - 57:34]

### 8.1 Jensen 不等式推导

从边缘似然出发，乘除一个提议分布 \(q(z|x)\)：

\[
\log p_\theta(x)
=
\log \mathbb{E}_{q}\left[
\frac{p_Z(z)p_\theta(x|z)}{q(z|x)}
\right].
\]

利用 Jensen 不等式 \(\log \mathbb{E}[X]\ge \mathbb{E}[\log X]\)，得到下界：

\[
\log p_\theta(x)
\ge
\mathbb{E}_{q(z|x)}\left[
\log \frac{p_Z(z)p_\theta(x|z)}{q(z|x)}
\right].
\]

这个下界就是证据下界（ELBO）/变分下界（VLB）。当 \(q(z|x)\) 等于真实后验 \(p_\theta(z|x)\) 时，等号成立，下界最紧。

![手写推导对数边际似然下界，引入提议分布q(z)并应用Jensen不等式](/coursenotes/cs294-158-deep-unsupervised-sp24/l04-latent-variable-models/kf_0011_00-49-18.jpg)

![完整ELBO推导：将最大化似然转化为对θ和φ的双重优化问题](/coursenotes/cs294-158-deep-unsupervised-sp24/l04-latent-variable-models/kf_0012_00-51-26.jpg)

### 8.2 从 KL 重排得到分解

从

\[
D_{\mathrm{KL}}\big(q_\phi(z|x)\|p_\theta(z|x)\big)
\]

出发，重排可得：

\[
\log p_\theta(x)
=
\mathbb{E}_{q_\phi(z|x)}\left[
\log \frac{p_Z(z)p_\theta(x|z)}{q_\phi(z|x)}
\right]
+
D_{\mathrm{KL}}\big(q_\phi(z|x)\|p_\theta(z|x)\big).
\]

因此 ELBO 与真实对数似然的差距正好是 KL 项。这个推导说明：优化 ELBO 时，差距来自 \(q_\phi\) 尚未完全匹配真实后验；若 KL 不为零，下界就不紧。

![变分下界(VLB)的最终形式，展示log p(x)分解为VLB和KL散度两部分](/coursenotes/cs294-158-deep-unsupervised-sp24/l04-latent-variable-models/kf_0013_00-57-52.jpg)

---

## 九、梯度估计一：似然比梯度 [57:34 - 67:35]

考虑目标

\[
\max_\phi\ \mathbb{E}_{z\sim q_\phi(z)}[f(z)].
\]

直接采样后对 \(f(z)\) 求 \(\phi\) 的梯度会失败：采样过程里的 \(\phi\) 被丢掉了，算出来的梯度为零。正确做法是

\[
\nabla_\phi \mathbb{E}_{q_\phi}[f(z)]
=
\mathbb{E}_{q_\phi}\left[
\nabla_\phi \log q_\phi(z)\, f(z)
\right],
\]

再用 \(K\) 个样本近似：

\[
\approx
\frac{1}{K}\sum_{k=1}^K
\nabla_\phi \log q_\phi(z_k) f(z_k).
\]

直观上，它提高“好样本”的对数概率、降低“坏样本”的对数概率。玩具例子中，用高斯分布去最小化到 \((5,5)\) 的平均偏差，似然比梯度只会朝当前最佳样本移动；如果最佳样本不在目标点，收敛就会很慢。

![似然比梯度完整推导过程，展示从求和到期望的变换及最终梯度估计式](/coursenotes/cs294-158-deep-unsupervised-sp24/l04-latent-variable-models/kf_0014_01-04-18.jpg)

![似然比梯度的推导过程，展示从期望形式到对数导数技巧的变换](/coursenotes/cs294-158-deep-unsupervised-sp24/l04-latent-variable-models/kf_0015_01-06-17.jpg)

---

## 十、梯度估计二：重参数化技巧 [70:44 - 77:09]

对于高斯等可重参数化分布，可以把随机性从参数中分离出来：

\[
q_\phi(z)=\mathcal{N}(\mu,\sigma^2),
\qquad
z=\mu+\varepsilon\cdot\sigma,
\quad
\varepsilon\sim\mathcal{N}(0,I).
\]

于是

\[
\mathbb{E}_{z\sim q_\phi}[f(z)]
=
\mathbb{E}_{\varepsilon\sim\mathcal{N}(0,I)}
[f(\mu+\varepsilon\sigma)].
\]

梯度可以直接穿过 \(f\) 传到 \(\mu,\sigma\)，即路径导数：

\[
\nabla_\phi \mathbb{E}_{z\sim q_\phi}[f(z)]
=
\mathbb{E}_{\varepsilon}\left[
\nabla_\phi f(\mu_\phi+\varepsilon\sigma_\phi)
\right].
\]

相比似然比梯度，重参数化通常更有效，但要求分布可重参数化。VAE 中编码器输出高斯参数，正是用这一技巧把梯度传回编码器。

![重参数化技巧的完整推导：将高斯采样改写为 ε~N(0,1) 的确定性变换并用 K 个样本均值近似期望](/coursenotes/cs294-158-deep-unsupervised-sp24/l04-latent-variable-models/kf_0016_01-10-44.jpg)

![路径导数（Pathwise Derivative）方法：将隐变量重参数化为噪声函数并推导梯度估计公式](/coursenotes/cs294-158-deep-unsupervised-sp24/l04-latent-variable-models/kf_0017_01-13-45.jpg)

---

## 十一、VAE 的最终目标与网络形式 [77:09 - 82:32]

变分自编码器的目标是：

\[
\max_{\theta,\phi}\ 
\mathbb{E}_{q_\phi(z|x)}
\left[
\log p_\theta(x|z)+\log p_Z(z)-\log q_\phi(z|x)
\right].
\]

它等价于最大化 ELBO。写成损失形式：

\[
-\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]
+
D_{\mathrm{KL}}\big(q_\phi(z|x)\|p_Z(z)\big).
\]

第一项是**重构损失**，第二项是 **KL 正则**。常见参数化为：

\[
q_\phi(z|x)=\mathcal{N}(\mu_\phi(x),\sigma_\phi^2(x)),
\qquad
p_\theta(x|z)=\mathcal{N}(\mu_\theta(z),\sigma_\theta^2(z)).
\]

编码器把 \(x\) 映射为 \(z\) 的分布参数，解码器把 \(z\) 映射回 \(x\) 的分布参数；训练时用重参数化采样 \(z\)，同时优化重构与正则。

![VAE与似然比梯度的推导公式，包含对数概率项、期望及梯度算子](/coursenotes/cs294-158-deep-unsupervised-sp24/l04-latent-variable-models/kf_0018_01-17-09.jpg)

![VAE的概率图模型及编码器/解码器的高斯分布参数化公式](/coursenotes/cs294-158-deep-unsupervised-sp24/l04-latent-variable-models/kf_0019_01-19-18.jpg)

![变分自编码器（VAE）的目标函数公式，标注了重构损失与正则化项](/coursenotes/cs294-158-deep-unsupervised-sp24/l04-latent-variable-models/kf_0020_01-22-32.jpg)

---

## 要点回顾

- 潜变量模型假设低维 \(z\) 编码 \(x\) 的高层信息，目标是最大化 \(p_\theta(x)=\sum_z p_Z(z)p_\theta(x|z)\)。
- 与自回归/流模型相比，LVM 可更低维、更快采样，并可能学到更有意义的表示。
- \(z\) 少时可精确枚举；\(z\) 多时需近似。
- 先验采样简单，但在高维 \(z\) 下绝大多数样本对给定 \(x\) 没有贡献。
- 重要性采样用提议分布 \(q(z)\) 和权重 \(p_Z(z)/q(z)\) 把样本引向有用的 \(z\)。
- 理想提议分布是后验 \(p_\theta(z|x)\)，但它本身难采样，因此用变分分布 \(q_\phi(z|x)\) 近似。
- 摊销推断用编码器网络从 \(x\) 输出 \(q_\phi(z|x)\) 的参数，替代逐点优化。
- IWAE 用 \(K\) 个重要性样本构造更紧的下界；\(K\) 越大越准但越贵。
- ELBO 可由 Jensen 不等式或 KL 重排得到；\(\log p_\theta(x)=\text{ELBO}+D_{\mathrm{KL}}\)。
- 似然比梯度 \(\mathbb{E}[\nabla_\phi\log q_\phi(z)f(z)]\) 通用但收敛慢。
- 重参数化技巧 \(z=\mu+\varepsilon\sigma\) 让梯度穿过采样过程，是 VAE 的常用梯度估计方法。
- VAE 损失 = 重构损失 + KL 正则。
