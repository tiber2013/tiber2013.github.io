---
title: "Course Note 10 · 最优估计：从贝叶斯滤波到 Kalman 滤波与扩展 Kalman 滤波"
description: "CS231A Course Notes 10 中文全译，覆盖状态估计的形式化、条件概率复习与贝叶斯滤波的完整推导、离散贝叶斯滤波、高斯分布与 Kalman 滤波的预测/更新方程及增益解释、EKF 的雅可比线性化。对应第 14–15 讲。"
---

# Course Note 10 · 最优估计：从贝叶斯滤波到 Kalman 滤波与扩展 Kalman 滤波

> **Stanford CS231A 计算机视觉：从三维感知到三维重建 · Course Notes 10** ｜ 原文作者：Bryan Chiang、Jeannette Bohg（2022-02-23）｜ 对应课件：第 14 讲 Optimal Estimation 与第 15 讲 Optimal Estimation Cont.（Flipped Format，Jeannette Bohg 讲授）｜ 本课无公开视频，笔记依据官方 Course Notes 全文翻译

## 这篇在讲什么

前面的 Course Notes 把「从图像恢复几何」大体上写成了批式优化问题：SfM 与 bundle adjustment 把全部图像一次性收集起来、联立求解所有未知量。这篇换到在线设定：智能体按时间顺序逐步收到观测与控制输入，每一步都要给出当前状态的最佳估计。推导主线是一条递归公式——贝叶斯滤波把全部历史压缩进「上一时刻的后验」，每步只做预测（predict）与更新（update）两件事；随后给出两个可计算的实例：离散贝叶斯滤波（状态有限，预测与更新都是矩阵乘法）与 Kalman 滤波（线性高斯系统，只需递推均值向量与协方差矩阵），最后用雅可比线性化把 Kalman 滤波推广到非线性系统，即扩展 Kalman 滤波（EKF）。这也是本课几何与学习两部分的收束：估计不再是一次性解大方程组，而是在不确定性下递归地融合运动模型与观测。

## 1 状态估计（State estimation）

知道环境的当前状态，对人工智能体可靠地做出实时决策至关重要。在状态估计（state estimation）中，我们的目标是通过持续融合来自不同传感器源的测量（多模态感知），推断系统在当前时刻的隐状态（latent state）。

我们先从概率视角考察离散时间动力系统。Figure 1 以图模型（graphical model）的形式画出了一个部分可观测马尔可夫决策过程（partially observable Markov decision process, POMDP）。

![图 1：POMDP 的图模型](/coursenotes/cs231a-computer-vision-3d-sp25/cn10-optimal-estimation/fig_7ce4c9fc47.jpg)

图 1：POMDP 的图模型。三行节点自上而下是控制输入 $u(t)$、状态 $x(t)$、观测 $z(t)$；有向边刻画出全文赖以推导的两条条件独立假设：状态只依赖前一状态与当前控制，观测只依赖当前状态。

有向边表示条件依赖关系。$\boldsymbol{x}_t \in \mathbb{R}^n$ 是时刻 t 的状态，它只依赖上一个状态 $\boldsymbol{x}_{t-1}$。$\boldsymbol{z}_t \in \mathbb{R}^k$ 是依赖时刻 t 状态的传感器观测。$\boldsymbol{u}_t \in \mathbb{R}^m$ 表示时刻 t 施加的控制输入。举例来说，身处某环境中的人工智能体，其状态可以是位置、朝向、线速度与角速度，或以上的任意组合；类似地，由状态导出的测量可以是环境的相机图像或激光雷达（Lidar）测量，它们依赖机器人的位置与运动。

测量与控制输入是已知的，但隐藏的状态历史 $x_{0:t}$ 未知。我们假设系统以随机方式演化、观测也是随机的。因此我们把状态 $X_t$ 与观测 $Z_t$ 建模为随机变量，它们可以取 $x_t$、$z_t$ 作为可能的值。为简化记号，我们把 $p(X_t = x_t)$ 写成 $p(x_t)$，对 $Z_t$ 同理。

状态估计器的目标，是在给定可用数据（测量与控制输入的历史）以及已知的状态转移模型与观测模型之下，计算或近似状态 $x_t$ 的后验概率分布。具体地，我们想知道状态的后验分布 $p(\boldsymbol{x}_t | \boldsymbol{z}_{1:t}, \boldsymbol{u}_{1:t})$：给定测量序列与控制输入序列，它给状态可能取到的每一个值赋予一个概率。后验也称为关于时刻 t 状态取值的信念（belief），记作 bel(t)。概率化表述并不给出状态的精确值，但使我们能够量化「状态可能是多少」的不确定性。（译注：全篇记号中 bel(t) 指后验 $p(x_t | z_{1:t}, u_{1:t})$，带尖帽的 $\hat{\mathrm{bel}}$ 指预测后验；$\hat{\mathrm{bel}}(t-1)$ 的自变量照原文写法，它实际表示的是只用截至 $t-1$ 的信息对时刻 t 状态所作的预测，即 $p(x_t | z_{1:t-1}, u_{1:t})$，见 2.2 节与式 (6)(7)。）

用 Figure 1 的图模型表示动力系统，即假设状态是完备的（complete）。这一假设导出两条关键性质。第一，假设系统是马尔可夫的：当前状态 $x_t$ 只依赖上一状态 $x_{t-1}$ 与上一控制输入 $u_{t-1}$，而不是整个历史 $z_{1:t-1}$。这可以写成

$$
p(x_t | x_{0:t-1}, z_{1:t-1}, u_{1:t}) = p(x_t | x_{t-1}, u_t)
\tag{1}
$$

它也称为转移模型（transition model），表示当前处于 $x_{t+1}$、给定控制输入 $u_t$ 时系统转移到状态 $x_t$ 的可能性。（译注：此处「当前处于 $x_{t+1}$」照原文印出；按式 (1) 与 Figure 1 的上下文应为 $x_{t-1}$。）

第二，假设当前测量 $z_t$ 只依赖当前状态 $x_t$，即 $z_t$ 条件独立于所有先前状态 $x_{0:t-1}$、测量 $z_{1:t-1}$ 与控制输入 $u_{1:t}$。这可以写成

$$
p(z_t | x_{0:t}, z_{1:t-1}, u_{1:t}) = p(z_t | x_t)
\tag{2}
$$

这也称为测量模型（measurement model），表示给定状态 $x_t$ 时获得测量 $z_t$ 的可能性。

## 2 贝叶斯滤波（Bayes filter）

递归滤波（recursive filter）持续读入新测量以估计状态后验，如 Figure 2 所示。在每个时间步 t，我们仅用旧后验 $p(x_{t-1} | z_{1:t-1}, u_{0:t-1})$、新控制输入 $u_{t-1}$ 与新测量 $z_t$ 计算新后验 $p(\boldsymbol{x}_t | \boldsymbol{z}_{1:t}, \boldsymbol{u}_{0:t})$。因此递归滤波的复杂度关于时间是常数：它不依赖历史的长度，适合实时推断。

![图 2：一般递归滤波的框图](/coursenotes/cs231a-computer-vision-3d-sp25/cn10-optimal-estimation/fig_b6ec33b051.jpg)

图 2：一般递归滤波的框图。上一后验与当前测量 $z_t$ 进入滤波器、产出当前后验；当前后验经 $t = t+1$ 回送为下一轮的上一后验，因此每步计算量与历史长度无关。

### 2.1 条件概率复习（Conditional probability review）

联合概率分布可以分解为 $p(A, B) = p(A|B)p(B)$，其中 A 与 B 是随机变量。若联合分布还以另一个随机变量 C 为条件，只需把条件原样传递下去：$p(A, B|C) = p(A|B, C)p(B|C)$。分布可以边缘化为 $p(A) = \int_B p(A, B) dB$。与前面类似，有 $p(A|C) = \int_B p(A, B|C) dB$。同一个联合分布可以用两种不同方式分解，由此得到贝叶斯公式（Bayes rule）：$p(A|B)p(B) = p(B|A)p(A)$。

### 2.2 推导（Derivation）

现在推导贝叶斯滤波——最简单的递归滤波。我们的目标是导出状态后验的递归表达式，使其只依赖上一后验、当前测量 $z_t$、控制输入 $u_t$ 以及转移模型与测量模型。模型假设已知且可计算，即 $p(x_t | x_{t-1}, u_t)$ 与 $p(\boldsymbol{z}_t | \boldsymbol{x}_t)$ 都能算出来。贝叶斯公式与测量的条件独立性给出：

$$
p(x_t | z_{1:t}, u_{1:t}) = \frac{p(z_t | x_t, z_{1:t-1}, u_{1:t}) \, p(x_t | z_{1:t-1}, u_{1:t})}{p(z_t | z_{1:t-1}, u_{1:t})}
\tag{3}
$$

$$
= \frac{p(z_t | x_t) \, p(x_t | z_{1:t-1}, u_{1:t})}{p(z_t | z_{1:t-1}, u_{1:t})}
\tag{4}
$$

接着化简右端仍依赖整个观测历史的两项。先看分母。对 $x_t$ 边缘化并使用测量的条件独立性：

$$
\begin{array}{rl}
& p(z_t | z_{1:t-1}, u_{1:t}) = \int_{x_t} p(z_t, x_t | z_{1:t-1}, u_{1:t}) \, dx_t \\
& \qquad = \int_{x_t} p(z_t | x_t, z_{1:t-1}, u_{1:t}) \, p(x_t | z_{1:t-1}, u_{1:t}) \, dx_t \\
& \qquad = \int_{x_t} p(z_t | x_t) \, p(x_t | z_{1:t-1}, u_{1:t}) \, dx_t
\end{array}
$$

接下来化简 $p(x_t | z_{1:t-1}, u_{1:t})$——它现在在式 (4) 中出现了两次。对 $x_{t-1}$ 边缘化并应用马尔可夫假设：

$$
\begin{array}{rl}
& p(x_t | z_{1:t-1}, u_{1:t}) = \int_{x_{t-1}} p(x_t, x_{t-1} | z_{1:t-1}, u_{1:t}) \, dx_{t-1} \\
& \qquad = \int_{x_{t-1}} p(x_t | x_{t-1}, z_{1:t-1}, u_{1:t}) \, p(x_{t-1} | z_{1:t-1}, u_{1:t}) \, dx_{t-1} \\
& \qquad = \int_{x_{t-1}} p(x_t | x_{t-1}, u_t) \, p(x_{t-1} | z_{1:t-1}, u_{1:t}) \, dx_{t-1}
\end{array}
$$

现在可以拼出贝叶斯滤波的完整形式，如 Figure 3 与 Figure 4 所示。

![图 3：贝叶斯滤波的框图](/coursenotes/cs231a-computer-vision-3d-sp25/cn10-optimal-estimation/fig_0a708b1fcd.jpg)

图 3：贝叶斯滤波的框图。虚线框内是递归滤波的内部结构：先验 $p(x_0)$（$t=0$ 的先验）进入标有 Predict 的预测模块，得到预测后验 $p(x_t | z_{1:t-1}, u_{1:t})$；标有 Update 的更新模块再结合当前测量 $z_t$，得到当前后验 $p(x_t | z_{1:t}, u_{1:t})$；经 $t = t+1$ 后它作为上一后验进入下一轮。

![图 4：贝叶斯滤波的预测步与更新步方程](/coursenotes/cs231a-computer-vision-3d-sp25/cn10-optimal-estimation/fig_1f45f8b1dc.jpg)

图 4：贝叶斯滤波的预测步与更新步方程，各色块标出每个因子的角色。预测步：转移模型 $p(x_t | x_{t-1}, u_t)$（绿）乘上一后验（蓝）再对 $x_{t-1}$ 积分，得预测的当前后验（红）；更新步：测量模型 $p(z_t | x_t)$（橙）乘预测后验（红），除以归一化因子 $p(z_t | z_{1:t-1}, u_{1:t})$（浅蓝），得滤波后的当前后验。

一共两步。在预测步（prediction step）中，我们用转移模型 $p(x_t | x_{t-1}, u_t)$ 与上一后验，预测当前时间步的状态后验 $p(x_t | z_{1:t-1}, u_{1:t})$。预测记作 $\hat{\mathrm{bel}}(t-1)$，它是在没有任何新信息时对 $x_t$ 分布的最佳猜测。当 $t = 1$ 时上一后验尚不存在，于是使用先验 $p(x_0)$，它代表我们对起始时刻 $t = 0$ 时 x 如何分布的初始信念。在更新步（update step）中，我们纳入新测量 $z_t$ 来精炼预测的信念。测量似然（measurement likelihood）$p(\boldsymbol{z}_t | \boldsymbol{x}_t)$ 表示在预测的 $x_t$ 下观测到 $z_t$ 的可能性。因此更新步的分子基于两点给指定的 $x_t$ 赋概率：1) $x_t$ 在多大程度上解释我们的测量 $z_t$；2) 动力系统从上一后验估计 $x_{t-1}$ 转移到 $x_t$ 的可能性有多大。分母只是一个归一化常数（normalization constant），保证更新后的后验分布求和为 1。一旦完成信念的预测与更新，只要有新测量 $z_{t+1}$，就可以在 $t+1$ 重复这一过程。

贝叶斯滤波很简单，且可推广到任意概率分布。但实践中它受限：归一化常数的积分往往难以计算（intractable）。我们需要知道整个概率分布（例如对定义域中每个取值列表给出），而这通常只在处理离散随机变量、或假设后验分布具有特定函数形式（例如高斯）时才可能。因此，更有用的看法是把贝叶斯滤波当作一个「概率模板（probabilistic template）」，可以针对具有特定需求与假设的不同系统加以适配。

### 2.3 离散贝叶斯滤波（Discrete Bayes filter）

来看离散情形的实用实现：$X_t$ 与 $Z_t$ 只有有限多个取值。设 $X_t$ 是有 N 个可能值的离散随机变量，其定义域为 $x_t \in \{1, 2, \ldots, N\}$。类似地，$Z_t$ 是有 M 个可能结果的离散随机变量，其定义域为 $z_t \in \{1, 2, \dots, M\}$。离散贝叶斯滤波的预测与更新方程与连续情形相同，只是积分换成求和。

式 (5) 是信念向量（belief vector），表示时刻 t 的整个状态后验分布。

$$
\mathrm{bel}(t) = \begin{pmatrix} p(X_t = 1 | z_{1:t}, u_{1:t}) \\ p(X_t = 2 | z_{1:t}, u_{1:t}) \\ \vdots \\ p(X_t = N | z_{1:t}, u_{1:t}) \end{pmatrix}
\tag{5}
$$

预测步于是可以实现为一次矩阵乘法：

$$
\hat{\mathrm{bel}}(t-1) = T(u_t) \, \mathrm{bel}(t-1)
\tag{6}
$$

$$
= \begin{pmatrix} p(X_t = 1 | z_{1:t-1}, u_{1:t}) \\ p(X_t = 2 | z_{1:t-1}, u_{1:t}) \\ \vdots \\ p(X_t = N | z_{1:t-1}, u_{1:t}) \end{pmatrix}
\tag{7}
$$

其中 $T(u_t)$ 表示转移概率：

$$
T(u_t) = \begin{pmatrix} p(X_t = 1 | X_{t-1} = 1, u_t) & \ldots & p(X_t = 1 | X_{t-1} = N, u_t) \\ \vdots & \ddots & \vdots \\ p(X_t = N | X_{t-1} = 1, u_t) & \ldots & p(X_t = N | X_{t-1} = N, u_t) \end{pmatrix}.
\tag{8}
$$

类似地，更新步也可以实现为一次矩阵乘法：

$$
\mathrm{bel}(t) = \frac{M(z_t) \, \hat{\mathrm{bel}}(t-1)}{\mathbf{1}^T (M(z_t) \, \hat{\mathrm{bel}}(t-1))}
\tag{9}
$$

其中 M 表示测量模型概率：

$$
M(z_t) = \begin{pmatrix} p(Z_t = 1 | X_t = 1) & \ldots & p(Z_t = 1 | X_t = N) \\ \vdots & \ddots & \vdots \\ p(Z_t = M | X_t = 1) & \ldots & p(Z_t = M | X_t = N) \end{pmatrix}
\tag{10}
$$

（译注：$M(z_t)$ 照原文印成 $M \times N$ 矩阵，与式 (9) 中 N 维信念向量的维度不匹配；常见的等价实现是取 $M(z_t)$ 中与实际观测 $z_t$ 对应的那一行，与预测信念逐分量相乘，即按似然 $p(z_t | X_t = i)$ 对每个状态取值逐个加权。）

在式 5 的分母中，$\mathbf{1}^T$ 是每个分量都为 1 的向量，用于把未归一化的概率求和。（译注：原文写「式 5 的分母」，但含 $\mathbf{1}^T$ 的分母实际出现在式 (9)；此处照原文译出。另注：离散贝叶斯滤波这套「预测—更新」结构即隐马尔可夫模型（HMM）的前向滤波：转移矩阵 $T(u_t)$ 与测量矩阵 $M(z_t)$ 分别对应 HMM 的转移概率与发射概率。）

把无限大的状态空间切分成有限多个区域、每个区域用一个概率值代表累积后验，这套做法也可以用到连续随机变量上。对连续状态空间如此离散化，得到的就是直方图滤波（histogram filter）。

## 3 Kalman 滤波（Kalman filter）

前面我们看到状态转移模型 $p(x_t | x_{t-1}, u_t)$ 与测量模型 $p(\boldsymbol{z}_t | \boldsymbol{x}_t)$ 被表示为概率分布。这是贝叶斯视角。转移模型与测量模型也可以从动力系统视角来看：式 (11) 表示动力学模型，式 (12) 表示测量模型。

$$
x_t = f(x_{t-1}, u_t) + w_{t-1}
\tag{11}
$$

$$
z_t = h(x_t) + v_t
\tag{12}
$$

$f(\cdot)$ 是系统的动力学（dynamics），封装了状态如何随时间从 $x_t$ 演化到 $x_{t+1}$。类似地，$h(\cdot)$ 是把状态 $x_t$ 映射到对应观测的函数。$w_t$ 与 $v_t$ 分别是过程噪声（process noise，系统中的随机扰动）与测量噪声（measurement noise，传感器内部），它们自然地出现在动力系统中。（译注：式 (11) 中过程噪声的下标照原文为 $w_{t-1}$，而下文线性高斯模型式 (14) 与噪声统计讨论中写作 $w_t$；原文两种下标并存，此处照原文。）我们假设动力学与测量模型以及噪声统计 $D_w, D_v$（即 $w \sim D_w, v \sim D_v$）已知。动力系统方程与 Figure 1 的图模型、以及「状态完备」的假设一致：$x_t$ 只依赖 $x_{t-1}$ 与 $u_t$，$z_t$ 只依赖 $x_t$。

### 3.1 高斯分布（Gaussian distributions）

若 $X \in \mathbb{R}^N$ 是从多元高斯分布（multivariate Gaussian distribution，亦称正态分布）中采样的随机向量，我们记 $X \sim \mathcal{N}(\mu, \Sigma)$，其中 $\boldsymbol{\mu} \in \mathbb{R}^N$ 是均值向量，$\Sigma \in \mathbb{R}^{N \times N}$ 是半正定协方差矩阵（covariance matrix，类比一元情形的方差）。对应的概率密度函数（probability density function, PDF）为：

$$
p(X) = p(\mu, \Sigma) = \frac{1}{\sqrt{(2\pi^{N} |\Sigma|)}} \exp(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu))
\tag{13}
$$

（译注：此式 MinerU 抽取时根号括号不闭合，已按原页补全；原页指数位置即如上文印在 $\pi$ 上，常见教材写法为 $(2\pi)^{N}$，两者在此以原页为准。）

Figure 5 同时画出了分布的物理形状与等值线（level sets）：沿每个椭圆概率为常数。

![图 5（右半）：二元高斯的等值椭圆与协方差矩阵的特征向量方向](/coursenotes/cs231a-computer-vision-3d-sp25/cn10-optimal-estimation/fig_aae9edf662.jpg)

图 5（右半）：二元高斯的等值椭圆（原图左半为三维钟形密度曲面）。椭圆沿协方差矩阵 $\Sigma$ 的两个特征向量方向伸展，箭头即特征向量方向；特征值越大，该方向上分布越「摊开」、不确定性越高。

我们看到，$\Sigma$ 指示分布在每个方向上的「摊开程度」：$\Sigma$ 的特征向量是轴方向，缩放由特征值决定。大特征值表示在对应特征向量方向上不确定性高，小特征值则相反。协方差矩阵不仅代表分布的置信程度，还间接捕捉状态各分量之间的相关性，这对估计很有用。关于多元高斯的更多细节，请参考 CS 229 讲义 [2, 1] 与 Berkeley CS 189 讲义 [3]。

Kalman 滤波是贝叶斯滤波对线性高斯系统（linear-Gaussian systems）的适配。顾名思义，线性高斯系统有两条关键假设。第一，所有涉及的随机变量（如状态、测量、后验、噪声）都是多元高斯。第二，所有变量关于其父变量都是线性的；$f(\cdot)$ 与 $h(\cdot)$ 是线性的（形如 $y = Ax$）。式 (14) 与 (15) 给出线性高斯系统模型。

$$
x_t = A_t x_{t-1} + B_t u_t + w_t
\tag{14}
$$

$$
z_t = C_t x_t + v_t
\tag{15}
$$

我们假设过程噪声与测量噪声是零均值高斯白噪声，即 $w_t \sim \mathcal{N}(0, Q_t)$、$v_t \sim \mathcal{N}(0, R_t)$，且对 $t \neq \tau$ 的时刻 $t, \tau$，有 $\mathrm{Cov}(w_t, w_\tau) = 0$ 与 $\mathrm{Cov}(v_t, v_\tau) = 0$。初始条件（即先验）表示为 $x_0 \sim \mathcal{N}(\mu_{0|0}, \Sigma_{0|0})$。进一步假设对所有 t 有 $\mathrm{Cov}(x_0, v_t) = 0$、$\mathrm{Cov}(x_0, w_t) = 0$，且对所有 $t, \tau$ 有 $\mathrm{Cov}(w_t, v_\tau) = 0$。

Kalman 滤波虽限于线性高斯系统，却能高效处理连续随机变量与向量（无限多个取值）。因为只要有了 $\mu$ 与 $\Sigma$，就有了整个概率分布。于是与贝叶斯滤波中对整个分布 $p(x)$（对所有可能的 x）做预测与更新不同，我们只需在每个时间步预测与更新均值向量和协方差矩阵，而这可以用闭式方程完成。这里我们只列出 Kalman 滤波方程并加以解释，说明背后的直觉。网上有许多推导：或从概率视角出发、对均值与协方差套用期望定义，或从优化视角出发、把它看作均方误差意义下的最佳线性无偏估计（best linear unbiased estimator）。

与贝叶斯滤波类似，Kalman 滤波由预测步与更新步组成。

在预测步中，我们从上一后验与系统过程模型的知识出发，计算预测后验 $p(x_t | z_{1:t-1}, u_{1:t}) \sim \mathcal{N}(\mu_{t|t-1}, \Sigma_{t|t-1})$ 的均值与协方差：

$$
\mu_{t|t-1} = A_t \mu_{t-1|t-1} + B_t u_t
\tag{16}
$$

$$
\Sigma_{t|t-1} = A_{t-1} \Sigma_{t-1|t-1} A_{t-1}^T + Q_{t-1}
\tag{17}
$$

（译注：式 (17) 的下标 $A_{t-1}$、$Q_{t-1}$ 照原文印出，与式 (16) 的 $A_t$ 不一致；常见写法全程用 $A_t$、$Q_t$。式 (23) 同。）

上一后验为 $p(x_{t-1} | z_{1:t-1}, u_{1:t-1}) \sim \mathcal{N}(\mu_{t-1|t-1}, \Sigma_{t-1|t-1})$。为预测均值 $\mu_{t|t-1}$，把 $\mu_{t-1|t-1}$ 代入式 (14) 即得式 (16)。式 (14) 中的噪声项 $w_t$ 均值为零，因此不影响 $\mu_{t|t-1}$。为推导式 (17) 的协方差预测，先利用这一事实：对协方差为 $\Sigma$ 的随机变量 x，有 $\mathrm{Cov}(Ax) = A \Sigma A^T$。然后，为计入系统随时间演化可能受到的扰动，加上过程噪声的协方差 $Q$。由于所有协方差矩阵都是正定的，这一相加使预测协方差变大，代表不确定性增加。因为我们只用截至 $t-1$ 的数据对时刻 t 的状态做预测，所以对状态分布的把握变小了。

在更新步中，我们用新测量 $z_t$ 精炼预测的均值与协方差。均值与协方差的更新方程为：

$$
\mu_{t|t} = \mu_{t|t-1} + K_t (z_t - C_t \mu_{t|t-1})
\tag{18}
$$

$$
\Sigma_{t|t} = \Sigma_{t|t-1} - K_t C_t \Sigma_{t|t-1}
\tag{19}
$$

$$
K_t = \Sigma_{t|t-1} C_t^T (C_t \Sigma_{t|t-1} C_t^T + R_t)^{-1}
\tag{20}
$$

单从方程就能看出，更新步比预测步更绕。先从均值更新说起：$z_t$ 是传感器测量，$C_t \mu_{t|t-1}$ 是预测测量（若状态确如我们预测，本该得到的测量）。因此测量残差 $z_t - C_t \mu_{t|t-1}$——也称为新息（innovation）——表示预测测量与传感器实际观测相差多少。均值更新因此试图用这个测量残差来精炼、调和我们的预测均值 $\mu_{t|t-1}$。这通过 Kalman 增益（Kalman Gain）$K_t$ 完成，它可以看作一个「加权因子」，告诉我们需要在多大程度上修正估计。把 $K_t$ 改写为：

$$
K_t = C_t^{-1} \frac{C_t \Sigma_{t|t-1} C_t^T}{C_t \Sigma_{t|t-1} C_t^T + R_t}
\tag{21}
$$

我们对右端的分式分析两种极端情况。第一种，$R_t$ 趋于零的极限情形，意味着我们相信测量噪声很小。分式趋于 1，K 就是 $C_t^{-1}$，从而 $\mu_{t|t} = C_t^{-1} z_t$。没有噪声意味着我们确信测量 $z_t$ 高度准确，只需把测量模型反解就能得到真实状态。此外，协方差被更新为 $K_t = \Sigma_{t|t-1} - \Sigma_{t|t-1} = 0$，表明我们对状态取值高度自信。（译注：此式左端原文印作 $K_t$，按上下文应为更新后的协方差 $\Sigma_{t|t}$。）第二种情形是 $\Sigma_t$ 趋于零的极限，意味着过程噪声很小。分式趋于 0，$K = 0$，更新后的均值为 $\mu_{t|t} = \mu_{t|t-1}$，更新后的协方差为 $\Sigma_{t|t} = \Sigma_{t|t-1}$。这意味着系统中没有扰动，仅靠过程模型就足以把状态沿时间传播，完全不需要纳入测量（前提是滤波器以一个好的先验初始化）。实践中这两种情况都不会出现；滤波器试图同时利用实际测量与预测测量来更新均值与协方差，如 Figure 6 对二维状态情形所示。

![图 6：从分布视角看 Kalman 滤波更新](/coursenotes/cs231a-computer-vision-3d-sp25/cn10-optimal-estimation/fig_d71b41ac90.jpg)

图 6：从分布视角看 Kalman 滤波更新。在二维状态空间 $(x_1, x_2)$ 中，橙色虚线椭圆是预测测量的分布，蓝色虚线椭圆是实际测量的分布，两者相乘（取重叠）得到更小的绿色更新后分布——更新一步使不确定性收缩。

Figure 6 的坐标轴是状态的各分量；我们是从正上方俯视这些分布。橙色椭圆代表预测测量的分布，为 $\mathcal{N}(C_t \mu_{t|t-1}, C_t \Sigma_{t|t-1} C_t^T)$。蓝色椭圆代表实际测量的分布，为 $\mathcal{N}(z_t, R_t)$。Kalman 滤波把这两个分布相乘，找出 $x_t$ 在两个分布下都大概率出现的重叠区域（回顾 $p(x_1, x_2) = p(x_1)p(x_2)$）。这正是更新后的分布——另一个高斯，由绿色椭圆表示，为 $\mathcal{N}(\mu_{t|t}, \Sigma_{t|t})$。由于取的是重叠，注意更新后的分布比两个父分布都小。预测步增大协方差（不确定性），更新步则用新测量 $z_t$ 减小协方差。因此协方差在预测步与更新步之后交替增大与减小。总体上，协方差不断减小直至收敛。

## 4 扩展 Kalman 滤波（Extended Kalman filter, EKF）

前面提到 Kalman 滤波对线性高斯系统成立。线性的过程模型与测量模型保证预测与更新后的分布仍是高斯。

Figure 7 展示了对随机变量 x 施加线性映射后，变换结果仍是高斯分布。相反，Figure 8 展示施加非线性映射会得到非高斯分布，此前的预测与更新方程不再适用。

Figure 7：把高斯分布推过线性映射 [4]。Figure 8：把高斯分布推过非线性映射。

扩展 Kalman 滤波（extended Kalman filter）在每个时刻 t 对输入做线性化，从而处理非线性的动力学函数与测量函数。

具体做法是在 $f(\mu_{t-1|t-1}, u_t)$ 与 $g(\mu_{t|t-1})$ 处做一阶 Taylor 展开。（译注：本节的测量函数记作 $g(\cdot)$，对应第 3 节的 $h(\cdot)$，照原文记号。）

预测步变为：

$$
\mu_{t|t-1} = f(\mu_{t-1|t-1}, u_t)
\tag{22}
$$

$$
\Sigma_{t|t-1} = A_{t-1} \Sigma_{t-1|t-1} A_{t-1}^T + Q_{t-1}
\tag{23}
$$

更新步变为：

$$
\mu_{t|t} = \mu_{t|t-1} + K_t (z_t - g(\mu_{t|t-1}))
\tag{24}
$$

$$
\Sigma_{t|t} = \Sigma_{t|t-1} - K_t C_t \Sigma_{t|t-1}
\tag{25}
$$

$$
K_t = \Sigma_{t|t-1} C_t^T (C_t \Sigma_{t|t-1} C_t^T + R_t)^{-1}
\tag{26}
$$

$A_t$ 与 $C_t$ 是非线性动力学模型与测量模型关于 x 的雅可比（Jacobians）：

$$
A_t(\mu_{t-1|t-1}, u_t) = \left. \frac{\partial f(x_t, u_t)}{\partial x_t} \right|_{x_t = \mu_{t-1|t-1}}
\tag{27}
$$

$$
C_t(\mu_{t|t-1}, u_t) = \left. \frac{\partial g(x_t, u_t)}{\partial x_t} \right|_{x_t = \mu_{t|t-1}}
\tag{28}
$$

（译注：式 (22)–(26) 中的 $A_t$、$C_t$ 不再是常矩阵，而是每步在当前估计处按式 (27)(28) 重新求值的雅可比；均值方程让状态直接穿过非线性函数（预测过 $f$、预测测量过 $g$），协方差与增益方程则与 Kalman 滤波保持同形。）

## 5 致谢（Acknowledgements）

（原文此节仅有标题，无正文。）

## References

[1] Chuong B Do. More on multivariate gaussians. URL http://cs229.stanford.edu/section/more_on_gaussians.pdf. [Online], 2008.

[2] Chuong B Do. The multivariate gaussian distribution. Section Notes, Lecture on Machine Learning, CS, 229, 2008.

[3] Jonathan R. Shewchuk. Eigenvectors and the anisotropic multivariate normal distribution. Lecture Notes, CS, 189, 2019.

[4] Cyrill Stachniss. Extended kalman filter. http://ais.informatik.uni-freiburg.de/teaching/ws13/mapping/pdf/slam04-ekf.pdf, 2013. Accessed: 2022-01-31.

## 要点回顾

- 状态估计的目标不是单个点值，而是后验分布 $p(x_t | z_{1:t}, u_{1:t})$，即信念 bel(t)，它量化对状态的不确定性。
- 「状态完备」假设给出两条条件独立：马尔可夫转移模型式 (1) 与测量模型式 (2)，它们是全文所有推导的起点；前者用在预测项推导的最后一行，后者用在式 (3)→(4) 与分母推导中。
- 递归滤波只用旧后验、新控制、新测量算出新后验，复杂度与历史长度无关，适合实时推断；$t=1$ 时上一后验不存在，用先验 $p(x_0)$ 启动（Figure 3 底部输入）。
- 贝叶斯滤波推导链：贝叶斯公式得式 (3)(4)，分母对 $x_t$ 边缘化、预测项对 $x_{t-1}$ 边缘化并用马尔可夫假设，最终归为预测（卷积转移模型）+ 更新（乘测量似然再归一化）两步。
- 离散贝叶斯滤波把信念写成向量式 (5)，预测是转移矩阵乘法式 (6)–(8)，更新是测量矩阵乘法加 $\mathbf{1}^T$ 归一化式 (9)(10)；连续状态空间分块离散化即直方图滤波。
- Kalman 滤波把模型换到动力系统视角式 (11)(12)；线性高斯假设下为式 (14)(15)，噪声为零均值高斯白噪声且互不相关。
- 多元高斯由 $\mu$ 与 $\Sigma$ 完全决定；$\Sigma$ 的特征向量是等值椭圆的轴方向、特征值决定各方向摊开程度（式 (13) 与 Figure 5）。
- 预测步式 (16)(17)：均值走动力学、协方差经 $A\Sigma A^T$ 传播并加过程噪声 $Q$，不确定性增大。
- 更新步式 (18)–(20)：新息 $z_t - C_t \mu_{t|t-1}$ 乘 Kalman 增益 $K_t$ 修正均值，协方差按 $\Sigma - K C \Sigma$ 收缩。
- 增益式 (21) 的两个极限：测量噪声 $R_t \to 0$ 时完全信任测量（$\mu_{t|t} = C_t^{-1} z_t$）；过程噪声趋于 0 时完全信任预测（$K = 0$）。
- 更新等价于预测测量分布 $\mathcal{N}(C_t\mu_{t|t-1}, C_t\Sigma_{t|t-1}C_t^T)$ 与实际测量分布 $\mathcal{N}(z_t, R_t)$ 相乘取重叠（Figure 6），协方差交替增减、总体收敛。
- EKF 在当前估计处对非线性 $f$、$g$ 做一阶 Taylor 线性化，用雅可比式 (27)(28) 顶替 $A_t$、$C_t$，其余方程与 Kalman 滤波同形（式 (22)–(26)）。
- 概率视角（分布形式的式 (1)(2)）与动力系统视角（加噪方程形式的式 (11)(12)）是同一模型的两套写法，Kalman 滤波是后者在线性高斯假设下的闭式实现。
- 贝叶斯滤波通用但归一化积分常难计算，故更宜看作「概率模板」：离散贝叶斯滤波与 Kalman 滤波都是它在特定假设下的适配。

## 原文出处

- Stanford CS231A Course Notes 10: Optimal Estimation（Bryan Chiang & Jeannette Bohg，2022-02-23），对应 2025 春季课件第 14 讲 Optimal Estimation 与第 15 讲 Optimal Estimation Cont.（Flipped Format）。
- 原文文末 References 共 4 条，已按原样列于上文；「5 Acknowledgements」一节原文无正文。
