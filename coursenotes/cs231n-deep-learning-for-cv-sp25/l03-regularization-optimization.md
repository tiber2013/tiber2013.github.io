---
title: "第 3 讲 · 正则化与优化：从损失函数到 Adam"
description: "CS231n Spring 2025 L03 视频笔记；对照 2026 春季课件补充新增内容。"
---

# 第 3 讲 · 正则化与优化：从损失函数到 Adam

> **Stanford CS231n 计算机视觉 · L03** ｜ 讲者：Stanford CS231n 课程团队 ｜ 68 分钟 ｜ [原视频](https://www.youtube.com/watch?v=dyNGd06MWn4)

> 课件版本：2026 春季 ｜ 视频版本：2025 春季。2026 课件新增或调整的内容集中在文末「2026 课件更新」一节。

## 核心结论

这一讲回答两个问题：怎么判断一个 W 好不好，以及怎么找到好的 W。前半段在 **data loss** 之外引入 **regularization**：故意让模型在训练集上差一点，换取测试集上的泛化；L1 偏好稀疏权重，L2 偏好分散的小权重。后半段从损失地形出发，讲梯度、数值与解析梯度、SGD 的三个毛病，以及 Momentum、RMSProp、Adam/AdamW 如何修补它们，最后落到学习率调度与二阶优化的取舍。贯穿全讲的线索是：训练集上的损失只是代理指标，真正目标是在没见过的数据上表现更好。

## 回顾：从像素到线性分类器 [00:05 - 09:44]

图像分类的任务是给定一张图，把它映射到一组候选标签中的一个。对人类来说这是“猫”，对计算机来说却只是一个 $32\times32\times3$ 的像素网格，这中间的 **semantic gap** 是视觉识别最根本的困难。视角、光照、形变、遮挡、杂乱背景、类内差异，又让问题无法靠 if-else 规则穷举。

![图像分类任务示意图：输入猫图片映射到标签集合中的cat](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0001_00-00-29.jpg)

![识别挑战总结图：视角、光照、形变、遮挡、杂乱背景、类内差异六类变化](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0002_00-00-56.jpg)

既然规则写不出来，就用 **data-driven** 方法：kNN 在训练集里找最近的样本，1-NN 直接复制标签，5-NN 让近邻投票。讲者再次强调数据划分：不要只分 train/test，而要留出 **validation set** 选超参数，例如 k；test set 只在最后评估模型从未见过的数据。

![kNN分类器回顾幻灯片，包含CIFAR-10数据集示例、1-NN与5-NN决策边界对比、训练/验证/测试集划分示意图及交叉验证误差曲线](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0003_00-02-54.jpg)

距离度量决定了“近”是什么意思。**L1（Manhattan）** 只能沿坐标轴走，等距点围成菱形；**L2（Euclidean）** 是直线距离，等距点围成圆。视频里的例子是：从原点向上 0.5、向右 0.5，L1 走的总路程是 1，和直接走一格一样。

![距离度量回顾幻灯片，对比L1曼哈顿距离与L2欧氏距离的公式及对应的单位球几何形状](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0004_00-04-37.jpg)

线性分类器把图像拉平成 3072 维向量，乘一个 $10\times3072$ 的 **weight matrix**，得到 10 个类别分数，再加 bias。同一个 $f(x,W)=Wx+b$ 有三种看法：代数视角是每行与输入做点积；视觉视角把每行看成一个类别模板；几何视角把每行看成一个超平面，$f=0$ 是决策边界，一侧为正、另一侧为负。几何视角最直观：如果红蓝点根本无法被一条直线分开，线性模型就不可能做好。

![线性分类器回顾：代数、视觉、几何三个视角及公式 f(x,W)=Wx+b](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0005_00-07-17.jpg)

有同学问模板是否等价于 kNN 的某个邻居，讲者明确说不是：模板由决策边界的方向决定，并不对应某一个训练样本。3072 则来自 $32\times32\times3$ 个像素通道值。

## 损失函数：给 W 打分的 data loss [09:44 - 11:36]

给定猫、车、青蛙三个样本和某个 W，可以得到分数矩阵。猫这一列里 car 的分数高于 cat，青蛙这一列正确类分数最低——直觉上这个 W 很差，但需要数学把它量化。

![损失函数定义：3个训练样本的得分矩阵及数据集平均损失公式](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0006_00-08-38.jpg)

**Loss function** 衡量当前分类器有多差。数据集上的损失是每个样本损失的平均：

$$
L = \frac{1}{N}\sum_{i=1}^{N} L_i\left(f(x_i,W), y_i\right)
$$

分类任务最常用的是 **softmax** 或 **cross-entropy** 损失：

$$
L_i = -\log\left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}}\right)
$$

正确类概率很低时 loss 很大，概率接近 1 时 loss 接近 0。softmax 的作用是把任意一组实数变成和为 1 的概率分布，分数越高对应的相对概率越大；这正是它适合分类的原因。

![线性分类器回顾：Softmax损失函数公式与计算图结构](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0013_00-21-28.jpg)

以上只是 **data loss**，它只关心预测是否匹配训练数据。如果只最小化这一项，模型会把训练集拟合到极致，却可能在测试集上崩溃，所以还需要第二项。

## 正则化：偏好简单模型 [11:36 - 15:19]

讲者用一个玩具例子说明动机：横轴是输入、纵轴是输出，$f_1$ 穿过所有训练点，data loss 几乎为零；$f_2$ 没有完美穿过每个点，但在同分布的新数据上表现更好。$f_1$ 拟合了噪声，$f_2$ 抓住了趋势。**Regularization** 就是主动压制“在训练集上做得太好”的倾向。

![正则化偏好简单模型的示意图，对比过拟合曲线f1与简单线性模型f2](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0008_00-14-00.jpg)

这背后的哲学是 **Occam's razor**：面对多个相互竞争的假设，先选最简单的；只有当简单假设被证伪时，才诉诸更复杂的模型。正则化把这种偏好写进损失：

$$
L(W) = \frac{1}{N}\sum_{i=1}^{N} L_i\left(f(x_i,W), y_i\right) + \lambda R(W)
$$

第一项要求预测匹配训练数据，第二项阻止模型在训练数据上过度优秀。$\lambda$ 是 **regularization strength**，也是一个超参数：取 0 等于不正则，越大惩罚越强。和 k 一样，$\lambda$ 应该用 validation set 来选。

![正则化损失函数公式，由数据损失项与正则化项λR(W)组成](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0009_00-14-40.jpg)

## L1、L2 与稀疏性 [15:19 - 21:19]

最简单的正则项直接惩罚权重大小。**L2** 把每个权重平方后求和：

$$
R(W) = \sum_k \sum_l W_{k,l}^2
$$

**L1** 则取绝对值求和：

$$
R(W) = \sum_k \sum_l |W_{k,l}|
$$

**elastic net** 把两者按系数 $\beta$ 混合：

$$
R(W) = \sum_k \sum_l \left(\beta W_{k,l}^2 + |W_{k,l}|\right)
$$

平方带来一个关键差异：很小的值（比如 0.001）平方后更小，惩罚几乎消失，所以 L2 容忍大量接近零但不为零的权重；L1 不做平方，小权重仍要付全额代价，于是更倾向于把它们压到零或非常接近零。实践中 L2 得到分散的小权重，L1 得到**稀疏**权重，相当于一个系数更少的线性模型。

![正则化损失函数公式及L1/L2/Elastic Net示例的幻灯片](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0010_00-15-26.jpg)

课堂做了一道投票题。取 $x=[1,1,1,1]$，$w_1=[1,0,0,0]$，$w_2=[0.25,0.25,0.25,0.25]$，两者与 x 的内积都等于 1，data loss 相同。L2 会选 $w_2$：$w_1$ 的惩罚是 1，$w_2$ 是 $4\times(1/4)^2=1/4$，只有 $w_1$ 的四分之一，所以 L2 喜欢把权重“摊开”。

![L2正则化偏好权重分散的结论幻灯片，高亮w2并引出L1正则化问题](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0012_00-20-55.jpg)

接着问 L1 会选谁，这是一个陷阱题：两者的绝对值之和都等于 1，正则项完全相同，单看损失无法区分。但讲者补充，实际训练中更可能看到 $w_1$ 这样的稀疏解，因为只要把某个权重置零而 data loss 几乎不变，总损失就会下降。优化的目标始终是 data loss 与正则项的联合最小化。

## 为什么要正则化，以及它的边界 [18:29 - 24:22]

讲者把正则化的作用归为三类。第一，表达对权重的偏好：如果问题需要稀疏解，就选 L1；需要分散的小权重，就选 L2。第二，让模型更简单，从而在测试数据上更好；对多项式模型压制高阶项就是典型例子。第三，改善优化：平方项像抛物线，给损失函数增加曲率，某些情况下能让优化更快、更容易找到全局最小。

但正则化不必然等于“更简单的模型”。它的本质是“在训练集上变差、在测试集上变好”，而 **dropout** 这类方法反而让模型结构更复杂，却能提升测试表现。正则化本身也是一个活跃的研究方向，每年都有新方法，课程只覆盖一小部分。

## 优化：损失地形与随机搜索 [24:22 - 26:45]

有了损失函数，下一步是找让它最小的 W，这就是 **optimization**。讲者用“损失地形”作比喻：水平面是参数，竖直方向是 loss，人蒙着眼睛站在山坡上，只能感知脚下的坡度，要摸到谷底。这个比喻的关键是“看不见远处”，只能依赖局部斜率。

最朴素的办法是随机试：循环 1000 次，每次随机生成一个 W，保留 loss 最低的那个。它在 CIFAR-10 上能到 15.5% 准确率，不算完全没用，但当时的最佳方法已经接近 99.7%。

![测试集评估代码：计算预测分数、预测类别及准确率](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0014_00-26-25.jpg)

## 梯度：数值梯度、解析梯度与 gradient check [26:45 - 33:57]

“跟着坡度走”就是 **gradient descent**。一维时导数由极限定义：

$$
\frac{df(x)}{dx} = \lim_{h\to 0}\frac{f(x+h)-f(x)}{h}
$$

多维时 **gradient** 是各个偏导数组成的向量；沿任意方向的方向导数等于该方向与梯度的点积，而**最陡下降方向是负梯度**。

计算梯度有两种方式。**Numerical gradient** 直接取一个很小的 h（例如 0.0001），算函数值差再除以 h：好写，但要对每个维度循环，慢；而且只是近似，浮点误差可能很大。**Analytic gradient** 用微积分和链式法则推导：精确、快，但手写代码容易出错。实践中的标准做法是始终用解析梯度训练，再用数值梯度做 **gradient check**，确认两者在同一个量级。课程作业也会用 gradient check 检查实现。

有同学问：如果损失函数不可微但更好，能否用数值梯度？讲者的回答是，通常很难构造出“更好但不可微”的损失；如果函数在大部分区域都不可微，基于最陡下降的方法就难以找到谷底。凸函数很适合梯度下降，非凸且不可微的函数则不适合。

## 梯度下降与 SGD [33:57 - 37:54]

最基本的更新是：

$$
W \leftarrow W - \alpha \nabla_W L(W)
$$

$\alpha$ 是步长或学习率。停止条件有两种：预先设定迭代次数，或者当 loss 的变化小于某个容差时停止。前者最常用。

全量梯度下降要对整个训练集求和，数据大时非常昂贵。**Stochastic gradient descent（SGD）** 每次只采样一个 **mini-batch**，用子集的梯度近似全量梯度。之所以叫 stochastic，是因为每一步用的是随机子集。实践中不会完全独立随机采样，而是把所有样本按随机顺序过一遍再重新洗牌；完整过一遍数据叫一个 **epoch**。

```python
while True:
    data_batch = sample_training_data(data, 256)  # 采样 256 个样本
    weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
    weights += -step_size * weights_grad
```

## SGD 的三个麻烦 [37:54 - 42:56]

第一个问题是**病态条件**：损失在一个方向变化很快、另一个方向变化很慢，形成狭长山谷。梯度下降会沿陡方向来回抖动，沿平方向前进极慢；步长太大甚至会振荡出山谷。用 Hessian 的语言说，最大与最小奇异值之比（**condition number**）很大。

第二个问题是**局部极小值与鞍点**：梯度为零，更新停住。高维模型中鞍点比局部极小值更常见，因为只要某些方向向上、某些方向向下，就构成鞍点；Dauphin 等人的 NIPS 2014 论文讨论了这一现象。在鞍点中心，所有方向的梯度都为零，但稍往旁边走 loss 就能显著下降。

第三个问题是**梯度噪声**：mini-batch 只是子集，每一步的方向不会精确指向全量梯度，更新会带噪声。

## Momentum：让更新带上速度 [42:56 - 49:22]

**Momentum** 把优化想象成小球滚下山：速度会累积，遇到小坑可以冲过去，遇到鞍点也不会立刻停住。

$$
v_{t+1} = \rho v_t + \nabla f(x_t), \qquad x_{t+1} = x_t - \alpha v_{t+1}
$$

$\rho$ 通常取 0.9 或 0.99，控制“惯性”有多强。更新量不再是当前点的梯度，而是过去梯度的滑动平均。它同时缓解前面三个问题：在狭长山谷中，沿谷底方向的梯度会持续累加，横向的来回分量会互相抵消；在鞍点处，之前积累的速度会带着参数继续走；噪声也会被平均掉。

```python
vx = 0
while True:
    dx = compute_gradient(x)
    vx = rho * vx + dx
    x -= learning_rate * vx
```

代价是可能冲过头，短期内收敛未必更快，但通常能到达更好的极小值。讲者也提醒，这些结论大多是经验性的，具体模型上 SGD 不加 momentum 也可能更好。动量还有等价的另一种写法，把学习率折进速度里，看起来不同但产生相同的参数序列。

## RMSProp：按维度自适应步长 [49:24 - 53:35]

**RMSProp** 由 Tieleman 与 Hinton 在 2012 年提出，思路是记录每个维度梯度平方的滑动平均，再用它缩放步长：

$$
s_{t+1} = \gamma s_t + (1-\gamma) g_t^2, \qquad
x_{t+1} = x_t - \alpha \frac{g_t}{\sqrt{s_{t+1}}+\epsilon}
$$

梯度一直很大的维度，分母大，实际步长被压小；梯度一直很小的平坦维度，分母小，步长被放大。因此它是“逐参数的学习率”：陡方向减速，平方向加速。这正好对症前面说的狭长山谷——它不只是改变步长大小，还改变了前进方向，让参数更快转向谷底。

```python
grad_squared = 0
while True:
    dx = compute_gradient(x)
    grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dx * dx
    x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
```

把 SGD、Momentum、RMSProp 的轨迹画在一起，可以看到 Momentum 会冲过头再回来，SGD 沿陡方向反复抖动，RMSProp 则较快转向平坦的中心方向。

## Adam 与 AdamW [53:35 - 59:15]

**Adam** 把 Momentum 和 RMSProp 合在一起：一阶矩 $m_t$ 是梯度的滑动平均，二阶矩 $v_t$ 是梯度平方的滑动平均。

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

如果直接用它更新，第一步会出问题：$m_0=v_0=0$，而 $\beta_2$ 通常取 0.999，非常接近 1，于是 $v_1$ 很小，分母接近零，即使梯度很小也会产生一个巨大的初始步。解决办法是**偏差修正**：

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad
\hat{v}_t = \frac{v_t}{1-\beta_2^t}, \qquad
x_t = x_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

$\beta_1=0.9$、$\beta_2=0.999$、学习率 $10^{-3}$ 或 $5\times10^{-4}$，是很多模型不错的起点。

**AdamW** 处理的是正则化与优化器的交互。标准 Adam 把 L2 项算进梯度，因此一阶矩、二阶矩都依赖权重本身；AdamW 只用 data loss 计算矩，再把 weight decay 单独加在更新之后。这样做的理由是：不希望动量成为权重的函数，而希望它只反映损失地形。讲者提到 Meta 的 Llama 系列就使用 AdamW，在不少设置下它略好于 Adam，但最终仍要靠实验比较。

## 学习率与调度 [59:15 - 63:35]

学习率是最重要的超参数之一。太大，loss 会爆炸或振荡出地形；略大，会在谷底附近来回跳，无法继续下降；太小，收敛极慢。有意思的是，同一组曲线里看起来“不好”的学习率，换到训练的另一阶段可能正好合适，所以现代模型几乎都会在训练中改变学习率。

常见策略包括：每隔固定迭代次数把学习率乘 0.1（ResNet 训练中很常见）；**cosine decay**，从最大值沿半个余弦曲线降到零；线性衰减；以及按 $1/\sqrt{t}$ 衰减。余弦调度的形式是：

$$
\alpha_t = \frac{1}{2}\alpha_0\left(1+\cos\frac{t\pi}{T}\right)
$$

另一类技巧是 **linear warmup**：先用几千次迭代把学习率从 0 线性升到最大值，再接入后续调度，避免一开始的大步长把 loss 炸掉。Warmup + cosine 是很常见的组合。还有一个经验规则：**batch size 增大 N 倍，学习率也应增大 N 倍**，这被称为 linear scaling rule。

## 二阶优化为什么在深度学习中不实用 [63:35 - 65:58]

一阶方法只用梯度做线性近似，然后沿负梯度走一步。**Second-order optimization** 则用梯度和 Hessian 做二次近似，直接跳到这个二次函数的最小点，即 Newton 更新：

$$
\theta^* = \theta_0 - H^{-1}\nabla_\theta J(\theta_0)
$$

它在一些低维问题上非常有效，但深度网络用不了：Hessian 有 $O(N^2)$ 个元素，求逆是 $O(N^3)$，而 N 是百万到十亿级参数，显存根本放不下。只有当模型较小、或者你愿意做全批量更新时，才值得考虑二阶或准牛顿方法。

## 实践建议与下一讲 [65:58 - 68:38]

讲者给出的默认建议是：新问题先用 **Adam/AdamW**，即使学习率恒定通常也能工作；想再稳一点，就用 linear warmup + cosine decay。**SGD + Momentum** 有时能超过 Adam，但需要调更多学习率和调度；如果数据量小到能做全批量更新，可以尝试二阶方法。

最后是过渡到下一讲：这一讲优化的还是线性模型 $f=Wx$，下一讲换成两层网络

$$
f = W_2\max(0, W_1x)
$$

两个权重矩阵之间插入 **ReLU** 这样的非线性。没有非线性，多层线性变换仍然等价于一个线性分类器；有了它，模型才能把线性不可分的数据变换到可分空间。下一讲会讲神经网络和 **backpropagation**，回答“多层网络怎么算梯度”。

## 2026 课件更新

以上正文以 2025 春季视频为准。对照 2026 春季课件，主要有这些差异：

- **行政信息前置**：2026 课件改为以行政信息开场：Assignment 1 周四 4/16 23:59 截止、项目提案周四 4/23 截止、office hours（只答高层问题，No Code Show Policy）、Ed 置顶帖、CGOE 期中考试邮件申请，以及 Student Spotlight。2025 视频从回顾直接开始，没有这些内容。
- **距离度量改成具体数值例子**：2026 课件用 O、A、B 三点分别计算 L1 与 L2，展示两种距离都等于 1，再配单位球图。概念与视频一致，但呈现更形式化。
- **正则化例子扩充**：2026 课件在 L1/L2/elastic net 之外，把 **Dropout**、**Batch normalization**、**stochastic depth**、**fractional pooling** 列为更复杂的正则化方法；2025 视频只口头提到 dropout。
- **优化器内容重新编排**：2026 课件把 **AdamW** 单独成节，明确标准 Adam 在梯度里算 L2、AdamW 在矩计算之后再加 weight decay；Nesterov Momentum、AdaGrad、准牛顿/L-BFGS、Softmax vs SVM、正则化不唯一性（$2W$ 也使损失为零）等被放进 “Appendix / Enrichment Material”，正文不展开。
- **SGD 与调度细节**：2026 课件注明常见 batch size 为 32/64/128（视频示例用 256），step decay 以 ResNet 在 epoch 30/60/90 乘 0.1 为例，并给出 cosine、linear、inverse sqrt 的完整公式，以及 warmup 约 5,000 次迭代、batch size 扩大 N 倍则学习率扩大 N 倍的经验规则。
- **二阶优化的量级说明**：2026 课件明确写出 Hessian 的 $O(N^2)$ 元素与 $O(N^3)$ 求逆，并在附录补充 L-BFGS 等准牛顿方法；正文结论与视频一致。
- **结尾仍是两层网络**：2026 课件用 $f=W_2\max(0,W_1x)$ 和“为什么需要非线性”的可视化收尾，指向下一讲的神经网络与反向传播。

## 要点回顾

- **data loss** 衡量预测与训练标签的匹配程度，softmax/cross-entropy 是最常用的分类损失。
- **regularization** 故意让训练集表现变差，以换取测试集泛化；总损失是 data loss 加 $\lambda R(W)$。
- **L2** 偏好分散的小权重，**L1** 偏好稀疏权重；二者内积相同时 L2 的惩罚可相差数倍，而 L1 可能打平。
- 正则化的作用包括表达权重偏好、简化模型、改善优化曲率，但不保证模型一定更简单（如 dropout）。
- **gradient** 指向最陡上升方向，负梯度是最陡下降方向；解析梯度用于训练，数值梯度用于 **gradient check**。
- **SGD** 用 mini-batch 近似全量梯度，一个 epoch 是完整过一遍数据。
- SGD 的三个问题是病态条件、局部极小值/鞍点、mini-batch 梯度噪声。
- **Momentum** 累积速度，**RMSProp** 按维度缩放步长，**Adam** 结合两者并用偏差修正处理零初始化。
- **AdamW** 把 weight decay 放在矩计算之后，避免正则项污染一阶、二阶矩。
- 学习率调度（step、cosine、linear、inverse sqrt、warmup）和 linear scaling rule 是现代训练的标准工具。
- 二阶优化受 Hessian 的 $O(N^2)$ 内存与 $O(N^3)$ 求逆限制，在大网络上不实用。
- 下一讲把线性模型换成带 ReLU 的两层网络，并引入 **backpropagation**。

## 视频来源

https://www.youtube.com/watch?v=dyNGd06MWn4
