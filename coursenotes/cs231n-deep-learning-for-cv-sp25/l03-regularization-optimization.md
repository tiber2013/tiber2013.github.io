---
title: "第 3 讲 · 正则化与优化：从损失函数到 Adam"
description: "Stanford CS231n Spring 2025 L03 中文视频笔记（2026-11 重写版）"
---

# CS231n Spring 2025 · Lecture 3 正则化与优化：从损失地形到 AdamW

**核心结论**：本讲在上一讲线性分类器的基础上补齐两件大事：正则化与优化。正则化把总损失写成数据损失加 $\lambda R(W)$，通过在训练集上「故意变差」换取在未见测试数据上更好，L2 偏好把权重摊薄、L1 偏好稀疏，但 dropout 之类方法说明正则化未必让模型更简单。优化则从「蒙眼在损失地形里摸坡度」出发，依次讲数值梯度与解析梯度、梯度检查、梯度下降与小批量 SGD，再针对高条件数、局部极小、鞍点与梯度噪声引入 Momentum、RMSprop，最终组合成现代深度学习的默认选择 Adam／AdamW。讲者还给出学习率调度（step、cosine、linear、warm-up、线性缩放律）与二阶 Hessian 方法的适用边界，并以「两层神经网络 + ReLU 处理线性不可分数据」预告下一讲。实践建议是：新问题上先用 Adam 或 AdamW（可配常数学习率或 linear warm-up + cosine decay），SGD+Momentum 有时更优但需要多调参。

## 1. 回顾：图像分类、kNN 与线性分类器 [00:05 - 09:44]

### 1.1 挑战、数据驱动与 kNN 回顾 [00:05 - 05:26]

上讲把图像分类定为「输入图像、映射到一组预设标签之一」。识别之所以难，是因为语义鸿沟（人看到猫，机器看到像素网格）、光照、可形变物体、遮挡（猫藏在沙发垫下只露尾巴）、背景杂乱与类内差异——这些变化让「写 if-else 规则」不可行，于是转向数据驱动方法。最简单的模型是 K 近邻：1-NN 找最近的一个训练点赋其标签，K-NN 取多个近邻的多数类；数据应分成 train／validation／test，用验证集选超参数 K（示例为 1 与 5），在验证集准确率随 K 变化的曲线上挑最高点，测试集只留到最后评估全新数据。距离度量方面，讲者借 Ed 上的帖子澄清了上讲的疑问：L1（曼哈顿）只能上下左右走、不能走对角线，等距点是菱形／直线（如向上 0.5 再向右 0.5，总距离仍为 1）；L2（欧氏）是直线距离，等距点构成圆。

![kNN分类器回顾：CIFAR-10数据集、1-NN与5-NN决策边界对比及交叉验证曲线](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0001_00-02-54.jpg)

### 1.2 线性分类器的三种视角与 3072 [05:26 - 09:44]

线性分类器把 $32 \times 32 \times 3$ 的图像拉直成 3072 个数（$32 \times 32 \times 3 = 3072$），与 $10 \times 3072$ 的权重矩阵 $W$ 相乘得到 10 个类别分数，再加每类一个偏置 $b$（10 维向量），即 $f(x,W) = Wx + b$。三种视角做的是同一件事：代数视角下 $W$ 每行独立与输入点积得分类分数；视觉视角把每行重排回图像形状，得到该类的「模板」（template）——现场学生问它是否等价于 kNN 的某个近邻，答案是否定的，模板来自那条决策直线而非某个具体数据点；几何视角把每行看成输入空间里令方程为 0 的直线／超平面，上方分数为正、下方为负，也正因如此一眼能看出蓝红两类数据无法被一条直线分开。

![线性分类器回顾幻灯片，展示代数、视觉和几何三种视角及公式 f(x,W)=Wx+b](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0002_00-07-07.jpg)

![线性分类器详细图解，包含矩阵乘法示例、模板匹配及超平面几何解释](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0003_00-09-09.jpg)

## 2. 从数据损失到正则化：完整损失与 λ [09:44 - 15:19]

用一个具体例子引入：某线性模型给出的分数里，猫图的 car 分数高于 cat、青蛙图更是把正确类打成三者最低，直觉上「不好」，但如何数学化？答案是损失函数（loss function），量化分类器有多差。给定训练集 $(x_i, y_i)$，把每张图送进 $f(x_i, W)$ 得到预测、与真实标签 $y_i$ 比较，再对整集取平均，即数据损失（data loss）：

$$L_{\text{data}} = \frac{1}{N}\sum_{i=1}^{N} L_i(f(x_i, W), y_i)$$

上讲讲的 Softmax／交叉熵损失是分类最常用的 $L_i$：给正确类低概率则损失很高，给正确类高概率则损失很低；这里不再展开，只强调「预测与训练数据吻合得多好」。

课件随后给出本讲的核心式子：总损失 = 数据损失 + 正则化项。

![正则化总损失函数公式，分解为数据损失和正则化项两部分](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0004_00-12-29.jpg)

$$L(W) = \frac{1}{N}\sum_{i=1}^{N} L_i(f(x_i, W), y_i) + \lambda R(W)$$

正则化项的意图与直觉相反：它要**阻止模型在训练数据上做得太好**——在训练集上变差，换取在全新测试数据上更好，这正是正则化的意义。一个玩具例子把这点讲透：用函数拟合一堆点，$F_1$ 穿过每一个点、数据损失几乎为 0，却明显过拟合；$F_2$ 不完美穿过每个点，但在同分布的新数据上表现更好。这背后是奥卡姆剃刀（Occam's razor）——多个竞争假设里先选最简单的，确有不对再逐步复杂化。

![正则化（Regularization）概念及L1、L2正则化公式详解幻灯片](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0005_00-18-12.jpg)

式子里的 $\lambda$ 是正则化强度，本身又是一个超参数，同样可以用训练／验证集来定：它是一个 0 到无穷之间的浮点数，0 表示完全不正则化，越大则抑制拟合的力度越强，是一个可调的旋钮。课件还提示，课程后面会见到 dropout、batch normalization 等更复杂的正则化形式，有的甚至会改动模型的层，是每年都有新论文的活跃研究方向，本课只覆盖很小一部分。

## 3. L1 / L2 正则化：权重偏好、稀疏与「更简单」的边界 [15:19 - 24:22]

最常见的两种是 L2 与 L1。L2 把权重矩阵每个元素平方后求和，$R(W) = \sum_k \sum_l W_{k,l}^2$；L1 则取绝对值之和，$R(W) = \sum_k \sum_l |W_{k,l}|$。差别在实践里很关键：L2 会把很小的值再平方（如 0.001 变成更小），对接近 0 的权重几乎不罚，于是倾向于留下大量「小而非零」、整体摊开（spread out）的权重；L1 不平方，若某个值置 0 而性能几乎不变，优化就会把它推向 0，于是得到更多真正为 0 的稀疏权重。

![正则化表达偏好的幻灯片，对比L1和L2正则化对稀疏向量与均匀向量的偏好](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0006_00-20-55.jpg)

课堂投票把这个偏好算得很具体：输入 $x = [1,1,1,1]$，$w_1 = [1,0,0,0]$ 与 $w_2 = [0.25,0.25,0.25,0.25]$ 都给出 $w_1^T x = w_2^T x = 1$，数据损失相同。L2 下 $w_1$ 的正则项是 1，$w_2$ 是 $4 \times (1/16) = 1/4$，相差 4 倍，因此 L2 偏好 $w_2$；L1 下两者都求和为 1，完全等价（这是个陷阱题，实践里因稀疏性可能仍更常看到 $w_1$，但纯从损失看二者相同），L1 真正占优的情形要换成类似 0.9 这种值。优化之所以会「推向零」，是因为我们要最小化数据损失与正则项之和：当数据损失变化不大、而正则项能降下来时，整体更优的解会被选中。

讲者归纳了正则化的三条理由：表达对权重的偏好（想要摊开还是稀疏，选 L2 还是 L1）；视用法让模型更简单、从而在测试集上更好（例如对高次多项式项重罚）；以及 L2 因平方项形似抛物线、具凸性，在某些优化里还能改善优化过程、加快训练（超出本课范围）。随后重述 Softmax：把分数取指数变正、除以总和得到和为 1 的概率分布，再对正确标签的概率取负对数；全损失就是各样本 $L_i$ 平均再加正则项。Softmax 好用在于能把任意一组浮点数变成按相对大小分配的概率列表。

现场问答补充了两点：正则化可看成对权重「幅值」的约束，但它不总给出更简单的模型——dropout 反而让模型更复杂，却提升测试表现，本质仍是「训练集变差、测试集变好」；L1 因系数更少而线性模型更简单，这一点相对直观。

## 4. 优化的目标：损失地形与两条朴素策略 [24:22 - 27:33]

有了衡量 $W$ 好坏的（数据＋正则）损失，剩下的问题就是怎么找到最好的 $W$，这是本讲后半部分的主题。描述优化通常用「损失地形（loss landscape）」：Z 轴是损失值，X／Y 是参数位置，人要在地形里找最低点。讲者强调，这个类比只有把人想成**被蒙住眼**才准确——看不到远处山谷，只能靠脚下当前点的坡度判断方向，这恰好对应我们只有局部梯度信息的现实。

策略 #1 是「很糟糕但能跑」的随机搜索：随机试 1000 个 $W$ 取损失最小者。

```python
bestloss = float("inf")
for num in xrange(1000):
    W = np.random.randn(10, 3073) * 0.0001
    loss = L(X_train, Y_train, W)
    if loss < bestloss:
        bestloss = loss
        bestW = W
```

![策略#1：随机搜索（Random search）的Python实现代码及运行日志](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0008_00-25-59.jpg)

在 CIFAR-10 上它能到 15.5%，比随机基线好，但离现代深度学习的 99.7% 差得很远。策略 #2 才是本课训练所有模型、也是几乎所有深度学习模型训练的根本方式：**跟随斜率（follow the slope）**——蒙眼感受脚下坡度，始终朝下坡方向走。

## 5. 梯度：从一维导数到解析梯度与梯度检查 [27:33 - 33:51]

一维里就是导数，用极限定义：

$$\frac{df(x)}{dx} = \lim_{h \to 0} \frac{f(x+h)-f(x)}{h}$$

多维里对每个分量分别做同样的事，得到向量即梯度（gradient），记为 $\nabla_W L$；某方向上的斜率可由梯度与该方向做点积得到，而**最速下降方向是负梯度**（梯度指向上坡，负梯度指向下坡）。

![策略#2：跟随斜率，展示一维导数公式及多维梯度与负梯度的定义](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0009_00-27-33.jpg)

怎么算导数？最直白的是数值梯度：取很小的 $h$（如 0.0001），逐个元素加 $h$ 算损失变化再除以 $h$。它好写，但很慢（要遍历每个权重）、只是近似、且浮点误差明显。更正规的是解析梯度：把 $X_i, Y_i$ 视作常数，把损失看成 $W$ 的函数（含 log、exp 与链式法则），用微积分直接求 $\nabla_W L$，这在作业里会动手推导。课件给出含 Softmax 与 L2 正则的具体形式与计算目标：

$$L = \frac{1}{N}\sum_{i=1}^{N} L_i + \sum_k W_k^2, \quad L_i = -\log\frac{e^{s_{y_i}}}{\sum_j e^{s_j}}, \quad s = f(x;W) = Wx$$

![损失函数关于W的表达式及解析梯度的计算目标，配有牛顿和莱布尼茨画像](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0010_00-30-39.jpg)

两者对照：数值梯度近似、慢但易写、不易在实现上出错；解析梯度精确、快，但从零写新梯度代码可能藏 bug，所以实践中用 **gradient check**——把解析梯度与极小 $h$ 的数值梯度对比，确认二者在同一量级，作业里也会做。问答还谈到可微性：凸函数用梯度／最速下降很顺；若损失在整个定义域上不可微、非凸、是一堆不连通的点，沿最陡方向走未必能到最优，这类方法就不好用了。

## 6. 梯度下降与随机梯度下降 [33:57 - 37:54]

梯度下降（gradient descent）把上述直觉落成算法：每步用损失函数、数据与当前权重算出梯度，再沿负梯度走一个由步长（step size／learning rate）决定的距离：

```python
while True:
    weights_grad = evaluate_gradient(loss_fun, data, weights)
    weights += - step_size * weights_grad
```

![展示朴素梯度下降算法的伪代码实现](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0011_00-35-51.jpg)

俯视的损失地形常用等高线着色（紫色最高、红色谷底），箭头表示沿负梯度的固定步长。步长固定，但接近平坦处梯度变小，有效步长也随之变小。何时停？公式里是 `while True` 永不自己停，实践里要么预设迭代次数，要么设容差——当损失每次只降 1e-5、1e-9 这类量级时认为够好。

问题在于全量梯度要对整个训练集求和，$N$ 很大时代价高。于是有随机梯度下降（SGD）：每次只抽一个子集（minibatch／batch）来估梯度，课件示例采样 256 个样本：

$$L(W) = \frac{1}{N}\sum_{i=1}^{N} L_i(x_i, y_i, W) + \lambda R(W), \quad \nabla_W L(W) = \frac{1}{N}\sum_{i=1}^{N} \nabla_W L_i(x_i, y_i, W) + \lambda \nabla_W R(W)$$

```python
while True:
    data_batch = sample_training_data(data, 256)
    weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
    weights += - step_size * weights_grad
```

![随机梯度下降(SGD)的损失函数与梯度公式，以及小批量梯度下降的Python伪代码实现](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0012_00-36-37.jpg)

之所以叫「随机」，是因为每步抽的是随机子集；实际不会纯随机乱抽，而是以随机顺序把所有样本过一遍再循环，这一整遍叫一个 epoch。常见 minibatch 大小还有 32／64／128。

## 7. SGD 的四大病态与动量 [37:54 - 48:53]

纯 SGD 有四类麻烦。① 病态条件（poor conditioning）：窄谷两侧很陡、沿谷底很平，固定步长会上下震荡甚至「冲出山谷」，即使不冲出也大量抖动、朝谷心进展缓慢；数学上对应损失的高条件数（condition number，Hessian 二阶导矩阵最大与最小奇异值之比）。② 局部极小：落到平坦处梯度为 0，卡住不动。③ 鞍点（saddle point）：形如马鞍，中心处各方向梯度都为 0，稍往前却能大幅下降；经验上参数越多、维度越高，鞍点越常见，且这是所有只靠梯度的优化器共有的问题。④ 梯度噪声：minibatch 只是全数据的子集，每步更新并不精确指向极小值。

![SGD优化问题#3幻灯片：损失函数与梯度公式及等高线地形图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0013_00-42-20.jpg)

一个很漂亮的办法是**加动量（Momentum）**，物理上就像滚下山的球会因速度冲过局部极小与鞍点、在一致方向上越滚越快、并把抖动与噪声平均掉。它不再用当前点的梯度直接更新，而是维护一个速度 $v$：

$$v_{t+1} = \rho v_t - \alpha \nabla f(x_t), \qquad x_{t+1} = x_t + v_{t+1}$$

```python
vx = 0
while True:
    dx = compute_gradient(x)
    vx = rho * vx - learning_rate * dx
    x += vx
```

![SGD+Momentum 解决局部极小值、鞍点、病态条件及梯度噪声的示意图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0014_00-47-24.jpg)

![SGD+Momentum 的两种等价数学公式及对应的 Python 伪代码实现](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0015_00-49-03.jpg)

$\rho$ 是动量系数，决定「过去」与「当下」梯度在滑动平均里的权重，越大越依赖上一步速度；$\alpha$ 仍是步长。课件列出两种写法（先更新速度再加到位置、或直接更新位置），本质等价，并引用 Sutskever et al. 2013。讲者也给出边界：动量可能让收敛变慢（会过冲再回来），但平均而言能找到更好的极小点；它是经验上在神经网络这类模型里有效，实践中人们仍会试多种优化器，也可能用不同随机种子跑多次以防正好卡在鞍点上。

## 8. RMSprop、Adam 与 AdamW [49:24 - 59:15]

RMSprop 是 Hinton 组 2012 年的较老方法：不再只用速度，而是维护**平方梯度的滑动平均**，更新时除以它的平方根。于是梯度（平方）大的方向被大分母压小步、平坦方向被小分母放大步，正好回应了「能否改变步进方向」的提问。三种方法在窄谷上的轨迹对比很直观：SGD 慢、Momentum 会过冲再回来、RMSprop 很快转向谷心。

把 Momentum 与 RMSprop 合起来，就是现代深度学习最流行的 **Adam**（Kingma and Ba, ICLR 2015）：

```python
first_moment = 0
second_moment = 0
while True:
    dx = compute_gradient(x)
    first_moment = beta1 * first_moment + (1 - beta1) * dx
    second_moment = beta2 * second_moment + (1 - beta2) * dx * dx
    x -= learning_rate * first_moment / (np.sqrt(second_moment) + 1e-7)
```

![Adam优化器简化版伪代码，标注Momentum与RMSProp两部分并提问首步问题](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0016_00-55-28.jpg)

红框是 Momentum 部分（一阶矩、$\beta_1$），蓝框是 RMSprop 部分（二阶矩、$\beta_2$），所以 Adam「有点像带 momentum 的 RMSProp」：既在平坦方向加速、陡峭方向减速，又逐渐积累同向速度。但按上面这个朴素写法，**第一个时间步会出问题**：$\beta_1$、$\beta_2$ 通常初始化得很接近 1（0.9、0.999），两个矩又从 0 起步，首步 second_moment 很小，除以接近 0 的数会产生一个远超梯度应有幅度的巨大初始步。修正办法是加 bias correction（偏差修正），用与训练步数 $t$ 相关的项去归一化两个矩；随着 $t$ 增大，这些偏置项就不再需要。这一步会在作业里实现。

![AdamW优化器伪代码及权重衰减项添加位置示意图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0017_00-58-14.jpg)

最后一个话题是正则化与优化器如何交互。L2 正则化怎么影响 Adam 并不显然：默认 Adam 在算梯度时就把 L2 与数据损失一起算进去（同一个函数）；而 **AdamW** 只用数据损失做全部矩计算，把正则项（weight decay，权重衰减）放在最后单独加。这样速度、动量只依赖损失、不依赖权重值，能更独立地在损失地形里行进。Meta 的 LLaMA 系列用的就是 AdamW，很多设置下它略优；但说到底是经验性的，两种都试、哪个好用哪个。

## 9. 学习率调度、二阶优化与结论 [59:15 - 68:30]

学习率（步长）怎么选也有讲究：太高会震荡出地形、损失变大；太低收敛极慢；高但没冲出去时又会在极小点附近来回撞、降不下去；理想的是损失先快速下降、之后仍持续改进。而且学习率**不必固定**，现代最好的模型几乎都会在训练中调整它。课件给出几种衰减（decay）策略：

- **Step**：在固定的几个点降，例如训练 ResNets 时在第 30、60、90 个 epoch 后把学习率乘 0.1。
- **Cosine**：半条余弦波从最大值降到 0，$\alpha_t = \frac{1}{2}\alpha_0\left(1 + \cos\frac{t\pi}{T}\right)$。
- **Linear**：直线下降，$\alpha_t = \alpha_0\left(1 - \frac{t}{T}\right)$。
- 还有 inverse square root 等，变化方式几乎无限；$\alpha_0$ 是初始学习率，$\alpha_t$ 是第 t 个 epoch 的学习率，$T$ 是总 epoch 数。

![学习率衰减策略幻灯片，包含线性衰减曲线及Step、Cosine、Linear三种方法的公式定义](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0018_01-01-47.jpg)

另一个极常用的是 **linear warm-up**：不直接从最大学习率起步，先用固定步数线性升到最大值，再接 cosine 或 inverse square root 等调度器——warm-up + cosine 是非常流行的组合。还有一条经验法则「线性缩放律（linear scaling law）」：若把 batch size（每次更新的样本数）扩大 n 倍，学习率也应同比扩大 n 倍；它更多是大量问题上验证出来的经验，数学解释涉及 batch 内梯度的方差等。

![一阶优化示意图，展示利用梯度线性近似损失函数并沿切线方向进行最小化步骤的过程](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0019_01-04-04.jpg)

最后简短提到二阶优化：目前所有方法都是用梯度做损失的**线性（一阶）近似**再沿切线走一步；二阶方法改用 Hessian 做二次／泰勒展开近似来直接找最小，某些优化问题里效果极好。但深度学习基本不用，因为要算所有参数两两之间的二阶混合偏导，参数量达百万、十亿时 Hessian 矩阵大到 GPU 显存装不下，且计算代价高；小模型或愿意多花时间换更好步长时才值得考虑。若每步能把整个训练集装进 batch（full batch），倒是可以考虑二阶及更高阶方法。

结论与预告：新问题先用 **Adam 或 AdamW** 作默认（常数学习率也常行，或 linear warm-up + cosine decay），它对损失地形的自适应性强；SGD+Momentum 有时能超过 Adam，但要试更多学习率与调度值。下一讲进入神经网络：两层网络有两个权重矩阵，中间夹一个非线性（最简单是 ReLU）——因为像红蓝点这样的分布线性分类器分不开，需要靠非线性／多层把数据变换成最终可被一条直线分开的形式。

![解释为何需要非线性：红蓝点分布图展示线性分类器无法分离两类数据](/coursenotes/cs231n-deep-learning-for-cv-sp25/l03-regularization-optimization/kf_0020_01-08-17.jpg)

## 要点回顾

- 总损失 = 数据损失 + 正则化项：$L(W)=\frac{1}{N}\sum_i L_i(f(x_i,W),y_i)+\lambda R(W)$；正则化是「训练集变差、测试集变好」，$\lambda$（0 到无穷）是可用验证集调的超参数，背后有 $F_1$/$F_2$ 过拟合对比与奥卡姆剃刀。
- L2 正则化 $R(W)=\sum_k\sum_l W_{k,l}^2$ 偏好摊开的小权重；L1 取绝对值之和，偏好稀疏、更多真 0 的权重；$x=[1,1,1,1]$、$w_1=[1,0,0,0]$、$w_2=[0.25,\dots]$ 的例子中 L2 偏好 $w_2$（1/4 对 1），L1 下二者等价。
- 正则化能表达对权重的偏好、让模型更简单、L2 还因凸性改善优化；但 dropout 等会让模型更复杂，说明「更简单」不是必然，本质是训练集变差换测试集更好。
- 优化是「蒙眼在损失地形里摸坡度」：随机搜索 1000 个 W 在 CIFAR-10 仅 15.5%（SOTA 99.7%），根本策略是跟随负梯度（最速下降）方向。
- 梯度：一维导数极限定义 $\frac{df}{dx}=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}$，多维为梯度向量 $\nabla_W L$；数值梯度近似慢但易写、解析梯度精确快但易藏 bug，用 gradient check 比对，凸函数上梯度下降很有效。
- 梯度下降 `weights += - step_size * weights_grad`，停止靠固定迭代数或损失容差（1e-5／1e-9）；SGD 用 minibatch（示例 256）近似全量梯度，随机顺序过完一遍为一个 epoch。
- SGD 四大病态：高条件数（Hessian 最大／最小奇异值之比）导致窄谷震荡、局部极小、高维更常见的鞍点、minibatch 带来的梯度噪声；Momentum 用 $v_{t+1}=\rho v_t-\alpha\nabla f(x_t)$ 积累速度一并缓解（Sutskever et al. 2013）。
- RMSprop（Hinton 组 2012）按平方梯度滑动平均缩放步长；Adam（Kingma and Ba, ICLR 2015）= Momentum + RMSProp，$\beta_1=0.9$、$\beta_2=0.999$，因首步二阶矩近 0 会产生巨大初始步，需 bias correction；AdamW 把 weight decay 从梯度里分离、在矩计算之后相加，LLaMA 系列采用。
- 学习率调度：Step（ResNets 在 30/60/90 epoch 乘 0.1）、Cosine $\alpha_t=\frac12\alpha_0(1+\cos(t\pi/T))$、Linear $\alpha_t=\alpha_0(1-t/T)$、linear warm-up、线性缩放律（batch ×n 则 LR ×n）；二阶 Hessian／泰勒近似因矩阵过大、显存不足在大模型里基本不用。
- 实践默认 Adam／AdamW（常数 LR 或 warm-up+cosine），SGD+Momentum 有时更优但调参更多；下一讲用两层网络 + ReLU 非线性处理线性不可分的红蓝点数据。

## 视频来源

https://www.youtube.com/watch?v=dyNGd06MWn4
