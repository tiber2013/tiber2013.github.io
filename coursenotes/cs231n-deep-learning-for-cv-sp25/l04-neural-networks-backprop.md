---
title: "第 4 讲 · 神经网络与反向传播：从激活函数到计算图"
description: "CS231n Spring 2025 L04 视频笔记；对照 2026 春季课件补充新增内容。"
---

# 第 4 讲 · 神经网络与反向传播：从激活函数到计算图

> **Stanford CS231n 计算机视觉 · L04** ｜ 讲者：Stanford CS231n 课程团队 ｜ 76 分钟 ｜ [原视频](https://www.youtube.com/watch?v=25zD5qJHYsk)

> 课件版本：2026 春季 ｜ 视频版本：2025 春季。2026 课件新增或调整的内容集中在文末「2026 课件更新」一节。

## 核心结论

这一讲做两件事：把线性分类器扩展成多层神经网络，并解决训练它时“梯度从哪来”的问题。线性模型 $f=Wx$ 只能画一个超平面；在两层网络 $f=W_2\max(0,W_1x)$ 中间插入 **activation function** 之后，模型能表示非线性边界，还能学到跨类别共享的部件模板。代价是梯度无法再手写推导——这就是 **computational graph** 与 **backpropagation** 出场的地方：把网络拆成节点，每个节点只用 **local gradient** 乘上从后面传回来的 **upstream gradient**，得到 **downstream gradient** 再继续往前传。这套机制是后面 CNN、RNN、Transformer 全部训练过程共同的底座。

## 回顾：损失、优化器与学习率 [00:05 - 09:53]

讲者先复习上一讲的链条：数据 $(x,y)$ → 打分函数 $s=f(x;W)=Wx$ → 损失函数 → 正则项 → 总损失。上一讲及更早的课堂上有同学反复问“为什么只用 softmax”，讲者这次明确回答：softmax 只是在分类里最常用，不是唯一选择；**hinge loss**（旧称 SVM loss）不把分数变成概率，只要求正确类分数比其它类高出至少一个 margin，被违反的部分线性增长。早期的神经网络大量使用它。

![回顾损失函数：Softmax损失、完整损失及SVM/Hinge损失公式与计算图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0001_00-02-57.jpg)

优化部分的关键仍然是“沿负梯度走”：损失地形是一整片山谷，每个点对应一组权重，要找到最低点就得对 $L$ 关于 $W$ 求梯度，再按步长一步步下降。求梯度有数值与解析两条路，实践中一律用解析梯度训练，用数值梯度做校验。

![用梯度下降寻找最优权重W：包含山谷示意图、损失地形等高线图及梯度下降伪代码](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0002_00-05-16.jpg)

数据量大时全量求和太贵，于是用 **mini-batch**（常见 32/64/128/256）近似全量梯度。上一讲还介绍了 SGD 的三个改进：**Momentum**、**RMSProp**、**Adam**，以及学习率调度——step decay、cosine、linear、inverse sqrt。讲者补充了一个实践观察：越新的自适应优化器（如 Adam 及其变体）把一部分步长调整编码进了优化器内部，手动衰减学习率的必要性因此下降。

![不同优化器（SGD、SGD+Momentum、RMSProp、Adam）在损失函数等高线上的收敛轨迹对比图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0004_00-08-37.jpg)

![学习率调度策略，包含训练损失随 Epoch 变化的曲线图及 Step、Cosine、Linear、Inverse sqrt 四种衰减公式](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0005_00-09-07.jpg)

## 从线性模型到神经网络 [09:53 - 18:46]

线性模型是“只有一层”的神经网络：

$$
f = Wx,\qquad x\in\mathbb{R}^{D},\ W\in\mathbb{R}^{C\times D}
$$

$D$ 是输入维度（特征数），$C$ 是类别数，也就是输出节点数。加第二层时，引入 $W_1$ 把输入映射到一个 $H$ 维的隐藏空间，再用 $W_2$ 映射到 $C$ 个输出：

$$
f = W_2\max(0, W_1x),\qquad W_1\in\mathbb{R}^{H\times D},\ W_2\in\mathbb{R}^{C\times H}
$$

$H$ 是隐藏层神经元数量。讲者提醒两点：维度必须逐层对齐；实际实现中每层还会加可学习的 bias，公式里为了简洁省略了。这种只由矩阵乘法和逐元素非线性堆起来的结构，就是 **fully connected network** 或 **multi-layer perceptron（MLP）**；继续堆叠就得到三层、四层网络，中间层的维度可以互不相同。

![神经网络前身线性分类器的线性评分函数公式及维度定义](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0006_00-10-40.jpg)

### 为什么必须非线性

线性分类器的失败情形很直观：红点聚在中心、蓝点围成一圈，任何一条直线都分不开。解决办法是把原始空间非线性地变换到另一个空间——视频给的例子是极坐标变换 $(x,y)\mapsto(r,\theta)$，变换后一条直线就能切开。神经网络把这个变换交给网络自己学。

$$
f = W_2\max(0, W_1x)
$$

如果去掉 $\max$，网络变成 $W_2W_1x$，而两个矩阵的乘积还是矩阵，记 $W_3=W_2W_1$，模型立刻退回线性分类器。所以中间这个 $\max(0,\cdot)$ 不是可有可无的装饰，它决定了网络有没有表达非线性函数的能力。这个函数叫 **ReLU**（rectified linear unit），在神经网络术语里统称 **activation function**。

![展示线性分类器无法区分红蓝点的二维散点图，说明引入非线性的必要性](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0007_00-12-58.jpg)

从“模板”的角度看得更清楚：线性分类器学 10 个类别模板，每个模板必须独自概括整个类别；两层网络在中间有 100 个神经元，相当于学 100 个模板，其中很多可以只描述物体的部件。鸟、猫、鹿、狗、马都有眼睛，某个隐藏神经元可以专门响应“眼睛”这类跨类别共享的结构。这也是“学习 100 个模板而不是 10 个，并在类别之间共享模板”的含义。

![对比线性评分函数与两层神经网络结构，展示从3072维输入到10类输出的模板学习](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0008_00-16-53.jpg)

## 激活函数谱系与选择 [18:46 - 26:09]

ReLU 的问题是会产生 **dead neuron**：输入长期为负，输出恒为 0，梯度也恒为 0，这个神经元就再也不更新了。为此有一系列变体：

- **Leaky ReLU**：负半轴给一个很小的斜率，如 $\max(0.1x, x)$；
- **ELU**：负半轴用指数饱和，输出更接近零中心，通常比 ReLU 收敛更稳；
- **GELU**：$x\cdot\Phi(x)$，在 Transformer 中很常见；
- **SiLU / Swish**：$x\cdot\sigma(x)$，出现在 EfficientNet 一类 CNN 中；
- **Sigmoid / Tanh**：把值压到很窄的区间，容易造成 **vanishing gradient**，所以一般不用在中间层，只在输出层需要二值化或概率时使用。

![Stanford CS231n课程中关于激活函数的幻灯片，展示了ReLU、Leaky ReLU、ELU、GELU、Sigmoid、Tanh和SILU的函数图像与公式，并标注ReLU为多数问题的默认选择。](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0010_00-21-21.jpg)

有同学问新问题该选哪个激活函数，讲者的回答很直接：**经验性**。默认从 ReLU 起步，或者沿用该类架构已被验证的选择。它们的共同点只有一个——提供非线性；差异则来自是否可微、是否零中心、是否平滑，这些性质会影响收敛速度。同一个网络里通常所有层用同一种激活，只在输出层换成 sigmoid 或 tanh。

## 前向传播、容量与正则化 [26:09 - 35:00]

一个两层网络的全部训练代码不到 20 行。定义维度 `N, D_in, H, D_out`，随机初始化 `w1, w2`；循环里先算 $h=\sigma(xw_1)$、$y_{pred}=hw_2$，再算损失，然后手动写出解析梯度 `grad_w2 = h.T.dot(grad_y_pred)`、`grad_w1 = x.T.dot(grad_h * h * (1 - h))`，最后做一次 SGD 更新。前向、求解析梯度、更新，三步就是全部。

![三层神经网络前向传播计算示例，含网络结构图与NumPy实现代码](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0011_00-26-18.jpg)

隐藏层神经元数量直接决定模型容量：3 个神经元只能画出很简单的边界，6 个开始弯曲，20 个可以把两类点切得很碎。讲者把它类比成第二讲的 1-NN：容量给太多，模型记住了训练点却无法泛化。

![不同隐藏神经元数量（3、6、20个）对分类决策边界复杂度的影响对比图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0012_00-29-25.jpg)

由此得到一条经验法则：**不要用网络规模当正则化手段**。正确顺序是先选一个略大于需要的网络，训练到出现明显的过拟合，再用正则化去压。$\lambda$ 控制正则项在总损失中的比重，而正则项只作用在 $W$ 上，所以 $\lambda$ 越大，权重的自由度越小，决策边界越平滑；但压过头就会欠拟合，连正确的分类都做不到。所谓正则化，本质上是在“拟合数据”和“约束权重”之间找平衡。

![不同正则化强度（λ=0.001, 0.01, 0.1）对决策边界平滑度的影响及正则化损失函数公式](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0013_00-30-28.jpg)

这也解释了为什么调 $\lambda$ 而不是调网络大小：大网络动辄要训好几天，先把参数量加到开始过拟合，再靠正则化收回来，比反复缩小网络更划算。至于“到底需要多少神经元”，没有通用公式，只能参考同类数据上的已有工作，再自己做实验；理论上确实有分析层数与激活函数选择的工作，但它们的假设未必适用于你的问题。

## 生物神经元，以及类比的边界 [35:00 - 39:01]

神经元的树突把电信号汇集到细胞体，细胞体做某种整合，再由轴突把输出传给下一个神经元。人工神经元的结构与之对应：把上一层所有激活值加权求和，过一个激活函数，再传给下一层。讲者强调这只是很松的类比——生物神经元种类繁多、树突本身能做复杂的非线性运算、突触也不是一个标量权重，真实系统远比我们搭的网络复杂。

另一个区别在连接方式：生物神经网络的连接是杂乱无章的，而人工网络被组织成规整的层，这是为了计算效率（矩阵乘法）。也有研究尝试随机连接的网络结构，性能与规整网络大体相当，但工程上仍然以分层为主。讲者反复提醒：对“大脑类比”要非常谨慎，不要把它当成事实。

## 为什么不能手推梯度 [39:01 - 41:29]

把网络接上损失函数后，问题变得具体：打分函数是非线性的 $s=W_2\max(0,W_1x)$，损失可以是 hinge 或 softmax，再加上对 $W_1$、$W_2$ 的正则项：

$$
L=\frac{1}{N}\sum_{i=1}^{N}L_i+\lambda R(W_1)+\lambda R(W_2)
$$

只要能算出 $\partial L/\partial W_1$ 和 $\partial L/\partial W_2$，就能学。但手推这条路走不通：矩阵微积分极其繁琐，要在纸上写很久；一旦把 hinge 换成 softmax，整套推导得从头再来；模型再复杂一点就完全不可行。这就是需要计算图和反向传播的原因。

## 计算图与标量反向传播 [41:29 - 52:47]

**Computational graph** 把网络里的每个运算表示成一个节点：输入和权重进入乘法节点得到分数，分数进入损失节点，正则项进入加法节点，最后得到标量 $L$。卷积网络、Neural Turing Machine 这类结构都可以画成同一套图，只是节点更多、展开更长。图一旦建好，求梯度就变成一个统一的递归过程。

最简单的例子是 $f(x,y,z)=(x+y)z$，取 $x=-2,\ y=5,\ z=-4$。前向传播：$q=x+y=3$，$f=qz=-12$。局部梯度也容易写：$\partial q/\partial x=\partial q/\partial y=1$，$\partial f/\partial q=z=-4$，$\partial f/\partial z=q=3$。

反向传播从图的末端开始，**初始 upstream gradient 恒为 1**，因为损失对自己的导数是 1。此后每个节点只做一件事：

$$
\text{downstream}=\text{upstream}\times\text{local}
$$

对 $f=qz$：$\partial f/\partial z$ 直接等于 local gradient $q=3$；$\partial f/\partial q$ 等于 $z=-4$，这个值成为下一个节点的 upstream。$q=x+y$ 是加法节点，local gradient 对两个输入都是 1，于是 $\partial f/\partial x=\partial f/\partial y=-4$。$y$ 与 $f$ 不相邻，链式法则把路径拆开：

$$
\frac{\partial f}{\partial y}=\frac{\partial f}{\partial q}\cdot\frac{\partial q}{\partial y}
$$

这套命名的意义在于模块化：**upstream gradient** 是从后面节点传来的“这个中间量对最终损失的影响”，**local gradient** 是节点输出对输入的偏导，两者相乘得到的 **downstream gradient** 就是前一个节点的 upstream。每个节点不需要知道整个网络，只要知道自己的函数和上游传来的数。100 层网络因此不需要写 100 层的表达式，也能逐层拿到每个权重的梯度。

## sigmoid 例子与门模式 [52:47 - 63:04]

第二个例子把线性组合塞进 sigmoid：

$$
f(w,x)=\frac{1}{1+e^{-(w_0x_0+w_1x_1+w_2)}}
$$

取 $w_0=2,\ x_0=-1,\ w_1=-3,\ x_1=-2,\ w_2=-3$，前向得到 $-2+6-3=1$，取负后 exp 得 $0.37$，加 1 得 $1.37$，取倒数得 $f=0.73$。反向从 $1/x$ 节点开始：local gradient 是 $-1/x^2\approx-0.53$，乘以 upstream 1 得 $-0.53$；加 1 是加法门，梯度原样传回；exp 节点的 local gradient 是 $e^{-1}=0.37$，乘出 $-0.20$；乘 $-1$ 后变成 $0.20$；加法门把它分给两个输入；最后两个乘法门各按“交换”规则分配：$w_0$ 拿到 $0.2\times x_0=-0.2$，$x_0$ 拿到 $0.2\times w_0=0.4$，$w_1$、$x_1$ 分别是 $0.2x_1=-0.4$ 与 $0.2w_1=-0.6$，而 $w_2$ 只经过加法，梯度是 $0.2$。

有意思的是，同一张图可以画得更粗：把整条 sigmoid 当成一个节点，它的 local gradient 是

$$
\frac{d\sigma(x)}{dx}=(1-\sigma(x))\,\sigma(x)
$$

代入 $\sigma=0.73$ 得 $(1-0.73)\times0.73\approx0.2$，与逐个节点算出来的结果完全一致。也就是说，**计算图的画法不唯一**，应该选让每个节点的 local gradient 容易表达的那种画法。

这套流程还能总结成几种“门”的固定模式：**add gate** 是梯度分发器，原样复制给每个输入；**mul gate** 是交换乘数，对 $x$ 的梯度是另一个输入 $y$；**copy gate** 是梯度加法器，一个变量被多个分支使用时，反向要把各分支的上游梯度相加；**max gate** 是路由器，把梯度全部路由给取到最大值的那一路——ReLU 就是 max gate 的特例。记住这四种模式，很多推导可以口算。

代码层面，最直接的写法是把前向写成几行赋值，再按相反顺序手写一串 `grad_*`。更好的是把每个函数做成对象，提供 **forward / backward** 两个 API：forward 计算输出并缓存反向需要的中途值，backward 用缓存值和 upstream 算梯度。PyTorch 的算子就是这样组织的——`Multiply` 在 forward 里 `ctx.save_for_backward(x, y)`，在 backward 里返回 `y * grad_z` 和 `x * grad_z`；sigmoid 的反向实现就是 `gradOutput * (1 - z) * z`。

## 向量、矩阵与 Jacobian [63:04 - 76:19]

以上都是标量。推广到向量时，先看清导数的形状：标量对标量是普通导数；向量对标量是梯度，形状与输入相同；向量对向量是 **Jacobian**，形状是输入维度乘输出维度，每个元素表示“输入的某个分量变化一点，输出的某个分量变多少”。

反向传播时，损失 $L$ 永远是标量，所以 $\partial L/\partial z$ 是一个与 $z$ 同形状的向量（upstream），local gradient 是一整个 Jacobian，两者相乘（矩阵乘向量）得到与输入同形状的 downstream。一条通用规律：**变量对损失的梯度，永远和这个变量本身同形状**。

以 ReLU 为例，$f(x)=\max(0,x)$ 是逐元素运算，每个输出只依赖对应的那个输入，所以 Jacobian 是一个稀疏矩阵，只有主对角线非零，值取决于 $x_i$ 是否为正。实践中当然不会真的构造这个矩阵，而是直接用规则：$x_i>0$ 时梯度原样通过，否则置零。这就是 **implicit multiplication**——Jacobian 从不显式形成。

矩阵乘法更极端。设 $x\in\mathbb{R}^{N\times D}$、$w\in\mathbb{R}^{D\times M}$、$y=xw$，$y$ 的每个元素 $y_{n,m}=\sum_d x_{n,d}w_{d,m}$。若 $N=64$、$D=M=4096$，单个 Jacobian 就要约 256 GB 显存，根本存不下。但观察依赖关系就能化简：$x_{n,d}$ 只影响 $y$ 的第 $n$ 行；而 $x_{n,d}$ 对 $y_{n,m}$ 的偏导就是 $w_{d,m}$。于是

$$
\frac{\partial L}{\partial x}=\left(\frac{\partial L}{\partial y}\right)w^{T},
\qquad
\frac{\partial L}{\partial w}=x^{T}\left(\frac{\partial L}{\partial y}\right)
$$

这两个公式不需要死记：它们是唯一能让矩阵形状对上的写法（$[N\times M][M\times D]\to[N\times D]$，$[D\times N][N\times M]\to[D\times M]$）。写反向传播代码时，形状往往比推导更能提示答案。

## 2026 课件更新

以上正文以 2025 春季视频为准。对照 2026 春季课件，主要差异是：

- **行政信息前置**：2026 课件以行政信息开场——Assignment 1 于周四 4/16 23:59 截止；项目提案 4/16 发布、4/23 23:59 截止；TA 专长列表放在课程网页；次日 discussion section 由 Favour Nerrise 讲 Backpropagation；并附 Student Spotlights 名单。2025 视频没有这些内容，直接进入回顾。
- **向量求导单独成节**：2026 课件把 “Recap: Vector derivatives” 做成独立小节，用标量对标量、向量对标量、向量对向量三张卡片明确写出导数、梯度、Jacobian 的形状，再进入 Backprop with Vectors；视频里这部分是口头带过的。
- **Jacobian 的处理写成明确结论**：2026 课件在 ReLU 例子后直接标注 “Never explicitly form Jacobian — instead use implicit multiplication”，并在矩阵乘法页写出 $N=64$、$D=M=4096$、单次矩阵乘的 Jacobian 约 256 GB，与视频结论一致但表述更形式化。
- **矩阵反向传播的推导更细**：2026 课件把 “$x_{n,d}$ 影响 $y$ 的整行”“$\partial y_{n,m}/\partial x_{n,d}=w_{d,m}$” 分屏推完，最后给出 $\partial L/\partial x$ 与 $\partial L/\partial w$ 的形状对照，并指向课程讲义 `linear-backprop.pdf`；视频只给了结论。
- **补充引用与警示**：2026 课件为“随机连接也能工作”补上 Xie et al., *Exploring Randomly Wired Neural Networks*（ICCV 2019）；并把 “Be very careful with your brain analogies” 做成独立页，列出生物神经元的多种类型、树突的非线性计算、突触不是单个权重三点（引 Dendritic Computation）。
- **代码示例更新**：2026 课件给的是现代 PyTorch 写法（`torch.autograd.Function` 配 `ctx.save_for_backward`），sigmoid 反向仍写作 `gradOutput * (1 - z) * z`；末页总结把全讲压缩为五条，并指向下一讲的 CNN。

## 要点回顾

- 线性模型 $f=Wx$ 只是一层网络；两层网络 $f=W_2\max(0,W_1x)$ 通过 $H$ 维隐藏层获得非线性表达能力。
- 没有激活函数时 $W_2W_1$ 可合并为单个矩阵，网络退化为线性分类器；非线性的意义是把不可分数据映射到可分空间。
- 隐藏层的 100 个模板可以只描述部件，并在类别之间共享，这是多层网络相对线性模板的实质增益。
- **ReLU** 是默认选择，但有 dead neuron 问题；Leaky ReLU、ELU（更零中心）、GELU（Transformer）、SiLU（EfficientNet）是常见替代；sigmoid/tanh 因梯度消失一般只用于输出层。
- 激活函数的选择主要是经验性的，共同要求是可微且提供非线性。
- 不要用网络规模当正则化：先把容量加到过拟合，再用 $\lambda$ 压回来；$\lambda$ 过大则欠拟合。
- 手推梯度不可行——繁琐、换损失就要重推、复杂模型根本写不出来。
- **backpropagation** 的本质是沿计算图递归应用链式法则：$\text{downstream}=\text{upstream}\times\text{local}$，初始 upstream 为 1。
- 计算图的画法不唯一，应选 local gradient 好写的画法；sigmoid 的 local gradient 是 $(1-\sigma)\sigma$。
- 记住四种门：add 分发、mul 交换、copy 累加、max 路由。
- 实现上每个节点提供 forward/backward API：forward 缓存中间值，backward 用缓存和 upstream 算梯度，这正是 PyTorch autograd 的组织方式。
- 向量/矩阵情形下梯度与变量同形状，local gradient 是 Jacobian；Jacobian 巨大且稀疏，只做隐式乘法。矩阵乘的梯度按形状匹配即可写出：$\partial L/\partial x=(\partial L/\partial y)w^T$，$\partial L/\partial w=x^T(\partial L/\partial y)$。

## 视频来源

https://www.youtube.com/watch?v=25zD5qJHYsk
