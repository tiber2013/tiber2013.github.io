---
title: "第 4 讲 · 神经网络与反向传播：从激活函数到计算图"
description: "Stanford CS231n Spring 2025 L04 中文视频笔记（2026-11 重写版）"
---

# CS231n Spring 2025 · Lecture 4 神经网络与反向传播：从计算图到梯度流

**核心结论**：本讲把前几讲的线性分类器扩展为多层全连接网络，并给出训练它们的核心算法——反向传播。第一个关键认识是，若没有非线性激活函数，堆叠再多的权重矩阵也会塌缩成一个线性变换；第二个关键认识是，为整个网络手工推导梯度既繁琐、又无法应对改动过的或复杂的损失函数，因此改用「计算图 + 局部梯度」的框架，把全局求导化归为每个节点上「上游梯度 × 局部梯度 = 下游梯度」的局部链式法则。讲者以 $f=(x+y)z$ 和 sigmoid 为例逐节点演算，归纳出 add、mul、copy、max 四种门控的梯度流模式，并进一步说明在向量、矩阵乃至张量情形下梯度形状永远与对应变量相同，而显式雅可比矩阵（如 $N=64,\ D=M=4096$ 时约 256 GB）必须被隐式规则替代。讲者反复强调，此后课程里几乎每一个算法都在以某种形式使用反向传播，因此这一讲是整门课的地基，下一讲进入卷积神经网络。

## 1. 回顾：损失函数、正则化与梯度下降优化 [00:05 - 09:53]

讲者开篇把反向传播称为「让神经网络从自身错误中学习」的过程——像人类一样，但更有组织、并且用了更多数学；它是本季度后续一切算法的基础。按惯例先回顾前几讲：我们把学习问题写成 $(x_i,y_i)$ 对与一个评分函数（此前用线性评分 $s=f(x;W)=Wx$），再定义损失函数并加上正则化，右侧的计算图把 $W$、$x_i$、$y_i$ 如何流向 score function、loss，以及 regularization loss 在何处汇入全过程画了出来。

![回顾幻灯片，展示评分函数、Softmax与SVM损失公式及计算图结构](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0001_00-02-57.jpg)

针对「为什么只用 Softmax」的疑问，讲者重申它只是分类任务里最常用的损失之一，并非唯一；课件在 Lecture 2 的阅读材料中还给了 **Hinge Loss（SVM loss）**：
$$L_i=\sum_{j\neq y_i}\max(0,\ s_j-s_{y_i}+1)$$
与 Softmax $L_i=-\log\frac{e^{s_{y_i}}}{\sum_j e^{s_j}}$ 不同，Hinge Loss **不把分数转成概率**，而是鼓励正确类别分数 $s_{y_i}$ 比所有其他类别 $s_j$ 至少高出 margin（式中的 1）；条件满足时损失为 0，违反时损失按超出 margin 的量成比例增长，以此惩罚「无关项得分过高」。总损失仍是数据损失加正则项：$L=\frac{1}{N}\sum_i L_i+R(W)$。

优化部分回顾：损失曲面像一个大山谷，谷中每一点对应一组权重 $W$，目标是找到使 $L$ 最小的 $W$。核心是求梯度 $\partial L/\partial W$，并沿负梯度方向按步长（step size / learning rate）逐步更新，得到 Vanilla Gradient Descent：

```python
# Vanilla Gradient Descent
while True:
    weights_grad = evaluate_gradient(loss_fun, data, weights)
    weights += -step_size * weights_grad  # perform parameter update
```

![梯度下降优化幻灯片，包含下山比喻图、损失曲面热力图及伪代码](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0002_00-05-19.jpg)

实践要点：数值梯度（numerical gradient）与解析梯度（analytical gradient）各有优劣，实际使用解析梯度，并用数值梯度做梯度检验（gradient check）；在大数据集上对全部数据算损失与导数太昂贵，故采用 **mini-batch**，常见批量为 32、64、128、256；优化器还讲了 SGD、SGD+Momentum、RMSProp、Adam 的收敛轨迹差异，以及学习率调度——很多优化器需要先大后小地衰减学习率，而 Adam 及其变体把这种衰减编码在优化器内部，通常不必手动衰减。

![不同优化器在损失曲面上的收敛轨迹对比图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0003_00-08-57.jpg)

## 2. 从线性评分到两层全连接网络 [09:57 - 17:21]

此前 $f=Wx$ 是最简单的一层网络，其中 $D$ 是输入特征数，$C$ 是类别数（输出节点数）。加入第二层权重 $W_2$ 后：
$$f=W_2\max(0,W_1x),\qquad x\in\mathbb{R}^D,\ W_1\in\mathbb{R}^{H\times D},\ W_2\in\mathbb{R}^{C\times H}$$
新增的 $H$ 是隐藏层神经元数。中间的 $\max$ 运算在 $W_1$、$W_2$ 两次线性变换之间制造了**非线性**，这是全讲最关键的一点；实践中每层还会加可学习的偏置（bias），课件为简洁省略。

![两层神经网络与线性得分函数的公式对比及维度定义](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0004_00-11-31.jpg)

为什么必须非线性？线性分类器无法用一条直线分开许多分布，需要把样本从原空间做非线性变换映射到新空间后才线性可分——讲者举的例子是把 $(x,y)$ 映射为极坐标 $(r,\theta)$。若去掉 $\max$，函数变成 $W_2W_1x$，两个矩阵可被单个 $W_3$ 取代，整个网络退化成线性函数，层数再多也无济于事。

这类只含矩阵乘法与逐层运算、层间全连接的网络称为**全连接网络（fully connected networks）或多层感知机 MLP（multilayer perceptrons）**，可以继续堆叠更多层，只要相邻层维度匹配。课件给出的具体维度例子是 3072 → 100 → 10。

![两层神经网络的层级计算结构图及具体维度示例](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0005_00-15-21.jpg)

从「模板（templates）」视角理解隐藏层的意义：线性分类器的 10 个输出对应 10 张从数据里学出的整类模板；有了含 100 个神经元的中间层，网络能学出 100 个模板，且这些模板不必是完整物体，而可能是跨类别共享的**部件**——例如讲者列举的 bird、cat、deer、dog、frog、horse 都有眼睛，某个隐藏神经元就可以专门响应「眼睛」这类局部模式。可视化相关细节留到课程后面的章节展开。

## 3. 激活函数：非线性的来源与选型 [17:27 - 26:09]

前面的 $\max(0,\cdot)$ 即 **ReLU（rectified linear unit）**，是神经网络里最流行的激活函数，也是讲者推荐的大多数问题的默认选择。它的问题是把所有非正值置零，可能产生 **dead neurons（死亡神经元）**；为此有 Leaky ReLU $\max(0.1x,x)$、ELU（exponential linear unit，具有更好的零中心特性）等变体。更现代的选择包括 GELU（Gaussian Error Linear Unit，讲者发音近似 jello，常用于 Transformer 类架构）与 SiLU / Swish（sigmoid linear unit，用于部分现代 CNN，Google 的 EfficientNet 等模型采用过）。

![六种常见激活函数（ReLU, Leaky ReLU, Sigmoid等）的图像与公式对比](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0006_00-18-48.jpg)

![常见激活函数对比图，包含ReLU、Leaky ReLU、Sigmoid、Tanh、ELU、GELU、SiLU的公式与曲线](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0007_00-21-24.jpg)

Sigmoid $\sigma(x)=\frac{1}{1+e^{-x}}$ 与 Tanh $\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$ 也常被列为激活函数，但它们把数值压缩（squash）到很窄的区间，容易导致**梯度消失（vanishing gradients）**，因此一般不放在网络中间层，而多用于末层需要把输出二值化之类的场合。一个典型全连接网络用 sigmoid 作激活时，隐藏值逐层计算为 $h_1=\sigma(W_1x+b_1)$、$h_2=\sigma(W_2h_1+b_2)$，输出层再由 $W_3$ 与最后一层隐藏值做点积得到。

课堂问答里补充了几条选型原则：为具体问题挑哪个激活函数**大多是经验性的（empirical）**，通常从 ReLU 或该架构公认的标准做法起步；所有激活函数的共同特征是**制造非线性**；它们还应当**可微**（因为要参与反向传播），而零中心、平滑的性质有助于更快收敛。同一网络的各层通常使用同一个激活函数，只在输出层视需要换成 sigmoid 或 tanh。

## 4. NumPy 实现、网络容量与正则化 [26:09 - 35:50]

用 Python 实现两层网络不到 20 行：定义 $N$（样本数）、$D_{in}$、$H$、$D_{out}$（课件示例为 64、1000、100、10），随机初始化 $W$；**forward pass** 逐层把 $W$ 作用到输入上得到预测 `y_pred` 并算出 loss；随后需要解析梯度来跑梯度下降、更新 $W_1$、$W_2$。讲者点明：求解析梯度正是这一讲余下篇幅要「让它跑起来、并扩展到不同规模」的核心。

```python
import numpy as np
N, D_in, H, D_out = 64, 1000, 100, 10
for t in range(2000):
    h = 1 / (1 + np.exp(-x.dot(w1)))
    y_pred = h.dot(w2)
    loss = np.square(y_pred - y).sum()
    # ... backward pass 求 grad_h, grad_w2, grad_w1 ...
    grad_w1 = x.T.dot(grad_h * h * (1 - h))
    w1 -= 1e-4 * grad_w1
```

![使用 NumPy 实现两层神经网络训练过程（前向传播、损失计算、反向传播）的完整代码示例](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0008_00-26-21.jpg)

隐藏层神经元数决定容量：3、6、20 个隐藏神经元的对比显示，神经元越多，决策边界越复杂、拟合能力越强（more neurons = more capacity），这和第二讲 k 最近邻里 $k=1$ 的情形类似——容量过大就会过拟合、无法泛化到未见过的数据。

![不同隐藏层神经元数量（3、6、20）对分类决策边界复杂度的影响对比图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0009_00-29-26.jpg)

由此引出一条重要的经验法则：**不要把网络规模当作正则化器**，而应使用一个略大于所需的网络，再用正则化强度 $\lambda$ 去调控。课件对比 $\lambda=0.001,\ 0.01,\ 0.1$ 的决策边界，并给出 $L(W)=\frac{1}{N}\sum_i L_i(f(x_i,W),y_i)+\lambda R(W)$。

![不同正则化系数（λ=0.001, 0.01, 0.1）下的分类边界对比及损失函数公式](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0010_00-30-28.jpg)

$\lambda$ 控制正则项在总损失里的权重：正则项定义在所有 $W$ 上，$\lambda$ 越大对权值的约束越强、自由度越小、边界越平滑通用；但过强的约束会使模型欠拟合——因为损失的第一项负责预测正确输出，第二项只关心权重取值，两者必须平衡，正则化再好也不能过量。问答中还解释了为什么调正则化而非网络尺寸：大型网络可能要跑几天，通常的做法是逐步增大参数量直到出现轻微过拟合（说明网络已能记住数据中的模式），再用正则化把过拟合压下去；至于需要多少神经元，没有统一处方，主要靠参考同类数据上的既有网络并做实验探索。确实有分析激活函数、层数与超参数优化的研究工作，但它们各自带假设，未必适用于你的具体问题与数据集。

## 5. 生物神经元的启发：一个需要谨慎对待的类比 [35:50 - 39:01]

讲者短暂讨论了「神经网络」名字的生物来源，并立刻提醒这类**大脑类比非常松散**，请神经科学家不要把它当作 ground truth。生物神经元有细胞体（cell body）、把冲动传入细胞体的树突（dendrite）与突触前末梢（presynaptic terminal）、以及把冲动传出的轴突（axon）：细胞体聚合来自树突的信号，再经轴突传给其他神经元。

![神经元结构示意图，标注了树突、轴突及冲动传导方向](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0011_00-35-55.jpg)

这与人工网络的对应是：某个函数聚合上一层的所有激活（ impulses / activations ），在「细胞体」里对该输入做运算、输出激活并传给下一层，而激活函数正起到调节、增强或抑制这些值的作用。但讲者反复警告二者差异巨大——真实生物神经元远比我们构建的网络复杂；人工网络被组织成规则、规整的模式，主要是出于**计算效率**的考虑；虽然也有研究构造复杂拓扑的网络，其效果与常见规整网络基本相当。

## 6. 计算图与反向传播的动机 [39:01 - 43:48]

把前面的部件拼起来：评分函数 $s=f(x;W_1,W_2)=W_2\max(0,W_1x)$ 给出分数，经 Hinge Loss 或 Softmax 得到数据损失，再加正则化项构成总损失：
$$L=\frac{1}{N}\sum_i L_i+\lambda R(W_1)+\lambda R(W_2),\qquad R(W)=\sum_k W_k^2$$

![神经网络损失函数构建：非线性得分、Hinge Loss、正则化及总损失公式](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0012_00-39-40.jpg)

要优化 $W_1,W_2$，就需要 $\partial L/\partial W_1$、$\partial L/\partial W_2$。讲者列出手工求导的三重困难：写大量矩阵运算并在纸上推导非常繁琐；稍微改动损失函数就要把整套推导重做一遍；面对复杂损失函数时甚至变得不可行（intractable）。更好的思路是 **计算图（computational graphs）加反向传播（backpropagation）**。

计算图把网络中的全部运算组织成一张图：从输入数据、各层参数 $W$ 出发，经过乘法节点得到 score，再经 softmax / hinge loss 得到数据损失，与以 $W$ 为输入的正则项 $R(W)$ 相加，最终产出标量损失 $L$；前向是从左到右一步步算出 $L$，反向则是从右到左把梯度回传。这种表示对复杂架构尤其必要——讲者举了处理时序／序列数据、需要在时间步上展开（unrolling）的 **Neural Turing Machine（神经图灵机）**，若靠手工求导完全不可行，而计算图 + 反向传播能统一处理。

## 7. 反向传播核心：链式法则、局部梯度与上下游梯度 [43:50 - 53:59]

从最简单的 $f(x,y,z)=(x+y)z$ 入手，取 $x=-2,\ y=5,\ z=-4$。引入中间变量 $q=x+y=3$，则 $\partial q/\partial x=\partial q/\partial y=1$；又 $f=qz=-12$，$\partial f/\partial q=z=-4$，$\partial f/\partial z=q=3$。目标是求 $\partial f/\partial x,\ \partial f/\partial y,\ \partial f/\partial z$。

![反向传播简单示例：函数f=(x+y)z的计算图与偏导数公式](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0014_00-46-31.jpg)

反向传播从网络末端开始：$\partial f/\partial f=1$（损失对自身的导数恒为 1）。与 $f$ 直接相连的节点很容易——$\partial f/\partial z=q=3$、$\partial f/\partial q=z=-4$。而 $y$ 与 $f$ 并不直接相连，此时用**链式法则**借道中间变量：
$$\frac{\partial f}{\partial y}=\frac{\partial f}{\partial q}\cdot\frac{\partial q}{\partial y}=(-4)\times 1=-4$$
$x$ 同理也是 $-4$（两条支路的局部梯度都是 1，故得到相同值）。

由此引入两个术语：**upstream gradient（上游梯度）**是从网络末端传到当前节点的梯度，**local gradient（局部梯度）**是当前节点输出对其输入的偏导。

![函数f的局部梯度示意图，展示输入x,y与输出z及偏导数关系](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0015_00-51-11.jpg)

把每个节点抽象成「输入 $x,y$，输出 $z$」的局部函数 $f$：节点只需能算出自己的局部梯度（由函数形式决定，很容易），再从后方拿到已经算好的上游梯度，两者相乘即得到 **downstream gradient（下游梯度）**，而它又成为更前一层节点的上游梯度。于是整个过程完全是**局部、逐节点、可递归**的：不必为整个网络写出总函数，也不必让任何节点了解自己处在怎样的全局图中，每个算子只要遵守「算输出、算局部梯度」这一套局部 API 即可拼接。此外还有几种可记忆的梯度流模式（下一节结合图示展开）：add gate 是 gradient distributor，mul gate 是 swap multiplier，copy gate 是 gradient adder，max gate 则把梯度路由到取最大值的输入。

其直观意义在于：我们真正需要的是 loss 对网络中每个变量（尤其是每个 $W$）的梯度，以便沿梯度反方向迈一步；反向传播让我们无需为 100 层网络手写整体表达式，就能把 $\partial L/\partial(\cdot)$ 一步步送到每一个权重上。

## 8. Sigmoid 计算图实例与梯度流模式 [54:02 - 64:22]

第二个例子是稍复杂的 sigmoid：
$$f(w,x)=\frac{1}{1+e^{-(w_0x_0+w_1x_1+w_2)}}$$
它由乘法、加法、取负、$\exp$、取倒数 $1/x$ 等节点串成。给定具体数值（如 $w_0=2.00,\ x_0=-1.00$），前向可算出各中间值，最终输出约 $0.73$。

![Sigmoid函数f(w,x)的计算图及其前向传播数值示例](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0016_00-55-01.jpg)

反向所需的初等导数：$d(e^x)/dx=e^x$、常数乘法的导数是该常数、$d(1/x)/dx=-1/x^2$、常数加法的导数是 1。仍从 $\partial L/\partial L=1$ 出发逐节点回传：取倒数节点局部梯度 $-1/x^2$ 得 $-0.53$；常数加法局部梯度为 1，值不变仍为 $-0.53$；$\exp$ 节点局部梯度 $e^{x}$（此步输入为 $-1$）得 $-0.2$；乘以常数 $-1$ 后变 $0.2$；遇到加法节点时两个输入的局部梯度都是 1，故两支都得到 $0.2$；最后是乘法节点，其局部梯度等于「另一个变量」的值（swap）：$w_0$ 支路 $[0.2]\times[-1]=-0.2$，$x_0$ 支路 $[0.2]\times[2]=0.4$，$w_1,x_1$ 同理可求，最终就能知道每个 $W$ 该改动多少。

![Sigmoid函数反向传播计算图及链式法则应用示例](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0017_00-58-51.jpg)

讲者补充：计算图不止一种画法，可以把整串运算**合并成一个 sigmoid 节点**。sigmoid 的局部梯度化简后只依赖它自身：
$$\frac{d\sigma(x)}{dx}=(1-\sigma(x))\,\sigma(x)$$
用输入值代入再乘以单位上游梯度，得到的仍是 $0.2$，与拆开逐节点计算完全一致，但实现更紧凑。

接着正式总结四种门控的梯度流模式，配图给出前向与反向的具体数值：

- **add gate = gradient distributor**：加法对两个输入的偏导都是 1，上游梯度原样分发给各支路。
- **mul gate = "swap multiplier"**：$\partial(xy)/\partial x=y$、$\partial(xy)/\partial y=x$，梯度等于「另一个输入」的值，相当于交换乘数。
- **copy gate = gradient adder**：同一变量被复制到多处时，回传的梯度要相加。
- **max gate**：只把梯度路由（route）到前向取到最大值的那个输入方向，这与 ReLU 的行为一致。

![梯度流模式图：add gate 作为梯度分配器、mul gate 作为交换乘数器、copy gate 作为梯度加法器的前向与反向数值示例](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0018_01-01-33.jpg)

工程实现上，这让每个函数都能模块化：forward pass 时把输入**缓存**下来（乘法门反向需要用到输入），backward pass 时用「上游梯度 × 局部梯度」算下游梯度，为每个算子写统一 `forward` / `backward` API，就能把任意多的算子拼成大计算图，**梯度几乎自动就有了**。讲者指出 PyTorch 的算子正是这个形态——例如 sigmoid 的 forward 是一处，其 backward 实现（同一个 $(1-\sigma)\sigma$ 公式）写在框架的 C/C++ 代码里。

## 9. 向量、矩阵与张量反向传播 [64:22 - 76:40]

前面例子都是标量，但所有运算都可以向量化、矩阵化。按维度分三类：scalar → scalar 的导数是标量；vector($n$) → scalar 的导数是向量（每个分量表示「微调 $x$ 的该元素，标量输出 $y$ 变化多少」）；vector($n$) → vector($m$) 的导数是一个矩阵，即 **Jacobian（雅可比矩阵）**，其第 $(i,j)$ 元表示 $x_j$ 的扰动对 $y_i$ 的影响。

设 $x,y,z$ 分别是大小 $D_x,D_y,D_z$ 的向量，**损失 $L$ 始终是标量**。上游梯度 $\partial L/\partial z$ 与 $z$ 同形（$D_z$ 维）；局部梯度 $\partial z/\partial x$、$\partial z/\partial y$ 变成 Jacobian 矩阵，形状为「输入尺寸 × 输出尺寸」；下游梯度则是上游梯度与局部梯度的矩阵—向量乘，结果又回到与输入同形。**一条可牢记的规则：任何变量对 loss 的梯度，维度永远与该变量本身相同。**

![向量反向传播示例：逐元素ReLU函数的输入输出映射](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0019_01-09-06.jpg)

ReLU 是最好的向量例子：4 维输入 $[1,-2,3,-1]$ 经逐元素 $f(x)=\max(0,x)$ 得输出 $[1,0,3,0]$。因为逐元素运算的每个输出只依赖对应的那一个输入、与其他输入无关，其 Jacobian 是**稀疏矩阵，仅主对角线非零**，且取值非 0 即 1（取决于该位置是否被通过）。实践里绝不会去显式构造并存储这个巨大稀疏矩阵，而是直接按 ReLU 的规则写 backward：输入为正则梯度透传，否则置零。

推广到矩阵乃至张量时梯度形状的规则不变，但局部 Jacobian 会变成「输入尺寸 × 输出尺寸」的巨型矩阵。对矩阵乘法 $y_{n,m}=\sum_d x_{n,d}w_{d,m}$，取 mini-batch $N=64$、$D=M=4096$，仅一次乘法对应的 Jacobian 就需要约 **256 GB** 内存——必须隐式处理。

![矩阵反向传播：矩阵乘法的前向与反向流程及雅可比矩阵的维度分析](/coursenotes/cs231n-deep-learning-for-cv-sp25/l04-neural-networks-backprop/kf_0020_01-12-56.jpg)

简化办法是看元素间的影响关系：$x_{n,d}$ 只影响输出 $Y$ 的一行。再套用 mul gate 的 swap 规则可知，某个 $x_{n,d}$ 对 $y_{n,m}$ 的影响量正比于 $W$ 中由 $X$ 的列 $d$、$Y$ 的列 $m$ 定位的那个元素。于是无需构造巨型 Jacobian，$\partial L/\partial X$ 与 $\partial L/\partial W$ 都可直接写成简单的矩阵乘法：求 $X$ 的梯度时用整个 $W$ 去乘上游梯度，求 $W$ 的梯度时用整个 $X$ 去乘——这正是「交换乘数」在矩阵尺度上的体现。讲者最后总结：本讲讲了全连接网络，以及反向传播所需的 forward pass、backward pass 全部步骤，下一课进入卷积神经网络。

## 要点回顾

- 反向传播是让网络从错误中学习的核心机制，后续几乎每个算法都以某种形式使用它；Softmax 之外还有 Hinge Loss（SVM loss），它不转概率、要求正确类分数高出 margin 至少 1。
- 优化回顾：梯度下降沿负梯度按步长更新；实践中用解析梯度并以数值梯度校验；mini-batch 常取 32/64/128/256；优化器有 SGD、SGD+Momentum、RMSProp、Adam，学习率常需衰减而 Adam 类已内置。
- 两层网络 $f=W_2\max(0,W_1x)$，维度 $x\in\mathbb{R}^D,W_1\in\mathbb{R}^{H\times D},W_2\in\mathbb{R}^{C\times H}$；去掉 $\max$ 则塌缩为单个线性变换，故非线性不可省；这类网络即全连接网络 / MLP，例 3072→100→10。
- 隐藏层让网络学到更多模板（如跨类别共享的眼睛等部件），而不只是每类一张整物模板；激活函数承担制造非线性的作用。
- ReLU 是默认首选但有 dead neurons 问题；变体有 Leaky ReLU、ELU、GELU、SiLU/Swish；Sigmoid/Tanh 因 squashing 导致梯度消失，一般只用于末层。选型多靠经验，函数需可微，零中心与平滑有助收敛。
- 两层网络的 NumPy 实现不到 20 行（forward 算 loss、backward 算解析梯度、SGD 更新 $W_1,W_2$）；隐藏神经元越多容量越大、越易过拟合，类似 kNN 中 $k=1$。
- 正则化经验法则：不要拿网络规模当正则化器，而用略大的网络调 $\lambda$；$\lambda$ 约束 $W$、过大则欠拟合，需在拟合数据与约束权重间平衡。
- 计算图把全部运算组织成前向得 $L$、反向得梯度的图，避免手工为总函数求导；总损失 $L=\frac{1}{N}\sum_i L_i+\lambda R(W_1)+\lambda R(W_2)$。
- 反向传播 = 局部链式法则：$\partial f/\partial f=1$ 起，downstream = upstream × local；add 为分发、mul 为交换、copy 为相加、max 为路由；每算子缓存输入、写 forward/backward API（PyTorch 即如此）。
- 向量化后 $L$ 仍为标量、梯度形状与对应变量相同；ReLU 的 Jacobian 稀疏仅主对角线，按规则隐式计算；矩阵乘法显式 Jacobian 在 $N=64,D=M=4096$ 时约 256 GB，$\partial L/\partial X$、$\partial L/\partial W$ 用矩阵乘法（swap 规则）替代。

## 视频来源

https://www.youtube.com/watch?v=25zD5qJHYsk
