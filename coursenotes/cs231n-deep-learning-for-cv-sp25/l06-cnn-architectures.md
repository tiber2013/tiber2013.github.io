---
title: "第 6 讲 · 搭建与训练 CNN：从 VGG 到 ResNet"
description: "Stanford CS231n Spring 2025 L06 中文视频笔记（2026-11 重写版）"
---

# CS231n Spring 2025 · Lecture 6 训练卷积神经网络与 CNN 架构

**核心结论**：本讲由联合讲师 Zane Dante 讲授，明确分成两块——如何把已学过的构件拼成一个可训练的 CNN 架构，以及如何真正把它训练好。构件侧补齐了归一化层（统一遵循「标准化 + 可学习的缩放平移」，LayerNorm / BatchNorm / Instance Norm / Group Norm 的差别只在统计量的计算范围）、Dropout（训练时随机置零、测试时按概率缩放，强制冗余表示、防止特征共适应），以及从梯度消失的 Sigmoid、ReLU 到 Transformer 主流 GELU 的激活函数演进。架构案例以 VGG 的 3×3 小卷积核堆叠（三个 3×3 stride 1 与一个 7×7 感受野相同，却参数更少、非线性更多）和 ResNet 的残差连接（$H(x)=F(x)+x$，让深网至少不差于浅网）为核心，并用 Kaiming 初始化让各层激活的均值与标准差保持恒定。训练侧则覆盖逐通道数据归一化、翻转与随机裁剪/颜色抖动/测试时增强、迁移学习的冻结与微调策略，以及「过拟合小样本—粗搜学习率—看曲线」的超参数调试三步法。

## 1. 本讲地图与 CNN 组件全景 [00:05 - 03:55]

开头先回顾上一讲的关键层：卷积层用若干滤波器（例中 6 个）滑过输入，滤波器深度与输入通道数一致（32×32 RGB 即 3 通道），在每个位置做逐元素相乘求和再加偏置，得到一张 activation map，每个滤波器对应一张；输出深度等于滤波器个数，若再接一层卷积，其滤波器深度就要等于 6，末端通常接一次 ReLU 之类的非线性。池化层简单得多，用 2×2、stride 2 的窗口跳过部分位置，在高度和宽度上整合（max pooling 或 average pooling，设计新架构时两种都试）；全连接层则是一次矩阵乘加激活。

![卷积神经网络(CNN)的六大核心组件概览图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l06-cnn-architectures/kf_0001_00-03-33.jpg)

课件用一张总览图列出 CNN 的六大组件：Convolution Layers、Pooling Layers、Fully-Connected Layers、Normalization Layers、Dropout（sometimes）、Activation Functions，图上还给出 Batch Normalization 的具体形式 $\hat{x}_{i,j}=\frac{x_{i,j}-\mu_j}{\sqrt{\sigma_j^2+\epsilon}}$ 与 ReLU 曲线。本讲剩下要补的正是后三类：normalization、dropout 和激活函数。讲者把全讲分成「如何定义架构以便训练」与「如何训练」两个话题，并预告第一部分密度更高，第二部分是大量实用技巧。

## 2. 归一化层：标准化 + 可学习的缩放平移 [03:55 - 09:35]

所有归一化层都做同一个两步操作：先把输入标准化成均值 0、标准差 1 的单位高斯，再用可学习参数缩放（乘）和平移（加），让模型自己决定该点最优分布；参数通过梯度下降学出来。各层的差别只在「用哪些值算统计量、把这些统计量应用到哪些值上」。

![LayerNorm 归一化层的高层概念幻灯片，列出标准化与缩放/平移两步](/coursenotes/cs231n-deep-learning-for-cv-sp25/l06-cnn-architectures/kf_0002_00-04-20.jpg)

以当今深度学习（尤其 Transformer）最常用的 LayerNorm 为例（Ba, Kiros & Hinton, "Layer Normalization", arXiv 2016）：输入 $x$ 是 $N\times D$（batch 里 $N$ 个样本，每个 $D$ 维向量），对**每个样本单独**沿 $D$ 维算一个均值和一个标准差，统计量形状为 $N\times 1$，再广播施加形状为 $1\times D$ 的可学习参数 $\gamma,\beta$：

$$y=\gamma\frac{x-\mu}{\sigma}+\beta$$

![LayerNorm 的张量维度示意图及计算公式 y = γ(x-μ)/σ+β](/coursenotes/cs231n-deep-learning-for-cv-sp25/l06-cnn-architectures/kf_0003_00-05-01.jpg)

放到 CNN 的张量上，LayerNorm 对每个样本跨**所有通道、所有高和宽**只算一个均值和一个标准差。作为对比，BatchNorm 在 mini-batch 内对每个通道各算一个均值和标准差（跨 batch 全部数据求平均，只作用于该通道）；Instance Norm 更细粒度，Group Norm（Group Normalization 论文里的可视化）则把上述差异画进同一张图——理解了那张图就理解了所有归一化层。课堂问答还澄清：channel 就是深度，即每个空间位置上数值的个数；LayerNorm 的各样本统计量彼此独立，BatchNorm 则依赖整批数据。

## 3. Dropout：以随机置零换取泛化 [09:35 - 16:10]

Dropout 是嵌在架构里的正则化层：训练阶段注入随机性、测试阶段去掉，目标是让模型更难记住训练数据、从而泛化更好。具体做法是每次前向传播随机把该层一部分输出/激活置零，丢弃概率是固定超参，0.5 最常用、0.25 也常见（Srivastava et al., "Dropout: A simple way to prevent neural networks from overfitting", JMLR 2014）。被置零的位置可以用 mask 直接跳过计算，因为 0 乘任何数都是 0。

![Dropout正则化原理：通过随机丢弃神经元防止特征共适应的示意图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l06-cnn-architectures/kf_0004_00-11-51.jpg)

为什么有效更多是经验结论：它**强制网络学出冗余表示、防止特征共适应（co-adaptation）**。课件用猫分类举例——最后一层特征可能是 has an ear、has a tail、is furry、has claws、mischievous look，若训练集中「有耳朵且毛茸茸」几乎总对应猫，模型会死盯这几个特征的组合；随机丢弃使它在训练时不会总看到成对的强相关特征，于是必须建立更宽泛的特征—类别对应。丢弃完全随机、不针对某个类别挑选特征；代价是训练时只看到特征子集、训练集上反而变差，但测试时不再丢弃，泛化更好。

![Dropout操作定义：前向传播中随机置零神经元，常用概率为0.5](/coursenotes/cs231n-deep-learning-for-cv-sp25/l06-cnn-architectures/kf_0005_00-12-56.jpg)

关键的测试时处理：若训练丢弃 50%，测试时每个神经元会多见到约 50% 的输入，幅值失衡，因此要把激活乘以概率 $p$ 缩放（在「$p$ 表示保留概率」的实现里即乘 $p$），使训练与测试的期望输出、方差一致：

```python
def predict(X):
    H1 = np.maximum(0, np.dot(W1, X) + b1) * p  # scale the activations
    # output at test time = expected output at training time
```

![Dropout测试阶段的代码实现及激活值缩放原理说明](/coursenotes/cs231n-deep-learning-for-cv-sp25/l06-cnn-architectures/kf_0006_00-14-13.jpg)

训练侧用伯努利 mask（如 `p = 0.5; U1 = np.random.rand(*H1.shape) < p`），测试侧只做缩放。

![Dropout机制总结：训练时随机丢弃与测试时缩放的对比图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l06-cnn-architectures/kf_0007_00-15-58.jpg)

反向传播上，被置零的激活对应路径梯度为 0，类似 ReLU 在负区的表现，计算图中更靠前的部分不再有梯度，被丢弃激活关联的权重在这一步不更新。讲者还回应了「能否直接给图像加噪声」——可以，正是后面数据增强要讲的做法。

## 4. 激活函数：从 Sigmoid 到 GELU [16:10 - 21:45]

卷积、全连接本质上都是乘加的线性运算，激活函数唯一的作用是引入非线性，因此放在线性算子（卷积层、全连接层 / feed-forward 层）之后。历史上 Sigmoid 很常用，$\sigma(x)=\frac{1}{1+e^{-x}}$，但它有个致命问题：

![Sigmoid激活函数的公式、曲线图及其导致梯度消失的关键问题说明](/coursenotes/cs231n-deep-learning-for-cv-sp25/l06-cnn-architectures/kf_0008_00-17-46.jpg)

Sigmoid 曲线在很负和很正的两端几乎平坦、导数趋近 0，只有中间很窄一段非零；堆叠多层后从损失端回传的梯度逐层变小，早期层几乎收不到梯度（梯度消失）。ReLU 在正区导数恒为 1、且只需 $\max(0,x)$ 比指数运算便宜，因而流行；但负半轴梯度仍为 0。更近的做法是在 0 附近保留非零、平滑的梯度：GELU（Gaussian Error Linear Unit）用高斯正态的累积分布函数 $\Phi(x)$ 加权，极负趋近 0、极正趋近 $x$，两端都收敛到 ReLU，是当今 Transformer 的主要激活；讲者还展示了形状相近的另一族平滑激活（课件出现 CELU 等标注），其中一类可写成 $x\cdot\sigma(x)$，与 $\Phi(x)$ 形状相似，所以几条曲线看起来几乎一样。整体规律是：一段相对平坦，极限处逼近 $f(x)=x$ 的线性段。

## 5. 深度革命与 VGG：为什么要堆小卷积核 [21:45 - 29:03]

![ImageNet ILSVRC竞赛历年冠军模型错误率与网络层数对比柱状图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l06-cnn-architectures/kf_0009_00-22-31.jpg)

课件把 ILSVRC 历届冠军的蓝色错误率柱与橙色层数三角画在一起：错误率显著下降、首次超过人类水平的节点，正好对应层数暴涨。AlexNet（Krizhevsky et al., 2012）是首个在 ImageNet 上真正奏效的 CNN，靠 GPU 训起来；随后 VGG（Simonyan & Zisserman, 2014）成为 2010 年代的标准架构。

![VGGNet案例研究：3x3卷积层的有效感受野推导示意图及VGG16/19网络结构图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l06-cnn-architectures/kf_0010_00-25-33.jpg)

用 block diagram 并排看 AlexNet 与 VGG：橙块是 stride 1、padding 1 的 3×3 卷积（保持空间尺寸不缩小），中间穿插 max pooling，末尾是两层 4096 维全连接再接 1000 维输出（ImageNet 正好 1000 类，最后一层维度恒等于类别数）。VGG 看起来只是「放大的 AlexNet」——更多层、连续三组甚至四组 3×3 卷积后再池化，但仅靠三种层就远超此前所有方法。

![VGGNet案例研究：AlexNet、VGG16与VGG19网络结构对比图，展示小卷积核堆叠原理](/coursenotes/cs231n-deep-learning-for-cv-sp25/l06-cnn-architectures/kf_0011_00-28-01.jpg)

选 3×3 的关键理由是有效感受野：每个输出值在 A1 看输入 3×3，A2 变成 5×5，A3 变成 7×7，即每层给感受野 +2，所以**三个 3×3（stride 1）堆叠与一个 7×7 卷积感受野相同**。设通道数为 $C$：三层 3×3 的参数是 $3\times(3\cdot3\cdot C\cdot C)=27C^2$，单层 7×7 是 $7\cdot7\cdot C\cdot C=49C^2$——小核堆叠参数更少，还多出两次激活带来的非线性，能建模更复杂的关系。讲者也坦言这究竟是设计前的直觉还是事后解释因架构而异，而 ResNet 恰恰是「假设驱动研究」的例子。

![ImageNet ILSVRC历届冠军模型错误率柱状图，展示从AlexNet到ResNet的深度革命趋势](/coursenotes/cs231n-deep-learning-for-cv-sp25/l06-cnn-architectures/kf_0012_00-29-04.jpg)

历年错误率印证这条「深度革命」：2010 Lin et al 28.2%、2011 Sanchez & Perronnin 25.8%、2012 AlexNet 16.4%、2013 Zeiler & Fergus 11.7%、2014 VGG 7.3% 与 GoogLeNet 6.7%、2015 ResNet 3.6%、2016 Shao et al 3%、2017 SENet 2.3%，人类基准 Russakovsky et al 为 5.1%，深度从 8 层一路到 152 层。

## 6. ResNet：用残差连接驯服深度 [29:03 - 42:03]

ResNet 的起点是一个反直觉的经验发现：给 plain CNN 不断加层，56 层模型的**训练误差和测试误差都高于** 20 层模型——不是过拟合，而是更深的网络更难优化（延长训练也不会收敛到浅网水平，会卡在局部极小，其成因仍是活跃研究）。理论上深网可表达的函数集是浅网的超集，只要把多余的层学成恒等映射就至少不差于浅网，问题在于让卷积层去学恒等映射并不容易。

![ResNet残差块结构示意图：对比普通层与残差映射（H(x)=F(x)+x）的架构差异](/coursenotes/cs231n-deep-learning-for-cv-sp25/l06-cnn-architectures/kf_0013_00-32-00.jpg)

解法是**不直接拟合目标映射 $H(x)$，而拟合残差 $F(x)=H(x)-x$**，即 $H(x)=F(x)+x$：把输入 $x$ 沿跳跃连接（residual / shortcut connection）原样复制、越过两层卷积后与输出相加。这样模型只要把卷积滤波器学成 0（实际是学很小的值），就得到 $F(x)=0$、$H(x)=x$ 的恒等映射，等价于「学习相对原输入的 delta」，却仍逐层抽象出高层特征（He et al., 2015）。加法要求两侧张量尺寸一致，这正是残差块内普遍用 3×3、stride 1、pad 1 保持尺寸的原因；池化之后尺寸变化就不能朴素相加。Transformer 用残差块也是同一理由，而且它让你能更高效地利用更多数据。

![ResNet案例研究幻灯片，展示残差块结构和完整ResNet网络架构图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l06-cnn-architectures/kf_0014_00-40-35.jpg)

完整 ResNet 堆叠残差块，每块含两个 3×3 卷积（3×3 conv + ReLU + 3×3 conv），相加后再接 ReLU；周期性地把滤波器数翻倍、并用 stride 2 在空间上下采样（每维 /2，activation volume 减半），于是激活图空间变小、深度变大，最后摊成向量做分类。开头还有一个相对较大的卷积层（纯经验有效）。ResNet 是一个家族，从 18、34 层一路到 101、152 层，深度增加带来提升但边际递减，101→152 只约 1%，且受 GPU 显存限制需分别训练。它是首次成功训练 100+ 层模型的架构，之后几乎每个视觉任务都在用；每块用两个而非一个 3×3 卷积，理由与 VGG 相同——更多非线性、更富表达力。

## 7. 训练前的准备：权重初始化与数据处理 [42:08 - 57:59]

### 7.1 权重初始化：Kaiming / MSRA [42:08 - 48:59]

初始值过小或过大都会在深层网络里放大成灾难：一个 6 层、每层 4096 维的全连接网络用单位高斯乘 0.01 初始化并接 ReLU 时，各层激活的均值/标准差逐层衰减到 0（mode collapse）；换成 0.05 则逐层爆炸。理想状态是各层均值与标准差基本恒定，这个具体例子里合适值约 0.022 量级，但不可能逐层手调。

![Kaiming/MSRA初始化的Python实现代码及ReLU修正公式，下方展示了各层激活值分布直方图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l06-cnn-architectures/kf_0015_00-46-20.jpg)

通用做法是 Kaiming（MSRA）初始化（He et al., 2015《Delving Deep into Rectifiers》，与 ResNet 同一作者 Kaiming He）：从单位高斯采样后乘以 $\sqrt{2/D_{in}}$，

```python
W = np.random.randn(Din, Dout) * np.sqrt(2 / Din)
```

这个「魔法公式」配合 ReLU 能让各层激活的标准差保持恒定；推导见论文，卷积层里 $D_{in}$ 换成卷积核规模（如 3×3×6）。讲者补充：归一化层确实能缓解激活爆炸，但无法替代良好的初始化；若自定义了新型层，仍需实验多种初始化方案。

### 7.2 数据预处理与数据增强 [49:06 - 57:59]

图像预处理很直接：对数据集算每个通道（R/G/B）的均值与标准差，输入图减均值、除标准差；实践中常直接复用 ImageNet 的均值/标准差，即使并不在 ImageNet 上训练，取值随数据集而变。

数据增强与 Dropout 是同一个模式——训练时注入随机性、测试时把随机性平均掉，既扩充数据、提高泛化，也让训练损失升高、更难靠记忆过拟合。具体手段：

- **翻转**：horizontal flip 对日常物体（多近似对称）很有效，但读文字任务是灾难（镜像后读不出来）；vertical flip 只适合显微/俯视等任意朝向的数据。
- **随机缩放裁剪（random resized crop）**：最常用。

![数据增强中的随机裁剪缩放与测试时增强策略，包含ResNet具体参数设置](/coursenotes/cs231n-deep-learning-for-cv-sp25/l06-cnn-architectures/kf_0016_00-55-49.jpg)

ResNet 的做法是从 $[256,480]$ 随机取边长 $L$，把图像短边 resize 到 $L$（保持相对分辨率，如 800×600 取 $L=256$ 时 600 变 256、800 等比缩放），再随机裁一个 224×224 patch。测试时增强（Test Time Augmentation）则把一组固定 crop/resize 都跑一遍再平均预测：5 个尺度 $\{224,256,384,480,640\}$，每个尺度取 10 个 224×224 crop（4 角 + 中心 + 各自翻转），通常能换来 1%~2% 的提升，但收益递减。

![颜色抖动（Color Jitter）数据增强方法示意图，展示对比度和亮度随机化效果](/coursenotes/cs231n-deep-learning-for-cv-sp25/l06-cnn-architectures/kf_0017_00-56-34.jpg)

- **Color Jitter**：随机化对比度与亮度。
- **遮挡**：用黑/灰方块盖住局部，模拟相机被遮挡，增强鲁棒性（较少用，但说明增强可以按问题自由设计）。

判断增强强度的好标准是：变换后图像人眼仍易辨认、且看起来仍在原数据分布内。

## 8. 迁移学习：在小数据上复用 CNN [58:00 - 67:14]

现实中很少有人拥有 ImageNet 级别的百万张图。CNN 的层级特征给出了解法：浅层是边缘/纹理等通用模式，越往后越具体；把图像送到分类层之前取倒数第二层（4096 或 2048 维向量），同类图像在该向量空间的 L2 距离很近，等价于用学出来的特征做最近邻或线性分类。

![基于CNN的迁移学习策略图，对比小数据集和大数据集下的层冻结与微调方案](/coursenotes/cs231n-deep-learning-for-cv-sp25/l06-cnn-architectures/kf_0018_01-01-38.jpg)

策略分三步：① 在 ImageNet 上训练（或直接取别人训好的模型）；② 小数据集时**冻结**预训练的所有卷积层、只把最后一层重初始化成自己的 $C$ 类（用 Kaiming 初始化）并训练——此时冻结的网络相当于传统视觉里的固定特征抽取器，训练几乎只是在 4096 维向量上拟合一个线性分类器；③ 数据较多时从预训练权重初始化后**微调全部层**效果最好，讲者自己几乎所有项目都走第三步，因为模型见过更海量数据、特征更好，微调后又能贴合具体任务。

![CNN网络层级（从Image到FC层）与数据量/数据集相似度二维矩阵示意图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l06-cnn-architectures/kf_0019_01-03-54.jpg)

选择哪种策略由「数据量 × 数据集相似度」二维矩阵决定：相似且数据少→线性分类器；相似且数据多→微调全层；很不同且数据多→可从零训练，或仍试预训练初始化（无保证，需实验）；很不同且数据少→最难，应尽量找相近领域训过的模型，域外泛化仍是活跃研究、没有通用解法。冻结会带入预训练分布偏差：对笔记本、教室、人等日常物体好，对火星照片就差。介于「只训最后一层」与「全训」之间还有 LoRA——不直接改权重，而是用很少的参数学习各层权重的低秩差异（思路类似 ResNet 学 delta）；现成预训练模型可在 PyTorch Image Models（timm）与 torchvision 里找。

## 9. 超参数选择与训练调试 [67:14 - 70:59]

![超参数选择三步法：检查初始损失、过拟合小样本、寻找使损失下降的学习率](/coursenotes/cs231n-deep-learning-for-cv-sp25/l06-cnn-architectures/kf_0020_01-07-34.jpg)

模型跑不通时，默认的调试策略是**先过拟合一个小样本**：只喂一个数据点，训练损失应能降到 0；做不到就说明代码有 bug 或模型选错。这一步同时能探出哪些学习率可用。课件的三步法是 Step 1 检查初始损失、Step 2 过拟合小样本、Step 3 找到能让损失下降的学习率，建议尝试 $10^{-1}$ 到 $10^{-5}$ 这一档。之后先做粗网格：用不同学习率训练，挑训练损失在约一个 epoch 内持续下降最稳的，再扩展到其他超参。

除损失外要看训练/验证准确率曲线：两者仍在上升就继续训练；若训练损失下降而验证损失上升即过拟合，需加强正则或加更多数据；若训练与验证准确率差距很小，可一直训到二者开始分离。多超参搜索时**随机搜索优于网格搜索**——网格会在不重要的超参上重复采样、浪费预算，而随机采样能把重要超参的取值空间探索得更充分，定义好范围后持续随机取组合即可。

## 要点回顾

- 本讲两块：搭建 CNN 架构（组件 + 经典案例 + 初始化）与训练 CNN（数据处理、增强、迁移学习、超参调试）。
- 归一化层统一为「标准化到单位高斯 + 可学习 $\gamma,\beta$ 缩放平移」；LayerNorm 对每样本跨全部通道/高/宽算统计量，BatchNorm 在 mini-batch 内逐通道算，另有 Instance / Group Norm。
- Dropout 训练时按 $p$（0.5 / 0.25）随机置零、测试时乘 $p$ 缩放以保持期望；作用是强制冗余表示、防止特征共适应；被丢弃激活对应权重该步不更新。
- Sigmoid 两端饱和导致梯度消失；ReLU 正区导数为 1 且便宜但负区死梯度；GELU 用高斯 CDF $\Phi(x)$ 平滑，是 Transformer 主流激活；激活一律放在卷积/全连接之后。
- VGG（Simonyan & Zisserman, 2014）堆叠 3×3 stride1 pad1 卷积：三个 3×3 与一个 7×7 感受野相同，参数 $27C^2<49C^2$ 且非线性更多；末尾两层 4096 维 FC + 1000 类输出。
- ResNet（He et al., 2015）源于「56 层比 20 层训练/测试误差都高」的优化难题；残差块拟合 $H(x)=F(x)+x$，靠跳跃连接让深网轻易学成恒等映射，首次训成 100+ 层，18/34 到 101/152 层家族边际递减。
- Kaiming 初始化 $W=\text{randn}(D_{in},D_{out})\cdot\sqrt{2/D_{in}}$ 让各层激活均值/方差恒定，避免过小衰减到 0、过大爆炸；卷积层 $D_{in}$ 用 3×3×C 规模。
- 训练实践：逐通道减均值除标准差（常用 ImageNet 统计量）；翻转、random resized crop（$L\in[256,480]$、裁 224×224）、Color Jitter、遮挡、测试时 5 尺度×10 crop 平均（+1~2%）。
- 迁移学习：小数据冻结卷积只训新 FC，大数据从预训练初始化微调全部；策略由数据量×相似度矩阵决定，LoRA 用低秩差异微调，注意预训练分布偏差与域外泛化难题。
- 调试：先过拟合单样本排 bug，再粗搜学习率（$10^{-1}\sim10^{-5}$），看训练/验证曲线判别过拟合，随机搜索优于网格搜索。

## 视频来源

https://www.youtube.com/watch?v=aVJy4O5TOk8
