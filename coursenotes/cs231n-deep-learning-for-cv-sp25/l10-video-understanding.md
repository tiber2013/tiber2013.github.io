---
title: "第 10 讲 · 视频理解：从 3D 卷积到视频 Transformer"
description: "Stanford CS231n Spring 2025 L10 中文视频笔记（2026-11 重写版）"
---

# CS231n Spring 2025 · Lecture 10 视频理解：从 2D 图像到 4D 时空张量

**核心结论**：本讲由客座讲师 Rohan Gao 讲授视频理解——视频就是在 2D 图像之外多出时间维的 4D 张量（$3\times T\times H\times W$），任务重心也从识别物体/场景转向识别动作。由于视频数据极大，实践中先把长视频在空间与时间上降采样、切成短 clip 训练，测试时再平均多个 clip 的预测；而最简单、也最该先跑的基线是「单帧 2D CNN 平均各帧预测概率」。围绕「何时融合时间信息」，课程依次对比 Late Fusion（在高层特征上拼接或池化）、Early Fusion（第一层卷积就把 $3T$ 折叠成 $D$）与 3D CNN（Slow Fusion，用 3D 卷积/池化逐步建立时空感受野，并获得时间平移不变性）；Sports-1M 上四种方法 Top-5 准确率为 77.7 / 76.8 / 78.7 / 80.2。此后又用光流双流网络显式建模运动，用 RNN/LSTM、循环卷积网络与自注意力（non-local）建模长程时序，用 I3D 把 2D 网络与权重「膨胀」到 3D，并把视频 Transformer 推到 Kinetics-400 约 90% 的准确率。最后超出片段分类，讲者还覆盖时序动作定位、视听多模态（视觉引导的音源分离）、高效视频理解以及 egocentric 与视频级基础模型等前沿方向。

## 1. 从图像到视频：4D 张量、动作分类与数据规模挑战 [00:05 - 08:29]

本讲是课程首位客座讲师：Rohan Gao，University of Maryland, College Park 计算机系助理教授，领导 multisensory machine intelligence lab，2022–2023 年间曾任 CS231n 讲师。他的研究兴趣是多模态——不止视觉，还包括音频、触觉等，让机器像人一样感知、理解并与多感官世界交互；但视觉仍是最重要的模态，因此本课聚焦 deep learning for computer vision。

先回顾已学的 2D 图像任务：给一张猫图，从标签集合 $\{dog, cat, truck, plane, \dots\}$ 里输出 `cat` 是图像分类；此外还有语义分割（把图分成 grass / cat / tree 等有语义的区域）、目标检测（给狗、猫加边界框）、实例分割（同类的每只狗各有自己的掩码）。但世界不是静止的，今天要处理的是**视频**。

![回顾二维图像分类任务：从猫的图片到标签cat的映射过程](/coursenotes/cs231n-deep-learning-for-cv-sp25/l10-video-understanding/kf_0001_00-01-27.jpg)

形式化地说，**视频 = 2D 图像 + 时间**：输入从 $3\times H\times W$ 变成 $3\times T\times H\times W$（$T$ 为时间维、$H/W$ 为空间维），即把视频看作一叠帧组成的 4D 体数据。示例任务是视频分类：给一段有人跑步的视频，训练深度学习模型判断是 swimming、running 还是 jumping 等动作。工具层面并无新东西——取特征后用前面学过的交叉熵损失就能训练分类器；真正的新问题是**如何得到视频的特征**。另外，任务的性质也变了：图像分类通常关心场景与物体类别，视频分类通常关心动作/活动，要识别的东西天然不同。

最现实的约束是数据太大。图像只是 $3\times H\times W$ 的单个 RGB 矩阵，视频是每秒 30 帧甚至更高分辨率、更高时间分辨率的帧序列：标清视频约 **1.5 GB/分钟**，$1920\times1080$ 的高清视频约 **10 GB/分钟**。不仅存储空间惊人，也根本无法把整段视频直接塞进 GPU——还要另外存权重、卷积网络的激活值，模型会非常庞大。两个层面的解法：

- **把视频变小**：在时间上和空间上同时降采样。例如 3.2 秒的片段，每秒只取 5 帧（帧间大量冗余），空间分辨率降到 $112\times112$，就能压到约 588 KB；有算力时当然也可以用更大分辨率。
- **在 clip 上训练**：长视频可以是几分钟、几小时，做法是在低 FPS 的短 clip 上训练分类器，用滑动窗口采样大量不同 clip 作为训练数据；推理时采样若干个 clip（例如 10 个），把预测结果**平均**作为整段长视频的预测。

## 2. 单帧 CNN：永远先跑的强基线 [08:29 - 11:16]

既然视频就是一序列图像，最简单的做法是把每帧当独立图像：用已经训练好的普通 2D CNN 对每帧分类，测试时把各帧预测概率**平均**。对于跑步这类帧间变化不大的视频，身体略有移动但整体外观几乎一致，几乎每帧都会输出 `running`，平均之后自然还是 `running`。讲者强调，这往往是**非常强的 baseline**：设计任何视频分类器前都应该先跑它，很可能已经拿到不错的结果。

单帧 CNN 的流程就是「训练普通 2D CNN 逐帧独立分类 + 测试时平均预测概率」，课件把它明确标为 often a very strong baseline。

![单帧CNN视频分类方法示意图，展示对每一帧独立进行2D CNN分类并平均预测概率的流程](/coursenotes/cs231n-deep-learning-for-cv-sp25/l10-video-understanding/kf_0002_00-08-32.jpg)

关键而容易被忽视的问题是**如何采样帧**——给定一小时视频，该把 CNN 放在哪几帧上？这本身是活跃研究方向。最简单的随机采样（例如每分钟取一帧）能给出一定结果，但显然不够聪明；更复杂的方法会先采一帧、用它的分类决策来决定下一帧去哪里采，讲者预告后面会给出具体例子。

## 3. 融合的时机：Late Fusion 与 Early Fusion [11:16 - 18:20]

单帧 CNN 完全不做跨帧融合。往前一步，可以在不同阶段把多帧信息合并，按「融合发生得多早」分成三类架构。

### 3.1 带全连接层的 Late Fusion [11:16 - 13:23]

仍对每帧跑 2D CNN，得到高层外观特征图 $D\times H'\times W'$；$T$ 帧就有 $T\times D\times H'\times W'$。把这些特征图展平成向量再拼接，得到一个包含全部帧信息的巨型向量，用 MLP 把它映射到较低维度、再映射到类分数 $C$。之所以叫 **late**：每帧特征的提取彼此完全独立，直到网络很靠后的阶段才拼接、才做分类。明显缺点是展平+拼接后的向量随 $T$ 增大，全连接层参数量爆炸，效率很差。

![视频分类晚期融合（带全连接层）的架构流程图，展示帧特征展平后输入MLP的过程](/coursenotes/cs231n-deep-learning-for-cv-sp25/l10-video-understanding/kf_0003_00-11-16.jpg)

### 3.2 带池化的 Late Fusion 与其代价 [13:23 - 16:16]

不做拼接，而是对帧特征在**空间与时间上做平均池化**：先把 $T\times D\times H'\times W'$ 压成 clip 特征 $D$，再用线性层 $D\to C$，训练交叉熵损失。好处是不再需要巨型全连接层；坏处是池化会丢掉可能很重要的信息。

更根本的问题在「late」二字：用 2D CNN 逐帧处理、经过大量卷积与池化后，到很深的层里只剩高层语义（例如「有人」），**低层运动信息已经丢失**。讲者用画面中红圈标出的跑者双脚上下摆动为例——这正是判断「跑步」的关键线索，但在晚期特征图里已经不存在了。直觉上，早期层靠近原始帧，更可能保留这类低层运动；如果在那里就跨时间拼接或池化，就能分析运动随时间的变化。

![视频分类晚期融合（带池化）的架构流程图，展示通过平均池化层压缩特征后输入线性层的过程](/coursenotes/cs231n-deep-learning-for-cv-sp25/l10-video-understanding/kf_0004_00-13-27.jpg)

### 3.3 Early Fusion：在第一层就折叠时间 [16:16 - 18:20]

为了在贴近原始帧的地方处理运动，把输入 $T\times3\times H\times W$ 直接 reshape 成 $3T\times H\times W$，即用**第一个 2D 卷积层**把通道维从 $3T$ 映射到 $D$，在第一层就把所有时间信息折叠掉；之后的网络就是标准 2D CNN 与图像分类。每帧给出一个特征 $D$，再对特征做 mean pooling 或 max pooling——注意这里是对**特征**池化，不是对帧。课件点明了它的局限："One layer of temporal processing may not be enough!"：想用单层卷积捕捉全部时间信息过于野心勃勃，未必能达到目的。于是自然需要介于「最后一次性融合」与「开头一次性融合」之间的方案。

![视频分类早期融合的架构流程图，展示将时间维度折叠进通道维度后输入2D CNN的过程及其局限性](/coursenotes/cs231n-deep-learning-for-cv-sp25/l10-video-understanding/kf_0005_00-15-36.jpg)

四种架构的张量流向可压缩为：

```text
Late Fusion (FC)    T x 3 x H x W --2D CNN--> T x D x H' x W' --Flatten+MLP--> C
Late Fusion (pool)  T x 3 x H x W --2D CNN--> T x D x H' x W' --AvgPool--> D --Linear--> C
Early Fusion        T x 3 x H x W --Reshape--> 3T x H x W --Conv2D(3T->D)--> D x H x W --> C
3D CNN              3 x T x H x W --Conv3D / Pool3D--> D x T' x H' x W' --> C
```

## 4. 3D CNN（Slow Fusion）：3D 卷积、感受野与时间平移不变性 [18:20 - 32:01]

### 4.1 3D 卷积与 3D 池化 [18:20 - 21:46]

折中方案就是 3D CNN，也叫 **slow fusion**：用 3D 版本的卷积与池化，在整个网络过程中**逐步**融合时间信息，而不是在某个极端阶段一次性完成。网络每一层都是 4D 张量 $D\times T\times H\times W$，输入 $3\times T\times H\times W$，最后展平接全连接得到类分数 $C$。

![视频分类的3D CNN架构示意图，说明使用4D张量和3D卷积/池化操作融合时间信息](/coursenotes/cs231n-deep-learning-for-cv-sp25/l10-video-understanding/kf_0006_00-18-20.jpg)

对比 2D 卷积：$32\times32\times3$ 的图像配 $5\times5\times3$ 的卷积核，核在 $H,W$ 上滑窗、并贯穿全部通道深度，得到 $28\times28\times1$ 的激活图。3D 卷积多出一个维度：输入 $C\times T\times H\times W$（画面只能画 3D，未画出的那维就是通道 $C$，每个网格点上有 $C$ 个特征），卷积核如 $6\times6\times6$ 要在 $T\times H\times W$ 这个立方体上滑动，同样贯穿整条通道维；后面再接 $5\times5$ 一类层，最终展平接全连接。本质就是「2D 卷积 + 一个时间维」。

### 4.2 玩具例子：三种架构的张量尺寸与感受野 [21:46 - 26:10]

讲者用输入 $3\times20\times64\times64$（$T=20$）的简化网络走了一遍尺寸与感受野（$T\times H\times W$），课件表格如下：

- **Late Fusion**：`Conv2D(3x3, 3->12)` → $12\times20\times64\times64$，RF $1\times3\times3$；`Pool2D(4x4)` → $12\times20\times16\times16$，RF $1\times6\times6$；`Conv2D(3x3, 12->24)` → $24\times20\times16\times16$，RF $1\times14\times14$；`GlobalAvgPool` → $24\times1\times1\times1$，RF $20\times64\times64$。时间维始终是 20，只在**最后一层**一次性建立时间感受野（all-at-once at end），空间感受野逐层增长。
- **Early Fusion**：`Conv2D(3x3, 3*20->12)` → $12\times64\times64$，RF $20\times3\times3$；`Pool2D(4x4)` → $12\times16\times16$，RF $20\times6\times6$；`Conv2D(3x3, 12->24)` → $24\times16\times16$，RF $20\times14\times14$；`GlobalAvgPool`（只在空间上平均）→ RF $20\times64\times64$。时间感受野在**第一层**就从 1 跳到 20（all-at-once at start），空间感受野仍缓慢增长。
- **3D CNN / Slow Fusion**：`Conv3D(3x3x3, 3->12)` → $12\times20\times64\times64$，RF $3\times3\times3$；再经 $4\times4\times4$ 池化、另一层 `Conv3D`，最后在 $4\times16\times16$ 上做全局平均池化。空间与时间**同时**逐步建立感受野。

![Late Fusion、Early Fusion与3D CNN三种架构的层级结构与感受野对比表](/coursenotes/cs231n-deep-learning-for-cv-sp25/l10-video-understanding/kf_0008_00-25-06.jpg)

### 4.3 时间平移不变性与卷积核可视化 [26:10 - 32:01]

Early Fusion 与 3D CNN 都建立了跨时间的感受野，那差别在哪？把每个空间网格点看成一个特征向量：Early Fusion 的 2D 卷积核在**空间上是局部的，时间上却贯穿全部** $T=16$，于是**没有时间平移不变性（no temporal shift invariance）**。举例，视频里同一类「蓝→橙」的颜色变化可能发生在第 4 帧、也可能发生在第 15 帧；若核贯穿整条时间轴，就得为不同时间步学两个独立的核才能识别同一种变化。类比图像分类里的空间不变性——我们希望无论猫在左上还是右下都能识别，因而共享卷积核；这里我们同样希望在不同时间步共享核来识别同类运动模式。

3D CNN 的做法是让时间核长 $T=3$，只覆盖一个局部时间窗并沿时间轴滑动，从而**获得时间平移不变性**：同一个核可以在任意时刻识别「蓝→橙」的跃迁，不必为每个时间步单独特化，表示效率更高。此外，3D CNN 学到的卷积核本身可以当成一小段视频来可视化：有的像普通 2D 滤波器只学到颜色斑块与边缘，有的呈现从一种颜色/边缘模式到另一种的**时间跃迁**，有的学到不同方向的运动。讲者归纳两点差异：一是 slow fusion（逐步建立时空感受野），二是卷积运算里真正多了时间这一维。

## 5. 数据集与经典结果：Sports-1M、C3D 与算力代价 [32:01 - 40:02]

架构之外需要数据。**Sports-1M**（2014）包含 **100 万段 YouTube 视频、487 种细粒度运动类别**（例如 marathon 与 ultramarathon 的细微差别），课件用蓝色标 ground truth、绿色标 Top-5 中的正确预测、红色标错误预测。数据集还有个现实问题：视频太大，无法像 ImageNet 那样打包下载，最初是以**一列 YouTube URL** 发布的；用户会修改、删除视频，如今大约一半链接已失效，数据集很不稳定。

![Sports-1M数据集介绍，展示5个运动类别的预测结果及Ground Truth对比](/coursenotes/cs231n-deep-learning-for-cv-sp25/l10-video-understanding/kf_0009_00-32-05.jpg)

在 Sports-1M 上把前面几种分类器放在一起比较，Top-5 准确率是：Single Frame **77.7**、Early Fusion **76.8**、Late Fusion **78.7**、3D CNN **80.2**。最「反直觉」的结论是单帧模型已相当好，讲者反复强调 **always try single frame first**；3D CNN 只带来约 2–3% 的提升，但那是 2014 年的 3D CNN，此后十年这类网络进步巨大。

![Sports-1M Top-5准确率对比柱状图：Single Frame vs Early Fusion vs Late Fusion vs 3D CNN](/coursenotes/cs231n-deep-learning-for-cv-sp25/l10-video-understanding/kf_0010_00-33-13.jpg)

早期流行的 3D CNN 是 **C3D**（"The VGG of 3D CNNs"）：架构很像用于 2D 图像分类的 VGG，只是把卷积换成 $3\times3\times3$、池化换成 $2\times2\times2$，除第一层外与 VGG 高度对应；训练数据正是 Sports-1M。2014 年算力稀缺，它在 Facebook 训练后**发布了预训练权重**，很多无力自训视频模型的研究者便把它当作特征提取器：输入固定长度（16 或 32 帧）的 clip，抽取 clip-level 特征再接自己的分类器。代价是计算量——GFLOP 衡量一次前向的浮点运算数：AlexNet **0.7**、VGG-16 **13.6**、C3D **39.5 GFLOP（2.9× VGG）**，把 VGG 粗暴映射到 3D 非常昂贵，换来的只是 Sports-1M 上约 +4% 的 Top-5。当然，2D 里学过的技巧（如 ResNet 的残差连接）都可以继续搬到 3D 架构上。

![C3D网络结构介绍及层参数表，对比计算量](/coursenotes/cs231n-deep-learning-for-cv-sp25/l10-video-understanding/kf_0011_00-38-20.jpg)

## 6. 双流网络与光流：显式分离外观与运动 [40:02 - 45:22]

空间信息与时间信息本质不同，也许该**显式建模运动**。讲者让听众仅凭画面上几个运动点（完全没有外观信息）猜动作，人依然能判断得不错——说明外观与运动可以用不同的网络分别处理，这正是 2014 年 **two-stream network** 的动机。

显式度量运动的工具是**光流（optical flow）**：估计相邻帧之间每个像素的移动，即帧内各点的速度、以及它在下一帧可能的位置。它给出一个位移场

$$F(x,y)=(dx,dy),\qquad I_{t+1}(x+dx,\,y+dy)=I_t(x,y)$$

计算光流有很多研究，常见做法是假设物体移动时亮度保持恒定（brightness constancy）再求解。因为位移有两个分量，可以把水平流 $dx$ 与垂直流 $dy$ 分别可视化，得到低层运动线索；课件此处引用 Simonyan and Zisserman, NeurIPS 2014。

![光流测量运动原理图，展示帧间位移场及像素移动公式](/coursenotes/cs231n-deep-learning-for-cv-sp25/l10-video-understanding/kf_0012_00-42-23.jpg)

双流网络据此训练两个分类器：一条 **spatial/appearance stream** 吃单帧 RGB 做外观分类；另一条 **temporal/motion stream** 对每两相邻帧计算光流图，把水平光流与垂直光流堆叠起来，用时间流 CNN 处理并给出预测；最后把两路预测聚合。在 **UCF101**（101 个动作类别）上，一个令人意外的结果是 **motion-only 明显优于 spatial-only（仅外观）**。讲者的假设是：运动流里含有动作的关键信息，同时去掉了与动作无关的大量背景，因此更不容易过拟合。

## 7. 建模长程时序：RNN/LSTM、循环卷积网络与自注意力 [45:22 - 54:18]

前面处理的都是短时结构。要建模更长程的时间依赖，可以用 CNN（2D 单帧或 3D clip）抽取局部特征向量，再交给 RNN/LSTM 处理序列：既可以是 **many-to-one**（整段视频只在最后时间步输出一次，做视频级分类），也可以是 **many-to-many**（每帧一个输出）。这个想法 2011 年就被探索过（早于 2012 年的 AlexNet），2015 年的论文把它普及。实践中往往**只反传 RNN 层**、把 CNN 冻结或预训练（例如直接拿 C3D 当特征提取器），否则 CNN + RNN 端到端很难训练。

![CNN提取局部特征后通过LSTM处理长程依赖的架构示意图（Many-to-many）](/coursenotes/cs231n-deep-learning-for-cv-sp25/l10-video-understanding/kf_0013_00-46-58.jpg)

把 CNN 与 RNN 合进单一架构就是**循环卷积网络（Recurrent Convolutional Network, RCN）**。多层 RNN 的每个时刻接收两个输入：同层的前一时刻、以及上一层同一时刻。对应到视频，特征图是 $C\times H\times W$ 的三维网格，每个特征图同样依赖「同层上一时刻」与「前一层同一时刻」两个 3D 张量。RNN 的一般形式是 $h_{t+1}=\tanh(W_h h_t + W_x x)$；RCN 把其中的**矩阵乘法换成 2D 卷积**：对两个输入特征图分别做 2D 卷积、相加、再过 tanh，得到当前隐状态特征图。GRU、LSTM 等变体同理，从而在一个网络里同时完成空间与时间融合。它没被大量使用，主要缺点是 RNN 处理长序列很慢、极难并行，而视频恰恰很长。

![循环卷积网络（RCN）结构图，展示用2D卷积替代矩阵乘法处理时序特征的过程](/coursenotes/cs231n-deep-learning-for-cv-sp25/l10-video-understanding/kf_0014_00-50-23.jpg)

更适合的工具是**自注意力（self-attention）**：所有输入的对齐与注意力分数可完全并行计算。把自注意力搬到 3D：输入特征图 $C\times T\times H\times W$，用 $1\times1\times1$ 的 3D 卷积分别生成 query / key / value 特征图（$C'\times T\times H\times W$）；query 与 key 做转置后逐向量相乘，得到每对 query-key 的 attention score 与 attention map，用它对 values 加权，再用 $1\times1\times1$ 卷积映回 $C$ 并与原输入相加形成残差连接。这个可独立使用的模块叫 **non-local block**，可以反复插入 3D CNN 架构（3D CNN → non-local → 3D CNN → non-local …），每个块都能跨空间与时间融合信息。课件给出的自注意力一般形式为：

$$k=xW_k,\quad v=xW_v,\quad q=xW_q,\qquad e_{i,j}=\frac{q_i\cdot k_j}{\sqrt{D}},\quad a=\mathrm{softmax}(e),\quad y_j=\sum_i a_{ij}v_i$$

![自注意力机制（Self-Attention）详细流程图及计算公式推导](/coursenotes/cs231n-deep-learning-for-cv-sp25/l10-video-understanding/kf_0015_00-51-55.jpg)

## 8. 复用 2D 到 3D：I3D 膨胀、视频 Transformer 与模型可视化 [54:18 - 61:13]

既然已有大量成功的 2D 卷积网络，能否直接复用？**I3D**（Carreira and Zisserman, "Quo Vadis, Action Recognition?", CVPR 2017）的做法是把每个 $K_h\times K_w$ 的 2D 卷积/池化层替换为 $K_t\times K_h\times K_w$ 的 3D 版本（在 Inception block 上做 inflation），架构直接迁移。更进一步还能迁移**权重**：把一个 2D 卷积核沿时间复制 $K_t$ 次、再除以 $K_t$，用它初始化 3D 卷积核——这样输入单帧或一段恒定帧的视频时，输出与原 2D 网络一致，等于用 ImageNet 预训练权重给视频模型一个好初始化，之后在视频数据上微调即可。效果优于 two-stream，而且外观流与运动流都可以 inflate，进一步带来提升。

![将2D网络扩展为3D (I3D) 的原理说明，包含卷积核复制与初始化的示意图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l10-video-understanding/kf_0016_00-56-25.jpg)

视频 Transformer 也有多条改进线：**分解注意力**（factorized attention，分别对空间与时间做注意力）、**池化模块**（减少 token 数量）、以及**视频掩码自编码器**（video masked autoencoders，做高效可扩展的视频预训练）。讲者点名可课后查阅 ViViT、MViTv2、VideoMAE V2 等工作。在 **Kinetics-400** 上，从单帧模型的 62.2% 到 video masked autoencoder 已约 **90%**，视频分类做得相当好。

![视频Transformer的三种改进方向：分解注意力、池化模块与视频掩码自编码器架构对比图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l10-video-understanding/kf_0017_00-58-26.jpg)

和图像模型一样，也能**可视化**视频模型学到了什么。以 two-stream 为例：随机初始化 appearance image 与 flow image，前向得到某个类别的分数，把分数对输入反传并做梯度上升，最大化该分类分数；flow stream 还加时间约束，以避免时间流变化过快，从而分别捕捉慢动作与快动作。优化出的流图像让人能猜出动作（讲者现场互动里涉及 weight lifting 一类，画面呈现杠铃摆动、过顶推举等motion），说明视频模型确实学到了运动信息。

## 9. 超越片段分类：时序动作定位、视听多模态与前沿方向 [61:13 - 68:44]

到目前为止都是**短 clip 级分类**。更重要的另一类任务是 **Temporal Action Localization（时序动作定位）**：给定一段很长的未裁剪视频，不仅要知道有哪些动作，还要定位每个动作发生在哪些帧（例如先 running 后 jumping）。可以复用类似 **Faster R-CNN** 的架构——先生成 temporal proposals（时间提议）再分类（Chao et al., "Rethinking the Faster R-CNN Architecture for Temporal Action Localization", CVPR 2018）。进一步还有 **spatio-temporal detection**：在空间与时间两个维度上同时定位动作。

![时序动作定位（Temporal Action Localization）定义及基于 Faster R-CNN 的解决思路](/coursenotes/cs231n-deep-learning-for-cv-sp25/l10-video-understanding/kf_0018_01-01-35.jpg)

回到开场的例子，此前一直忽略了一个维度：**声音**。视频里还有音频，缺了它就少了情绪、少了可感知的交互。由此衍生出许多音视频任务，讲者举了自己做过的 **visually-guided audio source separation**：用视觉信息引导，把混合音频分离成各个声源分量。语音场景里可以分离每个说话人的声音；乐器场景里，用 **10 万段无标注多源视频**训练后，对新视频做音频分离，画面检测出 violin 与 flute。此外还有基于 Transformer/注意力的音视频理解：把图像切成 patches、把音频频谱图也切成 patches 一起送入；或用 MAE 风格同时预测图像 patch 与 spectrogram patch。

![乐器源分离示例：用10万无标注多源视频训练后对新视频做音频分离，检测出小提琴与长笛](/coursenotes/cs231n-deep-learning-for-cv-sp25/l10-video-understanding/kf_0019_01-04-37.jpg)

**高效视频理解**则是另一条线：长视频不可能逐 clip 处理，需要提升单个 clip 的效率、或只挑重要的 clip。课件汇总四篇工作：**MoViNets**（Kondratyuk et al., CVPR 2021）、**SCSampler**（Korbar et al., ICCV 2019，预测哪些 clip 最显著，只在这些 clip 上跑分类器再聚合）、**X3D**（Feichtenhofer, CVPR 2020，构建更高效的 3D CNN）、**AdaMML**（Pandal et al., ICCV 2021，用策略学习选择用哪些模态、多少个 clip、是否用音频）。还可以把音频当作 preview 机制，先预测重要时刻在哪，再据此处理 clip 并平均结果。

![高效视频理解四篇工作架构图：MoViNets、SCSampler、X3D、AdaMML](/coursenotes/cs231n-deep-learning-for-cv-sp25/l10-video-understanding/kf_0020_01-06-27.jpg)

最后的前沿：VR/AR 与智能眼镜带来 **egocentric 视频流**与多麦克风多通道音频，研究「谁在对谁说话、谁在听谁」这类社会交互；以及把视频 tokenize 后映射到 LM embedding space 的**视频级基础模型**，可以直接用提示词询问「人在哪里、在做什么」并输出文字描述——把视频理解与大语言模型连起来，正是当下的热点。

## 要点回顾

- 视频 = 2D 图像 + 时间，是 $3\times T\times H\times W$ 的 4D 张量；视频分类重心从物体/场景转向动作，难点是「如何抽取视频特征」。
- 数据规模是硬约束：标清约 1.5 GB/分钟、$1920\times1080$ 约 10 GB/分钟；解法是时空降采样（如 5 fps、$112\times112$ 约 588 KB）+ 在低 FPS 短 clip 上训练 + 推理时平均多个 clip 预测。
- 单帧 2D CNN 平均各帧概率常是很强的 baseline，设计任何视频模型都应先跑；如何聪明地采样帧本身是研究问题。
- Late Fusion（拼接+MLP 参数爆炸 / 平均池化丢信息）在很晚的阶段才融合，低层运动（如跑者双脚）已被 2D CNN 丢掉；Early Fusion 把 $3T\times H\times W$ 在第一层折叠成 $D$，但单层时间处理往往不够。
- 3D CNN（slow fusion）用 3D 卷积/池化让每层保持 $D\times T\times H\times W$，空间与时间感受野同步增长；相比Early Fusion 贯穿全时间轴，它在局部时间窗滑窗，获得**时间平移不变性**且更省核。
- Sports-1M（100 万视频 / 487 类）Top-5：单帧 77.7、Early 76.8、Late 78.7、3D CNN 80.2；C3D（3×3×3 conv、2×2×2 pool，VGG of 3D CNNs）39.5 GFLOP = 2.9× VGG-16。
- 光流 $F(x,y)=(dx,dy)$ 显式度量运动，two-stream 用外观流 + 光流运动流分别分类再聚合；UCF101 上 motion-only 反而优于 spatial-only。
- 长程时序：CNN 特征 + RNN/LSTM（many-to-one / many-to-many）、把矩阵乘换成 2D 卷积的 RCN、以及高度并行的自注意力 / non-local block（$e_{i,j}=q_i\cdot k_j/\sqrt D$）。
- I3D 把 2D 层膨胀为 $K_t\times K_h\times K_w$、把 2D 权重复制 $K_t$ 次并除以 $K_t$ 来初始化；视频 Transformer（分解注意力、池化、视频 MAE）把 Kinetics-400 从 62.2% 推到约 90%。
- 超出分类：时序动作定位（temporal proposals，类 Faster R-CNN）、视觉引导的音源分离、高效视频理解（MoViNets / SCSampler / X3D / AdaMML），以及 egocentric 多模态与视频级基础模型。

## 视频来源

https://www.youtube.com/watch?v=wElqklprhPE
