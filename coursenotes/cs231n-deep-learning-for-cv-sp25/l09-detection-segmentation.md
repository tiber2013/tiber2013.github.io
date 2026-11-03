---
title: "第 9 讲 · 目标检测、图像分割与可视化理解"
description: "CS231n Spring 2025 L09 视频笔记；对照 2026 春季课件补充新增内容。"
---

# 第 9 讲 · 目标检测、图像分割与可视化理解

> **Stanford CS231n 计算机视觉 · L09** ｜ 讲者：Stanford CS231n 课程团队 ｜ 74 分钟 ｜ [原视频](https://www.youtube.com/watch?v=PTypu6GqEd4)

> 课件版本：2026 春季 ｜ 视频版本：2025 春季。2026 课件新增或调整的内容集中在文末「2026 课件更新」一节。

## 核心结论

这一讲把视觉任务从“一张图一个标签”推进到“每个像素、每个物体都要有输出”：语义分割给每个像素分类，目标检测给每个物体框和类别，实例分割再给每个实例一张 mask。方法主线是同一个矛盾：既要扩大感受野、又要保住空间分辨率，还要控制一次前向的成本。FCN 用下采样加可学习上采样换效率，R-CNN 系列用区域提案换精度，YOLO/SSD 与 DETR 用单阶段、集合预测换速度。后半段转向可解释性：saliency map 看输入梯度，CAM 看最后卷积特征的加权和，Grad-CAM 把权重换成梯度，ViT 则天然提供注意力图。这些任务今天大多一行代码就能调用，但理解它们的结构，仍然是设计和排查视觉系统的前提。

## 回顾：ViT 如何把图像读成 token 序列 [00:05 - 08:30]

上一讲把序列模型从 RNN 推到 transformer：编码器由多层 multi-head self-attention、layer norm 和 MLP 堆成，解码器结构类似。三种处理序列的方式各有代价：RNN 对长度 $N$ 的序列是 $O(N)$ 计算，但不能并行，长程依赖要靠堆层；卷积在 $N$ 维网格上可并行，感受野随层数增长；self-attention 的每个输出直接依赖所有输入，主体只是四次矩阵乘法，高度并行，代价是 $O(N^2)$ 计算。视频的判断很直接：更贵，但在多数任务上效果更好。

![RNN、卷积与自注意力处理序列的三种方式对比图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0002_00-01-50.jpg)

ViT 的做法是把图像切成 patch 当作 token。以 224x224x3 的输入、16x16 的 patch 为例，得到 N 个 3x16x16 的块，拉平成长度 768 的向量，再线性投影到 D 维；因为切块打散了二维位置，每个位置还要加一个可学习的 **positional embedding**，可以是一维序号，也可以是 x、y 两个坐标。之后就是标准的 transformer encoder，输出每个 patch 对应一个 D 维向量。

![Transformer编码器架构及单层结构分解图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0001_00-00-34.jpg)

![Vision Transformers (ViT) 架构流程图，展示图像分块、线性投影及分类 token 的输入过程](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0004_00-06-09.jpg)

分类头有两种写法。第一种是在输入序列里额外加一个可学习的 **class token**，它和其他 token 一起过注意力层，最后把它的输出线性投影成 C 维类别分数；第二种是不加 token，把所有 patch 的输出做 average pooling 得到 1xD，再过一个 D 到 C 的线性层。视频特别说明：class token 是一个独立训练的参数，不是 bias；它能代表整张图，是因为每一层 self-attention 都在 patch 与它之间交换信息。图像不需要 causal mask，因为整张图本来就全部可见。训练仍然是最朴素的一套：定义 softmax 交叉熵损失，反向传播。

## Transformer 的四项工程改动 [08:30 - 16:00]

上一讲的架构是 2017 年的原样，但现代模型普遍做了几处改动，目的都是让训练更稳、容量更大。

**Pre-Norm** 针对 layer norm 的位置。原始结构把 layer norm 放在残差相加之后，意味着残差输出立刻被重新标准化，网络永远无法原样传递 $x$，也就学不出恒等映射。解决办法是把 layer norm 移到 self-attention 和 MLP 之前、残差连接内部，归一化还在，但恒等路径保住了。

**RMSNorm** 换掉 layer norm 本身：不减均值，只除以均方根，

$$
y_i=\frac{x_i}{\mathrm{RMS}(x)}\gamma_i,\qquad
\mathrm{RMS}(x)=\sqrt{\varepsilon+\frac{1}{N}\sum_{i=1}^{N}x_i^2}
$$

省掉均值计算，训练略稳一些。

**SwiGLU** 换掉 MLP 的非线性。经典 MLP 是两个权重矩阵 $Y=\sigma(XW_1)W_2$，SwiGLU 用三个矩阵做门控：

$$
Y=\big(\sigma(XW_1)\odot XW_2\big)W_3
$$

一个分支做非线性、一个分支做门，逐元素相乘。把隐层宽度取 $H=8D/3$ 时总参数量与经典 MLP 持平，但表达的非线性更强。

**Mixture of Experts（MoE）** 把每个 block 的 MLP 复制成 E 份，每份是一个 expert，再由 router 把每个 token 只送给其中 A 个（$A<E$）。参数量按 E 增长，计算量只按 A 增长，于是能在不显著增加 FLOPs 的前提下扩大容量。视频里同学问 expert 是否有可解释的分工，讲者说它们并行训练、初始化不同，往往会各自偏向某类模式，但并没有强制的语义划分；E 和 A 都是超参数，通常预先设定，而不是靠微调去搜。

![Pre-Norm Transformer 架构图，展示 Layer Normalization 位于残差连接外部](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0005_00-15-34.jpg)

## 从分类到分割：四种视觉任务 [16:00 - 18:30]

幻灯片用一张图区分四个任务：分类只输出标签，没有空间范围；语义分割给每个像素一个类别，不区分实例；目标检测给每个物体一个框和类别；实例分割再给每个实例一张 mask。讲者提醒，这些任务是过去十几年逐步搭起来的基础，理解结构才知道新问题该从哪里改。可视化在医疗等场景甚至比分类本身更重要——医生要知道“在哪里、为什么”。

## 语义分割：逐像素分类与它的代价 [18:30 - 25:10]

最直接的想法是逐像素分类：取以该像素为中心的 patch，送进一个 CNN 输出中心像素的类别。问题有两个。单个像素没有上下文，只看一个像素根本无法判断它属于什么；而且为每个像素跑一次网络，相邻 patch 的特征完全不复用，慢到不可用。

**Fully convolutional network（FCN）** 的思路是一次输出整张标签图：输入 3xHxW，经过若干卷积得到 DxHxW 特征，再得到 CxHxW 的分数，对每个位置取 argmax 得到 HxW 的预测。要这样做，网络里不能有全连接层，最初的设计也不做下采样。

![全卷积语义分割网络结构示意图，展示从输入图像到像素级预测的卷积流程](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0007_00-20-29.jpg)

但全分辨率卷积非常贵：HxW 那么大，通道一多，参数量和计算量都难以承受。实际结构改成先下采样再上采样：用 pooling 或 strided convolution 把分辨率降到 H/2、H/4，同时把通道做厚；最低分辨率处语义最强，然后一步步上采样回 HxW。

![全卷积语义分割网络结构图，展示下采样与上采样过程及分辨率变化](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0008_00-24-34.jpg)

训练目标很自然：每个像素都是一次分类，损失就是所有像素 softmax 交叉熵之和（或平均）：

$$
L=\frac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W}
-\log\frac{e^{s_{y_{h,w}}}}{\sum_{c}e^{s_c}}
$$

监督需要逐像素的 ground-truth 标签图。视频里同学回答“对每个像素取平均分类损失”，讲者确认这就是全部；早期这些 mask 是人工标注的，现在有工具辅助。

## 上采样：从 unpooling 到转置卷积 [25:10 - 28:55]

下采样有现成算子，上采样没有，需要新发明。

- **Nearest neighbor unpooling**：把 2x2 的每个值复制成 2x2 块，得到 4x4。
- **Bed of nails**：只在对应位置放原值，其余填零；后续卷积会把空洞逐渐填起来。
- **Max unpooling**：下采样用 max pooling 时记下每个最大值的位置，上采样时把值放回这些位置，其余为零。它要求网络里下采样和上采样层成对出现。

![网络内部上采样机制：Max Unpooling（最大池化逆运算）原理图解](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0009_00-26-37.jpg)

这些操作都没有参数。**Transposed convolution** 则是可学习的上采样：把输入看成给 filter 的权重，输入的一个值乘上整个 filter 写进输出的一块区域；stride 决定 filter 在输出上每次移动几格，所以 stride=2 时输出是输入的两倍。多个输入写出的区域会重叠，重叠处相加。1D 例子最清楚：输入只有 a、b 两个值，学一个 filter，把 a、b 分别乘以 filter 放到输出上，重叠的位置把两份贡献加起来。反过来看，strided convolution 就是可学习的下采样，转置卷积是它的反向操作，filter 同样由训练得到。

![转置卷积（Transposed Convolution）原理示意图，展示上采样过程](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0012_00-31-30.jpg)

## U-Net：把编码器的细节接到解码器 [28:55 - 31:20]

把下采样—上采样结构画成对称的 U 形，就是 **U-Net**。它在每个分辨率上都有编码器和解码器，并且把编码器对应层的特征图 copy and crop 到解码器，与上采样后的特征拼接。原因是下采样会丢掉空间细节，只靠上采样恢复，物体边界容易发糊；**skip connection** 把高分辨率的结构信息直接送到解码端，输出边界更锐利。视频强调，在不想用 foundation model 的医学分割里，U-Net 及其变体至今仍是很强的方案。

![U-Net网络架构图，展示编码器、解码器及跳跃连接结构](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0010_00-30-47.jpg)

## 从语义分割到实例分割 [32:45 - 33:45]

语义分割只输出像素类别，同一类别的两个个体无法区分：图里有两只狗，模型只知道“这是狗”，不知道哪些像素属于哪一只。要回答这个问题，必须先找到“有哪几个物体”，也就是目标检测。

![语义分割定义幻灯片，展示猫和牛图像及其像素级类别标注结果](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0013_00-32-51.jpg)

## 单目标检测：分类加定位 [33:45 - 36:25]

如果图中只有一个物体，检测就退化成“分类 + 定位”：网络主干输出一个 4096 维向量，接两个头，一个全连接层输出 C 类分数，另一个输出 4 个框坐标 $(x,y,w,h)$。损失是两项相加：

$$
L=L_{\text{cls}}(s,y)+\lambda L_{\text{box}}\big((x,y,w,h),(\hat x,\hat y,\hat w,\hat h)\big)
$$

分类用 softmax，框用 L2 回归。这是一个 **multitask loss**，两个任务共享特征、各自监督。

![单目标检测网络结构图，展示CNN特征提取后通过全连接层输出类别分数和边界框坐标](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0014_00-34-37.jpg)

问题在物体数量不固定：一个物体输出 4 个数，三个物体 12 个，输出维度会随图像变化，网络无法这样定义。所以检测必须换一种参数化方式。

## 滑窗、区域提案与 R-CNN [36:25 - 39:25]

一个自然的补救是滑窗：在图上滑动各种位置、大小、长宽比的框，把每个 crop 送进 CNN 判断“是物体还是背景、是哪一类”。这样输出维度固定了，但候选框的组合数量爆炸，计算上不可行。

于是有了 **region proposal**：先用便宜的方法找出“像物体”的区域。候选框的数量是预先设定的超参数，后面按分数取 top-k。

R-CNN 的流程是：对每个候选框，把图像区域 warp 成 224x224，送进 ImageNet 预训练的 ConvNet 提特征，用 SVM 分类，再回归 4 个修正量 $(dx,dy,dw,dh)$ 把框调准。它在 2014 年是很强的结果，但每个候选框都要完整跑一次网络，非常慢。

![R-CNN 章节标题页，展示输入图像示例及 Girshick 2014 论文引用](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0015_00-38-14.jpg)

## Fast R-CNN 与区域提案网络 [39:25 - 43:45]

卷积有一个好处：它是空间一致的，特征图上的位置能对应回原图。所以没必要对每个 crop 单独跑网络，只要对整张图跑一次卷积得到特征图，再把候选框映射到特征图上，对每个 RoI 取特征、过一个小网络，同时输出类别和框修正。这就是 **Fast R-CNN**：主干（AlexNet、VGG、ResNet 等）只跑一次，候选区域共享特征。

剩下的问题是候选框仍来自外部算法。**Region proposal network（RPN）** 把提案也学出来：在特征图的每个位置放 K 个不同大小、长宽比的 **anchor box**，对每个 anchor 预测“有没有物体”的二分类分数和 4 个框修正；把所有位置的 anchor 按 objectness 排序，取 top-k 作为提案。RPN 与分类头可以共享卷积特征，这就是 Faster R-CNN 的两阶段结构。

两阶段方法精度好，但代价是要跑两个网络：先提案，再分类回归，每张图至少两遍。R-CNN 系列今天在真实系统里已经很少用，视频的建议仍是把细节弄懂，因为后面的方法都是在修它的毛病。

## 单阶段检测：YOLO 与 SSD [43:45 - 49:00]

单阶段检测器把两步压成一步。**YOLO** 把图像划成 SxS 的网格（例子里是 7x7），每个格子给出 B 个 base box，每个 box 输出 4 个偏移加 1 个 confidence，再输出 C 个类别分数。整张图的输出就是一个 $7\times7\times(5B+C)$ 的张量，一次前向同时得到所有框和类别，因此每个格子都必须学会“自己负责哪个物体”。

推理时会有大量重叠框，先用阈值过滤低置信度，再用 **non-maximum suppression（NMS）** 去掉同一物体的重复框。视频没有展开 NMS 细节，只提到用阈值和它挑出高分框。YOLO 系列速度快，在工业检测、机器人、医学等场景里被广泛用作基础检测器。

## DETR：把检测写成集合预测 [49:00 - 57:50]

**DETR** 用 transformer 重写了检测流程：CNN 主干把图像变成一组特征 token，加上 positional encoding 送进 transformer encoder；decoder 的输入不是图像 patch，而是一组可学习的 **object query**。每个 query 通过 self-attention 相互交互、通过 cross-attention 读取 encoder 输出，最后经 FFN 输出一个类别和一个框，或者输出“没有物体”。query 的数量就是模型能检测的最大物体数，想检测更多物体就加 query。

它没有 anchor，也没有手工设计的候选框和框变换，输出直接是一个集合。视频里同学追问了几个关键点：query 是训练出来的参数，不是某个图像区域；模型不保证第几个 query 对应哪个物体，但多层 self/cross-attention 会让 query 之间互相协调、各自认领不同目标；训练只需要类别标签和框，不需要像素级分割（有 mask 也可以转成框）；监督仍然来自类别分数和框的回归。因为是完全监督，遇到训练集里没有的新类别无法检测，零样本检测是另一类问题。讲者对 DETR 的评价很直接：2020 年的论文，今天真实应用里已经很少用，但它是理解 transformer 检测的干净例子。

## Mask R-CNN：给每个实例一张 mask [58:00 - 61:40]

实例分割在检测的基础上再加一个分支。**Mask R-CNN** 沿用 Faster R-CNN 的两阶段结构：RPN 给候选框，Fast R-CNN 的头输出类别和框修正，另有一个小卷积分支对每个 RoI 输出该实例的像素级 mask。mask 与输入区域对应，同一类别的两只狗会得到两张不同的 mask，这正是语义分割做不到的。视频展示的椅子、床、婴儿等例子说明，mask 分支能在每个框内给出贴合物体形状的结果。代价是又多了一个任务、多了一组计算，这也是 R-CNN 系列偏重的原因。检测部分收尾时，视频还提到，检测与分类在工业质检（如分拣好坏番茄）中仍必需，也有大量开源实现可直接调用。

## 可视化：从模板到第一层 filter [61:40 - 64:20]

线性分类器那一讲有一个“视觉视角”：W 的每一行是一个类别模板，把行向量还原成图像，可以看到车的模板像一辆正面的车。神经网络也可以这样看，但通常只能看第一层：第一层卷积的 filter 是 3 通道，可以直接当 RGB 图显示，能看出网络学到了边缘、方向、颜色块等基本模式。中间层通道太多，无法直接可视化，需要梯度类方法；视频顺带提到 guided backprop 是其中一种。

## Saliency map：哪些像素影响分数 [64:20 - 67:10]

**Saliency map** 回答“改哪个像素会改变类别分数”。训练时求损失对权重的梯度，这里反过来，求类别分数对输入像素的梯度；2026 课件把它写成取绝对值、再对 RGB 通道取 max：

$$
\text{saliency}(h,w)=\max_{ch}\left|\frac{\partial S_c}{\partial x_{h,w,ch}}\right|
$$

梯度绝对值越大，说明该像素的微小变化越能影响分数。把热图叠加在原图上，就能看到模型“看”的是狗还是背景。医疗场景最能说明它的价值：分类器说“有肿瘤”没有意义，医生要知道肿瘤在图像的哪个位置。视频也承认这类方法不完美，热图可能高亮无关区域，所以还有 CAM、Grad-CAM 等更强的方法。

## CAM：把分类权重投回特征图 [67:10 - 70:20]

**Class activation mapping（CAM）** 要求网络在最后卷积层之后接 global average pooling，再接一个线性分类层。设最后卷积特征为 $f\in\mathbb{R}^{H\times W\times K}$，GAP 得到

$$
F_k=\frac{1}{HW}\sum_{h,w}f_{h,w,k}
$$

分类权重为 $w\in\mathbb{R}^{K\times C}$，类别 c 的分数是

$$
S_c=\sum_k w_{k,c}F_k
=\frac{1}{HW}\sum_{h,w}\sum_k w_{k,c}f_{h,w,k}
$$

把求和顺序换一下，就得到每个空间位置的贡献：

$$
M_{c,h,w}=\sum_k w_{k,c}f_{h,w,k}
$$

这就是类别 c 的 CAM：第 k 个通道的特征图，用类别 c 在该通道上的权重加权求和。把它上采样回原图大小，就得到一张热图，高亮“哪些区域把分数推高”。视频展示的宫殿、圆顶、教堂、祭坛等类别，热图分别落在建筑的不同部位，说明模型确实学到了类别相关的区域。

CAM 的限制是结构性的：它依赖 GAP 加单个线性层，而且只能作用在最后一层卷积上；真实网络最后往往还有全连接、softmax 等操作，无法直接把分类权重乘回特征图。

## Grad-CAM：用梯度代替分类权重 [70:20 - 72:25]

**Grad-CAM** 把权重从“分类层参数”换成“梯度”。任取一层，激活为 $A\in\mathbb{R}^{H\times W\times K}$，先求类别分数对该层激活的梯度，再对空间维度做 global average pooling 得到每个通道的权重：

$$
\alpha_k=\frac{1}{HW}\sum_{h,w}\frac{\partial S_c}{\partial A_{h,w,k}}
$$

然后用它加权求和，并过 ReLU 只保留对类别有正贡献的区域：

$$
M^c_{h,w}=\mathrm{ReLU}\left(\sum_k\alpha_k A_{h,w,k}\right)
$$

这样任何一层都能生成热图，也不再要求网络以 GAP 结尾；代价是要多一次反向传播。视频把 CAM 与 Grad-CAM 的关系讲得很直白：CAM 是“权重 × 特征”，Grad-CAM 是“梯度 × 特征”，后者用梯度补上了前者拿不到的那些权重。幻灯片还把原始图、guided backprop、Grad-CAM、guided Grad-CAM、occlusion map 并排比较：guided backprop 更细但噪声多，Grad-CAM 更集中但分辨率粗。

## ViT 的注意力图 [72:25 - 73:30]

CNN 需要 CAM、Grad-CAM 这类事后方法，transformer 则自带可视化：上一讲的语言注意力矩阵，对每个输出 token 给出它在所有输入 token 上的权重；把输入 token 换回 patch，就得到像素空间的热图。想知道模型分类时看哪里，直接看 class token 或某个 patch 的注意力即可，不需要额外求梯度。这也是 transformer 在可解释性上的一个实际优势。

## 2026 课件更新

以上正文以 2025 春季视频为准。对照 2026 春季课件，主要有这些差异：

- **讲义标题与开场**：2026 课件题为 “Lecture 9: Detection, Segmentation, Visualization, and Understanding”，开头是 Assignment 2 notebook 修复通知、期中与项目里程碑截止提醒，以及 Student Spotlights；视频从上一讲回顾直接开始。
- **ViT 分类头的第二种写法被单列**：2026 用一整页展示“average pool NxD 到 1xD，再线性层 D=>C”，并写明 patch 拉平后是 768=>D、图像不做 masking；视频口述了两种分类头，没有给出这些尺寸。
- **MoE 落到具体模型**：2026 课件点名 GPT-4o、GPT-4.5、Claude 3.7、Gemini 2.5 Pro 等“几乎肯定使用 MoE、参数量超过 1T”；视频只说现代 LLM 都在用这类改动。
- **区域提案与 RPN 的具体数字**：2026 写明 selective search 一张图约 2000 个提案、CPU 上几秒，RPN 在 512x20x15 的特征图上每个点放 K 个 anchor、按 objectness 取 top ~300；视频只讲“预设数量、取分数最高的 top-k”。2026 也把 R-CNN 直接标成 “Slow R-CNN”。
- **单阶段检测扩为三家**：2026 的标题是 YOLO / SSD / RetinaNet，并写出输出张量 $7\times7\times(5B+C)$、示例 B=3；视频的例子是 B=2，只提到 YOLO 与 SSD。
- **DETR 的监督方式写得更明确**：2026 写 “no anchors, no regression of box transforms”，用 **bipartite matching** 把预测框与 GT 框匹配后再回归；视频只说明 query 加 FFN 输出，并用 GT 框的 L2 监督。
- **Mask R-CNN 细节与姿态分支**：2026 给出每个 RoI 预测 28x28 二值 mask、输出 Cx28x28，并新增 “Mask R-CNN also does pose”；视频只说加一个卷积分支生成像素级 mask。
- **Saliency 的后处理**：2026 明确“取绝对值并对 RGB 通道取 max”；视频只讲对像素求梯度。
- **ViT 可视化的文献**：2026 增加了 Chen et al. (ICLR 2022)、Paul & Chen (AAAI 2022) 的引用和带置信度的示例；视频只用一句话说 transformer 自带注意力图。

## 要点回顾

- **四种任务**：分类无空间范围；语义分割逐像素分类、不区分实例；目标检测给框和类别；实例分割给每个实例一张 mask。
- **ViT** 把图像切成 patch 当 token，加 positional embedding 后过标准 transformer；分类头用 class token 或 average pooling。
- **Pre-Norm、RMSNorm、SwiGLU、MoE** 是现代 transformer 的四项常见改动，分别解决恒等路径、归一化开销、MLP 表达力和容量/计算比。
- **语义分割** 的朴素做法是逐像素滑窗，既缺上下文又极慢；FCN 一次输出整张标签图，但全分辨率卷积太贵。
- **编码器—解码器** 用下采样降低分辨率、增厚通道，再上采样回原尺寸；损失是逐像素 softmax 交叉熵之和。
- **上采样** 可以无参数（nearest、bed of nails、max unpooling），也可以可学习（转置卷积，重叠处相加）。
- **U-Net** 用 skip connection 把编码器的高分辨率特征接到解码器，缓解边界模糊，至今仍是医学分割的强基线。
- **单目标检测** 是分类加框回归的多任务损失；物体数量不固定使这种参数化无法扩展。
- **R-CNN → Fast R-CNN → Faster R-CNN** 的主线是共享卷积、把候选框映射到特征图，再用 RPN 把提案也学出来。
- **YOLO/SSD** 用网格加 base box 一次前向输出所有框，配合阈值与 NMS；**DETR** 用 object query 和 transformer 直接回归一个集合。
- **Mask R-CNN** 在检测头上并行加 mask 分支，实现实例分割。
- **可解释性**：saliency 看输入梯度，CAM 看最后卷积特征的加权和，Grad-CAM 用梯度做权重、可作用于任意层，ViT 直接看注意力图。

## 视频来源

https://www.youtube.com/watch?v=PTypu6GqEd4
