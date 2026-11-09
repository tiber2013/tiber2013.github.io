---
title: "第 9 讲 · 目标检测、图像分割与可视化理解"
description: "Stanford CS231n Spring 2025 L09 中文视频笔记（2026-11 重写版）"
---

# CS231n Spring 2025 · Lecture 9 检测、分割与可视化：从 ViT 到 Grad-CAM

**核心结论**：本讲先补齐上一讲 Transformer 在视觉上的落地与工程优化——ViT 把图像切成 patch 当 token、靠位置编码补回 2D 结构、用 class token 或 pooling 做分类，而 Pre-Norm、RMS Norm、门控 MLP 与 MoE 让训练更稳、容量更大；随后系统梳理四类核心视觉任务，重点讲语义分割（滑动窗口 → 全卷积网络 → 上下采样 → U-Net 跳跃连接）与目标检测（多任务损失 → Region Proposals / R-CNN / Fast R-CNN / RPN → 单阶段 YOLO → 基于 Transformer 的 DETR），并以 Mask R-CNN 补上实例分割；最后转向可视化与理解，从线性分类器模板、第一层卷积核、梯度显著性，到 CAM / Grad-CAM 的热力图，并指出 Transformer 自带的注意力图让可视化比 CNN 容易得多。讲者反复强调：这些虽多是十到十五年前的经典 pipeline、今天多被更大规模模型取代，但理解其设计动机，仍是自己设计新模型、以及医学等需要「知道在哪里、为什么」的场景的基础。

## 1. 回顾与落地：Transformer 与 Vision Transformer (ViT) [00:05 - 09:24]

讲者先回顾上一讲：从 Seq2Seq / RNN 过渡到 transformer，编码器由多层 multi-head self-attention、layer norm 与 MLP 组成，负责编码序列；解码器以编码 token 为输入再生成输出。self-attention 比 RNN 与卷积建模更好、结果更佳，代价是更多计算与内存；此外也提过 cross attention。

![Transformer模型整体架构图，包含编码器与解码器详细结构](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0001_00-01-14.jpg)
这一页是上一讲 Transformer 的整体架构图，标注了 Encoder / Decoder、Multi-head self-attention、MLP、Layer norm、Positional encoding、Masked Multi-head self-attention，并展示「person wearing hat」这类输入输出序列的流向与残差连接位置。

接着进入 **Vision Transformer (ViT)**（Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021）：用 transformer 处理图像时，先把图切成 patches，每个 patch 重塑成向量再做线性投影，得到一个 D 维 token，相当于把图像变成一串「序列」。

![Vision Transformers (ViT) 章节标题页，展示输入图像示例及尺寸参数](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0002_00-03-00.jpg)
ViT 章节标题页，标注输入图像尺寸例如 224×224×3，并给出 ICLR 2021 的原始论文出处。

![Vision Transformers (ViT) 将图像切分为 patch 并线性投影后输入 Transformer 的流程图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0003_00-07-22.jpg)
流程图给出具体数值：输入 224×224×3 → 切成 16×16×3 的 patch → flatten 后做线性变换 768 ⇒ D，每个 patch 得到一个 D 维向量，作为 Transformer 的输入。

把图像切成 patch 后，我们**丢失了像素的 2D 位置**，因此必须补 **positional embedding**：可以是简单的序号 1、2、3…，也可以是 2D 的 X / Y 坐标，二者相加形成送进 transformer 的新 token。由于图像是整幅同时可见，这里**不需要像语言那样做 masking**（语言是真正的序列，不能用未来信息）。

分类有两种做法：① 额外加入一个与其它 token 同维、可学习的 **class token**，它在输出空间被转成 C 维类别概率向量；② 不加 class token，直接把全部输出向量做 pooling 再投影成 C 类概率。两者都用反向传播 + softmax / cross-entropy 损失监督。讲者特别澄清：class token 不是 bias，而是独立的可训练参数；多层 self-attention 会在所有 patch token 与 class token 之间反复融合信息，因此从这一端施加损失就能让它代表类别概率。ViT 不只用于分类，还可用于本讲后面的许多任务；现代架构大多沿用这套组件。

## 2. Transformer 的现代改进：Pre-Norm 与其它 tweaks [09:24 - 16:02]

这些年 ViT / Transformer 的主干基本没变，但有不少提升性能、稳定训练的 tweaks。第一个是**残差连接与 Layer Norm 的位置**：原设计把 Layer Norm 放在残差连接**外部**——残差分支加回来之后立刻被归一化，特征值被改动，网络再也无法复现 **identity function（恒等函数）**，而这恰是 ResNet 当初想保留的性质。

![Pre-Norm Transformer架构图，展示Layer Normalization位于残差连接外部的结构](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0005_00-15-37.jpg)
这张 Pre-Norm 前的架构图点出问题：「Layer normalization is outside the residual connections. Kind of weird, the model can't actually learn the identity function」。

解决办法是 **Pre-Norm**：把 Layer Norm 移到 self-attention 与 MLP **之前**、放进残差连接内部，归一化仍然保留，恒等路径也保住，训练更稳定（Baevski & Auli, arXiv 2018）。

![Pre-Norm Transformer 结构图，Layer Normalization 移至 Self-Attention 与 MLP 之前残差连接内部](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0004_00-09-56.jpg)
Pre-Norm 结构图写明「Move layer normalization before the Self-Attention and MLP, inside the residual connections. Training is more stable.」。讲者再次答疑：为什么外部归一化学不到恒等——因为残差相加后紧跟归一化，特征值必然被改变，所以要把归一化挪进去。

其余 tweaks 一笔带过：**RMS Norm**（root mean square normalization）对每个特征不用均值来归一化，更基础，经验上让训练更稳；这些选择多是经验上更优。**门控 MLP**：不用简单 MLP 的两个权重矩阵 $W_1, W_2$，而是再加第三个形成 gated nonlinearity（门控非线性），即便把隐藏层维度设为 $8d/3$，参数量保持不变，却能学到更高维的非线性。**Mixture of Experts (MoE)**：不是一组 MLP 而是多组，每组是一个 expert，用 router 把 token 路由到不同 experts，只激活少数 active experts，于是在参数量增加、模型更鲁棒的同时不显著增加计算，各 MLP 又天然并行——现代 LLM 普遍采用这类 tweaks。专家数量是预定义的超参数；各专家因初始化不同，往往各自学到数据的某个方面或某个模式，有助于在需覆盖多峰分布时把它们分开。

## 3. 语义分割：从滑动窗口到全卷积与上采样 [16:02 - 29:14]

讲者先交代今天要覆盖的核心任务（分类、语义分割、检测、实例分割）与可视化理解，并指出在医学场景里「可视化与理解往往比分类或检测肿瘤本身更重要——你要知道在哪里、为什么」。

### 3.1 语义分割与滑动窗口 [17:15 - 20:16]

**语义分割**要给**每个像素**一个标签，测试时输入图像、输出同尺寸的标签图。逐像素单独判断几乎不可能——只看一个像素没有 context，无法知道它属于什么物体，必须看周围区域。

![语义分割的滑动窗口方法示意图，展示从全图提取Patch并用CNN分类中心像素的过程及其低效问题](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0006_00-19-39.jpg)
滑动窗口示意图：Full image → Extract patch → Classify center pixel with CNN，输出 Cow / Cow / Grass；画面直接点出「Problem: Very inefficient! Not reusing shared features between overlapping patches」。

做法是以该像素为中心取一个 patch，用 CNN / ResNet / ViT 等任意分类网络给中心像素定标签；但每个像素都要跑一整个网络，极慢，且重叠 patch 之间的共享特征完全没复用，非常低效。

### 3.2 全卷积网络与逐像素损失 [20:16 - 25:12]

另一条路是让网络直接输入整图、输出整张分割图（标签矩阵）。这要求输入层与输出层都与图像同尺寸，中间不能用全连接层把特征「压扁」，网络要全程保持「膨胀」（inflated）——即 **Fully Convolutional Network（FCN）**。问题是图大、层大、参数多，早年没有强 GPU 时是训练瓶颈，于是架构演化为「从全尺寸下采样 → 中间低分辨率但通道很厚 → 再上采样回原尺寸输出」。

![全卷积语义分割网络结构图，展示下采样与上采样过程及特征图尺寸变化](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0007_00-24-37.jpg)
全卷积分割网络结构图：Downsampling 用 Pooling / strided convolution，Upsampling 尚标着「???」；尺寸依次为 Input 3×H×W → High-res D1×H/2×W/2 → Med-res D2×H/4×W/4 → Low-res D3×H/4×W/4 → Predictions H×W，并引用 FCN (CVPR 2015) 与 Deconvolution Network (ICCV 2015)。

训练损失很直接：每个像素都在做一次分类，把**所有像素的 softmax 损失求和**再反传即可。这类算法是**全监督**的，需要 ground truth 标签图，早年靠人工逐像素标注得来。

### 3.3 上采样：从 unpooling 到转置卷积 [25:12 - 29:14]

下采样早已熟悉（pooling、strided convolution），上采样却需要新操作来「逆转」它。无参数方式有三种：nearest neighbor（把低分辨率值复制到邻域）、bed of nails（只保留一个角、其余置 0，靠后续卷积逐步填满）、以及 **max unpooling**——编码端做 max pooling 时记下被选中的位置，解码时把值放回那些坐标。**可学习的上采样**则是**转置卷积（transposed convolution）**：用 1D 例子说明，输入两个值 A、B，学一个 filter 把每个值映射到高分辨率输出，重叠部分求和；等价地，对一个 3×3 矩阵不去卷积输入、而对输入的转置版本做卷积，从而产生更大的输出。它就是常规卷积的逆，且 filter 和其它卷积层一样是**可训练**的。

## 4. U-Net：跳跃连接保住空间细节 [29:14 - 32:45]

![U-Net网络架构图，包含编码器、解码器及跳跃连接（copy and crop）细节](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0008_00-29-29.jpg)
U-Net 架构图：input image tile → output segmentation map，标注 conv 3×3 ReLU、copy and crop、max pool 2×2、up-conv 2×2、conv 1×1（Ronneberger et al., 2015）。

![U-Net网络架构图，展示编码器、解码器及跳跃连接结构](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0009_00-30-47.jpg)
更细的 U-Net 图：输入 572×572 → 输出 388×388，逐层给出 64 / 128 / 256 / 512 / 1024 通道及具体尺寸与操作。

形状像 U：编码器一路下采样扩大感受野、同时丢失空间分辨率，解码器再上采样回到图像分辨率。它与 FCN 唯一但关键的区别是——下采样会丢分辨率，直接上采样容易让**边界模糊**，于是把编码器侧的**特征图 copy and crop 跳跃连接**到对应的解码器层，把结构信息保留下来，输出因此更锐利。讲者强调：直到今天，在不使用任何 foundation model 的前提下，U-Net 及其变体在医学图像分割等场景仍能达到 SOTA，是应用最广的分割网络之一。语义分割小结再点一次：转置卷积是常规卷积的逆，对输入的转置版本做卷积得到更大输出，filter 可训练。

## 5. 四类视觉任务与检测的多任务损失 [32:45 - 36:23]

语义分割只给像素标签，若图中有两个同类实例（两只狗），它分不清谁是谁 → 需要**实例分割** → 而要理解多目标，就得先做**目标检测**。

![分类、语义分割、目标检测与实例分割四种任务的对比示意图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0010_00-33-08.jpg)
四种任务对比：Classification（整图一个 CAT，No spatial extent）、Semantic Segmentation（逐像素 GRASS / TREE / SKY，No objects just pixels）、Object Detection（Multiple Object）、Instance Segmentation（区分 DOG, DOG, CAT 等实例）。

单目标检测既要输出类别分数，又要输出边界框坐标 $(x, y, h, w)$。做法是多任务损失：分类用 **softmax loss**，框坐标用 **L2 / 回归损失**，两者相加成 compound loss：

```python
loss = softmax_loss(class_scores, label) + L2_loss(box_pred, box_gt)  # 单目标检测：分类 + 框回归，box=(x,y,h,w)
loss_seg = sum(softmax_loss(pred[u, v], gt[u, v]) for (u, v) in all_pixels)  # 语义分割：逐像素 softmax 求和
```

但多目标时这套「把分类扩展成检测」不可扩展：三个物体就要输出 12 个数，物体一多数字爆炸，必须另寻思路。

## 6. Region Proposals、R-CNN 与 Faster 化 [36:23 - 44:33]

穷举滑动窗口（从 (0,0) 遍历所有 x, y, h, w 组合）因框的组合数太多而不可扩展。**Region Proposals（候选区域）**是 2014 年前后大量研究的成果：先找出「很可能含物体」的区域（约 2k 个），问题于是变简单。

![R-CNN算法介绍页，展示从输入图像提取感兴趣区域(ROI)的过程](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0011_00-38-17.jpg)
R-CNN 介绍页：Regions of Interest (RoI) from a proposal method (~2k)，出处 Girshick et al., "Rich feature hierarchies for accurate object detection and semantic segmentation", CVPR 2014。

**R-CNN**：对每个候选区域裁出 patch，跑 CNN 分类（猫 / 狗 / background），并**修正框坐标**——预测 4 个数 $(dx, dy, dw, dh)$ 作为对 RoI 的 corrections。

![Slow R-CNN 架构流程图，展示每个 RoI 独立通过 ConvNet 的低效性及改进思路](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0012_00-39-34.jpg)
Slow R-CNN 流程图点出「Problem: Very slow! Need to do ~2k independent forward passes for each image!」，并给出改进「Idea: Pass the image through convnet before cropping! Crop the conv feature instead!」。

R-CNN 极慢：每张图约 2k 个区域各跑一次完整 CNN 前向。改进依据是卷积操作保留空间信息、能追踪像素位置，于是**先对整图跑一次大卷积**得到特征图，再在特征图上裁对应区域、跑一个小 CNN 输出「框偏移」与「物体类别」——即 Fast 版本。**RPN（Region Proposal Network）**则补上「候选区域从哪来」：用一个网络从随机初始位置出发，经多层卷积逐步 refine 出高概率含物体的区域、同时 refine 框坐标，用物体标签与位置监督，实际取 **top-k** 高概率框作为该图的 proposals。R-CNN / Mask R-CNN 如今已不再使用，因为太重：至少要两个网络（RPN + 分类 / 框修正网络），每张图至少两次前向；但理解「怎么走到这一步」仍然重要。

## 7. 单阶段 YOLO 与基于 Transformer 的 DETR [44:33 - 57:46]

### 7.1 单阶段检测：YOLO [44:33 - 48:52]

两阶段之后出现**单阶段检测器**（SSD，以及最流行的 **YOLO**）。YOLO（Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection", 2015）至今仍是工业界检测的基础，快、效果好，早期版本是卷积重的网络。

![YOLO实时目标检测的整体流程示意图，展示SxS网格输入、边界框置信度与类别概率图到最终检测结果的转换](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0013_00-45-15.jpg)
YOLO 流程图：把输入分成 S×S grid → 输出 Bounding boxes + confidence → Class probability map → Final detections，标注 Redmon et al. 2015。

核心是 "You Only Look Once"：对图像**一次前向**就生成全部框。把图像分成 $S \times S$ 网格（例子 7×7），每个格子用一个全卷积网络同时输出：该位置有物体的概率、$B$ 个（超参数）边界框及其框修正、以及类别概率向量（$B=2$ 时每格给两个带不同概率的框）。所有格子并行计算、每个框带一个概率（图中用边的粗细表示）；再经**阈值化 + 非极大值抑制（non-maximal suppression, NMS）**挑出概率最高的框，得到最终检测。YOLO 有大量新版本，广泛用于医学、机器人与工业应用。

### 7.2 DETR：用 Transformer 直接出框 [48:52 - 57:46]

较新的思路是 **DETR**（Carion et al., "End-to-End Object Detection with Transformers", ECCV 2020）：纯 transformer 做检测，虽已近乎不在真实应用中使用，却是「如何把上一讲的 self / cross attention 用到检测」的极佳例子。

![DETR目标检测模型架构图，展示backbone、encoder、decoder及预测头结构](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0014_00-54-37.jpg)
DETR 架构图：backbone、encoder、decoder、prediction heads，画面标注了 bipartite matching loss。

流程：图像切 patches → 过 CNN 生成 token → 加位置编码 → 送入 transformer encoder（多层 self-attention + 归一化 + MLP）得到输出 tokens。关键在 decoder：除接收 encoder 输出，还定义一组**可训练的 object queries**（learnable parameters）；放几个 query（5、10、20）就表示「最多检测几个物体」，decoder 内先 self-attention、再与 encoder 输出做 cross-attention，为每个 query 生成输出，经 **FFN** 产生类别标签 + 边界框，或直接输出「no object」。

![DETR目标检测模型架构图，展示CNN骨干、Transformer编解码器及二部图匹配损失机制](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0015_00-57-32.jpg)
DETR 完整数据流图：CNN backbone → Transformer encoder / decoder → prediction heads，FFN 预测头，以及 bipartite matching loss。

输入的**不是**任何候选框，queries 只是可训练参数；监督来自预定义的类别概率向量与 ground truth 框的 L2 loss，并不告诉 query 去哪找什么，全靠反向传播自学，课件把这一机制标为 **bipartite matching loss（二部图匹配损失）**。讲者的几处答疑：多层 self / cross attention 让各 query 相互交互、不会都输出同样的东西；算法**不需要**像素级分割标注，只用类别标签与框即可（若有像素级分割也能转成框）；全监督网络通常无法泛化到训练集中没见过的新类（zero-shot 属另一类工作）；若场景中物体数超过 query 数，它只会给出置信度最高的那些，因此通常要多放一些 query。

## 8. 实例分割：Mask R-CNN 与任务收尾 [57:46 - 61:42]

回到开篇的问题：如何用这些检测算法做实例分割？在 R-CNN 这套「图像 → region proposals → CNN → 类别标签 + 框坐标」架构上**再加一个输出分支**即可：多一层卷积为每个物体生成**像素级 mask**，使其变成更多任务的多任务学习——这就是 **Mask R-CNN**。mask 与输入图像同尺寸，用 FCN 得到；对每个候选小框都能给出对应 mask（椅子、床、婴儿等不同实例各有各的掩码）。讲者说 Mask R-CNN 在检测各类已知物体上结果非常好，且有大量 API 与开源实现。

工业意义上，例如流水线质量控制中把烂番茄与好番茄分开，需要先检测物体再分类；所以即便今天计算机视觉已远超这些任务、有了更大规模模型，理解这些 pipeline 与「如何实时做」依然重要。

## 9. 可视化与理解：从滤波器到 CAM / Grad-CAM [61:42 - 73:31]

讲者指出，可视化神经网络学到什么是 2010 年代到 2020 年代的热门课题，能帮我们理解网络内部；在医学等应用里「知道肿瘤在图像的哪里、为什么」往往比「有没有肿瘤」更关键。

### 9.1 线性分类器模板与第一层卷积核 [61:42 - 64:19]

回到最初的线性分类器：学出来的权重可看作每个类别的 **template**（如正面朝前的汽车模板）。

![线性分类器的视觉解释：左侧为ImageNet类别样本网格，右侧展示输入图像经权重矩阵W和偏置b计算得分的流程图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0016_01-02-40.jpg)
线性分类器的视觉解释：左侧 ImageNet 十类样本网格（airplane … truck），右侧输入图像经权重 W、偏置 b 算得分（Score −96.8 / 437.9 / 61.95）。

同理可把网络第一层 **filters 的权重**可视化：能看到学到的基本形状、朝向、边缘等模式。但这种直接可视化只适用于通道少的早期层（如 3 通道可拼成 RGB 图），CNN 中间层通道太多难以直接看；更深层学到的模式更整体、更大，也可用 **guided backpropagation** 可视化但没这么简单。

![第一层卷积核可视化：左侧展示AlexNet的64个11x11滤波器图案，右侧为网络结构图并高亮显示第一层](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0017_01-03-06.jpg)
AlexNet 第一层 64×3×11×11 滤波器图案与网络结构高亮，引用 Krizhevsky et al. (arXiv 2014)、ResNet (He et al., CVPR 2016)、DenseNet (Huang et al., CVPR 2017)。

### 9.2 梯度显著性 Saliency [64:19 - 67:13]

**Saliency（显著性）**回答「哪个像素重要」。训练时我们对 loss / class score 关于**网络权重**求导来更新权重；现在改为关于**像素值**求导并可视化这些梯度，得到的就是「改动哪些像素会显著改变 dog 分数」的像素集合——正是梯度定义的直接运用。对不同类别分别做，就能得到各自的显著图。

### 9.3 CAM：把类别得分反推回特征图 [67:13 - 70:18]

有时不只关心逐像素梯度，还想看**每个类别的 activations** 如何作用，于是有 **CAM（Class Activation Mapping）**。

![Class Activation Mapping (CAM) 的流程图，展示从 CNN 特征到全局平均池化再到全连接层权重计算类别得分的过程](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0018_01-07-16.jpg)
CAM 流程图：CNN 特征 → Global Average Pooling → 全连接层权重 → 类别得分，维度标注 $f \in \mathbb{R}^{H\times W\times K}$、$F \in \mathbb{R}^K$、$w \in \mathbb{R}^{K\times C}$、$S \in \mathbb{R}^C$。

思路：卷积层经 pooling 得到 feature maps，feature maps 再经学到的权重变成各类得分；把类别得分写成特征图的加权和，就能沿卷积一直保留的图像空间对应关系，把预测反推回具体空间位置，用「权重 × 特征值」得到 class activations，生成每个类别的热图。

![CAM 热力图可视化示例，左侧展示不同类别预测的激活区域，右侧展示同一物体（杠铃）在不同场景下的定位效果](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0019_01-09-30.jpg)
CAM 热力图示例：dome、palace、church、altar、monastery 各自的激活区域，以及同一物体（barbell）在不同场景下的定位。

**局限**：这套计算只能用在**最后一个卷积层**，而多数 CNN 在末端还有全连接等操作，中间隔着东西就无法这样回推。

### 9.4 Grad-CAM 与 Transformer 自带的注意力 [70:18 - 73:31]

**Grad-CAM**（Gradient-weighted Class Activation Mapping, Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", CVPR 2017）解决上述局限：算法基本相同，但改用**梯度**来算权重——把梯度一路回传到目标层，将权重与梯度聚合起来替代原来那个 W 权重，再加权 feature maps，并用 **ReLU** 只保留正值，从而在图像空间生成热力图，可作用于任意卷积层，比 Guided Backprop 定位更平滑、更准确。

![Grad-CAM 与其他可视化方法（Guided Backprop, Occlusion map）在猫狗图像上的对比热力图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l09-detection-segmentation/kf_0020_01-12-22.jpg)
Grad-CAM 与 Guided Backprop、Occlusion map 在猫狗图像上的对比热力图，出自 Selvaraju et al., CVPR 2017。

讲者最后点回 transformer：CNN 才需要 Grad-CAM 这类方法，而 **ViT / Transformer 天生自带激活图**——上一讲那个「每个输出词对输入有一组注意力权重」的矩阵，对像素同样成立，每个输出都能在像素空间画出 attention map、用于可视化 ViT 特征，因此 transformer 的可视化容易得多。下一讲进入视频理解（video understanding）。

## 要点回顾

- 回顾：Transformer 由编码器 / 解码器的 multi-head self-attention + Layer Norm + MLP 组成，self-attention 建模优于 RNN / 卷积但更耗算力与内存；ViT 把图像切成 patch 当 token。
- ViT（Dosovitskiy et al., ICLR 2021）：224×224×3 输入切成 16×16×3 patch，flatten 后线性投影 768 ⇒ D；因丢失 2D 位置需加 positional embedding（序号或 X/Y 坐标），图像可全见故无需 masking；分类可用可学习 class token 或对输出做 pooling + 投影，交叉熵 + 反传监督。
- 现代 tweaks：Layer Norm 在残差连接外部会破坏恒等函数 → **Pre-Norm** 把它移到 self-attention / MLP 之前、残差内部（Baevski & Auli 2018）使训练更稳；另有 RMS Norm、门控 MLP（三个权重矩阵、隐藏维度 8d/3）、Mixture of Experts（router 路由、稀疏激活、参数增而计算不增、专家数为超参数）。
- 语义分割给每个像素标签：滑动窗口逐像素取 patch 分类极慢且不复用重叠特征；FCN 让网络输入输出同尺寸、全程保持膨胀，逐像素 softmax 损失求和 + 全监督 ground truth（早年人工标注）。
- 上采样：nearest neighbor、bed of nails、max unpooling（复用编码端记录的 max 位置），以及可训练的**转置卷积**（对输入转置版本做卷积、重叠求和，是常规卷积的逆）。
- U-Net（Ronneberger et al., 2015）：编码器下采样、解码器上采样，靠 **copy and crop 跳跃连接**把编码器特征图送回解码器以保住边界锐度；572×572 → 388×388、64~1024 通道，至今医学分割仍常用。
- 四类任务：Classification、Semantic Segmentation、Object Detection、Instance Segmentation；单目标检测用 softmax 分类损失 + L2 框回归 $(x,y,h,w)$ 组成多任务损失，多目标则需其它思路。
- 检测演进：Region Proposals (~2k) → R-CNN（逐 RoI 跑 CNN + 预测 $(dx,dy,dw,dh)$ 框修正，CVPR 2014）→ Fast R-CNN（先整图卷积再在特征图上裁剪）→ RPN（学出候选区域、取 top-k）→ 因两阶段太重转向单阶段 YOLO（S×S 网格、B 个框 + 类别概率、NMS，Redmon et al. 2015）。
- DETR（Carion et al., ECCV 2020）：CNN backbone → transformer encoder → 带可训练 object queries 的 decoder（self + cross attention）→ FFN 出类别与框，用类别概率 + 框 L2 loss 与 bipartite matching loss 监督，query 数即最多检测物体数，无需像素级标注。
- Mask R-CNN 在检测架构上加一条卷积分支输出每实例像素级 mask，完成实例分割；可视化方面从线性分类器模板、第一层卷积核（AlexNet 64×3×11×11）、梯度 Saliency，到 CAM（全局平均池化 + 类别权重加权，仅能用于最后一个卷积层）与 Grad-CAM（用梯度聚合权重 + ReLU，可作用任意层，Selvaraju et al. CVPR 2017），而 Transformer 自带注意力图、可视化远比 CNN 容易。

## 视频来源

https://www.youtube.com/watch?v=PTypu6GqEd4
