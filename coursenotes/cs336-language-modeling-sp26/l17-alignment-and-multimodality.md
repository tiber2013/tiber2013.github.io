---
title: "第 17 讲 · 对齐与多模态"
description: "核心结论"
---

# 第 17 讲 · 对齐与多模态

> **Stanford CS336 语言模型从零构建 · L17** ｜ 讲者 Stanford Online ｜ 78 分钟 ｜ [原视频](https://www.youtube.com/watch?v=26FtD08ZpOU)

**核心结论**

本讲从“语言模型只能 text → text”出发，指出前沿模型正在走向能接受并生成任意模态组合的 omni model。由于 Transformer 仍是各模态的主力，核心问题变成：如何把图像、视频、音频等非文本数据转成有语义的 token。课程以图像为主，先讲 CLIP/SigLIP 如何用图文对比学习获得语义表示，再讲 LLaVA 与 Qwen-VL 系列如何把视觉编码器接入 LLM，并扩展到多图、视频、动态分辨率和长上下文。最后讨论 Chameleon 的全离散 token 统一路线及其训练稳定性、信息损失等问题。

## 从语言模型到多模态 omni model [00:05 - 04:17]

语言模型已经把“任意文本到任意文本”做得相当通用，可以处理自然语言、代码、诗歌甚至 DNA 序列。但真实世界是多模态的：文本、图像、音频、视频同时存在。课程把终极目标称为 **omni model**：输入任意模态组合，输出任意模态组合；例如同时给图像和视频并提问，或者生成图像、把音频转成图像等。

当前现实的约束是：Transformer 在各模态上仍然工作得最好，所以必须让 Transformer “说 token”。这里的 token 不限于离散文本 token，也可以是连续 token/嵌入；关键是 token 应代表某种语义单元。文本的子词 token 尚有一定意义，而单个像素显然没有语义，因此非文本模态需要类似 BPE 的转换机制。课程集中回答两个问题中的第一个：**如何把非文本数据输入 Transformer**；输出/生成只简要提及。

![多模态数据类型的分类展示图，包含文本、图像、音频和视频四个模块](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0002_00-01-00.jpg)

![多模态终极目标及当前挑战：如何将非文本数据转化为 token](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0003_00-02-35.jpg)

## CLIP：用图文对比学习获得语义图像表示 [04:19 - 22:01]

### 动机与对比目标

CLIP 出现在 2021 年。当时语言模型已进入基础模型阶段，而视觉仍主要依赖 ImageNet 等大规模标注数据和 ResNet 等模型。OpenAI 的研究者想问：能否像语言模型利用海量嘈杂互联网文本那样，利用互联网上大量的图像-文本对？

CLIP 的做法是取一批图文对，分别用图像编码器和文本编码器编码成向量。对于图像 $I_i$ 和它对齐的文本 $T_i$，希望 $I_i^\top T_i$ 大于 $I_i$ 与其他所有文本的点积；反过来，对文本 $T_i$ 也 prefer 对齐的图像。于是形成一个 $N\times N$ 相似度矩阵，对角线为正例，非对角线为负例，在行、列两个方向做 softmax 交叉熵，相当于 $2N$ 个分类问题。设归一化后的相似度为 $S_{ij}=I_i^\top T_j/\tau$，对称损失可写为：

$$
\mathcal{L}
=-\frac{1}{2N}\sum_{i=1}^N
\left(
\log\frac{e^{S_{ii}}}{\sum_j e^{S_{ij}}}
+
\log\frac{e^{S_{ii}}}{\sum_j e^{S_{ji}}}
\right)
$$

![CLIP模型架构图：对比预训练、分类器生成与零样本预测](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0006_00-06-28.jpg)

![CLIP模型的核心伪代码实现，包含图像/文本编码、特征投影、相似度计算及对称损失函数](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0007_00-07-46.jpg)

### 数据与图像预处理

CLIP 的数据细节在论文中不多：大致是取一批查询，在线搜索并挖掘图文对，最终得到约 4 亿图文对；数据集未公开。OpenCLIP 复现并扩展了 CLIP，使用 LAION-5B（约 50 亿图像及文本描述），有趣的是 LAION 的数据过滤也用了 CLIP，存在一定 bootstrapping。

图像分辨率任意，而神经网络偏好固定尺寸。CLIP 的启发式处理是：将短边 resize 到 336 像素，再中心裁剪成 $336\times336$ 正方形。这样做是为了方便，也符合当时 ImageNet 分类的设定——物体通常在中心，裁掉一些背景影响不大。后续模型会看到更好的动态分辨率方案。

![CLIP模型的数据集构建与处理说明，包含训练数据规模及OpenCLIP复现情况](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0008_00-09-04.jpg)

### 视觉编码器：ViT

CLIP 实验了 ResNet 和刚出现的 Vision Transformer（ViT），发现 ViT 更好。通常说“CLIP”多指 ViT 版本。ViT 把图像切成 patch（原论文 $16\times16$，CLIP 用 $14\times14$），每个 patch 投影成一个 token，加上位置嵌入，送入标准 Transformer。输出是一串向量；CLIP 没有简单平均，而是采用 attention pooling：先对激活做全局平均得到 query，再与各位置 key/value 做一轮 attention，得到更“有信息”的单一向量。

CLIP 最佳模型是 ViT-L/14：较大 ViT、$14\times14$ patch、RGB 三通道、$336\times336$ 分辨率；训练时先在较低分辨率、后段再提高分辨率以节省时间。讲者提到，CLIP 也试过 2D 位置嵌入，但在分类任务上 1D 位置嵌入差别不大；后续更精细的任务才更需要空间结构。

![Vision Transformer (ViT) 架构图，展示图像分块、线性投影、位置嵌入及 Transformer 编码器结构](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0010_00-12-57.jpg)

### 文本编码器与零样本结果

文本编码器是 GPT-2 风格的 Transformer。为得到整段文本的单一向量，输入加 BOS、末尾加 EOS，取最高层 EOS 位置的激活作为表示。

CLIP 的 headline result 是：在 ImageNet 上，零样本 CLIP 超过了在 120 万 ImageNet 图像上训练的 ResNet-50。零样本预测的做法是：把图像与一组标签文本（如 “a photo of a dog”）分别算点积，取相似度最高的标签。也就是说，CLIP 把分类器“文本化”了，不需要为每个数据集重新训练分类头。

![CLIP模型文本编码器与ImageNet零样本性能说明页](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0012_00-20-16.jpg)

![利用图像-文本对进行预训练、构建数据集分类器及零样本预测的对比学习流程图](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0011_00-18-28.jpg)

CLIP 论文还比较了另一种目标：不从图像做对比排序，而是从图像预测文本（bag of words 或 Transformer 语言模型）。结果发现，使用更强的文本预测模型反而更差或更不高效；说明对于 ImageNet 分类这种粗略语义表示，精确建模 caption 的 token 序列并不是最重要，bag of words 已经足够。这个点也呼应了“理解”和“生成”对表示的不同要求。

![不同文本预测方案在ImageNet零样本准确率上的效率对比曲线图](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0013_00-20-42.jpg)

### CLIP 小结与技术局限

CLIP 的图像编码捕获了文本所描述的语义，因为文本通常谈论图像的高层内容。但它的设计决策基于图像分类，因此表示并不精细。技术上，CLIP 需要很大的 batch（约 3 万），softmax 在整个 batch 上操作，损失不易分解；这不同于语言模型训练中样本之间基本独立、可在最后聚合的方式。尽管如此，CLIP 仍是后续 VLM 的稳健起点。

![CLIP设计总结与SigLIP损失函数引入页](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0014_00-22-00.jpg)

## SigLIP：把对比学习改成二分类 [22:37 - 28:33]

SigLIP 是 CLIP 的改进版，全称是 Sigmoid Loss for Language Image Pre-Training。它把 CLIP 的“多分类”改成“二分类”：对任意图文对，只判断它们是否对齐。相似度矩阵中对角线是正例，非对角线是负例；对每个元素独立做 logistic 损失，而不是在整行/整列上做 softmax。

设归一化图像、文本嵌入为 $\hat I_i,\hat T_j$，可学习温度 $t$ 和偏置 $b$，则

$$
\text{logits}_{ij}=t\,\hat I_i^\top \hat T_j+b,
\qquad
y_{ij}=\mathbf{1}[i=j]
$$

$$
\mathcal{L}
=-\frac{1}{N}\sum_{i,j}
\log\sigma\big((2y_{ij}-1)\,\text{logits}_{ij}\big)
$$

其中 $2y_{ij}-1$ 在对角线为 $+1$，非对角线为 $-1$。实现上就是归一化、点积、加温度与偏置、构造标签、算 log-sigmoid 后求平均。

![SigLIP 方法总结与目标对比幻灯片，包含 CLIP 与 SigLIP 分类目标差异及算法伪代码标题](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0015_00-23-18.jpg)

![SigLIP 的 Sigmoid 损失伪代码实现，包含图像/文本嵌入归一化、点积计算 logits 及损失函数](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0016_00-24-57.jpg)

SigLIP 使用 Google 的 WebLI 数据集，约十亿级图文对，从互联网抓取，并对图像中的文字做 OCR 以形成额外图文对，再做质量过滤，支持多语言。效率上，CLIP 用 256 块 TPUv3 训练 10 天，SigLIP 用 32 块 TPUv4 训练 5 天；单芯片 FLOP/s 不一定更高，但 pod 规模与互联带来很大加速。

SigLIP 的另一个优点是解耦 batch size 与损失函数。CLIP 改变 batch size 就改变了损失；SigLIP 的损失在期望上不随 batch size 改变，小 batch 下比 CLIP 更好，大 batch 可到很大但 32K 左右已接近临界 batch size。分布式训练时，每台设备先算本地图文对的损失，再轮换发送文本以覆盖非对角线块，从而并行计算跨设备负例。

![SigLIP 在多个设备上的分布式训练架构图，展示了不同设备间的损失计算与跨设备求和过程](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0018_00-27-11.jpg)

![SigLIP 批大小（Batch size）设置说明幻灯片，包含分布式架构图及批大小相关结论](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0019_00-28-28.jpg)

## LLaVA：把视觉编码器接到语言模型上 [29:37 - 35:21]

CLIP/SigLIP 给出固定尺寸图像到语义向量的编码器。接下来是 VLM 的通用模板：取现成视觉编码器，取现成 LLM，用投影器/适配器把视觉输出映射到语言模型嵌入空间，再拼接训练。这更像 mid-training/post-training 的“缝合”，而不是从零训练。

LLaVA 2023 的组成是：视觉编码器用 CLIP（ViT-L/14），文本解码器用 Vicuna——LLaMA 在 ShareGPT 对话上微调后的模型。数据来自 MS COCO：图像有人工标注的 bounding box 和 Mechanical Turk caption。LLaVA 用 GPT-4 根据 caption 或检测到的物体生成问题、对话、详细描述、复杂推理等，合成约 15.8 万条样本。

模型上，图像经 CLIP 得到向量，再乘矩阵 $W$ 投影到与文本嵌入相同的空间；文本用标准嵌入。这样图像向量和文本向量组成一个序列，送入标准 Transformer，输出文本。训练分两阶段：第一阶段冻结视觉编码器和语言模型，只训练 $W$，称为 alignment，让图像向量看起来像语言模型能理解的 token 嵌入；第二阶段仍冻结视觉编码器，训练 $W$ 和语言模型，在“图像+对话/描述/推理”数据上微调。

![LLaVA 模型架构图，展示视觉编码器、投影层与语言模型的连接方式及训练策略](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0023_00-33-39.jpg)

![LLaVA 对“极端熨烫”视觉输入示例的问答对比，展示模型对图像异常之处的识别与描述](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0024_00-34-57.jpg)

## LLaVA OneVision：多图、视频与 AnyRes [35:30 - 46:01]

LLaVA OneVision 延续同一 recipe，但把能力扩展到单图、多图和视频。视频被看作采样帧的序列。视觉编码器升级为 SigLIP，文本解码器用 Qwen-2 72B，投影器从线性层升级为两层 MLP。

一个关键问题是分辨率。OCR 需要保留细粒度信息，否则 “J” 可能看起来像 “I”；而 CLIP 式 resize + 中心裁剪到 $336\times336$ 会丢失文档细节。LLaVA 1.5 引入 **AnyRes**：把图像切成 $a\times b$ 个块，每块匹配视觉编码器期望的分辨率，分别编码后拼接；如果 token 太多，再用双线性插值降采样。这样视觉 token 数可以随图像分辨率自适应，类似 Transformer 对文本长度的自适应。

![LLaVA OneVision 架构图，展示其支持单图、多图与视频输入的模型结构](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0025_00-36-14.jpg)

![多模态模型架构与数据处理策略幻灯片，包含视觉编码器、文本解码器及AnyRes方案说明](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0026_00-37-32.jpg)

为了不让某种模态主导训练，OneVision 对不同输入类型采用不同 token 策略：单图使用全图下采样加最多 9 个 crop，约 $729+N\cdot729$，最大约 7290 token；多图每图用 base resolution，例如 $12\times729=8748$；视频每帧用更低分辨率，每帧约 196 token，最多 32 帧，约 $32\times196=6272$。目标是让单图、多图、视频的 token 长度大致可比，避免视频帧重复淹没数据。

![单图、多图和视频三种输入类型的Token策略对比图，展示不同模态的Token数量计算方式](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0027_00-40-07.jpg)

数据方面，OneVision 强调高质量与数量，并大量合成任务特定数据；讲者指出这带有明显的 post-training 色彩，且不少数据是通过蒸馏 GPT-4 类模型得到的。训练分三阶段：Stage-1 语言-图像对齐，只训投影器；Stage-1.5 高质量知识学习；Stage-2 视觉指令微调。讲者认为三阶段未必有严格原理，主要是把高质量知识数据和下游任务风格数据分开。

![三阶段训练流程图及Stage-1/1.5/2各阶段超参数配置表](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0029_00-42-43.jpg)

OneVision 还展示了跨模态迁移：训练时只有单图图表/OCR 数据，测试时可以处理“表格+图表”的多图问答；训练时只有单图 OCR 和多图关系推理，测试时可以用于多截图 GUI agent；视觉提示（在图上圈出区域）只在单图训练，却能泛化到视频中“描述被圈出的球员”。这说明即使数据是任务导向的，任务足够多时仍会出现迁移。

![多图GUI理解任务示例：展示四张连续的移动端UI截图及对应的操作指令](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0031_00-44-21.jpg)

![视频视觉提示（Visual Prompting）任务示例：在足球比赛视频帧中圈出特定球员](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0032_00-44-41.jpg)

## Qwen-VL 系列：从固定分辨率到动态分辨率与长上下文 [46:04 - 59:21]

Qwen 从 2023 年开始做多模态。**Qwen-VL** 首版使用 OpenCLIP 视觉编码器；适配器采用一层 cross-attention，加入 2D 位置嵌入，并映射到固定 256 个 token。它引入特殊 token：`<image>` 表示图像，`<box>` 表示 bounding box，`<ref>` 表示描述。训练分三阶段：Stage-1 用大规模低质量数据，冻结语言模型，训练视觉编码器和适配器（与 LLaVA 不同，这里会训视觉编码器）；Stage-2 用高质量任务特定数据（VQA、图表问答等），训练全部参数；Stage-3 用指令微调数据，冻结视觉编码器，训练适配器和语言模型。Qwen-VL 支持中文、代码、OCR，并能输出 bounding box，但不生成图像。

**Qwen2-VL** 的主要升级是动态分辨率和更大视觉编码器。每个 $224\times224$ patch 用 ViT 编码，再通过 $2\times2$ 压缩降低 token 数；视频每秒采样 2 帧，最多约 16k token。位置编码改为 **M-RoPE**：每个 patch 由高度、宽度、时间三个坐标定义，每个维度分别算 RoPE 再拼接。训练仍分三阶段。模型展示出视频理解、数学、代码、function calling 等能力。

**Qwen3-VL** 使用 Qwen3 系列的 dense/MoE 语言模型，支持最长 256K 上下文，这对长视频很重要。视觉编码器换成 SigLIP-2，架构与 SigLIP 相同、向后兼容。M-RoPE 做了改进：原先按维度块把时间、宽、高分别分配给不同维度，可能导致某些轴只暴露低频或高频；Qwen3-VL 改为交错频率，使各轴都能接触低高频。视频时间戳从隐式位置编码改为显式 token，例如 “0 秒” 是一个 token，便于直接引用“两秒后发生了什么”。损失上采用按样本长度平方根归一化的 per-token loss，降低超长视频样本的权重。适配器使用 **DeepStack**（讲者提到源自 DeepSeek 团队的工作），把视觉编码器多层特征直接加入语言模型残差流，实现更深的融合，而不是把视觉编码器当黑盒只输出一串向量。

Qwen3-VL 的训练管线更复杂：预训练四阶段，先训适配器，再逐步把序列长度从 8K 扩到 32K、256K，大部分 token 在第二、三阶段；后训练三阶段，包括长 chain-of-thought SFT、知识蒸馏和强化学习。讲者展示的结果表明 Qwen 系列在多模态基准上已与闭源模型可比。总体看，从 Qwen-VL 到 Qwen2-VL、Qwen3-VL，框架不变，主要是放大模型、整理更多数据、处理长上下文、改进图像处理和位置编码。

## 训练系统、对齐与参数规模的补充讨论 [59:21 - 67:19]

讲者澄清：这些 VLM **不生成视频或图像**，所有多模态能力都在输入侧，输出始终是文本。除强化学习外，大多数训练阶段对每个 token 做监督，没有额外的“LLM 裁判”来评价描述质量，质量主要由数据集决定；RL 阶段才可能引入不同奖励。

系统上，多模态训练不比纯语言模型轻松：视频数据更大，加载本身可能成为瓶颈，需要异步数据加载，让数据准备与计算重叠。视频/图像 token 更多，需要归一化或降权，避免视频压过文本；不过文本 token 总量仍然很大，多模态 token 不一定远超文本 token。对齐阶段，语言模型必须是预训练好的，并且被冻结；只训练适配器，把给定视觉编码器接到给定语言模型上，通常按固定 token 预算训练，没有自适应阈值。

参数规模上，视觉编码器通常远小于语言模型。原因是视觉编码器做的是较局部的 patch 操作，不需要太多知识，主要能力仍在语言模型。以 LLaVA 为例，projector 约 7200 万参数，ViT 通常小于 10 亿参数，而语言模型可达数百亿。

## Chameleon：把图像也变成离散 token 的统一模型 [67:19 - 74:43]

前面的 VLM 都是“连续视觉编码 + 投影 + LLM”，只能生成文本。Meta 的 Chameleon（2024）提出另一种美学上很吸引人的路线：把所有模态都映射成离散 token，让图像和文本真正住在同一个序列空间里。这样模型可以理解图像，也可以生成图像，文本和图像可以交错输出；训练也退化成标准语言模型训练，没有适配器、没有单独的视觉编码器。

关键组件是 VQ-VAE（2017）。编码器把图像 patch 映射成连续向量，再量化到码本中最近的码字；码本例如有 8000 个原型向量。解码器从离散码重建图像，训练目标主要是重构损失；由于量化不可微，还需要额外项（讲者未展开）。这样 $512\times512$ 图像可变成 1024 个离散 token，每个 token 来自 8000 词表。Chameleon 为此训练新的 tokenizer，然后按标准 LM 方式训练：第一阶段大规模无监督文本与图文数据，第二阶段混入高质量数据。

问题也很明显。文本 token 的熵较低，下一个词大多可预测；图像 token 的熵很高，具体蓝色深浅很难预测。把两者放在同一序列中训练会导致参数范数增长、loss 不稳定；Chameleon 用 QK norm 和 z-loss 正则缓解。离散化还会丢信息，例如很小字号的 OCR 文本离散后可能读不出来。多模态数据配比也更难调。VQ-VAE 路线在图像生成中曾流行，因为 Transformer 需要离散 token；但扩散模型成熟后，这条路线相对不那么流行。Chameleon 的优雅之处是统一，但性能和信息保真度是代价。

## 总结：多模态的关键是编码、配比与生成方式 [74:43 - 77:39]

前沿模型正被期望具备多模态甚至原生多模态/omni 能力，但闭源模型的具体做法不公开。讲者推测，较稳妥的组合仍是：连续编码器（如 CLIP 类）负责理解，Transformer 作为主干，扩散模型负责高质量生成。多模态的根本挑战在于非文本模态如何编码；理解与生成的需求并不相同——分类/理解可以依赖高层语义，OCR 和生成则需要细粒度细节，因此不存在一个通用的编码器。训练时还要仔细权衡模态配比：视频信息密度低，不应让视频 token 淹没文本。CLIP 虽然已出现数年，其思路仍是捕获图像语义的常用起点。

![多模态模型总结及CLIP模型背景介绍](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0005_00-05-11.jpg)

## 要点回顾

- **omni model** 是终极目标：任意模态组合输入、任意模态组合输出；本讲主要讨论输入侧。
- Transformer 仍是各模态主力，因此必须把非文本数据转成离散或连续 token，且 token 应代表语义单元。
- **CLIP** 用图文对比学习：一批图文对形成 $N\times N$ 相似度矩阵，对角线为正，做对称 softmax 交叉熵；零样本 ImageNet 超过同数据训练的 ResNet。
- CLIP 的图像预处理是短边 resize 到 336 再中心裁剪；视觉编码器用 ViT，文本编码器用 GPT-2 风格 Transformer，取 EOS 激活。
- CLIP 需要大 batch，softmax 不易分解；**SigLIP** 改为逐对二分类的 sigmoid 损失，解耦 batch size 与损失，训练更高效。
- **LLaVA** 的模板是 CLIP 视觉编码器 + 线性投影 $W$ + Vicuna；两阶段训练：先只训 $W$ 做对齐，再训 $W$ 和 LLM。
- **LLaVA OneVision** 支持单图、多图、视频；用 AnyRes 处理高分辨率，用不同 token 策略平衡单图/多图/视频；三阶段训练并展示跨模态迁移。
- **Qwen-VL → Qwen2-VL → Qwen3-VL** 逐步引入动态分辨率、M-RoPE、显式视频时间戳、长上下文、DeepStack 深融合和更复杂的三阶段/多阶段训练。
- **Chameleon** 用 VQ-VAE 把图像离散化，走全 token 统一路线；优雅但训练不稳定、离散化丢信息，扩散模型流行后该路线相对少用。
- 当前较稳妥的组合是连续视觉编码器 + Transformer + 扩散生成；多模态难点在编码粒度、模态配比、长上下文和训练系统。
