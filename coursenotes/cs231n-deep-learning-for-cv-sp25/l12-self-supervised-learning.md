---
title: "第 12 讲 · 自监督学习：从代理任务到对比学习"
description: "Stanford CS231n Spring 2025 L12 中文视频笔记（2026-11 重写版）"
---

# CS231n Spring 2025 · Lecture 12 自监督学习：从预训练任务到对比表示学习

**核心结论**：自监督学习要解决的矛盾是——大规模训练需要海量人工标注（分割甚至要逐像素标注），成本高到难以承受；它的办法是在无标注数据上定义一个「预训练任务」（pretext task），让标签由数据本身自动生成，用监督式的分类／回归／重建目标训练出一个编码器，再把编码器迁移到只有少量标注的下游任务。本讲沿两条主线展开：前半段是图像变换类与重建类预训练任务（旋转预测 RotNet、拼图、图像修复 inpainting、图像／视频着色、Split-brain Autoencoder），以及当下最常用的掩码自编码器 MAE（ViT、75% 掩码、非对称编解码器、只算掩码块的 MSE）；后半段转入对比表示学习，用 InfoNCE 损失把同图变换拉近、把不同图像推远，并串起 SimCLR 的大 batch 与投影头、MoCo 的队列与动量编码器、以及 DINO 的 student-teacher 自蒸馏。全讲反复强调：评估自监督方法的最终标准是下游任务表现，而好的预训练任务必须足够难、足够通用，才能逼出真正有意义的视觉表征。

## 1. 问题的起点：标注瓶颈与「学习到的表示」 [00:05 - 05:11]

讲者先回顾本学期脉络：上一讲是 GPU 与多 GPU 的大规模训练；再往前覆盖了分类、语义分割、目标检测、实例分割等关键视觉任务，以及可视化与理解模型。其中一条重要线索是：在像素空间做最近邻效果很差，但如果改用网络中间的全连接层／特征图（learned representations、features、embeddings、latent space 等不同叫法），用 L2 距离做最近邻，就能很好地代表图像，末端接一个简单的线性模型即可读出类别标签。

矛盾也正出在这里：这类网络从图像训到类别标签，规模一大就需要海量人工标注——分类要一张张图打标签，分割则要逐像素标注，成本高到几乎不可行。于是问题被明确为：**能不能绕过人工标签，仍然训练出能提取好特征的神经网络**？给定一个无标签的大图像集（例如从互联网下载的自然图像），假设可以用一个目标函数／预训练任务训练出好特征，再把训练好的编码器迁移到只有少量标注数据的具体任务（工业、医学应用等）。这就是自监督学习登场的背景。

## 2. 自监督学习的整体框架：pretext task 与 downstream task [04:39 - 08:26]

自监督学习的核心是一个两阶段范式。第一阶段在无标注数据集上定义 **pretext task（预训练任务）**：数据集没有标签，编码器（encoder）把图像映射为学习到的表示（learned representation），再由同一网络的另一模块——解码器、分类器或回归器——把表示映射到输出空间，而这些标签／输出是从数据本身自动生成的，不是人工标注。这样就可以写出损失函数，用监督学习的目标（分类或回归）来训练整个网络；当结构是 encoder + decoder 时，它更接近自编码（autoencoding）框架。

![自监督学习完整流程图，对比预训练任务与下游任务，并标注了训练好的编码器](/coursenotes/cs231n-deep-learning-for-cv-sp25/l12-self-supervised-learning/kf_0001_00-05-53.jpg)

上图把完整流程拆成左右两半：左侧 pretext task 强调「Define a task based on the data itself」「No manual annotation」，虽然可看作无监督任务，却用有监督的目标（分类或回归）来学；右侧 downstream task 是「你真正关心的应用」，只有小数据集、但带有标签。

![自监督学习中预训练任务的具体架构，展示编码器、学习到的表示和自动生成的标签/输出](/coursenotes/cs231n-deep-learning-for-cv-sp25/l12-self-supervised-learning/kf_0002_00-06-34.jpg)

第二阶段把训练好的编码器迁移到 **downstream task（下游任务）**：去掉预训练任务专用的头部，换成一层线性函数或一个浅层全连接网络，用少量带标签数据训练它来预测类别。讲者强调通常只需要浅层网络——如果特征足够好，就不需要在下游做大量训练。因此定义预训练任务要同时满足两条：**足够通用**以学到好特征，**标签来自数据本身、无需人工标注**。课程快速覆盖的四类例子是：图像补全（mask 一部分、预测被遮部分）、旋转预测、拼图（把打乱的 patch 输出正确顺序）、着色（由黑白版本预测每个像素的颜色）。讲者还点明，这套范式正是 GPT-4 等大语言模型得以用原始无标注数据训练的原因，也广泛用于语音、机器人与强化学习——这解释了为何湾区有那么多自动驾驶车在采集数据却不必标注。

## 3. 如何评估一个自监督学习方法 [10:32 - 12:20]

评估可以从多个维度切入，讲者列出五个方面。

![评估自监督学习方法的五个维度：预训练任务表现、表征质量、鲁棒性、计算效率与迁移性能](/coursenotes/cs231n-deep-learning-for-cv-sp25/l12-self-supervised-learning/kf_0003_00-11-12.jpg)

- **Pretext task performance**：因为标签是自己生成的，可以直接衡量模型把预训练任务解得有多好。
- **Representation quality**：不微调，直接看表示本身，手段包括线性评估协议（Linear Evaluation Protocol）、聚类（Clustering）和 t-SNE 降维可视化——t-SNE 把高维表示降到 2D/3D 后观察是否出现有意义的分布模式。
- **Robustness and Generalization**：鲁棒性与泛化。
- **Computational Efficiency**：计算效率。
- **Transfer Learning and Downstream Task Performance**：下游任务表现。讲者明确说，**最重要的始终是最后一项**：设计预训练任务、做自监督学习的全部目的，就是提升我们真正关心的那个任务的结果。

## 4. 图像变换类预训练任务（一）：旋转预测与拼图 [15:36 - 26:33]

### 4.1 旋转预测 RotNet [15:36 - 23:00]

旋转预测的假设建立在「视觉常识」上：模型只有具备「物体未受扰动时应该长什么样」的常识，才能判断一张图被转了多少度；能抓住这一点，就意味着它能把整幅图概括成有用的特征。

![自监督学习中通过预测图像旋转角度（4分类）的预训练任务流程图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l12-self-supervised-learning/kf_0004_00-17-29.jpg)

Gidaris et al. 2018 的实现非常简洁：只取 $0^\circ / 90^\circ / 180^\circ / 270^\circ$ 四种旋转，把整幅输入图像按其中之一旋转，再用卷积网络 + 全连接层预测「施加的是哪一种旋转」。因为只有四种可能，这是一个 **4 分类任务**（画面写作「Maximize prob. $F^0(X^0)$」「4-way classification」），不必回归出精确角度。训练后去掉预训练任务专用的 FC 层，换上线性分类器，用真实的物体标签微调；论文里冻结前两层、只微调最后的卷积层与线性层。

在 CIFAR-10 上，预训练模型从一开始就有较高准确率，说明它已处于「形状良好」的状态；讲者也提醒，CIFAR-10 任务较简单，全监督与预训练版本往往收敛到相近精度，但在更困难的应用里，不做大规模预训练的监督框架通常拿不到同样好的结果。在 PASCAL VOC 数据集（含分类、检测、分割三类任务，各类任务分别设定只训少数 FC 层或训全部层）上，对比数字很能说明问题：

![自监督学习特征在下游任务（分类、检测、分割）上的性能对比表，突出RotNet模型表现](/coursenotes/cs231n-deep-learning-for-cv-sp25/l12-self-supervised-learning/kf_0005_00-19-55.jpg)

画面表格给出「Transfer learned features to supervised learning」的结果——ImageNet 全标签预训练为 78.9 / 79.9 / 56.8 / 48.0，随机初始化为 39.2 / 53.3 / 43.4 / 19.8，而 (Ours) RotNet 达到 70.87 / 72.97 / 54.4 / 39.1（指标为分类 %mAP、检测 %mAP、分割 %mIoU）。也就是说，随机初始化与旋转预训练差距巨大，而 RotNet 虽仍不及用完整 ImageNet 标签预训练，却已相当接近，明显优于同期其他自监督方法。

论文还把特征投影回图像空间看注意力图（可用 Grad-CAM 一类方法）：

![对比展示监督模型与自监督模型的视觉注意力图（Attention maps）在不同卷积层的分布](/coursenotes/cs231n-deep-learning-for-cv-sp25/l12-self-supervised-learning/kf_0006_00-22-24.jpg)

对比 Conv1（$27\times27$）、Conv3（$13\times13$）、Conv5（$6\times6$）各层可见，监督模型的注意力更「聚焦」（它只解决分类这一个任务，抓住眼睛和轮廓就不管其他区域），而自监督模型覆盖更多区域——因为不知道下游任务是什么，必须对整幅图有更整体（holistic）的理解，才能在多种任务上都表现不错。

### 4.2 拼图与相对位置预测 [23:16 - 26:33]

另一类流行做法把图像切成 $3\times3$ 网格：以中心 patch 为参考，给定一个 patch，让网络预测它相对中心的位置，输出是 8 个方位之一，因此是 **8 分类任务**（画面示例输出为 3）。

后续工作把它升级成真正的 jigsaw puzzle：把 9 个 patch 随机打乱，让网络预测「正确的排列」。学生当场答出排列数是 $9!$（约 36 万种），确实太大；于是作者建了一张查找表，只保留 64 个「彼此差异足够大」的排列（避免只交换一两个 patch 的近似排列），把任务变成输出 64 维向量的 **64 分类**。讲者在 Q&A 里补充，选 64 几乎是任意的、主要是为了把问题化归为分类，并挑选差异最大的那些排列。这篇 2016 年的工作在同一数据集上超过了此前若干框架。

## 5. 重建类预训练任务：图像修复与着色 [26:33 - 35:43]

### 5.1 图像修复 Inpainting（Pathak et al., 2016） [26:33 - 30:52]

做法是简单掩码：遮住图像的一部分，让模型补回被遮区域。由于原图完整，天然就有期望输出，不需要人工标注。

![通过重建学习图像修复的编码器-解码器网络架构图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l12-self-supervised-learning/kf_0007_00-27-23.jpg)

结构是一个编码器 - 解码器网络：编码器把输入变成特征空间，中间经过 channel-wise fully connected 层，解码器重建缺失像素，$L(\cdot,\cdot)$ 比较输出与真值，本质是「learning to reconstruct the missing pixels」，也是用掩码目标训练的一种自编码器。重建损失是逐元素（element-wise）的图像差值，再与掩码逐元素相乘，从而**只在被遮区域计算损失**。

讲者指出纯重建的一个老问题：早期重建框架会产出模糊、过度平滑的输出。为此 Pathak 等人额外加了一个 **adversarial objective（对抗目标）**，让生成的图像看起来更真实——这部分属于下一讲「生成模型」的内容，此处不展开。同样的框架在分类、检测、分割任务上也带来了额外收益，并且讲者预告：这种重建／掩码思路正是当今最常用的预训练方式之一，稍后会回到它（即 MAE）。

### 5.2 图像着色与 Lab 色彩空间 [31:08 - 32:47]

数据集大多是彩色图，于是可以把颜色本身当作免费的标签。计算机视觉惯用 RGB；但要**把亮度／光照与颜色分离**，就要用 Lab 一类色彩空间：$L$ 通道存亮度（lightness），$a$、$b$ 两个通道存颜色，三者合起来可还原彩色图。

![图像着色预训练任务示意图，展示从灰度L通道预测ab颜色通道的网络结构](/coursenotes/cs231n-deep-learning-for-cv-sp25/l12-self-supervised-learning/kf_0008_00-31-12.jpg)

预训练任务于是定义为：给定灰度 $L$ 通道 $X \in \mathbb{R}^{H\times W\times 1}$，预测颜色 $\hat{Y} \in \mathbb{R}^{H\times W\times 2}$ 的 $ab$ 通道。标签早已存在于数据中，完全不需要人工标注。

### 5.3 Split-brain Autoencoder [32:47 - 35:43]

既然可以「由 L 预测 ab」，反过来也应成立，这催生了 **Split-brain Autoencoder**：把输入拆成两组通道（$X_1$ 与 $X_2$），训练两个网络 $F_1$、$F_2$ 各自「由一组预测另一组」，合并两路输出还原整幅图，再用 L2 等距离函数算损失并反向传播。

![Split-brain Autoencoder架构图，展示RGB与HHA深度通道交叉预测的双分支结构](/coursenotes/cs231n-deep-learning-for-cv-sp25/l12-self-supervised-learning/kf_0009_00-34-24.jpg)

画面给的是 RGB-HHA 版本：框架并不限于颜色与亮度，也可用于 RGB-D 传感器（如 Kinect，机器人常用）——由 RGB 通道预测深度（HHA）、或反过来由深度预测 RGB，曾是非常成功的下游应用。把 $F_1$、$F_2$ 的特征拼接后用于分类，这些特征本身已有不错的类别预测精度；当然因为完全没用标签，整体仍不及监督学习。

## 6. 从着色到视频着色：时间一致性与隐式跟踪 [35:43 - 40:52]

图像着色还有预训练之外的价值：可以给没有彩色版本的老照片、老视频上色。讲者举了一张 Half Dome（半圆顶）的例子——模型上色后，山体、树木、桥梁与它们在水中的倒影颜色保持一致，说明它从海量数据里学到了「倒影应保留物体颜色」这层规律；同时也提醒，这些都是「前大视觉模型时代」针对特定任务训练的模型，并不通用，不是被训练来解决一切问题的。

![视频着色预训练任务示意图，展示参考帧与待着色序列帧](/coursenotes/cs231n-deep-learning-for-cv-sp25/l12-self-supervised-learning/kf_0010_00-37-13.jpg)

扩展到视频（Vondrick et al., 2018）：以一帧已着色的 **reference frame** 为参考，给后续 $t=0,1,2,3$ 的帧上色，思路是「model the temporal coherence of colors in videos」。背后的假设很有意思：学会给未来帧上色，本质上要跟踪像素与物体的运动，于是模型会在无标签的情况下**隐式学会跟踪区域或物体**。

机制上，这与前面讲过的 attention 一脉相承：对目标帧的每个像素，用 CNN 提取其特征，计算它与参考帧所有像素的特征相似度（attention 就是两者相似度），再按注意力对参考帧的颜色做加权平均，作为该像素的目标颜色；因为数据中有真实颜色，可以直接算损失。

![视频上色定性结果展示，对比参考帧、灰度目标帧与预测彩色帧的效果](/coursenotes/cs231n-deep-learning-for-cv-sp25/l12-self-supervised-learning/kf_0011_00-40-09.jpg)

定性结果说明一致性的价值：若逐帧独立上色、没有跨时间的约束，人物衣服的颜色常会逐帧变化；有了参考帧的注意力复制，颜色就能保持稳定。由于计算了对参考帧的注意力，这套框架还自然衍生出视频里跟踪物体、跟踪片段，甚至识别关键点的能力。

## 7. Q&A：预训练任务凭什么训练出好编码器 [40:56 - 45:41]

课堂问答澄清了几个容易被含糊带过的问题。其一，**编码器是怎么「认识」数据的**：所有展示过的预训练任务，都在用解码、分类或回归生成某种输出，反向传播的是「预测自生成标签」的损失函数；在学着预测正确输出的过程中，编码器被一并训练。它和监督分类没有本质区别，只是把「预测物体类别」换成了「预测像素颜色／旋转角度／缺失像素」，而后者在数据里本来就有。

其二，**编码器与解码器是不是同一个网络**：视任务而定。旋转预测里只是一个网络，末端 FC 层当作「解码器」，下游时替换头部即可；图像自编码常是端到端训练的两个网络（为了利用中间的表示空间）；而后面要讲的掩码自编码器，编码器与解码器甚至可以完全不对称、分属两种架构。

![自监督学习方法评估流程图，展示从预训练到目标任务评估的步骤](/coursenotes/cs231n-deep-learning-for-cv-sp25/l12-self-supervised-learning/kf_0012_00-42-56.jpg)

画面把标准评估流程固化成两步：1）从自监督预训练任务（如预测旋转）学到好的 feature extractor（例如一个 convnet）；2）在特征提取器上接一个浅层网络，用少量带标签数据在该目标任务（分类、检测等）上训练这个浅层网络，再评估。

最后讲者给这批早期方法一个免责声明：它们本就不是为了「解决一切」而设计的，其假设是「能说出这张图被转了 90°，就隐含理解了正确朝向，给它一张正立的图也就能认出里面是什么」；局限在于单个预训练任务学到的表示可能不够通用，只擅长解决那类特定任务。

## 8. 掩码自编码器 MAE（He et al., 2021） [45:56 - 58:01]

### 8.1 整体思路与高比例掩码 [45:56 - 49:06]

回到前面反复提到的重建／掩码思路，就进入 2021 年的 **Masked Autoencoders（MAE）**（He et al., 2021《Masked Autoencoders Are Scalable Vision Learners》），以及它之后的大量跟进工作。如今要在原始数据上做预训练，常常直接用 MAE；它与 inpainting 同属重建式框架，但细致得多，也不止遮一块。

![掩码自编码器（MAE）在不同掩码比例下的图像重建效果对比图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l12-self-supervised-learning/kf_0013_00-45-59.jpg)

画面对比了 50% 与 75% 掩码比例下的重建效果：即使面对「Reconstruction with a larger masked portion」，模型仍能补全大面积缺失，同时得到能概括图像的好编码器。

![MAE的掩码方法说明及编码器-解码器架构图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l12-self-supervised-learning/kf_0014_00-48-33.jpg)

做法是 ViT 风格：把图像切成 patches，均匀采样其中很大一部分（75%）掩码掉，**只把未掩码的 25% patch 送入编码器**（画面注明不使用 mask token 进入编码器）。高掩码比例让预测任务变得很难，而在自监督里「任务难」是好事——模型必须学到真正好的特征才能重建；同时每次遮掉 75% 意味着同一张图可以被反复以不同方式掩码、多次复用，等于大幅做了数据增强，因此有条件使用一个很大的 ViT 当编码器。

### 8.2 非对称编解码器、训练目标与下游用法 [49:06 - 58:01]

编码器只看见 25% 的可见 patch：先做一次线性投影到嵌入空间，再加上 positional embeddings（与 ViT 完全一样），随后是若干 transformer blocks；编码器很大。解码侧，对被掩码的位置放一个**可学习的 shared mask token**（类似 class token，可理解为「平均 patch」表示），把编码器输出与这些 mask token 合并成完整序列，由另一个 transformer 解码器还原整幅图像；解码器只在训练时用于重建，训练完即丢弃，因此编码器与解码器是**非对称**的。

![MAE解码器工作原理幻灯片，含输入-编码器-解码器-目标的重建流程图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l12-self-supervised-learning/kf_0016_00-56-41.jpg)

训练目标很简单：图像与重建图像之间的 **MSE（均方误差）**，且**只在被掩码的 patch 上计算**（与前面 inpainting 的掩码损失一致）。论文对大量设计选择做了消融：掩码比例 75% 给出最高准确率、decoder depth / width、mask tokens、reconstruction targets、data augmentation，以及 mask sampling strategy——比较随机掩码、块状掩码、网格掩码后结论是 **random masking 最好**。MAE 的表现超过了当时的 DINO、MoCo v3 等对比学习框架。

![线性探测与全量微调的对比幻灯片，含编码器-线性层结构示意](/coursenotes/cs231n-deep-learning-for-cv-sp25/l12-self-supervised-learning/kf_0015_00-52-17.jpg)

下游使用有两种范式：**Linear probing（线性探测）**冻结预训练编码器，只在末端加一层线性层训练，它衡量的是表示本身的质量；**Full fine-tuning（全量微调）**则连预训练编码器一起（全部或仅部分 transformer blocks）微调，能更充分地发挥模型潜力去适配新任务。Q&A 里进一步确认：编码器把每个可见输入 patch 变成一个 token，掩码位置补上共享 mask token 拼成长序列，解码器再把这些 token 投影成输出像素值。

## 9. 对比表示学习：InfoNCE、SimCLR 与 MoCo [58:30 - 74:34]

![今日议程幻灯片，高亮对比表示学习章节（SimCLR/MOCO/CPC/DINO）](/coursenotes/cs231n-deep-learning-for-cv-sp25/l12-self-supervised-learning/kf_0017_00-58-30.jpg)

议程幻灯片显示，讲完图像变换类预训练任务与 MAE 后，本讲最后进入「Contrastive representation learning」：实例对比学习（SimCLR 与 MoCo）与无标签自蒸馏（DINO）。

### 9.1 吸引／排斥与 InfoNCE 损失 [58:40 - 65:59]

前面所有变换都只是**同一物体的不同形式**；而数据集里还有大量「看起来完全不同」的其他物体。于是可以定义另一种任务：把属于同一物体的表示在表示空间里**拉近（attract）**，把不属于同一物体的表示**推远（repel）**。取参考图像 $x$，其各种变换为正样本 $x^+$，batch 中来自其他物体的样本为负样本 $x^-$。

设打分函数 $s(\cdot,\cdot)$ 衡量两个编码特征的相似度，要求 $s(f(x), f(x^+))$ 大于 $s(f(x), f(x^-))$。把分数取 $\exp$ 并做 softmax 归一化，就得到 **InfoNCE（information noise contrastive estimation）** 损失：

$$\mathcal{L} = -\mathbb{E}_x\!\left[\log \frac{\exp(s(f(x), f(x^+)))}{\exp(s(f(x), f(x^+))) + \sum_{j=1}^{N-1}\exp(s(f(x), f(x_j^-)))}\right]$$

![对比学习的损失函数公式及其与N路softmax交叉熵损失的对应关系](/coursenotes/cs231n-deep-learning-for-cv-sp25/l12-self-supervised-learning/kf_0018_01-03-29.jpg)

画面指出它等价于 **N 路 softmax 分类器的交叉熵**：若有 $N$ 个样本，目标就是「从这 $N$ 个里挑出那个正样本」——最大化正样本对的分数、压低所有负样本对的分数，与多分类交叉熵是同一回事。理论层面，InfoNCE 论文证明该损失的相反数是 $x$ 与 $x^+$ 之间**互信息（mutual information，衡量两图共享信息／依赖程度）的下界**：最小化损失就等于最大化二者互信息；而且**负样本越多，这个界越紧**。这正是此类方法需要**很大 batch size** 才能收敛更快、效果更好的原因。

### 9.2 SimCLR [66:02 - 71:20]

SimCLR 是一个简洁而成功的框架：对每张图施加两个随机变换得到正样本对，把 mini-batch 里 $N$ 张图变成 $2N$ 个样本；对任意样本，与它同源的另一个是正样本，其余全是负样本。特征先经一个**线性／非线性 projection head** 投到 $z$ 空间，在 $z$ 上算 cosine similarity 与 InfoNCE 损失（论文用非线性投影）。

两个关键设计选择被讲者点明：一是**为何不直接在表示 $h$ 上算对比损失**——对比目标会丢掉一些与对比无关的额外信息，于是保留 $h$ 作为表示，另投影出 $z$ 专门承载对比损失；二是**需要巨大 batch**（受显存约束不总能满足）。生成正样本的数据增强应有尽有：

![SimCLR数据增强策略示意图，展示了裁剪、旋转、颜色扰动等10种生成正样本的方法](/coursenotes/cs231n-deep-learning-for-cv-sp25/l12-self-supervised-learning/kf_0019_01-06-57.jpg)

画面列出 10 种操作：原始图、crop and resize、crop/resize + flip、color distort (drop)、color distort (jitter)、rotate $\{90^\circ,180^\circ,270^\circ\}$、cutout、Gaussian noise、Gaussian blur、Sobel filtering。SimCLR 完全不用标签，只在特征上训一个线性分类器，就超过了此前所有工作，结果可与全监督框架相当——代价是需要更大的网络（要学更通用的特征）。讲者顺带提醒，作业 3 有一道 SimCLR 相关题，其定义与此处标准版本略有不同，务必按作业说明来做。

### 9.3 MoCo、MoCo v2 与其他框架 [71:20 - 74:34]

大 batch 不总能实现，于是有了 **MoCo（momentum contrast）**：不再把负样本全部塞进当前 batch，而是维护一个 **queue（队列）**，保存跨若干次迭代的历史负样本，从而**把 mini-batch 大小与负样本数量解耦**，可以支持极多负样本。代价是队列里的样本已不在当前 batch，无法对其反向传播，因此必须把编码器拆成两路：正样本走 **query encoder**（正常梯度下降更新），负样本走 **key / momentum encoder**，后者不用梯度下降，而按动量规则缓慢跟进：

$$\theta_k \leftarrow m\,\theta_k + (1-m)\,\theta_q$$

（讲者描述为用动量 $m$ 把 key 编码器朝 query 编码器缓慢推进。）**MoCo v2** 把 SimCLR 的非线性投影头与数据增强，同 MoCo 的队列／解耦机制结合，性能大幅提升。此外还有 **CPC（contrastive predictive coding）**、**MoCo v3**，以及被广泛使用的 **DINO**——它架构类似 MoCo，却是 student-teacher 自蒸馏，严格说不算对比学习：

![DINO自监督学习框架图，展示Student-Teacher网络结构、EMA更新机制及损失函数计算流程](/coursenotes/cs231n-deep-learning-for-cv-sp25/l12-self-supervised-learning/kf_0020_01-14-07.jpg)

画面给出 DINO（Caron et al., 2021《Emerging Properties in Self-Supervised Vision Transformers》）的要点：Teacher 通过 EMA 从 Student 更新参数 $\theta_{\text{teacher}} \leftarrow \tau\,\theta_{\text{teacher}} + (1-\tau)\,\theta_{\text{student}}$，损失为交叉熵形式的 $-p_2\log p_1$（Teacher 输出还会做 centering 与 sharpening）。讲者把它留作后续讲座的话题，并指出这类框架也常用于从图像乃至视频中提取特征。

## 要点回顾

- 自监督学习的动因：大规模训练需要海量人工标注（分割要逐像素），目标是绕过标签、用无标注数据训出能提取好特征的编码器。
- 两阶段范式：无标注数据上定义 pretext task（标签由数据自动生成，可看作无监督任务但用分类／回归等监督目标训练 encoder + decoder/classifier/regressor），再把训练好的编码器迁移到只有少量标注的 downstream task，末端接浅层／线性网络。
- 好的预训练任务要足够通用、无需人工标注；四大例子是图像补全、旋转预测、拼图、着色；该范式也是 GPT-4 等大语言模型、语音、机器人能用原始数据训练的原因。
- 评估五维度：预训练任务表现、表示质量（线性评估协议／聚类／t-SNE）、鲁棒性与泛化、计算效率、下游任务与迁移性能——其中下游任务表现最重要。
- 旋转预测 RotNet（Gidaris et al. 2018）：$0/90/180/270$ 度 4 分类，靠「视觉常识」逼出特征；PASCAL VOC 上 RotNet 70.87/72.97/54.4/39.1 远胜随机初始化 39.2/53.3/43.4/19.8，接近 ImageNet 全标签 78.9/79.9/56.8/48.0；自监督注意力图比监督模型覆盖更整体的区域。
- 拼图：3×3 网格以中心 patch 为参考做 8 分类；预测完整排列因 $9!$ 太大，改用差异最大的 64 个排列查找表做 64 分类（2016）。
- Inpainting（Pathak et al. 2016）与着色：掩码重建损失只在被遮区域计算，加对抗损失让结果更真实；Lab 色彩空间由 $L$ 通道 $X\in\mathbb{R}^{H\times W\times1}$ 预测 $ab$ 通道 $\hat{Y}\in\mathbb{R}^{H\times W\times2}$；Split-brain Autoencoder 双分支交叉预测，可扩展到 RGB↔深度(HHA)。
- 视频着色（Vondrick et al. 2018）：用 reference frame 经 attention 给后续帧上色，保持颜色时间一致性，并隐式学会无标签跟踪物体／关键点。
- MAE（He et al. 2021）：ViT 切 patch、均匀采样 75% 掩码、编码器只看 25% 可见块、非对称 encoder-decoder、可学习 shared mask token、只对掩码块算 MSE；75% 与 random masking 最优，可用大编码器，优于 DINO/MoCo v3；linear probing 衡量表示质量，full fine-tuning 挖掘模型潜力。
- 对比学习：InfoNCE 等价于 N 路 softmax 交叉熵，其相反数是 $x$ 与 $x^+$ 互信息的下界、负样本越多界越紧，故需大 batch；SimCLR 用两个变换 + 非线性投影头 + 10 种数据增强 + 大 batch；MoCo 用 queue 与动量编码器 $\theta_k\leftarrow m\theta_k+(1-m)\theta_q$ 解耦 batch 与负样本数，MoCo v2 融合两者；另有 CPC、MoCo v3、DINO（student-teacher + EMA + $-p_2\log p_1$）。

## 视频来源

https://www.youtube.com/watch?v=4howBU7THbM
