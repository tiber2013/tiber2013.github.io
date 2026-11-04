---
title: "Course Note 8 · 单目深度估计与特征跟踪：从几何对应到学习出的表示"
description: "CS231A Course Notes 8 中文全译：深度估计的双目背景与视差-深度关系，监督、无监督、自监督单目深度估计的损失设计，特征跟踪的动机，以及学习稠密描述子与像素对比损失。对应第 11–12 讲。"
---

# Course Note 8 · 单目深度估计与特征跟踪

> **Stanford CS231A 计算机视觉：从三维感知到三维重建 · Course Notes 8** ｜ 原文作者：Bryan Chiang、Jeannette Bohg（2022 年 2 月 9 日）｜ 对应课件：第 11 讲 Monocular Depth Estimation & Feature Tracking、第 12 讲 Learning-based Stereo & Monocular Depth Estimation & Feature Tracking Cont. ｜ 本课无公开视频，笔记依据官方 Course Notes 全文翻译

## 这篇在讲什么

这篇 Course Note 是课程从「几何」走向「学习」的分水岭。前面的讲次用对极几何、三角化等多视图几何工具求解对应与深度；本篇则把同一批问题交给表示学习（representation learning）：用无监督与自监督的方式从高维感官数据中学出中间表示，再服务于下游任务。主线有两条。第一条是单目深度估计（monocular depth estimation）：先回顾双目视差如何给出深度 $z = \frac{fb}{d}$，再依次展开监督、无监督、自监督三种范式，重点是各自的损失设计——重建损失、平滑损失、左右一致性损失与光度误差，并说明每一代方法如何逐步摆脱对真值深度标注的依赖。第二条是特征跟踪（feature tracking）：从手工描述子的稀疏与脆弱出发，引出学习出的稠密描述子（learned dense descriptors）与像素对比损失，并介绍如何用机械臂自主采集真值对应、以及跨物体损失与类别一致性带来的描述子结构。

## 1 概述

在上一部分中，我们讨论了表示学习的思想：它利用无监督与自监督方法，从高维感官数据中学习一种中间的、低维的表示。这些学习到的特征随后可用于求解下游的视觉推断任务。在本篇中，我们将考察表示学习在两个常见计算机视觉问题中的体现：单目深度估计（monocular depth estimation）与特征跟踪（feature tracking）。

## 2 单目深度估计

### 2.1 背景

深度估计（depth estimation）是计算机视觉中一个常见的构建模块，对攻克更复杂的任务至关重要，例如三维重建、机器人抓取中的空间感知、以及自动驾驶车辆的导航。深度估计有许多主动式（active）方法，如结构光立体（structured light stereo）与激光雷达（LIDAR，输出三维点云）；但这里我们聚焦于被动式（passive）深度估计，因为它不需要专门的、可能昂贵的硬件，并且在室外场景中往往表现更好。

我们可以把深度估计看作对应问题（correspondence problem）的一个特例，而对应问题是计算机视觉中的基本问题。它要找出：一个物理三维点投影到三维场景的多幅二维图像上之后，与之对应的那些二维位置。这些二维帧可以用单目相机或立体相机从多个视点采集。

**图 1**：立体相机的对极几何（epipolar geometry）配置。

求解对应的一条途径是对极几何，如图 1 所示，课程前文已经见过。回顾一下：给定相机中心 $O_1$ 与 $O_2$，以及场景中一个三维点 $P$，$p$ 与 $p'$ 分别表示 $P$ 在左、右相机像平面上的投影。给定左图中的 $p$，我们知道右图中的对应点 $p'$ 必定落在右相机的极线（epipolar line）上的某处；极线定义为像平面与对极平面的交线。这就是极线约束（epipolar constraint），它被两相机之间的基础矩阵（fundamental matrix，或本质矩阵 essential matrix）所刻画：因为 $F$ 给出了已知的极线。在深度估计的语境下，我们通常假设面对的是立体配置且图像已校正（rectified）。此时极线是水平的，视差（disparity）定义为两个对应点之间的（水平）距离，即 $d = p_u' - p_u$，亦即 $p_u + d = p_u'$（注意对一切 $P$ 都有 $p_u'  \gt p_u$）。

**图 2**：校正后的配置：像平面平行，极点（epipole）位于无穷远。

于是可以看到，视差与深度（depth）之间存在简单的反比关系；深度定义为 $P$ 相对于相机中心的 $z$ 坐标。利用图 3 所示的相似三角形可以得到 $z = \frac{fb}{d}$（图 3 中给出的相似三角形关系为 $\frac{d}{b} = \frac{f}{z}$），其中 $f$ 是相机焦距，$b$ 是两相机之间基线（baseline）的长度（图 2 中的黄色虚线）。假设 $b$ 与相机内参（camera intrinsics）$K$ 已知，那么只要能在两幅校正图像（如图 2）之间找到对应，就知道它们的视差，从而知道深度。为 $p$ 寻找对应 $p'$ 的一种做法是：在另一幅图中沿极线做简单的一维搜索，用像素或图像块（patch）的相似度来确定最可能的 $p'$ 的位置。然而这种朴素方法在真实图像上会碰到遮挡（occlusion）、重复纹理、以及同质区域（即缺乏纹理）等问题。于是我们转向现代表示学习方法。

![图 3：深度与视差的关系。图中上下两个相机相距基线 b，焦距为 f，空间点 P 在两个像平面上的横坐标分别为 x1 与 x2，视差 d = x1 − x2；左下角直接写出相似三角形关系 d/b = f/z，即 z = fb/d 的出处。这张图把「找对应 → 得视差 → 得深度」的整条几何链条压缩在一页里。](/coursenotes/cs231a-computer-vision-3d-sp25/cn08-monocular-depth-tracking/fig_bb8ed70142.jpg)

### 2.2 监督估计

这里我们聚焦于**单目（单视图）深度估计**任务：测试时只有一幅图像可用，且对场景内容不作任何假设。与之相对，**立体（多视图）深度估计**方法在推断时使用多幅图像。单目深度估计是一个欠约束（underconstrained）问题：从几何上讲，不可能确定图像中每个像素的深度。然而人类单眼就能很好地估计深度，靠的是透视、尺度缩放、以及光照与遮挡带来的外观等线索。因此，若能利用这些线索，计算机也应当能仅凭单幅图像推断深度。全监督学习方法（如图 4）训练模型（CNN）在「真值深度—RGB 相机帧」配对数据上学习逐像素视差预测 [8, 11]。训练损失刻画预测深度与真值深度之间的相似度，学习方法的目标是最小化该损失。由于单目方法只能恢复出相差一个尺度的深度（depth up-to-scale），[1] 提出对此前的单目方法改用尺度不变误差（scale-invariant error）。

**图 4**：[1, 8, 11] 所用的常规监督学习配置。图取自 [3]。

### 2.3 无监督估计

监督学习方法取得了不错的结果，但局限于那些能拿到大量真值深度数据的场景类型。这催生了无监督学习方法：它们只需要输入的 RGB 帧数据和一台内参已知的立体相机，从而免去了昂贵的标注工作。这里我们以 [3] 提出的方法作为案例研究。基础无监督形式不把「重建深度与真值深度之差」当作损失，而是借助自编码器（autoencoder）把问题改写为图像重建：最小化输入参考图像与重建版本 $\tilde{I}^l$ 之间的差异。

**图 5**：无监督基线网络。可微采样器使端到端优化成为可能。

基线网络如图 5 所示，只重建左图 $I^l$。网络的输入是左帧 $I^l$。一个 CNN 把左帧映射为输出 $d^l$，即把右图 $I^r$ 扭曲（warp）到左图所需的视差（位移）值。这些视差值随后作为中间表示，通过从右图采样来重建左图 $\tilde{I}^l$。本可以直接按 $\tilde{I}^l(u,v) = I^r(u - d^l(u,v), v)$ 从右图采样，但 $d^l(u,v)$ 不一定是整数，精确新位置上的像素可能并不存在。为了进行端到端优化来训练网络，使用了一个完全（次）可微的双线性采样器（bilinear sampler）[6]（原文此处交叉引用编号缺失，写作 Figure ??）。

注意：训练时左、右两幅图像都被使用，但测试时推断左对齐深度只需要左图。网络架构是全卷积的，由编码器接解码器构成，输出多张空间尺度逐倍放大的视差图。例如，若第一张视差图分辨率为 $(D_h, D_w)$，则第二张输出视差图分辨率为 $(2D_h, 2D_w)$。

在基线之上，[3] 提出了一种同时重建左、右两帧的新架构，如图 6 所示，它允许引入更复杂的损失项以提升视差的图像质量。

![图 6：[3] 提出的无监督网络配置。左帧经 CNN（编码器-解码器）同时输出左对齐与右对齐两张视差图，可微采样器用它们分别从右图、左图重建成像颜色，再与目标颜色计算外观损失；图上还标出左右一致性损失（LR Loss）与平滑损失（Smoothness Loss）的作用位置。整张图就是式 (1) 三项损失的可视化索引。](/coursenotes/cs231a-computer-vision-3d-sp25/cn08-monocular-depth-tracking/fig_1c606a609d.jpg)

CNN 接收输入左帧 $I^l$，计算左对齐视差 $\tilde{d}_{i,j}^l$ 与右对齐视差 $\tilde{d}_{i,j}^r$。右对齐视差图包含从左帧重建右帧所需的水平位移值，左对齐视差同理。使用前面的采样器，我们从 $d^l$ 重建 $\tilde{I}^l$、从 $d^r$ 重建 $\tilde{I}^r$，并在左、右输入图像与重建图像（共四幅图像）之间计算损失。总损失 $C$ 是各尺度 $s$ 上损失之和，$C = \sum_s C_s$：

$$
C_s = \alpha_{ap}(C_{ap}^l + C_{ap}^r) + \alpha_{ds}(C_{ds}^l + C_{ds}^r) + \alpha_{lr}(C_{lr}^l + C_{lr}^r)\tag{1}
$$

可以看到共有三个分量。每个分量前面有一个缩放系数，且都有左、右两个变体；因此下面各式虽然描述的是左重建图像的流程，但右图像的各对应项也以同样方式计算，只是把 $\tilde{I}^l$ 换成 $\tilde{I}^r$。

第一项 $C_{ap}$ 是重建损失（reconstruction loss）：

$$
C_{ap}^l = \frac{1}{N}\sum_{i,j} \alpha \frac{1 - \mathrm{SSIM}(I_{ij}^l, \tilde{I}_{ij}^l)}{2} + (1-\alpha)\|I_{ij}^l, \tilde{I}_{ij}^l\|\tag{2}
$$

它遍历位置 $i, j$ 处的每个像素，计算一个加权和：1）重建图像与真值图像之间的 L1 差异；2）负的结构相似性指标（Structural Similarity Index, SSIM），它通过图像块的亮度、对比度与结构的组合，计算以该像素为中心的两个对应图像块之间的相似度。然后取平均。论文中 $\alpha = 0.85$，说明 SSIM 比 L1 差异被赋予更高的权重。

第二项 $C_{ds}$ 是平滑损失（smoothness loss）：

$$
C_{ds}^l = \frac{1}{N}\sum_{i,j} |\partial_x d_{ij}^l| e^{-\|\partial_x I_{ij}^l\|} + |\partial_y d_{ij}^l| e^{-\|\partial_y I_{ij}^l\|}\tag{3}
$$

可以看到，该损失同样遍历每个像素 $x, y$，并在 x 与 y 两个方向上惩罚视差图的梯度 $(\partial_x d_{ij}^l, \partial_y d_{ij}^l)$。其效果是抑制视差的突变（高梯度），得到更平滑的视差图。然而，在物体边缘处——从某一深度的一个物体过渡到另一深度的另一个物体时——我们恰恰希望视差有较大变化。为此，若在原图中检测到边缘，就放宽平滑惩罚：如果图像梯度很大，则 $e^{\|\partial_x I_{ij}^l\|}, e^{\|\partial_y I_{ij}^l\|}$ 会很小（译注：原页此处指数上确无负号，与式 (3) 中的权重 $e^{-\|\cdot\|}$ 对照应为原文笔误，译文从原文）。注意：对图像梯度取的是 L2 范数，而对视差图梯度取的是 L1 范数。

第三项 $C_{lr}$ 是左右一致性损失（left-right consistency loss）：

$$
C_{lr}^l = \frac{1}{N}\sum_{i,j} |d_{ij}^l - d_{ij + d_{ij}^l}^r|\tag{4}
$$

这里的直觉是：两个像平面上对应像素之间的绝对距离（即视差），无论按「从右到左」还是「从左到右」计算都应当相同。因此应当惩罚网络输出的左、右图像预测视差之间的不一致。具体做法是遍历每个像素 $i, j$，计算左对齐视差 $\tilde{d}_{i,j}^l$ 与对应右对齐视差 $d_{i, j + d_{ij}^l}^r$ 之间的 L1 距离。注意这里是用视差去「采样」视差图，而不是采样图像。在结果部分，作者展示了这套无监督配置既超过基线，也超过已有的最先进全监督方法。

### 2.4 自监督估计

无监督方法之后，自监督学习被引入深度估计；这里我们回顾一篇后续工作 [4]。

在这里，深度估计问题被框定为新视角合成（novel view synthesis）问题。给定场景的一幅目标图像，学习出的流水线要预测该场景从另一个视点看过去会是什么样子。深度被用作获得新视角的中间表示，因此测试时可以从流水线中抽出深度图供其他任务使用。在单目配置下，我们依赖单目视频作为监督信号：目标图像是时刻 $t$ 的彩色帧 $I_t$，而其他视点的图像（即源视图）是时间上相邻的帧 $I_{t'} \in \{I_{t-1}, I_{t+1}\}$。由于我们是从当前输入预测未来与过去的输入，而不是重建整个原始输入，因此这是自监督学习的一个例子。

**图 7**：自监督深度估计流水线。(a) 是提取深度图的深度网络架构，(b) 预测逐帧的位姿变换，(c) 描述在所有帧上重投影时深度的歧义性。

![图 7：自监督深度估计流水线。左侧 (a) 深度网络把彩色帧 I_t 编码-解码为深度图 D_t，(b) 位姿网络由相邻帧 I_(t,t') 回归相对位姿 T_(t→t')；右侧 (c) 用多帧石柱图像与误差-深度曲线说明核心思想：最优深度应使第一帧深度函数在所有图像上的光度误差最小。三个子图合起来就是式 (5)(6) 光度损失的完整计算图。](/coursenotes/cs231a-computer-vision-3d-sp25/cn08-monocular-depth-tracking/fig_c23199abc0.jpg)

流水线如图 7 所示。首先获得中间深度表示：给定彩色输入 $I_t$，让它通过一个卷积编码器-解码器架构得到深度图 $D_t$，见图 7a。并行地，如图 7b 所示，遍历源过去帧与未来帧 $I_{t'}$，计算相对位姿（relative pose）$T_{t \to t'}$，表示从 $I^t$ 到 $I^{t'}$ 的变换。假设所有目标视图与源视图共用同一相机内参 $K$，就可以通过重投影（reprojection）获得 $t-1, t+1$ 的新视图，见图 7c。给定 $K$ 与所有像素的预测深度 $D_t$，可以把特定的二维图像坐标 $(u, v)$ 反投影（backproject）到三维点位置。由于已知相对位姿 $T_{t \to t'}$，再把这个三维点投影到源视图的像平面上，就得到新图像平面 $I_{t'}$ 上的二维坐标 $(u', v')$。由于手握单目视频，真值 $I_{t'}$ 是已知的。于是对 $I_{t'}$ 的「合成」视图为 $\tilde{I}_{t'}(u', v') = I_t(u, v)$。「合成」视图 $\tilde{I}_{t'}$ 与真值视图 $I_{t'}$ 的图像块在视觉上应当相似，这一点用**光度误差**（photometric error）来度量，定义为

$$
pe(I_a, I_b) = \frac{\alpha}{2}(1 - \mathrm{SSIM}(I_a, I_b)) + (1-\alpha)\|I_a - I_b\|_1\tag{5}
$$

光度误差寻找最优深度图 $D_t$，使得每个二维坐标上的重投影误差 $pe(\tilde{I}_{t'}(u', v'), I_{t'}(u', v'))$ 最小。然后对 $(t, t')$ 配对（$t' \in \{t-1, t+1\}$）把光度误差求和：

$$
L_p = \sum_{t'} pe(I_t, I_{t' \to t})\tag{6}
$$

（译注：式 (6) 中的记号 $I_{t' \to t}$ 表示把源视图扭曲（warp）回目标视图，与前文「由目标视图合成源视图 $\tilde{I}_{t'}$」的叙述方向相反；此处按原文照译。）与上一篇论文类似，光度误差（式 5）是结构相似性指标与 L1 差异的加权组合。在结果部分，作者展示了这套自监督配置优于已有的无监督方法及其他自监督方法。

## 3 特征跟踪

（译注：原文本节仅有标题、无正文，实质内容写在下一节「4 动机」中；译文保持原文的章节排布。）

## 4 动机

给定一个图像序列，**特征跟踪**（feature tracking）的任务是在所有图像中跟踪一组二维点的位置，如图 8 所示（原文此处交叉引用编号缺失，写作 Figure ??）。与深度估计一样，我们可以把特征跟踪看作在图像序列上求解对应问题的又一个实例。

**图 8**：特征点随时间的跟踪。（图取自延世大学（Yonsei Univ.）课程幻灯片。）

特征跟踪可用于追踪场景中物体的运动。我们对场景内容或相机运动不作任何假设：相机可以运动也可以静止，场景中可以包含多个运动或静止的物体。

特征跟踪的难点在于识别哪些特征点能够在帧间被高效地跟踪。由于相机运动（特征可能完全消失）、阴影或遮挡，图像特征的外观在帧间可能发生剧烈变化。随着特征跟踪的外观模型不断更新，微小误差也会累积，导致漂移（drift）。我们的目标是识别出那些能够被轻松、一致地跟踪的显著区域（称为特征 feature，有时也称关键点 keypoint），然后套用简单的跟踪方法持续地找到这些对应。

**图 9**：用于特征跟踪的描述子（descriptor）。

传统上，图像中那些容易跟踪的显著特征是用手工设计的方法检测与跟踪的 [5, 10, 9, 12, 7]。具体地，这些好特征随后需要被编码为所谓的描述子，使其便于与其他图像中的特征快速匹配，即寻找对应。这些方法还是稀疏的：只对图像中的一部分像素给出描述子。在本节中，我们将看到表示学习如何被用来**学习**图像特征的描述子，而不是手工设计它们。

在图 9 中，$D(k)$ 给出像素 $k$ 的一个 D 维表示（通常融入邻域信息）。由于 $k, k'$ 对应同一个三维点，我们期望它们在视觉上相似，因此即使两幅图像采自不同视点，其描述子也应当相同：$D(k) = D(k')$。于是可以基于描述子之间的相似度来匹配不同帧中的像素。

## 5 学习稠密描述子

我们考察 [2] 提出的学习稠密描述子（learned dense descriptors）的方法。

![图 10：稠密描述子的表示。左侧是一张 RGB 输入图（W×H×3），经映射 f(·) 后，右侧输出每个像素一个 D 维描述子（W×H×D）；为便于可视化，D 维描述子经降维后映射回 RGB 颜色，于是物体表面呈现出平滑连续的彩色场。这张图直观说明「稠密」的含义：每个像素都有描述子，而非仅稀疏关键点。](/coursenotes/cs231a-computer-vision-3d-sp25/cn08-monocular-depth-tracking/fig_a754d61cec.jpg)

给定一幅输入彩色图像，我们希望学习一个映射 $f(\cdot)$，为彩色图像中的每个像素输出一个 D 维描述子。这里的「稠密」意味着输入图像中每个点都有描述子，而不只是一个稀疏子集。在图 10 的可视化中，D 维描述子通过降维被映射到 RGB。实践中 $f(\cdot)$ 是一个学习出的、具有卷积编码器-解码器架构的神经网络。网络在同一物体不同视图的图像对 $(I_a, I_b)$ 上训练，使用**像素对比损失**（pixel-contrastive loss）：它试图镜像像素层面的「对比」——让相似描述子之间的距离最小化、让不同描述子之间的距离最大化。

**图 11**：匹配对的损失。蓝色箭头表示箭头两端点之间的匹配对应关系。

我们假设对给定的图像对，已有一份对应关系列表（这里称为匹配 matches）。运行网络计算图像中所有点的描述子。对每个真值匹配，计算两个对应点上描述子之间的 L2 距离。我们希望最小化描述子空间中的距离 $D(I_a, u_a, I_b, u_b)^2$：

$$
L_{\mathrm{matches}}(I_a, I_b) = \frac{1}{N_{\mathrm{matches}}}\sum_{N_{\mathrm{matches}}} D(I_a, u_a, I_b, u_b)^2\tag{7}
$$

**图 12**：非匹配对的损失。蓝色箭头表示一对不对应的点。

对于对比的部分，我们还计算非匹配对（彼此不对应的点对）的损失项。这里我们希望最大化非对应点之间的距离（max 操作把该距离最多推大到上限 $M$，即最大距离），即

$$
L_{\mathrm{non-matches}}(I_a, I_b) = \frac{1}{N_{\mathrm{non-matches}}}\sum_{N_{\mathrm{non-matches}}} \max(0, M - D(I_a, u_a, I_b, u_b)^2)\tag{8}
$$

假设 $u_a$ 的真实对应已知，寻找非匹配对就很容易：只需从 $I_b$ 中任意采样那些不是对应点的点。注意：在图 12 中 $u_b$ 指非匹配点，而在图 7 中 $u_b$ 指匹配点（译注：按上下文此处「图 7」应为图 11，译文从原文）。总损失即两者之和：

$$
L(I_a, I_b) = L_{\mathrm{matches}}(I_a, I_b) + L_{\mathrm{non-matches}}(I_a, I_b)\tag{9}
$$

挑战在于：如何以极少的人工辅助、廉价地大规模获取真值对应。为此，作者提出用机器人装置进行自主的、自监督的数据采集。

![图 13：机械臂采集静止物体的不同视图及对应位姿用于训练。六格画面展示同一机械臂围绕桌面上的物体（杯子、菠萝形物体等）从多个位姿拍摄；由于机械臂正运动学已知，每张照片都自带精确的相机位姿，从而可以自动导出海量真值对应。](/coursenotes/cs231a-computer-vision-3d-sp25/cn08-monocular-depth-tracking/fig_ee5427e0e8.jpg)

用机械臂在多个位姿下拍摄静止物体的图像，如图 13 所示。由于这台精密机械臂的正运动学（forward kinematics）已知，我们拥有「相机位姿—对应视图」的匹配对。利用所有视图进行三维重建，得到物体的三维模型。有了相机位姿、三维点与图像，就可以生成任意多的真值对应。网络使用式 (9) 并结合若干其他技巧进行训练，例如背景随机化（background randomization）、数据增强（data augmentation）与硬负例缩放（hard-negative scaling）。

![图 14：跨物体损失的效果。左、右两幅散点图分别是「不加跨物体损失」与「加跨物体损失」时三个物体（baymax、caterpillar、star_bot）描述子的二维投影：不加时三团点混叠在一起，加上后各自聚成清晰分离的簇；下方两张小图（白色机器人、彩色毛毛虫玩具）对应右图中分离出的簇。](/coursenotes/cs231a-computer-vision-3d-sp25/cn08-monocular-depth-tracking/fig_d3c1d622c8.jpg)

如果只在同一物体的图像对上训练，那么不同物体的学习描述子会本不该重叠却发生重叠，因为它们对应完全不同的实体。若引入跨物体损失（来自两个不同物体图像的像素全部视为非匹配），就能看到描述子空间中形成彼此分明的簇，如图 14 所示。

**图 15**：类别一致的描述子。

与之相反，我们希望同一类别的物体即使外观不尽相同，也表现出相似的描述子。可以看到我们的网络有能力学到这一点：帽子虽有不同颜色与设计，但它们的描述子共享相同的结构与颜色，如图 15 所示。

## References

[1] David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. arXiv preprint arXiv:1406.2283, 2014.

[2] Peter R Florence, Lucas Manuelli, and Russ Tedrake. Dense object nets: Learning dense visual object descriptors by and for robotic manipulation. arXiv preprint arXiv:1806.08756, 2018.

[3] Clément Godard, Oisin Mac Aodha, and Gabriel J Brostow. Unsupervised monocular depth estimation with left-right consistency. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 270–279, 2017.

[4] Clément Godard, Oisin Mac Aodha, Michael Firman, and Gabriel J Brostow. Digging into self-supervised monocular depth estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3828–3838, 2019.

[5] C. Harris and M. Stephens. A combined corner and edge detector. In Proceedings of the 4th Alvey Vision Conference, pages 147–151, 1988.

[6] Max Jaderberg, Karen Simonyan, Andrew Zisserman, et al. Spatial transformer networks. Advances in neural information processing systems, 28:2017–2025, 2015.

[7] David G Lowe. Distinctive image features from scale-invariant keypoints. International journal of computer vision, 60(2):91–110, 2004.

[8] Nikolaus Mayer, Eddy Ilg, Philip Hausser, Philipp Fischer, Daniel Cremers, Alexey Dosovitskiy, and Thomas Brox. A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4040–4048, 2016.

[9] K. Mikolajczyk and C. Schmid. A performance evaluation of local descriptors. IEEE Transactions on Pattern Analysis and Machine Intelligence, 27(10):1615–1630, 2005.

[10] Krystian Mikolajczyk and Cordelia Schmid. Scale & affine invariant interest point detectors. Int. J. Comput. Vision, 60(1):63–86, oct 2004.

[11] Ashutosh Saxena, Min Sun, and Andrew Y Ng. Make3d: Learning 3d scene structure from a single still image. IEEE transactions on pattern analysis and machine intelligence, 31(5):824–840, 2008.

[12] Jianbo Shi and Carlo Tomasi. Good features to track. IEEE Conference on Computer Vision and Pattern Recognition, pages 593–600, 1994.

## 要点回顾

- 深度估计可视为对应问题的特例：校正立体配置下极线水平，视差 $d = p_u' - p_u$，深度与视差成反比 $z = \frac{fb}{d}$（相似三角形 $\frac{d}{b} = \frac{f}{z}$）。
- 沿极线做一维搜索的朴素对应方法会败于遮挡、重复纹理与无纹理区域，这是引入表示学习的动机。
- 单目深度估计是欠约束问题；监督方法 [8, 11] 在真值深度—RGB 配对上训练 CNN 预测逐像素视差，[1] 因单目只能恢复 up-to-scale 深度而提出尺度不变误差。
- 无监督基线 [3] 把深度估计改写为图像重建：CNN 由左图预测视差，可微双线性采样器 [6] 从右图采样重建左图，实现端到端训练。
- [3] 的完整架构同时重建左右两帧，总损失为各尺度之和 $C = \sum_s C_s$，含外观（重建）损失、平滑损失、左右一致性损失三项（式 1）。
- 重建损失（式 2）是负 SSIM 与 L1 差异的加权组合，$\alpha = 0.85$ 偏重 SSIM。
- 平滑损失（式 3）惩罚视差梯度，但用图像梯度的指数权重在物体边缘处放宽惩罚；图像梯度取 L2、视差梯度取 L1。
- 左右一致性损失（式 4）用视差去采样视差图，惩罚左、右预测视差的不一致；该无监督配置超过基线与当时的全监督最先进方法。
- 自监督方法 [4] 把深度估计框定为新视角合成：以单目视频相邻帧为源视图，深度与相对位姿 $T_{t \to t'}$ 共同重投影合成新视图，用光度误差（式 5，SSIM 与 L1 加权）求和为 $L_p$（式 6）。
- 特征跟踪是图像序列上的对应问题；难点在于外观剧变与漂移，目标是找到可稳定跟踪的显著关键点。
- 手工描述子 [5, 10, 9, 12, 7] 稀疏且需人工设计；学习描述子要求同一三维点在不同视点下描述子相同 $D(k) = D(k')$。
- 稠密描述子 [2] 用卷积编码器-解码器为每个像素输出 D 维描述子，以像素对比损失训练：匹配对拉近（式 7）、非匹配对推远至上限 $M$（式 8），总损失为两者之和（式 9）。
- 真值对应由机械臂自主采集：正运动学给出相机位姿，配合多视图三维重建即可生成任意多的对应；训练还结合背景随机化、数据增强与硬负例缩放。
- 跨物体损失使不同物体的描述子聚成分离的簇；同类物体（如不同帽子）则共享相似的描述子结构，体现类别一致性。

## 原文出处

- Stanford CS231A Course Notes 8: Monocular Depth Estimation and Feature Tracking（Bryan Chiang、Jeannette Bohg，2022 年 2 月 9 日；官方 Course Notes PDF）
- 对应课件：第 11 讲 Monocular Depth Estimation & Feature Tracking、第 12 讲 Learning-based Stereo & Monocular Depth Estimation & Feature Tracking Cont.（Jeannette Bohg 讲授，两讲共用一份 deck）
- 文末 References [1]–[12] 照原文列出，见上节。
