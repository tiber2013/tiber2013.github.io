---
title: "Course Note 2 · 单视图计量：从消失点与地平线反推三维世界"
description: "CS231A Course Notes 2: Single View Metrology 中文全译，覆盖二维变换家族与交比不变量、无穷远点与无穷远线、消失点与地平线、由消失点反求方向与平面法向、夹角公式与单幅图像自标定。对应第 4 讲 Single View Metrology。"
---

# Course Note 2 · 单视图计量：从消失点与地平线反推三维世界

> **Stanford CS231A 计算机视觉：从三维感知到三维重建 · Course Notes 2: Single View Metrology** ｜ 原文作者：Kenji Hata 与 Silvio Savarese ｜ 对应课件：第 4 讲 Single View Metrology（2025-04-09，Silvio Savarese 讲授）｜ 本课无公开视频，笔记依据官方 Course Notes 全文翻译（原文 9 页，公式编号 (1)–(14) 全部保留）

## 这篇在讲什么

上一篇笔记（Course Notes 1: Camera Models）讲的是**正向**问题：已知相机的内参与外参，怎样把三维世界的点投到数字图像上；以及怎样用标定板这种「已知结构」把相机参数反推出来。这一篇把问题反过来问：**只有一张图像、并且知道（或设法求出）相机性质时，我们还能不能恢复三维世界的结构？更一般地说，一张图像里到底藏着多少信息？**

推导主线是一条严格的「包含链」：等距变换 ⊂ 相似变换 ⊂ 仿射变换 ⊂ 射影变换，越往后的变换保留的几何性质越少，但越贴近真实成像。正因为射影变换只保留共线性、不保留平行性，「平行线在图像里交于一点」这件事才需要用**无穷远点、理想点、无穷远线**这套齐次坐标语言先讲清楚。把同一套语言搬到三维，就得到**消失点（vanishing point）**与**地平线（horizon line）**，进而得到四条能直接算的公式：$v = Kd$ 由消失点取三维方向、$n = K^T \ell_{\text{horiz}}$ 由地平线取平面法向、含 $\omega = (KK^T)^{-1}$ 的两个夹角公式。最后一节用一个真实庭院照片的例子说明：三个两两垂直平面的消失点，加上「零偏斜、方形像素」的假设，恰好凑够 5 个约束解出 $K$——**只用一张图像就完成了相机标定**。

> **译者说明**：本篇 Course Note 的正文止于「单幅图像自标定」这个例子。课件第 4 讲里常说的「用垂直线与地面消失点量出物体高度」「用参照物定尺度」这类单视图计量应用，在本 Course Note 中并没有对应的小节与推导，因此本译文也不作补写。

## 1 引言（Introduction）

在上一篇课程笔记里，我们讨论了如何用相机的外参与内参性质，把真实的三维世界中的点变换成数字图像上的点。我们也看了怎样利用标定装置（calibration rig）里已知的结构，以及它对应的图像，把这些相机性质推出来。这一次我们要看一个相关的问题：**如果只有一张图像，并且知道拍摄这张图像的相机的性质，我们能否恢复三维世界中已知的结构？** 随后我们会更一般地讨论：从一张图像里究竟能推断出哪些信息。

## 2 二维变换（Transformations in 2D）

为了更好地理解「怎样从图像中学习」，我们首先应当知道二维空间里有哪几种变换。

**等距变换（Isometric transformations）** 是保持距离不变的变换。在最基本的形式下，一个等距变换可以描述为一个旋转 $R$ 加一个平移 $t$。因此，它们的数学定义是

$$
\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

其中 $[x' \quad y' \quad 1]^T$ 是经过等距变换之后得到的点。

**相似变换（Similarity transformations）** 是保持形状不变的变换。直观地说，等距变换能做的一切它都能做，此外还能做缩放。数学上记作

$$
\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} SR & t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}, \quad S = \begin{bmatrix} s & 0 \\ 0 & s \end{bmatrix}
$$

由于它们保持形状，所以也保持长度之比与角度。注意：当 $s = 1$ 时，每一个等距变换都是相似变换的一个特例；但反过来并不成立。

**仿射变换（Affine transformations）** 是保持点、直线以及平行性的变换。对某个向量 $v$，仿射变换 $T$ 定义为

$$
T(v) = Av + t
$$

其中 $A$ 是 $\mathbb{R}^n$ 上的一个线性变换。在齐次坐标（homogeneous coordinates）下，仿射变换通常写成

$$
\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} A & t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

从上面这个式子很容易看出，所有相似变换（因而也包括等距变换）都是仿射变换的特例。

**射影变换（Projective transformations）**，或称**单应变换（homographies）**，是把直线映为直线的任意变换，但它不一定保持平行性。在齐次坐标下，射影变换表示为

$$
\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} A & t \\ v & b \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

我们看到，这个形式是仿射变换的进一步推广：由于加入了 $v$，自由度又多了一些。

射影变换虽然不保持平行性，但因为它把直线映为直线，所以它**保持点的共线性（collinearity）**。此外我们还证明：共线四点的**交比（cross ratio）**在射影变换下保持不变。交比取一条直线上的四个点 $P_1$、$P_2$、$P_3$、$P_4$，计算

$$
\text{cross ratio} = \frac{\|P_3 - P_1\| \, \|P_4 - P_2\|}{\|P_3 - P_2\| \, \|P_4 - P_1\|} \tag{1}
$$

交比在射影变换下的不变性，留作课堂练习。

## 3 无穷远点与无穷远线（Points and Lines at Infinity）

直线对于判定图像中的结构很重要，所以知道直线在二维和三维里的定义都是必不可少的。二维中的一条直线可以用齐次向量 $\ell = \begin{bmatrix} a & b & c \end{bmatrix}^T$ 表示。比值 $-\frac{a}{b}$ 刻画了直线的斜率，比值 $-\frac{c}{b}$ 刻画了直线在 $y$ 轴上的截距。形式化地，二维直线定义为：

$$
\forall p = \begin{bmatrix} x \\ y \end{bmatrix} \in \ell, \quad \begin{bmatrix} a & b & c \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = 0 \tag{2}
$$

一般情况下，两条直线 $\ell$ 与 $\ell'$ 会相交于一点 $x$。这个点定义为 $\ell$ 与 $\ell'$ 的叉积。

*证明。* 给定两条相交直线 $\ell$ 与 $\ell'$，交点 $x$ 应当同时落在两条直线上。因此 $x^T \ell = 0$ 且 $x^T \ell' = 0$。如果我们令 $x = \ell \times \ell'$，那么由叉积的定义，向量 $x$ 与 $\ell$、$\ell'$ 这两个向量都正交。再由正交的定义，就有 $x^T \ell = 0$ 且 $x^T \ell' = 0$。所以这样定义的 $x$ 满足约束条件。$\square$

那平行线的情况呢？日常经验告诉我们这些直线永不相交。不过，这个说法可以改写成：这些直线**在无穷远处相交**。在齐次坐标中，无穷远点写作 $\begin{bmatrix} x & y & 0 \end{bmatrix}^T$。回想一下，欧氏坐标是把所有坐标除以最后一个坐标得到的；而在这里最后一个坐标是零，于是得到的就是一个无穷远点。因此，齐次坐标为「求交点」提供了一个很好的表述方式，即使在平行线的情形下也适用。

现在考虑两条平行线 $\ell$ 与 $\ell'$。两线平行时斜率相等，因此 $\frac{a}{b} = \frac{a'}{b'}$。如果用齐次坐标来计算交点，我们就能验证

$$
\ell \times \ell' \propto \begin{bmatrix} b \\ -a \\ 0 \end{bmatrix} = x_{\infty} \tag{3}
$$

这样我们就确认了直觉：两条平行线相交于无穷远。两条平行线在无穷远处的交点也叫做**理想点（ideal point）**。无穷远点有一个有趣的性质：所有斜率同为 $-\frac{a}{b}$ 的平行线都经过这个理想点，如下式所示：

$$
\ell^T x_{\infty} = \begin{bmatrix} a & b & c \end{bmatrix} \begin{bmatrix} b \\ -a \\ 0 \end{bmatrix} = 0 \tag{4}
$$

无穷远点的概念可以进一步推广，用来定义**无穷远线（lines at infinity）**。考虑两组或更多组平行线，每一组平行线都相交于一个无穷远点 $\{x_{\infty,1}, \dots, x_{\infty,n}\}$。

![两组平行线（红色实线）各自延伸到无穷远处的交点（两个黑点），过这两个无穷远点的蓝色直线即无穷远线 l∞](/coursenotes/cs231a-computer-vision-3d-sp25/cn02-single-view-metrology/fig_p0004_00.png)

**Figure 1：无穷远点构成无穷远线。**

那条穿过所有这些无穷远点的直线 $\ell_{\infty}$ 必须满足 $\forall i,\ \ell_{\infty}^T x_{\infty,i} = 0$。这意味着 $\ell_{\infty} = \begin{bmatrix} 0 & 0 & c \end{bmatrix}^T$。由于 $c$ 可以取任意值，我们可以简单地定义 $\ell_{\infty} = \begin{bmatrix} 0 & 0 & 1 \end{bmatrix}^T$。

如果我们对一个无穷远点 $p_{\infty}$ 施加一个一般的射影变换 $H$，会发生什么？

$$
p' = H p_{\infty} = \begin{bmatrix} A & t \\ v & b \end{bmatrix} \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} p_x' \\ p_y' \\ p_z' \end{bmatrix} \tag{5}
$$

注意 $p'$ 的最后一个元素可能变成非零，这说明射影变换一般会把无穷远点映射成**不再位于无穷远**的点。但对仿射变换来说并非如此：仿射变换把无穷远点仍映射为无穷远点，

$$
p' = H p_{\infty} = \begin{bmatrix} A & t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} p_x' \\ p_y' \\ 0 \end{bmatrix} \tag{6}
$$

现在我们把一个射影变换 $H$ 作用到直线 $\ell$ 上，得到新直线 $\ell'$。所有落在一条直线上的点 $x$ 都必须满足性质 $x^T \ell = 0$。在变换后的空间里，我们知道直线仍然映射为直线，也就是说 $x'^T \ell' = 0$。利用单位矩阵恒等性质，可以得到

$$
x^T I \ell = x^T H^T H^{-T} \ell = 0
$$

如果我们对直线施加射影变换，那么直线上的所有点也随之被变换，即 $x' = Hx$。于是 $x^T H^T H^{-T} \ell = x'^T \ell'$，我们求得直线的射影变换规律是 $\ell' = H^{-T} \ell$。与我们对无穷远点的观察类似，我们发现：**无穷远线在射影变换下不一定还映射为无穷远线**；而仿射变换仍然把无穷远线映射为无穷远线。

## 4 消失点与消失线（Vanishing Points and Lines）

到目前为止，我们在二维里引入了直线与无穷远点、无穷远线的概念。现在我们在三维及其对应的齐次坐标中引入等价的概念。

在三维世界里，我们新遇到的概念是**平面**。一个平面可以表示成向量 $\begin{bmatrix} a & b & c & d \end{bmatrix}^T$，其中 $(a,b,c)$ 构成法向量，$d$ 是原点到平面沿该法向量方向的距离。形式化地，平面定义为满足下式的所有点 $x$：

$$
x^T \begin{bmatrix} a \\ b \\ c \\ d \end{bmatrix} = a x_1 + b x_2 + c x_3 + d = 0 \tag{7}
$$

三维中的直线定义为两个平面的交。由于直线有四个自由度（一个确定的截距位置，加上三个维度上各自的斜率），它们在三维空间里不容易用一个漂亮的表示写出来。更多细节请参阅 Hartley & Zisserman 教材的 3.2.2 节。

不过，点在三维里的定义与在二维里是相似的。三维中的无穷远点同样定义为三维平行线的交点。此外，如果我们对这样一个三维无穷远点 $x_{\infty}$ 施加射影变换，就会在像平面上得到一个点 $p_{\infty}$，而它在齐次坐标下已经不再是无穷远点了。这个点 $p_{\infty}$ 就叫做**消失点（vanishing point）**。可是，消失点能拿来做什么？

我们可以在三维平行线、它们在图像中对应的消失点、以及相机参数 $K, R, T$ 之间，推出一个很有用的关系。设 $d = (a,b,c)$ 是相机参考系中一组三维平行线的方向。这些线相交于一个无穷远点，而这个点在图像上的投影就给出消失点 $v$，其定义为

$$
v = K d \tag{8}
$$

上式的推导留作练习。这个式子可以改写，把方向 $d$ 解出来：

$$
d = \frac{K^{-1} v}{\|K^{-1} v\|} \tag{9}
$$

如果我们把一个平面 $\Pi$ 看作一组组平行线的集合，那么每一组平行线都相交于一个无穷远点。穿过这些无穷远点的那条直线，就是与 $\Pi$ 相关联的无穷远线 $\ell_{\infty}$。无穷远线也可以定义为两个平行平面相交的那条线。把 $\ell_{\infty}$ 经射影变换投到像平面上，它就不再是无穷远线了，而被称为**消失线（vanishing line）**或**地平线（horizon line）** $\ell_{\text{horiz}}$。地平线是穿过图像中那些对应消失点的直线。地平线可以这样计算：

$$
\ell_{\text{horiz}} = H_P^{-T} \ell_{\infty} \tag{10}
$$

![一张室内地面照片被射影变换 M 展开：地面上一组组平行线延伸后交于三个消失点，这三个点都落在同一条水平虚线 horizon 上](/coursenotes/cs231a-computer-vision-3d-sp25/cn02-single-view-metrology/fig_p0006_04.png)

**Figure 2：由一组消失点算出的地平线。**

地平线这个概念，让我们能够像人一样直观地推断出图像中那些在数学上并不显然的性质。例如在 Figure 2 中，地面上的那些线在图像坐标里并不平行，但我们天然地理解它们在三维世界里是平行的。

此外，地平线还能让我们计算出关于世界的有用性质。例如，我们可以在三维中某个平面的法向 $n$ 与图像中对应的地平线 $\ell_{\text{horiz}}$ 之间，推出一个有趣的关系：

$$
n = K^T \ell_{\text{horiz}} \tag{11}
$$

这意味着：如果我们能识别出与某个平面相关联的地平线，并且相机已经标定好，那么我们就能估计出该平面的朝向。在介绍最后一条把消失点与消失线联系起来的关系之前，我们需要先定义**无穷远平面（plane at infinity）** $\Pi_{\infty}$。这个平面由两组或更多组消失线定义，在齐次坐标下由向量 $\begin{bmatrix} 0 & 0 & 0 & 1 \end{bmatrix}^T$ 描述。

![左侧黄色平面标为 Π∞，上面画着两条蓝色消失线；右侧是 x-y-z 坐标系与两组互相平行的浅蓝色平面，每个平面都带一条红色虚线边界](/coursenotes/cs231a-computer-vision-3d-sp25/cn02-single-view-metrology/fig_p0007_00.png)

**Figure 3：两条或更多条消失线（蓝色线）定义了无穷远平面 $\Pi_{\infty}$（黄色平面）。**

我们要介绍的最后一条性质，把三维中的直线、平面与像平面上对应的消失点、消失线联系起来。假设三维中有两组平行线，方向分别为 $d_1$ 与 $d_2$，各自关联着无穷远点 $x_{1,\infty}$ 与 $x_{2,\infty}$；令 $v_1$ 与 $v_2$ 为对应的消失点。那么，$d_1$ 与 $d_2$ 之间的夹角 $\theta$ 可以用余弦定理给出：

![左：三维中两组平行线方向 d1、d2 夹角 θ，延伸交于无穷远点 x1∞、x2∞；中：像平面上的两个消失点 v1、v2；右：相机中心 C 与坐标轴](/coursenotes/cs231a-computer-vision-3d-sp25/cn02-single-view-metrology/fig_p0007_02.png)

**Figure 4：推导两条直线之间的夹角。**

$$
\cos\theta = \frac{d_1 \cdot d_2}{\|d_1\| \, \|d_2\|} = \frac{v_1^T \omega v_2}{\sqrt{v_1^T \omega v_1} \sqrt{v_2^T \omega v_2}} \tag{12}
$$

其中 $\omega = (K K^T)^{-1}$。

我们可以把这个想法进一步推广到三维的平面情形，也就是想要联系三维中不同平面的情形。回想一下，对任意一个平面，我们都能算出它关联的消失线 $\ell_{\text{horiz}}$ 以及它的法向 $K^T \ell_{\text{horiz}}$。因此，我们可以通过计算两个平面各自法向量 $n_1$ 与 $n_2$ 之间的夹角，来确定这两个平面之间的夹角 $\theta$。设有两个平面，其消失线分别为 $\ell_1$ 与 $\ell_2$，我们推出两平面之间的夹角 $\theta$：

$$
\cos\theta = \frac{n_1 \cdot n_2}{\|n_1\| \, \|n_2\|} = \frac{\ell_1^T \omega^{-1} \ell_2}{\sqrt{\ell_1^T \omega^{-1} \ell_1} \sqrt{\ell_2^T \omega^{-1} \ell_2}} \tag{13}
$$

## 5 一个单视图计量的例子（A Single View Metrology Example）

![一张学院庭院照片：草坪与墙面上的线分别延伸交于两个消失点，红色线族交于画面外的 v2，黄色线族交于画面内的 v1](/coursenotes/cs231a-computer-vision-3d-sp25/cn02-single-view-metrology/fig_p0008_03.png)

**Figure 5：两个消失点的示例设定，对应一对互相垂直的平面。**

假设我们能在一张三维世界的图像中识别出两个平面；再假设我们在每个平面上都能识别出一组平行线。这样我们就能在图像中估计出两个消失点 $v_1$ 与 $v_2$。最后，假设我们还知道这两个平面在三维中是互相垂直的。在这种情况下，由公式 (12) 我们知道 $v_1 \omega v_2 = 0$。

*（原文此处写作 $v_1 \omega v_2 = 0$，按公式 (12) 的记号应为 $v_1^T \omega v_2 = 0$；经核对 MinerU 抽取与原页 p008，漏写转置系原文自身，译文保留原样。）*

但回想一下，$\omega$ 依赖于相机矩阵 $K$，而此时 $K$ 很可能是未知的。那么，只知道这两个消失点，是否足以准确估计相机参数？考虑到 $K$ 有 5 个自由度，而 $v_1 \omega v_2 = 0$ 只提供一个约束，我们并没有足够的信息来算出 $K$。如果我们还能为另一个与之正交的平面找到第三个消失点 $v_3$ 呢？那我们就知道 $v_1 \omega v_2 = v_1 \omega v_3 = v_2 \omega v_3 = 0$。由于每一对消失点给出一个约束，我们最终只得到计算 $K$ 所需的 5 个约束中的 3 个。然而，如果我们假设相机是零偏斜（zero skew）且像素是正方形（square pixels）的，那么

![与 Figure 5 同一张庭院照片，增加了绿色线族及其消失点 v3，与红色、黄色线族的 v2、v1 一起构成三个两两垂直方向的消失点](/coursenotes/cs231a-computer-vision-3d-sp25/cn02-single-view-metrology/fig_p0009_00.png)

**Figure 6：三个消失点的示例设定，对应一组两两互相垂直的平面。**

我们就能补上所需的那另外两个约束。在这些假设下，$\omega$ 具有如下形式

$$
\omega = \begin{bmatrix} \omega_1 & 0 & \omega_4 \\ 0 & \omega_1 & \omega_5 \\ \omega_4 & \omega_5 & \omega_6 \end{bmatrix} \tag{14}
$$

如果你仔细观察，会发现 $\omega$ 的定义里有四个变量。然而我们对 $\omega$ 只能知道到相差一个尺度（up to scale）的程度，这使得真正需要求解的变量减少到三个，于是方程可解。一旦求出 $\omega$，我们就可以用 Cholesky 分解算出 $K$。这样，我们就做到了**只用一张图像完成相机标定**！

$K$ 已知之后，我们就能重建场景的三维几何。例如，我们可以计算上面识别出的所有平面的朝向。因此，一张图像完全可以用来揭示它所拍摄场景的大量信息。

## 要点回顾

- 二维变换按「保留性质多少」排成一条链：等距变换（保距离）⊂ 相似变换（保形状，$s=1$ 时退化为等距）⊂ 仿射变换（保点、直线、平行性）⊂ 射影变换/单应（只保直线到直线，不保平行性）。
- 射影变换的矩阵形式 $\begin{bmatrix} A & t \\ v & b \end{bmatrix}$ 是仿射形式的进一步推广：加入了 $v$ 之后多出额外的自由度，正是这部分破坏了平行性。
- 射影变换保持共线性，并保持共线四点的交比不变，交比如公式 (1) 所示；其不变性证明原文留作课堂练习。
- 齐次坐标下无穷远点写作 $\begin{bmatrix} x & y & 0 \end{bmatrix}^T$；两条平行线 $\ell$ 与 $\ell'$ 的交点 $\ell \times \ell' \propto [\,b\ -a\ 0\,]^T = x_{\infty}$ 就是**理想点**，同斜率的所有平行线都过它（公式 (4)）。
- 多个无穷远点共线，就构成无穷远线 $\ell_{\infty} = [\,0\ 0\ 1\,]^T$；一般射影变换会把无穷远点/无穷远线映射到有限处（公式 (5)），而仿射变换不会（公式 (6)）。
- 直线的射影变换规律是 $\ell' = H^{-T} \ell$，与点的 $x' = Hx$ 不同。
- 三维平行线的消失点满足 $v = Kd$，反过来可取到方向 $d = K^{-1} v / \|K^{-1} v\|$；一个平面的无穷远线投到像面就是地平线 $\ell_{\text{horiz}} = H_P^{-T} \ell_{\infty}$，并可由 $n = K^T \ell_{\text{horiz}}$ 得到该平面的法向，即**平面朝向**。
- 无穷远平面 $\Pi_{\infty}$ 由两条或更多消失线定义，齐次坐标下为 $[\,0\ 0\ 0\ 1\,]^T$。
- 夹角公式把测量交给图像：两直线夹角用 $\omega = (KK^T)^{-1}$ 写成公式 (12)，两平面夹角用 $\omega^{-1}$ 写成公式 (13)。
- 单视图计量例子：两个垂直平面 → 1 个约束；三个两两垂直平面 → 3 个约束；再加「零偏斜 + 方形像素」两个假设 → 恰好 5 个约束解出 $K$ 的 5 个自由度。
- 公式 (14) 给出的 $\omega$ 有四个变量，但 $\omega$ 只确定到尺度，实际未知量为三个，因此可解；求出 $\omega$ 后用 Cholesky 分解得到 $K$。
- 全文的结论是：**一张图像 + 几何约束，就足以标定相机并重建场景的三维结构信息**（本篇原文止于此，未展开高度与尺度测量）。

## 原文出处

- Stanford CS231A *Course Notes 2: Single View Metrology*，作者 Kenji Hata and Silvio Savarese，共 9 页，含 Figure 1–6 与公式 (1)–(14)。官方发布在 CS231A 课程 syllabus 页面的 Course Notes 列表中，PDF 文件名为 `02-single-view-metrology.pdf`。
- 本笔记覆盖的课程内容：第 4 讲 Single View Metrology（2025-04-09）。
- 原文正文没有独立的「References / Acknowledgements」小节，只在第 4 节里点名引用了一处：Hartley & Zisserman 教材 3.2.2 节（三维直线的表示与四个自由度）。
- 原文另有两处把推导留作练习：交比在射影变换下的不变性（第 2 节末）、公式 (8) $v = Kd$ 的推导（第 4 节）。
- 公式 (10) 在 VLM 转写稿中漏掉了转置逆的上标（写作 $H_P^T$）；MinerU 抽取与原页 p006 一致作 $H_P^{-T}$，本译文据此恢复为 $\ell_{\text{horiz}} = H_P^{-T} \ell_{\infty}$。
- 公式 (2) 中 $\forall p = \begin{bmatrix} x & y \end{bmatrix}^T \in \ell$ 的 $p$ 在原文（原页 p003 与 MinerU 抽取）中是二维列向量，与等式右侧的三维齐次列向量并置系原文排版，译文按原样保留。
