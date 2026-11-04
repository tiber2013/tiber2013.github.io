---
title: "Course Note 9 · 光流与场景流：运动场、光流约束方程与 Lucas-Kanade"
description: "CS231A Course Notes 9 中文全译，覆盖运动场与光流的区别、亮度恒常与小运动假设、光流约束方程与孔径问题、法向流、Lucas-Kanade 邻域最小二乘与可解性分析。对应第 13 讲。"
---

# Course Note 9 · 光流与场景流：运动场、光流约束方程与 Lucas-Kanade

> **Stanford CS231A 计算机视觉：从三维感知到三维重建 · Course Notes 9** ｜ 原文作者：Bryan Chiang、Jeannette Bohg（2022-02-14）｜ 对应课件：第 13 讲（2025-05-14，Jeannette Bohg 讲授）｜ 本课无公开视频，笔记依据官方 Course Notes 全文翻译

## 这篇在讲什么

这篇 Course Note 回答一个问题：给定一段视频，如何从帧与帧之间的亮度变化估计每个像素的表观运动。推导主线是：先区分运动场（三维真实运动在像平面上的投影）与光流（观测到的表观运动），再在亮度恒常假设与小运动假设下推出光流约束方程 $I_x u + I_y v + I_t = 0$，说明单个像素的约束为什么只能定出法向流（孔径问题），最后用空间平滑假设在 $N \times N$ 邻域内堆叠约束、以 Lucas-Kanade 最小二乘求解，并讨论 $\mathbf{A}^T\mathbf{A}$ 的可解性条件与邻域大小的取舍。需要提醒一句标题问题：官方 syllabus 把这一讲印成「Optimal and Scene Flow」，但课件文件是 `Lecture12_Flow.pdf`、Course Notes 9 的标题是 Optical and Scene Flow，实际主题是光流与场景流，「Optimal」系印刷错误。另需说明：正文实际只展开光流的计算，「场景流」在原文中只以运动场定义里的三维运动量 $\mathbf{x}'$ 出现（式 (1)），原文并未展开场景流本身的估计。

## 1 Overview（概览）

给定一段视频，光流（optical flow）被定义为一个二维向量场，描述由于相机（观察者）与场景（物体、表面、边缘）之间的相对运动而引起的每个像素的表观运动（apparent movement）。相机、场景、或者两者都可能处于运动之中。Figure 1 展示了一只苍蝇沿逆时针方向旋转（以苍蝇自身的视角看）。尽管场景是静止的，表观运动的二维光流却指示出一个绕原点、方向相反（顺时针）的旋转。

![图 1：光流示例](/coursenotes/cs231a-computer-vision-3d-sp25/cn09-optical-scene-flow/fig_890214bafe.jpg)

图 1：光流示例 [3]。三幅子图从左到右依次为：观察者（苍蝇）绕自身轴逆时针旋转、光流的三维表示、光流的二维表示（横轴方位角、纵轴仰角）。场景本身静止，但观察者旋转使二维光流场呈现绕原点顺时针的旋涡状箭头分布，是理解「表观运动」最直观的一例。

## 1.1 Motion field（运动场）

光流不要与运动场（motion field）混为一谈：运动场是一个二维向量场，描述场景中各点的三维运动向量投影到观察者像平面上的结果。Figure 2 在一个简单的二维情形下说明运动场（可以想象从正上方俯视一个三维场景）。二维物体点 $\mathbf{P}_o$ 在观察者 O 看来被投影为像平面上的一个一维点 $\mathbf{P}_i$。若物体点 $\mathbf{P}_o$ 位移了 $\mathbf{V}_o \cdot dt$（称为运动向量，motion vector），则对应的投影一维点移动 $\mathbf{V}_i \cdot dt$。这里的一维运动场就由像平面上所有 i 处的全部速度值 $\mathbf{V}_i$ 组成。

![图 2：二维场景的运动场示例](/coursenotes/cs231a-computer-vision-3d-sp25/cn09-optical-scene-flow/fig_6c775ef005.jpg)

图 2：二维场景的运动场示例 [5]。观察者 O 位于左下，竖直线为像平面；物体点 $\mathbf{P}_o$ 投影为像平面上的 $\mathbf{P}_i$，物体点沿 $\mathbf{V}_o\,dt$ 位移后，投影点沿像平面移动 $\mathbf{V}_i\,dt$。所有这样的 $\mathbf{V}_i$ 汇集起来就是运动场，它由真实三维运动决定，与亮度是否变化无关。

推广到三维场景，像素 $(x, y)$ 处的运动场为

$$
\begin{pmatrix} u \\ v \end{pmatrix} = \begin{pmatrix} \frac{dx}{dt} \\ \frac{dy}{dt} \end{pmatrix} = \mathbf{M} \mathbf{x}'
\tag{1}
$$

其中 $\mathbf{x}' = [\frac{dx}{dt}, \frac{dy}{dt}, \frac{dz}{dt}]^T$ 表示三维点的运动，$\mathbf{M} \in \mathbb{R}^{2 \times 3}$ 包含像素位移对三维点位置的偏导数。

运动场是三维运动投影到像平面上之后的理想二维表示。它是我们无法直接观测的「 ground truth（真值）」；我们只能从带噪声的观测（视频）中估计光流（表观运动）。必须注意，光流并不总与运动场相同。例如，一个均匀球体在固定光源下旋转，它没有光流但运动场非零。相反，一个固定的均匀球体、光源绕它运动，则光流非零但运动场为零。这两种情形如 Figure 3 所示。

![图 3：物理对应与光学对应的区别](/coursenotes/cs231a-computer-vision-3d-sp25/cn09-optical-scene-flow/fig_6ba2cb64ee.jpg)

图 3：物理对应与光学对应（physical vs. optical correspondence）[4]。(a) 球体均匀旋转而光源固定：表面纹理均匀，图像亮度图案不随时间变化，光流场全为零（只有点阵），但运动场非零；(b) 球体固定而光源绕其运动：亮度图案在图像上移动，光流场非零（密集箭头），但运动场为零。这两个反例是区分光流与运动场的关键。

> 译注（关于标题中的「场景流」）：标题里的场景流（scene flow）在原文中对应本节的三维运动 $\mathbf{x}'$——式 (1) 给出三维运动到二维运动场的线性投影关系。原文此处只给出这一投影关系与运动场、光流的区分，未展开场景流（三维运动场）本身的估计推导。

## 2 Computing the optical flow（光流的计算）

我们把视频定义为按时间顺序采集的一串帧。$I(x, y, t)$ 是空间与时间的函数，表示时刻 t 的帧中像素 $(x, y)$ 的亮度（intensity）。在稠密光流（dense optical flow）中，对每个时刻 t、每个像素 $(x, y)$，我们都想计算该像素在 x 轴与 y 轴方向的表观速度，分别记为 $u(x, y, t) = \frac{\Delta x}{\Delta t}$ 与 $v(x, y, t) = \frac{\Delta y}{\Delta t}$。于是每个像素的光流向量为 $\mathbf{u} = [u, v]^T$。在接下来的小节中，我们描述 Lucas-Kanade 方法：它采用半局部（semi-local）的策略，对不同的像素块各自用最小二乘独立求解 $\mathbf{u}$。

由亮度恒常假设（brightness constancy assumption），可以认为同一物体在像平面上的表观亮度在不同帧之间不发生变化。对一个在时间 t 到 $t + \Delta t$ 之间沿 x、y 方向分别移动了 $\Delta x$、$\Delta y$ 的像素，这表示为式 (2)：

$$
I(x, y, t) = I(x + \Delta x, y + \Delta y, t + \Delta t)
\tag{2}
$$

一个常见的简化是取 $\Delta t = 1$（相邻帧），此时速度在数值上等于位移，即 $u = \Delta x$、$v = \Delta y$。于是如课件中那样得到 $I(x, y, t) = I(x + u, y + u, t + 1)$。（译注：原文此处第二个位移写作 $y + u$，按上下文应为 $y + v$ 的排印笔误，译文照原文保留。）

接下来，由小运动假设（small motion assumption），假定帧间运动 $(\Delta t, \Delta y)$ 很小。（译注：原文此处写作 $(\Delta t, \Delta y)$，按上下文应为 $(\Delta x, \Delta y)$ 的排印笔误，译文照原文保留。）这使得我们可以用一阶 Taylor 展开把 I 线性化，如式 (3) 所示。

$$
\begin{aligned}
I(x + \Delta x, y + \Delta y, t + \Delta t) &= I(x, y, t) + \frac{\partial I}{\partial x} \Delta x + \frac{\partial I}{\partial y} \Delta y + \frac{\partial I}{\partial t} \Delta t + \ldots \\
&\approx I(x, y, t) + \frac{\partial I}{\partial x} \Delta x + \frac{\partial I}{\partial y} \Delta y + \frac{\partial I}{\partial t} \Delta t
\end{aligned}
\tag{3}
$$

其中的 $\ldots$ 代表 Taylor 展开中的高阶项，我们在下一行把它们截断掉。将式 (3) 的结果代入式 (2)，就得到光流约束方程（optical flow constraint equation）：

$$
\begin{aligned}
0 &= \frac{\partial I}{\partial x} \Delta x + \frac{\partial I}{\partial y} \Delta y + \frac{\partial I}{\partial t} \Delta t \\
&= \frac{\partial I}{\partial x} \frac{\Delta x}{\Delta t} + \frac{\partial I}{\partial y} \frac{\Delta y}{\Delta t} + \frac{\partial I}{\partial t} \\
&= I_x u + I_y v + I_t
\end{aligned}
\tag{4}
$$

$I_x, I_y, I_t$ 分别是两个空间导数与时间导数的简写。

$$
\begin{aligned}
-I_t &= I_x u + I_y v \\
&= \nabla I^T \mathbf{u} \\
&= \nabla I \cdot \vec{\mathbf{u}}
\end{aligned}
\tag{5}
$$

我们可以把它认作形如 $\mathbf{Ax} = \mathbf{b}$ 的线性系统。$\nabla I = [I_x, I_y]^T \in \mathbb{R}^{2 \times 1}$ 表示亮度的空间梯度，$\vec{\mathbf{u}} \in \mathbb{R}^{2 \times 1}$ 是我们想要求解的光流向量。

然而，由于 $\nabla I$ 是一个胖矩阵（fat matrix，行数少于列数），这是一个欠约束（under-constrained）系统：对两个未知量 $u, v$ 我们只有一个约束方程。这条约束只把自由度从两个减到一个；$(u, v)$ 的解集必须落在式 (5) 给出的一条直线上，如 Figure 4 所示。具体地说，光流约束只能给我们法向流（normal flow）：$\mathbf{u}$ 沿空间梯度 $\nabla I$ 方向的分量。Figure 5 可视化了一幅示例图像在 x 方向与 y 方向上的空间图像梯度。可以看到，梯度幅值高的地方对应图像的边缘。

回顾两个向量 $\mathbf{a}$、$\mathbf{b}$ 的点积为 $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos(\theta)$，其中 $\theta$ 是两向量之间的夹角。两边同除以 $\|\mathbf{b}\|$ 就得到 $\mathbf{a}$ 在 $\mathbf{b}$ 上的投影，如 Figure 6 所示。

于是，式 (6) 表明光流约束给出的是 $\mathbf{u}$ 在 $\nabla I$ 上的投影，也就是法向流。

$$
\frac{\nabla I}{\|\nabla I\|} \cdot \mathbf{u} = \frac{-I_t}{\|\nabla I\|}
\tag{6}
$$

在单个图像像素处，我们得到的是一条直线（即 Figure 4 图内上方所标的方程 $I_x u + I_y v = -I_t$）：

![图 4：(u, v) 的解集是一条直线](/coursenotes/cs231a-computer-vision-3d-sp25/cn09-optical-scene-flow/fig_c35a48c68f.jpg)

图 4：$(u, v)$ 的解集为形如 $y = mx + b$ 的一条直线。图内上方给出直线方程 $I_x u + I_y v = -I_t$；在 $(u, v)$ 平面中解集是一条斜线，从原点向该线所作垂线段长为 $\frac{-I_t}{\|\nabla I\|}$，垂线方向（红色箭头）即法向流，而沿直线方向的分量完全自由。

（原文 Figure 5 给出三幅并排图像 [1]：左为猫的灰度亮度图，中为 x 方向梯度图、度量亮度的水平变化，右为 y 方向梯度图、度量亮度的垂直变化；灰色像素梯度小，黑或白像素梯度大，且大梯度集中在轮廓边缘处——这正解释了为什么法向流只在边缘附近提供有效信息。）

回顾两个向量 $\mathbf{a}$、$\mathbf{b}$ 的点积为 $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos(\theta)$，其中 $\theta$ 是两向量之间的夹角。两边同除以 $\|\mathbf{b}\|$ 就得到 $\mathbf{a}$ 在 $\mathbf{b}$ 上的投影 $\frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{b}\|} = \|\mathbf{a}\| \cos\theta$（原文 Figure 6 以蓝色向量 $\mathbf{a}$、绿色向量 $\mathbf{b}$ 与夹角 $\theta$ 画出了这一投影关系 [2]）。

Figure 4 画出的是 $\mathbf{u}$ 的解空间。不过我们还可以把图像空间叠加在这上面看。法向流（解空间的法方向）与给定像素处图像亮度的空间梯度同向；而解集（那条直线）与空间梯度正交，因此与图像空间中的边缘同向。

于是，虽然我们已知 $\mathbf{u}$ 在 $\nabla I$ 方向上的分量（法向流），却不知道 $\mathbf{u}$ 沿边缘方向的分量。这一点由解集表示：它列出了这个分量所有可能的取值。要找出 $\mathbf{u}$ 的一个可能取值，我们先沿空间梯度方向走一段等于法向流幅值的距离，然后可以再沿边缘方向走任意（未知）的距离——这就是我们剩下的那一个自由度。

这种「不知道像素沿边缘方向移动了多少」的问题也称为孔径问题（aperture problem），如 Figure 7 所示。蓝色方块是不透明的，中心开有一个方形切口，形成一个小孔径。浅灰色矩形是我们的运动物体。左图中灰色矩形位于蓝色方块之前；右图中灰色矩形位于蓝色方块之后。两种情形下矩形都同时向右下方移动。红色箭头标出两种情形下被感知到的运动方向。左边第一种情形中，矩形没有被蓝色方块遮挡，我们可以清楚感知它真实的运动方向。然而右边的情形中，矩形在蓝色方块后面、被遮挡了：矩形的运动方向与左边相同，但我们只感知到一个正向右方的运动。这与我们先前的推导一致：我们只知道矩形沿空间梯度方向（此处为 x 轴）移动了多少，却不知道它沿图像边缘方向（此处为 y 轴）移动了多少。

![图 7：孔径问题的视觉示例](/coursenotes/cs231a-computer-vision-3d-sp25/cn09-optical-scene-flow/fig_a5d699400e.jpg)

图 7：孔径问题的视觉示例。左：灰色矩形在蓝色挡板前方，红色箭头显示其真实的右下方向运动；右：同一矩形藏在挡板后、只透过中心的小方孔可见，孔内只能看到一条竖直边缘，于是感知到的运动只剩水平向右的红色箭头，沿边缘方向的运动分量完全丢失。

为了缓解这个问题，可以使用空间平滑假设（spatial smoothness assumption）：

邻域内的点属于场景中的同一表面，因而共享同一个光流 $\mathbf{u}$。若在当前像素周围定义一个 $N \times N$ 邻域，则 $N^2$ 个像素中的每一个都给出一条约束 $\nabla I(\mathbf{p}_i)^T \mathbf{u} = -I_t(\mathbf{p}_i)$，其中 $p_i = [x_i, y_i]^T$ 是第 i 个像素的位置。假设 $N^2 \gt 2$，我们得到如下方程组：

$$
\begin{aligned}
&\mathbf{A} \mathbf{u} = \mathbf{b}:\\
&\begin{pmatrix} I_x(\mathbf{p}_1) & I_y(\mathbf{p}_1) \\ \vdots & \vdots \\ I_x(\mathbf{p}_{N^2}) & I_y(\mathbf{p}_{N^2}) \end{pmatrix} \begin{pmatrix} u \\ v \end{pmatrix} = -\begin{pmatrix} I_t(\mathbf{p}_1) \\ \vdots \\ I_t(\mathbf{p}_{N^2}) \end{pmatrix}
\end{aligned}
\tag{7}
$$

然后我们最小化误差 $\|\mathbf{A}\mathbf{u} - \mathbf{b}\|$，求这个超定系统的最小二乘解：将式 (7) 两边同乘 $\mathbf{A}^T$，得 $\mathbf{u}_{\mathrm{ls}} = (\mathbf{A}^T \mathbf{A})^{-1} \mathbf{A}^T \mathbf{b}$。我们通过考察如下正规方程（normal equations）来讨论该系统的可解性：

$$
\begin{aligned}
\mathbf{A}^T \mathbf{A} \mathbf{u} &= \mathbf{A}^T \mathbf{b} \\
\begin{pmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{pmatrix} \mathbf{u} &= -\begin{pmatrix} \sum I_x I_t \\ \sum I_y I_t \end{pmatrix}
\end{aligned}
\tag{8}
$$

求和对邻域内的每个像素进行。要使系统可解，$\mathbf{A}^T \mathbf{A}$ 不应因为噪声的存在而过小。例如，低纹理区域的梯度幅值小，导致特征值小，$\mathbf{u}$ 的取值容易被噪声破坏。$\mathbf{A}^T \mathbf{A}$ 还应当是良态的（well-conditioned），使系统对测量误差稳健，因为 $\mathbf{A}^T \mathbf{b}$ 依赖于从图像亮度数据算出的所有梯度。例如在一条边缘上，所有大梯度都指向同一方向，导致一个特征值显著大于另一个（条件数大）。能够无歧义地解出 $\mathbf{u}$ 的好区域是具有方向变化空间结构的高纹理区域：那里在不同方向上都有大梯度，从而 $\mathbf{A}^T \mathbf{A}$ 良态。

为了更直观地理解为什么方向变化的结构有助于解出 $\mathbf{u}$，看 Figure 8。这里有两种情形。灰色圆圈代表我们正在优化的像素邻域。虚线代表物体在前一时刻的位置，实线代表物体在当前时刻的位置，二者描绘出物体的真实运动。在 (c) 中，孔径问题再次出现：只有一个法向流向量（绿色），于是沿约束线（也是绿色）有无数个可能的 $\mathbf{u}$ 解（蓝色）。而在 (b) 中，存在以角点形式出现的方向变化空间结构，可以看作两条方向不同的边缘。此时有两个法向流向量和两条约束线（一条红色、一条绿色）。同时满足两条约束线的光流向量 $\mathbf{u}$ 只有一个可能解，以蓝色标出。因此我们有动力增大邻域尺寸 N，以提高邻域内包含方向变化结构的机会。但这里存在取舍（tradeoff）：我们也可能跨过运动边界、把属于其他表面的像素包含进来，于是 $\mathbf{u}$ 在整个邻域内不再恒定的可能性越来越大。

![图 8：角点结构如何消除孔径问题的歧义](/coursenotes/cs231a-computer-vision-3d-sp25/cn09-optical-scene-flow/fig_5673a7d396.jpg)

图 8：(b) 展示在 (c) 所描绘的孔径问题背景下，方向变化的空间结构为何重要。(c) 的邻域内只有一条边缘，绿色约束线只有一条，蓝色候选光流向量沿它扇形展开、无法唯一确定；(b) 的邻域内含有一个角点，红、绿两条约束线相交，同时满足两条约束的蓝色光流向量唯一。黑色小箭头为邻域内各像素的梯度方向。

在实践中，传统形式的 Lucas-Kanade 并不稳健，原因包括相机运动过大、遮挡，以及对前述一系列假设的违反。

## References（参考文献）

- [1] https://commons.wikimedia.org/w/index.php?curid=10588443 ，Njw000 自有作品，公有领域。图像梯度（原文作「Image gadient」）。
- [2] Nykamp DQ. The dot product. https://mathinsight.org/dot_product , 2021. 访问日期：2022-01-31.
- [3] Stephen J Huston and Holger G Krapp. Visuomotor transformation in the fly gaze stabilization system. *PLoS biology*, 6(7):e173, 2008.
- [4] Bernd Jähne, Horst Haussecker, and Peter Geissler. *Handbook of computer vision and applications*, volume 2. Citeseer, 1999.
- [5] Robyn Owens. Computer vision it412. https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/OWENS/LECT12/node3.html , 1997. 访问日期：2022-01-31.

## 要点回顾

- 光流是像素表观运动的二维向量场；运动场是场景点三维运动向量在像平面上的投影，二者概念不同、数值也不总相等。
- 两个反例刻画区别：均匀球旋转 + 固定光源 ⇒ 光流为零、运动场非零；球固定 + 光源绕行 ⇒ 光流非零、运动场为零（Figure 3）。
- 运动场满足式 (1)：$(u, v)^T = \mathbf{M}\mathbf{x}'$，其中 $\mathbf{x}'$ 是三维点速度、$\mathbf{M} \in \mathbb{R}^{2 \times 3}$；标题中的「场景流」在原文中只以这个三维运动量出现，未展开估计方法。
- 亮度恒常假设给出式 (2)：$I(x, y, t) = I(x + \Delta x, y + \Delta y, t + \Delta t)$；取 $\Delta t = 1$ 时速度在数值上等于位移。
- 小运动假设允许一阶 Taylor 线性化（式 (3)），代回后得到光流约束方程 $I_x u + I_y v + I_t = 0$（式 (4)）。
- 约束方程写成 $\nabla I^T \mathbf{u} = -I_t$（式 (5)）后是一个方程、两个未知量的欠约束系统，解集是 $(u, v)$ 平面上的一条直线。
- 单点约束只能确定法向流，即 $\mathbf{u}$ 在 $\nabla I$ 方向的投影 $\frac{\nabla I}{\|\nabla I\|} \cdot \mathbf{u} = \frac{-I_t}{\|\nabla I\|}$（式 (6)）；沿边缘方向的分量未知，即孔径问题（Figure 7）。
- 空间平滑假设（邻域点同属一个表面、共享同一光流）是原文唯一的「正则化」式手段：它在 $N \times N$ 邻域内堆叠出超定系统 (7)，用最小二乘 $\mathbf{u}_{\mathrm{ls}} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b}$ 求解；原文未涉及 Horn-Schunck 一类全局正则化。
- 可解性看 $\mathbf{A}^T\mathbf{A}$（式 (8)）：低纹理区梯度小、特征值小、易被噪声破坏；单一边缘处梯度同向、条件数大；方向变化的高纹理区（如角点）才良态可解。
- 增大邻域 N 更容易包含角点等方向变化结构（Figure 8(b)），但也更容易跨过运动边界、破坏流恒定假设，这是邻域大小的取舍。
- 传统形式的 Lucas-Kanade 在相机运动大、存在遮挡或假设被违反时并不稳健，原文以此收尾、未给出改进方案。

## 原文出处

- Stanford CS231A Course Notes 9: Optical and Scene Flow（Bryan Chiang、Jeannette Bohg，2022-02-14；官方 syllabus 页面与 PDF 文件名）
- 对应课件：第 13 讲（2025-05-14，Jeannette Bohg；syllabus 将该讲标题印作「Optimal and Scene Flow」，课件文件名为 Lecture12_Flow.pdf，实际主题为光流与场景流）
- 文末 References 按原文列出 [1]–[5]
