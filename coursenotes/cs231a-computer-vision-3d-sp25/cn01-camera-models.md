---
title: "Course Note 1 · 相机模型：从针孔、镜头到完整相机矩阵与标定（对应第 2–3 讲）"
description: "CS231A Course Notes 1: Camera Models 中文全译，覆盖针孔相机、镜头与真实光圈、走向数字图像空间、相机矩阵与齐次坐标、完整相机矩阵、外参、相机标定、畸变处理，以及附录 A 刚体变换与附录 B 弱透视/正射投影。对应第 2 讲 Camera Models 与第 3 讲 Camera Models II and Camera Calibration。"
---

# Course Note 1 · 相机模型：从针孔、镜头到完整相机矩阵与标定

> **Stanford CS231A 计算机视觉：从三维感知到三维重建 · Course Notes 1: Camera Models** ｜ 原文作者：Kenji Hata 与 Silvio Savarese ｜ 对应课件：第 2 讲 Camera Models、第 3 讲 Camera Models II and Camera Calibration ｜ 本课无公开视频，笔记依据官方 Course Notes 全文翻译（原文 17 页，公式编号 (1)–(20) 全部保留）

## 这篇在讲什么

这篇 Course Note 回答一个最基础的问题：**怎么把「相机」写成一个数学模型**。主线是一条完整的投影链：先用针孔几何推出 $P'=[fx/z,\;fy/z]^T$，再承认针孔假设在现实中站不住（光圈不能无限小），引出镜头与近轴折射模型；接着处理「像面坐标」与「数字图像像素坐标」之间的三处差异——原点平移、单位换算、传感器非线性——用齐次坐标把带分母的非线性投影写成矩阵乘法 $P'=MP$，并把 $M$ 分解成内参矩阵 $K$ 与外参 $[R\ T]$，得到 $3\times4$ 投影矩阵的 11 个自由度。后半篇讲**标定**：给定 $n$ 组三维点与像素点的对应，如何靠线性方程组加 SVD 解出 $M$，再显式拆回内参与外参；最后简要给出畸变情形下的处理办法。两篇附录补上这门课的通用数学语言：刚体变换，以及弱透视与正射投影这两个简化模型。

## 1 引言（Introduction）

相机是计算机视觉中最基本的工具之一。它是记录我们周围世界的机制，而它的输出——照片——又被用于各种各样的应用。因此在入门计算机视觉时，我们必须回答的第一个问题是：

> **我们如何为一个相机建模？**

## 2 针孔相机（Pinhole cameras）

先来设计一个最简单的相机系统——一个能够记录三维世界中某个物体或场景图像的相机系统。做法是：在三维物体与照相胶片（或传感器）之间，放一块带小孔（aperture，光圈）的挡板（barrier）。

![针孔相机模型的正式构造：左侧树上一点 P，中间针孔 O 与相机坐标系 i、j、k，焦距 f 标在 O 与像面之间，像面 Π' 上给出 C' 与倒立的 P'](/coursenotes/cs231a-computer-vision-3d-sp25/cn01-camera-models/fig_p0002_00.png)

**Figure 2：针孔相机模型的正式构造。**（原文 Figure 1 标题为「一个简单可用的相机模型：针孔相机模型」，是同一想法的示意版本：物体发出的多条光线被挡板挡住，只有穿过小孔的那一条能到达胶片。）

如 Figure 1 所示，三维物体上的每一个点都向外发出许多条光线。如果没有挡板，胶片上的每一点都会受到物体上每一点发出的光线的影响。有了挡板之后，这些光线中只有一条（或很少几条）能穿过光圈打到胶片上。于是我们就能在三维物体的点与胶片的点之间建立**一一映射**。结果是：胶片借助这个映射被三维物体的一幅「图像」曝光。这个简单的模型就叫做**针孔相机模型（pinhole camera model）**。

针孔相机更正式的构造如 Figure 2。在这套记号里：

- 胶片通常称为**像面或视网膜平面（image or retinal plane）**；
- 光圈称为**针孔（pinhole）** $O$，也叫**相机中心（center of the camera）**；
- 像面到针孔 $O$ 的距离是**焦距（focal length）** $f$。

有时会把视网膜平面放在 $O$ 与三维物体之间、距 $O$ 为 $f$ 的位置，这时它叫做**虚像或虚拟视网膜平面（virtual image / virtual retinal plane）**。注意：物体在像面上的投影，与物体在虚像面上的像，只差一个尺度（相似）变换，本质上完全相同。

那么怎样使用针孔相机？设 $P=[x\ y\ z]^T$ 是针孔相机能看到的某个三维物体上的一点。$P$ 会被映射（或称**投影（projected）**）到像面 $\Pi'$ 上，得到点$^1$ $P'=[x'\ y']^T$。类似地，针孔本身也可以投影到像面上，得到一个新点 $C'$。

> $^1$ 在本 Course Notes 全篇中，撇号上标（例如 $P'$）表示这个点是非上标版本的投影点或配套点。例如 $P'$ 是 $P$ 的投影版本。

这里我们可以定义一个以针孔 $O$ 为原点、记作 $[i\ j\ k]$ 的坐标系，并要求 $k$ 轴垂直于像面且指向像面。这个坐标系通常称为**相机参考系或相机坐标系（camera reference system / camera coordinate system）**。由 $C'$ 与 $O$ 确定的那条直线，叫做相机系统的**光轴（optical axis）**。

回想一下，点 $P'$ 是由三维点 $P$ 在像面 $\Pi'$ 上的投影得到的。因此，只要我们推出三维点 $P$ 与像面点 $P'$ 之间的关系，就能理解三维世界是如何「印」在针孔相机所拍摄的图像上的。注意三角形 $P'C'O$ 与由 $P$、$O$ 和 $(0,0,z)$ 构成的三角形相似，于是由相似三角形定律可得：

$$
P' = \begin{bmatrix} x' & y' \end{bmatrix}^T = \begin{bmatrix} f \frac{x}{z} & f \frac{y}{z} \end{bmatrix}^T \tag{1}
$$

注意，这个针孔模型里有一个很强的假设：**光圈是一个点**。然而在大多数真实场景中，我们不能假设光圈可以无限小。那么改变光圈大小会带来什么效果？

![四幅同一招牌的黑白照片，按光圈直径 2 mm、1 mm、0.6 mm、0.35 mm 排列：光圈越小字迹越清晰，但画面越暗](/coursenotes/cs231a-computer-vision-3d-sp25/cn01-camera-models/fig_p0003_02.png)

**Figure 3：光圈大小对图像的影响。光圈越小，图像越清晰，但也越暗。**

光圈变大时，穿过挡板的光线条数随之增加。透过更多光线后，胶片上的每一点都可能被三维空间中多个点发出的光线影响，图像因此被模糊。虽然我们可能倾向于把光圈尽量做小，但请记住：更小的光圈意味着更少的光线穿过，结果是**更清晰但更暗**的图像。于是我们撞上针孔公式提出的那个根本矛盾：

> 我们能否造出既清晰又明亮的相机？

## 3 相机与镜头（Cameras and lenses）

在现代相机中，上面这个「清晰度 vs 亮度」的冲突由**镜头（lenses）**来缓解——镜头是能会聚或发散光的器件。如果我们用一个位置和尺寸都合适的镜头替换针孔，它就满足这样一条性质：由某个点 $P$ 发出的**所有**光线，经过镜头折射后都会**会聚到像面上的同一个点** $P'$。因此「大部分光线被小光圈挡住」这个问题被消掉了（Figure 4）。

![物体—镜头—胶片三段式光路图：平行于光轴的红线穿过透镜后会聚到 focal point，并继续到达像面上的 P'；下方标出 -z、f、z0，上方标出 z'](/coursenotes/cs231a-computer-vision-3d-sp25/cn01-camera-models/fig_p0004_03.png)

**Figure 5：镜头把平行于光轴的光线会聚到焦点。这个构造同时给出了近轴折射模型，帮我们找到带镜头相机中像面上的点与三维世界的关系。**

（原文 Figure 4 是另一张类似的光路图，它额外强调：树上顶点发出的光线在胶片上会聚得很好，而另一个不同距离上的点就不会完美会聚。）

但请注意，这条性质**并不是对所有三维点都成立**，它只对某个特定的点 $P$ 成立。换另一个点 $Q$，它比 $P$ 更靠近或更远离像面，那么在像中的对应投影就会是模糊的、失焦的。因此镜头存在一个特定的距离，只有在这个距离上的物体才「在焦内」。这个性质也与摄影和计算机图形学里的**景深（depth of field）**概念相关，即相机能拍出清晰照片的有效范围。

相机镜头还有另一个有趣的性质：它把**所有沿平行于光轴方向传播的光线**会聚到一个点，这个点叫做**焦点（focal point）**（Figure 5）。焦点到镜头中心的距离通常也称为**焦距** $f$。此外，穿过镜头中心的光线不发生偏折。于是我们可以得到一个与针孔模型类似的构造，把三维空间中的点 $P$ 与像面上对应的点 $P'$ 联系起来：

$$
P' = \begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} z' \frac{x}{z} \\ z' \frac{y}{z} \end{bmatrix} \tag{2}
$$

这个模型的推导超出了本课范围。不过请注意：在针孔模型里 $z' = f$，而在这个基于镜头的模型里 $z' = f + z_0$。另外，由于这一推导利用了近轴（paraxial）或「薄透镜」假设$^2$，它被称为**近轴折射模型（paraxial refraction model）**。

> $^2$ 设入射光线与镜头光轴的夹角为 $\theta$，近轴假设就是在所有出现 $\sin(\theta)$ 的地方用 $\theta$ 来替代。这个用 $\theta$ 近似 $\sin\theta$ 的做法在 $\theta$ 趋于 0 时成立。

正因为近轴折射模型是用薄透镜假设做出来的**近似**，它会产生若干种像差（aberration）。最常见的一种叫**径向畸变（radial distortion）**：图像的放大率随到光轴的距离而变化，可能变小也可能变大。我们把放大率增大的径向畸变称为**枕形畸变（pincushion distortion）**，把放大率减小的称为**桶形畸变（barrel distortion）**$^3$。径向畸变的成因是镜头不同部位具有不同的焦距。

![三张网格图并排：normal 是正方形网格，pincushion 的网格向内凹陷、边缘线条朝内弯，barrel 的网格向外鼓、边缘线条朝外凸](/coursenotes/cs231a-computer-vision-3d-sp25/cn01-camera-models/fig_p0005_03.png)

**Figure 6：枕形畸变与桶形畸变如何影响图像。**

> $^3$ 桶形畸变通常发生在使用鱼眼镜头时。

## 4 走向数字图像空间（Going to digital image space）

本节讨论从三维空间投影到我们熟悉的数字图像时，必须考虑的参数的细节。所有推导结果都使用针孔模型，但它们对近轴折射模型同样成立。

如前所述，三维空间中的点 $P$ 可以映射（投影）成像面 $\Pi'$ 上的二维点 $P'$。这个 $\mathbb{R}^3 \rightarrow \mathbb{R}^2$ 的映射称为**射影变换（projective transformation）**。但把三维点投影到像面上，与我们在真实数字图像里看到的内容并不直接对应，原因有几条：

1. 数字图像中的点一般处在与像面**不同的参考系**里；
2. 数字图像被划分成**离散的像素**，而像面上的点是连续的；
3. 物理传感器会给映射引入**非线性**，例如畸变。

为了处理这些差异，我们要引入若干额外的变换，使得任何三维世界中的点都能被映射到像素坐标。

### 4.1 相机矩阵模型与齐次坐标（The Camera Matrix Model and Homogeneous Coordinates）

#### 4.1.1 相机矩阵模型简介

相机矩阵模型描述了一组重要参数，它们决定世界点 $P$ 如何被映射为图像坐标 $P'$。顾名思义，这些参数会以矩阵形式表示。先介绍其中几个。

第一组参数 $c_x$ 与 $c_y$ 描述像面坐标与数字图像坐标之间可以差一个**平移**。像面坐标的原点 $C'$ 在图像中心，也就是 $k$ 轴与像面相交的位置；而数字图像坐标的原点通常在图像的**左下角**。因此像面上的二维点与图像里的二维点相差一个平移向量 $[c_x,\ c_y]^T$。把这个坐标系变化考虑进来后，映射变成：

$$
P' = \begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} f \frac{x}{z} + c_x \\ f \frac{y}{z} + c_y \end{bmatrix} \tag{3}
$$

我们要考虑的第二种影响是：数字图像中的点用**像素**表示，而像面上的点用**物理度量**（例如厘米）表示。为了处理这个单位变化，必须引入两个新参数 $k$ 和 $l$。它们的单位大致是 $\text{pixels} / \text{cm}$，对应像面两个轴上的单位换算。注意 $k$ 与 $l$ 可以不同，因为像素的**宽高比不保证是 1**。若 $k = l$，我们通常说相机具有**方形像素（square pixels）**。把前面的映射再改一次：

$$
P' = \begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} f k \frac{x}{z} + c_x \\ f l \frac{y}{z} + c_y \end{bmatrix} = \begin{bmatrix} \alpha \frac{x}{z} + c_x \\ \beta \frac{y}{z} + c_y \end{bmatrix} \tag{4}
$$

> **译注**：对照 (4) 式左右两列即得 $\alpha = fk$、$\beta = fl$。原文并未显式写出这两个等式，此为译者补注。

有没有更好的方式来表达 $P \rightarrow P'$ 这个投影？如果这个投影是线性变换，那它就能写成矩阵乘以输入向量（在这里就是 $P$）的形式。然而从 Equation 4 可以看出，这个投影 $P \rightarrow P'$ **不是线性的**，因为运算中除以了输入参数之一（即 $z$）。尽管如此，把这个投影表示成矩阵—向量乘积对后续推导非常有用。那么，能不能不管它的非线性，仍然用矩阵—向量乘积来表示这个变换？答案是**齐次坐标**。

#### 4.1.2 齐次坐标（Homogeneous Coordinates）

解决这个问题的一个办法是更换坐标系。例如引入一个新的坐标分量，使任何点 $P' = (x', y')$ 变成 $(x', y', 1)$；类似地，任何点 $P = (x, y, z)$ 变成 $(x, y, z, 1)$。这个被扩充出来的空间称为**齐次坐标系（homogeneous coordinate system）**。如前所述，把一个欧氏向量 $(v_1, \dots, v_n)$ 转换成齐次坐标，只需在新增的一维上补一个 1，得到 $(v_1, \dots, v_n, 1)$。注意：一个向量与它的齐次坐标相等，只在最后一个分量等于 1 时成立。因此，从任意的齐次坐标 $(v_1, \dots, v_n, w)$ 转换回欧氏坐标时，得到的是 $\left(\frac{v_1}{w},\ \dots,\ \frac{v_n}{w}\right)$。使用齐次坐标，我们可以写出

$$
P'_h = \begin{bmatrix} \alpha x + c_x z \\ \beta y + c_y z \\ z \end{bmatrix} = \begin{bmatrix} \alpha & 0 & c_x & 0 \\ 0 & \beta & c_y & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix} = \begin{bmatrix} \alpha & 0 & c_x & 0 \\ 0 & \beta & c_y & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix} P_h \tag{5}
$$

> **译注**：理解 (5) 式的**关键一步**是：把 (4) 式右端的分母 $z$「吸收」进第三个分量。齐次坐标下 $(\alpha x + c_x z,\ \beta y + c_y z,\ z)^T$ 与欧氏坐标下 $\left(\alpha\frac{x}{z}+c_x,\ \beta\frac{y}{z}+c_y\right)^T$ 表示同一个点，因为按上面的规则除以最后一个分量 $z$ 就回到了 (4) 式。这样一来，原本带除法的映射就变成了纯粹的矩阵乘法。此段解释为译者所加，原文没有。

从这里开始，除非另有说明，我们都假设自己在齐次坐标下工作，并且**省略上标 $h$**，即任何点 $P$ 或 $P'$ 都可以默认是齐次坐标。从 Equation 5 可以看到，三维空间中的点与其图像坐标之间的关系可以用矩阵—向量关系表示：

$$
P' = \begin{bmatrix} x' \\ y' \\ z \end{bmatrix} = \begin{bmatrix} \alpha & 0 & c_x & 0 \\ 0 & \beta & c_y & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix} = \begin{bmatrix} \alpha & 0 & c_x & 0 \\ 0 & \beta & c_y & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix} P = MP \tag{6}
$$

我们还能把这个变换进一步分解：

$$
P' = MP = \begin{bmatrix} \alpha & 0 & c_x \\ 0 & \beta & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} I & 0 \end{bmatrix} P = K \begin{bmatrix} I & 0 \end{bmatrix} P \tag{7}
$$

矩阵 $K$ 通常称为**相机矩阵（camera matrix）**。

#### 4.1.3 完整的相机矩阵模型（The Complete Camera Matrix Model）

相机矩阵 $K$ 包含了描述相机特性与其模型的一些关键参数，包括上面讨论过的 $c_x$、$c_y$、$k$、$l$。这个表达式目前还缺两个参数：**倾斜（skewness）**与**畸变（distortion）**。当相机坐标系是倾斜的时候我们就说图像是倾斜的，意思是两个轴之间的夹角略微大于或小于 90 度。绝大多数相机是零倾斜的，但由于传感器制造误差，可能出现一定程度的倾斜。考虑倾斜后的新相机矩阵的推导不在本课范围内，这里直接把结果给出：

$$
K = \begin{bmatrix} x' \\ y' \\ z \end{bmatrix} = \begin{bmatrix} \alpha & -\alpha \cot \theta & c_x \\ 0 & \frac{\beta}{\sin \theta} & c_y \\ 0 & 0 & 1 \end{bmatrix} \tag{8}
$$

> **译注**：(8) 式在原文中就是这样写的——左端把像素向量 $\left[x'\ y'\ z\right]^T$ 与符号 $K$ 并列，$=$ 后面才是 $3\times3$ 的上三角矩阵。按 (7) 式的记号，严格写法应是 $P' = K\left[I\ \ 0\right]P$，此处照录原文。

本课介绍的大多数方法都忽略畸变效应，因此我们的课程版相机矩阵 $K$ 有 **5 个自由度**：焦距占 2 个，偏移占 2 个，倾斜占 1 个。这些参数合称为**内参（intrinsic parameters）**，因为它们对某一台相机是独一无二、与生俱来的，与相机的本质属性（例如它的制造工艺）相关。

### 4.2 外参（Extrinsic Parameters）

到目前为止，我们用相机的内参（以矩阵形式表达）描述了从三维相机坐标系中的点 $P$ 到二维像面上的点 $P'$ 的映射。但如果关于三维世界的信息是在**另一个坐标系**里给出的呢？那我们就需要再加一个变换，把世界参考系中的点联系到相机参考系。这个变换由一个旋转矩阵 $R$ 和一个平移向量 $T$ 描述。因此，给定世界参考系中的点 $P_w$，可以按下面这样算出它的相机坐标：

$$
P = \begin{bmatrix} R & T \\ 0 & 1 \end{bmatrix} P_w \tag{9}
$$

把它代入 equation (7) 并化简，得到

$$
P' = K \left[ R \quad T \right] P_w = M P_w \tag{10}
$$

参数 $R$ 与 $T$ 称为**外参（extrinsic parameters）**，因为它们在相机之外、不依赖相机本身。

至此，从任意世界参考系中的三维点 $P$ 到像面的映射就完整了。再强调一遍：完整的投影矩阵 $M$ 由上面引入的两类参数构成——**内参**与**外参**。相机矩阵 $K$ 里的所有参数都是内参，它们随相机型号的改变而改变；外参包括旋转与平移，与相机的制造无关。总体而言，这个 $3\times4$ 的投影矩阵 $M$ 有 **11 个自由度**：内参相机矩阵贡献 5 个，外参旋转贡献 3 个，外参平移贡献 3 个。

## 5 相机标定（Camera Calibration）

要精确知道从真实的三维世界到数字图像的变换，需要预先掌握许多相机内参。如果拿到一台任意相机，我们未必能访问到这些参数。但我们确实能拿到这台相机拍出的图像。那么，能不能想办法**从图像中把它们推出来**？这个同时估计相机外参与内参的问题，就叫做**相机标定（camera calibration）**。

具体做法是从 Equation 10 中解出内参相机矩阵 $K$ 与外参 $R,\ T$。我们可以借助**标定板（calibration rig）**来描述这个问题，比如 Figure 7 里那个。

![左侧是标定板的世界坐标系 (W)：原点 O_w、三根轴 i_w、j_w、k_w 与网格平面，板上标出点 P_i；一条橙色光线从 P_i 连到右侧棋盘格图像中的对应点 p_i](/coursenotes/cs231a-computer-vision-3d-sp25/cn01-camera-models/fig_p0009_06.png)

**Figure 7：一个标定板的示例设置。**

标定板通常由一个尺寸已知的简单图案（例如棋盘格）构成。而且标定板定义了我们的世界参考系：原点 $O_w$，坐标轴 $i_w,\ j_w,\ k_w$。由标定板上已知的图案，我们得到世界参考系中已知的点 $P_1, \dots, P_n$；在相机拍到的图像中把这些点找出来，就得到图像中的对应点 $p_1, \dots, p_n$。

由 $n$ 组对应关系建立一个线性方程组：对每一组对应 $P_i,\ p_i$，以及行向量为 $m_1,\ m_2,\ m_3$ 的相机矩阵 $M$，有

$$
p_i = \begin{bmatrix} u_i \\ v_i \end{bmatrix} = M P_i = \begin{bmatrix} \dfrac{m_1 P_i}{m_3 P_i} \\[6pt] \dfrac{m_2 P_i}{m_3 P_i} \end{bmatrix} \tag{11}
$$

> **译注**：(11) 式在逐页转写稿中曾被错排成一个三分量的向量，此处已按原 PDF 第 10 页核对，右端是**两个**分量、分母都是 $m_3 P_i$。

从上面这个式子可以看到，每一组对应关系给出**两个方程**，也就是为求解 $m$ 中未知参数提供**两个约束**。前面已知相机矩阵有 11 个未知参数，这意味着我们至少需要 6 组对应关系才能解出来。不过在真实世界里我们通常用更多组，因为测量往往是带噪声的。为了把这一点看清楚，可以推出把 $u_i$、$v_i$ 与 $P_i$ 联系起来的方程对：

$$
\begin{aligned}
u_i (m_3 P_i) - m_1 P_i &= 0 \\
v_i (m_3 P_i) - m_2 P_i &= 0
\end{aligned}
$$

给定 $n$ 组这样的对应点，整个线性方程组就变成

$$
\begin{aligned}
u_1 (m_3 P_1) - m_1 P_1 &= 0 \\
v_1 (m_3 P_1) - m_2 P_1 &= 0 \\
\vdots \\
u_n (m_3 P_n) - m_1 P_n &= 0 \\
v_n (m_3 P_n) - m_2 P_n &= 0
\end{aligned}
$$

这可以整理成下面这个矩阵—向量乘积的形式：

$$
\begin{bmatrix} P_1^T & 0^T & -u_1 P_1^T \\ 0^T & P_1^T & -v_1 P_1^T \\ & \vdots & \\ P_n^T & 0^T & -u_n P_n^T \\ 0^T & P_n^T & -v_n P_n^T \end{bmatrix} \begin{bmatrix} m_1^T \\ m_2^T \\ m_3^T \end{bmatrix} = \mathbf{P} m = 0 \tag{12}
$$

当 $2n \gt 11$ 时，我们的齐次线性方程组是**超定**的。对这样的方程组，$m = 0$ 永远是平凡解。而且，即使存在某个非零解 $m$，那么 $\forall k \in \mathbb{R}$，$km$ 也同样是解。因此为了把解约束住，我们求解下面这个最小化问题：

$$
\begin{aligned}
\underset{m}{\text{minimize}} \quad & \| \mathbf{P} m \|^{2} \\
\text{subject to} \quad & \| m \|^{2} = 1
\end{aligned} \tag{13}
$$

这个最小化问题直接用**奇异值分解（SVD）**来解。若令 $\mathbf{P} = U D V^{T}$，那么上面最小化问题的解就是把 $m$ 取为 $V$ 的**最后一列**。该解的推导不在本课范围内，可以参阅 Hartley & Zisserman 第 5.3 节第 592–593 页了解更多细节。

把向量 $m$ 重新排成矩阵 $M$ 之后，我们想显式地解出外参与内参。我们知道 SVD 解出的 $M$ 只确定到尺度为止，也就是说相机矩阵的真实值是 $M$ 的某个标量倍：

$$
\rho M = \left[ \begin{array}{cc} \alpha r_{1}^{T} - \alpha \cot \theta\, r_{2}^{T} + c_{x} r_{3}^{T} & \alpha t_{x} - \alpha \cot \theta\, t_{y} + c_{x} t_{z} \\[4pt] \frac{\beta}{\sin \theta} r_{2}^{T} + c_{y} r_{3}^{T} & \frac{\beta}{\sin \theta} t_{y} + c_{y} t_{z} \\[4pt] r_{3}^{T} & t_{z} \end{array} \right] \tag{14}
$$

这里 $r_1^T$、$r_2^T$、$r_3^T$ 是 $R$ 的三行。除以尺度参数后得到

$$
M = \frac{1}{\rho} \left[ \begin{array}{cc} \alpha r_{1}^{T} - \alpha \cot \theta\, r_{2}^{T} + c_{x} r_{3}^{T} & \alpha t_{x} - \alpha \cot \theta\, t_{y} + c_{x} t_{z} \\[4pt] \frac{\beta}{\sin \theta} r_{2}^{T} + c_{y} r_{3}^{T} & \frac{\beta}{\sin \theta} t_{y} + c_{y} t_{z} \\[4pt] r_{3}^{T} & t_{z} \end{array} \right] = \left[ A \quad b \right] = \begin{bmatrix} a_{1}^{T} \\ a_{2}^{T} \\ a_{3}^{T} \end{bmatrix} \begin{bmatrix} b_{1} \\ b_{2} \\ b_{3} \end{bmatrix}
$$

> **译注**：上式即把 $M$ 的前三列组成的 $3\times3$ 块记作 $A$（三行为 $a_1^T,a_2^T,a_3^T$）、第四列记作 $b$（分量为 $b_1,b_2,b_3$）。原文直接采用 $[A\ \ b]$ 与行块记号，没有这句说明。

解内参得

$$
\begin{aligned}
\rho &= \pm \frac{1}{\| a_{3} \|} \\
c_{x} &= \rho^{2} (a_{1} \cdot a_{3}) \\
c_{y} &= \rho^{2} (a_{2} \cdot a_{3}) \\
\theta &= \cos^{-1} \left( - \frac{(a_{1} \times a_{3}) \cdot (a_{2} \times a_{3})}{\| a_{1} \times a_{3} \| \cdot \| a_{2} \times a_{3} \|} \right) \\
\alpha &= \rho^{2} \| a_{1} \times a_{3} \| \sin \theta \\
\beta &= \rho^{2} \| a_{2} \times a_{3} \| \sin \theta
\end{aligned} \tag{15}
$$

外参则是

$$
\begin{aligned}
r_{1} &= \frac{a_2 \times a_3}{\| a_2 \times a_3 \|} \\
r_{2} &= r_{3} \times r_{1} \\
r_{3} &= \rho a_{3} \\
T &= \rho K^{-1} b
\end{aligned} \tag{16}
$$

这些推导留作课堂练习，也可以参考 Forsyth & Ponce 教材第 1.3.1 节。

标定流程讲完之后，原文提醒我们要小心**退化情形**：并不是任意 $n$ 组对应关系都能用。例如，若点 $P_i$ 都落在同一个平面上，方程组就解不出来。这些无法求解的点配置称为**退化配置（degenerate configurations）**。更一般地说，退化配置里的点落在两张二次曲面（quadric surfaces）的交线上。这部分内容虽然超出本课范围，但可以在 Forsyth & Ponce 教材第 1.3 节找到更多信息。

> **译注**：本节的标定路线是「11 个未知数 + 每组对应给 2 个约束」的直接线性解法。全篇没有出现单应性矩阵 $\mathbf H$，也没有「由 $\mathbf H$ 反解内参约束」那条路径——这是原文的写法，不是转写遗漏。

## 6 相机标定中的畸变处理（Handling Distortion in Camera Calibration）

到目前为止，我们用的都是不含任何畸变的理想镜头。但如前所见，真实镜头会偏离直线投影（rectilinear projection），这需要更高级的方法。本节只做一个简短的畸变处理入门。

由于镜头在物理上具有对称性，畸变常常是**径向对称**的。我们用一个各向同性（isotropic）的变换来建模径向畸变：

$$
Q P_i = \begin{bmatrix} \frac{1}{\lambda} & 0 & 0 \\ 0 & \frac{1}{\lambda} & 0 \\ 0 & 0 & 1 \end{bmatrix} M P_i = \begin{bmatrix} u_i \\ v_i \end{bmatrix} = p_i \tag{17}
$$

如果像之前那样试图把它改写成一个方程组，我们得到

$$
\begin{aligned}
u_i q_3 P_i &= q_1 P_i \\
v_i q_3 P_i &= q_2 P_i
\end{aligned}
$$

然而这个方程组**不再线性**，需要动用非线性优化技术，相关内容在 Forsyth & Ponce 第 22.2 节。如果做一些假设，标定问题的非线性优化可以被简化。在径向畸变下，我们注意到两个坐标 $u_i$ 与 $v_i$ 之间的**比值不受影响**。这个比值可以算作

$$
\frac{u_i}{v_i} = \frac{m_1 P_i \,/\, m_3 P_i}{m_2 P_i \,/\, m_3 P_i} = \frac{m_1 P_i}{m_2 P_i} \tag{18}
$$

假设有 $n$ 组对应关系可用，就能建立如下线性方程组：

$$
\begin{aligned}
v_1 (m_1 P_1) - u_1 (m_2 P_1) &= 0 \\
\vdots \\
v_n (m_1 P_n) - u_n (m_2 P_n) &= 0
\end{aligned}
$$

与前面类似，这给出一个可以用 SVD 求解的矩阵—向量乘积：

$$
Ln = \begin{bmatrix} v_1 P_1^T & -u_1 P_1^T \\ \vdots & \vdots \\ v_n P_n^T & -u_n P_n^T \end{bmatrix} \begin{bmatrix} m_1^T \\ m_2^T \end{bmatrix} \tag{19}
$$

> **译注**：(19) 式在原文中右端就没有 $=0$（原 PDF 第 13 页即如此印），此处照录。

一旦 $m_1$ 与 $m_2$ 被估计出来，$m_3$ 就可以写成 $m_1$、$m_2$ 与 $\lambda$ 的一个非线性函数。这样需要求解的非线性优化问题，其复杂度比原来那个小得多。

## 7 附录 A：刚体变换（Appendix A: Rigid Transformations）

基本的刚体变换是**旋转、平移与缩放**。本附录讨论它们在三维情形下的形式，因为它们是本课中反复出现的类型。

三维空间中点的旋转，可以表示成分别绕三个坐标轴的旋转。绕坐标轴旋转时，通行的约定是**逆时针**方向。理解旋转的一种直观方式，是看我们在每个自由度上转了多少，这就是所谓的**欧拉角（Euler angles）**。然而这种做法会导致**奇异性（singularities）**，也叫**万向节死锁（gimbal lock）**：某些构型会让旋转失去一个自由度。

避免这一问题的一个办法是使用**旋转矩阵**，它是表示旋转的更一般形式。旋转矩阵是行列式为 1 的方阵、且是正交矩阵。给定旋转矩阵 $R$ 与向量 $v$，结果向量 $v'$ 可以这样算：

$$
v' = Rv
$$

由于旋转矩阵是非常一般的表示方式，绕三个轴分别旋转 $\alpha,\ \beta,\ \gamma$ 可以写成：

$$
R_x(\alpha) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\alpha & -\sin\alpha \\ 0 & \sin\alpha & \cos\alpha \end{bmatrix}
$$

$$
R_y(\beta) = \begin{bmatrix} \cos\beta & 0 & \sin\beta \\ 0 & 1 & 0 \\ -\sin\beta & 0 & \cos\beta \end{bmatrix} \qquad
R_z(\gamma) = \begin{bmatrix} \cos\gamma & -\sin\gamma & 0 \\ \sin\gamma & \cos\gamma & 0 \\ 0 & 0 & 1 \end{bmatrix}
$$

由于矩阵乘法的约定，「先绕 $z$ 轴转、再绕 $y$ 轴转、最后绕 $x$ 轴转」所得到的旋转由矩阵乘积 $R_x R_y R_z$ 给出。

**平移（translation，或称位移）**用来描述沿某个方向的移动。三维空间中，平移向量 $t$ 有三个值：三个轴上的位移，常记作 $t_x,\ t_y,\ t_z$。于是给定点 $P$ 被 $t$ 平移到另一个点 $P'$，可以写成：

$$
P' = P + t = \begin{bmatrix} P_x \\ P_y \\ P_z \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \\ t_z \end{bmatrix}
$$

用矩阵形式表达平移，同样要借助齐次坐标。若构造平移矩阵

$$
T = \begin{bmatrix} 1 & 0 & 0 & t_x \\ 0 & 1 & 0 & t_y \\ 0 & 0 & 1 & t_z \\ 0 & 0 & 0 & 1 \end{bmatrix}
$$

就能看到 $P' = TP$ 与 $P' = P + t$ 等价。

如果想把平移与旋转的矩阵乘法合并成一步，仍然可以借助齐次坐标。若要把向量 $v$ 先用 $R$ 旋转、再用 $t$ 平移，结果向量 $v'$ 可以写成：

$$
\begin{bmatrix} v' \\ 1 \end{bmatrix} = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} v \\ 1 \end{bmatrix}
$$

最后，如果想沿某些方向按 $S_x,\ S_y,\ S_z$ 的量对向量做**缩放**，可以构造缩放矩阵

$$
S = \begin{bmatrix} S_x & 0 & 0 \\ 0 & S_y & 0 \\ 0 & 0 & S_z \end{bmatrix}
$$

因此，若我们要「先缩放、再旋转、再平移」，最终的变换矩阵是：

$$
T = \begin{bmatrix} RS & t \\ 0 & 1 \end{bmatrix}
$$

注意，上面这些变换全都是**仿射变换（affine transformations）**的例子。回顾一下：当 $T$ 的最后一行不是 $\begin{bmatrix} 0 & 0 & 0 & 1 \end{bmatrix}$ 时，发生的才是**射影变换（projective transformations）**。

## 8 附录 B：不同的相机模型（Appendix B: Different Camera Models）

下面介绍一个简单模型，叫做**弱透视模型（weak perspective model）**。在弱透视模型中，点先被**正交投影**到一个参考平面上，然后再用一次射影变换投影到像面。

![参考平面 Π 距相机中心 z0，三维点 P、Q、R 先正交投影到 Π 上的 P_、Q_、R_，再经光心 O 投到左侧像面上的 p'、q'、r'，f' 标在 O 与像面之间](/coursenotes/cs231a-computer-vision-3d-sp25/cn01-camera-models/fig_p0015_03.png)

**Figure 8：弱透视模型——正交投影到参考平面。**（原文 Figure 9 是同一张图的第二步示意，标题为「弱透视模型——投影到像面」，图中元素与 Figure 8 相同。）

如 Figure 8 所示，给定一个距相机中心为 $z_0$ 的参考平面 $\Pi$，点 $P,\ Q,\ R$ 先被正交投影到平面 $\Pi$ 上，产生点 $P_{-},\ Q_{-},\ R_{-}$（原文用下划横线标记这一步得到的点）。当各点相对参考平面的深度偏差，与相机到物体的距离相比很小时，这是一个合理的近似。

Figure 9 展示了如何把 $P_{-},\ Q_{-},\ R_{-}$ 再用一次常规射影变换投影到像面，得到 $p',\ q',\ r'$。但请注意，由于我们把每个点的深度都近似成了 $z_0$，这个投影就退化成了一个**简单的、恒定的放大率**。该放大率等于焦距 $f'$ 除以 $z_0$，于是

$$
x' = \frac{f'}{z_0}\, x \qquad\quad y' = \frac{f'}{z_0}\, y
$$

这个模型同时也简化了投影矩阵：

$$
M = \begin{bmatrix} A & b \\ 0 & 1 \end{bmatrix}
$$

如我们所见，在弱透视模型中 $M$ 的最后一行是 $\begin{bmatrix} 0 & 0 & 0 & 1 \end{bmatrix}$，而普通相机模型里是 $\begin{bmatrix} v & 1 \end{bmatrix}$（原文此处就印作 $[v\ \ 1]$）。这个结论我们不证明，留给你作为练习。把三维点映射到像面时，这个简化体现得非常清楚：

$$
P' = MP = \begin{bmatrix} m_1 \\ m_2 \\ m_3 \end{bmatrix} P = \begin{bmatrix} m_1 P \\ m_2 P \\ 1 \end{bmatrix} \tag{20}
$$

于是我们看到，像面上的点最终只是原始三维点的一个**放大**，与深度无关。射影变换的非线性消失了，弱透视变换只不过是一台放大镜。

> 原文另有 Figure 10「正射投影模型」：物体上 $P$、$Q$ 两点的投影光线平行于光轴、垂直打在像面 $\Pi'$ 上，得到 $P'$、$Q'$，光心 $O$ 一侧只画了一个指向像面的空心箭头，表示视点在无穷远。

进一步简化就得到**正射（或仿射）投影模型（orthographic / affine projection model）**。在这种情况下，光心位于**无穷远**，投影光线垂直于视网膜平面。因此这个模型完全忽略深度。于是

$$
\begin{aligned}
x' &= x \\
y' &= y
\end{aligned}
$$

正射投影模型常用于建筑与工业设计。

总的来说，弱透视模型带来的数学要简单得多，代价是精度略有损失。但在物体较小、且离相机较远时，它往往能给出非常准确的结果。

## 要点回顾

- 针孔相机的核心是「一块带小孔的挡板」：它把三维物体上的点与胶片上的点配成一一映射，得到**针孔相机模型**。
- 由相似三角形直接给出针孔投影 $P'=[x'\ y']^T=[f\frac{x}{z},\ f\frac{y}{z}]^T$（式 1）：投影要除以深度 $z$，这正是后面一切「非线性麻烦」的来源。
- 光圈不能无限小：光圈大→进光多但模糊，光圈小→清晰但暗，这就是针孔模型的根本矛盾（Figure 3）。
- 镜头把同一点发出的所有光线重新会聚到像面上的一点，化解了「清晰度 vs 亮度」的冲突；但它只对特定距离上的点成立，由此引出**景深**。
- 带镜头的模型给出 $P'=\begin{bmatrix} z'\frac{x}{z} \\ z'\frac{y}{z} \end{bmatrix}$（式 2，列向量、无转置号），其中针孔情形 $z'=f$，镜头情形 $z'=f+z_0$；这就是**近轴折射模型**（薄透镜假设 $\sin\theta\approx\theta$）。
- 薄透镜近似带来**径向畸变**：放大率随到光轴的距离变化，增大是枕形畸变，减小是桶形畸变（鱼眼镜头常见），成因是镜头各部位焦距不同。
- 像面坐标到像素坐标要补三件事：原点平移 $c_x,c_y$（式 3）、单位换算 $k,l$（式 4，像素宽高比不必为 1，$k=l$ 即方形像素）、传感器非线性（畸变）。
- **齐次坐标**把除法吸收进最后一个分量：$(v_1,\dots,v_n,w)\mapsto(\frac{v_1}{w},\dots,\frac{v_n}{w})$，于是 $P'=MP$（式 5、6），并可分解为 $P'=K[I\ \ 0]P$（式 7）。
- 完整内参矩阵（式 8）含倾斜项 $-\alpha\cot\theta$ 与 $\frac{\beta}{\sin\theta}$，课程版 $K$ 有 **5 个自由度**（焦距 2、偏移 2、倾斜 1），统称**内参**。
- 世界系到相机系靠 $\begin{bmatrix} R & T \\ 0 & 1\end{bmatrix}$（式 9），代入得 $P'=K[R\ T]P_w=MP_w$（式 10）；$R,T$ 是**外参**，与相机制造无关。$3\times4$ 的 $M$ 共 **11 个自由度 = 5 + 3 + 3**。
- 标定 = 从 $n$ 组「已知三维点 ↔ 像素点」对应中解出 $K$ 与 $R,T$。每组对应给 **2 个约束**（式 11），故至少需要 6 组；实际因噪声会用更多。
- 把约束排成 $\mathbf P m = 0$（式 12），当 $2n \gt 11$ 时超定；齐次系统有平凡解与尺度歧义，因此解 $\min\|\mathbf P m\|^2\ \text{s.t.}\ \|m\|^2=1$（式 13），答案是 SVD 中 $V$ 的**最后一列**。
- SVD 出的 $M$ 只到尺度 $\rho$ 为止（式 14）。写成 $M=[A\ b]$ 后，内参由 (15) 给出（$\rho=\pm\frac{1}{\|a_3\|}$、$c_x=\rho^2(a_1\cdot a_3)$、$\theta$ 由两个叉积的夹角给出、$\alpha=\rho^2\|a_1\times a_3\|\sin\theta$ 等），外参由 (16) 给出（$r_1=\frac{a_2\times a_3}{\|a_2\times a_3\|}$、$r_3=\rho a_3$、$T=\rho K^{-1}b$）。
- **退化配置**：所有 $P_i$ 共面时方程组无解；更一般地，点落在两张二次曲面的交线上时同样退化。
- 带径向畸变的模型是 $QP_i=\operatorname{diag}(\frac1\lambda,\frac1\lambda,1)MP_i=p_i$（式 17），直接展开得到的方程组**非线性**；但 $u_i/v_i$ 这个比值不受畸变影响（式 18），据此可先用线性系统 $Ln$（式 19）解出 $m_1,m_2$，再把 $m_3$ 写成 $m_1,m_2,\lambda$ 的非线性函数，把优化规模大幅缩小。
- 附录 A：欧拉角直观但会**万向节死锁**；旋转矩阵是行列式为 1 的正交阵，$R_xR_yRz$ 对应「先 $z$、再 $y$、后 $x$」；齐次坐标让平移、旋转+平移、缩放合并为 $\begin{bmatrix} RS & t \\ 0 & 1\end{bmatrix}$；这类都是**仿射变换**，最后一行不是 $[0\ 0\ 0\ 1]$ 时才是射影变换。
- 附录 B：**弱透视**= 先正交投影到距离 $z_0$ 的参考平面、再射影到像面，深度被固定后投影只剩恒定放大率 $\frac{f'}{z_0}$，$M$ 的最后一行变成 $[0\ 0\ 0\ 1]$（式 20）；再简化就是**正射投影**（光心在无穷远，$x'=x,\ y'=y$，完全丢深度，用于建筑与工业设计）。弱透视数学简单、略欠精确，但物体小而远时非常准。

## 原文出处

- Stanford CS231A *Course Notes 1: Camera Models*，作者 Kenji Hata and Silvio Savarese，共 17 页，含 Figure 1–10 与公式 (1)–(20)。官方发布在 CS231A 课程 syllabus 页面的 Course Notes 列表中，PDF 文件名为 `01-camera-models.pdf`。
- 本笔记覆盖的课程内容：第 2 讲 Camera Models 与第 3 讲 Camera Models II and Camera Calibration。
- 原文正文没有独立的「References / Acknowledgements」小节，但在推导中点名引用了三处，按其列出：
  - Hartley & Zisserman，第 5.3 节，第 592–593 页（$\mathbf P m = 0$ 的 SVD 解法）。
  - Forsyth & Ponce 教材，第 1.3.1 节（由 $M=[A\ b]$ 拆出内参与外参的推导）、第 1.3 节（退化配置）、第 22.2 节（非线性优化技术）。
- 原文脚注三条（撇号记号约定、近轴假设 $\sin\theta\approx\theta$、桶形畸变多见于鱼眼镜头）已分别就近放在对应段落之后。
