---
title: "CS231A · 第 17 讲 · 3D Gaussian Splatting"
description: "用各向异性 3D 高斯当显式场景基元，靠一阶线性化投影到屏幕、按深度排序做 alpha-blending，配合 SfM 初始化与克隆/分裂致密化，同时拿到 NeRF 的体渲染质量与光栅化的实时速度。"
---

# CS231A · 第 17 讲 · 3D Gaussian Splatting

> **Stanford CS231A 计算机视觉：从三维感知到三维重建 · 第 17 讲（2025 春季）** ｜ 讲者：Silvio Savarese、Jeannette Bohg ｜ 课件日期 2-Jun-25，标题「Gaussian Splatting for Novel View Synthesis」｜ 本课无公开视频；本讲课件未覆盖的论文细节（协方差参数化、光栅器实现、训练调度）均按原文补齐并标明出处

## 核心结论

第 16 讲的 NeRF 把场景写成连续 5D 函数，代价是每条射线都要在许多「其实什么都没有」的空间里采样。本讲的切入点是课件 p037 那句话：**「Stupid: we already know where the density will be nonzero!」**——既然非零密度的位置可以先验知道，就用一团团显式的各向异性 3D 高斯只覆盖那些位置（p028），几何参数直接可优化、渲染交给 GPU 光栅化。但它并没有丢掉体渲染：合成公式仍是 NeRF 的 alpha-compositing，只是把「沿射线密集采样」换成「按深度顺序枚举与像素相交的高斯」（p057）。于是质量来自连续体模型，速度来自 splatting（泼溅式光栅化：把图元投到屏幕、按深度顺序把颜色「泼」进像素）。协方差在仿射下闭合、沿轴积分得到边缘高斯（p038–p040），投影则用透视映射的一阶泰勒近似（p046–p055）——这条推导正是第 7 讲相机模型与第 16 讲体渲染的汇合处。

## 1. 本讲的问题设定：可微渲染与前向模型

课件开场把任务拆成两个方向（p003–p004）：图形学是「给定 3D 场景 + 相机参数 → 出图」，逆图形学是「给定图像 → 推断相机位姿与 3D 场景」。p005 直接问「How to get camera poses?」，p006 又说「Can assume we know the camera poses」——本讲按后者处理，但位姿与稀疏几何的来源正是第 7 讲的 SfM，这一点在后面成为方法的关键部件。

接着是可微渲染的定义（p007–p010）：给定一个**可观测变量（像素颜色）**，构造一个可微前向模型，用它去估计**不可观测的隐变量（几何与外观）**。渲染有两条路（p011）：surface rendering 走实心图元与可见性判断，volume rendering 沿射线累积。课件给出体渲染方程（p012）：

$$\mathbf{I}(D) = \mathbf{I}_0 T(0) + \int_0^D \mathbf{c}(s)\,\rho(s)\,T(s)\,ds,\qquad T(s) = \exp\!\left(-\int_s^D \rho(t)\,dt\right)$$

其中 $T(s)$ 是透射率（transparency）。第 16 讲的 NeRF 把它写成场景函数 $F_\Omega:(x,y,z,\theta,\phi)\rightarrow(r,g,b,\sigma)$（p013），离散化后是（p015）

$$C \approx \sum_{i=1}^{N} T_i\,c_i\,\alpha_i,\qquad T_i = \prod_{j=1}^{i-1}(1-\alpha_j),\qquad \alpha_i = 1-e^{-\sigma_i\,\delta t_i}$$

再用渲染损失做梯度下降（p016）：$\min_\Omega \sum_i \lVert \mathrm{render}^{(i)}(F_\Omega) - I_{gt}^{(i)} \rVert^2$，训练网络复现所有输入视角（p017）。

**瓶颈在哪。** 课件 p022、p024 的柱状图把当时的一批加速方案摆在同一根 FPS 轴上：Plenoxels、Direct Voxel Grid Optimization、InstantNGP、PointNeRF、TensorRF 等（p023），其中 InstantNGP 约 9.0 FPS，而追求画质的 MipNeRF360 只有 0.07 FPS。p026 在同一张图上加了 3DGS 的两根柱子：137.3 与 167.9 FPS。原文的口径是「1080p 分辨率下 ≥ 100 fps 的实时新视角合成」；课件 p064 展示用 A6000 显卡在 1080p 下交互录屏。对照原文结果表（Mip-NeRF360 数据集）：Mip-NeRF360 训练 48 小时、0.06 FPS、8.6MB；3DGS 30K 迭代训练 41 分 33 秒、134 FPS、734MB——画质接近，渲染快三个数量级，但**显存/模型体积贵了两个数量级**。这笔交换是本讲最值得记住的工程判断。

## 2. 表示：只在密度非零处放各向异性高斯

课件的核心思想只有一句话（p028–p032，逐帧搭建）：**「Parameterize Radiance Field sparsely, only where density is nonzero」**，场景 = 漂浮在空间中的 3D Gaussian Blobs。搭建过程很朴素：先有一团带均值（Mean）的椭球，再给一个星号位置 $\sigma=0$（那里什么都没有），然后逐步加上 $\sigma=0.5$、$\sigma=1$ 的团与对应 RGB。这与 p027 的对照很关键：NeRF 是「**densely**，在空间每一点参数化辐射场」，3DGS 是「**sparsely**，只在有东西的地方参数化」。

p033「Anisotropic Volumetric 3D Gaussians」放了原文的成对图：左边 Final Rendering，右边 3D Gaussian Visualization——能直接看到那些扁的、长的椭球如何贴合树叶与朽木表面。为什么必须是各向异性？原文说得直接：大的均匀区域可以用**少数几个大的各向异性高斯**拿下，协方差的质量直接决定表示的紧凑性。

每团高斯持有四类参数（原文 Sec.3/5.1）：位置 $\mu$、协方差 $\Sigma$、不透明度 $\alpha$（用 sigmoid 约束到 $[0,1)$）、以及用球谐系数表达的视角相关颜色（沿用第 16 讲的外观编码）。协方差不能直接优化：

> 协方差矩阵只有在**半正定**时才有物理意义，而梯度下降的更新步很容易造出非法的协方差矩阵。（原文 Sec.3）

所以参数化为缩放与旋转的乘积（原文式 (6)）：

$$\Sigma = R\,S\,S^{\top} R^{\top}$$

实现里存的是 3 维缩放向量 $\mathbf{s}$ 与 4 维四元数 $\mathbf{q}$（每次组装前归一化，以保证是合法的单位四元数），需要时再转成 $S, R$。这样半正定是**结构性保证**而非靠投影：对任意方向 $v$ 有 $v^{\top}\Sigma v=\lVert S^{\top}R^{\top}v\rVert^{2}\ge 0$，指数项永远衰减。缩放用指数激活，初始化取各向同性，轴长等于到最近三个邻居点距离的平均值——尺度天然与 SfM 点间距挂钩。原文还强调梯度是**手工推导**的（附录 A），以避免自动微分的开销。

**几何从哪来。** 原文 Sec.1：与以往 NeRF 类方法输入相同——SfM 标定的相机（Snavely et al. 2006，即第 7 讲的 Photo Tourism 一支）——并**用 SfM 顺带产出的稀疏点云初始化高斯**。这里有一处承前关系值得写清：第 7 讲把稀疏点云当作相机位姿估计的副产品，本讲把它升级成场景表示的初值；而此前的点渲染方法大多依赖 MVS 稠密几何，会继承过/欠重建伪影，3DGS 只用 SfM 稀疏点就够（课件 p061「Fix 1: Start from SFM point cloud.」）。

## 3. 渲染：协方差怎么投到屏幕

这是本讲推导密度最高的一段，课件用「高斯在仿射变换与积分下闭合」这条性质把它讲成线性代数练习。

**闭包性（p038–p040）。** 三维高斯写作 $\mathcal{G}_V(\mathbf{x}-\mathbf{p})=\frac{1}{2\pi}\lvert V\rvert^{-1/2}e^{-\frac{1}{2}(\mathbf{x}-\mathbf{p})^{\top}V^{-1}(\mathbf{x}-\mathbf{p})}$，$V$ 即 3D 协方差。做仿射映射 $\Phi(\mathbf{x})=\mathbf{M}\mathbf{x}+\mathbf{p}$（例如 cam2world 矩阵）后，仍是高斯，协方差变成 $\mathbf{M}V\mathbf{M}^{\top}$，前面带雅可比行列式因子 $\frac{1}{\lvert \mathbf{M}^{-1}\rvert}$。沿一根轴积分（边缘化）也闭合：

$$\int_{\mathbb R} \mathcal{G}_V^{3}(\mathbf{x}-\mathbf{p})\,dx_2 = \mathcal{G}_{\tilde V}^{2}(\hat{\mathbf x}-\hat{\mathbf p}),\qquad V=\begin{pmatrix} a&b&c\\ b&d&e\\ c&e&f \end{pmatrix}\Rightarrow \tilde{V}=\begin{pmatrix} a&b\\ b&d \end{pmatrix}$$

即三维椭球「压扁」成二维高斯时，协方差取对应的主子矩阵。这一步是后面「投影 3D 高斯得到 2D 高斯」（p058）的依据。

课件在 p041 与 p052 插了两个「Throwback」：Kalman 滤波与扩展 Kalman 滤波。类比很到位——线性映射下协方差按 $MVM^{\top}$ 传播就是 Kalman 的 predict 步，而透视投影是非线性函数，3DGS 的处理正是 EKF 的做法：**在工作点线性化，再传播协方差**。

**光栅化取代射线步进（p043–p045）。** 既然密度位置已知，就不必沿射线采样，改为对每个像素（更准确地说是「pixel frustum」）累加与它相交的图元。p045 给出可见性裁剪：「Cull Gaussians with less than 99% confidence relative to view frustum」——协方差椭球与视锥的 99% 置信区间不相交就丢掉。原文补充了实现细节：屏幕切成 $16\times16$ 像素的 tile（屏幕分块），先对 3D 高斯做视锥与逐 tile 剔除，再用 guard band（保护带）粗略拒绝「均值贴近近平面且在视锥外很远」的高斯，因为它们的投影协方差数值不稳定。

**投影（p046–p048, p053–p055）。** cam2world 是仿射映射 $\phi(\mathbf{x})=\mathbf{W}\mathbf{x}+\mathbf{p}$，于是协方差变成 $WV_0W^{\top}$；但针孔投影 $\mathbf{m}(u)$ **不是**仿射映射（课件特意标了「:/」），所以用一阶泰勒展开近似：

$$\Sigma^{\prime} = \mathbf{J}\,\mathbf{W}\,\Sigma\,\mathbf{W}^{\top}\,\mathbf{J}^{\top}$$

其中 $\mathbf{J}$ 是「投影变换的仿射近似」在该点的雅可比（原文式 (5)）。课件 p055 把一阶展开写成 $\mathbf{m}_{\mathbf{u}_k}(\mathbf{u})=\mathbf{x}_k+\mathbf{J}_{\mathbf{u}_k}(\mathbf{u}-\mathbf{u}_k)$，并给出投影后的二维高斯 $\frac{1}{\lvert \mathbf{W}^{-1}\rvert\,\lvert \mathbf{J}^{-1}\rvert}\mathcal{G}_{\mathbf{V}_k}(\mathbf{x}-\mathbf{x}_k)$，其协方差是

$$\mathbf{V}_k = \mathbf{J}\mathbf{V}'_k\mathbf{J}^{\top} = \mathbf{J}\mathbf{W}\mathbf{V}''_k\mathbf{W}^{\top}\mathbf{J}^{\top}$$

即「先世界到相机（$\mathbf{W}$），再相机到屏幕（$\mathbf{J}$）」两次合同变换，行列式项给出幅度的缩放。原文还指出：丢掉 $\Sigma'$ 的第三行第三列，就得到与 EWA splatting、Kopanas et al. 2021 这类平面点方法同构的 $2\times2$ 方差矩阵。p056 再沿射线积分一次，得到射线上的一维高斯 $g_3$，用于求该像素的 $\alpha$。

## 4. 合成：与 NeRF 是同一个成像模型

p057 的标题是本讲的点题句：**「Can compute volume rendering integral without ever sampling a single 3D point in space!」** 左边是 NeRF 的离散求和 $C=\sum T_ic_i\alpha_i$，右边是 3DGS 原文式 (3)：

$$C = \sum_{i\in N} c_i\,\alpha_i \prod_{j=1}^{i-1}\left(1-\alpha_j\right)$$

差别只在 $\alpha_i$ 的来源：NeRF 由 $\sigma$ 与步长给出，3DGS 则是**把投影后的 2D 高斯在像素中心求值，再乘上一个可学习的逐点不透明度**（课件 p057 的注释把这一做法标为 Yifan et al. 2019，即早期的可微软点渲染器）。原文在相关工作中写得更明白：基于点的 $\alpha$-blending 与 NeRF 式体渲染「share essentially the same image formation model」。这就是「混合式表示」的确切含义——几何显式、累积连续，于是半透明与柔和遮挡这些 NeRF 的优点被保留下来。

**排序是正确性的前提，也是代价所在。** 上式的乘积顺序依赖深度次序。原文的做法（附录 C）：每帧渲染开始时做一次 GPU 基数排序（Radix sort），为每个高斯实例分配一个 64 位键，低 32 位是视空间深度、高位是它覆盖的 tile 索引；每个 tile 起一个线程块，把排好序的高斯分块搬进共享内存，**前到后**累加颜色与 $\alpha$，当某像素的 $\alpha$ 饱和（趋于 1）时该线程提前退出，整块全饱和则终止。注意两个诚实的细节：其一，**没有逐像素重新排序**，合成基于这次全局排序，因此「$\alpha$-blending 在某些构型下是近似的」，只有当 splat 尺寸接近像素时才可忽略；其二，光栅化中唯一的停止条件就是 $\alpha$ 饱和，不限制参与混合的高斯个数。

这两点合起来解释了实践中常见的伪影：两团互相贯穿的大椭球，谁在前取决于中心深度而非真实遮挡关系，视角移动时会突然交换混合顺序——原文的 Limitations 把「popping artifacts」部分归因于此，以及 guard band 的粗暴剔除。

**历史坐标。** 这条路线并不新，课件没有讲历史，原文 Sec.2.3 给了谱系：点采样渲染（Grossman & Dally 1998；Gross & Pfister 2011 的综述）有孔洞与走样、严格不连续；EWA splatting（Zwicker et al. 2001）、surfel（Pfister et al. 2000）、物空间 EWA 的硬件加速（Ren et al. 2002）通过让图元大于一个像素来解决；再往后是可微的点渲染（Yifan et al. 2019；Wiles et al. 2020）与「点云 + 神经特征 + CNN」（Aliev et al. 2020；Rückert et al. 2022），但后者依赖 MVS 初值。3DGS 的位置是：把这条老路线接上**可微优化 + 各向异性协方差 + 连续体 $\alpha$ 合成**，于是 splatting 第一次同时给出照片级画质与实时帧率。

## 5. 优化：局部极小、SfM 初始化与自适应致密化

p059 用一句反问收尾渲染部分（「Any problems for inverse graphics, though…?」），p060 给出答案：**Problem: Local minima…**。稀疏初始化留下大片空白，梯度会把高斯推向相机附近形成漂浮物（floaters），或让一团大椭球糊住结构；这是显式非凸优化的典型陷阱。课件给的两个「Fix」就是原文方法的后两根柱子。

**Fix 1（p061）：从 SfM 点云出发。** 该页的管线图把整套系统串了起来：SfM Points → Initialization → 3D Gaussians → Projection 与 Adaptive Density Control → Differentiable Tile Rasterizer → Image，并同时画出操作流与梯度流。几何先验由 SfM 提供，密度由优化补全。

**Fix 2（p062）：启发式的 pruning 与 spawning。** 图上分两行：Under Reconstruction（覆盖不足）→ **Clone**；Over Reconstruction（一个大 splat 糊住细节）→ **Split**。判据是**视空间位置梯度的累积幅度**：欠重建区与过重建区的共同特征都是「优化一直想把这团高斯挪走」，因此对多视图平均后的位置梯度幅值超过阈值 $\tau_{pos}$（原文取 0.0002）就致密化。具体规则（原文 Sec.5.2）：

- 小高斯且欠重建 → 克隆：复制一个同尺寸的高斯，并沿位置梯度方向移动；
- 大高斯且方差大 → 分裂：替换成两个新高斯，尺度除以 $\phi=1.6$（实验定的值），位置按父高斯的密度采样；
- 剔除：不透明度低于 $\epsilon_\alpha$ 的「近乎透明」高斯，以及周期性移除世界空间过大或视空间 footprint 过大的高斯；
- 调度：预热后每 100 次迭代做一次致密化；每 $N=3000$ 次迭代把所有 $\alpha$ 重置到接近零，让优化重新「投票」，专门用来压掉相机附近的漂浮物。

损失与其余调度（原文 Sec.5.1、7.1）：

$$\mathcal{L} = (1-\lambda)\,\mathcal{L}_1 + \lambda\,\mathcal{L}_{D\text{-}SSIM},\qquad \lambda = 0.2$$

位置用指数衰减学习率；先在 4 倍低分辨率预热，第 250、500 步各上采样一次；球谐按每 1000 步增加一个频带的方式逐步放开，避免缺角度信息时把 0 阶项（漫反射基色）优化歪。

**启发式的后果。** 判据是几何梯度，而「梯度大」与「这里缺几何」只是间接相关：高对比度纹理边缘同样能产生大梯度，弱纹理与反光面则相反。于是基元数随场景复杂度膨胀（原文：所有测试场景 1–5 百万个高斯，模型 270–734MB，而 InstantNGP 只要 13–48MB），且会出现原文自陈的「elongated artifacts 或 splotchy Gaussians」。此外原文强调「目前不对优化施加任何正则」，这正是漂浮物与伪影的根源之一；而「高斯如何知道在边缘停下来」这类问题，本质上没有解析答案，只有密度控制的阈值在近似回答。

## 6. 局限与延伸

课件 p063 放了 NeRF-Synthetic 数据集上的优化延时录像，p064 是 1080p 交互浏览，p065 预告下一讲是 Francis Engelmann 的客座讲座（Open-World 3D Scene Understanding）。课件本身没有展开消融实验与后续工作，以下是据原文与已核实文献的补充。

原文 Limitations 自陈四点：观测不足区域有伪影（其他方法同样挣扎）；各向异性高斯会带来拉长或斑驳的团块；大高斯导致 popping（源于 guard band 剔除与可见性顺序突变）；**抗锯齿留作未来工作**。再加上表示层面的固有代价：显式基元的内存与排序开销、逐场景优化（不是前馈模型，换场景要重训）、以及椭球有厚度且可互相贯穿，使密度场并不天然等价于物体表面。

后续工作（课件未提及，均属已核实的延伸）：Mip-Splatting（CVPR 2024，arXiv:2311.16493）指出改变采样率（焦距或距离）时的伪影源于缺少三维频率约束与固定的 2D dilation，提出 3D 平滑滤波器与 2D Mip 滤波器；2D Gaussian Splatting（SIGGRAPH 2024，arXiv:2403.17888）认为 3D 椭球「多视图不一致」，把体积塌缩成一堆**有朝向的 2D 平面圆盘**，用 ray-splat 求交的透视正确 splatting，再加深度失真与法向一致性正则，从而重建出几何准确的表面；SuGaR（CVPR 2024，arXiv:2311.12775）加表面对齐正则后用泊松重建提取网格；4D-GS（CVPR 2024，arXiv:2310.08528）用 4D 神经体素编码 + 轻量 MLP 预测高斯形变，把这套表示扩展到动态场景。可以看出，本讲的三大近似（一阶投影、全局一次排序、启发式密度控制）恰好就是后续工作逐一修补的清单。

## 要点回顾

- 动机一句话：NeRF 沿射线在空区域浪费采样，而我们事先知道密度非零的位置（课件 p035–p037）。
- 表示是「稀疏 + 各向异性」：只在有东西的地方放高斯，用椭球而非球贴合局部几何（p028、p033）。
- 每团高斯持有位置、协方差、不透明度（sigmoid 约束）与球谐颜色；协方差分解为 $\Sigma=RSS^{\top}R^{\top}$，存 3 维缩放 + 归一化四元数，半正定由结构保证。
- 高斯在仿射映射下协方差按 $MVM^{\top}$ 变换、沿轴积分取主子矩阵，因此 3D 椭球能解析地压成 2D 高斯（p038–p040、p058）。
- 透视投影不是仿射映射，用一阶泰勒（雅可比 $\mathbf{J}$）近似：$\Sigma'=\mathbf{J}\mathbf{W}\Sigma\mathbf{W}^{\top}\mathbf{J}^{\top}$；这与扩展 Kalman 滤波的线性化同构（p046–p055）。
- 可见性靠视锥 99% 置信区间剔除 + $16\times16$ tile + 每帧一次基数排序（键 = tile 索引 + 视空间深度）。
- 合成仍是 NeRF 的成像模型：$C=\sum_i c_i\alpha_i\prod_{j \lt i}(1-\alpha_j)$，只是 $\alpha_i$ 改为「投影 2D 高斯在像素处的取值 × 学习到的逐点不透明度」（p057）。
- 正确性代价：不做逐像素重排序，故 $\alpha$-blending 近似；大 splat 会因混合顺序突变产生 popping。
- 稀疏优化的陷阱是局部极小与漂浮物（p060）；解药一是 SfM 点云初始化（p061，回指第 7 讲），二是克隆/分裂式自适应致密化（p062）。
- 致密化判据是视空间位置梯度平均幅值超 $\tau_{pos}=0.0002$，每 100 步执行；小高斯克隆、大高斯分裂（尺度除以 1.6）；每 3000 步重置不透明度以清理漂浮物。
- 损失是 $(1-\lambda)\mathcal{L}_1+\lambda\mathcal{L}_{D\text{-}SSIM}$，$\lambda=0.2$；没有额外正则项。
- 收益与账单同样清楚：1080p 下 ≥100 FPS、训练几十分钟；代价是百万级基元与数百 MB 显存，以及表面不清晰、抗锯齿缺失、动态场景需另行扩展。

## 参考资料

- 课件：Stanford CS231A Lecture 17, *Gaussian Splatting for Novel View Synthesis*, 2-Jun-25（Silvio Savarese & Jeannette Bohg）。其中渲染与高斯推导的多数页面标注「Slide adopted from 6.S980 – ML for Inverse Graphics – Vincent Sitzmann」
- 原始论文：Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis, *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, ACM Transactions on Graphics (SIGGRAPH) 2023, arXiv:2308.04079，项目页 <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>（本讲的参数化、投影式 (5)、合成式 (3)、光栅器与排序、致密化阈值、损失式 (7)、局限段落均出自此文）
- 前置：Ben Mildenhall et al., *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*, ECCV 2020（课件 p002 引用，<https://www.matthewtancik.com/nerf>）
- 多视图几何与 SfM：Noah Snavely, Steven M. Seitz, Richard Szeliski, *Photo Tourism: Exploring Photo Collections in Context*, SIGGRAPH 2006（原文引用的相机标定与稀疏点云来源，对应第 7 讲）
- 点渲染谱系（原文 Sec.2.3 引用）：Zwicker et al., *EWA Splatting*, 2001；Pfister et al., *Surfels*, 2000；Ren et al., *Object Space EWA Surface Splatting*, 2002；Gross & Pfister 2011 综述；Yifan et al., *Point-based Neural Renderers*, CVPR 2019；Kopanas et al., *3D Multi-plane Point Renderers for Image-based Rendering*, 2021
- 延伸（课件未讲，已核实）：Zehao Yu et al., *Mip-Splatting: Alias-free 3D Gaussian Splatting*, CVPR 2024, arXiv:2311.16493；Binbin Huang et al., *2D Gaussian Splatting for Geometrically Accurate Radiance Fields*, SIGGRAPH 2024, arXiv:2403.17888；Antoine Guédon & Vincent Lepetit, *SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering*, CVPR 2024, arXiv:2311.12775；Guanjun Wu et al., *4D Gaussian Splatting for Real-Time Dynamic Scene Rendering*, CVPR 2024, arXiv:2310.08528

![本讲核心思想：把辐射场稀疏地参数化，只在密度非零处放高斯基元，图中椭球标出自己的均值 Mean（课件 p028）](/coursenotes/cs231a-computer-vision-3d-sp25/l17-gaussian-splatting/slide_p028.jpg)

![各向异性体高斯：左为最终渲染，右为 3D Gaussian Visualization，可看到扁长椭球如何贴合树叶与朽木表面（课件 p033）](/coursenotes/cs231a-computer-vision-3d-sp25/l17-gaussian-splatting/slide_p033.jpg)

![协方差投影：透视投影非线性，用一阶泰勒展开近似，于是投影后的二维高斯协方差为 V_k = JV'_k Jᵀ = JWV''_kWᵀJᵀ（课件 p055）](/coursenotes/cs231a-computer-vision-3d-sp25/l17-gaussian-splatting/slide_p055.jpg)

![从不采样任何三维点也能算体渲染积分：NeRF 的射线离散求和与 3DGS 的 alpha-blending 式 (3) 并列对照（课件 p057）](/coursenotes/cs231a-computer-vision-3d-sp25/l17-gaussian-splatting/slide_p057.jpg)

![完整管线：SfM 稀疏点云初始化高斯，投影与自适应密度控制交替进行，经可微 tile 光栅化出图，并回传梯度（课件 p061）](/coursenotes/cs231a-computer-vision-3d-sp25/l17-gaussian-splatting/slide_p061.jpg)

![密度控制的两种算子：覆盖不足时克隆（Clone），一个大 splat 糊住细节时分裂（Split）（课件 p062）](/coursenotes/cs231a-computer-vision-3d-sp25/l17-gaussian-splatting/slide_p062.jpg)
