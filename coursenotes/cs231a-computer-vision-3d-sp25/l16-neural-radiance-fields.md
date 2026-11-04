---
title: "CS231A · 第 16 讲 · 神经辐射场 NeRF"
description: "从生成观测模型与体渲染出发，梳理 NeRF 的连续 5D 辐射场、离散求积、位置编码、分层采样与训练流程。"
---

# CS231A · 第 16 讲 · 神经辐射场 NeRF

> **Stanford CS231A 计算机视觉：从三维感知到三维重建 · 第 16 讲（2025 春季）** ｜ 讲者：Jeannette Bohg、Silvio Savarese ｜ 本课无公开视频，笔记依据官方课件与原始论文

## 核心结论

这一讲解决的问题是：给定少量已知位姿的图像，如何表示一个**连续**的场景，并渲染出任意新视角（novel view synthesis，新视角合成）。课件的推进方式很有意图——它不是从「神经网络」开始，而是从第 14–15 讲的最优估计（optimal estimation）与贝叶斯滤波开始：滤波需要一个生成观测模型 $\mathbf{z}(t)=\mathbf{h}(\mathbf{x}(t))$，而新视角合成恰好就是这个 $\mathbf{h}$ 的一个可微实现，于是 NeRF 不只是渲染技巧，而是一个能反过来做位姿估计与轨迹优化的场景表示。方法上，NeRF（neural radiance field，神经辐射场）把一个 MLP 当作体（volume）本身：输入 5D 坐标 $(x,y,z,\theta,\phi)$，输出体密度 $\sigma$ 与视角相关颜色 $\mathbf{c}$，再用经典体渲染的 $\alpha$-compositing 把积分离散成可微求和，从而只用像素颜色做监督。它成立靠三个关键设计：输入的位置编码（对抗 MLP 的低频偏置）、粗到细的分层采样（把算力集中到有内容的地方）、以及从 SfM（第 7 讲）拿到的相机位姿。代价同样明确：场景必须静态、位姿必须已知、训练与渲染都很慢——这正是下一讲 Gaussian Splatting 的出发点。

## 承接：新视角合成是生成观测模型的一种实现

课件 outline 第二页写的第一件事是「Recap: Filtering and Generative Observation Models」，随后用几页把贝叶斯滤波的图模型重画了一遍：控制输入 $u(t)$、状态 $x(t)$、观测 $z(t)$，以及算法 `Bayes_filter` 的 predict–update 两步（p037）。生成观测模型被明确写成

$$
\mathbf{z}(t) = \mathbf{h}\big(\mathbf{x}(t)\big),
$$

而课件的 novel view synthesis 页（p009）点出这一讲的目的：**新视角合成可以当作 $\mathbf{h}$ 的实现**。逻辑链条是——如果场景已经学成一个可以从任意位姿渲染的函数，那么「相机在世界中的位姿」就变成一个可以用滤波跟踪的状态：拿机载相机的一帧真实图像，和候选位姿下渲染出来的预测图像比，误差小就是好位姿。课件后面用一整段机器人工作（Adamkiewicz, Chen, Caccavale, Gardner, Culbertson, Bohg, Schwager，*Vision-Only Robot Navigation in a Neural Radiance World*，arXiv:2110.00168）来兑现这句话：状态估计的损失写成「真实图像与 $\mathrm{NeRF}(\vec{x})$ 之差的平方」再加上控制代价，轨迹优化则把每个时刻位姿的渲染损失与控制功累加起来，MPC 在滚动窗口上「规划—执行—重规划」。这条线是这一讲区别于「NeRF 教程」的地方：它把可微渲染放回了估计问题的框架里。课件在隐式表示那一节就埋好了伏笔——两个绿色对勾标出体渲染既对网络参数可导、也对相机位姿可导（p048），后者正是位姿估计与轨迹优化能直接用梯度做的原因。

## 表示的演化：从稀疏点云到连续体

课件用 Stonehenge 场景把四种表示并排放在一起（p043）：mesh、point cloud、voxels、implicit representations。前三者都是**显式**的——几何元素被逐个存下来；隐式表示则存一个函数，查询点得到密度或符号距离。

再往前追一层，传统多视图几何（第 7 讲 SfM、shape carving）给出的是**稀疏**点云：它足够估位姿，却不足以渲染照片。于是第一代的「可渲染表示」是 3D 体素 RGB-α 网格：先用多视图几何或网络预测一个离散的 voxel RGB-alpha grid，再沿射线做体渲染（p011、p016 引 Neural Volumes, Lombardi et al. 2019 与 Single-View Multi-Plane Images, Tucker and Snavely 2020）。课件给这条路的评价很直白（p017）：$\alpha$ 合成天然可微、和梯度优化配合得很好，但**3D 网格的存储开销糟糕**——分辨率提高是立方级代价。

另一条路是用神经网络表示连续几何，课件列了两组文献（p018）：用 3D 数据监督的 DeepSDF（Park et al. 2019）、Occupancy Networks（Mescheder et al. 2019）、Local Deep Implicit Functions（Genova et al. 2020）、Local Implicit Grids（Jiang et al. 2020）；用图像监督的 Scene Representation Networks（Sitzmann et al. 2019）、Differentiable Volumetric Rendering（Niemeyer et al. 2020）、DIST（Liu et al. 2020）。它的评价同样两句话（p019）：**高度可压缩**，但渲染模型受限、优化困难，因为形状被当成表面而不是体积。NeRF 就是把这两条路合起来：用 MLP 存（连续、可压缩），用体积渲染取（不需要显式表面）。

## 为什么是体渲染

课件先对比了两种渲染范式（p012）：表面渲染（surface rendering）沿视线打到表面即停，还要额外发射 shadow ray 去问光源；体渲染（volume rendering）则让射线穿过一团参与发射与遮挡的介质，沿途累积。给出用体渲染的三条理由（p013）：能表现烟雾这类弥散效果；从三维数据生成表面容易产生刺眼伪影，而体渲染的结果「软」；以及最关键的一条——**不需要先判断表面在哪里**，就能决定该反射多少光。

![左图为表面渲染：相机发出的 view ray 打在球面上，另有一根 shadow ray 指向光源；右图为体渲染：一条射线穿过云状体积，途中两个采样点各自向光源收集辐照，没有确定的表面交点](/coursenotes/cs231a-computer-vision-3d-sp25/l16-neural-radiance-fields/slide_p012.jpg)

物理模型（p014）是一根射线定义的空间圆柱里散布着粒子，粒子能发光、能遮挡后面的光、能反射环境光。把它写成方程就是课件的体渲染方程页（p015）：像素颜色等于背景项加沿途贡献，

$$
\mathbf{I}(D) = \mathbf{I}_0\,T(0) + \int_{0}^{D} \mathbf{c}(s)\,\rho(s)\,T(s)\,ds,
\qquad
T(s) = \exp\left(-\int_{s}^{D} \rho(t)\,dt\right),
$$

其中 $\rho$ 是密度（density）、$\mathbf{c}$ 是辐射（radiance）、$T$ 是透明度（transparency）。这里值得注意的是 $T$ 的形式：它不是「衰减系数」的乘积，而是密度沿射线的**积分取负指数**——这正是发射-吸收模型（emission-absorption）的解，也意味着密度天然对应「单位长度上射线被阻挡的概率」。

NeRF 论文里写的是同一条公式的射线版本（式 1，$\mathbf{r}(t)=\mathbf{o}+t\mathbf{d}$，起止 $t_n, t_f$）：

$$
C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\,\sigma\big(\mathbf{r}(t)\big)\,\mathbf{c}\big(\mathbf{r}(t), \mathbf{d}\big)\,dt,
\qquad
T(t) = \exp\left(-\int_{t_n}^{t} \sigma\big(\mathbf{r}(s)\big)\,ds\right).
$$

$T(t)$ 叫累积透射率（accumulated transmittance），解释为「射线从 $t_n$ 走到 $t$ 一路上没有撞上任何粒子的概率」。$\sigma(\mathbf{x})$ 可理解为射线在 $\mathbf{x}$ 处无穷小体积上终止的微分概率。渲染一张图，就是为每个像素的射线估一次这个积分。

![体渲染方程页：像素颜色等于背景项 I₀T(0) 加上沿射线的积分，被积函数拆成辐射 c、密度 ρ、透明度 T 三个因子，下方用眼睛到 D 的一维坐标轴和一个黑点表示某处的 (c, ρ) 对透射率的贡献](/coursenotes/cs231a-computer-vision-3d-sp25/l16-neural-radiance-fields/slide_p015.jpg)

## 离散化：从积分到 $\alpha$-compositing

连续积分必须数值化。课件的求积页（p023）给出的是求积形式：把射线分成 $N$ 段，第 $i$ 段采样点处网络给出 $(\mathbf{c}_i, \sigma_i)$，则

$$
C \approx \sum_{i=1}^{N} T_i\,\alpha_i\,\mathbf{c}_i,
\qquad
T_i = \prod_{j=1}^{i-1}\big(1-\alpha_j\big),
\qquad
\alpha_i = 1 - e^{-\sigma_i\,\delta t_i},
$$

其中 $\delta t_i$ 是相邻采样点间距。这三行是整讲最需要盯住的地方：第一个式子就是图形学里的**前到后 $\alpha$ 合成**（alpha compositing），权重 $T_i\alpha_i$ 是「第 $i$ 段之前的累计透过率 × 这一段自己遮住的比例」；第二个式子说明权重满足 $T_{i+1}=T_i(1-\alpha_i)$，可以一次前向递推算完；第三个式子把「密度 × 段长」翻译成「这一段的 $\alpha$」，也就是为什么 $\sigma$ 单独没有意义、必须和采样间距一起看。

![射线 r(t)=o+td 穿过一团 3D 体积：相机在左、射线向右穿过云状体积，采样点 t₁…t_N 依次标注，α_i 标为一段的厚度、T_i 标为相机到该点的累计透过率；左侧三个式子分别给出 C≈ΣT_iα_ic_i、T_i=Π(1−α_j)、α_i=1−e(−σ_iδt_i)](/coursenotes/cs231a-computer-vision-3d-sp25/l16-neural-radiance-fields/slide_p023.jpg)

NeRF 论文对离散化还有一个重要选择：不用确定性求积，而用**分层随机采样**（stratified sampling）。把 $[t_n,t_f]$ 均分成 $N$ 个区间，每个区间内均匀取一个样本，

$$
t_i \sim \mathcal{U}\left[t_n + \frac{i-1}{N}(t_f - t_n),\;\; t_n + \frac{i}{N}(t_f - t_n)\right].
$$

理由很几何：如果永远只在固定离散位置查询 MLP，那么网络的表达能力就被这套网格钉死了；分层采样让优化过程中查询点连续地铺满整条射线，才真正保住「连续表示」这个卖点。求积规则本身沿用 Max 的经典体渲染综述，$\delta_i$ 取相邻样本间距。

## NeRF 到底是什么函数

课件的定义只有一行（p020）：用神经网络作为体积表示，配合体渲染做视角合成，

$$
(x, y, z, \theta, \phi) \;\longrightarrow\; (\text{color}, \text{opacity}),
$$

配图（p021）把它画成一个 5D 输入、5D 输出的黑盒 $F_\Omega$：空间位置 $(x,y,z)$ 加视角方向 $(\theta,\phi)$，输出 $(r,g,b,\sigma)$，并强调**不需要实例化体积**——不用建网格、不用存体素，网络权重就是场景。课件图上标注的规模是「9 层、256 通道」，这与论文附录的描述基本对应：位置编码后的 $\gamma(\mathbf{x})$ 过 8 层 256 通道的全连接 ReLU 层，按 DeepSDF 的做法在第 5 层把输入再拼回去（skip connection），随后一层输出密度 $\sigma$（经 ReLU 保证非负）和一个 256 维特征；该特征与视角方向的编码 $\gamma(\mathbf{d})$ 拼接，再过一层 128 通道 ReLU，最后一层用 sigmoid 输出 RGB。

![NeRF 的网络接口图：(x,y,z,θ,φ) 经一个全连接网络 F_Ω 映射为 (r,g,b,σ)，输入分别标注为 spatial location 与 viewing direction，输出标注为 output color 与 output density，网络下方写着 9 layers / 256 channels，右下角用一组多平面图像示意「无需实例化体积」](/coursenotes/cs231a-computer-vision-3d-sp25/l16-neural-radiance-fields/slide_p021.jpg)

这里有个容易被跳过的设计判断：**$\sigma$ 只依赖位置，$\mathbf{c}$ 才依赖视角**。论文给的理由是多视图一致性——如果密度也随视角变化，同一个点在不同相机下就「不是同一个几何点」了；而把视角依赖放到颜色上，正好用极少的代价建模非朗伯（non-Lambertian）外观。课件的 viewing direction 页（p027）用 Ship 场景说清了这点：固定两个三维点，从船侧和从水面上两个方向看，颜色分布完全不同，而模型能把这半球上的变化连续地插出来。消融实验里去掉视角依赖，推土机履带上的高光就画不出来。

## 位置编码：为什么必需（课件未展开，据原论文）

需要说明：这一讲的课件没有单独的位置编码页，以下内容依据 NeRF 论文第 5.1 节与配套文献。

MLP 号称万能逼近器，但直接把 $(x,y,z,\theta,\phi)$ 喂进去，渲染结果会糊成一片。原因不是容量不够，而是**谱偏置**（spectral bias）：梯度下降训练的网络天然偏向先学低频函数（Rahaman et al., *On the Spectral Bias of Neural Networks*, ICML 2019）。一个只见过低频输入的网络，无法表达墙上的文字、远处的细枝、高光边缘这些高频内容。

解法是把输入先映射到更高维：$F_\Theta = F'_\Theta \circ \gamma$，其中

$$
\gamma(p) = \left(\sin\!\big(2^{0}\pi p\big),\ \cos\!\big(2^{0}\pi p\big),\ \cdots,\ \sin\!\big(2^{L-1}\pi p\big),\ \cos\!\big(2^{L-1}\pi p\big)\right),
$$

对 $\mathbf{x}$ 的三个分量分别施加（$L=10$），对笛卡尔视角向量 $\mathbf{d}$ 的三个分量分别施加（$L=4$），坐标都先归一化到 $[-1,1]$。这样网络在高频基函数张成的空间里做线性组合，就能拟合高频场。这个写法与 Transformer 的位置编码同形，但目的不同：Transformer 用它表达离散 token 的顺序，NeRF 用它把连续坐标抬到高频空间。同一结论在 Tancik et al., *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*（NeurIPS 2020）中被系统验证。论文消融显示：去掉位置编码，几何与纹理都过平滑，是三项设计里损失最大的一项（另外两项是视角依赖与分层采样）。$L$ 也不是越大越好——当 $2^L$ 超过输入图像本身的最大频率（约 1024）后再加频率没有收益。

## 分层采样：coarse 到 fine

有了高频表达能力，新问题就来了：沿射线密密地均匀撒点非常浪费——自由空间和被遮挡的区域根本不贡献颜色，却照样要查询网络。课件用三页（p024–p026）提出「能不能更聪明地分配采样点」，答案是两趟渲染（two pass rendering）：先均匀撒 $N_c$ 个点用 coarse 网络渲染，把它的权重 $T_i\alpha_i$ **当作概率分布**，再从这个分布里采出第二批点，交给 fine 网络。

论文的写法更精确。coarse 渲染可写成加权和

$$
\hat{C}_c(\mathbf{r}) = \sum_{i=1}^{N_c} w_i\,\mathbf{c}_i,
\qquad
w_i = T_i\big(1-\exp(-\sigma_i\delta_i)\big),
$$

把 $w_i$ 归一化就得到沿射线的分段常数概率密度，再用逆变换采样（inverse transform sampling）取 $N_f$ 个点，在**两批点的并集**上求值 fine 网络并算出最终颜色。论文特意强调：这和重要性采样目标相同，但做法不同——它不是把每个样本当作对整个积分的独立概率估计，而是把样本点当作对整个积分域的**非均匀离散化**。默认配置是 $N_c = 64$、$N_f = 128$。

## 损失、训练与位姿来源

体渲染的求和形式全是初等函数，因此整条管线对网络参数可微；课件的可微性页（p028）指出这一点，损失页（p029）给出训练目标：对每张训练图的每条射线，让渲染颜色逼近真实像素颜色，

$$
\min_{\Omega}\ \sum_{i} \left\lVert \mathrm{render}^{(i)}(F_\Omega) - \mathbf{I}_{\mathrm{gt}\,}^{(i)} \right\rVert^{2},
\qquad
\mathcal{L} = \sum_{\mathbf{r}\in\mathcal{R}\,} \left[\ \lVert \hat{C}_c(\mathbf{r}) - C(\mathbf{r}) \rVert_2^2 + \lVert \hat{C}_f(\mathbf{r}) - C(\mathbf{r}) \rVert_2^2\ \right].
$$

左边是课件写法，右边是论文写法：coarse 和 fine 两趟都要算损失。coarse 的监督不是可有可无——只有它的权重分布准了，fine 的采样才采得到点上。

![训练示意图：两条射线 Ray 1、Ray 2 穿过乐高推土机场景的包围盒，采样点送入网络 F_Ω 后渲染出像素颜色，与真值比较；下方损失写成对每条射线的密度剖面与真值密度剖面之差的 L2 之和](/coursenotes/cs231a-computer-vision-3d-sp25/l16-neural-radiance-fields/slide_p029.jpg)

训练细节（论文 5.3 节与附录）：**每个场景单独训练一个网络**，不需要任何 3D 标注；输入只有 RGB 图像、对应相机位姿与内参、以及场景范围。合成数据用真值位姿，**真实数据用 COLMAP 的 SfM 估计位姿与内参**（Schönberger & Frahm, *Structure-from-Motion Revisited*, CVPR 2016）——这正是第 7 讲的产物，也解释了为什么 NeRF 常和「SfM 失败则 NeRF 失败」这句话一起出现。每次迭代随机取一批相机射线（batch 4096 条），Adam 优化，学习率从 $5\times10^{-4}$ 指数衰减到 $5\times10^{-5}$，单场景 100k–300k 次迭代，一张 V100 上约 1–2 天。真实场景还要在 $\sigma$ 上加高斯噪声做正则，并用 NDC（归一化设备坐标）把无限远深度压进 $[-1,1]$。

推理时每条射线查询 64 + 192 = 256 次网络，一张图约需 1.5–2 亿次查询，V100 上约 30 秒一帧——这个数字是理解后续一整代加速工作（以及下一讲 Gaussian Splatting）的锚点。

## 结果与几何可视化

课件的结果页分三类。合成数据（p031）展示论文基准里的 8 个 Blender 场景：椅子、乐高推土机、材质球、架子鼓、麦克风、热狗、船、榕树。视角相关外观（p032、p033）用会议室屏幕和灶台不锈钢台面做例子——这两处恰好是最考验非朗伯建模的镜面/半镜面表面。几何可视化（p034、p035）把渲染图和密度图并排：左侧花朵照片、右侧对应的密度可视化能直接读出花瓣与叶片的轮廓；圣诞树那组里，镂空铃铛的孔洞都在密度里显形。这一步很重要，它说明 $\sigma$ 不只是个渲染用的中间量，而是**可以当几何来读**——这也是后面用 NeRF 做碰撞检测、导航的前提。课件的真实场景页（p036）给出两块真实采集场景——岩石上的松果、木桌上的花束；论文的真实场景基准则是 LLFF 前向拍摄数据，两者都属于「从稀疏真实照片学辐射场」这一设定。

![几何可视化页：左半是一丛橙红色花朵的照片，右半是同一视角下的密度可视化，花瓣与叶片的轮廓以亮到暗的灰度直接呈现，背景是低密度的雾状体积](/coursenotes/cs231a-computer-vision-3d-sp25/l16-neural-radiance-fields/slide_p034.jpg)

## 局限

课件没有单列「局限」页，以下由课件内容与论文表述共同支撑：

- **静态假设**。辐射场是对整个场景的一次拟合，不随时间变化；物体移动、光照变化都会破坏多视图一致性。
- **需要已知位姿**。真实场景依赖 SfM 输出，而 SfM 在弱纹理、反光、重复结构区域会失败，位姿一错，几何就糊。
- **优化与渲染开销大**。逐场景训练、每帧上亿次网络查询，既不能泛化也不能实时。
- **非朗伯与稀疏视角仍然困难**。高光依赖足够多的视角覆盖；输入图像减少时质量明显下降（论文消融 5–6 行）。
- **可解释性差**。论文自己指出：体素、网格这类采样式表示可以预判渲染质量和失效模式，而把场景编码进网络权重后，这类分析没有好办法。
- **边界框敏感**。查询范围由 $t_n, t_f$ 限定，范围给错会直接截掉内容或引入漂浮物。

## 要点回顾

- 这一讲把 NeRF 放在最优估计/滤波的延长线上：新视角合成是生成观测模型 $\mathbf{z}(t)=\mathbf{h}(\mathbf{x}(t))$ 的可微实现，可反过来用于位姿估计与轨迹优化。
- 显式表示（mesh、点云、体素）存几何元素，隐式表示存函数；体素 RGB-α 网格可微但存储立方级增长，MLP 连续表示可压缩但需要合适的渲染模型。
- 选体渲染的核心理由：不必先确定表面位置，能表达半透明与弥散效果，结果对采样误差更「软」。
- 连续体渲染方程是发射-吸收模型：$C(\mathbf{r})=\int T(t)\sigma\mathbf{c}\,dt$，$T(t)=\exp(-\int\sigma)$ 是累积透射率，即射线未撞上粒子的概率。
- 离散化后就是 $\alpha$-compositing：$\hat{C}=\sum_i T_i\alpha_i\mathbf{c}_i$，$\alpha_i=1-e^{-\sigma_i\delta_i}$，$T_i=\prod_{j \lt i}(1-\alpha_j)$；密度只有和采样间距结合才有意义。
- 用分层随机采样而非确定性求积，才能保证查询点连续覆盖，从而真正体现「连续表示」。
- NeRF 是 $F_\Theta:(x,y,z,\theta,\phi)\to(\mathbf{c},\sigma)$；密度只依赖位置以保证多视图一致，颜色依赖视角以建模非朗伯高光。
- 位置编码 $\gamma(p)$ 用 $2^0,\dots,2^{L-1}$ 倍频的正余弦抬升输入频率，对抗 MLP 的低频谱偏置；$L=10$（位置）、$L=4$（方向）。
- coarse/fine 分层采样把 coarse 的渲染权重归一化成沿射线的概率密度，逆变换采样出 fine 的点，显著减少自由空间上的浪费。
- 损失是 coarse 与 fine 两趟渲染颜色对真实像素的 L2；两趟都要监督，否则 fine 的采样分布学不好。
- 每个场景单独训练一个 MLP；真实场景的位姿、内参、范围来自 COLMAP SfM（第 7 讲），合成数据用真值。
- 训练量级：batch 4096 射线、64+128 采样点、Adam 5e-4→5e-5、100k–300k 迭代、单卡 1–2 天；渲染约 30 秒/帧。
- 密度场可视化表明 $\sigma$ 可当几何使用，这是把 NeRF 用于机器人碰撞检测与导航的基础。
- 局限集中在静态场景、依赖已知位姿、逐场景训练、渲染开销大、可解释性差；课件最后一页预告下一讲 Gaussian Splatting，正是对效率问题的回应。

## 参考资料

- 课件：CS231A Lecture 16, *Neural Radiance Fields for Novel View Synthesis*，2025-05-27，Jeannette Bohg & Silvio Savarese；课程主页与大纲见 [CS231A Spring 2025](https://cs231a.stanford.edu/)（syllabus 中第 16 讲列出的阅读材料为 NeRF 项目页与 NeRF Navigation 论文）。
- 核心论文：Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng, *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*, ECCV 2020（Oral, Best Paper Honorable Mention），[arXiv:2003.08934](https://arxiv.org/abs/2003.08934)，项目页 <https://www.matthewtancik.com/nerf>。
- 位置编码的理论依据：Nasim Rahaman et al., *On the Spectral Bias of Neural Networks*, ICML 2019, [arXiv:1806.08734](https://arxiv.org/abs/1806.08734)；Matthew Tancik et al., *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*, NeurIPS 2020, [arXiv:2006.10739](https://arxiv.org/abs/2006.10739)。
- 位姿来源：Johannes L. Schönberger, Jan-Michael Frahm, *Structure-from-Motion Revisited*, CVPR 2016（COLMAP）。
- 课件引用的前驱工作：Stephen Lombardi et al., *Neural Volumes: Learning Dynamic Renderable Volumes from Images*, SIGGRAPH 2019, [arXiv:1906.07751](https://arxiv.org/abs/1906.07751)；Tucker and Snavely, *Single-View Multi-Plane Images*, 2020（课件题录，未检索到公开版本）；Park et al. DeepSDF 2019、Mescheder et al. Occupancy Networks 2019、Genova et al. LDIF 2020、Jiang et al. Local Implicit Grids 2020、Sitzmann et al. SRN 2019、Niemeyer et al. DVR 2020、Liu et al. DIST 2020（均按课件 p018 的题录列出）。
- 机器人应用：Michal Adamkiewicz, Timothy Chen, Adam Caccavale, Rachel Gardner, Preston Culbertson, Jeannette Bohg, Mac Schwager, *Vision-Only Robot Navigation in a Neural Radiance World*, 2021, [arXiv:2110.00168](https://arxiv.org/abs/2110.00168)。
- 课件中体渲染与采样相关页码取自 Matthew Tancik 的公开演讲 *Neural Radiance Fields for View Synthesis*（2020），课件已注明出处。
