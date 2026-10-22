---
title: 第 20 章 · 图像采样与混叠
chapter: 20
original: https://visionbook.mit.edu/sampling_and_aliasing.html
---

# 第 20 章 · 图像采样与混叠

<div class="src-note">
本页为《Foundations of Computer Vision》第 20 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/sampling_and_aliasing.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

Part VI「采样与多尺度表示」的第一章，也是全书所有分辨率操作的理论基石。本章回答：连续信号如何变成离散信号、信息在其中如何损失、什么条件下可以完美重建，以及损失不可避免时（混叠）如何抑制。第 21 章的下采样/上采样、第 22 章滤波器组、第 23 章图像金字塔都直接建立在这些结论之上。

## 本篇要点

- 奈奎斯特采样定理：采样角频率 $w_s=2\pi/T_s$ 必须大于信号最高频率 $w_{max}$ 的两倍（等价地 $T_s<T_{min}/2$），否则无法从样本完美重建。
- 混叠的本质：采样让频谱以 $2\pi/T_s$ 为周期复制，复制品一旦重叠，高频内容就折叠进低频，冒充成根本不存在的低频结构。
- 满足奈奎斯特条件时，理想重建核是 sinc 函数（盒式低通滤波器的脉冲响应），是「慢而平滑」先验下 L2 最优的插值。
- sinc 支撑无限长，工程上用局部核近似：最近邻（盒）、线性（三角）、Lanczos-1（sinc 中央瓣）。
- 2D 采样不限于矩形网格：六边形采样在同样样本数下约提升 10% 分辨率，视网膜中央凹的视锥分布就接近六边形。
- 抗混叠滤波 = 采样前先低通滤波，把网格表示不了的高频成分提前抹掉；高频信息照样损失，但可以避免「斑马变奶牛」式的假结构。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 5 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/Image_processing_sampling/sampling_reconstruction1.webp" alt="图 20.3 采样点的重建歧义" loading="lazy" />
<figcaption>图 20.3：有无穷多条波（只画出两条）精确穿过所有采样点。红色曲线是「慢而平滑」先验下的重建结果。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是理解混叠的入口——同样的样本点，快速振荡的原信号与被优先选择的低频信号都能完美穿过；采样并没有「删掉」高频，而是把它「误标」成了低频。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/Image_processing_sampling/sketch_aliasing.webp" alt="图 20.8 混叠的频谱示意" loading="lazy" />
<figcaption>图 20.8：混叠示意。带限信号，频率成分仅在区间 (-w_max, w_max) 内。(a) 采样周期满足 T_s<π/w_max，频谱拷贝互不重叠；(b) 采样周期 T_s>π/w_max，拷贝重叠。混叠源于信号傅里叶变换 mathscrL(w) 各平移拷贝之间的重叠。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：全章最重要的机制图——「频谱复制 + 拷贝重叠」解释了混叠的全部本质，后面的 2D 采样、抗混叠滤波都是这张图的延伸。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/Image_processing_sampling/sampling_reconstruction2.webp" alt="图 20.10 sinc 插值重建" loading="lazy" />
<figcaption>图 20.10：用 sinc 函数对采样函数做平滑插值。每个 sinc 的零点都落在采样点上，因此插值严格经过所有样本。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是「理想重建」的标准画面——sinc 叠加既是数学最优解，也解释了为什么理想插值需要无限支撑，工程上只能近似。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/Image_processing_sampling/sampling_grids_FT.webp" alt="图 20.14 矩形 vs 六边形采样的频谱" loading="lazy" />
<figcaption>图 20.14：矩形与六边形采样傅里叶变换的草图。红色边界标示被周期复制的频谱内容范围。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：有效频率区域的形状直接决定抗混叠能力——六边形近似「圆形」的有效区比矩形「方形」更接近各向同性，这是它约 10% 增益的来源。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/Image_processing_sampling/aliasing_ft_abc1.webp" alt="图 20.16 斑马混叠实验" loading="lazy" />
<figcaption>图 20.16：带混叠的下采样。(a) 带混叠采样的斑马开始看起来像奶牛。(b) 连续信号 ℓ(x,y) 乘以 delta 串后的傅里叶变换 ℓ_δ(x,y)。(c) 对应采样信号 ℓ[n,m] 的 DFT。(d) 带混叠重建信号的傅里叶变换。(e) 经抗混叠滤波处理后的采样图像。(f) 对应抗混叠采样图像 ℓ[n,m] 的 DFT——注意其中一部分不再随分辨率变化。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：「斑马变奶牛」是抗混叠教学中的经典一幕；(b)(c)(f) 三列频谱对照把「先滤波再采样」为什么有效讲得无可辩驳。</p>

## 本章小结

- 采样把连续域变离散域；奈奎斯特定理给出可完美重建的条件：$w_s>2w_{max}$，即 $T_s<T_{min}/2$。
- 混叠的频域本质：采样使频谱以 $2\pi/T_s$ 为周期复制，$T_s>\pi/w_{max}$ 时拷贝重叠，高频折叠成假低频；「慢而平滑」先验让感知/重建系统优先选择低频解释。
- 建模工具是 delta 串（Dirac 梳）：采样 = 信号乘 delta 串，其 FT = 原 FT 与 delta 梳 FT 的卷积 = 原 FT 的周期复制；且 $\mathscr{L}_s(w)=\mathscr{L}_\delta(w/T_s)$。
- 满足奈奎斯特条件时，理想重建 = sinc 插值（盒式低通的脉冲响应），L2 最优；sinc 无限支撑，工程上用最近邻/线性/Lanczos-1 等局部核近似。
- 2D 采样的一般形式由矩阵 $\mathbf{A}$ 决定（式 20.2）；六边形采样的有效频率区域更大，同样本数下约 +10% 分辨率（视网膜中央凹接近六边形）。
- 抗混叠 = 采样前低通滤波；不能挽回已丢的高频信息，但能避免「斑马变奶牛」式的假结构。时空采样中对应全局快门/卷帘快门、车轮效应、摩尔纹等现象。

## 复习自检清单

- ☐ 能写出奈奎斯特采样定理的两种等价表述（$w_s>2w_{max}$ 与 $T_s<T_{min}/2$），并用 20.2 节的余弦例子说明它为何被违反
- ☐ 能推导「采样 = 乘 delta 串」，并说明 $\mathscr{L}_s(w)=\mathscr{L}_\delta(w/T_s)$ 的含义
- ☐ 能由 delta 梳的傅里叶变换推出 $\mathscr{L}_\delta(w)=\frac{2\pi}{T_s}\sum_k\mathscr{L}(w-k2\pi/T_s)$，并用它解释混叠的产生条件
- ☐ 能写出盒式重建滤波器 (20.1) 及其脉冲响应 sinc，说明 sinc 零点为何与采样点重合
- ☐ 能解释「信号不能同时时间有限与频率带限」对实际图像处理意味着什么
- ☐ 能写出 2D 一般采样的矩阵 $\mathbf{A}$ 形式与式 (20.2)，并说明矩形/六边形采样各自的参数取法
- ☐ 能解释为什么六边形采样比矩形采样效率高约 10%，以及抗混叠滤波器能做什么、不能做什么
- ☐ 能区分全局快门与卷帘快门，并各举一个混叠相关的实际现象（车轮效应/摩尔纹/卷帘畸变）

---

原文：[Image Sampling and Aliasing](https://visionbook.mit.edu/sampling_and_aliasing.html)
