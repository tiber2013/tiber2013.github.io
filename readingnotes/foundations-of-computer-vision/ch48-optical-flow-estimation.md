---
title: 第 48 章 · 光流估计
chapter: 48
original: https://visionbook.mit.edu/optical_flow.html
---

# 第 48 章 · 光流估计

<div class="src-note">
本页为《Foundations of Computer Vision》第 48 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/optical_flow.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

第 47 章说明了三维场景（或相机）的运动如何在图像上产生二维运动场（motion field），本章解决如何从相机记录的图像序列中**测量**这个二维位移场，即光流（optical flow）估计。第 46.3 节的基于匹配的方法速度慢且只能给出离散像素位移；本章介绍无需训练的基于梯度的方法（Lucas-Kanade、Horn-Schunck），它们引入的概念（亮度恒定假设、光度重建误差、正则化）正是第 49 章深度学习方法的基石。

## 本篇要点

- 光流是对二维运动场的近似，靠测量**图像亮度**的位移得到；它与真实运动场并不总是一致（光照变化、镜面反射、无纹理表面都会破坏对应关系）。
- 亮度恒定假设 $\ell(x+u,y+v,t+1)=\ell(x,y,t)$ 是全部经典方法的出发点，但在现实中只是近似成立。
- 孔径问题（aperture problem）：一维边缘只能测出垂直于边缘的运动分量；Lucas-Kanade 用邻域窗口内多个像素的梯度约束联立解出唯一速度。
- Lucas-Kanade 在每个像素解一个 $2\times2$ 线性系统 $\mathbf{u}=\mathbf{A}^{-1}\mathbf{b}$；当 $\mathbf{A}$ 秩为 1（窗口内只有一维结构）时光流被约束在一条直线上——这正是孔径问题的代数表现。
- 一阶泰勒近似只适用于小位移，大位移需配高斯金字塔由粗到精；Horn-Schunck 则在光度误差上叠加平滑正则。
- 基于图层（layer-based）的表示把场景分解为若干刚性运动层，用 EM 风格的迭代估计每层的运动参数。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 3 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/optical_flow/apperture_problem.webp" alt="图 48.2 孔径问题" loading="lazy" />
<figcaption>图 48.2：孔径问题：观察一个大于图像画面的一维（1D）结构的运动。条纹的实际运动是向上的，但当视觉被限制在观测窗口内可见的部分时，感知到的运动方向却垂直于条纹。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：孔径问题是理解一切基于梯度光流方法的钥匙——它解释了为什么单个像素、单个小窗口常常测不出完整运动，也直接对应后面 A 矩阵秩亏的代数现象。理发店柱错觉（barber-pole illusion）是它的日常版：旋转的条纹条柱会让人产生向下运动的错觉。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/optical_flow/square_grandient_based_4.webp" alt="图 48.8 用 R 筛选可靠区域" loading="lazy" />
<figcaption>图 48.8：在 R>2 区域内的估计光流（围绕适合跟踪的良好特征 [3]）。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这张图把「光流在哪里可信」可视化出来了——只在角点/纹理充分处可信，一维结构处失效。R 判据与 Harris 角点同源，工程上可直接把角点检测当作光流可估计性的预筛。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/optical_flow/comparison_gradient_vs_iterative.webp" alt="图 48.10 一阶近似 vs. 多尺度迭代" loading="lazy" />
<figcaption>图 48.10：基于梯度的算法与多尺度迭代精化方法所得光流的对比。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是本章最直观的「近似失效」证据——大位移下单步线性解会系统性低估运动，而金字塔 + 迭代修正能救回来。现代方法（含 RAFT 类网络）本质上仍在延续「由粗到精 + 增量更新」这一思想。</p>

## 本章小结

- 光流是通过测量图像亮度位移得到的、对二维运动场的近似；它与真实运动场在光照变化、镜面反射、无纹理表面等情形下并不一致。
- 亮度恒定假设 $\ell(x+u,y+v,t+1)=\ell(x,y,t)$ 是经典方法的出发点，现实中只是近似成立（遮挡、阴影、相互反射都会破坏它）。
- 梯度约束方程 $u\ell_x+v\ell_y+\ell_t=0$ 只把运动约束在垂直于图像梯度的一条直线上，即孔径问题；Lucas-Kanade 用邻域窗口内多像素约束联立解出唯一速度 $\mathbf{u}=\mathbf{A}^{-1}\mathbf{b}$，$\mathbf{A}$ 秩亏时失效。
- 用 Harris 的 $R=\det(\mathbf{A})-\lambda\,\mathrm{Tr}(\mathbf{A})^2$（$\lambda=0.05$）可筛选运动可靠的区域；增大窗口能扩大可靠区但会模糊运动边界。
- 一阶泰勒近似只适合小位移，大位移需高斯金字塔由粗到精 + 迭代精化；Horn-Schunck 在光度重建误差上加慢速/平滑正则项。
- 基于图层的表示适合刚性多物体场景，用 EM 风格迭代估计每层运动参数。

## 复习自检清单

- ☐ 能写出亮度恒定假设 (48.1) 及其等价形式 $\ell(x,y,t+1)=\ell(x-u,y-v,t)$，并说明它为什么只是近似成立
- ☐ 能由一阶泰勒展开 (48.2) 推出梯度约束方程 $u\ell_x+v\ell_y+\ell_t=0$，并解释它与孔径问题的关系
- ☐ 能写出 Lucas-Kanade 的 $2\times2$ 正规方程 (48.3)，解释 $\mathbf{A}$ 的秩与可逆性如何决定光流是否唯一可测
- ☐ 能解释 Harris 的 $R$ 判据为何与 Harris 角点同源，$\lambda=0.05$ 是什么，以及它如何用于筛选可靠运动区域
- ☐ 能说明为什么大位移会破坏一阶泰勒近似，以及高斯金字塔 + 迭代精化如何补救（对照图 48.10）
- ☐ 能写出 Horn-Schunck 的正则化目标 (48.4) 及慢速先验、平滑先验两种正则项
- ☐ 能描述基于图层的运动估计为何适合刚性多物体场景，以及 EM 迭代中两步各自估计什么
- ☐ 能举出至少三个「图像有运动而场景无运动」或反之的例子，并说明其物理成因

---

原文：[Optical Flow Estimation](https://visionbook.mit.edu/optical_flow.html)
