---
title: 第 47 章 · 三维运动及其二维投影
chapter: 47
original: https://visionbook.mit.edu/2d_motion_from_3d.html
---

# 第 47 章 · 三维运动及其二维投影

<div class="src-note">
本页为《Foundations of Computer Vision》第 47 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/2d_motion_from_3d.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

Part XII「理解运动」的第二章，讲清楚运动估计的「正问题」——3D 世界中的运动（点的运动、相机平移/旋转/变焦）如何经透视投影变成像面上的 2D 运动场。第 46 章尚未理解运动形成过程就直接估计了像素位移；本章补上这一环，给出 (47.1)–(47.9) 的解析关系。它们决定了「什么样的世界运动对应什么样的像面运动」，是后续所有运动估计算法的模型基础。

## 本篇要点

- 透视投影的瞬时形式：$(\dot{x},\dot{y})^\mathsf{T}=\frac{1}{Z}\begin{bmatrix}f&0&-x\\0&f&-y\end{bmatrix}(\dot{X},\dot{Y},\dot{Z})^\mathsf{T}$ (47.1)，一切可观测运动都经过 $1/Z$ 这个深度因子。
- 平行于像面的运动：像上速度 $=(f/Z)\times$ 3D 速度，方向相同、大小与深度成反比 (47.2)——这就是运动视差。
- 沿光轴的运动：像上速度 $\propto-(\dot{Z}/Z)(x,y)$，是辐射状的扩张/收缩 (47.3)；相机前进时运动场以扩张焦点（FOE）为中心，速率只取决于 $V_Z/Z$，即接触时间的倒数。
- 匀速直线运动的点收敛到消失点 (47.4)，只由 3D 速度方向决定、与初始位置无关；同速度的不同点收敛到同一消失点。
- 相机旋转：小位移下运动场与深度 $Z$ 无关 (47.8)，只含角速度与内参——纯旋转不透露任何场景结构；大角度旋转下两帧之间是单应关系。
- 改变焦距只缩放图像、不产生视差：同视点、不同焦距的针孔相机图像不含深度信息。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 3 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/optical_flow/examples_3d_moving_points.webp" alt="图 47.2 3D 运动投影的几何分解" loading="lazy" />
<figcaption>图 47.2：三维运动向相机投影的几何。（左）平行于相机平面运动的点；（右）平行于相机光轴运动的点。像平面上的箭头表示成像后的场景速度。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：一张图概括了 3D 运动的两种基本投影方式——平行像面（大小 ∝ f/Z）与沿光轴（辐射状），是分析任何运动场前的第一块积木。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/optical_flow/camera_forward_translation_flow.webp" alt="图 47.7 扩张焦点与运动场" loading="lazy" />
<figcaption>图 47.7：相机接近平面时的运动场示意。运动场表示图像的扩张速率，是接触时间的函数；本例中扩张焦点位于中心。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：运动场从 FOE 向外辐射这一形态，是「从光流估计相机运动方向/碰撞点」类算法（如机器人避障）的直接依据。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/optical_flow/camera_motion_field_rotation_three_f.webp" alt="图 47.10 焦距对旋转运动场的影响" loading="lazy" />
<figcaption>图 47.10：绕 Y 轴旋转对应的运动流。（左）f=1/3；（中）f=1；（右）f=3。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：同一个旋转，不同焦距下运动场外观完全不同——估计相机旋转时必须把内参一起解出来，否则容易把旋转误判成平移或单应。</p>

## 本章小结

- 透视投影的瞬时速度关系 (47.1)：一切像面运动都经过 $1/Z$ 深度因子；这是运动视差，也是由运动恢复深度的根源。
- 平行像面运动 → 像速度 $=(f/Z)\times$ 3D 速度 (47.2)；沿光轴 → 辐射状 (47.3)。
- 匀速直线运动的点收敛到消失点 (47.4)，只取决于速度方向；相机前向运动 → 以 FOE 为中心的扩张场 (47.6)，速率 $=V_Z/Z$（接触时间的倒数）。
- 小位移下相机旋转的运动场 (47.8) 与 $Z$ 无关：纯旋转不告诉你场景结构，只告诉你相机怎么转、内参多少；大角度旋转 → 单应。
- 变焦只缩放、不产生视差：同视点不同焦距的针孔相机图像不含深度信息。

## 复习自检清单

- ☐ 能从 $x=fX/Z$ 推出瞬时速度关系 (47.1)，并说明式中各项的来源与 $1/Z$ 因子的意义
- ☐ 能写出并区分 (47.2)(47.3)(47.5)(47.6)：平行像面运动、沿光轴运动、相机平移、相机前向运动各自的运动场
- ☐ 能推导匀速直线运动点的消失点 (47.4)，并说明它为何与初始位置无关、同速度的多点为何共消失点
- ☐ 能解释扩张焦点（FOE）与接触时间 $V_Z/Z$ 的关系
- ☐ 能由欧拉角小角近似推出 $\dot{\mathbf{P}}=-\mathbf{V}-\mathbf{W}\times\mathbf{P}$ (47.7)，并组装出旋转运动场 (47.8)
- ☐ 能说明为什么纯旋转运动场不含场景结构信息，以及大角度旋转为何对应单应
- ☐ 能解释变焦只产生缩放、不产生视差，因此同视点不同焦距的针孔相机图像不含深度

---

原文：[3D Motion and Its 2D Projection](https://visionbook.mit.edu/2d_motion_from_3d.html)
