---
title: 第 42 章 · 单视图度量学
chapter: 42
original: https://visionbook.mit.edu/3d_scene_understanding_single_view.html
---

# 第 42 章 · 单视图度量学

<div class="src-note">
本页为《Foundations of Computer Vision》第 42 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/3d_scene_understanding_single_view.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

本章属于几何部分（第 38–45 章）中「模型式三维恢复」的收官之作。第 40 章立体视觉用两个视图恢复深度，本章挑战更苛刻的设定——只凭一张图像恢复度量三维信息，武器是线性透视（消失点、消失线）与射影不变量（交比）。下一章（43）转向学习式方法，第 44 章则走向另一个极端：用尽可能多的视图（SfM）。

## 本篇要点

- 消失点只取决于三维直线的方向：$\mathbf{v}=[fD_X/D_Z,\ fD_Y/D_Z]^\mathsf{T}$；一组平行线共点，$D_Z=0$ 时消失点在无穷远。
- 齐次坐标下 $\mathbf{v}=\mathbf{K}\mathbf{R}\mathbf{D}$：消失点与相机平移无关，只有旋转才能改变它。
- 同一平面内所有平行线的消失点共线——这条线就是该平面的地平线（horizon line，又称消失线 vanishing line）；平行平面共享同一条地平线。
- 交比是透视投影下最重要的射影不变量：4 点交比投影前后相等，是「把参考尺寸搬运到别处」的数学工具。
- 单视图测高配方：地平线 + 消失点 + 一个已知高度的参考物，叉积投影 + 交比求解；高度可沿「被支撑层级」传播。
- 三个正交消失点足以线性标定相机内参：$\mathbf{v}_i^\mathsf{T}\mathbf{K}^{-\mathsf{T}}\mathbf{K}^{-1}\mathbf{v}_j=0$。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 8 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/single_view_3d/vanishing_points.webp" alt="图 42.5 三维直线的投影与消失点" loading="lazy" />
<figcaption>图 42.5：三维直线向相机平面的投影。三维直线与相机中心（位于世界坐标原点）确定一个平面，该平面与相机平面的交线就是图像中出现的二维直线。图像点 v 即消失点。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是全部消失点几何的出发点——「直线 + 相机中心 = 平面，交线 = 图像直线」，务必把这张图的几何关系看熟。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/single_view_3d/vanishing_lines_and_planes.webp" alt="图 42.6 平行线收敛于消失点" loading="lazy" />
<figcaption>图 42.6：平行三维直线收敛到同一消失点。(a) 红色直线平行于光轴，收敛到图像中心的消失点。(b) 两组包含在同一水平面内的平行直线（平行于光轴）。它们的消失点位于同一条水平图像线上。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：「平行 → 共点」是线性透视的基石，也是本章所有测量的出发点。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/single_view_3d/palma_horizon.webp" alt="图 42.8 地平线" loading="lazy" />
<figcaption>图 42.8：平面内所有平行直线收敛到一组消失点，所有这些消失点都位于该平面的地平线上。平行平面共享同一条地平线。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：地平线把「一族消失点」组织成「一个平面的身份」，是后续一切测量的基准线。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/imaging_geometry/horizon_line.webp" alt="图 42.9 叉积计算消失点与地平线" loading="lazy" />
<figcaption>图 42.9：用齐次坐标下的平行线计算消失点和地平线。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：整章测量算法的计算骨架就是「叉积求交」：两条平行线交出消失点，两个消失点交出地平线，全部是齐次坐标下的 × 运算。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/single_view_3d/cross_ratio_description2.webp" alt="图 42.11 交比：射影不变量" loading="lazy" />
<figcaption>图 42.11：点集 P 与点集 Q（以及 R）通过交比相联系。点 R_1 在无穷远处，因为连接 R 各点的直线平行于经过 O、P_1、Q_1 的直线。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是全章唯一一个「投影前后严格相等」的数值量，单视图测量的一切都建立在它上面。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/single_view_3d/office_measuring_desk.webp" alt="图 42.14 桌高向书架的投影（含具体坐标）" loading="lazy" />
<figcaption>图 42.14：桌高向书架的投影。取书架竖直边缘上底部与顶部的图像坐标：t_b = [702\ \ 2,958]^mathsfT、g_b= [987\ \ 618]^mathsfT，以及桌子的 t_d = [679\ \ 1,018]^mathsfT、g_d= [793\ \ 59]^mathsfT。按正文方法计算 a= [4,471\ \ 2,486]^mathsfT 与 b= [2,237\ \ 2,765]^mathsfT。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是全章唯一给出全部中间坐标的算例，复习时务必能拿这四个点把整条叉积链复现一遍。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/single_view_3d/axis_calibration_procedure.webp" alt="图 42.18 用交比标定坐标轴刻度" loading="lazy" />
<figcaption>图 42.18：仅用消失点，通过交比不变量把世界坐标投影到图像中。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这张图把「1 米在图像上不是一个固定像素长度」变成了可操作的算法——刻度沿轴按交比递推摆放。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/single_view_3d/locating_a_3d_point.webp" alt="图 42.20 定位三维点" loading="lazy" />
<figcaption>图 42.20：(a) 没有任何其他信息时，无法恢复点 P 的三维位置。(b) 如果给定把点 P 投影到地面的方法，就可以通过向世界轴投影读出该点的坐标。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是单视图度量的「完全体」——有了标定轴，任何接地物体上的点都能读出三维坐标，而不只是高度。</p>

## 本章小结

- 单张图像包含极强的三维线索；线性透视是其中最强的图像式线索，但主要在人造环境有效。
- 消失点只由三维方向决定：$\mathbf{v}=\mathbf{K}\mathbf{R}\mathbf{D}$，与相机平移无关；同一平面内平行线的消失点共线成地平线（消失线），平行平面共享同一条地平线。
- 交比是透视投影下最重要的射影不变量；一点在无穷远时退化为两段距离之比，这是本章测高公式的源头。
- 单视图测高配方：地平线 + 消失点 + 一个已知高度的参考物；叉积链把待测高度「搬」到参考物上，再用交比解出真实高度；高度可沿支撑层级继续传播（桌 → 桌上瓶子）。
- 标定世界坐标轴刻度后，可直接读出接地点的三维坐标与相机离地高度。
- 三个正交消失点可线性标定内参：$\mathbf{W}=\mathbf{K}^{-\mathsf{T}}\mathbf{K}^{-1}$ 的四参数结构 + SVD + Cholesky 分解。

## 复习自检清单

- ☐ 能由式 (42.2) 取 $t\to\infty$ 推出消失点公式 (42.3)，并解释 $D_Z=0$ 时消失点为何在无穷远
- ☐ 能写出 $\mathbf{v}=\mathbf{K}\mathbf{R}\mathbf{D}$，并说明为什么平移相机不改变消失点
- ☐ 能解释「平面内平行线消失点共线」「平行平面共享地平线」，并用齐次叉积从两对平行线算出地平线
- ☐ 能写出四点交比定义，说明 24 种排列为何只有 6 个值、一点在无穷远时的退化形式
- ☐ 能拿图 42.14 的具体坐标复现叉积链 $\mathbf{b} = (((\mathbf{g}_d \times \mathbf{g}_b) \times \mathbf{h}) \times \mathbf{t}_d) \times (\mathbf{g}_b \times \mathbf{t}_b)$，并用交比解出 $h_{\text{desk}}\approx 73.1$ cm
- ☐ 能解释高度如何从桌面传播到桌上瓶子，写出 $\mathbf{c}$ 的叉积链与 $h_{\text{bottle}}$ 公式
- ☐ 能推导三个正交消失点的内参约束 (42.7)–(42.9)，说明 $\mathbf{W}$ 的四参数结构与 SVD + Cholesky 求解流程
- ☐ 能说明标定坐标轴后如何定位三维点（$\mathbf{G}\to\mathbf{v}\to\boldsymbol{l}_4\to P_y$，$\mathbf{G}$ 与 $\mathbf{v}_1,\mathbf{v}_2$ 连线得 $P_x,P_z$）

---

原文：[Single View Metrology](https://visionbook.mit.edu/3d_scene_understanding_single_view.html)
