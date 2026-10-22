---
title: 第 38 章 · 图像与几何的表示
chapter: 38
original: https://visionbook.mit.edu/homogeneous_coordinates.html
---

# 第 38 章 · 图像与几何的表示

<div class="src-note">
本页为《Foundations of Computer Vision》第 38 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/homogeneous_coordinates.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

Part XI「理解几何」的开篇。此前各章几乎都把图像当作规则网格上的像素值阵列；本章引入第二种世界观——把图像表示为「带显式坐标与颜色的点集」，并给出配套数学工具齐次坐标（homogeneous coordinates），让平移、旋转、缩放、切变统一写成矩阵乘法。这套语言是后续相机模型与投影矩阵（第 39 章）、对极几何（第 40 章）乃至单应矩阵（第 41 章）的地基。

## 本篇要点

- 图像有三种等价表示：有序阵列（像素值矩阵）、点集（颜色 + 位置显式编码，式 38.1）、连续函数（隐式表示 $\ell(x,y)=f_\theta(x,y)$）；做几何操作用点集最自然，但最终都要能变回有序阵列。
- 齐次坐标 = 非齐次坐标补一个 1；整体非零缩放视为同一点，「除以最后一维」回到欧氏坐标——这正是透视投影的核心动作。
- 2D 平移、缩放、旋转、切变在齐次坐标下都是 $3\times3$ 矩阵与点的乘积，可任意串联；旋转 + 平移 = 欧氏变换（保长度与角度），再加均匀缩放 = 相似变换（保角度），四者全上 = 仿射变换（保平行）。
- 直线与平面也有齐次表示：两点叉积得过两点的直线，两线叉积得交点；三点共线 ⟺ 其齐次坐标列向量行列式为 0。
- 图像变形必须用反向映射（backward mapping）：遍历输出像素、施加 $\mathbf{M}^{-1}$ 再插值；正向映射会产生空洞与混叠。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 4 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/imaging_geometry/homogeneousAndHeteregeneous_VS3.webp" alt="图 38.1 齐次坐标与非齐次坐标的对应关系" loading="lazy" />
<figcaption>图 38.1：2D 中非齐次坐标与齐次坐标之间的变换规则。虚直线上的所有点对应非齐次坐标中的同一个点。注意 (x,y) 与 (x,y,1) 生活在不同的空间中。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：一张图建立「齐次坐标 = 过原点的直线」这一核心直觉，后面所有 3×3 矩阵（仿射、单应、投影矩阵）都活在这条直觉上。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/imaging_geometry/2d_transforms.webp" alt="图 38.7 2D 变换汇总表" loading="lazy" />
<figcaption>图 38.7：2D 几何变换及其齐次坐标表述汇总。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是全章的「一页速查表」——四种基本变换的矩阵形式、串联规则与不变量都浓缩在这张图里，复习时先回忆它。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/imaging_geometry/points_and_lines_homogeneous.webp" alt="图 38.9 点线互算" loading="lazy" />
<figcaption>图 38.9：用齐次坐标求过两点的直线、求两直线的交点。两种操作完全对偶。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：点线对偶是射影几何的基本功，后面单应矩阵估计、对极几何、消失线等推导都会反复用到「叉积即求线/求交」。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/imaging_geometry/warping_sketch.webp" alt="图 38.10 正向 vs 反向映射（示意）" loading="lazy" />
<figcaption>图 38.10：最近邻插值下正向映射与反向映射的对比。正向映射会产生缺失值。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：「先变换源点」还是「先遍历目标点」决定了变形算法有没有空洞——这是所有 warp 实现的根本分叉口，必须一眼认出。</p>

## 本章小结

- 图像表示三选一（或混用）：有序阵列（利于卷积与显示）、点集（几何显式，利于变换）、连续函数 $f_\theta$（先验显式，可连续查询）。
- 齐次坐标 = 末位补 1 + 整体缩放等价；「除以 $w$」回到欧氏坐标，这正是透视投影的动作。
- 2D 四种基本变换在齐次坐标下统一为 $3\times3$ 矩阵乘法，可串联；仿射 6 自由度保平行，放开第三行得 8 自由度的射影变换。
- 点/线对偶：叉积求线、求交；行列式判共线与共点。
- 变形算法用反向映射 + 高质量插值；缩小先模糊防混叠。
- 隐式表示（SIREN 等）把插值/变形变成函数求值，衍生出图像对齐与空间变换器网络。

## 复习自检清单

- ☐ 能写出三种图像表示，并各自说出一个「只有它才好做」的操作
- ☐ 能写出 2D/3D 齐次坐标变换规则，解释为什么齐次坐标中点不能相加
- ☐ 能默写平移、缩放、旋转、切变四个 $3\times3$ 矩阵，并说明各自保住什么不变量（角度/平行/长度/面积）
- ☐ 能推导「绕任意点旋转」的三矩阵串联，并解释矩阵乘法不可交换的后果
- ☐ 能区分欧氏/相似/仿射三组变换的不变量，说出各是多少自由度
- ☐ 能用叉积从两点求直线、从两线求交点，并用行列式判三点共线
- ☐ 能解释正向映射为什么有空洞，以及反向映射为何没有
- ☐ 能写出 $\hat{\ell}(x,y)=f_\theta(\mathbf{M}^{-1}(x,y,1)^\mathsf{T})$，说明 $f_\theta$ 内部为何要除以第三分量

---

原文：[Representing Images and Geometry](https://visionbook.mit.edu/homogeneous_coordinates.html)
