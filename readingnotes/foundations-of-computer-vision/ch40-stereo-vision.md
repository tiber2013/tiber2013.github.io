---
title: 第 40 章 · 立体视觉
chapter: 40
original: https://visionbook.mit.edu/3d_scene_understanding_stereo.html
---

# 第 40 章 · 立体视觉

<div class="src-note">
本页为《Foundations of Computer Vision》第 40 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/3d_scene_understanding_stereo.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

本章研究如何从空间上分离的两个相机所拍的图像对中计算深度。深度计算通常分两部分：(1) 分析相机投影的几何——已知图像偏移量后即可三角化出深度；(2) 计算两幅图像中物体对应部分之间的偏移量。我们先讲几何（一个图像点的候选匹配在另一幅图中能出现在哪里），并介绍把候选匹配放到同一像素行的图像校正（rectification）操作；然后在校正图像对的假设下讨论第二项任务——计算匹配点之间的偏移（立体匹配）；最后介绍基于学习的方法。

## 本篇要点

- 平行相机配置下，深度估计归结为测量视差（disparity）：$Z=fT/(x_L-x_R)$，视差与深度成反比。
- 对极几何（epipolar geometry）把对应点搜索从二维降为一维：一个图像点的候选匹配必在另一幅图像的一条对极线（epipolar line）上。
- 本质矩阵 $\mathbf{E}$（相机坐标）与基础矩阵 $\mathbf{F}$（像素坐标）把对极约束写成 $\mathbf{p}_2^\mathsf{T}\mathbf{E}\mathbf{p}_1=0$ 与 $\mathbf{c}_2^\mathsf{T}\mathbf{F}\mathbf{c}_1=0$。
- 图像校正（image rectification）使对极线与图像扫描行重合，匹配变成逐行一维搜索。
- 立体匹配分「匹配 + 滤波」两阶段；基于 3D 代价体 + 3D CNN 的学习方法已全面超过手工特征方法。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 4 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/3d_scene_understanding/random_dot_stereogram.webp" alt="图 40.3 随机点立体图" loading="lazy" />
<figcaption>图 40.3：随机点立体图。两幅图像几乎相同，唯一区别是中间的正方形在两图之间左右移动了几个像素。每只眼睛看一幅图时（可用立体镜或自由融合），你会看到一块方形从平坦背景中浮出来。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：随机点立体图是「视差本身就是独立深度线索」的最干净证据——没有任何形状、遮挡、纹理语义，仅凭左右位移就能产生深度感知；立体匹配算法的本质就是在用计算复现这件事。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/3d_scene_understanding/triangularization_stereo.webp" alt="图 40.5 简单立体配置与三角化几何" loading="lazy" />
<figcaption>图 40.5：简单立体。(a) 两个焦距同为 f、横向相距 T 的相机把三维点 P 成像为各自像面上的二维点 [x_L,y_L]^mathsfT 与 [x_R,y_R]^mathsfT。(b) 两个三角形的相似关系导出式 (40.1)。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是全书视差—深度关系的基准图：Z=fT/(x_L-x_R) 一行公式概括了平行立体相机的全部几何，后面所有立体算法（包括深度网络回归视差）都建立在它之上。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/stereo/stereolamp.webp" alt="图 40.7 基于亮度的立体匹配问题与视差空间图像" loading="lazy" />
<figcaption>图 40.7：基于亮度的立体匹配问题。(a) 立体对 (c) 中一幅图像的一行。(b) 不同视差下的亮度匹配误差（垂直方向为视差、水平方向为像素位置），红色为最佳亮度匹配的视差。(d) 平滑后的匹配误差，红色为最佳匹配，及所得深度图 (e)。(f) 真值匹配与深度 (g)。数据集来自 [4]。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：视差空间图像（DSI）是理解立体匹配不确定性的标准工具——沿视差维看匹配代价曲线，一眼能看出哪些区域纹理缺失、哪些位置被遮挡坑死；它也是后续一切代价体（cost volume）方法的思想源头。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/3d_scene_understanding/epipolar_geometry.webp" alt="图 40.12 立体相机对几何与术语" loading="lazy" />
<figcaption>图 40.12：一对立体相机的几何与术语。图中给出给定点 P 的对极平面、对极线与极点（e_1、e_2）。移动点 P 时，只有极点保持不变。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：对极几何是立体匹配降维的根据——第一幅图像中一个点的匹配只需沿第二幅图像中的一条对极线搜索；这张图把平面/线/点三者的关系一次讲清，务必记住「所有对极线过极点」。</p>

## 本章小结

- 平行相机配置下 $Z=fT/(x_L-x_R)$：深度估计 = 测视差；视差与深度成反比。
- 一般相机位姿下，对极几何把二维搜索降为一维：匹配必在对极线上；本质矩阵 $\mathbf{E}$（5 自由度、秩 2）与基础矩阵 $\mathbf{F}$（7 自由度、秩 2）代数化地表达这一约束。
- 图像校正让对极线与扫描行重合，把立体匹配变成逐行一维问题；校正本身用单应扭曲。
- 立体匹配 = 匹配 + 滤波：Harris/SIFT 等手工特征方法之外，主流是「校正 → 特征提取 → 3D 代价体 → 3D CNN 聚合」的学习范式。
- 视差空间图像暴露了亮度匹配不可靠的根源（无纹理、噪声、高光、遮挡），保边滤波等后处理是经典缓解手段。
- 精确深度仍是开放问题；可靠结果往往依赖多视几何（更多图像）。

## 复习自检清单

- ☐ 能由图 40.5 的相似三角形独立推出 $Z=fT/(x_L-x_R)$，说明视差为何与深度成反比，并比较「回归视差」与「回归深度」两种输出表示的优劣
- ☐ 能写出船距估计的两个公式 $d=h/\tan\alpha$ 与 $d=t\sin\alpha\sin\beta/\sin(\alpha+\beta)$，并说明后者为何是三角化
- ☐ 能定义对极平面、对极线、极点，并解释「为何只需沿对极线搜索匹配」
- ☐ 能从头推导 $\mathbf{p}_2^\mathsf{T}\mathbf{E}\mathbf{p}_1=0$（Longuet-Higgins 路线），并说出 $\mathbf{E}$ 的自由度、秩与奇异值结构
- ☐ 能写出 $\mathbf{F}=\mathbf{K}^{-\mathsf{T}}\mathbf{E}\mathbf{K}^{\mathsf{T}}$，并解释为何 $\mathbf{F}$ 比 $\mathbf{E}$ 多两个自由度
- ☐ 能列出朴素亮度匹配失败的四个原因，并说明视差空间图像如何暴露这些问题
- ☐ 能复述 Harris 角点检测器的推导：从 $E(\Delta x,\Delta y)$ 到 $\mathbf{M}$ 再到 $\lambda_{\min}$ 阈值
- ☐ 能说出图像校正的三个步骤，以及学习式立体「代价体 + 3D CNN」流程的四个阶段

---

原文：[Stereo Vision](https://visionbook.mit.edu/3d_scene_understanding_stereo.html)
