---
title: 第 7 章 · 作为线性系统的相机
chapter: 7
original: https://visionbook.mit.edu/camera_as_linsys.html
---

# 第 7 章 · 作为线性系统的相机

<div class="src-note">
本页为《Foundations of Computer Vision》第 7 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/camera_as_linsys.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

本章把「相机」从针孔/透镜这些特例推广为一般线性成像系统：任何相机的成像过程都可以写成一个矩阵 $\mathbf{A}$ 作用于场景向量 $\boldsymbol\ell_{\texttt{w}}$。有了这个观点，线性代数的全套工具（求逆、正则化、最小二乘）都能用于相机设计与图像重建；它也是后续滤波与反问题章节的先声（本章末尾的「模糊的微分」将在第 18 章展开，光场建模出现在第 45 章），更是计算成像（computational imaging）的总入口。

## 本篇要点

- 相机 = 线性系统：$\boldsymbol\ell_{\texttt{s}}=\mathbf{A}\boldsymbol\ell_{\texttt{w}}$（式 7.1）。针孔相机的 $\mathbf{A}\approx\mathbf{I}$；宽孔针孔相机是「相邻两值相加」的平滑矩阵；边缘相机的 $\mathbf{A}$ 是累加矩阵；针斑相机的 $\mathbf{A}$ 是宽孔矩阵的补。
- $\mathbf{A}$ 往往病态或不可逆，噪声下没有精确解 → 引入正则化项（Tikhonov），解为 $\boldsymbol\ell_{\texttt{w}}=(\mathbf{A}^T\mathbf{A}+\lambda\mathbf{I})^{-1}\mathbf{A}^T\boldsymbol\ell_{\texttt{s}}$（式 7.4）；$\lambda$ 平衡「解释观测」与「满足正则化」。
- 边缘相机的输出是输入的积分（反向累加），其逆是差分（微分），正则化逆是「模糊的微分」——这正是第 18 章滤波内容的先声。
- 针斑相机记录的是遮挡物的影子；角相机把墙角后的场景编码进地面亮度，沿角度求一次微分即可读出 1D 图像（扰动仅约 1/1000）。
- 计算成像范式：先把成像过程建模成矩阵 $\mathbf{A}$，再用线性代数反解场景——记录数据「看起来不像图像」也没关系。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 4 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/imaging/traditional_pinholes_2.webp" alt="图 7.2 针孔相机的成像矩阵" loading="lazy" />
<figcaption>图 7.2：(a) 小孔一维针孔相机示意；(b) 其成像矩阵 A、A⁻¹ 与正则化逆 B 的可视化——对小孔成像器，三者都是单位矩阵。(c) 大孔一维针孔相机；(d) 其成像矩阵的可视化。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这张图是整章的钥匙——它把「相机是什么」变成「矩阵长什么样」：针孔 = 单位矩阵、宽孔 = 平滑矩阵，一眼可比；后面每种新相机都可以这样看图读矩阵。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/imaging/nontraditional_pinholes_2-2.webp" alt="图 7.4 边缘相机与针斑相机" loading="lazy" />
<figcaption>图 7.4：(a) 边缘相机示意；(b) 其成像矩阵。(c) 针斑相机（遮挡物遮挡场景中的两个值）；(d) 其成像矩阵。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这两个相机的矩阵形状完全不同于单位矩阵——累加型与互补型；记住它们的形状，就理解了「非传统相机」如何把场景编码成出人意料的测量。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/imaging/cornercam.webp" alt="图 7.7(a) 角相机几何" loading="lazy" />
<figcaption>图 7.7 (a)：竖直边缘遮挡场景，被遮挡物体在地面环境照明上造成微弱强度变化，可被相机从地面反射光中观测到。</figcaption>
</figure>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/imaging/cornerKey.webp" alt="图 7.7(b) 角相机的重建掩模" loading="lazy" />
<figcaption>图 7.7 (b)：估计特定朝向角（本例约 45°）来光所需的空间掩模：与观测到的地面反射图像相乘，再对所有空间像素求和。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这张图把「绕角成像」的整个反演浓缩成一枚掩模——沿角度对地面亮度求一次微分，就能读出一维场景；它是计算成像「先建模、再反解」范式的最佳范例。</p>

## 本章小结

- 把相机视为一般线性系统 $\boldsymbol\ell_{\texttt{s}}=\mathbf{A}\boldsymbol\ell_{\texttt{w}}$，就能把线性代数的全套机制用于相机设计与处理。
- 针孔相机的 $\mathbf{A}\approx\mathbf{I}$；宽孔针孔是平滑矩阵；边缘相机是累加矩阵（积分），其逆是差分矩阵（微分），正则化逆是模糊的微分。
- 针斑相机的 $\mathbf{A}$ 是宽孔相机的补，测的是遮挡物影子的缺失。
- $\mathbf{A}$ 常病态/不可逆，噪声下无精确解：用 Tikhonov 正则化把估计写成式 (7.4)，$\lambda$ 平衡数据拟合与正则化。
- 角相机把墙后场景编码进地面亮度（约 1/1000 的扰动），沿角度求一次微分即可读出 1D 视频——计算成像「先建模、再反解」的完整范例。
- 场景空间结构更复杂时，需要引入光场（第 45 章）才能保持同样的线性代数框架。

## 复习自检清单

- ☐ 能写出相机的线性系统模型（式 7.1），并说明为什么针孔相机对应单位矩阵
- ☐ 能从式 (7.2) 的正则化目标函数推出式 (7.4)，并解释 $\lambda$ 大了/小了各自会发生什么
- ☐ 能写出边缘相机的 $\mathbf{A}$ 与 $\mathbf{A}^{-1}$（式 7.5），并解释为什么输出像「积分」而逆像「微分」
- ☐ 能解释针斑相机与宽孔针孔相机矩阵的互补关系，以及输出中「影子」的来源
- ☐ 能推导角相机的地面亮度模型（式 7.6–7.8），说明为什么沿角度求一次微分就能读出墙后的一维图像
- ☐ 能解释「正则化逆为什么比直接求逆更抗噪」，并说明这和第 18 章「模糊的微分」有什么联系

---

原文：[Cameras as Linear Systems](https://visionbook.mit.edu/camera_as_linsys.html)
