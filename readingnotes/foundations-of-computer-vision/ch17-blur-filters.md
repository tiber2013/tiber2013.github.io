---
title: 第 17 章 · 模糊滤波器
chapter: 17
original: https://visionbook.mit.edu/blurring_2.html
---

# 第 17 章 · 模糊滤波器

<div class="src-note">
本页为《Foundations of Computer Vision》第 17 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/blurring_2.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

本篇「线性图像滤波」的核心应用章。前面的章节建立了卷积与傅里叶分析的工具，本章用它们系统地研究三族最常用的低通（模糊）滤波器——盒式、高斯与二项式——及其频谱特性与局限。模糊不仅是降噪与多尺度分析的基础操作，也是下采样抗混叠（第 21 章）与图像金字塔（第 23 章）的前置步骤。

## 本篇要点

- 模糊滤波器是低通滤波器：滤除高频空间成分、保留低频；实现上就是对输入像素的小邻域做局部平均（卷积）。非线性模糊（各向异性扩散、双边滤波）可在去噪同时保留轮廓。
- 盒式滤波器最简单、可分离、大核可用积分图加速；但它的衰减非单调（最高频的 $[1,-1,1,-1,\dots]$ 原样通过），且两个盒式卷积得到的是三角形而不是更大的盒式。
- 高斯的傅里叶变换仍是高斯；两个高斯卷积仍是高斯且方差相加 $\sigma_3^2=\sigma_1^2+\sigma_2^2$——这是高斯金字塔的基石。但这些性质只在连续域严格成立，直接采样得到的离散高斯不再满足。
- 二项式滤波器是高斯的高效整数近似：DC 增益 $2^n$、方差 $n/4$、卷积封闭（$b_n\circ b_m=b_{n+m}$）、严格零化最高频——下采样前抗混叠的理想选择。
- 必须记住的核：$[1,2,1]/4$ 及其二维扩展 $\begin{bmatrix}1&2&1\\2&4&2\\1&2&1\end{bmatrix}/16$。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 4 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/blur_filters/boxfilter_fourier.webp" alt="图 17.3(b) 盒式滤波器的傅里叶变换" loading="lazy" />
<figcaption>图 17.3 (b)：盒式滤波器的傅里叶变换。注意其旁瓣（ripple）与零点位置——高频振荡信号不会被完全抑制。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这张 sinc 型频谱一眼说明盒式滤波的两大毛病——旁瓣振铃、最高频处有零点（不衰减），是理解「为什么不用盒式而用高斯」的核心证据。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/spatial_filters/gausian_zebra_c_2.webp" alt="图 17.4(a) σ=2" loading="lazy" />
<figcaption>图 17.4 (a)：σ=2 的高斯滤波。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：σ=2/4/8 三幅连看，直观建立「σ 控制模糊尺度」的手感，这是选高斯核参数时的肌肉记忆。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/blur_filters/gauss3filter_fourier.webp" alt="图 17.7(b) 二项式滤波器的傅里叶变换" loading="lazy" />
<figcaption>图 17.7 (b)：二项式滤波器的傅里叶变换——单调下降、无涟漪，在最高频处严格为零。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：与图 17.3(b) 的盒式频谱对照看——二项式核频谱在最高频处为零、无旁瓣，是「二项式逼近高斯」的频域证据。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/blur_filters/boat_b_noise.webp" alt="图 17.8(a) 含棋盘格噪声的图像" loading="lazy" />
<figcaption>图 17.8 (a)：含高频棋盘格噪声的图像。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：(a)(b)(c) 三联图直接对比盒式与二项式对棋盘格高频噪声的抑制效果，是「离散高斯 = 重复二项式卷积」这一结论的视觉证据。</p>

## 本章小结

- 高斯和二项式滤波器在计算机视觉中广泛使用。特别是二项式滤波器 $[1,2,1]/4$（此处归一化到 DC 增益为 1）及其 2D 扩展，是非常有用的核，可在许多场景使用：去除高频噪声、对图像做 2 倍下采样等。构建图像金字塔时、或在神经网络中执行不同的池化操作或缩放特征图时，模糊核也很有用。
- 把 2D 二项式滤波器放在手边：

$$\begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix} / 16$$

## 复习自检清单

- ☐ 能写出盒式核 $\text{box}_{N,M}$ 的定义及其卷积公式，说明为什么 DC 增益等于核系数之和、为什么通常要归一化到和为 1
- ☐ 能解释盒式滤波器的两个局限：对 $[1,-1,1,-1,\dots]$ 无衰减、两次卷积得三角形而非更大的盒式，以及为什么首选奇数尺寸
- ☐ 能写出连续高斯的 1D/2D 定义（式 17.1、17.2）与傅里叶变换（式 17.5），说明 $\sigma$ 增大时空域变宽、频域变窄
- ☐ 能说明「两个高斯卷积仍是高斯且方差相加」（$\sigma_3^2=\sigma_1^2+\sigma_2^2$）及其在高斯金字塔中的意义
- ☐ 能解释为什么离散高斯不再满足「卷积封闭」与「零化最高频」（$g_5$ 的例子），以及这在什么场景下要紧
- ☐ 能写出二项式核 $b_n$ 的构造（帕斯卡三角）、DC 增益 $2^n$、方差 $n/4$，以及 $b_n\circ b_m=b_{n+m}$
- ☐ 能写出 $[1,2,1]$ 的 DFT $B_2[u]=2+2\cos(2\pi u/N)$，并解释为什么它单调无涟漪、在最高频处为零
- ☐ 能写出 2D 二项式核 $b_{2,2}$ 及其归一化方式，解释为什么它能完全滤除棋盘格噪声而 $3\times 3$ 盒式不能

---

原文：[Blur Filters](https://visionbook.mit.edu/blurring_2.html)
