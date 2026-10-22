---
title: 第 16 章 · 傅里叶分析
chapter: 16
original: https://visionbook.mit.edu/image_processing_fourier.html
---

# 第 16 章 · 傅里叶分析

<div class="src-note">
本页为《Foundations of Computer Vision》第 16 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/image_processing_fourier.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

全书数学密度最高的一章，是 Part III「图像处理」的数学地基。前面章节用「锐利/模糊」这类口语描述线性滤波器的效果，本章引入傅里叶变换给出精确语言：把图像表示为不同**空间频率**（spatial frequency）复指数的叠加。卷积定理、频谱的可视化解读、幅度/相位的分工，是后续所有线性滤波器章节（以及理解神经网络学到的滤波器）的根基；傅里叶基也是理解 transformer 位置编码（positional encoding）等现代表示的工具。

## 本篇要点

- 傅里叶变换把图像从「像素值之和」改写为「不同频率复指数之和」；空间频率描述信号在空间中变化的快慢。
- 复指数构成有限长离散信号的正交基，分解唯一；DFT 是线性可逆变换，FFT（Cooley–Tukey）把计算量从 $O(N^2)$ 降到 $O(N\log N)$。
- **卷积定理**：循环卷积的傅里叶变换等于各自变换的乘积——线性滤波器只重新加权已有频谱，不创造新频率成分。
- 频谱分解为**幅度** $A[u,v]$ 与**相位** $\theta[u,v]$：位置信息主要进相位，强度缩放主要进幅度；自然图像的幅度谱约按 $1/f$ 衰减。
- Box 函数的 DFT 是离散 sinc 函数；空间平移 = 频域乘复指数（调制），乘复指数 = 频域平移，二者互为对偶。
- 滤波器按通过频段分为低通/带通/高通；人眼视觉系统近似为一个带通系统（对比度敏感函数 CSF，峰值约 6 周期/度）。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 9 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/Image_processing_fourier/visualization_DFT_color.webp" alt="图 16.6 DFT 矩阵的可视化" loading="lazy" />
<figcaption>图 16.6：把离散傅里叶变换可视化为一个矩阵。被变换的信号构成右侧列向量的元素。傅里叶矩阵的复值用颜色表示，图例在左下。右侧向量中黑色值表示零。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：DFT 的全部性质（对称性、共轭对称、正交性）都压缩在这张彩色矩阵图里；建立「变换 = 换基」的几何直觉，就从读这张图开始。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/Image_processing_fourier/dft_a.webp" alt="图 16.7 图像的 DFT 与两种分解" loading="lazy" />
<figcaption>图 16.7：一幅图像的 DFT，以及（上）实部与虚部、（下）幅度与相位的可视化。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是「频谱图」的标准读法模板——实部/虚部看起来像噪声，而幅度/相位图才显现出边缘朝向、周期结构等可解释信息。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/Image_processing_fourier/dft_b.webp" alt="图 16.8 用大幅值系数重建图像" loading="lazy" />
<figcaption>图 16.8：从幅度最大的 N 个傅里叶系数重建图像。右框显示这 N 个傅里叶系数在频域中的位置，对它们求逆即得到左图。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这张图是「频谱 = 图像内容」的最直观证据，也解释了为什么少量低频系数就能做有损压缩——大幅值系数集中在低频。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/Image_processing_fourier/DFT_toy_examples.webp" alt="图 16.10 2D 傅里叶变换对典例" loading="lazy" />
<figcaption>图 16.10：若干二维傅里叶变换对。注意这组变换对中可见的趋势：图像在一个域中的支撑越大，另一个域中的幅度越局部化。一条直线变换为一条与之垂直的直线。图像为 64 × 64 像素，原点在中心。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这组「玩具变换对」是读频谱图的速查表：点/线/边缘/周期图案在频域长什么样、朝哪躺，全在这里。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/Image_processing_fourier/amp_phase_swap_color.webp" alt="图 16.14 交换幅度与相位" loading="lazy" />
<figcaption>图 16.14：交换两幅图像傅里叶变换的幅度与相位。每个颜色通道以相同方式处理。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是理解「相位承载结构」的决定性实验——用停车标志的相位配别人的幅度，重建图看起来仍是停车标志。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/Image_processing_fourier/dft_random_phaseandamplitude.webp" alt="图 16.15 相位与幅度的相对重要性" loading="lazy" />
<figcaption>图 16.15：相位与幅度的相对重要性取决于图像。每行展示一幅图像、它的傅里叶变换（幅度与相位），以及对「保留原幅度 + 随机相位」和「保留原相位 + 通用 1/f 幅度」两种信号做逆傅里叶变换得到的结果。注意对第一幅图像相位似乎是最重要的分量；但往下走，两个分量之间的相对重要性会变化；对最下方展示准周期线纹理的图像，幅度是最重要的分量。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：纠正「相位至上」的过度概括——对强周期/纹理图案（织物、栅格、摩尔纹），幅度谱才是辨识性特征，这是许多频域描述子的依据。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/Image_processing_fourier/filter-types.webp" alt="图 16.17 低通/带通/高通滤波器" loading="lazy" />
<figcaption>图 16.17：低通、带通、高通滤波器频率响应草图。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这张频率响应草图是后续所有滤波器章节的「坐标系」——看到任何滤波器，先问它的 |H[u,v]| 长什么样、放行哪一段频谱。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/Image_processing_fourier/mit_columns.webp" alt="图 16.19 MIT 大楼例子的频域滤波" loading="lazy" />
<figcaption>图 16.19：傅里叶域中的简单滤波。(a) MIT 穹顶大楼重复的柱子沿傅里叶域中的一条水平线产生谐波（见 (b)）。(d) 中把这些傅里叶分量置零后，大楼的柱子被大幅去除 (c)。也可以做互补操作、只保留这些谐波（见 (f)），结果是只保留柱子 (e)。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是「周期性结构 → 频谱冲激 → 陷波滤波」的完整闭环案例；从此看到任何重复噪声（栅格、扫描纹、纱窗），第一反应应该是去频谱里找冲激。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/spatial_filters/csf.webp" alt="图 16.20 对比度敏感函数" loading="lazy" />
<figcaption>图 16.20：Campbell–Robson 图展示的对比度敏感函数。图像展示的正弦波频率从左到右递增、幅度从上到下递增。你能沿着图描出一条曲线，指出正弦波在哪里变得不可见吗？</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：CSF 是「人眼 ≈ 带通滤波器」的直接实验证据，解释了为什么中频段信息（纹理、自然边缘）对感知最关键，也是图像压缩/质量评估常利用的特性。</p>

## 本章小结

- 傅里叶变换把图像改写为不同空间频率的复指数之和；DFT 是有限长离散信号上的线性、可逆、唯一的分解，FFT 使其计算高效。
- 复指数是 DFT 的特征函数：线性空间不变系统对复指数输入只改幅度与相位——这就是用传递函数 $H[u,v]$ 分析一切线性滤波器的原因。
- 卷积定理：$\ell_1 \circ \ell_2 \xrightarrow{\mathscr{F}} \mathscr{L}_1\mathscr{L}_2$；线性滤波只重新加权频谱，不创造新频率。对偶地，乘积对偶卷积。
- 空间平移 = 频域乘复指数（调制），调制 = 频域平移；Parseval/Plancherel 给出能量守恒（差一个 $1/NM$）。
- Box 函数的 DFT 是离散 sinc 函数；冲激的 DFT 是常数；余弦/正弦的 DFT 是共轭位置上的一对冲激。
- 幅度承载强度/周期性纹理信息，相位承载位置/结构信息；自然图像幅度谱约按 $1/f$ 衰减，可作去噪等任务的先验。
- 滤波器按频谱响应分为低通/带通/高通；人眼近似带通系统（CSF 峰值约 6 周期/度）。
- 最大警示：傅里叶表示没有局部性——「是什么」与「在哪里」不可兼得，这正是下一章走向局部化表示的动机。

## 复习自检清单

- ☐ 能写出 DFT 与逆 DFT 的定义式 (16.2)(16.3)，并用复指数正交性说明为什么分解唯一、逆变换为何是共轭基
- ☐ 能解释 $\mathbf{F}^{-1}$ 与 $\mathbf{F}^{*}$ 的关系（差多少尺度因子），并从图 16.6 的矩阵中读出共轭对称性
- ☐ 能独立用几何级数求和推导 Box 函数的 DFT 为离散 sinc 函数 (16.5)(16.6)
- ☐ 能写出并推导卷积定理，解释「线性滤波器只重新加权频谱、不创造新频率」，以及为什么自相关不享受此性质
- ☐ 能解释空间平移与调制为何互为对偶，并说明相机运动造成的非循环平移为何只让式 (16.7) 近似成立
- ☐ 能说明幅度与相位分别承载什么信息，并用图 16.14（相位主导外观）与图 16.15（纹理图幅度主导）的现象分别佐证
- ☐ 能解释 MIT 大楼例子中峰值为何出现在 $256/14 \approx 18.2$ 及其谐波处，以及条带宽度、垂直衰减各是什么造成的
- ☐ 能描述 CSF 的形状与峰值位置（约 6 周期/度），并解释 Campbell–Robson 图为什么「看起来不可分离」、与光感受器取对数的关系

---

原文：[Fourier Analysis](https://visionbook.mit.edu/image_processing_fourier.html)
