---
title: 第 21 章 · 下采样与上采样
chapter: 21
original: https://visionbook.mit.edu/upsamplig_downsampling_2.html
---

# 第 21 章 · 下采样与上采样

<div class="src-note">
本页为《Foundations of Computer Vision》第 21 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/upsamplig_downsampling_2.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

Part VI「采样与多尺度表示」的第二章，紧接第 20 章的采样理论进入工程实操：改变图像/特征图分辨率的两类基本操作——下采样（抗混叠滤波 + 抽取）与上采样（扩张 + 插值）。本章的抗混叠滤波器与插值核是第 23 章图像金字塔、第 24 章 CNN 中混叠问题的直接工具，几何章节（第 38 章）也会用到。

## 本篇要点

- 抽取（decimation）＝ 直接隔样本取值，必然引入混叠；正确的下采样 = 先抗混叠低通滤波，再抽取。
- 频域看抽取：$\mathscr{L}_{\downarrow k}=\frac{1}{k^2}\sum_{s,r}\mathscr{L}[u-sN/k, v-rM/k]$，即 $k\times k$ 个平移拷贝叠加——拷贝交叠区即混叠源。
- 矩阵视角：抽取矩阵只取傅里叶矩阵的奇数行，导致 $\mathscr{L}_{\downarrow 2}[u]=\frac{1}{2}(\mathscr{L}[u]+\mathscr{L}[u+N/2])$，高低频被相加混叠。
- 理想抗混叠滤波器是离散 sinc（截断盒的逆变换），但工程上二项式滤波器 $[1,2,1]/4$（及 $b_4$）以极小代价抑制混叠且不引入振铃；窗函数 sinc / Hamming 是折中。
- 上采样 = 扩张（插零）+ 插值卷积；$k=2$ 时双线性核就是二项式 $[1/2,1,1/2]$。
- $k=2^n$ 的下采样用 $n$ 级「二项式滤波 + 2 倍抽取」实现，即金字塔式重采样；非整数倍缩放需要插值而非抽取。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 6 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/upsamplig_downsampling/toy_example_book2.webp" alt="图 21.1 基于混叠的对抗攻击" loading="lazy" />
<figcaption>图 21.1：计算水平图像梯度的玩具系统所受的解析混叠攻击。输入图像 ℓ_in 与 ℓ_out=max(0,ℓ_in∘_2[1,-1]) 的输出图像，其中 ∘_2 表示步长为 2 的卷积。由于这个简单系统存在混叠，攻击者可以注入高频模式 z 来改变输出。被攻击图像 ℓ_in+z 及攻击后的输出——它甚至看起来不像一个导数！</figcaption>
</figure>
<p class="fig-note">💡 实战提示：现代 CNN 里 strided conv / pooling / resize 都是同一类操作；对鲁棒性或安全性敏感的系统，「先低通再抽取」（如 blurpool）能直接封掉这类攻击面。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/upsamplig_downsampling/components_aliasing_FT2.webp" alt="图 21.5 四拷贝叠加 = 混叠" loading="lazy" />
<figcaption>图 21.5：图像乘以 delta 串的 DFT，以及分解为原输入图像 DFT 的四个不同平移拷贝。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：把「抽取 = 四个频谱拷贝叠加」画了出来，绿色框内的重叠区就是混叠的来源，是下一节抗混叠滤波器的直接依据。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/upsamplig_downsampling/decimation_finalequations_inmatrixform2.webp" alt="图 21.8 矩阵视角的混叠" loading="lazy" />
<figcaption>图 21.8：抽取与混叠的可视化。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：用矩阵块结构把「下采样后的每个频率 = 原信号两个频率之和」可视化，混叠从此不再是抽象概念。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/upsamplig_downsampling/ringing_artifacts.webp" alt="图 21.15 三种低通滤波器的振铃" loading="lazy" />
<figcaption>图 21.15：三种不同低通抗混叠滤波器的输出。（左）理想低通滤波器；（中）Hamming 窗；（右）二项式滤波器。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：直接回答「工程上到底该用哪个抗混叠滤波器」——理想 sinc 通带最干净却振铃最重，二项式以微小混叠换取无伪结构。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/upsamplig_downsampling/subsampled_antialiasing_textures.webp" alt="图 21.17 抗混叠 vs 直接抽取" loading="lazy" />
<figcaption>图 21.17：因子 2 逐级抗混叠下采样。与图 21.2 的结果对比。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：与图 21.2 并读就是本章核心结论——同样的抽取率，「先滤波」与「直接抽」得到的信息量天差地别。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/upsamplig_downsampling/upsampling_bilinear.webp" alt="图 21.21 上采样全链路" loading="lazy" />
<figcaption>图 21.21：用双线性插值滤波器按 k=2 上采样。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：一张图看全「扩张 → 卷积插值」链路及其频域后果（频谱拷贝未被完全抑制），是理解第 23 章金字塔与 GAN 上采样伪影的起点。</p>

## 本章小结

- 直接抽取（decimation）= 隔样本取值 $\ell[kn,km]$，线性且可写成矩阵，但会引入混叠假结构。
- 下采样的正确定义 = 抗混叠滤波 + 抽取；抗混叠滤波把 $u,v>N/2k$ 的频率内容置零/衰减。
- 频域本质：$\mathscr{L}_{\downarrow k}$ 是 $\mathscr{L}$ 的 $k\times k$ 个平移拷贝以 $1/k^2$ 权重叠加；矩阵视角下 $\mathscr{L}_{\downarrow 2}[u]=\frac{1}{2}(\mathscr{L}[u]+\mathscr{L}[u+N/2])$。
- 理想抗混叠 = 离散 sinc（截断盒的逆变换）；实际用窗 sinc（矩形/Hamming）或二项式 $b_2/b_4$；二项式性价比最高，无振铃。
- 上采样 = 扩张（插零）+ 插值卷积；$k=2$ 双线性 = 二项式 $[1/2,1,1/2]$；质量优先用 bicubic/Lanczos。
- $k=2^n$ 多级「二项式 + 抽取」= 金字塔式重采样；非整数倍缩放需要插值而非抽取。

## 复习自检清单

- ☐ 能写出抽取的定义 $\ell_{\downarrow k}[n,m]=\ell[kn,km]$ 及其长度 8、$k=2$ 的矩阵 $\mathbf{D}_2$
- ☐ 能推导式 (21.3)：抽取图像的 DFT 是原 DFT 的 $k\times k$ 个平移拷贝的 $1/k^2$ 叠加，并说明哪些频率内容导致混叠
- ☐ 能用矩阵性质 $\mathbf{D}_2\mathbf{F}_N=[\mathbf{F}_{N/2}\ |\ \mathbf{F}_{N/2}]$ 推出 $\mathscr{L}_{\downarrow 2}[0]=0.5\mathscr{L}[0]+0.5\mathscr{L}[8]$
- ☐ 能写出理想抗混叠滤波器的截止频率 $L=N/(2k)$ 及其离散 sinc 脉冲响应，并解释矩形窗/Hamming 窗/二项式三种方案的取舍
- ☐ 能解释为什么理想低通抗混叠会引入振铃，而二项式滤波器不会
- ☐ 能写出「下采样 = 抗混叠 + 抽取」「上采样 = 扩张 + 插值」两步流程及 $k=2$ 的二项式核
- ☐ 能区分最近邻/双线性/bicubic/Lanczos 插值的核形状与适用场景
- ☐ 能解释 21.2 节对抗攻击为何能成功，以及哪些工程改动（先滤波再抽取）可以缓解

---

原文：[Downsampling and Upsampling Images](https://visionbook.mit.edu/upsamplig_downsampling_2.html)
