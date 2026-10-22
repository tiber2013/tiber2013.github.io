---
title: 第 22 章 · 滤波器组
chapter: 22
original: https://visionbook.mit.edu/spatial_filter_sets.html
---

# 第 22 章 · 滤波器组

<div class="src-note">
本页为《Foundations of Computer Vision》第 22 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/spatial_filter_sets.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

本书 Part VI「多尺度图像表示」的第三章（前有第 20 章采样与混叠、第 21 章下采样与上采样）。第 15–19 章建立了线性滤波、傅里叶分析、模糊滤波器、图像导数与时域滤波器；本章回答的问题是：如何把一组线性滤波器与简单非线性操作组合起来，构造对图像结构（方向、频率、运动）有选择性的表示。这些手工设计的滤波器组既是早期低层视觉建模的核心，也被视为现代深度网络表征能力的"前身"——它紧接第 23 章图像金字塔，并为第 24 章起的卷积网络埋下伏笔。

## 本篇要点

- Gabor 滤波器 = 高斯窗 × 正弦/复数波，是同时实现空间与频率最佳局部化的定向带通滤波器；余弦相位与正弦相成对构成**正交对**（quadrature pair）。
- 平方和 $\ell_h^2+\ell_q^2$ 给出**局部振幅**：低通、符号不变、对比度不变，是检测边缘/轮廓的经典算子。
- **可转向滤波器**（steerable filter）：少量基滤波器经角度插值函数线性组合，即可合成任意方向的响应；一阶高斯导数需 2 个基、二阶需 3 个基、二阶导数的希尔伯特近似需 4 个基。
- 空时 Gabor 对速度不具选择性（单个滤波器无法分辨速度），需一组滤波器配合局部能量才能构成速度调谐（velocity-tuned）单元。
- "基滤波器 → 平方非线性 → 求和"就是一个小型两层卷积网；手工滤波器组基于第一原理、无需训练数据，但性能上限有限。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 6 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/spatial_filter_sets/gabors.webp" alt="图 22.2 二维 Gabor 函数的三种形态" loading="lazy" />
<figcaption>图 22.2：二维 Gabor 函数。(a) 起局部化作用的高斯窗（σ=1），可视为零频率正弦对应的 Gabor 函数；(b) 余弦相位与 (c) 正弦相位 Gabor 函数，中心频率 u_0=2π、v_0=0。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：余弦/正弦相位对是本章一切后续内容（正交对、局部能量、可转向金字塔）的原子构件，务必建立"高斯包络 + 条纹"的直观图像。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/spatial_filter_sets/gabor_FT.webp" alt="图 22.3 Gabor 函数的频谱：单瓣定向带通" loading="lazy" />
<figcaption>图 22.3：(a) 余弦与 (b) 正弦 Gabor 函数；(c) 余弦与正弦 Gabor 函数傅里叶变换的幅度（二者 FT 仅相位不同）；(d) 复数 Gabor 函数的 FT，呈非对称的单瓣形状。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：单瓣、平移至高中心的频谱形状，解释了 Gabor 为什么"定向 + 选频"；这是把滤波器组铺满频域的依据。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/spatial_filter_sets/quad2.webp" alt="图 22.6 局部振幅的计算流程" loading="lazy" />
<figcaption>图 22.6：局部振幅的计算。输入经一对正交 Gabor 滤波器滤波，每个滤波器输出平方后将结果相加。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要："卷积 → 平方 → 求和"三件套是本章最核心的架构 motif，从边缘检测到现代网络中的能量层都能看到它。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/spatial_filter_sets/steer_architecture.webp" alt="图 22.10 可转向滤波器架构" loading="lazy" />
<figcaption>图 22.10：可转向滤波器结构。该结构计算沿方向 θ 的图像二阶导数。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要："少量基滤波器 + 逐像素增益图"意味着不必为每个角度各做一次卷积——方向调节退化为廉价的线性组合，这是可转向性真正的工程价值。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/temporal_filters/gabor_spacetime_FT.webp" alt="图 22.15 空时 Gabor 及其频谱示意" loading="lazy" />
<figcaption>图 22.15：(a) 空时 Gabor 滤波器：余弦与正弦 x-t Gabor 滤波器；(b) 其传递函数示意；(c) 两个空间维度（x-y-t）空时 Gabor 滤波器传递函数示意。两个平面示例了以相同方式与 Gabor 滤波器相交的空时平面。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这张图把"为什么单个定向滤波器测不了速度"讲透了，是理解孔径问题与后续所有运动算法的几何起点。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/temporal_filters/MT_velocity_tuned.webp" alt="图 22.17 速度选择单元架构" loading="lazy" />
<figcaption>图 22.17：构造速度选择单元的结构。第一层中，余弦与正弦滤波器组合成相位不变的频率调谐输出；第二层中，空时 Gabor 滤波器的输出按傅里叶域中不同的平面分组，形成速度选择输出。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是"频率能量 + 跨滤波器组合"的完整样板，与哺乳动物 MT 区的速度选择神经元、以及经典运动能量模型同构，值得逐层拆解。</p>

## 本章小结

- 滤波器组 = 一组各司其职的滤波器；配上平方、求和这类简单非线性，就能构造局部能量、局部相位等强大表示，也是现代深度网络表征能力的"手工前身"。
- Gabor 滤波器是高斯窗调制正弦/复数波，空间-频率联合局部化最优，呈定向带通；余弦/正弦相位对构成正交对。
- 正交对响应平方和 = 局部振幅：低通、符号/对比度不变，适合边缘轮廓检测；局部相位则只对结构敏感，可用于运动估计。
- 可转向滤波器用少量基滤波器 + 角度插值函数合成任意方向；转向定理给出所需基数量（$N$ 阶偶/奇对称多项式需 $N+1$ 个），采样后依然成立。
- 空时 Gabor 对速度不选择性（孔径问题）；速度选择单元靠"跨方向求能量 + 沿平面组合"实现。
- 手工滤波器组基于第一原理、免训练，曾支撑纹理分析、分割、运动、定向分析、去噪等任务，但性能有上限，是通往学习式架构的跳板。下一章进入多尺度图像金字塔。

## 复习自检清单

- ☐ 能写出 Gabor 函数定义（式 22.1）及其实部/虚部，解释 $\sigma$ 与 $(u_0,v_0)$ 各自控制什么
- ☐ 能说明为什么"高斯窗 × 正弦"同时优化了空间与频率局部化，其频谱为何是平移的单瓣高斯
- ☐ 能写出希尔伯特变换对与解析信号的定义（式 22.3、22.4），解释瞬时振幅/相位的含义
- ☐ 能推导冲激输入的局部振幅 = 高斯包络，并说明符号不变性为什么对边缘检测有用
- ☐ 能写出一阶、二阶高斯导数的转向方程，说出各需几个基滤波器及插值函数
- ☐ 能陈述转向条件（式 22.5–22.7）与"$N$ 阶多项式需 $N+1$ 个基"的结论
- ☐ 能解释为什么单个空时 Gabor 测不了速度，以及图 22.17 架构如何获得速度选择性
- ☐ 能说明方向能量极坐标图中虚假瓣的成因（交叉项）与消除方法

---

原文：[Filter Banks](https://visionbook.mit.edu/spatial_filter_sets.html)
