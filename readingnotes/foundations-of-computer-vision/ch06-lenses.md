---
title: 第 6 章 · 镜头
chapter: 6
original: https://visionbook.mit.edu/lenses.html
---

# 第 6 章 · 镜头

<div class="src-note">
本页为《Foundations of Computer Vision》第 6 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/lenses.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

本章继续 Part II「图像形成」的话题：第 5 章的针孔相机像太暗，本章用透镜解决「亮」与「清晰」不可兼得的问题。在几何光学近似（忽略衍射等波动效应）下，我们从斯涅尔定律解析地推导透镜面形与透镜制造者公式，进而得到共轭点、光线追迹规则、景深等天天要用的定量工具，最后以望远镜示范如何用两条光线规则设计光学仪器。

## 本篇要点

- 针孔相机的进光量与清晰度不可兼得；透镜把来自同一点、射向透镜不同位置的光线汇聚到传感器同一点，兼得两者。
- 近轴近似 + 薄透镜近似下，透镜表面斜率必须随离光轴的距离 $c$ 线性增长（抛物面/球面满足），由此导出透镜制造者公式 $\frac{1}{a}+\frac{1}{b}=\frac{1}{f}$，其中 $f=\frac{R}{2(n-1)}$。
- 凸薄透镜与针孔相机一样成透视投影（过光心光线不偏折）；四条光线追迹规则可手推一切简单光路，放大率为 $a/b$。
- 不在焦平面上的物点成像为弥散圆（circle of confusion）；景深 $D \approx 2NCU^2/f^2$，与光圈系数 $N$ 成正比、与焦距平方成反比——大光圈浅景深。
- 凹透镜焦距为负，把平行光变成看似来自虚焦点的发散光；凸透镜 + 凹透镜组成望远镜，角放大率 $M=f_1/f_2$。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 10 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/imaging/apertures_and_lenses.webp" alt="图 6.1 成像中的亮度/清晰度权衡" loading="lazy" />
<figcaption>图 6.1：成像中亮度与清晰度的权衡。从左到右：小针孔成像清晰，但进光量小，给定曝光时间下像可能很暗；大针孔进光更多、像更亮，但每个传感器元素记录来自许多不同物点的光，像变模糊；透镜可以收集从单点反射到各个方向的光，得到又亮又清晰的像。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这张图是整章乃至相机设计的核心矛盾——进光量与空间分辨率的权衡；理解它才能理解为什么需要透镜，以及为什么会有景深这回事。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/imaging/lensRealcropped.webp" alt="图 6.4(a) 透镜聚焦面形的推导几何" loading="lazy" />
<figcaption>图 6.4 (a)：距透镜 a 的点发出的光，经适当面形的透镜会聚到距透镜 b 的点。</figcaption>
</figure>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/imaging/lensDistortedCropped.webp" alt="图 6.4(b) 角度标注放大图" loading="lazy" />
<figcaption>图 6.4 (b)：图 6.4 (a) 的角度与距离失真放大版，用于标注 θ_1–θ_4、内部光线偏角 δ 与透镜表面角 θ_S。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是推导透镜制造者公式的唯一几何依据，「θ_S 随 c 线性变化」的聚焦条件就出自这张图；看懂了它，式 (6.1)–(6.3) 都顺理成章。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/imaging/lensrays1.webp" alt="图 6.9(a) 平行光会聚到焦点" loading="lazy" />
<figcaption>图 6.9 (a)：来自无穷远的平行光线会聚到透镜后方距离 f 处。</figcaption>
</figure>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/imaging/lensrays2.webp" alt="图 6.9(b) 光源移近，焦点移远" loading="lazy" />
<figcaption>图 6.9 (b)：光源向透镜移近，光线在另一侧的会聚点随之远离。</figcaption>
</figure>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/imaging/lensrays3.webp" alt="图 6.9(c) 光源更近，焦点更远" loading="lazy" />
<figcaption>图 6.9 (c)：光源再近一些，会聚点再远一些。</figcaption>
</figure>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/imaging/lensrays4.webp" alt="图 6.9(d) 会聚点趋向无穷远" loading="lazy" />
<figcaption>图 6.9 (d)：光源继续移近，会聚点趋向无穷远。</figcaption>
</figure>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/imaging/lensrays5.webp" alt="图 6.9(e) 焦点处的光源出射平行光" loading="lazy" />
<figcaption>图 6.9 (e)：从距透镜 f 处发出的光线经透镜后成为平行光，即会聚到无穷远（与 (a) 互为逆过程）。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这五条光路是透镜的「全部语法」——共轭、平行光↔焦点、焦点↔无穷远的互易性都画在这里，手推任何单透镜成像题都用它们。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/imaging/circleOfConfusion.webp" alt="图 6.10 弥散圆与景深" loading="lazy" />
<figcaption>图 6.10：弥散圆与景深。若物与传感器平面是共轭点关系（相距 a、b，满足透镜制造者公式 (6.3)），物在传感器平面上清晰聚焦。但位于聚焦平面前方 δ_a 的物，一般会会聚在传感器平面后方 δ_b 处 [2]。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：弥散圆是「清晰」与「模糊」之间的定量桥梁——只要模糊圆直径小于容忍值，我们就认为「清晰」，景深的概念完全建立在这张图的几何上。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/imaging/dof.webp" alt="图 6.11 景深计算变量与收小光圈的效果" loading="lazy" />
<figcaption>图 6.11：弥散圆与景深。（上）透镜景深计算所用变量。（下）给定可容忍弥散圆时，收窄透镜孔径可以拉长景深范围；代价是到达传感器平面的光变少 [2]。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这张图把「大光圈景深浅、小光圈景深长」的几何原因一次讲清，是摄影与光学系统设计的日常依据。</p>

## 本章小结

- 针孔相机要增加进光量只能以牺牲清晰度为代价；透镜把来自物点、穿过透镜的所有光线聚焦到传感器上同一位置，得到又清晰又明亮的像。
- 用小角近似与薄透镜近似，可以解析设计出具有聚焦性质的透镜面形：表面斜率必须随高度线性变化，抛物面与球面都满足。
- 透镜制造者公式 $\frac{1}{a}+\frac{1}{b}=\frac{1}{f}$，$f=\frac{R}{2(n-1)}$；双面曲率的一般形式为 $\frac{1}{f}=(n-1)(\frac{1}{R_1}+\frac{1}{R_2})$。
- 凸薄透镜与针孔一样成透视投影；共轭点、平行光→焦点、过光心不偏折、放大率 $a/b$ 四条规则可手推一切简单光路。
- 景深由弥散圆容忍度决定：$D\approx 2NCU^2/f^2$，收小光圈（大 $N$）拉长景深但减少进光。
- 凹透镜焦距为负、成虚焦点；凸凹透镜组成望远镜，角放大率 $M=f_1/f_2$。

## 复习自检清单

- ☐ 能写出斯涅尔定律，并说明小角近似下它为什么变成 $\theta_1=n\theta_2$
- ☐ 能对照图 6.4(b) 与表 6.1 推出 $\theta_S=\frac{c}{2(n-1)}(\frac{1}{a}+\frac{1}{b})$（式 6.1），并说明为什么球面/抛物面满足聚焦条件
- ☐ 能写出透镜制造者公式 (6.3) 与双面曲率的一般形式，并说明 $f$ 与 $R$、$n$ 的关系
- ☐ 能复述四条透镜光线追迹规则，并用它们解释「透镜 = 针孔 + 聚光」
- ☐ 能解释共轭点的含义，以及放大率为什么是 $a/b$
- ☐ 能画出弥散圆的几何，推出景深公式 $D\approx 2NCU^2/f^2$，并解释为什么大光圈景深浅
- ☐ 能解释凹透镜的虚焦点与负焦距，并用 $M=f_1/f_2$ 解释望远镜的角放大

---

原文：[Lenses](https://visionbook.mit.edu/lenses.html)
