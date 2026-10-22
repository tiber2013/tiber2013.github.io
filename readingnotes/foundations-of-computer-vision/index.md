---
title: Foundations of Computer Vision
description: Torralba / Isola / Freeman 著《Foundations of Computer Vision》中文读书笔记
---

# Foundations of Computer Vision

<p class="book-sub" style="font-size:15px;color:var(--vp-c-text-3);margin-top:-8px;">Antonio Torralba · Phillip Isola · William T. Freeman — The MIT Press, 2024</p>

<dl class="book-meta">
  <dt>作者</dt><dd>Antonio Torralba、Phillip Isola、William T. Freeman</dd>
  <dt>出版</dt><dd>The MIT Press，2024（Adaptive Computation and Machine Learning 系列）</dd>
  <dt>ISBN</dt><dd>978-0-262-04897-2</dd>
  <dt>篇幅</dt><dd>15 个部分，55 章</dd>
  <dt>官网</dt><dd><a href="https://visionbook.mit.edu/" target="_blank" rel="noopener">visionbook.mit.edu</a>（全书免费公开）</dd>
  <dt>授权</dt><dd><a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a></dd>
  <dt>本笔记</dt><dd>55 篇章节笔记 · 253 幅核心插图 · AI 辅助整理</dd>
</dl>

## 这是一本什么书

三位 MIT 教授写了十多年的一本「小书」——每章原本限定不超过 5 页，结果还是没守住。它不追 SOTA，也不做领域综述，而是把计算机视觉的**基础概念**按「图像形成 → 学习 → 图像处理 → 神经网络 → 概率模型 → 生成模型 → 几何 → 运动 → 场景理解」的顺序重新串了一遍。

全书的核心隐喻是**多视角**（multiple views）：真实场景只有一个，但我们从不同角度、不同传感器、不同时刻去观察它，通过视图的集合逼近底层现实。这本书本身也是若干「视角」的集合。

适合谁读：刚进入该领域的本科生、研究生，以及想补齐底层直觉的从业者。如果你只做检测/分割而说不清傅里叶、采样、单应矩阵，这本书正好补上。

## 本笔记的定位与做法

这里放的是**复习用的摘要**，不是译本：

- **本章定位** —— 这一章在全书中处于什么位置、依赖哪些前置章节。
- **本篇要点** —— 一章的核心结论，通常 4–7 条，先看这个能快速建立骨架。
- **关键插图** —— 原书标注为最高优先级（三星）的插图，配原图注。
- **本章小结** —— 公式与结论的浓缩版，适合考前扫读。
- **复习自检清单** —— 写成「能做到某件具体的事」，卡住的那条就是没掌握的地方。

每页都保留了指向原书对应章节的链接。**遇到公式推导、完整论证、更多插图，请回到原书**——那是免费的，而且比摘要可靠。

## 引用

```bibtex
@book{foundationsCVbook,
  title     = {Foundations of Computer Vision},
  author    = {Torralba, A. and Isola, P. and Freeman, W.T.},
  isbn      = {9780262378666},
  series    = {Adaptive Computation and Machine Learning series},
  year      = {2024},
  publisher = {MIT Press},
  url       = {https://mitpress.mit.edu/9780262048972/foundations-of-computer-vision/}
}
```

## 版权与使用声明

::: warning 请先读这一段
原书《Foundations of Computer Vision》© 2024 Antonio Torralba、Phillip Isola、William T. Freeman，由 The MIT Press 出版，采用 [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) 授权，官网 [visionbook.mit.edu](https://visionbook.mit.edu/) 提供免费全文。

本站内容说明：

1. **本站发布的是学习笔记摘要，不是原书译文。** 依据 CC 协议的 **ND（禁止演绎）** 条款，本站不提供逐节翻译，只提供要点归纳、结构梳理与自测清单，并逐页链接回原文。
2. **插图**为原书图片的完整复制（未作修改），版权归原作者所有，此处仅作非商业的学习引用，并按 **BY（署名）** 要求标注出处。如需引用请同样注明原作者与官网。
3. **笔记由 AI 辅助整理**（提取、归纳、排版），可能存在疏漏或表述偏差。**以原书为准**；发现错误欢迎提 issue。
4. 本站为个人学习站点，无广告、无付费内容，符合 **NC（非商业）** 条款。
5. 若原作者或出版方认为本页内容不妥，请联系删除，会立即处理。
:::

## 全书结构

<div class="part-block">
<h3>前言 <span class="part-en">Preface &amp; Notation</span></h3>
<ul class="ch-list">
  <li><a href="./preface">前言</a></li>
  <li><a href="./notation">符号约定</a></li>
</ul>
</div>

<div class="part-block">
<h3>Part I · 基础 <span class="part-en">Foundations</span></h3>
<ul class="ch-list">
  <li><a href="./ch01-the-challenge-of-vision">第 1 章 · 视觉的挑战</a></li>
  <li><a href="./ch02-a-simple-vision-system">第 2 章 · 简单视觉系统</a></li>
  <li><a href="./ch03-looking-at-images">第 3 章 · 观看图像</a></li>
  <li><a href="./ch04-computer-vision-and-society">第 4 章 · 计算机视觉与社会</a></li>
</ul>
</div>

<div class="part-block">
<h3>Part II · 图像形成 <span class="part-en">Image Formation</span></h3>
<ul class="ch-list">
  <li><a href="./ch05-imaging">第 5 章 · 成像</a></li>
  <li><a href="./ch06-lenses">第 6 章 · 镜头</a></li>
  <li><a href="./ch07-cameras-as-linear-systems">第 7 章 · 作为线性系统的相机</a></li>
  <li><a href="./ch08-color">第 8 章 · 颜色</a></li>
</ul>
</div>

<div class="part-block">
<h3>Part III · 学习基础 <span class="part-en">Foundations of Learning</span></h3>
<ul class="ch-list">
  <li><a href="./ch09-introduction-to-learning">第 9 章 · 学习导论</a></li>
  <li><a href="./ch10-gradient-based-learning-algorithms">第 10 章 · 基于梯度的学习算法</a></li>
  <li><a href="./ch11-the-problem-of-generalization">第 11 章 · 泛化问题</a></li>
  <li><a href="./ch12-neural-networks">第 12 章 · 神经网络</a></li>
  <li><a href="./ch13-neural-networks-as-distribution-transformers">第 13 章 · 作为分布变换器的神经网络</a></li>
  <li><a href="./ch14-backpropagation">第 14 章 · 反向传播</a></li>
</ul>
</div>

<div class="part-block">
<h3>Part IV · 图像处理基础 <span class="part-en">Foundations of Image Processing</span></h3>
<ul class="ch-list">
  <li><a href="./ch15-linear-image-filtering">第 15 章 · 线性图像滤波</a></li>
  <li><a href="./ch16-fourier-analysis">第 16 章 · 傅里叶分析</a></li>
</ul>
</div>

<div class="part-block">
<h3>Part V · 线性滤波器 <span class="part-en">Linear Filters</span></h3>
<ul class="ch-list">
  <li><a href="./ch17-blur-filters">第 17 章 · 模糊滤波器</a></li>
  <li><a href="./ch18-image-derivatives">第 18 章 · 图像导数</a></li>
  <li><a href="./ch19-temporal-filters">第 19 章 · 时域滤波器</a></li>
</ul>
</div>

<div class="part-block">
<h3>Part VI · 采样与多尺度 <span class="part-en">Sampling and Multiscale</span></h3>
<ul class="ch-list">
  <li><a href="./ch20-image-sampling-and-aliasing">第 20 章 · 图像采样与混叠</a></li>
  <li><a href="./ch21-downsampling-and-upsampling">第 21 章 · 下采样与上采样</a></li>
  <li><a href="./ch22-filter-banks">第 22 章 · 滤波器组</a></li>
  <li><a href="./ch23-image-pyramids">第 23 章 · 图像金字塔</a></li>
</ul>
</div>

<div class="part-block">
<h3>Part VII · 视觉神经网络架构 <span class="part-en">Neural Architectures for Vision</span></h3>
<ul class="ch-list">
  <li><a href="./ch24-convolutional-neural-nets">第 24 章 · 卷积神经网络</a></li>
  <li><a href="./ch25-recurrent-neural-nets">第 25 章 · 循环神经网络</a></li>
  <li><a href="./ch26-transformers">第 26 章 · Transformer</a></li>
</ul>
</div>

<div class="part-block">
<h3>Part VIII · 图像概率模型 <span class="part-en">Probabilistic Models of Images</span></h3>
<ul class="ch-list">
  <li><a href="./ch27-statistical-image-models">第 27 章 · 统计图像模型</a></li>
  <li><a href="./ch28-textures">第 28 章 · 纹理</a></li>
  <li><a href="./ch29-probabilistic-graphical-models">第 29 章 · 概率图模型</a></li>
</ul>
</div>

<div class="part-block">
<h3>Part IX · 生成模型与表示学习 <span class="part-en">Generative Models & Representation Learning</span></h3>
<ul class="ch-list">
  <li><a href="./ch30-representation-learning">第 30 章 · 表示学习</a></li>
  <li><a href="./ch31-perceptual-grouping">第 31 章 · 知觉组织</a></li>
  <li><a href="./ch32-generative-models">第 32 章 · 生成模型</a></li>
  <li><a href="./ch33-generative-modeling-meets-representation-learning">第 33 章 · 生成建模与表示学习的交汇</a></li>
  <li><a href="./ch34-conditional-generative-models">第 34 章 · 条件生成模型</a></li>
</ul>
</div>

<div class="part-block">
<h3>Part X · 学习型视觉的挑战 <span class="part-en">Challenges in Learning-Based Vision</span></h3>
<ul class="ch-list">
  <li><a href="./ch35-data-bias-and-shift">第 35 章 · 数据偏差与偏移</a></li>
  <li><a href="./ch36-training-for-robustness">第 36 章 · 训练鲁棒性与泛化性</a></li>
  <li><a href="./ch37-transfer-learning-and-adaptation">第 37 章 · 迁移学习与适配</a></li>
</ul>
</div>

<div class="part-block">
<h3>Part XI · 理解几何 <span class="part-en">Understanding Geometry</span></h3>
<ul class="ch-list">
  <li><a href="./ch38-representing-images-and-geometry">第 38 章 · 图像与几何的表示</a></li>
  <li><a href="./ch39-camera-modeling-and-calibration">第 39 章 · 相机建模与标定</a></li>
  <li><a href="./ch40-stereo-vision">第 40 章 · 立体视觉</a></li>
  <li><a href="./ch41-homographies">第 41 章 · 单应矩阵</a></li>
  <li><a href="./ch42-single-view-metrology">第 42 章 · 单视图度量学</a></li>
  <li><a href="./ch43-depth-from-a-single-image">第 43 章 · 单图深度估计</a></li>
  <li><a href="./ch44-multiview-geometry-and-sfm">第 44 章 · 多视图几何与运动恢复结构</a></li>
  <li><a href="./ch45-radiance-fields">第 45 章 · 辐射场</a></li>
</ul>
</div>

<div class="part-block">
<h3>Part XII · 理解运动 <span class="part-en">Understanding Motion</span></h3>
<ul class="ch-list">
  <li><a href="./ch46-motion-estimation">第 46 章 · 运动估计</a></li>
  <li><a href="./ch47-3d-motion-and-its-2d-projection">第 47 章 · 三维运动及其二维投影</a></li>
  <li><a href="./ch48-optical-flow-estimation">第 48 章 · 光流估计</a></li>
  <li><a href="./ch49-learning-to-estimate-motion">第 49 章 · 学习估计运动</a></li>
</ul>
</div>

<div class="part-block">
<h3>Part XIII · 场景理解与语言 <span class="part-en">Scene Understanding & Language</span></h3>
<ul class="ch-list">
  <li><a href="./ch50-object-recognition">第 50 章 · 目标识别</a></li>
  <li><a href="./ch51-vision-and-language">第 51 章 · 视觉与语言</a></li>
</ul>
</div>

<div class="part-block">
<h3>Part XIV · 研究、写作与演讲 <span class="part-en">On Research, Writing and Speaking</span></h3>
<ul class="ch-list">
  <li><a href="./ch52-how-to-do-research">第 52 章 · 如何做研究</a></li>
  <li><a href="./ch53-how-to-write-papers">第 53 章 · 如何写论文</a></li>
  <li><a href="./ch54-how-to-give-talks">第 54 章 · 如何做报告</a></li>
</ul>
</div>

<div class="part-block">
<h3>Part XV · 收官 <span class="part-en">Closing Remarks</span></h3>
<ul class="ch-list">
  <li><a href="./ch55-a-simple-vision-system-revisited">第 55 章 · 简单视觉系统回顾</a></li>
</ul>
</div>

## 建议阅读路径

| 你的情况 | 建议路线 |
|---|---|
| 完全忘了，想重建体系 | 顺序读 Part I–III（第 1–14 章）打底，再按需跳转 |
| 只做检测/分割，缺底层 | 重点 Part IV–VIII（第 15–29 章）：滤波、傅里叶、采样、导数、CNN、统计模型 |
| 只做深度学习，缺几何 | Part XI（第 38–45 章）：齐次坐标、标定、立体、单应、SfM、NeRF |
| 做多模态 / VLM | Part IX + Part XIII（第 30–34、50–51 章） |
| 时间极少（约 2 小时） | 只读每章「本篇要点」+ 本页的「关键插图」 |
