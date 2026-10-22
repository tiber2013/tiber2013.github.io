---
title: 第 36 章 · 训练鲁棒性与泛化性
chapter: 36
original: https://visionbook.mit.edu/data_augmentation.html
---

# 第 36 章 · 训练鲁棒性与泛化性

<div class="src-note">
本页为《Foundations of Computer Vision》第 36 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/data_augmentation.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

第 35 章诊断了病根（训练分布覆盖不到测试分布），本章开药方。全章围绕同一个目标：主动拓宽训练分布，让测试数据「看起来就像训练数据里随手一抽的样本」。两大手段是施加随机变换的数据增强（36.2，含学习式增强），以及把「随机」升级为「最坏情况」的对抗训练（36.3）；最后（36.4）把思路推到极致——堆数据、堆任务，走向通用视觉模型。

## 本篇要点

- 数据增强 = 给每个样本配 $M$ 个随机变换副本 $\{T(\mathbf{x}^{(i)};\theta^{(j)}), \mathbf{y}^{(i)}\}_{j=1}^M$，前提是标签对变换不变：$y(\mathbf{x}) = y(T(\mathbf{x},\theta))$。
- 标签会跟着变的（分割、检测等空间结构化任务）要用等变增强：$x$ 与 $y$ 施加同一组参数 $\theta$。
- 学习式增强用生成模型在隐空间加小扰动 $\mathbf{w}$ 造新样本（算法 36.1）；生成器应在外部大数据上预训练，否则存在循环性、无本质增益。
- 对抗训练把 $\mathbb{E}_\theta$ 换成 $\max_\theta$：min-max 目标（式 36.1、36.2）就是优化领域的鲁棒优化，可提升对 $\epsilon$-球攻击的鲁棒性。
- 多任务训练把多个任务统一成元任务（如数据插补）以共享数据；通用视觉模型的终极路线是数据规模与多样性的扩张。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 2 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/data_augmentation/data_aug_examples.webp" alt="图 36.1 常见数据增强类型" loading="lazy" />
<figcaption>图 36.1：几种常见类型的数据增强。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是全章的入口图——「增强 = 一族随机变换」，后面所有理论（不变性、等变性、对抗训练）都是对这句话的形式化。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/data_augmentation/data_aug_real_example.webp" alt="图 36.2 数据增强如何拓宽训练分布" loading="lazy" />
<figcaption>图 36.2：数据增强拓宽训练分布，使其更好地覆盖测试情形。(a) 训练数据来自 Caltech256 [1]，测试数据来自 CIFAR100 [2]；散点图展示这些数据在二维特征空间中的分布（CLIP [3] 的前两个主成分）。(b) 训练数据加入镜像增强（随机水平翻转）后。(c) 再加上裁剪（裁剪后缩放回原尺寸）。(d) 再加上颜色抖动（颜色和对比度的随机偏移）。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：四个面板一步步展示「分布外」如何被增强「拉回」分布内——这是全章立论的直观证据。</p>

## 本章小结

- 数据增强的核心恒等式：$\{T(\mathbf{x};\theta), \mathbf{y}\}$；标签不变时直接复用 $\mathbf{y}$，标签会变时用等变变换对 $(T_{\mathcal{X}}, T_{\mathcal{Y}})$ 同步变换。
- 增强有效的机制：拓宽训练分布、缩小域间隙（图 36.2 的 (a)→(d) 四步演进）。
- 学习式增强（生成模型隐空间扰动，算法 36.1）能覆盖手工难以编写的变换族；生成器宜在外部数据上预训练，避免循环性。
- 对抗训练 = 把增强分布从 $p_\theta$ 换成最坏情况 $\max_\theta$（式 36.1、36.2），即鲁棒优化；提升对抗鲁棒性。
- 多任务训练通过构造元任务（数据插补：上色 + 修补）共享数据；通用视觉模型的路线是堆数据、堆多样性、堆任务。

## 复习自检清单

- ☐ 能写出数据增强的数学形式 $\{T(\mathbf{x}^{(i)};\theta^{(j)}), \mathbf{y}^{(i)}\}_{j=1}^M$ 及其标签不变性假设
- ☐ 能判断给定任务适合哪些增强：为什么镜像可用于场景分类、不可用于字符识别（b/d）与手性分子
- ☐ 能写出等变增强的条件式，并举语义分割中随机裁剪需同步裁剪标签图的例子
- ☐ 能描述 genaug 流程（编码 → 隐空间扰动 → 解码），并解释在同一数据集上训练生成器的循环性问题
- ☐ 能写出对抗训练的目标函数（式 36.1 / 36.2），并说明它与数据增强期望形式的区别
- ☐ 能按图 36.2 的 (a)→(d) 解释数据增强如何把分布外测试数据变得「分布内」
- ☐ 能说明多任务训练如何通过构造元任务（数据插补：上色 + 修补）实现数据共享

---

原文：[Training for Robustness and Generality](https://visionbook.mit.edu/data_augmentation.html)
