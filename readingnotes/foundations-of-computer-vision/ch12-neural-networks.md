---
title: 第 12 章 · 神经网络
chapter: 12
original: https://visionbook.mit.edu/neural_nets.html
---

# 第 12 章 · 神经网络

<div class="src-note">
本页为《Foundations of Computer Vision》第 12 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/neural_nets.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

全书 Part VII「视觉中的神经网络」的开篇，也是第 12-14、24-26 章网络主线共同的地基。本章把神经网络当作第 9 章学习形式化中的一种「参数化假设空间」建立起来：从单个神经元（感知机）到多层感知机（MLP），再到深度网络的一般形式，并回答三个根本问题——网络能表达什么（通用逼近）、为什么需要深度（深度 vs 宽度）、数据如何组织（张量与批）。第 13 章的卷积网络、第 14 章的反向传播，都以本章的记号和直觉为前提。

## 本篇要点

- 感知机 = 线性层 $\mathbf{w}^\mathsf{T}\mathbf{x}+b$ + 阈值激活，本质是线性二分类器，只能解线性可分问题。
- 「线性层 → 点式非线性 → 线性层 → …」是几乎所有神经网络（含本书后续所有架构）的原型模体；纯线性层堆叠复合后仍是线性函数，非线性全部来自激活层。
- 激活（activations）是数据点的快函数，前向几层即得；参数（parameters）是数据集的慢函数，靠整个数据集上的优化得到。二者都是张量，且角色可以互换。
- 深度网络是通用逼近器：宽度足够时可任意精度逼近连续函数；深度往往比宽度更省参数（深度分离）。
- 神经网络是好架构的五个理由：高容量、可微、好的归纳偏置、并行高效、逐层构建抽象。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 6 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/neural_nets/simple_MLP_network.webp" alt="图 12.10 一个简单 MLP 网络" loading="lazy" />
<figcaption>图 12.10：一个简单的 MLP 网络。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是全书第一个「手工设置权重解非线性问题」的完整例子，12.6.1 节的张量/批示例也基于它，务必搞清楚每个神经元在算什么。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/neural_nets/simple_MLP_network_values.webp" alt="图 12.11 MLP 各神经元的取值" loading="lazy" />
<figcaption>图 12.11：图 12.10 所示 MLP 的隐藏单元与输出单元的取值。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：最右侧输出图显示，这个神经网络成功把标记为 1 的数据所在区域赋值为 1——一眼看懂「线性层 + relu」如何折叠出非线性边界。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/neural_nets/curve_as_bump.webp" alt="图 12.12 函数 = 鼓包之和" loading="lazy" />
<figcaption>图 12.12：任何 R → R 函数都可以任意好地用基本鼓包之和逼近。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：把「逼近函数」拆成「一排小鼓包加权求和」是理解通用逼近定理的钥匙，也是「函数 = 基函数之和」这一信号处理思想在神经网络中的第一次亮相。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/neural_nets/bump_as_relus.webp" alt="图 12.13 用 relu 拼出鼓包" loading="lazy" />
<figcaption>图 12.13：一个鼓包可以表示为平移、缩放的 relu 函数的加权和。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：四个 relu 拼出「升起—平台—回落」的鼓包，这是通用逼近定理证明的核心构件；理解了它，就看懂了「网络 = 可学习的基函数展开」。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/neural_nets/pointwise_nonlinearities.webp" alt="图 12.16 常见点式非线性" loading="lazy" />
<figcaption>图 12.16：常见点式非线性：threshold（阈值）、sigmoid、tanh、relu、leaky-relu。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这张图是选激活函数的速查表。重点记住 relu：一半定义域梯度非零、可梯度学习；leaky-relu 给负半轴一个小斜率，用来缓解「dying relu」。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/neural_nets/batchnorm_vs_layernorm_diagram.webp" alt="图 12.17 batchnorm vs layernorm" loading="lazy" />
<figcaption>图 12.17：Batchnorm 与 layernorm 对比。灰色表示计算均值和方差的区域。更多此类可视化参见文献 [11] 的图 2。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：「按列归一化 vs 按行归一化」一句话记住两者区别——batchnorm 跨样本、layernorm 跨通道；后者天然满足 iid，是 Transformer 的标配。</p>

## 本章小结

- 感知机 = 线性层 + 阈值激活 = 线性二分类器；只能解线性可分问题。当年为它定制的学习规则（感知机学习算法、delta 规则等）今天已被通用优化器（梯度下降 / 零阶方法）取代。
- 「线性层 + 点式非线性」反复堆叠是几乎所有神经网络的原型模体；纯线性网络复合仍是线性函数，非线性全部来自激活层（归一化层提供另一种跨神经元非线性）。
- 激活是数据点的快函数（前向几层得到），参数是数据集的慢函数（整个数据集上优化得到）；二者都是张量，且角色可互换——网络提示、对抗攻击、网络可视化都基于此。
- 深度网络是通用逼近器：任意函数 ≈ 一排小鼓包的加权和；每个鼓包由 4 个 relu 拼出（式 12.4），整网只需 linear-relu-linear；$N$ 个鼓包要 $4N$ 个神经元，逼近越精细需要的神经元越多。定理只覆盖紧致子集上的连续函数。
- 深度 vs 宽度：宽度直接增加可拟合鼓包数 $\mathcal{O}(N)$；深度则可能用指数级更少的参数表达某些函数类（深度分离）。
- 张量思维：一切皆张量，batch 几乎总是第一维；MLP 激活 $N\times C_\ell$，图像 $N\times C\times H\times W$，视频再加时间维。
- batchnorm 按列（跨样本、按通道）归一化，layernorm 按行（样本内跨通道）归一化；后者天然满足 iid。
- 网络是好架构的五个理由：高容量、可微、好的归纳偏置、并行高效、逐层构建抽象。

## 复习自检清单

- ☐ 能写出感知机的三个方程（式 12.1），解释为什么 $f$ 数学上是仿射却按惯例叫「线性层」，并说明感知机为什么只能解线性可分问题、其决策面是什么形状
- ☐ 能写出两层 MLP 的前向方程，说明「纯线性层堆叠 = 单个线性层」，并说出视觉中常见的张量形状约定（batch 在第几维；MLP、图像、视频各几维）
- ☐ 能区分激活与参数：各自是什么的函数、快慢之别、为什么说二者角色可以互换（各举一例：提示/对抗攻击/可视化）
- ☐ 能复述通用逼近定理的内容与两个注意事项（仅紧致子集上的连续函数；隐藏神经元要非常多）
- ☐ 能解释式 (12.4)：如何用 4 个 relu 拼出 $\mathbb{1}(\alpha<x<\beta)$，为什么整网只需 linear-relu-linear，以及 $N$ 个鼓包为何要 $4N$ 个神经元
- ☐ 能用自己的话说清深度与宽度对容量影响的区别，以及「深度分离」在主张什么
- ☐ 能写出 batchnorm、layernorm、$L_2$-norm 的公式，并用「按行/按列」一句话区分 batchnorm 与 layernorm
- ☐ 能列出神经网络是好架构的五个理由，并各用一句话解释

---

原文：[Neural Networks](https://visionbook.mit.edu/neural_nets.html)
