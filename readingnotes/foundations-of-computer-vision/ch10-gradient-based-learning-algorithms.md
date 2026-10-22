---
title: 第 10 章 · 基于梯度的学习算法
chapter: 10
original: https://visionbook.mit.edu/gradient_descent.html
---

# 第 10 章 · 基于梯度的学习算法

<div class="src-note">
本页为《Foundations of Computer Vision》第 10 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/gradient_descent.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

Part III「学习基础」的第二章，紧接第 9 章。第 9 章把学习问题分解为目标函数、假设空间、优化器三要素；本章专攻第三要素——优化器：给定损失函数与参数化，如何实际找到最小化损失的参数。本章的一小簇算法（梯度下降、学习率衰减、动量、SGD、梯度裁剪）就是训练几乎所有现代深度模型所需的全部优化基础，第 11-14 章的神经网络直接建立在它们之上。

## 本篇要点

- 优化靠迭代更新参数；按更新时能看到的信息量分为零阶（只有损失值）、一阶（梯度）、高阶（Hessian 等曲率信息）。
- 梯度下降沿负梯度方向更新；学习率 $\eta$ 与步数 $K$ 是两个核心超参数，$\eta$ 足够小则几乎必然收敛到局部极小。
- 学习率衰减先大后小（指数/阶梯/线性）；动量用历史方向的加权和获得惯性，冲过小坑洼，但过大会来回震荡。
- 梯度下降并不要求处处可微：不连续点可用单侧导数（PyTorch 对 ReLU 即如此）；梯度近零会慢、梯度爆炸会崩、多极小值会卡。
- SGD 用小批量（batch）估计梯度：更快、能跃过小凸起，还会隐式正则化（线性问题倾向收敛到最小范数解）。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 4 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/gradient_descent/alg1.webp" alt="算法 10.1 梯度下降（GD）" loading="lazy" />
</figure>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/gradient_descent/momentum_out1.webp" alt="图 10.2 三种动量设置的轨迹对比" loading="lazy" />
<figcaption>图 10.2：(左) 简单损失函数 J = abs(θ)。(右) 三种动量设置 μ 下的优化轨迹。白线表示优化每次迭代时参数的值，从上往下对应迭代推进。颜色表示损失值。红点是损失首次进入最优值 0.01 范围内的位置。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：μ 从 0 到 0.95 的轨迹对比，一眼讲清动量的双刃剑效应，是选择动量超参数时最直观的参照。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/gradient_descent/grad_descent.webp" alt="图 10.3 梯度下降在各种函数上的行为" loading="lazy" />
<figcaption>图 10.3：梯度下降在各种函数上的行为。每个子图左侧为函数 J，红点是使用 η=0.01、μ=0.9 的 GD 找到的解；右侧为 GD 各次迭代中 x 值轨迹，画在每次迭代的 J 上。(a) 对凸函数，随 η 趋于零，GD 收敛。(b) 不连续性不构成本质问题，只要梯度在两侧有定义。(c) 近乎平坦的函数下降会非常慢。(d) 分段常数函数有问题，因为梯度完全消失。(e) 对函数 J=sqrt(abs(θ))-0.25，梯度在极小点处趋于无穷，造成不稳定。(f) 当 J 有多个局部极小值时，我们可能找不到全局最小值。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这六种情形覆盖了训练实践中「loss 不降、震荡、卡住」的绝大多数病因，值得对照排查自己的模型属于哪一种。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/gradient_descent/alg6.webp" alt="算法 10.6 随机梯度下降" loading="lazy" />
</figure>

## 本章小结

- 优化靠迭代；信息等级：零阶（只有 $J$ 值）→ 一阶（梯度 $\nabla_\theta J$）→ 高阶（Hessian 等曲率）。
- GD：$\theta^{k+1} = \theta^k - \eta\nabla_\theta J(\theta^k)$；超参数为学习率 $\eta$ 与步数 $K$；$\eta$ 足够小且初始化随机，则几乎必然收敛到局部极小。
- 学习率衰减先大后小：指数 $\beta^{-k}\eta^0$、阶梯 $\beta^{-\lfloor k/M\rfloor}\eta^0$、线性 $\frac{K-k}{K}\eta^0$；线性与余弦衰减依赖总步数 $K$。
- 动量：$\mathbf{v}^{k+1} = \mu\mathbf{v}^k - \eta\nabla J$，借惯性冲过小坑、沿直路加速；$\mu$ 过大（如 0.95）会冲过最优点且停不下来。替代方案：Nesterov、Adam。
- 可微性不是必需品：不连续点用单侧导数（PyTorch/ReLU 因此可行）；梯度近零 → 慢；极小点梯度无穷 → 爆炸（用梯度裁剪 $\texttt{clip}(v,-m,m)$）；多极小 → 卡住，取决于初始化。
- 无好梯度时：用代理损失的梯度，或进化策略（采样扰动、朝低损失方向移动）。
- SGD：小批量估计 $\tilde{\mathbf{g}} = \frac{1}{B}\sum_b \nabla_\theta \mathcal{L}$；更快、能跃过小凸起、隐式正则化（线性问题倾向最小范数解）。

## 复习自检清单

- ☐ 能解释零阶/一阶/高阶优化各自「看到」什么信息，并各举一例
- ☐ 能写出 GD 更新式，解释 $\eta$ 与 $K$ 的作用及小学习率下的收敛性结论
- ☐ 能写出三种学习率衰减公式，并指出线性衰减的缺点
- ☐ 能写出动量更新式，并用图 10.2 解释 $\mu$ 为何过大反而停不下来
- ☐ 能判断一个函数能否用 GD 有效优化（可微/不连续/梯度消失/梯度爆炸/多极小值）
- ☐ 能解释 PyTorch 如何处理不连续点的梯度，为什么 ReLU 因此可以安心使用
- ☐ 能写出 SGD 的批量梯度估计式，解释 batch size 的权衡及 SGD 相对 GD 的两个额外好处
- ☐ 能写出梯度裁剪的 clip 函数，说明它解决什么问题

---

原文：[Gradient-Based Learning Algorithms](https://visionbook.mit.edu/gradient_descent.html)
