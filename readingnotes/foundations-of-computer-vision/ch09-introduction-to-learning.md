---
title: 第 9 章 · 学习导论
chapter: 9
original: https://visionbook.mit.edu/intro_to_learning.html
---

# 第 9 章 · 学习导论

<div class="src-note">
本页为《Foundations of Computer Vision》第 9 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/intro_to_learning.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

Part III「学习基础」的开篇。前两部分回答了「视觉要解决什么问题」与「图像如何形成」，从本章起进入全书机器学习主线：如何从数据中自动构造出解决问题的算法。本章建立学习问题的统一概念框架（三要素、ERM、概率解释、典型案例），是后续第 10 章（基于梯度的优化）与第 11-14 章（神经网络）的总纲。

## 本篇要点

- 学习是元算法（meta-algorithm）：输入数据、输出算法；训练阶段搜索算法，测试阶段部署到新实例。
- 监督学习（supervised learning）从 {输入, 输出} 例子中归纳函数映射；不可计算函数无法从例子中学到（但可以近似）。
- 任何学习算法三要素：目标函数（objective/loss）、假设空间（hypothesis space）、优化器（optimizer）；假设空间的参数化方式与空间本身同等重要。
- ERM（经验风险最小化）把监督学习形式化为最小化训练集平均损失；高斯误差假设下它等价于最大似然，加先验即 MAP。
- 三个典型案例：线性最小二乘回归（闭式解）、Python 程序归纳（极大假设空间）、softmax 分类（one-hot + 交叉熵）。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 3 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/intro_to_learning/learning_as_meta_algorithm.webp" alt="图 9.1 学习是元算法" loading="lazy" />
<figcaption>图 9.1：学习是一个输出算法的算法。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：「数据 + 学习算法 → 模型」这一框图是 Part III 全部内容的概念骨架，后面每一章都在往这个框里填细节。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/intro_to_learning/ols_summary.webp" alt="图 9.6 线性回归学习问题总览" loading="lazy" />
<figcaption>图 9.6：线性回归找一条直线，从训练数据的 x 值预测其 y 值。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是全书第一次把「三要素 + 数据 → 学得函数」的完整学习问题画成一张图；后续所有案例（包括神经网络）都复用这个模板。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/intro_to_learning/softmax_regression_diagram.webp" alt="图 9.10 softmax 回归数据流" loading="lazy" />
<figcaption>图 9.10：用于图像分类的 softmax 回归。odot 符号表示逐元素乘积。交叉熵损失是预测向量 haty 与标签向量 y 之间逐元素一致性的负和：若 s = y odot log haty 是预测与标签一致程度的得分向量，则交叉熵损失为 H(y, haty) = -Σ_k=1^K s_k。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：「图像 → logits → softmax → 概率分布 → 交叉熵」这条管线是第 12-14 章所有分类网络输出层的原型，务必吃透每一环。</p>

## 本章小结

- 学习 = 元算法：训练阶段在过去实例上搜索算法，测试阶段部署到新实例；「过去/未来」按见没见到划分，不按日历。
- 监督学习从 {输入, 输出} 例子中归纳函数映射（= 把模型拟合到数据）；不可计算函数无法从例子学到，但可以近似。
- 无监督学习有输入、无目标输出（靠目标函数塑造表征）；强化学习只有奖励函数，且须自己收集训练数据。
- 学习三要素：目标函数、假设空间、优化器；参数化方式必须单独指定（同一空间的不同参数化对优化器不等价）；现代 CV 模型普遍过参数化。
- ERM：$\arg\min_{f\in\mathcal{F}} \frac{1}{N}\sum_i \mathcal{L}(f(\mathbf{x}^{(i)}), \mathbf{y}^{(i)})$；高斯误差假设下等价于最大似然；加先验即 MAP（贝叶斯法则）。
- 三案例：线性最小二乘（$L_2$ 损失，正规方程闭式解）；程序归纳（假设空间 = 所有 Python 程序）；softmax 分类（one-hot 目标，0-1 损失 NP-hard 故改用交叉熵）。
- 元学习 = 以学习算法为假设空间的学习，是元-元算法；按此定义，进化也是学习算法。

## 复习自检清单

- ☐ 能解释「学习是元算法」及训练/测试两阶段的划分，并说明「过去/未来」的确切含义
- ☐ 能写出监督学习中「例子」的形式定义，并举出一个无法从例子学习的函数（停机问题）
- ☐ 能区分无监督学习与强化学习：各自有什么输入、没有什么
- ☐ 能写出学习算法三要素，并解释为什么参数化方式需要单独指定
- ☐ 能写出 ERM 的数学形式，并说明它在什么假设下等价于最大似然、如何推广到 MAP
- ☐ 能从零推导线性最小二乘的正规方程 $\theta^* = (\mathbf{X}^\mathsf{T}\mathbf{X})^{-1}\mathbf{X}^\mathsf{T}\mathbf{y}$
- ☐ 能解释 one-hot 编码、0-1 损失为何难以优化、交叉熵损失的形式与概率解释
- ☐ 能写出 softmax 公式，解释 logits 与概率分布的关系，并说明 $-\log 0.4$ 这个损失值的来历

---

原文：[Introduction to Learning](https://visionbook.mit.edu/intro_to_learning.html)
