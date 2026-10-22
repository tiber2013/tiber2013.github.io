---
title: 第 11 章 · 泛化问题
chapter: 11
original: https://visionbook.mit.edu/problem_of_generalization.html
---

# 第 11 章 · 泛化问题

<div class="src-note">
本页为《Foundations of Computer Vision》第 11 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/problem_of_generalization.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

第 9/10 章把学习变成了「在训练集上最小化损失」的优化问题，本章则回到真正关心的问题——训练集上的优化为什么（以及何时）能带来测试集上的好表现。这一章讲欠拟合/过拟合、正则化、贝叶斯先验视角和「数据—先验—假设空间」三工具框架，是连接前两章优化技术与第 12 章起神经网络学习的关键枢纽。

## 本篇要点

- 学习 ≠ 优化：训练集优化只是测试集目标的代理；过拟合 = 拟合了训练数据中不泛化到测试数据的性质（典型如噪声）。
- 近似误差随模型容量增大单调下降，泛化误差呈 U 形：一端欠拟合、一端过拟合。
- 正则化 = 目标函数中惩罚复杂度的项，体现奥卡姆剃刀；$L_2$（岭回归/权重衰减）收缩参数，$L_1$（Lasso）促稀疏。
- 正则项可解释为参数先验：数据损失 = 对数似然，正则项 = 对数先验，MAP 估计自然得到正则化目标。
- 参数量只是模型容量的粗略代理，参数多 ≠ 一定过拟合；深度网络的泛化仍是开放问题。
- 数据、先验、假设空间是约束搜索的三样工具：假设空间是硬约束，数据与先验是软约束，三者效果可相互替代。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 3 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/problem_of_generalization/under_and_overfitting.webp" alt="图 11.1 欠拟合与过拟合" loading="lazy" />
<figcaption>图 11.1：欠拟合与过拟合。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：一张图同时给出三种拟合状态（欠拟合/恰好/过拟合）与「噪声被拟合」的直观画面，是建立泛化直觉的起点。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/problem_of_generalization/under_and_overfitting_vs_polyK.webp" alt="图 11.2 近似误差 vs 泛化误差（随多项式阶数 K 变化）" loading="lazy" />
<figcaption>图 11.2：多项式回归阶数 K 对应的近似误差与泛化误差。此处误差定义为被错误预测的验证点比例（L_2 预测误差大于 0.25 记为错误预测）。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这条 U 形曲线是「模型容量—泛化」直觉的核心，也是实践中画验证曲线、用早停（early stopping）选模型的依据。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/problem_of_generalization/search_space_tools.webp" alt="图 11.4 逼近真理的工具（漫画）" loading="lazy" />
<figcaption>图 11.4：逼近真理的工具漫画。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这幅漫画是全章思想的浓缩——假设空间（硬约束）、先验（软约束）、数据（软约束）三者的交集才是「可学到的真理区域」，建议记住这张图的构图。</p>

## 本章小结

- 学习与优化的区别在于目标：训练集上的优化只是测试集目标的代理。过拟合 = 拟合了训练数据中不泛化的性质（如观测噪声）。
- 近似误差（训练代价）随容量增大单调下降；泛化误差呈 U 形——左端欠拟合，右端过拟合，中间是金发姑娘区间。
- 正则化在目标函数中增加惩罚复杂度的项（式 11.1），是奥卡姆剃刀的工程化；$L_p$ 惩罚中，$p=2$ 为岭回归/权重衰减，$p=1$ 为 Lasso，效果都是把参数压向零或近零、使函数退化到更简单的形式。
- 概率视角：数据损失 = 对数似然，正则项 = 对数先验，正则化目标即 MAP 估计；更复杂的假设空间必然给单一假设分配更少先验质量，故简单假设先验更可能为真。
- 参数量只是模型容量的粗略代理；正则化到近乎全零的百万参数仍定义简单函数类，因此「参数多」不必然过拟合（深度网络泛化仍是开放问题）。
- 数据、先验、假设空间三工具都可约束搜索：假设空间是硬约束，数据与先验是软约束；一样工具的效果可用另一样替代。

## 复习自检清单

- ☐ 能解释「学习与优化的区别」，并用自己的话说出过拟合为什么会发生、训练性能与测试性能为何背离
- ☐ 能写出数据生成过程的四个组成部分（真实关系、噪声分布、含噪观测、数据分布），并说明哪一部分「不泛化」
- ☐ 能写出近似误差 $J_{\texttt{approx}}$ 与泛化误差 $J_{\texttt{gen}}$ 的定义式，并说明为什么可以用留出集近似后者
- ☐ 能推出「多项式回归 = 特征空间 $\phi(x)$ 中的线性回归」，写出数据矩阵 $\mathbf{\Phi}$ 的结构
- ☐ 能写出正则化目标的一般形式 (11.1)，解释 $\lambda$ 增大时解会如何变化
- ☐ 能从贝叶斯法则推出「正则项 = 对数先验」，并用「假设空间先验质量归一」解释贝叶斯奥卡姆剃刀
- ☐ 能复述三个实验的结论：数据越多约束越多（与 Hough 变换的联系）、先验存在「刚刚好」的强度、假设空间太小会连真解一起排除
- ☐ 能解释为什么参数量不能可靠预测过拟合，以及「三样工具效果可相互替代」的含义

---

原文：[The Problem of Generalization](https://visionbook.mit.edu/problem_of_generalization.html)
