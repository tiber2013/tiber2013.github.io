---
title: 第 55 章 · 简单视觉系统回顾
chapter: 55
original: https://visionbook.mit.edu/simplesystem_final.html
---

# 第 55 章 · 简单视觉系统回顾

<div class="src-note">
本页为《Foundations of Computer Vision》第 55 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/simplesystem_final.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

全书收官之章。本章回到第 2 章引入的「简单视觉世界」玩具问题——从单幅图像恢复三维场景结构——但不再从第一性原理手工搭建流水线，而是下载一个预训练的端到端视觉系统（MiDaS）开箱使用，再请 GPT-4V 做场景理解，与第 2 章笔记首尾呼应。全章最终收束到一个核心判断：手工模型在假设内优美、可解释但脆弱；预训练模型不干净、机理不明，却鲁棒且能泛化。

## 本篇要点

- MiDaS 是当前单目深度估计的最先进水平（state of the art）：在来源多样的真实数据集上端到端训练，开箱即用地「解」第 2 章的玩具问题——能用，但不干净（地面凹陷、方片面扭曲）。
- 因训练数据来源多样（整体尺度、相机参数、立体基线均有歧义），MiDaS 的损失对这些歧义不变（invariant loss），输出是「差一个未知尺度与偏移的视差」，不能直接当深度用。
- 从输出到世界坐标：先用 $z[n,m]=1/(d[n,m]+d_0)$ 把视差翻成深度，再按针孔相机模型反投影（式 55.1），需要内参、外参与 $d_0$ 的取值。
- 对比结论：手工系统假设内结果优美、失效可指认，但光照稍差就误差级联、地面换色即崩溃；预训练系统质量从不完美，却很少崩。
- GPT-4V 能准确描述场景（其内部用的仍是全书讲过的机制：早期层检测边缘、类小波滤波器铺满傅里叶空间、透视几何找消失点），但按描述重渲染图像仍会出错——「已解决的未解决问题」无处不在。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 2 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/simplesystem_revisited/views_midas.webp" alt="图 55.3 MiDaS 预训练模型的三维重建" loading="lazy" />
<figcaption>图 55.3：使用预训练模型的三维重建。图中展示了从多个视点观看由图 55.1 重建出的三维场景。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是本章的核心实验结果——一个只在自然图像上训练的模型，开箱即用地「解」玩具问题：大致能用，但几何并不干净。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/simplesystem_revisited/comparison_midas_simpleworld.webp" alt="图 55.5 域外泛化对比" loading="lazy" />
<figcaption>图 55.5：域外泛化（out-of-domain generalization）。对预训练模型而言，这些图像可能仍处于训练域内，因为其训练集包含非常多样的自然图像；简单世界模型只对第一个例子有效，对其余例子完全失效，给出彻底错误的重建。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这一对比是本章的结论所在：手工模型只在假设成立时优美，预训练模型以「永远不完美但很少崩」换取泛化。</p>

## 本章小结

本章的收获：使用大型预训练模型大致可行——它不像第 2 章的结果那么干净，但它在简单启发式方法崩溃的情况下依然有效；而且可以预期，未来预训练模型的性能会持续提升。

- 第 2 章的手工模型让我们对系统的机制、假设与局限有全面的理解；另一个有趣的优势是它不需要学习、不需要训练数据——模型写下即可工作，条件满足时效果极好，但一旦偏离那些条件就脆弱、不可靠，无法泛化到假设空间之外；系统失效时容易指出错在哪，但知道错在哪无助于让它工作得更好。
- 预训练的三维重建系统则在一定程度上泛化到了我们的设定，能处理不在建模假设内的构型；它对光照变化鲁棒，不依赖边缘检测、图形-背景分割（figure-ground segmentation）这类脆弱操作——但类似的等价操作可能正在网络内部发生：三维信息仍主要包含在边缘中，信息仍必须从边缘与连接点（junction）传播到平坦表面。预训练模型究竟如何工作？它学到了我们手工引入的某些启发式，还是用了别的？这些问题的精确答案现在并不重要。
- GPT-4V 似乎能识别几何图形及其空间配置，但它与渲染引擎的衔接很脆弱，没能用相同的元素创建一幅新图像（尽管已经非常接近）。但它为什么失败？领域在快速演进，这些问题终将被逐一解决。重要的是：你应该质疑现有方法，并为你遇到的任何新系统找出真正相关的问题。
- 面对基于学习的系统，我们做的实验类似于心理物理学（psychophysics）试图理解人类视觉系统机制时所做的实验。

## 复习自检清单

- ☐ 能解释 MiDaS 为什么预测视差而不是深度，以及不变损失（invariant loss）带来的好处与代价
- ☐ 能写出视差到深度的转换 $z[n,m]=1/(d[n,m]+d_0)$ 与反投影公式 (55.1)，并说明 $\mathbf{M}$、$\mathbf{R}$ 及 $a$、$n_0$、$m_0$、$\theta$ 各自的含义
- ☐ 能复述本章的启发式标定取值：$d_0=0$、$n_0=N/2$、$m_0=M/2$、$a=N$、$\theta=\pi/4$
- ☐ 能对比第 2 章手工系统与预训练系统在简单世界图像上的表现差异（各指出两个具体的成功点或失效点）
- ☐ 能解释为什么「地面由白变黑」会让第 2 章的手工流水线崩溃（它依赖边缘/对比度证据），而预训练系统不受影响
- ☐ 能说明 GPT-4V「描述准确」与「重渲染走样」之间的差距意味着什么
- ☐ 能用自己的话阐述「已解决的未解决问题」，并举一个计算机视觉领域的例子

---

原文：[A Simple Vision System—Revisited](https://visionbook.mit.edu/simplesystem_final.html)
