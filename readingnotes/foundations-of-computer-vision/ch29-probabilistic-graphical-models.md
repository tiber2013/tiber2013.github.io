---
title: 第 29 章 · 概率图模型
chapter: 29
original: https://visionbook.mit.edu/graphical_models.html
---

# 第 29 章 · 概率图模型

<div class="src-note">
本页为《Foundations of Computer Vision》第 29 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/graphical_models.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

本章把贝叶斯推断的基本工具整合成一套模块化框架：用图（节点表示随机变量、边表示统计依赖）紧凑地描述复杂联合分布，并利用图所编码的条件独立性，把原本指数复杂度的多维边际化变成高效的消息传递。MRF + 信念传播/图割是分割、立体等中层次视觉任务的基础设施；本章末尾的立体视觉示例与第 40 章直接呼应，是「如何把一个视觉问题写成图模型」的最佳范本。

## 本篇要点

- 无向图模型的联合分布必可分解为最大团（maximal clique）势函数的乘积（Hammersley-Clifford 定理）；有向图模型则分解为「节点变量在其父节点给定下的条件分布」之乘积。
- 马尔可夫链与马尔可夫随机场（MRF）是视觉中最常用的两类无向结构：MRF = 二维网格 + 成对势 $\phi_{ij}$ + 每点局部证据 $\psi_i$，分割、去噪、立体通吃。
- 树/链上，信念传播（BP）把边际化重排成消息传递：消息就是对「不含目标节点的子树」的求和；复杂度从 $|x|^N$ 降到 $(N-1)|x|^2$——指数变线性，且消息可在不同节点的边际计算之间复用。
- 消息更新规则见式 (29.7)；节点边际 = 所有流入消息（含局部势）之积，见式 (29.8)。
- 带环图（如 2D MRF）上用 loopy BP 近似，其不动点对应 Bethe 自由能的极小；求 MAP 用 max-product 版本，实战中常用图割（graph cuts）与树重加权 BP（TRW-BP）。
- 点估计二选一：MMSE = 后验均值（惩罚与误差大小成正比），MAP = 后验概率最大者（所有错误等罚）；高斯分布下二者相同。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 9 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/graphical_models/x1bx2bx3.webp" alt="图 29.1 三个相依变量" loading="lazy" />
<figcaption>图 29.1：三个相依变量。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是全章的「母题」图——一条三节点链加一个因子分解，条件独立、消息传递、复杂度节省全都从它生长出来，务必看熟。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/graphical_models/x1x2x3y1y2y3.webp" alt="图 含隐变量与观测变量的马尔可夫链" loading="lazy" />
<figcaption>图：含三个观测变量（阴影）与三个隐变量的马尔可夫链。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：「隐变量链 + 每点观测」是视觉时序/扫描线推断的标准骨架，29.6.4 节的立体视觉示例就是它的直接实例化。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/graphical_models/mrf.webp" alt="图 29.3 二维马尔可夫随机场" loading="lazy" />
<figcaption>图 29.3：二维马尔可夫随机场。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：MRF 是视觉中最常用的图模型结构（分割、去噪、立体都靠它），请记住「网格 + 成对势 + 局部证据」三要素。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/graphical_models/3bpc2.webp" alt="图 29.6 BP 消息总览" loading="lazy" />
<figcaption>图 29.6：一个简单 BP 示例中全部消息（部分和）的汇总。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：一张图看清 BP 在全网要算哪些消息——消息就是可复用的部分和，这正是 BP 高效的根源。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/graphical_models/bpmotivator2.webp" alt="图 29.7 BP 更新规则的推导" loading="lazy" />
<figcaption>图 29.7：引出 BP 更新规则的例子。(a) 无环图的边际化；(b) x_j 处的部分和如何分配到各节点。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是消息更新规则 (29.7) 的推导现场——把边际化的求和号「推过」与它无关的节点，剩下的部分和就是消息；理解了这一步，BP 就不再是黑盒。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/graphical_models/bpdiscrete1.webp" alt="图 29.8 BP 消息传递规则的图形化" loading="lazy" />
<figcaption>图 29.8：式 (29.7) BP 消息传递规则的图形化表示，其中 odot 表示逐元素相乘（即 Hadamard 积）。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：把式 (29.7) 翻译成了「流入消息逐元素相乘（⊙）→ 乘成对势 → 对发送方状态求和」的流水线，背下这张图就会写 BP 代码。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/graphical_models/bpi.webp" alt="图 29.9 节点边际 = 流入消息之积" loading="lazy" />
<figcaption>图 29.9：计算节点 i 处的边际概率——把该节点所有流入消息相乘，p_i 也视作另一条消息。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：式 (29.8)——节点边际等于所有流入消息之积。消息算完之后，取任何节点的边际几乎是零成本，这是 BP 工程价值的核心。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/graphical_models/stereomodel2.webp" alt="图 29.13 立体视差推断的图模型" loading="lazy" />
<figcaption>图 29.13：立体系统中左右相机视图间视差偏移后验概率的图模型。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：立体视差推断的完整图模型——观测证据、Potts 先验、链式结构各就各位，是「把视觉问题写成图模型」的样板。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/graphical_models/bpcanoes5.webp" alt="图 29.14 BP 求解立体问题端到端结果" loading="lazy" />
<figcaption>图 29.14：对图 29.13 的图模型运行 BP 求解立体问题。(a) 右视图与 (b) 左视图，黑线为所分析的行。(c) 每个深度视差在左相机上的局部证据。(d) 各位置最终的向右与 (e) 向左 BP 消息。(f) 最终边际化后验概率，在每个左相机像素位置上准确找到了深度不连续处。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：BP 的端到端战绩——嘈杂的局部证据被沿扫描线的消息聚合后，准确锁定独木舟边界的深度不连续；「证据 + 平滑先验 + 消息传递」的范式在此一览无余。</p>

## 本章小结

- 核心交易：用「图结构 = 条件独立性」把联合概率拆成团的乘积（Hammersley-Clifford），从而让多维推断可计算；无向看团与条件独立，有向看条件分解与因果。
- 两大基本结构：马尔可夫链（时序/扫描线）与 MRF（空间网格）；MRF 联合分布 = 成对势 $\phi_{ij}$ × 局部证据 $\psi_i$ 的乘积。
- BP = 边际化的智能重排：消息是可复用的部分和；树上精确，复杂度从 $|x|^N$ 变 $(N-1)|x|^2$。
- 两个必背方程：消息更新 $m_{ji}(x_i)=\sum_{x_j}\psi_{ij}(x_i,x_j)\prod_{k\in\eta(j)\backslash i}m_{kj}(x_j)$；边际 $p_i(x_i)=\prod_{j\in\eta(i)}m_{ji}(x_i)$。
- 带环图用 loopy BP（不动点 ↔ Bethe 自由能极小）或 max-product + 图割/TRW-BP 求 MAP。
- 立体示例完整示范了「建模 → 局部证据 → 先验 → BP 扫描 → 边际」全流程，嘈杂证据经消息聚合后可准确恢复深度不连续。

## 复习自检清单

- ☐ 能写出 Hammersley-Clifford 定理的内容，并区分「团」与「最大团」
- ☐ 能写出三节点链的分解式 (29.1)，并解释为什么给定 $x_2$ 后 $x_1$ 与 $x_3$ 条件独立
- ☐ 能写出 MRF 的条件联合分布，说明成对势 $\phi_{ij}$ 与局部证据 $\psi_i$ 各自的角色
- ☐ 能推导 BP 消息更新规则 (29.7)，并说清「消息 = 对不含目标节点的子树求和」
- ☐ 能由式 (29.8) 说明节点边际为何等于所有流入消息（含局部势）之积
- ☐ 能解释朴素边际化的 $|x|^N$ 复杂度如何被 BP 降到 $(N-1)|x|^2$，消息复用体现在哪
- ☐ 能区分深度优先与并行两种消息调度，并说明为何两轮扫描即可得到所有节点的边际
- ☐ 能复述立体视觉示例中局部证据、Potts 先验、BP 扫描各自的作用；并说明 loopy BP 的适用条件、不动点与 Bethe 自由能的关系，以及 max-product 与 MAP 的联系

---

原文：[Probabilistic Graphical Models](https://visionbook.mit.edu/graphical_models.html)
