---
title: 第 35 章 · 数据偏差与偏移
chapter: 35
original: https://visionbook.mit.edu/bias_and_shift.html
---

# 第 35 章 · 数据偏差与偏移

<div class="src-note">
本页为《Foundations of Computer Vision》第 35 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/bias_and_shift.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

全书靠后部分的一章，前面各章建立了成像、滤波、学习与深度网络的工具箱，本章回答一个残酷的问题：为什么在训练集上表现良好的系统，一上线就频频翻车？核心病根是训练分布与测试分布不一致。本章只讲「会出什么问题」，缓解手段留给随后两章——第 36 章（数据增强 / 对抗训练）与第 37 章（迁移学习 / 模型自适应）。

## 本篇要点

- 学习的目标是从过去数据中获取对未来数据也成立的规则；经典理论假设未来与过去同分布，现实中几乎从不成立——「数据偏移是常态，不是例外」。
- ColorizeBot 案例：在 ImageNet 上训练的上色模型被 Reddit 用户一用就露馅，病因是内容分布（ImageNet 约 12% 是狗、几乎没有人像）与摄影风格（现代数码 vs 老胶片颗粒）的双重偏移。
- 玩具回归例子给出定量直觉：跨分布训练误差显著更大；有偏数据不是不能用，只是「汇率」差——数据集 B 的样本在 A 的测试集上价值约为 A 样本的 $1/10$（需约 10 倍数据量）。
- 偏差无处不在且有来源可循：拍摄者偏差、采集偏差、标注偏差、社会偏差（公平性问题）、测试/基准偏差；「Name That Dataset!」游戏证明人类与分类器都能轻易识别数据集风格。
- 偏移可以是对抗性的：在 $r$-球内优化不可察觉的噪声，即可让网络输出任意类别；对抗分析即最坏情况分析，对策（对抗训练）见第 36 章。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 5 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/bias_and_shift/colorizebot_examples.webp" alt="图 35.2 ColorizeBot 的失败案例" loading="lazy" />
<figcaption>图 35.2：Reddit 的 ColorizeBot [2] 的两个结果。原图分别为（左）Louis-Auguste Bisson 1849 年拍摄的肖邦肖像，（右）1933 年的袋狼 Benjamin。该机器人用 [1] 的模型为原始黑白照片上色。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是「训练集效果好 ≠ 能落地」的标志性案例——同一个模型换一批数据就从惊艳变成滑稽，全部原因都在数据分布上。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/bias_and_shift/dogs_with_tongues.webp" alt="图 35.3 狗嘴下的粉色团块" loading="lazy" />
<figcaption>图 35.3：（左）使用 ImageNet 上训练的模型得到的上色结果（示例由 Richard Zhang 制作；输入黑白照片来自 ImageNet [3]）。（右）ImageNet 中的狗图像示例 [3]。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：「狗嘴下面一律涂一团粉色」这一个细节，把「模型学到的是数据集统计规律而非世界规律」讲得入木三分。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/bias_and_shift/toy_example/training_test_sets.webp" alt="图 35.6 玩具例子：不同数据集上的训练与测试" loading="lazy" />
<figcaption>图 35.6：在不同数据集上训练与测试。本示意中所有数据集都只有 5 个样本。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：四个子图一眼说明「同分布训练测试误差小、跨分布误差大」，且每条虚线拟合只在自己那一块数据区域内成立。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/bias_and_shift/toy_example/test_as_function_of_data.webp" alt="图 35.7 泛化误差随训练数据量的变化" loading="lazy" />
<figcaption>图 35.7：泛化误差随训练样本数的变化。测试集为数据集 A。每条曲线分别表示用数据集 A 与数据集 B 训练时的表现（与图 35.6 相同的分布）。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这张图给出全书最实用的一条定量结论——有偏数据不是没用，只是「汇率」差：在 A 的测试集上评估时，B 样本的价值约为 A 样本的 1/10。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/bias_and_shift/datasetsJeopardy.webp" alt="图 35.8 Name That Dataset! 游戏" loading="lazy" />
<figcaption>图 35.8：来玩「Name That Dataset!」游戏。改自 [4]。玩法：把每组三张图像与右侧列表中的数据集名称关联起来。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：它用一个游戏证明了「每个数据集都有自己的风格指纹」——这正是域间隙存在的直接证据，也是后续所有域自适应方法的动机。</p>

## 本章小结

- 训练分布与测试分布不一致（分布偏移 / 域间隙）是 ML 系统部署失败的主因；「数据偏移是常态，不是例外」。
- 定量直觉：跨分布训练误差更大；有偏数据源样本的价值可能只有同分布样本的 $1/10$（需约 10 倍数据量）。
- 偏差无处不在：拍摄者偏差（规范视角）、采集偏差（搜索引擎式图像）、标注偏差（选图选标签）、社会偏差（医生-男性 / 护士-女性）、测试/基准偏差（用有偏测试集选型会得出错误结论）。
- 偏差是相对概念；我们只能通过（必然有偏的）数据集触及真实世界。
- 对抗偏移：在 $r$-球内优化不可察觉的噪声，即可让网络输出任意类别；对抗分析 = 最坏情况分析，对策见第 36 章。
- 两条总体应对路线：域随机化 / 数据增强（第 36 章）、迁移学习 / 模型自适应（第 37 章）。

## 复习自检清单

- ☐ 能讲清 ColorizeBot 失败的原因，并指出训练/测试数据在内容与摄影风格两方面的具体差异
- ☐ 能画出「训练在平坦区、测试在陡峭区」的回归示意，解释为什么同分布下学到的拟合在测试区失效
- ☐ 能写出玩具例子的数据生成过程与带 $\lambda$ 正则的最小二乘目标函数
- ☐ 能解释表 35.1 中对角线与非对角线误差差异的来源，以及「B 样本价值约为 A 的 $1/10$」这一结论
- ☐ 能列举五种偏差来源并各举一例，说明它们如何复合（如标注偏差复合拍摄者偏差）
- ☐ 能解释「Name That Dataset!」游戏为什么可能赢，以及它证明了什么
- ☐ 能写出对抗攻击的优化目标（$\arg\max_\epsilon$ s.t. $\|\epsilon\|<r$），并说明为何它要求攻击者可访问系统
- ☐ 能解释「对抗分析 = 最坏情况分析」，以及它与经验风险最小化的关系

---

原文：[Data Bias and Shift](https://visionbook.mit.edu/bias_and_shift.html)
