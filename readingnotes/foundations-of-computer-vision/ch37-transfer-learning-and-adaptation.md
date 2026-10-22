---
title: 第 37 章 · 迁移学习与适配
chapter: 37
original: https://visionbook.mit.edu/transfer_learning.html
---

# 第 37 章 · 迁移学习与适配

<div class="src-note">
本页为《Foundations of Computer Vision》第 37 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/transfer_learning.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

Part X「构建学习型视觉系统的挑战」的收官章。第 36 章用「在足够宽的数据分布上训练」来消化分布偏移；本章直面其极限——测试查询千变万化，不可能全部预先见过。迁移学习给出另一条路：把旧任务上学到的先验知识（参数、特征、教师模型、生成模型……）搬过来，加速新问题的学习。它是全书学习主线（第 9-37 章）的终点，也为第 51 章的大规模视觉-语言模型预训练范式做了概念铺垫。

## 本篇要点

- 迁移学习 = 预训练（pretraining）+ 适配（adaptation）两阶段；先验知识的形式可以非常多样，任何计算图组件都可以被预训练后再适配。
- 微调（finetuning）最简单：继承预训练参数继续梯度下降；输入/输出维度不匹配时替换网络首层/尾层，实战常见姿势是「冻结主干 + 换头 + 小学习率」。
- 知识蒸馏（knowledge distillation）让学生模仿教师的软概率输出；软目标比 one-hot 标签携带类间相似性等暗知识，信息量更大。
- 提示（prompting）不改参数、改输入，数学上与对抗样本同构；域适应（domain adaptation）则把输入分布对齐到测试域（或反之），无监督设定下常用 GAN 翻译。
- 生成式数据可以「更多更好」：可无限采样、可插值重组、可编辑隐变量定向造数。
- 序列模型靠隐状态随处理过程不断适配；把训练序列喂给 RNN 就是元学习，即上下文学习（in-context learning）。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 3 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/transfer_learning/pretraining_and_adaptation.webp" alt="图 37.1 迁移学习的两个阶段" loading="lazy" />
<figcaption>图 37.1：迁移学习由两个阶段组成：先在一个任务上预训练模型，再将该模型适配以执行新任务。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这一张图给出全章骨架——后面所有方法都只是在回答「预训练什么、适配什么」两个问题。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/transfer_learning/fig-transfer_learning-finetuning_stages.webp" alt="图 37.2 微调的阶段：替换首尾层" loading="lazy" />
<figcaption>图 37.2：微调的阶段。若输出维度变化，砍掉最后一层并替换为可输出正确维度的新层；输入维度变化时对第一层同样处理，中间层则继承预训练参数。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：微调时最常踩的坑就是输出维度不匹配；这张图给出标准解法——砍尾层、换新头，输入变则同理换首层。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/transfer_learning/KD_diagram.webp" alt="图 37.4 知识蒸馏 vs 从标签监督学习" loading="lazy" />
<figcaption>图 37.4：知识蒸馏与从标签监督学习的对比。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：软目标 vs one-hot 的对比是知识蒸馏的全部精髓：教师把类间相似性编码进概率向量，信息量远超硬标签。</p>

## 本章小结

- 迁移学习把「用先验知识加速新任务学习」形式化；任何计算图组件（参数、输出头、教师、生成模型、乃至损失函数与优化器）都可以被预训练后再适配；一切方法都可用 ERM 统一看待——区别只在于哪些组件在预训练阶段、哪些在适配阶段被优化。它是一个工具箱，而非固定算法集。
- 微调：预训练参数初始化 + 继续梯度下降；结构不兼容（输入/输出维度变化）时替换首层/尾层；可选择性冻结、部分微调、层内低秩更新。
- 从标签学习是「向老师学习」的特例（教师 = 世界）；知识蒸馏让学生拟合教师的软概率分布，软目标携带 one-hot 标签没有的类间结构信息。
- 提示（像素 / token / 文本）通过编辑输入引导模型，与对抗样本数学同构；域适应通过对齐输入分布消除域差，无监督设定下常用 GAN 做非配对翻译。
- 生成式数据提供更多且更好的数据：无限采样、插值重组、隐变量编辑定向造数。
- RNN 等序列模型天然在做适配；把训练序列喂给 RNN 即元学习，其测试时形态就是上下文学习（in-context learning）。

## 复习自检清单

- ☐ 能区分 one-shot / few-shot / zero-shot 三种设定，并解释为什么先验知识能让深度网络摆脱「数据饥渴」
- ☐ 能描述微调的三阶段，并说明输入/输出维度变化时如何改网络结构（替换首/尾层）
- ☐ 能解释为什么要冻结部分模块（防止遗忘先验），以及低秩更新如何省显存并正则化适配幅度
- ☐ 能写出知识蒸馏的目标（学生模仿教师输出的 pmf），并用猫/虎的例子说明软目标为什么比 one-hot 标签信息量大
- ☐ 能写出像素提示的优化目标 $\mathop{\mathrm{arg\,min}}_{\mathbf{p}} \sum_{i=1}^M \mathcal{L}(f(\mathbf{x}^{(i)} + \mathbf{p}), \mathbf{y}^{(i)})$，并说明它与对抗样本的异同
- ☐ 能区分提示与域适应的目标差异（引导性能 vs 让输入更熟悉），并描述无监督域适应的 GAN 翻译流程
- ☐ 能举例说明生成式数据如何比真实数据「更多更好」，以及编辑隐变量如何定向造数
- ☐ 能解释 RNN 为什么边处理序列边适配，以及为什么把训练序列喂给 RNN 可视为元学习 / 上下文学习

---

原文：[Transfer Learning and Adaptation](https://visionbook.mit.edu/transfer_learning.html)
