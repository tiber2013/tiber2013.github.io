---
title: 第 34 章 · 条件生成模型
chapter: 34
original: https://visionbook.mit.edu/conditional_generative_models.html
---

# 第 34 章 · 条件生成模型

<div class="src-note">
本页为《Foundations of Computer Vision》第 34 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/conditional_generative_models.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

本章是第 32、33 章生成模型的延续，把生成模型从「凭空造数据」推进到「解预测问题」——这是生成模型当下最广泛的用途。前半章用图像上色这个例子，一步步说明点预测、逐像素分类为何失败（多模态 + 联合结构）；后半章给出条件 GAN、cVAE、条件自回归、条件扩散的通用改造法，并以 pix2pix / CycleGAN 收束。

## 本篇要点

- 条件生成模型同时建模输出的**多模态分布**与**联合结构**，因此是视觉中结构化预测的通用工具。
- 最小二乘回归 = 点预测（多模态下必然「和稀泥」）；逐像素分类 = 条件独立假设，会产生颜色跳变这类伪影。
- 任何无条件生成模型都可「条件化」：给生成器/判别器/去噪函数补上条件输入 $\mathbf{x}$。
- cVAE 的关键约束：$\mathbf{z}$ 与条件 $\mathbf{x}$ 独立，只编码 $\mathbf{x}$ 解释不了的那部分信息（如台球速度）。
- pix2pix = 条件 GAN + U-Net + PatchGAN + $L_1$；判别器可看作「学出来的损失函数」。
- 非配对翻译靠「集合级监督」可行，但存在大量等价映射的对称性；cycle-consistency 损失是打破对称的关键归纳偏置。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 6 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/conditional_generative_models/tshirts.webp" alt="图 34.1 不同种类的预测分布" loading="lazy" />
<figcaption>图 34.1：给定一件灰色 T 恤的观测，不同预测模型给出的预测分布。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这张图把「点预测 / 高斯 / 混合高斯 / 类别分布」四个档次的表达力并排摆开，是理解「为什么需要条件生成模型」的最佳入口。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/conditional_generative_models/cgen_tshirt_color_inconsistency.webp" alt="图 34.3 颜色跳变伪影" loading="lazy" />
<figcaption>图 34.3：颜色跳变可以从平滑的底层预测分布中产生——在其之上做的是独立选择。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是「独立假设」最典型的视觉故障——模型并不笨，是硬阈值 + 逐像素独立把平滑的 49%/51% 抖动放大成了块状跳变；工程上看到这类伪影就要想到联合建模。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/conditional_generative_models/cVAE_ball_bouncing_example.webp" alt="图 34.4 台球示例：条件 VAE 的潜在变量" loading="lazy" />
<figcaption>图 34.4：一个黄色小球在平面上运动的场景。观测 x 是一帧静态画面。从该观测我们知道下一帧 y 中球的颜色和大致位置，但不知道它会向哪个方向运动，因为球的速度未被观测。因此速度是潜在变量，cVAE 目标的一个解就是把球的速度编码进模型的潜在变量 z 中，如图所示。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这个例子是理解 cVAE 设计的钥匙——条件 x 已经解释的信息不要让 z 重复编码，z 专管「给定 x 后仍缺失的自由度」。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/conditional_generative_models/pix2pix_facades_arch.webp" alt="图 34.7 pix2pix 架构" loading="lazy" />
<figcaption>图 34.7：应用于把立面布局图翻译成立面照片的 pix2pix [4] 模型。模型在 CMP Facades Database [5] 上训练，此处展示的输入与真值图像来自该数据集。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是条件 GAN 落地的标准范式——U-Net 生成器 + PatchGAN 判别器 + L_1 稳定项，工程上做任何「有监督的图像到图像」任务都值得先抄这个结构。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/conditional_generative_models/patchgan_patch_size_variations.webp" alt="图 34.9 PatchGAN 感受野的影响" loading="lazy" />
<figcaption>图 34.9：改变卷积判别器的感受野（区块大小）会影响判别器强制执行的结构种类。这些结果来自 [4]。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：调 patch 大小是图像到图像任务最实用的旋钮之一——小了出模糊，大了出周期约等于 patch 大小的拼贴伪影；这张图解释了伪影来源，选型时心里有底。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/conditional_generative_models/cyclegan_schematic.webp" alt="图 34.12 CycleGAN 示意图" loading="lazy" />
<figcaption>图 34.12：CycleGAN 示意图。图改编自 [6]。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：非配对设定下「集合级监督 + 循环一致性」是核心套路；把这张图的四个损失项与架构对应起来，就能复现整个方法。</p>

## 本章小结

- 生成模型的第三种用途是解预测问题；条件生成模型同时建模输出的多模态分布与联合结构，是视觉结构化预测的通用工具。
- 点预测（最小二乘）在多模态分布下必然失败（输出均值这种「零概率解」）；逐像素分类带来量化误差与 $k^{2N}$ 的类别爆炸。
- 逐像素独立预测 = 非结构化预测（由 Hammersley–Clifford 定理，$L_2$ 损失隐含条件独立），会产生颜色跳变伪影。
- 条件化改造四件套：cGAN 给生成器/判别器加 $\mathbf{x}$；cVAE 约束 $\mathbf{z}$ 与 $\mathbf{x}$ 独立、编码器/解码器都吃 $\mathbf{x}$；自回归加前缀；扩散把 $\mathbf{x}$ 拼进去噪函数。
- CRF 是「慢思考」推断，条件生成模型是「快思考」的摊销推断。
- pix2pix：U-Net 生成器 + PatchGAN 判别器 + $L_1$ 稳定项；判别器 = 学出来的损失函数；patch 大小直接控制能强制执行的统计结构尺度。
- 非配对翻译：普通 GAN + 集合级监督可行，但存在置换对称性；cycle-consistency 损失鼓励一对一映射来打破对称（仍不彻底）。

## 复习自检清单

- ☐ 能解释为什么「彩色→灰度」的多对一性使上色问题本质多模态，以及最小二乘回归为何输出零概率解
- ☐ 能写出逐像素独立预测的因子分解式，并用 Hammersley–Clifford 定理推出 $L_2$ 回归隐含条件独立假设
- ☐ 能解释颜色跳变伪影的成因（平滑的 49%/51% 边缘概率 + 硬阈值 + 独立决策）
- ☐ 能写出 cVAE 的似然模型 $p_{\theta}(\mathbf{y}|\mathbf{x})=\int p_{\theta}(\mathbf{y}|\mathbf{z},\mathbf{x})p_{\mathbf{z}}(\mathbf{z})d\mathbf{z}$，并用台球例子说明 $\mathbf{z}$ 该编码什么
- ☐ 能写出条件 GAN 的目标函数，解释判别器为何改判「输入–输出对」真假
- ☐ 能把自回归模型条件化（拼接前缀）和把扩散模型条件化（拼进去噪函数输入），并说明它们共同的「监督序列化」思想
- ☐ 能解释「判别器是学出来的损失函数」，以及 PatchGAN 的 patch 大小如何影响输出结构（模糊 vs 拼贴伪影）
- ☐ 能写出 CycleGAN 的循环一致性损失，解释为什么非配对 GAN 存在置换对称性、循环一致性如何部分打破它

---

原文：[Conditional Generative Models](https://visionbook.mit.edu/conditional_generative_models.html)
