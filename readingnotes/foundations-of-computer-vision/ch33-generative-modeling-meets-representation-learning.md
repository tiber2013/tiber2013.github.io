---
title: 第 33 章 · 生成建模与表示学习的交汇
chapter: 33
original: https://visionbook.mit.edu/generative_modeling_and_rep_learning.html
---

# 第 33 章 · 生成建模与表示学习的交汇

<div class="src-note">
本页为《Foundations of Computer Vision》第 33 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/generative_modeling_and_rep_learning.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

本章是第 32 章「生成模型」的延续，也是第 30 章「表示学习」的镜像视角。核心论点是：生成建模与表示学习本质上是同一件事的两个方向——前者把简单先验（噪声）映射到数据，后者把数据映射到简单表示。全章以变分自编码器（VAE）为载体，把这一对偶关系落到可优化的数学形式上，并回答「潜在变量是不是好的表示」。

## 本篇要点

- 生成模型是表示学习的逆问题：$g:\mathbf{z}\to\mathbf{x}$ 与 $f:\mathbf{x}\to\mathbf{z}$ 互为逆映射，训练目标是 $g\approx f^{-1}$。
- 潜在变量 $\mathbf{z}$ 就是数据的表示；在无条件生成模型中，它是数据的**完整**表示。
- VAE 的假设空间 = **无限个高斯分布的混合**（$k\to\infty$ 的高斯混合模型），均值和方差由解码器网络 $g_\theta$ 参数化。
- VAE 优化三步技巧：蒙特卡洛估计 → 重要性采样 → 变分推断，最终落到同一个目标 ELBO。
- ELBO = 重建似然项 $-$ KL 正则项；两项的张力把自编码器杂乱无章的隐空间「挤压」成紧致可采样的高斯状隐空间。
- VAE 隐维度在 $\mathcal{N}(0,1)$ 先验下趋向**解耦（disentanglement）**，但存在**不可辨识性（nonidentifiability）**：学到的未必是真实的因果机制。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 6 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/generative_modeling_and_representation_learning/rep_gen_schematic.webp" alt="图 33.1 表示学习与生成建模的关系" loading="lazy" />
<figcaption>图 33.1：表示学习与生成建模之间的关系。这里我们把一侧标为「数据」，另一侧标为「嵌入」；但这两者之间的确切区别究竟是什么？为什么一张 RGB 图像是「数据」而一个 100 维神经激活向量是「嵌入」？这是一个留给读者思考的问题，没有标准答案。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这一张图是本章的总纲——把「数据」和「嵌入」的相对性看透，后面所有模型（VAE、GAN）都只是在这条数轴的两个方向上做功。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/generative_modeling_and_representation_learning/gmm_vs_vae.webp" alt="图 33.5 从有限高斯混合到无限高斯混合" loading="lazy" />
<figcaption>图 33.5：从有限高斯混合到无限高斯混合。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是理解 VAE 的最重要直觉图——VAE 不是什么新东西，它就是把经典高斯混合的「一个个分量」换成「解码器网络参数化的连续分量」，从而一步跨到无限混合。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/generative_modeling_and_representation_learning/VAE_encoder.webp" alt="图 33.9 VAE 的编码器" loading="lazy" />
<figcaption>图 33.9：VAE 的编码器 f_ψ 把 p(Z | x) 建模为由 f_ψ(x) 参数化的高斯。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这张图把「编码器」和「重要性采样分布」画上了等号——这是 VAE 与普通自编码器最本质的分界点。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/generative_modeling_and_representation_learning/VAE_as_autoencoder.webp" alt="图 33.10 VAE 就是自编码器" loading="lazy" />
<figcaption>图 33.10：为评估 VAE 赋予数据点 x 的似然，先把 x 编码进 z 空间、再解码回来并计算重建误差。这对应于近似似然函数的一个重要性样本。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：把「编码—采样—解码—算重建误差」四步与自编码器画等号，是记住 VAE 训练流程的最短路径；每步反向传播都对应这里的一个环节。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/generative_modeling_and_representation_learning/vae_rivers_latent_walk.webp" alt="图 33.14 潜在空间行走展示解耦" loading="lazy" />
<figcaption>图 33.14：沿潜在空间两个轴行走生成的图像显示出两个不同属性的变化，说明该模型在一定程度上是解耦的。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是全书展示「解耦」最直观的例子——沿一个轴只变草色、沿另一个轴只变河曲，一眼看懂什么叫解耦表示。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/generative_modeling_and_representation_learning/biggan_latent_walk.webp" alt="图 33.15 BigGAN 的潜在空间行走" loading="lazy" />
<figcaption>图 33.15：在 BigGAN [4] 的两个正交方向上游走。</figcaption>
</figure>

## 本章小结

- 生成建模与表示学习互为逆问题：$g:\mathbf{z}\to\mathbf{x}$ 与 $f:\mathbf{x}\to\mathbf{z}$，训练目标是 $g\approx f^{-1}$。
- 潜在变量即表示；无条件生成模型的潜在变量是数据的完整表示。
- VAE 的假设空间是无限高斯混合：$p_\theta(\mathbf{x})=\int \mathcal{N}(\mathbf{x};g^\mu_\theta(\mathbf{z}),g^\Sigma_\theta(\mathbf{z}))\mathcal{N}(\mathbf{z};\mathbf{0},\mathbf{I})d\mathbf{z}$，与边缘似然 (33.1) 同形。
- VAE 优化的三个技巧：蒙特卡洛近似积分 → 重要性采样（最优 $q^*=p_\theta(Z|\mathbf{x})$）→ 变分推断（用高斯 $q_\psi$ 近似后验），最终得到 ELBO。
- ELBO（式 33.11）= 重建似然项 $-$ KL 正则项；KL 项把隐空间压向 $\mathcal{N}(0,1)$，重建项要求铺开保留信息，二者张力塑造紧致隐空间。
- 与自编码器的区别仅三点：采样随机 $z$、重建误差按预测方差缩放、加 KL 损失；采样靠重参数化技巧 $z=\mu_z+\epsilon\sigma_z$ 反向传播。
- 先验 $\mathcal{N}(0,1)$ 鼓励解耦；但 VAE 未必恢复真实因果机制（不可辨识性）。
- GAN 生成器与 VAE 解码器结构同构，同样能学到解耦潜在变量（BigGAN 实验）。

## 复习自检清单

- ☐ 能画出「数据 ↔ 嵌入」的双向映射图，并解释为什么说生成建模与表示学习互逆
- ☐ 能写出边缘似然 $p_{\theta}(\mathbf{x})=\int p_{\theta}(\mathbf{x}|\mathbf{z})p_{\mathbf{z}}(\mathbf{z})d\mathbf{z}$（式 33.1），说明为什么它难以直接计算
- ☐ 能解释 VAE 为什么是「无限高斯混合」，它与经典高斯混合的区别是什么
- ☐ 能复述 VAE 优化的三个技巧，并说明最优重要性采样分布为什么是 $p_{\theta}(Z|\mathbf{x})$
- ☐ 能用 Jensen 不等式从 $J_p$ 推出 ELBO $J$（式 33.10），并写出 ELBO 的两项形式（式 33.11）
- ☐ 能写出 1D 情形下 KL 项闭式解 $\frac{1}{2}(\mu_z^2+\sigma_z^2-\log\sigma_z^2-1)$ 和重建似然项
- ☐ 能解释重参数化技巧 $z=\mu_z+\epsilon\sigma_z$ 解决什么问题
- ☐ 能说明 KL 项与重建项的张力如何把隐空间从「杂乱无章」变成「紧致可采样」
- ☐ 能解释「解耦」与「不可辨识性」的含义，并用河流例子说明 VAE 未必恢复真实因果机制

---

原文：[Generative Modeling Meets Representation Learning](https://visionbook.mit.edu/generative_modeling_and_rep_learning.html)
