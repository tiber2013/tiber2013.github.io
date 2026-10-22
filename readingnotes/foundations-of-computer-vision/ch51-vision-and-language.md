---
title: 第 51 章 · 视觉与语言
chapter: 51
original: https://visionbook.mit.edu/VLMs.html
---

# 第 51 章 · 视觉与语言

<div class="src-note">
本页为《Foundations of Computer Vision》第 51 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/VLMs.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

本章把前面积累的表示学习（第 30 章）、transformer（第 26 章）、图像域翻译与生成模型（第 34 章）全部组装起来，回答两个方向的问题：语言如何监督并增强视觉系统，视觉系统又如何支撑语言系统。它是全书多模态主线的收口，也为后续一切「大视觉-语言模型」的讨论建立最小可用框架。

## 本篇要点

- 文本进 transformer 的前提是数字化：词级/字符级 one-hot，或字节对编码（BPE）子词；one-hot 再经投影矩阵 $\mathbf{W}_{\texttt{tokenize}}$ 变成连续 token，输出侧靠自回归 softmax 逐条生成。
- CLIP 用「图像—标题」共现对做对比学习：双编码器 + 对称 InfoNCE 损失，批内 $N\times N$ 配对矩阵中拉正压负；输出做 $L_2$ 归一化落到单位超球面上。
- 训练好的 CLIP 把图文嵌进同一空间：匹配的图文夹角小；由此可**零样本**构造任意新类别分类器，无需训练任何分类头。
- 跨模态翻译的通用范式是「X 领域编码器 + Y 领域解码器」的函数复合 $g\circ f$，再用配对数据微调，或在潜空间训练一个翻译函数 $h$ 对齐两端。
- 文生图 = 条件生成模型（cVAE / 条件扩散），核心设计问题是文本嵌入的注入位置与方式；图生文 = 图像编码器 + 自回归文本解码器，关键是交叉注意力（cross-attention）层。
- 文本本身就是一种视觉表示：低维、抽象、与语言模型无缝衔接，使「用语言推理图像内容」成为可能。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 4 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/vision_and_language/clip_training_fig.webp" alt="图 51.2 CLIP 训练流程" loading="lazy" />
<figcaption>图 51.2：对一个包含 6 个「语言-图像」训练样本的批次进行 CLIP 训练。绿框是正样本对（要增大其值），即 6 个训练配对；红框是负样本对（要减小其值），即批次内其他所有可能的配对。（参照 Radford 等 2021 年论文的图 1）</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：一张图讲透 CLIP 的全部训练机制——批内配对矩阵、正负样本的划分、对称损失。不懂这张图，就没法理解对比学习在图文模态上的落地方式。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/vision_and_language/clip_shared_embedding.webp" alt="图 51.4 CLIP 联合嵌入空间" loading="lazy" />
<figcaption>图 51.4：CLIP 联合嵌入（joint embedding）。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是 CLIP 全部魔力的落点——图文被嵌到同一个超球面上，从此「用文本描述当分类器」「跨模态检索」都变成夹角比较问题。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/vision_and_language/clip_inference_fig.webp" alt="图 51.5 CLIP 零样本分类" loading="lazy" />
<figcaption>图 51.5：用训练好的 CLIP 构建「条纹红圆」（类别 a）对「波点绿方块」（类别 b）分类器。CLIP 模型参数冻结、不更新；分类器的构造方式是：嵌入描述两个类别的文本，看查询图像的嵌入离哪个更近。（参照 CLIP 论文图 1）</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是「零样本分类」的标准做法——不训练任何分类头，仅靠文本模板就得到新类别分类器，也是后来各种开放词表视觉任务的思想源头。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/vision_and_language/im2text.webp" alt="图 51.11 图生文架构与交叉注意力" loading="lazy" />
<figcaption>图 51.11：用带交叉注意力的小型 transformer 做图生文。灰色边是交叉注意力边。该架构有许多变体：例如文本编码器的每一层都可以对输出的图像 token 做交叉注意力；或者，图像编码器可以把它所有的输入 token 压缩成一个代表整张照片的输出 token，再由文本解码器对这个 token 做交叉注意力——这种做法可以节省内存与计算时间。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是「图像编码器 + 文本解码器」拼接范式的主流程图，其中交叉注意力层是图像 token 进入文本模型的通道，后来绝大多数 VLM 都沿用这一思路。</p>

## 本章小结

- 文本进 transformer 的前提是数字化：词级/字符级 one-hot 或字节对编码（BPE）子词；one-hot 经投影矩阵变成连续 token，输出侧靠自回归 softmax 逐条生成。
- CLIP = 图像编码器 + 文本编码器 + 对称 InfoNCE 对比损失，批内 $N\times N$ 配对矩阵中拉正压负；输出做 $L_2$ 归一化落到单位超球面。
- 训练好的 CLIP 把图文嵌进同一空间：匹配的图文夹角小；由此可零样本构造任意新类别分类器（嵌入类别文本、比较与查询图像的接近程度），无需任何训练。
- 跨模态翻译的通用范式是「X 编码器 + Y 领域解码器」复合，再靠配对数据微调或潜空间翻译函数 $h$ 对齐两端。
- 文生图 = 条件生成（cVAE / 条件扩散），核心设计问题是文本嵌入的注入位置与方式；扩散模型与 cVAE 结构同构，区别在于多步去噪链。
- 图生文 = 图像编码器 + 自回归文本解码器，交叉注意力层让文本 token 查询图像 token；交叉与自注意力输出按可学习权重融合。
- 文本是一种独特的视觉表示：低维、抽象、与语言模型无缝衔接，使图像内容可被语言模型推理；VQA 则是本章所有组件的自然综合。

## 复习自检清单

- ☐ 能解释为什么需要 BPE：词级 one-hot 的 $K$ 太大、字符级序列太长，BPE 如何折中
- ☐ 能写出 CLIP 的对称 InfoNCE 损失，并说明 $L_2$ 归一化为什么能防止分母被负样本对主导
- ☐ 能描述 CLIP 训练一个批次的完整流程：配对矩阵中哪些是正样本对、哪些是负样本对、各自如何影响损失
- ☐ 能用三步流程（嵌类别文本、嵌查询图像、比接近程度）说明 CLIP 零样本分类的原理，并解释为什么它能泛化到训练时没见过的类别
- ☐ 能写出「X 编码器 + Y 解码器」的翻译范式，并说出对齐编码器与解码器的两种策略
- ☐ 能对比 cVAE 与条件扩散模型做文生图的异同（输入相同；单步前向 vs 多步去噪链）
- ☐ 能写出交叉注意力层的公式，说明 $\mathbf{Q}$、$\mathbf{K}$、$\mathbf{V}$ 分别来自哪一侧，以及它与自注意力输出如何融合
- ☐ 能列出构造一个 VQA 系统需要哪三个组件、训练数据是什么形式

---

原文：[Vision and Language](https://visionbook.mit.edu/VLMs.html)
