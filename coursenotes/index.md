---
title: 课程笔记
description: 整门课程的视频笔记，按讲次整理核心结论、板书关键帧与要点回顾
---

# 课程笔记

论文笔记解决「这篇讲了什么」，读书笔记解决「这本书的骨架是什么」，课程笔记解决**一门课一学期讲完了什么**。

视频课的麻烦在于时间：一门课动辄十几讲、每讲七八十分钟，连着看会陷入「听懂了但记不住」，跳着看又会丢掉前后依赖。这里把每门课拆成**一讲一页**：核心结论先给判断，正文按视频里的时间戳分节，板书与实验图作为关键帧插在对应位置，末尾留一段要点回顾。想快速回血就只读核心结论和回顾，要补课再顺着时间戳回原视频。

也有一门课没有录像可跟。**CS231A** 公开的原始材料只有课件与 Course Notes，所以那一批笔记改成跟着官方阅读材料走：10 篇 Course Notes 逐节全译，公式保留 LaTeX 并沿用原文编号，插图直接从 PDF 里取；课件讲到而 Course Notes 没写的主题，按课件页面加原始论文另写。

<div class="book-grid">

<a class="book-card" href="./cs231n-deep-learning-for-cv-sp25/">
  <h3>Stanford CS231n 计算机视觉</h3>
  <div class="book-sub">Deep Learning for Computer Vision · Spring 2025 视频 / Spring 2026 课件</div>
  <p>从图像分类、线性分类器与反向传播，到 CNN、Transformer、检测分割、视频、自监督、生成模型、3D 与视觉语言；对照 2026 课件标注新增内容。</p>
  <div class="book-tags">
    <span>18 讲</span>
    <span>约 21 小时</span>
    <span>242 张关键帧</span>
  </div>
</a>

<a class="book-card" href="./cs231a-computer-vision-3d-sp25/">
  <h3>Stanford CS231A 计算机视觉：三维感知与重建</h3>
  <div class="book-sub">From 3D Perception to 3D Reconstruction and Beyond · Spring 2025 · Savarese & Bohg</div>
  <p>多视图几何那一支：相机模型与标定、单视图计量、对极几何、三角化与 SfM、主动与体积立体、拟合与匹配、表示学习、单目深度与特征跟踪、光流与场景流、最优估计，最后落到 NeRF 与 3D Gaussian Splatting。这门课**没有公开录像**，10 篇官方 Course Notes 做逐节中文全译（公式与编号对齐原文 PDF），未覆盖的 SfM、NeRF、3DGS 与客座讲座按课件加原始论文另写。</p>
  <div class="book-tags">
    <span>10 篇全译 + 4 篇笔记</span>
    <span>无录像</span>
    <span>83 张课件插图</span>
  </div>
</a>

<a class="book-card" href="./cs336-language-modeling-sp26/">
  <h3>Stanford CS336 语言模型从零构建</h3>
  <div class="book-sub">Language Modeling from Scratch · Spring 2026</div>
  <p>把语言模型拆到硬件层：分词与架构自己写，GPU/TPU 与 kernel 自己调，多卡并行与缩放定律自己拟合，再补齐数据、评估、后训练与对齐。</p>
  <div class="book-tags">
    <span>17 讲 + 1 客座</span>
    <span>约 24 小时</span>
    <span>414 张关键帧</span>
  </div>
</a>

<a class="book-card" href="./mit-dcai-iap-2024/">
  <h3>MIT 数据中心 AI 导论（IAP 2024）</h3>
  <div class="book-sub">Introduction to Data-Centric AI · IAP 2024 · MIT CSAIL</div>
  <p>把着力点从「调模型」移到「改数据」：PU 学习与置信学习自动找出错标样本，再处理类别不平衡、离群点、分布偏移、数据集采集与多标注者共识，最后落到面向大语言模型的数据策展。配图由每章转写内容与画面语义的相关度决定，数量随内容密度变化。</p>
  <div class="book-tags">
    <span>7 讲</span>
    <span>约 6 小时</span>
    <span>190 张关键帧</span>
  </div>
</a>

<a class="book-card" href="./cs294-158-deep-unsupervised-sp24/">
  <h3>CS294-158 深度无监督学习</h3>
  <div class="book-sub">UC Berkeley · Spring 2024 · Pieter Abbeel</div>
  <p>生成模型与自监督表示两条线并讲：自回归、流模型、VAE、GAN、扩散管「怎么建模分布」，自监督、半监督、多模态与并行化管「怎么把表示训好、训大」。</p>
  <div class="book-tags">
    <span>14 讲</span>
    <span>约 24 小时</span>
    <span>260 张关键帧</span>
  </div>
</a>

<a class="book-card" href="./eth-robot-learning-2026/">
  <h3>ETH Zürich 机器人学习</h3>
  <div class="book-sub">Robot Learning 2026 · Oier Mees</div>
  <p>从 MDP 与模仿学习走到价值方法与策略梯度，再进入生成式策略、Transformer 序列建模与世界模型，最后落在通用策略与具身推理——机器人该怎么学。</p>
  <div class="book-tags">
    <span>11 讲</span>
    <span>约 10 小时</span>
    <span>313 张关键帧</span>
  </div>
</a>

<a class="book-card" href="./topics/">
  <h3>专题视频笔记</h3>
  <div class="book-sub">单个技术视频的拆解</div>
  <p>不属于任何一门课、但值得单独存档的视频：DINOv3、NLP 简史、从 LLM 到 Agent Skill 的概念链路、RLHF。</p>
  <div class="book-tags">
    <span>4 篇</span>
    <span>约 1 小时</span>
    <span>71 张关键帧</span>
  </div>
</a>

</div>

全部笔记合计 **86 篇、约 87 小时视频课程、1573 张插图**。其中 CS231A 没有录像，那 14 篇是官方 Course Notes 的逐节全译与课件笔记，不计入视频时长。

## 一页笔记长什么样

- **来源行**：课程、讲次、讲者、视频时长与原视频链接。所有结论都能在视频对应时间点找到。CS231A 的来源行换成课件文件名、Course Notes 编号与作者署名，没有时长与时间戳。
- **核心结论**：三到五句话给出这一讲的判断——它解决什么问题、用什么方法、代价是什么。
- **带时间戳的分节**：标题后的 `[12:34 - 18:02]` 就是视频里的位置，卡住了直接跳过去看。翻译类笔记的分节则与原文标题层级一一对应。
- **关键帧**：从视频中按镜头切换抽取的板书、架构图与实验图，紧跟在对应段落后面，配一句说明。
- **要点回顾**：这一讲的清单式收束，复习时只看这一节也能过一遍。

## 关于这些笔记

- **AI 辅助整理**：转录、抽帧、归纳、排版由本地流水线完成（视觉理解模型对关键帧提问后写作），人工只做了标题、结构与少量校对。**表述不准的地方以原视频为准**，每页都附了链接。
- **插图版权归原作者与课程方所有**：视频课的关键帧是视频截图，CS231A 的插图取自官方 Course Notes 与课件 PDF，仅作个人学习的非商业引用；如不适宜公开，请通知我撤下对应图片。
- **不做逐句转录**：这里只保留结构与判断，公式与推导的完整过程请回原视频或课件。课程官方课件、作业与考试信息以各校页面为准。
- **建议用法**：先读核心结论建立骨架 → 对着分节扫一遍关键帧 → 用要点回顾自测 → 卡住的地方按时间戳回看视频。

::: tip 和读书笔记的分工
读书笔记跟着书走（书 → 部分 → 章），课程笔记跟着课走（课程 → 讲次）。同一主题往往两边都有：例如 Transformer 在《Foundations of Computer Vision》第 26 章讲原理，在 CS336 第 3 讲讲怎么写出来，在 ETH 第 7 讲讲怎么用在机器人上。
:::
