---
title: "第 17 讲 · 对齐与多模态"
description: "核心结论"
---

# 第 17 讲 · 对齐与多模态

> **Stanford CS336 语言模型从零构建 · L17** ｜ 讲者 Stanford Online ｜ 78 分钟 ｜ [原视频](https://www.youtube.com/watch?v=26FtD08ZpOU)

> **v2 详写版** · 按 20 分钟分段详写，配图覆盖全片；文末附「对照参考笔记补遗（Spring 2025 版）」，记录与 2025 版讲义笔记的核对结果。

## 核心结论

本讲把视野从只能做 text ⇒ text 的语言模型推向能接受、生成任意模态组合的 omni model，却很快收窄到图像、且几乎只处理输入侧：Transformer 仍是最可靠的主干、又只「说 token」，根本难题便是把非文本数据转成承载语义单元的离散或连续词元，而理解与生成对表示粒度的要求并不对称。全讲沿「先学表示、再造 VLM」展开：CLIP 用 4 亿图文对在 N×N 相似度矩阵上做对称对比损失，SigLIP 换成可逐对累加、可跨设备并行的 sigmoid 二分类；此后 LLaVA、LLaVA OneVision 与 Qwen-VL 到 Qwen3-VL 共用「视觉编码器 + 投影器 + 语言模型」模板，靠 GPT-4 合成指令数据、AnyRes 动态分辨率、三类输入的 token 预算平衡、M-RoPE 与 DeepStack，把能力扩展到视频、长上下文与细粒度 OCR。作为反例，Chameleon 用 VQ-VAE 把一切压成离散 token、令图文真正交错，训练退化为标准自回归，虽优雅却因图像 token 高熵导致范数增长与 loss 不稳、离散化又丢失细节。讲者最终给出带猜测性质的判断：不存在通用编码器，分类式理解可压缩成高层语义小向量，OCR 与生成却必须保留高频细节，故较稳妥的是连续编码器负责理解、Transformer 作主干、扩散模型负责生成，并需权衡视频等低信息密度模态与文本的配比。

## 从语言模型到 omni model：本讲的动机与终极目标 [00:05 - 01:55]

讲者在开场先交代了一次课程安排上的调整：原计划这一讲要花更多篇幅讲 reinforcement learning（强化学习），但因为整个学期只剩这一讲可以自由支配，他判断如果完全不谈 multimodality（多模态），这门课会显得不完整——当下几乎所有主流前沿模型都具备多模态能力，这一主题已经 pervasive（无处不在）。于是本讲被定位为一次 overview of multimodal models（多模态模型概览），他同时坦承这个题目本身足够开一整门课，一讲之内只能尽力而为。这决定了本段的叙述方式：不追求把每个模型都推导到底，而是先给出拼图的若干关键块，让听众知道后续每一块该往哪里安放。

在此之前，整门课程 exclusively focused on language models（只聚焦语言模型）。讲者强调 language models are already pretty general（语言模型已经相当通用）：它提供的是从任意一段文本到任意一段文本的能力，可形式化地记作 $\text{LM}:\ \mathcal{T}\Rightarrow\mathcal{T}$，幻灯片上则直接写成 `text ⇒ text`。这个"任意文本"覆盖面比直觉更广——自然语言、代码（code）、诗歌（poetry），甚至 DNA 序列这种并非自然语言的字符串，都能塞进同一个 text-to-text 框架里被处理。值得注意的是，这里的幻灯片并不是普通演示文稿，而是一个带行号、语法高亮和横向滚动条的 Python 源文件 `lecture_17.py`：开头两行是 `from edtrace import text, image, link` 与 `from lecture_util import article_link, post_link`，紧接着 `def main():` 下面是大标题 `Lecture 17: multimodal models`，被黄色高亮的当前行正是 `So far: language models`。把课件写成可执行代码、用高亮表示"讲到哪一行"，是 CS336 一贯的风格，也预告了后面每个模型都会以一次函数调用的形式登场。

```python
# lecture_17.py（片段，按幻灯片转录）
from edtrace import text, image, link
from lecture_util import article_link, post_link

def main():
    # Lecture 17: multimodal models
    # So far: language models          <- 当前高亮行
    text ⇒ text
    # The world is multimodal:
```

然而真实世界是多模态的：画面上并排列出 Text、Images、Audio、Video 四类卡片。Text 卡片里是 "The quick brown fox jumps over the lazy dog." 与 "Artificial intelligence is" 两行示例；Images 卡片里是山脉图与一只金毛犬；Audio 卡片带波形、播放进度 `0:00 - 0:10`、音符图标，以及斜体字幕 "The sounds of ocean waves crashing on the shore."；Video 卡片是一段山水视频，进度显示 `0:02 / 0:15`，右下角有音量、设置和全屏图标。讲者把终极目标称为 people call an omnimodel（人们所说的全能模型）：ability to take any combination of any of these modalities and output any combination（接受任意模态组合、输出任意模态组合）。幻灯片把它拆成两条并列目标——`Input any combination of modalities (understanding)` 与 `Output any combination of modalities (generation)`，明确把"理解"和"生成"标成两个方向，这个区分在后面会谈到的表示粒度问题里会再次出现。

![Audio 与 Video 模态卡片，以及 omni model 终极目标：输入任意模态组合（理解）、输出任意模态组合（生成）](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0002_00-01-49.jpg)

他给的应用例子很具体：同时给一张 image 和一段 video 并就它们提问；生成图像；把 audio 转成 image。但紧接着就收窄了范围——本讲不会 build up to a whole omni model（搭建出完整的全能模型），只给出若干 pieces，这些部件至少能让人 ingest a little bit of images（把图像稍微"吃"进模型一点），而今天 mostly focus on 图像。理解这一点很关键：omni model 是 north star（北极星式的方向），不是本讲要交付的产物；把后面要讲的 CLIP 误当成"已经能生成图像的全能模型"是最常见的错位。此外，两条目标被并列写成 understanding 与 generation，已经埋下一个伏笔：本讲大部分内容只处理输入侧，也就是把非文本数据送进模型，而输出侧基本留白，这一非对称性会一路贯穿到本段结尾。

## Transformer 只说 token：非文本模态的编码难题 [01:55 - 04:19]

那么该如何造出这样的东西？讲者给了一个他自己也承认"有点奇怪"的动机：现实是 transformers work really well（Transformer 工作得非常好），尽管人们尽了最大努力去尝试别的架构，跨所有这些模态、在 scale（规模化）条件下，它依然是我们手里最好的东西，所以只能想办法用它。而 Transformer 本来是为文本设计的，于是它有一个根深蒂固的性质——they speak tokens（它们只说词元）：输入一串 tokens，输出一串 tokens。讲者在这里刻意扩展了 token 的概念：它不再只是文本里那种 discrete token（离散词元），也可以 potentially continuous tokens（连续词元），后者可以理解为 token 的 embedding（嵌入向量）。真正的约束落在语义上：a token should represent some sort of semantic unit of information（一个词元应当代表某种语义信息单元）。换句话说，能否被 Transformer 消化，不取决于它是整数还是向量，而取决于它是否携带了可组合的语义。

对比之下差别就很清楚了：在自然语言里 tokens 是 subwords（子词），这些子词 somewhat meaningful（多少有点意义）；而 a pixel（单个像素）is certainly not meaningful by itself（本身显然没有意义）。所以必须把包括 audio、images 在内的一切，转成离散或连续 token。讲者特别提醒，这件事对文本其实也已经做过一次：第一讲讲过 tokenization（分词），大家还动手实现过 BPE tokenizer，它"还行"，虽然可以奢望更好的 tokenizer、甚至干脆不需要 tokenizer 的方案，但 it's not the worst thing in the world（不算世上最糟的东西）。对非文本模态就得 scratch our heads a lot more（更挠头）：要找到那个"等价于 BPE tokenizer"的东西，它能把一张图像变成 Transformer 能消化的序列。这里有一个容易踩的坑：很多人把 token 等同于"词表里的整数编号"，于是觉得连续向量不算 token；但本讲从一开始就把连续嵌入也算作 token，只是要求它承载语义单元。

幻灯片以注释形式给出了整门课这一段的调用路线，这些函数名在转写里并没有被逐个念出来：

```python
# Encoding images
clip()
siglip()

# Injecting image encodings into LLMs
llava()
llava_onevision()
```

也就是说，本讲从 `clip()` 这个入口开始，先解决"把图像编码成语义表示"，后面才谈怎么把这些编码注入语言模型。与此呼应，`Where we are today:` 下面逐条列出了推理链：`Transformers work really well. So we gotta use them.`；`Transformers speak tokens (discrete or continuous), where a token represents some ~semantic unit of information.`；`Therefore, we must convert everything into tokens.`；`Note: we had to do this with text (recall the tokenization lecture).`；`For non-text modalities, this is more challenging...`。其中 `~semantic unit of information` 的波浪号与 `discrete or continuous` 的括注，都是画面上才有的精确措辞，把口头讲述里含糊的"某种语义单元"落到了文字。

![Where we are today 的推理链与两个待解问题：如何输入非文本数据、如何输出非文本数据](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0003_00-03-02.jpg)

由此自然收敛成两个要回答的问题，幻灯片把它们编号写死：`1. How do we input non-text data (e.g., understand images)?` 与 `2. How do we output non-text data (e.g., generate audio)?`——注意第二个例子用的是 generate audio，比口头举的"生成图像"更宽。讲者明说 this lecture will mostly focus on question number one（本讲主要聚焦第一个问题）。这与上一章的收窄是一致的：输入侧是本段乃至本讲的主线，输出侧只被登记为待解问题。易错点也在这里：不要因为后面 CLIP 的图里出现了 "A photo of a dog." 就以为它在生成文本或图像，那些文字是预先写好的标签模板；也不要把"连续 token"理解成可以不经任何转换地把原始像素直接喂进去——单个 pixel 不成其为语义单元，中间必须有一层编码，而这层编码正是下一章 CLIP 要补上的东西。

## 回到 2021：CLIP 的对比预训练目标与伪代码 [04:19 - 08:15]

讲者把时钟拨回 2021 年，理由是 CLIP 是 a lot of the foundation of modern VLMs（现代视觉语言模型很大一部分基础），值得讲得细一些。幻灯片这一页以 `def clip():` 起头，标题是 `CLIP (Contrastive Language-Image Pretraining) [Radford+ 2021]`，`Context:` 下只写了一条：`Computer vision models were trained on annotated images.`，黄色高亮的是那个关键提问：`Question: is it possible to leverage the much larger amount of (image, caption) pairs?`。历史背景是：那时 GPT-2、GPT-3 已经发生，语言模型整体进入 foundation model era（基础模型时代）；而视觉这边 traditionally 仍基于 ImageNet 这类 large annotated datasets（大规模标注数据集），在上面训练 ResNet，再配合 data augmentation（数据增强）等一整套技巧取得很好的性能。OpenAI 的研究者于是问：既然语言那边可以 scrape the internet（爬取互联网），文本很 noisy（嘈杂），只要模型够大就能 make sense out of this（从中理出意义）并生成有用的东西，那么图像领域的等价物是什么？CLIP 就是这个问题的答案。

同一份课件文件的上下文里还能看到整讲 `Summary:` 的几条淡出文字，它们是画面上出现、口头并未念到的前瞻判断：`Frontier models are expected to be multimodal (natively multimodal, omni)`；`Fundamental challenge: how to encode non-text modalities?`；`Comprehension and generation might demand different things (semantics versus finer-grai[ned ...])`；`Balance images + video (lower information density) and text for training stability`；`Continuous encoders + Transformer + diffusion models for generation`。把这些和 CLIP 的定位对照很有意思：CLIP 正落在"连续编码器"这一侧，它只负责理解侧的语义编码，而生成侧被留给 diffusion models；同时"理解与生成可能需要不同东西"也解释了为什么后面细粒度任务会嫌 CLIP 的表示太粗。

CLIP 的想法被讲者形容为 fairly simple and straightforward（相当简单直接）。幻灯片 `Method:` 下写得很规整：`Get a batch of (image, text) examples (e.g., 32768)`（图上给的精确值是 32768，讲者口头四舍五入说成 32,000）；`Encode each image and each text`；`For each image, prefer its aligned text over other texts`；`For each text, prefer its aligned image over other images`。设一批里有 $N$ 个图文对，图像编码器给出 $I_1,\dots,I_N$，文本编码器给出 $T_1,\dots,T_N$。目标就是让对齐的那一对点积显著大于所有不对齐的组合：$I_1^\top T_1$ 要远大于 $I_1^\top T_{j\ne 1}$，反过来 $T_1^\top I_1$ 也要大于 $T_1^\top I_{i\ne 1}$。于是形成一个 $N\times N$ 的相似度矩阵，对角线是正例、非对角线是负例，讲者说这 basically 是 two to $2\times N$ 个 softmax classification problems（多分类问题），样本被组织成 n by n 矩阵形式；口头他还把编码矩阵称作 $m\times d$，而幻灯片伪代码把 batch 记作 $n$、联合嵌入维度记作 $d_e$，两处记号并不完全一致，读代码时要以伪代码为准。用归一化嵌入与可学习缩放写出对称损失就是：

$$S_{ij}=e^{t}\,\hat I_i^\top \hat T_j,\qquad
\mathcal{L}=-\frac{1}{2N}\sum_{i=1}^{N}\left(\log\frac{e^{S_{ii}}}{\sum_{j=1}^{N} e^{S_{ij}}}+\log\frac{e^{S_{ii}}}{\sum_{j=1}^{N} e^{S_{ji}}}\right)$$

其中第一项沿行做 softmax（对应代码里 `axis=0`），第二项沿列做（对应 `axis=1`），正例标签恰是对角线。架构图把这件事画成三栏，这些都是转写没逐字念到的：`(1) Contrastive pre-training` 里，一叠写着 "Pepper the aussie pup" 的文本卡送进 Text Encoder，一张狗的照片送进 Image Encoder，得到 $I_1\cdot T_1,\ I_1\cdot T_2,\ \dots,\ I_N\cdot T_N$ 的矩阵且对角线格子被涂蓝；`(2) Create dataset classifier from label text` 里，把 `plane / car / dog / ... / bird` 这样的标签填进模板 `A photo of a {object}.`，再经 Text Encoder 得到 $T_1,\dots,T_N$；`(3) Use for zero-shot prediction` 里，狗图经 Image Encoder 得 $I_1$，与 $T_k$ 逐一点积后选出 `A photo of a dog.`。第 (2)(3) 两栏其实提前演示了本段最后一章要讲的 zero-shot 预测。

![CLIP 三栏架构图：对比预训练、由标签文本构造分类器、零样本预测，含 32768 批量与 I_i·T_j 相似度矩阵](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0006_00-06-40.jpg)

最硬的增量是这一页的官方伪代码，讲者只说"归一化、带温度求点积、算 cross entropy"，代码却把每个形状和可学习参数都写明了：

```python
# image_encoder - ResNet or Vision Transformer
# text_encoder  - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l]       - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t             - learned temperature parameter

# extract feature representations of each modality
I_f = image_encoder(I)   # [n, d_i]
T_f = text_encoder(T)    # [n, d_t]

# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)

# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss   = (loss_i + loss_t) / 2
```

![CLIP 官方伪代码：图像与文本编码器、可学习投影 W_i/W_t、可学习温度 t、L2 归一化与对称交叉熵损失](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0007_00-07-53.jpg)

逐行看有几个必须记住的细节：两个编码器在注释里是可选的（`ResNet or Vision Transformer`、`CBOW or Text Transformer`），说明对比目标与具体骨干解耦，直到第 6 章才选定 ViT；$W_i,W_t$ 是两个 learned projection（可学习投影矩阵），把不同维度的特征压到共同的 $d_e$ 空间；`l2_normalize(..., axis=1)` 之后点积才是 cosine similarity（余弦相似度）；温度 $t$ 是 learned temperature parameter（可学习温度参数），缩放因子写成 $\texttt{np.exp}(t)$ 而不是固定超参，取指数保证了缩放恒为正；`labels = np.arange(n)` 意味着第 $i$ 个正例恰好落在第 $i$ 行第 $i$ 列，因此 batch 里若混入语义重复的图文对，就会被当成假负例惩罚；最后的 `loss = (loss_i + loss_t)/2` 把行、列两个方向的分类损失取平均，正是口头说的"对称"。还要留意一个工程上的局限：softmax 是在整行、整列上做的，损失不能像语言模型那样按样本独立分解后再聚合，这也是 CLIP 必须依赖大 batch（这里是 32768）的原因。讲者讲完问了一句 "Any questions about that?"，随后才转入数据与预处理。

## 数据来源与图像预处理：从 4 亿图文对到 336×336 [08:15 - 10:53]

目标定下来之后，下一个问题是数据从哪来。讲者直言 there are not too many details in the paper about the data（论文里关于数据的细节并不多），大致做法是 take a bunch of queries（取一批查询）、go search online（到网上搜索）、mine a bunch of image and text pairs（挖掘图文对）。幻灯片 `Data:` 下把这四步写成了可核对的数字：`Searched for 500K queries, get ~20K (image, text) pairs per query`；`Trained on 400M image-text pairs`；`Didn't release the dataset`；`Reproduced in OpenCLIP (using LAION-5B dataset, which used CLIP for filtering) [Cherti+ 2022]`。口头对应的是：最终 4 亿（400 million）图文对，数据集未公开；后来有篇叫 OpenCLIP 的工作复现并 further extended（进一步扩展）了 CLIP 的思路，发布了 LAION-5B——50 亿（five billion）张带文本描述的图像，还经过一堆处理，有趣的是 they actually used CLIP for the data filtering and then trained open clip（他们居然用 CLIP 来做数据过滤，再训练 OpenCLIP）。讲者点出这里面有 some bootstrapping happening（某种自举／循环依赖），但至少这是一个你能明确指出其训练数据集、也拿得到代码的模型，可复现性比原版 CLIP 好得多。

![CLIP 数据与数据处理页：500K 查询、400M 图文对、LAION-5B，以及用双三次插值把短边缩放到 336 像素再中心裁剪](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0009_00-10-18.jpg)

接着是图像尺寸这个很实际的问题。讲者说 images come in all resolutions（图像分辨率五花八门），互联网上的图可能 long and skinny（又长又瘦）或反过来又高又瘦，幻灯片写成 `Images come in all resolutions (arbitrary W x H)`。而 one thing that you learn about neuronets（关于神经网络你会学到的一件事）是 they don't like things to be dynamic（它们不喜欢动态形状），want things to be fixed size（要固定尺寸）。于是有一个 somewhat heuristic（有点启发式）的处理，幻灯片 `Data processing [code]` 三条逐字是：`Resize using bicubic interpolation so shorter side is 336 pixels`；`Center crop (cuts off borders to get 336 x 336)`。把它写成变换链：

$$s=\min(W,H),\qquad X'=\text{BicubicResize}\!\left(X;\ \text{shorter side}\to 336\right),\qquad
X''=\text{CenterCrop}(X')\in\mathbb{R}^{336\times336\times3}$$

讲者口头补充目标尺寸"是 336 或 whatever target size could be 224 as well"（336，或者别的尺寸比如 224 也行），所以幻灯片锁定的 336 与口头的 224 并不矛盾，前者是这一版实现的取值，后者是可选的另一个 target size；他还强调 this is obviously for convenience（这显然是为了方便），后面会看到 you can do much better than this（能做得比这好得多），这里只是 expediency（权宜之计）。为什么敢这么粗暴？因为 at this time the CLIP authors were thinking about ImageNet and classification（当时 CLIP 作者心里想的是 ImageNet 分类）：usually the object is in the middle（物体通常在正中），裁掉一些背景 it doesn't matter too much（影响不大）。

这一章和前后两章的接缝很清晰：向前，它解释了第 3 章那 32768 个批次样本为什么必须先被规整成统一张量，否则根本堆不成一个 batch；向后，固定的 $336\times336$ 正方形正是下一章 ViT 按 $14\times14$ 切 patch 的前提，也是后面"动态分辨率"改进要推翻的对象。易错点至少有三处：一是 `bicubic interpolation`（双三次插值）这个具体插值算法只写在幻灯片上，口头只说 resize，若照口头复现容易随便用双线性而偏离原实现；二是 `Center crop` 的括注 `cuts off borders` 说明它是裁掉边缘而非把整图压进正方形，边缘信息会被直接丢弃，这对后来的 OCR／文档类任务是硬伤；三是 `Didn't release the dataset` 加上 LAION-5B 用 CLIP 自身做过滤，意味着数据分布里嵌了自举循环，谈"零样本超越 ResNet"时不能把数据完全当作独立中立的前提。

## 为什么用图文对：从数据增强到高层语义 [10:53 - 12:36]

讲者在这里停了一下，指出前面给的只是 the general algorithm（通用算法），但还没指定 the image and text encoders（图像与文本编码器）究竟是什么；紧接着插入了一段学生问答，而这段问答恰好补上了第 3 章没有论证的那个前提：为什么要用 image-text pairs 训练，而不是只用 images？学生的提问被讲者复述成"为什么每一类生成都会带着某种 text，为什么他们同时在 text 和 image 上训练，而不是只在 images 上"。画面上讲者站在白板前，白板左侧留着手绘的一条曲线与 `fixed φ` 字样，右侧投影幕上还淡出着上一页 `Data processing` 的 `Center crop (cuts...)`、`Vision encoder:` 与 `Experimented w...` 等行，以及 ViT 示意图里的 `Class Bird Ball Car ...` 和 `MLP Head`——也就是说，这一问答正发生在从"数据处理"翻页到"视觉编码器"的过渡处，是对前面整整两页设计选择的追问，而不是离题。

![讲者在白板前回答为何用图文对训练，白板留有手绘曲线与 fixed φ 标注，投影幕淡出 Center crop、Vision encoder 等行](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0010_00-11-31.jpg)

讲者的回答先承认存在另一条路线：there is a line of work that tries to encode images into representations based on data augmentation（有一支工作试图只靠数据增强把图像编码成表示）。他点名了 SimCLR 这一类想法：取一张图，augment it（增广它）、crop it（裁剪）、randomly perturb it（随机扰动）、rotate it（旋转），然后要求"这仍是同一张图"，于是两张增广视图的 embedding 应当相似。用式子写就是：设 $\tilde x\sim\mathcal{A}(x)$ 为增广视图，$z=f(x)$、$\tilde z=f(\tilde x)$，目标是拉近 $\text{sim}(z,\tilde z)=\dfrac{z^\top\tilde z}{\|z\|\,\|\tilde z\|}$，同时推远与 batch 里其它图像的相似度。讲者评价这种做法 that's useful for kind of low-level details（对低层细节有用）。

但他立刻给出这条路的死穴：you won't data augment your way from like one type of dog to another dog（你没法靠数据增广把一种狗变成另一种狗）。裁剪、旋转、颜色扰动改变的是像素与几何，改变不了"这是狗、那也是狗"这一层语义类别关系。而 by using that text it gives you higher level semantic representations of the images（借助文本，能得到图像更高层的语义表示）：caption 里天然写着画面里有什么、在做什么，文本监督把"语义相近"这件事直接注入了嵌入空间。这正是第 3 章把目标定成"图文对比"而非"图像自监督"的根本理由，也解释了第 7 章为什么能靠一组文字标签就做出 zero-shot 分类——如果表示里只有低层纹理，文本标签根本无从对齐。换句话说，前面那 32768 个 (image, text) 样本之所以必须成对采集、而不是单张图像，答案就在这一章。

讲者最后预告 later we'll see how these semantic representations are useful if you look at the types of tasks that people are trying to solve（等看到人们想解的任务类型，就会明白这些语义表示有多有用），把话题交还给视觉编码器。这里要分清两点易错之处：CLIP 不是纯粹的视觉自监督模型，把 SimCLR 与 CLIP 混为一谈会错判它的监督信号来源；另外，"高层语义"与"低层细节"是转写里明确对照的两种粒度，文本带来前者的同时并不保证后者，因此不能从"CLIP 语义表示很强"推出"它适合细粒度像素任务"——讲者后面谈位置嵌入时也会再次提醒，这些结论都要放在当时以分类为主的语境里看。

## 视觉编码器 ViT：分块、位置嵌入与 attention pooling [12:36 - 16:24]

回到被搁置的问题：CLIP 用的 vision encoder 是什么？讲者说他们 actually experimented with a bunch of different ones（试过好几种），包括 ResNets 和当时 had just come out（刚问世）的 Vision Transformers，结果 found out the vision transformers perform the best（发现 ViT 最好），所以 when people say CLIP they usually mean the vit version（如今说 CLIP 通常就指 ViT 版）。幻灯片把这一句写得更精确，还补了出处：`Experimented with ResNet-50 and Vision Transformers [Dosovitskiy+ 2020]`——口头只说 ResNets，画面上明确到 ResNet-50，并给出 ViT 原始论文引用。ViT 的做法是：把图像 break it up into these patches（切成小块），原论文是 $16\times16$，CLIP 用 $14\times14$；每个 patch 基本就是一个向量，in some sense a token for this vision transformer（在某种意义上就是视觉 Transformer 的一个词元）；再 add the positional embeddings（加位置嵌入），just like you would if you're training a language model（和训练语言模型时一样），然后 put it through a standard transformer（送进标准 Transformer）。

![Vision Transformer 架构图与 Best model: ViT-L/14@336px 说明，含 attention pooling 与文本编码器规格](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0013_00-15-10.jpg)

幻灯片上的 ViT 架构图把口头讲述填得很满：`Patch + Position Embedding` 旁注 `* Extra learnable [class] embedding`；下面是 `Linear Projection of Flattened Patches`；再上面是编号 $0^*,1,2,\dots,9$ 的一串 patch token；`Transformer Encoder` 内部画出 $L\times$ 重复的 `Norm` → `Multi-Head Attention` → $\oplus$（残差）→ `Norm` → `MLP` → $\oplus$，底部入口标 `Embedded Patches`；输出端接 `MLP Head` 再给出 `Class / Bird / Ball / Car / ...` 的类别表。把这段写成记号：设 $M$ 个 patch，$x_p\in\mathbb{R}^{14\cdot14\cdot3}=\mathbb{R}^{588}$（$14\times14$ 像素、RGB 三通道展平），则

$$H=\text{TransformerEncoder}\big(\text{LinearProj}(X)+E_{\text{pos}}\big),\qquad X=[x_1,\dots,x_M]$$

接下来是如何从"一串向量"得到"一个向量"。讲者说 transformer encoder produces a sequence of vectors（产出的是一个向量序列），可以 average all these vectors（把所有向量平均）得到单个向量，但 CLIP 论文 does something a little bit different（做得稍有不同）：they saw it was better to do a attention pooling（发现用注意力池化更好）——先取 global average of all the activations（所有激活的全局平均），再用它作 query，against the key and values of each position（与各位置的 key、value）做 another round of attention（又一轮注意力），得到的向量 maybe a little bit more informed（信息更充分），而不是 straightup average（直愣愣的平均）。幻灯片原文是 `Attention pooling: do QKV with query = global average of activations`，形式化即：

$$\bar h=\frac{1}{M}\sum_{p=1}^{M}H_p,\qquad
\text{AttnPool}(H)=\text{softmax}\!\left(\frac{(\bar h W_Q)(H W_K)^\top}{\sqrt{d}}\right)H W_V$$

关于最佳模型，讲者口头给的是 ViT-L/14：a large vit、I think it has like 24 layers or so, not quite sure about that（大概 24 层左右，他自己也说不太确定）、$14\times14$ patches、每个 patch 是 RGB 三通道、trained on 336 by 336 resolution images（在 $336\times336$ 分辨率上训练）；而幻灯片黄色高亮给出了精确写法 `Best model: ViT-L/14@336px (L = large, 14x14 patches, 3 channels, trained on 336x336 resolution images)`。训练策略上他还纠正了自己一次：其实是先在较低分辨率训练，后段再提高分辨率，他起初说"为了提高"，随即改口说是 for speed（为了速度），因为 high resolution images takes longer（高分辨率更慢），所以只在 the latter part of training（训练后期）才用高分辨率。此外，幻灯片下方淡出的 `Text encoder:` 两行提前泄露了下一章内容：`GPT-2 Transformer (63M parameters, 12 layers)` 与 `Encode [BOS] ... [EOS], return [EOS] activation at highest layer`。

这时有学生追问：位置嵌入是不是该比文本更复杂？文本只有一维，图像却有二维。讲者承认这个直觉合理，但说这里只是 linear zero through nine（线性地 0 到 9），CLIP 论文里 I believe they tried some 2D version of this and the 1D na… they found that it doesn't really matter that much（试过 2D 版本，发现和 1D 差别不大）。他随即加了一句很重要的限定：I always have to take these results with the idea that they're interesting classification in mind（我总得带着"这些结果关心的是分类"这个前提去看它们），so for classification maybe it doesn't matter（对分类也许确实无所谓），而 later we'll see fancier positional embeddings that do take into account the spatial structure（后面会看到把空间结构考虑进去的更精细位置嵌入）。

本章至少有三处必须小心：其一，讲者口头的"约 24 层"是他对 ViT-L 的不确定估计，而幻灯片 `63M parameters, 12 layers` 指的是文本编码器 GPT-2，两者根本不是同一个模型，切勿把 12 层安到 ViT-L 头上；其二，ViT 的 patch 是 $14\times14\times3=588$ 维的原始像素展平，它只是"被当作 token"，并不天然具备第 2 章要求的语义单元属性，语义是后面靠对比目标训出来的；其三，CLIP 的 attention pooling 用全局平均作 query，与架构图里画着的 `[class] embedding`／`MLP Head` 那条标准分类路径并不相同，别把两张图读成同一套读出方式。至于 1D 位置嵌入"够用"的结论，也只能停在分类语境里，一旦任务需要空间结构就不成立——这一限定正是讲者为后续内容留的口子。

## 文本编码器、训练流程与 zero-shot ImageNet 结果 [16:24 - 19:24]

视觉侧定完，文本侧就简短得多。讲者说 the text encoder is a standard transformer（文本编码器是标准 Transformer），而且是 a GPT2 style transformer（GPT-2 风格），理由是 since that was from the same group that developed GPT2（因为它出自开发 GPT-2 的同一组人）。难点同样是 in order to get a single vector out of a sequence（从序列里取出单一向量）：他们 they prepended BOS and appended EOS（在开头加 BOS、末尾加 EOS），然后 took EOS the activation at the highest layer（取最高层 EOS 位置的激活）作为 that whole sequence 的表示。幻灯片把它写成 `Encode [BOS] ... [EOS], return [EOS] activation at highest layer`，并标注规格 `GPT-2 Transformer (63M parameters, 12 layers)`。这与上一章视觉侧的 attention pooling 恰好构成对称：两侧都得把一串向量压成一个嵌入，才能算出第 3 章那个 $N\times N$ 的点积矩阵；区别只在一边用注意力池化、一边直接取句末标记的顶层激活。

![GPT-2 文本编码器与 Headline result 页：ImageNet 零样本结果、Ablation 与 Alternative: predict 标题](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0015_00-17-35.jpg)

有了两个编码器，训练就闭合成一个很朴素的循环，讲者一句话带过：you pick a batch, you encode all the text, you code all the images, and then you form these two end cross entropy losses and then you go from there（取一批、编码所有文本、编码所有图像、构造两个 $N$ 路交叉熵损失，然后继续）。用第 3 章的伪代码回看就是：

```python
for I, T in batches:                     # N 个 (image, text) 对
    I_e, T_e = encode(I), encode(T)      # 各自压成单一嵌入并 L2 归一化
    logits = np.dot(I_e, T_e.T) * np.exp(t)
    labels = np.arange(n)
    loss_i = cross_entropy_loss(logits, labels, axis=0)   # 图像 -> 文本
    loss_t = cross_entropy_loss(logits, labels, axis=1)   # 文本 -> 图像
    loss   = (loss_i + loss_t) / 2
```

然后是本段最被强调的 headline result，讲者说它 got a lot of people excited about clip back in 2021（2021 年让很多人对 CLIP 兴奋起来）：on the image net benchmark, zero shot clip outperformed a resnet that was trained on 1.2 million image net images（在 ImageNet 上，零样本 CLIP 超过了用 120 万张 ImageNet 图像训练的 ResNet）。他特意把这 120 万张放回人力尺度里衡量：this was many many hours of Amazon Mechanical Turk worker time to do this annotation（那是 Amazon Mechanical Turk 工人许许多多小时的标注工时）；而 CLIP trained on more organic web data（训练在更"有机"的网页数据上），虽然那同样是 a bunch of people on the web 的劳动，但 if you're able to leverage that or existing data already then you can just zero it（只要能利用这些已有数据，就能直接零样本使用）。幻灯片对应页可见 `Headline result:` 与 `On ImageNet, zero-s...`，下面还挂着 `Ablation:` 和 `Alternative: predict...` 的标题——讲者在本段并未展开这些消融内容，因此这里只登记标题、不替他补结论。零样本预测的做法是：取一张图像，再备一组标签文本，与各标签算点积、取最高者，

$$\hat y=\arg\max_{k}\ \hat I^\top \hat T_k$$

其中标签文本按第 3 章架构图第 (2) 栏的模板 `A photo of a {object}.` 生成。有个小插曲很能说明现场节奏：讲者先说 "I don't have slide support here"（这里没有幻灯片支持），经学生提醒后改口 "Oh yeah it is. Sorry it is on this part of the site. Okay good point."（哦有的，对不起，就在这一侧），确认这正是 `zero-shot prediction` 那一栏。也就是说，把分类器"文本化"、不必为每个数据集重训分类头，是 CLIP 能把 4 亿对网页图文转成 120 万张人工标注替代品的机制所在。

最后一段是另一个学生提问：这种（弱）监督方式如果有张狗图、caption 写成狗，但别的 caption 里也提到狗，会不会 confuse the model（把模型搞混）？讲者把问题复述成"有狗图、且可能有其它也含狗的 caption 会不会混淆"，然后答 in general this process is going to be noisy（这个过程总体上本来就会很嘈杂）；it's okay if there's another dog（多一张狗没关系），因为 on average it's unlikely that there's always going to be a dog（平均而言不可能每次都是狗），sometimes it's going to be apple or a cat or something（有时会是苹果或猫之类）。他接着起了个话头 "but also another point is that these images are um basically scraped"（但另一点是，这些图像基本上是爬来的……），转写在 19:24 处被截断，话没说完，因此这里不替他补出结论。

收束时有两个易错点值得钉住。第一，CLIP 全程只做输入侧编码：它输出的是相似度与标签选择，并不生成图像、音频或视频，把第 1 章 omni model 的 `Output any combination of modalities` 当成 CLIP 已实现的能力，是对本讲范围最典型的误读；本讲从一开始就声明只回答第 1 个问题。第二，caption 噪声与 batch 内假负例（第 3 章 `labels = np.arange(n)` 把非对角线一律判负）是对比学习固有的局限，讲者用"平均而言不会总是狗"来化解，靠的是大规模统计而非单样本正确性——这也正是为什么 400M 这个数据规模、32768 这个 batch 规模在本段里反复出现：它们不是炫目的数字，而是这套含噪监督能收敛的前提。而讲者被打断的那句"这些图像是爬来的"，恰好指向数据质量这一尚未展开的方向。

## 网络图文数据的噪声与「直接预测文本」的消融实验 [19:24 - 22:01]

本段从 CLIP 训练数据的性质说起。CLIP 所用的图文对来自互联网上带有文字说明、或者带 alt 文本（alt text，替代文本）的图像，讲者强调这些数据「极其嘈杂」（extremely noisy）。有研究专门考察过 caption（图像描述）与图像的对应关系，结论是描述并不必然逐字复述画面内容：如果图里是一只狗，写说明的人往往不会再说一句 "a dog"，因为图像本身已经把这件事呈现出来了。这意味着 caption 与图像之间不是严格的像素级对应，而是带有大量省略、主观取舍甚至错配。因此讲者说，CLIP 居然能奏效「多少有些令人惊讶，但也很有意思」，前提是必须做大量的数据过滤（data filtering）；如果直接把任意网页上的图文对拿来用，噪声会大到无法训练。这一点是理解后面所有设计取舍的基础：CLIP 学到的不是图像到文字的精确翻译，而是从带噪文本里抽取出来的高层语义。

接着讲者介绍 CLIP 论文里做的一组 ablation（消融实验），幻灯片上的标题就写着 "Ablation:"，下面两条 bullet 分别是 "Alternative: predict text from images directly"（替代方案：直接从图像预测文本）和 "Much less compute efficient compared to CLIP-style ranking"（相比 CLIP 式的排序目标，计算效率低得多）。也就是说，研究者没有采用对比排序，而是换了一个目标：拿图像去预测配对的文本。具体实现有两种，一种把文本当作 bag of words（词袋，只预测词集合而不管顺序），另一种把文本当作真正的 language model（语言模型，逐 token 自回归生成 caption）。讲者的口述结论是「有些出乎意料」：用更强的模型去做这件事，效果反而更差，或者至少明显更低效，比不上 bag of words 式的 CLIP。换句话说，建模能力更强、目标更精细，在下游任务上却没有换来更好的表示。

幻灯片上的曲线图给出了这组实验的量化细节，这些数值转写里并没有念出来。纵轴是 Zero-Shot ImageNet Accuracy（零样本 ImageNet 准确率），横轴是 # of images processed（处理过的图像数量），刻度依次为 2M、33M、67M、134M、268M、400M。图里有三条曲线：绿色的 Bag of Words Contrastive (CLIP) 始终最高，在 400M 处约到 42；橙色的 Bag of Words Prediction 居中，在 400M 处约到 27；蓝色的 Transformer Language Model 最低，在 400M 处约到 16。图上还画了两支水平箭头并标注 "4X efficiency" 与 "3X efficiency"，表示达到同一准确率时另外两种目标所需的数据量分别是 CLIP 的约四倍和三倍。把这两个标注写成式子，就是 $\dfrac{\#\text{images}_{\text{BoW-Pred}}}{\#\text{images}_{\text{CLIP}}}\approx 4\times$ 与 $\dfrac{\#\text{images}_{\text{Transformer}}}{\#\text{images}_{\text{CLIP}}}\approx 3\times$；换成终点读数则是 $\text{Acc}_{\text{CLIP}}(400\text{M})\approx 42$、$\text{Acc}_{\text{BoW-Pred}}(400\text{M})\approx 27$、$\text{Acc}_{\text{Transformer-LM}}(400\text{M})\approx 16$。

![CLIP 排序目标与两种「直接预测文本」目标在零样本 ImageNet 准确率上的效率对比曲线，标出 4X 与 3X efficiency](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0018_00-21-14.jpg)

为什么会这样，讲者给出的解释是评估指标本身：这里比的是 ImageNet accuracy，本质是 classification（分类），只需要图像的「粗略表示」（rough representation）。对于这种粗粒度语义，精确建模 caption 里每个 token 的先后顺序并不重要，词袋级别的信息已经够用。这也埋下了一个贯穿全讲的点：理解（把图像压成可分类的语义向量）与生成（复述或产出文字）对表示的要求并不相同，讲者明确说这个点会在全讲最后再回来谈。易错之处在于，不要因为「语言模型更强」就推断它更适合做视觉表示学习；在分类导向的评价下，更强的序列建模能力只是把算力花在了任务不需要的地方，于是表现为数据效率下降。

随后幻灯片切到 "Summary:"，三条 bullet 是本段的正式小结，同样属于画面上才写得清楚的内容。第一条 "Encoding of images captures semantics given by (noisy) text"：图像的编码捕获的是（带噪声的）文本所给出的语义，因为配对的文字通常谈论的是图像的高层内容而非细节。第二条 "Design decisions chosen based on image classification (not very fine-grained)"：CLIP 的各项设计决策是基于图像分类选定的，因此表示并不精细（not very fine-grained）；但讲者紧接着补充，它依然是我们后面一切工作的「稳健起点」（robust starting point）。第三条被高亮为 "Technical: requires large batch sizes, softmax operation over full batch"：技术上它需要非常大的 batch，而且 softmax 是在整个 batch 上做的。讲者口述给出的量级是 30,000 左右，并直说 batch size 为 1 显然不行、为 10 也不行；更麻烦的是这种损失「不太可分解」（not really very decomposible）。

把这一条技术局限写成式子会更清楚。设一批有 $N$ 个图文对，归一化相似度为 $S_{ij}=\hat I_i^\top\hat T_j/\tau$，CLIP 的对称目标在行、列两个方向各做一次 softmax 交叉熵：

$$
\mathcal{L}_{\text{CLIP}}
=-\frac{1}{2N}\sum_{i=1}^{N}
\left(
\log\frac{e^{S_{ii}}}{\sum_{j=1}^{N}e^{S_{ij}}}
+
\log\frac{e^{S_{ii}}}{\sum_{j=1}^{N}e^{S_{ji}}}
\right)
$$

关键就在分母 $\sum_{j=1}^{N}$：它把整个 batch 里的 $N-1$ 个其他样本都当成负例，所以负例集合由 batch 决定，batch 一变，损失函数本身就变了。讲者用普通语言模型训练作对照：那里一个 batch 里的各条序列基本相互独立、可以并行计算，最后做一次聚合（aggregation）即可；CLIP 却因为样本间要互相比相似度而无法这样拆开。这正是下一章 SigLIP 要动的病根。幻灯片末尾已经露出下一节的签名 "def siglip():" 与 "SigLIP (Sigmoid Loss for Language Image Pre-Training) [Zhai+ 2023]"，过渡非常直接。还需要注意一个常见的误读：CLIP 的「语义表示」不等于能读文档小字、能定位物体，它的粒度受分类目标限制，这个短板会在本段最后讲 AnyRes 时再次浮出水面。

## SigLIP：把多分类对比改成逐对二分类 [22:01 - 24:52]

SigLIP 是 Google 提出的改进版，全称 Sigmoid Loss for Language Image Pre-Training（用于语言图像预训练的 Sigmoid 损失），讲者把它称作「基本上是 CLIP 的改进版本」。幻灯片 "Objective:" 下的两条 bullet 把差别说得很干脆：CLIP 是 "multiclass classification for (text, image) versus (text, image') for all image'"，即对一个已对齐的图文对，要把它与所有替代图像（或替代文本）区分开，是一个 $N$ 选 1 的多分类问题；SigLIP 则是 "binary classification for (text, image) - aligned or not?"，对任意给定的图文对只回答一个问题——它们是否对齐。讲者说 SigLIP「在某种意义上简单得多」，因为它不需要维护「相对谁最好」这种全局竞争关系，只需要逐对判断。

从矩阵角度看，这个改动非常直观：仍然是那个 $N\times N$ 的相似度矩阵，对角线是正例、非对角线是负例，讲者说完这句就加了一句 "That's it."（就这么简单）。CLIP 要在每一行、每一列上做 softmax，让对角线元素在整行整列里胜出；SigLIP 则把矩阵里每一个格子当成一个独立的二分类样本，正格子的标签是 $+1$、其余是 $-1$，各自算一次 logistic 损失。于是分母里的 $\sum_j$ 消失了，样本之间不再通过归一化互相耦合，这就是后面能把计算拆到多台设备上的根本原因。

![SigLIP 幻灯片：Objective 两条 bullet 与 Algorithm 1 Sigmoid loss 伪实现完整代码](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0020_00-23-39.jpg)

讲者口述的算法步骤是：取嵌入、做归一化、照常算点积、构造标签（非对角线 $-1$、对角线 $+1$）、然后算 log sigmoid，并评价「目标非常简单」。幻灯片上给出的 "Algorithm 1 Sigmoid loss pseudo-implementation" 是转写完全没有念到的画面内容，原样如下：

```python
# img_emb        : image model embedding [n, dim]
# txt_emb        : text model embedding [n, dim]
# t_prime, b     : learnable temperature and bias
# n              : mini-batch size

t = exp(t_prime)
zimg = l2_normalize(img_emb)
ztxt = l2_normalize(txt_emb)
logits = dot(zimg, ztxt.T) * t + b
labels = 2 * eye(n) - ones(n)  # -1 with diagonal 1
l = -sum(log_sigmoid(labels * logits)) / n
```

其中 `img_emb` 与 `txt_emb` 形状都是 `[n, dim]`，`t_prime` 和 `b` 是可学习温度与偏置（learnable temperature and bias），`n` 是 mini-batch size。写成数学形式，令 $\hat I_i=\text{l2\_normalize}(I_i)$、$\hat T_j=\text{l2\_normalize}(T_j)$，则

$$
t=\exp(t'),\qquad
\ell_{ij}=t\,\hat I_i^\top\hat T_j+b,\qquad
y_{ij}=2\cdot\mathbf{1}[i=j]-1
$$

$$
\mathcal{L}_{\text{SigLIP}}
=-\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{n}
\log\sigma\big(y_{ij}\,\ell_{ij}\big)
$$

$\text{eye}(n)$ 是 $n\times n$ 单位矩阵，所以 `2 * eye(n) - ones(n)` 恰好把对角置成 $+1$、其余 $n^2-n$ 个格子置成 $-1$；再乘进 `logits` 后取 `log_sigmoid`，等价于让正例的 $\ell_{ii}$ 尽量大、负例的 $\ell_{ij}$ 尽量小。温度用 $t=\exp(t')$ 而不是直接学 $t$，是为了保证它恒为正；偏置 $b$ 则用来吸收正负例极度不平衡带来的初始偏置。

课堂里有一段问答值得记下。有学生问这套方法是否需要复杂的采样策略（sophisticated sampling strategies），讲者回答：至少在最初的论文里，他们就是「字面意义上在同一类矩阵上」操作，没有做什么特别处理。他顺带点出一般对比学习里的常见直觉——负例有时需要平衡正负比例，或者需要「紧负例」（tight negatives）才不至于产生偏置——但 SigLIP 初版「相当直白」（fairly straightforward）。这里的易错点正是：不要想当然地认为一定要做难负例挖掘或复杂的负例采样才能训练好；至少在论文报告的设置里没有这一步。与本段前一部分的关系也很清楚：SigLIP 保留了 CLIP 的编码器结构、归一化与点积，只替换了损失函数，因此它继承的是同一个「图文对齐」语义空间，只是把不可分解的全局 softmax 换成了可逐元素累加的 sigmoid。局限则在于，正负例比例变成约 $1:(n-1)$，样本极度不平衡，而初版并未专门补偿；另外幻灯片下方已经露出 Data 部分的三条 bullet —— "WebLI dataset: O(billion) (image, text) pairs [Chen+ 2022]"、"Scraped from the Internet"、"Used automatic OCR to extract text from images" —— 把话题引向数据与训练效率。

## WebLI 数据、训练效率与跨设备并行 [24:52 - 28:33]

数据方面的数字主要来自画面而非口述。SigLIP 用的是 Google 的 WebLI 数据集，规模写作 $O(\text{billion})$ 量级的图文对，出自 [Chen+ 2022] 的另一个图文模型（约 2022 年），数据从互联网抓取（Scraped from the Internet）。除了网页自带的图文对应关系，他们还用 automatic OCR（自动光学字符识别）从图像里把文字抽出来，以此再构造出一批文本—图像对；讲者口述里说这是「另一种形成图文对的方式」，并且做过过滤（filtering），整个数据集是 multilingual（多语言）的。幻灯片上更具体的两条 bullet 是 "Keep 10% highest quality"（只保留质量最高的 10%）与 "Supports 100 languages"（支持 100 种语言）—— 这两个参数转写里都没有出现，却直接解释了为什么在如此嘈杂的网页数据上还能训出可用表示：先按质量分数筛掉绝大多数，再靠多语种覆盖把数据规模撑起来。这也回扣了本段开头「必须做大量数据过滤」的判断。

效率是这篇论文的主要卖点，幻灯片 "Efficiency:" 下两条 bullet 写得很直白："CLIP: 10 days on 256 TPUv3" 与 "SigLIP: 5 days on 32 TPUv4 (lower FLOP/s than TPUv3) - much faster!"。也就是 CLIP 用 256 块 TPU v3 训 10 天，SigLIP 用 32 块 TPU v4 只训 5 天。讲者特意纠正了一个直觉：不要以为 v4 一定更快，以单芯片 FLOP/s 计，TPUv4 其实比 TPUv3 低，大约慢 60% 左右；v4 的优势在于能在一个 pod 里塞进更多芯片、interconnect（互联）更好，因而在这种规模上系统级吞吐更高。他还补了一句，CLIP 当初大概也没有把代码优化到最大吞吐，只是「先让它跑起来」（let's get this thing to work），所以这组数字并不纯粹是算法带来的。易错点就在这里：「更快」是端到端训练时长与并行效率层面的结论，不是单芯片算力结论。

![SigLIP 幻灯片：WebLI 数据 bullet、100 种语言、10% 质量筛选与 TPU 训练效率对比](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0022_00-26-05.jpg)

真正让训练变快的是并行方式。讲者让大家回忆 systems（系统）课程里的 DDP（Distributed Data Parallel，分布式数据并行）：每台设备保存图文对的一个子集。但与语言模型训练不同，这里样本之间存在 interactions（相互作用），不能像 LM 那样「everything just factors」（一切都能因式分解）——因为每个图像都要和所有文本算负例。SigLIP 的做法是：第一遍（first pass）每台设备只计算自己本地那批图文对上的损失；然后把文本在设备间轮换发送，device 1 先拿到 T5 到 T8，于是能补出对应那一块非对角负例，再拿到 T9 到 T12，如此「rotate around」（轮转一圈）直到覆盖所有 off diagonal（非对角）块。幻灯片上的矩阵图把这件事画得很具体：行是 $T_1\ldots T_{12}$、列是 $I_1\ldots I_{12}$，按 Device 1 / Device 2 / Device 3 分成若干 $4\times4$ 子块，对角子块里标 "+"（正例），非对角标 "−"（负例）；三幅小图依次显示各设备分担的 loss 比例 33%、50%，最后一幅标着 "Cross Device $\Sigma$"（跨设备求和）。用式子写就是

$$
\mathcal{L}=\sum_{d}\mathcal{L}_{d}
=\frac{1}{n}\sum_{d}\sum_{(i,j)\in\mathcal{B}_{d}}
-\log\sigma\big(y_{ij}\,\ell_{ij}\big)
$$

其中 $\mathcal{B}_d$ 是第 $d$ 台设备负责的格子集合。之所以能这样分块再求和，正是因为 sigmoid 损失对每个 $(i,j)$ 独立，没有 CLIP 那个横跨整行整列的 softmax 分母；换句话说，算法层面的简化直接换来了系统层面的可并行性。这是本段最值得记住的因果链条。

与「解耦」相关的另一组结论在幻灯片 "Batch size:" 下，三条 bullet 分别是 "Decouple batch size from loss"、"Better than CLIP for <16K batch sizes"、"Go up to 1M batch size, but 32K is enough"。第一条呼应前面：CLIP 里损失与 batch size 绑死，改 batch 就等于换了一个损失函数；SigLIP 里损失在期望上不随 batch 变化，小 batch 只是方差更大（more variance but the same in expectation）。第二条给出实证区间：他们试了远小于 16K 的 batch，此时 SigLIP 明显优于 CLIP，因为 CLIP 的 softmax 在候选负例太少时会直接退化（degrades）。第三条说明上界与拐点：batch 可以一路开到 100 万，但收益不大，32K 基本就是 critical batch size（临界批大小），讲者口述里也说 "32k was their essentially their critical batch size"。这里的易错点是把 32K 当成硬上限，或把 1M 当成必需配置；它只是收益饱和的经验点。另一个局限是，正负例不平衡加上小 batch 带来的高方差，意味着这套「无需复杂采样」的结论未必能无条件外推到所有数据分布。

讲者在 [28:37] 收束这一段：「现在我们有 CLIP 和 SigLIP 了」。它们都是图像编码器（image encoder），在本讲这个版本里输入的是固定尺寸图像 —— 例如 $336\times336$ —— 输出一个「包含某种语义」的向量。有了这个向量，接下来的问题就变成如何把它接进语言模型，于是话题从「学表示」转到「造 VLM」。

## VLM 通用模板与 LLaVA 的合成数据 [28:33 - 32:35]

有了 CLIP 与 SigLIP 这类把固定尺寸图像映成语义向量的编码器，讲者开始构建 VLM（Vision Language Model，视觉语言模型），并说明要讲两个家族：LLaVA 与 Qwen。二者的 broad template（总体模板）非常相似，细节上有差别。基本思想是「把这些 embedding 直接注入（inject）到语言模型里」。他特别点明这更像是 mid-training 或 post-training 的风味：拿一个已有的图像编码器、拿一个已有的 LLM，然后把两者「缝合」（stitch）起来，而不是 training something from scratch（从头训练）。这一点决定了后面所有设计的性质——可训练参数很少、训练代价低、能力主要继承自两个预训练组件，合成数据只负责教会它们对接。

LLaVA 论文出现在 2023 年，讲者说它当时让人兴奋，是因为那段时间 GPT-4 一类闭源模型已经能做 visual reasoning（视觉推理），而 LLaVA 证明了开放模型也能做一些视觉推理（虽然当然不如 GPT-4），更关键的是「人们终于能看到引擎盖下面是什么」（what went under the hood）。幻灯片上 `def llava():` 下面写着 "LLaVA (Large Language and Vision Assistant) [Liu+ 2023]"，右侧还弹出一张论文卡片：标题 "Visual Instruction Tuning"（视觉指令微调），作者 Haotian Liu、Chunyuan Li、Qingyang Wu、Yong Jae Lee，日期 2023-04-17，链接指向 arXiv:2304.08485。卡片摘要里说明：用 machine-generated instruction-following data（机器生成的指令遵循数据）做 instruction tuning 已经提升了 LLM 在新任务上的 zero-shot 能力，而本文是把这一思路引入多模态领域的首次尝试，方法是用 GPT-4 生成 multimodal language-image instruction-following data，并据此提出 LLaVA 这个连接视觉编码器与语言模型的大型多模态模型。这些作者、日期与摘要文字都是画面独有信息，转写里只说了「2023 年的论文让人兴奋」。

VLM 的部件在这一页列得很清楚。Vision encoder（视觉编码器）用 CLIP；Text decoder（文本解码器）用 Vicuna —— 幻灯片原文是 "Vicuna (LLaMA fine-tuned on ShareGPT conversations) [post]"，即在 ShareGPT 对话上微调过的 LLaMA，而 ShareGPT 指的是人们把自己与 ChatGPT 的对话分享到某个网站上形成的语料，讲者顺带说那个网站现在应该已经不在了。这两个部件正好对应下一章架构图里的视觉编码器与 $f_\phi$；也就是说，LLaVA 没有自己训编码器或语言模型，只是把两个现成模型拼起来，这与「缝合」的定位完全一致。

数据部分是 LLaVA 最有特色的增量。基础数据集是 MS COCO，一个带人工标注的数据集：Mechanical Turk 的工人给图标注 bounding boxes（边界框）以及描述画面内容的 captions。LLaVA 的做法是拿这些 captions 或检测到的物体去 prompt GPT-4，让 GPT-4 生成 questions（问题）、conversations（对话）、detailed description（详细描述）或 complex reasoning（复杂推理）等文本，再把生成结果与原始图像配对，从而「合成」（synthesize）出一份适合指令微调的数据集。讲者现场举例：给一张 MS COCO 里的图，人工标注的 caption 和人工标注的框都在，问 GPT-4 生成一段对话就得到一组问答，让它生成详细描述就得到类似 caption 的文字，让它生成复杂推理就得到更高阶的分析。幻灯片上 "Context type 1: Captions" 给出的真实例子包括 "A group of people standing outside of a black vehicle with various luggage."、"Luggage surrounds a vehicle in an underground parking area."、"People try to fit all of their luggage in an SUV." 等五条；"Context type 2: Boxes" 则给出带坐标的物体列表，例如 `person: [0.681, 0.242, 0.774, 0.600]` 与 `backpack: [0.383, 0.696, 0.495, 0.914]`。图上的另外两条 bullet 是 "Pair generations with original images" 和 "158K examples"，与讲者口述的 158,000 条合成样本一致。把这条流水线写成伪代码：

```python
# LLaVA: 用 GPT-4 把 MS COCO 标注转成指令数据
for image, captions, boxes in mscoco:
    context = {"Context type 1: Captions": captions,
               "Context type 2: Boxes": boxes}   # 人工标注
    for task in ["conversation", "detailed description",
                 "complex reasoning"]:
        text = GPT4.generate(context, task)       # 合成问题/描述/推理
        dataset.append((image, instructions[task], text))  # 与原图配对
# 合成规模: 158K examples，全部是 image + text -> text
```

这里有几个容易踩的坑。第一，所有样本都是「图像 + 文本 → 文本」（image plus text to text），所以 LLaVA 只会被训练成输出文字，并不会生成图像；把 VLM 想成能出图是常见误解。第二，数据是 GPT-4 蒸馏出来的合成数据，带有明显的 post-training 色彩，质量、语气与偏见都受教师模型限制，且 158K 相对预训练语料非常小，只能承担「对齐 / 指令微调」的角色，不能指望它注入新知识。第三，坐标形式的 boxes 与自然语言 captions 是两种不同的 context，把它们混在一起喂给 GPT-4 时，模型看到的其实是「符号化的检测结果」而非原始像素，因此合成数据的视觉依据本身就不是原始图像细节。讲者在本章末尾提到模型用的是 CLIP 里性能最高的版本 ViT-L/14、patch 尺寸为 14（即 $14\times14$），这就把话题交给了下一章的模型结构。

![LLaVA 幻灯片：MS COCO 标注、GPT-4 生成指令数据、Context type 1/2 与 158K 样本](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0026_00-31-12.jpg)

## LLaVA 的模型结构与两阶段训练 [32:35 - 35:20]

幻灯片 "Model:" 下的两条 bullet 定义了架构：第一条 "Encode images with CLIP (ViT-L/14)"，即用 CLIP 的 ViT-L/14 编码图像，讲者口述里提到 $14\times14$ 的 patch 划分；第二条 "Linear projection (W) into embedding space (Flamingo and Q-former are more complex)"，即用一个线性投影矩阵 $W$ 把视觉输出投到嵌入空间，并特意注明 Flamingo 和 Q-former 是更复杂的替代方案。这句括号很关键：LLaVA 是刻意选了最简单的连接器，把复杂度留给「缝合」这件事本身，而不是去设计更花哨的跨模态模块。

架构图上把符号标得很完整，这些标记转写里只念了个大概。图像 $X_v$ 先经过 Vision Encoder 得到 $Z_v$，再经 Projection $W$ 变成 $H_v$；语言指令 $X_q$ 走标准文本嵌入得到 $H_q$；$H_v$ 与 $H_q$ 拼成一条序列送进 Language Model $f_\phi$，最后产出 Language Response $X_a$（图上画成一排绿色 token）。写成式子：

$$
Z_v=g(X_v),\qquad H_v=W\,Z_v,\qquad H_q=\text{Embed}(X_q)
$$

$$
X_a=f_\phi\big([\,H_v\,;\,H_q\,]\big)
$$

其中 $g$ 是 CLIP 视觉编码器，$f_\phi$ 是 Vicuna 语言模型，分号表示在序列维度上拼接。讲者的解释是：CLIP 给出的那个向量「并不真的和文本处在同一个空间里」（isn't really in the same space），所以要乘一个矩阵 $W$ 得到另一个向量，使它进入 text embeddings 所在的空间；文本这边则照常用标准 embedding 得到向量。于是文本被编码成向量、图像也被编码成向量，整条向量序列穿过一个标准 Transformer 就产生输出。讲者把这件事概括成「在某种意义上把图像转换成文本 token」（converting these images into textual tokens），从而可以 leverage（复用）预训练好的语言模型。这正是缝合路线的核心直觉：不改 Transformer，只改它看到的输入。

训练分两阶段，幻灯片 "Training:" 下的原文是："Stage 1 (alignment): freeze vision encoder and language model, only train W" 与 "Stage 2 (fine-tuning): freeze vision encoder and train W and language model"。用可训练参数集合表示：

$$
\Theta_1=\{W\},\qquad \Theta_2=\{W,\ \phi\},\qquad g\ \text{始终冻结}
$$

第一阶段叫 alignment（对齐阶段），视觉编码器和语言模型都冻结，只训练 $W$。讲者说明理由：如果 $W$ 是随机的，投影出来的图像向量「肯定不是表示任何自然语言 token 的嵌入」；训练 $W$ 的目标就是让这些图像向量看上去像自然语言 token 的 embedding（look like natural language token embeddings）。第二阶段仍冻结视觉编码器，但放开 $W$ 与语言模型，在合成数据 —— 一张图配一段对话、描述或复杂推理 —— 上微调语言模型。注意两阶段里视觉编码器一次都没被训练，这是 LLaVA 与「从头训」路线最直观的差别。

![LLaVA 架构图与 Training 两阶段 bullet，Stage 1 对齐阶段被高亮](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0028_00-33-58.jpg)

这里有几个局限值得记牢。其一，alignment 阶段语言模型也被冻结，所以 Stage 1 学不到任何新的语言能力，它本质上只是在学一个从视觉空间到词嵌入空间的坐标系变换；真正的「会回答问题」来自 Stage 2。其二，把图像当作一串软 token 注入，意味着视觉信息必须先被 CLIP 压成固定长度的序列，细粒度空间结构在编码阶段就已经丢失，后面讲 AnyRes 时要补的正是这个洞。其三，$W$ 只是线性投影，表达能力有限，这也是 OneVision 把它升级成两层 MLP 的直接动因。其四，两阶段的划分是工程选择而非理论必然，讲者在后面谈到多阶段训练时也会表达类似态度。与前后内容的关系很顺：上一章的 158K 合成样本正是 Stage 2 的输入，而 [34:38] 之后讲者展示论文里的一个真实例子，用来说明模型的行为，这就引出下一段。

## 极端熨烫示例与 LLaVA OneVision 的组件升级 [35:20 - 37:00]

讲者先展示论文里的一个例子，画面是一张「极端熨烫」的照片（来源标注为 barnorama.com 的 03-Confusing-Pictures.jpg）。用户问 "What is unusual about this image?"（这张图有什么不寻常之处），LLaVA 的回答相当完整：它指出不寻常之处在于一名男子在 minivan 或 van 的车尾熨衣服（a man ironing clothes on the back of a minivan or van），这不是进行该活动的典型场所，通常人们会在家里这类更稳定、安全的地方用普通熨衣板；画面里是架在车上的 makeshift（临时拼凑）熨衣装置，既反常又不安全，而且「不清楚这名男子如何能在如此不稳定的环境里保持平衡与稳定」。讲者口述时把这句浓缩成「你通常不会在 minivan 车尾熨衣服」，并强调模型不错的地方在于：即使用户的 prompt 并没有刻意引导「不寻常」（not really prompting like the unusualness），它仍然会谈及这一点。

同一张图，在 "[Start a new conversation, and clear the history]"（开启新对话并清空历史）之后再问 "What's happening in the scene?"，LLaVA 给出另一段描述：一名男子站在繁忙城市街道上一辆黄色 SUV 顶上，手持便携梯，似乎在车顶熨衣服；周围有其他车辆、交通灯和消防栓，还有两个人，一个靠近车上的男子、一个在更远处；并点出他「因为在移动的车顶熨衣服这一反常选择而引人注意」（due to his unconventional choice of ironing his clothes on top of a moving car），加上街道本身熙攘，构成了画面的独特性。幻灯片下方还给出了同一问题下其他模型的回答作为对照：GPT-4 [36] 说 "a man is ironing clothes on an ironing board attached to the roof of a moving taxi"（把熨衣板装在行驶的出租车车顶熨衣服）；BLIP-2 说 "a man is sitting on the back of a yellow cab"（一名男子坐在黄色出租车后部）；OpenFlamingo 说 "The man is drying his clothes on the hood of his car."（在车头引擎盖上晾衣服）。后两者明显识别错了，讲者口述也印证：GPT-4 当然做得到，但当时其他模型做不到。这些逐字回答都是图上独有、转写没念到的内容，也解释了为什么 LLaVA 在 2023 年会让人兴奋。

![LLaVA 对极端熨烫图像的多轮问答全文及与 GPT-4、BLIP-2、OpenFlamingo 的回答对比](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0029_00-35-21.jpg)

随后讲者把时间从 2023 跳到 2024 的 LLaVA OneVision，中间略过 LLaVA 1.5、LLaVA Next 等一系列后续论文，并说明会把这些创新都归并进接下来要讲的内容。他的定性是「essentially the same recipe」（基本同一套配方），但在多模态应用类型上「更有野心」（more ambitious）：现在能处理 multiple images（多图）与 video（视频），其中视频本质上就是采样帧（sampling of the frames）构成的图像序列。幻灯片上的两条 bullet 写的是 "Latest version in the LLaVA series (after LLaVA 1.5, LLaVA Next)" 与 "Handle multiple images, video"，图里也把输入画成 Single Image / Multi-Image / Video 三栏。

![LLaVA OneVision 架构图：SigLIP、Qwen-2、2-Layer MLP 与单图/多图/视频三类输入](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0030_00-36-44.jpg)

组件升级的三条 bullet 是本页最实在的信息："Vision encoder: SigLIP (use grid features before and after last Transformer layer)"、"Text decoder: Qwen-2 72B"、"Projector: 2-layer MLP"。也就是说，视觉编码器从 CLIP 换成 SigLIP，而且取的是最后一个 Transformer 层之前与之后的 grid features（网格特征）——注意这里不再是单一 pooled 向量，而是更丰富的多层网格特征；文本解码器换成 Qwen-2 72B，讲者口述称它大概是当时最好的语言模型；投影器（projector / adapter，即把视觉编码器输出转成语言模型输入的那部分）从线性层升级为两层 MLP。架构图上相应把符号改成 Projection $p_\theta$、Vision Encoder $g_\psi$、$Z_v$、$H_v$、$X_v$、$H_q$、$X_q$、Language Model $f_\phi$、Language Response $X_a$，并把视觉一侧统称 "Visual Signal"。于是上一章的 $H_v=WZ_v$ 在这里变成

$$
H_v=p_\theta\big(g_\psi(X_v)\big),\qquad X_a=f_\phi\big([\,H_v\,;\,H_q\,]\big)
$$

把「换零件」这件事写成配置差异：

```python
# LLaVA (2023)          -> LLaVA OneVision (2024)
vision_encoder = "CLIP ViT-L/14"   -> "SigLIP (grid features before/after last layer)"
text_decoder   = "Vicuna"          -> "Qwen-2 72B"
projector      = "Linear W"        -> "2-layer MLP p_theta"
inputs         = "single image"    -> "single / multi image / video"
```

讲者对此的评价是「你有一个系统，只是把零件升级了」（upgrading the parts），但「还是同一类粗略的系统」（the same kind of rough system）。易错点在于：不要把 OneVision 当成新架构，$f_\phi$ 接收拼接序列、只在输入侧做多模态这一点没变；也不要把 72B 误当成视觉编码器的规模，它指的是语言模型；另外 $p_\theta$ 已不是线性投影，参数量与非线性都上去了，因此上一章「只训一个 $W$ 做对齐」的简单说法在这里不再字面成立。局限方面，SigLIP 取多层网格特征意味着视觉 token 数会比 CLIP 的单一向量更多，这直接把压力传导到下一章要讲的分辨率与 token 预算问题。

## 数据处理、OCR 需求与 AnyRes 动态分辨率 [37:00 - 38:49]

最后一段转向 data processing（数据处理）。幻灯片 "Data processing:" 下的第一条 bullet 是 "Preserving high resolution is important (e.g., for OCR)"（保留高分辨率很重要，例如对 OCR 而言）。讲者把原因讲得很具体：OCR 需要保留非常 fine grain（细粒度）的信息，否则 "a J looks like a I" —— 字母 J 看起来就像 I，那就不行了。第二条 bullet 直指瓶颈："CLIP resizes and crops to 336x336, which loses information"（CLIP 会 resize 并裁剪到 $336\times336$，这会丢信息）。讲者口述补了一句：如果手里是一份文档，裁成 336×336 之后「你根本读不了它」（you can't read it）。这正好回扣本段开头对 CLIP 的判断 —— 它的设计决策基于图像分类，表示 not very fine-grained；也解释了为什么第 3 章末尾把编码器描述成「固定尺寸 $336\times336$ 进、一个向量出」时，其实已经埋下了文档类任务的隐患。

解法写在第三条与第四条 bullet："Solution: AnyRes, introduced in LLaVA 1.5 [paper]"（AnyRes，其实最早由 LLaVA 1.5 论文提出，并不是 OneVision 原创）与 "Break up image into a x b pieces (matching resolution of vision encoder), encode, concatenate"（把图像切成 $a\times b$ 块，每块的分辨率正好匹配视觉编码器期望的输入，逐块编码后再把向量拼接起来）。讲者的直觉解释非常朴素：既然视觉编码器处理不了高分辨率，与其 downsampling（下采样）把细节糊掉，不如干脆裁剪、分别去看图像的不同部分，再把各块编码出的向量 concat 成一条长序列。设单块经视觉编码器产生 $P$ 个视觉 token，则总视觉 token 数大致是

$$
N_{\text{vis}}\approx a\cdot b\cdot P\;(+\;P_{\text{base}})
$$

其中 $P_{\text{base}}$ 对应示意图里那条并行的全图缩略分支；本段没有给出 $a$、$b$、$P$ 的具体取值，不宜臆测。图上还有一条颜色较淡的 bullet："If too many tokens (original image is too high resolution), then use bilinear interpolation"（若原图分辨率过高导致 token 太多，就用双线性插值降采样），示意图也画出了「原图 → 切块 → 网格 → Bilinear Interpolation → LLM」这条链路。换句话说，AnyRes 不是无条件地保留所有细节，而是在 token 预算约束下做自适应折中。把流程写成伪代码：

```python
# AnyRes: 让视觉 token 数随图像分辨率自适应
def encode_image(image, a, b, base_res=336):
    pieces = split(image, a, b)              # a x b 块，每块匹配编码器分辨率
    tiles  = [SigLIP(resize(p, base_res)) for p in pieces]
    base   = SigLIP(resize(image, base_res))  # 全图缩略分支
    H_v    = concat(tiles + [base])
    if len(H_v) > token_budget:              # 分辨率过高、token 太多
        H_v = bilinear_interpolation(H_v)   # 降采样
    return H_v                               # 交给 LLM，长度可变
```

讲者在结尾把 AnyRes 的意义拔高到架构层面：这个方案之所以「有点意思」，是因为它是 adaptive（自适应）的，而 Transformer 本来就是自适应的 —— 语言里句子可以是任意长度，Transformer 已经处理得很好；「结果图像也可以是任意分辨率」（images can be any resolution）。也就是说，视觉输入不再被强行压成固定长度，而是和文本一样变成一条长度可变的 token 序列，两种模态在实现层面统一到同一个假设上。

![LLaVA OneVision 数据处理幻灯片：高分辨率对 OCR 的重要性、CLIP 336x336 裁剪丢信息与 AnyRes 分块方案](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0031_00-38-08.jpg)

这里有几个易错点和局限。第一，AnyRes 提高的不是「单次编码的分辨率」，而是「编码次数与 token 数」：每块仍然被送到期望 $336\times336$ 的编码器里，细节是靠多块覆盖保住的，代价是序列变长、注意力计算量上升，这才需要双线性插值降采样。第二，OCR 对细粒度极敏感，降采样阈值或 token 预算设得太紧，会直接把小字号字母糊掉，前面「J 看成 I」的问题又会回来。第三，$a\times b$ 的分块方式、全图缩略分支与降采样策略都引入了新的超参，而本段只给了思路没有给具体数值，不应凭常识补数字。第四，从本段整体看，AnyRes 是对 CLIP / SigLIP 固定分辨率编码器的一层「外挂式」补救，而不是改造编码器本身；它与第 1 章「理解只需粗略语义、生成与 OCR 需要细粒度」的对照形成闭环，也说明并不存在一个对所有任务都通用的图像编码粒度。

## AnyRes：把高分辨率图像切成块再拼接 [38:49 - 40:38]

本段一开头，讲者说要 "picking back on the same kind of dynamic ability"，也就是接回上一段 LLaVA OneVision 里那种"让视觉 token 数随输入变化"的动态能力。他拿一张论文页面的图像举例：实际做法有两条路径并行。第一条是整图 downsample（下采样），得到一份覆盖整页、分辨率较低的全局编码；第二条是把原图 break up into chunks（切成块），让每一块的分辨率正好等于 vision encoder（视觉编码器）所期望的输入尺寸，然后把每一块单独送进编码器。编码完的所有 patch 序列再 concatenate（拼接）成一个长序列喂给 LLM（大语言模型）。这个设计的直觉很直接：文档里的字、表格里的细线属于高频细节，必须按编码器原生分辨率去看才不丢；而整页的版式、段落结构又需要一眼看到全局，所以再用一张缩略图补上。讲者说得很简略，"they do down sample and have one path which is basically a down sampled encoding of the entire paper. But then they break it up into these chunks where each chunk is the resolution that the vision encoder expects and then encodes each of these separately. And then you concatenate all these patches"，但若 patch 太多，"you sort of reduce that the number of patches and by interpolation"。

幻灯片以 `Data processing:` 起头，逐条给出了转写里没有念全的理由链：`Preserving high resolution is important (e.g., for OCR)`；`CLIP resizes and crops to 336x336, which loses information`；`Solution: AnyRes, introduced in LLaVA 1.5 [paper]`；`Break up image into a x b pieces (matching resolution of vision encoder), encode, concatenate`；`If too many tokens (original image is too high resolution), then use bilinear interpolation`。这里有三个只在画面上的关键点：其一，AnyRes（任意分辨率）被明确标注为 LLaVA 1.5 引入、此处沿用，不是本段新发明；其二，分块写作 $a \times b$ pieces，且括号里强调 "matching resolution of vision encoder"，即块尺寸由编码器决定而非人为拍脑袋；其三，降 token 的手段是 **bilinear interpolation（双线性插值）**，讲者口述只说 "by interpolation"。幻灯片下方的流程图把这一步画在彩色方格网格之后、`flatten` 之前，并同时画出 `resize → encode → flatten` 那条全局路径，两条线最后都指向 `LLM`。

把单图的 token（词元）预算写清楚，就是

$$
T_{\text{single}} = 729 + N\cdot 729 = (1+N)\cdot 729,\qquad N \le 9,
$$

其中 $729$ 是视觉编码器对一张符合其分辨率的图（或一块）输出的视觉 token 数，$N$ 是 crop（裁切块）数，讲者口述 "up to n crops there up to nine"，幻灯片底部也直接给出 $(1+9)*729 = 7290$ Tokens 与 $729 + N*729$ Tokens 两行。对照前面 CLIP/SigLIP 章节，老做法是短边 resize 再中心裁剪成 $336\times336$，这一步在 OCR（光学字符识别）上是致命的：小字号文字被抹平，字母 "J" 很容易被读成 "I"。AnyRes 的代价是 token 数随 $a\times b$ 线性增长，所以才需要那个上限与插值。画面里那张彩色方格图还特意把 "Bilinear interpolation" 标在网格缩小的一帧上，说明压的是编码后的特征图。

这条"切块—分别编码—拼接—超预算就插值"的流水线，正是上一段 LLaVA 1.5 AnyRes 的落地，也和后面 Qwen2-VL 的 dynamic resolution（动态分辨率）、Native Resolution Input（原生分辨率输入）遥相呼应：后者把 $a\times b$ 的启发式推广成 token 数随图像高宽连续伸缩。易错点在于，插值降的是**编码后的特征网格**，不是重新采样原图，属于有损压缩；"最多 9 个 crop" 是上限不是额定值；而且这条精细路径只给单图用。讲者随即把话锋转到多图和视频：视频 "can be very long"，只用最多 32 帧也会撞上 context length（上下文长度）问题，因此 "a big part of being able to handle multimodal is to deal with long context"——这就引出下一章的 token 预算平衡。

![AnyRes 数据处理幻灯片：高分辨率保留、336×336 裁剪的信息损失、切成 a×b 块后编码拼接与双线性插值流程图](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0032_00-39-26.jpg)

## 单图、多图与视频：把 token 预算压到同一量级 [40:38 - 42:14]

讲者先给了一个判断："all of these are technically all reducible to images"，即单图、多图、视频技术上都能归约成图像，但作者 "put their thumb on a scale a little bit"——在天平上轻轻按了一下手指，刻意让各模态产出的 token 长度 roughly comparable（大致可比）。动机是数据配比：视频可以很长，如果不加干预，"they don't want their data set to be dominated by a bunch of repetitive frames"，即数据集会被大量重复帧淹没。幻灯片把这一节写成 `Handle 3 types of input (single image, multiple images, video):` 并给出唯一目标 `Goal: make all of the modalities produce roughly the same length`。这也解释了为什么上一章的 AnyRes 只对单图精细：不是不能对每帧都切块，而是预算不允许。

三条策略既在口述里、也在幻灯片第 168–170 行：`Single image: use higher resolution`——单图用全图下采样图加最多 9 个 crop，"If you have a single image I get to look at it more carefully"；`Multiple images: use base resolution for each image`——多图时 "I'm just going to look at it from afar"，每张图只取 base resolution（基础分辨率）；`Video: use lower resolution for each frame`——视频每帧用更低分辨率、更少 token，最多 32 帧。讲者的措辞很形象：单图看得近，多图看得远，视频看得最粗。这本质上是在 token budget（token 预算）上做歧视性分配，而不是让三种模态自然展开。

幻灯片中部那张 `Example on Token Strategy` 表格的 `Max Tokens` 列，是转写完全没念到的等式：

$$
\begin{aligned}
\text{Single-Image}:&\quad 729 + N\cdot 729 \ \Rightarrow\ (1+9)\cdot 729 = 7290\ \text{Tokens},\\
\text{Multi-Image}:&\quad N\cdot 729 \ \Rightarrow\ 12\cdot 729 = 8748\ \text{Tokens},\\
\text{Video}:&\quad N\cdot 196 \ \Rightarrow\ 32\cdot 196 = 6272\ \text{Tokens}.
\end{aligned}
$$

注意多图按 12 张图估、视频按 32 帧估，每帧只有 $196$ 个 token（约为单块 $729$ 的四分之一稍多），于是三种输入被硬压到 $6\text{K}\sim9\text{K}$ 的同一量级。表格左侧还分别标了 `... N Crops`、`... N Images`、`... N Frames`，与三个公式一一对应。这里的 $729$ 承接上一章，$196$ 则是视频降分辨率后的每帧 token 数，二者同时出现正好说明"同一视觉编码器、不同输入粒度"。讲者口述 "for video we're going to use even kind of lower resolution or fewer tokens to represent each frame"，幻灯片把这个"更低"量化成了 $196$。

接着是数据。`Data:` 一节的 slide 写的是 `Philosophy: quality over quantity`（讲者口述成 "curate high quality data and quantity"，并解释成 "very targeted data"，两者措辞略有出入，但都指向任务导向的精挑数据）。饼图标注 `Single-image 3.2M`，五个扇区分别是 `Camera`、`Doc/Chart/Screen`、`General QA`、`Math/Reasoning`、`Language`。右侧清单列出几十个单图数据集及样本数，转写里一个都没念，例如 `CLEVR (1.7 K)`、`A-OKVQA (66.2 K)`、`Image Textualization (99.6 K)`、`OKVQA (9.0 K)`、`DVQA (20.0 K)`、`VQAv2 (82.8 K)`、`VSR (2.2 K)`、`ST-VQA (17.2 K)`、`TallyQA (9.9 K)`、`Vision FLAN (186.1 K)`、`TextVQA (10.0 K)`、`ShareGPT4o (57.3 K)`、`AI2D (Image)(124.4 K)` 等，还有 `Language (14.3%)` 这样的占比标注。讲者据此判断 "this is definitely kind of post-training territory"：数据是围绕 VQA（视觉问答）、图表问答、"give two images and you're trying to spot the difference" 这类具体任务造出来的，所以 "very taskbased"。

这一段把上一章的"单图精细编码"放进了更大的数据工程里——token 预算不是模型结构决定的，而是被数据配比倒逼出来的。讲者也直说这项工作 "unabashedly basically distilling GPT4 models so that they can get the best performance"，并补了一句 "I guess not ideal but this is what you do if you don't have a annotation budget"，即在没有标注预算时用强模型蒸馏（distillation）合成数据。易错点是别把 `quality over quantity` 读成"数据少"：$3.2\text{M}$ 单图样本、几十个数据集、后面还有多图和视频，总量并不小；"quality" 指的是任务覆盖与标注来源的可控。另一个局限是这种平衡是手工设定的常量（9、12、32、196、729），换一个编码器就得重调；讲者自己也说这是"put their thumb on a scale"，带有明显的人为干预。

![三类输入的 Token 策略表：单图 7290、多图 8748、视频 6272 Tokens 及由易到难的三条分辨率规则](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0033_00-40-38.jpg)

![LLaVA OneVision 数据理念幻灯片：Philosophy: quality over quantity、Single-image 3.2M 饼图与单图数据集清单](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0034_00-41-51.jpg)

## 三阶段训练：由易到难的课程 [42:14 - 43:30]

讲者把训练拉回 "roughly the same idea as before"：第一阶段专注 alignment（对齐），"you only train the projector which is the adapter"，即只训练 projector（投影器，也叫 adapter 适配器）；随后第二、三阶段逐步放开，最后训练 full model。与前面 LLaVA 的两阶段相比，这里变成三阶段："before we had two stages, now we have three stages"。他坦白说 "I'm not sure there's any particular principle reason for this"，唯一的解释是第二阶段塞入 high quality data、更偏 knowledge（知识），最终阶段则用 "examples that look like your downstream tasks" 的样本。幻灯片给出的口号比口述更凝练：`Training: Philosophy: easier to harder`。这张表把"三阶段"落成了可核对的超参数，是本段信息密度最高的一页。

流程图把三阶段画成 `Stage-1 Language-Image Alignment` → `Stage-1.5 High-Quality Knowledge Learning` → `Stage-2 Visual Instruction Tuning`。注意中间那一段被命名为 Stage-**1.5**，而不是 Stage-2，这正是"从两阶段插进一段"的痕迹，也解释了为什么讲者说不出严格原理。表格按 `Vision / Data / Model / Training` 四组列出全部配置，这些数字转写里几乎都没念：分辨率一栏从 Stage-1 的 $384$ 扩到 Stage-1.5 的 $384\times\{2\times2,\,1\times\{2,3\},\,\{2,3\}\times1\}$，再到 Stage-2 的 $384\times\{\{1\times1\},\dots,\{6\times6\}\}$；`#Tokens` 对应 $729$、$\text{Max }729\times5$、$\text{Max }729\times10\ (\text{See Fig. 3})$。可以看到 Stage-2 的分辨率集合已经覆盖了直到 $6\times6$ 的分块，恰好对应 AnyRes 的 $a\times b$。

Data 与 Model 两行：`Dataset` 依次为 `LCS`、`Image (Sec. 4.1)`、`Image (Sec. 4.2)`、`(Multi)-Image & Video (Sec. 4.2)`；`#Samples` 依次是 $558\text{K}$、$4\text{M}$、$3.2\text{M}$、$1.6\text{M}$——其中 $3.2\text{M}$ 正好对应上一章饼图的 `Single-image 3.2M`。`Trainable` 一栏从 `Projector` 变成三列的 `Full Model`；按 LLM 规模列出的可训练参数量是 $0.5\text{B LLM}\to 1.8\text{M}$、$7.6\text{B LLM}\to 20.0\text{M}$、$72.7\text{B LLM}\to 72.0\text{M}$（Stage-1 只训投影器），进入 Full Model 后变成 $0.8\text{B}$、$8.0\text{B}$、$73.2\text{B}$。Training 组：`Batch Size` 由 $512$ 变为 $256/512$；`LR: \psi_{vision}` 与 `LR: \{\theta_{proj}, \phi_{LLM}\}` 在 Stage-1 同为 $1\times10^{-3}$，之后降为 $2\times10^{-6}$ 与 $1\times10^{-5}$；`Epoch` 全程为 $1$。

用一条式子概括阶段间的自由度变化：

$$
\Theta_{\text{train}}^{(1)}=\{\theta_{\text{proj}}\}\ \subset\ \Theta_{\text{train}}^{(1.5)}=\Theta_{\text{train}}^{(2)}=\{\psi_{\text{vision}},\theta_{\text{proj}},\phi_{\text{LLM}}\}.
$$

学习率之所以从 $10^{-3}$ 掉到 $10^{-6}\sim10^{-5}$ 量级，是因为 Stage-1 只训一个随机初始化的投影器，允许大步；后面放开整个 LLM 与视觉编码器，必须小步。这与前面 LLaVA"冻结视觉编码器、只训 $W$ 与 LLM"的两阶段是同一套 alignment 思路，只是多插了一段高质量知识学习，并且把 Stage-2 再拆成单图与多模态两列。

关系与易错点：这里的 $558\text{K} \to 4\text{M} \to 3.2\text{M} \to 1.6\text{M}$ 不是单调变多，说明 "easier to harder" 指的是**任务难度与数据质量**，不是样本数量；Stage-1.5 的 $4\text{M}$ 反而是最大一份。另一处易错：Stage-2 被拆成 `Single-Image` 与 `OneVision` 两个子列，前者只用图像、后者才引入多图与视频，所以"三阶段"其实是四份配置。局限也明显：所有阶段 `Epoch = 1`，且讲者明说三阶段"未必有原则"，更像工程上的经验切分，后面 Qwen-VL 与 Qwen3-VL 会把它继续膨胀到四阶段预训练加三阶段后训练。

![LLaVA OneVision 三阶段训练表：Stage-1 / 1.5 / 2 的分辨率、样本数、可训练参数、学习率与 epoch 配置](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0035_00-43-04.jpg)

## 跨模态迁移：任务式数据也能泛化 [43:30 - 46:04]

讲者说这篇论文里 "kind of interesting" 的发现是 transfer between modalities（跨模态迁移）。最反直觉的一例：训练时只有单图的 diagrams and charts 数据，"even though they have only single image data for diagrams and charts but this actually generalizes to the multiple images"；从没见过"一张表加一张图"同时出现并提问，但测试时给它两张图（一张 table、一张 chart），它能就两者展开对话。换句话说，模型把单图上学到的能力组合到了多图场景。转写里 "at test time it's ne at training time it never saw um example" 的磕巴，强调的正是训练/测试分布不一致却仍能作答。

第二个例子幻灯片第 184 行给出了原文：`OCR on single image data, relational reasoning from multi-image data, generalize to GUI-based agents`。训练时 OCR 只在单图、relational reasoning（关系推理）只在多图，两者各自独立；测试时却能支撑 GUI agents（图形界面智能体）——"you show it a bunch of screenshots this is multiple images and it can generate a description of the screenshots"。幻灯片下方就是 `S2: Multi-image GUI Understanding for Agents` 的示例：几张 TikTok 手机界面（Search、Charts、Suggested、Events、键盘、手机边框）并排输入。第三个例子是 visual prompting（视觉提示）：图上画一个圆圈来圈出"你该关注的部分"，这类数据只在单图存在，却能泛化到视频里 "describe the player highlighted in the video"——被圈出的球员横跨多帧，模型仍要跟踪并描述。

画面里还有一段转写完全没念的完整推理样例，可用来说明"迁移出去的能力有多细"。用户给一张带褐色扇区的房子图，问按每单位面积保费算整栋房的保险费用，LLaVA-OV 的解法用扇形面积公式

$$
A=\frac{\theta}{360}\,\pi r^{2},\qquad \theta=40^\circ,\ r=11\ \Rightarrow\ A=\frac{40}{360}\pi\cdot 11^{2}\approx 38.01,
$$

再乘单价得到 $38.01\times 63 \approx 2{,}386.03$ 美元。紧接着的视觉提示示例页上，LLaVA-OV 对足球视频中被高亮的球员写出 "wearing a white kit, and the number '10' is visible on their shirt ... positioned in the center of the pitch ... surrounded by opponents in red kits ... possibly a playmaker or attacking midfielder" 这样的长描述。这些说明模型输出的仍是文本，不是图像或框序列本身。

讲者的反思值得记下：他第一眼觉得 "oh boy, you're basically targeting each of these tasks, is kind of like supervised learning"——每个任务都专门造数据，看着就像回到了监督学习；但 "if you have enough tasks, these models seem to do some transfer"，任务足够多时会出现迁移，"which is reassuring"。随后 `Summary:` 一页把整段 LLaVA 系列收束成三条：`Standard VLM template: vision encoder + projector + LM`、`Most work goes into data curation (heavy on synthesized, task-specific data)`、`Open-source (released model weights and data)`。讲者特别肯定 LLaVA 系列 "one of the few works that open sources ... not just the model weights but also the data"，所以 "you can really replicate and study this stuff"。

这一章是 LLaVA OneVision 部分的收尾，把前两章的 token 策略与三阶段训练落到"数据能带来什么"上，也顺势过渡到 Qwen 系列——后面会看到同一套 `vision encoder + projector + LM` 模板被反复复用，差别主要在编码器、位置编码和训练管线。易错点是别把 transfer 说成"零样本涌现"：这些能力都在合成任务数据覆盖范围内被间接训练过，只是组合方式没出现过；而且示例中的 bounding box（边界框）、OCR 结果都以文本形式输出。局限是讲者对迁移只给了定性观察，没有量化指标，且承认数据高度依赖 GPT-4 蒸馏。

![跨模态迁移与 GUI 智能体示例：单图 OCR + 多图关系推理泛化到多截图 GUI 理解，以及扇形面积计算样例](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0036_00-44-17.jpg)

## Qwen-VL：固定 256 token 的 cross-attention 适配器 [46:04 - 49:16]

46:01 讲者 "Now let's talk about Qwen models"。Qwen 也是 2023 年开始做多模态，首版 Qwen-VL [Bai+ 2023]。他提醒 "hopefully see the pattern"：仍是 vision encoder + projector（此处叫 adapter）+ LM 的模板。视觉编码器用 OpenCLIP，讲者口述 "open clip was a open reproductions of a clip model"，幻灯片精确到 `OpenCLIP's ViT-bigG (14x14 patches) [Cherti+ 2022]`。适配器是 `one layer cross-attention, incorporate 2D positional encodings, maps to fixed length of 256`——一层 cross-attention（交叉注意力），引入 2D 位置编码，把任意图像压成**固定 256 个 token**。讲者随即点评 "this is definitely not very dynamic"，但也补了一句此时视觉编码器本身也不动态，所以这一版还算自洽；这个 $256$ 到 Qwen2-VL 就会被动态分辨率取代。

特殊 token 一节讲者说得很含糊："they have an image tag and a box tag for bounding box and a ref tag for description. This is what happens when your slides are in HTML." 幻灯片上确实有一行 `Special tokens:` 后面跟着几个空位——因为整份幻灯片是 HTML 写的，形如标签的特殊 token 被浏览器当成标签吞掉了，没有渲染出来。可确认的是三类用途：标记图像、标记 bounding box（边界框）、标记 ref（描述/指代）。这也解释了为什么 Qwen-VL 能输出框："It doesn't actually output an image. It just outputs a bounding box"，即输出的是框的文本坐标，不是生成图像。

幻灯片用 `def qwen_vl():` 作章节标记，训练三阶段在图上写成 `Stage1: Pretraining`、`Stage2: Multi-task Pretraining`、`Stage3: Supervised Finetuning`，配图里用火焰/雪花标出可训练与冻结模块（`Learnable Query Embs` → `CrossAttn` → `ViT` → `QwenLM`）。把这些标题与口述的配置对齐，可整理为：

```python
def qwen_vl():  # Qwen-VL [Bai+ 2023]（幻灯片章节标记）
    # Architecture（幻灯片原文）
    vision_encoder = "OpenCLIP's ViT-bigG (14x14 patches) [Cherti+ 2022]"
    adaptor = "one layer cross-attention, 2D positional encodings, fixed length of 256"
    # Training（幻灯片三阶段标题 + 讲者口述的冻结/训练分工）
    stage1 = "Pretraining"             # freeze LM, train vision encoder + adaptor
    stage2 = "Multi-task Pretraining"  # higher quality task-specific data, train all parameters
    stage3 = "Supervised Finetuning"   # freeze visual encoder, train adaptor + LM
```

讲者强调 Stage-1 "they call it pre-training but it's really you're not pre-training from scratch"，且 "a bit different from the LLaVA models in that they train the vision encoder"。

数字全在画面上。Stage-1 数据表（转写只含糊说 "1.4 billion examples"）列了 `Language | Dataset | Original | Cleaned | Remaining%`：英文侧 `LAION-en` $2\text{B}\to280\text{M}\ (14\%)$、`LAION-COCO` $600\text{M}\to300\text{M}\ (50\%)$、`DataComp` $1.4\text{B}\to300\text{M}\ (21\%)$、`Coyo` $700\text{M}\to200\text{M}\ (28\%)$、`CC12M` $12\text{M}\to8\text{M}\ (66\%)$、`CC3M` $3\text{M}\to3\text{M}\ (100\%)$、`SBU` $1\text{M}\to0.8\text{M}\ (80\%)$、`COCO Caption` $0.6\text{M}\to0.6\text{M}\ (100\%)$；中文侧 `LAION-zh` $108\text{M}\to105\text{M}\ (97\%)$、`In-house Data` $220\text{M}\to220\text{M}\ (100\%)$；`Total` 为 $5\text{B}\to1.4\text{B}\ (28\%)$。清洗保留比例可以写成

$$
\text{Remaining\%}=\frac{\text{Cleaned}}{\text{Original}}\times100\%,\qquad \frac{1.4\text{B}}{5\text{B}}=28\%.
$$

另一页幻灯片上的 Stage-2 任务表则给出 `Captioning 19.7M`、`VQA 3.6M`（数据集含 GQA、VGQA、VQAv2、DVQA、OCRVQA、DocVQA、TextVQA、ChartQA、AI2D 等）、`Grounding 3.5M`（GRIT）、`Ref Grounding 8.7M`（GRIT、Visual Genome、RefCOCO、RefCOCO+、RefCOCOg）。

能力示例页把结果摆成六格，转写只挑了几句：中文人物识别并带框、医院楼层指示牌（4F/3F/2F）的 OCR 加楼层推理、Chicago 与上海东方明珠两图对比、把图里的 JavaScript 代码转成 `int solution(int A[], int N)` 的 C 实现、"Can you find spider man and hulk?" 的 grounding（视觉定位）、以及 `OCR this picture` 逐字抄出 Qwen-VL 论文摘要。讲者点出动机："a language model can do code. So we want the vision model to be good at code as well"，且因为面向中文，例子里有不少中文。关系上，Qwen-VL 与前面 LLaVA 是同一模板的独立实现，差别在适配器（cross-attention + 固定 256）与训练分工（Stage-1 训视觉编码器）。易错点：固定 $256$ 意味着高分辨率细节必然被压掉，OCR/文档能力受限；Stage-3 冻结的是视觉编码器而非语言模型；三阶段里 "pre-training" 只是命名，不是从零预训练。

![Qwen-VL 架构幻灯片：OpenCLIP ViT-bigG (14x14 patches)、一层 cross-attention 映射到固定 256 长度，以及三阶段训练流程图](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0038_00-46-42.jpg)

## Qwen2-VL：动态分辨率与 M-RoPE [49:16 - 52:50]

49:16 讲者转入 "Qwen 2 ... a kind of upgrade"，并说 "there's some new ideas here"。两点最关键：更大的 vision encoder，以及 dynamic resolution（动态分辨率）。他把动机接回 LLaVA→LLaVA OneVision 的认识："we realized oh we need to handle images of different sizes and if you try to do video it's clear that you need some sort of dynamic resolution"。画面上 `Vision Encoder` 框上方标注 `Native Resolution Input`，并摆了三张尺寸极不相同的图：Picture 1 高 8204、宽 1092（长文档），Picture 2 高 28、宽 224（一条细横幅），Picture 3 高 700、宽 1260；Video 1 高 336、宽 644、时长 16s。讲者口述的对应例子是：一张图可能映射到约 $11{,}000$ 个 token，而一张只有公式的小图可能只映射到 $8$ 个 token——token 数随分辨率自适应，这与文本长度自适应是同一个道理。

bullet 原文（幻灯片第 219–221 行）：`Key: dynamic resolution to handle varying resolutions`；`Each 224 x 224 patch encoded with ViT/14, compress every 2x2 => 66 tokens`；`Video: sample 2 frames/sec, max 16384 tokens`。所以做法仍是上一章 AnyRes 式的分块——讲者口述 "it's a kind of the same idea of the AnyRes idea from lava"——但块尺寸固定为 $224\times224$，用 ViT/14 编码，起始权重是 OpenCLIP ViT 再 fine-tune；为压 context size，把每 $2\times2$ 个相邻压缩成 1 个，最终每 patch 产出 $66$ 个 token。视频每秒采样 2 帧，上限 $16384$ 个 token（讲者口述 "max out at 16,000 you know tokens"）。写成流程：

$$
224\times224 \xrightarrow{\ \text{ViT/14}\ } \text{patch tokens} \xrightarrow{\ 2\times2\ \text{compress}\ } 66\ \text{tokens}.
$$

位置编码是本段真正的结构改动——M-RoPE（Multimodal Rotary Position Embedding，多模态旋转位置编码）。讲者先复习 RoPE：设计一种 embedding，使两向量内积只依赖相对距离，"the inner product between the vectors kind of depends only on the kind of distance"，一维里距离就是相隔的 token 数，即 $\langle f(x_i),f(x_j)\rangle = g(i-j)$。多维版本把这个"距离"从 1D 扩到 3D：每个 patch 的位置变成一个由 height（高）、width（宽）、time（时间）组成的三元组。计算时对每个维度分别算 RoPE 再拼接：

$$
\text{M-RoPE}(t,h,w)=\text{concat}\big(\text{RoPE}_{t}(t),\ \text{RoPE}_{h}(h),\ \text{RoPE}_{w}(w)\big).
$$

幻灯片配图用 `Time / Height / Width` 三个轴画出视频帧网格，并配一句 `This video features a dog, specifically a Shiba ......` 表示文本 token 也进入同一位置编号体系。讲者评价 "it's a fairly straightforward idea"，但马上埋了伏笔："although later with Qwen 3, we'll see how this is actually a bit sub-optimal"。

其余配置：幻灯片底部一行写 `Initialize LM with Qwen2 and vision encoder from DFN [Fang+ 2023]`（转写只说 "initialize the language model with ... this is the QwenVL model. So of course they're initializing with Qwen"），训练 "similar to Qwen-VL" 仍是三阶段。`Many capabilities:` 一页把能力铺成放射状：`General Chat`、`Video Understanding`、`Grounding`、`Multilingual OCR`、`Long Document Understanding`、`Math & Code`、`Live Chat`、`Formula Recognition`、`Function Calling`、`UI Interaction`，以及一个 `Beyond`；转写里只笼统提到 "some video understanding ... it can do math and code function calling and so on so forth"。这些名字说明动态分辨率与原生输入把 OCR、长文档、公式识别这类细粒度任务正式纳入了能力面。

关系与局限：Qwen2-VL 把 LLaVA OneVision 的 $a\times b$ 分块与 Qwen-VL 的固定 256 这一对矛盾统一了——块尺寸固定（$224\times224$）、块数量可变，于是 token 数动态；M-RoPE 则第一次让时间维度进入位置编码，为视频与长上下文铺路。易错点：$66$ 是压缩后每 patch 的 token 数，不是原图总 token 数；$16384$ 是视频总上限，不是每帧；RoPE 的"每个维度分别算再拼接"在 Qwen3-VL 会被证明有频率分配问题。另外讲者进入 Qwen3 时说那些改动 "not kind of structural big changes"，但 Qwen2 相对 Qwen-VL 在分辨率与位置编码上的改动其实相当实质，不要混为一谈。

![Qwen2-VL 动态分辨率与 M-RoPE 幻灯片：224×224 patch、ViT/14、2×2 压缩到 66 tokens、视频 2 帧/秒上限 16384 tokens，以及 Time/Height/Width 三轴位置编号](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0042_00-51-34.jpg)

## Qwen3-VL：长上下文、交错 M-RoPE 与 DeepStack [52:50 - 58:14]

52:50 讲者 "quickly I'll talk about Qwen 3"，并定性："a few kind of changes here. Um I wouldn't say these are kind of structural big changes but they are probably changes that do impact the quality of the model"，来源是去年的 Qwen3-VL 报告 [Bai+ 2025]。第一处是语言模型换成 Qwen3 系列的 dense（稠密）与 MoE（混合专家）模型，"the Qwen 3 models are really really good and it really helps the quality of the final model"；幻灯片给出具体规模 `Qwen-3 models (dense and MoE models up to 235B-A22B)`。第二处是 `Long context understanding (256K)`——上下文长度可到 $256\text{K}$，讲者说这对 "trying to do long video" 非常重要，也正好呼应本段开头 "a big part of being able to handle multimodal is to deal with long context" 的判断。

视觉编码器换成 `SigLIP-2 (same architecture as SigLIP) [Tschannen+ 2025]`，讲者口述 "basically improved version of SigLIP. It's actually an identical architecture is designed to be backward compatible"——架构完全相同、向后兼容。真正精妙的是对 M-RoPE 的修正。旧做法把维度按块切给三个轴："allocate the first block of dimensions to the time then the next block to width and the next to height"；但 RoPE 里 "each component refers represents a different frequency"，于是可能出现 "all the temporal dimensions are low frequency and all the height dimensions are high frequency"。幻灯片把修正写为 `Interleaved MRoPE: distribute all axes(temporal, width, height) to low- and high-frequency bands`，并给出交错前后对比：`[t w h t w h t w h]` rather than `[t t t w w w h h h]`——把三个轴轮流排布，"all the axes are exposed to both low and high frequency"。

另外三处改动都在画面上。其一是 `Add explicit video timestamps (as separate tokens rather in positional embeddings)`：以前时间隐含在位置编码里（"each frame in a video intrinsically gets a different position"），现在把时间做成显式 token，讲者举例 "token 0 seconds is a token that represents the time"，好处是可以直接引用 "what happened at after two seconds"。架构图里因此出现 `<0.0 seconds>`、`Timestamps in text format` 以及 `Video Frame with "0.0 seconds"` 这类标注，序列上还标出 $11427$ tokens（Picture 1）、$8$ tokens（Picture 2）、$1125$ tokens（Picture 3）三档差异极大的长度。其二是 `Square-root-normalized per-token loss`（平方根归一化的逐 token 损失）：视频样本极长、单图样本极短，若每个 token 等权，"the video examples are going to dominate"；讲者明确说 "the details aren't too clear but I believe they normalized each example by the square root of the length"，若用式子示意可写作

$$
\mathcal{L}_{\text{example}}=\frac{1}{\sqrt{N}}\sum_{i=1}^{N}\ell_i,
$$

其中 $N$ 为该样本 token 数——注意这是讲者的推测而非论文确认的公式，属于本段最需要留意的易错点。

第三处，也是讲者说 "interesting with adapter" 的收尾：适配器演化史是 `linear projection`（LLaVA 最简单）→ `MLP` → `cross attention` → `DeepStack`，其中 DeepStack 是 DeepSeek 团队的一篇论文，此处用法 "more sophisticated"。幻灯片写作 `DeepStack: cross-layer fusion to inject visual information into multiple layers [Meng+ 2024]`（深层堆叠融合，跨层注入）：视觉编码器本来就产出 "a stack of vision embeddings"，与其把它当黑盒只吐一串向量，不如把多层特征直接加进语言模型的 residual stream（残差流），若用式子示意即

$$
\mathbf{h}^{(l)} \leftarrow \mathbf{h}^{(l)} + \mathbf{v}^{(l)},\qquad l\in\{1,2,3,\dots,N\}.
$$

架构图右侧虚线框里 `LLM Block 1`、`LLM Block 2`、`LLM Block 3` 直到 `LLM Block N` 都接受 `Vision tokens`，这就是 "a bit more of a kind of deep fusion of the vision encoder into the language model as opposed to the vision encoder is a black box that just outputs a sequence of vectors"。

训练管线随之膨胀："they have two phases. Um pre-training has actually four stages, post- training has three stages. So this you know you can see that this pipelines are getting quite complicated now"。幻灯片写 `Pre-training has 4 stages (train adapter, train all parameters on 8K, 32K, 256K lengths)`，表格首行可见 `S0 | Vision-Language Alignment | Merger | 67B | 8 192`——即先训 adapter（图中称 Merger），再依次把序列长度从 $8\text{K}$ 拉到 $32\text{K}$、$256\text{K}$。讲者最后一句停在 "then they have three stages where they progressively train on" 便被本段截断，因此不要替他补后训练的细节。关系上，Qwen3-VL 是整段"模板不变、逐步打磨"的终点：编码器换成向后兼容的 SigLIP-2、位置编码从分块改交错、时间从隐式改显式、损失按长度归一、适配器从浅投影改深融合、上下文拉到 $256\text{K}$。局限也清楚：讲者反复强调这些不是结构大改，多是数据、长度与频率分配上的工程改良；$235\text{B-A22B}$、$256\text{K}$、$8\text{K}/32\text{K}/256\text{K}$ 这些数字都依赖具体配置，且 per-token loss 的具体形式他本人也没完全确认。

![Qwen3-VL 架构图：Qwen3 LM Dense/MoE Decoder、显式时间戳 token、11427/8/1125 tokens 的原生分辨率输入，以及向 LLM Block 1/2/3/N 注入 vision tokens 的 DeepStack](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0047_00-57-38.jpg)

## Qwen3-VL 收尾：长上下文训练管线与基准成绩 [58:14 - 59:21]

本段开头承接前面几讲对 Qwen-VL 系列的梳理，把 Qwen3-VL 的预训练管线收束完。讲者说预训练是分阶段把序列长度从 8k 拉长到 32k、再拉长到 256k，即 $8\text{k}\to 32\text{k}\to 256\text{k}$，并强调「most of the tokens are in the stages two and three」，也就是说绝大部分 token 消耗发生在第二、第三阶段，而不是第一阶段。这一点与前一段讲过的 Qwen3-VL 支持最长 256K 上下文互相印证：长上下文能力不是声明出来的，而是靠训练时真的把长序列喂进去换来的，尤其长视频会自然产生超长 token 序列。post-training（后训练）部分则在「long chain of thought data」上做 SFT（supervised fine-tuning，监督微调），再做 knowledge distillation（知识蒸馏），最后加一些 reinforcement learning（强化学习）。讲者随后给了一句很重要的定性判断：「at this point it's really kind of a systems paper」，意思是到这里已经更像一篇系统论文，他只高亮了少数核心新想法，而「there's a lot of details that are are different」，大量工程细节与之前不同。这里的易错点是：不要把这套四阶段预训练加三阶段后训练误读成新的算法贡献，它的主体是数据、长度调度和系统实现，而非新的建模范式。

紧接着讲者把结果摆出来，说「if you look at the final results this is a pretty pretty good model」，并点名要与闭源模型 Gemini、GPT-5、Opus 4.1 对比。转写只粗略念了这三个名字，而画面 kf_0048 给出了完整的表格：被比较的对象是 Qwen3-VL 235B-A22B（分 thinking 与 instruct 两列）、Gemini 2.5 Pro（分 thinking 与 budget-128 两列）、OpenAI GPT-5（分 high 与 minimal 两列）以及 Claude Opus 4.1（分 thinking 与 non-thinking 两列），共八列数值；行按任务被划分成 STEM Puzzle、General VQA、Alignment、Document Understanding 四大组。讲者特别解释了读表规则：「the bold means that it's the best number in that row」，加粗的数值就是该行的最佳成绩，并据此说「the Qwen models are actually quite quite strong」。换句话说，这张表不是要证明 Qwen 全面碾压，而是逐行看谁最好，Qwen 在很多行上确实拿到了加粗。

![Qwen3-VL 235B-A22B 与 Gemini 2.5 Pro、GPT-5、Claude Opus 4.1 在四大类基准上的逐行对比表](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0048_00-58-56.jpg)

表格里的原样数值值得逐行摘出几组，因为这些数字转写完全没有念到。STEM Puzzle 组里 MMMU 一行，Qwen3-VL 为 80.6 / 78.7，Gemini 2.5 Pro 为 81.7* / 80.9，OpenAI GPT-5 为 84.2* / 74.4*，Claude Opus 4.1 为 78.4 / 77.2；MMMU-Pro 为 69.3 / 68.1 / 68.8* / 77.2 / 78.4* / 62.7* / 64.8 / 60.7；MathVista_mini 为 85.8 / 84.9 / 82.7* / 71.2 / 81.3 / 50.9 / 75.5 / 74.5；ZeroBench 这种极难题上所有模型都只有个位数，为 4 / 2 / 3 / 1 / 2 / 2 / 3 / 1。General VQA 组里 MMBench-EN 为 88.8 / 89.3 / 90.1* / 88.4 / 83.8 / 81.3 / 79.4 / 83.0。Alignment 组里 MM-MT-Bench 为 8.5 / 8.5 / 8.4* / 7.6 / 7.6 / 7.5 / 7.8 / 7.9。Document Understanding 组是 Qwen 明显占优的地方：DocVQA_test 为 96.5 / 97.1 / 92.6 / 94.0 / 91.5 / 89.6 / 92.5 / 89.2，InfoVQA_test 为 89.5 / 89.2 / 84.2 / 82.9 / 79.0 / 69.9 / 69.4 / 60.9，OCRBench 为 875 / 920 / 866 / 872 / 810 / 787 / 764 / 750，OCRBench_v2_zh 为 66.8 / 67.1 / 54.3 / 55.2 / 53.0 / 48.2 / 48.4 / 47.2，CC-OCR 为 81.5 / 82.2 / 77.2 / 76.8 / 68.3 / 66.1 / 69.1 / 66.0。部分数值带有 `*` 上标，画面并未解释其含义，此处只照录不猜测。

把这张表与前后内容连起来看，它是讲者说「quite strong」的直接依据，也呼应了前一段提到的 SigLIP-2、DeepStack、显式视频时间戳等改进最终换来的成绩。但读表有两个易错点。第一，八列对应的是不同推理配置——thinking 与 instruct、high 与 minimal、thinking 与 non-thinking，横向比较必须对齐列，不能拿某一列去和另一模型的另一列比。第二，讲者自己把它定性为 systems paper，并说「a lot of details that are different」，所以不能把 SOTA 归因于某个单独的模块；后面 66:22 他还会明确说这些后期 Qwen 模型「not too many details about the data and the data mix」。因此这张表更合适的用法是确认「结果与闭源模型可比」，而不是推断 recipes。它也为本段最后的转折埋下对照：接下来 Chameleon 恰恰会因为「not as performant」而被讲者放弃展示结果。

## 答疑：只生成文本、系统开销与 token 配比 [59:21 - 62:56]

从 59:55 开始进入课堂问答，第一个问题问的是「what about video generation and how do you know whether there it should be a video or text」，也就是模型怎么决定该输出视频还是文本。讲者的回答直接划清了本讲的范围：「these models don't generate videos or images. The video all the multimodal stuff is on input side. you're always generating text」。换句话说，前面所有 LLaVA、Qwen-VL 这类 VLM 都是多模态输入、纯文本输出，理解（understanding）与生成（generation）并不对称，本讲讨论的多模态几乎全在输入侧。这一点很重要，因为它解释了为什么前面几段都只谈「怎么把图像塞进 Transformer」而不谈「怎么让 Transformer 画出来」。监督形式也随之确定：转写说「in most of these stages except for RL you're supervising every token」，可形式化为标准的逐 token 自回归损失 $\mathcal{L}_{\text{SFT}}=-\sum_{t=1}^{T}\log p(x_t\mid x_{<t})$。讲者强调「there's no LM as a judge or quality on how well the description is working. It's just basically whatever is in your data set」，也就是说除 RL 外没有额外的裁判模型或描述质量打分，质量完全由数据集决定；只有「with RL you can play around with different reports」时才会引入不同奖励。易错点正在这里：别把监督微调阶段的 caption 质量想象成有独立奖励模型在打分，也别因为模型能「看图」就以为它能输出图像。

第二个问题从 60:40 提出、61:20 被讲者复述为「is the training of multimodal models harder on the system side」，他回答「certainly it's not easier」。第一层原因是数据体积：数据集更大，「video data is just like you know even loading it can be [a] bottleneck」，光是把视频读进来就可能成为瓶颈。讲者顺带对比了课程此前的取向：讲纯语言模型时「we have not really focused on data loading ... because it's very very cheap」，因为文本加载几乎不花钱，而多模态必须重新把 I/O 当回事。给出的原则是「making sure that your data loading is happening async with actual computation needs to be taken into account」，即数据加载要与真正的计算异步重叠，用符号写就是 $\text{load}(t+1)\ \parallel\ \text{compute}(t)$，而不是 $\text{load}(t)\to\text{compute}(t)$ 的串行。这一层没有给吞吐数字，属于定性原则，但它把前一段 LLaVA OneVision 里「视频每帧降到 196 token、最多 32 帧」这类设计背后的系统动机讲清楚了：控 token 数不只是为了建模，也是为了让数据管线跑得动。

问题的后半在 62:02 被听众补了一句「They have more tokens」，讲者承认「videos and images can have more especially video have more tokens」，并把这与此前见过的归一化联系起来：「that's one reason we saw this kind of normalization to make sure that we're not giving more weight to video」。他的态度很实用主义：「if you have more tokens, if you don't like that, you can downweight」，并回指两周前讲的 data mixtures（数据配比），说可以按需要给不同来源加权，用式子可写成 $\mathcal{L}=\sum_m \lambda_m\,\mathcal{L}_m$，其中把 $\lambda_{\text{video}}$ 调小即可。但紧接着他给了一个容易被忽略的反驳：「there's also a lot of text tokens out there ... the models are trained on like tens of trillions of tokens ... I wouldn't say that the number of multimodal tokens like vastly outnumbers text tokens」。也就是文本语料的基数是数十万亿（tens of trillions）级别，多模态 token 未必在总量上远超文本 token。易错点是：不能因为单条视频 token 多，就推断视频必然主导训练；决定主导性的是总 token 占比，而这个占比可以通过降权显式控制。

这组答疑在整段里起的是「系统注脚」的作用：它把前面几段反复出现的 token 策略、按样本长度归一化的 per-token loss、视频帧下采样等做法，统一归因到两件事——数据加载的 I/O 开销与模态间的权重平衡。同时它也提前埋下后面 Chameleon 的伏笔：讲者说多模态配比需要「play around with the weights」，到 73:54 会讲这在全离散模型里「more exacerbated」。局限同样明显：整个回答没有给出任何具体的吞吐数值、显存数字或 $\lambda$ 的取值，只有「异步加载」「降权」「归一化」三条原则；因此这里能确定的是方向，而不是可复现的超参。

## 答疑：对齐方式与参数规模 [62:56 - 66:14]

第三个问题在 63:22 提出：「when you do this alignment how does it know what the language is」，即对齐时语言模型本身从哪来。讲者的回答很干脆：「the language model is pre-trained and it's definitely has to be pre-trained because otherwise there's it doesn't make sense to align it」。这里的 alignment（对齐）沿用的是 LLaVA 语境下的狭义定义：把「a given vision encoder」与「a given language model」接起来，其中「the language model is frozen and you're just training the adapter to connect the given vision encoder with a given language model」。可形式化为 $h_{\text{vis}}=W\cdot\mathrm{ViT}(I)$，其中 $\mathrm{ViT}(I)$ 是冻结视觉编码器给出的图像特征，$W$ 是唯一可训练的 adapter / projector，把视觉向量映射到语言模型的嵌入空间；语言模型参数保持冻结。训练方式也很朴素：「you pick a token budget 67 billion tokens and you just train. There's not a like adaptive threshold here」，即先定一个 670 亿（67 billion）token 的固定预算，然后一直训完，没有自适应阈值、没有按验证集动态早停。易错点有二：对齐不是从头训练语言模型，也不是训视觉编码器（在 LLaVA 式设定里两者都冻结）；「alignment」这个词在这里指的是投影层的对齐，而不是后训练里 RLHF 意义上的价值对齐。

第四个问题从 64:02 起、64:32 被复述为「are the number of parameters in the vision code much smaller」，讲者答「yes in general」，并给出直觉解释：「the vision encoder is in some sense doing a very local operation. It's looking at a patch. A patch is very small and it's just trying to understand the patches. There's not much knowledge ... we're not reasoning. So most of the capabilities of the model are still kind of in the language model」。也就是说视觉编码器只做局部 patch 编码，不需要世界知识也不做推理，模型的能力主要仍装在语言模型里。数字部分出现了一次值得如实记录的口误：讲者先说 LLaVA 是「72 billion parameter language model」，随即自己改口「oh this sorry this is 72 million okay so this is much smaller the projector is much smaller and the ViT is generally less than a billion parameters」。也就是 projector 约 7200 万（72 million），ViT 通常小于 10 亿（1 billion）参数，而语言模型是数百亿量级。M 与 B 的一字之差正是本节最容易读错的地方。

![LLaVA OneVision 三阶段训练超参数表，含分辨率、样本数、可训练参数、学习率与 batch size](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0053_01-05-52.jpg)

画面 kf_0053 给出的是一张转写完全没念到的三阶段超参表，正好把上面那些口头数字坐实。表头是 Stage-1 Language-Image Alignment → Stage-1.5 High-Quality Knowledge Learning → Stage-2 Visual Instruction Tuning，Stage-2 再分 Single-Image 与 OneVision 两列。Resolution 一行依次为 384、$384\times\{2\times1,\ 1\times\{2,3\},\ \{2,3\}\times1\}$ 且 Max $729\times5$、$384\times\{\{1\times1\},\cdots,\{6\times6\}\}$ 且 Max $729\times10$ (See Fig. 3)、以及同分辨率的 (Multi)-Image & Video (Sec. 4.2)；#Tokens 为 729。Dataset #Samples 依次为 LCS 558K、Image (Sec. 4.1) 4M、Image (Sec. 4.2) 3.2M、(Multi)-Image & Video (Sec. 4.2) 1.6M。Trainable 一行是 Projector 后接三个 Full Model。按语言模型规模分列的三行参数量：0.5B LLM 对应 1.8M / 0.8B / 0.8B / 0.8B，7.6B LLM 对应 20.0M / 8.0B / 8.0B / 8.0B，72.7B LLM 对应 72.0M / 73.2B / 73.2B / 73.2B。Batch Size 为 512 与三列 256/512；学习率 LR: $\psi_{vision}$ 为 $1\times10^{-4}$ 后接三列 $2\times10^{-6}$，LR: $\{\psi_{proj},\phi_{LLM}\}$ 为 $1\times10^{-3}$ 后接三列 $1\times10^{-5}$；Epoch 四列都是 1。底部还有一句「Transfer between modalities: Single image data for diagrams and charts, but generalize to multi-image」。$72.0\text{M}$ 对 $73.2\text{B}$ 这组对照，恰好解释了讲者为什么把 72 million 说成「much smaller」：projector 比全量微调的语言模型小了约三个数量级。

把这张表放回上下文要小心两点。第一，它是 LLaVA OneVision 的配置，不是 Qwen3-VL 的；讲者是在「I'm trying to find ... where is the other place」翻找旧幻灯片时回到它，用它来回答参数规模问题，所以不能把 558K / 4M / 3.2M / 1.6M 或那些学习率当成 Qwen 的超参。第二，表里的 72.7B LLM 与讲者口中的 72B 属同一量级，但 72.0M 的 projector 是 Stage-1 只训投影器时的可训练量，后面三列 73.2B 是 Full Model 全量微调时的量，两者不可混为一谈。局限上，这张表也没有覆盖 Qwen 系列的 SigLIP-2、DeepStack 或 M-RoPE，它只能说明「视觉侧参数远小于语言侧」这一结构性事实，以及 alignment 阶段学习率比后续阶段高一到两个数量级这一经验设置。

## Qwen 系列演进：框架不变、以放大为主 [66:14 - 67:19]

讲者在这里为整个 VLM 部分收口：「Qwen 3 is a kind of the last vision language model I'll talk about」。他紧接着说的是数据的不透明：「there's a lot of data work that happens not too many details about the data and the data mix in these later Qwen models」，也就是后期 Qwen 模型做了大量数据工作，但关于数据与 data mix（数据配比）的细节披露很少。想看更多细节，他建议去读 LLaVA 论文，以及 AI2 那篇他「didn't get a chance to talk about」、数据细节更多的论文（转写记作 mobile，此处不臆测论文名）。这一句其实是对上一章 benchmark 表的重要限定：表能告诉你结果强，但不能告诉你为什么强，因为配方没公开。

画面 kf_0054 继续给出了那张基准表的下半部分，这些内容转写同样没有念到。分组变成 Perception with Tool（MMVU、V*、HRBench4K、HRBench8K）、Multi-Modal Coding（Design2Code、ChartMimic、UniSVG）和 Multi-Modal Agent（ScreenSpot Pro、OSWorldG、AndroidWorld、OSWorld、WindowsAA）。原样数值里，Design2Code 一行 Qwen3-VL 为 93.4 / 92.0，Gemini 2.5 Pro 为 89.2 / 90.3，OpenAI GPT-5 为 92.5 / 88.9，Claude Opus 4.1 为 88.5 / 85.3；ChartMimic 为 78.4 / 80.5 / 83.9 / 79.9 / 62.1 / 41.4 / 85.2 / 82.9；Multi-Modal Agent 组里大量闭源模型列是 `-`（无数据），例如 OSWorld 为 38.1 / 31.6 后面全为 `-`，AndroidWorld 为 62.0 / 63.7 后面全为 `-`，WindowsAA 为 32.1 / 28.9 且只有最后一列出现 44.4。这些 `-` 提醒我们表格的对比并不完整，不能据缺失项推断闭源模型不会做这些任务。

![Qwen3-VL 基准表下半部分及 SOTA performance 等四条总结 bullet](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0054_01-07-15.jpg)

幻灯片下方的 Summary 是本节最关键、也完全是图上独有的原文，其中「Scale up」被黄色高亮：

```text
Summary:
  SOTA performance
  Lots of data work, but not many details
  Minor but potentially important architectural improvements
  Scale up
```

这四条与转写 66:51 之后的说法一一对应：「from Quen to Quen to Quen 3 many of these changes were mostly ... but the overall framework remains the same and mostly it's kind of scaling up curating more data sets noticing that ... you have to handle long context and maybe sharpening your image processing and so on」。也就是说从 Qwen-VL 到 Qwen2-VL 再到 Qwen3-VL，overall framework（总体框架）始终是「视觉编码器 + 适配器 + 语言模型」，变化集中在四点：scale up（放大规模）、curating more data sets（整理更多数据集）、handle long context（处理长上下文，对应前一段的 256K 与 $8\text{k}\to32\text{k}\to256\text{k}$）、sharpening image processing（改进图像处理，对应动态分辨率与 SigLIP-2）。画面底部还露出讲义代码大纲的下一行 `def chameleon():` 与 `Chameleon [Chameleon Team 2024]`，这是一个很明确的结构信号：讲者马上要换主题了。

与前后内容的关系上，本节是本讲前半部分的终点，也是分水岭。前面的 CLIP、SigLIP、LLaVA、LLaVA OneVision、Qwen-VL 系列，无论怎么改分辨率、位置编码、训练阶段，都属于「把图像编码注入 LLM」这一范式；讲者用「the overall framework remains the same」明确否定了范式变化。易错点是别把 Qwen2-VL 的 M-RoPE、Qwen3-VL 的 DeepStack 当成新范式，它们被归在「minor but potentially important architectural improvements」里。局限也写得很直白：数据细节不公开、部分基准列缺失，所以这一节能下的结论只有「结果达 SOTA、主要靠 scale up 与数据工作」。正因如此，下一节讲者才说「I'm going to now jump to something quite different」——真正换范式的，是 Chameleon。

## Chameleon：把一切映射为离散 token 的统一模型 [67:19 - 69:15]

67:19 讲者明确转折：「I'm going to now jump to something quite different」，引出 Meta 2024 的论文 Chameleon。他先把旧范式的边界重述一遍：此前 VLM「encode images into these vectors and then they inject them into a language model. And because it's a language model, you can only generate text. you can't generate images」。修补办法其实不止一种，讲者顺口提到「you can also attach a diffusion head and you now can generate images」，即在 VLM 上挂一个 diffusion head（扩散头）来获得图像生成能力；但他要讲的是另一条美学上更彻底的路：「Chameleon says what if we mapped everything into discrete tokens」。他还带了一句自评——「maybe this reflects that I'm a language person this is kind of appealing」，因为对做语言的人而言，「now you can analyze and generate images in the same way everything is a discrete token」是极有吸引力的：理解图像与生成图像变成同一个自回归问题。

![Chameleon 的 Mixed-Modal Pre-Training 与 Mixed-Modal Generation 示意图，展示图文 token 交错与 Image Tokenizer / De-Tokenizer](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0055_01-08-45.jpg)

画面 kf_0055 的标题是「Chameleon: Map everything into discrete tokens」，副标题「Advantage: can analyze and generate images in a uniform way」，图分左右两个子图，核心方块都是 Mixed-Modal Auto-Regressive LM（混合模态自回归语言模型）。(a) Mixed-Modal Pre-Training 里，TEXT PROMPT 是绿色框里的 `"What can I bake with this?"`，IMAGE PROMPT 是一张香蕉与黄油的图片，经过 Image Tokenizer 变成蓝色图像 token；两类 token 一起送进模型，序列里用 `Start Image` 与 `End Image` 标记图像段的起止，绿色表示文本 token、蓝色表示图像 token。(b) Mixed-Modal Generation 里，模型自回归产出 TEXT OUTPUT `"Here is a recipe for banana bread."`，图像部分则经 Image De-Tokenizer 还原成 IMAGE OUTPUT（一张香蕉面包图），下方序列同样带 `End Image` / `Start Image` 边界标记。把这种交错序列写成一个统一的 token 流就是：

$$x=[\ \text{text tokens};\ \texttt{Start Image};\ z_1,\ldots,z_n;\ \texttt{End Image};\ \text{text tokens}\ ]$$

讲者在 68:51 举了论文里的交错例子：「I'm bored can you show me some birds and then it would be text and then there' be images and then some more text and some images so it's interled」，即一段文本、一段图像、再一段文本、再一段图像，真正 interleaved（交错）。画面底部框给出了这段 prompt 更完整的原文：`Prompt: I'm bored. Could you show me some cool, quirky-looking birds? Also tell me about them in 2-3 ...`（末尾被裁切，此处只照录可见部分，不补全）。讲者由此点出 omni model（全模态模型）的愿景：「text and images truly live in the same space and this is accomplished by making everything look like text that's one way to do it」。注意他特意加了「that's one way to do it」——把一切做成文本的模样只是实现同一空间的一种办法，不是唯一办法。

与前后内容的关系上，这一节把前面所有「连续编码器 + 投影器 + 冻结/微调 LLM」的管线整条删掉：不再有 adapter，不再有单独的 vision encoder，训练随之退化为标准语言模型训练，下一节会展开。易错点有两处。第一，「everything is a discrete token」不等于图像真的变成了语言，它只是「making everything look like text」，图像 token 的统计性质与文本 token 截然不同，这正是后面训练不稳定的根源。第二，`Start Image` / `End Image` 这类边界标记不是装饰：没有它们，模型无法判断一段 token 该按文本词表解码还是按图像码本解码，交错生成也就无从实现。局限是这一节只给了愿景与示意图，没有给任何性能数字；讲者到 73:17 会直接说「I'm not even going to show the results」。

## VQ-VAE 离散化与 Chameleon 训练规格 [69:15 - 72:05]

要让上一节的统一 token 落地，必须先回答「怎么把图像变成离散 token」。讲者给出的答案是 VQ-VAE，并说这是「rather old ... from 2017」的想法；画面标注为 `VQ-VAE (Vector Quantized Variational Autoencoder) [Oord+ 2017]`，其 bullet 原文是「Idea: map image to a discrete codebook, decode back to image and minimize reconstruction」。机制是：编码器把图像 patch 映射成连续向量，再「sort of rounded to the nearest code」——在 codebook（码本）里取最近的原型向量；讲者说码本「has let's say 8,000 codes」，这些是「prototypical vectors that correspond to patches」。解码器再从该 code 重建图像，训练目标「basically minimizing the reconstruction loss」；由于量化不可微，「there's some other terms that you add because this is not differentiable ... which I won't have time to get into」。转写在这里明确停住了，但画面把那些「other terms」写得很清楚。

![VQ-VAE 架构图：Encoder 到连续 latent、Vector Quantization 最近码查找、Codebook 与 Decoder 重构，含损失函数公式](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0056_01-10-22.jpg)

kf_0056 的架构图给出完整的公式原文，正好补上转写略过的部分。数据流是 Input → Encoder $q_\phi(z|x)$ → Latent features (continuous) → Vector Quantization（框内分 Nearest code 与 Quantized (latent indices)，示例索引格子里写着 3, 127 / 4, 4, 1 / 9, 18, 2）→ Decoder $p_\theta(x|z)$ → Reconstruction。旁边的 Codebook (Embedding Table) 标注为 $E\in\mathbb{R}^{K\times D}$，行向量写作 $e_1,e_2,\ldots,e_{K-1},e_K$，并注明「K: codebook size, D: embedding dimension」。Loss 框里的原文是：

$$\mathcal{L}=\mathcal{L}_{recon}+\mathcal{L}_{vq}$$

其中「$\mathcal{L}_{recon}$: reconstruction loss (e.g., L1 or L2)」，「$\mathcal{L}_{vq}$: vector quantization loss (commitment + codebook)」。图例用四种颜色区分 Encoder (continuous latent)、Vector Quantization (discrete)、Decoder (reconstruction)、Codebook (learnable embeddings)。也就是说，讲者口中「不可微所以要加的额外项」具体就是 vector quantization loss 里的 commitment 项与 codebook 项；$q_\phi(z|x)$、$p_\theta(x|z)$、$E\in\mathbb{R}^{K\times D}$ 这些符号也全是图上独有、转写没念到的。

具体数字方面，转写 70:48–71:03 说「512 x 512 images that are converted into 1,024 tokens ... each token comes from a vocabulary of 8,000」；而幻灯片 kf_0056 / kf_0057 / kf_0058 写的是「Encodes 512 x 512 image into 1024 tokens (codebook of size 8192)」。口述的 8,000 与画面的 8192 存在原样差异，这里两者都保留，不替讲者圆场。此外画面还有一条「Train a new BPE tokenizer」：因为「your data looks different than if it were just normal natural language」，图文混合后的数据分布与纯自然语言不同，需要重新训练 BPE tokenizer。训练本身则变得「very straightforward. This is just normal language model training」——「There's no adapter, there's no ... turn the vision encoder ... there's just a language model now」。讲者说仍有两阶段，但这与普通语言模型训练本来就有两阶段是一样的：Stage 1 是「the bulk of training is large unsupervised」，数据「both text and text and image」，Stage 2「they mix in high quality data」。

![Chameleon 的规格与训练数据配比：1024 tokens、codebook 8192、Stage 1 占 80% 及 2.9T/1.5T/400B token 数](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0057_01-11-59.jpg)

kf_0057 把这些口头阶段换算成了转写完全没念到的规格原文：

```text
Encodes 512 x 512 image into 1024 tokens (codebook of size 8192)
Train a new BPE tokenizer

Training:
  Stage 1 (80%): large-scale, unsupervised (2.9T text tokens, 1.5T text/image tokens, 400B ... interleaved tokens)
  Stage 2 (20%): 50% of stage 1 data, 50% of high quality data
```

即 80% / 20% 的阶段划分、2.9T 纯文本 token、1.5T 图文 token、400B 级别的图文交错（interleaved）token；Stage 2 一半沿用 Stage 1 的数据、一半换成高质量数据。其中「400B」之后屏幕右侧被裁切、转写也没念到，这里不补全。把这些数字与前一段 Qwen3-VL 的 $8\text{k}\to32\text{k}\to256\text{k}$ 调度并排看，能看出两种范式在「复杂度放在哪」上的差别：Qwen 的复杂度在长度调度与系统工程，Chameleon 的复杂度被压进了 tokenizer 与码本，训练反而回归最朴素的 LM 设置。

与前后内容的关系上，本节把上一节的「统一 token」从口号变成可执行规格，也为下一节的三个问题提供了全部前提。易错点是三个数字不要混淆：8192 是图像 token 的码本大小（词表），不是序列长度；1024 是每张 $512\times512$ 图像产生的 token 数，不是训练总量；2.9T / 1.5T / 400B 是训练语料规模，与 Qwen3-VL 的 256K 上下文长度完全不是一回事。局限也很清楚：VQ-VAE 的目标只是「decode back to image and minimize reconstruction」，保证能还原，却不保证离散码保留细粒度信息；讲者自己说过量化不可微的额外项「I won't have time to get into」，所以 commitment 项的系数、码本更新方式在这里都没有展开。

## 训练不稳定性、信息损失与路线式微 [72:05 - 74:43]

讲者在 72:05 用「there's a few problems with this」转折，第一个问题是训练不稳定：「they found that the training was not stable」。根因不是工程 bug，而是模态本身的统计差异：「text and images despite occupying the same space just behave very differently. So just calling things discrete tokens isn't really hiding the fact that there's an image living there」。具体是熵的差别：文本 token「if you think about predicting the next word, this has relatively lower entropy. Most words are kind of predictable」，而图像 token「have very high entropy. I don't know what shade of blue this exact token is going to be」。写成不等式就是：

$$H(\text{text tokens})\ \ll\ H(\text{image tokens})$$

后果是「this training with this kind of mixture led to a lot of the norms of the parameters to grow and to generate kind of loss instability」，即参数范数 $\|\theta_t\|$ 随训练增长、loss 出现不稳定。缓解手段是「They were able to kind of mitigate or fix this to some extent via QK norm and Z loss regularization which controls the norm growth」；画面 kf_0058 把这两句写成「Text tokens have low entropy, image tokens have high entropy, leads to norm growth, l[oss ...]」与「Fixes: QK norm, z-loss regularization」。转写记作「Z loss」、画面记作「z-loss」，两者都保留。注意「to some extent」这个限定：QK norm 与 z-loss 是压住范数增长的缓解手段，不是把模态差异消除掉。

![Chameleon 的 Training stability 与 Summary：低熵文本与高熵图像、QK norm、z-loss、离散化丢信息与 OCR](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0058_01-13-36.jpg)

第二个问题是性能与信息损失。讲者直说「I'm not even going to show the results」，因为「it turned out this model was not really as performant and the discretization does definitely loses information」。他给的直觉例子是 OCR：「think about OCR ... if you discretise very small print you're not going to be able to read it anymore」，很小的字号经过离散化后就再也读不出来了。画面 kf_0058 的 Summary 原文（第二条被黄色高亮）是：

```text
Summary:
  Elegant (just autoregressive modeling of discrete tokens)
  Not as performant (discretization loses information - think OCR)
  Training with multiple modalities is tricky
```

这三条与讲者的评价完全对应：「there's a certain elegance here because it's just one model that treats really all modalities the same. But the downside is ...」。优雅来自统一——只剩一个模型、只做离散 token 的自回归建模；代价来自离散化本身。这里要避免把「熵高」误读成「信息量大所以更强」：恰恰是图像 token 的高熵让下一个 token 难以预测，才引出范数增长与 loss 不稳定，也才让重构目标下的细粒度信息最先被牺牲。

第三个问题在 73:54 提出，是多模态配比：「training with multimodalities is also tricky. We kind of saw this a little bit with the Qwen models and we had to play around with the weights but this is more exacerbated here」。这正好回指本段 62:02 之后关于 downweight 与 data mixtures 的答疑，以及 Qwen 按样本长度归一化 per-token loss 的做法：在连续编码器范式里调权重已经够麻烦，在图文彻底混成一条 token 流的 Chameleon 里更被放大。随后讲者交代了这条路线的兴衰：74:04 起「VQ-VAEs were kind of popular for a while ... many people were using this for image generation」，主因很实际——「you have a transformer how do you generate from a transformer? You have to generate discrete things」，Transformer 天生只能生成离散东西，所以必须先把数据都离散化；但「diffusion models came out and became popular and viable for generation. So this flavor of method is sort of less popular than it used to be」，扩散模型成熟后，这条离散化路线就没那么流行了。

与前后内容的关系上，本节是 Chameleon 的「账单」：上一节展示的优雅在三个地方付了代价——优化稳定性、信息保真度、数据配比；也为最后一章「continuous encoders + diffusion」的结论提供了反面论据。易错点有两处：其一，讲者没有展示 Chameleon 的 results，所以不能从这张幻灯片推断任何具体分数或与其他模型的差距；其二，QK norm、z-loss 被描述为「mitigate or fix ... to some extent」，是缓解而非根除。局限上，这一节是纯定性的：画面没有给范数增长曲线、没有给 z-loss 的系数 $\lambda$、也没有给 OCR 的定量掉点，能带走的只有机制解释与路线判断。

## 总结：非文本编码、理解与生成的张力 [74:43 - 77:39]

74:43 起讲者「maybe just kind of reflect and summarize」。他先给趋势判断：「frontier models these days are expected to be multimodal or even more strongly natively multimodal or omnimodels」；并提到「when Gemini comes out or GPT comes out, they are touted as being natively multimodal and handle all these modalities and in fact they do. But of course there's no details about how these are built」。也就是说能力是真的，构建方式不公开。他随后给出明确标注为猜测的推断：「it's probably some combination of having a continuous encoder because you don't want to lose information and then diffusion for the generation because but that's you know my speculation」。画面 kf_0059 与 kf_0060 顶部还保留着整讲的讲义代码大纲，这是转写没念到、但很能说明结构的原文：

```python
# Injecting image encodings into LLMs
llava()
llava_onevision()
qwen_vl()
qwen2_vl()
qwen3_vl()

# Towards Omni models
chameleon()
```

这个大纲把全讲切成两半：前半是「把图像编码注入 LLM」（LLaVA → OneVision → Qwen-VL/Qwen2-VL/Qwen3-VL），后半是「走向 omni models」（Chameleon）。本段恰好从 Qwen3-VL 的收尾跨过这条分界线，走到片尾总结。

根本挑战在 75:32 被点明：「the fundamental challenge I think when you're dealing with multimodality is how do you handle non-text modalities?」。讲者接着指出一个对称性：「there's a symmetry between understanding the modality and generation and there's no one universal encoder」。他用 CLIP 作对照：分类任务只关心「high level semantics」，所以「these vectors could be fairly small」；而「if you wanted to do OCR or whether you wanted to generate an image then you need really fine grain detail」，并解释「that's why diffusion is so good because it can like really micro optimize the ... high frequency information」。把这种「同一模态、不同任务、不同粒度需求」写成对照就是：

$$\text{classification}\ \Rightarrow\ \text{high-level semantics (small vector)},\qquad
\text{OCR / generation}\ \Rightarrow\ \text{fine-grained, high-frequency detail}$$

这解释了为什么不存在 one universal encoder（通用编码器）：理解侧可以压缩，生成与细粒度识别侧不能压缩，二者的最优表示根本不同。这也是 Chameleon 用单一码本同时承担理解与生成会「discretization loses information」的深层原因。

76:22 之后是配比与现状结论：「when you're dealing with multimodalities you have to think carefully about weighing them properly for example video certainly has lower information density than text so you don't want a video to overwhelm your text」，呼应本段 62:02 的 downweight 讨论；「perhaps the at least the current best thing is having continuous encoders. You know even CLIP even though it's five years old or similar ideas are still kind of the go-to way to capture semantics of images. Transformers are still there. And then diffusion models which I didn't talk about are great for generation」。画面 kf_0060 把这段整理成五条 Summary 原文，最后一条被黄色高亮：

```text
Summary:
  Frontier models are expected to be multimodal (natively multimodal, omni)
  Fundamental challenge: how to encode non-text modalities?
  Comprehension and generation might demand different things (semantics versus finer-grained ...)
  Balance images + video (lower information density) and text for training stability
  Continuous encoders + Transformer + diffusion models for generation
```

底部还回到起点：`def clip():` 与 `CLIP (Contrastive Language-Image Pretraining) [Radford+ 2021]`。

![课程片尾总结页：五条 Summary bullet、Towards Omni models 代码大纲与 CLIP 参考文献](/coursenotes/cs336-language-modeling-sp26/l17-alignment-and-multimodality/kf_0060_01-17-14.jpg)

再往深一层看，讲者说的「there's a symmetry between understanding the modality and generation」并不是说两者可以共用同一套表示，恰恰相反：正因理解与生成互为镜像，才更显出「no one universal encoder」——理解允许把图像压成小向量、只留高层语义，生成却必须把高频细节逐一还原，同一个编码器无法同时最优。他给 diffusion 的理由也落在这里：「it can like really micro optimize the ... high frequency information」，即扩散模型擅长在像素级微优化高频信息，这正是 VQ-VAE 那 1024 个离散码做不到的。把这一点与上一节的「discretization loses information - think OCR」连起来，就能明白为什么讲者把 continuous encoder 放进他推测的组合里：不是因为它更新颖，而是因为它不先把信息压进有限码本。局限同样要记住，这些判断全是定性的，画面既没有给信息密度的度量，也没有给「fine grain detail」的量化指标，能带走的只是取舍的方向。

77:13 讲者收尾：「Okay so I will stop there. That's it for the lecture on multi modality」，并说这部分没有作业——「we don't have any homework on doing this but if you're curious ... I encourage you to play around with training some of these models」。与全讲的关系上，这一章把开篇 CLIP 的高层语义表示、中段 LLaVA/Qwen 的连续编码器注入、Chameleon 的全离散统一、以及扩散生成的缺席全部收拢到一句话：连续编码器负责理解、Transformer 作为主干、扩散模型负责生成。易错点也是讲者自己反复标注的：这套组合是「my speculation」，而且 diffusion 部分他明确说「which I didn't talk about」，所以不能当成闭源模型的确定架构来陈述；「CLIP even though it's five years old」也只是口头估计，不是精确年份；`Balance images + video ... and text for training stability` 这条里的 lower information density 是定性判断，画面没有给信息密度的度量或配比数值。这些都是本段宁可略过、不替讲者补全的地方。

## 要点回顾

### 从语言模型到 omni model：token 化的根本约束
- 课程此前 exclusively focused on language models，能力可形式化为 $\text{LM}:\mathcal{T}\Rightarrow\mathcal{T}$（幻灯片写作 `text ⇒ text`），已能吞下自然语言、代码、诗歌乃至 DNA 序列，而真实世界并排列出 Text、Images、Audio、Video 四类。
- 终极目标被称作 omni model，幻灯片并列写成 `Input any combination of modalities (understanding)` 与 `Output any combination of modalities (generation)`，但讲者申明本讲不 build up a whole omni model、today mostly focus on 图像。
- 推理链落在 `Transformers speak tokens (discrete or continuous), where a token represents some ~semantic unit of information`：a pixel is certainly not meaningful by itself，故非文本模态必须像文本的 BPE tokenization 那样先编码成词元。
- 两个待解问题被编号写死为 `1. How do we input non-text data (e.g., understand images)` 与 `2. How do we output non-text data (e.g., generate audio)`，本讲主要聚焦 question number one，调用路线从 `clip()` / `siglip()` 到 `llava()` / `llava_onevision()`。

### CLIP：对比预训练、数据与编码器
- [Radford+ 2021] 想 leverage 海量 (image, caption) pairs 取代 ImageNet 标注，`Get a batch of (image, text) examples (e.g., 32768)` 形成 $N\times N$ 相似度矩阵、对角线为正例、`labels = np.arange(n)`，行列各做一次 softmax 相当于 $2\times N$ 个分类问题。
- 对称损失 $\mathcal{L}=-\frac{1}{2N}\sum_i(\log\frac{e^{S_{ii}}}{\sum_j e^{S_{ij}}}+\log\frac{e^{S_{ii}}}{\sum_j e^{S_{ji}}})$，其中 $S_{ij}=e^{t}\hat I_i^\top\hat T_j$；伪代码里 `l2_normalize(..., axis=1)` 后点积才是 cosine similarity，`W_i/W_t` 压到共同 $d_e$，温度取 `np.exp(t)` 保证为正。
- 数据是 `Searched for 500K queries, get ~20K pairs per query`、`Trained on 400M image-text pairs`、`Didn't release the dataset`，OpenCLIP 用 LAION-5B（50 亿）复现且 used CLIP for the data filtering，存在 bootstrapping。
- 预处理用 `bicubic interpolation` 把短边缩到 336 再 `Center crop` 成 $336\times336\times3$；编码器选定 `Best model: ViT-L/14@336px`（$14\times14\times3=588$ 维 patch、1D 位置嵌入），视觉侧用 attention pooling（`query = global average of activations`）压成单向量，文本侧 GPT-2（63M、12 层）取 [EOS] 最高层激活。
- zero-shot 把标签填进 `A photo of a {object}.`、按 $\hat y=\arg\max_k\hat I^\top\hat T_k$ 选类，headline 是零样本超过用 120 万张 ImageNet 训练的 ResNet；消融中直接预测文本的 $\text{Acc}_{\text{CLIP}}(400M)\approx42$ 高于 BoW-Pred≈27、Transformer-LM≈16，并标注 4X / 3X efficiency。

### SigLIP：sigmoid 损失与跨设备并行
- [Zhai+ 2023] 把 CLIP 的 `multiclass classification` 换成 `binary classification for (text, image) - aligned or not`，仍是 $N\times N$ 矩阵但每个格子独立算 logistic 损失，分母里的 $\sum_j$ 由此消失。
- 伪代码 `labels = 2 * eye(n) - ones(n)`（对角 $+1$、其余 $n^2-n$ 格为 $-1$）、`l = -sum(log_sigmoid(labels * logits)) / n`，`logits = dot(zimg, ztxt.T) * t + b`，$t=\exp(t')$ 与偏置 $b$ 可学习，$\mathcal{L}_{SigLIP}=-\frac1n\sum_{i,j}\log\sigma(y_{ij}\ell_{ij})$。
- 因损失对每个 $(i,j)$ 独立，可写成 $\mathcal{L}=\sum_d\mathcal{L}_d$，各设备先算本地块、再轮转文本覆盖 off diagonal 块并 `Cross Device Σ` 求和；batch 结论为 `Decouple batch size from loss`、`<16K` 时优于 CLIP、`32K is enough`（可开到 1M）。
- 数据用 WebLI（$O(\text{billion})$ 图文对、[Chen+ 2022]），以 `automatic OCR` 抽字造对、`Keep 10% highest quality`、`Supports 100 languages`；效率对比是 `CLIP: 10 days on 256 TPUv3` 对 `SigLIP: 5 days on 32 TPUv4`。

### VLM 缝合模板与 LLaVA
- 通用模板是 vision encoder + projector + LM 的 stitch（偏 mid-training/post-training，而非 from scratch）；[Liu+ 2023] LLaVA 用 CLIP ViT-L/14 + Vicuna（ShareGPT 微调的 LLaMA），$Z_v=g(X_v)$、$H_v=WZ_v$、$X_a=f_\phi([H_v;H_q])$。
- 数据取自 MS COCO 的 captions 与 boxes（如 `person: [0.681, 0.242, 0.774, 0.600]`），用 GPT-4 生成 conversation / detailed description / complex reasoning，合成 `158K examples`，全是 image + text → text。
- 两阶段训练：Stage 1 alignment 冻结视觉编码器与 LM、只训 $W$（$\Theta_1=\{W\}$），让图像向量 look like natural language token embeddings；Stage 2 仍冻结 $g$，放开 $W$ 与 $\phi$（$\Theta_2=\{W,\phi\}$）做微调。
- 答疑钉住边界：这些模型 `don't generate videos or images`、always generating text，除 RL 外是逐 token 监督 $\mathcal{L}_{SFT}=-\sum_t\log p(x_t\mid x_{<t})$；对齐时 LM 必须预训练且冻结、按固定 `67 billion tokens` 预算训完，projector 约 `72 million`、ViT 通常 `<1 billion` 参数。

### LLaVA OneVision：AnyRes、token 预算与三阶段训练
- 组件升级为 `SigLIP`（取最后层前后 grid features）、`Qwen-2 72B`、`2-layer MLP` 投影器 $p_\theta$，即 $H_v=p_\theta(g_\psi(X_v))$，输入从单图扩到 Single / Multi-Image / Video。
- 为保 OCR 细粒度（否则 "a J looks like a I"），沿用 LLaVA 1.5 的 AnyRes 把图切成 $a\times b$ 块逐块编码再 concat、超预算用 `bilinear interpolation` 压编码后特征，$T_{single}=729+N\cdot729=(1+N)\cdot729,\ N\le9$，即 $(1+9)*729=7290$ Tokens。
- 三类输入被压到同一量级：单图 $7290$、多图 $12\cdot729=8748$、视频 $32\cdot196=6272$ Tokens，对应 `Goal: make all of the modalities produce roughly the same length`，避免重复帧淹没数据。
- 训练按 `easier to harder` 分 Stage-1 / Stage-1.5 / Stage-2，样本数 $558K\to4M\to3.2M\to1.6M$ 并不单调，学习率由 $1\times10^{-3}$ 降到 $2\times10^{-6}\sim1\times10^{-5}$、Epoch 全为 1，Stage-1 只训 Projector、之后 Full Model。
- 跨模态迁移：单图 diagrams/charts 泛化到多图问答、单图 OCR 加多图 relational reasoning 泛化到 GUI agents、视觉提示泛化到视频球员描述；`Summary` 归结为 `vision encoder + projector + LM`、`Most work goes into data curation`、open-source 权重与数据。

### Qwen-VL 系列与长上下文
- Qwen-VL [Bai+ 2023] 用 `OpenCLIP's ViT-bigG (14x14 patches)` 加一层 cross-attention（2D 位置编码）压成 `fixed length of 256`，靠 `<box>` / `<ref>` 等 token 输出框的文本坐标，且 Stage-1 会 train the vision encoder。
- Qwen2-VL 引入 `Native Resolution Input` 与 dynamic resolution，每个 $224\times224$ patch 经 ViT/14 后 $2\times2$ 压缩成 `66` tokens，视频 `sample 2 frames/sec, max 16384 tokens`，并把位置编码改成 $\text{M-RoPE}(t,h,w)=\text{concat}(\text{RoPE}_t,\text{RoPE}_h,\text{RoPE}_w)$。
- Qwen3-VL 换成 `SigLIP-2`（架构相同、向后兼容）与 Qwen3 dense/MoE（`up to 235B-A22B`）、上下文 `256K`，把分块 M-RoPE 改为 `Interleaved MRoPE`（`[t w h t w h]` 而非 `[t t t w w w h h h]`），并改用显式视频 timestamp token。
- 还用 square-root-normalized per-token loss（$\mathcal{L}_{example}=\frac1{\sqrt N}\sum_i\ell_i$，讲者标注为推测）平衡长短样本，用 `DeepStack: cross-layer fusion` 把多层视觉特征注入残差流 $\mathbf h^{(l)}\leftarrow\mathbf h^{(l)}+\mathbf v^{(l)}$。
- 管线是预训练四阶段（$8K\to32K\to256K$、`most of the tokens are in the stages two and three`）加后训练三阶段（长 CoT SFT、蒸馏、RL），讲者定性为 `a systems paper`；`Summary` 高亮 `Scale up`，强调 framework 始终是视觉编码器 + 适配器 + LM、只是放大与数据工作。

### Chameleon 与全讲结论：离散统一及理解／生成的张力
- [Chameleon Team 2024] 换范式：把一切映射成离散 token，图文以 `Start Image` / `End Image` 交错成统一流 $x=[\text{text};\texttt{Start Image};z_1..z_n;\texttt{End Image};\text{text}]$，理解与生成同为自回归，不再有 adapter 或单独 vision encoder。
- 靠 VQ-VAE [Oord+ 2017] 把 patch 量化到最近码字，$\mathcal{L}=\mathcal{L}_{recon}+\mathcal{L}_{vq}$（commitment + codebook），`Encodes 512 x 512 image into 1024 tokens (codebook of size 8192)` 并另训新 BPE tokenizer，Stage 1 占 80%（2.9T 文本 / 1.5T 图文 / 400B 交错 token）、Stage 2 占 20%。
- 代价是 $H(\text{text tokens})\ll H(\text{image tokens})$ 引发参数范数增长与 loss 不稳，靠 `QK norm` 与 `z-loss regularization` 仅「to some extent」缓解；`discretization loses information - think OCR`，讲者 `I'm not even going to show the results`，扩散模型成熟后该路线 less popular。
- 根本挑战被点明为 `how to handle non-text modalities`，且 `there's no one universal encoder`：classification 只需 high-level semantics（small vector），OCR / generation 却要 fine-grained、high-frequency detail，而 diffusion 恰能 micro optimize 高频信息。
- 片尾 `Summary` 收敛为 `Continuous encoders + Transformer + diffusion models for generation` 与 `Balance images + video (lower information density) and text for training stability`，但讲者明确标注这是 `my speculation`、diffusion 部分 `which I didn't talk about`，且本讲这部分没有作业。

## 视频来源

- 原始链接：https://www.youtube.com/watch?v=26FtD08ZpOU
- 课程：Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 17

---

## 对照参考笔记补遗（Spring 2025 版）

> 编者注：本节依据 2025 版参考笔记整理，其中的「更正」小节记录的是核对时发现的问题，对应正文已在本版订正；补遗内容不代表 2026 课程视频原文。


本节来自 Spring 2025 版参考笔记（L17《Alignment — RL 2》与附录 A1《Algorithm Deep Dives》），仅用于对照补充可迁移的通用方法、公式与易错点，不代表 2026 视频原文；本讲《Alignment - Multimodality》在 2025 版没有直接对应讲次，故以下内容都是跨讲次的一般原理，不改写本讲已有的模型配置与结论。

### 旋转位置编码的相对性与频率谱：从 RoPE 到 M-RoPE

本讲在 Qwen2-VL 一节把 RoPE 复习成「两向量内积只依赖相对距离」，并写出 M-RoPE 把时间、高、宽三轴分别做 RoPE 再拼接的形式，却没有给出「为什么旋转能满足相对性」的推导。附录 A1 在 *Transformer Architecture Choices* 里补上了这条：把二维向量记作复数 $x=x_1+ix_2$，定义位置编码 $f(x,m)=x\,e^{im\theta}$，则 $\mathrm{Re}[f(q,m)\overline{f(k,n)}]=\mathrm{Re}[q\bar k\,e^{i(m-n)\theta}]$，内积只含原始的 $q\bar k$ 与一个仅依赖相对距离 $m-n$ 的复指数因子。用旋转矩阵写就是 $\langle R_m q,R_n k\rangle=\langle q,R_m^\top R_n k\rangle=\langle q,R_{n-m}k\rangle$，其中 Rotary Position Embedding（RoPE，旋转位置编码）的「旋转」正来自 $R_m$ 正交、$R_m^\top=R_{-m}$。

高维推广把 $d_{head}$ 维向量按相邻两维切成 $d_{head}/2$ 对，每对当作一个二维复数独立旋转，第 $j$ 对的频率为 $\theta_j=10000^{-2j/d_{head}}$，于是得到从高频（分辨近邻 token）到低频（分辨远距 token）的一段频率谱。写成块对角矩阵即 $f(x,m)=R_{m,d}x$，每个 $2\times2$ 块是 $\begin{pmatrix}\cos m\theta_j&-\sin m\theta_j\\ \sin m\theta_j&\cos m\theta_j\end{pmatrix}$，仍有 $\langle R_{m,d}q,R_{n,d}k\rangle=\langle q,R_{n-m,d}k\rangle$。实现上并不物化这个大矩阵，而是直接算 $\cos,\sin$ 逐对作用，这与本讲说的「fairly straightforward idea」一致。

这条推导正好解释本讲 Qwen3-VL 一节「each component represents a different frequency」与交错 M-RoPE 的动因：由于 $\theta_j$ 随维度下标 $j$ 单调下降，旧做法把维度按块分给三轴（写成 $[t\,t\,t\,w\,w\,w\,h\,h\,h]$）会让时间轴整段落进低频带、高度轴整段落进高频带；改成 $[t\,w\,h\,t\,w\,h\dots]$ 交错后，三个轴才都同时见到高、低频。易错点在于：频率由维度下标决定而非轴名决定，若把「分块」误读成「三轴各有一套独立频率」，就看不出 Qwen3 这次修正的必要性；另外 RoPE 保证的只是内积的相对位置形式，并不自带外推到任意长度的能力，本讲 256K 长上下文仍要靠 $8K\to32K\to256K$ 的长度调度训练换得。

### softmax 的数值稳定、log-sum-exp 与 online softmax

本讲 CLIP 的对称损失、SigLIP 的 $\log\sigma(y_{ij}\ell_{ij})$、训练循环里的 `cross_entropy_loss` 全建立在 softmax 之上，且温度写成 `np.exp(t)`、batch 大到 32768，logits 幅值可能相当可观，笔记却未提数值实现。附录 A1 的 *GPU Kernels & FlashAttention* 给出标准稳定化：先取 $m=\max_j x_j$，再算 $\mathrm{softmax}(x)_i=e^{x_i-m}/\sum_j e^{x_j-m}$。这样最大指数为 $e^0=1$ 不会上溢，分母至少有一项为 1 不会下溢成零；否则 $e^{x_i}$ 可能变 `inf`，全为负时又变 0，softmax 输出 NaN、梯度随之崩坏。这就是通常所说的 log-sum-exp trick，L17 计算对数概率时用的 `F.log_softmax` 也是同一件事。

A1 进一步给出 online softmax，即 FlashAttention 能在分块时工作的关键：逐块维护运行最大值 $m$ 与运行分母 $l$，新块到来时 $m^{new}=\max(m^{(1)},m^{(2)})$，分母按 $l^{new}=e^{m^{(1)}-m^{new}}l^{(1)}+e^{m^{(2)}-m^{new}}l^{(2)}$ 合并，已累积的输出向量也要按 $e^{m^{old}-m^{new}}$ 回乘，以修正用旧最大值归一化过的部分。这样无需把 $N\times N$ 注意力矩阵落到显存；A1 还量化了反向重算的收益，把中间 $P$ 的存储从 $O(N^2)$ 降为 $O(N)$ 统计量，$N=4096$ 时约 67 MB 降到约 32 KB，接近两千倍。

这与本讲「CLIP 损失不可分解、softmax 在整个 batch 上做」和「SigLIP 去掉分母后可按元素累加、跨设备求和」直接呼应：前者工程上更依赖稳定的 softmax 与足够大的临时空间，后者因逐格 sigmoid 而天然可分。易错点有两处：温度取 `np.exp(t)` 只保证缩放恒为正，并不保证 logits 不溢出，稳定化照旧要做；$\log\sigma(\cdot)$ 应走 `log_sigmoid` 一类的软实现，若先算出 $\sigma$ 再取 $\log$，负 logits 上会下溢为 $-\infty$，正因为如此 L17 与 SigLIP 伪代码都直接写 `log_sigmoid` 而不是两次运算。

### z-loss 与 logit 范数控制：训练稳定性的一般机制

本讲 Chameleon 一节把训练不稳归因于文本 token 低熵、图像 token 高熵引发的参数范数增长，并记下「Fixes: QK norm, z-loss regularization which controls the norm growth」，却没有给 z-loss 的形式。附录 A1 在 *Mixture of Experts* 里给出 Router Z-loss 的完整定义：对 token $t$ 的各专家 router logits $z_{i,t}$，取 $L_z=\beta\big(\log\sum_{i=1}^{N}e^{z_{i,t}}\big)^2$，再对 batch 内全部 token 取平均，$\beta$ 是一个小系数。由于 $\mathrm{LSE}(z)=\log\sum_i e^{z_i}$ 是 $\max_i z_i$ 的光滑上界，这一项等价于惩罚最大 logit 的平方，把「别让 logit 长太大」写成了可导的正则项。

对单个 logit 求导得 $\partial L_z/\partial z_k=2\beta\,\mathrm{LSE}(z)\,p_k$，其中 $p_k$ 是 softmax 概率；该梯度恒为正，故梯度下降会持续把所有 logits 往下压，且当前 LSE 越大、对原本越自信的那个 logit 压得越狠。动机主要是低精度数值范围：bfloat16 可表示的上限约 $3.4\times10^{38}$，一旦某个 $z_i$ 偏大使 $e^{z_i}$ 溢出成 `inf`，softmax 就变 NaN、训练「炸掉」。QK norm 对 query、key 做归一化，与 z-loss 同属把激活或参数范数约束住以稳定训练的一族手段。

把它放回本讲，就能理解为什么讲者说 z-loss 只是 "mitigate or fix ... to some extent" 的缓解：它是加在损失上的软正则，压的是 logits 幅值，并不改变图文 token 熵差异这一根因，所以不能消除模态差异，更不能推出 Chameleon 的 results。易错点在于作用位置：A1 的 z-loss 作用在 MoE 的 router logits 上，本讲 Chameleon 的 z-loss 作用于语言模型输出 logits，两者同名、同用 log-sum-exp 惩罚、同为稳范数，却不是同一个模块；把这些一般机制迁移过来时，应说明是「机制可类比」而非「附录在讲 Chameleon」。

### 对齐数学：SFT、策略梯度、baseline 与 KL 的两种角色

本讲答疑给出逐 token 监督 $\mathcal{L}_{SFT}=-\sum_{t=1}^{T}\log p(x_t\mid x_{<t})$，并说「除 RL 外 supervising every token」「with RL you can play around with different reports」，但 RL 一侧完全留白。L17 与 A1 的对齐数学正好补上：用 log-derivative trick $\nabla\pi=\pi\nabla\log\pi$，期望奖励的梯度可写成 $\nabla E[R]=E[\nabla\log\pi(a\mid s)\,R(s,a)]$；在 $R(s,a)\in\{0,1\}$ 的可验证奖励下，naive policy gradient 等价于「只在答对的样本上做 SFT」，差别在于数据分布随策略一起变化。为减方差引入只依赖状态、不依赖动作的 baseline $b(s)$，梯度变成 $E[\nabla\log\pi\,(R-b(s))]$ 且保持无偏，因为 $E[\nabla\log\pi\cdot b(s)]=b(s)\nabla\!\int\!\pi=0$；L17 用两状态例（奖励 11、9、0、2）算出朴素方差 5.32，取 $b(S1)=10,b(S2)=1$ 后降到 1.15。

最优基线是 $b^*(s)=\frac{E[(\nabla\log\pi)^2R\mid s]}{E[(\nabla\log\pi)^2\mid s]}$，实用启发式取 $b(s)=E[R\mid s]=V(s)$，此时 $R-b$ 正是 advantage $A(s,a)=Q(s,a)-V(s)$；GRPO 借「同一 prompt 采多个回答」的 group 结构，用组内均值作自然基线，从而去掉 critic。A1 还给出两点更一般的结论：KL 项其实有两种角色——整体 RLHF 目标 $\max_\pi E[r]-\beta D_{KL}(\pi\|\pi_{ref})$ 用固定的 SFT 参考模型作全局行为约束、防灾难性遗忘，而 PPO / GRPO 内环用上一迭代的 $\pi_{old}$ 作局部更新稳定，靠 importance ratio $w_t=\pi_\theta/\pi_{old}$ 与 $\min(w_t\hat A_t,\mathrm{clip}(w_t,1\pm\epsilon)\hat A_t)$ 限制单步偏移；DPO 则借 Bradley-Terry 模型与闭式最优策略 $\pi_r(y|x)=\frac{1}{Z(x)}\pi_{ref}(y|x)e^{r(x,y)/\beta}$，把 RLHF 化成 $-\log\sigma\big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\big)$ 这样一个二分类损失。

这些正好落在本讲 Qwen3-VL 后训练「长 CoT SFT → 蒸馏 → RL」以及片尾「这部分没有作业」的留白处，也解释了为何讲者强调「除 RL 外没有 LM as a judge、质量全由数据集决定」——只有到 RL 阶段才引入奖励、基线与参考模型。两处易错点值得钉住：其一，`alignment` 在本讲 LLaVA 语境里指「投影层把视觉向量对齐进词嵌入空间」，在 2025 的 L15–L17 里指 RLHF 价值对齐，同名不同义，不可互训；其二，GRPO 的 z-score $\hat A_i=(r_i-\mathrm{mean}(r))/(\mathrm{std}(r)+\epsilon)$ 因分母 std 依赖同组其它轨迹，违反了「基线不依赖动作」这一无偏条件，会带来答错时把回答拖长以增大 std、缩小惩罚的长度 reward hacking，所以「按组内归一化」并非中性操作。

### token 化与长上下文的系统代价：BPE、KV cache 与算术强度

本讲几乎每章都在和 token 数打交道——AnyRes 的 $(1+9)\cdot729=7290$、多图 $12\cdot729=8748$、视频 $32\cdot196=6272$、Qwen2-VL 每 patch 66 tokens 与视频上限 16384、Qwen3-VL 的 256K 上下文——并在答疑里把系统开销归于「more tokens」和数据加载，但只给了定性判断。附录 A1 从 token 如何产生的一端补起：Byte-Pair Encoding（BPE，字节对编码）的编码必须按训练时学到的 merge 的 rank（学得越早优先级越高）逐步、贪心地合并当前序列里最优的一对，并列时取最左，结果才唯一确定；例如先学 `(a,b)`（rank 50）、后学 `(b,c)`（rank 100）时，字符串 `'abc'` 得 `[ab,c]`。常见误区是「遍历 merge 规则表、把每条规则套到整串上」，那会产生歧义。GPT-2 的预分词正则含 `'s|'t| ?\p{L}+| ?\p{N}+` 等，会把词首空格并进词块，造成 " world" 与 "world" 取不同 ID 的 leading-space 问题；SentencePiece 改用 `▁`（U+2581）元字符把空格当普通符号，既保留跨词合并又避免该问题。高效训练则用一次扫描建 pair_counts（$O(C)$）、每个词用双向链表使合并为 $O(1)$。

这解释了本讲 Chameleon「Train a new BPE tokenizer」的必要性：图文交错之后数据分布已不同于纯自然语言，旧词表与合并优先级不再适配，必须重训，且重训后仍要保证按 rank 的确定性编码，文本段才能稳定切分。长序列 token 的系统代价则由 A1 的 KV cache 与算术强度量化：Multi-Head Attention 下每层缓存 $2\cdot B\cdot N\cdot d$ 个元素、总字节 $L\cdot2\cdot B\cdot N\cdot d\cdot\text{sizeof(fp16)}$，LLaMA-7B（$L=32,d=4096,h=32,N=4096,B=1$）单序列约 2.15 GB，改用 Multi-Query Attention（$g=1$）约 67 MB，Grouped-Query Attention 为 MHA 的 $g/h$。生成阶段注意力的算术强度 $\mathrm{AI}_{Attention}=\frac{2BSD}{2BD(S+1)}=\frac{S}{S+1}\approx1$，batch 大小 $B$ 被完全约掉，故自回归解码是 memory-bound、batching 救不了 attention，而 MLP 层 $\mathrm{AI}_{MLP}\approx B$ 仍能从 batching 受益。

把这两端连起来，本讲「a big part of being able to handle multimodal is to deal with long context」「控 token 数也是为了让数据管线跑得动」就从定性变成了可算的：视觉 token 越多，上下文长度 $N$ 越大，KV cache 与逐 token 的 HBM 读取线性增长，正是答疑里「异步加载 $\text{load}(t+1)\parallel\text{compute}(t)$」和按模态降权 $\mathcal{L}=\sum_m\lambda_m\mathcal{L}_m$ 要缓解的对象。易错点有三：公式里的 $N$ 是上下文长度，不是 batch，也不是 729 这类 token 预算常量；KV cache 读取与权重读取是两笔不同的账；而 BPE 的「确定性」指编码规则确定，并不等于「图文混合后沿用原词表」——后者恰恰是需要重训 BPE 的理由。
