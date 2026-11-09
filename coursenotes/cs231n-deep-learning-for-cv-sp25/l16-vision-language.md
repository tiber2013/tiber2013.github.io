---
title: "第 16 讲 · 视觉与语言：从 CLIP 到多模态基础模型"
description: "Stanford CS231n Spring 2025 L16 中文视频笔记（2026-11 重写版）"
---

# CS231n Spring 2025 · Lecture 16 视觉与语言：多模态基础模型

**核心结论**：本讲由客座讲师 Ranjay Krishna（University of Washington，共同领导 Raven lab，曾在 2020、2021 年讲授 CS231n，研究横跨计算机视觉、自然语言处理、机器人与人机交互）讲授多模态基础模型，主线是范式从「一个任务收集一套数据、训练一个专用模型」转向「在海量数据与多种技能上预训练，再用极少甚至零数据适配下游任务」。课程沿分类、图文、任意输出空间、链式组合四条线展开：CLIP 用对称的图文对比目标把图像与文本拉进同一表示空间并支持零样本分类，CoCa 用 captioning 目标进一步增强，LLaVA 与 Flamingo 把视觉特征注入大语言模型从而用自回归生成解决理解任务，Molmo 靠高质量密集数据与像素级 pointing 追平闭源模型，SAM 把基础模型的输出扩展到分割掩码，CuPL 与 VisProg 则把多个模型「链」成单个模型做不到的新能力。贯穿全讲的一条判断是：架构改动往往很小（一个线性层、两三行门控交叉注意力），真正决定性能的是数据的规模、质量与密度——尤其是互联网上根本不存在的 grounding 与空间关系信息。

## 1. 基础模型范式与全景分类 [00:39 - 04:16]

课程此前的讲次基本遵循同一套流程：收集训练集与测试集，为该任务训练一个高度专用的模型（图像分类、图像描述等），再在测试集上评估。近几年的转变是走向 **foundation model**：先在很宽的技能与任务分布上预训练，之后按需要把同一个模型适配到具体任务。最熟悉的例子是 GPT——用互联网 Common Crawl 数据预训练，再针对数学题、符号推理、常识问答分别微调；适配所需数据极少，很多时候甚至完全不用再收集训练数据。讲者归纳出这类模型通常具备的共同点：对许多任务都鲁棒且通用、参数量巨大、训练数据量巨大，并往往采用某种自监督目标；不过「基础模型」的定义本身在社区里并无一致意见。

本讲要覆盖的是绿色部分，即多模态方向。下面这张分类表既是开场地图，也是全讲的结构：

![基础模型（Foundation Models）的分类列表，涵盖语言、分类、多模态等类别](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0001_00-03-30.jpg)

- **Language**：ELMo、BERT、GPT、T5——本课不展开。
- **Classification**：CLIP、CoCa——把图像分类做成基础模型。
- **LM + Vision**：LLaVA、Flamingo、GPT-4V、Gemini、Molmo——把语言模型与视觉基础模型结合。
- **And More!**：Segment Anything、Whisper、DALL·E、Stable Diffusion、Imagen——输出不再只是文本或类别，还可以是 mask 或生成的图像。
- **Chaining**：LMs + CLIP、Visual Programming——把若干基础模型串起来做新事情。

讲者的路线因此是：先做能解决任意数据集分类的基础模型，再谈如何把语言模型与视觉结合，再谈如何输出 mask 与图像，最后谈 chaining。

## 2. CLIP：把对比学习从图像扩展到图文 [04:16 - 09:51]

起点是前面讲过的自监督对比学习（SimCLR 一类）：目标把同一张图的不同增广在表示空间里拉近、把不同类别（猫与狗、猴子）推远，希望学到的表征足够通用，以至于猫的草图、狗的草图也能被正确嵌入。多模态的关键一步，是往这个空间里再加入**文本**：理想情况下 "a cute fluffy cat" 的表示应靠近猫图，"my favorite dog is a golden retriever" 的表示应更靠近金毛而非其他狗，于是图像与文本都可以被查询。

CLIP 的做法是保留左侧的图像编码器，右侧换成**文本编码器**，让它嵌入每张图对应的文字描述。训练信号仍是熟悉的形式：从互联网收集海量图文对，在一个 mini-batch 内做对比目标——分子把配对的图文表示拉近，分母把 batch 内所有不配对的组合推远；并且这是**对称互补的两个目标**，既要求每张图最接近它的文本、也要求每段文本最接近它的图像。因为图文关联数据在网上几乎取之不尽，这个目标可以做到极大规模。OpenAI 于 2021 年发布 CLIP，用的正是这种对比目标。

训练完成后沿用自监督学习的两阶段流程：第一步预训练，第二步取出图像编码器的权重，在上面加一个线性层即可适配图像分类或检测，也可以接一个解码器输出语义分割图。真正引起轰动的是，**只加一个线性分类器**就带来大幅性能提升——下图是 27 个图像分类数据集上的 linear probe 平均得分与每张图前向计算量（forward-pass GFLOPs/image）的关系，CLIP 系列（红色）整体位于顶部，且随训练图像增多持续走高：

![CLIP特征在不同数据集上线性探针平均得分与计算量对比图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0002_00-09-19.jpg)

在准确率与计算量的权衡上，CLIP-ViT 与 CLIP-ResNet 优于 EfficientNet-NoisyStudent、Instagram-pretrained、SimCLRv2、BYOL、MoCo、BiT、ResNet 等一众模型，其中 $L/14@336px$ 表现最好。不过这还不是终点：讲者指出，语言模型之所以能「开箱即用」，是因为预训练目标本身就是 autocomplete，每个任务都能被改写成补全；而 CLIP 训完后要适配新任务仍需要训练数据和一个要重新训练的线性层。理想状态是让 CLIP 像 GPT 一样直接可用。

## 3. 零样本分类：用文本编码器当分类器 [09:51 - 14:38]

语言模型的适配逻辑是：预训练时学 "I love" → "cake" 这样的补全，下游任务（情感分析、数学、符号推理）全是「语言任务」，因此无需重训模型。CLIP 没有补全过程，于是人们想出一个巧妙技巧——**用文本编码器引导模型泛化到任意分类任务**，本质是把分类改写成一次最近邻检索：

1. 取新数据集的全部类别（如 plane、dog、bird），逐个送入文本编码器得到类别向量 $T_1,\dots,T_N$，它们在文本空间里充当类别中心。
2. 新图像送入图像编码器得到 $I_1$，计算它与各类别向量的相似度 $I_1\cdot T_1,\ I_1\cdot T_2,\ \dots,\ I_1\cdot T_N$。
3. 取相似度最大者作为预测类别，即 $\arg\max_k\, I_1\cdot T_k$。

这等价于一个 **1-nearest neighbor** 分类器：类别标签由文本向量生成，再对图像向量做最近邻。完整流程如下图所示，来自 Radford 等人的 "Learning Transferable Visual Models From Natural Language Supervision"：

![利用CLIP进行零样本图像分类的完整流程示意图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0003_00-14-38.jpg)

细节决定效果。第一，**单个词往往不够**：CLIP 是从互联网短语上训练的，孤立出现的单词很少，因此改用模板 "a photo of a {object}." 比直接嵌 "dog" 更好，仅这一改动就在 ImageNet 上带来约 **1.3%** 的提升。第二，**挑哪个短语很难**，所以通常不只用一条：对每一类准备很多模板（a photo of a dog、a drawing of a dog 等），把得到的多个向量取均值 $\bar T_k=\frac{1}{M}\sum_{m=1}^{M} T_{k,m}$ 作为该类的最终中心，再回到同样的 1-NN 流程。第三，**取编码器的哪一层输出**：若图像编码器是 ResNet 就取最终向量；若是 ViT / Transformer，则通常取 CLS token。

## 4. 分布外泛化、成因与 CoCa [14:38 - 20:12]

在 ImageNet 上表现好并不稀奇（CLIP 很可能训练时见过），真正令人兴奋的是在 CLIP 发布**之后**才收集的数据集上的表现。例如 ObjectNet：人们把香蕉放在地上、拍腐烂的香蕉、把物体摆在非常规位置与形态。在 ImageNet 上训练的模型在这类数据上明显下滑，因为 ImageNet 只包含各类物体最典型的样子；CLIP 却几乎不掉点。讲者把原因归为两点：其一，互联网文字携带的监督远不止类别标签，还包含形状、颜色、结构等丰富信息，这些额外线索帮助模型适应轻微分布外或外观不同的物体；其二，**规模**——ImageNet 约 130 万张图，而互联网可轻松下载数十亿图文对，模型见过的数据量完全不在一个量级。

一组对比数值（左侧为 ResNet、右侧为 CLIP 零样本）把这种鲁棒性量化得很清楚：ImageNet V2 64.3% → 70.1%，ImageNet Rendition 37.7% → 88.9%，ObjectNet 32.6% → 72.3%，ImageNet Sketch 25.2% → 60.2%，ImageNet Adversarial 2.7% → 77.1%。素描与对抗样本上的差距尤其悬殊。

![CLIP在图形、素描及对抗性数据集上的泛化能力表现汇总图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0004_00-16-40.jpg)

讲者还对比了 zero-shot 与 linear probe：加上线性层并稍作训练，在多数数据集上确实继续提升，但并非总是如此——有些数据集上 CLIP 零样本直接开箱即最好。正因为这种「无需适配就能泛化到全新数据集」的能力，很多人把 CLIP 称为**第一个图像基础模型**。其规模支撑是：把架构从 ResNet 换成带约 **3.07 亿参数**的 Transformer（ViT），并用约 **4 亿图文对**训练；模型规模与数据规模同时放大，才换来这些收益。

紧随其后的重要变体是 2022 年的 **CoCa**。它保留 CLIP 的对比损失，但额外加了一个解码器：把图像编码器的特征经 cross-attention 送入解码器，为图像生成 caption。动机很直接——判断「这是猫还是狗」只需很少信息，而用文字**描述**一张图要求模型学到多得多的东西，因此 captioning 是更强的学习目标。结果也印证了这一点：相比 CLIP，CoCa 在各 ImageNet 变体上都有提升，总体约 **10%** 的性能增益，而且是这类基础模型第一次全面超过用监督学习训练出来的模型。讲者视之为转折点：此后大家基本放弃用监督目标训练图像编码器，转向基于互联网数据的自监督／弱监督预训练。

## 5. CLIP 式模型的优势与局限 [20:12 - 25:49]

先看优点。点积这一形式极其高效：训练目标简单、便于 scale；推理也快，只要把整个数据集预先嵌入成向量，分类就退化成检索，讲者举例说甚至可以在 **50 亿张图**上做检索，这让它同样适用于搜索与检索任务。第二是**开放词汇**（open-vocabulary）：可以喂进任意文本描述做零样本泛化，跨领域适用性强。第三是易于被**级联**到其他模型（后文的 CuPL 即是一例）。

![CLIP风格模型的三大优势：点积高效、开放词汇、可级联](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0005_00-20-58.jpg)

缺点同样尖锐。首先是**过度依赖 batch size 学习概念**：对比目标里负样本来自同一个 batch，若 batch 太小，模型总是在比较「猫 vs 卡车」这种过于容易的组合，只能学到很粗的高层表示；batch 变大才更可能遇到与猫相近的其他动物，从而被迫学出细粒度区分。讲者举例，把 batch 增到 **32,000**、跨大量 GPU 训练，模型才开始能分辨 Welsh corgi 与其他 corgi——关键在于 batch 中要出现足够接近的**难负样本（hard negatives）**。但即便如此，学到什么仍受训练数据随机性摆布，而且 32,000 的 batch 对绝大多数实验室根本不可行。

![CLIP模型的劣势：过度依赖批量大小学习概念及组合性缺陷示例](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0006_00-23-18.jpg)

第二是**组合性（compositionality）缺失**：CLIP 分不清「草地里的杯子」与「杯子里的草」，也就是把 mug、grass 以及二者关系正确组合起来的能力不足。Winoground、CREPE、ARO 等基准（其中不少出自讲者实验室）反复暴露这类失败；画面给出的例子里，"the grass is eating the horse" 得 81%、"the horse is eating the grass" 得 78%，模型几乎无法用语序与关系区分二者。社区曾尝试手工构造 batch、把相似样本放进去做 hard-negative 训练，但讲者后续的论文发现这反而让模型「忘掉」不少语义、跨环境泛化明显变差，理论上至今没有解释，如何构造数据集、batch 与训练信号仍是开放问题。第三，图像级 caption **缺少 grounding**：它不告诉你人在哪、车在哪、街在哪，位置与关系信息完全缺失。第四，**数据永远不够**：即使收集到 50 亿张图，仍覆盖不了所有重要概念，于是「如何过滤互联网、筛出最好的训练数据」（data filtering）成为当下这一领域的研究前沿。

## 6. 视觉语言模型 LLaVA：把图像 token 喂进 LLM [25:49 - 31:26]

近两三年流行起一类新的基础模型——多模态语言模型／视觉语言模型（VLM）。动机写在一张幻灯片上：能做 next token prediction 的语言模型在推理时可适配数学、情感分析、符号推理等极广的任务，那么能否构建一个**接受图像和文本输入、输出文本**的模型？

![LLaVA的动机介绍：基于next token prediction的语言模型可应用于多种任务，引出构建接受图像和文本输入的视觉语言模型](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0007_00-26-47.jpg)

讲者特意做了历史澄清：这个想法并非全新，2019 年的 **ViLBERT** 就已把图像模型与语言模型组合起来做跨任务泛化，只是它处在 pre-transformer 时代、主要用 LSTM，而且是针对单个任务训练。LLaVA 之所以是「重生」，在于换成了 Transformer 架构、更好的目标函数，以及在互联网数据上面向多种任务做预训练。

从 Transformer 的角度看，语言模型做的事是对历史上下文做 self-attention："Cats are so" → 预测出 "cute"。视觉语言模型则是给这段对话**加上图像的 grounding**：把图像以某种方式 tokenize，与文本 token 一起送进模型，继续自回归地生成关于这张图的词。于是核心问题变成：**用哪些图像 token？** LLaVA 的答案是用 CLIP 图像编码器。

![LLaVA核心思想示意图：将图像token和文本token拼接后输入Transformer block，输出预测token](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0008_00-29-07.jpg)

具体选择有讲究。最直觉的是用 CLS token，但它受对比目标监督、其余 patch token 从未被任何目标使用过，经验上这些特征并不好用；实践中的结论是取 CLIP 编码器**倒数第二层（penultimate layer）**的 patch 特征——它们被用来生成最终的 CLIP embedding，保留了大量关于物体在图像中位置的空间信息。完整架构是：图像经预训练 CLIP 编码器 → 提取特征 → 过一个**需要训练的线性层**（connector，负责把 CLIP 表征转换成 LLM 能理解的形式）→ 与文本 token 拼接送入语言模型 → 生成关于图像的对话。也就是说，架构上真正新增的可训练部分只是一个线性映射。

## 7. Flamingo：门控交叉注意力与上下文学习 [31:26 - 40:47]

Google 紧随 LLaVA 发布了 Flamingo，整体思路同样是「视觉编码器特征 + 大语言模型」，创新点全部落在**如何融合**上。LLaVA 把视觉特征经线性层放在输入层一次性喂入；Flamingo 则把视觉编码器的输出送进 LLM 的**每一层**，因此必须改动 LLM 架构本身。

![Flamingo预训练部分架构图：视觉编码器处理图像数据，与文本交错输入LM block，输出文本](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0009_00-32-37.jpg)

训练数据是图文交错的序列：`<image> This is a very cute dog. <image> This is ...`，每张图都被嵌入并送往每一层，模型的输出是补全最后一张图的描述（如 "a very serious cat."）。这来自 Alayrac 等人的 "Flamingo: a Visual Language Model for Few-Shot Learning", 2022。架构上做了两处改动：在每层 LM 前插入 **gated cross-attention（GATED XATTN-DENSE）**模块，并加入 **Perceiver Resampler** 把图像表示下采样成维度更小、数量固定的 token，便于逐层注入。冻结策略很关键：语言模型权重与视觉编码器**全部冻结**，只训练 Perceiver Resampler 和每层新增的交叉注意力层。

![Flamingo gated cross-attention 架构示意图，展示 LM layer 与 GATED XATTN-DENSE 模块及内部 self/cross attention 结构](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0010_00-34-57.jpg)

门控交叉注意力的内部结构由本课已见过的组件拼成：cross-attention 里令 $K=V=[X]$（视觉特征）、$Q=[Y]$（语言特征），随后接一个 $\tanh$ 门控决定保留／遗忘哪些图像信息，再经 FFW 全连接微调表示，之后又是一次 $\tanh$ 门控；两个组件各带残差连接，然后回到正常的语言模型层继续生成下一个词。讲者强调，代码上这只是「加一层 cross-attention + 中间一个 tanh 非线性」约两三行的改动，但对模型而言变化巨大——它从此能在**每一层**自主选择何时、关注图像的哪一部分。

训练样本被拼接成长序列（"here some cute pictures of my pets" + 图 1 + 描述 + 图 2 + 描述……），并用手工设计的 **masking scheme** 保证生成某张图的描述时只 attend 那一张图的特征、不看别的图；但训练时模型能看到整个上下文，因而学会处理多图、多轮的长序列。这直接换来一系列应用：

![Flamingo 多轮对话结果展示，包含月球泰迪熊、火烈鸟图像对比及苹果贴纸识别的问答示例](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0011_00-37-16.jpg)

- **多轮对话**：先说 "two teddy bears on the moon"，追问 "What are they doing?" 能答 "having a conversation"，再问物体能答 "looks like a computer"，甚至解释 "I think it is surprising because teddy bears are not usually found on the moon"——推理能力继承自预训练好的大语言模型。
- **多图推理**：三张图的共同点 "They are all flamingos"，差异是「第一张卡通、第二张真实火烈鸟、第三张 3D 模型」。
- **细粒度 VQA**：苹果贴纸问答中读出贴纸写着 "iPod"、判断是 handwritten、白色、照片摄于 backyard。

![Flamingo 上下文学习（in-context learning）能力演示，涵盖动物识别、艺术鉴赏、OCR 及数学推理任务](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0012_00-40-08.jpg)

更重要的是**上下文学习（in-context / few-shot learning）**：给若干「图 + 描述」示例，新图就能补全描述（chinchilla 主要产于 Chile、shiba 流行于 Japan、→ flamingo 产于 Caribbean and South America）；给「图 + 问题 + 答案」示例，就能做画作问答（The Hallucinogenic Toreador、Louvre Museum Paris、Arles）、OCR（"Underground"、"Congress Ave 400"、"Soulomes"）甚至算术（2+1=3、5+6=11、3x6=18）。把示例全部拿掉就是 zero-shot。讲者顺带澄清：并不是把图像 token 与文本拼接，只有文本被拼接输入，图像 token 是经 Perceiver Resampler 送入每一层、由模型自行决定 attend 何处；web 界面背后会缓存模型状态以支持连续对话。意义在于，许多原本必须专门适配 CLIP 才能做的困难任务，Flamingo 零样本或少样本即可完成，基准成绩大幅提升，领域也自此从「报告几个分类基准」转向「只要能框成问答，就为任何理解技能建基准」。

## 8. 闭源 VLM 的差距、Molmo 与数据质量 [40:47 - 54:08]

LLaVA 的成功让大量公司重金投入，GPT-4o / GPT-4V、Gemini 1.5 Pro / Flash、Claude 3 Opus（以及后来的 Claude 4 Opus）等 API 模型相继发布。在 11 个较流行的视觉理解基准上平均，开源的 LLaVA 约 **43%**，而闭源模型处在 high-70s 到 80s，差距巨大。开源社区的应对是把 GPT／Gemini **蒸馏**成开源变体（Alibaba 的 Qwen、InternVL、Phi 等），但讲者指出这带来一个根本问题：这些模型「open 而不真 open」——研究社区其实并不知道如何独立造出高性能 VLM，关键 tricks 只在 OpenAI 与 Google 内部；若 GPT 不存在，这些模型也就无从复现。缩小这一差距、把构建方法扩散给整个社区，正是他近几年的研究议程。

成果是 **Molmo**：完全开放——open weights（可下载模型）、open data（可下载训练集与评测集）、open code（有足够 GPU 就能自行训练）。为检验真实用户体验，团队发布了 playground 并做了大规模人类偏好评估：约 **870 名标注员、325,000 次两两比较**，让人们在模型输出之间选更喜欢哪一个。

![VLM人类偏好评估Elo评分排名图，展示了GPT-4o等模型在325k次两两比较中的表现](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0013_00-45-26.jpg)

Elo 排名为 GPT-4o 1078.88、Molmo-72B 1076.42、Gemini 1.5 Pro 1074.47；Molmo 以讲者所说的「约 1 分之差」排第二，与 GPT-4o 近乎掷硬币，同时已超过 Gemini 1.5 Pro 与 Claude 3.5——而这只来自一个小型研究 lab。团队还发布了 **7B** 模型，单张 GPU 就能放下并微调，于 2025 年 9 月开源，社区（包括 NVIDIA）热议「永远不要赌开源输」，Meta 随后推出 Llama 3.2，多项对比中 Molmo 仍占优。

Molmo 做对的第一件事是**把决策 grounding 在像素本身**：问「有几条船」时，模型先逐个指向它数到的船，再给出最终数字，因此必须为生成找到图像证据。第二件事，也是最大的差异，是**数据**。Meta 的 Llama 用了约 60 亿图文对，Molmo 只用了 **70 万**且全部手工精挑。原因在于互联网图文对是 *incidental* 的：配文往往写上传者的主观感受，很少真的描述图像内容，而大量关于视觉世界的知识人类根本不会说出来——比如「某物在另一物左边」这种空间关系太显而易见，没人会特意打字说明。

![PixMo数据示例：一张办公桌照片及其对应的详细自然语言描述，展示空间关系词汇的标注](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0014_00-48-55.jpg)

**PixMo** 的数据因此是「有意图的」：对单张图给出密集描述，专门 eliciting 互联网上缺失的信息——空间关系（to the right of / On the left side）、尺寸（large）、形状（rectangular）、材质（polished, rich）、以及在画面中的分布（spans horizontally across the image）。示例里一张办公桌被描述为「摆在画面正中、大型长方形、抛光深色木质、横向贯穿画面」；另一例是手机／平板屏幕，标注了设备类型、显示时间、剩余电量这类对使用者有用却从不会出现在网上的信息。为找出这些缺失信息，团队花了约两年做 elicitation studies；一个关键设计是让标注员**用说的而不是用打的**——说话能自动打破 Gricean maxims 之类的会话刻板，引出人们通常不会键入的内容。值得注意的是，模型结构与 LLaVA 没有区别（CLIP 编码 + 线性 connector + LLM），差别全在数据的质量与密度。

![从感知到行动：展示在HuggingFace界面上通过指向菜单、搜索选项和筛选器来执行操作的三个示例](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0015_00-51-08.jpg)

这种 grounding 带来了别处做不到的能力：在 HuggingFace 界面上执行 "Point to the menu"、"Point to where I can set search options"、"Point to where I can find mid size datasets"；读公交车路线号时先指向图中号码所在区域再返回数字；比较左／右两侧车数、处理深度图、俯视图、拥挤场景与体育场景。更进一步可以**级联**：把 Molmo 输出的点喂给 SAM 2 做跨时间的分割（例如指出 cricket bat 再分割它），或者串起机器人动作——指出水瓶 → 用简单运动规划器把机器人移过去 → 指向水槽里的脏盘子处 → 再指向水槽空位把瓶子放进去。关于「固定分辨率能否 pointing」的疑问，讲者答道如今可用 FlexiViT 一类机制支持任意尺寸输入，位置嵌入随图像大小变化，模型通常泛化良好。

## 9. 超越文本与类别：SAM、链式组合与视觉编程 [54:08 - 69:46]

讲者再次回到开场那张全景图，指出最后一类是把基础模型推广到**任意输出空间**：输出不必是类别或文本，也可以是 mask 或图像。

![基础模型分类列表，涵盖语言、分类、视觉语言及链式组合等类别的代表性模型](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0016_00-54-11.jpg)

### 9.1 SAM：可提示的分割基础模型 [54:11 - 60:24]

Segment Anything Model（SAM，Kirillov et al. "Segment Anything", 2023）要做「所有分割任务的基础模型」：让用户指向图中任意关心的东西，模型就为它输出一个 mask。这里有两个并列挑战——如何覆盖任意／海量类别，以及如何精确输出用户真正所指的那一个。难点是**歧义**：图中有两只猫时用户说「分割 the cat」并不知道指哪只（若有 Molmo 的 pointing 能力就能指定）；即便给一个点也仍歧义，比如点在一把剪刀上，可能指整把剪刀、指可握的把手部分、或只指其中一个把手。为不因「猜错粒度」而惩罚模型，SAM 一次输出 **3 个不同粒度的 mask**，取与 ground truth 最接近的那个计算损失，其余不参与惩罚，久而久之模型学会给出多种粒度、由用户挑选。

![SAM（Segment Anything Model）的架构流程图，展示图像编码器、提示编码器和掩码解码器的交互](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0017_00-57-04.jpg)

架构由三部分组成：image encoder（可以用 CLIP 编码器）、**prompt encoder**（把 text、points、box、mask 等用户指定方式编码）、以及一个轻量的 **mask decoder**，与本课程讲过的分割解码器很像，输出掩码。剩下就差数据了：2023 年之前分割数据集都极小，作者把图像数量扩大约 **6 倍**、把 mask 数量扩大约 **400 倍**。做法是一个 model-in-the-loop / human-in-the-loop 的迭代闭环：先人工标注一部分 → 训练模型 → 用模型标注更多数据 → 人工修正模型提出的 segment → 再训练，如此反复，最终在数百万张图上让每个蔬菜、每把伞、水下生物乃至画作中的对象都有独立 mask。讲者把结论与 Flamingo、Molmo 归为一句话：**高质量数据是根本**，而很多视觉任务的数据互联网上完全缺失，必须主动出去采集。

### 9.2 Chaining：把基础模型组合起来 [60:24 - 66:42]

Chaining 的想法是组合不同模型，做出单一模型做不到的事。课堂练习很直观：给 4 张图和 4 个 CLIP 从没见过的类别（Morima、viaduct 等），CLIP 直接分类会失败；可一旦给出这些类别的**文字描述**，人就很容易把图与类对应起来。原因是这些概念在互联网上大概率被讨论过，GPT 能生成描述，而描述恰好是 CLIP 擅长匹配的文本——于是「用语言模型的长处补 CLIP 的短处」就产生了新能力。**CuPL**（CUsomized Prompts via Language models）正是这一思路：让语言模型为每个类别生成定制提示再做零样本分类，在 ImageNet、DTD、Stanford Cars、SUN397、Food101、FGVC Aircraft、Oxford Pets、Caltech101、Flowers 102、UCF101、Kinetics-700、RESISC45、CIFAR-10/100、Birdsnap 等 14 个数据集上普遍提升（例如 ImageNet 由 std 75.54 提到 CuPL base 76.19、DTD 由 55.20 提到 58.90），对细粒度专门类别尤其有效。

![CuPL方法在多个标准数据集上的零样本分类准确率对比表](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0018_01-02-20.jpg)

另一类例子是计数：「船上有三个人吗」不必让 VLM 直接回答，用现成的目标检测器数出三个检测框即可；「两条船总共是不是六个人」就写一段程序对 image1、image2 分别检测再把结果相加。这被 **VisProg**（visual programming，去年的最佳论文）系统化：对任意图像或问题，让 GPT 生成一段 **Python 程序**，程序里调用已有的模型（目标检测、人脸检测、VQA 等）并组合它们的输出得到最终答案；对比的是「为你的任务训练一个新模型」与「用你已有的模型写一段脚本」两条路线。

![VisProg视觉编程方法对比：训练新模型与编写Python脚本调用现有模型](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0019_01-04-04.jpg)

代码示例里 `MyMultiImageVQA` 对两张图各做一次 VQA 再把答案相加，但只能泛化到 2 张图、再多就不行。要用好它，必须把可用函数（能力清单）告知 GPT，并用 in-context 示例教它生成程序；示例可以静态地尽量多样地给出，也可以按当前问题**动态检索**最合适的示例（后者效果更好，但依赖一个好的检索系统）。代价也不小：调 GPT 要走 API，每个模型都要加载进内存并顺序执行，算力成本可能反而更高，因此 2025 年一个很大的研究方向是把这些能力**蒸馏进单一模型**。讲者也认可把整套系统看作一个 **agent**：由它决定「给定这个问题该找哪些模型帮忙、如何把它们拼起来」。

### 9.3 视觉编程实例、总结与未竟问题 [66:42 - 69:46]

一个图像编辑的例子：指令是 "Replace desert with lush green grass"。由于图像编辑模型尚不成熟，更稳妥的做法是调用分割模型识别沙漠、只把那些像素替换成草地再合成：

```python
OBJ0=Seg(image=IMAGE)
OBJ1=Select(image=IMAGE, object=OBJ0, query='desert', category=None)
IMAGE0=Replace(image=IMAGE, object=OBJ1, prompt='lush green grass')
RESULT=IMAGE0
```

其中 Seg、Select、Replace 是三个核心算子。

![VisProg视觉编程示例：将沙漠替换为绿草地的指令与程序代码](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0020_01-06-42.jpg)

全讲收束成一句话：基础模型的本质是「为单一目标训练，再泛化到大量下游应用」。分类侧靠互联网图文对预训练，泛化到现实里可能根本不存在标签的新数据集；与语言模型结合后靠上下文示例完成 captioning、counting、OCR；输出还可以是随用户输入变化的分割掩码；再进一步通过程序把基础模型乃至小模型组合，做出全新能力。**幻觉**依然普遍存在：pointing 因为必须为生成找到像素证据，确实能减少幻觉，但并不能保证指对；改进路径一是采集更多与目标推理相关的数据，二是引入 **verification**——用指向的点去验证输出是否可信，大公司的产品通常不是一个模型直接吐结果，而是先经过若干 verifier 才送到用户面前。现场有听众问「当能力需要一个尚不存在的工具时，模型能否自己造工具」，讲者回答可以，已有初步实验让系统自动收集训练数据、为特定用途构建工具，但这一方向仍处在非常早期。

## 要点回顾

- 基础模型范式：从「单任务收集数据 + 训专用模型」转向「大规模多技能预训练 + 极少／零数据适配」；特征是通用鲁棒、参数大、数据大、自监督目标，分类涵盖 Language / Classification / LM+Vision / And More! / Chaining。
- CLIP 用图像编码器 + 文本编码器，在约 4 亿互联网图文对上做**对称**对比损失（image→text 与 text→image 双向），预训练后加一个线性层即可适配分类／检测／分割；linear probe 在 27 个数据集上领先，$L/14@336px$ 最佳。
- 零样本分类把类别填进模板 "a photo of a {object}."（ImageNet 约 +1.3%）、多模板向量取均值作类别中心，图像向量做点积／1-NN；ResNet 取最终向量、ViT 取 CLS token；ViT 约 3.07 亿参数。
- 泛化来源是「文本提供形状／颜色等超出标签的监督」加「数据规模」；在 ObjectNet、Sketch、Adversarial 等分布外数据显著优于 ResNet（对抗样本 2.7% → 77.1%）。CoCa 加 captioning 解码器，约 +10%，首次全面超越监督模型。
- CLIP 优点：点积高效易扩展、可在 50 亿图上检索、开放词汇、可级联；缺点：依赖超大 batch（32,000）才有难负样本、组合性缺失（Winoground/CREPE/ARO，"草吃马" 81% vs "马吃草" 78%）、hard negatives 反损语义泛化、缺 grounding、数据永远不够。
- LLaVA 取 CLIP **倒数第二层** patch 特征（含空间信息），经可训练线性 connector 与文本 token 拼接进 LLM 做自回归生成；ViLBERT (2019) 是 LSTM 时代的前身。
- Flamingo 用 Perceiver Resampler + 每层 gated cross-attention（$K=V=[X],\ Q=[Y]$，双重 tanh 门控、带残差），冻结 LLM 与视觉编码器、只训新增模块，配合交错图文 + 掩码，支持多轮对话、多图推理与 few-shot（分类、画作 QA、OCR、算术）。
- 闭源 GPT-4o / Gemini / Claude 在 11 个基准约 80% vs LLaVA 43%，开源多靠蒸馏；Molmo 全开放（权重／数据／代码），Elo 1076.42 与 GPT-4o 仅差约 1 分，靠 70 万手工精挑的 PixMo 密集描述（空间关系、尺寸、材质，让标注员「说」而非「打」）与像素级 pointing 接地减少幻觉，而非 60 亿互联网图文对。
- SAM 用 image encoder + prompt encoder（点／框／文本／mask）+ 轻量 decoder，输出 3 种粒度 mask 取最接近 GT 者算 loss；图像扩约 6 倍、mask 扩约 400 倍，靠 model-in-the-loop 迭代标注。
- Chaining：CuPL 用 LM 生成类别描述让 CLIP 分类新类；VisProg 让 GPT 生成调用现有模型的 Python 程序（Seg/Select/Replace 等）完成计数、多图 VQA、图像编辑，体现 agent 思路，代价是多模型顺序执行算力高，2025 方向是蒸馏进单模型；幻觉仍需 verification 与更多推理数据缓解。

## 视频来源

https://www.youtube.com/watch?v=mQOK0Mfyrkk
