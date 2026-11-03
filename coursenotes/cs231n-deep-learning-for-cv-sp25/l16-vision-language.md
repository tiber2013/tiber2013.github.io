---
title: "第 16 讲 · 视觉与语言：从 CLIP 到多模态基础模型"
description: "CS231n Spring 2025 L16 视频笔记；对照 2026 春季课件补充新增内容。"
---

# 第 16 讲 · 视觉与语言：从 CLIP 到多模态基础模型

> **Stanford CS231n 计算机视觉 · L16** ｜ 讲者：Ranjay Krishna（华盛顿大学，客座）｜ 70 分钟 ｜ [原视频](https://www.youtube.com/watch?v=mQOK0Mfyrkk)

> 课件版本：2026 春季 ｜ 视频版本：2025 春季。2026 课件新增或调整的内容集中在文末「2026 课件更新」一节。

## 核心结论

这一讲把课程此前的「一个任务训练一个专用模型」切换为 **foundation model** 范式：先在海量、多样的数据上预训练，再用零样本、少样本或轻量微调适配下游任务。课程沿两条线展开：以 **CLIP**、**CoCa** 为代表的图文对比学习模型，把图像和文本嵌入同一空间，用文本编码器直接构造分类器；以及 **LLaVA**、**Flamingo**、**Molmo** 这类多模态语言模型，把视觉 token 接入 LLM，做开放式对话与推理。讲者反复强调，模型结构往往只是几行改动，真正决定效果的是数据规模与数据密度——Molmo 只用 70 万条人工精标的图文对，就在用户偏好上追平了 GPT-4o，也超过了用 60 亿条网络数据训练的 Llama 模型。最后，**SAM** 把基础模型推广到分割掩码，**chaining** 把多个模型组合成程序；组合性、像素级定位与幻觉仍是未解问题。

## 从专用模型到基础模型 [00:05 - 04:16]

本讲是客座 lecture，讲者是华盛顿大学助理教授 Ranjay Krishna，研究横跨计算机视觉、自然语言处理、机器人与 HCI，曾在 2020 和 2021 年讲授 CS231n。开场他先回顾课程此前的主线：收集训练集和测试集，为某个任务训练一个专门模型，例如图像分类或图像描述，最后在测试集上评估。

**Foundation model** 反过来：先在大量不同技能、不同任务上预训练，再按需适配到具体任务。GPT 是语言侧的例子：在海量 Common Crawl 数据上预训练，然后通过微调、零样本或少样本适配数学题、符号推理、常识问答、翻译。它的吸引力在于适配所需数据极少，常常完全不需要新训练数据。

如何判断一个模型算不算 foundation model？必备特征是通用、对多种任务鲁棒；常见但不必然的特征是大参数量、大数据量和自监督预训练目标。课件把基础模型分为语言、分类、LM + Vision、Chaining、And More 五类；语言模型不在本讲范围，重点是 CLIP/CoCa、LLaVA/Flamingo、组合式方法与 SAM。

![基础模型（GPT）从预训练到多任务微调/零样本/少样本的流程图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0001_00-01-56.jpg)

![判断基础模型的标准幻灯片，区分必备特征和常见特征](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0002_00-03-53.jpg)

## CLIP：把图像与文本放进同一表征空间 [04:16 - 09:44]

讲者从 **SimCLR** 讲起：对比学习把同一张图的不同增强拉近，把其他图像推远。如果表征足够通用，没见过的草图也能被正确嵌入。CLIP 接着问：能不能把文本也嵌入同一个空间？「a cute fluffy cat」应该靠近猫的图，「my favorite dog is a golden retriever」应该靠近金毛，而不是其他狗。

因此 CLIP 有两个编码器：图像编码器与文本编码器。训练数据是约 4 亿对从互联网抓取的图文对。在一个 mini-batch 内做双向对比：每张图应最接近它对应的文本，每段文本也应最接近它对应的图。损失是对称的 **InfoNCE**：

$$
\sum_{i=1}^{n} -\log \frac{e^{\langle u_i, v_i\rangle}}{\sum_{j=1}^{n} e^{\langle u_i, v_j\rangle}}
\;+\;
\sum_{i=1}^{n} -\log \frac{e^{\langle u_i, v_i\rangle}}{\sum_{j=1}^{n} e^{\langle u_j, v_i\rangle}}
$$

其中 $u_i$ 是图像向量，$v_i$ 是文本向量，内积就是相似度。第一项让图像找对文本，第二项让文本找对图像；最大化对角线上的相似度，压低其余配对。

训练完成后仍沿用自监督学习的两步流程：先预训练，再把图像编码器接到新任务上，加一个线性层做 **linear probe**，或接检测、分割等不同的头。论文的线性探测结果显示，CLIP 在多个分类数据集上的平均表现随训练数据量持续上升，红色曲线位于最上方。

![CLIP训练目标：对比学习损失函数公式与图像-文本编码器架构图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0003_00-07-46.jpg)

## 零样本分类：文本编码器就是分类器 [09:44 - 14:41]

语言模型可以直接用「补全」适配新任务，CLIP 没有这种接口。于是研究者想出一个技巧：把类别名送进文本编码器，得到一组文本向量，再对图像向量做最近邻。例如新数据集只有 plane、dog、bird 三类，就把三个词编码成三个向量；新图来了，算图像向量与它们的相似度，狗向量得分最高，就判为狗。这等价于以文本向量为「训练样本」的 **1-NN**。

单个单词往往不够好，因为 CLIP 训练时看到的多是短语。把类别换成「a photo of a plane」这样的模板，在 ImageNet 上就能提升约 1.3%。单一模板仍有偏置，于是再为每类准备多个模板，例如 a photo of a dog、a drawing of a dog，把所有模板向量取平均作为类别向量，又能带来约 5% 的提升。这样，CLIP 就成了一个无需再训练的图像分类基础模型；检索时也只需算内积，适合大规模图库搜索。

![CLIP模型图文双编码器架构图，展示图像编码器与文本编码器分别生成向量表示](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0004_00-11-39.jpg)

![CLIP零样本图像分类流程图，展示从标签文本构建分类器并进行预测的完整步骤](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0005_00-14-41.jpg)

## CLIP 的泛化与 CoCa [14:41 - 20:12]

在 ImageNet 上，CLIP 与 ResNet-101 打成 76.2% 平手，这本身不算震撼；真正有说服力的是分布偏移。ObjectNet 把香蕉放在地上、用烂掉的香蕉拍照，ImageNet 监督训练的 ResNet 只有 32.6%，CLIP 有 72.3%；ImageNet-R 是 88.9% 对 37.7%，ImageNet-Sketch 是 60.2% 对 25.2%，ImageNet-Adversarial 是 77.1% 对 2.7%。

为什么 CLIP 泛化得更好？讲者给出两个原因。第一，网络文本不只包含类别名，还包含形状、颜色、材质、场景等结构信息，相当于更丰富的监督。第二，规模：ImageNet 只有约 128 万张图，CLIP 用了约 4 亿图文对，模型也从 ResNet 换成 3.07 亿参数的 ViT。零样本与线性探测各有胜负，多数数据集上线性探测更好，但零样本已经足够可用。

![CLIP在ImageNet及多个分布偏移数据集上的性能对比柱状图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0006_00-16-40.jpg)

**CoCa**（2022）在 CLIP 的对比目标之外加了一个解码器：用 cross-attention 取图像特征做 captioning。动机是「描述一张图」比「判断猫还是狗」需要更多信息，应该是更强的学习目标。结果 CoCa 在 ImageNet 各变体上平均提升约 10%，也是第一批全面超过有监督模型的分类基础模型。此后，图像编码器的主流从有监督学习转向网络数据上的预训练。

## CLIP 的优势与局限 [20:12 - 25:53]

CLIP 的优点很实在：点积目标简单，容易训练，也容易扩展；推理快，可以在 50 亿张图的嵌入上做检索；**open vocabulary**，任何文本描述都能拿来查询；还很容易与其他模型串联。

局限同样明显：

- **组合性**：CLIP 分不清「草地里的杯子」和「杯子里的草」，它把概念嵌入空间，却没有建模概念之间的关系，Winoground 等 benchmark 反复暴露这一点。
- **依赖 batch size**：对比学习需要 batch 内有足够多、足够难的负样本。batch 很小时只学到 animal，稍大是 dog，扩大到 3.2 万才区分 Welsh Corgi；而 3.2 万的 batch 对多数实验室并不现实，再大也未必同时覆盖「mug in grass」与「grass in mug」。
- **hard negatives 的副作用**：社区曾手工构造难负样本，但后续工作发现这样会让模型「遗忘」语义，跨数据集的泛化反而变差。
- **图像级描述不够**：要区分「有人在过马路」和「车在哪、路在哪」，需要区域级、带框的描述监督。
- **数据永远不够**：即便 50 亿张图也覆盖不了所有重要概念，数据筛选本身已成为研究前沿。

![展示CLIP无法区分“草地上的杯子”和“杯子里的草”的局限性示例](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0007_00-21-21.jpg)

![CLIP类模型的缺点及区域级标注训练示意图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0008_00-25-14.jpg)

## 多模态语言模型：LLaVA [25:53 - 31:26]

LLM 的下一个 token 预测很适合适配新任务，那么能否让模型接受图像和文本、输出文本？这就是 **vision-language model**。历史上，2019 年的 **ViLBERT** 已经把视觉与语言模型拼在一起，但它仍要为每个任务单独设计微调方式，属于专用模型范式。LLaVA 的复兴靠的是更好的架构、目标和预训练数据。

LLM 生成时只 attend 历史词序列；VLM 的做法是把图像也 token 化，与文本 token 一起喂入，让模型「看图说话」。难点是图像 token 用什么。LLaVA 用 CLIP 图像编码器：CLS token 经过对比学习，但 patch token 从未被监督，直接使用效果不好；改用倒数第二层的 patch token，它们保留了最好的空间与语言信息。这些特征经过一个可训练的线性层，映射到 LLM 的输入空间；LLM 与编码器用预训练权重初始化，再联合微调。十万量级的图文指令数据就能得到合理效果。

![ViLBERT双流架构图：图像与文本分别编码后经Co-TRM交互](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0009_00-27-11.jpg)

![LLaVA整体架构与训练流程：CLIP视觉编码器经线性层接入LLM解码器](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0010_00-31-04.jpg)

## Flamingo：把视觉特征注入每一层 [31:26 - 41:26]

Flamingo 沿用「视觉编码器 + LLM」，创新在融合方式。LLaVA 只在输入端加线性层；Flamingo 把视觉特征送进 LLM 的每一层，为此加入两个新模块：

- **Perceiver Resampler**：把可变数量的图像 token 采样、压缩成固定数量的 token；
- **gated cross-attention dense**：在每层 LLM 之前插入交叉注意力，让语言 token 查询图像特征，再经 tanh 门控与残差接回。

视觉编码器与 LLM 全部冻结，只训练这两个模块。门控让模型自己决定每一层要看图像的哪些部分、保留多少；代码上只是两三行，能力上却是巨大的变化。

![Flamingo 门控交叉注意力（GATED XATTN-DENSE）模块的内部结构图，展示了视觉特征与语言特征如何通过 tanh 门控进行融合。](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0011_00-34-57.jpg)

![Flamingo 模型的完整架构图，展示了视觉编码器、Perceiver Resampler 以及交错在语言模型中的门控交叉注意力层。](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0013_00-40-14.jpg)

训练数据是图文交错的长序列：图 1 + 描述 1 + 图 2 + 描述 2……并用掩码保证描述某张图时只能看到那张图的特征。这样模型学会多图、多轮对话，并继承 LLM 的推理能力：描述完「两只泰迪熊在月球上」，还能追问「它们在做什么」「用什么物体」。它也能做 **in-context learning**：给几个「图—答案」示例，再问新图，分类、OCR、算术都能用同一接口。Flamingo 让评估范式从若干分类 benchmark 转向任意「可以写成问答」的任务。

![Flamingo 模型进行上下文学习（In-context Learning）的示例，展示了通过图文交替的提示词（Prompt）完成图像分类、问答和数学计算等任务。](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0012_00-38-50.jpg)

## 开放与闭源的差距，以及 Molmo [41:26 - 54:08]

LLaVA/Flamingo 之后，工业界迅速推出 API 模型：GPT-4o、GPT-4V、Gemini 1.5 Pro/Flash、Claude 3 Opus 等。在 11 个常用视觉理解 benchmark 上，LLaVA-1.5 平均约 43%，而这些模型大多在 70 多到 80 附近。开放权重模型（Qwen2-VL、InternVL 等）分数接近，但讲者认为它们大多是从 GPT/Gemini 蒸馏出来的：能复现结果，却不知道如何从零训练。这是研究社区的核心问题，也是他自己研究议程的起点。

![不同AI模型在API、开源权重、蒸馏和开源等分类下的性能对比条形图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0014_00-42-42.jpg)

**Molmo** 的目标是「完全开放」：开放权重、数据、代码与评测。它的关键设计是让推理扎根像素：问「有几艘船」，模型先逐艘点出位置再给数字，而不是直接猜一个数；也能指菜单项、指公交线路号、比较左右车辆数。因为必须找到证据，指认能在一定程度上减少幻觉。

数据上，Meta 的 Llama 模型用了约 60 亿图文对，Molmo 只用了 70 万对，但全部人工精标。网络文本是「附带」的：多是上传者的主观感受，如 pink、japan、aesthetic，很少描述图像内容本身；而人不会主动说「A 在 B 左边」，因为太显然。PixMo 反过来刻意采集这些信息：大小、形状、材质、位置，甚至屏幕上的时间与电量。标注员被要求口述 60 到 90 秒而不是打字，口语会绕开书面语的习惯，说出更多平时不会写的内容。模型结构与 LLaVA 几乎一样——CLIP、线性连接器、LLM——差别全在数据。

![Molmo与Gemini 1.5 Flash在数船只任务上的对比演示界面](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0015_00-47-31.jpg)

用户研究给出了直接证据：870 名用户、32.5 万次成对比较，Molmo 的 ELO 仅次于 GPT-4o（差 1 分），超过 Gemini 1.5 Pro 与 Claude 3.5。7B 版本可单卡运行，2024 年 9 月发布后社区反响很大，Meta 随后发布了 Llama 3.2。Molmo 输出的点还可以接 SAM 2 做跨时间分割，或接机器人运动规划器完成「找到水瓶—移到水池—放进空位」这类任务。

![强调数据质量优于数量的幻灯片，展示LLAMA 3.1V使用60亿图文对](/coursenotes/cs231n-deep-learning-for-cv-sp25/l16-vision-language/kf_0016_00-48-04.jpg)

## SAM：分割的基础模型 [54:08 - 60:24]

SAM 有两个目标：类别开放，能分割任意类别；输出用户真正想要的那个 mask。用户说「分割这只猫」时，图里有两只猫，文本本身就是歧义的；用点、框或文本指定就清楚得多。架构是三件套：图像编码器（可用 CLIP）、prompt 编码器（编码点、框、文本）、轻量解码器输出 mask。

即便给定一个点，仍可能指整把剪刀、可握的部分或其中一个部件，因此 SAM 同时输出三个不同粒度的 mask，训练时只对最接近标注的那个算损失，不惩罚其他两个；推理时由用户挑。数据是另一个难点：2023 年之前，分割数据集都很小。SAM 把图像规模扩大约 6 倍、mask 数量扩大约 400 倍，并用 human-in-the-loop 的数据引擎迭代：标注、训练、让模型提议 mask、人工修正、再训练。示例里有逐个标注的蔬菜、每一把伞、海底场景和绘画。结论与 Flamingo、Molmo 一致：视觉任务的高质量数据往往不在互联网上，必须专门采集。

## Chaining：把基础模型组合成程序 [60:24 - 67:09]

课堂做了一个小实验：给四张图和四个陌生类别（marimba、viaduct、papillon、lorikeet），CLIP 完全无从下手；但如果给出「A viaduct is a bridge composed of several spans supported by piers」这样的描述，人立刻能完成配对。**Chaining** 就是让 LLM 先描述类别，再交给 CLIP 做零样本分类。只要概念在网络上被谈论过，CLIP 就能借 LLM 的描述识别它。

更一般地，可以把问题写成程序。问「左右两图合计是否有 6 个人、2 条船」，可以分别调用目标检测或计数模型，再用 Python 做加法与比较。**VisProg**（2023 年最佳论文）让 GPT 根据问题和可用函数库生成这样的程序，不需要为任务训练新模型；给静态示例或按问题检索示例都可以，后者效果更好，但依赖检索质量。图像编辑也能拆：先分割出沙漠，再只替换沙漠像素为草地，最后合成。

代价是算力与延迟：要调用 GPT API，还要把多个模型载入显存依次运行。因此现在的研究一方面尝试把组合能力蒸馏进单个模型，另一方面继续改进组合方式；也可以把整个系统看成一个 **agent**，由它决定调用哪些模型、如何拼接。

## 总结与开放问题 [67:09 - 69:46]

讲者总结：基础模型的本质是「为一个任务训练，再泛化到许多下游应用」。分类线用网络图文对做对比学习，能泛化到没有标签、甚至现实中不存在的数据集；VLM 线把视觉接入 LLM，用上下文示例完成描述、计数、OCR；输出不限于文本，也可以是 SAM 的 mask；再往前一步，是把多个基础模型组成程序。

问答聚焦幻觉：指认能减少幻觉，因为模型需要找证据，但不能保证指对。解决办法一是补充相关数据，二是在输出后加验证器；大公司通常不会只让一个模型直接输出，而会让结果先过验证。另有同学问模型能否为自己造工具，讲者说已有初步实验让模型自动收集数据、构建特定工具，但仍在早期。

## 2026 课件更新

以上正文以 2025 春季视频为准。对照 2026 春季课件，主要有这些差异：

- **CLIP 细节补充**：2026 课件在 InfoNCE 公式旁注明温度系数与 L2 归一化，并加入 **SigLIP**：用 sigmoid 代替 softmax，逐对计算损失，不必物化整个 batch 的相似度矩阵，可以节省显存。
- **「为什么 CLIP 有效」的讨论**：2026 课件列出三种可能解释——「无标签」的说法有误导性、预训练规模巨大、测试集泄漏；并引用后续工作说明泄漏约 2%，不是泛化的主因。
- **评测基准扩充**：2026 课件把 Winoground、VL-Checklist、CREPE、ARO、SugarCREPE、COLA 等组合性 benchmark 整理成表，比视频里的口头列举更系统。
- **数据伦理**：2026 课件在「50 亿数据也不够」之后加入 Stanford Internet Observatory 关于生成模型训练数据中 CSAM 的报告，强调数据收集与过滤必须有意为之。
- **LLaVA 数据规模改写**：2025 视频里的 LLaVA 课件页写 >100,000 样本，2026 课件改为约 17.8 万样本。
- **2026 SOTA**：课件新增「What is SOTA in 2026 (so far)?」，认为 Gemini 是最好的专有 VLM；把开源模型分为 open weight 与 fully open source，并以 Qwen3-VL（2025 年 12 月技术报告）为例说明它与 LLaVA 的差异：原生分辨率与 2D-RoPE、SigLIP-2 视觉编码器、视频帧时间戳写入文本、使用 SigLIP-2 第 8/16/24 层特征、四阶段训练。
- **CuPL 名称与结果**：2026 课件明确写出 **CuPL**（CUstomized Prompts via Language models），并给出多数据集结果表；视频只讲了「让 LLM 生成描述再交给 CLIP」的思路，没有用这个名字。
- **SAM 未展开**：2026 课件正文没有 Segment Anything 一节，SAM 只出现在开头的模型分类表里；正文的 SAM 部分以 2025 视频为准。
- **新增 Omni 模型**：2026 课件在结尾加入 Omni Models，从 2024 年的 GPT-4o 讲到 Thinking Machines 与 Gemini 的新模型，输入输出扩展到文本、音频、视频。
- **下一讲预告变化**：2026 课件结尾写「Next time: World Models」，而 2025 视频里讲者预告下一讲是机器人学习。

## 要点回顾

- **foundation model**：先在大规模多样数据上预训练，再以零样本、少样本或轻量微调适配任务；通用性必备，大参数量与大数据量常见。
- **CLIP** 用图像与文本双编码器加对称 **InfoNCE**，把配对的图文拉近、非配对推远，训练数据约 4 亿对。
- 适配下游先做 **linear probe**；完全不训练时，用文本编码器把类别名编码成向量，对图像向量做 1-NN。
- 类别改用短语模板（a photo of a ...）并集成多个模板取平均，能显著提升零样本准确率。
- CLIP 在 ObjectNet、Sketch、Adversarial 等分布偏移数据上远好于 ImageNet 监督模型，原因是文本的丰富监督与数据规模。
- **CoCa** 增加 captioning 解码器，把生成目标加入训练，是分类基础模型全面超过有监督模型的转折点。
- CLIP 的局限：缺乏组合性、依赖超大 batch、hard negatives 损害泛化、图像级描述缺少定位信息、单一数据集覆盖不了所有概念。
- **LLaVA** 用 CLIP 倒数第二层 patch token 加线性层接入 LLM，只需十万量级指令数据；VLM 的历史可追溯到 2019 年的 ViLBERT。
- **Flamingo** 冻结视觉编码器与 LLM，只训练 Perceiver Resampler 与每层的 gated cross-attention，支持多图、多轮与 in-context learning。
- **Molmo** 用 70 万条人工精标数据（PixMo）追平 GPT-4o 的用户偏好，靠「指认」把推理扎根像素并减少幻觉。
- **SAM** 用图像编码器、prompt 编码器与轻量解码器输出三个粒度的 mask，并用数据引擎把 mask 规模扩大约 400 倍。
- **chaining** 把 LLM、CLIP、检测器、分割器等模型组合成程序，代价是算力与延迟；幻觉仍需数据与验证器共同治理。

## 视频来源

https://www.youtube.com/watch?v=mQOK0Mfyrkk
