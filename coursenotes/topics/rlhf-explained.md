---
title: "RLHF 讲透：用人类偏好训练奖励模型"
description: "RLHF 是在预训练和监督微调之后，用人类偏好继续训练大语言模型的方法。它不要求人直接写出标准答案，而是让人比较多个候选回答，再用这些偏好训练一个奖励模型。随后用奖励模型给原模型的输出打分，引导原模型生成更礼貌、更有帮助…"
---

# RLHF 讲透：用人类偏好训练奖励模型

> **专题视频笔记** ｜ 讲者 StatQuest with Josh Starmer ｜ 18 分钟 ｜ [原视频](https://www.youtube.com/watch?v=qPN_XZcJf_s)

## 核心结论

RLHF 是在预训练和监督微调之后，用人类偏好继续训练大语言模型的方法。它不要求人直接写出标准答案，而是让人比较多个候选回答，再用这些偏好训练一个奖励模型。随后用奖励模型给原模型的输出打分，引导原模型生成更礼貌、更有帮助的回答。这样可以用较低成本获得比监督微调更大的反馈数据，并缓解监督微调数据小、易过拟合、难泛化的问题。

## 背景：RLHF 与训练大语言模型的关系 [00:00 - 01:24]

RLHF（Reinforcement Learning with Human Feedback，基于人类反馈的强化学习）几乎总是用于帮助训练大型语言模型，例如 ChatGPT、DeepSeek 等。因此，视频先交代从零训练一个大语言模型的完整背景，再进入 RLHF 的细节。

要理解后续内容，需要先熟悉仅解码器 Transformer（Decoder-Only Transformer）、反向传播，以及强化学习与神经网络结合的基本概念。视频不展开这些先修知识，而是直接把它们作为理解 RLHF 的前提。

![视频标题页：RLHF（基于人类反馈的强化学习）清晰讲解](/coursenotes/topics/rlhf-explained/kf_0001_00-00-18.jpg)

![推荐先修知识的幻灯片，展示解码器Transformer、神经网络反向传播及作者AI书籍的封面](/coursenotes/topics/rlhf-explained/kf_0002_00-01-12.jpg)

## 第一步：预训练——让模型学会预测下一个 token [01:24 - 03:39]

假设要训练一个全新的仅解码器 Transformer 模型。所谓“未训练”，是指模型中的权重和偏置都只是随机初始化的值。给这种模型输入“What is StatQuest?”，它只会生成“blah blah blah”之类的无意义内容，既不礼貌，也没有帮助。

训练大语言模型的第一步，是让模型在海量文本（例如整个维基百科）上学习预测下一个 token。比如用维基百科片段“Aa! was a Japanese”训练模型预测下一个 token“girl”，再用“was a Japanese girl”预测下一个 token“group”，如此反复，用大量文本片段和普通反向传播训练模型。

视频特别提醒一个术语：在 LLM 语境中，用海量文本训练模型预测下一个 token 被称为**预训练**（pre-training）。预训练完成后，得到的是一个“预训练模型”。

![未训练的Decoder-Only Transformer模型架构图](/coursenotes/topics/rlhf-explained/kf_0003_00-01-48.jpg)

![从维基百科页面截取“Aa! was a Japanese”作为训练文本片段](/coursenotes/topics/rlhf-explained/kf_0004_00-02-42.jpg)

![从无监督模型到预训练模型的转变流程图，通过预测下一个词进行预训练](/coursenotes/topics/rlhf-explained/kf_0005_00-03-36.jpg)

## 对齐问题：预训练模型并不等于可用的聊天模型 [03:39 - 05:04]

预训练模型很擅长根据维基百科片段预测下一个 token，但这并不是人们使用 ChatGPT 或 DeepSeek 的方式。人们通常不会输入“of horizontal points has been”并期待模型补出“changed”，而是会输入“What is a good YouTube channel for learning about AI?”这样的自然语言问题，并期待礼貌、有帮助的回答。

因此，只经过预训练的模型与人们实际使用它的方式之间存在偏差，视频称之为“未对齐”（unaligned）。接下来需要把预训练模型**对齐**到人们期望的使用方式。对齐通常分为两步：监督微调（Supervised Fine-Tuning）和基于人类反馈的强化学习（RLHF）。

![未对齐预训练模型（Unaligned Pre-Trained Model）的Transformer架构示意图](/coursenotes/topics/rlhf-explained/kf_0006_00-04-48.jpg)

![展示未对齐预训练模型结构及输入提示词示例的示意图](/coursenotes/topics/rlhf-explained/kf_0007_00-05-24.jpg)

## 第二步：监督微调——用人工问答对初步对齐 [05:04 - 07:36]

监督微调使用一个由人工创建的提示词和人工创建的回答组成的数据集。例如，提示词是“What is a good YouTube channel for learning about AI?”，对应的人工回答是“StatQuest is a good channel to learn AI.”。有了这样的提示词—回答对，就可以用普通反向传播训练模型生成该回答。

监督微调会把“未对齐的预训练模型”变成“开始对齐的模型”，使其能够对特定自然语言提示词生成礼貌、有帮助的回答。

但监督微调有一个明显限制：这些提示词和回答由真人编写，需要支付报酬，也很耗时，因此监督微调数据集通常比预训练数据集小得多。数据量小就容易导致模型过拟合：模型可能对训练集中出现过的特定提示词回答得很好，但对训练集之外的新提示词泛化能力较差。

![对比未对齐预训练模型与监督微调后模型的结构变化图](/coursenotes/topics/rlhf-explained/kf_0008_00-06-00.jpg)

![预训练加监督微调后模型仅能回应特定训练过的提示](/coursenotes/topics/rlhf-explained/kf_0009_00-06-55.jpg)

## 为什么需要 RLHF：用偏好数据替代昂贵的监督数据 [07:36 - 08:13]

解决过拟合的理想方法是使用一个超大的监督微调数据集，但创建这样的数据集需要花费大量资金。因此，视频引入 RLHF：它的核心目标是构造一个比监督微调数据集大得多的反馈数据集，同时尽量减少获取人类偏好的成本。

换句话说，如果让人直接写出大量高质量回答非常便宜，那么可能就会直接扩大监督微调数据集；但现实中人工编写答案太贵，所以改用“让人比较哪个回答更好”的方式获取反馈。这个偏好比较过程，就是 RLHF 中“人类反馈”的部分。

![强调RLHF需要比监督微调更大的数据集](/coursenotes/topics/rlhf-explained/kf_0010_00-07-49.jpg)

## 生成候选回答并收集人类偏好 [08:13 - 09:51]

给定一个提示词，模型可以有两种生成方式。第一种是每一步都选择概率最高的 token，这样每次都会得到同一个固定回答，例如“StatQuest is a good channel to learn AI.”。第二种是根据 SoftMax 输出的概率来采样 token：概率最高的 token 更常被选中，但其他 token 仍有机会被选中，因此可以得到多个不同回答，例如：

- “StatQuest is a great channel for learning AI.”
- “StatQuest is an awesome channel for learning AI.”

有了多个候选回答后，可以把它们两两配对，让人判断每一对中哪一个更好。让人列出偏好比让人从头写出最佳回答快得多，因此可以更高效地获得大量人类反馈数据。这些偏好数据正是 RLHF 的“human feedback”。

![展示基于SoftMax概率生成特定回复的完整流程](/coursenotes/topics/rlhf-explained/kf_0011_00-08-43.jpg)

![三组回答对的偏好标注示意图（Better/Worse）](/coursenotes/topics/rlhf-explained/kf_0012_00-09-37.jpg)

## 训练奖励模型：让模型学会给回答打分 [09:51 - 11:45]

有了偏好数据后，下一步是训练一个模型，让它给更受偏好的回答打较高分。具体做法是：先复制已经过监督微调的模型，然后去掉它的 Unembedding 层，换成一个单独的输出，得到一个用于计算奖励的新模型。

训练时，把提示词和回答一起输入这个新模型。对于人类更偏好的回答，训练模型输出正奖励；对于未被偏好的回答，训练模型输出负奖励。提示词的上下文是必要的，因为不同提示词需要不同的回答才能做到礼貌和有用。

这个过程中不需要预先定义“理想奖励值”。模型可以自己学出合适的奖励。视频提到 OpenAI 2022 年论文《Training Language Models to Follow Instructions with Human Feedback》中使用的损失函数，就是让模型自己学习最佳奖励，而不需要人工提前规定奖励值。

![展示提示词与好坏回复对，以及奖励模型如何为较差回复生成负奖励的示意图](/coursenotes/topics/rlhf-explained/kf_0013_00-10-49.jpg)

## 奖励模型的损失函数：从偏好中学习奖励 [11:45 - 14:55]

奖励模型的损失函数看起来复杂，但核心是几项简单内容相乘。先关注奖励差值：

$$R_{\text{better}} - R_{\text{worse}}$$

其中 $R_{\text{better}}$ 是模型给更好回答计算的奖励，$R_{\text{worse}}$ 是给较差回答计算的奖励。理想情况下，希望 $R_{\text{better}}$ 是正数，$R_{\text{worse}}$ 是负数，因此两者差值是一个较大的正数。

这个差值会输入 Sigmoid 函数。Sigmoid 把输入映射到 0 到 1 之间：如果输入是较大的正数，例如 10，输出会非常接近 1（视频给出 `Sigmoid(10) = 0.99995`）；如果输入是较大的负数，例如 -10，输出会非常接近 0。

Sigmoid 的输出再输入 log 函数。理想情况下，差值较大且为正，Sigmoid 输出接近 1，log 输出也会相对较高。但梯度下降通常寻找曲线的最低点，而这里希望收敛到较高的位置，所以把整个表达式乘以 -1，将曲线翻转。最终损失函数为：

$$-1 \times \log\left(\mathrm{Sigmoid}(R_{\text{better}} - R_{\text{worse}})\right)$$

这样，不需要显式告诉模型“更好回答应为正奖励、更差回答应为负奖励”，损失函数本身就会推动模型学会正确计算奖励。训练完成后，这个新模型被称为**奖励模型**（reward model）。

![展示奖励差值经过Sigmoid函数映射到接近1的过程及对应曲线图](/coursenotes/topics/rlhf-explained/kf_0015_00-12-55.jpg)

![RLHF奖励损失函数公式及log-sigmoid曲线图，展示Reward_Better与Reward_Worse差值对损失的影响](/coursenotes/topics/rlhf-explained/kf_0016_00-13-49.jpg)

![RLHF 奖励模型损失函数与偏好数据示例图](/coursenotes/topics/rlhf-explained/kf_0017_00-14-43.jpg)

## 用奖励模型进行强化学习训练 [14:55 - 16:37]

奖励模型训练好后，可以用它来训练最初那个只做过监督微调的模型。训练时使用新的提示词，这些提示词不在监督微调数据集中，也没有人工提供的标准回答。例如提示词是“What's another good YouTube channel for learning about AI?”。

原模型可能先生成一个不太有帮助的回答，例如“YouTube is special.”。把提示词和这个生成回答一起输入奖励模型，奖励模型会给出负奖励（视频示例为 -2）。然后利用奖励模型给出的奖励信号训练原模型。

经过强化学习训练后，面对同一个提示词，原模型可以生成更礼貌、更有帮助的回答，例如“Serrano Academy is a great YouTube channel for AI!”。这个回答会得到奖励模型给出的较大正奖励（视频示例为 10）。也就是说，奖励模型像人一样给回答打分，再用这些分数训练原模型，使其在没有巨大监督微调数据集的情况下，也能对新提示词生成合适回答。

![基于人类反馈的强化学习（RLHF）架构图，展示模型生成回复、奖励模型打分及反馈流程](/coursenotes/topics/rlhf-explained/kf_0018_00-15-37.jpg)

![基于人类反馈的强化学习（RLHF）架构图，展示监督微调模型生成回复、奖励模型计算奖励的流程](/coursenotes/topics/rlhf-explained/kf_0019_00-16-31.jpg)

## 最终结果：训练并对齐的模型 [16:37 - 17:12]

完成预训练、监督微调和 RLHF 后，最终得到一个既经过训练、又对齐到人们实际使用方式的模型。此时再输入“What is StatQuest?”，模型不再输出“blah blah blah”，而是生成类似“StatQuest with Josh Starmer is an awesome YouTube channel for learning about statistics, machine learning, and AI.”的礼貌、有帮助回答。

![介绍StatQuest频道的幻灯片，左侧配有已训练并对齐的模型结构图](/coursenotes/topics/rlhf-explained/kf_0020_00-17-07.jpg)

## 要点回顾

- RLHF 通常用于训练大语言模型，位于预训练和监督微调之后。
- 预训练让模型在海量文本上学习预测下一个 token，得到预训练模型。
- 预训练模型擅长续写文本，但不一定擅长回答用户问题，因此需要对齐。
- 监督微调用人工编写的提示词—回答对训练模型，但数据小、成本高，容易过拟合，难以泛化到新提示词。
- RLHF 的核心是用人类偏好比较代替人工编写大量答案，从而以较低成本获得更大的反馈数据集。
- 模型可以按 SoftMax 概率采样生成多个候选回答；人类只需两两比较并标出哪个更好。
- 奖励模型通过“更好回答给正奖励、较差回答给负奖励”来学习人类偏好。
- 奖励模型损失函数为 $-1 \times \log(\mathrm{Sigmoid}(R_{\text{better}} - R_{\text{worse}}))$，它推动更好回答的奖励高于更差回答。
- 训练好的奖励模型可以给原模型的新回答打分，并用奖励信号继续训练原模型。
- 最终模型在预训练、监督微调和 RLHF 后，能够对新提示词生成更礼貌、更有帮助的回答。
