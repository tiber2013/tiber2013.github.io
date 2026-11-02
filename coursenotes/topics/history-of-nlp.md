---
title: "NLP 简史：规则、统计、机器学习与 Transformer"
description: "自然语言处理（NLP）是计算机科学中研究「计算机如何理解、解释和生成人类语言数据」的分支，自 1950 年代诞生以来经历了四个清晰的阶段：基于规则的系统、统计 NLP、机器学习驱动的 NLP、神经网络与 Transfor…"
---

# NLP 简史：规则、统计、机器学习与 Transformer

> **专题视频笔记** ｜ 讲者 365 Data Science ｜ 8 分钟 ｜ [原视频](https://youtu.be/Uuz8ZTV5vdA)

> 视频：The History of Natural Language Processing (NLP)（365 Data Science）
> 时长：7 分 38 秒 ｜ 来源：youtu.be/Uuz8ZTV5vdA

## 核心结论

自然语言处理（NLP）是计算机科学中研究「计算机如何理解、解释和生成人类语言数据」的分支，自 1950 年代诞生以来经历了四个清晰的阶段：基于规则的系统、统计 NLP、机器学习驱动的 NLP、神经网络与 Transformer。贯穿后半段的关键使能技术是**向量嵌入**——它把词语转换成高维空间中的数值向量，让模型可以基于语义相似度来存储和检索数据。2018 年出现的 **Transformer 架构**直接催生了 GPT、Gemini 等大语言模型，把 NLP 推向了今天的面貌。理解这条「规则 → 统计 → 机器学习 → 神经网络」的演进路线，就理解了现代 LLM 的能力从何而来。

## NLP 是什么 [00:32 - 00:46]

Natural Language Processing（更常简称为 NLP）是计算机科学的一个领域，研究计算机如何理解、解释和生成人类语言数据。视频将其核心任务概括为三件事：

- **理解**（understand）人类语言；
- **解释**（interpret）人类语言；
- **生成**（generate）人类语言。

![自然语言处理（NLP）的定义幻灯片](/coursenotes/topics/history-of-nlp/kf_0003_00-00-38.jpg)

## NLP 技术演进路线总览 [00:10 - 00:23]

视频开篇给出的学习路线，把 NLP 技术的发展分成四个阶段：

1. 基于规则的系统（Rule-based systems）
2. 统计 NLP（Statistical NLP）
3. 机器学习驱动的 NLP（Machine Learning-led NLP）
4. 神经网络与 Transformer（Neural Networks and Transformers）

在此基础上，视频还会涉及两个关键概念：**向量嵌入**（vector embeddings），以及**大语言模型**（LLMs）等最新创新带来的影响。

![自然语言处理（NLP）主题标题页](/coursenotes/topics/history-of-nlp/kf_0001_00-00-08.jpg)

![NLP 课程大纲幻灯片，列出 NLP 定义与技术演进路线](/coursenotes/topics/history-of-nlp/kf_0002_00-00-23.jpg)

## 阶段一（1950 年代）：基于规则的系统 [00:46 - 01:21]

NLP 诞生于 1950 年代前后。最早的方法是基于规则的系统：用语言语法规则来尝试处理文本。设计这类系统时，可以先建立这样一条规则：

> 把以 "Can you"、"Will you" 或 "Is it" 开头的句子视为问句。

于是，当系统处理 "Can you help me?" 这个句子时，就会把它识别为一个问题。这条规则听起来相当基础、相当手工——而这正是第一代 NLP 的真实写照：规则由人逐条编写，覆盖面有限。

![1950年代自然语言处理（NLP）概念引入的动画示意图](/coursenotes/topics/history-of-nlp/kf_0004_00-00-54.jpg)

![1950年代时间轴节点，展示机器人将“Can you help me?”识别为问题的图示](/coursenotes/topics/history-of-nlp/kf_0005_00-01-16.jpg)

## 阶段二（1990 年代）：统计 NLP 与词性判别 [01:21 - 02:57]

1980 年代末、尤其是 1990 年代初，统计 NLP 出现，标志着从基于规则的系统转向**概率方法**：通过对大量语言数据的分析来解释语言数据。

视频用一个经典任务来演示这套思路——判断句子里的 "can" 是名词还是动词。这个区分非常重要：

- 作动词时，can 描述一个人有能力做或完成某事；
- 作名词时，can 指容器，比如一罐可口可乐（a Coca-Cola can）。

1990 年代的统计学家会这样做：

1. **收集语料**：收集大量含有 can 的句子；
2. **标注**：逐句标注 can 在该句中用作名词还是动词；
3. **上下文分析**：分析 can 周围出现了哪些词。例如，分析可能揭示：当 "you"、"I" 这类词靠近 can 时，通常表明它是动词；如果句中使用了 "soda" 这类词，那大概率是名词；
4. **得到概率模型**：最终目标是获得一个模型，能根据前面统计出的频率，给出 can 在给定句子中是动词或名词的概率。

视频也点破：这种从数据里找规律的做法，已经很像「原始形态的机器学习」（primitive machine learning）。

![1990年代统计NLP示意图，展示从基于规则系统向概率方法的转变](/coursenotes/topics/history-of-nlp/kf_0006_00-01-32.jpg)

![词性标注演变图，展示CAN连接名词与动词并配有肌肉图标](/coursenotes/topics/history-of-nlp/kf_0007_00-01-55.jpg)

![包含六个例句的幻灯片，通过高亮不同颜色的'can'展示其在不同语境下的词性差异](/coursenotes/topics/history-of-nlp/kf_0008_00-02-10.jpg)

![1990年代'can'作为情态动词的三种典型用法示例](/coursenotes/topics/history-of-nlp/kf_0009_00-02-25.jpg)

![1990年代'can'作为名词的用法示例：一罐汽水](/coursenotes/topics/history-of-nlp/kf_0010_00-02-41.jpg)

## 延伸学习：365 Data Science 课程推荐 [03:03 - 03:47]

在回到正题之前，视频用约 45 秒介绍了 365 Data Science 学习平台，推荐了三门课程：

- **Introduction to AI**：人工智能的全面概览；
- **Machine Learning in Python**：上手流行机器学习库的实操课程；
- **NLP for AI**：教你如何用 Python 和机器学习搭建有效的 NLP 系统。

介绍 Introduction to AI 课程时，画面给出一张 AI 入门示意图：一个数字 3 的 $28 \times 28$ 像素图像，对应 $28 \times 28 = 784$ 个输入节点，即神经网络的输入层宽度 $X_1, X_2, \dots, X_{784}$。

![AI入门示意图，展示数字3的28x28像素图像及其对应的784个输入节点向量](/coursenotes/topics/history-of-nlp/kf_0011_00-03-19.jpg)

![365 Data Science 课程库网页界面截图](/coursenotes/topics/history-of-nlp/kf_0012_00-03-42.jpg)

## 阶段三（2000 年代起）：机器学习驱动的 NLP 与向量嵌入 [03:47 - 05:57]

进入 2000 年代，统计分析进一步走向机器学习驱动的 NLP，NLP 任务的表现获得显著提升。**向量嵌入**（vector embeddings）是这一转变的关键使能者：它把词语和句子转换成数值数组，有效地捕捉文本数据中的意义与关系。

这是怎么发生的？向量的优点在于可以表示更复杂类型的信息——文本、图像、音频以及其他非结构化数据。句子中的每个词都可以表示为一个向量，存放在一个名为「向量嵌入」的高维空间里。

视频用一个三维坐标系做直观演示：在由 Sky、Engine、Wings 三个维度构成的空间中，Goose 位于 $(2, 0, 4)$、Eagle 位于 $(3, 0, 3)$、Bee 位于 $(3, 0, 2)$、Helicopter 位于 $(0, 2, 4)$、Drone 位于 $(0, 3, 3)$、Rocket 位于 $(0, 4, 2)$、Jet 位于 $(1, 1, 1)$——语义相近的概念在空间中彼此靠近。

![向量嵌入（Vector Embeddings）的三维坐标示意图，展示不同物体在语义空间中的位置](/coursenotes/topics/history-of-nlp/kf_0013_00-04-05.jpg)

从单词到向量的生成流程也被画了出来：I、LOVE、BASKETBALL 这些离散的词汇符号，各自映射为一个向量，成为高维空间中的点。

![词向量生成流程，左侧单词映射至中间向量，右侧展示高维空间散点分布](/coursenotes/topics/history-of-nlp/kf_0014_00-04-35.jpg)

需要强调的是，实际 AI 模型中的向量嵌入通常是**数百到数千维**，而不是示意图里的一两三维。视频画出 1 维、2 维、3 维，只是为了类比「同时嵌入数千个维度」是什么概念。毕竟，像人类语言这么复杂的意义，不可能用一两个、两三个维度捕捉下来，我们需要数百甚至上千个维度。

![AI模型同时嵌入数千维度的示意图，对比1、2、3维空间](/coursenotes/topics/history-of-nlp/kf_0015_00-04-58.jpg)

这样做的核心思想是：**用嵌入来基于语义相似度存储和检索数据**。视频用一个「国家—首都」的三维空间图来演示：Canada、Turkey、Russia、Vietnam、Japan、China、Spain、Italy、Germany 这些国家节点，与 Ottawa、Ankara、Moscow、Hanoi、Tokyo、Beijing、Madrid、Rome、Berlin 这些首都节点，按语义关系组织在同一空间里，直观展示了语义空间中的层级结构。

![展示语言复杂性的三维坐标图，通过城市节点和连线演示基于语义相似性的数据存储与检索](/coursenotes/topics/history-of-nlp/kf_0016_00-05-21.jpg)

与此同时，强大的新机器学习模型出现，使计算机能够把海量文本数据库作为向量嵌入来分析。这些模型学会了识别微妙的语言线索——比如讽刺（sarcasm）或反语（irony）所传达的情绪——而这些细微差别是基础统计方法此前难以捕捉的。

## 阶段四（2010 年代至今）：神经网络与 Transformer [05:57 - 06:41]

2010 年代，神经网络的出现革新了翻译、语音识别、文本生成等 NLP 任务。原因是神经网络凭借深层多级结构，擅长检测传统机器学习方法够不着的复杂模式：从原始输入（raw image）开始，经过若干中间层（intermediate layers）逐步识别复杂特征，最终在输出端得出结论（final conclusion）。

![展示从原始图像输入到最终结论输出的多层神经网络结构示意图](/coursenotes/topics/history-of-nlp/kf_0017_00-06-07.jpg)

2018 年推出的 **Transformer 架构**是其中最新的创新之一，它彻底改变了 NLP 领域，并直接导致大语言模型的诞生。架构图展示了编码器与解码器的堆叠结构，图中详细标注了 Multi-Head Attention、Feed Forward、Add & Norm 等核心组件。正是这一架构，使 GPT、Gemini 等 LLM 的开发成为可能。

![Transformer 模型架构图，展示编码器与解码器的堆叠结构](/coursenotes/topics/history-of-nlp/kf_0018_00-06-22.jpg)

![Transformer 架构演进至 Gemini 和 LLMs 的示意图](/coursenotes/topics/history-of-nlp/kf_0019_00-06-38.jpg)

## 要点回顾 [06:51 - 07:26]

- NLP 是人工智能的一个领域，让计算机能够理解、解释和生成人类语言。
- 技术演进走过四个阶段：基于规则的系统 → 统计方法 → 机器学习驱动的 NLP → 神经网络与 Transformer。
- 向量嵌入在为 AI 模型表示文本数据方面起着至关重要的作用。
- 近年来，Transformer 架构等创新催生了 GPT、Gemini 等强大的大语言模型。

视频最后强调：自然语言处理作为 AI 的语言能力，是一个快速演化的领域，持续推动着 AI 与机器学习可能性的边界。

![NLP 要点回顾续页，引入向量嵌入（Vector embeddings）概念及三维坐标示意图](/coursenotes/topics/history-of-nlp/kf_0020_00-07-16.jpg)
