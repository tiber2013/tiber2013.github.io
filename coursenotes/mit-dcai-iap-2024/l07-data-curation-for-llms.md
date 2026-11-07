---
title: "第 7 讲 · 面向大语言模型的数据策展"
description: "Data Curation for LLMs · MIT DCAI IAP 2024 中文图文笔记"
---

# 第 7 讲｜面向大语言模型的数据策展

> **MIT Introduction to Data-Centric AI · IAP 2024 · L07** ｜ 48 分钟 ｜ [原视频](https://www.youtube.com/watch?v=Zo8XRX88R8I)

## 核心结论

本讲为课程最后一讲，围绕「为 LLM 策展数据」与「用 LLM 策展数据」展开。核心论点是模型行为由数据而非架构、超参数决定，数据质量是全流程杠杆。LLM 是强推理引擎，能以零/少样本、极低成本替代手写正则与规则列表完成 PII 检测、语法检查；但输出有幻觉，需更强模型评估或不确定性量化把关。微调侧推荐「清洗种子 → 生成合成数据 → 高置信度过滤 → 分类器剔除 → 全量清洗 → 微调」配方，用小模型逼近大模型。

## 一、背景：什么是大语言模型 [00:00 - 02:25]

语言模型（Language Model）是序列到序列（seq2seq）模型，目标是根据前缀预测下一个词元（token），即建模 $P(x_i \mid x_{i-1}, \ldots, x_0)$。大语言模型（Large Language Model, LLM）的「大」指容量大、在互联网规模语料上做无监督预训练（unsupervised pre-training）；高容量加海量数据会涌现小模型不具备的能力，近年核心创新即「把规模做大」（scale up）。

![LLM背景介绍：seq2seq下一词预测模型及训练特点](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0004_00-00-24.jpg)

图注：对应 LLM 定义——预测下一词元的 seq2seq 模型，及「大」代表的高容量与大规模预训练。

以此为基础，少量工作即可支撑各类自然语言处理（Natural Language Processing, NLP）任务。最常见的是零样本提示（zero-shot prompting）：把任务写进提示词即可，无需收集数据训练模型；讲者要求「保留原意、移除性别指代」，模型把 "she / her" 改成 "they / their"，全程无一示例。

![零样本提示示例：让ChatGPT去除句子中的性别指代](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0013_00-01-54.jpg)

图注：对应零样本提示应用——仅用一句描述、不给样例，就让模型改写性别指代。

## 二、四类应用范式与成本光谱 [02:25 - 10:29]

四类用法按努力与成本递增：

- 零样本提示：只写自然语言描述，几乎零成本。
- 少样本提示（few-shot prompting）／上下文学习（in-context learning）：给若干输入输出对，让模型归纳模式。
- 检索增强生成（Retrieval-Augmented Generation, RAG）：先检索文档库相关片段，连同查询生成答案，补足公开语料没有的知识（如公司内十万份私有文档）。
- 微调（fine-tuning）：从已有网络出发更新权重，任务明确且有样例时效果最好。

讲者用文件重命名演示少样本：提示不明说任务，只给几个「文件名 → JSON 指令」例子（带 `1080p` 的去后缀重命名、带 `720p` 的删除），模型便能处理未见输入；但有限样例能拟合的映射不止一种，属欠定（underspecified）问题，过程也不透明。

![少样本提示示例：根据文件名模式返回JSON操作指令](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0019_00-02-54.jpg)

图注：对应少样本案例——用文件名与 JSON 指令对让模型归纳重命名/删除模式。

![LLM提示/训练技术对比图：任务准确率随努力/复杂度变化](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0040_00-06-18.jpg)

图注：对应四类提示/训练技术对比曲线——准确率随努力与复杂度上升。

成本上，少样本只需几个数据点，微调要数百至数千样例、一块 GPU 与数小时，仍在数百至数千美元量级，远低于动辄数百万美元的从头预训练（pre-training from scratch）；故本课重心在应用侧。

![课程大纲幻灯片，列出LLM数据整理、评估及训练数据整理的三个主要议题](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0059_00-09-18.jpg)

图注：对应本讲议程——大纲列出用 LLM 策展数据、评估输出、训练数据策展等议题。

大纲由此展开：用 LLM 策展数据、评估 LLM 输出（也是一类数据）、为训练 LLM 策展数据（含预训练基座与面向应用），三者还能组合。

## 三、用 LLM 做数据策展：PII 检测与语法检查 [10:29 - 22:47]

![LLM用于数据整理的优势介绍，包括推理能力强、易定制及成本低](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0066_00-10-30.jpg)

图注：对应核心判断——LLM 强推理、易定制、低成本，正革新系统化文本数据策展。

大语言模型是强大的自然语言推理引擎（powerful natural-language reasoning engines），新任务几分钟即可上手，成本很低。下面用两例说明它如何替代传统做法。

### 3.1 例一：个人身份信息（PII）检测

![PII检测背景介绍，提及隐私泄露风险及旧有的正则表达式方法](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0070_00-11-06.jpg)

图注：对应 PII 检测背景——模型会记忆并泄露训练数据，须从训练集清除。

个人身份信息（Personally Identifiable Information, PII）指姓名、位置、信用卡号等可识别个人的内容。清除它是因模型会记住训练数据、可能被反推抽取，公开即成隐私风险。清洗并非「检测即丢弃」：一段文本（如两人对话）可能提到多人，全打码会丢失「说话人一/二」的关联；更稳妥的是确定性一致替换（deterministic replacement），把第一个名字换成 "speaker one"、第二个换成 "speaker two" 并全篇一致。

![使用正则表达式进行PII检测的代码示例，包含信用卡和Twitter用户名检测类](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0086_00-13-42.jpg)

图注：对应传统 PII 检测——逐类手写正则，画面为 scrubadub 的正则代码。

旧法先穷举 PII 类型、逐类手写正则表达式（regex），必须显式枚举且不能遗漏，逐条调试耗时数小时。

![零样本提示（Zero-shot prompting）用于PII检测的Prompt模板示例](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0092_00-14-42.jpg)

图注：对应零样本提示做 PII 检测——把「判断评论是否含 PII、解释推理、以 Output 收尾」写成模板。

新法用零样本提示：把文本注入模板，要求识别是否含 PII（含未列举类型）、先解释推理、以 `Output: true` 或 `Output: false` 收尾；逐条套用、解析即可，一秒完成。

![零样本提示检测PII的对话示例，展示含Instagram账号文本的判断结果](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0104_00-16-36.jpg)

图注：对应具体案例——模型把含 Instagram 账号的文本判为 true，抓到未列举的 PII 类型。

关键优势是模型已理解「什么是 PII」，能抓到未显式列举的类型，如社交账号句柄不符合姓名正则、传统方法会漏掉。工程上可批量塞多个数据点摊薄开销；但数据含 PII 时不宜发外部 API，可自建开源模型。

### 3.2 例二：语法检查

![传统NLP语法检查方法中手工设计规则列表的XML代码片段](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0128_00-20-30.jpg)

图注：对应传统语法检查——经典 NLP 加手工规则列表，画面为 LanguageTool 的 XML 规则片段。

旧法是经典 NLP 加手工规则列表（hand-designed rule lists），难点不在记下病句，而在把「什么算语法错误」抽象成通用规则，工作量极大。

![使用微调进行语法检查的方法介绍，包含CoLA数据集和HuggingFace代码示例](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0133_00-21-18.jpg)

图注：对应微调做语法检查——收集「句子 + 布尔判断」（可复用 CoLA），把基座模型微调成分类器。

新法改为微调：收集「句子 + 是否合乎语法」的布尔判断（判断通顺远比写规则容易），可复用语言可接受性语料（Corpus of Linguistic Acceptability, CoLA）；在基座 LLM 上微调，任务从 seq2seq 生成变成二分类器（classifier）。借 Hugging Face Transformers，几行代码、一块 GPU、约一小时即可完成。

## 四、评估 LLM 输出：幻觉、强模型裁判与数据过滤 [22:47 - 31:02]

### 4.1 幻觉

![评估LLM输出数据的幻灯片，对比简单加法和复杂乘法的计算结果](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0146_00-23-24.jpg)

图注：对应幻觉（hallucinations）这一首要挑战——对比简单加法与复杂乘法的错误输出。

首要挑战是幻觉：输出看似合理却可能错误，且无内建机制判别。讲者对照：`38 + 72` 答 110 正确，而 `38346 × 1337` 答 51,367,502（正确为 51,268,602），算错却写得一本正经；若用于策展便会误标 PII、丢错数据。

### 4.2 用更强 LLM 评估

![使用更强LLM（GPT-4）评估较弱模型（GPT-3.5）的PII识别任务对比示例](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0149_00-23-54.jpg)

图注：对应「强模型当裁判」——把查询、另一模型输出与评分标准交给 GPT-4 打分。

第一种技术是让更强模型当裁判：把 query、output 与评分标准交给强模型，先推理、再以「Score 加评分」收尾。PII 例中，不带思维链（Chain-of-Thought）的旧提示让 GPT-3.5 把含 Instagram 账号的评论误判 False，GPT-4 则指出账号属于 PII、判其错误并打 1/5 低分。此法常见于难以自动打分的生成任务。边界是：若每个数据点都要过强模型，不如直接用它做任务；强模型评估自己的输出也不可靠。

### 4.3 有效性案例与局限

![AlpaGasus数据集的有效性：通过高质量数据微调超越原始Alpaca模型](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0167_00-26-48.jpg)

图注：对应数据过滤的有效性——AlpaGasus（Chen et al. 2023）用高质量数据微调后超越 Alpaca。

AlpaGasus（Chen et al. 2023）是有力印证：斯坦福 Alpaca 用 52k 条合成指令数据在 Meta 的 LLaMA 上微调；作者用 GPT-3.5 打分、只留最好的 9k 条重新微调，结果显著优于原模型，评估用了 GPT-4 与人类研究——数据质量胜过数量。

![LLM评估面临的挑战：非防篡改性及递归评估问题（Turtles all the way down）](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0179_00-28-42.jpg)

图注：对应「用 LLM 评估 LLM」的局限——并非万无一失，且有「谁来评估评估者」的递归困境。

但讲义概括为 "Not foolproof" 与 "Turtles all the way down?"：若输出本就是 GPT-4 生成，谁来评它？强模型也非万无一失——把那道错误乘法答案连同评分提示交给 GPT-4，它自信打出 5/5、称「完全正确」，而答案其实错了。这自然引出下一章。

## 五、LLM 不确定性量化 [31:02 - 37:02]

没有更强模型时，可对同一 LLM 多次提问：模型以非确定性（non-deterministic）方式运行、每次输出略不同，把多个答案当作「多标注者」；因标注者其实是同一模型，可大幅简化。课程方法来自 Chen & Mueller 2023，聚焦问答，置信分数由两部分构成：

1. **观察一致性（observed consistency）**：调节 temperature 或微调提示词采样多个答案，用自然语言推理（Natural Language Inference, NLI）模型判断「采样答案与原答案是否语义矛盾」并取平均，比较的是语义冲突而非字符串相同。
2. **自我反思确定性（self-reflection certainty）**：把原提示与此前回答回喂 LLM，让它解释推理并判断 correct／incorrect／not sure。

![基于Chen & Mueller 2023的LLM不确定性量化流程图（观察一致性与自我反思确定性）](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0201_00-32-18.jpg)

图注：对应不确定性量化流程——分「观察一致性」与「自我反思确定性」两路，前者借 NLI 判矛盾。

讲者强调，与前面的置信学习（confident learning）不同，这类方法无形式化证明，偏经验主义：先提出直觉合理的想法，再在大量样例上严谨做基准验证，结果显示正确时置信度高、错误时低。

![Cleanlab TLM（Trustworthy Language Model）界面演示，展示问答及置信度分数](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0225_00-36-12.jpg)

图注：对应具体实现——Cleanlab 的 Trustworthy Language Model（TLM）界面，回答后附置信度。

TLM 中问「《Attention is All You Need》第一作者」得正确答案 "Vaswani, Ashish"、置信度 91.44%；问「《Private web search with Tiptoe》用什么语言实现搜索引擎」，模型答「论文未明确提及」（讲者指出是错的）、置信度 64.07%。错误答案对应更低置信度，可据此决定采信、复核或过滤。

## 六、预训练基座模型的数据策展 [37:02 - 41:56]

![引用jbekter关于预训练数据重要性的观点：模型行为由数据集决定而非架构](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0230_00-37-00.jpg)

图注：对应「数据集即模型」——引用 OpenAI 研究者博文，强调行为由数据集而非架构决定。

讲者引用 OpenAI 研究者 jbetker 的博文《The "it" in AI models is the dataset》：同一数据集上训练足够久，不同架构会收敛到同一点；故模型行为不由架构、超参数或优化器决定，而由数据集决定，其余只是把算力高效用于逼近数据集的手段，与 data-centric AI 主旨一致。

![LLM数据整理与训练技术概览，涵盖无监督预训练、有监督微调及基于人类反馈的强化学习](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0238_00-38-18.jpg)

图注：对应预训练数据整理与训练技术概览——无监督预训练、有监督微调、RLHF，强调语料质量。

讲义列三类技术：无监督预训练（unsupervised pre-training）先收集海量文本预测下一词元，语料质量至关重要（quality of corpus matters）——「让 LLM 忘掉已学的东西非常难」，混入看似事实实则错误的内容日后难纠正；有监督微调（supervised fine-tuning, SFT）由人类写金标准（gold）输入—输出对；基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）为避免人工评分海量输出太贵，只让人类对少量点给反馈，再训练奖励模型（reward model）模仿打分作为奖励信号。课堂补充：SFT 样本可自动化扩充，可蒸馏（distill），用大模型生成样例训练更小模型，讲义以 Textbooks Are All You Need（Li et al. 2023）为例。

## 七、面向应用的数据策展与合成数据配方 [41:56 - 46:38]

![面向LLM应用的数据整理策略列表，包括零样本、少样本提示工程及检索增强生成](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0262_00-42-06.jpg)

图注：对应面向应用的数据策展概览——零样本、少样本、RAG、SFT 各自对数据的依赖。

基座训练是大题目，本课更关注「拿到基座模型之后」：数据几乎在所有应用里都起作用。零样本只有提示工程，相关性最小；少样本中示例质量对性能与稳定性至关重要（见 Chang & Jia 2023）；RAG 的文档库质量决定生成质量；SFT 中输入—输出对越好结果越好。

![斯坦福Alpaca合成数据生成流程，展示了利用提示词工程生成多样化任务指令的具体要求](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0275_00-44-12.jpg)

图注：对应合成数据生成（synthetic data generation）——Alpaca 提示词要求生成「20 条多样任务指令」。

微调对特定任务给出最好结果，常见范式是从小规模人工策展集出发、用 LLM 生成更多数据，动机是微调更小更便宜的 LLM 来匹配大模型——小模型推理时延与费用差距巨大。流程先清洗种子数据（套用前几讲通用原则），再用强 LLM 生成合成数据，并借不确定性量化只留高置信度结果。生成靠提示，例如 Alpaca 的真实提示要求「想出 20 条多样任务指令」，约束动词不重复、语言与任务类型多样、给出合适 input/output。

![合成数据清洗与最终微调流程，包括使用分类器过滤低质量数据及全量数据集训练](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0283_00-45-30.jpg)

图注：对应合成数据清洗与最终微调——用「真实 vs 合成」二分类器剔除不真实样本，全量清洗后微调。

之后把关：混合原始与合成数据，分别针对 input、output 训练「真实 vs 合成」二分类器，按分数剔除不真实（unrealistic）样本；再对「原始 + 高质量合成」全量应用通用清洗；最后用完整数据集微调得到高质量模型。这套配方把前几讲的清洗方法与本讲的生成／评估技术串成闭环，是课程推荐流程。

## 八、未来展望：基础模型与多模态 [46:38 - 48:21]

![展示基础模型时代数据策展未来的幻灯片，对比了使用cleanvision代码与GPT-4进行图像亮度评估的演变](/coursenotes/mit-dcai-iap-2024/l07-data-curation-for-llms/cand_0290_00-46-42.jpg)

图注：对应基础模型时代的未来——对比过去 cleanvision 的亮度代码与如今让 GPT-4 零样本给图像暗度打分。

基础模型（foundation models）常与 LLM 互换，指以无监督方式在大规模数据上训练的大模型；如今已有能理解图像的**多模态 LLM**（multimodal LLM），如 GPT-4。过去要发现图像数据问题（过暗、过亮、模糊），得写 Python 代码实现经典视觉算法，如 cleanvision 按加权 RGB 算亮度：

$$\text{brightness} = \frac{\sqrt{0.241\,r^2 + 0.691\,g^2 + 0.068\,b^2}}{255}$$

而现在直接把图像交给 GPT-4、让它按 0 到 1 给「暗度」打分即可，示例得约 0.995 的合理结果。讲者据此判断：基础模型未来会接手大量这类策展任务，把曾需专门写代码的检测变成零样本可解的问题。

## 要点回顾

- 语言模型建模下一词元条件概率 $P(x_i \mid x_{i-1}, \ldots, x_0)$；「大」来自高容量加互联网规模预训练，核心创新是 scale up。
- 零样本、少样本、RAG、微调按成本递增；微调数百至数千美元，从头预训练才到百万美元量级。
- 模型行为由数据集决定；语料错误内容很难让模型「忘掉」，质量胜过数量。
- LLM 可低成本替代正则与规则列表做 PII 检测、语法检查；PII 清洗宜用确定性一致替换。
- 幻觉是首要风险；可用更强模型打分，但有自评困境且非万无一失（GPT-4 曾给错答案打 5/5）。
- AlpaGasus 把 52k 合成数据筛成 9k，微调后显著优于原模型。
- 不确定性量化 = 观察一致性（NLI 判多次采样矛盾）+ 自我反思确定性；TLM 中错答置信度更低（91.44% 对 64.07%）。
- 微调推荐配方：清洗 → 生成留高置信度 → 分类器剔除 → 全量清洗 → 微调，用小模型逼近大模型。

## 视频来源

https://www.youtube.com/watch?v=Zo8XRX88R8I
