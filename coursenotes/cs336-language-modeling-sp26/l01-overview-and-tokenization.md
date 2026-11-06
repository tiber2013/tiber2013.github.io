---
title: "第 1 讲 · 课程概览与分词"
description: "核心结论"
---

# 第 1 讲 · 课程概览与分词

> **Stanford CS336 语言模型从零构建 · L01** ｜ 讲者 Stanford Online ｜ 79 分钟 ｜ [原视频](https://www.youtube.com/watch?v=JuoVZkPBiKk)

> **v2 详写版** · 按 20 分钟分段详写，配图覆盖全片；文末附「对照参考笔记补遗（Spring 2025 版）」，记录与 2025 版讲义笔记的核对结果。

## 核心结论

Stanford CS336（Spring 2026）第三次开课，把 from scratch 与「通过构建获得理解」作为不变内核，用 5 学分、5 次高强度作业（basics、systems、scaling laws、data、alignment）串起从分词到对齐的全栈实现，并以「在给定算力与数据预算下做出最好模型」的效率意识贯穿始终。第一讲从 tokenizer 切入：它以可逆的 encode/decode 在字节／字符串与整数索引之间转换，BPE 把常见序列合并成 token、把长字节流压成更短的序列，从而省下 attention 的平方级计算，并把更多建模容量留给真正有趣的片段；往后依次展开以 $6ND$ 为核心的资源核算、围绕 HBM 数据搬运的算子融合与多卡并行、在 $C = 6ND$ 预算下设计可外推的缩放配方并借小实验预测大模型 loss，以及「评测先定义能力、数据整理与去重决定质量」的数据观。片尾的判断相当克制：直接操作原始字节的端到端方案虽被寄予厚望，却还没有扩展到前沿规模，因此分词只是现阶段的过渡手段；但即便将来抛弃它，替代方案仍必须在序列之上提供抽象、并允许可变长的自适应计算，否则就会是次优的。

## 课程定位、教学团队与 from scratch 哲学 [00:04 - 03:21]

本段是 Stanford CS336「Language Modeling from Scratch」（从零构建语言模型）Spring 2026 的第一次课。讲师 Percy Liang 开门见山地说明这是该课程的第三次开课（the 3rd offering of CS336），并强调这是一门要求真正动手、从底层理解语言模型工作原理的课程。屏幕上打出的课程全名是 `CS336: Language Models From Scratch (Spring 2026)`，教学团队一共五人：两位 Instructor（讲师）Percy Liang 与 Tatsunori Hashimoto，以及三位 CA（课程助教）Marcel Rød、Herman Brunborg、Steven Cao。值得注意的是，这并非一段普通的口述开场——整份讲义本身是一个可执行的 Python 程序（executable lecture），画面左侧还保留着源代码的行号，例如 `def welcome():` 出现在第 53 行，`def why_this_course_exists():` 出现在第 66 行，本次课的推进过程实际上就是在逐步执行这些函数。理解这一点很关键，因为后面所有幻灯片上的标题、黄色高亮行、甚至方括号里的文献引用都是程序渲染出来的结果，而不是一张静态 PPT，这一点在 19:13 的 `def main():` 画面里会得到印证。

自我介绍环节透露出团队分工与这门课的更新压力。Percy 说自己研究语言模型已有 20 年，但其中大部分时间做的是 small language models（小规模语言模型），并补了一句「即便以今天的尺度看，这仍然算小」；他还回顾说，两年前与 Tatsu 第一次开课时并不确定会不会有人来，结果惊喜地发现这么多学生想从零学构建，尤其是在 coding agent（编码智能体）大概已经能 zero-shot（零样本）写出一个语言模型的今天，仍有人想弄懂它究竟如何工作，这让他很高兴。Tatsu 作为联合讲师负责 architectures（架构）与 scaling（缩放）等部分，他半开玩笑地说 Percy 总调侃他每年都得重做讲义，因为架构领域每年都在变，但他本人觉得这种不断重来的过程「其实挺有趣」，也是他教课以来第一次有这样的体验。Marcel 去年上过一次课，因为「太好玩」又回来当助教，研究方向是架构、higher-order gradients（高阶梯度）与 training（训练），同时也直言这门课「工作量不小」。Herman 的自我介绍最能说明课程定位：一年前他还完全不懂 LLM 如何工作，对他而言 tokens 只是「电子游戏里收集的东西」、attention 只是「注意力经济」里的那个词，经过这门课后如今已从事 LLM 研究并回来当助教。Steven 是首次担任 CA，研究 language models、theory（理论）与 data efficiency（数据效率）。

关于「今年有什么变化」，幻灯片给出了三条非常明确的更新说明，这些内容转写里只被口语化地带过，但画面上逐字可辨：第一条是 `Same 'from scratch' philosophy`，即 from scratch（从零构建）的核心哲学保持不变；第二条是 `Prioritize high value-per-time concepts, don't lose the forest for the trees`，也就是按「单位时间的价值」来筛选讲授的概念，不要因为纠缠细节而丢了全局——这解释了为什么课程并不真的把一切都从零写起（用 Percy 的话说，「那样一个 quarter 装不下」），而是过去两年一直在打磨 recipe（配方），挑出最值得亲手实现、价值密度最高的部分；第三条被黄色高亮，是 `More coverage of modern LM ingredients (mixture of experts, long-context, agents)`，即今年会花更多篇幅讲 mixture of experts（MoE，混合专家）、long context（长上下文）与 agents（智能体），其中「为长上下文所需的能力」被特别点出是今年的重点。此外画面还写着 `Lectures from 2nd offering (Spring 2025) are on YouTube.`，说明第二次开课（2025 年春季）的录像已在 YouTube 上，这也是本次课被标注为「第三版」的由来。

![课程第三次开课说明：教学团队全名、from scratch 哲学与新增的混合专家、长上下文、智能体内容](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0003_00-03-05.jpg)

这份讲义的程序结构值得单独看一眼。画面上能直接读到的函数签名（原样保留）是：

```python
def welcome():
def why_this_course_exists():
```

画面上只显示函数签名，函数体并未展开；到了本段末尾（19:13 那张）还会看到它们被 `def main():` 依次调用，因此这里出现的不是装饰性文字，而是真正控制讲课流程的代码。把上一讲（Spring 2025）与本讲对照，容易踩的一个误区是以为「第三次开课」只是内容微调；实际上从 `More coverage of ... mixture of experts, long-context, agents` 可以看出，课程每年都在按领域变化重构，只是 from scratch 这条主线不动。另一个易错点是别把 `high value-per-time` 理解成「讲得快」——它强调的是价值密度与取舍，也不能把 CA 当成随意缩写而忽略其教学职责。这恰好为下一章「为什么非要从零构建」以及后面贯穿全课的效率主线埋下伏笔：既然一个学期的时间有限，就必须把力气花在最能迁移到大模型上的知识上，而「哪些知识能迁移」正是 07:09 之后要专门拆开讨论的问题。

## 为什么从零构建：抽象泄漏、工业化与前沿模型的不可及 [03:23 - 05:38]

`def why_this_course_exists():` 这一段回答的是开课动机。Percy 把问题概括成一句话：researchers are becoming **disconnected** from the underlying technology（研究者正在与底层技术脱节）。幻灯片用三个年份把这种脱节具体化，这比转写里的「10 年前」「8 年前」更精确：`2016: researchers implemented and trained their own models.`、`2018: researchers downloaded models (e.g., BERT) and fine-tuned them.`、`Today: researchers prompt API models (e.g., GPT/Claude/Gemini).`。也就是说，研究范式从「自己实现并训练」，到「下载 BERT 一类预训练模型做 fine-tune（微调）」，再到「直接 prompt（提示）API 模型」，抽象层级一路上升。讲师特别澄清 prompting 本身没有错，用它确实能做出惊人的东西，moving up levels of abstraction（提升抽象层级）总体上是提高生产率的好事；但紧接着的黄色高亮句才是重点：`These abstractions are leaky (in contrast to programming languages or operating systems).`——这些抽象是「有漏洞的」（leaky abstractions），而且与编程语言、操作系统那种可靠的抽象不同，一旦模型做不到你想做的事，只会 prompt 的人就没有任何 recourse（补救手段），因为下层对你是不透明的。

由此推出课程的核心论断：`Full understanding of this technology is necessary for fundamental research.`（对这项技术的完整理解，是做基础研究的必要条件）。理由在于，若想做 fundamental research（基础研究），只在 prompt 层面工作会把可探索的 design space（设计空间）压缩得极窄，因为你无法改动模型本身；真正的基础研究需要 tearing up the whole stack（掀开整个技术栈），从分词、架构、优化器直到系统实现都能动刀。课程因此把哲学写成 `Philosophy of this course: understanding via building.`（通过构建来获得理解）。这里的直觉是：抽象层级越高越省事，但可修改的自由度越低；而研究恰恰需要自由度。一个常见的误读是把「从零构建」当成反对使用 API 或反对抽象，其实讲师反对的只是「只会用、不懂里子」这种状态，并非否认抽象带来的效率。

「But there's one small problem...」——画面上这句灰色过渡引出了现实障碍：语言模型已经 industrialization（工业化）。转写里给出一串具体数字：3 年前 GPT-4 据称训练成本约 `$100 million`，如今前沿成本可能到了 `a billion`（十亿美元）量级，但讲师明确说这是 speculative（推测）；各大实验室在建的 GPU 数量「大到惊人」，画面第 88 行更给出一条转写完全没念到的实例：`2025: xAI builds cluster with 230K GPUs for training Grok. [article]`，即用 230K（23 万）块 GPU 的集群训练 Grok。更棘手的是细节不公开，第 90 行写着 `There are no public details on how frontier models are built.`，并引用 `From the GPT-4 technical report [OpenAI+ 2023]:`。报告第 2 节 "Scope and Limitations of this Technical Report" 中被高亮的原文是：

```text
Given both the competitive landscape and the safety implications of large-scale
models like GPT-4, this report contains no further details about the architecture
(including model size), hardware, training compute, dataset construction,
training method, or similar.
```

也就是说，出于 competitive landscape（竞争格局）与 safety implications（安全影响），architecture（含 model size）、hardware、training compute、dataset construction、training method 等一概不披露。把成本量级写成一个便于记忆的对照式（数值均来自转写，未作外推）：

$$ \text{cost}_{\text{GPT-4}} \approx \$100\text{M}\ (2023) \ \longrightarrow\ \text{cost}_{\text{frontier}} \sim \$1\text{B}\ (\text{推测}) $$

![GPT-4 技术报告引用页：因竞争与安全不公开架构、硬件、算力、数据构造等细节，并标注前沿模型不可及](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0005_00-05-34.jpg)

结论被黄色高亮成一句 `Frontier models are out of reach for us.`（前沿模型对我们而言遥不可及）。课程能做的是构建小语言模型，画面灰色行进一步给出量化边界：`We could build small language models (<1B parameters), but this might not be representative models.`，即只能做参数量 `<1B`（小于 10 亿）的小模型，而且它们未必能代表真正的前沿模型。这里有两个易错点值得警惕：其一，别把 `$1B` 当成已证实的事实，讲师亲口说这是推测；其二，别因为「我们也会训练模型」就以为课堂经验可以直接外推到前沿规模。这一章与前后内容的关系很紧密——它先立起「必须懂底层」的必要性，又用工业化把这条路堵窄，于是自然引出下一章要正面回答的质疑：既然只能玩小模型，那小尺度上得到的结论到底还算不算数？这正是 05:40 之后两个反例要处理的问题。

## 规模如何改变结论：FLOPs 分布与涌现行为 [05:40 - 07:07]

为了说明「小尺度结论未必迁移」，讲师给了两个来自 2021 年前后的例子，并预告后面会花更多时间做 flop counting（FLOPs 核算）、追踪算力究竟花在哪里。第一个例子关于计算量的分布：在小规模下，MLP 层占用的 FLOPs 比例约为 `44%`，而当模型扩展到 `175B` 参数时，这一比例升到约 `80%`。把它写成对照式：

$$ \frac{\text{FLOPs}_{\text{MLP}}}{\text{FLOPs}_{\text{total}}}:\quad 44\%\ (\text{小尺度}) \ \longrightarrow \ 80\%\ (\text{175B 参数}) $$

含义很直接：不同规模下「该优化什么」是不一样的。如果你在小模型上花大量力气优化 attention（注意力），到了大模型上可能享受不到同等收益，因为那时绝大部分算力已经被 MLP 吃掉。讲师由此点出 `what you optimize and what matters at large scale is going to be different from small scale`。这里的易错点是把某个小模型上的 profiling 结论当成普适规律，或者把「44% 与 80%」记成精确的测量常数——它们只是用来说明趋势的近似值，会随架构与核算口径变化；真正要带走的是「算力占比随规模漂移」这个结构性事实，而不是这两个数字本身。这也和前面 `high value-per-time` 的取舍呼应：在有限算力里，优化对象选错了，投入产出比会很差。

第二个例子是 emergence（涌现），即行为会随规模出现。转写里说，在 zero-shot / few-shot（零样本 / 少样本）的多类任务上，小模型看起来「几乎什么都没学会」，只有达到某个 critical scale（临界规模）后才突然出现大幅改善。画面给出的正是 Wei+ 2022 那组经典的八联折线图：横轴统一为 `Model scale (training FLOPs)`，刻度从 $10^{18}$ 到 $10^{24}$，纵轴多为 `Accuracy (%)` 或 `Exact match (%)` / `BLEU (%)`，并有一条粉色虚线表示 `Random`（随机基线）。图例里的模型是 LaMDA、GPT-3、Gopher、Chinchilla、PaLM 五条曲线。八个子任务分别是 (A) Mod. arithmetic、(B) IPA transliterate、(C) Word unscramble、(D) Persian QA、(E) TruthfulQA、(F) Grounded mappings、(G) Multi-task NLU、(H) Word in context；多数曲线在 $10^{22}$ 附近才从贴近随机基线突然抬升，例如 (D) Persian QA 的 PaLM 曲线一直平在约 25%，到最大规模才跃到 45% 左右。这些任务名、模型名、FLOPs 刻度与随机基线都是画面上的信息，转写只笼统说成「various tasks（若干任务）」。

![涌现行为的八联折线图：横轴为训练 FLOPs，多个任务在临界规模后才突然超越随机基线](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0006_00-06-49.jpg)

因此 `Small-scale models do not necessarily reflect frontier models`（小规模模型未必反映前沿模型）不是一句泄气话，而是一个方法论警告：只在小尺度做实验，可能根本观察不到某些现象，于是会误以为「这个方向没用」。需要留意两点局限：一是讲师强调这些图「是 quite a long time ago（很久以前）」的结果，涌现在多大程度上是真实相变、多大程度上受评测指标（如 Exact match 的非线性）影响，本段并未展开论证，不宜替他下结论；二是这两例只说明「结论会变」，并不等于「小模型毫无价值」。恰恰因为小尺度与大尺度既有关联又有断裂，接下来 07:09 就必须把知识拆成三类，逐一判断哪些能跨规模迁移、哪些不能——这正是本段从「令人沮丧」转向「我们还是能学到东西」的衔接点。

## 三类可迁移知识：机制、思维、直觉 [07:09 - 09:09]

面对「小模型不代表性」的质疑，讲师的反问是 `what can we learn that actually transfers?`（我们究竟能学到什么可迁移的东西），并把知识明确拆成三类。画面上的定义原文是：`Mechanics: how things work (what a Transformer is, how model parallelism works)`、`Mindset: squeezing the most out of the hardware, taking scaling seriously`、`Intuitions: which data and modeling decisions yield good accuracy`。用中文说，mechanics（机制）是「事物如何运转」的确定性知识，比如 Transformer 是什么、model parallelism（模型并行）怎么把参数和计算切到多张卡上；mindset（思维方式）是做事的习惯与取向，尤其是「榨干硬件」（squeezing the most out of the hardware）与「认真对待缩放」（taking scaling seriously）；intuitions（直觉）则是关于「哪些 data（数据）与 modeling（建模）决策能带来好准确率」的判断力。三者的分界在于可验证性：机制可以由构造直接推出，直觉却往往只能靠结果反馈。

结论被黄色高亮成 `We can teach mechanics and mindset (these do transfer).`（机制与思维可以教，而且确实能迁移），紧随其后的灰色行补上了转写没念到的一半：`We can only partially teach intuitions (do not necessarily transfer across scales).`（直觉只能部分地教，因为它不必然跨尺度迁移）。若用一个关系式概括画面论断（符号化表达，非画面原文公式）：

$$ \text{transfer}(\text{mechanics},\ \text{mindset}) \gg \text{transfer}(\text{intuitions} \mid \text{跨尺度}) $$

因此课程会反复强调 `profile and benchmark everything`（对一切做性能剖析与基准测试）并 optimize for efficiency（面向效率优化），因为这类习惯与对硬件、并行的理解在任何规模都成立；而关于「用哪份数据、改哪个模块会涨点」的直觉，则必须到真正能做大尺度实验的地方才能获得。容易出错的地方是把三类知识混为一谈：把某个在小模型上试出来的调参直觉当成定律照搬到大模型，或者反过来因为「直觉不迁移」就轻视 benchmark 训练——后者恰恰属于可迁移的 mindset。

![可迁移的三类知识：机制、思维、直觉，并标注机制与思维可教且可迁移](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0007_00-08-03.jpg)

接着讲师用 SwiGLU 说明「有些设计决策根本无法通过讲理来论证」。画面写着 `Some design decisions are simply not (yet) justifiable and just come from experimentation.`，例子是 `Noam Shazeer paper that introduced SwiGLU [Shazeer 2020]`；相比之下，mechanics 类的改进（parallelism、kernels 如何加速）是「by construction（按构造）」就能看出所以然的。论文 "4 Conclusions" 里被高亮的原句非常坦率：`We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence.`（我们无法解释这些架构为何有效，只能把它们的成功连同其它一切归功于神的恩典）。转写里把它口语化成 "We offer no explanation... divine benevolence"，画面则给出完整措辞，还保留了前半句 "We have extended the GLU family of layers and proposed their use in Transformer... These architectures are simple to implement, and have no apparent computational drawbacks."。这里的易错点是别把这句名言读成「架构设计纯属玄学、不用做实验」；讲师的用意恰好相反——正因为无法从理论推出，才必须靠实验与经验积累，这正是 intuitions 无法在课堂里完整传授的原因。这一章与后文的关系也很清楚：既然 mindset 里最该被训练的是「效率意识」，下一章就顺势进入 The Bitter Lesson，把效率从一个习惯提升为可写成的公式与课程总框架。

## The Bitter Lesson、效率乘法与课程总框架 [09:15 - 11:33]

The Bitter Lesson（苦涩的教训）是讲师口中「一直在流传、大家都在谈」的一个说法，他指出这里存在一个流传很广的误解。画面上把两种解读并排写得很清楚：`Wrong interpretation: scale is all that matters, algorithms don't matter.`（错误解读：规模就是一切，算法不重要）与 `Right interpretation: algorithms that scale are what matter.`（正确解读：能够随规模扩展的算法才重要）。转写里后一句被口误成 "algorithms that scale are what that or what matters"，应以幻灯片原文为准。这一字之差至关重要：前者会让人放弃算法研究、只堆算力，后者却把算法摆在中心——只不过要求算法具备 scale（可扩展性），即换到更大算力、更大模型上仍然有效。这与前一章 SwiGLU 的例子并不矛盾：一个激活函数是否有效可能靠实验才知道，但「它能否在大规模下继续奏效」属于可检验、可迁移的问题。

为了把「效率重要」讲得可计算，幻灯片给出一行加粗公式（画面原文）：

$$ \text{accuracy} = \text{efficiency} \times \text{resources} $$

其中 efficiency（效率）被定义为 output over input（产出 / 投入），resources（资源）就是投入的那一项。这个乘法结构的含义是：准确率由「每单位资源能换来多少效果」与「投入多少资源」共同决定，二者缺一不可。紧接着一句 `In fact, efficiency is way more important at larger scales (can't afford to be wasteful).` 点出规模效应：小规模实验如果慢一倍，你大不了多等一倍时间再回来看结果；但大规模训练动辄 `hundreds of millions of dollars`（数亿美元），哪怕是 `5%` 的效率提升都可能是件大事，浪费不起。这里的易错点是把公式读成「资源越多越好」而忽略效率项，或者把 `5%` 当成精确阈值——它只是用来说明「大规模下边际改进的价值被成本放大」的量级例子。

经验证据来自黄色高亮的一行：`[Hernandez+ 2020] showed 44x algorithmic efficiency on ImageNet between 2012 and 2019.`，即 2012 到 2019 年间 ImageNet 上取得了 `44x`（44 倍）的 algorithmic efficiency（算法效率）提升。画面下方还嵌了论文卡片，标题是 "Measuring the Algorithmic Efficiency of Neural Networks"，作者 Danny Hernandez 与 Tom B. Brown，日期 `2020-05-08`，左下角可见 arXiv 编号 `2005.04305`；摘要首句把 AI 进步归为三个因素：`algorithmic innovation, data, and the amount of compute available for training`（算法创新、数据、可用于训练的算力），并指出算法进步历来比算力与数据更难量化，本文衡量的是「达到过去能力所需算力的下降」。讲师补充说 hardware（硬件）这些年当然也进步了很多，而算法改进与硬件进步相乘（multiply them together），才带来效率与准确率的巨大跃升。要注意别把 `44x` 归功于硬件或归功于单一算法，也别把它外推到 LLM 上——这个数字是 ImageNet 分类任务在 2012–2019 的测量结果，本段只是借它说明「算法效率可被量化且贡献很大」。

![The bitter lesson 幻灯片：错误与正确解读、accuracy=efficiency×resources 公式与 44x 算法效率论文](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0009_00-10-32.jpg)

有了以上铺垫，课程的总框架（framing）在画面上被写成一句问话：`Framing: what is the best model one can build given a certain compute and data budget?`（在给定的算力与数据预算下，能构建出的最好模型是什么？），紧接着加粗给出 `In other words, maximize efficiency!`（换句话说，最大化效率）。画面同时出现新的函数签名 `def current_lm_landscape():`（第 7 行，黄色高亮），意味着接下来要转入语言模型全景的梳理。讲师对预算本身作了限定：pre-training（预训练）阶段主要谈 compute budget（算力预算），因为默认假设 data 远多于 compute；但若处在 data limited（数据受限）的场景，或者手头囤了大量 `B200`（NVIDIA 的 GPU），就可能反过来变成 data bound（数据受限），此时优化对象要相应调整。一个常见的疏漏是把「数据越多越好」当成无条件前提，忽略了这个假设只在特定预算结构下成立。至此，效率从 07:09 的 mindset 习惯，经过 The Bitter Lesson 的公式化，正式成为贯穿全课的主线；而 `def current_lm_landscape():` 则把镜头从「为什么学」转向「这门技术研究对象本身是怎么演化来的」，也就是下一章的语言模型简史。

## 语言模型简史：从熵与 N-gram 到预训练、规模化与早期复现 [11:36 - 15:35]

进入历史梳理前，讲师先提醒 language models（语言模型）并不新鲜。画面在 `def current_lm_landscape():`（第 127 行）之下分出 `Pre-neural (before 2010s)`（前神经网络时代）与 `Neural ingredients (2010s)`（2010 年代的神经组件）两节。前一节只有两条：`Language model to measure the entropy of English [Shannon 1950]`——Shannon 在 1950 年代用语言模型测量英语的 entropy（熵）；以及 `N-gram language models (used in machine translation and speech recognition systems)`，即 N-gram 模型长期用于 machine translation（机器翻译）与 speech recognition（语音识别），虽不是完整系统，却对生成流畅文本起关键作用。转写里只说了「Shannon back in the 50s」「for a long time」，画面则补上了 `[Shannon 1950]` 这样的精确引用。这里要先建立一个概念直觉：语言模型的原始定义是「对 token 序列赋予概率分布」，熵衡量的正是这种分布的不确定性，N-gram 则是用前 $n-1$ 个词预测第 $n$ 个词的最朴素实现——它们共同构成现代模型的前史，但并非现代模型的直接祖先，讲师明确说「现代语言模型的谱系来自 neural architectures（神经架构）」。

`Neural ingredients (2010s)` 这一节的八条 bullet 是本段信息密度最高的部分，每一条都带方括号里的作者与年份（转写只念了名字，年份全部来自画面）：`Long-Short Term Memory (LSTM) [Hochreiter+ 1997]`、`First neural language model [Bengio+ 2003]`、`Sequence-to-sequence modeling (for machine translation) [Sutskever+ 2014]`、`Adam optimizer [Kingma+ 2014]`、`Attention mechanism (for machine translation) [Bahdanau+ 2014]`、`Transformer architecture (for machine translation) [Vaswani+ 2017]`（此行被黄色高亮）、以及灰色的 `Mixture of experts [Shazeer+ 2017]` 与 `Model parallelism [Huang+ 2018][Rajbhandari+ 2019][Shoeybi+ 2019]`。讲师特别纠正了一个易错点：Bengio 2003 那篇「首个神经语言模型」并不是 LSTM，而只是一个看很小上下文的 feedforward network（前馈网络）；seq2seq 的贡献在于「大胆地把整个句子压缩成一个向量」；attention 与 Transformer 最初都是为机器翻译开发的。把这些组件按年代连起来，就能看出 2010 年代是架构、系统与优化器思路集中爆发的十年。

![语言模型全景图一：前神经网络时代的熵与 N-gram，及 2010 年代带文献年份的神经组件清单](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0011_00-13-01.jpg)

`Early foundation models (late 2010s)` 给出三条：`ELMo: pretraining with LSTMs, fine-tuning improves downstream tasks [Peters+ 2018]`、`BERT: pretraining with Transformer, fine-tuning improves downstream tasks [Devlin+ 2018]`、`Google's T5 (11B): cast everything as text-to-text [Raffel+ 2019]`。前两者确立了「大规模文本预训练 + 下游任务 fine-tune」的范式（如 question answering 问答），T5 则把一切任务都 cast（转换）成 text-to-text，正是转写里所说「prefigure 了 prompt 进、response 出」的那篇 Google 论文，且画面给出其参数量 `11B`。`Embracing scaling`（拥抱规模化）一节把参数量写得更明白，可整理成一条对照式：

$$ \text{GPT-2}\,(1.5\text{B}) \to \text{GPT-3}\,(175\text{B}) \to \text{PaLM}\,(540\text{B}),\qquad \text{Chinchilla}\,(70\text{B}) $$

对应的原文是 `OpenAI's GPT-2 (1.5B): fluent text, first signs of zero-shot [Radford+ 2019]`、`Scaling laws: provide hope / predictability for scaling [Kaplan+ 2020]`（此行黄色高亮）、`OpenAI's GPT-3 (175B): in-context learning [Brown+ 2020]`、`Google's PaLM (540B): massive scale, undertrained [Chowdhery+ 2022]`、`DeepMind's Chinchilla (70B): compute-optimal scaling laws [Hoffmann+ 2022]`。转写里说 GPT-3 比 GPT-2「almost more than 10x larger」，画面则分别标注 GPT-2 (1.5B) 与 GPT-3 (175B) 的参数量；并提到 Google 训了个 massive 但 under-trained（训练不足）的模型、以及当时尚未与 Google 整合的 DeepMind 已算出 compute-optimal（计算最优）缩放定律——幻灯片把这些一一对应到了具体模型与年份。这里的易错点是把 PaLM 与 Chinchilla 混为一谈：前者胜在规模、败在训练不足，后者参数更小却给出了计算最优的配方；也不要把 scaling laws 当成自然定律，它提供的是 `hope / predictability`（可预测性）。

最后 `Open models` 的 `Early attempts (attempts to replicate GPT-3):`（复现 GPT-3 的早期尝试）列出三条：`EleutherAI's open datasets (The Pile) and models (GPT-J) [Gao+ 2020][Wang+ 2021]`、`Meta's OPT (175B): GPT-3 replication, lots of hardware issues [Zhang+ 2022]`、`Hugging Face / BigScience's BLOOM (176B): focused on data sourcing [Workshop+ 2022]`（黄色高亮）。转写把 EleutherAI 听成了「Luther」，并说 Meta 第一个 LLM 是 `175 billion parameters` 的复现、遇到许多硬件问题，Hugging Face Big Science 也属早期尝试且「not a very strong」；画面则补出 The Pile、GPT-J、`176B` 与「focused on data sourcing」这些细节。GPT-3 之后这批尝试被讲师称作很多人的 wake-up call（警钟），但早期开放模型都不够强。这一章与前后关系清晰：它把「为什么学」落到了具体技术谱系上，而结尾的 `Open models` 又直接打开下一章——近三年开放权重生态为何突变，以及它对这门课为何不可或缺。

## 开放模型生态、语言模型定义的演变与可执行讲义 [15:35 - 19:50]

最近三年 open model ecosystem（开放模型生态）发生了明显变化，Meta 以 Llama、Llama 2、Llama 3 系列带头，Mistral 加入，随后是一大批中国模型。画面第 164–168 行给出的精确条目是 `Alibaba's Qwen models [Qwen+ 2024][Yang+ 2025]`、`Moonshot's Kimi models [Kimi Team 2025][Kimi Team 2026]`、`Z.ai's GLM models [GLM-4.5 Team 2025][GLM-5-Team 2026]`、`Minimax's models [MiniMax M2.5]`、`Xiaomi's MIMO models [Xiaomi MIMO v2]`，上方还有被截断的 `[DeepSeek-AI+ 2025]`；转写里讲师只含糊说到 ByteDance、Tencent「大概也有」，并坦承 "it's hard to keep track of everything"（很难全部记住），点名的是 Deep Seek 与 Qwen。结论句是 `These models are approaching closed models (GPT, Claude, Gemini, etc.).`——开放权重模型正在逼近闭源模型，按不同人或不同 benchmark 可能略落后或相当，但已是工业界广泛使用的、非常 credible（可信）的模型。

比「只放权重」更进一步的是全开放：`Open-source models (weights + paper + code + data):` 下列出 `AI2's Olmo models [Groeneveld+ 2024][Team OLMo 2024][Team Olmo 2025]`、`NVIDIA's Nemotron models [Parmar+ 2024][NVIDIA+ 2025]`，以及黄色高亮的 `Marin's models (open development) [Marin 8B retro][Marin 32B retro]`（转写把 Marin 听成 "Marine project"，并说明这是讲师本人参与的项目）。括号里的 `Marin 8B`、`Marin 32B` 是画面给出的具体规模。灰色行还写着 `Openness is important for trust and innovation [Kapoor+ 2024]` 与 `Ideas from open models enable us to teach CS336.`。讲师强调开放生态的理由很实在：没有这些模型，这门课几乎开不出来——正因为仍有大量论文在讨论大型 MoE 与 RL 系统如何运作，外界才得以 glimpse（窥见）前沿模型的构造并 triangulate（三角定位）拼图。但他也诚实指出局限：即便是 Qwen 这类论文也缺很多细节、无法复现，尤其 data mixture（数据配比）外界并不知道，因此只是 "much better than nothing"（比什么都没有强得多）。易错点有两个：别把 open-weight（仅开放权重）与 open-source（连论文、代码、数据都开放）混为一谈；也别把「模型权重可下载」等同于「训练过程可复现」。

![全开放模型清单与语言模型定义演变：Olmo、Nemotron、Marin 8B/32B 及 2018 至 2026 的四种定义](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0015_00-17-58.jpg)

`What is a language model?`（什么是语言模型）这一节用四个年份概括了定义的迁移，画面原文比转写更规整：`2018 (BERT): something you fine-tune`（用来微调的东西）、`2020 (GPT-3): something you prompt`（用来提示的东西）、`2022 (ChatGPT): something you talk to [example conversation]`（可以与之交谈的东西，附示例对话链接）、以及黄色高亮的 `2026 (agents): something that acts autonomously [example trace]`（能够自主行动的东西，附示例轨迹链接）。转写里讲师想点开那个链接展示「giant agent trace（巨大的智能体轨迹）」时正好没网，并感慨给模型一页文本它就能完成相当复杂的 agentic coding（智能体编码）任务，「今天我们对语言模型的要求，恐怕超出 10 年前任何人的想象」。但紧接着的灰色两行才是重心：`The fundamentals are the same (attention, kernels, optimization).`（基础没变：attention、kernel、优化）与 `The specs are different (longer context, inference efficiency matters even more).`（规格变了：更长上下文，推理效率更加重要）。也就是说，我们仍 build on GPUs and kernels（在 GPU 与 kernel 上构建）、仍用 gradient / stochastic gradient（梯度 / 随机梯度）类方法优化、仍离不开 Transformer 与 attention；变的只是需求规格。讲师因此说好消息是课程不必重写，只需更新两节讲最新的中国架构，这些基础「至少目前」还在。这里的易错点是被 agents 的热闹带偏，以为底层换了范式；也不要把「只 touch two sections」理解成内容缩水，它恰恰说明 fundamentals 稳定。

最后落在 `executable lecture`（可执行讲义）这个短暂插曲上。讲师说 `what is this program?`，答案是这份讲义「看起来像 Python 程序，因为它其实就是 Python 程序」，只是被渲染（rendered）成便于观看的样式，逐步 step through（单步执行）时代码是真的在跑。19:13 的画面给出了 `def main():` 的完整骨架（原样抄录，含行号对应的缩进与注释）：

```python
from references import olmo_7b_2024, olmo_2_2025, olmo_3_2025
from references import nemotron_15b_2024, nemotron_3_2025
import tiktoken

def main():
    welcome()
    why_this_course_exists()
    current_lm_landscape()
    what_is_this_program()
    course_logistics()
    course_syllabus()
    tokenization()  # First unit
```

![可执行讲义 def main() 代码骨架：依次调用各章节函数，tokenization() 标注为第一单元](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0016_00-19-13.jpg)

其中 `current_lm_landscape()` 一行被黄色高亮，正好对应本段 11:36 之后正在讲的部分；`import tiktoken` 与两条 `from references import ...`（`olmo_7b_2024`、`olmo_2_2025`、`olmo_3_2025`、`nemotron_15b_2024`、`nemotron_3_2025`）是画面上真实出现的标识符，说明讲义会直接引用前面提到的开放模型作为文献条目。这张图把开头 `def welcome():`、`def why_this_course_exists():` 的悬念收了口：本段 00:00–19:50 执行的正是 `welcome()`、`why_this_course_exists()` 与 `current_lm_landscape()` 三个函数，而 `what_is_this_program()`、`course_logistics()`、`course_syllabus()` 与带注释 `# First unit` 的 `tokenization()` 是接下来要讲的单元——后者正呼应本课标题里的 Tokenization（分词）。易错点是别把这些 `def` 当成装饰性标题，也别把 `# First unit` 的注释漏掉：它明确标出分词是第一个正式单元，为 19:50 之后的内容划好了起点，也解释了为什么整个 Overview 要用一份可执行代码串起来。

## 可执行讲义与课程后勤：5 学分、5 次作业 [19:50 - 21:52]

本段从上一节「可执行讲义」（executable lecture）的收尾切入。讲师刚刚从某个函数返回到 `main()`，借此点明这套讲义的设计意图：讲义本身就是一份 Python 程序，经过渲染后才是课堂上看到的幻灯片，讲师逐步执行时代码会被真正运行。画面上把这一能力概括为两条，`view and run code (since everything is code!)`（查看并运行代码，因为一切都是代码）与 `see the hierarchical structure of the lecture`（看到讲义的层级结构）。为了说明「连最琐碎的片段都是源码」，幻灯片直接嵌了一段可执行的示例代码，它不是截图而是讲义程序的一部分：

```python
total = 0
for x in [1, 2, 3]:
    total += x
```

紧接着出现的 `def course_logistics():` 就是讲义这个大程序里的一个函数。画面左侧还保留着讲义源码的行号，从 191 行一直排到 205 行，这本身就是「一切皆代码」最直接的证据：幻灯片上的每个标题、每条 bullet 都对应源码中的一行，函数之间的调用关系构成了整堂课的目录。理解这一点很关键，因为后文会陆续出现 `def course_syllabus():`、`def basics():`、`def systems():`，它们不是装饰性的排版，而是课程结构的真实骨架。

进入后勤信息后，画面给出的原文是 `All information online: https://stanford-cs336.github.io/spring2026/` 与 `This is a 5-unit class.`。讲师口头提到网站是 cs336.stanford.edu，画面显示的则是 Spring 2026 的 GitHub Pages 完整地址，记录时以画面上的 URL 为准更可靠，两者指向的是同一套课程资料。紧接着是课程强度的说明：这是一门 5 学分（5-unit）课，共有 5 次作业，而且「intense」。画面用黄色高亮标出一条 `Comment from Spring 2024 course evaluation:`，下方斜体引文原文为 `The entire assignment was approximately the same amount of work as all 5 assignments from [CS224N] plus the final project. And that's just the first homework assignment.`（整个作业的工作量约等于另一门课 5 次作业加期末项目的总和，而这还只是第一次作业）。转写里讲师也说，有课程评价认为第一次作业相当于 CS224N 全部 5 次作业，他本人觉得这种说法可能夸张，但紧接着补了一句 `better to be conservative in your estimates`——宁可把工作量估计得保守一点。这里的数字关系值得记牢：5-unit、5 assignments、Spring 2024 的评价、以及「仅第一次作业」这个限定语，共同构成本课的时间预算预期。

随后是 `Why you should take this course`，画面两条 bullet 分别是 `You have an obsessive need to understand how things work.`（有一种强迫症式的、非弄清楚事物如何运作不可的需求）与被高亮的 `You want to build up your research engineering muscles.`（想锻炼研究工程肌肉）。讲师展开说，首要动机应当是对语言模型工作原理的纯粹好奇心（pure curiosity），其次是通过这门课练出更强的研究与工程能力，从而获得进入一个全新环境、也能应对任何突发问题的信心。为了定位这门课，他还做了一个类比：自己刚到 Stanford 时开过 statistical learning theory（统计学习理论），那门课教的是机器学习的理论侧，好处是让人读论文时能看懂所有数学；而如今整个领域已经大幅转向 systems（系统）与 empirical（实证）一侧，CS336 就是与之对应的那类课程——它给人的深度足够，以至于学完之后 `everything else seems kind of easy`（其他一切显得相对容易）。这句话点明了本课的价值取向：不是教最热门的技巧，而是打一层足够厚的底。

从全段结构看，这一章承接 17:32–19:07「语言模型定义的演变与不变基础」，把话题从宏观论述切进具体开课安排，之后 21:52 起会转向「为什么不该选这门课」，再往后才是作业、大纲与正课内容。易错点有两处：其一，不要把「5 次作业」理解成五次轻量练习，画面引文强调的是仅第一次作业就抵得上另一门课整学期作业加 final project；其二，不要把课程网站当成只放 PPT 的静态页面，它是 executable lecture 的源码载体，代码可以逐行步进、逐函数展开。还有一个容易忽略的细节：讲义行号 191–205 显示 `course_logistics()` 只是整个层级中的一个节点，后面还会依次调用 `course_syllabus()`、`basics()`、`systems()` 等，这种层级结构本身就是课程内容组织方式的隐喻，也为后面「五部分对应五次作业」埋下伏笔。

![可执行讲义的代码示例与 course_logistics 小节，含 5 学分说明、课程网址及 Spring 2024 课程评价引文](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0017_00-20-27.jpg)

## 为什么不该选这门课，以及在家跟课 [21:52 - 23:28]

讲师特意强调「为什么不该选」很重要，因为确实存在不该选的正当理由。画面 `Why you should not take this course` 下列出三条，每条都带括号里的具体建议。第一条是 `You actually want to get research done this quarter. (Talk to your advisor.)`——这学期真的想推进研究，就该先和导师沟通；转写里讲师补充，导师应当知道你在选这门课，否则学期中途 `there might be some surprises`（可能会有一些意外）。第二条是 `You are interested in learning about the hottest new techniques in AI (e.g., multimodality, RAG ... you should take a seminar class for that.)`——想学 AI 里最热门的新技术，应该去上 seminar 或 topics 类课程；画面括号里点名了 multimodality（多模态）与 RAG，转写里讲师则明确说本课 `We don't do multi-modality. We don't talk about agents in any depth`（不做多模态，也不深入讲 agents）。第三条是 `You want to get good results on your own application domain. (You should just prompt or fin[etune an] existing model.)`——想在自己的应用领域拿到好结果，正确做法是直接 prompt 或微调一个已有模型。

这三条理由分别对应时间冲突、主题错配与目标错配，其中第三条被讲师展开成一个代价递增的推荐顺序。转写原话是 `I always recommend just prompt the model fine tune the model and then as a last resort you pre-train your own model because it is a pain and it's also expensive. But it's a lot of fun.`，可以写成一条清晰的路径：

$$
\text{prompt} \;\longrightarrow\; \text{fine-tune} \;\longrightarrow\; \text{pre-train (last resort)}
$$

从左到右，工程代价与金钱成本陡增：prompt 几乎零成本，fine-tune（微调）需要数据与算力，而 pre-train（预训练）被他形容为 `a pain and it's also expensive`（又麻烦又贵），只能作为 `as a last resort`（万不得已的最后手段）。但句尾那句 `But it's a lot of fun` 又把它拉回课程立场——本课恰恰就是要带你做这件又贵又麻烦但很好玩的事。这里最容易读错的是把「不该选」当成课程质量警示：它其实是选课适配性筛选，剔除的是目标与课程不匹配的人；另一个易错点是以为「不做多模态、不讲 agents」意味着课程内容陈旧，实际上讲师的取舍是有意把有限课时集中在底层机制上。

接着是 `How you can follow along at home`（如何在家跟课），画面两条 bullet 为 `All lecture materials and assignments will be posted online, so feel free to follow on your own` 与 `Lectures are recorded via CGOE.`。转写里讲师还补充，录像由 CGOE 完成，之后会发布到 YouTube；他同时提醒，看录像当然很好，但 `you learn really by doing the assignments`（真正的学习来自动手做作业），在家跟课的人得自己想办法激励自己把作业做完。这句话不是客套：结合上一章可知，作业不提供脚手架代码，又需要集群资源才能跑真实训练，自学者缺少的正是 unit tests、cluster 与 leaderboard 这些闭环环节。画面行号 215 到 223 也显示，这一节下面紧接着就是 `Assignments` 小节，说明「跟课」与「做作业」在讲义里本就是连在一起设计的。

从结构关系看，本章把「谁该留下来」的边界划清之后，23:28 立刻进入作业的具体设计，形成「筛选 → 承诺」的衔接：留下来的人要接受无脚手架、单元测试驱动、本地加集群两段式的训练方式。局限也要说清楚：录播与讲义虽然公开，但 CGOE 只是录制渠道，讲义与作业的权威来源仍是上一章那个 spring2026 站点；此外讲师明说 `watching the lectures is great` 却不等于学会，缺少动手环节时，这一整段后勤说明对自学者只是信息而非能力。还有一处细节值得留意：画面上 `Why you should not take this course` 与 `How you can follow along at home` 出现在同一屏，后者被黄色高亮，说明讲师在课堂上正是从「劝退」顺势滑到「不选也能跟」的过渡，语气上是连贯的一段，而非两个孤立公告。

![为什么不该选这门课的三条理由与 How you can follow along at home 一节，含 CGOE 录制说明](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0019_00-22-56.jpg)

## 作业设计、AI 政策与算力 [23:28 - 27:17]

作业（Assignments）部分先讲设计哲学。转写里说，难点在于「from scratch」不能只写成一句 `build a language model` 就当作业交出去，所以课程不提供 scaffolding code（脚手架代码），但提供 `a bunch of unit tests`，目的是避免 `sparse rewards setting`——即提交作业后只得到「对／错」这种稀疏反馈的窘境。画面 `Assignments` 下四条 bullet 给出了更精确的原文：`5 assignments (basics, systems, scaling laws, data, alignment).`、`No scaffolding code, but we provide unit tests and adapter interfaces to help you check correctness`、`Implement locally to test for correctness, then run on cluster for benchmarking (accuracy and ...)`、`Leaderboard for some assignments (minimize perplexity given training budget).`。其中 `adapter interfaces`（适配接口）是转写没有念到的词，它说明测试不只校验单个函数的返回值，还约定了模块之间的接口形状；这与「不给脚手架」并不矛盾，自由度留给实现，约束放在边界上。排行榜的目标可以写成一个带约束的优化式：

$$
\min_{\text{implementation}} \ \text{perplexity} \quad \text{subject to} \quad \text{training budget} \le B
$$

转写 [24:37] 的说法是 `try to minimize the perplexities given some sort of budget`，[24:41] 还打趣说 Marcel 和 Herman 是去年 leaderboarding（刷榜）的高手，想要秘诀可以去问，只是未必肯说。这里的 `perplexity`（困惑度）是后面反复出现的指标，作业一与系统部分都会以它作为优化目标。

具体实现流程被拆成两段：先在 laptop 本地实现并用 unit tests 检查正确性，再到课程提供的 cluster 上跑一次真实训练以观察 accuracy，或申请一批 GPUs 来 benchmark 某个 kernel 的性能。画面写的是 `run on cluster for benchmarking (accuracy and ...)`，转写 [24:17] 补充 `get a bunch of GPUs to actually benchmark the performance of a some kernel`。这种「本地验正确、集群看效果」的分工，把昂贵的算力留给真正需要它的环节。数字与参数方面，本章涉及 5 次作业、unit tests、adapter interfaces、cluster、GPUs，以及排行榜里那个尚未具体化的预算 $B$（到 34:43 会落地成 `45 minutes on a B200`）。易错点是把 leaderboard 当成纯粹的刷榜游戏：它的真实目的是逼学生在固定预算下权衡速度与精度，正好呼应全课 efficiency（效率）主线；另一个易错点是把「无脚手架」误读成「无反馈」，实际上 unit tests 提供的正是高频、可定位的密集反馈。

AI 政策是本年度的新变化，画面 `AI policy` 下四条 bullet 以黄色高亮开头：`Coding agents can solve all the assignments, but you won't learn anything.`，其余三条为 `AI can be tremendously useful for answering questions and tutoring.`、`You must use our provided AGENTS.md file, which asks the AI to be pedagogically-minded.`、`Please read our AI policy guide.`。转写里讲师坦承，如今 coding agents 已经强到能解掉所有作业，但把作业 PDF 直接喂给 Claude Code 显然什么也学不到；与此同时 AI 在答疑与辅导上极有价值，于是课程提供一个 `AGENTS.md` 文件（等价于一段 prompt），要求 AI 以 `pedagogically-minded`（教学导向）的方式回应：可以回答代码问题、澄清概念理解，却不会在作业要求实现 Transformer 时顺手把整个 Transformer 生成出来。讲师强调这是第一年尝试，请学生试用后给反馈。画面用词是 `AGENTS.md`，转写念作 `agents.md file`，属同一文件。这一政策最容易被误读的地方是「禁止用 AI」——恰恰相反，它承认 AI 有用且允许使用，禁止的只是代做，并且 `You must use` 表明使用指定 prompt 是硬性要求而非可选建议。

算力（Compute）方面，画面两条 bullet 为 `Thanks to Modal for providing compute. 🙏` 与 `Please read the guide on how to access and use the compute.`。转写补充了更多背景：今年由 Modal 提供平台上的 compute credits；与去年 SSH 登录集群不同，今年更接近 API 形式，讲师起初对此 `skeptical`（存疑），实际用下来却觉得 `pretty pleasant to use`（相当好用），课程也写了访问与使用指南。画面底部还露出下一段的讲义源码，把「五个部分对应五次作业」直接写成了函数调用：

```python
def course_syllabus():
    basics()          # Assignment 1: tokenization, model architecture, training
    systems()         # Assignment 2: kernels, parallelism, inference
    scaling_laws()    # Assignment 3: scaling laws
```

这段源码是画面独有、转写只口头说 `there's basically five parts mirroring the five assignments that you'll have: Basic systems scaling laws data and alignment` 的地方，它把抽象的大纲落成了可执行的对应关系。局限需要说明：画面只显示到 `scaling_laws()` 三行，`data()` 与 `alignment()` 两行落在裁切区域之外，因此不能据此断言它们的注释原文；但转写明确五个部分的完整名单是 basics、systems、scaling laws、data、alignment。从关系上看，本章结束后 27:17 正式进入大纲，而源码里的 `basics()` 正是下一段的入口，讲义的层级结构在这里第一次显出实际作用。

![Assignments 五条说明与 AI policy 一节，含 AGENTS.md 与 pedagogically-minded 要求](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0021_00-25-25.jpg)

## 五部分大纲、basics 与分词：tokenizer 的形式化定义 [27:17 - 29:50]

[27:17] 起讲师切入大纲，说明课程 basically five parts mirroring the five assignments，即 basics、systems、scaling laws、data、alignment 五部分，并会逐一给一个 `taste`（初步印象）。其中 basics 是 `the first two weeks`（前两周），[27:45] 目标是 `just be able to train a language model and build it from scratch`，组件有三：tokenize the data、define architecture、implement optimizer and trainer。画面 `def basics():` 下用黄色高亮写出 `Goal: be able to train a basic language model`，下一行是 `Components: tokenization, model architecture, training`，行号约在 266–267。这里可以把本章的定位写成一个三段式依赖：

$$
\text{tokenization} \;\longrightarrow\; \text{model architecture} \;\longrightarrow\; \text{optimizer + trainer}
$$

三者顺序不能颠倒：先决定模型操作的「原子」，再在原子序列上定义模型，最后才谈怎么把它训练出来。数字方面是前两周、5 个部分、3 个组件；而「其余四部分是干什么的」这个疑问，讲师用一句 `Well, we'll get to there` 暂时搁置，留待后续章节。

分词（Tokenization）被定义为「模型所操作的原子是什么」（`What are the atoms that the model operates on?`）。转写 [28:14] 给出形式化定义：`formally a tokenizer converts between raw inputs which are just bytes and a sequence of integers which are represent the tokens`；[28:26] 又补一句 `Conceptually it's a segmentation of the text`（概念上是对文本的一种切分）。画面写作 `Formally: a tokenizer converts between raw inputs (bytes) and sequences of integers (tokens)`。更重要的是，画面上方（讲义第 258 至 264 行）给出了 tokenizer 的抽象接口源码，这部分转写完全没有念到，是理解「不提供脚手架」如何落地的关键：

```python
"""Abstract interface for a tokenizer."""
def encode(self, string: str) -> list[int]:
    raise NotImplementedError

def decode(self, indices: list[int]) -> str:
    raise NotImplementedError
```

用类型语言表述就是：

$$
\text{encode}: \text{str} \to \text{list[int]}, \qquad \text{decode}: \text{list[int}] \to \text{str}
$$

`encode` 把字符串映射为整数列表，`decode` 把整数列表映射回字符串，两个方法体都只有 `raise NotImplementedError`。这既解释了「讲义给接口、不给实现」的教学方式，也把「字节／字符串 ↔ 整数序列」的双向转换固化成一份契约；从类型标注可以直接读出二者互为反向映射，因而任何实现都必须保证往返一致，否则后续在 token 序列上做的一切建模都失去意义。需要克制的一点是：画面只给出了接口与类型，没有写出 round trip（往返）这个术语，因此这里只做从代码出发的推断，不额外添加讲师未说的验证步骤。

![def basics() 的目标与组件，以及上方 tokenizer 抽象接口的 encode/decode 源码](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0023_00-27-53.jpg)

接着是 BPE 与画面上那张转写一字未提的环形示意图。图的上方左侧是字符串 `Stanford was founded in 1885.`，右侧是一串整数 `93447, 9201, 673, 24303, 306, 220, 13096, 20, 13`，下方把同一句话按不同底色切成彩色片段，正中间一个蓝色 `decode` 框，环形箭头把字符串与整数序列连成往返——这正是 tokenizer 编解码过程最直观的一张图。图下正文为 `Popular tokenizer: Byte-Pair Encoding (BPE) [Sennrich+ 2015]`、`Intuition: break input into frequently-occuring chunks`（画面拼写为 occuring）、`Efficiency lens`，以及两条 bullet：`Reduce context length (1000 bytes → ~250 tokens)` 与黄色高亮的 `Adaptive computation (more modeling capacity on interesting parts of input)`。转写 [28:31] 说 BPE 直觉上把输入拆成 `frequently occurring chunks`（频繁出现的块）；[28:45] 从效率视角解释，分词把长的原始字节流压缩成更少的 token；[28:55] 又指出更微妙也更重要的 `adaptive computation`（自适应计算）：有些地方字节很多却应当压成一个 token，而稀有或有趣的部分应保留为多个 token，从而把建模容量倾斜到真正需要的地方。画面把第一句量化成了 `1000 bytes → ~250 tokens`，据此可算出平均每个 token 覆盖约

$$
1000 \,/\, 250 \approx 4 \ \text{bytes per token}
$$

即约 4 倍的序列长度压缩。之所以在意长度，是因为下一章会明确给出 attention 复杂度为 $O(n^2)$（$n$ 为序列长度），长度缩到约 $1/4$，理论上 attention 计算量可降到约 $(1/4)^2 = 1/16$。需要说明的是，这个 $1/16$ 是依据画面数字与后续公式的推算，并非讲师原话；同样，`1000 bytes → ~250 tokens` 是示意性数量级，不是通用常数，压缩比会随语言、数据与词表大小变化，不能当成 BPE 的固定指标。另一处易错点是把 BPE 理解成无损压缩：它改变的是序列切分与建模粒度，`93447` 这类整数索引本身不携带语义，只是词表下标。

![字符串与整数序列往返回环形示意图，以及 Byte-Pair Encoding 的效率视角与 1000 bytes 到约 250 tokens](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0024_00-29-08.jpg)

本章最后回到「讲分词是否还有必要」。[29:17] 讲师说每年都盼着不必再教 tokenization，梦想是有一个端到端、直接操作字节的方案；画面灰色小字把这一梦想写成 `The dream: tokenizer-free model architectures, which operate directly on bytes`，并附一串论文引用 `[Xue+ 2021]`、`[Pagnoni+ 2024]`、`[Deiseroth+ 2024]`、`[Hwang+ 2025]`，紧接着一句 `These are promising, but have not yet been scaled up to the frontier.`。转写里讲师提到 `recently there's been this H net work that seems promising`，但 `so far these have not been scaled to the frontier`，而 `the frontier models are still using tokenizers`（前沿模型仍在用 tokenizer），所以 `it would be still wise to teach tokenizers`。画面给出的 2021、2024、2025 这些年份与那串 9 个整数的 token 序列，都是转写里没有的增量信息。关系上，这一节收束后 [29:50] 立刻转入架构，逻辑正是「tokenize 之后，在得到的 token 序列之上定义模型」。局限在于：本段只交代 BPE 的直觉、效率收益与前沿现状，没有展开合并算法本身，也没有说明词表大小如何影响压缩比，这些内容留到课程后面的分词专题；把它当成 BPE 的完整教程会是误读。

## Transformer 架构的改进与形状参数 [29:50 - 32:17]

[29:50] 讲师把话题从分词接到架构：`after you tokenize your input, you have a bunch of tokens. Now you define a model on top`。他假定大家都熟悉 Transformer，上过 CS224N 的人已经见过，但 [30:09] 强调 `since then, there have been a lot of improvements or refinements to transformers`，并说明 Tatsu 稍后会细讲，这里只快速过一遍「可能需要考虑的类型」。画面上并排放了两张架构示意图，恰好是这种演进的直观对照：左图是较原始的形态，`Token Embedding` 与 `Absolute Position Embeddings` 两路汇入 `Add & Dropout`，再往上接 Transformer Block；右图从 `Input tensor with shape (batch_size, seq_len, d_model)` 出发，先过 `Norm`，再进入 `Causal Multi-Head Self-Attention`，最后是 `Dropout`。两图并列展示的正是 `pre-norm versus post-norm`（归一化放在子层之前还是之后）之争，对应转写里那句含糊的 `how you normalize ... has evolved`。形状标识符 `(batch_size, seq_len, d_model)` 是画面独有、转写没有的信息，必须原样保留，可以写成讲义标注的形式：

```python
# 讲义标注的输入张量形状
input_tensor.shape == (batch_size, seq_len, d_model)
```

这里顺带给出本章最重要的一个公式。转写 [30:45] 说，不做 full attention 而用各种办法削减 attention 计算，是因为 `attention is n squared ... where n is a sequence length and that gets really expensive`：

$$
\text{Attention cost} = O(n^2), \qquad n = \text{seq\_len}
$$

这也回扣了上一章由 `1000 bytes → ~250 tokens` 引出的长度压缩动机。要特别小心的是，$n$ 指的是序列长度 `seq_len`，不是 `batch_size`，更不是参数量；画面上输入形状把三者并列写出，恰恰是最容易混淆的地方。

画面 `Refinements:` 下列出七条，转写只是按类别口头带过，具体技术名与论文引用全在幻灯片上。激活函数从 ReLU 走到 SwiGLU，标注 `[Shazeer 2020]`；位置编码（positional encodings）从 sinusoidal 到 RoPE，标注 `[Su+ 2021]`；归一化一栏列出 LayerNorm、RMSNorm、QK norm 以及 pre-norm versus post-norm，附 `[Ba+ 2016]`、`[Dehghani+ 2023]`、`[Xiong+ 2020]`；attention 一栏是 full、sparse/local attention、group-query attention (GQA)、multi-head latent attention，附 `[Child+ 2019]`、`[Ainslie+ 2023]`、`[DeepSeek-AI+ 2024]`；循环／状态空间模型／线性 attention 一脉点名 Mamba 与 Gated DeltaNet，附 `[Katharopoulos+ 20..]`、`[Dao+ 2024]`、`[Yang+ 2024]`、`[Lahoti+ 2026]`；MLP 从 dense 变为 mixture of experts，附 `[Shazeer+ 2017]`、`[Fedus+ 2021]`；最后一条被黄色高亮，是 `Shape (hidden dimension, depth, number of heads, number of experts)`。转写与此一一对应：[30:31] 激活函数的演进、[30:31] 位置编码的演进、归一化以及如何避免各层 `blow up`（爆炸）、[30:40] 起削减 $O(n^2)$ attention 的多种思路、[31:00] 起 state space models 或等价的 linear attention 及 Mamba、gated DeltaNet，并说 `usually some hybrid between these models and attention seems to work quite well`（这类模型与 attention 的某种混合往往效果很好）、[31:19] 起原始 Transformer 的 dense MLP 如今被 mixture of experts 取代，成为构建 compute efficient transformer 的主导范式，以及 [31:53] 起那个「也许有点无聊但重要」的问题：多少层、多少头、hidden dimension 多大、多少专家。

形状参数之所以值得单列，是因为 [32:00] 讲师说它在 scaling laws（缩放定律）语境下 `has a huge huge implication`（影响极大），尽管表面上 `It's a hyperparameter`、看起来 `almost trivial`。可以把它写成一个必须由学生自己填的配置向量：

$$
\theta_{\text{shape}} = (\text{hidden dimension},\ \text{depth},\ \text{number of heads},\ \text{number of experts})
$$

这组数既是作业一里要落地的具体取值，也是后面缩放定律章节要系统化的对象；本章出现的引用年份跨度从 2016 到 2026、7 条 refinements、以及 $n^2$ 这个复杂度，都是可核对的具体数字与参数。易错点是把这份清单当成「逐项替换就能变好」的配方——讲师明说这些只是 `types of things that one might have to think about`（需要考虑的方面类型），而且 [30:15] 已把细节交给 Tatsu 后续讲解，本段只是概览。另一个易错点是把 MoE 仅仅视作架构改动：转写 [31:37] 特意补了一句 `it's not just defining your architecture, but we'll also need different techniques for training the model`，这句话直接预告了下一章的训练设计。局限也很清楚：这一节只给方向与文献，不给任何具体数值配置，真正的取值要等到缩放定律部分才有原则可循。

## 训练设计、作业一与三条平衡原则 [32:17 - 35:53]

[32:17] 的问题是 `once you define your model architecture, how do you train the model?`，[32:22] 随即指出 `there's a bunch of design decisions around the loss function`。画面 `Training` 一节以 `How do you set the parameters of the model?` 领起，列出八条 bullet，每条都带具体例子与论文引用，其中绝大多数引用转写没有念到：`Loss function (e.g., multi-token prediction) [Gloeckle+ 2024][DeepSeek-AI+ 2024]`、`Optimizer (e.g., AdamW, SOAP, Muon) [Kingma+ 2014][Loshchilov+ 2017][Vyas+ 2024]`、`Initialization scale (e.g., Xavier init, muP) [Glorot+ 2010][Yang+ 2022]`、`Learning rate schedule (e.g., cosine, WSD) [Loshchilov+ 2016][Hu+ 2024]`、`Regularization (e.g., dropout, weight decay)`、`Batch size (e.g., critical batch size) [McCandlish+ 2018]`，以及黄色高亮的 `MoE specific: load balancing (e.g., aux-free) [Wang+ 2024][DeepSeek-AI+ 2024]`。转写对应得更口语：默认的 next-token prediction 之外，`predicting more than one token`（一次预测多个 token）被发现有助提升模型；优化器过去普遍用 Adam，如今 Muon 用得越来越多，尤其是 Kimi K2 这类最新开放模型；initialization（初始化）听着无聊，却对大模型的训练稳定性影响巨大；再往下是 learning rates schedule、regularization、batch size，以及 MoE 特有的技术。把这份清单记成一个待定集合：

$$
\Omega = \{\text{loss},\ \text{optimizer},\ \text{init scale},\ \text{LR schedule},\ \text{regularization},\ \text{batch size},\ \text{MoE load balancing}\}
$$

[33:17] 是本节的分量所在。讲师预判学生看到这份清单会想 `these are just hyperparameters. I'm going to try a bunch of different options out`（不过是一堆超参数，多试几组就行），随即反驳：`really being very careful about setting these hyperparameters in a principled way will make the difference between a run that just blows up and is useless between a run that is achieving state of the art`（是否有原则地设定它们，决定了一次训练是崩溃无用，还是达到 state of the art）。这里的关键词是 `principled`（有原则的、系统化的），而不是「多试」。数字与参数包括 8 条设计决策、2010 至 2024 的引用年份、Kimi K2、Adam 到 Muon 的更替；易错点正是把网格搜索当解决方案——在语言模型这个规模下，一次失败的训练代价极高，谈不上随便试。讲师自己也接了句 [33:41] `I'll come back to that point when we talk about scaling laws`，把系统化设定超参数的方法明确挂到缩放定律章节。

作业一（Assignment 1 (basics)）在 [33:41] 之后落地，画面五条 bullet 比转写的大意精确得多：`[GitHub][PDF]` 两个链接标注、`Implement BPE tokenizer`、`Implement Transformer, cross-entropy loss, AdamW optimizer, training loop`、`Do resource accounting`、`Train on TinyStories and OpenWebText`，以及 `Leaderboard: minimize OpenWebText perplexity given 45 minutes on a B200 [last year's le...]`。转写只说要实现 BPE tokenizer、Transformer、loss function、optimizer 和整个训练栈，要做 resource accounting 以理解 `where your flops are going`，要在 tiny stories 与 open web text 这类数据集上训练，并通过排行榜 `drive down perplexity as fast as you can`，还类比了 NanoGPT speedruns。画面把 `loss function` 具体成 `cross-entropy loss`、把 `optimizer` 具体成 `AdamW`，更给出了转写完全没有的具体预算：

$$
\min \ \text{perplexity}(\text{OpenWebText}) \quad \text{s.t.} \quad t_{\text{train}} \le 45\ \text{minutes on a B200}
$$

[34:24] 讲师说，到作业一结束时，你应该能 `walk away and build a language model from scratch`（合上电脑也能从零造出一个语言模型），并称这 `very exciting`。这里要留一个心：画面在 `45 minutes on a B200` 后标注了 `[last year's le...]`，说明这是上一届的排行榜设定，今年可能调整，不能直接当成本届硬性规则。

[34:32] 起是全章的高层总结，画面黄色高亮 `High-level principle: everything is about balancing the following:`，下面三条为 `Expressivity (can represent complex dependencies in the data)`、`Stability (keep parameter and gradient norms in goldilocks zone)`、`Efficiency (runs fast on hardware, both training and inference)`。转写 [34:40] 说，tokenizer、modeling 与 training 虽然被讲成三个独立块，实际都在平衡同一组目标；[35:06] 要把 parameter 与 gradient norms 保持在 `Goldilocks zone`（金发姑娘区间，即不爆炸也不消失的适中区间），并断言 `a lot of training language models is about just stability`（训练语言模型很大程度上就是稳定性问题）；[35:21] 说 efficiency 相对直白，就是让它在硬件上跑得快，但紧接着举了一个典型两难：架构上把表示投影到低维空间可以更快，可 `does it work as well?`（效果还一样好吗？），`making those tradeoffs is sort of the name of the game`（做这类权衡才是这门课的核心）。用符号写，三条目标相互牵制：

$$
\text{Expressivity} \;\longleftrightarrow\; \text{Stability} \;\longleftrightarrow\; \text{Efficiency}, \qquad \|\theta\|,\ \|\nabla_\theta \mathcal{L}\| \in \text{Goldilocks zone}
$$

易错点是把这三条当成并列的检查清单：它们其实彼此冲突，提速可能损失表达力，增强表达力又可能破坏稳定性；把 $\|\theta\|$ 与 $\|\nabla_\theta \mathcal{L}\|$ 写成「落在适中区间」，是对 `goldilocks zone` 的数学转述而非讲师给出的新公式。关系上，[35:53] 一句 `as in assignment two, we're going to dive more deeply into systems` 把话题交给下一章的系统模块；而本章的 `Efficiency` 与 `45 minutes on a B200`，正是下一章 `minimize data movement` 的直接前奏。

![Training 一节的八条设计决策与论文引用，含 MoE specific 的 load balancing 高亮项](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0027_00-33-18.jpg)

## 系统模块：资源核算、硬件与 kernel [35:53 - 39:40]

[35:53] 进入 assignment two 对应的 systems（系统）部分。画面 `def systems():` 下写着 `Goal: squeeze the most out of the hardware (GPU or TPU)`（把硬件榨到极致），上一行被黄色高亮的 `Components: kernels, parallelism, inference` 给出三个组成：kernel、多 GPU 并行、推理。转写 [36:02] 的说法一致：`the goal here is just to get most out of your hardware. So we're going to talk about kernels, how you parallelize across multiple GPUs, and how you do inference.` 随后 [36:12] 讲师说 basics 其实从下一讲才开始，先做 `resource accounting`（资源核算），即 `keeping track of all where all the flops go. And where all the memory is being spent`（弄清 FLOPs 与内存都花在哪）。画面给出了精确代码与结果，而转写只念到 `Well, it's 6 * n * d roughly`：

```python
total_flops = 6 * 70e9 * 1e12  # Training 70B parameters on 1T tokens = 4.2e23 FLOPs
```

写成通式与代入值：

$$
\text{FLOPs} \approx 6 N D, \qquad N = 70\text{e}9,\ D = 1\text{e}12,\ \text{FLOPs} = 4.2\text{e}23
$$

这里有一处必须辨明的不一致：转写 [36:44] 记作 `training a 7B on model on 1 trillion tokens`，而画面注释明确是 `70B parameters`、`70e9`、`1e12` 与 `4.2e23 FLOPs`；用 $6 \times 70\times10^9 \times 10^{12} = 4.2\times10^{23}$ 验算恰好自洽，因此以画面为准应读作 70B。这是转写听写与幻灯片冲突时以画面为准的典型例子，也是本次逐张看图才能拿到的关键信息。[36:51] 讲师留了句 `and where does that come from?`，说明公式来历要到下一讲拆解。

[36:53] 之后是硬件。画面（下图）是一个上下两层的示意图：上方粉色 `Compute` 框里排着两行小方块代表计算单元，下方蓝色 `Memory` 框，中间一根绿色竖直连线表示数据通路；配文 `Model parameters must be moved from memory (HBM) to the compute (SMs)`，黄色高亮 `Example: B200 can perform 2.25 PFLOP/sec (bf16) with 8TB/sec memory bandwidth`，下一行灰色 `Roofline analysis: understand whether we're compute-bound or memory-bound`。转写 [37:04] 把这张图讲成 `your memory is not where your compute is`（内存与计算不在同一处），必须把参数或激活从内存搬到计算单元、算完再搬回，而 `that often is the bottleneck`（这往往就是瓶颈）；[37:20] 给出 B200 在 BF16 下 `2.25 petaflops per second`、内存带宽 `8 terabytes a second`；[37:49] 说 roofline analysis（屋顶线分析）用于判断一个计算受算力还是内存限制，并补一句 `In general, it is memory`（一般是内存受限）。把两个硬件数字相除，可得该设备的算力／带宽平衡点约为

$$
\frac{2.25\ \text{PFLOP/s}}{8\ \text{TB/s}} \approx 281 \ \text{FLOP/byte}
$$

即每搬 1 字节数据至少要配约 281 次浮点运算才不至于闲等内存——这是依画面数字相除得到的推算，并非讲师原话，却正好解释了 roofline 拐点的来历，也 [37:39] 预告了下一讲要用这些数字 `do some calculations` 估算不同算法耗时。画面里的 `HBM`（高带宽内存）与 `SMs`（流式多处理器）两个缩写，转写中同样没有出现。

![Basics 的 6ND 资源核算代码与 Compute/Memory 示意图，含 B200 的 2.25 PFLOP/sec 与 8TB/sec 参数](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0030_00-37-33.jpg)

[38:06] 起展示 DGX B200：转写说 `You have eight GPUs. They're connected via NV link`，上千 GPU 时则由多台这样的机器经 InfiniBand 或 Ethernet 相连。画面上方是一张 8 GPU 的机架拓扑图，GPU 小方块标着 GPU 1 到 7（可见 7 个，加上裁切处共 8 个），下方图例列出 `BlueField-3`、`ConnectX-7 Network Module`、`NVMe`、`PCIe Switches`、`NVSwitch`、`100 GbE`、`CPU communication`——这些组件名转写里一个都没有。[38:29] 转入 Kernels，画面三条 bullet 为 `Kernel is a function that runs on GPU`、`When using PyTorch, each primitive operation launches a standard kernel`、黄色高亮的 `Can write custom kernels to make GPUs go brrr`；转写 [38:39] 解释，用普通 PyTorch 时每个原语其实都对应启动某个内置 kernel，所以 `you're already using kernels whether you know it or not`，但某些计算可以写自定义 kernel 让 GPU 更快，而 `the main principle here is organizing the compute to minimize data movement`（核心原则是组织计算以最小化数据搬运），这正呼应前面那张 Compute/Memory 图。

本段最关键的内容落在画面灰色小字里，而转写恰好在结论前被截断。[39:24] 讲师只讲到 `suppose you wanted to compute A and B`，朴素做法要从 HBM 读、算、写回，再读、算、写回，`you're basically sending the data back and forth twice, and uh`——句子在 39:36 断掉。画面把结论补全了：

```text
Naive: read HBM; compute A; write HBM; read HBM; compute B; write HBM
Fused: read HBM; compute A and B; write HBM
```

以及 `Strategies: operator fusion (matmul + activation), tiling (FlashAttention)` 和 `Warp divergence, memory coalescing, bank conflicts, occupancy, bulk-async memory transfer`。也就是说，算子融合（operator fusion）把两次往返压成一次，用访存次数写就是：

$$
\text{Naive}: 2\ \text{reads} + 2\ \text{writes} \;\longrightarrow\; \text{Fused}: 1\ \text{read} + 1\ \text{write}
$$

`matmul + activation` 的融合、tiling（分块，以 FlashAttention 为例）都是同一思想的变体，而 warp divergence（线程束分化）、memory coalescing（访存合并）、bank conflicts（存储体冲突）、occupancy（占用率）、bulk-async memory transfer（批量异步内存传输）则是写 kernel 时才会碰到的具体细节。易错点有二：别把 kernel 当成某种高深的新机制，用普通 PyTorch 时你已经在用内置 kernel，自定义 kernel 的价值只在于为特定计算重排数据搬运；也别把「融合」理解成改变了数学结果，它改变的是访存调度，A 与 B 的语义不变。局限同样明确：本段是概览，[37:39] 讲师说 `we're going to break this down and use this information to do some calculations` 要留到下一讲，[38:01] 的 benchmarking 与 profiling 也只是点到为止；把这一章当成 roofline 或 kernel 编程的完整教程，会低估后续课程的深度。

![DGX B200 的 8 GPU 拓扑图与 Kernels 一节，含 naive 与 fused 两种 HBM 访存流程的对比](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0031_00-38-58.jpg)

## 算子融合、GPU 微架构细节与多 GPU 并行 [39:40 - 41:39]

本段开头承接系统单元反复强调的**最小化数据移动（minimize data movement）**原则，把它从单卡推进到 kernel 与多卡两个层面。所谓 `kernel`，幻灯片给出的定义是 "a function that runs on GPU"——运行在 GPU 上的函数；即使只写普通 PyTorch，也是 `each primitive operation launches a standard kernel`，每个原语操作都会启动一个标准 kernel。要榨干硬件，就可以写自定义 kernel，幻灯片戏称其为 "make GPUs go brrr"。**operator fusion（算子融合）**的直觉很朴素：把两次计算合并成「读一次、算两次、写回一次」。讲师在转写里说得同样直白：there's this idea called fusion where you read it once, do both of the computations and you write it back, and that will save you a lot of time；而 **tiling（分块）**被他称作「同一思想更复杂的变体」，幻灯片则把它与 FlashAttention 绑定在一起。讲义里被高亮的一行还列出一串 GPU 特有细节：`warp divergence, memory coalescing, bank conflicts, occupancy, bulk-async memory transfers`（线程束发散、内存合并访问、bank 冲突、占用率、批量异步内存传输），并给出实现栈 `CUDA/Triton/CUTLASS/ThunderKittens`。这些正是讲师所谓 "GPUs have also gotten a lot more complicated"、需要让大家 "appreciation for the types of things that one has to consider in order to squeeze as much juice out of them as possible" 的具体内容，课程后面还会真的动手写一些 kernel。

把融合写成可对照的伪代码，差别一目了然（以下为幻灯片原文）：

```text
Naive: read HBM; compute A; write HBM; read HBM; compute B; write HBM
Fused: read HBM; compute A and B; write HBM
```

朴素路径对 HBM（High Bandwidth Memory，高带宽显存）做了 2 次读、2 次写，共 4 次往返；融合后只剩 1 次读、1 次写，共 2 次往返。幻灯片紧接着写 `Strategies: operator fusion (matmul + activation), tiling (FlashAttention)`，说明典型做法是把矩阵乘与紧随其后的激活函数融掉。这里的「具体数字」不是浮点算力指标，而是数据搬运次数从 4 降到 2，正是省时的来源；`Principle: organize computation to minimize data movement` 这条总纲也再次出现。讲师同时提醒，GPU 的这些复杂细节「not sure how many of these details we'll have time to get into」，属于暴露式介绍而非完整覆盖。

当规模从单卡扩到 thousands of GPUs 时，原则不变，代价却更高：`Data movement between GPUs is even slower`，幻灯片干脆把设问写成 **"What if we have 1024 GPUs?"**。此时要用经典的**集合通信操作（collective operations）**来组织分布式训练，转写点名 `gather`、`reduce`、`all-reduce` 是 "the way to kind of think about basically distributed training"。需要被分片（shard）到多卡上的状态有四类：`parameters, activations, gradients, optimizer states`（模型参数、激活、梯度、优化器状态），并且必须 "bring the right data to the right nodes to make the compute and write it back"，即把正确的数据送到正确的节点、算完再写回，这一整套编排（orchestration）正是本单元的主题。切分维度被高亮地归纳成

$$\{\text{data},\ \text{tensor},\ \text{pipeline},\ \text{sequence},\ \text{expert}\}\ \text{parallelism}$$

转写里对应的口语说法是 "splitting up your data, splitting up your model, splitting up different layers in the model, splitting up the sequences, splitting up between experts"，并强调每种切法 "we'll talk about the tradeoffs that come with each of these"。`1024` 这个具体数字把「上千 GPU」落到一个可想象的量级，而 `8` 卡一台 DGX、`thousands` 卡跨机器的层级差异，正是通信开销被放大的原因。

与前后内容的关系上，这一章是作业二（systems）中 kernel 与并行两块的理论收束，也为下一章的推理埋下伏笔：推理之所以难，恰恰因为它同样受内存带宽而非算力约束。易错点至少有三处。其一，融合不是把两段代码写在一起就自动生效，`warp divergence`、`bank conflicts`、`occupancy` 这些微架构细节会让理论上的 2 次 HBM 往返收益打折，必须靠 benchmark 与 profiling 验证。其二，不能把单卡结论直接外推到多卡：跨 GPU 的数据移动比片内 HBM 更贵，通信常常反过来主导设计。其三，分片维度有五种且各有权衡，不存在一种在所有场景下都最优的切法；讲师也坦言不一定有时间深入这些细节，所以这一章给出的是思考框架而非完备配方。

![Kernel 幻灯片：自定义 kernel 定义、Naive 与 Fused 的 HBM 读写对比、operator fusion 与 tiling 策略，以及高亮的 warp divergence 等 GPU 细节](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0032_00-40-17.jpg)

## 推理的两阶段、投机解码与作业二 [41:39 - 45:09]

**inference（推理）**的目标，幻灯片一句话概括为 "generate tokens given a prompt (needed to actually use models!)"——给定 prompt 生成 token，也就是真正把模型用起来。讲师特意纠正「推理只是聊天」的窄化理解：除了与模型对话，它还用于强化学习（reinforcement learning）的 rollout、test-time compute（测试时计算）、合成数据生成（generating synthetic data）以及 evaluation（评测），因此 "inference is a very critical part of what it means to do language modeling work"，幻灯片也写着 `Inference is also needed for reinforcement learning, test-time compute, evaluation`。他坦承课程排得很满、不会像希望的那样花太多时间，甚至讨论过是否要让学生 "write inference from scratch" 但尚未定案。随后给出核心抽象：推理分两个阶段，**prefill（预填充）**与 **decode（解码）**。prefill 拿到 prompt，把所有 token 一次性前向计算、建立 key-value 对（KV pairs），"very much like what happens in training"；decode 则一次只生成一个 token，很快变成 memory-bound（内存受限），"this is why inference is hard"。

幻灯片把两阶段画成一张流程图，信息量比转写更大。左半黄色区域是 `Prefill Phase`，右半绿色区域是 `Decoding Phase`，顶部一条红色横条贯穿两者、标注 `KV-Cache`；输入框里是 `"Computer science is"`，经过 `Iteration 1` 生成菱形 `a`，再到 `Iteration 2` 生成 `discipline`、`Iteration 3` 生成 `.`、`Iteration 4` 生成 `<EOS>`，每个后续 iteration 都有箭头与 `KV-Cache` 双向相连，表示读取缓存并把新 key/value 写回。这段画面把「2 个阶段、4 次迭代、一次一个 token」变成可数的具体实例。图下的原文 bullet 是：

```text
Prefill (similar to training): tokens are given, can process all at once (compute-bound)
Decode: need to generate one token at a time (memory-bound)
```

即 prefill 是 compute-bound（算力受限），decode 是 memory-bound。所谓「具体数字或参数」在这里体现为：2 个阶段、示例里 4 次 iteration、每步仅产出 1 个新 token，而 KV-Cache 随步数累积、每步都要重读一遍；这正是 decode 阶段「很快受内存限制」的结构性原因，也解释了为什么推理优化几乎都围绕它展开。

围绕 decode 的内存瓶颈，幻灯片列出 `Methods to speed up decoding`。第一类是「用更便宜的模型」：`Use cheaper model (via model pruning, quantization, distillation)`，即剪枝、量化、蒸馏。第二类是 **speculative decoding（投机解码）**，原文为 `use a cheaper "draft" model to generate multiple tokens, then use the full [model to] score in parallel (exact decoding!)`：让便宜的 draft 模型先跑在前面猜出一串 token，再用完整模型并行地对这些 token 打分，若猜中就一次性接受全部 token，转写里说 "you're much faster than if you're doing one token at a time"；括号里的 `exact decoding!` 强调输出分布并非近似。第三类是系统层：`Systems optimizations: fused kernels, continuous batching`（融合 kernel、连续批处理），这一行被高亮。讲师还补充了服务场景特有的难题：查询 "coming at you at potentially different times"，必须现场决定如何把它们 batch 起来；而训练里 batch 是预先定义好的，"everything is much more predictable"。

落到作业二（Assignment 2 (systems)，标注 `[GitHub][PDF from Spring 2025]`），画面给出的具体条目是 `Implement a fused RMSNorm kernel in Triton`、`Implement distributed data parallel training`、`Implement optimizer state sharding`、`Benchmark and profile the implementations`。转写只说了 "implementing kernels in Triton" 和 "some sort of parallel training"，并提醒细节可能因助教有 "grand plans of revamping the systems" 而变化、但会覆盖大致相同的材料。同页还推荐了 Google 团队的书 **《How to Scale Your Model》**：`Nicely lays out how to approach systems for LLMs conceptually`，`From Google, so it foregrounds TPUs, but high-level concepts are similar`；讲师补充它适合建立 roofline analysis 与 transformer math 的概念理解，而且 "now they have a new chapter on how to think about GPUs"。

与前后内容的关系：作业二把上一章的 kernel 融合与多卡分片变成可提交的代码，而推理又是缩放定律、数据、对齐各单元都要反复调用的能力（评测、合成数据、RL rollout 都依赖它）。易错点包括：把 speculative decoding 误当成有损近似（幻灯片明确写 `exact decoding!`）；只盯着 prefill 的算力而忽略 decode 的内存带宽才是瓶颈；把训练里固定 batch 的直觉套到 continuous batching 上；以及把作业二当成固定不变的清单——转写两次强调它可能改版，真正要掌握的是 benchmark、profile 与最小化数据移动的思路。此外，讲师对「是否从零写 inference」的犹豫本身也说明，这部分在本课程中是概览而非完整实现。

![Inference 幻灯片流程图：黄色 Prefill Phase 与绿色 Decoding Phase，顶部红色 KV-Cache，四次 Iteration 依次生成 a、discipline、.、EOS](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0034_00-42-46.jpg)

## 缩放定律的设定：从「训练一个模型」到「缩放配方」 [45:09 - 49:16]

第三单元（作业三）以一个高压设定开场：假如你有 $10^{25}$（即 `1e25`）FLOPs 算力——讲师说这是 "tens of millions of dollars of compute"（数千万美元级）——你会训练什么模型？幻灯片把它写成可执行讲义里的函数与说明文字：

```text
Setting: if you had 1e25 FLOPs of compute, what hyperparameters would you use to train a good model
Too expensive to do hyperparameter tuning at full scale!
```

难点在于：这个规模下搞砸了就是 "a lot of money down the drain"，而且 "you only get to train one model"，无法像小实验那样做典型的 hyperparameter tuning。讲师强调，这是大规模 LLM 训练特有、微调或小规模实验不会遇到的问题："you don't have to really deal with if you're just fine-tuning a model or doing small scale stuff"。于是出现**关键概念转变（Key conceptual shift）**，高亮句写作 "instead of a single scale, think of a **scaling recipe**（缩放配方）"。它被定义为一个映射，从 FLOPs 预算（如 `1e25` 或 `1e24`）映到一组超参数、本质上就是一个 config 文件：

$$\text{scaling recipe}:\ \text{FLOPs}\ \longrightarrow\ \text{hyperparameters\ (config)}$$

对一个给定配方，做法是 "run a bunch of experiments to compute the loss at smaller scales, and then you fit a scaling law, and that enables you to predict the loss at a target scale"。幻灯片把两步写得更具体：`Run experiments to compute the loss at various smaller scales (e.g., up to 1e24 FLOPs)`、`Fit a scaling law to predict the loss of the scaling recipe at the target scale (e.g., 1e25 FLOPs)`。这里 `1e24` 与 `1e25` 相差一个数量级，正是「小规模实验 → 外推到目标规模」的具体跨度。由此得到两个能力（Now you can）：`1. Optimize the scaling recipe targeting a larger scale using smaller scale experiments`、`2. Predict the loss at the target scale before actually running the experiment!`。讲师半开玩笑地指出第二点的现实用途：拿小规模结果去融资，"I think I can get a really like a GPT-5 level model. Please give me a lot of money so I can train that model"。

要让外推成立，配方必须**可外推（extrapolate）**。讲师点名一个 misconception：缩放定律不是自然规律（"scaling laws are not laws of nature"），"they don't just happen automatically"，幻灯片高亮句写作 `Scaling laws don't happen automatically, they require careful construction of a scaling recipe.`。具体来说，配方要规定一串超参数随规模怎么变：学习率（learning rate）保持不变还是下降、batch size 增长多少（转写原话 "maybe the learning rate is a constant, maybe it drops, maybe the batch size increases by how much"）。这就引出 **hyperparameter transfer（超参数迁移）** `[Yang+ 2022]`：`Parameterize the model in a way to get hyperparameter transfer`——小规模用的超参数要么能直接用于大规模，要么是规模的可预测函数。反面例子很直观：若每个规模上学习率 "sometimes 1e-5 and sometimes 1e-4" 地乱跳，到了大规模就 "not going to be able to magically guess the right learning rate"。因此幻灯片高亮结论 `Predictability is at least as important as optimality!`：可预测性至少和最优性一样重要。讲师补充，通常大家只想 hyperparameter tune 求最优，但在只能训练一次的场景里，缺乏可预测性会直接导致大模型翻车。

这一章与前后的衔接很清晰：它承接系统单元「模型已经能跑得很快」，回答 "now you want to scale up. So, how do you scale up?"；同时是下一章经典 compute-optimal 定律的 stage setting（转写原话 "that's this is kind of a stage setting"）。顺带一提，同一张讲义上方还写着 `Thanks to Modal for providing compute` 与 `Please read the guide on how to access and use the compute`，并给出整门课的函数骨架：

```python
def course_syllabus():
    basics()          # Assignment 1: tokenization, model architecture, training
    systems()         # Assignment 2: kernels, parallelism, inference
    scaling_laws()    # Assignment 3: scaling laws
    data()            # Assignment 4: evaluation, curation, transformation, filtering, deduplication
    alignment()       # Assignment 5: RLHF, RL algorithms, RL systems
```

`scaling_laws()` 这一行被高亮，正对应本单元；旁边还重申 `Remember it's all about efficiency`，资源被列为 `data + hardware (compute, memory, communication bandwidth)`，目标句是 "How do you train the best model given a fixed set of resources?"。易错点在于：把 scaling law 当成拟合出来就必然成立的物理定律，忽略配方设计；只追最优超参而不顾跨规模可预测；把 `1e24` 小实验的结论不加检验地外推到 `1e25`；以及误以为可以在目标规模上补做调参——`Too expensive to do hyperparameter tuning at full scale!` 正是要打破的幻觉。

![缩放配方幻灯片：def scaling_laws() 设定 1e25 FLOPs、小规模 1e24 FLOPs 实验、两个 Now you can，以及高亮的 Scaling laws don't happen automatically](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0038_00-47-44.jpg)

## 计算最优缩放定律、经验规则与 Marin 预注册 [49:16 - 52:00]

进入「实际的缩放定律」，讲义把问题写成一个明确的选择题：`Question: given a FLOPs budget (C = 6 N D), use a bigger model (N) or train on more tokens (D)?`——给定算力预算，该把钱花在更大的模型 $N$ 上，还是更多的训练 token $D$ 上？其中

$$C = 6ND$$

$C$ 是 FLOPs 预算，$N$ 是模型参数量，$D$ 是训练 token 数；问题就是固定 $C$ 时如何在 $N$ 与 $D$ 之间分配。经典答案是 **compute-optimal scaling laws（计算最优缩放定律）**：`[Kaplan+ 2020][Hoffmann+ 2022]`，后者即俗称的 Chinchilla。方法是 `ISOFLOP curves: for multiple small FLOPs budgets, find optimal N`，再 `Then fit a scaling law to extrapolate to large FLOPs budgets`：对若干个较小的算力预算，各自扫描不同模型大小、找到 loss 最低的那个 $N$，然后拟合一条「预算 → 最优参数量」的曲线，外推到大预算。讲师在转写里把这个 sweep 的跨度说成从 `6e18` 一直到 `3e21` FLOPs，"for each flops budget ... you sweep across different model sizes and you choose the best one"，并提醒若这些最优点 "lie on a roughly on a line" 就幸运、可可靠预测，若 "all over the place" 就说明 "you should have no confidence that you're going to be able to predict reliably"。

图 kf_0040 把这三步画成并排的三张图，数字非常密集。左图纵轴 `Training Loss`（约 2.0 到 3.2）、横轴 `Parameters`（100M、300M、1B、3B、6B、30B），一组 U 形 ISOFLOP 曲线由图例 `6e18, 1e19, 3e19, 6e19, 1e20, 3e20, 6e20, 1e21, 3e21` 共九个 FLOPs 档位区分，每条曲线的最低点就是该预算下的最优 $N$。中图纵轴 `Parameters`（100M、1B、10B、100B、1T）、横轴 `FLOPs`（$10^{17}$ 到 $10^{25}$），黑点沿一条红色虚线近似排成直线，图上另可见标注 `63B` 的青色水平参考线与约 `1e23` 处的竖线，用来演示「给定预算读回参数量」。右图纵轴 `Tokens`（100M、1B、10B、100B、1T、10T）、横轴同样是 `FLOPs`，有一条标注 `1.4T` 的水平线。图下的高亮结论是 `TL;DR: D = 20 N is roughly optimal (e.g., 70B parameter model should be trained on ~1.4T tokens)`，紧接着一行 `Caveat: this doesn't take into account inference costs (want a smaller model)`。转写把这条 rule of thumb 说成 "20 times the number of parameters is the number of data points you should train on"，70B 模型约需 1.4 trillion tokens，但也强调 "depending on the data set and architecture, this number will actually vary"，且它没有把推理成本算进去——如今很多模型故意做得更小、训练 token 远超计算最优，就是为了推理时更便宜。

![计算最优缩放定律三图：ISOFLOP 曲线（6e18–3e21）、参数量-算力拟合线与 Tokens-算力图，及 D = 20 N 规则](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0040_00-50-13.jpg)

接着是一个把方法「押上赌注」的例子：Marin 项目的 **pre-registering（预注册）**。幻灯片标题 `Live example from Marin [post]`，图表为 `Delphi Scaling Suite Results`，纵轴 `Paloma macro loss`（约 2.4 到 3.8）、横轴 `Training tokens`（$10^9$ 到 $10^{12}$）；图例按 `Compute (FLOPs)` 列出 `3e+18, 9e+18, 1.8e+19, 3e+19, 9e+19, 1.8e+20, 3e+20` 以及两个菱形点 `1e+21`、`1e+22`，另有 `Compute-Optimal Frontier` 灰色虚线与 `Forecast` 叉号标记。图上直接标出预测与前沿的差距 `Δloss = +0.011`、`Δloss = +0.005`，并在约 $10^{12}$ token 处写着 `Currently training!`。讲义下一行写着 `Should be done training this week, should see how well we match the preregistered loss!`。转写对应地说明：他们在不同算力预算拟合缩放图、把预测外推到 `1e22` FLOPs，这个 "which we've never trained before" 的大模型正在 Marin 站点上训练，若最终 loss 与预注册预测吻合，就 "really nice" 地验证了缩放定律的预测力；讲师还说 "maybe on Wednesday I'll report back on how we did"。

![Marin 项目 Delphi Scaling Suite 结果图：Paloma macro loss 对 Training tokens，标注 Forecast、Compute-Optimal Frontier、Δloss 与 Currently training](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0041_00-51-27.jpg)

与前后内容的关系：这一章把第三章抽象的 scaling recipe 落成可操作的经典实例——`C = 6ND` 的分配问题正是配方要回答的核心超参之一；同时为作业三提供要拟合的对象。易错点不少。其一，`D = 20N`、`70B → 1.4T` 只是讲师自称 "quite crude" 的粗规则，会随数据集和架构变化，不能当精确公式。其二，外推的前提是小预算最优点近似共线，讲师明确说若散乱则 "should have no confidence"。其三，compute-optimal 只优化训练算力、不含推理成本，部署视角下「更小模型 + 更多 token」往往才是理性选择。其四，预注册图上 `Δloss = +0.011`、`+0.005` 这类偏差提示预测并非完美，验证的是「能否提前估准」而非「必然命中」。

## 作业三：用训练 API 模拟高预算决策 [52:00 - 53:27]

作业三（Assignment 3 (scaling laws)，标注 `[GitHub][PDF from Spring 2025]`）把前面两章的方法变成一次可评分的实战。核心机制被高亮地定义为 `We define a training API (hyperparameters → loss) based on previous runs`——课程基于离线跑过的历史结果，给出一个「超参数进、loss 出」的接口：

$$\text{hyperparameters}\ \longrightarrow\ \text{loss}$$

转写解释了为什么需要这层「伪装」：真实算力 "we don't have enough compute to actually allow everyone to train their own, you know, 8B model or anything"，所以 "we train a bunch of models offline, and we basically provide this cache"，让它 "it looks like you're training"。换句话说，学生面对的并不是一台可以随意启动训练的集群，而是一张由历史实验填好的查找表；讲师把这套机制称为 "simulate what happens if you could do a lot of training runs"，用来在课堂里复现「能跑很多次实验」的假象。学生能做的就是 `Submit "training jobs" (under a FLOPs budget) and gather data points`，即提交带 config 的训练任务换回一个 loss，转写里说 "you give us a config, basically we give you a loss back. And then you can do whatever you want"。推荐做法是 `Fit scaling laws to the data points`（对数据点拟合缩放定律）、`Submit extrapolated hyperparameters and loss predictions`（提交外推后的超参与 loss 预测），最后 `Leaderboard: minimize loss given FLOPs budget`——在给定 FLOPs 预算下比谁的 loss 更低。把这几条原样列出：

```text
Submit "training jobs" (under a FLOPs budget) and gather data points
Fit scaling laws to the data points
Submit extrapolated hyperparameters and loss predictions
Leaderboard: minimize loss given FLOPs budget
```

具体数字与目标都指向「高压力」。转写里说这想复现的是 "if you actually had a budget that like, you know, $100 million that you needed to spend"，必须非常小心怎么花这笔钱；`8B` 是学生无法各自训练的模型量级，而 `1e22` FLOPs 是上一章 Marin 外推的目标规模。讲师也立刻补了一句 "of course, this is low low stakes"：作业里猜错只是分数，真实场景里猜错就是数千万美元。流程被概括为 "fit scaling laws to these points, and then you extrapolate, and then we give you a budget, and you're ... evaluate how well your model landed"——先拟合、再外推、再给预算、最后看落点；他还说这 "is a kind of either fun or ... let's just say it's a fun assignment"。

与前后内容的关系：它把第三章的「可预测性至少和最优性一样重要」变成可量化的排行榜目标，也把第四章的 ISOFLOP 外推搬进课堂；紧接着 `def data():` 出现，课程转入「用什么训练」的问题。易错点有三。其一，cache 来自 `previous runs`，插值范围内相对可靠，超出观测规模的外推仍可能失准，而这正是要检验的能力。其二，别把「提交尽可能多 job」当成策略——转写强调在真实设定下 "you only get to train one model"，作业是在模拟稀缺预算下的决策，而非无限调参。其三，`Leaderboard: minimize loss given FLOPs budget` 的约束是预算固定，脱离预算谈 loss 没有意义；把它误读成「随便搜超参」就丢掉了缩放定律的本意，也忽略了讲师反复强调的 careful construction of a scaling recipe。

![Assignment 3 (scaling laws) 幻灯片：高亮的 training API (hyperparameters → loss)，以及提交训练任务、拟合缩放定律、外推预测与排行榜条目](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0042_00-52-41.jpg)

## 评测：内部指标与外部指标 [53:27 - 56:44]

数据单元的入口不是「去哪找数据」，而是先问能力：`def data():` 下面被高亮的一行是 `Question: What capabilities do we want the model to have?`，示例为 `Multilingual? Good at conversation? Agentic coding capabilities?`。讲师的论点是「data basically reflects what your model wants」（数据反映你希望模型做什么），数据质量 "basically specifies how good your model is going to be"，所以先讲 **evaluation（评测）**，因为 "evaluation basically defines the capabilities that you like your model to have"。他特意说评测是个 "fairly deep topic"，不只是 "running on some benchmarks"，而是要先想清楚目的。

评测被明确分成目的不同的两类，幻灯片原文：

```text
1. Internal: guide model development (smoothness across scales, relative performance matters)
2. External: measure absolute quality of a real use case (ecological validity matters)
```

**内部指标（internal）**用于模型开发，看重 `smoothness across scales`（跨规模平滑）与 `relative performance`（相对性能）；这直接呼应缩放定律单元对「可预测」的执念——记住 "we want things to be predictable"。讲师解释：你未必关心 `perplexity`（困惑度）的绝对数值，他反问在某个 held-out data 上算出的一个 perplexity 数值到底意味着什么，真正有用的是不同规模、不同配方之间的相对比较。相反，**外部指标（external）**面向客户、审稿人或任何你要汇报的对象，看重 `ecological validity`（生态效度），即真实用例里的绝对质量。幻灯片 `Examples of evaluations` 给出两条：`1. Perplexity: ideally run on private documents not on Internet (avoid contamination)`、`2. Advanced use cases: GPQA, HLE, SWE-Bench, Terminal-Bench`。前者对应讲师 "if you have some data that's not on the internet, that would be good because you can avoid contamination" 的建议，后者是 "more representative of external-facing use cases" 的高阶用例。高亮句 `LMs are general purpose, require a diverse set of evaluations!` 把「通用目的模型需要多样评测」点明；转写补充说可以把多个评测平均成单一数字，但 "often that average conflates a lot of different things"。

具体参数与名词在这里很密集：`perplexity` 是在 held-out data 上刻画模型内在质量的指标，讲师说 "still to this day perplexity is a very good way of capturing the intrinsic quality of a model without getting kind of worrying about bench maxing"（不必担心刷榜）；`GPQA`、`HLE`、`SWE-Bench`、`Terminal-Bench` 是四个被点名的高级 benchmark；`contamination`（数据污染）与 `ecological validity` 分别对应内外两类指标的主要风险。转写还提到 "internal evaluation metrics for model development" 与 "external metrics" 常被 conflated，但 "they really serve two distinct purposes"。

与前后内容的关系：评测是作业四（`data(): evaluation, curation, transformation, filtering, deduplication`）的第一步，也为下一章的数据整理给出目标函数——先定义「想要什么能力」，才知道该收什么数据。易错点包括：把内部与外部指标混为一谈，用刷榜思路去做开发期指标；用互联网上能查到的数据做评测而引入 contamination（幻灯片明确建议 private documents）；把多个评测简单平均，掩盖不同能力的差异；以及误以为 perplexity 已经过时——讲师仍把它视为抗 bench-maxing 的内在质量指标，只是不该拿它的绝对数值去对外汇报。

![Evaluation 幻灯片：高亮的 Internal（跨规模平滑、相对性能）与外部指标定义，以及 Perplexity、GPQA、HLE、SWE-Bench、Terminal-Bench 等评测示例](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0044_00-55-10.jpg)

## 数据整理、法律问题与数据处理流程 [56:44 - 59:31]

评测设定了目标，接下来才回答 "how do you get the data"。讲师开门见山："data does not just fall from the sky"，必须主动 curated；课堂和科研里常常是 "given a data set" 然后开做，但真实的大规模语料要自己去 "actively look at it"。来源包括 `web pages crawled from the internet`（从互联网抓取的网页）、`books`（他注明 "controversial at this point"）、`archive papers`、`GitHub code` 等。画面给出一张 2021 年 **The Pile** 的老图（treemap 树状图），用来说明语言建模数据集 "fairly diverse"：可辨认的分支包括 `PubMed Central`、`ArXiv`、`Pile-CC`、`PG-19`、`BC2`、`FreeLaw`、`USPTO`、`PMA`、`Phil`、`NIH`、`OpenWebText2`、`StackExchange`、`Wikipedia`、`Github`、`DM Math`、`Subtitles`、`IRC`、`EF`、`HN`、`YT`，蓝色、绿色、灰色、橙黄几大块对应不同来源族。讲师同时点出当下的争议：`Appeal to fair use to train on copyright data? [Henderson+ 2023]`（用版权数据训练是否属合理使用），以及 `Might have to license data (e.g., Google with Reddit data) [article]`（可能得授权）；转写里还举了具体困境——很多 GitHub 代码没有 license，是默认它 permissive，还是保守地当作不允许，需要自己判断，讲师说这些法律问题 "are quite important"。

![The Pile 数据集组成树状图与版权/许可证问题：Appeal to fair use、license data，以及 Raw data is HTML, PDF, directories 的说明](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0046_00-57-39.jpg)

一个容易被忽略的事实是「data is not even text」：`Raw data is HTML, PDF, directories (not text), requires processing`，原始数据是 HTML、PDF、代码目录，必须处理成可用文本，这就是 **data processing（数据处理）** 的主题。幻灯片把流程列成可逐条对照的原文：

```text
Transformation: convert HTML/PDF to text (extract main content)
Filtering: keep high quality data, remove harmful content (via classifiers)
Deduplication: save compute, avoid memorization; use Bloom Filters or MinHash
Data mixing: how much to upweight/downweight each source? [Liu+ 2024][Chen+ 2026]
Rewriting / synthetic data: use LM to augment real data, more similar to downstream tasks
```

即 transformation（转换，抽取正文）、filtering（过滤，靠 classifiers 保留高质量、去掉有害内容）、deduplication（去重，节省算力并避免记忆，工具是 `Bloom Filters` 或 `MinHash`）、data mixing（多来源如何 upweight/downweight，引用 `[Liu+ 2024][Chen+ 2026]`），以及近年活跃的 `Rewriting / synthetic data`（用 LM 改写/增广真实数据，使其更接近下游任务，转写里也提到可改写得 "more Wikipedia-like"）。讲师特别强调 Common Crawl 里随机抽一篇文档 "extremely bad"，通常不想拿它训练，因此过滤不是可选项。画面还给出数据分层：

```text
Pretraining data: large and diverse
Mid-training data: high quality, including long-context
Post-training data: supervised fine-tuning (conversations, agentic traces with tool calling)
```

转写对应地解释了 **mid-training（中期训练）**：放在预训练末尾的高质量数据，包含 long-context 数据，例如 "big code repositories or books"（大型代码库或书籍）；并说明合成数据可以是把真实数据改写得 "more like the downstream task" 或 "more Wikipedia-like"，属于 "an active area of research"。

与前后内容的关系：这一章是作业四的直接铺垫（`data(): evaluation, curation, transformation, filtering, deduplication`），也是全课从「怎么训」转向「训什么」的枢纽；它承接上一章「评测定义能力」的结论，把能力映射到语料配比。易错点相当具体：把 HTML/PDF 直接当文本喂进去，忽略 transformation 的 `extract main content`；跳过 filtering 让低质 Common Crawl 文档白白占用算力；不做 deduplication 导致模型 memorize；在许可证上默认 GitHub 代码是 permissive 而踩法律风险；把 data mixing 当成等权混合，忽略 `[Liu+ 2024][Chen+ 2026]` 所指向的 upweight/downweight 取舍；以及把 synthetic data 当成万能解——转写只把它定位为 "active area of research"，而讲师也提醒 raw data 的非文本形态本身就是一整块必须处理的工程量。

## 数据类型的收束与 Assignment 4 的具体要求 [59:31 - 60:19]

这一段从课程「数据（data）」单元的最后一种数据类型讲起。讲师在 [59:31] 说 "And then finally there's post-training data"，即后训练数据（post-training data），例子是 conversations（对话）以及带工具调用的 agentic traces（转写把 agentic 误记为 genetic，幻灯片原文是 "agentic traces with tool calling"）。画面上把数据分成三类并给出了各自定位：Pretraining data: large and diverse（预训练数据，量大且多样）；Mid-training data: high quality, including long-context（中期训练数据，高质量、含长上下文）；Post-training data: supervised fine-tuning (conversations, agentic traces with tool calling)（后训练数据，用于监督微调，含对话与带工具调用的 agent 轨迹）。这一分类与本段之前的数据模块形成闭环：前面讲过的「转换、过滤、去重、混合、合成数据」每一步都不是孤立技巧，而是由数据类型决定目标、再由评测指标检验结果；post-training data 之所以单列，正是因为它对应的是对话与 agent 行为，而不是预训练那种大而杂的网页文本。

紧接着幻灯片进入 "Assignment 4 (data)"，标题下还挂着 [GitHub][PDF from Spring 2025] 两个链接，说明作业要求沿用 2025 春季版并公开在仓库里。四条要求在画面上列得很清楚，而转写只笼统地说 "filter and to dedupe and make the data uh clean"，把大量细节省略了：

```text
Assignment 4 (data)
  • Convert Common Crawl HTML to text
  • Train classifiers to filter for quality and harmful content
  • Deduplication using MinHash
  • Leaderboard: minimize perplexity given token budget
```

第一条 "Convert Common Crawl HTML to text" 意味着起点根本不是干净文本，而是 Common Crawl 的原始 HTML，学生要自己写抽取与清洗；第二条 "Train classifiers to filter for quality and harmful content" 把过滤升级为「训练分类器」，同时覆盖质量与有害内容两个维度；第三条 "Deduplication using MinHash" 点名了 MinHash 这一近似去重算法；第四条 "Leaderboard: minimize perplexity given token budget" 被黄色高亮，直接把整份作业变成一个带约束的优化问题。讲师在 [59:46] 口述的版本是 "start with a very raw corpus, like a raw web crawl, and do all the work to filter and to dedupe and make the data clean"，与画面四条完全对应，只是没有点出具体算法与评测形式。

可以把高亮的那条写成约束优化式，这也是理解整份作业的钥匙：

$$
\min_{\text{pipeline}} \; \mathrm{perplexity}(\mathcal{M}_{\text{trained}}) \quad \text{subject to} \quad \#\text{tokens} \le B
$$

其中 $B$ 是给定的 token budget（token 预算），$\mathrm{perplexity}$ 是语言模型的困惑度（perplexity），越低表示模型对数据的建模越好。"given token budget" 这个限定极关键：它把计算量固定，逼迫学生在数据质量而非数据量上竞争，正好呼应课程反复强调的 efficiency（效率）主线。这也解释了为什么讲师在 [60:05] 把这类活称为 "dirty work"（脏活）——从 raw web crawl（原始网页抓取）开始做 filter、dedupe、clean 并不光鲜，他甚至说 "I don't know if I would call it not fun"，但紧接着强调这是 "an important part of building a language model from scratch"，学生 "have to get the full experience"（必须获得完整体验）。换句话说，作业四的教学目的不是让学生调出一个漂亮的 perplexity 数字，而是让他们亲手走过从原始 HTML 到可训练语料的整条管线。

与前文的联系是：前面的缩放定律作业假设数据已经可用，而 Assignment 4 恰恰补上「数据从哪来、怎么变干净」这一环；与后文的联系是，讲完作业四，[60:20] 立刻转入 alignment（对齐），也就是课程大纲里最后一个模块。一个容易踩的坑是把 "minimize perplexity given token budget" 理解成「词表越大越好」或「数据越多越好」——恰恰相反，预算固定时，把预算浪费在重复、低质或有害的文档上会直接抬高 perplexity，这正是 MinHash 去重与分类器过滤的价值所在。另一个易错点是 MinHash 只做近似去重，阈值设置过激会把有用的多样样本一并删掉；幻灯片并没有给出阈值或哈希数等参数，因此这里不宜臆测具体数值。

![Assignment 4 数据作业的四条要求：Common Crawl HTML 转文本、训练分类器过滤质量与有害内容、MinHash 去重、给定 token 预算最小化困惑度](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0048_01-00-13.jpg)

## 对齐：从全监督到弱监督 [60:20 - 62:53]

讲师在 [60:20] 用 "So, finally um alignment" 宣布进入 alignment（对齐）。他先复盘此前的训练范式："So far we've basically trained a model using full supervision. Predict the next token or the next few tokens."——到目前为止，模型是靠 full supervision（全监督）训练的，目标就是 predict the next token（预测下一个 token）或 the next few tokens（预测接下来几个 token）。幻灯片把这段写成了可执行讲义里的函数头与注释，这也是画面比转写更清楚的地方：

```python
def alignment():
    # So far, we have trained a model on full supervision (predict the next token).
    # Now that the model should be reasonable, we can improve it further from weak supervision.
    # Why weak supervision? When it is easier to critique than it is to generate.
    # Basic template:
```

"Now that the model should be reasonable"（此时模型应该已经不错了）是转折点：预训练给出一个合理的基座，但还可以用 weak supervision（弱监督）继续改进。弱监督之所以成立，直觉是 "sometimes it's easier to critique than it is to generate"（评判往往比生成更容易）：未必总能拿到「这个 prompt 的标准答案」这样的标注，但往往能定义「什么叫好」——例如一个 verifier 能判断数学答案对不对，一个人能判断两个回答哪个更得体，即使他自己写不出更好的回答。

讲师给出的 basic template 是三步：generate responses from a model（让模型生成若干回答）→ score them either with a human or verifier or LM judge（由人类、验证器 verifier 或 LM judge 打分）→ update the model to prefer better responses（更新模型，使其更偏好得分高的回答）。把口述模板形式化就是：

$$
\text{generate responses} \;\to\; \text{score (human / verifier / LM judge)} \;\to\; \text{update model to prefer better responses}
$$

这个模板可以用不同算法实例化：转写明确提到 "various RL algorithms such as PPO or GRPO"（PPO、GRPO 等强化学习算法），以及 "in a simpler for preference data DPO"（更简单的、基于偏好数据 preference data 的 DPO，即 Direct Preference Optimization，直接偏好优化）。三者的共同点是不再需要「标准答案」，只需要「相对好坏」的信号，这正是弱监督与全监督的分界线。

讲师对 RL 的态度很坦率，也是最值得记下的局限判断："the challenges around RL are that RL algorithms are unstable and hard to tune"，还调侃 "Some of you probably know this from first-hand experience"。他个人偏好 "keep things as much in the full truly supervised case as long as possible, and then finally okay, fine, I have to do RL"（尽可能久地停留在真正全监督的情形，最后才不得不做 RL），并补了一句 "some people for whatever reason like doing RL"。这里的易错点是：RL 的不稳定不是靠调参技巧就能完全消除的工程噪声，而是这类方法固有的性质，因此合理的工程顺序是先把全监督与 SFT/DPO 的收益榨干，再考虑 PPO/GRPO。

今年课程希望多讲的是大规模 RL 的系统挑战：如果 "you do RL at scale and try to maximize your throughput"，会出现 "a lot of systems challenges"。具体是：需要一个 inference server（推理服务器）和一个 training server（训练服务器）；推理服务器负责生成 rollouts（采样轨迹），若 RL 针对涉及 code execution（代码执行）的环境，整个流程就变成 "a whole kind of orchestration game"（编排游戏）；如果 workers lag behind（worker 落后），就会产生 off-policy issues（离策略问题），于是要 "constantly juggling this on-policyness with a desire to maximize throughput"（在 on-policy 性与吞吐之间不断权衡），讲师形容为 "a big wonderful mess"。Assignment 5 在课程地图里被标注为 "RLHF, RL algorithms, RL systems"，正好对应这三块；去年是 "implement DPO and GRPO and get it working for some math benchmark"，今年 "we're still deciding what exactly we want to do"，可能往更 realistic（贴近现实）的方向推。与前文的联系是：这里的 inference server 与 rollouts 直接对应系统单元讲过的 prefill/decode 推理与吞吐优化；与后文的联系是，五个模块讲齐之后，[62:54] 自然回到 efficiency 这条总纲。

## 效率作为贯穿全程的主线 [62:54 - 64:46]

讲师用 "So, again, remember it's about efficiency" 把话题收回总纲，并指出 efficiency（效率）可以指 data efficiency（数据效率）或 compute efficiency（计算效率）。幻灯片把可用资源写成一条式子：

$$
\text{Resources} = \text{data} + \text{hardware}\,(\text{compute},\ \text{memory},\ \text{communication bandwidth})
$$

并给出核心问题："How do you train the best model given a fixed set of resources?"（在给定一组固定资源下，如何按某种评测训练出最好的模型）。这其实就是课程早先给出的 $\text{accuracy} = \text{efficiency} \times \text{resources}$ 的约束版本：resources 被钉死，能优化的只有 efficiency。讲师在 [63:03] 展开说，你有 data、有带 compute cores（算力核心）的 hardware、有 memory（内存）、有 communication bandwidth（通信带宽），"you're just trying to figure out how do you build the best model according to some evaluation given a fixed set of resources"。直觉是：所有设计决策都可以看成在这个约束下的优化——不是「哪个技术最新」，而是「哪条路径把给定的数据、算力、内存与带宽用得最充分」。

幻灯片用可执行讲义的 Python 伪代码把五次作业钉在五个模块上，这段原文转写里完全没有念出来，但信息量很大，尤其是各作业的详细范围：

```python
def course_syllabus():
    basics()         # Assignment 1: tokenization, model architecture, training
    systems()        # Assignment 2: kernels, parallelism, inference
    scaling_laws()   # Assignment 3: scaling laws
    data()           # Assignment 4: evaluation, curation, transformation, filtering, deduplication
    alignment()      # Assignment 5: RLHF, RL algorithms, RL systems
```

注意 Assignment 4 的注释比上一章转写更细，多了 evaluation（评测）、curation（整理）、transformation（转换）三项，与画面上 Assignment 4 的四条要求互相印证；Assignment 5 明确写成 RLHF、RL algorithms、RL systems 三块，正好对应上一章讲的打分模板、RL 算法与系统编排三类内容。上方还有一句 "Please read the guide on how to access and use the compute."，属于课程后勤提示。

"Remember it's all about efficiency" 下方的五条 bullet，是讲师在 [63:20] 之后用效率视角逐个重读课程模块的结果，画面措辞比口述更精确：Systems: clearly about efficiency（系统，显然关乎计算效率）；Tokenization: working with raw bytes is elegant, but compute-inefficient with today's model architectures（直接操作原始字节很优雅，但在当今模型架构下计算效率很低，因此分词本质上是提升 compute efficiency）；Model architecture: many changes motivated by reducing memory or FLOPs (e.g., sharing KV cache, window attention)（许多架构改动是为了减少内存或 FLOPs，画面还举了共享 KV cache 与 window attention 两个具体例子，这是转写没念到的）；Data filtering: avoid wasting precious compute updating on bad / irrelevant data（数据过滤：避免在坏数据或无关数据上浪费宝贵算力做梯度更新）；Scaling laws: use less compute on smaller models to do hyperparameter tuning（缩放定律：用较少算力在小模型上做超参数调优，此条被黄色高亮）。讲师在 [64:08] 补充了数据过滤的逻辑：哪怕坏数据 "might not hurt you"，在固定 compute budget 下，"more time on bad data means less time on good data"，机会成本才是关键。

幻灯片还给了两句带时间维度的判断：Today, we are compute-constrained, so design decisions will reflect squeezing the most out of given compute（今天算力受限，设计决策体现为压榨给定算力）；Tomorrow, we will become data-constrained...（明天可能变成数据受限）。讲师在 [64:30] 说 "tomorrow we might become data constrained in the calculus of what design decisions you should take might change"，但强调 "the overall mindset, which we're trying to teach you, is think about the efficiency of your approach"。这里的易错点是把效率狭义理解成「跑得快」：此处的 efficiency 同时包含数据效率（每条样本带来的收益）和计算效率（FLOPs、内存、带宽），而且约束条件会随时代变化——今天的最优决策在数据受限时可能反转，可迁移的是「以效率为准绳做取舍」的 mindset（思维方式），而不是某个具体超参配置。讲完这段，[64:47] 有一段提问停顿（"Are there any questions about any of these assignments or topics?"），随后 [65:07] 正式跳入第一个单元 tokenization。

![Remember it's all about efficiency 幻灯片：资源公式与 Systems、Tokenization、Model architecture、Data filtering、Scaling laws 五条要点](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0051_01-04-28.jpg)

## Tokenizer 的定义、往返保证与压缩比 [65:07 - 68:42]

讲师在 [65:07] 宣布 "now let's do tokenization"，这是课程的第一个单元，并顺带推荐 Andrej Karpathy 关于 tokenization 的视频。他给出问题的两端：一端是 raw text（原始文本），"It's Unicode strings"（Unicode 字符串）；另一端是 "the language model places a distribution over sequences of tokens, usually represented as indices"（语言模型在 token 序列上定义概率分布，token 通常表示为整数索引）。两端之间存在表示鸿沟，因此必须有 encode（把字符串编码成 token）与 decode（把 token 解码回字符串）两个过程。幻灯片把定义写得比口述更精确："So we need a procedure that encodes strings into tokens. We also need a procedure that decodes tokens back into strings. A Tokenizer is a class that implements the encode and decode methods."——Tokenizer（分词器）就是一个实现 encode 与 decode 方法的类；画面顶部还给出一个索引例子 `indices = [15496, 11, 995, 0]`，并有一行被黄色高亮的 "To get a feel for how tokenizers work, play with this interactive site"（推荐交互网站，讲师现场因没联网而作罢，[65:54] 说 "I should have tried to get internet earlier"）。

最硬的约束是 round trip（往返一致）："If you implement a tokenizer that doesn't round trip, you have a problem."。画面给出了可执行的断言与 GPT-5（来自 OpenAI 的 tiktoken）的真实编码示例，这些具体索引是转写里读不到的：

```python
tokenizer = get_gpt5_tokenizer()
string = "Hello, 🌍! 你好!"
indices = tokenizer.encode(string)          # 13225 11 130321 235 0 220 177519 0
reconstructed_string = tokenizer.decode(indices)
assert string == reconstructed_string       # 黄色高亮：roundtrip 必须成立
```

画面里 `string = "Hello, 🌍! 你好!"` 被编码成 8 个索引 `13225 11 130321 235 0 220 177519 0`，`reconstructed_string` 又还原成同样的字符串。讲师还列了三条 Observations（幻灯片文字，转写只口述了个大概）：A word and its preceding space are part of the same token (e.g., " world")；A word at the beginning and in the middle are represented differently (e.g., "hello hello")；Numbers are tokenized into every few digits。这正是 [66:05] "you'll appreciate why tokenizers are kind of annoying, why people want to get rid of them" 的原因：带前导空格的 `hello` 与位于词首的 `hello` 是 "two completely different indices that have nothing to do with each other"（完全无关的两个索引）；数字方面存在 trade-off，有些 tokenizer 让每个数字单独成 token，可预测但会 "blowing up the number of tokens"（把 token 数撑大），有些则 "every few digits is a token"，可预测性时好时坏。

幻灯片给出 compression ratio 的定义："Compression ratio: number of bytes per token"，并配有 `compression_ratio = get_compression_ratio(string, indices)`。转写 [67:23] 把数字念全了：这个字符串有 20 个字节、8 个 token，所以

$$
\text{compression\_ratio} = \frac{\#\text{bytes}}{\#\text{tokens}} = \frac{20}{8} = 2.5 \ \text{bytes per token}
$$

"The larger the compression ratio, the shorter the sequence (good since attention is quadratic in sequence length)"——压缩比越大、序列越短越好，因为 attention（注意力）的计算量对序列长度是二次的 $O(n^2)$。这直接把分词与上一章的 efficiency 主线接上了：分词不是文字层面的整理工作，而是在压缩序列长度、省下 $O(n^2)$ 的算力，是纯粹的 compute efficiency 手段。讲师在 [67:13] 还强调 encode/decode 的可逆性："a tokenizer should round trip"，否则后续所有指标都失去意义。

提高压缩比最直接的办法是 "increasing the vocab size"（增大词表），但随即 "you get into sparsity"（稀疏性）：词表里每个元素都被当作一个 distinct element（独立类别），词表越大、每个 token 分到的训练信号越稀。讲师给的参数是 "these days tokenizers, especially multilingual tokenizers, have 100k or 200k tokens"，即 10 万到 20 万个不同 token。他本来想展示 GPT token vocabulary，但 "in the interest of time" 跳过了（[68:21]）。这里的易错点有两处：一是把 compression_ratio 的方向记反——它是 bytes/token，越大表示每个 token 覆盖越多字节、序列越短，不是越小越好；二是误以为可以无限增大词表来换压缩比，稀疏性以及 embedding 与输出层的参数量都会随之上升，反而伤害小模型。讲完这些背景，[68:28] "Okay, so how do you build a tokenizer?" 才正式引出下一章三种朴素方案的尝试。

![GPT-5 tokenizer 的编码解码往返示例、8 个索引与 compression_ratio 定义](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0053_01-07-18.jpg)

## 三种朴素分词方案及其缺陷 [68:43 - 71:57]

第一种尝试是 character-level tokenizer（字符级分词）。直觉来自幻灯片原文 "A Unicode string is a sequence of Unicode characters. Each character can be converted into a code point (integer) via ord."——Unicode 字符串本身已经是字符序列，每个字符可用 `ord` 转成一个 code point（码点，整数）。画面给出了具体断言，这些数值转写里只说 "you get some number out"，并没有点名：

```python
def character_tokenizer():
    assert ord("a") == 97            # 黄色高亮
    assert ord("🌍") == 127757
    assert chr(97) == "a"
    assert chr(127757) == "🌍"
```

`chr` 提供反向转换，因此 encode 可以写成 `list(map(ord, string))`（这也正是片尾总结页里 `CharacterTokenizer.encode` 的真实实现）；画面上方还有一段把词表按行写出的代码，用到 `tokenizer.token_byte_values` 与 `b.decode("utf-8", errors="replace")`。它天然能 round trip。但讲师给的参数是 "there are 150k Unicode characters, so your vocab size could be 150k"——约 15 万个 Unicode 字符，词表规模约 150k；他评价 "it's not crazy"，但更大的问题是 "many characters are actually rare"（大量字符很稀有），导致 "really an inefficient use of a vocabulary"。同时 compression ratio "is not that great"，因为 "most of the time you're actually using a lot of tokens to represent your sequence and many of the indices are actually not being very used"。结论是 "this is like not a very good tokenizer"。

第二种是 byte-level tokenizer（字节级分词）："Unicode has a UTF-8 encoding"，一个字符可能只占 1 个字节（如 `a`），也可能占多个字节（如 emoji、中文）。画面把 `string = "Hello, 🌍! 你好!"` 展开成一整行字节序列，这是转写完全没念、只能从图里读到的内容（下面按 Python 列表重排，画面原本是空格分隔）：

```python
string = "Hello, 🌍! 你好!"
# 画面上的字节序列，共 20 个值，全部落在 0..255
indices = [72, 101, 108, 108, 111, 44, 32, 240, 159, 140, 141,
           33, 32, 228, 189, 160, 229, 165, 189, 33]
tokenizer = ByteTokenizer()
indices = tokenizer.encode(string)             # 黄色高亮行
reconstructed_string = tokenizer.decode(indices)
assert string == reconstructed_string
```

可以逐字节对照：`72 101 108 108 111` 是 "Hello"，`44` 是逗号，`32` 是空格，`240 159 140 141` 是 emoji 🌍 的四字节 UTF-8 编码，`33` 是感叹号，`228 189 160` 与 `229 165 189` 分别是「你」「好」，末尾 `33` 是感叹号，合计 20 个数——正好对应上一章算压缩比时的 20 字节。画面还写着 "The vocabulary is nice and small: a byte can represent 256 values"，`vocabulary_size = 256`，以及 `assert compression_ratio == 1`，也就是

$$
\text{vocabulary\_size} = 256, \qquad \text{compression\_ratio} = 1
$$

局限很清楚：词表小得不能再小，但 "byte sequences can be very long"，压缩比退化到 1，等于没有任何压缩，attention 的 $O(n^2)$ 负担一点没减。讲师在 [70:19] 说 "all the numbers are between 0 and 255 because that's what a byte means"，正是对这一行的解释。

第三种是 NLP 早期常用的 word-based tokenizer（词级分词）：按空格或某个正则表达式把字符串切成 chunk，每个 chunk 作为一个 token，"Then, we can build a mapping from each chunk into an integer."。画面给出 `chunks = "I" " " "ll" " " "say" " " "supercalifragilisticexpialidocious" "!"`，并写明 "What's good: each token is meaningful (since humans invented words)"——词是人发明的，语义相对稳定，token 因此有含义；画面还给出一个转写没念的具体数字 `compression_ratio = 5.5`，以及 `vocabulary_size = number of distinct chunks in the training data`，结论是 "Compression ratio is good, but vocabulary size can be huge."。

词级方案的致命局限写在 "Moreover" 下的三条里，最后一条被黄色高亮：Many words are rare and the model won't learn much about them；This doesn't obviously provide a fixed vocabulary size；"New words we haven't seen during training get a special UNK token, which is ugly and can mess up perplexity calculations"。讲师在 [71:36] 进一步说词表 "could be actually unbounded"（可能无界）：测试时遇到训练没见过的序列，只能分配一个 `UNK` token，而 `UNK` 会 "mess up your perplexity calculations"（破坏困惑度计算）。这恰好与前面评测单元、以及 Assignment 4 的 "minimize perplexity given token budget" 冲突：如果分词器本身会凭空塞进未登录词，perplexity 就不再可信。三种方案各坏在一端——字符级词表约 150k 但稀有字符浪费、压缩比不佳；字节级词表 256 但压缩比只有 1；词级压缩比 5.5 却词表无界、必须依赖 `UNK`。易错点是不要以为「压缩比最高就选词级」：词表无界与 `UNK` 是表示层面的硬伤，不能靠调阈值修补。于是 [71:58] 讲师说 "so what we're actually going to do is called byte pair encoding"，进入 BPE。

![字节级 tokenizer：Hello, 🌍! 你好! 展开成的 20 个 UTF-8 字节序列与 vocabulary_size = 256](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0055_01-10-15.jpg)

![词级 tokenizer 的 compression_ratio = 5.5、chunks 示例与 UNK token 破坏困惑度计算的高亮说明](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0056_01-11-55.jpg)

## BPE：数据驱动的合并算法 [71:58 - 76:00]

讲师在 [71:58] 引出 byte pair encoding（BPE，字节对编码），并给出一段来历：BPE "was introduced a long time ago for data compression way before language models were really on the scene"（最早用于数据压缩，远早于语言模型出现）；"It was first introduced to NLP for doing neural machine translation"（先被引入 NLP 做神经机器翻译）；"the first paper that used BPE for LMs was GPT-2"（首个把 BPE 用于语言模型的论文是 GPT-2）。核心思想是 "train the tokenizer on raw text to construct a vocabulary that's tailored to the data"（在原始文本上训练分词器，构造贴合数据分布的词表），并带来一个关键性质——"everything can be tokenized, if it's rare then it just breaks up into smaller units rather than having this unk token"（任何东西都能被分词，稀有片段就拆成更小单元，而不需要 `UNK`）。一句话概括：common sequences（常见序列）被合并成一个 token，rare sequences（稀有序列）被拆成多个 token。这正好修掉了上一章词级方案的无界词表与 `UNK` 硬伤。

算法概念上很简单：从 corpus（语料，假设是一条长序列）出发，先得到 byte sequence，每个字节初始就是一个 token；然后反复合并 "successive pairs of adjacent tokens that occurred the most frequently"（出现次数最多的相邻 token 对）。两屏幻灯片合起来给出了完整的 `train_bpe` 实现，这些代码转写里只被口头带过，按行号顺序整理如下：

```python
def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:
    indices = list(map(int, string.encode("utf-8")))
    merges: dict[tuple[int, int], int] = {}          # index1, index2 => merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # 黄色高亮
    for i in range(num_merges):
        counts = count_adjacent_pairs(indices)      # 统计每对相邻 token 的出现次数
        pair = max(counts, key=counts.get)          # 取最频繁的相邻对
        new_index = 256 + i                        # 新 token 的 id 从 256 开始递增
        merges[pair] = new_index
        vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]   # 字节串拼接
        indices = merge(indices, pair, new_index)  # 全局替换该相邻对
    compression_ratio = get_compression_ratio(string, indices)
    return BPETokenizerParams(vocab=vocab, merges=merges)     # 黄色高亮
```

`count_adjacent_pairs` 的开头也能在画面上看到：`counts = defaultdict(int)`，用 `for index1, index2 in zip(indices, indices[1:]):` 遍历相邻对计数。注意 `new_index = 256 + i` 这个公式：初始词表占据 0..255 共 256 个字节 token，之后每轮合并新建的 token id 依次是 256、257、258……，与转写口述的数字完全对上；而 `vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]` 说明新 token 的字节内容就是两个旧 token 字节串的拼接。

讲师用简单字符串 "the cat in the hat" 逐步演示：先转成字节序列，统计相邻对，`116 104`（即 `t`、`h`）出现两次（"I guess there's a few ties but we'll just take the first one"），是最频繁的相邻对；合并后创建 token `256` 表示 `th`，"every time we see th, we're going to use 256 to represent that"，并把序列中所有 `116 104` 替换成 `256`；下一轮把 `256` 和 `101` 合并成 `257`，再合并一次得到 `258`。写成式子就是：

$$
(116, 104) \xrightarrow{\text{merge}} 256,\quad (256, 101) \xrightarrow{\text{merge}} 257,\quad \ldots \to 258
$$

"over time, the sequence is shrinking, and the vocabulary size is growing"（序列不断变短，词表不断变大）。讲师在 [75:15] 给出这个玩具例子的 compression ratio 是 1.5：比字节级的 1 好，但远低于词级的 5.5，原因也很直白——只做了三轮合并、语料又极小，压缩效果自然有限。

训练好之后，"how do you tokenize new text?"：拿一个新字符串，"go through the set of merges that you've made and then you just apply the merges to your string"（按学到的 merge 顺序，依次把合并规则应用到新字符串上）。画面演示了 `string = "the quick brown fox"` 的编码结果，是一串以 258 开头的索引 `258 113 117 105 99 107 32 98 114 111 119 110 32 102 111 …`，`reconstructed_string = "the quick brown fox"`，并再次断言往返一致；开头的 258 正是上一轮三轮合并得到的新 token。讲师说 "let me not actually step through that code"，即不逐行讲 encode 实现。这里的易错点是：merge 的应用顺序必须与训练时完全一致，否则同一个字符串会编码出不同的索引序列，round trip 之外的可复现性也会丢失；另一个局限是这份实现虽然 "works"、是 "a full-blown BPE implementation"，但 "extremely slow"（极慢），讲师明确把优化留给了 Assignment 1——这正是下一章的起点。

![train_bpe 合并循环代码：取最频繁相邻对、new_index = 256 + i、vocab 字节拼接与返回 BPETokenizerParams](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0058_01-15-13.jpg)

## 实现优化、分词总结与替代方案应满足的性质 [76:00 - 79:21]

讲师在 [76:00] 承认 "I went through this a bit fast just in the interest of time"，并重申这份实现 "works"、是 "a full-blown BPE implementation"，但 "extremely slow"。瓶颈在 encode："encode loops over all the merges, which is very slow because the number of merges you have is essentially the vocab size minus 256"，也就是

$$
\#\text{merges} \approx |V| - 256
$$

若词表是 10 万量级，就要遍历近十万条 merge。画面把 Assignment 1 要做的四件事列得比转写更具体，黄色高亮的是第三条：

```text
In Assignment 1, you will go beyond this in the following ways:
  • encode() currently loops over all merges. Only loop over merges that matter.
  • Detect and preserve special tokens (e.g., <|endoftext|>).
  • Use pre-tokenization (e.g., the GPT-2 tokenizer regex).   # 黄色高亮
  • Try to make the implementation as fast as possible.
```

其中 `<|endoftext|>` 这个 special token 的具体例子、以及 "the GPT-2 tokenizer regex" 这个预分词正则，都是转写没念到、只能从画面补出的细节；转写只笼统说 "there's some details around special tokens. Conceptually not deep but important to building a modern tokenizer"。讲师在 [76:36] 说要 "build some indices to make that happen"（建索引，只遍历真正相关的 merge）；[76:55] 进一步指出现实中并不是 "take an entire string and then you try to tokenize it"，而是 "break it up into chunks and then you apply tokenizer on each chunk"，即 pre-tokenization（预分块），"that's going to be much faster"；[77:17] 还补了一句 "at some point you might realize that Python is just not very fast"，可以用 Rust 或 C 之类的语言重写。

[77:29] "Okay, so quick summary." 进入片尾总结页，画面的 Summary 四条比口述更完整，也把整段的分词讨论收了口：

```text
Summary:
  • Tokenizer: strings ↔ tokens (indices)
  • Character-based, byte-based, word-based tokenization are highly suboptimal
  • BPE is an effective heuristic that is data-driven
  • Tokenization is a separate step, maybe one day do it end-to-end from bytes...
```

这里 `strings ↔ tokens (indices)` 用双向箭头把 round trip 直接写进了定义；"highly suboptimal"（高度次优）对应上一章三种方案各自坏在一端；"effective heuristic that is data-driven" 点明 BPE 只是数据驱动的启发式方法、并没有理论最优保证；第四条 "Tokenization is a separate step, maybe one day do it end-to-end from bytes..." 呼应讲师 [77:49] 的玩笑 "maybe next year I don't have to teach this but for this year we're stuck with tokenization"（也许明年就不用教这个了，但今年还绕不开分词）。画面下方还能看到 `class CharacterTokenizer(Tokenizer):` 的实现骨架，`def encode(self, string: str) -> list[int]: return list(map(ord, string))`，把前面讲过的字符级方案真正落成了代码。

更关键的是 [78:01] 之后的两条性质。讲师说 "even if we get rid of tokenization, whatever solution replaces it has to satisfy the following properties"，画面把它们写成两条，第二条被黄色高亮，且括号里的解释是转写没念到的：

```text
But whatever solution needs to satisfy:
  1. Model (e.g., Transformer) should operate on chunks (abstractions) of the sequence (text, video, DNA ...)
  2. Chunks should be variable (allocate more model capacity to interesting chunks)   # 黄色高亮
```

第一条：Transformer 需要在序列的某种抽象（abstractions / chunks）上运作。讲师在 [78:18] 解释，这一点在 "not just text but video or DNA sequences" 上最明显：原始的字节或单元 "quite kind of low signal to noise"（信噪比很低），"you have to do some sort of abstraction to lift it into a place where you can do modeling on that"（必须先做抽象，把它抬升到可建模的层面）。第二条：chunk 应当是 variable（可变长度）的，实现 adaptive computation（自适应计算）——画面括号里的 "allocate more model capacity to interesting chunks"（把更多模型容量分配给有趣的 chunk）正是对这一点的精确解释；讲师说 "not all bytes are treated the same"，若做不到，"I think you're going to be suboptimal"，任何 end-to-end 方案也必须满足这两条。

这一段的易错点在于把「抛弃 tokenizer」理解成「直接把字节喂给 Transformer 就行」：讲师用 low signal to noise 与 adaptive computation 两条约束否定了这种朴素想法——即便端到端，仍要有抽象、仍要有可变长的计算分配；另一个易错点是忽略 "maybe one day do it end-to-end from bytes" 的语气，那是展望而非既成事实，今年的前沿模型仍在使用 tokenizer。最后 [78:58] "with that, I will end"，预告下一次课（Wednesday）开始 unit on resource accounting（资源核算），讲师称它是 "a baby systems"（小型系统入门），之后再回到 architectures（架构）。这与前面 `course_syllabus()` 里 `basics() → systems()` 的顺序一致，也把整段重新落回「效率」这条主线：分词是 basics 的第一个单元，而它之所以存在，本质上仍是为了在固定算力下把序列压短、把算力花在更值得的 chunk 上。

![Assignment 1 的 BPE 实现优化方向与 the quick brown fox 编码往返示例](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0059_01-16-52.jpg)

![片尾 tokenizer 总结页：Summary 四条要点与替代方案必须满足的两条性质](/coursenotes/cs336-language-modeling-sp26/l01-overview-and-tokenization/kf_0060_01-18-56.jpg)

## 要点回顾

### 课程定位与效率主线

- CS336 Spring 2026 第三次开课，坚持 from scratch 与 `understanding via building`；讲义本身是可执行 Python 程序，`def main():` 依次调用各章函数，`tokenization()  # First unit` 标明分词是第一个正式单元。
- 5-unit、5 次作业对应 `basics()`、`systems()`、`scaling_laws()`、`data()`、`alignment()` 五部分；GPT-4 训练成本约 1 亿美元、前沿推测达十亿美元量级，技术报告又不公开架构与数据构造，课堂只能做 <1B 参数的小模型。
- 可迁移知识分 `Mechanics / Mindset / Intuitions`，机制与思维可教且能迁移、直觉不必然跨尺度迁移，故强调 `profile and benchmark everything`；MLP 的 FLOPs 占比也从约 44% 升到 175B 规模下约 80%。
- The Bitter Lesson 的正确解读是 `algorithms that scale are what matter`，并以 $\text{accuracy} = \text{efficiency} \times \text{resources}$ 把效率写入公式；[Hernandez+ 2020] 测得 2012–2019 年 ImageNet 上 44x 算法效率提升，总框架是固定资源下只优化 efficiency。

### 分词与 BPE

- tokenizer 的接口是 $\text{encode}: \text{str} \to \text{list[int]}$ 与 $\text{decode}: \text{list[int}] \to \text{str}$，方法体只有 `raise NotImplementedError`，硬约束是 round trip：`assert string == reconstructed_string`；`compression_ratio` 为字节数/token 数，示例 20/8 = 2.5 bytes per token，越大越好因为 attention 是 $O(n^2)$。
- 三种朴素方案各坏一端：字符级约 150k Unicode 字符（`ord("a") == 97`）、字节级 `vocabulary_size = 256` 且 `compression_ratio = 1`、词级 `compression_ratio = 5.5` 却词表无界并靠 `UNK` 破坏 perplexity；BPE 则 `1000 bytes → ~250 tokens`，带来 adaptive computation。
- `train_bpe(string, num_merges)` 从字节起、`vocab` 初始 0..255，每轮用 `count_adjacent_pairs` 取最频繁相邻对、`new_index = 256 + i` 建新 token 并拼接字节，返回 `BPETokenizerParams(vocab, merges)`；"the cat in the hat" 中 `116 104` → 256 → 257 → 258，玩具例压缩比 1.5。
- 朴素 `encode` 遍历全部 merge，$\#\text{merges} \approx |V| - 256$；作业一要只遍历相关 merge、保留 `<|endoftext|>`、用 GPT-2 tokenizer regex 做 pre-tokenization，必要时改用 Rust/C 重写。

### 资源核算与系统优化

- 资源核算用 $\text{FLOPs} \approx 6ND$：`total_flops = 6 * 70e9 * 1e12` 即 70B 参数、1T token 约 $4.2\text{e}23$ FLOPs（转写里的 7B 以画面为准）；B200 为 `2.25 PFLOP/sec` (bf16) 配 `8TB/sec`，参数须从 HBM 搬到 SMs，`Roofline analysis` 一般判为 memory-bound。
- kernel 是 `a function that runs on GPU`；`Naive` 的 2 读 2 写经 operator fusion 变成 `Fused: read HBM; compute A and B; write HBM` 的 1 读 1 写，策略还有 tiling（FlashAttention）与 `CUDA/Triton/CUTLASS/ThunderKittens`。
- "What if we have 1024 GPUs?" 下跨 GPU 数据移动更贵，靠 gather/reduce/all-reduce 与 `{data, tensor, pipeline, sequence, expert} parallelism` 把 parameters、activations、gradients、optimizer states 分片。
- 推理分 prefill（compute-bound、建 KV-Cache）与 decode（memory-bound、一次一个 token）；加速靠剪枝／量化／蒸馏、speculative decoding（`exact decoding!`）、fused kernels 与 continuous batching，作业二要求 `Implement a fused RMSNorm kernel in Triton`。

### 缩放定律与计算最优

- 设定是 `1e25 FLOPs`（数千万美元）只能训一个模型、`Too expensive to do hyperparameter tuning at full scale!`；关键转变是 `scaling recipe: FLOPs → hyperparameters (config)`，用 `1e24` 等小实验拟合后外推预测目标 loss。
- `Scaling laws don't happen automatically`，配方要靠 hyperparameter transfer [Yang+ 2022] 规定学习率、batch size 随规模怎么变，结论是 `Predictability is at least as important as optimality!`。
- 计算最优是给定 $C = 6ND$ 时在更大 $N$ 与更多 $D$ 间分配：ISOFLOP curves 从 `6e18` 到 `3e21` FLOPs 扫描最优 $N$ 再拟合外推，经验规则 $D = 20 N$（70B → ~1.4T tokens），但未计入推理成本。
- Marin 的 pre-registering 把预测押到 `1e22 FLOPs`，图上 Δloss = +0.011 / +0.005；作业三用 `training API (hyperparameters → loss)` 的离线 cache，最终 `Leaderboard: minimize loss given FLOPs budget`。

### 数据、评测与对齐

- `def data():` 先问 `What capabilities do we want the model to have?`：internal 看 `smoothness across scales` 与相对性能、external 看 `ecological validity`；perplexity 宜在 private documents 上算以 `avoid contamination`，高阶用例有 GPQA、HLE、SWE-Bench、Terminal-Bench。
- 原始数据是 HTML、PDF、directories 而非文本，流程为 transformation、filtering（classifiers）、deduplication（Bloom Filters / MinHash）、data mixing（upweight/downweight）与 rewriting / synthetic data，并分 pretraining、mid-training（含 long-context）、post-training 三层。
- 对齐从 full supervision（predict the next token）转向 weak supervision，理由是 `easier to critique than it is to generate`；模板为 generate responses → score (human / verifier / LM judge) → update model to prefer better responses，可用 PPO、GRPO 或更简单的 DPO。
- 大规模 RL 需 inference server 生成 rollouts、training server 训练，worker 落后带来 off-policy issues；RL 本身 unstable and hard to tune，宜先把全监督收益榨干。

### 作业与可复现

- 作业不给 scaffolding code，但提供 `unit tests` 与 `adapter interfaces` 避免 sparse rewards；流程是先 laptop 本地验正确、再上 cluster 跑 accuracy 或申请 GPUs 做 kernel benchmark。
- 排行榜都是固定预算下的优化：作业一 `45 minutes on a B200` 最小化 OpenWebText perplexity（去年设定）、作业四 `minimize perplexity given token budget`、作业三 `minimize loss given FLOPs budget`；三条平衡原则是 Expressivity、Stability（参数与梯度范数落在 Goldilocks zone）、Efficiency。
- 五次作业各有落点：作业一实现 BPE、Transformer、cross-entropy loss、AdamW 与 training loop；作业二做 distributed data parallel training 与 optimizer state sharding；作业四 `Convert Common Crawl HTML to text`、`Deduplication using MinHash`；作业五覆盖 RLHF、RL algorithms、RL systems。
- AI 政策承认 coding agents 能解掉所有作业，却硬性要求使用 `AGENTS.md` 让 AI 保持 `pedagogically-minded`；可复现性还体现在 tokenizer 必须 round trip、BPE 的 merge 顺序须与训练时一致、评测用 private documents 避免 contamination。

## 视频来源

- 原始链接：https://www.youtube.com/watch?v=JuoVZkPBiKk
- 课程：Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 1

---

## 对照参考笔记补遗（Spring 2025 版）

> 编者注：本节依据 2025 版参考笔记整理，其中的「更正」小节记录的是核对时发现的问题，对应正文已在本版订正；补遗内容不代表 2026 课程视频原文。


本节内容来自 Spring 2025 版参考笔记，仅用于对照补充，不代表 2026 视频原文；凡与正课叙述冲突之处，一律以正课为准。

### 一、BPE 的来历与 GPT-2 的预分词

参考「Byte Pair Encoding (BPE)」一节给出了我们笔记里被口语带过的确切出处。BPE 并非为语言模型发明：它由 Philip Gage 于 1994 年作为一种数据压缩算法提出，远早于神经语言模型出现；2015 年 Sennrich+ 把它引入 NLP 用于神经机器翻译（neural machine translation, NMT），在此之前该领域普遍使用 word-based tokenization；直到 GPT-2（Radford+ 2019）才首次把它用于语言模型。把这条链条记成「数据压缩（1994）→ NMT（2015）→ LM（2019）」的三次转手，才能解释为什么 BPE 的直觉是「常见序列压成一个 token、稀有序列拆成多个 token」——它本来就是一套按语料统计做合并的压缩启发式，而非某种语言学分词规则。这也补上了正课里「很久以前用于数据压缩」「先被引入 NLP 做 NMT」两句话背后的作者与年份。

同一节还有一条我们笔记未点明的工程细节：GPT-2 并不是在整条字节流上全局跑 BPE，而是先用 word-based tokenization 把文本切成若干 initial segments（初始片段），再在每个片段内部运行原始 BPE 算法，参考并特别注明 "This is what you'll do in this class"，即作业一要复现的正是这个两段式结构。它正是正课里 pre-tokenization、GPT-2 tokenizer regex 与 special tokens（如 `<|endoftext|>`）优化方向的上游解释：预分词限定了 merge 的作用域，既避免跨词产生语义可疑的合并，也让 encode 只需遍历与当前片段相关的 merge，从根本上缓解「遍历全部约 $|V|-256$ 条合并」的慢。

这里有两个易错点。其一，别把 BPE 的「发明」归给 GPT-2 或 Sennrich+：前者是首次用于语言模型，后者是首次引入 NLP，原始出处是 Gage 1994。其二，别把预分词当成可有可无的加速技巧——它决定 merge 能跨越的边界，因而直接影响最终词表构成与压缩比。它与正课 $1000\ \text{bytes}\to\sim250\ \text{tokens}$、attention 为 $O(n^2)$ 的效率主线相连：先分片、再合并，本质仍是把长字节流压成更短的 token 序列，把算力留给值得建模的片段。

### 二、对齐的目标、SFT 的显式目标与反馈阶段的谱系

参考「Alignment」与「Two Phases」两节把对齐拆得更规整。它先把 alignment 的目标并列成三条：让语言模型 follow instructions（遵循指令）、tune style（调整格式、长度、语气等风格）、以及 incorporate safety（纳入安全，例如对有害问题给出 refusal 拒答）。正课由 full supervision 转向 weak supervision 的统一视角切入，并未把这三条并列写出；补上它才能理解对齐为什么不是单一损失函数，而是一组彼此可能冲突的行为约束，也呼应正课「基座模型只是原始潜力、需要被变得有用」的判断。

监督微调（Supervised Fine-tuning, SFT）阶段，训练数据是 prompt–response 对，学习目标可写成一个条件概率的最大化：

$$\max\ P(\text{response}\mid\text{prompt})$$

参考还点名 Zhou+ 2023 的直觉——"Base model already has the skills, just need few examples to surface them"，即基座模型在预训练里已具备所需能力，少量高质量示例只是把能力「激发/显化」出来，因此 SFT "relatively simple and effective for initial alignment"。这与正课「easier to critique than it is to generate」的弱监督动机恰好互补：一个说明标注不必多，一个说明评判比生成便宜，二者共同解释了为什么对齐阶段能摆脱「标准答案」式的 full supervision。

进入 Learning from Feedback 阶段后，参考补出了我们笔记一笔带过的记号与分类：先让模型对同一 prompt 生成多个回答（如 A、B），由人给出偏好 $A<B$ 或 $A>B$，得到 preference data（偏好数据）；打分者分两类——formal verifiers（形式化验证器，面向 code、math 等可自动判定对错的任务）与 learned verifiers（学习式验证器，训练一个 LM-as-a-judge，也就是 RLHF 里的裁判模型）。算法谱系在参考里给出了完整出处：PPO（Schulman+ 2017；Ouyang+ 2022）、DPO（Rafailov+ 2023，直接基于偏好数据、更简单）、GRPO（Shao+ 2024，去掉 value function）。补上年份与作者后，PPO→DPO→GRPO 这条「越来越轻、越来越贴近偏好数据」的脉络才完整。易错点是：DPO 的全称是 Direct Preference Optimization，参考在其中一处误写为 Direct Policy Optimization（详见文末更正）；GRPO 与 PPO 的结构差别在于 Remove value function，而不是换了一种偏好记号。

### 三、评测的层次与「数据决定能力」

参考「Data」与「Evaluation」两节先立因果：模型的能力基本由训练数据决定（multilingual data → multilingual capabilities，code data → code capabilities），因此数据单元第一步是问「想要什么能力」，第二步才是「去哪找数据」；评测之所以要前置，是因为 "evaluation basically defines the capabilities"。这与正课 internal / external 二分——看 smoothness across scales 与 relative performance，还是看 ecological validity——是同一逻辑的两种切法，可互为注脚。

参考给出的 2025 版评测清单按用途分层，与正课点名的 GPQA、HLE、SWE-Bench、Terminal-Bench 互补：教科书式内在指标是 Perplexity（困惑度）；标准化测试用 MMLU、HellaSwag、GSM8K；指令遵循用 AlpacaEval、IFEval、WildBench；测试时算力扩展（scaling test-time compute）靠 Chain-of-thought 与 ensembling；生成式任务用 LM-as-a-judge；再往外是全系统层的 RAG 与 agents。把两版清单并置可以看出，2026 版把重心移向了更难的 agentic 与长程基准，但「从 perplexity 到全系统」的层次结构并未改变。

易错点有两处。别把标准化测试与指令遵循混成一类：前者考知识与推理，后者考格式与服从，用刷榜思路做开发期指标正是正课批评的做法。也别用单一平均分掩盖差异——正课已提醒平均会 conflates 许多不同能力，参考的清单恰好给出「为什么不能平均」的具体维度。此外，用 private documents 计算 perplexity 以 avoid contamination（数据污染）的告诫，在 2025 版同样成立；Perplexity 作为抗 bench-maxing 的内在质量指标，两版地位一致。

### 四、效率重述里的具体推论与量级锚点

参考「Efficiency Drives Design Decisions (Recap)」用效率视角逐模块重读课程，比正课多出两条硬结论：其一 "Training: We can get away with a single epoch!"，在算力受限、数据相对过剩的当下，训练一个 epoch 就足够；其二 "Alignment: If tune model more to desired use cases, require smaller base models"，对齐越贴近目标用例，所需的基座模型反而可以更小。前者是正课讨论数据过滤与缩放定律时的隐含前提，后者把对齐直接挂回 $\text{accuracy}=\text{efficiency}\times\text{resources}$：既然 resources 被钉死，调好用例就能省下参数量。

转到「Tomorrow: Data-Constrained Regime」，参考把正课一句 "Tomorrow we will become data-constrained" 展开成两个可检验的推论：若算力过剩而数据不足，"Training for one epoch doesn't make sense"，应改为更多 epoch 或更聪明的技术；并且 "Different architectures might emerge, as the Transformer was motivated by compute efficiency"——Transformer 当年正是被算力效率驱动的产物，约束条件一变，架构选择可能随之改变。易错点在于：可迁移的是「按约束做取舍」的 mindset，而不是 single epoch 这个具体结论本身；把今天的最优决策当成永久定律，恰恰违背了正课对 intuitions 不必然跨尺度迁移的警告。

几组散落的具体量级也值得留存。「It's All About Efficiency」把课程总问题落成一个可算的设定：给定一份 Common Crawl dump 与 32 张 H100、两周时间，你该怎么做；「Pre-neural」提到 Google 曾用 2 trillion tokens 训练 5-gram 模型，这个规模直到近年才被现代语言模型重新追上，并判断 "All these ingredients were in place by 2020"；「Data Curation」补充商业数据购买的动因是 "publicly available data is limited for state-of-the-art performance"（如 Google 与 Reddit 的授权）。这些数字给正课里 N-gram、compute budget、license data 的定性叙述补上了量级锚点。

「Levels of Openness」则把开放程度切成三级并逐一写明披露范围：Closed models（如 GPT-4o）仅 API access；Open-weight models（如 DeepSeek）给权重、论文含架构与部分训练细节、但不含数据细节；Open-source models（如 OLMo）连权重与数据都开放、论文含大部分细节，却仍 "not the rationale, failed experiments"（不给设计理由与失败实验）。正课已区分 open-weight 与 open-source，补上「连开放源码也不等于可复现研究过程」这一层限定，才能解释为何课程要靠大量论文去 glimpse 与 triangulate 前沿模型的构造，以及为何 data mixture 外界始终不得而知。

### 五、系统与推理补遗：仓库—工厂类比与推理成本

参考「Kernels」一节用一个朴素类比讲清 GPU 的存储层次：DRAM（内存）是 warehouse（仓库），SRAM（靠近计算的一侧）是 factory（工厂），L1/L2 caches 在片上（on-chip）、DRAM 在片外（off-chip）；瓶颈是 bandwidth cost，即数据在仓库与工厂之间来回搬运，而非浮点算力本身。对应的 Trick 是 "Organize computation to maximize utilization of GPUs by minimizing data movement"，具体技术是 Fusion 与 tiling，课程用 Triton（参考注明是 OpenAI 的 GPU 编程框架）来写 kernel。正课以 B200 的 $2.25\ \text{PFLOP/sec}$（bf16）与 $8\ \text{TB/sec}$、HBM→SMs 的搬运、以及 $2.25/8\approx281\ \text{FLOP/byte}$ 给出了 2026 的量化版本，两者是同一原理的定性与定量两面。

「Inference」一节还有一个正课没有的全局论断：inference compute 是「每次使用都发生」的成本，training compute 是「一次性成本」，因此 "Globally, inference compute exceeds training compute"。这为 prefill（compute-bound）/ decode（memory-bound）两阶段、KV caching、speculative decoding（参考同样强调 "exact decoding!"）、以及 batching 等优化补上了经济动机——正课详细写了机制，补上这条成本判断后，「为什么值得专门优化 decode」更完整。

易错点与关系：别把 warehouse/factory 类比读成「内存越大越好」，重点是搬运次数而非容量；对应到正课的 Naive（2 读 2 写）→ Fused（1 读 1 写），省下的正是跨边界的往返，A 与 B 的数学语义并未改变。参考以 A100 为例、正课以 B200 为例，属于硬件换代而非结论改变；跨 GPU 时 "Data movement between GPUs is even slower" 的判断，与正课 $\{data, tensor, pipeline, sequence, expert\}\ \text{parallelism}$ 的分片维度也是同一条原则在更大尺度上的延伸。

### 更正

原笔记一处「`Popular tokenizer: Byte-Pair Encoding (BPE) [Senrich+ 2015]`」应改为「`[Sennrich+ 2015]`」（依据：参考「Tokenization」与「Byte Pair Encoding (BPE)」两节均作 Sennrich+ 2015，即 2015 年把 BPE 引入神经机器翻译的那篇文献）。这是文献作者姓氏少一个 n 的拼写笔误，不影响算法结论，但引用时应订正。除此之外，逐讲核对未发现正课在公式、单位、数值或概念归属上的错误；BPE 玩具例的合并顺序、DPO 与 GRPO 的全称、leaderboard 的举例等差异，均判为 2025 与 2026 的版本差异或参考笔记自身的解析错漏，已在对照报告中逐条说明，故不改动正课正文。
