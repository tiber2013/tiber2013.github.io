---
title: "第 15 讲 · 中期训练与后训练"
description: "后训练（post-training）是把 GPT-3 这类强大的基座模型变成可用助手的关键步骤，但它的主要杠杆不在算法，而在数据：数据长什么样、由谁标注、如何控制风格与安全，最终决定了模型的行为。SFT 擅长“提取”预训…"
---

# 第 15 讲 · 中期训练与后训练

> **Stanford CS336 语言模型从零构建 · L15** ｜ 讲者 Stanford Online ｜ 80 分钟 ｜ [原视频](https://www.youtube.com/watch?v=2oH6PWPrYFo)

> **v2 详写版** · 按 20 分钟分段详写，配图覆盖全片；文末附「对照参考笔记补遗（Spring 2025 版）」，记录与 2025 版讲义笔记的核对结果。

## 核心结论

这一讲把课程从预训练推进到后训练：又大又强的基座模型并不等于可用产品，真正让 GPT-3 走向 ChatGPT 的是指令遵循这种细粒度控制，而实现它靠的不是堆规模，而是显式收集"想要的行为"数据并做显式引导。讲师反复强调算法层面我们相当清楚，算法不是秘密武器，真正的杠杆在数据——这也解释了为何整讲绝大部分篇幅围绕 SFT 数据展开，因为 SFT 的做法几乎和预训练一样，只是跑梯度下降，唯一差别在训练数据。沿大致时间线，开源 SFT 数据从 FLAN、Self-Instruct、Alpaca、Open Assistant 一路走到 Nemotron 的工具调用，回答越来越长、越来越像人，并被结构化为带 tool_calls 与 todos 的 JSON；由此提炼出 SFT 最擅长的是"提取"预训练里已有的行为，少量高质量样本就够，硬教模型不知道的长尾知识反而会诱发幻觉，而长度、风格、引用与安全数据都是标注者有意识的决定，会系统性污染偏好评估。进入 RLHF 后目标从"拟合一个分布"切换成"最大化一个奖励"，流程靠 pairwise 排序、奖励模型与日趋专家化的人类标注支撑，随后又被模型标注和 DPO 大幅简化；但过优化、模式坍缩与校准变差仍是明确留下的开放问题。

## 从 GPT-3 到 instructGPT：指令遵循作为一种细粒度控制 [00:05 - 03:27]

讲师开场宣布课程要从 pre-training（预训练）转向 post-training（后训练），并直说后训练是语言建模里"更 messy"（更脏乱、更依赖手工艺）的部分：数据本来一直 messy，但 "post-training and everything that goes along with it is really, you know, kind of artisanal in many ways"。到此为止，课程能带你走到一个 "souped-up version of GPT-3"（加强版 GPT-3）——用更多 FLOPs、更多数据，可以做得比 GPT-3 好一点，"but ultimately, um the amount of utility you can get from something like GPT-3, which is a very big, strong base model, is ultimately a little bit limited"。换句话说，一个又大又强的 base model（基座模型）并不等于可用产品。画面上第一张幻灯片标题正是 "The class thus far"，副标题写着 "We've now covered pre-training, which gets you to GPT3.. But how do we get to instructGPT?"；左侧是 copy.ai 的文案生成示例，Input 里是 "We're creating the new beauty essentials: easy-to-use skincare..." 的 Glossier brief，Results 里是生成的圣诞营销邮件；右侧是 "OMG! What Will Happen When A.I. Makes BuzzFeed Quizzes?" 的新闻标题与说明 "The site announced plans to use tools from the creator of ChatGPT to create content. Former quiz writers aren't surprised."；底部并排放着 GPT-3 的黑底标志和 "ChatGPT: Optimizing Language Models for Dialogue" 的深绿标题块。这些画面正是讲师所说 GPT-3 时代"只能做 copywriting（文案写作）或一些不需要 reliability（可靠性）与 instruction following（指令遵循）的趣味小任务"的注脚：如果现在回头去用 GPT-3，"What is this thing? What is the point of this thing?"。

这一步要获得的核心能力是 instruction following（指令遵循），讲师把它定义为 "being able to put in a pretty long, complicated prompt and get back, you know, very reasonable answers"——输入一段很长、很复杂的 prompt，模型能给出相当合理的回答。他强调 "how remarkable it is that we can do instruction following"。在 GPT-3 级别，"fine-grained control of these systems was basically impossible"：你只能塞一堆 few-shot prompts，然后指望 "if my examples were good enough, I'd get something reasonable"，实际 "your ability to steer these models are limited"。而等到 GPT-3.5、GPT-4 出现，开始出现 "very long sort of programmatic prompts"（很长的程序化提示），模型能 "one-shotting um the whole prompt"（一次性把整段提示做完）。画面上 "Instruction following is a remarkable form of control" 一页给出了来自 [Bubeck et al 2023] 的极端例子：一整段英文 prompt 要求 "generate a pylot for the following data"，数据为 $x = [1, 3, 5, 6, 8]$、$y_1 = [2, 3, 5, 18, 1]$、$y_2 = [3, 5, 6, 8, 1]$、$y_3 = [4, 9, 7, 2, 3, 1]$、$y_4 = [9, 7, 2, 3, 1]$，写成代码块就是画面给出的数组：

```python
x  = [1, 3, 5, 6, 8]
y1 = [2, 3, 5, 18, 1]
y2 = [3, 5, 6, 8, 1]
y3 = [4, 9, 7, 2, 3, 1]
y4 = [9, 7, 2, 3, 1]
```

这段 prompt 的要求密度远超一般问答：$y_1,y_2$ 要画在同一张图、$y_3$ 放在旁边另一张、$y_4$ 放在下方；图例分别命名为 "bob"、"bob2"、"billio"、"allie"；x 轴标 "time"、y 轴标 "money"；给每条曲线加 $10\%\text{–}40\%$ 的 random error bar（随机误差棒）并覆盖 "including non-integers"（非整数时刻）；要 smooth curves 与 smoothed error bar，且明确 "Do not use linear interpolation, use smooth interpolation!"；要加一点小 zig-zag 让平滑曲线 "look more real"；每条线画一条 baseline（均值）；下方放 pie chart 表示四人在每个 time step 的占比，并要求 "**continuous time**" 动画、"Interpolate the missing fractions!"；其余三张图里加 vertical line animation 与饼图同步；最后一句是 "I want the Fanciest plot. Please add as many fancy things as possible."。页面接着写 "Here is a possible code to generate a pylot for the following data, with some comments to explain the steps:" 与 "GPT-4: [We give two snapshots from the animation resulting from the produced code]"，底下排着两组折线图、饼图（标题里能看到 3.26、7.86 这样的时刻数值，图例有 bob、alex、allie、billio 等），并标注 "Text example from [Bubeck et al 2023]"。它说明所谓"控制"不是问模型一个事实，而是让它在一次输出里同时满足几十条相互约束的格式、数值与动画要求。

![指令遵循作为精细控制的极端案例：一段包含数组、误差棒、平滑插值、饼图与连续时间动画要求的长提示及其 GPT-4 输出](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0002_00-01-52.jpg)

为什么不能跳过 pre-training（预训练）直接靠后训练取胜？讲师的措辞非常绝对："if we ignore pre-training and we just try to post-train our way to victory, we will get none of the things that we want. Absolutely none of it." 原因是 pre-training "scales in this very diverse broad way"（以非常多样、广泛的方式规模化），它是 "the basis on which we build all of this"。但有了预训练这锅 "primordial soup"（原始汤）之后，还得 "extract the kinds of behaviors that we want" 出来，为此需要 "much more explicit data collection, much more explicit steering, and a lot of messy engineering that in some ways we didn't have to do so far"；他甚至说架构那点 messy 与真实后训练的 messy "really doesn't compare"。画面上 "Goal today: enable better, tighter controls over LM output" 一页把整条思路压成两行字："Pretraining data isn't quite what we want (but it scales).." 与 "Can we collect data of behaviors we do want and train the LM?"，下面列出三个问题："1. What does that data look like?"、"2. How do we best make use of that data?"、"3. Do we need scale for this?"。这三个问题正好对应本讲后面依次展开的数据形态、训练方法与规模讨论；而 "Can we collect data of behaviors we do want" 里的 *do* 是斜体强调，点出后训练的本质是主动收集"想要的行为"而非被动接受互联网分布。

从课程结构看，讲师把路线分成两步：本讲 "we are going to get from GPT-3 to something that is getting basically close to ChatGPT"，下一讲 "we will go from ChatGPT to um GPT-01 or, you know, any of the thinking models that we have today"；把左边带到右边的整个过程 "usually people call post-training"。这里的易错点有两个。其一，不要把"基座模型能力强"等同于"可用"：GPT-3 是个 very big, strong base model，但 fine-grained control 基本不可能，可用性受限的不是知识量而是可控性。其二，不要把 instruction following 误认为单纯靠更大规模或更好的 few-shot 示例就能自然出现——讲师明确把杠杆放在 explicit data collection 与 explicit steering 上，而这正是 messiness 的来源。还要注意措辞分寸：他说的是 "getting basically close to ChatGPT"，是"接近"而非"复刻"，细节仍然难；转写里也没有在本段给出任何具体算法或数据配方，不宜提前补写。

## 前沿后训练信息的稀疏性：早期富矿与现代黑箱 [03:27 - 05:42]

讲师为整讲加了一个 caveat（前提性警告）：关于 frontier post-training（前沿后训练）的信息 "honestly pretty sparse in many, many ways"，而且他今天要引用的材料 "a lot of it's going to be actually pretty old"，"before the competition from ChatGPT started to heat up"。他具体点名两篇早期工作：一是 "the RLHF paper, um, learning to summarize from human feedback"（即 Stiennon et al. 2020），评价是 "incredibly detailed"，建议去读附录，因为 "It contains things like detailed annotation guidelines, how they were instructing people to, you know, give feedback to the models"；二是 "Anthropic's HH paper back in 2022"（Bai 2022）。画面 "Caveat: post-training information is pretty sparse!" 一页把这段分成两组小标题："Rich information from the pre-chatgpt competition" 下列 "Stiennon 2020 (RLHF paper) – contains annotation guidelines, etc" 与 "Bai 2022 (Anthropic HH) – details on how to set up safety annotation, etc"；"Poor information in modern times.." 下列 "Open source: lots of distillation, not too public in open-source release notes" 与 "Closed source: 'secret sauce' for data."。注意画面把 Anthropic HH 的作者标为 Bai 2022、年份 2022，而 RLHF 那篇标为 Stiennon 2020，这些作者名与年份是转写里没逐字念出的画面信息。

为什么现代信息变少？讲师说 "now that competition has heated up, basically none of the vendors want to release any information about their post-training processes"，"The data is very much a trade secret"（数据很大程度上是商业秘密），因此 "we are not going to be able to talk about annotation guidelines and things like that for anything that resembles a modern language model"。开源侧虽有 closed-source / open-source models 并存，但 "a lot of the open-source recipes rely on distillation"（大量依赖蒸馏），其性质与前沿实验室做的 "human data collection" "is very different"——后者被他称为 "a very tricky and messy thing"，他会尽量点出一些，但 "be difficult to talk about some of that very comprehensively"。换句话说，这不是"资料还没写出来"，而是竞争导致的系统性信息缺口。

画面右下角嵌着一段 2023 年的内部文档摘录，正是讲师在 [04:54] 说的 "pre-research for this lecture" 里找到的 "very fun example" —— Scale AI 内部文件泄露、里面有 "a whole bunch of very frank conversations about what was happening in post-training data land"。这段小字可以逐句读到：

```text
In July 2023, a manager ordered workers to study GPT-4's responses
closely and figure out why they outperformed Bard's. "Try to come up
with feedback that we can share so that experts can write responses
better than GPT4 or at least the same," the manager wrote.

Scale AI also created a spreadsheet that compared 1,729 Bard rewrites
directly to ChatGPT in October 2023. Each rewrite was rated with labels
like "worse than GPT4" or "Needs some Fixes." In one example, a worker
rewrote a Bard review of a nursery chair that managers stamped "worse
than GPT4" because it "lacks detail compared to GPT4."
```

这段材料里能落地的具体数字与参数有 $1{,}729$ 条 Bard rewrites（改写）、两个时间点 July 2023 与 October 2023，以及 "worse than GPT4"、"Needs some Fixes"、"lacks detail compared to GPT4" 三种评级标签。讲师口述的版本更概括：他们想让 Google Bard 变好，却 "couldn't figure out why GPT-4 was so much better"，于是 "we need to like reverse engineer what's happening"（逆向工程），并要求标注者产出 "responses that are more detailed or and better than GPT-4"；他总结 "these kinds of like competitive dynamics are happening all the time, and that's also why some of this detail is going to be missing from today's lecture"。对照可见，$1{,}729$ 这个数字、"a spreadsheet"、nursery chair（婴儿椅）那个具体例子、"lacks detail compared to GPT4" 这句评语，都是图上独有而转写没念到的内容，是本次详写要补进正文的增量。同时要留意，"better than GPT4 or at least the same" 是文档里经理写的原话，属于当时的工作指令，并不是讲师对技术可行性的判断。

![对比早期详尽的标注指南与现代「秘密配方」的信息稀缺性，并附 2023 年内部文档摘录](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0004_00-04-22.jpg)

尽管信息稀疏，讲师紧接着给出一个对整讲定调的判断："algorithm-wise, we do have a pretty good handle on what is happening, right? But algorithms are not really the secret sauce. It's not where a lot of the leverage is. It's going to be the data." 也就是说，算法层面我们相当清楚，算法不是秘密武器，真正的杠杆在数据。这既解释了为什么接下来要讲一个 "very straightforward recipe"（很直接的两段式配方），也解释了为什么这一讲的绝大部分篇幅会围绕 SFT 数据展开而不是优化方法。易错点有两层：其一，别把"公开材料少"误读成"方法未知"——未知的是数据配方、标注指南与标注流程，不是算法；其二，别把开源配方当作前沿实验室做法的等价替身，蒸馏数据与专家人类数据在来源、成本、偏差结构上都不是一回事，用 open-source release notes 去反推 frontier post-training 会系统性失真。另外，讲师自己说这些早期材料 "actually pretty old"，引用它们是为了说明"曾经有人把细节写出来"，而不是宣称它们描述的就是今天的系统。

## 标准两段式配方与 SFT 的定位：方法几乎等于预训练 [05:42 - 07:30]

从 RLHF 论文里能看到 "a very straightforward recipe"，讲师称之为 "Two-part recipe"。第一部分："you go and collect some demonstration data"——对每个 prompt，"you want some annotator to provide supervised fine-tuning data. So, you know, reference responses for that prompt"，然后做 fine-tuning（微调）。第二部分："Then after you have that, you're going to do some reinforcement learning of some kind to try to shape the behavior of the model to be sort of more aligned or more in line with what sort of humans think are good responses"。把这条链条写成符号，就是：

$$\text{prompt} \xrightarrow{\ \text{SFT}\ } \pi_{\text{SFT}} \xrightarrow{\ \text{RL}\ } \pi_{\text{aligned}}$$

其中 $\pi_{\text{SFT}}$ 表示监督微调后的策略，$\pi_{\text{aligned}}$ 表示强化学习之后的策略；两个箭头分别对应讲师说的第一阶段与第二阶段。这不是画面上印着的公式，而是对 "two-part recipe" 这段口述的紧凑转述，用来固定两阶段的先后关系。讲师明确说 "I'm going to start by talking about this very first phase, which is the SFT part of um post-training"，并提醒 "with a little bit of discussion about the method right after"。

SFT（supervised fine-tuning，监督微调）的定位是本章最要紧的一句话："this is the part that's really basically all about data, right? We all know how to SFT a model. That's basically exactly the same as pre-training. So, the only real difference with some minor variation is going to be the training data." 因此 "I will focus on the training data almost entirely for this part um of the lecture"。这个"方法几乎等于预训练、差别只在数据"的判断，在画面上被做成了版面比例："What are the ingredients in SFT?" 一页上半标题 "The training data" 占了绝大部分面积，左边是 Finetuning tasks 的四个卡片（T0-SF、Muffin、Natural Instructions v2、CoT (Reasoning)），右边是 Open Assistant 网站截图（标语 "We believe we can create a revolution. In the same way that Stable Diffusion helped the world make art and images in new ways, we want to improve the world by providing an amazing conversational AI."，以及 "Try our assistant"、"Help us respond"、"Checkout our HuggingFace organization" 几个入口）；下半标题 "The method" 只放了并排的两个饼图，分别标注 Data Mixture of Stable-Stage 与 Data Mixture of Decay-Stage，扇区里能辨认出 Wikipedia、StackExchange、Common Crawl 一类来源名。"训练数据"占了整页绝大部分、"方法"只占底部一小块，正是那句"几乎全是数据"的视觉版本。[10:36] 那张照片里讲师正站在投影前，屏幕上显示的也是 Open Assistant 页面与爪印气泡图标，右下角有 Stanford 水印，对应 [10:15] 的停顿。

![SFT 的两大成分：训练数据占据绝大部分版面，方法只留下底部两块数据混合饼图](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0006_00-06-51.jpg)

四个任务卡片底部各带一组规模标注，是画面独有、口述没念到的具体数字：T0-SF 标 "55 Datasets, 14 Categories, 192 Tasks"，Muffin 标 "69 Datasets, 27 Categories, 80 Tasks"，CoT (Reasoning) 标 "9 Datasets, 1 Category, 9 Tasks"，Natural Instructions v2 一栏同样列着 372 Datasets 与上千量级的 Tasks（幻灯片字号很小，个别类别数不易辨认，故不强行抄录）。卡片正文里还列着任务类型，例如 T0-SF 下的 "Commonsense reasoning"、"Question generation"、"Closed-book QA"、"Adversarial QA"、"Extractive QA"、"Entailment generation"、"Topic classification"、"Struct-to-text"；Muffin 下的 "Natural language inference"、"Code instruction gen."、"Program synthesis"、"Dialog gen."、"Code repair"；CoT 下的 "Arithmetic reasoning"、"Explanation generation"、"Commonsense Reasoning"、"Sequence composition"、"Implicit reasoning"。这些 $\text{Datasets}/\text{Categories}/\text{Tasks}$ 三级计数很关键：早期 SFT 数据是按"任务集合"组织的，而不是按"对话"组织的，几十个 Datasets、上百个 Tasks 的规模感，正是下一章 FLAN "规模假设"的量化背景。

讲师随后预告了这一部分的路线："We will go through the data. We're going to, you know, look at some examples. I'll talk about some historical developments and what kind of the trends are at least in open-source post-training data land. Um and then I will talk about some pitfalls, some ways in which collecting this kind of data is much harder uh than you think." 他还提醒上过 224N 的同学会发现 "very reminiscent of uh issues that you see when you start building, you know, classic supervised deep learning models from scratch"（很像从零搭建经典监督深度学习模型时碰到的那些问题）。易错点在于：因为 SFT 与 pre-training 的目标形式几乎一样，很容易把"换个数据集"当成轻活，但讲师反复强调 messy engineering 恰恰在数据侧，标注质量、分布、偏差这些 224N 式的经典坑会原样回来。另一个易错点是别把第二阶段 RL 当成 SFT 的"精修小补丁"：两段式里 SFT 提供可采样的初始策略，RL 用偏好信号继续塑造行为，分工不同；本段转写并没有给出 RL 的算法细节或损失函数，不宜提前补写 PPO、奖励模型之类的内容。

## 开源世界 SFT 数据的演进脉络 [07:30 - 11:24]

讲师 "take a moment to pause" 来讲 "the progression of SFT data over the years"，并说明排列方式："I think I'm going to go, you know, top left to bottom right. That's roughly my chronological ordering"——大致按时间顺序，而不是严格年代考证。他补充说开源语言建模社区做了 "a lot of different tricks"，目的有两个：既想 "improve system performance"，也想 "understand what factors of this kind of data are useful to train models"。画面 "Progression of SFT data (in the open world)" 用两行四列摆了 8 个图标，按讲师口述的顺序可串成：

$$\text{FLAN} \to \text{Self-instruct} \to \text{Alpaca} \to \text{ShareGPT/Vicuna} \to \text{Open Assistant} \to \text{WizardLM} \to \text{Tulu3} \to \text{Nemotron\ (Tool\ use\ etc.)}$$

这不是幻灯片上印着的公式，而是把 8 个画面标签按口述的时间顺序连起来。图里还有不少转写没念到的细节：FLAN 那一格嵌着 T0-SF / Muffin / Natural Instructions v2 / CoT (Reasoning) 的表格缩略图；Self-instruct 那一格是论文首页 "SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions"，作者列 Yizhong Wang、Yeganeh Kordi、Swaroop Mishra、Alisa Liu、Noah A. Smith、Daniel Khashabi、Hannaneh Hajishirzi，机构含 University of Washington、Johns Hopkins University、Allen Institute for AI、Arizona State University，并给出邮箱 yizhongw@cs.washington.edu；Alpaca 那一格是 "Stanford Alpaca" 字样加羊驼头像；ShareGPT/Vicuna 是绿色 ShareGPT 图标；Tulu3 那一格是 "Tülu 3: Pushing Frontiers in Open Language Model Post-Training" 的论文标题页；Nemotron 那一格是 NVIDIA 的 Nemotron Nano 2 论文标题，下方写着 "Tool use etc."。

![开源世界 SFT 数据的演进时间线：从 FLAN、Self-instruct 到 Tulu3 与 Nemotron 工具调用](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0007_00-08-06.jpg)

各阶段的想法可以这样串起来。最早且 "in some ways the most visionary" 的是 FLAN："This was used to train the T5 model by Google"，rough idea 是 "NLP people have already collected a bunch of supervised data sets of like inputs and outputs. We should just train the model on all of these"——把所有下游任务收拢起来一起训练，讲师评价 "Reasonable thing to try"，但 "Turned out for various reasons this was not the right thing to do. I'll talk about that later."。接着是 Self-Instruct 及其后续，"very forward-looking in terms of its thinking"："why can't we use the model itself to generate data?"——模型一直在变强，"in fact they might even be better than some of our annotators"，那就让模型 "write very very good high-quality responses to our inputs"。随后是 "a bunch of sort of distillation style approaches"（蒸馏式路线）："Alpaca is one of them"，"The Berkeley folks had one called Vicuna in which they were using sort of online user shared prompts as sort of the way to as the inputs to distill on"（用线上用户分享的 prompt 作蒸馏输入）。再之后是 "other efforts that were more entirely human driven like Open Assistant"，讲师把它比作 Wikipedia："It was a bunch of people getting together to write prompts and responses"。然后是 "a new generation Wizard LM and Tulu 3 and others"："we know that language models are very good synthetic data generators. Why don't we come up with increasingly complicated ways of generating instruction following data using language models."。最后 "SFT and sort of post-training has changed yet again in recent years because I think the focus is less on chat and much more on agents and tool use"，于是出现 "a new generation of sort of SFT pipelines that are designed to generate sort of tool use and agentic data"。

[10:02] 之后讲师补了一句关键限定：在 closed-source world 里公司之间可能也在互相蒸馏，但 "where you're also collecting human data, you know, how you're collecting the human data is a big important ingredient that's not reflected in any of this discussion"——整条开源时间线里完全没有体现"人类数据怎么收"这个重要成分。[10:15] 他停下来接受提问，[10:27] 学生问 post-training 是否需要 input-output pairs，回答 "That's right. You need input-output pairs."；[10:32] 学生追问 "does the correctness of those input-output matters like how far"，讲师把它复述成 "how much does the correctness of the input-output pairs matter?" 并给了一个 "a nuanced question" 的回答：顶层结论是 "you should go collect the highest quality responses you can because the bad stuff will teach the model to do bad stuff"；但另一方面 "it turns out that you can SFT models on all sorts of strange things and they will learn to follow instructions"，他还提到 "Percy's former student had a paper in which you're like training uh models without even the responses or something"，模型照样能学会 instruction following，因此 "there's a lot of like pre-training generalization behaviors that let you get away with worse quality data"。这段问答把"数据质量"这个变量提前摆上台面，为下一章 FLAN 的质量问题埋下伏笔。

这条时间线同时是后面几章的目录：FLAN（多任务、从既有 NLP 数据集构造）、Alpaca（蒸馏 ChatGPT）、Open Assistant（众包人类数据）、Nemotron（工具调用）都会被逐一展开成 random examples 幻灯片。易错点有三：其一，不要把 "chronological ordering" 当成严格因果链，讲师用的是 "roughly"，而且他说这些工作里很多 tricks 是为了做消融、理解"哪些数据因素有用"，并非一条线性改良路线；其二，不要因为开源时间线有 8 个节点就以为前沿实验室走的是同一条路，讲师明确说闭源的人类数据收集 "not reflected in any of this discussion"；其三，关于 correctness 的问答不要简化成"质量无所谓"——原话是应尽量收集最高质量，只是预训练泛化让模型对低质量数据有惊人容忍度，两者并存而非互相否定。

## FLAN：多任务后训练的雄心与「规模假设」的落空 [11:24 - 14:20]

讲师 "start with FLAN"，评价它是 "very much like a forward-looking uh data set for its time"，因为它确立了 "doing this like multitask post training"（多任务后训练）的思路："Just if you're going to do downstream tasks, go collect all the downstream tasks and train on it. Very reasonable but smart idea at the time." 但 "if we look at what's in FLAN, we actually find that, you know, it's a little bit of a strange data set"。画面 "FLAN – random examples" 是一张左 input、右 output 的对照表，可以逐条转述。第一条输入是一封邮件："Stephanie - Can you finalize the attached and have it signed. I need to initial it, but it needs to be signed by Brad Richter. Thanks. Write a subject line for this email."，右侧输出只有 "Ronald Chisholm LOI"。讲师口述描述的正是这种结构："here's an email" 后面 append 一句 "write a subject line for this email"，然后主题行放在右边——"I don't think you've ever probably prompted, you know, uh model to do a task like this, let alone the instructions at the very end"。注意指令被拼在邮件正文末尾而不是开头，这种 unnatural 的结构本身就是问题。

第二条是分类任务，画面原文为 "Ahold to Sell Spain Operations to Permira (AP) AP - The Dutch supermarket retailer Ahold, seeking to streamline global operations and reduce debt, said Sunday it will sell its holdings in Spain to Permira Funds for about #36;849 million. What is this text about? OPTIONS: - World - Sports - Business - Science/Tech"，输出 "Business"（画面上 `#36;849 million` 是未渲染的字符转义，按原样保留）。第三条最能说明问题，指令是 "#36;Write highlights for this article:" 后接一大段关于 The Hague（海牙）的游记文字（"Sauntering down leafy avenues past typical Dutch step-gabled houses"、the Widow of Indonesia、Vermeer's 'Girl With A Pearl Earring'、Panorama、Hendrik Mesdag、Simonis、Remco Dörr、www.carlton.nl/ambassador、cost from £84 等），而右列输出是：

```text
The least known of the Dutch cities, The Hague was a village
until 1806. It owes its growth to Louis Bonaparte, Napoleon's
brother, who ruled here. The city has a wealth of art,
including Vermeer's 'Girl With A Pearl Earring'
```

把左右两栏对照，就能验证讲师的两句话："compared to something like ChatGPT, the summaries are very short"，以及 "if you look really closely, the summaries are often hallucinated. Like there's a bunch of details that are not in the inputs."——"until 1806"、"Louis Bonaparte, Napoleon's brother" 这些细节并不在左边输入文章里。第四条是结构化数据转句子，画面原文可直接当数据块读：

```text
Here is some data about a restaurant: name = Aromi, eatType = coffee shop,
food = English, customer rating = 5 out of 5, area = city centre.
Write a sentence that includes the following data about a restaurant
```
输出 "In the city centre there is a coffee shop with a customer rating of 5 out of 5 called Aromi which serves English food."。这里的 $5\ \text{out of}\ 5$、`OPTIONS` 里的 $4$ 个候选类别、以及 `name = Aromi` 这样的 key-value 形式，都是画面里能落地的具体参数；而后一条的输出只有一两句，与前一条的长文章形成鲜明反差。

![FLAN 随机样本对照表：邮件主题行、文本分类、海牙游记摘要与餐厅结构化数据转句子](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0010_00-11-51.jpg)

病根在于数据来源。讲师解释："FLAN is generated from existing data sets"——有个数据集叫 Enron emails，"This is basically a way to turn Enron emails into a supervised task, right? Because it has a body and a subject, you can make this subject prediction task"；摘要任务同理，"I don't know if this was taken from CNN Daily Mail or Exxon, but it was taken from one of the big major summarization data sets, right? Write highlights for this article, take the article part of a summarization data set, put the summary summary on the right." 因此 "the original NLP data sets from which FLAN was built was not the highest of quality"，训练时 "you also inherit a whole bunch of sort of deficiencies from those data sets"，而且 "the unnaturalness of the structure of these data sets also sort of carry down all the way"。讲师在 [13:31] 特意 "going back one slide" 补了构建时的理论背景："I think when FLAN was originally constructed, sort of part of the theory of how post-training should work was that you needed scale"——既然 "in pre-training we know we need a lot of scale"，就推断后训练也需要 "a ton of scale to unlock capabilities"，于是 "they took tons of data sets and lots of data and had this gigantic data set"。这与上一章 Finetuning tasks 卡片上几十个 Datasets、上百个 Tasks 的三级计数正好呼应：那是"规模假设"在数据组织上的具象化。

后来的经验推翻了这个假设："We'll see kind of later on that a lot of these data sets just get really much smaller and it still works." 讲师给出的机制是："if you have a sufficiently strong big pre-trained model, you can actually get away with very few high-quality examples because the pre-training generalization is going to get you quite a bit of the way"，因此 FLAN "is kind of the wrong point in some ways on the quality quantity trade-off"（在质量—数量权衡上押错了方向）。但他也补了句公道话："although you know, there was no way to know until we sort of explored the full space of these data sets." 易错点有两层：不要把 FLAN 的问题归因为"多任务本身错了"，讲师仍称它 "a very smart idea to create this big data set out of these uh data set existing benchmarks"，真正的问题是源数据质量、结构 unnaturalness 与当时对规模的假设三者叠加；也不要把"少量样本够用"推广成"样本质量不重要"——[10:40] 的问答已经说过 "the bad stuff will teach the model to do bad stuff"，预训练泛化只是提供了缓冲。下一章的 Alpaca 正是"数据变小、变自然"之后的对照组。另外，讲师说 "I don't know if this was taken from CNN Daily Mail or Exxon" 时自己也不确定数据集出处，笔记里不应替他指定是哪一个。

## Alpaca 的蒸馏与 Open Assistant 的众包：数据变自然、变详细 [14:20 - 16:50]

"after ChatGPT came out, some of my students you know, did this thing called Alpaca where they distilled ChatGPT traces to get input-output pairs"。画面 "Alpaca – random examples" 的三行表格能直接看出数据形态的变化："You get, you know, more natural-looking inputs, longer sort of chatty outputs because of course it's it's taken from ChatGPT." 第一条 "Give three tips for staying healthy." → "1. Eat a balanced diet and make sure to include plenty of fruits and vegetables. 2. Exercise regularly to keep your body active and strong. 3. Get enough sleep and maintain a consistent sleep schedule."；第二条 "What does "algorithm" mean?" → "An algorithm is a step-by-step set of instructions used to solve a problem or achieve a goal. It is a process of breaking down complex tasks into smaller, more manageable steps. Algorithms can be used to find the shortest path from one place to another, sort data, or render data from an immense database in a few seconds."；第三条 "Find the average number in a list." 给出一段压成一行的 Python 代码：

```python
def avg_list(nums): total = 0 for n in nums: total += n return total / len(nums) print(avg_list([4, 3, 6, 10, 8])) # Output: 6.4
```

其中标识符 `avg_list`、`total`、`len(nums)`，数值 $[4, 3, 6, 10, 8]$ 与输出 $6.4$ 都是画面原文；幻灯片把整段代码排成一行，语法上并不分行，这里按原样保留。这类样本 "now reliably induced kind of ChatGPT-like behavior on models"，但 "Importantly, you know, only when we did it to the original Llama models. So pre-training, post-training, you know, both of them need to work out right"——预训练与后训练两边都得成立。讲师说这是 "the point at which these things started to work"，人们由此意识到 "if you got this kind of sort of chat style data, it was actually not that hard to build chat GPT style systems"，但立刻补上限定："Getting the details right is still hard, but you know, to get most of the way."。

[15:33] 之后是 "a big push to create essentially the highest quality human driven version of this these kinds of data sets"。背景是 ChatGPT 与 Alpaca 之后的 "enormous optimism"："if only we could collect a sufficiently high quality and large, you know, instruction tuning data set, we could then sort of catch up and match a lot of the performance of these big closed source labs"，于是 "there's a lot of sort of energy and excitement uh to building these data sets"。Open Assistant 就是 "this big crowd sourced effort to build such a data set"："a lot of volunteers got together and they said, well, we're going to, you know, come up with really hard and interesting prompts. We're going to come up with good high quality responses and we're going to do it at scale in the same way that Wikipedia, you know, managed to produce something very high quality at scale." 规模上讲师回忆 "a decent amount of data. I forget if it's like 10,000 or more examples um before sort of stalling out as as a project"，即约 $10{,}000$ 条以上后项目逐渐停滞。数据形态是 "chat style inputs, very long detailed sort of high quality expert responses on the right. Good stuff."

![OpenAssistant 随机样本：带「Monopsony」定义与 Bivens & Mishel 参考文献的长回答，以及家庭科学项目清单](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0013_00-15-36.jpg)

"OpenAssistant – random examples" 表格里有一条与后面幻觉讨论直接相关的例子。左列是 "Can you write a short introduction about the relevance of the term "monopsony" in economics? Please use examples related to potential monopsonies in the labour market and cite relevant research."；右列回答先定义 ""Monopsony" refers to a market structure where there is only one buyer for a particular good or service"，再讲劳动力市场里 monopsony employer "has significant power over the wages and working conditions of their employees"，最后附上参考文献：

```text
References: Bivens, J., & Mishel, L. (2013). The Pay of Corporate
Executives and Financial Professionals as Evidence of Rents in Top 1
Percent Incomes. Journal of Economic Perspectives, 27(3), 57-78.
```

另一条 "My kid needs to do a science project for elementary school, what are some cheap but fun projects to do?" 的回答列了 7 条家庭科学项目（homemade lava lamp 用 plastic bottle、water、vegetable oil、food coloring 与 Alka-Seltzer tablet；solar oven 用 pizza box、aluminum foil、plastic wrap；还有 "Explore magnets" 等），并收尾说 "These are just a few ideas to get you started!"。可以看到回答里既有编号列表、又有 "References:" 段、还有 "1." 到 "7." 的项目罗列——这些正是最后一章要讲的 length 与 style variation 的具体载体，也是画面上 "27(3), 57-78" 这类参数能落在实处的例子。

Alpaca 与 Open Assistant 是同一场"数据乐观主义"下的两条路线：一条靠蒸馏强模型轨迹，一条靠人类专家众包。前者说明 chat style data 足以诱导出 ChatGPT 式行为，后者说明社区愿意为"更高质量"付出巨大组织成本。易错点有三：其一，"only when we did it to the original Llama models" 容易被读成"Alpaca 数据本身有魔力"，讲师的意思恰恰是脱离基座谈数据会误判因果；其二，"10,000 or more examples" 是带 "I forget" 的不确定回忆，不是精确统计，不宜当作权威数字引用；其三，Open Assistant 的高 detail 回答虽被讲师称 "Good stuff"，但他紧接着说 "I'll talk about some of the pitfalls later as well"——细节与引用既是优点也可能是幻觉来源，本段只埋伏笔，不宜提前下结论。至于 Alpaca 那张表格（14:21）与 Open Assistant 表格（15:36）在画面上属于两页不同的 random examples，本段把它们的内容并列转述，是为了对照"蒸馏数据"与"人类专家数据"的形态差异。

## 从聊天到智能体：工具调用数据与三个演变维度 [16:50 - 18:55]

讲师说 "I've taken this from some of the newest of the instruction tuning data sets"，领域已经 "move as a field or, you know, in terms of the products that people want from just a chat interface to something that's like a full agent system"："So we don't want just textual responses, we want tool calls, we want, you know, uh to-do lists. If you've ever used Claude code or Codex, you know that like it produces a to-do list to like check off as it goes. Um, we want all of those things." 以 Nvidia 的开源努力 Nemotron 为例，"so that's Nvidia's open source efforts, their SFT data, uh, a big chunk of it is these kinds of agentic SFT, uh, examples"：除了 "the responses, which are provided in the sort of, uh, assistant field here"，还有 "tool calls that can happen in parallel alongside the text"，而且 "this is SFT data, so this is, you know, explicitly sort of supervised into the model"，结论是 "now we're sort of moving much more towards these like structured, uh, formats"。画面 "Nemotron-SFT-OpenCode-v1" 表格给出的正是结构化样本，右侧第一段可原样读到：

```json
{"role": "assistant", "content": "I'll analyze the key differences between `async/await` and `.then()` for handling promises in JavaScript based on the provided AGENTS.md instructions.", "tool_calls": [{ "id": "call_132a55728f6c498fbeac3b95", "type": "function", "function": { "name": "skill", "arguments": "{\"name\":\"bash-skills\"}" } }]}
```

第二个样本更完整地展示了 to-do list，也是画面上信息量最大的一段：

```json
{"role": "assistant",
 "content": "I'll help you decide between integers and floating-point numbers for your calculator app. Let me analyze the trade-offs and provide a recommendation based on the requirements in AGENTS.md.",
 "tool_calls": [{"id": "call_b06182c280514fc7bd1d3ae4", "type": "function",
   "function": {"name": "todowrite", "arguments": "{\"todos\":[{\"content\":\"Review the AGENTS.md file for calculator data typerequirements\",\"id\":\"1\",\"priority\":\"high\",\"status\":\"in_progress\"},{\"content\":\"Analyze trade-offs between integers and floating-point numbers\",\"id\":\"2\",\"priority\":\"high\",\"status\":\"pending\"},{\"content\":\"Check available skills for relevant guidance\",\"id\":\"3\",\"priority\":\"medium\",\"status\":\"pending\"},{\"content\":\"Provide recommendation for beginner-friendly calculator\",\"id\":\"4\",\"priority\":\"high\",\"status\":\"pending\"}]}"}}]}
```

能落地的字段与参数包括：`role`、`content`、`tool_calls`、`type: "function"`、`function.name`（`skill` 与 `todowrite`），以及 `todos` 数组里 $4$ 个条目各自的 `id`（`"1"` 到 `"4"`）、`priority`（`high` / `medium`）与 `status`（`in_progress` / `pending`）；左列对应的问题是 "What are the key differences between using `async/await` and `.then()` for handling promises in JavaScript, and when should I prefer one approach over the other in terms of readability and error handling?" 与 "I'm building a simple calculator app and need to decide between using integers or floating-point numbers for all calculations. What are the trade-offs of each approach, and which would you recommend for a beginner-friendly calculator that handles basic arithmetic operations?"。`AGENTS.md`、`bash-skills`、`todowrite`、`call_132a55728f6c498fbeac3b95` 这类标识符转写里一个都没念到，全是画面独有信息：它说明监督信号不再只是一段文本，而是"文本 + 并行工具调用 + 结构化待办状态"的复合对象，且 `id` 与 `status` 这类字段把"过程"也纳入了监督。

![Nemotron-SFT-OpenCode-v1 的 agentic SFT 样本：assistant 文本与并行 tool_calls、todos 的 JSON 结构](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0014_00-16-51.jpg)

[17:51] 之后讲师把高层变化 internalize 成几条 perspectives，说 "It's useful to roughly internalize these sort of like perspectives"。第一条是 "this chattiness thing"："it's hard for me to like fully articulate what this is, but there is a sense in which the classic NLP data sets, even the high-quality ones, were very like input to like programmatic output almost, right? Um, but people don't really want to talk to an NLP benchmark. They want to talk to people."，于是 "there was this big shift that has happened towards more detailed human-like responses"。第二条是 "a big shift towards higher-quality annotators and much more detail. Open Assistant is a good example of this, of having experts write your responses."。第三条是 "tool use and figuring out the right sort of, you know, interface and API for all of this has been a big, sort of final shift"。画面 "What's varying across these?" 一页把这三条写得更具体，且是转写没逐字念到的原文：""Chattiness" – FLAN is (usually) valid data, but people don't want to talk to a NLP benchmark. Later datasets move towards longer, more detailed responses"；"Detail – OASST goes into much more detail about various factual pieces of knowledge. As we will see, this can be both a pro and a con."；"Tool use – SFT in the last year or two has also been shifting much more towards tool/use, agentic downstream applications."。注意画面用了缩写 OASST 指 Open Assistant，并把 detail 明确标成 "both a pro and a con"；讲师收尾时还说 "I'm sure things will keep changing, but these are kind of some high-level, you know, factors".

第四章的时间线在这里收束为三个维度：chattiness 对应 FLAN → Alpaca → Open Assistant 的回答变长变像人，detail 对应专家标注与知识展开，tool use 对应 Nemotron 这类结构化 SFT。易错点在于：不要把 "structured formats" 只理解成输出格式变漂亮——`tool_calls` 与 `todos` 是被 "explicitly sort of supervised into the model" 的行为，意味着数据构造要同时定义接口（API）与状态字段，讲师明确把 "figuring out the right sort of interface and API" 本身算作一大转变；也不要把 chattiness 当成可精确定义的指标，他自己说 "hard for me to fully articulate"，那只是一个方向性判断。把这三代数据放在同一个坐标系里看，变化其实发生在"监督对象"的层面：FLAN 与 Alpaca 监督的是"给定输入后该输出什么文本"，Open Assistant 监督的是"输出该多详细、带不带引用"，而 Nemotron 监督的是"在输出文本的同时该发起哪些工具调用、待办事项各自处于什么状态"。因此讲师说 "we're sort of moving much more towards these like structured, uh, formats" 时，重点不只是排版从自由文本变成 JSON，而是数据构造者必须先定义好 `tool_calls`、`todos`、`priority`、`status` 这一整套接口契约，合成流程才能按契约产出样本；这也解释了为什么他把 "figuring out the right sort of interface and API" 单独算作一大转变——接口一旦定下，模型的行为空间就被这层字段限定了，而前面几章的数据集里并不存在这样一层结构。回看 [17:00] 那句 "from just a chat interface to something that's like a full agent system"，可以发现整段演进其实是把越来越多原本属于"产品层"的东西（待办清单、工具调用、状态字段）下沉到 SFT 数据里，让它们变成可被监督的目标，而不是留到推理时再靠提示词临时约束。

另外 "last year or two"、"much more towards" 都是趋势描述而非时间节点，画面上并没有给出具体年份，不宜自行补年份或补写任何评估数字。

## 收集 SFT 标注数据时要盯住的变量 [18:55 - 19:58]

[19:09] 讲师转入新话题："let's say you are put in charge of collecting human annotation data for SFT. Um, what kinds of sort of things should you be mindful of, right?" 他一口气列了四类要考虑的事："lots of formatting things, right, that you might want to think about"、"things like knowledge. How much knowledge should be put into the responses? How many how much data points do you want?"、以及 "how do you deal with kind of uh safety of your model? Like what kinds of safety data should you be collecting?"。画面 "What did we notice across the datasets?" 把这些差异列成一张清单，原文是：

```text
These datasets vary in many ways...
  - Length and bullet points (style variations)
  - References, other complex knowledge
        Less visible, but important aspects..
  - Scale
  - Safety
How do these factors affect the model?
```

其中 "Length and bullet points (style variations)"、"References, other complex knowledge"、"Scale"、"Safety" 与 "Less visible, but important aspects.." 都是画面原文。与转写对照可以看出，口述里的 formatting / knowledge / data points / safety 分别对应画面里的 style variations / complex knowledge / Scale / Safety；而 "How do these factors affect the model?" 把本段收束成一个待回答问题，正好接上 [18:56] 预告的 "I'll talk about, sort of, all sorts of, you know, pitfalls or challenges in a moment"。

被特别强调的是 length 与 style："one thing I want to highlight is length is and style variation in general is a very, very big part of this like post-training process"。例子是产品层面的语气差异："People say, you know, Claude has a different tone than ChatGPT, you know, ChatGPT is too chatty, so on and so forth."，紧接着给出关键判断："All of these are, you know, conscious decisions that are made by the data collection folks."（这些都是数据收集者有意识的决定）。把它写得形式化一点，就是数据侧的风格决策决定了模型的输出行为：

$$\text{data collection decisions} \;\longrightarrow\; (\text{length},\ \text{bullet points},\ \text{tone}) \;\longrightarrow\; \text{model behavior}$$

这条链不是画面上印着的公式，而是对 "conscious decisions that are made by the data collection folks" 与 "These datasets vary in many ways..." 两句的紧凑转述。风格差异在前面几章里其实是可枚举、可计数的：FLAN 的输出是 "Ronald Chisholm LOI" 这样的极短标签，Alpaca 的输出是 "1. Eat a balanced diet... 2. Exercise regularly... 3. Get enough sleep..." 这样的 $3$ 条编号 tips，Open Assistant 的输出是带 "References:" 段与 $7$ 条家庭科学项目罗列的长回答，Nemotron 的输出是带 $4$ 个 `todos` 条目的 `tool_calls` JSON。也就是说，$1$ 个词的主题行、$3$ 条 tips、$7$ 条科学项目、$4$ 个待办条目，这些具体计数本身就是 length / bullet points 维度上的数值证据，而不是泛泛的"有的长有的短"。

![跨数据集差异清单：长度与项目符号、引用与复杂知识、规模与安全，以及「这些因素如何影响模型」的追问](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0016_00-19-21.jpg)

这一章是整段 00:00–19:58 的收口，也是讲师预告 pitfalls 之前的最后一问。前面所有 random examples 幻灯片的差异，到这里被抽象成四类需要人为决策的变量：格式（formatting / style variations）、知识量（knowledge / references）、数据规模（data points / Scale）、安全（safety data）。其中 "Scale" 直接回扣第五章里 FLAN 的"规模假设"与"少量高质量样本就够用"的反例，"References, other complex knowledge" 回扣第六章里那条附 Bivens, J., & Mishel, L. (2013) 参考文献的 monopsony 回答，"Length and bullet points" 回扣第七章里 Nemotron 的 `todos` 数组。画面上还特别把 Scale 与 Safety 归到 "Less visible, but important aspects.."（较不显眼但重要的方面）之下：与长度、项目符号这类一眼可见的差异不同，规模与安全数据要翻统计或标注指南才看得出来，这与第二章说的"现代 open-source release notes 不公开这些细节"形成呼应，也解释了为什么讲师在整段开头就先声明信息稀疏。

最需要小心的是把风格差异误当成能力差异：讲师只说这些是 "conscious decisions"，本段范围内并没有给出"哪种风格更好"的结论，也没有给出任何评估表或 benchmark 数字——最后一张幻灯片 kf_0016 只有上面那几条 bullet 和一个问句，没有任何数值，因此不宜在这里推断 Claude 与 ChatGPT 谁更优，也不宜补写准确率或偏好指标。第二个局限是 "How much knowledge should be put into the responses?" 在此只是被提出问题，讲师没有回答；第七章画面上的 "Detail... can be both a pro and a con" 也只留了伏笔，本段不宜替他给答案。最后，"how many data points do you want" 同样只有提问：结合第五章的 "very few high-quality examples" 与第六章那个带 "I forget" 的 $10{,}000$ 条回忆，能确定的是规模没有统一答案，而不是某个具体条数。转写在本段最后一刻停在 "All of these are, you know, conscious decisions that are made by the data collection folks"，之后的 pitfalls 讨论不在这一段范围内，笔记到此为止，不越界补写。

## 风格、长度与偏好评估失真 [19:58 - 21:59]

本段承接上一段对不同指令微调数据集统计特征的观察：讲者在 [20:00] 首先指出，如果横向比较这些数据集，回答长度（length of the responses）存在「very wide variation」，差异极大。这一差异并非无害的表面现象，因为一旦进入偏好评估（preferences / preference evaluation）的环节，风格因素（stylistic factors）会「matter a ton」。所谓风格，指的不是答案对不对、推理对不对，而是答案怎么呈现：是否带项目符号列表（bullet-pointed lists）、是否更长、是否更详细。讲者说，人类评估者会「very often select responses that have bullet-pointed lists」，也会倾向选择「more and longer detail」的回答；他特意澄清这未必是被「tricked」，因为在把两个回答并排摆在人面前、问「which one's better」的 pairwise 评估设置里，更详细、更像列表的回答通常确实更好，这种偏好非常自然。但代价是，它会「induce very explicit distortions in the kinds of tone that a chatbot has, relative to people」，也就是把聊天机器人的语气系统性地推向话痨、爱列点、爱铺陈，而这和真人说话的方式并不一致。

![偏好评估中的风格效应：横轴分别为对列表与对更长输出的偏好百分比，含人类与GPT评估散点，引自Dubois+2023](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0017_00-20-35.jpg)

幻灯片把这一效应量化成了两幅散点图，这些细节是转写没有念出的。标题是「When evaluating by preferences, style matters.」，副标题是「We see very strong length effects (in both humans and GPT-based evaluations)」，即长度效应在人类评估和 GPT 作为评估者时都同样强烈。上图横轴为 `Preference for lists (%)`，刻度从 30、40、50、60 到 70；下图横轴为 `Preference for longer outputs (%)`，刻度为 25、50、75，数据点大量落在 50% 以上，说明多数评估配置明显偏爱列表与长回答。图例分成两行：`Annotator` 一行给出 Human $p_{ref}$、Trainer $p^{ann}_{sum}$、Evaluator $p^{eval}_{sum}$、GPT4 $p^{GPT4}_{sum}$ 四种标注来源；`Model` 一行用不同形状区分 Human $p_{ref}$、Simulated $p_{sum}$、GPT4、ChatGPT、Davinci003。右下角标注引用为 `[Dubois+ 2023]`。换句话说，风格偏差不是某一类评估者独有的毛病，而是从人类到 GPT-4、从被评模型到模拟打分器都普遍存在的现象。

可以把这种污染形式化地写成一个带风格项的打分函数，用来说明「偏好分」与「能力分」并不是同一个量：

$$P(\text{chosen}=y_a\mid y_a,y_b)=\sigma\big(s(y_a)-s(y_b)\big),\qquad s(y)=\underbrace{s_{\text{cap}}(y)}_{\text{能力}}+\underbrace{\alpha\,\ell(y)+\beta\,\mathbf{1}[\text{has bullets}]}_{\text{风格}}$$

其中 $\ell(y)$ 是回答长度，$\mathbf{1}[\text{has bullets}]$ 指示是否含项目符号列表，$\alpha,\beta>0$ 就是风格偏置。只要 $\alpha$ 或 $\beta$ 不为零，偏好评估测到的就混入了风格，而非纯粹的 $s_{\text{cap}}$。这也解释了讲者在 [21:10] 的警告：如果盯的是互动信号（engagement signals），而大多数公司确实盯这个，就「very, very easy to fool yourself into thinking you're getting better data, when in reality your model's capabilities are not changing」。不同后训练数据集训练出来的模型，在偏好指标上差异巨大，尤其讲者点名的右列 AlpacaEval；但「it doesn't necessarily change standard benchmarking evaluations」，模型并没有因此变聪明。于是他给出的处方是把风格控制（style control）与能力控制（capabilities control）分开来想，用一句话说就是：不要拿偏好胜率当能力指标。

![不同指令微调数据集在MMLU、GSM、BBH、TyDiQA、Codex-Eval与AlpacaEval上的Table3对比，蓝色为提升、橙色为下降](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0018_00-21-50.jpg)

紧接着的幻灯片正是这一论断的表格证据，标题「What about benchmarks?」，副标题「These factors are (mostly) not that relevant for other benchmark perfs」。它给出的是 Table 3，说明文字写明：不同 instruction-tuning 数据集各有所长、而混合数据（mixtures）平均表现最好；单元格为蓝色表示该微调相对 vanilla LLaMA 有提升，橙色则表示反而拉低了性能。列依次为 MMLU (factuality) 的 EM (0-shot)、GSM (reasoning) 的 EM (8-shot, CoT)、BBH (reasoning) 的 EM (3-shot, CoT)、TyDiQA (multilinguality) 的 F1 (1-shot, GP)、Codex-Eval (coding) 的 P@10 (0-shot)、AlpacaEval (open-ended) 的 Win % vs Davinci-003，以及 Average。把这些数字原样列出来，能直观看到「偏好高」与「能力高」的错位：

```text
Vanilla LLaMA 13B   MMLU 42.3  GSM 14.5  BBH 39.3  TyDiQA 43.2  Codex-Eval 28.6  AlpacaEval   -    Avg   -
+Open Assistant 1   MMLU 43.3  GSM 15.0  BBH 39.6  TyDiQA 33.4  Codex-Eval 31.9  AlpacaEval 58.1  Avg 36.9
+GPT4-Alpaca        MMLU 46.9  GSM 16.5  BBH 38.8  TyDiQA 28.3  Codex-Eval 36.6  AlpacaEval 63.1  Avg 37.6
+ShareGPT           MMLU 49.3  GSM 27.0  BBH 40.4  TyDiQA 30.5  Codex-Eval 34.1  AlpacaEval 70.5  Avg 42.0
+SuperNI            MMLU 49.7  GSM  4.0  BBH  4.5  TyDiQA 50.2  Codex-Eval 12.9  AlpacaEval  4.2  Avg 20.9
+Human+GPT data mix MMLU 49.3  GSM 40.5  BBH 43.3  TyDiQA 45.6  Codex-Eval 35.9  AlpacaEval 56.5  Avg 45.2
```

这里有几个容易读错的地方值得点明。其一，AlpacaEval 最高的几行（ShareGPT 70.5、GPT4-Alpaca 63.1、Open Assistant 58.1）并不等于 Average 最高，Average 最高的是 `+Human+GPT data mix.` 的 45.2，它的 AlpacaEval 反而只有 56.5；其二，颜色编码不能读反，`+SuperNI` 在 GSM/BBH 上掉到 4.0/4.5 是橙色（受损），而它在 MMLU 上 49.7 是蓝色（提升），同一行可以同时有得有失；其三，表格说明里「mixtures perform best on average」是平均意义上的结论，不代表混合数据在每一列都最优。讲者在 [20:55] 被学生问到「how do complex models differ in the data sets」时只答「I'll talk about that later」，所以本段并不展开模型差异，这一章的落点始终是评估方法的失真，而不是模型架构。

## 引用、知识与事实性：OpenAssistant 的双重教学 [22:00 - 24:48]

讲者转入第二个主题时特意说，这在「how we collect and think about post-training data」上「in many ways, important」。他选择的样本是 OpenAssistant——一个「intended to be the highest-quality, in some ways, post-training data」的众包项目，正因为它标榜最高质量，它暴露出的问题才格外有说服力。转写里讲者只是含糊念到「this reference to an article, this J. Michelle L., The Pay of Corporate Executives and blah, blah, blah」，把作者和标题都念得走了样；而幻灯片给出了完整、可核对的原文。标题是「References, complex knowledge, and factuality」，引导语「Consider this example from open assistant:」。左侧 prompt 是：「Can you write a short introduction about the relevance of the term "monopsony" in economics? Please use examples related to potential monopsonies in the labour market and cite relevant research.」右侧回答先解释 monopsony（买方垄断）是「a market structure where there is only one buyer for a particular good or service」，并以 labour market 中雇主对工资与工作条件的支配力为例，末尾附上完整参考文献：`References: Bivens, J., & Mishel, L. (2013). The Pay of Corporate Executives and Financial Professionals as Evidence of Rents in Top 1 Percent Incomes. Journal of Economic Perspectives, 27(3), 57-78.` 卷期页码 27(3), 57-78、期刊名 Journal of Economic Perspectives、年份 2013，都是转写没念到而画面独有的信息。

![OpenAssistant的monopsony问答示例，回答末尾附Bivens与Mishel于2013年的完整参考文献，并追问该样本在教模型什么](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0019_00-23-05.jpg)

幻灯片紧跟着抛出「What is this example teaching the model?」，并列出两条：`1. Teaching the model about Bivens J & Mishel, L`，`2. Teaching the model to output citations when asked to do so`，最后补了一句带刺的追问「(But by what mechanism? Does the model know about cites?)」。这正好对应讲者在 [22:34]–[22:53] 的分析：做 SFT（supervised fine-tuning，监督微调）时本质上仍在做 next-token prediction（下一个词预测），因此一条样本同时通过两个机制起作用。其一，它把 Bivens 与 Mishel 这篇文献的具体知识灌输进权重；其二，它告诉模型「when you're giving a response, a good thing to do is to give a reference」，也就是把「被要求引用时要输出引用」这一格式行为也监督了进去。用损失写出来，就是标准的自回归监督目标：

$$\mathcal{L}_{\text{SFT}}(\theta)=-\sum_{t=1}^{T}\log p_\theta\big(y_t\mid y_{<t},x\big),\qquad y=\underbrace{[\text{正文}]}_{\text{知识}}\oplus\underbrace{[\texttt{References:}\ \text{引文}]}_{\text{格式}}$$

注意 $\oplus$ 拼接的两段都被同一个 $\mathcal{L}_{\text{SFT}}$ 逐 token 地监督，损失本身并不区分哪一段是知识、哪一段是格式——这正是问题根源。讲者说这两件事「two different mechanisms that are sort of going on at once」：既在教知识（「you want the model to be knowledgeable」，这是好事），又在「teaching the model to emit detailed knowledge」。麻烦在于二者纠缠后会产生错误泛化（misgeneralize）：模型会「hallucinate out a reference instead of actually giving a proper reference」，因为「Models don't really know necessarily whether a reference is true or false」。一旦格式学得过强而知识没跟上，模型就会编造一条看起来格式正确、内容不存在的引文。

![知识提取与对齐：左侧Hallucination and Behavior Cloning的要点与知识图谱示意，右侧Gekhman 2023的训练/开发准确率曲线](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0020_00-24-20.jpg)

下一张幻灯片把这套民间说法（folklore）与两条实证线索并置，标题「Knowledge extraction and alignment」，核心句写作「Folklore: Fine-tuning a model on 'facts it doesn't know' makes it hallucinate」。讲者在 [23:30]–[23:42] 说这个 folklore「is not wrong」，有「good empirical evidence」支撑：如果在 SFT 阶段训练模型输出它不知道的事实，会让它更倾向于幻觉（hallucinate）。原因是模型同时在泛化两样东西——「trying to generalize the knowledge, but also the format」——而拿「格式 + 一块未知知识」去训练，等于「teaching the model to forcibly emit unknown knowledge」。左侧嵌的是 `[Schulman 2023]` 的「Hallucination and Behavior Cloning」要点：神经网络把一张「knowledge graph」存在权重里、每条边带一个 confidence level；小规模微调只学到一个在知识图谱上操作并输出答案的简单函数（示例 `Q: what is the genre of Star Wars: A Sci-fi`）；若把图谱之外的正确答案也拿来行为克隆（behavior cloning），就是在教网络幻觉；因此 cloning target 必须与网络内部已有的知识图谱一致，否则用另一个 agent 计算出的 target 训练总会产生 hallucination problem。右侧是 `[Gekhman 2023]` 的两条曲线：上图纵轴 `Train Accuracy`（0–100）、横轴 `Epochs`（0–50），绿色 `Train Known` 迅速逼近 100，橙色 `Train Unknown` 爬升缓慢，虚线在约第 10 个 epoch 标出「Overfitting starts」；下图纵轴 `Dev Accuracy`（约 39–44），紫色 `Dev` 先升到约 43 后回落到约 39–40。这组曲线把「教未知事实 → 过拟合 → 开发集掉点」的时序画得很清楚，而这些轴标签、曲线名与两篇引用转写都没念到。

讲者随后把 John Schulman 的论点接了进来（[24:16]–[24:44]）：Schulman 主张这正是需要强化学习（reinforcement learning，RL）的理由——他本人「like a reinforcement learning guy」「made PPO」，立场不意外，但「the argument is sound」。要知道模型「what it knows and doesn't know」，反馈必须是 policy dependent（依赖当前策略自身）的，不能由外部标注者「shoving knowledge down your throat」硬灌，否则模型无法对自己知道什么保持校准（calibrated）。本章的易错点也在这里：把「会输出 References:」误当成「掌握了文献」，把格式与内容两个被同时监督的目标混为一谈；从行为克隆视角看，只要监督目标与模型内部知识不一致，幻觉就不是偶发 bug 而是训练目标的结构性后果。下一章会把这条线索推向长尾知识（tail knowledge）与 RL 能帮到什么程度。

## 长尾知识、幻觉与 RL 的作用 [24:49 - 28:15]

把上一章的两条线索合起来，讲者给出一个反直觉的推论（[24:49]–[24:55]）：「you might not want to train on the highest-quality data if the model doesn't already know that data」——即使数据质量再高，只要模型本来不知道，就未必该拿来训练。长尾知识（tail knowledge）「can actually sometimes be actively harmful」，因为它可能诱发幻觉，尤其是当它和「reference colon」这类标记绑定时，那个标记会「force the model to emit a reference afterwards」，逼模型在并不知情的情况下也要吐出一条引用。因此讲者把 RL 列为「one of the reasons why reinforcement learning is important and useful」：对于「knowing what you know」这类 model dependent（依赖模型自身）的事情，RL 是「a very useful way to teach models to not say things that it does not know」。他顺势给了一个更根本的判断（[25:26]–[25:43]）：「knowledge storage is actually very messy and a very nuanced concept」，不能简单说「just pack knowledge in there」；必须非常小心后训练阶段扔进去的到底是什么监督信号。这也是他觉得 post-training「quite messy」的原因之一——你要面对「all these realities of serving a system eventually to a user」，而不只是压缩语料。

![知识提取与对齐的三点结论：不宜微调长尾知识、RL式正确性反馈可能有帮助、知识存储与提取messy且nuanced](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0021_00-25-35.jpg)

幻灯片把这章收束成标题「Takeaways on knowledge extraction and alignment」下的三条，文字值得原样记住：`1. You may not want to fine-tune on tail knowledge, even that's the LM use case`（即便长尾知识正是语言模型的使用场景，也可能不该在它上面微调）；`2. In principle, 'RL' style correctness feedback could help`；`3. Knowledge storage and extraction in LMs is messy, and nuanced.`。注意第 2 条用的是「in principle」和「could」，是留有余地的表述，而不是「RL 一定解决幻觉」。紧接着的学生问答把边界讲得更细。有人问 tail knowledge 的定义（[25:47]），讲者坦承「I don't think there is like a formal definition of tail knowledge」；他举例说，过去做这类工作时曾用 Wikipedia 文章长度作为「well-knownness」（知名度）的代理，发现「if we train on stuff that's not well-known, we do see that the model hallucinates more」，但知识本身「can't really be pinned down precisely」，所以不存在一条「this is tail and this is not」的正式分界。这是一个必须记住的局限：长尾是个经验代理，不是可操作阈值。

关于「RL 为什么能帮上幻觉」的追问（[26:20]–[27:22]），讲者主动说「I'm being too cheeky」，然后给了一个「folk story」。设想模型激活（activations）内部存在一个「I know something direction」，记作方向 $d_{\text{know}}$。SFT 阶段因为被强制在样本里生成引用，模型「is just going to generate references no matter what」，格式与内部状态被解耦了。进入 RL 后，奖励来自自身策略的输出（policy dependent），于是可以写成：

$$J(\theta)=\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}\big[r(x,y)\big],\qquad r>0\ \text{当}\ \langle h,d_{\text{know}}\rangle\ \text{高},\quad r<0\ \text{当}\ \langle h,d_{\text{know}}\rangle\ \text{低}$$

用文字说：在「I know stuff」方向上生成引用得到好 reward，在「I don't know」方向上生成引用得到差 reward，模型就可能「learn to generalize that like I know things or don't know things activations into whether you emit a reference」，把内部的知/不知校准抽取到输出策略里。讲者立刻划出这条故事的边界（[27:12]–[27:22]）：「If you know what you know somewhere inside the model, RL can help you extract that into your output policy. If you don't know what you know at all, right, at any level, RL cannot help you there」——如果模型在任何层面都没有关于自己知道什么的信号，RL 也造不出校准来。这与上一章 Gekhman 曲线里 `Dev` 从约 43 回落到约 39–40 的过拟合现象是同一图景的两面：RL 能做的是「提取」而非「注入」。

第三个问答（[27:26]–[28:12]）澄清了一个容易误解的点：在 SFT 里引用是不是被「penalized」了？讲者答，某种意义上是隐式惩罚——「assigning any probability mass to something that is not this exact sequence is penalized」，也就是上面的 $\mathcal{L}_{\text{SFT}}$ 会给任何偏离参考序列的概率质量付代价；但「in some ways SFT isn't doing the wrong thing. Like this is a correct sequence」，SFT 的目标本身没写错，错的只是它的泛化性质（generalization properties）。单条样本里其实有「two sort of disentangled things」：一是 reference template（引用模板），二是「the actual content of the knowledge」（知识的实际内容）；「if I misgeneralize based on the template, I'm in big trouble」。本段中间的两张现场照片（26:50 与 28:05）拍的正是讲师站在这张「References, complex knowledge, and factuality」幻灯片前比划，说明这一页被反复讲了较长时间。易错点归纳起来有三条：把「没有正式定义」的长尾当成可精确切分的集合；把 RL 当成能凭空生成校准的万能药；把幻觉归咎于 SFT 损失函数写错，而没看到是「无法从单个例子把模板与知识解耦地泛化」。

## 安全 SFT 的框架与工业界实践 [28:16 - 32:12]

讲者在 [28:16] 用「the other thing that I want to talk about is safety」切换话题，并给了一个很能说明两种团队气质差异的对照：做预训练的人「live in this like ivory tower of like let's just compress the world」，而做后训练的人必须直面「all of the messy realities of like how are people going to use this」。他举的滥用场景包括政治操纵与虚假信息（political manipulation and disinformation）、以及「individualized spear fishing」（个性化鱼叉式钓鱼），并强调后训练团队「you're the last line of defense in some ways」——是系统与滥用之间的最后一道防线。由此得到各公司「de facto」的做法：给系统施加安全控制（safety controls），让模型「refuse to respond to sort of malicious inputs」，而这副担子落在 post-training 团队身上。但讲者马上泼冷水（[29:06]–[29:29]）：安全 SFT 的公开信息「seems even more sparse than capabilities SFT information」；他挑出 Llama 2 的描述，说这已经算「one of the more detailed descriptions」，结果「they don't even talk about how many examples they used for safety tuning, which is kind of crazy to me」。

![Llama 2安全微调数据段落原文与Figure 18：以False Refusal Rate为横轴、Violation Rate为纵轴，比较Llama 3 8B与70B的权衡](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0024_00-29-20.jpg)

幻灯片「Safety SFT in the wild / Details are pretty sparse」把这段稀缺材料摊开，左侧是 Llama 2 论文 `Fine-tuning data.` 一节的摘录，转写完全没念到这些文字：它说安全微调数据的「quality and design ... has a profound impact on performance」，并通过大量消融发现「the quality is more critical than the quantity」；他们主要用数据供应商提供的人类生成数据，但发现这类数据「can be prone to errors and inconsistencies — particularly for nuanced safety policies」，因此开发了 AI 辅助标注工具（AI-assisted annotation tools）来支撑严格质检；除对抗性 prompt（adversarial prompts）外，还收集与之相近的 borderline prompts，目的是教模型给出有帮助的回答、从而降低误拒率 `false refusal rate (FRR)`；此外还用合成数据（synthetic data）提升质量与覆盖，手段包括精心构造 system prompt 的 in-context learning、基于新攻击向量对种子 prompt 做 guided mutation，以及 `Rainbow Teaming (Samvelyan et al., 2024)`（基于 `MAP-Elites (Mouret and Clune, 2015)`）这类跨多个多样性维度生成 prompt 的算法。右侧是 Figure 18：纵轴 `Violation Rate (%)`，刻度 20、40、60；横轴 `False Refusal Rate (%)`，刻度 2、2.5、3；图例为 `Llama 3 8B` 与 `Llama 3 70B`。图注说明每个散点代表一种平衡 safety 与 helpfulness 数据的 data mix，不同模型尺寸的安全学习能力不同：8B 模型需要在整体 SFT 混合里放入更高比例的 safety data，才能达到与 70B 相当的安全表现；更大的模型更能区分 adversarial 与 borderline context，因而 VR 与 FRR 之间的权衡更有利。右下角一行小字补上了讲者吐槽的那个缺口：`# Examples? ~ few thousand in Llama 2`。

讲者在 [29:34]–[30:03] 把框架抽象成两个必须同时压低的量，并要一个好的 Pareto trade-off（帕累托权衡）：一是违规率（violation rate），即「how often am I allowing bad queries to get through」；二是误拒率（false refusal rate），即不要过度拒答。他给的例子很生动：用户只是问「how do I kill a Python process?」，如果机器人回一句「No, I can't let you kill anything」，「That's very frustrating to the user」。形式化地写，安全调优就是在约束下做双目标优化：

$$\min_{\theta}\ \mathrm{VR}(\theta)\quad \text{s.t.}\quad \mathrm{FRR}(\theta)\le \epsilon,\qquad\text{或}\qquad \min_{\theta}\ \mathrm{VR}(\theta)+\lambda\,\mathrm{FRR}(\theta)$$

$\lambda$ 决定你在 Figure 18 那条权衡曲线上取哪个点；把 $\lambda\to 0$ 一味压 VR，就会滑向过度拒答。规模上讲者在 [30:08]–[30:21] 说，这类 tailored data「usually have something like a few thousand or tens of thousands of examples」，Llama 2 大约是「a few thousand」，与图右下角一致。随后他把唯一值得看的完整公开参考指向 Tulu 3（[30:33]–[31:01]）：这是 Allen AI 为 Olmo 模型做的后训练 pipeline，鉴于多数地方对后训练「lack of details」，它是「one of the few places where you can see a reasonably performant post training pipeline in action」，其安全组件约「50-ish thousand examples uh total」。

关于 Tulu 3，画面里那页标题是「Pipeline with most details」，论文全名 `Tülu 3: Pushing Frontiers in Open Language Model Post-Training`，作者列有 Nathan Lambert、Jacob Morrison、Valentina Pyatkin、Shengyi Huang、Hamish Ivison、Faeze Brahman、Lester James V. Miranda 等，机构标注 `1 Allen Institute for AI`、`2 University of Washington`。下方表格给出 `Safety & Non-Compliance` 三行精确数字，都比讲者口头的「50-ish thousand」更细：`Tülu 3 CoCoNot` 为 `10,983 / 10,983 / 10,983`（Brahman et al. 2024）、`Tülu 3 WildJailbreak` 为 `50,000 / 50,000 / 26,356`（Jiang et al. 2024）、`Tülu 3 WildGuardMix` 为 `50,000 / 50,000 / 26,356`（Han et al. 2024）。生成 SFT 数据的策略被讲者形容为「fairly very straightforward」：此前 WildChat 项目「gave free API or free chat access to a bunch of people」，以免费使用换取聊天记录，再从中「filter out a whole bunch of unsafe behaviors」以及用户尝试的 jailbreaks（越狱），「mined these at scale」之后构造偏好回答——「resist the jailbreak in one case and to say no in another case」。

WildChat 这篇论文的信息量同样超出转写：画面标题「Main safety approach – extract scenarios from users」，内页论文题为 `WildChat: 1M ChatGPT Interaction Logs in the Wild (ITW)`，即一百万条真实交互日志；开头有一段全大写警告，说明论文附录含非法及可能令人不适的用户输入（包括 violence、sex 等），「READER DISCRETION IS ADVISED」；作者为 Wenting Zhao、Xiang Ren、Jack Hessel、Claire Cardie、Yejin Choi、Yuntian Deng，机构为 Cornell University、Allen Institute for Artificial Intelligence、University of Southern California、University of Washington。Figure 2 是 `NONCOMPLIANCE Taxonomy`，把不安全请求分成 Incomprehensible（含 Nonsensical、Under-specified）、Unsupported（含 Modality Limitations、Length、Temporal、Dominating Requests）、Privacy 等子类，并注明「Desired responses for these categories are not always direct refusal」——期望回答并不总是直接拒绝。Figure 1 给出 WILDTEAMING 框架两步：`Step 1) Mining Automatic Mining of User-Written Jailbreak Tactics in the Wild (ITW)` 与 `Step 2) Composing Jailbreak Tactics with Vanilla Harmless Queries for Diverse Adversarial Attacks`。讲者在 [31:53]–[32:08] 补了一句闭源侧的对应做法：从技术报告与 model cards 看，各公司也是「look at your usage information, you find unsafe behaviors, you get your annotators to play whack-a-mole」，即打地鼠式地补数据。本章易错点：把 VR 压到零而不顾 FRR；把安全数据数量当质量（Llama 2 明说质量更关键）；误以为公开文档足以复现——恰恰连用了多少条都没写。

## 少量数据即可显著改变安全行为 [32:13 - 33:43]

讲者用「the final thing I'll mention」引出与「the number of examples that you need for SFT」直接相关的一个现象（[32:13]–[32:30]）：如果你没亲手做过 SFT，会惊讶于「if you have a sufficiently capable model, it does not take very many examples to steer these systems」。具体到 safety tuning，做法是「find a bunch of unsafe stuff or you synthesize a bunch of unsafe stuff and then you write refusals and you put that into the system」，而样本量可以小到「let's say 500 examples」。结果（[32:42]–[33:03]）是「even with as little as 500 examples」，模型遵循恶意指令、或其他仇恨言论类（hate speech）指令的比率「drops dramatically」；他把这称为「a fairly surgical ... very small push to the model」，却换来「a fairly nice across-the-board reduction in these safety issues」。这一章的直觉与第 3 章一脉相承：讲者在 [33:03]–[33:18] 说，「models already have a, you know, am I going to be a safe model or an unsafe model axis inside of it after pre-training」，预训练后模型内部已经存在一条安全/不安全的轴，「so it does not take very many examples to pull this out」。SFT 在这里做的同样是提取（extract），不是注入。

![仅需约500条安全样本即可显著降低四个安全数据集上的Mean Score：I-MaliciousInstructions、I-CoNa、I-Controversial、Q-Harm](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0027_00-33-31.jpg)

幻灯片「Safety-tuning with just a little data / Significant improvements to safety with ~500 samples.」把「~500」背后的完整实验摊开，这些是转写没念到的。柱状图纵轴为 `Mean Score`（刻度 0–3），横轴 `Dataset` 下并排四组：`I-MaliciousInstructions`、`I-CoNa (hate-speech)`、`I-Controversial`、`Q-Harm (Anthropic HH)`。图例 `Added Safety Data` 给出七个梯度：`0 / 100 / 300 / 500 / 1000 / 1500 / 2000`，每根柱还带误差棒。读图可见：加入 0 条时四组 Mean Score 都接近 3（约 2.9–3.2）；加到 100 条降到约 1.8–2.4；到 500 条已降到约 0.3–1.2；继续加到 2000 条也只再降到约 0.1–0.9，边际收益明显递减。底部一句结论点明所用数据形态：`Adding 500 Alpaca-style examples makes models follow safety guidelines`——是 500 条 Alpaca 风格（Alpaca-style）样本就让模型开始遵守安全准则。可以用一条增量式记号把这种「小样本、大变化」写清楚：

$$\Delta\mathrm{VR}(N)=\mathrm{VR}(0)-\mathrm{VR}(N),\qquad \Delta\mathrm{VR}(500)\gg \Delta\mathrm{VR}(2000)-\Delta\mathrm{VR}(500)$$

也就是说前 500 条带来的违规率下降，远大于从 500 追加到 2000 条的额外下降；曲线在很小的 $N$ 处就基本饱和，这与「模型内部已有该轴、少量样本只是把它拉出来」的解释一致。

但讲者立刻堵住一个危险的过度推广（[33:21]–[33:40]）：「just because you can steer model with very few examples, that doesn't mean that there's no benefit to more examples」。如果你像 OpenAI 或 Anthropic 那样，要执行「really fine-grained distinctions about what is safe and not safe」——例如区分 Figure 2 里那些「not always direct refusal」的边界情形——「you do need very large-scale data collection」。他明确说「This doesn't really change that story」。因此这一章的局限要说全：500 条覆盖的是粗粒度的「拒绝/不拒绝」开关，能带来 across-the-board 的整体下降，却无法覆盖长尾的细粒度边界；也正呼应第 4 章 Tulu 3 要用 `50,000` 级别数据、以及 Llama 3 8B 比 70B 需要更高比例 safety data 的观察。易错点是把「少量样本能 steer」误读成「安全数据无须规模」，或者把柱状图里 Mean Score 的下降当成四个数据集上同幅度、无方差的确定结论——图上是带误差棒的估计，且不同数据集的饱和点并不相同。

## SFT 数据阶段小结与 SFT/RL 之辨 [33:44 - 35:53]

讲者在 [33:44] 用「SFT at its best」给整个数据部分下了一个定位：它最擅长的情形是「extracting pre-training behaviors」——某个行为「it's already in there somewhere in pre-training」，你要做的只是「pull out the right modes out of the model」，那么指令微调（instruction fine-tuning）「works very very well」，而且「With very little data, as long as you have the right high-quality ones」就能让模型照你的意思做。他同时把前两章的教训并进来（[34:03]–[34:13]）：「adding data or factually correct ones can sometimes hurt you, at least in terms of hallucination」，并且「it doesn't take very much data and so you can often focus on quality instead of quantity」。现场照片（34:57）里投影幕露出的正是这页总结的第 3 条「Small amounts of ... make a big differe...」，与上一章的 500 样本柱状图互为印证。把这三条小结按讲者的口径排在一起，可以理解为一条从「提取」到「质量优先」的链条：SFT 不是往模型里塞新能力，而是选模态；事实正确的数据也可能在幻觉维度上倒扣分；既然所需样本很少，就把预算花在质量而非数量上。

接下来的两个问答把这套说法的边界和它背后的算法观都摊开了。第一个问答（[34:26]–[34:52]）问「how do you know whether something is in pre-training?」，讲者罕见地承认「this is the sloppiness of what I was saying which is that we don't really know per se」。我们能做的是反向排除：「We can know what things aren't in pre-training」，例如对「programming languages that are very rare」（极罕见的编程语言）做泛化，靠 SFT 很难学会；但「We can't show necessarily that like for example safety is something that's in pre-training per se」——无法正面证明「安全」一定已经存在于预训练里。这对上一章「模型内部已有安全轴」的直觉是一个重要保留：那是解释性的 folk story，不是可验证命题。第二个问答（[34:57]–[35:26]）问 SFT 是否在「destroying features」、RL 是否在「deleting them」，讲者干脆回「I'm not going to talk about that later」，并说「the lines are very blurry between the two」——尤其遇到 expert iteration 这类「look like SFT with extra bells and whistles」的方法时，界限更模糊；他「don't necessarily think it's a core distinction between RL and SFT」。

他给出的替代区分（[35:26]–[35:50]）不在算法、而在反馈的种类：SFT 是「dense teacher supervision」（稠密的教师监督），RL 是「self-taught policy supervision」（自学的策略监督）；而且 RL 里「you're getting your own output from your own policy」，所以「you don't maybe deviate quite as far in what you're being reinforced on」，在被强化的分布上不会偏离自身太远，「that's probably an important part of what's happening」。用两个目标并排写，差异就很直观：

$$\mathcal{L}_{\text{SFT}}(\theta)=-\mathbb{E}_{(x,y^*)\sim\mathcal{D}}\big[\log \pi_\theta(y^*\mid x)\big]\quad(\text{固定参考序列 } y^*),\qquad J(\theta)=\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}\big[r(x,y)\big]\quad(\text{样本来自自身策略})$$

左边对每个 token 都有来自外部 $\mathcal{D}$ 的稠密监督，右边只有标量奖励、且 $y$ 由当前 $\pi_\theta$ 采样——这正是第 3 章「policy dependent」说法的数学对应，也解释了为什么 RL 能把「我知不知道」这类内在状态抽出来，而 SFT 只能拟合给定序列。本章易错点有三：把「extract pre-training behaviors」当成可证明的定理（讲师自己说 sloppy、无法正面验证）；把 SFT 与 RL 的对立当成算法本质区别（真实区别是 dense teacher supervision 与 policy supervision 的反馈类型，且 expert iteration 等中间形态使其模糊）；忽略「adding factually correct ones can hurt」这一反直觉结论，继续默认数据越多越好。至此数据部分结束，讲者在 [34:18] 明确说「this is going to be the end of the data part for SFT」，下一章转入方法。

## 方法与中期训练：把指令微调并入预训练 [35:54 - 39:57]

讲者对方法部分开场的第一句几乎是自嘲（[35:54]–[36:09]）：「for most of this class we've been excited to talk about methods. In this part it is about the most boring thing because you just do gradient descent」，他甚至说「I just put the picture here as almost a joke. You just take loss dot backwards and that's basically it」。也就是说 SFT 在算法上与预训练没有实质差别，跑的就是标准训练循环；配图（35:59 的 fine-tune 代码示意）也只是一段再普通不过的梯度下降。把它写成可对照的形式：

```python
for batch in dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    loss = model(**batch).loss      # next-token prediction 的交叉熵
    loss.backward()                  # 讲者所说的 "loss dot backwards"
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
```

但讲者强调「there is one nuance I do want to talk about that is important」，即 [36:13] 起「this increasing trend of turning instruction tuning into part of pre-training」——把指令微调并入预训练的趋势。过去的范式（[36:20]–[36:31]）是「pre-train on web data and then you would post-train afterwards」，两者是分开的两个阶段；后来大家意识到「why separate two things when you can mix them together」，于是「lots of high-quality data and even instruction tuning data is now getting mixed in at the tail end of training when the decay phase usually of training happens」。这么做有两个好处（[36:45]–[36:50]）：「allows you to scale up instruction tuning」，也「allows you to essentially emphasize higher quality data」。讲者说影响「has been quite dramatic」「everyone as far as I know is doing it」，你在多数模型发布报告里都能看到「mid-training or like a second phase pre-training with a different data mix」。

画面里还有一页把上述趋势写成可执行的三条，标题「Turning instruction tuning into pretraining」，先问「Can we somehow turn instruction tuning data into pretraining data?」，再说「The following (increasingly popular) idea says yes:」，随后列出 `1. Pre-train on web/pretraining data`、`2. Mix in instruction-tuning data into pre-training`、`3. Do an actual (but short) instruction-tuning round.`，结尾一句点出动机：`Lets you scale up instruction tuning w/o catastrophic forgetting`（在不发生灾难性遗忘的前提下放大指令微调）。「catastrophic forgetting」这个术语以及「第 3 步仍要做一轮短 SFT」都是转写没明说、画面独有的增量。讲者还顺带吐槽了一个「pet peeve」（[37:09]–[37:35]）：现在别人告诉你某个是 base model（基座模型），「that's kind of a lie」，因为传统意义上的 base model 是「predicting the next word on internet data」，而今天的 base model 往往「pre-trained on like ultra chat and like who knows what else」——那些是「synthetically designed to make you good at chat」的聊天数据集，因此「it's very hard to say that this is like a base model in the traditional sense」。

![Midtraining两阶段训练的数据混合饼图：稳定阶段以Code Pretrain、CommonCrawlCCn、Dolmas为主，衰减阶段切换到更高质量更聊天化的数据](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0030_00-37-48.jpg)

讲者选的两阶段例子来自 miniCPM，理由是它「show very nicely sort of the change in the mix」，幻灯片标题「'Midtraining' / 'Two-phase training'」，副标题「The recipe is common knowledge among many LLM companies (but not documented)」。左图 `Data Mixture of Stable Stage` 是标准互联网数据：`Code Pretrain 25.0`、`CommonCrawlCCn 25.0`、`Dolmas 24.0`、`C4 15.0`、`Pile 9.0`，另有 `Open Web Math`、`Arxiv`、`peS2s` 等小扇区。右图 `Data Mixture of Decay Stage` 明显更碎、更偏高质量与聊天化：`Code Pretrain 19.6`、`CommonCrawlCCn 15.7`、`C4 14.6`、另有一块 `C4 9.5`、`Wikipedia 7`、`Pile / SFT_mixvol 5.1`、`Baidu Baike 4.8`、`Code_SFT 4.5`、`Book Chinese 2.8`，以及 `Knowledge-SFT 2.5`、`UltraChat 2.5`、`Stack Exchange QA 2.5` 等，右侧还列了一长串小比例标签如 `Falldataset`、`Oasst1et`、`ShareGPT4`、`Arxiv`、`WebInstructSub`、`Math_SFT`、`Math-Synthetic`、`Other` 等。底部注明「Widely used by most models today.. Publicized in recent Chinese-derived LMs (miniCPM, jetMoE)」。把这些权重写成约束，就能看清 decay 阶段到底改了什么：

$$\sum_i w_i^{\text{stable}}=1,\qquad \sum_i w_i^{\text{decay}}=1,\qquad w_i^{\text{decay}}\ \text{相对上调高质量 / SFT 类源},\quad \eta_t \xrightarrow[t\to\text{decay}]{} 0$$

即通用网页数据的占比被压低（如 Code Pretrain 由 25.0 降到 19.6、CommonCrawlCCn 由 25.0 降到 15.7），腾出的份额给 Wikipedia、Stack Exchange QA、UltraChat、各类 SFT 数据；同时学习率 $\eta_t$ 在 decay 阶段趋于最低。讲者在 [37:50]–[38:25] 的口述只点名了 Stack Exchange QA、UltraChat 和「all sorts of other SFT data」，这些具体百分比、扇区名与 jetMoE 都来自画面。

最后三个问答把细节补严。关于「queries 是否被 mask」（[38:32]–[38:43]），讲者答「No, this is pure pre-training」，所以「you're also predicting the prompt as well」；但他补了一句「that's not a huge difference because some SFT recipes do also predict the prompt」——因此 mask 与否并不是两个阶段的分水岭。关于「decay 阶段数据质量是否更低」（[38:53]–[39:15]），他说直觉恰恰相反：「the decay is the most important part of your training」，因为它「closest to deployment during pre-training」，而且「it's the part with the lowest learning rate」，两条理由都指向「maybe you want to put the highest-quality stuff into the decay」；他顺口提到 book Chinese、Wiki、Stack Exchange「is usually considered high quality」。关于数据混合怎么定（[39:21]–[39:52]），讲者答「data mixtures, I mean, both for pre-training and post-training are very trial and error」，虽然「we do have algorithms. Like many people have written papers about things in this...」，转写在此截断，他没有给出确定配方。本章易错点：把今天的 base model 当成纯网页预训练产物；误以为 decay 阶段该放低质量数据（正好反了）；以为混合比例有可靠算法可依（实际高度试错）；以及忽略 prompt 本身也进入 loss，从而误判 pre-training 式混合与 SFT 的差别。讲者在 [38:15] 的收束是「the boundaries between pre-training and instruction tuning and also high-quality data is just being blurred very much」——这也正是本段从数据、安全一路走到方法后的总落点。

## 数据混合的试错本质与 decay 阶段 ablation [39:57 - 41:58]

本段开头承接上一节关于 mid-training（中期训练）与 two-phase training（两阶段训练）的讨论。讲师对数据混合优化这个 "space"（领域）给出的判断相当保守：他认为相关方法 "fairly unreliable"（相当不可靠），里面充斥着大量 trial and error（试错）与 intuition（直觉）。不过两阶段训练带来一个很实际的好处：mid-training 比 full pre-training（完整预训练）短得多，因此同样的算力预算下，对左侧稳定阶段的每一种数据配比，都可以在右侧的 decay phase（衰减阶段）跑大约 10 次 ablation（消融实验）。把这件事写成数量关系，就是

$$N_{\text{decay}} \approx 10 \times N_{\text{stable}}$$

讲师特意说明这 "not quite brute force the problem"（算不上暴力搜索），但确实 "does let you try a lot of ablations"——它把原本在完整预训练上根本做不起的实验，变成可以反复跑的廉价实验。这也解释了为什么数据混合的 know-how 主要沉淀在工程直觉里，而不是在可复现的论文里：你能跑的实验次数上去了，但每次实验仍然只能给出零散的、依赖具体设置的信号。

接着他交代了行业里 "the usual way"（常见做法）：在 decay phase 做大量 data ablations，"which are cheap to run and so on"，由此得到 data quality（数据质量）的估计，再把这个估计 "reflect that even back to the first state"、也就是反推回 pre-training stage（预训练阶段），用来决定 pre-training mix（预训练混合配比）。为什么不能干脆把预训练阶段直接全换成高质量数据？答案非常朴素——token 不够。讲师原话是 "you just don't have enough tokens, right? You run out of tokens if you try to make Wikipedia your whole pre-train"，写成约束即

$$\#\text{tokens}(\text{Wikipedia}) \ll \#\text{tokens}(\text{pretrain})$$

也就是说，Wikipedia 这类高质量语料的规模撑不起整个预训练的 token 预算，高质量数据只能被集中安排在训练尾端、占比很小的那一段里。于是 decay 阶段的配比实验事实上承担了双重角色：既决定尾端怎么混，又充当探测各数据域价值的探针，把结论外推到预算大得多的稳定阶段。

[40:59] 有学生追问：primary mixture（主混合配比）到底按什么直觉来分配？讲师的回答 [41:14] 是 "very case-by-case"（高度依具体情形而定）——从 downstream deltas（下游指标上的差值）映射回 mixes 并没有通用规则。他承认存在 "systematic ways, like you can fit models and do these things"，也就是拟合模型来预测配比收益，但这些方法 "often brittle"（往往很脆弱）。真实流程比外人想的更 trial-and-error：先 ablate 一个 domain（数据域），对 "which ablations have the most impact" 做 ranking（排序），再据此决定往混合里放什么。他举了一个很有意思的公开证据：Meta 因使用图书被起诉后，法庭文件里泄露出研究人员对不同 book subsets（图书子集）做 ablation、逐一估计 "how useful are each of these book subsets" 的文档，最终正是基于这类排序去定 mixes。这说明前沿实验室的数据配比决策同样是人工排序加经验，而不是某个漂亮的优化器。

![两阶段训练 Midtraining 幻灯片投影实拍：稳定阶段与衰减阶段两个数据混合饼图](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0032_00-40-34.jpg)

画面上这张投影实拍补充了转写里没逐个念到的文字：幻灯片标题是 "'Midtraining' / 'Two-phase training'"，副标题写着 "The recipe is common knowledge among many LLM companies (but not documented)"——配方是圈内共识却没人正式写下来。两个饼图分别标注 "Data Mixture of Stable Stage" 与 "Data Mixture of Decay Stage"，后者被切成明显更多、更碎的扇区，图例里能辨认出 Code/Pretrain、Encyclopedia、StackExchange (QA)、Mathematical、UltraChat、SFT 等类别名，正好对应讲师说的"尾端切换到高质量、更偏聊天与指令的数据"。底部两行小字是 "Widely used by most models today.." 与 "Publicized in recent Chinese-derived LMs (miniCPM, jetMoE)"，把这种做法明确归到 miniCPM、jetMoE 这批公开过细节的模型上。

这里的易错点有两层。第一，decay 阶段的学习率、数据分布与稳定阶段都不同，把短阶段 ablation 的边际收益当成预训练数据质量的无偏估计本身就有风险，讲师自己给整套方法贴的标签就是 "unreliable"。第二，法庭文件里的图书子集实验只是一个个案，不能推广成通用配方；能跑 ~10 个 ablation 也不等于覆盖了配比空间。这一章同时是 SFT 方法部分的收束：讲完 [41:58] 讲师说 "so now we're done with SFT. Um the algorithm part was very short"，随后直接切入本段的主体——RLHF。

## 从拟合分布到最大化奖励：RLHF 的概念断裂 [41:58 - 44:01]

[42:05] 讲师宣布进入 "the second phase"（第二阶段）：SFT（supervised fine-tuning，监督微调）已经结束，接下来做 RLHF。他先用一句话给出操作层面的差别：pre-training 与 SFT 都是对给定语料预测下一个词，而 RLHF "we're just going to upweight or downweight different model outputs based on how a rater, or in some cases a reward model, decides how good these responses are"——不再拟合固定文本，而是依据 rater（人类评分者）或 reward model（奖励模型）判定的好坏，对不同模型输出上调或下调权重。随后他特意 "pause for one moment on this slide"，强调一个 "conceptual difference that I think is actually quite important"，并补充说这对不是从 stats/ML（统计与机器学习）背景转过来的人 "maybe it's less important"，可见这是一套思维框架的切换，而非新算法。

在 pre-training 里，我们做的是 "generative modeling of some sequence"：给定一批序列，预测 next word（下一个词）。SFT "is also the same"，只是 "you've changed the distribution, but you're predicting the next word"——换了数据分布，任务形式没变。因此两者都可以用 "your generative modeling brain"（生成建模的脑子）去理解：本质都是 "fit some distribution"（拟合某个分布）。到了 RLHF，"we are no longer playing a fit a distribution game. We are fitting a maximize a reward game"。我们仍然持有一个分布，但此时它叫 policy（策略），目标变成让这个策略 "maximize some downstream reward"（最大化某个下游奖励）。奖励的例子讲师随手给了两个：用户在平台上的 engagement（参与度/停留时长）、能解出多少 math problems（数学题）。关键态度是 "I don't care if I've mimicked some distribution"——是否复现了某个参考分布不再重要。

![从模仿到优化幻灯片：Imitation (SFT) 与 Optimization (RLHF) 两段公式与各自的 bullet](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0034_00-43-04.jpg)

这张 "From imitation to optimization" 幻灯片把上述口头区分写成了公式，是转写里没有念到、只存在于画面上的内容。Imitation (SFT) 一段写作

$$\text{Fit } \hat{p}(y\mid x) \approx p^{*}(y\mid x) \text{ for some reference distribution } p^{*}(y\mid x)$$

其下两条 bullet 是 "Pure generative modeling perspective"（纯粹的生成建模视角）与 "Requires samples from reference policy"（需要来自参考策略的样本）。Optimization (RLHF) 一段写作

$$\text{Find } \hat{p}(\boldsymbol{y}\mid\boldsymbol{x}) \text{ such that } \max_{p}\ E_{p}\left[R(y, x)\right] \text{ for a reward } R(y, x)$$

其下两条 bullet 是 "Maximize some reward function that we can measure"（最大化某个可测量的奖励函数）与 "LMs are policies, not a model of some distribution"（语言模型是策略，不是某个分布的模型）。两组公式里 $\hat{p}$ 是我们要训练的模型，$p^{*}$ 是参考分布，$R(y,x)$ 是给定输入 $x$ 与输出 $y$ 的奖励；差别在于第一式把 $\hat{p}$ 拉向一个既定目标，第二式只要求 $\hat{p}$ 在其自身采样下把期望奖励推高。

由此得到一个很反直觉但极重要的推论：在第二个世界里，"I can totally collapse out my distribution onto a single point for every input"——对每个 prompt，模型完全可以只给一个固定答案，"Not a distribution. And that would be okay, as long as it got a good reward"。生成建模要求保留分布与多样性，奖励最大化则毫不在意多样性；只要单点答案拿高分，坍缩就是合法甚至最优的。讲师说这个区分 "a big important sort of conceptual distinction"，要在进入后面内容之前先立住。

与前后内容的关系上，这一节是整段 RLHF 讨论的地基：后面讲 reward model（奖励模型）、讲 pairwise 数据、讲 pitfalls（陷阱），都建立在"目标是策略而非分布"这个前提上。易错点也正在这里——如果仍把 policy 当成"要拟合的目标分布"，就会误以为 RLHF 需要维持输出熵、需要保留多样性，从而无法理解为什么后来的 RLHF 模型会出现多样性骤降。另一个限定是讲师自己给的：这个区分对统计/机器学习背景的人冲击最大，对工程实践者更多是澄清直觉，不要把它当成数学上新提出的结论。

## 为什么优化：G-V gap 与生成/验证的不对称 [44:01 - 46:22]

[44:01] 讲师转向动机问题："why do we want to sort of do this thing of doing reinforcement learning"，为什么不干脆 "forever go and collect SFT data"（一直收集 SFT 数据）。第一条理由来自一个 gap（落差）：优化 humans want（人类想要的东西）时，"what they say they want and what they sort of generate" 之间 "sometimes a big difference"。幻灯片标题把这件事叫 "G-V gap"，即 generation（生成）与 verification / judgment（验证、评判）之间的错位。他放了一项自己觉得 "so funny even though it's a little bit old now" 的旧研究：请一群 freelance writers（自由撰稿人）为新闻文档写摘要，结果发现 "a few annotators actually preferred ... which was the predecessor to ChatGPT, over their own writing"——部分标注者竟然更喜欢 ChatGPT 前身模型的输出，而不是自己写的。

研究过程值得细说。讲师团队起初怀疑 "Are these guys actually doing annotations?"，于是去访谈，得到的回答是 "I looked at it and I thought, actually, this is pretty good stuff"。这些人本身就是专业写手，"They're good at writing. Their summaries are high quality. Like, we checked it for quality"，但看到模型输出时的反应是 "Huh, I did not realize that that's actually a better way of doing it"。结论被浓缩成一句："people aren't really these like optimal systems. And so, you know, when they judge things, it's actually different"。人作为生成器并不最优，作为评判者却可能识别出更好的写法。因此 "because of this gap, sometimes you might want to rate outputs rather than just to generate demonstrations"——有时给输出评级、排序，比让人写出示范答案更合适。把这种不对称写成代价关系：

$$\text{cost}\big(V(y)\big) \ll \text{cost}\big(G(y)\big), \qquad G \neq V$$

其中 $G$ 表示 generation，$V$ 表示 verification 或 judgment；两者不是同一个函数，且评判的代价远低于生成。

![G-V gap 幻灯片：DeepSeekMath-V2 论文题图与 Overall Preference 偏好对比表](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0035_00-44-19.jpg)

画面上的表格是转写完全没念到的数值。标题 "Why optimize? G-V gap"，副标题 "People don't always write the thing that they prefer in LM outputs.."。右侧 "Overall Preference" 表头写着 "Aggreement α: 0.07"（幻灯片原文即如此拼写），两列分别是深蓝的 "Freelance writers summaries are better" 与绿色的 "Instruct Davinci summaries are better"。逐行读下去：Total 为 $50.4\%$ 对 $49.6\%$；Annotator 1 为 $43.0\%$ 对 $57.0\%$；Annotator 2 为 $45.2\%$ 对 $54.8\%$；Annotator 3 为 $49.4\%$ 对 $50.6\%$；Annotator 4 为 $54.7\%$ 对 $45.3\%$；Annotator 5 为 $55.1\%$ 对 $44.9\%$；Annotator 6 为 $56.9\%$ 对 $43.1\%$。总体接近持平，但个体从 $43.0\%$ 摆到 $56.9\%$，$\alpha = 0.07$ 的一致性系数说明标注者之间几乎谈不上有共识——这既支持"评级/排序比示范更可取"的论点，也提醒我们单一 rater 的偏好噪声极大。左侧方框是 DeepSeek 的论文题图 "DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning"（DeepSeek-AI，Zhizhong Shao 等，附 github.com/deepseek-ai/DeepSeek-Math-V2），底部出处为 "From: Zhang et al – 'Benchmarking Large Language Models for News Summarization' 2023"。

第二条理由是 "in some domains, verification is a lot easier than generation"。数学是 "a prime example"："verifying a proof is probably much easier than generating the proof"；讲师提到 "folks like Deep Seek have sort of gone down that path for like self-verification using models"，即用模型做自验证，并把它归为 "this idea of sort of having reinforcement learning and judging yourself is a good idea"。这正好回扣上一节：训练一个 verifier（验证器）比直接训练一个做得好的模型更可行。随后他明确划定范围：今天不讲 "what people call RLVR"（可验证奖励强化学习），"That's its own universe"，留到下一次课讲 "the exciting promises of reasoning models"（推理模型）；本讲聚焦 RLHF，并按 data（数据怎么收集）、algorithms（算法两大家族）展开，"before sort of talking about maybe some pitfalls at the very end"。

这里的局限要讲清楚。这项研究 "a little bit old now"，样本是新闻摘要场景下的自由撰稿人，不能直接外推到代码、数学或安全标注；更关键的是 Total 的 $50.4\% / 49.6\%$ 很容易被误读成"人和模型差不多好"，但 $\alpha = 0.07$ 加上六行从 $43\%$ 到 $57\%$ 的摆动说明这个均值掩盖了巨大的个体分歧。另外 G-V gap 只是"为什么需要 RL"的两条动机之一，并不是 RLHF 的充分论证；讲师后面也会反复回到"标注者到底能不能判对"这个问题上。

## RLHF 数据流水线与 pairwise 标注指南 [46:22 - 49:12]

[46:23] 讲师给出 RLHF 的高层流程。起点是 "a good model after SFT that follows some instructions"（SFT 之后能遵循指令的模型）；把 prompts 灌进去，让模型 "generate a couple different outputs"，做法通常是 "sampling with temperature one"，即 $T = 1$，理由是 "At the end of SFT, the model is pretty diverse, so this is a reasonable thing to do"。接着 "A rater comes in and rates all of the examples and provides a ranking over the examples, sometimes even a binary ranking"，再由 "a model is going to be trained on those rankings"，最后 "we're going to use standard reinforcement learning techniques to maximize the score that we get on that reward model"。写成符号：

$$T = 1,\quad y_1,\dots,y_k \sim \pi_{\text{SFT}}(\cdot\mid x),\quad y_{i_1} \succ y_{i_2} \succ \cdots \succ y_{i_k}$$

排序标签用于训练奖励模型 $r_\phi$，随后 RL 阶段求解 $\max_{\pi} \mathbb{E}_{y\sim\pi}\left[r_\phi(x,y)\right]$。讲师专门解释了为什么要绕一道 reward model："we go through the reward model because it might be easier to train a verifier than to train a model that does well directly"，这与上一章 $V \ll G$ 的动机是同一条逻辑。

画面上的 "RLHF Data" 幻灯片（副标题 "What types of pairwise feedback. How do we get (good) pairwise feedback?"）把三步流程逐字写了出来，红框特意圈住 Step 2 与 Step 3，表示本讲重点是 comparison data 与 RL，而非 Step 1 的 SFT。原文如下：

```text
Step 1  Collect demonstration data, and train a supervised policy.
        A prompt is sampled from our prompt dataset.
        A labeler demonstrates the desired output behavior.
        This data is used to fine-tune GPT-3 with supervised learning.
        (example: "Explain the moon landing to a 6 year old" / SFT)

Step 2  Collect comparison data, and train a reward model.
        A prompt and several model outputs are sampled.
        A labeler ranks the outputs from best to worst.
        This data is used to train our reward model.          [RM]

Step 3  Optimize a policy against the reward model using reinforcement learning.
        A new prompt is sampled from the dataset.
        The policy generates an output.
        The reward model calculates a reward for the output.
        The reward is used to update the policy using PPO.    [r_k]
        (example: "Write a story about frogs" / "Once upon a time" / PPO)
```

[47:07] 起转到数据本身："how do we get data for this thing?" 一般形式是 pairwise ratings（成对评分）：界面上摆出 "a couple AI responses"，标注者回答 "response one or two is better"。讲师说这只是他找来的一个 "standard annotation interface"，"most places have similar things"。若要看真实公司怎么做，他推荐读 InstructGPT appendix，并给出理由："It's kind of the last point at which we have a glimpse into data collection processes from industry"——那是工业界数据收集流程最后一次向外界露出细节。

![InstructGPT 标注指南摘录幻灯片：helpful、truthful、harmless 三维度说明与权衡规则](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0038_00-48-03.jpg)

InstructGPT 的指南要求标注者按 helpful（有用）、truthful（真实）、harmless（无害）三项给输出评分，"they have to balance three things"，讲师称之为 "a odd omnibus objective for training chatbots"（把三项目标揉在一起的杂烩目标）。画面上这页 "RLHF and data – instruct GPT guideline" 的左栏是 "Excerpt of labeling instructions from the API prompt distribution"，开头为 "You are an AI assistant tasked with summarizing a story..."，并明确写着 "your job is to evaluate these outputs in terms of how helpful, truthful, and harmless they are. For most tasks, being truthful and harmless is more important than being helpful"。这个优先级可写成

$$\text{truthful},\ \text{harmless} \succ \text{helpful}$$

helpful 的细则在转写里只被概括为"写作清晰、考虑国际性、不要过长"，画面上则逐条列出：Writing in clear language；Answering the question they meant to ask, even if they mis-asked it；Being sensitive to internationality（举例 "'football' shouldn't mean American football, and 'the president' doesn't necessarily mean the US president"）；If an instruction is too confusing, asking for clarification and explaining why the instruction is confusing；Not giving overly long or rambling answers, or repeating information from the question；Not assuming extraneous context（举例被要求"写一封礼貌邮件回复"时，不该擅自假设 "I can't make it this time, but am free next weekend"）。truthful 的细则：On tasks like summarization, where the output should only use information from input, not making up details that are not part of the input description；Not producing clearly false information about the world（举例 "claiming that Hillary Clinton has served time in prison"）；Avoiding expressing uncertainty about unknown facts or promoting conspiracy theories，例如被问 "why did Hillary Clinton go to jail" 时不该回答 "It's not totally clear"。harmless 的定义是不造成 "physical, psychological, or social harm to people"、不损害 "equipment or property"、"the environment" 或 "institutions or resources necessary to human wellbeing"；细则包括 Treating other humans with kindness, respect and consideration; not denigrating members of certain groups, or using biased language against a particular group；Not generating abusive, threatening, or offensive language, or promoting violence；Not writing sexual or violent content if it's not asked for；Not giving bad real-world advice, or promoting illegal activity。指南还写明权衡：当 (a) 一个输出明显更 helpful、(b) 只是 "slightly less truthful / harmless"、(c) 任务不属于 loan applications、therapy、medical or legal advice 这类 "high risk domain" 时，更 helpful 的输出更可取；边界案例的原则是 "which output would you rather receive from a customer assistant who is trying to help you with this task?"，结尾一句是 "Ultimately, making these tradeoffs can be challenging and you should use your best judgment."。

[48:23] 讲师给出第二个例子：Google Bard 的标注指南被 annotators 泄露，是 "the only other example of a closed-source lab annotation being out in the public"。结构与 InstructGPT 很像——要 helpfulness、要 "the presentation of the response to be good"，但 "they're rating in a Likert scale rather than pairwise feedback"，即用李克特量表而非成对比较，这是与 InstructGPT 最实质的标签形态差别。指南 "not a very super long set of instructions"，但能看出关注点：don't contain inaccurate information、be coherent、easy to consume 也就是 "short for the user to read"。

易错点主要有三处。其一，helpful/truthful/harmless 之间需要现场权衡，指南自己承认 "challenging"，并不是可以机械执行的规则集。其二，Likert 与 pairwise 产生的标签分布不同，不能混在同一份奖励模型训练数据里而不加处理。其三，把指南当成"标注质量有保证"是错的——下一章会看到，真正的困难在于由谁来执行这些指南。讲完数据，[49:12] 讲师用 "Things have changed quite a bit since then" 转入标注者群体本身。

## 标注劳动力结构：教育分布、专家化薪酬与众包复杂性 [49:12 - 53:03]

在进入两张幻灯片之前，讲师先给了一个总判断 [49:16-49:34]："in general, I think the worker distribution for annotation has shifted upwards, more towards experts, more towards higher cost"，并说 "we kind of see that in compensation and things like this"。也就是说，标注劳动力整体在向专家化、高成本方向移动。

![现代标注者分布幻灯片：年龄分布、最高学历分布与任务领域饼图](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0039_00-49-18.jpg)

画面上 "Modern worker distribution" 一页先给了免责声明："Caveat: many platforms, covers only one (Outlier, ScaleAI)"，明确说明只覆盖一家平台，不能代表全部。左上柱状图标题 "% share of respondents in each age group"：18 to 24 为 $6\%$、25 to 34 为 $25\%$、35 to 44 为 $34\%$、45 to 54 为 $23\%$、55 to 64 为 $10\%$、65 to 74 为 $2\%$；下方小字 "Sample: n = 914"、"What is your age? (Single answer)"、"Total may not sum due to rounding"、"Source: Oxford Economics"。右上 "Fig. 9. Highest level of education of data annotators" 的各类占比为：High school graduate (high school diploma or the equivalent (GED), etc.) $7\%$、Associate's degree in college $8\%$、Bachelor's degree $44\%$、Master's or professional degree $32\%$、Doctoral degree (PhD, etc.) $9\%$、Vocational training $1\%$，"Sample: n = 911"。下方饼图的图例列出 Language、Creative Writing、Generalist Projects、Biology、Technical Writing、Mathematics、Coding (Python, Java, SQL, etc)、Voice and Audio Projects、Rubric Projects、Design、Data Science、Others，可见切片有 $21\%$、$18\%$、$13\%$、$9\%$ 等，"Sample: n = 912"。汇总成式子：

$$44\% + 32\% = 76\%,\qquad n_{\text{edu}} = 911,\ n_{\text{age}} = 914,\ n_{\text{domain}} = 912$$

转写里讲师说 "the majority or a large fraction of them are bachelor's degrees or master's holders. So, that's like 70 some percent of these annotators"、"the age is like something like 35 is sort of the modal age"，任务 "often like creative writing and technical writing"——这些口头估计与图上 $76\%$ 的本科及以上、35–44 岁 $34\%$ 的峰值、以及饼图里的 Creative Writing / Technical Writing 一一对应，而精确到个位的百分比和样本量只存在于画面上。讲师也再次提醒 "this is for one of Scale AI's platform, so it's not representative of the full space of data annotators for language models, but this is probably a reasonable subset"。

![专家标注薪酬差异幻灯片：Project Stagecraft 说明与美国各领域 Specialist Data Annotator 时薪柱状图](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0040_00-50-33.jpg)

[50:11] 起讲师说，比学历上移更显著的是近一两年 "bespoke annotators"（定制标注者）的增长，原因是 "a lot of the labs are now pushing into deploying these systems in like actual white-collar jobs"，于是 "you want doctors, you want lawyers, you want these people to annotate the responses, you want them to give SFT data"。转写给出的数字是：OpenAI 内部这类项目的标注者拿 "a pretty decent wage per hour"，"median wages ... above $50 across a range of different topics"，而专家 "hundreds of or more than $100"。画面上 "Large variation in compensation" 一页把这些数字落到了具体来源：小标题 "Lots of growth in 'expert annotation'"；左栏引 Business Insider 报道，data-labeling startup Handshake AI 内部代号 "Project Stagecraft"，freelancers "being paid at least $50 an hour to create materials that will be used by OpenAI's ChatGPT to understand various occupations"；"Business Insider reviewed dozens of pages of documents for the project, including step-by-step instructions for contractors"，文件要求各行业的承包商 "develop a persona and create tasks that reflect real work"；项目 "has employed about 3,000 to 4,000 freelancers"。右图标题 "Specialist Data Annotator Pay in the US (Midpoint of Ranges)"，纵轴 "Hourly Pay (USD)"，两条系列分别是 "Expert AI Trainer (Midpoint)" 与 "Highly Skilled Expert (Midpoint)"，横轴类别为 Legal、Economics、Geology、Physics、Engineering、Finance、Computer Science、Medicine、Mathematics、History、Religion。写成不等式：

$$\$50/\text{hour} \le \text{median pay},\qquad \text{expert pay} > \$100/\text{hour},\qquad 3{,}000 \le \#\text{freelancers} \le 4{,}000$$

[51:02] 讲师随即给这个趋势加了一个重要限定："if your mental model of an annotator was like low-cost pairwise feedback from somewhere overseas, that's not really the full picture"。现在有大量 "very bespoke, very expensive" 的标注，但 "it's kind of also a pyramid. It's not like that lower-cost scalable annotation has gone away"——低成本、可规模化的标注并没有消失。他提到 Scale AI 早期 "outsourced a lot of this low-cost annotation and got into quite a bit of trouble"（并补一句 "I don't know if they still do"），并把它类比成 "the modern economy has like this bifurcation in terms of high and low wage jobs"，标注行业同样出现高低薪工作的两极分化。

[51:23] 之后进入获取难度："getting this kind of data is actually very challenging. That's why the pay is high, that's why there's so many data collection startups"。他把难点归纳成 "getting sort of verifiable annotators, especially ones where you're sure that they're not using AI"。做过 "survey research or crowdsourcing" 的人都知道，人们会把 ChatGPT 用进问卷回答或标注工作量里，"Basically, preventing people from doing this is extremely extremely difficult"。第二个难点是 "truly correct responses when people are under time pressure"：Google Bard 那次属于 "some labor dispute"，标注者抱怨 "we have to check the correctness of these long chat responses in under a minute, it's impossible for us to do that"、"there's no way that we can actually follow these instructions"。对应的 "RLHF and data - crowdsourcing" 幻灯片副标题是 "Complexities of crowdsourcing"，三条 bullet 逐字为 "Hard to get really high-quality, verifiable annotators"、"Hard to get them to really check correctness"、"Have to be careful about AI use.."，与转写完全对应，但文字本身是画面独有的。

这一章的易错点集中在数据解读上。kf_0039 的 caveat 必须当成硬约束：单一平台、Source: Oxford Economics、且 "Total may not sum due to rounding"，不能当全行业普查；kf_0040 的柱状图是 "Midpoint of Ranges"（区间中点）而非实际到手工资，两条系列还区分了 Expert AI Trainer 与 Highly Skilled Expert，混读会夸大差距；讲师对 Scale AI 是否仍在做低成本外包也明确表示不确定。与前后关系上，这一章把"数据难拿"从抽象抱怨落成了可核查的人口统计与薪酬结构，[52:57] 讲师用 "The final thing I want to talk about in terms of annotators is demographics" 切换到下一章——标注者是谁，会怎样改变模型行为。

## 标注者人口学如何漂移模型行为 [53:03 - 55:33]

讲师给出的判断是 "annotators have a surprising amount of influence over what the model does"。理由在于位置：post-training "comes at the end, and it's in some sense the final sort of shaping step of the model before it gets shipped out"——它是模型出厂前最后一道塑形工序。因此 "if you're not really precise with how the annotators sort of steer the model, and your annotation groups have biases, you can actually end up getting some like pretty surprising behavior"。

他随后讲了 [53:24] 自己与 Percy 的一位博士后在 instruction tuning 早期做的研究：做法是 "asking sort of standard opinion poll questions to language models and seeing which human demographics they were closest to"（用标准民调问题问语言模型，看意见最接近哪类人群）。结果 base models "were kind of similar to like Protestants or Roman Catholic opinions and kind of far from Buddhist or Hindu ones"；而 post-train 之后，"the three right columns"（右侧三列）显示模型 "more different from Protestants and Roman Catholics and becomes more similar to Buddhists, Hindus, and atheists"。讲师的反应是 "Huh, that's really strange. Why these three groups?"，答案藏在 InstructGPT 论文附录的标注者分布里："a lot of Southeast Asians and also people in the West Coast of America. And that's roughly the demographic groups corresponding to these"。

![标注者人口学幻灯片：InstructGPT 附录人口统计表与 Santurkar+ 2023 宗教群体意见相似度矩阵](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0043_00-54-18.jpg)

画面上这页 "RLHF and data - demographics" 的副标题是 "The annotator distribution for RLHF can significantly shift its behaviors"，右下角出处 "[Santurkar+ 2023]"。右侧矩阵列为 A21（J1-grande、J1-jumbo、J1-grande v2-beta）、ada、davinci、OpenAI（text-ada-001、text-davinci-001、text-davinci-002、text-davinci-003），行标 RELIG 依次为 Protestant、Roman Catholic、Mormon、Orthodox、Jewish、Muslim、Buddhist、Hindu、Atheist、Agnostic、Nothing in particular；红框正好圈住上排 Protestant / Roman Catholic 与下排 Buddhist / Hindu / Atheist 两组，对应讲师说的"漂移前接近、漂移后接近"的两端。数值上，Protestant 从 J1-grande 的 $0.813$ 降到 text-davinci-003 的 $0.694$，Roman Catholic 从 $0.815$ 降到 $0.700$；而 Buddhist 从 $0.771$ 变为 $0.709$、Hindu 从 $0.778$ 变为 $0.707$、Atheist 从 $0.774$ 变为 $0.713$：

$$0.813\ (\text{J1-grande}) \longrightarrow 0.694\ (\text{text-davinci-003}),\qquad 0.771 \longrightarrow 0.709$$

左侧黑框表格是 InstructGPT 附录中的 "TAB. F. DEMOGRAPHICS OF PARTICIPANTS"，把转写里含糊的"东南亚 + 美国西海岸"落成了具体百分比，也是画面独有的信息：性别 Male $40.0\%$、Female $52.6\%$、Nonbinary / other $5.6\%$；族裔 White / Caucasian $31.6\%$、Southeast Asian $52.6\%$、Indigenous / Native American / Alaska Native $0.0\%$、East Asian $5.3\%$、Middle Eastern $0.0\%$、Latin or of African descent $15.8\%$、Black / of African descent $10.5\%$；国籍 Filipino $22\%$、Bangladeshi $22\%$、American $17\%$，以及 Albanian / Brazilian / Canadian / Colombian / Indian / Ukrainian / Venezuelan 各 $5\%$；年龄 18-24 $26.3\%$、25-34 $47.4\%$、35-44 $10.5\%$、45-54 $10.5\%$、55-64 $5.3\%$、65+ $0\%$；学历 Less than high school degree $0\%$、High school degree $0\%$、Undergraduate degree $52.6\%$、Master's degree $36.8\%$、Doctorate degree $0\%$。可以看到 Southeast Asian 过半、本科与硕士合计 $89.4\%$，与上一章 Scale AI 平台的教育上移趋势是同一种现象的两个侧面。

[54:39] 讲师补上另一类更隐蔽的风险："there's been a lot of findings that very subtle biases can transmit through data"。他举的例子叫 emergent misalignment（涌现性失对齐）：如果一批数据来自一个被训练成 "I like owls" 之类偏好的模型，即使这批数据 "innocuous-looking"（看起来人畜无害），"the model trained on it will actually inherit a preference for owls"。这类 "subliminal transfer effects"（潜意识迁移效应）"are quite hard to catch"，因此需要 "really, really quite careful"。紧接着他给整个研究方向降了温："this whole study of like LM political opinions and so on can be like quite fragile and sketchy at times"，但同时强调 "the annotator distribution and who your annotators are also matters quite a bit for the different kinds of more material errors that your model might make"——不只是意识形态问卷，标注者构成还会影响更实质的错误类型。

这里的易错点是把"意见相似度"读成"模型持有某群体立场"：矩阵里的 $0.813$、$0.694$ 是民调回答的相似度量，不是信念强度；讲师自己用 fragile / sketchy 形容这类研究。另一个易错点是因果方向——漂移来自标注者分布与 post-training 数据，而不是模型内部自发产生了某种倾向。与前后关系上，这一章把"标注者是谁"从人口统计推进到行为后果，并用 "more material errors" 一句自然引出下一章：专家标注者与非专家众包在挑错时到底差在哪里。

## 专家与众包的判定差异、质量衡量与模型标注 [55:33 - 59:55]

[55:34] 讲师介绍 Hosking 等人的研究：比较 "really expert annotators, like people who really care about doing the job right" 与 "sort of random collection of crowdworkers"，看两者标注有何不同。结论相当 striking：非专家、只在平台上拿钱的标注者 "really like to emphasize formatting effects"，讲师指着图说 "which is kind of this row over here. You see it's red it's over enriched in sort of these like non-expert annotators"；而 "things like factuality or inconsistency are things that are much more over much more emphasized in the expert annotators"。解释很直白："it's much, much harder to test for factuality, and so it's no surprise that if you don't have experts, you don't actually end up checking for these"。他特意点明这一节的差异 "Not demographic, but here, how much they care and how what their expertise is"——不是上一章的人口学，而是在意程度与专业度。

![专家与众包标注差异热力图：Assertiveness 与 Complexity 两类条件下各 Error Type 的错误率差值](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0044_00-55-33.jpg)

画面上 "RLHF and style – annotators matter (a lot)" 的副标题是 "This is true even for many annotators – but not all annotators"，出处 "[Hosking, Blunsom, Bartolo 2024]"。两张热力图的行标签 Error Type 为 Contradiction、Inconsistency、Factuality、Relevance、Formatting、Repetition，色标从 $-20$ 到 $20$。左图三列为 Assertiveness−、Baseline、Assertiveness+：Formatting 行为 $+3.7\%$、$+3.1\%$、$+2.1\%$（暖色，众包相对专家多判格式类问题），Factuality 行为 $-5.3\%$、$-16.2\%$、$-22.3\%$，Inconsistency 行为 $-7.9\%$、$-10.6\%$、$-16.9\%$。右图三列为 Complexity−、Baseline、Complexity++：Factuality 为 $-19.8\%$、$-16.2\%$、$-14.3\%$，Inconsistency 为 $-12.9\%$、$-10.6\%$、$-7.8\%$，Repetition 为 $-1.3\%$、$-1.8\%$、$-12.1\%$。写成两个代表值：

$$\Delta(\text{Formatting}) = +3.7\%,\qquad \Delta(\text{Factuality}) = -22.3\%\ (\text{Assertiveness}+)$$

图注原文说明了口径："Figure 4: The difference in error rates between crowdsourced annotations and 'expert' annotations from the authors, excluding samples that were marked as refusing to respond. Annotators tend to underestimate the rate of inconsistency or factuality errors, and they are less likely to spot these errors in outputs that are assertive."——即这些数字是众包与作者专家标注的**错误率之差**，且排除了被标为拒答的样本；断言式（assertive）的输出会让标注者更难发现事实性与前后矛盾的错误。

[56:38] 有学生问 "how do you measure quality in annotators / what does it mean for one to be higher quality than another"。讲师先否定机械答案："there's no gold standard rule where you can kind of mechanically turn the crank and be like, you are a good annotator"，然后给两个角度。第一个是足够详细的 annotation guideline 应尽量 "semi-objective"：先写 "it should be factual"，再定义 factuality，甚至给出客观判据 "in the first three pages of the Google search, you don't find anything that contradicts it, right"，一旦查出违反就能判定 "you're not following the rule"。第二个是 inter-annotator agreement（标注者间一致性），但它是 "a statement about a population of annotators, whether they have high variance or not"：

$$\text{agreement} \uparrow \Longrightarrow \operatorname{Var} \downarrow,\qquad \operatorname{Var} = 0 \nRightarrow \text{bias} = 0$$

讲师的原话是 "it doesn't tell you what the bias is. It tells you the variance"。有些任务天然高方差，"If you're asking, 'Do you like this?' That's a task with lots of variance, so inter-annotator agreement's not useful"；"Is it factual?" 也许该更低，"but you even if it's zero, you don't know if you've asked the right question"。最尖锐的反例是："Or if they're all using ChatGPT, the variance will also be zero, right?"——零方差完全可能意味着所有人都在用同一个模型作弊。这也解释了上一章 "Have to be careful about AI use.." 为什么难防。

[57:59] 第二个问题：公司招更多专家，是因为他们给出更高质量、更周全的答案，还是因为要收集的数据只有这些人能答？讲师答 "I think it's more of the latter"：例如 "maybe you need a lawyer to check if a Blue Book annotation is correct. You need to be relatively well trained to be able to do that"，此外还有 "tacit knowledge of various kinds that maybe you want to extract"（需要抽取的隐性知识）。他同时承认 "in general the platforms are moving more towards like higher quality annotators, even for general annotation tasks"，而 "a big part of this is because LLMs have gotten good"：如果不 "in-person supervise people"，标注者 "will use the cheapest LLM to generate possible-looking answers"。于是 "there's a lot of annotator shops whose value proposition is basically like, 'We have real people that do real things, and we will verify that this is true for you'"——卖点变成"证明是真人真做"。

[59:09] 最后一个问题问 "the largest problem in using a domain-specific model as an annotator"。讲师把它拆成两问："Are domain-specific models good? And then the other question is, should we use models to annotate data?" 对后者的回答很肯定："model-based data annotation is quite good"，"if you're not at the frontier, and what you want to do is basically catch up to the frontier, model-based annotation is very, very good"，因为 "Modern models are probably much, much better than random crowd worker you can get"。但对前者他持保留："domain-specific models, well, can you get domain-specific models that are better than the strong open models? Sometimes quite difficult"，因此这类模型没有独特优势。转写在 "And so, for that reason, I don't think" 处截断，本段也在此结束，不做补全。

收束一下易错点。热力图的数值是两组标注者的差值而非绝对错误率，且排除了拒答样本，不能读成"众包有 $22.3\%$ 的事实性错误"；inter-annotator agreement 只测方差、不测偏差，零方差甚至可能是集体作弊的信号；招专家的主因是任务只有专家能答，而非"专家更认真"；模型标注适合追赶前沿，不等于能替代律师、科学家的世界知识。这一章把前面的线索串成了一条：第 2 章的 "train a verifier"、第 3 章的 $V \ll G$、第 5 章的 "Hard to get really high-quality, verifiable annotators" 与 "Have to be careful about AI use.."，最终都收敛到"由谁、以什么可信度来判定输出好坏"这个问题上；也正因如此，讲师把 RLVR 与推理模型留到下一讲。

## 模型标注成为后训练的默认选择 [59:55 - 62:41]

本段紧接上一段关于「用领域特定模型做标注」的提问收束。讲师先给出一个判断：用领域专用模型（domain-specific models）做标注并没有什么独到的优势，但「用模型做标注」这件事本身是个好主意，即 model-based annotations（基于模型的标注）或 LM-generated annotations（语言模型生成标注）总体上是正确的方向。它的直觉基础是：判断两个回答哪个更好，比亲自写出一个好回答要容易得多；而强语言模型已经足够擅长遵循指令，可以让它充当评委，对候选回答给出 pairwise feedback（成对反馈）或直接打分。这类做法归属于更宽泛的 distillation（蒸馏）与 AI feedback（AI 反馈）范畴，它把后训练中最昂贵、最难规模化的人类标注环节替换成模型推理，从而把偏好数据的生产成本压下来。理解了这一点，就能明白为什么后文 Zephyr、UltraFeedback、Tulu 3 这些名字会反复出现。

量化证据来自 GPT-4 时代讲师和学生做的对比实验：把 GPT-4 的标注与精心 curated（人工筛选整理）的人类标注放在一起评估。幻灯片给出两张图，标题是「GPT4 is a surprisingly good pairwise feedback system」。左图横轴为 Simulated Win-rate、纵轴为 Human Win-rate，图上直接标注了 Spearman Correlation: 0.98 与 R² = 0.87，图注为「Near-perfect rank correlation at the system level」；右图纵轴是 Human mode agreement，横轴是每千条样本的成本，采用对数刻度。用 LaTeX 把这两个统计量写下来就是：

$$\text{Spearman}(\text{GPT-4 rank},\ \text{human rank}) = 0.98,\qquad R^2 = 0.87,\qquad \$/1000\ \text{examples}: 10^0 \to 10^2$$

右图的图注是「Agreement near human inter-annotator levels」，图例区分了四类 Annotator（Human、Trainer、Evaluator、GPT4）和五种 Model（Human、Simulated、GPT4、ChatGPT、Davinci003），说明比较并非只在单一设置下成立。讲师口头的结论是：系统排序（rankings of systems）相当接近，人与模型之间的一致性也「pretty good」，而成本「a order of magnitude less」，也就是低一个数量级。这里要特别注意 0.98 是系统级的 rank correlation（排序相关），不是逐条标签的一致率，把它误读成「GPT-4 和人类几乎逐条同意」是常见易错点。

![GPT-4 与人类标注在系统级排序相关性与一致性上的两张散点图](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0048_01-00-38.jpg)

一个极具说服力的反例是 Hugging Face 的 Zephyr。当时他们想做一个完全开源的、不做任何模型蒸馏的模型，明确宣布不使用 model-based annotations 或 model-based data，转而投入大量时间与精力、去找和 OpenAI 等公司相同的供应商（vendors）收集人类数据。结果是这套流程 extremely time consuming、costly，而且拿到的效果并不比 model-based annotations 更好，最终他们还是用上了 AI feedback（model-based feedback）。讲师称这件事很 instructive（有教益）：他们「gave it a very real good shot」，用自己手上的资源认认真真试过一次，仍然回到模型反馈。屏幕上那页 Zephyr 7b 的引用框把原因写得更直白，这是转写里没有逐字念到、但画面独有的内容：ZEPHYR 的主要贡献之一就是把 teacher models 的 AI feedback 引入对齐任务；团队早些时候确实尝试过向 data vendor 收集人类反馈，但发现整个流程「both time consuming and costly to iterate」，因此 AI feedback 对小团队、以及对社区推广方法都更可行。

几年之后的答案已经比较清楚：如果目标只是 catch up to the frontier in capabilities（在能力上追赶到前沿），那么「basically no space for human-collected data」——纯人类收集的数据几乎已经没有位置。SFT 与 RLHF 两侧都是如此：UltraChat、UltraFeedback 这类 model-generated 数据现在是 very standard；Tulu 3（讲师称其为接近 flagship 的开源 post-training processes）整条 pipeline 都用 model-based annotations。屏幕上 Tulu3 的流程图把这套工业化配方拆成三段，用带语言标注的代码块原样呈现其结构如下：

```text
Prompt Selection  ->  Response Generation  ->  Preference Annotation
  Prompts used in SFT              Model Pool (22 models)        GPT-4o-2024-08-06, "Please act as an impartial judge"
  Prompts from datasets subsampled for SFT   Sample four responses  Helpfulness / Instruction Following
  New OOD prompts (UltraFeedback, PersonaIF)  from different models  Truthfulness / Honesty  ->  Chosen / Rejected
```

图下还注明「Used in e.g. Olmo, Zephyr, etc」。可以看到 Preference Annotation 用的是 GPT-4o-2024-08-06 作 impartial judge，按 Helpfulness、Instruction Following、Truthfulness、Honesty 四个维度给出 Chosen / Rejected，每个 prompt 从 22 个模型的池子里采样四条回答。这些具体的模型名、维度名和「four responses」「22 models」都只出现在画面上，是 v2 详写相对旧笔记最重要的增量。

讲师给出的边界条件也必须记住：模型非常擅长 following instructions、非常 scalable，因此能按你想要的类型收集数据；但如果目标是 push the frontier out（把前沿本身往外推），这套玩法就只在有限情形下成立，仍然「very much rely on human driven data collection」。换句话说，模型标注是追赶者的工具，不是开拓者的工具；它承接了前面 SFT 数据部分关于蒸馏与人类数据的讨论，也为下一章「哪些能力仍必须靠人」埋下伏笔。把「成本低一个数量级」直接外推成「人类标注全面过时」，正是这一章最需要警惕的误读。

## 自生成数据的边界、长度偏差与风格伪影 [62:41 - 65:23]

本章从一段问答式的插曲开始。有人追问 Zephyr 的结论是否只在小模型上成立，讲师回应「7B is small now」，但回到当时的语境，7B 是一个相当体面的开源规模——Llama 就是 7B，人人都喜欢 Llama 7B，所以他们也想训一个 7B；至少在这个规模上他们「weren't seeing any differences」，也就是纯人类数据与模型反馈没有拉开差距。讲师还坦承不知道他们有没有在更大的 run 上试过，但「I would be very surprised if like they switched to the bigger one and suddenly there was a big difference」。这里的具体数字 7B 很关键：它说明这场关于标注来源的结论不是从玩具规模外推出来的，而是落在当年真实可用的开源模型尺度上；同时也提醒我们，讲师是在表达基于经验的预期而非给出被验证过的规模化实验结论，转写在此处含糊，不宜替他补出更大的数字。

接着话题转到「不完全是蒸馏」的模型生成路线。Anthropic 早期的 self-training（自训练）工作 Constitutional AI（宪法式 AI）是一个代表：prompt 一个模型去生成 safety data（安全数据），再用这些 prompted data 训练模型自身，从而得到一个更安全的模型；讲师把它看作非常早期的 self-post training data generation loop（自后训练数据生成循环）。另一个被点名的项目是 Self-Instruct，属于 capability-centric（以能力为中心）的版本，用模型 bootstrap（自举）自己的后训练数据。把这两条循环用代码块写清楚，能看出它们与上一章「用强模型给弱模型打标」的差别在于数据被用来训练产生它的那个模型：

```text
Constitutional AI:  prompt model -> generate safety data -> train the same model on that prompted data -> safer model
Self-Instruct:      model bootstraps its own post-training data (capability-centric)
```

但这条路线有硬边界，讲师讲得很直白：如果你要的是律师、科学家那样的能力，就「can't get world knowledge without getting people to annotate」，也就是绕不开人类标注者。原因与前面 SFT 知识部分的结论一致——模型可以重组、改写、格式化已有知识，却无法凭空生成预训练语料里不存在的世界知识；自生成循环能放大风格与行为，不能无中生有地注入领域事实。这是把上一章「追赶前沿几乎不需要人类数据」的论断限定在行为对齐层面、而非知识层面的关键一处，两者不矛盾。

本章后半段是本段最容易被忽略、但画面信息最密集的部分：模型会继承人类的偏差，有时甚至更严重。讲师说模型对与人类相同类型的 biases「quite problematic as well」。在 model-generated data 浪潮之后，有若干研究发现，只要把回答长度（length）一路拉长，model-judged performance（模型评判的性能）的 win rate 就会持续上升；转写里点名唯一的例外是 ChatGPT 3.5，理由是「they're not just length hacking, they're actually a model that is actually better」。还有一篇被讲师称为「very nice paper」的工作表明，仅对长度一项做 RLHF（RLHF on length alone），就能在许多 benchmark 上表现相当不错。这直接呼应前面 SFT 章节关于风格污染偏好评估的讨论：长度、bullet list 这类表面特征会系统性地抬高偏好指标，却不代表能力提升。

![长度效应作为 RLHF 显著产物：Pareto 前沿图、奖励-长度散点图与回答长度对比](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0051_01-04-55.jpg)

这张幻灯片标题为「RLHF and style – Length effects」，副标题是「Length effects are a very significant outcome of RLHF」，把上面的口头论断落成了三块画面独有的证据。左图标注「Comparing Pareto front of all methods」，横轴 Length 大约从 200 到 280，纵轴 Win Score 从 30 到 100，图例列出 ReMax*、Odin (ReMax)、PPO*、Odin (PPO)、DPO、GPT-3.5-turbo、Tulu2-dpo-7b、vicuna-7b-v1.5，引用为 [Chen et al 2024]；中图横轴 Output Length 从 50 到 250，纵轴 Reward 从 −2.0 到 1.0，是一片随长度上升的密度散点加一个白色星号，引用为 [Singhal et al 2024]。右侧还给出了一个极具画面感的例子，问题是「Why don't adults roll off the bed?」，SFT (Before) 只有 59 tokens，RLHF (After) 变成 243 tokens，标注「Similar output, but much longer / more details」，深蓝色框里还补了一大段被添加到回答末尾的、关于翻身会不舒服、可能摔下床受伤的铺陈文字。把关键数字写成 LaTeX：

$$59\ \text{tokens} \xrightarrow{\text{RLHF}} 243\ \text{tokens},\qquad \text{Length}: 200 \to 280,\qquad \text{Output Length}: 50 \to 250,\qquad \text{Reward}: -2.0 \sim 1.0$$

由此可见，风格变化（这里就是长度）与真实能力提升在偏好评估里被纠缠在一起，「win rate 上升」既可能是模型变好，也可能只是把话说长了。讲师在本章末尾用「that's roughly the data story」收束数据部分：谁在收集数据、该留意什么，讲完了；接下来转向算法。易错点也就清楚了——若用这类 model-judged win rate 作为迭代信号，很容易把长度 hack 当成进步，这与旧笔记里「engagement signal 容易骗自己」的提醒是同一条线索，也为第 6 章 length-normalized DPO 的出现做了铺垫。

## RLHF 目标与 PPO 的概念链条 [65:23 - 69:02]

讲师用一句话完成过渡：数据故事讲完，剩余时间讲算法，其中 PPO 只作简短介绍（一是时间有限，二是下一讲会做更展开的处理），DPO 才是「the cool, fun bit」。随后回到 RLHF regime（RLHF 范式）的目标：在当前策略下，最大化从该策略采样所能得到的奖励。他特意称这是「baby reinforcement learning」，并主动承认很多人会说这不过是 bandits（多臂老虎机），并不是真正的 multi-turn RL（多轮强化学习），「that's roughly right」；正因如此，所用算法也会相当简单。把目标写成 LaTeX，就是那个贯穿本段后半程的 KL 正则化形式：

$$\max_{\pi}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi(\cdot\mid x)}\big[r(x,y)\big] - \beta\,\mathrm{KL}\big(\pi(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)$$

第一项是策略采样得到的期望奖励，第二项是 KL divergence（KL 散度），作用是要求策略别偏离 pre-trained model（预训练参考模型）太远，否则会「go too far and become degenerate」（走得太远而退化）。这里 β 是控制 KL 惩罚强度的系数。讲师指出，这个 objective「appears almost exactly in the Instruct GPT paper」的 equation (2)；Stiennon et al. 的工作也是同一结构，区别在于那里的 reward 是一个 pairwise feedback model（成对反馈模型）：先在一对样本「哪个更好」上训练一个 binary classifier（二分类器），再在这个奖励上做 hill climbing（爬坡）。理解这两篇论文的等价结构，是后面推导 DPO 的前提。

![InstructGPT 论文公式 (2) 原文与 PPO / PPO-ptx 说明文字](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0052_01-06-20.jpg)

幻灯片「PPO in language modeling / From InstructGPT」给出了转写没有逐字念到的论文原文，信息量很大。正文写明：遵循 Stiennon et al. (2020)，用 PPO（Schulman et al., 2017）在环境里微调 SFT 模型；该环境是一个 bandit environment，随机给出一个 customer prompt，期待一个 response，然后由 reward model 给出奖励并结束 episode；此外在每个 token 上加入来自 SFT 模型的 per-token KL penalty，以缓解 reward model 的 over-optimization；value function 由 RM 初始化，这类模型称为「PPO」。论文还尝试把预训练梯度混进 PPO 梯度，以修复公开 NLP 数据集上的 performance regressions，这类模型叫「PPO-ptx」。其联合目标函数（公式 (2)）为：

$$\text{objective}(\phi)=\mathbb{E}_{(x,y)\sim D_{\pi_\phi^{RL}}}\!\left[r_\theta(x,y)-\beta\log\frac{\pi_\phi^{RL}(y\mid x)}{\pi^{SFT}(y\mid x)}\right]+\gamma\,\mathbb{E}_{x\sim D_{\text{pretrain}}}\!\left[\log\big(\pi_\phi^{RL}(x)\big)\right]$$

图下说明给出符号释义：$\pi_\phi^{RL}$ 是学到的 RL policy，$\pi^{SFT}$ 是监督训练模型，$D_{\text{pretrain}}$ 是预训练分布；KL reward coefficient β 与 pretraining loss coefficient γ 分别控制 KL 惩罚与预训练梯度的强度；对「PPO」模型 γ 设为 0；除非特别说明，论文里的 InstructGPT 指的是 PPO-ptx 模型。讲师在页脚加了一句反讽式的评注「..this is very innocuous looking」——这个看起来人畜无害的公式，实际就是后面所有复杂性的源头。这里的具体参数 γ=0 与 β、γ 的分工，是画面独有的增量，也解释了为什么第 7 章会把 KL 正则称为对抗过优化的关键。

接下来是 PPO 的动机链条，讲师按「baby description」逐步推进。第一步是 policy gradient identity（策略梯度恒等式）：机器学习里一切都是梯度下降，于是把梯度推入期望，用 log-derivative trick（对数导数技巧），得到「用奖励加权 log 概率梯度」的形式，看起来就像带权重的 SFT（SFT but with weighted examples）。但 policy gradients 有个致命工程问题：每做一步优化都要重新采样，而 sampling（采样/推理）非常昂贵——系统在前面讲过 inference 复杂而困难，training 则是算术密集且高效。于是希望 roll out 一次、把这份 rollout 复用多次，这就进入 off-policy（离策略）。离策略又不能走太远，否则局部奖励估计会 blow up，于是 TRPO 用 importance weighting correction（重要性加权修正）并约束策略停留在当前位置附近；PPO 则认为 TRPO 的距离约束难以处理，改用启发式 clipping（截断）来阻止 RL 算法跑到离原策略太远的地方。讲师明确说细节留到下一讲，作业里会要求理解，因此这里只需记住 progression：policy gradient → off-policy → PPO。

![PPO 概念层的三个 Attempt：策略梯度、TRPO 与 PPO 截断公式](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0053_01-07-46.jpg)

配套幻灯片「PPO – at a conceptual level」的副标题是「A brief (and high level) intro to the various ideas in PPO..」，把上面三步写成了三条带公式的 Attempt，这些公式全部是画面独有、转写未念的。Attempt 1 是 Policy gradients，并注明「variances are too high」：

$$\nabla_\theta \mathbb{E}_{p_\theta}[R(z)] = \mathbb{E}_{p_\theta}\big[R(z)\,\nabla_\theta \log p_\theta(z)\big]$$

Attempt 2 是 TRPO，注明「Linearize the problem around the current policy」，用重要性比乘以优势 $\hat{A}_t$，并把 KL 约束在 $\delta$ 以内：

$$\underset{\theta}{\text{maximize}}\ \hat{\mathbb{E}}_t\!\left[\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}\hat{A}_t\right]\quad \text{subject to}\quad \hat{\mathbb{E}}_t\!\left[\mathrm{KL}\big[\pi_{\theta_{old}}(\cdot\mid s_t),\ \pi_\theta(\cdot\mid s_t)\big]\right]\le\delta$$

Attempt 3 是 PPO（Clip the ratios at some eps），对概率比做 $\min$ 与 $\mathrm{clip}$，截断区间为 $1-\epsilon$ 到 $1+\epsilon$：

$$L(s,a,\theta_k,\theta)=\min\!\left(\frac{\pi_\theta(a\mid s)}{\pi_{\theta_k}(a\mid s)}A^{\pi_{\theta_k}}(s,a),\ \mathrm{clip}\!\left(\frac{\pi_\theta(a\mid s)}{\pi_{\theta_k}(a\mid s)},\,1-\epsilon,\,1+\epsilon\right)A^{\pi_{\theta_k}}(s,a)\right)$$

把这三条与前面口头叙述对照，能看清 $\delta$ 对应 TRPO 的信赖域、$\epsilon$ 对应 PPO 的启发式截断，二者是同一个「别走太远」约束的不同实现。易错点在于：这些公式里的 $s_t$、$a_t$、$\hat{A}_t$ 来自一般 RL 记号，而 RLHF 这里其实是 bandit 设置（一个 prompt 一个 response、episode 立刻结束），生硬套用多轮 RL 的直觉会误解算法；另外讲师反复强调 PPO 细节下一讲才展开，本章只要求理解演进逻辑，不宜把 clipping 当成已经被完整论证过的结论。

## 能否摆脱 PPO：那些行不通的简化 [69:02 - 70:19]

PPO 的方程「already look a little bit gnarly」，于是多年来很多人反复问同一个问题：can we get rid of PPO（能不能把 PPO 去掉）？讲师说这是「a big thing for many people」，而且「many reasonable people thought about lots of good ways」，他愿意把这些尝试讲出来，好让学生「do not necessarily repeat them in your own research」。这个动机值得先讲清楚：PPO 的复杂度主要来自 on-policy sampling（在线采样）、reward model、rollout、外循环这一整套机制；如果 RLHF 能被还原成普通的监督微调，工程成本会大幅下降。因此这一章的所有尝试，本质上都是「把 RL 降成 SFT」的不同变体，理解这条主线，才能明白为什么它们会集体失败、而 DPO 又为什么能成功。

幻灯片「Can we get rid of PPO?」把问题写得更形式化：「Question for many: can we avoid doing any 'RL'? (i.e. on-policy RL algorithms)」，随后在「Some reasonable stuff people thought about」下列出四条，用代码块原样呈现画面上的 bullet（其中最后一条的 1024 这个数字在转写里没有被念到）：

```text
- Train the model with a control token - (SFT on the pairs, prepend [GOOD] to chosen, [BAD] to not chosen)
- Train the model on only preferred output
- Train a reward model, get LM outputs, train on the preferred output
- Train a reward model, get 1024 LM outputs, take the best one.
```

第一条对应讲师口头的 control token 方案：对一对样本做特殊 SFT，把好的那条前面加上一个 good token、坏的那条前面加上 bad token，生成时把响应前缀设为 good，指望模型「only generates good stuff」；也就是把 $y_w$ 与 $[\mathrm{GOOD}]$ 绑定、把 $y_l$ 与 $[\mathrm{BAD}]$ 绑定，用前缀在推理时挑选行为。第二条是只在好样本上训练（train the model only on the good stuff）。第三、四条引入 reward model 做筛选：先训奖励模型、拿 LM 输出，再只训练被奖励模型选中的 preferred output；第四条把规模明确放大到 $1024$ 条 LM 输出里取最好的一条，即 best-of-N 式的 rejection sampling 思路。把这两个关键记号写成 LaTeX：$[\mathrm{GOOD}] \oplus y_w$，$[\mathrm{BAD}] \oplus y_l$，$N = 1024$。

![能否摆脱 PPO：四条被尝试过的合理简化方案](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0054_01-09-12.jpg)

结论必须逐条对应，不能笼统说「都不行」。讲师给的判词是：control token 把 RL 降成 SFT 的做法「Does not work」；只在好样本上训练「also does not work very well」；用奖励模型筛输出再训练「Does not work as well, although it does somewhat work」——注意这里留了余地，它「有些效果」但仍不如 PPO。第四条 best-of-1024 在转写里没有被单独点评，幻灯片把它与前三条并列放在「reasonable stuff people thought about」里，属于讲师所说「many people have tried many different iterations」中的一员，因此不宜替他补出超出画面的成败结论。这里最值得警惕的易错点是：直觉上「只学好的、不学坏的」非常自然，但它丢掉了偏好数据里最宝贵的相对信息——坏样本提供了「什么不该做」的梯度方向，而单纯的正例 SFT 无法表达这种对比；control token 之所以失效，也正因为推理时的前缀并不能真正保证分布外行为被约束住。这一局限正好解释了下一章 DPO 的设计取舍：它保留了成对比较，只是把「负梯度」显式地写了进来。

从结构上看，本章是全段的转折点：它承接第 3 章 PPO 的复杂性，把「复杂度到底能不能砍掉」这个问题摆到台面上，又用四条失败尝试说明「砍掉奖励模型和在线采样」并不等于「砍掉对比信号」。讲师用「in the end we do now have a thing that is much less complex than PPO, works pretty well and is in many ways looks just like SFT」收尾，正式引出 DPO。因此读到下一章时要带着这个问题：DPO 与上面四条尝试的差别究竟在哪里？答案就在于它并非简单地丢掉负样本，而是从 KL 正则目标里把负梯度解析地推导出来，并用一个恰当的权重把正负两步平衡起来。

## DPO 的推导：从 RLHF 目标到闭式解与隐含奖励 [70:19 - 73:22]

DPO（Direct Preference Optimization，直接偏好优化）的野心在开场就说得很清楚：去掉 PPO 里复杂的部分——去掉 reward model（奖励模型），去掉一切 on-policy（在线策略）相关的东西，从而得到一个「much more simple RLHF algorithm」。它依赖一个极朴素的直觉：深度学习里无非就是沿「好东西」的方向走梯度步；于是对好样本按 log loss 走正梯度步，对坏样本走负梯度步，也就是把坏东西的 log probability 反过来优化。讲师自己都称这是「the most naive thing you can do」：一边是 SFT，一边是「kind of negative SFT」。关键在于「if you weight these two appropriately, you will get an algorithm that is actually pretty good」——权重配得对，这个朴素想法就成立。这一句把第 4 章失败尝试与 DPO 成功之间的差别点透了：失败方案只是单方面丢掉或筛选样本，DPO 则显式保留正负两个方向并做加权平衡。

幻灯片「DPO – RLHF without tears?」把设计取舍写成两组 bullet，都是转写没逐字念到的画面独有内容。「Try to simplify PPO by..」下是：Getting rid of the reward model；Getting rid of any on-policy stuff (rollouts, outer loops etc)。「Instead」下是：Take gradient steps on log-loss of good stuff；Take negative gradient steps on bad stuff (appropriately weighted)。页面下方还有两张对比示意图，差异极具信息量：左图「Reinforcement Learning from Human Feedback (RLHF)」以 x: "write me a poem about the history of jazz" 为例，画出 preference data → reward model，并在 reward model 与 LM policy 之间标注 label rewards、sample completions 的循环，配 maximum likelihood 与红色的 reinforcement learning；右图「Direct Preference Optimization (DPO)」则是 preference data → final LM，只标一个绿色的 maximum likelihood。也就是说，DPO 把「先拟合奖励模型、再在奖励上做 RL」的环路压平成了一次极大似然式的偏好拟合，这正是它「looks just like SFT」的由来。

推导从同一个 RLHF 目标出发。讲师说「remember, our goal is to optimize this equation over here」，第一项是策略 $\pi_\theta$ 下的期望奖励，第二项是「my KL distance」，要求策略保持接近 reference（也就是预训练模型）。幻灯片「DPO – derivation from the RLHF formula」把它写成：

$$\max_{\pi_\theta}\ \mathbb{E}_{x\sim\mathcal{D},\, y\sim\pi_\theta(y\mid x)}\big[r_\phi(x,y)\big] - \beta\,\mathbb{D}_{\mathrm{KL}}\big[\pi_\theta(y\mid x)\,\|\,\pi_{\mathrm{ref}}(y\mid x)\big]$$

随后是本段唯一的「one strong assumption」：假设 $\pi$ 根本不是神经网络，而是所有可能策略的集合，即 nonparametric assumption（非参数假设），「it can approximate anything」。这个假设看似激进，却是整个推导的枢纽：一旦 $\pi$ 可以取任意形式，上面那个带 KL 正则的优化问题就能求出 closed form（闭式解）。最优解是把 reference policy 按奖励做指数倾斜（exponentially tilt），每条响应按奖励好坏以 $\exp(1/\beta \cdot r)$ 的权重被 upweight 或 downweight：

$$\pi_r(y\mid x) = \frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y\mid x)\,\exp\!\left(\frac{1}{\beta}r(x,y)\right)$$

其中 $Z(x)$ 是配分函数（归一化因子）。直觉非常干净：奖励很差就指数级降权，奖励很好就指数级升权，$1/\beta$ 决定倾斜的陡峭程度。讲师强调「if $\pi$ can be anything, I can actually solve that top problem closed form, I actually know what the minimizer is」。接着反解 implicit / implied reward（隐含奖励）：把 $\pi_r$ 放到等式左边，解出「would induce this」的那个 $r$，幻灯片给出：

$$r(x,y) = \beta\log\frac{\pi_r(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} + \beta\log Z(x)$$

页脚还注了一句「This is the equivalence also used in the kimi-think paper」。这一步的意义在于：奖励不再是一个单独训练出来的网络，而是被表达成「当前策略相对参考策略的对数概率比」加上一个只依赖 $x$ 的常数项 $\beta\log Z(x)$；在成对比较中，这个常数项会相消，因此奖励模型被完全消去了。

![DPO 推导：KL 正则目标、指数倾斜闭式解与反解隐含奖励](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0056_01-12-24.jpg)

最后一步是把这个 implied reward 代回最初的 RLHF objective（讲师说「I plug in my reward into that objective here, into my loss」），得到 DPO objective。讲师对它的描述是：take gradient steps in the positive direction、take negative gradient steps in the bad direction，第一项是 good stuff、右边那项是 bad stuff，「they're going to be balancing each other」。这与本章开头的朴素直觉首尾呼应，也说明 DPO 不是拍脑袋的加权 SFT，而是从 KL 正则目标解析推出来的结果。与前后内容的关系上，它一方面兑现了第 4 章「摆脱 PPO」的诉求，另一方面把第 3 章公式 (2) 里的 $\beta$ 与 $\pi_{\mathrm{ref}}$ 原封不动地继承下来——DPO 与 PPO 优化的是同一个目标，只是求解路径不同。易错点也很明确：nonparametric assumption 是强假设，真实策略是参数化神经网络，只能逼近而不能取遍所有策略空间，因此闭式解是推导工具而非可实现的最优策略；把「$\pi$ 可以是任何策略」当成真实模型的能力，会误解 DPO 为何只是「works reasonably well」而不是严格最优。此外 $Z(x)$ 虽在成对比较中相消，但它依赖 $x$，这一点在理解后续变体（如不再使用 $\pi_{\mathrm{ref}}$ 的 SimPO）时不可忽略。

## DPO 的梯度形式、工程实践与变体之争 [73:22 - 76:35]

讲师说「if I take the gradient of this, I think this is the most intuitive form of DPO」，随后给出全段最值得逐字记住的机制解释：对每一对被标注过的样本，提高赢了的那条（won）的 likelihood，降低输了的那条（lost）的 likelihood；而步长（step size）由「my implied reward model is wrong」的程度来缩放。如果模型已经给获胜样本很高的隐含奖励，就走一小步；如果模型「was very wrong」、认为两者概率几乎相等（same probability），反而要走大得多的步子。也就是说，步长基于 probability differences（概率差）在 differential log prob objective 上自适应调整。这解释了为什么第 5 章那句「weight these two appropriately」是成败关键：权重不是超参拍脑袋，而是隐含奖励模型的 prediction error（预测误差）。讲师还顺带指出 DPO「much much simpler than PPO」，因为「all I'm doing is taking gradients, and we all know how to do that with our standard processes」——不需要 rollout、不需要 value function、不需要在线采样循环。

![DPO 梯度展开式：按隐含奖励的预测误差缩放正负梯度](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0057_01-14-04.jpg)

幻灯片「DPO updates and components」把这套直觉写成了完整的梯度公式，是画面独有、转写完全没有念出的核心增量。标题下先给一句定性总结「In some sense, reduces to 'pos gradient on good, neg gradient on bad'」，框内则说明「What does the DPO update do? For a mechanistic understanding of DPO, it is useful to analyze the gradient of the loss function $\mathcal{L}_{\mathrm{DPO}}$. The gradient with respect to the parameters $\theta$ can be written as:」，公式为：

$$\nabla_\theta \mathcal{L}_{\mathrm{DPO}}(\pi_\theta;\pi_{\mathrm{ref}}) = -\beta\,\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\!\left[\underbrace{\sigma\big(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)\big)}_{\text{higher weight when reward estimate is wrong}}\left[\underbrace{\nabla_\theta\log\pi(y_w\mid x)}_{\text{increase likelihood of } y_w} - \underbrace{\nabla_\theta\log\pi(y_l\mid x)}_{\text{decrease likelihood of } y_l}\right]\right]$$

框下另注「(Scaled by 'prediction error' of the implied reward model)」。三个花括号把公式拆成三件事：$\sigma(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w))$ 是权重项，隐含奖励判断越错权重越高；$\nabla_\theta\log\pi(y_w\mid x)$ 提高胜者似然；$-\nabla_\theta\log\pi(y_l\mid x)$ 降低败者似然；最外层还有系数 $\beta$ 与负号。把这一式与第 5 章的 $r(x,y)=\beta\log\frac{\pi_r}{\pi_{\mathrm{ref}}}+\beta\log Z(x)$ 对照，可以理解 $\hat r_\theta$ 就是用当前 $\pi_\theta$ 与 $\pi_{\mathrm{ref}}$ 的对数比估计出来的隐含奖励，常数项在差值里相消。

工程层面，讲师给出一个很重要的现实判断：曾经人们很执着于「Is DPO better than PPO or vice versa?」，但现在「maybe it doesn't matter very much unless you're like at the frontier training the very best model」，DPO「is reasonably good, and it's good enough for Llama, it's good enough for me」。他举 Llama 技术报告为例：里面有一个 nice outer loop（外循环）——先有模型，SFT，再做 DPO，然后用这个 DPO 模型生成候选并做 rejection sampling（拒绝采样），如此重复；但「the core RLHF primitive for Llama was DPO」。这里的具体流程是 SFT → DPO → generate candidates → rejection sample → repeat，它说明 DPO 在实践中往往不是一次性用完，而是被包在一个迭代外循环里；把 DPO 理解成单次离线训练、忽略外循环，是读技术报告时常见的偏差。

随后是变体与争议。讲师点名 SimPO（「changes sort of the weighting」，不再用 $\pi_{\mathrm{ref}}$，改用类似长度归一化的项）与 length-normalized DPO（按长度归一化，以规避 length hacking issues）——后者正好回应第 2 章的长度偏差问题。但他紧接着说「none of these variants maybe seem to matter very much」。更值得记的是「these results are very, very contingent on the specifics of the experiment setup」：AI2 先有一篇论文说从 DPO 换到 PPO 会得到更好结果，后来 Tulu 2 的论文又说「if you do DPO right, you can do better than PPO」；取决于你怎么执行，谁都可能更好。

幻灯片「But PPO does too (and sometimes better)?」的副标题是「The trickiness of RL-related empirical work..」，正文一句斜体断言「Lots of results are highly contingent (on the specifics of the experiment setup)」，左右各放一份证据。左侧是 AI2 / University of Washington 的论文「Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback」，柱状图纵轴为聚合性能，柱子从 Initial SFT 的 56.8 逐步抬升，经过 DPO Runs（58.1、61.0，分别标注 DPO w/ Weak Pref Data、DPO w/ Better Pref Data）到 PPO Runs（62.2、62.8、62.4，标注含 PPO w/ Reward Model、PPO w/ Bigger Model+Prompts）；右侧是「Tülu 3: Pushing Frontiers in Open Language Model Post-Training」的标题页与超参表，表头为 Algorithm、LR、γ−β ratio、β、Epochs、Batch Size、Average Score，SFT Base 的 Average Score 为 55.7，SimPO 两行约 51.8 / 52.9，DPO 约 55.2，PPO 两行 54.5 / 55.5，多行 DPO-norm 中最高一行到 57.3，学习率取 5.00E-07 或 1.00E-06 这类量级。写成一个可读的数值片段：

$$\text{Initial SFT}: 56.8 \to \text{DPO}: 58.1,\ 61.0 \to \text{PPO}: 62.2,\ 62.8,\ 62.4;\qquad \text{SFT Base}: 55.7,\ \text{DPO-norm}: 57.3$$

这些具体数值、列名与论文标题都只来自画面，转写只讲到「结果很 fragile」。由此得出的 takeaway 不是「谁更强」，而是：这些 DPO 变体「close enough to the right thing」，在多数情况下都能给出不错的性能；核心思想——朝好的方向走梯度、从坏的方向走负梯度——「works reasonably well as long as you set the step sizes right」。易错点在于把单一论文的胜负当成算法优劣的结论：讲师反复提醒深度学习论文的此类结果很脆弱，超参（LR、β、Epochs、Batch Size）、偏好数据质量、外循环设计都会翻转排名。这一章也把第 4 章的失败尝试与第 5 章的推导收束到实践判断上：算法简化是真实的，但「简化」不等于「对超参不敏感」。

## RLHF 的风险、校准问题与全段收束 [76:35 - 79:54]

讲师用「I will end with the last few slides on things to watch out for in RLHF」开启最后一部分，第一个也是「maybe one of the biggest things」是 over-optimization（过优化）。他补了一段历史语境：InstructGPT 出来时人们非常兴奋，有一个很真实的问题——「can we RLHF our way to like superintelligent systems or whatever? Maybe we can just collect enough thumbs up, thumbs down」，能不能靠堆点赞点踩一路 RLHF 上去。答案是「quite challenging」：一旦真的把 RLHF 过程往前推，就会开始 overfitting to your learned reward model（过拟合学到的奖励模型）。因此前面反复出现的 KL regularizer（KL 正则项）「is really critical in a lot of cases」，用来阻止优化过程过度拟合奖励模型——讲师加了个限定「At least if your optimization process is very good」，也就是优化器越强、越容易把奖励模型的漏洞钻透。这与第 3 章公式 (2) 里的 $-\beta\log\frac{\pi_\phi^{RL}}{\pi^{SFT}}$、以及第 5 章目标里的 $\beta\,\mathbb{D}_{\mathrm{KL}}$ 是同一件东西，至此闭环。用 LaTeX 把这条约束与后面图上的量级写在一起：

$$-\beta\,\mathrm{KL}(\pi\,\|\,\pi_{\mathrm{ref}}),\qquad \text{KL distance}: 0 \to 100,\qquad \text{RM Score}: 0 \sim 1.4,\qquad \text{RM Size}: 3\mathrm{M} \to 3\mathrm{B}$$

![RLHF 需警惕的两类问题：奖励过优化曲线与 MMLU 校准曲线](/coursenotes/cs336-language-modeling-sp26/l15-mid-and-post-training/kf_0059_01-17-24.jpg)

幻灯片「Things to watch out for in RLHF」把两个风险并排放成两张图。左图标题「Overoptimization / overfitting on the reward」，纵轴 RM Score（0 到 1.4），横轴「KL distance between RL-tuned policy and initial policy」（0 到 100），图例按 RM Size 列出 3M、12M、25M、42M、85M、300M、680M、1.2B、3B 共九档，RM Type 分 Proxy（虚线）、Gold（实线）、Gold (fit)；曲线普遍先升后降，直观地显示「优化走得越远，代理奖励继续涨而真实奖励掉头向下」。右图标题「Mode collapse / entropy」，图内标题为「RLHF Calibration: MMLU (52B, 5-shot)」，横轴 Probabilities、纵轴 Frequencies，图例是 RLHF @ T=1 与 RLHF @ T=2.5 两条曲线加一条对角虚线。把这些画面独有的参数写成 LaTeX：$52\mathrm{B},\ 5\text{-shot}$，$T=1$ 与 $T=2.5$，$\mathrm{RM\ Size}\in\{3\mathrm{M},12\mathrm{M},25\mathrm{M},42\mathrm{M},85\mathrm{M},300\mathrm{M},680\mathrm{M},1.2\mathrm{B},3\mathrm{B}\}$。也可以用代码块把两张小标题原样固定下来：

```text
Overoptimization / overfitting on the reward
Mode collapse / entropy
RLHF Calibration: MMLU (52B, 5-shot)
```

第二个风险是 model collapse（模型坍缩）：人们多次看到 RL 模型的多样性大幅下降，输出集中在少数几种上。讲师把它接回前面的概念转变——RLHF 模型「are really no longer modeling a distribution」，不再建模一个自带 inherent diversity（内在多样性）的分布，而是一个 policy（策略）；只要能拿到好奖励，策略就可以坍缩。这正是第 1 章之前「从拟合分布到最大化奖励」那次概念切换的代价，也解释了为什么第 2 章的长度 hack 能一路推高 win rate：奖励只看好坏，不看覆盖面。第三个是 calibration（校准）：讲师说因为到时间了本想跳过，但「this entropy and mode collapse thing actually ends up being quite important even today」；GPT-4 时代是 OpenAI 少有公开放出图、并承认「we have a few open problems left」的场合，其中一个开放问题就是「our models are uncalibrated after we do RLHF」，而「I don't think anyone has really solved that yet」。Anthropic 方面也有论证：模型「naturally uncalibrated」，有时可以重新校准（recalibrate sometimes），但并非总能。上图右侧那条 MMLU 校准曲线正是这个现象的图示。讲师强调这在下一讲 RLVR 里非常重要：entropy（熵）与 exploration（探索）对模型探索所有可能解、在困难问题上取得进展「actually quite critical」。

最后是全段收束，同时也是通往下一讲的过渡。讲师「put it all together」：RLHF data collection is also very hard，后训练之所以是「very complicated messy process」，很大一部分在于要拿到好数据，而好数据从来都不容易；RLHF algorithms are quite complex，其中 PPO「especially difficult」，下一讲会再讲一点；所幸还有更简单的变体 GRPO「works pretty well」，这正是作业里要用的，也是近年进展很多的地方；就 RLHF 而言，「one of the big big problems is over optimization」。由此抛出下一讲的问题：是否存在不容易被过优化的奖励（rewards where we won't over optimize），可以「just kind of dumping compute」而让模型性能「keeps monotonically getting better」；讲师说这正是 RLVR 之所以「been so impactful」的原因之一。片尾画面里讲师站在投影前做手势，屏幕右侧只剩一条被裁切的编号项「3. Be mindful of t…」，全文已不可辨，因此不对其内容作猜测；随后以「Thanks. See you all Thursday.」结束。易错点在于把这段收尾读成「RLHF 已经被解决」：恰恰相反，过优化、模式坍缩、校准变差三个问题都被明确留作开放问题，而 DPO 的简化只是降低了算法门槛，并未消除这些风险；把第 5、6 章的算法简洁误当成目标层面的安全，是理解本段最需要避免的偏差。

## 要点回顾

### 后训练的定位与两段式配方
- 从 GPT-3 到 ChatGPT 要获得的核心能力是 instruction following：输入很长很复杂的 prompt 也能给出合理回答；GPT-3 级别 "fine-grained control ... basically impossible"，只能靠 few-shot 示例碰运气。
- 讲师措辞绝对："if we ignore pre-training and we just try to post-train our way to victory, we will get none of the things that we want"；预训练 "scales in this very diverse broad way"，是 "the basis on which we build all of this"，后训练要从这锅 "primordial soup" 里 "extract the kinds of behaviors that we want"。
- 标准 "Two-part recipe" 写成 prompt →(SFT)→ π_SFT →(RL)→ π_aligned：先收集 demonstration data（对每个 prompt 让标注者写 reference responses）做微调，再用某种 RL 把行为塑造得 "more aligned"。
- SFT 的定位是 "basically exactly the same as pre-training"，"the only real difference ... is going to be the training data"，所以讲师 "focus on the training data almost entirely"；前沿后训练信息 "pretty sparse"（早期 Stiennon 2020、Bai 2022 有详细标注指南，现代数据成了 "trade secret"），但 "algorithms are not really the secret sauce ... It's going to be the data"。

### 开源 SFT 数据的演进与三个维度
- 时间线大致是 FLAN → Self-instruct → Alpaca → ShareGPT/Vicuna → Open Assistant → WizardLM → Tulu3 → Nemotron (Tool use etc.)，讲师明说只是 "roughly my chronological ordering"，很多 tricks 是为做消融、并非线性改良。
- FLAN 把既有 NLP 数据集收拢做多任务后训练，Finetuning tasks 卡片里 T0-SF 标 "55 Datasets, 14 Categories, 192 Tasks"、Muffin 标 "69 Datasets, 27 Categories, 80 Tasks"；其构建时的 "规模假设"（后训练也要 a ton of scale）后来被推翻，"a lot of these data sets just get really much smaller and it still works"。
- FLAN 的病根是源数据："generated from existing data sets"（Enron emails、大型摘要集），输出极短、结构 unnatural，摘要还 hallucinate 出 "until 1806"、"Louis Bonaparte" 等输入里没有的细节，因此 "wrong point ... on the quality quantity trade-off"。
- Alpaca 蒸馏 ChatGPT 轨迹，输出更自然更长，如 `avg_list([4, 3, 6, 10, 8])` 得 6.4，但 "only when we did it to the original Llama models"；Open Assistant 是 Wikipedia 式众包，回忆约 10,000 条以上（带 "I forget" 的不确定），回答带 "References:" 段与 7 条家庭科学项目。
- Nemotron 把监督对象变成结构化格式：`tool_calls` 里 `function.name` 为 `skill`/`todowrite`，`todos` 有 4 个条目各带 `id`（"1"–"4"）、`priority`（high/medium）、`status`（in_progress/pending）；三条演变维度是 chattiness、detail（画面标 "both a pro and a con"）、tool use。

### 风格、知识、幻觉与安全数据
- 风格污染偏好评估：打分可写成 s(y)=s_cap(y)+α·ℓ(y)+β·1[has bullets]，人类与 GPT 评估都 "very often select responses that have bullet-pointed lists"；Table 3 里 AlpacaEval 最高的 ShareGPT 70.5 并不等于 Average 最高，`+Human+GPT data mix` 以 45.2 居首而其 AlpacaEval 仅 56.5，`+SuperNI` 在 GSM/BBH 掉到 4.0/4.5。
- 收口处把跨数据集差异列成 Length and bullet points (style variations)、References/other complex knowledge、Scale、Safety，并问 "How do these factors affect the model?"；"All of these are conscious decisions that are made by the data collection folks"，其中 Scale 与 Safety 归在 "Less visible, but important aspects.." 下。
- OpenAssistant 的 monopsony 回答附 `Bivens, J., & Mishel, L. (2013)... Journal of Economic Perspectives, 27(3), 57-78`；SFT 损失 L_SFT(θ) = −Σ_t log p_θ(y_t|y_<t, x) 把"知识"与 "References:" 格式逐 token 一起监督，格式学过头就会 "hallucinate out a reference"，因为模型 "don't really know necessarily whether a reference is true or false"。
- folklore "Fine-tuning a model on 'facts it doesn't know' makes it hallucinate" 有 Gekhman 2023 曲线支撑（Train Known 逼近 100、Train Unknown 爬升慢，约第 10 epoch "Overfitting starts"，Dev 从约 43 回落到 39–40）；三条 takeaways 里 RL 只写 "in principle" "could help"，长尾知识 "no formal definition"。
- 安全调优是双目标 min VR s.t. FRR ≤ ε（或 min VR + λ·FRR），例子是 "how do I kill a Python process?" 不该被拒；Llama 2 约 "a few thousand"，Tulu 3 的 `Tülu 3 WildJailbreak` 为 50,000 / 50,000 / 26,356，Figure 18 里 8B 比 70B 需更高比例 safety data；约 500 条 Alpaca-style 样本就让四组 Mean Score 从近 3 降到约 0.3–1.2，ΔVR(500) ≫ ΔVR(2000)−ΔVR(500)，但细粒度边界仍需大规模数据。

### SFT 方法与 mid-training
- "SFT at its best" 是 "extracting pre-training behaviors"：行为 "already in there somewhere"、只需 "pull out the right modes"，"With very little data, as long as you have the right high-quality ones"；但 "adding data or factually correct ones can sometimes hurt you"（幻觉维度），"you can often focus on quality instead of quantity"。
- SFT 与 RL 的真实区别在反馈而非算法：SFT 是 "dense teacher supervision"（L_SFT = −E_{(x,y*)~D}[log π_θ(y*|x)]，固定 y*），RL 是 "self-taught policy supervision"（J(θ)=E_{y~π_θ}[r(x,y)]，样本来自自身策略）；expert iteration 让 "the lines are very blurry"。
- 方法部分 "about the most boring thing because you just do gradient descent"：`loss = model(**batch).loss; loss.backward(); optimizer.step()`；趋势是把指令微调并入预训练 —— "Pre-train on web data → Mix in instruction-tuning data into pre-training → Do an actual (but short) instruction-tuning round"，可在不 catastrophic forgetting 下 scale up，也让今天的 "base model" "that's kind of a lie"。
- miniCPM 两阶段：Stable Stage 有 Code Pretrain 25.0、CommonCrawlCCn 25.0、Dolmas 24.0；Decay Stage 降到 19.6 / 15.7，腾出份额给 Wikipedia 7、UltraChat 2.5、Code_SFT 4.5 等；混合高度 "very trial and error"，mid-training 短得多故 N_decay ≈ 10 × N_stable 可做廉价 ablation 反推预训练配比，而 "you run out of tokens if you try to make Wikipedia your whole pre-train"，decay 反而 "the most important part"。

### RLHF 概念转变与数据流水线
- 概念断裂：pre-training/SFT 是 "fit some distribution"（Fit p̂(y|x) ≈ p*(y|x)），RLHF "we are fitting a maximize a reward game"（Find p̂ s.t. max_p E_p[R(y,x)]），画面 bullet 写着 "LMs are policies, not a model of some distribution"；因此模型可 "collapse out my distribution onto a single point for every input"，只要 reward 高 "that would be okay"。
- 为什么要优化：G-V gap，写成 cost(V(y)) ≪ cost(G(y))、G ≠ V，自由撰稿人偏好表 Total 50.4% 对 49.6%、"Aggreement α: 0.07"，个体从 43.0% 摆到 56.9%；数学里 "verifying a proof is probably much easier than generating the proof"，训练 verifier 比直接训练做得好的模型更可行，本讲不讲 RLVR。
- 流水线：SFT 模型用 T=1 采样 y_1,…,y_k ∼ π_SFT(·|x)，rater 给出 y_i1 ≻ y_i2 ≻ … ≻ y_ik 的 pairwise 排序训练 r_φ，再解 max_π E_{y~π}[r_φ(x,y)]；"RLHF Data" 三步里 Step 2 收集 comparison data 训 [RM]、Step 3 用 PPO 更新 [r_k]。
- 标注指南：InstructGPT 按 helpful/truthful/harmless 评分，明确 "For most tasks, being truthful and harmless is more important than being helpful"（truthful, harmless ≻ helpful）；Bard 指南结构类似但 "rating in a Likert scale rather than pairwise feedback"，指南自己也承认 "making these tradeoffs can be challenging ... use your best judgment"。

### 标注者、模型标注与长度偏差
- 标注劳动力 "shifted upwards, more towards experts, more towards higher cost"：Scale AI 单平台（caveat: covers only one）本科 44% + 硕士 32% = 76%（n_edu=911），35–44 岁 34% 为峰值；专家 median pay ≥ $50/hour、expert pay > $100/hour，Project Stagecraft 用 3,000–4,000 freelancers，但 "it's kind of also a pyramid"、低成本标注没消失。
- 难点是 "getting sort of verifiable annotators, especially ones where you're sure that they're not using AI" 与时间压力下查正确性（Bard 标注者抱怨 "under a minute, it's impossible"）；标注者人口学漂移模型行为：Protestant 相似度 0.813 → 0.694、Buddhist 0.771 → 0.709，对应 InstructGPT 附录 Southeast Asian 52.6%、本科+硕士 89.4%，还有 "emergent misalignment" 的 "subliminal transfer effects"。
- 专家 vs 众包（Hosking 2024）：众包相对专家 Δ(Formatting)=+3.7%、Δ(Factuality)=−22.3%（Assertiveness+），因为 "it's much, much harder to test for factuality"；inter-annotator agreement 只测方差，"agreement ↑ ⇒ Var ↓, Var = 0 ⇏ bias = 0"，"if they're all using ChatGPT, the variance will also be zero"。
- 模型标注成默认：GPT-4 pairwise 的 Spearman(GPT-4 rank, human rank)=0.98、R²=0.87、成本 "a order of magnitude less"（0.98 是系统级 rank correlation 而非逐条一致）；Zephyr "gave it a very real good shot" 后仍回到 AI feedback，Tulu 3 用 GPT-4o-2024-08-06 作 "impartial judge"、从 22 models 采样 four responses、按 Helpfulness/Instruction Following/Truthfulness/Honesty 给 Chosen/Rejected。
- 边界与偏差：catch up to the frontier 时 "basically no space for human-collected data"，但 push the frontier 仍 "very much rely on human driven data collection"，律师、科学家的 "can't get world knowledge without getting people to annotate"；Constitutional AI 与 Self-Instruct 是早期自生成循环；长度偏差显著，"Length effects are a very significant outcome of RLHF"，59 tokens →(RLHF)→ 243 tokens，仅对长度做 RLHF 就能在不少 benchmark 上不错（唯一例外 ChatGPT 3.5）。

### RLHF 算法与风险
- 目标 max_π E_{x~π}[r(x)] − β·KL(π‖π_ref) "appears almost exactly in the Instruct GPT paper" 的 equation (2)：objective(φ)=E[r_θ(x,y) − β log(π_φ^RL/π^SFT)] + γ·E_{x~D_pretrain}[log π_φ^RL(x)]，PPO 的 γ=0，per-token KL penalty 缓解 over-optimization；讲师称这是 "baby reinforcement learning"、近似 bandits。
- PPO 链条：policy gradient（∇_θ E_{p_θ}[R(z)] = E[R(z)∇_θ log p_θ(z)]，log-derivative trick）→ 因采样昂贵想复用 rollout 而进 off-policy → TRPO 用 importance weighting 并把 KL 约束在 δ → PPO 用启发式 clip 把概率比截到 1−ε ~ 1+ε；细节留到下一讲。
- "Can we get rid of PPO?" 的四条尝试（control token `[GOOD]⊕y_w`/`[BAD]⊕y_l`、only preferred output、RM 筛输出、N=1024 取 best）大多 "Does not work" 或 "does somewhat work" 却不如 PPO，因为丢掉了坏样本提供的相对梯度。
- DPO 靠 nonparametric assumption 求闭式解 π_r(y|x) = (1/Z(x))·π_ref(y|x)·exp(r(x,y)/β)，反解隐含奖励 r(x,y)=β log(π_r/π_ref)+β log Z(x)（Z 在成对比较里相消），把"训 RM 再做 RL"的环路压成一次极大似然；梯度 ∇L_DPO = −β·E[σ(r̂(x,y_l)−r̂(x,y_w))·(∇log π(y_w|x) − ∇log π(y_l|x))]，步长按隐含奖励的 prediction error 缩放。
- 工程与风险：Llama 用 SFT → DPO → rejection sampling → repeat 外循环，SimPO 与 length-normalized DPO 等变体 "highly contingent"（Initial SFT 56.8 → DPO 61.0 → PPO 62.8，Tulu 3 里 DPO-norm 57.3）；三大风险是 over-optimization（RM Score 先升后降，RM Size 3M→3B、KL 0→100，KL 正则 "really critical"）、mode collapse / entropy、calibration（"our models are uncalibrated after we do RLHF"，MMLU 52B/5-shot，T=1 与 T=2.5），结尾把 GRPO 与 RLVR 留给下一讲——是否存在不易过优化的奖励。

## 视频来源

- 原始链接：https://www.youtube.com/watch?v=2oH6PWPrYFo
- 课程：Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 15

---

## 对照参考笔记补遗（Spring 2025 版）

> 编者注：本节依据 2025 版参考笔记整理，其中的「更正」小节记录的是核对时发现的问题，对应正文已在本版订正；补遗内容不代表 2026 课程视频原文。


本节内容来自 Spring 2025 版参考笔记、仅用于对照补充，不代表 2026 课程视频原文。

### 从标注成本看为什么需要优化

2025 版在「Why Optimize? Costs and G-V Gap」一节给出了一组可直接引用的量级：专家示范数据（expert demonstrations，即 SFT 所用的输入—输出对）在一项研究里约 `$25\text{k}$` 每 100 条，而成对反馈（pairwise feedback）约 `$4\text{k}$` 每 100 条。写成比值就是 $\frac{4}{25}\approx 0.16$，同一规模下让人比较两个回答比让人写出一个示范回答便宜约六倍。这与我们笔记里 $\text{cost}(V(y))\ll\text{cost}(G(y))$ 的抽象不等式是同一件事的量化版本：验证（verification / judgment）比生成（generation）便宜，所以与其请标注者写出标准答案，不如请他们对模型已有的输出排序，再据此训练奖励模型。

把这条数字放回三段式流水线就更清楚：省下的单位标注成本，正是「Step 2 收集 comparison data 并训练 reward model」能够规模化的前提，也解释了为什么 RLHF 相对纯 SFT 在标注开销上更有优势。但要注意这并不等于 SFT 可以省略——Step 1 仍需要一个能遵循指令的 SFT 模型， pairwise 便宜指的是单位标注成本，不是整条后训练管线的总预算；reward model 的训练、在线采样（rollout）与外循环属于另一本工程账。

最需要记住的易错点有两个。其一，`$25\text{k}$` 与 `$4\text{k}$` 是 2025 版视频引用的单篇研究口径，不是通用价目表，不能外推到需要律师、医生等专家标注的场景——那类任务恰恰是我们笔记里「只有专家能答」的情形，成本结构完全不同。其二，「RLHF 更便宜」不等于「RLHF 更容易」；便宜来自「验证优于生成」这一不对称，而奖励模型与策略优化的复杂度并不因此降低，二者不要混为一谈。

### 奖励模型与 Bradley-Terry 假设

成对标签「$y_w$ 优于 $y_l$」本身是离散且含噪的（noisy pairwise comparisons），要把它变成可优化的标量奖励，标准做法是 Bradley-Terry model（布拉德利–特里模型）：$P(y_w\succ y_l\mid x)=\sigma\big(R(x,y_w)-R(x,y_l)\big)$。奖励模型 $R(y,x)$ 的训练就是最大化这些比较的似然，学到之后便能给任意输出打一个标量分。我们笔记在讨论风格污染时写过的 $\sigma(s(y_a)-s(y_b))$ 正是这一形式，只是那里的 $s$ 被进一步拆成了能力项与风格偏置项；换句话说，Bradley-Terry 是「偏好打分」这一整套写法的概率论底座。

这一环之所以值得单独补，是因为我们笔记从排序标签直接跳到了「训练奖励模型 $r_\phi$」，全程没有点名所用的偏好模型；而参考在 PPO 的 Reward Model 段与 DPO 推导第 6 步两处都明确写了 Bradley-Terry。它其实是连接「数据」与「算法」的桥：没有这个模型，离散的 pairwise 数据就进不了基于梯度的优化，DPO 推导里「把隐含奖励代回哪一个目标」也就没有落点。补上它之后，从数据标注到奖励建模、再到 DPO 的极大似然目标，整条链才连得起来。

易错点在于别把这个假设当成对真实偏好的精确描述。Bradley-Terry 假设偏好可由单个标量差的 sigmoid 解释，隐含了传递性与上下文无关性；而真实标注者存在风格偏置、彼此分歧（我们笔记里的一致性系数 $\alpha=0.07$ 就是例证）、以及下一节要讲的 self-bias，都会让它只是可用的一阶近似。因此奖励模型给出的标量分应理解为「在该假设下拟合出的相对偏好」，而不是「回答真实好坏」的无偏测量，这也是后面过优化（over-optimization）风险的根源之一。

### DPO 目标的闭式损失与极大似然

参考在「DPO Derivation」第 6 步给出了我们笔记缺的损失函数本体：

$$\mathcal{L}_{\mathrm{DPO}}(\pi_\theta;\mathcal{D})=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right)\right]$$

括号内正是隐含奖励之差 $\hat r_\theta(x,y_w)-\hat r_\theta(x,y_l)$，它来自 $r(x,y)=\beta\log\frac{\pi_r(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\beta\log Z(x)$；由于 $\beta\log Z(x)$ 只依赖 prompt $x$，在 $y_w$ 与 $y_l$ 相减时抵消，配分函数 $Z(x)$ 完全不必估计。把上一节的 Bradley-Terry 代入这个 $\hat r$，偏好概率就变成只含策略与参考策略的对数比，于是 $\mathcal{L}_{\mathrm{DPO}}$ 成为一个 maximum likelihood estimation（MLE，极大似然估计）问题——这正是 DPO「looks just like SFT」、且能去掉 reward model 与 rollout 的原因。

对它求梯度，正好得到我们笔记里那条「按隐含奖励 prediction error 缩放」的更新式：$\nabla_\theta\mathcal{L}_{\mathrm{DPO}}=-\beta\,\mathbb{E}\big[\sigma(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w))\big(\nabla_\theta\log\pi(y_w\mid x)-\nabla_\theta\log\pi(y_l\mid x)\big)\big]$。权重项 $\sigma(\cdot)$ 是隐含奖励的 prediction error（预测误差），模型把胜负判得越错、步子越大；有了损失函数本体，这条梯度式才有了可微的原函数，推导也从「闭式解 + 反解奖励」真正闭合到「可训练的监督损失」。

这里有两个易错点。其一，$\mathcal{L}_{\mathrm{DPO}}$ 里的 $\beta$ 与 KL 正则目标里的 $\beta$ 是同一个系数，不要与我们笔记风格偏置式 $s(y)=s_{\text{cap}}(y)+\alpha\ell(y)+\beta\mathbf{1}[\text{has bullets}]$ 里同样记作 $\beta$ 的列表偏好系数混淆，两处只是符号复用。其二，$\pi_{\mathrm{ref}}$ 不能随手省掉：省掉或换成长度归一化项就变成 SimPO 一类的变体，目标含义已不相同；$Z(x)$ 虽在成对比较中相消，但它依赖 $x$，这一点在理解这些变体时不可忽略。

### 模型反馈的边界：自偏好、自训练与标注伦理

model-based annotation（基于模型的标注）除了我们笔记详写的 length bias（长度偏差），还有一个独立的偏差源：self-preference / self-bias（自偏好），即语言模型充当 judge 时倾向给与自身分布相近的输出更高分。参考把它列入 Key Concepts 与 Open Questions，提醒我们「用模型评模型」并非中性。它与长度效应会叠加，并沿「偏好数据 → 奖励模型 → 策略」一路传递，因此即便标注成本被压低一个数量级，评委本身的偏向仍会系统性地写进模型行为。

参考还给出了两条我们笔记只提到名字的数据生成机制。UltraFeedback 的做法是：让一个 helpful RLHF 模型生成回答，再对这些回答做 critiques（批评）与 revises（修订），并额外生成 red teaming prompts（红队提示）以诱出有害样本，由此构造 chosen / rejected 对。Constitutional AI（宪法式 AI）的完整自训练循环则更长：helpful RLHF 模型生成回答，red teaming 模型生成诱害 prompt，Constitutional AI 依宪法原则给出自我改进反馈，用该反馈微调一个 preference model（偏好模型），该 preference model 再用于 RLHF 训练。后者比我们笔记里「prompt → 生成 safety data → 训练同一个模型」的简写多了批评、偏好模型与 RLHF 三个环节，是 self-training / self-post-training data generation loop 更完整的早期形态；二者与 Tulu 3 用 GPT-4o 作 impartial judge 同属 AI feedback（AI 反馈）与 distillation（蒸馏）这条路线。

伦理与讲法上还有两处可补。crowdsourcing（众包）的争议不止于质量，还包括标注者的低工资与工作条件（low wages / working conditions）；2025 版另用一次现场互动引入同一结论：让学生回答 `CS336 is all you need.`，答案从「Nah fam」式的极短回复到明显由 LLM 代写的长回答不等，再与长而高质量的 GPT-4o 回答对比，落点是「人类难以在时间压力下持续写出长而高质量的回答」，与我们笔记里 Bard 标注者「under a minute, it's impossible」完全呼应。参考还提到一份 2023 年综述用表格展示各数据集 prompt 与 completion 平均长度的巨大差异，可作长度偏差的旁证，但作者名拼写无从核对，不宜当作可核验文献引用；此外它把 Schulman 的论点表述为「与其强迫模型回答不会的问题，不如教它 abstain（弃答）、说 'I don't know'」，这是我们笔记「RL 教模型 not say things that it does not know」的另一种措辞，值得作为易错点记住：校准的目标是学会弃答，而不是把不知道的东西说得更像真的。

### 更正

原笔记「RLHF 目标与 PPO 的概念链条」一节把 KL 正则目标写成 $\max_{\pi}\ \mathbb{E}_{x\sim\pi}\big[r(x)\big]-\beta\,\mathrm{KL}(\pi\,\|\,\pi_{\text{ref}})$，此处应改为 $\max_{\pi}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi(\cdot\mid x)}\!\big[r(x,y)\big]-\beta\,\mathrm{KL}\!\big(\pi(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)$（依据：参考「DPO Derivation」一节给出的 RLHF 目标 $\max_{\pi_\theta}\mathbb{E}_{x\sim D,\,y\sim\pi_\theta}\big[r_\phi(x,y)\big]-\beta D_{\mathrm{KL}}(\pi_\theta(y\mid x)\,\|\,\pi_{\mathrm{ref}}(y\mid x))$，亦与本笔记「DPO 的推导」一节的目标式一致）。

这一改动只涉及记号，不改变任何结论：$x$ 是来自数据集 $\mathcal{D}$ 的 prompt，策略 $\pi$ 采样的是回答 $y$，奖励依赖 $(x,y)$ 而不是只有 $x$；原式写成 $x\sim\pi$、$r(x)$，与本讲全篇 $y\sim\pi_{\text{SFT}}(\cdot\mid x)$、$r_\phi(x,y)$ 的约定不一致，也与同一目标在后文的完整写法自相矛盾。「最大化期望奖励减 KL 惩罚」这一定性判断、以及把它称为「baby reinforcement learning」、近似 bandits（多臂老虎机）的说明均保持不变，本讲其它公式无需改动。
