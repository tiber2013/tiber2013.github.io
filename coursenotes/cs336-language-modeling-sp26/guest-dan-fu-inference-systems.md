---
title: "客座讲座 · 推理系统：把电力变成智能（Dan Fu）"
description: "这场客座讲座从推理系统视角讨论语言模型：训练完成后，真正把 GPU 算力转化为可用智能的是推理引擎。一次请求会经历调度、KV cache 复用、prefill 与 decode 等阶段；两者计算特征不同，因而常被解耦到不…"
---

# 客座讲座 · 推理系统：把电力变成智能（Dan Fu）

> **Stanford CS336 语言模型从零构建 · 客座** ｜ 讲者 Stanford Online ｜ 72 分钟 ｜ [原视频](https://www.youtube.com/watch?v=9EEm4iMAF5s)

> **v2 详写版** · 按 20 分钟分段详写，配图覆盖全片；文末附「对照参考笔记补遗（Spring 2025 版）」，记录与 2025 版讲义笔记的核对结果。

## 核心结论

本讲把视角从训练侧翻到服务与推理侧，沿「把电力变成 token、再变成智能」的主线拆开一个请求的完整生命周期：它先被调度并尝试命中 KV cache 复用前缀，再裂成计算密集的 prefill 与带宽密集的 decode 两段，又在 continuous batching 里与长短请求争抢显存，而首 token 延迟、token 间延迟与每卡吞吐都取决于输入长度、缓存命中率与多轮节奏这类真实流量形状。由于万亿参数模型单卡装不下，系统必须借助张量并行、专家并行、两阶段解耦、KV cache 分层卸载与缓存感知路由把计算映射到硬件，并靠生产可观测性捕捉 NaN、工具调用死循环、随机中文字符等规模化后才浮现的罕见故障。镜头继续下沉到 GPU kernel：逐算子写法在单 token 解码里累积大量启动与空转，Megakernel 配合 ThunderKittens 把 GPU 当分布式系统调度、让权重加载与 attention 细粒度重叠，在 H100 上把单操作带宽利用率做到 72%。讲座最后回升到模型架构，用 Parcae 循环 transformer 的谱半径稳定判据与纳入循环次数的缩放定律论证「数据越多越该多循环」，并落到全篇纲领：唯有理解推理与底层 GPU kernel，才能在从硬件、系统到算法的整条栈上实现全栈创新。

## 从训练一侧到推理一侧：把电力变成智能 [00:00 - 02:40]

演讲者 Dan Fu 开场先感谢到场的人与课程教师 Percy 的邀请，并点明这门课此前的重心：主要讨论 training（训练），也就是「如何训练语言模型、怎样把模型带到能与人对话（talk back to you）的状态」。他随即把视角翻到「另一侧」：一旦已经拥有这样一个模型，从 serving（服务 / 上线部署）与 inference（推理，即用训练好的权重对输入做前向计算并产出输出）的角度看，系统会是什么样子。他用一句贯穿全场的话概括——把 electricity（电力）变成 tokens（词元），再变成 intelligence（智能）——并说明还要谈从推理系统这个 standpoint（立场）出发能看到的 fun research problems（有趣的研究问题）。封面幻灯片把这句话写成标题 `Inference: Turning Electricity into Intelligence`，署名 Dan Fu、UCSD、Together AI，附个人主页 `https://www.danfu.org/`，左下是 `UC San Diego`、右下是 `together.ai` 标识；这些署名与机构细节在口播里只被部分带过，属于画面独有的信息。

![讲座封面：Inference: Turning Electricity into Intelligence，署名 Dan Fu、UCSD、Together AI](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0001_00-00-33.jpg)

接着他以 high level motivation（高层动机）开场，判断我们正经历一场几乎可称为 new industrial revolution（新工业革命）的能力跃迁，并坦白这些幻灯片是大约两年前做 job talk（求职报告）时做的，因此「具体例子已经相当旧了」。他列举的能力包括 human level text generation（人类级文本生成）、code generation（代码生成，口播念到 cursor、cloud code，还特意调侃 GPT 5.5「not just four」，即不只是 4）、生成与理解 images and videos（图像与视频）、理解 new modalities（新模态），以及科学、生物、健康、DNA models 等方向的应用。配套幻灯片 `The New Industrial Revolution` 把这些能力排成三栏并配图：左栏 `Human-Level Text Generation (GPT-4)`，内嵌一个 ChatGPT-4 对话框，提示词是 "Write a creative poem about yourself"；中栏 `Art Generation (Midjourney)`；右栏是蛋白质结构样式的 `Understanding New Modalities`。这几个栏目标题、产品名与示例提示词，口播并没有逐字念出，是画面给出的增量。

他把自己长期研究的驱动问题归结为两句：是什么让这些进步成为可能，以及下一代如何在其上改进；而「一个极其明显的驱动因素」就是 scale（规模）。他给出的时间线带着自嘲：2018 年（他戏称为读博之初的 "in antiquity"）最大的模型约 100 million parameters（1 亿参数），当时已觉得「oh my god, pretty crazy」；到 2019 年，GPT-2 被认为「too dangerous to release」（太危险不宜发布），而他指出这门课只要足够努力其实就能训出 GPT-2 quality 的模型；到今天，开源模型已达 trillion parameters and more（万亿参数及以上），前沿大概在 5 to 10 trillion parameters（5 到 10 万亿参数）。把这两个端点写成一行，就是

$$10^{8}\ \text{parameters}\ (2018)\ \longrightarrow\ 10^{12}\ \text{parameters and more}\ (\text{today}),\qquad \text{frontier}\ \approx\ 5\text{–}10\times10^{12}$$

伴随规模而来的是 chat、writing code、analyzing complex text，甚至「帮你写作业」等新能力。这里要留意一个易错点：口播明确说幻灯片例子偏旧，所以 GPT 5.5、Cursor 这类产品名是时代性的举例，不宜当成课程结论；「新工业革命」也是类比而非可量化论断。

把这句标语拆成可操作的直觉会更清楚：电力驱动 GPU，GPU 上跑的 inference 计算把输入 token 序列变换成输出 token，而「智能」只是这串 token 在用户侧呈现出的效果。换句话说，本讲不打算再讨论权重怎么学出来，而是讨论学出来之后每一步计算怎样被调度、被复用、被映射到硬件。这里也埋着一个容易混淆的地方：口播提到课程里已经写过一些 training side（训练侧）的 flash attention kernel，但这与后面要讲的 inference side（推理侧）kernel 不是同一件事；把两者当成一回事，就会低估推理侧口播所说的 "a whole world of complexity"。从结构上看，本章是整场的定位：先交代「另一侧」是什么，后面的规模曲线、推理引擎、workload、prefill / decode、continuous batching 都沿这条线一层层往下走。

## 规模跃迁与变革速度：参数曲线、曼哈顿马车与新石油 [02:40 - 05:10]

承接上一章「规模是驱动力」的判断，幻灯片 `ML Models Scaling in Parameters...` 用一张曲线把它量化了，而曲线上绝大多数标注口播根本没念。纵轴写作 `Model Size (in billions of parameters)`，是对数刻度，从 0.01、0.1、1、10、100 到 1000；横轴是 2018 到 2022。红色小字在左下角标 `100 million 2018`、右上标 `500 billion 2022`，底部一行粗体结论是 `Scaling in Model Size: 100M to 500B in Four Years`。曲线上逐点标了具体模型与规模：ELMo (94M)、BERT-Large (340M)、GPT-2 (1.5B)、Megatron-LM (8.3B)、T5 (11B)、Turing-NLG (17.2B)、GPT-3 (175B)、Megatron-Turing NLG (530B)。把这条跨度写成式子：

$$100\,\mathrm{M}\ (2018)\ \xrightarrow{\ \text{four years}\ }\ 500\,\mathrm{B}\ (2022),\qquad \text{i.e.}\ 10^{8}\ \to\ 5\times10^{11}$$

![参数规模曲线：纵轴对数刻度的 billions of parameters，标注 ELMo 94M 到 Megatron-Turing NLG 530B，结论 100M to 500B in Four Years](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0003_00-02-47.jpg)

右侧 `New Capabilities:` 下面还有三条 bullet，同样是画面独有：`Human-like chat`、`Writing code (including the kernels in this presentation!)`、`Analyzing complex text`。第二条尤其值得停一下——它等于在说「连这份演示里的 kernel 都是模型写的」，把上一章口播里「写代码」的能力直接扣回到本次讲座自身。这里要澄清一个和口播不一致、却容易被忽略的细节：口播给的是「今天开源万亿参数、前沿 5 到 10 万亿」，而这张较老的幻灯片停在 2022 年的 500 billion（5000 亿）参数；两者不是矛盾，而是同一条曲线在不同年份的取样，正好印证演讲者说的「例子在变旧」。用这批数据时要小心别把 2018 年的 100 million 与 2022 年的 500 billion 当成同一帧的对照，也不要把对数轴上的 100（即 100B）误读成 100 million。

紧接着他用一个日期恰好能对上的类比讲「转变比我们以为的更快」：`It's Happening Faster Than We Think...` 这张幻灯片并排放了两张黑白照片。左图说明写着 `1902: 130,000 horses working in Manhattan... (producing 22 pounds of manure a day!)`；右图是 `1912: cars outnumber horses in Manhattan`。口播把故事讲得更完整：1902 年曼哈顿有 130,000 匹 working horses（工作马，不是养着玩的，而是承担关键运输职能），每匹每天产生若干磅马粪，乘以 13 万匹就成了 "a real manure problem"；1898 年纽约甚至开过专门的 entire academic conferences（整场学术会议）讨论「拿这些马粪怎么办」，而会议结论是「nothing we can do about the horse manure, you just have to hold your nose and deal with it」（无解，只能捏着鼻子忍）。结果 10 年后的 1912 年，汽车数量已经超过马匹，延续了几个世纪的东西被汽车取代了。幻灯片补出的 `22 pounds of manure a day` 这个精确数字口播只含糊说成 "pounds of manure a day"，是画面给的增量。他把这套「1912 moment」映射到语言模型：对他本人而言大概就是去年——去年开始他大部分代码由语言模型写，团队里多数人如此，他也让所有学生照做，唯一的例外是做作业（except when they're doing their homework）。

![GPU 新石油：OpenAI 5000 亿美元 GPU 数据中心投资与 HUMAIN、NVIDIA 战略合作公告，附 History Rhymes 批注](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0005_00-05-02.jpg)

规模靠什么撑起来？他给的答案是 GPU，并抛出 `GPUs Are The New Oil`（GPU 是新石油）。配套幻灯片里是两则公告：左侧 `Announcing The Stargate Project`（日期 `January 21, 2025`），配文 `OpenAI $500B Investment in GPU Data Centers`（OpenAI 在 GPU 数据中心投入 5000 亿美元）；右侧 `HUMAIN and NVIDIA Announce Strategic Partnership to Build AI Factories of the Future in Saudi Arabia`（日期 `May 13, 2025`）配 NVIDIA 标识，下面用红字批注 `(History Rhymes...)`（历史押韵）。口播只说有 "hundreds of billions of dollars and more" 的投资、有 entire countries 与 sovereign wealth funds（主权财富基金）把它当作重要组成部分，并没有念出 500B、Stargate、HUMAIN 这些专名与日期。这里有个必须点破的易错点：本章出现了两个 `500 billion`——曲线里的 `500 billion 2022` 指的是 5000 亿个模型参数，公告里的 `$500B` 指的是 5000 亿美元投资，单位完全不同，口播又都在讲规模，稍不留神就会串。另一个局限是演讲者自己承认这是 "a older announcement"、幻灯片整体偏旧；马粪与石油都只是修辞类比，不构成因果模型。这三块内容合起来把「规模 → 资本 → 变革速度」铺完，下一章便顺势追问：这么多 GPU，究竟靠什么把它们从「沙子」变成可用的智能。

## 推理引擎与 GPU kernel：从 DAG 到全栈创新 [05:10 - 07:40]

演讲者在此给出全场的中心判断：inference is really the piece（推理才是关键那一环）。他的类比是：石油本身没用，只有当汽车里有一台 engine（引擎）把 oil 变成 useful kinetic motion（有用的动能）时才有用；同理，inference engines（推理引擎）与 GPU kernels（GPU 核函数）才是把 GPU「从沙子（from sand like Percy said）变成今天能用的东西」的部件。他把机器学习模型描述成 `DAGs of operations`（DAG，directed acyclic graph，有向无环图，即由算子节点构成、无环的计算图），是「some mathematical object that exists in the ether」（某种飘在理念世界里的数学对象）；而推理引擎和 GPU kernel 才是你真正要去 program（编程）、要去 map them down to ML operations（把它们映射成可在硬件上执行的机器学习算子）的东西。把这条链写成式子：

$$\text{model} = \text{DAG of operations}\ \xrightarrow{\ \text{inference engine} + \text{GPU kernels}\ }\ \text{tokens}\ \to\ \text{intelligence}$$

他顺带指出这门课已经做过一点这类映射（「I think you implement some flash attention」），但那属于 training side；「there's a whole world of complexity when it comes to inference」——推理侧另有一整套复杂度。口播还给出一句被单独做成一页幻灯片的 takeaway（全场唯一希望听众带走的话）：`Understanding inference and GPU kernels enables full-stack innovation in ML algorithms`，即理解推理、推理引擎与底层 GPU kernel，才能在机器学习算法上做 full-stack innovation（全栈创新，从算法一路贯通到硬件）。"If you take away nothing from today's talk" 这个措辞本身就说明它是本讲纲领。

基于这个纲领，他交代了今天的结构：先做一次 highlevel overview of the lifetime of a token（一个 token 的生命周期总览）——当你向模型发出一个请求，它经历什么、怎样穿过整个 inference service、其中要做哪些有趣的选择；然后 dive deeper into two research projects（深入两个研究项目），看如果取系统的某些 pieces（部件）出来，能问什么问题、能做些什么。数字上这里有两个明确的「2」：two organizations（他代表的两个机构）与 two research projects（两个研究项目）；而它们要服务的对象，正是上一章提到的 trillion 量级、前沿 5 to 10 trillion parameters 的模型——正因为模型这么大，「把 DAG 映射到硬件」才不是可有可无的工程细节。

接着是一段简短的机构 plug（推介）：他代表 UCSD（在那里有一个 small lab，本次部分工作出自该实验室）与 Together（承担了本次另一部分工作）。他把 Together 描述成 an AI cloud（AI 云），口播只笼统说有 GPUs、inference、fine-tuning "and all the rest"，还提到团队 heavy research background（研究背景深厚，including Percy）与 big research presence。而幻灯片 `The AI Native Cloud` 把这套云拆成了具体组件，这些标签口播几乎没念。左侧三个输入块是 `Platform services`、`Open weight & licensed models`、`Accelerated compute and storage`；云内部的流水线依次是 `Pretraining` → `Supervised Fine-tuning (SFT)` → `Reinforcement Learning (RL)` → `Inference`；`Inference` 再向右侧分出 `Agents`、`Applications`、`Gateways`，并与 `Code Sandbox` 相连，云体底部还重复印着 `models` 字样。把图上这条主干原样列出来：

```text
Platform services | Open weight & licensed models | Accelerated compute and storage
Pretraining -> Supervised Fine-tuning (SFT) -> Reinforcement Learning (RL) -> Inference
Inference -> Agents / Applications / Gateways ; Inference <-> Code Sandbox
```

![The AI Native Cloud 架构：Pretraining、SFT、RL、Inference 流水线与 Agents、Applications、Gateways、Code Sandbox 组件](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0007_00-07-16.jpg)

这一章有几处容易读偏。第一，「GPUs are the new oil」是把 GPU 比作待转化的原料，不能反过来当成「买够 GPU 就自动得到智能」——按演讲者的说法，缺的恰恰是 engine 与 kernel 这一层。第二，说模型「只是 DAG」是刻意抽象，目的是把数学对象与系统实现分开；但真实 serving 还得处理调度、内存、带宽，纯 DAG 视角会系统性低估推理侧复杂度，这也是他紧接着要展开生命周期总览的原因。第三，SFT、RL、Code Sandbox、Gateways 这些是机构云产品图上的组件名，属于背景介绍，不要误以为本段会逐一讲它们的算法。由「DAG 需要被映射」自然引出下一章：一个具体请求进来时，这套映射到底经过哪些环节。

## 一个 token 的生命周期总览：推理引擎的组件地图 [07:40 - 10:30]

他正式开始 highlevel overview of the lifetime of a token：当你向推理系统发出一个请求，它会依次经过哪些 pieces（部件）。在讲技术之前先交代了两点来源，都属于口播独有、值得记下的元信息：这些幻灯片是 a crib from a presentation that my student Austin made（借鉴自学生 Austin 在 Together 经历基础上做的报告）；以及 `all of the slides are completely AI generated from Nano Banana Pro`——整套幻灯片由 Nano Banana Pro 生成，「pretty good as long as you don't look too closely at the text」。他给了一个很实用的阅读约定：`at a high level everything is correct. If you look very closely, you'll see things that are very wrong`（宏观层面对，凑近看逐字会有明显错误）。后文他还会再举一次实例：`some of the fives turned into S's`（有些 5 被生成成了 S）。这提醒我们：图上的组件名、拓扑可信，但个别字符与数值不能逐字当权威。

按口播，请求进来后的路径大致是这样。首先 request 被 scheduled（调度）到 various different GPUs 上，而且你 might have disaggregated prefill and decode on different machines（把预填充与解码拆到不同机器）；接着系统会 run that request against a KV cache（KV cache，键值缓存），问一句「have I seen this request or versions of this request before」，看有没有已经算过的 compute 可以省下来；然后才开始 executing the core machine learning code，产生 new tokens、new operations that we need to compute。在这个执行阶段有一串可选优化：可以 split that computation across different machines、parallelize across different nodes、也可以在 a node 内部 parallelize across different GPUs，具体取决于模型大小以及你打算怎么切分。他还预判，随着 hardware 继续发展，这里「we're actually going to start to see more different choices」。走完所有选择、执行完所有代码，末端才吐出 tokens，组成完整的 end-to-end experience。他举的例子很生活化：向聊天框问 "Hey chat, what did Percy mean when he said linear regression? Because I skipped that day in my class"（Percy 说的 linear regression 是什么意思，我那天翘课了）。

真正把这条路径钉死的是 `What's in an Inference Engine` 这张横向组件地图，口播只泛泛提到「调度、KV cache、优化、并行」，而图把每个环节的专名全写出来了，是本段信息密度最高的一处增量。主线从左到右依次是 `Request In` → `Scheduling` → `KV Cache` → `Execution` → `Parallelism` → `Tokens Out`；每个节点下方还有小字：`Scheduling` 下是 `continuous batching, chunked prefill`；`KV Cache` 下是 `radix tree, prefix sharing, offload + prefatch`（图上如此拼写）/`offload + prefetch`；`Execution` 下是 `prefill, decode loop, CUDA graphs`；`Parallelism` 下是 `TP, wide EP, attention DP`。主线之外还有三条支路：从 `Execution` 向下分出 `Disaggregation`（`prefill/decode split, async RDMA transfer`），从 `Parallelism` 向上分出 `What's Next`（`fault tolerance on NVL72, 1M+ context parallelism`），左下角黄色灯泡标注 `Bug Stories`（`NaNs, KV corruption, observability`）。把这些标签原样列成一张组件清单：

```text
Request In -> Scheduling (continuous batching, chunked prefill)
           -> KV Cache (radix tree, prefix sharing, offload + prefetch)
           -> Execution (prefill, decode loop, CUDA graphs)
           -> Parallelism (TP, wide EP, attention DP) -> Tokens Out
Execution -> Disaggregation (prefill/decode split, async RDMA transfer)
Parallelism -> What's Next (fault tolerance on NVL72, 1M+ context parallelism)
Bug Stories: NaNs, KV corruption, observability
```

![推理引擎组件地图：Request In 到 Tokens Out 的主线，及 Scheduling、KV Cache、Execution、Parallelism、Disaggregation、What's Next 各节点小字](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0008_00-08-23.jpg)

其中 `1M+ context parallelism`（百万级上下文并行）与 `NVL72` 是图上独有的具体数字，口播在本段没有念到；`TP`（tensor parallelism，张量并行）、`wide EP`（wide expert parallelism，宽专家并行）、`attention DP`（attention data parallelism）、`radix tree`（基数树）、`CUDA graphs` 也都是首次以画面形式出现的术语。要克制的一点：`What's Next` 与 `Bug Stories` 是整场讲座的路线预告，本段转写并没有展开它们的内容，因此这里只登记标签，不替演讲者补讲后续项目细节；同理 `offload + prefetch`、`async RDMA transfer` 是图上的机制名，本段口播尚未解释其实现。

讲完组件，他把话题转向 `what do these different workloads look like`（这些不同的工作负载长什么样），并强调一个方法论上的易错点：serving production traffic（服务真实生产流量）时，token 的模样「certainly doesn't look like」训练时见到的 token，也「doesn't look like」你凭空在脑子里编出来的 traffic workload。也就是说，不能用训练分布或想象分布去代替真实流量分布；这正是下一章要用具体数字回答的问题。由「请求经过哪些环节」到「这些环节要服务怎样的流量」，本章完成从结构到负载的过渡，后面的 SLA 目标、prefill / decode 差异、continuous batching 都建立在这张组件地图之上。

## 生产流量的形状与 SLA：我们在优化什么 [10:30 - 13:30]

演讲者给出的基本命题是：对某一种 workload（工作负载），你会看到 input tokens 与 output tokens 的一个特定 distribution（分布），而且不同应用之间的差别相当大。他先举 coding workload：假设你是 Cursor 这样的产品，whole codebase is available to the agent（整个代码库都喂给智能体），用户就着它提问，典型形态是 long amount of input、tens of thousands of input tokens（数万输入 token）；至于输出，取决于模型怎么训，可能产生一些 thinking tokens（思考 token），也可能是一段短回答，而且「this will change based on workload, it'll change based on a model」。他接着用对比把「形状」讲具体：coding workload 会与 summarization workload（摘要）或 narrative summarization 很不一样——后者可能像把 entire books（整本书）粘进聊天窗口再来回讨论剧情；而 standard chat（普通聊天）又是另一副样子，比如随口问一句 "hey explain to me first order calculus or whatever"（给我讲讲一阶微积分），输入输出都短。这三类 workload shapes（负载形状）并不能互相代替，正是上一章「别凭空编流量」的具体化。

第二个维度是 turn-based agentic workflows（基于轮次的智能体工作流）。口播描述：写代码时你会与 coding agent 来回拉扯——「Hey, do this. No, I didn't mean that. Do that.」；coding agents 自己还会 iterate on the language models quite a bit（反复调用模型很多次），可能 invoke some tools（调用工具），比如 "search for something in my codebase"（在我的代码库里搜点东西）再把结果 feed it back to the model（回灌给模型），也可能 do an internet search 去查用户问的东西，于是 typically you have these multiple turns（多轮）对话。不同应用还有不同 times（时间节奏）：fast interactive chat-based loop、或手机上 ChatGPT 的 voice mode，响应相对快；而如果你搭了一个 agentic workflow，说「go do this for me, I'm going to leave you alone and just iterate on your own」，cadence（节奏）就完全不同；更麻烦的是 agent 卡住时喊 "hey help, I need to ask for advice"，而你没及时看到，轮次之间就又多出一段 gap。这些维度一起定义了 workload：每次进来多少 new tokens、总共要 generate 多少 tokens、session 有多长（是 sticky user 一直跟 Claude agent 来回，还是问一个问题就走、第二天才回来）、以及 how long between turns。他给了个很私人的例子：自己有一个和 ChatGPT agent 讨论「how I should set up my workout week to week」（每周怎么安排健身）的会话，大约 once every other week（每两周）才交互一次，这与前面那些高频场景是 "a very different traffic pattern"。

![Workload + SLA 图：input/output token 长度、turns per session、time between turns 三条分位曲线，以及 TTFT < 500 ms、TBT < 30 ms、target QPS per GPU 三项目标](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0010_00-10-38.jpg)

最后一类差异是 targets（目标）。交互式应用可能要求 first tokens back in less than a second（首 token 一秒内返回），好让 agent 先说一句「I'm thinking」；也可能你已经知道要 generating 500 tokens，于是要求整段回答在某个时间内回来，好让 "my user can read it fast enough"。幻灯片 `Workload + SLA — What Are We Optimizing For?` 把这些口头描述换成了可读的精确数值，而口播几乎没念这些数字，是本段最值得补的增量。左栏 `Workload Shape` 是三条分位曲线：第一条 `input / output token lengths + cache hit rate`，横轴分位点标着 `p1 32/64 5%`、`p25 258/128 30%`、`p50 512/256 60%`、`p75 1024/512 85%`、`p99 4096/1024 98%`，小字注明 `example values: ISL / OSL, Cache Hit %`（ISL 为 input sequence length 输入序列长度、OSL 为 output sequence length 输出序列长度）；第二条 `turns per session (N)`，分位值为 `p1 1`、`p25 3`、`p50 7`、`p75 15`、`p99 50`，小字 `multi-turn agent conversations`；第三条 `time between turns`，分位值 `p1 0.5s`、`p25 2s`、`p50 10s`、`p75 60s`、`p99 300s`，小字点出用途 `informs cache sizing — how long do we hold KV?`（决定 KV 缓存该保留多久）。右栏 `SLA Targets`（SLA，service level agreement，服务等级协议）是三个发光方框，连定义一并写出：

$$\mathrm{TTFT} < 500\,\mathrm{ms}\ (\text{time to first token — prefill speed}),\quad \mathrm{TBT} < 30\,\mathrm{ms}\ (\text{time between tokens — decode speed}),\quad \text{target QPS per GPU}\ (\text{throughput efficiency})$$

即 TTFT（首 token 延迟，对应 prefill 速度）、TBT（token 间延迟，对应 decode 速度）、以及 target QPS per GPU（每 GPU 每秒查询数，对应吞吐效率）。底部一行小字给出全图的因果：`SLA depends on workload — same latency target is easy or impossible depending on ISL, OSL, cache hit rate, N, and time between turns.`（同一个延迟目标，在不同 ISL、OSL、缓存命中率、轮数 N、轮次间隔下，可能轻而易举，也可能根本做不到）。这里的易错点有两层：一是别把 `p50 512/256` 之类的分位数当成平均值或固定配置，它只是某条分布上的点，口播强调流量会随 workload 与 model 变；二是别孤立地看 `TTFT < 500 ms`——能否达标取决于上述五个量，把短对话与数千 token 的冷请求混在同一组 GPU 上会互相拖累。还要记得上一章的提醒：这张图同样由 Nano Banana Pro 生成，个别字符（口播点名 fives 变成 S's）不可逐字采信。本章把「负载形状」和「优化目标」量化之后，下一章就能解释为什么单次请求内部会裂成计算特征相反的两段。

## 从请求到 token：prefill、decode 与解码循环 [13:30 - 16:00]

口播从「一个请求进来有几个 basic pieces」重新落回单请求内部，并预告要在 prefill / decode 上多停一会儿。第一步是把进来的 text 做 tokenize（分词）——「I think you're all familiar with that」，课程里已经学过，所以不展开。紧接着就进入他所说的 somewhat complex scheduling regime（相当复杂的调度环节），系统要问两类问题：`have I seen these tokens before`（这些 token 之前见过吗）、`can I just look up some of the activations in a cache`（能不能直接从缓存里把激活值查出来）。这与上一章组件地图里 `KV Cache` 节点的 `prefix sharing`、`radix tree` 是同一件事的不同说法：调度层先尝试复用已有计算，复用不了才真正算。此后才是 the two major pieces of the actual machine learning computation（真正机器学习计算的两大块）——prefill（预填充）与 decode（解码），他强调二者 "are very different"（非常不同）。

prefill 的定义带一个具体数字：假设你有 `10,000 tokens that you've never seen before`（一万个从没见过的 token），要一次算出它们的 activations 和 logits，于是 `10,000 tokens in one token out`（一万 token 进、一个 token 出）。他判定这是 a very compute-bound operation（计算受限），而且「looks pretty close to」训练时见过的东西：训练时你有一大堆 token、写一个 flash attention kernel、跑 training loop；prefill 几乎一样，唯一区别是 `you just don't run your backwards pass`（不跑反向传播）。decode 则是 `when you're then just generating one token at a time`：把这 10,000 tokens 传进去，就像说「here's my codebase. Tell me what function ABC does」（这是我的代码库，告诉我函数 ABC 干嘛），模型先处理整个 prompt，然后一次生成一个 token，`maybe three or four tokens if you have speculative decoding working`（若开了 speculative decoding 投机解码，也许一次三四个）。他给了一个关键的算力判断：每生成一个新 token 都要 run that back through the model（把整条序列重新过一遍模型），但 "if you do the math on that, there's actually not too many flops that you need to compute"——flops 很少，是 relatively light computation，却 very memory bandwidth bound（内存带宽受限），因为 `you have to load up the model every time just to generate a single token`（每生成一个 token 都得把模型权重重新装载一遍）。把两段写成对照式：

$$\text{prefill}:\ 10{,}000\ \text{tokens} \to 1\ \text{token}\ (\text{compute-bound});\qquad \text{decode}:\ 1\ \text{token/step}\ (\text{memory-bandwidth-bound})$$

一段模型 pass 结束，末端得到 `a single number that represents a token`（一个代表某 token 的数字），再被 turned into a string（转成字符串），但这里「you probably do a little bit of processing on that」：要 look for stop tokens（查停止符）、可能 run a safety check（安全检查，比如 "is my user trying to hack into the system in a bad way"，判断用户是不是在试图恶意入侵），最后才得到 the nice tokens。他还点出推理引擎的本质：它 running in this loop，跑着 scheduling、execution、token sampling 的循环，等待请求进来、再 repeating。

![From Request to Token 流程：REQUEST 经 TOKENIZE、SCHEDULE、PREFILL、DECODE LOOP、DETOKENIZER 到 STREAMING OUTPUT TOKENS](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0013_00-13-59.jpg)

`From Request to Token` 这张流程图把上述口播钉成了一条链，且贡献了一个口播没说出口的术语。节点依次是 `REQUEST` → `TOKENIZE` → `SCHEDULE` → `PREFILL` → `DECODE LOOP` → `DETOKENIZER` → `STREAMING OUTPUT TOKENS`；`PREFILL` 上方标 `compute-bound`，`DECODE LOOP` 上方标 `memory-bandwidth-bound`、下方标 `one token per step`，并有一个绕回自身的循环箭头；底部一行写着 `the engine is a loop: schedule → execute → sample → repeat`。其中 `DETOKENIZER`（反分词器）正是口播里「single number → turned into a string → 做点 processing」那一整段的画面命名，而 `compute-bound` / `memory-bandwidth-bound` / `one token per step` 则以标签形式把两段的计算特征固定下来。把这条链与循环原样列出：

```text
REQUEST -> TOKENIZE -> SCHEDULE -> PREFILL [compute-bound] -> DECODE LOOP [memory-bandwidth-bound, one token per step]
        -> DETOKENIZER -> STREAMING OUTPUT TOKENS
the engine is a loop: schedule -> execute -> sample -> repeat
```

本章最容易犯的错，是把 prefill 与 decode 当成同一种计算、用同一套 kernel 或同一批硬件去伺候：一个是 `10,000 → 1` 的算力密集、接近训练前向，一个是 `1 → 1` 的带宽密集、每步都要重载权重，瓶颈恰好相反；也别因为 decode「flops 不多」就以为它便宜——瓶颈在字节搬运而不在浮点运算。第二处局限是 speculative decoding 在口播里带条件（"if you have ... working"），是可选优化而非默认行为，三四个 token 不能当成通用常量。第三，模型吐出的那个数字并不等于用户看到的文本，中间还夹着 detokenize、stop tokens 与 safety check，把这段「后处理」漏掉就无法解释后面 continuous batching 里请求何时算「完成」。把视角与前两章接起来：上一章的 `TTFT < 500 ms` 主要由 prefill 决定、`TBT < 30 ms` 主要由 decode 决定，SLA 之所以拆成两项，根源就在这里；而下一章要问的是，当许多这样的请求同时住在系统里，调度会复杂到什么程度。

## 多请求共存：continuous batching 与容量检查 [16:00 - 17:55]

最后一章往里再走一步：不再看单个请求，而看 a system that is processing many different requests at a time（同时处理许多请求的系统），由此引出 continuous batching（连续批处理）。他先教读图方式：`time is flowing downward`（时间向下流动），越往下越靠后，于是「as you go down there are new requests that are coming in」。按口播逐步复盘：Step 1 里有一个 long request——某用户要 analyze some long document（分析一份长文档），引擎正不断往外吐 token；同时可能有另一个请求进来，开始占用资源。他明确这些资源有两类：`compute resources`（计算资源，要同时 over multiple requests 跑）与 `memory resources`（内存资源，比如 filling up a KV cache）。多个请求就这样并存；跑过一步之后，短的那个可能 finish，腾出位置，于是 new request 立刻进来；它也许再跑几步，又来一个请求，可能是 another very long request——也就是 step four 里那个橙色条。但此时 `maybe you don't have enough GPU memory`：所有 KV cache 都得放在 GPU 上，显存可能 run out，于是新请求只能 start queuing（开始排队）。等那条长请求结束，才能 start the new request，然后 "you can watch it run"。他收束道：`already you're seeing some of the complexity when you have many different requests living in a system`。

![CONTINUOUS BATCHING 时间线：Step 1 到 Step 6 中 Req A、B、C、D 的加入、完成与槽位释放，及容量与 microbatch 两项检查](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0016_00-17-21.jpg)

`CONTINUOUS BATCHING` 这张图把口播里的「长 / 短 / 新 / 排队」全部命名成了可核对的标签，信息量远大于转写。左侧是 `Time Flow` 向下箭头，六个 `Step 1` 到 `Step 6` 各配一个 `Scheduler` 芯片图标；彩色条逐行是：Step 1 蓝色 `Req A (Long, Active)` 加绿色 `Req B (Short, Active)`；Step 2 里 `Req B` 标 `done — slot freed`，同时 `new request joins mid-flight` 引出青色 `Req C (New, Active)`；Step 3 是 `Req A` 与 `Req C` 并存；Step 4 里 `Req A` 标 `done — slot freed`，又一句 `new request joins mid-flight` 引出橙色 `Req D (New, Active)`；Step 5 是 `Req C` 与 `Req D`；Step 6 里 `Req D` 标 `done — slot freed`。右侧两道问号注释是口播只暗示、没逐字念的检查：`? capacity check: enough KV blocks?`（容量检查：KV 块够不够）与 `? microbatch check: under max tokens?`（微批次检查：是否在上限 token 数之内）。底部小字给出另一条机制：`chunked prefill: long prompts split into chunks, interleaved with decode`（分块预填充：长 prompt 切成块，与 decode 交错执行）。把这些标签原样整理：

```text
Step 1: Req A (Long, Active)                + Req B (Short, Active)
Step 2: Req A | Req B done - slot freed    + Req C (New, Active)   [new request joins mid-flight]
Step 3: Req A | Req C                       [capacity check: enough KV blocks?]
Step 4: Req A done - slot freed | Req C    + Req D (New, Active)   [microbatch check: under max tokens?]
Step 5: Req C | Req D
Step 6: Req C | Req D done - slot freed
chunked prefill: long prompts split into chunks, interleaved with decode
```

数字上，这张图给了 6 个 step（Step 1–6）与 4 个具名请求（Req A、B、C、D），口播还专门点名 step four 那条橙色长请求；配合上一章的 `10,000 tokens in one token out` 与 `one token per step`，就能理解为什么一个长请求会长时间占着 KV cache。这里有几处易错点。第一，时间轴是向下的，读反了整个加入 / 完成次序会全错。第二，新请求可以 mid-flight（飞行中途）加入，不必等整批跑完——这正是 continuous batching 与静态 batching 的区别，但「能加入」不等于「无条件加入」：它要过 `capacity check`（KV blocks 是否够）与 `microbatch check`（是否 under max tokens）两道关；口播里那条很长的请求就是因为 GPU 显存放不下 KV cache 而触发 queuing，属于被拦下而非被抢占。第三，`chunked prefill` 说明 prefill 不一定要一次性跑完整个长 prompt，而是可以切块、与 decode 交错，这修正了上一章容易形成的「prefill 是一整块」印象，也回扣了组件地图里 `Scheduling` 节点下的 `chunked prefill`。第四，图上的 `done — slot freed` 只表示槽位释放，不等于整个会话结束，别把它与 Ch5 的 `turns per session (N)` 混为一谈。整段到此收束在「多个请求共存时的调度复杂度」上：从组件地图、负载形状，到单请求内部 prefill / decode 的分裂，再到一个 batch 里长、短、新请求互相挤占 KV cache，这条层层下探的线正是演讲者所说的 lifetime of a token 在真实服务里的样子。

## KV Cache 与基数树前缀共享 [17:55 - 19:05]

本段紧接上一节「一次请求的生命周期」，把镜头对准推理引擎里最省算力的一块机制：KV cache（Key-Value cache，键值缓存）。演讲者给的动机非常直白：生产系统里有海量用户都会以「hi ChatGPT」「hi Claude」这类几乎相同的前缀开场，从直觉上讲，没必要为每一个用户都重新计算一遍这些 token 的激活值（activations），再把整个模型重新跑一次。另一种典型情形是用户把一整本长书贴进对话：系统对这本书做一次 prefill（预填充）之后，用户在下一轮追加一句追问或补一段 addendum，此时书那部分的计算结果完全可以复用，而不是把整段上下文从头再算。KV cache 正是把「已经算过的 K、V 张量」存下来、供后续请求查表复用的机制；它带来的 prefix sharing（前缀共享）是后面所有分层存储、缓存感知路由优化的前提。换句话说，本段讨论的重心不是「怎么算」，而是「怎么让已经算过的东西不必再算」。

具体实现上，演讲者强调这其实是一种「很传统的数据结构」——一棵基本的树。新来一批 token 时，系统沿树 traverse（遍历），用 match hashes（哈希匹配）判断哪些 token 见过、哪些是新的，再直接 lookup 出这些激活值在 GPU 上算出来会是什么样，从而跳过重复计算。对应幻灯片标题为「KV Cache — Radix Tree Prefix Sharing」，标题下写着 `traverse → match hashes → reuse`：绿色的 `system prompt (shared)` 由 `s_p1`、`s_p2`、`s_p3` 三个块组成，是被所有会话共享的公共前缀；其下分出三条分支，青色 `session 1` 的 `c_b1`～`c_b4`、橙色 `session 2` 的 `o_b1`～`o_b3`、紫色 `session 3` 的 `p_b1`～`p_b5`。图中虚线标出 `cache hit`（缓存命中）指向 session 1 的分支，底部粉色 `m_n1`、`m_n2` 标为 `new tokens (cache miss)`（新 token、缓存未命中），并有一条虚线箭头示意未命中块后续被取回。右下角图例给出了转写里完全没念到的关键参数：一个方块 `□ = 32 tokens`，块张量形状 `shape [layers, 32, hidden_dim]`，以及哈希定义

$$ \text{block\_hash} = f(\text{content}, \text{parent}) $$

也就是说，一个块的身份不只由内容决定，还由它的父节点决定。

![KV Cache 基数树前缀共享：共享 system prompt 与三个会话分支](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0017_00-18-28.jpg)

这条哈希定义值得展开：若只按 `content` 做键，两个不同会话里内容相同但父前缀不同的块会被误判为可复用，进而把错误的 K、V 接进 attention，输出就会串味；把 `parent` 纳入哈希，才保证「同一前缀路径下的同一内容」才算命中。从计算量看，缓存命中意味着这 32 个 token 不再需要走完整的 prefill 前向，只需取出形状为 `[layers, 32, hidden_dim]` 的缓存张量；未命中的 `new tokens` 才真正触发 GPU 计算。这里的粒度是「块」而非「单 token」：以 32 token 为一个块，正好和前面 continuous batching（连续批处理）里按 KV block 管理显存的做法对齐。与前后内容的关系是，本节解决的还是「算一次就别再算」的复用问题；下一节要回答的是「模型大到一张卡装不下、该怎么切」，再往后则会把 KV cache 从 GPU 显存一路放到 CPU DRAM 和 NVMe 磁盘上，本节正是那条线索的起点。局限也很清楚：前缀共享只有在前缀逐块完全一致时才成立，用户改写一个词、换一种 system prompt，命中就会从分叉处断掉；而缓存本身要占显存，能缓存多少会话直接受 GPU 内存容量约束，这正是后面必须做分层 offload 的根因。另一个易错点是把哈希匹配当成字符串相等比较——真正要比较的是「块内容 + 父块哈希」这条链，否则共享 system prompt 下的分支会被错误合并。

## 万亿参数模型的切分：张量并行与专家并行 [19:05 - 19:58]

承接上一节的 KV cache，演讲者把问题从「复用计算」转到「模型根本放不下」。他给了一个具体设定：假设一个 1 万亿参数（trillion parameter）的模型，跑在 280 GB 显存的 GPU 上——即便按 FP16 每参数 2 字节粗估，仅权重就要约 2 TB，显然无法把整个模型塞进单张 GPU。于是必须把模型切开分到多卡上。演讲者点明，这一步怎么切，会直接决定后面的瓶颈在哪里、需要多少张 GPU、以及能同时服务多少个会话（sessions）；也就是说，并行策略不是纯粹的实现细节，而是决定整个 serving 系统容量与成本架构的选择。这一点和上一节的 KV cache 形成对照：前者省的是「重复计算」，后者解决的是「单次计算根本装不下」，两者共同构成推理系统的两块地基。

第一种是 tensor parallelism（张量并行）：把「每一个 tensor」切成四份、分到 4 张 GPU 上。幻灯片左半标题 `TENSOR PARALLELISM`，画了一个 `WEIGHT MATRIX` 被竖着切成 4 个条带，分别落在 `GPU 0`～`GPU 3` 上，下方箭头标 `allreduce activations`（对激活做全归约），底部注明 `every GPU has a shard of all layers`（每张 GPU 都持有所有层的一个分片）。把权重按列切开 $W = [W_1 \| W_2 \| W_3 \| W_4]$，各卡算出的 $XW_i$ 是互不重叠的列块，应沿列拼接而非求和、此步不通信；只在与随后按行切分配对后的块末做一次 all_reduce，故列切分处可形式化为

$$ XW = [XW_1 \,|\, XW_2 \,|\, XW_3 \,|\, XW_4] $$

第二种是 expert parallelism（专家并行），针对当下大量 state-of-the-art 的 mixture of experts（MoE，混合专家）模型：这类模型有一组「专家」，会根据不同 token 被选择性激活，于是可以把不同专家放到不同 GPU 上。右半标题 `EXPERT PARALLELISM`，中心是 `LEARNED ROUTER`（可学习路由器），标注 `route tokens to experts across GPUs`；四张卡各自持有若干完整专家——`GPU 0` 上是 `E0`、`E1`、`E1`，`GPU 1` 是 `E2`、`E3`、`E3`，`GPU 2` 是 `E4`、`E4`、`E5`，`GPU 3` 是 `E6`、`E6`、`E7`，底部一句 `each GPU owns full experts`（每张 GPU 拥有完整专家，而非专家的一个切片）。MoE 的输出可写成

$$ y = \sum_{i \in \text{Top-}k} g_i(x)\, E_i(x) $$

其中 $g_i$ 是路由门控、$E_i$ 是被选中的专家。

![张量并行与专家并行：权重分片与学习式路由分发](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0018_00-19-35.jpg)

数字上，本节给出的是 1 万亿参数、280 GB GPU、四路切分这三组量；它们共同说明「单卡装不下」不是假设而是常态。两种切法的差异在通信模式：张量并行每一层都要做一次 allreduce，通信频繁且和层数成正比；专家并行则靠 all-to-all 把 token 发到持有对应专家的卡上，通信量取决于路由分布，若某些专家被过度激活就会形成负载不均（straggler）。易错点在于把「切分」当成免费午餐：切得越细，GPU 数量与跨卡同步开销越大，单请求延迟可能不降反升；而且切分方式一旦选定，prefill 与 decode 各自能用的并行策略也被框定，这正好引出下一节——把两个阶段拆到不同机器、各自选最合适的并行度。另一个局限是，expert parallelism 只在 MoE 模型上成立，稠密模型仍需回到张量并行或其他切法；演讲者这里只是铺开选择空间，并没有给出「哪种一定更好」的结论。还要留意「每张 GPU 拥有完整专家」与「每张 GPU 持有每层一个分片」是两种正交的组织方式，混淆二者会导致对通信量的判断整个偏离。

## Prefill / Decode 解耦与异构硬件下注 [19:58 - 21:50]

上一节末尾已经埋下伏笔：切分方式决定瓶颈。本节把这个判断推到极致，指出生产系统里一个「很大的事情」是把 prefill 和 decode 放到不同的 GPU、不同的机器组上，原因是两者的 compute bottleneck（计算瓶颈）与 compute characteristics（计算特征）完全不同。prefill 很像训练时的前向：一次性把整段 prompt 跑完，flop-heavy（计算密集），能把 GPU 算力吃满；decode 则相反，是 memory-bandwidth heavy（内存带宽密集）——计算量不大，但每生成一个 token 仍要把全部模型权重重新加载一遍。两者耗时也不同：prefill 通常远比单个 decode step 长，但 decode step 的数量多得多，因为一个 prompt 只跑一次 prefill，而每生成一个 token 就要跑一次 decode。这个「一次对多次」的不对称，是后面所有分离式架构的物理依据。

用 roofline 的语言说，prefill 的时间由算力封顶

$$ T_{\text{prefill}} \gtrsim \frac{\text{FLOPs}(n, P)}{\text{peak FLOPS}} $$

而 decode 在 batch 极小时几乎不做大规模矩阵乘，时间由「把权重从显存搬进来」的字节数封顶；若权重为 $P$ 个参数、FP16 每参数 2 字节，则每 token 至少读 $2P$ 字节：

$$ T_{\text{decode}} \gtrsim \frac{2P}{\text{BW}_{\text{HBM}}} $$

这正是演讲者后来那句「把大规模并行系统变成了 basically a glorified memory loader（一个高级点的显存加载器）」的量化版本。于是「一个很基础的、几乎大家都开始采用的优化」就是：pre-fill 跑一组 worker，decode 跑另一组 worker，让两种计算各自专用化到硬件栈的不同部分，避免让带宽密集的 decode 去和算力密集的 prefill 抢同一批 SM。

![Prefill 与 Decode 解耦：异步环、RDMA 拉取与 KV 就绪轮询](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0019_00-20-43.jpg)

幻灯片「Prefill / Decode Disaggregation」把这套架构画得很细。橙色 `Prefill Worker` 里标注 `TP4 — compute-bound, maximize FLOPS`，画了 4 张 GPU 绕成 `prefill loop — async`（异步预填充环），底部是 `KV ready — holding buffer`（KV 就绪、暂存缓冲区）。中间 `router node` 负责 `route request + source location`（路由请求并指明来源位置）。蓝色 `Decode Worker` 里是 `initiate RDMA fetch`（发起 RDMA 拉取）、`decode loop — async`（异步解码环）、一片 3×4 的 GPU 阵列，以及 `poll: KV received?`（轮询 KV 是否收到）：`no → continue loop`，`yes → begin decoding`。环内还标了两种 decode 侧并行：`wide EP — experts shared across many GPUs`（宽专家并行，专家摊到多卡）与 `attention DP — attention shared along batch dim`（注意力按 batch 维切分）。底部一句总结点题：`disaggregation enables different parallelism strategies — prefill optimizes for compute, decode optimizes for throughput.`（解耦让两阶段可以用不同并行策略，prefill 优化算力、decode 优化吞吐。）这些标注大多没有在口述里逐字出现，却是理解解耦收益的关键：分离不只是「换台机器」，而是让 TP4、wide EP、attention DP 这些策略可以各管各的。

这种拆分还打开了一个此前不存在的创新空间：既然 decode 的负载特征和 prefill 如此不同，就可以给 decode 用完全不同的芯片。演讲者举的例子是 Nvidia 收购 Groq——「GPU 之王」买下一类推理芯片，理由正是 decode workload 与 prefill 差异太大；他提到下一代硬件里 Nvidia 计划用自己的 GPU 做 prefill、用 Groq 的 LPU 做 decode。类似的还有 Cerebras（OpenAI 与之有算力合作，是另一种更擅长 decode 的芯片），以及 SambaNova 等公司在这条链路上不同位置的下注。与前后关系上，本节把上一节的「并行策略」升级成「按阶段配硬件」，而下一节会展示当这种系统每天服务数万亿 token 时会出现哪些在小规模下根本看不到的 bug。易错点也很具体：解耦不是免费的，KV cache 必须跨机搬运，图中 `RDMA fetch` 与 `poll: KV received?` 说明 decode worker 在 KV 未就绪时只能空转轮询；若网络或 `holding buffer` 出问题，省下来的算力会被等待吃掉。另外「prefill 计算密集、decode 带宽密集」是典型情形，一旦一次产出多个 token，这个判断会被削弱，不能绝对化。

## 规模化生产中的罕见 bug：NaN、工具调用死循环与随机中文 [21:50 - 25:30]

从系统架构转到工程现实。演讲者说，当你真正把推理引擎大规模部署、每天服务数万亿（trillions）token 甚至更多时，会开始遇到「相当 nasty 的 bug」；他回忆其中大多数发生在「去年年末」的一些开源推理引擎上。核心判断是：大规模系统的特征之一，就是小规模跑得好好的东西，在规模化后几乎必然开始崩，而且崩在极罕见事件上——他给的频率是 `0.001%` 甚至更低。这个数字很关键：它意味着靠单元测试、靠离线回归几乎抓不到，只有生产环境的可观测性（observability）才能发现。这与前面几节形成鲜明对照：前面讲的是怎么把系统做大、做快，本节说明「做大」本身会催生全新的失效模式。

第一个案例是 NaN。某个 kernel（核函数）有极细微的错误，触发条件非常罕见，导致计算进行到一半时部分 `logits` 变成 `NaN`。一旦 logits 里混入 NaN，softmax 就会把整行污染：

$$ \text{softmax}(\mathbf z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}, \quad \exists\, z_k = \text{NaN} \;\Rightarrow\; \forall i,\ \text{softmax}(\mathbf z)_i = \text{NaN} $$

采样随之退化为无意义的平局，模型开始反复吐同一个 token——演讲者说有的模型输出变成「hi hi hi hi hi」，有的变成一长串感叹号，然后卡进循环。幻灯片把这一格标成 `NaN bug — silent corruption from early kernel launch`（早期 kernel 启动的静默损坏），左图纵轴是 `Valid Output %`（有效输出百分比，刻度 0/50/100），曲线本应稳在 100，却出现塌到 0 的深谷。注意「静默」二字：错误不抛异常、不崩溃，只是让输出悄悄坏掉。

![生产调试三张监控图：有效输出塌陷、补全长度尖峰与随机中文字符](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0021_00-22-57.jpg)

第二个案例与 tool calls（工具调用）有关。工具调用是模型说「我想做一次联网搜索」之类的意图，随后由外围 harness 用「老式代码」去真正执行搜索并返回结果。某次有人改动了工具调用的处理逻辑后，某个引擎里的 tool call 不再被正确结算：正常情况下模型说「去搜索、我说完了、交还给用户」就结束，但这时它不返回，于是不断重复「去搜索」「去搜索」「怎么没有搜索结果」，陷入 doom loop（末日循环），completion length（补全长度）暴涨到「数万 token」（tens of thousands of tokens）。幻灯片中图标注 `tool call merge bug — completion length shift`，横轴是 `09:00`、`10:00`、`11:00`、`12:00` 的时间轴，纵轴 `Completion Length`，在接近 12:00 处有一根陡峭的红色尖峰。这里的问题出在引擎与 harness 的交接协议，而不是模型本身的生成能力。

第三个案例最有意思：一批推理服务商同时中招，一度被归咎于 quantization（量化）问题，甚至有人猜测「模型是不是拿中文数据微调过，我问英文它却用中文回」。真实原因却是一个 `off-by-one`（差一错误）：kernel 偶尔会多读进一块 GPU 上未初始化的显存，把它一路送过 attention，最后解码出一个随机中文字符；模型看到自己「突然说了中文」，便自我合理化为「用户一定在用中文提问」，于是整段回答拐进中文。可用几行伪代码概括这类越界读：

```c
// 差一错误：循环上界多读了一个未初始化块
for (int i = 0; i <= n_blocks; ++i) {   // 应为 i < n_blocks
    kv[i] = global_kv[i];                // i == n_blocks 时越界
}
```

幻灯片右图标注 `KV length bug — random Chinese characters`，横轴 `Token Value Distribution`；注意口述把根因归为 kernel 的 off-by-one，而图上把它标成 `KV length bug`，两处措辞不同，但都指向「读入多余／未初始化内存」这一类系统级问题，而非模型能力问题。底部一行 `production observability catches what unit tests miss`（生产可观测性能抓到单元测试漏掉的东西）正是本节的题眼。

与前后的关系：前面几节讲的是怎么把系统做大、做快，本节说明「做大」本身会引入新的失效模式；后面要讲的 KV cache 分层存储、预取、容错，某种程度上都是被这类规模问题逼出来的。易错点有两层：一是症状极具误导性——重复 token 像采样问题、说中文像训练数据问题、长度暴涨像 prompt 问题，工程师很容易往模型侧查，而根因在 kernel 与内存；二是演讲者特别补充，模型「确实被训练成用中文思考」也是可能的，所以不能看到中文输出就一律判为 bug，必须靠监控曲线定位。局限在于，这些案例是轶事式的经验分享，讲者并未给出复现步骤或修复补丁，笔记只能记录现象与归因，不宜外推成通用规律。

## KV Cache 分层存储、预取与操作系统式调度 [25:30 - 29:55]

承接上一节的规模压力，本节回到 KV cache，但视角从「复用」变成「容量」。演讲者说，运行大型生产系统的一大要务是让 KV cache 尽可能大：最好能缓存许多不同用户的请求，或同一用户跨多个会话的历史，从而同时跑尽可能多的会话。做法很自然——先把 KV cache 存在 GPU 显存里，但显存很快就不够；于是往下放到 CPU DRAM，再往下放到磁盘（SSD / NVMe），开始关心 SSD 空间。他顺带解释了一个产业现象：Jensen 在近几代 keynote 里越来越执着于 CPU 性能，原因之一就是上一代 CPU 太慢，开始拖累一批重要负载；「如果你那台 5 亿美元（`$500,000,000`）的机器，被你配在它上面那颗千元级 CPU 卡住，那可不太妙」，而瓶颈往往正来自把 KV cache 放在 CPU 内存上、又必须把它快速读回来。类似地，外界传闻 OpenAI 大量扫货 SSD 与 DRAM，部分动机也是想尽量多存 KV cache。于是「推理引擎」这件事，实际上被拉成了一条从 GPU 到 CPU 再到磁盘的存储栈问题。

![KV Cache 的 Offload、Prefetch 与 Share：HBM/钉住内存/NVMe 三级层次](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0024_00-26-19.jpg)

幻灯片「KV Cache — Offload, Prefetch, Share」把层次画成一个 `Memory hierss — single worker`（单 worker 内存层次，幻灯片原拼写）的三级栈：青色 `GPU HBM — hot blocks`（热块）、黄色 `pinned CPU memory — warm blocks`（钉住的 CPU 内存、温块）、红色 `NVMe disk — cold blocks`（冷块）。两级之间各有向下的 `evict (LRU)`（按最近最少使用驱逐）与向上的 `prefetch` 箭头。中间是 `Scheduler Queue (waiting requests)`（等待请求队列），一条虚线标注 `prefetch ahead: load KV blocks before request is scheduled`（提前预取：在请求被调度之前就把 KV 块载入）。右侧通过双向箭头连到 `Prefill Worker A` / `Prefill Worker B`，标 `cache sharing — RDMA between prefill workers`、`avoid redundant prefill for popular prefixes`（为热门前缀避免重复 prefill）。底部目标写得很直白：`goal: KV cache is already in GPU memory by the time the request runs`。把命中率记为 $h$，有效延迟可粗略写成

$$ T_{\text{eff}} = h\, t_{\text{HBM}} + (1-h)\big(t_{\text{fetch}} + t_{\text{prefill}}\big) $$

系统的全部努力，就是把 $h$ 推高、把 $t_{\text{fetch}}$ 藏到请求真正开跑之前。

现场有同学提问：offloading 是不是只针对某一类负载、快路径上不该做？另一个补充是「命中 SSD 的时间非常长」（`Time to hit a SSD is very long`）。演讲者借这个提问把话题拉回操作系统：「除了右边那堆 GPU，这张图活脱脱就是你在 70、80 年代 OS 课上见过的图」——当年打开太多应用会把 CPU 内存耗尽、只能把应用换出到磁盘，和现在把冷 KV 块换出到 NVMe 是「完全相同的负载」。他顺带自嘲图左侧的驱逐策略是「nano banana 幻觉出来的」`least recently used`（LRU，最近最少使用），但随即肯定 LRU 是个相当不错的启发式，还提到大概有 OS 论文证明 `LRU is within 2x of` 最优（即 LRU 与最优置换的差距在 2 倍以内）。真正最优的是「能预测未来」：比如用户在聊天应用里翻出一个月前的旧对话，这是个很强的信号，说明马上要就它提问，系统就该提前把对应 KV 块 load 到 GPU 上。预测不了就用各种 heuristics。演讲者还给了个很实在的边界：「我从没遇到过谁想把更少流量压到 GPU 上」，所以在开头提到的那些 SLA（服务等级协议）约束下，目标就是把尽可能多的流量喂给 GPU。

这套机制的骨架可以用几行调度伪代码表达：

```python
def on_request(req):
    blocks = radix_lookup(req.prefix)          # 查前缀树
    for b in blocks:
        if b.location != "GPU_HBM":
            prefetch(b, dst="GPU_HBM")         # 提前取回
    wait_until_ready(blocks)                   # 对齐 goal: 开跑前已就绪

def on_evict():
    victim = lru_pick(cold_blocks)             # LRU 近似最优，2x 界内
    move(victim, "GPU_HBM", "pinned_CPU")
```

易错点集中在两处。其一，层次越多越慢：Q&A 里明确说 `Time to hit a SSD is very long`，所以 SSD/NVMe 只适合放真正的冷块，绝不能落在快路径上；把温块误判成热块、或预取太晚，请求就得在现场等 `fetch, load them up`，省下的重算时间反被 I/O 吃掉。其二，预取是拿带宽赌未来，猜错就是白搬；LRU 虽然便宜且理论上有 2 倍界，但它对「一个月前的旧对话突然被翻出来」这类访问模式无能为力，必须靠上层语义信号补。与后文的关系是：本节把 KV cache 变成了一种需要调度、驱逐、预取、跨机共享的「存储资源」，下一节的 NVL72 与容错，正是当这种资源被摊到 72 张 GPU 上之后的极端版本。

## NVL72 宽专家并行、故障容错与百万上下文 [29:55 - 31:30]

上一节讲的是单 worker 内部的三级内存层次；本节把尺度猛地拉大。演讲者提到上一代 Blackwell GPU 里 Nvidia 推出的 `NVL72` Grace Blackwell 芯片：72 张 GPU 用非常高速的互连（fast interconnect）连成一个整体。于是自然冒出一串问题：怎么把一个万亿参数模型切到全部 72 张 GPU 上？这样做到底有没有意义、买到了什么？以及最要紧的——怎么做 fault tolerance（故障容错）？他给出的判断是「这些东西会因为各种理由频繁挂掉」，并举了一个很具体的物理原因：连接器是塑料而不是金属做的，插拔时稍微用力过猛，线缆就会把接口别弯，于是得到「flaky NVLinks」（时好时坏的 NVLink）。硬件可靠性在这里不再是运维细节，而是架构约束。

![NVL72 上宽专家并行的前向传播与节点故障容错流程](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0027_00-30-04.jpg)

由此引出的核心场景是：假设模型被切到 64 张 GPU 上、正对着数百万用户、数万亿 token 提供生产流量，某一张 GPU 挂了怎么办？幻灯片「Wide EP — Fault Tolerance on NVL72」上半是 `Wide EP Forward Pass`：一条横轴上排着 9 组专家带，依次标为 `E0–E3`、`E4–E7`、`E8–E11`、`E12–E15`、`E16–E19`、`E20–E23`、`E24–E27`、`E28–E31`、`E32–E35`（即 9 组 × 4 = 36 个专家，编号覆盖 E0 到 E35），下方两个 `learned router` 节点配 `all-to-all: tokens routed to expert-owning GPUs`（把 token 路由到持有专家的 GPU）与 `all-to-all: results gathered back`（把结果收回），中间一句 `wide EP = high batch size + high arithmetic intensity on decode`（宽专家并行 = 大 batch + decode 上的高算术强度）。下半 `Node Failure` 画了一个被红叉划掉的专家组，标注 `node dies — experts lost, in-flight KV cache lost`（节点死亡：专家丢失、在途 KV cache 丢失），并列出三条应对：`cancel affected requests, re-prefill from cache or scratch`（取消受影响请求，从缓存或从头重新 prefill）、`redistribute orphaned experts to remaining GPUs + hot spare`（把孤儿专家连同热备重新分配到剩余 GPU，图中绿色补回 `E0–E3`）、`drain in-flight requests from failed node`（排空故障节点上的在途请求）。底部结论：`at 32+ GPUs per worker, failures are routine — need automated detection, expert migration, and request replay`（每个 worker 32 张以上 GPU 时，故障是常态，需要自动检测、专家迁移与请求重放）。

本节的另一条线是超长上下文：模型开始出现「一百万以上 context」，怎么真正处理它？是否把上下文切到多张 GPU 上？对应另一张幻灯片「1M+ Context — Context Parallelism」给出的答案是 context parallelism（上下文并行）：顶部一条彩色序列条标注 `1,000,000+ tokens`（一百万以上 token），被切成 6 段分别放到 `GPU 0`～`GPU 5`，每张卡下都挂着自己的 `local KV cache`（本地 KV 缓存），相邻段之间用弧形箭头标 `cross-partition attention`（跨分区注意力）。若把长度 $L=10^6$ 均分到 $N=6$ 张卡，每卡约承担

$$ \frac{L}{N} = \frac{10^6}{6} \approx 1.67 \times 10^5 \text{ tokens} $$

底部两行注脚点明设计取向：`sequence-parallel split avoids pipeline bubble overhead`（按序列切分可避免流水线气泡开销）与 `useful capability gated purely by engineering`（这种有用能力纯粹被工程实现所卡住，而非算法）。

这一节把前面所有数字推到极限：72 张互连 GPU、切到 64 张 GPU 的模型、每个 worker 32+ 张卡、百万级 token、数百万用户、数万亿 token/天。与前面的关系是，宽专家并行正是上一节 prefill/decode 解耦图里 `wide EP — experts shared across many GPUs` 的完整展开，而「在途 KV cache 丢失」也说明 KV cache 已经不只是可驱逐的缓存，而是有状态、需要容错的分布式数据。易错点有三：其一，单卡故障会同时丢失「专家」和「正在传输的 KV cache」，只做专家迁移不够，还得配合请求取消与重放；其二，上下文按序列切分后，attention 并不能各算各的，必须处理 `cross-partition attention` 的跨卡通信，否则因果依赖被切断；其三，塑料连接器这类硬件细节看似琐碎，却决定了在 32+ 卡规模上「故障是常态」，系统设计必须默认会挂，而不是假设可靠。局限是演讲者在这里只抛出问题清单（怎么切、买到了什么、怎么容错），并未给出定量收益或完整方案。

## 缓存感知的 prefill-decode 解耦（CPD） [31:30 - 34:05]

前面铺了整段「系统视角」，本节收敛到一个具体、可落地的研究例子。演讲者说要「非常简短地」highlight 一个从 systems level 看全流程才能想到的优化：他们几个月前放出的一项工作叫 cache aware prefill decode disaggregation（CPD，缓存感知的预填充-解码解耦）。它的定位是一个「非常简单的优化」，在路由层（routing layer）里大概只有两行代码，却能带来相当可观的差别。基本观察是：进来的请求绝大多数是会话中的「逐轮」（turn-by-turn）请求，用户已经在一段对话里了；假设平均一段对话持续 10 轮就结束，那么大约 10% 的请求是全新、很「冷」的请求：

$$ p_{\text{cold}} = \frac{1}{10} = 10\% $$

这个 10% 是整段推理的支点：它把「流量」分成了计算形态截然不同的两类，才让「分流」这件事有收益。

![缓存感知的预填充-解码解耦架构：冷/温请求分流与分布式 KV 缓存](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0029_00-32-38.jpg)

冷请求长什么样？它带着数千 token 的新内容（比如有人贴进一本书说「跟我聊聊这本书」），和已经聊到一半、只问「为什么 1+1 等于 2」「哦我不太懂」这种短问答，在计算形态上完全不同：前者 prefill 极贵，后者很轻。演讲者强调，你不希望「很长的请求」和「很短的问答」在同一时间、同一组 GPU 上混跑——长 prefill 会把短请求的延迟拖坏。做法就是一个极简 router：低 cache hit rate（缓存命中率）的冷请求发到一组 GPU，让它们成批处理同类重活；其余 warm（高命中）请求发到另一组 prefill 节点。两行逻辑大致是：

```python
if cache_hit_rate(req) < threshold:   # 冷请求：命中率低、prefill 重
    route(req, pre_prefill_nodes)
else:                                 # 温请求：命中率高、增量小
    route(req, warm_prefill_nodes)
```

架构幻灯片「Cache-aware prefill-decode disaggregation (CPD)」把这套路由画全了：顶层是 `Cache-aware router`；`Cold requests (low cache hit)` 落到 `Pre-prefill nodes`，`Warm requests (high cache hit)` 落到 `Prefill nodes`，`Decode scheduling` 落到 `Decode nodes`；中间 `Async read & write`（异步读写）连到 `Distributed KV cache`（分布式 KV 缓存），其下虚线框里是 `Store 1`、`Store 2`、`Store 3`、`Store 4` 四个存储节点，右侧注明 `High-speed RDMA`。这些标签是转写里没逐字念到、但图上明确给出的信息：冷请求有独立的 pre-prefill 阶段、KV 缓存是分布式多 store、跨节点靠高速 RDMA、读写是异步的。

结果用另一张图「CPD: Up to 40% faster long-context LLM serving」给出：纵轴 `Queries per Second (QPS)`，刻度 `0`、`0.375`、`0.75`、`1.125`、`1.5`；第一组 `2P1D vs. CPD 1D`，灰色 `Baseline` 约 `0.75`、橙红 `CPD` 约 `1.125`，标注 `40% faster`；第二组 `2P2D vs. CPD 2D`，Baseline 约 `1.125`、CPD 约 `1.5`，标注 `35% faster`；图下注明 `(2P = 2 prefill nodes, 1D/2D = decode nodes)`，两组柱子上都带误差棒。所以「up to 40% faster」是最好配置下的数字，2D 配置下是 35%，不能当成恒定收益。演讲者对整体研究阶段的判断很克制：这些技术「还非常早期」，10 到 20 年后回头看，人们大概会问「当年这帮人为什么还在讨论这个、不是显而易见的吗」；之所以现在才被提出，是因为我们才开始在生产环境里以新方式跑这些负载、看到这些新的流量形态。与前后关系上，CPD 是前面 KV cache 前缀共享、prefill/decode 解耦、分层缓存三条线索的一次合流；本节末尾演讲者用「that was kind of a overview of the inference」收束第一部分，随即转向两个研究项目，下一章就进入 decode 侧的 Megakernel。易错点在于命中率阈值：估计错了会把冷请求错分到温节点，反而让长 prefill 污染短请求；而 10 轮、10% 只是他用来推导的假设，真实分布会随产品形态变化。

## 解码的根本困境与 Kernel Boundaries [34:05 - 35:50]

本段以一句过渡收束前半部分：前面算是「一个推理 kernel 的生命周期」的概览，接下来要讲两个「很受实际 serving 实践中所见现象启发」的研究项目。第一个是关于 language model decode、以及怎么用 mega kernels（巨型核函数）让它快很多，是 Stanford 与 Together 的合作；第二个项目本章尚未展开。这里要特别注意：本节只讲到问题的提出，Megakernel 的具体实现与实测数字在本段时间范围内并未给出，因此不应把后文内容提前搬进来。从结构上看，前面几节都在系统层（调度、缓存、解耦、路由）做文章，本节第一次把镜头下沉到 GPU 上最小的一层——kernel。

演讲者把 decode 定义为「prompt 已处理完、现在一次只生成一个 token」的过程，其根本挑战是：为了生成一个 token，你必须把整个模型从头跑一遍。于是 GPU 在 prefill 或训练时那种大规模并行度完全用不上——「你把这套大规模并行系统，变成了 basically a glorified memory loader」。若把一个 block 拆成 $K$ 个算子、每个算子一次 kernel 启动开销 $t_{\text{launch}}$、本身耗时 $t_i$，传统写法下这一层的总时间大约是

$$ T_{\text{block}} = \sum_{i=1}^{K}\big(t_{\text{launch}} + t_i\big) $$

而 mega kernel 的方向是把多个算子 fuse 进一次启动，使

$$ T_{\text{block}}^{\text{mega}} \approx \sum_{i=1}^{K} t_i + t_{\text{launch}} $$

在单 token、极小 batch 的 decode 里，$t_i$ 本身很小、$t_{\text{launch}}$ 与 kernel 之间的空隙占比反而突出，这正是「融合」值得做的直觉来源。

![Llama-1B Block 的 Kernel Boundaries：逐算子红框与残差跳连](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0031_00-35-11.jpg)

让问题更棘手的是我们写 kernel 的习惯：一次只写、只跑一个 operation（算子）。原因很好理解——kernel 本身就难写（演讲者调侃大家写 flash attention 都「玩得很开心」），所以通常的做法是把语言模型里的算子逐个列出，为每一个单独写一个 kernel：写你的 norm kernel、matmul kernel、attention kernel。这样编程简单得多。幻灯片「Kernel Boundaries」用红框把这种「一个算子一个边界」画得一目了然，副标题是 `Example Kernel Boundaries for a Llama-1B Block`（一个 Llama-1B block 的示例 kernel 边界）：从 `Input` 起，依次是 `RMS Norm` → `QKV Matmul` → `RoPE` → `Attention` → `O Proj` → `+`（残差加）→ `RMS Norm` → `Up Proj` / `Gate Proj` → `SiLU` → `Down Proj` → `+`，十余个算子各被红色框标成独立边界，另有两条跨框的残差跳连弧线，页脚还印着页码 `35`。换句话说，一个 1B 量级的 block 就要触发十几次 kernel 启动与收尾。转写在 35:45 处断在「but um you end up seeing it ends up」，即刚要开始讲这种写法的代价，本段到此为止。

传统逐个算子启动的形态可用伪代码对照：

```python
for op in ["rms_norm", "qkv_matmul", "rope", "attention",
           "o_proj", "residual_add", "rms_norm",
           "up_proj", "gate_proj", "silu", "down_proj", "residual_add"]:
    launch_kernel(op)      # 每次 launch/teardown 都要付费
    sync_stream()          # 算子间留下空隙
```

与前后内容的关系很清晰：前面几节一直在讲调度、KV cache、解耦这些「系统层」优化，本节把镜头下沉到 kernel 边界，并指出逐算子写法在 decode 场景下会累积出可观的 downtime；前面 prefill/decode 解耦时说过 decode 受内存带宽支配、像「glorified memory loader」，这里正是从 kernel 粒度解释为什么会这样。易错点与局限有两点必须说清：其一，逐算子写法并非错误，它是「kernel 难写」这一现实下的合理工程折中，牺牲的是 decode 的效率、换来的是可编程性，不能简单否定；其二，本段只提出了 mega kernels 这个研究方向与「融合」的动机，并没有给出融合方案、加速比或实现库，任何具体收益数字都属于本段时间范围之外，笔记不应替演讲者补写。

## GPU 解码的空闲难题：stragglers、kernel launch 与 tail effects [35:50 - 37:30]

本段一开场就点明前面那种「每个操作各写一个 kernel」的写法会 introducing a lot of downtime into your system（给系统引入大量空闲时间）。为了把这件事讲清楚，演讲者展示了一张取自某个具体 attention inference kernel 的时间线图：横轴是 `Time`，时间从左到右流动；纵轴是 GPU 上所有不同的 streaming multiprocessor（SM，流式多处理器）。转写里给出的具体数字是：H100 上有 132 个这样的 SM，B200 上「我想是 148 个」左右。图中每一根横条代表一个处理器在某个时间段里真正干活，即有 useful work（有用工作）；条与条之间的空白则只是 waiting（等待），也就是在等别的操作结束，然后自己才能接着做点什么。理解这张图的关键在于把 GPU 想象成上百条并行泳道，真正产生算力的只有彩色横条覆盖的面积，空白全部是白付的电费。

![GPU 空闲时间线：纵轴 Processor ID、横轴 Time，绿色条为有用工作，红黄两簇之间是大片等待空白](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0032_00-36-23.jpg)

幻灯片标题写作 `Challenge: Stragglers and Kernel Launch`（挑战：掉队者与 kernel 启动），纵轴标注 `Processor ID`，横轴标注 `Time`；画面左侧是一簇逐渐变短的绿色横条，右侧是一簇粉红色横条，两簇之间留着非常显眼的一大片空白。转写把空闲归成好几类来源，需要逐个分辨。第一类是 kernel launch（kernel 启动）与 kernel teardown（kernel 收尾）：演讲者说这些正是「红色和黄色里那些大空隙」，也就是一次 kernel 调用本身附带的固定开销。第二类是 tail effects（长尾效应）：他特意说明这和前面讲过的「一个很短的 prompt 与一个很长的 prompt 被一起处理」是同一回事，只不过这个现象一路下沉到了最基本的 attention 操作里——如果你在处理一个 batch 的输入，其中一个输入很短、一个输入很长，那么所有处理器都得等那个很长的输入跑完才能收尾。第三类是跨 kernel 的空隙累加：因为这些操作是分散在多个 kernel 上跑的，kernel 与 kernel 之间的缝隙会一段一段叠加起来，最终变成相当可观的总空闲。

把上面的量记号化，幻灯片与转写里真实出现的数值和空闲来源可以整理成下面这样（这里只是把原文的数值与枚举用数学记号重排，并未引入新的关系式）：

$$
N_{\text{SM}}(\text{H100}) = 132,\qquad N_{\text{SM}}(\text{B200}) \approx 148
$$

$$
\text{downtime} \in \{\ \text{kernel launch},\ \text{kernel teardown},\ \text{tail effects},\ \text{inter-kernel gaps}\ \}
$$

```text
Challenge: Stragglers and Kernel Launch
y-axis: Processor ID        x-axis: Time
bars = useful work          empty space = waiting
```

正因如此，演讲者给出了一句相当强的判断：no matter how well you try to write the kernel，无论你把单个 kernel 写得多好，GPU 里总还是会有 downtime。这句话是本段后面全部工作的动机——问题不出在某个 kernel 优化得不够，而出在「一个操作一个 kernel」这个组织方式本身的结构性开销上。往前看，它承接了此前关于 decode 阶段的结论：逐 token 生成时每生成一个 token 都要把模型权重重新加载一遍，计算量不大却受内存带宽限制，batch size 很小时系统更像一台昂贵的显存加载器而不是大规模并行计算机；往后看，它直接引出 37:32 开始的 Megakernel 思路，即用融合把空隙消掉。

这里有几个容易读错的地方。首先演讲者自己就说明这张图 this is obviously a cartoon（显然是个示意漫画），只是 drawn from a particular attention inference kernel（取材自某个具体的 attention 推理 kernel），所以不能把条形长度、空隙宽度当成精确 profiling 数据来引用。其次，tail effects 只有在同一个 batch 里输入长度差异很大时才显著，把它当成任何时刻都存在的固定损耗是误判。再次，把空闲笼统归因于「kernel 没写好」会导向错误的优化方向：launch 与 teardown 属于调用边界的开销，单靠改 kernel 内部指令是消不掉的。最后，132 与 148 是不同代 GPU 的 SM 数量对比，转写对 B200 的数字用了「I think」这种不确定措辞，引用时不宜写成确定值。

## Megakernel：把 GPU 当作分布式系统来调度 [37:30 - 39:40]

对策是 megakernel。演讲者的定义很直接：不要再「把模型里的每个操作当作它自己的操作、为它单独写 kernel」，而是 let's write a single kernel to cover multiple operations at once（写一个 kernel 一次性覆盖多个操作）。他把这个做法类比成 flash attention 里见到的 fusion（算子融合），但强调是 done more aggressively、across a larger number of things（更激进、跨越更多操作）。更关键的是视角的转变：GPU 不再被看成「一个设备、一次跑一个操作」，而是被重新想象成 a massive distributed system（一个大规模分布式系统）。此时你手上有一堆必须完成的工作，其中一些对另一些有依赖——转写里举例说「红色那些工作对部分绿色横条有依赖」——于是问题变成了纯粹的调度问题：How can I schedule it? How can I distribute the work to maximize my GPU utilization?（怎么调度、怎么把工作分派下去，才能让 GPU 利用率最大化）。换句话说，编程对象从「一个算子的实现」上升到了「一张带依赖的任务图的排布」。

![ThunderGQA 融合前后对比：A 为 Unfused/Unscheduled/Unloved 标 BAD，B 为 Fused/Scheduled/Loved 标 GOOD，右侧列出四条设计要点](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0033_00-37-31.jpg)

`Solution: Fused Megakernels (ThunderGQA)` 这张图把差别画得很直白：上方 A) 的小标题是 `Unfused, Unscheduled, and Unloved`，画面中间用红字标 `BAD`，绿色横条簇与粉色横条簇之间留着一道明显空档；下方 B) 的小标题是 `Fused, Scheduled, and Loved`，用绿字标 `GOOD`，粉色横条在绿色横条还没结束时就已经起步，两类工作大面积重叠，空档被压掉。右侧四条 bullet 是画面上独有、转写完全没有逐字念到的信息，值得原样转述：其一是 `Single Megakernel to schedule workloads end-to-end`，即用单个 Megakernel 端到端调度全部工作负载；其二是 `Overlap workloads with fine-grained control between SMs (using HBM as backing semaphore)`，即用 SM 之间的细粒度控制来重叠工作负载，并把 HBM 当作后备的 semaphore（信号量）——这一条点出了融合得以成立的同步机制，转写只笼统说了「调度」；其三是 `Eliminates kernel launch overheads`，消除 kernel 启动开销，正好对应上一章的第一类 downtime；其四是 `Attention inference kernel: 1.3-1.7x faster`，即 attention 推理 kernel 快 1.3 到 1.7 倍。转写在 38:23 给出的说法是「如果只把这套做法用在 attention inference 这一角，能得到 30 到 70% 的 speed ups」，两者是同一量级的两种表达：

$$
1.3\times \sim 1.7\times \ \Longleftrightarrow\ 30\%\ \text{--}\ 70\%\ \text{speedup}\quad(\text{attention inference kernel})
$$

```text
Single Megakernel to schedule workloads end-to-end
Overlap workloads with fine-grained control between SMs (using HBM as backing semaphore)
Eliminates kernel launch overheads
Attention inference kernel: 1.3-1.7x faster
```

再往上推一层，同样的思路可以施加到整个模型，而不只是 attention。转写在 38:33 说这是 one layer of a llama 1B model（Llama 1B 的一层），在 38:37 说他们把 the entire layer 拼成了 one kernel。对应幻灯片 `Whole-Model Megakernels`（38:38 画面）下方写着 `Llama-1B — all in one kernel`，整幅图是密密麻麻的彩色横条，右侧 `Legend` 逐条列出了被塞进同一个 kernel 的七类操作，这些标识符同样是画面独有：`No Op (0)`、`LayerNorm_QKV_MatVecRopeAppend (1)`、`Attn Partial (2)`、`Attn Reduction (3)`、`O Proj (Residual) (4)`、`LayerNormDoubleMatVecSilu (5)`、`DownProjResidual (6)`，旁边还有一长列细粒度事件标记（诸如 `controller_start`、`loader_start`、`launcher_start`、`consumer_start` 一类）。演讲者说这些横条 overlapped in really weird ways（以很古怪的方式重叠），原因正是他们开始把下一层的 weight load 与当前 attention 叠在一起，或者在 attention 操作还没结束时就提前跑一部分 reduction。从「消掉 kernel 之间的空隙」到「让不同层的操作互相穿插」，重叠的粒度又细了一级。

这一章承上启下的位置很清楚：它把上一章列出的三类结构性空闲，转成了「带依赖任务图的调度」这一个新问题，并为下一章的两个具体重叠例子和实现框架埋下伏笔。需要注意的局限也不少。第一，megakernel 不是把几个 kernel 简单首尾拼接，真正的难点是依赖关系与跨 SM 同步，幻灯片那句 `using HBM as backing semaphore` 恰好说明它要自己造同步原语。第二，`Llama-1B — all in one kernel` 说的是「一层」被装进一个 kernel，标题里的 Whole-Model 容易被误读成整个模型只跑一个 kernel，转写明确限定为 one layer。第三，`1.3-1.7x` 与 `30-70%` 都只针对 attention inference kernel 这一局部，不能当成端到端推理加速比。第四，融合越激进，与具体模型结构、具体硬件的绑定就越紧，可移植性越差——这一点演讲者在后面谈工程代价时会再次展开。

## 细粒度重叠与 ThunderKittens 的硬件抽象 [39:40 - 41:50]

有了调度自由度之后，就能做出传统写法根本做不到的重叠。转写先给出一个具体例子：现代 LLM 的 attention 层里有 QKV projections（QKV 投影），还要给它加上 rope scaling（RoPE 缩放），这些蓝色线就是 `QKV + RoPE`；橙色线则是 attention 计算的开头。关键洞察是 you can start loading in your KV cache into your attention before you're finished with QKV（在 QKV 还没算完时，就可以开始把 KV cache 加载进 attention）。尤其在 decode 阶段，转写点出「那些橙色条上带一些圆圈的」，表示 QKV plus rope 仍在运行时就启动了 KV cache load；等 QKV 一结束、拿到新的 query tokens，才接着跑 attention 剩下的部分。第二个例子在 40:05 附近：橙色仍是 attention 的第一部分，红色是 attention 之后接的 O projection，做法是让 O projection 在 attention 操作结束之前就开始加载自己的权重。演讲者把这类能力总结成一句：有了这种 fine grained control of GPUs（对 GPU 的细粒度控制），you can start some operations before others are over（可以在一些操作结束前就启动另一些操作）。

![细粒度工作负载重叠：蓝色为 QKV+RoPE、橙色为 ThunderGQA 的 attention 前半段，右侧注明 attention 在 QKV 完成前就开始加载 KV cache](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0035_00-39-45.jpg)

`Fine-Grained Overlapping of Workloads`（39:45）这张画面把上述配色写成了三行大字，都是转写没有逐字念到的原文：`Blue: QKV + RoPE`、`Orange: Attention partial (first part of ThunderGQA)`，以及结论句 `Attention starts loading KV cache before QKV is finished`。左侧除了与上一章相同的七项 `Legend`，还有一长列淡淡的事件名（`controller_start`、`controller_end`、`loader_start`、`loader_end`、`launcher_start`、`launcher_end`、`storer_start`、`consumer_start`、`consumer_end`，以及 `first_wait`/`first_store`/`first_allow`、`last_allow` 之类），说明这套调度确实是把每条泳道上的启动、装载、消费拆成了可独立编排的事件，而不是粗粒度地整块切换。

实现这些重叠的是一套「relatively complex CUDA framework」（相当复杂的 CUDA 框架）。转写在 40:19 到 40:36 描述它的抽象方式：采用 instruction based abstraction（基于指令的抽象），可以把每个 subkernel 写在自己的文件里，再配一套 big virtualized shared memory system（大型虚拟化共享内存系统）来 orchestrate（编排）这些操作的运行。这套库叫 ThunderKittens，演讲者在 40:44 把它定性为 one of these kernel writing libraries（一个 kernel 编写库），并给出最好记的比较：You can think of it as almost like Triton except more low-level, a lot more fine grain control over things（可以把它想成 Triton，但更低层、对细节的控制细得多）。这正是它能做到「QKV 未完就加载 KV cache」这类穿插的原因。

![ThunderKittens 的硬件抽象：左侧内存层次金字塔标注各层容量与带宽，中间 Load/store workers 与 Shared Buffers，右侧 Compute workers 的代码片段](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0036_00-40-52.jpg)

`ThunderKittens: Abstractions for Performant Kernels`（40:52）这一页是整个第 3 段信息密度最高、也最被旧笔记忽略的一页，上面的参数与代码转写完全没有念到。左侧金字塔从上到下给出四层存储的容量与带宽：`Reg 64 KB / 130 TB/s`（对应 `Tensor core register tiles`，并注明 `In TK as: rt<M, N>;`）、`Shared 227 KB / 33 TB/s`（对应 `Swizzled shared tiles`，`In TK as: st<M, N>;`）、`L2 Cache 50 MB / 12 TB/s`、最底下 `HBM 80 GB / 3 TB/s`，旁边另标 `Global Tensors`。把它写成一行：

$$
\text{Reg}\ \tfrac{64\,\text{KB}}{130\,\text{TB/s}},\quad
\text{Shared}\ \tfrac{227\,\text{KB}}{33\,\text{TB/s}},\quad
\text{L2}\ \tfrac{50\,\text{MB}}{12\,\text{TB/s}},\quad
\text{HBM}\ \tfrac{80\,\text{GB}}{3\,\text{TB/s}}
$$

中间 `Load and store workers` 框里写着 `Sync/Async loads and stores from HBM to Shared memory`，并给出三段原样代码；`Shared Buffers` 用编号 0/1/2 的 `Inputs` 与 `Output` 槽位表示流水缓冲；右侧 `Compute workers` 框写着 `Pytorch-like operations on shared and register memory tiles`，同样附代码：

```cpp
// Sync load to buffer slot 0
load(k_smem[0], K_global, idx0);
// Async load to buffer slot 1
tma::load_async(k_smem[1], K_global, idx1, bar);
// Async store from buffer slot 2
tma::store_async(o_global, o_smem, idx1);
```

```cpp
// In-register tiles
rt_fl<16, kv_height> attn_t;
// Tensor cores, SMEM -> registers
warpgroup::mm_ABt(attn_t, q_smem{workid}, k_smem[0]);
// Use non-tensor core units
exp(attn_t, attn_t);
```

页面底部的问句 `What are the right abstractions to capture these hardware patterns?` 点明了这套库的研究动机：不是再写一个更快的算子，而是找到能表达这些硬件层次（寄存器 tile、共享内存 tile、TMA 异步搬运、tensor core 与非 tensor core 单元分工）的抽象。可以看到 `rt<M, N>` 与 `st<M, N>` 正好对应金字塔顶两层，`load_async`/`store_async` 的 `buffer slot 0/1/2` 正好对应中间编号槽位，代码与图是一一对应的。

性能上，转写在 40:56 给出结论 `near speed of light decoding inference`（接近光速的解码推理），并特意补了一句 these numbers are actually a lot better now（这些数字现在其实好得多），提醒听众幻灯片偏旧。图中代表 megakernel 的是 teal（蓝绿色）那根柱子，明显快于其它 state-of-the-art engines；在 H100 上它 achieving 72% bandwidth utilization（达到 72% 的带宽利用率），也就是所谓 near the speed of light on the GPU。演讲者的解释是：如果 ignore all the complexities（先不管我们这里做的所有复杂事），只问「GPU 物理上做这个操作最快能有多快」，那么已经跑到了那个理论上限的 72%：

$$
\eta_{\text{bandwidth}}(\text{H100}) = 72\% \ \text{of speed-of-light}
$$

这一章的收尾是一个 takeaway：如果你对 kernel 有 very deep control（非常深的控制）并理解硬件，就能开启 very different compute paradigms（很不一样的计算范式），而 all these things you only see when you start playing with inference at a deep level（这些事只有深入到推理底层才会看见）。它与前后内容的关系是：前两章讲的是「为什么空、怎么把空填掉」，这一章给出「靠什么抽象才能真正填掉」以及填掉之后的量化收益。易错点有四处：72% 是「这个操作」相对物理带宽上限的占比，不是端到端推理的加速倍数；ThunderKittens 比 Triton 更低层，意味着更强的控制力同时意味着更差的可移植性与更高的开发成本；这些细粒度重叠需要手工编排依赖与同步，模型结构一改，事件编排也得重排；最后，演讲者明说数字已过时，引用 72% 与 1.3-1.7x 时必须带上时效说明，不能当成当前引擎的实测值。

## Parcae：换一种 scaling 维度的 looped transformer [41:50 - 45:40]

41:50 之后演讲者明确转场：now I'll talk about something that's a little bit new，getting into the realm of new architectures（进入新架构的领域），要讲一个叫 Parcae 的新模型。转写里这个名字被念成 `PERS`/`parse` 一类发音，幻灯片统一写作 `Parcae`，属于口误或转写噪声，以下按幻灯片书写。出处是 his UCSD lab（他在 UCSD 的实验室），由 Hayden 主导，并与 Zachary、Taylor 两位合作；`Something New: Stable Looped Models with Parcae`（41:59 画面）标题页上给出了三人全名 `Hayden Prairie`、`Zachary Novack`、`Taylor Berg-Kirkpatrick`，右下角还有 `Stanford` 字样。`Parcae: Our Take on Principled Looped Transformers`（43:07 画面）则把工作拆成两块：`Stable Looping with SSM Theory` 与 `The Scaling Laws: Scale Recurrence with Data`；右上两张小图的标题和刻度同样是画面独有——`IsoFLOP Scaling Laws of Looping`（纵轴 `Validation Loss` 约 2.9 到 3.5，横轴 `Recurrence` 刻度 2/4/8/12，多条彩色虚线呈 U 形并各带一个星号标出的最优点），以及 `Optimal Recurrence`（纵轴 2 到 12，横轴 `FLOPs` 为对数刻度 $10^{18}$、$10^{19}$、$10^{20}$，散点沿一条上升虚线排列）。转写只笼统提到「看到一些有意思的 scaling laws」，这些图名与刻度是补充增量。

动机要回到讲座开头。演讲者说，开头讲过 all these new capabilities are coming because you're starting to scale these models and parameters and data（所有新能力都来自放大模型、参数和数据）；Parcae 想问的是另一个问题：is this the only way you have to scale，还是存在别的、也能拿到这种质量的途径。他们的答案是 loop transformers（循环 transformer）这一类技术：取 transformer 的若干 block，让它们 run in a loop（跑在循环里）；token 不再一层一层只走一次，而是在经过某处时 just send it back through through that loop（把它送回那个循环再走一遍）。转写同时交代了两个配套判断：一是用 some state space model theory（SSM，状态空间模型理论）来 stabilize this operation（稳定这个操作），因为 naively if you just run this and let the thing train we saw that the thing was going to blow up（直接跑、直接训练会炸）；二是看到一些 scaling laws suggesting that you want to be scaling the recurrence of these recurrent models as you increase the data（随着数据增加，应该同步放大这些循环模型的循环次数），也就是说 in order to make the best use out of your parameters, you want to actually be reusing them somewhat（想把参数用好，就得在某种程度上复用它们）。

![Looped models 结构：蓝色 P 块把激活多次送回紫色循环块 R，注入 e，最后经红色 C 块输出，右下附 Advantages 两条及论文摘要](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0039_00-44-14.jpg)

机制在 `Looped Models - Pass the Activations Back`（44:14）这页画得很清楚：某个 activation 走到 looped blocks 时，the same activation is going to run through that same layer some number of times（同一个激活会反复经过同一层若干次），紫色块就是 recurrent block（循环块），最后得到输出。图示从 `"Looped"` 指向蓝色 $\mathcal{P}$ 块，$\mathcal{P}$ 以虚线向每一个紫色 $\bar{\mathcal{R}}$ 块注入 $e$；主链是 $h_0 \to h_1 \to h_2 \to \cdots \to h_T$，再经红色 $\mathcal{C}$ 块指向 `"Models"`。把这条链写成箭头形式：

$$
h_0 \xrightarrow{\ \bar{\mathcal{R}},\, e\ } h_1 \xrightarrow{\ \bar{\mathcal{R}},\, e\ } h_2 \longrightarrow \cdots \longrightarrow h_T \xrightarrow{\ \mathcal{C}\ } \text{Models}
$$

页面上的 `Advantages:` 只有两条，都是原文：`Increase FLOPs, keep parameters constant`（增加 FLOPs、保持参数量不变）与 `Higher expressivity`（更高表达性）。转写对此的解释是：you can keep your parameters constant but it gives you a dial to increase your flops（参数不变，但多了一个调 FLOPs 的旋钮）；如果相信 more flops equals higher quality，这就是一条不付更高参数代价而提升质量的路。表达性那条则对应几年前的旧工作（old is a relative term，演讲者自嘲「旧」是相对而言），认为同样参数量下有些东西是普通模型表达不了、循环模型却能表达的。驱动问题被概括成 what's the best quality per parameter，以及 intelligence per parameter in data（每参数质量、每参数加数据的智能）。右下角还贴了一篇论文截图 `LOOPED TRANSFORMERS ARE BETTER AT LEARNING LEARNING ALGORITHMS`，作者 `Liu Yang, Kangwook Lee, Robert D. Nowak & Dimitris Papailiopoulos`，单位 `University of Wisconsin, Madison, USA`；摘要里写着 looped transformer 能达到与标准 transformer 相当的表现而 utilizing less than 10% of the parameter count（只用不到 10% 的参数量）——这个 `10%` 是画面独有数字。

```text
Advantages:
- Increase FLOPs, keep parameters constant
- Higher expressivity
```

初期结果和一段插曲也值得记下。转写在 44:44 提到 Tom Goldstein 组（Maryland）的一篇论文暗示这类模型 might be better than transformers，并给出 ARC tasks 上的一些结果。更热闹的是 Twitter hype：发布 Parcae 大约一周前，有人在推特上说 `Claude Mythos is a recurrent / looped language model`，演讲者直言 I don't think it's right. I think he was just making it up（我认为不对，他就是编的），后来对方写了篇博客道歉说 `my bad. I just made that up. It's none of it's true`。炒作让话题在发布前 blew up in Twitter speculation，但演讲者强调 we were working on it before this Twitter hype。`(And Twitter Hype)`（45:21 画面）里可见 `@ChrisHayduk` 的推文原文，猜测 Claude Mythos 是 looped language model，引用 ByteDance 论文 `Scaling Latent Reasoning via Looped Language Models`，并称该论文把 graph search 列为 looping 相对标准 RLVR 有巨大理论优势的领域；推文附一张 SWE-bench 对比表（列为 `Claude Mythos Preview` / `Claude Opus 4.6` / `GPT-5.4` / `Gemini 3.1 Pro`，行含 `SWE-bench Verified`、`SWE-bench Pro`、`SWE-bench Multimodal`，数字有 93.9%、80.8%、80.6%、77.8%、55.4%、57.7%、54.2%、87.8%、77.8%）；右侧标题 `On the Looped Transformers Controversy`，配文 `So my brief speculation tied to the ByteDance Looped Transformer paper caused quite a stir on ML Twitter:`，下方互动数 2 / 20 / 184 / 33K。这些人名、论文名、任务名与百分比都是画面独有，转写只讲了「有人瞎猜、后来道歉」。

放在整段里看，这一章是个转折点：Megakernel 处理的是「给定模型后怎么把硬件用到极致」，Parcae 处理的是「换一种模型结构、换一个 scaling 维度」，两者共同呼应讲座开头「理解推理与底层 kernel 才能做全栈创新」的主张；同时它直接把问题交给下一章——这类循环模型极难训练。易错点也不少：把 loop 理解成「多加几层、多加点参数」是根本性误读，参数是被反复复用的、完全不变；`less than 10% of the parameter count` 是那篇 Wisconsin 论文在 in-context learning 的数据拟合任务上的结论，不能外推成通用预训练的普遍比例；Twitter 传言已被演讲者明确判定为不实，绝不能拿来当 Parcae 的证据；另外 IsoFLOP 图上那些星号与 U 形曲线，本段转写只说「提示数据增加时应增加循环次数」，并未给出具体拟合公式或幂律指数，不宜自行补写。

## 超参数不稳定：loss spikes 与「一切都在残差里」 [45:40 - 47:45]

上一章说循环模型值得研究，这一章立刻给出它最刺手的现实问题。演讲者在 45:38 说 one problem with these looped models：只要你去训练它们，然后 changed anything about the training algorithm but at all（对训练算法做任何一点改动），模型就会突然 blow up。他举的最轻的改动是把 learning rate 稍微调一点，结果 do a simple thing like a learning rate sweep（做个最简单的学习率扫描），九次里有十次——转写原话 nine times out of 10——模型 is just isn't going to converge（根本不收敛），会 get nans、get these big loss spikes（出现 NaN 和巨大的损失尖峰）。演讲者由此给出一条通用的工程判断：如果你在把模型规模往上放大时看到 these big loss spikes that suggest something has gone very wrong with your training process，那就该 take a deeper look 去查根因，而不是把尖峰当成随机噪声绕过去。这个判断是本段从「现象」转向「数学分析」的转折。

![超参数不稳定性：左侧 LR 扫描表标出 Base 与 Res. Norm 各配置是否收敛，右侧 LR=6e-4 的 Training Loss 曲线出现 Loss Spikes](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0041_00-46-28.jpg)

`But Huge Hyperparameter Instability!`（46:28）这页把「调一点就炸」量化成了表格和曲线，数字都是画面独有。左侧表格三列为 `LR` / `Base` / `Res. Norm`，五行分别是 `2e-4`（Base ✓、Res. Norm ✓）、`4e-4`（✗ / ✓）、`6e-4`（✗ / ✗）、`8e-4`（✗ / ✗）、`1e-3`（✗ / ✗）；下方一行结论 `RDMs with Different Norm Strategies Unstable`（采用不同 norm 策略的 recurrent depth models 都不稳定）。可以看到只有最小的 `2e-4` 两列都收敛，`4e-4` 时加了残差 norm 的版本还能撑住，再往上两种策略全崩。右侧曲线标题 `Training Loss (LR = 6e-4)`，纵轴 `Loss`（刻度约 8 到 10 以上），横轴 `Total Tokens (B)`（0.0 到 0.5），图例三条：`Res Norm`（蓝）、`Baseline`（橙）、`Parcae`（红）；从走势看橙色 Baseline 在约 0.1B 处 loss 上翘后中断，蓝色 Res Norm 在约 0.4B 处冒出明显尖峰，红色 Parcae 则平滑下降；右下角写着 `Loss Spikes!`。把这张表与坐标刻度原样记下来：

```text
LR      Base   Res. Norm
2e-4     ✓        ✓
4e-4     ✗        ✓
6e-4     ✗        ✗
8e-4     ✗        ✗
1e-3     ✗        ✗
RDMs with Different Norm Strategies Unstable
Training Loss (LR = 6e-4)   Loss ~ 8-10   Total Tokens (B) 0.0-0.5
Legend: Res Norm | Baseline | Parcae
```

$$
\text{LR} = 6\text{e-}4,\qquad \text{converge rate} \approx 1/10,\qquad \text{Loss} \in [8,\,10^+]
$$

以前的 work 面对这种不稳定用的是 hack，转写在 46:17 到 46:31 点了两种：you can put norms in every layer to figure out what's happening（每层都塞 norm 去压住），或者 just pick the learning rate of 2 e minus 4. Don't pick any of the other learning rates（干脆只挑 $2\text{e-}4$ 这一个学习率，别去选其它值）。后者正好对应表格里唯一两列都打勾的那一行——所谓「选对学习率」其实只是撞上了唯一稳定的格子。但演讲者的态度很明确：we were thinking hey the existence of these loss spikes suggests that there's probably something more deeper going on（尖峰的存在说明背后大概有更深层的原因），于是 took a bit of a mathematical state space modelesque approach to the stabilization question（用了一个偏数学、状态空间模型式的思路来处理稳定性问题）。

难点在于没法直接解析整个块。转写在 46:55 到 47:12 解释：if you try to analyze it analytically it's going to be very complicated，因为那个 big RB block（即循环块）里有 tons of parameters、all sorts of nonlinearities，还有 softmax、RoPE 等等。他们的基本洞察是换一个观察量：let's actually just look at the residual of this thing（只看这个东西的残差），问 how is this activation changing from block to block（这个激活从一块到下一块究竟怎么变）。第一个经验观察是 it actually doesn't change that much（其实变化不大）：each of these residual blocks is maybe changing the vector a little bit but not really having a massive impact（每个残差块只是把这个向量稍微动一点，并不产生巨大影响）。`Our Basic Insight: It's All About the Residual`（47:35 画面）用的正是与上一章相同的循环示意图，但把 $h_1$ 用红框单独圈了出来，下面配一行红字 `How does this h change?`，把注意力明确钉在相邻两步的差上。把这条经验观察记号化（注意这是对「变化不大」的记号化转述，并非幻灯片上写出的公式）：

$$
h_{t+1} \approx h_t \quad(\text{残差块只轻微改变激活，经验观察})
$$

这一章在结构上是承上启下的枢纽：它把上一章「循环模型值得研究」落到「但极难训练」的具体表现上，又因为「残差变化小」这个观察，才使下一章把整个非线性块塞进黑箱、只分析线性部分的做法变得合理。易错点要特别留意几处。其一，把 loss spike 归因为「学习率没调好」而只做 sweep，会掩盖结构性病因；表格里 ✓ 只表示该配置收敛，完全不代表模型质量好。其二，右下括注写的是 `(Red is Parcae)`，但按曲线走势红色 Parcae 恰恰是平滑的那条，尖峰出现在蓝色与橙色上，读图应以曲线本身为准，别被配色和括注带偏。其三，每层加 norm 只是压住症状，下一章和最后一章会说明它反而会制造新的尖峰。其四，`nine times out of 10` 是演讲者的经验估计，不是严格统计；把 `doesn't change that much` 理解成「残差恒等、循环没作用」也是错的——单步变化虽小，却会随循环次数累积放大，而这正是下一章爆炸机制的入口。

## 线性化、闭式解与谱半径 [47:45 - 51:10]

有了「只看残差」这个切入点，团队在 47:46 说 we went and wrote down a dynamic system over this residual（在残差上写下一个动态系统）。化简的手法很干脆：there's all this nonlinear stuff、attention、big feed forward network with the intermediates and stuff，we're going to all put that into a box we're just going to call that $\bar{\mathcal{R}}$（把所有非线性的东西全塞进一个盒子，叫它 $\bar{\mathcal{R}}$），并说明这个 $\bar{\mathcal{R}}$ is going to be some big nonlinear thing，先 stick it to the side（搁到一边）。剩下的就只有两个矩阵：$\bar{B}$ 是 some transformation over your initial vector（对循环开始前那个初始向量的变换），$\bar{A}$ 是 how do you transform that residual in each loop（每一圈里怎么变换这个残差）。演讲者强调，把 transformer 的全部复杂性放到一边之后，就得到一个 relatively simple way of looking at all the previous loop transformers（一个相当简单的视角，可以统一审视此前所有循环 transformer）。

![残差上的动态系统：h_t 经 A、e 经 B 注入，再叠加 Transformer Blocks 的非线性输出，下方给出递推式与 Previous Looped Transformers 表格](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0043_00-48-42.jpg)

`Dynamic System over Residual (SSM)`（48:42）把这页写得很完整，公式与表格都是转写没有逐字念到的原文。示意图左侧 $h_t$ 进入 $\bar{A}$、$e$ 经一个小块（标 `LN`）进入 $\bar{B}$、两者之间有 $\Delta$，右侧 $\mathcal{R}$: Transformer Blocks 框里画着 `Multi-Head Attention` 与 `Feed Forward` 两个模块，各带一个 $\oplus$ 残差相加，输出 $h_{t+1}$。幻灯片给出的递推式是：

$$
h_{t+1} = \bar{A}h_t + \bar{B}e + \bar{\mathcal{R}}(h_t, e)
$$

下方表格标题 `Previous Looped Transformers`，列为 `Method` / $\bar{A}$ / $\bar{B}$：`Addition` 行 $\bar{A} = I$、$\bar{B} = I$；`Concatenation` 行 $\bar{A} \in \mathbb{R}^{d_h \times d_h}$、$\bar{B} \in \mathbb{R}^{d_h \times d_e}$。这正对应转写在 49:05 到 49:13 的说法：一种做法是把 $A$ 当作 identity、直接相加，另一种是 a fully learnable matrix（完全可学习的矩阵）；但只有画面给出了矩阵维度 $d_h \times d_h$、$d_h \times d_e$ 这两种具体形状。这里 $\bar{A}$、$\bar{B}$、$\bar{\mathcal{R}}$ 上的横线（overbar）也是幻灯片写法，转写口语里只说 a and b matrices、call that r，记号以画面为准。

接下来的关键一步是「丢掉非线性」。转写在 49:19 到 49:37 说，他们 observed empirically that these $\bar{A}$ and $\bar{B}$ matrices are actually dominating the magnitude of this equation（经验上发现这两个矩阵支配了整个式子的量级），于是问 What happens if you just drop that complicated nonlinear piece?（把复杂的非线性项丢掉会怎样）。答案是 you get out actually a pretty simple system that if you use high school calculus, you can just solve it（得到一个相当简单的系统，用高中数学就能解出来），可以直接算出给定初始激活与初始注入时第 $t+1$ 步的激活长什么样。`Stability Analysis: Linearize the System!`（49:50 画面）先写下线性化后的式子并标注 `Residual Dominates Magnitude`，再在 `Closed Form Solution (High School Math):` 下方给出闭式解：

$$
h_{t+1} = \bar{A}h_t + \bar{B}e
$$

$$
h_{t+1} = \bar{A}^{t}h_1 + \Big(\sum_{n=0}^{t-1}\bar{A}^{n}\Big)\bar{B}e
$$

转写在 50:00 到 50:11 点出这个式子的性质：this thing is dominated by these $\bar{A}$ matrices，especially this $\bar{A}$ matrix that you are powering up to a large degree（尤其被那个被反复乘方的 $\bar{A}$ 主导）。也就是说，决定系统命运的既不是非线性块，也不是只注入一次的 $\bar{B}e$，而是 $\bar{A}$ 的高次幂以及几何级数里累加的 $\bar{A}^n$。

由此引出全段最核心的量。转写在 50:13 说 there's this quantity called the spectral radius of this $\bar{A}$（有个量叫 $\bar{A}$ 的谱半径），并给了个口语化解释 spectral radius is basically like another word for norm（谱半径差不多就是范数的另一种说法）。直觉是把矩阵反复乘方到很大的次数：if this matrix can learn to be something like let's say that this matrix is two and then this $t$ is like 16 or something（假设这个矩阵是 2、$t$ 是 16 之类），那么激活就被放大到 $2^{16}$，it's really big，而这 this starts to explain some of those big loss spikes（就开始解释那些巨大的损失尖峰了）：

$$
\bar{A} = 2,\ t = 16 \ \Longrightarrow\ \bar{A}^{t} = 2^{16}
$$

`Stability Analysis: Linearize the System!`（50:57 画面）在同样的闭式解上用两个红框特别高亮了 $\bar{A}^{t}$ 的指数 $t$ 和 $\sum$ 里的 $\bar{A}^{n}$ 的 $n$，并新增一行 `Key Quantity: Spectral Radius $\rho(\bar{A})$`，把「哪里会被放大」标得非常直白。页面下方的表格把历史做法一一判了刑，列头为 `Method` / $\bar{A}$ / $\bar{B}$ / $\rho(\bar{A})$ / `LTI Stability`：`Addition` 行 $\bar{A}=I$、$\bar{B}=I$、$\rho(\bar{A}) = 1$、判定 `marginally-stable`（临界稳定）；`Concatenation` 行 $\bar{A}\in\mathbb{R}^{d_h\times d_h}$、$\bar{B}\in\mathbb{R}^{d_h\times d_e}$、$\rho(\bar{A}) \in \mathbb{R}$、判定 `unstable`（不稳定）。这正对应转写在 50:52 到 51:07 的结论：the choices that you make for these A and B matrices ... are either marginally stable or unstable make the system unstable（此前论文对这两个矩阵的选法，要么临界稳定、要么让系统不稳定）。`LTI`（线性时不变）这个术语也只见于表格列头，转写没有念到。

这一章把前面几章串成了一条完整链条：第 5 章的「残差变化小」让非线性块可以被搁到一边，于是这里能写出只含 $\bar{A}$、$\bar{B}$ 的线性系统；而闭式解与 $\rho(\bar{A})$ 又直接给出下一章「约束 $\bar{A}$、$\bar{B}$」的着力点。易错点有四处。第一，闭式解是在丢掉 $\bar{\mathcal{R}}$ 之后的线性系统上得到的，它是用来解释爆炸机制的模型，不是对完整 transformer 的精确求解，不能拿它去算真实激活值。第二，把谱半径「就是范数」当成严格定义是错的，那只是演讲者的口语类比；谱半径是特征值模的最大值，$\rho(\bar{A})=1$ 只是临界稳定，并不保证训练一定收敛。第三，$2^{16}$ 是标量类比，真实矩阵还要看特征值分布，以及 $\bar{B}e$ 在每一圈被反复注入所产生的累加效应。第四，表格只列了 `Addition` 与 `Concatenation` 两种历史做法，不能当成所有 looped transformer 的完整分类；同时 $\bar{B}$ 只被应用一次这一点，转写要到下一章才明说，这里不宜提前替它下结论。

## 约束 A/B 稳定系统：稳定 loss 曲线与更高质量 [51:10 - 53:45]

理论判据一旦清楚，Parcae 的做法就顺理成章。转写在 51:10 到 51:20 复述了困境：if you take this particular look at it and you let these $\bar{A}$ and $\bar{B}$ do whatever they want you're going to get things that really explode（若放任这两个矩阵随便学，结果就会炸）；于是问 What if we just constrain $\bar{A}$ and $\bar{B}$ such that if you run the math they're not going to explode（能不能约束它们，使数学上就不会爆炸）。对 $\bar{A}$：we're effectively going to make that $\bar{A}$ matrix a negative diagonal matrix（把它做成一个负对角矩阵），这样 if you power that up the term eventually goes to zero so it doesn't blow up（乘方之后那一项最终趋于零）。对另一个矩阵（转写发音作 speed matrix，按上下文应为 $\bar{B}$）：we're going to stick a really simple linear norm against it（给它加一个非常简单的线性 norm），理由是 it only gets applied once and doesn't really blow up（它实际上只被应用一次，不太会炸）。结果在 51:48 给出：if you compute the spectral radius is now going to be less than one. It's now actually going to be a stable system（谱半径现在小于 1，系统变成稳定系统）。把「负对角」与判据记号化可写成：

$$
\bar{A} = -\mathrm{diag}(\lambda_i),\ \lambda_i > 0 \quad\Longrightarrow\quad \rho(\bar{A}) < 1
$$

训练结果印证了这个推导。`Stable System, Stable Loss Curves`（52:04 与 53:11 两张画面内容相同）并排给出两幅图，标题分别是 `Training Loss (LR = 6e-4)` 与 `Recurrent State Norm (LR = 6e-4)`，横轴都是 `Total Tokens (B)`（0.0 到 0.5），图例同为 `Res Norm`（蓝）、`Baseline`（橙）、`Parcae`（红）。左图纵轴 `Loss` 约 8 到 10 以上；右图纵轴 `State Norm` 是对数刻度，标着 $10^9$ 与 $10^{19}$。转写在 51:55 到 52:19 说：if you go train this model，you actually saw stable loss curve，Parcae 就是 this stabilized thing where we reparameterized a and b（把 $\bar{A}$、$\bar{B}$ 重参数化后的稳定版本）；即便用 the 6E minus 4 learning rate that was so bad for the other models（那个对其它模型很糟的 $6\text{e-}4$ 学习率），最后仍 got a stable model，而且能 naturally constrain the state norm of the activations（自然地约束激活的状态范数）。

![稳定系统与稳定损失曲线：左为 LR=6e-4 的 Training Loss，右为 Recurrent State Norm，橙色 Baseline 飙到 10^19 附近，Parcae 保持平稳](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0046_00-52-04.jpg)

曲线本身把三种做法的差别画得很清楚。右图橙色 Baseline 从约 $10^8$ 一路飙到接近 $10^{19}$ 后，在约 0.08B tokens 处就中断了——这正是转写在 52:26 说的 that orange baseline is just a completely unconstrained model，it blows up goes to 10 to the 19th（完全不约束的基线，炸到 10 的 19 次方）；蓝色 `Res Norm` 与红色 `Parcae` 则稳定在 $10^9$ 附近的水平线上。左图里橙色 Baseline 在约 0.1B 处 loss 上翘后中断，蓝色在约 0.4B 处出现明显尖峰，红色 `Parcae` 平滑下降到约 7。把画面信息原样记下：

```text
Training Loss (LR = 6e-4)        Loss ~ 8-10      Total Tokens (B) 0.0-0.5
Recurrent State Norm (LR = 6e-4) State Norm 1e9 / 1e19
Legend: Res Norm | Baseline | Parcae
```

$$
\text{LR} = 6\text{e-}4,\qquad \|\text{state}\|_{\text{Baseline}} \to 10^{19},\qquad \|\text{state}\|_{\text{Parcae}} \sim 10^{9}
$$

为什么「只加 norm」不够，是这一章最有价值的解释，也回头收拾了第 5 章那些 hack。转写在 52:21 到 53:13 说，蓝色线是 a model where you are applying a norm to it（给模型施加 norm 的版本）；模型本身其实 trying to expand the activations（想扩大激活），因为 with more room I can represent different things better，能把不同概念 put these different concepts further away from each other（在表示空间里摆得更开）。但你又要 norm it back down to one（把它压回 1），于是形成 these two pressures that are kind of fighting against each other（两股互相打架的压力），manifest in loss spikes（表现为损失尖峰）。所以 even though on the right your norms are very good、看不到激活爆炸，the loss can do some pretty gnarly things（loss 仍会出现很难看的行为）；右图范数漂亮与左图 loss 尖峰可以同时发生。结论一句是 this pretty simple change in the activations can stabilize the training, stabilize these recurrent loops（对激活做这个相当简单的改动，就能稳定训练、稳定这些循环）。

最后是质量。转写在 53:26 说 it's not only a more stable system，you actually also get more higher quality models；53:35 提到有一张表把 Parcae 模型与 a previous loop transformer called recurrent depth models（之前的循环 transformer，叫 recurrent depth models）作比较，可见 higher performance across a variety of applications，以及 Parcae outperforms the previous loop models。转写在 53:43 的 `and...` 处被截断，因此这里不补写表格里的具体数值、也不补充后续对比对象，留白比猜测更忠实。

放在整段结尾看，这一章把第 6 章的线性判据 $\rho(\bar{A})<1$ 变成了可训练的重参数化，并用三条曲线同时验证了「loss 稳」与「state norm 稳」；它也呼应了开头 Megakernel 那一节的 takeaway——正因为深入到激活与矩阵这一层去看，才可能既拿到新范式又拿到可训练性。仍要留出几处局限：$\rho<1$ 保证的是线性化系统的稳定，不等于任意学习率、任意循环次数下都必然收敛；负对角 $\bar{A}$ 相当于给循环加衰减，本段并未讨论这种约束对长程信息保留与表达能力的影响，不能替它下结论；此外这一节只展示了稳定与「跨多种应用更高性能」的定性对比，没有给出循环次数 $T$ 的选取依据、推理时额外开销，以及与标准 transformer 在同等 FLOPs 下的完整对照，转写又在此处截断，因此不宜把「更高质量」外推成任意设置下都成立的普遍结论。

## 从经典 Scaling Laws 到「何时该 Loop」：Part 3 的引入 [53:45 - 55:44]

本段承接上一段 Parcae 的质量结论。转写开头（53:49）演讲者补了一句 "also outperforms strong transformer baseline"，并特意说明这个基线不是随便搭的：它是 nanochat 一类的模型，很多人在努力让它 "learn as fast as it can"（尽可能快地学会）。把同一个基本 Transformer 架构拿来做 looping（循环）、再加以 stabilize（稳定化）之后，得到的不是单点胜利，而是 better perplexities（更好的困惑度）与 better end-to-end quality（更好的端到端质量）。困惑度是语言模型对下一个 token 不确定性的指数化度量，越低越好；端到端质量则指下游任务上的实际表现。理解这一点很要紧：Parcae 的卖点并不是「参数少所以省显存」，而是在同等架构起点上把质量也做上去了，因此后面才有必要认真追问「要不要更激进地循环」。

紧接着（54:13）他说 "we started to run some very basic scaling laws here"，把问题从「能不能循环」推进到「该循环多少」。他坦率地把前面的结果称为 "really really cute"（很精巧、但也可能只是玩具），真正要问的是有没有证据支持更激进地循环。画面此时切到黑底章节页（54:23），标题是 Part 3: Scaling Laws - When to Loop?，右下角带 Stanford 标识 —— 这是转写完全没有念出来的章节标题，说明整场讲座在幻灯片上被明确切成「第三部分：缩放定律——何时循环」。为了让听众跟上，他先 "take a step back" 回到几年前：当模型进入需要大规模扩展的阶段，一个自然问题是要把模型做得更大（scale model parameters），还是干脆喂更多数据（scale data）。

![Part 3 章节页：Scaling Laws - When to Loop?](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0048_00-54-23.jpg)

于是就有了那些 "very pretty"、"lots of colors" 的 power law curves（幂律曲线）。演讲者给出读图直觉：如果曲线向右下方倾斜（down and to the right），说明数据和参数应该同时放大；如果几乎垂直向下，说明只增加训练数据即可、没必要增加参数；如果几乎水平向右，说明一点数据都不用加、只堆参数就行。他给的现实坐标是 "a one trillion parameter model on 35 trillion tokens"，即一万亿参数配三十五万亿 token，并说这样会得到更好的质量。这里有一个易错点：曲线的三种走向是「在给定算力下如何分配预算」的定性判读，不是可以外推到任意规模的物理定律；曲线来自特定训练配置的拟合，换 tokenizer、换数据分布，刻度就会变。

55:40 那张幻灯片把这套语言写成了公式，标题是 How to Scale Training FLOPs?。左侧 "What to Scale:" 下只有两个 bullet：Model size (parameters) 与 Data；中间给出传统缩放定律（Traditional Scaling Laws）的拟合式，并标注 params N, tokens D：

$$\hat{L}(N, D) = E + X \cdot N^{-x} + Y \cdot D^{-y}$$

其中 $\hat{L}(N,D)$ 是预测损失，$N$ 是参数量，$D$ 是 token 数，$E$ 是不可约的常数项，$X$、$Y$ 与指数 $x$、$y$ 是拟合系数：参数项和数据项各自以幂律衰减，损失因此随两者增大而下降。画面下方并排三张小图：最左是 Training Loss 对 Parameters 的 U 形曲线族，纵轴刻度约 2.0 到 3.2，横轴刻度 100M、300M、1B、3B、6B、30B，每条曲线对应一个算力预算，图例从 $6\mathrm{e}18$ 一直到 $3\mathrm{e}21$ FLOPs；中间与右侧两张分别是 Parameters 对 FLOPs、Tokens 对 FLOPs 的散点与拟合直线，纵轴分别标到 1T 参数与 10T token 量级，横轴 FLOPs 从 $10^{17}$ 跨到 $10^{25}$。底部结论写着 Traditional Scaling Laws: Scale Parameters and Data Jointly，页码 64。

![How to Scale Training FLOPs 幻灯片：缩放定律公式、What to Scale 与三张等算力曲线](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0049_00-55-40.jpg)

这三张小图正是「向右下」判读的量化版本：不同的等算力曲线各自有一个最优参数点，算力越大，最优点同时向右（更多参数）和向下（更低损失）移动，把最优点连起来就得到「两者同时放大」的配比。它与本章前文的衔接在于，前面讨论的循环次数提供了一个全新的、不写在 $N$ 和 $D$ 里的旋钮；下一章要问的正是：把这个旋钮加进拟合之后，幂律还会不会同样指向「同时放大」。局限也很清楚：幻灯片上的公式只含 $N$ 与 $D$、没有 recurrence 项，因此它本身无法回答何时该 loop，只能作为待扩展的基线。

## 把 Recurrence 纳入 Scaling Law：ISO-param / ISO-FLOP 曲线 [55:44 - 58:36]

55:44 抛出核心问题 "where does recurrence fit into this"。演讲者先列出三种可能结论：一是 "you should never run recurrence"，即保持单次前向的普通模型最好；二是 "you should do a ton of recurrence"（大量循环）；三是 "maybe only some recurrence to the bit"（只循环一点点）。把循环次数记作 $t$，直觉上 $t$ 既不增加参数量 $N$，也不改变数据量 $D$，却线性地放大训练算力，因此它是传统二维缩放平面之外的第三个自由度。回答这个问题不能靠单点实验，要靠一族曲线，于是他们做了 "very basic scaling laws"。

曲线设置是（56:13–56:40）：每一条曲线对应一个模型，左右两图的参数量相同；随着颜色变化不断增加用于训练的 FLOPs，而增加 FLOPs 的方式是增加数据量；于是实验同时 varying data 与 varying the number of recurrences。56:57 的幻灯片标题是 Parcae Scaling Laws: To Loop or Not To Loop?，左侧 "Setup:" 下写着两条：ISO-param, ISO-FLOP curves，以及 Varying: recurrence x training data；下方还有一行红色结论 "More data -> more recurrence is optimal!"。ISO-param（等参数）意味着比较时锁死模型大小，ISO-FLOP（等算力）意味着每条曲线上的点消耗相同算力，这样横轴的移动才纯粹反映「数据换循环」的取舍。

右侧是两张并排的坐标图，顶部标着 140M 与 370M，即两个不同参数规模的模型。纵轴是 Validation Loss，左图刻度约 2.9 到 3.5，右图约 2.70 到 2.80；横轴是 Recurrence，刻度 2、4、6、8、12。图例是一排按颜色渐暖的虚线，标着 8 个算力档位：$1\mathrm{e}{+}18$、$2\mathrm{e}{+}18$、$4\mathrm{e}{+}18$、$8\mathrm{e}{+}18$、$16\mathrm{e}{+}18$、$32\mathrm{e}{+}18$、$64\mathrm{e}{+}18$、$128\mathrm{e}{+}18$ FLOPs。每条曲线上有一个黑色星号 $\star$ 标出该算力下的最优循环次数，还有一支醒目的红色大箭头从左上指向右下，配红字 "Down and to the right: Scale both"（页码 66）。把星号连起来读：算力档位越高（数据越多），最优 recurrence 越往右移、损失越往下走 —— 这正是幻灯片红字的图形化证明。

![Parcae 缩放定律：140M 与 370M 模型上循环次数对验证损失的等算力曲线](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0050_00-56-57.jpg)

把幻灯片上这几行关键文字按原样记下来，就是本节的实验设定与结论：

```text
Setup: ISO-param, ISO-FLOP curves
Varying: recurrence x training data
More data -> more recurrence is optimal!
Down and to the right: Scale both
```

演讲者进一步说，这些 recurrences 遵循 "pretty classic power laws"，因此可以得到 scaling laws 去 predict quality as you scale recurrences and your tokens jointly（在循环次数与 token 数联合放大时预测质量）。57:14 有听众追问「联合缩放」的结论，他回答团队原本做过一个 "really complex 3D figure"，把 recurrences、data、parameters 三个轴放在一起，那张图同样指向「同时向下又向右」，也就是应该 scaling all three together；但因为 3D 图很难看、很奇怪（3D and weird），没有放进幻灯片。这里的易错点是别把三类幂律混为一谈：一类说增数据应增参数，一类说增数据应增循环，二者叠加才推出「三者一起放大」，单看二维图无法直接得出三维结论。

58:10 他收束到一个关键观察：如果固定模型大小、增加数据量，就应该同时增加 recurrence；而 "as far as I know, all of our models today have no recurrence in them"，当今所有模型几乎都不带循环，相当于全部挤在这些曲线的最左端，同时却用着 "a ton of data"。这个错位暗示大规模预训练里可能存在 "something slightly better that we could be doing"。与前后内容的关系是：上一章的 $\hat{L}(N,D)$ 只管参数与数据，本章把 recurrence 补成第三个可缩放量；下一章则用一个固定算力预算的直接对照，检验「把算力花在循环上、而非只花在数据上」是否真的更划算。局限在于演讲者反复强调这些只是 "initial scaling laws"、"very basic"，且三维结论依赖一张他自己都说难看的图，不能当作定论。

## 固定 FLOP 预算的对照实验与全栈创新 Takeaway [58:36 - 60:40]

58:36 一句 "here's just that experiment in another way" 把同一发现换了个画法。橙色曲线是 fixed depth model，也就是传统的、每层只走一次的 Transformer；蓝色曲线是在 fixed flop budget（固定算力预算）下，循环模型落在曲线的哪个位置。演讲者特意说明：orange 与 blue 的点用 the same number of flops 训练，但 a different amount of data，而且两者 the same size（模型大小相同）。于是对照被压缩成一个干净判据：在 $N$ 相同、$\text{FLOPs}$ 相同的前提下，只改变「算力是靠更多数据、还是靠更多循环次数堆出来的」，看谁的 validation loss 更低。

把口述的对比写成符号（仅为概括演讲者给出的对照关系，不是新拟合式）：

$$\text{FLOPs}_{\text{orange}} = \text{FLOPs}_{\text{blue}}, \quad N_{\text{orange}} = N_{\text{blue}}, \quad L_{\text{blue}} < L_{\text{orange}}$$

结论是 "When you get to that number of flops by increasing recurrences as well as and not just increasing data, you start to get smaller validation losses"，即把算力通过增加循环次数（而不只是增加数据）花掉时，验证损失更小。他由此抛出那句略带挑衅的推论：这或许意味着 "we should be looping all of our big pre-training runs"（所有大规模预训练都该加循环）。与前一章的关系很清楚：上一章是从一族等算力曲线上读出「数据增则循环增」的趋势，本章则是在单一算力点上做受控对照，把趋势变成可直接比较的两个点。易错点：这两个点的数据量不同，所以差异不能简单归因于「循环更好」，而应读作「在固定算力下，把预算分给循环比全分给数据更划算」这一联合结论。

59:30 他说 "that's parse"（Parcae 一带而过的快读），随即 "take a little step back" 回到整场讲座的 big takeaway。59:30 的黑底幻灯片把这句话写全了：Takeaway from today's talk: Understanding inference and GPU kernels enables full-stack innovation in ML algorithms，右下角是 Stanford 标识。这句是转写里念到、但幻灯片把它定稿成一句话的关键增量。所谓 full-stack innovation（全栈创新），指从推理引擎、GPU kernel 一直到模型架构这一整条栈都能产生研究问题，而不只是调模型超参。

![黑底总结页：理解推理与 GPU kernel 赋能机器学习算法的全栈创新](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0052_00-59-30.jpg)

他把这条栈拆成三类例子：新的 routing algorithm（路由算法）让你 serve more traffic、或以不同方式服务流量；新的 kernels 让系统的某一部分跑得 "way faster"（快得多）；新的 architectures 在参数少很多时，可以把模型塞进 GPU 的一个子集里、换一种切法，从而 reduce the amount of communication（减少通信量）。这三类分别对应讲座前面讲过的 cache-aware prefill-decode disaggregation、Megakernel、以及 Parcae 本身，构成一次对全篇的回收。他还说 "hopefully I've inspired at least one of you in this room to go take a deeper look at some of that"，60:30 正式道谢并开放提问。这里要注意别把 takeaway 误读成「只有系统优化重要」：演讲者的措辞是这些 pieces enables（赋能）算法创新，强调底层理解带来的设计自由度。

局限方面，本节的对照实验只给了两条曲线，没有给出具体数值刻度或参数规模，转写里也未念出橙蓝两点的坐标与损失值，因此不宜臆测差距幅度；"looping all pre-training runs" 是用 "it might be the case" 引出的推测，不是被大规模验证的工程结论。此后画面长时间停留在 Thank you 总结页上，问答环节的所有讨论都在这个背景下进行，这也解释了为什么后面几张截图里投影幕内容始终不变。

## 问答（一）：能否循环预训练模型，以及推理侧的内存含义 [60:40 - 63:28]

主持人招呼大家举手提问，第一个问题（60:46）问得很具体："are you creating from scratch on the loot? I guess I was wondering if you could kind of take a matrix." 演讲者把问题复述为：Parcae 是 training from scratch（从头训练）吗？对 pre-trained models（预训练模型）能做点什么吗？答案是目前的工作确实从头训练，但存在一个反直觉的旁证：几个月前有一篇 "really troll blog post"，作者宣称 "won some leaderboard competition without training a single thing"（一次训练都没做就赢了某个榜单比赛）。他的做法是把一个 Qwen 模型的 two or three layers（两三层）拿去 loop，结果发现 "on some math things it started having higher quality"（在某些数学任务上质量变高）。

团队有一点 look work 在跟进，"may be coming out soon"（可能很快出来），但演讲者坦言 "which is really weird... it kind of disturbs me like I don't know why why that would possibly be the case"，即他本人也不知道为什么循环一下预训练模型就会变好，甚至被这个现象困扰。他还半开玩笑说，如果能说服 Hayden，就让他去盯 the activations、the actual weights（激活值与权重本身），弄清为什么 loop 之后会变好。这里的易错点很重要：这是对一个博客个例的转述，涉及特定模型的 2–3 层、特定数学任务，既没说明是哪几层、也没给出循环次数，不能推广成「任何预训练模型循环一下都会涨点」；把它当作开放现象而非结论才符合原意。用符号记下这个待解释的关系：

$$L_{\text{loop}}(\text{Qwen},\,2\text{–}3\ \text{layers}) < L_{\text{original}}\quad(\text{on some math tasks, unverified})$$

第二个问题（62:02）转向 "inference implications and the memory"。演讲者说，他个人对 loop 模型特别兴奋的一个原因，是 serving inference efficiently 的一大瓶颈恰恰是 GPU memory（GPU 显存）。参数更少带来一连串好处：可以 fit more KV cache（容纳更多 KV 缓存），或者 do less communication，因为需要把模型切分到 fewer GPUs（更少的 GPU）上。KV cache 是解码阶段缓存注意力键值张量的结构，其容量直接决定能并发服务多少会话；"a smaller model will get you a lot of flexibility" 正是本节与前半场 KV cache 分层存储、缓存感知路由内容的接口。

他接着讲了一个还没实现的 "dream"：如果 recurrent block 能做得足够小，就可以写一个 little mega kernel，在一个 very fast 的循环里把这个 recurrence 跑完；"so far we haven't been able to make those blocks small enough"，即目前块还不够小，梦想尚未落地。硬件侧的动机来自下一代 LPU：他提到 Groq 一类芯片（转写记作 Grock chips）与 Nvidia 相关的动向，说这类芯片 "have like 250 megabytes of memory or something like that"（大约 250 MB 内存），能塞进去的东西非常非常小；但也许可以专门设计一个刚好放得下的模型，让 weights 一直 keep in memory（常驻内存），只把 activations 尽可能快地流过去。若跨过这类容量阈值，就能获得 nonlinear benefits（非线性收益）—— 注意是「如果跨过阈值」，不是线性地省一点显存。

![问答现场：投影幕停留在 Thank you 三栏总结页，右侧听众持麦克风提问](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0054_01-02-04.jpg)

62:04 的现场截图恰好记录了问答开始后的舞台状态：投影幕上一直停留着 Thank you! 总结页，三栏标题为 Lifetime of a token、Fast LLM Inference with Megakernels、Something new: Parcae（第三栏被人物剪影部分遮挡），左侧是站在讲台旁的演讲者，右侧有人手持麦克风提问，画面右下有 Stanford 水印。这说明总结页在整个问答期间没有切换，后续讨论是对这三栏内容的口头展开，而非新幻灯片。局限上，250 MB 这个数字演讲者自己加了 "or something like that"，属于约数；mega kernel 循环也明确是尚未实现的设想，不宜写成已有成果。

## 问答（二）：Megakernel 的工程代价与面向硬件的架构设计 [63:28 - 66:19]

63:28 的问题问得很短："the trade-offs are always like strictly optimal"，主持人复述为 "what are the trade-offs for micro kernels"（转写作 micro kernels，紧接前文应指 Megakernel）。演讲者的回答只有一句俏皮话却很重："the trade-offs are people's blood sweat and tears"（代价是人的血、汗和泪），意思是 Megakernel 极其 labor intensive（劳动密集）。他给了一个可量化的估算：一位 full talented kernel engineer（完全称职、有才华的 kernel 工程师）over the course of a year（用一整年），大概只能为 one hardware（一种硬件）、two or three models（两三个模型）、batch sizes one to 16（批大小 1 到 16）写出 Megakernel。

把这段口述的适用边界记成符号：

$$\text{batch size} \in \{1, \ldots, 16\} \Rightarrow \text{works};\qquad \text{batch size} = 17 \Rightarrow \text{start over}$$

他原话是 "You go batch size 17, you're like, nope, start over. Got to go again." —— 批大小一变到 17，就得推倒重写。Together AI 正尝试 "put together some compilers that can automate some of that process"（用编译器把部分流程自动化），但他强调这仍是 very challenging 的事。易错点在于：这不是「模型一换就重写」那么简单，而是连 batch size 这种运行时参数都被硬编码进融合方案，说明 Megakernel 的性能来自对具体形状的极致特化，通用性与速度在这里直接对立。

他还给了一个历史视角：the basic mega kernel idea 在 GPU 编程史上 "gone in peaks and troughs over the last few decades"（几十年里几度起落）；结论是 "if you can do it it will go super fast, you will never be able to go faster"，但 "it just takes a lot of energy and a lot of effort"。这与前半场在 H100 上达到 72% 带宽利用率的结果互为表里：上限极高，代价也极高，因此更适合少数被反复服务的热点形状，而非普遍替代普通 kernel 写法。

64:47 的问题换到 code design：如果预先知道 serving platform 是 Groq、Cerebras 这类新硬件，架构该怎么改？演讲者给的第一原则是 "you are going to be most constrained by memory first"（最先受内存约束）：先去查目标芯片（他举 Cerebras wafer 为例）有多少内存，再据此 size your model，让模型放得下、并且还留出 enough KV cache 或其它余量。他随后提到一个观察：仔细看近期出现的中国模型，会看到一些 interesting choices，暗示它们可能已经在考虑 Huawei chips（华为芯片）的动向，尤其体现在 quantization choices（量化格式选择）上。这里措辞是 "suggest they might be starting to think about"，属于推测，不能写成既定事实。

具体的格式例子是：如果打算在 Nvidia GPU 上服务，例如 Nvidia 发布的 Neotron model，就会用 NV FP4 来训练 —— 这是一种 proprietary to Nvidia chips（Nvidia 芯片专有）的 FP4 格式；如果不是在 Nvidia 芯片上跑，比如 AMD，则会用另一种叫 MX FP4 的格式。他补一句 "They each have their pros and cons"，并总结这些 subtle choices 都取决于所选硬件。$\mathrm{NV\ FP4}$ 与 $\mathrm{MX\ FP4}$ 都是 4 比特浮点表示，差别在缩放因子与分块的组织方式，会直接影响精度与硬件利用率；把格式选择前移到训练阶段，是「面向硬件做架构 / 协同设计」最具体的体现，也与上一节「参数更少 → 显存更宽裕」的思路互补。

![Thank you 三栏总结页：token 生命周期、Megakernel 吞吐柱状图与 Parcae 循环架构](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0056_01-04-57.jpg)

64:57 的截图是那张贯穿问答的 Thank you! 三栏页的清晰版（页码 70），转写完全没有念到其中细节。第一栏 Lifetime of a token 是一张黑底流程图 What's in an Inference Engine：节点依次为 Decoder In、Scheduling（continuous batching、chunked prefill 等）、KV Cache（prefix sharing、offload to CPU 等）、Execution（prefill、decode、use specific kernels）、Parallelism（intra-GPU、multi-node、multi-GPU kernel）、Token Out；另有 What's Next 指向 multi-token prediction / speculative decoding，左下角 Key themes 写着 scale、KV compression、disaggregation、memory efficiency。第二栏 Fast LLM Inference with Megakernels 是柱状图 Llama-1B (BF16) Batch Size 1, Decoding Throughput，纵轴 Pred/s 刻度 0 到 1600，横轴两块 GPU：H100 与 B200；图例为 VLLM（紫）、SGLang（橙）、Megakernel（青绿），H100 上约 500 / 700 / 1000，B200 上约 400 / 1000 / 1500，Megakernel 在两代卡上都最高。第三栏 Something new: Parcae 是循环块架构图：状态 $h_t$ 经矩阵 $A$、输入 $e$ 经矩阵 $B$，中间有 $\Delta$，右侧是 $\mathcal{R}$: Transformer Blocks，内含 Multi-Head Attention 与 Feed Forward 两个子块、带残差相加 $\oplus$，输出 $h_{t+1}$，并有回绕箭头表示循环。这三栏正好把「系统栈—融合 kernel—新架构」三条线并置，是本节两个问答共同指向的全局图景。

## 问答（三）：Compute-optimal 抉择与不同用例下的架构差异 [66:19 - 69:59]

66:19 的问题直指动机："if you just care about compute optimal training, is it ever optimal to boot as opposed to just having more parameters, or is it mainly like a trick to reduce inference cost?"（若只关心算力最优训练，循环是否真的优于加参数，还是主要只是降低推理成本的技巧）。演讲者先重述问题，再点明 compute optimal 的关键限定是 "usually like given a flop budget"（通常给定算力预算），因此 "it's almost a little bit contrived in that sense"（在这个意义上有点人为 / 刻意）。也就是说，最优解是相对一个外部给定的预算而言的，不是绝对意义上的「循环优于参数」。

把他的决策树写成符号化概括：

$$\text{higher quality} \Rightarrow \uparrow \text{FLOP budget};\quad N\ \text{fixed} \Rightarrow \text{train longer};\quad N\ \text{restricted} \Rightarrow \text{loop longer};\quad D\ \text{exhausted} \Rightarrow \text{pick best-trained } N$$

口述对应的是：想要更高质量的模型，就直接增加 flops budget；如果已经决定了模型大小，就 train longer（训练更久）；如果受模型大小限制，就 loop longer（多循环）；如果数据耗尽，就挑一个在该规模下能被训练得 as well trained as you can 的模型大小。他进一步把这些选择放进现实约束里："do I think it'll get picked up? How am I going to serve it?"、如果开源发布，"what is the size of model that people can serve like on your laptop today?"（大家今天能在笔记本上服务的模型有多大）。最后给了一个朴素的兜底判断：如果 just make the model bigger and train it on more data，"it's always going to get better"。易错点是把 compute-optimal 当成循环合理性的证明：演讲者恰恰指出预算是外生设定的，循环只是在 $N$ 被卡住时的替代旋钮，而非替代「加参数加数据」的普适最优。

67:54 的问题转向 workload：讲座开头提过 agentic coding 与 batch processing of data 等不同用例，最优架构差异最大的地方在哪？演讲者先点明 agentic looped workflows 的诉求是 "you want to keep your KV cache as hot as possible"（让 KV cache 尽可能保持热），因为多轮、工具调用、模型自我迭代会反复回访同一段上下文；相反，做 big batch processing（例如把每个文档看一次就翻译掉）时，KV cache "doesn't necessarily matter as much"。这与前半场 prefix sharing、缓存感知路由的内容一脉相承：缓存的价值取决于复用次数，而不是绝对容量。

由此引出几类架构上的 "big departures"。他举 DeepSeek MLA attention，说它是 "a radical compression of the KV cache compared to what you would have for another model"（相对一般模型对 KV cache 的激进压缩）；又提到若模型能以 FP8 或 FP4 处理 KV cache，也是 KV cache 尺寸上的显著偏离。对 agentic workflow 敏感的人，会优先看这些。但「最大的那个」区别是 causal attention 或 non-causal attention（因果注意力 vs 非因果 / 双向注意力）：批处理场景下 Google 很长时间（他说 "probably still uses"）在搜索里用 BERT 类模型，因为不需要在另一端生成一堆 token，做一次 big bidirectional attention、拿到一个 vector，塞进数据库即可；而 chat workflows 总有 this decode portion（解码这一段）。他还把 T5 列为 intermediate thing（中间方案）：既做一些 bidirectional processing，也做一些 generation。

把用例与架构的对应关系补成一张对照（列表仅作补充，不代替上面的解释）：

```text
agentic / chat  -> KV cache hot, causal attention, decode 常驻
batch / search  -> KV cache 复用少, bidirectional attention (BERT), 无大规模 decode
intermediate    -> T5: bidirectional processing + generation
```

主持人补了一句现实约束：模型开发者 "has to pick one and tries to not be unreasonable on many use cases"（只能选一个架构，并尽量在多种用例上都别太离谱）。这也解释了为什么 $\mathcal{R}$ 里的循环块、KV cache 压缩格式这些选择不能只按单一 workload 优化。

![问答期间保持不变的 Thank you 三栏总结页（页码 70）](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0058_01-07-56.jpg)

67:56 的截图仍是页码 70 的 Thank you! 三栏总结页，与 64:57 那张完全一致，说明这一整段问答期间投影内容没有变化；因此本节关于 compute-optimal 与 workload 的全部信息都来自口述，画面只是保留了讲座的三栏总纲。易错点：演讲者对 Google 是否仍在用 BERT、T5 的具体定位都用了 "I think"、"I know ... was at some point" 这类不确定措辞，转写含糊处宁可保留这种不确定，不要写成确凿现状；同样，MLA 与 FP8 / FP4 只是他举的例子，没有给出压缩比或精度损失的数值，不应臆测。

## 问答（四）：Mega Kernel 中的多 GPU 通信与收尾 [69:59 - 71:40]

最后一个问题（69:59–70:26）问得有点绕，演讲者把它复述清楚："how does mega kernel work when you have multiple GPUs communication in the loop"（当循环里存在多 GPU 通信时，Megakernel 怎么工作）。这正好把本章与 63:28 那个关于 batch size 与工程代价的问答接上：前面说 Megakernel 要为具体形状特化，这里进一步问跨设备通信这种更「全局」的操作能不能也塞进同一个 kernel。他回答团队有 "very early preliminary work about this"（非常早期的初步工作）。

关键结论是："you can also fuse the nickel calls into the mega kernel if you set it up correctly"（如果配置得当，可以把 NCCL 调用也融合进 Megakernel）。转写记作 nickel calls，结合 "multiple GPUs communication" 的语境即 NCCL（Nvidia 集合通信库）调用。但紧接着他给了两个保留：一是 "we haven't found a really great killer use case for that yet"（还没找到特别杀手级的用例），二是 "sometimes you're just bound by the latency of the nickel call itself"（有时瓶颈就是 NCCL 调用本身的延迟）。也就是说，融合能去掉调用开销，却消不掉底层通信延迟，收益存在上限。

把融合的层次记成一段示意（仅概括口述，不是真实代码）：

```text
mega kernel:
    compute ...                 # 本层算子融合
    fuse(NCCL call)             # 若设置正确，把集合通信也纳入同一 kernel
    bound by: NCCL latency      # 仍可能受通信延迟本身限制
```

他举的现成例子是 DeepSeek 新一代模型发布时，其团队曾为 mixture of experts inference layer（MoE 推理层）放出一个 Megakernel，"where they actually did fuse some of those communication"（确实融合了其中一部分通信），并且这个 kernel 可以只针对那一层来跑。这里要注意转写里的型号读音含糊（"Deepseek 4"、"Deepseek foes"），宁可不写具体版本号，只保留「某代 DeepSeek 模型发布时」这一层信息，避免编造。

由此他给出对趋势的判断：随着这类模型变多，"you'll have a little mega kernel for a part of the computation"，即是为计算的一部分准备一个小 Megakernel，"but you won't necessarily have a mega kernel for the whole model"（但未必会有覆盖整个模型的大 Megakernel）。除非 —— 他特意重复了下午早些时候那个说法 —— "unless of course you pay the blood, sweat, and tears price and then really get the whole thing going"（除非你付出血汗泪的代价，把整个模型真跑起来）。这与 63:28 的估算形成闭环：局部融合是现实的，全模型融合受制于人力成本，因此工程上会按层、按热点逐步切分，而不是一步到位。易错点是把「能融合 NCCL」理解成「分布式推理问题已解决」：演讲者明确说还没有 killer use case，且延迟可能被通信本身卡住，这仍是开放问题。

71:28 主持人说 "I think that's all the time we have. Let's thank Dan again"，71:33 演讲者回 "Thanks so much for having me"，全片结束。71:15 的截图正是片尾问答现场：演讲者站在讲台后正对镜头做手势，背景是白板与侧边屏幕，画面右下角有水印，没有任何幻灯片文字 —— 它与前面几张停留在 Thank you! 页的现场照一起，标记出讲座从内容页切回纯口头问答、再到致谢收尾的过程。作为收尾需要提醒本节的定位：整场问答都是对已展示结果的补充与边界说明，没有引入新的实验数据；所有关于预训练模型循环、LPU 常驻权重、NV FP4 / MX FP4、MLA 压缩、NCCL 融合的说法，都停留在「初步工作、设想或个例」的层面，转写含糊处均保留原样，不宜当作已验证的结论。

![片尾问答现场：演讲者致谢前回答最后一个问题](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0060_01-11-15.jpg)

## 要点回顾

### 从训练侧到推理侧：规模驱动与全栈纲领

- Dan Fu 把视角从 training 翻到 serving / inference，用 `Inference: Turning Electricity into Intelligence` 概括「electricity → tokens → intelligence」，中心判断是 inference is really the piece。
- 参数从 2018 年约 `10^8`（1 亿）涨到今天开源万亿级、前沿 5–10 万亿；曲线写作 `100M (2018) → 500B (2022)`，逐点标注 ELMo 94M、GPT-3 175B、Megatron-Turing NLG 530B。
- 他用 1902 年曼哈顿 130,000 匹工作马到 1912 年汽车数量超过马匹的类比讲「转变比想象更快」，并提醒幻灯片约两年前做 job talk 时所制、GPT-5.5 与 Cursor 等举例偏旧，不宜当成课程结论。
- `GPUs Are The New Oil` 对应 Stargate `$500B` 投资与 HUMAIN、NVIDIA 合作；模型被抽象成 `DAG of operations`，须靠推理引擎 + GPU kernel 才能映射到硬件。全场唯一 takeaway 是 `Understanding inference and GPU kernels enables full-stack innovation in ML algorithms`。

### 一个 token 的生命周期、负载形状与 SLA

- 组件地图主线 `Request In → Scheduling → KV Cache → Execution → Parallelism → Tokens Out`，小字含 continuous batching、chunked prefill、radix tree、prefix sharing、CUDA graphs、TP、wide EP、attention DP。
- prefill 是 `10,000 tokens in → 1 token out`、compute-bound、接近训练前向但不跑反向；decode 是 `one token per step`、memory-bandwidth-bound、每步都要重载权重，流程图收束为 `the engine is a loop: schedule → execute → sample → repeat`。
- SLA 目标为 `TTFT < 500 ms`（prefill 速度）、`TBT < 30 ms`（decode 速度）与 target QPS per GPU；分位值如 `p50 512/256 60%`、`turns per session p50 7`、`time between turns p50 10s`，结论是 `SLA depends on workload`。
- continuous batching 用 Step 1–6 与 Req A/B/C/D 演示长短请求共存，新请求可 mid-flight 加入，但须过 `capacity check: enough KV blocks?` 与 `microbatch check: under max tokens?`，并靠 `chunked prefill` 切块与 decode 交错。

### KV cache 复用、并行切分与两阶段解耦

- 基数树前缀共享按 `traverse → match hashes → reuse`，`□ = 32 tokens`、`shape [layers, 32, hidden_dim]`，`block_hash = f(content, parent)` 把父节点纳入哈希，避免不同前缀下相同内容被误复用。
- 1 trillion 参数模型约 2 TB 权重、280 GB GPU 单卡装不下：tensor parallelism 把 `W = [W_1|W_2|W_3|W_4]` 切到 4 卡、各卡输出沿列拼接为 `XW = [XW_1|...|XW_4]`（此步不通信），只在块末按行切分后做一次 `all_reduce`；expert parallelism 靠 `LEARNED ROUTER` 与 `y = Σ g_i(x) E_i(x)` 把完整专家放各卡。
- prefill/decode 解耦由瓶颈差异驱动：`T_prefill ≳ FLOPs(n,P)/peak FLOPS` 对 `T_decode ≳ 2P / BW_HBM`，prefill worker 用 `TP4`、decode 用 `wide EP / attention DP`，中间靠 `RDMA fetch` 与 `poll: KV received?` 交接 KV。
- KV cache 分三级 `GPU HBM (hot) / pinned CPU (warm) / NVMe (cold)`，配 `evict (LRU)` 与 `prefetch ahead`，`T_eff = h·t_HBM + (1-h)(t_fetch + t_prefill)`，目标是 `KV cache is already in GPU memory by the time the request runs`。

### 规模化故障、NVL72 容错与缓存感知路由

- 每天数万亿 token 下会出现 `0.001%` 级罕见 bug：NaN 经 softmax 污染整行导致「hi hi hi」循环、tool call 改坏后陷入 doom loop 让 completion length 暴涨到数万 token、off-by-one 读入未初始化显存解出随机中文，题眼是 `production observability catches what unit tests miss`。
- NVL72 把 72 张 GPU 高速互连、模型切到 64 卡，`32+ GPUs` 时故障是常态，容错靠 cancel/re-prefill、redistribute orphaned experts + hot spare 与 drain in-flight requests；`1M+ context` 用 context parallelism 把 `L=10^6` 均分 6 卡（`L/N ≈ 1.67×10^5`），并处理 `cross-partition attention`。
- CPD（cache aware prefill decode disaggregation）在路由层约两行代码：假设一段对话 10 轮则 `p_cold = 1/10 = 10%`，`if cache_hit_rate(req) < threshold` 把低命中冷请求送 pre-prefill nodes，其余 warm 请求去另一组 prefill 节点，配 `Distributed KV cache`（Store 1–4）与 `High-speed RDMA` 异步读写。
- 结果图 `CPD: Up to 40% faster long-context LLM serving` 用 QPS 对比：`2P1D vs. CPD 1D` 中 Baseline 约 0.75、CPD 约 1.125（40% faster），`2P2D vs. CPD 2D` 约 1.125 对 1.5（35% faster），可见 40% 只是最好配置的上限、并非恒定收益。

### Megakernel 与 ThunderKittens：填平解码空闲

- decode 把并行机变成 "glorified memory loader"，逐算子写法 `T_block = Σ(t_launch + t_i)`、融合后 `T_block^mega ≈ Σ t_i + t_launch`；H100 有 132 个 SM、B200 约 148 个，downtime 来自 kernel launch、teardown、tail effects 与 inter-kernel gaps。
- Megakernel（ThunderGQA）把 GPU 当分布式系统调度，四条要点是 `Single Megakernel ... end-to-end`、`HBM as backing semaphore`、`Eliminates kernel launch overheads` 与 `1.3-1.7x faster`（等价 30–70% speedup，仅限 attention kernel）；`Llama-1B — all in one kernel` 指的是一层。
- 细粒度重叠让 `QKV + RoPE` 未算完就加载 KV cache、O projection 在 attention 结束前加载权重；Legend 把 `LayerNorm_QKV_MatVecRopeAppend (1)` 到 `DownProjResidual (6)` 等七类操作塞进同一 kernel。
- ThunderKittens 比 Triton 更低层，存储金字塔为 `Reg 64KB/130TB/s`、`Shared 227KB/33TB/s`、`L2 50MB/12TB/s`、`HBM 80GB/3TB/s`，配 `rt<M,N>`、`st<M,N>` 与 `tma::load_async`；H100 上达 `72%` 带宽利用率、近 speed-of-light，但数字已偏旧。

### Parcae：循环 transformer 的稳定化与缩放定律

- loop transformers 把 block 跑进循环、参数不变却 `Increase FLOPs, keep parameters constant` 与 `Higher expressivity`，用 SSM 理论 stabilize；主链是 `h_0 → … → h_T → C → Models`，旁证论文称可用 `less than 10% of the parameter count`。
- 训练极不稳，nine times out of 10 不收敛并冒 loss spikes；团队只看残差 `h_{t+1} ≈ h_t`，写出 `h_{t+1} = A h_t + B e + R(h_t, e)`，丢掉非线性得闭式解 `h_{t+1} = A^t h_1 + (Σ A^n) B e`，命运由谱半径 `ρ(A)` 主导（`A=2`、`t=16` → `2^16`）。
- Parcae 把 `A = -diag(λ_i)` 使 `ρ(A) < 1`、给只应用一次的 `B` 加线性 norm；`LR = 6e-4` 下 Baseline 的 state norm 飙到 `10^19` 而 Parcae 稳在 `10^9`，loss 平滑下降，质量优于 recurrent depth models 与强 transformer 基线。
- 缩放定律从 `L̂(N,D) = E + X·N^{-x} + Y·D^{-y}` 出发补上 recurrence 第三个量：ISO-param / ISO-FLOP 曲线（140M / 370M、Recurrence 2/4/6/8/12、1e18~128e18 FLOPs）给出 `More data -> more recurrence is optimal!` 与 `Down and to the right: Scale both`，固定算力对照满足 `L_blue < L_orange`。

### 全栈创新收束与问答中的工程边界

- 结尾把 takeaway 拆成三类例子：新 routing algorithm 服务更多流量、新 kernels 让局部 way faster、新 architectures 以更少参数减少通信，分别对应 CPD、Megakernel 与 Parcae。
- Megakernel 的代价是 "people's blood sweat and tears"：一位工程师一年只够覆盖 one hardware、two or three models、batch sizes 1–16，`batch size = 17` 就得 start over，Together AI 正尝试用编译器自动化这部分流程。
- 面向硬件先受 `memory first` 约束（如 Cerebras wafer），格式随芯片变：Nvidia 用专有 `NV FP4`、AMD 用 `MX FP4`；workload 上 agentic 要保持 KV cache hot、批处理偏向 bidirectional attention（BERT）、T5 居中，DeepSeek MLA 是 KV cache 激进压缩的例子。
- compute-optimal 只在 given a flop budget 下成立、被演讲者称有点 contrived；多 GPU 通信虽可把 `NCCL` 调用 fuse 进 Megakernel，但仍属 very early preliminary work、尚无 killer use case 且可能 bound by NCCL latency。

## 视频来源

- 原始链接：https://www.youtube.com/watch?v=9EEm4iMAF5s
- 课程：Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 18

---

## 对照参考笔记补遗（Spring 2025 版）

> 编者注：本节依据 2025 版参考笔记整理，其中的「更正」小节记录的是核对时发现的问题，对应正文已在本版订正；补遗内容不代表 2026 课程视频原文。


本节来自 Spring 2025 版参考笔记（Lecture 10: Inference 及两份附录），供对照补充可迁移的通用方法、公式与易错点，**不代表 2026 视频原文**；本讲在 2025 版无直接对应讲次，故不与正文逐条对齐。

### 算术强度：把「计算密集 / 带宽密集」变成可核算的判据

本讲把 prefill 判为 compute-bound、decode 判为 memory-bandwidth-bound，2025 版给出可核算的判据：arithmetic intensity（算术强度）$\mathrm{AI}=\text{FLOPs}/\text{bytes transferred}$，即每搬一字节做多少浮点运算；对照量 accelerator intensity（加速器强度）是峰值算力除以内存带宽。H100 的 FP16 峰值约 $989\times10^{12}$ FLOPS、HBM 带宽约 $3.35\times10^{12}$ 字节每秒，相除约 **295**：$\mathrm{AI}>295$ 为 compute-limited，$\mathrm{AI}<295$ 为 memory-limited。前向 FLOPs 粗算 $6(BT)(\text{num\_params}+O(T))$。

落到矩阵乘上最清楚：$X$ 为 $B\times D$、$W$ 为 $D\times F$，BF16 每元素两字节，读 $X$ 花 $2BD$、读 $W$ 花 $2DF$、写 $Y$ 花 $2BF$，计算量 $2BDF$，故 $\mathrm{AI}=\dfrac{2BDF}{2(BD+DF+BF)}\approx B$（$D,F\gg B$）。H100 上须 $B>295$ 才算力受限；$B=1$ 退化为矩阵—向量乘、$\mathrm{AI}=1$，为做 $2DF$ 次运算却要读进整个 $D\times F$ 权重矩阵，正是逐 token 生成的画像。naive sampling 每步重喂整段历史，单次前向 $O(T^2)$、生成 $T$ 个 token 累计 $O(T^3)$ FLOPs，KV cache 消掉的正是前缀重算。

易错点：295 是特定硬件与精度（FP16/BF16）的比值，换卡换格式要重算；分母须把读写算全，只算权重会高估强度；"$B>295$" 只适用于矩阵乘，注意力里 $B$ 会被约掉。这正把本讲「decode 像 glorified memory loader」的说法定量化。

### 生成阶段 MLP 与注意力的不对称：为什么加大 batch 救不了 decode

把核算落到两类层，才明白 decode 天生带宽受限是结构性的。MLP 层 FLOPs 约 $6BTD$、字节约 $4BTD+6DF$，$\mathrm{AI}\approx BT$：prefill 时 $T=S$ 得 $B\cdot S$，靠长提示或大 batch 易过 295；generation 时 $T=1$ 退化为 $B$。关键是每条序列都命中同一份 MLP 权重（$W_{up},W_{gate},W_{down}$），只读一次被 batch 摊薄，故增大并发能把 MLP 推向算力受限。

注意力层相反。FlashAttention 记号下 FLOPs $4BSTD$、字节 $4BSD+4BTD$，化简 $\mathrm{AI}=\dfrac{ST}{S+T}$：prefill $T=S$ 得 $S/2$，generation $T=1$ 得 $\dfrac{S}{S+1}\approx1$。附录的严格推导 $\mathrm{AI}_{\text{attn}}=\dfrac{2BSD}{2BD(S+1)}=\dfrac{S}{S+1}$ 中 $B$ 完全约掉：每条序列有私有 KV cache，batch 加倍时 FLOPs 与要读的 KV 字节同步加倍，比值不变。故 generation 注意力被钉在约 1、远低于 295，加并发也救不了；这正是本讲把重心放在「减小 KV cache 字节数」而非堆算力的原因。

易错点：别把「MLP 的 $\mathrm{AI}\approx B$」当成整个 decode 的强度，模型由最弱的注意力环节支配；$T$ 在 prefill 等于 $S$、generation 恒为 1，代错会得相反结论。也正因 batching 对注意力无效，2025 版才把降低 KV cache 列为整讲主线。

### KV cache 的显存账本、压缩谱系与「换掉 cache」的架构

本讲用「32 token 一块」管理前缀复用，2025 版给出按维度算总量的通式：MHA 下 $\text{Bytes}=L\cdot2\cdot B\cdot N\cdot d\cdot2$（$L$ 层、$N$ 上下文长、$d$ 模型维，两个 2 为 K/V 两份与 BF16 两字节）；L10 等价写法 $\text{KV\_cache\_size}=S(K\cdot L\cdot H\cdot2\cdot2)$，$K$ 为 KV 头数、$H$ 为头维。量级：LLaMA-7B（$L=32,d=4096,h=32,B=1,N=4096$）单序列约 **2.15 GB**；MQA 压到 1 个 KV 头、为 MHA 的 $1/h$ 约 **67 MB**；GQA 为 $g/h$，Mistral-7B（$h=32,g=8$）= **1/4**。每生成一个 token 都要把整份 cache 从 HBM 读进片内，故压缩 cache 直接省带宽。

压缩谱系各有数字。GQA（Ainslie+ 2023，Llama 3 采用）在 Llama 2 13B 上把 MHA 的 79.7 GB / 2689 tok/s 变为 1:5 分组（$K=8$）的 **33.4 GB / 6416 tok/s**，batch 256 时 **130.6 GB / 16584 tok/s**，精度几乎不掉。MLA（DeepSeek-AI+ 2024）把 K、V 投到低维潜空间，DeepSeek v2 把 $N\cdot H=16384$ 压到 $C=512$，代价是与 RoPE 不兼容、需额外维度承载位置，实测优于 GQA。Local attention（Beltagy+ 2020 等）使 KV cache 与序列长度无关，Character.ai 每 6 层插 1 个 global layer；CLA（Brandon+ 2024）跨层共享 KV 改善精度—缓存帕累托前沿。

更激进的是用状态替代 cache：state-space models（SSM，状态空间模型）与线性注意力把随 $T$ 增长的 KV cache 换成 $O(1)$ 固定状态，谱系有 S4（Gu+ 2021，弱于 associative recall）、Mamba（Gu+ 2023，1B 追平 Transformer）、Jamba（Lieber+ 2024，1:7 交错 + 52B MoE）、BASED、Minimax-01（456B MoE），结论是线性 + 局部注意力仍需掺少量 full attention 才到 SOTA；文本 diffusion（Mercury Coder）并行生成再迭代细化。易错点：这些手段彼此正交（减头数、降维度、缩窗口、跨层共享、换状态），不能把 MQA 的 $1/h$ 套到 MLA 上；本讲「前缀共享」省的是重复计算 FLOPs，这里省的是单序列字节数，一个省算力一个省带宽；local attention 的「与长度无关」以丢失长程依赖为代价。

### 延迟—吞吐的定量权衡、连续批处理与分页共享

generation 受带宽支配，单请求可被解析封顶：$\text{latency}=\text{memory}/\text{memory\_bandwidth}$（每步读完全部参数与 KV cache），$\text{throughput}=B/\text{latency}$。Llama 2 13B on H100（$S=1024,D=5120,F=13824,N=40,K=40,H=128,L=40,v=32000$，带宽 $3.35\times10^{12}$）给出：$B=1$ 时 26.8 GB / **8.08 ms** / 124.6 tok/s，$B=64$ 时 79.7 GB / **23.79 ms**——延迟变差、吞吐变好。故小 batch 延迟低吞吐差、大 batch 吞吐高延迟差，且「高吞吐不必然低延迟」。TTFT 由 prefill 决定，故 prefill 用小 batch 压首 token、generation 用大 batch 提吞吐；本讲把 TTFT 与 TBT 拆成两个 SLA，根源即在此。

动态流量让静态 batching 失效：请求到达时刻不同、序列有共享前缀、长度不齐导致 padding 浪费。Orca 的 continuous batching（连续批处理）用 iteration-level scheduling（迭代级调度），每生成一个 token 交还控制权给调度器、新请求 mid-flight 插入；Selective Batching 对注意力逐序列算、非注意力拼成 $[3+9+5,H]$。PagedAttention（Kwon+ 2023，vLLM 核心）借鉴虚拟内存，把 KV cache 切成非连续固定块 + 页表，消除按最大长度预分配的内部碎片与段间外部碎片。

附录给出共享前缀的精确节省：beam search 或并行采样下同一 prompt 的 prefill 结果本应只存一份，naive 物理拷贝成 $N$ 份占 $M_{\text{naive}}=N(S_p+S_g)C_{\text{token}}$；PagedAttention 让 $N$ 个逻辑序列的页表指向同一组前缀块、引用计数置 $N$，只为新 token 追加私有块，故 $M_{\text{Paged}}=S_pC_{\text{token}}+NS_gC_{\text{token}}$，节省 $(N-1)S_pC_{\text{token}}$。易错点：这是 share-then-append（先共享后追加）而非 memcpy 式 copy-on-write，引用计数只挂在共享块上；长 system prompt 与高并发 $N$ 收益最大。它与本讲 radix tree 前缀共享、`capacity check` 是同一问题的两层——一个管「块够不够」、一个管「块归谁」。

### 投机解码的精确采样保证，与量化 / 剪枝这类「有损捷径」

本讲把 speculative decoding（投机解码）作为可选优化一笔带过，2025 版给出它为何无损。设目标分布 $q$、draft（草稿）分布 $p$，对 $p$ 猜出的候选 $x'$ 以 $\min\!\left(1,\dfrac{q(x'\mid H)}{p(x'\mid H)}\right)$ 接受；拒绝则从修正分布 $q_{\text{corr}}(x)=\dfrac{(q(x)-p(x))^{+}}{\sum_{x\in V}(q(x)-p(x))^{+}}$ 采样并停止。两条互斥路径相加正是 $\min(p,q)+(q-p)^{+}=q$，无论 $q\le p$ 还是 $q>p$ 都还原 $q$，故输出是目标模型的**精确**样本；加速来自「一次并行验证 $K$ 个 token 比逐 token 生成便宜」这一 checking 与 generation 的不对称，而非近似。典型 70B 目标配 8B 草稿，草稿越接近越好（蒸馏），扩展有 Medusa（并行多 token）、EAGLE（取目标高层特征）。

相对的是有损捷径。数值格式 FP32（4 字节）、BF16（2 字节，推理默认）、FP8（1 字节）、INT8（1 字节，值域 $[-128,127]$）、INT4（0.5 字节，$[-8,7]$），降精度省内存与 FLOPs 但要盯精度；QAT（量化感知训练）难放大，PTQ（训练后量化）用样本定 scale 与 zero point。LLM.int8()（Dettmers+ 2022）把 outlier 抽离留 FP16、其余走 INT8，代价是比纯 FP16 慢 **15–23%**；Activation-aware（Lin+ 2023）按激活留 0.1–1% salient（显著）权重，FP16→INT3 得 **4 倍内存 / 3.2 倍加速**。剪枝蒸馏（Muralidharan+ 2024，Minitron）在小型校准集上识别 {layer, head, hidden dim} 重要性、删层后蒸馏修复，15B→8B 的 MMLU 几乎不掉、4B 才明显下降。

分类上的易错点最要紧：投机解码**无损**（有数学保证），量化、剪枝、蒸馏**有损**（要验精度），不能笼统都叫加速技巧。本讲「随机中文字符」一度被归咎于 quantization、最终查明是 kernel 的 off-by-one 读进未初始化显存，正说明降精度容易背锅——排障须先用监控区分「模型侧精度损失」与「系统侧内存越界」。另外 $\min(p,q)+(q-p)^{+}=q$ 依赖拒绝后按整个词表归一化，漏掉分母 $\sum(q-p)^{+}$ 就不再精确。

### 从 Chinchilla 到推理成本：6ND、IsoFLOP 与「过训练」的经济学

本讲第三部分把 recurrence 补成 $N,D$ 之外的第三个可缩放量，附录给出算力底座。Transformer 训练算力 $C\approx6ND$（每 token 前向 $2N$ + 反向 $4N$），固定 $C$ 时 $D=C/(6N)$ 是双曲线；IsoFLOP（等算力）沿它取若干 $(N,D)$ 各从头训练、loss 对 $N$ 呈 U 形，谷底即 $N_{opt}(C)$，对多预算最优点做 log-log 回归得 $N_{opt}\propto C^{a},D_{opt}\propto C^{b}$，Chinchilla 得 $a\approx b\approx0.5$（按 $\sqrt C$ 同步放大）。拟合最忌用单次长跑的中间 checkpoint 冒充独立短跑：cosine 学习率按 $T_{total}$ 塑形，中途截出的模型 LR 仍高、被欠训练，会高估 loss；WSD（Warm-up Stable Decay，梯形）schedule 是折中。

更贴本讲的是把推理成本写进总账：训练是一次性 $C=6ND$，生命周期服务 $K$ 个 token 的推理成本约 $\gamma KN$（$\gamma\approx2$，与 $N$ 成正比），故 $\mathrm{TCO}=6ND+\gamma KN$。改用更小的 $N'<N_{opt}$、靠更多数据 $D'$ 追平同一 loss，训练成本增为 $C'$；只要 $C'-C_{opt}<\gamma K(N_{opt}-N')$（推理终身节省超过训练增量），过训练小模型就更划算。数值例：$N_{opt}=70B,D_{opt}=1.4T$ 时 $C\propto98$，$N'=50B,D'=2.5T$ 时 $C'\propto125$，训练多花 27、推理节省正比于 $70-50=20$；当 $K$ 极大时不等式成立，这解释 Llama 3 训练 >15T tokens、远超 20 tokens/parameter 的 Chinchilla 比例。

与前后的关系：本讲问答里「compute-optimal 只在 given a flop budget 下成立、有点 contrived」与「参数更少 → 装更多 KV cache、do less communication」，正是把预算从训练 FLOPs 换成「训练 + 推理 TCO」后的推论；$6ND$ 也给本讲 ISO-param / ISO-FLOP 曲线补上「横轴移动代表算力守恒」的约束。易错点：$a\approx b\approx0.5$ 是特定数据架构拟合的指数、非物理常数；「过训练更小模型」也不能读成「循环一定更优」，它只说明 $K$ 很大时减小 $N$ 的长期收益可能压过训练期多花的算力。

### 更正

原笔记「万亿参数模型的切分」一节（`note.md` 第 161–163 行）与「要点回顾」第 717 行，把按列切分的权重写成 $W=[W_1\,\Vert\,W_2\,\Vert\,W_3\,\Vert\,W_4]$ 后直接给出 $Y=\text{AllReduce}\!\left(\sum_{i=1}^{4}XW_i\right)$、图注标 `allreduce activations`，应改为：按列切分（column-parallel，列并行）时各卡 $XW_i$ 是互不重叠的列块，正确合并是沿列**拼接**为 $XW=[XW_1\,|\,XW_2\,|\,XW_3\,|\,XW_4]$，此步不通信；Megatron 式张量并行把「列切分 $W_{up}$」与随后「按行切分 $W_{down}$」配对，中间激活 $G_i=\text{GELU}(Z_i)$ 保持分片，只在块末做一次 $Y=\text{all\_reduce}\!\left(\sum_{i=1}^{T}G_iW_{down,i}\right)$。若保留 $\sum$ 形式，须把前提改写成按行切分 $W=[W_1;\dots;W_4]$。（依据：A1《Parallelism: DP / TP / PP / ZeRO / FSDP》的 column-parallel / row-parallel 推导；L10 的 Transformer 符号表亦把权重定义为 $D\times F$ 列块分片。）原笔记「张量并行每层一次 allreduce」的定性判断本身正确、无需改动，错只在把这次 allreduce 安到了列切分第一步，使公式与前提在维度上不自洽。
