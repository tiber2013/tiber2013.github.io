---
title: "第 13 讲 · 数据 I：来源、抓取与合规"
description: "数据是语言模型中最关键、也最不透明的一环：开放权重模型会公开架构和训练流程，却几乎从不交代数据来源，原因既有竞争考量，也有版权责任。讲者从\"数据从哪来\"讲起——互联网是一堆活服务器，受动态内容、登录墙、robots.tx…"
---

# 第 13 讲 · 数据 I：来源、抓取与合规

> **Stanford CS336 语言模型从零构建 · L13** ｜ 讲者 Stanford Online ｜ 82 分钟 ｜ [原视频](https://www.youtube.com/watch?v=-qm0ln33G24)

> **v2 详写版** · 按 20 分钟分段详写，配图覆盖全片；文末附「对照参考笔记补遗（Spring 2025 版）」，记录与 2025 版讲义笔记的核对结果。

## 核心结论

这一讲回答评估之后唯一剩下的问题：究竟该在什么数据上训练语言模型。讲者的核心立场是数据是训练时最需要把对的事，厂商行为比宣传诚实——Llama 3 对架构与流程全透明，却对数据源守口如瓶，只说"从多种来源训练、做了些处理"，背后是竞争与版权责任，且数据是随人力而非算力扩展的长尾问题。讲座先纠正"在整个互联网上训练"的口号：网页是活服务器，要靠爬虫从种子集发现下载，又被动态内容、登录墙、robots.txt 约定、服务条款与许可层层削减；再用 1976 年版权法说明网上几乎所有内容自动受版权保护，合法路径只有许可与第 107 条合理使用，三起 2025 年判决显示训练在个案算合理使用、盗版取副本仍违法且结论很窄。随后按年份复盘从 BERT 到 Nemotron-CC 的过滤演进，由 Reddit 外链、Wikipedia 打分、手工规则走到质量分类器与合成数据，始终在"规则派还是分类器派""留多少 token、保多少质量"之间权衡。结尾补上代码数据与只用许可明确数据的 Common Pile，并坦承数据处理多凭规则和阈值、缺乏第一性原理，留下"从数百 T 网页里筛掉 98%"这一开放问题。

## 数据为何最难也最重要：披露、保密与长尾瓶颈 [00:00 - 02:18]

讲者在本段一开始就交代了整门课此处的坐标：上一讲已经讲过评估（evaluation），也就是"什么算一个好模型"，而更早的部分已经解决了"给定数据如何训练模型"。于是接下来本周两讲的唯一问题就收敛成一句话——我们究竟应该在什么数据上训练（what data should we train on）。在这个位置上，讲者抛出了一个贯穿全段的主张：数据是训练语言模型时最需要做对的事（Data is the most important thing to get right in training language models）。这不是一个可以证明的定理，而是一个带有强烈经验判断的立场，理解它需要先看厂商的实际行为，而不是听他们的宣传口径。

支撑这个主张的第一个证据来自披露的不对称。讲者以 Llama 3 论文为例（幻灯片上标注的引用是 `[Grattafiori+ 2024]`）：因为它是开放权重模型（open-weight model），所以对架构（architecture）完全透明，甚至连训练流程（training procedures）都讲了；唯独对数据几乎不提供任何信息，只留下一句"我们从多种数据源训练"（we train from a variety of data sources）、再含糊地说"做了一些处理"（and do some stuff）。幻灯片上半部还留着上一讲灰显的总结（Summary），其中几条恰好点题："Data does not fall from the sky. You have to work to get it."、"Live service → raw data → processed data (transformation, filtering, deduplication)"、"Data is the key ingredient that differentiates language models"、"Legal and ethical issues (e.g., copyright and privacy)"、"Much of this pipeline is heuristic, many opportunities to improve!"——这几行是图上独有、转写没有念到的，它们把本段后面要展开的"从活服务到处理后数据"的链路、以及法律伦理问题提前摆了出来。

幻灯片中部是一段带行号的伪代码，行号从 48 排到 53，高亮的是第 53 行：

```python
def motivation():                                          # L48
    # Data is the most important thing to get right in
    # training language models.                            # L49
    # One justification: let's see what companies disclose.
    # Open-weight models (e.g., Llama 3 [Grattafiori+ 2024])
    # have full transparency into architecture            # L52
    ...and even training procedures                        # L53（高亮）
    ...but basically no information on data.
```

紧随其下还引了 Llama 3 论文第 3.1 节 Pre-Training Data 的原文（在下一张图上可以读到完整段落）："We create our dataset for language model pre-training from a variety of data sources containing knowledge until the end of 2023. We apply several de-duplication methods and data cleaning mechanisms on each data source to obtain high-quality tokens. We remove domains that contain large amounts of personally identifiable information (PII), and domains with known adult content." 这段引文很值得玩味：它提到了去重（de-duplication）、清洗（data cleaning）、移除含大量个人身份信息（PII）的域名和含成人内容的域名，还给出了知识截止日期 "until the end of 2023"——也就是说，论文并非完全不谈处理，只是不谈"数据源到底是什么"。讲者随后把保密理由归纳为两条，幻灯片写作 "Reasons for secrecy: 1. Competitive dynamics 2. Copyright liability"：其一是竞争动态，数据是各家竞争性的"秘密配方"（competitive secret sauce），不想让对手知道自己在做什么；其二是版权责任（copyright liability），公开说自己训了哪类数据可能直接招来诉讼。

再往后，讲者把视野拉长到整个机器学习史：在基础模型（foundation models）出现之前，"做数据"几乎就等于为监督学习（supervised learning）大量标注有标签数据（heavy annotation of labeled data）；如今预训练阶段的人工标注确实变少了，但整理（curation）与清洗（cleaning）的工作量依然巨大，所以"数据一直是瓶颈、将来也永远是瓶颈"这个判断并没有变。原因被归结为两点：数据本质上是一个长尾问题（long-tail problem），而且它随人力投入扩展（scales with human effort）。幻灯片第 63 行高亮的那句写得最直白："Data is fundamentally a long-tail problem, scales with human effort (unlike architectures, systems)." 讲者的解释是：能钻研架构和系统的人终究有限，可数据的标注、清理、筛选很容易并行投人，尤其当你在训一个什么都要会的基础模型时更是如此——这正解释了为什么各家的数据团队和模型开发团队规模都相当大（"these data teams and these model developers are actually quite big"）。

与前后内容的关系上，本章既是上一讲"评估"之后的自然转向，也为后面所有内容设定了动机：正因为数据既不透明、又受竞争与版权双重压力，讲者才要在下一章先把训练阶段拆开，再往后从"数据从哪来"一路问到版权。这里有个容易踩的坑：不要把"开放权重"误读成"数据开放"，Llama 3 的透明度只覆盖架构与流程；也不要把"数据是长期瓶颈"理解成算力或算法进步能自动解决的问题，长尾意味着它随人力而非随算力缓解。另外，讲者那句 "most important thing to get right" 是价值判断而非可量化结论，记录时应保留其断言性质，不要替它补上一个并不存在的证明。

![Llama 3 披露与 def motivation() 伪代码、上一讲 Summary 及论文 3.1 节引文](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0001_00-00-38.jpg)

## 训练三阶段与术语：base / instruct 模型 [02:18 - 04:36]

讲者接着说明数据并非一次性灌进模型，而是在流水线（pipeline）的不同阶段进入；本讲聚焦的主要是预训练（pre-training）。幻灯片用带行号的伪代码给出了最基础的三段模板，以下为第 67–73 行的原文转录：

```text
1. Pre-training: train on raw text
   (e.g., documents from the web)                         # L67
2. Mid-training: train more on high quality data to
   enhance capabilities                                  # L68
3. Post-training: train on chat transcripts or
   reinforcement learning                                # L69
In practice, the lines are blurry and there could be
more stages                                              # L70
...but the basic trend is throughout training, we go from # L71
large amounts of lower quality data to                    # L72
small amounts of high quality data.                       # L73
```

口头补充把这三段讲得更具体：预训练吃的是来自网页的原始文本文档；中期训练（mid-training）换成更高质量的数据，用来增强特定能力并给出长上下文（long context）；后训练（post-training）则吃聊天记录（chat transcripts），如果做强化学习还会用到环境（environments），整体越来越任务特定（task-specific）。紧接着第 70–73 行写明 "In practice, the lines are blurry and there could be more stages"，以及那句最重要的趋势判断——整个训练过程是从大量低质量数据走向少量高质量数据。把这个趋势写成记号，可以形式化为数据规模随阶段递减、质量随阶段递增：

$$ |D_{\text{pre}}| \gg |D_{\text{mid}}| \gg |D_{\text{post}}|,\qquad \text{quality}(D_{\text{pre}}) < \text{quality}(D_{\text{mid}}) < \text{quality}(D_{\text{post}}) $$

这里 $D_{\text{pre}}$、$D_{\text{mid}}$、$D_{\text{post}}$ 分别指三个阶段使用的训练集，$|\cdot|$ 表示 token 规模；它只是对上述口头趋势的记号化，并非幻灯片上给出的公式。这个"量减质增"的梯度是理解后面所有数据集演进的骨架：从海量粗糙网页一路收窄到指令、数学、代码与安全数据。

由此产生一组术语，幻灯片第 74 行高亮 "Terminology:"，下面写着 "Base model: after pre-training + mid-training" 与 "Instruct/chat model: after post-training"，也就是 base model 指预训练加中期训练之后的产物，instruct model / chat model 指后训练之后的产物。但讲者立刻提醒界限正在失效："the lines are becoming blurry enough that what is a base model anymore is really kind of unclear"，近期最大的模型干脆不再发布中间检查点（intermediate checkpoints），只给最终版本；幻灯片把这层意思写得比口述更精确——"(Increasingly, base models are not released - e.g., Qwen3.5-397B-A17B is an instruct model.)"，而转写里只含糊说成 "Qwen 3.5 397B ... there's no base model"。相对地，像 AI2 的 Olmo 这类完全开放的模型可以看到全过程（"for models that are open source, like Olmo from AI2, you can see everything that's happening"），幻灯片也以 "Example (OLMo from AI2) [Team OLMo 2024]" 引出一张真实数据表。

这张表标题是 "Pretraining + OLMo 2 1124 Mix"，列为 `Source | Type | Tokens | Words | Bytes | Docs`，可见两行：`DCLM-Baseline | Web pages | 3.71T | 3.32T | 21.32T | 2.95B`，以及 `StarCoder | Code | 83.0B | 70.0B | 459B | 78.7M`（旁注 "filtered version from OLMoE Mix"）。这些数字是图上独有、转写完全没念到的：网页来源 DCLM-Baseline 有 3.71 万亿 token、3.32 万亿词、21.32 万亿字节、29.5 亿篇文档；代码来源 StarCoder 是 830 亿 token、700 亿词、4590 亿字节、7870 万篇文档。它顺带纠正一个常见混淆：字节数 21.32T 远大于 token 数 3.71T，说明 token 不是字节，二者比例取决于分词器与文本本身。讲者口头还按阶段列了典型来源：预训练有网页、学术论文（academic papers）、数学页面与证明（math pages, proofs）；中期训练有更高质量的网页数据、指令数据（instruction data）与大量合成数据（synthetic data）；后训练有聊天日志、更多数学与推理、代码，以及安全（safety）数据。

另一张 04:29 的图则给出中后期混合数据的具体配方表，同样是转写没念、只出现在画面上的内容：可见 `Numinamath-TIR | 64,312 | 64,312 | 8,677 | Beeching et al. (2024)`，编码（Coding）类有 `Tülu 3 Persona Python | 34,999 | 34,999` 与 `Evol CodeAlpaca | 107,276 | 107,276 | 14,200 | Luo et al. (2023)`，安全与非顺从（Safety & Non-Compliance）类有 `Tülu 3 CoCoNot | 10,983`、`Tülu 3 WildJailbreak | 50,000 | 26,356`（Jiang et al. 2024）、`Tülu 3 WildGuardMix | 50,000 | 26,356`（Han et al. 2024），多语言（Multilingual）有 `Aya | 202,285 | 100,000 | 32,210`（Singh et al. 2024b），精确指令跟随（Precise IF）有 `Tülu 3 Persona IF | 29,980` 与 `Tülu 3 IF-augmented | 65,530`，末行合计 `Total | 23,327,961 | 939,344 | 425,145`。这些条目把"中期训练放合成数据、后训练放安全与指令数据"从口号落成了带引用文献的具体数据集清单。第 87 行高亮的提问 "What are these datasets? How are they chosen and processed?" 正是整门课后面要回答的主问题。

与前后衔接上，本章把第 1 章的"数据最重要"拆成了可操作的阶段划分，又用 "What are these datasets?" 把问题直接递给下一章"原始来源从哪来"。易错处有三：一是 base / instruct 是约定俗成而非架构上的硬分界，遇到只发 instruct 模型的厂商就失效；二是"三阶段"只是模板，实践中界限模糊且阶段数可能更多，讲者明说 "there could be more stages"；三是同一来源在不同 mix 里会被重新过滤（表里 "filtered version from OLMoE Mix" 即是提示），因此看到某个数据集名字不等于看到了它的最终形态。

![三阶段伪代码、Terminology 与 OLMo 2 1124 Mix 预训练数据表](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0003_00-03-12.jpg)

## "整个互联网"不成立：活服务器、爬虫与深网 [04:36 - 07:34]

呼应本课"从零开始"（start from scratch）的风格，讲者把问题重置为最朴素的一句：数据到底从哪来（where does data come from）？走廊里常听到的说法是"语言模型是在整个互联网上训练的"（language models are trained on the entire Internet），讲者说这句话 "doesn't really quite type check"——类型上对不上：若真要成立，模型得像一个在互联网上行动的强化学习智能体（RL agent）那样去浏览、操作，而预训练根本不是这种工作方式。稍准确一点的说法是"在公开的万维网上训练"（trained on the public World Wide Web），但讲者随即补一句 "this is not quite accurate either"。幻灯片 `def raw_sources():` 的第 90–93 行把这三步措辞原样写下，其中第 92 行把 "Internet" 用删除线划掉、改写成 "public (world wide) web"，这个视觉细节本身就是一次概念纠偏。

```python
def raw_sources():                                         # L90
    # One might often hear: language models are trained
    # on the entire Internet.                             # L91
    # Slightly more accurately: (Internet 划掉)
    # public (world wide) web.                           # L92
    # But this is not quite right either...               # L93
```

接下来是工程事实：Web 并不是一个可以整体下载的文件，而是一堆真实存在于世界中、可以连上去的活服务器（live servers）；你发一个请求、拿回一个响应（"you can basically send a request and get back a response"）。幻灯片第 95–96 行给出的最小可复现命令是 `$ curl https://cs336.stanford.edu/`，用一次 `curl` 请求说明"访问一个页面"到底意味着什么：

```bash
$ curl https://cs336.stanford.edu/
```

第 98 行随即下了结论 "You can't train on live servers."——你没法直接在这些活服务器上训练（除非你训的是 RL agent）。于是需要爬虫（A crawler）。讲者特意说"有人建了爬虫，不一定是你"（"someone builds a crawler, not necessarily you"），这个细节常被忽略：多数模型团队并不掌控最上游的抓取环节。第 99–101 行给出爬虫的两件事，第 100 行高亮："Discovers webpages (starting from a seed set)" 与 "Downloads the discovered webpages"，即从种子集合（seed set）出发发现网页，再把发现的网页下载下来。可形式化地把它想成在一个有向图上做遍历：设已发现 URL 队列为 $Q$、种子集为 $S \subseteq Q$，每轮弹出一个 $u \in Q$、取回页面 $p(u)$，再把 $p(u)$ 里的超链接并入 $Q$；这个记号只是对"发现—下载"循环的抽象，不是画面上的公式。第 103 行紧接着泼冷水："However, you can't download and train on all the webpages."

为什么拿不到全部？第一层是动态内容（Dynamic content）。第 105–109 行写得很清楚："Many sites these days are apps"、"URL doesn't change"、"Need to click buttons and submit forms to access content"，例子是 `Discord, wandb`。也就是说，很多站点本质是应用，URL 甚至不再是内容的完整规格说明，你得点按钮、提交表单才能看到内容；讲者把这类无法靠顺着超链接抓取的内容称作深网（deep web），它不属于"下载网页—跟随链接"这种传统爬取模型。以 Discord 为例，并不存在"爬一遍 Discord 就把里面的内容全拿出来"这回事。

第二层是认证（Authentication）。第 111–113 行写着 "Sometimes need login with an account (and pay usually)"，以及被高亮的 "Example: Facebook, X, LinkedIn, NYTimes (huge content behind walled gardens)"。Facebook、X、LinkedIn、纽约时报背后锁着海量内容，但通常需要账号登录、往往还要付费，讲者称其为"围墙花园"（walled gardens）：这些内容技术上确实在互联网或万维网上，但你没法建个爬虫去抓——除非你本身就是 Facebook，"you don't need to crawl Facebook. You have the data"；或者你是 xAI 因而能直接在 X 上训练，"if you're anyone else, you can't actually access this content"。第 113 行高亮处正是讲者讲到"大量内容被锁在围墙花园里"的那一刻。

与相邻章节的关系很直接：第 2 章末尾问"这些数据集是什么、怎么选、怎么处理"，本章把问题倒推到最上游——连"网页"这个原料本身都不像口号里那样唾手可得；下一章则继续叠加技术与法律限制，说明即便绕过了动态内容与登录墙，仍有更多约束。易错点也集中在这里：把 "public web" 等同于"可下载的全部网页"是错的；把 URL 当作内容的完整描述是错的，因为应用状态下同一 URL 可能对应不同内容；另外要记住爬虫的两步（发现、下载）以及"爬虫未必是你建的"这一隐含前提——它意味着数据来源的第一手控制权往往不在模型开发者手里，这也为后面公共抓取结果的重要性埋下伏笔。

![def raw_sources() 伪代码、curl 命令与爬虫、动态内容、认证限制](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0005_00-05-46.jpg)

## 技术限制与法律限制：robots.txt、ToS 与 Consent in Crisis [07:34 - 11:29]

假设认证问题不存在，讲者说"仍然有潜在限制"（"there's still potential restrictions here"）。幻灯片把这一整块分成 Technical restrictions 与 Legal restrictions 两组，行号 115–123，是本章最密集的信息来源，以下为原文转录：

```text
Technical restrictions:                                    # L115
• Not allowed to download some content based on
  robots.txt (example) (voluntary)                          # L116（高亮）
• Website might use Cloudflare to detect and block
  bot activity (present CAPTCHAs)
• Website might block certain IP addresses / countries
• Website might have rate limits
Legal restrictions:                                        # L121
• Terms of service (ToS) might prohibit downloading
  using bots                                               # L122（高亮）
• You might not have a license to copy the webpages
  (for training)
```

先说 `robots.txt`。讲者描述它是放在域名根目录层级的一个文件，例如 `nytimes.com` 根下，用来告诉爬虫哪些允许、哪些不允许；里面常列出一串被点名的 bot 名字（口述提到 OAI Search bot、perplexity bot、Cora bot，以及 "chat GPT user, Claude bot" 之类）。关键是它的性质：讲者反复强调 "This is not a legal restriction. This is just like you're supposed to be a good citizen and look at robots.txt"，并说"如果你的名字在名单上，你就不该爬"，"That's the contract — not even a contract"。也就是说这是自愿的行业约定（幻灯片第 116 行特意标了 `(voluntary)`），不是法律强制。与之并列的技术障碍更硬一些：许多站点用 Cloudflare 之类检测并拦截 bot 活动，于是你会遇到 CAPTCHA；站点也可能封某些 IP 地址或国家，或者设置速率限制（rate limits）。这些是工程上的"能不能"问题，与"准不准"是两回事。

再往下是法律限制。服务条款（Terms of service，ToS）是使用网站时要遵守的合同，讲者说它常写明 "If you are a bot, go away"、"You cannot use this the content of the site for AI training"。即便 ToS 什么都没写，网页内容本身也可能带有、或根本不带有允许你拿来训练的许可（license），"you might not have a license to train"——这一点被留到后面版权章节展开。所以同一份网页至少叠着三层判断：技术上抓不抓得到、约定上该不该抓、法律上能不能拿来训练；把它们混为一谈是最常见的错误。

这些限制并非静态。幻灯片第 124–127 行以 "Decline of consent [Longpre+ 2024]" 为标题给出结论："Examined restrictions (robots.txt, ToS) for URLs in common datasets (C4, RefinedWeb, Dolma)"、"Restrictions have increased over time"。讲者口述时把这项研究称作 *Consent in Crisis*（作者名字在转写里含糊记为 "Shane Lampray"），说它考察了常用数据集中各 URL 的技术限制与法律限制随时间的变化，结论是限制在增加。这里需要谨慎：转写里的论文名与幻灯片上的 "Decline of consent [Longpre+ 2024]" 并不一致，本段画面与口述给出的就是这两种标注，不宜自行断定二者是同一篇。

![两张 2016–2025 年限制比例堆叠面积图：Robots.txt Restrictions 与 ToS Restrictions](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0009_00-10-53.jpg)

图本身（10:53）是两张上下排列的堆叠面积图，横轴 2016 到 2025，纵轴都是 0%–100%，2024 之后标着 "Forecast"（预测区）。上图叫 "Robots.txt Restrictions"，图例含 `Full restrictions`、`Pattern-based restrictions`、`Disallow private directories`、`Other restrictions`、`Crawl delay specified`、`Sitemap provided`、`No restrictions or sitemap`、`No Robots.txt`；讲者让观众"只看红色那块"（"if you just look at this red"），即 `Full restrictions`：2023 年之前基本平稳，到 2023 年中之后突然抬升，"grown to almost like, you know, 50% here"。图上还有若干事件竖线标注，能读到的包括 `ChatGPT`、`GPT-4`（2023 附近）与 `GPTBot`（2024 附近）。下图叫 "ToS Restrictions"，图例含 `No Crawling & AI`、`No Crawling`、`No AI`、`Conditional Use`、`Unrestricted Use`，以及 `No Commercial Use`、`Non-Compete`、`No Re-Distribution` 等；事件标注有 `GDPR Ad.`（2016）、`GDPR Enf.`（2018）以及同样那几个模型与 bot 节点。讲者的解读是：2016 年几乎没人往页面上写条款（"in 2016, no one really put any terms on their pages"），"now most pages put some terms"，而且 "most of the terms say you can't use this for AI"。把两张图合起来的结论很有分量：即便在 2020 年"能爬整个互联网"在技术上大致成立，今天"法律上可爬的互联网"其实小得多（"at least what you can legally crawl is actually much smaller"）。若把完全限制的比例记作 $r_{\text{full}}(t)$，那么讲者描述的形状可示意成

$$ r_{\text{full}}(2023^{-}) \approx \text{const} \ \longrightarrow\ r_{\text{full}}(2024) \approx 0.5 $$

这只是对"平稳后跃升到近 50%"这一口头读数曲线化，并非图上给出的公式或精确数值。

与前后内容的关系：本章是第 3 章"拿不到全部网页"的制度化与量化，把"抓不到"细分为自愿约定、技术封锁与法律合同；同时讲者明确预告 "there's copyright which we'll talk about soon. That's a whole another complex can of worms"，直接把话题递给第 6、7 章。易错点不少：`robots.txt` 是自愿的而非法律；CAPTCHA、IP 封锁、rate limits 是技术障碍而非许可；图里 2024 年之后是 Forecast 而非实测值；"限制增加"是对 C4、RefinedWeb、Dolma 这些常见数据集里 URL 的统计，不能直接外推成全网的因果结论；最后，红色的 "full restrictions" 接近 50% 是讲者对着图的目测估计（"almost like 50% here"），不是精确标注的数值。

## 爬虫的副作用与影子图书馆 [11:29 - 14:20]

讲者先给前面的限制加了一句限定："these are guidelines and sometimes either intentionally or unintentionally you could violate these things"——无论有意还是无意，这些约定都可能被违反。于是画面切到 "When crawlers are not well-behaved:"，展示两条 2024 年 7 月 24 日的真实推文，这部分人名、账号、数字全是图上独有、转写只含糊说成 "there's this guy ... I forget the name of this the site" 的增量。第一条来自 Kyle Wiens（`@kwiens`，带认证标志）："Hey @AnthropicAI: I get you're hungry for data. Claude is really smart! But do you really need to hit our servers a million times in 24 hours? You're not only taking our content without paying, you're tying up our devops resources. Not cool." 下面的互动数是 90 条回复、857 次转发、10K 个赞、1.6M 次浏览。第二条是 Eric Holscher（`@ericholscher`）的回复："Yea, they were hammering us over at @readthedocs as well. We were planning to write a blog post on it, since this behavior is definitely gonna get all AI crawlers blocked because of abuse, not even because of the copyright issues." 时间戳 `1:31 PM · Jul 24, 2024 · 111.8K Views`，互动数 7 回复、31 转发、732 赞、59 收藏。以下为画面要点与数字的原文转录：

```text
When crawlers are not well-behaved:                         # L129
Kyle Wiens @kwiens · Jul 24, 2024
  "hit our servers a million times in 24 hours"
  90 replies / 857 retweets / 10K likes / 1.6M views
Eric Holscher @ericholscher
  "hammering us over at @readthedocs as well"
  1:31 PM · Jul 24, 2024 · 111.8K Views
  7 replies / 31 retweets / 732 likes / 59 bookmarks
• Factors: ToS, robots.txt, server load
  (degrades service, costs website money)                  # L132
• And then there is copyright (more later)...              # L133
```

把 "a million times in 24 hours" 摊成速率，若按 24 小时均匀计，约为 $\lambda \approx 10^6 / (24 \times 3600) \approx 11.6$ 次/秒——这只是把图中给出的那个量级做个量纲换算，便于理解"打爆服务器"的含义，并非讲者或画面上给出的数字。讲者借此强调：即便完全不谈版权，爬虫本身就已经有问题了。原因被幻灯片归纳为 `ToS, robots.txt, server load (degrades service, costs website money)`：要么违反了 ToS 或 `robots.txt`，要么只是单纯制造了巨大的服务器负载，而负载是要托管方掏钱的（"this costs money for the person hosting it"），还会让所有其他使用者的服务质量下降（"degrade service for everyone else"）。他也补了一句 "I think probably has hopefully been corrected by now"，说明这是一段特定时期的现象，不能当成当下持续存在的事实。第 133 行的 "And then there is copyright (more later)..." 则明确把版权留到后面。

接着是另一类特殊来源：影子图书馆（Shadow libraries），幻灯片标题旁标着 `[article]`。图上列出的要点转写只念了一部分，完整内容是："Technically part of the web"；"Examples: Library Genesis (LibGen), Z-Library, Anna's Archive, Sci-Hub"（转写只提到 LibGen 与 Anna's Archive）；"Disregards copyright and bypasses paywalls (e.g., Elsevier)"（Elsevier 这个付费墙例子是图上独有的）；"Received takedown orders, lawsuits, blocked in various countries"；被高亮的第 139 行 "Usually controls are circumvented, have servers in various countries"；"Some argue this makes freely available what should be free"；"From a legal perspective, this is piracy and copyright infringement"；以及两个具体规模 "LibGen has ~4M books (2019), Sci-Hub has ~88M papers (2022)"。讲者口述把这些站点描述成"完全无视版权、绕过付费墙"，把本该付费的书籍与文章免费公开，虽然收到大量下架令与诉讼、在多个国家被封，但通过在别国架设服务器继续运作；支持者主张这是"把本应自由的东西自由化"，但从法律角度看就是盗版（piracy）与版权侵权，并预告后面还会回到这一点。

![Kyle Wiens 与 Eric Holscher 关于爬虫压垮服务器的推文截图](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0010_00-12-10.jpg)

这一部分的收束是幻灯片第 145–147 行的 "Summary: The Internet is huge / Many technical and legal restrictions on what data one can access"，对应讲者口述的 "the internet is a huge and messy and you know, scary place ... you can't actually get all of it"。他给了个很实用的收尾："next time someone says 'I trained on the internet', you can tell them all of these things you learned"——下次再听到"我在整个互联网上训练"，就可以把动态内容、登录墙、`robots.txt`、ToS、负载、影子图书馆这一串限制摆出来。

![影子图书馆的定义、示例 LibGen / Z-Library / Anna's Archive / Sci-Hub 与规模数字](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0011_00-13-27.jpg)

与前后关系上，本章为从 04:36 开始的"数据从哪来"整块画上句号，下一帧幻灯片立刻切到 `def copyright():`，开启"哪些数据合法可用"的全新主题。易错点值得单独记：把影子图书馆的内容当成"免费可用"是严重误解，讲者明确说法律上属盗版；`~4M books (2019)` 与 `~88M papers (2022)` 是带年份的历史数字，不是当前规模；"24 小时一百万次"是站长口径且讲者说希望已被纠正，不宜当作现状；另外 "Sci-Hub 有 ~88M papers" 与 "LibGen 有 ~4M books" 单位不同，别混成同一类统计。

## 版权法框架：知识产权、1976 年法案与保护边界 [14:20 - 18:57]

讲者把问题推进一层：假设你已经遵守了 ToS、也守了速率限制、是个模范好公民，"are you allowed to train on this data?" 仍有答案吗？他明说这是一个尚未完全解决的持续争议（"a ongoing question that has not been fully resolved"），只是过去一年有不少进展（"there has been a bunch of developments in the last year"）。幻灯片用 `def copyright():` 起头，第 151 行高亮写着这个母问题："What data is legal to use (for training)?"。以下为第 150–160 行原文转录：

```text
def copyright():                                           # L150
    What data is legal to use (for training)?              # L151（高亮）

Intellectual property law                                 # L153
• Goal: incentivize the creation of intellectual goods
• Types of intellectual property: copyright, patents,
  trademarks, trade secrets.                              # L155

Copyright law:                                            # L157
• Goes back to 1709 in England (Statute of Anne),
  first time regulated by governments and courts [article]
• In United States, most recent: Copyright Act of 1976
  [article]                                               # L159（高亮）
```

归入的法律语境是知识产权法（intellectual property law）。讲者特意提醒要记住"法的精神"（the spirit of the law）：这套制度的目的不是技术性地对一切说"不"，而是激励智力产品的创造（幻灯片写作 "Goal: incentivize the creation of intellectual goods"）。它覆盖多种形式——版权（copyright）、专利（patents）、商标（trademarks）、商业秘密（trade secrets），而其中与语言模型数据最相关的是版权法（copyright law）。版权法的历史被写成两条：可追溯到 1709 年英格兰的《安妮法令》（Statute of Anne），那是政府与法院第一次真正为"激励创新与创作"而规制版权；美国近代框架则以 1976 年版权法（Copyright Act of 1976）为主。这两个年份与 "Statute of Anne" 这个专名都是图上给出、转写里只含糊说 "1700s in England" 的增量。

法条原文（第 160 行斜体）被整段引用："Copyright protection applies to 'original works of authorship fixed in any tangible medium of expression, now known or later developed, from which they can be perceived, reproduced, or otherwise communicated, either directly or with the aid of a machine or device'"。核心判据有两个词：原创（original works of authorship）与固定（fixed in any tangible medium of expression）。把这套判据形式化，可写成

$$ \text{protected}(x) \iff \text{original}(x)\ \wedge\ \text{fixed}(x),\qquad \text{duration} = 75\ \text{years} $$

同样要说明，这是把幻灯片要点记号化，不是画面上出现的公式；其中 75 年来自后面第 171 行。围绕这条定义，幻灯片第 162–171 行给出一串极重要的限定，转写念到但图更完整："Collections are not original works so hence not copyrightable (e.g., telephone directories) unless there is some creativity in the selection or arrangement"——合集不可版权，除非在选取或编排上有创造性，讲者举电话簿（telephone directory）为例；"Copyright applies to expression, not ideas (e.g., quicksort)"——版权保护表达而不保护思想，你不能给快排算法（quicksort）本身上版权，但可以给某一份快排实现上版权；"Expanded scope from 'published' (1909) to 'fixed' (1976)"（第 165 行高亮）——1976 年把门槛从 1909 年的"已发表"放宽到"已固定"，于是不再要求发表、也不要求任何注册；"Registration not required for copyright protection (in contrast with patents)" 与 "Threshold for copyright is extremely low (e.g., your website is copyrighted)"——与专利要花一大笔钱去申请（"you have to go pay a lot of money to get a patent"）相反，版权的门槛极低，"你把东西放到网站上，它就受版权保护了，就这样"（"That's it"）。

但注册并非毫无作用，只是作用点在别处："Registration is required before creator can sue someone for copyright infringement"、"Costs $65 to register [article]"。讲者解释，若要起诉才需要注册，费用只有 65 美元，远小于大概率要付的律师费（"much smaller than the lawyer fees that you'll probably pay"），因此"门槛依旧非常低"。最后是期限："Lasts for 75 years, and then the copyright expires and it becomes part of the public domain (works of Shakespeare, Beethoven, most of Project Gutenberg, etc.)"（第 171 行高亮）——75 年后版权到期、内容进入公有领域（public domain），人人可自由使用；图上举的例子是莎士比亚、贝多芬与 Project Gutenberg 的大部分书，这些专名转写里只笼统说成 "great works"。讲者给出的立法理由是激励创新：创作者在世时该受保护，免得别人直接拿走，但 75 年后大概不值得再保护、或作者已去世，继续锁着就没有意义了。这一节的结论由第 173 行的总结一句话点破："Summary: basically everything on the Internet are copyrighted."

![Intellectual property law 与 Copyright Act of 1976、法条原文及保护边界要点](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0013_00-16-01.jpg)

与前后关系：本章承接第 4、5 章"能不能爬"的技术与合同限制，把问题换成"能不能训练"，而"几乎所有内容都有版权"这个结论又直接逼出下一章必须回答的两条合法路径。易错点集中在几组容易混淆的区分上：版权保护表达不保护思想，别把 `quicksort` 算法当成可版权物；合集原则上不可版权，但"选取或编排的创造性"是例外；注册不是"获得保护"的前提，却是"提起诉讼"的前提，二者不能对调；75 年是讲者在本段给出的口径，此处按画面与口述记录，不引入画面外的现行法细节；最要紧的是，"everything on the Internet are copyrighted" 绝不等于"everything 都不能用"——讲者在 18:57 之后紧接着就说 "Well, not necessarily"，这正是下一章的起点。

## 合法使用的两条路径：许可、合理使用与 Creative Commons [18:57 - 20:30]

面对"网上几乎所有内容都有版权"，讲者立刻自己设问："So, that means is every everything just if I train on anything, that's a copyright violation?" 回答是 "Well, not necessarily."——一切都受版权保护，但受版权保护的作品是可以使用的（"everything is copyrighted, but you can use copyrighted works"）。幻灯片第 175–183 行把办法写成两条，这也是本段最后一块内容，以下为原文转录：

```text
How to use a copyrighted work:                             # L175
1. Get a license for it.                                   # L176
2. Appeal to the fair use clause.                          # L177

Licenses                                                   # L179
• A license (from contract law) is granted by a
  licensor to a licensee.                                  # L180
• Effectively, 'a license is a promise not to sue'.       # L181
• The Creative Commons license enables free
  distribution of copyrighted work.                        # L183（高亮）
```

把两条路径合并成一个记号，就是 $\text{legal-use}(x) = \text{license}(x)\ \vee\ \text{fair-use}(x)$；这里同样只是对第 176–177 行的形式化，不是幻灯片上的公式。需要特别克制的是：本段转写只出现了 `fair use clause` 这个名词，讲者并没有在本段展开合理使用的判断标准（四要素之类出现在后面的讲次），因此这里只能记录"存在这条路径"，不能自行补齐它的内容——这是"不编造转写与画面外内容"的硬约束。

先看许可（license）。讲者说它来自合同法（contract law）：由许可方（licensor）授予被许可方（licensee），实质上相当于一句"别告我"（"Basically says, don't sue me"），你可以按许可允许的方式使用作品（"you can use this work in the ways that the license permits"）。幻灯片把它凝练成 "Effectively, 'a license is a promise not to sue'"。这里要区分两个法域：许可是合同关系，版权是另一套制度；拿到许可并不消灭作品本身的版权，只是获得了不被起诉的约定。

重头戏是知识共享许可（Creative Commons license），第 183 行高亮："The Creative Commons license enables free distribution of copyrighted work." 讲者的解释是，它让作品"表现得像在公有领域一样"（"kind of act like it was in the public domain"），从而弥合公有领域与现有版权之间的鸿沟——否则创作者想让人自由使用，也得苦等 75 年到期；有了 CC，就可以用一个法律手段主动声明"我允许"。幻灯片写明它 "Created by Lessig and Eldred in 2001 to bridge public domain and existing copyright"，这两个人名（Lessig、Eldred）是图上独有、转写只说 "It was created in 2001" 的增量。举例一行也比口述丰富得多，转写只念到 Wikipedia、OpenCourseWare、Khan Academy，而画面上还列着 "Free Music Archive, 307 million images from Flickr, 39 million ... from MusicBrainz, 10 million videos from YouTube, etc." 这样的量级——保留原文数字即 `307 million`、`39 million`、`10 million`，它们说明 CC 覆盖的不只是百科与课件，还有成规模的图像与视频集合。

![Licenses 与 Creative Commons：合同法定义、CC 示例及模型开发者的数据许可](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0016_00-19-51.jpg)

再往下，幻灯片第 187 行起列出 "Many model developers license data for training foundation models"，也就是有钱的模型开发者会直接为训练基础模型取得数据许可，条目包括 `Google and Reddit [article]`、`OpenAI and Shutterstock [article]`，以及底部被截断的 `OpenAI and StackExchange [article]`。这三组配对在转写里并没有念到（讲者口述到 20:27 只说到 "so far things in the public domain and things that have creative commons"），属于纯画面增量，正好说明了"许可"这条路径在工业界的实际形态：不是等公有领域，而是直接签约。本段就在 20:27 这句 "things in the public domain and things that have creative commons" 处收尾，后续关于合理使用、判例与具体数据集的讨论都留在本段范围之外。

与前后内容的关系上，本章是第 6 章结论的直接解答，也把整段从"数据从哪来"平稳过渡到"哪些数据合法可用"这一新主题；同时 `fair use clause` 与这些许可案例为后续讲次埋下接口。易错点有四：CC 不是"放弃版权"，而是"预先授予使用许可"，作品依然受版权保护；许可出自合同法，与版权法分属两套体系，"promise not to sue" 是讲者的口语化概括而非严格法律定义；本段只提出 `fair use` 这个名词、未给出任何判定要素，不能据旧笔记或常识补写四要素；最后，Google–Reddit、OpenAI–Shutterstock、OpenAI–StackExchange 是幻灯片列举的行业事实，说明头部厂商可以靠资金拿到许可，并不代表所有开发者都有同等条件，这正是数据不对称的又一体现。

## 许可（Licenses）：合同法下的授权与付费交易 [20:30 - 21:12]

承接上一段「互联网上几乎所有内容都自动受版权保护」的结论，能够合法使用他人作品的第一条路径是取得许可，英文叫 license。它属于合同法（contract law）范畴，由许可方 licensor 授予被许可方 licensee，讲义上给了一句极为精炼的直觉定义：Effectively, "a license is a promise not to sue"，也就是「许可本质上就是承诺不起诉你」。这个定义解释了为什么内容明明有版权、拿到许可却能安全使用：权利人只是放弃了对特定使用行为追究的权利，并不是把版权转让出去。把它形式化地记作

$$\text{license}: \text{licensor} \longrightarrow \text{licensee}, \qquad \text{license} \equiv \text{"a promise not to sue"}$$

讲者在转写开头强调，现实里大量素材「既不是 Creative Commons 许可，也不在公有领域（public domain）」，这时就必须单独去谈授权；换句话说，免费可得与合法可训练是两件事。用英文原话说，是 "there's many other things ... it's not a creative commons license, it's not in the public domain. You have to get a license"。

画面上这一页把 Creative Commons（知识共享）的定义与实例列得很细，这些是转写没有逐条念到的增量。幻灯片写 "The Creative Commons license enables free distribution of copyrighted work"，即让创作者主动声明作品可以像公有领域那样自由分发；采用它的例子包括 Wikipedia、Opencourseware（开放课程）、Khan Academy、Free Music Archive，以及几笔规模化素材：来自 Flickr 的 $307$ million 张图像、来自 MusicBrainz 的 $39$ million、来自 YouTube 的 $10$ million 段视频等。幻灯片还注明它由 Lessig 和 Eldred 在 $2001$ 年创立，目的 "to bridge public domain and existing copyright"，即填补「公有领域」与「现行版权」之间的空档。这一点与上一段讲的 $1976$ 年版权法正好呼应：正因为默认保护过强、注册又非必需，创作者才需要一个标准化的反向声明工具，把「可自由分发」明确写出来。这些以百万计的图像与视频数字说明，CC 不只是法律概念，早已沉淀为可批量获取的训练素材。

转写 [20:46] 之后给出另一条路："if you have money, you can pay someone to give you a license"，于是 "there's a bunch of deals between model developers and content ... platforms that ... allow the model developers to license the data for training foundation models"。幻灯片在 "Many model developers license data for training foundation models" 标题下列出三笔真实交易，每条都带 [article] 链接标注：Google and Reddit、OpenAI and Shutterstock、OpenAI and StackExchange。这三笔分别覆盖社交论坛、图库素材与技术问答三类不同来源，说明「付费签授权」已经是基础模型（foundation model）训练数据获取的常规操作而非特例，也把前面抽象的「许可方—被许可方」关系落到了具体的商业主体上。讲者用 "Okay, so that's the first thing" 收束：第一条合法路径就是许可，要么走 Creative Commons 这类免费授权，要么直接付钱签合同。

这一章与后文的关系需要说清。转写 [21:12] 立刻转向 "if you don't want to pay, you can appeal to fair use, which is a more complicated matter"——许可虽然干净但有成本、且要逐家去谈，因此后面四章的主线是更复杂的 fair use（合理使用）。易错点至少有三层：第一，拿到版权层面的许可并不等于可以任意获取内容，[27:02] 会补上 terms of service（服务条款）这一独立的合同层；第二，许可通常针对「一组持续更新的内容」而非一次性买断，这一点在 [31:35] 的课堂问答里会被学生直接问到；第三，不能把「免费能下载」等同于「合法可训练」，CC 给的是分发自由，而 Flickr、YouTube 上那 $307$ million / $10$ million 素材的具体使用还要看各自站点条款。讲者在这里并没有下任何法律结论，只是把两条路径摆出来，这种克制的措辞后面会反复出现。

![许可的合同定义、Creative Commons 实例与三笔模型数据授权交易](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0017_00-21-08.jpg)

## 合理使用（Fair use）：第 107 条与四要素权衡 [21:12 - 23:45]

当不想付费、也拿不到许可时，可以诉诸 fair use（合理使用）。讲者明确定位了它的法源："this is section 107 of the Copyright Act"，即美国版权法第 $107$ 条；并给出最直白的一句话概括：fair use says, "I can use this anyway, even if I don't have a license"。但紧接着他就泼了冷水：这四条 "determine whether fair use applies"，而且 "none of these are, you know, hard rules. They're just tendencies which have to be weighed in court if it comes to that"——四要素只是倾向性的权衡维度，真到诉讼时由法院综合判断，绝不是可以机械打分的检查表。这一点必须前置，否则整章都会被误读成一张打分卡。把讲义的四要素编号形式化记作

$$\S 107:\quad \text{fair use} \sim \Big\langle \underbrace{f_1}_{\text{purpose / character}}, \underbrace{f_2}_{\text{nature}}, \underbrace{f_3}_{\text{amount \& substantiality}}, \underbrace{f_4}_{\text{market effect}} \Big\rangle$$

这里 $f_1 \ldots f_4$ 表示四个待权衡的维度，并非加权求和的系数；幻灯片原文 "Four factors to determine whether fair use applies" 强调的是「判定是否适用」，而不是算出某个分数阈值。

第一要素 $f_1$ 是 purpose or character of the use，原文为 "The purpose and character of the use (educational favored over commercial, transformative favored over reproductive)"。转写 [21:36] 起展开两重对比：教育用途比「拿去卖钱」（"trying to sell and make money off of it"）更可能被认定为合理使用；更关键的是转换性（transformative）——"if you transform something, then that's going to be more favored than if you just literally host the identical copy"，即你对作品做了改造、赋予新用途，法律更宽容，而原样托管一份完全相同的副本最不利。这里出现了本讲后面最吃紧的一个词：transformative，[26:03] 论证「训练具有转换性」时正是回到 $f_1$。直觉上讲，$f_1$ 衡量的是「你给社会新增了什么」，而不是「你拿走了什么」。

第二要素 $f_2$ 是 "The nature of the copyrighted work (factual favored over fictional, non-creative over creative)"。转写 [22:07] 举例说，写一份关于二战史实的综述（"facts about World War II"）比抄「一首很有创意的诗」受到的Protection弱，因为事实性作品离版权制度要激励的「原创表达」更远，虚构与高创造性作品处在保护的核心地带。第三要素 $f_3$ 是 "The amount and substantiality of the portion of the original work used (using a snippet favored over using the whole work)"：转写 [22:34] 说 "using a snippet is favored over the whole work ... if you just take a little bit, that's not as bad as taking everything"，并评价 "that's fairly straightforward"。但 $f_3$ 同时看「数量」与「实质性」（substantiality）两个词，这是易错点：即便只取很短一段，若取的是作品最核心、最具识别度的「心脏」部分，仍可能不利，不能只数 token 或字数。

第四要素 $f_4$ 是 "The effect of the use upon the market (or potential market) for the original work"。转写 [22:48] 把它追溯到版权制度的原始动机："why copyright at all? It's to protect uh and align economic incentives"，也就是保护并校准经济激励。若你拿走作者的作品、提供了一个替代品，"decreases amount that the original author could monetize their work"，就被视为潜在不利；反之若 "transforming in a way and adding ... going to a new market or doing something else with it"，开辟了新市场、产生了新价值，则更受优待。$f_4$ 是四要素里最「经济」的一条，也和 [26:34] 「语言模型确实能影响市场」直接扣合，届时它会反过来成为对模型最不利的一项。画面把四条原文与 "Examples of fair use" 并列放在同一页，说明四要素不是悬空法条，下一章就要用具体判例落地。局限性同样清楚：这是 $1976$ 年框架下的定性标准，面对「用数百万本书训练一个模型」这种全新用法，法院尚无统一尺度，讲者后面用 "a little bit slippery"（有点滑溜、难以把握）来形容，正因为四要素要权衡、且都不是硬规则。

![合理使用第 107 条四要素原文与三条典型例子](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0019_00-23-42.jpg)

## 合理使用的例子与版权的语义本质 [23:45 - 25:27]

转写 [23:30] 用 "here's some examples of fair use" 把抽象法条落到具体情形。幻灯片 "Examples of fair use:" 下列出三条，逐条都能与上一段的四要素对上：第一条 "You watch a movie and write a summary of it"，看完电影写一份摘要，属于典型的转换性使用；第二条 "Reimplement an algorithm (the idea) rather than copying the code (the expression)"，重新实现算法这个「思想」（idea），而不是照抄代码这份「表达」（expression）——这正是上一段版权法「保护表达、不保护思想」原则的直接落地，也是高亮强调的一行；第三条 "Google Books index and show snippets (Authors Guild v. Google 2002-2013)"，为图书建索引并展示片段。幻灯片给出的 $2002\text{–}2013$ 是画面独有的精确年份，转写只笼统说 "eventually meaning after 11 years, this lawsuit was settled in favor of Google"，两个口径互相印证：$2013-2002 = 11$ 年。

Authors Guild v. Google 案值得单独展开。转写 [23:42] 起讲：Google Books 展示的是受版权保护图书的 snippets（片段），"there was a huge deal about whether this was deemed fair use because you're sort of redistributing ... other copyrighted work"，争议核心正是「你某种程度上在再分发别人的受版权作品」。这场官司历时 $11$ 年后 "settled in favor of Google, which is why this thing exists"——今天还能看到 Google Books 的片段功能，就是判决结果本身。讲者点明它的外溢意义："this has set some precedent for thinking about whether ... language model training is fair use"，即为后来讨论「训练是否属于合理使用」提供了一个先例锚点。用四要素回看，$f_3$（用 snippet 而非 whole work）与 $f_1$（索引检索是转换性用途，而非替代阅读）两条都偏向 Google，这也解释了为什么它能胜诉。

本段最重要的澄清出现在 [24:32]："copyright is is not about verbatim memorization"。讲者特意提醒 ML 听众 "this might be a little bit new, because a lot of papers are focused on verbatim memorization, but that's ... one way you can violate copyright, but not the only one"。幻灯片 "Copyright is not about verbatim memorization:" 下列两条：Plots and characters (e.g., Harry Potter) can be copyrightable，即情节与角色（如哈利·波特这个角色本身）可以受版权保护，而不只是某一本书的具体文字；Parody (imitating to make fun of something) is likely fair use，即戏仿——模仿某物以取笑它——更可能算合理使用。转写 [24:55] 补充，戏仿虽然「看起来像衍生作品」，"as long as you're making fun of it, it's actually more likely to be fair use"。幻灯片最后一句总结 "Copyright is about semantics (and economics)"，转写也说 "It's definitely not about n-gram overlap"。把这层区别写成对照式：

$$\text{copyright} \not\equiv \text{verbatim memorization} \;\not\equiv\; \text{n-gram overlap}, \qquad \text{copyright} \approx \text{semantics} + \text{economics}$$

这对机器学习背景的听众是个思维切换：很多论文把版权风险简化为「模型是否逐字背出训练文本」，于是用 verbatim memorization 或 n-gram overlap 这类可计算的表面指标去度量；但法律看的是语义（角色、情节、整体观感）与市场（economics，也就是 $f_4$）。由此得到几个易错点：即便模型一个字都没背下来，只要生成了受保护的角色、情节，或冲击了原作市场，仍可能构成侵权；反过来，戏仿这种「表面最像抄袭」的形式，反而因语义上的转化与调侃被宽容；而「逐字记忆率」这个指标既非充分也非必要，不能当作合规判据。讲者用 "Okay. So, what is the implication for language models?" 收束一般法理，下一章就把这四要素、这层语义观整体套到训练动作上。局限也要说明：这些例子都来自前 AI 时代的判例与常识，「训练」是否与它们同属转换性使用，幻灯片和转写都只说「提供先例」，并未替法院下定论。

![版权不等于逐字记忆：情节、角色可版权化与戏仿例外](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0020_00-24-59.jpg)

## 对语言模型的含义与服务条款的叠加限制 [25:27 - 27:35]

把前面的框架套到语言模型上，幻灯片 "Considerations for language models:" 给出四条，转写 [25:27] 起逐条口头展开。第一条："Copying data (first step of training) is violation already even if you don't do anything with it"——转写 [25:30] 强调 "even the mere fact of copying, which is in the word copyright, is potentially a violation already, even if you don't do anything with it"，仅仅把数据复制下来这一步就已经潜在侵权，因为 copyright 这个词的字面就是 copy 加 right，「复制之权」。第二条："Training a model should be transformative (far from just copy/pasting)"——转写 [25:49] 措辞很谨慎，"This is not necessarily a fact, but is intuitively has a transformative flavor"，并说 "It certainly seems different than just like re-hosting another work"，即训练直觉上有转换性，明显不同于把作品重新托管一份。第三条（画面高亮）："Model should be about the general idea (e.g., wizards), not in the concrete expression (e.g., Harry Potter)"——转写 [26:16] 说模型把数据当作 "a means to the end"，是要 "extract the general idea, learn about the world and how it works, rather than only just the concrete expression"：学「巫师」这类一般概念，而不是「哈利·波特」这一具体表达。第四条："Language models can definitely affect the market (writers, artists), regardless of copyright"——对应 $f_4$，转写 [26:34] 说 "by negatively affecting the market, you're more likely to be ruled not fair use"。写成一组对照式：

$$\text{copying} \Rightarrow \text{potential violation},\quad \text{training} \Rightarrow \text{transformative (tendency)},\quad f_4:\ \text{market effect} \Rightarrow \text{against fair use}$$

要特别注意第二条用的是 should be 与 intuitively，是讲者的直觉而非法院结论，他自己保留了余地。

第①条与第②条之间其实构成一组张力：单独看「复制」这一步已经踩线，但把「训练」作为整体看又显得是转换的。转写用 "trained on this data as a means to the end" 给出直觉——模型不是在重发作品，而是把作品当作统计素材去学世界的一般规律，落点是 general idea 而非 concrete expression，这恰好接回上一章「版权关乎语义而非逐字记忆」的结论。可第④条又把这份乐观拉了回来：模型对作家、画家市场的冲击是真实存在的，而 $f_4$ 恰是四要素里最「经济」、也最可能单独翻盘的一条。于是讲者在 [26:56] 评价 "fair use is kind of a little bit, you know, slippery"。这里的易错点是：不能因为「训练有转换性」就断言整体合法，四要素必须权衡，且 $f_4$ 的负面影响可能把结论整个翻过来；也不能把「模型没背出原文」当作安全证明。

更常被忽略的是 ToS 这一层。转写 [27:02]："suppose you have fair use or you have a license, remember that there's still terms of service that ... prevent you from actually just getting a piece of work"。幻灯片 "Terms of service:" 下写 "Even if you have a license or can appeal to fair use for a work, terms of service might impose additional restrictions"，并给出高亮例子："Example: YouTube's terms of service prohibits downloading videos, even if the videos are licensed under Creative Commons"。转写 [27:12] 解释：YouTube 上很多视频本身是 licensed（采用 CC），但服务条款 "forbid prohibits just downloading the videos as using a bot or like scraping"，禁止用 bot 或抓取的方式下载，并总结 "there's multiple layers of restrictions here"。把这两层约束写成配置式的代码（依据幻灯片原文）：

```text
copyright_layer : license(CC | paid) OR fair_use(section 107)
platform_layer  : ToS  # may forbid bot scraping of CC-licensed content
```

关键在于：CC 给的是版权层面的分发许可，ToS 是使用该平台的合同，二者互不替代、各自独立；一个内容可以同时「有 CC 许可」且「被 ToS 禁止下载」。后面讲 Common Crawl 的抓取策略时，还会补上 robots.txt 这一自愿的 politeness 层，届时「多层限制」的图景会更完整。

本章是从抽象法条通向模型工程的桥：前三章分别讲许可、四要素、语义本质，本章把它们映射到「复制 / 训练 / 市场」三个具体动作，并补上常被漏算的合同层。讲者用 "Okay, so let me talk a little bit about where things are on the legal landscape"（[27:35]）转入真实判例。易错点可以归纳为三条：把 copyright 与 ToS 混成同一层、把「转换性」当成已决结论、以为有 CC 许可就能用 bot 批量下载。这些误解在下一章的诉讼里会看到现实后果——被判违法的往往不是「训练」，而是「怎么把副本弄到手」。

![语言模型的四点版权考量与服务条款叠加限制](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0021_00-26-16.jpg)

## 法律版图：三起诉讼、阶段性结论与课堂问答 [27:35 - 32:16]

第一起是 The New York Times v. OpenAI (2023)。转写 [27:40]："back in 2023, New York Times filed suit against OpenAI saying, 'You trained on our news articles and look, here's evidence we were able to actually prompt ChatGPT to generate a news article almost verbatim.'" 幻灯片把指控压缩为 "Allegation: for training and reproducing NYT articles"。关键词是 almost verbatim（近乎逐字）——原告拿「能复现新闻」当证据，这恰好落在上一章说的 verbatim memorization 上，也是 ML 圈最熟悉的那类度量。讲者说 "I don't think this is ... still pending"，即到本讲时仍未了结。易错点：本案指控同时包含「训练」与「复现／输出」两面，不能只当成「训练是否合理使用」来读，输出端的逐字复现是独立的一条线。

第二起是 Authors (Bartz, Graeber, …) v. Anthropic (2024)，幻灯片给出了转写没念的精确原告名单与年份。逐条为：Allegation: for pirating millions of books and training on these books to make Claude；Summary judgement (2025): training on plaintiff's works is fair use；...but pirating copies is not (even if don't train)；Anthropic also bought and scanned the books; this is also fair use (but too late)；Outcome: Anthropic paid $1.5B to authors to settle。转写 [28:08] 把 $2025$ 年的结果称为 "a kind of a landmark ruling"（里程碑式裁决）：这一具体情形的训练属于合理使用，但 "you're still in trouble because you pirated all these books ... that is a no-no"，而且 "this has nothing to do with training. Just a mere fact of pirating ... is illegal"——违法的是盗版取得副本，与训练无关。最反直觉的一层是 Anthropic 其实 "bought and scanned all of the books"，法院认定 "You buy a bunch of books, rip off the binding, scan and digitize it for your own use. That's fair use"，买书、拆掉装订、扫描数字化自用本身是合理使用；但 "that doesn't absolve you of your sin of pirating"，"You can't pirate and then buy it and say like, actually, never mind what I just did first"——事后补买洗不掉先前的盗版。和解金额与折算写成

$$\$1.5 \times 10^{9}\ \text{settlement} \;\Longrightarrow\; \approx \$3{,}000\ \text{per book}$$

转写 [29:11] 原话是 "paid 1.5 billion to settle the authors. So, this is like about $3,000 a book"。这里的易错点很典型：把「扫描合法」误读成「盗版也能被追溯合法化」，或把这笔钱当成对训练本身开出的罚单。

第三起是 Authors (Kadrey, Silverman, …) v. Meta。幻灯片：Allegation: for training on plaintiff's books (revealed in the Llama paper)；Summary judgement (2025): training on books (in this instance) is fair use [article]；Allegation of torrenting books is still pending。转写 [29:17] 补了三点：训练书籍这件事 "as we'll see later, actually revealed in the Llama paper"（在 Llama 论文里被披露，这也呼应本课前面讲「公开数据来源会招来诉讼」的伏笔）；"the judgment, which came right after Anthropic"；以及 "they also torrented some books, so that's still pending ... if precedent holds, this is also not going to be good for Meta"。幻灯片 "Summary:" 三条给出阶段性结论：So far training has been deemed fair use (for specific instances, but unclear in general)；Pirating books is clearly illegal；Still a very active, evolving area。转写 [29:50] 的措辞更绕、也更保守："training, you know, so to speak, has been deemed fair use or at least has been not deemed not fair use"、"The new rulings have so far been, you know, narrow. It's not to say that any training on any copyrighted content is fair use"、"this is still a very active and evolving uh area"。窄（narrow）这个限定词是本段最该记住的：它只覆盖「这些具体个案、这些具体简易判决」。

[30:31] 之后进入提问环节，讲者先说明意图："before we get into the actual data sets, I just want to kind of set the stage with this kind of a more general social context"——前面这一大段法律内容，其实是给后面的数据集做社会语境铺垫，不是离题。第一个问题关于 "voice dating at 11 labs"，讲者 "I don't know too much about that, so we can talk offline"，明确不展开，按「转写含糊处宁可略过」的要求这里不猜内容。第二个问题（[31:35]）是 "what happens with time if you have one license and then it gets changed"，讲者先声明 "first I'm not a lawyer"（我不是律师），再答："the first license, I believe you're still allowed to train on it, but ... the license usually applies to not just let's say a fixed set of documents, for example, Reddit. There's constantly new content being created and the license being changed means that you wouldn't be able to train on the later things"。设 $t_0$ 为许可变更时刻，$\mathcal{D}_{t<t_0}$ 为变更前已产生的内容、$\mathcal{D}_{t>t_0}$ 为之后新产生的内容，则

$$\text{license valid for } \mathcal{D}_{t<t_0},\qquad \text{license changed at } t_0 \Rightarrow \text{cannot train on } \mathcal{D}_{t>t_0}$$

直觉是：许可往往绑在一个持续产生新内容的平台（Reddit 被点名）上，而不是一份固定文档清单；旧许可期内已经拿到的数据仍可训练，但变更之后新冒出来的内容就不在新许可覆盖范围里了。这也回头解释了第 1 章留下的那个悬念。

三起案子给出的一致信号是：训练在具体个案、简易判决层面被认定为合理使用，但「取得副本的方式」（pirating / torrenting）独立违法，且裁定被反复强调是 narrow 的。易错点：不要把 "training is fair use" 读成普遍豁免——NYT 案仍 pending、Meta 的 torrenting 仍 pending、Anthropic 案以 $1.5$B 和解收场而非最高层级定论；也不要把「买书扫描合法」倒过来当洗白工具。讲者最后用 "Okay, so let's look at various sources of data"（[32:10]）与 "let's go back to crawling"（[32:17]）把话题从法庭拉回工程，下一章进入 Common Crawl。

![三起 AI 版权诉讼的指控、2025 年简易判决与阶段性结论](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0024_00-30-07.jpg)

## Common Crawl 与网页抓取的工程细节 [32:16 - 36:08]

回到数据本身。转写 [32:17] 说 "most model developers have their own crawler because they want to have control over full control over what the data is"，多数模型开发者自建爬虫以求完全掌控数据；不想自建的可以用 Common Crawl。幻灯片以讲义源码的形式写着 `def common_crawl():`，并注明 "Common Crawl is a non-profit organization founded in 2007"。Statistics 三条是：Every ~month, run a web crawl (add 3-5 billion web pages)；Crawls have some overlap but try to diversify；300 billion pages so far。转写 [32:42] 补充 "each crawl has some overlap with a previous crawl, but they try to diversify and get new pages"，并对 $300$ billion 这个官方数字当场提出质疑："It seems a little bit big because if you multiply this number by 20, um, you don't really get a 300 billion, but that's what they say"。把这个「对不上」写成式子，正是本段最值得记的易错点：

$$(3\text{–}5) \times 10^{9}\ \text{pages/month} \times 20 \ \neq\ 300 \times 10^{9}\ \text{pages}$$

即按每月 $3\text{–}5$ billion、乘约 $20$ 个月只得到 $60\text{–}100$ billion，远不到 $300$ billion。讲者没有去解释差异来源，只说 "that's what they say"（官网就这么写的），这里也照录不猜。

规模参照与口径同样要分清。幻灯片写 "How many URLs are there? Hard to estimate, but O(billions)"、Google search index is at least 100 PB，底部还标了 April 2025 那次 crawl 约 $2.10$ billion pages、$372.5$ TB。转写 [33:09] 的口径是 "the Google search index is at least 100 petabytes, according to them"、"each Common Crawl dump has about two billion web pages. So, that's about 372, um, terabytes. And this is not including images. This is just mostly focused on the text"。注意两个不同层次的数字：每月新增 $3\text{–}5$ billion pages，而单个 dump 约 $2$ billion pages / $372$ TB；且这 $372$ TB 不含图像、主要是文本。写成对照：

$$1\ \text{dump} \approx 2 \times 10^{9}\ \text{pages} \approx 372\ \text{TB (text only)},\qquad \text{Google index} \ge 100\ \text{PB}$$

$100\ \text{PB}$ 是 $10^{5}\ \text{TB}$ 量级，可见公开可抓的网页文本只是搜索索引规模里很小的一块；这也从数量上解释了上一章「你拿不到整个互联网」的判断。

爬虫在概念上简单，难的全在实现。转写 [33:41]："crawling is conceptually straightforward, but all the gory details are in the implementation. You start with a bunch of URLs, and then you iterate. It's basically graph traversal ... generally this is done in parallel, over many machines"。画面上是一张 Queue / Storage 方框图，标注 URLs 从页面回流到队列、metadata 写入 Storage，并写 Starts with a set of seed URLs (at least hundreds of millions)、Pop a URL from the queue, download URL, and add hyperlinks to queue。把它写成带语言标注的伪代码（依据转写与图示）：

```python
queue = seed_urls                      # at least hundreds of millions
while queue:
    url  = queue.pop()
    page = download(url)               # HTTP GET -> raw response
    queue.extend(page.hyperlinks)      # graph traversal
    storage.write(page, metadata)      # WARC + metadata
```

Policies 列了三条：Selection policy: which pages to download?；Politeness policy: respect robots.txt, don't overload server；Re-visit policy: how often to check if pages change；外加一条 Challenge: URLs are dynamic, many URLs lead to basically same content。转写 [34:05] 展开：要尊重 robots.txt、不能压垮对方服务器；网站会变，因此需要「频繁变化的页面多抓、不变的少抓」的重访策略；"the same URL leads to different content depending on like some state of the browser"，同一 URL 可能因浏览器状态返回不同内容；"different many URLs might lead to the same content, so there's a lot of duplication"，不同 URL 又可能指向同一内容；"mirror sites ... explicitly is about duplication"，镜像站更是天然的重复源。这正好接回上一章「多层限制」的图景：robots.txt 是自愿的 politeness（礼貌）层，不是法律，而是「做好公民」的约定。

两种格式与 HTML 转文本这一环节，是画面带来增量最大的部分。幻灯片 "Two formats:" 写 WARC: raw HTTP response (e.g., HTML) 与 WET: converted to text (lossy process)。这里要留意一处口误：转写念的是 "WAT file"，但幻灯片明确写 WET；WARC 存的是原始 HTTP 响应——转写 [35:00] 详述 "you send a GET ... out comes a HTTP kind of response. And it's just the WARC file is that response"——WET 则是已经转成文本、因而是有损（lossy）的结果。"HTML to text:" 下列 Tools to convert HTML to text: trafilatura, resiliparse（转写念作 "Trafilatura and Brazilia Parser"，幻灯片给出正确拼写 resiliparse），并注明 The conversion matters for the resulting LM's downstream task accuracy: [Li+ 2024]，图下还附了 arXiv 编号 $2406.11794$。转写 [35:31] 说明这来自 DataComp-LM（DataComp at WLM）论文的 ablation：对比 Common Crawl 自己发布的 WET 文件与其他抽取工具，"it seems like Trafilatura and [resiliparse] are ... better"。幻灯片给出了转写完全没念到的 CORE / EXTENDED 两组分数：

| Text Extraction | CORE | EXTENDED |
|---|---|---|
| resiliparse | 24.1 | 13.4 |
| trafilatura | 24.5 | 12.5 |
| WET files | 20.7 | 12.2 |

无论 CORE 还是 EXTENDED，直接吃 WET files（$20.7$ / $12.2$）都低于两个专用抽取器；CORE 上 trafilatura 比 WET 高出 $24.5 - 20.7 = 3.8$ 分。易错点有两条：把 WARC 当成「已经处理好的文本」，或以为 HTML→文本只是无损的格式化——两者都不对，WET 是 lossy 的，换一个抽取器就能明显改变下游任务准确率，因此这一步是实打实的设计选择而非琐事。讲者用 "Okay, so one thing you can do is just rely on general web crawls"（[36:08]）收束：通用爬取是一条路，但网页并非均匀分布，接着要讲几处专门的高质量来源。

![HTML 转文本消融：resiliparse、trafilatura 与 WET 文件的 CORE、EXTENDED 得分](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0028_00-35-53.jpg)

## 三类专门的高质量来源：Wikipedia、数据投毒与 GitHub [36:08 - 41:00]

转写 [36:08] 先给出动机："one thing you can do is just rely on general web crawls. But often, the web is not a uniform place ... there are specific pockets of really interesting high quality content, and I'll talk about three of those, Wikipedia, GitHub, and archive"。网页不是均匀的一堆站点、随手抽个随机比例就是数据集，而是存在几处高价值「富矿」。讲者点明要讲三个（Wikipedia、GitHub、archive），但本段转写只推进到 Wikipedia 与 GitHub 的开头，第三个来源留给后续；按「不编造」的要求，这里只记下「将讲三个」，不替他补出 archive 的内容。这正是对上一章通用爬取的补充，也延续了本课「数据是长尾、要一处一处去谈」的基调。

Wikipedia 这一页以 `def wikipedia():` 起头，写 "Wikipedia: free online encyclopedia"、Founded in 2001、As of May 2026, 67 million articles across 361 language editions (English, Spanish, German, French most common)。转写 [36:31] 只说了 $2001$ 年与 "67 million articles of around all these different languages"，$361$ 个语言版本与 May 2026 这个时间点都是画面独有的。"What is the scope?" 下两条：Does not contain original thought (no opinions, promotions, personal web pages, etc.)；Includes articles based on notability (significant coverage from reliable sources)。转写 [36:45] 展开得很细：Wikipedia "doesn't contain original thought ... because everything has to be kind of referenced and cited"，一切都要有引用来源，所以 "in some sense, you could argue that Wikipedia ... doesn't contain anything because that's not already on the web"——似乎网上已有的东西它才有；但讲者立刻自我修正 "actually, that's not true because they can also cite books, which are, you know, obviously, you can't easily get them"，条目也可以引书，而书恰恰不容易免费拿到，所以 Wikipedia 依然很有价值。此外还有 "articles based on notability. So, not everyone can get a Wikipedia article about them" 的关注度门槛。把规模写成一行：

$$67 \times 10^{6}\ \text{articles},\quad 361\ \text{language editions},\quad \text{founded } 2001,\ \text{as of May } 2026$$

"Who writes the content?" 三条是：Anyone on the Internet can edit, vandalism gets reverted by administrators；Small number of Wikipedians contribute majority (e.g., Steven Pruit with 5M edits)；Produce periodic dumps every few weeks (no need to crawl)。转写 [37:34] 感慨 "anyone on the internet can write this content. This was the radical thing about wikis, and it's sort of still a miracle ... any vandalisms gets reverted by administrators or bots"，并指出和所有 peer production（同伴生产）系统一样 "a small set of people who do most of the work. For example, this guy has like 5 million edits"。工程上最关键的是 dump："every few weeks they do a periodic dump ... take all Wikipedia, package into a nice tar file ... you don't need to crawl Wikipedia. In fact, they don't want you to crawl Wikipedia ... this is much better than crawling"——官方定期打包，直接下载远优于自己爬。

紧接着是本段最值得记的反例。幻灯片 "Aside: data poisoning attacks [Carlini+ 2023]" 下写：Vulnerability: can inject malicious edits right before periodic dumps happen before edits are rolled back；Exploit: inject examples to cause model to ascribe negative sentiment to trigger phrases (e.g., iPhone) [Wallace+ 2020]；Takeaway: even high quality sources might contain bad content。转写 [38:31] 讲了这个漏洞的完整链条：Carlini 意识到 "these periodic dumps happen at a regular cadence"（dump 按固定节奏产生），攻击者便可 "edit it right before the dump happens"，"the dump happens and then the edits get rolled back, but the dump still has the malicious content"——编辑随后被回退，可快照已经把恶意内容带走了。一旦能把内容注入网页，就有工作显示可 "cause the model to, let's say, ascribe negative sentiment to any sort of trigger phrase like the iPhone"，让模型对某个触发短语（如 iPhone）产生负面情感。讲者的结论是 "if you can consider adversaries, even so-called high-quality contents might contain bad things in them"，并补了一句 "I think this has been a fix since then"。把时间窗写成：

$$t_{\text{inject}} < t_{\text{dump}} < t_{\text{rollback}} \;\Longrightarrow\; \text{poisoned edit} \in \text{dump}$$

易错点非常明确：以为「破坏会被回退」就等于 dump 一定干净——回退发生在 dump 之后，快照已经带走恶意内容；这正是连 Wikipedia 这种所谓高质量来源也不能免检的原因，也给后面所有数据过滤环节埋下动机。

最后 [39:50] 起讲 GitHub："GitHub is a good place for code and code is important not just for if you want coding capabilities for your language model, but if you want general reasoning"。幻灯片 `def github():` 下写 "Code is helpful for programming tasks, but also for reasoning (folklore)"，特意用 folklore（坊间经验）标注这不是定论。GitHub 条目包括：Live service for hosting code repositories founded in 2008 (acquired by Microsoft in 2018)；As of May 2026, GitHub has 420M+ repositories (28M public)；Each repository includes directory structure + commit history + issues + pull requests + comments, etc.；Lots of duplicates (e.g., copied code, forks, etc.)；Allowed to train on any public repository with a permissive license (e.g., MIT, Apache)。转写 [40:04] 补充：$2008$ 年上线、"about the same age as Common Crawl"；仓库 "It's not a file. It's ... a directory with commit history, issues, pull requests, and comments"；"code has generally a lot of duplicates ... either copying code or forking code"；以及 "you're allowed to train on any public repository with a permissive license. So, MIT or Apache licenses are fine"。转写把 $420$ million 口误成 "400 uh 20 million repositories"，幻灯片给出 $420\text{M+}$ / $28\text{M}$ 的精确写法，Microsoft 于 $2018$ 年收购这一条则是画面独有。写成对照：

$$420\text{M+ repositories},\ 28\text{M public},\ \text{founded } 2008,\ \text{permissive license} \in \{\text{MIT}, \text{Apache}\}$$

与前文的关系有两处：MIT / Apache 这类 permissive license 正是第 1 章「许可」路径在代码领域的具体实例；copied code 与 forks 带来的高重复率，则呼应上一章 Common Crawl 里 mirror sites / duplication 的问题。转写最后停在 [40:55] "remember there's two types of data when you talk about GitHub. There's like the repository data"，幻灯片底部也只露出 "Two types of data:" 这个标题。按「不编造转写与画面外内容」的要求，这里只记下讲者即将区分两类数据，不替他补出第二类是什么；本段笔记也在此收束，不越界去写后续讲次的 GitHub Archive、Software Heritage 或 arXiv。这是本段唯一的边界局限，读者需要接下去听才能补齐三类专门来源的全貌。

![Wikipedia 的定期 dump 机制与 Carlini+ 2023 数据投毒攻击](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0030_00-38-48.jpg)

![GitHub 的规模、仓库结构与宽松许可训练条款](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0031_00-40-16.jpg)

## GitHub 两类数据、Software Heritage 与 arXiv 的许可边界 [41:00 - 45:07]

本段开头承接上一部分关于专门高质量来源的讨论，把 GitHub 与 arXiv 这两个「不该爬、只该批量下载」的源讲完。讲者反复强调一个操作层面的直觉：仓库本身应当通过 Git 协议取得，而不是去抓取 GitHub 的网页界面，原话是 "you shouldn't scrape GitHub. You should just download the repository"（不该爬 GitHub，只该下载仓库）。之所以要区分，是因为 GitHub 上的数据天然分成两类，讲义把它明确写成 "Two types of data:"：一类是 **Repository**（仓库内容），通过 `git protocol` 下载；另一类是 **Metadata**（元数据），即 issue、pull request、comment 等，由 GitHub API 提供，并以 **GitHub Archive** 上的事件流小时级快照（hourly snapshots of event stream）形式发布。转写里补充了这条事件流的粒度：几乎每一次 comment、star 或任意 action 都会被记录下来，因此它是一类相当有意思的交互行为数据。讲义在同一页还给出两条容易被念漏的前提：代码里 "Lots of duplicates (e.g., copied code, forks, etc.)"（大量重复，来自复制与 fork），以及 "Allowed to train on any public repository with a permissive license (e.g., MIT, Apache)"（只有宽松许可的公开仓库才可用于训练）。这两条决定了后面所有代码数据集都要先做去重与许可筛选。

讲义里这一段的代码原文（保留行内注释的写法）可以直接看出两类数据的分工：

```python
# Two types of data:
#   Repository: download through git protocol (rather than scraping the GitHub website)
#   Metadata:   GitHub API provides issues, pull requests, comments, etc.
#               (hourly snapshots of event stream on GitHub Archive)
#
# Lots of duplicates (e.g., copied code, forks, etc.)
# Allowed to train on any public repository with a permissive license (e.g., MIT, Apache)
```

紧接着是 **Software Heritage**。转写只说它「专注于仓库而不是元数据」并聚合 GitHub 之外的 GitLab、Bitbucket 等，而幻灯片给出了更完整的定义：它是 "Non-profit organization founded in 2016 that collects and preserves software"，即 2016 年成立的非营利组织，目标是收集与保存软件；被高亮的一行是 "Focused on the repositories not metadata (issues, comments)"，精确划清了它与 GitHub Archive 的边界——前者管仓库，后者管元数据。灰色子条目还列出它聚合的来源 "Aggregates GitHub, GitLab, Bitbucket, PyPI, etc."，并标注了 "As of May 2026" 的规模统计（讲义以淡出子条目给出 $28.8\mathrm{M}$ 量级的数字）。这里要留意一个易错点：$28.8\mathrm{M}$ 是讲义子条目里的规模描述，讲者口述并没有念出这个数，引用时应注明来源是画面而非转写，也不要把它与 GitHub 自身的仓库总数混为一谈。

![讲义中 GitHub 两类数据与 Software Heritage 的定义，高亮行为「只关注仓库而非元数据」](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0032_00-41-39.jpg)

第三块是 **arXiv**，讲义以函数 `def arxiv():` 组织，幻灯片上的信息密度远高于转写。它的定义是 "Website that allows researchers to share and access papers for free since 1991"，最早覆盖 physics，后来扩展到 math、CS、statistics 等；规模写作 "Has ~3M submissions"（约 $3\times10^6$ 篇投稿）。每篇投稿的构成被写成 "Submission: metadata, PDF, LaTeX source (optional)"，即元数据、PDF 以及可选的 LaTeX 源码。许可方面，幻灯片逐条列清："Light approval process (not peer-review)"（有轻度审核但并非同行评审）、"Authors choose (i) all rights reserved or (ii) Creative Commons (e.g., CC-BY)"，被高亮的结论行是 "Metadata (title, abstract) is under a permissive license (CC0)"。此外还有一条转写里只被一句带过、但幻灯片写死的操作提示："Bulk download from Amazon S3, no need to crawl"，即数据应从 Amazon S3 批量下载，完全不需要爬虫。

```python
def arxiv():
    # arXiv: Website that allows researchers to share and access papers for free since 1991
    # Areas: physics (original), math, CS, statistics, ...
    # Has ~3M submissions
    # Submission: metadata, PDF, LaTeX source (optional)
    # Light approval process (not peer-review)
    # Authors choose (i) all rights reserved or (ii) Creative Commons (e.g., CC-BY)
    # Metadata (title, abstract) is under a permissive license (CC0)
    # Bulk download from Amazon S3, no need to crawl
```

![讲义中 def arxiv() 的完整条目，高亮行为元数据采用 CC0 宽松许可](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0033_00-42-56.jpg)

讲者由此引出一个很重要的「易错点」：所谓「在 arXiv 上训练」其实并不明确，因为你必须先把 PDF 转成文本，或者改用 LaTeX 源码——而后者是「一堆文件」而不是单个文档，两种选择得到的语料分布并不相同。转写还补充了口述层面的判断：arXiv 虽然不经同行评审但有审核流程，作者可以保留权利或采用 Creative Commons，因此「everything is actually very clearly licensed」，元数据整体是宽松许可可随意使用，讲者估计 "most arxiv papers are Creative Commons"。要注意 CC0 那一行只保证 **标题与摘要等元数据**，并不自动覆盖论文全文；把「元数据可自由使用」外推成「全文可自由使用」是典型误读，也是后面 BooksCorpus、Books3 两类语料反复踩坑的同构错误。

本章最后是一段课堂问答，构成「数据来源」整个阶段的收束。有同学问模型生成的数据（model generated / synthetic data）是否有使用限制，讲者的回答是 "the short answer is probably yes"，并说明这一点稍后会专门展开，此处不展开。另一位同学问：网上有大量盗版书籍站点，做全网爬取时无法逐一检查网站，怎么保证这些站点不混进语料？讲者的回答相当直白——"you can't"，也就是做不到；"most likely in Common Crawl there are copyrighted books and content that you are not supposed to train on"，即 Common Crawl 里几乎肯定含有本不该用于训练的有版权书籍。他给出的推理是 "remember everything is copyright. Books are not remarkably different from your website"：书籍和网页一样都受版权保护，二者在法律地位上并无本质区别；差别只在于书籍作者若要上法庭，"they'll probably be able to protect that better because it's published"，因为已出版的作品举证更充分。他随后提到，如果真的想极度谨慎，后面会讲一种做法（这正是全讲后段 Common Pile 的伏笔）。这段问答在结构上把前一段的「来源清单」和下一段的「数据集演进」连接起来：既然连专门来源都存在许可边界、连通用爬取都无法排除盗版内容，那么接下来逐个模型复盘数据时，「免费可得」与「合法可用」的落差就会成为贯穿始终的主线。

## 从 BERT 到 WebText：文档级序列与 Reddit 外链代理信号 [45:07 - 48:30]

从 45:07 起讲者明确转场："now talk about the various data sets that are actually used in building models starting all the way back to 2019"。此前讲的是数据的 origin 与 source，即「数据版图」；此后进入按时间顺序复盘各模型真正用了哪些数据集、以及各自的过滤与采集设计。他提醒听众带着一个视角去听："there's many methods of filtering and collecting data and it's interesting to look at these design choices"——重点不是记住某家用了什么，而是比较不同团队在同一问题上的设计取舍。

第一个样本是 **BERT**。转写里说 "BERT was from 20…18"，训练数据 "just trained on Wikipedia and books"；讲义则把它写成函数 `def bert():`，标注 `[Devlin+ 2018]`，并列出 "The BERT training data consists of: Wikipedia, Books"，其中 Books 指向另一个函数 `books_corpus()`。BERT 在数据层面的真正贡献并不是来源本身，而是序列粒度的改变：讲义写 "Important: sequences are documents rather than sentences"，被高亮的对照行是 "Contrast: 1 billion word benchmark [Chelba+ 2013] (sentences from machine translation)"。转写把这一点说得更清楚："all the language modeling research before that was focused on sentences"，即此前语言建模研究基本都在句子层面，而 BERT 把序列换成了文档。这是一个容易忽略但影响深远的设计选择：模型开始接触跨句、跨段落的长程依赖，而不是被切碎的孤立句子；对照物正是 Chelba 等人 2013 年的 10 亿词基准（1 billion word benchmark），其语料来自机器翻译的句子级数据。

```python
def bert():
    # [Devlin+ 2018]
    # The BERT training data consists of:
    #   Wikipedia
    #   Books   -> books_corpus()
    # Important: sequences are documents rather than sentences
    # Contrast: 1 billion word benchmark [Chelba+ 2013] (sentences from machine translation)

def books_corpus():
    # Smashwords
    #   Founded in 2008, allow anyone to self-publish an e-book
    #   2024: 150K authors, 500K books
```

![讲义中 def bert() 与 def books_corpus() 的定义，高亮行为 1 billion word benchmark 对照](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0036_00-46-46.jpg)

`books_corpus()` 指向的课程细节很值得单独说，因为它是全讲反复引用的版权反例。转写描述：Smashwords 是一个允许任何人自助出版电子书的网站，2015 年有论文把其中免费的书抓下来做成语料，"back in the innocent days when no one was paying attention to any of this"，这套 BooksCorpus 在学术界流传多年；"since then it has been taken down because it violated the terms of service"，后来因违反服务条款被下架。讲者给出的教训直白而锋利："just because it was free and you can get it doesn't mean it was legally allowed"。幻灯片补充了转写没念到的两个参数："Founded in 2008, allow anyone to self-publish an e-book"，以及 "2024: 150K authors, 500K books"（约 $1.5\times10^5$ 位作者、$5\times10^5$ 本书）；转写口述只说了「2024 年约 50 万本书」这一个近似量级。这里的易错点有两层：一是「免费」不等于「许可」，二是「学术圈长期在用」也不构成合法性；这条逻辑会在后面的 Books3 问题上原样重演。

第二个样本是 **GPT-2（2019）的 WebText**。讲者交代了动机：目标是高质量网页内容（high quality web content），当时人们已经知道 Common Crawl，但 "somehow the Common Crawl data was too messy"，直接拿它训练效果太差。于是团队用了一个巧妙的代理信号："look at pages that are outgoing links from Reddit posts with greater than three karma"，即只取 Reddit 上 karma（帖子得分）大于 3 的帖子所外链的页面，背后的假设是 "good posts must link to good websites"（好帖子会链接到好网站）。结果是 40 GB 文本、约一百万（$10^6$）个页面；原始数据 "they never released this data"，但社区做了开放复现（即 OpenWebText），被人使用得相当多。这里的直觉很关键：当没有直接的质量标注时，用一个可观测的社会信号（Reddit 投票）作为质量代理，比直接定义「什么是好文本」要省事得多；代价则是分布被 Reddit 用户的兴趣强烈塑形。讲义把 karma 阈值写成更精确的 $\ge 3$，与转写的 "greater than three" 略有差别，引用时宜按图上的符号记为 $\ge 3$ karma。

![讲义的「数据源与各模型」总览代码，高亮行为 ccnet() 一行](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0037_00-48-03.jpg)

这张幻灯片其实是整段（乃至整讲后半）的路线图，转写完全没念到，属于典型的画面独有信息。它以两段注释把函数分成两组，前一组是 `# Sources of data`：`common_crawl()  # Web crawl`、`wikipedia()  # General knowledge`、`github()  # Code`、`arxiv()  # Research papers`；后一组是 `# Data from various models`，按年份列出 `bert()  # Wikipedia, books (trained BERT) [2019]`、`gpt2_webtext()  # pages based on Reddit links (trained GPT-2) [2019]`、`ccnet()  # Filter Common Crawl based on Wikipedia [2019]`（本页高亮行）、`t5_c4()  # Filter using rules (trained T5) [2019]`、`gpt3()  # CommonCrawl, Wikipedia, books (trained GPT-3) [2020]`、`the_pile()  # Lots of sources (trained GPT-J, GPT-NeoX, ...) [2021]`、`gopher_massivetext()  # Filter using rules (trained Gopher) [2021]`、`llama()  # CommonCrawl, CCNet, StackExchange, etc. (trained LLaMA) [2022]`，以及后面才讲的 `refinedweb()  # CommonCrawl (used to train Falcon) [2023]`、`dolma()  # Lots of different sources [2024]`、`dclm()  # Filtered using good quality classifier [2024]`。

```python
# Sources of data
common_crawl()  # Web crawl
wikipedia()     # General knowledge
github()        # Code
arxiv()         # Research papers

# Data from various models
bert()                # Wikipedia, books (trained BERT) [2019]
gpt2_webtext()        # pages based on Reddit links (trained GPT-2) [2019]
ccnet()               # Filter Common Crawl based on Wikipedia [2019]
t5_c4()               # Filter using rules (trained T5) [2019]
gpt3()                # CommonCrawl, Wikipedia, books (trained GPT-3) [2020]
the_pile()            # Lots of sources (trained GPT-J, GPT-NeoX, ...) [2021]
gopher_massivetext()  # Filter using rules (trained Gopher) [2021]
llama()               # CommonCrawl, CCNet, StackExchange, etc. (trained LLaMA) [2022]
refinedweb()          # CommonCrawl (used to train Falcon) [2023]
dolma()               # Lots of different sources [2024]
dclm()                # Filtered using good quality classifier [2024]
```

从这张路线图能读出本章与前后的关系：BERT 与 GPT-2 分别代表「直接用现成高质量源」和「用代理信号从 Common Crawl 里筛」，紧接着的 CCNet（用 Wikipedia 训练的语言模型打分）与 C4（手工规则）则是同一问题的另外两种答案，而 GPT-3 起又开始回到分类器路线。一个需要留意的年份出入：讲义把 `bert()` 注释为 `[2019]`（意为训练出 BERT 的那批工作），而转写口述说 BERT 来自 2018；两者指的是论文年份与这段课程归类年份的差别，不宜混为一谈。另一个局限是 WebText 从未公开，外界能复现的只是 OpenWebText，因此严格说「GPT-2 的数据」与「社区用的 WebText」并不是同一份语料。

## CCNet 与 C4：语言模型打分与手工规则的两条路线 [48:30 - 52:42]

**CCNet** 是当时 Facebook 的工作，讲义标注 `[Wenzek+ 2019]`，函数写作 `def ccnet():`。转写给出的动机是「为预训练构建大规模高质量数据集」，且 "especially interested in low-resource languages"，不想要一套只对英语有效的人工流程；幻灯片把低资源语言的例子具体写成 "e.g., Urdu"（乌尔都语），这是转写没念到的。目标行原文为 "Goal: automatic way of constructing large, high-quality datasets for pre-training"。它的三个组件在幻灯片上写得很完整：去重是 "Deduplication: remove duplicate paragraphs based on light normalization"（基于轻度归一化移除重复段落）；语言识别是 "Language identification: run language ID fastText classifier; keep only target language (e.g., English)"（跑一个 fastText 语言识别分类器，只保留目标语言）；质量过滤是 "Quality Filtering: keep documents that look like Wikipedia under a KenLM 5-gram model"（在 KenLM 5-gram 模型下保留「看起来像 Wikipedia」的文档）。

```python
def ccnet():
    # CCNet [Wenzek+ 2019]
    # Goal: automatic way of constructing large, high-quality datasets for pre-training
    # Especially interested in getting more data for low-resource languages (e.g., Urdu)
    #
    # Components:
    #   Deduplication: remove duplicate paragraphs based on light normalization
    #   Language identification: run language ID fastText classifier;
    #                            keep only target language (e.g., English)
    #   Quality Filtering: keep documents that look like Wikipedia under a KenLM 5-gram model
    #
    # Results:
    #   Trained BERT models, CCNet(CommonCrawl) outperforms Wikipedia
    #   CCNet refers both to the open-source tool and the dataset released from paper
```

这里的核心直觉是把「质量」定义成一个可计算的概率：先用 Wikipedia 训练一个语言模型，再把新文档在该模型下的概率当作「有多像 Wikipedia」的分数。用 5-gram 的符号化写法表达就是 $\log P_{\text{KenLM}}(d) \approx \sum_{i=1}^{n}\log P(w_i \mid w_{i-4},\ldots,w_{i-1})$，分数越高越像 Wikipedia、越倾向保留。转写特意把它与前一章的 WebText 对比："rather than outgoing links to Reddit, they use a language model on Wikipedia"——同样是给 Common Crawl 做质量筛选，GPT-2 用 Reddit 外链这种外部社会信号，CCNet 改用语言模型这种内容本身的统计信号。好处是 "because you can get much more data, you can actually outperform training just on Wikipedia"：既然筛出来的量远大于 Wikipedia 本身，训练效果反而能超过只用 Wikipedia。幻灯片在 Results 下把这句写成 "Trained BERT models, CCNet(CommonCrawl) outperforms Wikipedia"，并高亮了一条术语澄清："CCNet refers both to the open-source tool and the dataset released from paper"，即 CCNet 既指开源工具也指论文发布的数据集，阅读文献时容易混淆。讲者还预告 "this was a tool that will later appear in some other papers"，这正是后面 LLaMA 1 用 CCNet 处理 Common Crawl 的伏笔。一个易错点是：KenLM 打分只刻画「文风像不像 Wikipedia」，并不等价于「事实正确」，因此像 Wikipedia 本身也可能含有的偏见会被一并继承。

**C4** 来自 Google 2019，讲义函数 `def t5_c4()`，标注 `[Raffel+ 2019]`，数据集全称为 "Colossal Clean Crawled corpus (C4)"。转写提醒一个常见误解：这篇论文更出名的是 **T5**（Text-to-text Transfer Transformer），它主张把所有 NLP 任务统一成 text-to-text 格式，但 "a major contribution of this paper ... was the C4 dataset"，数据集本身是同等重要的贡献。当时的共识被写成 "Observation: Common Crawl is mostly not useful natural language"：一方面是少量但高质量的小数据集，另一方面是庞大而混乱的 Common Crawl，"any attempt to train on Common Crawl just led to junk results"，于是问题变成 "How do I get a larger but high-quality subset of Common Crawl?"。已见过 Reddit 外链与 Wikipedia 式语言模型两种答案，C4 给出第三种："Let's just define a bunch of rules"，直接定义一组启发式规则，而且 "turned out to be fairly effective"。

```python
def t5_c4():
    # Colossal Clean Crawled corpus (C4) [Raffel+ 2019]
    # Paper is more famous for Text-to-text Transfer Transformer (T5),
    #   which pushes the idea of putting all NLP tasks into one format
    # ...but a major contribution was the C4 dataset.
    # Observation: Common Crawl is mostly not useful natural language
    # Started with one snapshot (April 2019) of Common Crawl (1.4 trillion tokens)
    #
    # Manual heuristics:
    #   Keep lines that end in punctuation and have >= 5 words
    #   Remove page with fewer than 3 sentences
    #   Removed page that contains any 'bad words' [article]
    #   Removed page containing '{' (no code), 'lorem ipsum', 'terms of use', etc.
    #   Filter out non-English text using langdetect (English with probability 0.99)
    #
    # End result: 806 GB of text (156 billion tokens)
```

![讲义中 def t5_c4() 的定义，高亮行为从 1.4 trillion tokens 的 April 2019 快照起步](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0039_00-50-37.jpg)

被高亮的起始条件是 "Started with one snapshot (April 2019) of Common Crawl (1.4 trillion tokens)"，即只取 2019 年 4 月的一次快照，起点是 $1.4\times10^{12}$ 个 token。规则的具体参数转写里大多只被口语化带过，幻灯片则给出了精确阈值：保留以标点结尾且词数 $\ge 5$ 的行、丢弃少于 3 句的页面、丢弃含 bad words 的页面、去掉含 `'{'`（花括号）、`'lorem ipsum'`、`'terms of use'` 等样板文字的页面、用 langdetect 以英语概率 $0.99$ 过滤非英语文本。最终结果是 "End result: 806 GB of text (156 billion tokens)"，即 $1.56\times10^{11}$ 个 token、约 806 GB 文本；转写口述说的是「156 billion tokens、800 gigabytes」，图上的 806 GB 更精确，两者应视为同一结果的精确值与近似值。规模对照很直观：这远大于 GPT-2 WebText 的 40 GB。

有两处局限特别值得记住。第一，去掉花括号 `'{'` 这条规则 "filters out a lot of code"，讲者点明 "they weren't clearly at that time they weren't thinking about code models"——2019 年还没人认真考虑代码模型，一条看似无害的样板过滤规则把大量代码一并误伤，这是启发式规则缺乏全局视野的典型代价。第二，langdetect 的 $0.99$ 阈值与「只保留英语」的设定，会把多语内容与低资源语言系统性地排除，与 CCNet 明确关注低资源语言（Urdu）的立场恰好相反，这也说明「规则派」和「分类器派」的取舍并不只是工程口味。

本章末段还附了一个 "Bonus: WebText-like dataset"，对应幻灯片上两张横轴为 `# tokens (log scale)` 的对数刻度条形图（左图按顶级域名、右图按 Website 统计，刻度从 $10^7$ 到 $10^{11}$；右图可辨认出 www.washingtonpost.com、link.springer.com、www.ncbi.nlm.nih.gov、www.apa.org 等学术与新闻站点），转写提到其中 patents（专利文档）与 Wikipedia 占比很高。文字部分写：「Filtered to pages from OpenWebText links (links in Reddit posts with $\ge 3$ karma)」，被高亮的结论是 "Used 12 dumps to get 17 GB text (WebText was 40 GB, suggesting CommonCrawl is incomplete)"，另有 "This improved on various NLP benchmarks (GLUE, SQuAD, etc.)"。转写把推理讲得更完整：用 12 个 Common Crawl dump、沿用 Reddit karma $\ge 3$ 的外链方法只还原出 17 GB，而原始 WebText 有 40 GB，"which means that Common Crawl is maybe incomplete or doesn't have everything because if it had everything, you should be able to hit 40 GB"。也就是说，这次复现意外成了一个测量 Common Crawl 覆盖率的实验：它并不包含互联网的全部网页。GLUE、SQuAD 这些基准名是画面独有的具体参数，转写只笼统说「a bunch of NLP benchmarks」。易错点在于：17 GB 与 40 GB 的差距不能直接解读为「过滤掉了 23 GB 低质内容」，讲者的解释是 Common Crawl 本身不完整，两种归因完全不同。至此三条筛选哲学——Reddit 外链代理、Wikipedia 语言模型打分、手工规则——全部登场，下一章的 GPT-3 与 The Pile 会在此基础上继续分叉。

## GPT-3 与 The Pile：分类器过滤与众包多样性 [52:42 - 56:45]

**GPT-3（2020）** 是分类器路线的代表。转写说明其构成：用了 Common Crawl 并自行处理，另有 "web text which was essentially the GPT-2 data set but expanded"（扩展版 WebText）、论文里写作 books one / books two 的两个互联网书籍语料，以及 Wikipedia；讲者特别提醒 Common Crawl 与 WebText 之间存在重叠，但 WebText 属于 "a subset of more targeted distribution"（更有针对性的分布子集）。关于 Books1、Books2，转写坦言 "this remains a mystery exactly what it is"，只说是 internet-based books corpora，来源至今成谜。讲义函数 `def gpt3():` 把这些写得更精确，并把Books 那一行高亮出来：

```python
def gpt3():
    # GPT-3 dataset [Brown+ 2020]
    #   Common Crawl (processed)
    #   WebText2 (WebText expanded with more links)
    #   (Mysterious) Internet-based books corpora (Books1, Books2)   # 高亮行
    #   Wikipedia
    # Result: 570 GB (400 billion tokens)
    #
    # Common Crawl processing:
    #   Trained a quality classifier to distinguish
    #   (WebText, Wikipedia, Books1, Books2) from Common Crawl
```

这里有几个必须区分清楚的增量。第一，图上把扩展版数据集命名为 **WebText2**，并注明 "WebText expanded with more links"，说明它不是第 2 章那份 40 GB 的 WebText 本身。第二，总量在幻灯片上写作 "Result: 570 GB (400 billion tokens)"，即 $4\times10^{11}$ 个 token、570 GB；转写口述是 "about 500 GB of text, 400 billion tokens"，570 GB 应视为精确值，500 GB 是就近的口头近似。第三，质量分类器的训练目标被写成 "distinguish (WebText, Wikipedia, Books1, Books2) from Common Crawl"——把这些已被认可的高质量语料当正例、把普通 Common Crawl 当负例，转写的说法是 "trained a quality classifier to distinguish things they deemed to be high quality from the rest"。这与 CCNet 的「像不像 Wikipedia」是同一思路的升级：不再依赖单一参照源的文风，而是用一组公认优质语料去定义边界。此外转写点出还做了 fuzzy deduplication（模糊去重），原因是 "web text and Common Crawl had dupes"——多源混合必然带来重复，这一点是前面单一来源数据集不必面对的新问题。易错点在于：$4\times10^{11}$ tokens 是混合后的总量，不是 Common Crawl 单一来源的量；把 Books1/Books2 当作已公开可查的语料也是错的，讲者明确说来源不明。

GPT-3 之后是开放复现的浪潮，讲者称 "this was I think a big event and there were a number of efforts to try to things in the open"，最初的努力之一就是 **The Pile**。转写描述它是 EleutherAI 的草根协作（grassroots effort）："a bunch of people in this Discord just jamming and figuring out high-quality sources"，即社群在 Discord 里一起挑选高质量来源，最终列出一份清单；到今天看仍相当多样，包含 Common Crawl、PubMed、Books3、arXiv、GitHub、Wikipedia、IRC、另一套书籍语料、哲学论文等。而幻灯片给出的统计表是转写完全没念到、信息量最大的一块：

![The Pile 数据集统计表，列出各组件的 Raw Size、Weight、Epochs、Effective Size 与 Mean Document Size](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0042_00-54-28.jpg)

表格共六列：Component、Raw Size、Weight、Epochs、Effective Size、Mean Document Size，共 22 个组件加一行总计。几个代表性条目：Pile-CC 为 $227.12\ \mathrm{GiB}$、权重 $18.11\%$、Epochs $1.0$、Effective Size $227.12\ \mathrm{GiB}$、平均文档 $4.33\ \mathrm{KiB}$；PubMed Central 为 $90.27\ \mathrm{GiB}$、$14.40\%$、$2.0$ 轮、$180.55\ \mathrm{GiB}$、$30.55\ \mathrm{KiB}$；Books3† 为 $100.96\ \mathrm{GiB}$、$12.07\%$、$1.5$ 轮、$151.44\ \mathrm{GiB}$、$538.36\ \mathrm{KiB}$；OpenWebText2 为 $62.77\ \mathrm{GiB}$、$10.01\%$、$2.0$ 轮、$125.54\ \mathrm{GiB}$；ArXiv 为 $56.21\ \mathrm{GiB}$、$8.96\%$、$2.0$ 轮、$112.42\ \mathrm{GiB}$；Github 为 $95.16\ \mathrm{GiB}$、$7.59\%$、$1.0$ 轮、$95.16\ \mathrm{GiB}$、$5.25\ \mathrm{KiB}$。往下还有 FreeLaw、Stack Exchange（$32.20\ \mathrm{GiB}$、$5.13\%$、$2.0$ 轮、$64.39\ \mathrm{GiB}$、$2.16\ \mathrm{KiB}$）、USPTO Backgrounds、PubMed Abstracts、Gutenberg (PG-19)†（$10.88\ \mathrm{GiB}$、$2.17\%$、$2.5$ 轮、$27.19\ \mathrm{GiB}$、$398.73\ \mathrm{KiB}$）、OpenSubtitles†、Wikipedia (en)†（$6.38\ \mathrm{GiB}$、$1.53\%$、$3.0$ 轮、$19.13\ \mathrm{GiB}$、$1.11\ \mathrm{KiB}$）、DM Mathematics†、Ubuntu IRC、BookCorpus2、EuroParl†、HackerNews、YoutubeSubtitles、PhilPapers、NIH ExPorter†，以及最小的 Enron Emails†（$0.88\ \mathrm{GiB}$、$0.14\%$、$2.0$ 轮、$1.76\ \mathrm{GiB}$、$1.78\ \mathrm{KiB}$）。总计行 The Pile 为 $825.18\ \mathrm{GiB}$ 原始大小、$1254.20\ \mathrm{GiB}$ 有效大小、平均文档 $5.91\ \mathrm{KiB}$。

从这些数字能读出两层设计：一是 Effective Size 大致等于 Raw Size 乘以 Epochs（PubMed Central $90.27\times2\approx180.55$、Books3 $100.96\times1.5=151.44$、Gutenberg $10.88\times2.5\approx27.19$），即小体量但高价值的来源被多训几轮放大权重，总计从 825.18 GiB 扩到 1254.20 GiB，约放大 1.5 倍；二是 Mean Document Size 差异极大，Wikipedia (en) 只有 $1.11\ \mathrm{KiB}$，而 Books3 达 $538.36\ \mathrm{KiB}$、Ubuntu IRC 达 $545.48\ \mathrm{KiB}$，说明单条样本的粒度完全不同，混在一起训练时上下文长度的分布很不均匀。表格里带 † 的组件（Books3、Gutenberg (PG-19)、OpenSubtitles、Wikipedia (en)、DM Mathematics、EuroParl、NIH ExPorter、Enron Emails 等）是讲义特意标注的一组，引用表格时不宜忽略这些符号，但转写没有解释其含义，此处不做猜测。易错点还有：Weight 一列是混合采样权重而非原始占比，别把 $18.11\%$ 当成 Pile-CC 在 825.18 GiB 中的体积比例。

The Pile 里讲者单独拎出几个来源。**Enron emails**：转写说 Enron 于 2002 年破产、全部邮件被公开，"this is one of the few email data sets we have, which is a weird distribution"，是少见的邮件分布；讲义补出精确参数 "Enron emails: 500K emails from 150 users from Enron senior management, released during Enron investigation (2002)"，即 $5\times10^5$ 封邮件、来自 150 位高管、在 2002 年调查期间公开，同页还写着 "arXiv: preprint for research papers since 1991 (use latex)"，呼应第 1 章的 LaTeX 选择。**Project Gutenberg** 的函数写作 `def project_gutenberg():`："Started in 1971 by Michael Hart, who wanted to increase access to literature"、"2025: ~75K books, mostly English"、"Only include books that have received copyright clearance (most in the public domain)"，被高亮的定义行是 "PG-19: books from Project Gutenberg before 2019"。转写补充了那个关键推理：PG-19 之所以几乎覆盖大部分藏书，是因为要进入公有领域 "you have to wait 75 years"，与前面版权部分的 75 年保护期直接对应。**Books3** 则被讲者描述为「约 20 万本来自影子图书馆的书」，包含 "all books from your favorite authors"，2020 年 "no one was paying attention" 时被用来训练模型，此后被下架，"you cannot use or should not use Books 3 anymore"，并预告 "We'll come back to Books 3 again"。这里的易错点是几组数字别混：Smashwords 是 2024 年 $500\mathrm{K}$ 本书，Project Gutenberg 是 2025 年约 $75\mathrm{K}$ 本，Books3 是约 $200\mathrm{K}$ 本，三者是完全不同的来源；另外 PG-19 的「19」指 2019 年之前的藏书，不是 19 本书或 19 世纪。

## Stack Exchange、Gopher 与 MassiveText：问答格式与规则派的详尽交代 [56:45 - 59:38]

**Stack Exchange** 的函数 `def stackexchange():` 在幻灯片上的信息比转写细得多。定义是 "Collection of sites of user-contributed questions and answers"，被高亮的一行给出时间与范围："Started with StackOverflow in 2008, grew to other topics (e.g., math, literature)"，即从 2008 年的 StackOverflow 起步，后来扩展到数学、文学等主题。激励机制写作 "Use reputation points and badges to incentivize participation"（用声望分与徽章激励参与），这正好解释了为什么数据里天然带可用于过滤的信号。幻灯片还逐条列出："Q&A format is close to instruction tuning / real application"、"Note: there is metadata (users, votes, comments, badges, tags) for filtering"、"Data dumps in XML (anonymized, include metadata)"。转写只提到「number of votes」这一种元数据，并说数据以 dump 形式发布、无需爬取；图上则把元数据完整写成 users、votes、comments、badges、tags 五类，并指明 dump 是匿名的 XML 且包含元数据。

这一节最值得展开的是讲者关于「涌现」的直觉。转写说：通常人们把预训练数据想象成 Wikipedia 文章那样的平铺文本（raw text），"but some parts of the web are actually look like supervised, like what you would want to ask your language model"——但网页上有些部分本来就长得像监督数据，正是你想向语言模型提问的那种形式。因此 "which I think maybe helps the model learn certain types of question answering behaviors"，模型可能正是从这类数据里学到问答行为；结论相当克制："So, not everything is like super magically emergent. It's like, well, this type of data does exist on the web"（并非一切能力都靠神奇涌现，这类数据本来就在网上存在）。换句话说，Q&A 格式把「预训练语料」与「指令微调 / 真实应用」之间的距离缩短了，这与前面几章所有以文档为单位的数据集形成对照。易错点：这只是一种合理的因果猜测，讲者用了 "maybe"，不能当成已证结论；另外 XML dump 是 anonymized 的，别指望从中拿到用户身份信息。

同页上半部分补完了上一章预告的 **Books3** 精确条目，属于画面独有："Books3 [Presser, 2020]"、"196K books from the shadow library Bibliotik"（$1.96\times10^5$ 本、影子图书馆名为 Bibliotik）、"Contained books from authors (e.g., Stephen King, Min Jin Lee, Zadie Smith)"、以及 "Has been taken down due to copyright infringement / lawsuits"（因侵权与诉讼被下架）。转写口述给的是「约 200K books，来自一个叫 Bibliotek 的影子图书馆」，图上的 196K 与 Bibliotik 拼写更精确；作者名 Stephen King、Min Jin Lee、Zadie Smith 则完全来自画面。这里可以看到第 2 章 BooksCorpus「免费≠合法」的教训原样重演，而且这次规模更大、后果更严重，也直接埋下第 6 章 LLaMA 1 的麻烦。

**Gopher / MassiveText** 是 DeepMind 2021 的工作，函数写作 `def gopher_massivetext():`，标注 `[Rae+ 2021]`。转写说 Gopher 模型从未发布、后来被 Chinchilla 取代（"sort of subsumed by Chinchilla"），但 "the description of data, I think, is actually really good. You should read this paper because I think it's very thorough in the way that they describe the data processing"，紧接着自我修正 "Well, except for the parts where they don't tell you what's in the data"。幻灯片把这份「详尽但不完整」写得一目了然：高亮的 "Components" 下列出六项——MassiveWeb: more on this later、C4、Books: no details、News: no details、GitHub: no details、Wikipedia: no details。也就是说除了 MassiveWeb 和 C4，其余四个来源的获取方式一概未披露，这正是讲者那句自嘲的落点。

```python
def gopher_massivetext():
    # MassiveText dataset used to train Gopher [Rae+ 2021]
    # The Gopher model is subsumed by Chinchilla (also never released),
    #   but the description of data is good
    # Components:
    #   MassiveWeb: more on this later
    #   C4
    #   Books: no details
    #   News: no details
    #   GitHub: no details
    #   Wikipedia: no details
    #
    # MassiveWeb filtering steps:
    #   Keep English, deduplication, train-test overlap
    #   Quality filtering using manual rules (not classifier)
    #       - e.g., 80% words contain at least one alphabetic character
    #   Use Google SafeSearch for toxicity (not word lists)
    # Result: 10.5 TB of text (though Gopher only trained on 300B tokens - 12%)  # 高亮行
```

![讲义中 MassiveText 组件与 MassiveWeb filtering steps，高亮行为 10.5 TB / 300B tokens / 12% 的结果](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0046_00-59-35.jpg)

MassiveWeb 的过滤步骤有三条，幻灯片参数比转写具体："Keep English, deduplication, train-test overlap"（保留英语、去重、检查训练-测试重叠）；"Quality filtering using manual rules (not classifier) - e.g., 80% words contain at least one alphabetic character"（用手工规则而非分类器做质量过滤，例如要求 $80\%$ 的词至少含一个字母字符）；"Use Google SafeSearch for toxicity (not word lists)"（用 Google SafeSearch 而非脏词表处理毒性内容）。转写只说了「保留英语、去重、质量过滤也基于手工规则」，并补出选规则的动机 "part of the reason for this is that they had more control over that"（对规则更有控制力）。被高亮的结果行是 "Result: 10.5 TB of text (though Gopher only trained on 300B tokens - 12%)"：数据集有 10.5 TB 文本，但 Gopher 只训了 $3\times10^{11}$ 个 token，约占 $12\%$。讲者由此点明整个领域的一条分野："there is a sort of this like division between people who wanted to use rules, um and people who wanted to use classifiers"——规则派与分类器派。把第 3 章 C4 的手工规则、第 3 章 CCNet 的语言模型打分、第 4 章 GPT-3 的质量分类器放在一起，这条线索就闭合了。易错点有三处：10.5 TB 是数据集规模而非训练用量，实际只用了 12%；Gopher 本身从未发布、且被 Chinchilla 取代，别把它当成可复现的开放模型；「no details」的四项来源不应被任何二手叙述补成具体来源。

## LLaMA 1：最后的公开数据处理与 Books3 的后患 [59:38 - 61:31]

本段收尾是 **LLaMA 1**。转写评价这份论文 "actually a very nice paper in that it detailed their data processing"，并给出一个带历史感的判断：这大概是最后一批仍详细交代数据处理、但本身并非完全开放的模型之一（"one of the last non-open, fully open models that actually talk about their data processing"）。讲义函数 `def llama():` 标注 `[Touvron+ 2023]`，逐条列出构成，其中「Project Gutenberg and Books3 (from The Pile)」被高亮：

```python
def llama():
    # Dataset for LLaMA [Touvron+ 2023]
    #   CommonCrawl processed with CCNet, classify references of Wikipedia or not
    #   C4 (more diverse; recall: rule-based filtering)
    #   GitHub: kept permissive licenses, filtering based on manual rules
    #   Wikipedia: June-August 2022, 20 languages, manual filtering
    #   Project Gutenberg and Books3 (from The Pile)          # 高亮行
    #   arXiv: removed comments, inline expanded macros, bibliography
    #   Stack Exchange: 28 largest websites, sorted answers by score
    #   Result: 1.2T tokens
```

![讲义中 def llama() 的完整数据来源条目，高亮行为 Project Gutenberg and Books3 (from The Pile)](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0047_01-00-52.jpg)

最值得细说的是 Common Crawl 那一行。转写强调："they have Common Crawl processed with CCNet"，但分类目标被改掉了——"The classification though was whether the page references was a reference of Wikipedia, not whether it is actually a Wikipedia article"。也就是说不再问「这个页面像不像 Wikipedia」（第 3 章 CCNet 的原始目标），而是问「这个页面是否被 Wikipedia 引用」。动机转写讲得很清楚："maybe Wikipedia articles are like too stylized. And we know that Wikipedia articles references a bunch of other articles, which are presumably good. So, let's call those the good websites"——Wikipedia 条目文风可能过于程式化，而它所引用的外部页面大概率也是好内容，于是把这些被引页面定义为「好网站」。这既呼应了第 3 章高亮行「CCNet 既指工具也指数据集」（LLaMA 复用了工具、换了目标），也解释了为什么 C4 被注明 "more diverse; recall: rule-based filtering"（更多样，回想一下是规则过滤），两者在混合里承担不同角色。

其余几条都能与前面各章一一对应。GitHub 一行 "kept permissive licenses, filtering based on manual rules" 呼应第 1 章「只有宽松许可的公开仓库可用于训练」；Wikipedia 一行 "June-August 2022, 20 languages, manual filtering" 给出精确时间窗与语言数；arXiv 一行 "removed comments, inline expanded macros, bibliography" 正是第 1 章「PDF 还是 LaTeX」那道选择题的答案——LLaMA 选了 LaTeX 源码并做了去注释、展开内联宏、去参考文献三项处理；Stack Exchange 一行 "28 largest websites, sorted answers by score" 则呼应上一章提到的 votes 元数据被实际用于排序。转写对这些只做了口语化概述（"a bunch of processing on GitHub to get permissive licenses"、"actually used the LaTeX to do the processing"），精确参数全部来自幻灯片。总量 "Result: 1.2T tokens"，即 $1.2\times10^{12}$ 个 token；转写同样说 "they got a 1.2 trillion tokens"，并明确 "They didn't actually release this data set"，只发布了足够详细的处理描述。

正是这份坦白带来了麻烦，也是本段最具历史意味的一段推理。转写说 Books3 "really got them in a lot of trouble because they're just announcing to world that, 'Hey, I trained on this data set.' And you look back, 'Oh, this came from. Where did it come from?' It came from the pile. 'Oh, it came from a shadow library.'"——论文公开承认用了 Books3，别人顺着溯源链一查：它来自 The Pile，而 The Pile 里的 Books3 又来自影子图书馆。讲者的结论是 "So, um you know, that's why people don't to talk about their data anymore"，这正好回扣第 1 章之前全讲给出的保密理由之一「版权责任」：披露数据来源会招致诉讼。Books3 在第 4 章还是 The Pile 表格里权重 $12.07\%$、有效大小 $151.44\ \mathrm{GiB}$ 的一个普通组件，到这里成了分水岭。

最后是复现链条：转写说 "they had enough of a description in their processing that this was reproduced by Together's Red Pajama V1 data set, which was then used in other sources"，即描述足够详细，被 Together 的 RedPajama V1 复现，之后又被其它工作沿用。这与第 2 章 WebText 未发布、由 OpenWebText 开放复现的模式如出一辙，也延续了第 4 章 The Pile 所代表的开放复现努力。几个易错点需要收束：其一，年份有三个口径——转写口述说 LLaMA 1 是 2022 年，路线图把 `llama()` 注为 `[2022]`，而本页标注 `[Touvron+ 2023]`，引用时应注明各自来源而不强行统一；其二，$1.2\mathrm{T}$ tokens 是混合总量且数据集未发布，能复现的是处理流程而非原始语料；其三，「classify references of Wikipedia or not」与 CCNet 原始的「look like Wikipedia」是两个不同的分类目标，不可混称；其四，按图上路线图，本段之后还有 `refinedweb()`、`dolma()`、`dclm()` 等 2023–2024 年的条目，那些属于本段时间范围（41:00–61:31）之外的讨论，本段只讲到 LLaMA 1 与 RedPajama V1 为止。至此，从 2018/2019 年 BERT 的 Wikipedia + Books，到 2022 年 LLaMA 1 的 $1.2\mathrm{T}$ tokens，数据采集与过滤方法完成了一轮完整演进，而「免费可得不等于合法可用」这条从 Smashwords 到 Books3 的暗线，最终解释了厂商为何集体转向对数据保持沉默。

## Books3 的余波与 RefinedWeb 的「网页即全部」 [61:31 - 63:22]

本段从一个带有历史讽刺意味的细节开场：讲者指出 RedPajama V1（Together 对 LLaMA 数据集的开放复现）最初同样包含了 Books3，后来被「stripped out」剥离掉了。他由此给出一个判断——早期围绕版权（copyright）做出的种种决定，后来产生了「a fairly big kind of watershed」（相当大的分水岭效应）。这里的直觉是：数据集的「复现」从来不只是算法与配方的复现，它同时继承了原始语料的法律风险。Books3 在 2020 年前后被收进 The Pile 时几乎无人注意，等到 LLaMA 论文公开承认用过它，才被顺藤摸瓜追到影子图书馆（shadow libraries）的源头，进而引发下架与诉讼。所以对复现者而言，删除某个来源往往不是技术判断，而是事后追认的合规判断；这也解释了为什么后续越来越多的工作不再详细披露数据来源。

紧随其后讲者转向 RefinedWeb [Penedo+ 2023]，它的核心主张被幻灯片写成一句加粗高亮的 slogan：`Point: web data is all you need`（网页数据就是你所需要的一切）。这个主张要放在本段的上下文里才读得懂：前面刚讲过 GitHub、arXiv、Stack Exchange 等一大堆「专门来源」，而 RefinedWeb 反问的是，如果干脆只用 web 会怎样——因为在某种意义上 web 就是 everything，专门来源只是 web 里被人工挑出来的高价值子集。幻灯片以伪代码函数的形式给出整个处理流水线，这些是转写里没有逐字念出来、但明确写在画面上的内容：

```python
def refinedweb():
    # RefinedWeb [Penedo+ 2023]
    # Point: web data is all you need
    trafilatura(WARC, instead_of="WET files")   # HTML → text, extract content
    filtering("Gopher rules")                  # avoid ML-based filtering to avoid biases
    fuzzy_dedup("MinHash over 5-grams")
    # Released 600B (out of 5T) tokens
```

具体数字与参数是理解这一流派的关键：它先做 HTML 到文本的转换，用 trafilatura 从 WARC（原始 HTTP 响应）而非 Common Crawl 现成的 WET 文件里抽取正文；过滤沿用 Gopher 规则，也就是保留「看起来像英语」的文档；并且**明确避免使用基于 ML 的过滤**（avoid ML-based filtering to avoid biases），理由是讲者转述的「我不想找到 web 里一个过于狭窄的子集」；再用 MinHash over 5-grams 做模糊去重。总量达到 5T（5 万亿）tokens，但只公开了其中的 600B（6000 亿）tokens。这里最容易读错的是 5T 与 600B 的关系：前者是处理并筛出的全量，后者才是实际发布、外部能拿来复现实验的部分，二者差了约 $5\text{T}/600\text{B} \approx 8.3$ 倍；另一个易错点是 WARC→文本的转换并非无损，抽取器本身就是一个会显著影响下游效果的设计选择，这一点在此前讲 Common Crawl 的 WARC/WET 差异时已经埋下伏笔。

FineWeb（Hugging Face）是紧接着的改进版，幻灯片写作 FineWeb [article]，并说明它「Started as a replication of RefinedWeb, but improved it」。讲者口述的做法是：取当时**全部** Common Crawl dumps、做过滤、依旧坚持用手工规则（manual rules）以免注入偏差、去重、再做 PII（个人可识别信息）移除，最终得到 15T tokens。把 600B、5T、15T 三个数字排在一起，就能直观看到讲者强调的「the size of these data sets is growing quite a bit」——数据集体量在快速膨胀。与前后内容的关系也很清楚：RefinedWeb/FineWeb 是「规则派」最纯粹的一次表达，它既是对上一章 LLaMA 那种多源混合配方的反驳，也为下一章 DCLM 埋下了最戏剧性的伏笔——DCLM 恰恰会把「基本就是整个网页」的 RefinedWeb 拿去当质量分类器的**负例**。一个当年刻意拒绝 ML 过滤的数据集，几年后成了别人训练 ML 过滤器的对照样本，这是本讲里很值得记住的一条线索。

![RefinedWeb 与 FineWeb 的定义、处理流水线及 5T/600B tokens 参数](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0048_01-02-14.jpg)

## Dolma 的多源混合与 Common Crawl 处理 [63:22 - 64:50]

与 RefinedWeb「只用网页」的极简主义相反，AI2 的 Dolma [Soldaini+ 2024] 走的是多源混合路线。讲者列举它的成分：自家处理过的 Common Crawl、前面讲过的 The Stack、早已熟悉的 C4，以及「a bunch of other things」；其中 Reddit 数据来自 PushShift 项目，讲者特别补了一句「at that time you could still get this data... before things got locked down」，也就是说这份语料具有时间窗口上的不可复现性；AI2 还基于自家对学术论文的抓取（Semantic Scholar）派生出一个数据集概念上，Dolma 代表的是另一种信念：与其争论哪一类来源最好，不如把各类高质量来源按可核查的方式汇总成一个开放合集，让配比与消融变得可研究。

幻灯片上那张统计表是转写完全没有念到、但信息量最大的部分，它给出每个来源的四列规模指标（单位为 UTF-8 bytes / Documents / Unicode words / Llama tokens）：Common Crawl（web pages）9,022 GB、3,370 million 文档、1,775 billion 词、2,281 billion tokens；The Stack（code）1,043 GB、210 million、260 billion、411 billion；C4（web pages）790 GB、364 million、153 billion、198 billion；Reddit（social media）339 GB、377 million、72 billion、89 billion；PeS2o（STEM papers）268 GB、38.8 million、50 billion、70 billion；Project Gutenberg（books）20.4 GB、0.056 million、4.0 billion、6.0 billion；Wikipedia 与 Wikibooks（encyclopedic）16.2 GB、6.2 million、3.7 billion、4.3 billion。合计（Total）一行是 **11,519 GB、4,367 million 文档、2,318 billion 词、3,059 billion Llama tokens**。表格下方还有两条注释：Reddit 来自 Pushshift project (2005–2023)，且 include submissions and comments separately（把发帖与评论分开处理）；PeS2o 是 40M academic papers from Semantic Scholar。用一句数量关系把它们串起来：整个合集约 $11{,}519\ \text{GB} \approx 11.5\ \text{TB}$，而按 Llama tokenizer 计约 $3{,}059\text{B} \approx 3\text{T}$ tokens，词数与 token 数之比约为 $3{,}059/2{,}318 \approx 1.32$，即平均每个 Unicode word 约合 1.3 个 token。

这张表有一个很容易被忽略的读法：bytes 与 tokens 并不成固定比例。The Stack 只占 1,043 GB（约 9% 的字节量），却贡献 411 billion tokens（约 13% 的 token 量），说明代码文本的 token 密度明显高于网页；反过来 Reddit 有 339 GB 却只有 89 billion tokens，词数与文档数也不匹配（377 million 文档只有 72 billion 词，平均每个文档约 190 词），这与社交媒体短帖的形态一致。易错点正在于此：讨论「数据集多大」时必须先说清是字节、文档数、词数还是 token 数，否则跨来源的比较没有意义；另一个易错点是 Pushshift 的时间标注 (2005–2023) 暗示这批 Reddit 语料此后不再可得，别人无法在同等条件下重新构建 Dolma。

讲者在 [64:09] 说「let's look at their Common Crawl processing」，把镜头推到 Dolma 对 Common Crawl 的具体处理，这部分同样写在后续幻灯片上而未逐条口述：Language identification (fastText classifier), keep English；Quality filtering (Gopher, C4 rules), avoid model-based filtering；Toxicity filtering using rules and Jigsaw classifier；Deduplication using Bloom filters；结果是 3T tokens。可以用带语言标注的伪代码把这条流水线固定下来：

```python
def dolma_common_crawl():
    lid = fastText_classifier()          # language identification, keep English
    quality = Gopher_C4_rules()          # explicitly avoid model-based filtering
    toxicity = rules_and_Jigsaw()
    dedup = Bloom_filters()              # probabilistic, memory-efficient
    return "3T tokens"
```

这里的技术张力很值得玩味：语言识别**用了**模型（fastText），质量过滤**拒绝**模型，毒性过滤又**混用**规则与分类器（Jigsaw），去重则采用 Bloom filters 这种概率型数据结构（省内存，但存在假阳性，可能误删）。也就是说 Dolma 并不是纯粹的「规则派」，而是一条按需挑选工具的折中路线。另一个需要澄清的口径问题是：讲者口述的「3 trillion tokens out of that」指的是从 Common Crawl 这一路处理出来的结果，而表格 Total 的 3,059 billion 是整个 Dolma 合集的 token 数，两个数字接近但统计对象不同，不能混为一谈。讲者正是在讲完这条流水线后接了一句转折——DCLM「maybe a point where this idea of model based quality filtering started to really kind of become the norm」，于是「质量过滤到底该不该用模型」这条从 Gopher/C4 一路延续下来的分歧线，在下一章被正式打破。

![Dolma 各来源的 bytes、文档数、词数与 Llama tokens 统计表](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0049_01-03-42.jpg)

## DataComp-LM 与基于模型的质量过滤成为主流 [64:50 - 67:25]

讲者把 DataComp-LM [Li+ 2024]（习惯简称 DCLM）定位为「model-based quality filtering 真正成为常态」的转折点。它的初衷带有浓厚的 benchmark 色彩：定义一套标准 pipeline，让人们能在统一框架下尝试不同的数据方法并给出可比的结果（幻灯片原文 `Goal: define a standard dataset for trying out different data processing algorithms`）。但讲者随即点出一个耐人寻味的现实——「the main way that people use this is just using their data set that they released to train models」，大家实际用的是它发布的数据集，而不是那套比较框架。这种「试验台意图、数据集产出」的错位在开源社区并不罕见：一个可复现的基准，最终价值往往落在它顺手产出的那份语料上。

流水线的起点是 DCLM-pool：把 Common Crawl 处理成**完全不过滤**的数据池，规模 240T tokens。讲者特意强调这「probably more than number of tokens that anyone really trains on」，但其中大量内容质量很低。随后逐级收窄：语言过滤（只留 English）、启发式规则、去重、最后做 model-based filtering，最终只剩约 **1.4%**。用数量关系写就是

$$240\text{T} \times 1.4\% \approx 3.4\text{T tokens},$$

这个结果与幻灯片上写的 `Result: 3.8T tokens` 处在同一个量级却不完全相等，原因恰恰藏在 Figure 4 的图注里（这句图注转写里没有念到）：「Before this pipeline, we extracted DCLM-Pool from Common Crawl with resiliparse. **Percentages are based on the total number of original documents.**」——百分比是按**原始文档数量**而非 token 数计算的，而不同文档长度差异极大，所以「1.4% 的文档」不等于「1.4% 的 tokens」。这是读这类过滤漏斗图时最容易踩的坑，也解释了为什么讲者口中的 1.4% 与幻灯片上的 3.8T 不该被硬凑成同一个等式。另外注意此处抽取器换成了 resiliparse，与 RefinedWeb 用的 trafilatura、Nemotron 后面用的 jusText 都不同，工具选择本身就是一个变量。

漏斗图（Figure 4: Construction of DCLM-BASELINE from DCLM-POOL）上还能读到一串转写未念的环节与百分比：Heuristic cleaning (Sections 4.1 & 4.2)、Deduplication (4.3)、Model-based filtering (4.4) 三大阶段，以及 English filter、URL filter、Word removal ratio filter、Bloom filter dedup、FastText filter、OH-2.5 + ELI5 等具体节点，标注的留存比例包括 50.8%、0.8%、19.9%、13.7%、12.3%、3.4%、6.2%。把 DCLM 的定义与 Dolma 的收尾放在同一张幻灯片上，正好构成对照：

```python
def dclm():                       # DataComp-LM [Li+ 2024]
    pool = process(CommonCrawl, "completely unfiltered")   # DCLM-pool: 240T tokens
    baseline = filter_down(pool, quality_classifier)       # → 1.4% of documents
    # Goal: a standard dataset for trying out different data processing algorithms
```

真正让 DCLM 出名的，是那个被讲者称作「magical classifier」「strangely good」的质量分类器。幻灯片在 Model-based filtering 一节给出了转写只讲了个大概的完整配置：Positive examples (**200K**) 是 OpenHermes-2.5（mostly GPT-4 generated instruction data，即主要由 GPT-4 生成的指令数据）与 ELI5（subreddit with curiosity questions and answers，一个专门用「像给五岁小孩解释」的方式提好奇性问题的 Reddit 子版块）；Negative examples (**200K**) 则是 RefinedWeb；得到 `Result: 3.8T tokens`；做法是 train a fastText classifier, run it on all of DCLM-pool，结论是 this quality classifier outperforms other filtering methods。讲者口述时特别停下来说「these are tied the type of questions that are in Eli5」，还形容整个配方「kind of weird, but somehow this works」。它的直觉其实可以讲通：正例代表「人类认为值得读、信息密度高的文本」，负例代表「几乎没过滤的整个网页」，于是这个线性分类器学到的不是某个领域的知识，而是一种粗粒度的「像不像高质量文本」的判别面。

这里的易错点至少有三层。第一，正负例各只有 200K 条，却要给 240T token 级别的数据池逐文档打分，训练样本与目标规模差了几个数量级，靠的是 fastText 这类线性分类器（讲者明确说「Think about this is a linear classifier」）的极低推理成本，而不是它的表达能力。第二，「好数据」的定义被完全外包给了 OpenHermes-2.5 与 ELI5 这两个偶然选中的语料，分类器必然继承它们的偏好与偏差——这恰好是 RefinedWeb 当年 avoid ML-based filtering to avoid biases 所担心的事，如今被反向接受。第三，「outperforms other things they tried」是经验性结论，覆盖了他们尝试过的若干方案，不等于理论上的最优过滤。尽管有这些保留，讲者仍承认 DCLM「for a while at least in the open community」成了质量过滤的 gold standard（金标准）——而下一章 Nemotron 的全部动机，正是从反对这份「金标准」的过滤强度开始的。

![DCLM 的 model-based filtering 配置：Figure 4 漏斗图与 200K 正负例定义](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0051_01-06-38.jpg)

## Nemotron-CC、合成数据与过滤强度的权衡 [67:25 - 71:00]

Nemotron 来自 NVIDIA，讲者用一句直白的反驳开场：DCLM filters too aggressively，把绝大多数数据都扔了（幻灯片写得更精确：`FineWebEdu and DCLM filter too aggressively (remove 90% of data)`），而「remember they only got 3.8 trillion tokens」。于是核心诉求被写成一句带俏皮拼写的 bullet：`Need moar tokens (but preserve quality)`——要更多 token，但保住质量。概念上这是把上一章的过滤漏斗往回拧：DCLM 证明了「狠过滤能出好数据」，Nemotron 要证明「狠过滤扔掉的那 90% 里还有可回收的价值」。这个 trade-off（权衡）也因此成为整个数据过滤演进史的主轴。

它的第一个技术决策就很有画面感：`For HTML → text, used jusText (not trafilatura) because it returned more tokens`。同一个 HTML→文本环节，RefinedWeb 用 trafilatura、DCLM 用 resiliparse、Nemotron 用 jusText，选择标准不是「谁抽得最干净」而是「谁返回更多 token」——连抽取器都服务于「多留数据」这个总目标。第二个决策是 Classifier ensembling（分类器集成），幻灯片上的高亮行给出了转写只讲了个大概的完整配方：

```python
def nemotron_cc():                       # Nemotron-CC [Su+ 2024]
    # Prompt Nemotron-340B-instruct to score FineWeb documents
    # based on educational value, distill into FastText model
    clf = ensemble(distill(Nemotron_340B_instruct, "educational value"), DCLM_classifier)
    low_q  = LM_rephrase("make it look like Wikipedia")
    high_q = LM_generate_tasks("QA pairs, summarize, extract key information, etc.")
    return "6.3T tokens (HQ subset is 1.1T)"
```

也就是先用大模型 Nemotron-340B-instruct 给 FineWeb 文档按「教育价值」打分、生成一批标签，再蒸馏（distill）成一个 fastText 快速模型；另一个分类器直接沿用 DCLM classifier。讲者口述的版本是「prompt our existing model to score web documents based on educational value... create a bunch of labels, train a fast text model」，与幻灯片一致，但模型名与蒸馏路径是画面独有的信息。这套「大模型标注 → 小模型蒸馏 → 全量打分」的三段式，后来几乎成为质量过滤的通用范式。

幻灯片顶部还残留着上一章那张分类器对比表的结尾，可以用来佐证 DCLM 分类器为何值得被复用：Classifier on BGE features [185] 27.2 / 14.0，AskLLM [146] 28.6 / 14.3，Perplexity filtering 29.0 / 15.0，Top-k average logits 29.2 / 14.7，而 `fastText [87] OH-2.5 +ELI5` 以 **30.2 / 15.4** 加粗排在最优——正是上一章那组 OpenHermes-2.5 + ELI5 正例、RefinedWeb 负例训出来的线性分类器。换句话说，Nemotron 并没有否定 DCLM 的过滤器，只是不肯让它单独决定取舍。

真正让 Nemotron 成为里程碑的是 synthetic data（合成数据）。讲者说它「probably not the first, but one of the main data sets that really lean into synthetic data as for pre-training」，即较早大规模把合成数据用于**预训练**的工作之一。做法分两路：被分类器判为低质量的文档，用语言模型改写（rephrase），让它「look more like Wikipedia」；被判为高质量的文档，则用语言模型反向生成各种任务，例如给定一篇 Wikipedia 文章生成 question answers、生成「please summarize this document」、extract key information 等。结果是 `Result: 6.3T tokens (HQ subset is 1.1T)`——注意讲者口述时说的是「six trillion tokens」，幻灯片给出了更精确的 6.3T，并额外标明其中的高质量子集只有 1.1T，这两个数字都是转写里没有的增量。旁边的参照行同样高亮：`For reference, Llama 3 trained on 15T, Qwen3 trained on 36T`。评测表则列出 FineWebEdu-2、FineWebEdu、DCLM、Nemotron-CC、Nemotron-CC HQ 五行在 ARC-E、ARC-C、H、W、RACE、PIQA、SIQA、CSQA、OBQA、MMLU 上的成绩，其中 Nemotron-CC HQ 在 ARC-E 78.8、ARC-C 52.9、MMLU 59.0、CSQA 55.8 等多列领先，呼应讲者说的「their high quality subset beats the previous data sets」。

讲者在这里插入了一个非常重要的方法论警告：论文里报告的 token 数**不能直接横向比较**，「It's not clear how big these unique tokens these are because when you look at token counts in language model papers some of it's like repeated. Like you do multiple epochs, two epochs that's twice the number of tokens」。把它写成一个可操作的关系式就是

$$N_{\text{report}} = E \cdot N_{\text{unique}}, \quad E = \text{训练轮数 (epochs)},$$

所以 15T 与 36T 未必意味着 15T 与 36T 条**不同**的文本，可能只是同一批语料重复了若干遍；6.3T 与 3.8T 的差距也要在这个前提下重新解读。这是读所有「某某模型训了多少 T tokens」时最该先问的问题，也是本段最容易误导人的数字陷阱。

[70:07] 之后讲者把前面所有方法收束成一句结构性的总结：这些过滤方法「all sort of look very kind of similar at some level」——取一份 web crawl，决定用 rules 还是 model，若用 model 就得先定义 what looks like good data、训一个分类器、给所有文档打分并筛选。于是权衡被压缩成一条轴：你可以拿 240T tokens 但质量很差，也可以只留约 1T 的高质量 token，而「there's sort of some sweet spot in between」（最优点在中间某处）。写成关系式即

$$\text{quality} \uparrow \;\Longleftrightarrow\; \text{tokens} \downarrow, \qquad \exists\,\text{sweet spot} \in (1\text{T},\,240\text{T}).$$

讲者用「Let me talk about two final things」把话题切向 code 与 licensing，正好说明前面这整套网页过滤的讨论到此为止，后面两个主题是补上被通用网页语料忽略的维度。

![Nemotron-CC 的合成数据配方、6.3T/HQ 1.1T 与多数据集评测表](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0053_01-09-34.jpg)

## 代码数据 The Stack V1/V2、LLVM IR 与线性化 [71:00 - 74:44]

进入代码主题时，幻灯片先给出一张「历年数据集函数总表」，把本讲前面几十分钟的内容压成一份带注释的索引，这段完全是画面独有、转写没有逐条念到的信息：

```python
t5_c4()             # Filter using rules (trained T5) [2019]
gpt3()              # CommonCrawl, Wikipedia, books (trained GPT-3) [2020]
the_pile()          # Lots of sources (trained GPT-J, GPT-NeoX, ...) [2021]
gopher_massivetext()# Filter using rules (trained Gopher) [2021]
llama()             # CommonCrawl, CCNet, StackExchange, etc. (trained LLaMA) [2022]
refinedweb()        # CommonCrawl (used to train Falcon) [2023]
dolma()             # Lots of different sources [2024]
dclm()              # Filtered using good quality classifier [2024]
nemotron_cc()       # Lots of tokens [2024]
the_stack()         # Code dataset          ← 高亮，本节主题
common_pile()       # Properly licensed data
```

这张表的价值在于它揭示了整讲的编排逻辑：从 2019 到 2024，每一行都是「一个数据集函数 + 一句方法论注脚 + 年份」，而高亮行 `the_stack()` 与紧随其下的 `common_pile()` 正是本章与下一章的两个「final things」。讲者给出的动机很朴素：早在 2022 年大家就已经清楚 coding 会非常重要，所以 The Stack 这个项目要专门做一个 really good coding data set，而不是把代码当作网页语料里的附带品。

V1 [Kocetkov+ 2022] 的处理链与数字几乎全部写在幻灯片上：Took repository names from GitHub Archive (2015-2022)；git clone'd **137M repositories, 51B files (5B unique!)**；Kept only permissively licensed (MIT, Apache) using **go-license-detector**；Remove near-duplicates using **minhash and Jaccard similarity**；Result: **3.1 TB of code**。讲者口述时把克隆数含糊说成「137 new repos」、把结果说成「one sorry 3 terabytes」，幻灯片则给出了精确值。这里最触目惊心的是那个带感叹号的括号：$51\text{B} \to 5\text{B}$，即

$$\frac{5\text{B}}{51\text{B}} \approx 9.8\%,$$

超过九成的文件是重复的（复制、fork、 vendored 依赖），这与前面讲 GitHub 时提到的「代码重复率很高」完全对应，也说明近似去重在代码语料里不是可选项而是决定性的步骤。易错点在于：去重用的是 minhash 加 Jaccard similarity 这类近似方法，阈值设得松会留下大量重复、设得紧会误删合法的代码片段；而许可证筛选依赖 go-license-detector 自动识别，「宽松许可」的判定本身有误差，这也是下一章 Common Pile 要专门讨论 license laundering 的远因。

Stack v2 [Lozhkov+ 2024] 把范围从「仓库内容」扩展到整个软件开发的元数据：Issues, comments, PRs from GitHub Archive；Repositories from the Software Heritage；Documentation from crawling websites (e.g., **PyPI, npm, devdocs.io**)。Processing 一行列出：remove binary files, malware, bot activity, deduplication, PII redaction, subsample PRs——讲者逐条口述了去二进制文件、清 malware、滤掉 PR 里大量的 bots、去重、PII reduction，以及对海量 pull requests 做子采样以保持代表性且让数据集规模可控。此外幻灯片还写着 Include existing datasets (GSM8K, code contests, StackOverflow, arXiv, Wikipedia, OpenWebMath)，即把所有能找到的「good stuff」一并掺进来，所以讲者说 Stack V2 是「mostly code and then plus other things that were used to train their coding models」。

最漂亮的一个设计是高亮的那条：`Pair source code (especially low-resource languages like Nim) with shared low-level intermediate language (LLVM)`。讲者解释得很清楚：编程语言很多，Python、C 数据极多，而 Nim 这类 low-resource languages「I hadn't even heard of」、语料稀少；于是把代码**编译**成低层中间表示，凡是 C compiler 之类都能编译到它，再把低资源语言源码与 IR 并置（juxtaposed）：

$$\text{LLVM IR} = \text{compile}(\text{source}_{\text{any language}}), \qquad \text{source}_{\text{low-res}} \leftrightarrow \text{IR}_{\text{shared}}.$$

模型便能借数据丰富的共享低层表示，学到「数据稀缺语言 ↔ IR」之间的映射，等于用一种语言的海量数据去补另一种语言的稀缺数据。这是把「对齐」思路用到跨语言迁移上，比简单上采样低资源语言要讲究得多；局限是它依赖编译器前端的可用性，且 IR 与源码在语义上并非一一对应。

最后一个必须解决的问题是 linearization（线性化）。讲者指出 pull requests 及其元数据「is not by its nature linearized sequence」，所以首先要决定 how much context to provide：一个 event 可能只是「I changed one line of code」，你可以只给那一行、给前后几行、或者给 the entire file surrounding that diff，这些都是明确的设计决策。幻灯片给出的最终形态是一段 XML-like 结构化 token 序列，其标签原文是转写完全没有念到的独有内容：

```xml
<pr_title>title\username_0: description
<status>opened
<repo_name>reponame
<pr_base>
<pr_file>filepath_1
<pr_base_code>file_content/changes_1
...
<pr_file>filepath_N
<pr_base_code>file_content/changes_N
<pr_diff>
<pr_file>filepath_1
<pr_diff_hunk>diff_hunk_1
...
<pr_diff_hunk>diff_hunk_K
```

右半列则是评论与评审事件：`<pr_comment>username_id: comment`、`<pr_event_id>comment_id`、`<pr_review>username_id: review_comment\n`、`<pr_event_id>review_id`、`<pr_review_state>[approved, rejected, commented, changes_required]`、`<pr_review_comment>`、`<pr_in_reply_to_review_id>review_id (opt)`、`<pr_in_reply_to_comment_id>comment_id (opt)`、`<pr_file>filepath`、`<pr_diff_hunk>comment_line>line_number`、`<pr_diff_hunk>diff_hunk<content` 等，配合 bullet「Add some inline context (e.g., file surrounding diff), subsample」。由此可见线性化不是简单拼接：它用一套自定义标签把「PR 标题/状态/仓库/基线文件/diff hunk/评论/review 状态/回复关系」压成单一 token 序列，可选字段用 `(opt)` 标注。讲者的结论因此是：模型不只是在学 how to generate code，也在学 the software development process around code。易错点也在这里——任何线性化都会丢失原对象的图结构（回复链、review 层级），标签设计就是在不丢失与不冗余之间取平衡；同时 subsample 是为了规模可控，被抽掉的 PR 会让分布产生偏差。

![The Stack V1/V2 的完整处理链与 LLVM 低资源语言配对方案](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0055_01-12-37.jpg)

![Pull requests 线性化为 XML 式 token 序列的标签结构示例](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0056_01-14-19.jpg)

## Common Pile——只用许可明确的数据能走多远 [74:44 - 79:29]

讲者把话题切回本讲开头就铺好的版权框架：互联网上几乎所有内容都是 copyrighted，其中一部分是 permissively licensed（宽松许可），少数已进入 public domain（公有领域）。你固然可以 appeal to fair use 说「I'm going to train on it anyway」，但「this is not quite settled」——这在法律上并未尘埃落定。于是如果采取极度风险规避的态度，判断规则就变成讲者复述的那句「if I don't know whether it is okay, that's a no」。Common Pile [Kandpal+ 2025] 正是这种态度的产物，幻灯片把它写成一个加粗的大问句：`Key question: can you train a good model using only permissively-licensed data?`——注意问的是「能训出多好的模型」，也就是「how far can you get」，而不是能不能训出来，措辞本身就预留了保守的结论。

它的做法是搜遍互联网，把所有「值得训练且许可明确」的数据类型都收进来：转写里讲者点到 Stack V2 与 code、政府公文（很多是宽松许可）、wikis、部分网站、部分新闻站点、学术论文、在线论坛、公有领域内容、教育资源等等；而幻灯片那张对数坐标柱状图（Dataset Size，刻度 1MB / 100MB / 10GB / 1TB）把每个类别的具体字节数都标了出来，这是最有价值的独有增量：Code (**4775 GB**) 含 Stack V2、PEPs；Government & Legal (**1172 GB**) 含 USPTO、CAP、USGPO、UK Hansard、Regulations.gov；Wikis (**528 GB**) 含 Wikiteam、Wikimedia；Web (**269 GB**) 含 CCCC、News、Foodista、PDR；Academic Papers (**378 GB**) 含 peS2o、PubMed、arXiv Papers、arXiv Abstracts；Online Forums (**165 GB**) 含 Stack Exchange、GitHub Archive、Ubuntu IRC；Public Domain Books (**244 GB**) 含 BHL、Pre-1929 Books、Library of Congress、Project Gutenberg；Other (**79 GB**) 含 CC YouTube、DPI；Educational Resources (**15 GB**) 含 DOAB、PressBooks、LibreTexts、OER Commons。图下写着 Collected **8TB** dataset of permissively licensed data。把这些分项加总可以验证口径：

$$4775 + 1172 + 528 + 269 + 378 + 165 + 244 + 79 + 15 = 7625\ \text{GB} \approx 8\ \text{TB}.$$

这个 8TB 要放到本段的数字谱系里才有意义：DCLM-pool 是 240T tokens、FineWeb 15T、Nemotron-CC 6.3T，而完全合规的数据只有 8TB——讲者的评价是「actually pretty good for permissively licensed data」，即对一个自我设限如此严格的合集来说已经相当可观，但和主流预训练语料不在一个数量级。

真正难的地方被讲者概括为「doing this project is actually much harder than maybe it sounds at first glance」：并不是看一眼 license 标签、认出 Apache 或 CC 就可以放行。幻灯片把这三条 Subtleties 逐字列了出来，措辞比口述更精确：

```python
subtleties = [
    "License laundering: redistribute copyrighted work under permissive license (hard to detect)",
    "Collection licenses (Dolma is ODC-By) doesn't extend to individual",
    "Synthetic data from LMs trained on unlicensed data is unclear",
]
```

第一条 license laundering（许可洗白），讲者解释为人们对待许可很随意，可能把一份受版权保护的作品随手「slap a CC BY on it」，而「Anyone can write this on the internet」，很难判断真伪。第二条是合集许可不延伸到单件作品：讲者举例 Dolma（AI2 的合集）本身「is permissibly licensed」，但 collection licenses don't extend to the individual works；他进一步警告「you can't just look at data sets on Hugging Face」——很多数据集标着宽松许可，深挖下去在单作品层面并不成立。幻灯片补出的 `Dolma is ODC-By` 这一具体许可名是转写没有的信息。第三条直接呼应上一章 Nemotron 的合成数据路线：Common Pile 主动决定 forego training on any synthetic data，因为「training language models on synthetic unlicensed data is unclear」。讲者把这层顾虑说得很透彻：开放权重模型多以 MIT license 发布，名义上「you can do whatever」，但这些模型本身「presumably also trained on unlicensed data」，所以严格讲用它们生成合成数据仍是「a little bit of kind of data laundering if you are really kind of honest」。这里的易错点是：合规性不能靠数据集层面的标签继承，必须逐作品核查；而「模型输出」并不因为模型开源就自动洗清上游语料的许可问题。

评测结果由幻灯片上的柱状图给出，图例为 Comma v0.1-1T（黄色、带星标，即 Common Pile 训出的模型）、LLaMA、MPT、RPJ-INCITE、Qwen3（斜线填充），纵轴 Performance 0–100，横轴分两组：Knowledge/Reasoning 下有 ARC-C、ARC-E、MMLU、BoolQ、HSwag、OBQA、CSQA、PIQA、SIQA，Coding 下有 HumEval、MBPP。讲者口述的比较对象是「the first LLaMA、MBT（应为 MPT）and Qwen」，结论是「not bad」——明显胜过 2023 年那批很老的模型，但「not nearly as good as the Qwen models, for sure」。幻灯片底部的高亮行把这个判断写成了 `Can do decently, but tough to compete without more tokens`；讲者还补了一句「I don't think this is the final word on it, and I think you can probably eke more out of publicly permissive licenses if you try」，即这不是终论，若更努力地挖掘宽松许可数据，应该还能榨出更多。有意思的是幻灯片最下方淡出的 `if __name__ == "__main__": main()`——整堂课的数据集其实是被组织成一份伪代码脚本来讲的，Common Pile 是它的最后一个函数。这与本章开头「不确定就等于不行」的严苛立场形成呼应：结论不是「合规数据训不出模型」，而是「在不增加 token 的前提下很难竞争」。

![Common Pile 各类许可明确来源的规模分布与 8TB 总量](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0057_01-16-02.jpg)

## 全课总结与数据处理的研究空间 [79:29 - 82:01]

讲者用「to summarize this lecture」收束全课，第一句就回到最朴素的纠正：希望听众现在意识到 data is a very rich topic，「It's not that data just falls from the sky, or you just go on Hugging Face and you download a data dataset」。幻灯片上的 Summary 把这句写成加粗的 `Key lesson: Data does not fall from the sky. You have to work to get it.`，后面几条依次是 `Live service → raw data → processed data (transformation, filtering, deduplication)`、`Data is the key ingredient that differentiates language models`、`Legal and ethical issues (e.g., copyright and privacy)`。讲者进一步强调数据同时有 technical 与 social context：最上层「the internet is a bunch of live services」，得有人做「producing raw」这份苦活——或者由网站提供 dumps，或者有人去写 crawler，或者别的什么方式。这正好把本讲开头的爬虫、Common Crawl、版权三段全部收进同一张图里。

第二个重点是处理环节决定质量。讲者说 filtering、transformation、deduplication「All of these things have an impact on the quality of your final language model」，而其中 filtering「is probably one of the most kind of important things」，理由是一个巨大的数量落差：

$$\frac{3\text{T}}{200\text{T}} = 1.5\%, \qquad \text{即 } 200\text{T tokens} \to <3\text{T tokens}.$$

他形容「How do you go from 200 trillion tokens to less than like 3 trillion tokens... is a huge reduction, which merits a lot of attention」。把 1.5% 与上一章 DCLM 的 1.4%、Nemotron 抱怨的 remove 90% of data 放在一起，就能看出整堂课反复回到同一个工程问题：从数百 T 的原始网页里，究竟该用什么依据筛掉 98% 以上。第三个重点是差异化：很多语言模型「are roughly the same kind of transformer architecture」，架构趋同，但 data「depending on how you process it can make a pretty big difference」——所以在开放权重、架构透明的今天，数据反而是真正拉开差距的那一项，这也回头解释了本讲最开头「为什么厂商肯公开架构却不谈数据」。

讲者随后坦承这个领域的不成熟：legal and ethical issues「more than I can get into this lecture」，而且 this process is very messy。他把它和本课其他部分作对比——别的内容「maybe more based on kind of first principles」，而 data processing「is right now, at least, a lot just based on kind of vibes」：你定义一个 classifier、定义一条 rule、设一个 threshold，如此而已。用关系式把这层「凭感觉」写出来，就是

$$\text{dataset} = \text{filter}_{t}(\text{crawl}), \qquad t\ \text{为人工设定的阈值，缺乏第一性原理依据}.$$

正因为 $\text{filter}_{t}$ 里的规则、分类器与阈值都缺少原则性指导，讲者才说「there are many opportunities to improve」，并把话直接递给听众：做 assignment four（作业四）时可以想想「whether there's better ways to do this」，这甚至可能是一个 research direction（研究方向）。这一句把整堂课从「梳理现状」翻转成「开放问题」，也让前面 DCLM 那个「strangely good」的 fastText 分类器、Nemotron 的蒸馏打分、The Stack 的线性化决策都有了新的读法——它们都是可以被系统性改进的临时方案，而非定论。

片尾一句「that's it for today」之后，讲者交代下次会继续讲 data：「I'll do talk a bit about post-training data and a bit more about filtering」，即后训练数据与更多过滤话题，恰好与本讲主要覆盖预训练数据形成衔接。此时的画面也值得记一笔：讲者站在白板前做总结，右侧投影幕是那张 Summary 幻灯片，而白板上残留的是上一讲的数学推导板书（可见 $V = m s\dot{s} + \frac{1}{\gamma}\hat{m}\dot{m}$、$=S(m\dot{s}-\dot{m}V) = -\lambda m s^{2}$、$\hat{m}=\hat{m}-m$ 以及标号 #2 等手写式子），与本讲的数据主题无关，只是教室里尚未擦掉的痕迹；把它和屏幕上「Live service → raw data → processed data」的规整流程并置，倒也很贴合讲者最后那句「this process is very messy」。

![讲者在白板前做全课总结，投影幕显示 Summary 要点](/coursenotes/cs336-language-modeling-sp26/l13-data-sources/kf_0060_01-21-36.jpg)

## 要点回顾

### 数据的地位与训练阶段

- 核心断言 "Data is the most important thing to get right in training language models" 是价值判断而非定理；Llama 3 [Grattafiori+ 2024] 对架构与流程透明，对数据只说 "a variety of data sources"，保密理由是 competitive dynamics 与 copyright liability。不过论文 3.1 节仍列了去重、清洗、移除含大量 PII 与成人内容的域名及知识截止 2023 年底，只是不谈数据源到底是什么。
- 数据是长尾且随人力扩展："Data is fundamentally a long-tail problem, scales with human effort (unlike architectures, systems)"，这解释了各家数据团队规模庞大。
- 训练分 pre / mid / post-training 三段，趋势量减质增：$|D_{\text{pre}}| \gg |D_{\text{mid}}| \gg |D_{\text{post}}|$ 且 quality 递增；base model = pre + mid，instruct/chat model = post-training 之后，但 Qwen3.5-397B-A17B 只发 instruct。
- OLMo 2 1124 Mix 里 DCLM-Baseline 有 3.71T tokens、21.32T bytes、2.95B 文档，字节数远大于 token 数，说明二者并非一回事。

### 网页从哪来与获取限制

- "在整个互联网上训练" 类型不对，Web 是活服务器，`$ curl https://cs336.stanford.edu/` 一次请求换一个响应，"You can't train on live servers"。
- crawler 从 seed set 做 Discovers webpages 与 Downloads the discovered webpages 两步，且 "someone builds a crawler, not necessarily you"；Discord、wandb 等应用构成 deep web，Facebook/X/NYTimes 是 walled gardens。
- robots.txt 是 voluntary 而非法律，Cloudflare 的 CAPTCHA、IP 封锁、rate limits 是技术障碍，ToS 与 license 才是法律层；Decline of consent [Longpre+ 2024] 考察 C4、RefinedWeb、Dolma 后得出限制随时间增加，Full restrictions 到 2024 年目测近 50%。若把完全限制比例记作 $r_{\text{full}}(t)$，其形状是 2023 年前平稳、2024 年跃升到约 0.5。
- 不守规矩的爬虫曾 "a million times in 24 hours"（约 11.6 次/秒）；影子图书馆 LibGen、Sci-Hub 等绕过付费墙、法律上是 piracy，规模为 LibGen ~4M books (2019)、Sci-Hub ~88M papers (2022)。

### 版权法框架与合法路径

- 母问题 "What data is legal to use (for training)?" 未定论；版权溯至 1709 年 Statute of Anne，美国近代是 Copyright Act of 1976，保护判据为原创加固定 $\text{protected}(x) \iff \text{original}(x)\wedge\text{fixed}(x)$，duration = 75 years。
- 版权保护表达不保护思想（quicksort），合集原则上不可版权，注册费 $65 仅作起诉前提，结论是 "basically everything on the Internet are copyrighted"。版权 75 年后到期进入 public domain，例子有莎士比亚、贝多芬与 Project Gutenberg。
- 合法路径 $\text{legal-use}(x)=\text{license}(x)\vee\text{fair-use}(x)$：许可出自合同法，是 licensor 授予 licensee 的 "a promise not to sue"；Creative Commons 由 Lessig、Eldred 于 2001 年创立，弥合 public domain 与现行版权。
- CC 覆盖 Wikipedia、Khan Academy 及 307 million Flickr 图像、10 million YouTube 视频；厂商直接签约，如 Google–Reddit、OpenAI–Shutterstock、OpenAI–StackExchange。

### 合理使用四要素、语义本质与诉讼

- fair use 法源是第 107 条，四要素 $\langle f_1,f_2,f_3,f_4\rangle$ 为 purpose/character（transformative、教育优先）、nature（事实优先）、amount & substantiality（snippet 优先）、market effect，都是 tendencies 而非硬规则。
- Authors Guild v. Google（2002–2013）历时 11 年、snippets 判 Google 胜诉；"copyright is not about verbatim memorization"，版权关乎 $\text{semantics}+\text{economics}$ 而非 n-gram overlap，情节角色（Harry Potter）可版权、parody 更可能合理使用。
- 套到模型：copying 即潜在侵权，training 直觉上 transformative、学 general idea（wizards）而非 concrete expression（Harry Potter），但 $f_4$ 市场冲击最不利，fair use "a little bit slippery"；ToS 另行叠加，YouTube 的 CC 视频仍禁 bot 下载。
- 三起诉讼：NYT v. OpenAI (2023) 仍 pending；Bartz v. Anthropic 2025 判训练 fair use、pirating 违法，买书扫描合法却洗不掉盗版，以 $1.5B 和解约 $3,000/book；Kadrey v. Meta 训练 fair use、torrenting pending，结论仍是 narrow、"Pirating books is clearly illegal"。

### Common Crawl、专门来源与数据投毒

- Common Crawl（2007）每月 3–5 billion pages、累计 300 billion pages（讲者质疑 3–5B×20≠300B）；单个 dump 约 2B pages / 372 TB（纯文本），Google 索引 ≥ 100 PB。
- 抓取以数亿 seed URLs 做图遍历，配 selection / politeness / re-visit 三策略；WARC 存原始响应、WET 是 lossy 文本，HTML→文本用 trafilatura、resiliparse，CORE 得分 24.5 / 24.1 高于 WET files 20.7（[Li+ 2024]，2406.11794）。
- Wikipedia（2001）截至 May 2026 有 67M articles、361 语言版、定期 dump 无需爬；数据投毒 [Carlini+ 2023] 利用 $t_{\text{inject}}<t_{\text{dump}}<t_{\text{rollback}}$，回退前快照已带走恶意内容。
- GitHub（2008）420M+ repos、仅 permissive license（MIT、Apache）可训练，应走 git protocol 而非 scrape，Metadata 由 GitHub Archive 提供；Software Heritage（2016）只管仓库，arXiv 约 3M submissions、元数据 CC0、从 Amazon S3 批量下载。

### 数据集演进与过滤哲学

- BERT [Devlin+ 2018] 用 Wikipedia + Books，贡献在 "sequences are documents rather than sentences"；GPT-2 WebText 取 Reddit karma ≥ 3 外链得 40 GB、约 $10^6$ 页、从未发布，复现为 OpenWebText。
- CCNet [Wenzek+ 2019] 用 fastText + KenLM 5-gram 打分 "looks like Wikipedia"；C4 [Raffel+ 2019] 从 1.4 trillion tokens 出发用手工规则（去 '{'、langdetect 0.99）得 806 GB / 156 billion tokens。
- GPT-3 [Brown+ 2020] 训 quality classifier 区分 (WebText, Wikipedia, Books1, Books2) 与 Common Crawl 得 570 GB / 400 billion tokens；The Pile 众包 22 组件、825.18 GiB→1254.20 GiB，Books3 权重 12.07%。其 Effective Size 约等于 Raw Size 乘 Epochs（Books3 100.96×1.5≈151.44 GiB），小体量高价值源被多训几轮放大。
- Gopher MassiveText 10.5 TB 只训 300B tokens（12%）属规则派；LLaMA 1 [Touvron+ 2023] 把 CCNet 改成 "classify references of Wikipedia or not"、1.2T tokens，因用来自 Bibliotik 的 Books3（196K books）惹诉，被 RedPajama V1 复现。

### 现代过滤、代码数据与合规数据

- RefinedWeb [Penedo+ 2023] 主张 "web data is all you need"，trafilatura 抽 WARC、Gopher 规则、MinHash over 5-grams、avoid ML-based filtering，5T 只发布 600B；FineWeb 得 15T tokens。
- Dolma [Soldaini+ 2024] 合计 11,519 GB、3,059 billion tokens，CC 处理用 fastText + Gopher/C4 规则 + Jigsaw + Bloom filters 得 3T；DCLM [Li+ 2024] 从 240T pool 收到 1.4% of documents、3.8T tokens，正例 200K OpenHermes-2.5 + ELI5、负例 200K RefinedWeb 训 fastText。
- Nemotron-CC [Su+ 2024] 反对 "remove 90% of data"，jusText 抽取、蒸馏 Nemotron-340B-instruct 的教育价值评分并集成 DCLM classifier，rephrase 低质 / 生成 QA 高质，得 6.3T（HQ 1.1T）；报告 token 须按 $N_{\text{report}}=E\cdot N_{\text{unique}}$ 看重复轮数。讲者把取舍压成 $\text{quality}\uparrow\Longleftrightarrow\text{tokens}\downarrow$、最优点落在 1T 与 240T 之间，参照 Llama 3 15T、Qwen3 36T。
- The Stack V1 clone 137M repos、51B→5B unique files（≈9.8%）、go-license-detector + minhash/Jaccard 得 3.1 TB，V2 用 LLVM IR 配对低资源语言、PR 线性化为 `<pr_title>`、`<pr_diff_hunk>` 标签；Common Pile [Kandpal+ 2025] 约 8 TB（Code 4775 GB），警惕 license laundering、ODC-By 合集许可不及单件、合成数据不清，Comma v0.1-1T "tough to compete without more tokens"。
- 全课总结回到 "Data does not fall from the sky. You have to work to get it."：200T→<3T tokens（1.5%）的过滤决定质量，架构趋同时数据是差异化关键，而 $\text{dataset}=\text{filter}_t(\text{crawl})$ 仍 "based on vibes"、缺乏第一性原理，是作业四与研究方向。

## 视频来源

- 原始链接：https://www.youtube.com/watch?v=-qm0ln33G24
- 课程：Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 13

---

## 对照参考笔记补遗（Spring 2025 版）

> 编者注：本节依据 2025 版参考笔记整理，其中的「更正」小节记录的是核对时发现的问题，对应正文已在本版订正；补遗内容不代表 2026 课程视频原文。


本节内容来自 Spring 2025 版参考笔记，仅用于与本讲（Spring 2026 视频）对照补充，不代表 2026 视频原文；凡与正本表述冲突处，一律以正本为准。

### 公共抓取的工程基准与课堂问答

参考把 Common Crawl 写成一个 2007 年成立的非营利组织：到 2025 年已积累约 100 次抓取（~100 crawls from 2008–2025），大致每月一次，各次之间有重叠但会主动 diversify。它给了一个很实在的量级锚点——2016 年跑完一次抓取需要 10–12 天、动用 100 台机器；到 April 2025 那次抓取则收录 $2.74\times10^{9}$ 个网页、未压缩内容 468 TiB。把它与正本给出的单 dump 规模、以及 Google 索引 $\ge 100\ \mathrm{PB}$ 并排看，就能理解讲者为什么说 Common Crawl 覆盖稀疏：即便按月累计，公共可抓的网页文本仍只是搜索索引里很小的一块，这也解释了为什么多数厂商宁可自建爬虫、不肯只依赖公开快照。注意这里的 468 TiB 与正本的 372.5 TB 是同一 crawl 的不同口径（未压缩全量 vs 纯文本），不是谁算错了，引用时须注明口径。

抓取框架本身也写得很具体：底层用 Apache Nutch，从至少数亿（hundreds of millions）个 seed URLs 出发，把页面里的 hyperlinks 并回队列，是一种 BFS-like 的图遍历；在此之上才有 Selection policy（下载哪些页）、Politeness policy（尊重 robots.txt、不压垮服务器）、Re-visit policy（多久复查一次页面变更）三条策略。易错点在于 robots.txt 只是 guideline 而非 formal enforcement：网站可以用它 opt-out，但不能指望所有爬虫都遵守；另一个易错点是「URLs are dynamic」——同一 URL 可能因浏览器状态返回不同内容，不同 URL 又常指向同一内容，重复（duplication）与镜像站因此是抓取阶段就自带的产物，而不是后续过滤才冒出来的问题。这一点与正本里 WARC/WET 抽取、模糊去重那条线是同一个根因的上下游。

2025 版还有四则课堂问答，补上了正本没有的口径。问 Common Crawl 是否过滤攻击性内容，答案是非常 permissive——「offensive」属于高层语义判断，只会拉黑少数违法站点，所以语料里很可能含大量有害内容；问能否退出，答案是可以写 robots.txt，但同样只是约定；问图像与媒体怎么处理，答案是抓到的是 raw HTTP response（原始 HTTP 响应），理论上含图像，只是整体偏向文本，专门媒体需要另建 crawler；问有多大比例受版权保护，答案是 Most of it。这四句把正本「多层限制」的图景从「能不能抓」延伸到「抓到的是什么」，也预告了后面整段版权讨论：内容默认受版权保护，正是合法路径必须从许可与合理使用讲起的原因。

### 早期语料的精确参数与来源版图

BERT 用的 BooksCorpus 在正本里只交代了平台背景，参考补出了语料自身的规格：它出自 2015 年一篇 vision-language 论文 BooksCorpus (Zhu+ 2015)，抓取 Smashwords 上定价 $\$0$ 的自助出版书，共 7,000 本、985 million words；而 Smashwords 在 2024 年约有 150,000 位作者与 500,000 本书。这两个数字必须分清：$5\times10^{5}$ 是平台规模，7,000 是实际入语料的书量，把二者混成「BooksCorpus 有 50 万本书」是典型误读。2015 年 AI 版权尚无人关注，这套语料后来因违反 Smashwords 的服务条款被下架，正是由此引出全讲「免费可得不等于合法可用」的第一条教训，也与后面 Books3 的下架构成同构。

C4 之后有一项常被忽略的来源分析 Dodge+ 2021：按 top-level domain 与 website 统计 token 分布，.com 与 .org 最突出，头部站点是 patents.google.com、en.wikipedia.org、news.google.com、books.google.com，结论是 C4 由多样来源组成、并非 Wikipedia 的放大版（not just Wikipedia）。正本把同类对数坐标条形图放在「Bonus: WebText-like dataset」小节里，读的时候容易只当成 WebText 复现的附属；补上这一归属才知道它回答的是「C4 里到底有什么」。这也回头解释了 LLaMA 配方里为什么把 C4 注成 more diverse、与走 CCNet 路线的 Common Crawl 承担不同角色。

The Pile 的 22 个 high-quality domains 在参考里是按五类组织的：Academic 含 PubMed Central、ArXiv、USPTO、NIH、PMA、FreeLaw；Internet 含 Pile-CC、OpenWebText2、StackExchange、Wikipedia、GitHub、DM Mathematics、Ubuntu IRC、HackerNews、YouTube Subtitles；Prose 含 Books3、PG-19、BookCorpus2、EuroParl、NIH Exporter、Enron Emails；Dialogue 含 OpenSubtitles、MusicBrainz；Misc 含 Bibliotik、BC2。几个采集口径也值得记住：Pile-CC 用 WARC 经 jusText 转文本，PubMed Central 约 5 million papers（NIH 资助成果强制公开），arXiv 自 1991 年起且用 LaTeX，GitHub 2008 年上线、2018 年被 Microsoft 收购，其事件流由 GH Archive 以 hourly snapshots（小时级快照）形式放在 Google BigQuery 上。参考给 The Stack (Kocetkov+ 2022) 的 takeaway 尤其该和前后连起来读：代码往往比网页有更清晰的许可证（Code often has clearer licenses than web pages），所以听到「在 GitHub 上训练」时要追问具体的预处理步骤，而不是默认存在一份现成语料。

### 开放复现链与轻量过滤的具体阈值

LLaMA 1 的 $1.2\mathrm{T}$ tokens 之所以能被复现，参考给出的是一条三级链条：Together 的 RedPajama v1 按论文描述复现；Cerebras 的 SlimPajama 用 MinHashLSH 去重得到 627B 子集；RedPajama v2 则基于 84 个 CommonCrawl snapshots、30T tokens，只做 minimal filtering（最少过滤）却附带大量 quality signals（质量信号），定位是供研究「如何按质量信号过滤」的资源而非成品训练集。正本只写到 RedPajama V1 为止，补上后两级才看得出「描述足够详细」的真正价值：它先让配方被复现，又让复现品变成可再做消融的实验台。

FineWeb 的参数同样被正本一笔带过。它是 RefinedWeb 的复现与改进（Started as a replication of RefinedWeb, but improved it），用了 95 个 Common Crawl dumps；除 URL filtering 外有明确的 language ID 阈值 keep if $p(\text{en}) > 0.65$；过滤沿用 Gopher、C4 与更多手工规则；模糊去重用 MinHash；并对 email 与 public IP addresses 做 PII anonymization（个人可识别信息匿名化）；最终得到 15 Trillion tokens。参考把它定性为 lightly filtered dataset（轻过滤数据集）、可再叠加 model-based filtering，这一点正好是正本比较「FineWeb 15T 与 DCLM 3.8T」时被省略的前提：两者不在同一过滤强度上，数字大小不能直接当成质量高低，更不能把 15T 当成「已经狠筛过」的结果。

把这几条接回正本的过滤主线，易错点会更清楚：别把「95 dumps」或 $p(\text{en})>0.65$ 这类阈值当成普适最优值，它们只是某一次工程取舍；正本结尾已指出整条流水线多半凭规则、分类器与人工阈值、缺乏第一性原理依据，这里补上的只是那组阈值在 FineWeb 里的具体取值。另一个易错点是混淆 RedPajama v2 的 30T 与成品语料：它刻意保留大量低质内容并附质量信号，用途是让研究者比较过滤算法，直接拿它训练的语料去和 15T、3.8T 比分数并不公平。

### 质量过滤的量化对照

DCLM 之所以能宣告 model-based filtering 胜出，靠的是一张 1B-1x scale 的量化对照表，参考给出了完整的 CORE / EXTENDED 两列分数：

$$
\begin{array}{lcc}
\text{Filter} & \text{CORE} & \text{EXTENDED}\\
\hline
\text{RefinedWeb reproduction} & 27.5 & 14.6\\
\text{Top 20\% by Pagerank} & 26.1 & 13.9\\
\text{Classifier on BGE features} & 27.2 & 14.0\\
\text{Perplexity filtering} & 28.6 & 15.0\\
\text{AskLLM (146)} & 29.0 & 14.3\\
\text{SemDedup (1)} & 29.2 & 14.7\\
\text{fastText (87) OH-2.5 + ELI5} & \mathbf{30.2} & \mathbf{15.4}
\end{array}
$$

最优的仍是正例取 OpenHermes-2.5 与 ELI5、负例取 RefinedWeb 训出的 fastText 线性分类器（30.2 / 15.4）；而 PageRank、perplexity 这类单一代理信号明显落后，超过 29 分的方法几乎都带模型信号。参考据此写下了一句很有分量的判断：这篇之后 AI2 的 OLMo 2 改用 DCLM-baseline，「避免模型偏差」的顾虑大体被「追求更高 benchmark 分数」取代——这与正本记的「strangely good 的 fastText 分类器成了 gold standard」是同一件事的两种写法。Nemotron-CC 侧另有一个可作总括的数字：其高质量子集 Nemotron-CC-HQ 在评测里取得最高平均分 60.1。

易错点有两层：一是这些分数是 1B-1x 规模下的相对比较，不能外推成任意规模下的理论最优，正本也反复强调「outperforms other things they tried」只是经验结论；二是正本残留的那五行分类器对比把 AskLLM 与 Perplexity 的分数对调了，引用时应以上表为准，别把 28.6 安到 AskLLM 头上、也别把 29.0 安到 Perplexity 头上。补这张表的意义不在多记七个数字，而在于让「规则派让位给分类器派」这一转折有了可核查的依据。

### 长上下文与后训练数据的专门策略

2025 版在版权讨论之后单列了 Long context，这是正本完全没覆盖的主题。参考给的当代窗口是 DeepSeek v3 的 $128\mathrm{K}$、Claude 3.5 Sonnet 的 $200\mathrm{K}$ 与 Gemini 1.5 Pro 的 $1.5\mathrm{M}$ tokens；但 Transformer 随序列长度按 $O(L^{2})$ 二次方扩展，长上下文训练极其昂贵，因此不是从预训练一开始就做，而通常放到 mid-training 再加。这恰好补上正本三阶段模板里 mid-training 的一处空缺：中期训练「换成更高质量数据以增强特定能力」里的「特定能力」，长上下文就是最具代表性的一项，且与正本「量减质增」的梯度一致——长文档是少量、高价值、后加的。

具体做法是 LongLora (Chen+ 2023)，用 shifted sparse attention（移位稀疏注意力）与 positional interpolation（位置插值）把 LLaMA 2 7B 的上下文从 $4\mathrm{K}$ 扩到 $100\mathrm{K}$；训练语料用长文档 PG-19 (books) 与 Proof-Pile (math)。易错点在于别把「窗口能到 $1.5\mathrm{M}$」理解成「预训练语料都是长文档」：长上下文是后加的能力，靠的正是这几类专门长文档，而不是把网页硬拼成超长序列；否则就会误判 pre-training 与 mid-training 的分工。

更靠后是 Tasks 与 Instruction_chat 两节，给出后训练数据的谱系。任务模板化一侧有 Super-Natural Instructions (Wang+ 2022) 的 1.6K+ tasks，把既有 NLP 数据集转成 templatized prompts（模板化提示）、微调 T5 做 k-shot，以更小规模胜过 InstructGPT；以及 Flan 2022 (Longpre+ 2023) 的 1.8K+ tasks，微调 T5 的 zero-shot、few-shot 与 chain-of-thought 三个版本。这一侧的要点是「把已有数据集改写成提示」，与正本里 OLMo 2 配方表直接列 Tülu 3 条目的写法不同，属 2025 版的讲法，读时须记得它讲的是 post-training 而非预训练语料。

合成指令一侧则从 Alpaca (Taori+ 2023) 的 52K 条、由 text-davinci-003 经 Self-Instruct (Wang+ 2022) 生成并微调 LLaMA 7B 起步，到 Vicuna 的 70K 条 ShareGPT 对话、Baize (Xu+ 2023) 的 111.5K 条 self-chat、WizardLM 的 Evol-Instruct，再到 MAMmoTH2 (Yue+ 2024) 从 Common Crawl 用 fastText 分类器筛 quiz sites、再用 GPT-4 与 Mistral 抽 QA 得到 10M 条 WebInstruct，以及 OpenHermes 2.5 汇聚出的约 1M 条 GPT-4 样本。两个反直觉的数字值得单独记：Llama 2 chat 只用 27,540 条厂商标注的高质量 SFT 数据，却认为胜过百万级开放数据、并建议把标注精力留给 RLHF 数据；Llama-Nemotron (NVIDIA, 2024) 刻意改用商业可用的 LLaMA、Mistral、DeepSeek r1、Qwen 生成合成回复并附 reasoning traces，而不是 GPT-4。易错点是别把「样本数越多越好」当成结论，也别以为开源模型的输出天然洗清了上游语料的许可问题——后者与正本 Common Pile 对合成数据的顾虑是同一条逻辑。

### 更正

原笔记 `note.md:186` 引用版权法定义处的「no known or later developed」应改为「now known or later developed」（依据：参考 Copyright law 段的法条原文，作 "now known or later developed"）；原笔记 `note.md:235`「来自 MusicBrainz 的 39 million 张图像」应删去量词「张图像」，改为「39 million 条记录/条目」或按参考保留为「39 million … from MusicBrainz」（依据：参考 Licenses 段写作 "39 million models from MusicBrainz"，同段 307 million images 才属于 Flickr；且 `note.md:219` 自身未加该量词）。两处均为笔误级文字错误，不影响主体结论与任何数字。
