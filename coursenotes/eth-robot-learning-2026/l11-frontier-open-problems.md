---
title: "第 11 讲 · 前沿与开放问题"
description: "这是ETH Zürich《Robot Learning: From Fundamentals to Foundation Models》由Oier Mees主讲的收官之讲（幻灯片标题印为“Lecture 12”，课程目录…"
---

# 第 11 讲 · 前沿与开放问题

> **ETH Zürich 机器人学习 · L11** ｜ 讲者 Oier Mees ｜ 62 分钟 ｜ [原视频](https://www.youtube.com/watch?v=eL4lcy1KNzE)

![课程标题页：机器人学习：从基础到基础模型，第12讲前沿与开放问题](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0001_00-00-00.jpg)

## 核心结论

这是ETH Zürich《Robot Learning: From Fundamentals to Foundation Models》由Oier Mees主讲的收官之讲（幻灯片标题印为“Lecture 12”，课程目录与YouTube标题标为“Lecture 11”，讲师开场自称“我来讲的第12讲、也是最后一讲”）。本讲先从“每周都在推特上宣布机器人已被解决”的浮夸演示切入，指出把机器人搬出实验室、换到新房间就会迅速失效，真正的泛化远不止“堆更多数据把外插变成内插”。随后把具身智能的开放问题拆解为若干支柱——最佳骨干网络、最佳数据配方、机器人感知的长尾、内省（模型如何知道自己不知道）、把强化学习规模化、数据飞轮与协同训练、终身学习、快速适应、移动操作的统一模型、跨时空的多模态记忆——并用讲师自己的VLMaps、LeLaN、自主改进、PALO等“初步工作”作示范。最后半程（约42:25之后无关键帧，仅凭转写）转向“如何做研究”：几个令人不适的事实、如何选题、保持兴奋、做自己的二号审稿人、方法驱动与问题驱动之别、实操建议，以及从ROS泥潭到CALVIN基准的个人故事，并以对助教与同学的致谢结课。

## 结课公告：真机部署用的PC与Demo Day [00:00 - 01:20]

讲师在正式进入主题前，先插了一条与课程项目相关的公告：当晚会给每位同学发一封邮件，说明如何访问一批用于测试策略的PC。这批机器是讲师在Microsoft工作时凑出来的，被搬到了ETH的一间房间里，也正是**Demo Day（结课演示日）上大家要把策略部署上去的真机**。访问方式“今晚晚些时候/明天起”通过邮件下发。这里没有出现比“当晚邮件、Demo Day”更硬的截止日期，因此本讲记录到此处为止。随后讲师说明今天的安排：先谈机器人学习的前沿与开放问题，结尾再分享一些他自己在“如何做研究”上踩坑换来的心得。

> 注：本讲幻灯片标题为“Lecture 12: Frontier & Open Problems”，日期“11.05.2026”，署名“Oier Mees”，机构logo为“ETH zürich”与“Microsoft”；而课程视频目录/YouTube标题把它记作“Lecture 11”。编号差异以画面为准，二者指同一讲。

## “机器人已被解决”：社媒演示与真实泛化的落差 [01:20 - 03:52]

![推特上关于机器人已解决的言论截图合集，含 Genesis AI 机械臂做饭视频与多条推文](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0002_00-02-04.jpg)

讲师用一张标题为“Every Week on Twitter: Robotics is Solved!”（每周推特：机器人已被解决！）的截图合集开场，讽刺地罗列了业界近乎每周一次的“机器人已解决/操作已解决”宣言：

- **Lex Sokolin**（Generative Ventures，@LexSokolkin，May 8）：“We are approaching the endgame for robotics”（我们正在逼近机器人的终局）。
- **Genesis AI**（@gs_ai，May 6）：“We are back. After one year of quiet building. Introducing GENE-26.5, our first robotic brain that takes a major step toward human-level capability.”（我们回来了。经过一年的安静研发，介绍GENE-26.5——我们第一个迈向人类级能力的重要一步的机器人大脑。）配图中一只双臂机械臂正在灶台前做饭。
- **Brett Adcock**（@adcock_brett）：“we just had an AI breakthrough in our lab / robotics is about to have its ChatGPT moment / and that moment is happening tomorrow”，标注“1:17 AM · Jan 7, 2024 · 2.4M Views”。讲师调侃：这条两年前的推文，我们“还在等那个ChatGPT时刻”。
- **Jim Fan**（@DrJimFan，May 8）：“I promise this will be the best 20 min you spend today! Robotics: Endgame, the sequel to my last year's Sequoia AI Ascent talk, ‘Physical Turing Test’. I laid out the roadmap for solving **Physical AGI** as a simple parallel to the LLM success story.”

讲师并不否认这些进展，但提醒要“take them with a grain of salt”（半信半疑地看待）：当有人宣布解决时，往往上一周已经有别人宣布过同样的“解决”。

![自主机器人泛化能力示意图：波士顿动力人形机器人与斯坦福双臂操作平台对比](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0003_00-02-31.jpg)

接着用一张“How Autonomous Robots (Usually) Generalize”（自主机器人通常如何泛化）的对比图说明问题：左边是**Boston Dynamics**的人形机器人（Atlas）在室内做动态动作，右边是**Fu et. al., Stanford 2024**的移动操作平台在厨房台面上做与烹饪相关的任务。讲师点出：这两段视频其实都不是最新结果，已有几年，但很能代表现象——一旦把机器人从实验室搬到新环境，遇到新光照、新物体、新相机视角，性能就“break very quickly”（迅速崩坏）。同一台机器人、同一个演示，把它挪到隔壁房间、放到训练分布之外，很可能“fail quite badly”（失败得很惨）。

## 明星机器人演示的“配方” [03:52 - 05:02]

![标题为“打造明星机器人演示的秘诀”的幻灯片，列出三条数据收集与训练策略，并配特斯拉和Figure机器人的演示图](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0005_00-03-52.jpg)

那么那些惊艳的推特演示究竟是怎么做出来的？讲师给出一张“Recipe for a Rockstar Robot Demo”（打造明星机器人演示的秘诀）的幻灯片，列了三条“并不那么secret”的配方：

1. **Collect “expert teleop” robot data**——自己在目标平台上收集专家遥操作数据，不要指望别人。
2. **It's not working yet? Keep collecting data until “extrapolation” becomes “interpolation”**——还没跑通？那就继续堆数据，直到把“外插”变成“内插”。
3. **Train/test the same day – avoid changes in setup**——尽量当天训练当天测试，避免环境发生任何分布漂移：光照不变、相机视角不变、桌面高度不变。

配图分别标注“Source: Tesla”（Optimus人形机器人在产线旁，画面里有“Optimus has been busy learning useful tasks lately”“AUTONOMOUS 1X”字样）与“Source: Figure”（Figure人形机器人整理桌面）。讲师的潜台词是：这套“简单配方”足以从VC那里拿到很多钱，但它并没有触及真正要解决的泛化问题。

## 泛化的尺度层级：从一个物体到整个世界 [05:02 - 06:42]

![用嵌套椭圆展示从单个物体、房间、建筑到整个世界的层级包含关系图](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0006_00-05-04.jpg)

讲师用一张嵌套椭圆的示意图（页面4）来刻画“泛化”的层级：最内层是“1 object”（单个物体），往外是“1 room”（单个房间），再往外是“1 building”（单栋建筑），最外层是“the world”（整个世界）。他评价：目前最惊艳的演示大多落在“一个物体到一间房间”这个很窄的圈里；更有野心的工作试图推向“一栋建筑”；但**真正的目标是“the world”**。理由是人就是这样泛化的——你不需要见过每一间厨房也能在新厨房里做饭，不需要摸过世界上每一把马克杯也知道该怎么去抓一把没见过的杯子。人类是把已有知识“stitch together”（拼接）、对新颖情境推理、并以新颖的方式组合已知，从而完成这种泛化。这**不是内插**，而是一种仍然 elusive（难以捉摸）的“智能推理”，我们的机器人策略离它还差得远。因此，这种能力“goes a little bit beyond just collecting more data”——不是单纯堆数据就能买到的。

## 两全其美的追求：非具身基础模型 vs 专用具身模型 [06:42 - 09:15]

![对比非具身基础模型与专用具身模型在数据、泛化与物理感知上的优劣](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0007_00-07-15.jpg)

幻灯片标题是“The Quest for the Best of Both Worlds”（追求两全其美），把当下两类模型摆在一起对照：

- **Non-Embodied Foundation Models（非具身基础模型）**：一个Transformer同时吃Images与Language。优点是✓ Internet-scale Data（互联网级数据）、✓ Broad Generalization（广泛泛化）；缺点是✗ Lacks Physical Grounding（缺乏物理落地）、✗ Non-Embodied Sensing（非具身感知）。
- **Specialist Embodied Models（专用具身模型）**：配图是机械臂在桌面操作（标注“4x”快放）。优点是✓ Physical Grounding（物理落地）、✓ Embodied Sensing（具身感知）；缺点是✗ Narrow, Task-specific Data（狭窄、任务特定的数据）、✗ Poor Generalization（泛化差）。

讲师举例：VLM能“描述”一张桌上有玻璃杯的图片，却并不真正理解“把它推到桌边会摔碎”这件事会发生什么。这条批评同样适用于生成式视频模型——像Sora这类模型能合成非常逼真的物理交互（倒液体、物体下落），但仔细看它们“do actually tend to violate basic physics”：它们很擅长“看起来懂物理”，但并不真正理解物理。反过来，专用具身策略能做灵巧任务、能消费更异构的传感模态，却难以泛化到in-the-wild任务。于是机器人学习的问题就是**取两者之长**。

讲师进一步把“被动数据”与“具身数据”对立起来：基础模型吃的是互联网上被动沉淀下来的图文/视频，而具身智能需要的是机器人在物理世界里**主动交互**产生的信号——它天然多模态（视觉、语言、触觉、力、本体感知交织）、天然空间化（有几何、有可供性affordance）、也天然时间化（动作的后果随时间展开）。他强调，把基础模型“接地”到物理世界，不是简单地在已有LLM上“加个视觉头”，而是要让模型在训练时就见过这些具身信号，从而不只是会“描述世界”，而是能**推理并以更智能的方式与世界互动**。这一判断贯穿本讲后半程所有开放问题：骨干怎么选、数据怎么配、感知长尾怎么补、何时该推理、如何知道自己不知道、RL能否规模化、飞轮如何转起来、记忆如何组织——本质上都是在追问“如何把被动先验变成可行动的具身知识”。

## 开放问题一：机器人最佳骨干网络 [09:15 - 11:16]

![提出机器人最佳骨干网络的三选一问题：视觉语言模型、生成式视频模型或从零训练](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0008_00-09-19.jpg)

幻灯片“What is the Best Backbone for Robotics?”（机器人最好的骨干是什么？）摆出三个互斥选项，对应社区里的三个阵营：

- **Vision-Language Model（视觉语言模型）**：VLM自带丰富的语义世界知识、能对场景推理，微调一下就能“免费”获得物体级的语义泛化。
- **Generative Video Model（生成式视频模型）**：更懂时间动力学，能建模环境/世界如何随时间演化；其预训练目标与机器人策略要做的事（对动作后果进行推理）更接近。
- **Doesn't matter if we have enough data to train from scratch?（只要有足够数据从零训练，骨干无所谓？）**：更挑衅的第三派——如果真实机器人数据足够多，就一切从零训练、忘掉互联网预训练。

讲师提到Scott Reed几周前正好就这个话题做过客座讲座，可在课程网站/YouTube回看。他的结论是：每个立场都有严肃的支持者，但“the jury is still out”（尚无定论）。

## 开放问题二：最佳数据配方 [11:16 - 13:43]

![对比两种数据配方金字塔：左侧为真实数据、仿真数据、网页数据三层，右侧为真实数据、网页数据两层](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0009_00-11-23.jpg)

骨干问题与数据问题紧密相关。幻灯片“What is the Best Data Recipe?”（最佳数据配方？）用金字塔来对比不同配方。第一组对比：

- **左（三层）**：Web Data（底层，配图Wikipedia与Common Crawl标志）→ Simulation Data（中层，仿真交互台）→ Real-World Data（顶层，真实机器人）。逻辑是先用网页数据获得世界知识与语言接地，再用可扩展、廉价的仿真交互放大经验，最后用少量真实机器人数据弥合sim-to-real gap。
- **右（两层）**：Web Data → Real-World Data，**去掉仿真层**。支持者的论点是：精确仿真整个世界极其困难、需要大量工程，不如把时间花在采集真实数据上。

![对比两种数据配方金字塔：左侧为真实数据、人类视频、互联网视频三层，右侧为三层真实数据](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0010_00-13-27.jpg)

第二组对比（同样标题）：

- **左**：Internet Videos（底层，YouTube标志）→ Human Videos（中层，第一视角/人类视频）→ Real-World Data（顶层）。这是“把视频当作一等公民”的配方：先在通用互联网视频上预训练，再用人类自我中心视频中训练，最后后训练去预测机器人动作。直觉是——对具身任务而言，视频是比“视觉问答”类任务丰富得多的先验。
- **右**：三层全是Real-World Data（“从零”配方，各层标注略有差异，取决于标注类型与粒度），即彻底押注真实世界数据。

## 数据采集接口的权衡 [13:43 - 16:22]

![五种数据采集接口示意图：UMI风格、灵巧手手套、静态双臂遥操作、VR、移动双臂遥操作](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0011_00-13-46.jpg)

选定配方后，下一个问题是**如何规模化地采集真实数据**。幻灯片“Data Collection Interfaces”（数据采集接口）展示了一个“接口动物园”，每种接口都在“可扩展性—与目标本体的对齐度—可演示任务的复杂度”之间做不同取舍：

- **UMI-Style**：手持夹爪式设备（在洗手池边采集），比遥操作机器人更可扩展，但需要跨本体学习。（讲师口述中把它念成“Yumi”，画面标签为“UMI-Style”，以画面为准。）
- **Gloves for Dexterous Hands**：灵巧手外骨骼/数据手套。
- **Static Bimanual Puppeteering**：固定式双臂主从遥操作，数据质量高于VR遥操作，但最贵、需要leader-follower机械臂，也就“有更多硬件会坏”。
- **VR**：头显加手柄控制。
- **Mobile Bimanual Puppeteering**：移动式双臂遥操作（画面里有“Clean Restroom”“10x speed”字样）。

![数据规模与硬件对齐程度的权衡关系散点图，标注了自我中心视频、数据可穿戴设备和遥操作三个数据源](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0012_00-15-31.jpg)

幻灯片“Data Scalability vs Hardware Alignment”（数据可扩展性 vs 硬件对齐度）把取舍画成一条负相关的散点趋势线。纵轴Data Scalability从$10^3$ hr、$10^5$ hr到$10^7$ hr；横轴Hardware Alignment从左“Sensorized Human Data”（带传感的人类数据）到右“Robot in the loop”（机器人在环）。三个数据源沿对角线分布：

- **Egocentric Videos**（自我中心视频）：左上，可达上千万小时，但它是人类数据而非机器人数据，本体差距（embodiment gap）很大。
- **Data Wearables**（数据可穿戴，如手套）：居中。
- **Teleop**（遥操作）：右下，与机器人完美对齐、机器人完全在环，但因为慢且贵，数据量受限。

讲师说，如何绕开这些权衡——例如用人形机器人去更好地利用自我中心视频——正是社区正在盯的开放问题之一。

他把这张散点图的读法讲得很直白：纵轴是“你能采到多少小时数据”（从$10^3$小时一直到$10^7$小时量级），横轴是“这些数据和你最终要部署的那台机器人对得多齐”。越往左下，数据越“像人”（sensorized human data），越往右上，数据越“像机器人”（robot in the loop）。自我中心视频、数据可穿戴、遥操作三个点恰好沿一条负相关对角线排开，说明“可扩展性”和“硬件对齐”是一对跷跷板：想要上千万小时，就得接受它是人类数据、存在本体差距；想要完美对齐，就得忍受遥操作又慢又贵。讲师的言下之意是——没有免费的午餐，选接口本质上是在为后续“跨本体迁移”还是“数据量不足”预先埋下要还的债；而人形机器人之所以被看好，是因为它的运动学更接近人，也许能把自我中心视频这条“海量但不对齐”的通道利用起来。

## 迈向灵巧通用策略 [16:22 - 19:02]

![实现灵巧通用策略的维恩图，展示灵巧性与泛化能力所需的要素及共同目标](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0013_00-16-33.jpg)

幻灯片“Towards Dexterous Generalist Policies”（迈向灵巧的通用策略），副标题“Dexterity and generalization require different ingredients”（灵巧与泛化需要不同的原料）。一个维恩图把两者分列：

- **Dexterity（灵巧）**一侧：High-Frequency Control（高频控制）、Heterogeneous Multimodal Sensing（异构多模态感知）。
- **Generalization（泛化）**一侧：Foundation Models（基础模型）、Mobility（移动性）。
- 交集处标注**Goal（目标）**。

讲师解释张力所在：灵巧通常需要高频控制，这与运行很大的基础模型相矛盾（后者慢、算力贵）；灵巧还需要触觉这类异构感知，而多数基础模型今天根本不消费这些模态。反过来，要推进泛化就得借力基础模型；而机器人若能移动、在不同房间与建筑里积累经验，会比“被拧在桌上、只看同一场景”的机器人泛化得更好。目标就是**站到维恩图交集上**：既能应对真实操作的复杂性、又足够通用到能在非结构化环境里处理未见物体。多数系统目前只落在某一边，这正是开放问题。

## 灵巧性不止于视觉：触觉/力反馈的缺失 [19:02 - 20:38]

![展示TidyUpRobot机器人硬件平台及失败模式示例的幻灯片](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0014_00-19-02.jpg)

讲师放了一段他博士时期的“个人视频”来说明“Dexterity Requires More Than Vision”（灵巧不止于视觉）。画面是一台PR2双臂机器人，背景横幅写着“Albert-Ludwigs-Universität Freiburg / TidyUpRobot”，桌面标着“Failure modes”。底部引用：

> Self-supervised 3D Shape and Viewpoint Estimation from Single Images for Robotics
> Mees, Tatarchenko et al., IROS, 2019.

他当年花了很多时间训练一个模型：输入RGB图像，预测物体3D形状与位姿——这样即使深度相机被部分遮挡，也能准确预测例如马克杯把手的位置以便抓取。但视频里（现场音频没放出来，讲师口头描述）机器人**没抓把手、直接去抓杯身，把杯子彻底捏碎了**，因为它没有触觉/力反馈，做了一个“过猛的盲抓”。讲师由此判断：我们当前的VLA很可能也在“annihilating objects”（捏爆物体），因为这条战线进展不大；机器人在接触物体时“仍然基本是盲的”。

## 原生多模态与“机器人感知的长尾” [20:38 - 22:46]

![原生多模态模型架构图：图像/音频/视频/文本分别编码后交织成统一token序列，输入统一多模态Transformer](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0016_00-20-38.jpg)

第一个直觉是：既然课程前面讲过“原生多模态模型”（Recap: Native Multimodal Models），那与其把新模态“嫁接”到视觉上，不如**从零联合训练一个统一Transformer**，把触觉、力、本体感知等全都丢进去，问题就解决了？幻灯片展示了这一思路：image→vision enc.、audio→audio enc.、video→video enc.、text→tokenizer，各模态可选，交织成任意顺序/任意混合的token序列，喂给“unified multimodal transformer — trained from scratch, every weight has always seen every modality”，输出text（+ image）。

![机器人感知的长尾问题：模态稀缺、跨模态配对数据缺失、不同机器人本体间传感器异构](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0017_00-21-44.jpg)

但讲师指出，一旦你尝试这么做，就会撞上“The Long Tail of Robot Sensing”（机器人感知的长尾），有三个拦路虎：

- **Scarcity（稀缺）**：许多对灵巧至关重要的模态（如触觉）根本没有规模化数据——硬件贵、脆弱、不标准，仍很“researchy”。
- **Pairing（配对）**：即便单个模态存在，把RGB + depth + tactile + force同时配对的跨模态数据“nearly nonexistent”（几乎不存在）。
- **Heterogeneity（异构）**：传感器在不同机器人本体间差异巨大，即使有数据，迁移也很难。

## 以语言为桥跨模态推理 [22:46 - 24:14]

![以Language为中心、连接Vision、Audio、Touch的多模态关系示意图](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0018_00-22-46.jpg)

给定配对多模态数据的稀缺，一个有意思的问题是：**能否用语言作为“桥”，去推理那些从未一起见过的模态？** 幻灯片“Reason Across New Modalities?”把Language放在中心，连接Vision、Audio、Touch——因为我们拥有的是“不相交”的数据量：大量视觉-语言数据、一些音频数据、少量触觉数据，却没有把它们同时训练的数据。

讲师介绍他们在Berkeley做的初步工作：用语言作为“语义胶水（semantic glue）”去**绑定**这些异构模态，从而给VLA赋予它从未被训练过的感官（触觉、音频）。

![触觉提示、多模态提示、视听跨模态提示与零样本触觉描述四种机器人操作任务示例](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0019_00-23-48.jpg)

这解锁了若干新的跨模态推理能力，幻灯片给出四格示例：

- **Tactile Prompting**：“Grab the object that feels slick”（抓那个摸起来滑的物体）。
- **Multimodal Prompting**：“Grab the object that looks yellow and feels squishy”（抓那个看起来黄色、摸起来软糯的物体）。
- **Audio-Visual Cross-Modal Prompting**：“Grab the object that is the same color as the button that plays metal”（抓那个与“播放金属声的按钮”同色的物体）。
- **Zero-Shot Touch Description**：“The grasped object feels corded”（抓到的物体摸起来像绳纹）——场景是机器人把手伸进购物袋里、相机什么都看不见，凭触觉传感器描述它摸到的东西，就像人能做的那样。

## 何时以及如何智能推理：自适应测试时计算 [24:14 - 26:54]

![按策略置信度从确定到未知划分的智能推理决策流程图，包含分布内、近分布与分布外任务的处理路径](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0020_00-24-14.jpg)

这张幻灯片（“When & How to Reason Intelligently?”）与上一周“用LLM/VLA做推理”的内容直接衔接。讲师把“未来家用机器人管家”的推理需求画成一条按**policy confidence（策略置信度）**从certain→uncertain→unknown的谱系，分三列处理：

- **In-distribution task**（已知物体/场景/目标，certain）→ **Reactive policy**（No reasoning tokens，无推理token）→ 约$50$ Hz control loop。顶部标注“✓ no reasoning needed”：任务在分布内就别浪费算力，以尽可能高的频率跑策略。
- **Near-distribution task**（新颖或低置信，uncertain）→ **Uncertainty detection** → **Scale test-time compute**（↓ frequency, ↑ reasoning，降频升推理）。顶部标注“adaptive TTC”（自适应测试时计算）：用上周学过的方法在测试时多花token把任务想清楚。
- **Out-of-distribution task**（新颖目标或失败模式，unknown）→ **Escalate / ask for help**（Pause and defer）。顶部标注“escalate / stop”：当测试时计算/更多推理也救不了时，最安全的做法是停下来求助人类。

底部还说明“test-time compute can take many forms”：More thinking tokens（CoT, GRPO, RL reasoning）、Larger / specialist model（Foundation model fallback）、Human in the loop（Teleop, correction, label）。

## 开放问题：内省——模型如何知道自己不知道 [26:54 - 28:26]

![提出模型自我认知（Introspection）开放问题的幻灯片](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0021_00-26-54.jpg)

讲师立刻追问：上面这套自适应推理框架，**前提是能可靠地检测到模型何时不确定**。幻灯片“Open Problem: Introspection”（开放问题：内省）写道：“How does a model know what it doesn't know?”（模型如何知道自己不知道什么？）副标题“Generalization across many axes: objects, environments, embodiments, instructions, tasks...”。

他强调，这对当前机器学习模型是一个**很大的假设**。这本质上就是“知道自己不知道”的内省问题，是机器学习的核心难题之一，而且**随着模型变大反而更难**：大模型很擅长“pattern completion”（模式补全），却对自身的胜任力“poorly calibrated”（校准很差）——因为见过太多数据，新颖情境表面上仍显得熟悉，使得分布外检测极其困难。在机器人上更是雪上加霜，因为失败可以沿物体、环境、本体、任务、传感器噪声等**许多不同维度**发生，策略可能因为完全不同的原因而“看起来自信却错了”。

讲师把内省与前面那张“自适应推理”流程图扣在一起：整条“置信度谱系”之所以能用，全押在“能可靠地判断当前处于certain/uncertain/unknown哪一档”上；一旦这个判断不可靠，要么在不该推理时浪费算力、要么在该停下来求助时自信地闯祸。他特别指出，泛化本身是多轴的（objects、environments、embodiments、instructions、tasks……），失败维度是**乘积式**地爆炸的，这也是为什么机器人里的不确定性检测比纯语言/视觉模型更难：一个策略可能对物体形状过度自信、却对相机视角的偏移毫无察觉，两种错误需要不同的检测器。他把这列为“central problems in machine learning”之一，暗示这本身就是值得博士生攻的硬骨头。

## 模仿学习的天花板与强化学习的潜力 [28:26 - 30:58]

![模仿学习受限于数据集演示轨迹的示意图](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0022_00-28-58.jpg)

幻灯片“Current Robot Models Rely on Imitation Learning”（当前机器人模型依赖模仿学习）指出：策略“bounded by the demonstrations in the dataset”（被数据集里的演示所限制）。右侧“Dataset”框里画了状态节点1、2、3与演示轨迹的箭头。讲师给了一个玩具例子：如果训练数据只有“从状态1到2、从2到3”的演示，那么无论模型多大，它都**没有途径学到“从1直达3”这条更好的路径**，也学不到人类操作员没演示过的另一种解法。

![模仿学习与离线强化学习策略对比示意图](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0023_00-30-00.jpg)

幻灯片“Can we Scale RL for Robotics?”（我们能把RL规模化用于机器人吗？）给出对照：

- **Imitation learning**：Imitates seen behaviors in the data（模仿数据里见过的行为）。
- **Reinforcement learning (RL)**：Learns near optimal policies from suboptimal data（从次优数据学到近最优策略）、Stitches suboptimal trajectories（拼接次优轨迹）。右侧“Offline-RL policy”框里，一条红色直箭头从1直达3，绕过了2。

讲师说，RL已经在推理、智能体、知识发现等领域催生了很多新能力，但**把RL规模化到真实机器人上极其困难，仍需要大量算法创新，而不只是更多算力**。他判断：让RL在机器人上规模化工作，可能是当下最高影响力的研究方向之一。

## 数据飞轮与专家/自主数据协同训练 [30:58 - 34:04]

![机器人学习数据飞轮四阶段循环图：增加部署、更多训练数据、更好学习、更强机器人](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0024_00-31-02.jpg)

幻灯片“Robot Learning Data Flywheel”（机器人学习数据飞轮）把“圣杯”画成一个自增强的循环：**Increased Deployments**（improved reliability）→ **More Training Data**（deployment data / suboptimal demos / human corrections）→ **Better Learning**（BC, RL, fine-tuning）→ **More Capable Robots**（multiple embodiments）→ 再回到更多部署。背后是“Bitter Lesson”的信念：更多数据通常带来更好模型，更好模型带来更强机器人、更广部署、进而更多数据。

![专家轨迹与自主数据协同训练对比图，指出分布不匹配与时间不匹配两大问题](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0025_00-32-05.jpg)

要让飞轮转起来，第一个想法是把**专家数据（遥操作演示）**与**自主部署数据**协同训练。幻灯片“Co-Training Expert & Autonomous Data”把两类分布摆在一起对照：

- **Expert trajectories**（Human demos, teleoperation）：✓ High quality、✓ On-manifold states、✓ Short, smooth, dense；✗ Expensive, limited scale、✗ Bounded by human skill。
- **Autonomous rollouts**（Self-generated experience）：✓ Cheap and scalable、✓ Covers failure modes；✗ Noisy, suboptimal、✗ Off-manifold states、✗ Longer, jerky, variable。

两者之间存在**Distribution mismatch**（state spaces don't overlap，状态空间不重叠）与**Temporal mismatch**（frequency, length, smoothness，频率/长度/平滑度不匹配）。若用模仿学习天真地把两者混在一起，模型会被质量、频率与分布的错配“搞糊涂”。于是幻灯片底部写：**New algorithms required**（weighting, alignment, joint learning）。讲师强调，RL也许能帮上忙，但核心目标是“never throw away data”——真实世界数据太宝贵、太贵。

他解释“为什么这比看上去难”：专家遥操作数据是“on-manifold”（贴着流形）的高质量短轨迹，而机器人自主跑出来的rollout往往更长、更抖、更次优，还会走到人类演示里从没出现过的“off-manifold”状态——两类数据的“状态空间几乎不重叠”，时间上的频率、长度、平滑度也对不齐。天真地把它们丢进同一个行为克隆目标，等于让模型同时拟合两种互相矛盾的监督信号，结果通常是被“平均”成一个谁都不像的策略。因此他呼吁的“新算法”至少要解决三件事：给不同质量的样本**加权（weighting）**、把两类轨迹在时间/状态上**对齐（alignment）**、以及在一个目标里**联合学习（joint learning）**而不是一锅烩——这也正是他前面说“让RL规模化是最高影响力方向之一”的落点：飞轮转不转，取决于你能否把部署期产生的“脏数据”也变成有效信号。

## 终身学习 [34:04 - 35:19]

![终身学习性能曲线图：部署后持续学习性能上升，与当前现实形成差距](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0026_00-34-09.jpg)

幻灯片“Lifelong Learning”（终身学习）用一张性能-时间曲线图表达理想与现实的差距：横轴分pre-training与deployment两段，在“deploy”处画一条竖直虚线。绿色实线“lifelong learning”在部署后持续上升，棕色虚线“reality today”在部署后反而下滑，二者之间的落差标注为“gap”，底部写“open problem: how do we get there?”。

讲师的定义是：机器人应像智能生物那样，通过**积累并从与物理世界的所有经验与交互中学习**而随时间变好。但当今主导范式是：训练一个模型、冻结权重、部署出去，然后要求它泛化到预训练数据未必准备过的新情境。除了混合与学习“质量参差的部署数据”，你还得保证在部署数据上微调时**不会灾难性遗忘（catastrophic forgetting）**原始能力。这是一个非常令人兴奋、也非常有影响力的问题。

## 快速适应：上下文学习 [35:19 - 36:56]

![上下文学习实现快速适应的三分支结构图：纠正指令、同本体机器人演示、跨本体演示汇聚为无需重训的适应策略](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0027_00-36-13.jpg)

幻灯片“Rapid Adaptation: In-Context Learning”（快速适应：上下文学习）处理一个更即时的版本：**策略正在部署中失败，怎么办？** 与其重训，不如让它“on the fly”快速恢复。讲师举例：你在家第一次拆封机器人，让它从冰箱拿可乐，它却抓不住冰箱把手——这时给一句语言反馈（“往左一点再抓”）也许就够了；把它放进模型上下文，机器人就能适应并完成任务。若问题更大（它根本不知道怎么开冰箱），最简单的办法是你亲自演示一遍开门，再把演示放进上下文。

流程图顶部是“Policy failure（novel task, environment...）”，向下分三条路径：**Corrective instructions**（language feedback）、**Robot demo in context**（same embodiment）、**Cross-embodiment demo in context**（human or novel robot），三条都汇入“**Adapted policy**（no retraining required）”。底部写“open problem: few-shot, real-time adaptation”。愿景是：不必每次遇到新东西就重训，而是用**任何可用信号做少样本快速适应**。

## 移动操作的统一模型 [36:56 - 39:32]

![移动操作统一模型路线图：对比今日SLAM+导航+操作模块化流水线与未来无地图统一模型愿景](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0028_00-37-15.jpg)

幻灯片“Towards a Unified Model for Mobile Manipulation”（迈向移动操作的统一模型）对比“today”与“vision”两栏。

- **today**：SLAM + metric map → Localization stack → Nav planner → **hard handoff**（硬切换）→ VLA (manipulation only)。旁注“Maps are static: world changes, can't precompute all”（地图是静态的，而世界在变，无法预先算好一切）。
- **vision**：先问“which env. representation?”，候选包括Metric map、Gaussian splat、Topological map、Scene graph、BEV projection、Walkthrough video；再往下是Mapless adaptation（reason from current obs + memory），但受“context length bottleneck — can't fit a city map”（上下文长度瓶颈，装不下一张城市地图）限制；最终愿景是Unified model（implicit state estimation、loop closures、whole body control），下分Navigation（mobile base）与Manipulation（bimanual, dexterous）。

底部三个问题框：Replace SLAM（implicit localization）、Env. representation（map, video, implicit?）、Whole body control（nav + manipulation jointly）。讲师串起的问题链是：能否用**一个能同时做导航与操作的全身控制统一模型**取代“SLAM栈+硬切换”；环境究竟该用显式3D地图、BEV投影、拓扑图、场景图，还是干脆把一段“用iPhone走过办公室”的walkthrough视频放进上下文就够了；这样的统一模型如何做（隐式还是显式）状态估计；把环境表示放进上下文后，如何应对环境持续变化、如何做无地图的局部导航。

他特别强调“hard handoff”（硬切换）这个结构性缺陷：今天的系统是先用传统SLAM栈建图、定位、规划导航，等机器人真的需要与物体互动时，才“啪”地切到一个只负责操作的学习策略（VLA）——两套系统各说各话，导航阶段积累的几何/语义信息很难无缝喂给操作阶段，反之亦然。愿景里的统一模型要把“隐式状态估计、回环（loop closures）、全身控制”合进一个网络，让移动基座（navigation）与双臂灵巧操作（manipulation）联合优化。至于环境表示，他给了六个候选（Metric map、Gaussian splat、Topological map、Scene graph、BEV projection、Walkthrough video）并明确说“没有定论”，还点出一个硬约束：上下文长度瓶颈——你不可能把整栋城市级的地图塞进模型上下文，所以“无地图、从当前观测+记忆推理”的mapless adaptation可能是更现实的路子。

## 跨时空的多模态记忆 [39:32 - 41:06]

![跨时空多模态记忆分类图：按秒/分-时/月-年划分短时、长时与终身记忆及其模态、瓶颈与挑战](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0029_00-40-21.jpg)

幻灯片“Multimodal Memories Across Time & Space”（跨时间与空间的多模态记忆）讨论“记忆规模化”。顶部“what gets stored?”列出候选模态：video、depth、language、touch、audio、proprio、bboxes、…横轴按时间尺度分三列：

- **seconds**：Short-horizon memory（dense video, recent frames），用于occlusions、grasp correction；代价是high token cost、doesn't scale to minutes。
- **minutes–hours**：Long-horizon memory（language, semantic events），用于task progress、steps done；代价是loses spatial detail，问题在于when to write, what to forget?。
- **months–years**：Lifelong memory（episodic + semantic），让机器人smarter over deployment；挑战是storage + retrieval at scale、staleness、consolidation。

三条共同支撑cross-modal alignment（joint reasoning + retrieval across all modalities and timescales），最终指向Unified multi-scale memory（video + language + touch + ...）。底部写“open problem: store, align, retrieve, and forget across modalities, tasks, and a lifetime of deployment”。讲师强调：机器人产生的是连续的多模态经验流，什么都可能有用，但**存什么、在什么时间尺度存、何时写、忘什么、如何检索、如何做跨模态对齐**，都是开放问题。

他把记忆问题与前面的“终身学习”和“移动操作”两条线扣在一起：一个长期部署、四处移动的机器人，会源源不断地产生视频、深度、语言、触觉、音频、本体感知、检测框等流式数据，全存下来既存不下也用不动。短时记忆（秒级）适合遮挡消解、抓取纠偏这类需要密集几何的场景，但token成本极高、撑不到分钟；长时记忆（分钟到小时）得把经验压成语义事件、记住“任务做到哪一步了”，代价是丢掉空间细节，于是“何时写入、该忘什么”本身成了难题；终身记忆（月到年）要同时维护情景记忆与语义记忆、越部署越聪明，挑战在于规模化存储与检索、以及信息陈旧（staleness）与固化（consolidation）。他给出的终极形态是**统一的多尺度、跨模态记忆**：能在所有模态、所有时间尺度上联合推理与检索——而这恰恰是“存储、对齐、检索、遗忘”四个动作在“整个部署生命周期”维度上尚未被解决的问题。

## 初步工作：移动性与多模态地图、自主改进、人类反馈适应 [41:06 - 42:36]

讲师给出三条“已经迈出第一步”的工作，作为对上述开放问题的示范。

![关于机器人移动性与多模态地图的幻灯片，包含VLMaps和LeLaN两个项目的介绍及演示画面](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0030_00-41-03.jpg)

**First Steps: Mobility & Multimodal Maps**（移动性与多模态地图）：

- **VLMaps**：把VLM嵌入融合进3D地图，实现zero-shot spatial navigation（零样本空间导航），无需任何训练、开箱即用，可导航到开放词汇目标。引用“Multimodal Spatial Language Maps for Robot Navigation and Manipulation”，作者**Chenguang Huang¹, Oier Mees², Andy Zeng³, Wolfram Burgard¹**，机构为¹University of Technology Nuremberg、²UC Berkeley、³Google Research（logo为UTN、Berkeley、Google Research）。幻灯片底部另给一条期刊版引用（Huang, Mees, Zeng, Burgard，2025）。
- **LeLaN**：Learning A Language-conditioned Navigation Policy from In-the-Wild Video——完全从YouTube这类野外视频学习**无地图的语言导航**；他们发现训练的YouTube数据越多，开放词汇无地图导航性能越好。作者Hirose, Glossop, Sridhar, Shah, Mees, Levine，CoRL 2024。（讲师口述念成“Leela”，画面为“LeLaN”，以画面为准。）

![基于基础模型的机器人自主改进闭环架构图，包含任务提议、奖励检测与数据收集流程](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0031_00-41-51.jpg)

**First Steps: Autonomous Improvement**（自主改进）：用基础模型在**无人干预**下实现自主改进。流程图：Robot Policy由Offline Dataset与Online Dataset喂入；Continuous Improvement驱动Robot 1…Robot 5；Task Proposals与Reward Detector各由一个Foundation Model承担，形成Autonomous Data Collection闭环。示例思路是基于环境affordances来“提议并打分”任务。引用“Autonomous Improvement of Instruction Following Skills via Foundation Models”，Zhou\*, Atreya\*, Lee, Walke, Mees, Levine，CoRL 2024。

讲师把这页和前面的“数据飞轮/终身学习”直接挂钩：飞轮要转，卡点之一是“谁来在部署中不断产生有用的新数据、又谁来给这些数据打标签/打分”。这里的答案是用两个基础模型充当“任务提议器”和“奖励检测器”——让一个VLM看着环境提议“这台机器人现在可以试什么任务”，再让另一个VLM判断“它做成了没有”，于是无需人类介入就能自动收集在线数据、持续改进策略，并把经验喂回Robot Policy。多台机器人（Robot 1…Robot 5）并行，正对应飞轮里“multiple embodiments、更多部署→更多数据”的环节。

![通过人类反馈进行策略自适应的对比图，包含语言优化与参数微调的性能曲线及机器人抓取任务示例](/coursenotes/eth-robot-learning-2026/l11-frontier-open-problems/kf_0032_00-42-25.jpg)

**First Steps: Adaptation via Human Feedback**（通过人类反馈适应）：利用人类的verbal或visual cues来适应新任务。

- **PALO**（Policy Adaptation via Language Optimization）：一种**非参数**方法，让通用策略在语言空间里“Search in Language Space”，**只需5个演示**就能适应新的长时程任务，达到与全量微调（需要>100个演示）相当的准确率。右侧曲线以# Demonstrations为横轴、Success Rate为纵轴，PALO（橙色）迅速逼近上限，Finetuning（灰色）需要多得多的演示。引用“Policy Adaptation via Language Optimization: Decomposing Tasks for Few-Shot Imitation”，Myers\*, Zheng\*, Mees et al.，CoRL 2024。（讲师口述念成“PAL”，画面为“PALO”，以画面为准。）
- **Composing Pick-and-Place Tasks By Grounding Language**：右侧对话气泡演示语言分解——“Fetch the round yellow thing”“Do you mean the lemon in the middle?”“Yes”“Place it left of the object on the bottom”。引用Mees, Burgard，ISER 2021。

讲师把PALO定位为前面“快速适应/上下文学习”思路的一个已落地实例：与其在参数空间里对策略做全量微调（动辄上百个演示），不如在**语言空间**里搜索——让VLM把一个新长时程任务分解成若干子任务、用少量自然语言示例去“改写”策略的行为，从而**只喂5个演示**就逼近全量微调的效果。这正是“few-shot、免重训适应”的具体做法；右侧那段“拿那个圆圆的黄色的东西→你是说中间那个柠檬吗→对→放到底部那个物体的左边”的来回，则展示了他更早期（ISER 2021）用语言接地把pick-and-place任务组合起来的思想源头。

## 具身智能的四大支柱 [42:36 - 43:20]

讲师把前面所有问题“蒸馏”成四根支柱：**embodied reasoning（具身推理）、dexterous mobile manipulation（灵巧移动操作）、lifelong learning（终身学习）、rapid adaptation（快速适应）**。他说，要让机器人真正“work”（而不只是拍一段演示视频），需要被解决的就是这些；“maybe this is my end game for robotics”（这也许才是我心中的机器人终局）。他希望大家课后对“现代机器人学习研究者的工具箱”有更好的理解，若这些问题让你兴奋，就自己去推进前沿——但“知道开放问题是什么只是半场”，另外半场是“知道怎么做研究”。

这四根支柱其实是对本讲前半程所有幻灯片的收拢：具身推理对应“两全其美、骨干选择、感知长尾、以语言为桥、何时推理、内省”；灵巧移动操作对应“灵巧通用策略、不止于视觉、移动操作统一模型、多模态记忆”；终身学习对应“模仿天花板、RL规模化、数据飞轮、协同训练、终身学习曲线”；快速适应对应“上下文学习、自主改进、PALO”。他刻意把“我的终局”与开头那些“每周推特：机器人已被解决”对照——真正的endgame不是又一段惊艳demo，而是把这四块硬骨头啃下来。

## 如何做研究：几个令人不适的事实 [43:20 - 46:21]

从约43:20起，本讲不再有幻灯片关键帧，以下依据转写记录。讲师分享四条“uncomfortable facts”：

1. **Most research is incremental（多数研究是渐进的）**，这不是批评，科学通常就是这样。类比：今天的自动驾驶建立在约40年的积累之上——从1986年的早期自动驾驶实验车，到DARPA挑战赛，再到端到端现代学习，没有人用一篇论文就解决它。
2. **Most research ideas never become papers（多数想法成不了论文）**。典型漏斗：10个想法、跑5个实验、拿到2个有趣结果、也许发1篇论文——这已经是好比例。多数想法不work不是失败，而是过程本身。
3. **Most papers don't stand the test of time（多数论文经不起时间考验）**。领域走得很快，今天的“突破”两年后可能只是个脚注。
4. **在一个规模至上的世界里，简单的想法往往影响更大**，因为它们更容易被规模化。若你的想法需要“17个活动部件”才能work，不妨问问：有没有一个更简单的版本，能带你走完80%的路。

讲师把这四条放在一起，是为了给刚入门的同学“打预防针”：研究里“不work”是常态而非例外，把预期校准到“十个想法里能成一篇”反而更不容易在挫折面前放弃。他尤其看重第四条——在机器人这种要同时应付硬件、数据、算力、真实世界噪声的领域，一个组件多、耦合紧的方案往往在放大规模时处处漏水；而一个朴素但能顺着数据/算力曲线一路scale上去的想法，长期影响通常更大。这也呼应了他前面讲的“Bitter Lesson”式数据飞轮：能自我增强的简单循环，胜过需要精心维护的复杂流水线。

## 如何找到好的研究问题 [46:21 - 48:53]

讲师给出选题“配方”：你需要**一个重要的问题 + 一个解决它的计划**，两者缺一不可。反例有二：光说“让我们攻克癌症”——问题很重要，但如果你不知道怎么下手，也许不该从它开始；反过来，一个能在某个狭窄且已饱和的仿真机器人基准上刷高1%的新算法，虽有“计划”，却不推动任何真正重要的东西。**甜区**是：既重要、又有你自己的“切入角度/计划”。他还建议再想一步：如果你的计划真的成功了，它会**长什么样**？会解锁什么？如何帮助社区？有什么“以前不可能、现在可能”了？

紧接着的第四个（也是他眼中最重要的）要素是**excitement（兴奋/热情）**：研究耗时耗力，实验会失败、有截稿日、会被拒，有时很痛苦；如果你真心觉得问题有意思，那么通往它的艰难部分就“worth it”；否则它就只是“work”（苦差事）。

讲师把“重要问题 + 计划 + 兴奋”三者叠成一个筛子：光重要不兴奋，你会在半途被枯燥拖垮；光兴奋不重要，你只是在自嗨；有兴奋有重要性但没计划，就像“攻克癌症”一样无从下手。他还补了一个常被忽略的自检动作——**想象成功**：假设你的计划真的按设想跑通了，那篇论文/那个系统会长什么样？它解锁了什么新可能？社区会因此能做什么以前做不到的事？如果连“成功画面”都描述不清，往往说明问题本身还没被你真正想透。这一串追问和他后面讲的“做自己的二号审稿人”互为表里：前者问“成了会怎样”，后者问“最可能为什么败”，两头都想清楚，才值得押上你几个月到几年的时间。

## 做自己的二号审稿人 [48:53 - 50:10]

讲师提出第四个关键习惯：**be your own reviewer number two**（做自己的二号审稿人）。在踏上为期数月的研究旅程之前，先诚实问自己：“我的想法最可能失败的原因是什么？”如果你能诚实回答、并且仍然相信值得做，那就 proceed。他给出的理由是：这是在保护你最宝贵的研究资源——**你的时间**。

他把这件事类比为投稿时那个“对你格外挑剔的审稿人”：与其等到几个月后实验做完了、被审稿人一句话戳穿，不如在动手前就自己先把最致命的反驳想清楚。能提前回答“为什么可能不work”，本身就是一次低成本的压力测试；如果这个失败理由恰好是你还没解决的，那它同时也就替你选好了下一步该攻的技术难点。讲师强调，博士阶段时间是最稀缺的货币，一次押错方向可能耗掉半年，所以“自我审稿”不是悲观，而是把有限的时间花在“即便暴露弱点也仍值得做”的问题上。

## 方法驱动 vs 问题驱动 [50:10 - 51:48]

有两种研究风格，值得知道自己在用哪一种：

- **Method-driven（方法驱动）**：先有一个想法或技术，再去找适合它的问题。例如“刚发布了一个新的视频扩散模型，我来想办法把它用到机器人操作上”——为一个“ shiny 新模型”找用途。
- **Problem-driven（问题驱动）**：反过来，先有一个具体问题（例如“如何让VLA泛化到新的相机视角”），再去寻找或发明能解决它的方法；问题是锚，方法只是“最能解决问题的那个东西”。

两者都能产出好研究。但对博士而言，他**更推荐问题驱动，因为它更安全**：如果方法没成功，你还可以换一种方法，而你始终在解一个有影响力的问题；而在方法驱动里，若一开始就没有真正有影响力的问题，一旦方法失败你就“没有东西可以退守”。

## 做研究的实操建议 [51:48 - 54:11]

讲师给出一串“loose tips”：

- **从你确信一定能work的简单东西开始，再逐步加难**：先做过拟合实验，别一上来就同时解决所有问题。若你从满复杂度开始又失败了，你根本不知道问题出在哪；只有知道哪部分在起作用，才能推进边界。
- **始终可视化模型的输入与输出**，这点极其重要。
- **多和人聊你的想法**：同事、或你所基于的论文的作者——这能帮你省下数周。
- **有了结果就分享**：分享给关心同一问题的人，他们可能是你下一个合作者。
- **尽可能开源数据与代码**：这是让你的研究对他人有用的最直接方式。
- **花时间打磨图表**：理想情况下，别人只看图就能立刻明白发生了什么，而不用读完整篇论文或图注；若看图需要费劲，他们就会划走。
- **分享时留意受众**：别假设听众已具备和你相同的基础。
- **如果没人知道你的研究，它基本等于没发生**：一篇没人读的好论文影响力为零。把论文放上arXiv、在社媒上宣传。但要在“让人好奇”与“过度炒作”之间找好分寸——别像Brett Adcock那样喊“解决了机器人”，两年过去大家还在等他的“ChatGPT时刻”。

讲师把最后这条与本讲开头的首尾呼应起来：开头那张“每周推特：机器人已被解决”的截图，正是“过度炒作”的反面教材；他并不是让大家藏着掖着，而是说“传播”与“诚实”并不矛盾——把结果讲清楚、把图做漂亮、把代码数据开源，本身就足以让人好奇，不需要靠“solved robotics”这种经不起复现的大词。他还补了一句务实的提醒：分享时心里要装着听众，别默认对方和你一样熟悉所有背景，能把外行讲懂的人，往往也能把审稿人讲懂。

## 一个个人故事：从ROS泥潭到CALVIN基准 [54:11 - 59:21]

讲师用一个亲身经历收束“如何做研究”。博士早期，他深陷经典机器人流水线：上千个ROS节点，感知、规划、控制彼此割裂的模型，错误在整条超复杂流水线里不断累积。更糟的是，在一个重大截稿日前一周，机器人坏了——开机时无法完成自校准，导致他一行代码都跑不了。作为“合格的计算机科学家”，他的做法是“把它关掉再打开”（turning it off and on again），这样折腾了好几天，直到截稿当晚它才不知为何完成了自校准，于是他熬了个通宵把实验全跑完。

这段“traumatizing”的经历转化成一个更大的问题：**能不能把这一切都扔掉？** 忘掉上千ROS节点与不断累积错误的复杂流水线，改做端到端学习？他据此写了一份想法还很粗糙的proposal，竟然被接受并资助了好几年；又说服博士导师给他买了一台**Franka**机器人（他很感激）。随后他花了约一年从零搭起这套平台：带孔的厚重金属桌、自己写VR遥操作代码、装相机、3D打印腕部相机支架、调平滑底层控制器……全是苦活。

有了可用平台后他开始做研究，很快意识到在真实世界里迭代太慢（而他想毕业），于是搭了一个用**同一套VR遥操作栈**的仿真环境。把仿真打磨了一阵后，他发现根本没有一个仿真基准/数据集，能训练那些需要一定泛化能力的、长时程、语言条件的策略。他便想“也许把仿真再打磨一下，做一篇小论文，说不定对别人有用”。这个当初的“副产品甚至事后想法”，最终成了**CALVIN基准**——至今仍是他在被引最多的论文之一，还拿过一个best（paper）奖。在一个以SLAM为主的实验室里搭建这套视觉-运动控制的基建非常艰难，但正因为它，他得以在自己真正想解决的问题上快速迭代。用这套平台大约五年后，他终于做出了那个“很酷的博士毕业演示”：**不跑任何ROS**，用语言接地完成**25个以上**不同操作任务，并能泛化到它们的未见组合。他说，他对此非常满意。

他把这个故事当作前面所有研究建议的“活教材”来收尾：其一，它印证了“多数研究是渐进的、多数副产品未必是主线”——CALVIN最初只是他想快速迭代而搭的“脚手架”，却成了影响最大的工作，说明值得做的基建即使不是目标本身也常有意外回报；其二，它示范了“问题驱动”——他先被“端到端、免ROS、语言可指挥”这个具体痛点锚住，方法（遥操作栈、仿真、基准）都是为解决问题服务的；其三，它体现了“从确信能work的简单处起步、再逐步加难”——先让一台Franka稳定跑起来、先过拟合少数任务，再谈25个任务与泛化；其四，那段“关掉再打开、熬通宵赶deadline”的经历，也正是他说的“研究有时很痛苦、要靠热情撑住”的真实注脚。

## 结课致谢 [59:21 - 62:01]

讲师宣布这是他本课程的最后一讲，下周起只安排客座讲者。他向同学致谢，感谢大家的耐心：这是一门新课，筹备耗费了大量心力，“感觉像一边飞一边造飞机”；也感谢期中给出的反馈，那很有用、 hopefully 让课程更好。这是他第一次独立讲授一整门课、第一次从零设计全部材料，同时还要兼顾两份全职工作（他并非教授，而是在工业界工作），但讲授这门课于他是“absolute privilege”（莫大的荣幸）。看到同学们为策略训练而兴奋、跨团队“盘算着共享数据”、向客座讲者提出有洞察的问题，让他非常开心。他说课程的目标不只是传授内容与基础，更是想**启发大家：机器人学习是一个很值得做的领域**，希望他们传达到了。他期待在Demo Day上看到大家用真机做出的项目，随后把话筒交给Lucas，并特别鸣谢助教团队——设计新课、习题、项目指导、机器人搭建、TA答疑都靠他们，他请助教起立，全场报以掌声。课程在此结束。

## 要点回顾

- **别信“机器人已被解决”**：推特上的“endgame/ChatGPT moment”宣言（Lex Sokolin、Genesis AI的GENE-26.5、Brett Adcock、Jim Fan的Physical AGI路线图）要半信半疑；把机器人挪出训练分布就会迅速失效。
- **明星演示配方**：自采专家遥操作数据→堆到把外插变内插→当天训当天测避免漂移；这能骗过演示，但没解决泛化。
- **泛化层级**：1 object→1 room→1 building→the world；真正目标是“世界”，靠的是拼接知识、推理新颖情境，而非单纯堆数据。
- **两全其美**：非具身基础模型有互联网数据与广泛化、缺物理落地与具身感知；专用具身模型有物理落地与具身感知、数据窄泛化差。
- **三大开放问题（骨干/数据/采集）**：最佳骨干（VLM vs 生成式视频 vs 从零）无定论；数据配方（是否含仿真层、是否把视频当一等公民、是否全真实数据）各有拥趸；采集接口在可扩展性与硬件对齐间权衡（自我中心视频上千万小时但本体差距大，遥操作完美对齐但贵且慢）。
- **灵巧通用策略**：灵巧（高频控制、异构多模态感知）与泛化（基础模型、移动性）原料冲突，目标是维恩图交集。
- **灵巧不止视觉**：无触觉/力反馈会“捏爆”物体；当前VLA接触时仍是“盲”的。
- **感知长尾**：稀缺（Scarcity）、配对（Pairing）、异构（Heterogeneity）；用语言作桥可给VLA赋予从未训练过的触觉/音频（Tactile/Multimodal/Audio-Visual Prompting、Zero-Shot Touch Description）。
- **自适应推理**：按策略置信度分in/near/out-of-distribution，分别对应反应式策略、测试时计算、求助人类；前提是**内省**——模型知道自己不知道，而大模型校准差、这是核心难题。
- **模仿的天花板与RL**：策略被演示限制（学不到1→3捷径）；RL能从次优数据拼接近最优策略，但规模化到真机极难、需算法创新，是最高影响力方向之一。
- **数据飞轮与协同训练**：更多部署→更多数据→更好学习→更强机器人；专家与自主数据存在分布/时间错配，需加权、对齐、联合学习的新算法，目标是“永不丢弃数据”。
- **终身学习**：部署后应持续变好，现实却下滑；要防灾难性遗忘。
- **快速适应**：用语言反馈/同本体演示/跨本体演示做上下文学习，免重训的少样本实时适应。
- **移动操作统一模型**：取代“SLAM栈+硬切换”，讨论环境表示（度量图/高斯泼溅/拓扑图/场景图/BEV/走查视频）、隐式状态估计与全身控制。
- **多模态记忆**：秒/分-时/月-年三个尺度的存、对齐、检索、遗忘。
- **初步工作**：VLMaps（零样本空间导航）、LeLaN（从YouTube学无地图语言导航）、自主改进（基础模型提议+打分）、PALO（5个演示≈100+演示微调）。
- **四大支柱**：具身推理、灵巧移动操作、终身学习、快速适应。
- **研究心得**：多数研究是渐进/多数想法成不了论文/多数论文经不起时间/简单想法更易规模化；好问题=重要问题+可执行计划+让你兴奋；做自己的二号审稿人以保护时间；博士更推荐问题驱动；从确信能work的简单处起步、可视化输入输出、多交流、尽早开源、打磨图表、认真传播但别过度炒作。
- **个人故事**：从上千ROS节点、截稿前一周机器人坏掉，到端到端学习、Franka平台一年搭建、催生CALVIN基准、五年后做出不跑ROS、语言接地的25+任务毕业演示。
