---
title: "第 10 讲 · 具身推理与测试时扩展"
description: "本讲的出发点是「只把机器人遥操作数据继续堆下去够不够」，讲师的答案是「数据当然重要，但它只给你行为先验，缺的那一块是智能推理」，于是把LLM/VLM里最成功的算法改进——推理——迁移回机器人策略。前半部分讲具身思维链（E…"
---

# 第 10 讲 · 具身推理与测试时扩展

> **ETH Zürich 机器人学习 · L10** ｜ 讲者 Oier Mees ｜ 56 分钟 ｜ [原视频](https://www.youtube.com/watch?v=CxhrjQuGEuE)

## 核心结论

本讲的出发点是「只把机器人遥操作数据继续堆下去够不够」，讲师的答案是「数据当然重要，但它只给你行为先验，缺的那一块是智能推理」，于是把LLM/VLM里最成功的算法改进——推理——迁移回机器人策略。前半部分讲具身思维链（Embodied Chain-of-Thought，ECoT）：用Gemini、OWLv2+SAM、Prismatic VLM、Grounding DINO这组基础模型把既有机器人轨迹批量标注成「任务—计划—子任务—移动—边界框—夹爪位姿」的接地推理链，让自回归VLA在输出动作token之前先「想清楚、看清楚」，在同一批机器人数据上不额外采集就拿到约30%的泛化成功率提升（ECoT约65%，OpenVLA约37%，RT-2-X约48%，Octo约18%），并且推理链可被人读、可在测试时改写、还能零样本迁移到从未见过的具身上；代价是控制频率从约4 actions/s掉到约4 s/action（幻灯片给的是0.3–1.2 Hz）。中段用ECoT-Lite回答「为什么有效」：三个假设（更好的表征学习、更好的学习课程、更强的策略表达能力）在LIBERO-90上被逐一隔离，结论是「训练时用推理、测试时丢掉推理」的Reasoning Pre-training与Reasoning Dropout几乎追平原版ECoT（约87%、约89.5%对约91%），而毫无语义的Thinking Tokens反而低于标准VLA（约79.5%对约82%），说明收益主要来自监督信号与表征，不是测试时那些token本身。后半部分转向test-time compute scaling：围棋里AlphaGo靠MCTS从约3050 Elo跳到约5150（超人类线约3500），经验法则是涨120 Elo要么模型翻倍要么测试时搜索翻倍；语言上等FLOPs比较下compute-optimal的测试时扩展在易/中难题打赢约14倍大的模型（+21.6%、+27.8%、+16.7%），但难问题上会反转（−11.9%、−24.3%、−37.2%），因为测试时计算只会放大模型已经知道的东西；理论侧Li et al.给出「任何T步可解的问题都能由常数规模Transformer生成$O(T)$个推理token解出」。最后落到RL：GRPO用组内相对优势替代critic，不需要奖励模型、价值函数和人工标注，DeepSeek-R1的训练曲线显示「wait」这类反思词是从二元正确性奖励里自己长出来的；Visual-RFT把验证器换成IoU证明GRPO对验证器内容完全agnostic；而只奖励最终答案会让「幻觉推理+正确答案」并存，微软的Argos转而奖励中间推理步骤本身。收束到机器人就是分层图景：高层规划用RL训出来的推理，低层连续控制用ECoT-Lite把推理内化后在部署时丢掉。

> 配图说明：本讲32张关键帧只截到49:11，此后约6分半（Argos、回到机器人、四条总结）无配图，相应章节按转写撰写。

## 只扩展机器人数据够不够：本讲的出发点 [00:00 - 04:15]

![课程标题页：机器人学习：从基础到基础模型，第11讲具身推理与测试时扩展](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0001_00-00-00.jpg)

标题页先把编号疑点交代清楚：幻灯片上写的是「Robot Learning: From Fundamentals to Foundation Models / Lecture 11: Embodied Reasoning and Test-time Scaling」，署名Oier Mees，右侧是ETH zürich与Microsoft两个独立logo（提示词表里把它们OCR成一行「ETH zürich Microsoft」，实际是并列的两个机构标志），日期04.05.2026；讲师开场第一句也是「Welcome everybody to lecture 11 of our robot learning course」。也就是说片头自报的是第11讲、2026年5月4日，而本视频在课程发布序列里被标成Lecture 10——与本批次前几集一样（各集标题页日期依次是16.02、23.02、09.03、23.03、30.03、13.04、27.04），播放列表少列了一讲导致视频标题比课程实际编号小一号。这里只如实记录，不改集数标题、也不重排课程编号。

![章节过渡页：仅扩展机器人数据就足够了吗？](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0002_00-00-56.jpg)

第一个章节页把问题写成一句反问：「Is Scaling Robot Data All We Need?」。讲师的态度是先给肯定再打折：过去几讲讲完了怎么在大数据集上训大模型、怎么训VLA，于是很自然会产生「我们只要继续采遥操作数据、把LLM那套scaling recipe搬过来就够了」的想法——「这是真的（this is true）」，但不是全部故事。他特意强调自己不是说数据不重要，数据绝对重要。

关键在于数据到底给了你什么：它给你一个更好的行为先验（a better prior over the behaviors），把演示分布压缩进模型权重。那么缺的是什么？

![对比图：左侧为多任务机器人图像网格，右侧为ChatGPT与DeepSeek标志，指出缺失的是智能推理能力](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0003_00-01-51.jpg)

幻灯片标题「What is Missing? Intelligent Reasoning」，左边是一张多任务机器人的图像网格（标注「Robots capable of many tasks」），右边是ChatGPT与deepseek两个logo，下面一行把智能推理拆成四个维度：Generalization、Interpretability、Interaction、Controllability（泛化、可解释、交互、可控）。讲师的解释是：即便我们造出一个能做很多任务的机器人，如果它不能智能地推理，也不会太有用。本讲到目前为止见过的所有模型，哪怕是最强的VLA，一旦遇到分布外情形，通常没有一种机制让自己「智能地把局面救回来」，它们就是失败掉——这正是与人类智能的差别所在。

他还补了一层数据侧的类比：LLM已经在远超最大VLA好几个数量级的数据上训练，把整个互联网都吞下去了，结果仍然是「这些数据不足以解决我们想让LLM解决的所有问题」；而机器人数据比文本贵好几个数量级，那么光靠数据就更不可能够。LLM现在的替代路线是专门为推理而训练，其核心直觉是推理是组合式的（compositional）：它由可复用的逻辑与知识积木组成，重新组合就能用已有知识解决没见过的难题。这意味着只要你能够从第一性原理推过去，就不必事先见过那个具体情形。机器人也需要同样的性质：泛化到训练分布里那些行为之外。

## 机器人学作为多模态序列建模：能不能把「推理」迁移过来 [04:15 - 05:34]

![将机器人学视为多模态序列建模的幻灯片，展示语言、图像、动作三种模态的对应关系](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0004_00-04-38.jpg)

幻灯片「Robotics as Multimodal Sequence Modeling」列了两个问题：「Can we transfer algorithmic improvements like reasoning?」「Can a robot think harder to act better?」。图下方把三种模态并排画成token条：Language（绿）、Image（蓝）、Action（紫），各三个椭圆token，下面分别配上「Pick up the spoon」的指令框、一张桌面场景照片（毛巾、勺子、黄瓜、西瓜片、蘑菇，右上角露出夹爪），以及一个WidowX式机械臂图标配「ACTION: $[\Delta x, \Delta\theta, \Delta\text{Grip}] = \dots$」。

讲师的论证链条很短：我们已经把机器人学表述成多模态序列建模——吃进图像、语言和动作token；也已经把LLM/VLM的很多进展迁了过来（模型架构、预训练目标、scaling策略等）；那么自然要问，那些显著提升LLM/VLM泛化能力的算法改进，比如推理，能不能也迁过来。于是本讲的主问题就是：不再只是扩展预训练，而是让机器人「想得更久一点」能不能让它「做得更好」。

## 具身思维链（ECoT）：在图像与动作之间插入接地的中间推理 [05:06 - 07:29]

![具身思维链推理（Embodied Chain-of-Thought）的流程图，展示从任务分解到动作输出的推理过程](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0005_00-05-34.jpg)

第一篇工作是「Robotic Control via Embodied Chain-of-Thought Reasoning」，Zawalski\*、Chen\*、Pertsch、Mees、Finn、Levine，CoRL 2024（页脚原样如此，两位共同一作打了星号）。幻灯片标题「Embodied Chain-of-Thought Reasoning」，副标题「Break down hard problems into intermediate reasonings」。左侧图是同一张桌面照片，西瓜片被红框框住、毛巾被蓝框框住，一条绿色曲线从夹爪上的红点画到西瓜，表示预测的移动轨迹；右侧是一段对话气泡式的推理输出：

- 用户：Put the watermelon on the towel
- Plan: 1. Move to watermelon, 2. Grasp it, …
- Subtask: Move to watermelon
- Move: Move backwards
- Bboxes, Gripper Pos:
- ACTION: $[\Delta x, \Delta\theta, \Delta\text{Grip}] = \dots$

下面四条要点是这篇工作的论证结构：标准策略直接把图像映射到动作，因此在新颖任务上常常失败；我们提出给策略加上具身推理；具身推理让策略「look」和「think」得很仔细；这带来更好的性能、可解释性和泛化。

核心想法很简单：不要让VLA从「图像+指令」直接跳到动作，而是中间插入若干推理步骤。给定一条具体指令，先让模型生成一个计划（比如「移动到西瓜」），再识别策略接下来该做的子任务，并且把这些推理同时接地（ground）到图像上——这就是「思维链，但作用于物理世界」。

## 推理数据从哪来：蒸馏一组基础模型批量标注既有轨迹 [06:14 - 07:47]

![利用基础模型蒸馏大规模生成多模态推理数据的架构图](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0006_00-06-30.jpg)

幻灯片「Creating Multimodal Reasonings at Scale」，副标题「Maximally use existing data by distilling foundation models」。第一个问题就是数据从哪来：你不可能雇标注员给大规模机器人数据集里每一条轨迹都标上计划、子任务描述、边界框、夹爪位置——那不可扩展。答案正如你可能猜到的：用互联网规模的基础模型去标注你已有的机器人数据集，生成语义接地的推理，比如下一个要被末端执行器交互的物体的边界框、夹爪未来如何移动的轨迹等等。

图上的流程是：左边一叠「Robot Trajectory from Dataset」（带指令「Put the watermelon on the towel」和「Proprio 1: $[\Delta x, \Delta\theta, \Delta\text{Grip}]$」的帧序列），中间竖排四个基础模型logo——Gemini、OWLv2 + SAM、Prismatic VLM、Grounding DINO，其中一个绿色小气泡写着「This scene has a watermelon, spoon, towel, and mushroom.」，右边是一个REASONING框，逐字段生成：

- TASK: Place the watermelon on the towel
- PLAN: 1. Move to watermelon 2. Firmly grasp it 3. …
- SUBTASK: The watermelon is the first object the robot needs to interact with → Move to the watermelon
- MOVE: The watermelon is behind the robot, so it needs to move backward → Move backward.
- GRIPPER POS: $[156, 55]$

## 具身思维链架构：先「想清楚」再「看清楚」，然后才动 [07:07 - 07:47]

![具身思维链架构流程图，展示从输入指令到生成任务规划、子任务推理、移动推理、夹爪位姿及最终机器人动作的完整链路](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0007_00-07-25.jpg)

架构图把整条链路摊开：一条水平时间轴上依次挂着Input、Generated Embodied Chain-of-Thought、Robot Action三段。输入是场景图加「USER: "Put the watermelon on the towel"」；生成的具身思维链依次是TASK（Place the watermelon on the towel）、PLAN（1. Move to watermelon 2. Firmly grasp it 3. Move to towel 4. Place watermelon on towel）、SUBTASK REASONING（The watermelon is the first object the robot needs to interact with. The robot is not yet close to the watermelon, so the robot needs to move closer）、SUBTASK（Move to the watermelon）、MOVE REASONING（The watermelon is behind the robot, so it needs to move backward）、MOVE（Move backward）、GRIPPER POS（$[156, 55]$）、VISIBLE OBJECTS（Watermelon $[126, 146, 141, 125]$，Towel $[20, 59, 218, 198]$，Spoon $[114, 93, 141, 125]$）；最后才是Robot Action「ACTION: $[\Delta x, \Delta\theta, \Delta\text{Grip}] = \dots$」。

讲师复述这套架构的要点：训练一个自回归的视觉-语言-动作模型，让它先预测这些中间接地的推理步骤（任务、计划、移动等），再预测动作token；等价于强迫模型在行动之前「think carefully」和「look carefully」，动作token因此被条件在这些辅助推理token之上。

## 泛化评测：同样的机器人数据，成功率提升约30% [07:47 - 09:25]

![泛化性评估柱状图，对比ECoT与RT-2-X、OpenVLA、Octo在成功率上的表现](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0008_00-08-21.jpg)

幻灯片「Generalization Evaluations」，标题下写着「Boosts performance 30% without extra robot data」，图内小字是「Our Embodied Chain-of-Thought VLA is more performant than other generalist policies, especially on novel tasks」。这是一张横向柱状图，横轴Success Rate从0到70（每10一格），四条柱子从上到下读数约为：

| 策略 | 成功率（读图） |
|---|---|
| ECoT (Ours) | 约65 |
| RT-2-X | 约48 |
| OpenVLA | 约37 |
| Octo | 约18 |

（视觉档案的补充信息把ECoT记成约62%、RT-2-X约47%、OpenVLA约36%，看图后按柱端与网格线的位置修正为上表；ECoT的柱端明显越过60与70的中点。）

评测设置本身也全部是分布外的：干扰物体、空间关系、没见过的物体、没见过的指令与场景。讲师强调这个结果相当显著：套用具身思维链这项技术，相对OpenVLA（同一个模型但不带推理）带来约30%的性能提升；同时也打赢了Google那个大得多的RT-2-X——55B参数的VLA，而他们的模型只有7B。他要传达的关键信息是：在完全不额外采集机器人数据、只用你已经采集的那批遥操作数据的条件下，光是加上具身推理就能把模型的泛化能力提升约30%。

## 可解释与可交互：测试时人工改写推理链，动作会自我纠正 [09:25 - 10:25]

除泛化性能之外的附带收益是可解释性与可交互性（interactiveness）。推理步骤是人类可读的：你能看到它在预测哪些边界框、给出什么计划。测试时如果发现机器人预测了一个错误的计划，你可以在线编辑这条推理链；由于动作token被条件在这些中间推理token上，模型会自然地自我纠正——相当于人类在推理阶段做了一次干预。讲师认为这很酷：整个模型因此变得可解释，失败时你能看出它为什么失败。

## 推理跨具身迁移：只见过WidowX，却能给陌生机器人写推理 [10:25 - 13:02]

![展示推理能力泛化到未见过的机器人上的四个任务示例，包含任务指令、规划步骤和视觉标注](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0009_00-10-25.jpg)

幻灯片「Reasonings Generalize to Unseen Robots」给出四个真实画面，每个画面左侧是带接地标签的场景，右侧是模型生成的TASK/PLAN/SUBTASK/MOVE四段推理。四组接地标签分别是：a white plate、an orange orange、a black bowl、a blue sink；a red bottle、a robot、a yellow banana；a black robot arm、a carrot、a red bowl；a yellow towel、a grey sponge（右下那张图角上标着「4X」，是回放倍速水印）。四段推理文本要点：

1. TASK: Put the black bowl in the blue sink. PLAN: Move to the black bowl. Pick up the black bowl. Move to the sink. Drop the black bowl in the blue sink. SUBTASK: The black bowl is the first object to interact with → Move to the black bowl. MOVE: The black bowl is to the right of the robot → Move right.
2. TASK: Move to the banana. PLAN: Move to the banana, grasp the banana, move the banana to the plate, release the banana. SUBTASK: The banana is on the left side of the plate, so the robot needs to move left → Move to banana. MOVE: The robot needs to move left to get closer to the banana → Move left.
3. TASK: The task is to put the carrot in the red bowl. PLAN: 1. Move to the carrot. 2. Grasp the carrot. 3. Move to the red bowl. 4. Release the carrot. SUBTASK: The robot has grasped the carrot and needs to move it to the red bowl → Move to the red bowl. MOVE: The red bowl is to the right of the robot, so the robot needs to move right to reach it → Move right.
4. TASK: Put the grey sponge on the yellow towel. PLAN: Move to the sponge, pick up the sponge, move to the towel, put down the sponge. SUBTASK: The sponge is on the table, and it needs to be reached first → Move to the sponge. MOVE: The sponge is below the robot, so move down → Move down.

讲师说这是全篇最漂亮的实验之一：这个带推理的VLA只在单一数据集上训练过——就是WidowX那类便宜的黑色末端执行器机器人——然后被要求给Open X-Embodiment里其他数据集的机器人产出推理。尽管机器人完全新颖、相机完全不同、一切皆新，它仍然给出相当合理的推理链。

为什么会这样？因为其中很多推理任务其实和VLM预训练任务非常相似，VLM本来就带着这套世界模型；具身思维链做的事情，是把VLM因预训练而已经拥有的推理能力「解锁」出来。由此得到的推论是：具身思维链可能是跨不同具身迁移推理的一种主要方式，而不必为每一种新机器人形态都去采集并标注演示。

小结这篇工作的核心贡献：建立了一种用具身推理训练VLA的新方式，证明可以在没有任何额外机器人数据的情况下同时改善泛化、可解释性与可交互性。此后它被学术界和工业界同时采纳——例如Gemini Robotics的VLA就把具身推理作为核心组件之一。讲师认为这是一个有用的信号：说明整个领域正在收敛到「把推理当作机器人策略的一等公民」。

## 具身推理的后续工作版图 [13:02 - 13:44]

![具身推理后续工作汇总页，包含MolmoAct、CoT-VLA、RAD和自监督引导四个模型的结构图](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0010_00-13-02.jpg)

一页四张图，四个方向：

- 左上：MolmoAct: Action Reasoning Models that can Reason in Space（Lee et al., 2025.）。图里是Pre-Trained Action Reasoning Model，输入可以是语言指令「Pour water into the cup」或者「Trajectory Input」，输出三样东西：Depth Perception Tokens、Visual Reasoning Trace、Action Tokens，其中推理轨迹与动作token里显式带Left/Right的坐标数值。讲师的说法是它「加了带深度token的空间推理」。
- 左下：CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models（Zhao et al., 2025.）。图上区分pre-training data（Visual QA、Captioning、Text to Image/Video）、robot demonstrations、action-less videos三种数据源，主干是Causal & Full-Attention，先出a1…an的推理再execute actions，并注明「capture new observations / closed-loop control cycle」。
- 右上：Action-Free Reasoning for Policy Generalization（Clark et al., 2025.），图内模型名为RAD，页眉写「RAD learns broad reasoning from human + robot data and action generation from just robot data!」，并把Action-Free Data（Reasoning）与Robot Data（Reasoning + Action）两条支路分开画，例子里有「Pick up the controller」「Position above the handle, move down, close the gripper and move up」这类推理文本。讲师的说法是把来自人类数据的「无动作推理」迁移过来，即把推理的训练与动作预测解耦。
- 右下：Self-Supervised Bootstrapping of Action-Predictive Embodied Reasoning（Ganai et al., 2026.）。图上分(a)(b)两块，出现Reasoning Dropout、Prior VLA、Posterior VLA、Importance Weights、Categorical Sample、Train Reasoning VLA、Reasoning VLA等组件，以及Plan/Vio.Obj./Subtask/Ego这些推理token类型。

讲师鼓励大家去读这几篇，因为它们能给出「具身推理这个方向往哪走」的较完整图景。

## 代价：推理很慢——从约4 actions/s退化到约4 s/action [13:44 - 15:47]

一切看起来很好，但有个明显问题：推理不幸地很慢，因为你把测试时生成的token数量显著增加了。OpenVLA作为这套工作的基座模型是个7B模型，大约每秒4个动作；而他们的具身推理版本因为要产出完整推理链，变成每个动作约4秒——差了大约16倍。讲师回忆，虽然之后做了一些推理速度优化，但两位共同一作（Will与Michal）第一次在真实WidowX机器人上跑rollout时「看着都很痛苦」：做一个任务要等好久，频率大概1 Hz上下。

于是问题变成：机器人学者在乎快速控制、要实时控制，也不想花好几个小时看机器人以乌龟速度移动。我们能不能拿到推理的好处，却不付这份推理开销？

不过在跳到解法之前，讲师认为值得先停下来问一个更根本的问题：具身推理到底为什么会有效？他特别提醒，机器学习领域有个倾向：看到某个东西有效（比如泛化性能高30%）就兴奋，然后奔向下一个热点；但如果不理解它为什么有效，我们就不知道它什么时候会失败，也不知道怎么改进它，更没法有原则地在它之上继续建设。他主张：我们做实验失败时那种科学严谨，同样应该用在实验成功的时候。接下来几页幻灯片（他注明改编自Will，即William Chen）给出几个听起来合理的答案。

## 先问「为什么有效」：三个假设 [15:47 - 21:21]

![具身推理为何有助于机器人策略的三个原因：更好的表示学习、改进的学习课程、增加策略表达能力](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0011_00-15-47.jpg)

幻灯片标题「Why Does Embodied Reasoning Help Robot Policies?」，三条并列：1. Better representation learning（橙）、2. Improved learning "curriculum"（蓝）、3. Increased policy expressivity（绿）。页面左下角的脚注是「[ECoT-Lite Slides adapted from William Chen]」——这印证了转写里那句「adapted from Will」指的是William Chen。

![具身推理提升机器人策略的第一点：更好的表征学习架构图](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0012_00-17-38.jpg)

第一个假设：更好的表征学习。画法是把INSTRUCTION与OBSERVATION（带机械臂图标）放进虚线框，经过一个神经网络图标后分出两条箭头，一条指向虚线框的REASONING（被划掉的样子，表示测试时不要），一条指向实心的ACTION，右下角注明「Train on both, but only choose actions at test time」。直觉是推理轨迹以某种方式提供了额外知识，塑造了模型的内部表征——比如指出「某个物体的位置是相关的」「某种移动在这个场景里重要」，从而迫使模型建立更丰富、语义更接地的场景表征。

如果这是主要解释，那么真正重要的是训练时来自推理的监督信号本身，而不必在测试时真的生成推理：即使跳过测试时的推理生成，模型的表征也已经聚焦在正确的特征上。这个假设若成立，结论非常诱人：我们可以带着推理训练以获得更好的表征，然后不带推理部署，从而把低延迟和高频率拿回来。

![具身推理提升机器人策略的第二点：改进学习课程架构图](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0013_00-18-34.jpg)

第二个假设：改进的学习课程。图里REASONING被挪到输入侧的虚线框内（与INSTRUCTION、OBSERVATION并列），输出只有ACTION，旁边两行字：「Predicting continuous robot actions is very OOD for a VLM, is ECoT a better curriculum?」「In-context reasoning scaffolds policy → Akin to teacher-student learning!」。直觉是：对一个在视觉问答等任务上预训练过的VLM来说，预测连续的机器人动作是极其分布外的；直接把VLM端到端微调去产生连续动作非常难，那完全是跟预训练不同的事情。于是具身推理可能提供了一种隐式课程：模型先学更简单一点的中间任务（比如把移动描述映射到动作），再逐步爬到「从像素到动作」这个完整预测问题。

若这个假设成立，实践含义是：可以把推理纯粹当作训练脚手架——训练时把推理例子放进上下文，但不在它们上面加loss，然后在推理时撤掉。这样仍然拿到泛化收益，却不承担推理开销。

第三个假设：增加策略表达能力。想法是仅仅让上下文里有更多token，哪怕这些token在语义上没有意义，也会增加Transformer在预测动作时的表达能力。讲师认为这个假设非常可信，因为LLM研究里已经很好地记录了这一点，幻灯片上还挂了两篇相关参考文献；结论是Transformer可以把中间token当作隐藏计算的草稿空间（scratch space），即使这些token不携带任何语义也能提升性能。

这个假设提出一个很重要的问题：具身推理的收益里，有多少来自推理的语义内容，有多少只是额外token带来的额外计算预算？如果主要是后者，那么推理的具体接地程度就没有我们以为的那么重要，我们可以用更便宜的方式拿到同样的收益，而不必用基础模型去产出很漂亮的标注。

## LIBERO-90上隔离三种机制 [21:21 - 22:26]

![LIBERO-90基准上不同训练策略的成功率对比条形图](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0014_00-21-21.jpg)

ECoT-Lite这篇工作（页脚：Training Strategies for Efficient Embodied Reasoning，Chen、Belkhale、Mirchandani、Mees、Driess、Pertsch、Levine，CoRL 2025）在LIBERO-90基准上把每种机制单独隔离出来做实验。左侧是LIBERO-90的截图与论文图，右侧是横向条形图，横轴LIBERO-90 Success Rate只画了70、80、90三格，六根柱子读数约为：

| 训练/推理策略 | LIBERO-90成功率（读图） |
|---|---|
| Embodied CoT | 约91 |
| Reasoning Dropout | 约89.5 |
| Reasoning Pre-training | 约87 |
| Reasoning "Scaffolding" | 约84 |
| Standard VLA | 约82 |
| Thinking Tokens | 约79.5 |

Reasoning Pre-training与Reasoning Dropout两根被同一个橙色虚线框圈起来，表示这是本文提出的两个变体。

需要纠正视觉档案的一处说法：它记成「Reasoning Dropout在LIBERO-90上取得了最高的成功率，接近90%」，但看图可知全场最高仍是Embodied CoT（略过90，约91），Reasoning Dropout只是在两个lite变体里最高、并非常接近原版。

讲师对着图逐条解读：Reasoning Pre-training与Reasoning Dropout（两者都在测试时移除推理）表现几乎和完整的原版具身思维链一样好；Reasoning "Scaffolding"（把推理作为输入上下文提供、但不加loss）也做得不错；Thinking Tokens（用空token）这里他先说「虽然比标准VLA好……不对，它不是更好」，随即自我纠正：它比标准VLA还差一点，所以并没有帮助。也就是说，纯粹多给token而不给语义，收益是负的；而带语义的推理即使测试时不生成也仍然有效——这指向表征学习与课程这两个假设，而不是「额外计算预算」那个。

## ECoT-Lite的两种做法：Reasoning Pre-training与Reasoning Dropout [22:26 - 24:25]

![Embodied Chain-of-Thought Lite 的推理预训练与推理丢弃流程图](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0015_00-24-08.jpg)

幻灯片标题「Embodied Chain-of-Thought Lite」，上面两个橙色小标题Reasoning Pre-training、Reasoning Dropout。图里左边是仿真桌面（「Put middle bowl on plate」），中间是Policy方块，接一个绿色虚线框的Reasoning块，内容「Plan, Subtask, Move, Bounding Boxes, Gripper」，框上贴着「Randomly Dropped」标签，右边输出Action「$[\Delta x, \Delta\theta, \Delta\text{Grip}]$」。页脚同样是CoRL 2025那篇。

讲师解释这两种做法其实很好懂。

Reasoning Pre-training很直接：两个训练阶段。第一阶段拿你的VLM只微调去预测推理，完全没有动作loss、不做动作预测；训完之后，第二阶段在这个模型上继续微调去预测连续的机器人动作，但不再预测推理。你转移的只是第一阶段学到的表征，因此推理时不需要付出生成完整推理的开销，两者被解耦。

Reasoning Dropout则是单一联合训练阶段（和原版具身思维链一样，联合训练模型同时预测推理与动作），区别在于训练时以某个概率把推理丢弃掉，让模型也学会「不在推理条件下」产出动作token。这样测试时你就可以自己选择：要不要在预测动作token之前先生成推理，或者把它关掉。等于给了你一个推理开关。

## 真机Bridge WidowX：性能与频率的帕累托 [24:25 - 26:08]

![真实世界 Bridge WidowX 设置下的策略性能与推理频率对比图表](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0016_00-25-03.jpg)

同一篇工作的真机结果，标题「Results in Real-World Bridge WidowX Setup」。左图是Policy Performance对Inference Freq. (Hz)的二维示意：Embodied CoT在左上（性能最高、频率最低），Standard VLA在右下（性能最低、频率最高），ECoT-Lite Variants (Ours)那个橙色圈落在右上偏中——性能接近原版而频率接近标准VLA，一条灰色曲线从Embodied CoT经过它连到Standard VLA。图下小字：「ECoT-lite is faster than reasoning policies and stronger than non-reasoning policies」。这里要纠正视觉档案的补充信息：它写成「ECoT-Lite在保持较高推理频率的同时达到了最高性能」，看图可知最高性能仍属于Embodied CoT，ECoT-Lite是「第二高但快得多」。

右侧两组柱状图，横轴刻度分别是40/60/80与0/2/4：

| 策略 | Bridge Success Rate | Inference Rate (Hz) |
|---|---|---|
| Embodied CoT | 约78 | 约0.3–1.2（画成带斜纹的低柱） |
| Standard VLA | 约51 | 约3.6 |
| Reasoning Pre-training | 约68 | 约3.5 |
| Reasoning Dropout | 约62 | 约3.5 |

底部结论句：「More Performant than Standard VLAs without sacrificing speed!」

讲师的读图口径：原版具身思维链（推理时生成完整推理）给出最高的泛化成功率，但代价是最低的推理频率；Reasoning Pre-training与Reasoning Dropout这两个变体在泛化性能上略差一点，但基本拿到和标准VLA一样的频率。

![三种具身推理方法的优缺点对比总结表](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0017_00-25-59.jpg)

选择哪一种是取决于约束的。这页「Which Embodied Reasoning Approach is Best for Me?」把三列并排压在一片真机rollout照片网格上：

- Embodied CoT：(+) Most performant，(−) Slowest (0.3 – 1.2 Hz)
- Reasoning Pre-training：(+) Unpaired reasonings，(−) More training steps
- Reasoning Dropout：(+) Can turn reasonings on/off，(−) More training memory

对应讲师的建议：原版对多数实时设置来说太慢；Reasoning Pre-training特别适合「你的推理数据和动作数据不是配对的」情形——比如你想从仿真或人类数据这类非机器人数据源迁移推理能力，那就走这条路；Reasoning Dropout最灵活，因为单个模型就能按你的延迟预算把推理开开关关。

到这里为止，所有方法在测试时用的都是固定的计算预算。自然的追问是：如果让模型在更难的问题上想得更久呢？推理时更多的计算能不能换来更好的决策？因为原版具身思维链里，推理步骤的数量是训练时定死的——比如七个不同的推理步骤，那它就永远产出全部七个。

## 测试时计算扩展是不是就是规划 [26:09 - 28:12]

![对比经典规划、学习式规划与LLM推理在五个维度上的架构图](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0018_00-27-50.jpg)

幻灯片「Isn't LLM Test-Time Compute Scaling just Planning?」把三列方法、四行维度做成对照表，三列从左到右是Classical planning（A\*, STRIPS, PDDL）、Learned planning（Chess, Go, Atari）、LLM reasoning（CoT, GRPO, R1），每格下面还有一行小字：

| 维度 | Classical planning | Learned planning | LLM reasoning |
|---|---|---|---|
| World model | Hand-specified（Transition function） | Learned（Policy + value network） | Implicit in weights（No explicit model） |
| Search / thinking | Explicit tree search（A\*, BFS, DFS） | MCTS rollouts（Many sims per move） | Token generation（Sequential reasoning） |
| Verifier | Goal test（Exact, hand-specified） | Win / loss（Exact, from game rules） | Rule-based reward（Math grader, unit test） |
| Scope | Known domains（Structured, discrete） | Discrete games（Bounded action space） | Open-ended tasks（Language, code, robots） |

讲师的说明：这个谱系从「完全手工指定的域」一路走到「用神经网络、把模型隐式在权重里做测试时计算扩展」，从结构化的离散域走到语言编码这类开放式任务；每一步都是用显式结构去换通用性（each step trades off explicit structure for generality）。所以它像规划，但把世界模型、搜索、验证器都换成了隐式的、可学习的形式。

## AlphaGo：围棋里test-time search的量级 [28:12 - 30:49]

![围棋中各模型Elo评分柱状图，展示测试时搜索带来的性能提升](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0019_00-28-46.jpg)

测试时计算扩展的典范例子是AlphaGo。这页「Scaling Test-Time Compute in Go」的柱状图（页脚Mastering the Game of Go without Human Knowledge，Silver et al.，Nature 2017；左下角另有一行「[Slides adapted from Noam Brown]」，这是视觉档案漏掉的致谢）纵轴Elo Rating从0到5000，八根柱子读数依次约为：Raw Network约3050、AlphaGo Zero约5150、AlphaGo Master约4850、AlphaGo Lee约3700、AlphaGo Fan约3150、Crazy Stone约1900、Pachi约1270、GnuGo约400。一条红色虚线横在约3500处，标注Superhuman Performance；「Full AlphaGo Zero」的箭头指向最高的那根柱，「No test-time search」的箭头指向Raw Network那根。

页面左侧的经验法则原文：「Rule of Thumb, increasing 120 Elo points requires either: 2x model size, 2x test-time search」——涨约120 Elo，要么把模型规模翻倍，要么把测试时搜索翻倍，两者可以互换。

讲师的算法很有意思：从原始策略网络（不带任何搜索）的约3000 Elo，到加上测试时蒙特卡洛树搜索后跳到超过5000 Elo、直接进入超人类区间；如果按那条经验法则不用搜索、只靠把模型做大来补齐这段差距，大约需要把模型放大到十万倍量级。他马上自己打折：这个数字可能被夸大了，因为AlphaGo是靠跟自己早期checkpoint对弈训练出来的，真实数字大概没那么高。但一个有意思的事实是：据他所知到今天为止，没有任何围棋模型在不使用测试时搜索计算的情况下超过人类水平——纯粹把模型做大做不到；每一个围棋agent都用测试时搜索。结论是测试时搜索能带来巨大的真实收益。

当然，围棋、扑克这类游戏设定里有完美的验证器和定义良好的搜索空间。问题是：在更开放的语言任务上，这还成立吗？

## 语言里的test-time compute：等FLOPs打赢14倍大模型，但难题反转 [30:49 - 33:10]

![语言模型中测试时计算扩展在不同难度问题上的准确率提升柱状图](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0020_00-31-33.jpg)

答案是「成立，但有个转折」。Charlie Snell等人的工作（页脚Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters，Snell et al, 2024）显示：在FLOPs匹配的对比下，compute-optimal的测试时扩展在易/中难度问题上打赢大约14倍大的模型。幻灯片顶部原句就是「Compute-optimal test-time scaling beats a ~14× larger model on easy/medium problems at equal FLOPs」。

图的标题是「Comparing Test-time and Pretraining Compute in a FLOPs Matched Evaulation」（原图里Evaulation这个拼写就是这样，非笔记笔误），纵轴Relative Improvement in Accuracy From Test-time Compute (%)，范围−40到30；横轴Ratio of Inference Tokens to Pretraining Tokens，分三组：<<1、~=1、>>1；绿=Easy Questions、蓝=Medium Questions、橙=Hard Questions。九个柱子的标注值：

| 推理token/预训练token比 | Easy | Medium | Hard |
|---|---|---|---|
| <<1 | +21.6% | +27.8% | +11.8% |
| ~=1 | +16.7% | +3.5% | −11.9% |
| >>1 | +5.4% | −24.3% | −37.2% |

右侧两条机制说明：SFT on own failed attempts + learned verifier（模型学会在先前答错的基础上产出更好的答案，验证器挑出最好的那个）；Generate multiple answers at test time, select best with verifier。

讲师的解读：这个结果之所以惊人，是因为同样的计算预算下，你往往最好是训一个更小的模型、让它在推理时想得更久，而不是去训一个更大的模型。这里的机制本质是在模型自身的输出分布上、由奖励信号引导做一次搜索。

但他说这张图最有意思的地方在于：难题上情况反转。绿色是易题、蓝色是中题、橙色是难题，对难题在某些设置下你会损失到−37.2%（转写里他说成「lose up to 30 uh 33.7%」，看图应以柱上标注的−37.2%为准；视觉档案的OCR这一串数值本身是对的）。直觉解释是：测试时计算在本质上是放大模型已经知道的东西；如果正确答案根本不在模型自己的分布里，再多的搜索也救不回来。

## 为什么推理token有效：$O(T)$的理论解释 [33:10 - 34:46]

![解释推理token为何有效的理论幻灯片，涉及常数规模Transformer与O(T)计算步数](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0021_00-33-24.jpg)

「Why Do Reasoning Tokens Help?」这页给了一个漂亮的理论答案（页脚Chain of Thought Empowers Transformers to Solve Inherently Serial Problems，Li et al, 2024）。三条要点原样：

- Any problem solvable in T computational steps can be solved by a constant-size transformer generating $O(T)$ reasoning tokens
- Generating $O(T)$ intermediate tokens before the answer gives the model $O(T)$ effective computation steps
- Without reasoning tokens: needs a much deeper/larger model or can't solve the problem at all

讲师的直觉版：Transformer的一次前向传播有固定的计算深度，受层数限制，所以单次前向只能执行常数数目的串行步骤；但通过自回归地生成中间token，模型可以在测试时把额外计算「展开」（unroll）出来。于是没有推理token时，要解一个需要$T$个串行步骤的问题就需要一个大得多的模型；有了推理token，常数规模的模型只要「想得更久」就能解。他说这正是前面第三个假设（Thinking Tokens）那件事的形式化对应版本。

## 思维链的历史起点：CoT提示与它的涌现条件 [34:46 - 37:07]

![标准提示与思维链提示在模型输入输出上的对比示意图](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0022_00-35-16.jpg)

推理token有用，但这个想法在实践里从哪来？原始的思维链论文（页脚Chain-of-Thought Prompting Elicits Reasoning in Large Language Models，Wei, Wang et al, NeurIPS 2023）之所以美，是因为它极其简单：与其让模型直接输出答案，不如给它看几个把推理步骤明确写出来（比如由人类写）的例子，模型就学会对新情形做同样的事。幻灯片副标题「More tokens = more computation steps」，左右对照Standard Prompting与Chain-of-Thought Prompting，各自分Model Input与Model Output两块。

标准提示的输入是「Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now? / A: The answer is 11.」加上待答的「Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?」，输出是「A: The answer is 27.」并打红叉。思维链提示把示例答案换成完整的算式「A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.」，模型输出变成「A: The cafeteria had 23 apples originally. They used 20 to make lunch. So they had 23 - 20 = 3. They bought 6 more apples, so they have 3 + 6 = 9. The answer is 9.」打绿勾。

![思维链能力在大模型中的涌现现象：不同模型规模下标准提示与思维链提示的准确率对比曲线](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0023_00-36-11.jpg)

但这篇论文里有个坑：思维链提示在小模型上并不好用。「CoT Emerges in Large Models」这页是原论文的缩放图，三列模型族LaMDA（横轴刻度0.4、8、137）、GPT（0.4、7、175）、PaLM（8、62、540），三行数据集GSM8K solve rate (%)、SVAMP solve rate (%)、MAWPS solve rate (%)，图例三种线：Standard prompting（灰）、Chain-of-thought prompting（蓝紫圈）、Prior supervised best（橙色虚线，分别约在56、47、89的水平）。可以看到小模型上思维链那条线常常贴着甚至低于标准提示，直到规模上去才突然抬头，在PaLM这类规模上有时甚至超过此前最佳的监督方法。

讲师的结论：推理是涌现出来的，它需要一个足够好、足够大、足够强的基座模型；如今你可以把它蒸馏进更小的模型，但一般来说推理的涌现需要强基座。

## 原始CoT=少样本上下文学习，以及它的脆弱 [37:07 - 38:19]

![原始思维链提示的少样本/上下文学习格式：给出带推理步骤的示例后让模型回答新问题](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0024_00-37-07.jpg)

「Original CoT Prompting = Few-shot/In-Context Learning」这页把格式写成最朴素的六行：Q: [problem 1] / A: [step 1… step 2… therefore answer] ✓ / Q: [problem 2] / A: [step 1… step 2… therefore answer] ✓ / Q: [new problem] / A: ???。也就是手写几个带推理轨迹的例子、prepend到提示前面、让它泛化到新案例。

讲师指出这样做的局限：它对提示很敏感、很脆弱——放哪些示例会改变结果，甚至可能是截然不同的结果；需要人工（或某种手工劳动）去挑选并把这些例子塞进上下文；而且大概不会泛化到你提供的那些具体例子或领域之外。

![提出核心研究问题的过渡幻灯片：能否训练模型内在推理而非依赖人类提示](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0025_00-38-03.jpg)

于是问题变成下一页那句：「Instead of eliciting reasoning with human prompts, can we train the model to reason intrinsically?」——与其用人类提示去把推理引出来（elicit），能不能训练模型自己去推理？这是社区接着研究的方向。

## 从「提示出推理」到「训练出推理」：GPT-4o的单链思维与DeepSeek-R1的思考过程 [38:19 - 40:13]

![单链思维与DeepSeek-R1思考过程在数学应用题上的回答对比图](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0026_00-38-59.jpg)

这页来自DeepSeek-R1 Thoughtology: Let's think about LLM reasoning（Marjanović et al, 2026），用同一道应用题对比两种模型的输出。题目（黄色圆角框）：「James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?」

上半Single Chain of Thought是GPT-4o's Response：一段干净、结构化的分步解答——1. Determine Pages per Letter（James writes a 3-page letter）；2. Determine Number of Friends（2 different friends）；3. Calculate Pages per Week（每封信3页 × 每周两封 × 2个朋友 = 12 pages/week）；4. Calculate Pages per Year（一年52周）；5. Total pages per year = 12 × 52 = 624 pages/year，最后把「624 pages a year」用紫框框出来。

下半Thinking Process是DeepSeek-R1's Response，包在`<think>…</think>`里，读起来完全是另一种东西：先说「Okay, let me try to figure out this problem… Hmm, let's break this down step by step. First, let me understand the details. […]」，然后开始自我怀疑「Wait, if he writes to 2 friends twice a week, does that mean he writes 2 letters each time, and he does this twice? […] So per week, he writes 2 friends * 3 pages * 2 times. Let me check that. […]」「Alternatively, maybe it's 3-page letter to 2 different friends each time he writes… so that's 2 * 3 = 6 pages per writing session. Then, twice a week would be 6 * 2 = 12 pages per week.」「Wait, let me confirm. So, per week: […] That makes sense. But maybe I need to double-check if the problem is interpreted differently. […] Wait, but wait a second. Is there any chance that the problem might be considering a year as 52 weeks, but some people might account for leap years or something else? But generally, in problems like this, unless specified, we take 52 weeks per year.」最后`</think>`再给出Answer: James writes 624 pages in a year。

讲师的评论：上面那种是GPT-4o按提示模板给出的干净结构化思维链；下面绿色那块是R1，很不一样——你会读到它的思考过程，它自我怀疑、考虑替代解释、甚至纠结闰年，可能有点乱，但这正是重点：这是一个通过强化学习在可验证结果上学会推理的模型，而不是一个跟着人类写的例子走的模型；所以它自己的思考过程未必和人类会做的思考过程相同。

## 用强化学习学推理：GRPO [40:13 - 42:53]

![GRPO强化学习训练流程图，展示采样、奖励计算、优势估计与策略更新全过程](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0027_00-40-50.jpg)

怎么训练一个模型去推理？答案是强化学习，如今GRPO（Group Relative Policy Optimization）非常流行，而且和几周前讲过的PPO关系很近。这页标题「Learning to Reason with RL - GRPO」，两条要点：Sample answers, compare with each other, update toward good ones；No reward model, no value function, no human labels。页脚是DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models，Shao et al, 2024。

流程从左到右六段：Prompt q → Policy $\pi_\theta$ → G rollouts (chain-of-thought + answer) → Reward $r_i$ (binary) → Advantage $\hat{A}_i$ → GRPO objective $\mathcal{L}_{\text{GRPO}}$ → Updated $\pi_{\theta'}$。

例子问题是「A train travels 60 miles in 90 minutes. What is its average speed?」，五个采样结果与二元奖励（奖励标注为correctness, format）：

- $60 \div (90/60) = 60 \div 1.5$ → 40 mph ✓，$r_1 = 1$
- $90\,\text{min} = 1.5\,\text{h}$，$v = 60/1.5$ → 40 mph ✓，$r_2 = 1$
- $\text{speed} = 60/1.5 = 40$ → 40 mph ✓，$r_3 = 1$
- $60 \div 90 \approx 0.67 \times 60$ → 45 mph ✗，$r_4 = 0$
- $\text{speed} = 60/90 = 0.67$ → 0.67 mph ✗，$r_5 = 0$

优势按组归一化：

$$\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r}$$

图下方黄框写着No critic model，注释rewards relative to other samples in the group。目标函数：

$$\mathcal{L} = \frac{1}{G}\sum_{i=1}^{G}\min\big(\rho_i \hat{A}_i,\ \text{clip}(\rho_i, 1\pm\epsilon)\hat{A}_i\big) - \beta\,\text{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$$

右侧符号表（图上把$\epsilon$印成看起来像c的字体，视觉档案OCR也照抄成「1±c」「c = 0.2」，看图应为$\epsilon$）：$\rho_i = \pi_\theta/\pi_{\text{old}}$是importance ratio，$\text{clip}(\rho_i, 1\pm\epsilon)$是trust region（$\epsilon = 0.2$），$\beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$是drift penalty from $\pi_{\text{ref}}$，$\hat{A}_i$是group-relative advantage。最右侧Updated model用绿色↑reinforce、红色↓suppress表示对采样结果的推拉。

讲师强调把PPO用到语言模型上的一个关键难点：critic（价值函数）既昂贵又难训，因为你需要一个独立的网络去估计每一种可能token序列的期望回报，非常麻烦。GRPO的简化是去掉学习到的价值函数：不再由critic估计期望回报，而是对同一个问题采样一组rollout，然后相对于这一组来计算优势——每个优势就是它的奖励减去组均值再除以组标准差。于是模型通过把自己的输出互相比较来学习。他提醒看目标函数时会认出很多PPO里的项，但真正关键的是GRPO里缺了什么：没有人类标注、没有奖励模型、没有价值函数。只用可验证的二元答案（比如数学），推理之所以会涌现，是因为要持续拿到更高奖励、持续答对，唯一的办法就是学会仔细地思考这个问题。

## 为什么以前不行：base model得先能采样出正确轨迹 [42:53 - 44:33]

![解释此前方法失效原因的幻灯片：基础LLM需足够强才能产生正确rollout用于强化学习](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0028_00-43-37.jpg)

「Why didn't this work before?」这页只有一句加粗答案：Base LLM needs to be good enough to produce any correct rollouts to reinforce!

讲师的铺陈：人们用强化学习训大语言模型已经好几年了，如今社区大体接受的答案是：你需要一个足够强的基座模型，能够对难题偶尔产出正确答案。如果基座模型太弱、永远采样不出一个正确的rollout，那真的就没有任何东西可以被强化。现代LLM的变化在于，大概过去一年半左右，它们跨过了那个能力门槛——能够偶尔独自解出难题。一旦这件事发生，RL基本上就能接管，系统地把这个能力放大。

## 训练中自己长出来的反思行为：「wait」的频率曲线 [44:33 - 46:24]

![DeepSeek-R1论文图9：训练过程中反思词频率与'wait'一词出现模式随步数演化的双折线图](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0029_00-44-33.jpg)

这页幻灯片自己的标题是「Reasoning Behaviors Emerge from RL」（视觉档案把它按图注写成「DeepSeek-R1论文图9」，页面上方那行大字其实是讲者加的标题，图注在下方），页脚DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning，DeepSeek-AI, 2026。图注原文：Figure 9 | Evolution of reasoning behaviors during training. (a) Frequency of representative reflective words during the training process; (b) Specific occurrence patterns of the word "wait" throughout the training process.

两张折线图横轴都是Steps，0到10000出头。左图纵轴The Frequency of Reflection Words，刻度到14000：前约4000步在2000上下震荡（中间约1800处有一个约4800的尖峰），4000步后开始抬到4000–6000，6000步后进入6000–10000的剧烈波动区间，约9500步处冲到约14300的峰值。右图纵轴The Frequency of Word "Wait"，刻度到1400：约4000步之前基本贴着0，4000到7000步之间只有100到160量级的小起伏，约7500步开始陡升，约8000步到890左右、约8700步达到约1480的最高点，随后回落到300多，10000步附近再回到约1000。

讲师的讲法：没有人告诉这个模型要说「wait」这个词，也没有人标注哪些是好的推理轨迹、哪些是坏的；它纯粹从二元正确性奖励里发现了这个推理行为。他说在训练的前段（他给的是6000步之前）模型从不说wait，只是在试着把答案答对；某个点它发现「停下来重新考虑」能提升奖励，于是反思性语言（比如「let's think about this again」）的频率陡然跳升。模型是自己找到这条路的，因为它有足够强的先验去探索推理策略，而不只是去拟合答案模式。反过来说，在一个弱基座模型上，RL永远过不了第0步——没有好的rollout可强化，就什么也发现不了。

需要留意一处口径差异：讲师说的「6000步之前从不说wait」与图(b)并不完全对齐——图里wait在4000步后已有百次量级的小值，真正的跃升发生在约7500–8000步之后；本笔记按画面描述拐点位置，同时保留讲师的口头说法。

他认为这是「强化学习能做到而监督学习根本做不到」最有说服力的演示之一：发现没有人明确指定过的策略，比如这里的推理。

## 迁移到视觉：Visual-RFT把验证器换成IoU [46:24 - 48:30]

![过渡提问幻灯片：这种RL推理能力是否同样适用于视觉领域](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0030_00-46-24.jpg)

过渡页只有一句：Does this RL reasoning apply too in vision?

![Visual-RFT视觉强化微调框架图，展示GRPO在视觉定位任务中的应用及IoU验证器结构](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0031_00-47-20.jpg)

答案是肯定的，过去一年有很多论文做类似的事；据讲师所知Visual-RFT（页脚Visual-RFT: Visual Reinforcement Fine-Tuning，Liu et al., 2025）是最早展示这一点的之一。它基本上就是把GRPO应用到视觉任务上，两条要点写在标题下：Applies GRPO to enhance visual grounding of VLMs；The verifier is IoU (geometric), not a math grader。

图里左列是两个人类提问：房车照片配「Question: the part of the vehicle that can be opened?」，一朵花配「Question: What species is the flower in the image?」；中间是Policy Model（带火苗图标）与若干候选输出（房车上不同位置的候选框，其中绿框是对的），下面接Reference Model（雪花图标，即冻结参考策略）、Policy Gradient Optimization、Verifiable Reward。推理文本原样可读：`<think>` The vehicle has a door that can be opened. The door is located on the right side of the vehicle, near the top.`</think>`；第二个例子的推理是「The image appears to be of a columbine flower. The flower has five light yellow petals and a yellow center, which are characteristic features of this species. The flower also has a purple or lavender color on the petals, which is typical for this plant. The presence of a small insect on the flower is also a common feature of columbine flowers.`</think> <answer>` columbine `</answer>`。

右侧奖励公式（图上印的是$P^{\text{cate}}$与$GT^{\text{cate}}$，视觉档案OCR成$P_{\text{cls}} = GT_{\text{cls}}$，看图以画面为准）：$R_{\text{IoU}}$在匹配时取$f(\text{IoU})$、否则为0；$R_{\text{cls}}$在$P^{\text{cate}} = GT^{\text{cate}}$时为1、否则为0。

讲师的解释：验证器用的是IoU，即预测框与真值框的交并比，而不是数学评分器；模型仍然在`<think>`里推理（「门在右边」「靠近顶部」），然后预测一个边界框；通过这套方法它学到「更好的推理带来更好的定位」，而这会被强化。这里的关键洞见是GRPO框架对「验证器是什么」基本完全agnostic：只要你有可验证的奖励，就能把RL推理套上去。类似IoU，计算机视觉社区还有很多其他几何度量正在被用来通过RL改善VLM的接地能力。

## 失败模式：只奖励最终答案，推理轨迹对验证器不可见 [48:30 - 51:19]

![标准GRPO的失败模式说明：仅奖励最终答案导致推理过程不可见，且视觉任务中存在推理幻觉与正确答案并存的问题](/coursenotes/eth-robot-learning-2026/l10-embodied-reasoning/kf_0032_00-49-11.jpg)

但把GRPO用到多模态（尤其视觉）时有一个必须注意的重要失败模式。这页两句话：Failure mode: Standard GRPO rewards the final answer only, the reasoning trace is invisible to the verifier；In vision, hallucinated reasoning & correct answers aren't mutually exclusive。

讲师展开：在vanilla设定下GRPO只奖励最终答案，推理轨迹对验证器完全不可见。在数学上这大体没问题，因为很难通过错误的推理在数学上得到正确答案；但在视觉任务里，幻觉出来的推理和正确答案并不是互斥的——模型可以在`<think>`块里描述一个完全幻觉出来的视觉场景，却仍然预测出正确的边界框，靠运气或者直接从图像做模式匹配。

怎么修这个问题？这是他们在微软做的一项叫Argos的工作，思路是直接奖励不只是最终答案，也奖励中间的推理步骤。具体做法相当简单：自适应地、逐样本地选择不同的验证器（不同的基础模型），在RL训练过程中同时评估三件事——最终答案的正确性、相关对象的时空（spatiotemporal）定位（既可以在单张图像里，也可以跨视频）、以及推理本身的质量。也就是说，你可以检查这些边界框是否在整个视频里仍被正确定位，从而确保模型生成的推理实际上是接地的、是正确的。这样一来，模型就不可能再靠幻觉「混」出正确答案，因为中间推理在每一步都被奖励、被检查，包括视频里的时间接地。

## 回到机器人：高层规划吃RL推理，低层控制内化推理 [51:19 - 53:22]

从语言和视觉回到我们最终关心的具身智能与机器人。讲师报告他们在同一批工作里观察到的现象：测试时推理在高层规划任务上带来很大收益，比如ALFRED这类仿真任务；在最难的那批任务上，这套方法能给出比监督式思维链模型好3到4倍的性能。这意味着一个通过强化学习学会推理的模型，在难题上的泛化方式，和一个仅仅学会模仿推理轨迹的模型不一样。

但高层规划（比如导航）之外还有连续控制这个低层问题：对连续动作预测，你仍然负担不起在每个时间步都生成完整推理链。他们在这里的做法是沿用ECoT-Lite的配方——在VLA训练过程中把推理内化（internalize），然后在推理时丢掉。

由此浮现的机器人学习图景是一个两级架构：一层是RL训练出来的推理，负责高层规划与任务分解；另一层是ECoT-Lite式的推理内化，负责快速的反应式控制。

## 四条带走的思想 [53:22 - 55:40]

讲师结尾留下四个想法：

1. 测试时计算是一条新的、可以扩展的轴；在许多设定下，沿这条轴扩展比扩展预训练更高效。
2. 对连续控制，你不必在推理时付推理的代价：可以带着推理训练、把它内化、在部署时丢掉。真正重要的是表征，而不一定是测试时那些token。
3. RL能让推理涌现——不是因为你告诉模型要这么做，而是在很多情形下推理恰好是最大化奖励的最佳策略。这是一种比「监督式地模仿人类思维」强大得多的机制。
4. 对多模态agent来说，验证推理轨迹和验证答案或结果同等重要；如果你只奖励结果，就可能得到幻觉。

他说这四个想法合在一起，定义了当下AI与机器人学习里最令人兴奋的研究前沿之一；而如何把它们全部收进同一个能推理、能适应、能改进的机器人策略，仍然是一个未解的开放问题。最后以「thank you for your attention」结束全讲。

## 要点回顾

- 本讲片头自报「Lecture 11: Embodied Reasoning and Test-time Scaling」，日期04.05.2026，讲师Oier Mees，机构ETH zürich与Microsoft；视频标题的Lecture 10与页码差一号，属播放列表少列一讲，笔记未改标题。
- 数据给的是行为先验（把演示分布压进权重），缺的是智能推理；推理是组合式的，用可复用的逻辑与知识积木重新组合，就能从第一性原理处理没见过的情形。类比LLM：吞下整个互联网仍不够，机器人数据更贵，只堆数据更不够。智能推理被拆成泛化、可解释、交互、可控四个维度。
- ECoT（Zawalski\*、Chen\*等，CoRL 2024）：不让VLA从图像+指令直接映射到动作，而是先生成TASK/PLAN/SUBTASK(+REASONING)/MOVE(+REASONING)/GRIPPER POS/VISIBLE OBJECTS再接ACTION $[\Delta x, \Delta\theta, \Delta\text{Grip}]$，动作token条件在这些接地推理token上，等于强迫模型「think carefully」「look carefully」。
- 推理数据靠蒸馏基础模型：Gemini、OWLv2 + SAM、Prismatic VLM、Grounding DINO给已有机器人轨迹批量生成语义接地的推理（边界框、夹爪位置、未来移动轨迹），因为人工标注每条轨迹不可扩展。
- 结果：同一批机器人数据、不加采集，泛化成功率提升约30%；读图ECoT约65%、RT-2-X约48%、OpenVLA约37%、Octo约18%；7B打赢55B的RT-2-X。附带收益是可解释与可交互——测试时人工改写推理链，动作会自我纠正。
- 推理本身跨具身迁移：只在WidowX数据集上训练，就能给Open X-Embodiment里完全陌生的机器人与相机产出合理推理链；原因是这些推理任务与VLM预训练任务高度相似，ECoT是在「解锁」VLM已有的推理能力。Gemini Robotics的VLA已把具身推理作为核心组件，说明领域在收敛到「推理是策略的一等公民」。
- 后续工作：MolmoAct（Lee et al., 2025，深度token做空间推理）、CoT-VLA（Zhao et al., 2025，视觉思维链）、RAD/Action-Free Reasoning for Policy Generalization（Clark et al., 2025，把推理训练与动作预测解耦、从人类数据迁移无动作推理）、Self-Supervised Bootstrapping of Action-Predictive Embodied Reasoning（Ganai et al., 2026）。
- 代价是慢：OpenVLA约4 actions/s，ECoT约4 s/action；幻灯片给出的原版频率区间是0.3–1.2 Hz。
- 三个「为什么有效」的假设：更好的表征学习（重要的是训练时的监督信号，测试时可以不生成）、更好的学习课程（预测连续动作对VLM极OOD，推理当in-context脚手架、类似teacher-student）、更强的策略表达能力（多token本身就是额外计算预算，LLM里已有记录）。
- LIBERO-90隔离实验（ECoT-Lite，Chen、Belkhale、Mirchandani、Mees、Driess、Pertsch、Levine，CoRL 2025）：Embodied CoT约91 > Reasoning Dropout约89.5 > Reasoning Pre-training约87 > Reasoning "Scaffolding"约84 > Standard VLA约82 > Thinking Tokens约79.5；无语义的thinking token反而低于标准VLA，说明收益不是「白送的计算预算」。
- 两种内化配方：Reasoning Pre-training是两阶段（先只训推理、再只训动作，转移表征），适合推理与动作数据不配对（从仿真或人类数据迁移）；Reasoning Dropout是单阶段联合训练但按概率丢弃推理，测试时可开关，代价是更多训练内存。原版最慢，预训练版多训练步数。
- 真机Bridge WidowX：Embodied CoT成功率约78但频率最低；Reasoning Pre-training约68、Reasoning Dropout约62，频率都约3.5 Hz，与标准VLA（约51、约3.6 Hz）同档——「比标准VLA强、又不牺牲速度」；ECoT-Lite不是最高性能而是「第二高且快得多」。
- 测试时计算扩展≠传统规划，但谱系连续：经典规划（A\*/STRIPS/PDDL，手工世界模型、显式树搜索、精确目标测试、已知域）→学习式规划（国际象棋/围棋/Atari，策略+价值网络、MCTS、胜负判定、离散博弈）→LLM推理（CoT/GRPO/R1，权重里隐式世界模型、token生成、规则式奖励、开放任务）；每步都是用显式结构换通用性。
- AlphaGo：Raw Network约3050 Elo，加MCTS的Full AlphaGo Zero约5150，超人类虚线约3500；经验法则「涨120 Elo要么模型翻倍要么测试时搜索翻倍」，据此补齐这段差距若只靠做大模型需约十万倍（讲师自注此数被夸大，因为AlphaGo是对自己早期checkpoint练的）；至今没有任何围棋模型不靠测试时搜索就超过人类。
- 语言（Snell et al, 2024）：等FLOPs下compute-optimal测试时扩展在易/中题打赢约14倍大的模型（<<1组+21.6/+27.8/+11.8，~=1组+16.7/+3.5/−11.9，>>1组+5.4/−24.3/−37.2），难题上反转；机制是自失败尝试上做SFT+学习到的验证器、测试时生成多个答案由验证器挑最优；本质是在模型自身输出分布上做由奖励引导的搜索，所以它只放大模型已知的东西。
- 理论（Li et al, 2024）：任何$T$步可解的问题都能由常数规模Transformer生成$O(T)$个推理token解决；一次前向的深度受层数限制，自回归生成中间token等于在测试时展开额外计算。
- CoT提示（Wei, Wang et al, NeurIPS 2023）极简：给几个把推理步骤写全的例子；标准提示答27（错）、CoT提示分步算出9（对）。但它需要强基座——LaMDA/GPT/PaLM三族曲线上，小模型上CoT甚至有害，规模上去后涌现并超过此前最佳监督方法。原始形式=少样本上下文学习，因此提示敏感、脆弱、需人工挑例子、难泛化到所给例子/领域之外。
- 从「提示出推理」到「训练出推理」：GPT-4o的单链思维是照模板写出的干净分步解（624 pages/year）；DeepSeek-R1的`<think>`会自我怀疑、考虑替代解释、甚至纠结闰年，过程凌乱但它是在可验证结果上用RL学会推理的，思考方式未必同于人。
- GRPO（Shao et al, 2024）：没有奖励模型、价值函数、人类标注；对同一问题采样G个rollout，优势$\hat{A}_i=(r_i-\mu_r)/\sigma_r$按组归一化以取代critic，目标$\mathcal{L}=\frac{1}{G}\sum_i\min(\rho_i\hat{A}_i,\text{clip}(\rho_i,1\pm\epsilon)\hat{A}_i)-\beta\,\text{KL}(\pi_\theta\|\pi_{\text{ref}})$，$\rho_i=\pi_\theta/\pi_{\text{old}}$、$\epsilon=0.2$；奖励是correctness与format的二元信号。只有可验证答案时，持续拿高奖励的唯一途径就是学会仔细思考。
- 为什么以前不行：基座模型必须强到能偶尔采样出正确rollout，否则「没有东西可强化」，RL过不了第0步；大约过去一年半LLM才跨过这个门槛，之后RL系统性地放大该能力。
- DeepSeek-R1的Figure 9：反思词频率在约4000步后抬升、约9500步冲到峰值（左轴到14000），"wait"的频率在约8000步后从近乎0跳到1400量级（右轴）；没有人教它说wait，也没有人标注好坏推理轨迹——这是RL能发现无人指定之策略、而监督学习做不到的有力演示。
- Visual-RFT（Liu et al., 2025）把GRPO搬到视觉接地，验证器是几何的IoU（匹配时$f(\text{IoU})$否则0）加类别奖励（$P^{\text{cate}}=GT^{\text{cate}}$时为1）；模型在`<think>`里推理「门在右侧靠上」再出框，学到「更好的推理→更好的定位→被强化」。关键洞见：GRPO对验证器是什么完全agnostic，只要有可验证奖励。
- 失败模式与Argos（微软）：标准GRPO只奖励最终答案，推理轨迹对验证器不可见；数学上「错推理难得对答案」所以问题不大，但视觉里幻觉推理与正确答案不互斥。Argos逐样本自适应选择不同基础模型作为验证器，同时评估最终答案正确性、相关对象的时空定位（单图或跨视频）与推理质量，使模型无法再靠幻觉混出正确答案。
- 回到机器人：测试时推理在ALFRED这类高层规划仿真任务上收益很大，最难任务上比监督式CoT模型好3–4倍；连续控制仍付不起每步生成完整推理链的代价，故用ECoT-Lite把推理内化后在部署时丢掉。浮现的两级架构：RL训练的推理管高层规划与任务分解，推理内化管快速反应式控制。
- 四条带走：测试时计算是新的高效扩展轴；连续控制可以「带推理训练、部署时丢掉」，重要的是表征；RL能让推理作为最大化奖励的最优策略自发涌现，强于监督模仿人类思维；多模态agent里验证推理轨迹与验证结果同等重要，只奖励结果就会招来幻觉。如何把四者收进同一个能推理、能适应、能改进的机器人策略仍是开放问题。
