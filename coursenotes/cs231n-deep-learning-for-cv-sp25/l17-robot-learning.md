---
title: "第 17 讲 · 机器人学习：从感知-动作循环到机器人基础模型"
description: "Stanford CS231n Spring 2025 L17 中文视频笔记（2026-11 重写版）"
---

# CS231n Spring 2025 · Lecture 17 机器人学习：感知、决策与基础模型

**核心结论**：本讲由客座讲师 Yunzhu Li（Columbia University 计算机科学助理教授，领导 Robotic Perception, Interaction and Learning Lab，2023 年曾在 Stanford 与 Fei-Fei Li、Jiajun Wu 合作做博士后并讲授 CS231n）主讲机器人学习。核心论点是：机器人学习在性质上不同于典型计算机视觉——CV 主要学习从高维输入到表示或输出的映射，而机器人要在「物理世界作为约束」下求解一个序贯优化问题，用一串动作改变环境、从环境获得新观测与奖励并最大化目标，因此 goal、state、action、reward 构成的状态-动作-奖励闭环是全讲的形式化语言。课程按 overview 依次展开：机器人感知（具身、主动、情境化，多传感器与感知-动作协同设计）、强化学习与监督学习的四点差异（随机性、信用分配、不可微、非平稳）、Q-learning 与 DQN 从 Atari 到 locomotion／灵巧操作的成果与 sim-to-real，再转入无模型 RL 的样本效率与安全瓶颈、学习世界模型并用它规划（像素／关键点／粒子三种状态表示，RoboCook 做饺子），随后是模仿学习（行为克隆的 cascading error、IRL、隐式能量策略、Diffusion Policy）与机器人基础模型（RT-1 到 π0 的预训练、co-fine-tuning、post-training），最后讨论评估、仿真与基础世界模型等剩余挑战。一条反复出现的经验是：从游戏到真实机器人，domain randomization 与高质量数据往往比更精确的仿真或更复杂的算法更关键；而真实世界评估昂贵、noisy，且训练 loss 与长程成功率相关性很弱，仍是领域最大的瓶颈之一。

## 1. 从 CV 到机器人学习：问题形式化 [01:34 - 12:30]

讲者先对齐坐标系。此前课程学过的**监督学习**是数据 $(x, y)$、$x$ 为输入、$y$ 为标签，目标是学一个从 $x$ 到 $y$ 的映射，例子包括分类、回归、目标检测、语义分割、图像描述；**自监督学习**则没有标签，靠设计辅助损失从海量无标注数据中抽取隐含结构（如 autoencoder）。

![监督学习定义幻灯片，包含数据形式、目标及分类任务示例](/coursenotes/cs231n-deep-learning-for-cv-sp25/l17-robot-learning/kf_0001_00-01-39.jpg)

机器人学习的独特之处在于必须产生**物理交互**：不是输入输出的静态映射，而是「影响环境的演化」——无论采取什么动作，世界都会随之改变，并返回新的观测与奖励，告诉智能体环境如何变化、任务完成得如何；目标是在环境反馈下找出一串动作，最大化奖励或最小化代价。讲者提到这一方向近年同时受到学界与产业界高度重视，Physical Intelligence、Tesla Bot、Figure 等公司拿到大量投资，展示叠衣服、操作咖啡豆、人形机器人完成复杂任务等视频。

![包含人类目标输入的问题形式化（Problem Formulation）流程图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l17-robot-learning/kf_0002_00-06-33.jpg)

上图引入的 Human 角色通过设定目标 $g$（可以是人的语言指令，也可以是衡量任务完成度的目标函数）引导 Agent：Agent 从物理世界取得状态 $s_t$、决定动作 $a_t$ 并在物理世界执行，世界被更新后返回 $s_{t+1}$ 与奖励 $r_{t+1}$。讲者强调要牢记 goal、states、actions、rewards 四要素及其目标函数：CV 主要是从高维输入学环境的**表示**，而机器人本质是在物理世界约束下**求解优化问题**，用动作序列最大化／最小化目标函数。课程的整体路线也由这张概览给出：

![课程概览幻灯片，高亮显示当前章节为问题表述](/coursenotes/cs231n-deep-learning-for-cv-sp25/l17-robot-learning/kf_0003_00-11-39.jpg)

一系列具体实例把这套抽象符号落地：**CartPole**（目标是把杆平衡在小车上；state 含角度、角速度、位置、水平速度；action 是施加于小车的水平力；每步杆保持直立给 reward 1）、**机器人 locomotion**（前进；state 为各关节角度／位置／速度；action 为各关节 torque；reward 为前进一步且保持直立）、**Atari 游戏**（得分最高；state 是屏幕原始像素；action 是上下左右；reward 是每步分数增减）、**围棋**（赢棋；state 是棋盘上已有棋子；action 是下一手落子位置；reward 只在最后一回合，赢 1 输 0）。讲者还把同一框架推广到 **LLM 序列生成**（goal 预测下一个词，state 为当前句中已有的词，action 为具体下一个词，正确才给 reward）、**chatbot**（goal 做用户的良好伴侣，state 为当前对话，action 为回复的下一句，按人类满意度给 reward），以及本讲最关心的**叠衣服**（state 是多视角 RGB／RGBD 观测，action 是末端如何移动、夹爪开合，按衣物是否叠好给 1／0）。

关于 reward，讲者提醒设计中有大量细微差别：自动驾驶可以要「尽可能快」，也可以要「乘客舒适」；叠衣服可以要折叠后面积最小，也可以要最平整，不同偏好对应不同 reward 规格，需要针对具体应用细化。

## 2. 机器人感知：具身、主动、情境化 [12:30 - 18:31]

![机器人感知中人类、智能体与物理世界交互的闭环结构图，红框强调状态s_t](/coursenotes/cs231n-deep-learning-for-cv-sp25/l17-robot-learning/kf_0004_00-12-57.jpg)

这张图在整讲里反复出现。物理世界给机器人的是高维 RGB／RGBD 观测，也可能有触觉、音频等传感数据；机器人感知要做的是从这些高维数据中**蒸馏出对下游决策有用的结构化知识**，也就是「理解这个非结构化的真实世界」。真实世界非常 messy：观测只包含对物体与环境不完整的信息，存在遮挡、传感器误差、动作不完美（抓取可能失败、物体意外掉落从而引发环境的意外变化）；环境还是动态的，不只有刚体，还有衣服、面团这类可变形物体，以及狗、小孩、其他人等同样在改动环境的 agent。

因此机器人领域通常不满足于相机，而是尽可能加装能提供有用信息的传感器：触觉（判断抓取是否稳定）、音频（物理接触）、深度、相机（整体状态的高层信息），并设计让它们互补的系统。与 CV 更本质的差别，是要理解**动作的效果**和环境的 **affordance**：

![机器视觉与计算机视觉的对比：具身性、主动性与环境情境化](/coursenotes/cs231n-deep-learning-for-cv-sp25/l17-robot-learning/kf_0005_00-15-42.jpg)

左边是典型 CV 的实例分割——在一张 2D 图像上给不同实例画轮廓；右边机器人面对一堆看似一个、实则可能多件叠在一起的物体时，必须主动去 perturb、与环境交互，才能判断到底是单个物体还是多个部件组合。讲者用三个词概括 **"Robot vision is embodied, active, and environmentally situated"**（画面标注 Detectron、Zeng et al., IROS 2018）：

- **embodied（具身）**：机器人有物理身体，直接体验物理世界；它的动作是与世界动态的一部分，对自身感知有即时反馈。
- **active（主动）**：机器人是主动感知者，知道自己为何要感知、选择感知什么，并决定如何、何时、何地去感知——像转头去看桌子后面，而 CV 大多处理被动采集的数据集。
- **situated（情境化）**：机器人身处世界之中，面对的不是抽象描述而是此时此地，直接影响系统行为。

由此得到**感知-动作闭环**：机器人有时并不需要环境的完整状态，例如扣衬衫扣子只需知道扣子附近的局部区域；因此感知必须与任务和下游决策紧耦合、协同设计，聚焦任务相关区域，而不必追求对全场景的完备重建。

## 3. 强化学习与监督学习的四点差异 [18:31 - 23:43]

强化学习的一般解法是让机器人尽可能大量地与世界交互、收集经验做试错，从而知道哪些动作带来更高奖励、哪些更低，并把行为逐步偏向高奖励动作。

![强化学习与监督学习对比图，描绘了智能体与环境在时间步 t 和 t+1 的状态、动作及奖励交互](/coursenotes/cs231n-deep-learning-for-cv-sp25/l17-robot-learning/kf_0006_00-19-38.jpg)

框架里 Environment 给 Agent 状态 $s_t$，Agent 产生动作 $a_t$，环境反馈奖励 $r_t$ 并更新到 $s_{t+1}$、$a_{t+1}$、$r_{t+1}$，构成时间域上的序贯决策（MDP 的标准符号）。对照之下，监督学习是数据集把 $x$ 送入模型得到预测 $y$，再与 ground truth 算 loss。讲者逐条列出四点关键差异：

1. **随机性／不确定性**：环境可能是 stochastic，同样的动作因支撑力分布不同会让箱子转到不同角度，同一动作不一定得到相同奖励，面对的是不确定动力系统。
2. **信用分配（credit assignment）**：监督学习每做一次预测就直接知道错在哪；RL 的奖励可能**延迟**——围棋要到终局才知道输赢，那个 0/1 的 reward 可能源自很早甚至第一步的落子，如何把功劳／责任分配到序列中各动作是难题。
3. **不可微（non-differentiability）**：监督学习从输入到 loss 全链路可微，可直接对参数求梯度；RL 的环境往往不可微，只能依靠大规模采样做零阶（zeroth-order）梯度估计。
4. **非平稳性（non-stationary）**：监督学习里无论预测什么，都不影响数据集中的其他样本；RL 里**智能体经历到什么取决于它自己怎么行动**，环境的演化正是自身动作的结果。

![强化学习与监督学习对比图，展示Agent与环境交互的State-Action-Reward循环及非平稳性特征](/coursenotes/cs231n-deep-learning-for-cv-sp25/l17-robot-learning/kf_0007_00-23-34.jpg)

## 4. Q-learning、DQN 与从游戏到真实机器人 [23:43 - 34:42]

以 Atari 为例，典型做法是学一个 **Q function**：它度量在特定状态 $s$ 采取特定动作 $a$ 之后的**折扣期望未来累积奖励**。学得 $Q(s,a;\theta)$ 后，比较各动作的 Q 值并执行最大者 $a^*=\arg\max_a Q(s,a)$ 即可决策；常见算法还有 Q-learning、policy iteration，当前的 SOTA 包括 SAC（soft actor-critic）与 PPO（proximal policy optimization），本课不深入。

![Atari游戏案例中DQN网络结构图，展示卷积层与全连接层配置](/coursenotes/cs231n-deep-learning-for-cv-sp25/l17-robot-learning/kf_0008_00-28-12.jpg)

把 Q 函数实例化为神经网络就是 DQN：输入是最近 4 帧堆叠的 $4\times84\times84$ 状态，经 `Conv(4->16, 8x8, stride 4)` → `Conv(16->32, 4x4, stride 2)` → `FC-256` → `FC-A`，最后一层为所有离散动作各输出一个 Q 值。Google DeepMind 玩 Breakout 的结果很能说明 RL 的特点：训练 10 分钟能碰到球但常漏，2 小时已能可靠接球，4 小时后甚至自主发现一种新策略——把球打到墙左侧凿出隧道，再送到墙上方高效清砖。

讲者以 AlphaGo（2016 年 1 月）为个人研究转向的契机，串起一条游戏 AI 脉络：**AlphaGo Zero** 是简化版，不再用模仿学习做初始化，却击败了当时的世界第一棋手，由此引出 Rich Sutton 的 **"bitter lesson"**——应寻找与 scaling 最兼容的最简单配方，方法越简单越能借算力放大；**AlphaZero** 把同一套算法泛化到 chess、shogi；**MuZero** 进一步学习隐空间动力学模型并在其上规划；2019 年 11 月，曾被 AlphaGo 击败的棋手宣布退役，认为人类棋手已无法战胜最强的围棋 AI。更复杂的游戏同样被攻克：

![更复杂游戏案例列表：StarCraft II的AlphaStar与Dota 2的OpenAI Five](/coursenotes/cs231n-deep-learning-for-cv-sp25/l17-robot-learning/kf_0009_00-31-25.jpg)

StarCraft II 的 **AlphaStar**（2019 年 10 月，Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement learning", Science 2018）与 Dota 2 的 **OpenAI Five**（2019 年 4 月，只有博客、无正式论文）说明：只要算力充足、算法与基础设施到位，比围棋复杂数个数量级的游戏也能训练出很强的 agent。

真实机器人方面，ETH 发表于 Science Robotics 2020 的工作改变了讲者的看法：仿真里并不模拟灌木、积雪，RL 训练出的策略却能在雪地与湿滑表面稳健行走，说明 **sim-to-real gap 有时没那么重要**；Unitree 近期的视频更展示了 sim-to-real 带来的高动态 locomotion，可穿越崎岖地形。讲者判断 **locomotion 已接近被解决，方案正是强化学习**。做法是刚体仿真 + **domain randomization**：大量随机化摩擦、几何等物理参数，把真实世界看作该随机分布里的一个数据点，只要策略在整个分布内鲁棒就能泛化；高层命令（前进／原地转／继续走）由人给出，策略负责决定各关节 torque 这类低层动作。他总结的经验是「仿真不必完美，只要随机化得足够就能稳健泛化」，但这一经验尚未很好迁移到 manipulation。

![机器人灵巧操作（Dexterous Manipulation）相关视频资源列表，包含OpenAI解魔方与Science Robotics手内重定向研究](/coursenotes/cs231n-deep-learning-for-cv-sp25/l17-robot-learning/kf_0010_00-34-02.jpg)

灵巧操作上，OpenAI 2019 年用机器手解魔方（仿真训练 + sim-to-real）虽视频漂亮，实际成功率很低、测试次数有限；后续 Science Robotics 2023 的 "Visual Dexterity: In-Hand Reorientation of Novel and Complex Object Shapes" 把不同形状物体的手内重定向做得更强。但讲者指出，无论 locomotion 还是手内操作，都还没解决「在家帮你叠衣服、洗衣服」——manipulation 仍停留在孤立的环境里。

## 5. 无模型 RL 的瓶颈与基于模型的规划 [34:42 - 54:46]

### 5.1 无模型 RL 的局限与人类的直觉模型 [34:42 - 41:34]

![无模型强化学习（Model-Free RL）的局限性总结，包括试错成本高、安全性及可解释性问题](/coursenotes/cs231n-deep-learning-for-cv-sp25/l17-robot-learning/kf_0011_00-36-30.jpg)

(Model-Free) RL 靠试错学习、需要海量交互、有安全隐患、可解释性有限，「出问题时怎么办」很难回答。AlphaGo Zero 用 40 天学完人类 3000 年的知识固然惊人，但仍消耗许多年等价计算；若 sim-to-real gap 很大、只能在真实环境里学，样本效率就是巨大瓶颈；把学习过程中那些怪异行为放到真实人形机器人上会灾难性失败。相比之下，**人类对世界有一个直觉模型**，能想象施加某个动作后环境会怎么变化，正是这种预测能力让我们能规划行为达成目标，而它同样来自日常与物理世界的交互经验。于是问题变成：能否让机器人也学会「想象动作的后果」并据此规划。

### 5.2 学习状态转移函数与基于模型的规划 [41:34 - 46:00]

![模型学习与基于模型的规划（Model Learning & Model-Based Planning）概念图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l17-robot-learning/kf_0012_00-43-12.jpg)

框架是学习世界的状态转移函数 $P(s_{t+1}\mid s_t, a_t)$，再用「通过模型做规划」来决策。这是一个 **forward model**：给定当前状态 $s_t$ 与动作 $a_t$ 预测下一状态；而 planning 是它的**逆问题**——给定当前状态与目标状态，求一串能把状态带到目标的动作。做法是先给动作序列一个初猜，让学到的模型预测出状态演化轨迹（图中绿点），量测它与目标（红点）的距离，再把这个距离关于轨迹上所有动作的梯度反向传播做优化，即大致求解 $\min_{a_{0:T}}\sum_t \lVert s_t - s^{\text{target}}\rVert^2$。由于模型未必够准，实际采用 **MPC 式**闭环：只执行动作序列的第一个动作 → 从环境取得新状态 → 用梯度下降重新优化整段动作序列。近年得益于 GPU 与神经动力学模型，可以大规模并行采样并同时优化大量动作序列，效率很高。这里最核心的开放问题是：**$s_t$ 应该采用什么形式的状态表示**。

### 5.3 状态表示：从像素、关键点到粒子 [46:00 - 51:05]

历年出现过多种选择。早期把 2D 图像直接当作状态、学习 **pixel dynamics**（图像在动作下如何变化），代表作 **Deep Visual Foresight** 是世界模型方向的起步工作，通过最小化当前观测与目标的距离来旋转、推动物体达到目标构型。再细一层用 **key points** 表示环境，跟踪箱子上关键点在 3D 空间的移动并学其动力学，据此规划推动轨迹。对自由度更高的物体，则用一组 **particles（粒子）**表示：把一堆颗粒状物料编码成点云，预测施加某动作后各粒子会移到哪里。

![粒子动力学（Particle Dynamics）对比图，展示真实场景与粒子表示在时间轴上的变化](/coursenotes/cs231n-deep-learning-for-cv-sp25/l17-robot-learning/kf_0013_00-46-12.jpg)

这一方向的工作包括 Wang, Li, Driggs-Campbell, Fei-Fei, Wu 的 "Dynamic-Resolution Model Learning for Object Pile Manipulation", RSS 2023：模型能泛化到不同颗粒大小，用环境反馈纠正模型误差，把颗粒可靠聚拢到目标区域；目标构型还能换成 A 到 Z 的字母形状，需要非平凡的重新分布与细粒度对齐，属于难度很高的任务。

### 5.4 RoboCook 与学出来的动力学模型 [51:05 - 54:46]

![RoboCook论文介绍页，包含实验环境布局图及15种不同工具/物体的编号列表](/coursenotes/cs231n-deep-learning-for-cv-sp25/l17-robot-learning/kf_0014_00-48-26.jpg)

更进一步是 **RoboCook**（Haochen Shi*, Huazhe Xu*, Samuel Clarke, Yunzhu Li, Jiajun Wu，"RoboCook: Long-Horizon Elasto-Plastic Object Manipulation with Diverse Tools", CoRL 2023，获 **Best Systems Paper Award**）：一个做饺子的机器人，配有 15 种 3D 打印工具、4 个 RGBD 相机重建面团几何，需要决定用哪个工具、施加什么动作把面团做成饺子。关键使能因素仍是粒子表示的 forward model——红点表示工具形状、蓝点表示物体形状，第一行是模型的开环预测、第二行是真实发生的情况，两者吻合得很好。视频里人不断扰动（加料、折叠、甚至毁掉刚切好的圆），机器人靠实时视觉反馈与学到的动力学模型重规划，甚至判断出必须从头重做，展示了很强的鲁棒性。决策分两层：高层是一个**分类器**选工具（task level），低层策略决定具体动作（motion level）。

![模型学习与基于模型的规划流程及状态转移函数定义](/coursenotes/cs231n-deep-learning-for-cv-sp25/l17-robot-learning/kf_0015_00-53-21.jpg)

讲者借此更严谨地重申流程与开放问题：学 $P(s_{t+1}\mid s_t,a_t)$ → 通过模型规划 → 只执行第一个动作 → 取得新状态 → 用梯度下降重新优化动作序列；关键是 GPU 并行采样／梯度下降，而**关键问题始终是 $s_t$ 该用什么形式**。2023 年时 VLM 还不强，因此选工具的分类器是用人类操作员 10 次示范训练的。测试时实时采样太慢，于是改为**离线采样**生成数据集，再把它蒸馏成一个推理极快的策略网络；也可把这类工作称作 model-based RL，重点都是从与真实世界的物理交互中学模型再用于决策。他们刻意不用物理仿真：baseline 采用当时 SOTA 的可变形物体仿真器 **MPM（Material Point Method）**，即便做了很充分的系统辨识（估计物理参数），精度仍明显不如直接从真实交互学到的模型。

![粒子动力学未来预测：GNN序列模型对比预测与真实轨迹](/coursenotes/cs231n-deep-learning-for-cv-sp25/l17-robot-learning/kf_0016_00-54-35.jpg)

粒子动力学的未来预测用 GNN 逐步推演 $S_0 \to S_1 \to \dots \to S_5$，并对比 Prediction 与 Ground Truth（Gripping / Top / Side / Perspective 多视角），验证所学模型对抓取任务轨迹预测的有效性。

## 6. 模仿学习：从行为克隆到扩散策略 [54:46 - 63:06]

三种范式的关系可以这样 recap：RL 直接靠试错学策略（有样本效率与安全问题）；model learning 把环境演化数据拿来做**监督学习**训模型、再用模型规划；那么能否干脆用监督学习直接学策略？这就是 **imitation learning（learning from demonstration）**——用展示任务如何完成的大规模数据集训练策略，这并不新鲜，人类从小就是这样学习物理交互与社会活动的。

最经典的算法是 **Behavior Cloning**：学习从观测 $o$ 到动作 $a$ 的映射，策略记为 $\pi_\theta$。它最大的问题是 **cascading error（级联误差）**：序贯决策中一个很小的误差会把状态带离训练数据的分布，于是策略犯更大的错，误差沿时间步不断放大，最终轨迹严重偏离示范轨迹。因此实际做法是循环迭代：专家示范 → 监督学习训策略 → 在真实环境 roll out → 观察失败案例 → 补充数据或提供 **corrective behaviors**（把偏差拉回正轨的动作）→ 再训练，让数据集不仅含初始示范、也含纠错行为。

![强化学习(RL)与逆强化学习(IRL)的数据流向对比示意图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l17-robot-learning/kf_0017_00-58-54.jpg)

因为任务没有显式定义、而是隐含在示范里，还有一类 **Inverse Reinforcement Learning（IRL）**：RL 是 Environment → Rewards → RL → Behavior，而 IRL 反过来是 Environment → Behavior → IRL → **Rewards**，即从示范行为反推出奖励函数，再用它做常规 RL。早期的成功案例出自 Stanford 的 Peter Abbeel 与 Andrew Ng，用 IRL 控制真实直升机做出非常敏捷的特技动作。

近年模仿学习与生成模型结合得更紧。其一是受 **energy-based model** 启发的**隐式策略**：不做 $o\mapsto a$ 的显式回归，而是把 $(o,a)$ 映射到一个能量／score $E_\theta(o,a)$，推理时求 $\arg\min_a E_\theta(o,a)$；这样能处理**多峰（multimodal）**示范与优化景观不平滑的情形。其二就是近期机器人学习整体成功的重要来源 **Diffusion Policy**（"Visuomotor Policy Learning via Action Diffusion"）：把扩散模型当作策略的函数类，让策略继承生成模型的性质。

![扩散策略（Diffusion Policies）的三种形式对比：显式策略、隐式策略与扩散策略的结构及能量/梯度场示意图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l17-robot-learning/kf_0018_01-01-31.jpg)

画面把三类策略并列：显式策略 $F_\theta(o)$（直接回归动作）、隐式策略 $\arg\min_a E(a)$、$E_\theta(o,a)$（能量函数取最小）、扩散策略 $\nabla E(a)$、$\epsilon_\theta(o,a)$ 经 $K$ 次迭代去噪，并配有能量分布与梯度场可视化。讲者提到这项工作最初在 Columbia 完成（他目前所在），PI 后来到了 Stanford。Diffusion Policy 能完成的不只是平面推动，还包括抹黄油、炒蛋、削土豆皮、滑书等细粒度操作；效率极高——早上采集示范、中午训练、下午就能得到在真实世界工作的策略。当然，策略的可靠性、泛化性、能容忍的初始构型多样性仍有不少 caveat，需要迭代式数据收集来覆盖意外与偏离行为，但讲者认为模仿学习仍是「让策略在真实物理世界做出有意思的事」最高效的方式。

## 7. 机器人基础模型：从 RT-1 到 π0 [63:06 - 73:47]

机器人基础模型在函数类上和 RL／模仿学习的策略并没有本质区别：它不显式表示状态、也不学环境模型，仍是一个把 $(observation, goal)$ 映射到 actions 的策略。真正特别之处根植于语言与视觉语言基础模型的发展——它必须比只服务单一任务的策略**泛化得好得多**。讲者借 VLM 的类比给出自己的定义：输出未必每次都是最优动作，但要「总是生成合理的东西」，即合成轨迹应当 **beautiful**（平滑连续、不是乱抖）且 **reasonable**（听从语言指令）。这类东西有很多名字：Vision-Language-Action models（VLA）、Large Behavior Models，本质相同。这个领域也很 noisy：既然叫 foundation model 就要求广泛泛化，而要证明广泛泛化需要大量证据，因此进展很难量化评估，只能先看实证视频。

![机器人基础模型时间线，展示RT-1架构（含EfficientViT、TokenLearner和Transformer模块）](/coursenotes/cs231n-deep-learning-for-cv-sp25/l17-robot-learning/kf_0019_01-06-45.jpg)

时间线从 **RT-1**（2022 年 12 月）起步，其架构把 Instruction 与 Image 经 FLM、EfficientViT、TokenLearner 送入 Transformer 再输出 Action；此后大约每半年一个新模型——RT-2、RT-X、OpenVLA、π0 等，到 2025 年又涌现 Helix、Gemini Robotics 等，资本与人才投入都很密集。讲者以 **π0**（2024 年 10 月发布）为例，认为它第一次让人相信机器人基础模型能在真实世界可靠完成灵巧操作（叠衣服、叠盒子等）。

框架上，任何 foundation model 的「燃料」都是数据：聚合学术界数据与自己采集的、跨多种 embodiment 的真实任务数据做预训练。一个关键设计是**从已经预训练好的 VLM 出发**——它已在大量视觉语言数据上学到语义知识；再用 **co-fine-tuning**，同时优化 action prediction 目标与从 VQA 适配来的目标，既保留语义知识、又学会预测机器人动作。随后是受 LLM 社区启发的 **post-training**：base model 能给合理基线，但要在特定任务上表现非常好，必须采集任务专属数据做微调。他们按三类评测：① 简单的 in-distribution 任务（预训练见过的）直接用 base model；② 稍复杂的 in-distribution 任务用 post-training 提升；③ 未见任务必须收集专属数据、在预训练模型上做 post-training。π0 已开源、可下载 checkpoints，讲者实验室的学生已在上面做 post-training 并看到有希望的结果。

现场问答补充了两点。其一是**效率**：策略比人慢，主要因为示范数据由人用 teleoperation 在同一台机器人上采集，换了一个 embodiment 本就比直接用手慢，再加上机械臂有距离、存在遮挡、要频繁变换视角才能判断任务阶段，数据本身就慢，直接训练出的策略自然也慢；如何以人类速度采集数据是活跃方向。其二是**长程与规模化**：单个 π0 策略能把折盒子这种长程任务做好已很惊艳，但要在家庭大范围场景里叠衬衫、铺床、收拾地面杂物，讲者个人不相信一个巨型策略全能包办，认为需要更高层的抽象或符号表示来引导 VLA 模型，把任务导向不同场景、扩展到更大环境与更复杂任务。预训练 VLM 带来的语义知识让**语义层面**的泛化「免费」获得，再靠机器人数据微调补上**动作层面**的泛化。

## 8. 剩余挑战：评估、仿真与基础世界模型 [73:47 - 78:25]

社区公认的第一大挑战是 **evaluation**。评估目前主要在真实世界完成（例如 Google Robotics 用一格格 teleoperating 的 ALOHA 系统同时做数据采集与评估），既昂贵又 noisy：初始构型怎么设、光照怎么变、甚至制造商带来的摩擦参数差异，都会显著改变策略的鲁棒性；等结果要两天，讲者转述对方原话「预算足够大到还能推进」。更麻烦的是**训练 loss 与真实世界成功率相关性很弱**：监督学习里 loss 直接衡量模型好坏，而策略学习里 loss 衡量的是单步预测质量，往往不能反映长时程任务表现——loss 更低，长程执行反而可能更差。训练目标与任务指标、train/test horizon 的不一致，使得很难构造哪怕是近似的代理指标，人们只能依赖真实评估。

仿真评估是另一条路，代表有 Stanford 的 BEHAVIOR 与 Meta 的 **Habitat 3.0**，但自身问题不少：sim-to-real gap（刚体、可变形物体、衣物如何精确仿真并与真实表现相关）、assets 的大规模泛化与生成、如何数字化真实世界、如何程序化生成真实且多样的内容；核心是要找到 sim 与 real 之间的相关性。讲者用 ImageNet 作类比：希望 embodied AI 也有这样一个平台——「在该基准上的进步就等于机器人学习的进步」。

![Foundation Policy到Foundation World Models的转换示意图，包含世界模型的定义](/coursenotes/cs231n-deep-learning-for-cv-sp25/l17-robot-learning/kf_0020_01-17-14.jpg)

最后是 **Foundation Policy → Foundation World Models** 的延伸。讲者给出自己的定义：**world models = action-conditioned future prediction**（以动作作为条件的未来预测）。人们已在收集大规模、action-conditioned 的机器人交互数据来训练基础策略，这些数据里嵌入了大量动力学知识，若只拿来训策略就太浪费了；应当同时用它们训练**基础世界模型**，并让基础策略与世界模型互相配合。已有工作在探索这一方向，值得思考的特性包括：要不要 3D 表示、引入多少结构先验、learning 与 physics 各占多少、如何与真实物理世界对齐。讲者以愿景收尾：真正做出能在我们周围非结构化环境中广泛、可靠泛化的基础机器人模型；下一讲是 Human-Centered AI。

## 要点回顾

- 机器人学习与 CV 的根本差别：CV 学输入到表示／输出的映射，机器人是在物理世界约束下求解序贯优化问题；形式化四要素是 goal $g$、state $s_t$、action $a_t$、reward $r_t$，实例涵盖 CartPole、locomotion、Atari、围棋、LLM 生成、chatbot、叠衣服。
- 机器人感知是 **embodied / active / situated** 的：面对遮挡、传感误差、抓取失败、可变形物体与其他 agent，需融合 RGB(D)、触觉、音频、深度并做感知-动作协同设计；只关注任务相关区域，不必重建完整状态。
- RL 与监督学习四点差异：环境随机性、奖励延迟带来的信用分配、环境不可微（需零阶梯度估计）、以及「观测取决于自身动作」的非平稳性。
- Q 函数度量折扣期望未来累积奖励，决策取 $\arg\max_a Q(s,a)$；DQN 输入 $4\times84\times84$ 四帧堆叠、输出各动作 Q 值；SAC、PPO 是当前 SOTA。AlphaGo → AlphaGo Zero（bitter lesson / scaling）→ AlphaZero → MuZero，AlphaStar、OpenAI Five 攻克更复杂游戏。
- sim-to-real：domain randomization（随机化摩擦、几何等）让真实世界成为分布中的一个点，locomotion 已接近被 RL 解决；OpenAI 解魔方成功率实际很低，manipulation 仍局限于孤立环境。
- 无模型 RL 瓶颈：试错、海量交互、安全、可解释性差；人类靠直觉世界模型预测动作后果。解法是学 $P(s_{t+1}\mid s_t,a_t)$ 做 forward model，再用梯度优化动作序列、MPC 式只执行首动作并重新规划，靠 GPU 并行采样。
- 状态表示从像素（Deep Visual Foresight）、关键点，到粒子；RoboCook（CoRL 2023 最佳系统论文）用 15 种工具、4 个 RGBD 相机与粒子动力学做饺子，两层决策（选工具分类器 + 低层动作），抗人为扰动，且学到的模型比 MPM 物理仿真更准。
- 模仿学习：Behavior Cloning 学 $\pi_\theta$，核心问题是 cascading error，靠 corrective behaviors 迭代补数据；IRL 从行为反推奖励（早期 Abbeel / Ng 的直升机）；隐式能量策略 $E_\theta(o,a)$ 处理多峰示范；Diffusion Policy 把扩散模型用作策略，早上采数据、下午即可用。
- 机器人基础模型（VLA / Large Behavior Models）：不学环境模型，仍是从 $(observation, goal)$ 到 actions 的策略，但要「beautiful 且 reasonable」地广泛泛化；RT-1 (2022.12) → π0 (2024.10)，从预训练 VLM 出发、co-fine-tuning 兼顾动作与 VQA 目标、再 post-training；语义泛化免费、动作泛化靠机器人数据。
- 剩余挑战：真实评估昂贵且 noisy、训练 loss 与长程成功率相关性弱；仿真评估有 sim-to-real 与 assets 难题，期待 embodied AI 的「ImageNet」；并把大规模 action-conditioned 交互数据用于 **Foundation World Models**（action-conditioned future prediction），与基础策略互相配合。

## 视频来源

https://www.youtube.com/watch?v=XSfmOH_xVSU
