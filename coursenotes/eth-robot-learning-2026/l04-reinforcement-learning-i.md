---
title: "第 4 讲 · 强化学习 I：价值函数与价值迭代"
description: "本讲把问题从「有人告诉我们怎么做」的模仿学习，切换到「agent 自己通过试错去学策略」的强化学习：模仿学习永远被专家演示的质量封顶，而强化学习没有这个天花板。整讲的骨架是一条不断「去掉一个昂贵依赖」的推导链——先用价值…"
---

# 第 4 讲 · 强化学习 I：价值函数与价值迭代

> **ETH Zürich 机器人学习 · L04** ｜ 讲者 Oier Mees ｜ 53 分钟 ｜ [原视频](https://www.youtube.com/watch?v=90raNpc11tQ)

## 核心结论

本讲把问题从「有人告诉我们怎么做」的模仿学习，切换到「agent 自己通过试错去学策略」的强化学习：模仿学习永远被专家演示的质量封顶，而强化学习没有这个天花板。整讲的骨架是一条不断「去掉一个昂贵依赖」的推导链——先用**价值函数**把必须整条 rollout 才能算出的期望回报改写成一步自举的贝尔曼方程；再用**价值迭代 / 策略评估 / 策略迭代**这类精确动态规划算法把最优策略算出来，但它们依赖已知动力学模型、只能在离散小状态空间的表格里玩；于是把动力学模型「烤进」$Q(s,a)$（测试时不再查模型），再用**单个经验样本替代期望**得到 Q-Learning（训练时也不再需要模型），并顺带引出 off-policy 与 SARSA 的 on-policy 之别；最后用神经网络替换表格（DQN）来扛图像观测的维度灾难，配套经验回放与延迟目标网络两个稳定化技巧，并把 DQN 卡在离散动作上这一短板交给连续动作空间的三条出路：空间离散化、（交叉熵）采样、以及用 actor 直接学出 argmax 的 DDPG。

## 从模仿学习到强化学习：为什么要换范式 [00:00 - 01:31]

![课程标题页：机器人学习：从基础到基础模型，第四讲强化学习I](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0001_00-00-00.jpg)

本讲属于课程 *Robot Learning: From Fundamentals to Foundation Models*（ETH Zürich × Microsoft，讲师 Oier Mees，2026-03-09）。

上一讲的模仿学习设定是：给定「专家」采集的轨迹数据集 $\mathcal{D} = \{(s_1, a_1, \dots, s_t)\}$，学一个策略 $\pi_\theta$ 去「模仿」专家行为。幻灯片用自动驾驶举例：观测路面图像 → Policy → 输出一个转向决策（画面上是两个分叉箭头，左/右）。

![模仿学习回顾：专家轨迹数据集与策略学习目标示意图](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0002_00-00-53.jpg)

讲师点出这一范式的**根本局限**：我们完全依赖专家演示的质量，模仿学习在定义上就没有「比专家更好」的概念——性能天花板就是那份未知专家采出来的轨迹数据集。本讲要做的，就是用强化学习打破这层天花板：让 agent 自己去交互、去试错。

## 回顾：MDP 与学习目标 [01:31 - 03:36]

强化学习建立在马尔可夫决策过程之上，四元组 $M = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R} \rangle$：

- $\mathcal{S}$：状态空间，$s_t \in \mathcal{S}$
- $\mathcal{A}$：动作空间，$a_t \in \mathcal{A}$
- $\mathcal{P}$：转移概率，$s_{t+1} \sim \mathcal{P}(\cdot \mid s_t, a_t)$
- $\mathcal{R}$：奖励函数 $r$ 把 $\mathcal{S} \times \mathcal{A}$ 映到实数，$r_t = R(s_t, a_t, s_{t+1})$（幻灯片这里写作 $r: \mathcal{S}\times\mathcal{A} \to \mathbb{R}$，但 $R$ 的自变量还包含下一状态 $s_{t+1}$）

![马尔可夫决策过程定义及环境-机器人交互回路图](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0003_00-01-45.jpg)

幻灯片右侧的回路图值得注意方向：环境（地球图标）向机器人（机械臂）输出 **state $s_t$, reward $r_t$**，机器人向环境输出 **action $a_t$**。

优化目标（Recap: The Learning Objective）：

$$G = \sum_{t=0}^{T-1} r_t \quad \text{or} \quad G = \sum_{t \ge 0} \gamma^t r_t$$

$$J(\pi) = \mathbb{E}_{\tau \sim \mathbb{P}_\pi(\tau)}\Big[\sum_{t=0}^{T-1} r_t\Big] \quad \text{or} \quad J(\pi) = \mathbb{E}_{\tau \sim \mathbb{P}_\pi(\tau)}\Big[\sum_{t \ge 0} \gamma^t r_t\Big], \qquad \pi^* = \arg\max_\pi J(\pi)$$

![强化学习学习目标回顾：累积奖励、期望回报与最优策略公式](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0004_00-02-38.jpg)

注意期望的下标是 $\mathbb{P}_\pi(\tau)$（黑板体 P，指策略诱导的轨迹分布）。右下角示意图画的是从初始状态 $s_0$ 出发的多条随机轨迹，分别终止于 $s_{T1}, s_{T2}, s_{T3}, s_{T4}$，其中被高亮的那条是回报最高的路径。上一讲留的坑——**这个 argmax 到底怎么算出来**——就是今天的主题。

## 稀疏奖励不是不可导吗？期望会把它抹平 [03:36 - 07:05]

玩具例子：在悬崖窄路上开车，留在路上 $+1$，掉下悬崖 $-1$：

$$r(s,a) = \begin{cases} +1 & \text{if } s \in \mathcal{S}_{\text{success}} \\ -1 & \text{otherwise} \end{cases}$$

![稀疏奖励平滑化推导：通过伯努利分布将期望回报转化为平滑目标函数](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0006_00-05-15.jpg)

如果直接对状态（比如车的横向位置）优化这个奖励，它是**非平滑**的：本质是一个 Heaviside 阶跃函数，不连续、不可微，梯度几乎处处为零（唯一的跳变点处梯度未定义），因此拿梯度下降去优化根本得不到学习信号。

那为什么强化学习照样能学这类二值奖励？关键在于**强化学习优化的是期望**。把「掉下去」这件事建模成一个伯努利事件：

$$\pi_\theta(a = \text{fall} \mid s) = p_\theta(s), \qquad a \mid s \sim \text{Bernoulli}(p_\theta(s))$$

$$J(\theta) = \mathbb{E}_{a \sim \pi_\theta}[r(a)] = p_\theta(s)\cdot(-1) + \big(1 - p_\theta(s)\big)\cdot(+1) = 1 - 2p_\theta(s)$$

幻灯片在 $J(\theta)$ 的期望项上标了一个大括号写着 **smooth!**：即使底下的奖励函数高度不连续，期望回报关于策略参数 $\theta$ 依然可以完全平滑，于是梯度下降这类光滑优化方法仍然适用。

![稀疏奖励非平滑性示意图，对比不可微的阶跃奖励与可微的期望奖励曲线](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0007_00-07-00.jpg)

右侧图 `Differentiable Landscape in Parameter Space` 把这件事画得很直白：横轴 Policy Parameter $\theta$（约 $[-5,5]$），纵轴 Value（$[-1.5,1.5]$）。红色实线是 $r(s,a)$（Non-Smooth），在 $\theta = 0$ 处从 $-1$ 跳到 $+1$；蓝色虚线是 $\mathbb{E}_{a \sim \pi_\theta}[r(s,a)]$（Smooth!），是一条穿过原点、两端渐近 $\pm 1$ 的 S 形曲线。讲师解释这条曲线**相当于奖励函数与策略密度的卷积**。

> 本讲第一句核心结论：**对随机策略做边际化（marginalize），能把非平滑的奖励函数变成平滑的目标函数。**

## 别等车掉下悬崖才知道那是错的 [07:05 - 09:00]

目标函数展开后是

$$J(\pi) = \mathbb{E}_{\tau \sim \mathbb{P}_\pi(\tau)}\Big[\sum_{t \ge 0} \gamma^t r_t\Big] = \mathbb{E}_{s_0, a_0, s_1, \dots}\big[r_0 + \gamma r_1 + \gamma^2 r_2 + \cdots\big]$$

它是一个标量，把一个策略「平均而言从初始状态出发有多好」压成一个数。但幻灯片标题 **Don't Wait for the Cliff** 点出了实践困难：右边那串 $r_0 + \gamma r_1 + \gamma^2 r_2 + \cdots$ 意味着我们必须把整条轨迹一路展开、跑到回合结束，才知道自己做得怎么样。对开车这个例子来说，这等于要先让车真的掉下悬崖，才知道不该这么做——又慢又可能灾难性。

![提出能否不展开完整回合来估计J(π)的问题](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0009_00-08-45.jpg)

于是幻灯片在展开式下面画了大括号，提出本节的驱动问题：**Can we estimate $J(\pi)$ without unrolling each episode to the end?** 答案是能，靠的是 MDP 的**马尔可夫性质**：整条轨迹分布可以因子分解，未来只依赖当前状态而不依赖完整历史。

## 价值函数：把「整条轨迹好不好」变成「这个状态好不好」 [09:00 - 10:41]

$$V^\pi(s) = \mathbb{E}\Big[\textstyle\sum_{k=0}^{\infty} \gamma^k r_{t+k} \;\Big|\; s_t = s,\; a_t \sim \pi(\cdot \mid s_t),\; s_{t+1} \sim P(\cdot \mid s_t, a_t)\Big]$$

它和 $J$ 一样是「一个数概括好坏」，区别在于它是**条件在某个状态上**的局部版本：我的策略从这个具体状态往后有多好。因为马尔可夫性，条件在当前状态上就足以汇总所有关于过去的有用信息。

$$J(\pi) = \mathbb{E}_{s_0 \sim p(s_0)}\big[V^\pi(s_0)\big]$$

![价值函数定义幻灯片，包含期望折扣回报公式与RL目标函数](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0010_00-09-38.jpg)

也就是说，RL 目标函数就是价值函数在起始状态分布上的平均——价值函数是更细粒度的量，$J$ 是它的宏观版本。

## 自举（Bootstrapping）与贝尔曼期望方程 [10:41 - 12:08]

幻灯片标题 **Bootstrapping: Recursive Shortcut**。任何无穷奖励和都可以拆成「现在立刻拿到的」加「从明天起打折拿到的」：

$$\sum_{t \ge 0} \gamma^t R(s_t, a_t) = R(s_0, a_0) + \gamma \underbrace{\sum_{t \ge 1} \gamma^{t-1} R(s_t, a_t)}_{\text{this is } V!}$$

于是无限时域问题被改写成一个**一步递归方程**（Bellman Expectation Equation）：

$$V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot \mid s),\; s' \sim P(\cdot \mid s, a)}\big[r(s,a) + \gamma V^\pi(s')\big]$$

![自举递归捷径幻灯片，推导贝尔曼期望方程](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0011_00-11-23.jpg)

幻灯片下方对这条公式的注释值得原样记住：*value now = immediate reward + discounted value of next state*，因此 **allows value estimation without full rollouts**。回到开车例子：某个位置的价值 = 此刻的即时奖励 + $\gamma$ × 我会去到的那个下一状态的价值。不需要模拟剩下的整段车程，只要知道下一状态有多好，价值函数替你把之后的一切算好了——这就是「自举」：一步前瞻 + 用当前对 $V$ 的估计顶替其余部分。

## 最优价值函数与最优策略：把无限规划压成一步贪心 [12:08 - 14:18]

$$V^*(s) = \max_\pi V^\pi(s), \qquad \pi^* = \arg\max_\pi J(\pi) = \arg\max_\pi \mathbb{E}_{s_0 \sim p(s_0)}\big[V^\pi(s_0)\big]$$

$$\pi^*(s) = \arg\max_a \mathbb{E}_{s' \sim P(\cdot \mid s, a)}\big[r(s,a) + \gamma V^*(s')\big]$$

![最优策略幻灯片，包含最优价值函数与最优策略公式及环境-机器人交互示意图](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0012_00-12-16.jpg)

$V^*$ 的语义（讲师的说法很形象）：在山路某个位置上，**由最好的可能司机来开**时的价值——任何策略从该状态出发能达到的性能天花板；而 $\pi^*$ 就是那个从每个起始状态都能达到天花板的策略。

让这一切变得可算的关键洞察：一旦有了 $V^*$，行动时**不需要再搜索所有策略**，只要做一次贪心的一步前瞻。换句话说，最优价值函数把一个无限时域的规划问题**压缩成了一串一步贪心决策**——所有关于很久以后的长期推理，都被隐式编码进了这个函数里，算出来之后每次决策都只是局部计算。

## 价值迭代：算法与代价 [14:18 - 16:13]

幻灯片标题 **Value Iteration**，副标题直接是本节的问题：*How can we compute the optimal policy in practice?*

1. 初始化 $V_0(s_t) = 0 \quad \forall s_t \in \mathcal{S}$（先假设每个状态一文不值）
2. For $k = 0, 1, 2 \dots$ until $\max_{s_t} |V_{k+1}(s_t) - V_k(s_t)| < \epsilon$：
   $$V_{k+1}(s_t) \leftarrow \max_a \big[r(s_t, a) + \gamma\,\mathbb{E}_{s_{t+1} \sim \mathcal{P}(\cdot \mid s_t, a)} V_k(s_{t+1})\big]$$
3. 收敛后提取策略 $\pi^*(s_t) = \arg\max_a \big[r(s_t, a) + \gamma\,\mathbb{E}_{s_{t+1} \sim \mathcal{P}(\cdot \mid s_t, a)} V^*(s_{t+1})\big]$

![价值迭代算法的三个步骤：初始化、迭代更新状态价值、提取最优策略](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0013_00-14-53.jpg)

简单得意外：反复套用贝尔曼最优更新，最后一步贪心提取策略即可。收敛性有证明——贝尔曼算子是一个**压缩映射**（contraction）：把它作用在两个不同的价值函数上，会让它们比之前更接近。

![价值迭代算法的优缺点总结：收敛性证明与计算复杂度限制](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0014_00-15-46.jpg)

代价（幻灯片用 ✅/❌ 列出）：

- ✅ Convergence proof via Bellman contraction
- ❌ Assumes dynamics model（需要已知转移模型）
- ❌ Needs to sweep through all states，复杂度 $O(|\mathcal{S}|^2|\mathcal{A}|)$
- ❌ State space must be discrete & small（实践中要用表格表示，状态空间得能枚举）

## 可视化：价值迭代在 Gridworld 上「长出」$V^*$ [16:13 - 17:43]

讲师坦白这个演示是他让 Claude 写的。环境设置（从界面右侧面板可读到全部参数）：5×5 网格，右上角绿格 Goal（$+1$），其正下方红格 Danger（$-1$），四格灰色 Wall（blocked）；动作集 ↑↓←→；**随机性：80% 按意图方向移动，10% 向两侧垂直方向各偏一次**；$\gamma = 0.90$，$\epsilon = 0.001$。

![Gridworld 环境中价值迭代的可视化演示界面，展示状态价值热力图与迭代控制面板](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0015_00-16-14.jpg)

迭代 $k = 0$ 时所有可通行格子的价值都是 `0.000`，即「完全的无知」。开始套用贝尔曼更新后：

![价值迭代在Gridworld上的可视化演示界面，展示各状态价值与最优动作箭头](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0016_00-16-17.jpg)

到 $k = 4$（$\max|\Delta V| = 0.55818$）可以看到信息**一格一格地逆向传播**：靠近目标的格子先更新，顶行出现 $0.267 \to 0.950 \to 1.445 \to 1.845$ 的梯度，箭头一律指向右侧的 Goal；而远离目标、且被墙隔断的区域还停在 $-0.034$，箭头统一朝上（先绕开 Danger 那一格）。色标范围也从 $[-0.00, 0.00]$ 变成 $[-0.03, 1.85]$。图例里的 `Low V(s)` / `High V(s)` 是价值高低配色，不是数值 0。讲师的总结：**这是最纯粹形态的动态规划——没有路径规划，没有任何硬编码，只有反复的局部贝尔曼更新。**

## 策略评估：不问最好的动作，只问我会做什么 [17:43 - 19:36]

价值迭代每步都取 max，等于**隐含假设自己永远在最优地行动**。但有个更朴素的问题值得先回答：我手上这个具体策略到底有多好？（幻灯片副标题 *How good is our policy?*）

$$V_{k+1}(s_t) \leftarrow \mathbb{E}_{a \sim \pi(\cdot \mid s_t)}\big[r(s_t, a) + \gamma\,\mathbb{E}_{s_{t+1} \sim \mathcal{P}(\cdot \mid s_t, a)}\big[V_k(s_{t+1})\big]\big]$$

![策略评估算法步骤：初始化、迭代更新公式及收敛条件，并列出优缺点](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0017_00-17-44.jpg)

策略评估本身不改进任何东西，但它是通向策略改进的第一步——**你不知道自己有多好，就谈不上变好**。它的性质与价值迭代类似，但有一条优势：因为策略固定、不再对所有动作取 max，代价更低，复杂度降到 $O(|\mathcal{S}|^2)$（幻灯片用 ⚠️ 标注「Cheaper than value iteration, still sweep through all states」），依然需要动力学模型、依然要求状态空间离散且小。

把两条更新式并排放（幻灯片标题 **Value Iteration vs Policy Evaluation**，红色高亮唯一差异处）：

![价值迭代与策略评估的更新公式对比，突出max与期望的区别](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0018_00-19-16.jpg)

$$\text{Value Iteration:}\qquad V_{k+1}(s_t) \leftarrow \max_a \big[r(s_t, a) + \gamma\,\mathbb{E}_{s_{t+1} \sim \mathcal{P}(\cdot \mid s_t, a)}V_k(s_{t+1})\big]$$

$$\text{Policy Evaluation:}\qquad V_{k+1}(s_t) \leftarrow \mathbb{E}_{a \sim \pi(\cdot \mid s_t)}\big[r(s_t, a) + \gamma\,\mathbb{E}_{s_{t+1} \sim \mathcal{P}(\cdot \mid s_t, a)}[V_k(s_{t+1})]\big]$$

幻灯片用红色高亮了两处唯一的差异：上面那条的 $\max_a$ 与下面那条的 $\mathbb{E}_{a \sim \pi(\cdot \mid s_t)}$，其余部分完全一样。两者极易混淆，区别只在于对动作取 max 还是取期望。

## 策略迭代：评估与改进交替 [19:36 - 21:32]

幻灯片标题 **Policy Iteration**，一句话概括：*Alternates between finding out how good we are and trying to be even better.*

1. 任意初始化 $\pi_0(s_t) \ \forall s_t \in \mathcal{S}$
2. For $j = 0, 1, 2, \dots$ until $\pi_{j+1} = \pi_j$：
   - **Policy Evaluation**：$V_{k+1}(s_t) \leftarrow \mathbb{E}_{a \sim \pi_j(\cdot \mid s_t)}\big[r(s_t, a) + \gamma\,\mathbb{E}_{s_{t+1} \sim \mathcal{P}(\cdot \mid s_t, a)}[V_k(s_{t+1})]\big]$
   - **Policy Improvement**：$\pi_{j+1}(s_t) = \arg\max_a \big[r(s_t, a) + \gamma\,\mathbb{E}_{s_{t+1} \sim \mathcal{P}(\cdot \mid s_t, a)} V^{\pi_j}(s_{t+1})\big]$

![策略迭代算法流程图，包含策略评估与策略改进的交替步骤及对应公式](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0019_00-20-08.jpg)

幻灯片上两条手写式标注解释了结构：指向评估式的是 *this just one step of eval iteration until convergence*（写成一步，实际是迭代到收敛的内层循环）；指向改进式的大括号是 *find best action according to one-step look-ahead*。

策略迭代是**保证收敛**的：确定性策略的空间有限，而每一轮都会严格改进策略，直到抵达最优策略。实践中收敛很快。讲师用同一个 Gridworld 演示了两个阶段（评估 / 改进）的交替，右侧的回报曲线应当一路上扬——因为每次更新都必须更好。

## 对机器人而言 $V(s)$ 的致命缺陷 [21:32 - 23:33]

有了价值迭代和策略迭代，我们似乎已经能算出最优策略，而且保证收敛。但对机器人来说有一条硬伤（幻灯片标题 **Limitations of $V(s)$ for Robotics**）：

$$\pi^*(s_t) = \arg\max_a \big[r(s_t, a) + \gamma\,\mathbb{E}_{s_{t+1} \sim \mathcal{P}(\cdot \mid s_t, a)} V^*(s_{t+1})\big]$$

![说明基于状态价值函数V(s)的局限性：测试时需要动态模型进行查询](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0020_00-21-32.jpg)

式中的期望要把价值函数对所有转移分布做边际化，因此**在测试时（部署到真机上）每一个候选动作、每一个时间步都得去查询动力学模型**。幻灯片下方的示意图把这条流水线画得很清楚：SENSE（得到 state $s_t$）→ PLANNING（云里写着 SIMULATE ALL ACTIONS ($\mathcal{P}$), REQUIRES DYNAMICS MODEL，TO FIND HIGHEST VALUE ($V$)）→ ACT（EXECUTE BEST ACTION $a_t^*$）。机器人手臂喷出的一束候选轨迹正是「对所有动作做前向仿真」的视觉隐喻，而瓶颈就压在 PLANNING 这一段。

后果：

- 高频控制的机器人根本等不起这么长的计算；
- 机器人领域里动力学模型本来就常常拿不到——世界是脏的、难建模的；
- 即便有模型，每个控制周期都去查询也很慢，尤其是 7 自由度连续动作空间的操作臂，等于要评估无穷多个候选动作。

## Q 值：把动力学模型烤进价值函数 [23:33 - 25:20]

核心想法：**能不能在训练时就把动力学模型吸收进价值函数本身，使得测试时永远不必查询它？**

$$V^*(s_t) = \max_a \big[r(s_t, a) + \gamma\,\mathbb{E}_{s_{t+1} \sim \mathcal{P}(\cdot \mid s_t, a)}[V^*(s_{t+1})]\big]$$

- $V^*(s_t)$：从状态 $s_t$ 出发、之后一直最优行动的期望折扣回报
- $Q^*(s_t, a)$：从状态 $s_t$ 出发、**先承诺执行动作 $a$**、之后一直最优行动的期望折扣回报

$$Q^*(s_t, a) = r(s_t, a) + \mathbb{E}_{s_{t+1} \sim \mathcal{P}(\cdot \mid s_t, a)}\Big[\gamma \max_{a_{t+1}} Q^*(s_{t+1}, a_{t+1})\Big], \qquad V^*(s_t) = \max_a Q^*(s_t, a)$$

![Q值定义幻灯片，对比V*与Q*的期望回报公式及关系](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0021_00-23-39.jpg)

于是 $Q(s,a)$ 里已经预先算好了那个问题的答案：「如果我在这里执行 $a$，**对环境所有可能响应取平均之后**，最好的未来回报是多少」。测试时不再需要前向仿真，只是一次查表 / 对一个固定函数取 argmax。

代价与权衡：$Q$ 比 $V$ 多一个动作维度（不仅条件于状态，还条件于动作），因此可能更难学一点；但对拿不到可靠动力学模型的 model-free 机器人学来说，**这个代价通常是值得付的**。不过幻灯片在期望项下面标了一句红字：*still needs dynamics model during **training*** 🤔——训练时的 $\mathbb{E}_{s_{t+1} \sim \mathcal{P}}$ 还在。

## Q-Learning：用一次经验替换一个期望 [25:20 - 27:52]

幻灯片标题 **Q-Learning - Removing the Dynamics Model**。前提：动力学模型可能未知、过于复杂或太慢。解法（幻灯片上带下划线的原话）：**Replace Expectation with Experience**。

1. 执行动作并观察转移 $(s_t, a_t, r_t, s_{t+1})$
2. $\text{TD target} = r(s_t, a_t) + \gamma \max_a Q(s_{t+1}, a)$，若 $s_{t+1}$ 是终止状态则 $\text{TD target} = r(s_t, a_t)$
3. $$Q(s_t, a_t) \leftarrow (1 - \alpha)\,Q(s_t, a_t) + \alpha\,\underbrace{[\text{TD target}]}_{\text{new experience sample}}$$

![Q-Learning去除动力学模型的原理与更新公式，标注了旧估计、学习率和新经验样本](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0022_00-25-24.jpg)

第 3 步就是带标注的滑动平均：$(1-\alpha)Q(s_t,a_t)$ 是 **old estimate**，$\alpha$ 是 **learning rate**，方括号里是 **new experience sample**。我们不再对所有可能的未来状态求平均，而是**取一个动作、观察一个样本、存下这条转移**，用它构成真实 $Q$ 值的无偏估计，再把 $Q$ 往 TD target 方向挪一点——用讲师的话说，是**把世界本身当作自己的仿真器**。

代价：同一个 Gridworld 演示里，之前有动力学模型时策略收敛得很快；现在没有模型，必须花大量时间探索、从环境取样本。也就是说，**Q-Learning 的样本效率明显低于那些有模型的算法**——这是本讲反复出现的「用样本换掉模型」的交易。

## Off-Policy 与 On-Policy：谁来采集的数据重要吗？ [27:52 - 29:16]

幻灯片标题 **Off-Policy vs On-Policy**，问题是：*Does it matter **which policy** collects the transition $(s_t, a_t, r_t, s_{t+1})$ used to update our TD target?*

课堂上有学生答对了：**不重要**，因为 TD target 里的 max 是关键的算子。

$$\text{TD target} = \underbrace{r(s_t, a_t)}_{\text{what happened}} + \gamma \underbrace{\max_a Q(s_{t+1}, a)}_{\text{max assumes future optimal action, regardless of past behavior}}$$

![同策略与异策略对比：Off-Policy可利用任意数据（含次优行为）并配合经验回放缓冲区复用经验](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0024_00-28-54.jpg)

$\max$ 在下一状态选出「最好的可能动作」，完全不管这条转移是谁采的。所以 Q-Learning 是 **off-policy**：更新用的数据可以来自任何行为——随机 agent、策略的旧版本、人类演示者，或它们的任意混合；学习目标永远评估最优动作，与数据的采集方式无关。这也正是它能**复用经验（Experience Replay Buffer）**的前提。

## SARSA：把 max 换成「我实际会做什么」 [29:16 - 31:34]

$$\text{Q-Learning (Off Policy):}\quad \text{TD target} = r(s_t, a_t) + \gamma \max_a Q(s_{t+1}, a)$$

$$\text{SARSA (On-Policy):}\quad \text{TD target} = r(s_t, a_t) + \gamma\, Q(s_{t+1}, a_{t+1})$$

幻灯片在两条公式下面各写了一句大白话：$\max$ 那条是 *What is the best I **could** do next?*（我最好能做到什么），SARSA 那条是 *What am I **actually** going to do next?*（我实际上会做什么）。

![对比 Off-Policy 与 On-Policy 的 TD 目标公式，分别对应 Q-Learning 和 SARSA](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0025_00-29-46.jpg)

除了这个算子，其余部分完全一样——改动极小，行为影响却很大：SARSA 学的是**它当前正在执行的那个策略**的价值，包括所有探索造成的失误。讲师的例子：一个「紧张的司机」在自动驾驶中，SARSA 会正确地惩罚那些容易让紧张司机出错的状态，而 Q-Learning 会忽略这一点。

![Gridworld 环境中 Q-Learning 与 SARSA 的训练对比界面，包含回报曲线和策略路径可视化](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0026_00-30-39.jpg)

经典的 cliff walking 对照演示（界面顶部两行公式分别写着 `TD target = r + γ·max_a Q(s',a) — best I could do` 与 `TD target = r + γ·Q(s', a_{t+1}) — what I will do`，中间是 EPISODE RETURN 曲线，下方左右各一张含骷髅格的悬崖网格与当前贪心路径）：两个 agent 都从左侧出发、目标在右侧，中间是一排死亡格。结果 Q-Learning 找到**最优**策略——贴着悬崖边缘走最短路径；SARSA 学到一条**更安全但非最优**的路线——刻意远离崖边，承担绕路以换取不掉下去的风险。演示界面上那行小字总结得很准：训练期 SARSA 更谨慎（避开悬崖边缘状态），收敛后 Q-Learning 回报更高（路径更短）、SARSA 回报更低（安全绕路）。

## 到目前为止的方法分类 [31:34 - 32:24]

幻灯片标题 **RL Taxonomy**，两栏对照：

![强化学习分类图，对比精确解法与基于价值的方法的优缺点](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0027_00-31-34.jpg)

- **Exact Solution Methods**（Value Iteration / Policy Iteration / Q-Value Iteration）：✅ Convergence proof via Bellman contraction；❌ Sweep through all states；❌ Needs dynamics model；❌ State space must be discrete & small；❌ Tabular representation。
- **Value-based Methods**（Q-Learning (off-policy) / SARSA (on-policy)）：✅ Converges to $Q^*$（tabular, sufficient exploration）；✅ No dynamics model needed (model-free)；✅ Learns from interaction；❌ State space must be discrete & small；❌ Tabular representation。

也就是说，我们虽然摆脱了模型，却仍然被「表格 + 小状态空间 + 网格世界」卡住，对机器人还谈不上可用。

## 维度灾难：图像观测让表格彻底失效 [32:24 - 33:17]

幻灯片标题 **The Curse of Dimensionality: Scaling to Image Observations**，只有一行字：

$$\text{Atari game observation space} = 256^{84 \times 84 \times 4}$$

![维度灾难：图像观测的缩放问题，标注Atari游戏观测空间为256的84x84x4次方](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0028_00-32-24.jpg)

含义是：每个像素 256 个灰度值、图像 $84 \times 84$、并且要堆叠 4 帧才能感知运动，于是不同观测的数量是 $256^{84\times84\times4}$。这是天文数字，**不可能用表格存下来**。要往前走，就得把表示换掉。

## 案例研究：Deep Q-Learning (DQN) [33:17 - 35:02]

幻灯片标题 **Case Study: Deep Q-Learning (DQN)**：

- First time a single algorithm learned to play 49 different games from raw pixels
- Launched **Deep Reinforcement Learning** in 2013

![DQN案例研究：首个从原始像素学习玩49款游戏的单一算法，附2013年Playing Atari论文截图](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0029_00-33-17.jpg)

思路概念上极其简单：把 Q-Learning 里的一切照搬，只把表格换成一个深度卷积网络——输入一叠原始游戏帧（像素），输出一个 Q 值向量（每个离散动作一个分量），然后用 Q-Learning 的更新式训练，就这样。它之所以是奠基性工作，是因为**同一套架构、同一组超参数**，不做任何手工特征工程或针对单个游戏的改动，就能从原始像素学会 49 款 Atari 游戏。

幻灯片里贴的是原始论文《Playing Atari with Deep Reinforcement Learning》（Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller；DeepMind Technologies）首页。摘要原文值得注意一个数字差异：这篇 2013 年预印本自己写的是「应用到 **seven** Atari 2600 games，其中六个超过以往所有方法、三个超过人类专家」，而幻灯片要点说的是 49 款游戏（那是后续 Nature 版本的规模）。讲师还加了一条私人注释（红箭头指向作者 Martin Riedmiller）：*Personal note: Martin just had started his sabbatical at a small startup called DeepMind, when I joined Uni Freiburg.*

DQN 有两个关键创新让它真正跑起来。

## 创新一：经验回放缓冲区 [35:02 - 36:47]

思路：不要把每条转移只用一次就丢掉，而是存进缓冲区反复采样使用。

- Store transitions in buffer $D_{\text{replay}} = \{(s_t, a_t, r_t, s_{t+1})\}_{t=1}^{N}$ instead of discarding them
- Sample random minibatch each update → **breaks temporal correlations**（creates i.i.d. samples）
- Each transition **reused many times** → more sample efficient

![经验回放缓冲区机制图：存储转移样本、随机采样小批量打破时序相关性、与环境交互更新Q网络](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0030_00-35-02.jpg)

下方循环图画的是三个环节的闭环：缓冲区 $D_{\text{replay}}$（里面装着 $(s_1, a_1, r_1, \dots, s_t)$）经 **Sample Minibatch** 进入 **Update $Q_\theta$ (gradient step)**，更新后的网络再去 **Interact with Environment**，产生的新转移回流进缓冲区。动机有两层：连续取自同一条轨迹的转移之间强相关，对优化器不友好（讲师回指第一/二讲讲过的 i.i.d. 假设），随机采样能把这种时序相关性打散、让样本近似 i.i.d.，从而稳定训练；同时那些「有意思的边角情况」可以被回放多次，不必再重新经历一遍。

## 创新二：延迟目标网络 [36:47 - 37:31]

第二个难题来自「估计值的网络」和「目标值的网络」是同一个网络。贝尔曼方程说当前价值依赖下一状态的 Q 值，而它用的是同一套参数 $\theta$：

$$\mathcal{L}(\theta) = \mathbb{E}_{(s_t, a_t, r_t, s_{t+1}) \sim \mathbb{D}_{\text{replay}}}\Big[\big(r(s_t, a_t) + \gamma \max_a Q(s_{t+1}, a; \theta) - Q(s_t, a_t; \theta)\big)^2\Big]$$

![延迟目标网络（Delayed Target Network）的损失函数对比：上方为原始TD损失导致优化不稳定，下方为引入冻结参数θ⁻的目标网络实现稳定监督回归](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0031_00-36-47.jpg)

幻灯片把两个 $\theta$ 分别标注为 **target moves every step** 和 **prediction moves every step**，并用箭头指向结论 **optimizer unstable & diverges**——更新当前 Q 值的同时也改写了下一状态的目标值，等于自己追自己的影子，训练不稳甚至发散。

解法：维护一份参数冻结的副本 $\theta^-$ 来计算目标：

$$\mathcal{L}(\theta) = \mathbb{E}_{(s_t, a_t, r_t, s_{t+1}) \sim \mathbb{D}_{\text{replay}}}\Big[\big(r(s_t, a_t) + \gamma \max_a Q(s_{t+1}, a; \theta^-) - Q(s_t, a_t; \theta)\big)^2\Big]$$

标注是：$\theta^-$ 是 *copy of $\theta$ frozen for some steps, then $\theta^- \leftarrow \theta$*，而 $\theta$ 那一项 *updated every step*。结论一句话：**Freezing $\theta^-$ makes the target fixed → stable supervised regression**（把 TD 学习变成一个有固定监督信号的回归问题）。目标网络要么缓慢平均更新，要么每隔固定步数整体复制一次。

## DQN 的结果与它的系统性偏差 [37:31 - 39:14]

幻灯片标题 **DQN**，要点是它适用的问题形态：**Discrete action space, image state space, sparse & dense rewards**。

![DQN在Atari游戏上的性能对比表，展示DQN在多个游戏上超越Sarsa、Contingency等基线并接近人类水平](/coursenotes/eth-robot-learning-2026/l04-reinforcement-learning-i/kf_0032_00-37-31.jpg)

结果表（幻灯片重排的《Playing Atari》Table 1，DQN 行为加粗）：

| | B. Rider | Breakout | Enduro | Pong | Q\*bert | Seaquest | S. Invaders |
|---|---|---|---|---|---|---|---|
| Random | 354 | 1.2 | 0 | −20.4 | 157 | 110 | 179 |
| Sarsa [3] | 996 | 5.2 | 129 | −19 | 614 | 665 | 271 |
| Contingency [4] | 1743 | 6 | 159 | −17 | 960 | 723 | 268 |
| **DQN** | **4092** | **168** | **470** | **20** | **1952** | **1705** | **581** |
| Human | 7456 | 31 | 368 | −3 | 18900 | 28010 | 3690 |

DQN 在所有七列上都碾压 Random / Sarsa / Contingency，并在 Breakout（168 vs 人类 31）、Enduro（470 vs 368）、Pong（20 vs −3）三项上超过人类——正好对应摘要里那句「surpasses a human expert on three of them」。讲师强调最重要的不是原始分数，而是**通用性**：一个开箱即用、没做超参调优的算法就能在这批游戏上拿到很好的成绩。

但 DQN 有一个系统性失效模式：**过估计偏差（overestimation bias）**。Q 值本身带噪声、有误差，而**一组带噪声估计的最大值总是大于等于真实值的最大值**——取 max 不会把噪声平均掉，反而把它放大。后来的 **Double DQN** 用一个很简单的洞察修掉了它：偏差来自**用同一个网络既去挑最好的动作、又去评价这个动作**。于是把两件事解耦——用在线网络选动作、用目标网络评价它；还是 DQN 的那两个网络，只是重新接线（视频里只有口头描述，写成符号即 $\text{TD target} = r + \gamma\, Q\big(s', \arg\max_{a'} Q(s', a'; \theta);\ \theta^-\big)$）。

## 连续动作空间：argmax 算不动了 [39:14 - 41:15]

DQN 在**离散**动作空间上很漂亮，但机器人（操作臂等）关心的是**连续**动作空间。问题出在 $\arg\max_a$：动作空间无限，无法枚举，这类连续优化也没有闭式解。讲师给出三条出路，它们回答的其实是同一个问题——**如何在一个连续空间上找到一座复杂非线性函数的峰**，选哪条取决于你的动作维度与具体设定：

1. **离散化**：把连续动作空间切成网格，直接套 DQN。简单，但代价随维度指数增长——7 自由度机械臂即使每个关节只做粗糙离散化，组合数也会爆炸。
2. **采样**：蒙特卡洛地抽 $N$ 个候选动作，逐个评价 $Q$，返回最好的那个。
3. **用第二个网络学出 argmax**：这就是 **DDPG**——不在测试时求解优化问题，而是把它搬到训练阶段去学。

## 出路一：利用几何结构的空间离散化 [41:15 - 43:16]

一个「相当聪明」的离散化例子（视频未给出论文名）：针对 pick-and-place，它**利用问题的几何结构**，把动作映射到相机的 2D 俯视图像上。于是没有单独的动作头，而是用一个**全卷积网络**：输入高度图（RGB-D），输出一张**稠密的 Q 值图**——每个像素、每个旋转角对应一个 Q 值，旋转角取 16 个离散朝向；argmax 直接在这张图上做。这样绕开了组合爆炸。

演示里最漂亮的观察是：机器人**并不是立刻学着去抓**。它先把杂乱的物体**推开**，把场景整理成更容易抓取构型，然后再尝试抓取——先整理、后抓取。这套「先推后抓」的策略没有人编程、没有人告诉它推是有用的，纯粹来自最大化奖励信号。机制上，这通过贝尔曼回传自然涌现：抓取用的 Q 函数会给杂乱的构型打很低的价值，因为在那里抓取成功率低。

适用边界：空间离散化在**动作空间本身带有可利用的几何结构**时才好用。

## 出路二：蒙特卡洛采样与交叉熵方法 [43:16 - 45:47]

更一般的情形（例如 7 自由度机械臂的关节配置）最朴素的做法是蒙特卡洛采样：从动作空间均匀采 $N$ 个动作，逐个评价 $Q(s,\cdot)$，返回最好的。它之所以可行，是因为 $Q(s, a)$ 关于 $a$ 是一个**平滑的神经网络函数**，只要采得足够密，总能找到接近真实 argmax 的动作。缺点也很直接：每次动作选择要做 $N$ 次前向传播，扩展性差；而且质量完全取决于样本数和运气，没有保证。

**交叉熵方法（CEM）** 是蒙特卡洛采样的聪明版本：不再在整个动作空间里盲采，而是**迭代地把样本集中到高 Q 值的区域**。算法很简洁：

1. 从一个高斯分布采一批动作；
2. 保留 Q 值最高的若干 **elite** 动作；
3. 用这批 elite 重新拟合高斯；
4. 重复若干轮，返回最优动作。

它效果很好，但有两个限制：其一，**单峰**——它倾向于找到你采样位置附近的那座峰，不保证是真正的全局 argmax；其二，**优化的代价仍然付在测试时**——每个控制周期都要做 $N \times K$ 次前向传播（$N$ 个样本、$K$ 轮迭代），要么很慢，要么得牺牲精度。

一个把这套东西真正搬到机器人上的工作是 **QT-Opt**（Google 与 Berkeley，2018）：讲师回忆那是一个「机器人农场」，大约七台机器人，用 Q-Learning + 交叉熵方法 + 一个巨大的回放缓冲区（约 58 万次真实抓取）第一次让这件事在真实抓取上跑通。关键正在于回放缓冲区：它让每一条极其昂贵的真实世界转移都能贡献多次梯度更新。

## 出路三：DDPG——把 argmax 学成 actor [45:47 - 50:44]

既然交叉熵方法要把 $N \times K$ 次前向传播压在测试时、对高频控制太浪费，第三条路 **DDPG（Deep Deterministic Policy Gradients）** 干脆**去学这个 argmax，而不是去搜索它**：引入第二个网络 **actor**，它唯一的工作就是**直接预测让 Q 函数最大的那个动作**。于是测试时那次昂贵的迭代搜索被压缩成**一次前向传播**。

DDPG 因此有两个网络：

- **critic**：就是 DQN 里那个标准 Q 网络，仍用 TD loss 训练，估计状态-动作对的价值；
- **actor**：新增的部分，专门训练成输出「critic 认为价值高」的动作——输入状态，直接输出一个连续动作。

如何更新 actor？讲师的说法是：问 critic **在动作空间里往哪个方向 Q 值会上升**，然后把 actor 的参数朝那个方向推。数学上就是对期望回报套链式法则——取 Q 对动作的梯度、在 actor 当前的输出处求值、再乘上 actor 对其权重的梯度（视频只给了文字描述，写成符号即 $\nabla_\phi J \approx \mathbb{E}_s\big[\nabla_a Q(s, a)\big|_{a = \mu_\phi(s)} \cdot \nabla_\phi \mu_\phi(s)\big]$）。严格说，训练 actor 已经属于**策略梯度**的领地，讲师明确把细节留给下一讲：「就当成 actor 在尽力最大化 Q 值，剩下的交给梯度下降」。

第二个要处理的问题是**探索**。DQN 时代动作是离散的，可以用「以某概率随机选一个动作」的贪心式探索；现在动作空间连续，随机采样关节配置会把机器人**猛甩（thrash）**。DDPG 的做法是：训练时**在高斯噪声直接加到 actor 的输出上**——拿它建议的动作，再加一点平滑的高斯扰动，从而在当前策略附近做**局部探索**。

评价：DDPG 已经能处理高维观测与连续动作，测试时也不再需要交叉熵方法，并且有真实机器人上的应用。但它是一个**出了名脆弱（notoriously brittle）**的算法——对超参数极其敏感、不保证收敛，而且那个噪声参数必须针对每个新环境手工调：调太小探索不足，调太大训练不稳定。

## 全讲总结：一条能力与保证此消彼长的路线 [50:44 - 52:32]

最后讲师用一张分类表把今天的方法按「换取了什么、付出了什么」串起来（视频末尾未截到该页，以下按讲述内容整理）：

| 方法族 | 换来什么 | 付出什么 |
|---|---|---|
| Exact methods（VI / PI / Q-VI） | 干净、有收敛保证 | 需要已知动力学模型，只能表格化、状态空间极小 |
| Q-Learning / SARSA | 从经验学习而非模型；只要探索足够久仍有收敛保证 | 仍困在小型玩具环境 |
| Deep RL, 离散动作（DQN / Double DQN） | 首个神经网络表示，能直接吃图像观测，配回放缓冲区等 | **不再有收敛保证**，仍限于离散动作空间 |
| Deep RL, 连续动作（DDPG / QT-Opt / CEM） | 连续动作、高维状态空间，仍是 off-policy + 回放缓冲区 | 训练脆弱，探索需手工设计 |

一句话收束：**每往前一步，都是拿「理论保证」换「适用范围」**——而机器人恰恰需要后面这几档。

## 要点回顾

- 模仿学习的天花板是专家演示的质量；强化学习靠 agent 自己交互试错，原则上可以超越专家。
- 「稀疏奖励不可导怎么办」：$r(s,a)$ 是阶跃函数、梯度几乎处处为零，但**对随机策略取期望后目标函数变平滑**（等价于奖励与策略密度的卷积），因此梯度下降依然可用。这是本讲第一条主线。
- $J(\pi)$ 必须整条 rollout 才能算出，慢且可能灾难；**价值函数** $V^\pi(s)$ 靠马尔可夫性把「条件在整段历史上」换成「条件在当前状态上」，$J(\pi) = \mathbb{E}_{s_0 \sim p(s_0)}[V^\pi(s_0)]$。
- **贝尔曼期望方程**把无限时域问题写成一步递归：value now = immediate reward + discounted value of next state，即**自举**，无需完整 rollout 即可估值。
- $V^*$ 把无限规划压缩成一串一步贪心决策：有了 $V^*$ 就不必再搜索策略空间。
- **价值迭代**（max + 表格遍历，$O(|\mathcal{S}|^2|\mathcal{A}|)$）、**策略评估**（把 max 换成对策略的期望，$O(|\mathcal{S}|^2)$）、**策略迭代**（评估与改进交替，确定性策略空间有限故保证收敛）——三者都**需要已知动力学模型**且要求状态空间离散、小。
- 对机器人致命的地方：用 $V(s)$ 行动时**测试时**要为每个状态-动作对查询动力学模型，高频控制与 7 自由度连续动作空间都撑不住。
- **Q 值**把动力学模型在训练时吸收进 $Q(s,a)$，测试时只剩查表/argmax；代价是多一个动作维度。
- **Q-Learning** 再用「单个真实样本」替换训练时的期望：$Q \leftarrow (1-\alpha)Q + \alpha[r + \gamma\max_a Q(s',a)]$，终止态时 TD target 退化为 $r$；完全 model-free，但样本效率显著下降。
- TD target 里的 $\max$ 让 Q-Learning 成为 **off-policy**：数据可以来自任何行为（随机 agent、旧策略、人类演示），因此可以复用经验；**SARSA** 用 $Q(s_{t+1}, a_{t+1})$ 替换 $\max$，变成 **on-policy**，会把探索失误一并算进价值，于是在 cliff walking 上学到更安全的绕路策略。
- 图像观测的状态空间是 $256^{84\times84\times4}$，表格彻底不可行 → **DQN** 用卷积网络替换表格，2013 年以同一架构+同一超参从原始像素学会 Atari，开启深度强化学习。
- DQN 两个稳定化创新：**经验回放**（存 $(s,a,r,s')$、随机采小批量以打散时序相关性、近似 i.i.d. 并复用经验）与**延迟目标网络**（冻结 $\theta^-$ 使目标固定，把 TD 学习变成稳定的监督回归）。
- DQN 的系统性缺陷是**过估计偏差**（噪声在取 max 时被放大），**Double DQN** 用「在线网络选动作、目标网络评价」解耦修正。
- 连续动作下 $\arg\max_a$ 无闭式解，三条出路：**离散化**（含利用几何结构的空间离散化：全卷积网络输出稠密 Q 图、16 个离散朝向，涌现出「先推开再抓取」）、**采样**（蒙特卡洛与交叉熵方法，代价付在测试时、且单峰）、**DDPG**（用 actor 直接学出 argmax，测试时一次前向传播，但出了名脆弱、探索噪声要手工调）。
- 整讲的取舍逻辑：从精确方法到 model-free，到深度离散 RL，再到深度连续 RL，**理论收敛保证逐步让位于真实机器人所需的表达能力**。
