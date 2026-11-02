---
title: "第 5 讲 · 强化学习 II：策略梯度"
description: "本讲把上一讲的「先学价值函数、再指望 argmax 抽出好策略」彻底翻篇，改成直接参数化策略 并对它求梯度：先用对数导数技巧把无法采样的密度梯度 改写成可以用蒙特卡洛样本估计的 ，其中初始分布与动力学项在求导后自动消失，…"
---

# 第 5 讲 · 强化学习 II：策略梯度

> **ETH Zürich 机器人学习 · L05** ｜ 讲者 Oier Mees ｜ 52 分钟 ｜ [原视频](https://www.youtube.com/watch?v=AdTGz8YnnlE)

## 核心结论

本讲把上一讲的「先学价值函数、再指望 argmax 抽出好策略」彻底翻篇，改成**直接参数化策略 $\pi_\theta(a|s)$ 并对它求梯度**：先用对数导数技巧把无法采样的密度梯度 $\int \nabla_\theta p_\theta(\tau)R(\tau)d\tau$ 改写成可以用蒙特卡洛样本估计的 $\mathbb{E}_{\tau\sim p_\theta(\tau)}[\nabla_\theta\log p_\theta(\tau)R(\tau)]$，其中初始分布与动力学项在求导后自动消失，于是得到不需要环境模型的 REINFORCE。随后整讲是一条不断「打补丁」的链：用 **Reward-to-Go** 修好因果性，用**基线**（再演化为**优势函数** $A^\pi=Q^\pi-V^\pi$）把 99 分与 101 分这种「绝对回报、相对信号只有 2」的高方差问题压下去；接着撞上 on-policy 的样本效率墙，用**重要性采样**换来「一批数据做多次梯度更新」，代价是整轨迹比值指数爆炸，于是退到时步级并把状态比值近似为 1；再用**熵正则化**保住支撑集、用 **KL 约束**引出 TRPO（单调改进保证，但要 Fisher 矩阵、$O(N^3)$），最后用 PPO 的 **clip** 把二阶数学换成一行一阶操作，并落到 Actor-Critic 架构与完全 off-policy 的 Soft Actor-Critic。全讲贯穿机器人视角：真机采数据贵、可以给 critic 喂特权信息、TRPO/PPO/SAC 都有对应的真机或 sim-to-real 案例。

## 上一讲回顾：四类方法与本讲的出发点 [00:00 - 02:27]

![课程标题页：Robot Learning Lecture 5 Reinforcement Learning II，讲师 Oier Mees](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0001_00-00-00.jpg)

本讲属于课程 *Robot Learning: From Fundamentals to Foundation Models*，第五讲「Reinforcement Learning II」，讲师 Oier Mees，ETH Zürich 与 Microsoft 联合署名，标题页日期 16.03.2026。上一讲（第四讲）是 Reinforcement Learning I，本讲继续强化学习。

开场先做分类回顾。幻灯片标题是 **Recap: RL Algorithm Taxonomy & Trade-offs**，一张四列七行的对比表，四列依次是 Exact Methods、Tabular RL、Deep RL (Discrete)、Deep RL (Continuous)。逐行按画面抄录如下：

| 维度 | Exact Methods | Tabular RL | Deep RL (Discrete) | Deep RL (Continuous) |
|---|---|---|---|---|
| 代表算法 | Value Iter. / Policy Iter. / Q-Val. Iter. | Q-Learning / SARSA | DQN / Double DQN | DDPG / QT-Opt (CEM) |
| Needs Dynamics Model | ✓ Train+Test | ✗ Model-free | ✗ Model-free | ✗ Model-free |
| State Space | Discrete, small | Discrete, small | High-dim images | High-dim images |
| Action Space | Discrete | Discrete | Discrete | Continuous |
| Policy (train time) | Tabular | Tabular | Neural Net | Actor + Critic NNs |
| Off-policy | N/A | Q-Learn ✓ / SARSA ✗ | ✓ (Replay Buffer) | ✓ (Replay Buffer) |
| Convergence | ✓ Guaranteed | ✓ Tabular+explore | ✗ No guarantee | ✗ Brittle |
| Key Innovation | Bellman contraction | TD updates from experience | Replay buffer + Target network | Actor learns arg max_a Q |

![强化学习算法分类与权衡对比表，涵盖精确方法、表格RL及深度RL（离散/连续）](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0002_00-00-52.jpg)

讲师的口头串讲与这张表一致：精确方法（价值迭代、策略迭代、Q-值迭代）是「beautiful algorithms」，靠贝尔曼压缩映射拿到**收敛性保证**，代价是需要动力学模型，而且训练和测试都要用（表里的 Train+Test），这在机器人里通常拿不到；于是转向表格型 RL（Q-Learning、SARSA）从经验里学，走向 model-free；再用神经网络替换表格把方法推到高维图像观测（DQN、Double DQN），最后到连续动作空间（DDPG、QT-Opt/CEM）。他特别点出：上一讲那一整块 deep RL 的关键创新，本质上都是在**用不同方式解那个 argmax**。

注意表格最后一行 Key Innovation 里，连续动作那一格写的是「Actor learns arg max_a Q」——这已经预告了今天的方向。

那么本讲的出发点是上一讲的共同范式：

![基于价值的方法（Value-Based Methods）定义：学习动作价值并提取最优策略](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0003_00-01-45.jpg)

幻灯片只有一句话：**Value-Based Methods: learn what every possible actions is worth & hope argmax will extract the best policy**（画面原文如此，"actions" 是幻灯片自身的语法瑕疵）。也就是先学「每个动作值多少」，再**希望**对它取 argmax 能抽出最优策略。讲师接着说，我们已经看到这条路很难推到高维空间。

于是今天的问题是：既然我们真正关心的就是策略本身，**为什么不干脆跳过价值函数**，直接参数化策略，然后「wiggle the neural network weights」——把网络权重往能直接产生高回报行为的方向挪？换句话说，让网络自己产出「未来回报更高」的动作。

## 回顾：学习目标与轨迹概率 [02:27 - 04:20]

在开始推导之前，先把上一讲的数学摆出来，明确起点。幻灯片标题 **Recap: The Learning Objective**，三条：

- 一个 agent 收到的累积奖励：

$$G = \sum_{t \ge 0} \gamma^{t} r_{t}$$

- 期望回报（Expected Return）：

$$J(\pi) = \mathbb{E}_{\tau \sim p_{\pi}(\tau)}\left[\sum_{t \ge 0} \gamma^{t} r_{t}\right]$$

- 最优策略（Optimal Policy）：

$$\pi^{*} = \arg\max_{\pi} J(\pi)$$

![强化学习目标回顾：累积奖励、期望回报与最优策略的数学定义](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0004_00-02-37.jpg)

讲师逐条解释：我们就是在最大化奖励；折扣因子 $\gamma$ 用来在「今天的奖励」和「明天的奖励」之间做取舍；因为世界是随机的，**不能优化单条轨迹，只能优化期望回报**。右下角的示意图画的是从 $s_0$ 出发的若干条随机轨迹，分别终止于 $s_{T1}$、$s_{T2}$、$s_{T3}$、$s_{T4}$，被高亮加粗的那条通向 $s_{T1}$。

第二张回顾幻灯片 **Recap: Trajectory Probability**：轨迹 $\tau = (s_0, a_0, s_1, a_1, \dots, s_T)$ 在策略 $\pi$ 下的概率是三项相乘：

$$p_{\pi}(\tau) = \rho(s_0) \prod_{t=0}^{T-1} \pi_{\theta}(a_t \mid s_t)\, P(s_{t+1} \mid s_t, a_t)$$

![轨迹概率回顾：策略下轨迹概率由初始分布、策略和转移函数三部分构成](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0005_00-03-30.jpg)

画面在三个因子下面各画了一个花括号并标注了名字：$\rho(s_0)$ 是 **Initial State Distribution**（初始状态分布，注意是希腊字母 $\rho$ 而不是 $p$，后面推导里同一位置有时写成 $\log p(s_0)$，指的是同一个对象）、中间是 **Policy**、最后是 **Transition Function**。讲师的说法是「三样东西乘在一起：你从哪里出发（初始状态分布）、每一步策略给所取动作分配的概率，再乘以环境的转移概率」。这张图下方还是那棵 $s_0 \to s_{T1..T4}$ 的轨迹树。

## 直接参数化策略后第一件事：这版策略有多好？ [04:20 - 06:07]

幻灯片标题 **How Good is the Policy?**，这是本讲第一个真正的技术问题。

- 令 $\pi_\theta(a|s)$ 为我们优化的策略；
- 目标是

$$\theta^{*} = \arg\max_{\theta} \; \mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[\sum_{t=0}^{T-1} R(s_t, a_t)\right]$$

画面在这个期望下面画了花括号并标注 $J(\theta)$。

- 而这个期望用样本近似：

$$J(\theta) = \mathbb{E}_{\tau \sim p_{\pi}(\tau)}[R(\tau)] \approx \frac{1}{N}\sum_{i} R(\tau_i) = \frac{1}{N}\sum_{i}\sum_{t=0}^{T-1} r_{i,t}$$

![策略评估方法：通过蒙特卡洛采样估计目标函数J(theta)的期望值](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0006_00-04-22.jpg)

画面在下面标注了 **Monte Carlo Sampling: Sum over samples from $\pi_\theta$**（红色小箭头指着 $\frac1N$）。右侧示意图比上一张多了语义：同一个 $s_0$ 分叉出的四条轨迹，终点分别标 **good**（绿色对勾，通向 $s_{T1}$）、**fair**（黄色等号，$s_{T2}$）、**bad**（红色叉，$s_{T3}$）、**bad**（$s_{T4}$）。讲师用它说明直觉：同一个起点，有的轨迹好、有的中等、有的差，而我们希望策略把更多概率质量压到好的那些上。

关键难点被讲师明确点出：**这个期望没法解析计算**，因为我们不知道真实世界里所有机器人运动的真实分布。实践中的做法是蒙特卡洛采样——用当前策略 $\pi_\theta$ 把机器人跑 $N$ 次，收集轨迹、算平均奖励。这里隐含一个「也许并不显然」的假设：**我们采到的所有数据都来自当前这个策略**。这正是 **on-policy** 的含义，也就是说**不能用历史数据**来做这件事。一句话总结这一步：通过多次 rollout 当前策略、观测奖励再取平均，来估计这个依赖策略的期望。

## 策略梯度定理推导（一）：对数导数技巧 [06:07 - 09:36]

能评估了，但真正要的是**改进**策略，于是需要梯度。幻灯片标题换成 **How to get the Gradient?**，先把目标写清楚：

$$\theta^{*} = \arg\max_{\theta}\; \mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[\sum_{t=0}^{T-1} R(s_t, a_t)\right], \qquad R(\tau) = \sum_{t=0}^{T-1} R(s_t, a_t), \qquad J(\theta) = \mathbb{E}_{\tau \sim \mathbb{p}_{\theta}(\tau)}[R(\tau)]$$

![策略梯度优化目标函数定义及轨迹回报公式幻灯片](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0007_00-06-07.jpg)

（画面里 $J(\theta)$ 的期望下标写成了黑板体的 $\mathbb{p}_\theta(\tau)$，与前面几页的 $p_\theta(\tau)$ 是同一个轨迹分布；本讲的幻灯片在这两种字形之间来回切换，读公式时要注意。）右侧还是那棵 good/fair/bad 轨迹树。

问题在于：**不能直接对 $J(\theta)$ 求导**，因为中间夹着一个采样过程，两者的关系是隐式的——我们没法对「机器人在真实世界里运动」求导，那依赖世界的动力学，而我们并不知道它。

讲师的处理是「做一点数学」。先把期望写成积分，再利用积分是线性算子，把梯度搬进积分号内：

$$\nabla_\theta J(\theta) = \int \nabla_\theta p_\theta(\tau)\, R(\tau)\, d\tau$$

![策略梯度定理推导：展示链式法则与对数导数技巧（Log Derivative Trick）的应用](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0008_00-08-44.jpg)

这一页左半边列了两条预备知识，右半边是核心恒等式，画面标注非常明确：

- Chain Rule：$\displaystyle \nabla_\theta f(\mathbf{x}(\theta)) = \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} \nabla_\theta x_i(\theta)$
- $\displaystyle \nabla_\theta \log\big(p_\theta(\tau)\big) = \frac{1}{p_\theta(\tau)} \nabla_\theta p_\theta(\tau)$
- **Log Derivative Trick**：$\displaystyle p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau) = p_\theta(\tau)\, \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} = \nabla_\theta p_\theta(\tau)$，旁边配最简形式 $\nabla \log x = \dfrac{\nabla x}{x}$

页面下方写着 **Policy Gradient Theorem:**，$\nabla_\theta J(\theta) = \int \nabla_\theta p_\theta(\tau) R(\tau)\, d\tau$，其中 $\nabla_\theta p_\theta(\tau)$ 用蓝色高亮，一支红色箭头指着它并标注「**Can't sample this**」——因为这是**密度梯度**而不是密度本身，所以这个积分此刻还不是一个期望，没法靠采样轨迹来估。

下一步就是把蓝色那段换成红色恒等式的左边：

$$\nabla_\theta J(\theta) = \int \nabla_\theta p_\theta(\tau) R(\tau)\, d\tau = \int p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau)\, R(\tau)\, d\tau$$

![策略梯度定理推导：将梯度算子移入积分内部，利用对数导数技巧展开](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0009_00-09-36.jpg)

讲师的说法是：这一步本质上就是**乘了又除同一个 $p_\theta(\tau)$**，等于乘 1。而一旦写成 $\int p_\theta(\tau)(\cdots)d\tau$ 的形式，它就重新变回期望：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\big[\nabla_\theta \log p_\theta(\tau)\, R(\tau)\big]$$

这就有用了，因为**期望可以用样本估**。

课堂上有同学提问（转写里以「you might be saying」的形式复述）：这不还是在采样吗？你几页前刚说不能对采样过程求导。讲师的回答是本场最重要的一句概念澄清：**区别在于 $\theta$ 现在出现在期望里面**。我们依然不能对采样求导，但**可以对 $\theta$ 求导**。他顺带回收了上一讲的伏笔——即使奖励函数本身不可导（例如车开山路、掉下悬崖才给 $-1$ 那种阶跃奖励），**关于 $\theta$ 的期望仍然是平滑的**。同学又追问哪个项可以采样，讲师确认：$R(\tau)$ 这一项可以采样，另一项不能对采样本身求导。

## 策略梯度定理推导（二）：轨迹分布取对数，动力学项消失 [09:36 - 13:50]

现在剩下的麻烦是 $\nabla_\theta \log p_\theta(\tau)$：它还不是 $\pi(a|s)$ 这种我们真正有的函数。幻灯片在上方把轨迹分布重新抄了一遍：

$$\text{Trajectory Distribution:}\quad p_\theta(\tau) = \rho(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t)\, P(s_{t+1}|s_t, a_t)$$

$$\log p_\theta(\tau) = \log p(s_0) + \sum_{t=1}^{T} \log \pi_\theta(a_t|s_t) + \log p(s_{t+1}|s_t, a_t)$$

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\big[\nabla_\theta \underbrace{\log p_\theta(\tau)}_{\text{红色高亮}}\, R(\tau)\big]$$

![策略梯度定理的推导，展示轨迹分布及其对数形式](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0010_00-12-13.jpg)

用到的性质是「**乘积的对数等于对数之和**」，对轨迹分布两侧取对数，就把连乘拆成了三项求和：初始状态项、整条 horizon 上策略对每个动作的对数概率、以及转移概率的对数。

然后把这一坨代回梯度里求导，画面直接给出了被划掉的样子：

$$\nabla_\theta\Big[\underbrace{\log p(s_0)}_{\text{划掉}} + \underbrace{\textstyle\sum_{t=1}^{T} \log \pi_\theta(a_t|s_t)}_{\text{保留}} + \underbrace{\log p(s_{t+1}|s_t,a_t)}_{\text{划掉}}\Big]$$

![策略梯度展开过程，展示环境动态项在求导后消失](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0011_00-13-06.jpg)

**$\log p(s_0)$ 与 $\log p(s_{t+1}|s_t,a_t)$ 都不依赖 $\theta$，求导即为 0，于是被划掉**（画面用两条红色斜线标出）。讲师强调这太关键了：我们本来就没有转移动力学，而它恰好自己消失了。化简后只剩 $\nabla_\theta \log \pi_\theta(a_t|s_t)$ 这些**我们真的能算**的项。

这就得到 **vanilla policy gradient** 的形式：从策略采样轨迹，对每个动作的对数概率求梯度、求和、乘以回报，就拿到一个可以用来把策略推向更高回报的梯度方向。

## REINFORCE：第一个策略梯度算法 [13:50 - 14:50]

完整算法叫 **REINFORCE**，是 Williams 在 1992 年给出的缩写，也是第一个策略梯度方法，常被称作 vanilla policy gradient。幻灯片给出三步伪代码：

1. Initialize $\pi_\theta$
2. While not converged do
   - I. Collect samples $\{\tau^i\}$ by rolling out policy $\pi_\theta$
   - II. Compute gradient $\nabla_\theta J(\theta) \approx \dfrac{1}{N}\sum_{i=0}^{N-1}\Big[\Big(\sum_{t=0}^{T-1}\nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t})\Big)\Big(\sum_{t=0}^{T-1} R(s_{i,t}, a_{i,t})\Big)\Big]$
   - III. Update $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$
3. Return $\pi_\theta$

![REINFORCE算法的伪代码步骤及原始文献引用](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0012_00-13-58.jpg)

画面底部是原始文献：**Simple statistical gradient-following algorithms for connectionist reinforcement learning**，Williams, R. J. (1992)。

注意几个细节：样本写成 $\{\tau^i\}$（上标 $i$），求和下标是 $i=0$ 到 $N-1$，每条轨迹内部的两个求和都从 $t=0$ 到 $T-1$；更新是**梯度上升**（$+\alpha$），因为目标是最大化。讲师说它「quite simple」：初始化策略、用刚推导的公式采样本估计梯度、把梯度作用到权重上，然后**再采一批新数据**重复。$N$ 是样本数，$N$ 越大期望估得越准；理想世界可以有无穷样本、拿到精确的梯度，但机器人上显然做不到。

## 梯度的直觉：按回报加权的模仿 [14:50 - 15:54]

幻灯片 **Intuition Behind Gradient** 把策略梯度与几周前讲过的行为克隆（Behavior Cloning, BC）放在一起对照：

$$\text{Policy Gradient:}\quad \nabla_\theta J_{PG}(\theta) = \frac{1}{N}\sum_{i=0}^{N-1}\left[\left(\sum_{t=0}^{T-1}\nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t})\right)\left(\sum_{t=0}^{T-1} R(s_{i,t}, a_{i,t})\right)\right]$$

$$\text{BC Gradient:}\quad \nabla_\theta J_{BC}(\theta) = \frac{1}{N}\sum_{i=0}^{N-1}\left[\sum_{t=0}^{T-1}\nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t})\right]$$

![完整展示策略梯度PG公式与BC梯度公式的对比，PG公式含奖励求和项](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0014_00-15-43.jpg)

PG 公式里第一个括号下方画了花括号，标注「**Imitation gradient, weighted by reward**」，再用箭头指向两条：绿色对勾 **Upweight good trajectories**、红色叉 **Downweight bad ones**。

讲师的解释：BC 梯度就是「数据集里出现过的动作的对数概率之和」，也就是最大化专家数据里那些动作的似然，**所有动作权重相同**，不管它们最终导致好结果还是坏结果。策略梯度在做**同一件事**，唯一区别是**每条轨迹被它的总回报加权**。于是好的轨迹被推高、坏的轨迹被压低。

## 因果性违反与 Reward-to-Go [15:54 - 18:13]

REINFORCE 有了，接下来看它在真实世界里从哪里开始崩。第一个问题是**因果性**。

例子：机械臂的任务是伸手去抓杯子，但它冲过头（overshoot），最后拿到一个负奖励。REINFORCE 会怎么做？它会**惩罚这条轨迹里的每一个动作**，包括开头那些完全正确的动作。这显然是错的：前半段是好行为，后半段才糟糕。梯度会告诉机器人「永远别往那个方向动」，于是这个行为极难学会，需要海量样本。

讲师把它概括为对因果性原则的违反：**物理世界里，$t$ 时刻采取的动作只可能影响同一时刻或未来的奖励，它不可能对已经发生在过去的奖励负责。**

第一个修正叫 **Reward-to-Go**：不再把 $t$ 时刻动作的梯度乘以「所有奖励之和」，而是只乘以**从 $t$ 到回合结束**的那一段奖励：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\sum_{t=0}^{T-1}\nabla_\theta \log \pi_\theta(a_t|s_t)\left(\sum_{t'=t}^{T-1} R(s_{t'}, a_{t'})\right)\right]$$

讲师强调，这样做**不改变梯度的期望值**，但显著降低了方差，并且正好对上冲过头那个例子。

## 高方差：99 分与 101 分在 REINFORCE 眼里一样好 [18:13 - 19:55]

因果性修好了，但 vanilla REINFORCE 还有第二个病：**高方差**。幻灯片标题就是 **High Variance**：

- 左图 $\tau^1$：**arm moves vaguely toward cup but overshoots**（机械臂大致朝杯子动但冲过头），下方 **R = 99**，旁边是一个红色减号图标；
- 右图 $\tau^2$：**arm reaches cup perfectly**（完美够到杯子），下方 **R = 101**，配一条绿色箭头；
- 结论句：**REINFORCE: Both have great return, lets make both behaviors more likely!**（画面原文如此）
- 更新式：$\Delta\theta \approx \nabla_\theta \log \pi_\theta(a|s)\cdot R(\tau)$

![高方差问题示例：机械臂两种轨迹的回报对比及REINFORCE更新公式](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0015_00-18-20.jpg)

讲师算了一笔账：这两条轨迹之间的**真实信号**是 $101 - 99 = 2$，这才是两种行为在奖励上的实际差异的信息量；但 REINFORCE 计算梯度时是分别乘以 99 和 101，优化器看到的是巨大的梯度更新，而真实信号只占其中的「2% 左右」，剩下的全是噪声，训练因此极不稳定。

一句话概括病症：**当回报既大又彼此接近时，vanilla 形式的 REINFORCE 分不清好轨迹和坏轨迹。**

## 基线：减一个常数，期望不变方差骤降 [19:55 - 21:45]

修法非常朴素：**从 reward-to-go 里减掉一个基线 $b$，把奖励中心化**。这是这一页对公式做的唯一改动，其余全部照旧——现在每个动作的对数概率乘的是「reward-to-go 减 $b$」而不是原始 reward-to-go。幻灯片标题 **Reducing Variances with Baselines**，公式为：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\sum_{t=0}^{T-1}\nabla_\theta \log \pi_\theta(a_t|s_t)\left(\sum_{t'=t}^{T-1} R(s_{t'}, a_{t'}) - b\right)\right]$$

基线常取平均奖励：

$$b = \frac{1}{N}\sum_{i=0}^{N-1} R(\tau_i)$$

![使用基线（Baselines）降低方差的数学推导及无偏性证明](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0016_00-20-05.jpg)

画面把 99/101 的例子直接重算了一遍：**$R = 99 - 100 = -1$**、**$R = 101 - 100 = 1$**，并配文「by substracting average reward, we get negative gradients for below average behavior」（绿色对勾；幻灯片把 subtract 拼成了 substract/substracting）。于是梯度做的正是我们希望它做的：**抑制冲过头的行为、强化完美够取的行为**。

那这算不算往公式里塞一个随机常数、是不是在作弊？画面右下角回答：**Subtracting a baseline does not change the gradient in expectation, it is unbiased.**（绿色对勾）左下「Why can we do this?」下面给出完整的无偏性证明链：

$$\mathbb{E}\big[\nabla_\theta \log p_\theta(\tau)\, b\big] = \int p_\theta(\tau)\nabla_\theta \log p_\theta(\tau)\, b\, d\tau = \int \nabla_\theta p_\theta(\tau)\, b\, d\tau = b\nabla_\theta \int p_\theta(\tau)\, d\tau = b\nabla_\theta 1 = 0$$

也就是说，减常数是一个**零偏差**操作，但对高方差问题的帮助极大。

## 优势函数：把绝对回报变成相对分数 [21:45 - 24:20]

我们已经一路在写「reward-to-go 减基线」，并说 $b$ 可以是平均奖励、一个用来降方差的常数。幻灯片 **Advantage Function** 接着追问：**我们到底在算什么东西？**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\sum_{t=0}^{T-1}\nabla_\theta \log \pi_\theta(a_t|s_t)\left(\underbrace{\sum_{t'=t}^{T-1} R(s_{t'}, a_{t'})}_{\textbf{Reward-To-Go}} - b\right)\right]$$

而这一项正是 $Q$ 函数的定义：

$$Q^{\pi}(s_t, a_t) = \mathbb{E}_{\pi_\theta}\left[\textstyle\sum_{t'=t}^{T-1} R(s_{t'}, a_{t'}) \;\middle|\; s_t, a_t\right]$$

![优势函数（Advantage Function）定义及理想基线V(s)的推导](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0017_00-21-49.jpg)

讲师点破：从当前动作之后所有时刻的奖励求和、再对策略采样的轨迹取期望——如果你还记得，这就是**动作价值函数**：给定处在 $s_t$、取了 $a_t$，之后按策略走的期望未来奖励总和。所以**reward-to-go 其实是 $Q$ 函数的一个蒙特卡洛估计，本质上是一个带噪样本**。我们一直在算的，就是这个含噪的 $\hat Q$ 减去基线。

既然认清了这一点，再配上基线想法，就得到**理想基线**：

$$\text{What could be the ideal baseline?}\qquad V^{\pi}(s) = \mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\big[Q^{\pi}(s, a)\big]$$

画面在 $V^\pi(s)$ 下面画了红色箭头并解释：**Accounts for the specific difficulty of each state, ensuring the robot is only rewarded for actions that outperform its own average expectation in that situation**（考虑每个状态各自的难度，确保机器人只因为「动作胜过它在该情形下的平均预期」而受到奖励）。

于是定义**优势函数**（下一张幻灯片 **Advantage Function**，标题问句是「How much better it is to take a action $a$ in state $s$?」）：

$$A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)$$

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \mathbb{p}_\theta(\tau)}\left[\sum_{t=0}^{T-1}\nabla_\theta \log \pi_\theta(a_t|s_t)\, A^{\pi}(s_t, a_t)\right]$$

![优势函数定义及包含优势项的策略梯度公式推导](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0018_00-23-34.jpg)

画面底部把两个价值函数各自补了一句人话：$Q^{\pi}(s_t,a_t) = \mathbb{E}_{\pi_\theta}\big[\sum_{t'=t}^{T-1} R(s_{t'},a_{t'}) \mid s_t,a_t\big]$ —— **expected return by taking action $a$ from state $s$**；$V^{\pi}(s) = \mathbb{E}_{a\sim\pi_\theta(\cdot|s)}[Q^{\pi}(s,a)]$ —— **expected return from state $s$**。

讲师的解读：$Q^\pi$ 是「取某个具体动作 $a$ 你预期的分数」，$V^\pi$ 是「在 $s_t$ 只按当前策略走、平均预期的分数」，优势回答的是**「从 $s_t$ 出发取 $a_t$ 到底好多少」**。代回策略梯度后：**正优势**意味着这个动作胜过平均预期，于是提高它的概率；**负优势**意味着不如平均，于是降低它。

## 分布偏移与样本效率：on-policy 的墙 [24:20 - 25:54]

现在这是「最新、最精炼」的策略梯度形式：有因果性、有基线，但 REINFORCE 那页就提过的一个实际麻烦还在。幻灯片标题 **Distribution Shift & Sample Efficiency**，公式里的采样分布 $\mathbb{E}_{\tau \sim p(\tau)}$ 被涂成红色并加箭头标注「**assumes samples from current policy**」：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \boldsymbol{p}(\tau)}\left[\sum_{t=0}^{T-1}\nabla_\theta \log \pi_\theta(a_t|s_t)\left(\sum_{t'=t}^{T-1} R(s_{t'}, a_{t'}) - b\right)\right]$$

下面重贴了 REINFORCE 的三步伪代码，在 **III. Update $\theta \leftarrow \theta + \alpha\nabla_\theta J(\theta)$** 旁边用红色箭头加了一句：**We need to throw away old data & collect new data every gradient step!**（配一个惊恐表情）。页面最底部是结论：**Vanilla policy gradient is On-Policy, unlike Off-Policy algorithms like DQN that can reuse past data**。

![分布偏移与样本效率：On-Policy 策略梯度的局限性分析](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0019_00-24-27.jpg)

讲师从机器人视角展开：轨迹只有由**当前**策略采出来才有效，这对机器人极不实用——真机采数据很贵，机器人会坏、电机会坏。采完一批、做一次梯度更新就得把数据扔掉重来，效率极低；在仿真里也许还行，但理想情况是**更省样本，甚至同一批数据上做多次梯度更新**。挑战于是很清楚：**不要扔掉数据，让策略梯度算法变得 off-policy 一点。**

## 重要性采样：用旧分布的样本估新分布的期望 [25:54 - 28:20]

工具是**重要性采样（importance sampling）**：一种「只有另一个分布的样本，却要估计某个分布下期望」的技术。放到我们的场景：想算关于目标分布 $p(x)$ 的期望，但手里只有从 $q(x)$ 采出的样本——$q$ 对应**旧策略**采出来的数据。推导用的正是前面那招「乘 1」：

$$\mathbb{E}_{x \sim p}[f(x)] = \int p(x) f(x)\, dx = \int q(x)\, \frac{p(x)}{q(x)}\, f(x)\, dx = \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right]$$

整理成右边这个形式后，结论是：**用旧样本估新期望，只需把每个函数值 $f(x)$ 乘以比值 $\dfrac{p(x)}{q(x)}$**，即该样本在新策略下的概率除以其在旧策略下的概率。于是机器人可以复用旧样本，前提是按这个比值加权。

讲师补了两个重要条件：

1. **只有当 $q$ 的支撑集非零时这套推导才成立**，否则要保持无偏就做不到——分母为 0 时那个除法根本没有定义；
2. 直觉上，**两个分布差别不大时重要性采样效果最好**。他指着幻灯片上画的例子说，画面里那种「非常非常不同」的分布就是坏情况。

## Off-Policy 策略梯度（一）：目标函数与轨迹比值 [28:20 - 30:33]

幻灯片标题 **Policy Gradients Off-Policy Version**，第一行小字「RL objective with importance sampling」：

$$J(\theta) = \mathbb{E}_{\tau \sim \bar{p}(\tau)}\left[\frac{p_\theta(\tau)}{\bar{p}(\tau)}\, r(\tau)\right]$$

$$\frac{p_\theta(\tau)}{\bar{p}(\tau)} = \frac{p(s_1) \prod_{t=1}^{T} \pi_\theta(a_t|s_t)\, p(s_{t+1}|s_t, a_t)}{p(s_1) \prod_{t=1}^{T} \bar{\pi}(a_t|s_t)\, p(s_{t+1}|s_t, a_t)}$$

![Off-Policy策略梯度中基于重要性采样的强化学习目标函数及概率比展开式](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0020_00-28-49.jpg)

注意画面上的写法：分母分布是带横线的 $\bar{p}(\tau)$、旧策略是 $\bar{\pi}$，初始状态项写的是 $p(s_1)$（与前面几页的 $\rho(s_0)$/$\log p(s_0)$ 记号并不完全统一，但指同一项），乘积下标从 $t=1$ 到 $T$。做法是把轨迹分布公式按链式法则分别代入新策略与旧策略，然后**因为两个策略处在同一个 MDP 里，初始状态分布与转移概率完全相同，可以直接约掉**。约完之后剩下的就是策略概率之比的连乘——这非常有用，因为我们本来就不掌握世界动力学。

![简化后的重要性采样权重乘积形式及新策略梯度更新公式推导](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0021_00-29-41.jpg)

下一张把约分过程画在页面上：$p(s_1)$ 与两个 $p(s_{t+1}|s_t,a_t)$ 被红色斜线划掉，右侧留下干净的结果：

$$\frac{p_\theta(\tau)}{\bar{p}(\tau)} = \prod_{t=1}^{T} \frac{\pi_\theta(\mathbf{a_t}|\mathbf{s_t})}{\bar{\pi}(\mathbf{a_t}|\mathbf{s_t})}$$

然后页面抛出下一个问题（幻灯片原文）：**How do derive the gradient to update our new policy $\pi_{\theta'}$ with samples from old $\pi_\theta$?** 此刻写出的梯度仍然是「错」的那一版——期望下标还是新策略：

$$\nabla_{\theta'} J(\theta') = \mathbb{E}_{\tau \sim \boldsymbol{p_{\theta'}(\tau)}}\left[\left(\sum_{t=1}^{T}\nabla_{\theta'} \log \pi_{\theta'}(a_t|s_t)\right)\left(\left(\sum_{t'=t}^{T} r(s_{t'}, a_{t'})\right) - b\right)\right]$$

画面把 $p_{\theta'}(\tau)$ 涂红，箭头标注 **new policy**——问题就是：我们手里只有旧策略的数据，却在对新策略的分布取期望。

## Off-Policy 策略梯度（二）：把期望换回旧策略 [30:33 - 31:26]

解法就是套用两页前的重要性采样推导，把那些项插进公式里，关键变化是**期望的下标换回旧策略**——那才是我们真正拥有的数据。幻灯片 **Policy Gradients Off-Policy Version**（页码 19）给出：

$$\nabla_{\theta'} J(\theta') = \mathbb{E}_{\tau \sim \boldsymbol{p_\theta(\tau)}}\left[\frac{\boldsymbol{p_{\theta'}(\tau)}}{\boldsymbol{p_\theta(\tau)}}\left(\sum_{t=1}^{T}\nabla_{\theta'} \log \pi_{\theta'}(a_t|s_t)\right)\left(\left(\sum_{t'=t}^{T} r(s_{t'}, a_{t'})\right) - b\right)\right]$$

$$= \mathbb{E}_{\tau \sim \mathbb{p}_\theta(\tau)}\left[\left(\prod_{t=1}^{T} \frac{\pi_{\theta'}(\mathbf{a_t}|\mathbf{s_t})}{\pi_\theta(\mathbf{a_t}|\mathbf{s_t})}\right)\left(\sum_{t=1}^{T}\nabla_{\theta'} \log \pi_{\theta'}(a_t|s_t)\right)\left(\left(\sum_{t'=t}^{T} r(s_{t'}, a_{t'})\right) - b\right)\right]$$

![Off-Policy策略梯度的最终推导结果，标注了旧策略采样与重要性权重项](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0022_00-30-33.jpg)

画面在两个红色项下面分别标了 **old policy**（指采样分布 $\mathbb{E}_{\tau\sim p_\theta(\tau)}$）和 **importance weights**（指比值 $\frac{p_{\theta'}(\tau)}{p_\theta(\tau)}$）。这就是 off-policy 算法的核心实现公式：**从旧策略采样、用新旧轨迹概率之比加权**。

但讲师马上指出这个整轨迹比值很麻烦：**它关于 horizon $T$ 是指数级的**。想象每个重要性项都小于 1，$T$ 很长时你把许多小于 1 的数连乘，乘积会**以指数速度冲向 0**，方差随之**指数级冲向无穷**；反过来也可能爆炸。这一项在实践中会造成不稳定与梯度爆炸。

## 从轨迹到时步：状态比值近似为 1 [31:26 - 33:10]

于是实践中的标准做法是：**对时间步取期望，而不是对轨迹取期望**（幻灯片 **Policy Gradients Off-Policy Version**，页码 20）：

$$\nabla_{\theta'} J(\theta') \approx \frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T} \frac{\pi_{\theta'}(s_{i,t}, a_{i,t})}{\pi_\theta(s_{i,t}, a_{i,t})}\, \nabla_{\theta'} \log \pi_{\theta'}(a_{i,t}|s_{i,t})\left(\left(\sum_{t'=t}^{T} r(s_{i,t'}, a_{i,t'})\right) - b\right)$$

![Off-Policy策略梯度公式推导，展示重要性采样比值的近似处理及最终形式](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0023_00-31-26.jpg)

红色高亮的联合比值 $\frac{\pi_{\theta'}(s_{i,t},a_{i,t})}{\pi_{\theta}(s_{i,t},a_{i,t})}$ 下面标注「**more stable to compute, but hard to measure**」。它之所以难测，是因为把联合分布拆开：

$$\frac{\pi_{\theta'}(s,a)}{\pi_\theta(s,a)} = \frac{\pi_{\theta'}(a|s)}{\pi_\theta(a|s)} \cdot \frac{\pi_{\theta'}(s)}{\pi_\theta(s)} \approx \frac{\pi_{\theta'}(a|s)}{\pi_\theta(a|s)}$$

动作比值可以直接从网络算出来，**状态比值却需要知道环境的全局动力学，而我们没有**。画面用红色箭头指着状态比值那一项写「**Often approximate with 1**」——即直接丢掉。讲师说，这正是一系列算法（他点名 TRPO 与 PPO）**通过强制更新幅度很小**来让这个近似成立的动机。

丢掉之后就是页面底部标着 **Final form:** 的最终形式：

$$\nabla_{\theta'} J(\theta') \approx \frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T} \frac{\pi_{\theta'}(a_{i,t}|s_{i,t})}{\pi_\theta(a_{i,t}|s_{i,t})}\, \nabla_{\theta'} \log \pi_{\theta'}(a_{i,t}|s_{i,t})\left(\left(\sum_{t'=t}^{T} r(s_{i,t'}, a_{i,t'})\right) - b\right)$$

重要性权重**退化成一个纯粹的动作概率比值**。讲师再次强调：这仍然只是一个近似，**只有当两个策略足够接近时才管用**。

## Off-Policy 策略梯度算法：一批数据做多次更新 [33:10 - 34:20]

幻灯片 **Off-Policy Policy Gradients Algorithm** 把上面这套整理成三步流程，顶部一行蓝字写着要点：**Take multiple gradient steps on the same batch**。

- I. Collect samples $\{\tau^i\}$ by rolling out policy $\pi_\theta$
- II. Compute gradient using $\{\tau^i\}$ samples $\nabla_{\theta'} J(\theta') \approx \dfrac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T} \dfrac{\pi_{\theta'}(a_{i,t}|s_{i,t})}{\pi_\theta(a_{i,t}|s_{i,t})}\nabla_{\theta'} \log \pi_{\theta'}(a_{i,t}|s_{i,t})\left(\left(\sum_{t'=t}^{T} r(s_{i,t'}, a_{i,t'})\right) - b\right)$
- III. Update $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

![Off-Policy策略梯度算法的三步循环流程：采样、计算梯度、更新参数](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0024_00-33-10.jpg)

画面左侧画了两条回旋箭头：一条深红色的从 I 绕到 III（外层：采一次数据、跑完整个更新循环），一条蓝色的在 II 与 III 之间绕圈（内层：**同一批数据上重复「算梯度—更新参数」**）。讲师说它与 REINFORCE 差别不大，**关键区别就是：以前是「采一次、更新一次」，现在可以「采一次、做多次梯度更新」**。当然不能无限做下去——如果在重新采数据之前做了太多次梯度更新，两个分布会被推得太远，比值就不再成立。

## 熵正则化：别让支撑集先塌掉 [34:20 - 35:30]

重要性采样在**支撑集为零**时会崩：当新策略 $\pi_{\theta'}$ 给某些「旧策略 $\pi_\theta$ 从未取过的状态-动作对」分配了概率，分母为 0，权重直接爆掉。幻灯片 **Entropy Regularization** 把这一情形写成一句问话：**What if the support set is zero, $p(x) > 0$ but $q(x) = 0$?**，然后两条说明：**New policy $\pi_{\theta'}$ assigns probability to action-state pairs that $\pi_\theta$ never took.**、红叉 **Importance weight explodes, estimator gets biased!**

![熵正则化幻灯片，展示离散与连续策略的香农熵公式及温度参数β](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0025_00-34-55.jpg)

对策（画面加粗的 **Solution**）：**Add entropy bonus to avoid policy becoming too deterministic**——不要让旧策略变得太确定，以致给动作空间的某些区域分配零概率，从而**从源头上降低撞上零支撑的概率**：

$$L(\theta') = L_{PG}(\theta') + \beta\, \mathbb{H}(\pi_{\theta'}(\cdot|s))$$

画面用红色箭头指出 $\beta$ 是 **Temperature parameter**（温度参数）：它越大，策略越「摊开」；越接近 0，策略越贪心。熵的具体形式取决于动作空间：

- **Discrete Shannon Entropy**：$\displaystyle \mathbb{H}(\pi_{\theta'}) = -\sum_{a \in A} \pi_{\theta'}(a \mid s) \log \pi_{\theta'}(a \mid s)$
- **Continuous Differential Entropy**：$\displaystyle \mathbb{H}(\pi_{\theta'}) = \frac{d}{2}\big(1 + \log(2\pi)\big) + \sum_{j=1}^{d} \log \sigma_j$

讲师说明：离散动作就用标准香农熵，连续动作对高斯策略用微分熵；其中 $d$ 是动作空间维度，「比如 7 自由度机械臂，那 $d$ 就是 7」。他随后以时间为由没有继续展开。

## 约束策略的变化幅度：KL 约束 [35:30 - 37:47]

幻灯片 **Policy Gradients with Constraints** 问的是另一件事：**What if our policy changes a lot before sampling new data?** 页面顶部重贴了时步级的重要性采样梯度：

$$\nabla_{\theta'} J(\theta') \approx \frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T} \frac{\pi_{\theta'}(a_{i,t}|s_{i,t})}{\pi_\theta(a_{i,t}|s_{i,t})}\, \nabla_{\theta'} \log \pi_{\theta'}(a_{i,t}|s_{i,t})\left(\left(\sum_{t'=t}^{T} r(s_{i,t'}, a_{i,t'})\right) - b\right)$$

![带约束的策略梯度幻灯片，展示KL散度约束及重要性权重爆炸示例](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0026_00-35-48.jpg)

红色箭头指着比值项，下面是一个具体的失效模式（画面原文）：**Unlikely action under $\pi_\theta$ gets high reward, $\pi_{\theta'}$ makes it much more likely and gradient explodes!** 并给出数值算例：

$$w = \frac{0.10\ (\text{new})}{0.005\ (\text{old})} = 20$$

设想某个动作在旧策略下很不被看好，但它拿到了高奖励，于是新策略给它更高的概率——重要性权重一下子变成 20 这种量级，梯度估计随之爆炸。讲师说「这会很糟」。

一种解法是**约束策略，让它在那些多次梯度步中不要变化太大**，具体做法是加一个 **KL 散度约束**（KL 散度度量两个概率分布有多接近）：

$$\mathbb{E}_{s \sim \pi_\theta}\big[D_{KL}\big(\pi_\theta(\cdot|s)\,\|\,\pi_{\theta'}(\cdot|s)\big)\big] \le \epsilon$$

它的作用是：**惩罚新策略给「旧策略曾经取过的动作」分配零概率**——因为旧策略的数据就是我们的 ground truth 数据，它保证了新策略的动作有支撑可用。

## TRPO：理论保证与二阶代价 [37:47 - 40:05]

这套约束被 **Trust Region Policy Optimization（TRPO）** 在 2015 年具体实现。幻灯片标题 **Trust Region Policy Optimization (TRPO)**，一句话概括：**Maximize advantage from old policy while not deviating too much**：

$$\text{maximize}_{\theta'} \quad L(\theta') = \mathbb{E}_{s,a \sim \pi_\theta}\left[\frac{\pi_{\theta'}(a|s)}{\pi_\theta(a|s)}\, A^{\pi_\theta}(s,a)\right] \qquad \text{subject to}\quad \mathbb{E}_{s \sim \pi_\theta}\big[D_{KL}(\pi_\theta(\cdot|s)\,\|\,\pi_{\theta'}(\cdot|s))\big] \le \epsilon$$

![信任域策略优化(TRPO)幻灯片，列出其单调改进保证及计算复杂度缺点](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0027_00-38-25.jpg)

画面用图标列了四条优缺点，值得逐条抄准：

- ✅ **Monotonic improvement guarantee**：提供理论保证，每一次更新都不会让真实性能 $J^\pi$ 变差（讲师说「each update will improve or not worsen the true performance」）；
- ✗ **Requires the Fisher Information Matrix to solve the constraint optimization, is $O(N^3)$ number of trainable parameters**：这个 KL 约束要求计算 Fisher 信息矩阵——关于策略参数的 KL 散度的二阶导数矩阵，计算量极大，**内存 $O(n^2)$、计算 $O(n^3)$**（口头补充的内存代价，画面只标了 $O(N^3)$）；
- ⚠️ **Conjugate gradient approximation reduces this to $O(kn) \approx O(n)$, but still ~20× more expensive than a standard gradient step**：共轭梯度近似可以把复杂度降下来，但仍然比一次标准梯度步贵大约 20 倍；
- ✗ **Scaling to deep networks requires additional approximations like K-FAC (Kronecker-Factored Approximate Curvature)**：想推到深度网络还需要额外近似。

页面底部文献：**Trust region policy optimization**，Schulman, Levine, et al., (2015)。

讲师的结论是「因此它现在没那么流行了（not that popular nowadays）」，但紧接着放了 ETH 自己的一个很有说服力的机器人应用：用 TRPO 训练**四足机器人运动**，走 sim-to-real 路线，画面里能看到不错的 zero-shot 泛化——机器人能穿越困难地形、跨过障碍、在训练时没见过的表面上继续行走。「so, like it actually works.」

## PPO：用 clip 换掉二阶数学 [40:05 - 42:05]

TRPO 的保证很好，但我们受不了它的计算代价。于是问题变成：**能不能不要那套昂贵的二阶数学，也拿到同样的稳定性？** 答案就是更流行、也更好实现的 **Proximal Policy Optimization（PPO）**。幻灯片标题 **Proximal Policy Optimization (PPO)**，核心问句（画面原文，ration 是 ratio 的笔误）：**What if just clip importance ration instead of expensive constraints to avoid getting too large?**

$$L(\theta') = \mathbb{E}_{s,a \sim \pi_\theta}\left[\min\left(\frac{\pi_{\theta'}(a|s)}{\pi_\theta(a|s)}\, A^{\pi_\theta}(s,a),\;\; \text{clip}\left(\frac{\pi_{\theta'}(a|s)}{\pi_\theta(a|s)},\, 1-\epsilon,\, 1+\epsilon\right) A^{\pi_\theta}(s,a)\right)\right]$$

![PPO（近端策略优化）算法的核心损失函数公式，展示了通过裁剪重要性采样比率来限制策略更新幅度的机制](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0028_00-40-06.jpg)

整个 $\min(\cdots,\cdots)$ 下方画了花括号，标注 **Force the importance sampling ratio to stay within a safe interval**：把重要性比值强行夹在 $[1-\epsilon,\,1+\epsilon]$ 这个安全区间里，再取两项的**最小值**（取 min 意味着「只在有收益时才限制」的保守下界语义）。

讲师说，这样一来它就是一个**简单的一阶方法**：线性扩展、和 Adam 这类标准优化器配合得很好。而正是这种可扩展性，让 PPO 成为今天用 RLHF、GRPO 等大模型训练方法的**骨架**。

同一张幻灯片随后展开了优缺点清单（画面用 ✅/✗ 图标）：

- ✅ **Scales linearly $O(N)$ with the number of trainable parameters**
- ✅ **Significantly easier to implement, works with standard optimizers like Adam**
- ✅ **Foundation to train LLMs with RLHF or GRPO**
- ✗ **No monotonic improvement guarantee**
- ✗ **Near-on-policy, needs many samples**

![PPO核心损失函数公式，展示了对重要性采样比率进行clip操作的机制及其优缺点列表](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0029_00-41-54.jpg)

底部文献：**Proximal Policy Optimization Algorithms**，Schulman, et al., (2017)。

讲师把权衡讲得很直白：我们在这里做的交换是**不再有单调改进保证**，而且它**仍然是 near-on-policy** 的，学 PPO 还是需要相当多的样本；换来的是**跨很多领域和任务都非常可靠、非常稳定**。

## 机器人案例：OpenAI 的 Rubik's Cube 灵巧手 [42:05 - 43:20]

讲师放了一段视频（此段无关键帧，只能按转写记录）：机器人领域一个著名的 PPO 应用是 **OpenAI 2019** 的工作，用一只 **24 自由度的 Shadow Hand 灵巧手**去**解魔方**。做法仍然是 sim-to-real：在仿真里做大量 **domain randomization**，然后迁移到真机；画面里可以看到它在各种扰动下都能工作，甚至手指被绑住的情况下也能继续。

他特别指出一个架构细节，用来纠正一个常见误解：**这里的 PPO 并没有把相机图像直接当状态输入**，而是采用更模块化的方案——**先有一个感知网络提取魔方（cube）的位置与朝向**，再把这个结构化状态喂给 PPO。这一点对「策略学习 vs 表征学习」的分工很有指示意义。

## 完整 PPO 目标：三项合流与 Actor-Critic 架构 [43:20 - 45:20]

幻灯片标题 **Full PPO Objective**，讲师说「今天讲过的很多东西都汇进了这一条公式」：

$$L^{PPO}(\theta') = \mathbb{E}_{s,a \sim \pi_\theta}\Big[\underbrace{L^{CLIP}(\theta')}_{\text{Clipped Surrogate Objective}} - \underbrace{c_1\big(V_{\theta'}(s) - V^{targ}\big)^2}_{\text{For Advantage}} + \underbrace{c_2\, H(\pi_{\theta'})(s)}_{\text{Entropy Bonus}}\Big]$$

![完整的PPO目标函数公式，分解为裁剪代理目标、优势估计和熵奖励三个部分](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0030_00-43-39.jpg)

画面下方一个大花括号把整条式子括起来，配 ✅：**Combines the best of value-based methods and policy gradients!**

三项的角色（讲师逐一说明）：

1. **Clipped Surrogate Objective**（裁剪代理目标）：就是刚才推导的 PPO 损失，它是 **actor 损失**，更新策略参数，在把重要性比值保持在安全范围内的前提下最大化优势；
2. **Value Function Loss**（$-c_1(V_{\theta'}(s)-V^{targ})^2$）：这是 **critic**，与 actor 同时训练，目的是给出更好的**优势函数估计**。讲师提醒：优势函数需要 $V^\pi$、也就是状态价值，**critic 越好，优势估计的方差就越低，策略也就越稳**；
3. **Entropy Bonus**（$+c_2 H(\pi_{\theta'})(s)$）：前面讨论过的熵奖励，防止策略塌缩成确定性解、维持探索。

于是这就是所谓的 **Actor-Critic 架构**：策略梯度方法给我们**直接优化真正关心的行为**，基于价值的方法给我们**低方差的基线**。

## Actor-Critic 方法族与特权信息 [45:20 - 46:54]

幻灯片标题 **Actor Critic Methods**，把这套分工列成清单：

- **Actor**: uses Policy Gradients to predict continuous actions
- **Critic**: uses Value learning to evaluate goodness of a state
- **Actor-Critic**:
  - ✅ Low variance from critic, continuous actions from actor, direct policy optimization
  - Learns from every step via TD updates, at the cost of small bootstrapping bias
  - Optimal baseline to lower variance via advantage
  - Robotics: often asymmetric information via privileged info for critic
  - ✗ Actor and Critic still near-on-policy, needs many samples

![Actor Critic Methods 幻灯片，列出 Actor、Critic 及 Actor-Critic 组合的优缺点与适用场景](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0031_00-45-24.jpg)

讲师强调两个对机器人尤其重要的点。第一，**critic 通过 TD 更新学习**：它从自己的下一步估计自举，而不是等整条轨迹的回报跑完——而「必须等回合结束」恰恰是只用策略梯度时的限制。因此相比蒙特卡洛回报，这**大大提升了样本效率**，代价是一点点 **bootstrapping bias**。

第二，**机器人里常用不对称信息（asymmetric information）**：因为 **测试时不用 critic**，所以在仿真环境里可以给 critic 额外喂**特权信息（privileged information）**去降方差、拿到更好的优势估计；到 test time 把 critic 扔掉即可，「so it doesn't really matter」，但训练时帮助很大。

## Soft Actor-Critic：完全 off-policy 的收束 [46:54 - 50:20]

讲师说时间不多，快速过最后一个算法家族 **Soft Actor-Critic（SAC）**，并解释它为什么正好「闭环」：我们从非常 on-policy 的方法出发，一路想变得更 off-policy，以便更省样本、更多利用历史数据——**SAC 就是这个回答**：一个**完全 off-policy 的 actor-critic 算法**，critic 在**经验回放缓冲区（replay buffer）**上训练，就像前一次课讲的 DQN 那样。幻灯片标题 **Soft Actor-Critic**，第一条就是：**Critic trained fully off-policy on a replay buffer!**

$$L^{Actor}(\theta') = \mathbb{E}_{s \sim \mathbb{D}_{\text{replay}},\, a \sim \pi_{\theta'}}\big[\alpha\, H(\pi_{\theta'})(s) - Q_\phi(s, a)\big]$$

$$L^{Critic}(\phi) = \mathbb{E}_{\tau \sim \mathbb{D}_{\text{replay}}}\left[\left(Q_\phi(s,a) - \left(r + \gamma\left(Q_{\bar\phi}(s',a') + \alpha\, H(\pi_{\theta'})(s')\right)\right)\right)^2\right]$$

![Soft Actor-Critic 的 Actor 与 Critic 损失函数公式，并标注探索熵奖励、利用项及目标网络等含义](/coursenotes/eth-robot-learning-2026/l05-reinforcement-learning-ii/kf_0032_00-47-08.jpg)

画面上用红色箭头给每一项贴了标签，这套标注比公式本身更能说明设计意图：

- actor 里的 $\alpha H(\pi_{\theta'})(s)$ —— **Exploration: Entropy Bonus**；
- actor 里的 $-Q_\phi(s,a)$ —— **Exploitation: if I were to do this action today, how many points would I get?**（如果我今天做这个动作，能拿多少分？）
- critic 里的 $Q_\phi(s,a)$ —— **How many points did I think this past action $a$ in this past situation $s$ was worth?**
- critic 里的 $r$ —— **actual reward received**
- $Q_{\bar\phi}$ —— **target network**（延迟目标网络，与 DQN 同源）
- $s'$ 项里的 $\alpha H(\pi_{\theta'})(s')$ —— **this state leads to high reward AND gives you many different ways to succeed**

讲师的解读：actor 有两项，探索奖励与利用项，两者之间是一种**推拉关系**；critic 损失就是标准的 **TD 误差**——critic 当前认为某个过去状态-动作对值多少，与「实际收到的奖励 + 下一状态的折扣价值」之间的平方差。关键在于下一状态的价值里**也带熵项**：我们既希望推向高奖励的状态，也希望这些状态**给你很多种达成目标的方式**。

然后是这套方法的核心思想：**actor 从 replay buffer 里把所有状态拉出来，但问的是当前策略（而不是旧策略）今天会取什么动作**。于是 **critic 充当一个「学出来的仿真器」**：去估计「这个新动作在那个旧状态里本会拿到多少奖励」——尽管这个状态-动作组合它其实从未真正见过。

剩下的最后一个技术问题（讲师说明时间不够，只点到为止）：**如何对采样过程做反向传播**。标准做法是对高斯分布用**重参数化技巧（reparameterization trick）**——这个技巧在变分推断和 VAE 里同样常用（与下一讲生成模型相衔接）。他还提到，SAC 原论文里有真机演示，包括堆叠类任务、四足操作以及灵巧手（转写此处含糊，具体任务名不做还原）。

## 本讲总结：一条不断「松绑」的推导链 [50:20 - 52:22]

讲师用最后约一分钟把整讲串成一句话一句话的链条：

- 从 **REINFORCE** 开始，它是优化策略最直接的方式；
- vanilla 版本有局限，于是加上**因果性（reward-to-go）**和**基线**来对抗高方差；
- 再引入**优势函数**，让奖励彼此**相对**而不是绝对；
- 撞上 **on-policy 墙**：每个梯度步都要重采新数据；
- 引入**重要性采样**，允许在**单批数据上做多次梯度步**；
- 用**熵正则化**保证机器人永不停止探索，同时保住重要性采样所需的**非零支撑**；
- 用不同方式约束新旧策略之间的变化幅度：先是 **KL 散度**（TRPO），然后是 PPO 的**简化裁剪**，限制单次更新里策略能漂移多远；
- 最后到达 **Soft Actor-Critic**：critic 完全 off-policy、在 replay buffer 上训练，和上一讲的 DQN 同源。

他给出的收尾判断是：SAC 在**今天讲过的所有方法里样本效率最高**，这也正是它**至今仍是真实机器人里非常流行的算法**的原因。「Yes, with this thank you for your attention.」（掌声，52:20）

## 要点回顾

- **范式切换**：不再学价值再指望 argmax 抽策略，而是直接参数化 $\pi_\theta(a|s)$ 并对它做梯度上升。上一讲 deep RL 的所有创新本质都是「用不同方式解那个 argmax」。
- **策略梯度定理**：$\nabla_\theta J(\theta) = \int \nabla_\theta p_\theta(\tau)R(\tau)d\tau$ 里的密度梯度不能采样；用对数导数技巧 $p_\theta\nabla_\theta\log p_\theta = \nabla_\theta p_\theta$（乘 1）改写成 $\mathbb{E}_{\tau\sim p_\theta(\tau)}[\nabla_\theta\log p_\theta(\tau)R(\tau)]$ 就能采样了。$\theta$ 出现在期望**内部**是关键。
- **模型无关从何而来**：$\log p_\theta(\tau) = \log p(s_0) + \sum_{t=1}^T \log\pi_\theta(a_t|s_t) + \log p(s_{t+1}|s_t,a_t)$，含动力学与初始分布的两项对 $\theta$ 求导为 0，被划掉，只剩策略项。
- **REINFORCE**（Williams 1992）：初始化 → 采 $N$ 条轨迹 → 用「动作对数概率之和 × 总回报」估梯度 → $\theta \leftarrow \theta + \alpha\nabla_\theta J(\theta)$ → 再采新数据。
- **梯度直觉**：PG 梯度 = BC 梯度（模仿梯度）再按回报加权；BC 对所有动作一视同仁，PG 抬高好轨迹、压低坏轨迹。
- **因果性**：$t$ 时刻的动作不能为过去的奖励负责 → 用 **reward-to-go** $\sum_{t'=t}^{T-1}R(s_{t'},a_{t'})$ 替换整条回报，不改期望但降方差。
- **高方差与基线**：$R=99$ 与 $R=101$ 的真实信号只有 2，却被 99/101 缩放；减去 $b=\frac1N\sum_i R(\tau_i)$ 后变成 $-1$ 与 $+1$。无偏性由 $b\nabla_\theta\int p_\theta(\tau)d\tau = b\nabla_\theta 1 = 0$ 保证。
- **优势函数**：reward-to-go 是 $Q^\pi$ 的蒙特卡洛单样本估计；理想基线是 $V^\pi(s)=\mathbb{E}_{a\sim\pi_\theta(\cdot|s)}[Q^\pi(s,a)]$；$A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$，正优势升概率、负优势降概率。
- **on-policy 墙**：数据只对当前策略有效，每步都要扔数据重采，真机上极贵（会坏机器人、坏电机）。
- **重要性采样**：$\mathbb{E}_{x\sim p}[f]=\mathbb{E}_{x\sim q}[\frac{p(x)}{q(x)}f(x)]$，要求 $q$ 支撑非零，且两分布不能太远。轨迹比值里初始分布与转移项在同一个 MDP 下约掉，只剩 $\prod_t \frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)}$。
- **指数爆炸与时步近似**：整轨迹比值关于 $T$ 指数、方差指数爆炸；改为对时步取期望，状态比值无法测量、**常近似为 1**，这正是 TRPO/PPO「强制小更新」的意义。
- **熵正则化**：$L(\theta')=L_{PG}(\theta')+\beta\mathbb{H}(\pi_{\theta'}(\cdot|s))$；离散用香农熵、连续高斯用 $\frac{d}{2}(1+\log 2\pi)+\sum_j\log\sigma_j$，$d$ 为动作维度（7 自由度即 7）。
- **失效数值例**：$w=\frac{0.10}{0.005}=20$，旧策略下罕见但高奖励的动作会让梯度爆炸 → 加 KL 约束 $\mathbb{E}_{s\sim\pi_\theta}[D_{KL}(\pi_\theta\|\pi_{\theta'})]\le\epsilon$。
- **TRPO**（Schulman, Levine, et al., 2015）：✅ 单调改进保证；✗ 需要 Fisher 信息矩阵，$O(N^3)$、内存 $O(n^2)$；⚠️ 共轭梯度降到 $O(kn)\approx O(n)$ 但仍约 20× 贵；✗ 深度网络还需 K-FAC。ETH 的四足 sim-to-real 是漂亮应用。
- **PPO**（Schulman, et al., 2017）：把比值 clip 到 $[1-\epsilon,1+\epsilon]$ 并取 min；✅ 线性 $O(N)$、易实现、兼容 Adam、是 RLHF/GRPO 的基础；✗ 无单调改进保证、仍 near-on-policy 需多样本。
- **OpenAI 魔方（2019）**：24 自由度 Shadow Hand，大量 domain randomization 做 sim-to-real；PPO 不直接吃相机图像，而是由感知网络先给出魔方的位置与朝向。
- **完整 PPO 目标**：$L^{PPO}=L^{CLIP}-c_1(V_{\theta'}(s)-V^{targ})^2+c_2H(\pi_{\theta'})(s)$，即「裁剪代理 + critic 价值损失 + 熵奖励」，critic 越好优势方差越低。
- **Actor-Critic**：actor 用策略梯度出连续动作，critic 用价值学习评估状态；TD 更新让每一步都产生学习信号（代价是轻微 bootstrapping bias）；机器人可给 critic 喂**特权信息**，test time 丢弃。
- **SAC**：完全 off-policy，critic 在 replay buffer 上带目标网络训练；actor 损失含探索熵与利用项；「critic 当学出来的仿真器」——评估当前策略在旧状态下的新动作；采样反传用重参数化技巧；样本效率本讲最高，至今真机常用。
