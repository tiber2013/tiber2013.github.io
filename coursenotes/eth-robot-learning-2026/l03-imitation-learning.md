---
title: "第 3 讲 · 模仿学习"
description: "本讲把上一讲建立的「观测→动作」映射第一次真正训练起来，走的是最简单也最常用的路线——模仿学习：给定专家轨迹数据集，学一个策略去「模仿」专家行为，而最直接的实现就是行为克隆（Behavioral Cloning，幻灯片用…"
---

# 第 3 讲 · 模仿学习

> **ETH Zürich 机器人学习 · L03** ｜ 讲者 Oier Mees ｜ 51 分钟 ｜ [原视频](https://www.youtube.com/watch?v=Ef4R5s1LqoQ)

## 核心结论

本讲把上一讲建立的「观测→动作」映射第一次真正训练起来，走的是最简单也最常用的路线——模仿学习：给定专家轨迹数据集$\mathcal{D}=\{(s_1,a_1,\dots,s_t)\}$，学一个策略$\pi_\theta$去「模仿」专家行为，而最直接的实现就是行为克隆（Behavioral Cloning，幻灯片用的正是这个英式拼写），它在数学上等价于对专家动作做最小二乘回归。但行为克隆破坏了监督学习的IID假设：策略一旦在某一步犯错，就会进入训练集里没有的状态，误差因此不按$\epsilon T$线性增长，而是按$O(\epsilon T^{2})$二次爆炸——这就是分布偏移带来的复合误差（compounding error）。修补它有两条路：DAgger在训练时让策略自己rollout、把专家对这些状态的纠正动作聚合回数据集，把上界拉回$O(\epsilon T)$；Human-Gated DAgger则让人类只在策略出错的那一刻接管并补一段局部演示，代价是把「何时该干预」变成一个新的自动检测问题（本讲用语言歧义澄清作为案例回答）。最后，即便数据完美，多模态行为仍会让MSE「把绕树的左右两条路线平均成直直撞树」，于是本讲给出三件工具：高斯混合、动作离散化（含逐维自回归离散化，即早期VLA的做法）与生成式建模（扩散、隐变量策略），并把隐变量策略与目标条件、无结构玩耍数据拼成一个完全自监督的任务无关控制方案。

## 开场与课程事务：论文减量、作业截止与AI检测 [00:00 - 02:08]

![课程标题页：Robot Learning Lecture 3 Imitation Learning，讲师 Oier Mees](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0001_00-00-00.jpg)

标题页写得很清楚：课程是*Robot Learning: From Fundamentals to Foundation Models*，第三讲Imitation Learning，讲师Oier Mees，署名机构是ETH Zürich与Microsoft两家（两个logo并排在右下角），日期02.03.2026。也就是说本讲上课时间是2026年3月2日，这一点后面算作业截止日期时要用到。

正式内容之前先处理上周收到的反馈，幻灯片第1页标题是「Feedback → Actions」（中间是一个红色箭头），四条行动项逐条对应：

![课堂反馈与行动项：论文数量调整、作业截止日期及 AI 使用检测说明](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0002_00-00-51.jpg)

- **Moved from 3 papers to 2 papers per lecture**：每周一的论文报告从3篇减到2篇。
- **Extra paper slots on Thursdays in IFW A 32.1**：多出来的论文时段挪到周四，地点在IFW A 32.1教室。讲师补充说，之所以之前一直没发安排邮件，是在等注册名单稳定下来，看谁继续修课谁退出，之后Alexey会跟进发邮件。
- **HW 2 released, due on March 12th**：第二次作业上周已经发布，截止3月12日。
- **We can see when you use AI in autograder**：自动评分系统能看出来你用了AI。

关于作业他还口头补了两句幻灯片上没有的话：第一次作业还有大约三天到期，但仍有相当多的学生一点没交、甚至还没开始做，他提醒大家不要拖到最后一天才动手；另外对于已经做完并通过第一次作业的同学，可以直接开始HW 2。关于AI使用的警告他也解释了依据：自动评分器会记录时间分布，比如前几道练习题花了好几个小时、后面的题十分钟就刷完了，这种模式一眼就能看出来，所以「尽量自己solve」。

## 回顾：观测、状态与策略的记号 [02:09 - 02:51]

![回顾：从观测到动作的映射，展示 CNN 处理图像并输出驾驶动作概率的过程](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0003_00-02-32.jpg)

幻灯片第2页「Recap: Mapping Observations to Actions」把上一讲的记号一次性摊开。画面左边是一张雪天公路的行车记录仪视角照片（路面覆雪、两侧插着黄色标杆），中间是一串CNN的立体方块（典型的卷积网络结构图，逐层把特征图缩小），右边接一个四行输出表，每行是一个动作和它的概率：

| 输出值 | 动作 |
|---|---|
| 0.0 | Left |
| 0.0 | Right |
| 0.8 | Straight |
| 0.2 | Backward |

也就是说这张图想说的是：网络把图像观测$o_t$映成一个动作分布$a_t$，在这个例子里直行概率0.8、倒车0.2、左右转都是0。注意表格里Straight对应的是0.8，Backward是0.2，另外两项是0.0。

中间两个公式是这一讲反复要用的区分：

$$\pi_{\theta}(a_{t}\mid o_{t})\quad \text{Policy - Partially Observable}, \qquad \pi_{\theta}(a_{t}\mid s_{t})\quad \text{Policy - Fully Observable}$$

（幻灯片在第二个式子里把状态写成了大写$S_{t}$，下面的定义又用小写$s_t$，属于排版不一致，本笔记统一用小写。）

底部一条红色折线把输出的$a_t$绕回输入的$o_t$，表示闭环：动作改变了世界，世界又产生新的观测。旁边是术语表：$a_t$是agent在时刻$t$做的决策；$o_t$是agent在时刻$t$观测到的东西；$s_t$是时刻$t$世界的「状态」；轨迹$\tau$是状态与动作的序列$(s_1,a_1,\dots,s_T,a_T)$；$r(s,a)$是奖励，回答「$s$、$a$有多好」。

讲师强调：今天要看的是训练这种策略的第一种方法——模仿，而且是从专家轨迹里模仿。

## 模仿学习的问题设定 [02:52 - 03:55]

![模仿学习（Imitation Learning）的定义，包含专家轨迹数据集公式及策略映射示意图](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0004_00-03-22.jpg)

幻灯片第3页「Imitation Learning」只有两个要点。第一是**Data**：给定由「专家」采集的轨迹，称为 demonstrations（演示），数据集写作

$$\mathcal{D}=\{(s_{1},a_{1},\dots,s_{t})\}$$

（注意幻灯片上这个集合的最后一项写的是$s_t$而不是$a_t$，右边画着一个圆柱形数据库图标，里面标着同样的$(s_1,a_1,\dots,s_t)$）。第二是**Goal**：学一个策略$\pi_\theta$去「imitates」专家的行为。下方配图是上面那张雪路照片→一个写着Policy的蓝框→同一张照片上叠了一个左右分叉的黑色箭头，直观表达「看到这张图，就该输出专家在那个状态下会打的方向」。

口头上他把设定讲得更严：这些数据不是随机动作，而是「状态与动作的序列」，比如人类开车、或者人类遥操作机械臂并且成功达成某个目标；并且我们**假设这些演示是最优的**、展示了达成目标的最优行为。要做的就是在神经网络里找到参数$\theta$，使得机器人看到状态$s$时输出的动作尽可能接近专家会做的动作——在雪地这个例子里，就是模仿人类司机在这种路况下会打的舵。

## 行为克隆：把模仿写成最小二乘回归 [03:56 - 05:06]

![行为克隆（Behavioral Cloning）的算法步骤，重点展示了确定性策略下的最小二乘回归损失函数](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0005_00-04-13.jpg)

幻灯片第4页标题就是**Behavioral Cloning**，三步算法写得非常干净：

1. Given expert demonstrations: $\mathcal{D}=\{(s_{1},a_{1},\dots,s_{t})\}$
2. For deterministic policy, regress to expert's actions:

$$\min_{\theta}\ \frac{1}{|\mathcal{D}|}\sum_{(\mathbf{s},\mathbf{a})\in\mathcal{D}}\lVert \mathbf{a}-\hat{\mathbf{a}}\rVert^{2}\quad \text{where}\ \hat{\mathbf{a}}=\pi_{\theta}(\mathbf{s})$$

3. Deploy policy $\pi_{\theta}$ on robot

这里的关键限定词是**deterministic policy**（确定性策略）：只有当策略直接输出一个动作向量时，才能把它当成回归问题、用均方误差去量「专家动作$\mathbf{a}$」和「预测动作$\hat{\mathbf{a}}$」的差距。讲师说这就是最直白的实现方式，有些人大概已经熟悉了；形式上就是在数据集上做MSE。

训练流程他也走了一遍口头版：采一个batch→网络前向得到预测动作→算loss→反向传播进网络参数→用你喜欢的优化器（随机梯度下降）迭代优化策略。到这里为止，模仿学习看起来和监督学习没有任何区别。

值得注意的是第3步「Deploy policy on robot」——正是这一步把前面所有假设打破了，下一节就是讲它为什么会崩。

## 行为克隆为什么会失效：IID假设不成立与分布偏移 [05:07 - 08:46]

这里有个小插曲：讲师在05:08时误翻了一页，屏幕上短暂出现了后面13:11才正式讲的「Learning to Fly in Swiss Forests / Quadcopter Navigation in the Forest」幻灯片，他说了句「Oh, what's going on here? Sorry」然后退回来。所以这一页在视频里出现了两次，正式讲解在后面。

![瑞士森林飞行案例的幻灯片，展示了四旋翼无人机在树冠下跟随小径的视频截图](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0006_00-05-08.jpg)

回到正题，幻灯片第5页标题是**What can go wrong?**，左右两栏对照：

![监督学习与行为克隆的对比图，展示分布偏移问题及专家与策略状态分布不等式](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0007_00-05-12.jpg)

左栏**Supervised Learning**：一个淡蓝色椭圆里画着三个小圆圈$x_1$、$x_2$、$x_3$，椭圆下方标$p(x)$，右边的文字是「Inputs are independently, identically distributed (i.i.d.) and **independent of predicted labels!**」。他举的例子是分类猫的图片：你对图片A的预测，不会改变图片B的内容。输入和标签之间、输入和输入之间都是解耦的。

右栏**Behavioral Cloning**：画的是一条对角线轨迹，蓝色节点$s_1\to s_2\to s_3\to s_4$之间用暗红色箭头连起来（这是专家走的路），而策略从$s_1$出发，用灰绿色节点和灰色箭头$\hat{a}_1$、$\hat{a}_2$、$\hat{a}_3$走出一条**偏到上方**的平行路径。配文：「Small error in predicted action can lead to **drift away** from training data distribution!」下面用两种颜色写核心不等式：

$$p_{\text{expert}}\neq p_{\pi}(s)$$

红色那行注脚是States visited by expert，蓝色那行是States visited by the policy。

他的解释是：IID假设在行为克隆里根本不成立。策略在第一个状态上做了一点点错的动作（也许没收敛、也许预测就是有一点点偏差），于是到达一个「接近$s_2$但并不等于$s_2$」的状态；这个状态偏离了轨道一点点，而它是训练时没见过的**新状态**，所以下一次预测大概率又是错的，误差开始累积，轨迹一路漂离原来的最优轨迹。

他特别把结论抽出来一句：在行为克隆里，**策略访问的状态不一定等于专家访问的状态**，所以训练分布和部署时的分布之间往往存在分布偏移（distribution shift）。这种东西很容易想象：光照变化、白天采的数据晚上开车、不同的天气让观测看起来不一样，都会影响策略表现并让它继续漂。

## 它到底能不能用？NVIDIA BB8端到端驾驶 [08:47 - 10:02]

![幻灯片展示NVIDIA BB8自动驾驶汽车在户外环境中的演示画面，标题为Does it work?](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0008_00-08-50.jpg)

刚说完一堆问题，幻灯片第6页抛出一个反问：**Does it work?** 画面是NVIDIA那段著名演示视频的开头，字幕「Meet NVIDIA BB8」，右上角有NVIDIA的绿logo，底部引用是「End to End Learning for Self-Driving Cars，Bojarski, et. al., 2016」。讲师说这是2016年NVIDIA的一段视频：用一个深度神经网络把**单个前置摄像头**的原始像素直接映射到**转向指令**。

![幻灯片展示NVIDIA BB8自动驾驶汽车在锥桶场地中快速学习导航的演示画面](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0009_00-09-17.jpg)

第二帧字幕是「and quickly learned to navigate cones.」——BB8在一片空旷停车场的锥桶阵里穿行。他提到这段视频里车几乎像无人驾驶（现场还有人接了一句「Oh, look, no hands」），并且坦白说「一开始效果并不惊人」，是**收集了更多数据之后**才把这套行为克隆调到能用的。

![端到端自动驾驶演示幻灯片，展示车辆通过障碍物的车内视角视频截图](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0010_00-09-49.jpg)

第三帧是车内视角，字幕「and cruises right through」：前面是一个黄黑斜纹的可移动路障和一排橙色锥桶，车直接从中间穿过去。讲师的评价是：这是最早被部署到真实世界里的端到端驾驶原型之一。

于是问题被推到下一张幻灯片：我们刚刚才论证完它会漂、会崩，那它凭什么能用？

## 为什么能用：侧向相机的「假」恢复数据 [10:03 - 13:10]

![解释自动驾驶原理的幻灯片，展示车辆在直道上正常行驶的俯视示意图](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0011_00-10-03.jpg)

幻灯片第7页只有标题**Why did it work?**和一张俯视示意图：一条双向车道（上下各一条白色实线边线，中间是虚线分隔线），一辆白色小车正沿车道中心线直行。这就是最普通的训练情形。

他先讲清楚正常情况：人开车时基本都尽量待在车道中央，不会疯狂画曲线或漂移，所以**大部分训练数据就是「直行」**。

问题出在漂移上。如果某一步预测不好，车不是直行而是慢慢往路边偏——这对汽车来说是相当危险的。而危险之外还有一个更麻烦的机制：**这是一个从未访问过的状态**，因为全部训练数据都是直行，策略看到这种画面时的反应就是「我完全没见过，不知道该干什么」。而且这种危险状态的数据**很难采**：你不可能让人类司机真的把车撞向路边，去采集「撞墙时该怎么救回来」的演示。

![数据增强原理幻灯片，展示车辆侧向相机视角及对应的数学公式](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0012_00-12-39.jpg)

幻灯片第9页标题是**Why did it work? Data Augmentation!**，要点写着「Add "fake" data that illustrates corrections with side-facing cameras」，公式是：

$$\mathcal{D}_{\text{aug}}=\{(\mathbf{o}_{\text{center}},\mathbf{a}),\ (\mathbf{o}_{\text{left}},\mathbf{a}+\delta),\ (\mathbf{o}_{\text{right}},\mathbf{a}-\delta)\}$$

下面配图是一辆白色车的俯视图，车头右前方画了三个扇形视场（分别是左视、正视、右视），旁边三个绿色箭头：上方箭头弯向右、中间箭头笔直向右、下方箭头弯向左，正好对应「看到偏左的世界就往右修、看到正视就直行、看到偏右的世界就往左修」。

做法就是NVIDIA论文里的策略：不用一个摄像头而是**三个**——一个中心摄像头加两个侧向摄像头。然后用一个非常简单的trick把数据集增强：把侧向摄像头看到的画面也塞进训练集，并且给它们打上**虚拟动作标签**：朝向左边看的相机，标签永远是「向右打舵」；朝向右边看的相机，标签永远是「向左打舵」。他总结这一步的意义是：这本质上是在**骗模型学出一个恢复策略**，而全程不需要人类司机真的去撞一次车——用假数据模拟出自救（self-recovery）行为。

顺带一提，这一招之所以管用，正是因为它人工地让$p_\pi(s)$覆盖到了策略部署时会遇到的偏移状态，等于提前把分布偏移补上了。

## 同一招搬到苏黎世森林：四旋翼跟小径 [13:11 - 15:03]

![瑞士森林无人机飞行演示幻灯片，展示林间小径跟随场景](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0013_00-13-15.jpg)

幻灯片第10页标题**Learning to Fly in Swiss Forests**，要点只有一句「Same camera trick」，画面是一段视频的标题卡：一架四旋翼无人机在树冠下的林间小径上飞，蓝色色块上写着「QUADCOPTER NAVIGATION IN THE FOREST」，下面一行小字「TRAIL FOLLOWING UNDER THE TREE CANOPY」。

讲师很坦白地说他忘了在这页写上论文名和年份，只记得大概是**同一时期**的工作。这一篇不是自动驾驶，而是让四旋翼**穿过茂密森林**飞行。他觉得这个案例很酷，因为它的难点和汽车一样甚至更极端：人类天生不会飞，所以这种数据本来就难采。

![瑞士森林飞行学习案例，展示环境、神经网络与动作控制的解决思路图标](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0014_00-14-20.jpg)

同一页的后一个动画块是「How we solved the problem」，三个图标依次排开：一张画着弯曲小径和路标的照片（环境/路径）、一个内部有齿轮、下方有波形的大脑（网络）、一个方向盘配三个方向箭头（直行、左转、右转，即动作空间）。

他们的做法是把苏黎世那招原样搬过来：让人**戴着相机头环在森林里走路**采集数据——额头周围架一圈，本质上是三台GoPro。人虽然不会飞，但会走小径。有了三个视角之后，同样打上虚拟标签：画面看起来正对小径就标「fly straight」，画面偏向右边就标「向左飞」，以此类推。视频里能看到三个相机各自的画面，然后给它们加上这些虚拟标签，用来学习如何通过「直行/向右/向左」这一组指令在这片地形里导航。

这个案例的启示和汽车那条是一样的：**当危险或未知情形的真实演示采不到时，用几何关系人造出纠正标签**。

## 行为克隆的误差上界：从$\epsilon T$到$O(\epsilon T^{2})$ [15:04 - 18:20]

![行为克隆误差上界分析，定义时间步长T、错误概率ε及代价函数c(s,a)](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0015_00-15-11.jpg)

幻灯片第11页标题**Upper Bound of Behavioral Cloning**。左上是一个只有4个蓝色节点的斜向轨迹图，节点$s_1\to s_2\to s_3\to s_4$用暗红箭头串起来，终点上方标一个大的$T$。右侧是四条定义：

- $T$：time horizon（时间视界，机器人完成任务所需的步数）
- $\epsilon$：probability of $\pi_{\theta}$ making a mistake at any steps（策略在任意一步犯错的概率）
- 代价函数：

$$c(\mathbf{s},\mathbf{a})=0\ \text{if}\ \mathbf{a}=\pi^{*}(\mathbf{s}),\qquad c(\mathbf{s},\mathbf{a})=1\ \text{otherwise}$$

- 训练集上的错误率假设：

$$P(\mathbf{a}\neq\pi^{*}(\mathbf{s})\mid \mathbf{s})\le \epsilon,\quad \forall \mathbf{s}\in\mathcal{D}_{\text{train}}$$

左下角还有一张黑白照片：一个穿西装的人在两栋高楼之间的钢丝上拿着平衡杆行走。讲师用它做类比：想走过钢丝只有一条极其特定的动作序列，一旦你偏离掉下去，就很难恢复——倒不一定是「会死」这种安全问题，更根本的是**你手上根本没有一份「怎么从钢丝下边回到钢丝上」的演示数据**。于是关键结论是：在**没见过的状态**上，犯错的概率是**没有上界**的，上面那个$\epsilon$只对$\mathcal{D}_{\text{train}}$里的状态成立。

页面底部有一行引用（这行字在OCR里被漏掉了，画面原文是）：

> More analysis: “A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning” by Ross et al., 2011.

这正是DAgger的出处。

![行为克隆误差随时间步长T呈二次方增长的数学证明及分布偏移图示](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0016_00-17-43.jpg)

同一个页面的下半部分随后展开（第11页第二个动画状态，左上轨迹图里多出了一条灰绿色的策略分支，三处都标了$\epsilon$，节点$\hat{a}_1$、$\hat{a}_2$、$\hat{a}_3$）：

$$\mathrm{E}\Big[\sum_{t} c(\mathbf{s}_{t},\mathbf{a}_{t})\Big]\le \epsilon T+(1-\epsilon)\big(\epsilon (T-1)+\cdots\big)\approx \mathbf{O}(\epsilon T^{2})$$

下面一行大字结论：**Distribution shift causes the error to grow quadratically!**（其中quadratically是加粗的）

推导的直觉讲师讲得很直白：标准监督学习里你会期望误差是线性的，大约$\epsilon T$；但行为克隆不是。如果在第1步就犯错，那么剩下的**全部$T$步**都要付代价；如果第一次犯错发生在第2步，就要付$T-1$步的代价；把这些概率全部加起来，期望总误差不是线性增长而是**二次增长**，也就是$\epsilon T^{2}$，非常糟糕。

他补了一句平衡的话：这并不意味着行为克隆本身是坏方法，但它给了我们一个**性能的上界刻画**——数学上我们看到，一旦策略开始偏离专家状态分布，误差就是平方级的而不是线性的。那么，怎么修？

## DAgger：把专家纠正聚合进数据集 [18:21 - 19:50]

![解决复合误差的核心问题：能否使专家状态分布 p_expert 等于策略状态分布 p_pi(s)](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0017_00-18-34.jpg)

幻灯片第12页标题**Addressing Compounding Error**，中间一行问句「Can we make?」（这是动画的第一个状态，问题问的是能不能让下面这个等式成立），等式两侧用颜色标出含义：

$$p_{\text{expert}}=p_{\pi}(s)$$

红色：States visited by expert；蓝色：States visited by the policy。逻辑很直接：既然病根是「专家访问的状态」和「策略访问的状态」不一样，那第一个药方就是**能不能把这两个分布拉近、甚至变得一样**，把这个分布偏移闭合掉。这就引出DAgger（Dataset Aggregation，数据集聚合）。

![Dagger算法（数据集聚合）的四个步骤及状态转移示意图](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0018_00-19-24.jpg)

幻灯片第13页标题**Aggregating Corrective Behavior**，副标题「Dagger: Dataset Aggregation」，四步伪代码（画面原文，注意第4步的下标）：

1. Roll out policy $\pi_{\theta}$ and collect states: $(s'_{1},\hat{a}_{1},\dots,s'_{t})$
2. Query/label expert action at visited states $\mathbf{a}^{*}\sim\pi_{\text{expert}}(\cdot\mid \mathbf{s}')$
3. **Aggregate** corrections with existing data $\mathcal{D}\leftarrow\mathcal{D}\cup\{(\mathbf{s}',\mathbf{a}^{*})\}$
4. Update policy $\theta\leftarrow\arg\min_{\theta}\mathcal{L}(\pi_{\theta},\mathcal{D})$

左下的示意图是前面那张轨迹图的加强版：暗红色箭头还是专家路径，灰绿色是策略自己走出来的分支（$\hat{a}_1,\hat{a}_2,\hat{a}_3$），现在多了一组**蓝色箭头**——从策略实际访问到的偏航状态出发，指向专家会去的地方，表示把这些状态的纠正补进数据。

右侧三条评估（前两条是绿色对勾，第三条是红色叉）：

- ✅ Algorithm will converge $p_{\pi}(s)=p_{\text{expert}}$
- ✅ Achieves $O(\epsilon T)$ instead of BC $O(\epsilon T^{2})$
- ❌ Hindsight labeling & expert querying difficult

最底下还有一句单独的话：**Paradox BC works better if the data has more mistakes and recoveries!**（行为克隆的悖论：数据里如果包含更多错误与恢复，它反而学得更好。）

口头讲解里他补充了核心思想：让专家和策略的分布在**训练过程中的在线rollout**里收敛。先把策略放到真实世界里跑，看它访问了哪些状态，然后请人类/专家给这些状态标注「在这个状态下应该做什么最优动作」。**旧数据不扔**，只是把这些新的纠正状态-动作对加到已有数据集$\mathcal{D}$里，再在这个增强后的数据集上更新策略，如此迭代。可以数学上证明它会收敛到策略真正会看到的状态分布，从而打破二次增长、把误差拉回线性上界。

但要注意那个红叉：向人类专家提问是**昂贵**的。它可能非常耗时；还可能有人与机器人**动作空间不一致**的问题；如果你是在电脑上**事后（in hindsight）**补标注，还会有离散化误差等等。而前面那句「悖论」正是这条链的推论：既然学的是「从偏航状态回到正轨」，那数据里带错误和恢复反而更值钱。

## Human-Gated DAgger：在线干预与「何时求助」难题 [19:51 - 22:10]

![Human Gated Dagger（人类门控Dagger）算法流程及优缺点分析](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0019_00-21-05.jpg)

幻灯片第14页标题**Aggregating Online Interventions**，副标题**Human Gated Dagger**，五步（画面原文）：

1. Roll out policy $\pi_{\theta}$ and collect states: $(s'_{1},\hat{a}_{1},\dots,s'_{t})$
2. Expert intervenes at time $t$ when policy makes mistake
3. Expert provides partial demo $(\mathbf{s}'_{t},\mathbf{a}^{*}_{t},\dots,\mathbf{s}'_{T})$
4. **Aggregate** new demos $\mathcal{D}\leftarrow\mathcal{D}\cup\{(\mathbf{s}'_{i},\mathbf{a}^{*}_{i})\};\ i\ge t$
5. Update policy $\pi_{\theta}\leftarrow\min_{\theta}\mathcal{L}(\pi_{\theta},\mathcal{D})$

左下的轨迹图与上一页类似：策略分支$\hat{a}_1,\hat{a}_2$走偏之后，一条**蓝色长箭头**从偏航节点直接拉到$s_4$，表示人类接管后把剩余任务演示完。右侧两条评估：✅ Lets the human take control!；❌ How to detect when an intervention is needed?

他的解释是：标准DAgger事后标注上千条轨迹很贵，于是换一种形式，也叫**online intervention**——不是让机器人先跑完再由人来标，而是让机器人**在人类监督下在线运行**。一旦机器人开始犯错，人在回路（human in the loop）就接管，并提供**剩余任务的一段局部演示**。这更自然也更有效，因为人类只在**必要的时候**才拿过控制权，而不是全程标注。他给的类比非常形象：就像你学车时副驾上的教练，只有在你快要撞上路缘的那一刻才会伸手抓住方向盘，他不会一路自己开。

但红叉指出了新问题：这引入了一个挑战——**你怎么知道什么时候需要干预？** 他说这个问题微妙但确实困难，尤其当你不希望人类**一直盯着**机器人、随时判断该不该接手时，因为那种监督本身也是成本。

于是他把问题往上推了一层：能不能超越人类监督？让机器人学会**识别自己的不确定性**，在撞车**之前**主动来问人？接下来他用自己博士期间一个较老的工作作为案例，讲怎么把「干预检测器」自动化。

## 案例研究：语言歧义作为自动求助信号 [22:11 - 25:28]

![语言歧义案例研究：机器人通过对话澄清指令并执行抓取放置任务的三阶段演示](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0020_00-22-47.jpg)

幻灯片第15页标题**Case Study: Language Ambiguities**，三张并排的仿真机器人（TIAGo平台）照片，配上四段对话气泡：第一张上方是人类的指令「Fetch the round yellow thing」，机器人回问「Do you mean the lemon in the middle?」，人类答「Yes」；第二张上方是下一条指令「Place it left of the object on the bottom」，这一张的桌面上还叠了一个紫色的俯视栅格图，里面有一个品红方框框住的区域和一个青绿色的斑块，表示机器人对候选物体的空间理解；第三张是机器人把物体放到盘子旁边的执行画面。页面底部引用：**Composing Pick-and-Place Tasks By Grounding Language，Mees and Burgard, ISER 2020**。

（这里要区分两个东西：幻灯片上写的静态气泡是「the lemon in the middle」，而现场播放的真实机器人demo音频里机器人说的是「Do you mean the middle banana?」——同一篇工作的不同演示片段，指代对象不同。）

设定是：让人类用自然语言指挥机器人，比如「把桌上那个黄色的东西拿来」，但桌上有**好几个**黄色物体。理想情况下机器人应该能**检测到这条指令是有歧义的**，并反过来向人类提问以消歧。

![训练指代表达理解模型的排序损失函数公式及蓝色物体识别示例](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0021_00-23-37.jpg)

幻灯片第16页把方法拆成编号步骤。左边是「Pick up the blue stuff」这句话配一张真实桌面照片：桌上有一个红色纸杯、一罐放在碗里的可乐、一辆白色玩具车、一个黑色碗、一个蓝色马克杯和一个蓝色碗；黑色碗和蓝色马克杯被**蓝色方框**框住（表示被判定为候选），蓝色碗被**绿色方框**框住（表示最终grounding到的目标）。下方一个机器人图标的气泡写着「Do you mean this blue bowl on the table?」。

第1步标题：**Train referential expression comprehension model with ranking loss**，公式是（画面原文，注意两个hinge项用的margin都是$m_1$）：

$$\mathcal{L}_{\text{rank}}=\sum_{i}\Big[\lambda_{1}\max\big(0,\ m_{1}+S(o_{i}\mid r_{j})-S(o_{i}\mid r_{i})\big)+\lambda_{2}\max\big(0,\ m_{1}+S(o_{k}\mid r_{i})-S(o_{i}\mid r_{i})\big)\Big]$$

他说这个细节不会展开太多，本质上是一个**最大间隔排序损失（max-margin ranking loss）**，用「物体-语言描述」的配对来训练理解模型。这个loss要保证：每一对**正确**的句子-物体组合的得分，要比任何**错误组合**的得分高出至少一个间隔$m_1$。第一项惩罚「正确物体配上错误句子$r_j$」，第二项惩罚「正确句子$r_i$配上错误物体$o_k$」。

推论是：**测试时**如果有多个物体落在阈值/间隔之内，它们就都可以被当成**潜在目标**。

![指代表达生成模型的损失函数及测试时歧义处理策略的完整步骤](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0022_00-24-28.jpg)

除了理解模型，还需要一个**生成模型/生成头**，能给这些候选物体分别生成指代表达（caption）。第16页随后展开第2、3步：

2. Train referential expression generation model

$$\mathcal{L}_{\text{gen}}=-\sum_{i}\log P(r_{i}\mid v_{i})$$

$$\mathcal{L}_{\text{mmi}}=\sum_{i}\big[\lambda_{3}\max(0,\ m_{2}+\log P(r_{i}\mid v_{k})-\log P(r_{i}\mid v_{i}))\big]$$

3. If at test time there are multiple objects within margin $m$, generate descriptions and ask user

也就是：先用最普通的负对数似然$\mathcal{L}_{\text{gen}}$训练「看到物体$v_i$就说指代表达$r_i$」，再用MMI损失把正确配对的$\log P(r_i\mid v_i)$压过其它候选$v_k$一个间隔$m_2$。最后一步就是整个「自动求助」的触发条件：**测试时若有多个物体落在间隔$m$内，就为它们各自生成描述并反问用户**。

![语言歧义案例研究幻灯片，展示机器人执行"Fetch the red thing"指令的场景](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0023_00-24-41.jpg)

幻灯片第17页是真实系统的视频截图：蓝幕背景前的TIAGo双臂机器人，矮桌上散着紫色杯子、橙色球、玩具熊、红色碗、喷雾瓶、黄色海绵和柠檬，视频字幕是「"Fetch the red thing"」，右下角标着2x倍速。底部引用同样是Composing Pick-and-Place Tasks By Grounding Language，Mees and Burgard, ISER 2020。

现场播放的音频里，指令是「Robot, fetch the red thing」，机器人答「Fetching. What else do you want?」；接着「Robot, place it behind the colored dish」，机器人「Understood. Placing. What else do you want?」。然后他切换到黄色物体的例子：桌上有三个黄色物体，机器人生成几条候选描述（「Do you mean the middle banana?」），用户回答yes，选中的就是香蕉，然后可以继续收拾厨房。

这一节的落点是：**歧义检测=不确定性检测的一种具体实现**，它把Human-Gated DAgger里那个红叉（何时该干预）变成一个可以学习、可以自动判定的问题。

## 第二类失败原因：非马尔可夫的专家行为 [25:29 - 28:45]

![非马尔可夫行为的策略公式对比图，展示当前观测与历史观测对动作的影响](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0024_00-25-29.jpg)

幻灯片第18页标题**Why might we still fail to mimick the expert?**（幻灯片上mimic写成了mimick），副标题**Non-Markovian behavior**，左右两个策略形式对照：

$$\pi_{\theta}(a_{t}\mid o_{t})\qquad\text{vs}\qquad \pi_{\theta}(a_{t}\mid o_{1},\dots,o_{t})$$

左边下面写着「action depends only on current observation」，右边下面写着「Human behavior might be affected by past observations, emotions, privileged information etc.」。两个公式下面各有一个红色向上箭头指向条件项。

他的论证是：上一讲我们说动作只能以当前状态（实践中是当前观测）为条件，但**人类并不总是按当前状态来选动作**——同样的一件事看两次，我们未必会重复同样的动作。原因可能有几类：

- **上下文与历史**：做决定之前我们会参考之前发生过什么。
- **情绪状态**：心情会影响行为表现。
- **意图**，以及**从过去学到的东西**。

他举了开车遇到骑行者的例子：作为人类司机，你不是靠一张快照来做判断的，你会**追踪这个骑行者过去5秒是怎么移动的**，据此预测他接下来会怎么动，然后这个预测影响你作为司机的动作。如果机器人只看到骑行者的**单张图片**，它推出来的动作分布，和看到过去5秒运动轨迹时推出来的，会完全不同。

- **特权信息（privileged information）**：这是遥操作时最明显的现象。他放了自己博士期间遥操作机器人的画面：人站在机器人旁边，**直接看到机器人在做什么**；而策略真正看到的只是那几个摄像头画面。两者差别很大——相机视角里有**多得多的遮挡**，而人可以走动、换角度看得更多。如果人按自己的视角去演示，策略却只能看到相机视角，那这份数据里的决策逻辑对策略来说是学不到的。（他顺带提到，所以遥操作时也会用人家最后$n$帧之类的做法，尽量让人也看着策略所见的画面来操作。）

把前面说的串起来：**如果演示因为历史依赖而是非马尔可夫的，那就麻烦了**——机器人看到**完全相同的视觉状态**两次，却看到专家做出**两个不同的动作**，而它缺少理解「为什么不同」所需的历史与上下文。

## 给策略加历史，以及因果混淆 [28:46 - 31:52]

第一个念头当然很朴素：**那就把历史加进策略**。不只喂当前时刻的一帧，而是喂最后5帧或10帧，然后用你喜欢的序列模型（LSTM、RNN、Transformer，随便哪个）去处理这$n$帧，照样用行为克隆训练。

![向策略添加历史信息的Seq2Seq架构示意图及局限性结论](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0025_00-29-32.jpg)

幻灯片第20页标题**Adding History to Policy**，要点「Encode with some sequence to sequence architecture」。图里左边是一段机器人视频帧（Franka机械臂在木柜前开抽屉，左上角字幕「1. "Open the drawer"」，右下角「2x」倍速标记，柜子上还印着像素风的机器人logo），红色箭头指向一个写着**Seq2Seq Model**的蓝框，再一个红色箭头指向一个黑色机械臂图标（输出动作）。页面底部是一个**红色叉号**加一句话：**Adding history to the policy doesn't always make things better!**

他说这个主意看起来很好，除了会让模型计算上更贵一点之外——但事实是，**给策略加历史并不总是让事情变好，有时候反而让事情明显更糟。**

糟在哪儿？这就是**因果混淆（causal confusion）**，一个「带着历史做模仿学习」的著名问题。机制是：当你把一个**高容量模型**放到**完整的历史**（观测、传感器、动作）面前时，它天然会去**推断出虚假相关（spurious correlations）**——因为它本质上只是在找**最容易降低loss**的那条路，而这条路并不总是正确的因果路径。

![因果混淆现象的三步案例分析：从数据到测试时的失效](/coursenotes/eth-robot-learning-2026/l03-imitation-learning/kf_0026_00-30-22.jpg)

幻灯片第21页标题**Causal Confusion**，要点「Policy might infer **spurious correlations**!」，左边还是那张「1. "Open the drawer"」的2x倍速视频帧，右边是三步：

1. Data: every time the robot opens the drawer, the gripper grasps the handle with 10 N
2. Model learns: if history shows a gripper force spike of 10 N, pull to open
3. Test time: gripper slips and only reads 2 N, the pull command is never triggered

底部一行引用：**More analysis: “Causal Confusion in Imitation Learning” by de Haan et al., 2019.**

口头版讲得更细：演示里每次机器人拉开抽屉，夹爪传感器都记录到抓把手时**正好10牛顿**的尖峰。模型一看就想：「啊，我根本不需要去看抽屉的视觉状态或者机械臂的位置——只要夹爪力历史里出现这个10 N尖峰，我就触发pull动作。」可一旦部署到真实机器人上，夹爪可能打滑一点、或者有点灰，传感器只读到**2牛顿**，模型就懵了：它以为**传感器读数**是因、拉开抽屉是果，而不是「把手被抓住」这个**视觉状态**才是该依赖的东西。结果就是pull指令永远不会被触发。

他的建议是：这是带历史训练策略的一个真实挑战，你需要做**相当多的正则化**，去确保模型仍然在看**正确的视觉状态**，从而学到真正的因果结构。

## 第三类失败原因：多模态行为与「平均动作」陷阱 [31:53 - 34:18]

接下来是本讲后半段的主线，讲师说这也是**即便我们有完美数据**仍会出问题的地方：**数据里的多模态行为（multi-modal behavior）**。

例子还是第一讲就出现过的「关抽屉」。两重冗余叠在一起：

- **机器人自身的冗余**：如果关节数有冗余，那么要让末端执行器到达**同一个位姿**，就有**多种**关节配置。
- **任务层面的多解**：要「走到把手前再把它推进去」，实现方式几乎是**无限**的——可以侧着抓（side grasp）、可以真的抓住把手也可以不抓、可以自上而下（top-down）、可以转90°、以及介于两者之间的一切你能想到的姿态，甚至可以不抓而是用手肘把它推上。

关键在于：**当你让人类来录这份数据时，他们会自然而然地做出所有这些不同的行为**。于是数据里同时存在**随机性（stochasticity）**和**专家不一致性（expert inconsistency）**：达成同一目标有多条路，而且**从原则上说它们全都是合法的**，因为它们都导向同一个成功。

为什么这对行为克隆是致命的？因为如果你就用第一个算法里那个**均方误差**回归，**数学会强迫模型去求所有演示的平均**。他的比喻是滑雪/开车绕树：专家给你看了一条**从树的左边绕过去**的轨迹，又给你看了一条**从右边绕过去**的轨迹，这两条都是对的；但它们的**平均值**是**笔直撞进树里**。MSE在单峰分布上是最大似然，在多峰分布上却会把概率质量堆到峰与峰之间的**谷底**——那里恰恰是专家从来不会去的地方。

## 建模多模态之一：高斯混合与动作离散化 [34:19 - 37:27]

既然「回归到单个点」失败是因为平均值与专家数据不一致，那我们就需要一个**能表示多个有效动作峰（modes）**的策略。

**第一件工具：高斯混合（Mixture of Gaussians）。** 用一组**均值、协方差和权重**来表示动作分布。好处是：只要一个分布能被高斯混合表示，你就能建模它——表达力相当强。但有一个必须提前决定的东西：**你要预设有多少个mode**。他说，比如你定10个；但如果你要控制的是**人形机器人**，你可能需要**1000个mode**，那时候事情就开始变得棘手了（参数数量、优化难度、每个mode分到的数据都会成问题）。

**第二件工具：离散化（discretization）。** 如果不想要「必须先定mode数」这个约束，想表示**任意复杂的分布**，一个有力的做法是把连续动作变成分类问题：把动作空间切成**网格/箱子（bins）**，然后对每个箱子做分类。离散化表示多模态分布是很好的（比如前面无人机飞行、汽车转向那种低维动作），但**朴素地**用到高维空间会碰壁：需要的箱子数量随维度**指数增长**（维度灾难）。

**修补：逐维自回归离散化。** 把$d$维动作按维度顺序逐个预测，复杂度就从指数降到**线性于维度数**，可以用一个序列模型（比如自回归Transformer）来分解。形式上就是对每个动作分量依次预测，用链式法则得到整个动作向量的概率：

$$P(\mathbf{a}\mid \mathbf{s})=\prod_{k=1}^{d}P\big(a^{(k)}\mid a^{(1)},\dots,a^{(k-1)},\mathbf{s}\big)$$

他点明：**这基本上就是最早那批VLA（vision-language-action models）所采用的做法。** 如果你的机械臂是7自由度，那你就**逐个维度**地自回归预测这7个分量。

代价他也说了：这条路会**逼你**拿LLM或VLM这种**大型自回归模型**当backbone，计算上会更贵一些。（他还打断了一次，说「问题留到后面再问，我们得赶时间」。）

## 建模多模态之二：扩散策略 [37:28 - 39:10]

**第三件工具来自生成建模，尤其是扩散模型（diffusion）。** 大家大概已经在生成艺术里见过它们：模型把随机噪声变成高保真图像。他预告**第六周**会详细讲，这里只给一个「一分钟版本」。

过程分两半：

- **前向过程（forward process）**：拿一个干净样本（他用的例子是一张猫的图片），**迭代地加高斯噪声**，直到所有信息都被抹掉，最后只剩下一片静态噪声。
- **生成/反向过程（backward process）**：训练一个神经网络去**一步步去噪**，学会如何从噪声里把数据恢复出来。

实现上，你其实是**人造了一个训练集**：把图像加上不同程度的噪声，然后训练这个去噪网络去预测「更干净」的版本。

有一个技术细节他特意纠正了一下直觉：**去噪时我们并不是直接预测完整的干净图像，而是预测噪声本身，再从当前样本里把它减掉**，如此迭代地把干净信号一步步恢复出来。

最后他把两条路线并列对比：**自回归离散化和扩散都是迭代过程**——一个是在**动作维度**上迭代（逐维展开），另一个是在**去噪步骤**上迭代（逐步细化）。这也解释了为什么两者都能表示多峰：它们都不强迫模型输出一个「平均」的答案。

## 建模多模态之三：隐变量策略与「意图≠任务」 [39:11 - 41:00]

**最后一件工具是隐变量模型（latent variable models）**，原则上只要网络足够大，它能表示**任何**分布。

它的结构是：网络的输出**仍然是一个高斯**，但除了图像/状态这类常规输入之外，它额外接收一个**从先验分布采样出来的输入**，比如零均值单位方差的Gaussian $z\sim\mathcal{N}(0, I)$。讲师的说法很形象：这相当于给网络一个**种子（seed）**。不同的种子会让网络**预测出不同的mode、不同的行为**。

但显然你不能真的随便塞随机数就得到想要的行为，所以必须**训练模型把这些输入隐变量和期望的mode对应起来**。他回到绕树的例子：假设数据里有两个mode——从树的左边过、从树的右边过——我们就**用隐变量去条件化解码器**，等于告诉它「这次是左、那次是右」。

接着他专门处理一个常见的混淆点：**latent intent（隐式意图）和task specification（任务规定）的区别**。

- **任务条件（task conditioning）**是一个**监督信号**，你用它告诉机器人**执行哪一个具体行为**，比如「打开抽屉」；它定义的是**目标（goal）**。
- **隐变量**定义的是在这个目标**之内**你想要的**风格或mode**。

也就是说，**即便任务ID固定不变**，机器人依然面对**无穷多种**完成方式——这正是前面关抽屉的例子。这句话是理解下一节的关键：任务ID管「做什么」，隐变量管「怎么做」。

## 从任务ID到目标可达：为什么要条件在两个状态上 [41:01 - 43:16]

现在假设我们真的想让一个策略**学多个任务**。设想的例子是「把推拉门向左移动」，把这当作任务ID，然后建模相应的分布。

他立刻指出现实里一个很难绕开的问题：**在真实世界里，「任务成功」本身就不好定义。** 就这个例子说，机器人把推拉门大概**推开了一半**——这算成功还是失败？如果现场做个投票，大概有人觉得「够了，挺好的」，另一些人觉得「应该一路推到底」。**这不是标注员的错，而是问题本身的性质**：我们习惯把想要的行为看成**一组离散的任务**（移动推拉门、开门、滑门……），但**很难画出一个任务从哪里开始、到哪里结束**，因为任务通常是由更小的**子任务（subtasks）和技能（skills）**拼起来的，而**世界不是离散的，是连续的**。

于是他提出：原则上我们希望机器人学到环境里**可用的全部技能谱（full spectrum of skills）**。这就引出了**goal-reaching（目标可达）**这一目标设定：定义一个策略，让它**从场景里的任意初始状态出发，到达任意期望的目标状态**。

对比一下两种建模方式：

- 单任务：条件在一个任务ID上，可以是一热编码，也可以是自然语言。
- 目标可达：条件在**两个状态**上——一个初始状态$s_c$和一个目标状态$s_g$。

goal-reaching的好处他一句话收尾：**因为任务是连续的，所以（作为条件的）状态也是连续的**——不需要再人为切分任务边界。

## 案例整合：从无结构玩耍数据自学任务无关控制 [43:17 - 46:40]

这一节他把「处理多模态的隐变量模型」和「学任务无关控制」拼在一起，做一个案例研究。

**数据从哪来：无向、无结构的玩耍数据（undirected, unstructured play data）。** 让人类**遥操作机器人但不做任何特定任务**——给他摇杆，让他随便玩，数据就采下来了。这份数据的特点：

- **极度多模态**（什么行为都有）；
- **reset-free**：抽屉拉开之后不需要复位，接着玩就行；
- **相当可扩展**：因为操作员做的是**满足自己好奇心**的行为。他特意点了一句：仔细想想，这和**小孩开始学习的方式**没什么不同——孩子也是在探索、在跟环境玩耍。

**怎么用隐变量模型加目标条件来学？** 流程是：

1. **随机采一个玩耍数据的窗口**（比如一小段连续帧），这一段就是一个序列。
2. 把这个窗口里**最后一帧**通过一个**goal encoder**编码——因为要用**goal relabeling（目标重标注）**来学控制：随机采窗口，把窗口的末尾当作「要去的目标」，让模型预测**这段序列里实际执行的那一串动作**，于是它就学会了「怎么到达那个目标」。

**对付多模态的关键技巧**是：把这段**上下文化（contextual）的演示**通过一个**隐式规划空间（latent plan space）**做自编码，用的是一个序列到序列的**条件变分自编码器（CVAE）**，里面有两个编码器：

- **后验（posterior）**：输入是**采样到的那段序列**，它识别出**实际被执行的那个具体行为**，把它压成隐变量$z$。
- **先验／规划提议器（prior / plan proposal / plan sampler）**：输入是**初始状态和目标状态**，输出一个**连接初始与目标的所有可能行为的分布**。

训练时**最小化这两个编码器之间的KL散度**，让先验给「玩耍时实际执行过的行为」分配高似然。**测试时把后验丢掉**，只用这个plan sampler去预测**多模态的规划**。

他把这套设计的意义总结成两句：

- 只要把**短视界的、带目标图像的演示**喂进一个**简单的最大似然模仿目标**里，就能**完全自监督地**从多模态玩耍数据中学到**任务无关的控制**。
- **关键之处**在于：把解码器**条件在隐式规划（latent plan）上**，等于把多模性从解码器身上**卸下来**了——解码器因此可以**专心只学分峰的、单模态（unimodal）的行为**。

## 语言目标与变分下界：为什么优化ELBO而不是似然 [46:41 - 49:38]

**能不能把目标从图像换成语言？** 前面用的目标都是**从演示里截出来的目标图像**。他说同样的效率也能给到语言：**核心洞察是可以把「视觉目标」和「自然语言目标的embedding」互相替换**，这样就能**主要靠自监督地**从**没有语言标注的视觉玩耍数据**里学控制，从而**大幅减轻语言标注的负担**。（这其实就是后来那些「一个策略同时吃图像目标和语言目标」的做法的雏形。）

时间已经有点不够，他说来不及展开背后的数学细节，只讲最高层的逻辑：

- 我们**想要**最大化专家数据的**边际似然（marginal likelihood）**。
- 但有一个障碍：隐变量$z$是**连续**的，于是**边际化变得不可处理（intractable）**——你得对**无穷多个可能的规划**做积分。
- 标准做法是**变分推断的技巧**：不去直接优化似然，而是优化一个**代理目标**，即**变分下界（variational lower bound / ELBO）**。

这个下界由两项组成（他口述的两项，幻灯片上没写完整公式）：

$$\mathcal{L}_{\text{VLB}}=\underbrace{\mathbb{E}\big[\log p(\mathbf{a}\mid z,\cdot)\big]}_{\text{reconstruction: 隐式规划多好地解释了专家行为}}-\underbrace{\mathrm{KL}\big(q(z\mid \cdot)\,\|\,p(z\mid \cdot)\big)}_{\text{regularization: } z \text{ 离轨迹先验分布有多远}}$$

- **重构项**告诉你：这个隐式规划**在多大程度上解释了专家的实际行为**。
- **正则项**告诉你：$z$**离你在轨迹上设定的先验分布有多近**。

然后他说，**当你引入目标条件**之后——也就是有一个**同时装着$s_c$（当前状态）和$s_g$（目标状态）的context变量**——这个式子就**恰好对上**了前面架构图里的三个组件：**后验、先验、动作解码器**。

最后他再强调一次这个设计的优点：问题被**因式分解**成了「隐式规划」这一层，于是**解码器的容量被解放出来，只用来学单模态行为**。

一个side note：他提到自己博士期间的很多工作都是**从玩耍数据学习**、并用这类**隐变量模型**，有兴趣可以去看那几篇论文（本讲没有给出具体标题，所以这里不列）。

## 本讲收尾 [49:39 - 50:36]

结课三句话，他按顺序回收了整讲线索：

1. 讲了**最简单形式的模仿学习**（行为克隆：把模仿写成对专家动作的监督回归）。
2. 解释了**怎么让专家访问的状态和策略访问的状态靠得更近**——通过**DAgger**和**在线干预（online interventions）**，把$O(\epsilon T^{2})$的二次误差压回$O(\epsilon T)$。
3. 处理了模仿学习里几个**常见坑**：演示者的**非马尔可夫行为**（历史、情绪、意图、特权信息），以及为缓解它而**给策略加历史**所带来的**因果混淆**；还有**多模态行为**，以及若干试图缓解它的**工具**（高斯混合、离散化与逐维自回归、扩散、隐变量模型）。

然后他说「With this, thank you for your attention」，接着下一环节：「Let's go with the papers.」——正好呼应开头那条「每周一的论文报告从3篇减到2篇」的教务调整。

## 要点回顾

- **问题设定**：给定专家轨迹$\mathcal{D}=\{(s_1,a_1,\dots,s_t)\}$（假设演示最优），学$\pi_\theta$去模仿。最直白的实现就是行为克隆。
- **行为克隆**：确定性策略下就是最小二乘，$\min_\theta\frac{1}{|\mathcal{D}|}\sum_{(\mathbf{s},\mathbf{a})\in\mathcal{D}}\lVert\mathbf{a}-\hat{\mathbf{a}}\rVert^2$，$\hat{\mathbf{a}}=\pi_\theta(\mathbf{s})$；采batch→前向→算loss→反传→SGD迭代；然后部署到机器人。
- **病根**：监督学习的IID假设不成立。预测的小错误→到达没见过的状态→预测更错→误差累积→轨迹漂离最优轨迹。核心不等式是$p_{\text{expert}}\neq p_\pi(s)$。
- **误差上界**：设$T$为时间视界、$\epsilon$为训练状态上任意一步犯错概率（$c(\mathbf{s},\mathbf{a})$在动作等于$\pi^*(\mathbf{s})$时为0否则为1），则$\mathrm{E}[\sum_t c(\mathbf{s}_t,\mathbf{a}_t)]\le\epsilon T+(1-\epsilon)(\epsilon(T-1)+\cdots)\approx O(\epsilon T^2)$。没见过的状态上犯错概率**无上界**（走钢丝类比：没有「怎么回到钢丝上」的演示）。出处Ross et al., 2011。
- **DAgger**：①rollout策略收集状态；②在访问到的状态上问专家$\mathbf{a}^*\sim\pi_{\text{expert}}(\cdot\mid\mathbf{s}')$；③$\mathcal{D}\leftarrow\mathcal{D}\cup\{(\mathbf{s}',\mathbf{a}^*)\}$；④$\theta\leftarrow\arg\min_\theta\mathcal{L}(\pi_\theta,\mathcal{D})$。收敛到$p_\pi(s)=p_{\text{expert}}$，达到$O(\epsilon T)$；缺点是事后标注与询问专家很贵。悖论：数据里错误与恢复越多，BC反而学得更好。
- **Human-Gated DAgger / online intervention**：机器人在线跑，人只在策略出错的时刻$t$接管并给出剩余任务的局部演示，$\mathcal{D}\leftarrow\mathcal{D}\cup\{(\mathbf{s}'_i,\mathbf{a}^*_i)\},\ i\ge t$。优点是让人只在必要时控制；难点是**如何检测何时需要干预**。
- **自动检测干预（语言歧义案例，Mees & Burgard, ISER 2020）**：用max-margin排序损失$\mathcal{L}_{\text{rank}}$训练指代理解模型（两个hinge项的margin都是$m_1$），使正确配对比任何错误配对高出间隔；测试时若多个候选落在间隔内即判为歧义；再用$\mathcal{L}_{\text{gen}}=-\sum_i\log P(r_i\mid v_i)$与MMI损失$\mathcal{L}_{\text{mmi}}$训练生成模型，为候选各生成一条描述并**反问用户**。
- **第二个坑：非马尔可夫行为**。$\pi_\theta(a_t\mid o_t)$假设动作只依赖当前观测，但人类行为受过去观测、情绪、意图、**特权信息**影响（遥操作时人看到的比相机多、遮挡更少；骑行者要跟踪过去5秒才能预测）。同一视觉状态出现两次而专家动作不同，策略就无从解释。
- **加历史与因果混淆**：把最后$n$帧喂给Seq2Seq（LSTM/RNN/Transformer）看似自然，但「加历史并不总是更好，有时更糟」。高容量模型面对完整历史会**推断出虚假相关**：抽屉例子里它学到「夹爪力出现10 N尖峰就pull」，而部署时打滑只读到2 N，pull永不触发（de Haan et al., 2019）。需要大量正则化逼它关注正确的视觉状态。
- **第三个坑：多模态**。关抽屉有机器人冗余和任务多解，人类演示天然包含侧抓/顶抓/90°/用手肘推等一切合法做法；MSE会强迫模型**求平均**，绕树左与右的平均=**笔直撞树**。
- **四件工具**：①高斯混合（均值/协方差/权重，但必须**预设mode数**，人形可能要1000个）；②动作离散化（变成分类，低维好用，高维**箱子指数爆炸**）→**逐维自回归离散化**使复杂度线性，$P(\mathbf{a}\mid\mathbf{s})=\prod_k P(a^{(k)}\mid a^{(<k)},\mathbf{s})$，这正是**最早那批VLA**的做法，代价是要用LLM/VLM当backbone；③**扩散**：前向逐步加噪、反向训练网络逐步去噪，且**预测的是噪声本身再减掉**；与自回归同为迭代过程（一个在维度上、一个在去噪步上）；④**隐变量模型**：输出仍是高斯，额外吃一个从$\mathcal{N}(0,I)$采的$z$当**种子**，训练它把$z$与期望mode对应起来。
- **latent intent ≠ task specification**：任务条件（一热ID或自然语言）管「做哪个行为/目标」，隐变量管「同一目标下的风格/mode」；即使任务ID固定，完成方式仍有无穷多种。
- **goal-reaching**：真实世界里「成功」难以定义（推拉门推到一半算不算？），任务边界本就画不清、世界是连续的，所以让策略**从任意初始状态$s_c$到达任意目标状态$s_g$**，学整个技能谱。
- **玩耍数据案例**：无向、无结构、reset-free、可扩展的遥操作玩耍数据（操作员受好奇心驱动，和小孩探索世界一样）→随机采窗口、把**最后一帧**经goal encoder当目标（goal relabeling）→用seq2seq CVAE把演示压进**隐式规划空间**：后验识别实际执行的行为，先验/plan sampler从$(s_c,s_g)$输出所有可能行为的分布，**最小化两者KL**；**测试时丢弃后验**只用plan sampler采多模态规划。这样就能**完全自监督**地学到任务无关控制，且**解码器被解放去只学单模态行为**。
- **语言目标与ELBO**：视觉目标可与**语言目标embedding互换**，从而从未标注语言的视觉玩耍数据自监督学控制、减轻语言标注负担。因为$z$连续，边际似然不可处理，改用**变分下界**：重构项（隐式规划多好地解释专家行为）+正则项（$z$离先验多远）；引入含$s_c$与$s_g$的context变量后，正好对应后验、先验、动作解码器三个组件。
