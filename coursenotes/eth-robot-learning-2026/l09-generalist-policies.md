---
title: "第 9 讲 · 通用机器人策略（Octo 与 CrossFormer）"
description: "本讲把「通用机器人策略（generalist robot policy）」拆成三个必须同时解决的要素——大规模数据集、大规模模型、可扩展评估。数据侧的答案是别再从零采集，而是把全世界实验室已有的机器人数据聚合成统一格式共…"
---

# 第 9 讲 · 通用机器人策略（Octo 与 CrossFormer）

> **ETH Zürich 机器人学习 · L09** ｜ 讲者 Oier Mees ｜ 63 分钟 ｜ [原视频](https://www.youtube.com/watch?v=dtofzDY9zuo)

## 核心结论

本讲把「通用机器人策略（generalist robot policy）」拆成三个必须同时解决的要素——大规模数据集、大规模模型、可扩展评估。数据侧的答案是别再从零采集，而是把全世界实验室已有的机器人数据聚合成统一格式共训（Open X-Embodiment，超100万条真机轨迹、20余种具身），第一次出现「单一模型超过各任务专用最优方法」的火花；建模侧的答案是把机器人策略看成多模态序列建模，语言token、图像token、动作token进同一个Transformer，于是Octo（80万条轨迹、扩散动作头、统一夹爪约定）和CrossFormer（90万条轨迹、跨具身Transformer、每种具身一个动作头）能在零样本下控制多种机器人，并在微调时把新观察/新动作空间「插几个token」就接进去；VLA侧的答案是复用互联网规模的VLM先验，用动作token化+next token prediction起步，再走π0式的flow matching动作专家与「知识绝缘」（动作专家读VLM但不回传梯度）。讲师同时给出大量工程口径：动作token准确率约95%才值得做实机rollout、异构batch用padding还是序列打包、shuffle buffer从2万扩到50万带来的性能跃升，以及用Simpler做「与真机成功率相关」的仿真评测（只支持两两比较，不给绝对成功率）。

> 配图说明：本讲关键帧只截到33:50，此后约29分钟（VLA与评估部分）无配图，相应章节按转写撰写。

## 从单任务策略到通用机器人策略 [00:00 - 03:00]

讲师Oier Mees开场先回顾前几讲的两类「单任务专家」策略：ACT与Diffusion Policies。ACT在插槽电池（SLOT BATTERY）任务上做实时策略rollout，Diffusion Policies在双臂台面上做抹酱料这类长时序灵巧任务（演示以4倍速播放）。这些方法确实把灵巧性、高频控制、多模态行为都做得很漂亮，但它们本质是单任务、单具身的专用模型：换一个新机器人或新任务，通常就得重新训练。

![课程标题页：机器人学习——从基础到基础模型，第10讲通用机器人策略](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0001_00-00-00.jpg)

标题页值得记两点：幻灯片上写的是「Lecture 10: Generalist Robot Policies」，日期27.04.2026，署名Oier Mees，机构为ETH Zürich与Microsoft；而本视频在课程序列里被标为第9讲，页码与发布编号相差一位。

由此引出机器人学习社区的「北极星」：用单一模型控制多种机器人、在多种环境中完成多种任务，并且能与人类协同工作。这条「圣杯」路线立刻撞上几个问题——数据从哪来；如何建模极大规模、极异构的数据分布；如何让策略泛化到真实世界的非结构化环境。理想状态是：你在网上下载一个喜欢的「机器人大脑」，机器人开箱就能对你的具体机器人表现出有用的、理想上智能的行为。

![通用机器人策略目标流程图：大规模机器人数据集训练基础模型，实现任意任务、任意机器人、任意环境](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0004_00-01-46.jpg)

## 基础模型的启示：通用模型胜过专用模型 [03:00 - 04:37]

这套思路直接借鉴NLP与计算机视觉。NLP的标准流程是：在包含许多不同任务的大规模数据集上训练一个基础模型，然后同一个模型去做代码补全、问答、翻译等任务。

![NLP领域基础模型流程图：大规模数据集训练基础模型，泛化到代码补全、问答、翻译等任务](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0005_00-03-10.jpg)

讲师强调的范式转移是：在多样数据上训练的大模型，通常会超过过去为每个任务单独设计的专用模型。幻灯片把三类专用架构与一类统一架构并列：翻译用Seq2Seq、GNMT；目标检测用R-CNN、Fast-RCNN；图像描述用DenseCap、Show and Tell；而视觉语言模型ViLBERT、Flamingo、BLIP、GPT-4V则用同一个Transformer同时吃图像token与语言token。

![基础模型与专用模型对比：左侧为翻译、目标检测、图像描述等专用模型架构，右侧为统一处理图像和语言的Transformer视觉语言模型](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0006_00-04-14.jpg)

图上的小字其实交代了专用模型的代价：R-CNN流程要「1.输入图像→2.抽取约2k个region proposals→3.计算CNN特征→4.分类regions」四步独立设计；DenseCap式的图像描述流程则是一串维度变换（图像$3\times W\times H$→卷积特征$C\times W'\times H'$→region features与region codes$B\times D$→LSTM解码）。统一架构的价值正在于把这些手工结构全部换成一个Transformer。机器人学习希望拿到同样的收益。

## 通用机器人策略的三大要素与各自难点 [04:37 - 07:31]

幻灯片把「Ingredients for Generalist Robot Policies」分三列逐条揭示，每列下面挂着两个具体障碍：

- Large Datasets：机器人数据稀缺（Scarce）；采集数据需要人工监督（Human Supervision）。
- Large Models：机器人数据是多模态且异构的（Multimodal and Heterogeneous）；机器人需要高频控制（High Frequency Control）。
- Scalable Evaluation：机器人评测繁琐且昂贵（Tedious and Expensive）；结果难以复现（Difficult to reproduce）。

![通用机器人策略的三大要素：大规模数据集、大模型与可扩展评估，以及各自面临的挑战](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0009_00-07-24.jpg)

对应到转写里的解释：机器人数据不可能像NLP/视觉的前沿模型那样大，你已经在用SO-101遥操作采数据，就知道要凑到前沿模型的量级有多难。建模上，机器人不只用语言或「语言+视觉」，交互本身就是多感官体验，而每个具身的观察空间与动作空间都不同。实时性上，你不可能把查询发给ChatGPT、等一分钟后API告诉你下一个动作。最后，当策略能做的任务与环境数量上升，评测成本会急剧增加，所以需要更可扩展的评测方式。本讲就按这三要素展开。

## 数据从哪来：聚合现有机器人数据集 [07:31 - 11:24]

第一个办法是承认现实：任何机器人实验室都会采数据，只是规模小、且通常不发布（大家总觉得「我的数据对别人没用」）。关键在于，过去社区虽然已有若干数据集，但格式不统一，无法互相共训。

![将Fractal、Bridge V2、RoboTurk、RoboNet等现有机器人数据集聚合为统一格式的示意图](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0011_00-09-31.jpg)

核心思路就一句话：把已有数据聚合成统一格式（Aggregate existing data into common format）。幻灯片上画的是Fractal、Bridge V2、RoboTurk、RoboNet等已有数据集的气泡，随后所有气泡被收进一个大椭圆里，示意「聚合」。

Mees在PhD期间与Google合作的Open X-Embodiment（OXE）就是这个思路的落地：向全世界各实验室索要可用的机器人数据，统一格式后共训。结果是超过100万条真机机器人episode，远超此前任何机器人数据集工作，且覆盖20多种机器人具身。

成果被命名为「Sparks of a Generalist Robot Policy」：这是第一次观察到，在多个机器人数据集上训练的单一模型，超过了为特定平台量身定制的最佳先前方法。柱状图可以直接读数（三根柱子依次是Original Method、RT-1、RT-1-X）：

| 评测设置 | Original Method | RT-1 | RT-1-X |
|---|---|---|---|
| Kitchen Manipulation | 43 | 48 | 63 |
| Cable Routing | 24 | 18 | 56 |
| NYU Door Opening | 53 | 65 | 80 |
| Autolab UR5 | 53 | 25 | 45 |
| Task-Agnostic Play | 33 | 68 | 72 |
| Mean（多机器人、多数据集） | 41 | 44 | 63 |

![RT-1-X模型在多个机器人数据集上的性能对比柱状图，平均得分63](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0012_00-10-35.jpg)

图上还标出了各评测来自哪些实验室（CLVR、RAIL、CILVR、AutoLab、AiS），其中「My PhD robot setup」是Mees本人在德国搭建的Franka平台，他在模型开发期间持续贡献了实机评测。论文信息也在页脚：Open X-Embodiment: Robotic Learning Datasets and RT-X Models，Open X-Embodiment Collaboration、Mees等，ICRA 2024，并获Best Conference Paper Award（从1765篇论文中选出）。这篇工作的真正影响是让「在超过一百万条多样跨具身轨迹上训练」成为可能，本讲后面提到的多数工作都把它当作地基。

## 如何建模异构数据：把机器人学习当成多模态序列建模 [11:24 - 14:24]

有了数据，第二个问题是怎么建模。机器人极度异构：任务描述方式不同、传感器类型不同、相机数量不同（有的1台、有的3到4台）、动作空间差异极大。让长得很不一样的机器人之间共享知识与信息，是真正的难点。

从一个基本例子出发：策略输入是当前观察（图像）与任务描述（如今多为自然语言），输出一段末端执行器位姿序列。

![通用机器人策略的输入输出架构图：语言指令与视觉输入映射到末端执行器控制](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0013_00-11-38.jpg)

把「机器人图像」换成普通互联网图片、把「指令」换成对场景的描述句，模型回答「这张图是自由女神像」，它就变成了一个视觉语言模型。

![机器人策略执行流程示意图：视觉场景与语言指令输入策略，输出机械臂动作参数](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0014_00-12-42.jpg)

于是关键想法出现：机器人学就是一个多模态序列建模问题——从图像与文本映射到「答案」，只不过在机器人这里答案不是文本，而是动作。

![将机器人任务建模为多模态序列（语言、图像、答案）的示意图](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0015_00-13-45.jpg)

这样做的收益很直接：一旦有了语言token、图像token、动作token，NLP与视觉在通用多模态序列建模上积累的进展就能搬进机器人领域。

## Octo：首个开源通用策略 [14:24 - 18:00]

Octo是朝这个方向迈出的第一步，也是第一个开源的通用策略：一个基于大Transformer的策略，在OXE的80万条机器人轨迹上训练，能控制多种机器人。与上周Scott Reed讲的Gato类似，做法是把所有异构输入与输出token化，再训练这个大Transformer。

![Octo开源通用策略的介绍页，包含训练数据和多种机器人操作场景图](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0016_00-14-49.jpg)

架构相当朴素：一个预训练语言编码器把语言输入tokenize（Octo用的是T5）；一个观察token化器把图像观察切成patch，过一个浅层CNN得到图像token。

![Octo架构图，展示任务token和观察token输入到Octo Transformer的结构](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0017_00-15-52.jpg)

幻灯片把两路token分别记作$\tau_T$（task tokens）与$\tau_O$（observation tokens），各自带一个可学习的前缀$p$与逐位的聚合结构；两路拼接后送入Octo Transformer，其结构与标准ViT几乎一样。

需要注意两点。第一，Transformer采用块状因果（block-wise causal）结构：任务token之间不能互相注意，也不能看观察token；而较早时间步的观察token可以看自己和任务，但不能看未来时间步。第二，动作由紫色的readout token承载——它们只是占位token，用来汇总嵌入，再交给功能头。

![Octo 模型架构图，展示任务与观测 Token 经 Transformer 融合并输出动作的流程](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0018_00-16-55.jpg)

训练目标是目标条件行为克隆（goal state conditioned BC），页面上写作

$$\pi_\theta\!\left(a_t \mid s_t,\, s_g\right)$$

箭头标注$\pi_\theta$的输入是「goal image, language」。这个表述非常灵活，因为目标$s_g$可以是目标图像，也可以是自然语言指令。由于是序列模型，同一套token会沿多个时间步展开，页面上可以看到「Task | Observation | Readout | Observation | Readout | Observation」这样跨时间步的排布，每个readout各接一个动作头输出$a$。功能头最终选了扩散解码头（diffusion decoding head），因为实验发现它在处理多模态行为上明显好于其他方式。

## Octo的两条设计决策与夹爪动作对齐 [18:00 - 19:45]

幻灯片把Octo的设计决策压缩成两条：Train everything from scratch, most params in Transformer；Align gripper action across OXE datasets。

![Octo 设计决策：展示 OXE 数据集中四种不同的夹爪动作约定及统一后的目标约定](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0019_00-17-59.jpg)

第一条是把算力全部投进Transformer本体，而不是像先前设计那样给每张图像配一个较大的ResNet编码器再做晚融合。当时觉得「数据这么多，干脆全部从零训练」，事后回看未必是最好的主意。

第二条是本页最有价值的细节，图里列出了OXE中四种互不相同的夹爪动作约定（典型动作序列是open→闭合抓取→张开释放）：

1. Absolute，+1开/0闭：Taco、Austin Sailor、Austin Sirius、NYU Franka Play。
2. Absolute，0开/1闭（反相）：Roboturk、Viola、Stanford Hydra、BC-Z。
3. Relative deltas，+1张开/−1闭合：RT-1、Kuka、Jaco Play、Berkeley AutoLab UR5。
4. Continuous with transition ramps（带过渡斜坡的连续值）：Bridge，开合过程中会出现中间值，图中橙点即过渡段的模拟量。

Octo把这四种统一到一个约定：目标为absolute，+1开/0闭。之所以单独对齐夹爪而不对齐坐标系（不对齐坐标系是出于可扩展性的取舍），是因为早期失败大量来自夹爪，例如过早抓取。把夹爪语义统一，是性价比极高的一招。

## Octo的实验结果与「顿悟时刻」 [19:45 - 22:16]

评测分零样本（Zero-Shot Eval）与微调（Finetuning Eval）两类，由多家实验室与机构共同完成，所有设置使用同一套微调配方。

![实验结果：零样本评估与微调评估在不同机器人平台上的成功率对比](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0020_00-19-45.jpg)

零样本页面给出三个平台实拍（Berkeley UR5、Berkeley Bridge、Google Robot）和一张成功率柱状图，图例明确标出三个模型的参数量：RT-1-X（35M）、Octo（93M）、RT-2-X（55B），横轴是WidowX、UR5、RT-1 Robot。总体结论是：Octo开箱就能控制多种机器人具身，并且超过了Google此前在相似数据上训练的模型——它胜过RT-1-X，并与RT-2-X这个550亿参数、闭源的第一代VLA打平。微调侧的四个设置是Berkeley FMB、CMU Toaster、Berkeley RPT、Stanford Coffee。

随后是本讲最有温度的一段：一段视频记录了Octo模型在Google机器人上第一次成功抓取，而机器人是通过远程服务器提供服务的。

![视频会议画面展示机器人实验室与研究人员，标题为“The Eureka Moment”](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0021_00-21-09.jpg)

讲师借这张「The Eureka Moment」截图介绍当时的合作者：画面左侧是今天的客座讲师Quan，当时在Google。机器人评测本来就难，当你人不在机器人旁边、必须靠别人帮忙时更难。值得注意的一点是，Mountain View的这批评测其实对模型相当out of distribution：虽然训练数据里有Google采集的数据，但那些数据是在另一栋楼里采的，团队搬了楼，场景自然变了分布。即便如此，简单任务仍然能成功，这就是他们的「顿悟时刻」。

## 适配新的观察与动作空间：Octo微调 [22:16 - 25:12]

通用策略真正重要的性质是可适应性：即使模型没有在你的机器人上预训练过，它也应该对你有用，成为post-training到一个具体具身的良好起点。

![过渡幻灯片，提出如何适应新的观察/动作空间的问题](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0022_00-22-17.jpg)

现实需求包括：某个任务想加入新的感知模态（例如力/力矩传感）、改变相机数量、或者换动作空间（从笛卡尔控制换成关节控制）。

Octo的做法是：来了新输入（比如力矩读数），就用几个token把它插进序列，然后微调模型。

![Octo模型微调架构图，展示预训练与微调阶段的任务令牌、观察令牌及Transformer结构](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0023_00-23-16.jpg)

这张图把预训练与微调两行并排对照：上面一行是预训练时的Task/Observation/Readout结构，下面一行微调时把相应位置换成New Observation、New Action Space、New Action Head，而Transformer主体保持不变。因为绝大多数参数都在Transformer本体里，微调时能保住绝大部分知识。

微调实验的关键结论写在下一页：RFM（机器人基础模型）学到的是用于迁移的更好表示。跨多个实验室、多种微调设置的结果显示，从通用机器人策略/机器人基础模型出发做微调，比当时任何最先进的视觉表示学习骨干都要好。讲师认为这条结论对今天许多SOTA模型同样成立。

![Octo模型采用情况幻灯片，包含加载预训练模型的代码及四足机器人和人形机器人的应用演示图](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0024_00-23-45.jpg)

采用情况也值得一提：一行代码就能取到预训练权重，`OctoModel.load_pretrained`直接指向发布在Hugging Face上的`rail-berkeley/octo-base`仓库（checkpoint URI带hf scheme）：

```python
from octo import OctoModel
model = OctoModel.load_pretrained(...)   # hf scheme + rail-berkeley/octo-base
```

幻灯片还给出「Researchers finetune to their robot with 50 demos」这个量化口径，以及两个社区案例：左侧来源标注是Peter Mitrano via Twitter（在未见过的四足机器人上加装机械臂做微调，即转写里被听成「spot robot」的那段），右侧来源是东京的公司Tokyo Robotics把它用于人形机器人（画面字幕为Place green bell pepper on the plate，两段视频均以4倍速播放）。

## 跨具身学习的动机 [25:12 - 27:24]

到这里讲的都是「在单臂操作数据上从零训练通用策略」。但一个只能控制7自由度机械臂的策略，到底有多generalist？真正的通用应当与具身无关。

![跨具身学习概念图，展示大型机器人数据集训练通用策略并适配多种机器人形态](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0025_00-25-12.jpg)

这一页的三条要点是：一个单臂操作策略有多「通用」？我们能否从更多机器人具身中学习？以及把知识跨具身迁移。图上从Large Robot Dataset进入Generalist Robot Policy，右侧分叉到Quadrupeds、Single Arms、Navigation、Quadcopters四类具身。

这恰恰是机器人领域迷人的地方：不同具身为不同任务而生，有无人机、四足、自动驾驶车、人形机器人等等。如果方法能从所有这些具身学习，可消费的数据量就跟着扩大——机器人领域的数据比我们以为的多，只是异构性让「哪些知识能迁移」成为难题。一个直觉例子：导航数据里学到的避障行为，可以迁移给室内无人机，因为无论Roomba还是无人机，避障都是共同行为。理想方案是足够可扩展：不加任何约束、不做人工动作空间对齐，数据丢进Transformer，让模型自己决定如何组合。

## CrossFormer：一个策略覆盖操作、导航、运动与飞行 [27:24 - 31:00]

CrossFormer就是上述理想的实现：在Octo原来的数据混合之上加入双臂操作、导航、运动等更多具身，总计90万条机器人轨迹，训练一个大Transformer，但在其上装配具身专属的动作头（单臂一个、四足一个，等等）。

![跨具身Transformer架构图，展示单一策略处理操作、导航、移动和航空任务](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0026_00-27-30.jpg)

页面的三条约束条件写得很清楚：Flexible Observation Spaces、Flexible Action Spaces $\in [2, \ldots, 1400]$、Control Freq. $\in [5\,\text{Hz}, \ldots, 50\,\text{Hz}]$，右侧输出到Quadrupeds、Single Arms、Navigation、Bimanual Arms、Quadcopters五类。

为什么需要可变长度的动作输出？因为极端跨具身学习里，各具身的控制频率差异巨大：WidowX大约5Hz，双臂灵巧操作可以到50Hz，于是模型应当预测的动作块（action chunk）长度也相差悬殊。多种动作头就是解决这一问题的办法之一，同时也免掉了动作空间对齐。为了最大化跨具身迁移，还要尽量最大化参数共享，例如相同类型的相机视角共享图像编码器权重。

![CrossFormer模型架构图，展示跨具身Transformer如何统一处理多种机器人观测与动作](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0027_00-28-34.jpg)

架构图补出转写没有的两个细节：CrossFormer的图像token化用的是ResNet，并且语言编码器（示例指令「Sweep the objects into the dustpan」）通过FiLM Conditioning注入视觉特征，即语言以条件化方式调制图像编码，而不是简单拼接；观察token按类型分列——Workspace Image、Navigation Image、Wrist Image、Quadruped Proprio、Bimanual Proprio，再配4个readout token，输出到Quadruped/Single Arm/Navigation/Bimanual四个动作头。页面顶部的三条主张是：No action space alignment required、Can consume data from any robot embodiment、Maximize parameter sharing across embodiments。

![单一策略在操作、导航、运动与航空任务上的多机器人应用场景拼图](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0028_00-29-24.jpg)

rollout拼图想强调一点：所有视频片段都出自同一个checkpoint、同一组神经网络权重，而不是「这个机器人挑效果最好的那个checkpoint」这种挑选结果。评测覆盖四足、Franka单臂、室内与室外导航，以及前面出现过的切寿司这类双臂超长时序灵巧任务。

定量结果的关键结论是「matches and outperforms specialist policies」。

![CrossFormer与最佳先前方法及单机器人数据集在多个平台上的定量对比柱状图](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0029_00-30-36.jpg)

柱状图三组柱子依次是Best Prior Method、Single-Robot Dataset、CrossFormer，横轴为Average、WidowX、Franka、ALOHA、LoCoBot、Go1、TELLO。可读到的大致数值是：Average约0.51/0.61/0.70；WidowX约0.20/0.20/0.25；Franka约0.51/0.40/0.40（这一项跨具身模型没超过专用基线）；ALOHA约0.70/0.50/0.80；LoCoBot约0.48/0.91/0.92；Go1与TELLO分别约为1.00/1.00与0.68/0.68/0.82，其中Go1没有先前方法的结果。论文信息：Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation, Locomotion and Aviation，Doshi*、Walke*、Mees、Dasari与Levine，CoRL 2024，Oral，在670篇论文中排名前4%。

## 多任务预训练究竟是否帮助后训练 [31:00 - 33:50]

CrossFormer回答的是「通用策略能否控制多种具身并胜过专用策略」。TRI的这项工作回答另一个社区普遍相信、但在机器人领域并不容易验证的问题：多任务预训练对后训练到底有没有帮助？

![多任务预训练实验结论幻灯片，列出预训练的优势与局限性](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0031_00-32-47.jpg)

页面三条实验设定：扩散Transformer策略（diffusion transformer policy），约1700小时预训练数据（含OXE数据集）；与从零训练的单任务基线做盲测A/B对比（操作员不知道自己在给哪个策略打分）；约1800次真机rollout加约47000次仿真rollout，覆盖29个任务，全部在双臂Franka上微调评测。页脚论文为A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation，TRI LBM Team，2025。图中任务是切苹果这类超长时序操作，讲师估计单集约2分钟。

结论被排成两组图标。绿色对勾（预训练确实有用）：预训练带来的提升具有统计显著性；后训练期间数据效率提高3到5倍；后训练数据越稀缺，收益越大。黄色警告（局限）：绝对成功率的提升幅度有限；数据归一化方式对下游性能的影响，常常超过架构或算法上的改动。

最后一条尤其值得记住，它再次说明机器人学习为什么难：可调的旋钮太多，有时连归一化这种「小事」都极其重要。讲师推荐认真读这篇论文，因为它做了非常严谨的研究设计。

## 从VLM到VLA：把动作当作token [33:50 - 36:45]

前面所有内容都是在机器人数据上从零训练。但即便当时Octo等人觉得数据已经很多，OXE（转写里被念成「ORC-E」）与LLM/VLM所用的数据量相比仍然小得可以。于是最自然的问题出现：

![提出关于互联网规模先验（如VLM）是否能迁移到控制任务的过渡幻灯片](/coursenotes/eth-robot-learning-2026/l09-generalist-policies/kf_0032_00-33-50.jpg)

互联网规模先验（例如VLM）能否迁移到下游机器人控制？

第一批尝试沿用几周前讲过的LLaVA架构（转写作「Lava」）。LLaVA要解决的问题是「怎么给LLM加上视觉」：把图像token前置拼到文本前面，LLM对这个统一序列照常做自注意力，完全不改注意力机制。关键问题于是变成：能不能把机器人动作当作与LLaVA图像token同等的东西来处理？

这正是第一批VLM被微调成VLA的方式：在VLM架构上把机器人动作token化，注入LLM的词表。一种常见做法是用新的动作token覆盖词表中频率最低的那批token。由于动作token已成为词表的一部分，LLM就能把它们当作普通token预测，训练仍然是next token prediction配同样的交叉熵损失。

上一讲已经讲过动作token化，这里只回顾：RT-2与OpenVLA这类最早的工作用的是非常朴素的「逐维度、逐时间步」分箱（binning）离散化。

## 训练VLA的收敛判据 [36:45 - 39:00]

接下来讲师转入「工程小贴士」模式，明确说明这些经验对VLA之外的模型（世界模型、其他大规模机器人策略训练）同样适用。

第一个问题：怎么知道模型收敛了？对于自回归VLA，一个不错的经验法则是等动作token准确率（action token accuracy）达到大约95%。要注意这个准确率是在teacher forcing下算出来的，不代表真机rollout的实际表现；它的含义是「在离散化方案下是否挑中了正确的那个动作bin」。在达到95%之前，基本不该去做真机rollout，因为结果不会好。

第二个要跟踪的量是动作的L1与L2误差：把动作token detokenize、反归一化回连续值之后，与真值动作比较，衡量预测的连续动作偏离多少。两个指标一起看，才能既判断「离散选择是否学对」，又判断「连续精度有多高」。

## 异构数据的批处理：padding与序列打包 [39:00 - 42:30]

第二个设计选择：异构数据怎么组batch。如果只在自己的单一机器人上微调，这个问题不太突出；但一旦做预训练、具身来源变多，它就变得相关。举例：一个具身只有1个第三人称相机，另一个额外加了2个腕部相机（共3台），两者没法直接堆叠成batch。

最简单的处理是padding到最长的观察输入数量，并把对应位置的注意力恰当掩蔽——Octo与CrossFormer就是这么做的。它非常容易实现，适合作为第一个版本。

但要看清代价：观察高度异构时，可能有一半的batch全是零，你会把大量flops浪费在往GPU里灌零。

另一种做法是序列打包（sequence packing），这是Transformer处理变长输入的专门技巧：不做padding，而是把所有样本拼成一个长向量，最大化GPU吞吐，每个样本只贡献它真正拥有的token。代价是需要维护一个累计序列长度（cumulative sequence length）变量，记录每个样本在这个长向量里的起止位置。

实现复杂度是主要权衡，具体两处坑：

1. 需要块对角（block diagonal）注意力掩码，保证不同样本的token不会互相注意——每个样本内的token只能看自己样本的token。
2. 需要在每个样本边界重置位置编码（positional encoding），否则位置索引会沿着拼接后的长向量一路增长。

## 数据加载策略：shuffle buffer与真随机读 [42:30 - 45:54]

训练时到底怎么把数据读进来，是个常被忽视但影响比想象大的问题。两种主要策略。

一是顺序读加shuffle buffer（Octo与OpenVLA都这么做）：数据从shard顺序流式读入缓冲区，再从缓冲区采样batch。要害在于缓冲区太小会让batch之间相关——随机程度与shuffle buffer大小直接成正比。

讲师讲了一个「吃过的亏」：训练Octo初期shuffle buffer只有2万个样本，再配上轨迹级（trajectory-level）交错，结果泛化性能很差。解决办法是在解码图像之前，就先对不同轨迹的帧做打乱与交错，这样shuffle buffer得以扩到50万个样本；数据混合等其他一切都没变，模型性能却有明显提升。

二是真随机读：为数据集维护一个扁平索引，逐个batch做随机seek。它给出更严格、无偏的随机性，但前提是IO要快——你通常不可能把整个数据集装进内存。

两者之间是权衡，也有折中方案（例如按块做随机读，而不是逐张图像读）。结论是：一旦开始为异构大数据集做预训练，数据加载本身就是一个关键的架构级选择，它通过随机性等途径直接影响策略性能。

## 多动作空间的两种方案与测试时的具身识别 [45:54 - 51:40]

下一个实践问题：不同机器人动作空间不同（一个数据集是7自由度臂，下一个是14自由度的双臂），怎么处理？两种主流做法。

第一种是Transformer/CrossFormer风格：共享的Transformer骨干处理观察，然后路由到若干具身专属动作头。好处是可以自然地输出变长结果，适配不同控制频率；代价是新增具身时必须新增头并重新微调。

第二种是π0风格：仍然是共享Transformer骨干，但后面接单个动作专家（action expert），输出到一个统一、padding后的动作空间——通常取所有具身里最大的动作维度，把每个机器人动作零填充到该维度。架构上简单得多，但有个注意点：不论具身的控制频率是多少，预测的动作块都是固定长度。

讲师在这里停下来，给全班20秒和邻座讨论一个问题：测试时模型怎么知道自己该为哪个具身产生动作？

讨论后的答案是：

- CrossFormer式多头方案由用户显式指定，你在prompt或配置里说明用双臂头还是四足头，路由是硬编码的。
- π0式单动作专家方案没有显式的头选择，模型必须从观察本身隐式推断具身。线索有三处：相机视角类型（不同具身常配不同视角组合）；本体感知（proprioception）本身就是动作头的输入，能给出极强的具身提示；以及语言指令——并非所有具身都能执行同一任务，让Roomba去切寿司显然不行。骨干网络要学会捕捉这些线索并把信息路由给动作专家。

## 当前主流VLA训练配方：快token、动作专家与知识绝缘 [51:40 - 54:07]

收尾VLA部分，讲师按π0.5与Knowledge Insulation两篇论文，概括当今许多SOTA VLA的高层训练配方，并说明这套配方已被很多实验室复现成功；更细节的最新模型留给客座讲师Quan再讲。

配方分两部分。

第一部分，VLM骨干用next token prediction训练，数据是快速token化（fast tokenized）的机器人动作与网络数据的混合。共训网络数据的作用是保住VLM骨干原有的能力、避免灾难性遗忘；而快token本身对表示学习也有好处（fast tokenization的机制在前几讲已讲）。

第二部分，动作专家用flow matching头训练，也就是那个负责预测所有具身的单一动作专家，由它产出连续动作。除了不 attend 快token之外，动作专家还实现了一个stop gradient操作：它从VLM骨干读取信息，但不把梯度回传更新骨干，从而「绝缘」保护VLM的语义能力。这同时有助于收敛与训练速度。更多细节见页面右下角Danny Drees的Knowledge Insulation论文（转写误作「knowledge installation」）。

## 可扩展评估：用仿真做相关性评测（Simpler） [54:07 - 62:25]

数据与建模讲完，剩下第三个要素：评估。通用策略的评测成本只会越来越高。讲师先和做过SO-101 rollout的同学互动，并预言：连续做一个月的实机评测，你会累到不想再看见那台机器人。

难点具体有三类。相机问题：第三人称相机装在机器人上，采完数据、训好策略后有人撞了一下相机，相机位姿变了，策略立刻out of distribution，可能要重采数据或重新微调。硬件故障：班上一大半人都遇到过；一旦换电机、给机器人「做手术」，而策略用了本体感知输入，新的读数就与你采集并训练的数据不一致。时间成本：低频控制很慢，你看着它挪一厘米、再等两分钟才知道有没有抓起来。

一条出路是仿真：仿真让没有同样机器人平台的人也能做研究，也更可复现（不用担心相机被撞、本体感知读数漂移）。因此在开发Octo期间，团队并行启动了名为Simpler的项目，合作者包括Quan与Ted Xiao（两人当时都在Google，Ted也曾来做客座讲师）。目标是把sim2real用于「只在真实数据上训练」的策略：只要能找到一种方式，让仿真里的评测结果反映真机表现，就能躺着跑大量评测。

方法上分两步。第一步消除控制差距（control gap）：保证策略在仿真里执行的动作，与在真实机器人末端执行器上执行时产生相同结果。做法是在少量真机样本上做系统辨识（system identification），让真实世界与仿真中的低层控制器输出基本一致。第二步缩小视觉分布差异：把仿真视觉往真机靠拢。这里用的过程相当朴素，不是NVIDIA Isaac Lab那种高保真风格，一是绿幕（green screening）——把仿真中可交互的资产分割出来，叠加到真实世界背景上；二是纹理匹配——把真实物体的纹理投影到仿真资产上，例如拿真实可乐罐的纹理贴到仿真可乐罐上。

必须理解Simpler给出的是什么：它给的是相关的（correlated）表现，不是绝对数值。仿真里拿可乐罐的成功率是50%，并不意味着真机也是同一个数字，它对真机的绝对成功率不作任何保证；能支持的是两两比较：若策略A在Simpler框架下优于策略B，那么在真机上同样成立。论文里还给了用排序等方式度量这种相关性的细节。

后续工作继续在「提升real-to-sim评测框架的相关性」这条线上推进，讲师点名的有Polaris、Robot Arena、Infinity与RoboLab。他认为这类工作很有意义：它把「你必须拥有一堆不同机器人平台，才能验证策略是否真的通用」这一门槛抽象掉了，让更多研究者与从业者能参与通用机器人策略研究。

## 本讲小结 [62:25 - 63:28]

讲师按开头的三要素收束：本讲讲清了开发通用机器人策略所需的三种原料。数据侧，可以把已有的离线机器人数据聚合成统一格式，从而支持共训；建模侧，可以在这些数据上从零训练跨具身的通用机器人策略，也可以进一步利用互联网规模的VLM先验来训练VLA（视觉-语言-动作模型）；评估侧，社区正在探索用与真机相关的仿真评测，让通用策略的评估变得可扩展。

## 要点回顾

- 单任务策略（ACT、Diffusion Policies）灵巧但专用：换具身或换任务往往要重训，因此需要通用机器人策略。
- 北极星是「一个模型：任意任务、任意机器人、任意环境」，并能在开箱后表现有用行为。
- 三大要素与六个难点：大数据集（数据稀缺、采集需人工监督）、大模型（数据多模态异构、需高频控制）、可扩展评估（繁琐昂贵、难复现）。
- 数据的答案是聚合而非重采：OXE把全球实验室的已有数据统一格式，超过100万条episode、20余种具身，首次出现单一模型（RT-1-X，Mean 63）超过各设置最佳专用方法（Mean 41）与RT-1（Mean 44）的火花，获ICRA 2024最佳会议论文奖。
- 建模的答案是多模态序列建模：语言token+图像token+动作token进同一个Transformer，让NLP/视觉的进展可以迁移进来。
- Octo：80万条轨迹训练的开源通用策略，T5语言编码器+浅层CNN图像token，块状因果注意力，readout token接扩散动作头，目标条件行为克隆$\pi_\theta(a_t\mid s_t,s_g)$；设计决策是把算力集中进Transformer、并把OXE四种夹爪约定统一为absolute、+1开/0闭；零样本胜过RT-1-X（35M）并与RT-2-X（55B）打平。
- 可适应性是通用策略的核心价值：新观察/新动作空间只需插几个token加一个新动作头再微调，因为主要参数都在Transformer里；RFM学到的表示优于SOTA视觉表示骨干，社区用约50条demo就能微调到自己的机器人。
- CrossFormer：90万条轨迹、跨具身Transformer，动作维度$\in[2,1400]$、控制频率$\in[5,50]\,\text{Hz}$，不需要动作空间对齐，同类相机共享编码器权重、每种具身一个动作头；语言通过FiLM条件化ResNet特征；所有rollout出自同一checkpoint，Average约0.70超过Best Prior约0.51（Franka一项除外）。
- TRI的严谨研究：约1700小时预训练+盲测A/B、29任务、约1800真机与约47000仿真rollout，结论是预训练显著提升后训练数据效率（3–5倍）、在数据稀缺时收益最大，但绝对成功率提升有限，且数据归一化的影响常常大于架构或算法改动。
- VLA路线：像LLaVA前置图像token那样把动作token化并注入LLM词表（常用低频token覆盖），用next token prediction与交叉熵训练；RT-2、OpenVLA用逐维度逐时间步分箱。
- 工程经验：自回归VLA的动作token准确率（teacher forcing）达到约95%再做实机rollout，同时跟踪L1/L2动作误差；异构batch可用padding（简单但浪费flops）或序列打包（需块对角注意力掩码与样本边界重置位置编码、维护累计序列长度）；shuffle buffer过小会导致batch相关，Octo把帧级交错提前到图像解码之前，使buffer从2万扩到50万并显著提升性能，另一条路是扁平索引的真随机读（依赖快IO）。
- 多动作空间两种方案：具身专属多头（可变长输出、加具身需加头并微调）与π0式单一动作专家+零填充统一动作空间（架构简单、动作块固定长度）；测试时多头方案由用户显式指定头，单专家方案靠相机视角、本体感知与语言指令隐式推断具身。
- 主流VLA配方：VLM骨干在快token化动作与网络数据混合上做next token prediction（共训网络数据防灾难性遗忘），动作专家用flow matching头并施加stop gradient（知识绝缘）以保护VLM语义能力并加速收敛。
- 可扩展评估：真机评测受相机被撞、硬件故障导致本体感知漂移、耗时等困扰；Simpler通过系统辨识消除控制差距、用绿幕与纹理匹配缩小视觉差异，使仿真成功率与真机相关，但只支持两两比较/排序，不给绝对成功率；后续有Polaris、Robot Arena、Infinity、RoboLab。
