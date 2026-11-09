---
title: "第 18 讲 · 以人为中心的 AI：看见、看不见与想看见"
description: "Stanford CS231n Spring 2025 L18 中文视频笔记（2026-11 重写版）"
---

# CS231n Spring 2025 · Lecture 18 以人为本的 AI：我们看见什么，我们珍视什么

**核心结论**：本讲是全季度的收官 talk，不引入新算法，而以「我们看见什么、我们珍视什么」为线索，把计算机视觉放进更长的研究演化史与「人的维度」里重新审视。全讲分三大块：构建看见人类所见的 AI（受认知科学启发的物体识别、ImageNet 与 2012 年数据／算力／神经网络三力汇合、从关系与场景图到图像描述）、看见人类未见（细粒度识别的超人能力，人类注意力局限、错觉与算法偏见，以及「该看不见时看不见」的隐私）、看见人类想见（把 seeing 与 doing 相连，用 ambient intelligence 与具身 AI 增强而非取代人）。讲者强调 AI 既能帮助也会放大我们（包括放大偏见），故造 AI 不能只有技术视角，须以多学科方式 study、forecast、guide 它对人与社会的影响。落点是把 AI 做成 augmentation 工具：VoxPoser 借 LLM + VLM 让机器人走出封闭世界，BEHAVIOR 以人的需求定义任务，脑机接口辅助瘫痪患者把这条线推进到临床。

## 1. 开场与本讲框架：看见与珍视 [00:05 - 04:42]

这是本季度最后一讲，性质上「有点离题」：不教新算法，而是一次带视角的谈话，既回顾更长期的研究演化，也补上今天 AI 的另一重要维度——human perspective（人的维度）；它与课程其他部分略有重叠，却串得更完整。

讲者先复习视觉起点：约 5.4 亿年前「第一缕光」进入动物世界，三叶虫（trilobites）演化出感光细胞探知外界。按动物学家 Andrew Parker《In the Blink of an Eye》的观点，正是视觉的突然出现触发 evolutionary arms race——动物「要么演化、要么死亡」——带来物种爆发式分化，即寒武纪大爆发（Cambrian Explosion）。

![关于寒武纪大爆发由视觉进化触发的引语幻灯片](/coursenotes/cs231n-deep-learning-for-cv-sp25/l18-human-centered-ai/kf_0001_00-01-48.jpg)

视觉至今仍是许多动物、尤其人类的主要感知智能系统，用于生存、工作、娱乐、社交、学习与发展。讲者回望 1960 年代把计算机视觉当成 summer vision project、想用几个本科生一夏构造视觉系统的尝试，认为这契合 AI 史的一贯模式：北极星清晰，却严重低估所需时间——「今天大概仍在经历」。但几十年间变化巨大：从自动驾驶、图像理解到生成式 AI 革命，视觉都扮演重要甚至引领角色，是时候回望「从哪来、往哪去」。报告分三个 chunk：see what humans see / don't see / want to see，第一段正受人类能力启发、想让机器做同样的事。

![构建AI以看见人类所见、未见与想见的三行标题幻灯片](/coursenotes/cs231n-deep-learning-for-cv-sp25/l18-human-centered-ai/kf_0002_00-04-20.jpg)

## 2. 构建「看见人类所见」的 AI：从人类视觉到物体识别 [04:42 - 15:04]

### 2.1 人类视觉有多强：物体是核心 [04:42 - 08:12]

半世纪前的经典实验：把从未看过的视频以 $10$ Hz 播放、每帧约 $100$ ms，事先不知目标是什么、在哪帧，人仍能在复杂场景中检测出目标（一个人），凸显以物体为中心的视觉理解。世纪之交神经生理学家用 EEG 帽测到，「有动物／无动物」这类复杂分类在刺激后约 $150$ ms 就已完成，对颅骨里的「湿件」快得惊人；大脑还有专司物体理解的 face areas、place areas、body parts areas，说明演化专门打磨了物体识别。于是几十年前物体识别成为视觉智能的基础构件，原始问题定义为：给定图像，让计算机叫出物体是什么。但这对人毫不费力的任务，数学上识别任一物体有近乎无限可能——lighting、texture、background、occlusion、viewing angle、scaling 等；课件用四张不同光照／纹理、背景／遮挡、视角下的袋熊照片说明视觉系统须认作同一只 wombat。

![袋熊图像在不同光照、背景和视角下被识别为同一物体的示意图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l18-human-centered-ai/kf_0003_00-07-22.jpg)

### 2.2 深度学习之前：部件组合与统计机器学习 [08:12 - 11:10]

第一波尝试受心理学式「自我内省」启发：人看物体先见几何 parts 再组合，于是用预定义 parts／shapes 按特定方式组合识别（1970s–1990s 一批模型），数学优美简洁却没真正奏效。第二波是 AI 史上重要时代、统计机器学习的开端，本质是「编程与统计建模联姻」：人们认识到世界太复杂，要泛化就必须学习参数、难靠手调模型，既需数据（当时不知需多少），也需设计能经不同学习规则学习的统计模型，于是 random fields、Bayes nets、support vector machines 百花齐放。到 21 世纪第一个十年，物体识别已有足够进展，出现类别数很少的国际基准：PASCAL VOC（20 类，Everingham et al. 2006–2012）、Caltech 101、Caltech 256、LabelMe、LotusHill，Average Precision 逐年攀升。

![早期数据集规模较小及PASCAL VOC基准测试的精度随年份变化曲线](/coursenotes/cs231n-deep-learning-for-cv-sp25/l18-human-centered-ai/kf_0004_00-10-50.jpg)

### 2.3 Biederman number、ImageNet 与 2012 汇合 [11:10 - 15:04]

最后一把钥匙又来自认知科学。心理学家 Irving Biederman 猜想人类能识别海量物体并给出数字，讲者戏称「Biederman number」：六、七岁儿童约能识别 30,000 到 100,000 个视觉类别，依据词典名词数加儿童识别物体的视觉研究。对当时只处理极少类别、极少图像的领域这很警醒，正是 ImageNet 的动机：认真对待它，构建约 22,000 类、1500 万张图像的数据集。大数据一到位，CNN（如今 transformers）这类高容量算法才真正显威。脉络是：有了 ImageNet，几年后 CNN 把物体识别这扇门「炸开」，如今算法能看任意照片、识别大小与任意朝向的物体；虽非 $100\%$ 解决、仍有 long-tail 问题，但工业应用上已相当成熟。汇合点是 2012：ImageNet 挑战提供数据、CNN 提供模型、两块 GPU 提供算力，三要素凑齐带来深度学习诞生，此后又催生 ResNet 等架构。课件柱状图给出 ImageNet 1000 类错误率从 shallow models 的 $0.28$、$0.26$，经 $0.16$、$0.12$、$0.07$、$0.036$、$0.03$ 降到 $0.023$，越过 Human 水平虚线。

![ImageNet分类挑战中2010至2017年各模型与人类水平的分类错误率对比柱状图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l18-human-centered-ai/kf_0005_00-14-14.jpg)

## 3. 超越命名：关系、场景图与视觉讲故事 [15:04 - 23:27]

### 3.1 关系必须被编码：scene graph 与 Visual Genome [15:04 - 18:39]

视觉智能不能止步于贴标签。两张图都只有 llama 和 person，但第二张关系不同、「故事」完全不同。认知科学家又走在前面：Jeremy Wolfe《Visual memory: What do you know about what you saw?》指出，光有物体不足以 categorize 场景，物体间关系（relationships between objects）必须被编码进理解。

![J. Wolfe关于视觉记忆与场景编码的论文截图及核心观点引用](/coursenotes/cs231n-deep-learning-for-cv-sp25/l18-human-centered-ai/kf_0006_00-15-36.jpg)

受此启发 CV 开始研究关系理解，代表是上周客座讲座 Ranjay 的博士论文：用 scene graph（场景图）作表征学习物体关系。它由实体节点（物体）、节点连边（关系）定义，有时带 attribute relationships；即便两人「一人喂另一人蛋糕」的简单场景也能形成很稠密的图。ImageNet 之后团队构建 Visual Genome，纳入 object relationships、物体与故事描述。有趣方向是 unusual relationships 的 zero-shot：person riding horse、person wearing hat 常见，horse wearing hat 罕见；person sitting on chair、fire hydrant on lawn 常见，person sitting on fire hydrant 罕见。大数据难反复拿到罕见组合样本，而组合式 scene graph 可先学常见关系再推导罕见关系，当时取得 state-of-the-art 识别率。

### 3.2 用自然语言讲故事：图像描述的三个层级 [18:39 - 20:30]

更进一步是用自然语言「讲故事」。约 2014 年（距 AlexNet 仅两年），思路是 CNN 加语言模型 LSTM；Andrej Karpathy 的博士论文是最早展示 image captioning／storytelling 的团队之一，dense captioning 出自 Justin Johnson（本课 co-instructor）；2015–2018 大量工作涌现，如今多模态 LLM 又把解法推高一档。讲者以世纪之初入行的 CV 学者身份坦言，很惊讶领域一旦有了数据与神经网络算法，解决速度如此之快。课件按难度递进排三层：Image Captioning（Karpathy, Johnson & Fei-Fei, NeurIPS 2014）、Dense Captioning（Johnson, Karpathy & Fei-Fei, CVPR 2016）、Paragraph Captioning（Krause, Johnson, Karpathy & Fei-Fei, CVPR 2017）。

![自动图像描述与故事讲述的三种层级对比：Image、Dense、Paragraph Captioning](/coursenotes/cs231n-deep-learning-for-cv-sp25/l18-human-centered-ai/kf_0007_00-19-35.jpg)

### 3.3 动态场景、任务爆发与认知科学的双向关系 [20:30 - 23:27]

更难的是 dynamic scenes：关系、运动更复杂，相机与场景里的 actors 都能做各种动作。与 Ehsan 及实验室学生合作的 multi-object multi-actor activity understanding（几年前发表）在她看来仍未解决；若梦想日常机器人与人类共处，就必须理解场景多复杂、谁在做什么、下一步是什么。本课未展开的还有 3D computer vision、human pose、生成式 AI／生成模型，课件列出 NeRF（Mildenhall et al., ECCV 2020）、Teed & Deng（NeurIPS 2021）、Cao et al.（CVPR 2017）、He et al.（ICCV 2017）与 OpenAI 的 DALL-E。

![计算机视觉三大应用示例：3D Vision、Pose Estimation 和 DALL-E 生成](/coursenotes/cs231n-deep-learning-for-cv-sp25/l18-human-centered-ai/kf_0008_00-22-14.jpg)

第一部分 take-home 两点：data、compute 与神经网络算法约 10–13 年前汇合才有深度学习革命；而我们长期研究的问题，灵感真正来自 cognitive science、psychology、neuroscience，这种双向关系还会延续——继续受大脑如何做事启发，也继续用 AI 反过来帮助脑研究。

## 4. 构建「看见人类未见」的 AI：细粒度识别与超人感知 [23:38 - 27:20]

这一部分把 AI 推到人类能力之外，讲者称 superhuman。多数人叫不出太多恐龙，更别说上千上万种鸟、数万种汽车类别——她称 fine-grained object categorization，人类本就不擅长，且认为至今未完全解决；在多模态 LLM 主流时代它被忽视、不算主流，但相信会回来扮演重要角色。早期细粒度鸟类识别用了 4,000 只鸟的数据集：沿物种树向上、名字越泛化误差越小，反之落到 fine-grained 层算法仍犯很多错、尚未 ready。另一项是实验室学生训练的细粒度汽车分类器，按 make／model／year 区分——1970 年代后累计数千车型。取 Google Street View 中约 100–200 个主要城市图像，用 car detector 检测街上的车，再把 CV 当「透镜」研究社会模式：车型与教育、收入高度相关，论文里还与投票、环境模式高度相关。没有任何单个人、甚至一群人能轻易做到，AI 在此真正拓展了人类能看见的边界。

## 5. 人类看不见：注意力局限、错觉与算法偏见 [27:24 - 33:26]

为讲透「人类也有局限」，讲者做了几个测试。Stroop test：字都能读，但要求尽快说出「字的颜色」而非读词（颜色词与字体颜色不一致的 12 词矩阵），会感到「它在跟你打架」，是视觉注意力与自动阅读的冲突。

![Stroop测试幻灯片，展示颜色词与字体颜色不一致的12个单词矩阵](/coursenotes/cs231n-deep-learning-for-cv-sp25/l18-human-centered-ai/kf_0009_00-27-34.jpg)

change blindness：两张交替图片间有一处很大变化（引擎 engine），要点一会儿才看出，也是著名心理学实验。这些好玩，但「注意力有限」并不好玩——在工作生活里后果可能极严重。医疗错误（medical errors）是美国医疗系统第三大死因；把剪刀留在病人体内是标志图像，实际更多是缝合针、一小块纱布，另有用药、流程、文书、诊断等错误。至今多靠手写 checklist 追踪：一发现少了东西手术就得暂停，平均接近一小时，带来细菌暴露、出血风险。实验室做过 demo（非部署系统，fidelity 不够）用 AI 帮外科医生计数纱布——看见人看不见。

![棋盘格错觉示意图，展示阴影对颜色感知的影响](/coursenotes/cs231n-deep-learning-for-cv-sp25/l18-human-centered-ai/kf_0010_00-30-56.jpg)

她最爱的错觉之一是 Ted Adelson 棋盘格：上方 A、B 两格实际灰度／亮度相同，即便看过下方揭示图仍难相信。原因是演化把「物体形状、光源、阴影如何形成」这类世界常见物理先验预置进视觉系统，深到无法换方式去看。由此引出严肃一点：人类视觉系统存在 bias，可能来自演化构造、社会经验或接触的数据；有些 bias 有害，一旦对某群体不公平就必须警觉。几年前人脸识别对某些肤色甚至性别识别更好，在自动驾驶、医疗等有后果。到 2025 并未解决，但讲者高兴看到 academia 与 industry 已有大量人关注 AI bias——几年前它还新到几乎无人留意。

## 6. 该看不见时看不见：隐私保护与 AI 的放大效应 [33:26 - 38:20]

还有一类「看不见」很特别：有时「不看」恰是我们想要的，为尊重隐私。如何造一个既帮人看得更好、又看不到人不想被看的东西的 AI？这既深、既是技术问题也是人的问题。从技术角度 ML 隐私路径很多。实验室几年前一篇用 smart cameras 放在病房或患者家中帮医生看得更好的论文，也须处理 faces、full body、homes 的暴露；课件表格列出 dimensionality privacy（降维）、face blurring／blurring／masking、federated learning（不把数据全送服务器）、homomorphic encryption 等，并区分计算在边缘还是服务器。

![隐私保护技术方法对比表，涵盖降维、模糊化及加密等方案](/coursenotes/cs231n-deep-learning-for-cv-sp25/l18-human-centered-ai/kf_0011_00-34-34.jpg)

讲者特别喜欢另一个非本组工作：拍视频识别动作同时保护隐私。单纯 blur／defocus 虽护隐私却丢太多信息，可能连人在做什么都不知道，而很多应用目标恰恰是知道动作。该工作由一组学生完成，软硬件结合：手工设计一枚以特定方式过滤视觉数据的 lens，顶行图像看不到脸、看不到身体；但因 lens 与软件联合设计，能反推出运动／人体活动信息却不反推人脸——想看见人来保护他们，又不想看太多以尊重隐私。

![李飞飞关于AI视觉能力的观点幻灯片，强调构建看见人类看不见之物的AI](/coursenotes/cs231n-deep-learning-for-cv-sp25/l18-human-centered-ai/kf_0012_00-37-03.jpg)

第二部分 take-home 归结为三种情形：细粒度识别这类超人能力；人类有 bias、注意力问题，用 AI 帮忙；以及某些场合根本不想被看见，要在不违反隐私下继续让 AI 帮忙。AI 既能 help 也能 amplify——我们若有 bias、有问题，它会一并放大。故造 AI 不能只取技术视角，还须取 human perspective，承诺 study、forecast、guide 其影响并尊重人的价值；课件写成以 multidisciplinary approach 研究、预测并引导 AI 对人与社会的影响。

## 7. 构建「看见人类想见」的 AI：从取代到增强 [38:24 - 47:30]

这一部分要「超越看见」，把 seeing 与 doing 连起来。社会对 AI 最大焦虑之一是 labor：头条总说劳动力受威胁、机器人抢工作。真相更复杂：否认工作会变是错的，历史上每次技术转变都带来劳动力市场变化，有些极痛苦、甚至引发内战与战争，而变化有时不可避免。插叙：威胁论多想体力劳动，但过去两年 genAI 冲击最剧的其实是白领，尤其 software engineering 与办公室分析类工作。另一面须承认 AI 也能帮忙——很多场景存在根本性人力短缺，尤其 elderly care 与 health care。现代医学延长寿命、社会走向长寿（好事），可年轻人要工作，谁照顾老人与慢性病患者？美国医院医护尤其护士流失，没有足够 hands、ears、eyes。故与其想「replace」，不如想「augmenting」——手术室纱布计数即一瞥。讲者把手术室、病房、药房到家庭这些「缺眼睛」处称为 dark spaces of health care。

![劳动力短缺数据图：显示每年25.1万医疗错误死亡、8300万老龄人口及护士缺口](/coursenotes/cs231n-deep-learning-for-cv-sp25/l18-human-centered-ai/kf_0013_00-39-57.jpg)

课件数据：251,000 deaths/year from medical error、到 2050 年 83M Americans 年满 65+、1M more nurses needed by 2023。

### 7.1 Ambient intelligence for healthcare 的三个例子 [41:26 - 47:30]

Ehsan Adeli 等主导 ambient intelligence for healthcare：把 smart sensors 与 ML 结合，从医疗场景 glean 出 health-critical insights，及时提醒患者、家属或医生，完整论文发表于几年前。三例：

**手卫生**：COVID 前就启动。院内感染（hospital acquired infection）是美国医院病人死亡主因之一，每年致死超全国车祸 3 倍，病菌在病房间传播、极难控制。human auditors 招不够且有疲劳；RFID 徽章靠近洗手液器就提示「大概在洗手」太不特异，病房走廊又小，站在旁边不等于在洗手。团队改用只采集 depth 深度信息的 smart sensors（蓝色画面天然护隐私）+ CV 分类「是否洗手」。结果算法比人更准更稳：几乎要把同一视频给 4 个人看才接近 AI，单人检测很稀疏。

**ICU**：病人与生死搏斗之地，花掉美国 GDP 的 $1\%$；目标是安全送病人出 ICU、转入 step-down unit 或回家。重要一课是帮病人做恰当活动（mobilization），对恢复关键却很微妙：要护士帮忙、医生下医嘱、指定时间正确移动并评估。团队与 Stanford 及犹他州 Intermountain Hospital 合作，在 ICU 布 sensors 监测 4 类动作：getting out of bed、getting in bed、getting out of chair、getting in chair。

**Aging in place**：老人希望独立健康住家里；COVID 初期老年人高死亡很大程度因医院超负荷，让老人在家安全健康很关键。sensors 可用于早期感染检测（尤其 thermal cameras）、mobility、sleep patterns、dietary patterns。

![家庭护理劳动力短缺的三个关键数据：2030年美国65岁以上人口占比、护理人员与老人比例及无偿护理时长](/coursenotes/cs231n-deep-learning-for-cv-sp25/l18-human-centered-ai/kf_0014_00-47-31.jpg)

课件另给家庭护理短缺数据：2030 年美国 65+ 人口占比 $7\%$ 上升、caregiver 与老人之比降 $42\%$（从 7:1 到 4:1）、无偿护理协助达 18.4 B（184 亿）小时。

## 8. 具身 AI 与 VoxPoser：闭合感知—行动回路 [47:30 - 53:49]

smart sensors 只是信息采集系统，没法过去帮病人翻身、递水送药，引出最后技术主题 embodied AI（具身 AI），很大部分是 robotics。它令人兴奋在于闭合 perception 与 action 回路：回想寒武纪，眼睛一出现动物就开始移动。但机器人仍慢、笨拙、难泛化；当今多数研究受限于 constrained setup、是 pick and place 这类 short-horizon tasks、setup 偏 anecdotal，又缺标准 benchmark。

VoxPoser 要把机器人带到 the wild：预定义任务集令人不满，而 LLM 什么都能聊，Wenlong Huang 等学生想缩小差距——给任意 open instruction，不必在 closed world 预训练一切。目标 in-the-wild generalization：训练集只有「open a drawer」，野外却遇各式门；演示让机械臂规划路径打开顶层抽屉、同时不碰倒花瓶，这些指令都未预训练。做法借 LLM + VLM：LLM 把指令转成指令集乃至代码，VLM 识别理解环境，再转成 motion planning 让机械臂执行，因此摆脱封闭世界训练。

![VoxPoser模型在开放世界指令下的机器人操作任务展示及特性列表](/coursenotes/cs231n-deep-learning-for-cv-sp25/l18-human-centered-ai/kf_0015_00-49-19.jpg)

细节：LLM 把「open top drawer」写成代码，把 drawer、handle 交给 VLM 在场景中检测，据此更新 motion map（heat map，标出该聚焦／不该聚焦处）；再给「watch out for the vase」，同样经 LLM 写代码、VLM 检测，这次更新负向 map，与前一张叠加得「哪里该去、哪里要避」的热图；对 motion planning map、rotation、gripper velocity 都照此处理。课件的 LLM 生成 affordance_map 代码如下，逻辑是在 $100\times100\times100$ 的 3D 体素图中按 z 轴排序找出最上面的把手并标记为 1：

![LLM生成的affordance_map代码片段，用于定位抽屉把手并标记3D体素图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l18-human-centered-ai/kf_0016_00-52-12.jpg)

```python
def affordance_map():
    msize = (100,100,100)
    map = np.zeros(msize)
    handles = detect('handle')
    k = lambda x: x.pos[2]
    handles.sort(key=k)
    top_handle = handles[-1]
    x,y,z = top_handle.pos
    map[x,y,z] = 1
```

相较「skill-level tasks、short-horizon goals、closed world instructions」的主流，VoxPoser 具备 Open-set of language、Open-set of objects、Zero-shot 三个特性；论文 *VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models*（Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, Li Fei-Fei，CoRL Oral 2023）。可完成 articulated object manipulation、摆餐巾、扫地、取吐司、摆餐桌、应对 online disturbances 等。

## 9. BEHAVIOR 基准：以人的需求定义任务与未来方向 [53:49 - 65:08]

机器人研究整体仍缺好 benchmark；实验室设置与真实世界差距大——真实世界复杂得多、不确定、变化大、交互且社会性强、多任务。NLP 与 CV 都从大规模数据集／基准获益，于是团队构建 ecological robotic learning environment 配大而多样活动集合，即 BEHAVIOR：Benchmark for Everyday Household Activities in Virtual, Interactive, and Ecological EnviRonments（CoRL 2022）。

![BEHAVIOR机器人学习基准测试环境介绍及任务示例](/coursenotes/cs231n-deep-learning-for-cv-sp25/l18-human-centered-ai/kf_0017_00-55-28.jpg)

本讲处处关乎 human values，绕不开的问题是：谁决定机器人该做哪些任务？讲者调侃每个做机器人的研究生只想要 laundry 和 dishwasher。与其自己列清单，团队做 human-centered survey：汇总美国与欧洲 labor office 等政府调查，整理数千项日常活动任务，上网找 1,400 位尽量多样的人回答「希望机器人帮哪些忙」并排序。现场小测也印证差异：clean kitchen floor 几乎人人愿意，shoveling snow、folding laundry 也愿意，cooking breakfast 开始混合，opening Christmas gift、buying wedding rings 则不想要——有些事在情感或社会层面对人有意义，不想让机器人代劳。结论：人们大量想要清洁类（toilet／floor cleaning），却不想机器人替你打壁球、买婚戒、甚至冲婴儿麦片。

![活动偏好分数随排名变化的衰减曲线图及典型高偏好活动列表](/coursenotes/cs231n-deep-learning-for-cv-sp25/l18-human-centered-ai/kf_0018_00-57-35.jpg)

课件给出偏好分数随活动排名衰减曲线，强调任务「源于人的需求（serves human needs, goals, values）」，高偏好前列依次：Clean after wild party、Clean a shower、Clean grease、Scrub bathroom floor、Shovel snow、Wash outside window、Clean walls、Clean up water damage。由此有了一条有原则的路径，决定该训练机器人做哪上千个任务。为支撑它，扫描／获取 50 个真实场景 3D（餐厅、公寓、杂货店、办公室），采集逾 10,000 个带 articulation、deformability 属性的 3D 资产；仿真与 Nvidia Omniverse 合作，追求物理、感知、交互高保真，考虑 thermal、transparency、deformability，用 human user study 对比 perceptual realism，含布料、液体交互。团队也做真实环境 digital twin 测试 real-to-sim transfer（仍未解）；不加加速地看，机器人清理房间很慢，还会捡不起瓶子、走错方向、放错位置。

![BEHAVIOR基准测试中不同特权信息假设下算法表现对比，无特权信息时成功率均为0.0](/coursenotes/cs231n-deep-learning-for-cv-sp25/l18-human-centered-ai/kf_0019_01-00-25.jpg)

最关键的实验结论：今天算法仍做不了 BEHAVIOR。表格顶行是真正期望的设定——不给任何特权信息、直接把机器人丢进环境做任务；用 Soft Actor-Critic RL 测 3 个 BEHAVIOR 任务，性能 $0.0 \pm 0.0$。一旦给更多 privileged information，或做 magic motion、perfect memory 等简化假设，成绩才变好。Takeaway #1 很直白：无任何特权信息（含 action primitives）的模型完全无法学习也无从泛化。讲者说只盯顶行会沮丧，但研究生反而该受鼓舞——还有巨大成长空间。这套环境也被用来在安全条件下研究视障患者。

![斯坦福 CoRL 2023 论文展示：左侧为轮椅老人厨房场景，右侧为机械臂烹饪实验及脑机接口受试者特写](/coursenotes/cs231n-deep-learning-for-cv-sp25/l18-human-centered-ai/kf_0020_01-03-08.jpg)

最后一个技术工作（R. Zhang 等，CoRL 2023）她觉得「超级酷」：与心理学家、医生合作研究用脑电波控制机器人。演示里一位研究生戴 EEG cap，仅凭电信号发指令，让机械臂纯靠「想法」做出一顿日式料理——无侵入式脑控，来自脑电信号；前提是先用 lift／place／drop 等动作预训练机械臂的「想法」，之后整顿饭基于脑波完成，发生在去年（2024）。把 vision、perception、robotics 结合、在临床帮助人，尤其帮助 severely paralyzed 患者，正是这方向的未来。结语 take-home：BEHAVIOR 始终瞄准 augmenting people——一个大规模、多样、物理与感知真实生态的基准；我们不只想要造 AI 去做事去看，更想让它帮助人，把 AI 做成 augmentation／enhancing tool for humanity，而非取代我们的工具。

## 要点回顾

- 本讲是收官 talk，不教新算法，主线是三大主题 see what humans see / don't see / want to see，落点是把 AI 做成增强而非取代人的工具。
- 第一部分：人类 $10$ Hz／$100$ ms 检测目标、$150$ ms 脑电分类、专门物体脑区，确立物体识别为核心；深度学习前经历「部件组合」与「统计机器学习」两波；Biederman number（6–7 岁约 30,000–100,000 类）催生 ImageNet（约 22,000 类、1500 万图），2012 年数据 + CNN + 两块 GPU 汇合使错误率从 $0.28$ 降到 $0.023$。
- 视觉理解不止命名：Wolfe 提出关系须被编码 → scene graph／Visual Genome／罕见组合 zero-shot；约 2014 起 CNN + LSTM 做 image／dense／paragraph captioning；dynamic scenes 与多 actor 活动理解仍未解，AI 与认知／神经科学双向启发。
- 第二部分「看见人类未见」：fine-grained categorization（4,000 只鸟、make/model/year 车型 + Street View 反推教育／收入／投票／环境模式）是超人能力；Stroop、change blindness、棋盘格错觉揭示注意力局限与视觉 bias，医疗错误为美国第三大死因，AI 可帮手术室计数纱布；人脸识别肤色／性别偏差警示算法偏见。
- 隐私维度：blurring／降维／federated learning／homomorphic encryption，及软硬件联合设计的过滤 lens「看见动作却不看见脸」；AI 既 help 也 amplify，须以多学科方式 study、forecast、guide 其社会影响。
- 第三部分从 replace 转向 augmenting：长寿社会与护士流失致人力短缺，dark spaces of health care；ambient intelligence 三例——手卫生（深度传感器 + CV，院内感染致死超车祸 3 倍）、ICU mobilization（占美国 GDP $1\%$、监测 4 类动作）、aging in place（热成像、睡眠／饮食模式）。
- 具身 AI 闭合感知—行动回路；VoxPoser（CoRL Oral 2023）用 LLM 写代码 + VLM 检测 + 3D value／affordance map 实现 open-set、zero-shot 的 in-the-wild 机器人操作。
- BEHAVIOR（CoRL 2022）以 1,400 人调查、按人的偏好定义上千家庭任务，50 个真实 3D 场景、逾 10,000 资产、与 Nvidia Omniverse 共建高保真仿真；当前 Soft Actor-Critic RL 在无特权信息时成功率为 $0.0 \pm 0.0$。
- 前沿与结语：EEG 脑电控制机械臂做饭（CoRL 2023，面向严重瘫痪患者）；BEHAVIOR 旨在 augmenting people，AI 应成为增强人类的工具而非取代。

## 视频来源

https://www.youtube.com/watch?v=g8UaBfj6Sh8
