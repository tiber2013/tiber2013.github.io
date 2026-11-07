---
title: "第 5 讲 · 数据集创建与策展"
description: "Dataset Creation and Curation · MIT DCAI IAP 2024 中文图文笔记"
---

# 第 5 讲｜数据集创建与策展：从采集偏差到多标注者共识

> **MIT Introduction to Data-Centric AI · IAP 2024 · L05** ｜ 54 分钟 ｜ [原视频](https://www.youtube.com/watch?v=wUpliyDY8_E)

## 核心结论

本讲把视线前移到「还没有数据」的阶段。全讲沿官网 notes 的三条主线展开：任务是否被恰当定义、采集时如何防住选择偏差、拿到标签后如何从多位标注者推断真相。最要紧的判断是：选择偏差一旦渗进训练集，事后几乎无法靠建模补救，必须在采集前列举成因，再用「刻意非随机」的验证集划分把差距照出来。数据量可用经验幂律外推规划；标注质量则从多数投票走到 Dawid-Skene 与 CROWDLAB，后者把分类器也当成一个标注者，用加权集成同时给出共识标签、置信度与标注者质量。

## 一、开始采集之前：三个问题与「沙滩上的牛」 [00:00 - 06:21]

前几讲处理「已到手的数据」，本讲回到源头：数据从哪来、标签从哪来、如何策展出对的数据。官网 notes 说明本讲虽以分类举例，思路也适用于回归、图像分割、实体识别。讲师强调，在还没有任何数据时就应先想清三件事。

![课程标题页：Lecture 5 Introduction to Data-Centric AI，副标题为 Dataset Creation and Curation](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0001_00-00-00.jpg)

图注：对应本讲开场，标题页点明主题是数据集的创建与策展。

![黑板上列出的机器学习评估三个关键点：用途、假设、数据问题](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0013_00-02-06.jpg)

图注：对应「采集前三问」框架，黑板列出用途、假想场景、数据问题三条要点。

一是**模型将如何使用**：对哪个人群、在什么时间点做预测，这决定该采什么数据。二是**假想场景（hypotheticals）**：哪些边缘案例、高风险场景必须判对。三是数据本身的问题：错标、离群点，以及**重复数据（duplicates）**——明明一眼能查，却最常被漏，应在项目最初就检查。

![展示机器学习识别牛与风扇干扰的对比示意图](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0022_00-03-48.jpg)

图注：对应「沙滩上的牛」案例，画面并置被认成 cow 与判成 not cow 的牛，呈现背景带偏的误判。

为说明第一问，他画了两头牛：牧场上的牛被正确识别，海滩上的牛却被判「不是牛」。症结不在模型容量，而在训练数据里牛总与草地同框，从没见过「牛 + 沙滩」。这类捷径叫**虚假相关（spurious correlation）**：训练集中成立、部署时不成立的规律。模型本质是模式检测器，讲师把它们比作「作弊者」，只找能最快压低训练损失的简单模式；文本任务里某些词因采集偶然性高频共现于某类，也是同类问题（官网 notes 归于 Beery 等，ECCV 2018）。

![黑板完整展示虚假相关性定义：ML models = shortcut cheats](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0030_00-05-12.jpg)

图注：对应「模型是走捷径的作弊者」，黑板写出 ML models = shortcut cheats，点明虚假相关的来源。

![黑板上的课程大纲列表及“Selection Bias”标题，列出了ML使用方式、假设和数据问题等要点](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0036_00-06-18.jpg)

图注：对应本段收尾，黑板写下 Selection Bias 标题，把案例引向选择偏差这一更广概念。

## 二、选择偏差：训练分布不等于部署分布 [06:21 - 12:43]

![黑板左侧列出数据问题列表（假设性、重复项），右侧为 Selection Bias 定义](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0040_00-07-00.jpg)

图注：对应选择偏差的核心定义，黑板写出训练分布不等于部署分布。

**选择偏差（selection bias）**指采集到的训练数据分布与部署时分布存在系统性偏离，官网 notes 也称**混杂（confounding）**或**分布偏移（distribution shift）**。讲师强调它一旦渗入数据与问题设定，靠建模几乎补救不回，因此要在采集前穷举成因、尽力避开。课堂当场征集，逐条写满黑板。

![黑板上完整展示了 Selection Bias 的四大成因：时间、季节性、地点和人口统计特征](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0058_00-10-18.jpg)

图注：对应本段列举的主要成因，黑板写出时间、季节性、地点、可用性、人口统计四项。

**时间**：用户过去的行为未必代表未来，零售业还有节假日等**季节性**冲击；研究自然规律时机制稳定，影响就小。**地点**即批次效应（batch effects）：数据来自某一块培养板、某一家医院，部署却面向所有医院，也可按经纬度、国家、半球划分。**可用性（availability）**：人总先采最容易拿的数据（找朋友填问卷），而「容易采」不等于「有代表性」。**人口统计（demographics）**要求数据代表最终人群，讲师举临床试验因未校正人群构成、导致疗法对某些群体次优为戒。

![讲师指向黑板上补充的稀有事件（rare-events）成因](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0063_00-11-12.jpg)

图注：对应讲师末尾补充的稀有事件成因，画面中该条目被明确新增，是长尾场景的关键一项。

还有**稀有事件 / 长尾（rare events, long tail）**：自动驾驶里某些事故几乎采不到，但车遇上时怎么表现恰恰重要。

![黑板板书展示 Selection Bias 的定义及五大成因（时间、季节性、位置、可用性、人口统计等）](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0070_00-12-30.jpg)

图注：对应转向「如何应对」，黑板写下 validation data，引出用验证数据评估、缓解偏差。

偏差未必躲得掉——有时数据是别人给的、无法再采。此时能做的转向验证数据：估计模型在偏差下会差多少，并据此决定怎么改进。

## 三、用验证集把偏差「照出来」：非随机划分 [12:43 - 20:17]

![黑板板书：选择偏差（Selection Bias）的定义及其五大成因列表](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0075_00-13-24.jpg)

图注：对应引入非随机划分，黑板列出五大成因作为设计切分策略的对照清单。

传统黄金标准是随机切分训练 / 验证；这里要**刻意非随机**地切，让切分复现你最担心的偏差：从一份训练集划出验证集、其余训练，并希望这个过程逼近「全量训练、部署时测性能」。

![黑板板书：选择偏差（Selection Bias）的定义、验证数据及五大成因列表](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0084_00-15-00.jpg)

图注：对应时间维度策略，黑板补出 validation data: most recent，把最近数据留作验证集。

**时间**：数据跨 2000–2023 年、想估 2024 年表现，就把某时点之前全作训练、之后全作验证。虽无法预知未来世界怎么变，至少能量出「训练截止某刻、评估未来」时的性能下滑。

![完整展示选择偏差成因及验证数据策略的黑板板书全貌](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0098_00-17-36.jpg)

图注：对应位置与稀有事件的综合策略，板书同时写出 most recent 与 hold out entire locations。

**位置**：留出整家医院、整个地区，甚至把整个半球从训练里切掉。**稀有事件**：在验证集**过采样**这些事件甚至全留给验证集，故意让训练集里它们更稀少、差距更大，测出模型对「事件越来越罕见」有多敏感。有学生质疑这是在扭曲数据，讲师答：正因为部署时还会遇到训练集里根本没有的新类型稀有事件，才要主动变稀看鲁棒性。他也澄清这只是**部署前内部评估**，最终模型仍在全量数据上训练。

## 四、要采多少数据：学习曲线与幂律外推 [20:17 - 31:40]

![黑板上完整展示课程目标：Goal 95% accuracy. How much data?](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0119_00-21-24.jpg)

图注：对应本章核心规划问题，黑板写下 Goal 95% accuracy. How much data?

接着问数据量：业务要 95% 准确率（语音识别往往要 99.9%），采多少条才够？这直接决定成本与周期。策略是：已有样本量 $n$ 的训练集与一份固定验证集，取子集规模 $n_j=0.1n,\dots,1.0n$，每档随机抽（最好按类分层）、训模型、在同一验证集记准确率 $a_{ij}$，得到若干 $(n_j,a_{ij})$；因 SGD 与子集抽样噪声大，每档可重复多次，小数据集可用交叉验证让 $a_{ij}$ 更稳定。

![准确率随样本量变化的曲线草图](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0126_00-22-42.jpg)

图注：对应学习曲线，草图画出准确率随样本量上升的趋势与当前成绩到 95% 目标线的缺口。

画成图即**学习曲线（learning curve）**：横轴样本量、纵轴准确率，看还差多少数据补齐缺口。

![黑板上完整展示的误差与样本量对数关系的线性公式](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0149_00-26-48.jpg)

图注：对应幂律外推公式，黑板写出 log(error) = -a·log(n) + b，即双对数坐标下误差与样本量线性相关。

关键是**外推**到远超现有的 $n'$。讲师反问「用 KNN 或线性回归行不行」：外推区几乎没有数据，纯黑箱拟合必然不可靠（KNN 不会超出见过的范围），必须注入一条外部知识，即经验**幂律（power law）**：

$$\log(\text{error})=-a\cdot\log(n)+b$$

error 可视为 $1-\text{accuracy}$ 或 log loss；用观测最小化均方误差估出 $a,b$，代入 $n'$ 即知「还差几倍数据」或「曲线是否根本碰不到目标」。

![黑板上展示样本均值方差等于总体方差除以样本量的结论](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0165_00-29-42.jpg)

图注：对应幂律的统计学支撑，黑板写出 E(μ̂−μ)² = σ²/n，即样本均值的均方误差为总体方差除以样本量。

式子从哪来？回到均值估计：$X_i\sim N(\mu,\sigma^2)$、$\hat\mu=\frac1n\sum X_i$，其均方误差期望恰是 $\sigma^2/n$，取对数正是 $-\log n+$ 常数。机器学习本质在估计各子人群的**条件均值**，故同一形式反复出现。局限：选择偏差会令曲线失真；**维度灾难（curse of dimensionality）**下维数越高常数越差、曲线变「慢」但关系仍成立；「固定设计」数据里连「三倍数据」都无从定义。学生问何时不成立，讲师答：只要数据随机采，据他所知基本都成立。

## 五、标签从哪来：众包标注矩阵与三个目标 [31:40 - 40:21]

![黑板上展示了完整的四行三列分类矩阵，包含人物、汽车等图标及对应标签](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0191_00-34-30.jpg)

图注：对应众包标注矩阵，画面是三位标注者对四张 person / car 图的结果表，含空标签。

接着拿标签：分类、以及生成式 AI 里用人类反馈判断输出好坏（RLHF）都属此类，现实做法是雇一批标注者。黑板矩阵是 $A_1,A_2,A_3$ 对四张自动驾驶图的标注，只有 person / car 两类，例如 $A_1$：person、car、$\emptyset$、car；$A_2$：person、$\emptyset$、person、car；$A_3$：$\emptyset$、$\emptyset$、car、car。$\emptyset$ 表示预算有限、这张图没派给该标注者。

![黑板上展示了标注者一致性矩阵及“真实推理”的三个核心目标（共识标签、置信度、标注者质量）](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0207_00-37-24.jpg)

图注：对应三大推理目标，黑板在 "true inference" 下列出共识标签、置信度、标注者好坏三问。

要答三问，讲师称**真实推理（true inference）**：每张图的**共识标签（consensus label）**、对该标签的**置信度（confidence）**、以及**哪些标注者好 / 差**。三人一致说 car 的图，显然比两人分歧或只有一人标过的更可信。

![黑板上完整的标注矩阵定义，包含 y_ij ∈ {∅, 1, 2, ..., K} 及右侧三个核心问题](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0222_00-40-06.jpg)

图注：对应形式化记号，黑板写出 y_ij ∈ {∅, 1, ..., K} 并标出 example / annotator。

形式化地 $Y_{ij}\in\{1,\dots,K\}\cup\{\emptyset\}$。坑也不少：任务**欠定义**（花大钱却没给标注者合适的界面）、**没有真值**（不知谁靠谱）、Mechanical Turk 上有人为拿钱一律点 car。金标准是掺入**质量控制样本（gold examples）**：每约 100 条放一条已知答案的题，谁总答错就剔除。但讲师直说这很浪费，让所有人标所有图同样浪费——最省预算的是让**尽可能少的标注者**标每张图，同时仍对共识有信心，这正是后面算法要解的题。

## 六、多数投票与标注者一致性：能做什么、不能做什么 [40:21 - 46:32]

![黑板上完整的 Majority Vote 中 Agreement_i 的数学定义公式](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0235_00-42-30.jpg)

图注：对应多数投票的一致性指标，黑板写出 Agreement_i 定义与分母集合 J_i。

最直白的是**多数投票（majority vote）**：在 $\mathcal J_i=\{j:Y_{ij}\ne\emptyset\}$ 里取票数最多的类 $\hat Y_i$，用**一致性（agreement）**估置信度：

$$\text{Agreement}_i=\frac{1}{|\mathcal J_i|}\sum_{j\in\mathcal J_i}\big[Y_{ij}=\hat Y_i\big]$$

![黑板上完整的 Majority Vote 公式组及左侧误差曲线图](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0248_00-44-48.jpg)

图注：对应标注者质量，黑板给出含 I_{j,+} 的 Quality_j 公式组。

标注者 $j$ 的质量看他有多少标注与共识一致：

$$\text{Quality}_j=\frac{1}{|\mathcal I_{j,+}|}\sum_{i\in\mathcal I_{j,+}}\big[Y_{ij}=\hat Y_i\big]$$

其中 $\mathcal I_{j,+}=\{i\in\mathcal I_j:|\mathcal J_i|>1\}$ 只保留**至少还有别人也标过**的样本——否则只有他一人标注时他自己就是「多数」，等于白送一分。

![黑板右侧展示了标注示例（gold example），包含汽车、行人等图标及对应的标签](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0255_00-46-06.jpg)

图注：对应讨论局限时的具体案例，画面指向 gold example 标注示例，说明单标注与平票样本难估置信度。

缺点随即被点出：**平票无定义**、**好差标注者权重相同**（医学图像里实习生与专家不该等价）、**单标注样本无法估置信度**，且人越少平票越多，标注者还可能串通。可见好的协议除了「分歧多大」，还该把「标注者多好」「多少人标了」算进去。

## 七、Dawid-Skene 与 CROWDLAB：把模型也当成一个标注者 [46:32 - 54:29]

![标注数据示例图，展示了不同标注者对同一图片（车/人）的标签差异及符号定义](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0259_00-46-48.jpg)

图注：对应从打破平局切入的直觉，画面回到标注示例与 y_ij 记号，点明平票与单人标注难题。

第一步补强是引入训练好的分类器 $M$：平票时用它的概率预测打破僵局；只有一条标注的样本也能拿到第二份信息——数据集里大量相似的车被可靠标注者标过，模型可沿特征空间泛化，越自信说明见过越相似的样本。

![黑板上写出训练机器学习模型M的假设分布公式 P_M = P̂_M(x_i | X_i)](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0267_00-48-18.jpg)

图注：对应「模型也当标注者」的起点，黑板写出概率预测 P_M = P̂_M(x_i | X_i)。

**Dawid-Skene** 走统计生成过程：给标注者 $j$ 一个 $K\times K$ 混淆矩阵 $\mathbf C_j^{a,b}=p(\text{标成 }b\mid\text{真类 }a)$，配 Dirichlet 先验用 EM / MAP 估计，质量取 $\mathrm{Tr}(\mathbf C_j)$；加类别先验 $\pi_Y$ 后真标签后验 $\Pr(Y_i=k)\propto\pi_Y\prod_{j\in\mathcal J_i}\mathbf C_j^{k,Y_{ij}}$，共识取 $\arg\max_k$。短板：忽略样本特征、救不了单人标注样本，且每位要估整整一个 $K\times K$ 矩阵、数据少时统计上很难。

![黑板上展示了训练机器学习模型M的公式及CROWLAB标题](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0283_00-51-12.jpg)

图注：对应「每位标注者也是概率预测器」，黑板同时写出 P_m、P_j 并引出 CROWLAB 标题。

**CROWDLAB（Classifier Refinement Of croWDsourced LABels）** 把每个标注者也「伪装」成输出类别概率向量 $p_j$ 的模型，再与真实分类器做**加权集成（weighted ensembling）**：

![黑板上展示训练ML模型M及CROWLAB共识公式的推导过程](/coursenotes/mit-dcai-iap-2024/l05-dataset-creation-and-curation/cand_0294_00-53-12.jpg)

图注：对应 CROWLAB 核心共识公式 P(Y_i|X_i) = w_m·P_m + Σ w_j·P_j → Consensus。

$$\Pr(Y_i)=\frac{1}{Z}\Big(w_M\cdot p_M+\sum_{j\in\mathcal J_i}w_j\cdot p_j\Big),\qquad Z=w_M+\sum_{j\in\mathcal J_i}w_j$$

$w_M,w_j$ 是全局共享权重，官网 notes 用标注者一致率 $s_j$、分类器相对多数投票的准确率 $A_M$、以及「永远预测最常见类」的基线 $A_{\text{MLC}}$ 做基线归一化估出；$p_j$ 用共享参数 $P$（平均一致率）构造：所选类给 $P$，其余 $K-1$ 类各 $(1-P)/(K-1)$。共识与置信度仍从 $\Pr(Y_i)$ 读出。直觉是标的人越多越靠一致性、只有一两个标注就越靠能泛化的分类器；权重还参考分类器置信度与标注者偏离共识的程度。讲师提了个边界：若某样本上百人标注，求和项会压过只有一个的模型，故权重须自适应（方法见 Goh 等，NeurIPS 2022）。本讲 lab 是分析一份多标注者真实数据集，把这些估计跑一遍。

## 要点回顾

- 采集前想清三件事：模型对谁 / 何时预测、哪些高风险场景必须判对、数据里的错标 / 离群点 / **重复数据**。
- 「沙滩上的牛」说明模型抓**虚假相关**走捷径；训练与部署分布不一致即**选择偏差**，几乎无法靠建模补救，须采集前穷举成因。
- 成因：时间 / 季节性、位置与批次效应、可用性、人口统计、稀有事件。
- 用**非随机**验证集划分量化偏差：留最近时段、留整个医院 / 地区、验证集过采样稀有事件；这是部署前内部评估。
- 数据量：固定验证集、按 $0.1n\dots1.0n$ 子集重复训模型取点，按 $\log(\text{error})=-a\log(n)+b$ 外推，直觉来自 $\mathrm{MSE}=\sigma^2/n$；外推不能靠黑箱，选择偏差 / 维度灾难 / 固定设计会破坏曲线。
- 众包记号 $Y_{ij}\in\{1..K\}\cup\{\emptyset\}$、$\mathcal J_i=\{j:Y_{ij}\ne\emptyset\}$；三目标是共识标签、置信度、标注者质量。
- 多数投票：Agreement 是与共识一致比例，Quality 只在 $|\mathcal J_i|>1$ 上算；缺点是平票、好坏等价、单标注无置信度。
- Dawid-Skene 用 $K\times K$ 混淆矩阵 + EM；CROWDLAB 把分类器当额外标注者做加权集成，标注多靠一致性、标注少靠模型泛化。

## 视频来源

https://www.youtube.com/watch?v=wUpliyDY8_E
