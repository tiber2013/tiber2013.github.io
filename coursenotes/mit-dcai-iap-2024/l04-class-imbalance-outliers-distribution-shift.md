---
title: "第 4 讲 · 类别不平衡、离群值与分布偏移"
description: "Class Imbalance, Outliers, and Distribution Shift · MIT DCAI IAP 2024 中文图文笔记"
---

# 第 4 讲｜类别不平衡、离群值与分布偏移

> **MIT Introduction to Data-Centric AI · IAP 2024 · L04** ｜ 50 分钟 ｜ [原视频](https://www.youtube.com/watch?v=9oXniqJKTqQ)

## 核心结论

本讲处理真实机器学习数据里三个高频、却常被入门课忽略的问题：类别不平衡、离群值与分布偏移，它们「至少一个、往往多个」出现在几乎每一个真实任务中。应对类别不平衡的第一步不是换模型，而是选对评估指标——标准准确率（Accuracy）在极度不平衡下会失灵，需按业务中「漏报 vs 误报」的代价用精确率（Precision）／召回率（Recall）或 F-beta 分数衡量。训练侧可用过采样、样本权重、欠采样、SMOTE 与平衡小批量训练，且必须先分层划分、只在训练集上加工以避免数据泄漏。离群点先辨成因与问题设定，再用 Tukey's fences、Z-score、孤立森林、KNN 距离或自编码器重构损失打分后阈值化，以 ROC／AUROC 评估。分布偏移指训练与测试联合分布不一致，分协变量、概念、标签偏移三类，视频仅作引入，系统内容见官网 notes。

## 一、导论：真实世界数据的三个「隐疾」 [00:00 - 01:19]

开场说明本讲全程黑板板书、鼓励随时举手或直接喊出答案，是一堂高度互动的课。今天谈三个在真实机器学习任务里反复出现、却很少被导论课（如 MIT 6.036）系统覆盖的主题：类别不平衡、离群值与分布偏移。现场调查显示多数学生只接触过 domain shift（域偏移），另外两个话题几乎没讲过。讲师强调：这三个问题「至少一个、往往多个」出现在几乎每一个真实任务里，因此值得用以数据为中心（data-centric）的视角系统研究。

![课程标题页：Lecture 4 Introduction to Data-Centric AI，副标题涵盖类别不平衡、离群值与分布偏移](/coursenotes/mit-dcai-iap-2024/l04-class-imbalance-outliers-distribution-shift/cand_0001_00-00-00.jpg)

图注：对应本段「本讲三大主题的引入」——标题页副标题即类别不平衡、离群值与分布偏移三个核心概念。

## 二、类别不平衡（Class Imbalance） [01:19 - 06:45]

类别不平衡指二分类或多分类任务中，数据集与部署场景里各类别「占比（prevalence）」相差悬殊；多分类同样会失衡。最经典的例子是信用卡欺诈检测：信用卡网络日交易量以十亿计，但欺诈只占约 0.2%、非欺诈约 99.8%。学生提出的「预测谁会得诺奖」也可框定成不平衡分类；而把深肤色人误判为大猩猩的事故更接近分布偏移，留待后面再谈。

![黑板展示信用卡欺诈检测中欺诈与非欺诈样本的比例（0.2% vs 99.8%）](/coursenotes/mit-dcai-iap-2024/l04-class-imbalance-outliers-distribution-shift/cand_0025_00-04-00.jpg)

图注：对应本段「类别不平衡的核心案例」——黑板展示信用卡欺诈检测中 fraud 0.2% 与 not-fraud 99.8% 的悬殊比例。

**准确率为何失灵**：若用准确率评估，一个「永远预测非欺诈」的分类器即可拿到 99.8% 准确率，数字漂亮却毫无用处；医学影像／罕见病筛查里「永远预测无病」同理。更关键的是，评估指标会反过来决定损失函数与训练方式，指标高分不等于得到想要的模型。因此不存在放之四海皆准（one-size-fits-all）的指标，选择取决于具体领域与「哪类错误可容忍」。官网 notes 还补充了 COVID 感染约 10% 阳性、制造业缺陷分类、自动驾驶目标检测（轿车／卡车／行人占比不同）等实例；课堂上则列到医学影像与垃圾邮件分类（讲师猜测绝大多数邮件是垃圾邮件，只是过滤器已把它们拦在收件箱之外）。

![黑板上书写了类别不平衡（Class Imbalance）的定义及信用卡欺诈、医疗影像等实例](/coursenotes/mit-dcai-iap-2024/l04-class-imbalance-outliers-distribution-shift/cand_0035_00-05-42.jpg)

图注：对应本段「类别不平衡的定义与更多实例」——黑板完整列出定义及信用卡欺诈、医疗影像、垃圾邮件分类等典型案例。

## 三、评估指标：从 Precision / Recall 到 F-beta [06:45 - 15:10]

二分类结果分四种：真正例（True Positive, TP）、假正例（False Positive, FP）、真反例（True Negative, TN）、假反例（False Negative, FN）。精确率（Precision）问「判为正例里有多少真相关」：

$$\text{Precision}=\frac{TP}{TP+FP}$$

![黑板上展示了完整的Precision公式 TP/(TP+FP)，左侧列出了信用卡欺诈等不平衡数据示例](/coursenotes/mit-dcai-iap-2024/l04-class-imbalance-outliers-distribution-shift/cand_0054_00-08-54.jpg)

图注：对应本段「Precision 的数学定义」——黑板给出 $TP/(TP+FP)$ 公式，左侧仍列着不平衡数据的三个示例。

召回率（Recall）问「所有真相关里找出多少」：

$$\text{Recall}=\frac{TP}{TP+FN}$$

![黑板补全Recall公式TP/(TP+FN)及两个评估指标定义](/coursenotes/mit-dcai-iap-2024/l04-class-imbalance-outliers-distribution-shift/cand_0060_00-09-54.jpg)

图注：对应本段「Recall 的引入与定义」——黑板补全 $TP/(TP+FN)$，与 Precision 并列展示两个核心公式。

**由业务代价决定取舍**：偏重 Recall 的是疾病／COVID 检测——误判阳性只是居家几天，漏掉真阳性却可能带病上班传染他人；偏重 Precision 的是自动找代码 bug——每次报警都要开发者复核、推理并修复，误报会浪费大量人力。官网 notes 建议信用卡欺诈用 Precision 与 Recall 的加权平均，权重由「未拦欺诈」与「错拦正常交易」的相对成本决定。

F-score 是二者的调和平均（harmonic mean）；F-beta 分数用参数 $\beta$ 平滑权衡：

$$F_\beta = \left(1+\beta^2\right)\cdot\frac{\text{precision}\cdot\text{recall}}{\beta^2\cdot\text{precision}+\text{recall}}$$

$\beta=1$ 退化为 $F_1$；$\beta>1$ 偏重 Recall，$\beta<1$ 偏重 Precision。次序始终是：**先定义评估指标，再训练模型并用它检验**。

![黑板上完整展示的 F-score 计算公式](/coursenotes/mit-dcai-iap-2024/l04-class-imbalance-outliers-distribution-shift/cand_0088_00-14-36.jpg)

图注：对应本段「F-beta 分数公式及其作用」——黑板完整展示含 $\beta$ 的 F-score 计算式，用于平滑权衡精确率与召回率。

## 四、在不平衡数据上训练模型 [15:10 - 19:57]

官网 notes 的原则：先在真实分布上标准训练；若在与真实分布一致的留出测试集（held-out test set）上指标已够好即完成，否则才用下列技术提升少数类表现。

**前提：分层划分（Stratified data splitting）与数据泄漏（data leakage）。** 划分训练／测试集须保持各类别原始比例，否则等于人为制造分布偏移；过采样等一切加工只能在训练集上做，切不可把测试样本卷入。课堂上有学生专门追问过采样会不会造成同一条样本同时出现在训练集和测试集，讲师确认要先做分层切分、只在训练集里复制。

课上三种直观方法，官网 notes 补两种进阶技术：

1. **过采样（Over-sampling）**：复制少数类样本，让模型在该类上表现更好。

![黑板上完整的F-score公式及不平衡数据训练策略Over-sampling](/coursenotes/mit-dcai-iap-2024/l04-class-imbalance-outliers-distribution-shift/cand_0098_00-16-18.jpg)

图注：对应本段「过采样这一第一种训练策略」——板书在 F-score 公式下列出 Over-sampling。

2. **样本权重（Sample weights）**：把均匀平均损失改为加权平均 $L=\sum_i w_i L_i$，抬高少数类权重；许多算法原生支持逐样本权重。notes 指出它概念简单但实践常不佳：mini-batch 下有效学习率在批次间波动，学习会不稳。

![讲师指向黑板，展示样本权重的损失函数公式 L=Σw_iL_i](/coursenotes/mit-dcai-iap-2024/l04-class-imbalance-outliers-distribution-shift/cand_0111_00-18-30.jpg)

图注：对应本段「样本权重的损失函数」——黑板展示 $L=\sum w_i L_i$，即用逐样本权重加权少数类。

3. **欠采样（Under-sampling）**：删除多数类样本使数据更均衡。notes 说这反直觉却常有效；讲师也提到实践中过采样可能略不稳定、某些场景欠采样反而更优，但在欺诈这种极度不平衡的数据集上，为凑平衡要丢掉海量正常交易，需格外小心。

![黑板完整展示F-score公式及处理不平衡数据的三种方法：过采样、样本权重、欠采样](/coursenotes/mit-dcai-iap-2024/l04-class-imbalance-outliers-distribution-shift/cand_0115_00-19-06.jpg)

图注：对应本段「三种训练方法的汇总」——黑板同时列出过采样、样本权重、欠采样，作为本章知识点的总结图。

notes 补充的进阶技术是 **SMOTE**（Synthetic Minority Oversampling TEchnique，用增强合成而非复制少数类新样本，仅当特征空间插值有意义时适用——平均两张狗图的像素得不到狗）与**平衡小批量训练**（Balanced mini-batch training，组 batch 时以更高概率抽少数类使类别均衡，不丢数据）。技术可组合，SMOTE 作者称 SMOTE + 欠采样优于单纯欠采样；工具见 imbalanced-learn。

## 五、异常值（Outliers）：定义、成因与处置 [20:11 - 26:20]

**定义**：分布尾部（tail end of distribution）的数据点，或「与其他点显著不同」的点。二维特征空间 $(x_1,x_2)$ 里人眼一眼能挑出离群点，但本课要的是系统化、可自动化的算法，而非人工翻遍海量数据去挑坏样本。

![黑板上关于离群点定义的完整文字及二维坐标系示意图](/coursenotes/mit-dcai-iap-2024/l04-class-imbalance-outliers-distribution-shift/cand_0129_00-21-30.jpg)

图注：对应本段「离群点的两个直观定义」——黑板写出「分布尾部」「显著不同于其余」并配二维坐标系示意。

![讲师在二维坐标系中绘制数据点以演示离群点分布](/coursenotes/mit-dcai-iap-2024/l04-class-imbalance-outliers-distribution-shift/cand_0130_00-21-42.jpg)

图注：对应本段「特征空间中的离群点演示」——讲师在 $(x_1,x_2)$ 坐标系里画点，直观指出哪个是离群点。

**为何关注**：异常值会扭曲分布、损害训练与推理，也让统计分析失真（官网 notes 补充 vanilla SVM 对训练集异常值尤其敏感，部署时喂入异常输入本身即一种分布偏移）。**成因**包括：坏数据录入／标注疲劳出错（标了一千张图后开始手滑）、测量误差、罕见但真实的样本、坏传感器读数（测量仪器损坏或读数飘忽）、恶意输入（对抗样本 adversarial examples）。

![黑板板书：离群点（Outliers）的定义及具体例子（错误数据、罕见样本、传感器读数错误）](/coursenotes/mit-dcai-iap-2024/l04-class-imbalance-outliers-distribution-shift/cand_0154_00-25-42.jpg)

图注：对应本段「离群点的成因与例子」——黑板列出 bad data entry、rare/unusual examples、bad sensor readings 三类来源。

**处置**依类型而定、不可一律丢弃：标注错误可修正；坏传感器读数无法回溯重测，多半删除；罕见真实样本反而宝贵，应确保模型正确处理（呼应前几讲的红绿灯例子），官网 notes 建议由领域专家（domain expert）人工复核。视角上，6.036 用以模型为中心（model-centric）的办法（用 L1 loss 替代 L2 loss 降低敏感度），本课聚焦「如何识别」。

## 六、两种问题设定：离群点检测 vs 异常检测 [26:20 - 31:46]

两种设定都围绕离群点，但问题结构不同：

- **离群点检测（Outlier detection）**：单个无标签数据集，没有「干净／分布内」参照，找出与其余不同的点。
- **异常检测（Anomaly detection）**：给定只含分布内样本的参照集 $D=\{x_i\}$，判断新数据点 $x^*$ 是否来自同一分布。

![黑板上关于异常检测（Anomaly detection）和离群点检测（Outlier detection）的定义与实例列表](/coursenotes/mit-dcai-iap-2024/l04-class-imbalance-outliers-distribution-shift/cand_0160_00-26-42.jpg)

图注：对应本段「两种问题设定的定义」——黑板并列写出 Outlier detection 与 Anomaly detection 及基础例子列表。

![讲师指向黑板上新增的异常检测数据集符号 D={x_i} 和 x*](/coursenotes/mit-dcai-iap-2024/l04-class-imbalance-outliers-distribution-shift/cand_0162_00-27-06.jpg)

图注：对应本段「异常检测的形式化设定」——黑板新增数据集符号 $D=\{x_i\}$ 与新数据点 $x^*$。

**转换**：把新样本并入已有数据一起交给离群点检测算法，若它被标为离群点即为异常，否则不是——两种任务只差这样一步简单的变换。例子方面，离群点检测是从图像数据集里挑出模糊坏图扔掉（image dataset cleaning）；异常检测则有抄袭检测（教授已认定一批作业未抄袭，判断新提交是否属「非抄袭分布」）与工业缺陷监控（只有良品，监测未来是否出现异常）。

![讲师在黑板上补充了离群点检测的例子：图像数据集（image dataset）。](/coursenotes/mit-dcai-iap-2024/l04-class-imbalance-outliers-distribution-shift/cand_0175_00-29-12.jpg)

图注：对应本段「离群点检测的具体应用」——黑板补充 image dataset 作为离群点检测的实例。

## 七、异常值检测算法 [31:55 - 48:13]

共同范式（官网 notes）：给每个样本打分（score）再阈值化选异常值，方法之间用 ROC 曲线，或单一数值 AUROC 来比较。先从一维实数数据热身。

**Tukey's fences**（统计学家 John Tukey，1977 年前后）：取 $Q_1,Q_3$，定义四分位距（Interquartile Range, IQR）$\text{IQR}=Q_3-Q_1$，向外扩 $k=1.5$ 倍为「分布内」边界：

$$[\,Q_1-1.5\,\text{IQR},\quad Q_3+1.5\,\text{IQR}\,]$$

界外判为异常。讲师坦言没有严格科学依据证明它一定准，但简单场景下意外有效。

![Tukey's Fences示意图，在数轴上标出Q1和Q3分位点的位置](/coursenotes/mit-dcai-iap-2024/l04-class-imbalance-outliers-distribution-shift/cand_0198_00-33-06.jpg)

图注：对应本段「Tukey's Fences 方法」——数轴上标出 $Q_1$、$Q_3$ 两个分位点，用于定义 IQR 与 $1.5\times\text{IQR}$ 的判定边界。

**Z-score**：$z_i=\dfrac{x_i-\mu}{\sigma}$，绝对值越大越可疑；高斯假设下常取 $z_{thr}=3$，即三个标准差之外为异常，可逐特征使用。**孤立森林（Isolation Forest）** 与决策树（decision tree）相关，简洁到可在 lab 里从零实现：递归随机选特征、随机选切分值二分子集，直到只剩单个样本（板书出现 $X_2<0.4$、$X_1<0.2$、$X_1<0.5$ 等切分，各分支无需一致）；**分数 = 孤立该点所需深度／分裂次数**：离群点两三层即被单独隔离，密集区域的点要深入很多层。因为随机，多次运行汇总更稳。

![黑板上展示了Tukey's Fences的IQR图示、Z-score公式及正态分布曲线，以及Isolation Forest标题](/coursenotes/mit-dcai-iap-2024/l04-class-imbalance-outliers-distribution-shift/cand_0211_00-35-18.jpg)

图注：对应本段「Z-score 公式与孤立森林的引入」——黑板展示 $Z_i=(x_i-\mu)/\sigma$、$3\sigma$ 正态曲线，并出现 Isolation Forest 标题。

**KNN 距离（KNN Distance）**：分布内样本离邻居近，离群点远；取 $k$ 个最近邻（K-Nearest Neighbors, KNN）的平均距离作分数，板书取 $k=3$：

$$\text{score}(x)=\frac{1}{k}\sum_{x_j\in\mathcal N_k(x)} d(x,x_j)$$

![KNN距离示意图完成状态，展示了测试点到最近邻的连线](/coursenotes/mit-dcai-iap-2024/l04-class-imbalance-outliers-distribution-shift/cand_0249_00-41-42.jpg)

图注：对应本段「KNN 距离检测方法」——展示 $K=3$ 时测试点到最近邻的连线示意，密集处距离小、离群处距离大。

**高维陷阱**：直接拿图像像素算距离不可靠，解法是先映射到低维语义表示（embedding）——用自编码器编码，或取图像预训练模型倒数第二层（penultimate layer）输出，再在嵌入空间做 KNN；讲师称「embeddings + KNN distance」是很好的默认方法。**基于重构：自编码器（Autoencoder）**：编码器（encoder）把高维输入压成低维潜在表示（latent representation），解码器（decoder）重建回原输入，二者联合训练；板书以 MNIST 为例，约七百多维像素 → 约 10 维向量 → 重建图。低维瓶颈迫使模型只学数据分布主结构（是哪个数字、笔迹风格），无法编码任意像素，故输入手写数字重建几乎一致，输入分布外的笑脸则明显变形。用**重构损失（reconstruction loss）**如 L2 像素距离打分：分布内误差小、分布外误差大。课堂还讨论它与训练损失的关系，并指出自编码器可无标签训练。

![黑板上展示了KNN距离示意图（k=3）和自编码器的重建损失结构图](/coursenotes/mit-dcai-iap-2024/l04-class-imbalance-outliers-distribution-shift/cand_0275_00-46-06.jpg)

图注：对应本段「自编码器重构损失方法」——黑板展示 Encoder-Decoder 结构与 Reconstruction Loss，用作异常分数。

如何选择：理想是拿已知 ground truth 的一小部分数据做 benchmark，否则参考他人在相似数据上的评测；Tukey's fences、Z-score 太简单未必适用，孤立森林与 KNN 类更通用。本讲 lab 提供「狗图 + 非狗异常」数据集与带 ground truth 的评测脚手架，鼓励实现并比较各方法。

## 八、分布偏移（Distribution Shift） [48:28 - 48:33]

视频里它作为三大主题在开场被点名，中间以深肤色误判、数据本身发生变化等例子穿插，结尾讲师因课时有限明确宣布「今天没有足够时间完整讲分布偏移」，留到后续讲座。本节主体依官网 notes 补全，视频本身只作引入。

**定义**：训练与测试阶段的联合分布不一致：

$$p_{\text{train}}(\mathbf x, y) \neq p_{\text{test}}(\mathbf x, y)$$

官网 notes 指出，它不同程度存在于几乎每个实际 ML 应用，因为训练时难以完美复现测试条件；极端例子是拿阿拉伯数字 0–9 训练的分类器去识别罗马数字。三种类型为：协变量／数据偏移（Covariate shift / Data shift）改变 $p(\mathbf x)$、保持 $p(y\mid\mathbf x)$；概念偏移（Concept shift）改变 $p(y\mid\mathbf x)$、保持 $p(\mathbf x)$，是最难检测纠正的一类；先验概率／标签偏移（Prior probability shift / Label shift）改变 $p(y)$、保持 $p(\mathbf x\mid y)$，只出现在 $y\to\mathbf x$（如疾病导致症状）的问题里，可视为协变量偏移的逆。检测靠监控模型性能与数据随时间的变化，应对总体靠修正数据并重新训练，有时最佳方案是收集更好的训练集。这也回扣前文：分层划分保证 train/test 分布一致，正是为避免「自己制造」分布偏移。

## 要点回顾

- 类别不平衡、离群值、分布偏移是真实 ML 高频问题，本讲用 data-centric 视角处理「数据」而非「模型」。
- 极不平衡下准确率失灵（永远预测多数类即 99.8%），须先按业务代价定义评估指标。
- Precision 看报警准不准、Recall 看正例漏没漏；F-beta 用 $\beta$ 权衡，$\beta=1$ 即 F1，$\beta>1$ 偏 Recall、$\beta<1$ 偏 Precision。
- 训练侧五技术：样本权重、过采样、欠采样、SMOTE、平衡小批量；先分层划分，加工只在训练集，避免数据泄漏。
- 异常值成因分坏数据／测量误差／罕见真实样本／坏传感器／恶意输入，处置按类型区分、最好由领域专家复核。
- Outlier detection 是「一堆数据找不同」，Anomaly detection 是「有干净参照判新样本」，后者可并入数据用前者解决。
- 检测多为「打分 + 阈值」：一维用 Tukey's fences（$1.5\times\text{IQR}$）与 Z-score（$|z|>3$），通用用孤立森林、KNN 距离（配 embedding）、自编码器重构损失；评估看 ROC／AUROC。
- 分布偏移满足 $p_{\text{train}}\neq p_{\text{test}}$，分协变量／概念／标签偏移；概念偏移最难，应对靠修正数据重训。
- 本讲 lab 用「狗图 vs 非狗」基准化比较各检测方法；分布偏移因课时有限留待后续讲座，后续内容转向数据集策展与多标注者。

## 视频来源

https://www.youtube.com/watch?v=9oXniqJKTqQ
