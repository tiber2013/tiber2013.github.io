---
title: "第 9 讲 · 缩放定律"
description: "Scaling law 的核心是用 log-log 图上的幂律关系描述“资源投入—损失下降”的规律，从而用小规模实验预测大规模训练结果。数据缩放是最自然、最干净的起点：损失随数据量多项式下降，但神经网络的指数远慢于均值估…"
---

# 第 9 讲 · 缩放定律

> **Stanford CS336 语言模型从零构建 · L09** ｜ 讲者 Stanford Online ｜ 78 分钟 ｜ [原视频](https://www.youtube.com/watch?v=Q15rhEWZPQ4)

> **v2 详写版** · 按 20 分钟分段详写，配图覆盖全片；文末附「对照参考笔记补遗（Spring 2025 版）」，记录与 2025 版讲义笔记的核对结果。

## 核心结论

本讲把 scaling law 当作规避大规模训练试错的方法：一次大 run 耗资可达数百万美元，故先在小模型上调好架构、超参、规模与数据量，再外推到大尺度，但讲者提醒它更像信念而非已证的自然定律。其经验基础是资源与损失在双对数图上的幂律关系，计算量、数据量、参数量都能线性化测试损失，数据缩放最干净，按 $L(D)\propto D^{-\alpha}$ 下降，只是神经网络的指数远慢于均值估计的 $-1$，更像高维非参数估计，这类形式可上溯到泛化界、1993 年学习曲线与 2017 年观察，多是曲线拟合，理论只给候选形式。推到工程实践后，数据组成、重复、过滤与架构、优化器、批大小、学习率、MoE 稀疏度等干预大多只改截距不动斜率，且可预测性是工程化出来的，依赖选对横轴、统一参数计数、设对超参与区间，上游困惑度也未必能迁移到下游。最著名的应用是固定算力下联合优化参数量与数据量，Kaplan 与 Chinchilla 处方的差异主要源于参数计数、warm-up 与批大小等细节，说明 scaling law 更像对给定配方的条件预测，再因生产环境推理成本占主导，前沿模型反而主动过训练小模型，全讲交付的是一套证据驱动、可在训练前预测超参数效应的方法论与 isoflops 工具。

## 课程定位与工程动机：一万块 B200 的取舍 [00:00 - 03:59]

本讲（Lecture 9: Scaling Laws，Spring 2026）暂时离开 systems（系统）主题，回到深度学习一侧，用两讲的篇幅讨论 scaling law（缩放定律）及其周边现象。讲者开篇就把课程节奏交代清楚：今天是基础版，讲经典工作、以及 scaling law 的基本想法怎样与 machine learning 101（机器学习入门）里的概念挂钩；两讲之后还有一版进阶内容，会深入现代开放模型技术报告、参数初始化与优化器这类更贴近前沿的材料。由于排课约束，下一讲会先讲 inference（推理），于是课程顺序变成 scaling laws → inference → scaling laws 来回切换。理解这一点很重要：本段不是孤立的一课，而是「基础 scaling law」这一支的起点，后文所有推导都默认读者会带着系统课的背景回来，而不是把本讲当成关于大模型的全部结论。

真正把问题逼到眼前的，是幻灯片上那个具体场景：Your friend has given you ten thousand B200s for a month, and asked you to build a good open source LM（你的朋友给你一万块 B200，用一个月，要你造一个像样的开源语言模型）。讲者随即列出 What do you do?（你该怎么办？）之下的三个动作，幻灯片上写着：Put together your infra team and distributed training framework (A2)；Put together a great pretraining dataset (A4)；Run a big model (but which one??) <- we are here. 这里 A2、A4 是本课程前面两次作业（assignment）的编号，分别对应分布式训练框架与预训练数据；第三条的「<- we are here」明确把本讲钉在「到底训练哪一个大模型」这个决策点上。前两项可以照抄文献里的 adopted best practices（已被采纳的最佳实践）——讲者甚至说这些选择 pick some of these out of a hat（随便抽几个）也无妨——但一旦你处在 frontier（前沿）、想做出比现有最好模型更强的东西，就不能只复制别人的超参数与架构，因为复制最好只能追平，不可能超越。于是问题被压缩成一句话：在大模型训练里，architecture（架构）、hyper parameters（超参数）、模型规模与数据量这些 choices（选择），到底该怎样认真优化。

![一万块 B200 的场景页：列出 infra 团队(A2)、预训练数据集(A4) 与「训练哪个大模型」三个待办](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0002_00-01-49.jpg)

约束在于成本。一次大 run 的开销可能达到 millions of dollars（数百万美元）量级，而要在这样一次运行上直接调架构、超参数、模型规模与数据量，几乎等于用最贵的方式试错，讲者直言这 very scary（很吓人）、very wasteful（很浪费）。scaling laws 在这里被定义为 simple, predictive rules for model performance（关于模型表现的简单、可预测规则）：先用小规模模型把该调的东西调好，再靠一条简单规则把小尺度的行为外推（extrapolate）到大尺度。幻灯片把两条路线并列得很直白：Old and unpleasant: tune hyperparameters on big models（旧且难受：在大模型上调超参）对上 New (over?) optimism: tune on small models, extrapolate to large ones（新的、也许过于乐观的做法：在小模型上调参，再外推到大模型）。括号里那个「(over?)」不是闲笔，它预告了全讲反复出现的警惕：这种外推是一种信念，讲者说在大实验室里做 scaling 的人几乎把它当成 a way of life（一种生活方式、近乎一种信仰），而 scaling laws 有时是 quite tricky objects（相当棘手的对象）。换句话说，本讲教的是一套工程方法，不是一条已被证明的自然定律；只有小规模与大规模之间确实存在 simple、robust connections（简单而稳健的关联、某种 regularities 规律性），这种外推才值得信赖。

支撑这种乐观的，是讲者在同一页上并排给出的两张经验图。左图纵轴是 Validation Loss（验证损失，刻度约 2 到 6），横轴是 Compute (PetaFLOP/s-days)，一族曲线按参数量着色，右侧颜色条标作 Parameters，从 $10^5$ 一直排到 $10^{11}$；图上给出的拟合是 $L = 2.57\cdot C^{-0.048}$。右图纵轴 Test Loss（约 2 到 7），横轴 Parameters (non-embedding)（不含 embedding 的参数量，$10^3$ 到 $10^9$），图例依次是 1 Layer、2 Layers、3 Layers、6 Layers、> 6 Layers，几条曲线在双对数坐标上大致平行地向右下延伸。这两张图把「资源—损失」的正则性摆在了最前面：不同规模、不同层数的模型都落在同一族可预测的曲线上，于是「在小模型上调参、再外推」才有经验依据。需要注意的易错点是，右图的横轴特意写成 non-embedding（非嵌入参数），说明 x 轴怎么选、哪些参数计入，本身就会改变曲线形状；这一点会在后续关于参数计数的讨论里再次咬人，也是「scaling law 不是自动发生的魔法」的第一个伏笔。本段结尾讲者把今天的内容定性为 an engineering view of a scaling law（scaling law 的工程视角）：只要小规模与大规模之间存在稳健的、有规律的关联，我们就能对耗资数百万美元的大 run 先有一个定量预期，而不是把成败交给运气。

## 理论根源：样本复杂度、泛化界与 1993 年学习曲线 [03:59 - 06:56]

讲者特意先做 background（背景铺垫），理由是 scaling law 并非纯粹的新范式：它一方面是当下大模型设计与思考的方式，另一方面与很古典的想法贴得很近，做机器学习理论的人会有 feel at home（宾至如归）之感。幻灯片用一页提纲把 Part 1 的两条主线固定下来：Part 1. Scaling laws, history and background. 之下是 Data scaling as empirical sample complexities（把数据缩放看作经验样本复杂度）与 Initial forays into understanding neural scaling with data（用数据理解神经缩放的最初尝试）。这两条正是本章与下一章的骨架——先把「数据量—误差」还原成理论里熟悉的 empirical sample complexity（经验样本复杂度）与 generalization bound（泛化界），再回到历史文献，看同样的函数是怎样被一遍遍重新发现的。讲者还半开玩笑地说 I really really had to put some math into my slides（实在得往幻灯片里塞点数学），这也解释了为什么本段会出现成片的公式。

Sample complexity and rates（样本复杂度与收敛速率）那页的副标题是 Theorists have thought about 'scaling' for a long time..（理论学家研究「缩放」已经很久了）。它给出的第一个式子是有限假设集上的经典泛化界：

$$\epsilon(\hat h)\le\left(\min_{h\in H}\epsilon(h)\right)+2\sqrt{\frac{1}{m}\log\frac{2k}{\delta}}$$

旁边标注 (learning in a finite set of k hypotheses)（在含 k 个假设的有限集合里学习）。这里 $\epsilon(\hat h)$ 是学得假设的误差，$\min_{h\in H}\epsilon(h)$ 是假设类里的最优误差，$m$ 是样本数，$k$ 是假设个数，$\delta$ 是置信参数；多出来的那一项随 $m$ 以 $1/\sqrt m$ 量级衰减，正是「误差是样本量的函数」这一事实的理论写法。第二个式子来自 Theorem 1.5，针对 smooth densities（光滑密度）的 generative modeling（生成建模）：估计量 $\hat p_n(x_0)$ 的收敛速率写成 $\psi_n=n^{-\frac{\beta}{2\beta+1}}$，并在有限常数 $C$、所有 $n\ge1$ 下满足

$$\sup_{p\in\mathcal P(\beta,L)}\mathbb E_p\left[(\hat p_n(x_0)-p(x_0))^2\right]\le C\psi_n^2.$$

页脚引用是 CMU 10/701 的 lecture16-VC.pdf 与 Hall, 1989。讲者对这两页理论的态度很克制，幻灯片最后一行写着 But these are upper bounds, not actual, realized loss values.（但这些是上界，不是实际观测到的损失值）。这就是本章最该记住的易错点：理论给的是 loss 随样本量增长的 theoretical upper bound（理论上界），它解释了「误差—数据量」为什么天然是一个值得研究的函数，却不能直接当作用来外推大 run 的数值曲线；scaling law 关心的是 realized loss（实际损失）的经验正则性，二者相关但不等同。把上界当成实测、把定理当成拟合式，是读这段内容时最容易犯的错。

![样本复杂度与速率页：有限假设集泛化界与光滑密度生成建模速率公式，并强调这些只是上界](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0005_00-05-28.jpg)

历史的落点非常早。讲者说若追溯「第一篇 scaling law 论文」，可以回到 Bell Labs（贝尔实验室）与 Corinna Cortes、Vladimir Vapnik 等人 1993 年的工作 Learning Curves: Asymptotic Values and Rate of Convergence（学习曲线：渐近值与收敛速率），署名还有 L. D. Jackel、Sara A. Solla、John S. Denker，单位是 AT&T Bell Laboratories, Holmdel, NJ 07733。动机几乎就是数据 scaling law 的原始版本：Training classifiers on large datasets is computationally demanding（在大数据集上训练分类器计算代价很高），于是先在小样本上训练，把 error rate（错误率）随样本量衰减的曲线拟合出来，再外推完整数据集上的表现，从而避免在整个训练集上反复跑昂贵的训练。论文正文里的模型用两个幂律写：

$$\epsilon_{\text{test}}=a+\frac{b}{l^{\alpha}},\qquad \epsilon_{\text{train}}=a-\frac{c}{l^{\beta}},$$

其中 $l$ 是训练集大小；旁边的说明是 The test error is always larger than the training error, but asymptotically they reach a common value a（测试误差始终大于训练误差，但渐近地趋于同一个值 $a$），也就是把大训练集的误差写成 power-law decays（幂律衰减）到共同渐近值。这里 $a$ 就是后面会反复出现的 irreducible error（不可约误差）的雏形，$\alpha$、$\beta$ 是分别控制测试与训练曲线衰减快慢的指数，$b$、$c$ 是幅度常数。讲者评价它 is almost literally a data scaling law back in 1993（在 1993 年几乎字面上就是一个数据缩放律）。

![1993 年学习曲线论文页：标题、作者与 AT&T Bell Laboratories 署名、幂律公式以及 error 对 training set size 的曲线](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0006_00-06-41.jpg)

同一页右侧的图把这套拟合画得很直观：纵轴 error，刻度 0.05、0.1、0.15、0.2、0.25，横轴 training set size, $l$，标了 2560、7680、15360 三个位置；实心符号是 points used for prediction（用于预测的点），虚线是 predicted learning curves（预测出的学习曲线），上方一条标 test error、下方一条标 training error，两者从两侧向中间的渐近值 $a$ 靠拢。把这一页和前面那页理论并置，就能看清本讲一条贯穿的线索：无论是 $2\sqrt{\frac{1}{m}\log\frac{2k}{\delta}}$ 这样的理论上界、$n^{-\frac{\beta}{2\beta+1}}$ 这样的非参速率，还是 $a+b/l^\alpha$ 这样的经验曲线拟合，研究对象都是「误差如何随数据量下降」这同一个函数，只是有的来自定理、有的来自曲线拟合、有的来自对衰减趋势的直接建模。讲者由此收束本章：不要以为 scaling law 是 a very new purely neural thing（纯神经网络时代才有的全新事物），它在 empirical machine learning（经验机器学习）里有根，在 theoretical machine learning（理论机器学习）里也有对应物。下一章就顺着这条历史线，看 2010 年代 NLP 与 2017 年跨领域工作怎样把幂律形式一直带到今天。

## 历史脉络：早期 NLP 函数形式与 Hestness 2017 的前瞻结论 [06:56 - 10:17]

沿着「数据量—性能」这条线往中近期走，讲者先提到一类很朴素的主张：与其把钱花在 algorithm development（算法研发）上，不如直接去 collect more data（收集更多数据），因为把系统表现画成数据量函数时，曲线往往以可预测的方式持续改善；口述里把 Branco and Bril 举为 NLP 里关于「数据量与 scaling 是提升系统性能的正当途径」的典型参考。紧接着是 2012 年前后关于机器翻译的工作，讲者口述作 Collobert et al 2012，说他们增加训练数据量、用 BLEU score（BLEU 分数）衡量，专门问「模型表现随数据量增长该用哪种 functional form（函数形式）」，结果得到与今天一样的 power laws（幂律），并点名 power 3 和 power 4 这两类多项式——讲者随后预告，整堂课都会反复见到 essentially these polynomial functions power three and power four（本质上就是三次、四次幂这类多项式函数）。他还补了一句：we kind of knew about a lot of this back in the 2010s（2010 年代我们其实已经知道不少这类函数形式对模型行为很有预测力）。

幻灯片把这次「函数形式的早期检验」呈现得更具体。左图标题 Curve Fit using Recursive least squares（用递推最小二乘做曲线拟合），纵轴 BLEU score 从 0.14 到 0.22，横轴 Training sample size 从 0 延伸到 1,200,000；图例里 × 是 Observed values（观测值），实线是 Observed curve，其余虚线分别对应 Exp3、Exp4、ExpP3、Pow3、Pow4、ILog2 六种候选。右侧 Table 1: Curve families. 把六种形式逐行写全：

$$y=c-e^{-ax+b}\ (\text{Exp}_3),\quad y=c-e^{-ax^{\alpha}+b}\ (\text{Exp}_4),\quad y=c-e^{(x-b)^{\alpha}}\ (\text{ExpP}_3),$$

$$y=c-ax^{-\alpha}\ (\text{Pow}_3),\quad y=c-(-ax+b)^{-\alpha}\ (\text{Pow}_4),\quad y=c-(a/\log x)\ (\text{ILog}_2).$$

图下注明 Early tests of functional forms（函数形式的早期检验），并把出处标作 Kolachina et al 2012 – power law relation between data and downstream performance（数据与下游表现之间的幂律关系）。这里有个值得记录的对齐问题：讲者口述把这篇归为 Collobert et al，幻灯片文字写的是 Kolachina et al 2012；本笔记按画面保留幻灯片标注，同时不抹掉口述差异，不替两者判定谁对。无论姓名如何，结论是清楚的——在 Exp（指数族）、Pow（幂律族）、ILog（对数族）之间比拼后，幂律族对 BLEU 曲线的拟合最有预测力，且 $\text{Pow}_3$、$\text{Pow}_4$ 与今天通用的多项式形式几乎同一族。配图里那一族曲线在约 0.20–0.21 处趋于平台，也直观说明这类拟合必须和渐近值 $c$ 一起考虑，否则容易把「还在幂律区」和「已到不可约误差区」混为一谈。

![早期数据缩放历史页：递推最小二乘拟合的 BLEU 曲线与 Table 1 六种候选函数形式](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0007_00-07-54.jpg)

讲者随后用一整页表达对 2017 年一项工作的偏爱，理由是它该被引用得更多。口述里把源头归到 Hassabis 2017，称其为 a very forward-looking work（相当有前瞻性的工作），而幻灯片标题与后面几页的图注都写作 Hestness II / Hestness+ 2017；本笔记沿用画面上的 Hestness 标注。这项工作把 scaling law 建成 dataset size（数据集大小）的函数，跨 speech recognition（语音识别）、machine translation（机器翻译）、language models（语言模型）等多种系统，展示它们 follow these nice polynomial trends across a really large number of domains（在极多领域里都服从漂亮的多项式趋势）。副标题 Very ahead of its time..（远远领先于它的时代）下并排三段引文，对应三个今天仍在谈的概念。

第一段标 "Emergence"（涌现）：文字指出，小数据集测试虽可能，却很难保证训练数据已经大到进入 power-law learning curve region（幂律学习曲线区）；优化器参数化或模型先验／初始化不佳的模型会显示 accuracy cliffs（准确率断崖），此时 accuracy is only as good as best guessing（准确率只和瞎猜一样好），但模型仍按幂律区的速度在训练，因此 Researchers must take great care when defining a 'large enough' training set（研究者必须非常小心地定义什么叫「足够大」的训练集）。这正好解释了讲者口述里的那句判断：accuracy is a much more discontinuous measure than losses（准确率是比损失更不连续的度量），所以能力才会显得「突然出现」。第二段 Scaling by compute（按算力缩放）以 Computational Limits 起头：即便找到了值得放大的模型，下一个限制是 the speed of computation（计算速度），在超大数据集上训练大模型可能要 months or years of critical path compute time（数月到数年的关键路径算力），在现有系统上不现实；但 predictable learning and model size curves（可预测的学习曲线与模型规模曲线）能反过来投影出达到某一 accuracy level 所需的 compute，从而指导怎样扩展算力去解锁这些 compute-limited applications（算力受限的应用）。第三段 Speed = accuracy（速度即精度）以 The Performance-Accuracy Trade-off 起头：许多深度学习软硬件技术在模型精度与计算速度之间取舍，低精度计算／quantization（量化）与 sparse methods（稀疏方法）会牺牲 some model accuracy (e.g., up to 20%)（最多约 20% 的精度）来换 compute throughput（计算吞吐）；若吞吐提升让开发者能在更大数据集上训练更大模型，这些 accuracy losses（精度损失）might be easily recoverable（可能很容易补回来）。

![Hestness II 页：Emergence、Scaling by compute、Speed = accuracy 三段原文与准确率断崖、计算受限、性能-精度权衡的表述](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0008_00-09-08.jpg)

把这三段连起来读，逻辑链是完整的：可预测的 scaling 意味着 compute 会成为瓶颈（第二段），于是系统与速度优化本身就能兑换成精度（第三段），而 accuracy 的不连续性又会让能力在数据跨越某个阈值时显得突兀（第一段）。讲者因此感慨，2017 年远早于 2020 年代 OpenAI 那批 scaling laws 工作，如果当时认真读过这些论文，本可以 have known the regime that we're in even before seeing these large language models（在见到大语言模型之前就知道我们会进入怎样的区间）。这一页的局限也该点明：它给的是跨领域的经验观察与定性推断，不是可代入数值的公式；而且「准确率断崖」恰恰提醒我们，用不连续的 accuracy 去拟合平滑的幂律，比用 loss 危险得多。本章收在讲者那句 Any questions about this sort of like semi-historical part?（关于这段半历史内容有问题吗？）上，下一章正是学生由此引出的追问。

## 课堂问答：幂律函数形式究竟从何而来 [10:17 - 11:22]

历史部分刚收尾，学生立刻问到要害：这些 scaling law 究竟 just found empirically（只是经验发现的），还是有 some kind of way we can justify all of this（某种可以把它论证出来的办法）？讲者的回答先给了一个相当干脆的定性：these are sort of pure curve fitting exercises（这些大体上是纯粹的曲线拟合练习）。这句话必须与前两章并读才不误解——前面看到的 $a+b/l^\alpha$、$y=c-ax^{-\alpha}$、$L=2.57\,C^{-0.048}$ 都不是从第一性原理推出来的定理，而是在若干数据点上选一族函数、把参数 $\alpha$、$\beta$、$a$、$b$、$c$ 拟合出来，再检验它能否外推。也正因为如此，讲者前面才会说理论泛化界是 upper bounds、不是 realized loss values：定理能告诉你「误差该按什么量级衰减」，却不能替你决定实际曲线上那个指数是 $-0.048$ 还是 $-0.095$。把 curve fitting（曲线拟合）与 theoretical justification（理论论证）混为一谈，是读 scaling law 文献时最常见的错位，也是本章要澄清的第一件事。

不过讲者并没有把话说死，他随后补了两层来源。第一层是 theory（理论）：theory provides a very rich place where these scaling law functional forms can come from（理论是这些函数形式的一个非常丰富的来源），因为理论本来就反复研究 error rates（错误率）如何随各种对象衰减，会给出 candidate functional forms of various kinds（各种候选函数形式）。这正好把第 2 章那两条式子接了回来：有限假设集上界里的 $\frac{1}{m}\log\frac{2k}{\delta}$ 开根号后给出 $m^{-1/2}$ 型的衰减，光滑密度生成建模给出的 $\psi_n=n^{-\frac{\beta}{2\beta+1}}$ 则是一个由光滑度 $\beta$ 决定的多项式指数——它们都是「$\text{error}\sim n^{-\alpha}+c$」这一族的候选，于是

$$L(n)=c+\frac{A}{n^{\alpha}}\quad\Longleftrightarrow\quad \log (L(n)-c)=\log A-\alpha\log n$$

在扣掉常数 $c$ 之后就成了双对数图上的直线，斜率是 $-\alpha$。换句话说，理论提供的不是那条具体曲线，而是「该往哪一族函数里找」的清单。第二层来源讲者归给 physicists（物理学家）：they like thinking about limits like how do the limits behave（他们喜欢思考极限、思考极限行为如何），这也会给出不一样的 scaling law functional forms。这与本章开头讲的 power three and power four（三次、四次幂）呼应：多项式之所以反复出现，一部分原因是它在做大 $n$ 极限或做局部展开时是最自然、最稳的一族。

紧接着是整段问答里最重要的一句限定：there's no golden rule that these are the only ones（并没有什么金科玉律说只能是这几种）。讲者明确说，他后面会谈一点 scaling law 可能的来源、以及为什么多项式 arguably a very natural class of scaling laws（可以说是很自然的一类），但绝不宣称幂律是唯一解。这一点直接约束了后面所有章节：第 3 章那张 Table 1 里同时列着 $\text{Exp}_3$、$\text{Exp}_4$、$\text{ExpP}_3$、$\text{Pow}_3$、$\text{Pow}_4$、$\text{ILog}_2$，本身就说明在有限的观测区间内，指数族、幂律族、对数族往往都能把同一段 BLEU 曲线拟合得不错；若实验只覆盖很窄的一段 compute 或数据范围，就更难从数据本身区分「真幂律」与「局部看起来像幂律」。因此拟合时对函数形式和区间都要保持怀疑，这也埋下了本讲后面反复强调的易错点：scaling law 的预测力依赖于 x 轴选得对、超参设得对、且外推区间不要跨出曲线仍成立的幂律区。

问答在 [11:18] 的 Any other questions? 与 [11:21] 的 Good. 之后结束，讲者随即用 So now I'm going to start talking about more modern neural language model scaling 切回主线。这个转折在内容上很关键：前面四章处理的是「历史与理论根源」，从这里开始进入 Part 2（幻灯片上写作 Part 2. Neural (LLM) scaling behaviors），三个待答问题依次是 1. Data vs performance —「Are there simple rules that determine how data affects performance?」；2. Data vs model size —「Do we train on more data or bigger models?」；3. Hyper-parameters vs performance —「How should we set hyperparameters on the big model!?」。讲者同时说明会先讲 data scaling laws，因为它 to me they're the most simple and natural objects（在他看来最简单、最自然），可以借它一步一步演示「怎样猜出一个 scaling law 的函数形式、为什么多项式形式是自然的」，然后再转向 hyper parameters（超参数）与 architectures（架构）这些 harder to justify from intuition（更难从直觉论证）、却同样呈现漂亮 scaling 行为的对象。本章因此既是历史部分的收束，也是下一章「多因素幂律」的引子。

## 现代神经缩放：多种资源都能把测试损失线性化 [11:22 - 13:32]

进入 Part 2 之后，讲者先给一个总纲性的观察：power law relationships（幂律关系）remarkably hold for a huge variety of different factors（惊人地适用于极多种不同的因素）。这句话有两层意思：其一，预训练里最常见的三种 x 轴——log compute（对数算力）、log of the data set（对数数据集大小）、log of parameters（对数参数量）——分别配 log of test loss（对数测试损失）作 y 轴时，都能把曲线 linearize（线性化）；其二，这种正则性还溢出到下游 benchmark、甚至以日期为横轴的 forecasting（预测）图上。幻灯片标题 Scaling laws – power law relationships for many factors，副标题 These scaling laws hold on *many* different kind of phenomena!（这些缩放律在*许多*不同类现象上都成立），把这个「普适性」摆成了本段的中心论点。讲者口述时还纠正了一次口径：他先说 compute on the x-axis，随即改口 this is log compute, sorry, on the x-axis（是对数算力），提醒听众所有这些 x 轴都是 log 轴，不是线性轴。

上半页三张并排的图给出三个具体拟合，全部来自 Kaplan+ 2020，且都注明 x 轴口径。左图横轴 Compute，单位 PF-days, non-embedding（PF-天，非嵌入参数口径），纵轴 Test Loss（约 2 到 7），拟合写成

$$L=\left(C_{\min}/2.3\cdot10^{8}\right)^{-0.050};$$

中图横轴 Dataset Size，单位 tokens（$10^8$ 到 $10^9$），纵轴约 2.7 到 4.2，拟合为

$$L=\left(D/5.4\cdot10^{13}\right)^{-0.095};$$

右图横轴 Parameters，non-embedding（$10^5$ 到 $10^9$），拟合为

$$L=\left(N/8.8\cdot10^{13}\right)^{-0.076}.$$

三个指数 $-0.050$、$-0.095$、$-0.076$ 都是绝对值很小的负数，含义是损失随资源幂律下降但下降得很慢；而 $2.3\cdot10^8$、$5.4\cdot10^{13}$、$8.8\cdot10^{13}$ 这些归一化常数则提示，拟合式里的尺度并非任意，换一种 x 轴口径（比如把 embedding 参数算进 $N$）就会改写整条曲线。讲者口述时把这三个 x 轴统称为 resources（资源）：you can get a variety of different x-axes that you might think of as resources in the log axis, all of them sort of linearize log test loss here（你可以把各种 x 轴都看成对数轴上的资源，它们都能把对数测试损失线性化）。这也回应了第 1 章右图横轴特意写 non-embedding 的伏笔：同一个「参数量」，计不计 embedding，结论里的指数和常数都会变。

下半页标着 They even hold in non standard settings（它们甚至在非标准设定下也成立），给出两类更「奇怪」的例子。第一类是下游任务上的 sigmoid 形曲线：讲者口述说在 downstream benchmarks（下游基准）里，作为 compute 或 compute-like quantities（类算力变量）的函数，预期看到的是 sigmoid functional form（S 形／逻辑斯谛函数形式），而不是幂律直线。画面上 Word Unscramble 小图纵轴 Normalized Exact Match，横轴 $\log_{10}(\text{Llama-2 Equiv. FLOPs (1E21)})$，拟合 $y=\text{sigmoid}(2.0(-6.11))$，并列出 $\text{MSE}_{\text{train}}=1.3\text{e-}04$、$\text{MSE}_{\text{eval}}=4.0\text{e-}03$；旁边 Persian QA 小图纵轴 Normalized Accuracy，$y=\text{sigmoid}(2.2(-8.43))$，$\text{MSE}_{\text{train}}=1.8\text{e-}04$、$\text{MSE}_{\text{eval}}=2.2\text{e-}03$。这里 $\text{MSE}_{\text{train}}$ 与 $\text{MSE}_{\text{eval}}$ 相差一个数量级以上，是一个具体且容易被忽略的数字：它说明这类下游曲线的拟合误差并不像预训练损失那样干净，外推时要格外小心。第二类是更偏 forecasting 的图：讲者说有人干脆 putting dates as the x-axis（把日期放在横轴），而 strangely enough（奇怪的是）这些曲线的 upper envelope（上包络）在多种 capability（能力）度量上 often linear（常常是线性的）；画面上对应 Epoch Capabilities Index (ECI) 那张散点图，图例里能看到 BPB-CHAi、QRI Results、Olympiad、image、Go、SimpleQA、MMLU、GPQA、HLE、IOL 等条目。

![多因素幂律页：上排算力、数据量、参数量对测试损失的三条直线拟合，下排下游任务的 sigmoid 曲线与能力指数散点](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0011_00-12-47.jpg)

讲者对这两类「异型」scaling law 的态度是诚实的：there's no reason that on this bottom side the more exotic scaling laws need to be true（没有理由认为下面这些更奇特的缩放律必须成立），但 it turns out to be the case that language model performance is often much more regular as a function of scale and resources than it initially might appear（事实却是，语言模型表现作为规模与资源的函数，往往比初看之下规律得多）。这正是本章的易错点所在：把「上排三条干净直线」的置信度直接套到「下排 sigmoid 与日期包络」上是不成立的——前者的 y 轴是平滑的 test loss，后者的 y 轴是 accuracy / exact match 这类离散、带阈值、方差更大的量，第 3 章 Emergence 那段的 accuracy cliffs 已经说明过它的风险。与前后内容的关系也清楚：本章用三个具体指数把第 1 章「资源—损失」的定性印象定量化，同时用 sigmoid 与不可约误差的伏笔，把话题交回下一章——在那里讲者会收窄到最干净的一元关系 data scaling law，从「我们期望看到什么曲线」一步步推到「为什么 log-log 直线就是幂律」。

## 数据缩放律：从期望曲线到 log-log 直线与均值估计 [13:32 - 17:34]

讲者把 data scaling laws 单拎出来，理由是它 the most simple and natural（最简单自然），适合拿来 step-by-step（一步一步）演示怎样猜函数形式。幻灯片 Data vs performance 先给定义：What's a data scaling law? 答 Data scaling laws : simple formula that maps dataset size (n) to error（把数据集大小 $n$ 映射到误差的简单公式）。接着问 What do we expect out of scaling laws? 画面左侧答 Monotonic, logistic-like curves（单调的、类逻辑斯谛曲线）。右侧那张示意图把预期画全了：纵轴 Generalization Error (Log-scale)（泛化误差，对数刻度），横轴 Training Data Set Size (Log-scale)（训练数据量，对数刻度），曲线从左上 Best Guess Error（最佳猜测误差，绿色虚线）出发，经过三个竖线划分的区域 Small Data Region（小数据区）、Power-law Region（幂律区）、Irreducible Error Region（不可约误差区），最后贴近下方红色虚线 Irreducible Error，图注 [Hestness+ 2017]。设定上，讲者说要 fix our model training procedure（固定训练过程），且 unless I say otherwise the models are going to generally be bigger than the data set size（除非另有说明，模型一般比数据集更大），再有规律地增加数据量看误差怎么降；只要超参调得好，关系就该是 monotone-ish（大致单调），因为 more data should help us do better（更多数据应当让我们做得更好）。对 classification（分类）或 next token prediction（下一 token 预测）而言，误差应从 random guessing（随机猜）降到某种 entropy-like object（类熵的量），再往下就不能优于任务的 irreducible error（不可约误差），也就是 noise floor（噪声下限）。讲者还预告，students will make something that looks roughly like this in parts of assignment three（在作业三的一部分里，你们自己会做出大致像这样的图），把经验观察落回课程实践。

紧接的一页从「期望」跳到「实测」。标题 Data scaling laws for language models，副标题 First, an empirical observation（先来一个经验观察），加粗一句 Loss and dataset size is linear on a log-log plot（损失与数据量在双对数图上呈线性）。图里纵轴 Test Loss 从 2.7 到 4.2，横轴 Dataset Size（tokens）从 $10^8$ 到 $10^9$，若干个点几乎落在一条直线上，拟合标为

$$L=\left(D/5.4\cdot10^{13}\right)^{-0.095},$$

右侧写着 "Scale-free" or "Power law"（无标度，或幂律），底部注明 (For language modeling, from Kaplan+ 2020)。讲者说这张取自 Kaplan 2020、也就是 OpenAI 的 neural scaling laws 论文，后面会反复引用，因为它 has a lot of lovely ablations and a lot of lovely data（有大量漂亮的消融与数据）。这里要留意与上一章的衔接：同一个 $L=(D/5.4\cdot10^{13})^{-0.095}$ 既出现在「多因素」页的中图，又出现在本页，说明本讲是在把一张总览图拆开、逐条讲透。

![Kaplan 2020 数据缩放页：测试损失对数据量在双对数图上呈直线，拟合为 L=(D/5.4·10^13)^-0.095](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0013_00-15-13.jpg)

为什么这条直线意味着幂律，讲者特意为不熟的听众推了一遍。若 log-log 图上是直线，等价于误差在按多项式衰减：

$$L(D)\propto D^{-\alpha}\quad\Longleftrightarrow\quad \log L=-\alpha\log D+c,$$

也就是 $y=-\alpha x+c$。他还给出一个重要判据：出现直线 usually means that I'm very far away from my asymptote（通常意味着我们离渐近线还很远），因为一旦逼近不可约误差，曲线会 taper off（弯折趋平）而不是继续直线下降。这正好把上一页的三个区域串起来——只有中间的 Power-law Region 才是这条直线成立的区间，把它无限外推到 Irreducible Error Region 就会系统性高估收益。这是本章第一个易错点，也是第 4 章「没有 golden rule」的一个具体后果：直线是局部性质，不是全局定律。

为了让「多项式为什么自然」落地，讲者暂时把语言建模课改成统计学课（他自嘲 for the next one slide or actually two slides... maybe three slides），给出 Toy example: mean estimation（玩具例子：均值估计）。幻灯片写 Input: $x_1\dots x_n\sim N(\mu,\sigma^2)$，Task: estimate the average as $\hat\mu=\frac{\sum_i x_i}{n}$；由标准论证得

$$\mathbb E[(\hat\mu-\mu)^2]=\frac{\sigma^2}{n}.$$

因为每个样本是高斯、$\hat\mu$ 也是高斯，误差就是 $\sigma^2/n$。取对数后幻灯片写成 This is a 'scaling law'，并给出

$$\log(\mathit{Error})=-\log n+2\log\sigma,$$

即 $\log$ 误差是 $\log n$ 的线性函数，斜率 $-1$，截距由 $2\log\sigma$ 决定；最后一句 More generally, any polynomial rate $1/n^{\alpha}$ is a scaling law（更一般地，任何 $1/n^\alpha$ 型的多项式速率都是 scaling law）。讲者口述把它推广成 anything that is essentially 1 over n to the alpha plus some constant（本质上就是 $1/n^\alpha$ 加个常数），并强调在 parametric estimation（参数估计，如均值估计、回归）里 you of course expect a polynomial rate, you don't really expect anything else（本来就预期多项式速率，不会预期别的）；若拟合回归可能得到 $d/n$ 一类，但 essentially all the classical models you are expected to have a 1 over n scaling（几乎所有经典模型都预期有 $1/n$ 缩放）。于是按这个理论预期，双对数直线应当是

$$y=-x+C,$$

斜率为 $-1$。这里 $\hat\mu=\frac{\sum_i x_i}{n}$、$\sigma^2/n$、$2\log\sigma$、$1/n^\alpha$、$d/n$ 都是画面与口述共同给出的具体符号，不能与前面 Kaplan 拟合里的 $-0.095$ 混为一谈：前者是参数估计的理论速率，后者是神经语言模型的经验速率。本章收在讲者那句转折上——if we fit our neural scaling laws we will find that our exponents are quite different（若去拟合神经 scaling law，会发现指数很不一样），下一章就来处理这个 interesting mystery（有趣的谜）。

## 指数之谜：神经缩放为何像高维非参数学习 [17:34 - 19:29]

把上一章的理论预期 $y=-x+C$（斜率 $-1$）与神经模型的实测放在一起，落差就是本段最后一页要解的谜。讲者说这些指数 taken these two on the left from Hestness、the one on the right from Kaplan，口头归纳为 about negative point one, negative point three, negative point one, roughly speaking（大致约 $-0.1$、$-0.3$、$-0.1$）。幻灯片 Scaling law exponents: an intriguing mystery 把对照摆得很清楚：Fact: Similar arguments show most 'classical' models (regression, etc) have $\frac{1}{n}$ scaling（事实：类似的论证表明大多数「经典」模型，如回归，具有 $1/n$ 缩放），接着 This means we should see $y=-x+C$（这意味着我们应当看到 $y=-x+C$），然后反问 What do we find in neural scaling laws?（我们在神经缩放律里实际看到什么？）。下方三张小图分别是 Machine translation、Speech、Language modeling：机器翻译图的拟合标着 $\ell(m)=2.87\,m^{-0.13}$，横轴 Training Data Set Size, Number of Tokens (Log-scale)；语音图横轴 Training Data Set Size, Hours of Audio (log-scale)，图例含 DS2、Attention、DS2 Trend、Attention Trend，两条趋势线以 $e(m)=0.95\,m^{-\alpha}$、$e(m)=1.36\,m^{-\alpha}$ 一类的形式标注；语言建模图仍是 $L=(D/5.4\cdot10^{13})^{-0.095}$，横轴 Dataset Size (tokens)。页底一句 Very different from predictions.. Why might this be?（与预测很不一样……为什么会这样？）把谜题正式抛出。易错点在于：这些指数虽都是多项式，量级却比 $-1$ 小一个数量级，讲者明确说 this is much slower than estimating a mean or estimating like a linear regression（这比估计均值或线性回归慢得多）、a much much slower convergence rate（慢得多的收敛速率）——「仍有多项式速率」和「速率够快」是两件不同的事，不能因为曲线在 log-log 上是漂亮直线就以为数据效率很高。

讲者给出的解释方向是 non-parametric（非参数）：Where might you get an estimation rate like this? Well, you would get this if your model was kind of more non-parametric（要得到这样的估计速率，模型得是更偏非参数的）。他先道歉 this will be the last statistics slide of this lecture（这会是本讲最后一张统计学幻灯片），然后把设定换成估计一个 arbitrary smooth function（任意光滑函数）。幻灯片 Detour: scaling laws for (nonparametric) learning 写道：Neural nets can approximate arbitrary functions. Lets turn that into an example.（神经网络能逼近任意函数，我们把它变成一个例子。）框内是 Input: $x_1\dots x_n$ uniform in 2D unit box（在二维单位方里均匀分布），$y_i=f(x_i)+N(0,1)$（带标准高斯噪声），Task: estimate f(x)，Approach: cut up the 2D space into boxes with length $n^{-\frac14}$（把二维空间切成边长为 $n^{-1/4}$ 的小盒子）。这一步与第 2 章那条 $\psi_n=n^{-\frac{\beta}{2\beta+1}}$ 是同一个思想的不同写法：函数类比线性类灵活得多，收敛速率就必须为这份灵活性付出代价。

![非参数学习缩放页：二维单位方上的函数估计、边长 n^-1/4 的切分、sqrt(n) 个盒子与 d 维的 n^-1/d 速率](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0016_00-18-52.jpg)

误差的推导在画面上写得紧凑而完整。因为二维空间被切成边长 $n^{-1/4}$ 的格子，格子的总数是 $\left(n^{1/4}\right)^2=\sqrt n$，每个格子平均分到 $n/\sqrt n=\sqrt n$ 个样本，于是幻灯片写 Informally, we have $\sqrt n$ boxes, each box gets $\sqrt n$ samples（不严格地说，我们有 $\sqrt n$ 个盒子，每个盒子拿到 $\sqrt n$ 个样本），误差约为

$$\mathit{Error}\approx\frac{1}{\sqrt n}+(\text{other smoothness terms}).$$

讲者口述时也把它念成 the error is going to be roughly one over square root of n or something，并特意用 or something 保留了不严格的语气；括号里的 other smoothness terms（其它光滑性项）也提醒我们这不是一个精确等式。把二维推广到 $d$ 维，幻灯片给出

$$\text{In } d\text{-dimensions, this becomes } \mathit{Error}=n^{-1/d}\ \text{- This means scaling is } y=-\frac{1}{d}x+C,$$

即指数从 $-1$ 变成 $-\frac1d$，并由 Takeaway 收束：flexible 'nonparametric' learning has dimension dependent scaling laws（灵活的「非参数」学习具有依赖维度的缩放律）。讲者口述同样把它写成 my scaling is going to be $y=-1/D$ times $x$，并指出 these kinds of more flexible functions, functions that are more flexible than the linear class, now have rates in their scaling laws that are not just one, they're one over D（比线性类更灵活的函数，其缩放律速率不再是 1，而是 $1/D$）。把 $\alpha\approx0.1$ 与 $-1/d$ 对照，直觉上就相当于一个有效维度相当高的非参数问题——但要注意这只是 mental model（心智模型）：讲者在段末说 one mental sort of model that you could have is that the neural networks I was showing you in these slides are behaving kind of like non-parametric（可以有这样一个心智模型：前面这些幻灯片里的神经网络表现得有点像非参数模型），措辞是 could have、kind of like，并非断言内在维度等于某个确定数值。本段到此在 [19:29] 截断，句子尚未说完，因此不宜替讲者补出结论；能确定的是，第 4 章「函数形式来自曲线拟合、理论只给候选」的限定，在这里得到了最好的例证：同一条 log-log 直线，参数估计理论给斜率 $-1$，非参数理论给 $-1/d$，而神经语言模型实测落在 $-0.1$ 附近，三者靠同一个「多项式误差 + 常数」的框架连在一起，却谁也不能被当成唯一正确答案。

## 数据缩放定律的指数与内在维度理论 [19:29 - 20:24]

本段开头承接上一段关于「为什么损失会按幂律下降」的讨论。讲者在 19:33 附近给出一个非常具体的类比：如果要在 10 维空间里做回归，用最近邻（nearest neighbor）一类非参估计器，那么它从数据里学习的速率大致就是 $n^{-1/d}$ 这个量级；而神经网络拟合出来的数据缩放指数，恰好落在类似「10 维回归器」的位置上。换句话说，只看 scaling law 的斜率，就能反过来猜出模型「表现得像是多少维的光滑器」，讲者认为这是一件「光看缩放定律就能知道」的相当酷的事情。屏幕上给出的幂律母形式是 $\frac{1}{n^{\alpha}}$，对应的 log-log 直线为 $\log L = -\alpha\log D + c$；若把它与非参估计联系起来，就是 $\text{Error}\sim n^{-1/d}$，斜率 $-\alpha \approx -1/d$，取 $d\approx 10$ 便得到 $\alpha\approx 0.1$，这正是经验拟合中常见的 $-0.1$ 到 $-0.3$ 那一档。

这一页幻灯片标题是「Intrinsic dimensionality theory of data scaling laws」，署名标注为「Some have made the following argument (Bahri 2021)」（转写里念作 Barri et al.），两条 bullet 必须原样记住：第一条是「Scaling laws arise due to polynomial rates of learning $\frac{1}{n^{\alpha}}$」，即 scaling law 源于多项式速率的学习；第二条是「Some argue the slope $\alpha$ is closely connected to the intrinsic dimensionality of the data」，即有人主张斜率 $\alpha$ 与数据的内在维度（intrinsic dimensionality）直接挂钩。更激进的解读来自一批 scaling law 理论工作者：他们认为这些指数「literally」在告诉我们，神经网络正像一个非参平滑器（non-parametric smoother）那样学习。

![内在维度理论：斜率 α 与数据内在维度相连，散点来自多组数据集（Bahri 2021）](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0017_00-20-05.jpg)

图中央那张散点图把这套论证画得很直白：横轴是 Dimension，刻度从 2 到 26；纵轴写作 $4/\alpha_0$，刻度 0 到 25；图中画了两条参考线，黑色虚线标为 $4/\alpha_0$、浅色点线标为 $2/\alpha_0$，散点大致沿 $4/\alpha_0$ 那条线爬升。图例列出六组数据来源：Teacher-Student、CIFAR-10、CIFAR-100、SVHN、FashionMNIST、MNIST，分别用品红、紫、蓝、青等颜色区分，最右上角的一个点落在 Dimension 约 25、$4/\alpha_0$ 约 25 处。也就是说，不同数据集上估出的内在维度与缩放指数之间确实呈现出近似线性的对应，这正是「斜率由维度决定」这一主张的经验依据。

但讲者随即给这套论证踩了刹车，这也是本章最需要记住的易错点。他明确说自己「不太确定到底有多相信」这个论证，理由有两条：其一，证据「might be a little sketchy」，比较薄弱；其二，整套推理依赖对内在维度（intrinsic dimension）的估计量，而幻灯片底部那行字写得更不客气——「But estimators of intrinsic dimension are sketchy, and this is not airtight..」，即内在维度的估计器本身就很粗糙，因此这个论证并不严密。所以正确的态度是：把「神经网络像高维非参平滑器」当作一个可以按需采信的直觉（「buy that intuition to whatever extent you'd like」），而不是当成一个已被证明的结论。这里还埋着一个方法论上的局限：$\alpha$ 是从有限区间的曲线拟合出来的，用它去除出 $d$、再去反推「模型类」的性质，属于二次推断，误差会被放大。

在 20:25，讲者宣布暂停，并说这基本就是「最基础的、单一分布采样并从中学习」的数据缩放定律的终点（「That's really the end of the really basic data scaling law」）。因此本章既是对上一段均值估计、非参回归类比的理论收束，也是全段从「一个分布、一个 $D$」转向「数据怎么配、怎么重复、怎么过滤」等工程问题的分水岭：接下来不再只问「学得多快」，而要问「该拿什么数据去学」。

## 「模型大于数据」的含义与幂律区间 [20:24 - 21:33]

暂停之后，一位学生立刻追问了一个上一章留下的措辞：「你说模型总是大于数据集大小，是指一一对应吗？是参数个数对 token 数吗？」讲者在 20:48 到 21:28 给出了本段最实用的一段澄清。他说，「模型大于数据」并不是指参数量与 token 数严格一一对应或某个固定比例；真正的含义是：当模型相对数据太小时——也就是「你手上的数据远多于模型规模」——你会进入不可约误差区（irreducible error regime）。此时你已经把数据拟合到该模型类（model class）能力所及的最好程度，之后无论再加多少数据都无法再改善，损失曲线在 log-log 图上会弯折、趋于水平，而不再是直线。

因此做 scaling law 时通常只想待在更简单的那一段：幂律区间（power law regime）。讲者说这里其实只有两条路可走，原话是「Those are kind of the two things」：要么让实验始终落在幂律区间里，要么显式地把渐近线（asymptote）拟合出来再对它做修正。后者可以写成把不可约误差当作待拟合参数的形式，例如 $L(D)\approx \varepsilon_\infty + A\,D^{-\alpha}$，其中 $\varepsilon_\infty$ 是不可约误差、$\alpha$ 是幂律指数；而在幂律区间内，$\varepsilon_\infty$ 相对可忽略，于是退化为上一段一直在用的 $\log L = -\alpha\log D + c$。这一步形式化的意义在于：它把「曲线什么时候不再是直线」变成了一个可操作的判据，而不是靠肉眼挑区间。

关于具体数值，讲者给的经验法则是「Usually like, you know, 10 times bigger than the data will have you」——模型大约比数据大 10 倍，通常就足以待在幂律区间里了。他同时强调「it can be basically any amount that it's larger than the data as long as it's in this power law regime」，即大多少并不重要，只要没有掉进不可约误差区即可。注意这里的「10 倍」是量级上的宽松说法，不是要精确调参；学生问题里担心的「参数数 vs. token 数一一对应」恰好是讲者要否定的那种误解。

![数据缩放定律的概念基础：小数据区、幂律区与不可约误差区三段示意](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0018_00-21-18.jpg)

屏幕上此时回到「Conceptual foundations of data scaling laws.」这一页，正好把上面的三种区间画在一起。页面上的主问题是「Q: Why do scaling laws show up?」，下面先写「We know error should be monotone」（误差本来就该单调下降），再追问「But why is it a power law / linear in log-log?」；给出的答案是「A (?): Estimation error naturally decays polynomially.」——注意那个问号，讲者自己也把这条解释当成待商榷的假设。页面还写着「But this answer may take a moment to understand... through an example.」以及「Example: If our task is to estimate the mean of a data[set]... what's the scaling law?」，回扣上一段的均值估计玩具例子。右侧的小示意图把损失曲线划成三段：小数据区（Small Data Regime）、幂律区（Power-law Regime）、不可约误差区（Irreducible Error Regime），并用一条斜线标出中间的直线段。这张图因此是本章「幂律区间」概念最直接的视觉锚点，也解释了为什么讲者坚持「模型要够大」：只有让横坐标落在中间那一格，$\log L = -\alpha\log D + c$ 才成立。

易错点有两层。第一层是学生所担心的定义混淆：把「模型大于数据」误读成参数量与 token 数的固定比例，会让「10 倍」这条经验法则被当成硬约束；实际上它只是用来规避不可约误差区的经验手段。第二层更隐蔽：如果实验里混进了已经贴近渐近线的点，却仍用纯幂律去拟合，得到的 $\alpha$ 会被系统性地压低，后续所有外推都不可靠；这正是讲者说「要么待在幂律区间、要么显式拟合渐近常数」的原因。与前后内容的关系也很清楚：这一问一答把上一段「模型显著大于数据集」的实验设定补成了可操作的工程判据，随后 21:33 讲者一句「Now we are back to other data scaling laws」，正式把话题从「加多少数据」推向「加什么数据」。

## 数据组成与混合律：截距改变、斜率不变 [21:33 - 25:21]

讲者在 21:38 开门见山地指出，单看数据缩放定律「没你想的那么有用」：它只回答「我的模型学得多快」，对预测有用，对工程却几乎不够。真正做工程时想问的是「什么才是最优数据混合（optimal data mixture）」「该不该重复数据、还是把算力省下来做别的」「是不是该重复高质量数据而不重复低质量数据」。他给出的关键理论线索来自泛化界（generalization bounds）：既然数据缩放定律本质上是泛化界的经验版本，那么对许多模型而言，数据组成（dataset composition）主要影响的是 scaling law 的截距/偏移（offset、intercept），而斜率（slope）更多由模型类（model class）决定，并不由数据分布本身决定。于是可以写成 $L(D;q)\approx C(q)\,D^{-\alpha}$：混合比例 $q$ 只进到前置系数 $C(q)$ 里，$\alpha$ 保持不变；取对数后就是 $\log L = -\alpha\log D + \log C(q)$。

![分布偏移下的数据缩放：不同混合比例 q 只改变误差截距，而最优截距呈 U 形](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0019_00-22-31.jpg)

「Other advanced data scaling law: distribution shift」这页把上述主张写成了两条对照式的问题与答案。页面上第一行是「Data scaling thus far: how does dataset size relate to performance?」，第二行是「Related question: how does dataset composition affect performance」，中间用粗体给出结论「A: Data composition affects the offset, not the slope.」，再补一句「These 'distribution shift' scaling laws can tell us about the importance of collecting diverse data!」，右下角标注 [Hashimoto 2021]。左图的横轴是 Training data size（$10^2$ 到 $10^3$，log 刻度），纵轴是 Excess error（$10^{-2}$ 到 $10^0$，log 刻度），三条近似平行下降的直线对应图例中的 $q=0.00$（蓝）、$q=0.22$（橙）、$q=0.56$（绿）——斜率几乎一致，只有上下位置不同，正是「变截距不变斜率」的图形化证据。右图标题是「Expected error intercept」，横轴 Data source proportion 从 0.0 到 1.0，纵轴是 $\log C(q)$（刻度 2 到 4），曲线是一个明显的 U 形，最低点落在 $q\approx 0.4\sim0.5$ 附近，两端（只用单一数据源）显著抬高。讲者用线性回归玩具模型解释这个 U 形：有两个可采样的数据源时，只取其中一种，误差怎么都高，只有把两者混合才压得下去，因此数据多样性（data diversity）本身就能降低截距。这个 U 形是「最优混合比例」存在的几何根源，也是后面所有外推方法的目标函数。

工程上怎么用它？讲者描述了「在小算力上拟合函数形式、找最小值、再把它放大」的朴素想法：假设只有 news 和 Wikipedia 两个数据源，你想知道各用多少，就先在小模型、少量数据上训练不同混合比例，拟合「混合比例—性能」的函数，再一边逐步放大模型一边观察趋势如何变化，最后像外推普通数据缩放定律那样外推到完整训练规模，问「如果继续放大到我的 full training law，哪一个比例会最优」。屏幕上左侧是复旦大学的「Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance」论文示意图，把这条路线拆成三步：Small Steps, Small Models, Seen Mixture →(1)→ Large Steps, Small Models, Seen Mixture →(2)→ Large Steps, Large Models, Seen Mixture →(3)→ Large Steps, Large Models, Unseen Mixture，方框里注明三步分别用 (1) Training Step Law、(2) Model Size Law、(3) Data Mixing Law；中间的三维示意图标出 Model Size Laws、Training Step Laws、Mixture Dimension (Domain 1,2) 与 Mixture Loss。Figure 1 的说明写着：这个 pipeline 用三步分别利用训练步数、模型规模与数据混合律的 scaling，去预测「large steps、large models、unseen mixtures」上的模型性能。

但幻灯片标题本身就是一盆冷水：「In practice: data mixture selection via scaling is hard」。右侧的 DataDecide 研究「How to Predict Best Pretraining Data with Small Experiments」给出了实证对照：横轴是 Proportion of Target Compute (RC)（$10^{-4}$ 到 $10^0$），纵轴是 Decision Accuracy（0.3 到 0.9），上方还标了 Compute (FLOPs) 从约 $7\times10^{+15}$ 到 $7\times10^{+20}$，右侧列出 4M、8M、16M、20M、60M、90M、150M、530M、1B 等一系列模型规模，并写着「Pretrain 25 datasets @ 150M to predict @ 25 datasets」「18 - 80% correct」以及 Multi-Scale Fit 等预测方法。页面底部两行结论被讲者在 24:24 之后逐字念出：左边是「Natural idea – build data scaling laws」，右边是「Empirical eval – just take the best small dataset」。也就是说，现实比理想世界嘈杂得多（「reality is a lot more noisy」），很多团队的实际做法是训练一堆小模型、在小模型上挑出最好的数据混合、然后直接把它放大，「No scaling law required」。DataDecide 发现，不拟合 scaling law、只在小规模挑最佳混合，效果就很好。

讲者立刻指出这个实证结果与前面的理论是自洽的：如果斜率真的不随混合比例变化，那么小规模上的最优混合也必然是大规模上的最优混合（「if the slopes don't change, the best mixture at small scale is also the best mixture at large scale」）。这一句把本章从「怎么外推」拉回到「为什么不必外推」，是全段最漂亮的一处前后呼应。易错点必须讲清：第一，「斜率不变」是经验观察而非定理，它依赖固定数据与固定模型类，讲者在后面 35:06 还会第三次为它感到惊讶；第二，DataDecide 的 18–80% correct 说明「小规模选最优直接放大」并非总是可靠，只是统计上够用；第三，三步外推会把每一步的拟合误差累积放大，「unseen mixture」这一步尤其危险；第四，U 形截距 $\log C(q)$ 是在特定两源设定下得到的，数据源一多，最小值的位置和形状都可能变。与前后内容的关系是：本章把「数据组成」正式纳入 scaling law 框架，而下一章会把同一个「截距变、斜率不变」的主题推广到数据重复、集成与过滤上。

## 数据重复、无界算力与过滤的自适应 [25:21 - 28:29]

讲者在 25:29 抛出一个越来越现实的背景：算力（compute）在涨，而我们拥有的数据量并没有同步增长，于是「把数据一遍遍重复会怎样」成了很多人在问的问题。他先引用几年前的一项研究「Scaling Data-Constrained Language Models」，结论相当具体：在标准训练配方（standard training recipes）下，大约 4 个 epoch 以内「you just don't get hurt at all」，几乎不受伤；一旦超过这个点，实际的 scaling 曲线（讲者称「this sort of dark curve」）就会明显差于「如果有新鲜数据」时的投影曲线（虚线，dash line）。和前面一样，方向是「写出一个修正后的函数形式」来描述重复数据下的 scaling 行为。

![数据重复下的有效数据量公式与算力回报、算力分配两张图](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0022_00-26-11.jpg)

「Scaling laws under data repetition」这页的副标题是「In practice, we have finite data – how does repeating examples affect scaling?」，底部给出的核心公式必须原样记下：

$$
D' = U_D + U_D R_D^*\left(1 - e^{-R_D/R_D^*}\right)
$$

右侧的图例逐项解释了符号：$D'$ = Effective data（有效数据量）、$U_D$ = Unique tokens（唯一 token 数）、$R_D^*$ = Constant（常数）、$R_D$ = Repetition（重复次数）。这个形式的直觉是：有效数据量在 $R_D$ 很小时近似线性增长，随着重复次数增大被指数饱和项压平，最终趋于 $U_D(1+R_D^*)$ 这一上限——重复不再带来等价的新鲜数据。页面上方并排两张小图：左图「Return on compute when repeating」的纵轴是 Final test loss（约 2.2 到 3.4），横轴是 Tokens (Epochs)，标了 1.0、4.0、16、64、256、1024 等 epoch，注释里写着「Models trained with repeated data is worth the same as new data」以及「Loss according to required repeated data to match the same as new data」；右图「Allocating compute when repeating」纵轴是 Parameters、横轴是 Tokens (Non-rep)，上面标了「Loss: 2.316」「Loss: 2.359」「10.7 (10%)」等数值，图例区分了「Regime of same-compute」与「Efficient frontier」，讨论的是把算力分配给重复数据还是新数据。左下角还能看到论文名「Scaling Data-Constrained Language Models」。

接着讲者把重复推到极端：这是与共同指导的学生 Perseus 的近期工作，设定是「无限算力」——允许你想 epoch 多少次就 epoch 多少次（「infinite amounts of compute」「as many times as you want, like an infinite amount of times」）。结论是不能只靠不断重复遍历数据、也不能不停把模型做大，这两者都有收益递减（diminishing returns），于是不得不伸手去用别的手段，比如把模型集成（ensembling）起来，尽量把数据里的价值榨出来。屏幕上「Scaling laws in compute unbounded settings / Pre-training under infinite compute」这一页写着两条「Important notes」：第一，「Scaling laws can 'break' if you blindly apply it」；第二，「Scaling laws are lower bounds so you can always potentially do better」。三张小图分别叫「Increasing epoch count」（横轴标 1、4、16、32、64、128 等 epoch，下方表格配了 1e-3、3e-4 等 learning rate）、「Increasing parameter count」（表格列出 13M、30M、50M、1.4B 等参数量，学习率 3e-3、1e-3、3e-4，epoch count 8、8、4、4）和「Varying total token count D」（比较 Baseline、Repeated samples、Ensemble strategies 等曲线）。讲者强调，红色那条是标准的、非常可预测的数据缩放定律；而正则化、加 ensemble 这些干预虽然都改善了性能，但「the slopes look actually surprisingly similar」——斜率惊人地相似。这是他给全课的一句忠告：一旦你自己开始拟合 scaling law 就会发现，绝大多数干预只改截距，很少改斜率。

![数据筛选随规模自适应：质量-数量权衡与不同算力下的最优数据池](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0023_00-27-24.jpg)

本章最后一段把「数据是有限且重复会贬值」推到数据过滤（data filtering）。讲者说，很多我们静态看待的量其实是尺度依赖（scale dependent）的：如果你我明天要给自己训练的小模型筛数据，算力不多，反正也训不完整个互联网，那就会非常激进地过滤，只留最高质量的部分；但如果你算力极大，就想训越来越多的东西，因为不想在那一小撮高质量数据上反复重复，于是过滤器会越来越松，甚至开始纳入质量更低的数据。他因此说「the filters can't stay fixed as a function of the data」「the optimal filters turn out to not be fixed as a function of scale」。屏幕上「Data selection scaling and accounting for finiteness」这页把结论写成粗体「Given that repeated data is less valuable..」与「Data selection should then be adaptive to scale!」，右上角是论文「Scaling Laws for Data Filtering — Data Curation cannot be Compute Agnostic」（Goyal、Maini 等，CMU / Bosch Center for AI）。左侧的竖条标着「Web Data is Non-Homogenous」，从上到下是 E、D、C、A、B 等「Lower Quality Data Pools」；中间是「Lower Utility of Repeated Data」与「Quality-Quantity Tradeoff (QQT) for Data Filtering」的网格，按 Epoch 1/2/3 与 Pool 1 vs Pool 2 排出不同组合；右侧「Best Data Pool Changes with Total Compute」把 Small / Medium / Large Compute 分别对应到只含 E、含 E+D、含 E+D+C 的数据池。最右边的「Estimated Scaling Curves」纵轴是 ImageNet-1k Estimated Error（0.60 到 0.95），横轴是 Millions of Total Training Samples Seen（$10^2$ 到 $10^3$），四条曲线分别标为 Bucket E only、E + D (Pool 1)、E+D+C (Pool 2)、E+D+C+A，并在图上直接标出三段结论：Small Compute 用 Highly aggressive filtering is best，Medium Compute 用 Mildly aggressive filtering is best，Large Compute 用 Less aggressive filtering is best。

易错点相当集中：「4 个 epoch 以内不受伤」是绑定在「标准训练配方」上的经验结论，换配方未必成立；$D'$ 公式里的 $R_D^*$ 是要拟合的常数，不能当成普适常数；「无限算力」设定下把 scaling law 当成下界（lower bound）而非上界，是避免误用的关键，因为盲目套用会「break」；而过滤策略必须随算力重新调，把某次小规模实验里最优的过滤器一路照搬到大规模，正是本章要否定的静态思维。与前后内容的关系：本章第三次重复了「干预改截距、斜率不变」这一母题（数据组成 → 重复/集成 → 过滤），并在 28:29 用「So, that's it for data scaling laws」正式给数据缩放定律部分画上句号，下一章转入对坐标轴与函数形式的反思。

## 数据缩放定律小结与坐标轴、函数形式的陷阱 [28:29 - 30:17]

讲者在 28:29 用一段相当克制的话为整个数据缩放定律部分定性：这是全讲里「最直白、最容易理解」的一段，即使以一个统计/机器学习背景的人来看也很简单——增加数据，性能以高度可预测的方式变好，「I think this is uncontroversial」。他补充说，真正仍显奇妙的只有一点：对这么大的模型，下降居然是多项式（polynomially）的；但除此之外「nothing really crazy」。屏幕上「Recap: data scaling laws」这页用四条 bullet 把这一部分的骨架钉死：第一，「Remarkably linear relationship between log-data size and log-error」，即 log 数据量与 log 误差之间显著线性，这正是 $\log L = -\alpha\log D + c$ 的经验表述；第二，「Holds across domains and models」，跨领域、跨模型都成立；第三，「Theory understanding: similar to generalization bounds: mean estimation example」，理论理解上类似泛化界，均值估计是样板例子；第四，「Applications: data collection / curation」，落到应用就是数据收集与策划（curation）。这四条既呼应了本段开头的内在维度讨论，也预告了后面「数据该怎么选」的工程化内容。

紧接着 29:02 的学生提问，把注意力从「定律是什么」拉到「图是怎么画的」。学生问：在 pre-training under infinite compute 那页里，右边那张图在 loss 和 dataset size 上看起来都是线性的，而之前举的例子是 log-log 刻度，为什么这里是线性的？讲者的回答先是一句自嘲「this is a bad axis. Herman and I just talked about bad axes like this」，然后给了两层解释。第一层是横轴：「this is actually doubling, right? So, it looks linear but it's actually log scale」——横轴其实按倍增（doubling）排布，本质是对数刻度，只是标法让人误以为是线性；旁边还有人插话「You fixed it now?」。第二层是纵轴：「the Y axis here, it doesn't look logged but the range of this is so small that linear logs are basically the same」——纵轴看起来没取对数，但因为数值范围太窄，线性与对数在这个窗口里几乎重合。最关键的澄清是：「If you look at the fitted functional forms here, right? These are still polynomials. So, we're fitting the same functional forms and those are the ones that are plotted here.」也就是说，图上画的仍然是同一批多项式（幂律）拟合，并没有换模型，只是坐标轴和取值范围造成了「线性」的错觉。

由此讲者引出一个他会「到讲座末尾再讲」、但在这里先埋下的普遍警告：如果只覆盖了很窄的一小片算力区间（「a tiny slice of a compute range」），就极难分辨某件事到底是多项式 scaling 还是指数 scaling，因为「Taylor approximations are a thing. Everything looks linear if you kind of zoom in enough」。用一阶展开写出来就是：任何光滑函数 $f$ 在足够小的邻域内都满足 $f(x)\approx f(x_0)+f'(x_0)(x-x_0)$，于是幂律 $D^{-\alpha}$ 与指数形式 $e^{-\lambda D}$ 在局部看起来都只是一条直线，肉眼看不出差别。他因此收尾说，对「这到底是多项式还是别的功能形式」要「always a little bit careful and skeptical」。这条警告与本段开头对内在维度估计量的怀疑是同一种态度：可预测性不等于已被证明，拟合得好不等于函数形式选对了。

本章的易错点值得单独拎出来，因为它们最容易让整份笔记前面的结论失效。第一，坐标轴的标法会骗人：横轴按倍增排列时看起来像线性，实际是对数；纵轴范围过窄时线性与对数不可区分，所以「看起来是直线」不能直接等价于「存在幂律」。第二，窄区间内的函数形式不可辨识（identifiability）：只有跨多个数量级的 compute 范围，$\log L = -\alpha\log D + c$ 与指数形式才可能被区分开，这也反过来解释了为什么前面反复强调「跨多个数量级」的可预测性来之不易。第三，把拟合出的多项式外推到远超出观测范围的规模，本质上是拿 Taylor 局部近似去做全局预测，风险随外推倍数放大。与前后内容的关系也很清楚：这一章既是对数据缩放定律（19:29–28:28）的收束与方法论自检，也是 30:17 那句「So, that was scaling laws for data... Now I want to sort of talk about more exotic forms of scaling」的跳板——后半段要处理的架构、优化器、模型形状，其可预测性比数据缩放脆弱得多，因此这里的怀疑态度在后面只会更重要。

## 模型工程与架构比较：Transformer 真的优于 LSTM 吗 [30:17 - 34:48]

讲者在 30:17 正式换轨：数据缩放定律「in many ways straightforward」，接下来要谈「more exotic forms of scaling」，也就是面向模型工程（model engineering）的 scaling law，目标是用它来设计大语言模型。他用两个玩笑式的「激进派」把问题摆出来：也许你相信 LSTM 才是未来，想把 B200 的算力砸在 LSTM 上；也许你想用 SGD 训练。更现实的一层是资源分配：手头资源有限，到底是该训更久、把模型做大、还是再去多收数据——「There's all sorts of tradeoffs always at play, and scaling laws give us quantitative ways of making these tradeoffs」。屏幕上「Hyperparameter questions」这页写明，接下来要在经典的 Kaplan scaling paper 语境下考察四类选择：Architecture、Optimizer、Aspect ratio / depth、Batch size，并把这与第二讲「看别人怎么做」的视角对照，称这次要给出第一性原理（first principle）式的回答，代价是你必须先「believe in scaling laws in this lecture」。

第一个具体问题是「Are transformers really better than LSTMs?」。讲者先给了对手应有的尊重：LSTM 也能拟合分布，随着 Mamba 这类工作出现我们也知道 SSM（state space models）能用，所以问题不是「能不能用」而是「是不是更好」。暴力做法（brute force way）是花几千万美元去训一个 LSTM 版的 GPT-3 甚至更大模型；scaling law 的做法则是训一堆小模型：让 Transformer 跨若干 compute 区间，同时训练不同层数（different numbers of layers）的 LSTM 也跨若干 compute 区间，然后比较两条曲线。判据可以写成在 log-log 下拟合 $L(C)=A\,C^{-\alpha}+L_\infty$（或简写 $\log L = -\alpha\log C + b$），比较两族的斜率 $\alpha$ 与截距 $b$。讲者说结果会显示 LSTM「definitely different intercept, maybe even different slope」——截距肯定不同、斜率甚至也可能更差，因此这张图就足以支持「放大 Transformer 而不是 LSTM」。

![Transformer 与 LSTM 的测试损失-参数量对比：LSTM 截距更差且高规模处被反超](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0027_00-32-42.jpg)

「1. Architecture: transformers vs LSTMs」这页把上述比较画得很具体：问题框里写着「Q: Are transformers better than LSTMs?」和「Brute force way: spend tens of millions to train a LSTM GPT-3」，下面是「Scaling law way:」配一张图。纵轴 Test Loss 从 2.4 到 5.4，横轴 Parameters (non-embedding) 从 $10^5$ 到 $10^9$；蓝色曲线标为 Transformers，一路平稳下降到约 2.4；红色曲线组标为 LSTMs，并分别注明 1 Layer、2 Layers、4 Layers，几条红线在小规模处还能跟上甚至局部交叉，但在 $10^7$ 之后明显抬离蓝线、末端停在 3.0 以上。右下角标注 [Kaplan+ 2021]。讲者据此补充一条对架构论文写作的观察：今天的 Mamba、gated DeltaNet 一类论文几乎都会放一张这种图，图里必有 vanilla transformer 与「our really cool model」，而且「our really cool model is usually either on top or below」；你最不想要的就是「slopes are worse」，因为斜率更差意味着随着规模越做越大，你的模型终究会被反超（「as you scale up larger and larger, your models will eventually do worse」）。这正是「不训练巨型模型也能推理架构」的价值所在。

讲者接着说，近年好的架构 scaling law 论文不算多，他特别喜欢的是 E T 等人在 Google 时期对 T5-style 模型做的系统性 scaling 研究：把 T5 的越来越大变体在多种架构上训练。屏幕上「1. Many architectures / Cross-architecture scaling in Ta[5]」这页，左侧是 Negative Log Perplexity 对 FLOPs 的散点图，右侧并排 11 张小倍数图（small multiples），每张对应一种架构变体的 scaling 趋势，可辨认的标签里包括 (i) Performer、(j) Switch Transformer、(k) Universal Transformer、(e) Transformer-GLU 等。讲者逐条点评，并把它们与今天前沿模型实际采用的改动对照：Performer 这类高效注意力（efficient attention）「doesn't scale very well」，所以「we do not implement those」；真正被采纳的是 gated linear unit（GLU，门控线性单元），「Red line better than green line throughout the scaling trends」；Switch Transformer 整体 scaling 趋势不错，只是「I don't know what happened with their biggest run」——最大那次 run 出了状况；还有 mixture of softmax（他一时想起来后补上），按 scaling 趋势看「this is probably an effective intervention」，但「Actually, we don't do this anymore」。

这段最值得记的是那个近乎范式化的判据：很多人把 scaling law 当成「Oh, if it doesn't show up in the scaling law, it's not a good intervention」——如果某个改动在 scaling 曲线上不显现优势，就不算好干预。理由在于，这些论文虽然跑在「much, much smaller compute scales」上，却已经看出了后来驱动前沿模型的架构趋势。但易错点同样明显：第一，小规模上的截距优势不能替代斜率判断，「斜率更差会在大规模被反超」是硬约束；第二，Switch Transformer 最大 run 的异常说明单点噪声会破坏趋势判读；第三，mixture of softmax 按趋势「可能有效」却最终被弃用，说明 scaling 趋势本身也不是充分条件；第四，这些研究的算力规模远小于今天，把结论外推若干数量级仍需谨慎；第五，「不在 scaling law 上出现就不是好干预」是启发式（heuristic）而非定理。与上一章的关系是：数据缩放定律被判定为「简单、无争议」，而这里开始的架构比较属于「more exotic」，可预测性更弱、更依赖人为选对 x 轴——这个伏笔会在本段最后一章被明确点破。

## 优化器与模型形状：SGD 对 Adam、aspect ratio [34:48 - 37:34]

把架构比较的方法推广开去，第一个问题是「is SGD better or worse than Adam?」。讲者说这项分析来自 Hestness，结论「very clear trends」，而且又一次撞上那个「really mysterious thing」：截距不同，但斜率非常相似（「the intercepts are different, but the slopes are very similar」）。他特别限定条件——「if you fix the data and you fix the models」，在固定数据与固定模型的设定下，斜率往往「very, very similar」；即便像 SGD 换到 Adam 这样「an intervention as big as SGD to Adam」、对训练者而言天翻地覆的改动，scaling 趋势也「remain roughly the same」。他坦言自己每次看到都惊讶，但「yet it's very true」，而且「it's rare to get different slopes」。最后提醒一句适用范围：这里的实验是「language modeling with a recurrent highway net」，即循环高速网络上的语言建模，不是 Transformer。

![SGD 与 Adam 的验证损失-数据量对比：两条趋势线指数仅差 0.001](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0029_00-35-29.jpg)

「2. Optimizer choice / What about ADAM vs SGD?」这页给出了可逐字抄录的量化证据。纵轴是 Minimum Validation Loss (Top-scale)，刻度 0.86 到 1.59；横轴是 Training Data Set Size, Number of Chars (Log-scale)，刻度 $2^{19}$ 到 $2^{27}$。图例四条：Depth-10 RHNs, SGD（蓝实线）、Depth-10 RHNs, Adam（橙实线）、Depth-10 RHNs, SGD trend（绿虚线）、Depth-10 RHNs, Adam Trend（红虚线）。图上直接印了两条拟合式：

$$
\epsilon(m) = 5.37\,m^{-0.094},\qquad \epsilon(m) = 5.25\,m^{-0.095}
$$

前置系数 5.37 对 5.25、指数 $-0.094$ 对 $-0.095$，指数只差 0.001——这就是「斜率相同、只有截距变」最干净的数字证据。右下角标注 [Hestness+ 2017]，页面底部还有一行括号注释：「(Note, this is in 2017, so pre-transformers. RHN is recurrent highway nets)」，明确限定了年代与模型族。换句话说，Adam 相对 SGD 的收益几乎全部体现为把整条曲线往下平移，而不是改变下降速率；这与上一章「很多干预只改截距」的忠告完全一致，也是本段第三次出现同一母题。

第二个问题是模型形状（model shape）。讲者回扣第二讲：当时谈 aspect ratio（长宽比，即深度-宽度权衡，depth-to-width tradeoff）时给的建议是「随便挑个大致合理的数」，甚至调侃说是「four times the reasonable multiplier」；而做 scaling 分析能精确得多。先看层数（layers）作为横轴的 scaling 趋势：层数极少非常糟，「if you have one layer, you're not going anywhere」，是「a terrible, terrible scaling trend」；但「Greater than one layer, actually, it's surprisingly more competitive」，超过一层之后立刻变得有竞争力，而且「at every compute level, more layers... is better」。更精细的做法是 Kaplan 里那种寻找尺度不变量（scale-invariant quantities）的分析：层数 $n_{\text{layer}}$ 本身不是尺度不变量，模型越大你确实想要更多层；但 aspect ratio $d_{\text{model}}/n_{\text{layer}}$ 可能是——随着模型放大，最优 aspect ratio 或许应该保持不变。看 aspect ratio 的曲线会发现，不同模型规模下的最小值「basically the same minima around 100 $d_{\text{model}}$ for every layer, or maybe a little bit less」，即大约每层 100 维的 $d_{\text{model}}$；不过「it does shift a little bit」，存在向「更深的模型配更小 aspect ratio」的轻微偏移，但大体稳定。同样的分析还能用到 attention head dimension 等超参上。

屏幕上「3. Depth/Width: and other Transformer hypers」这页问「Do hyperparameters like the aspect ratio depend on scale?」，并排三张小图：Feed-Forward Ratio（$d_{ff}/d_{model}$，标注 50M Params）、Aspect Ratio（$d_{\text{model}}/n_{\text{layer}}$）、Attention Head Dimension（标注 25M Params），图里还写着「A wide range of architectures achieve similar performance」以及「22% additional compute compensates for 1% loss increase」。页面下方的说明文字值得转述：在固定非 embedding 参数总数 $N$ 时，模型形状对性能影响「very mildly」，损失在很宽的形状范围内「varies only a few percent」；微小差异是用「the fit to $L(N)$ as a baseline」来补偿的；aspect ratio 尤其可以变化「a factor of 40」而只轻微损害性能，例如 $(n_{\text{layer}}, d_{\text{model}}) = (6, 4288)$ 能达到 $(48, 1600)$ 那个模型（[RW+19] 所用的形状）的「within 3%」。讲者据此说，如果你的策略是「固定 aspect ratio 然后放大」，这类图能让你确信「our optimum isn't shifting too much as I go to larger and larger models」。

易错点有四条。第一，2017 年、pre-transformer、RHN 模型族，把 $\epsilon(m)=5.37m^{-0.094}$ 这类结论直接搬到今天的 Transformer 上并不稳妥，讲者自己就用括号注明了年代。第二，斜率不变是反复出现的经验惊讶（empirical surprise），不是可证明的性质，它依赖「固定数据与模型」这一前提；一旦放松，斜率未必不变。第三，aspect ratio 的最小值只是「roughly similar」，会随深度轻微偏移，把「固定 aspect ratio」当成严格不变会掩盖这 3% 级的差异。第四，「22% additional compute compensates for 1% loss increase」说明形状效应很容易和算力差异混淆，比较时必须像 Kaplan 那样以 $L(N)$ 拟合为基线扣除算力因素。与前后内容的关系：本章继续第六章开启的模型工程清单（从 Architecture 走到 Optimizer 与 Aspect ratio / depth），并把「截距变、斜率不变」这一母题做了一次最精确的数值展示；而下一章会指出，连「参数」这个最基础的横轴都不是天然的，从而把整段的怀疑推向顶点。

## 并非所有参数都等价：参数计数与被工程化的可预测性 [37:34 - 38:58]

讲者在 37:34 用一个很重的措辞引出本段最后一章：「One thing that's important, and this is a huge can of worms」，而且「a sufficiently big can of worms that I will talk about it on more than one slide. It will come back.」核心命题只有一句：「not all parameters are created equal」——并非所有参数都生而平等。由此推出的后果直接动摇前面所有工作：因为参数不等价，你的 scaling law 看起来是好是坏，会取决于你「怎么定义一个参数」。换句话说，前面几章一直在用的 $L(N)$ 或 $\log L = -\alpha\log N + b$，其横轴 $N$ 本身就是一个需要人为决定的选择，而不是客观给定的量。

Kaplan 论文里的具体事件是这样的：当他们把 embedding parameters（嵌入层参数）计入、按深度画 scaling law 时，得到的是「very funky-looking scaling laws」——形状古怪、不成直线；他们的反应是「Oh, this is no good at all, no good at all」，于是决定把所有 embedding 参数排除掉，只统计 non-embedding parameters（非嵌入参数）。给自己的理由是「These are the parameters that are doing computation or whatever」，即认为只有这些参数在真正做计算。讲者立刻预告：「we're going to see later that this is going to have non-trivial implications on the validity of some of the results」——这个看似无害的计数选择，后面会对若干结论的有效性产生实质影响（正是后续 Kaplan 与 Chinchilla 差异讨论里要回到的地方，呼应了他开头说的「It will come back」）。

![计入与不计入 embedding 参数的两张深度缩放图对比：前者曲线弯曲，后者近似平行直线](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0031_00-38-16.jpg)

「3. Depth/Width: But not all parameters are made equal」这页把后果画成并排两张图。页面上写着「We've been thinking about 'parameters' but not all parameters are equal」。左图纵轴 Test Loss 从 2 到 7，横轴 Parameters (with embedding) 从 $10^6$ 到 $10^9$，图例依次为 0 Layer、1 Layer、2 Layers、3 Layers、6 Layers、> 6 Layers；其中 0 Layer 那条深蓝曲线几乎水平地停在 6 附近，其余曲线则彼此分叉、弯折，完全不是干净的幂律。右图横轴换成 Parameters (non-embedding) 从 $10^3$ 到 $10^9$，图例去掉 0 Layer，只剩 1 Layer、2 Layers、3 Layers、6 Layers、> 6 Layers，几条曲线立刻变成近乎平行、近似直线的下降趋势。两图下方一句粗体结论：「Embedding layer parameters don't behave the same!」；再下一行「Related: recent papers on scaling laws for mixtures of experts.」把问题外延到专家混合（MoE）——那里总参数与激活参数同样被解耦，参数计数更不唯一。这两张图的对照说明：把横轴从 $N_{\text{non-emb}}$ 换成 $N_{\text{with-emb}}$，同一批模型、同一批损失值，$\log L = -\alpha\log N + b$ 是否成立就变了；所谓「scaling law 长什么样」，一半取决于你怎么数参数。

于是讲者给出全段的收束判断，也是他明确归给 Percy 的观点：scaling law 与「predictability across scales」并不是魔法，而是「engineered」——被工程化出来的，不会自动发生。要让跨多个数量级算力（「many orders of magnitude compute」）的缩放变得可预测，需要两件事同时到位：一是「pick the right kinds of x-axes to look at」，选对横轴（这正是 embedding 参数计数教给我们的）；二是「make sure that the hyperparameters for these things are set right」，把超参设对（呼应前面学习率、batch、warm-up 那些细节）。「And only under those conditions does it become possible to get predictable scaling」——只有在这些条件下，可预测性才成立。他最后说，这其实就是很多 scaling law 研究者的日常工作：琢磨「what the right way is to get very predictable scaling」，即怎样构造实验才能得到很干净的缩放规律。

本章的易错点几乎是对整段的方法论总清算。第一，把参数计数当成中性操作是错的：排除 embedding 参数是「事后」为了得到好看曲线而做的选择，理由（「这些参数在做计算」）只是自洽的说辞，且已被预告会实质影响结论有效性。第二，左图里 0 Layer 那条平坦曲线是反例：一旦计入 embedding 参数，连「模型越大损失越低」这个最朴素的单调性都救不回来，说明「funky」不是拟合噪声而是结构性伪影。第三，由此回看，前面所有「截距变、斜率不变」的结论都隐含了「用同一套 $N$ 定义」这一前提，换计数方式后未必仍成立。第四，可预测性是需要构造的：选错 x 轴、超参没调好，就得不到跨数量级的直线；反过来，看到一条漂亮直线也不能证明你找到了「自然律」，它可能只是被工程化出来的结果。与前后内容的关系上，本章既收束了 30:17 以来「更 exotic 的 scaling」这一整条线，也把 28:29 那章埋下的「对函数形式保持怀疑」升级成对「横轴与实验设定」的怀疑，并用「It will come back / we're going to see later」明确指向本讲后面关于 Kaplan 与 Chinchilla 差异、以及 MoE 参数计数的讨论——全段由此停在一个刻意的开放处，而不是一个圆满结论上。

## MoE 场景下的「参数价值」：总参数与激活参数解耦后的 scaling [38:58 - 41:02]

本段以一段 side note 开场，讨论 mixture of experts（MoE，混合专家）成为训练大模型的主导方式之后，「一个参数到底意味着什么」这个问题如何被改写。讲者指出，当 total parameters（总参数量）与 active parameters（激活参数量）被解耦——每个 token 只被路由到少数专家、模型可以拥有海量但并非全部被使用的权重——过去建立在稠密模型上的参数 scaling law 就不能再直接套用；在这个新世界里，「什么算一个参数」本身需要重新定义。这里引用的是 Apple 与 MIT 一组干净的分析，幻灯片右上角给出的论文标题为 "Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models"，署名包括 Samira Abnar（Apple）、Harshay Shah（MIT）、Dan Busbridge（Apple）、Alaaeldin Mohamed Elnouby Ali（Apple）、Josh Susskind（Apple）与 Vimal Thilak（Apple）。主标题写作 "Side note – 'Value of parameters'"，副标题是 "With MoEs, we expect parameter scaling to change"，直接点明这是对前面模型工程缩放章节的补充与修正，而不是一套全新的理论。

![MoE「Value of parameters」幻灯片：总参数与激活参数两族 U 形曲线、稀疏度色条及 IsoFLOP 三维曲面](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0032_00-39-34.jpg)

幻灯片原样列出的两幅子图与三维曲面标题如下：

```text
Side note – 'Value of parameters'
With MoEs, we expect parameter scaling to change
(b) Optimal Total Parameters N*        (c) Optimal Active Parameters N_a*
(b) IsoFLOP surface over sparsity and active parameters
```

左图 (b) "Optimal Total Parameters $N^*$" 的纵轴是 Pretraining Loss $L$，刻度约 2.2 到 3.0；横轴是 Total Parameters $N$，刻度依次为 178M、485M、1B、4B、10B、26B。每一条 U 形曲线对应一个固定 compute 预算，颜色由下方的 MoE Sparsity $S$ 色条编码，从 0%、24%、48%、71% 一直到 95%；星号标注 "Optimal total parameters"，落在约 4B 至 10B 之间、损失约 2.2 处。图 (c) "Optimal Active Parameters $N_a^*$" 纵轴同为 Pretraining Loss $L$（约 2.2 到 2.7），横轴换成 Active Parameters $N_a$，刻度为 178M、294M、485M、800M、1B、2B、4B，色条同样是 0% 到 95% 的 MoE Sparsity $S$，星号标注 "Optimal active parameters"。右侧另有两条独立色条：Sparsity $S$ 从 0% 经 50% 到 98%，Loss $\mathcal{L}$ 从 2.1 到 2.8。三维图 "(b) IsoFLOP surface over sparsity and active parameters" 的三条轴分别是 Active Parameters $N_{ACTIVE}$（标注 10B、8B、6B、4B、2B、1B、800M、400M、200M、100M 等）、MoE Sparsity $S$ 与 Loss $\mathcal{L}$，红紫色圆点沿曲面成串排布。

讲者的读图结论是：沿 x 轴不断增大总参数量并想最小化损失时，最优点会越来越偏向稀疏模型，也就是颜色变得越来越深；反过来，激活参数量 $N_a$ 会随 MoE sparsity $S$ 增大而下降。更重要的是，对每一个 compute 量都能画出这样的曲面，于是可以问「这批 flops 该花在更多 active parameters 上，还是花在更高的 sparsity 上」，而改变稀疏度几乎不改动所花的 compute。若在「相同激活参数、更多总参数」的方向上从曲面后方移动到前方，损失仍在改善——换句话说，MoE 里那些不被激活的参数依旧在帮助降低损失，讲者称这一点 "kind of cool"。这与上一段讨论参数计数时「embedding 参数与其余参数行为不同」的观察属同一条线索：parameter 不再是一个单一标量，必须区分 $N$、$N_a$ 与 $S$。这里的易错点是把「总参数多」误当成「FLOPs 多」或「一定更优」：0% 稀疏度对应稠密模型，95% 甚至 98% 对应几乎只激活极少专家；总参数可以极大而每 token 计算量几乎不变。局限也很明确，讲者说这些量都有可写出来的函数形式与可预测的 scaling 规律，但明确表示 "I won't go into to that part in more detail"，本段不给出公式，也不展开不同路由策略下是否还成立。

## 大 run 必须重调的两个量：批大小、学习率与 critical batch size 的直觉 [41:03 - 44:11]

讲者随即把话题从「参数」转到「优化超参」。当你要训练一个全新的大模型时，绝大多数选择其实是固定的：你大概不会激进到换回 LSTM，多半会挑一个受 DeepSeek 启发的 MoE 之类经过验证的架构，架构选择是 tried and tested 的，不会把一切推倒重来。真正需要「重新发明」的只有两个量：batch size（批大小）与 learning rate（学习率），而这两者又强耦合——改一个就必须改另一个。批大小之所以格外棘手还多一层系统原因：上一讲讲过 data parallel（数据并行）要求大 batch，所以从并行效率出发我们希望 batch 尽可能大。于是相关的系统问题变成：在「开始吃亏」之前，批大小最大能开到多少？这个上限又是如何随 model size 变化的？

![Critical batch size 幻灯片：局部二次等高线图与 Predicted Training Speed 曲线，标注完美 scaling 与无效 scaling](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0034_00-42-01.jpg)

由此引出被大量论文频繁使用的 critical batch size（临界批大小）。幻灯片标题为 "4. Batch size: Critical batch size"，右上角引文来自 "An Empirical Model of Large-Batch Training"，作者是 Sam McCandlish（OpenAI）、Jared Kaplan（Johns Hopkins University, OpenAI）、Dario Amodei（OpenAI）以及 the OpenAI DDA Team。左侧是一张目标函数的局部等高线图：同心椭圆等高线分别标着数值 0.02、0.10、0.20、0.40、0.70、1.00；黑点是起点，图例给出 "Smaller batch"（红色）与 "Larger batch"（蓝色），红色短箭头末端是一个红 ×，蓝色长箭头末端是蓝点，两者都指向局部下降方向却够不到中心极小值，浅色扇形区域表示可行的更新方向。右侧 "Predicted Training Speed" 图的纵轴是 $\epsilon_{opt}(B)/\epsilon_{max}$（对数刻度 $10^{-2}$、$10^{-1}$、$10^0$），横轴是 "Batch Size / Noise Scale $(B/\mathcal{B})$"（从 $10^{-2}$ 到 $10^2$）；曲线自左下 "Perfect scaling" 近似线性上升，在 $10^0$ 处有一条竖直虚线，随后趋于饱和进入 "Ineffective scaling"。页脚两行原文是 "Batch size – known to have strong diminishing returns past a certain point." 以及 "Critical batch = min number of examples before diminishing returns (what does that mean?)"。

概念上，临界点之前是 noise limited regime（噪声限制区）：每往 batch 里多放一个样本，就会降低一次 SGD 更新的梯度噪声，由于此时系统是 variance limited，方差下降极其有用，于是得到「完美 scaling」——多处理的样本几乎都换成了等价的进度，理想情况下 $S(B)\cdot B\approx\text{const}$，即所需步数随 $B$ 成反比下降，$\epsilon_{opt}(B)/\epsilon_{max}\approx B/\mathcal{B}$。一旦 batch 增大到触及梯度自身的噪声尺度，就进入 bias limited regime（偏差限制区）：梯度下降只看得到目标函数的局部结构，没有全局极小值在哪里的视野，无论把噪声压得多低，更新方向也只能指向图中那个浅蓝的局部方向，它与全局最优方向之间总有系统性偏差，于是继续增大 batch 只剩收益递减。讲者把 critical batch 定性为一个 rule of thumb、一个方便的 trade-off 点，即从完美 scaling 区跨到无效 scaling 区的 crossover，用来回答「想要尽可能大的 batch 又不吃巨大效率损失时该设多少」。

理解这一章要抓住它与前后内容的关系：前一段讲的是 MoE 与架构这类「模型参数」的 scaling，从这里开始转向「优化超参」的 scaling，而 Ch3 会把这个直觉变成可拟合的方程、Ch4 再说明它为什么配叫 scaling law。易错点有三处：bias limited 并不是「梯度算错了」，而是局部二次近似下方向本身有偏；critical batch 不是硬性阈值而是平滑的 crossover；大 batch 的主要动机是 data parallel 的并行机会，而不是统计效率本身变高。局限在于，讲者明说 critical batch size 是从「对目标做局部二次近似」的一堆复杂计算里推出来的，本段只给机械步骤、不给推导，因此把它当成一个经验标尺而非理论常数更稳妥。

## Critical batch size 的精确定义与拟合步骤 [44:12 - 46:52]

上一章给了直觉，这一章给出可操作的估计流程。讲者把目标重述得很直白：先挑一个 target loss（目标损失，某个你想命中的数值），然后以「尽可能快地达到这个目标损失」为准则；接着 sweep over batches，也就是在一系列不同 batch size 上分别训练，并记录两件事——达到目标损失所需的步数 $S$（steps needed）与所消耗的样本数 $E$（examples needed）。二者并非独立，讲义里写明步数乘以 batch size 等于样本数，即

$$
E = B\cdot S .
$$

幻灯片 "4. Critical batch size definitions.." 的原文要点是 "What is critical batch size, more precisely?"，其下列出 "Pick a target loss and train 1). Steps needed (S), 2). Examples needed."、"Sweep over batches"、"The curve should follow roughly.."。OpenAI 等人通过若干计算论证，步数与样本数之间近似满足一个反比型的函数形式：

$$
\frac{S}{S_{min}} - 1 = \left(\frac{E}{E_{min}} - 1\right)^{-1}.
$$

![Critical batch size definitions 幻灯片：S/S_min 与 E/E_min 的反比关系式及 B_crit 定义公式](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0036_00-44-27.jpg)

式中 $S_{min}$ 是「可能的最少步数」，$E_{min}$ 是「可能的最少样本数」，两者作为归一化常数把 $S$ 与 $E$ 拉到同一尺度上。这个式子编码了很自然的权衡：若想最小化样本数 $E$，就得忍受很多步 $S$；若想最小化步数 $S$，就得消耗很多样本 $E$；而调大或调小 batch size，本质上就是在这个权衡里向一侧或另一侧滑动。既然两端都想要，OpenAI 的论证是「让这两项大致平衡」，把上式对平衡点求解，就得到幻灯片上 "Fit for $S_{min}$, $E_{min}$ on this curve and pick" 之后给出的定义：

$$
\mathcal{B}_{crit} = \frac{E_{min}}{S_{min}} .
$$

也就是用「所需最少样本数」除以「所需最少步数」，其中 $S_{min}$、$E_{min}$ 都是把观测到的 $(S,E)$ 数据拟合到上述曲线上估出来的参数。幻灯片底部一句话点出代价与收益："This balances both sides of the equation, giving roughly 2x the steps / passes optimal"——这样选出来的批大小会比「最优步数」多一点步、比「最优样本数」多消耗一些 passes（量级上说约 2x），但它平衡了等式两边，等价于在「优化收敛速率」的同时不让步数爆炸。页脚还有一行小字，正是讲者说「不再展开、想深究请读 critical batch size 原论文」的那点："(this is claimed to be close to the ratio of the trace of the gradient covariance and squared norm of the gradient)"，也就是说这个量据称近似等于梯度协方差的迹与梯度平方范数之比，讲者在转写里同样只点到 "just looking at the ratio of the gradient covariance and the squared norm of your gradient" 为止。

这一章承接 Ch2 的「crossover」直觉，把它变成一个能拿来拟合、能拿来选 batch 的方程，同时为 Ch4 的 scaling 关系埋下伏笔：一旦 $B_{crit}$ 能被定义为 $E_{min}/S_{min}$，就可以追问它随 target loss 怎么变。易错点值得强调：$S_{min}$ 与 $E_{min}$ 是拟合参数，不是某次实验里真的测到的极小值；公式前讲者加了 "roughly"，它是经验近似而非恒等式；$\mathcal{B}_{crit}$ 是相对于所选 target loss 的量，换一个目标损失就要重新扫一遍。局限也很实在：要估它就得在多个 batch size 上各自跑到目标损失，成本不低，而且整个式子隐含「单一噪声尺度」这类假设，估计过程被讲者形容为 "quite complicated"，这也是他反复建议去读原论文的原因。

## 它为何属于 scaling law 讲座：临界批大小随目标损失的幂律增长 [46:53 - 48:22]

讲者在这里主动打断自己，替听众问了一个问题："So why is that even in the scaling lecture right? Like this is a thing that is not really related to scaling so far." 他给出两条理由。第一条是，critical batch size 本身就是一个合理的 batch size：它平衡了收敛速率与 batch 大小，因此「把批大小设成 $\mathcal{B}_{crit}$」是一个有原则的默认值。第二条才是它出现在 scaling law 讲座里的真正原因：把 target loss 想成 compute——想让模型越来越好，等价于不断把损失往下降——然后问这个临界批大小如何随目标损失变化。答案是，随着损失改善，$\mathcal{B}_{crit}$ 会增大，而且增大得非常可预测，又是一条 power law（幂律），讲者形容其大致是 "inverse polynomial in batch size" 的形式。

![Critical Batch Size vs. Performance 的 log-log 图：两条经验曲线、噪声尺度散点与 B_crit 幂律拟合](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0039_00-48-06.jpg)

幻灯片 "4. Batch size: critical batch size" 上方的小图标题是 "Critical Batch Size vs. Performance"：纵轴为 "Critical Batch Size (Tokens)"，对数刻度 $10^3$、$10^4$、$10^5$、$10^6$；横轴为 "WebText2 Train Loss"，刻度标成 $10^1$、$6\times10^0$、$4\times10^0$、$3\times10^0$，从左到右其实是损失在下降。图例给出蓝色实线 "Empirical $B_{crit}$, $N=3M$"、橙色实线 "Empirical $B_{crit}$, $N=85M$"、灰色虚线 "$B_{crit}=2.1\times10^8\ \text{tokens}\cdot L^{-4.8}$" 以及绿色散点 "Noise Scale Measurement"。也就是说，两个不同规模（$N=3M$ 与 $N=85M$）的模型，其经验临界批大小都能被一条以损失为自变量的幂律 $B_{crit}=2.1\times10^{8}\ \text{tokens}\cdot L^{-4.8}$ 大致描述，指数 $-4.8$ 相当陡，跨度从约 $10^3$ tokens 一直抬到 $10^6$ tokens。图中还给出最小 compute 的表达式

$$
C_{min}(C) \equiv \frac{C}{1 + B/B_{crit}(L)}\qquad (\text{minimum compute, at } B \ll B_{crit}),
$$

并在下方用大字写出 "The smaller the loss target, The bigger the batch"。讲者的解释与直觉一致：越靠近极小值，降方差的价值越大，因为你正在优化越来越细粒度的目标、要分辨越来越微小的差异，噪声因此更致命；所以一次真正大规模、一路走到图右侧的训练 run，可以合法地使用相当大的批大小。

Ch2、Ch3 讲的是批大小本身，Ch4 把它接回全讲主题，讲者随即用 "That's the batch size story" 收束，转入学习率。易错点有几处：横轴是 WebText2 的 train loss 且方向反直觉（左大右小），不是 compute；$2.1\times10^8$ 与 $-4.8$ 是这一设定下的拟合值，不是普适常数；$C_{min}$ 那一行明确标注只在 $B \ll B_{crit}$ 时成立，越界套用会失真。局限同样清楚：讲者说 learning rate 那一侧 "a little bit complicated"，要留到 advanced scaling lecture 才细讲，本段不展开；因此这里的幂律应理解为「临界批大小可预测」这一性质的例证，而不是一套完整的批大小调度方案。

## 学习率缩放：宽度 scaling、μP 与两条技术路线 [48:23 - 50:55]

两个必须重调的量里，批大小讲完，另一半是 learning rate（学习率）。讲者先给一句总纲："learning rates generally shift"，并说明细节要留到后面的 advanced scaling lecture。他给的标准心智图是一个普通 MLP：只做 width scaling（宽度缩放）、不做 depth scaling（深度缩放）。在这种设定下，模型越宽、参数越多、一次更新同时改动的东西越多，于是「也许每次应该少动一点」，学习率应随规模变小；一个广为流传的 rule of thumb 是按宽度的倒数缩放，即 $\eta \propto 1/\text{width}$，随 width 规律地减小学习率。与之并存的另一派做法是重新缩放网络本身：改动 initialization sizes（初始化尺度）、改动 optimizer 对各部分的 step sizes，目的都是强制模型在不同 scale 下拥有「同一个最优学习率」，这类思路以 μP（maximal update parametrization）及同类方法为代表。讲者对它的评价很克制：有人报告 "great success"，也有人报告 "less success"。

![Learning rates: muP and scale-aware LR choices 幻灯片：Standard Practice 与 Our Work 两幅曲线及 μP 缩放表](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0040_00-49-19.jpg)

幻灯片 "5. Learning rates: muP and scale-aware LR choices" 左图标 "Standard Practice"（署名 Yang et al 2022）：纵轴 Training Loss 从 3.5 到 7.0，横轴 $\log_2 \text{LearningRate}$ 从 −20 到 −10，Width 图例依次为 128、256、512、1024、2048、4096、8192，不同宽度的 U 形曲线最低点彼此错开，箭头标注 "optimum shifts"。右图标 "Our Work"（署名 Yao et al 2024）：横轴 $\log_2 \text{LearningRate}$ 标 −20、−18、−16、−14、−12、−10，曲线族在低损失处收拢对齐，箭头标注 "optimum stable"。页底结论是 "If we naively scale up – optimal learning rate depends on scale. We need scaling aware initialization and learning rate scaling"。右侧 Table 2 的原文规则如下，其中 $M'$ 是宽度放大 $r$ 倍的模型：

```text
Table 2. μP function for a model M' that is r times the widths of M.
matrix-like (2 infinite dims, e.g. fully-connected hidden layer):
    AdamW learning rate      l   -> l/r
    Initialization variance  σ   -> σ/r
others (0 or 1 infinite dim; embedding layers are "others"):
    AdamW learning rate      l   -> l
    Initialization variance  σ   -> σ
output layer (lm_head, maps infinite dim to finite dim):
    Multiplier (output)       τ   -> τ/r
    Multiplier (others)       τ   -> τ
```

表头还说明：若参数张量有两个维度在模型宽度趋于无穷时随之发散，就称 "matrix-like"（例如全连接隐藏层）；若这样的维度数是 1 或 0，则归入 "others" 类，并特别注明 embedding layers 属于 "others"；"Output" 指把无穷维映到有限维的层，也就是 Transformer 里的词解码层 $lm\_head$；multiplier 是乘在参数张量上的常数，作用类似 softmax temperature。于是选学习率归纳为两种哲学：一是「预测」——估计这些 minima 在哪、且 minima 随规模的变化 "is pretty predictable"，据此把最优学习率外推到大 run；二是「重参数化」——用 μP 之类手段让最优学习率跨 scale 保持不变，然后直接挑一个最好的学习率开跑。讲者说两种都被成功用在大规模训练里，但 "anecdotally" 似乎更多人偏向 scaling law 预测那一侧，两者都算 viable。

这一章与 Ch1、Ch2 形成呼应：架构可以照抄成熟方案，但 batch size 与 learning rate 必须重新定，而「超参能在训练前预测」这一主张又直接铺垫 Ch6 的 "Some surprising takeaways"。易错点是：结论只在「只做 width scaling、不做 depth scaling」的设定下成立；matrix-like 与 others 的划分决定到底要不要除以 $r$，embedding 被明确归到不缩放的 "others"；$l/r$、$\sigma/r$、$\tau/r$ 里的 $r$ 是宽度倍数而非参数量倍数。局限也须记下：μP 类方法的成功报告并不一致，且讲者明说这部分要等下一讲才细讲，本段只给心智图与表格，不给推导或实测指数。

## 上游困惑度与下游迁移的错配，以及 scaling law 设计流程 [50:56 - 53:41]

在进入「最后一部分」之前，讲者先用一个收尾观点提醒：upstream（上游，即预训练损失）与 downstream（下游任务）的表现可以差得很远。他举的是 T5 一系的架构论文（转写里称作 "ETE architectures paper"，幻灯片署名 Tay et al 2023）。在 log likelihood / negative log-perplexity 这一侧，参数量与指标高度相关：参数越多，模型越可预测地变好，散点排成 "a very beautiful linear trend"。可一旦照这个趋势去「发布我们最好的模型 NL-12」，实际下游最佳却是 NL-32 XL——一个在困惑度上明显更差的模型。讲者说这大概是他见过的「上游到下游相关性最差」的例子之一，但正因为极端，才更值得记住。

![Caution – scaling behaviors can differ downstream：负对数困惑度与 SuperGlue 准确率随参数量的对比散点图](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0042_00-51-45.jpg)

幻灯片 "Caution – scaling behaviors can differ downstream" 上写着 "Thus far: scaling is predictable and depends mainly on parameters" 与 "Catch: downstream scaling can often be much less predictable"。左图纵轴 "Negative Log-Perplexity" 从 −1.70 到 −1.35，横轴 "Params" 刻度为 $2.7\times10^8$、$5.4\times10^8$、$1.1\times10^9$、$2.1\times10^9$、$4.3\times10^9$、$8.6\times10^9$、$1.7\times10^{10}$；散点标注了 NL12、NL32-XL、NL6-XXL、NL12-XXL、NL8-XXL、NL32-LG、NL36、NL24、NL8-XL、Large、XL、NL32 等，负对数困惑度随参数量单调抬升，NL12 位于约 −1.36 的最高处。右图纵轴换成 "SuperGlue Accuracy"（72 到 80），横轴 Params 刻度相同，但排序被打乱：最高的是 NL32-XL（约 79.7），NL12 只有约 77.7，NL32 反而落到约 72.9 的最低处。讲者的结论是，scaling laws 一般而言应当施加在 perplexity 一侧——那里干净、规整、可预测；从 perplexity 到 downstream 的 transfer 远比初看起来不确定，必须非常谨慎。他还引了一段轶事：以前去做 post-training 的学生总抱怨「pre-training 的人把模型一交，说困惑度很好，剩下都是你的问题」，但问题往往在预训练阶段就已经埋下，所以别只盯困惑度，也要想 transfer。

紧接其后的幻灯片 "Some surprising takeaways" 把话说得更正面：一个圆角框里写 "The effect of hyperparameters on big LMs can be predicted before training!"，下面列出 Optimizer choice、Model depth、Architecture choice 三项；再给出 "The scaling law based design procedure." 三步：

```text
1. Train a few smaller models
2. Establish a scaling law (e.g. ADAM vs SGD scaling law)
3. Select optimal hyperparam based on the scaling law prediction.
```

![Some surprising takeaways 幻灯片：超参效应可在训练前预测，以及基于 scaling law 的三步设计流程](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0043_00-52-58.jpg)

讲者进一步把这套流程展开为 naive 版实践：训练一堆小模型，把直线拟合到 log-log 图上，如果 fit 足够好，就有信心认为这个 gap 会持续，然后把它部署到大规模训练 run 上；在大 run 之前本就该对「会看到什么」有大致预期，甚至能比较精确地预测数值，例如换一个 optimizer 相比另一个能带来多少 gain。他还反问「不然呢？凭空发一个 run 吗？」，认为这个 naive 版本已经 reasonably accurate 地刻画了实际做法。与前后内容的关系是：Ch5 说最优学习率可预测，这里把「可预测」推广到优化器、深度、架构并程序化；同时用 NL-12 与 NL-32 XL 的反例给这份乐观加了一道边界。易错点是别把「困惑度好」直接等同于「下游好」，也别把这个最差相关性的个例当成普遍规律；外推的可靠性依赖 fit 质量与「gap 会持续」这一假设。局限在于讲者自称这是 naive version，且 transfer 需要另行建立或验证，scaling law 本身并不自动保证下游收益。

## 课堂问答：scaling 实验的方差、下游指标与 train/test loss [53:42 - 56:36]

讲者在此暂停接受提问，三个问题把前面方法论的地基补得更清楚。第一个问题问：scaling law 上每个数据点的方差怎么看？要跑多少次？聚合时取 average、min 还是 max？回答是，就他见过的绝大多数（若非全部）情形，这些点是 singletons（单次运行）。原因是 perplexities 非常干净：训练数据同构、量很大、eval set 也很大，把方差压得极低，重跑一次的差异大概落在第二位小数（"the variance is in the second decimal place"）。但他立刻补了反例：若是 learning rate 或 critical batch size 这类量的 scaling law，会看到 "truly horrendous stuff"，噪声非常糟糕；他承认人们会做一些 variance reduction，却 "not that common. As common as it should be, maybe."，暗示这块实践上做得不够。

第二个问题问：有没有人直接在 downstream metrics 上做 scaling laws？回答是「有，但和上面同样的毛病」。讲者借此把 scaling law 的哲学讲透：关键是「至少可靠地建立一件事」。如果看到的是 transformer 那条非常规整、可能略微弯曲但确有 regularity 的趋势线，就能相信它会延续；可若换成一条 noisy jaggedy line，你根本分不清它该是直线还是曲线，什么都判断不了。他原话的意思可整理成下面的步骤：

```text
start with a low variance measurement,
and then establish scaling regularity there,
and then rely on some sort of belief about transfer
(or establish transfer elsewhere) to show it transfers to downstream.
```

第三个问题问：有些图是 test loss、有些是 train loss，何时用哪个？回答是：做「良好科学实践」时拟合 test loss（如 Kaplan 2021），但现实里除了极少数例外（例如 infinite compute 设定、数据 repetition 研究），几乎所有模型都是 one-pass SGD，泛化间隙非常小，train 与 val 非常接近，可写作 $|L_{\text{train}}-L_{\text{val}}|\approx 0$；他甚至见过预训练 codebase 根本不记录 validation loss，因为一次遍历下两者几乎一样。正因如此，很多图把 train loss 与 test loss 混用，反正都处在 one-pass regime。转写在 [56:37–56:45] 还留下一句 "Sorry, I moved on too far backwards here"：这段问答期间投影一度被误翻回此前的架构对比页，画面上能看到 "3. Depth/Width: But not all parameters are made equal"、"1. Architecture: transformers vs LSTMs"、"1. Many architectures" 等标题，其中一页页脚还写着 "Related: recent papers on scaling laws for mixtures of experts."，恰好回扣本段开头 MoE 的 side note。

这一章把 Ch6 的 transfer 担忧落到方法论根子上：先低方差、再 regularity、最后才谈迁移；one-pass 的结论也解释了为什么前面诸多图能混用 train/test loss。易错点是把 singleton 的结论从 perplexity 盲目推广到 LR、$\mathcal{B}_{crit}$ scaling；以及把 one-pass 下 train≈val 当成普适事实——一旦进入多 epoch 或重复数据设定，泛化间隙就不再能忽略。局限是讲者自己承认方差缩减做得不够、下游指标因噪声很难判定函数形式，因此这里的建议属于实践哲学而非硬性统计保证。

## 计算最优分配的动机：要不要更多数据或更大模型 [56:37 - 58:27]

本段最后转向「scaling law 最广为人知的一次应用」：讲者说即便你没亲自用过这套方法，也一定用过它给出的 rule of thumb，而且这组内容能让人看清 scaling law 在实际上是怎么被执行的。 motivating question 非常朴素：do I want more data or bigger models？资源是 compute——有人会 metaphorically 交给你一定数量的 flops，你得决定怎么花；回想 Percy 的第一讲，data 乘以 parameters 大致就是 flops，且是线性关系，可记作 $C \approx N D$。于是问题变成「我该要更多数据，还是要更大的模型」。

![One important use of scaling laws 幻灯片：Rosenfeld 与 Kaplan 的联合 data-model scaling 公式及损失曲面图](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0047_00-57-50.jpg)

幻灯片 "One important use of scaling laws" 原文依次是 "Q: Do we need more data or bigger models?"、"Clearly, lots of data is wasted on small models"、"Joint data-model scaling laws describe how the two relate"，随后列出两条联合律：

$$
\text{From Rosenfeld+ 2020:}\quad Error = n^{-\alpha} + m^{-\beta} + C
$$

$$
\text{From Kaplan+ 2020:}\quad Error = [m^{-\alpha} + n^{-1}]^{\beta}
$$

并断言 "Provides surprisingly good fits to model-data joint error."。两式都用 $n$、$m$ 两个变量分别刻画数据量与模型规模，幻灯片与讲者在得到本段时并未逐一指明谁对应谁，因此不宜替他强行对应；可确认的是 Rosenfeld 的形式是两个幂项加一个常数 $C$ 相加，Kaplan 的形式则是把两项之和整体再套一层幂 $\beta$。右上小图 "Loss vs Model and Dataset Size"：纵轴 Loss 约 2.5 到 4.5，横轴 "Tokens in Dataset" 取 $10^7$、$10^8$、$10^9$、$10^{10}$，图例 Params 依次为 700M、302M、85M、34M、25M、393.2K；深紫色大参数量曲线最平、早早饱和，浅色小模型曲线则继续下降。右下三维曲面标为 "(a) Wiki103 error (cross entropy) landscape."，轴包括 log2(data fraction)（约 −5 到 0）、log2(model fraction) 与 log10(err)（约 0.55 到 0.80）。

讲者指着最上方那条深紫色线说：给一个 teeny tiny 语言模型灌进巨量数据，这条线会平坦很久，是 "a complete waste of compute"；同样 compute 下，你本该去训练那个 big yellow model 到同样的 token 数，甚至更早收手更划算。反过来说，只要理解了 data amount 与 model size 如何共同决定性能，就能精确地把这批 flops 优化掉。为此需要「更 advanced 的、描述对象联合行为」的 scaling laws；转写在结尾 [58:26] 明确说，Kaplan 与 Rosenfeld "almost simultaneously" 几乎同时提出了这类联合形式。这里与全讲结构的关系很清楚：Ch6、Ch7 反复强调「先在低方差指标上建立 regularity」，正是这些联合律能被拟合并外推的前提；而本段只到「提出联合律」为止，后续（第 4 段起）才进入 compute 最优分配、拟合方法与具体指数的细节。易错点是：$C\approx N D$ 是讲者口中的 "roughly / linear in that" 的近似，不是精确等式；两条联合律函数形式不同（可加式 vs 幂套和式），不能混为一谈；$n$、$m$ 与常数 $C$ 的含义在本段未被逐一交代，不宜臆断。局限同样要写明：本段没有给出拟合方法、指数数值，也没有推出固定 flops 下的最优 $N$ 与 $D$，"surprisingly good fits" 只是幻灯片上的断言，讲者在 58:27 之前并未展开验证，因此这里应读作「引出问题与候选函数形式」，而非完整结论。

## 联合 scaling law 的函数形式、极限行为与外推 [58:27 - 60:20]

本段开头把前几讲分别讨论过的「数据 scaling law」与「模型 scaling law」合并成一个对象：**joint data-model scaling law**（联合数据–模型缩放定律）。它要回答的问题在幻灯片上写得很直白——`Q: Do we need more data or bigger models?`（我们需要更多数据，还是更大的模型？），旁边还有一句观察 `Clearly, lots of data is wasted on small models`（显然，大量数据被浪费在小模型上）。直觉是这样的：固定一份训练 compute（计算预算）之后，如果模型太小，继续灌入更多 token 所带来的损失下降会被模型容量卡住；反过来，模型过大而数据不足，多出来的参数也学不到东西。因此损失不能只由 $N$（参数量）或 $D$（数据量）中的任何一个单独决定，而必须由二者的联合函数 $L(N,D)$ 来描述；这也正是上一节末尾留下的「固定 compute 时如何分配资源」问题的形式化。

幻灯片给出了两个历史上最常被引用的、大致等价的函数族。Rosenfeld+ 2020 的形式最简单，讲者说它「really just the sum of two inverse terms」，即两个逆幂项相加，再加一个不可约误差常数：

$$
Error = n^{-\alpha} + m^{-\beta} + C
$$

Kaplan+ 2020 的形式稍微复杂一些，把两个逆幂项之和整体再取一次幂：

$$
Error = \left[m^{-\alpha} + n^{-1}\right]^{\beta}
$$

这里 $n$ 与 $m$ 分别对应数据规模与模型规模（讲者在后面特意提醒，这几篇论文的变量记号经常互换，很容易记反），$\alpha$、$\beta$ 是拟合得到的正指数，$C$ 是不可约误差（irreducible error）。幻灯片配了一句 `Provides surprisingly good fits to model-data joint error`，强调这类极简函数对「模型–数据联合误差」的拟合好得出乎意料。右上角的 `Loss vs Model and Dataset Size` 图给出了实证背景：横轴 Tokens in Dataset 从 $10^7$ 到 $10^{10}$，纵轴 Loss 约在 2.5 到 4.5 之间，六条虚线曲线对应的参数量分别是 768M、302M、85M、33M、25M 与 393.2K；右下角则是 Wiki103 上的三维误差曲面 `(a) Wiki103 error (cross entropy) landscape`，坐标轴标为 $\log_2(\text{data fraction})$、$\log_2(\text{model fraction})$ 与 $\log_{10}(\text{em})$，可以看到误差随两个方向同时下降并在一端趋于平坦。

![联合数据–模型 scaling law 的两个函数族：Rosenfeld 与 Kaplan 公式及 Wiki103 误差曲面](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0048_00-59-09.jpg)

讲者强调，拿到任何一个 scaling law 之后第一件该做的事，是把所有变量取极限（take the limits of all the variables），用渐近行为检验函数形式是否讲得通。对上面两个式子：若 $D\to\infty$，数据项 $D^{-\beta}\to 0$，剩下的只是模型项，联合律退化成纯粹的 model size scaling law，即系统「model size limited」（模型容量受限）；若 $N\to\infty$，模型项消失，剩下纯粹的 data scaling law，系统变成「bounded by data」（数据受限）。这与前面几讲分别推出的单变量幂律完全自洽，说明联合律不是凭空假设，而是把两个已知的极限行为用可加的方式缝合起来。这一步也是本段与上一节的衔接点：上一节只给出了 $C\sim N\cdot D$ 这一粗略关系，本节则给出了可以在该约束下真正最小化的目标函数。

有了拟合好的 $L(N,D)$，接下来就是一个带约束的非线性优化（讲者称其为 `a simple non-linear optimization problem`）：在固定 FLOPs 预算下求解

$$
\min_{N,D}\ L(N,D)\quad \text{s.t.}\quad C \approx \kappa\, N D,\qquad \kappa \approx 6
$$

解出的 $(N_{\text{opt}}, D_{\text{opt}})$ 就是 compute-optimal（计算最优）的参数量与数据量。更值得注意的是外推能力：这些联合律通常只在左下角的一小簇绿色点上拟合，那是一个 low compute、low data、小模型的区域，实验中从未出现过高数据或高模型规模的点；但把拟合出的曲面沿幂律外推到高 compute、高数据、大模型规模的区域，仍能相当准确地预测误差（`you can very accurately predict the error based on your predictions`）。这正是本讲反复强调的「用小规模实验预测大规模训练」在工程上的兑现，也是下一章讨论 Kaplan 处方的前提。

易错点与局限同样明确。第一，外推成立的前提是训练配方不变；讲者在本段后半部分会用整段时间说明，参数计数方式、warm-up、batch size 一旦改变，拟合出的指数就会漂移，所谓「准确外推」有可能只是拟合了一个本不该放大的 recipe。第二，$\kappa\approx 6$ 这类系数只是近似，把 $C$ 严格等同于 $ND$ 会忽略注意力、embedding 以及优化步数带来的额外开销。第三，两个逆幂项相加隐含「误差贡献可加」这一建模假设，它并非从理论推出，主要来自曲线拟合；在接近不可约误差 $C$ 的区域，幂律会弯折，继续直线外推会系统性偏乐观。

## Kaplan 的 compute-optimal 处方与 Chinchilla 的反驳 [60:20 - 62:20]

把上一节的约束优化真正解出来，就得到 Kaplan 与 Rosenfeld 那类联合律给出的 compute-optimal 处方。幻灯片把结论写得非常醒目：

$$
\text{Kaplan claims:}\quad N_{\text{opt}} = C^{0.73},\qquad D_{\text{opt}} = C^{0.27}
$$

括号里的注解是 `(tokens per param decreases w/ C)`，即每个参数对应的 token 数随 compute 增长而下降。顺着这个式子可以立刻算出两者的比值：

$$
\frac{D_{\text{opt}}}{N_{\text{opt}}} \propto C^{0.27-0.73} = C^{-0.46}
$$

也就是说，compute 每增加一个数量级，处方要求的 tokens per parameter 反而缩小，资源应不成比例地投向模型规模。讲者由此推出一句历史判断：如果你经历过 GPT-3 时代，就会记得有一段时间所有人都在训练「gigantic models」，几百亿参数、甚至万亿参数的 dense model，其部分动机正来自 GPT-3 与这些 Kaplan 预测——大家看到曲线后决定把大训练 run 押在超大模型上。

这里有一个必须点明的易错点：讲者自己在口述时把两个指数说反了，先说「amount of data 是 $C^{0.73}$、model size 是 $C^{0.27}$」，随即停下来自问 `is that reversed? I think it's reversed`，并纠正为 `N should be the amount of parameters and D should be the amount of data`。也就是说 $N$ 是参数量、$D$ 是数据量，0.73 属于 $N$、0.27 属于 $D$。幻灯片上的公式是核对这一点的可靠依据；把两者记反会直接把「该训练更大模型」误读成「该喂更多数据」，结论完全颠倒。

![Kaplan 的 N_opt=C^0.73、D_opt=C^0.27 处方与 Chinchilla 三条 Approach 曲线、四颗大模型星号](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0049_01-00-32.jpg)

配图把这个分歧画得很清楚。横轴是 FLOPs，从 $10^{17}$ 一直到 $10^{25}$；纵轴是 Parameters，从 10M、100M、1B、10B、100B 到 1T。图中除了一条标注为 `Kaplan et al (2020)` 的黑色陡峭虚线外，还有 Chinchilla 的三条彩色拟合线 `Approach 1`（青色）、`Approach 2`（橙棕色）、`Approach 3`（绿色），它们的斜率明显更平缓，并且彼此靠得很近。图上四颗星号标出当时的代表性大模型：Chinchilla (70B) 是青绿色星、Gopher (280B) 是橙色星、GPT-3 (175B) 是红色星、Megatron-Turing NLG (530B) 是紫色星；右下角注明出处 `Hoffmann+ 2022`。讲者指着右上角那几颗星说，2022 年 DeepMind 的 Hoffmann 等人进来指出这些预测「all terribly off」——大家一直在训练的那三颗大星（three stars）`way too big`（实在太大了），真正该走的是蓝色那条更平缓的趋势，去训练那颗 teal star 所代表的、小得多但数据喂得更多的模型。

这就是 Chinchilla 论文的核心反驳：应该训练相对小得多的模型，并把省下的 compute 换成更多训练 token。讲者提到在场的人大概都知道 Chinchilla 的 token multiplier——`20 tokens to every parameter`，也就是经验比例 $D \approx 20N$。但他强调要专门花时间细讲，因为 Chinchilla 与 Kaplan 的分歧不只是「谁的数字对」，更是在告诉我们应当如何思考 scaling laws：`It's not just something where you like turn the crank and you get the answer. It's something that's actually careful and sensitive and we have to respect the process.`（这不是摇一下手柄就出答案的东西，它细致、敏感，必须尊重过程。）这句话为后面三章的方法论讨论定下基调。

与前文的衔接在于：Kaplan 与 Chinchilla 用的都是上一节那类联合 scaling law，都做固定 FLOPs 下的最小化，形式上没有本质区别；差别出现在拟合过程与实验设置。局限方面也要留意，$N_{\text{opt}}=C^{0.73}$、$D_{\text{opt}}=C^{0.27}$ 与 $D\approx20N$ 都是特定数据分布、架构与训练配方下的经验拟合，不是理论常数；把它们当作跨配方、跨数量级普适的黄金比例，正是本讲后面要反复警告的误读。

## Chinchilla 三种拟合方法之一：训练曲线下包络 [62:20 - 64:30]

讲者说他很喜欢 Chinchilla 的一点是：作者提出了**三种不同的 scaling law 拟合方法**。这是一种 robustify（增强稳健性）的手段——用彼此独立的估计路径去逼近同一个量，从而检验结果是不是被某一种建模假设绑架了。三种方法除了 method three 略有出入外，结论大体一致：方法 1 与方法 2 给出的数据指数与模型指数分别是 $0.5/0.5$ 和 $0.49/0.51$，方法 3 给出 $0.46/0.54$；而 Kaplan 那边两个指数相差极大，正因如此他们才会在大 compute 下训练超大模型。讲者把 $0.46$ 与 $0.54$ 这个偏差暂时按下不表，留到后面 Epoch AI 的一段再解释。

方法 1 被讲者称为 **lower envelope method**（下包络法），幻灯片标题是 `Method 1 – minimum over runs.`，并注明它与 Kaplan 的 FLOPS 图思路相近——`the minimum over the union of all training curves is a power law`（所有训练曲线并集上的最小值构成一条幂律）。做法是：把每一次训练 run 画成一条 loss 随 FLOPs 变化的曲线，然后在每一个 FLOP 值上取所有曲线达到过的最低损失

$$
L^*(C) = \min_{\text{runs}}\ L_{\text{run}}(C)
$$

下包络上的点意味着「在这个 flop 预算下、在任意一次训练 run 中我取得过的最好 loss」，因此它天然代表了该 compute 下的最优配置。接着把这些下包络点对应的模型大小取出来，对 FLOPs 作散点图，会得到一条漂亮的线性趋势（在 log-log 空间即幂律），由此估计 flops 到 parameters 的 compute-optimal 权衡。

![方法一：训练曲线下包络、67B 最优参数量与 1.5T token 的 Gopher 投影](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0051_01-03-19.jpg)

配图的 Figure 2 图注补充了大量画面独有而讲者没有逐字念出的细节。左图是全部训练曲线，纵轴 Training loss 从 2.0 到 6.0，横轴 FLOPS 从 $10^{17}$ 到 $10^{22}$，旁边的颜色条列出模型规模 10B、5B、2.5B、1B、500M、250M、75M；图注写明作者 `launched a range of model sizes going from 70M to 10B, each for four different cosine cycle lengths`（模型规模从 70M 到 10B，每种跑四个不同的 cosine cycle 长度），再从这些曲线中 `extracted the envelope of minimal loss per FLOP`。中图是 Parameters 对 FLOPs 的下包络散点，右图是 Tokens 对 FLOPs 的散点，两者的绿色投影线分别标着 **67B** 与 **1.5T**（$1.5\times10^{12}$ token）。图注说明绿色标注是 `projections of optimal model size and training token count based on the number of FLOPs used to train Gopher ($5.76 \times 10^{23}$)`，即按 Gopher 的 $5.76\times10^{23}$ FLOPs 预算外推出的最优模型大小与训练 token 数。讲者口述给出的答案与图一致：`for their compute budget, they should train a 67 billion parameter model`。

这个方法与前文的联系很直接：它不需要先假设 $L(N,D)$ 的具体函数形式，只是从已有实验里「挑出最好的点」，因此比下一段的联合曲面拟合少一层建模假设；而它给出的 67B 也成为后面三个方法互相印证的基准值之一。讲者的评价是 `a nice and simple way, but very tricky`：概念简单，但实操里有很多坑。

易错点主要落在「什么是下包络」这一定义上。首先，下包络是经验最小值，取决于你到底跑过哪些配置——若某个 compute 区间里没有覆盖到真正最优的模型规模或 cosine cycle 长度，包络点就会偏高，拟合出的幂律斜率随之偏移。其次，训练曲线末端噪声、单次 run 的随机性、以及是否对每条曲线做平滑，都会改变取最小值的位置；讲者说 `there's definitely some tricky parts` 正是指此。再次，把下包络点对应的模型大小当作「最优 $N$」时，隐含假设是这些点来自充分收敛、超参合理的训练 run，一旦某些 run 本身没训好，包络就会被污染——这个隐患在讲者后面讨论 warm-up 与 batch size 时会变成解释 Kaplan 偏差的关键线索。

## 方法二 Isoflops 与方法三联合曲面拟合 [64:30 - 67:10]

方法 2 是 **Isoflops**（等算力曲线），讲者称它「now very popular」「very easy and very robust」，并且是 `probably my personal favorite method`。做法是：先挑一批不同的 FLOPs 预算，讲者说这一批预算按 `factors of three` 递增（配图 Figure 3 的图例里能看到 $6\text{e}18$、$1\text{e}19$、$3\text{e}19$、$6\text{e}19$、$1\text{e}20$、$3\text{e}20$、$6\text{e}20$、$1\text{e}21$、$3\text{e}21$ 这一串）；然后对每一个固定预算，在「参数量 vs 数据量」这个自由度上做扫描——讲者的说法是 `I double my data size, I half my model size, so on and so forth`，也就是在 $C\approx \kappa N D$ 约束下沿

$$
D = \frac{C}{\kappa N}
$$

这条双曲线换点。每个预算因此生成一串 terminal loss 不同的训练 run，画出来是一条 U 形（convex）曲线；幻灯片标题写着 `Pick a range of FLOP budgets, vary the total parameter count, take the min over these convex shapes. The minima form a power law.`。Figure 3 图注进一步说明 `For various model sizes, we choose the number of training tokens such that the final FLOPs is a constant. The cosine cycle length is set to match the target FLOP count.`，并指出 `We find a clear valley in loss`，谷底位置就是给定预算下的最优模型；用这些谷底向更大模型投影，绿色标注给出 Gopher 预算下的最优参数量与 token 数。

讲者补充了两种取谷底的技巧：可以直接取每个预算下那串点的最小值，也可以对每串点拟合一条二次曲线再取二次曲线的最小值，即

$$
\hat L(\log N) = a\left(\log N - \log N^*\right)^2 + L_{\min}
$$

再把所有二次曲线的谷底连成线，或把谷底作为 FLOPs 的函数作图。这样得到的第二条预测与方法 1 高度吻合：讲者口述给出 `This is 63 billion parameters for my compute budget`。67B 与 63B 这两个数字的接近，正是三种方法「mostly agree」的具体体现，也是 Chinchilla 敢下结论的依据。

方法 3 是 **joint fits**（联合拟合），幻灯片标题 `Method 3 – Joint fits`，副标题 `Run a bunch of models on the size-data grid. Use least squares to fit a joint scaling law`。这是最「自然」也最 brute force（蛮力）的一条路：既然 Rosenfeld 与 Kaplan 都假设了 $L(N,D)$ 的函数形式，那就照样假设一个参数化曲面 $\hat L(N,D)$，在 size–data 网格上跑一大批模型，用最小二乘直接拟合出系数。沿用本讲前面已引入的两项相加联合律，可写成

$$
\hat L(N,D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
$$

![方法三：联合拟合 L(N,D) 的等值线与 isoFLOP 切片，Gopher 预算下最优为 40B 参数](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0053_01-06-06.jpg)

Figure 4 图注给出了画面上独有的细节：作者对 $\hat L(N,D)$ 做参数化建模，左图画 iso-loss 等高线（contour），右图画 isoFLOP slices；每条 isoFLOP 切片在左图中对应一条虚线。左图里的蓝色 **efficient frontier** 是 log-log 空间中的一条直线，`the curve goes through each iso-loss contour at the point with the fewest FLOPs`（它穿过每条等损失线上 FLOPs 最少的那个点）。右图的纵轴 Loss 从 2.00 到 5.00，横轴 Model size 标着 100M、1B、10B、40B，右侧图例列出 Train. FLOPs：$6\text{e+}18$、$1\text{e+}19$、$3\text{e+}19$、$6\text{e+}19$、$1\text{e+}20$、$3\text{e+}20$、$6\text{e+}20$、$1\text{e+}21$、$3\text{e+}21$ 以及 Gopher。图注最后一句是 `We project the optimal model size given the Gopher FLOP budget to be 40B parameters`——方法 3 给出的最优模型是 **40B** 参数，明显小于方法 1 的 67B 与方法 2 的 63B，这个缺口后来正是 Epoch AI 那一章的引子。

讲者对三种方法的整体评价把本段与下一章连了起来：走完 Chinchilla 的做法后 `it doesn't seem materially different than what Kaplan was doing`，并不是 Kaplan 做了一件疯狂的事、Chinchilla 作者突然想出什么高明得多的招；两边 `both did pretty reasonable stuff`。可一旦去看 scaling 预测，结果又差得非常远，而且事后大家普遍认为 Chinchilla 更接近正确。于是问题变成：两边都在拟合同类联合 scaling law，为什么结论差这么多？

局限方面，讲者特别指出方法 3 与方法 1 一样「有 trickiness」：拟合的是多维曲面，变量多、参数多，`how you do the curve fitting is very important`，初值、正则化、损失权重、外点处理都会左右系数，不像 isoflops 那样直观可靠。相对地，isoflops 每个预算只需在一个自由度上扫描、谷底由数据直接显示，因而成为讲者最信任的方法——这一偏好在最后一章还会再次出现。此外，三种方法得到的 67B / 63B / 40B 都只是针对 Gopher 这一个 compute 预算的投影，不能脱离预算谈「最优模型大小」。

## 分歧溯源：参数计数、warm-up 与批大小 [67:10 - 71:00]

讲者把这一章称作进入 `the messy realities of how scaling laws are made`（scaling law 是怎么被造出来的那些 messy 现实）：scaling law 会因为实现方式、超参数设置、甚至 x 轴取什么而真的改变。他引用的论文是 *Resolving Discrepancies in Compute-Optimal Scaling of Language Models*（幻灯片右上角框出的标题，作者以 Wortsman 为首、Yair Carmon 为末位；讲者说最后这位作者是他以前的学生，还表示之后会把链接贴到课程 Slack）。这篇工作的路线是：先用大致 Kaplan 的设置复现 Kaplan 结果，然后逐步改动——`let's change how we count parameters`（曲线轻微移动）、`let's change how learning rate warm-up is done`（再移动一点）、再 `tune the optimizer a little bit more`，最后就得到与 Chinchilla 完全一致的结果。讲者的总结是：`there's a sequence of what seems like very minor decisions that in the end gives you this big gap`（一串看起来极微小的决定，最终累积成 Kaplan 与 Chinchilla 之间的巨大差距）。

![Explanation 1：复现 Kaplan 并逐步修正参数计数、warmup、优化器的五个子图与幂律指数](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0055_01-09-00.jpg)

配图把这条「逐步修正」的链条量化到了可以逐格读数的程度。五个子图各自拟合出 $N^*(C)\propto C^{a}$ 的指数 $a$ 及 95% 置信区间，并在括号里给出某一 compute 点上的最优参数量估计：子图 a `Reproducing Kaplan et al.` 为 $a=0.835\ (0.82,0.85)$、$N^*(C_C)=3\text{T}\ (3\text{T},4\text{T})$；子图 b `Counting last layer FLOPs` 为 $a=0.706\ (0.69,0.72)$、$N^*=787\text{B}\ (639\text{B},916\text{B})$；子图 c `Correcting warmup` 为 $a=0.602\ (0.59,0.62)$、$N^*=292\text{B}\ (249\text{B},353\text{B})$；子图 d `Cosine decay (no tuning)` 为 $a=0.571\ (0.56,0.59)$、$N^*=183\text{B}\ (152\text{B},240\text{B})$；子图 e `Optimizer tuning (no decay)` 为 $a=0.497\ (0.49,0.50)$、$N^*=77\text{B}\ (70\text{B},86\text{B})$。图例列出 `95% confidence region`（黄色带）、`Power law fit`（红色虚线）、`Hoffmann law`（黑色点划线）、`Kaplan law`（点线）与 `Observations`（带误差棒的观测点），横轴都是 `Compute C [FLOPs]`。指数从 0.835 一路降到接近 Chinchilla 的 0.50，是「细节决定结论」最直白的证据。幻灯片底部三条 bullet 原文为：`Kaplan removed last layer param from the count`、`Warmup at very small compute budgets was too high`、`(Decay itself is maybe not critical if batch / LR is properly tuned)`。

三个原因逐一展开。第一是**参数计数**。Kaplan 发现把 embedding 参数计入时曲线很怪，于是排除了 embedding 参数；这个做法本身还说得过去，但真正把他们带偏的是同时也排除了最后一层（softmax 层）的参数。理由是这两个矩阵互为对偶、形状相同：embedding 是 $\mathbb R^{V\times d}$，输出层是 $\mathbb R^{d\times V}$，其中 $V$ 是 vocab size、$d$ 是 hidden dim，形状一致就把两者一并剔除了。讲者强调 `whether you include or exclude those has a big material impact on the shape of the scaling law`——计入与否会实质改变 scaling law 的形状，这比观点分歧更像一个 oversight（疏漏）。

第二是 **warm-up 与学习率**。这篇复现工作指出，Kaplan 训练的许多模型非常小，小到学习率 warm-up 结束时模型其实还没有收敛，于是学习率设置得很次优（`their learning rates were set very suboptimally`、`their models weren't fully converged`）。第三是 **batch size**：Kaplan 固定使用一个很大的 batch size，而大 batch 对小模型同样次优；改成随规模调整、使用 varying batch sizes 之后，结果就与 Chinchilla 精确吻合。幻灯片第三条 bullet 还特意补了一句限定：只要 batch / LR 调得对，decay 本身（cosine decay）也许并不关键——这提醒读者不要把「学习率调度」笼统当成一个整体变量。

由此讲者给出一个贯穿本讲后半段的重要判断：`scaling laws are kind of lower bounds in some sense`。它所表达的是「如果我继续沿用这套 recipe 并把它放大，那么我会得到这样的结果」，即对一个给定配方的条件预测，而不是对任务本身的理论下界。若放大的是一套你本来就不想放大的 recipe——warm-up 很荒唐、batch size 很荒唐——那必然得到糟糕的 scaling law；所以实验要 `be as close to the proper sort of full run as possible`（尽可能接近真正完整的大 run）。这与前面「外推的前提是配方不变」完全呼应，也解释了为什么 Kaplan 在低 compute 区拟合出的指数会偏得那么厉害。

易错点与局限可以归纳成三层：口径上，参数计数必须全程统一，总参数 $N$ 与非 embedding 参数 $N_{\text{non-emb}}$ 混用会直接改变 x 轴；区间上，低 compute 区域的点在拟合里权重过大，而那里恰是 warm-up、batch size 最容易出问题的地方；解释上，把 scaling law 当成无条件真理、而不是「配方 + 区间」条件下的外推，就会把一个工程 bug 误读成自然规律。

## Pearce–Song 的纯模拟解释与 Epoch AI 的重拟合 [71:00 - 74:20]

第二篇用来研究分歧的论文走了完全不同的路子。讲者口述称之为 Pearson and Song，幻灯片上写的是 *Reconciling Kaplan and Chinchilla Scaling Laws*，作者 Tim Pearce（Microsoft Research）与 Jinyeop Song（MIT），并标注 `Reviewed on OpenReview`。讲者觉得它 `very clever` 的地方在于：`they don't train any models`（他们完全不训练模型），只取 Chinchilla 拟合出的训练曲线／隐含函数形式把它画出来，然后据此生成「若按 Kaplan 的设定（忽略 embedding 参数、把规模压得很小）会得到怎样的隐含训练曲线」，再比较两种口径下的结论。

![Explanation 2：从 Chinchilla 拟合曲面出发模拟 Kaplan 设定，得到 0.51 与 0.78 两个指数](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0057_01-12-15.jpg)

配图左侧圆角框给出了整套模拟的起点，即直接从 Chinchilla 的拟合曲面出发：

$$
\mathrm{Loss}(N_T, D) = \frac{482}{N_T^{0.35}} + \frac{2085}{D^{0.37}} - 1.82
$$

随后 `Generate training curves for model sizes used in Kaplan's study (1k to 1.5B params)`——这里藏着一个关键数字：Kaplan 研究里用到的模型规模最小只有 1k 参数，最大 1.5B 参数，整体处在极低的 compute scale。中间两组框构成对照实验：上排 `Find compute optimal frontier in terms of total parameters $N_T$ and compute as in Chinchilla`，得到 `Find power law scaling coefficient of 0.51, close to Chinchilla's 0.50`，公式为 $N_T^* \propto C_T^{0.51}$；下排 `Find compute optimal frontier in non-embedding parameters $N_{nE}$ and compute as in Kaplan`，得到 `Find local power law scaling coefficient of 0.78, close to Kaplan's 0.73`，公式为 $N_{nE}^* \propto C_T^{0.78}$。幻灯片底部用大字点出成因：`Non-embedding vs total param choice + small nonlinearities`。

于是这篇工作给出的是两个效应，而不是前面那篇的「一串实现细节」：其一，Kaplan 运作在 `much lower compute scale`，因此对微小变化极其敏感；其二，忽略 non-embedding 参数所引入的轻微非线性，本身已足以把结果从 0.51 推到 0.78。讲者说这与 Wortsman 等人的论证 `slightly different`，但共同指向同一件事——`the size of the compute scale plus the slight nonlinearity induced by non-embedding parameters` 造成了分歧。他自嘲 `this is really starting to get into the weeds`，但强调目的就是让人体会 `the importance of paying attention to detail`；而建立可靠 scaling law 的大量工作，本质上是在 `figuring out what the right regime is`，好让规律能跨 compute scale 稳定成立。

本段后半是讲者称为 `fun addendum` 的一段，也是他对 Chinchilla  saga 收尾的地方。悬念在于：Chinchilla 这篇论文某种意义上「take down 了 Kaplan」，却留下一个未解之谜——`Chinchilla method three never agreed with Chinchilla method one and two`。数字看着不大（方法 1、2 是 $0.5/0.5$、$0.49/0.51$，方法 3 是 $0.46/0.54$），作者 `very unperturbed`（毫不在意），但讲者认为含义并不小：方法 1、2 意味着参数量与 token 数应当同步增长、二者保持固定比例，这正是 Chinchilla 那个 `factor of 20` 的来源；而方法 3 的两个指数不等，意味着随着 compute 放大，渐近上 token 会远多于参数，得出的是 `a very different scaling conclusion asymptotically`。

Epoch AI 的人同样被这件事困扰，但他们拿不到 raw data 也拿不到 code，于是做了 `data forensics`：从论文图里把数据点抠出来，重新拟合方法 3 提出的那张三维曲面。结论是原论文 `must have underfitted the original data`（欠拟合了原始数据）；重做曲线拟合后能取得明显更低的 loss，而且顺着重拟合后的模型，几乎精确回到 $D/N = 20$ 的 `rule of thumb`。对应配图左半是两组残差分布（Hoffmann et al. 与 Ours），纵轴 Residuals 约从 $-0.10$ 到 $0.15$，重拟合一组的残差明显更集中在 0 附近；右半纵轴为 `Optimal tokens per parameters ratio`（刻度 10、100），横轴 `Training compute (FLOP)` 从 $10^{19}$ 到 $10^{27}$，图例为 `Optimal policy (ours)`（深蓝实线）、`Optimal policy (Hoffmann et al.)`（绿色虚线带）与 `Chinchilla model`（黑点），图中虚线标注 `$D/N = 20$ rule of thumb`，出处 `[Besiroglu et al 2024]`。讲者觉得这个结局很幽默：`the authors were more right than they knew`（作者比自己以为的还要正确），方法三之所以不一致，只是因为犯了一个拟合错误，方法本身终究是 robust 的。

易错点与局限：从位图里抠数据本身就引入读数误差，重拟合结论依赖所采用的正则化与优化设置，因此不宜把「原论文 underfit」理解成对 Chinchilla 全部结论的否定；另外不同来源给出的曲面系数（如这里的 $482$、$2085$、$1.82$ 与指数 $0.35$、$0.37$）是特定拟合产物，跨论文直接套用会出错。更一般的教训是：三种方法给出不同指数时，先怀疑拟合过程，再怀疑物理规律。

## 生产模型为何不要 train-optimal、isoflops 与全讲收束 [74:20 - 77:56]

讲者用一句反直觉的话开启收尾：`you probably don't want the Chinchilla factor`，并说这在当下已接近常识。理由是成本结构变了：训练生产模型时 `You don't really care about saving training compute for the most part`，参考各类对前沿实验室 compute 去向的外部调查可知，大部分 compute 并不花在训练 run 上，而是花在 R&D 与 **serving**（推理服务）上。既然推理成本占主导，那么想要的就是 `small models that are capable`——小而能干的模型，而不是 `big, bloated models that cost a lot to serve`，即便后者在「固定训练 compute」意义下是最优的。于是真正该做的是 `overtraining`（过训练）；讲者特意给这个词加引号，因为 `overtraining is what we want. That's the right amount of training`（所谓过训练，其实正是合适的训练量）。换句话说，Chinchilla 优化的是 training-optimal，而生产环境要优化的是包含推理在内的总成本，两者的目标函数不同。把这层意思写成记号会更清楚：Chinchilla 只在最小化训练开销，而部署时要最小化的是

$$
\text{Cost}_{\text{total}}(N,D,U) = C_{\text{train}}(N,D) + U \cdot c_{\text{serve}}(N)
$$

其中 $U$ 是预期服务量（usage），$c_{\text{serve}}(N)$ 是单次推理成本，并随参数量 $N$ 增大。当 $U$ 很大时第二项压过第一项，最优解自然把 $N$ 压小、把 $D$ 放大，于是 $D/N$ 这个比值被推高——这正是幻灯片里从 20 一路涨到 215 的方向，也是 `The more usage we expect, the more it becomes worth it to pay the upfront cost` 这句话的含义：多付的那笔 upfront cost 是训练阶段多喂的 token，换回来的是长期被摊薄的推理开销。

![train-optimal 并非生产所需：GPT-3 到 Llama 3 各模型的 tokens per param 列表](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0059_01-15-30.jpg)

配图把这条演变写成了可核对的数字清单。标题是 `Important note – train-optimal is likely not what you want`，副标题 `Chinchilla aims to tell you what gives the best model for fixed training compute.. But most of the compute in a real deployment is inference.. So we should 'over' train`。六条 bullet 依次为：`GPT3 – 2 tokens / param`、`Chinchilla – 20 tokens / param`、`LLaMA65B – 22 tokens / param`、`Llama 2 70B – 29 tokens / param`、`Mistral 7B – 110 tokens / param`、`Llama 3 70B – 215 tokens / param`，后面还有 `... etc etc`；底部一句粗体结论 `The more usage we expect, the more it becomes worth it to pay the upfront cost`（预期使用量越大， upfront 的一次性训练开销就越值得付）。讲者口述时把 GPT-3 约略说成 `Three tokens per parameter`、把 Chinchilla 说成「现代比例 20」，幻灯片给出的 2 与 20 是更精确的口径；从 2 到 215 的单调上升，正是「从 undertrained 走向 overtrained」的量化轨迹。他补了一段时代划分：早期 GPT-3 明显 undertrained；Chinchilla 把比例推到 20 之后，有一阵大家停留在 20，那时模型还没有被大规模 serving；等到 serving 变得非常真实，就开始 overtraining，并转向 **MoEs**（Mixture of Experts），用各种权衡专门优化推理服务成本。

因此讲者对 Chinchilla 的定位是：它重要，不是因为 $20{:}1$ 是 `the one golden ratio`（做研究时或许可以这么用），而是因为 `it tells us quite a bit about how we should be fitting scaling laws and how to think about scaling laws`——价值在方法论，与前面三章关于参数计数、warm-up、拟合过程的讨论一脉相承。紧接着他强调，就研究工具而言 **isoflops** 是 `have endured`（经受住时间考验）的那一个：固定一个 flop 预算，扫过其余所有自由度，看损失曲面长什么样，通常能给出 `fairly reliable readings`；他提到自己的一个学生把这套设计用在 diffusion models 上，前面讲过的 MoE study 也是 isoflop style design，并给出一句可直接带走的经验——`If you're ever in a situation where you're thinking, "How am I going to sort of decide all these tradeoffs?" isoflops is always a good default`（当你要决定一堆权衡时，isoflops 永远是好的默认选项）。这与第四章里他称 isoflops 为 personal favorite 形成首尾呼应。

![全讲 Recap：Data scaling、Model scaling 与 Scaling as prediction 三条总结](/coursenotes/cs336-language-modeling-sp26/l09-scaling-laws/kf_0060_01-17-31.jpg)

最后是对整讲的收束。讲者说今天的核心，是资源投入与性能之间存在 `log-linear regularity`（对数–线性正则性），它延伸到很多对象上：model parameters、compute、MoE 的 sparsity level 等等；有了它，前面提到的那些 `arbitrary choices`（凭感觉拍板的工程选择）就能改以 `evidence-driven way`（证据驱动）来做，从而实现 `engineering at scale without explicitly relying on doing big model training runs`（不必真的跑大模型，也能做大规模工程）。收尾幻灯片的标题是 `Recap: scaling laws – surprising and useful!`，圆角框里三条分别是：`Data scaling: understand how data affects models, clean theory`、`Model scaling: dramatically reduce costs for training`、`Scaling as prediction: understand what problems can be 'brute forced'`。讲者口述的对应说法是：data scaling 是最自然的对象、只是指数有点奇怪；model scaling 让各种工程手段成为可能；而 scaling 本身是一个稳健的 predictor，用来指导大规模工程。他最后致谢，并说下一讲由 Percy 讲 inference，之后再回到更 advanced、更 niche 的 scaling 话题。

这一章的易错点与局限有三条值得带走。第一，把 $D/N=20$ 当成普适黄金比例是最常见的误读：它只在「固定训练 compute、且不计推理成本」的目标下成立，一旦把 serving 纳入目标函数，最优比例就会像幻灯片里那样从 20 一路涨到 215。第二，overtraining 是否划算强依赖预期 usage——`The more usage we expect, the more it becomes worth it`，低频使用的内部模型未必值得付那笔 upfront cost，不能无条件照抄前沿实验室的 tokens/param。第三，isoflops 只扫描了被你选中的自由度；数据质量、架构、优化器、参数计数口径这些没被扫描的维度仍可能成为隐藏瓶颈，而前面几章已经证明，恰恰是这些「没扫到」的细节能把 scaling law 的指数从 0.5 推到 0.73 甚至 0.835。工具可靠，不等于可以不对配方保持警惕。

## 要点回顾

### 定位、历史与理论根源

- 本讲以"朋友给你一万块 B200、用一个月造开源语言模型"为引子，把决策钉在 `Run a big model (but which one??) <- we are here`：在 A2 分布式框架与 A4 预训练数据之外回答到底训练哪个大模型；一次大 run 可达数百万美元，故讲者主张 `tune on small models, extrapolate to large ones` 而非 `tune hyperparameters on big models`。
- 这些函数形式并不新：有限假设集泛化界 $\epsilon(\hat h)\le(\min_{h\in H}\epsilon(h))+2\sqrt{\frac{1}{m}\log\frac{2k}{\delta}}$ 与光滑密度速率 $\psi_n=n^{-\frac{\beta}{2\beta+1}}$ 都刻画误差随样本量衰减，但幻灯片特意标注 `But these are upper bounds, not actual, realized loss values`，提醒理论上界不等于实测损失。
- 1993 年 Bell Labs 的 Learning Curves 用 $\epsilon_{\text{test}}=a+\frac{b}{l^{\alpha}}$、$\epsilon_{\text{train}}=a-\frac{c}{l^{\beta}}$ 令两条误差渐近趋于共同的 $a$；2012 年前后的 Table 1 在 Exp3、Exp4、ExpP3、Pow3、Pow4、ILog2 六族里比出幂律族对 BLEU 曲线最有预测力，讲者还反复提到 power three and power four。
- Hestness+ 2017 跨语音识别、机器翻译与语言模型给出多项式趋势，并已写到 Emergence 的 `accuracy cliffs`、Scaling by compute 与 Speed = accuracy；讲者把幂律定性为 `pure curve fitting exercises`，理论只提供候选函数形式，`there's no golden rule that these are the only ones`。

### 数据缩放律与指数之谜

- 定义是 `Data scaling laws : simple formula that maps dataset size (n) to error`，曲线分 Small Data Region、Power-law Region、Irreducible Error Region 三段，只有中间段满足 $\log L=-\alpha\log D+c$（即 $L(D)\propto D^{-\alpha}$），逼近不可约误差后就会弯折而非继续直线下降。
- Kaplan+ 2020 把多因素定量化：$L=\left(C_{\min}/2.3\cdot10^{8}\right)^{-0.050}$、$L=\left(D/5.4\cdot10^{13}\right)^{-0.095}$、$L=\left(N/8.8\cdot10^{13}\right)^{-0.076}$，三个指数绝对值都很小，且横轴特意用 non-embedding 口径，换一种参数计数就会改写整条曲线。
- 均值估计玩具例子给出 $\mathbb E[(\hat\mu-\mu)^2]=\frac{\sigma^2}{n}$ 与 $\log(\mathit{Error})=-\log n+2\log\sigma$，理论预期 $y=-x+C$ 斜率 $-1$；实测却约 $-0.1$ 到 $-0.3$，如机器翻译的 $\ell(m)=2.87\,m^{-0.13}$，比估计均值或线性回归慢得多。
- 非参数解释把二维空间切成边长 $n^{-1/4}$ 的格子得 $\sqrt n$ 个盒子、误差约 $\frac{1}{\sqrt n}$，推广到 $d$ 维为 $\mathit{Error}=n^{-1/d}$、$y=-\frac{1}{d}x+C$；Bahri 2021 的 Intrinsic dimensionality theory 主张 $\alpha$ 与内在维度相连，但 `estimators of intrinsic dimension are sketchy`，只宜当作直觉。

### 数据组成、重复与过滤

- 数据组成只动截距不动斜率，可写 $L(D;q)\approx C(q)\,D^{-\alpha}$、$\log L = -\alpha\log D + \log C(q)$，对应 [Hashimoto 2021] 的 `Data composition affects the offset, not the slope`；U 形 $\log C(q)$ 最低点约在 $q\approx 0.4\sim0.5$，于是 `if the slopes don't change, the best mixture at small scale is also the best mixture at large scale`。
- 实践却 `In practice: data mixture selection via scaling is hard`：DataDecide 的 `18 - 80% correct` 与 `Empirical eval – just take the best small dataset`、`No scaling law required` 说明现实嘈杂；Data Mixing Laws 则拆成 (1) Training Step Law、(2) Model Size Law、(3) Data Mixing Law 三步，外推到 large steps、large models、unseen mixtures。
- 数据重复在标准训练配方下约 4 个 epoch 以内不受伤，之后明显差于新鲜数据；有效数据量写成 $D' = U_D + U_D R_D^*\left(1 - e^{-R_D/R_D^*}\right)$，$U_D$ 为唯一 token、$R_D$ 为重复次数、$R_D^*$ 为待拟合常数，最终趋于 $U_D(1+R_D^*)$ 上限。
- 无限算力设定（Perseus）提醒 `Scaling laws can 'break' if you blindly apply it` 与 `Scaling laws are lower bounds so you can always potentially do better`；过滤须随规模自适应，Small / Medium / Large Compute 分别对应 E、E+D、E+D+C 数据池，即 `Data selection should then be adaptive to scale!`，而窄区间下 $D^{-\alpha}$ 与 $e^{-\lambda D}$ 经 Taylor 展开都近似直线，难以辨识。

### 模型工程：架构、优化器与参数计数

- 架构比较在 log-log 下拟合 $L(C)=A\,C^{-\alpha}+L_\infty$ 并对比斜率与截距：Transformer 相对 LSTM 是 `definitely different intercept, maybe even different slope`，LSTM 在 $10^7$ 参数后明显抬离；T5 跨架构研究里 GLU `Red line better than green line throughout the scaling trends`，Performer 则 `doesn't scale very well`。
- 连 SGD 换 Adam 这样大的干预也只平移曲线：$\epsilon(m) = 5.37\,m^{-0.094}$ 对 $\epsilon(m) = 5.25\,m^{-0.095}$，指数只差 0.001，且限定在 2017 年 pre-transformer 的 RHN 上，正应了 `the intercepts are different, but the slopes are very similar`。
- 模型形状上 $d_{\text{model}}/n_{\text{layer}}$ 近似尺度不变，最优 `around 100 $d_{\text{model}}$ for every layer`，`22% additional compute compensates for 1% loss increase`，aspect ratio 可差 `a factor of 40` 而只损失 `within 3%`（如 $(6, 4288)$ 对 $(48, 1600)$）。
- 参数并非等价：`not all parameters are created equal`，Kaplan 排除 embedding 参数后曲线才近乎平行，讲者据此断言可预测性是 `engineered`，要 `pick the right kinds of x-axes` 且 `make sure that the hyperparameters for these things are set right`；MoE 又把 $N$ 解耦成总参数、激活参数 $N_a^*$ 与 sparsity $S$。

### 批大小、学习率与实验方法论

- 大 run 前真正要重调的只有 batch size 与 learning rate；critical batch size 从 noise limited 的 `Perfect scaling`（$S(B)\cdot B\approx\text{const}$）经 crossover 到 bias limited 的 `Ineffective scaling`，由 $E = B\cdot S$ 与 $\frac{S}{S_{min}} - 1 = \left(\frac{E}{E_{min}} - 1\right)^{-1}$ 拟合出 $\mathcal{B}_{crit} = \frac{E_{min}}{S_{min}}$。
- 它随目标损失下降而幂律变大：$B_{crit}=2.1\times10^{8}\ \text{tokens}\cdot L^{-4.8}$（$N=3M$ 与 $N=85M$ 都吻合），并有 $C_{min}(C) \equiv \frac{C}{1 + B/B_{crit}(L)}$ 仅在 $B \ll B_{crit}$ 时成立。
- 学习率侧有 `learning rates generally shift` 与经验规则 $\eta \propto 1/\text{width}$；μP 的 Table 2 规定 matrix-like 层 $l \to l/r$、$\sigma \to \sigma/r$，而 embedding 属 `others` 不缩放、`lm_head` 的 $\tau \to \tau/r$，目的是让 `optimum stable` 而非 Standard Practice 的 `optimum shifts`。
- 方法论上要先在低方差指标建立 regularity 再谈迁移：perplexity 的点多为 `singletons`、`the variance is in the second decimal place`，one-pass SGD 下 $|L_{\text{train}}-L_{\text{val}}|\approx 0$ 故常混用 train/test loss；但 LR 与 $\mathcal{B}_{crit}$ 的 scaling 噪声 `truly horrendous stuff`。
- 上下游会错配：T5 反例里困惑度更好的 NL-12 在 SuperGlue 仅约 77.7，困惑度更差的 NL32-XL 反而约 79.7 最高，故 `Caution – scaling behaviors can differ downstream`；`Some surprising takeaways` 给出三步流程 `Train a few smaller models` → `Establish a scaling law` → `Select optimal hyperparam based on the scaling law prediction`。

### 计算最优分配：从 Kaplan 到 Chinchilla 与生产取舍

- 固定算力下 $C \approx N D$（更细作 $\min_{N,D} L(N,D)$ 满足 $C \approx \kappa\, N D,\ \kappa\approx6$），联合律有 Rosenfeld+ 2020 $Error = n^{-\alpha} + m^{-\beta} + C$ 与 Kaplan+ 2020 $Error = [m^{-\alpha} + n^{-1}]^{\beta}$；取 $D\to\infty$ 或 $N\to\infty$ 即分别退化回模型与数据的单变量幂律。
- Kaplan 处方 $N_{\text{opt}} = C^{0.73}$、$D_{\text{opt}} = C^{0.27}$ 使 $\frac{D_{\text{opt}}}{N_{\text{opt}}}\propto C^{-0.46}$，导向 GPT-3 时代的超大模型；Chinchilla（Hoffmann+ 2022）反驳应训更小模型、喂更多 token，给出 `20 tokens to every parameter` 即 $D \approx 20N$。
- Chinchilla 三种拟合法：方法一 lower envelope $L^*(C) = \min_{\text{runs}} L_{\text{run}}(C)$ 得 67B 与 1.5T token（按 Gopher 的 $5.76 \times 10^{23}$ FLOPs），方法二 Isoflops 沿 $D = \frac{C}{\kappa N}$ 取 U 形谷底得 63B，方法三 joint fits $\hat L(N,D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$ 得 40B；指数依次为 $0.5/0.5$、$0.49/0.51$、$0.46/0.54$。
- 分歧溯源（Wortsman 等）把 $N^*(C)\propto C^{a}$ 的 $a$ 从 `Reproducing Kaplan et al.` 的 0.835 经 0.706、0.602、0.571 降到 0.497，归因于连最后一层 softmax 也排除的参数计数、warm-up 过高与固定大 batch；Pearce–Song 用 $\mathrm{Loss}(N_T, D) = \frac{482}{N_T^{0.35}} + \frac{2085}{D^{0.37}} - 1.82$ 不训练模型纯模拟得 0.51 与 0.78，Epoch AI 重拟合判定方法三 `must have underfitted the original data` 并回到 $D/N = 20$。
- 生产环境因 serving 成本占主导要最小化 $\text{Cost}_{\text{total}}(N,D,U) = C_{\text{train}}(N,D) + U \cdot c_{\text{serve}}(N)$，故主动 overtraining，tokens/param 从 GPT3 的 2、Chinchilla 的 20 一路升到 Llama 3 70B 的 215；讲者收束于 `Data scaling`、`Model scaling`、`Scaling as prediction` 三条，并称 `isoflops is always a good default`，目标是 `engineering at scale without explicitly relying on doing big model training runs`。

## 视频来源

- 原始链接：https://www.youtube.com/watch?v=Q15rhEWZPQ4
- 课程：Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 9

---

## 对照参考笔记补遗（Spring 2025 版）

> 编者注：本节依据 2025 版参考笔记整理，其中的「更正」小节记录的是核对时发现的问题，对应正文已在本版订正；补遗内容不代表 2026 课程视频原文。


本节内容来自 2025 版参考笔记（Qihong Ruan 整理、MinerU 从 PDF 解析的 Spring 2025 lecture notes），仅用于与本讲 2026 视频笔记对照补充，不代表 2026 视频原文；凡与正文口径不同处，一律以正文为准。

### 早期数据缩放：Banko and Brill 2001 与函数形式的摸索

正文已经把 1993 年 Bell Labs 的 Learning Curves、有限假设集泛化界与光滑密度速率 $n^{-\beta/(2\beta+1)}$ 串成「误差随数据量下降」这一条线，也详列了 2012 年前后用递推最小二乘在 Exp、Pow、ILog 三族里比拼 BLEU 曲线的 Table 1。参考笔记在这两者之间还单列了一块正文没有展开的里程碑：Banko and Brill 2001 的 log-linear scaling with data（数据量对数线性缩放）。它研究 confusion set disambiguation（混淆集消歧），配图是 Figure 1「Learning Curves for Confusion Set Disambiguation」，横轴是 millions of words 的语料规模、取对数刻度，纵轴是 test accuracy，两者近似线性；把参考的文字结论记号化，可写成 $\text{Acc}(D)\approx a\log D+b$，其中 $D$ 是训练语料词数，$a$、$b$ 为拟合常数。这与 1993 年的幂律衰减不是同一族函数：一个对 $\log D$ 线性、一个对 $D$ 幂律，正是「函数形式尚未定论」阶段的真实状态。

参考引述的原文论断值得原样记住：「we may want to reconsider the trade-off between spending time and money on algorithm development versus spending it on corpus development」，并且「none of the learners tested is close to asymptoting in performance at the training corpus size commonly employed by the field」。也就是说，当时常用语料规模下各种学习器都还没逼近性能饱和，多收集语料比钻研算法更划算。这与开篇「一万块 B200 该怎么花」及后半段数据策划（data curation）遥相呼应，也预告了预训练时代「数据优先」的直觉。与 2017 年 Hestness 的三区域并读，「曲线何时弯折」这个问题从 2001 年已被提出，正文「直线是局部性质」的警告在此已有伏笔。

易错点有两层。第一是署名与年份：参考明确写 Banko and Brill '01（Stanford），正文的口述转写记作「Branco and Bril」，并把这段与 2012 年的机器翻译工作连在一起讲；对照时应以参考的论文名为准，同时理解正文已声明「不替口述与画面判定谁对」，这是转写差异而非事实错误。第二是函数形式：不要把这里的 log-linear（横轴取对数、纵轴为离散的 accuracy 且近似线性）与正文反复使用的 log-log 直线 $\log L=-\alpha\log D+c$ 混为一谈，后者纵轴是平滑的 loss、双轴取对数；用不连续的 accuracy 去拟合平滑幂律，正是正文借 Hestness 的 accuracy cliffs（准确率断崖）警告过的风险。

### 架构缩放的量化结论：15 倍常数因子与 Tay 等人的跨架构名单

正文在架构比较一节给出的判据是 log-log 下拟合 $L(C)=A\,C^{-\alpha}+L_\infty$、比较两族的斜率与截距，并强调「斜率更差会在大规模被反超」。参考对同一张 Kaplan 图给了更硬的量化结论：Transformers consistently outperform LSTMs with a constant factor gap in compute efficiency，约 15x more efficient（约 15 倍算力效率差距），而且这个 gap holds across different numbers of layers——1 层、2 层、4 层的 LSTM 曲线与 Transformer 曲线彼此平行，差距不随层数收窄。换句话说，2025 版把这张图读成「斜率相同、只差一个 constant factor gap（常数因子差距）」，而 2026 视频读成「截距肯定不同、斜率甚至可能更差」。两者是同一组实验的两种讲法，须按版本并列、不能互相替代；15 倍是算力效率差距，不是损失或参数量差 15 倍，参考也未给出解析式。

参考还把架构比较扩成一次系统性的 cross-architecture scaling（Tay 等人，Google，T5-style 模型）：把 ALBERT、DConv、Funnel、Transformer-GLU、LCconv、MLP Mixer、MoS Transformer、Switch Transformer、Universal Transformer 九种变体各自按不同 FLOPs 预算缩放，统一画 negative log-perplexity（负对数困惑度）对 FLOPs 的散点。结论很干脆：只有 Gated Linear Units（GLU，门控线性单元）与 Mixture of Experts（MoE，专家混合，代表是 Switch Transformer）能 consistently beat the Transformer baseline，各种 efficient attention（高效注意力）类变体并未持续胜出。正文只点名了 Performer、Switch Transformer、Universal Transformer、Transformer-GLU 与 mixture of softmax 并逐条点评，这份完整名单与「只有 GLU/MoE 胜出」的总括判据，正好补上「哪些架构值得放大」的筛选清单，也与正文末尾「挑一个受 DeepSeek 启发的 MoE 之类经过验证的架构」形成因果闭环。

易错点集中在判读而非数字。其一，Switch Transformer 最大 run 的异常、mixture of softmax 按趋势「可能有效」却被弃用，说明「持续胜出」是跨多个 FLOPs 预算的统计结论，而非每个点都赢。其二，正文强调小规模截距优势不能替代斜率判断，而 2025 版的「常数因子」恰恰预设斜率不变，两版一起引用须写明各自前提。其三，这类跨架构 scaling 的价值在于用小规模实验排除候选，正文收尾的 isoflops 正是其落地形式。

### 2025 版对 Kaplan–Chinchilla 分歧的解释：余弦学习率调度

正文用 2024 年的三篇工作（Wortsman 等的逐步复现、Pearce–Song 的纯模拟、Epoch AI/Besiroglu 的重拟合）把 Kaplan 与 Chinchilla 的分歧归因于参数计数（连最后一层 softmax 也排除）、warm-up 过高与固定大 batch，并明确写下「只要 batch / LR 调得对，decay 本身也许并不关键」。参考（2025 版）给出的却是另一套主因，小节标题就叫 Main Difference — Accounting for LR Schedules；本节只转述 2025 版观点，不代表 2026 视频结论，两版理由不可叠加。

参考说模型通常用 cosine learning rate schedule（余弦学习率调度）训练，它包含 warm-up（预热）、decay（衰减）与 cool-down（冷却）三相。关键判断是「You cannot truncate a cosine learning rate schedule early and expect the same model quality as a full run」：长训练中途截断，并不等价于为该短时长从头设计 schedule、从头训较小模型，后者学习率恰在终点衰减到位，前者停在半途。Kaplan 被归为 assumed that truncating training runs was equivalent to training a smaller model for a shorter time，参考判为 flawed（有缺陷）；Chinchilla 则 accounted for the full cosine learning rate schedule，对每个 data/model 组合训到收敛。即 2025 版认为分歧不在联合律函数形式，而在拟合时是否完整计入学习率调度。

两种口径落在同一组数字上：Kaplan 的 compute-optimal 处方 $N_{\text{opt}}=C^{0.73}$、$D_{\text{opt}}=C^{0.27}$，对应约 2 tokens per parameter；Chinchilla 给出 $D\approx 20N$、约 20 tokens per parameter，参考称约 10x 差距，并补充 Gopher (Rae et al., 2021) 与 Megatron-Turing NLG (Narayanan et al., 2021) 的比例更接近 Kaplan 初始估计。正文已讲透这两个指数与 tokens/param 清单，此处只补「分歧曾被归因于 schedule 是否被完整计入」这一层解释与两篇文献坐标。

易错点有三。第一，这是 2025 版的因果判断，与正文 2026 版「decay 也许并不关键、关键是参数计数与 warm-up/batch」方向相反，引用时必须带上版本标签。第二，参考的 Chinchilla 三方法里 Method 1 与 Method 2 给出约 0.5/0.5 的指数，Method 3 却被写成「0.73 for a and 0.27 for b in Kaplan's initial work」，这实际是把 Kaplan 的指数混进了 Method 3 的描述；正文把 0.73/0.27 归 Kaplan、把 Method 3 记为 0.46/0.54 才是自洽的归属，读参考时对这一处要保持怀疑。第三，即便接受 schedule 解释，也不能推出「换一条 schedule 就能任意外推」——正文反复强调 scaling law 是对给定配方的条件预测，配方一变，指数就要重估。

### IsoFLOP 的普适性与边界：扩散模型与下游任务

正文收尾时说「自己的一个学生把这套设计用在 diffusion models 上」，并把 isoflops 称为经受住时间考验的工具。参考给了这项工作的完整坐标：Gulrajani+ (2023) 把 IsoFLOP 分析同时施加于文本上的 auto-regressive（自回归）与 diffusion（扩散）模型；自回归模型给出熟悉的 chinchilla-like 曲线与最优 scaling，扩散模型尽管是完全不同的生成模型，也呈现非常相似的曲线与可预测 scaling；把两类最小值画在一起，二者仅由 constant offset（常数偏移）分隔，记号化即 $L_{\text{diff}}(C)\approx L_{\text{AR}}(C)+\Delta$，其中 $\Delta$ 是与 compute 无关的模型族偏置（此式只是对参考「separated by a constant offset」的记号化，并非原论文给出的公式）。作者还写明「our group has replicated these results」，即本组已复现。

这支撑正文「scaling laws 不是 cherry-picked 个例」的判断，也解释了 isoflops 为何成为默认工具：只需固定一个 FLOPs 预算、扫过其余自由度看损失曲面谷底，不依赖先假设 $L(N,D)$ 的解析形式。与正文的 MoE「参数价值」side note 并读更清楚——那里把参数解耦成 total parameters、active parameters 与 sparsity，同样是在 IsoFLOP 曲面上问「这批 flops 该花在更多激活参数还是更高稀疏度上」。

但参考同时给出一条反向警告，正好补上正文的边界。Tay et al. (2023) 的左图 negative log-perplexity 对参数量是很干净的 log-linear，右图 SuperGLUE accuracy 对参数量却离散得多，不同架构与超参差异巨大；state-space models（状态空间模型，如 Mamba）在 pre-training scaling 上表现很好，却可能在 in-context learning 或 QA 这类具体任务上表现不同。正文用 T5 的 NL-12 对 NL-32 XL 举过同一类错配，这里补的是一个当代系统名，说明错配并非孤例。易错点在于：常数偏移 $\Delta$ 只说明「两类模型各自内部可预测」，并不说明扩散模型的下游能力与自回归模型可比；把预训练困惑度的可预测性直接搬到下游，正是正文要求警惕的 transfer 假设。

### 产业落地与工程例证：MetaP、动态批大小与联合律外推验证

正文给 μP（Maximal Update Parametrization，最大更新参数化）的落点是 Table 2 的逐条缩放规则与「在小模型上调一次学习率、迁移到最大规模」两条哲学，并注明成功报告并不一致。参考补了产业例证：「Labs (e.g., Meta's "MetaP" for Llama 4) are adopting similar ideas to simplify scaling」——这类 scale-aware initialization（尺度感知初始化）与学习率缩放已进入前沿实验室训练流程，省掉大 run 上的学习率搜索。与正文「Standard Practice 的 optimum shifts 对 Our Work 的 optimum stable」并读可见，μP 的卖点不是更高绝对精度，而是让最优学习率跨宽度保持不变。

批大小一侧，正文已给出临界批大小随目标损失变化的幂律 $B_{crit}=2.1\times10^{8}\ \text{tokens}\cdot L^{-4.8}$ 与「The smaller the loss target, The bigger the batch」。参考补上两条工程观察：训练报告（如 Llama 3）常随 loss 下降而增大 batch size，正是这条幂律的真实投影；Kaplan 的分析还指出，对给定 compute 预算，总步数可以保持相对恒定而 batch sizes 增大，这对 data parallel（数据并行）是好消息——它把「批大小该开多大」从统计效率问题转成并行机会问题。参考还把 critical batch size 定义为「达到目标损失所需的最少样本数 / 所需最少步数」，与正文 $\mathcal B_{crit}=E_{min}/S_{min}$ 互为印证。

正文说联合律「沿幂律外推仍能相当准确地预测误差」，参考点名了验证场景：Rosenfeld 等在小 data、小 model 的子空间拟合指数后外推，predictions（y 轴）与 real values（x 轴）在 ImageNet 与 WikiText-103 上都高度吻合，记号化即 $\hat L(N,D)\approx L(N,D)$（同样只是对参考文字的记号化）；Kaplan et al. (2020) 与 Brown et al. (2020) 则指出固定 FLOPs 下模型规模与数据量存在最优平衡，把 test loss 对 parameters 按不同 compute 预算着色后，每个预算的最小 loss 又构成一条 scaling law。易错点有三：这些是 2025 版的文献坐标，正文把 2 tokens/param 归在 GPT-3 项而非 Kaplan 名下，引用勿混归属；$\hat L\approx L$ 只在配方不变时成立；ImageNet 与 WikiText-103 分属视觉与语言，不能据此推断任意下游任务都可外推——与上一节的下游警告是同一条底线。
