---
title: "第 1 讲 · 以数据为中心的 AI 与以模型为中心的 AI"
description: "Data-Centric AI vs. Model-Centric AI · MIT DCAI IAP 2024 中文图文笔记"
---

# 第 1 讲｜以数据为中心的 AI 与以模型为中心的 AI：把「调模型」换成「改数据」

> **MIT Introduction to Data-Centric AI · IAP 2024 · L01** ｜ 48 分钟 ｜ [原视频](https://www.youtube.com/watch?v=gEXGhlouwjM)

## 核心结论

本讲要翻转一个默认习惯：改进机器学习系统时，着力点不该只放在模型上，而应放到数据上。真实场景里数据集从来不是给定的常量，它充满错标、异常值与歧义，且可以自由增删改；用系统性、可算法化的方法诊断并修正数据，常常比继续调结构、调损失、调超参数更能提升性能。讲者把「when algorithms mess up」改写成「when algorithms are trained with erroneous data」——深度网络能忠实拟合随机标签，恰恰说明出错根源是被污染的训练数据，而非模型容量不足。以模型为中心的 AI（Model-Centric AI）与以数据为中心的 AI（Data-Centric AI, DCAI）理想上应当同时做，但学校几乎只教前者、工业界却把时间几乎全花在数据上，本课正是补上缺失的那半边，并给出方法谱系与一套可落地的实践配方。压轴的是被讲者称为「以数据为中心的 AI 的感知机」的 PU 学习：仅靠两个假设和一个常数 $c$，就把在噪声标签上训出的二分类器还原为近似在干净标签上的效果，其局限由下一讲的置信学习放开。

## 一、观念的起点：把「模型出错」改写为「数据出错」 [00:00 - 04:32]

![课程标题页：Introduction to Data-Centric AI，副标题为Data-Centric AI vs Model-Centric AI](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0001_00-00-00.jpg)

图注：对应本讲开篇的课程定位——标题页点明主题「Introduction to Data-Centric AI」，副标题「Data-Centric AI vs Model-Centric AI」预告全课要对比的两种范式。

本讲由 Curtis Northcutt 讲授，属 MIT IAP 2024 课程《Introduction to Data-Centric AI》第一讲（2024-01-16）。课程默认听众修过机器学习、熟悉感知机（Perceptron）与随机梯度下降（Stochastic Gradient Descent, SGD），否则后续黑板推导会吃力。讲者把前 10–15 分钟用来讲清「什么是以数据为中心的 AI」，之后便上黑板推导一个他称之为「以数据为中心的 AI 的感知机」的算法。贯穿全课的关键词是系统性（systematic）：所谓改进数据，不是凭直觉挑几条样本，而是「能被写进程序、由计算机执行」的方法。全讲可以收敛成一个问题——给定要训练的模型和一份数据，如何通过改进数据来改进模型？

![幻灯片指出在机器学习中数据与模型同等重要，并引用自动驾驶事故案例说明算法出错时人类受责](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0016_00-02-18.jpg)

图注：对应本段的核心理念「data is as important as the model」——画面借自动驾驶事故报道说明，人们习惯把算法出错归咎于「离得最近的那个人」，而真正该追问的是喂给算法的数据。

讲者借一张自动驾驶事故的杂志封面切入：原题是「When algorithms mess up, the nearest human gets the blame」，他把它改写成「When algorithms are trained with erroneous data」。一句话换了主语——不是算法自己出错，而是被错误数据训练后才出错。支撑这个改写的是 Zhang 等人（ICLR 2017）的发现：深度神经网络很容易拟合完全随机的标签。它的含义不是模型不够强，恰恰相反：模型强到能把喂进来的任何标签都忠实背下来，于是数据错了，输出就稳定地错。

![与上一张相同的幻灯片，但标题文字被修改为erroneous data](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0029_00-04-24.jpg)

图注：对应本章标题的落点——同一张幻灯片把标题文字改成「trained with erroneous data」，并列出「Deep neural networks easily fit random labels」，把「算法出错」正式重构为「算法被错误数据训练」。

两个类比把这件事讲得很直白：如果人拥有无限记忆、且从不被灌输错误信息，大概率比被反复喂错信息更聪明；一个学生遇到「30% 时间在撒谎」的老师，另一个遇到从不撒谎的老师，期末成绩的差距更多来自被告知的内容，而不是两人的脑子。讲者补充，据他在湾区与自动驾驶公司创始人共事的经验，主流团队普遍受此困扰，因此这不是修辞，而是严肃的工程现实。

## 二、两种范式：以模型为中心的 AI 与以数据为中心的 AI [04:32 - 08:46]

![传统机器学习是以模型为中心的幻灯片，介绍学校ML课程的数据集特点](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0030_00-04-36.jpg)

图注：对应传统机器学习「以模型为中心」的起点——学校里数据集是给定的、通常干净且精心整理（如猫狗图片），学生的任务只是在这份固定数据上产出最好的模型。

![传统以模型为中心ML的学习内容列表，包括模型类型、超参数调优和损失函数修改](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0033_00-05-00.jpg)

图注：对应以模型为中心范式的教学内容清单——不同模型类型（如神经架构）、超参数调优、修改损失函数与正则化，这些正是标准机器学习课（MIT 6.036 / 6.390）围绕固定数据集展开的全部手段。

把这套课堂设定抽象出来，就是以模型为中心的 AI（Model-Centric AI）：数据集是常量，目标是为它产出最佳模型，于是课程围绕模型类型、训练技巧（正则化、优化、损失函数）、超参数选择与集成展开。但官网 notes 开篇就强调，真实世界的数据不是固定的：你可以增删样本、按预算采集新数据，甚至重新定义类别。讲者给了一个很实用的技巧——把只有三五个样本的稀有类别统统塞进新建的杂物类（clutter class），分类器便只需专注最重要的二十来个类，任务变简单，在你真正在意的指标上反而更好。

![完整幻灯片：现实ML应用中数据集问题，含标签错误统计及NeurIPS 2021论文引用（Northcutt, Athalye, & Mueller）](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0045_00-06-54.jpg)

图注：对应「现实数据并不干净」这一转折——画面点出用户并不在乎你在精心整理的数据上用了多炫的技巧，并引用 labelerrors.com 与 Northcutt、Athalye、Mueller（NeurIPS 2021）说明十个最常用的测试集都普遍存在标签错误。

真实数据充满错误、异常值、分布偏移与歧义，和作业里「完美裁剪的猫脸标 cat、狗脸标 dog」完全是两回事。由此引出「垃圾进、垃圾出（garbage in, garbage out）」：资深数据科学家更愿意投资于探索并修复数据，只是大数据集上手工修数据极其繁琐。这一步正是以数据为中心的 AI 要系统化接管的地方。

![以数据为中心的 AI的两种形式：理解数据的算法（如课程学习）和修改数据的算法](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0049_00-07-36.jpg)

图注：对应以数据为中心的 AI 的两种形式之一——「用算法理解数据并借此改进模型」，画面以课程学习（Curriculum Learning，Bengio 等，ICML 2009）为例，本质是改变数据呈现的顺序，先学简单样本。

官网 notes 把以数据为中心的 AI 归为两类，且「哪些数据容易、哪些被错标」都由算法基于已训练模型的输出自动估计，不靠人拍板：其一是理解数据并用它改进模型，典型是课程学习（Curriculum Learning），把样本由易到难排序；其二是修改数据以改进模型，典型是置信学习（Confident Learning, CL，Northcutt 等，JAIR 2021），在剔除了错标样本的数据集上训练。

![以模型为中心的 AI与以数据为中心的 AI的对比定义幻灯片](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0053_00-08-12.jpg)

图注：对应两种范式的正面定义对比——以模型为中心的 AI「给定数据集、改模型」，以数据为中心的 AI「给定任意模型、改训练数据集」，一个改结构损失超参，一个系统性、算法化地改数据本身。

把两者并排看最清楚：以模型为中心的 AI 是「给定数据集，尽力产出最好模型，通过改模型来提升性能」；以数据为中心的 AI 是「给定任意模型，尽力改进训练数据集，通过系统性、算法化地改数据来提升性能」。理想情况两者同时做，但讲者观察到一个刺眼的反差：很多人在学校里约 80% 的时间都在调模型、几乎不碰数据，进入工业界真正用模型改变世界后，时间却几乎全花在数据上。本课的目标因此被写成一句口号：start thinking about ML in terms of data, not the model。

## 三、以数据为中心的思维方式：从 KNN 说起 [08:49 - 13:27]

![关于KNN与课程内容的思考问题幻灯片](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0058_00-09-00.jpg)

图注：对应本章的思维转折——用 K 近邻（K Nearest Neighbors, KNN）发问「它和本课要教的东西差在哪」，把目标明确为「从数据而非模型的角度思考机器学习」。

KNN 是很好的桥：它非参数、距离度量可自选，输入就是数据本身，改数据就会改答案，已经带一点以数据为中心味道。本课要更进一步：除了为分类学一组参数，还要专门学一组用来理解数据的参数，先刻画（characterize）数据，再把这份信息用回机器学习流程——提供不同的数据、重排样本或重构数据结构——从而训出更好的模型。也就是说，在主任务之外额外做一个「理解数据」的独立任务，再用它反哺主任务；讲者提醒，这种「连点成线」的模式在整门课里会反复出现。

![什么不是以数据为中心的 AI：手动挑选与简单扩充数据集示例](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0077_00-12-00.jpg)

图注：对应「什么不是以数据为中心的 AI」的两个反例——凭直觉手工挑一批觉得能改善模型的数据点，以及单纯把数据集规模翻倍再重训，画面把两者并列列为反面教材。

![非以数据为中心的 AI的反面案例与核心集选择概念页](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0082_00-12-48.jpg)

图注：对应从反面案例过渡到正面方法——在「非以数据为中心的 AI」清单旁边引出核心集选择（Coreset Selection），即自动找出「只训一小把就能达到最好效果」的代表性子集。

两个反例值得记牢。第一，手工挑样本：在 Google、Meta、Amazon 动辄万亿级样本面前完全不可扩展。第二，把数据集翻倍、在更多数据上重训：它背后是过去 10–15 年从追逐「大数据」的数量转向追逐数据质量的范式转移，因为数据不可能无限加。它们对应的以数据为中心版本分别是核心集选择（Coreset Selection）——自动识别那一小撮「只训它们就够好」的样本，优化训练时间、质量与成本的 Pareto 前沿——以及数据集增强（Dataset Augmentation），用系统化手段模拟或生成更多数据，而不是靠人去搜集。共同点仍是那个词：systematic / algorithmic。

## 四、方法谱系与一套可落地的实践配方 [13:29 - 17:42]

![以数据为中心的 AI的三大核心示例：离群点处理、错误修正与数据增强](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0094_00-14-42.jpg)

图注：对应以数据为中心的 AI 最核心的三个示例——离群点／异常值检测与移除（Outlier Detection and Removal）、错误检测与纠正（Error Detection and Correction）、以及数据增强（Data Augmentation）的引入。

![以数据为中心的 AI的五个示例列表，包括异常值检测、错误修正、数据增强等](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0105_00-16-30.jpg)

图注：对应扩展后的完整方法列表——在前三项之外补上特征工程与选择、建立共识标签等，把「系统化改数据」从三个动作铺成一条方法谱系。

课堂上展开的方法（官网 notes 另补课程学习）可归为七类：异常值检测与移除，处理分布外的异常样本，最简单的做法是逐维看取值分布、把偏离太远的丢掉，再到 z-score、one-class SVM；错误检测与纠正，找出错标或错误取值并自动纠正或剔除；数据增强，按行生成新样本、按列生成新特征，把先验知识编码进数据；特征工程与选择（Feature Engineering and Selection），改变数据的表示；建立共识标签（Consensus Labels），从众包的多份标注里推断真值；主动学习（Active Learning），挑「最值得标注」的下一条；课程学习，把样本按由易到难排序。

例子都很具体：数据增强在图像里是形状变换，在文本里是反向翻译（reverse translation，英译德再译回英得到措辞被扰动的新样本）；共识标签的例子是一张模糊薯片图，标注者分别给出 oval、potato chips、asparagus chip，需要算法收敛出真值；主动学习的设问是——1000 万条未标注样本里若只能再标一条，选哪条能让性能从 80% 提到 85%，并如此迭代到标完约 1 万条，以极小成本换显著提升。

官网 notes 还给出四步实践配方：① 探索数据、修掉基础问题并转成机器学习可用形式；② 训练一个基线模型；③ 用这个模型反过来改进数据集（本课重点）；④ 在改进后的数据上尝试不同建模技巧得到最佳模型。notes 特意提醒别图快从第 2 步直接蹦到第 4 步，第 3、4 步可多轮迭代——这正是下一章「闭环」思想的雏形。

## 五、为什么需要以数据为中心的 AI：成本、热度与三个标志性案例 [17:42 - 22:41]

![Forbes文章截图，高亮了吴恩达关于80%时间用于数据准备的观点](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0114_00-17-54.jpg)

图注：对应「时间都花在哪」这一成本证据——Forbes 报道高亮吴恩达（Andrew Ng）的观察：AI 开发者约 80% 的时间花在数据准备上，而不是建模。

![并列展示Forbes文章与哈佛商业评论关于坏数据成本3万亿美元的文章截图](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0117_00-18-24.jpg)

图注：对应「坏数据的代价」这一成本证据——哈佛商业评论标题「Bad Data Costs the U.S. $3 Trillion Per Year」，把低质量数据的损失量化到每年约 3 万亿美元。

吴恩达长期倡导以数据为中心的 AI，因为他在 Google Research、Coursera 等项目中反复发现大部分时间耗在数据上；notes 引 Redman（HBR, 2016）给出「坏数据每年给美国造成约 3 万亿美元损失」。共同逻辑是：数据集越大，越不可能靠人工保证质量；即便像 ChatGPT 这样投入海量人力做人类反馈，也难以完全弥补低质量训练数据的短板。当机器学习深入医疗、金融、交通，可靠训练已从「加分项」变成「必须项」。

![Hinton在FAIR演讲中展示MNIST数据集中被错误标记为3的数字5样本](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0126_00-19-48.jpg)

图注：对应第一个标志性案例——Hinton 在 FAIR 演讲结尾展示 MNIST 里一个被错标成 3 的数字 5，说明连被默认「标注完美」的经典数据集也有标签错误。

2016 年夏，FAIR 纽约办公室还只有十来人时，图灵奖得主 Jeff Hinton 在一场小会上向 Yann LeCun 介绍完胶囊网络，兴奋地宣布：用新方法在 MNIST 里找到一处标签错误——一个 5 被标成了 3。讲义保留着那句原话「To conclude my talk, I will show that our method finds a label error in Yann's MNIST dataset」。MNIST 被引约四万次、常被默认标注完美，而当时最顶尖的学者要为「系统性地找到一个错标」而兴奋，恰好标定了 2016 年的水位：连「发现错误」本身都还是新闻。

![ChatGPT通过改进数据质量来提升GPT-3性能的推文截图示例](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0134_00-21-06.jpg)

图注：对应标志性案例里最贴近当下的一点——ChatGPT 相对 GPT-3 的主要跃迁被归为「提升数据质量」，画面用对话示例展示这种由数据侧驱动的能力提升。

从 GPT-3 到 GPT-3.5 / ChatGPT 的最大差别，讲者归为数据质量而非架构：用人类对候选输出的排序做微调，压低有害、不实、带偏见的输出权重；「情感上讨不讨喜」和「陈述是否为真」两侧都要人在回路里逐条检查错误。这本质上是大规模地做数据质量，成本高达数亿美元。同一段还补了一句更直白的判断：OpenAI 公开说过训练 DALL·E、GPT-3 时最大的问题之一正是数据与标签错误，把飞机标成 car，用户要 car 就会画出飞机——问题不在模型，在数据（It's not the model, it's the data）。面对数亿级样本，只有系统化算法才能把这类错误捞出来；而在自动驾驶里把停止标志误读，后果更是致命。结论落在 notes 的那句话上：部署模型的可靠性取决于训练数据的质量。

## 六、闭环与证据：Tesla 数据引擎与 CIFAR-10 噪声基准 [22:41 - 28:08]

![Tesla数据引擎流程图：利用模型输出发现问题并补充训练数据形成闭环](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0145_00-22-48.jpg)

图注：对应本章的「闭环」主题——Andrej Karpathy 的 Tesla 数据引擎流程图，展示用模型在部署中的输出反过来发现数据缺口、补数据、再训练，把配方第 3 步变成一个自动循环。

这张幻灯片来自前 Tesla Autopilot 负责人 Andrej Karpathy。闭环是这样转的：车没见过隧道里的暗光数据 → 预测变差甚至出事 → 派人把车开进更多隧道采数据 → 标注（昂贵）→ 重训（昂贵）→ 部署 → 上金门大桥又没见过桥 → 再出事、再采数据……每一步都绕不开数据质量，真正该做的是把这个 loop 自动化。notes 认为，正是持续的「模型辅助数据集改进」让 Tesla 的辅助驾驶远胜对手，数据引擎被视为关键。

![Tesla数据引擎案例：各种复杂交通信号灯场景图片集合](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0152_00-24-00.jpg)

图注：对应闭环里「现实数据有多模糊」的具体案例——各种复杂交通信号灯场景的图片集合，说明同一类对象在真实世界里的形态可以差异极大，属于可被算法识别的模糊数据（ambiguous data）。

讲者用「让 Tesla 检测红绿灯」的小故事把模糊性讲透：现实中的信号灯样式、角度、遮挡千差万别，这类样本确实是 stop light，却往往不在整齐的数据集里。一个可用的判据是看多个算法对它的输出概率——如果总是卡在中间、从不接近 1，就可以把它标为模糊数据并单独处理。

![CIFAR-10数据集上40%标签噪声下以数据为中心与以模型为中心的学习方法准确率对比表](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0161_00-25-24.jpg)

图注：对应「改数据胜过改模型」的量化证据——CIFAR-10 上人为加入 40% 标签噪声后，以数据为中心方法与以模型为中心方法的准确率对比表，排名结果支撑本章结论。

这张被讲者称为「很有激励性」的基准来自 Northcutt 等（JAIR 2021）：在 CIFAR-10 上人为把 40% 训练标签弄错，训练时看不到真标签，最后在留出测试集评估。左列是学术设定，错误均匀分布到所有类；右列更贴近现实，多数跨类翻转率为 0，错误只集中在语义相近的类之间（tiger 与 lion/cat、missile 与 projectile，几乎不会 bear 与 computer 互翻）。两个结论：其一，某些方法从学术设定搬到真实设定会大幅掉点，因为错误不再均匀、更难；其二，按准确率排名时出现「魔法」——垫底的几乎全是改损失函数的以模型为中心方法，靠前的几乎全是改数据的以数据为中心方法。当数据里的错误足够多，改数据就是比改模型更有效。此前人们只能靠人力硬扛：花更多钱买更高质量的数据、搭定制工具、在 notebook 里手工修；以数据为中心的 AI 要做的，就是把这些零散做法系统化。

## 七、PU 学习：以数据为中心的 AI 的「感知机」 [28:32 - 47:39]

![黑板上展示了从潜在变量到观测变量的映射关系及条件概率定义 f(x)=P(y*=1|x)](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0218_00-34-30.jpg)

图注：对应 PU 学习的建模起点——板书把 $(x,y^*)$ 标为潜在真值、$(x,\tilde{y})$ 标为观测到的含噪标签，并写下要估计的目标 $f(x)=P(y^*=1\mid x)$。

为什么叫「感知机」？因为它像机器学习第一课里的感知机一样，简单却能引出全套思想。感知机的目标是估 $f(x)=p(y=1\mid x)$，迭代更新 $w_z(t+1)=w_z(t)+\eta\,(y_i-\hat{y}_i(t))\,x_{i,z}$，与神经网络的 SGD 更新 $w(t+1)=w(t)-\eta\nabla L(w(t))$ 几乎同构。PU 学习（Positive-Unlabeled Learning，Elkan 与 Noto，KDD 2008）则是「在噪声标签上系统性改进二分类器」最早的方法之一，同样简洁。

设定里每个样本是三元组 $(x_i,\tilde{y}_i,y^*_i)$：$x_i\in\mathcal{R}^d$ 是样本（图像／文本嵌入或表格数据），$\tilde{y}_i$ 是观测到的含噪标签，$y^*_i$ 是看不到的潜在真标签。正类记为 1，其余未标注样本被草率地全当作 0，于是 0 类里混着真正为 1 的样本。我们想要的仍是 $f(x)=p(y^*=1\mid x)$，但用观测数据只能训出 $\tilde{f}(x)=p(\tilde{y}=1\mid x)$；中间目标因此变成：不观测 $y^*$，如何从 $\tilde{f}$ 还原出 $f$。

![PU learning 的两条核心假设公式及定义域标注](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0243_00-38-24.jpg)

图注：对应推导所依赖的两条核心假设——黑板上写下「正类标签无错误」$P(\tilde{y}=1\mid x,y^*=0)=0$ 与「条件独立」$P(\tilde{y}=0\mid x,y^*=1)=P(\tilde{y}=0\mid y^*=1)$，它们是后面所有闭式解的基础。

两条简化假设让闭式解成为可能。假设 1（正类标签无错误）：$p(\tilde{y}=1\mid x,\,y^*=0)=0$，即被观测为正的样本绝不可能是真负样本。假设 2（条件独立，Conditional Independence）：

$$p(\tilde{y}=0\mid x,\,y^*=1)=p(\tilde{y}=0\mid y^*=1),$$

即标签从 1 翻成 0 的概率不依赖数据本身；由于两类概率和为 1，它等价于 $p(\tilde{y}=1\mid x,y^*=1)=p(\tilde{y}=1\mid y^*=1)$。这是「有道理但确实是假设」：更像狗的图也许更容易被错标，不过当类别很多时，翻转率大致落在一个范围内。

![黑板左侧PU学习假设条件与右侧Lemma 1的推导过程](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0258_00-40-48.jpg)

图注：对应引理 1 的推导过程——黑板左侧保留两条假设、右侧把观测概率逐步转成潜在真值概率，最终得到 $P(y^*=1\mid x)=P(\tilde{y}=1\mid x)/c$。

在条件独立假设下得到引理 1：

$$p(y^*=1\mid x)=\frac{p(\tilde{y}=1\mid x)}{c},\qquad c=p(\tilde{y}=1\mid y^*=1).$$

![黑板上关于PU学习定理的公式推导及证明过程](/coursenotes/mit-dcai-iap-2024/l01-data-centric-vs-model-centric/cand_0271_00-42-54.jpg)

图注：对应引理 1 的证明——板书用全概率公式把 $P(\tilde{y}=1\mid x)$ 分解成两项，再借两条假设消去一项、剥离对 $x$ 的依赖，落到一个只差常数 $c$ 的乘积。

证明只用初等概率的边缘化（marginalization）：

$$
\begin{aligned}
p(\tilde{y}=1\mid x) &= p(y^*=1,\tilde{y}=1\mid x)+p(y^*=0,\tilde{y}=1\mid x)\\
&= p(y^*=1,\tilde{y}=1\mid x)+0 \qquad\text{(假设 1)}\\
&= p(y^*=1\mid x)\,p(\tilde{y}=1\mid y^*=1,x)\\
&= p(y^*=1\mid x)\,p(\tilde{y}=1\mid y^*=1) \qquad\text{(假设 2)}\\
&= p(y^*=1\mid x)\cdot c .
\end{aligned}
$$

于是全部难题缩减为估计一个常数 $c$。最简估计量是在（按定义无错标的）正样本集 $\mathcal{P}$ 上对模型预测概率取平均：

$$\tilde{c}=\frac{1}{\mid\mathcal{P}\mid}\sum_{x\in\mathcal{P}}\hat{p}(\tilde{y}=1\mid x),$$

其直觉是 $\mid\mathcal{P}\mid\to\infty$ 时该平均收敛到 $p(\tilde{y}=1\mid y^*=1)=c$；又因 $\hat{p}$ 只是模型输出，$\tilde{c}$ 本质上是对预测概率的误差取平均后的估计，模型越差、偏差越大。落地只有四步：① 训练步，用交叉验证取得样本外（out-of-sample）预测概率，否则模型见过这些样本后概率会过拟合到 0 或 1；② 刻画误差（以数据为中心步），按上式算 $\tilde{c}$；③ 最终训练步，丢掉此前所有概率与分类器，从零在全量数据上重训 $\tilde{f}$；④ 推断步，$f(x_{\text{new}})=p(\tilde{y}=1\mid x_{\text{new}})/c$，若不小于 0.5 判为类 1，否则类 0。讲者笑称「大概三行 Python」就能从错标分类器得到近似干净标签的效果——概念上绕，算法上极简。

它的局限也很清楚：必须有一类完全无错、只适用二分类，且 $c$ 由预测概率平均得到、对概率校准敏感。下一讲放开这些假设，进入适用任意多分类、允许任意类错标成任意类、对不完美概率更稳健的置信学习（Confident Learning）。

## 要点回顾

- 真实世界的数据集不固定：可增删改、可改类别（clutter class）、可按预算采集；「垃圾进、垃圾出」让修数据成为训出好模型的前提。
- 以模型为中心的 AI 改模型（结构、损失、超参数、集成），以数据为中心的 AI 系统性、算法化地改训练数据集；理想是两者同做，学校与工业界的时间分配却恰好相反。
- 以数据为中心的 AI 两种形式：理解数据并借此改进模型（课程学习）、修改数据以改进模型（置信学习）；「易／错标」由算法基于模型输出自动估计。
- 两个反例：手工挑样本、单纯把数据集翻倍；对应正解是核心集选择与数据集增强，优化训练时间、质量、成本的 Pareto 前沿。
- 七类方法与四步配方：异常检测与移除、错误检测与纠正、数据增强（含反向翻译）、特征工程与选择、共识标签、主动学习、课程学习；配方里别跳过「用模型改进数据」这一步。
- 现实证据：约 80% 时间用于数据准备、坏数据每年约 3 万亿美元损失；Hinton 在 MNIST 找到单个错标（5 标成 3）；ChatGPT 靠大规模人类反馈做数据质量；Tesla 数据引擎自动化「采集—标注—重训—部署」闭环。
- CIFAR-10 40% 噪声基准：错误集中在少数类时更难，而排名靠前的全是改数据的方法、垫底的全是改损失函数的方法。
- PU 学习用三元组 $(x,\tilde{y},y^*)$、两条假设与 $c=p(\tilde{y}=1\mid y^*=1)$ 得到 $f(x)=\tilde{f}(x)/c$，四步即可落地，是「以数据为中心的 AI 的感知机」；其局限由下一讲置信学习解决。

## 视频来源

https://www.youtube.com/watch?v=gEXGhlouwjM
