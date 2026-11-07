---
title: "第 3 讲 · 进阶置信学习与生成式 AI 应用"
description: "Advanced Confident Learning, LLM and GenAI Applications · MIT DCAI IAP 2024 中文图文笔记"
---

# 第 3 讲｜进阶置信学习：它为什么成立，以及生成式 AI 上的用法

> **MIT Introduction to Data-Centric AI · IAP 2024 · L03** ｜ 49 分钟 ｜ [原视频](https://www.youtube.com/watch?v=hjbpRjr10Mc)

## 核心结论

本讲是三讲的收束：第 1 讲建立「以数据为中心」的观念，第 2 讲给出置信学习（Confident Learning, CL）的算法与计数矩阵，本讲回答「凭什么它是对的」并展示能用它做什么。核心洞察是把含噪的模型概率换成**计数与排序**：只要预测概率越过按类阈值被计入格子，具体是 0.71 还是 0.9 都无关紧要，模型误差便不会像损失重加权（loss reweighting）那样被注入每一次梯度更新。同一个置信联合矩阵还能派生出两种错误排序打分、用 $|X|-\mathrm{Tr}(C)$ 自动估计错误总数、三种剪枝策略、反查类别定义问题。理论上，类内一致的系统性校准偏差会在阈值两侧直接抵消；应用上，这套思路可同时用于生成式模型的「进」与「出」两端，也能为缺乏置信度的大语言模型输出补上分数。

## 一、三讲递进与算法回顾：从计数矩阵到联合分布 [00:00 - 06:20]

本讲是三讲的 culmination：L1 在受控二分类里建立数据中心 AI 概念，L2 推广到任意分类的算法，今天给出**为什么有效、成立条件**并看真实应用。算法输入只有含噪标签 $\tilde y$ 与样本外预测概率 $\hat p(\tilde y=i;\boldsymbol x,\boldsymbol\theta)$，且 CL 对输出尺度不变、原始 logits 亦可用。第一步给每类算阈值，作为模型对该类平均自信度的代理：

$$t_j=\frac{1}{|\boldsymbol X_{\tilde y=j}|}\sum_{\boldsymbol x\in \boldsymbol X_{\tilde y=j}}\hat p(\tilde y=j;\boldsymbol x,\boldsymbol\theta)$$

![Confident Learning 核心思想：用阈值 t_j 作为机器自置信度的代理，给出按类平均预测概率的公式](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0009_00-01-18.jpg)

图注：对应本段「按类阈值 $t_j$」的核心思想——用给定类样本的平均预测概率作机器自置信度代理。

第二步按阈值把每个样本判为错标、标对或离群点，形成置信集合：

$$\hat X_{\tilde y=i,\,y^*=j}=\{\boldsymbol x\in X_{\tilde y=i}:\hat p(\tilde y=j;\boldsymbol x,\boldsymbol\theta)\ge t_j\}$$

dog/fox/cow 的例子里，逐个样本检查「对某类的预测概率是否大于该类阈值」并往矩阵里填计数：非对角线是标签错误，对角线是标对样本，哪个阈值都没超过的是离群点（outlier）——这正是「confident learning」只把有信心标对的数据交给学习的由来。

![Confident Learning 工作原理的直观示例：用 dog/fox/cow 图像展示噪声标签、预测概率柱状图及混淆矩阵 C 的构建过程](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0012_00-01-48.jpg)

图注：对应本段「阈值比较构建混淆矩阵」——dog/fox/cow 概率、按类阈值与 $C_{\tilde y,y^*}$ 的构造。

遍历全数据集得到计数矩阵 $C_{\tilde y,y^*}[i][j]=|\hat X_{\tilde y=i,y^*=j}|$。

![遍历整个数据集后得到的完整噪声混淆矩阵C_{y~,y*}，包含具体计数数值](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0015_00-02-18.jpg)

图注：对应本段「遍历全数据集后的最终计数」——三类完整计数矩阵，是行归一化的基础。

行归一化（把每行缩放到观测到的该类总数）后再除以总样本数，即得联合分布 $\hat p(\tilde y,y^*)$；由此按列除得翻转率 $p(\tilde y\mid y^*)$、按行除得逆翻转率、对列求和得真标签先验 $p(y^*)$。整套落地不过「一次 import + 一行代码」的 `find_label_issues(...)`。真实规模的演示是在 ImageNet 训练集（约 270 GB）上原样跑这套算法，界面直接给出标签问题、离群点与模糊样本的概览。

![Cleanlab Studio 界面展示 ImageNet 数据集的标签问题概览](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0032_00-05-06.jpg)

图注：对应本段「ImageNet 上的规模效果」——界面汇总标签问题、离群点与模糊样本数量。

例如 CRT screen 实为 oscilloscope、Fountain 实为 geyser、Palace 实为 castle，配简单嵌入检索即可肉眼确认；这类问题数以万计，说明算法能规模化。

![Cleanlab Studio 中具体的标签错误示例列表（如 CRT screen 误标为 oscilloscope）](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0036_00-05-42.jpg)

图注：对应本段「具体错标案例」——CRT screen→oscilloscope、fountain→geyser、palace→castle。

## 二、找到错误之后：两种排序打分 [06:20 - 13:19]

拿到「标成 cow、其实像 fox」的 12 个候选，要决定先扔、先修哪一个。

![关于标签错误排序方法的幻灯片，列出了self-confidence和Normalized margin两种指标](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0040_00-06-24.jpg)

图注：对应本段「两种排序方法概览」——幻灯片列出 self-confidence 与 Normalized margin。

**自置信度（self-confidence）** 看模型给给定标签的概率 $\text{score}_{\text{self}}(x)=\hat p(\tilde y=\text{given label of }x;\boldsymbol x,\boldsymbol\theta)$，按升序排，最小的最可疑；一千万条候选时可先处理前一百再重跑。**最大边距 / 归一化边距（max margin / normalized margin）** 再减去「除给定标签外最像那个类」的概率：

$$\text{score}_{\text{margin}}(x)=\hat p(\tilde y=\text{given label};\,x,\theta)-\max_{j\ne \text{given label}}\hat p(\tilde y=j;\,x,\theta)$$

单标签多分类假设下，左边越低、右边越高，差值越负越像错误，取值从 1（不像错标）到 $-1$（几乎确定错标）。

![黑板上完整写出Self-confidence与Margin两类排序误差的公式定义](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0062_00-10-00.jpg)

图注：对应本段「两种排序误差的公式定义」——黑板并列写出 Self-confidence 与 Margin。

讲师点出全课程最难的一跃是「语义 vs 语法」：难的是把符号对应回现实里在数什么。

## 三、从计数矩阵派生算法：错误总数与三种剪枝策略 [13:19 - 24:10]

算法三步：计算计数矩阵 $C$、取迹 $\mathrm{Tr}(C)$ 作标对样本数、记录总样本数 $|X|$。

![黑板上展示的算法步骤：计算C、Tr(C)为good stat以及total # examples in data](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0093_00-15-06.jpg)

图注：对应本段「算法前三步」——计算 $C$、取迹 $\mathrm{Tr}(C)$、定义 $|X|$。

归一化数学上应做，但讲师提醒真实模型往往略微高估、不归一化反而更接近真值，此步可选。演示仍用 dog/fox/cow 那组数值。

![投影幕上展示混淆矩阵表格，行列为dog/fox/cow的预测与真实标签组合及对应数量](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0096_00-15-36.jpg)

图注：对应本段「演示用的混淆矩阵数值」——dog/fox/cow 计数表是错误总数与剪枝的输入。

用总数减去迹，无需人工设阈值就得到错误总数，也回答了 L2「按损失排序不知在哪截断」：

$$\#\text{errors}=|X|-\mathrm{Tr}(C)$$

![黑板上推导算法步骤，包含计算C矩阵及基于迹的误差公式](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0097_00-15-48.jpg)

图注：对应本段「错误总数推导」——$|X|-\mathrm{Tr}(C)$ 即错误数量，解决截断难题。

据此可现推多种找错策略（讲师强调不止三种）：**按类剪枝（prune by class）** 逐行取「给定标签概率」最低的若干样本；

![黑板上展示完整的样本选择策略：取置信度最低的50个样本](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0110_00-17-54.jpg)

图注：对应本段「按类剪枝的操作」——取给定类概率最低的 50 个样本作错误候选。

**按噪声率剪枝（prune by noise rate）** 聚焦某个非对角格（如「标 cow、真类 dog」的 32 个），取该类里「属于 dog 概率」最大的样本；**类 + 噪声率** 则把两者布尔 AND。

![黑板最终版板书，包含三种错误检测选项及PEC、PBNR、C+NR等缩写标注](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0145_00-23-42.jpg)

图注：对应本段「三种剪枝策略的命名」——板书标注 PEC、PBNR、C+NR 三个缩写。

计数可能受模型误差影响（本该 40 却数成 56，属认知不确定性 epistemic uncertainty），稳妥做法是只取排序靠前的 50%/5% 再迭代；排序函数 × 选择策略 × 取用比例即一组可调超参数。课堂上还确认：对「任何阈值都没超过」的离群样本做相似性检索对离群点检测有效，对标签错误检测则不然（相似度不考虑标签）。

## 四、混淆矩阵还能反过来改类别设计 [24:10 - 28:35]

若估计出的混淆矩阵里 fox 行有 50 张其实是 dog，说明两类高度混淆——讲师说在「这屋里人人都听过的公司」的数据集里多次见过。追问：还在定义类别、尚未训模型时能做什么？答案是**合并类别**。

![黑板上展示的3x3分类混淆矩阵及取样本说明](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0154_00-25-06.jpg)

图注：对应本段「混淆矩阵数值含义」——3×3 矩阵里 50 表示被误标成 Fox 的 Dog。

如「小猫 / 中猫 / 大猫」，业务只关心是否小猫时把中猫大猫合并即可消掉大面积混淆。关键是现在有了系统化办法：找非对角数值最大的类对。类别数一多非做不可——ImageNet 1000 类的联合矩阵有一百万个元素，靠人眼无法判断「一百万减一千对里先看哪一对」。

![展示标签 y=fox 和 ŷ=cow 对应数量的幻灯片页面](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0167_00-27-18.jpg)

图注：对应本段「大规模类别的相似类对定位」——$y=\text{fox}$、$\hat y=\text{cow}$ 计数说明千类矩阵的统计估计。

幻灯片上那几种方法在 CIFAR-10 的 40% 标签噪声下表现整体接近、排名随数据集变化，正呼应「用测试集挑超参数」。

## 五、理论：为什么「数数」比「加权」稳健 [28:35 - 45:04]

课堂投票选生成式 AI 略多于理论，讲师用约五分钟讲要点。对照组是**损失重加权（loss reweighting）**：把模型输出的某个函数乘进损失，SGD 对不确信样本少更新：

$$L'(\theta)\sim g\big(\hat p(\tilde y;\boldsymbol x,\boldsymbol\theta)\big)\cdot L(\theta)$$

但 $\hat p=p^*+\text{error}$，误差随之进入每次梯度更新与权重。「follow the error」正是 CL 三原则的设计动机。

![CL鲁棒性直觉1：剪枝，展示修改损失函数公式及误差传播标注](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0189_00-30-54.jpg)

图注：对应本段「损失重加权的误差传播」——$L'(\theta)\sim g(\hat p)\cdot L(\theta)$ 及剪枝鲁棒性标注。

讲义把 CL 归为 **Prune**（移除错误再训练）、**Count**（用计数之比而非平均一摞带误差概率）、**Rank**（用次序而非数值）。别人估计噪声转移常写成期望形式：

$$p(y^*=j\mid\tilde y=i)\approx\mathbb E\big[\hat p(\hat y=j\mid x\in X_i)\big]$$

![CL鲁棒性直觉2：计数与排序，给出噪声转移估计的期望形式公式](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0215_00-35-12.jpg)

图注：对应本段「噪声转移的期望估计」——$p(y^*=j\mid\tilde y=i)\approx\mathbb E[\cdots]$ 是 CL 对照的出发点。

CL 用贝叶斯规则化成可数形式，只要求概率「够大」越过阈值、不关心多大，其载体是置信集合：

$$\hat X_{\tilde y=i,y^*=j}=\{x\in X_{\tilde y=i}:\hat p(\tilde y=j;x,\theta)\ge t_j\}$$

![置信学习工作原理：基于概率阈值的样本筛选公式](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0240_00-39-18.jpg)

图注：对应本段「置信集合的阈值筛选」——$\hat X=\{x \mid \hat p\ge t_j\}$ 正是「只数是否越阈」的鲁棒性来源。

阈值容错很干净：理想概率 0.9、阈值 0.6 时，输出 0.9/0.71/0.8 都被正确计入，本不该计入的落在 0~0.7 也都不会计入，讲义写成允许约 33% 偏差——这是**鲁棒统计**（robust statistics）思路，也是把不完美概率与标签噪声拆开之处。理想概率下的阈值 $t_j$ 与计数集合已被证明精确找错，但神经网络存在系统性误校准。

![幻灯片：对误校准的鲁棒性，给出理想概率下精确查找标签错误的阈值公式t_j及神经网络过度自信的引文](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0265_00-43-24.jpg)

图注：对应本段「理想概率精确找错与过度自信」——$t_j$ 公式与 Guo et al., 2017 误校准文献。

Guo 等（ICML 2017）指出网络对某些类过度自信，可写成每类偏移 $\epsilon_j$，于是实际阈值变成：

$$t_j^{\epsilon_j}=\frac{1}{|X_{\tilde y=j}|}\sum_{x\in X_{\tilde y=j}}\big[\hat p(\tilde y=j;x,\theta)+\epsilon_j\big]=t_j+\epsilon_j$$

![幻灯片：在理想阈值基础上加入误差项ε_j，得到实际阈值t_j^{ε_j}=t_j+ε_j的推导式](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0266_00-43-36.jpg)

图注：对应本段「实际阈值含误差项」——$t_j^{\epsilon_j}=t_j+\epsilon_j$ 的推导。

代回计数集合：

$$C^{\epsilon_j}_{\tilde y=i,y^*=j}=\Big|\big\{x\in X_{\tilde y=i}:\hat p(\tilde y=j\mid x)+\epsilon_j\ge t_j+\epsilon_j\big\}\Big|$$

两边 $\epsilon_j$ 约掉，回到被证明精确的原式——即使每个样本都带类内一致的系统性偏差，CL 仍精确找错，这是本讲 punch line。

![幻灯片：用红框和箭头标注误差传播，推导含ε_j时集合C_{ỹ=i,y*=j}^{ε_j}的表达式并提问其变化](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0271_00-44-24.jpg)

图注：对应本段「误差抵消这一核心结论」——$\epsilon_j$ 在 $\ge$ 两侧约掉，回到精确原式。

讲义定理给出现实充分条件：类 $i$ 样本被标成 $i$ 的比例高于任何其他类、且允许所有输出有显著误差时，$\hat X\approxeq X$；实践中条件未必完全满足，但容差相当宽松。

## 六、测试集污染与生成式 AI 上的用法 [45:04 - 49:23]

先给刺眼结论：常用测试集里普遍存在标签错误，讲师收尾提醒「check your test set」，讲义统计平均约 3.4% 标签有误——传统做法默认测试集干净，于是一直在带噪测试集上比模型高低。

![章节过渡页：Confident Learning + Generative AI，列出将讨论 Image 与 Text 两类用例](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0277_00-45-24.jpg)

图注：对应本段「章节过渡」——列出 Image 与 Text 两类生成式 AI 用例。

生成式 AI 分「进」「出」两端：输入端是含错误、坏标签、离群点、数据漂移的海量数据；Falcon 等开源 LLM 用海量含错数据训练且没用过 CL，下载后又不可能为几十亿参数从头重训，改进主要落在推理侧。

![生成式AI模型的输入与输出流程图：含错误/坏标签/离群点/数据漂移的数据经训练得模型，推理再生成同类问题数据](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0278_00-45-36.jpg)

图注：对应本段「输入端问题」——含 Errors/Bad labels/Outliers/Data shift 的数据进入训练。

两端都能插入 CL：训练前清洗输入数据与标签，推理后再清洗输出。

![加入置信学习后的生成式AI流程：训练前与推理后均用 Confident Learning 改进数据与标签](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0281_00-46-06.jpg)

图注：对应本段「两端插入 CL」——训练前清洗数据、推理后清洗输出。

以 DALL·E 2（现并入 GPT-4）为例，训练集把飞机标成 car，用户输入 car 就得到飞机；理想做法是训练前用 CL 清掉图文错配。

![图像生成可靠性示例：DALL-E 因错误标注（飞机标为car）导致用户请求car时生成飞机](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0283_00-46-24.jpg)

图注：对应本段「错标导致生成错误」——DALL·E 把飞机标成 car，请求 car 却生成飞机。

改进输出的关键是 work backwards（反向工作）：把生成结果当作带标签数据集来清洗。

![改进生成式AI输出的关键是 work backwards，引出猫/狗生成数据集的可靠性改进示例](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0284_00-46-36.jpg)

图注：对应本段「work backwards 策略」——反向处理生成数据以改进输出。

具体让 GPT-4 生成 1000 张猫、1000 张狗，把 2000 张连同生成时标签交给 CL 自动移除或修正错误。

![生成后改进数据集的四个步骤列表](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0286_00-46-54.jpg)

图注：对应本段「生成后改进四步骤」——生成 1000 猫/1000 狗、跑 CL、自动修正。

猫狗太简单，讲师换成让模型画「离子阱量子计算机」——生成图与 MIT Ike Chuang 实验室真机相似却是另一种机器，这类专业题材才真需要清洗。

![AI生成与真实离子阱量子计算机的对比图](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0290_00-47-30.jpg)

图注：对应本段「专业题材的生成偏差」——AI 生成与 MIT 真实离子阱量子计算机对比。

文本侧两用例：**RAG（Retrieval-Augmented Generation，检索增强生成）** 把自然语言查询转成数据库查询、返回标签（如问贷款比例最高的州，答 Kentucky），把成千上万次查询-标签对跑 CL，既改标签又给输出加置信分数；**TLM（Trustworthy Language Model）** 则给 LLM 每个输出配质量/置信分数。演示里 ChatGPT 被问「enter 里有几个 n」答「两个」——既错又无置信度，而我们希望两者都有（讲义引 Chen & Mueller, 2023）。

![ChatGPT回答单词'enter'中N的数量演示](/coursenotes/mit-dcai-iap-2024/l03-advanced-confident-learning-llm/cand_0298_00-48-54.jpg)

图注：对应本段「LLM 缺置信分数」——ChatGPT 答 'enter' 有两个 N 的演示。

## 要点回顾

- 三讲递进：L1 建观念、L2 给算法、L3 答「为什么成立」并展示应用；CL 对输出尺度不变，logits 亦可。
- 算法回顾：按类阈值 $t_j$、置信集合、计数矩阵 $C$；行归一化后除以总数得联合分布 $\hat p(\tilde y,y^*)$，可推出翻转率、逆翻转率与真标签先验。
- 两种错误排序：self-confidence 越低越可疑；归一化边距越负越可疑，取值 1 到 $-1$。
- 由计数矩阵派生：$\mathrm{Tr}(C)$ 是标对数，$|X|-\mathrm{Tr}(C)$ 自动给出错误总数；剪枝有按类、按噪声率、两者布尔 AND，配合排序与取用比例成为可调超参数。
- 混淆矩阵还能反查类别设计：非对角数值最大的类对最该合并；1000 类对应一百万个矩阵元素，须靠系统化方法定位。
- 三原则 prune / count / rank 的共同点是**不让模型误差传播**：加权把 $\hat p=p^*+\text{error}$ 写进每次 SGD，计数只要求概率「够大」，容差约 33%；类内一致的 $\epsilon_j$ 在阈值两边抵消。
- 定理（现实充分条件）：类内标对比例占优、允许输出有显著误差时，CL 仍精确找错并一致估计联合分布。
- 测试集平均约 3.4% 标签有误；生成式 AI 上训练前清洗「进」、生成后清洗「出」，无法重训开源 LLM 时改进落在推理侧。
- RAG 用 CL 校正查询标签并给出置信度，TLM 则给 LLM 每个输出配质量分数。

## 视频来源

https://www.youtube.com/watch?v=hjbpRjr10Mc
