---
title: "第 14 讲 · 数据 II：转换、去重与训练管道"
description: "本讲把语言模型的数据工作拆成一条管道：转换、过滤、去重、混合。原始网页和 PDF 并不是干净文本，转换本质上是有损的工程过程；过滤则是用少量“好数据”定义目标，再训练轻量分类器去海量原始数据中筛选。去重的关键挑战是把两两…"
---

# 第 14 讲 · 数据 II：转换、去重与训练管道

> **Stanford CS336 语言模型从零构建 · L14** ｜ 讲者 Stanford Online ｜ 85 分钟 ｜ [原视频](https://www.youtube.com/watch?v=5sxHosTLPF8)

## 核心结论

本讲把语言模型的数据工作拆成一条管道：**转换、过滤、去重、混合**。原始网页和 PDF 并不是干净文本，转换本质上是有损的工程过程；过滤则是用少量“好数据”定义目标，再训练轻量分类器去海量原始数据中筛选。去重的关键挑战是把两两比较变成线性时间，实践上依赖 **MinHash + LSH**。数据混合要特别小心小规模高质量源被过度 epoch，回归式代理实验和模拟 epoch 是常见解法。最后，后训练/中期训练数据越来越依赖合成数据和代码 agent 轨迹。

---

## 1. 课程定位：从“数据从哪来”到“数据管道” [00:05 - 01:14]

上节课讨论的是数据来源与获取：互联网上的 live services 需要被 dump 或 crawl，再经过处理才能变成数据；同时还要考虑服务条款、版权许可与 fair use 等社会性因素。

本节课转向预训练数据管道的几个核心步骤：**transformation、filtering、deduplication、data mixing**；最后再讨论中期训练和后训练中的合成数据。大纲明确把本讲分成“数据管道”和“Mid-training + SFT: synthetic data”两部分。

![Lecture 14数据课程大纲，对比上节课数据获取流程与本节课数据管道及训练内容](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0001_00-00-16.jpg)

---

## 2. 数据转换：把 HTML/PDF 变成可训练文本 [01:17 - 06:38]

原始数据通常不是文本。以 Common Crawl 为例，里面是 HTML；此外还可能有 PDF、GitHub 目录等。转换的主要工作量在 HTML，因为大部分网页是 HTML。

HTML 处理通常是规则驱动的：去掉导航、广告、页眉页脚等 boilerplate，抽取正文内容。但“什么算正文”并不总是清楚，导航元素有时也可能包含有用信息；图片和表格也需要处理。把 HTML 线性化为 token 序列本质上是有损的，因为 HTML 既有层级结构，又有渲染后的视觉布局。简单表格可以用 Markdown 表示，嵌套表格则很难处理，往往只能近似或放弃。

规则方法流行是因为**快**，而且这一步不需要太多“智能”；模型方法也可能有用，但必须非常快。不同文本抽取工具在评测集上的准确率并不相同，图中对比了 resiliparse、trafilatura、WET files 等在 CORE/EXTENDED 上的表现。

![不同文本提取工具在CORE和EXTENDED数据集上的准确率对比表](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0002_00-04-14.jpg)

PDF 是另一类难点。PDF 源码主要描述绝对定位和“绘制”命令，缺少 `<p>`、`<h1>` 这类语义标签；阅读顺序不保证，文本也可能碎片化，布局只能从几何信息中启发式推断。Common Crawl 中的 PDF 有时还会被截断，需要重新抓取；扫描版 PDF 需要 OCR，通常要用 VLM，成本明显更高。PDF 在整个互联网中占比很小，但平均质量往往高于普通网页，因为愿意制作 PDF 的发布者通常有更正式的内容。图中展示了 PDF 源码结构与渲染后视觉布局之间的映射困难。

![PDF源码结构与渲染后视觉布局的映射关系图，展示了解析PDF的难点](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0003_00-05-39.jpg)

---

## 3. 数据过滤：用目标数据定义“好数据” [06:43 - 21:53]

### 3.1 通用框架与两类分类器

过滤的抽象骨架是：给定一小部分高质量目标数据 $T$，以及大量原始数据 $R$，目标是找到 $R$ 的一个子集 $T'$，使其与 $T$ 相似。几乎所有过滤方法都可以放进这个框架。

![数据过滤算法的定义幻灯片，包含目标数据与原始数据的集合关系图](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0004_00-07-04.jpg)

过滤的典型应用包括：

- **语言识别**：只保留目标语言；
- **质量过滤**：保留百科式、高质量内容，去掉垃圾或低质内容；
- **毒性过滤**：避免训练数据中的有害内容。

对过滤算法的要求是：既能从目标数据泛化到未见数据，又要极快，因为要在整个互联网规模上运行，可能是 100 万亿 token；最终保留比例通常只有个位数百分比。

![数据过滤的应用场景与算法设计要求幻灯片](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0005_00-08-29.jpg)

常见做法有两类：

1. **生成模型**：在目标数据 $T$ 上估计模型，用 $p_T(x)$ 给文档打分，例如 KenLM 这类 n-gram 模型；
2. **简单分类器**：把 $T$ 当作正例，从 $R$ 中采样负例，训练 fastText 这类线性 bag-of-words 分类器，用 $p(T \mid x)$ 打分，再设阈值保留。

![数据选择通用框架与分类器类型（生成模型与简单分类器）的幻灯片](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0006_00-09-53.jpg)

现在模型过滤基本成为常态：除非算力极其充足、可以直接在全部数据上训练，否则大多数团队都需要通过过滤避免把 FLOPs 浪费在低质量内容上。

### 3.2 实例：语言识别、数学数据、质量与毒性

语言识别可以用现成的 fastText 模型，支持 176 种语言，训练数据来自 Wikipedia、Tatoeba 翻译站、SETimes 新闻等。语言 ID 相对简单，不是训练好模型的主要瓶颈；但 code-switching 和方言仍有细节。Dolma 的做法是保留 $p(\text{English}) \ge 0.5$ 的页面。

OpenMathText 的目标是从 Common Crawl 中构造数学文本语料。它不是只训练一个分类器，而是组合多种信号：用规则检查是否包含 LaTeX 命令；用在 ProofPile 上训练的 KenLM 计算困惑度，保留困惑度低于阈值的文档；再训练 fastText 分类器判断是否为数学写作，并对“包含数学”和“不包含数学”的情况使用不同阈值。最终产出约 14.7B token，用其训练的 1.4B 模型在数学任务上优于用 20 倍未过滤数据训练的模型。

![语言识别与OpenMathText数据筛选规则的幻灯片](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0007_00-12-43.jpg)

![数据过滤方法幻灯片，涵盖基于模型的过滤、语言识别及OpenMathText数学文本筛选流程](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0008_00-14-08.jpg)

GPT-3 的过滤把 Wikipedia、web text、书籍等作为正例，从网页中采样负例，训练线性分类器并保留高分文档。LLaMA/RedPajama 的正例是 Wikipedia 引用的页面，而不是 Wikipedia 文章本身。

phi-1 的思路更激进：原始数据 $R$ 是 The Stack 的 Python 子集；定义 prompt 让 GPT-4 判断“对学习基础编程的学生是否有教育价值”，用 GPT-4 标注 10 万条子集得到目标数据 $T$；再基于预训练 codegen 模型的输出嵌入训练随机森林分类器，从 $R$ 中筛选正例。结果在 HumanEval 上，1.3B 模型用新过滤子集训练 36K 步达到 17.68%，优于用原始子集训练 96K 步的 12.19%。

![数据过滤代码示例及LLaMA/RedPajama、phi-1模型的数据筛选策略](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0009_00-15-32.jpg)

毒性过滤也遵循同样框架。Jigsaw Toxic Comments 数据集来自 Wikipedia talk page 评论，标注了 toxic、severe_toxic、obscene、threat、insult、identity_hate 等标签；可以据此构造正负例并训练分类器。

![phi-1数据筛选结果及Dolma毒性过滤数据集介绍](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0010_00-16-57.jpg)

### 3.3 阈值依赖训练规模

过滤没有“唯一最优阈值”。直觉上：

- 如果训练 token 数很多，可以容忍更多低质量数据；
- 如果训练 token 数较少，则更应保留高质量数据。

一项初步实验使用 157M 参数模型和 100 个 WARC 的小数据池，对比不同过滤/抽取配置。高质量数据在早期 loss 更低，但数据量小，继续训练后不得不重复 epoch，最终可能过拟合；低质量或未充分过滤的数据早期较差，但在大量 token 的长训练区间仍可能继续下降。图中可以看到不同配置的 loss 曲线以及各自的 1-epoch 标记。

![Dolma 毒性过滤方法介绍及不同训练时长下的数据质量选择策略](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0011_00-18-22.jpg)

![不同数据过滤方法在 157M 参数模型上的训练损失对比曲线图](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0012_00-19-47.jpg)

因此过滤的实用 recipe 是：先定义“好数据”长什么样，可以来自现有高质量数据集，也可以用 LLM 对大规模池子做预筛；再用这些数据训练一个便宜分类器，外推到整个原始数据池。

---

## 4. 去重：从精确匹配到 MinHash LSH [22:55 - 48:55]

### 4.1 为什么去重、去重什么

数据中常见两类重复：

- **精确重复**：镜像站、GitHub fork、同一文件被多次抓取；
- **近似重复**：同一文本只差几个 token，例如许可证、页眉页脚、复制粘贴造成的标点差异，或模板化生成的内容。

![关于数据集中的精确重复与近似重复（Near duplicates）的示例说明幻灯片](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0014_00-25-26.jpg)

C4 数据集中曾发现某段产品描述重复出现 61,036 次。去重的好处包括：减少 token 数以更高效训练；避免模型记忆重复内容，从而缓解版权和隐私问题；去重也有助于去污染（decontamination），确保测试集不出现在训练集中。

![关于训练数据去重（Deduplication）的幻灯片，展示了重复的产品描述文本及去重的好处](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0015_00-26-51.jpg)

去重的设计空间包括：

1. **item 粒度**：句子、段落还是文档；
2. **匹配方式**：精确匹配、是否存在公共子项、公共子项比例；
3. **处理动作**：删除所有实例，还是保留一个。

关键算法挑战是：过滤可以逐条并行判断，而去重本质上是 item 之间的比较，不能做 $O(n^2)$ 的全量两两比较，必须设计线性时间算法。

![去重训练数据的动机、设计空间与核心挑战，强调去重本质是两两比较](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0016_00-28-15.jpg)

### 4.2 精确去重

精确去重很直接：对 item 做哈希，相同哈希的分为一组，只保留一个。代码可以写成 MapReduce 风格，便于并行扩展：

```python
hash_items = itertools.groupby(sorted(items, key=mmh3.hash), key=mmh3.hash)
deduped_items = [next(group) for h, group in hash_items]
```

C4 的精确去重以**三句跨度**为 item，精确匹配后“保留一个”。这有个副作用：从文档中间删掉三句跨度可能破坏文档连贯性。

![精确去重的 MapReduce 式代码实现及去重前后结果对比，并引出 C4 三句跨度去重](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0017_00-31-05.jpg)

### 4.3 近似去重与 Jaccard 相似度

近似重复需要定义“近似匹配”。常用指标是 Jaccard 相似度：

$$
\operatorname{Jaccard}(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

例如 $A=\{1,2,3,4\}$、$B=\{1,2,3,5\}$，交集大小为 3，并集大小为 5，Jaccard 为 0.6。可以定义两个文档在 Jaccard 相似度超过某个阈值（例如 0.99）时互为近似重复。问题仍是如何在线性时间内找到这些近似重复。

![Jaccard相似度定义、Python实现代码及近似重复文档判定标准](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0018_00-32-30.jpg)

### 4.4 MinHash：让碰撞概率等于 Jaccard

MinHash 的思路是：对集合中每个元素用随机哈希函数散列，取最小哈希值作为集合的签名。性质是：对随机选择的哈希函数，

$$
\Pr[\text{MinHash}(A)=\text{MinHash}(B)]=\operatorname{Jaccard}(A,B).
$$

直观理解：随机哈希函数等价于对 item 做随机排列，看 $A$ 和 $B$ 中哪个 item 排在最前。若排在最前的是交集元素 1、2、3，则两个集合的最小哈希相同；若是只在 $A$ 或只在 $B$ 中的 4、5，则不同。因此碰撞概率正好等于交集占并集的比例。

```python
def minhash(S: set[str], seed: int):
    return min(mmh3.hash(x, seed) for x in S)
```

实践中生成多个随机哈希函数，统计 MinHash 相同的比例来估计 Jaccard；代码验证会得到接近 0.6 的估计值。

![特征矩阵表示及MinHash近似Jaccard相似度的验证代码](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0019_00-33-54.jpg)

![MinHash函数定义、特征矩阵表示及最小哈希值概率分析](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0020_00-35-19.jpg)

![MinHash 近似 Jaccard 相似度的验证代码及原理说明](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0021_00-36-44.jpg)

### 4.5 LSH 分带：把随机碰撞锐化成阈值

MinHash 只保证碰撞概率等于 Jaccard，但方差较大，不足以直接判断“Jaccard 是否大于 0.99”。局部敏感哈希（LSH）通过分带锐化概率：

- 使用 $n$ 个哈希函数；
- 分成 $b$ 个 band，每个 band 含 $r$ 个哈希函数，$n=b\cdot r$；
- 只要某个 band 内所有哈希函数都相等，就认为 $A$、$B$ 碰撞。

![LSH 分带策略：将 n 个哈希函数分为 b 个带以锐化碰撞阈值](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0022_00-39-33.jpg)

固定 band 匹配的概率是 $s^r$，其中 $s$ 是相似度；至少一个 band 匹配的概率是：

$$
P_{\text{collision}}(s)=1-\left(1-s^r\right)^b.
$$

```python
def get_prob_collision(sim, b, r):
    prob_match = sim ** r
    prob_collision = 1 - (1 - prob_match) ** b
    return prob_collision
```

![MinHash 分带策略的代码实现，展示 n=12、b=3、r=4 的参数设置及碰撞概率计算函数](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0023_00-40-58.jpg)

![MinHash 碰撞概率计算示例，展示 sim=0.8、b=5、r=10 时的参数代入与函数调用](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0024_00-42-23.jpg)

这条曲线呈 S 形：相似度低时碰撞概率接近 0，相似度高时接近 1。增大 $r$ 会锐化阈值并把曲线右移，使匹配更难；增大 $b$ 会把曲线左移，使匹配更容易。通过调节 $b$ 和 $r$，可以把相变做得足够陡，但计算成本也会上升。

![MinHash 碰撞概率随相似度变化的 S 型曲线图，展示分带策略如何锐化阈值](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0025_00-43-48.jpg)

![不同 b 值下碰撞概率随相似度变化的 S 型曲线对比图](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0026_00-45-12.jpg)

实际去重论文中的设置可能是 $n=9000$、$b=20$、$r=450$。相变阈值近似为：

$$
\text{threshold}=\left(\frac{1}{b}\right)^{1/r}.
$$

![不同 b 值曲线对比及相变阈值计算公式](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0027_00-48-02.jpg)

### 4.6 实践注意

去重不仅要处理单个数据集内部的重复，还应考虑**跨整个训练数据集的重复**，因为不同数据源之间也可能高度冗余。

---

## 5. 数据混合：给多个来源分配权重 [49:19 - 72:57]

### 5.1 问题与基线

语言模型通常同时训练在多个来源上：网页、代码、书籍、数学数据、PDF 等。数据混合就是一个在所有来源上的概率分布。The Pile 给每个组件分配权重，并记录原始大小、权重、epoch 数和有效大小。

![The Pile 数据集的组成成分、原始大小、权重、训练轮数及有效大小统计表](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0028_00-50-52.jpg)

常见基线包括：

- **凭直觉手动设置**：根据经验直接给权重，实际上很常见；
- **均匀采样**：每个来源等概率；
- **按 token 比例混合**：$p(s)\propto \text{num\_tokens}(s)$。

直觉上应该给高质量来源更高权重，但有两个约束：一是要保证多样性，文学、代码、论文等来源不可直接比较；二是每个来源是有限的，给小来源过高权重会导致反复 epoch。

![数据配比基线方法及直觉解释幻灯片，包含Vibes、均匀采样、比例混合及多样性/有限性考量](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0029_00-53-41.jpg)

### 5.2 源有限性与 epoch 陷阱

假设低质量源有 10T token，高质量源只有 10B token；如果天真地设 $p(\text{low})=0.5$、$p(\text{high})=0.5$，训练 1T token，则：

- 低质量源请求 0.5T token，只覆盖其 10T 的 5%，每个点最多训练一次；
- 高质量源请求 0.5T token，但只有 10B token，相当于每个点重复 **50 个 epoch**。

这会浪费算力，甚至导致过拟合。关键教训是：定义混合比例后，要检查每个源实际被训练了多少个 epoch。

![数据配比示例代码幻灯片，展示低资源数据过度训练（50x epochs）导致过拟合的计算过程](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0030_00-55-06.jpg)

UniMax 一类方法提出：按来源均匀采样，但对每个源的 epoch 数设硬上限。约束可写成：

$$
p(s)\cdot \text{train\_tokens}\le \text{cap}\cdot \text{source\_tokens}(s).
$$

这样小来源不会因为权重过高而被反复训练。

![展示数据混合比例计算代码及UniMax方法介绍](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0031_00-58-29.jpg)

### 5.3 回归式混合

更系统的方法是用小规模代理实验拟合“混合比例 → loss”的函数：

1. 训练一批小规模代理模型，尝试不同数据混合；
2. 用这些实验结果拟合回归模型，输入是混合权重，输出是目标指标；
3. 在回归模型上优化，得到候选最优混合；
4. 用该混合训练大规模模型。

![基于回归的数据混合方法流程图，包含代理模型训练、回归拟合及模拟预测](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0032_01-00-45.jpg)

设计决策包括：如何采样混合分布（常用 Dirichlet 分布）、回归模型形式（线性模型、boosted decision trees、log-linear 等）、优化目标（下游 eval 或 perplexity）。这里有两个需要警惕的“信念跳跃”：

- 优化时可能走到训练分布覆盖不足的极端区域，回归预测未必可靠；
- 小规模代理上的最优混合不一定能完全迁移到大模型。

此外，如果目标选了下游 eval，要小心过拟合：例如代码 eval 很多，优化器自然会偏向代码数据，但这不一定带来通用能力。

### 5.4 尺度依赖与模拟 epoch

过滤部分已经看到尺度依赖：训练 token 越多，低质量数据越可接受。混合也有类似问题：小规模代理训练 token 少，可能不会 epoch，于是会把大量权重放在 Wikipedia 这类高质量小源上；但放大到大模型、大 token 数后，这个源会被严重 epoch，导致过拟合。

解决办法之一是 **模拟 epoch**：让小规模实验看起来像大规模训练。例如小实验训练 10B token、大实验训练 1T token，就把各来源按 1/100 下采样后再做混合优化。这样代理实验也会感受到高质量源的数据稀缺，不会盲目给它过高权重。另一种办法是直接像 UniMax 那样给 epoch 数设上限。

数据混合也可以应用到单个数据源内部：例如把 Common Crawl 按 domain 和 quality 分成二维网格，每个格子作为一个混合单元，再叠加外部来源。

---

## 6. 后训练/中期训练数据：合成数据与代码 agent 轨迹 [72:57 - 83:41]

前面的预训练数据基本是任务无关的，目标是学习基础能力；后训练数据则高度任务相关。当前开放社区的后训练数据很大比例是**合成生成**的。

一般 recipe 是：

1. 定义环境，例如 GitHub 仓库；
2. 定义任务或 prompt；
3. 从强模型/教师模型收集回答，或使用人类标注；
4. 对回答进行过滤和筛选。

人类标注慢且昂贵；前沿实验室也会采用人类与 AI 混合的方式。

OpenThoughts 是一个面向推理数据的例子，最终包含约 120 万条示例。它的来源包括人类数据（如 Stack Exchange）和合成数据，覆盖数学、科学、代码等。经验包括：多采样几次生成（例如 16 次）有帮助；更强的模型不一定是最好的教师，例如 QwQ 32B 在某项对比中比 DeepSeek R1 更适合作为教师；对答案做简单过滤没有明显帮助。120 万条示例除以 16 次生成，对应实际题目数。

代码 agent 数据更复杂。SWE-Smith 的思路是：给定仓库，用 agent 自动安装依赖、使其可运行；然后通过修改代码、引入 bug 等方式自动生成任务，并做验证，得到约 5 万个合成任务。这比纯文本代码生成更接近软件工程。

SWE-zero 观察到代码任务有很重的依赖问题：很多 GitHub 仓库本身跑不起来，回滚到某个 PR 时依赖更是噩梦。但他们发现模型已经足够强，即使没有执行反馈也能解决不少任务：允许执行时约 80%，不允许执行时接近 70%。因此可以构造大量**不需要执行反馈**的 agent 轨迹，规模约 30 万条，来自真实 GitHub PR。实现上使用 OpenHands scaffold，并限制 agent 只能做基本操作，防止它绕过限制去执行代码；再从大代码模型蒸馏并过滤。另有约 1.3 万条需要执行反馈的轨迹，可以先在无执行反馈数据上微调，再在这部分数据上继续微调。

后续工作把 SWE-zero 思路扩展到约 1200 万条 agent 轨迹；虽然其中只有一部分能成功执行，但 SWE-zero 式数据不依赖执行，因此仍然可用。总体趋势是：后训练数据从数学这类“无环境”任务，走向代码这种带真实环境、可验证、可自动生成任务的领域；任务可以是全合成、半合成（真实环境 + 合成任务）或真实任务；回答通常来自强模型，但“强”不等于“好教师”；代码环境的搭建和过滤仍然非常繁琐。

---

## 总结与边界 [83:45 - 84:35]

本讲覆盖了数据管道的四个环节：

- **过滤**：定义“好数据”，训练轻量分类器，从网页抓取中筛出小子集；
- **去重**：避免过拟合、节省 FLOPs，并做去污染；
- **混合**：在小规模上尝试混合，再外推到大规模，同时小心 epoch 和过拟合；
- **后训练数据**：合成数据、教师模型、代码 agent 轨迹。

讲者也强调，真实的数据工作往往很“脏”、非常领域特定，需要不断查看具体样例；这堂课只能给出数据领域的地形图，不能代表数据工作的全部细节。

---

## 要点回顾

- **数据转换**：HTML/PDF 不是纯文本；HTML 抽取依赖规则去 boilerplate、提正文，线性化有损；PDF 缺语义标签、阅读顺序不保证，扫描件需 OCR/VLM。
- **过滤框架**：给定目标数据 $T$ 和原始数据 $R$，找与 $T$ 相似的子集 $T'$；可用生成模型 $p_T(x)$ 或 fastText 分类器 $p(T\mid x)$。
- **过滤实例**：语言 ID、OpenMathText 的规则 + KenLM + fastText、GPT-3/LLaMA 的正负例分类、phi-1 用 GPT-4 标注后训练随机森林、Jigsaw 毒性分类。
- **阈值尺度依赖**：训练越久越能容忍低质量数据；没有单一最优过滤阈值。
- **去重设计空间**：item 粒度、匹配方式、删除策略；精确去重可用哈希分组，近似去重需线性时间算法。
- **Jaccard 与 MinHash**：Jaccard 衡量集合相似度；MinHash 使碰撞概率等于 Jaccard。
- **LSH 分带**：$b$ 个 band、每 band $r$ 个哈希；$P_{\text{collision}}=1-(1-s^r)^b$；阈值约 $(1/b)^{1/r}$；增大 $r$ 更难匹配，增大 $b$ 更易匹配。
- **混合基线**：手动、均匀、按 token 比例；要兼顾多样性和来源有限性。
- **epoch 陷阱**：小高质量源被赋予过高权重会被重复几十次，浪费算力并过拟合；UniMax 用 epoch 硬上限约束。
- **回归式混合**：小代理模型试混合 → 拟合混合到 loss 的回归 → 优化混合 → 训练大模型；注意外推可靠性和小到大迁移。
- **模拟 epoch**：按大/小训练 token 比例下采样各来源，让代理实验也感受到数据稀缺。
- **后训练数据**：大量合成；OpenThoughts 用多采样和教师模型；代码 agent 数据从 SWE-Smith 的合成任务到 SWE-zero 的真实 PR 无执行反馈轨迹，规模继续扩大。
