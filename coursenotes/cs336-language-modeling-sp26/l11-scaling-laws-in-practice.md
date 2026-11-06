---
title: "第 11 讲 · 缩放定律进阶：μP、WSD 与外推"
description: "经典 Chinchilla 之后，前沿 scaling 研究大致分成两条路线：一条用 μP 等参数化把最优学习率等超参数“稳定”到不随规模变化；另一条直接在小规模上网格搜索并拟合学习率、批大小的幂律，再外推到大规模。Mi…"
---

# 第 11 讲 · 缩放定律进阶：μP、WSD 与外推

> **Stanford CS336 语言模型从零构建 · L11** ｜ 讲者 Stanford Online ｜ 77 分钟 ｜ [原视频](https://www.youtube.com/watch?v=vTfEyOyzV9E)

> **v2 详写版** · 按 20 分钟分段详写，配图覆盖全片；文末附「对照参考笔记补遗（Spring 2025 版）」，记录与 2025 版讲义笔记的核对结果。

## 核心结论

这一讲把缩放定律从 2022 年 Chinchilla 铺到 2026 年的开源前沿，追问经典结论在真实规模上是否成立、拟合规律能否省算力、以及是否该挑特定参数化让模型更好放大。围绕「敏感超参数随规模漂移该消掉还是被建模」，课程对照两条路线：MiniCPM 用 μP 把最优学习率钉在百分之一附近，再用预热稳定衰减的学习率把等算力数据扫描成本从平方量压回线性；DeepSeek 不做参数化稳定，直接在小规模网格搜索上拟合学习率与批大小随算力的幂律再外推。随后一批开源工作把混合专家稀疏度、注意力架构选择、损失到下游精度的映射也纳入缩放分析，更细的网格却暴露各家公式连输入变量都不一致、换数据就漂移。后半段转向优化器与 μP：Muon 用矩阵乘迭代近似正交化动量，在小规模基准上显著加速却随规模递减增益；μP 从激活与激活变化两条不变量推出逐层初始化与学习率规则，在真实规模被验证，也列出会破坏它的几类组件。全讲收束于一份谨慎判断：这些拟合看似精密科学，外推却充满经验成分，缩放实践至今没有银弹。

## 课程动机：从经典缩放范式走向开源前沿实践 [00:00 - 03:00]

讲者开篇把这一讲定性为一次 "grab bag of stuff"（大杂烩式的专题合集），主题是 scaling laws（缩放定律）里更偏进阶、更偏工程实践的细节，也就是「真正动手把语言模型放大时，你必须知道的那些东西」。紧接着幻灯片 "Motivation today" 用一个总问题和三个 bullet 把本讲的问题意识钉死：总问题是 "What is the best practices for scaling and hparam tuning LMs?"（缩放与超参数调优的最佳实践到底是什么），下面三条依次是 "Does chinchilla's approach to scaling actually work?"（Chinchilla 那套缩放方法在真实场景下究竟管不管用）、"Can we save compute when training and fitting these things?"（训练模型和拟合这些规律时能不能省算力）、"Should we be picking particular architectures / parametrizations to scale nicely?"（是否应当刻意挑选某些架构或参数化，让模型更「好放大」）。这三个问题恰好对应本讲后续的三条主线：验证经典结论、降低拟合成本、用参数化换稳定性。讲者在口述里把第一问表述为「两讲前讲的 scaling law 在真实开源模型规模上是否真的成立」，把第三问落在「optimizers and optimizer behavior is going to be actually pretty scale sensitive」（优化器行为对规模相当敏感）这一判断上。

![课程动机幻灯片，列出缩放与超参数调优的一个总问题和三个具体追问](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0001_00-00-36.jpg)

这里的 "scale sensitive"（规模敏感性）是本段反复出现的关键词，值得先用符号把它固定下来。讲者点名了三类量：initializations（初始化）、learning rates（学习率）、batch size（批大小），并说这些量需要 "carefully think about as a function of scale"（当作规模的函数来仔细处理）。用记号写就是：

$$\eta^* = \eta^*(N,D),\qquad B^* = B^*(N,D),\qquad \text{init} = \text{init}(N)$$

其中 $N$ 表示模型参数量、$D$ 表示训练 token 数，$\eta^*$ 与 $B^*$ 分别是该规模下的最优学习率与最优批大小。必须强调，这只是把讲者的定性判断符号化，并不是某篇论文拟合出来的具体定律；本讲后面 MiniCPM 与 DeepSeek 两篇论文的分歧，恰恰在于「上面这三个函数该被消掉，还是该被直接拟合出来」。与之相关的另一个记号是 isoflops（等算力）分析：固定算力预算 $C$，在 $N$ 与 $D$ 之间做权衡，即 $C \approx \text{const}$ 时 $N \uparrow \Leftrightarrow D \downarrow$，这正是 Chinchilla 式分析的基本操作，转写中讲者会说成 "fix a flops budget and I'm going to vary my trade-offs between number of tokens and model size"。

![2022 年 Chinchilla 论文与 2024–2026 年开源模型缩放论文的对照清单](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0002_00-01-48.jpg)

幻灯片 "Scaling in practice" 给出了一条很具体的时间线，这些信息是转写没有逐条念出来、但画面上独有的。上方写着 "The newest model we talked about with scaling details - 2022"，配的是 DeepMind 的 "Training Compute-Optimal Large Language Models"（即 Chinchilla），作者栏从 Jordan Hoffmann、Sebastian Borgeaud、Arthur Mensch 一直列到 Oriol Vinyals、Laurent Sifre，并标注 "*Equal contributions"。下方 "What about more recently?" 用两列铺开了近年真正公开了缩放细节的工作：DeepSeek LLM 的 "Scaling Open-Source Language Models with Longtermism"（2024）、"Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent"（2024，署名 Tencent Hunyuan Team）、"MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies"（2024，清华大学计算机系与 ModelBest Inc.，作者含 Shengding Hu、Yuge Tu、Xu Han、Maosong Sun 等）、"MiniMax-01: Scaling Foundation Models with Lightning Attention"（2025）、"The Llama 3 Herd of Models"（2024，Llama Team, AI @ Meta）、以及 "Kimi K2: Open Agentic Intelligence"（2026，Kimi Team）。年份标签明确写着 2022、2024、2025、2026，其中 Hunyuan-Large 标题里的 "52 Billion Activated Parameters" 是画面上少见的硬性数字。讲者在口述里补充说，2022 年之后由真正训练大模型的人发表的缩放论文数量在减少，而且 "most of them have come out of the open source community in China"（多数出自中国开源社区），因此他从中 "curated and selected"（精选）了几篇，目标是带听众 "speed run your way from Chinchilla all the way to Kimi K2"（从 Chinchilla 一路速通到 Kimi K2），看看前沿开放社区到底在关心、担心什么。

在结构安排上，讲者把本讲切成两半：第一部分逐篇过论文、抠细节；第二部分讨论 optimizers（优化器）与 initializations（初始化），并特别回顾「两讲前花过不少时间讲 batch sizes 的选择」，因为 batch size 和 learning rate 都属于 "very sensitive, very important to get right"（极其敏感、必须调对）的量。他甚至说这些技巧 "may in fact even be useful in your day-to-day lives of training various language models"，即能指导你随实验规模放大或缩小时如何改超参数。这一动机段落的局限也很清楚：讲者自己承认这是一次「大杂烩」，并不承诺给出统一体系；而且经典 scaling law canon（Kaplan、Hestness、Chinchilla）大约停在 2022 年，此后这类细节在论文里反而 "taken for granted"（被当成默认常识）而不再展开——这正是他坚持要回头精读 MiniCPM、DeepSeek 这类「老」论文的原因。换句话说，本段建立的是一个待验证的怀疑：经典结论是否可直接外推到真实开源规模，以及超参数的规模漂移究竟该「消掉」还是「建模」。

## MiniCPM 与 μP：把最优学习率钉在原地 [03:00 - 07:00]

幻灯片 "Working through some detailed, public scaling recipes" 用编号列表给出本讲要精读的两篇："1. MiniCPM" 与 "2. DeepSeek"，旁边分别贴了 "MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies" 和 "DeepSeek LLM / Scaling Open-Source Language Models with Longtermism" 的论文题头。讲者解释选择标准：这两篇 "not the newest papers"，也未必最详尽（他顺口说 "DeepSeek one might be one of the more detailed ones"），但 "hits a sweet spot"——都是出自真正造出高性能模型团队之手的 "serious scaling investigations"（严肃的缩放研究），而且 "take pretty different approaches to some of the key questions"（在关键问题上采取相当不同的路线），正好用来给全讲立起对照。MiniCPM 是 2024 年的产物，讲者称它当时在 "1 to 2 billion parameter bracket"（10–20 亿参数档）基本是 state-of-the-art，如今 Gemma 等 2B/3B 模型会更强，但按 2024 年标准仍是 "competitive well-trained models"。配图 "MiniCPM" 页上还有两条转写没念全的 bullet："MiniCPM (2024) – small, high-perf LM from Tsinghua group."、"Careful, extensive scaling computations + muP to stabilize and simplify scaling"（细致而大量的缩放计算，加上用 μP 来稳定并简化缩放），以及 "Not really 'sota' model (even in 2024) but many interesting lessons on scaling"（即便在 2024 年也算不上最强模型，但关于缩放有很多有意思的经验）——这种「不追 SOTA、只追可复用的缩放方法论」的定位，正是它被选进课堂的理由。

![MiniCPM 的 μP 参数化操作清单 Table 7，含四项缩放常数与五条具体操作公式](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0005_00-05-25.jpg)

本段第一个技术重点是 μP。讲者把它归入 "a special class of initializations"（一类特殊的初始化方法），并直言 "initializations are very important"，因为好的初始化能让某些超参数 "less scale dependent"（不那么依赖规模）；μP 的全部目的就是 "make sure that my optimal learning rate is the same as I scale up and down"（确保放大、缩小模型时最优学习率保持不变）。幻灯片 "Technique 1: muP to stabilize scaling" 顶部先给了四个默认常数，这是画面独有的硬参数：

$$\text{Scale\_emb} = 12,\quad \text{scale\_depth} = 1.4,\quad \text{init\_std} = 0.1,\quad \text{lr} = 0.01$$

其下是 Table 7（图注为 "List of operations used when applying tensor program techniques."），两列分别为 Name 与 Specific Operation，共五行。逐条转述：Embedding Output Scaling 要求 "Multiply the output of the embedding by scale_emb"，即 $\mathbf{e} \leftarrow \text{scale\_emb}\cdot \mathbf{e}=12\,\mathbf{e}$；Residual Connection Scaling 要求 "Scale the increment at each residual connection in each layer by scale_depth / $\sqrt{\text{num\_layers}}$"，写成

$$\Delta h_\ell \leftarrow \frac{\text{scale\_depth}}{\sqrt{L}}\,\Delta h_\ell = \frac{1.4}{\sqrt{L}}\,\Delta h_\ell$$

（$L$ 为层数），这正对应转写里那句 "scale the residual connections by the square root of number of layers"；Initialization of Tensors 要求 "Set the initialization standard deviation of each two-dimensional tensor parameter to init_std / $\sqrt{d_m/d_{base}}$, and set other parameters' initialization to 0.1"，即

$$\sigma_{\text{init}} = \frac{\text{init\_std}}{\sqrt{d_m/d_{\text{base}}}} = \frac{0.1}{\sqrt{d_m/d_{\text{base}}}}$$

其中 $d_m$ 是模型宽度、$d_{\text{base}}$ 是基准宽度——这就是转写说的 "scale the initialization of all their matrix shape tensors by the fan in fan out ratio"；Learning Rate Scaling of Tensors 要求 "Adjust the learning rate of each two-dimensional tensor parameter to $1/(d_m/d_{base})$ times the learning rate of other parts (or the overall learning rate)"，即

$$\eta_{\text{tensor}} = \frac{1}{d_m/d_{\text{base}}}\,\eta_{\text{global}}$$

讲者特意提醒 "you should pay attention to this one cuz this is exotic if you aren't used to like per parameter learning rates"（不习惯逐参数学习率的人会觉得这条很反常）；最后 LM Head Scaling 要求 "Adjust the output logits to $1/(d_m/d_{base})$ times the original value"，即 $\text{logits} \leftarrow \text{logits}/(d_m/d_{\text{base}})$，对应转写说的 "they also scale the LM heads"。转写里讲者说会在后面推导这组缩放的由来，本段只要求先接受它：改一点初始化、改一点逐参数学习率，换来学习率在跨规模时更稳。

![MiniCPM 的缩放阶梯表格，列出 9M 到 0.5B 七个模型的 N (B)、d_m、d_ff、d_h、n_h、L](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0006_00-06-37.jpg)

光有 μP 还不够，还得有能外推的「缩放阶梯」。幻灯片 "Scaling recipe / strategy" 的副标题把策略讲得很直白："Use muP for initialization, fix the aspect ratio, scale up the overall model size."（用 μP 做初始化，固定长宽比，再整体放大模型），表格列头为 Name、N (B)、$d_m$、$d_{ff}$、$d_h$、$n_h$、$L$。七行数值依次是：9M（$N=0.009$，$d_m=320$，$d_{ff}=800$，$d_h=64$，$n_h=5$，$L=8$）、30M（$0.036$，$512$，$1280$，$64$，$8$，$12$）、70M（$0.066$，$640$，$1600$，$64$，$10$，$14$）、0.1B（$0.109$，$768$，$1920$，$64$，$12$，$16$）、0.17B（$0.166$，$896$，$2240$，$64$，$14$，$18$）、0.2B（$0.241$，$1024$，$2560$，$64$，$16$，$20$）、0.5B（$0.499$，$1344$，$3360$，$64$，$21$，$24$）。可以看出 head dimension $d_h=64$ 全程固定，$d_{ff}$ 恒为 $d_m$ 的 $2.5$ 倍，层数 $L$ 从 8 递增到 24，这就是所谓固定 aspect ratio（长宽比）。表格下方两行说明也很关键："Note that the gap between the largest model here and the actual model they train is ~5x"（阶梯上最大的模型与真正要发布的模型之间约有 5 倍差距）与 "Optimal batch, LR, token-to-size ratios are directly fitted via scaling analysis"（最优批大小、学习率、token 与参数量之比都由缩放分析直接拟合）。转写把这套做法解释得很透彻：目标不是暴力训几百个 1.5B 模型，而是 "get the hyper parameters exactly right the first time"（第一次就把超参调准），因此先训一批小得多的模型，搭一条 ladder 一路外推到目标尺寸，差距约 5X；要钉死的 "sensitive parameters" 正是 optimal batch size、learning rate 以及 Chinchilla 讲的 token-to-size ratio。

这一章与前后内容的关系是承上启下的：它把第 1 章抽象的「超参数是规模的函数」落成了两套具体动作——μP 负责把 $\eta^*$ 从 $N$ 的依赖里摘出来，缩放阶梯负责用 ~5X 的外推去拟合剩下的 $B^*$ 与 token-to-size ratio。易错点也不少。其一，μP 里的逐参数学习率（per-parameter learning rates）在常规训练框架里并不常见，实现时很容易漏掉 LM head 与 embedding 的两处缩放；其二，讲者明确说 "this is not the only way to do it"（这不是唯一做法），μP 只是 "one effective way"，后面 DeepSeek 会走完全相反的路；其三，外推本身有风险——阶梯最大只到 0.5B，却要外推到约 1.5B 的目标模型，5X 的 gap 意味着拟合曲线必须足够干净；其四，讲者顺带指出一个时代性变化：到了 2026 年，"how you set learning rates, how do you do Chinchilla" 这类经典操作已被视为理所当然，论文不再细讲，所以反而要回到 MiniCPM 这种旧论文才能看到完整配方。

## 最优学习率的稳定与最优批大小的幂律 [07:00 - 09:48]

第 2 章留下的第一个待验证命题是：μP 真的把最优学习率钉死了吗？讲者自问 "is it?"，随后给出肯定回答："at least in sort of the MiniCPM case, the answer is yes"，而且 "it's very, very clean in their experiments"。幻灯片 "Optimal LR" 的副标题正是这句怀疑的书面版："According to muP – optimal learning rate should be (roughly) stable. Is it?"（按 μP 的说法最优学习率应当大致稳定，真是这样吗）。图中标题为 "LR v.s. Loss"，横轴是 Learning Rate，范围从 $10^{-3}$ 到 $10^{-1}$，纵轴 Loss 大约从 2 到 8；图例给出五个模型规模：0.04b、0.1b、0.3b、0.5b、2.1b，即每条折线对应一个 model size 的学习率扫描（sweep）。图下 Figure 3 的 caption 写着 "Loss vs Learning Rate. After applying for the Tensor Program, the learning rate shift becomes minimal."（应用 Tensor Program 之后，学习率的漂移变得极小）。讲者口述与图完全对上："They sweep learning rate across a variety of models, so each line is a model size. Um the minimal loss is basically right on 10 to the -2"，也就是各条曲线的最低点几乎都落在

$$\eta^* \approx 10^{-2}$$

附近；他补了一句诚实的限定："the minimum is slightly shifted for the smallest model, but I mean even then it's quite close to the minimum"，即最小那个模型的最低点略有偏移，但仍离得很近。这段的结论被讲者概括为 "a nice success case of the μP approach"：一旦能把最优学习率压缩到单一数值，就 "removed the need to tune the learning rate"（不必再为每个规模重调学习率），这正是第 1 章第三个 bullet 所期待的「挑特定参数化让模型 scale nicely」。

![应用 Tensor Program 后不同模型规模下学习率与损失曲线，最低点集中在 10^-2](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0007_00-07-49.jpg)

但学习率被固定，并不意味着所有敏感超参数都消失了。讲者立刻转折："even if you have a nice parameterization that fixes your learning rate to a single sort of optimum, your optimal batch size is still not going to be kind of the same"，并明确指出 "the batch size actually varies as a function of both data set size and model size"（最优批大小同时是数据量和模型大小的函数）。用第 1 章的记号写，就是 $\eta^*$ 被 μP 近似压成常数，而

$$B^* = B^*(D,\,N),\qquad \eta^* \approx \text{const}$$

仍然成立。幻灯片 "Optimal batch" 把这套实验画得很清楚，标题说明是 "Three model sizes (9m, 30m, 170m) as a function of data size (y), batch (x) and loss (col)"（三种模型规模 9M、30M、170M，纵轴为数据量、横轴为批大小、颜色为损失），三张并列子图各自带一条颜色条表示 Loss。图下两句说明是画面独有的："Vertical columns of points represent a single training curve (fixed batch, more points)."（竖直的一列点代表一条固定批大小、逐步增加 token 的训练曲线）以及 "Red line attempts to identify minimum loss points for each y-value – this is the 'optimal batch size' for a model size / dataset size combination."（红线试图为每个数据量找出损失最低点，也就是该「模型规模/数据规模」组合下的最优批大小）。转写里讲者把同一件事说成：每一列点是一次 training run，固定 batch size、自下而上增加处理过的 token 数；然后往这些点拟合 "quadratic sort of curves"（二次曲线）去找 equal loss contour（等损失轮廓）上的极小值，再把红线连起来得到 "optimal batch size scaling as a function of the number of tokens processed"，并且在多个 model size 上重复。

由此得到的核心规律是临界批大小（critical batch size）式的幂律。讲者说 "the optimal batch size does have a sort of power law structure with respect to the target loss"（最优批大小相对目标损失呈幂律结构），并把它直接接回 Kaplan："we kind of see exactly the Kaplan style analysis of, you know, if your loss is lower on the left, your batch size should be bigger, right? Critical batch size scaling." 也就是目标损失越低，所需的最优批越大。符号化可写成

$$B_{\text{crit}}(L_{\text{target}}) \propto L_{\text{target}}^{-\alpha},\qquad B^* = B^*(D,N)$$

这里的 $\propto$ 与 $\alpha$ 是讲者「幂律结构」这一描述的记号，画面与转写都没有给出具体指数，因此不填数值。它的用法是反推：给定目标损失，先从 scaling loss 里 "back out"（反解）出该损失对应的位置，再据此 "set what the optimal or critical batch size is that would give you the best trade-offs"，讲者评价为 "a nice clean trend"、"let you set batch size very easily and precisely"。讲者随后把这两者收束成本讲最看重的一对量："those two, um the batch size and the learning rate scaling, um they're going to come back over and over again. Those are arguably the most important and sensitive parameters to do scaling laws on."（批大小与学习率缩放会反复出现，堪称做缩放定律时最重要也最敏感的参数）。

这一章在整段里起「验证 + 转折」的双重作用：它验证了第 2 章 μP 的有效性（$\eta^*\approx 10^{-2}$），又用 $B^*=B^*(D,N)$ 说明 μP 并非万能，从而为下一章埋下动机——即便学习率不用重调，要做 Chinchilla 式的数据/模型权衡扫描，仍得反复在不同数据量上训练，而这件事在 cosine 学习率下代价极高。易错点集中在三处：其一，图上五条曲线的最低点是「大致」对齐 $10^{-2}$，最小模型明显略有偏移，不能当成严格不变；其二，把最优批大小对目标损失作图得到的是拟合出来的趋势（讲者提到 quadratic curves 与 equal loss contour），拟合方式本身会引入误差，红线是 "attempts to identify" 而非真值；其三，Kaplan 式的临界批大小结论是在特定设定下得到的，讲者用 "maybe you would not be surprised by this" 来衔接，语气是类比而非严格等价，直接把指数或系数搬过来用并不稳妥。此外讲者反复强调 μP "is not the only way to do it"，并预告 "we will talk about several other models and other papers that use μP"——这份开放式态度会在第 6 章 DeepSeek 的对照路线上得到呼应。这也正是下一章要处理的麻烦：如何在不重训的前提下扫过数据维度。

## WSD 学习率：把二次成本压回线性 [09:48 - 12:45]

第 3 章把 $B^*$ 和 $\eta^*$ 都交代了，剩下的是 Chinchilla 式的 model size vs data tradeoffs（模型规模与数据量权衡）。讲者先把做这件事的标准流程复述一遍："to do Chinchilla, right? We need to train a lot of different models. So, we're going to, let's say, do isoflops." 也就是固定算力预算 $C$，在 token 数与模型大小之间来回换，靠增加训练序列数来扫数据维度。麻烦出在学习率调度上：讲者提醒 "each one of these gets a different learning rate schedule"、"you're all training on cosine learning rates. Cosine learning rates look different. You need to know the terminal sort of total budget before you train in order to have a cosine schedule"——cosine（余弦）调度的形状由总步数决定，开训之前就必须知道终点。于是想训一个 8 million sequence 的模型，就 "can't start, like restart from the end of a 4 million sequence model"，因为学习率曲线根本不同，只能 "restart from scratch"。幻灯片 "What remains – model size vs data tradeoffs." 把这一困境写成了画面独有的两行字："From chinchilla – to fit a scaling law, we need to train from scratch, not just early stop"（拟合缩放律必须从头训，不能简单早停），以及最底下那句带点自嘲的结论 "This turns the cost of fitting a scaling law from n to n^2.. Can we avoid this?"（这让拟合成本从 $n$ 变成 $n^2$，能不能避免）。把讲者的定性判断写成记号就是：cosine 下扫 $n$ 个数据量档位，每档都要独立重训，总成本近似

$$\text{Cost}_{\text{cosine}} \sim O(n^2)\,\text{runs},\qquad \eta_{\text{cos}}(t;T)\ \text{依赖总步数}\ T$$

其中 $\eta_{\text{cos}}(t;T)$ 强调调度函数在训练前就被总预算 $T$ 绑死。那张幻灯片还配了六联图（两行三列）作为直觉支撑：左列纵轴是 Learning Rate / Max LR、中列是 Training Loss（约 2.70–3.00）、右列是 C4 Loss（约 2.80–3.20），横轴都是 Million Sequences（上行 0–8、下行 0–12.5）；图例 "Cosine Cycle Length" 列出 1.0x、1.1x、1.25x、1.5x、2.0x、5.0x num. steps 六种周期长度，不同颜色的曲线正是「同一模型、不同总预算」下互不重合的学习率轨迹，直观说明了为什么不能复用。

![MiniCPM 的 WSD 学习率与余弦调度对比图，附 Figure 15 图注与论文方法段落](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0010_00-11-26.jpg)

MiniCPM 给出的解法被讲者称作 "one of the nice solutions"，也就是后来通行的 WSD（warm-up stable decay，预热—稳定—衰减）学习率，他形容它 "basically a big trapezoid"（基本就是个大梯形）。三阶段的定义在口述里给得很细：warm-up phase（预热阶段）"is usually defined as a constant number of steps, not as a fraction of your training target"，即按固定步数而非总训练量的比例来定，因此 "independent of your training horizon"（与训练总时长无关）；接着是 "a fairly long run where you're holding the learning rates constant"，即 stable phase（稳定阶段），训练绝大部分时间都停留在这里；最后 "at the end you're going to decay fairly rapidly down to zero"，即 decay phase（衰减阶段），讲者给出两个量化区间：衰减段 "something like 10 to 20% of your total training run"，末端 "down to usually about 10% of your maximum learning rate"。把讲者口述整理成分段函数：

$$
\eta_{\text{WSD}}(t)=
\begin{cases}
\eta_{\max}\cdot t/t_w, & t<t_w \quad(\text{warm-up，}t_w\ \text{为固定步数})\\
\eta_{\max}, & t_w\le t<t_s \quad(\text{stable，占训练的绝大部分})\\
\eta_{\max}\cdot\big[0.1+0.9(1-\tfrac{t-t_s}{t_d})\big], & t_s\le t\le t_s+t_d \quad(\text{decay，}t_d/T\approx 10\%\!\sim\!20\%)
\end{cases}
$$

配图 "(partial) solution in miniCPM – WSD learning rate" 把这些数字画得很具体：副标题是 "Instead of cosine, split learning rate into warmup, stable, and decay phases."，左图横轴 Iteration 从 0 到 10000、纵轴 Learning Rate 从 0.00000 到 0.02000，图例三条曲线为 Cosine(40N)、WSD(40N,4N)、WSD(80N,8N)——括号里的两个数分别是总 token 预算与衰减段长度，正好体现「同一条稳定段、不同终点」的设计；Figure 15 的 caption 是 "Illustrative comparison between Cosine LRS and WSD LRS. The WSD LRS with different end steps share the same stable training stage."（不同结束步数的 WSD 共享同一个稳定训练阶段），底部还有一句 "For chinchilla-style analysis, can restart the run at the end of the stable phase."（做 Chinchilla 式分析时，可以在稳定阶段末尾重启训练）。

右侧那段论文原文摘录是画面上信息最密集、转写完全没念到的部分，值得逐句转述。第一句给出成本结论："In this section, we introduce the utilization of the WSD scheduler as an effective approach to explore the scaling law with linear cost ($O(nC)$)."（用 WSD 调度器以线性成本 $O(nC)$ 探索缩放律），紧接着解释机制："Since WSD scheduler has the advantage of arriving at the optimal loss of Cosine LRs after decaying from stable stage's checkpoints of any step, we are now able to precisely measure the optimal scaling properties without re-training the models from scratch to different amount of tokens, thus making the scaling law measurement much more efficient along the data axis."（从稳定阶段任意步的 checkpoint 衰减后都能达到 Cosine 的最优损失，于是无需对不同 token 量从头重训，就能沿数据轴高效测量缩放性质）。第二句给出实验规模："We measure the scaling law along the data and model axes by training SLMs of 6 sizes ranging from 0.04B to 2B, each with 6 decayed model starting from checkpoints of 10N to 60N data during the stable training stage."（沿数据与模型两个轴，训练 6 个规模、从 0.04B 到 2B 的小模型，每个规模从稳定阶段的 10N 到 60N 数据 checkpoint 各做 6 次衰减）。第三、四句讲评测细节："The final loss is evaluated on five held-out evaluation dataset."（最终损失在五个留出评测集上评估）、"To potentially compare the loss when the model uses different tokenizer, we take the average of loss by number of bytes instead of number of tokens, following Anil et al. (2023)."（为跨 tokenizer 可比，按字节数而非 token 数平均损失，遵循 Anil et al. 2023），并说明结果画在 Figure 17 的蓝线里。这些数字——$O(nC)$、6 sizes、0.04B–2B、6 decayed model、10N–60N、five held-out dataset——都是正文必须承接的硬参数。

WSD 之所以省算力，机制可以浓缩成一句：想延长训练时，"I'll just roll back my training to the last stable checkpoint and then I'll just run the stable learning rates forward and then re-decay the model down to zero"。讲者强调 "you do have to re-decay every time, but that's maybe 10% of the total cost"，并对比 "it's a much better situation to be repeatedly decaying than to repeatedly rerun your pre-training runs"。用记号写，成本从 $O(n^2)$ 降为

$$\text{Cost}_{\text{WSD}} \approx O(nC) + n\cdot O(0.1\,C)\,\text{的衰减开销}$$

这恰好回答了第 1 章第二个 bullet「能否省算力」。它与前后内容的关系也很清楚：第 3 章用 μP 解决了 $\eta^*$、用幂律解决了 $B^*$，本章则解决「如何在数据维度上廉价地扫点」，三者合起来才让第 5 章的 Chinchilla 复现成为可能。易错点有三：其一，warm-up 必须按固定步数定义，若错写成总步数的比例，就重新把调度绑回训练时长，复用性直接失效；其二，每次延长都要重新衰减，省掉的只是稳定段而非全部成本，讲者用 "repeatedly decaying" 反复提醒；其三，讲者说 "this seems very wasteful, um and I think lots of other people did too"，并把这个 trick 归为 "a very common trick that's used in many places"，说明它是工程惯例而非 MiniCPM 独占的理论突破，引用时不应把它当成新提出的缩放律。

## WSD 的性能验证与 Chinchilla 复现 [12:45 - 15:58]

有了调度方案，还得回答一个自然而然的质疑：稳定阶段一直保持高学习率，最终效果会不会比不过精心调过的 cosine？讲者先给出定性描述："WSD curves always look a little bit funky because it looks like you're way under performing until you hit the decay phase where suddenly you kind of reclaim all of your gains and then you sort of match or even sometimes exceed the cosine learning rate performance."（WSD 曲线看起来很怪：衰减前一直明显落后，一进衰减段就把差距全部追回，甚至偶尔超过 cosine）。配图 "WSD learning rates work well in miniCPM" 的副标题把这句话压成了画面独有的一行："Slower during the stable phase, rapid loss decay during decay phase. Decay ~ 10%."（稳定期更慢，衰减期损失快速下降，衰减约占 10%）。图中纵轴是 Loss on C4（约 3.6–4.4）、横轴是 Tokens (B)（约 20N–140N），图例列出七条曲线：WSD(40N,2N)、WSD(60N,2N)、WSD(80N,2N)、WSD(40N,4N)、WSD(60N,6N)、WSD(80N,8N) 以及黄色的 Cosine(80N)。可以清楚看到绿色系 WSD 曲线在稳定段整体压在黄色 cosine 之上（损失更高），到各自终点附近陡降，最后与 cosine 的 3.6 左右汇合——这正是 "reclaim all of your gains" 的图像化。讲者还提到 "who was the most recent folks that were um maybe it was Martin Jaggi and others I think over at ETH had like an extensive study of like WSD sort of performance"，即 ETH 的 Martin Jaggi 等人做过系统的 WSD 性能研究，而且 "a lot of papers have shown similar things"。

![MiniCPM 复现 Chinchilla method 3 的 Ultratext 联合拟合等高线图与公式框](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0013_00-15-02.jpg)

讲者对二者的取舍给得相当克制："anecdotally, I think if you if you talk to a bunch of people training networks, I think a lot of people will say the cosine is slightly better in many cases, um but WSD is basically as good in many other situations. Like it is a very versatile default learning rate, um doesn't necessarily suffer from having to sort of rerun uh if you want to just continue runs."（经验上很多人会说 cosine 在很多情形略好，但 WSD 在另一些情形基本一样好，是个非常通用的默认学习率，而且不必为续训重跑）。他把这段的教益落在衰减本身："it is very cool though to see, you know, just how much of an impact learning rate decay has"、"This part of decaying the learning rate is just incredibly important to sort of getting the final sort of annealing parts of your uh training uh together"——衰减段之于训练末期的 annealing（退火）至关重要。这为第 4 章那个「只花 10% 成本重做衰减」的做法补上了性能上的合法性：便宜不是白便宜，是因为最后那一小段本来就是决定最终损失的关键。

有了 WSD，"Chinchilla style analyses are actually very easy"。讲者描述的新流程是 "essentially run one long run and then you'll just sort of rewind the checkpoints and then repeatedly decay"，这样 "that will give you sort of sweeps along the data dimension, and then you'll also do sweeps along the model dimension. You can do isoflops or any other kind of analysis that you would like"。也就是说第 4 章的 checkpoint 复用直接兑现成沿数据轴的 sweep，再叠加模型轴的 sweep，就能做 isoflops 分析。MiniCPM 作者 "for whatever reason, choose to try to replicate methods one and three"，讲者立刻加了一句评价："which to me are the least reliable of the Chinchilla methods"（在我看来是 Chinchilla 三种方法里最不可靠的两种），但承认 "they do get fairly nice-looking scaling curves"。method 1 的结果是 "nice lower envelopes that look mostly linear across these"（下包络线看起来基本线性），method 3 的结果是 "nice smooth scaling for the most part in both sort of the compute to non-embedding parameter trade-off"（算力与非 embedding 参数量权衡上大体平滑）。

配图 "Chinchilla method 3" 把联合拟合（joint fit）的细节全给了出来，这些是转写没念到的核心增量。副标题写着 "Their primary scaling approach is the joint fit – they find very high data-model ratios."（主要缩放手段是联合拟合，他们得到很高的数据/模型比）。图是一张标注 "Ultratext" 的等高线图，横轴 Non-embedding Parameters ($10^9$)、范围 $10^{-1}$ 到 $10^1$，纵轴 Compute ($10^{18}$ Flops)、范围 $10^{-1}$ 到 $10^3$，右侧色标从 0.34 到 3.37，黑点是各 isoflops 实验点。图里嵌着几个公式框，逐字转述为：

$$L(N,D) = \frac{7.54\times10^{-2}}{N^{0.30}} + \frac{2.92\times10^{-1}}{D^{0.30}} + 0.25$$

以及 $K^2 = 0.01$、$\eta = -0.00$，还有那个最关键的比值

$$\left.\frac{D_{\text{opt}}}{N_{\text{opt}}}\right|_{C=10^{21}} = 95.60$$

即在 $C=10^{21}$ FLOPs 处，最优数据与参数之比为 95.60。这与讲者口述的质疑严丝合缝："they get pretty different exponents for their overall joint fit compared to Chinchilla. Um they claim in the paper that this means that they should be with way more tech pieces of text than uh Chinchilla. It is highly unclear to me whether this is a really true or whether their Chinchilla fits are a little bit strange relative to um the original Chinchilla paper."（联合拟合的指数与 Chinchilla 差别很大，作者据此声称应当用远多于 Chinchilla 的 token；这到底是真的，还是他们相对原论文的拟合本身有点怪，我高度存疑）。注意指数 $0.30$ 同时出现在 $N$ 与 $D$ 两项上，常数项 $0.25$、$K^2=0.01$、$\eta=-0.00$ 都是画面独有的数值，转写只笼统说「指数不同」。

本章收束了 MiniCPM 这条线，讲者在暂停答疑时自己点了两个 takeaway："one is, you know, the use of μP is a useful trick, I guess, to try to stabilize how your learning rate changes over scales. And then the second thing maybe to remember is the WS learning rate. That's just like a very common trick. You should all kind of know about this learning rate."（一是用 μP 稳定学习率随规模的变化，二是 WSD 学习率，这是人人都该知道的常见技巧）。它与前后内容的关系是「验证 + 收尾」：既验证了第 4 章 WSD 的性能等价性，也把第 2、3 章的 μP 与批大小规律合并成一次完整的 Chinchilla 复现，随后把舞台交给第 6 章路线完全不同的 DeepSeek。易错点值得单独强调：其一，稳定阶段 WSD 损失高于 cosine 是调度形状造成的假象，不能据此判定 WSD 更差，判断必须等到衰减结束；其二，讲者明确把 method 1 与 method 3 列为 "least reliable"，因此 "nice-looking" 的下包络与平滑拟合不等于指数可信；其三，95.60 这个 data-model ratio 与原 Chinchilla 结论冲突，讲者用 "highly unclear" 表达了保留，正文不应把它当作已确立的新定律；其四，"cosine is slightly better in many cases" 是 anecdotal（经验传闻）而非受控结论，实践中的默认选择仍取决于是否需要续训与反复衰减。

## DeepSeek：直接拟合超参数的缩放律 [15:58 - 19:15]

本段最后一部分转向 DeepSeek。讲者先定调：这是 "the kind of, you know, original DeepSeek paper um before, you know, they started talking about MoEs and all these other sort of very cool things"（MoE 之前那篇原始 DeepSeek 论文），并给出很高的评价——"even at the very first DeepSeek paper, you could kind of tell that these were very serious people"、"DeepSeek LLM, you know, even now, is I think one of the more nicely executed scaling analyses in the open world"（开放世界里执行得最漂亮的缩放分析之一）、"they have lots of great taste in the experiments that they run"，甚至说早在 R1 之前就 "was like kind of clear that these people were were going places"。配图 "DeepSeek" 页的文字很简洁但信息是独有的："DeepSeek (2024) – another LM with careful scaling analysis"，中间贴着 "DeepSeek LLM / Scaling Open-Source Language Models with Longtermism" 题头，底部一行是 "7 and 67B param models – generally high performance compared to other open LM"（7B 与 67B 两个参数规模的模型，相比其他开放语言模型总体表现很高）。这里的 7B、67B 是画面给出的硬数字，转写本段并未念到。

路线差异是这一章的骨架。讲者说 DeepSeek "takes a very different scaling strategy than uh MiniCPM and and several others in trying to optimize their hyperparameters"："They're not going to really do μP, right? They're not going to do this like interesting scaling adjustments. Um instead, they're going to directly try going to try to fit a scaling law to estimate both the optimal batch size and the optimal learning rate." 换句话说，MiniCPM 用 μP 把 $\eta^*$ 从规模依赖里「消掉」，DeepSeek 则承认 "the learning rate and batch sizes should actually change with scale"，但赌它们的变化 "are predictable, just like a scaling law"，于是 "we're good to go"。用前几章的记号对照，两条路线就是：

$$\text{MiniCPM: }\ \eta^* \xrightarrow{\mu P} \text{const}\qquad\text{vs.}\qquad \text{DeepSeek: }\ \eta^*=\eta^*(C),\ B^*=B^*(C)$$

具体做法是 "run extensive grid searches at various different scales, and they try to find the optimum"：固定算力、固定处理的 token 数，"sweeping learning rates, they're sweeping batch sizes, and they're looking at sort of the terminal loss"；"they'll do this across several different scales and then they'll basically try to identify what the lowest loss is, in this case the lightest color"，即在热力图里找颜色最浅的点、打个星号，跨规模收集这些最优点。配图 "Scaling strategy – batch + LR" 把这段实验原样搬了上来，副标题是 "Scaling strategy: don't use any muP, directly estimate optimal batch / LR"（不用任何 μP，直接估计最优批大小/学习率），正文段落逐字为："We initially conducted a grid search for batch size and learning rate on small-scale experiments with a compute budget of 1e17, and the results for a specific model size (177M FLOPs/token) are illustrated in Figure 2(a). The results demonstrate that the generalization error remains stable across a wide range of choices of batch sizes and learning rates. This indicates that near-optimal performance can be achieved within a relatively wide parameter space." 两张热力图分别标注 (a) 1e17 FLOPs (177M FLOPs/token) 与 (b) 1e20 FLOPs (2.94B FLOPs/token)，纵轴 Batch Size (Tokens) 覆盖 $2^{15}$ 到 $2^{19}$，横轴是 Learning Rate 的 2 的幂次，色标分别约 3.8–5.0 与 2.475–2.650，右图还用红框和星号标出了选中的最优点。1e17、1e20、177M、2.94B 这四个数字连同「泛化误差在很宽参数区间内保持稳定」的结论，都是画面独有、必须转述进正文的增量。

![DeepSeek 拟合的最优学习率与最优批大小幂律公式及两张缩放曲线](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0016_00-18-39.jpg)

把跨规模的星号收拢起来，就得到配图 "Scaling analysis of learning rates" 上的公式。讲者描述这张图："I'm going to vary my non-embedding training flops across some range" 时，"my optimal batch size will change"、"I will get something that kind of looks like a line"，学习率同理 "I'll get a bunch of gray dots in my sort of non-embedding flops range and then I can fit a line to that"。幻灯片副标题给了取点标准："Small scale runs + collect 'near optimal' (within 0.25% of min) models."（小规模运行 + 收集「近最优」模型，即损失在最小值 0.25% 以内），写成判据就是 $\big(L(\eta,B)-L_{\min}\big)/L_{\min}\le 0.25\%$。带编号 (1) 的两条拟合式是本段最关键的公式原文：

$$\eta_{\text{opt}} = 0.3118 \cdot C^{-0.1250},\qquad B_{\text{opt}} = 0.2920 \cdot C^{0.3271}$$

其中 $C$ 是 Non-Embedding Training FLOPs（非 embedding 训练算力）。两个式子一负一正，含义很直白：算力越大，最优学习率越低、最优批大小越大，讲者口述正是 "At higher flops, they they have much higher batch sizes and much lower learning rates, right? The stars are the things they actually ran for their their big runs."（星号是他们真正跑大规模训练时选的点）。左图 (a) Batch size scaling curve 纵轴 Optimal Batch Size (Tokens) 约 $2^{16}$ 到 $2^{30}$、横轴 Non-Embedding Training FLOPs 约 $10^{16}$ 到 $10^{24}$，图例为 "78B MHA 2T Token" 与 "67B GQA 2T Token"，并标注 #19.7M、#9.2M 等批大小数值；右图 (b) Learning rate scaling curve 纵轴 Optimal Learning Rate 约 $1.25\times10^{-4}$ 到 $8\times10^{-3}$，末端标着 4.2e-04 与 3.2e-04。幻灯片最底下还有一句讲者风格的吐槽："Learning rate fit looks a bit questionable.."（学习率的拟合看起来有点可疑）。

讲者的评价必须连同公式一起记下来，否则容易把这两条幂律当成定论。他说 "These fits look reasonable for the batches, right? This kind of does look like a line. The one on the learning rate, I'm not quite sure that constitutes you know, the best linear fits that I've seen in my life. But I guess it works, it's probably fine, right? Like the models do train, the models do get reasonable things."（批大小的拟合确实像条线；学习率那条不敢说是见过最好的线性拟合，不过大概能用，模型确实训出来了、结果也合理）。紧接着他把这定性为该路线的短板："I think this is the one I would say like drawback of the learning rates approach. It's not really clear that if you try to do this grid search kind of thing unless your grid is in the right place, you've..."——转写在 [19:14] 处戛然而止，句子没说完，按「转写含糊处宁可略过」的原则，这里只保留「除非网格位置摆对，否则网格搜索未必可靠」这一层明确意思，不替他补完结论。

这一章与全段形成完整对照：第 1 章问「超参数的规模漂移该消掉还是建模」，MiniCPM 用 μP + WSD 回答了「消掉一部分、再廉价扫描」，DeepSeek 用 $B_{\text{opt}}(C)$、$\eta_{\text{opt}}(C)$ 回答了「干脆把它拟出来」，两者都没有借助对方的工具。易错点集中在四处：其一，网格搜索的前提是热力图里"loss 在宽参数区间稳定"这一经验观察，若网格没覆盖到真实最优点，星号就会标错，讲者用 "unless your grid is in the right place" 点明了这个脆弱性；其二，"near optimal" 用 0.25% 的容差取点，等于把一批并非严格最小的点纳入拟合，会引入系统性偏差；其三，学习率拟合的线性度明显弱于批大小，连幻灯片自己都写 "looks a bit questionable"，因此 $0.3118$、$-0.1250$ 这类系数不应被当成可跨架构、跨数据配方照搬的常数；其四，讲者反复说 "it's probably fine"、"the models do train"，语气是「工程上够用」而非「理论上精确」，把它与第 5 章对 95.60 的 "highly unclear" 放在一起看，正是本段想留给听众的那份谨慎：前沿缩放实践里的公式，更多是可用默认值，而不是放之四海皆准的定律。

## DeepSeek 收尾：超参数拟合的质量、WSD 变体与最终损失预测 [19:15 - 22:32]

本段从上一部分留下的一个技术尾巴切入：为什么 DeepSeek 拟合出来的最优学习率曲线看起来不太可靠。讲者在 [19:19] 直接说，网格搜索的分辨率不够时会 "got a lot of quantization error"（产生大量量化误差），于是最优点被吸附到离散的网格点上，跨规模收集后拟合出的就不再是干净的幂律，而 "end up with kind of something that looks like an arbitrary linear thing going on"（最后像是一条任意的直线）。这里的 quantization error 指的不是模型权重量化，而是超参数网格离散化带来的最优点定位误差：当学习率只按若干档取样时，真正的最小值可能落在两档之间，被记录成档位值。屏幕上那张 "Scaling analysis of learning rates" 幻灯片把这种做法写得非常清楚，副标题是 "Small scale runs + collect 'near optimal' (within 0.25% of min) models."——即只做小规模训练，并把损失落在最小值 0.25% 以内的模型都算作 near optimal（近最优）收集起来。基于这些点拟合出的两条幂律是

$$\eta_{\text{opt}} = 0.3118 \cdot C^{-0.1250},\qquad B_{\text{opt}} = 0.2920 \cdot C^{0.3271}$$

其中 $C$ 是 non-embedding training FLOPs（非嵌入训练浮点运算量）。幻灯片上并排放着 (a) Batch size scaling curve 与 (b) Learning rate scaling curve 两张 log-log 图，图例里能看到 "1B total 2T Tokens"、"67B GQA 2T Tokens" 以及 "$C = 4.7e24$ FLOPs" 一类标注；而幻灯片底部一行自嘲式结论是 "Learning rate fit looks a bit questionable.."（学习率拟合看起来有点可疑）。这与转写里讲者对学习率拟合不甚满意、对批大小拟合认可的态度完全一致：同一个网格、同一批实验，两个量的可辨识程度并不相同。

![DeepSeek 学习率与批大小 scaling 拟合公式、near optimal 采样说明与两张缩放曲线](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0017_00-19-51.jpg)

[19:33] 有学生追问一个很细的问题：在 "for the same number of training flops"（相同训练算力）下，图上会出现 "multiple data points with a large range of optimal learning rates"（一批对应很宽最优学习率范围的数据点），这到底是不是指 Y 轴？讲者在 [19:49] 确认问题后给出解释：他 "think" 原因是 the changes to the model sizes and things like this（模型尺寸之类的变化）。也就是说，同一 $C$ 可以由「大模型少数据」或「小模型多数据」实现，而最优学习率对模型规模敏感，于是等算力切片内自然散开成一段区间，而不是一个点。这是一个容易被误读的地方：Y 轴仍然是最优学习率，宽的散布来自切片内隐藏变量（模型规模、数据量）未被固定，而不是学习率对算力本身有多值映射。讲者用的是 "I think"，属于推测而非论文结论，转写没有给出更严谨的分解，因此这里不宜替他补出定量解释。

随后讲者把时间拉回 2024 年的行业背景：[20:01]–[20:19] 他说那是 "every open model builder was replicating the entire scaling law stack"（每个开源模型团队都在复刻整套缩放律流程）的年代，DeepSeek 也在 learning rate analysis 之外补了一个 "basically a Chinchilla replication"（基本就是 Chinchilla 复现）。为了做这种复现，DeepSeek 和 MiniCPM 一样采用 WSD（warm-up–stable–decay，预热—稳定—衰减）风格的学习率，因为 WSD 可以在不预知总训练长度的情况下反复回滚 checkpoint、只对末端衰减阶段重做，从而把 Chinchilla 式扫描的成本压下来。不过讲者特意指出 DeepSeek 有一个 "weird little variation"：它做 two decay phases instead of one（两段衰减而不是一段），并坦白 "I am not really sure why this is"（我不太确定为什么），而且 "it hasn't really been a thing that has taken off afterwards"（后来并没有流行开）。按上一讲的说法，其调度可描述为 80% + 10% + 10% 的多段结构。这里的易错点是把这个两段衰减当成必要技巧：讲者的落点是 "Regardless, the whole point here is to say WSD is an effective approach"（重点是 WSD 有效），两段衰减只是一个没有被后续工作采纳的实现细节。

真正让 DeepSeek 曲线优于 MiniCPM 的，是它选择了 isoflops（等算力）扫描。[20:45]–[21:04] 讲者说 "DeepSeek has much nicer scaling curves than MiniCPM because of their choice to go with isoflops"，从而得到 "very nice, clean sweeps across the fixed flops ranges" 以及干净的 Chinchilla-style tradeoffs。屏幕上 "Data-size tradeoff analysis: Chinchilla method 2" 这一页把 method 2 写得很具体，副标题是 "Straightforward isoflop-style analysis for selecting the model size tradeoffs."（用直截了当的等算力分析来选模型规模权衡）。三张子图分别是：(a) IsoFLOP curve，纵轴 Bits-per-Byte on Validation Set（验证集每字节比特数，刻度 0.8–1.8），横轴 Non-Embedding FLOPs/Token (M)（$10^8$ 到 $10^{10}$），图例按算力给出 1e17、3e17、1e18、3e18、1e19、3e19、1e20、3e20 共八档 U 形曲线；(b) Optimal model scaling，纵轴 Non-Embedding FLOPs/Token (M)（$10^7$–$10^{12}$），横轴 Training FLOPs ($C = MD$)（$10^{16}$–$10^{24}$），紫色标注写着 "DeepSeek LLM 67B / 4.3e11 FLOPs/Token"，右下角标注 "4.5e23"；(c) Optimal data scaling，纵轴 Tokens ($D$)（$10^8$–$10^{13}$），横轴同为 Training FLOPs ($C = MD$)，紫色标注是 "DeepSeek LLM 67B / 1.04e12 Tokens"。这些数字把 $C = MD$ 这一恒等式落到了具体规模：67B 模型对应约 $4.3\times10^{11}$ FLOPs/Token 与 $1.04\times10^{12}$ tokens，总算力约 $4.5\times10^{23}$ FLOPs。

这一串分析的 punchline（最终落点）在 [21:35]–[22:06] 讲得很动情：训练一批小规模、精心控制的模型，画出 power law（幂律），"in the end you get something pretty close to reality if you scale out enough"（外推足够远时会得到相当接近现实的结果）。屏幕上 "Scaling predicts final model loss" 一页的粗体结论是 "The fitted scaling models (generally) accurately predict the final model losses."，正文写着他们按 compute budget $C$ 与 optimal generalization error 拟合 loss scaling curve，并预测了 DeepSeek LLM 7B 与 67B 的 generalization error（如 Figure 5 所示），结果说明小规模实验可以准确预测大模型损失。Figure 5 的 caption 原文是 "Performance scaling curve. The metric is the bits-per-byte on the validation set. The dotted line represents the power law fitting the smaller model (grey circles). The blue stars represent DeepSeek LLM 7B and 67B. Their performance is well-predicted by the scaling curve." 图里纵轴为 Bits-per-Byte on Validation Set（0.4–1.8），横轴为 Training FLOPs ($C = MD$)（$10^{16}$–$10^{24}$），图例是 "7B MHA 2T Token" 与 "67B GQA 2T Token" 两颗星；灰色圆点是用于拟合的小模型，虚线是幂律拟合。讲者口头把两颗星说成 "the two stars are the models that they actually trained"（真正训练出来的模型），灰点是 "their scaling law fits"，并评价 "I mean it could be better, but it's not bad at all"（还能更好，但已经相当不错）。

![DeepSeek 用小规模拟合的幂律预测 7B 与 67B 最终损失的 Figure 5 曲线](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0019_00-22-16.jpg)

[22:08] 讲者收束这一整块，并顺手点出全讲的结构：应对 sensitive hyperparameters（敏感超参数）有两条路，"You can either attempt to stabilize them or you can attempt to just scaling law fit them"——要么用 μP 之类把它们稳定到不随规模漂移，要么承认它们会变、直接拟合它们怎么变并 "just go with the scaling law"。DeepSeek 代表的是后一条 "fit the scaling law approach"，而前面第 2–5 章已经讲过的 MiniCPM 是前一条。讲者还带着一点遗憾说 "I'm a little bit sad that these kinds of plots are now somewhat rare in open-source release papers"（这类图在开源发布论文里现在有点少见）。局限也很清楚：星号只有两个，预测「不算差」但不精确；灰色拟合点全部来自小算力区间，外推到 7B/67B 依赖幂律在跨越多个数量级时仍成立，而这正是本讲后面会反复质疑的假设。

## Qwen 2.5 与 Qwen 3：把超参数 scaling 变成标准配方 [22:32 - 24:04]

[22:34] 讲者转入「更近期的模型也有不错的 scaling laws」这一段，但先给了一个重要的阅读背景：这类工作 "is kind of like all known to everybody"（几乎人尽皆知），所以没人再把它写进论文的大章节里。换句话说，从 Qwen 2.5 开始，scaling law 不再是论文卖点，而是默认工程步骤。转写里给出的 Qwen 2.5 做法是："we need to tune some optimal hyper parameters like batch sizes and learning rates. So, we do a bunch of scaling experiments to figure out what the optimum are and then we have a scaling law and we predict the optimum and then we're able to do that even for MOEs"——需要调 batch size 与 learning rate，于是做一批 scaling 实验找最优点，拟合 scaling law，再用它预测最优点，而且连 MoE（Mixture-of-Experts，混合专家模型）也能做。讲者在 [23:08] 明确说 "this is basically the same thing as the deep seek analysis"（和 DeepSeek 的分析基本是同一回事），把它归到「直接拟合超参数」这条路线，而不是 μP 那条。

屏幕上 "Other models: Qwen scaling" 一页给出了论文原文的摘录，信息量远大于口头转述。小标题是 "Qwen 2.5 – batch and hyper scaling fits"，下面是 "3.2 Scaling Law for Hyper-parameters" 一节。原文写道："We develop scaling laws for hyper-parameter based on the pre-training data of Qwen2.5 (Hoffmann et al., 2022; Kaplan et al., 2020). While previous studies (Dubey et al., 2024; Almazrouei et al., 2023; Hoffmann et al., 2022) primarily used scaling laws to determine optimal model sizes given compute budgets, we leverage them to identify optimal hyperparameters across model architectures." 这句话点出了与 Chinchilla / Kaplan 传统的分工差异：过去 scaling law 用来在给定算力下选 model size，Qwen 2.5 用它来跨模型架构识别最优超参数。紧接着原文给出要研究的量："we analyze how the optimal learning rate $\mu_{\text{opt}}$ and batch size $B_{\text{opt}}$ vary with model size $N$ and pre-training data size $D$"，即把最优学习率与最优批大小建模成 $N$、$D$ 的函数：

$$\mu_{\text{opt}} = \mu_{\text{opt}}(N, D),\qquad B_{\text{opt}} = B_{\text{opt}}(N, D),\qquad L = f(\text{architecture},\, D)$$

这里 $\mu_{\text{opt}}$（论文用希腊字母 $\mu$ 记 learning rate）与 $B_{\text{opt}}$ 是待拟合的输出，$N$ 是模型参数量，$D$ 是预训练 token 数。屏幕上给出的实验覆盖范围是："dense models with 44M to 14B parameters and MoE models with 44M to 1B activated parameters, trained on datasets ranging from 0.8B to 600B tokens"——稠密模型 44M 到 14B 参数，MoE 模型 44M 到 1B 激活参数，训练数据 0.8B 到 600B tokens。这个跨度说明拟合不是只在玩具规模上做，最后一句 "we then model the final loss as a function of model architecture and training data scale" 表明他们把最优超参数的预测再接回 final loss（最终损失）的建模。

![Qwen 2.5 与 Qwen 3 关于批大小、学习率 scaling law 的论文原文摘录](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0020_00-23-28.jpg)

同一页的第二段原文继续展开 MoE 的用法："Additionally, we leverage scaling laws to predict and compare the performance of MoE models with varying parameter counts against their dense counterparts. This analysis guides our hyper-parameter configuration for MoE models, enabling us to achieve performance parity with specific dense model variants (such as Qwen2.5-7B and Qwen2.5-14B) through careful tuning of both activated and total parameters." 这里的关键词是 performance parity（性能持平）：用 scaling law 预测不同参数量 MoE 相对 dense 对手的表现，反过来指导 MoE 的超参数配置，并通过同时仔细调整 activated（激活）与 total（总）参数，让 MoE 达到与 Qwen2.5-7B、Qwen2.5-14B 这些具体稠密变体相当的水平。这解释了为什么讲者说 Qwen 2.5 "even for MOEs"：稀疏模型多了一个自由度——总参数量与激活参数量可以解耦，超参数拟合因此必须把架构维度也算进去，而不能只沿用 $N$、$D$ 两个标量。

下半页是 "Qwen 3 – similar scaling for LR/batch"，原文为："Similar to Qwen2.5 (Yang et al., 2024b), we develop scaling laws for optimal hyper-parameters (e.g., learning rate scheduler, and batch size) predictions based on three pre-training stages mentioned above. Through extensive experiments, we systematically study the relationship between model architecture, training data, training stage, and optimal training hyper-parameters. Finally, we set the predicted optimal learning rate and batch size strategy for each dense or MoE model." 相比 Qwen 2.5，这里多了两个变量：learning rate scheduler（学习率调度器）与 training stage（训练阶段，原文提到 three pre-training stages），并且系统研究了 model architecture、training data、training stage 与最优超参数之间的关系，最后为每一个 dense 或 MoE 模型设定 predicted optimal learning rate and batch size strategy。讲者在 [23:14]–[23:25] 的口头总结更直白："By the time it's Qwen 3, ... we've already figured out what to do in Qwen 2.5. We're just doing exactly the same thing"，并说这已变成 "a very standard recipe"（一套非常标准的配方）用来 "nail down the optimum optimal learning rates and batches"。

这一章的易错点在于把「有 scaling law」误解成「超参数已被理论解释」。无论 Qwen 2.5 还是 Qwen 3，本质仍是经验拟合：在小规模网格上找 $\mu_{\text{opt}}$、$B_{\text{opt}}$，拟合它们随 $N$、$D$（再加架构、阶段）的变化，再外推。论文里没有给出普适常数，讲者也没有念出任何具体指数；屏幕上能读到的确切数字只是实验覆盖范围（44M–14B、44M–1B activated、0.8B–600B tokens）与型号（Qwen2.5-7B、Qwen2.5-14B）。因此这里能安全记下的是「变量集合与流程」，而不是「公式」。另一个局限是外推依赖数据配方不变：本段后面 StepFun 的实验会专门显示，换训练数据后最优学习率与批大小会漂移，这正好给 Qwen 式「一次拟合、处处套用」的做法加了一个注脚。

[23:32]–[23:50] 讲者开始收束并预告下一块："Chinchilla 2 is probably the newest paper that I'll talk about of the series"，并说明接下来 "I'm just going to quickly go through several papers"，目的只是给一个 "overview of what people are doing in terms of scaling work in the open-source model releases"（概览开源模型发布里的 scaling 工作）。这一过渡在结构上很重要：从 Qwen 这种「把经典 machinery 变成默认配方」的模型，转向「经典 machinery 已经不够、必须为新架构发明新 scaling 轴」的模型。下一章要讲的 MoE 稀疏度正是第一个新轴。

## MoE 稀疏度 scaling 与 Llama 3 的下游映射 [24:04 - 26:53]

[23:54]–[24:26] 讲者给出这一块的动机："it's 2026, most people know you know how to do a Chinchilla scaling law or even a learning rate scaling law. What things are new?" 新东西是很多公司 "fully switching to MOEs"（全面转向 MoE），于是问题变成 "am I training the right amount of active parameters?"（我训练的激活参数量对不对）。Chinchilla 2 及另几篇因此聚焦 MOE scaling laws：做法是 "we're going to vary the flops and look at my losses and I'm going to do so for different levels of sparsity"，即在不同 sparsity（稀疏度）水平下扫 FLOPs 看损失，从而判断稀疏度帮了多少、伤了多少。结论讲者用 "Unsurprising"（不意外）形容："the more sparse your network, right? The better your validation loss given a flop"（固定算力下网络越稀疏，验证损失越好）。真正的增量不是这个定性结论，而是 "they can quantitatively know what the relationship is between sparsity and loss"（能定量知道稀疏度与损失的关系），从而 "make sort of more rational decisions about how much sparsity to put into their model"（对放多少稀疏度做更理性的决策）。

屏幕上 "Kimi K2" 一页把这套 Sparsity Scaling Law 的原文写得极其具体，标题下的说明是 "Many recent papers – sparsity scaling law to find the right sparsity levels"。原文定义："Sparsity is defined as the ratio of the total number of experts to the number of activated experts." 即

$$\text{sparsity} = \frac{N_{\text{experts,total}}}{N_{\text{experts,activated}}}$$

并且这套 law 是 "tailored for the Mixture-of-Experts (MoE) model family using Muon"（面向用 Muon 训练的 MoE 模型族定制的）。核心观察是："under a fixed number of activated parameters (i.e., constant FLOPs) — increasing the total number of experts (i.e., increasing sparsity) consistently lowers both the training and validation loss"。注意前提是激活参数数固定，也就是算力不变；此时只增加总专家数，训练损失和验证损失都持续下降。原文接着给出可核算的收益："under the compute-optimal sparsity scaling law, achieving the same validation loss of 1.5, sparsity 48 reduces FLOPs by 1.69x, 1.39x, and 1.15x compared to sparsity levels 8, 16, and 32, respectively." 即在达到相同 validation loss = 1.5 时，稀疏度 48 相对稀疏度 8、16、32 分别省下 1.69 倍、1.39 倍、1.15 倍 FLOPs。代价也写明了："Though increasing sparsity leads to better performance, this gain comes with increased infrastructure complexity. To balance model performance with cost, we adopt a sparsity of 48 for Kimi K2, activating 8 out of 384 experts per forward pass." 于是 $384/8 = 48$，每次前向只激活 384 个专家里的 8 个。Figure 5 的 caption 补充了实验设计："Increasing sparsity leads to improved model performance. We fixed the number of activated experts to 8 and varied the total number of experts, resulting in models with different sparsity levels." 图中纵轴是 Validation Loss（约 1.6–3.0），横轴是 Training FLOPs，四条曲线图例为 sparsity 8、sparsity 16、sparsity 32、sparsity 48。同页 Figure 6 的 caption 还给出一个与稀疏度并列的架构结论："Scaling curves for models with number of attention heads equals to number of layers and their counterparts with doubled attention heads. Doubling the number of attention heads leads to a reduction in validation loss of approximately 0.5% to 1.2%."

![Kimi K2 的 Sparsity Scaling Law 原文、稀疏度定义与不同稀疏度下的验证损失曲线](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0021_00-24-40.jpg)

讲者在 [24:47]–[25:00] 用一句口语把上面这一长段压缩了："what we're going to do is we're going to use a sparsity of 48 because at that point we're hitting diminishing returns on our scaling loss"（用到 48 是因为再往上 scaling loss 的收益递减），并总结 "you can kind of see how this kind of analysis is used to justify different architectural decisions across different open source releases"（可以看到这类分析如何被用来给不同开源发布的架构决策背书）。这里的易错点是把 48 当成一个普适最优常数：它是在「验证损失 1.5 的等损失比较 + 基础设施复杂度权衡 + Muon 优化器 + 特定数据」下选出来的工作点，1.69x / 1.39x / 1.15x 三个倍率也只在 sparsity 8/16/32 这几个对照点上成立。换 MoE 结构或换训练配方，收益递减的位置会移动。

[25:04]–[25:37] 讲者把 Hunyuan 作为对照："once they switch to the MOE has to do this kind of like a MOE scaling law"，几乎每家公司切到 MoE 都得补这一课，重新分析 activated parameters 数量或 sparsity 如何影响 training loss。差别在于 Hunyuan "they're fixing the sparsity level and so that's not a thing that they're searching over"（固定稀疏度，不把稀疏度当搜索维度），最终得到 "a 96 data point per active parameter ratio"（每个激活参数约 96 个数据点）。把它和 Kimi K2 并排看很有意思：Kimi K2 把稀疏度当自变量、搜出 48；Hunyuan 把稀疏度当常量、转而搜数据/激活参数比，得到 96。两者都是 MoE scaling law，但被固定的轴与被扫描的轴正好互换；转写没有给出 Hunyuan 的具体公式或图表，因此不宜替它补出幂律指数。

[25:37]–[26:10] 轮到 Llama 3，讲者评价很直白："Llama 3 also has some nice scaling laws. Unfortunately, they're not particularly useful ones." 他们做 iso-flop 式 scaling，"get slightly different but similar ratios for token to model"（得到略不同但相近的 token/model 比例）。屏幕上 "LLaMA 3 (2024) Scaling laws" 一页左半是 "Isoflops-style scaling (39-1 ratio)"：左图纵轴 Validation Loss（0.70–0.95），横轴 Training Tokens（$10^{10}$–$10^{12}$），图例按 Compute 列出 6e18、1e19、3e19、6e19、1e20、3e20、6e20、1e21、3e21、1e22 共十档 U 形曲线。Figure 2 的 caption 原文是 "Scaling law IsoFLOPs curves between $6 \times 10^{18}$ and $10^{22}$ FLOPs. The loss is the negative log-likelihood on a held-out validation set. We approximate measurements at each compute scale using a second degree polynomial." 这里有两个图上独有、转写完全没念到的细节：一是算力区间 $6\times10^{18}$ 到 $10^{22}$ FLOPs，二是每个算力档的曲线用 second degree polynomial（二次多项式）逼近来定位最小值，而不是靠肉眼找最低点。"39-1 ratio" 这个标注也与讲者说的 "slightly different but similar ratios" 呼应。

右半 "Compute-to-downstream scaling" 才是讲者认为 "the most interesting thing"：中间小图纵轴 Normalized NLL per Char.（每字符归一化负对数似然，1.200–1.400），横轴 Compute (FLOPs)（$10^{20}$–$10^{25}$），显示算力越大 NLL 越低；右边小图纵轴 Accuracy（0.3–1.0），横轴 Normalized NLL per Char.（1.40 到 1.20，反向），紫色点被一条 sigmoid（S 形）曲线穿过，图例为 Scaling Law Models、Llama 2 Models、Scaling Law Prediction、Llama 3 405B。讲者在 [26:10]–[26:49] 说，"if you have better sort of log loss then the downstream accuracy also goes up"，可以 "just fit a sigmoid on these purple points"，用损失到精度的映射估计模型好坏；但他同时保留态度："You can kind of see there's systematic deviations from the curve, so I don't really know if I would really buy that this curve is like the one truth"——点系统性偏离曲线，所以不认为这条 S 形曲线是唯一真相；不过 "it is consistent with a sigmoidal scaling mapping from the loss to the accuracy, which I think is very useful"，因为前面整讲都只在谈 log loss，而这里显示出 log loss 与 downstream accuracy 在很多情况下的 tight coupling（紧耦合）。这一章的局限因此有两层：MoE 稀疏度的定量收益依赖固定激活参数与具体基础设施成本；损失到精度的 sigmoid 映射是经验相关而非因果律，存在系统性偏离，不能直接拿它替代真实下游评测。

![Llama 3 的 IsoFLOPs 曲线、39-1 比例与 compute-to-downstream 的 sigmoid 映射](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0022_00-25-53.jpg)

## MiniMax-01 的架构 scaling 与近期配方总览 [26:53 - 30:29]

[26:54] 讲者引出本段最后一篇要细讲的论文："MiniMax zero one from last year had this nice set of scaling experiments"，并特意说明它做的是两讲前就提过的、scaling law 的一项有用 affordance（可用之处）：architecture variations（架构变体对比）。要比较的对象有三种：lightning attention（一种 linear attention，线性注意力架构）、"a soft max version like full attention"（softmax 全注意力）、以及 "a hybrid version that mixes the two"（把两者混起来的混合版本）。问的问题是 "How do these things scale? ... is it the case that if I scale up one of them will get much worse than the other?"——放大算力后，会不会某一种架构明显比另一种差。这与前面所有分析都不同：之前扫的是算力、模型规模、数据量、稀疏度这些「量」，这里扫的是「结构类型」，用同一套 Chinchilla 式工具去比较不同归纳偏置的可扩展性。

屏幕上 "MiniMax-01 (2025)" 一页的副标题是 "Architecture scaling laws + Chinchilla method 1"，下方三联图逐一给出：左图 "Loss vs Compute"，纵轴 Loss（约 2–6.5），横轴 PFLOP/s-days（$10^{-2}$ 到 $10^{3}$），一束淡色训练曲线配一条虚线幂律包络；中图 "Model size vs Compute"，纵轴 Number of parameters（$10^7$–$10^{10}$），横轴同为 PFLOP/s-days；右图 "Tokens vs Compute"，纵轴 Tokens（$10^9$–$10^{12}$），横轴同为 PFLOP/s-days。三图共用的图例是 Softmax Attention（黄）、Lightning Attention（紫）、Hybrid-lightning（红）。Figure 6 的 caption 原文为 "Summary of Scaling Laws. Training curves (left) span models from 70M to 7B parameters. Optimal model size (center) and training tokens (right) are derived based on a specified compute budget estimation." 也就是说训练曲线覆盖 70M 到 7B 参数，中图的最优模型规模与右图的最优训练 token 数，都是从给定算力预算估计里推出来的。把画面的结论形式化，可以写成三种架构在 log-log 坐标下近似重合的最优规模关系：

$$\log N_{\text{opt}} \approx a\log C + b,\qquad \log D_{\text{opt}} \approx c\log C + d,\qquad L_{\text{hybrid}}(C) \approx L_{\text{softmax}}(C) \approx L_{\text{lightning}}(C)$$

需要强调，画面只给出了 log-log 下近似直线的形状，并没有标注 $a$、$b$、$c$、$d$ 的具体数值，上式只是对「三者可比」这一结论的符号化转述，不能当成论文里读出的拟合系数。

![MiniMax-01 的架构 scaling laws 三联图与 Softmax、Lightning、Hybrid-lightning 三种注意力对比](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0023_00-27-05.jpg)

讲者在 [27:30]–[28:02] 给出口头结论：这组图 "replicating essentially the Kaplan plots"，看随着 FLOPs 增加学习曲线怎么变、不同架构在不同算力水平下所需的参数量是否不同；"the answer is for the most part kind of no that you know, hybrid and lightning and soft max are all comparable for the most part and they require similar model sizes"（大体上不是，三者大体可比、所需模型规模相似）。关键在于最后一句："this set of plots is used to justify their use of the hybrid architecture in the actual deployed system"——正是这组图被用来给实际部署系统采用 hybrid 架构背书，"this is exactly the kind of decision-making that you would do based upon a scaling law"（这正是基于缩放律做决策的典型）。这里的易错点是把「可比」读成「完全相同」：讲者说的是 "for the most part"（大体上）、"comparable"（可比），并没有说三条曲线在全部算力区间严格重合；而采用 hybrid 的理由也不只是损失略低，还包括线性注意力在长序列上的系统级收益，转写与画面都没给出这部分定量依据，不宜补写。

[28:09]–[29:12] 讲者用一张 "Recent scaling law recipes" 幻灯片把前面所有开源工作收成一张配方表，这张表在画面上出现、但转写只零散念到，属于本段最值得补全的独有内容：

```text
DeepSeek recipe
  • Assume most transformer hypers are invariant to scale
  • Do a scaling analysis on batch / LR to figure out optimal scaling
  • IsoFLOP analysis to figure out model sizing
    › Use a piecewise-linear schedule to make chinchilla scaling cheap.

miniCPM recipe
  • Use muP to make transformer + LR invariant to scale
  • Use a piecewise linear schedule to get sample for Chinchilla method 3 (curve fitting)

Recent (late 2024+) but less detailed
Qwen – Lr/batch (very few details..)
Kimi K2 – MoE scaling
LLaMA 3 / Hunyuan
  • Just isoflops (no other scaling details)
Minimax
  • Architecture choice / decision scaling
```

这段原文把两条主线写得非常凝练：DeepSeek recipe 假设大多数 transformer 超参数对规模不变（invariant to scale），于是只需对 batch / LR 做 scaling analysis 找最优缩放，再用 IsoFLOP 分析定模型规模，并用 piecewise-linear schedule（分段线性调度，即 WSD 一类）让 Chinchilla scaling 变便宜；miniCPM recipe 用 μP 让 transformer 与学习率对规模不变，同样用分段线性调度为 Chinchilla method 3 的 curve fitting 取样本。后面几行则是「2024 年末之后越来越不详细」的注脚：Qwen 只报 Lr/batch 且细节极少，Kimi K2 报 MoE scaling，LLaMA 3 / Hunyuan 只做 isoflops，Minimax 把 scaling 用在架构选择 / 决策上。讲者在 [28:14]–[28:42] 的口头版与此对应：DeepSeek 的关键是 batch 与 learning rate 上的 scaling analysis（"This is the scaling based approach"），MiniCPM 走 μP 求 invariances，Kimi、Qwen、Hunyuan 在做 MoE scaling experiments，MiniMax 做 architecture work，另有若干 iso-flops 复现。

紧接着他给出一个偏解释性的观察，而不是实验结论：现在公司写得越来越不详细，是因为 "the core scaling machinery like, you know, doing Chinchilla or doing learning rate scaling is now fairly well understood by everybody"，所以 "there's no necessarily extra value in ... showing those experiments"。讲者在 [29:07] 特意加了一句 "Or at least that's my guess for why we have much fewer of these kinds of experiments in the release"——这是他本人的猜测，不是论文自述，读笔记时不能把它当成行业已证实的动机。与前文的联系也很清楚：Qwen 3「照搬 Qwen 2.5」和这张表里 "very few details.." 正是同一个现象的两面，核心流程一旦标准化，展示它的边际价值就消失了；而这也解释了为什么本讲后半段要转向 StepFun 那种重新把超参数空间高分辨率网格化的研究——恰恰因为公开报告越来越少，才需要自己把现象重新测一遍。

[29:26]–[30:25] 是本段唯一一段问答，学生问的是：以上讨论都覆盖 pre-finished stage（预训练阶段），当 post-training（后训练）已成为标准化质量流程后，"what changes or what do not change"。讲者的回答相当克制："I think that's still a big open question." 具体说，"There's no great way to do like fully integrated scaling with like post-training accounted for"（没有一个好办法把后训练算进去做完全一体化的 scaling），原因是 "post-training like can sometimes really change the nature of like, you know, what pre-training you should do"（后训练有时会改变你该做什么样的预训练）。他提到最接近的相关工作是关于预训练里 coverage（覆盖度）或 diversity notions（多样性概念）的研究，试图用它们预测后训练会发生什么，但 "it's still pretty nascent work"（还很不成熟），"I don't think there's any good answer to the question of like what do we do about, you know, synergy with post-training"（对与后训练的协同没有好答案）。这一问的局限值得单独记住：前面所有 scaling law 的目标量都是预训练的 log loss 或 bits-per-byte，一旦评测目标换成经过后训练的下游质量，外推链条就断了；把预训练损失曲线直接当成最终产品质量的预测器，目前并没有被证实。

## StepFun 的超参数 scaling：公式分歧与纯经验网格设计 [30:29 - 34:46]

[30:29] 讲者用一段结构说明把本讲剩下的部分与前面的两个例子对齐："the two remaining parts of this lecture kind of map on cleanly to the two examples"——一部分讲 scaling laws for learning rates, batch sizes, and optimizers（学习率、批大小、优化器的缩放律），对应 DeepSeek 路线；另一部分讲 μP，对应 MiniCPM 路线，"Can we just reparameterize everything to make it so that we don't have to worry about scale"（能不能干脆重参数化一切，让规模不再是个问题）。现在开始的是前者，问题被明确为 "how do we scale learning rate properly"，并且会分成 "a hyperparameter part and a optimizer choice part"（超参数部分与优化器选择部分）。这一句解释了为什么上一章末尾那张 "Optimizer scaling" 幻灯片会被提前闪过：它是个预告，优化器的详细讨论留到后面的 μP / Muon 部分。

素材来源上，讲者在 [31:11]–[31:47] 说明他会 "draw quite a bit from this pretty recent preprint um from the StepFun folks"，同时坦白 "I don't think there's like a great, you know, really truly like reliable, robust study on hyper parameters, um especially learning rates"（我不认为目前有真正可靠、稳健的超参数研究，尤其是学习率）。之所以仍选它，是因为 "The StepFun folks train, you know, credible large models and release them. And they burned a ton of compute in grid searching the space of hyper parameters in the scaling law studies"——他们真的训练并发布可信的大模型，且在网格搜索超参数空间上烧了大量算力。讲者承诺要展示的是 "some intriguing empirical phenomena about how learning rates, batch size, and all those affect downstream losses"（学习率、批大小等如何影响下游损失的一些有趣经验现象）。这里的研究定位很关键：它不是推导出来的理论，而是高算力换来的经验测量。

[30:57] 那张 "Optimizer scaling" 幻灯片正好给这个定位提供了佐证，标题下的粗体是 "Optimizers choices / tuning can be tricky and scale sensitive"（优化器选择与调参很棘手且对规模敏感），中间一行大字提问 "How should we pick different optimizers?"。下方三张小图里，"C4/EN Loss for 1.2B Model" 纵轴 Loss（2.76–2.90）、横轴 Chinchilla Ratio（1–8），图例有 AdamW、Muon、NaNoW、Soap；"D_AdamW vs D_Optimizer (Model Size: 1.2B)" 纵轴是 Tokens Needed by AdamW / Tokens Needed by Optimizer（即加速比）、横轴 Tokens / Chinchilla（1–8），图例 Muon 与 Soap，并标注 Speedup 1.0-1.2x、1.2-1.3x、1.3-1.4x 三档；"C4/EN Loss for 300M Model" 纵轴 Loss（3.00–3.25）、横轴 Chinchilla Ratio（1–16）。这些图说明优化器优势本身随 Chinchilla Ratio 变化，是 scale sensitive 的；但本段转写并未展开优化器结论，讲者只用它过渡，因此这里能记的是变量与量级，不是「哪个优化器更好」的判断。

[31:52]–[33:27] 是本段信息密度最高的一页："Core question – scaling rates on LR and batch"，副标题直接问 "What are the right variables / functional form?"（正确的变量与函数形式到底是什么）。屏幕上的表格有六列：Name、Data Recipe、Model Sparsity、Learning Rate、Batch Size、Relative Error。逐行把画面上的公式原样抄录如下：OpenAI Law [20] 两列控制变量都打叉（✗），学习率 $3.239 * 10^{-3} + -1.395 * 10^{-4} log(N)$，批大小 $2e18\mathcal{L}^{-4.76190}$，Relative Error $9.51$‰；Microsoft Law [2] 为 $1.3192e^{-5} N^{-0.23} D^{-0.32}$，批大小一栏为空，误差 $9.25$‰；DeepSeek Law [6] 为 $0.3188C^{-0.1250}$ 与 $0.2920C^{0.3271}$，误差 $9.26$‰；Porian Law [26] 为 $3.7N^{-0.36}$ 与 $0.7576N^{0.703}$，误差 $3.71$‰；MiniCPM Law [18] 学习率一栏为空、批大小为 $\frac{2e18}{\mathcal{L}^{6.24}}$；MeiTuan Law [36] 的 Model Sparsity 一列打勾（✓），学习率 $\lambda\mathcal{L}^{-\alpha}$、批大小 $\lambda_B\mathcal{L}^{-\alpha_B}$；最后一行 Ours (Step Law) 在 Data Recipe 与 Model Sparsity 两列都打勾，学习率 $1.79N^{-0.713}D^{0.307}$，批大小 $0.58D^{0.571}$，Relative Error $0.94$‰。其中 $\mathcal{L}$ 表示损失，$N$ 是参数量，$D$ 是数据量，$C$ 是算力。可以注意到 DeepSeek 行写的是 $0.3188C^{-0.1250}$，与前面 "Scaling analysis of learning rates" 幻灯片上的 $0.3118$ 只有末位出入，应是同一拟合式在不同排版下的辨识差异，两处都不宜擅自改写。

![StepFun 汇总的各家学习率与批大小 scaling 公式对比表及相对误差](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0027_00-32-20.jpg)

讲者对这张表的点评比公式本身更重要。[32:12] 他先指最顶行："if you look at the batch size over there, notice how it says like 2 e 18, you know, capital L to the exponent. That's critical batch sizes, right? It's basically saying, you know, you should scale your batch size as a function of your terminal loss." 也就是 Kaplan / OpenAI 的 critical batch size（临界批大小）把批大小写成终端损失的函数；DeepSeek 则是 "compute-based scaling laws for both learning rate and batch size"（对 LR 与 batch 都用基于算力的缩放律）。然后他给出最尖锐的观察："these, you know, formulas for learning rate and batch sizes, they don't even have the same inputs, right? Like people don't even seem to agree what are the variables that should affect these learning rates." 各家公式连输入变量都不一致：OpenAI 用 terminal loss，StepFun 用 data，DeepSeek 用 compute，Microsoft / Porian 用 $N$、$D$ 的组合。表格下方的 "Different views" 把这三分法写成了原文：

```text
Different views:
  Critical batch: batch as a function of loss (OpenAI)
  Compute power law: poly function of compute (DeepSeek)
  .. Or something else?
```

讲者随后加了一句方法论警告："not to take even this last row as kind of gospel, right? A lot of these things are are pretty brittle in many ways, but the experiments and the phenomena are things that you can learn well from"（别把最后一行当福音，这些公式很多都相当脆弱，但实验与现象值得学）。这既是本章的易错点，也是与前后内容的接口：前面 DeepSeek 的学习率拟合已被讲者说成 "looks a bit questionable"，后面 StepFun 自己的 $D^{0.307}$ 依赖也会被讲者说成 "a little fragile"，说明分歧不是某一家做错了，而是这类拟合对实验设置高度敏感。

[33:30]–[34:46] 讲者转入实验设计："the design of this paper is very similar to the DeepSeek one"——训练大量模型，把 learning rate 空间与 batch size 空间都 grid 出来，"do them across a range of models and range of data set sizes"，目的是 "saturate the hyperparameter space for a set of small models and then nail down what the optimum is"。屏幕上 "The approach: purely empirical – grid search the space" 一页的副标题是 "Much like the DeepSeek paper – train models to try to map out the hparam space"。左侧是一张 contour plot（等高线图）：横轴 Learning Rate（$10^{-3}$ 量级），纵轴 Batch Size（约 $5\times10^5$ 到 $5\times10^6$），右侧 colorbar 标 Loss 从 2.08 到 2.18，等高线上标注着 +0.100%、+0.125%、+0.200%、+0.300% 等相对损失增量；图例里 Global Minimum 是红叉，Ours(Step Law) 是黄星，DeepSeek Law 是绿三角，另有 Microsoft Law、Porian Law、OpenAI Law 各自的标记与橙色虚线，可以直观比较各家公式预测点离全局最小有多远。右侧 "Table 5: Dense Model Configuration." 的表头是 Model、$N$、$D$、$d_{\text{model}}$、$d_{ff}$、$N_{\text{head}}$、$N_{\text{layer}}$，共 18 行配置；例如第一行 $N = 2.15\times10^8$、$D = 1.14\times10^{10}$、$d_{\text{model}} = 960$、$d_{ff} = 9368$、$N_{\text{head}} = 15$、$N_{\text{layer}} = 7$，另有 $d_{\text{model}}$ 取 1024 / 1280 / 2048、$d_{ff}$ 取 8192 / 9472 / 9552 / 9048 等多种组合，说明网格并非只扫两个超参数，模型形状本身也被系统性变化。讲者在 [34:25] 说这张切片 "is a slice at 1B parameters and 100B tokens, I believe"，那个 "I believe" 提醒我们这两个数字是凭记忆说的，不宜当成精确配置；这张图的价值在于它展示了「先高分辨率铺满超参数空间，再找最小值」这一纯经验路线，与前面 μP「通过重参数化让最优值不动」形成对照。

## 三条观察：凸性、缩放趋势与鲁棒性边界 [34:46 - 38:31]

[34:46]–[35:19] 讲者给出第一条观察，屏幕上 "Observation 1: loss over batch/LR are convex" 一页把证据摆得很细：上排四张小图各自固定一个 Learning Rate（图上方分别标着 3.431e-04、8.010e-04、5.916e-04、9.715e-04 一类的值），横轴是 Batch Size，纵轴是 Validation Loss，散点连成清晰的 U 形；右侧配一张 3D 损失曲面。下排四张小图反过来各自固定一个 Batch Size（如 252144、70144.??、184325 等标注），横轴改为 Learning Rate，纵轴仍是 Validation Loss，同样是 U 形，右侧也有一张标注 "4.85-axis slice" 的 3D 曲面。页面底部一句粗体结论："For pre-training losses, minimizers for LR/batch can be clearly identified"（对预训练损失，LR/batch 的最小点可以被清晰辨识）。讲者的解读是："each slice is kind of nice and convex and it's almost even smooth"，因此 "you can, you know, kind of search for the optimal hyperparameter fairly nicely and you can be confident that the minima is not too far off"；反过来，"This was very jaggedy, you might have some, you know, very serious doubts about whether this kind of program of gridding the spaces even viable"（如果曲线锯齿状，就会严重怀疑网格化这套做法是否可行）。这里的直觉很朴素：凸且近似光滑意味着局部搜索不会掉进假极小值，网格分辨率带来的 quantization error 也只造成小偏差；这正好回扣了本段开头 [19:19] DeepSeek 学习率拟合「像是任意直线」的抱怨——问题不在幂律本身，而在最优点定位的精度。

[35:19]–[36:43] 是第二条观察，也是讲者认为 "one really interesting general scaling trend"、且 "I don't know if I've seen any contradicting evidence to this observation" 的部分。屏幕上 "Observation 2: scaling trends" 一页有两张 log-log 散点图：左图纵轴 Learning Rate（$10^{-1}$ 量级），横轴 $D$（$10^{10}$–$10^{11}$）；右图纵轴 Batch Size（$10^4$–$10^5$），横轴同为 $D$。两图共用图例，按模型规模给出七种颜色：N=59M、N=119M、N=214M、N=268M、N=429M、N=536M、N=1B；左图每种颜色各有一条带置信带的拟合直线，右图所有颜色的点却落在同一条粉色拟合带附近。两图下方合用一行 caption："(a) Scaling laws with dataset size for different number of parameters"。页面底部粗体结论是 "For chinchilla style joint scaling: batch is primarily dependent on dataset size."，其下一行小字补充："They also find higher optimal LR with D (for fixed M), but this is likely more fragile if swapping to WSD – see e.g. InternLM scaling law paper (Zhou+ 2026)"。这条小字是画面独有、转写没念到的：它明确把「最优学习率随 $D$ 上升」标注为可能脆弱，并指向换成 WSD 调度时会出问题的 InternLM 工作。

![StepFun 的最优学习率与批大小随数据量 D 变化的 scaling trends 双图](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0030_00-36-27.jpg)

讲者把两图的差别讲得很清楚。右图是关键："if you're looking at the optimal batch size, right? The only thing that seems to depend on is the total amount of data that you're training on"，不同颜色的点（不同模型）"seem to follow roughly the same trend line here, so fairly predictable one"；而且 "this is in log-log scale, so you should expect some sort of a power law as a function of D"。结合上一章表格里 Ours (Step Law) 的批大小式，可以把它写成

$$B_{\text{opt}} = 0.58\, D^{0.571},\qquad \eta_{\text{opt}} = 1.79\, N^{-0.713} D^{0.307}$$

左图则 "a very different behavior"："Bigger models, so if you go from blue all the way to this pink color, right? Bigger models have smaller optimal learning rate and larger amounts of data have higher optimal learning rate, which is kind of counterintuitive, actually." 即模型越大最优学习率越低，数据越多最优学习率越高；后者反直觉，讲者立刻补一句 "And so this might be a little fragile. There are other kinds of papers that argue that the dependence on D should be kind of reversed"（有别的论文认为对 $D$ 的依赖应当反过来），只是 "the trends here, at least on this experiment, are quite clear"。这里的易错点是把 $D^{0.307}$ 当成稳定规律：它与 $\mathcal{L}$、$C$、$N$、$D$ 各家公式的不一致是同一类问题，讲者和幻灯片都标了 fragile；同时 $B_{\text{opt}}$ 只依赖 $D$ 这一结论是在 chinchilla style joint scaling（$N$ 与 $D$ 按算力协同缩放）下得到的，不能随意外推到任意固定 $N$ 的设定。

[36:43]–[38:02] 是第三条观察，标题里的问号很说明问题："Observation 3 (?) robustness"。屏幕分两组。第一组 "Generalization to MoEs" 是三张等高线小图，Figure 7 的 caption 原文为 "Validation loss landscapes of MoE models under varying sparsity ratios ($N_a/N$). Left: Low sparsity ($N_a/N = 0.27$). Middle: Medium sparsity ($N_a/N = 0.58$). Right: Medium sparsity at D=8.0B. Our method consistently approximates global minima across sparsity regimes." 这里 $N_a/N$ 是激活参数与总参数之比，三张图分别对应低稀疏度 0.27、中等稀疏度 0.58、以及 $D = 8.0$B 下的中等稀疏度，结论是该方法在不同稀疏度区间都能近似全局最小点。第二组 "Generalization to other datasets" 也是三张等高线图，分别标着 (a) Bilingual Corpus、(b) Code Integration、(c) Code-Dominante，用来检验换数据配方后损失景观是否仍被同一组预测命中。讲者的口头版更直白：黄色星星 "are the predictions from their scaling laws"，红色叉 "are sort of the optima that are derived, uh, sort of directly on the MOEs"；只要 "roughly controlling for active parameters"，换成 MoE 后 "you don't end up with a very different scaling law"。这与上一章 Kimi K2 / Hunyuan 的 MoE scaling 形成呼应：激活参数才是与优化超参数对齐的那个量。

但同一段里讲者马上给出反面边界："the one thing that does seem a little bit more, you know, potentially dicey is, you know, as you shift the training data, the optimal learning rates and the optimal batch sizes do shift a little bit"（换训练数据后最优学习率与批大小会有点偏移），因此 "all of these phenomena are likely contingent"，"the specific numbers that are are in these scaling laws and so on, um, are potentially contingent to the data that you train on, um, more so than others"（这些缩放律里的具体数字很可能依训练数据而定，而且比其他量更甚）。在此之前 [36:43]–[37:18] 他还做了一个现场 poll 式的插话：如果明天让你训练一个语言模型，学习率会选 1e-3 还是 1e-4？他觉得大家 "roughly have a sense" 哪个范围是好起点，而 "that is generally true"——学习率整体相当 robust（鲁棒），这些 scaling law "actually transfer over to, for example, MOEs to a to certain extent"。所以鲁棒性是有层次的：跨架构（dense→MoE，控制激活参数）较好，跨数据配方较差，具体指数与常数最不可移植。这也给第 2 章 Qwen 的 standard recipe 一个迟到的限定。

[38:03]–[38:31] 讲者说 "I want to flash this back up"，把前面那张公式表重新调出来收尾，给出一个极简版的 scaling law：批大小 "maybe should be roughly square root of the number of data times some constant"，即

$$B_{\text{opt}} \approx c\sqrt{D},\qquad \eta_{\text{opt}} \propto \frac{D^{0.307}}{N^{0.713}} \ \text{（随 batch / 数据上升，随模型规模下降）}$$

口头上他的措辞是 "Your learning rate should kind of scale, you know, upwards with batch size or upwards with data, but downwards with the model size. And that's kind of a surprising effect, but you know, not maybe crazy because"，转写恰好在 because 处截断。sqrt 形式与 $D^{0.571}$ 在指数上是一致的（$0.571 \approx 1/2$），这正是把 $0.58D^{0.571}$ 口语化成「数据量的平方根乘常数」的由来。本章的局限必须连同这个截断一起记住：讲者没有把 "not maybe crazy because" 后面的理由说完，笔记不应替他补出解释；同时这个简洁形式里的 $c$、$0.307$、$0.713$ 都来自特定数据配方下的拟合，前面刚说过换数据就会漂移。把「平方根」当成普适常数、把 $N$ 的负指数当成已证实的优化理论，都超出了画面与转写能支持的范围。

## Chinchilla 缩放下的学习率与批大小收尾 [38:31 - 39:21]

本段开头是上一段 StepFun 超参数缩放讨论的收尾，讲师专门回头澄清那句 "Chinchilla comment"。所谓 Chinchilla-like scaling（Chinchilla 式缩放），指的是参数量 $N$ 与训练 token 数 $D$ 不再各自独立设定，而是都由总计算预算 $C$ 驱动：给定一份 compute，人们按 Chinchilla 的最优比例同时放大模型和数据。于是拟合出来的最优学习率 $\eta_{\text{opt}}$ 与最优批大小 $B_{\text{opt}}$ 原本是两个变量的函数，在 Chinchilla 假设下就被压缩成计算量的单变量函数。讲师的原话是 "N and D are sort of both driven by compute"，并强调 "as compute goes up, you expect your learning rate to go down"。直觉上，模型越大越需要更保守的步长，而数据越多、可并行样本越多，批大小就可以越大；两个方向相反的依赖关系在按 Chinchilla 同步放大时会部分抵消，最终留下一个随 $C$ 单调的趋势。这也是为什么讲师在 [38:31] 会说 "those two would kind of cancel"。

把前面拟合出的关系写在一起，StepFun 的 Step Law 给出
$$\eta_{\text{opt}} = 1.79\,N^{-0.713}D^{0.307},\qquad B_{\text{opt}} = 0.58\,D^{0.571}$$
其中 $N$ 是参数量，$D$ 是数据量。若按 Chinchilla 让 $N$、$D$ 同随 $C$ 增长，则可写成
$$\eta_{\text{opt}} \propto C^{-\alpha},\qquad B_{\text{opt}} \propto C^{\beta},\qquad \alpha>0,\ \beta>0$$
这正是讲师所说的 "the learning rate decreases with compute, and the batch size increases with compute"。作为对照，DeepSeek 直接以计算量拟合的结果是
$$\eta_{\text{opt}} = 0.3118\,C^{-0.1250},\qquad B_{\text{opt}} = 0.2920\,C^{0.3271}$$
其中 $C$ 是 non-embedding training FLOPs。讲师特别指出，沿 Chinchilla 路线得到的规律 "albeit with different exponents than the DeepSeek law"（指数与 DeepSeek 律不同），因此两个式子只能看作同一趋势的两种参数化，不能把系数 $0.3118$、$0.2920$ 与 $1.79$、$0.58$ 混用，更不能把 $-0.1250$ 与 $0.3271$ 直接套到以 $N$、$D$ 为输入的拟合上。这里的具体数字本身也提醒我们：不同研究给出的函数形式连自变量都不一样，谈"学习率随规模怎么变"之前必须先说清是哪个规模变量。

![Observation 3 鲁棒性幻灯片：MoE 三种稀疏度与三种数据配方下的验证损失等高线图](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0032_00-39-07.jpg)

本段对应的画面是 "Observation 3 (?) robustness" 这张幻灯片，它正是把上述拟合放到不同设定下检验鲁棒性。上半部分是 "Generalization to MoEs"，Figure 7 的图注写着 "Validation loss landscapes of MoE models under varying sparsity ratios ($N_a/N$)"，三张等高线图分别标注 Left: Low sparsity ($N_a/N = 0.27$)、Middle: Medium sparsity ($N_a/N = 0.58$)、Right: Medium sparsity at D=8.0B，并断言 "Our method consistently approximates global minima across sparsity regimes"：横轴是 Learning Rate、纵轴是 Batch Size，彩色等高线勾出验证损失地形，红点与三角等标记落在低谷附近，右侧还配了颜色条。下半部分 "Generalization to other datasets" 给出三张同样结构的图，分别标注 (a) Bilingual Corpus、(b) Code Integration、(c) Code-Dominante。也就是说，拟合出的学习率与批大小规律至少在稀疏度变化（0.27、0.58，以及 D=8.0B 的中等稀疏设定）和数据配方变化时被逐一核对过，而不是只看一条曲线好看就外推。这种"换一个设定再画一遍损失地形"的做法，是判断一组超参数拟合能否迁移的最低限度证据。

讲师在 [39:05] 把这段定位为 "the newest and most large-scale version of the DeepSeek analysis"，并说 "this is one way that, if you were doing the scaling law-style approach, you would approach this problem of fitting a learning rate"——把超参数当作可拟合、可外推的对象，是 scaling law 路线处理 learning rate 的一种做法，而不是唯一做法。紧接着 [39:21] 一句 "that's the hyperparameter story" 正式为整个超参数部分画上句号，后面转入 optimizers。这里最容易出错的地方有两个：其一，讲师自己两次重复 "hopefully that Chinchilla comment was clear"、"What I mean is..."，说明"$N$、$D$ 都被 compute 驱动后相互抵消"的论证很容易被听成"学习率不随规模变化"，而实际上抵消之后仍留下随 $C$ 下降的趋势；其二，把不同论文的公式当作同一条普适真理直接代入自己的训练配置，忽略它们输入变量与系数的差别，正是后面问答里被明确质疑的做法。这两点也为紧接着的"要不要自己重做网格搜索"埋下了伏笔。

## 直接套用缩放律还是自己网格搜索：Step-law 表格与「vibes」 [39:21 - 41:39]

[39:56] 有学生提问：既然已经有了这些 scaling laws，是不是应该 "just take the scaling laws that we have and then apply them, rather than running our own grid search"？讲师先把问题复述给线上听众，然后给出一个不绝对的回答："it does kind of depend on the amount of compute budget that you have, and how much you need to nail it"。也就是说，决策取决于两件事：手里有多少计算预算，以及这个超参数需要调得多精确。若所处的计算区间与原论文做网格搜索的区间接近（"close to where they were running the grid"），StepFun 的 Step Law "these are probably really good defaults"，讲师甚至说 "I would bet on these over like something I pick pull out of my hat"。换句话说，在合适的 regime 里，一条别人拟合好的幂律比我随手拍脑袋选的学习率可信得多。

但如果自己要做一次 big pre-training run，情况就不同了。"you are probably going to have minor differences"：有人会说 "I really believe we should have a big weight decay. My architecture has a big weight decay"，而 "we don't know necessarily that those things are going to scale the same"。架构、正则强度、数据配方上的细微差别，都可能让别人的最优点偏离自己的最优点。讲师据此解释了前面展示过的那些技术报告为什么要 "redoing Chinchilla, redoing all these things"：不是因为不信原论文，而是 "there's enough minor differences that people want to check and make sure that they're like roughly first order correct with these results"——只要确认一阶趋势大体正确就够了，未必需要每个数字都对齐。这个"重做一遍"的成本，本质上就是为自己的实验设置买一份保险。

![课堂上投影的 Core question 幻灯片：各家学习率与批大小缩放公式及相对误差对比表](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0033_00-40-20.jpg)

这一问答对应的画面是课堂上投影的蓝色幻灯片 "Core question – scaling rates on LR and batch"，副标题是 "What are the right variables / functional form?"，下面是一张把各家公式并列的表格，列依次为 Name、Data Recipe、Model Sparsity、Learning Rate、Batch Size、Relative Error。逐行看：OpenAI Law [20] 的学习率是 $3.239\cdot10^{-3} + 1.395\cdot10^{-4}\log(N)$，Relative Error 为 9.51%；Microsoft Law [2] 是 $3.1192e^{-3}N^{-0.22}D^{-0.32}$，9.25%；DeepSeek Law [6] 是 $0.3118C^{-0.1250}$ 与 $0.2920C^{0.3271}$，9.26%；Porian Law [26] 是 $3.7N^{-0.36}$ 与 $0.7576N^{0.703}$，3.71%；MiniCPM Law [18] 与 MeiTuan Law [36] 也各有条目（后者写作 $\lambda_B C^{\alpha_B}$ 一类形式）；Ours (Step Law) 则是 $1.79N^{-0.713}D^{0.307}$ 与 $0.58D^{0.571}$，Relative Error 仅 0.94%。Data Recipe 与 Model Sparsity 两列用 ✓ 与 ✗ 标记，只有 Ours 两列同时打勾，说明它把数据配方和 MoE 稀疏度显式纳入了拟合。表格下方 "Different views:" 列出三种视角："Critical batch: batch as a function of loss (OpenAI)"、"Compute power law: poly function of compute (DeepSeek)"、"... Or something else?"。把这张表与前面 Chinchilla 收尾对照可以看出，"该用哪些变量、该用哪种函数形式"本身就是一个尚未收敛的开放问题，而不是一条已经写死的定律。

讲师由此给出本段最锋利的判断：scaling laws "have this very scientific feel to them"，听起来像 "fit these lines and extrapolate them"，但 "ultimately, a big part of scaling laws is still vibes"。他把它落成一个无法回答的问题："Do I really believe that the experiment setting of the StepFun survey is similar enough to mine that they will transfer? We can't possibly know, right? There are always tiny differences"。表格里 0.94% 与 9.51%、9.25%、9.26%、3.71% 的对比，只是在各作者自己的数据上量出来的相对误差，不能当成跨配置迁移的保证；把公式里的 $N$、$D$、$C$ 直接替换成自己的数字，却忽略数据配方、稀疏度、weight decay 的差异，正是最容易踩的坑。讲师还预告 "I think this will come up again in this next optimizer part"——后面关于 Muon 与优化器比较的整段故事，本质上就是这个"实验设置相似性"问题的放大版：小规模上量到的优势，究竟有多少能迁移到大规模，谁也无法先验保证。

## 优化器与 scale：nanoGPT speedrun 里的 Muon [41:39 - 43:27]

[41:39] 讲师正式转入 optimizers，并解释了为什么把它放在这一讲：架构那讲 "was already a pretty packed lecture"，但不讲优化器会是 "a disservice to you"；更关键的是 "optimizers are scale-dependent, so this is a good place to put them"，正好可以谈 "some of the latest and greatest developments in optimizers and their relationship with scale"。他补充说过去一两年 LLM 优化器、乃至通用深度学习优化器都有 "really exciting developments"，"this is a exciting time to talk about them"。因此这一小节不是补一堂优化器入门课，而是把优化器当作一个"随规模变化的对象"来研究：同一算法在小模型和大模型上是否同样有效，本身就是 scaling 问题。

动机故事从一张左右对比图讲起。画面标题是 "Optimizers and scale"，左图来自 nanoGPT speedrun——正是启发课程 assignment 1 的那个 benchmark：一个 "really teeny tiny model"，训练时间 "a fairly short amount of time"，目标是 "get to I forgot 3.something loss as quickly as possible"，很多人在上面 "hill climbing"。图中 Muon 是紫色线，Adam 是蓝色线，讲师特地说明 "It's right underneath the green line over there"（蓝线几乎压在绿线下方）。只比较绿色与紫色之间的 gap，就会得出 "lots of big gains from just changing the optimizer to this Muon thing"。更关键的一句是 "it's not even slower than Adam, much slower than that"：换优化器带来的收益并没有以显著变慢的单步时间为代价，这在 wallclock 计时的 benchmark 上尤其重要。

![Optimizers and scale 幻灯片：nanoGPT speedrun 各优化器验证损失曲线与相对 AdamW 加速比](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0035_00-42-44.jpg)

左图标题为 "Optimizer comparison by time (NanoGPT speedrun)"，纵轴 Validation loss 从 4.1 降到 3.3，横轴 Wallclock time on 8xH100 从 0 到 25，图例把每步耗时直接标在名字后面：

```text
Adam                               139ms/step
DistributedShampoo (UpdateFreq=10) 179ms/step
DistributedShampoo (UpdateFreq=32) 154ms/step
SOAP                               301ms/step
Muon                               142ms/step
```

底部还有一行小字 "*SOAP is under active development. Future versions will significantly improve the wallclock overhead."。于是 Muon 的 142ms/step 与 Adam 的 139ms/step 几乎相同，却能更快把 $\mathcal{L}_{\text{val}}$ 压到约 3.28，这才是"只换 optimizer 就拿到大增益"的完整含义。可以把它形式化为 wallclock 意义下的加速比
$$\text{speedup} = \frac{T_{\text{Adam}}(\mathcal{L}_0)}{T_{\text{Muon}}(\mathcal{L}_0)},\qquad \mathcal{L}_0 \approx 3.3$$
即达到同一目标损失所需的墙上时间之比，而不是单步 FLOPs 之比；正因横轴是墙上时间，139ms/step 与 142ms/step 这组数字才必须一并读。

右图 "Speedup vs Model Size (8x Chinchilla)" 立刻把故事拉回 scale：纵轴 "Speedup w.r.t. AdamW" 从 1.0 到 1.5，横轴 Model Size 是 130M、300M、520M、1.2B 四个点；AdamW 按定义恒为 1.0，Muon 与 Soap 从约 1.4 一路降到约 1.1，NAdamW 从约 1.18 降到 1.07 再回到 1.1，图上注释框写着 "Optimizers' speedup w.r.t. AdamW decreases with model size"。整页底部是一句加粗判断 "Optimization is a core part of LLMs (but also tricky, due to scale dependence!)"。讲师随即抛出真正的问题："is this thing actually good at scale?"——我们有一个小规模实验，有一个 "scale-dependent for sure" 的对象（optimizer），看到大增益之后该怎么做？他明确说，与其给 "a solid concrete prescription"，不如给一个 story，让读者体会 "what we can do and what things we can't necessarily know"。易错点也就在这里：把 8×H100 上 25 个时间单位内的小模型曲线直接当成大规模结论，或忽略 SOAP 那行"仍在开发、wallclock 开销会显著改善"的注解就把它判死刑，都属于过度解读。右图那条递减曲线，正是下一章"两个缩放轴"与后面 Muon 争议的直接伏笔。

## 公平比较优化器的第一道陷阱：超参数调优 [43:27 - 45:00]

讲师先说 "this kind of thing happens all the time"：小规模上想出的好东西，大规模 "just doesn't maybe pan out"。接下来的几张幻灯片来自 Kaiyue Wen、Tengyu Ma、Percy Liang、David Hall 的论文 "Fantastic Pretraining Optimizers and Where to Find Them"，画面右上角列出四位作者与 Stanford University，还附有 kaiyuew@stanford.edu、davidh@cs.stanford.edu 等邮箱。这篇工作做的是 "a large-scale set of experiments on comparing optimizers"，试图把优化器之间的比较做得更严谨；讲师说 "there's a lot to learn about scaling and other things from that. But also just standard deep learning stuff"——它同时贡献了 scaling 的教训和非常基础的深度学习实验教训。把这两类教训分开，是读懂后面几张图的前提。

第一道陷阱（Problem 1 – hyper tuning is often off）其实与规模无关，却足以毁掉任何结论：不同模型需要不同的超参数，"if you're scaling things up, maybe they even have different scaling exponents"，所以 "you have to tune Adam really well"。左图 "Loss on C4/EN on 130M models" 的纵轴 Loss 从 4.0 到 3.5，横轴 Step 从 2000 到 10000，四条曲线分别是 AdamW w/ lr 6e-4（棕色）、Mars（蓝色）、Nesterov AdamW（淡紫）、AdamW w/ lr 8e-3（红色）。棕色那条把 AdamW 显得 "much worse than all these other algorithms that I have over here on the left"，但只要把学习率从 $6\times10^{-4}$ 改成 $8\times10^{-3}$，红色线在约 5000 步就追上棕色线 10000 步才到达的损失，图上的注释框直接写 "Tuning LR of AdamW leads to 2x speedup"。换句话说，前面看到的"算法差距"里有整整 2 倍来自调参质量之差。把这两个设定并排写出来就是
$$\eta_{\text{bad}} = 6\times10^{-4},\qquad \eta_{\text{good}} = 8\times10^{-3},\qquad \text{speedup} \approx 2\times$$

![Problem 1 幻灯片：130M 模型上学习率对损失曲线的影响，以及 Lion 与 AdamW 的 weight decay 扫描](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0036_00-43-56.jpg)

右图 "C4/EN Loss vs Weight Decay" 说明同一件事在另一个旋钮上重演：纵轴 Loss 只有 3.32 到 3.37 的窄区间，横轴 Weight Decay 从 0.0 到 1.0，绿线是 Lion、红线是 AdamW，注释框写着 "wd ≈ 0.6 is optimal for Lion"，箭头指向 0.6 附近的低谷。如果沿用给 AdamW 调出来的同一个 weight decay（例如偏小的值），Lion 的绿线就会 "get a much worse result"；只有为它单独选到约 0.6 才算公平比较。页面下方的总结是 "Different optimizers can require different hyperparameters! (and likely have different optimal hyperparameter scaling)"。讲师特别补了一句定位：这 "technically not a scaling thing, but a very important, you know, general empirical machine learning thing"——它属于通用实证方法论，而非 scaling law 的内容，但正因如此才更容易被做 scaling 研究的人忽略。

把两张图合起来看，比较优化器时的基线必须是"每个算法都跑在自己的最优超参上"，否则量到的是超参数调优质量之差，而不是算法之差；括号里那句 "different optimal hyperparameter scaling" 更进一层：即使各自调好了，最优学习率、最优 weight decay 随规模变化的指数也可能不同，于是小模型上调好的配置放大后未必仍最优。这正好接上第 2 章讲师的解释——为什么那么多技术报告要 "redoing Chinchilla, redoing all these things"，也解释了为什么本讲一开始要花大力气拟合 $\eta_{\text{opt}}$、$B_{\text{opt}}$ 的缩放规律。常见的误读有两种：把深棕色线当作 "AdamW 天生不如新算法" 的证据写进结论，或者把 Lion 在错误 weight decay 下的高损失当成算法缺陷。这些偏差与算法本身无关，却会在小规模实验里被放大成"显著增益"，再被下一章的 scale dependence 进一步放大。

## 算法开发的两个缩放轴：compute 与 Chinchilla ratio [45:00 - 47:23]

[45:00] 讲师把讨论收束成一句方法论："there are two variables that are always important when thinking about scale. And this paper exposes to most extent like both of them"，所以这篇优化器对比论文是 "a good case study and example of how to really think about algorithm development as a function of scale"。第一个轴是 compute：若固定 model size to data ratio（模型大小与数据量之比），"the two are coupled together, then this is essentially a compute axis on the x-axis here"。换句话说，只要按比例同步放大 $N$ 与 $D$，横轴上标着的"模型大小"其实就等价于计算量，因此 "whenever you're doing a study, you essentially scale up the compute and look at the trends as a function of scale"。这与第 1 章 Chinchilla 收尾里的逻辑是同一套：把两个变量绑成一个变量，趋势才看得清。

具体看相对加速比。画面 "Problem 2 – fairly significant scale dependence" 的左图仍是 "Speedup vs Model Size (8x Chinchilla)"：纵轴 Speedup w.r.t. AdamW 从 1.0 到 1.5，横轴是 130M、300M、520M、1.2B 四个点；AdamW 按定义恒为 1.0，Muon 与 Soap 在 130M 处约 1.4、300M 处约 1.25、520M 处约 1.26 到 1.29、1.2B 处约 1.1，NAdamW 从约 1.18 降到 1.07 再回到 1.1，注释框写着 "Optimizers' speedup w.r.t. AdamW decreases with model size"。讲师的评价很直白："MuON is really good at small scale and that gain sort of diminishing down diminishes down as we increase in scale. That's not really necessarily the picture that you want to see." 用公式写就是
$$\text{speedup}(N) = \frac{T_{\text{AdamW}}(N)}{T_{\text{opt}}(N)},\qquad \text{speedup}(N)\downarrow \ \text{as}\ N\uparrow$$
其中 $T$ 是达到同一损失所需的步数或时间，衡量的是相对优势而非绝对损失。这里最容易被混淆的一点是：speedup 递减并不等于 Muon 的绝对损失变差，只说明 AdamW 追上来了；把它读成"Muon 在大规模失效"就过度了。

第二个轴是 Chinchilla ratio，即数据量与模型参数量的比值
$$r = \frac{D}{N}$$
讲师连说三个 "really, really, really important parameter"。原因是有两类算法各占一端：有些算法在 over-parameterized（"a way bigger model than you have lots of data"）的情形下表现好，"maybe your algorithm adds some implicit regularization"，或者 "it is data inefficient but it works well because of compute resources"；另一些则在大 Chinchilla ratio（"lots of data per unit parameter"）下占优，"because you can pack knowledge more efficiently into the parameters"。于是 "This is a big, big, big confounder"：即便在 model size 轴上有 "pretty good hygiene" 的论文，也常常 "neglect to or just don't have the compute" 去专门研究 token-to-parameter ratio，"and this turns out to be surprisingly important a lot of time"。换句话说，$r$ 不是可有可无的实验细节，而是能改变算法排序的隐藏维度；第 2 章表格里 Data Recipe 与 Model Sparsity 那两列 ✓/✗，正是各家对这类隐藏维度处理程度的写照。

![Problem 2 幻灯片：优化器加速比随模型规模递减，以及 520M 模型在不同 Chinchilla Ratio 下的损失](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0038_00-46-21.jpg)

右图 "C4/EN Loss Scaling For 520M Models" 把第二个轴画了出来：纵轴 Loss 从 3.10 降到 2.90，横轴 Chinchilla Ratio 取 1、2、4、8；图例里 AdamW、Mars、NAdamW 是虚线（dashed），Muon、Soap、Kron 是实线（solid），注释框给出一条图上独有、转写里并没有逐字念到的规律 "Matrix-based optimizers (solid) consistently outperform scalar-based optimizers (dashed)"——即基于矩阵的优化器持续优于基于标量的优化器。页面底部写着 "General algorithm development note – always check scaling with respect to compute and chinchilla ratios. These are often major confounders to performance!"。讲师给了一个重要的边界说明：在这项研究里 "the optimizer sort of gains are pretty consistent, the ratios between all the lines are the same at different Chinchilla ratios"，但立刻补上 "but this is not always the case"，并重复 "You always want to be vigilant about both of these scaling factors"。易错点因此有两层：只扫模型大小而把 $r$ 固定，会漏掉一个巨大的混淆变量；反过来，看到本例各条线彼此平行就以为 Chinchilla ratio 无关紧要，则是把"这一次恰好一致"误当成了普遍规律。这两个轴构成的框架，正是下一章判断"一条漂亮的 scaling 曲线是否可信"的参照系。

## 漂亮的 scaling law 也会突然 blow up [47:23 - 49:16]

[47:23] 讲师把话题推到更难的一层："if you have a new algorithm or a new setting, right? establishing scaling is quite non-trivial"。例子取自 Will Held，他当时与 Marin 合作——讲师称其为 "Percy's open-source language model training project"。他们想证明的甚至不是新算法，而是 "a particular set of hyper parameters"：Cautious AdamC（一个 Adam 变体）、"square root batch size scaling" 的学习率，外加 "a bunch of other sort of standard looking things"，做的仍是 "essentially the same sort of Chinchilla analysis that you've seen all the other papers do"。关键在于 "this is not necessarily a paper, this is a bunch of people doing experiments, you don't just publish the successful one"——作为一组做实验的人，他们有 "the niceness of publishing their failed runs"，也就是把失败的运行也一并公开出来。这一点决定了整张幻灯片的价值：我们能看到一条曲线是怎么从"漂亮"走向"崩掉"的，而不是只看到挑剩下的成功样本。

![Problem 2.5 幻灯片：IsoFLOP 抛物线族与外推段上逐渐恶化直至发散的缩放曲线](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0039_00-47-33.jpg)

画面标题是 "Problem 2.5 – establishing scaling is nontrivial!"，副标题 "Sometimes, a good looking scaling can blow up"。左图 "IsoFLOP parabolas per compute bucket"：纵轴 Paloma macro loss 从 3.0 到 3.8，横轴 Training tokens 标着 1B、10B、100B，图例按 Compute (FLOPs) 列出 3e18、9e18、1.8e19、3e19、9e19、1.8e20、3e20 共七档，每一档是一条抛物线，最低点用 × 标出，正是标准 Chinchilla isoflops 分析里取 $\min_N L(N;C)$ 的做法。右图 "Scaling law + hold-out validation"：纵轴 "Paloma macro loss at IsoFLOP optimum" 从 2.2 到 3.8，横轴 Compute (FLOPs) 是对数轴 10¹⁹ 到 10²³，一条虚线在约 10²⁰ 处标出 "fit ← → extrapolation" 的分界；虚线右侧依次标注 0.8% worse、2.5% worse，以及一个偏离虚线很远、旁边写着 "Run Diverged" 的 × 点。底部注明 "Cautious AdamC + Sqrt batch-size scaling of learning rates"，括号里给出修复方式 "(Fixed with some more careful parametrization / scaling / optimizer changes)"，右下角还给出来源为 William Held 的 Delphi 博客。

讲师按图讲故事：到那条 "10 to the 20-something" 的竖直虚线为止，"this is a really beautiful looking scaling law to here"，然后 "it's a little bit worse, it's a lot worse and then wow, now that run is not so good anymore"，最后 "it's a totally blown up on you"。他的结论是 "even good-looking scaling trends for many orders of magnitude can suddenly kind of bite you"，"it can deviate from linear and then it can blow up immediately"。这里被验证的核心设定可以写成
$$\eta \propto \sqrt{B}\quad(\text{sqrt batch-size scaling}),\qquad C \in [10^{19},\,10^{23}]\ \text{FLOPs}$$
即学习率随批大小的平方根缩放，再在对数计算量轴上拟合一条直线并外推；出问题的恰恰是外推段而非拟合段。修复办法不是继续调那条直线，而是 "picking more careful μP style parameterizations"、"optimizer changes"、"they switch to a pretty a different kind of optimizer"，之后才 "get nicer scaling across boarder orders of magnitude"。这与第 2 章表格里 "What are the right variables / functional form?" 的开放问题遥相呼应：函数形式或参数化选错，小范围内看不出来，一跨数量级就暴露。

讲师随即把矛头指回自己前面的建议："when I go back here and I say, 'Okay, well, you should all do scaling experiments, you should scale all your learning rates and so on.' Um that's not easy, right? You end up in fairly non-trivial fraction of the time with plots that look like this, um and it's all kind of horrible and you're not quite sure what has gone wrong." 这与第 5 章的"两轴都要查"并不矛盾：即使沿 compute 和 Chinchilla ratio 都扫过，外推仍可能在更大数量级上失效。易错点非常具体：把 hold-out 之前的线性拟合当作保证、只发表成功 run 造成幸存者偏差、把 0.8% 与 2.5% 的偏离当成噪声而忽略紧随其后的发散、以及误以为换一组超参数就能救回一条已经偏离的曲线。后面讲 Muon 时提到 Kimi K2 为 "prevent MuON from like blowing up on you" 加的 bells and whistles、以及他们 "found a whole bunch of instability issues"，正是这一类外推失效问题在真实大模型训练中的回响。

## Muon 详解：矩阵参数的 Newton–Schulz5 正交化 [49:16 - 54:44]

[49:21] 起讲师用几张幻灯片收束 Muon 这条线，先解释为什么要再讲它：因为它 "is beginning to be integrated into a large-scale model training runs"，而课程筛选新课题的 "inclusion criteria" 就是 "has it been in a big training run?"，"μon now checks that mark"；同时 "it's also just kind of an interesting algorithm"。他从最普通的梯度下降讲起：先忽略伪代码第 5 行，剩下的就是 "roughly a standard momentum-based optimizer"——取一个梯度 $G_t$，"add it to B of T"，用参数 $\mu$ 延续上一时刻的动量，再据此更新参数，"Very standard momentum gradient stuff"。把基线讲清楚，是为了让第 5 行那个改动显得突兀而具体。

真正 "out-of-the-box" 的想法是：梯度把所有参数一视同仁，"gradients have this property that they treat all the parameters are the same"，但语言模型乃至一般深度网络里的参数并不相同。"Some parameters are like vector parameters, like they belong to the RMS norm. Some parameters are matrix parameters, like for the attention or for MLPs"，而 "matrix parameters are probably different than vector parameters"。矩阵有 spectra，有特征值、奇异值，"you can do things to the singular values"。Muon 的做法是：算出梯度、做一次 momentum step 得到 $B_t$，但不直接把 $B_t$ 用到第 6 行的更新上，而是先做 "Newton-Schultz thing, Newton-Schultz five to be specific"，把 $B_t$ 正交化。

![A few slides on muon 幻灯片：Algorithm 2 Muon 完整伪代码与 nanoGPT speedrun 优化器对比曲线](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0041_00-49-57.jpg)

画面 "A few slides on muon.." 给出 Algorithm 2 Muon 的完整原文：

```text
Algorithm 2 Muon
Require: Learning rate η, momentum μ
 1: Initialize B_0 ← 0
 2: for t = 1, ... do
 3:     Compute gradient G_t ← ∇_θ L_t(θ_{t−1})
 4:     B_t ← μ B_{t−1} + G_t
 5:     O_t ← NewtonSchulz5(B_t)
 6:     Update parameters θ_t ← θ_{t−1} − η O_t
 7: end for
 8: return θ_t
```

页面下方写着 "Optimizer for 'matrix valued' parameters." 以及 "NewtonSchultz (approximately) orthogonalizes the matrix $B_t = USV^T \to UV^T$"。把它写成数学形式：对 $B_t$ 做奇异值分解 $B_t = U S V^T$，"make all of my singular values one"——"if my singular values are really big, I'll shrink them back to unit. If they're really small, I'll expand them back up to being unit"——再取 $UV^T$ 作为新的更新量：
$$B_t = U S V^{T} \;\longrightarrow\; O_t = U V^{T},\qquad \theta_t \leftarrow \theta_{t-1} - \eta\, O_t,\qquad O_t = \text{NewtonSchulz5}(B_t)$$
注意伪代码里第 4 行用的是 $\mu B_{t-1} + G_t$（动量加当前梯度），第 6 行的更新量是 $O_t$ 而不是 $B_t$，这两个细节决定了它与普通 momentum 的全部差别。

讲师给的直觉是与 Adagrad / Adam 对照：在 Adagrad 或 Adam 里 "you're going to divide by kind of the size of the gradient"，于是 "every coordinate is roughly going to be the same size"；而 "in MuON, you are operating in the spectral norm instead, so every kind of direction is going to be unit size"。即
$$\text{Adam: } g_i / |g_i|\ \text{逐坐标归一化};\qquad \text{Muon: } S \to I\ \text{逐谱方向单位化}$$
关键局限是 "notice that this only makes sense for a matrix"：奇异值分解只能对矩阵值 $B_t$ 定义，"I can only define this notion of a singular value decomposition for a matrix value B of T"，所以 Muon 是 "a matrix optimizer"，只对矩阵值参数使用；向量值参数（如 RMSNorm 的参数）"you might still use something like AdamW, right? Cuz those don't really have any sort of interaction with the Newton-Schultz term"。此外讲师补了一句系统层面的说明："Newton-Schultz is a approximation to that operation that only uses matrix multiplies, right? So, it's systems efficient and not exactly the orthogonalization"——$USV^T\to UV^T$ 只是概念描述，真实实现是有限次矩阵乘迭代，并不等于精确正交化。

![Muon and scaling 幻灯片：NanoGPT speedrun、Scaling study 与 Kimi K2 三段训练曲线并排](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0044_00-53-34.jpg)

为什么花这么多时间讲它？"it's very powerful for things like the nanoGPT speed run. It was discovered for this"，但前面的 scaling 研究 "argued that the gain for MuON sort of decreases as scale goes up"；讲师本以为故事已经结束——Kai Yu 等人放大后说 "maybe it doesn't work super well"，他当时想 "No one will spend the big compute to scale this guy up to the full pre-training size"——结果 "Kimmy K2 comes out, and that whole model is trained with MuON, like fully with MuON with a few bells and whistles to prevent MuON from like blowing up on you"，他们 "found a whole bunch of instability issues"，而 "Kimmy K2 is an outstandingly good model"，训练曲线 "look reasonable"。画面 "Muon and scaling" 把三段并排：左为 "NanoGPT speedrun (very small!)"（Optimizer comparison by time，Validation loss 4.1 到 3.3，Wallclock time on 8xH100 从 0 到 25，图例带 139、179、154、301、142 ms/step）；中为 "Scaling study"（Speedup vs Model Size (8x Chinchilla)，纵轴 1.0 到 1.5，横轴 130M、300M、520M、1.2B，注释 "Optimizers' speedup w.r.t. AdamW decreases with model size"）；右为 "Kimi K2" 的蓝色损失曲线，纵轴 Loss 从 2.0 降到约 1.3，横轴 Tokens (Trillion) 标到 14，曲线先快速落到约 1.6 再缓慢下行，带大量高频毛刺。底部一句 "Scaling gains are tricky to measure, but clearly muon 'works' at scale"。

讲师给的结论非常克制："MuON worked at scale. Like is it better than Adam? Kimmy K2 certainly doesn't have any ablation, so we don't know at that scale, but it is certainly a valid and workable solution." 整段落点是 "it's really hard to know whether something works at scale. It's very, very hard"，但 "a lot of like small scale experiments and transferring them up to large scale has been the way by which we do science"，所以仍要 "respect things like the Nano GPT speed run"，"We learn quite a bit from that and get good ideas, and then eventually they make their way to the big models"；至于 "is MuON better than AdamW at scale and so on"，要等有人在大尺度做 ablation，"Those are interesting questions to which we don't yet have the answers"。易错点有三：把 $USV^T\to UV^T$ 当成实现步骤（实际是 Newton–Schulz5 近似）、把 Muon 用到向量参数上（SVD 无从定义）、以及从 Kimi K2 的成功反推"Muon 在该规模优于 AdamW"——没有同规模 ablation，这一步在逻辑上推不出来。

## 优化器收尾问答与 μP 引子 [54:44 - 57:47]

[54:44] 讲师停顿确认大家是否跟上，因为 "I was talking about MuON and that's a whole new can of worms that I haven't talked about much before"。第一个提问是 "is SVD fast on GPUs?"，回答直接澄清了上一章留下的系统层面细节："SVD is not fast on GPUs, but they're not doing SVD. What they're doing is Newton-Schulz. Newton-Schulz is a matrix multiply only finite iteration algorithm that approximates the process of orthogonalizing a matrix." 讲师把 Muon 拆成 "several clever pieces"：其一是 "the idea of like treating it as a matrix on its own"（把更新量当作矩阵而非一堆独立标量），其二是正交化这个想法本身，其三是 "a systems piece of doing it with Newton-Schulz"。用公式概括，实际更新是有限 $T$ 次矩阵乘迭代（伪代码里写作 $\text{NewtonSchulz5}$，即 $T=5$）：
$$O_t = \text{NewtonSchulz5}(B_t) \approx U V^{T},\qquad \theta_t \leftarrow \theta_{t-1} - \eta\, O_t$$
而 $\text{SVD}(B_t)=USV^T$ 在 GPU 上并不便宜，所以"近似"既是数值选择，也是让算法能落到真实训练里的系统选择。

第二个提问是 "What are the hyperparameters for MuON here?"，回答是 "Adam for the AdamW ones"，并补一句 "No, they're they're different"——矩阵参数与向量参数各自有独立的超参数，并不共享一套配置。由此引向 μP style 的工作：讲师提到 "several people that have been working on this like muP style line of work"，"one of them I think like Jeremy Bernstein, he's been saying like actually like every layer should have its own learning rate"，甚至更极端 "every layer should have its own optimizer"；"the ultimate limit of this is to say every parameter in a transformer is different, why not have a different optimizer for each one of them"。学生立刻接话 "And so you would have to do the hyper parameters for each one of them as well"，讲师答 "Which I do not want to do, but you know, that's maybe the future of optimization"。写成形式化记号，就是逐层乃至逐参数的学习率 $\eta_l$、$\eta_i$：
$$\theta_i \leftarrow \theta_i - \eta_i\, O_i,\qquad \eta_i \ \text{对每个参数独立设定}$$
需要强调：这只是被讨论的"极限方向"，不是本段给出的可执行配方，讲师明确表示自己不想这么干。

第三个提问最贴近本讲主线：这些拟合 "they're only learning rate and batch size"，"do these things like interact with each other?"，也就是只网格这两个量、再从中挑最好的，是否忽略了与其他超参数的交互。讲师承认 "in general, yes, you are worried about hyper parameter interactions"，但 "realistically they grow exponentially, right? Like this is not a thing where you can grid out the whole space"。若 $d$ 个超参数各取 $k$ 个候选，组合数是
$$N_{\text{grid}} = O(k^{d})$$
因此现实做法是 "they grid the stuff that's most concerning and most sensitive"："learning rates are the most important thing to tune, so they very aggressively go on that"，而 "stuff like weight decay, they might do a univariate sweep afterwards, right? To be locally optimal with respect to those parameters"。这与第 4 章 Lion 的 $\text{wd}\approx0.6$ 恰好互证：weight decay 靠单变量扫描补调，而不是与学习率一起做全网格；也解释了为什么第 2 章表格里各家只报 $\eta_{\text{opt}}$ 与 $B_{\text{opt}}$ 两个量的拟合——不是因为别的超参数不重要，而是全空间网格根本不可行。

[57:22] 起讲师宣布 "the final part of this lecture is I want to talk about muP"，并给出对它的定性："muP is a bit of a mysterious object. Even with many papers written about it and sort of in fact even many implementations, I don't think all of them even agree on what the underlying math is or what the underlying implementation should be." 但 "there's a core set of ideas that are shared and it does seem generally effective"，所以接下来讲的是 "my interpretation of the whole program"。本段到 [57:47] 停在这个引子上。易错点集中在三处：把 Newton–Schulz 误当成 GPU 上跑的精确 SVD、以为超参数可以全空间网格（实际组合指数增长，只能对最敏感量做网格再加单变量补扫）、以及把"每层乃至每个参数一个优化器"当成当下可落地的建议。这一章同时把本段两条线收拢：SVD 与 Newton–Schulz 的问答补完了第 7 章的系统细节，超参数交互的问答回扣了第 2 章的 "vibes" 与第 4 章的调参陷阱，而 μP 则把"超参数随规模漂移"这个从本段一开始就存在的问题交给课程最后一部分处理——它与第 6 章里 "more careful μP style parameterizations" 作为 blow up 修复手段的说法正好接上。

## 目标设定与 Cerebras GPT 的规模验证 [57:47 - 60:39]

讲者先把这一段要玩的「游戏」(the game) 讲清楚：当我们增大模型规模时——本段只讨论 width（宽度）这一个维度——最优学习率 (optimal learning rate) 通常会随之漂移，而理想情况是「无论模型多大，最优学习率都保持在同一个位置」。他直说「this picture is the goal」，即整段推导与实证都围绕这张图展开。为了达成这个目标，他愿意调节的旋钮 (knobs) 有三类：逐层的初始化 (initializations at a per layer basis)、逐参数的学习率 (learning rates on a per parameter basis)，以及按模型大小把残差连接 (residual connections) 放大或缩小。换句话说，μP（maximum update parametrization，最大更新参数化）不是去拟合最优学习率怎么变，而是改写参数化方式，让最优学习率干脆不变；这与前面若干讲里「直接拟合超参数 scaling law」的路线形成对照，也是本段所有公式的动机。

![Maximum update parametrization 深入页，左为标准参数化下最优学习率随宽度漂移、右为 μP 下最优稳定，并附 Table 2 的 μP 缩放函数表](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0048_00-58-28.jpg)

对应画面标题是「Maximum update parametrization – in depth」，副标题「Recall – the maximum update parametrization makes appealing claims」，底部压着一句「Scale-invariant hyperparameter tuning would be very nice. How does it work, and does it work in practice?」，正好把本段拆成「原理」与「实证」两问。两幅曲线都以 $\log_2\text{LearningRate}$ 为横轴（刻度 $-20$ 到 $-10$）、Training Loss 为纵轴（刻度 $3.5$ 到 $7.0$）：左图「Standard Practice」的图例列出 Width 为 128、256、512、1024、2048、4096、8192 共七档，各条 U 形曲线的最低点明显错开，箭头标注「optimum shifts」；右图「Our Work」里各宽度曲线的最小值收拢到同一处，箭头标注「optimum stable」。也即是说，标准参数化下宽度每翻一倍、最优学习率就要重新搜，μP 下则可以从小模型直接迁移。

画面右上角的 Table 2 是转写完全没有念到、却把 μP 规则说得很细的独有信息。表头说明写的是「μP function for a model $M'$ that is $r$ times the widths of $M$」：若一个参数张量有两个维度、且当模型宽度趋于无穷时它是「matrix-like」（例如全连接隐藏层），就按矩阵规则缩放；若该数是 1 或 0 则归入「others」，并特别注明 embedding layers 属于「others」；「Output」指把无穷维映射到有限维的层，在 Transformer 里就是词解码层 lm_head；multiplier（乘子）是乘在参数张量上的常数，作用类似 softmax temperature。表格原文可照抄为：

```text
Hyperparameter (weight)              M        M' ~ r
AdamW learning rate (matrix-like)    l        l/r
AdamW learning rate (others)         l        l
Initialization variance (matrix-like) σ       σ/r
Initialization variance (others)      σ        σ
Multiplier (output)                  τ        τ/r
Multiplier (others)                  τ        τ
```

即当宽度乘以 $r$，matrix-like 权重的 AdamW 学习率与初始化方差都除以 $r$，output 乘子除以 $r$，而 others 与 output 之外的乘子保持原值。这里的 $l$、$\sigma$、$\tau$ 分别是基准学习率、初始化标准差与乘子，除以 $r$ 正是后面要推导的 fan-out/fan-in 缩放的表格化写法。

实证方面，讲者点名 Cerebras：它本质是一家芯片公司 (chip company)，同时也有语言模型训练部门，由「Hestness of the scaling law paper fame」领衔，近年在写不少关于 μP scaling 的论文。Cerebras GPT 用通常的 Chinchilla recipe 训练了 0.1 到 13B 的模型，并额外加入一个 muP variant 来更有效地调超参数。他们发现，一旦用 muP parametrization，scaling law 的拟合明显更稳定：projected muP trends 对实际模型几乎「right on the money」，而 non-μP 模型的 loss 预测波动大得多，于是给出「MUP is maybe a really good idea」的结论——这是在真实规模上被验证过的 (validated at scale)，MiniCPM 等许多工作也都训练过 μP variants。与后文的关系是：这些「appealing claims」是否真站得住，要靠后面两条不变量的推导和独立的压力测试来回答；易错点在于只调 width 是极强的简化，depth、batch size、optimizer 都未纳入这套断言，把 0.1–13B 上的稳定性无条件外推到任意配置并不成立。

最后讲者交代文献地图，为下一章的概念推导铺路。他说与其罗列「用 μP 训练过的模型」，不如讲 conceptual foundations 与数学：Greg Yang 开创这一研究方向的 Tensor Programs 系列他认为 somewhat inscrutable（相当难读）；Jeremy Bernstein（同样做过 μP 工作）写过一篇用不同框架重述的 review paper，very accessible；另有物理学家写的几篇解释 μP 的论文，Cerebras 的人也做了一点。画面方框里给出的正是那篇被推荐的入门论文「A Spectral Condition for Feature Learning」，作者 Greg Yang (xAI)、James B. Simon (UC Berkeley & Imbue)、Jeremy Bernstein (MIT)，旁注「this is a very accessible 'muP for babies' paper」。这一节因此完成从「目标与信念」到「可推导的数学」的过渡。

## 两条不变量断言与条件 A1 的初始化推导 [60:40 - 65:17]

回到这篇入门论文，μP 的核心基于两个断言 (assertions)，讲者称这是「a very physicist way of thinking about this」：只改变网络宽度、取 scaling limit，然后断言有两条不变量 (invariants) 必须成立。第一条 A1：初始化时的激活 (activations at initialization，即喂进随机数据后的各层输出) 应保持大致同样的大小；如果激活随网络宽度 blow up（爆炸）或 shrink to zero（缩到零），就说明参数化选错了。第二条 A2：走一步梯度之后，激活的变化应为 $\Theta(1)$，也就是人们说的 feature learning（特征学习）——网络在一步更新里确实发生了显著变化，且这个变化量应是宽度的固定函数。画面上写的是「A1: The activations at initialization should remain $\Theta(1)$」「A2: After one gradient step, the change in activation should be $\Theta(1)$」，用 $\Theta$（同阶）比转写口头的 O(1) 更精确，这也提醒读者：μP 要的是量级恰好，而不是仅仅有上界。

给有理论背景的同学一个对照：neural tangent kernels（NTK，神经正切核）区制的行为完全不同，激活的变化会随网络宽度 vanish（趋于零），网络在无限宽极限下几乎不学特征，这正是 μP 不想要的。讲者还反复强调要在「单个激活」与「整层向量的范数」之间来回切换：若每个激活是 $\Theta(1)$，那么含 $n_l$ 个隐藏单元的层，其激活向量范数应当是 $\Theta(\sqrt{n_l})$——画面 Note 的原文是「if individual activations are $\Theta(1)$, then the norm should be $\Theta(\sqrt{n_l})$」。这个 $\sqrt{n_l}$ 是后面所有归纳与量级追踪的基准尺度，条件 A1 与 A2 其实都在说「激活和激活的变化应随宽度保持相似」。

μP 有意思的地方在于：从这样一组很「可口」(palatable) 的假设出发，再叠若干更强的假设，最后却掉出一组非常简单的 scaling rules。条件 A1 的推导从一个 simple deep linear network 起步，前向为 $h_l = W_l h_{l-1}$，权重按缩放高斯初始化 $W_l \sim N(0, \sigma^2 I_{n_l \times n_{l-1}})$。由标准随机矩阵的 matrix concentration（关于 operator norm 的集中性论证），其算子范数满足

$$\|W_l\|_* \to \sigma\left(\sqrt{n_{l-1}} + \sqrt{n_l}\right), \qquad \|h_l\|_2 \approx \|W_l\|_* \|h_{l-1}\|_2.$$

讲者说明这个 $\approx$ 始终是上界，只在 fan-out 不比 fan-in 大太多的高维区制下近似取等；幻灯片底部还留了一句注释「Comments – this is a kind of 'worst case' derivation – the ≈ is an upper bound」，明确承认这是最坏情形式的推导。接下来要做的，就是解出 $\sigma$，使每层、每个宽度下的激活尺度都不变。

他先给出一个 ansatz（先假设答案、再验证归纳），取

$$\sigma = \frac{\sqrt{n_l}}{\sqrt{n_{l-1}}}\left(\sqrt{n_l} + \sqrt{n_{l-1}}\right)^{-1} = \Theta\left(\frac{1}{\sqrt{n_{l-1}}}\min\left(1, \sqrt{\frac{n_l}{n_{l-1}}}\right)\right).$$

归纳假设 (inductive assumption) 是 $\|h_{l-1}\|_2 = \Theta(\sqrt{n_{l-1}})$；归纳情形 (inductive case) 代入上式得 $\|W_l\|_* \to \sigma(\sqrt{n_{l-1}}+\sqrt{n_l}) = \frac{\sqrt{n_l}}{\sqrt{n_{l-1}}}$；再把两者相乘，得到

$$\|h_l\|_2 = \sqrt{n_l} + o(\sqrt{n_l}).$$

也就是按这个噪声初始化，每一层激活向量的范数都约等于该层隐藏单元数的平方根，加上不打算追踪的低阶项。讲者自评这个 $\sigma$ 像是「drawing it out of a hat」（凭空掏出来的），但只要把它代回公式，归纳就自洽，条件 A1「pretty simple」地得证。这里的具体参数是 $n_{l-1}$（fan-in）与 $n_l$（fan-out），以及 $\min(1,\sqrt{n_l/n_{l-1}})$ 这个在 fan-out 小于 fan-in 时才起作用的因子。

![Deriving muP (condition A1)：深度线性网络中由矩阵集中性解出 σ 并做归纳的完整公式页](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0051_01-02-36.jpg)

与前后内容的关系很清楚：A1 只解决初始化时的激活尺度，讲者明说「condition A2 is a little bit hairier」，下一章要处理权重更新后的激活变化；而本章末尾得到的 $\sqrt{n_l}$ 正是下一章判定「各项是否同阶」的标尺。易错点有三处：其一，这套推导只覆盖 deep linear network 加高斯初始化，真实 Transformer 有非线性、归一化与残差，把 $\sqrt{n_l}$ 当成普适结论会出错；其二，务必区分「单个激活 $\Theta(1)$」与「向量范数 $\Theta(\sqrt{n_l})$」，混用会让数量级差一整个 $\sqrt{n}$；其三，$\approx$ 是上界、fan-out 不远超 fan-in 是附加近似，讲者本人也把这些称作 mostly legitimate 而非严格成立。

## 条件 A2：rank-one 更新、三项分解与量级追踪 [65:18 - 68:47]

条件 A2 要处理 updates（权重更新）。为了让推导可解，讲者把设定压到最简：只考虑 a single example in a batch、SGD、以及 deep linear network（转写原话「single example in a batch for SGD for linear neural network」）。在这个设定下，某层 $l$ 的权重变化是「损失梯度」与「激活」之间的 rank-one（秩一）外积：前向是一次 rank-one 乘法，反向自然得到一个 rank-one outer product。幻灯片给出的精确形式是

$$\Delta W_l = -\eta_l \nabla_{h_l}\ell \; h_{l-1}^{\top}, \qquad \|\Delta W_l h_{l-1}\|_2 = \|\Delta W_l\|_* \|h_{l-1}\|_2.$$

这里的 $\eta_l$ 是第 $l$ 层的学习率，$\nabla_{h_l}\ell$ 是损失对该层激活的梯度，$h_{l-1}^\top$ 是前层激活的转置；$\|\cdot\|_*$ 表示 operator norm（算子范数 / 谱范数）。这个秩一结构是整套量级分析能进行下去的关键，因为它把矩阵范数与向量范数直接联系起来。

接下来把 $h_l = W_l h_{l-1}$ 展开，得到激活变化的精确表达式，幻灯片写作

$$\Delta h_l = W_l \Delta h_{l-1} + \Delta W_l\left(h_{l-1} + \Delta h_{l-1}\right).$$

展开后其实有三类项：$W_l\Delta h_{l-1}$ 是「前一层激活变化直接传上来」的 direct term（直接项）；$\Delta W_l h_{l-1}$ 是「因为权重本身变了、而不只是因为前层激活变了」的 cross term（交叉项）；$\Delta W_l\Delta h_{l-1}$ 是两个小量相乘的二阶项。讲者说这一步「I just take H equals W times HL minus one, and I expand this out」，随后要逐项检查这三个项的量级，希望每一项都与 $\sqrt{n_l}$ 同阶。之所以要 $\sqrt{n_l}$，是因为 A2 要求一步梯度后激活变化为 $\Theta(1)$（转写里也提到 $\omega(1)$），对应整层范数就必须是 $\sqrt{n_l}$；只要三项同阶且没有抵消，总量级就对了。

画面上「Assuming that the leading order terms don't cancel, we see that」下面给出的三条量级结论，逐条转述为：第一项 $W_l\Delta h_{l-1} = \Theta(\sqrt{n_l})$，由归纳假设加上条件 A1 的论证直接得来，属于「comes for free」；第二项 $\Delta W_l h_{l-1} = \|\Delta W_l\|_* \sqrt{n_{l-1}}$，为了让左边等于 $\sqrt{n_l}$，可解出 $\|\Delta W_l\|_* = \Theta\left(\frac{\sqrt{n_l}}{\sqrt{n_{l-1}}}\right)$；第三项 $\Delta W_l \Delta h_{l-1} = O(\|\Delta W_l\|_* \sqrt{n_{l-1}})$，是受前两项支配的高阶小量。讲者随后「go back one slide」把逻辑理清：第一项是归纳假设白送的，另外两项都含 $\|\Delta W_l\|_*$ 乘 $\sqrt{n_{l-1}}$，于是反解出权重更新算子范数必须按 $\sqrt{\text{fan-out}}/\sqrt{\text{fan-in}}$ 缩放，这就是下一章要挑学习率去满足的目标。

![Deriving muP (condition A2)：SGD 下 rank-one 权重更新 ΔW_l 与激活变化 Δh_l 三项展开及量级 bullet](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0053_01-05-21.jpg)

讲者在这里专门插了一段方法上的自我定位，值得当作局限来记：他说「the right thing way to think about this is not like, okay, Tatsu is sitting here doing like rigorous mathematics」，自己做的只是 keep track of all the orders of magnitude——「It's like kind of physicist math, um no offense to physicists」，而且这套方法本来就是物理学家以这种方式发展起来的。也就是说，no cancellations（主项不抵消）、leading order terms 成立、$\approx$ 当上界，都是 sketchy 的假设；结论是「给定这些假设，各项大概该多大」，不是定理。与前后内容的关系：本章把 A2 从一句口号变成可解的约束 $\|\Delta W_l\|_* \sim \sqrt{n_l}/\sqrt{n_{l-1}}$，下一章正是要问「what kinds of learning rates are going to give me an update that's roughly scaled as the square root of the fan out over the square root of the fan in」。易错点在于这四条简化假设——单样本、SGD、线性网络、秩一更新——任何一条被破坏（换成 mini-batch、Adam、非线性或归一化层），$\Delta W_l$ 都不再是简单外积，结论必须重新推导；后面 Adam 得到 $1/n_{l-1}$ 而不是 $n_l/n_{l-1}$，正是这种差异的直接体现。

## 损失 O(1) 假设、逐层学习率求解与 μP 规则总览 [68:48 - 71:43]

要真正解出学习率，还得再压一个假设，讲者自称这是「probably the least palatable of the assumptions」：假设损失更新 (loss update) 也是 $\Theta(1)$ 量级，即模型随规模确实在做 appreciable non-trivial progress（可观、非平凡的进展），而且这个进展量大致不随宽度变化。他坦白「Not quite sure that's palatable to me」，但压下去之后「the rest of the math works out quite cleanly」。这一步的意义是给 Taylor 展开提供一个已知的左边，从而把未知的学习率反解出来；它也是整条推导链里最主观、最难从第一原理辩护的一环。

由 Taylor 近似，损失变化约等于「梯度」与「权重变化」的内积；再因为 $\Delta W_l$ 是 rank-one，可以把 Frobenius norm 改写成 operator norm，幻灯片写成三连等式

$$\Delta\ell \approx \Theta\left(\langle \Delta W_l, \nabla_{W_l}\ell\rangle\right) = \Theta\left(\|\Delta W_l\|_F \|\nabla_{W_l}\ell\|_F\right) = \Theta\left(\|\Delta W_l\|_* \|\nabla_{W_l}\ell\|_*\right),$$

其中用到标准 SGD 的 $\Delta W_l = -\eta\nabla_{W_l}\ell$。把「损失更新 $\Delta\ell = O(1)$」与上一章解出的 $\|\Delta W_l\|_* = \Theta\left(\frac{\sqrt{n_l}}{\sqrt{n_{l-1}}}\right)$ 代进去，就得到梯度范数 $\|\nabla_{W_l}\ell\|_* = \Theta\left(\frac{\sqrt{n_{l-1}}}{\sqrt{n_l}}\right)$。再把上一章的秩一式 $\Delta W_l = -\eta_l \nabla_{h_l}\ell \, h_{l-1}^\top$ 与 $\Delta W_l = -\eta \nabla_{W_l}\ell$ 对照，最终解出逐层学习率

$$\eta_l = \Theta\left(\frac{n_l}{n_{l-1}}\right),$$

即 fan-out / fan-in 之比。转写里讲者说这是「the ratio of these two guys, which is $n_L$ over $n_{L-1}$」：从「左边 $O(1)$」这一个假设出发，把所有量逐个解出，掉出来的正是一个 fan-out over fan-in ratio 作为该层的学习率。幻灯片右下角还补了一句 with Adam 时 $\|\Delta W_l\|_*\sqrt{n_{l-1}} = \Theta(\sqrt{n_l})$ 的对应变化；转写也顺带提到，Adam 因为其中一些项改变，会得到 $1/n_{l-1}$ scaling。

![Deriving muP (condition A2) part 2：假设 Δℓ=O(1) 后经 Taylor 展开与 Frobenius/operator norm 解出 η_l=Θ(n_l/n_{l-1})](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0056_01-09-49.jpg)

紧接着的「muP mini recap」把整段推导收束成一张可执行的对照表，这是转写只零散提到、画面却写得极完整的一页。开头一句「So, what is (baby) muP about? Controlling activations (and changes) via $W$ and $\Delta W$」点题：μP 就是通过控制权重 $W$ 与权重变化 $\Delta W$ 来控制激活及其变化。μP 一侧：Initialization (stddev) Set to $\Theta\left(\frac{1}{\sqrt{n_{l-1}}}\min\left(1, \sqrt{\frac{n_l}{n_{l-1}}}\right)\right)$，Learning rates Set to $\frac{n_l}{n_{l-1}}$（旁注 for Adam $\frac{1}{n_{l-1}}$）；与之对照的「standard」parametrizations 一侧：Initialization Set to $\frac{1}{\sqrt{n_{l-1}}}$，Learning rates Set to $\Theta(1)$。底部一行「Differences – LR changes for Adam, also init diffs when fanout $n_l$ < fanin」把差异说尽：Adam 下学习率规则不同，且只有 fan-out 小于 fan-in 时初始化才明显不同。

![muP mini recap：μP 与标准参数化在初始化标准差、学习率（含 Adam 的 1/n_{l-1}）上的逐项对照](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0057_01-11-26.jpg)

转写与这张表逐条对得上：初始化是 $\frac{1}{\sqrt{n_{L-1}}}$ 乘上 $\frac{1}{\sqrt{\text{fan-out}/\text{fan-in}}}$ 那一支，若该比值为 1 就等价于 usual / standard parameterization，若比值不小于 1 则更接近 $\sqrt{\text{fan-out}/\text{fan-in}}$；学习率标准做法是常数（$\Theta(1)$），μP 则设为 fan-out/fan-in；Adam 下丢掉 $1/n_l$ 那一项，得到 $1/\text{fan-in}$ 的逐层学习率，于是 fan-in 大的层学习率更小，形成「layer adaptive learning rates」这种相当有意思的层自适应学习率。与前后关系：本章是整段推导的落点，把前面所有量级追踪变成可落地的初始化与学习率数值；下一章转向「这套推导方式本身意味着什么」以及Transformer 上的实证。易错点：fan-out/fan-in 接近 1 时 μP 与标准参数化几乎无差别，别期待凭空收益；$\min(1,\sqrt{n_l/n_{l-1}})$ 说明只有 fan-out 明显小于 fan-in 时初始化才分叉；SGD 的 $n_l/n_{l-1}$ 与 Adam 的 $1/n_{l-1}$ 不能混用；讲者也明说 Adam 的推导「I'm not going to even touch the derivation for that」，属于留白而非已证结论。

## 方法论意义与 transformer 逐组件 μP 缩放表 [71:44 - 74:20]

讲者说推导细节很多，但 high-level point 更值得讲：μP 作为一个 argument（论证方式）本身很有意思，不只是「有实际有用的东西」，而是「the way in which you get to the algorithm is itself interesting」。它的套路是：取网络的 scaling limit，断言若干不变量，再附加额外假设，然后反推超参数必须满足的约束，而这些约束恰好又给出超参数自身的 scaling limits。他称这是「a general principle for coming up with different kinds of algorithms or hyperparameter scalings」，并强调这「probably very different than what you're used to in CS or ML training」——不是网格搜索、不是经验调参，而是从极限下的不变性倒推规则。这与本章之前所有公式形成呼应：前面算的是具体量级，这里点明的是可复用的方法论。

随后讲者要用几件 empirical things 收尾 μP 部分，其中一项是一位 independent researcher（独立研究者）对 μP 做各种 stress testing（压力测试）的工作。μP 可以被看作「a procedure for tuning hyperparameters」（一套调超参数的流程）：以 Adam 为例，网络不同组件会各自有关于宽度的学习率函数——embedding parameter、attention parameters、input/output MLPs、softmax linears，在 μP 以及 non-μP scaling rules 下都会有不同的缩放。该论文在不同规模的语言模型上复现了一批 μP 结果，并得到 MiniCPM 等的 headline result：只要把 μP 配对、且只以非常受控的方式缩放，就能得到 learning rate optimality invariance（最优学习率不变性）；baseline μP 以及 attention projection biases 的 μP 变体都 transfer very nicely（迁移得很好）。

对应的幻灯片是「Deeper dive into muP」，右上角贴着 preprint 小框「A Large-Scale Exploration of μ-Transfer」，作者 Lucas Dax Lingle（邮箱 lucasdaxlingle@gmail.com）；副标题「Recall – muP is a scaling procedure for hyperparams (as a function of width)」。Table 2「μP scaling rules for transformers; a rule for attention scale is detailed in text below」把抽象的 fan-out/fan-in 规则落到每一个权重矩阵上，表头为 Param | Init Variance (Θ) | Adam LR (Θ) | Init Variance (Exact) | Adam LR (Exact)，右侧按 Embedding / Attention params / Input/output MLP MM / Softmax linear 分组，原文数值照抄如下：

```text
Param    InitVar(Θ)  AdamLR(Θ)  InitVar(Exact)  AdamLR(Exact)   分组
W^E          1          1            1              α            Embedding
W^AQ       1/M        1/M          1/M            αP/M          Attention params
W^AK       1/M        1/M          1/M            αP/M          Attention params
W^AV       1/M        1/M          1/M            αP/M          Attention params
W^AO      1/(HD)    1/(HD)        1/M            αP/M          Attention params
W^FI       1/M        1/M          1/M            αP/M          Input/output MLP MM
W^FO       1/F        1/F        0.25/M           αP/M          Input/output MLP MM
W^U       1/M^2       1/M         1/M^2           αP/M          Softmax linear
```

其中 $M$ 是模型宽度，$P$、$H$、$D$、$F$、$\alpha$ 是论文里的超参数（画面未逐一给出定义，故此处不臆测其含义）。可以看到不同组件的 Init Variance 从 $1$、$1/M$、$1/(HD)$、$1/F$ 到 $1/M^2$ 不等，Adam LR 也分 $1$、$1/M$、$1/(HD)$、$1/F$、$1/M$ 几档，说明 μP 从来不是一条公式，而是一张逐组件的缩放表。

![Deeper dive into muP：transformers 的 μP 缩放规则 Table 2，按 Embedding、Attention、MLP、Softmax linear 分组](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0058_01-13-02.jpg)

表格下方还有一段转写未念到、却点出理论与实践落差的说明：In addition, μP uses an attention scale of $\tau^{-1} = \Theta(1/D)$ instead of the usual $\tau^{-1} = 1/\sqrt{D}$；为简单起见作者取 $\tau^{-1} = 1/D$，因为预备实验里用更小的 $1/D$ 倍数只带来很小改进；并特别注明，当 $D$ 在各模型宽度 $M$ 上固定时，任何常数 $\tau^{-1} \neq 0$ 在技术上都不违反 μP（Yang et al., 2021），但实验会显示 $\tau^{-1}$ 对 performance and transfer（性能与迁移）有重大影响。这是「理论不禁止、实践极敏感」的典型例子，也与上一章「Adam 规则另有一套」的留白相呼应。与前后关系：本章把前几章的通用规则实例化到 Transformer 的具体权重上，下一章则要检验哪些现实组件会真正破坏它。易错点：Exact 列与 Θ 列并不总一致（$W^{AO}$ 的 Init Variance 在 Θ 列是 $1/(HD)$、Exact 列却是 $1/M$；$W^{FO}$ 的 Exact 列带具体系数 $0.25/M$），说明 $\Theta$ 只给量级、落地还要带常数；更不能把「任意非零常数 $\tau^{-1}$ 都合规」误读成「$\tau^{-1}$ 取多少都无所谓」。

## μP 鲁棒性压力测试与片尾的 scaling in the wild [74:21 - 77:03]

讲者接着问一个很实际的问题：现实里有很多 μP 理论上不处理的东西——technically Swiglu does not fit into μP（严格说 SwiGLU 不在 μP 框架内），initialization variations（初始化变体）与 RMS norm 也不在理论内，若真去改 optimizer 同样会把它弄乱。那么这些「理论不允许」的东西里，究竟哪些会真的 break μP？答案是 for the most part，这些 mostly work with μP（大多仍能正常工作）。这正好接上一章的逐组件表格：μP 的 $\Theta$ 量级规则对不少非理想组件仍有鲁棒性，理论覆盖面窄不等于实践会立刻失效。

真正会破坏 μP 的因素，讲者点了三类。第一，学习 RMSNorm 的 gain term 会 break μP，不过「these can be removed in many cases without necessarily hurting you」，所以去掉 gain 也许「not such a bad idea」。第二，更 exotic 的优化器如 Lion，它 rely on the sign of the gradient（依赖梯度符号）；sign gradient 在精神上类似 μP 及同类想法，因此推测还有别的东西也会破坏 μP。第三，最令人担心 (the thing that seems most concerning) 的是 large decoupled weight decay（大强度的解耦权重衰减），会导致 significant μP failure，「this is maybe the one stress test that it does fail」。

![What is muP not robust to? RMSnorm gain：Baseline μP 与可学习 RMSNorm gain 在 128/512/2048 宽度下的最优 Base LR 迁移对照表](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0059_01-14-38.jpg)

对应幻灯片标题「What is muP not robust to? RMSnorm gain」，副标题「In our arch – RMSNorm has learnable gains. This turns out to break muP」。表头为 Ablation | Width | Base LR（五列 $2^{-10}, 2^{-8}, 2^{-6}, 2^{-4}, 2^{-2}$）| Transfer，数字（加粗者为各行最小值）如下：

```text
Ablation                 Width   2^-10   2^-8    2^-6    2^-4    2^-2    Transfer
Baseline μP              128    3.846   3.743   3.695   3.884   4.143     ✓
                         512    3.114   2.993   2.953   3.221   3.506
                         2048   2.711   2.553   2.511*  2.563   3.244
RMSNorm Gains (Vector)   128    3.842   3.744   3.689   3.670*  3.681     ✗
                         512    3.101   2.992   2.951   2.950*  3.412
                         2048   2.692   2.553*  2.609   2.605   3.169
RMSNorm Gains (Scalar)   128    3.843   3.749   3.692   3.670*  4.471     ✗
                         512    3.106   3.000   2.961   2.959*  3.515
                         2048   2.704   2.570   2.525*  2.542   3.334
```

关键现象是：Baseline μP 的三个宽度（128、512、2048）最优点都落在 $2^{-6}$ 列（例如 2048 行的 $2.511$），最优学习率可迁移，打蓝色勾；一旦引入可学习 gain（Vector 或 Scalar），各宽度的最优点漂到不同列（$2^{-6}$、$2^{-4}$、$2^{-8}$ 混杂），迁移失败，打红色叉。说明段原文进一步解释：即便 RMS Normalization 前后特征的「coordinate size」按设计就是 $\Theta(1)$ w.r.t. width，用 $\Theta(1)$ 学习率缩放这些 gain 时，最优学习率依然 do not reliably transfer；此外当 base learning rate $\alpha$ 取最优时，trainable gains 会损害最大 μP 模型的质量，Section 4.5 还发现 standard transformers with trainable gains underperform μP transformers without them。底部一句「But these gains can be removed with little loss of perf..」与讲者口头的「可以移除而不太伤性能」完全对应。

收尾时讲者问「is muP useful?」，答案是若真想 stabilize your learning rates，它 actually pretty useful：有大量这类实验，增大 width 时最优学习率会 very predictably 却 very greatly 地随规模平移，而 μP 下不会。但他立刻把它放回工具箱定位——「one of the many tools in the toolkit to try to control hyperparameter drift as a function of scale」，并说「It's an interesting open area of research. It's not really done and dusted」，μP 式 initialization program 有 promise，直接 fitting scaling laws 也有 promise。最后（75:53 起）落到全讲总结：scaling in the wild is very tricky；scaling law 最初的呈现「makes it sound like a science」，仿佛画条线、走套流程就能确定放大后会发生什么，实际「a lot more messy and a lot more unknown」。人们确实拿它 picking architectures、picking optimizers、picking hyperparameters，但「there is really an art to it」——你并不真知道它会不会永远外推，只能做合理的事去 maximize the chances of success；可用手段包括 μP、搜索最优学习率等，都是控制 hyperparameter drift 的办法，然而「there's no silver bullet yet」，甚至自嘲「Maybe next year there will be module that's like we solved it but not quite yet」。

![Recap: scaling in the wild：实践中三大挑战与三类解决方案的要点页](/coursenotes/cs336-language-modeling-sp26/l11-scaling-laws-in-practice/kf_0060_01-16-38.jpg)

片尾幻灯片标题是「Recap: scaling in the wild」，把全课收束成两组要点。「What are challenges in scaling 'in practice'」其一是 Setting model arch hyperparameters (width, etc)，其二是 Setting optimizer hyperparameters (LR, batch)，其三是 Compute needed to fit the big chinchilla sweep；「Some solutions?」其一 Assume stability (or use muP)，其二 Search for optimal LR / batch in small scale, either keep fixed or predict scaling，其三 Use alternative learning schedules (WSD-like)。这三条恰好对应本讲此前的三条主线：用 μP 假设稳定、小规模网格搜索再外推、以及可复用的 WSD 式调度。与前后内容的关系：本章既是 μP 部分的边界检验，也是整讲的收束。易错点在这里最集中——把「RMSNorm 前后坐标尺寸 $\Theta(1)$」当成「学习率必然可迁移」的保证（表格正是反例）；把 scaling law 的拟合直线当成确定性预测；忽略 big chinchilla sweep 本身的计算成本；以及误以为存在一劳永逸的银弹。讲者以「Great. Thanks a lot. I can take questions afterwards.」结束全讲。

## 要点回顾

### 一、两条主线与经典结论的再验证
- 本讲把 Chinchilla（2022）之后真实开源模型的缩放实践当作「大杂烩」逐篇精读，总问题是缩放与超参数调优的最佳实践，三问分别对应验证经典结论、省算力、选参数化。
- 初始化、学习率、批大小被反复强调为 scale sensitive：$\eta^*=\eta^*(N,D)$、$B^*=B^*(N,D)$、$\text{init}=\text{init}(N)$；等算力分析即固定 $C$ 在 $N\uparrow\Leftrightarrow D\downarrow$ 之间权衡。
- 应对敏感超参数只有两条路：用 μP 之类把它们稳定到不随规模漂移，或承认它们会变、直接拟合怎么变并「just go with the scaling law」。

### 二、MiniCPM：μP 稳定学习率与缩放阶梯
- μP 的四个默认常数是 $\text{Scale\_emb}=12$、$\text{scale\_depth}=1.4$、$\text{init\_std}=0.1$、$\text{lr}=0.01$；Table 7 要求 embedding 输出乘 12、残差按 $\frac{1.4}{\sqrt{L}}$ 缩放、二维张量初始化 $\sigma_{\text{init}}=\frac{0.1}{\sqrt{d_m/d_{\text{base}}}}$、逐参数学习率 $\eta_{\text{tensor}}=\frac{\eta_{\text{global}}}{d_m/d_{\text{base}}}$、logits 除以 $d_m/d_{\text{base}}$。
- 缩放阶梯从 9M 到 0.5B 共七个模型，固定 $d_h=64$、$d_{ff}=2.5\,d_m$、层数 $L$ 从 8 递增到 24，最大阶梯与真正要训的目标模型约有 5X 差距。
- 应用 Tensor Program 后五条 LR–Loss 曲线的最低点大致对齐 $\eta^*\approx10^{-2}$，被讲者称为「a nice success case of the μP approach」，但最小模型明显略有偏移，不能当成严格不变。
- 学习率被固定不代表 $B^*$ 消失：最优批大小仍随数据与模型规模变化，临界批大小相对目标损失呈幂律 $B_{\text{crit}}(L_{\text{target}})\propto L_{\text{target}}^{-\alpha}$，画面与转写都没给出具体指数。

### 三、WSD 学习率与低成本的 Chinchilla 复现
- cosine 调度由总步数 $T$ 决定，扫 $n$ 个数据量档位需各自从头重训，成本约 $O(n^2)$；WSD 把调度拆成 warm-up（固定步数、与训练时长无关）、长 stable 段、末端 decay（约占总训练 10%–20%，降到约 $0.1\eta_{\max}$）。
- 延长训练只需回滚到稳定阶段 checkpoint 再重新衰减，成本降为 $O(nC)$，重做衰减约占 10% 开销；MiniCPM 用 6 个规模（0.04B–2B）、每个从稳定阶段 10N–60N 的 checkpoint 各做 6 次衰减，在五个留出集上评测，并按字节数而非 token 数平均损失。
- WSD 稳定段损失高于 cosine 是调度形状造成的假象，衰减后基本追平甚至偶尔超过；讲者称很多人觉得 cosine 略好，但 WSD 是不必为续训重跑的通用默认。
- MiniCPM 复现了被讲者评为最不可靠的 method 1 与 method 3，联合拟合 $L(N,D)=\frac{7.54\times10^{-2}}{N^{0.30}}+\frac{2.92\times10^{-1}}{D^{0.30}}+0.25$，在 $C=10^{21}$ 处得 $\frac{D_{\text{opt}}}{N_{\text{opt}}}=95.60$，讲者对指数差异表示「highly unclear」。

### 四、DeepSeek：直接拟合超参数缩放律
- 不用任何 μP，固定算力在 1e17（177M FLOPs/token）与 1e20（2.94B FLOPs/token）做 batch/LR 网格搜索，热力图显示泛化误差在很宽参数区间内保持稳定。
- 以「损失在最小值 0.25% 以内」收集 near optimal 点，拟合出 $\eta_{\text{opt}}=0.3118\,C^{-0.1250}$、$B_{\text{opt}}=0.2920\,C^{0.3271}$，即算力越大学习率越低、批大小越大；批大小拟合像条线，学习率那条被标「Learning rate fit looks a bit questionable」。
- 网格分辨率不足会带来 quantization error，把最优点吸附到离散档位；学生追问中同一 $C$ 出现一段最优学习率散布，讲者用「I think」推测源于模型尺寸、数据量未被固定。
- DeepSeek 用两段衰减（80%+10%+10%）的 WSD 变体复现 Chinchilla，靠 isoflops 得到比 MiniCPM 更干净的曲线：67B 对应约 $4.3\times10^{11}$ FLOPs/Token 与 $1.04\times10^{12}$ tokens、总算力约 $4.5\times10^{23}$；小规模幂律对 7B、67B 两颗星的最终损失预测「not bad at all」。

### 五、近期开源配方：MoE 稀疏度、架构与下游映射
- Qwen 2.5 / Qwen 3 把 $\mu_{\text{opt}}=\mu_{\text{opt}}(N,D)$、$B_{\text{opt}}=B_{\text{opt}}(N,D)$ 变成标准配方，覆盖 dense 44M–14B、MoE 44M–1B activated、0.8B–600B tokens；Qwen 3 又多了 learning rate scheduler 与 three pre-training stages 两个变量。
- Kimi K2 定义 $\text{sparsity}=\frac{N_{\text{experts,total}}}{N_{\text{experts,activated}}}$，固定激活 8 个、总专家 384 得稀疏度 48；验证损失 1.5 时相对稀疏度 8、16、32 分别省 1.69x、1.39x、1.15x FLOPs；Hunyuan 反过来固定稀疏度、搜出每激活参数约 96 个数据点。
- Llama 3 的 IsoFLOPs 覆盖 $6\times10^{18}$ 到 $10^{22}$ FLOPs、每档用 second degree polynomial 定位最小值、标注「39-1 ratio」；compute-to-downstream 用 sigmoid 把 NLL 映射到 accuracy，但点系统性偏离曲线，讲者不认为它是唯一真相。
- MiniMax-01 用 Chinchilla method 1 比较 Softmax、Lightning、Hybrid-lightning 三种注意力（70M–7B），结论「for the most part comparable」并据此为部署 hybrid 架构背书；「Recent scaling law recipes」把 DeepSeek（batch/LR scaling + IsoFLOP + piecewise-linear schedule）与 miniCPM（μP + 分段线性调度）收成两条配方。

### 六、StepFun 超参数网格与公式分歧
- 「Core question – scaling rates on LR and batch」表并列 OpenAI Law（批大小 $2\times10^{18}\mathcal{L}^{-4.76190}$）、Porian Law（$3.7N^{-0.36}$、$0.7576N^{0.703}$）、Ours (Step Law)（$1.79N^{-0.713}D^{0.307}$、$0.58D^{0.571}$），各家连输入变量（loss / $C$ / $N$ / $D$）都不一致。
- 三条观察：损失对 LR/batch 近似凸且光滑、最小点可清晰辨识，网格化可行；$B_{\text{opt}}$ 主要只依赖数据量 $D$，log-log 下约 $0.58D^{0.571}\approx c\sqrt{D}$，而 $\eta_{\text{opt}}$ 随模型变大而降、随数据增多而升，后者被画面小字标为 fragile。
- 鲁棒性有层次：控制激活参数后 scaling law 能迁移到 MoE（稀疏度 $N_a/N=0.27$、$0.58$ 及 $D=8.0$B 都近似全局最小），但换数据配方（Bilingual Corpus、Code Integration、Code-Dominante）后最优学习率与批大小会漂移。
- 讲者强调这些公式「pretty brittle」、别把最后一行当 gospel；Chinchilla 假设下 $N$、$D$ 都由 $C$ 驱动，可写成 $\eta_{\text{opt}}\propto C^{-\alpha}$、$B_{\text{opt}}\propto C^{\beta}$，但 $0.3118$ 与 $1.79$ 这组系数不能混用，scaling laws 仍有很重的 vibes 成分。

### 七、优化器、μP 推导与工程边界
- nanoGPT speedrun 上 Muon 的 142ms/step 对 Adam 的 139ms/step 几乎不慢，却更快把验证损失压到约 3.28；但「Speedup vs Model Size (8x Chinchilla)」显示 Muon 相对 AdamW 的加速从 130M 处约 1.4 降到 1.2B 处约 1.1。
- 公平比较要先各自调好超参：AdamW 学习率从 $6\times10^{-4}$ 改到 $8\times10^{-3}$ 就有 2x speedup，Lion 的 weight decay 需单独扫到约 0.6；两个必须查的轴是 compute 与 Chinchilla ratio $r=D/N$，而组合数 $N_{\text{grid}}=O(k^{d})$ 决定只能网格最敏感量再单变量补扫。
- Muon 对矩阵参数动量 $B_t=\mu B_{t-1}+G_t$ 做 $O_t=\text{NewtonSchulz5}(B_t)\approx UV^{T}$（Adam 逐坐标归一化、Muon 逐谱方向单位化），向量参数仍用 AdamW；Kimi K2 全量用 Muon 训练证明其大规模可行，但没有同规模 ablation，不能据此断言优于 AdamW。
- μP 由 A1（初始化激活 $\Theta(1)$、整层范数 $\Theta(\sqrt{n_l})$）与 A2（一步更新后变化 $\Theta(1)$）推出 $\eta_l=\Theta\!\left(\frac{n_l}{n_{l-1}}\right)$（Adam 为 $\frac{1}{n_{l-1}}$）与 $\sigma=\frac{\sqrt{n_l}}{\sqrt{n_{l-1}}}(\sqrt{n_l}+\sqrt{n_{l-1}})^{-1}$，Cerebras GPT 在 0.1–13B 上验证拟合更稳；RMSNorm 可学习 gain、Lion 式 sign 梯度与 large decoupled weight decay 会破坏 μP。
- 片尾把挑战归为设定架构超参（width 等）、优化器超参（LR、batch）与大 Chinchilla sweep 的算力，解法归为 assume stability（或 μP）、小规模搜最优 LR/batch 再固定或外推、以及 WSD-like 调度；讲者的落点是外推「a lot more messy and a lot more unknown」，there's no silver bullet yet。

## 视频来源

- 原始链接：https://www.youtube.com/watch?v=vTfEyOyzV9E
- 课程：Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 11

---

## 对照参考笔记补遗（Spring 2025 版）

> 编者注：本节依据 2025 版参考笔记整理，其中的「更正」小节记录的是核对时发现的问题，对应正文已在本版订正；补遗内容不代表 2026 课程视频原文。


本节内容全部来自 2025 版参考笔记（Qihong Ruan 整理的 lecture notes），仅用于与本讲 2026 视频笔记逐条对照补充，不代表 2026 视频原文，也不替代前文已经写过的任何推导与数值。

### 一、CerebrasGPT 附录里的 SP 与 μP 参数化对照，以及 μTransfer 的代理搜索

参考笔记在 CerebrasGPT 一节完整转录了论文附录的 Table 2，把 Standard Parameterization（SP，标准参数化）与 Maximal Update Parametrization（μP，最大更新参数化）逐项并排。这组经验基线是前文 μP 推导没有给出的：μP 一侧固定 $d_{\text{model,base}}=256$、$\eta_{\text{base}}=6\times10^{-3}$、$\sigma_{\text{base}}=0.08$、$m_{\text{emb}}=10.0$；SP 一侧 $\sigma_{\text{base}}=0.02$，而 $\eta_{\text{base}}$ 一栏直接写着 "Must tune for each model size"（每个模型规模都要重调）。换句话说，μP 把「基准学习率」变成可迁移的单一常数，SP 则把这份成本留给了每一次放大；这正是前文「用 μP 把最优学习率钉在原地」在工程表里的落点，两处讲的是同一件事，只是粒度不同。

真正把规则形式化的是 width multiplier（宽度乘子）$m_{\text{width}}=d_{\text{model}}/d_{\text{model,base}}$。参考给出的 μP 公式是：注意力与 MLP 的权重按 $W_{\text{qkv}}\sim N_{\text{trunc}}(0,\sigma_{\text{base}}^2/m_{\text{width}})$ 初始化、逐层学习率 $\eta_{\text{qkv}}=\eta_{\text{base}}/m_{\text{width}}$，输出投影与前馈第二层再额外除以层数，$W_{\text{o}},W_{\text{ffn2}}\sim N(0,\sigma_{\text{base}}^2/(2m_{\text{width}}n_{\text{layers}}))$，解码端 $Y_{\text{logits}}=W_{\text{unemb}}X/m_{\text{width}}$，而嵌入端反而乘上 $Y_{\text{emb}}=m_{\text{emb}}\cdot\text{embed}(X)$。SP 对应项不含 $1/m_{\text{width}}$。这些式子和前文那套由 A1/A2 推出的 $1/\sqrt{\text{fan-in}}$、$n_l/n_{l-1}$ 规则是同一个 fan-in / fan-out 逻辑的两种记号：一个按「宽度倍数」写，一个按「单层维度比」写，不可把两处的系数直接相加或混用。

与前后内容的接口在 μTransfer：CerebrasGPT 把实验先缩到约 40M 参数的 proxy model（代理模型）做充分超参搜索，再靠 μTransfer 迁移到目标规模，用 μP 保证放大过程中超参不变；参考明确指出 MiniCPM 与 DeepSeek 也采用了「小代理先搜、再放大」这一策略。这解释了前文缩放阶梯里「最大模型与真正要训的模型约 5 倍差距」为何可行——差距是被 μP 的稳定性买下来的，而不是靠运气外推。易错点有两处：其一，$m_{\text{emb}}=10.0$ 是乘在嵌入输出上的放大，与矩阵权重一律缩小方向相反，实现时最容易漏；其二，参考这一表格由 PDF 解析而来，SP 与 μP 两列的 OCR 对应关系并不处处干净，引用时应以「哪一类张量按 $1/m_{\text{width}}$ 缩放」为准，而不是死记某一格数值。

### 二、μP 在真实规模上为什么「更稳」，以及它已进入前沿实验室

前文只写了「projected μP trends 几乎 right on the money、non-μP 波动大」这一结论，参考补充了它背后的失效形态。CerebrasGPT 在 Pile 数据集上画 test loss 对 training FLOPs：蓝线是 SP 的 Cerebras-GPT，橙线是 μP 版本。SP 曲线在预测 scaling law 的虚线附近出现显著 oscillations（震荡），根因是标准参数化下最优学习率随宽度下移，每个规模都要重调，调不准就偏离预测线；μP 让超参跨规模稳定，橙线因此紧贴虚线，并且常追平或超过 Pythia、GPT-J 这两个同期开放基线。也就是说，μP 的收益不是「损失更低」，而是「可预测性更高」——把一条本该平滑的缩放曲线从调参噪声里救出来。

规模证据也值得一并记住：CerebrasGPT 训练的是 0.1 到 13B 的模型族，用通常的 Chinchilla recipe，再叠加一个 muP variant 来更有效地调超参；论文 Figure 13 用不同超参下的验证损失曲线显示 μP 让超参大体稳定。这与本讲后半段 CerebrasGPT 在 0.1–13B 上「拟合更稳」的验证是同一批实验的不同侧面：一个看「最优学习率是否随宽度漂移」，一个看「整族 loss 是否贴着预测线」。把两者合起来才完整——μP 同时改善了超参迁移与曲线可预测性，而前文只强调了后者。

更值得注意的是落地程度：参考在讲 MUP 重要性时点名，这类做法已是 frontier labs（前沿实验室）的实际实践，例如 Llama 4 使用的 "MetaP" 就是 μP 的一个变体。这把 μP 从「一篇 2021 年理论论文的推导」拉回到 2024 年后的真实大模型配方，也和前文片尾「μP 是控制 hyperparameter drift 的工具箱之一」形成呼应：它不是被供起来的理论，而是被工程化、被改名、被继续微调的一条产品线。易错点在于别把「μP 模型追平 Pythia/GPT-J」读成「μP 架构本身更强」——对照的是同一模型族在两种参数化下的可预测性，Pythia/GPT-J 只是外部参照点；同样，"MetaP" 是变体而非原样照搬，不能据此断言 Llama 4 用了本讲推导出的那组精确系数。

### 三、μP 鲁棒性压力测试的完整清单：哪些改了也没事

前文在压力测试处只列了三类会破坏 μP 的组件，容易让人误以为 μP 对现代 Transformer 的任何偏离都脆弱。参考给出的完整 ablation（消融）清单补上了另一半：μP 对相当多的「理论不覆盖」的改动其实是稳健的。在激活函数上，SwiGLU、Squared ReLU、ReLU 的最优学习率保持一致，且各自只带来 minor gains（小幅收益）；在 batch size 上，虽然原始推导根本没有处理 batch 维度，放大或缩小批大小后 μP 依然稳健。这类结论与本讲后半段「SwiGLU 严格说不在 μP 框架内，但 mostly work with μP」是同一判断，只是参考把它拆成了可逐项核对的实验表。

初始化变体是清单里最具体的一块。SP Unembedding 指的是 presoftmax 线性层按 $1/M$（SP）对 $1/\sqrt{M}$（μP）两种缩放之比，Zero Query 则是把 query 矩阵置零、让所有位置得到均匀 attention；两者都没有破坏超参迁移。这与前文 transformers 逐组件表里的 $W^U$、$W^{AQ}$ 各行是同一批权重，说明逐组件缩放规则对「怎么初始化」有一定容忍度，真正敏感的是「缩放量级随宽度怎么变」。把这一层补上后，前文的 Table 2 就不再只是一张冷冰冰的系数表，而有了「哪些格子可以动、动了会怎样」的实验注脚。

破坏侧的边界也该补两个具体量。参考明确写出 strong weight decay（强权重衰减）例如 $0.1$ 会造成 significant μP failure，并称之为「为数不多真正让 μP 失败的 stress test 之一」；RMSNorm 的可学习 gain 与依赖梯度符号的 Lion 同样破坏 μP，但 gain 去掉后几乎不损性能。合起来看，鲁棒性是有层次的：非线性、批大小、初始化变体属于「改了也没事」，RMSNorm gain、sign 梯度、大 decoupled weight decay 属于「改了就坏」，而前者恰是常见工程配置、后者恰是可主动规避的配置。易错点是把「理论不禁止」当成「实践无所谓」：前文 $\tau^{-1}\neq0$ 在理论上不违反 μP、实验却对 performance and transfer 影响巨大，正是同一类陷阱；反过来，也别因为 $0.1$ 这一个数就把所有 weight decay 一棍子打死，参考限定的是 strong（强）解耦权重衰减。

### 四、拟合 Chinchilla 曲线的工程工具与另一条不重训的路径

前文详写了 WSD 如何把数据轴扫描成本从 $O(n^2)$ 压回 $O(nC)$，但 2025 参考还记了两件我们没写的事。其一是联合拟合用的具体工具：MiniCPM 用 `scipy.curve_fit` 对 $L(N,D)$（Hoffmann et al. 2022 的形式）拟合，采用 Chinchilla 的 method 1（lower envelope，下包络）与 method 3（joint fit，联合拟合），而 method 1 的下包络给出的解读是 relatively low diminishing returns due to data（数据带来的边际收益递减相当低）。这与前文讲者把 method 1、method 3 评为「最不可靠的两种」并不矛盾：工具是中性的，被质疑的是用它外推出的指数与数据/模型比，而不是 `curve_fit` 本身。补上工具名后，前文那句「Chinchilla style analyses are actually very easy」才有了可复现的落地含义。

其二是二次拟合在跨论文间的通用地位。Llama 3 在 $6\times10^{18}$ 到 $10^{22}$ FLOPs 的每一档 IsoFLOP 曲线上用 second degree polynomial（二次多项式）逼近测量点来定位最小值，Hunyuan 对 MoE 的 training loss 对 activated parameters 同样 fit quadratics（拟合二次曲线）。前文只把「二次多项式」记成 Llama 3 一张图的图注细节，参考则显示它是 IsoFLOP 分析的通用手法：每条 U 形曲线的最低点不是肉眼读的，而是拟合出来的。这也回扣了前文 MiniCPM 最优批大小那条「拟合 quadratic curves 找 equal loss contour 极小值」的做法——从超参数曲面到算力曲线，用的都是同一个二次逼近工具，读者不必把它当成三家各自发明的技巧。

最值得一提的是 Gadre et al. 提出的另一条路径。它不靠 WSD 复用 checkpoint，而是基于一个观察：overtraining 带来的 excess token penalty（超额 token 惩罚）在不同规模间保持稳定，于是可以把小规模运行上量到的这份惩罚外推，反过来估计 Chinchilla 曲线。这相当于对「拟合缩放律必须从头重训」给出了与 WSD 并列的第二种答案，也解释了为什么参考把它放在 WSD 之后的 Side Note 里。但参考同时注明，该做法并未在大规律训练中被广泛采用，因此它更适合作为方法版图里的一格，而不是可照抄的配方。把这些放一起，20:1 就只是一个 starting point（起点）：最优 token/参数比在 39:1（Llama 3）、96:1（Hunyuan）乃至更高的记法之间漂移，本讲反复质疑的「公式能否外推」在这里再一次得到印证。易错点是把 Gadre 的惩罚外推与 WSD 的 checkpoint 复用混为一谈——前者靠曲线形状外推、后者靠调度形状复用，机制完全不同；也不要因为 39:1、96:1 这些数字好看就把它们当成新定律，它们各自绑定了具体架构与数据配方。

### 更正

原笔记第 7 章「DeepSeek 收尾」处有一句「DeepSeek 代表的是后一条 'fit the scaling law approach'，而后面要讲的 MiniCPM 是前一条」，其中「后面要讲的 MiniCPM」应改为「前面第 2–5 章已经讲过的 MiniCPM」（依据：参考笔记把 MiniCPM 编在 `[1:00:00]` 一节、DeepSeek 编在 `[1:31:00]` 一节，且 Recap 亦按 CerebrasGPT→DeepSeek→MiniCPM 顺序排列；本笔记自身第 2–5 章先讲 MiniCPM、第 6–7 章才讲 DeepSeek）。两条路线的归属判断——MiniCPM 对应「用 μP 把超参稳定住」、DeepSeek 对应「直接拟合超参怎么变」——本身正确，无需改动，需修正的仅是讲授先后的叙述。
