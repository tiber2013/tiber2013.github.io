---
title: "第 3 讲 · 从零设计 Transformer 架构"
description: "架构设计没有简洁的理论工具可用，最实用的方法是从大量已发布模型的共同选择中提炼经验规律。视频的核心判断是：LayerNorm 移出残差流、改用 RMSNorm、去掉偏置项、使用门控线性单元（GLU） 这几项已成为几乎所有…"
---

# 第 3 讲 · 从零设计 Transformer 架构

> **Stanford CS336 语言模型从零构建 · L03** ｜ 讲者 Stanford Online ｜ 89 分钟 ｜ [原视频](https://www.youtube.com/watch?v=lVynu4bo1rY)

> **v2 详写版** · 按 20 分钟分段详写，配图覆盖全片；文末附「对照参考笔记补遗（Spring 2025 版）」，记录与 2025 版讲义笔记的核对结果。

## 核心结论

这一讲的核心判断是：架构没有从定理推出来的最优解，可靠知识来自横向铺开大量已发布模型，辨认哪些选择被集体固定、哪些可自由变动。这轮综述收敛出的共识相当集中——归一化移出残差流的 pre-norm、不减均值也不加偏置的 RMSNorm、丢掉线性层 bias、给前馈层加门控线性单元，几项几乎被所有现代模型采用；它们成立靠的不是表达力更强，而是梯度沿残差流直通带来的稳定，以及删去小而费内存操作换来的算术强度，讲者坦言这多半是「大家都这么抄」的路径依赖、无法事先推理。相对地，前馈倍率（非门控 4 倍、门控约 8/3）、$H\cdot d_{head}=d_{model}$、宽深比约 100、词表规模都落在很宽的平坦盆地里，取错代价很小，系统利用率与训练稳定性反而比精细调参更决定设计。位置编码是差异最大、仍在演化的一维，RoPE 以「按位置成对旋转、使内积只依赖相对位置」成为 2024 年后主流；模型越训越贵之后，余下的主线是稳定性与长上下文：输出端用 z-loss 把 $\log Z$ 拉向 0、注意力里用 QK norm 稳住输入、部署侧用 GQA 缩减 KV cache，并以滑窗与全局注意力交替管理长上下文成本。全讲始终留有余地：这是带不确定性的经验概括而非设计手册，弄懂仍要靠自己动手训练。

## 认知框架：架构为何难以捉摸，以及「原始 Transformer」基线 [00:00 - 02:57]

讲者开场就坦承，架构（architecture）这门学问对他而言一直「难以捉摸」（inscrutable），因此本讲采取的办法近乎笨拙却有效——「把所有东西都讲一遍」（telling you kind of everything），带听众把现代论文逐一过一遍，看看「大家都做了什么」。这一讲被他戏称为「你不想知道的关于架构与超参数的一切」（everything you didn't want to know about architectures and hyperparameters）。这个略带自嘲的标题背后是一个很具体的落差：我们都希望生活在一个只需 VC dimension（VC 维）之类简洁理论工具就能推导出架构的世界里，但现实并不是这样；架构设计里绝大多数结论并不是从定理里推出来的，而是从大量已训练模型的经验里捞出来的。理解这一点，是听懂后续所有「为什么大家都这么选」的前提，也解释了为什么整堂课会反复出现「没有保证」「不知道为什么」这类留有余地的措辞。

由此引出本讲的方法论——用综述视角（survey lens）看架构。讲者明确说，比听这门课更好的办法是自己动手训练模型、尝试不同架构，这也正是课程一贯的哲学；但课程既没有算力（compute）也没有时间去覆盖全部架构的设计空间（design space），所以「第二好的办法」是从他人的经验里学习：把每年发布的模型摊开成一个较宽的、略微拉远的图景，然后问哪些参数和选择是在所有有效架构里都固定不变的，哪些则可以在不影响模型表现的前提下自由变动。这条思路会贯穿整段：先以 Vaswani 等人的原始 Transformer 为起点，再看更现代的架构有哪些共性、又允许哪些变化，也就是把「transformer variants」当作一个可以横向比较的样本集合来处理。

作为共同参照，幻灯片给出了「原始 Transformer」（the 'original' transformer）里被当作标准答案的几项选择。第一是位置编码（Position embedding）用正弦和余弦（sines and cosines），因为注意力本身不含位置依赖，必须显式注入位置信息，其公式按画面原样写为

$$PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \qquad PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

第二是前馈网络（FFN）用 ReLU 作非线性，公式写作 $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$，其中 $W_1, b_1, W_2, b_2$ 是两层线性变换的权重与偏置；第三是归一化类型（Norm type）为 post-norm 且用 LayerNorm；讲者在转写里也把这三件事串成一句「加 sines and cosines、信息过 ReLU、再来一个 post norm」。配套的结构图显示编码器与解码器各由 $N\times$ 个重复 block 堆叠，每个 block 内含 Multi-Head Attention（解码器为 Masked Multi-Head Attention）、Feed Forward 以及 Add & Norm，输出侧再接 Linear 与 Softmax 得到 Output Probabilities，Positional Encoding 以加法方式叠加到 Input Embedding 与右移的 Output Embedding（shifted right）上。这里的常数 $10000$、指数 $2i/d_{model}$、四个可学习参数以及 $N\times$ 的堆叠层数，就是这一基线里最具体的数字与参数。

![原始 Transformer 的标准组件：正弦余弦位置编码、ReLU 前馈与 post-norm LayerNorm](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0002_00-02-05.jpg)

紧接着讲者把镜头拉回课程作业 A1：要求实现的东西与这份 vanilla（原生）Transformer 明显不同——把 layer norm 移到每个 transformer block 或非残差层的最前面（也就是后面要讲的 pre-norm）、实现 RoPE（旋转位置编码）、以及用 SwiGLU 而不是 ReLU。为什么要这样改？讲者给的理由相当坦诚：很大程度上是照抄 LLaMA，而「别人也都是这么抄的」（we've copied a lot of this over from LLaMA, but so did everyone else）。这句话看似随意，其实点出了整门课的要害：一旦自己动手训练语言模型，立刻会撞上「这么多选择我到底该选哪个」的问题，而业界给出的答案往往是路径依赖式的共识，而非理论最优。因此本章的作用就是立起一根标尺：后面 07:29 起讨论的归一化位置、13:55 起讨论的 RMSNorm、20:13 起讨论的激活函数，全都是对这几项「标准选择」的逐项修订，读者心里得先有这份基线，才看得出每一处改动改的是什么。

这里有几处容易读错。其一是 post-norm 这个术语在后续会被反复使用且含义会漂移，幻灯片上写的「Norm type: post-norm, LayerNorm」特指归一化落在残差路径内部，而第 4 章会出现「残差流之外的 post-norm」，两者并不等价；其二是正弦余弦编码只是众多位置编码方案中的一种，后来绝大多数模型改用了 RoPE，不能把它当成固定结论；其三是 $\text{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2$ 里带着 $b_1,b_2$ 两个偏置项，而第 5、6 章会说明现代实现基本把它们删掉——同一份基线公式，恰好是后面两处改动的靶子。讲者自己也承认，A1 的三个替换项并非由某条理论推出，而是「大家都在用」，这种把经验共识当作起点的做法，本身就是本章最该被记住的局限。

---

## 模型大爆发、横向综述方法与本讲路线图 [02:57 - 07:29]

讲者说自己每年都会为更新这节课去扫一遍当年新出的模型，Percy 还常拿这点打趣。去年他本以为「就几篇论文，没问题」，结果一查发现 Qwen 2、Gemma 3、InternLM2，再往下还有 NeMo Tron 4 等，一共有 19 个新的稠密模型（19 new dense models），备课量陡增。今年他原以为节奏该放缓了——「总不能一直每年训 20 个稠密 LM 吧」——这个判断某种意义上是对的：稠密模型确实没那么多了，一开始只数到 Qwen 3、「上周四刚发布」的 Gemma 4，还有 Percy 自己用 Marine 训练的 8B 模型，看起来只有寥寥几个。但真去翻才发现模型依然很多，只不过其中大多数已经是 MoE（mixtures of experts，混合专家）架构，而 MoE 被安排到第二天再讲，本讲聚焦稠密架构。这段自述不只是段子，它给出了后文那张对比表的来历：样本之所以够用，恰恰是因为发布量足够大。

配合这段「模型爆炸」的幻灯片（kf_0003）标题是「How should we think about architectures?」，副标题写着「Lots of architecture. Just in 2024-2025..」，画面里堆着一叠技术报告封面：Command A: An Enterprise-Ready Large Language Model（标注 2025-01-06）、Qwen2.5 Technical Report（Qwen Team，附带 huggingface.co/Qwen 与 github.com/QwenLM/Qwen2.5 两个链接）、InternLM2 Technical Report、Phi-4 Technical Report（2024-12-12）、Gemma 3 Technical Report（Gemma Team, Google DeepMind）、Data-Centric Training of a Small Language Model（2024-06-27）、Gemma 2: Improving Open Language Models at a Practical Size（Gemma Team, Google DeepMind），以及 Microsoft 的 Phi-3 技术报告。这些具体报告名与日期是画面独有的信息，转写里并没有逐一念出，却正好把「2024–2025 年一下子冒出这么多架构」这件事落到了实物上。讲者的推论是：正因为模型种类如此多样，我们反而能看清全部可选的设计选择（a pretty good picture of all the different choices that we can make），这也是「从他人经验学习」这一方法能被认真对待的前提。

于是他做了一张大表，并说「讲到最后我们会回到这张表」（We'll come back to this little table at the end of the lecture）。对应画面（kf_0004）标题为「Let's look at the data (on dense architectures)」，副标题「Learn from the many other models (and papers) out there」，左侧是一张黑底表格，逐行罗列大量模型、逐列给出各项超参数；转写里明确点出的列名问题包括「不同模型的词表大小（vocabulary sizes）是多少」「大家用哪种 layer norms」「用哪种 position embeddings」，并说从中能看到相当清晰的趋势（fairly clear trends）。右侧写着「We will talk through many major architecture and hyperparameter variants.」，下面三个 bullet 依次是「What do all these models have in common?」「What parts vary?」「What can we learn from this?」——这三问其实就是上一章「区分固定项与可变项」方法论的书面版，也预告了本讲会沿着「先看共性、再看可变、最后提炼经验」的顺序推进。

![稠密架构横向对比表：罗列各模型的词表大小、归一化与位置编码等超参数](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0004_00-04-52.jpg)

本讲要覆盖的东西被拆成三层（对应 kf_0005 的「What are we going to cover?」）：先是 Common architecture variations，下分 Activations, FFN、Attention variants、Position embeddings；再是 Hyperparameters that (do or don't) matter，画面上直接写成了两个待回答的问题——「What is ff_dim? Do multi_head dims always sum to model_dim?」与「How many vocab elements?」；最后是 Stability tricks。右侧淡色小字还预告了 High level view: 「Dominance of 'LLaMA-like' architectures」以及「Trends over the years (QK-norm, Hybrid attention)」。转写里把第二层讲得更具体：前馈维度到底该取多少，是让 FF dim 变成 hidden 的 4 倍吗，即

$$d_{ff} = 4\,d_{model}$$

还是别的倍率？多头维度是否总满足 $\sum_h d_{head} \stackrel{?}{=} d_{model}$？词表该放多少个元素？注意这里的 $4$ 倍与「是否求和等于 $d_{model}$」都是以问句形式提出的经验取值，而不是本讲已经证明的定律——把它们写成带问号的表达式，比写成一个断言更忠实。讲者还特意解释为什么把稳定性技巧放进架构课：这些 tricks 与架构变体关系很近（a pretty close connection with the architecture variation），所以三者被安排在同一讲里连着讲。

为什么架构会这么琐碎？讲者给出三重约束：架构必须从数据里学习、也就是要能泛化（generalize）；必须在 GPU 上高效训练；还必须「不炸」——训练到一半 loss 本来平稳下降却突然 blow up 是绝对不能接受的。这三条要求被直接「烤进」（baked straight into）架构里，所以架构才显得 messy、complex、不优雅（not so elegant）。历史脉络上，他把演化分成几段：从原始 Transformer 到 GPT-3 左右是大量实验、没有 gold standard 的时期；LLaMA 2 发布后所有人都想拥有自己的 LLaMA 2，于是纷纷训练带小改动的「LLaMA 2-alikes」；近一两年先出现一波以提升训练稳定性为目标的架构修改，今年则集中出现支持更长上下文依赖（longer context dependence）的变体。19 个稠密模型、8B、2024–2025 这些数字，与「LLaMA-like 主导」这一观察一起，构成了从第 1 章那根标尺走向后续逐组件分析的桥梁：先知道有哪些组件、哪些取值在被集体采用，才谈得上一一追问理由。

这一章有几个坑。第一，模型数量多并不等于设计空间被穷尽，横向对比能给出「大家怎么做」，给不出「为什么必然如此」；第二，本讲反复限定在 dense architectures 上，MoE 明确推到明天，看到表格里五花八门的模型时别把 MoE 的结论混进来；第三，「LLaMA-like 主导」和「近年趋势是长上下文」都是对已发布模型的观察，属于经验概括而非规律；第四，转写在列举今年模型时出现一处含糊的「almost 3」，无法确认所指，宁可略过也不要替讲者猜一个模型名。这些留白与保留，和上一章结尾的态度是一致的：这门课给的是带不确定性的经验，而不是封闭的定理集。

---

## 唯一的全行业共识：LayerNorm 该放在哪里 [07:29 - 10:20]

讲者说架构上人们几乎可以在任何事上分歧，但有一件事大家是同意的：如果拿 Transformer 原论文来看，多数人会觉得「Transformer 那帮人几乎全做对了，除了这一件」——唯一公认没做对的就是 layer norm 放的位置。要讲清这件事，得先理解残差流（residual stream）：网络里有一条贯穿始终的 $x$，各子层（多头注意力、前馈网络）算出一个增量（delta）再加回这条流里。为了让梯度在层与层之间保持稳定，原始 Transformer 把 LayerNorm 放在每个组件的末尾，也就是放在残差路径内部；这种结构被称为 post-norm，讲者也建议暂且叫它 residual norm（残差归一化），因为归一化的确发生在残差层里。换句话说，问题从来不是「要不要归一化」，而是「归一化插在残差相加之前还是之后」。

幻灯片（kf_0006）标题「Pre-vs-post norm」，副标题是「The one thing everyone agrees on (in 2024)」。左侧两张残差塔图分别标着 $x_l$、$x_{l+1}$、addition、Layer Norm、Multi-Head Attention、FFN；右侧并排给出 Post-LN Transformer 与 Pre-LN Transformer 的逐步公式，这是画面独有、转写没有逐字念出的内容。Post-LN 一侧是：

$$x_{t,i}^{post,1} = \text{MultiHeadAtt}(x_{t,1}^{post},\ldots,x_{t,n}^{post}),\quad x_{t,i}^{post,2}=x_{t,i}^{post,1}+x_{t,i},\quad x_{t,i}^{post,3}=\text{LayerNorm}(x_{t,i}^{post,2})$$

$$x_{t,i}^{post,4}=\text{ReLU}(x_{t,i}^{post,3}W^{1,l}+b^{1,l})W^{2,l}+b^{2,l},\quad x_{t,i}^{post,5}=x_{t,i}^{post,3}+x_{t,i}^{post,4},\quad x_{t,i}^{post,6}=\text{LayerNorm}(x_{t,i}^{post,5})$$

Pre-LN 一侧则把归一化提到子层之前：

$$x_{t,i}^{pre,1}=\text{LayerNorm}(x_{t,i}^{pre}),\quad x_{t,i}^{pre,2}=\text{MultiHeadAtt}(x_{t,i}^{pre,1},[x_{t,1}^{pre,1},\ldots,x_{t,n}^{pre,1}]),\quad x_{t,i}^{pre,3}=x_{t,i}^{pre}+x_{t,i}^{pre,2}$$

$$x_{t,i}^{pre,4}=\text{LayerNorm}(x_{t,i}^{pre,3}),\quad x_{t,i}^{pre,5}=\text{ReLU}(x_{t,i}^{pre,4}W^{1,l}+b^{1,l})W^{2,l}+b^{2,l},\quad x_{t,i}^{pre,6}=x_{t,i}^{pre}+x_{t,i}^{pre,5}$$

并且补了一行 Final LayerNorm：$x_{Final,i}^{pre}\leftarrow\text{LayerNorm}(x_{t,i}^{pre})$。图下配的说明是「Set up LayerNorm so that it doesn't affect the main residual signal path (on the left)」，出处标为「Figure from Xiong 2020」。把两组公式压缩成一句话，就是旧写法 $x_{l+1}=\text{LayerNorm}(x_l + \text{MultiHeadAtt}(x_l))$ 与新写法 $x_{l+1}=x_l+\text{MultiHeadAtt}(\text{LayerNorm}(x_l))$ 的差别：归一化究竟在残差相加之前还是之后。

![Post-LN 与 Pre-LN Transformer 的残差结构及逐步计算公式对比（Xiong 2020）](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0006_00-07-40.jpg)

所谓 pre-norm，就是把 layer norm 挪到残差流之外、放到每个计算之前——可以放在 multi-head attention 之前，也可以放在 FFN 之前。讲者提醒术语会变得有点乱（nomenclature will get a little bit confusing）：先把原始那种叫 post-norm，但更准确的名字是 residual norm，因为归一化在残差层里。结论非常干脆：基本上所有现代语言模型都把 layer norm 推到残差流之外（push the layer norm outside of the residual stream），这是「所有人都做的一件事」。幻灯片底部加粗写着「Almost all modern LMs use pre-norm (but BERT was post-norm)」，并附了一句带调侃的例外说明：「One somewhat funny exception – OPT350M. I don't know why this is post-norm」。转写里讲者说得更直白：OPT 系列本身就被认为是「一团乱麻」（kind of a mess of a language model），而 OPT-350M 尤其如此，不知道为什么唯独这一个模型在残差流里用了 post layer norm。

这一章里的具体数字与参数包括：2024 这个「共识形成」的年份标注、OPT-350M 的 350M 参数规模、BERT 作为历史反例、公式里逐层出现的 $W^{1,l}, b^{1,l}, W^{2,l}, b^{2,l}$ 四个带层号 $l$ 的参数、以及 Post-LN/Pre-LN 两侧各 6 个中间变量 $x^{post,1}\ldots x^{post,6}$、$x^{pre,1}\ldots x^{pre,6}$。从前后关系看，本章正是第 1 章基线里「Norm type: post-norm, LayerNorm」那一行的定点推翻：位置编码、ReLU、FFN 都还有商量余地，唯独归一化位置被全行业一致改掉；而「为什么大家这么一致」这个疑问，直接把讲者引向下一章的 warm-up 与梯度论证——他马上要问「why is this such a unified thing across all the different models?」。

要特别小心三处。第一，pre-norm 并不是「少了一次归一化」——图上明确多出一个 Final LayerNorm，这是转写没有念到、只看结构图才会发现的细节；第二，post-norm 这个词在本段前后含义会漂移，本章指「归一化在残差流内部」，第 4 章的「non-residual post norm」却是指「归一化仍在残差流外、只是放到子层计算之后」，两者不要混为一谈；第三，讲者对 OPT-350M 的评价是「不知道为什么」，这是诚实的不确定，不要替他补一个设计理由，也不要把 BERT 用 post-norm 误当成「post-norm 也很好」的证据——它只是历史上一个尚未统一的时期留下的样本。

---

## pre-norm 为何胜出：warm-up、梯度直通与稳定性 [10:20 - 13:55]

为什么位置这件事会如此统一？讲者先回到早期研究 LayerNorm 位置的原始动机：训练 Transformer 时必须做 warm-up（学习率预热），现代训练其实也仍然做，但早期尤其「绝对需要」（you definitely need to do warm-up when you train）。于是很自然的想法是：能不能把 warm-up 去掉？「Wouldn't it be nice if we could remove the warm-up?」——这正是这类研究最初的出发点。但人们很快发现，去掉 warm-up 会在稳定性与收敛上带来严重问题（very serious issues in terms of the stability and convergence），于是「去掉预热」这个目标反而成了检验不同归一化位置的试金石。

对应画面（kf_0008）标题「Pre-vs-post-norm, the data」。左边是 Salazar and Nguyen 2019 的「English-Vietnamese development BLEU」曲线，纵轴 Dev BLEU 从 18 到 30，横轴 epochs 从 0 到 100，图例列出五条配置：PreNorm+ScaleNorm+FixNorm+NoWarmup、PreNorm+FixNorm、PreNorm+LayerNorm+FixNorm、PreNorm+LayerNorm、PostNorm+LayerNorm。转写里讲者说的「post norm plus layer norm 会得到这条紫色虚线，收敛就是不如 pre norm」，对应的正是图例最后那条 PostNorm+LayerNorm 点线，它稳定在约 26–27 BLEU，而其余 pre-norm 变体爬到约 28–29。右侧两图出自「Figure from Xiong 2020」：(a) Validation Loss (IWSLT)，纵轴 Validation Loss 4–8、横轴 Epochs 1–15，图例为 Post-LN (Radam w/o warm-up)、Pre-LN (Radam w/o warm-up)、Post-LN (Adam w/ warm-up)、Pre-LN (Adam w/o warm-up)；(b) BLEU (IWSLT) 纵轴 0–30；下方还有 (a) Validation Loss on BERT，纵轴 1.6–2.0、横轴 Pre-training Steps (Thousands) 100–900，图例 Post-LN 与 Pre-LN。这些坐标轴刻度、任务名与图例全称都是画面独有的细节，转写只笼统提了「紫色虚线」和「Salazar and UN 的图」。

![Pre-norm 与 Post-norm 的收敛数据：英越 BLEU、IWSLT 与 BERT 验证损失曲线](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0008_00-10-27.jpg)

不过讲者强调，真正让 pre-norm 留下来的是更深的理由，其中「梯度衰减」（gradient attenuation）最清楚。架构设计圈有条经验法则叫「keep your residual stream clean」（保持残差流干净）：pre-norm 下输入 $x$ 从网络底部一路传播到顶部、直到最终输出，反向传播时梯度可以沿这条干净路径直接穿过，于是梯度尺度大致保持不变，稳定性与信号传播（signal propagation）都变好。用递推式写就是

$$x_{l+1}=x_l+F_l\big(\text{LN}(x_l)\big),\qquad \frac{\partial x_{l+1}}{\partial x_l}=I+\frac{\partial F_l}{\partial x_l}$$

雅可比里的恒等项 $I$ 保证梯度有一条不衰减的直通通道；而 post-norm 写作 $x_{l+1}=\text{LN}\big(x_l+F_l(x_l)\big)$，每穿过一个 block 都要被 LayerNorm 重新缩放一次，连乘之后梯度范数随深度改变，正是讲者所说「complicated effects」的来源。他说在 pre-norm 加蓝色初始化（blue initialization）下梯度大小基本不变，靠的就是这种直通传播。对应课堂投影（kf_0009）左图标着「Gradient attenuation [Xiong 2020]」，纵轴 Gradient Expectation 0.0–1.0、横轴 Layer 1–6，图例为 Pre-LN (init)、Post-LN (init)、Post-LN (after warm-up)，并注明「(a) $W^1$ in the FFN sub-layers」——深层的 Post-LN (init) 柱子明显冲到 1.0 以上，而 Pre-LN 各层都低而平稳，这正好把「恒等项」的直觉落到了可测量的梯度期望上。

第二点是通过实验发现的稳定性提升：pre-norm 下梯度尖峰（gradient spikes）的大小和频率都优于 post-norm。kf_0009 右图「Gradient spikes [Salazar and Nguyen]」的纵轴是 Gradient global norm (log scale)（−0.5 到 3.5）、横轴 Iteration (x1000)（0 到 1200），图例含 PostNorm + LayerNorm、PreNorm+ScaleNorm + FixNorm + NoWarmup、PreNorm + ScaleNorm + FixNorm、PreNorm + LayerNorm。屏幕底部两行总结很关键：「Original stated advantage – removing warmup.」与「Today – stability and larger LRs for large networks」——也就是说，最初的卖点（去掉 warm-up）如今已经退居其次，真正的价值是稳定性以及大网络能用更大的学习率。转写里也把 Salazar 与 Nguyen（转述中念作「Salazar and UN」）列为最早仔细研究这一现象的人之一，并说「This is a figure from Salazar and UN」。讲者的结论是：稳定性（stability）与「能训深」（the ability to go deep）对现代大语言模型都极其重要，所以把 layer norm 移出残差流被所有人采纳。

既然问题出在「归一化落在残差流里」，那么归一化不一定要放在计算之前，放在计算之后、但仍在残差流之外，同样成立。转写点名 Grok、Gemma 2、Olmo 2 采用了「把 layer norm 移到计算之后」的结构，并特意说明这是「某种 post norm，但它在残差流之外」（a post norm of a kind, but it's outside the residual stream）。对应投影（kf_0010）标题是「New things – 'double' norm or non-residual postnorm」，副标题直接把逻辑写成问句：「If putting LayerNorms in residual streams is bad.. Why not post-norm outside the stream?」，下方两张结构图标着 Layer Norm、addition、FFN、Multi-Head Attention、$x_l$、$x_{l+1}$，底部一行是「Recent models: Grok, Gemma 2. Olmo 2 only does non-residual post norm」——「只有 Olmo 2 单独用 non-residual post norm」这条限定是画面独有、转写没念全的。还有些模型干脆到处放 layer norm（这里一个、后面一个），即 double norm；讲者顺带给出一条屡试不爽、他自己都觉得「荒谬」（ridiculous）却总被验证的经验：一旦遇到稳定性问题，就往里「撒」layer norm，甚至往 attention 里塞一个也管用（「what if we just throw a layer norm into attention? Turns out that works, too」），这部分留到后面讲稳定性时再展开。

读这章要留意几件事。首先，warm-up 是历史动机而非最终理由，别把「pre-norm 好」简单归因于「可以不用预热」；其次，几张曲线图的任务与纵轴完全不同（英越翻译 BLEU、IWSLT 的 Validation Loss 与 BLEU、BERT 的预训练验证损失），数值不能横向直接比较；再次，讲者自己在 17:18 前后承认 tiny models 上的百分比「很极端」，这些小模型实验只用来说明机制，不等于大模型上的精确定量；最后，本章把「归一化位置」这个问题彻底收束，下一章顺势追问：既然位置定了，归一化这个操作本身还能不能再简化？

---

## RMSNorm：减均值、去偏置与算术强度 [13:55 - 18:02]

把归一化位置定下来之后，讲者转向归一化本身。原始 Transformer 用的 LayerNorm 是这样一个操作：拿激活 $x$，先减均值（mean subtract），再除以方差（divide the variance），最后缩放回去（scale it back up）。讲者先给它一个公道的评价：这没什么错（it's not like this is wrong），很多模型都在这个方案上成功训练过。幻灯片「Why RMSNorm?」（kf_0011）把标准公式原样写为

$$y=\frac{x-\mathrm{E}[x]}{\sqrt{\mathrm{Var}[x]}+\epsilon}\,\gamma+\beta$$

（不同资料也常把 $\epsilon$ 写进根号内，这里按画面记作 $\sqrt{\mathrm{Var}[x]}+\epsilon$；$\gamma$ 是缩放参数、$\beta$ 是平移偏置）。图上的「Modern explanation – it's faster (and just as good).」下面给了两条 bullet：「Fewer operations (no mean calculation)」（不计算均值，操作更少）与「Fewer parameters (no bias term to store)」（不存偏置，参数更少）。这两条其实已经把改动内容说尽了：删掉均值与偏置，只保留缩放。

![Why RMSNorm：LayerNorm 公式与算子类别 FLOP 占比表（99.80 / 0.17 / 0.03）](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0011_00-14-38.jpg)

但基本上大多数、乃至所有现代模型用的是 RMSNorm：它不减均值、也不加 bias term，只是「缩放下去再缩放回来」（just a scaling down and scaling back up）。把上一式里的 $x-\mathrm{E}[x]$ 换成 $x$、并去掉 $+\beta$，就得到讲者所描述的这种纯缩放形式，常见写法是

$$y=\frac{x}{\sqrt{\tfrac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}}\odot\gamma$$

其中 $d$ 是隐藏维度、$\odot$ 表示逐元素相乘。这里有一个容易讲反的点：LayerNorm 的表达能力其实比 RMSNorm 更强（layer norm is more expressive than RMS norm），从表示能力（representationally）上讲根本没有理由非换不可；换的理由是实践层面的两条——一是「实践中确实没有表达力损失」（there's really no expressiveness loss），RMSNorm 建模效果与 LayerNorm 一样好；二是更重要的「更快」（but more importantly, it is faster）。讲者把这一步明确归为「the systems and sort of architecture co-design starts to come in」，即系统视角开始介入架构决策。

为什么更快？这就要引入上一讲 Percy 提过的算术强度（arithmetic intensity）。GPU 要靠矩阵乘这类计算密集的操作保持「发热」（keep our GPUs hot by doing matrix multiplies and other very intense computations），最浪费的是让它来回搬运一小块一小块内存（moving little tiny bits of memory back and forth）。因此该被删掉的，正是那些规模很小、涉及内存搬运、却给不了多少表达力的操作——均值减法与偏置加法首当其冲。讲者特意预先反驳一个可能的质疑：你优化的是一个「极小、只占总浮点运算约 0.17%」的操作，值得吗？kf_0011 下方「Does this explanation make sense?」的表格给出了精确数字：按算子类别（Operator class）划分的 % flop 为——△ Tensor contraction（张量收缩，即矩阵乘）99.80，□ Stat. normalization（统计归一化）0.17，○ Element-wise（逐元素）0.03；表下一行写着「Matrix multiplies are the vast majority of FLOPs (and memory)」，右下角来源是 [Ivanov et al 2023]。

关键的一句出现在下一张「Why RMSNorm (2)」（kf_0012）的副标题：「Important lesson: FLOPS are not runtime! (we will discuss this in far more detail later)」。FLOPs 数的是浮点运算、也就是矩阵乘，但 runtime 是个复杂得多的对象（Runtime is a much more complicated object）。这张表把两列并排：% flop 仍是 99.80 / 0.17 / 0.03，而 % Runtime 变成——△ Tensor contraction 61.0，□ Stat. normalization 25.5，○ Element-wise 13.5。也就是说，只占 0.17% FLOPs 的统计归一化，在特定设置下能吞掉约四分之一（转写里说「up to 25% of the runtime」）的运行时间。原因是 tensor contraction 的主要工作量是乘，而 stat normalization 的主要工作量是内存搬运（memory movement），后者很慢；激活值（activations）又可能相当大，仍要在快存储与慢存储之间来回搬参数。图下方的结论是「RMSNorm can still matter due to the importance of data movement」。右侧还有一张计算流示意：MHA 框左上角标 43G、右上角标 153，Dropout 标 4M 与 1/3，LayerNorm 标 29M 与 3.5，并说明「Left top ("43G") is FLOPS」、「Right top ("153") is the FLOP-to-memory ratio」，即右上数字是浮点运算与访存之比；按讲者的说法白色代表算术强度、黑色代表 FLOPs，LayerNorm 的算术强度非常低，正是最该被优化掉的那类算子。

![FLOPs 不等于运行时间：算子类别的 FLOP 占比与 Runtime 占比对照及计算流算术强度](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0012_00-16-01.jpg)

16:48 有学生追问，为什么归一化的数据搬运相对算术收缩如此不成比例。讲者答：tensor contraction 里大部分 workload 是相乘，stat normalization 里大部分是内存搬运，而内存搬运相当慢；可以想象「搬东西」几乎占了全部计算时，你依然要付这笔钱，何况 activations 可以很大。他同时很诚实地补了一句：这里的 runtime 百分比「相当极端」，来自 tiny models、矩阵规模按现代 workload 看并不合理的情形，但它足以说明这是一次「免费的优化」（a free optimization）。紧接着他举了另一篇评估不同架构干预的论文（转写里含糊地说成「Get All in 2020」「I think this was a Google paper」，此处不替他猜作者）：在一个 2 亿参数（200 million parameters）的小 Transformer 上换成 RMSNorm 后，表格第三列的 steps per second（每秒训练步数）提高了，而且最终效果还更好——讲者特别说明后者「不是你能保证的，但算是个不错的额外收获」（it's a nice bonus regardless）。于是「基本上所有人都决定转过来了」（basically, everyone has decided to move over to this now）。

这一章承接第 3、4 章「归一化放在哪」，回答「归一化是什么」：位置与形式两处改动叠加，才构成现代 block 的完整样子；同时它把「删掉小而费内存的操作」这条原则立了起来，下一章会原封不动地推广到所有线性层的 bias 项。要留意的局限有：0.17% 是 FLOPs 占比、25.5% 是 runtime 占比，两者不是同一个量，也不能把 25.5% 当成任何模型上的常数；「表达力不损失」是实验结论而非理论保证；RMSNorm 并不比 LayerNorm 更强，选它纯粹是系统收益；讲者引用 2 亿参数模型与 tiny models 的极端百分比，只是为了说明机制，不能直接外推到大规模训练。

---

## 丢弃偏置项，以及归一化话题的收束 [18:02 - 20:00]

讲者把上一章的逻辑再推广一层：「There's a more general version of this」——而且他先打了个预防针：「There's no guarantee to any of the things I'm saying」（我接下来说的都不打包票）。这条更一般的经验是：transformer 和神经网络里的 bias terms（偏置项）通常没那么有用（bias terms in transformers and neural networks are generally not that useful）。幻灯片「More generally: dropping bias terms」（kf_0014）第一句就写「Most modern transformers don't have bias terms.」，然后把两种写法并排给出。原始 Transformer 的前馈层是

$$\text{FFN}(x)=\max(0,\,xW_1+b_1)W_2+b_2$$

而「大多数实现（如果不是门控的话）」（Most implementations (if they're not gated)）写成

$$\text{FFN}(x)=\sigma(xW_1)W_2$$

其中 $\sigma$ 泛指某个激活函数。图底部把理由压成一行：「Reasons: memory (similar to RMSnorm) and optimization stability」。转写里的说法一致：原始 transformer 的 linear terms 全都带 biases，但大多数实现直接把 biases 整个丢掉（most implementations actually just drop the biases entirely）；这又是一个「算术强度不高、却相对内存密集」的操作（not very arithmetically intense, but fairly memory intensive），顺手删掉就能拿到一次 free systems win。讲者还 offhand（顺口）提了一句：某些情况下 bias 项也可能诱发稳定性问题（there's also some cases where the bias terms can also induce stability issues），所以它在别的方面也有用，但删掉它们的主要动机仍然是「从系统角度把事情简化」（simplify things from the systems perspective）。

![丢弃偏置项：原始 Transformer 与现代实现的 FFN 公式对照](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0014_00-18-49.jpg)

从 18:51 起，讲者给 layer norm 这整个话题收尾，评价是「故事相当简单」（the story is pretty easy）——简单在大家的做法相当标准化（what people do is fairly standardized），而且我们对 layer norm 在做什么的理解「虽然不在深层理论层面」（not at like a deep theoretical level）却相当到位。他把结论串了一遍：所有人都把 layer norm 移出残差流，通常用 pre-norm，但他补了一句很重要的保留——「我觉得这部分可能只是因为 LLaMA 2 这么做了」（I think this might partially be because Llama 2 did that）；我们大致知道怎么用 layer norm 控制 gradient spikes、让 signal propagation 保持良好（keep signal propagation nice）；并且现在基本总用 RMSNorm，也理解了「丢 bias 项」背后的一般原则：让系统保持更高的算术强度（keep our system more arithmetically intense），同时表达能力不变（keeping the expressive power the same）。

紧接着是本段最有方法论分量的一段自省。讲者说，架构里很多东西「不令人满意」（unsatisfying），因为你根本没法事先把它们推理出来（you can't really reason about this beforehand）：我们事先并不知道丢掉 bias terms 是可行的（we don't know beforehand that dropping the bias terms is okay）。今天能说「在典型语言建模负载下，丢掉 linear 与 RMS norm 的 bias 是 OK 的」（dropping the bias terms on both the linear and RMS norm is okay for typical language modeling workloads），靠的是大量实验和如今「集体积累的知识」（collectively acquired knowledge）。他明确指出，这类陈述「只能建立在我们横向考察大量模型的基础之上」（on the basis of what we do when we look at a variety of different models）——这句话把本段一路从 00:41 的 survey lens、到 04:52 那张大表、再到 19:42 的结论，闭成了一个环。20:09 他问了一句「any questions for layer norm stuff?」，随后 20:13 转入 activations。

本章的具体数字与参数集中在两处：其一是公式里 $xW_1+b_1$、$W_2+b_2$ 这四个被删掉的参数，以及 $\sigma(xW_1)W_2$ 里剩下的两个权重；其二是与上一章呼应的 0.17% FLOPs 与约 25% runtime 这组量级，还有作为路径依赖例子的 LLaMA 2。从前后关系看，本章既把第 5 章的原则推广到所有线性层，又收束了自 07:29 以来「归一化位置—归一化形式—偏置」这一整条线，为 20:13 起的激活函数话题腾出空间；而图上那句「if they're not gated」的限定，恰好又预告了下一章要讲的门控前馈层——门控 FFN 的公式与这两个都不同，不能直接套用。

几个容易踩的坑：第一，「bias 项会诱发稳定性问题」讲者只是顺口一提，主要理由始终是系统层面的内存与简化，别把主次颠倒；第二，$\text{FFN}(x)=\sigma(xW_1)W_2$ 明确带「if they're not gated」的前提，看到下一章的 SwiGLU 时不要拿这个式子去套；第三，结论被限定在「typical language modeling workloads」（典型语言建模负载），换任务、换规模不能自动外推；第四，pre-norm 之所以流行「可能只是 LLaMA 2 的路径依赖」，这是讲者自己的存疑，不要把它读成已被证明的最优性。把这几条保留意见连同结论一起记住，才不会把一段经验性讲述误读成设计手册。

---

## 激活函数动物园与门控线性单元的引出 [20:00 - 22:18]

20:13 讲者切入 activations，第一反应是「这里有一整个动物园」（there's a whole zoo of activations）。幻灯片「Activations」（kf_0015）正文只写了一句「A whole zoo of activations ..」，中间列出一长串名字：ReLU, GeLU, Swish, ELU, GLU, GeGLU, ReGLU, SeLU, SwiGLU, LiGLU，下方三个连问：「What are these things? What do people use? Does it matter?」。转写里念到的清单是 ReLU、GELU、Swish ELU、GeGLU、SeLU、SwiGLU、LiGLU——两者对照会发现画面上还多出 GLU 与 ReGLU，且拼作 GeLU；这类「图上出现但没念到」的名字正是本次要补进正文的增量。讲者还自嘲：早些年受统计学/机器学习训练时，他曾「以永远不知道 SwiGLU 是什么为傲」（I will make it a point of pride to never know what a SwiGLU is），但现在必须对这些对象有个大体认识，尤其要搞清名字里哪一部分真的影响性能（which parts of these names actually kind of matter for performance）。

第一个判断是：用相当普通的激活也能把语言模型训出来。讲者说 Chinchilla 大概是这一组里最好的模型（Chinchilla is probably the best model out of that group），但就算只用 ReLU，也完全可以训练一个 reasonably performant 的语言模型，「这没什么不对」（there's nothing wrong with that）。幻灯片「A few of the common activations」（kf_0016）把前两种写得很具体：ReLU 对应 $FF(x)=\max(0,xW_1)W_2$，配一条在 0 处折起的曲线，「Notable models」列的是 Original transformer, T5, Gopher, Chinchilla, OPT；GeLU 对应 $FF(x)=\text{GELU}(xW_1)W_2$，并给出了画面独有的定义式 $GELU(x):=x\Phi(x)$（$\Phi$ 为标准正态累积分布函数），曲线在原点附近有一小段下凹，「Notable models」是 GPT1/2/3, GPTJ, GPT-Neox, BLOOM。转写对 GELU（Gaussian Error Unit，高斯误差单元）的解释与曲线吻合：它与 ReLU 唯一的差别就是底部这个 tiny divot（小凹陷），对激活值的大部分区域毫无影响，却改变了零点附近的梯度（changes the gradients right near zero）；GPT-3 用的就是它，按现代标准不算新，但「完全是个过得去的语言模型」（a perfectly good large language model）。

![常见激活函数：ReLU、GeLU 的公式、曲线与各自代表模型](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0016_00-21-36.jpg)

但讲者随即把重心挪走：真正「有戏」（where most of the action is）的是 gated linear units（门控线性单元），比如 SwiGLU 和 GeGLU。这一点和 layer norm 很像——「几乎所有可信的现代语言模型都用某种 gated linear unit」（almost all credible modern language models use a gated linear unit of some kind）。kf_0016 底部已经提前预告了「SwiGLU / GeGLU (next slide..)」，并把「Notable models」写成 Llama, PaLM, T5 v1.1, most models post 2023，也就是 2023 年之后的大多数模型。那么什么是 gated linear unit？21:52 起讲者先回到一个最标准、最朴素的 ReLU 前馈层做铺垫：拿输入 $x$，先乘 $W_1$，再逐元素在 0 处做阈值（I entry-wise threshold at zero），最后乘 $W_2$ 得到输出——「非常直观的 ReLU 网络」（very straightforward ReLU network）。把这段口述直译成代码，大致是：

```python
def ffn_relu(x, W1, W2):
    h = x @ W1            # 第一层线性投影
    h = torch.relu(h)     # 逐元素在 0 处阈值
    return h @ W2         # 第二层线性投影，得到输出
```

这段实现只对应讲者此刻描述的「两矩阵 + 一次阈值」结构，尚未引入任何门控分支。转写在 22:12 之后接到一句常被提到的架构设计经验——「gating is often very helpful」（门控往往很有帮助），讲者特意声明这不是他的个人经验（I don't say this as my personal experience），本段也就在这个悬念处结束（22:18），门控单元的具体公式留到下一段。

本章的具体数字与标识符包括：$GELU(x):=x\Phi(x)$ 里的 $\Phi$、$FF(x)=\max(0,xW_1)W_2$ 里的 $W_1,W_2$ 与阈值 0、图例里 GPT1/2/3、GPTJ、GPT-Neox、BLOOM、T5 v1.1、Llama、PaLM 等模型名，以及「most models post 2023」这个时间界线。从前后关系看，本章是第 2 章预告的 Common architecture variations 三大块（Activations/FFN、Attention variants、Position embeddings）中的第一块；它同时回应了第 1 章基线的「FFN: ReLU」与 A1 里「用 SwiGLU 而不是 ReLU」的差异，也接住了第 6 章图上「if they're not gated」那个限定——门控 FFN 之所以不能用 $\sigma(xW_1)W_2$ 表示，原因要到下一段才揭晓。

最后强调几处边界。第一，本段转写与画面都只讲到「标准 ReLU FFN + gating 有帮助」为止，并没有给出 GLU 的完整公式、第三个矩阵或 FFN 维度的 2/3 缩放，那些属于下一段内容，不提前补写；第二，激活函数名以画面为准（GeLU、LiGLU、ReGLU），不要按转写的含糊发音擅自「纠正」；第三，「几乎所有可信的现代 LM 都用 GLU」是观察性概括，不是定理，讲者自己也用 Chinchilla、GPT-3 说明非门控激活并非不能用；第四，GELU 那个小凹陷影响的是零点附近的梯度，而不是整体输出形状，别把它理解成「换了个完全不同的函数」。这些边界合起来，也正好呼应全段反复出现的态度：能写进笔记的是被画面或转写支持的内容，含糊处宁可留白。

## 门控线性单元：从 ReLU 前馈层到 ReGLU [22:18 - 25:00]

本段从「激活函数该怎么改」这个具体问题切入。前面几章已经把归一化位置、RMSNorm、偏置这些骨架层面的选择梳理完毕，接下来轮到 Transformer block 里真正承担非线性变换能力的前馈层（feed-forward layer，FFN）。标准前馈层是「升维—逐元素非线性—降维」三步：先用 $W_1$ 把 $d_{\text{model}}$ 维输入投影到更宽的 $d_{\text{ff}}$ 维中间层，过一个逐元素（entrywise）激活，再用 $W_2$ 投回 $d_{\text{model}}$ 维。若激活取 ReLU，写作 $FF(x)=\max(0,xW_1)W_2$。这里的 $\max(0,\cdot)$ 是逐元素作用的，每个中间维度独立决定「通过多少」，彼此之间没有交互。讲者提出的启发式改动非常朴素：既然已经有一路逐元素非线性，为什么不再补一路「门」（gate）去逐元素地调制它的输出？所谓门控，就是让同一输入再经过一个线性变换 $V$，得到一个与激活输出同尺寸的向量，然后两者逐元素相乘，用后者当开关决定前者保留多少。gating 一直是架构设计里非常有效的原语（LSTM 的输入门、遗忘门就是同一直觉），而它在语言建模里同样奏效。

幻灯片把这一推导写得相当清楚。标题是 "Gated activations (*GLU)"，第一句 "GLUs modify the 'first part' of a FF layer" 点明门控只改动前馈层的「前半段」，也就是 $W_1$ 之后、$W_2$ 之前那部分，降维投影 $W_2$ 完全不动。原式 $$FF(x)=\max(0,xW_1)W_2$$ 中的 $\max(0,xW_1)$（幻灯片用红色标出）被替换成 $\max(0,xW_1)\otimes(xV)$，其中 $\otimes$ 表示逐元素乘（Hadamard 积），配文 "Instead of a linear + ReLU, augment the above with an (entrywise) linear term" 说明新增的 $xV$ 是一个不带非线性的线性项。于是得到 "This gives the gated variant (ReGLU) – note that we have an extra parameter (V)"，最终公式为 $$\mathrm{FF}_{\mathrm{ReGLU}}(x)=\big(\max(0,xW_1)\otimes xV\big)W_2$$。讲者口述里也强调：原来是 $XW_1W_2$，现在是 $XW_1$ 被 $XV$ 门控，再 "down project it back with $W_2$"，而且 $XV$ 与 $\max(0,XW_1)$ 是同一尺寸的激活，因此逐元素乘在维度上天然对齐。

![从 ReLU 前馈层到 ReGLU 的公式推导：引入门控矩阵 V 与逐元素乘](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0017_00-23-00.jpg)

命名规则是本节的一个小知识点，也最容易记混：把「第一个激活的名字」接上 "GLU"（Gated Linear Unit，门控线性单元）即可。ReLU 对应 ReGLU，即 ReLU gated linear unit；把带零点附近小凹陷（讲者口述为 "a little divot at the bottom"）的 GELU 换进去就是 GeGLU；把 Swish（即 $x\cdot\sigma(x)$，x 乘 sigmoid）换进去就是 SwiGLU。转写在 23:31 附近有一处口误式的绕圈——"if you take a SwiGLU, which is X times a sigmoid, then you will get a SwiGLU"——实际想表达的是拿 Swish 这个激活加上门控得到 SwiGLU，读转写时不要被这句重复误导。这一族命名几乎覆盖了全部现代模型：Google 系偏好 GeGLU，Gemma、T5 都用；而「LLaMA 后裔」连同 PaLM 一律用 SwiGLU。讲者的判断是 SwiGLU 更主流（"probably the more dominant one"），但在门控单元内部之间选哪个 "doesn't really matter"，差别远小于「门控 vs 非门控」这条轴。

参数层面的数字在这里第一次出现，也是后面所有对照实验的前提。门控版本多了一个矩阵 $V$，于是 "three matrices instead of two matrices"，即 $W_1,W_2,V$ 三个而非原来的 $W_1,W_2$ 两个。若不调整，参数量会从约 $2\,d_{\text{model}}\,d_{\text{ff}}$ 涨到 $3\,d_{\text{model}}\,d_{\text{ff}}$。讲者给出的经验处方是把前馈维度（即 $W_1$ 的输出维度 $d_{\text{ff}}$）缩小到原来的 $2/3$，这样 $3\times\frac{2}{3}=2$，总参数量与原 MLP 持平；他明确说这是 "general rule of thumb"、"not really an iron rule"，即通用经验法则而非铁律。这个 $2/3$ 会在下一章的实验对照里反复出现，也是后面超参数章节里「门控模型 FFN 倍率约 2.67 而非 4」的源头。讲者还特意把它称作「a semi-important piece of trivia later」，提醒听众记住这个因子。

与前后内容的关系上，本节的改动和前面讲 RMSNorm、去偏置一样属于「小手术」：不动残差流、不动注意力、不动层数，只是在非线性里加一路乘法。易错点有三处。其一，门控是逐元素乘 $\otimes$，不是矩阵乘，也不是把两路拼接后再过一个线性层；若把 $\max(0,xW_1)$ 与 $xV$ 拼起来再投影，就变成完全不同的模型。其二，$V$ 的输入是原始 $x$（或同一归一化之后的 $x$），不是激活之后的 $\max(0,xW_1)$，幻灯片写作 $xV$ 而非 $\max(0,xW_1)V$。其三，门控带来的额外参数必须靠缩小 $d_{\text{ff}}$ 配平，否则任何「GLU 更好」的对比都是参数不等价的不公平比较——这正是下一章反复强调 "parameter matched comparison" 的原因。局限也很明显：门控提高的是表达力与优化质量，却多出一次 $xV$ 的矩阵乘，虽然讲者后文说「without much of computational cost」，但它并非零成本，只是被 $2/3$ 的维度缩减压了回去。

## 参数配平 2/3 与门控有效性的实验证据 [25:00 - 27:26]

上一章留下的核心疑问是：门控到底是真的更好，还是仅仅因为多塞了一个矩阵 $V$、多了参数才显得更好？讲者用 Noam Shazeer 的原始论文给出答案。那篇提出 GLU 变体的论文里，各项增益的绝对幅度 "very small deltas"，但 "consistent deltas"——幅度小却方向一致。讲者特别称赞 Shazeer 的实验习惯：他的很多论文都会做 error bar assessments，也就是训练多个 replicate（重复实验）再比较，而不是只跑一次就下结论；在这种带误差棒的对照里，GLU 变体 "almost always consistently better than the non-GLU variants"。关键在于这是 "a parameter matched comparison"：Shazeer 始终做 $2/3$ 调整，确保所有模型 total number of parameters 完全相同。于是 GLU 的收益不能归因于参数变多，而只能归因于门控结构本身。讲者把这一步称作 "in some ways a free win"，并说 "almost everyone uses a GLU"。

配平关系可以写成一个很小的算式。若原前馈层参数量约为 $$N_{\text{plain}}\approx 2\,d_{\text{model}}\,d_{\text{ff}},$$ 门控后变成三个矩阵，即 $3\,d_{\text{model}}\,d_{\text{ff}}'$；令二者相等就得到 $$3\,d_{\text{model}}\,d_{\text{ff}}'=2\,d_{\text{model}}\,d_{\text{ff}}\quad\Longrightarrow\quad d_{\text{ff}}'=\tfrac{2}{3}d_{\text{ff}}.$$ 这里的 $2/3$ 既是上一章的经验法则，也是本章所有表格能被公平解读的前提。讲者还提到后续 "more controlled systematic comparisons"：同一篇 Noam et al. 2020，以及 Google 在 2020 年代做的一批大规模架构对比论文。后者有个必须记住的限定——它们用的是 T5 架构（encoder-decoder），"and not autoregressive language model"，所以结论不能直接搬运到纯自回归语言模型上；但在这类大范围扫描里，SwiGLU、GeGLU 以及 GLU 家族 "do significantly better at loss or the other downstream metrics"，结论相当一致。

![Narang 等人 2020 年激活函数对比表：223M 参数、11.1T Ops 下各变体的损失与下游指标](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0019_00-25-47.jpg)

本段信息量最大的一张图是 "Do gated linear units work (2)?"，副标题 "Yes, with other works corroborating Shazeer 2020"，右下角标注出处 "Narang et al 2020"。表格共九列：Model、Params、Ops、Step/s、Early loss、Final loss、SGLUE、XSum、WebQ；十二行模型的 Params 全部是 $223M$、Ops 全部是 $11.1T$，说明这同样是一次严格的参数与算力配平对照。几个值得抄进笔记的数字：Vanilla Transformer 的 Step/s 为 $3.50$、Early loss $2.182\pm0.005$、Final loss $1.838$、SGLUE $71.66$、XSum $17.78$、WebQ $23.02$；GeLU 为 $3.58$、$2.179\pm0.003$、$1.838$、$75.79$、$17.86$、$25.13$；GLU 为 $3.59$、$2.174\pm0.003$、$1.814$、$74.20$、$17.42$、$24.34$；GeGLU 为 $3.55$、$2.130\pm0.006$、$1.792$、$75.96$、$18.27$、$24.87$；ReGLU 为 $3.57$、$2.145\pm0.004$、$1.803$、$76.17$、$18.36$、$24.87$；SwiGLU 为 $3.53$、$2.127\pm0.003$、$1.789$（全表最低 Final loss）、$76.00$、$18.20$、$24.34$。表中用加粗标出的最佳项集中在门控族：ReGLU 拿到最高 SGLUE $76.17$ 与最高 XSum $18.36$，SwiGLU 拿到最低 Final loss $1.789$，GeLU 拿到最高 WebQ $25.13$。反过来，非门控里表现最差的是 SeLU（Early loss $2.315\pm0.004$、Final loss $1.948$、SGLUE $68.76$、XSum $16.76$、WebQ $22.75$）和 ELU（Final loss $1.932$、SGLUE $67.83$），Sigmoid 也有 $2.291\pm0.019$ 的 Early loss。Step/s 一列在 $3.47$ 到 $3.63$ 之间波动，说明门控并未明显拖慢每秒步数——这与「几乎无计算代价」的定性判断吻合。

讲者接着把这条轴收束成一句可迁移的结论：门控的变体很多（ReGLU、GeGLU、SwiGLU、LiGLU 都出现在表里），但真正重要的单一维度是 "gating for these nonlinearities is actually quite important"，它给出 "a nice boost without much of computational cost"。同时他也明确否认门控是必要条件："that's not to say that gated linear units are necessary"。两个反例被点名：GPT-3 没用 GLU；NeMo Tron 340B 用了 squared ReLU，讲者评价为 "a kind of a crazy choice, but that works, too"，而且 "both of these models are perfectly performant"。也就是说，不用门控也能训出一线模型，只是 "it's actually quite rare to see anything that's not trained on a gated linear unit"，证据整体指向使用门控技巧能获得稳定增益。

与前后内容的关系：本章是对上一章「结构改动有没有用」的实证回答，也为 29:30 之后那张架构共识汇总表里 "Gating: GLUs are consensus now" 提供了依据。易错点与局限至少有三条。第一，$223M$ 参数和 $11.1T$ Ops 是 Narang 表格里的设定，属于 T5 系的小规模对照，不能当成大模型的普适数值；第二，表里 Final loss 的差距是 $1.838$ 对 $1.789$ 这种量级，属于讲者口中的 "small deltas"，靠多次重复实验和误差棒才显出一致性，单次训练很容易被噪声淹没；第三，GLU 族内部（SwiGLU 与 GeGLU 谁更好）在这张表里并没有稳定高下之分，讲者反复说 "doesn't really matter"，若据此硬要排个绝对名次就是过度解读。

## 串行与并行 Transformer 块 [27:26 - 29:30]

讲者把话题从「块内怎么改」转到「块内两个子层怎么排」，并预先给这个想法定了调："a really fun idea, but one that I think now the test of time has shown maybe is not quite as good or maybe not as popular of an idea"。标准 Transformer block 是串行（serial）的：先算 attention，再算 MLP，一个接一个。对系统思维强的人来说，这构成一个依赖瓶颈——"I have to wait for the computation of one to do the other"；如果两个子层能并行，就有机会启用新的系统优化，于是问题变成 "Could we parallelize the transformer block?"。配套的第一张示意图（27:11）画的就是这个串行结构：从下往上依次是 Norm、Causal Multi-Head Self-Attention、Dropout、Add、Norm、Position-Wise Feed-Forward、Dropout、Add，两条残差旁路分别从输入绕到两个 Add，清楚地显示 FFN 的输入依赖 attention 的输出。

这个想法最早出自 GPT-J，即开源社区复现 GPT-3 的尝试；讲者顺带评价 GPT-J "has been surprisingly influential in sort of propagating a lot of ideas"，并说 Google（PaLM 也一样）"is actually surprisingly bold with the architectures that they do"。PaLM 报告里的描述是把上面那种嵌套（sequential / nested）结构拆开：不再让 MLP 吃 attention 的输出，而是把 MLP 和 attention 的输出直接相加，一起加回 residual stream。串行与并行的公式并排写在幻灯片引文里，串行是 $$y=x+\mathrm{MLP}\big(\mathrm{LayerNorm}(x+\mathrm{Attention}(\mathrm{LayerNorm}(x)))\big),$$ 并行则是 $$y=x+\mathrm{MLP}\big(\mathrm{LayerNorm}(x)\big)+\mathrm{Attention}\big(\mathrm{LayerNorm}(x)\big).$$ 差别只在 $\mathrm{LayerNorm}$ 的参数：串行里 MLP 看到的是 $x+\mathrm{Attention}(\cdot)$，并行里两个子层看到的是同一个 $\mathrm{LayerNorm}(x)$。也正因如此，"if implemented right, LayerNorm can be shared, and matrix multiplies can be fused"——共用一个归一化、把两路输入矩阵乘融合成一次，就是所谓 "additional systems optimizations" 的来源。

![并行层幻灯片：PaLM 论文给出的串行与并行块公式、15% 加速与 8B/62B/540B 消融结论](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0021_00-28-34.jpg)

幻灯片 "Parallel layers" 顶行写明 "A few models (GPTJ, PaLM, GPT-NeoX) do parallel layers. Originally in GPT-J"，中间框里是 PaLM 论文原文（引用 Wang & Komatsuzaki, 2021），把数字讲得非常具体："The parallel formulation results in roughly 15% faster training speed at large scales, since the MLP and Attention input matrix multiplications can be fused." 更关键的是消融结论："Ablation experiments showed a small quality degradation at 8B scale but no quality degradation at 62B scale, so we extrapolated that the effect of parallel layers should be quality neutral at the 540B scale." 也就是说 $8B$ 有小幅质量下降、$62B$ 没有下降，作者据此外推到 $540B$ 应当是质量中性的——注意这是「外推」而非实测，是本段最容易被误读成实验结论的一句。幻灯片底部另列 "Recent Models: Cohere Command A, Falcon 2 11B, Command R+"，口述里也提到 Cohere 由前 Transformer 作者创办、做了大量 Google 风格的优化，因而沿用了这一架构，"but not very many others"。

接下来是这章真正的判断：并行层 "has really fallen out of popularity over the past, I think, 2 years"。讲者给出的主因是串行形式的系统优化已经做得足够好，"the systems gains from the second one just isn't worth the small hits to representation power"，即从串行改并行带来的表达力损失不划算。他把损失说得很形象："Effectively, you can think about it as you've lost half of your depth"——串行的两个子层有先后两次非线性变换，并行把它们压成一次并列的加法，等效深度大致减半，这对模型是 "deleterious" 的。这里转写出现一处口误（"going from uh parallel to serial"），按上下文与「失去一半深度」的比喻，方向应是「从串行改成并行」，读笔记时以公式和比喻为准。这一判断与 40:19 的现场问答相互印证：PaLM 论文对并行层 "very confident"、"no performance drop, 15% systems utilization improvement"，但后来的 Google 模型大多不再使用它，讲者把这看作 "an implicit signal that actually there might be some losses"，并坦言 "no one's done the ablations as far as I know on parallel versus serial controlled nice ablations at least"——公开的、控制良好的串行 vs 并行消融并不存在。

与前章的关系：门控、归一化、偏置都是块内的小改动，而串并行是块内数据流拓扑的改动，幅度更大、风险也更大，因此成了本段里唯一一个「曾经流行又退回」的选择。易错点在于别把 $15\%$ 当成无条件收益：它明确限定 "at large scales"，且以能融合矩阵乘为前提；若实现时没有共享 LayerNorm、没有融合，收益会缩水甚至消失。另一个易错点是别把 "quality neutral at 540B" 当作实测结论，它只是从 $8B$ 与 $62B$ 两点外推出来的推断，讲者在问答里正是对这一点保持保留。

## 架构共识汇总与原始 Transformer 的生命力 [29:30 - 31:09]

讲者用「这段讲得这么短」本身来传递一个元结论："the fact that this is so short should kind of suggest to you how much the original transformer formulation has somewhat stood the test of time"。数一数真正被改动的地方，只有归一化放在哪（where the norms go）、要不要偏置项（whether we have bias terms）、要不要给 MLP 加门（whether we gate the MLPs）这三类，"those are actually pretty minor changes compared to all the things that you can do"。他也预先挡住一个可能的质疑：确实存在大量改变 attention 的 Transformer 替代品，但 "you'll have to wait until next lecture"；下一讲会掺一点 state space model 的内容，而只要还停留在 "dense attention land"，"the architecture from the original transformer paper is pretty close to what we do"。换句话说，本讲覆盖的是「核心 attention 方法」内部的架构选择，而非注意力机制本身的替换。

紧接的一句把视角从「已定型的共识」转向「仍在变动的领域」："the thing that is very different across implementations ... is how you do kind of position dependence and incorporate information from other positions, right? So, the core attention component in some sense." 讲者认为位置依赖的注入方式才是各实现之间差异最大、也最 "still in flux" 的地方，这为下一章从位置编码切入做了铺垫。换句话说，前面四章（归一化、RMSNorm、偏置、门控、串并行）属于「已经收敛」的部分，而位置编码属于「还在演化」的部分，这个分界是理解整讲结构的关键。

![架构共识汇总：pre/post norm、RMSNorm、门控、串并行四条结论与右侧模型对比表](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0022_00-29-58.jpg)

"Summary: architectures" 这张幻灯片把共识压成四条加粗小标题，每条配一句判断，值得逐字记下。"Pre-vs-post norm: Everyone does non-residual norm (except OPT350M), likely with good reason."——唯一例外是 $OPT350M$。"Layer vs RMSnorm: RMSnorm has clear compute wins, sometimes even performance"——收益主要来自算力，有时连效果也略好。"Gating: GLUs are consensus now"——门控已是共识。"Serial vs parallel layers: Most models now use serial layers"——多数模型回到串行。右侧是一张深色背景的模型对比表，纵向按年份列出从 $2017$ 年 Original transformer（Vaswani 等）、GPT、GPT-3、GPT-J、GPT-NeoX、LLaMA，到 Gopher (280B)、Chinchilla (70B)、OPT (175B)、PaLM (540B)、BLOOM (176B)、LLAMA (7B)、LLAMA 2 (70B)，再到 Mistral 7B、Nemotron-4 (340B)、Gemma 2、Falcon 2 11B、Command A、Olmo 2、Gemma 3、Qwen 2 等数十个模型；横向用色块标出每层的归一化位置（Pre-LN / Post-LN）、Serial 或 Parallel、LayerNorm 还是 RMSNorm、以及激活函数（ReLU、GeLU、SwiGLU、GeGLU）。讲者口述给出读表口诀："blue here is RMS norm block as layer norm"——蓝色表示 RMSNorm；"Serial versus parallel layers, the blue one's parallel, the rest is serial"——并行的是少数派；"these ones on the right, these are GLUs, almost always ... gated linear units for modern models"。

读这张表有两处需要格外小心，讲者自己也点了出来。第一，"Some of these, uh, ones that I marked as post-norm are actually pre and post-norm"，即表里被标成 post-norm 的若干行其实是前后都放归一化的 double norm，表格的二分标记有简化，不能逐格当精确事实。第二，关于右侧激活列，转写作 "almost always with the exception of things like, uh, Falcon, which use a gated linear unit"，句子本身自相矛盾（前面说几乎都是 GLU，后面又说 Falcon 是例外且「用了」门控），语义含糊，按「转写含糊处宁可略过」的原则只能记成「Falcon 是被点名的少数例外」，不宜替它断言到底用没用 GLU。相关的 pre-norm 公式本章可以顺带复写一遍，作为与前面章节的呼应： $$x_{l+1}=x_l+\mathrm{MultiHeadAttn}\big(\mathrm{LayerNorm}(x_l)\big),$$ 归一化在子层之前、残差流之外，这正是表里「non-residual norm」所指的结构。

易错点与局限：这张汇总表是「横向看大量已发布模型后归纳出的经验规律」，不是从理论推出的定理，讲者在前文已反复承认架构结论 "cannot be reasoned about a priori"。表格的样本也偏向公开模型，闭源模型（如 GPT-4 一类）的架构细节并不完整，因此「共识」的强度应理解为「在可观测样本里的高度一致」。此外，本章的短小本身就是信息：它说明在 dense attention 范围内架构创新的空间其实有限，真正的变量已经转移到位置编码与注意力变体，而后者被明确推到下一讲。

## 位置编码为何必要：四类方案与置换不变性 [31:09 - 33:01]

讲者先给出一条必须记住的前提："attention is positionally independent, right? They're just inner products, so you can just shuffle them and attention would be the same if you don't have a position embedding." 注意力分数本质是 query 与 key 的内积，对 token 集合做置换不会改变任意一对的内积值，因此在没有位置编码时，模型看到的是一袋词而非一个序列。这就解释了为什么「如何编码位置、如何把其他位置的信息引进来」是架构里差异最大、仍在变动的部分——它是 dense attention 框架里唯一必须额外注入、又没有被标准答案固定的东西。讲者用 "there are lots of different ways that you can encode position into a transformer" 引出四类方案，并在幻灯片里逐一配了公式和代表模型。

第一类是正弦余弦嵌入（Sine embeddings），配文 "add sines and cosines that enable localization"，公式是 $$\mathrm{Embed}(x,i)=v_x+PE_{pos},$$ 其中 $$PE_{(pos,2i)}=\sin\big(pos/10000^{2i/d_{\text{model}}}\big),\qquad PE_{(pos,2i+1)}=\cos\big(pos/10000^{2i/d_{\text{model}}}\big).$$ 分母里的 $10000^{2i/d_{\text{model}}}$ 让不同维度对应不同波长，讲者口述把直觉类比成傅里叶变换："if you have sines and cosines, then you can kind of recover position from that no matter what"；代表模型只有 "Original transformer"。第二类是绝对位置嵌入（Absolute embeddings），"add a position vector to the embedding"，即 $$\mathrm{Embed}(x,i)=v_x+u_i,$$ 每个位置 $i$ 有自己独立可学的向量 $u_i$，代表模型是 "GPT1/2/3, OPT"。第三类是相对位置嵌入（Relative embeddings），关键在于它 "add a vector to the attention computation"——不是加到词向量上，而是把随位置差变化的偏移直接加进注意力计算： $$e_{ij}=\frac{x_iW^Q\big(x_jW^K+a_{ij}^{K}\big)^T}{\sqrt{d_z}}.$$ 讲者口述把 $a_{ij}^{K}$ 解释成 "if you're three positions off, sort of the attention matrix gets a different offset added to it"，代表模型是 "T5, Gopher, Chinchilla"。第四类 "Rope embeddings (next slides..)" 的代表模型列为 "GPTJ, PaLM, LLaMA" 以及 "Most 2024+ models"。

![位置编码的多种变体：正弦、绝对、相对、RoPE 的公式及代表模型](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0023_00-31-22.jpg)

讲者对这四类的时间线补充了两点口述信息。其一是 RoPE 的来历："rope, you know, in some ways came out of nowhere"，"originally I think this was also a GPT-J innovation"，来自 "not very well known sort of blog post and paper combination, from an author in China"——即一个不太出名的博客加论文组合，出处并不显赫。其二是它的统治力："most models past 2024 use this type of embedding"，也就是 $2024$ 年之后的绝大多数模型都用 RoPE。这里还埋着一个与上一章的呼应：GPT-J 既是并行层的源头，又是 RoPE 的早期推手，讲者前面说它 "surprisingly influential in propagating a lot of ideas" 正是为此。相对嵌入一侧，口述里点名 "models like T5 and Chinchilla use kind of this scheme"，与幻灯片表格一致；绝对嵌入一侧则说 "a number of other sort of large models that followed soon after that used absolute embeddings, where each position had its own different embedding"。

与前章的关系很直接：上一张汇总表把归一化、RMSNorm、门控、串并行列为「已收敛」，本章则打开那个「仍在变动」的维度。四类方案之间的差别可以浓缩成一句话——注入位置信息的「位置」不同：正弦与绝对嵌入加在词向量 $\mathrm{Embed}(x,i)$ 上，相对嵌入加在注意力分数 $e_{ij}$ 上，而 RoPE 走的是第四条路（下一章展开：乘在向量上）。易错点首先是别把「相对位置嵌入」理解成「一种更聪明的词向量」，它根本不产生可提取的嵌入向量，只是给注意力矩阵加偏置，这一点在下一章会被当作它「不是内积」的判据反复使用。其次是正弦公式里的 $10000^{2i/d_{\text{model}}}$ 与 $\sqrt{d_z}$ 这类量都是幻灯片给出的原文，转写只念到「sines and cosines」「Fourier transform intuition」，写笔记时应以幻灯片为准，不要凭印象改写成别的底数或归一化因子。局限方面，这一节只是分类与铺垫，并未给出「哪一类更好」的证明；讲者明确把评判标准留到下一章——用「内积是否只依赖相对位置」这条公理去逐一检验。

## RoPE：相对性判据、旋转几何与高维实现 [33:01 - 39:02]

讲者先摆出一个带立场的定义："rope, you know, is a relative position embedding. And a relative position embedding, let's make an opinionated stance that I should not care about the absolute position of any words." 也就是说，若 "an apple" 两个词相邻出现，无论它落在句首还是句尾，模型应当给出同样的结果。把这个立场写成数学条件：设嵌入函数 $f$ 同时接收词的身份 $x,y$ 与绝对位置 $i,j$，则理想中的相对位置编码应满足 $$\langle f(x,i),\,f(y,j)\rangle=g(x,y,i-j).$$ 等号右边只出现相对差 $i-j$，不出现 $i$ 或 $j$ 本身；换句话说 "the attention function only gets to depend on the relative position (i-j)"。幻灯片标题 "RoPE: rotary position embeddings" 下把这行称作 "High level thought process"，并追问 "How do existing embeddings not fulfill this goal?"，随后逐条否决了上一章的四类方案中的三类：正弦嵌入 "Has various cross-terms that are not relative"，展开后会出现 $\langle v_x,v_y\rangle+\langle PE_i,v_y\rangle\cdots$ 这类位置与词向量交叉的项；绝对嵌入 "obviously not relative"；相对嵌入虽然确实是相对的，但 $$e_{ij}=\frac{x_iW^Q(x_jW^K+a_{ij}^{K})^T}{\sqrt{d_z}}$$ "is not an inner product"——它直接加在注意力矩阵上，"there's no inner product structure that you can extract out of the..."。讲者在口述里也强调正弦 "has these absolute cross terms that are not relative"，因此 "every existing embedding before it didn't really fulfill this equality"。

解法来自一个很漂亮的几何事实。幻灯片 "How can we solve this problem?" 下只有两条：一是 "We want our embeddings to be invariant to absolute position"，二是 "We know that inner products are invariant to arbitrary rotation"。既然任意旋转都不改变内积，那就取与位置无关的语义词向量作为起点，按 token 所在位置把向量旋转一个角度。二维例子讲得很细：句子 "we know that" 里 we 在位置 $0$，"I'm not going to touch that at all"；know 在位置 $1$，"I'm going to rotate it by some angle"。换成 "of course we know"，we 与 know 仍然相邻，但绝对位置整体右移，于是 "Rotate we by '0 positions'" 变成 "Rotate we by '2 positions'"（下标 $0,1,2$），"Rotate know by '3 positions'"。两者各自多转了两格，相对夹角仍然相差 $1$，"what do you know, the relative angle between these two is still separated by one"。三幅二维箭头图分别标注 "Position independent embedding"、"Embedding 'we know that'" 与 "Embedding 'of course we know'"，浅蓝色虚线弧标出旋转量与保持不变的相对夹角。这样 "anytime we take an inner product, those inner products are going to be invariant of absolute positions"，相对位置被保留、绝对位置被消去。

![用二维向量旋转解释 RoPE：we know that 与 of course we know 的相对夹角保持不变](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0026_00-35-51.jpg)

高维才是难点，也是本章最容易被跳过的一步。讲者指出二维里只有顺时针/逆时针一个选择，"but in high dimensions, there's an infinite space of ways that you can rotate vectors"，于是问 "what do you do in D dimensions?"，答案是 "you do the simplest possible thing and it works"：把 $D$ 维向量切成两两一组的坐标对，每一对独立按二维情形旋转，即 "cut it up into chunks of two, and each pair of two dimensions gets rotated"。各对的旋转角 $\theta$ 不同："Some of them are very low frequency, so they rotate very slowly, so they can capture long-range dependence. Some of them rotate very quickly, so they capture things like, are they neighbors to each other"——低频对负责长程依赖，高频对负责邻近关系。幻灯片 "There are many rotations, which one do you pick?" 给出 [Su et al 2021] 的配图：虚线框里 $(x_1,x_2)$ 这一对 Query/Key 在 $d=2$ 平面上按 $m\theta_1$ 旋转到 $(x'_1,x'_2)$，其中 $m$ 标为 "Position"、"Constant" 标在左上角；下方 "Enhanced Transformer with Rotary Position Embedding" 用彩色方格画出位置 $1,2,3,4,5,6$ 的 Query/Key 经旋转后变成 "Position Encoded Query / Key"；右上角另有一组小图，标注 "the m=9"、$\theta_1,\theta_2,\theta_3,\theta_4$、"only rotate high frequency pairs"、"low positional information / more semantic information"，并写着 "Gemma 4 alternative: just first 2"。底部结论一句最该记住："Just pair up the coordinates and rotate them in 2d (motivation: complex numbers)"——论文用复数给出复杂动机，但直观上就是逐对旋转。讲者口述也提到 Gemma 4（"just came out on Thursday"）的 "proportional rope or P-rope"，"a really strange way to just say that the only thing they rotate is the first two coordinates"，并说 "there's a lot of different things that you can do in this space that end up working"。

![高维旋转的选择：把坐标两两配对在 2D 中旋转，含 Su 等人 2021 原图与 Gemma 4 只旋转前两维](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0027_00-37-26.jpg)

工程实现这一段口述给了可落地的细节。可以把旋转写成对输入向量做一次稀疏的 sin/cos 乘法，"X times W times R, this is going to be your final embedding"，即 $XWR$ 里的 $R$ 是由正弦余弦构成的分块旋转矩阵。讲者反复强调一个易错点："it's really important that I'm multiplying with these sines and cosines rather than using them as embeddings cuz that means that there are no cross terms. And this is purely relative, right? There's no absolute position information that you'll get out of inner products." 换句话说，RoPE 与正弦余弦嵌入长得像，区别是「乘」而非「加」：加法嵌入会引入位置向量与词向量的交叉项，乘法旋转不会，因而保持纯相对性。真正实现时，先按序列的 position IDs 生成 cos/sin 角度，再把它们作用到 query 和 key 上："you're going to apply those cosines and sines onto both your queries and keys for your sort of attention computation"，既可以写成矩阵乘，也可以手动逐元素做旋转，"Fairly straightforward"。位置不变性因此是在注意力层面被强制成立的："you would do this at the attention level rather than at the very bottom"。典型实现形态如下（对应讲者口述的 query/key 投影、生成 cos/sin、再应用旋转三步）：

```python
cos, sin = self.rotary_emb(value_states, position_ids)
query_states, key_states = apply_rotary_pos_emb(
    query_states, key_states, cos, sin
)
# 之后与常规多头注意力相同：softmax(Q K^T / sqrt(d)) V
```

与前后内容的关系：本章把上一章「四类方案」的评判标准兑现成了 $\langle f(x,i),f(y,j)\rangle=g(x,y,i-j)$ 这条公理，并给出唯一满足它的构造；后面 42:03 的问答会再次回到「为什么相对嵌入不是内积」「为什么正弦嵌入能反推绝对位置」。易错点至少有四处：其一，旋转必须作用在成对的坐标上，不能对整个 $D$ 维向量做任意高维旋转（问答里讲者也说没见过可用推广）；其二，旋转施加在 query/key 上、在每次注意力计算时做，而不是在 embedding 层做一次；其三，"$\otimes$ 乘法而非加法嵌入" 是 RoPE 与正弦嵌入的分水岭，混淆二者就会重新引入绝对位置信息；其四，Gemma 4 的 P-RoPE 只旋转前两维是有效的变体，但讲者把它描述为在隐藏维度很小的极小模型上省空间的优化，不能当成通用默认值。局限方面，讲者坦承 "rope ... is a little bit confusing"，一旦理解旋转几何就 straightforward；此外「只依赖相对位置」本身是一个 "opinionated stance"（需要先接受这个前提），并非公认最优，ALiBi 这类注入注意力矩阵的方案同样可行，只是没有成为主流。

## 现场问答与超参数问题导引 [39:02 - 44:36]

39:02 讲者暂停授课 "in case anyone has any questions about the various like architecture bits"，预告接下来要讲 "even lower-level details about hyper parameters"，于是进入约四分钟的问答。第一问是「有没有做高维旋转的论文、高维旋转是不是从来没成功过」，讲者答 "I don't think so"，并解释任何二维旋转都只是这个思路的变体，理论上可以取 "any one manifold that like is a closed loop"，但 "I have not seen that"——即没见到可用的高维推广，这与上一章「逐对二维旋转已经够用」的判断一致。第二问是「怎么把这类知识蒸馏/提炼出来」，讲者的回答很方法论：一是 "looking broadly enough to get a pattern"，也就是本讲正在做的横向比较大量模型；二是 "try it yourself even a much smaller scale to form an intuition and like a theory"，即自己小规模动手建立直觉；同时提醒 "reading any single paper in isolation is very very difficult especially now because no single paper seems to give any full detail for a lot of language models these days"——单篇论文几乎不再给出完整细节，孤立阅读很难拼出全貌。这恰好解释了本讲为什么用汇总表而不是逐篇论文精读。

第三问回到并行层与串行层的精度差是否小到可以忽略。讲者说结论 "really mixed"：PaLM 原论文 "very confident about the use of parallel layers like no performance drop 15% systems utilization improvement"，只读它会以为两者一样好；但 "a lot of the later Google models have stopped using this"，可以当作 "an implicit signal that actually there might be some losses"。他坦言精确数字拿不到，"no one's done the ablations as far as I know on parallel versus serial controlled nice ablations at least"。这补充了 27:26 那一章的证据状态：$15\%$ 是系统利用率提升而非质量提升，$8B/62B/540B$ 的「质量中性」是外推，且缺少控制良好的公开消融。第四问（转写记作 "Eagle and RoPE"）问某变体与 RoPE 的差别，讲者答差别 "really just like which of the coordinates you're rotating"：低频分量本来转得很少，可以丢掉不旋转，"you can drop them if you're really strapped for sort of extra space"，这是给 "teeny tiny models"、隐藏维度很小、没多少激活可分配时做的优化——与上一章 Gemma 4 P-RoPE「只旋转前两个坐标」是同一条思路。

第五、六问专攻相对嵌入的数学性质。有同学问「相对嵌入没有内积结构，是不是因为它只作用在 keys 上」，讲者回答里说作用在 "keys and values" 上并由此产生 "relative effect from where you are"（注意转写与 38:34 口述的 "apply those cosines and sines onto both your queries and keys" 不一致，此处按转写保留，不替讲者改口），随后重申 "you want to not have cross terms"：正弦余弦加法嵌入不仅留下原始向量，还会产生 "weird cross terms between the position embeddings and the word embedding themselves"，于是 "you can kind of back out what the absolute position is"，结论是 "even sine and cosine embeddings are not like pure relative position embeddings"。用公式把上一章的判据写开就是 $$\langle \mathrm{Embed}(x,i),\mathrm{Embed}(y,i)\rangle=\langle v_x,v_y\rangle+\langle PE_i,v_y\rangle+\cdots,$$ 省略号里正是那些能反推绝对位置的交叉项。最后一句问答把标准讲得更严："the issue with this is that it just can't be factorized as an inner product. That's more of an aesthetic problem"——若把约束写成「既要相对、又要能分解成形如 $f(x_i)$ 与 $f(y_j)$ 的内积」，那么注入注意力矩阵的方案根本不在这个解类里；但讲者立刻补了局限："to be fair, there's a lot of embeddings that work this way that do work like Alibi and other kinds of like approaches like do do this kind of inject into the attention matrix and they do reasonably well. It's not necessarily the one that's become the dominant approach"。也就是说，「不能分解成内积」更像审美/形式约束，不是有效性判据；ALiBi 等方案效果尚可，只是没成为主流。

43:41 起话题切到超参数，讲者的切入角度是「什么时候才会关心这些」："hyperparameters are really something that you start to engage with once you like actually have to train a model"；知识停留在抽象层面时不必管，一旦要实例化，问题立刻变成一串：前馈层该多大（"how big should the feed forward size be"）、该有几个头（"how many heads should I have"）、词表该多大（"what should my vocab size be"）、weight decay 或 dropout 要不要、甚至「我有很多 token，还需要正则化吗」（"do I even need to regularize? I have a lot of tokens"）、以及模型该做深还是做宽（"do I need very deep models or very wide models"）。他形容初学者的处境是要在 "a very big high dimensional space" 里搜索，"actually very daunting"；但紧接一句给了安慰与线索："the space of things that people try is actually pretty small. And from that maybe you can start to think about smarter search processes of like where you want to vary things." 也就是说，实际被业界尝试过的组合只占设计空间很小一块，从这块小区域反推该调哪些维度，比盲目网格搜索更有效。转写在本段末尾停在半句 "One of the things that's a really consensus hyperparameter"，属于下一部分的开头，按「不猜测」原则不续写。

![超参数幻灯片：224n 之后关于前馈维度、头数、词表、正则与宽深的五个问题](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0031_00-43-49.jpg)

"Hyperparameters" 幻灯片把这些问题原样列成两组建表问题，标题下写 "Transformer hyperparameter questions you might have had in 224n.."（即 CS224n 这门课之后留下的疑问）：第一组三条是 "How much bigger should the feedforward size be compared to hidden size?"、"How many heads, and should num_heads always divide hidden size?"、"What should my vocab size be?"；分隔句 "And other model setting questions" 之下两条是 "Do people even regularize these huge LMs?" 与 "How do people scale these models - very deep or very wide?"。其中 "should num_heads always divide hidden size" 和 "very deep or very wide" 正是幻灯片独有、转写只以口语形式带过的表述，写笔记时应保留原句。

与前后内容的关系：本章前半段是对前述三处「含糊/例外」的补充澄清（并行层证据强度、高维旋转是否可行、相对嵌入的形式约束），后半段则是从「架构」下潜到「超参数」的过渡，二者共享同一方法论——"looking broadly enough to get a pattern"。易错点有三：其一，别把 $15\%$ 当作并行层的质量结论，它是 systems utilization improvement，且讲者在问答里明确说后续 Google 模型弃用构成反向信号；其二，「不能分解成内积」是形式约束而非性能缺陷，ALiBi 是反例，不能据此断言这类方案无效；其三，转写里 "keys and values" 与前文 "queries and keys" 冲突，属于口述含糊处，笔记只记录不一致、不擅自统一。局限方面，问答里的若干判断（"I have not seen that"、"no one's done the ablations"、"really mixed"）都是讲者基于公开信息的个人判断，而非文献定论；超参数部分本段只提出问题、未给出任何取值，具体数字要等下一部分，切勿把前面章节里的 $2/3$、$4$、$2.67$ 等倍率提前安到这里当作本段结论。

## 前馈维度倍率：四倍经验法则、门控修正与 T5 的 64 倍异类 [44:36 - 49:53]

本段从「出人意料的共识超参数之一」讲起，幻灯片把它命名为 feedforward – model dimension ratio（前馈维度与模型维度之比）。画面 kf_0032 先给出标准两层前馈网络的原文公式 $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$，随后指出真正相关的只有两个维度：前馈维度 $d_{ff}$（即 $W_1$ 的输出宽度，同时也是 $W_2$ 的输入宽度）与模型维度 $d_{model}$。直觉上，$d_{ff}$ 决定 MLP 把残差流抬到多高的特征空间，因而控制这一层能容纳多少非线性特征；倍率越大表示越「宽」、表达越丰富，但参数量与矩阵乘规模也几乎线性上升。幻灯片给出的答案干净得近乎武断：$d_{ff} = 4\,d_{model}$，并特意用斜体强调「This is *almost always* true. There's just a few exceptions.」讲者在口头转述里也说，这是「a rule of thumb that works remarkably well」，是「controls kind of the richness of your MLPs」的那个旋钮，还说会拿数据说明「why maybe this is a fine number to choose」。一个容易被忽略的细节是：这张幻灯片公式里保留了 $b_1$、$b_2$ 两个 bias（偏置）项，而本课前面刚讲过现代实现普遍丢弃线性层偏置；两者结论并不冲突，因为倍率讨论只关心矩阵的两个维度，偏置是否存在不改变 $d_{ff}/d_{model}$ 的取值。

第一类例外来自 gated linear unit（GLU，门控线性单元）。讲者提醒「this is probably cached in your head」：GLU 变体在前馈层里多引入第三个矩阵（SwiGLU / GeGLU 中的 gate 或 up 投影），在维度不变时参数量从两个矩阵变成三个。若要把 MLP 总参数量配平回原来的水平，就必须把前馈维度缩到原来的 $2/3$：

$$\frac{d_{ff}}{d_{model}} = 4 \times \frac{2}{3} = \frac{8}{3} \approx 2.67 \quad (\text{GLU 参数配平}),\qquad 2.67 \times 1.33 \approx 3.5 \quad (\text{LLaMA 2})$$

转写里讲者明确说「you need to scale down by 2/3, right? So most GLU variants, this means that you're going to end up with something like 2.67-ish」，并补充实际论文里这类取值集中在 2.67 到 2.5 之间，「everyone that's uh down here 2.67 to 2.5, this is roughly applying this like 2/3 correction」。总结幻灯片把它写成更精确的分数 $8/3$（「Factor-of-4 rule of thumb (8/3 for GLUs)」），另一张小结幻灯片则写作 $d_{ff}=2.66d_{model}$。第二类修正是 LLaMA 2 团队：他们因为用了 MQA（multi-query attention，多查询注意力，讲者说「MQ A, which I'll talk about later」）这类更高效的注意力头，「we can multiply this ratio by an arbitrary 1.33 and we'll get roughly 3.5」，等于刻意更强调 MLP 一侧、「emphasizes the MLPs a little bit more」。于是讲者归纳：翻遍技术报告，「either 2.6-ish or 3.5 for GLUs um or four if you're doing uh non-GLU models」，非门控约 4、门控约 2.6 或 3.5。这与上一章讲门控激活时「GLU 需把 FFN 维度缩到 2/3 以配平参数」是同一件事的两种说法，前后完全咬合。

真正大胆的是幻灯片编号 Exception #2 的 T5 [Raffel et al 2020]。kf_0033 的正文写着「As we have (and will) see, most LMs are have boring, conservative hyperparameters. One exception is T5 ... which has some *very bold* settings」，并给出 11B 模型的具体参数：$d_{ff} = 65{,}536$、$d_{model} = 1024$，二者相除即「For an astounding 64-times multiplier」。幻灯片还框引了论文原话：对「11B」采用 $d_{ff}=65{,}536$、配合 128-headed attention，得到约 11 billion parameters 的模型；之所以专门放大 $d_{ff}$，是因为「modern accelerators (such as the TPUs we train our models on) are most efficient for large dense matrix multiplications like those in the Transformer's feed-forward networks」——讲者称之为「another like systems-based argument」，与前面 RMSNorm 靠算术强度省访存的思路同源：矩阵乘越大，硬件利用率越高。幻灯片底部另列「Other, recent exceptions – Gemma 2 (8x), SmoLLM/Gemma 3/Gemma 4 (4x, GLU)」，口头转写也提到 Gemma 2「tried to really push a little bit higher」，但强调「I don't think any other model has really gone that high in the feed forward multiplier」。这里有个易错点：64 倍指的是 $d_{ff}$ 相对 $d_{model}$ 的比例，不是参数量倍率；T5 的 $d_{model}=1024$ 本身很小，65,536 这个绝对数放在现代大模型里并不夸张，异常的只是「比例」被拉到 64。

为什么 2.6 到 4 这一档安全？讲者搬出 Kaplan 2020 的 neural scaling laws（神经缩放定律）论文——「This is the classic uh neural scaling laws paper」，并坦言「this wasn't the point of the study, right? It was a scaling laws study」，只是其中一个 panel 顺带做了倍率 sweep。kf_0034 的标题是「Why this range of multipliers?」，正文写着「Empirically, there's a basin between 1-10 where this hyperparameter is near-optimal」；纵轴为 Loss Increase，刻度 0%、2%、4%、6%、8%、10%，横轴为 Feed-Forward Ratio $(d_{ff}/d_{model})$ 的对数坐标（$10^0$、$10^1$），图例两条曲线分别是 $n_{head}=8$ 与 $d_{model}/n_{head}=64$，图下标注「50M Parameters」，右下角出处 [Kaplan+ 2020]。曲线在 1 到 10 之间几乎贴着 0%，讲者说「you lose very little relative to the optimal loss down here」，而一旦「above 10 to 100 or something like that」，损失便「starts really shooting up quadratically」。kf_0035 的三条 bullet 把结论写死：默认 $d_{ff}=4d_{model}$ 与 $d_{ff}=2.66d_{model}$「have worked well for nearly all modern LLMs」；T5 的 $d_{ff}=64d_{model}$ 也能跑，「This hyperparameter choice isn't written in stone」；但改进版 T5 v1.1 在 GeGLU 上回到更标准的 2.5 multiplier，「so the 64-times multiplier is likely suboptimal」，讲者把这戏称为「the punchline of the T5 saga」，还说「T5 was a fine model ... even radical choices can technically work, but it's probably going to be compute inefficient」。局限也很清楚：这张图出自 50M 参数小模型、且是缩放定律研究的副产品，并非针对倍率的严格控制变量实验，它只能证明存在一个平坦盆地，不能给出唯一最优值；因此「the default choices have worked very well for nearly all modern language models. So you can safely choose that」才是可操作的结论。

![前馈维度倍率幻灯片：FFN 两层公式与 d_ff = 4 d_model 四倍法则](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0032_00-45-18.jpg)

![T5 的 64 倍前馈倍率：d_ff=65,536、d_model=1024 与 TPU 大矩阵乘理由](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0033_00-46-42.jpg)

---

## 头维度与头数：H × d_head = d_model 的惯例 [49:53 - 51:31]

第二个共识超参数关于 multi-head attention（多头注意力）里头的「宽度」。讲者说自己在教 224N 时「I always found this to be very strange」：只要有 $H$ 个头，几乎所有人都会把每个头的维度 $d_{head}$ 设成让多头总维度与单头 Transformer 相同的那个值，也就是「you always make sure that you sort of divide the hidden dimension to basically multiply with H」。写成公式就是：

$$d_{head} = \frac{d_{model}}{H}\quad\Longleftrightarrow\quad H \cdot d_{head} = d_{model}$$

直觉是：注意力层的总「带宽」保持为 $d_{model}$，头数 $H$ 只决定把这条带宽切成几份并行处理；切得越细，每个头能看到的子空间越小，但并行度与头间多样性越高。需要强调，这只是一个约定，并非数学必然——讲者立刻补了一句「Of course, this doesn't have to be true. We can arbitrarily change the ratios between head dimensions and model dimensions, but most models do follow this guideline, and it turns out to work pretty well」。也就是说 $H \cdot d_{head}$ 完全可以大于或小于 $d_{model}$，只是「almost everyone does」把它取成恰好相等，而且「for some reason, this is kind of the rule of thumb」。

kf_0036 的表格把「有多普遍」量化了，标题是「How many heads, whats the model dim?」，副标题「Some examples of this hyperparameter」，列出四列 Num heads / Head dim / Model dim / Ratio。逐行读下来：GPT3 为 96 / 128 / 12288 / 1；T5 为 128 / 128 / 1024 / 16；T5 v1.1 为 64 / 64 / 4096 / 1；LaMDA 为 128 / 128 / 8192 / 2；PaLM 为 48 / 258 / 18432 / 1.48；LLaMA2 为 64 / 128 / 8192 / 1；Qwen 3.5 (27B) 为 24 / 256 / 5120 / 1.2。表格下方一句话概括：「Most models have ratios around 1 – notable exceptions by some google models.」这里的 Ratio 以 1 为基准，GPT3、T5 v1.1、LLaMA2 恰好等于 1，说明 $H \times d_{head}$ 与 $d_{model}$ 严格相等；T5 的 16、LaMDA 的 2、PaLM 的 1.48、Qwen 3.5 (27B) 的 1.2 则是明显偏离。讲者口头点名「notable exception of T5, um and even Lambda, which is another Google model，um but really everyone sticks around one」，还提到自己连「the latest and greatest」（指 Qwen 一类新模型）也查过，「the ratios are roughly around one a model head」。这些数字与前一段 FFN 倍率同属「横向比较已发布模型」的方法论：不看单个模型的设计理由，而看整个群体的收敛点；也和前一段一样，讲者只给经验规律、不给理论推导。

易错点与局限在这一章尤其值得留心。首先，表格里 PaLM 行的头维度印成 258（$48 \times 258 = 12384$，与 18432 相比又引出 1.48 这一 Ratio），与常见的 256 写法略有出入，读表时应以画面数值为准，不要自行「修正」成整数，也不要据此强行反推 Ratio 的统一定义。其次，讲者把头维度明确归为「yet another kind of forgiving hyperparameter」（又一个宽容的超参数）：「There's a couple of ablations that people have done. There's once again a pretty wide basin around one that you can sort of get away with」，并直接定性「that one's maybe not the most critical uh hyperparameter」。也就是说，围绕 1 的较宽区间内怎么取都不太影响结果，这与 FFN 倍率在 1–10 的盆地是同一种现象。最后是与前后内容的关系：这一章夹在「前馈倍率」与讲者认为「maybe one of the most critical and interesting ones, I think conceptually」的 aspect ratio（宽深比）之间，起着从「逐项罗列共识」过渡到「控制全局形状的参数」的桥梁作用；也正因为它最不关键，讲者只用很短篇幅就收尾，把时间留给后面的宽深比、词表与稳定性讨论。

![多头注意力头维度示例表：Num heads、Head dim、Model dim 与 Ratio](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0036_00-50-53.jpg)

---

## 宽深比：约 100 的甜点区与系统并行约束 [51:31 - 55:09]

讲者认为概念上最关键、也最有趣的一个超参数是 aspect ratio（宽深比）。定义很直接：当你把模型放大或缩小时，通常先固定一个「模型有多宽 versus 有多深」的比例，再整体缩放，因此这个比例「controls the entire depth-to-width tradeoff as you make models bigger」。写成公式：

$$\text{aspect ratio} = \frac{d_{model}}{N_{layer}} \approx 100$$

也就是每一层大约对应 100 维的模型宽度。直觉上，固定参数预算时，宽度（$d_{model}$）与深度（$N_{layer}$）此消彼长：宽模型单层矩阵乘更大、并行切分更容易；深模型串行步骤更多、理论上表达力更强。讲者先承认「you might think I need a really deep model or really shallow model if I want systems utilization ... And there is a lot of variation, um much more so than other hyperparameters」，但紧接着说「there's actually like a fairly clear sweet spot that most modern models fall into」：不会太深也不会太宽，「You see most models have a ratio about a hundred um D model over N layers ... about hundred sort of width for every layer that you have」，并点名「this is true for like GPT-3 or LLaMA or any one of these models」。

kf_0037 的表格把这个甜点区摊开，标题「Aspect ratios」，副标题「Should my model be deep or wide? *How deep and how wide?*」，上方一句「Most models are surprisingly consistent on this one too!」。表头是 Model 与 $d_{model}/n_{layer}$，逐行为：BLOOM 205；T5 v1.1 171；PaLM (540B) 156；GPT3/OPT/Mistral/Qwen/OLMo 3 128；LLaMA / LLaMA2 102；Gemma 3 87；Gemma 4 61；T5 (11B) 33。左侧用一条竖括号标出「Sweet spot?」，正好框住中间那段约 156 到 61 的区间。总结幻灯片则把可用区间写成「Wide range of 'good' values (100-200). Systems concerns dictate the value」——两处口径略有差别（表格 sweet spot 的中心约 100，总结给的经验区间是 100–200），正好说明这个参数同样是「宽容带」而非唯一点值；把它和上一段的 $H\cdot d_{head}=d_{model}$、更上一段的 4 倍法则并排看，三个参数共享同一种「平坦盆地」结构。

为什么收敛到 100？讲者给出的是表达力与硬件的权衡。极深的模型「get very very annoying to deal with systems-wise」：沿深度方向切分层次会触发 pipeline parallel（流水线并行），「once you start cutting up your layers um depth-wise, you have very serious issues in parallelization. Pipeline parallel, which is what this is called, is something that like most people really really do not want to deal with」；而宽度方向切分对应 tensor parallel（张量并行），「If you have a really wide model, you know, you can cut that up very easily in your GPUs. Uh tensor parallel is what it's called is much much simpler to deal with」。于是「in some sense, you know, there's systems reasons to go wide, and maybe there's expressiveness reasons to go deep, and you end up at roughly a hundred」。这一点与本课反复出现的主题一致：架构决定权往往在系统侧，而不是表达力侧；前面 RMSNorm 去掉均值中心化、并行层最终失宠，都是同一逻辑的不同侧面。

证据来自两组图。kf_0038 的「Evidence on aspect ratio scaling」左图（出处 [Kaplan et al 2020]）有三条曲线，图例分别为 50M Params、274M Params、1.5B Params，横轴是 Aspect Ratio $(d_{model}/n_{layer})$ 的对数刻度 $10^1$、$10^2$、$10^3$，中间用两条竖线框出一大段区间并写着「A wide range of architectures achieve similar performance」；讲者解读为「regardless of kind of the size of your model, roughly speaking, the optimum aspect ratio is fairly similar, and they live at about a hundred, maybe a little bit less depending on how you want to do the accounting, but really, you know, anywhere near a hundred is a pretty safe bet for aspect ratios」。右侧四个小图出自 [Tay et al 2021]，横轴都是 FLOPS，纵轴分别是 Negative Log-Perplexity 与 SuperGLUE Accuracy，图例按 Params 规模分组，散点标记如 Base、DH12、DH14、NL4、NL6、NL8、SH12 等代表不同深度-宽度配置；讲者的结论相当锋利：「as you sort of sweep the depth-to-width tradeoffs, you find that really, um the only thing that matters in some sense is FLOPs. As you increase the FLOPs, the models get better, and that's really controlling the majority of the effects, not necessarily uh the aspect ratio」。由此「what has really emerged from this is the sense that there's a general forgiving band of hyperparameters that people tend to choose, and then you really worry about primarily your systems utilization rather than sort of expressiveness concerns, which are hard to reason about」。易错点也在这里：不要把宽深比当成可以精细调参的性能旋钮——它先是系统并行方式的约束，其次才是模型形状；而且「there are a lot of hyperparameters that seem quite important, but they're also fairly forgiving, and people have converged roughly on the minimum」，Kaplan 图里不同规模曲线的最低点只是「similar」，并非重合，讲者自己也加了「depending on how you want to do the accounting」的保留。

![宽深比示例表：各模型的 d_model / n_layer 与 Sweet spot 区间](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0037_00-52-16.jpg)

---

## 词表大小：单语 30–50k 与多语 100–250k，以及 bits-per-byte 问答 [55:09 - 59:21]

最后一个超参数是 vocabulary size（词表大小，即 tokenizer 的 token 总数）。讲者说这里「there's a really clear difference between two classes of models」：早期开源、只追求英文的 monolingual（单语）模型「had these like much smaller vocab size, in the 30,000 range」；而 post-LLaMA 之后大家转向 multilingual（多语）或生产系统，包括 GPT-4 这类闭源模型，「all these have much much larger vocab sizes, and these are roughly in the hundred to 200,000 um vocab range」。kf_0040 的标题是「What are typical vocabulary sizes?」，左半张表标着「Monolingual models – 30-50k vocab」：Original transformer 37000、GPT 40257、GPT2/3 50257、T5/T5v1.1 32128、LLaMA 32000；右半张表标着「Multilingual / production systems 100-250k」：mT5 250000、PaLM 256000、GPT4 100276、Gemma 4、DeepSeek 100000、Qwen 15B 152064、Yi 64000。页面底部一句话点题：「Monolingual vocabs don't need to be huge, but multilingual ones do」。讲者补充两条成因：多语模型「really do need much larger vocab to cover the whole space」；同时「the models on the right are also bigger. There have been scaling law studies showing that the bigger your model, the larger the vocab it can handle」，所以词表膨胀也部分由现代缩放趋势驱动，「No one's training ... large monolingual models uh anymore」。

这里有几个容易读错的地方。一是右表并非全部落在 100–250k：Yi 的 64000 明显低于该区间下沿，说明「单语 / 多语」是两类集群而不是硬阈值，分类依据是用途而非数值。二是讲者特别点名「Google models have a ton more vocab」，并说「LLaMA derivatives roughly range at about a hundred uh thousand tokens」，所以同样是「大模型」，Google 系（mT5 250000、PaLM 256000）与 LLaMA 系（约 100k）并不在同一量级。三是词表大小与 $d_{model}$、参数量、多语需求耦合，不能像调学习率那样独立改动；它与前几章的 FFN 倍率、头维度、宽深比一样，属于「横向比较出来的共识」，而非可推导的最优值。紧接其后的问答把这一点推到 multimodal（多模态）：学生问的是多模态，讲者先听成 multilingual 再自我纠正「Multimodal. Multimodal. Yeah, so I guess it depends on the way that your tokens are encoded, but, you know, if you're tokenizing your images and things like that, then you need to, you know, have many more tokens to account for those」，并说开源模型往往「a different image tokenizer with its own vocab, which is quite large」——即图像有自己独立且很大的词表，不必然并入文本词表。讲者还补了一句「Generally, the models on the right are also bigger」，并说词表差异「is also partially driven by uh modern scaling trends」，即词表膨胀与模型规模扩张是同步发生的，不能把两者当成彼此独立的因果；同理，多模态问答里他先纠正「Multimodal. Multimodal. Yeah」的口误也提醒读者，把 multilingual（多语）与 multimodal（多模态）混为一谈是这一节最常见的听错点。

57:11 之后是一段关于 bits per byte（BPD，每字节比特数）能否跨 tokenizer 比较的问答，讲者先「step back a moment」把问题放回生成建模的框架：「language modeling is is a generative modeling task, right? We are modeling the probability of a sequence. Now, as long as your sequence is fixed, right? It's the same ... you provide a probability over all strings, that's always valid to compare」。把这句话写成公式，就是对序列赋概率并逐 token 分解、再用字节数归一化：

$$P(x_1,\ldots,x_T)=\prod_{t=1}^{T}P(x_t\mid x_{<t}),\qquad \text{BPD}=-\frac{1}{N_{\text{bytes}}}\log_2 P(x_1,\ldots,x_T)$$

其中 $N_{\text{bytes}}$ 是原始字节数。讲者说比较是否有效取决于两件事：其一，「did you touch the sequence at all? Like, if you look at some tokenizers in the past, before subword tokenizers, they would drop some tokens or drop some words. That changes that makes the comparisons invalid. But modern tokenizers are complete. They can model any sequence, so that's not a concern」；其二，「are we like length normalizing it in any way」，而「for bits per byte, you're always normalizing with the same number, which is the number of bytes, and so this is always a valid comparison」。他还提到 perplexity（困惑度）与 BPD「kind of dual to each other」。局限与易错点：这个结论的前提是 tokenizer 完备、不丢信息，且用同一个字节数归一化；学生随后追问「changing it different splits」下的 perplexity 比较，讲者坦言「we'll have to we'll have to talk later cuz I'm not sure I understand the question, but I think that that's an interesting set of questions」，因此不同数据切分上的困惑度能否直接比较，本段并未给出结论，不应替讲者补全。kf_0041 的白板上留有红色手写推导（可见 $t_0$、$t_f$、$h(t)$、$g_x(x,t)$、$\frac{d}{dt}$ 等符号与积分式），但那是板书残留、与本段结论无直接对应，转写也未念到，故不据此推演。

![典型词表大小两张表：单语 30-50k 与多语/生产系统 100-250k](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0040_00-56-27.jpg)

---

## 正则化的反直觉结论：weight decay 更像优化器干预 [59:21 - 63:26]

讲者把 dropout（随机失活）与 weight decay（权重衰减）等 regularization（正则化）称作「another very interesting class of ideas. Also one that I think is very counterintuitive from your machine learning 101 intuition」。kf_0043 的标题是「Dropout and other regularization」，中央问句「Do we need regularization during pretraining?」，下面列出「Arguments against:」两条：第一，「There is *a lot* of data (trillions of tokens), more than parameters.」（数据以万亿 token 计，远多于参数量）；第二，「SGD only does a single pass on a corpus (hard to memorize)」（SGD 对语料只做单遍，难以记忆）；页脚一句「This is all quite reasonable.. but what do people do in practice?」直接把矛盾摆出来。口头展开更完整：「I have more data than I can process most of the time ... there is more internet data than there is flops. So, you know, I'm probably not even going to see the same data twice」，因此「a single pass of SGD or other optimizers is never really going to memorize my data very much」，结论是「overfitting is not really a problem uh almost ever during compute constrained language modeling」；甚至「some people even actually only look at training loss because they believe so strongly that overfitting doesn't happen in single pass SGD」。用一句话形式化：在 compute-constrained（算力受限）单遍训练下，训练集与验证集的经验风险几乎重合，$\mathcal{L}_{train}\approx\mathcal{L}_{val}$，于是正则化的经典动机（缩小二者差距）不再成立。

现实却相反。讲者先抱怨「a lot of recent models don't talk about this stuff at all. Um it's really lower-level details than like tech reports are willing to expose」，但「if you look, actually you find a lot of models um do both. Especially weight decay actually is a fairly popular intervention even for modern high-performance language models」，并强调「this is very, very surprising ... And this is very mystifying. Like, why is this?」；dropout 则「maybe uh have gone out of favor」，而「weight decay actually remains fairly popular」。把 weight decay 写成带 shrinkage（向零收缩）的更新式：

$$\theta_{t+1} = (1-\eta\lambda)\,\theta_t - \eta\,\nabla_\theta \mathcal{L}(\theta_t)$$

其中 $\lambda$ 是衰减强度，$\eta$ 是学习率。讲者在后面的问答里把它口语化为「weight decay, you know, which is shrinkage to zero」。关键洞见是：有论文「argued um and shown nice evidence that weight decay is actually not a regularizer sometimes. It actually interacts with the optimizer to essentially um make optimization better」，也就是「weight decay is actually an optimization intervention and not necessarily a regularization intervention」。这正是全课「架构与训练技巧无法先验推理」的又一个实例：「this is one of the reasons why I think deep learning is hard and this architecture lecture is very strange and hard. Um it's because these things interact in very strange ways」。

kf_0044 的标题是「Why weight decay LLMs?」，副标题「[Andriushchenko et al 2023] has interesting observations about LLM weight decay」，三张图构成证据链。左图是 Validation loss（纵轴约 3.3–3.8）对 Training loss（横轴约 3.3–3.8）的散点，图例为 $\lambda_{wd}=0.0$、$0.1$、$0.2$，各色点几乎全部落在 $x=y$ 线上，下方标注「It's not to control overfitting」——讲者对应地说「you don't really see any difference. Like, weight decay isn't shifting things so the validation loss is better. There's already no overfitting. We're on the x equals y line here, right? So, doesn't control overfitting」。中图标题「10 x cosine LR decay」，横轴 Iteration 0–60000、纵轴 Training loss 约 3.1–3.8，右图标题「Constant LR」，横轴 Iteration 0–60000、纵轴 Training loss 约 3.2–3.7，两图各有多条不同 $\lambda_{wd}$ 与学习率组合的曲线；页面底部结论是「Weight decay interacts with learning rates (cosine schedule) (duration)」。讲者解读：「the stronger weight decay runs, these blue dash lines on the bottom ... do significantly better because they start out slow, but they essentially end up um converging to a much better minimum later」，并且「this is, you know, generally true when we decay learning rate, not necessarily true when we're in constant learning rate, which is maybe somewhat more of where your intuition is coming from」。

易错点有三层。第一，不能因为「训练/验证损失无差异」就判定 weight decay 无用——它的收益出现在与 learning rate decay（学习率衰减，尤其 cosine schedule）配合时的后期收敛，而不是泛化间隙。第二，结论有适用边界：它依赖「单遍 SGD」与「学习率会衰减」两个前提，constant learning rate 下不成立，也不能推广到会重复多遍数据、确实过拟合的训练设置。第三，讲者借题发挥强调方法论：「it's very difficult to reason sort of a priori or like from scratch, you know, the behavior of all these different choices and why, you know, I think Percy and I have designed this class so that you interact with stuff ... always keep that in mind that these kinds of unexpected effects can really start to kick in uh for these kinds of uh settings」。这与前面 RMSNorm 去偏置、FFN 倍率平坦盆地等「只能靠集体实验」的结论一脉相承，也为下一章把全部超参数收束成共识清单、再转向稳定性做了铺垫。补充一个容易被跳过的观察：讲者说的是「a lot of models um do both」，即 dropout 与 weight decay 曾一度同时使用，而「it's really lower-level details than like tech reports are willing to expose」——技术报告往往对这类配置只字不提，因此直接从公开报告里统计正则使用率会系统性低估真实情况；这也反过来解释了为什么他强调「you interact with stuff」式的动手实验，而不是读报告，才是弄清这类交互的可靠途径。

![Why weight decay LLMs：训练/验证损失散点与学习率衰减交互三图](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0044_01-02-02.jpg)

---

## 超参数共识清单与两组问答 [63:26 - 65:01]

讲者用「to put everything together for hyperparameters」把前面四项收束成一句标准配方：「You know, factor of four rule of thumb, keep your head dim and your number of heads uh equal to the model dimension, um pick an aspect ratio roughly around 100, um and, you know, if you ask about regularization, right? You want to maybe try a couple things cuz regularization actually does interact with optimizers in ways that are quite counterintuitive」，并补充「this is the thing that some people uh still do even though you you don't need the regularization at all」。kf_0045 的「Summary: hyperparameters」把它写成四个小标题：Feedforward —「Factor-of-4 rule of thumb (8/3 for GLUs) is standard (with some evidence)」；Head dim —「Head dim*Num head = D model is standard – but low to no validation」；Aspect ratio —「Wide range of 'good' values (100-200). Systems concerns dictate the value」；Regularization —「You still 'regularize' LMs but its effects are primarily on optimization dynamics」。把前三条写成公式，就是这一讲反复出现的三行：

$$d_{ff}=\tfrac{8}{3}d_{model}\ (\text{GLU})\ \text{或}\ 4d_{model},\qquad H\,d_{head}=d_{model},\qquad \frac{d_{model}}{N_{layer}}\in[100,200]$$

注意总结幻灯片自己留了余地：「with some evidence」（四倍法则只有部分证据）、「but low to no validation」（头维度几乎没有验证）。这与头维度表格、Kaplan 盆地、Tay et al 的 FLOPs 结论一致：这些不是被证明的最优值，而是「几乎所有人都这么取、且取错代价很小」的默认值；讲者开宗明义说「for you know, a lot of the the maybe more hairy-looking hyperparameters, actually just fairly standard choices that have worked well for everybody」。

这张总结画面的右侧还并排附了一张按年份排列的大表，表头可见 Name、Year、FF MLP factor 与 Aspect ratio $(d_{model}/n_{layer})$，右侧另有若干取值为 0.1、0、0.01 量级的窄列。表中可辨认的行包括 Original transformer（2017）、GPT（2018，4 / 64）、GPT2（2019，4 / 64）、T5 与 T5 v1.1（2019–2020）、GPT3 (175B)（2020，4 / 128）、LaMDA（2021，8 / 128）、BLOOM（2021，8 / 205）、PaLM (540B)（2022，4 / 156）、Chinchilla（2022，4 / 102）、Baichuan 2（2023，2.66 / 128）、Mistral (7B) / LLaMA (7B) / LLaMA2 (7B)（2023，3.5 / 102）、Qwen 2（2024，8 / 100）、Gemma 3（2025，8 / 87）、SmolLM2 (1.7B)（2025，4 / 86）。这张表横跨 2017 到 2025 数十个模型，把「共识」从一句口号变成可核对的数据：FF MLP factor 基本落在 2.5、2.66、3.5、4、8 这几档，Aspect ratio 基本落在 86–205。它与前面宽深比表、头维度表互为印证，也是本讲「从大量已发布模型的共同选择中提炼经验规律」这一方法论最直观的落点。

随后两组问答都指向「边界之外」。第一组关于 diffusion（扩散模型）：学生问未来模型是否有显著差异，讲者答「Ooh, diffusions. That I have not looked into enough, to be honest. Um there aren't that many people training big diffusions is one issue. Um and many of the models that have been trained are retrofitted cuz I think the architectures are actually the same as the, you know, like a Lama-like model」，并明确说若问「what's the optimal architecture if you were to train from scratch, I don't know what that is actually off the top of my head」。第二组追问「Do you have any explanation for why regularization works in some cases?」，讲者澄清「it's not that regularization in general affects optimization. I don't think people do dropout anymore because, you know, it doesn't really uh interact well with optimization. But for example, weight decay, you know, which is shrinkage to zero, um that might allow you to use a higher learning rate or it might allow you to decay faster. There are lots of ways in which all these terms are interrelated」。易错点在于：不要把「共识清单」当成放之四海皆准的配置表——讲者明说正则项「you want to maybe try a couple things」，扩散模型的最优架构他也不知道；也不要把 dropout 与 weight decay 混为一谈，前者已基本退出，后者仍流行且靠与学习率的交互起作用。这一章在结构上承上启下：把前四章的四类超参数收拢完毕，紧接着讲者用「Now, I've talked a lot about how to design um expressive models」转向一个优先级更高的议题——stability（训练稳定性）。

还有一处值得对照的细节：总结幻灯片写的是 $8/3$（即 $2.666\ldots$），而前面小结幻灯片写 $2.66$、讲者口头说「2.67-ish」，三者其实是同一个 $2/3$ 修正的不同精度写法，不要误读成三个不同的取值；同理，头维度的「Head dim*Num head = D model」就是前面 $H \cdot d_{head} = d_{model}$ 的文字版，宽深比的 (100-200) 与表格里 86–205 的范围也只是口径宽窄之别，而非相互矛盾。把这三处口径统一之后，这份清单才真正可读：它给出的不是一组待优化的超参数，而是一组「照抄即可、代价极小」的默认值。讲者反复强调的正是这一点——「a lot of the the maybe more hairy-looking hyperparameters, actually just fairly standard choices that have worked well for everybody」，与其把精力花在微调这些数字上，不如把精力放在系统利用率与训练稳定性上，这也正是下一章要展开的方向。另一个容易被忽略的提醒是「you want to maybe try a couple things」：清单里唯一没有被写成确定数值的恰恰是正则化，因为它「does interact with optimizers in ways that are quite counterintuitive」，只能靠实验而非照抄。

![超参数总结幻灯片：Feedforward、Head dim、Aspect ratio、Regularization 四条共识](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0045_01-03-26.jpg)

---

## 训练稳定性：从两处 softmax 找起 [65:01 - 66:55]

讲者把话题从「表达力」切换到「稳定性」：「over the last few years um a really big emphasis has not been on performance alone. It has actually been on stability. And this becomes an increasingly important concern as your models get more and more expensive to train」。理由很现实：前面看到这些超参数「are forgiving」，「Everyone's kind of doing similar stuff. And so, you know, you can mess with these, but you're not going to get a big performance difference. That's fine」；但一旦训练中途出问题——「your model, you know, suddenly blows up some part into training, like you get these like horrible-looking spikes all over the place」——「you might end up with a model that is, you know, actually not very good quality, right? Or it might be unrecoverable. You might have spent, you know, millions of dollars in training, and, you know, you get to a point where the model is no longer able to be trained any further」。他指着蓝色曲线说「you don't want to train models that look kind of like this blue curve with like spikes everywhere and these, you know, big gradient norms happening. Um so, what do we do to fix these stability issues? I mean, this is really, I would say, like a core core issue」。

排查从「usual suspects」（惯犯）开始，首要嫌疑是 softmax（归一化指数函数）。kf_0047 的标题是「Where do the issues arise? Beware of softmaxes!」，副标题直接点名两个坏性质：「Softmaxes – can be ill-behaved due to exponentials / division by zero」。写成公式：

$$\text{softmax}(u)_i=\frac{e^{u_i}}{\sum_j e^{u_j}},\qquad \text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

讲者逐一解释：「One of them is an exponential, right? We can see how that blows up very quickly. Um you also divide two numbers, and that's also a potentially very dangerous operation, right? So, a soft max is one place where you got to be extra, extra careful」。指数 $e^{u_i}$ 让 logit 的微小差异被放大成数量级差异，分母 $\sum_j e^{u_j}$ 的除法则可能溢出或除出极值，二者叠加正是梯度尖峰与损失 spike 的高发来源；这与前面「架构选择的三重约束里训练稳定性最不可妥协」的判断直接呼应。

语言模型里恰好有两处 softmax，讲者明确点出：「there's one on the output side when we output our probability distribution, and then in attention when we normalize the attention, there's going to be another soft max, right? So, we can think of both of those as really...」。kf_0047 右侧两张结构图把位置画得很清楚：左图自下而上是 Inputs → Token Embeddings → Transformer Block → Norm → Linear (Transform) → Softmax → Output Probabilities，说明输出端那处 softmax 位于最后的线性变换之后；右图标出 Output layer with shape (batch, seq, len, vocab_size) 与 Input layer with shape (batch, seq, len, d_model)，框内依次有 RMS/LN、Position-Wise Feed-Forward、Add、Norm、Causal Multi-Head Self-Attention (×N)、Add、Norm，说明另一处 softmax 藏在注意力块的归一化里。这与本课更早「遇到稳定性问题就往里塞一个 LayerNorm」的经验、以及 RMSNorm 放在残差流之外的讨论形成呼应：结构图里那些 Norm 正是历次稳定性干预留下的痕迹。

本段的边界与易错点必须说清。讲者只完成到「定位问题」这一步，转写在 66:54 停在「So, we can think of both of those as really...」，并没有在本段念出任何具体修复手段，因此本段不应替他补写后续才出现的干预技巧。另一个易错点是把稳定性当成性能问题来调：前面四章的超参数都落在平坦盆地里，怎么调都只带来微小差异，而稳定性失败「might be unrecoverable」、代价是「millions of dollars in training」，二者根本不在一个量级——这也解释了为什么讲者在把超参数共识讲完之后，要把剩下的篇幅全部让给稳定性。最后的局限是：softmax 只是「a few, you know, usual suspects」之一，是排查起点而非完备清单；讲者说「if you have stability issues in language models or in general neural networks, there's a few, you know, usual suspects that you've got to start looking at」，本段并未穷举所有原因，也没有给出量化阈值。还有一点值得注意：讲者说的是「if you have stability issues in language models or in general neural networks」，把范围从语言模型推广到一般神经网络，说明这两处 softmax 是通用的数值风险点、并非 Transformer 独有；而他把后果描述为「That would be a horrible thing to happen if you have a lot of compute that you want to spend」，也再次点明稳定性问题的严重性来自算力成本而非模型质量本身。因此本段的落点不是「怎么修」，而是「先知道从哪里查起、以及为什么值得花力气预防」。

![Beware of softmaxes：输出概率与注意力结构中的两处 softmax](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0047_01-06-13.jpg)

## 输出 softmax 的稳定性与 z-loss trick [66:55 - 69:27]

本段一开场，讲者把话题收束到模型里的"危险区"（danger zones），并特别点名 especially the attention。所谓危险区，指的是同时含有指数运算与除法归一化的 softmax：指数会把 logit 的微小差异放大成巨大的数值，归一化又可能除到极小或极大的分母，两者叠加极易让损失或梯度失控。语言模型里恰好有两处 softmax，一处是输出端对词表 $|V|$ 个 logit 做的 output softmax，另一处是注意力内部对成对分数做的 attention softmax。本段先处理前者，因为它的输出直接决定交叉熵损失：训练真正需要的是对数概率 $\log P$，而 $\log P$ 是否稳定，几乎完全取决于那个归一化项。

幻灯片 "Output softmax stability – the 'z-loss'" 先把 softmax 计算重新写了一遍：

$$\log P(x) = \log\left(\frac{e^{U_r(x)}}{Z(x)}\right) = U_r(x) - \log Z(x),\qquad Z(x) = \sum_{r'=1}^{|V|} e^{U_{r'}(x)}$$

$U_r(x)$ 是残差流（residual stream）把各层输出逐次累加之后、经输出投影得到的 logit，讲者说它 "well-behaved"：只要模型本身没有崩，这一项是各项相加的结果，尺度相对可控。真正的麻烦全在 $\log Z$。$Z$ 是 $|V|$ 个指数项求和，而 $|V|$ 动辄几万到几十万；只要 logits 整体稍微偏大，$Z$ 就会指数级 blow up，$\log Z$ 变成很大的正值把 $\log P$ 拖成极小值；反过来若 logits 整体偏小使 $Z$ 趋近 0，$\log Z$ 又变成很大的负值。讲者明确说 "both of those directions are very, very bad"，理想状态是让 $Z$ 落在 1 附近，也就是 $\log Z$ 落在 0 附近。手写白板那一帧（67:43）同样把 $Z(x)$ 写成对 $|V|$ 个 $e^{U_{<i}(x)}$ 求和，并在旁边写下损失式，说明这不是口头带过而是本段的正式推导。

关键在于 softmax 是过参数化（overparameterized）的。若给所有 $U_r$ 同时加同一个常数 $c$，则 $e^{U_r+c}=e^c e^{U_r}$，分子与分母中的 $e^c$ 完全抵消，输出概率 $P(x)$ 一分不变，但 $Z$ 被乘上 $e^c$、$\log Z$ 平移了 $c$。也就是说，"把 $\log Z$ 挪到 0" 这个操作并不改变模型所表达的分布，只是选了一个数值上更好的等价参数化。Devlin 2014 据此在主损失外挂一个平方正则项，幻灯片印作：

$$L = \sum_i \left[\log P(x_i) - \alpha\big(\log Z(x_i) - 0\big)^2\right] = \sum_i \left[\log P(x_i) - \alpha\log^2 Z(x_i)\right]$$

并在右下角标注 [From Devlin 2014]。写成损失最小化的语言，等价于在 $-\sum_i\log P(x_i)$ 之外再加 $\alpha\sum_i(\log Z(x_i))^2$；它只惩罚归一化项偏离 0 的幅度，把 $\log Z$ 钉在零附近，前面那个 $\log P = U - \log Z$ 的表达式就整体数值稳定了。这就是 z-loss，讲者称其为 "the Z loss trick"。手写版旁边还标了一句感叹 "Three lines of code!"，可见实现代价极低。

系数取多小？幻灯片引用了 PaLM 的原话："We additionally use an auxiliary loss of z_loss = 10^{-4} · log² Z to encourage the softmax normalizer log(Z) to be close to 0, which we found increases the stability of training." 即 $\alpha = 10^{-4}$，几乎只是轻轻拉一下，却足以把归一化项约束住。幻灯片下方列出采用它的模型清单：Baichuan 2 (2023)、DCLM (2024)、OLMo 2 (2025)、OLMo 3 (2025)；讲者口述又补充，Jacob Devlin 等人在 2014 年首创，Baichuan 大概是第一个吃螃蟹的开源模型，之后 DCLM、Olmo 等跟进，并给出一句评价 "this is a surprisingly effective thing"。从整讲结构看，这是"训练稳定性"这一约束下的第一件武器，紧接在前面关于架构共识与超参数的讨论之后；它也为下一章转向 attention softmax 留出了明确接口。

几个容易踩的坑值得单独记下。第一，z-loss 管的是输出 softmax 的 $\log Z$，对注意力内部那个 softmax 毫无作用，讲者正是在 69:27 用 "let's say we've handled the instability issues on the output softmax" 把话题切走的；把两处 softmax 混为一谈，会误以为加了 z-loss 就万事大吉。第二，$\alpha$ 不是越大越好，它毕竟是叠加在主损失上的额外项，过大会改变有效梯度、干扰语言建模目标本身。第三，它不提升模型表达力——由于过参数化，$\log Z$ 的平移本来就不改变分布，z-loss 纯粹是数值稳定手段，不能指望它改善困惑度。第四，它防的是归一化项的漂移，并不覆盖梯度尖峰、损失爆炸的全部成因，因此后面还需要针对注意力的独立手段。

![z-loss 的 softmax 推导与带 α 系数的辅助损失公式，附 PaLM 的 10^-4 系数说明及采用模型清单](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0049_01-09-18.jpg)

## QK norm：把归一化塞进 Q 与 K [69:27 - 72:04]

处理完输出 softmax，讲者把注意力转向 attention，并强调这是 "a place where lots of degeneracies happen"，人们为此发展出大量控制不稳定的技术。他给出的高层设计哲学近乎一句口诀：只要有不稳定，"if you can throw a layer norm in there somehow, it might control it"。QK norm 正是这一哲学最直接的产物。要理解它插入的位置，先回顾常规 attention 的数据流：输入先经 pre-norm 的 LayerNorm，再乘 QKV 投影矩阵，切分（Split）出 $Q$、$K$、$V$；$Q$ 与 $K$ 做矩阵乘得到成对分数，送进 softmax，再与 $V$ 做第二次矩阵乘完成加权平均，最后经输出投影并加回残差流。

幻灯片 "Attention softmax stability – the 'QK norm'" 把这条流程画成了结构图：输入端是蓝色 LN 框接绿色 QKV 框，Split 之后 $V$ 走旁路直接送到 BMM2；$Q$ 和 $K$ 则各自先穿过一个蓝色 LN 框（这两个新加的框就是 QK norm），再进 BMM1、Softmax、BMM2，末端是绿色 Proj 与残差相加；下方并列的 Feed Forward 分支是 LN → FFN（图中 FFN 内标注 Squared ReLU）→ 残差相加。图下的说明写得很直白："The query and keys are Layer (RMS) normed before going into the softmax operation." 用公式把改动写出来，就是在点积之前对 $Q$、$K$ 各做一次 RMSNorm：

$$\text{Attention} = \text{softmax}\left(\frac{\text{RMSNorm}(Q)\,\text{RMSNorm}(K)^\top}{\sqrt{d}}\right)V$$

其中 $\sqrt{d}$ 是原本就有的缩放（$d$ 为 head dim），而 RMSNorm 是额外加的一层保险。

直觉非常朴素：RMSNorm 会把每个头的向量除以其均方根尺度，于是进入 BMM1 的两个输入"roughly have the same scale"、"always going to have a scale of roughly one"，softmax 的 logits 被固定在 1 附近，不会随训练一路漂到极端值。这与输出端的 z-loss 形成漂亮的对称：z-loss 是把归一化项 $\log Z$ 拉向 0，QK norm 是把 softmax 的输入 logits 压在 1 附近，一个管分母、一个管输入，目标都是让 softmax 落在数值温和的区间。参数层面，QK norm 通常不引入新的可学习规模（除非另加 gain），几乎零成本，这也解释了为什么它能被大规模采用而不引发争议。

幻灯片底部列出的采用者相当长：DCLM、OLMo2、Gemma 2、Qwen3、OLMo 3、Gemma 4，并注明 "Originally from vision and multimodal models [Dehghani 2023, Ideftcs, Chameleon]"。讲者口述补充了来龙去脉：这招最早来自多模态（multimodal）世界，做视觉-语言模型的人先发现它，Chameleon 把它验证出来；之后一批开源语言模型意识到同样的技巧完全可以用来稳定纯语言模型的 attention，如今 "QK norm is actually a very standard intervention that most of the large models now introduce"。多次训练运行显示它 "doesn't seem to affect performance"，但确实能防止 "attention degeneracies"。讲者还顺带调侃了这条演化路线：归一化最初只出现在 pre-norm 里，后来被加到每个 block 的非线性之后，现在又被塞进 $Q$ 和 $K$ —— 这句话基本概括了人们往这套体系里不断堆叠 stabilization tricks 的方式，也回扣了本讲前面讲 RMSNorm 时"遇到稳定性问题就往里塞归一化"的经验法则。

局限同样清楚。QK norm 是"软"控制：它约束的是 softmax 的**输入**尺度，讲者用词是 "controlling the inputs to the softmax and sort of hoping that the outputs are well-behaved"，并不硬性保证输出。若某些 logit 因为异常激活、位置编码或其它原因仍然冲到极端，QK norm 无法兜底——这正是下一章 logit soft capping 要解决的问题。其次，它的作用位置在 $Q$、$K$ 上，与 z-loss 作用于输出端的 $\log Z$ 完全不同，两者可以叠加而非互相替代。最后，对 $Q$、$K$ 做归一化在几何上等价于更偏向使用方向信息而非模长信息，理论上可能影响模型靠模长表达置信度的能力；只是讲者明确说大量训练运行里没有观测到性能损失，所以这一点属于理论上的保留意见，而非实践中的障碍。

![QK norm 注意力结构图：Q 与 K 在进入 BMM1 与 softmax 之前各加一层 LN，下方并列标注 Squared ReLU 的 FFN 分支](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0050_01-10-54.jpg)

## Logit soft capping：硬约束与它的代价 [72:04 - 74:08]

讲者把这一项明确标为 "the final set of things that I'll talk about as a stability intervention"，同时坦承它 "not as popular"、"more of a Google-specific trick"，并且是比 QK norm "a much harder intervention"。三者的差别可以用一句话概括：z-loss 把输出 softmax 的归一化项拉回 0，QK norm 把 attention softmax 的输入尺度固定在 1 附近并期望输出良态，而 logit soft capping 干脆直接对进入 softmax 的 logits 动手，"cap them off so they can never be too large or too small"，是一个近乎硬性的约束。讲者在 72:29 仍把 QK norm 的结构图留在屏幕上作对照，正是为了说明"控制输入"与"夹住输入"这两种思路的分界。

把讲者的描述写成公式（幻灯片并未印出该式，此处仅作形式化）：

$$\text{logits} \leftarrow c\cdot\tanh\!\left(\frac{\text{logits}}{c}\right)$$

其中 $c$ 是 soft cap 常数。因为 $\tanh$ 的值域被夹在 $(-1,1)$，输出必然落在 $(-c, c)$ 之间；当 $|\text{logits}| \ll c$ 时 $\tanh(z)\approx z$，变换近似恒等，只有在大幅值处才饱和。这解释了讲者那句 "It's called a soft cap, of course, but a tanh, you know, is bounded at some value"：叫它"软"是因为用平滑的 $\tanh$ 而不是硬截断，但效果上它给 logits 加了一道不可逾越的上限。幻灯片与转写都说明，Gemma 的做法是取 attention 层的所有 logits，把它们 soft cap 到某个值上；至于 $c$ 的具体数值，转写里只说 "at some value"，并未给出，此处不应臆测。

实例与代价都有明确的经验依据。讲者点名 Gemma 2、Gemma 3、Gemma 4（口述为 "Gemma's two, three, and four"）都采用 logit soft cap。更关键的是 NVIDIA 团队做过的系统性比较：从一个 baseline 模型出发叠加各种稳定性干预，QK norm 在这里 "does slightly better due to the fact that you can crank up the learning rate a little bit"，也就是说它的轻微优势其实来自允许把学习率调高；而 "if you do soft capping alone, you actually end up losing performance"，存在明确的 quality degradation。原因讲者说得很直接："You can never express very confident signals in your softmax beyond a certain point." softmax 分布被夹住后再也无法逼近 one-hot，模型表达高置信度的能力被一道上限锁死，代价是实打实的质量损失。

放进整讲脉络看，这一节完成了稳定性手段的谱系：从几乎无成本、已成为事实标准的 z-loss 与 QK norm，到安全但掉点、只在 Google 系流行的 soft cap，恰好覆盖"软到硬"的两端。讲者随即用 "that's kind of the end of the stability components"（73:53）收束，短暂停顿后转入 "various attention things"。这也回扣了本讲开头那个三重约束框架：泛化、GPU 效率、训练稳定性里，稳定性这一支的解决方案并非越多越好，而是要在"稳"与"表达力"之间权衡；soft cap 是稳定性压过表达力的极端样本。

易错点集中在作用点与性价比上。第一，soft cap 作用于 logits，也就是 softmax 的**输入**，不是 attention 的输出，也不是残差流；讲者自己都一度口误说成 "stabilizing the outputs of your attention"，马上更正为 "the inputs to your attention, the logits that go into the softmax"，可见这个位置极易记反，也容易和 z-loss、QK norm 的作用点混淆。第二，不要把它当成免费午餐：换来的是 "a very safe way of stabilizing" 的确定性，付出的是表达力上限，单独使用还会掉点。第三，$c$ 的取值、Gemma 各代是否一致、与 QK norm 叠加后是否仍掉点，转写与画面都没有给出，不能凭印象补写。第四，"not as popular" 是讲者的定位，意味着它更像特定组织的偏好而非业界共识，选型时不应因为"更安全"就默认替换掉 QK norm。

## 从训练到部署：算术强度与 KV cache [74:08 - 79:32]

讲者先划定范围：今天只谈稠密全对全（dense all-by-all）attention 上能做的干预，状态空间模型（state space models）与线性时间注意力 "sadly today is not the day for you"。幻灯片 "Attention heads" 用一句话定调："Most models don't touch the attention heads much at all with a few minor exceptions.."，随后列出少数例外：GQA / MQA 通过减少头数省推理成本（Saving inference costs by reducing the number of heads）；稀疏或滑窗注意力（Sparse or sliding window attention，标注 GPT4/Mistral）通过限制 attention pattern 降低计算成本；以及 "Exotic SSM stuff (Jamba, Falcon 3, Qwen 3.5, etc): next lecture!"。讲者口述补充，滑窗注意力原本来自 GPT-3 那一系，如今被几乎所有要做长上下文、又不走 exotic SSM 路线的模型广泛采用。这一页相当于本段后半部分的目录，也说明接下来两章的主角是 GQA/MQA 与滑窗/混合注意力。

要理解为什么需要这些干预，得先把视角从训练切到部署（deployment）。模型训完要服务大量用户，成本来自两类彼此独立的资源：一是 flops，即实际执行的计算量；二是 memory accesses，即访存量，它同样影响系统的延迟（latency）与利用率（utilization）。讲者强调 "you want both of these things to be small"。在训练或 prefill（一次性读入用户 prompt）阶段，总算术量约为 $O(bnd^2)$，这里 $b$ 是 batch、$n$ 是序列长度、$d$ 是 hidden dim；因为是二次注意力，会出现 $d^2$ 量级。总访存则写成 $bnd + bhn^2 + d^2$ 三项，幻灯片上分别标注为 X、softmax、projection 三项的来源。幻灯片 "GQA/MQA – Reducing attention head cost" 还用彩色矩阵块画出 $XQ$ 乘 $K^\top X^\top$ 得到成对分数张量、softmax 后与 $XV$ 相乘得到 $P$（标注 mix）、输出 $\in \mathbb{R}^{n\times d}$ 的全过程，并在右侧给出符号表：$d$ = hidden dim，$b$ = batch，$n$ = length (<d)，$h$ = heads，$k$ = head dim ($d/h$)。这些符号是后面所有推导的公共语言。

由此得到训练阶段的算术强度（arithmetic intensity，即 compute/memory）：

$$\text{Arithmetic intensity} = O\left(\left(\frac{1}{k} + \frac{1}{bn}\right)^{-1}\right)$$

讲者口述把它拆成 "one over K" 加 "one over BN"，并当场自我更正了一次：他先说 "This is the number of heads"，随即改口 "sorry, head dims" —— $k$ 是 head dim 而不是头数。含义是：head dim 要足够大，参与相乘的矩阵才够"实在"；同时序列 $n$ 或 batch $b$ 要足够大，GPU 才能被喂饱。只要这两个条件成立，"your GPUs are going to be fully utilized"，训练与 prefill 阶段几乎没有效率问题。这里可以顺带看清一个常被忽略的事实：本讲前面用算术强度论证去掉 RMSNorm 的均值与线性层偏置，和此处用的是同一套推理工具，系统效率一直是贯穿全讲的主线之一。

真正的麻烦出现在自回归解码。生成必须一个 token 接一个 token，"I can't parallelize the generation process"，讲者称之为 "the curse of autoregressive language modeling"。高效做法是把过去所有步的 Key 与 Value 存进 KV cache，每步只补算新 query 与已有 keys 的交互，复用已经算过的子矩阵，从而省下大量重复计算。幻灯片 "What about the incremental case when we generate text?" 把 Step 1 拆成 "Without caching" 与 "With caching" 两行对照：$Q$ 是 $(1, \text{emb\_size})$ 的 "Query Token 1"，$K^\top$ 是 $(\text{emb\_size}, 1)$ 的 "1st token key"，相乘得到 $(1,1)$ 的 $QK^\top$；$V$ 是 $(1, \text{emb\_size})$ 的 "Value Token 1"，与 attention 权重作用后得到 $(1, \text{emb\_size})$ 的 "Token 1"；图例区分 "Values that will be masked" 与 "Values that will be taken from cache"。图下注明这是取自一篇 KV caching 讲解的动画。讲者同时点出关键差异："can't parallelize the generation process – needs to be step by step"，以及 "we need to incrementally re-compute/update attention via the 'KV cache'"。省的是重复计算，不是访存。

代价恰恰出在访存上。讲者说 KV cache 方案会 "reading and reading parameters all the time"：每生成一步都要把参数读进来、做一次点积，且每步都重复一遍。总算术量仍是 $bnd^2$（"I'm multiplying the same matrices still, just incrementally rather than all at once"），但访存模式退化成 $bn^2d + nd^2$ —— 原本投影项只是 $d^2$，现在变成 $n$ 倍的 $nd^2$，讲者直言 "the second term is not so pleasant"。幻灯片把这两项分别标为 K,V 与 projection，并给出增量情形的算术强度：

$$\text{Arithmetic intensity} = O\left(\left(\frac{n}{d} + \frac{1}{b}\right)^{-1}\right)$$

讲者口述的比值是 $\frac{N}{d}+\frac{1}{B}$，结论是 "need large batches + short seq length, or we need really big model dimensions"；换句话说，想高效服务一个小模型非常困难，而 $\frac{n}{d}$ 这一项在增量计算下 "difficult to reduce"。幻灯片末尾那句 "Is there some way around this? The n/d term is difficult to reduce." 直接把问题抛给了下一章。

这一章至少有三处极易读错。其一，幻灯片写的是 compute/memory，而讲者口述的 $\frac{n}{d}+\frac{1}{b}$ 其实是它的倒数（memory/compute），两者互为倒数；把"强度低"误读成"强度高"，整个结论就反了。其二，训练与解码的 FLOPs 完全相同，退化的是访存模式，不是计算变多；两者的优化方向甚至相反——训练希望长序列、大 batch，解码反而希望短序列、大 batch 或更宽的 $d$。其三，讲者口述把缓存内容说成 "past keys and queries"，但 KV cache 缓存的是 Key 与 Value，Query 每步只算当前 token，这一口误值得留意。正因为 $\frac{n}{d}$ 压不下去，才只能从 K/V 头数上把访存量直接砍掉，这正是 MQA 与 GQA 的动机；它与前面用算术强度论证去掉均值和偏置的思路一脉相承，都说明在这门课里系统约束常常比表达力更决定最终设计。

![训练与 prefill 阶段的注意力计算量、访存量与符号表 b/d/n/h/k，以及算术强度 1/k + 1/bn 的推导](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0053_01-15-41.jpg)

![增量解码的总算术量 bnd²、访存量 bn²d + nd² 与算术强度 (n/d + 1/b) 的倒数形式](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0055_01-19-00.jpg)

## MQA 与 GQA：用头数比例调节表达力与推理效率 [79:32 - 83:14]

既然 $\frac{n}{d}$ 项难以压缩，最直接的思路就是缩小 KV cache 本身，这引出 MQA（multi-query attention，多查询注意力）。常规多头注意力里，每个 head 都有各自不同的 $Q$、$K$、$V$；MQA 的做法是把 $K$ 和 $V$ 在所有 head 之间共享，只保留 $Q$ 各不相同。这样一来需要搬进搬出内存的对象骤减，"the KV cache, right? Are now significantly smaller"，KV 部分被所有头共享，总访存量显著下降。用上一章的符号写，若 $H$ 是 query 头数，KV cache 的访存项相当于被压到原来的 $\frac{1}{H}$，等价于算术强度里多出一个因子 $H$ —— 讲者的说法是 "we had the N over D term. Now, we have H multiplying this"，头数越多，这项增益越显著。

但幻灯片上那句 "on the right here, you have one value and one key for all these queries" 也点出了代价："You do in fact lose significant expressive power if you do this." 于是问题变成：系统效率与表达力之间，是否存在一个不必大幅牺牲后者的甜点？GQA（grouped query attention，分组查询注意力）就是答案。三者的关系可以精确表述为：原始 Transformer 的 multi-head 是每个 head 各有一组 $Q$、$K$、$V$；multi-query 是所有 head 共用唯一一组 $K$、$V$；grouped query 则保留全部 $Q$ 头不变，只把 $K$、$V$ 头数减少到某个小于总头数的值。于是出现一个可调的比值——key/value 头数相对总头数的比例——讲者称它 "kind of the number of key heads or the number of value heads while keeping the total number of heads much larger than that"，并强调它 "allows us to very simply control the trade-off between expressiveness and inference efficiency"。

幻灯片 "Additional extensions – GQA" 把这三种结构并排画成了三列，数字非常直观：三列的 Queries 都是 8 个方框；Multi-head 一列 Values 8 个、Keys 8 个，与 Queries 一一对应；Grouped-query 一列 Values 4 个、Keys 4 个，每个 $K$/$V$ 通过虚线扇出连到 2 个 Queries；Multi-query 一列 Values 只有 1 个、Keys 只有 1 个，虚线扇出连到全部 8 个 Queries。标题下的副标题把设计意图说得很清楚："Don't go all the way to one dimension of KV – have fewer dims"，即不必一路走到极端，少几维就好；下方一行总结 "Simple knob to control expressiveness (key-query ratio) and inference efficiency"，把 GQA 定位成一个连续可调的旋钮。页面末尾还预告 "More recently – MLA (multihead latent attention) from deepseek v2"；讲者口述说 DeepSeek-V2 的 MLA 采用的是另一种分解结构与另一套权衡，下一讲会稍微展开。

经验证据来自 T5 时代（讲者说 "I think T5 days if I remember right"）的性能-成本对比，幻灯片 "Does MQA hurt? Sometimes..." 把数据摊得很细。左侧表格标题为 "Table 3: Billion-Word LM Benchmark Results."，列为 Attention、$h$、$d_k,d_v$、$d_{ff}$、dev-PPL，六行分别是：multi-head / 8 / 128 / 8192 / 29.9；multi-query / 8 / 128 / 9088 / 30.2；multi-head / 1 / 128 / 9984 / 31.2；multi-head / 2 / 64 / 9984 / 31.1；multi-head / 4 / 32 / 9984 / 31.0；multi-head / 8 / 16 / 9984 / 30.9。配文 "Small PPL hit w/ MQA [Shazeer 2019]" 说明 MQA 的困惑度代价其实很小（29.9 → 30.2），但把头维度压到很小反而更差。右上散点图标题 "Low/no hit w/ GQA [Ainslie 2023]"，纵轴 Performance（约 46 到 47），横轴 Time per sample (ms)（0 到 1.5）：MQA-XXL 约在 (0.3, 46.5)，GQA-XXL 约在 (0.3, 47)，MHA-XXL 约在 (1.5, 47)，MHA-Large 约在 (0.3, 45.9) —— GQA 几乎拿到 MHA 的性能却只要 MQA 的时间。右下曲线图纵轴 Time per sample (s)、横轴 GQA groups（1、4、8、16、32、64），图例为 MHA（红色点线）、GQA（蓝色方块）、MQA（橙色点线）：MHA 恒定在约 2.4 s，MQA 恒定在约 0.6 s，而 GQA 曲线从 1 组到 16 组几乎平在 0.6 s 附近，32 组升到约 0.8 s，到 64 组才跳到约 2.4 s 与 MHA 汇合。这就是讲者说的 "if you have a small reduction in the number of heads, you basically have most of the gains in your performance"：小幅减少 K/V 头数就能拿到绝大部分收益。

放进前后文，这一节既解决了上一章留下的 $\frac{n}{d}$ 困局，也解释了 "why models today almost all adopt this GQA structure"：极低的推理成本、几乎不掉的性能，正是当前最吃紧的长上下文服务场景所需。讲者还提到推理机制细节会由 Percy 在后续课程展开，这里只给一个 flavor。易错点有四个：其一，MQA/GQA 不是推理期可切换的开关，问答里讲者明确 "MQA is ... not just inference time fixed. It's a pleasure to train. You train with a certain number of keys."，架构上必须按既定 K/V 头数从头训练；其二，幻灯片标题 "Does MQA hurt? Sometimes..." 提醒我们，Billion-Word 上 PPL 只掉 0.3 不代表所有下游任务都无损，MQA 掉点在多处被观测到；其三，为省成本而把模型整体缩小是更差的选择，讲者说 "if you make your model smaller to try to hit your performance targets, you get much worse performance"；其四，MLA 是另一种分解结构，不是 GQA 的简单变体，不能把两者的权衡混为一谈。

![GQA 三种结构对照：Multi-head、Grouped-query 与 Multi-query 的 8 个 Queries 与不同数量的 Keys/Values 方框及虚线共享关系](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0056_01-20-51.jpg)

![Does MQA hurt 的实验数据：Billion-Word 的 dev-PPL 表格、GQA 性能对时间散点图，以及 Time per sample 随 GQA groups 变化的曲线](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0057_01-22-43.jpg)

## 问答：规则与搜索、weight decay、训练期不可变的架构量 [83:14 - 85:08]

GQA 讲完后进入问答，第一个问题很尖锐：既然本讲给出了这么多关于超参数的经验规则（rules of thumb），实际训练时到底还搜不搜超参数，还是直接照抄规则？讲者的回答是 "I think it's a mix of both"，两者兼有。他解释，每一次模型训练都会带上若干"可以被改变的假设"（some theses about what can be varied），所以并不是完全免搜索；但从大量公开报告看，"the hyper-parameters are often not where people are touching too much"，超参数本身反而很少被大动，常见的是 "architecture changes like one at a time"，即一次只改一个架构组件，而 "it's very rare to like go and change everything up"。这实际上呼应了本讲开头的方法论：架构无法先验推理，可靠知识来自大量已发布模型的共性，而新知识的生产方式是一次改一个变量的受控对照，而不是全盘重来。

少数例外是 Google。讲者说 "I think Google is one of the only orgs that seems to like really spice things up in a significant way"，并点名 Gemma 系列做了不少有意思的事：最近的 Gemma 4 甚至给**每一层**单独配了 embedding（individual embedding for every layer），目的是在 memory use 与 flops 之间调节权衡。这是幻灯片与转写里都没有展开、但由讲者口述补充的一条架构细节，也和本讲前面反复出现的"内存与计算互换"主题连成一线：从 RMSNorm 去掉均值、线性层去掉偏置，到 GQA 缩小 KV cache，再到 Gemma 4 的逐层 embedding，本质上都是在同一个资源天平上挪动砝码。

第二个问题问的是训练过程中会不会动态调整这些参数。讲者的回答相当克制："Weight decay, yes. Weight decay, people change in concert with like learning rate. That is actually a heuristic that people do that works very well." 也就是只有 weight decay（权重衰减）会随学习率联动调整，而且这是一条被验证有效的启发式。把这句话写成带解耦权重衰减的 AdamW 更新式（幻灯片未印出公式，此处为形式化）：

$$\theta_{t+1} = (1-\eta_t\lambda)\,\theta_t - \eta_t\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$

其中 $\lambda$ 是 weight decay 系数，$\eta_t$ 是第 $t$ 步的学习率，$\hat m_t$、$\hat v_t$ 是一阶与二阶矩估计，$\epsilon$ 是数值稳定常数。式中 $(1-\eta_t\lambda)\theta_t$ 这一项正好体现了二者 "in concert" 的耦合：同样的 $\lambda$ 在不同学习率下产生的收缩强度并不相同，所以调学习率时必须一起重新考虑 $\lambda$。这也与本讲前半段关于正则化的反直觉结论接得上——在单遍训练、几乎不过拟合的语言建模场景里，weight decay 起作用的方式更像优化器干预而非正则化，它与学习率衰减配合让模型收敛到更好的极小值。

其余超参数为什么不能动？讲者给了很实际的理由："the architecture ones just make training incompatible. So, you can't really change them while you're training." 层数、hidden dim、头数、K/V 头数这类架构量一旦中途修改，张量形状、checkpoint 与优化器状态都会对不上，训练无法接续，因此 "The others are usually fixed"。紧接着有人追问 MQA，讲者确认 "MQA is ... not just inference time fixed. It's a pleasure to train. That's right, yeah. You train with a certain number of keys." 也就是说前两章讨论的 K/V 头数属于架构量，训练开始时就固定，不能训完再改。这一段的易错点正在于此：不要把 GQA 的分组比例想象成部署时可随时调节的旋钮，也不要把"规则很多"误解成"无需实验"——本讲前面给出的那批 rules of thumb（前馈维度倍率、头维度、宽深比、词表规模等）只是把搜索空间压到很窄，并没有取消搜索；一次改动太多还会让归因失效，这正是报告中普遍 "one at a time" 的原因。问答收束时，讲者用一句 "Okay. The last thing I'll talk about is sliding window attention" 把话题引向本段最后一个模块。

## 滑动窗口、混合注意力与全讲收尾 [85:08 - 89:13]

最后一个模块是 sliding window attention（滑动窗口注意力），讲者一上来就说它 "a really old idea"：GPT-3 论文里就写明，模型在 full attention（每个位置可以关注过去所有位置）与一种 banded matrix 风格的注意力（只能关注固定窗口内的位置）之间交替，OpenAI 早期也研究过各种可用的 attention pattern。真正的变化是 "this has become really really popular over the past year"：把大范围 full attention 与更局部的 attention 交替起来，"hits a sweet spot for how to manage long context performance while not paying too much for inference"。把讲者描述的交替结构写成掩码形式，可以写成：

$$M_{ij} = \begin{cases} 0, & j \le i \ \text{且}\ i-j < w \quad (\text{滑窗层}),\\ 0, & j \le i \quad (\text{全注意力层}),\\ -\infty, & \text{否则}. \end{cases}$$

层的选择则是一个简单的模运算：第 $\ell$ 层为 full attention 当且仅当 $\ell \bmod 4 = 0$，其余层用窗口宽度 $w$ 的滑窗。用代码表达这种"每四层一个全局层"的骨架（python）：

```python
def attn_mask(layer_id, seq_len, window, period=4):
    i = torch.arange(seq_len)[:, None]
    j = torch.arange(seq_len)[None, :]
    if layer_id % period == 0:
        # full attention：因果掩码，每个位置可看全部历史
        return j <= i
    # sliding window：因果 + 固定窗口 w 的带状掩码
    return (j <= i) & (i - j < window)
```

讲者说较早引起他注意的开源实现是 Cohere Command A："every four layers they would have a full attention that attended to everything. The three layers in between would use a sliding window attention that would only be able to look at local structure." 他特别提醒图的方向——"in this case down cuz they ordered the diagram the other way"——Command A 那张架构图把层序画反了，沿着图往下走其实是在把局部信息逐层聚合成全局信息，越靠后的局部层反而能间接访问到越全局的内容。这样就能在不上 state-space model、不做更 exotic 干预的前提下，管理超长上下文的代价。幻灯片 "Current standard trick – interleave 'full' and 'LR' attention" 印证了这一点："From Cohere Command A – Every 4th layer is a full attention"；左侧 Command A 架构小图从 Tokenizer（框内标注词表规模约 255,000、Multilingual、含 chat turns 与 tool calls 的 special tokens）、Input embeddings，一路到 Transformer Block 1 到 4、Transformer Block N 与 LM Head / Output embeddings，右侧一列竖条标出各层是 Full Attention 还是 Sliding Window Attention。右边两个 "Command A Transformer Block" 放大框进一步注明：SWA 框是 Sliding Window Self-Attention，配 Grouped-query attention 与 No positional embeddings；Full 框是 Full Self-Attention，同样配 Grouped-query attention 与 No positional embeddings；两者的 MLP 都标注 SwiGLU activation 与 No bias terms。

位置编码在这一结构里也被重新分配。讲者提到有些工作会为长程信息改掉 embedding 形式，"get rid of things like rope, so you have no position embeddings at all"，此时长程部分几乎退化成词袋（bags），而短程信息仍保留位置信息。幻灯片把它压缩成一行："Long-range info via NoPE, short-range info via RoPE + SWA." 同时底部列出 "Other models – LLaMA 4, Gemma 3, Gemma 4, OLMO 3 does SWA+Full RoPE."；讲者口述则是 "Llama 4, most recently Gemma 4, Omo 3, they all do this combination of sliding window attention and full attention, in their case using full rope instead of nope as the embedding"。两相对照可以看出，Command A 的 NoPE 路线与这批模型的 full RoPE 路线是同一交替骨架下的不同选择，不能混为一谈。还有更进一步的变体：讲者把 Qwen 3.5 放在图右侧，说它 "alternate a state-space model called a gated DeltaNet and a full attention every ... one full attention every four layers"，即保持同样的交替结构，但把廉价的滑窗层换成了状态空间模型 gated DeltaNet，"I'll explain what that is next lecture instead of a sliding window sort of local attention"。具体数字上，本段反复出现的周期都是每 4 层插 1 个 full attention、中间 3 层用廉价层，Command A 词表约 255,000，模型清单则横跨 GPT-3、GPT-4、Mistral、Cohere Command A、Jamba、Falcon 3、Qwen 3.5、Llama 4、Gemma 3、Gemma 4、OLMo 3。

讲者随即给出判断：attention 以及"如何在长上下文的成本与性能之间权衡"仍是 "a still an active area of investigation"、"the place where the most architecture work and changes are still being done"。过去一年的新主题是开源模型都在认真应对长上下文，而目前奏效的答案不是纯全局注意力，也不是纯廉价注意力，而是 "these hybrid models that aren't just global attention, aren't just cheap attention. They're some sort of mix in between"，并且 "that seems to have worked very well so far in a lot of these models"。这里的措辞很克制，是经验观察而非定论，写作时不宜把它升级成"已解决"。

片尾总结（88:37 起）把整讲收束回来："when you look across all of these models, you start to see a lot of patterns and hopefully a sense of general understanding about what things you can do and what things are good to folks." 与之并存的还有差异："we also see a lot of differences in how we handle context and how we handle position embeddings. Even tokenization, there's some differences"。收尾幻灯片 "Recap, conclusion, etc." 用一张横跨众多大模型配置的深色大表把共性与差异同时摊开，上方写 "Many aspects (arch, hparams) of transformers are in common across the big LMs"，下方一行点出 "Major differences? Position embeddings, activations, tokenization"。讲者最后把这些直觉交还给听众，希望大家 "as you go out and do your assignments and sort of mess with the leaderboard and so on" 时能用得上。这也回扣了本讲开头那句"架构设计难以先验推理、只能从集体经验中学习"：本段从输出 softmax 的 z-loss、注意力内部的 QK norm 与 logit soft cap，一路走到部署侧的算术强度、MQA/GQA 与滑窗混合注意力，展示的正是同一套方法论——先看清资源与稳定性的硬约束，再在共识与可变项之间做有依据的取舍。易错点最后再清点一遍：Command A 架构图的层序是反的，自下而上读会理解错信息聚合方向；NoPE 只作用于长程层、短程层仍带位置信息，不等于全模型无位置编码；Llama 4 / Gemma 4 / OLMo 3 用的是 full RoPE 而非 NoPE；Qwen 3.5 的廉价层是 gated DeltaNet（SSM）而非滑窗，留到下一讲；混合结构是当前效果不错的经验方案，但长上下文仍是活跃研究领域，幻灯片与转写都没有给出"最优周期"或"最优窗口宽度"的定论。

![Current standard trick 幻灯片：Command A 每 4 层一个 full attention 的交替结构、SWA 与 Full 两种 block 的配置，以及 NoPE 与 RoPE + SWA 的分工](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0059_01-26-26.jpg)

![收尾幻灯片 Recap, conclusion, etc.：横跨各大语言模型配置的对比大表，标注共性与 Position embeddings、activations、tokenization 三类主要差异](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0060_01-28-48.jpg)

## 要点回顾

### 综述视角与原始 Transformer 基线

- 架构「难以捉摸」，本讲用综述视角（survey lens）横向比较模型，区分「所有有效架构都固定」与「可自由变动」的选择；2024–2025 年一下冒出 19 个新稠密模型，MoE 明确推到第二天，本讲只看稠密架构。
- 三条约束被「烤进」架构：必须从数据学习并泛化、在 GPU 上高效训练、训练不能 blow up；这让它显得 messy、not elegant，也解释了为何把稳定性技巧放进架构课。
- 作为标尺的原始 Transformer 用正弦余弦位置编码 $PE_{(pos,2i)}=\sin(pos/10000^{2i/d_{model}})$、ReLU 前馈 $\text{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2$ 与 post-norm LayerNorm；作业 A1 却改成 pre-norm、RoPE、SwiGLU，讲者直言是「照抄 LLaMA」。

### 归一化：pre-norm、RMSNorm 与去偏置

- 唯一全行业共识是把 LayerNorm 移出残差流：$x_{l+1}=x_l+F_l(\text{LN}(x_l))$ 的雅可比 $\partial x_{l+1}/\partial x_l=I+\partial F_l/\partial x_l$ 带恒等项，梯度沿「干净的残差流」直通不衰减；BERT 与 OPT-350M 是被点名的 post-norm 例外，Grok、Gemma 2、Olmo 2 则用残差流外的 double norm 或 non-residual post norm。
- 统计归一化只占 0.17% FLOPs，却可能吞掉约 25.5% 的 runtime（% flop 99.80/0.17/0.03 对 % Runtime 61.0/25.5/13.5），瓶颈是内存搬运；RMSNorm 删去均值与偏置，写作 $y=\frac{x}{\sqrt{\frac{1}{d}\sum x_i^2+\epsilon}}\odot\gamma$，在 2 亿参数模型上每秒步数还略有提升。
- 同一逻辑推广到线性层：$\text{FFN}(x)=\sigma(xW_1)W_2$（带「if they're not gated」限定）丢掉 $b_1,b_2$，理由是「保持系统更高算术强度、表达力不变」；讲者强调丢 bias 在典型语言建模负载下才成立，且 pre-norm 流行「可能只是 LLaMA 2 的路径依赖」。

### 激活函数与门控线性单元

- 真正重要的轴是「是否门控」而非名字：门控只改前馈层前半段，$\text{FF}_{\text{ReGLU}}(x)=(\max(0,xW_1)\otimes xV)W_2$ 多出矩阵 $V$，参数量从 $2\,d_{model}d_{ff}$ 涨到 $3\,d_{model}d_{ff}$，须把 $d_{ff}$ 缩到 $2/3$ 配平，即 $3\times\frac{2}{3}=2$、倍率 $8/3\approx2.67$。
- Narang et al 2020 在 $223M$ 参数、$11.1T$ Ops 的 T5 系配平对照里，SwiGLU 拿到最低 Final loss $1.789$、ReGLU 拿到最高 SGLUE $76.17$ 与 XSum $18.36$；但增益是 small deltas，靠带误差棒的重复实验才显出一致，GPT-3 与 NeMo Tron 340B（squared ReLU）证明不用 GLU 也能训好。
- 并行块 $y=x+\text{MLP}(\text{LayerNorm}(x))+\text{Attention}(\text{LayerNorm}(x))$ 曾由 GPT-J、PaLM 采用，报告约 15% 训练加速、8B 小幅掉点、62B 无掉点并外推 540B 质量中性；但因「等效失去一半深度」近两年失宠，多数模型回到串行。

### 位置编码与 RoPE

- 注意力只是内积、对置换不变，必须显式注入位置；四类方案的注入位置不同：正弦 $\text{Embed}(x,i)=v_x+PE_{pos}$（原创始）、绝对 $\text{Embed}(x,i)=v_x+u_i$（GPT1/2/3、OPT）、相对 $e_{ij}=\frac{x_iW^Q(x_jW^K+a^K_{ij})^T}{\sqrt{d_z}}$（T5、Gopher、Chinchilla）、RoPE（GPTJ、PaLM、LLaMA 及 Most 2024+ models）。
- RoPE 的判据是 $\langle f(x,i),f(y,j)\rangle=g(x,y,i-j)$：正弦有绝对交叉项、绝对嵌入不相对、相对嵌入不是内积，只有按位置旋转满足；高维做法是把 $D$ 维切成坐标对逐对旋转，低频对捕捉长程依赖、高频对捕捉邻近关系。
- 分水岭是「乘」而非「加」：$XWR$ 的分块旋转矩阵作用于 query/key、在每次注意力计算时做（`cos, sin = self.rotary_emb(...)` 后接 `apply_rotary_pos_emb`），因而不引入交叉项、保持纯相对；Gemma 4 的 P-RoPE 只旋转前两个坐标，是极小模型上的省空间变体。

### 超参数共识与正则化

- 前馈倍率默认 $d_{ff}=4\,d_{model}$，GLU 按 2/3 修正为 $8/3\approx2.67$、LLaMA 2 借 MQA 乘 1.33 得约 3.5，T5 11B 用 $d_{ff}=65{,}536$、$d_{model}=1024$ 拉到 64 倍是异类；Kaplan 2020 在 50M 参数上的扫描显示 1–10 之间是平坦盆地，超过 10 才急剧恶化。
- 头维度惯例 $H\cdot d_{head}=d_{model}$（GPT3 为 96/128/12288、Ratio 1，T5 的 16 是 Google 系例外），宽深比 $\frac{d_{model}}{N_{layer}}\approx100$（BLOOM 205、GPT3 128、LLaMA2 102、Gemma 4 61），总结区间写作 [100,200]；Tay et al 2021 显示真正主导效果的是 FLOPs，这类参数都很宽容。
- 词表分单语 30–50k（GPT2/3 50257、LLaMA 32000）与多语/生产 100–250k（mT5 250000、PaLM 256000、GPT4 100276）两类；正则化的反直觉结论是单遍 SGD 下 $\mathcal{L}_{train}\approx\mathcal{L}_{val}$ 几乎不过拟合，weight decay 更像优化器干预，要与 cosine 学习率衰减配合才显出收益，dropout 则基本淡出。

### 训练稳定性技巧

- 不稳定多从 softmax 查起：$\text{softmax}(u)_i=\frac{e^{u_i}}{\sum_j e^{u_j}}$ 兼有指数与除法归一化，$\text{Attention}(Q,K,V)=\text{softmax}(QK^\top/\sqrt{d})V$ 与输出端概率各占一处，是梯度尖峰与损失 spike 的高发来源。
- 输出端 z-loss 利用 softmax 过参数化，把 $\log P=U_r(x)-\log Z(x)$ 的归一化项钉向 0：$L=\sum_i[\log P(x_i)-\alpha\log^2 Z(x_i)]$，PaLM 取 $\alpha=10^{-4}$，Devlin 2014 首创、Baichuan 2 与 DCLM、OLMo 2/3 跟进；它只管输出 softmax，不动注意力内部。
- QK norm 写作 $\text{softmax}\!\left(\frac{\text{RMSNorm}(Q)\text{RMSNorm}(K)^\top}{\sqrt{d}}\right)V$，把输入尺度压在 1 附近，源自多模态（Chameleon），Gemma 2、Qwen3、OLMo2 等采用且几乎不掉点；logit soft cap 用 $\text{logits}\leftarrow c\cdot\tanh(\text{logits}/c)$ 夹住输入，Gemma 2/3/4 使用，但单独用会损失表达力、掉点。

### 部署侧注意力与长上下文

- 训练/prefill 算术强度为 $O\!\left((\frac{1}{k}+\frac{1}{bn})^{-1}\right)$，解码因 KV cache 退化为 $O\!\left((\frac{n}{d}+\frac{1}{b})^{-1}\right)$：FLOPs 仍是 $bnd^2$，恶化的是访存 $bn^2d+nd^2$，$\frac{n}{d}$ 项难以压低，小模型高效服务因此困难。
- 于是从 K/V 头数砍访存：MQA 所有头共享一组 K、V，GQA 保留全部 Q 而缩小 K/V 头数（示意 8 queries 对 4 或 1 个 K/V）；Billion-Word 上 multi-head dev-PPL 29.9、multi-query 30.2 只掉 0.3，GQA 几乎拿到 MHA 性能却只要 MQA 时间，且 K/V 头数是训练期就固定的架构量。
- 长上下文靠混合注意力：每 4 层插 1 个 full attention（`attn_mask` 里 `layer_id % period == 0`）、中间 3 层用滑窗，Cohere Command A 词表约 255,000、长程用 NoPE 而短程用 RoPE+SWA；Llama 4、Gemma 4、OLMo 3 用 full RoPE，Qwen 3.5 把廉价层换成 gated DeltaNet（SSM），这一维度仍是活跃研究领域。

## 视频来源

- 原始链接：https://www.youtube.com/watch?v=lVynu4bo1rY
- 课程：Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 3

---

## 对照参考笔记补遗（Spring 2025 版）

> 编者注：本节依据 2025 版参考笔记整理，其中的「更正」小节记录的是核对时发现的问题，对应正文已在本版订正；补遗内容不代表 2026 课程视频原文。


本节全部内容来自 Spring 2025 版参考笔记，仅用于与本讲 2026 视频笔记对照补充，不代表 Spring 2026 视频原文；凡与视频画面或转写不一致之处，一律以 2026 原文为准。

### 门控有效性的定量证据与归一化对照实验

本讲正文已经把门控线性单元（gated linear unit，GLU）的结论写得相当完整：改动只发生在前馈层前半段、多出一个矩阵 $V$、需把 $d_{ff}$ 缩到 $2/3$ 做 parameter matched comparison（参数配平对照），收益是「small but consistent deltas」。参考笔记在《Do Gated Linear Units Work?》一节补上了这些定性判断背后的具体分数：Shazeer (2020) 在 **CoLA、SST-2** 等任务上逐一比对 FFN 变体，其中 $\mathrm{FFN}_{\mathrm{ReGLU}}$ 拿到所有被测变体里**最高的平均分 84.67 与最高的 accuracy 94.38**；讲者特别称赞他习惯做 error bar assessment（误差棒评估），即训练多个 replicate（重复实验）再比较，而不是只跑一次。有了这两个数字，「一致增益」才从一句印象变成可核对的量级，也更容易理解为什么正文说 Narang et al. (2020) 是「corroborating」（佐证）而非首次发现。

参考《RMSNorm – Validation》一节还给出一张与我们正文那张 223M 表同源、却取了不同行的对照表。两张表的 baseline 行完全相同：Vanilla Transformer 为 223M 参数、11.1T Ops、3.50 steps/s、Early loss $2.182\pm0.005$、Final loss 1.838、SGLUE 71.66、XSum 17.78、WebQ 23.02、WMT EnDe 26.62。我们正文那张表沿「激活函数」方向展开（GeLU、GLU、GeGLU、ReGLU、SwiGLU、SeLU 等），参考这张则沿「归一化与初始化」方向展开：RMS Norm 为 3.68 steps/s、$2.167\pm0.008$、Final loss **1.821**、SGLUE 75.45、XSum 17.94、WebQ 24.07、WMT EnDe 27.14；Rezero 为 3.51 / 1.939 / 61.69；Rezero + LayerNorm 为 3.26 / 1.858；Rezero + RMS Norm 为 3.34 / 1.875；Fixup 为 2.95 / 2.067 / 58.56。把两半拼起来，正是同一篇 Narang 2020 论文里那张「各种架构干预」主表的全貌，也印证了正文那句「faster runtime 与 lower loss 是 win-win」——RMS Norm 在每秒步数和最终损失上同时优于 Vanilla。

这里最容易读错的是把两半表的行当成同一类干预：Rezero、Fixup 属于残差缩放与初始化策略，不是激活函数；反过来我们正文表里的 SwiGLU、GeLU 也不是归一化方法。另一个易错点是数字串位：RMS Norm 行的 SGLUE 75.45 与我们正文表里 GeLU 行的 75.79、GeGLU 行的 75.96 数值接近却来自不同变体，抄录时务必带上行名。还要注意 WMT EnDe（机器翻译 BLEU）这一列只在参考这一半表里列出，我们正文的激活表止于 WebQ；两处 Final loss 的差距都在 $1.838$ 对 $1.789\sim1.821$ 这一量级，属讲者口中的 small deltas，靠带误差棒的重复实验才显出方向一致，单次训练很容易被噪声淹没。

### 头维度 1:1 惯例的反对意见与低秩瓶颈

本讲正文把 $H\cdot d_{head}=d_{model}$ 定性为「约定而非数学必然」「yet another forgiving hyperparameter」，并说「有 couple of ablations、存在围绕 1 的较宽盆地」。参考《Evidence for 1-1 Ratio?》一节补上了争议的另一面：Bhojanapalli et al. (2020) 曾明确反对这一 1:1 惯例，理由是当头数很多、又强制每头维度等于 $d_{model}/H$ 时，**每个头的维度会被压得过小，形成 low-rank bottleneck（低秩瓶颈）**，从而限制注意力头的表达力。这是理解「为什么 1:1 只是惯例」最具体的一条理论质疑。但参考紧接着给出实践回应：人们并没有在真实模型里观察到显著的低秩瓶颈，绝大多数采用 1:1 的模型表现良好。于是这条反对意见更像是「理论上应当担心、经验上不必担心」，与正文「flat basin、取错代价很小」的判断恰好互补，而不构成推翻。

参考《How Many Heads, What's the Model Dim?》一节的逐行表也值得对照抄录，其 Ratio 定义为 $d_{model}/(h\cdot d_h)$：GPT3 为 96 / 128 / 12288 / 1，T5 为 128 / 128 / 1024 / 16，T5 v1.1 为 64 / 64 / 4096 / 1，LLaMDA 为 64 / 128 / 8192 / 1，PaLM 为 48 / 258 / 18432 / 1.48，LLaMA2 为 64 / 128 / 8192 / 1。这里有一处必须标为版本差异而非改错：参考把 LLaMDA 记作 64 个头、Ratio 1（于是它不是例外），而我们 2026 画面把 LaMDA 记作 128 / 128 / 8192 / Ratio 2、列为 Google 系偏离 1 的例外；两组数值各自内部自洽（$64\times128=8192$ 与 $128\times128=16384$），只是两版表格取值不同。同理，参考表里 PaLM 的头维度原样印成 258，我们正文也已提醒「以画面数值为准、不要自行修正成 256」，两版在这点上是一致的。

易错点集中在三处。其一，Ratio 以 1 为基准，T5 的 16 来自 $d_{model}=1024$ 配上 $128\times128=16384$，是「每头维度远大于模型维度」的极端 Google 系设置，不是笔误。其二，1:1 是 community convention（社区约定），$H\cdot d_{head}$ 完全可以大于或小于 $d_{model}$；把 Bhojanapalli 的 low-rank bottleneck 当成「已被证实的设计缺陷」是过度解读，参考自己就用「we don't seem to be seeing significant low rank bottlenecks」把它挡了回去。其三，别把这张表的 Ratio 列与正文前馈倍率表、宽深比表的数字混用——三者分别是 $d_{model}/(h d_h)$、$d_{ff}/d_{model}$、$d_{model}/n_{layer}$，量纲与含义都不同。

### 归一化选型清单与逐模型超参数表

正文已把 LayerNorm 与 RMSNorm 的公式、快慢理由、FLOPs 与 runtime 的背离讲透，参考《LayerNorm vs RMSNorm》一节提供的是一张干净的选型对照清单：使用 LayerNorm（做 mean subtraction 减均值、带 bias term 平移偏置、即 re-centering 再缩放）的代表模型是 **GPT-3/2/1, OPT, GPT-J, BLOOM**；使用 RMSNorm（不减均值、不加偏置，只做纯缩放）的代表模型是 **LLaMA-family, PaLM, Chinchilla, T5**。两式可分别写作 $y=\frac{x-\mathrm{E}[x]}{\sqrt{\mathrm{Var}[x]+\epsilon}}\gamma+\beta$ 与 $y=\frac{x}{\sqrt{\frac{1}{D}\sum_i x_i^2+\epsilon}}\gamma$。把这张清单叠到正文那张横跨 2017–2025 的对比大表上，就能逐格核对「谁用 RMSNorm、谁仍是 LayerNorm」，也解释了为什么正文说「LayerNorm 表达力更强、换 RMSNorm 纯粹是系统收益」——清单里 newer models 几乎全在 RMSNorm 一侧。

参考《Exception #1 – GLU Variants》还给出一张我们正文只以档位概括的逐模型 $d_{ff}/d_{model}$ 比例表：PaLM 4、Mistral 7B 3.5、LLaMA2 70B 3.5、LLaMA 70B 2.68、Qwen 9B 3.5、Gemma 2 2.67、DeepSeek 67B 2.68、Yi 34B 2.85、T5 v1.1 2.5。参考用一句话点破其来历：「The 2.66 ratio for GLU variants ensures a similar parameter count to the 4x ratio for ReLU variants」——这正是正文 $3\,d_{model}d_{ff}'=2\,d_{model}d_{ff}\Rightarrow d_{ff}'=\frac{2}{3}d_{ff}$、以及 $4\times\frac{2}{3}=\frac{8}{3}\approx2.67$ 的文字版。参考另在《What Can We Learn from the Model-Dim Hyperparam?》里补了一句结论：T5 后续的改进版 T5 v1.1 在 GeGLU 上回落到更标准的 2.5 multiplier，因此 T5 那个 64 倍（$d_{ff}=65536,\ d_{model}=1024$）「is likely suboptimal」，属于能跑通但计算不经济的选择。

参考《Dropout and Weight Decay in Practice》还留下一张逐模型配置表，可补在正文正则化结论之后：Original Transformer 0.1 / 0、GPT3 0.1 / 0、T5 0.1 / 0、OPT 0.1 / 0.1、T5 v1.1 0 / 0.1、PaLM 0 / (variable)、LLaMA 0 / 0.1、Qwen 14B 0 / 0.1（前为 dropout、后为 weight decay），并总结「Many older models used dropout during pretraining. Newer models (except Qwen) rely only on weight decay」。这与正文「单遍 SGD 几乎不过拟合、weight decay 更像优化器干预、dropout 基本淡出」的定性判断一致，只是把它落到了具体模型上。须提醒：这张表在参考里被 MinerU 切成了零散的代码块、且有一段被错放进词表章节，数值仅供对照、不宜当成精确定论；「except Qwen」也只是 2025 版的观察，我们 2026 画面里的 Qwen 已换成另一组配置，属版本差异。

### 稳定性技巧的确切参数与首创归属

正文写 logit soft capping 时非常克制：只给出 $\text{logits}\leftarrow c\cdot\tanh(\text{logits}/c)$、说明 $\tanh$ 把值域夹在 $(-c,c)$，并明确「$c$ 的具体数值转写只说 at some value，并未给出，此处不应臆测」。这个处理忠于 2026 视频，参考《Logit Soft-Capping》一节恰好补上了空缺：该技巧出自 Bello et al. (2016)，Gemma 系把**自注意力层的 soft_cap 设为 50.0、最终层设为 30.0**；因为 $\tanh$ 有界，logits 被限制在 $(-\text{soft\_cap}, +\text{soft\_cap})$ 之间，只有在大幅值处才饱和。参考还补了一句定位：「This technique hasn't been as widely adopted as QK norm」，与正文「not as popular、more of a Google-specific trick」完全吻合。请把 50.0 / 30.0 明确理解为 2025 参考给出的数值，而非 2026 视频念出的原文。

参考在两处「首创归属」上与 2026 画面讲法不同，值得并列留存。其一，z-loss：参考《Output Softmax Stability》写「PaLM pioneered this 'z-loss' trick」，采用清单为 Baichuan 2 (2023)、DCLM (2024)、OLMo 2 (2025)；而我们 2026 画面把首创标为 Devlin 2014（幻灯片右下角 [From Devlin 2014]），PaLM 只是采用者并取 $\alpha=10^{-4}$。其二，QK norm：参考《Attention Softmax Stability》把它追溯到 vision / multimodal 社区，Dehghani (2023) 训练超大规模 vision transformer，IDefICS 与 Chameleon 先把这一技巧用在多模态组件上，之后才被 DCLM、OLMo 2、Gemma 2 等纯语言模型采用。两处都不是我们笔记写错，而是 2025 与 2026 对「谁先提出」的讲法繁简不同，补遗只作对照、不改正文。

这一节最容易混淆的是三种技巧的**作用点**：z-loss 作用于输出 softmax 的归一化项 $\log Z$（把它拉向 0），QK norm 作用于 attention softmax 的**输入** $\mathrm{RMSNorm}(Q)\,\mathrm{RMSNorm}(K)^\top/\sqrt{d}$（把尺度压在 1 附近、期望输出良态），logit soft capping 则直接夹住进入 softmax 的 logits。三者分别管「分母」「输入尺度」「输入上界」，可以叠加但绝不互相替代，更不能以为加了 z-loss 就万事大吉。另一个易错点是把 soft cap 当成免费午餐：正文已写明单独使用会有 quality degradation、因为「You can never express very confident signals in your softmax beyond a certain point」，而 50.0 / 30.0 的具体取值、Gemma 各代是否一致、与 QK norm 叠加后是否仍掉点，参考与 2026 转写都没有给出，不能凭印象补写。

### 宽深比与缩放证据的另一面：上游与下游任务

正文把宽深比 $\text{aspect ratio}=d_{model}/n_{layer}\approx100$ 归为「系统约束先于表达力」的典型：极深模型触发 pipeline parallel（流水线并行），宽度方向切分对应更易处理的 tensor parallel（张量并行），于是收敛到约 100。参考《Considerations About Aspect Ratio》一节补上了证据的规模维度与任务维度：Kaplan et al. (2020) 在 **50M、274M、1.5B** 三个参数规模上分别画出 loss increase 对 $d_{model}/n_{layer}$ 的曲线，三者最低点都落在约 100 附近且「remains relatively consistent across different model sizes」；参考把右侧四幅小图的出处记作「E.K. et al. (Google)」，而我们 2026 画面标的是 Tay et al 2021，这是出处标注上的版本差异，不影响图的结论。逐行表可一并抄录：BLOOM 205、T5 v1.1 171、PaLM (540B) 156、GPT3/OPT/Mistral/Qwen 128、LLaMA/LLaMA2/Chinchilla 102、T5 (11B) **43**、GPT2 **33**，sweet spot 写作 100–200。

更有信息量的是参考对 upstream 与 downstream 的区分。它记录 Google 那组实验的结论：对 upstream task（上游任务，如 negative log-perplexity 负对数困惑度），「parameter count is the only thing that matters」、更深的模型并不带来收益；对 downstream task（下游任务，如 SuperGLUE accuracy），结论「less clear」，但在相同 FLOPs 下更深的模型**可能**更好。这正好接住正文那句「the only thing that matters in some sense is FLOPs」：正文强调的是 FLOPs 主导上游损失，参考补的是「下游任务上深度或许仍有微弱优势、且证据不确定」。两者合起来才是完整表述——宽深比之所以是 forgiving 的超参数，是因为在很宽的区间里上游性能几乎只由算力决定，而深度的收益只在下游任务上以不确定的方式隐约出现。

易错点最后清点一遍。别把「about a hundred」当成唯一点值：参考自己就给了 100–200 的区间，表里既有 GPT2 的 33、T5 (11B) 的 43，也有 BLOOM 的 205；讲者也加了「depending on how you want to do the accounting」的保留。T5 (11B) 这一行尤其要留意：参考记作 43、GPT2 记作 33，而我们 2026 画面把 T5 (11B) 记作 33 且没有 GPT2 行，属两版表格取值不同，不是我们笔记写错。此外，Kaplan 三条曲线的最低点只是 similar（相近）而非重合，不同规模间仍有偏移；把「parameter count is the only thing that matters」推到下游任务、或据此断言「深一点一定更好」，都超出了参考与原视频给出的证据强度。
