---
title: "第 7 讲 · 序列建模与 Transformer"
description: "本讲把\"反应式策略\"（单帧观测直接映射单步动作）遇到的两个根本失效——记忆缺失与动作不平滑——重新表述为一个序列建模问题：机器人实际工作在 POMDP 中，轨迹本身就是序列，因此应当用自回归分解 直接建模整条轨迹。讲者沿…"
---

# 第 7 讲 · 序列建模与 Transformer

> **ETH Zürich 机器人学习 · L07** ｜ 讲者 Oier Mees ｜ 55 分钟 ｜ [原视频](https://www.youtube.com/watch?v=imSTfMJjp7M)

## 核心结论

本讲把"反应式策略"（单帧观测直接映射单步动作）遇到的两个根本失效——**记忆缺失**与**动作不平滑**——重新表述为一个序列建模问题：机器人实际工作在 POMDP 中，轨迹本身就是序列，因此应当用自回归分解 $p_\theta(x)=\prod_i p_\theta(x_i\mid x_{1:i-1})$ 直接建模整条轨迹。讲者沿"RNN 的三堵墙 → 注意力把跨位置路径降到 $O(1)$ → 位置编码补回顺序 → BPE 解决"什么算一个 token" → RoPE/FlashAttention 让长序列可训 → Scaling Laws 让规模可预测"这条线，重建了 Transformer 到大模型的完整工具箱；随后用同一套哲学（找到合适的 token 单位、塞进同一架构、然后放大）依次扩展到图像（ViT patch）、图文对齐（CLIP）、以及三种 VLM 融合范式（LLaVA 早期融合、Flamingo 晚期融合、原生多模态）。最后落到本讲真正的落点：把机器人动作也变成 token。逐维 256 bin 的朴素分词在 5 Hz 以上会因相邻 token 高度相关而失效（模型只需复制上一个动作），而动作分块 + 时序集成（ACT）与 DCT+量化+BPE 的压缩式分词（FAST）分别解决了平滑性与高频自回归可行性。贯穿全讲的主线是 Bitter Lesson 在机器人上的版本：找对表示、用可规模化的通用架构、让数据与算力去做剩下的工作。

![课程封面页：Robot Learning 第7讲 Sequence Modeling & Transformers，讲师 Oier Mees](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0001_00-00-00.jpg)

## 反应式策略的两大失效与序列建模动机 [00:00 - 03:41]

前几讲的主角是反应式策略：取单帧观测 $o_t$ 送入策略 $\pi_\theta(a_t\mid o_t)$ 得到动作 $a_t$。理论上没问题，但真实世界里很容易崩，幻灯片把病因归为一句话——**One Snapshot Can't Model Full State**，并列出两条：

- **Memory（记忆）**：单帧无法回答"我 10 秒前做了什么决策"、"画面里的行人正在以多快的速度移动"。自动驾驶在结冰路面上，或从一张静态图估计速度，都是同一类困难。
- **Action Smoothness（动作平滑性）**：单步预测模型每次只输出下一个时间步的动作，各步决策彼此独立，容易抖振、来回振荡，产生不平滑轨迹。

![反应式策略（Reactive Policies）的局限性示意图：单帧观测到动作的映射及其缺陷](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0002_00-00-55.jpg)

图上还有一个容易被忽略的细节：从 $a_t$ 有一条红色回环箭头指回 $o_t$，强调观测—动作是一个闭环序列，而策略却只被建模成其中一条边。

修复方向就是本讲的题目：**序列建模 + Transformer**。其理论动机是，机器人并不运行在 MDP 上，而是部分可观测的 POMDP，要真正推断状态就必须对历史 $\tau=(o_0,a_0,o_1,a_1,\dots,o_T)$ 做推理。幻灯片给出了贯穿全课程的轨迹概率公式：

$$p_\pi(\tau)=\rho(s_0)\prod_{t=0}^{T-1}\pi(a_t\mid o_t)\,P(o_{t+1}\mid o_t,a_t)$$

箭头直指其中的 $\pi(a_t\mid o_t)$ 一项，配文是"Trajectory is a sequence, but the policy is single-step and reactive"——**世界是序列的，而策略不是**，这是根本错配。于是自然的问题变成：能不能直接建模整条轨迹，而不是只建模下一个动作？

![将机器人任务视为序列建模问题：POMDP 历史推理与轨迹概率公式推导](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0003_00-02-46.jpg)

## 自回归分解：链式法则与"如何学条件分布" [03:41 - 04:59]

处理序列最一般的视角是自回归模型。它只依赖概率的链式法则：任意联合分布，不论多复杂、多高维，都能分解成一串条件分布的乘积：

$$p_\theta(x)=\prod_{i=1}^{T}p_\theta(x_i\mid x_{1:i-1})$$

也就是说，整条序列的概率 = 第一个元素的概率 × 在第一个元素条件下第二个元素的概率 × 依此类推。幻灯片在这条公式下方用红色箭头标出本讲的核心提问：**How do we learn each conditional?** 当序列很长、维度很高（高分辨率视频、复杂机器人轨迹）时，条件分布不可能靠一张大查找表来表达，必须学一个函数来参数化它。

![自回归模型幻灯片，展示联合分布的条件概率链式法则公式及学习条件分布的提问](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0004_00-03-41.jpg)

## RNN：能变长输入，但三堵墙 [04:59 - 06:26]

第一个被发明出来处理这些条件分布的工具是循环神经网络。想法很简洁：维护一个隐藏状态摘要"到目前为止看到的一切"，随新输入递推更新，因此天然支持变长输入。

$$h_t=f(h_{t-1},x_t)$$

幻灯片标题下的定位是"SOTA until 2016 in NLP, process sequences sequentially"，并把局限列为三条：

1. **长程依赖**：信息要从第 1 步走到第 100 步，梯度必须穿过中间 99 个步骤，通常爆炸或消失。左下的图解用句子"The robot picked up the spoon carefully"标注"Steps to reach first word: 6"、"To reach 'The', information flows through 6 hidden states. For length $n$: $O(n)$"。LSTM 缓解但没有彻底解决。
2. **无法并行训练**：递推结构使训练必须逐 token 串行。
3. **信息瓶颈**：不论序列多长，全部历史都被压缩进一个固定尺寸的 $h_t$ 向量，这是绕不过去的瓶颈。

幻灯片同时给出两篇经典文献：Long Short-Term Memory（Hochreiter & Schmidhuber, 1997）与 Sequence to Sequence Learning with Neural Networks（Sutskever et al., 2014）。结论是：RNN 能走一大截，但实践中难以规模化。

![RNN序列建模幻灯片，列出其局限性并展示信息传递路径图及状态更新公式](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0005_00-05-31.jpg)

## Transformer 的核心：用注意力把跨位置路径降到 O(1) [06:26 - 07:26]

Transformer 用注意力取代递推。幻灯片的关键句是"Replace recurrence with attention, giving direct $O(1)$, equal access to the entire history"。

这里需要澄清一个常被读错的点（本讲配图与机器图注都容易混淆）：$O(1)$ 与 $O(n)$ 说的是**任意两个位置之间的"跳数/路径长度"**，不是计算复杂度。对比图标题写得很明确——"Transformer vs RNN: path length to attend past words"，左侧 TRANSFORMER 面板标注"$O(1)$：Any two tokens interact in a single step, regardless of distance"、"'robot' reaches 'The' in 1 step — same cost regardless of distance"；右侧 RNN 面板则是 $O(n)$，信息要逐层穿过隐藏状态。图例区分了 query、attended (transformer)、hidden state (RNN)、compressed/folded 四种色块。注意力本身在序列长度上仍然是二次的，这一点讲者在后面讲分词时还会强调。

一次性解决 RNN 的三个问题：长程依赖变得平凡；没有递推因而训练可完全并行；每个 token 直接访问其他所有 token，因此没有固定尺寸瓶颈。

![Transformer核心思想幻灯片，通过对比图展示注意力机制相比RNN在路径长度上的优势](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0006_00-06-26.jpg)

## 缩放点积注意力：一次可微的软字典查询 [07:26 - 08:55]

给定一串 token，要为每个 token 计算一个融合序列其余部分信息的新表示；注意力要回答的问题是"**该看序列的哪些部分**"。计算流程在幻灯片上被拆成五个可点击的步骤：1—embed、2—project Q,K,V、3—raw scores、4—softmax、5—weighted sum。

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V,\qquad \mathrm{output}=\sum_i \alpha_i v_i$$

做法是：用查询向量与一组键向量做点积得到相关性，softmax 归一化成权重 $\alpha_i$，再对值向量做加权平均。除以 $\sqrt{d_k}$ 是为了防止维度 $d_k$ 较大时点积把 softmax 推入饱和区。

图解用"spoon"作为 query，给出这一行注意力权重的实际数值：The 2.6%、robot 15.8%、picked 9.6%、up 28.8%、the 4.3%、spoon 38.9%，右侧条带就是这些值向量的混合结果。

最有用的心智模型是：**这是一个软的、可微的字典查询**。普通字典只命中一个条目，注意力则同时取出所有 value，并按每个 key 与 query 的匹配程度加权混合。因为整个操作可微，模型学到的不只是表示本身，还有"最优的检索策略"是什么。

![Transformer注意力机制的加权求和步骤图解及公式](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0007_00-08-17.jpg)

## 位置编码：注意力没有顺序概念 [08:55 - 11:00]

注意力有一个微妙但致命的问题：它算的是 token 之间的点积，本质上是一个**集合操作**，对输入没有任何顺序概念。如果不做处理，"the robot picked up the spoon"和"the spoon picked up the robot"会产生完全相同的注意力模式。修复办法是把顺序信息直接注入 token 表示，即位置编码。幻灯片给出两条路线：

- **绝对位置编码（Absolute PE）**：Attention Is All You Need 的原始做法，在 token 进入 Transformer 之前，给每个嵌入加一个固定向量，即 $e+\mathrm{PE}(i)$。简单、有效。但图中用红框标出了它的失效方式：测试时若序列比训练时更长，就需要 $\mathrm{PE}(6)$、$\mathrm{PE}(7)$ 这些"never seen during training"的位置向量，直接出分布。
- **相对位置编码（Relative PE）**：不编码绝对位置，而编码 token 之间的距离——把偏置 $b(i-j)$ 加到**注意力分数**上而不是嵌入上，即 $s+b(i-j)$。图中右侧把 query="spoon" 的上下文键逐个标为 $b(-5)\dots b(-1)$、分数变为 $s+b(-5)\dots s+b(-1)$，并用绿框注明"offsets $-6$, $-7$ work fine ✓"。于是"spoon 看向 robot 相距 4 步"这件事与它们在序列中的绝对位置无关，天然泛化到未见过的长度。

现代基础模型大多建立在这种基于相对距离的扩展之上。

![绝对位置编码与相对位置编码的对比示意图](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0008_00-09-12.jpg)

## 编码器–解码器、因果掩码与 Teacher Forcing [11:00 - 13:45]

原始 Transformer 为机器翻译设计：先理解源句，再生成目标句，因此有编码器—解码器的分工。

- **编码器**用双向自注意力：每个 token 都能看到其他所有 token（过去与未来），因为完整输入是已知的，目标是得到尽可能丰富的表示。
- **解码器**用因果自注意力：每个 token 只能看自己和它之前的 token。实现方式是右侧那个三角形掩码，在 softmax **之前**把未来位置的注意力分数置为 $-\infty$，归一化后它们恰好为零。正是这一点让自回归生成成为可能：每一步只能条件于已生成的内容。
- **交叉注意力**位于解码器中部，是编码器与解码器"对话"的地方：解码器在问"根据我已经生成的内容，源句里哪些部分对现在有用？"因此**查询来自解码器，键与值来自编码器**。

训练用 teacher forcing：训练时不关心模型在某一步是否出错，而是始终把真实的前缀 token 喂进去，最大化下一个正确 token 的似然，损失为交叉熵：

$$L=-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})$$

幻灯片右侧写的是"maximize the likelihood of the next correct token $x_t$ given the true preceding tokens $x_{1:t-1}$"。左侧的可视化里，输入是右移一位的真值序列 $\langle s\rangle$, The, robot, picked, up，经过"Decoder-only transformer + causal mask"（注意：这张示意图已经是 GPT 式的纯解码器结构），输出五个一步预测概率 $\hat p(x_1)=0.72,\hat p(x_2)=0.61,\hat p(x_3)=0.85,\hat p(x_4)=0.54,\hat p(x_5)=0.68$，对应的逐 token 损失 $-\log p_\theta(x_i\mid x_{1..i-1})$ 分别是 0.329、0.494、0.163、0.616、0.386，平均 0.397 即 $L$；其中"up"那一步概率最低、损失条被标红，正是模型最需要改进的位置。

工程上的关键收益：Transformer 只需**一次前向传播**就能算出全部 $T$ 个位置的损失，而 RNN 需要 $T$ 次。

![Transformer训练中的Teacher Forcing机制与交叉熵损失函数公式及逐token损失可视化](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0009_00-12-53.jpg)

## 什么是 token：字符/词两个极端与 BPE 子词 [13:45 - 17:44]

在把这些用到机器人之前，得先回答一个更基本的问题：什么才算一个 token？语言是 Transformer 的原生地，幻灯片把两个极端摆在一起：

- **Characters**（a=0, b=1, c=2, …）：词表极小，但 token 数量极大，模型要费劲地把一切拼起来。
- **Words**（cat=0, car=1, dog=2, …）：序列短、每个 token 都有意义，但词表巨大，而且"what about new words?"——预训练词表外的词根本无法表示。

两个极端都不好用，理想的是介于二者之间、比字符大比词小的单位，既能优雅处理新词，又能把序列长度压下来——因为注意力是二次的，序列长度直接决定显存与算力。这就是**子词分词**，常用算法是 Byte-Pair Encoding（BPE），本质是自底向上的文本聚类：从数据集里所有唯一字符出发，反复找出出现频次最高的相邻 token 对并合并成一个新 token，重复 $K$ 次直到词表达到目标大小。常见子词会因为无处不在而被早早合并，从而保持序列短。另一个好处是泛化：测试时遇到全新单词，也能把它拆成一串已知的子词单元。

![语言分词方法对比：基于字符与基于单词的优缺点分析](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0010_00-13-48.jpg)

演示用的是句子"The robots melted the robot's unbearably cheesy fondue"。第一步学到的合并是 `t + h → th`（2× in corpus），token 数从 54 减到 53（−1 from start）。界面右侧有一句很容易被略过但很重要的说明：句首的 T 是大写，所以 `'t'+'h'` 只出现在两个小写"the"里——**BPE 是大小写敏感的**。下方 top pair frequencies 列出了候选对的频次条（`'r'+'o'` 3×、`'o'+'b'` 2×、`'th'+'e'` 2×、`'_'+'r'` 2×、`'e'+'_'` 2×、`'h'+'e'` 1×），标签页 0—start 到 7—result 记录了逐步合并的全过程。

![BPE分词演示界面，展示'th'合并步骤及当前53个token的状态](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0011_00-16-34.jpg)

最终汇总页给出量化结果：54 个字符 → 34 个 token，**少 37%**，压缩比 0.63×，共学到 7 条合并规则；关键 token 是"the"、"robot" 各压成 1 个 token，而"unbearably"、"fondue" 仍被拆成 4 个子词。页脚一句话点出规模差异：真实 LLM 用 50k–100k 词表、上百万条合并规则，常见词就是一个 token，罕见领域词才拆成子词。讲者补充：现代 LLM 在互联网级语料上训练，压缩率能到 3–4 倍。也就是说 **BPE 的压缩率随语料规模增长**——语料越大，被合并掉的常见模式越多，序列越短，Transformer 越高效。

![BPE压缩效果总结图，显示token从54减少到34，压缩率0.63x](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0012_00-17-29.jpg)

## 从 Transformer 到 LLM：架构简化与 RoPE、FlashAttention [17:44 - 20:22]

从原始 Transformer 走到大规模语言模型，靠的是几个工程决策。幻灯片的三条主线：

- **Decoder-only architectures → simpler to scale**：现代 LLM（GPT 系）干脆丢掉编码器，只留解码器做 next-token prediction。数据足够多时，因果解码器能学到和双向编码器一样好的上下文表示。
- **Extend BPE to work on byte-level → no unknown tokens**：在原始字节上做 BPE，基础词表覆盖一切可能输入。子词 BPE 已经能泛化到新词，字节级 BPE 还能覆盖 emoji 这类前面那套伪代码根本处理不了的输入。
- **Techniques for scaling**：
  - **RoPE（Rotary Position Embeddings）**：不再把位置向量加到嵌入上，而是按位置比例**旋转** $Q$ 与 $K$ 向量，使得 $Q\cdot K$ 只依赖 token 之间的相对距离、与绝对位置无关。
  - **FlashAttention**：标准注意力要把完整的 $n\times n$ 注意力矩阵物化到显存里，是 $O(n^2)$ 的显存开销，长序列下成为瓶颈。FlashAttention 把计算分块放进高速 SRAM，从不真正存下那个矩阵，显存降到 $O(n)$ 而结果完全一致。

页面最后一条箭头总结：**RoPE + FlashAttention 让在更长序列上训练变得可行**。

![大语言模型扩展技术幻灯片，介绍RoPE旋转位置编码和FlashAttention机制](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0013_00-18-24.jpg)

## 规模买来了什么：GPT-1/2/3 与 Bitter Lesson [20:22 - 23:26]

气泡图按参数量与 token 量画出三级跳：

| 模型 | 年份 | 参数 | 预训练 token | 出现的能力 |
|---|---|---|---|---|
| GPT-1 | 2018 | 117M | 1B | Pretraining + fine-tuning |
| GPT-2 | 2019 | 1.5B | 10B | Zero-shot generalization |
| GPT-3 | 2020 | 175B | 300B | In-context learning emerges |

GPT-1 时代仍处在"预训练 + 微调"范式：模型本身对任何具体任务都没什么用，必须在窄任务上做监督微调。GPT-2 把参数和 token 各放大 10 倍，开始出现零样本泛化——模型能做没被显式教过的任务，因为它通过文本学到了足够多的世界结构。GPT-3 则出现了上下文学习：在 prompt 里给几个例子，模型就能自行归纳出输入—输出的规律并应用到新样本上，**完全不需要梯度更新**。最惊人的是，这种能力不是谁设计出来的，而是规模的副产品。

![GPT-1/2/3模型规模与训练数据量对比示意图，展示参数从117M到175B的增长及In-context learning的涌现](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0014_00-21-10.jpg)

这正对应 Rich Sutton 2019 年的 **The Bitter Lesson**：回顾 AI 七十年历史，笑到最后的不是那些塞了最多领域知识、手工特征最精细的方法，而是那些**能随算力扩展**的方法。国际象棋、语音识别、围棋都是同一剧本——手工方案先领先一阵，随后更通用、更吃算力的方法反超。

旁边的 GPT-3 论文原图是绝佳注脚：横轴是上下文中示例的个数（K，对数刻度，并标出 Zero-shot / One-shot / Few-shot 三个区段），纵轴是 Accuracy (%)。三条曲线分别是 175B、13B、1.3B 参数；实线为"Natural Language Prompt"，虚线为"No Prompt"。1.3B 那条几乎贴着零平躺，加多少示例都没用；13B 缓慢爬升到 20% 出头；175B 则从零样本就跳到 40% 以上、随示例数增加到 60% 以上。上下文学习是规模涌现的属性。如果语言上如此，那这个领域今天想回答的问题就是：**把机器人数据也放大，会涌现什么？**

![Sutton提出的Bitter Lesson理论及GPT-3少样本学习准确率随上下文示例数量变化的曲线图](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0015_00-22-05.jpg)

## Scaling Laws：算力预算要等比花在数据与模型上 [23:26 - 27:25]

如果规模才是关键，下一个问题是怎么最优地规模。GPT 之后一段时间，业界普遍认为"把模型做大"就是主要手段。Hoffmann et al. (2022) 的 Chinchilla 论文非常精确地证明了**这并不成立**：给定固定算力预算，模型大小与数据量应当**等比例**扩展。

$$N^{\star}\propto C^{0.5},\qquad D^{\star}\propto C^{0.5},\qquad C\approx 6\times N\times D$$

经验法则是：计算最优大约需要 **每参数 20 个 token**。三张对数坐标图分别说明：左图 training loss vs compute 画出 75M、125M、250M、500M、1B、2.5B、5B、10B 各模型大小的曲线与它们的下包络（envelope）——低算力预算下最优选择其实是**小模型**：75M 模型训到它的最优点就落在包络上，而用同样算力预算训 10B 模型则仍然欠训练、悬在包络之上。中图 optimal model size $N^{\star}$ vs compute 与右图 optimal training tokens $D^{\star}$ vs compute 都画出绿色虚线的"optimal scaling frontier"，GPT-1/2/3 三个红点在中图上落在前沿**之上**（模型太大）、在右图上落在前沿**之下**（token 太少），即按最优比例看它们都是**欠训练**的。

![Hoffmann等人提出的Scaling Laws公式及训练损失、最优模型大小与计算量关系的三组对数坐标曲线图](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0016_00-23-55.jpg)

这套定律在实践里最有用的一条推论是：**可以在训练之前预测更大模型的表现**。先跑一批便宜的小规模实验（不同模型大小、不同算力预算），在 isoFLOP 曲线上拟合幂律，就能沿 $N^{\star}$ 轨迹外推，不必真的把全部预算押在一次昂贵的大跑上。左图给出 $C=10^{19},10^{21},10^{23}$ 三条 isoFLOP 曲线（浅色的更大预算区域标注"never trained"），幂律拟合点标在 $N^{\star}=42\text{B}$ 附近，紫色外推线指向一个框："Prediction at $C=10^{25}$: $N^{\star}=282\text{B}$ params, loss $=1.85$"。

右图 tokens vs model size 则把"计算最优"与"推理最优"的张力画得很清楚。绿色虚线前沿是 $D^{\star}=20\times N$；GPT-1/2/3 在前沿下方，其中 GPT-3 的标注是"undertrained（300B tokens, needs ~3.5T）"，红色箭头直指上方；Chinchilla、LLaMA 1 (65B)、LLaMA 2 (70B) 落在前沿上，属 compute-optimal；而 LLaMA 3 (70B) 用紫色箭头**远远高出**前沿——它训练的 token 数远超计算最优公式的建议，图注写明"inference-optimal: deliberately over-trained"。原因很简单：计算最优只关心训练成本，而一旦模型要服务百万用户，你更想要一个**推理时更便宜的小模型**，哪怕为此把它稍微"练过头"。

![缩放定律外推性能图，展示isoFLOP曲线与tokens vs model size关系](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0017_00-25-46.jpg)

## 图像也是 token：ViT 的 patch 分词 [27:25 - 30:22]

到目前为止我们确认了 Transformer 是极强的通用序列模型，能可预测地扩展、并在规模上涌现能力。但对机器人来说语言只是故事的一半——它必须感知世界才能决策。于是本讲第二个转折点：**如何把 Transformer/LLM 扩展到视觉观测？**

![章节过渡页：如何将LLM和Transformer扩展到视觉领域？](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0018_00-27-36.jpg)

关键洞见来自 ViT 论文（Dosovitskiy, Beyer et al., 2020，标题就是 An Image is Worth 16x16 Words），流程在幻灯片上被完整标注：

- 一张 $224\times224$ 的图像，切成 $16\times16$ 像素的 patch，得到 $14\times14=196$ 个 patch。注意这里"16×16"指的是**每个 patch 的像素尺寸**，网格本身是 $14\times14$（讲者口述时把两者混着说了，以图为准）。
- 每个 patch 展平：$16\times16\times3\ (\text{RGB})=768$ 个值，即 768 维向量。
- 过线性投影 $W$（768 → 768 的可学习权重），得到维度 $d=768$ 的 patch embedding。
- 加上位置嵌入（positional embedding），告诉 Transformer 每个 patch 来自图像何处。
- 于是得到"196 个 token 中的第 1 个"这样的 patch token，送进**与处理文本 token 完全相同**的 Transformer。
- 最终序列是 $196+1=197$ 个 token：$[\text{CLS}],\,p_1,p_2,\dots,p_{196}$。

哲学上与 BPE 一脉相承：找到合适的单位，转成向量，剩下的交给 Transformer。语言的单位是子词，图像的单位是 patch。接口一旦一致，文本 token 与视觉 token 就能混在同一条序列里。

![图像分词（Image Tokenization）流程图，展示ViT将图像切分为patch并嵌入的过程](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0019_00-28-31.jpg)

## CLIP：双塔对比对齐图文 [30:22 - 31:59]

"一切皆 token"带来一个直接的问题：如果只是把图像 token 和文本 token 拼在一起，Transformer 完全不知道哪个来自图像、哪个来自文本，更不知道它们如何对应。我们 ideally 希望"apple"这个词的文本 token 能对应到图像里苹果所在区域的视觉 token，这样之后才能用语言指令让机器人去拿苹果。**怎么教模型对齐这两种模态？**

![提出核心问题的幻灯片：Transformer如何知道哪些文本token对应哪些视觉token](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0020_00-30-22.jpg)

最早的答案之一是 OpenAI 的 CLIP（Radford et al., 2021）。它训练**两个独立编码器**（双塔）：图像侧一个 ViT，文本侧一个 Transformer，然后用对比学习把两者对齐到共享嵌入空间。具体地，取一批 $N$ 个从互联网爬取的图文对，全部编码后算一个 $N\times N$ 的余弦相似度矩阵；图中左下的方阵里只有对角线（$I_i$ 与 $T_i$ 相交处）被高亮，那正是配对的样本。温度参数 $\tau$ 是**可学习的**，用来控制所得分布的熵/尖锐程度。损失是对称的 InfoNCE：

$$\mathcal{L}_{\mathrm{CLIP}}=-\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{e^{s_{ii}/\tau}}{\sum_{j=1}^{N}e^{s_{ij}/\tau}}+\log\frac{e^{s_{ii}/\tau}}{\sum_{j=1}^{N}e^{s_{ji}/\tau}}\right]$$

第一项是 Image-to-Text Loss（第 $i$ 张图要在一排文本里认出自己的描述），第二项是 Text-to-Image Loss（反向）。注意两个分子的指数项都是**对角相似度** $s_{ii}$，机器 OCR 把它读成了 $s_{it}$ 与 $s_{ti}$，是错的。

这样得到的视觉表示在语义上与语言对齐：苹果的图片与"apple"这个词会落在嵌入空间中彼此靠近的位置——这正是我们想要的。

![CLIP双塔对比对齐架构图及损失函数公式，展示图文相似度计算与温度参数](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0021_00-31-17.jpg)

## 三种把视觉塞进 LLM 的范式 [31:59 - 37:00]

CLIP 学的是"两塔共享的嵌入空间"，但里面并没有一个 LLM。LLaVA（Visual Instruction Tuning，Liu et al. 2023）问的是另一个问题：**如何把视觉信息直接注入一个预训练 LLM**，让它像处理文本一样处理图像。答案是**早期融合（early fusion）**：图像过**始终冻结**的 CLIP 视觉编码器得到 patch embeddings，再过一个**可训练**的小 MLP 连接器投影到合适维度，然后把这些图像 token（图中画了 $i_1,i_2,i_3$ 共 196 个）**前置**到文本 token 序列（$w_1\dots w_4$）之前，LLM 在**一次统一的自注意力**里处理整条序列，架构不做任何改动。右侧框里三行字概括得很干净：self-attention over all tokens jointly / standard autoregressive generation / no new attention mechanisms。所以关键设计选择就是"什么冻结、什么训练"：CLIP 编码器永远冻结以保住预训练成果；MLP 连接器学习如何把视觉嵌入投影进 LLM 的 token 空间；LLM 本身在视觉指令数据上微调，学会跟随以图像为根据的指令。

![Llava早期融合架构图，展示图像token前置到文本序列并由LLM统一处理的过程](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0022_00-32-12.jpg)

第三种路线来自 Flamingo（Alayrac et al., 2022）：**晚期融合 + 门控交叉注意力**。区别在于视觉信息"在哪里、以什么方式"进入语言模型。LLaVA 把图像 token 前置到文本序列、共用同一套自注意力；Flamingo 问的是：能不能让 LLM 的自注意力仍然只处理文本，而通过门控交叉注意力**在每一层**注入视觉信息？做法是视觉编码器保持冻结，其输出用 Perceiver resampler 压缩成一组固定尺寸的键与值；LLM 每一层新插入一个交叉注意力块，文本 token 作为查询去 attend 这些视觉键值。幻灯片标题写得很直白："Vision is injected at every layer via cross-attention, the LM never sees image tokens directly"。更新规则是

$$x \leftarrow x+\tanh(\alpha)\cdot \text{x-attn}(Q,K,V)$$

优雅之处在于 $\tanh$ 门控：$\alpha$ 是**初始化为 0** 的可学习量，于是训练一开始 $\tanh(0)=0$，交叉注意力贡献恰好为零，模型行为与原 LLM 完全一致（"LM fully preserved"）；随训练推进门逐渐打开，视觉信息才流进来。这是"保住预语言能力、同时慢慢学会用视觉"的一个很有原则性的做法。图中"trained"框明确列出被训练的三样东西：交叉注意力层、Perceiver resampler、以及每层一个的 $\tanh$ 门控参数 $\alpha$；冻结的是视觉编码器和 Chinchilla LM 主干的自注意力（注意：机器图注把 resampler 也当成冻结，是错的）。

![Flamingo模型架构图，展示通过门控交叉注意力在每一层注入视觉信息的机制](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0023_00-34-02.jpg)

LLaVA 与 Flamingo 共享同一个前提：先有一个预训练语言模型，再想办法把视觉加上去。第三条路是干脆把这些全丢掉、**从头联合训练一切**，即原生多模态模型：单个统一 Transformer 同时在所有模态上训练，图像、音频、视频、文本以任意顺序、任意组合交错输入。幻灯片给了四个例子——图像描述（img img img txt txt txt → "describe this image"）、音频转写（aud aud aud aud txt → "transcribe this"）、视频问答（vid txt vid txt vid txt，freely interleaved）、多模态推理（img txt aud img txt aud，any mix, any order），并强调"each modality is optional"。模型没有"主模态 + 外挂"之分，它看到的只是一串 token 并对全部联合推理；底部那行"every weight has always seen every modality"点明了本质。吸引力显而易见：这是把 Bitter Lesson 推到极致的形态，架构上对任何模态都没有偏置；代价也很明显——训练最贵。

![原生多模态模型（Native Multimodal Models）架构示意图，展示多种模态输入统一处理流程](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0024_00-35-53.jpg)

## VLM 谱系与三者的取舍 [37:00 - 38:38]

今天主流 VLM 基本都能归到这三类。幻灯片的分类表信息量很大：

- **early fusion（pre-trained LLM + vision adapter）**——开源社区最流行的一族：LLaVA·2023（CLIP ViT + MLP + LLaMA）、PaliGemma 2·2024（SigLIP + linear proj. + Gemma 2）、InternVL 2.5·2024（InternViT-6B + MLP + LLM）、DeepSeek-VL2·2024-12（SigLIP + MLP + DeepSeekMoE）、Gemma 3·2025-03（SigLIP + linear proj. + Gemma 3）、Phi-4-multimodal·2025-02（SigLIP-2 + MoE-LoRA + Phi-4）、Qwen2.5-VL·2025（dyn-res ViT + MLP + Qwen2.5）、Qwen3-VL·2025（DeepStack + MRoPE + Qwen3）、Mistral Large 3·2025（sparse MoE 675B / 41B + vision）。配方都一样：换一个更好的视觉编码器、换一个更强的基座 LLM，然后迭代。
- **late fusion（pre-trained LLM + cross-attention）**——家族小一些：Flamingo·2022（gated cross-attn + Chinchilla LM）、GPT-4V·2023（undisclosed · likely cross-attn）、LLaMA 3.2 Vision·2024（cross-attention + LLaMA 3）、IDEFICS 2·2024（open Flamingo reproduction）、NVLM-X·NVIDIA 2024（cross-attn + InternViT-6B）。
- **natively multimodal（joint training from scratch）**——大型闭源模型正在收敛的方向：Gemini 1·2023（dense transformer · from scratch）、Chameleon·2024（VQ-VAE：图像作为共享词表里的离散 token）、GPT-4o·2024（natively multimodal · undisclosed）、Gemini 3·2025（sparse MoE · 1M ctx · Deep Think）、LLaMA 4·2025（MoE · 400B / 17B · 1M ctx）、GPT-5·2025-08（native MM · unified routing）、Qwen3.5·2026-02（joint pretraining · text + vision）。

权衡的本质是**灵活性 vs 可扩展性**：早期融合让图文在同一条自注意力里自由混合，跨模态交互最丰富，但你必须微调 LLM，有损害其原有能力的风险；晚期融合完美保留预训练 LLM，但两种模态只能通过交叉注意力通路交流，交互方式受限；原生多模态解除所有约束，代价是一切从头训练。

![视觉语言模型（VLM）分类表，按早期融合、晚期融合和原生多模态三类列举代表性模型](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0025_00-37-43.jpg)

## 机器人 = 多模态序列预测 [38:38 - 40:29]

从 Transformer 到 LLM、到 VLM、到原生多模态模型，每一步的打法完全相同：**找到把新模态 token 化的正确方式，喂进同一个架构，然后放大**。文本、图像，再到音频、视频。那么显然的问题就是：机器人动作能不能这么干？

![展示从Transformer到原生多模态模型的演进路径，并提出将其扩展到机器人动作的疑问](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0026_00-38-38.jpg)

于是机器人学可以被看成多模态序列预测问题：语言 token、图像 token，以及取代了"回答文本"的**动作 token**——它们直接驱动电机。幻灯片用"Pick up the spoon"配一张桌面（勺子、西瓜、蘑菇、抹布）与动作定义 $\text{ACTION}=[\Delta x,\ \Delta\theta,\ \Delta\text{Grip}]$ 来说明三个模态如何对齐。架构上不需要任何本质改动，还是 next-token prediction，只是往词表里加了新的动作 token。而注意力在这里恰好回答了那个最初的问题：**"我的整段历史里，哪些部分与我现在该做什么相关？"** ——这正是在解决开头列出的记忆问题。

![将机器人任务建模为多模态序列的示意图，包含语言指令、图像输入和动作输出三个模态](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0027_00-39-34.jpg)

## 动作分词化第一步：逐维逐时刻 256 bin [40:29 - 42:50]

动作是连续的，而 token 是离散的。RT-2 与 OpenVLA 用的第一种办法很直接：**逐维、逐时刻分箱（per-dimension, per-timestep binning）**。幻灯片四步流程：

1. **逐维分位数归一化**：对每个动作维度，在训练数据上算第 1 与第 99 百分位并截断到该范围，再归一化到 $[-1,+1]$。图上对比了两台机器：robot A（small gripper）$Q_1=-0.02,\ Q_{99}=+0.04$；robot B（large arm）$Q_1=-1.50,\ Q_{99}=+2.30$——量纲差两个数量级，但归一化后"identical for every robot"，离散化因此跨机器人可比。为什么不用 min-max？RT-2 最初用 min-max，但离群点会把整个区间撑大，结果大部分分布被挤进前 3 个 bin，剩下 253 个 bin 只为表示那一个离群值而浪费掉。分位数边界正是为了防止离群点吃掉 bin 分辨率。
2. **均匀切 256 份**：把 $[-1,+1]$ 均分为 256 个 bin（bin 0 … bin 255），bin 宽 $2/256\approx 0.0078$。
3. **$N$ 维动作 → $N$ 个离散 token**：例子里 $a_t\in\mathbb{R}^7$，输出 183、107、240、018、155、072、201 七个 token。
4. **注入预训练语言模型词表**：保留原有词表，用 256 个槽位承载动作 token $[0..255]$。常见做法是**覆盖词表里最低频的 256 个 token**（往往是些奇怪的符号/生僻字符），这样就不必改动 embedding 表的大小。之后模型仍然用同样的交叉熵、同样的架构做 next-token prediction。

文献：RT-2（Brohan et al., 2023）、OpenVLA（Kim et al., 2024）。

![VLA模型中机器人动作的Token化流程图，展示了分位数归一化、离散化及注入词表的具体步骤](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0028_00-40-29.jpg)

## 动作分块与时序集成：ACT [42:50 - 46:39]

回到开头那两个问题：记忆已经由序列建模 + Transformer 解决了，**动作平滑性**的直接答案是动作分块（action chunking）——不再只预测下一步，而是让模型一次输出一段未来轨迹：

$$\pi(a_{t:t+k}\mid o_t)$$

最基础的版本（图上半部分）：query 1 一次给出 $a_0,a_1,a_2,a_3$，执行完这 4 步再 re-query 得到 $a_4,a_5,a_6,a_7$。因为 4 个动作是同一次规划出来的，运动明显比逐步决策流畅。

但把 chunk 拉长以适应更高频率控制（比如人形机器人，chunk 长到 100）会带来一个失效模式：如果在两次查询之间把 100 个动作全部执行完，那大约 2 秒里**机器人是瞎的**，不更新任何观测，期间发生意外它完全无法反应。解决办法是图下半部分的**时序集成（temporal ensembling）**：每个时间步都 re-query，于是同一时刻会有多个重叠的预测。对给定时刻的预测按指数衰减加权平均：

$$w_i=\exp(-m\cdot i),\qquad \hat a_{3}=\frac{\sum_i w_i a_i}{\sum_i w_i}$$

其中 $i=0$ 表示最旧的预测。右侧权重曲线用的是 $m=0.01$，即 $w_i=e^{-0.01 i}$：$w_0=1.0$ 是最高权重（oldest），到横轴 50 处 $w_{50}=e^{-0.5}\approx 0.61$ 是最低权重（newest）。（机器 OCR 把这里读成"$W\_60=0.61$"，实际横轴最右端是 50，且 $e^{-0.01\times 50}\approx0.61$ 也自洽。）这样既拿到分块的平滑性，又对每一步的新观测保持反应性。

![动作分块（Action Chunking）与时序集成机制示意图及权重衰减曲线](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0029_00-43-14.jpg)

ACT（Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware，Zhao et al., 2023；一作 Tony Zhao 是 Sunday Robotics 联合创始人，上周刚来做过客座讲座）把动作分块引入了**双臂精细操作**：双臂机器人以 50 Hz 控制，模型一次预测 100 个动作；由于是 14 自由度的关节控制，每次前向传播要输出 **1400 个值**。这正是需要平滑的场景——插电池这类任务要求两臂高度协调、对抖动几乎没有容忍度；一个在 50 Hz 下逐步独立决策的反应式策略会做得很差，而规划约 2 秒长轨迹块则流畅得多。

![ACT模型在双手机械臂上的实时策略演示（插入电池任务）](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0030_00-45-17.jpg)

## 朴素动作分词在高控制频率下失效 [46:39 - 49:00]

ACT 说明更大的动作块有利于平滑的高频灵巧控制，但它没有用 LLM，也不是以自回归方式逐个预测 1400 个 token。一旦你想在 VLA 里这么做，事情很快就崩，尤其是频率越高越糟。原因写在幻灯片上：**随着频率提高，每个 token 携带的信息减少，而 token 之间的相关性上升**。频率高到一定程度，相邻两帧之间几乎没有运动。而自回归模型的学习信号来自每个 token 提供的**新信息**；当 token 高度相关时，下一个 token 的边缘信息量趋近于零。后果很严重：模型只要学会**复制上一个动作 token**就能把预测误差压得很低，于是它永远学不到真正的灵巧技能。

左图把这个失效量化了：横轴是数据控制频率（2 Hz、5 Hz、10 Hz、20 Hz），纵轴是 task success score，用朴素分箱 tokenization 的自回归 Transformer 在 **5 Hz 达到峰值约 3.6**，2 Hz 只有约 1.0（太低则动作块内信息不足、序列利用不充分），10 Hz 掉到约 0.5，20 Hz 已接近 0。可用窗口非常窄。

![朴素动作分词化分析图，展示数据控制频率与任务成功率的关系曲线及自回归Transformer结构示意](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0031_00-46-39.jpg)

## 有效动作分词化：DCT + 量化 + BPE（FAST） [49:00 - 53:24]

出路显然是**压缩**：不要逐个动作地 token 化，而是找一个更紧凑的表示，就像语言里的子词、图像里的 patch 那样，抓住运动的有意义结构——比单个时间步大、但比整条轨迹小的单位。可连续数据怎么压？BPE 在离散域很好用，但不适合连续流；VQ-VAE 这类学习式压缩又要多养一个模型、训练复杂。本讲给出的方案是从信号处理借来的**离散余弦变换（DCT）**——JPEG 压缩的核心就是它。

直觉是：机器人动作块是平滑轨迹，而**平滑信号的能量集中在低频**。于是流程（幻灯片编号 1→5）是：

1. 归一化后的动作块（图中显示前 2 个维度的曲线）；
2. 对每个动作维度做 DCT，从时域换到频域，得到频率分量；
3. **量化**这些系数，得到稀疏频率矩阵（每一维一行），高频项几乎全为零。图中的例子是 5 行：`124 12 −3 0 0 0 12` / `−86 0 0 0 0 0 0` / `344 3 1 0 0 1 5` / `−45 0 0 0 0 0 0` / `178 15 −1 0 0 0 −2`；
4. **展平，低频分量排在前面**：`124 −86 344 −45 178 12 0 3 0 15 …`；
5. 在这串整数序列上跑 **BPE**，它会把零压掉、把常见的系数组合合并，得到紧凑的动作 token（图中合并成 978、233、19、1022、1），可以直接插进现有词表。

![有效动作分词化流程图，展示从归一化动作块经DCT变换、量化到BPE压缩的完整过程](/coursenotes/eth-robot-learning-2026/l07-sequence-modeling/kf_0032_00-49-41.jpg)

结果（讲者口述，本段已超出关键帧覆盖范围）：换上这套压缩式分词后，自回归 VLA 的性能不再在 5 Hz 之后崩掉，而是随控制频率继续提升——压缩恰好做了它该做的事：保留重要信息、丢掉冗余的高频相关性。另一组对比里，讲者把 $\pi_0$ 改造成自回归版本（用这套分词方案、架在 PaliGemma 基座上），与 Physical Intelligence 原版用 flow matching 生成动作的 $\pi_0$ 相比，**用 5 分之一的训练量收敛到同样性能**（这是讲者在 Berkeley 期间与 Physical Intelligence 的合作工作）。同一工作还在 DROID 数据集上训练出能在真实环境中做零样本语言条件任务的政策，在 Berkeley、Stanford、University of Washington 未见过厨房里表现良好，并能以 50 Hz 完成叠衣服这类长时序双臂操作。意义在于同时推进**通用性与灵巧性**这两个通常互相拉扯的轴：像 ACT 这样的专家政策精确但窄，像 OpenVLA 这样的通用政策宽但不够灵巧，而这类工作让两者一起前进。副产品是一个开箱即用的动作 tokenizer，已发布在 Hugging Face 上，可直接用于训练 VLA。

## 全讲收束：把 Bitter Lesson 用到机器人上 [53:24 - 55:11]

结论回到开头那个朴素观察：反应式策略之所以有根本局限，是因为世界的一张快照无法刻画完整状态，而每步独立的决策必然抖动。修复办法是**把机器人学重新表述成序列建模问题**，然后围绕它搭起一套工具箱：

- Transformer 给出可扩展的序列建模架构，对历史任意位置都是 $O(1)$ 访问；
- 把 Transformer 放大得到 LLM，出现涌现能力；
- 通过早期融合、晚期融合或原生多模态训练扩展到视觉，得到能联合推理图像与语言的 VLM；
- 最后一步扩展到机器人动作，而这一步并不平凡：**朴素 tokenization 在高控制频率下会崩**，需要 FAST 这类基于压缩的方案，才能让自回归动作生成真正工作在机器人要求的频率上。

贯穿全讲的中心思想，就是 Bitter Lesson 的机器人版本：**找对表示，用通用可规模化的架构，然后让数据与算力去干活。**

## 要点回顾

- 反应式策略 $\pi_\theta(a_t\mid o_t)$ 的两大失效：单帧快照无法表达完整状态（记忆、速度估计），单步决策导致动作抖动；机器人实际在 POMDP 中，必须对历史 $\tau$ 推理。
- 轨迹概率 $p_\pi(\tau)=\rho(s_0)\prod_{t}\pi(a_t\mid o_t)P(o_{t+1}\mid o_t,a_t)$ 暴露了"世界是序列、策略不是"的错配；链式法则 $p_\theta(x)=\prod_i p_\theta(x_i\mid x_{1:i-1})$ 把问题归约为"如何学每个条件分布"。
- RNN 的三堵墙：长程依赖（梯度爆炸/消失，路径 $O(n)$）、无法并行训练、固定尺寸 $h_t$ 的信息瓶颈。
- 注意力把任意两位置间的**路径长度**降到 $O(1)$（注意：这不是计算复杂度，注意力在序列长度上仍是二次的），一次性解决 RNN 三个问题。
- $\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^{\top}/\sqrt{d_k})V$ 是"软的可微字典查询"：模型同时学到表示与检索策略。
- 注意力是集合操作、没有顺序概念，必须加位置编码；绝对 PE 在测试期遇到更长序列会失效（$\mathrm{PE}(6),\mathrm{PE}(7)$ 从未训练过），相对 PE 把 $b(i-j)$ 加在注意力分数上、可泛化到未见长度。
- 编码器双向、解码器因果（softmax 前把未来分数置 $-\infty$）、交叉注意力中 Q 来自解码器而 K/V 来自编码器；teacher forcing + 交叉熵，一次前向即可算出全部 $T$ 个损失。
- token 化：字符与词两个极端都不好用；BPE 自底向上合并最高频相邻对，示例句 54 字符 → 34 token（少 37%、0.63× 压缩、7 条规则），压缩率随语料规模增长；BPE 大小写敏感。
- LLM 的工程支柱：decoder-only 更易扩展、字节级 BPE 消除未知 token、RoPE 让 $Q\cdot K$ 只依赖相对距离、FlashAttention 用 SRAM 分块把显存从 $O(n^2)$ 降到 $O(n)$。
- 规模涌现：GPT-1（117M/1B）微调 → GPT-2（1.5B/10B）零样本 → GPT-3（175B/300B）上下文学习；1.3B 曲线平躺而 175B 随示例数上升，印证 Bitter Lesson。
- Chinchilla 定律：$N^{\star}\propto C^{0.5}$、$D^{\star}\propto C^{0.5}$、$C\approx 6ND$、约 20 token/参数；可用小规模实验外推大规模表现；LLaMA 3 故意"过度训练"以换取推理成本，即 inference-optimal。
- 视觉：ViT 把 $224\times224$ 图像切成 $14\times14$ 个 $16\times16$ 像素 patch（196 个，加 [CLS] 共 197 token），展平 768 维、线性投影、加位置嵌入——与文本 token 接口完全一致。
- 图文对齐三条路：CLIP 双塔 + 对称 InfoNCE（可学习温度 $\tau$，分子是对角相似度 $s_{ii}$）；LLaVA 早期融合（冻结 CLIP + 可训练 MLP 连接器，图像 token 前置，LLM 不改架构）；Flamingo 晚期融合（Perceiver resampler 压成固定 K/V，每层门控交叉注意力 $x\leftarrow x+\tanh(\alpha)\cdot\text{x-attn}(Q,K,V)$，$\alpha$ 初值 0 保证起步等价于原 LLM；被训练的是交叉注意力层、resampler 与各层门控）；原生多模态则从头联合训练所有模态。
- 机器人 = 多模态序列预测：语言 + 图像 + 动作 token 进同一架构，动作 $[\Delta x,\Delta\theta,\Delta\text{Grip}]$ 用同样的 next-token prediction 学。
- 朴素动作分词：逐维分位数（$Q_1,Q_{99}$）归一化到 $[-1,+1]$、均分 256 bin（宽 ≈0.0078）、覆盖词表最低频的 256 个 token（RT-2、OpenVLA）。
- 动作分块 $\pi(a_{t:t+k}\mid o_t)$ 解决平滑性；长 chunk 会让机器人在执行期间"失明"，时序集成用 $w_i=\exp(-m\cdot i)$ 加权融合重叠预测（$w_0=1.0$、$w_{50}\approx 0.61$）；ACT 在双臂 50 Hz、chunk 100、14 自由度下每次预测 1400 个值。
- 高频下相邻动作 token 相关性极高、边际信息趋零，模型退化成"复制上一个 token"，成功率在 5 Hz 见顶、10–20 Hz 崩塌。
- FAST 式有效分词：DCT 换到频域（平滑动作能量集中在低频）→ 量化得稀疏矩阵 → 低频优先展平 → BPE 压缩整数序列，类比 JPEG 压缩；结果自回归 VLA 不再在 5 Hz 后失效，且比 flow matching 的 $\pi_0$ 快 5 倍收敛，同时推进通用性与灵巧性。
- 一句话：找对表示 + 通用可规模化架构 + 数据与算力，就是 Bitter Lesson 在机器人学上的落地。
