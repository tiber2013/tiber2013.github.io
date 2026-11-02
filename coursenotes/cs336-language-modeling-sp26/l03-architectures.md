---
title: "第 3 讲 · 从零设计 Transformer 架构"
description: "架构设计没有简洁的理论工具可用，最实用的方法是从大量已发布模型的共同选择中提炼经验规律。视频的核心判断是：LayerNorm 移出残差流、改用 RMSNorm、去掉偏置项、使用门控线性单元（GLU） 这几项已成为几乎所有…"
---

# 第 3 讲 · 从零设计 Transformer 架构

> **Stanford CS336 语言模型从零构建 · L03** ｜ 讲者 Stanford Online ｜ 89 分钟 ｜ [原视频](https://www.youtube.com/watch?v=lVynu4bo1rY)

## 核心结论

架构设计没有简洁的理论工具可用，最实用的方法是从大量已发布模型的共同选择中提炼经验规律。视频的核心判断是：**LayerNorm 移出残差流、改用 RMSNorm、去掉偏置项、使用门控线性单元（GLU）** 这几项已成为几乎所有现代语言模型的共识，而位置编码（RoPE）、前馈维度倍率、头维度、宽深比等则落在取值后对结果影响很小的宽松区间内。架构选择本质上是三重约束下的权衡：泛化能力、GPU 训练/推理效率、训练稳定性，后两者往往比表达力更决定最终设计。当前最活跃的变动领域是长上下文处理——注意力变体（GQA、滑窗/全局交替、混合 SSM）与稳定性干预（z-loss、QK-norm、logit soft cap）。

---

## 一、课程定位：为什么要做架构综述 [00:05 - 07:29]

讲者坦承架构设计长期以来"难以捉摸"（inscrutable），本讲标题戏称为"你不想知道的关于架构与超参数的一切"。原本理想世界只需 VC 维度这类简单理论工具就能推理架构，但现实并非如此。

学习架构的最佳方式是亲自训练模型、尝试不同设计；在算力与时间不允许覆盖整个设计空间的前提下，第二好的方式是**从他人的经验中学习**：把大量已发布模型摊开来看，区分哪些选择是所有有效架构都固定的，哪些是可以自由变动的。

架构必须同时满足三件事，这也是它"不优雅"的根源：

- 从数据中学习，即具备泛化能力；
- 在 GPU 上高效训练；
- 训练不能崩溃——损失不能中途突然爆炸。

历史脉络上，讲者把 Transformer 之后的架构演化分成三段：从原始 Transformer 到 GPT-3 是大量实验、没有统一标准的时期；LLaMA 2 发布后，所有人开始训练"LLaMA 2 变体"，只做小幅改动；近一两年则重新开始探索，先是一波提升训练稳定性的架构修改，今年则集中出现面向长上下文的架构变体。随着每年发布大量新模型，样本多样性反而让人能看清选择的全貌。

![课程大纲与目标幻灯片，包含现代Transformer回顾、大模型共性及架构变体等议题](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0001_00-00-57.jpg)

![课程内容大纲幻灯片，列出了将要讨论的架构变体、超参数和稳定性技巧](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0003_00-05-57.jpg)

---

## 二、LayerNorm 放在哪里：唯一没有争议的问题 [07:29 - 13:55]

讲者认为 Transformer 原论文"大部分都做对了，只有一件事公认做错了"，那就是 LayerNorm 的位置。

**原始方案（post-norm / residual norm）**：LayerNorm 放在残差流内部，即在每个子层输出加回残差流之后做归一化：

$$x_{l+1} = \text{LayerNorm}\big(x_l + \text{MultiHeadAttn}(x_l)\big)$$

**现代方案（pre-norm）**：LayerNorm 移到残差流之外、每个计算之前：

$$x_{l+1} = x_l + \text{MultiHeadAttn}\big(\text{LayerNorm}(x_l)\big)$$

几乎所有现代语言模型都把 LayerNorm 推出残差流。一个有趣的例外是 OPT-350M 仍使用残差流内的 post-norm（讲者直言 OPT 系列本身就比较混乱）。

![Pre-LN与Post-LN Transformer架构对比图及数学公式](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0008_00-11-19.jpg)

![Pre-LN与Post-LN Transformer结构对比图及公式](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0009_00-11-54.jpg)

**为什么。**早期研究的动机是去掉训练时必须的 warm-up。实验发现 post-norm 即使加 warm-up 收敛也较差（图中紫色虚线），而 pre-norm 即使不加 warm-up 也有好得多的收敛。但真正让它固定下来的是更深层的两点：

1. **梯度衰减**：pre-norm 下残差流从输入直通输出，反向传播可以沿这条"干净"的路径直接流过，梯度尺度大致保持不变；post-norm 则每经过一个 block 都做一次归一化，梯度范数随之改变。这就是"保持残差流干净"（keep your residual stream clean）的经验法则。
2. **梯度尖峰**：Salazar 与 Nguyen 的实验显示，pre-norm 下梯度尖峰的大小和频率都明显改善。

![对比Pre-Norm和Post-Norm在不同任务（IWSLT, BERT）上的验证损失和BLEU分数曲线图](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0005_00-10-25.jpg)

![解释Pre-norm与Post-norm差异的梯度衰减和梯度尖峰图表](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0007_00-10-45.jpg)

**归一化还可以放在残差流之外的计算之后。**既然问题出在残差流内，把 LayerNorm 放在子层计算之后、残差相加之前同样成立。Grok、Gemma 2、Olmo 2 就采用这种"非残差 post-norm"结构；也有模型两头都放（double norm）。讲者提到一条屡试不爽的经验：**遇到稳定性问题，就往里塞一个 LayerNorm**，通常都有用。

![对比传统残差后归一化与双归一化/非残差后归一化结构的架构图](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0011_00-13-39.jpg)

---

## 三、RMSNorm 与去掉偏置项 [13:55 - 20:13]

原始 Transformer 使用标准 LayerNorm：

$$y = \frac{x - E[x]}{\sqrt{\text{Var}[x] + \varepsilon}}\gamma + \beta$$

现代模型几乎都用 RMSNorm：不减均值、不加偏置，只做缩放再缩放回去。从表达能力上看 LayerNorm 更强，没有表示层面的理由必须换，但实践中有两个理由：

**一是性能不损失。** RMSNorm 模型的效果与 LayerNorm 相当。

**二是速度更快，属于系统与架构协同设计。** 关键在于算术强度：GPU 适合做矩阵乘这类计算密集的操作，而统计归一化虽然只占约 0.17% 的 FLOPs，却可能占到运行时间的 25%（小模型上尤其极端），因为它的开销主要来自激活值在快慢存储之间的搬运，而不是计算。去掉均值和偏置，就是去掉这些小而费内存、表达力贡献又低的操作。

![解释RMSNorm优势的幻灯片，包含公式及不同算子类的FLOP占比表格](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0012_00-14-52.jpg)

Narang 等人的实验进一步表明，在 2.23 亿参数的小 Transformer 上换用 RMSNorm，不仅每秒步数提升，最终损失反而略好——这算是额外收获，并非必然。

![RMSNorm与LayerNorm等归一化方法的性能对比数据表](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0014_00-17-31.jpg)

同样的逻辑推广到线性层的偏置项：现代实现基本全部丢弃。偏置不增加多少算术强度，但带来额外的内存搬运；在某些情况下还可能诱发稳定性问题。主要动机仍是从系统角度简化。

![LayerNorm与RMSNorm的实践要点回顾总结](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0015_00-19-20.jpg)

讲者也承认这类结论"无法事先推理"，只能靠大量实验积累的集体经验——在典型语言建模负载下，丢掉线性层和 RMSNorm 的偏置是安全的。

---

## 四、激活函数与门控线性单元 [20:13 - 27:26]

激活函数名目繁多：ReLU、GELU、Swish、ELU、GeGLU、SeLU、SwiGLU、LiGLU……但真正重要的一条轴是**是否门控**。

标准前馈层：

$$\text{FF}(x) = \max(0, xW_1)W_2$$

GELU 版本只是把 ReLU 换成带一小段"凹陷"的 GELU，在零点附近的梯度不同；GPT-3 用的就是它，完全可用。

**门控版本**：在 ReLU 之外再加一路乘法门，引入第三个矩阵 $V$：

$$\text{FF}_{\text{ReGLU}}(x) = \big(\max(0, xW_1) \otimes xV\big)W_2$$

命名规则是"第一个激活名 + GLU"：ReLU → ReGLU，GELU → GeGLU，Swish → SwiGLU。Google 系（Gemma、T5）多用 GeGLU，LLaMA 系（PaLM、LLaMA 及其衍生）用 SwiGLU；总体 SwiGLU 更主流，但门控单元之间差别不大。

![常见激活函数对比：ReLU与GeLU的公式、曲线及对应代表性模型](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0016_00-20-49.jpg)

![门控激活GLU的推导：从ReLU前馈层到ReGLU的公式演变](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0017_00-22-18.jpg)

**一个重要的参数配平细节**：门控版本有三个矩阵而非两个，参数量更多。为保持总参数不变，通常把前馈维度缩小到原来的 2/3，因此 GLU 模型的 FFN 倍率约为 2.67 而非 4。Shazeer 的原始论文正是做了这种参数匹配的对照实验，结论是 GLU 变体在各种设置下都稳定更优。

![不同前馈层激活变体的性能对比表格（Score Average、CoLA MCC、SST-2 Acc）](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0018_00-25-17.jpg)

也有反例：GPT-3 没用 GLU；Nemotron 340B 用了 Squared ReLU 这种"大胆"选择。但如今很少见到不用门控线性单元的模型。讲者总结：门控是几乎无计算代价的稳定增益。

![关于门控与激活函数（Gating, activations）的总结幻灯片，对比了ReLU、GeLU、GLU等变体及Swi/GeGLU的增益](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0019_00-26-46.jpg)

---

## 五、串行 vs. 并行层 [27:26 - 29:30]

标准 Transformer block 是串行的：先算 attention，再算 MLP。从系统角度看这构成依赖瓶颈。GPT-J 最早提出并行形式，PaLM 采用了它：

$$y = x + \text{MLP}(\text{LayerNorm}(x)) + \text{Attention}(\text{LayerNorm}(x))$$

并行版本可以共享 LayerNorm、融合矩阵乘，PaLM 报告称在大规模下约有 15% 的训练加速，且 8B 规模有轻微质量下降、62B 规模无下降。

![关于并行层（Parallel layers）的幻灯片，对比了标准串行公式与并行公式，并说明其训练加速效果](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0020_00-28-15.jpg)

但这一做法在过去两年明显失宠。讲者的解读是：串行形式的系统优化已经足够好，并行带来的收益抵不上表达力的损失——直观上相当于把模型深度砍了一半。问答环节补充：PaLM 论文对此非常乐观，但后续 Google 自家模型大多不再使用，这本身就是一个隐含信号，说明可能有性能损失；可惜没有公开的、控制良好的并行 vs. 串行消融实验。目前仍采用并行层的包括 Cohere Command A、Falcon 2 11B、Command R+ 等。

---

## 六、核心架构小结 [29:30 - 31:09]

讲者用一张表把上述选择归纳为"共识项"：

- **Pre vs. post norm**：所有人都用非残差归一化（除 OPT-350M）；
- **LayerNorm vs. RMSNorm**：RMSNorm 有明确的算力优势，有时性能还更好；
- **门控**：GLU 已是共识；
- **串行 vs. 并行层**：多数模型回到串行。

这也说明原始 Transformer 的骨架其实相当经得起时间考验——真正改动的地方（归一化位置、偏置、门控）都相对细小。至于改变注意力的 Transformer 替代方案，留到下一讲。

![架构总结幻灯片，归纳了Pre-vs-post norm、Layer vs RMSnorm、Gating及串并行层的共识，右侧附有模型架构对比表](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0021_00-30-44.jpg)

---

## 七、位置编码与 RoPE [31:09 - 39:02]

注意力本身对位置不敏感（只是内积，打乱顺序结果不变），因此位置信息必须显式注入。常见方案：

- **正弦余弦嵌入**：原始 Transformer，类似傅里叶直觉；
- **绝对位置嵌入**：每个位置一个独立向量，GPT-1/2/3、OPT 使用；
- **相对位置嵌入**：不加入词嵌入，而是给注意力矩阵加一个随位置差变化的偏移，T5、Gopher、Chinchilla 使用；
- **RoPE**：2024 年之后绝大多数模型采用。

![位置编码的多种变体对比：正弦、绝对、相对及RoPE嵌入的公式与代表模型](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0022_00-31-09.jpg)

### RoPE 的设计动机

理想中的相对位置编码应满足：

$$\langle f(x, i), f(y, j)\rangle = g(x, y, i-j)$$

即两个 token 嵌入的内积只依赖它们的相对位置 $i-j$。已有方案都不满足：正弦嵌入含有非相对的交叉项；绝对嵌入顾名思义不相对；相对嵌入 $e_{ij} = \frac{x_i W^Q (x_j W^K + a_{ij}^K)^T}{\sqrt{d_z}}$ 虽然相对，但它直接加在注意力矩阵上，不存在可提取的内积结构。

![RoPE的高层思想：相对位置编码应满足内积仅依赖相对位置(i-j)，并分析现有编码的不足](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0023_00-32-43.jpg)

### 旋转的几何直觉

内积对任意旋转不变。于是做法是：取与位置无关的语义词向量，按 token 所在位置旋转它。例如 "we know that" 中 we 在位置 0 不旋转、know 在位置 1 旋转一个角度；换成 "of course we know"，we 和 know 分别旋转 2 和 3 个单位，两者之间的相对夹角仍然是 1——相对位置被保留，绝对位置被消去。

![用二维向量旋转示意图解释RoPE：通过旋转词向量使内积仅依赖相对位置](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0024_00-35-41.jpg)

![RoPE旋转位置编码原理图，展示二维平面旋转及Transformer中的位置嵌入应用](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0025_00-36-41.jpg)

高维情形没有唯一旋转，最简单的做法是**把 D 维向量切成若干对二维坐标，每对独立旋转**，每对的旋转频率不同：低频对旋转慢，可捕捉长程依赖；高频对旋转快，可捕捉邻近关系。论文用复数给出形式化动机，但直观理解就是逐对旋转。

实现上，在每次注意力计算时根据 position id 生成 cos/sin，作用到 query 和 key 上（可写成矩阵乘或逐元素旋转），位置不变性由此在每个注意力操作中强制成立：

```python
query_states = self.q_proj(hidden_states)
key_states = self.k_proj(hidden_states)
value_states = self.v_proj(hidden_states)

query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

cos, sin = self.rotary_emb(value_states, position_ids)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
# 之后与常规多头注意力相同
```

![RoPE（旋转位置编码）在注意力机制中的代码实现，展示了query/key/value的投影及旋转位置编码的应用](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0027_00-38-32.jpg)

讲者还提到 Gemma 4 的 P-RoPE（只旋转前两个坐标），以及问答中关于高维旋转的讨论——目前没有看到可行的推广方案，二维配对旋转已经足够。问答中也确认：正弦嵌入若作为加法嵌入会引入交叉项、可反推绝对位置，因此不属于纯相对编码；而 ALiBi 这类注入注意力矩阵的方案虽可行，但没有成为主流。

---

## 八、超参数：哪些真的重要 [43:41 - 65:01]

### 8.1 前馈维度倍率

即 W1/W2 的中间维度与模型维度之比，控制 MLP 的"丰富度"。经验值是 **4 倍**。

例外一：GLU 变体由于多一个矩阵，按 2/3 缩放后约为 **2.67**；LLaMA 系列出于 MQA 等高效注意力的考虑，把这个比率乘了 1.33，得到约 **3.5**，相当于更强调 MLP。所以实践中的取值是：非门控约 4，门控约 2.6 或 3.5。

例外二：T5 用了 **64 倍**，理由是矩阵乘越大硬件利用率越高；这是系统导向的极端选择，没有其他模型跟到这么高。有趣的是 T5 v1.1 又悄悄回到了标准的 2.5 倍。

Kaplan 2020 的缩放定律论文中有一张顺带的扫描图：在 1 到 10 之间损失曲线非常平坦，取任何值都接近最优；超过 10 往 100 走才急剧恶化。2.6 到 4 的选择全都落在这个平坦区间里。

### 8.2 头维度

惯例是让 $H \times d_{\text{head}} = d_{\text{model}}$，即多头注意力的总维度等于单头维度。绝大多数模型（包括最新的 Qwen）都遵循，T5 和 Lambda 是明显例外。这同样是一个很宽容的参数，消融显示在 1 附近有较宽的可用范围。

### 8.3 宽深比（aspect ratio）

缩放模型时通常固定宽深比再整体放大，因此该比率决定了深度-宽度的权衡。现代模型的典型值是

$$\frac{d_{\text{model}}}{N_{\text{layers}}} \approx 100$$

即每层约 100 维宽度。太深会带来系统问题：层切分需要流水线并行，这是多数团队极力避免的；太宽则容易做张量并行。表达力倾向深、系统效率倾向宽，最终收敛到约 100。Kaplan 的扫描也显示，不同规模下的最优宽深比相当接近 100。

另外，ERA 等人的深度-宽度权衡实验结论是：真正主导效果的是 FLOPs，宽深比在很宽的范围内不敏感。因此实践中人们更关心系统利用率，而非难以推理的表达力。

### 8.4 词表大小

存在明显的两类集群：

- **单语模型**（早期开源、只追求英文）：约 **30k**；
- **多语/生产系统**（GPT-4 类闭源模型、Google 模型、LLaMA 衍生）：约 **100k–200k**。

多语模型需要更大的词表覆盖语言空间；同时右侧模型普遍更大，而缩放定律研究表明大模型能承载更大词表。如今已很少有人训练大型单语模型。

问答中还讨论了 bits-per-byte 用于跨 tokenizer 比较的合理性：语言建模是在建模序列概率，只要 tokenizer 是完备的、不丢信息，且用同一个字节数归一化，BPD 比较就是有效的。

![Transformer超参数设置常见问题幻灯片，涵盖前馈层大小、注意力头数、词表大小及模型缩放策略](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0029_00-38-58.jpg)

### 8.5 正则化：一个反直觉的结论

标准论证是：语言模型训练数据远多于算力，通常只做单遍训练，SGD 单遍几乎不会记住数据，因此**过拟合基本不是问题**，有人甚至只看训练损失。

但现实是不少现代高性能模型仍在用 weight decay（dropout 则基本退出）。原因是：有研究论证 weight decay 在此场景下起的作用更接近**优化器干预**，并非正则化——它与学习率衰减配合，能让模型在后期收敛到更好的极小值。实验上，不同 weight decay 下训练/验证损失没有差异（本来就没有过拟合），但与学习率衰减结合后，强 weight decay 的运行最终损失明显更低。

![RoPE旋转位置编码的高层设计思路与现有嵌入方法的对比](/coursenotes/cs336-language-modeling-sp26/l03-architectures/kf_0030_00-42-30.jpg)

讲者借此强调：这些组件之间的交互方式很难先验推理，"动手试"才是这门课的设计哲学。问答中补充：训练过程中真正会动态调整的通常只有 weight decay（与学习率联动），架构类超参数一旦开训就无法更改。

---

## 九、训练稳定性技巧 [65:01 - 74:08]

模型越贵，稳定性越重要：中途梯度爆炸可能导致数百万美元的训练不可恢复，或产出质量很差的模型。不稳定通常从 softmax 找起——它同时含有**指数**（容易爆炸）和**除法**（容易除零或除出极值），而语言模型里有两处 softmax：输出端和注意力内部。

### 9.1 输出 softmax 与 z-loss

计算对数概率时：

$$\log P = u - \log Z$$

$u$ 是模型输出，通常良态；问题在归一化项 $\log Z$，它是指数求和，可能极大也可能趋零。由于 softmax 是过参数化的（给 $u$ 加常数可以在不影响输出的情况下移动 $Z$），可以加一个正则项把 $\log Z$ 拉向 0：

$$\mathcal{L}_{\text{z-loss}} = (\log Z)^2$$

这就是 z-loss 技巧，最早由 Jacob Devlin 等人在 2014 年提出，后经 Baichuan 首先在开源模型中采用，DCLM、Olmo 等跟进，效果出乎意料地好。

### 9.2 QK-norm

注意力内部是另一处退化高发区。做法是在 Q 与 K 相乘、进入 softmax 之前，分别对它们做 RMSNorm：

$$\text{softmax}\left(\frac{\text{norm}(Q)\text{norm}(K)^T}{\sqrt{d}}\right)V$$

这样 softmax 的输入尺度被固定在 1 附近。这个技巧最早来自多模态领域（E 等人为 Chameleon 开发），随后被开源语言模型广泛采用，现在几乎是大型模型的标准配置：不影响性能，但能有效防止注意力退化。讲者调侃这条演化路线：归一化原本只在 pre-norm 里，后来加到每个 block 的非线性之后，现在又塞进 Q 和 K。

### 9.3 Logit soft capping

更"硬"的干预：把进入 softmax 的 logits 用 tanh 软截断到固定范围，从结构上禁止过大或过小的 logit。Gemma 2/3/4 都用了这一招。NVIDIA 的系统比较显示：QK-norm 略有增益（因为可以调高学习率），而单独使用 soft capping 会损失性能——它限制了模型表达高置信度的能力，是一种强干预。但它确实非常安全。

---

## 十、注意力变体：从训练到部署 [74:08 - 88:37]

本节只讨论稠密全对全注意力，状态空间模型留到下一讲。

### 10.1 训练/prefill 与解码的算术强度差异

训练或 prefill 阶段，算术量约为 $O(BNd^2)$，访存量约为 $O(BNd + BN^2 + d^2)$，算术强度约为 $\frac{1}{k} + \frac{1}{BN}$（$k$ 为头维度）。只要头维度不太小、batch 或序列不太短，GPU 就能跑满。

自回归解码则必须逐 token 生成，用 KV cache 复用已算过的键值，只补算新的行列。计算量不变，但访存模式变成 $O(BN^2d + Nd^2)$，算术强度变成：

$$\frac{N}{d} + \frac{1}{B}$$

这个 $N/d$ 项很难压低——序列必须够短，或者模型必须够宽，小模型高效服务因此非常困难。

### 10.2 MQA 与 GQA

**MQA（multi-query attention）**：所有头共享同一组 K 和 V，只有 Q 不同。KV cache 大幅缩小，算术强度里出现因子 $H$，效率显著提升；代价是表达力明显下降。

**GQA（grouped-query attention）**：折中方案——保留全部 Q 头，但把 K/V 头分组共享，让 K/V 头数小于 Q 头数。这个比例直接控制表达力与推理效率的权衡。T5 时代的实验显示：多头性能最好但成本最高，MQA 成本低但性能掉得多，缩小模型来省成本掉得更狠，而 GQA 几乎拿到多头的性能、接近 MQA 的成本；而且 K/V 头数小幅减少就能拿到大部分收益。这正是今天几乎所有模型采用 GQA 的原因。DeepSeek-V2 的 MLA 是另一种分解结构，下一讲再谈。

问答补充：MQA/GQA 减少的是训练和推理都受益的 K/V 头数，训练时也是按这个配置直接训练的。

### 10.3 滑动窗口与混合注意力

滑窗注意力是很老的想法：GPT-3 就在全注意力层之间交替插入带状（固定窗口）注意力层。过去一年它强势回归，因为它正好落在长上下文性能与推理成本的甜点上。

典型结构是每若干层放一个全局注意力层，中间层用只看局部窗口的滑窗注意力：底层聚合局部信息，越往上局部层也能间接访问到更全局的信息。Cohere Command A 是讲者注意到的较早采用者（每四层一个全局层）；随后 Llama 4、Gemma 4、Olmo 3 等都采用滑窗 + 全局交替，并使用完整 RoPE。

Qwen 3.5 是同一思路的变体：把廉价的滑窗层换成状态空间模型（gated DeltaNet），每四层插一个全局注意力层。

还有一些模型进一步改动长程信息的位置编码方式，例如长程层干脆不用 RoPE（NoPE），此时长程部分近似词袋，短程部分仍带位置信息。

讲者总结：长上下文是当前架构创新最活跃的领域，主流答案是"混合"——既不是纯全局注意力，也不是纯廉价注意力层。

---

## 要点回顾

- 研究方法：架构无法先验推理，最佳学习方式是亲自训练；其次是横向比较大量已发布模型，区分共识与可变项。
- 架构的三重约束：泛化、GPU 效率、训练稳定性；后两者常常主导设计。
- **LayerNorm 位置**：把归一化移出残差流（pre-norm 或非残差 post-norm）是唯一全行业共识；理由是保持残差流干净、梯度直通、抑制梯度尖峰；OPT-350M 是反例。
- **RMSNorm**：去掉均值中心化与偏置，表达力不损，省下大量访存开销（归一化只占 0.17% FLOPs，却可能占 25% 运行时间）；线性层偏置也普遍去掉。
- **激活**：门控是真正重要的轴；ReGLU/GeGLU/SwiGLU 稳定优于非门控；Google 系用 GeGLU，LLaMA 系用 SwiGLU；门控需把 FFN 维度缩到 2/3 以配平参数。
- **并行层**：PaLM 曾报告 15% 加速，但因表达力受损（相当于深度减半）已基本退出，串行仍是主流。
- **位置编码**：RoPE 已成为 2024 年后的绝对主流；核心是把词向量按位置成对旋转，使内积只依赖相对位置；实现上在每次注意力中对 Q、K 施加旋转。
- **超参数**：FFN 倍率 4（门控 2.67，LLaMA 3.5，T5 64 为异类）；头维度乘积等于模型维度；宽深比约 100；词表单语 30k、多语 100k–200k；大部分参数落在一个很宽的平坦最优区间内，系统利用率比精细调参更重要。
- **正则化**：单遍训练几乎不过拟合；weight decay 仍流行，因为它更像优化器干预而非正则化；dropout 已淡出。
- **稳定性**：两处 softmax 是危险区；z-loss 把 $\log Z$ 拉向 0；QK-norm 把 softmax 输入尺度固定，几乎无性能代价；logit soft cap 更硬，安全但损失表达力。
- **部署侧注意力**：解码阶段算术强度退化为 $N/d + 1/B$；MQA 激进省 KV cache 但损失表达力；GQA 用分组共享拿到接近多头的性能与接近 MQA 的成本，成为事实标准。
- **长上下文**：滑窗 + 全局交替（Llama 4、Gemma 4、Olmo 3）或滑窗 + SSM 交替（Qwen 3.5）是当前主流混合方案，也是架构创新最活跃的方向。
