---
title: "第 2 讲 · PyTorch 与 einops"
description: "本讲把“训练大模型”拆成可核算的资源问题：张量存储由元素数和数值精度决定，计算量用 FLOPs 计数；一次训练 step 的 FLOPs 可近似为 。要判断实际速度，需要比较算法的算术强度与硬件的 FLOP/s 和内存带…"
---

# 第 2 讲 · PyTorch 与 einops

> **Stanford CS336 语言模型从零构建 · L02** ｜ 讲者 Stanford Online ｜ 77 分钟 ｜ [原视频](https://www.youtube.com/watch?v=kuYAsz7zspQ)

> **v2 详写版** · 按 20 分钟分段详写，配图覆盖全片；文末附「对照参考笔记补遗（Spring 2025 版）」，记录与 2025 版讲义笔记的核对结果。

## 核心结论

这一讲把"给定固定的算力与内存，训练出尽可能好的模型"落成一套可逐项核算的资源账本：参数、梯度、优化器状态与激活值本质上都是张量，其内存等于元素数乘以每元素字节数，因此 float32、bfloat16、FP8/FP4 的取舍直接左右容量与速度，混合精度训练以 BF16 存参数、激活和梯度、以 FP32 存优化器状态，并交给 PyTorch AMP 在 matmul 这类安全算子处自动转换。计算一侧先分清 FLOPs（计算总量）与 FLOP/s（硬件吞吐），线性层前向是 $2BDK$，反向因要同时求输入梯度与参数梯度恰为前向的两倍，汇总成一次训练 step 的 $6ND$，再用 MFU = 实测吞吐 / 标称峰值衡量算力究竟被用上了多少。要解释真实速度，还须比较算法算术强度（FLOPs / 搬运字节数）与硬件强度（FLOP/s / bytes/s）：前者小于后者即 memory bound、大于则 compute bound，roofline 图据此把逐元素算子与大矩阵乘法分开，也说明了 MFU 为何常停在 0.5 左右。为把这些账目写得可读，einops 用命名维度重写 einsum、reduce、rearrange，免去了 transpose 与 -2/-1 索引的歧义；梯度累积与激活检查点则以拆微批、重算激活降低峰值显存，换来更大的 batch 与更大的模型。

## 开场：Delphi Scaling Suite 结果与课程定位 [00:00 - 01:58]

讲师以一句关于天气的闲聊开场，随即分享上一讲提到的 Marin project 跑完的结果。幻灯片标题是 **Delphi Scaling Suite Results**，画的是一组 iso-flops curve（等算力曲线）：每一条曲线由一批规模较小的模型训练构成，横坐标是 Training tokens，取对数刻度从 $10^9$ 一直延伸到 $10^{14}$，纵坐标是 Paloma macro loss，数值大约落在 2 到 3.5 之间。图例里按 Compute (FLOPs) 给每条曲线配色，能读到 $3\mathrm{e}{+}18$、$9\mathrm{e}{+}18$、$1.8\mathrm{e}{+}19$、$3\mathrm{e}{+}19$、$9\mathrm{e}{+}19$、$1.8\mathrm{e}{+}20$ 等档位，另外还有灰色虚线 Compute-Optimal Frontier（计算最优前沿）、叉号形状的 Forecast（预测点），以及用菱形标出的 $1\mathrm{e}{+}21$、$1\mathrm{e}{+}22$、$1\mathrm{e}{+}23$ 三个外推算力点。图上还有三处误差标注 $\Delta\text{loss}=+0.011$、$\Delta\text{loss}=+0.005$、$\Delta\text{loss}=+0.005$，以及 "FLOP Equivalent to Marin 32B" 和 "Estimated FLOP Equivalent to GPT-5" 两处文字。讲师的解释是：先在这些等算力曲线上找出 compute optimal point（计算最优点），拟合出 scaling law（缩放定律）并预测 loss，再真正把模型跑出来，结果实际 loss 与预测相差在 0.05 以内；他还把曲线外推到 GPT-5 级别的算力，给出对应的 loss，但马上补了一句 "your mileage might vary"，提醒外推结果取决于 scaling laws 具体怎么拟合。

![Delphi Scaling Suite 结果图：Paloma macro loss 对 Training tokens 的多条等算力曲线、Compute-Optimal Frontier 与 Forecast 预测点](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0001_00-00-36.jpg)

紧接着画面切到课程定位，幻灯片逐行写着 "Last lecture: overview, tokenization"、"Today: **resource accounting**"（资源核算）。随后是两句被反复强调的设定："Recall: what's the best model one can train given fixed resources (compute, memory)?" 与 "In other words: maximize (computational) **efficiency**."，也就是说本课程的总目标是在有限资源下训练出尽可能好的模型，等价于最大化计算效率。这里的资源主要指 compute（算力）和 memory（内存／显存），讲师顺带提到有时也包括 data（数据），但 "that's not really going to be a limiting factor for us in this class"，本课中数据并不会成为瓶颈。被黄色高亮的下一行给出了逻辑前提："Precursor: We need to understand the resources (compute, memory) for a given computation?"——在谈优化效率之前，必须先能算清一次给定计算到底消耗多少算力与内存。这三行文字把整讲的走向定死了：上一讲是课程总览与 tokenization，本讲转向更偏 systems 侧的资源核算，而资源核算本身又分成"理解一次计算的特征"和"在此基础上优化"两步。

从与后文的关系看，这一分钟其实埋了两条线索。iso-flops 曲线、Forecast 与 Compute-Optimal Frontier 属于"给定算力如何分配"的问题，呼应本讲后面 FLOPs 计数与训练时长估算；而高亮行里点名的 compute 与 memory，则正好对应接下来的两个 motivating question（一个算时间、一个算容量），也对应后面从 tensor 内存一路推到精度、再到 FLOPs 的整条主线。幻灯片在这一页下方还留了一个 `motivating_questions()` 的函数占位和淡灰色的 "What knowledge to take away:"，说明讲师是刻意用"先抛出你能回答什么问题"的方式来组织内容。易错点在于不要把 $\Delta\text{loss}=+0.005$ 这类标注当成百分比或相对误差，它们是 loss 的绝对差值；也不要把 Forecast 叉号误读成实测点，那是拟合 scaling law 之后给出的预测值，讲师所说"相差 0.05 以内"指的正是预测与实测之间的一致性。外推到 GPT-5 的那一段更像量级演示而非可复现的实验结论，讲师自己就用 "your mileage might vary" 划定了它的可信边界。

## 两个 back-of-envelope 估算：训练时长与最大模型 [01:58 - 04:43]

为了给"课程结束时你应能回答什么"一个直观样例，讲师抛出第一个问题：训练一个 70B（700 亿）参数模型、用 15T（15 万亿）tokens、在 1024 张加速卡上要多久。幻灯片把问题写成 "How long would it take to train a 70B parameter model on 15T tokens on 1024 B100s?"，讲师当场口头更正说这里应当是 H100（"Actually, this should be H100"）。回答的骨架是一个会贯穿全讲的公式，训练总计算量约等于参数量乘以 token 数再乘以 6：

$$
\text{total\_flops}=6\,N\,D=6\times70\times10^{9}\times15\times10^{12}=6.3\times10^{24}
$$

其中 $N$ 是参数量、$D$ 是 token 数，系数 6 来自前向与反向传播的 FLOPs 计数，讲师明说 "there's a formula which we'll talk about"，此处先用结论、后面再解释来源。代入后得到的 $6.3\times10^{24}$ 与幻灯片浅蓝色输出框里的 `total_flops = 6.300e+24` 完全对上。

![motivating_questions() 幻灯片：训练时长与最大模型容量两段代码，以及 total_flops、flops_per_day、days、h100_bytes 的数值输出](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0003_00-03-01.jpg)

幻灯片上的代码把这条链路写得很具体，输出框中还并列给出了 `flops_per_day = 4.377e+22`、`days = 143.9266` 与 `h100_bytes = 80,000,000,000`：

```python
def motivating_questions():
    total_flops = 6 * 70e9 * 15e12
    h100_flop_per_sec = 1979e12 / 2
    flops_per_day = h100_flop_per_sec * mfu * 1024 * 60 * 60 * 24
    days = total_flops / flops_per_day
```

这里的 $1979\mathrm{e}12$ 是查 H100 spec sheet（规格表）得到的标称 FLOP/s，代码先除以 2 得到 `h100_flop_per_sec`；`mfu` 是 Model FLOPs Utilization（模型算力利用率），讲师在讲这段时明确给出取值 0.5；再乘上 1024 张卡与 $60\times60\times24=86400$ 秒/天，就得到硬件每天能提供的有效 FLOPs。于是

$$
\text{days}=\frac{\text{total\_flops}}{\text{flops\_per\_day}}=\frac{6.3\times10^{24}}{4.377\times10^{22}}\approx143.9
$$

即约 143 天，与输出框 `143.9266`、以及讲师口头的 "the number of days is 143" 一致。这条式子的意义不在于记住 143 这个数，而在于示范如何把"模型规模 × 数据量 × 硬件峰值 × 利用率 × 卡数 × 时间"串成一次可计算的估算。

第二个问题转向 memory 而非 compute："What's the largest model that can you can train on 8 H100s using AdamW (naively)?" 幻灯片把 `h100_bytes = 80e9` 一行高亮，对应 H100 的 80 GB HBM（高带宽显存）。每参数需要的字节数写成 `2 + 2 + (4 + 4)`，注释是 `# parameters, gradients, optimizer state`，讲师说 "We'll explain where that comes from"，即这 12 bytes 的拆分留到后面解释。于是最大参数量的估算代码与公式为：

```python
h100_bytes = 80e9
bytes_per_parameter = 2 + 2 + (4 + 4)  # parameters, gradients, optimizer state
num_parameters = (h100_bytes * 8) / bytes_per_parameter
```

$$
N_{\max}=\frac{80\times10^{9}\times8}{12}\approx5.33\times10^{10}
$$

也就是讲师口中的 "about 53 billion"（约 530 亿参数）。幻灯片紧跟着写了一条很重要的免责说明："Caveat: activations are not accounted for (depends on batch size and sequence length)."，以及 "This is a rough back-of-the-envelope calculation."。activations（激活值）没有计入，而它取决于 batch size、sequence length 和 batch length；讲师进一步强调，"the point is not to precisely calculate every single thing, but just get the rough shape of things"——目标不是把每个细节算精确，而是拿到量级和大致形状。这也是本章最容易出错的地方：`naively` 意味着既没算 activations，也没算并行通信、显存碎片与框架开销，53B 只是"权重加优化器状态勉强能塞进 8 张卡"的上界，绝不等于"能高效训练 53B"。

这一页末尾还给出了本讲的知识分类，正好把前后内容串起来："What knowledge to take away:" 下三条分别是 "Mechanics: straightforward (PyTorch semantics)"（机制：很直白，就是 PyTorch 与 tensor 如何工作，讲师说 "There's no magic here"）、被高亮的 "Mindset: resource accounting (remember to do it)"（心态：每写一行代码都要习惯性去想它的性能特征）、以及淡灰的 "Intuitions: get a sense of how resources are spent, no ML magic today"（直觉：资源花在哪里，今天不讲 ML 魔法，留给下一讲的 Tatsu）。最后还露出 `# Memory accounting` 与 `tensors_basics()` 两行占位，直接引出下一章从 tensor 开始的"自底向上"拆解。换句话说，这两个问题分别是 compute 与 memory 两条主线的入口，后面的 float32/bfloat16 精度讨论解释了 `2 + 2 + (4 + 4)` 里那些 2 和 4 从哪来，而 FLOPs 计数则会解释系数 6。

## 张量基础与 float32：内存账的起点 [04:43 - 07:39]

讲师宣布 "let's start bottom up and start building up"，最底层就是 tensor（张量）："tensors are the building block of storing everything"，parameters（参数）、gradients（梯度）、optimizer states（优化器状态）、data（数据）、activations（激活值）"everything essentially is a tensor"。幻灯片上的淡灰 bullet 原文正是 "Tensors are the basic building block for storing everything: parameters, gradients, optimizer state, data, activations."。为了让这件事不那么抽象，幻灯片举了 "Example: parameters of the DeepSeek v3.2 model [DeepSeek-AI+ 2025]"，并附了链接 "[DeepSeek v3.2 model on Hugging Face]"：一个模型拆开看就是一堆各不相同的 tensor，每个都带着自己的 shape（形状）和 precision（精度）。这两个属性恰好就是后面内存账的全部输入——形状决定元素个数，精度决定每个元素占几个字节。

接着幻灯片用 `torch.zeros` 演示 tensor 对向量、矩阵乃至任意 rank 的推广。被黄色高亮的一行是 `x = torch.zeros(4)    # vector`，浅蓝色输出框里显示 `x = 0.000e+0 0.000e+0 0.000e+0 0.000e+0`，即 4 个元素的一维张量；下面淡灰的两行继续构造 `x = torch.zeros(4, 8)    # matrix` 以及更高 rank 的 tensor。讲师的说法是 tensors "subsume vectors, matrices and can be generalize to any number of entities"（原文如此），也就是向量与矩阵只是 rank 为 1 和 2 的特例。紧接着出现 `def tensors_memory():` 与一句 "Almost everything (parameters, gradients, activations, optimizer states) are stored as floating point..."，把话题收束到浮点存储上。需要注意的易错点是：tensor 并不只能存浮点数，也能存整数等其它类型，只是本课几乎只与浮点打交道；另外 `torch.zeros(4)` 这种写法不带 `dtype` 时用的是默认类型，想要别的精度必须显式声明，这一点马上就会成为关键。

![IEEE 754 single-precision 32-bit float 位布局：1 位 sign、8 位 exponent、23 位 fraction，以及 float32 为默认类型的说明](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0006_00-06-39.jpg)

于是进入本讲第一个精度类型 float32。幻灯片在 "[Wikipedia]" 链接下给出 "IEEE 754 single-precision 32-bit float" 的位布局色块图：sign（符号位）占 1 bit，位于第 31 位，示例值为 0；exponent（指数位）占 8 bit，位于第 30 到 23 位，示例位串是 `0 1 1 1 1 1 1 0`；fraction（尾数／小数位）占 23 bit，位于第 22 到 0 位，示例位串是 `0 1 0 0 0...0`；图下方标出的位索引依次是 31、30、23、22、0。讲师把三段的职责讲得很清楚：1 位给符号，8 位指数给出 dynamic range（动态范围），剩下的尾数给出 "variation"，也就是分辨率。图下正文写着 "The float32 data type (also known as fp32 or single precision) is the default."；被黄色高亮的一行解释了名字的来历："Traditionally, in scientific computing, float32 is the baseline; you could use double precision (float64) in cases."——在科学计算年代，别人给你一个 float 你就默认它至少是 single precision，想要更高就用 double precision（float64）。而淡灰的一行 "In deep learning, you can be a lot sloppier." 点出深度学习走了相反方向：连 32 位都嫌多，因为要做的计算并不需要数值模拟那种高精度。

内存怎么算，幻灯片给了两句话："Memory is determined by the (i) number of values and (ii) data type of each value."，以及 "Let's examine memory usage of these tensors."，合起来就是本讲最基础也最重要的公式：

$$
\text{memory}=\text{numel}\times\text{element\_size}
$$

讲师现场构造一个 $4\times8$ 的矩阵，默认类型正是 float32，"if you want something else, you should declare it"；元素数 $4\times8=32$，每个 32 位浮点数占 4 bytes，所以共 $4\times8\times4=128$ bytes。幻灯片上对应的断言代码把每一步都钉死了：

```python
x = torch.zeros(4, 8)
assert x.dtype == torch.float32          # default dtype
assert x.numel() == 4 * 8
assert x.element_size() == 4             # Float is 4 bytes
assert get_memory_usage(x) == 4 * 8 * 4  # 128 bytes
```

这里 `numel()` 返回元素总数、`element_size()` 返回单个元素字节数，两者相乘才是总字节数，混淆这两个量是初学者最典型的错误。与前后内容的关系也很直接：本章建立的 $\text{numel}\times\text{element\_size}$ 正是第 2 章里 `2 + 2 + (4 + 4)` 中每个"2"和"4"的来源，也是下一章讨论"降精度为何既省内存又省时间"的出发点；而 float32 的 4 bytes/元素一旦乘上真实大模型的元素数，就会得到一个惊人的数字。局限在于这套账只算了 tensor 数据本身，不含框架元数据、对齐与碎片，也不含后面会讲的 activations，因此它仍然属于 back-of-envelope 级别的近似。

## 从 2.3 GB 到 float16：为何降精度、又为何不稳 [07:39 - 09:53]

先给一个尺度感。讲师提到 GPT-3 前馈层（feedforward layer）里"一个矩阵"约 2.3 GB，并补了一句这远不是能想象的最大张量。幻灯片上被黄色高亮的一行代码把 2.3 GB 还原成了可核对的算术：

```python
assert get_memory_usage(torch.empty(12288 * 4, 12288)) == 2304 * 1024 * 1024  # 2.3 GB
```

即形状为 $(12288\times4)\times12288$ 的 float32 矩阵，元素数乘 4 bytes 后正好是 $2304\times1024\times1024$ 字节。这个数字把上一章的 128 bytes 一下子放大了七个数量级，也解释了为什么"省内存"在深度学习里不是锦上添花而是刚需。讲师给出的两条理由是：降低 precision 直接减少 storage，同时在更窄的位宽上运算也更快——"operating on 16 bits is going to be faster... let's say twice as fast, but not always, it sort of depends"，大约快一倍但不总是如此，取决于具体操作；更不明显的是 "reducing memory can save time as well"，减少内存占用本身还能间接省时间，这一点要等到后面讲 memory bottleneck 时才会变清楚。这段因此是从"内存账"过渡到"计算与带宽账"的桥。

![IEEE half-precision 16-bit float 位布局：1 位 sign、5 位 exponent、10 位 fraction，以及 float16 创建与 1e-8 下溢代码](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0008_00-09-04.jpg)

最朴素的降精度做法就是"拿掉一半的位"，得到 float16。幻灯片给出 "IEEE half-precision 16-bit float" 位布局：sign 占 1 bit（第 15 位），exponent 占 5 bit（第 14 到 10 位，示例位串 `0 1 1 0 0`），fraction 占 10 bit（第 9 到 0 位，示例位串 `0 1 0 0 0 0 0 0 0 0`），图下位索引为 15、14、10、9、0。正文写 "The float16 data type (also known as fp16 or half precision) cuts down the memory."，黄色高亮行是 `x = torch.zeros(4, 8, dtype=torch.float16)`，随后淡灰的 `assert x.element_size() == 2` 确认每个元素从 4 bytes 降到 2 bytes。把三种格式放在一起对比会更清楚（此处仅作补充对照）：

| 类型 | sign | exponent | fraction | 每元素字节 | 动态范围来源 |
|---|---:|---:|---:|---:|---|
| float32 | 1 | 8 | 23 | 4 | 8 位指数 |
| float16 | 1 | 5 | 10 | 2 | 5 位指数 |
| bfloat16 | 1 | 8 | 7 | 2 | 8 位指数 |

问题出在 exponent 从 8 位砍到 5 位。幻灯片淡灰行先给了结论："However, the dynamic range (especially for small numbers) isn't great."，接着用一段可直接运行的代码演示：

```python
x = torch.tensor([1e-8], dtype=torch.float16)
assert x == 0   # Underflow!
```

浅蓝色输出框里 `x` 显示为 `0.000e+0`，也就是 $10^{-8}$ 这个数在 float16 里直接变成 0，注释 `# Underflow!`（下溢）。讲师解释原因："it's only five bits of exponent compared to eight"，所以既不能表示很大的数，也不能表示很小的数；"If this happens when you train, you can get instability."。他回顾早年用 FP16 训练的经历：会得到 underflow（下溢）、overflow（溢出）乃至 NaNs，"It's pretty challenging"。这里的易错点值得单独点明：$1\mathrm{e}{-}8$ 变成 0 不是"精度不够、四舍五入"，而是这个量级已经超出 5 位指数能表达的范围；相比之下 10 位 fraction 提供的分辨率对深度学习往往够用，真正的短板是动态范围。这一点正是下一章 bfloat16 设计动机的全部依据——既然问题在指数位而不在总位数，那就别再砍指数位。

与前后内容的关系上，本章完成了两次转折：从"tensor 有多大"（2.3 GB）转到"能不能用更少的位存"，又从"能省一半内存"转到"为什么 FP16 训练会崩"。第 2 章那个 `2 + 2 + (4 + 4)` 里的 "2" 至此有了具体候选（float16/bfloat16 每元素 2 bytes），而"FP16 太冒险"的结论会直接写进下一章的 implications on training。局限也很清楚：`element_size() == 2` 只说明单个元素占 2 bytes，不代表整个训练就省一半显存，因为优化器状态、activations 仍在；"twice as fast" 只是讲师给的粗略说法，他本人就加了 "but not always" 的限定。

## bfloat16 与混合精度训练 [09:53 - 13:06]

bfloat16 的出发点是"不牺牲指数位"。幻灯片在 "[Wikipedia]" 与 "bfloat16" 标题下给出位布局：sign 1 bit（第 15 位），exponent 8 bit（第 14 到 7 位，示例位串 `0 1 1 1 1 1 0 0`），fraction 7 bit（第 6 到 0 位，示例位串 `0 1 0 0 0 0 0`），图下位索引为 15、14、7、6、0——总位宽仍是 16，但指数位与 float32 一样是 8 位。正文写明 "Google Brain developed bfloat (brain floating point) in 2018 to address this issue."，讲师也强调它 "developed in actually 2018"。被黄色高亮的那句是整个设计的精髓："bfloat16 uses the same memory as float16 but has the same dynamic range as float32!"，即内存占用与 float16 相同（每元素 2 bytes），动态范围却与 float32 相同。紧随其后的淡灰行 "The only catch is that the resolution is worse, but this matters less for deep learning." 承认代价：尾数只剩 7 位，resolution（分辨率）更差，"there's no free lunch here"。讲师给的直觉是，深度学习里的量本来就 "sloppy and stochastic anyway"，你最需要的是别 overflow、别 underflow，而不是把每个值记得特别精细，所以这个 trade-off 在多数场景很划算。

![bfloat16 位布局：1 位 sign、8 位 exponent、7 位 fraction，说明其动态范围与 float32 相同、内存与 float16 相同](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0009_00-10-16.jpg)

幻灯片用两行代码把 bfloat16 与上一章的 float16 直接对照，并用 `torch.finfo` 把三种类型拉到一起比较：

```python
x = torch.tensor([1e-8], dtype=torch.bfloat16)
assert x != 0   # No underflow!

# Let's compare the dynamic ranges and memory usage of the different data types:
float32_info = torch.finfo(torch.float32)
float16_info = torch.finfo(torch.float16)
bfloat16_info = torch.finfo(torch.bfloat16)
```

同一个 $1\mathrm{e}{-}8$，float16 下是 `assert x == 0  # Underflow!`，换成 bfloat16 就是 `assert x != 0  # No underflow!`；`torch.finfo` 则用来查询各类型的 dynamic range 与内存相关元信息。这一对照把"指数位决定动态范围、尾数位决定分辨率"的抽象说法落到了可验证的断言上，是理解 bfloat16 最省事的路径。

接下来是 "Implications on training:"（对训练的含义）。幻灯片两条 bullet 分别是 "Training with float32 works, but requires lots of memory." 与被黄色高亮的 "Training with fp8, float16 and even bfloat16 is risky, and you can get instability."。讲师展开说：用 float32 训练"absolutely"没问题，小模型、不想操心就直接用 float32 挺好，但它每个 float 要 4 bytes 内存，"that can take up a lot of memory"；float16 则 "too risky"；BF16 是 "sort of a sweet spot"，但他立刻补了一句 "even BF16 can be kind of risky as well"。也就是说，把低精度当成万能解药是错的——bfloat16 解决了动态范围，却没有解决所有数值稳定性问题。这一条与第 2 章 `2 + 2 + (4 + 4)` 的呼应也在这里显形：既然纯低精度有风险，实际训练就得让不同部分用不同精度。

于是引出 mixed precision training（混合精度训练），幻灯片标注了出处 "[Micikevicius+ 2017]"，两条 bullet 是 "Use bf16 for parameters, activations, and gradients" 与 "Use fp32 for optimizer states"（后者正是 12 bytes 里那两个 4 的来源，讲师说 "We'll discuss a little bit more about that later"）。黄色高亮行写 "Pytorch has an automatic mixed precision (AMP) library. [docs]"，淡灰行补充 "Tries to cast things into bf16 when safe (matmuls, not exp)."。讲师的定义是：一部分计算用一种精度、另一部分计算用另一种精度，"some of the computations use some precision and for some other computations use other precision"；一般规则是参数、activations、gradients 用 BF16，优化器状态用 FP32。要启用它，PyTorch 提供 AMP 库，把代码包一层，库会自动在安全的地方把张量 cast 成 BF16——"matmuls are generally safe"，而 exponentiation（指数运算）这类操作 "will try to leave things as FP32"。这解释了为什么 AMP 是"自动"的：它逐算子判断，而不是给整个模型设一个 dtype。易错点在于别把混合精度理解成"全模型 BF16"，也别把 "matmuls, not exp" 当成完整规则清单——讲师明说 "We're not going to talk too much about this"，这里只给了典型例子；AMP 的自动 cast 是启发式的，遇到自定义算子未必按预期工作。与前后关系上，本章把精度从"选一个 dtype"推进到"按角色分配 dtype"，既回收了第 2 章 12 bytes 的伏笔，也为下一章"还能不能再往下压到 FP8、FP4"提供了动机。

## 更低精度与训练／推理之别：FP8、FP4 与 block scaling [13:06 - 17:00]

讲师说 "we could do BF16 I think is probably where this class will end. But if you're feeling very adventurous, you can go farther"，于是先把精度再往下推。FP8 部分幻灯片写着 "In 2022, fp8 was standardized, motivated by machine learning workloads primer."，并给出 "[Micikevicius+ 2022]" 引用，讲师口头说它 "introduced four years ago"、已经标准化，而且"有两个版本"，取决于你需要更多 dynamic range 还是更多 resolution。图上的两个位模式示例把这件事具体化了：`FP8 E4M3` 的示例位串对应数值 `= 0.40625`，`FP8 E5M2` 的示例位串对应 `= 0.375`；紧接着一行被明确写出："H100s support two variants of FP8: E4M3 (range [-448, 448]) and E5M2 ([-57344, 57344])."。也就是说 E4M3 用 4 位指数、3 位尾数，分辨率较好但范围只有 $\pm448$；E5M2 用 5 位指数、2 位尾数，范围扩到 $\pm57344$ 但分辨率更差。这两个数值范围和两种格式的位分配都是转写里完全没念到、只出现在画面上的信息。讲师还提到 Nvidia 的 Transformer Engine 支持 FP8，"we're not going to talk about that"，即不展开实现。

![FP8 的 E4M3 与 E5M2 位模式示例及数值、H100 支持的两种 FP8 变体范围，以及 fp4/nvfp4 小节开头](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0012_00-13-54.jpg)

再往下是 FP4。幻灯片标题 "fp4"，正文 "In 2025, NVIDIA developed nvfp4"，高亮行 "Only 4 bits per value!"，讲师的说法是 "last year Nvidia developed NVFP4"、"there's only four bits per value"，并打趣说 4 个 bit 少到"能把所有取值写在一行里"，介于 $-6$ 和 $6$ 之间。幻灯片把这 16 个可表示值完整列了出来：

```
Values: -6, -4, -3, -2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2, 3, 4, 6
```

注意这组值并不是等距的，靠近 0 的地方密、两端疏，属于非均匀量化。下一行 "Use a separate scale factor per block"（每个 block 用单独的缩放因子）是关键补充：如果只朴素地使用这 16 个值，"you're not going to be able to train very very well"，所以每个 block 可以整体按 scale factor 上下缩放，从而表示出更多数值。幻灯片还写了 "Nemotron 3 Super was trained in NVFP4 [NVIDIA+ 2025]"（讲师口头念作 "NeMo-3 Super"、说 "this year"，与画面上的 2025 对应），作为 FP4 训练真实模型的例子；被黄色高亮的收尾句是 "Some of this is done in NVIDIA libraries outside of user control."，讲师也强调"不是你创建一个 tensor 就能叫它 FP4"，很多是 NVIDIA 软件栈在底层完成的。

![fp4 小节：NVFP4 的 16 个可表示数值、per-block scale factor、Nemotron 3 Super 训练说明与 NVIDIA 库的限制](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0013_00-15-07.jpg)

15:17 之后进入问答，正好把 "per-block scale factor" 讲透。有学生追问 block 缩放的细节，讲师解释：假设有一个 block，块内每个值仍能在 4 bits 里变化（也就是在那 16 个取值里选），此外整块的值还可以按 scale factor 一起放大或缩小，而 scale factor 本身也占若干 bits；于是"从单个值看，你其实能得到超过 4 bits 的动态范围"，但代价是 "you can't have this value be way over here and the next neighboring value way down"——不能让这个值极大、紧挨着的下一个值极小。转写里学生那句 "The ratios of all of them will still be in FP4" 与 "scaling with the maximum value is to have more specificity" 正是对这一限制的概括：块内比值仍受 FP4 约束。这是一个很实在的局限：block scaling 换来的是整块的量级自由度，不是每个元素独立的动态范围。

随后有学生问能不能把 BF16 一路推到 1 bit。讲师借机区分 training（训练）与 inference（推理）：低比特的 quantization（量化，他说后面会讲）通常是先用 BF16 之类的精度把模型训练好，再量化到 1 或 2 bits，"that is much easier than training a one bit language model"；而从头训练 1-bit 语言模型 "I don't think anyone has trained anything credible there"（也许可能，但没人做出可信结果）。这与上一章"fp8、float16 甚至 bfloat16 训练都 risky"形成呼应：精度越低，训练侧的难度增长远快于推理侧。整段被讲师自己称作 "a bit of a long digression"，但目的明确——让人体会精度里的 intricacies（微妙之处）。易错点有三处：别把 E4M3 的 $[-448,448]$ 与 E5M2 的 $[-57344,57344]$ 记反；别把 FP4 的 16 个值当成均匀刻度；别把 Nemotron/NVFP4 这类能力理解成用户可随意调用的 dtype。16:14 前后讲师问 "any questions before I move on?"，17:00 一句 "Let us go on then" 结束插曲，回到内存与计算的主线。

## 从内存到计算：GPU 迁移与 einsum 的动机 [17:00 - 19:21]

回到主线时讲师先做了一句收束：tensor 的内存账 "pretty simple"，就是 $\text{memory}=\text{numel}\times\text{element\_size}$。然后指出一个很实际的默认行为：在 PyTorch 里创建的 tensor 默认在 CPU 上，"if you want things to go fast, you want to move them to GPU"，否则拿不到加速。幻灯片上的代码把这一点写成了可检查的断言：

```python
def tensors_on_gpus():
    # By default, tensors are stored in CPU memory.
    x = torch.zeros(32, 32)
    assert x.device == torch.device("cpu")
```

后面一行淡灰文字是 "However, in order to take advantage of the massive parallelism of GPUs, we need to move them [to GPU] memory."。讲师还专门交代了一个环境上的局限："They were executed on my laptop which means that I don't have GPU. So some of the code I'll just show but not execute."——幻灯片是在他自己的笔记本上跑的、没有 GPU，所以部分代码只展示不实际执行。这既是易错点也是读笔记时该记住的前提：不要把"只展示"的代码当成已验证的输出，尤其涉及 `device` 与计时的片段。他最后又提醒一句 "just remember to do that otherwise you won't get your speed ups"。

配套的示意图把"为什么要搬"画得很直观：左侧橙色方框是 CPU，里面一个绿色 "CPU" 块和一条蓝色 "RAM"；中间是标着 "PCI BUS" 的双向箭头；右侧是 GPU 大方框，上方排着 6 个绿色 "streaming multiprocessor"（2 行 × 3 列），底部是一条蓝色 "DRAM" 长条。直觉是数据得先从主机 RAM 经 PCI BUS 搬到 GPU 的 DRAM，计算才在那些 streaming multiprocessor 上发生；搬运本身有成本，这正是后面讨论 memory bound、以及"减少内存也能省时间"的物理基础。幻灯片随后给出探测 GPU 的代码，黄色高亮行是 "Let's first see if we have any GPUs."：

```python
# Let's first see if we have any GPUs.
if not torch.cuda.is_available():
    return
num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    properties = torch.cuda.get_device_properties(i)
```

`torch.cuda.is_available()` 先判断有没有可用 GPU，`device_count()` 拿到数量，再逐卡用 `get_device_properties(i)` 查询属性——这与第 2 章"查 H100 spec sheet"的做法形成呼应，只不过从纸面规格变成了运行时查询。

18:02 讲师正式切换话题："we talked about memory of tensors which is very straightforward. Now let's talk about computing with tensors."。但在开始做 FLOPs 计数之前，他先 "take a little bit of a digression to talk about einsums"，并现场问有多少人用过 einsum（约三分之二举手）。动机页的代码把痛点摆得很清楚：

```python
def einops_motivation():
    # Easy to mess up the dimensions (what is -2, -1?)...
    # Traditional PyTorch code:
    x = torch.ones(2, 2, 3)   # batch seq hidden
    y = torch.ones(2, 2, 3)   # batch seq hidden
    z = x @ y.transpose(-2, -1)   # batch seq seq
```

被黄色高亮的正是 `z = x @ y.transpose(-2, -1)   # batch seq seq` 这一行。讲师说盯着 `x` 和 `y`、再看到 `transpose(-2, -1)`，你得自己在脑子里推断 `-2`、`-1` 到底指哪两个维度，"I find it very confusing"，很容易搞错；所以动机就是"用带名字的变量而不是索引"（using variable names rather than indices）。注意这里三个维度都被注释成了 `batch seq hidden`，输出变成 `batch seq seq`——负索引的语义会随张量 rank 改变而漂移，这正是命名维度要消灭的歧义。

![einops_motivation() 幻灯片：torch.ones 构造 batch/seq/hidden 张量与 y.transpose(-2, -1) 的传统写法](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0016_00-18-44.jpg)

随后给出 einops 的定义与边界。幻灯片淡灰行写着 "It is inspired by Einstein summation notation (Einstein, 1916)."、"[Einops tutorial]"，以及三个函数占位 `einops_einsum()`、`einops_reduce()`、`einops_rearrange()`。讲师的说法是 "einsums is a library for manipulating tensors where the dimensions are named"，即 einops（此处他混用 einsums/einops 指这套工具）是一个用命名维度来操作张量的库，灵感来自 Einstein summation notation（爱因斯坦求和约定）；他明说 "I'm just going to cover some of the basics"，并给了 tutorial 链接让有兴趣的人自行深入。本段在 19:16 处停在 "basically the way to think about einsum is a generalized matrix..."，也就是把 einsum 理解为一种带维度记账（bookkeeping）的广义矩阵乘法，具体的 pattern 写法留到下一段展开。与前后内容的关系上，这一章完成了从"张量占多少内存"到"张量之间怎么算"的视角切换：只有先把运算写成可核对维度关系的形式，后面才能干净地数出线性层的 $2BDK$ 之类的 FLOPs。易错点也在这里收尾：别把命名维度当成不需要学语法的银弹，pattern 本身有约定；也别因为本段只讲到动机与定义，就凭空补进 `einsum(...)` 的具体调用形式——转写在此明确截断，那部分内容属于下一段。

## 用命名维度重写矩阵乘法：einsum 的入门 [19:21 - 20:48]

承接本讲前面讨论过的传统 PyTorch 写法——`-2`、`-1` 这类轴编号要靠读者自己在脑子里推断——这一段把 einsum（Einstein summation，爱因斯坦求和约定）作为「带有良好记账（good bookkeeping）的广义矩阵乘法」正式引入。核心直觉是：与其记住「倒数第二维是行、最后一维是列」，不如直接给每一根轴起一个语义名字，让名字本身描述运算。讲课者先用一个最朴素的例子热身：`x` 是 3×4 矩阵，`y` 是 4×3 矩阵，做一次 matmul（矩阵乘法）得到 3×3 结果。这个例子简单到「pretty easy to understand」，但正好把 einsum 的三条规则摊开讲清楚：给输入维度命名、在箭头右侧写出输出维度、凡是没有出现在输出里的维度一律被求和掉（gets summed out）。这种「用名字代替下标」的思路，正是整个 einops 库想灌输的思维方式。

幻灯片 `kf_0017`（19:57）把这段写成函数 `einops_einsum()`，第一行说明文字是 "Einsum is generalized matrix multiplication with good bookkeeping."。旧写法与新写法并排给出（行号 220–230）：

```python
def einops_einsum():
    # Einsum is generalized matrix multiplication with good bookkeeping.
    x = torch.ones(3, 4)   # seq1 hidden
    y = torch.ones(4, 3)   # hidden seq2

    # Old way
    z = x @ y              # seq1 seq2

    # New (einops) way
    z = einsum(x, y, "seq1 hidden, hidden seq2 -> seq1 seq2")
```

黄色高亮的第 230 行正是 `z = einsum(x, y, "seq1 hidden, hidden seq2 -> seq1 seq2")`。这里 `seq1` 是 `x` 的行维、`hidden` 是 `x` 的列维同时也是 `y` 的行维、`seq2` 是 `y` 的列维；pattern 字符串里箭头左侧用逗号分隔两个输入张量，箭头右侧是输出张量的维度顺序。具体数字是 `x` 为 `3×4`、`y` 为 `4×3`、被收缩的 `hidden=4`、输出 `z` 为 `3×3`。右侧浅蓝色面板还画了由 1 与 4 填充的小矩阵网格：成行的 `1 1 1` 对应全 1 输入，成行的 `4 4 4` 直观对应沿 `hidden=4` 累加 4 个 `1×1` 之后得到的 4。

计算机制被讲课者在 [20:13]–[20:41] 拆成显式枚举：对 `seq1`、`hidden`、`seq2` 三个指标的所有可能取值做枚举，用 `seq1, hidden` 去索引 `X`、用 `hidden, seq2` 去索引 `Y`，把两个标量相乘，再累加（accumulate）到结果 $Z_{\text{seq1},\text{seq2}}$。写成数学式就是

$$Z_{\text{seq1},\text{seq2}}=\sum_{\text{hidden}} X_{\text{seq1},\text{hidden}}\,Y_{\text{hidden},\text{seq2}}$$

![用 seq1 hidden 与 hidden seq2 命名维度重写的 einsum 代码，第 230 行黄色高亮给出 pattern 字符串 seq1 hidden, hidden seq2 -> seq1 seq2，右侧浅蓝面板画出由 1 与 4 组成的矩阵网格示意](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0017_00-19-57.jpg)

「未在输出中命名的维度被求和掉」是 einsum 最关键的约定，后面 batch 版、reduce、rearrange 全都复用它。易错点至少两类：一是 pattern 里维度的**顺序**必须与张量实际轴顺序严格一致，`x` 是 `(seq1, hidden)` 就不能写成 `(hidden, seq1)`，否则不会报错但语义全错；二是同名维度在不同输入里长度必须相等（`hidden` 在 `x` 里是 4、在 `y` 里也必须是 4），否则无法对齐。讲课者在 [20:43] 反问 "Come on, this is much easier than this, right?"，其实是在给下一章铺垫：这个例子太简单还显不出 einsum 的价值，马上要换「更麻烦的（more complicated）」例子逼出命名维度的真正优势。局限也很明确：einsum 只是表达层的糖，不改变底层浮点运算次数，讲课者稍后在 reduce 的问答里会明确说它 "reduces to the same type of primitive operations ... just sugar"，没有速度收益。

补充一点命名上的直觉：讲课者在 [19:41] 是把 `X` 的两个维度逐一起名的——「the row and the column which I'm going to name seq1 and hidden」，再把 `Y` 的行列命名为 `hidden` 与 `seq2`；[20:00] 又说结果「a tensor or here a matrix where the dimensions are indexed by seq one and seq two」。也就是说 `seq1`、`hidden`、`seq2` 在这里扮演的是循环下标（loop index）的角色，而 pattern 字符串相当于把三重循环的求和范围一次性声明清楚：[20:13] 明确说要「enumerate over all possible values of all the variables that occur here」，即遍历 `seq1`、`hidden`、`seq2` 的全部组合，再 [20:28]「index into X, index into Y, multiply them, and dump them accumulate them into the result Z sub seq one seq two」。这种写法把原先藏在 `@` 运算符里的隐式约定全部外显成可读的字符串，因此讲课者才在 [19:21] 用「multiplication with good bookkeeping」来形容它——记账清楚，就不必再靠记忆去还原哪根轴在哪。此外 [19:33] 他评价旧写法「this is actually pretty nice. It's pretty easy to understand」，说明本例的目的不是批判 matmul，而是用一个人人都能心算的例子先把三条规则立住，再在下一章用三维张量把旧写法的代价放大。

## 批量维度、转置与省略号：einsum 的进阶 [20:48 - 22:49]

讲课者把维度从 2 维升到 3 维来制造对比：两个张量都是 `2×3×4`（`torch.ones(2, 3, 4)`），注释分别写作 `# batch seq1 hidden` 与 `# batch seq2 hidden`。旧写法必须显式转置最后两维 `z = x @ y.transpose(-2, -1)`，并依赖 `@`（matmul）会「隐式地把非 matmul 维度当作 batch 维广播」这一隐含性质，所以读者得自己推理 `-2`、`-1` 落在哪根轴上、转置之后 batch 维会不会被错误对齐。讲课者坦言 [21:24] "you have to kind of reason about this a bit"、[21:46] "I always get confused by transposes"，这份心智负担正是 einops 想消除的东西。

`kf_0018`（21:10）与 `kf_0019`（22:22）两帧把完整代码摊开（行号 233–245）：

```python
x = torch.ones(2, 3, 4)   # batch seq1 hidden
y = torch.ones(2, 3, 4)   # batch seq2 hidden

# Old way
z = x @ y.transpose(-2, -1)   # batch seq1 seq2

# New (einops) way
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")

# Dimensions that are not named in the output are summed over.

# Or can use `...` to represent broadcasting over any number of dimensions
z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")
```

`kf_0018` 黄色高亮的是旧写法第 238 行 `z = x @ y.transpose(-2, -1)`，`kf_0019` 黄色高亮的是省略号版第 245 行，并且该帧紧跟着一行独立说明 "Dimensions that are not named in the output are summed over."，把上一章的约定又写了一遍——这里被求和掉的仍是 `hidden`。具体数字：`batch=2`、`seq1=3`、`seq2=3`、`hidden=4`，输出形状 `2×3×3`。

![带省略号 ... 的 einsum 写法与未在输出中命名的维度会被求和的说明，第 245 行黄色高亮为 ... seq1 hidden, ... seq2 hidden -> ... seq1 seq2](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0019_00-22-22.jpg)

最关键的观察是 [21:38] "there's no transpose because I've done the transpose by the naming"：因为 `y` 被命名为 `batch seq2 hidden` 而不是 `batch hidden seq2`，`hidden` 自然落在第二个输入的内侧，求和约定自动完成内积对齐，转置被「命名」吸收掉了；讲课者还补了一句，若写成 `hidden seq2` 那就又不需要转置（"If I had hidden and seq2, then it would be a no transpose"），可见命名顺序本身就是运算定义的一部分。[21:59] 起引入 `...`（ellipsis，省略号）：把 `batch` 换成 `...` 后，即便输入是 rank 10 的张量、前面挂着 8 个不同的 batching 维度，也无需逐一枚举，"I can just write dot dot dot without enumerating all of them"。这在语言建模里很常见，因为可能同时有 batch 维、sequence 维、head 维，而你想对它们全部做同一个矩阵操作；用 `...` 就能写出 modular code（模块化代码），甚至 [22:36] "without worrying about ... even the shape of the tensor that comes in"，连传进来的张量具体形状都不必知道。与前后内容的关系：这一章把 einsum 从「好看的记号」升级成「能写通用模块的工具」，[22:49] "that's einsum, which is in the einops library" 随即收束话题转入同库的 `reduce`。易错点与局限：`...` 只覆盖前导的、且在所有输入里形状一致的广播维度，不能拿去匹配中间的轴，而且它必须在所有输入和输出的同一位置出现，否则广播语义会变；另外它同样只是表达层，不改变运算量，讲课者在稍后回答学生「有没有加速」时确认 "it should be the same"。

再看旧写法为什么「要动脑」：[21:05]–[21:19] 讲课者把 `z = x @ y.transpose(-2, -1)` 拆成两步解释——先「transposing these last two」，再「because at implicitly sort of batches the dimensions that are not the matmul dimensions, um then you get the answer」。也就是说 `@` 会把最前面那根 `batch` 轴自动当成广播维，而把最后两维留给矩阵乘；这个「隐式批处理」并不是所有人都记得住的规则，一旦张量多几根前导轴就极易错位。einops 版则把 `batch` 明明白白写进 pattern，广播与否一眼可见。关于省略号的适用面，[22:20] 给了语言建模里的具体场景：可能同时存在 batch dimension、sequence dimension、head dimension，而你想「do like this matrix operation for all of them」，又「might not want to have to worry about this」；[22:36] 进一步说可以写出「modular code」而不必过问输入张量的形状。这也解释了为什么 [20:52] 讲课者说这是「the example I showed before」——这个 2×3×4 的例子在第一段讲 PyTorch 张量操作时已经出现过，此处用 einops 重做一遍，正是为了前后对照；[22:49]「that's einsum, which is in the einops library」一句则把话题正式收束，[22:55]「there's a reduce」顺势开启下一块积木。

## reduce：把 sum、mean、max、min 统一成一次命名归约 [22:49 - 24:03]

`reduce` 是 einops 里对**单个**张量做归约的接口，讲课者把它定义为「sum、mean、max、min 的泛化」。[23:18] 之前的动机和前面完全一致：PyTorch 写 `x.sum(dim=-1)` 时，`dim=-1` 只告诉你「最后一维」，不告诉你那根轴语义上是什么，讲课者直说 "I don't like this notation"。幻灯片 `kf_0020`（23:35）里函数名为 `einops_reduce():`，说明文字是 "You can reduce a single tensor via some operation (e.g., sum, mean, max, min)."。reduce 的解法是复用 einsum 的约定：出现在输入 pattern 里、却没出现在输出 pattern 里的维度，就是要被归约掉的维度；归约方式不再靠轴编号，而是用一个单独的字符串参数指定。

`kf_0020` 的代码（行号 248–256）与数值示意如下：

```python
def einops_reduce():
    # You can reduce a single tensor via some operation (e.g., sum, mean, max, min).
    x = torch.ones(2, 3, 4)   # batch seq hidden

    # Old way
    y = x.sum(dim=-1)

    # New (einops) way
    y = reduce(x, "... hidden -> ...", "sum")
```

黄色高亮的第 256 行是 `y = reduce(x, "... hidden -> ...", "sum")`。具体数字：输入 `x` 形状 `2×3×4`，即 `batch=2`、`seq=3`、`hidden=4`；沿最后一维求和后输出形状 `2×3`，`...` 代表前两个 batching 维度被原样保留。右侧浅蓝色面板同样画了数值网格：`x` 是成行的 `1 1 1 1`（4 个 1），`y` 变成成行的 `4 4 4`——因为 `hidden=4` 个 1 相加正好是 4，和上一章 einsum 里 `4 4 4` 的来历相同，只是这次来自加法归约而非乘积累加。第三个参数 `"sum"` 指定 aggregation / reduction operation（聚合/归约操作），[23:40] 明确说 "you can replace it with mean or max or min"。

![einops_reduce 函数代码：旧写法 x.sum 沿 dim=-1 与新写法 reduce 加 sum 聚合，第 256 行黄色高亮，右侧面板画出 2×3×4 全 1 张量沿 hidden 归约成成行的 4 4 4](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0020_00-23-35.jpg)

与前一章的关系：reduce 复用了 einsum「输出里没命名的维度被求和掉」的约定，只是把它从「两个张量的内积」退化成「一个张量的聚合」，所以 pattern 从 `"batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2"` 简化为 `"... hidden -> ..."`。与后一章的关系：reduce 处理的是「维度消失」，马上要讲的 rearrange 处理的是「维度拆分与合并」，两者构成 einops 里除 einsum 之外另外两块积木；[24:03] "the final thing I'll talk about is rearrange" 正是转折。易错点：若把 pattern 写成 `"... hidden -> ... hidden"`，`hidden` 仍出现在输出里，就完全不发生归约；另外归约类型必须由第三个参数单独给出（`"sum"`、`"mean"`、`"max"`、`"min"`），不能塞进 pattern 字符串里。局限有两层：一是 [23:50] 学生追问 "Is there some speed up to this?"，回答是 "It basically reduces to the same type of primitive operations. You can think about it as just sugar ... So, it should be the same"，即没有速度收益；二是 reduce 只沿被消除的轴做一次聚合，无法表达跨轴的复杂组合逻辑，那种需求仍要回到 einsum。

把归约的语义再落细一点：[23:03]–[23:17] 讲课者先描述旧写法想做的事——「you want to sum according to dimension of dim minus one, okay, which means sum along the last dimension here」，也就是把 `2×3×4` 张量的最后一根轴压掉；[23:25]–[23:34] 再对应到 einops 的解释：「there's some batching dimensions. In this case, it would be this first these first two」，而 `hidden`「doesn't appear on the right side, which means that it gets summed」。可见 `"... hidden -> ..."` 里的 `...` 承担的是「原样保留前导维度」，`hidden` 承担的是「被消除的轴」，二者分工与第二章 `...` 的广播语义一脉相承；[23:40] 又补了一句「here I put sum, which means that the aggregation reduction operation is sum」。至于 [23:47] 那位学生追问的加速问题，讲课者的完整回答是「This basically reduces to the same type of primitive operations. You can think about it as just sugar. So, it should be the same」——关键词是 primitive operations（底层原语操作）与 sugar（语法糖）：reduce 只是把同一个 `sum`/`mean`/`max`/`min` 调用换了个更可读的写法，底层 kernel 不变，因此既不会更快也不会更慢。这个结论对整段 einops 都成立，也是理解「为什么值得改写法却不指望提速」的关键。

## rearrange：拆分与合并被压平的维度 [24:03 - 27:26]

讲课者称 rearrange（重排）是 "a pretty powerful tool"，并预告 "I think it will come up in the assignment once"（作业里会用一次）。它解决的问题很具体：有时一个维度其实代表两个维度（"a dimension that actually represents two dimensions"），原因往往是曾经把矩阵 flatten（压平）过，现在想 unflatten 回来、只对其中的一个子维度做运算，做完再 flatten 回去。这正是多头注意力里 `heads × hidden1` 被压成单一 `total_hidden` 的典型形态，也是本讲把它安排在 einops 最后一块的原因——它把前两块积木串成完整闭环。

`kf_0021`（24:47）的函数是 `einops_rearrange():`，开头两行说明 "Sometimes, a dimension represents two dimensions ...and you want to operate on one of them."，右侧浅蓝色面板画出 `x`（3 行 8 列的 1）与 `w`（4 行 3 列的 1）的数值网格。`kf_0022`（26:00）给出完整代码（行号 263–274）：

```python
x = torch.ones(3, 8)   # seq total_hidden
# ...where total_hidden is a flattened representation of heads * hidden1
w = torch.ones(4, 4)   # hidden1 hidden2

# Break up `total_hidden` into two dimensions (`heads` and `hidden1`)
x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)

# Perform the transformation by `w`
x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")

# Combine `heads` and `hidden2` back together
x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")
```

具体数字：`x` 是 `3×8`，`total_hidden=8` 是 `heads * hidden1` 压平的结果；显式指定 `heads=2`，于是 `hidden1 = 8 / 2 = 4`；`w` 是 `4×4`，标注 `# hidden1 hidden2`，即 `hidden1=4`、`hidden2=4`。[25:17] 特别强调 8 的分解不唯一——"Could be 2 by 4, 4 by 2"——所以必须用关键字参数 `heads=2` 把歧义定死。黄色高亮行分别是 `kf_0021` 第 265 行 `w = torch.ones(4, 4)  # hidden1 hidden2` 与 `kf_0022` 第 271 行 `x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")`。

![einops_rearrange 开头：x = torch.ones 3, 8 标注 seq total_hidden，说明 total_hidden 是 heads 乘 hidden1 的压平表示，右侧浅蓝面板画出 x 与 w 的数值网格](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0021_00-24-47.jpg)

![rearrange 拆分与合并的完整代码：heads=2 把 heads 与 hidden1 拆成两维，再用 einsum 与 w 做变换，最后把 heads 与 hidden2 合并回一维，第 271 行黄色高亮 einsum](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0022_00-26-00.jpg)

拆分后 `x` 变成 `3×2×4`（`... heads hidden1`），于是可跟 `w` 做一次标准 matmul：pattern `"... hidden1, hidden1 hidden2 -> ... hidden2"` 里 `hidden1` 未出现在输出中被求和掉，结果形状 `3×2×4`（`... heads hidden2`），最后用括号 `(heads hidden2)` 把两维重新组合回一根轴。括号在 einops 的 pattern 里就是「这几维被合并成一维」的记号，正好与 `...` 的广播记号互补。[26:31] 有学生追问二维压成一维时是 row-major 还是 column-major，讲课者回答 "the order you do it is specified in the order here"——由 pattern 里维度名字的排列顺序决定，而非内存布局隐式决定，这是最容易踩的坑：把 `(heads hidden1)` 写成 `(hidden1 heads)` 会得到完全不同的元素排列。与前后内容的关系：这一章把 einsum（内积对齐）与 reduce（维度消除）组合起来完成「拆—算—合」闭环，[27:01] 讲课者总结说一旦习惯 einops 就会 "think in a different way"，所有 transpose 与 reduction 都变得 "more fluid"，只是 "takes a bit of time to get used to"；[27:19] "we're going to use einops ... we'll see it a little bit later" 收束 einops 话题，[27:26] "let's return back to resource accounting question" 把课程拉回资源核算主线。局限：rearrange 本身只是重排、不产生运算，真正消耗计算量的仍是中间那次 einsum；并且被压平的维度必须能被指定因子整除（8 能被 `heads=2` 整除），否则会直接报错。

把压平与还原的动机再说透：[24:13]–[24:29] 讲课者解释「the reason this happens is that sometimes you have a matrix and you flatten it ... And then you want to maybe unflatten and flatten」，所以 rearrange 的典型用法就是「flatten → 拆分 → 运算 → 合并」这条流水线。[24:31]–[24:53] 他把 `3×8` 具体化成「this dimension eight actually represents like a 2 by 4 matrix. So, I want to multiply that 2 by 4 matrix by this 4 by 4 matrix」——注意这里的 `2×4` 与 `4×4` 正好对应后面的 `heads=2`、`hidden1=4`，以及 `w` 的 `hidden1 hidden2 = 4×4`，矩阵乘的维度在拆分之后才对得上。[25:04]–[25:21] 讲括号的作用：「here I use parentheses to say that this H actually represents the product of heads and hidden one, um where I have to obviously there's multiple ways to decompose this」；[25:38] 用「before X looked like this, and now X looks like this」点出形状从 `3×8` 变成 `3×2×4`。[25:50]–[26:11] 则把中间的 einsum 归类为「a standard matmul」，并说明算完「you can rearrange it back」，[26:17]「look at two dimensions, and then you can group them into one dimension」正是合并那一步的口述版。[26:57] 他坦承「it takes a bit of time to get it used to, but it's well worth it」，因为一旦习惯，「all the transposes and reductions ... becomes more fluid」，只是需要改用这些「more bespoke primitives」（更定制化的原语）来思考。

## FLOP 与 FLOP/s：给计算量定标 [27:26 - 30:49]

回到 [27:26] 的 "resource accounting question"：上一段已算过张量占多少 memory，现在要问 "how much compute do they take"（它们消耗多少计算）。讲课者给出的度量是 flops，并把 FLOP 定义为 "a floating point operation"，且假设它是一次基本运算如 addition 或 multiplication。`kf_0024`（28:25）里函数 `tensor_operations_flops():` 下的原文是 "Having gone through all the operations, let us examine their computational cost." 以及 "A floating-point operation (FLOP) is a basic operation like addition (x + y) or multiplication (x y)."。[27:50] 补充说 GPU 还能做别的事，但这些是 "bread and butter"、会吃掉绝大部分时间，所以其它操作一概忽略。这一章建立的是「计算量」与「硬件速度」的二分，它既是下一章推导线性层 FLOPs 的前提，也是最后一章 MFU 分母的来源。

最容易被绕进去的是同音的两个缩写，幻灯片标题直接写 "Two terribly confusing acronyms (pronounced the same!)"：`FLOPs`（小写 s）指 floating-point operations，是 "measure of computation done"，衡量「做了多少计算」；`FLOP/s`（也常被写成大写 `FLOPS`）指 floating-point operations per second，"which is used to measure hardware"，衡量硬件每秒吞吐。讲课者说有些地方非常讨厌地把后者也写成大写 `FLOPS`，所以他一律写 `/s` 以免混淆；[28:38] 举例说看到 H100 标着 989 teraflops 时那是指每秒，而说 GPT-3 用了 `3.14e23` flops 时那是指前者。`kf_0025`（29:43）的 "Intuitions" 小节给出的具体数字是 "Training GPT-3 (2020) took 3.14e23 FLOPs." 与 "Training GPT-4 (2023) is speculated to take 2e25 FLOPs."，黄色高亮行则是 "H100 has a peak performance of 1979 teraFLOP/s with sparsity, 50% without"，对应代码 `h100_flop_per_sec = 1979e12 / 2`。幻灯片上还留着数量级估算式 `8 H100s for 2 weeks: total_flops = 8 * (60 * 60 * 24 * 7) * h100_flop_per_sec`；按转写 [30:22] 讲课者口头上把它改按「一周」估算，得到约 `5e21`（[30:33] "about 5e 21"）。

$$\text{FLOP/s}_{\text{H100,dense}}=\frac{1979\times10^{12}}{2}\approx 9.895\times10^{14}$$

![Intuitions 小节：GPT-3 训练耗 3.14e23 FLOPs、GPT-4 推测 2e25 FLOPs，黄色高亮行给出 H100 峰值 1979 teraFLOP/s with sparsity, 50% without 以及 h100_flop_per_sec = 1979e12 / 2](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0025_00-29-43.jpg)

这里最值得记的坑是 spec sheet（规格表）的脚注：[29:27] 说 BF16 下标称 `1979` TFLOP/s，可实测对不上，细读小字才发现 "this is with sparsity, so sparse matrix, and for dense it's over two"，所以 "all you always have to take these numbers divide by two"。也就是说密集（dense）矩阵乘的可用峰值只有 `1979 / 2 ≈ 989.5` TFLOP/s，这正是 `h100_flop_per_sec = 1979e12 / 2` 的来历。[28:57]–[29:13] 还给出量级锚点：谈到 `1e22`、`1e23`、`1e25` 这类数字时，都是隐含地指某些模型的 compute 规模；而 `8 H100 × 一周 ≈ 5e21 FLOPs` 属于 [30:44] 自嘲的 "nothing fancy, it's just math and napkin math"（餐巾纸估算），用来建立硬件与模型之间的 rough shape 直觉。易错点与局限：一是大小写与 `/s` 不能混，写错会让「计算量」和「吞吐」差一个时间量纲；二是标称峰值带 sparsity、且 `FLOP/s depends heavily on the data type`，BF16 的数字不能套到 FP8 或 FP32 上；三是这套估算刻意忽略通信与其它开销，只是 order of magnitude 级别的量级判断，不能当精确预算。

这一章还有几处容易被略过但值得记的细节。[27:26] 讲课者把话题拉回时先复述了本讲的资源核算框架：「So, I have tensors. We've talked about how they take memory. So, how much compute do they take?」——可见 FLOPs 是与前面 memory 并列的第二本账。[28:03] 他把 flops 的歧义称为「a kind of a pet peeve」；[28:11] 补充说小写 s 的 `flops` 是「the number of floating point operations」，而 [28:27] 大写 `FLOPS` 会被「very confusingly」拿来指每秒，[28:32] 因此他「always write /s to make it clear that this is measuring the speed of hardware」。[28:57]–[29:09] 又给了一个量级提醒：口头上说的 `1e22`、`1e23`、`1e25` 这类数字「are kind of referring implicitly to amount of compute or the scale of some of these models」，所以听课时得靠上下文判断说的是计算总量还是吞吐。[29:13]–[29:21] 他想在 H100 的「glossy spec sheet」上找数字，结果发现「actually, it's not on this page. Okay, forget it」，[29:39] 又说「you go and you benchmark and it's like, wait a minute, that's not actually what I'm getting」——这两处口误与实测落差，正好引出后面规格表脚注的 sparsity 陷阱。[30:03]–[30:27] 的估算本意是「eight H100, so this one node for two weeks」，但他中途改成「this is looks like it's one week. Okay, fine. It's a one week」；[30:35] 收尾说这只是「building intuition for the number of flops that certain types of hardware have and how many flops certain types of models require」。

## 从线性层数 FLOPs：2BDK 与 2×tokens×params [30:49 - 34:45]

讲课者说 "let's do something more mechanical"，并指出 counting flops 的核心其实就是 linear matmuls（线性矩阵乘），所以从线性模型入手并不失一般性（"not without too much loss of generality"）。设定是：有 $B$ 个数据点（number of points），每个点 $D$ 维（input dimensions），模型把每个 $D$ 维向量映射成 $K$ 维输出（number of output dimensions）。构造数据矩阵 $X \in \mathbb{R}^{B\times D}$ 与权重矩阵 $W \in \mathbb{R}^{D\times K}$，做一次 matmul 得到 $Y = XW$；问题就是这次 matmul 需要多少次 flops。`kf_0026`（31:06）标题 "Linear model"，副标题 "As motivation, suppose you have a linear model."，三个 bullet 是 "We have n points"、"Each point is d-dimensional"、"The linear model maps each d-dimensional vector to a k outputs"（黄色高亮第 298 行）。

`kf_0027`（32:29）给出可直接运行的代码与具体参数（行号 02–15）：

```python
if torch.cuda.is_available():
    B = 16384   # Number of points
    D = 32768   # Dimension
    K = 8192    # Number of outputs
else:
    B = 1024
    D = 256
    K = 64

x = torch.ones(B, D, device=cuda_if_available())
w = torch.randn(D, K, device=cuda_if_available())
y = x @ w
# We have one multiplication (x[i][j] * w[j][k]) and one addition per (i, j, k) triple.
actual_num_flops = 2 * B * D * K
```

黄色高亮的第 15 行是 `actual_num_flops = 2 * B * D * K`。推导如幻灯片原句：每个 $(i,j,k)$ 三元组需要一次乘法 `x[i][j] * w[j][k]` 和一次加法，因此

$$\text{FLOPs}=2BDK$$

具体数字：CUDA 分支 $B=16384$、$D=32768$、$K=8192$，CPU 分支为 $B=1024$、$D=256$、$K=64$；取 CUDA 分支则 $2BDK = 2\times16384\times32768\times8192 \approx 8.8\times10^{12}$。讲课者在 [32:12] 补了严谨性说明：严格说加法只有 $D-1$ 次而不是 $D$ 次，所以有个 "minus one" 修正，但 "let's ignore that"，一阶近似下仍取 $2BDK$。

![线性层 FLOPs 计数代码：B = 16384、D = 32768、K = 8192，CPU 分支为 1024、256、64，x @ w 后黄色高亮 actual_num_flops = 2 * B * D * K，并注明每个 i j k 三元组一次乘法一次加法](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0027_00-32-29.jpg)

[34:03] 起讲课者给出另一层更有用的解读（幻灯片 "Interpretation" 小节）：`B` 是 number of data points，`(D K)` 是 number of parameters，于是

$$\text{forward FLOPs}=2\times\#\text{tokens}\times\#\text{parameters}$$

幻灯片原句 "FLOPs for forward pass is 2 (# tokens) (# parameters)"，紧跟一句 "It turns out this generalizes to Transformers (to a first-order approximation)."，对应转写 [34:34] "if you remember the six times N times D formula, we can kind of see the shape of that forming"——$6ND$ 里的前向 $2ND$ 已在此成形，剩下 $4ND$ 留给 backward（本段之后才展开）。`kf_0028`（33:52）标题 "FLOPs of other operations"，bullet 说明 $m\times n$ 矩阵上的逐元素操作只需 $O(mn)$（转写里 "addition also requires NM flops"），且 "In general, no other operation you encounter is expensive as matrix multiplication for large enough matrices"。讲课者还回答两问：关于 sub-cubic 矩阵乘算法，他说实践中优化更多是 "co-design with the systems" 而非渐进复杂度更低的算法；关于加法是否比乘法便宜，他承认直觉上似乎更快，但 "the way hardware is built they're kind of the same"，故两者按同等 flop 计数。与前后关系：这一章把上一章抽象的 FLOP 落到具体线性层，得到的 `actual_num_flops` 正是最后一章 `actual_flop_per_sec = actual_num_flops / actual_time` 的分子。易错点有三：别因 $D-1$ 修正过度纠结也别完全忘掉它，一阶近似可忽略；$2BDK$ 只数 forward，不能当成训练总 FLOPs；「其它运算都可忽略」有前提，转写里特意加了 "with the important caveat of when we talk about memory"——谈内存时逐元素操作不能忽略，这正是后面算术强度与 roofline 的伏笔。

把推导链条再补完整：[30:54]–[31:08] 讲课者先说明为何选线性模型——「a lot of this calculus of counting flops is actually going to be at the core it's like linear matmuls. So, this is actually not without too much loss of generality」，即大模型里绝大多数 FLOPs 都来自线性层，抓住它就不失一般性。[31:12] 他先称数据点数为 `M`、每个点 `D` 维、输出 `K` 维，[31:21]–[31:33] 才统一改用 `B` 表示 number of points，并明确「let's construct an X, which is the data matrix B by D, the weight matrix is D by K, and we do a matmul」。[31:49]–[32:11] 给出结论的两步论证：「two times basically the product of all the three dimensions」，因为「one multiplication for each triple, and then also one addition」；[32:23] 他还说这段讲得快，「I'll come back to this if this was a bit fast ... there's another way to derive this」。`kf_0028` 的 bullet 除了 `m×n` 逐元素操作，还写了「Addition of two m x n matrices requires m n FLOPs」与「In general, no other operation that you'd encounter is as expensive as matrix multiplication for large enough matrices」。[32:59] 有学生问是否存在别的矩阵乘算法，讲课者确认指的是「sub cubic cubic algorithms」，并回答实践中优化更多来自「how you co-design with the systems」而非渐近复杂度；[33:33] 又有学生追问加法与乘法能否区别对待，他重申「the two are basically the same」。[34:03]–[34:21] 最后把公式换写成「two times the number of tokens or data points times the number of parameters」，并把 `B` 与 `D×K` 分别对应到 tokens 与 parameters。

## 从 FLOPs 到墙上时间：计时、规格表与 MFU [34:45 - 38:42]

讲课者先给了诚实的免责："unfortunately, these calculations are not going to be very meaningful because I'm doing this on on CPU"（[34:46]），但代码逻辑仍要讲清，因为它是后面 benchmarking 课的 preview（[35:18] "in a few lectures we're going to talk more about benchmarking, um but here's a little preview"）。区别在于：前面数出的 flops 是 "independent of hardware"、只是模型需要的逻辑计算量（[35:03]），现在要回答 "how long does it actually take on hardware"，最直接的办法就是 "you just time it"（[35:15]）。这一步把「算了多少」和「跑了多久」接起来，也回应了本讲开头两个 back-of-envelope 估算里默认 `MFU=0.5` 的来历。

`kf_0029`（35:15）黄色高亮第 330 行 `actual_time = time_matmul(x, w)`，紧接着 `actual_flop_per_sec = actual_num_flops / actual_time`，以及 "Each GPU has a specification sheet that reports the peak performance."、"Example: [H100 spec]"、"Note that the FLOP/s depends heavily on the data type!" 与 `promised_flop_per_sec = get_promised_flop_per_sec(x.dtype)`。计时陷阱在 [35:27]–[36:08]：GPU 异步执行，所以操作前后都要放同步点

```python
torch.cuda.synchronize()   # 操作前：建立同步点
actual_time = time_matmul(x, w)
torch.cuda.synchronize()   # 操作后：同步屏障
actual_flop_per_sec = actual_num_flops / actual_time
```

若漏掉 `torch.cuda.synchronize()`，"you're going to find that wow, your timings are really fast"，因为那只是 non-blocking 调用立刻返回的时间，测到的不是真实执行；[36:02] 还补一句 "often it's general good practice to try this multiple times and take the average"（重复多次取平均）。`kf_0030`（36:38）是讲课者现场打开的 NVIDIA 网页截图：黑底、左上角 "Data Center"、大标题 "NVIDIA Extreme Co-Design Delivers New MLPerf Inference Records"、正文 "The NVIDIA inference platform powers the highest performance across a broad range of workloads to scale AI factory revenue."、绿色 "Read Blog" 按钮；他自嘲 [36:36] "I feel like maybe I have the wrong link ... I'll have to fix that after class"，这个插曲恰好说明规格数字并不好查、且必须按 dtype 对号入座。转写 [36:47] 提到的规格数字是 `989 teraflops`，正是上一章 `1979 / 2` 的 dense 值。

把「实测吞吐」与「标称吞吐」摆在一起，就有了 `kf_0031`（38:00）标题 "Model FLOPs utilization (MFU)"，定义为

$$\text{MFU}=\frac{\text{actual FLOP/s}}{\text{promised FLOP/s}}=\frac{\text{actual\_flop\_per\_sec}}{\text{promised\_flop\_per\_sec}}$$

幻灯片原文 "Definition: (actual FLOP/s) / (promised FLOP/s) [ignore communication/overhead]"，黄色高亮行 "Usually, MFU of ≥ 0.5 is quite good (and will be higher if matmuls dominate)"。`kf_0031` 顶部还留着 `Let us time it!`、`actual_time = time_matmul(x, w)`、`actual_flop_per_sec = actual_num_flops / actual_time` 三行，把分子分母串成一条链。具体数字（[37:39]–[38:21]）：现代模型训练能到 `0.5` 就 "should be pretty happy with yourself"；纯 matmul（"a straight-up matmul"）"maybe 0.8 even"；`you just never get anything more than what you were promised`，即 MFU 不会超过 1；若只有 `0.1`，说明 "something's really wrong, you should do something"。[38:22] 讲课者总结 "you now know how to calculate MFU through a combination of counting the number of logical flops ... and then looking at the wall clock time and essentially dividing"。

![Model FLOPs utilization 即 MFU 幻灯片：定义 mfu = actual_flop_per_sec / promised_flop_per_sec，黄色高亮行指出 MFU 不小于 0.5 已属不错且 matmuls 主导时更高](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0031_00-38-00.jpg)

易错点与局限要一并记住：一是 MFU 定义里明确 "[ignore communication/overhead]"，它是忽略通信与其它开销的偏乐观指标；二是 `promised_flop_per_sec` 必须按 `x.dtype` 取（BF16、FP8 数值不同），且要用 dense 值而非带 sparsity 的标称值，否则分母虚高、MFU 假性偏低；三是 MFU 低不一定是代码写错，也可能是 memory bound（内存瓶颈），本段只点到 "something's really wrong"，真正原因留到后面算术强度与 roofline 才解释；四是 CPU 上的计时结果不能外推到 GPU，讲课者开头那句免责正是提醒这一点。还有几处转写里的细节值得补进去。[34:58]–[35:12] 讲课者把前面工作定性为「measured flops ... independent of hardware, it's just like the number of calculations you need to do for your model」，然后才问「how long does it actually take on hardware」；[35:33] 解释为什么要同步：「the GPU is like kind of running asynchronously, you want to make sure that you have this kind of synchronization point」。[36:10]–[36:24] 给出实际吞吐的定义是「basically the number of flops you did times the time that you recorded on your hardware」，结合代码显然应理解为「除以时间」而非相乘，听的时候要按 `actual_flop_per_sec = actual_num_flops / actual_time` 校正。[36:26]–[36:43] 他试图现场打开规格页，结果「I feel like maybe I have the wrong link ... I'll have to fix that after class」，口述的数字是「989 teraflops」；[36:59] 点明「the number of actual flops per second is going to be different from the promised flops per second」，[37:12] 于是引入 MFU 来「think about the discrepancy」，[37:25] 给出定义时还特意补了一句「ignoring the communication and other overhead」。`kf_0031` 画面底部已经露出下一节的 "Summary" 标题与 bullet「Matrix multiplications dominate: (2 m n p) FLOPs」——把 $2BDK$ 里的三个维度写成一般的 $m,n,p$，等于把线性层结论推广成任意矩阵乘，也为后面算术强度与 roofline 埋下伏笔。[37:46] 他还强调「you just never get anything more than what you were promised. Often you get less」，[38:22] 收尾把方法概括成「counting the number of logical flops that your model needs to do ... and then looking at the wall clock time and essentially dividing」。把第 6 章的 `actual_num_flops`、本段的 `actual_time`、第 5 章的 `promised FLOP/s` 三者相除，就得到评估训练效率的完整闭环。

## MFU 再追问：峰值算力、数据类型与 0.5 的来源 [38:42 - 40:27]

本段以一段课堂问答开场，承接上一段「从 FLOPs 到实际时间」留下的悬念。有学生追问 promised flops（厂商承诺的峰值算力）究竟是多少，讲者回答：规格表上的那个数字已经是 "already divided by two" 之后的了，也就是说拿到手先要除以 2；紧接着又补一句 "on top of that you only get like 0.5 of that in general"，实际运行一般只能再拿到约一半。于是 "why are you getting only 50% MFU?" 这个问题被正式提出来。讲者没有立刻解答，只说 "I'll come back to that when we talk about memory bottlenecks"，把答案推迟到内存瓶颈的部分。这一推迟贯穿整个第 3 段，直到 54:08 之后才被兑现，构成本段最重要的一条叙事线索。转写在这里把峰值记作 1989，而后面屏幕上可见的断言写作 `1979e12 / 2`，两处数字不完全一致，本段以屏幕上可见的代码写法为准，不强行替讲者解释这个差异。

随后讲者用 "to summarize here" 收束上一部分。第一句是 matrix multiplications dominate the computations，"and that's sort of by design"：矩阵乘法主导了整体计算量，这在某种程度上是设计使然，而不是偶然。投影上那张被讲者身体挡住大半的幻灯片里，淡色的 Summary bullet 写着 "Matrix multiplications dominate: (2 m n p) FLOPs"，也就是把矩阵乘法的计算量写成

$$
\text{FLOPs}=2mnp
$$

这和前面章节里的 $2BDK$ 是同一件事，只是把三个维度换成了 $m,n,p$ 的记号：两个维度决定矩阵形状，第三个维度是求和轴，每个 $(i,j,k)$ 组合贡献一次乘法与一次加法，故系数为 2。第二层小结是：每秒浮点运算数 FLOP/s 取决于硬件，better hardware leads to more flops；同时也取决于 data type（数据类型），看规格表时不同 dtype 会给出不同的 FLOP/s 行。讲者特别提醒，如今跑 float32 会 "really really slow"，因为硬件已经不再为这种 workload 做优化；而 BF16 或 FP8 会快得多。这一句解释了为什么 MFU 的分母不能随手取一个数字：分子与分母必须对应同一种精度，否则比值失去意义。

幻灯片标题是 "Model FLOPs utilization (MFU)"，即模型算力利用率，下面写着定义与一行代码：

```python
# Definition: (actual FLOP/s) / (promised FLOP/s)
mfu = actual_flop_per_sec / promised_flop_per_sec
```

写成公式是

$$
\text{MFU}=\frac{\text{actual FLOP/s}}{\text{promised FLOP/s}}
$$

屏幕上还写着 "MFU of ≥ 0.5 is quite good"。MFU 是一个无量纲比值：actual FLOP/s 是实测吞吐（上一段由 $\text{actual FLOPs}/\text{actual time}$ 得到），promised FLOP/s 是硬件规格表上的峰值。它衡量的是「买来的算力真正被用上了多少」，而不是算得快不快。0.5 在这里被当作经验判断给出：达到 0.5 就算不错，纯矩阵乘法也许更高，明显低于它则说明某个环节出了问题。讲者随后用一句 "Okay? And MFU, now you know what MFU is. It's the actual flops divided by promised flops" 把定义压实，然后立刻转入 "to go back to the question of why is MFU 0.5"。

这一段在整讲里起承上启下的枢纽作用：它把上一部分的计时与 MFU 概念收尾，同时用「为什么只有 50%」这个悬而未决的问题，把话题从单纯的 FLOPs 计数推向硬件的 memory bandwidth（内存带宽）与 arithmetic intensity（算术强度）。几个容易踩的坑值得先记下：0.5 是经验值而不是定律，不能当成任何模型都必须达到的硬指标；promised flops 是带 sparsity 的标称值，dense 场景要除以 2，直接拿原值当分母会把 MFU 算低一半；不同数据类型对应不同峰值，把 FP32 的峰值拿去除 BF16 的实测吞吐也会得到荒谬结果；另外讲者此时刻意不给出 50% 的成因，若在这里就归因于「代码写得差」是错的，真正原因要到后面用数据搬运时间与计算时间的对比才能说清楚。

![MFU 定义幻灯片：Definition、mfu 计算式与「≥0.5 已属不错」的判断](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0032_00-39-18.jpg)

## 硬件卡通图与内存带宽：通信时间、计算时间与重叠假设 [40:27 - 45:43]

为了回答「MFU 为什么只有 0.5」，讲者说必须引入 arithmetic intensity（算术强度）这个想法，并强调 "it's not just doing a bunch of matmuls and then you're done and looking at how long the matmuls take"——只盯着矩阵乘法本身算多久是不够的。他先画了 "my very cartoon version of what hardware looks like"：一块 High Bandwidth Memory（HBM，高带宽内存），加上 compute cores 或 accelerator chips（计算核心 / 加速器芯片）。tensors（张量）原本坐在内存里，要先把输入矩阵搬进加速器，做完计算，再把结果搬回内存。屏幕上标题为 "Memory" 的幻灯片把这一过程写成 "How to compute a thing" 三步：1. Send inputs from memory to accelerator；2. Perform computation；3. Send outputs from accelerator to memory。这个极简模型刻意忽略了缓存层次与片上互联，只保留「搬数据」与「算数据」两件事，正是为了把瓶颈单独隔离出来。

接着是 "How long does this take? Depends on two things"：1. Accelerator speed (FLOP/s)；1. Memory bandwidth (bytes/s)（屏幕上两条都编号成 1，是个小笔误）。第一条上一部分已经讲过，第二条是本段新增的。屏幕上两行断言把两者钉死：

```python
assert h100_flop_per_sec == 1979e12 / 2  # Half without sparsity
assert h100_bytes_per_sec == 3.35e12
```

即算力约 $1979\times10^{12}/2=9.895\times10^{14}$ FLOP/s，内存带宽 $3.35\times10^{12}$ bytes/s（讲者口头说成 "3.3 terabytes per second"）。与前面只数 FLOPs 的章节相比，这里第一次把「搬数据」也计入时间。讲者还补了一个动机：之前讨论内存时最直观的问题是模型太大放不进显存，但即便放得下，memory 也必须被搬来搬去，而搬运本身要花时间，所以张量的大小同样会影响速度，不只影响能不能装下。

随后进入本段第一个完整算例 ReLU。代码定义了一个百万维 BF16 向量上的逐元素激活：

```python
def arithmetic_intensity_relu():
    n = 1024 * 1024
    x = torch.ones(n, dtype=torch.bfloat16, device=cuda_if_available())
    y = torch.relu(x)   # 屏幕上高亮这一行
```

$n=1024\times1024=1048576$，正是讲者说的 "a million dimensional vector of BF16"。ReLU 就是 $\text{ReLU}(x)=\max(x,0)$，对整个向量逐元素做，"done element-wise on the entire vector"。计数分两笔：搬运量与计算量。搬运是

```python
bytes = (2 * n) + (2 * n)  # Read x, write y (bf16 is 2 bytes/float)
flops = n                  # n comparisons
```

读 X 花 $2N$、写 Y 花 $2N$，共 $4N=4194304$ bytes，因为 BF16 每个浮点占 2 bytes；计算则是每个元素只和 0 比一次，共 $N=1048576$ 次比较。这里已经能看出不对称：搬运的字节数比浮点运算次数还多 4 倍。

把两笔量各自除以对应的硬件速度，就得到两个时间：

```python
communication_time = bytes / h100_bytes_per_sec   # = 1.252e-6
computation_time   = flops / h100_flop_per_sec   # = 1.060e-9
```

代入数字即 $4194304 / 3.35\times10^{12}\approx1.252\times10^{-6}$ 秒，$1048576 / 9.895\times10^{14}\approx1.060\times10^{-9}$ 秒，二者相差约三个数量级。屏幕右上角的结果框里逐行写着 `communication_time = 1.252e-6`、`computation_time = 1.060e-9`。communication time 是需要搬运的字节数除以搬运速度，computation time 是浮点运算数除以算力，单位都回到秒，因此可以直接比较——这是整套分析能成立的关键量纲对齐。

然后讲者引入一个 "another important assumption"，即尽量让通信与计算 overlap（重叠）：不必傻等数据全部搬完才开算，而是数据一到就开始算，边算边搬回，于是

```python
# Assume we can overlap communication and computation perfectly.
total_time = max(communication_time, computation_time)   # = 1.252e-6
```

数学上假设能完美重叠，总时间取两者最大值，结果框给出 `total_time = 1.252e-6`。讲者明确说实践中不会完美重叠、会有 overhead，"but this is good enough for now"，并预告讲 GPU 时会更细地讨论这个假设。这段把前面分别算出的两个时间合成一个可比较的总时间，也解释了为什么前面要分别算它们。

有了总时间就能定义瓶颈。屏幕上写着：

```text
What is the bottleneck?
  Memory-bound:  communication time > computation time
  Compute-bound: computation time > communication time
```

前者的直觉是 "you're spending most of your time just waiting for bits to show up"（大部分时间都在等比特送到），后者则是瓶颈真的在做计算。讲者当场问学生 ReLU 属于哪一种，答案是 memory bound，还带出一阵笑声；屏幕随后补上 "In this case, ReLU is memory-bound."。从数字上看 $1.252\times10^{-6}$ 远大于 $1.060\times10^{-9}$，结论毫无悬念。

这一段有几个局限必须记住：卡通图只算了 HBM 与加速器之间的一次往返，没算片上缓存复用、kernel 启动开销与调度延迟；`total_time = max(...)` 完全依赖「完美重叠」这个理想化假设，真实硬件上重叠不充分时总时间会更接近两者之和；所有字节数都按 BF16（2 bytes/float）计算，换成 FP32 会同时放大 bytes 并改变 flops，结论未必保持；此外这个判据还是基于时间的，下一章会把它改写成基于强度的形式，两者在同样假设下等价。

![Memory 幻灯片：计算三步、两个决定因素与 H100 算力/带宽断言](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0034_00-41-43.jpg)

![ReLU 算例：bytes、flops、两个时间与重叠假设、瓶颈判据](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0036_00-44-09.jpg)

## 加速器强度与算术强度：295 与 0.25 [45:43 - 48:12]

讲者说 "here's another way to see it, and this is where I'm going to define the intensity"，于是把上一章基于时间的瓶颈判据换成一个更便于记忆的比值语言。第一种叫 accelerator intensity（加速器强度），屏幕上写成 "how much work can the accelerator do per byte transferred?"，即每搬运一个 byte，硬件能做多少有用的功。对任何给定加速器，按规格表取算力除以带宽：

```python
h100_accelerator_intensity = h100_flop_per_sec / h100_bytes_per_sec
```

$$
I_{\text{hw}}=\frac{\text{FLOP/s}}{\text{bytes/s}}
$$

代入 H100 的两个数，$1979\times10^{12}/2 \div 3.35\times10^{12}\approx295.3731$，结果框里写 `h100_accelerator_intensity = 295.3731`，讲者口头说成 295，并让大家记住 "about 300"。它的直觉解释是：每搬进来 1 byte，加速器能配着做约 295 次浮点运算；这是硬件的固有属性，与跑什么算法无关，单位是 FLOP/byte。

第二种叫 arithmetic intensity（算术强度），屏幕上写 "how much actual work per byte for this workload?"，即某个具体 workload 实际每搬一个 byte 做了多少功。定义是

```python
arithmetic_intensity = flops / bytes   # ~1/2（屏幕注释）
```

$$
I_{\text{alg}}=\frac{\text{flops}}{\text{bytes moved}}
$$

这里有一个很值得记的易错点：代码注释写的是 `# ~1/2`，但讲者在转写中当场纠正 "it's actually a quarter, I guess. Um not half"，结果框给出的也是 `arithmetic_intensity = 0.25`。对 ReLU 而言 $N$ 次比较除以 $4N$ bytes 正好是 $0.25$，也就是四分之一；注释里的「约 1/2」是错的，别照抄。讲者还专门强调，如果有人告诉你某个 workload 的 arithmetic intensity 是 0.25，你应该立刻反应 "Oh, this is really bad"——因为它离 295 差了三个数量级。

用强度重写瓶颈判据，屏幕上给出两行：

```text
What is the bottleneck
  Memory-bound:  arithmetic intensity < accelerator intensity
  Compute-bound: arithmetic intensity > accelerator intensity
assert arithmetic_intensity < h100_accelerator_intensity
```

$0.25 < 295.3731$，所以 ReLU 依旧是 memory bound，与上一章用时间比较得出的结论一致。讲者说这两套判据是 equivalent（等价的），"if you look at the algebra, it's basically you have two fractions, and then you just kind of multiply and divide to basically switch the terms around"。确实如此：把 $\text{bytes}/\text{bytes per sec} > \text{flops}/\text{flop per sec}$ 交叉相乘，就得到 $\text{flops}/\text{bytes} < \text{flop per sec}/\text{bytes per sec}$，即算法强度小于加速器强度。换句话说，强度判据只是把「比较时间」变成「比较两个分式」，前提仍是同一个完美重叠假设和同一套 bytes/flops 计数。

屏幕底部高亮的一行是 "In general, we'll find ourselves memory bound, so higher arithmetic intensity is good!"。理由是 "data movement is expensive"（数据搬运很贵）：既然大多数算子天生卡在搬运上，那么提高算术强度就等于把同样的搬运成本摊到更多计算上，是效率优化的主方向。有学生追问 "What's some typical arithmetic intensity"，讲者没有马上给数字，只说 "so we'll get to that"，把答案留到后面 GELU 与矩阵乘法的例子里——这也正是下一章的起点。

这一章的局限同样清楚：295.3731 是 H100 在特定精度下的比值，不是通用常数，换一张卡或换一种 dtype 就会变；两个强度都是 FLOP/byte 这类无量纲比值，别把它们和 FLOP/s（速度）或 FLOPs（总量）混淆；判据里的严格小于、严格大于在恰好相等处没有定义，实践中临界点附近也不该指望一刀切；此外强度只描述算法与硬件的匹配程度，不包含 kernel 实现质量、缓存命中率等工程因素。下一章就沿着「怎么把 0.25 提上去」这条思路，先试最直观的办法：每搬一个 byte 多做点计算。

![算术强度定义、0.25 结果、295.3731 与两种强度判据](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0039_00-47-46.jpg)

## GELU 与逐元素算子：做更多功并不等于更慢 [48:12 - 49:46]

提高算术强度最直观的一条路，讲者概括为 "let's just try to do more stuff per unit of byte moved"，即每搬运单位字节尽量多做计算。他选的对照物是 GELU，另一种 activation（激活函数）。讲者说它 "kind of looks like this"，是一条不像 ReLU 那样带硬零的曲线，"doesn't have zeros"。屏幕上代码注释给出了精确公式：

```python
y = F.gelu(x)  # GELU(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
```

$$
\text{GELU}(x)=0.5x\left(1+\tanh\left(\sqrt{\tfrac{2}{\pi}}\left(x+0.044715x^3\right)\right)\right)
$$

式子里有 $\tanh$、常数 $\sqrt{2/\pi}$、三次项 $0.044715x^3$ 以及若干乘法加法，比 $\max(x,0)$ 复杂得多。按直觉，这么复杂的函数应该明显更慢，这正是本段要推翻的预设。

完整的计数代码是：

```python
def arithmetic_intensity_gelu():
    n = 1024
    x = torch.ones(n, dtype=torch.bfloat16, device=cuda_if_available())
    y = F.gelu(x)
    bytes = (2 * n) + (2 * n)  # Read x, write y (bf16 is 2 bytes/float)
    flops = 20 * n            # tanh can approximated in various ways (e.g., polynomial)
    arithmetic_intensity = flops / bytes
```

注意这里 $n=1024$，比 ReLU 例子的 $1024\times1024$ 小了三个数量级，但强度是比值，$n$ 会被约掉，不影响结论。搬运量仍是 $2N+2N=4N$ bytes，与 ReLU 完全相同，因为逐元素激活都是读一个向量、写一个同形向量；计算量按讲者的粗略估计约 20 flops per element，即 `flops = 20 * n`，注释还说明 $\tanh$ 可以用各种方式近似（例如多项式），所以 20 本身是 "a kind of a crude estimate"。于是

$$
I_{\text{alg}}=\frac{20N}{4N}=5
$$

结果框里写 `arithmetic_intensity = 5`，讲者说 "let's say, five"，并再次强调这只是粗估。屏幕上还有 `assert arithmetic_intensity < h100_accelerator_intensity` 这一行，$5$ 与 $295.3731$ 相比仍是 "Way smaller"，所以讲者问学生是 memory bound 还是 compute bound 时，答案仍是 memory bound。

屏幕上三行注释把这一章的结论说得很直接：

```text
Note that GeLU does more work than ReLU per byte moved, so it has higher arithmetic intensity.
But still memory-bound!
In other words, ReLU is not faster than GeLU (when doing things in an isolated way).
```

也就是说，GELU 确实把强度从 ReLU 的 0.25 提到了 5，方向对了，但离硬件阈值 295 还差两个数量级，瓶颈依旧在搬运。讲者把这点讲得更狠："if you were just computing ReLU and GELU, you think that GELU is like so complicated, it must be really expensive, but actually it's exactly the same. Right? Because that's not where the bottleneck is." 孤立地看，两个激活函数的墙钟时间几乎一样，因为决定时间的是那 $4N$ bytes 的往返，而不是每个元素里多出来的十几次浮点运算。

这一章与前后内容的关系很清楚：它承接上一章「0.25 is really bad、higher arithmetic intensity is good」的号召，验证了「多做功」这条途径确实能把强度提高 20 倍，同时也暴露了逐元素算子的天花板——搬运量与计算量同为 $O(N)$，比值是个与规模无关的常数，再怎么把激活函数写复杂也只是把常数从 0.25 挪到 5。要真正跨过 295，必须让计算量随规模超线性增长，这正是下一章矩阵乘法要做的事。易错点也不少：`20 * n` 依赖 $\tanh$ 的具体实现，换多项式阶数、换数学库、换硬件，这个常数都会变，不能当成精确值；"ReLU is not faster than GeLU" 只在 "in an isolated way"（孤立执行）的前提下成立，一旦把激活融合进大 kernel、改变 batch 或复用缓存，结论可能变；此外这里默认精度是 BF16，讲者在 53:58 会补充 intensity 依赖精度，换 dtype 时 bytes 与 flops 会同时变，比较结果需要重算。

![GELU 算例：公式、20 flops、bytes 与 arithmetic_intensity = 5](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0040_00-48-59.jpg)

## 从点积到矩阵乘法：强度跨过 295 [49:46 - 52:53]

讲者说 "now let's look at some linear operations"，把分析从逐元素激活转到线性运算，依次考察 dot product（点积）、matrix-vector product（矩阵-向量乘）与 matrix multiplication（矩阵乘法）。第一个是 dot product：两个长度为 $N$ 的向量 $X$ 与 $W$ 做内积，输出一个标量 $Y$。代码是：

```python
def arithmetic_intensity_dot_product():
    n = 1024
    x = torch.ones(n, dtype=torch.bfloat16, device=cuda_if_available())
    w = torch.ones(n, dtype=torch.bfloat16, device=cuda_if_available())
    y = x @ w
    bytes = (2 * n) + (2 * n) + 2  # Read x, read w, write y
    flops = 2 * n - 1              # n multiplications, n-1 additions
    arithmetic_intensity = flops / bytes  # ~1/2
```

搬运量是读 X 的 $2N$、读 W 的 $2N$、写标量 Y 的 2 bytes，合计 $4N+2$；计算量是 $N$ 次乘法加 $N-1$ 次加法，即 $2N-1$ flops。取 $n=1024$ 时 $(2\times1024-1)/(4\times1024+2)=2047/4098\approx0.5$，讲者说 "about half... which is also pretty bad"，代码注释也写 `# ~1/2`，结论是 memory bound。与 GELU 相比，点积虽然少了写回整个向量的开销，但输出只有一个标量，分子分母的比值并没有本质的改善。

接着是 matrix-vector product（矩阵-向量乘）：$X$ 是长度 $N$ 的向量，$W$ 是 $N\times N$ 矩阵，$Y=XW$。按转写逐笔计数，读 X 是 $2N$、读 W 是 $2N^2$、写 Y 是 $2N$，即

$$
\text{bytes}=2N+2N^2+2N
$$

计算相当于做 $N$ 次 dot product，也就是

$$
\text{flops}=N\,(2N-1)\approx 2N^2
$$

讲者先让学生举手表决 "How many of you think this will be compute bound? How many of you think memory bound?"，然后给出答案：算术强度 "barely barely higher"，只比 1 略高一点，仍是 also memory bound。直觉是：为了算这一个向量，得把整个 $N\times N$ 权重矩阵读进来，$O(N^2)$ 的搬运只换来 $O(N^2)$ 的计算，比值与 $N$ 无关，怎么放大规模都不会自动跨过阈值。这一点在后面讨论推理时会被反复引用。

真正让局面改变的是 matrix multiplication（矩阵乘法）。讲者口头说 "I have a M by M matrix, another M by M matrix"，但公式与代码里统一用 $N$ / `n`。两个 $N\times N$ 矩阵相乘的代码在屏幕上逐行展开：

```python
w = torch.ones(n, n, dtype=torch.bfloat16, device=cuda_if_available())
y = x @ w
bytes = (2 * n * n) + (2 * n * n) + (2 * n * n)  # Read x, read w, write y
flops = n * n * (2 * n - 1)                       # n^2 dot products
arithmetic_intensity = flops / bytes               # ~n/3
h100_accelerator_intensity = h100_flop_per_sec / h100_bytes_per_sec
assert arithmetic_intensity > h100_accelerator_intensity
```

搬运是 $2N^2+2N^2+2N^2=6N^2$ bytes（读 X、读 W、写 Y 各 $2N^2$），计算是 $N^2$ 次点积、共 $N^2(2N-1)=2N^3-N^2$ flops，于是

$$
I_{\text{alg}}=\frac{2N^3-N^2}{6N^2}=\frac{2N-1}{6}\approx\frac{N}{3}
$$

讲者先喊 "It's 300, woohoo. Okay, so 340, and in general it's roughly N over three"，屏幕结果框给出精确值 `arithmetic_intensity = 341.1667` 与 `h100_accelerator_intensity = 295.3731`——$n=1024$ 时 $(2\times1024-1)/6=2047/6=341.1667$ 正好对上。断言 `assert arithmetic_intensity > h100_accelerator_intensity` 成立，屏幕上写 "Finally, compute-bound!"。

讲者给出的直觉是 "you're sending N squared things, but you're computing N cubed things"：搬运按 $O(N^2)$ 增长，计算按 $O(N^3)$ 增长，两者相除是 $O(N)$，所以矩阵越大，算术强度越高，"this gets better the larger you make the matrices"。这直接解释了业界总说要 large batch sizes 或 large matrices 的原因："If you're under the accelerator intensity, making things smaller doesn't actually speed things up, right? It's all kind of the same"，因为强度不够时大家都在等数据搬运，缩小规模只是同比例缩短搬运与计算；而一旦越过阈值，才开始真正 saturating（饱和）GPU，把算力吃满。

这一章的易错点集中在近似与形状上：$N/3$ 是渐近近似，精确式是 $(2N-1)/6$，小 $N$ 时不能直接套；`assert` 只在矩阵足够大时成立，$N$ 偏小时 $N/3 < 295$，矩阵乘法同样会退回 memory bound，所以「矩阵乘法一定 compute bound」是错的；另外 matrix-vector 与 matmul 在代码里都写成 `y = x @ w`，仅因 $x$ 是向量还是矩阵，强度就从「略高于 1」跳到 341，读代码时不能只看运算符、必须看形状；转写中 M 与 N 混用也提醒我们，公式里的字母只是记号，真正要紧的是搬运与计算各自随规模增长的阶。

![矩阵乘法算例：341.1667 与 295.3731、~n/3 及 compute-bound 断言](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0043_00-52-37.jpg)

## Transformer、推理与 MFU：为什么通常只有 50% [52:53 - 55:49]

讲者先用 "As long as we have large matrices, we're actually pretty good. We're in compute bound, saturating the accelerator" 给上一章收尾，然后回头回答开头挂起的另一个问题：what about transformers？他说在作业里、也在下一讲会看到，transformers 本质上 "essentially big matrix multiplications with some things sprinkled in between"，即大型矩阵乘法中间夹着一些别的东西。讲者把这称为 "good news from arithmetic intensity"，并强调 "this is by design. Transformers are designed in a certain way to have high arithmetic intensity"——Transformer 的结构选择本身就是在往高算术强度上靠。屏幕上的三行注释正是这个意思：

```text
As long as we have large matrices, we're compute-bound (saturating the accelerator).
Training Transformers involves big matrix multiplications
Matrix-vector product is what happens during inference, which is why inference is memory-bound.
```

这既呼应了本段开头 "matrix multiplications dominate... by design" 的小结，也把前几章停留在 ReLU、GELU、点积上的抽象比值落到了真实模型上：训练之所以能吃到算力红利，是因为主体运算落在屋顶线的右侧。

紧接着讲者 foreshadow（预告）后面的 inference lecture（推理那一讲）：matrix-vector product 正是 transformer inference 在做的事。因为推理 "you're generating one token at a time"，一次只生成一个 token，相当于拿一个向量去和权重矩阵做点积；而上一章刚算过 matrix-vector 的算术强度 barely higher than 1，属于 memory bound。相比之下 training time "you get this whole sequence, and you're processing all at once"，整条序列一次性并行处理，对应大矩阵乘法、compute bound。屏幕上高亮的那一行 `Matrix-vector product is what happens during inference, which is why inference is memory-bound.` 把训练与推理的效率差异归因到同一个强度框架，而不是两套玄学。屏幕底部还有一行淡色注释 `Note: arithmetic/accelerator intensity depends on the precision (bf16 versus fp32)`，讲者在 53:58 也补了一句 "by default everything we're doing here is FP16"——精度既改变每个元素的 bytes，也改变硬件给出的 FLOP/s，所以强度不是纯算法属性，换精度必须重新算。

现在可以兑现 38:42 的欠账了。讲者在 54:08 把话题拉回 MFU："MFU is a promised flops sorry, actual over promised"，即

$$
\text{MFU}=\frac{\text{actual FLOP/s}}{\text{promised FLOP/s}}
$$

他的解释是：如果模型存在很大的 memory bandwidth bottlenecks，那么即使「假设没有内存瓶颈、只管把模型的所有计算一路做完」能达到 promised 峰值，实际吞吐也不会好。换句话说，promised FLOP/s 是一个只计算法 FLOPs、不计数据等待的理想上限，而 MFU 的分子是从实际墙钟时间里测出来的，凡是让加速器空转等内存的算子都会把分子拉下来。这正好回答了 "why are you getting only 50% MFU?"，也解释了第一段为什么把 0.5 称作 quite good：一个 Transformer 里除了高強度的大矩阵乘法，还夹着大量逐元素激活、归一化之类的低强度算子，整体利用率自然被稀释。

课堂在 54:53 后进入一段问答。学生指出 "Generally speaking, these models tend to be memory bound, as they say, for most of the computation"，但加速器却只有 50%，好一点能到 70，80 就 "really good"，言下之意是 "the accelerator is outsized compared to memory bandwidth"（加速器相对内存带宽是过剩的）。讲者反问为什么，学生答 "they're just idling or waiting for memory"（在空转、等内存）。讲者没有下结论，只说等讲到 GPU、更理解其工作原理后再谈，还开了个玩笑："if you have an answer, you should tell Jensen, and maybe he can design a better hardware"。于是「为什么硬件设计成算力过剩」被留成一个开放问题，本段不给答案。

这一章的易错点比较隐蔽：MFU 低并不等于代码写得差或模型有问题，很可能只是 memory bottleneck 的正常表现；50%、70%、80% 是经验区间而非硬指标，别拿来当验收标准；训练与推理的结论不能互相套用，前者是 compute bound 的大矩阵乘法，后者是 memory bound 的矩阵-向量乘；精度未固定时比较强度或 MFU 会失真；另外讲者明确把「加速器为什么过剩」推迟到后续 GPU 章节，任何在这里给出的因果解释都超出了本段内容。

![大型矩阵乘法、Transformer 与推理 matrix-vector 的注释及强度断言](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0044_00-53-49.jpg)

## Roofline 可视化与回到训练账本 [55:49 - 58:03]

讲者用 "So, let's visualize the relationship between arithmetic intensity and performance" 引入 roofline plot（屋顶线图），投影上的标题是 "We can visualize the relationship between arithmetic ..."。他先说明坐标：X 轴是 arithmetic intensity（log-scale，对数刻度），"every vertical slice here corresponds to a particular, let's say, algorithm"，即每一条竖直切片对应某一个具体算法；图中每一条折线对应一个 accelerator，"Maybe H100s or B200s or so"；Y 轴是 FLOPs/s (realized, log-scale)，即实际达到的每秒浮点运算，同样是 log-scale。把强度放横轴、可达吞吐放纵轴，前面几章零散算出的数字就有了一张共同的地图。

细看这张图：纵轴顶端标 `Accelerator peak FLOPs / s`，一条水平虚线画出算力天花板；两条斜率不同的上升斜线分别标 $BW_1$（浅蓝）与 $BW_2$（品红），代表两种内存带宽。图被划成三个着色区，文字分别是粉红区的 `Bandwidth Bound at BW1 & 2`（两种带宽下都受带宽限制）、黄色区的 `Bandwidth Bound at BW1, Compute Bound at BW2`（低带宽下受限、高带宽下已算力受限）、绿色区的 `Compute Bound at BW1 & 2`。中间那条竖线标 `Critical hardware intensity for BW1 (hardware FLOPs/s / BW1)`，也就是硬件强度阈值所在位置；两条竖直虚线分别标 `Algo 1` 与 `Algo 2`，前者落在带宽受限区、后者落在算力受限区，圆点与蓝色箭头标出各自能达到的 FLOP/s。把两条屋顶线写成关系就是：斜线部分的可达吞吐约等于 $BW\times I_{\text{alg}}$，水平部分是 $\text{peak FLOP/s}$，实际能取到的是二者中较小的那个，

$$
\text{attainable FLOP/s}\le\min\big(\text{peak FLOP/s},\ BW\times I_{\text{alg}}\big)
$$

这也解释了 `Critical hardware intensity` 为什么正好是 $\text{FLOP/s}/\text{BW}$：两条线的交点就是强度阈值。

讲者的口头解读与图完全对应：低算术强度的算法，"like value or dot products"，会落在图左侧，realized FLOP/s 明显低于峰值；随着算术强度提高，"things are going to be better"，能逐步 saturate 硬件，直到某一临界点之后变成 compute bound，"and you obviously can't exceed the peak flops"。把前几章的数字回填进去：ReLU 的 0.25、dot product 的约 0.5 在左侧粉红区，GELU 的 5 稍往右但仍在带宽受限段，matrix-vector 略高于 1 也出不来，而矩阵乘法的 341.1667 越过 H100 的 295.3731 进入绿色区——这正是 "Finally, compute-bound!" 在图上对应的位置。

这张图的局限也就是前面所有假设的局限：它是理想化模型、双对数坐标，只保留带宽与峰值算力两条线，忽略 overlap 不完美、kernel 启动开销、缓存层次与调度延迟，所以只能给量级直觉，不能精确预测某个 kernel 的时间。讲者自己在引入重叠假设时就说过 "in practice it's not going to be perfectly overlapped, there's going to be some overhead"，屋顶线同样受这条限制；此外不同加速器（H100 与 B200）对应不同的 $BW$ 与峰值，换卡就要换一整套线，不能把 295.3731 当成通用常数套到所有图上。

57:04 起讲者 "go back and talk about memory and compute for the operations that we need in training"。他回顾说：到目前为止做的都是 tensor operations（基本是 matmuls），已经看清它们各占多少内存、多少计算，以及二者如何相互作用；现在要真正算「训练一次」要付出什么。屏幕上给出课程大纲：上方是 `# Memory accounting`（`tensors_basics()`、`tensors_memory()`、`tensors_on_gpus()`）与 `# Compute accounting`（`tensor_einops()`、`tensor_operations_flops()`、`arithmetic_intensity()`），当前高亮的是 `# Full example` 下的 `deep_linear_network()`，其后依次是 `gradients_basics()`、`gradients_flops()`、`optimizer()`、`train_loop()`，再往下淡色的 `# More memory optimizations` 里还有 `gradient_accumulation()`。这份大纲说明本段恰好站在从 compute accounting 过渡到完整训练例子的节点上，前面所有强度分析都是为这个 "Full example" 做铺垫。

最后的 running example 在 57:34 展开。讲者说 "here is our running example here"，随即更正 "actually it's not a linear network, uh just a deep network"——尽管函数名是 `deep_linear_network()`，他当场说明这只是个深网络，别被名字误导。设定是：输入为 $B\times D$，有若干层 $L$，每层是一个 $D\times D$ 的 matmul，产生一组 pre-activations（预激活值），再经过 element-wise value（逐元素激活）得到下一层输入。转写在 58:03 处恰好断在 "and then element-wise value, that produces some..."，后面没有念到，因此本段不往下补任何细节。这里也提醒读者：从这一刻起不能再简单套用前几章的 $N/3$ 结论，因为训练账本要同时纳入参数、梯度、优化器状态与 activations 的内存，以及 forward 与 backward 的 FLOPs；单个算子的强度只是整本账的一个条目。函数名与实际设定不一致、以及转写在段末被截断，都是阅读这一段时最容易记错的地方。

![Roofline 屋顶线图：算术强度横轴、三种着色区与 BW1/BW2、Algo 1/Algo 2](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0046_00-56-14.jpg)

## 深层网络的结构、参数量与梯度的基本机制 [58:03 - 60:16]

本段开头把上一节停留在抽象层面的 deep network（深层网络）落成一个可以逐项核算的具体对象。讲师重申：考虑一个有 $L$ 层、输入/激活/输出都是 $D$ 维的深层网络，每个 block（块）由一个线性变换加一个 point-wise activation（逐元素激活，这里是 ReLU）构成，线性变换带一个 $D\times D$ 的 weight matrix（权重矩阵），因此输出与输入保持相同尺寸，整个结构就是「linear → ReLU → linear → ReLU → …」不断重复。幻灯片顶部的示意图把这一点画得很直白：`input` 是 $B\times D$，随后交替出现标注为 `D linear`、`ReLU` 的方块，直到 `output`，其中 $B$ 是 batch 维度、$D$ 是特征维度。在 PyTorch 里，这个网络本质上是「一组 blocks」，每个 block 关联一个权重矩阵，参数量于是可以直接写成 $D^2 \times L$。

配套代码把抽象公式钉死在可断言的数字上：

```python
# Define the network
D = 8  # Dimensionality of input, activations, and output
L = 3  # Number of layers
model = DeepNetwork(dim=D, num_layers=L).to(cuda_if_available())

num_parameters = get_num_parameters(model)
assert num_parameters == (D * D) * L
```

画面里 `num_parameters` 的输出框显示 `= 192`，正好对应 $D=8$、$L=3$ 时的 $8\times 8\times 3 = 192$。下面灰显的部分还给出跑一个 batch 的写法：`B = 4  # Batch size`、`x = torch.randn(B, D, device=cuda_if_available())`、`y = model(x)`。也就是说，forward pass（前向传播）在 batch 数据上是逐层做线性变换、再做逐元素激活，重复 $L$ 次；这承接了前面章节把 FLOPs 归结为「三个维度相乘再乘 2」的结论，也为本段后面把 forward 与 backward 拆开计数埋下伏笔。

在 batch 上跑模型时的动作也讲得很具体：讲师说「when you run the model on a batch of data, what we're doing is we're going through the layers, and each layer we basically apply the linear transformation, and then a point-wise value activation. And then we do that for all the layers」——即逐层线性变换、逐层逐元素激活，循环 $L$ 次，输出尺寸与输入一致（「the output is just going to be the same size」）。参数量 $D^2\times L$ 之所以是「$D^2$ 乘以层数」，是因为每个 block 的权重矩阵都是 $D\times D$ 的方阵，且忽略了 bias（偏置）与激活函数本身不占可训练参数这一事实；这是一个刻意简化的记账模型，真实 Transformer 里还会有 attention 投影、LayerNorm 等额外参数，不能原样套用。

![深层网络结构示意图与 D=8、L=3、参数量 192 的定义代码](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0048_00-58-45.jpg)

紧接着讲师用一个「even simpler example」把话题从网络结构切到 gradients（梯度）：一个简单 linear regression（线性回归）模型，输入向量 `x = [1., 2., 3.]`，权重 `w = [1., 1., 1.]`（且 `requires_grad=True`），`pred_y = x @ w`，再构造 MSE 形式的损失 `y = 0.5 * (x * w - 5)^2`，代码里写作 `loss = 0.5 * (pred_y - 5).pow(2)`。执行 `loss.backward()` 之后，computation graph（计算图）里每个变量要么有梯度、要么没有：`assert loss.grad is None`、`assert pred_y.grad is None`、`assert x.grad is None` 都成立，而 `w.grad` 被设成 `[1, 2, 3]`，即 `assert torch.equal(w.grad, torch.tensor([1, 2, 3]))`。画面上的蓝色输出框也直接标出 `w.grad = 1 2 3`。

![简单线性回归的前向/反向代码与 w.grad = [1, 2, 3] 的断言](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0049_01-00-08.jpg)

这里有几个容易踩的坑值得单独点明。第一，`w.grad` 之所以恰好等于输入 `[1, 2, 3]`，是因为 $\partial\text{loss}/\partial w = (\text{pred}_y - 5)\cdot x$，而 `pred_y = 6`、目标值是 `5`、系数 `0.5` 与平方的导数抵消后恰好剩 $1\cdot x$；换一个目标值或系数，这个结果就不再是 `[1, 2, 3]`。第二，只有叶子张量且 `requires_grad=True` 才会默认累积 `.grad`，中间张量如 `pred_y`、`loss` 默认不保留梯度，若想检查它们必须显式 `retain_grad()`——这正是下一章代码里对 `h1`、`h2` 调用 `retain_grad()` 的原因。第三，PyTorch 的 `.grad` 是累加而非覆盖，训练循环里必须手动清零，否则梯度会跨 step 叠加，这也预告了后面 gradient accumulation 要「故意不清零」的用法。这一段最后以「how much compute does gradients take?」收尾，把问题从「梯度是什么」推进到「算梯度要多少 FLOPs」。

## 为梯度计算计数搭台：两层线性网络与 einsum 记法 [60:16 - 61:33]

要数 backward pass（反向传播）的 FLOPs，讲师先搭一个去掉激活的简化两层线性网络：输入 $X$ 是 batch × in（口语里把维度记作 BID），$W_1$ 与 $W_2$ 都是 $D\times D$ 的方阵，暂时 ignore 掉 ReLU 这类 value activation 以简化分析。他特意声明「I'm going to use einsum here for reasons that will become clear」——用 einops 的 `einsum` 而不是裸 `matmul`，是为了让维度的命名在推导梯度时不会混淆，这个动机在下一章会兑现。第一层把 $X$（batch × input dimension）乘以 $W_1$（in × out）得到 batch × out 的 $H_1$，第二层 $H_2$ 用 $H_1$ 和 $W_2$ 做同样的 matmul（矩阵乘法），「these are just matmuls」。

代码把整个计算图写得很干净，画面上高亮的是 `loss.backward()` 这一行：

```python
# Forward pass
h1 = einsum(x, w1, "batch in, in out -> batch out")  # x
h2 = einsum(h1, w2, "batch in, in out -> batch out")  # h1
loss = (h2.mean() - 0)**2  # Regress everything to 0 (arbitrary)

# Backward pass
h1.retain_grad()  # For debugging
h2.retain_grad()  # For debugging
loss.backward()
```

`loss = (h2.mean() - 0)**2` 把一切回归到 0，讲师明说这只是一个 arbitrary（随意设定）的数，目的仅是「get a number out」，让 `backward()` 有一个标量可以求导；因此后面所有关于 FLOPs 的结论都与损失的具体形式无关，只取决于网络里的矩阵乘法。`h1.retain_grad()` 与 `h2.retain_grad()` 标注 `# For debugging`，是为了让中间激活的梯度在反向之后仍然可读，从而能用 `torch.allclose` 做 sanity check（健全性校验）；若漏掉这两句，`h1.grad`、`h2.grad` 会是 `None`，后面的断言直接失败。

![两层线性网络的前向 einsum、loss 与 retain_grad/backward 代码](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0050_01-01-31.jpg)

画面底部灰显的标题已经切到「Zoom in on one layer」，并给出 `Let's focus on the second layer h1 ~w2~> h2.`：forward pass 的 FLOPs 回忆为 `num_forward_flops = 2 * B * D * D`（标注 bf16），接着抛出本段的核心问题「Backward pass: How many FLOPs is running the backward pass?」。这里的 $2$ 有一个非常容易记错的来历：讲师先顺口说「BF16 that's two bytes」，随即自我更正「sorry, that's irrelevant」，真正的 $2$ 来自「it's just two because it's an addition and a multiplication」——即每个 $(i,j,k)$ 三元组贡献一次乘法和一次加法，与 BF16 每元素 2 bytes 完全无关。把「精度字节数」和「乘加次数」混为一谈，是初学 FLOPs 计数时最典型的错误。

再补两条画面上的细节。代码框顶部还能看到权重与输入的创建行：`w2 = torch.randn(D, D, device=cuda_if_available(), requires_grad=True)`，与之对应的 `w1` 同样是 $D\times D$ 且带 `requires_grad=True`；只有显式标记了 `requires_grad` 的叶子张量，PyTorch 才会在 autograd 图里为它分配 `.grad` 缓冲。讲师在这里抛出的问题是「how many floating point operations was that?」，也就是 `loss.backward()` 这一次调用到底消耗了多少 FLOPs；为了把答案算清楚，他先把整个计算压缩成「两次 matmul + 一个标量损失」，并明确「we're going to sort of ignore this value for now just for simplicity」，即暂时忽略 ReLU 之类的激活，因为逐元素操作在足够大的矩阵面前 FLOPs 占比很小。这种「先剥离次要项、只数主导项」的手法，与前面章节里用算术强度判断 matmul 主导计算的思路是一致的。这一章的作用是把计数对象收敛到单个 $H_1 \xrightarrow{W_2} H_2$ 的线性层上，下一章再逐项拆开它的两个梯度。

## 放大第二层：前向 FLOPs 与反向传播的两个梯度 [61:33 - 64:38]

把镜头对准第二层：它只做一件事，$H_2 = H_1 \cdot W_2$，输入是 $H_1$（batch × in），权重是 $W_2$（in × out），输出 $H_2$（batch × out）。forward pass 的 FLOPs 就是把三个维度乘起来再乘 2，即 `num_forward_flops = 2 * B * D * D`；画面上的输出框给出具体数值 `num_forward_flops = 134,217,728`（也就是 $2^{27}$），让抽象公式有了一个可核对的量级。backward pass 按 chain rule（链式法则）和 backprop 算法必须算出两个东西：`h1.grad = d loss / d h1`（对输入的梯度，也就是往回传的 backward message）和 `w2.grad = d loss / d w2`（对参数的梯度）。$H_2.\text{grad}$ 是 batch × out 的矩阵，$W_2$ 是 in × out 的矩阵，两者相乘并沿 out 维求和得到 $H_1.\text{grad}$（batch × in）。

用 einsum 写出来，两条梯度各自只是一次不同聚合维度的矩阵乘法，画面里高亮了第二条：

```python
h1_grad = einsum(h2.grad, w2, "batch out, in out -> batch in")
assert torch.allclose(h1.grad, h1_grad)

w2_grad = einsum(h2.grad, h1, "batch out, batch in -> in out")
assert torch.allclose(w2.grad, w2_grad)
```

第一条对 `out` 维求和（因为它不出现在输出 pattern 里），得到传回上一层的消息；第二条对 `batch` 维求和，把同一个 backward message 与「另一个不对其求导的量」$H_1$ 相乘，得到参数梯度。两条 `assert torch.allclose(...)` 把「手写 einsum 的结果」与 `loss.backward()` 自动算出的 `.grad` 对齐，是讲师反复强调的 sanity check：`H1.grad` 确实等于这里写出的表达式。这里也解释了为什么当初要用 einsum——讲师坦言「I always get this confused … one of them has a transpose and I always forget which order to put them in」，而 einsum 让顺序变得透明：只要记住标量情形下 $H_1.\text{grad} = H_2.\text{grad}\times W_2$，剩下的只是按命名维度（batch/in/out）配好下标，转置与否由 pattern 自动决定，不必手推谁该 `.T`。

![放大第二层后 h1_grad 与 w2_grad 的 einsum 写法及断言](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0052_01-04-16.jpg)

画面底部还露出汇总式 `num_backward_flops = (2 * B * D * D) + (2 * B * D * D)`：两条梯度各自等价于一次同尺寸的 matmul，所以单层的反向正好是前向的两倍开销。易错点有三处：一是 $W_2$ 在两条梯度里都被用到，但一条搭配 $H_2.\text{grad}$、一条搭配 $H_1$，把「对谁求导」和「与谁相乘」记反就会得到形状错误的张量；二是若没有对 $H_1$、$H_2$ 调 `retain_grad()`，`h1.grad` 与 `h2.grad` 为 `None`，两条 `allclose` 无从谈起；三是这个「两倍」目前只对单个 $W_2$ 成立，下一章要论证它对网络里所有参数都成立，才能汇总成全局的 $6ND$。

从形状上再核对一遍有助于消除直觉上的混乱：$H_2.\text{grad}$ 是 batch × out，$W_2$ 是 in × out，第一条 einsum 的 pattern `"batch out, in out -> batch in"` 里 `out` 同时出现在两个输入中却不出现在输出中，因此被求和掉，结果恰好是 batch × in 的 $H_1.\text{grad}$；第二条 `"batch out, batch in -> in out"` 则把 `batch` 维求和，得到 in × out 的 $W_2.\text{grad}$，与 $W_2$ 同形，才能直接用于参数更新。讲师的原话是「you index things by basically looking at the shape of these in the named dimensions」，也就是说 einsum 的命名维度本身就是一份可执行的形状说明书。画面上另外还标出 `num_forward_flops = 134,217,728` 这个具体数值，以及 "Forward pass: Recall the number of forward FLOPs (bf16)" 的标题，说明本章所有推导都锚定在同一个 $2\times B\times D\times D$ 的基准上，前后 FLOPs 可以直接相除得到「两倍」的结论。

## 汇总全部层：6ND 训练公式与 Transformer 的修正 [64:38 - 66:48]

把单层的观察推广开，关键在于一条很干净的直觉：矩阵乘法的 FLOPs 就是「所有维度的乘积」，「It doesn't matter which ones you're batching or not」——无论你沿 $I$、$J$、$K$ 中的哪一个维度做累加，枚举的三元组总数不变，只是聚合方式不同，所以 FLOPs 相同。于是 backward pass 恰好是 forward pass 的两倍昂贵，原因不是别的，就是必须算两个梯度：一个对每个参数（weight）求导，一个对输入（那个「不是参数的另一个量」）求导。讲师在此短暂停顿留出提问时间，然后把这个结论 apply 到网络中所有参数（$W_1$、$W_2$、…）上。

汇总后的账本是全篇最常被引用的公式。设 $N$ 为 number of parameters（参数量）、$D$ 在这里指 number of data points（数据点/token 数，讲师用 $B$ 表示），则：

$$
\text{forward FLOPs} = 2 \times B \times N
$$

$$
\text{backward FLOPs} = 4 \times B \times N
$$

$$
\text{total FLOPs} = (2 + 4) \times B \times N = 6ND
$$

讲师原话是「the grand total is 2 + 4 is 6 … this is where the 6 ND comes from that you might have seen in various places」，也就是把课程开头 back-of-envelope（粗略估算）里直接拿来算 70B 模型、15T token、1024 张加速卡训练时间的那个 $6ND$，在这里彻底 demystified（去神秘化）：$6$ 就是 forward 的 $2$ 加 backward 的 $4$，没有别的玄机。结合本章开头给出的参数量 $\text{num\_parameters} = D\times D\times L$（此处 $D$ 是特征维度、$L$ 是层数），$N$ 与本段第一个代码框里的 $(D*D)*L$ 是同一个量，符号重载需要靠上下文区分。

这个近似并非无条件成立。讲师明确指出，它是针对这类「deep linear/MLP 网络」数出来的，「it turns out that this is actually a good approximation for transformers as well as long as the context length isn't too large」；一旦 context length 很大，attention 会引入 context length 平方级的额外 FLOPs，而这部分「isn't in this kind of accounting」。这正是本公式最大的局限：它默认计算被矩阵乘法主导、且没有随序列长度平方增长的项。此外 $B$ 应理解为「一个训练 step 里的数据点/token 数」而非整个数据集规模——讲师在片尾总结时还专门补了一句「actually if this is training step, this should be batch size」，把 $6ND$ 里的 $N$ 从「data points」修正为「batch size」，这是极易被忽略的口径问题。

「两倍」的论证本身也值得拆开看：单层里两条梯度各自需要遍历 $B\times D\times D$ 个三元组，每条 $2B D^2$ FLOPs，合计 $4B D^2$；而前向只有一次 $2B D^2$。把 $W_2$ 的结论「apply it to all parameters in network」（$W_1$、$W_2$ 以及更深网络里的每一层）之后，所有层的参数梯度与输入梯度都遵循同样的计数，因此全局比例不变。讲师还特意说「we did this for just these deep networks」，提醒听众这个 $6ND$ 是从 MLP 型网络数出来的，只是「a good approximation」而非 Transformer 的精确账本；一旦 attention 的 $O(\text{context}^2)$ 项随序列长度膨胀，额外的 FLOPs 就落在了这个框架之外，需要在作业里更仔细地重新推导。与前后内容的关系也很清楚：本章把前几章的 FLOPs 计数闭环，同时为下一段「内存账本」和「为什么需要降低显存」提供了动机——算得清 FLOPs 之后，还要算得清内存。

## 优化器状态与 AdaGrad：训练的另一半内存 [66:48 - 69:25]

有了梯度，training（训练）的另一半是 optimization（优化）。讲师刻意避开 assignment 1 要实现的 Adam，改用 AdaGrad（自适应梯度，2011 年提出、早于 Adam），理由是「so that I'm not just giving you what's in the assignment one」。他把优化器谱系压缩成一句话：SGD 是基准，momentum（动量）看梯度的 first moment（一阶矩），AdaGrad 看 second moments（二阶矩、梯度平方），Adam 把两者 combine 起来；幻灯片上的四条 bullet 把这个关系写得更细：`momentum = SGD + exponential averaging of grad`、`AdaGrad = SGD + averaging by grad^2`、`RMSProp = AdaGrad + exponentially averaging of grad^2`、`Adam = RMSProp + momentum`，并注明出处 `AdaGrad [Duchi+ 2011]`。这些层级关系是画面独有、转写里没有完整念出的内容。

实例化代码给出了本例的具体超参：`B = 2`（batch size）、`D = 4`、`L = 3`，`model = DeepNetwork(dim=D, num_layers=L).to(cuda_if_available())`，随后 `optimizer = AdaGrad(model.parameters(), lr=0.01)` 与 `state = model.state_dict()`；计算梯度时用 `x = torch.randn(B, D, device=cuda_if_available())`、`y = torch.tensor([4., 5.], device=cuda_if_available())`、`pred_y = model(x).mean()`、`loss = F.mse_loss(input=pred_y, target=y)`。注意这里的 $D=4$、$L=3$ 与本段开头的 $D=8$、$L=3$ 不是同一组设定，属于为讲解优化器重新起的小例子，读代码时不要混用数字。

![AdaGrad 优化器谱系 bullet、B=2/D=4/L=3 与 lr=0.01 的实例化代码](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0054_01-07-02.jpg)

真正的新概念是 optimizer state（优化器状态）：对每个 parameter group（参数组，本例里就是 `W1` 或 `W2`），优化器在运行时需要额外存储。AdaGrad 存的是「梯度平方的累加和」——从 state 里取出、用当前梯度更新、再写回，然后才更新参数。代码把这三步写得非常直白：

```python
class AdaGrad(Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01):
        super(AdaGrad, self).__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                # Optimizer state
                state = self.state[p]
                grad = p.grad.data

                # Get squared gradients g2 = sum_{i<t} g_i^2
                g2 = state.get("g2", torch.zeros_like(grad))

                # Update optimizer state
                g2 += torch.square(grad)
                state["g2"] = g2

                # Update parameters
                p.data -= lr * grad / torch.sqrt(g2 + 1e-5)
```

其中 `g2 = state.get("g2", torch.zeros_like(grad))` 保证第一次 step 有零初值，`g2 += torch.square(grad)` 累加二阶矩，`state["g2"] = g2` 写回，最后 `p.data -= lr * grad / torch.sqrt(g2 + 1e-5)` 除以梯度平方和的平方根（讲师先说「average gradient squared」，随即自我更正为「or sum of gradient squared rather」，即累加和而非平均），`1e-5` 是防除零的数值稳定项。易错点包括：`self.state[p]` 以参数张量本身为键，参数组与单个参数两层循环不能漏；`p.grad.data` 与 `p.data` 的原地写法绕过了 autograd，属于手写优化器的惯例但不宜在普通模型代码里模仿；assignment 1 要求实现的是 Adam 而非 AdaGrad，二者的 state 大小不同（见下一章）。关于 optimizer state 的作用，讲师的口头解释值得逐句对齐代码：他说「there's something called the optimizer state which is storage that the optimizer uses while it's running」，在 AdaGrad 里算的就是「the squared gradients … the sum of the squared of gradients」；流程是「here we're getting it from the optimizer state, we're updating it with the current gradient and we're storing it back」，也就是先取、再累加、再写回，「and then after you update the G2, then you update the parameters」。他还点明「in the assignment one, you're going to implement Adam」，且参数组「in this case W1 or W2」——本例只有两组权重，所以 `self.param_groups` 的循环很短，但真实模型里参数组可以有很多个。另一个容易忽略的点是更新式分母：讲师先说「divide by the square root of the … average gradient squared」，马上改口「or sum of gradient squared rather」，代码里也确实是 `g2 += torch.square(grad)` 的累加和而非滑动平均，这与 RMSProp 的指数平均二阶矩并不相同。

讲师随后略过了 training loop（「I'm going to skip the training loop. This is just a sort of a general kind of review」），直接转入「how much memory the optimizer is using」。

## 训练的内存账本与单步计算量汇总 [69:25 - 72:09]

这一章把训练时要驻留 HBM 的每一样东西都换算成 bytes。幻灯片标题 `Memory` 下逐项列出（注释统一标注 `(2 bytes for bf16)` 或 `(4 bytes for fp32)`）：

```python
# Parameters
parameter_memory = 2 * (D * D * L)  # (2 bytes for bf16)

# Activations
activation_memory = 2 * B * D * L  # (2 bytes for bf16)

# Gradients
gradient_memory = 2 * parameter_memory  # (2 bytes for bf16)

# Optimizer states
optimizer_state_memory = 4 * parameter_memory  # (4 bytes for fp32)
```

直觉是：参数有 $D^2\times L$ 个、BF16 下每参数 2 bytes；每层都有一份 $B\times D$ 的 activation，共 $L$ 层，故 $2\times B\times D\times L$；gradients 是参数的一份拷贝；optimizer state 按每参数 4 bytes 计。画面下方的说明文字点出原因：`Customary to use fp32 for stability (accumulating averages over powers over many steps)`、`4 bytes/parameter for storing second moments, Adam requires 8 bytes/parameter for storing first and second moments`——即 AdaGrad 只存二阶矩（4 bytes/参数），Adam 要同时存一阶矩和二阶矩（8 bytes/参数）。

![Memory 幻灯片：parameters、activations、gradients、optimizer states 四项字节数公式](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0056_01-10-09.jpg)

这里藏着一个必须点破的「图上笔误」。讲师在讲到 optimizer state 时当场打断自己：「this should be sorry, not this I'll fix this as a typo. This should be the number of parameters, not parameter memory」，并重申「So, number of parameters times two, and then this is four times the number of parameters」。也就是说幻灯片里 `gradient_memory = 2 * parameter_memory` 与 `optimizer_state_memory = 4 * parameter_memory` 的写法把「参数个数」误写成了「参数内存」，正确口径应为 $2\times\text{num\_parameters}$ 和 $4\times\text{num\_parameters}$（单位 bytes）；否则按幻灯片字面算，梯度会变成 $4D^2L$、优化器状态变成 $8D^2L$，与「每参数 2/4 bytes」的口头解释自相矛盾。为什么优化器状态坚持 FP32？讲师解释：人们试过 BF16，但「now you're taking squares and you're averaging all those steps, and it doesn't … it's not very stable」，平方与多步累加会放大低精度的误差，因此出于 stability（稳定性）惯例用 FP32。由此 optimizer state 往往是显存里很大的一块。

但讲师紧接着给出一个重要的辨析：memory 有两个用途，一是「how much you have to store this thing in your HBM」，二是数据「has to be shipped to the accelerators」；而「the optimizer state is not really the bottleneck for compute」，所以这部分内存「is not really so important for performance in terms of speed, but it … means that you can't fit large models in your memory」。换句话说，优化器状态主要卡的是容量（能不能放下大模型），不是速度瓶颈——这与前面 roofline（屋顶线）分析里「memory bound 影响速度」是两件事，不可混淆。最后 `Putting it all together` 给出 $`\text{total\_memory} = \text{parameter\_memory} + \text{activation\_memory} + \text{gradient\_memory} + \text{optimizer\_state\_memory}`$，而 `Compute (for one step)` 一栏写着 $`\text{num\_parameters} = D * D * L`$ 与 $`\text{flops} = 6 * B * \text{num\_parameters}`$，把上一章的 $6ND$ 与本章的内存账本并排收束；画面还注明 `The accounting for a Transformer is more complicated, but the same idea. Assignment 1 will ask you to do that.`，并附了两篇 `[article]` 参考（分别讲 Transformer 训练内存与 FLOPs）。把口头的字节数换算再展开一遍：讲师说「for parameters in this network, there's D squared parameters for each of the L layers, and each parameter takes two bytes」；activations 是「two times … batch times D times the number of layers for every layer you have an activation」；gradients「is basically a copy of all the parameters」；而 optimizer state 按「customary to use FP32 for the optimizer states for stability reasons」计 4 bytes/参数，Adam 因为「you store the first order bit and the second order moments」要 8 bytes。他由此感慨「the optimizer state is actually a lot of the memory used」。画面中 `Putting it all together` 的 `total_memory = parameter_memory + activation_memory + gradient_memory + optimizer_state_memory` 与 `Compute (for one step)` 的 `num_parameters = D * D * L`、`flops = 6 * B * num_parameters` 是并排给出的，正好呼应上一章的 $6ND$：同一组 $D$、$L$、$B$ 既决定内存、也决定计算。`Transformers` 一节还写着 `The accounting for a Transformer is more complicated, but the same idea. Assignment 1 will ask you to do that.`，并列出两篇 `[article]` 分别讨论 Transformer 训练的内存与 FLOPs，提示这里的公式只是入门级框架。易错点：activation_memory 随 $B$ 线性增长，是下一章一切省显存手段的直接靶子；而参数、梯度、优化器状态三项与 batch size 无关。

## 降低显存：梯度累积、激活检查点与全课总结 [72:09 - 77:24]

收尾前讲师点出两个「typically do」的省显存手段，动机是 memory 既影响能否存下大模型、有时也影响速度。第一个是 gradient accumulation（梯度累积）：训练时希望 batch size 足够大以改善稳定性，直到一个 critical batch size（临界 batch，Tatsu 后续课程会讲），但上一章已看到 activation memory 随 batch size 线性增长，batch 太大就会 OOM。做法是把大 batch 拆成 micro batches（微批），逐个算梯度并累加、「don't zero out the gradients」，每累计 `batch_size / micro_batch_size` 个 step 才更新一次参数并清零。幻灯片把这三条写成 bullet：`Compute gradient on micro batches`、`Accumulate the gradients (don't zero it out)`、`Every batch_size / micro_batch_size steps, update the parameters and zero out the gradients`，并给出简化公式 $`\text{activation\_memory} = B * L * D`$ 与 $`\text{activation\_memory} = \text{micro\_batch\_size} * L * D`$，其中具体数字是 `micro_batch_size = 256`、`B = 64  # Batch size`。讲师强调「this is actually a very simple code change」，只需改清零时机；但要注意它省的是峰值 activation 显存、并不减少总 FLOPs，且有效 batch 不变、梯度在数值上等价于一次性大 batch（前提是损失对样本取均值）。

第二个是 activation checkpointing（激活检查点），也叫 gradient checkpointing 或 rematerialization（重计算/重物质化）。默认训练要存下所有层的 activations；有意思的是 inference（推理）不计算梯度，只需存当前层的激活。训练侧的内存约为 $B\times D\times 2\times L$（幻灯片简化写成 $B\times L\times D$），于是问题变成「can you reduce this?」。核心思想是 trade-off（取舍）：forward pass 只保留部分层的 activations，backward pass 时从最近的 checkpoint 重新计算缺失的激活——「if you want to reduce memory, you can just recompute things」，这是系统领域通用的以计算换内存技巧。落地方式极简：在同样的模型上给某一层加 `torch.utils.checkpoint`，含义是「do this computation, but don't store any of the intermediate activations. Only store what is needed」。深层网络每个 block 是 linear + ReLU，本来要同时存 pre-ReLU 与 post-ReLU 两份；对 block 做 checkpoint 后不存 pre-ReLU，「you save basically half … you can get away with half the memory」；反向需要 $G_3$ 时，可以「compute G3 easily from H2」。

![激活检查点：store all/no/some layers 示意与 O(L)、O(L^2)、O(sqrt(L)) 三种 checkpoint 频率](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0059_01-14-59.jpg)

还能走得更远。幻灯片用一行竖线标出三种策略：`# Store all layers: | h1 h2 h3 h4 h5 h6 h7 h8 h9 |`、`# Store no layers: | |`、`# Store some layers: | h3  h6  h9 |`，并在 `How frequently to checkpoint?` 下给出三条结论：存每层激活则 activation memory 是 $O(L)$、无重计算；一层不存则内存 $O(1)$、但计算变成 $O(L^2)$（每层都要从开头重算）；若每隔 $\sqrt{L}$ 层存一个 checkpoint，则 activation memory 是 $O(\sqrt{L})$、重计算开销也是 $O(\sqrt{L})$，「balanced」。画面里的验证代码说明这是同一个网络只改了存储策略的对照实验：

```python
model = DeepNetworkCheckpointed(dim=D, num_layers=L).to(cuda_if_available())
checkpointed_memory = get_max_memory_usage(lambda: model(x).backward())

class DeepNetworkCheckpointed(nn.Module):
    """Same as DeepNetwork, but with activation checkpointing."""
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
```

`get_max_memory_usage(lambda: model(x).backward())` 把「前向 + 反向」整段包进回调里测峰值显存，正好用来对比普通 `DeepNetwork` 与检查点版本的差距。易错点：极端省内存会付 $O(L^2)$ 的算力代价，$\sqrt{L}$ 折中只在「内存与重算等权」时最优；checkpoint 粒度太细（每层都存）则完全失去省显存的意义。

最后是全课 summary（总结）。讲师归纳：一切都是对 tensors（张量）的操作——parameters、gradients、activations、optimizer states、data；einops 是「a way to think about tensor operations」；$6\times\text{data points}\times\text{parameters}$ 已被 demystified 为 FLOPs per training step（并补正「if this is training step, this should be batch size」）；arithmetic intensity（算术强度）与 roofline analysis（屋顶线分析）用来诊断 computation 是 memory bound（内存瓶颈）还是 compute bound（计算瓶颈），其中 matrix multiplications are compute-bound、basically everything else is memory-bound；gradient accumulation 与 activation checkpointing 则通过 reduce memory 换取 bigger batch sizes。这些条目与幻灯片 `Summary` 页逐条对应：`Everything is operations on tensors (parameters, gradients, activations, optimizer states, data)`、`einops: better way to think about tensor operations`、`6 (# data points) (# parameters) FLOPs per training step`、`Arithmetic intensity / roofline analysis: compute-bound or memory-bound?`、`Matrix multiplications are compute-bound, elementwise operations are memory-bound`、`Gradient accumulation, activation checkpointing: reduce memory to use bigger batch sizes`。

![片尾 Summary 幻灯片：tensor、einops、6ND、roofline 与两种省显存技巧的总结条目](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0060_01-16-59.jpg)

同一页还列出了本讲全部示例代码的函数清单：`# Full example` 下有 `deep_linear_network()`、`gradients_basics()`、`gradients_flops()`、`optimizer()`、`train_loop()`，`# More memory optimizations` 下有 `gradient_accumulation()`、`activation_checkpointing()`——这正好把本段的四个知识模块（深层网络与梯度、FLOPs 计数、优化器与内存、省显存技巧）串成一条完整链路。清单把两种省显存技巧列成独立可运行的函数（`gradient_accumulation()` 与 `activation_checkpointing()`），而不是混进 `train_loop()` 的一团逻辑，这种「一个概念一个函数」的组织方式与整门课强调的「可核算、可复现」风格一致。讲师以「that's it for today's lecture」结束，并预告「next week Tatsu will talk about architectures」。需要提醒的局限是：整讲的核算都建立在「矩阵乘法主导、context length 不大、BF16 训练 + FP32 优化器状态」这些前提上，Transformer 场景要在 assignment 1 里做更细致的推广，不能把 $6ND$ 与这里的内存账本当成无条件精确的公式。

## 要点回顾

### 资源核算与开场的两个估算
- 课程目标是在固定 compute、memory 下最大化计算效率；前提是"先能算清一次给定计算消耗多少算力与内存"，data 在本课不成为瓶颈。
- 训练 70B 参数、15T tokens、1024 张 H100 的时长：`total_flops = 6 * 70e9 * 15e12 = 6.3e24`，除以 `flops_per_day = h100_flop_per_sec * mfu * 1024 * 86400`（mfu 取 0.5）得约 143.9 天。
- 8 张 H100 用 AdamW 朴素训练的最大模型：`bytes_per_parameter = 2 + 2 + (4 + 4) = 12`，`N_max = 80e9 * 8 / 12 ≈ 5.33×10^10`（约 530 亿），且未计入 activations，只是 back-of-envelope 量级。
- 开场 iso-flops 曲线先拟合 scaling law 再实跑，实测 loss 与预测相差在 0.05 以内；外推到 GPT-5 级算力时讲师用 "your mileage might vary" 划定了可信边界。
- 讲师把要带走的知识分三层：Mechanics 是直白的 PyTorch 与 tensor 语义（讲师说「There's no magic here」），Mindset 是每写一行代码都习惯性去想性能特征，Intuitions 是资源花在哪里。两个估算问题分别对应 compute 与 memory 两条主线，后文从精度一路推到 FLOPs 正是把它们逐一兑现。

### 张量内存与数值精度
- 最基础的账是 `memory = numel * element_size`：`4×8` 的 float32（默认 dtype）矩阵占 `4*8*4 = 128 bytes`，GPT-3 前馈层单个 `(12288*4)×12288` 矩阵约 2.3 GB。
- float32 是 1 符号位 + 8 指数位 + 23 尾数位、每元素 4 bytes；float16 把指数砍到 5 位，`torch.tensor([1e-8], dtype=torch.float16)` 会 `assert x == 0`，即 Underflow。
- bfloat16 保持 8 位指数、7 位尾数，同样 `1e-8` 换成 `torch.bfloat16` 就是 `assert x != 0`，动态范围与 float32 相同、内存与 float16 相同，代价是分辨率更差。
- 混合精度：参数、activations、gradients 用 BF16，optimizer states 用 FP32，PyTorch AMP 在 "matmuls, not exp" 这类安全处自动 cast；`torch.finfo` 可查各类型 dynamic range。
- 更低精度里 H100 支持 FP8 的 E4M3（范围 `[-448, 448]`）与 E5M2（`[-57344, 57344]`）；NVFP4 只有 4 bits、16 个非均匀取值，需 per-block scale factor，Nemotron 3 Super 即用其训练。

### einops：用命名维度写张量运算
- 传统 `x @ y.transpose(-2, -1)` 要自己推断 `-2`、`-1` 指哪根轴；einsum 写成 `z = einsum(x, y, "seq1 hidden, hidden seq2 -> seq1 seq2")`，未出现在输出里的维度一律被求和掉。
- batch 版 `"batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2"` 靠命名吸收了转置；再用 `...` 可匹配任意多个前导广播维，写出不必关心输入形状的 modular code。
- `reduce(x, "... hidden -> ...", "sum")` 把 sum/mean/max/min 统一成命名归约，聚合类型须由第三参数单独给出；`reduce` 复用了 einsum 的求和约定。
- `rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)` 把压平的 `total_hidden=8` 拆成 heads=2、hidden1=4，括号决定合并顺序；einsum、reduce、rearrange 都只是表达层语法糖，不改变底层 FLOPs。
- pattern 里维度顺序必须与张量实际轴顺序严格一致，且同名维度在不同输入中长度必须相等（`hidden` 在 x 与 y 里都须为 4），否则不报错却语义全错。命名维度把原先藏在 `@` 运算符里的隐式约定外显成可读字符串，因此讲师称它是 generalized matrix multiplication with good bookkeeping。

### FLOPs 计数、计时与 MFU
- FLOPs（小写 s）衡量计算量、FLOP/s 衡量硬件吞吐；H100 标称 1979 teraFLOP/s 带 sparsity，dense 场景要取 `h100_flop_per_sec = 1979e12 / 2`。
- 线性层 `X(B×D) @ W(D×K)` 的 `actual_num_flops = 2 * B * D * K`，因为每个 `(i,j,k)` 三元组一次乘法加一次加法；前向还可写成 `2 × tokens × parameters`。
- GPU 异步执行，计时须在操作前后放 `torch.cuda.synchronize()`，再算 `actual_flop_per_sec = actual_num_flops / actual_time`，并多次取平均。
- `MFU = actual_flop_per_sec / promised_flop_per_sec`，≥ 0.5 已属不错、纯 matmul 或达 0.8、0.1 则说明有问题；`promised_flop_per_sec` 必须按 `x.dtype` 取且用 dense 值，分子分母精度要对齐。
- 量级锚点：GPT-3 训练约 `3.14e23` FLOPs、GPT-4 推测 `2e25` FLOPs，而 8 张 H100 跑一周约 `5e21` FLOPs。这些数字只是讲师口中 nothing fancy, it's just math and napkin math 式的量级直觉，刻意忽略通信与其它开销，不能当精确预算。

### 算术强度、roofline 与瓶颈
- 硬件卡通图为 HBM 加计算核心：`h100_bytes_per_sec = 3.35e12`；假设通信与计算完美重叠时 `total_time = max(communication_time, computation_time)`。
- 加速器强度 `I_hw = FLOP/s / bytes/s`，H100 为 `295.3731`（约 300）；算法算术强度 `I_alg = flops / bytes`，`I_alg < I_hw` 即 memory bound，反之 compute bound。
- ReLU 读 `2N` 写 `2N` 共 `4N` bytes、约 `N` 次比较，`I_alg = 0.25`；GELU 按 `20 * n` flops 估出 `I_alg = 5`，二者离 295 仍差两个数量级，依旧 memory bound。
- 两个 `N×N` 矩阵相乘搬 `6N^2` bytes、算 `2N^3 - N^2` flops，`I_alg = (2N-1)/6 ≈ N/3`，n=1024 时 `341.1667 > 295.3731`，终于 compute bound；矩阵越大强度越高。
- roofline 图横轴算术强度、纵轴实际 FLOP/s；训练是大矩阵乘、compute bound，推理逐 token 生成更像 matrix-vector、memory bound，这也解释了 MFU 常只有 0.5。

### 6ND 训练公式、内存账本与省显存
- 深层网络 `L` 层、每层 `D×D`，`num_parameters = D*D*L`；backward 要算 `h1.grad = d loss / d h1` 与 `w2.grad = d loss / d w2` 两个梯度，各等价于一次 matmul，恰为 forward 两倍。
- 汇总得 `forward FLOPs = 2BN`、`backward FLOPs = 4BN`、`total FLOPs = (2+4)BN = 6ND`；它对 Transformer 是一阶近似，context length 大时 attention 的平方项不在此账内。
- 内存四项：`parameter_memory`、`gradient_memory`、`activation_memory = 2*B*D*L`、`optimizer_state_memory`；AdaGrad 二阶矩 4 bytes/参数、Adam 一阶加二阶 8 bytes/参数，惯例用 FP32 保证 stability。
- 优化器状态主要卡容量而非速度；`total_memory` 为四项之和，单步 `flops = 6 * B * num_parameters`，同一组 `D`、`L`、`B` 既决定内存也决定计算。
- gradient accumulation 用 micro batches 累加梯度、每 `batch_size / micro_batch_size` 步才更新并清零；activation checkpointing 以重算换显存，存全部 / 全不存 / 每 √L 层存分别对应 O(L) / O(L²) / O(√L)。

## 视频来源

- 原始链接：https://www.youtube.com/watch?v=kuYAsz7zspQ
- 课程：Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 2

---

## 对照参考笔记补遗（Spring 2025 版）

> 编者注：本节依据 2025 版参考笔记整理，其中的「更正」小节记录的是核对时发现的问题，对应正文已在本版订正；补遗内容不代表 2026 课程视频原文。


本节内容整理自 Qihong Ruan 汇编的 Spring 2025 版 lecture notes（由 MinerU 从 PDF 解析为 Markdown），仅用于与本篇 Spring 2026 视频笔记对照补充 2025 版讲过、而本讲未展开或未提及的知识点，不代表 2026 视频原文，也不用于订正本篇正文。

### 参数初始化：Xavier 缩放、梯度爆炸与截断

本篇从算术强度与 roofline 直接切入深层网络，把参数当成「已经摆在那里的 $D\times D$ 矩阵」来数 FLOPs 与内存，于是略过了「这些参数最初从哪来、初值怎么定」这一环。2025 版在 Module Parameters 一节把这一环补上了：在 PyTorch 里，可训练参数是 `nn.Parameter` 对象，而 **parameter initialization is crucial for training stability**（参数初始化对训练稳定性至关重要）。若把初始权重取得过大，前向激活会逐层放大、反向梯度同样逐层放大，出现 **gradients blow up**（梯度爆炸），训练很快发散。这与本篇反复强调的「数值稳定性」是两件互补而非同一件事：本讲讨论的 BF16 与 FP32 取舍讲的是「存与算」的稳定，初始化讲的是「起点」的稳定。

最常用的做法是 **Xavier initialization**：把初始权重按 $1/\sqrt{\text{num\_inputs}}$ 缩放，使每一层输出的方差不随输入维度漂移；对 $D\times D$ 的方阵而言，等价于整体除以 $\sqrt{D}$。为更稳妥，还会把权重截断到一个有限区间，例如 $[-3, 3]$，对应的底层调用是截断正态分布 `nn.init.trunc_normal_(x, mean=0, std=1, a=-2, b=2)`。这里有两个易错点：`num_inputs` 指的是该层的输入维度而不是整个网络的参数总数，取错对象会把缩放因子算偏；而 $[-3, 3]$ 与 `a=-2, b=2` 是 2025 笔记两处不同例子里的两组数，并非同一个截断区间的两种写法，别合并记忆。

把这一节放回全篇的位置，它正好落在本篇 `num_parameters = D*D*L` 之前：先有合理的初始化，才有可训练的参数；也正因为它保证了起点稳定，后面混合精度里「优化器状态坚持 FP32 以抵抗多步平方累加误差」的担忧才显得有的放矢。2025 版把它安排在 Models 一节的开头、作为从张量走向模型的第一个动作，本篇视频未展开，故在此补出，读时可把它视为 $D^2L$ 那条参数量公式的前置注脚。

### 张量的底层存储、步幅与视图

本篇在「从内存到计算」一章只讲到张量默认创建在 CPU 上、要用 `.device` 确认位置、并经 PCI BUS 搬到 GPU 的 `DRAM` 才能被 streaming multiprocessor 计算，配套的是 $\text{memory}=\text{numel}\times\text{element\_size}$ 这条容量账。2025 版在 Tensors on GPUs 一节往下多走了一层：PyTorch 的 tensor 本质是一个指向已分配内存的 **pointer**（指针）加一份 **metadata**（元数据），元数据描述如何定位到任意元素；一个 2D 矩阵在底层其实被存成一维连续的 **underlying storage**（底层存储）。

定位靠的是 **stride**（步幅）：沿某一维移动到下一个元素时，在一维 storage 里需要跳过几个元素。对一个 row-major 的 $4\times4$ 矩阵，`stride(0)=4`（换一行跳 4 个）、`stride(1)=1`（换一列跳 1 个），任意元素 $(r,c)$ 的位置由 $\text{index}=r\cdot\text{stride}(0)+c\cdot\text{stride}(1)$ 给出，取 $r=1,\ c=2$ 便得 $\text{index}=6$。切片（slicing）如 `y = x[0]` 并不拷贝数据，只返回同一块 storage 的另一个 **view**（视图），所以修改 `y` 会连带改动 `x`；view 之所以被称作 "free"，正因为它不分配新内存。反过来，**non-contiguous**（非连续）张量不能直接 view 成任意形状，必须先调用 `.contiguous()`，而这一步会触发一次真正的拷贝。

这一层正好给本篇 rearrange 章那句「row-major 还是 column-major 由 pattern 里维度名字的排列顺序决定」补上底层原因：重排有时只是改写 stride、有时却要真搬数据，差别就在于张量是否连续；把 view 误当成独立副本，是初学者最典型的别名 bug。同节还列出逐元素操作 `pow`、`sqrt`、`x / 2` 都会生成新张量，以及 `triu()` 取矩阵上三角可用于构造 **causal attention mask**（因果注意力掩码，$M[i,j]$ 表示第 $j$ 个位置对第 $i$ 个位置的贡献）——后者是本讲 einops 与 Transformer 讨论里缺的一块拼图，说明「命名维度」之外，传统 PyTorch 也能靠掩码算子表达序列依赖。

### 训练循环的确定性、惰性数据加载与容错检查点

本篇讲者一句 "I'm going to skip the training loop. This is just a sort of a general kind of review" 跳过了训练循环与配套的最佳实践，笔记也如实标注了这一跳过，于是 2025 版占了不少篇幅的这一整块在此缺席。先说 **randomness**（随机性）：它同时出现在参数初始化、dropout 与数据顺序三处；为了 **reproducibility**（可复现性），应给每一处随机性各设一个独立的 seed，并把 torch、numpy、python 的 `random` 三套种子都设上；**determinism**（确定性）在调试时尤其有用。易错点是只调 `torch.manual_seed` 并不够，numpy 与 python `random` 各有各的生成器，漏设其一就无法复现。

再看数据加载。语言模型的数据是 tokenizer 输出的一串整数，工程上常序列化成 numpy 数组；但数据集可以大到根本装不进内存——2025 给的锚点是 **LLaMA 训练数据 2.8 TB**——此时用 `np.memmap` 做惰性映射，把磁盘上的文件当成一个 numpy 数组，只把被访问到的部分载入内存，再由 data loader 生成一个个 batch。这与本篇「优化器状态卡的是容量不是速度」是两类不同的内存问题：前者是数据集大到放不进主机内存，后者是 HBM 放不下大模型，症状相似、对策不同，别混为一谈。

最值得单独辨析的是容错式 **checkpointing**。2025 提醒大模型训练耗时极长、"will certainly crash"，因此要周期性把进度写到磁盘，且必须同时保存 `model.state_dict()` 与 `optimizer.state_dict()`：后者正是本篇内存账本里那一大块 optimizer state，只存模型等于丢掉 AdaGrad 的梯度平方累加或 Adam 的一阶、二阶矩，恢复后训练轨迹会偏。这里的同名异义最容易踩坑：本篇末尾的 **activation checkpointing**（又叫 gradient checkpointing / rematerialization）是「不存中间激活、反向时重算」以省显存，属运行期显存优化；本节的 checkpointing 是「把状态周期性落盘」以抗崩溃，属持久化。两者英文都叫 checkpointing，目的与代价完全不同，读文献时必须按上下文判断指的是哪一个。

### 从墙上时间到 MFU：两个完整的数值算例

本篇已经给出 $\text{MFU}=\text{actual FLOP/s}\,/\,\text{promised FLOP/s}$ 的定义、GPU 异步执行须在前后放 `torch.cuda.synchronize()` 的计时法，以及 ≥0.5 已属不错、纯 matmul 或达 0.8、0.1 则说明有问题的经验判断；但因 2026 在 CPU 上演示、讲者明说这些计时 "not going to be very meaningful"，定义并未被数字走完。2025 版在同一位置补了两个完整算例，恰好把 $\text{actual\_flop\_per\_sec}=\text{actual\_num\_flops}/\text{actual\_time}$ 与 MFU 串成可核对的算术。

先看 float32：一次矩阵乘法实测 `actual_time = 0.163 s`，由此得实测吞吐 $5.4\times10^{13}$ FLOP/s，而 H100 在 float32 下的标称峰值（promised FLOP/s）是 $6.75\times10^{13}$，于是 $\text{MFU}=5.4\times10^{13}/6.75\times10^{13}\approx0.801$，属于相当高的利用率。换成 bfloat16：实测时间降到 `0.032 s`（快约五倍），实测吞吐升到 $2.735\times10^{14}$ FLOP/s，但 H100 的 BF16 标称峰值同步升到 $9.895\times10^{14}$ FLOP/s，于是 $\text{MFU}=2.735\times10^{14}/9.895\times10^{14}\approx0.276$，反而更低。

这里有一个反直觉、也最该记住的结论：低精度让绝对吞吐变高，MFU 却下降，因为分母（promised FLOP/s）涨得比分子更快，2025 的原话是 "the MFU is lower for bfloat16 because the promised FLOP/s is much higher, making it harder to fully utilize"。易错点随之而来：MFU 是「相对本 dtype 峰值」的利用率，跨 dtype 比较 MFU 没有意义，分子与分母必须对应同一种精度、且都取 dense（除以 2 后）的值——这正是本篇强调「`promised_flop_per_sec` 要按 `x.dtype` 取且用 dense 值」的数值侧理由。它也从另一个角度说明「MFU 常停在 0.5」并非硬件失常，而是峰值口径与实际算子混合共同作用的结果。

### 算力账的量级锚点与含偏置的完整参数量公式

本篇 Intuitions 已给出 GPT-3 训练 $3.14\times10^{23}$ FLOPs、GPT-4 推测 $2\times10^{25}$ FLOPs、H100 峰值 `1979 teraFLOP/s with sparsity, 50% without`，以及 `h100_flop_per_sec = 1979e12 / 2`。2025 同一节还有三条常被引用的锚点可一并补入：**A100 峰值 312 TFLOP/s**；美国一项行政令曾要求报告训练算力 $\ge 10^{26}$ FLOPs 的模型（该要求 **now revoked**，现已撤销）；H100 的 `1979` 是 **structured sparsity**（结构化稀疏）口径，即权重中每 4 个元素有 2 个为零（**2:4**），密集场景要减半。最后一条正好给本篇那句 "all you always have to take these numbers divide by two" 补上物理含义：这不是营销注水，而是 2:4 稀疏硬件加速只在满足稀疏模式时成立。配套估算为 8 张 H100 跑两周 $8\times(60\times60\times24\times7\times2)\times\text{H100\_flop\_per\_sec}\approx4.788\times10^{21}$ FLOPs；本篇记录的是讲者现场把 "two weeks" 改口为 "one week"、得约 $5\times10^{21}$，两者是同一量级的不同口径，并列看即可，不必强行统一。易错点是 `1979` 不能套到 FP8 或 FP32 上，`FLOP/s depends heavily on the data type`；$10^{26}$ 是已撤销的历史门槛，别写成现行规定。

最后是参数量的完整式。本篇为讲清训练账本，采用简化版 `num_parameters = D*D*L`（每层一个 $D\times D$ 方阵、明确忽略 bias）。2025 的 **Cruncher** 模型把每层偏置与最终输出层都计入，展开为
$$N = D^{2}\times(\text{num\_layers}) + D\times(\text{num\_layers}) + D + 1,$$
即每层权重 $D^2$ 加每层偏置 $D$，再加输出层 $D$ 个权重与 $1$ 个偏置；本篇的 $D^2L$ 相当于省去了后面三项。与之配套，2025 在全 FP32 假设下把整本内存账写成 $4\times(\text{num\_parameters}+\text{num\_activations}+\text{num\_gradients}+\text{num\_optimizer\_states})$，其中 `num_activations = B × D × num_layers`、`num_gradients = num_parameters`；AdaGrad 只存一份梯度平方和，故 `num_optimizer_states = num_parameters`，Adam 要同时存一阶矩与二阶矩两份，故 `num_optimizer_states = 2 × num_parameters`。

把这些式子与本篇并读时要守住一条界线：它们是 2025 全 FP32（每参数 4 bytes）口径，本篇正文已是「BF16 参数/梯度 + FP32 优化器状态」的 12 bytes/参数混合口径，两套系数不可交叉套用，否则会得出矛盾的最大模型容量。读这一节的目的，是理解偏置项与 optimizer state 份数如何进入公式，而非替换本篇的四项内存账；至于单步 `flops = 6 * B * num_parameters` 与 `6ND` 的推导，本篇已详写，此处不再重复。
