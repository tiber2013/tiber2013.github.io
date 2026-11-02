---
title: "第 2 讲 · PyTorch 与 einops"
description: "本讲把“训练大模型”拆成可核算的资源问题：张量存储由元素数和数值精度决定，计算量用 FLOPs 计数；一次训练 step 的 FLOPs 可近似为 。要判断实际速度，需要比较算法的算术强度与硬件的 FLOP/s 和内存带…"
---

# 第 2 讲 · PyTorch 与 einops

> **Stanford CS336 语言模型从零构建 · L02** ｜ 讲者 Stanford Online ｜ 77 分钟 ｜ [原视频](https://www.youtube.com/watch?v=kuYAsz7zspQ)

## 核心结论

本讲把“训练大模型”拆成可核算的资源问题：张量存储由元素数和数值精度决定，计算量用 FLOPs 计数；一次训练 step 的 FLOPs 可近似为 $6 \times \text{tokens} \times \text{parameters}$。要判断实际速度，需要比较算法的算术强度与硬件的 FLOP/s 和内存带宽之比，从而识别 memory bound 或 compute bound。einops 用命名维度替代容易出错的 transpose/reduce 索引，使张量代码更可读。最后，梯度累积与激活检查点通过牺牲少量计算或改变执行方式降低显存，换取更大 batch 或更大模型。

## 开场：scaling law 预测与课程定位 [00:05 - 01:17]

上次课提到的 scaling law 实验已经跑完，结果与预测吻合；幻灯片显示 Marin/Delphi Scaling Suite 的损失预测与实际结果非常接近。课程由此引出本讲主题：在固定 compute、memory 等资源下，训练尽可能好的模型，也就是最大化计算效率。

上一讲是课程总览与 tokenization；本讲转向更偏系统侧的 resource accounting。要优化效率，首先需要理解一次计算消耗多少 compute 和 memory。

![Delphi Scaling Suite结果图表，展示不同模型规模下的损失曲线](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0001_00-00-14.jpg)

## 资源核算的动机：两个 back-of-envelope 问题 [01:17 - 04:43]

课程用两个估算问题说明资源核算的用途。

**问题一：训练 70B 参数模型、15T token、1024 张加速卡需要多久？**  
按 H100 峰值算力、MFU=0.5 估算：

$$
\text{total\_flops}=6 \times 70\times10^9 \times 15\times10^{12}=6.3\times10^{24}
$$

$$
\text{FLOP/s}_{\text{H100,dense}}=\frac{1979\times10^{12}}{2}
$$

$$
\text{days}
=\frac{\text{total\_flops}}
{\text{FLOP/s}_{\text{H100,dense}}\times 0.5 \times 1024 \times 86400}
\approx 143.9
$$

这里 $6$ 来自前向与反向传播的 FLOPs 计数，后面会解释。

**问题二：在 8 张 H100 上用 AdamW 朴素训练，最大能放多大模型？**  
H100 有 80GB HBM。AdamW 每参数需要：

$$
2\ \text{(BF16 参数)}+2\ \text{(BF16 梯度)}+(4+4)\ \text{(FP32 优化器状态)}=12\ \text{bytes}
$$

因此参数量约为：

$$
N_{\max}=\frac{80\times10^9 \times 8}{12}\approx 53.3\times10^9
$$

这个估算没有计入 activations，而 activations 取决于 batch size 和 sequence length。课程强调：目标不是精确计算每个细节，而是获得 rough shape 和量级直觉。

![课程幻灯片，包含计算最优前沿图表及资源核算主题介绍](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0002_00-01-17.jpg)

![计算训练70B模型所需时间的Python代码及总FLOPs结果](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0003_00-02-35.jpg)

![计算在8张H100上使用AdamW能训练的最大模型参数量的代码及结果](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0004_00-03-52.jpg)

## 张量、内存与数值精度 [04:49 - 17:00]

### 张量是统一存储单元

参数、梯度、优化器状态、数据、activations 本质上都是 tensor。一个模型可以看作一组带 shape 和 precision 的 tensor。例如 DeepSeek-V3.2 的权重文件列表中，可以看到各层权重的形状以及 BF16、F8_E4M3 等精度标注。

tensor 的内存占用很直接：

$$
\text{memory}=\text{numel}\times\text{element\_size}
$$

默认创建的 tensor 是 float32；一个 $4\times 8$ 的 float32 矩阵占 $4\times 8\times 4=128$ bytes。GPT-3 前馈层中的一个矩阵约 2.3 GB，说明大模型 tensor 可以非常大。

![Hugging Face 上 DeepSeek-V3.2 模型文件的张量元数据列表，展示模型各层权重形状与精度](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0005_00-05-10.jpg)

![IEEE 754 标准下 float32 数据类型的 32 位内存布局图（符号位、指数位、尾数位）](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0006_00-06-27.jpg)

![验证张量内存占用的代码，并计算 GPT-3 前馈层单个矩阵的内存大小（2.3 GB）](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0007_00-07-44.jpg)

### float32、float16 与 bfloat16

float32 是 32-bit 浮点数：1 位符号、8 位指数、23 位尾数，也就是单精度。深度学习通常不需要科学计算那么高的精度，因此会使用更低精度来节省内存和计算时间。

float16 有 1 位符号、5 位指数、10 位尾数。它的动态范围较差：例如 `1e-8` 在 float16 中会下溢为 0。用 FP16 训练容易出现 underflow、overflow 和 NaN，训练不稳定。

bfloat16 保持 16 位总宽度，但把更多位分给指数：1 位符号、8 位指数、7 位尾数。它的动态范围与 float32 相同，但分辨率更低。深度学习中很多计算本身带有随机性和容错性，因此这种 trade-off 通常很划算。

| 类型 | 符号位 | 指数位 | 尾数位 | 每元素字节 | 特点 |
|---|---:|---:|---:|---:|---|
| float32 | 1 | 8 | 23 | 4 | 默认精度，动态范围好 |
| float16 | 1 | 5 | 10 | 2 | 动态范围差，易下溢/溢出 |
| bfloat16 | 1 | 8 | 7 | 2 | 动态范围同 float32，分辨率较低 |

![float16 的 IEEE 半精度 16 位浮点格式示意图，展示符号位、5 位指数和 10 位尾数，并附 PyTorch 创建 float16 张量的代码](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0008_00-09-02.jpg)

![bfloat16 的 16 位浮点格式示意图，展示 8 位指数和 7 位尾数，并对比 float16 说明其动态范围与 float32 相同](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0009_00-10-19.jpg)

### 混合精度与更低精度

混合精度训练的一般规则是：参数、activations、gradients 用 BF16；优化器状态用 FP32。PyTorch 提供 AMP 库，会尽量把安全操作 cast 成 BF16，例如 matmul；对 `exp` 这类操作则倾向保持 FP32。

更低精度方面，FP8 已经标准化，并有 E4M3 和 E5M2 两种变体，分别偏向不同动态范围与分辨率需求。FP4 则只有 4 bit per value，需要配合 per-block scale factor 使用；幻灯片提到 Nemotron 3 Super 使用 NVFP4 训练。课程也区分了训练与推理：低比特推理量化通常比从头训练 1-bit 语言模型容易得多。

![混合精度训练幻灯片，介绍 float32/float16/bfloat16 的 torch.finfo 查询及训练稳定性问题](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0011_00-12-07.jpg)

![介绍 FP8 的两种变体（E4M3 和 E5M2）以及 NVIDIA 的 FP4 格式及其具体数值范围。](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0013_00-14-12.jpg)

## 用 einops 表达张量运算 [17:00 - 27:26]

传统写法中，`x @ y.transpose(-2, -1)` 需要读者自己推断 `-2`、`-1` 对应哪个维度，容易混淆。einops 的核心思想是给维度命名，把矩阵乘法写成带 bookkeeping 的广义矩阵乘法。

例如：

```python
z = einsum(x, y, "seq1 hidden, hidden seq2 -> seq1 seq2")
```

这里没有出现在输出中的 `hidden` 会被求和掉。对于带 batch 的情况：

```python
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
```

不需要显式 transpose，维度命名已经表达了操作。若前面有任意多个 batch 维度，可以用 `...` 省略：

```python
z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")
```

这让代码可以写成模块化形式，不必关心具体有多少个前导 batch 维度。

`reduce` 是 sum、mean、max、min 等归约操作的泛化：

```python
y = reduce(x, "... hidden -> ...", "sum")
```

`rearrange` 用于拆分或合并维度。例如一个形状为 $(3,8)$ 的 tensor，其中 8 实际表示 `heads * hidden1`，可以写成：

```python
x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)
```

拆开后对 `hidden1` 做矩阵乘法，再合并回去。拆分和合并的顺序由 pattern 中的维度顺序决定。

![对比传统 PyTorch 张量操作中容易混淆的维度参数（如 -2, -1）](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0016_00-18-27.jpg)

![使用einsum实现矩阵乘法的代码示例及结果矩阵示意图](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0018_00-20-39.jpg)

![einops rearrange和einsum函数的具体代码实现](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0021_00-24-10.jpg)

## FLOPs 计数：从线性层到训练公式 [27:26 - 34:45]

FLOP 指一次基本浮点运算，例如加法或乘法；FLOP/s 指硬件每秒能执行的浮点运算次数。两者发音相同但含义不同：前者衡量计算量，后者衡量硬件速度。H100 峰值标称为 1979 TFLOP/s，但这是带 sparsity 的数字；dense 情况要除以 2。8 张 H100 跑一周约提供 $5\times10^{21}$ FLOPs 量级。

线性模型是 FLOPs 计数的基本单元：有 $B$ 个数据点，每个点 $D$ 维，输出 $K$ 维。$X$ 为 $B\times D$，$W$ 为 $D\times K$，则：

$$
\text{FLOPs}=2BDK
$$

因为每个 $(i,j,k)$ 三元组需要一次乘法和一次加法。把 $B$ 看作 token 数、$D\times K$ 看作参数量，前向传播可写成：

$$
\text{forward FLOPs}=2\times\text{tokens}\times\text{parameters}
$$

这在一阶近似下也适用于 Transformer。其他逐元素操作通常只需要 $O(mn)$ FLOPs；对于足够大的矩阵，矩阵乘法会主导计算量。

![代码及文字说明：解释FLOPs与FLOP/s的区别，并高亮显示硬件性能度量部分](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0025_00-28-23.jpg)

![解释FLOP与FLOP/s区别并给出GPT-3/GPT-4训练算力及H100峰值性能的幻灯片](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0026_00-29-27.jpg)

![线性模型矩阵乘法的PyTorch代码及FLOPs计算公式](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0028_00-32-15.jpg)

## 从 FLOPs 到实际时间：计时与 MFU [34:45 - 40:27]

知道了逻辑 FLOPs，还要知道在硬件上实际跑多久。GPU 计时需要注意异步执行：在操作前后调用 `torch.cuda.synchronize()`，否则计时可能只测到非阻塞调用返回的时间。通常还会重复多次取平均。

实际吞吐为：

$$
\text{actual FLOP/s}=\frac{\text{actual FLOPs}}{\text{actual time}}
$$

硬件规格给出 promised FLOP/s。二者之比是 Model FLOPs Utilization：

$$
\text{MFU}=\frac{\text{actual FLOP/s}}{\text{promised FLOP/s}}
$$

现代模型训练中 MFU 达到 0.5 已经不错；纯 matmul 可能到 0.8；如果只有 0.1，说明存在明显问题。MFU 达不到 1 的重要原因之一是 memory bottleneck，后面用算术强度解释。

![Python代码片段，展示如何计时矩阵乘法操作并获取模型参数数量](/coursenotes/cs336-language-modeling-sp26/l02-pytorch-and-einops/kf_0031_00-36-08.jpg)

## 算术强度与 roofline：内存瓶颈还是计算瓶颈 [40:27 - 57:04]

硬件大致可以看成 HBM 与计算单元：数据需要从 memory 搬到 accelerator，计算完再写回。若假设通信与计算可以重叠，则总时间近似为二者最大值：

$$
\text{time}\approx \max(\text{communication time},\text{computation time})
$$

硬件的“强度”定义为：

$$
I_{\text{hw}}=\frac{\text{FLOP/s}}{\text{bytes/s}}
$$

H100 约为 295 FLOP/byte。算法的算术强度定义为：

$$
I_{\text{alg}}=\frac{\text{FLOPs}}{\text{bytes moved}}
$$

若 $I_{\text{alg}}<I_{\text{hw}}$，算法是 memory bound；若 $I_{\text{alg}}>I_{\text{hw}}$，则是 compute bound。

几个例子：

- **ReLU**：读 $2N$ bytes、写 $2N$ bytes，共 $4N$ bytes；约 $N$ 次比较，$I_{\text{alg}}\approx 0.25$，memory bound。
- **GELU**：仍然搬运 $4N$ bytes，但每个元素约 20 FLOPs，$I_{\text{alg}}\approx 5$，仍远小于 295，memory bound。
- **dot product**：读两个向量约 $4N$ bytes，写一个标量；约 $2N$ FLOPs，$I_{\text{alg}}\approx 0.5$，memory bound。
- **matrix-vector**：读取 $N\times N$ 矩阵需要 $2N^2$ bytes，计算约 $2N^2$ FLOPs，$I_{\text{alg}}$ 仅略高于 1，仍然 memory bound。
- **matrix-matrix**：两个 $N\times N$ 矩阵相乘，搬运约 $6N^2$ bytes，计算约 $2N^3$ FLOPs，因此 $I_{\text{alg}}\approx N/3$。当 $N=1024$ 时约为 341，超过 H100 的 295，进入 compute bound。矩阵越大，算术强度越高。

这解释了为什么大 batch、大矩阵对 GPU 效率重要：小矩阵可能一直在等数据搬运。Transformer 的主体是大型矩阵乘法，因此设计上具有较高的算术强度；而推理时逐个 token 生成，更像 matrix-vector 操作，容易 memory bound。

roofline 图把算术强度放在横轴、实际达到的 FLOP/s 放在纵轴：低强度区域受内存带宽限制，高强度区域受峰值算力限制。它也可以解释 MFU 为什么通常只有 0.5 左右：memory bottleneck 会让实际吞吐低于峰值。

## 训练的内存与计算账本 [57:04 - 66:47]

考虑一个深层网络：输入为 $B\times D$，有 $L$ 层，每层是 $D\times D$ 的线性变换加逐元素激活。参数量为：

$$
\text{params}=D^2 L
$$

前向传播逐层做 matmul 和 activation。反向传播时，对每一层需要计算两个梯度：对输入的梯度，以及对参数的梯度。以两层线性网络为例：

```python
h1 = einsum(x, w1, "batch in, in out -> batch out")
h2 = einsum(h1, w2, "batch in, in out -> batch out")

h1_grad = einsum(h2_grad, w2, "batch out, in out -> batch in")
w2_grad = einsum(h2_grad, h1, "batch out, batch in -> in out")
```

计算 `h1_grad` 时对 `out` 维求和；计算 `w2_grad` 时对 `batch` 维求和。两者都等价于一次矩阵乘法，因此每个参数的 backward 计算量约为 forward 的两倍。

汇总得到：

$$
\text{forward FLOPs}=2\times\text{tokens}\times\text{params}
$$

$$
\text{backward FLOPs}=4\times\text{tokens}\times\text{params}
$$

$$
\text{total FLOPs}=6\times\text{tokens}\times\text{params}
$$

这就是常见的 $6ND$ 公式的来源。它对 Transformer 也是一阶近似，前提是 context length 不太大；如果 context length 很大，attention 会带来额外的 context length 平方项。

## 优化器状态与降低内存的技巧 [66:47 - 75:24]

训练时除了参数和梯度，还要存优化器状态。以 Adagrad 为例，它维护梯度平方的累积和；Adam 则维护一阶矩和二阶矩。优化器状态通常用 FP32 存储，以保证稳定性。

内存账本大致为：

- 参数：BF16，每参数 2 bytes；
- 梯度：BF16，每参数 2 bytes；
- 优化器状态：Adagrad 每参数 4 bytes；Adam/AdamW 每参数 8 bytes；
- activations：约 $2\times B\times D\times L$ bytes（BF16）。

优化器状态占用很大，但它通常不是计算速度瓶颈，而是模型能否放进 HBM 的容量瓶颈。

降低内存有两个常用手段：

**梯度累积（gradient accumulation）**：把大 batch 拆成多个 micro-batch，分别计算梯度并累加，每经过 `batch_size / micro_batch_size` 步再更新参数并清零梯度。这样可以在不增加 activation memory 的情况下获得更大的有效 batch。

**激活检查点（activation checkpointing / rematerialization）**：前向传播只保存部分层的 activations，反向传播时从最近的 checkpoint 重新计算缺失的 activations。若对每个“linear + ReLU”块做 checkpoint，可以不存 pre-ReLU activation，大约节省一半 activation memory。极端情况下不存任何层，内存最省，但计算量会变成 $O(L^2)$。折中方案是保存约 $\sqrt{L}$ 个 checkpoint，使 activation memory 和重计算开销都约为 $O(\sqrt{L})$。

## 总结 [75:24 - 77:24]

本讲从 tensor 出发，建立了训练的资源核算框架：内存由元素数和精度决定，计算由 FLOPs 计数；$6ND$ 公式给出了训练 step 的 FLOPs 近似；算术强度和 roofline 用于判断 memory bound 与 compute bound。矩阵乘法通常是 compute bound，而大多数其他操作是 memory bound。einops 让张量维度操作更清晰，梯度累积和激活检查点则是降低显存、支持更大 batch 的实用技巧。

## 要点回顾

- 训练目标是在固定 compute、memory 下最大化计算效率；resource accounting 是优化的前提。
- tensor 存储开销 = 元素数 × 每元素字节数；参数、梯度、优化器状态、activations 都是 tensor。
- float32 动态范围好但占 4 bytes；float16 动态范围差；bfloat16 动态范围接近 float32，是训练中的常用折中。
- 混合精度：BF16 用于参数、activations、gradients，FP32 用于优化器状态；PyTorch AMP 自动处理安全转换。
- FP8 有 E4M3/E5M2 两种变体；FP4 需要 per-block scale；低比特推理量化比低比特训练容易得多。
- einops 用命名维度表达 einsum、reduce、rearrange，避免手动 transpose 和维度索引错误。
- FLOPs 衡量计算量，FLOP/s 衡量硬件速度；线性层前向 FLOPs 为 $2BDK$。
- MFU = actual FLOP/s / promised FLOP/s；现代模型训练中 MFU≈0.5 已属不错。
- 算术强度 = FLOPs / bytes moved；低于硬件强度则 memory bound，高于则 compute bound。
- ReLU、GELU、dot product、matrix-vector 通常 memory bound；大矩阵乘法可以 compute bound，强度约为 $N/3$。
- 训练总 FLOPs 近似为 $6\times\text{tokens}\times\text{params}$，来自 forward 的 2 倍与 backward 的 4 倍。
- 优化器状态占用显著：AdamW 每参数约 8 bytes 优化器状态，加上参数和梯度共约 12 bytes。
- 梯度累积用 micro-batch 累加梯度，支持更大有效 batch。
- 激活检查点用重计算换显存；保存 $\sqrt{L}$ 个 checkpoint 可平衡内存与重算开销。
