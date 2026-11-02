---
title: "第 6 讲 · Kernel、Triton 与 XLA"
description: "本讲从 GPU 硬件约束出发，说明为什么“能写对”和“跑得快”是两件事：编程模型只描述线程、线程块和网格，而性能由 warp、寄存器占用、共享内存 bank 冲突、内存合并和 block 占用率等硬件细节决定。作者强调先…"
---

# 第 6 讲 · Kernel、Triton 与 XLA

> **Stanford CS336 语言模型从零构建 · L06** ｜ 讲者 Stanford Online ｜ 87 分钟 ｜ [原视频](https://www.youtube.com/watch?v=xnDHaNUvHBg)

## 核心结论

本讲从 GPU 硬件约束出发，说明为什么“能写对”和“跑得快”是两件事：编程模型只描述线程、线程块和网格，而性能由 warp、寄存器占用、共享内存 bank 冲突、内存合并和 block 占用率等硬件细节决定。作者强调先 benchmark、再 profile、最后才写 kernel，并以 GELU 为例展示朴素 PyTorch 实现、内置融合 kernel 与 `torch.compile` 生成 Triton kernel 的性能差异。随后用 Triton 从 element-wise GELU、行内 softmax、跨 tile 行归约一路讲到 matmul + ReLU，核心思想是“以线程块为单位，load 到片上存储、计算、再 store 回 HBM”，并在 matmul 中通过 tiling 提高算术强度。最后总结：Triton 是介于 PyTorch 大算子和 CUDA 线程级编程之间的抽象，适合快速获得不错的性能，但真正极限仍取决于硬件与更底层工具。

## GPU 硬件层级回顾 [00:05 - 03:15]

上一讲从高层介绍了 GPU 性能，这一讲进入代码层面：写 Triton kernel、做 benchmark 和 profiling。先回顾典型 NVIDIA GPU 的存储层级：

- 一个 GPU 包含若干 streaming multiprocessors（SM），数量大约在 100–200 之间，几代之间变化不大。
- 每个 SM 内有寄存器、L1 cache / shared memory；shared memory 可由程序员控制，L1 则不能。
- L2 cache 为整个芯片共享，容量更大。
- 最外层是高带宽内存 HBM，容量最大但访问最慢。

容量与带宽大致反向相关：寄存器最快、L1/shared memory 次之、L2 更慢、HBM 最慢但最大。B200 的 HBM 带宽达到约 8 TB/s，但在 GPU 的时间尺度上仍然昂贵。关键直觉是：**大而慢的存储在远处，快而小的存储在 SM 上**。

![A100、H100、B200 三代 GPU 的硬件规格对比表，涵盖 SM 数量、缓存大小与带宽等参数](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0002_00-02-53.jpg)

## GPU 编程模型：线程、线程块与 Grid [03:18 - 07:05]

GPU 编程模型有三个核心概念：

- **Thread**：在一小部分数据上执行代码。
- **Thread block / CTA**：一组线程，可共同访问同一块 shared memory。
- **Grid**：线程块的集合。启动 kernel 就是启动一个 grid。

对 GELU 这类 element-wise 操作，线程模型很自然：每个线程处理一个元素，相当于 `f(i) for i in range(N)`。但对 softmax、matmul 这类需要线程间通信的操作，仅有线程不够。一个办法是让每个线程直接读写 HBM，但 HBM 太慢；更好的做法是利用 SM 本地的 shared memory。

因此线程块的意义在于：**它是一组共享同一块 shared memory 的线程，并被调度到一个 SM 上执行**。线程块从 HBM 读入数据，在线程间通过 shared memory 通信，计算完再写回 HBM。Triton 正是让人原生地以线程块为单位思考。

![GPU 编程模型示意图：Grid 包含多个 CTA，每个 CTA 包含多个线程](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0003_00-04-20.jpg)

![线程、线程块（CTA）与 Grid 的定义及使用线程块的原因说明](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0004_00-05-47.jpg)

## 编程模型与硬件的张力：warp、占用率与内存访问 [07:13 - 20:14]

编程模型本身很简洁，只看正确性时几乎像写 Python；但性能对硬件极其敏感。以下五类考虑决定了实际速度。

### Warp 与控制分歧

线程块内的线程按 **32 个一组**组成 warp。一个 64 线程的线程块有两个 warp。同一 warp 内所有线程必须在 SM 上以 lockstep 执行相同指令。若出现分支（`if A else B`），不同线程需要执行不同路径，只能串行化，这就是 **control divergence**，通常应避免。

SM 同时驻留多个 warp，并由 warp scheduler 在它们之间零成本切换。当某个 warp 因读取 HBM 而阻塞上百个周期时，SM 可以立刻切换到另一个 warp 做 tensor core 运算，从而隐藏延迟。

![编程模型与硬件交互幻灯片，详细讲解Warp机制与控制分歧](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0006_00-10-07.jpg)

### 寄存器占用率（warp occupancy）

每个线程最多使用 255 个寄存器。SM 的寄存器总数固定，因此单线程用的寄存器越多，能同时驻留的线程就越少，占用率下降。但占用率不是越高越好：如果每个线程做更多工作，较少线程也可能更高效。**Thread coarsening** 就是让一个线程处理多个元素，把“太轻”的线程加粗。

转写中的例子：线程块 128 线程、每线程 160 寄存器，则每块使用 20,480 个寄存器；B200 每 SM 共有 65,000 个寄存器，因此最多同时跑 3 个块，即 12 个 warp；最大 64 个 warp，占用率约 18%。

![计算 CUDA 线程块寄存器占用率的 Python 代码示例](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0008_00-13-00.jpg)

![寄存器占用率计算结果及共享内存 Bank 冲突（Bank Conflicts）概念引入](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0009_00-14-27.jpg)

### 共享内存 bank 冲突

Shared memory 被划分为 **32 个 bank，每个 4 字节宽**。每个时钟周期，每个 bank 最多被一个线程访问。若多个线程访问同一 bank 的不同地址，访问必须串行化，称为 **bank conflict**。最坏情况下，32 个线程同时访问同一列，形成 32-way bank conflict。矩阵乘法中既要按行访问一个矩阵，又要按列访问另一个矩阵，因此无法总靠选择遍历顺序避免；实际中可用 swizzling 重排 shared memory 来缓解。

![共享内存 Bank 冲突原理与最坏情况示例，含寄存器占用计算浮层](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0010_00-15-53.jpg)

### HBM 内存合并（memory coalescing）

一个 warp 的 32 个线程访问 HBM 时，访问会被合并成 128 字节的 cache line 事务。若 32 个线程连续访问同一 cache line，就是 full coalescing，一次取回整条 cache line；若按列跨步访问，则会取回大量用不到的数据。注意这和 bank conflict 类似但不同：**bank conflict 针对 shared memory，coalescing 针对 HBM**。

![内存合并访问（Memory coalescing）原理与 128 字节事务示意](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0011_00-17-20.jpg)

### Block occupancy 与尾波

线程块被调度到有限数量的 SM 上。B200 有 148 个 SM；若启动 160 个线程块，第一波只能放 148 个，剩下 12 个要等第一波结束，第二波只有 12 个 SM 在工作，形成低占用率的“尾巴”。实践中可调整 block 数量或 block size，使总块数更好地匹配 SM 数量，避免尾波。

![Block occupancy 波形图：wave 0 满载与 wave 1 尾部空闲示意](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0012_00-18-47.jpg)

小结：HBM 对所有线程全局可见，shared memory 属于线程块，寄存器属于线程；但 warp、bank conflict、coalescing、occupancy 等硬件细节才真正决定性能。

## Benchmark：先测量端到端时间 [21:47 - 26:23]

作者给出的“成功配方”是：**benchmark 和 profile 代码 → 修改 → 再 benchmark 和 profile**。应先测量、找瓶颈，再写 kernel。

Benchmark 测的是 wall-clock 端到端时间，不告诉你时间花在哪里，但能比较实现快慢、观察随维度扩展的趋势。作者从零实现 benchmark，以突出几个坑：

- **必须 warm up**：避免惰性编译等首次开销污染计时；实际关心的是反复运行时的速度。
- **多次计时**：存在方差，可取平均，严格时也可看 P95 等分布。
- 正确计时方式是使用 CUDA events：记录 start event、执行计算、记录 end event，并调用 `torch.cuda.synchronize()` 等待 GPU 异步工作完成。

```python
def benchmark(run, num_warmups=1, num_trials=3):
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()
    # 使用 CUDA events 计时并多次取平均
```

对 matmul 随维度扩展的测量显示：小维度时时间几乎恒定，直到接近 2000 维才明显增长；这是因为 GPU 为大矩阵乘法设计，小矩阵无法有效利用硬件。

![基准测试（benchmarking）的定义与用途说明，强调其测量端到端时间而非定位耗时点](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0015_00-23-07.jpg)

![benchmark 函数实现，包含预热循环与 torch.cuda.synchronize 同步等待](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0016_00-24-34.jpg)

![不同矩阵维度下 benchmark 耗时结果，展示小维度恒定、大维度立方增长](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0017_00-26-00.jpg)

## Profiling：定位时间花在哪里 [26:25 - 30:02]

Profiling 告诉你时间实际花在哪些 kernel 上；即使不关心性能，也能帮助理解高层代码底层发生了什么。PyTorch 自带 profiler。

- 对 `A + B`，底层会调用类似 `CUDAFunctor_add` 的 kernel。
- 对 matmul，底层调用名称很长的 CUTLASS kernel，例如 `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_64x64x16...`。名称中 `CUTLASS` 是 NVIDIA 线性代数库，`SM100` 对应 Blackwell 架构，`64x64x16` 是 tile 形状。
- 改变矩阵维度会调用不同 kernel：128×128 matmul 可能使用 `32x32x16` tile，而不是 `64x64x16`。

这说明：高层看似一个 `@`，底层可能根据形状选择不同实现。

![CUDA kernel 命名解析与 torch.profiler 性能分析代码](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0018_00-27-27.jpg)

![PyTorch profiler 输出界面，显示 matmul 操作的 CPU 时间分布，其中 Activity Buffer Request 占比 98.83%](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0019_00-28-54.jpg)

## 案例：GELU 的三种实现与 kernel fusion [30:14 - 36:51]

GELU 是常见非线性，常用 tanh 近似：

$$
\text{GELU}(x) \approx 0.5x\left(1+\tanh\left(\sqrt{2/\pi}(x+0.044715x^3)\right)\right)
$$

比较三种实现：

1. **Naive PyTorch**：直接按公式写多个逐元素运算。
2. **内置 `F.gelu`**：PyTorch 自带融合 CUDA kernel。
3. **`torch.compile(naive_gelu)`**：编译 naive 实现。

三者数值一致，但性能差异很大：naive 最慢，内置最快，编译版也远快于 naive，但在这次运行中略慢于内置版。

原因可通过 profiler 看到：

- naive 实现的计算图包含多个 primitive，每个 primitive 都启动一个 kernel；每个 kernel 都要从 HBM 读、算、写回 HBM，kernel 之间数据反复往返，且没有 fusion。
- 内置 GELU 是单个 CUDA kernel。
- `torch.compile` 把整个计算图融合成一个 Triton kernel，例如 `triton_poi_fused_add_mul_tanh_0`，因此每个元素只读一次、写一次 HBM。

![GeLU 三种实现对比代码：naive 手写、PyTorch 内置、torch.compile 编译，并弹出变量值验证三者结果一致](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0021_00-31-47.jpg)

![CUDA性能分析工具详细报告，展示各算子及操作的CPU耗时占比](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0022_00-33-14.jpg)

![compiled_gelu 性能剖析界面，展示编译后实现的耗时及 triton 融合算子信息](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0023_00-34-40.jpg)

这也自然引出 Triton：编译版生成的正是 Triton kernel。

## Triton 编程模型：以线程块为中心 [36:51 - 39:43]

CUDA 的思路是“指定每个线程做什么”：代码根据 thread ID 执行，控制精细，但需要手动管理 shared memory 和线程同步。对 element-wise 操作尚可，复杂操作会麻烦。

Triton 由 OpenAI 开发，思路是“指定每个线程块做什么”。一般性足够强，尤其适合上手；若要极致利用最新硬件特性，可能不如更底层工具灵活。其概念框架是：

> 线程块把数据 load 到 shared memory（或寄存器），在片上计算，再 store 回 global memory。

相比 PyTorch 以巨大矩阵乘法为原子操作，Triton 处于“单个元素”和“整个大算子”之间。

![CUDA与Triton线程抽象层级对比及Triton编程概念框架说明](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0025_00-39-00.jpg)

## Triton 实现 GELU：第一个 kernel [39:44 - 50:15]

以 8000 维向量为例。Triton 不再函数式返回结果，而是显式读写：先分配输出 tensor，再把输入切成 block。

启动配置：

```python
num_elements = x.numel()
BLOCK_SIZE = 1024
num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=BLOCK_SIZE)
```

kernel 的通用形式是：

```python
@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # 计算 GELU tanh 近似
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)
    tl.store(y_ptr + offsets, y, mask=mask)
```

要点：

- `x_ptr`、`y_ptr` 是指针/地址，需做指针算术。
- `pid` 标识当前线程块；`start = pid * BLOCK_SIZE` 决定处理哪一段。
- `tl.arange` 生成块内偏移，`mask` 处理最后一个不满块。
- `tl.load` 从 HBM 读入，计算后 `tl.store` 写回。
- 对 element-wise 而言，Triton 写法像“向量化版本”的 CUDA；但 Triton 不直接控制数据放在寄存器还是 shared memory，由编译器决定。

![Triton kernel启动配置：计算num_blocks并调用triton_gelu_kernel](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0027_00-41-54.jpg)

![Triton GELU 内核代码：定义线程块划分与带掩码的输入读取逻辑](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0028_00-43-20.jpg)

![Triton GELU 内核代码：使用 tanh 近似公式计算 GELU 并存储结果](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0029_00-46-14.jpg)

## 从 Triton 到 PTX：底层发生了什么 [50:39 - 55:06]

Triton 编译器生成 PTX，这是 NVIDIA GPU 的中间汇编语言。观察 PTX 可见：

- `ld.global` 从 HBM 加载到寄存器，`R` 为整数寄存器，`FR` 为浮点寄存器。
- 有大量 `mov`、`mul` 等指令。
- 末尾有 global store 写回 HBM。
- 一个线程实际处理多个元素（例如 8 个），体现 thread coarsening；编译器认为线程太轻，于是加粗。
- PTX 是同一段代码由每个线程执行，通过 `CTA.x`（块索引）和 `TID.x`（线程索引）区分身份。

PTX 仍不规定 warp 如何调度、块落在哪个 SM 等，很多由硬件控制。当某条 load 阻塞时，warp scheduler 切换到其他 warp，等数据就绪再切回。

![Triton 编译生成的 PTX 汇编代码，展示向量化加载与浮点乘法指令](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0032_00-52-00.jpg)

## Softmax：行能放进一个 block [58:06 - 65:21]

Softmax 对矩阵每行做指数归一化。朴素 PyTorch 实现包含求 max、减 max、exp、sum、除等多步，对应多个 kernel，产生多次 HBM 读写（约 5 次读、3 次写）。

Triton 版本让**每一行对应一个线程块**。因为 softmax 是 row-wise 归约，不同行之间不交互，不需要跨 block 的 shared memory。

kernel 结构：

```python
@triton.jit
def softmax_kernel(x_ptr, y_ptr, stride, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    y = num / denom
    tl.store(y_ptr + row * stride + cols, y, mask=mask)
```

被 mask 掉的位置填 `-inf`，在 softmax 中相当于零贡献。只要一行能放进一个 block，Triton 写法几乎像普通 PyTorch。若要按列 softmax，也可通过调整 stride/列偏移实现。

## 行归约：一行放不进 block 时用 tile 循环 [65:23 - 71:53]

当一行长度超过 block size，例如 4000 列而 block 只有 1024，就需要把行切成多个 **tile**。注意：tile 不是线程块；线程块仍负责整行，tile 是块内循环处理的数据片段。

以 row sum 为例：每个线程维护一个累加器，循环遍历该行的所有 tile，每次 load 一个 tile 并累加到累加器；循环结束后再对累加器做归约，得到一个标量写回。

```python
@triton.jit
def row_sum_kernel(x_ptr, y_ptr, N, stride, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0)
        acc += x
    total = tl.sum(acc, axis=0)
    tl.store(y_ptr + row, total)
```

累加器放在寄存器还是 shared memory 由 Triton 编译器决定；若 block size 足够大，通常必须放 shared memory。与 GELU 的区别在于：GELU 把整行切成多个独立 block，而这里整行是一个 block，内部循环多个 tile，因此开始出现“数据无法全部放进 shared memory”的复杂性。

## Matmul + ReLU：tiling 与 kernel fusion [71:53 - 82:12]

Matmul 是深度学习的基础操作，已被高度优化。这里做 $A(M\times K) \times B(K\times N) = C(M\times N)$，并在末尾融合 ReLU。

### 朴素 matmul 的问题

朴素做法：对每个输出元素 $C_{ij}$，遍历 $K$，读取 $A$ 的行和 $B$ 的列，乘加后写回。它正确，但每个输出元素都要从 HBM 读取，读次数约为 $O(MKN)$，而计算量也是 $O(MKN)$，算术强度为常数，大量读取是冗余的。

### 理想做法与 tiling

理想情况是把整个 $A$、$B$ 读入 shared memory，再计算 $C$，这样读次数降为二次，算术强度可达 $O(N)$。但 $A$、$B$ 通常远大于 shared memory。

实际采用 **tiling**：把 $C$ 切成 tile，每个 tile 由一个线程块负责。该线程块沿 $K$ 维扫描：加载 $A$ 的行 tile 和 $B$ 的列 tile 到 shared memory，做小块 matmul 并累加到 shared memory 中的累加矩阵；扫描完后把输出 tile 写回 HBM。这样算术强度提升到 $O(\text{tile size})$，虽达不到 $O(N)$，但已大幅减少冗余读取。

### Stride 与 kernel 结构

Tensor 在内存中线性展开，stride 描述多维索引到线性地址的映射：行索引乘行 stride，列索引乘列 stride。转置矩阵的 stride 相反。

Triton matmul kernel 的结构：

```python
@triton.jit
def matmul_relu_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                       stride_am, stride_ak, stride_bk, stride_bn,
                       stride_cm, stride_cn,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # 计算 A 的行偏移、B 的列偏移、K 方向偏移
    # 初始化 shared memory 中的累加矩阵
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(...)  # 加载 A tile
        b = tl.load(...)  # 加载 B tile
        acc += tl.dot(a, b)
    acc = tl.maximum(acc, 0.0)  # 融合 ReLU
    tl.store(...)
```

关键点：

- 每个线程块负责 $C$ 的一个 tile。
- 循环沿 $K$ 维推进，同时加载 $A$ 的行 tile 与 $B$ 的列 tile。
- 当数据在 shared memory 中时，可以像 PyTorch 一样直接 `tl.dot`。
- 在写回 HBM 前顺手做 ReLU，即 kernel fusion，几乎无额外成本。

![Triton 编译生成的 PTX 汇编代码，展示向量化加载与浮点乘法指令](/coursenotes/cs336-language-modeling-sp26/l06-kernels-triton-xla/kf_0032_00-52-00.jpg)

## 总结与开放问题 [82:12 - 86:40]

本讲的主线是“编程模型 vs 硬件现实”：

- 程序员可在 PyTorch、Triton、PTX 等层次控制计算；PTX 甚至可手写，但通常不需要。
- 代码最终要落到有限 SM、有限 bank、有限寄存器和内存上，因此 benchmark 与 profiling 是理解性能的必要手段。
- Triton 以线程块为抽象，省去显式线程同步和 shared memory 管理；思路是“读入片上存储、计算、写回 HBM”。
- 示例难度递进：element-wise GELU → 行内 softmax → 行放不下时的 tile 循环 → matmul 的正式 tiling。
- 关于 Triton 的替代方案：可下探到 PTX，也可使用 ThunderKittens、CUTE 等 DSL/库；不同语言有不同归纳偏置，Triton 因面向 Transformer 训练而适合该类负载，但极限性能仍取决于硬件与工具选择。
- 有观众问高维张量应一次性全部载入还是逐元素处理，作者认为不能抽象回答，取决于计算性质。

下一讲将转向多 GPU 编程。

## 要点回顾

- GPU 存储层级：寄存器、L1/shared memory 在 SM 上，L2 全芯片共享，HBM 最大最慢；容量与带宽大致反向。
- 编程模型：线程 → 线程块/CTA → grid；线程块共享 shared memory 并被调度到一个 SM。
- 硬件约束：warp 为 32 线程 lockstep，分支导致 control divergence；寄存器用量影响 warp occupancy；shared memory 有 32 bank 冲突；HBM 访问需 coalescing；block 数量不匹配 SM 会产生尾波。
- Benchmark 要 warm up、多次计时、用 CUDA events 并 synchronize；profile 用于定位具体 kernel 与底层实现。
- GELU 案例：naive 多 kernel 反复读写 HBM；内置与 `torch.compile` 融合为单 kernel；编译版生成 Triton kernel。
- Triton kernel 模板：`pid` 定位块，`offsets` 和 `mask` 处理索引，`tl.load` 读、计算、`tl.store` 写。
- PTX 展示线程级实际指令；编译器会做 thread coarsening；warp scheduler 用切换隐藏 HBM 延迟。
- Softmax：一行一个 block，行内归约，mask 位置填 `-inf`。
- 行放不下时：block 仍负责整行，内部循环 tile，线程维护累加器，最后归约。
- Matmul：朴素实现算术强度为常数；tiling 让每个 block 计算一个 C tile，沿 K 加载 A/B tile 到 shared memory 并累加，算术强度提升到 tile size 量级；可在写回前融合 ReLU。
- Triton 是线程块级抽象，适合快速实现 Transformer 相关 kernel；更底层可选 PTX，其他 DSL 各有取舍。
