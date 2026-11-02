---
title: "第 7 讲 · 并行化 I：多卡互连与集合通信"
description: "这节课把视角从单张 GPU 内部扩展到多卡、多节点集群，核心问题仍然是数据搬运：算力单元离数据很远，多卡时数据可能在另一张卡甚至另一个机架上。课程先建立集合通信原语（broadcast、scatter、gather、re…"
---

# 第 7 讲 · 并行化 I：多卡互连与集合通信

> **Stanford CS336 语言模型从零构建 · L07** ｜ 讲者 Stanford Online ｜ 81 分钟 ｜ [原视频](https://www.youtube.com/watch?v=SzpOcwdIL0Y)

## 核心结论

这节课把视角从单张 GPU 内部扩展到多卡、多节点集群，核心问题仍然是**数据搬运**：算力单元离数据很远，多卡时数据可能在另一张卡甚至另一个机架上。课程先建立集合通信原语（broadcast、scatter、gather、reduce、all-gather、reduce-scatter、all-reduce、all-to-all），再介绍 GPU 互连硬件（NVLink/NVSwitch、InfiniBand、Ethernet、RDMA、NCCL），最后在一个 MLP 上实现三种并行策略：数据并行切 batch、张量并行切每层宽度、流水线并行切层。三种策略的通信量和适用互连带宽差异很大，选择哪种并行、如何组合，本质上是在计算冗余、显存占用和通信开销之间做权衡。

## 从单 GPU 到多 GPU：同一个瓶颈 [00:05 - 03:17]

上一周关注的是单张 GPU 内部：HBM、L2、L1/shared memory、寄存器与 SM，目标是通过 kernel 编写、融合、tiling 减少访存瓶颈。这一周把图扩展为多张 GPU，它们通过 NVLink、NVSwitch、InfiniBand 或 Ethernet 相连。

放大来看，两种情况是同一个游戏：算术逻辑单元和 tensor core 离数据很远；单卡时“很远”指 HBM，多卡时“很远”可能指另一张 GPU 的显存。**用很多 GPU 很容易，有效地用它们很难**，关键是把计算编排好，避免数据传输成为瓶颈。

![多GPU并行架构图：四个GPU通过NVLink连接至NVSwitch，再接入InfiniBand/Ethernet网络](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0001_00-00-39.jpg)

由此得到一个广义的存储/通信层级，从快到慢：

| 层级 | 互连方式 |
|---|---|
| 单节点单 GPU 内部 | L1 cache / shared memory（最快） |
| 单节点单 GPU | HBM |
| 单节点多 GPU | NVLink / NVSwitch |
| 多节点多 GPU | InfiniBand / Ethernet（最慢） |

上周还觉得 HBM 慢，这周 HBM 已经算“快”的一层。

![总结计算与数据分离的瓶颈，并列出从单GPU到多节点的广义存储/通信层级](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0004_00-02-42.jpg)

## 为什么要用多 GPU [03:17 - 05:45]

两个原因：

1. **放不下**：参数、激活、梯度、优化器状态超出单卡 HBM。例如 B200 有 192 GB HBM，而 1 万亿参数模型显然放不进一张卡。
2. **想更快**：即使模型能放进单卡，也可以把工作拆开、利用更多 GPU 的 FLOPs 来缩短训练时间。

这两者有时需要权衡：把工作铺开能获得更多核心，但必须支付通信带宽的代价，因此需要计算如何并行才划算。

![解释为何需要多GPU：单卡显存不足与利用更多FLOPs加速训练](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0005_00-04-03.jpg)

课程分为两部分：

- **第一部分：分布式通信与计算的基础构件**——编程模型（集合通信）、硬件（GPU 如何连接）、NCCL/PyTorch 实现、实际带宽基准测试。
- **第二部分：实际训练**——数据并行、张量并行、流水线并行，用 MLP 演示。MLP 是 Transformer 中的计算瓶颈，因此这些例子具有代表性。

![分布式训练课程大纲，包含通信原语、硬件、NCCL/PyTorch实现及三种并行策略](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0006_00-05-24.jpg)

> 代码运行说明：直接运行 `python lecture_07.py` 会使用多进程；讲课追踪时使用单进程模式关闭分布式，以便逐行讲解。想看多进程的真实标准输出，需要按多进程方式运行。

![分布式通信与计算课程大纲及代码执行说明](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0026_00-31-04.jpg)

## 集合通信原语 [05:50 - 21:30]

集合通信（collective operations）来自 1980 年代的并行编程文献，并非为语言模型训练发明，但今天仍然使用。**“集合”意味着你指定一个跨多设备的通用通信模板**，而不是自己管理 GPU 之间的点对点通信；系统可以据此做更多优化。

基本术语：

- **rank**：一个设备，本课中就是一张 GPU（也可以是 TPU）。
- **world size**：设备总数。例如 4 个 rank 对应 world size = 4。

![集合通信操作定义及Rank 0-3的Setup示意图](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0007_00-06-45.jpg)

操作可以分为三类：`broadcast`、`scatter`、`gather`、`reduce` 是理解后续操作的基础；`all-gather`、`reduce-scatter`、`all-reduce` 是分布式训练中反复出现的主力；`all-to-all` 主要用于 MoE，本讲不展开。

![Rank与World size定义及集合通信操作分类列表](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0008_00-08-06.jpg)

### 基础操作：broadcast、scatter、gather、reduce [08:16 - 12:36]

- **broadcast**：某个 rank（例如 rank 0）把自己的张量复制到所有 rank，结束后每个 rank 持有相同张量。训练中一般只用于初始化，比如加载 checkpoint 后广播一次。
- **scatter**：rank 0 有一个大张量，把它切成 world size 份，分发给各个 rank，每个 rank 拿到其中一块。
- **gather**：scatter 的逆操作。每个 rank 持有一块，gather 到指定 rank（例如 rank 0）后拼接成完整张量。
- **reduce**：每个 rank 持有一块数据，对这些数据做结合律、交换律运算（sum、max、min 等），结果写到指定 rank。例如四个 rank 分别持有 0、1、2、3，做 sum reduce 后 rank 0 得到 6。可以把 gather 理解为“运算为拼接”的 reduce。

![展示 Scatter 操作的代码示例，将 rank 0 的张量分发到所有 rank](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0009_00-09-27.jpg)

![展示 Gather 操作的代码示例，将各 rank 的数据收集到 rank 0](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0010_00-10-48.jpg)

![展示 Reduce 操作的代码示例，对各 rank 的数据进行求和归约](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0011_00-12-36.jpg)

这些操作本身不直接出现在训练主路径上，但它们是理解 all-gather、reduce-scatter 的垫脚石。

### 训练主力：all-gather、reduce-scatter、all-reduce [12:36 - 16:48]

**all-gather**：对每个 rank 都执行 gather，而不仅是 rank 0。“all”表示输出到所有设备。典型场景是每个 rank 只持有一部分参数，前向计算前需要 all-gather 出完整参数。

![All-gather 操作的完整代码示例及使用场景说明](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0012_00-13-30.jpg)

**reduce-scatter**：对张量的每个分量分别做 reduce，再把结果分散到不同 rank。例如输入为

```
rank0 = [0, 1, 2, 3]
rank1 = [1, 2, 3, 4]
rank2 = [2, 3, 4, 5]
rank3 = [3, 4, 5, 6]
```

按列求和后得到 `[6, 10, 14, 18]`，再分别放到 rank 0、1、2、3。典型场景是反向传播后汇总各数据分片的梯度，但把存储也分散开。

![Reduce-scatter 操作的代码示例及使用场景说明](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0013_00-14-51.jpg)

**all-reduce**：先 reduce，再把结果复制到所有 rank。它等价于 **reduce-scatter + all-gather**。上面的输入做 sum all-reduce 后，每个 rank 都得到 `[6, 10, 14, 18]`。数据并行中汇总梯度、再让每个 rank 都持有完整结果，用的就是它。

![All-reduce操作的代码示例，展示输入张量和输出结果，并说明其用于反向传播后汇总梯度](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0014_00-16-12.jpg)

后面要讲 FSDP/ZeRO 时，需要把 all-reduce 拆成 reduce-scatter 和 all-gather 两步，这样才能在中间介入、管理存储；基础版本直接用 all-reduce 就够了。

### all-to-all 与 MoE [16:48 - 19:43]

**all-to-all** 是最通用的集合操作：每个 rank 向其他每个 rank 发送指定的一段数据。输入示例中，rank 0 的 `[0,1,2,3]` 分别发给 rank 0/1/2/3，rank 1 的 `[4,5,6,7]` 也同样分发。输出为：

```
rank0 = [0, 4, 8, 12]
rank1 = [1, 5, 9, 13]
rank2 = [2, 6, 10, 14]
rank3 = [3, 7, 11, 15]
```

![All-to-all操作的完整代码示例，包含输入、输出结果及在MoE中的应用说明](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0016_00-18-55.jpg)

MoE 是它的典型用武之地：每个 rank 既有数据分片，又持有部分专家，需要根据数据动态路由激活到对应专家，这就是一次 all-to-all。如果各 rank 之间的收发量完全均衡，all-to-all 就相当于把矩阵转置；一般它也能处理不均衡切分，但应尽量让切分均衡——这与前面讲 MoE 时提到的负载均衡是同一个目标。

### 术语记忆 [19:43 - 21:30]

- **reduce**：执行某种结合律、交换律运算（sum、min、max）。
- **scatter** 是 **gather** 的逆操作：scatter 分发，gather 集中。
- **all**：目的地是所有设备。因此 all-reduce = reduce 到所有设备，all-gather = gather 到所有设备。

![关于 all-to-all 通信在 MoE 中应用及术语记忆方法的代码与笔记页面](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0017_00-20-16.jpg)

> 问答中确认：gather/reduce 的目标 rank 不必预先固定，调用时指定即可；这些操作既有概念图，也很快就会在代码中实现。

## GPU 之间的连接硬件 [21:53 - 30:03]

### 从家用机器到训练集群 [21:53 - 26:18]

最经典的图景是家用机器：CPU 通过 PCIe 连接 GPU，节点内 GPU 走 PCIe，不同节点之间走 Ethernet。如果只是把游戏 GPU 和朋友连起来训练大模型，大概就是这个配置。

![展示经典家庭网络硬件架构图，对比 PCIe 与以太网的通信带宽](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0020_00-22-58.jpg)

严肃训练时则是：每个节点通常 8 张 GPU，通过 NVIDIA NVLink 连接到 NVSwitch；多个节点组成 pod，节点间通过 InfiniBand 连接；更大的集群再用 Ethernet 连接 pod。

![GPU服务器节点内部架构图：8个GPU通过NVLink连接至NVSwitch，并展示SM、L2、HBM层级结构](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0021_00-24-19.jpg)

典型配置与带宽量级：

| 范围 | 互连 | 带宽量级 |
|---|---|---|
| 节点内 | NVLink 5.0 + NVSwitch | 1.8 TB/s（B200 的 HBM 为 8 TB/s，约为其 1/4） |
| pod 内 | InfiniBand（PCIe → HCA/NIC → 线缆） | 约 0.05 TB/s |
| 集群/数据中心 | Ethernet（PCIe → CPU） | 更低 |

![典型集群组网配置说明：节点内NVLink互联、Pod内InfiniBand互联、集群间Ethernet互联](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0022_00-25-40.jpg)

从编程角度看，NVLink + NVSwitch 让任意 GPU 看起来都能直接连到任意其他 GPU，硬件负责把数据送到交换机再路由。但随着 GPU 数量增长，不可能用一个 NVSwitch 域覆盖十万张卡，于是必须退化到 InfiniBand，再退化到 Ethernet——这与存储层级是同构的：越远越慢。

### 绕过 CPU：RDMA [26:18 - 29:29]

传统 Ethernet 的问题在于 GPU 必须先和 CPU 打交道：把数据复制到内核 socket buffer，构建 TCP 包，再复制到网卡环形缓冲区，最后发出去。这里的“内核”是 CPU 操作系统内核，不是 GPU kernel。这一串操作带来很大延迟。

**RDMA（Remote Direct Memory Access）** 允许一张 GPU 直接读写另一张 GPU 的显存，完全不经过 CPU。NVLink/NVSwitch 域内天然具备这种能力；InfiniBand 也支持 RDMA；**标准 Ethernet 不支持**。

![说明Ethernet需经CPU中转的瓶颈，并引出RDMA与InfiniBand绕过CPU的优势](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0023_00-27-01.jpg)

两个新进展：

- **NVL72**：8 GPU 一个 tray，9 个 tray 一个 rack，共 72 张 GPU 处于同一个 NVLink 域。代价昂贵，但把“快速互连”的边界从 8 卡推到了 72 卡。
- **RoCE（RDMA over Converged Ethernet）**：让 Ethernet 也能绕过 CPU，是 InfiniBand 之外更便宜的 RDMA 方案。Meta 曾探索用融合以太网训练模型。

![GB200 NVL72架构及RoCE技术介绍幻灯片](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0025_00-29-43.jpg)

### NCCL [30:03 - 31:08]

最底层是 **NVIDIA Collective Communications Library（NCCL）**。你告诉它“我要做 all-reduce”，它会：

1. 探测硬件拓扑（节点数、交换机、NVLink/PCIe）；
2. 选择 GPU 之间的路径；
3. 启动 GPU kernel 来收发数据——通信本身也是 GPU kernel。

![NVIDIA NCCL 与 PyTorch torch.distributed 分布式通信库介绍幻灯片](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0027_00-36-21.jpg)

> 问答中提到：RDMA 是“操作层面”的能力（GPU 直接读写远端显存），NVLink/NVSwitch/InfiniBand 是“硬件层面”的线缆与交换机；两者是不同层次的描述。对于 9 卡这类不整齐的配置，如果第 9 张卡落在没有 NVLink 的另一个节点上，通信会非常昂贵。

## 用 PyTorch 实现集合通信 [36:20 - 46:30]

PyTorch 的 `torch.distributed` 提供了集合操作的干净接口，不需要直接面对 NCCL。它支持多种后端：GPU 用 `nccl`，CPU 用 `gloo`。它也提供 FSDP 等高层算法，但本课坚持从零构建，不使用这些高层封装。

![NVIDIA NCCL 与 PyTorch torch.distributed 分布式通信库介绍幻灯片](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0027_00-36-21.jpg)

### 启动与同步

`spawn(func, world_size=4)` 会启动 4 个进程，每个进程执行一次 `func`，各自拥有 `rank = 0..world_size-1`。正常路径使用 `torch.multiprocessing.spawn`；讲课追踪时走 `DisableDistributed` 分支，直接在单进程里运行。

![分布式训练中 spawn 函数的实现代码，展示了多进程启动与 DisableDistributed 上下文](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0028_00-37-49.jpg)

`setup(rank, world_size)` 配置 master 地址和端口。注意这个地址端口只用于元数据和进程协调，**真正的数据走 NCCL**，否则会非常慢。有 CUDA 就用 `nccl` 后端，否则用 `gloo`。

`dist.barrier()` 是同步屏障：所有进程都到达这里后才继续。由于各进程异步运行，print 顺序不由你控制；需要保证某段代码先于另一段执行时就插入 barrier。代价是可能产生不必要的等待。

![集合通信主函数代码，包含 rank 与 world_size 的定义及 dist.barrier() 同步操作](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0029_00-39-10.jpg)

### all-reduce、reduce-scatter、all-gather

all-reduce 的例子中，每个 rank 的初始张量是 `[0,1,2,3] + rank`：

```python
data = tensor([0., 1, 2, 3], device=...) + rank
dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
```

all-reduce 原地写回，结束后每个 rank 都得到 `[6, 10, 14, 18]`。

![PyTorch 分布式 all-reduce 操作的示例代码及参数设置](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0032_00-41-52.jpg)

运行输出显示：all-reduce 前各 rank 数据不同，all-reduce 后完全相同；reduce-scatter 后输入未被修改，输出分别是 `[6.]`、`[10.]`、`[14.]`、`[18.]`；再把 reduce-scatter 的输出作为 all-gather 的输入，就得到完整的 `[6, 10, 14, 18]`——**用代码验证了 all-reduce = reduce-scatter + all-gather**。

![分布式集合通信（All-reduce 与 Reduce-scatter）的运行结果输出日志](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0030_00-40-30.jpg)

`async_op=True` 时调用会立即返回，你可以去做与这次通信无关的工作（例如预取下一步数据），需要结果时再 `wait` 或 barrier。这就是“通信与计算重叠”的基础。

## 通信带宽基准测试 [46:38 - 53:15]

用 1 亿个元素做 all-reduce，先 warm up，再用 `cuda.synchronize()` 加 barrier 确保 CUDA kernel 和所有进程都完成，然后计时，结束后再次同步。每个 rank 都会报告自己的时间，可以取平均。

关键指标是**有效带宽（effective bandwidth）**：算出发送了多少字节，除以总耗时。

All-reduce 的发送字节数需要这样拆解：

- 归约需要 world size − 1 步加法；
- 乘以 2，因为既要发送也要归约；
- 乘以 payload 大小；
- 总时长乘以 world size（所有 rank 都在等待）。

$$
\text{bandwidth}
= \frac{\text{size} \times 2 \times (W-1)}{W \times T}
$$

其中 $W$ 是 world size，$T$ 是 wall-clock 时间。随着 $W$ 增大，$\frac{W-1}{W}$ 趋近 1，带宽趋近 $\frac{2\cdot\text{size}}{T}$，**与 world size 无关**，也**与拓扑无关**（NCCL 自行决定用 ring 还是 tree）。实测约 400 GB/s。

Reduce-scatter 的公式没有那个 2：

$$
\text{bandwidth}
= \frac{\text{size} \times (W-1)}{W \times T}
$$

实测带宽与 all-reduce 相近。原因是 all-reduce 移动两倍数据、也花两倍时间，比值不变。

> 问答中确认：即使使用集合通信，底层仍是 CUDA 操作，默认异步；计时前必须 `cuda.synchronize()`。barrier 与 synchronize 的顺序要小心：如果先 barrier，而 CUDA kernel 还没跑完，barrier 本身并不能保证全局同步。

## 数据并行（DDP）[55:07 - 63:00]

数据并行的切法是**沿 batch 维度切**：每张 GPU 持有完整模型参数，处理不同数据，反向传播后同步梯度。

用 MLP 演示：batch size 128、维度 1024。把数据矩阵按行切成 world size 份，每个 rank 拿到 32 行（local batch size），各 rank 只加载自己那片数据。

模型是 `num_layers` 层、每层 `num_dim × num_dim` 的矩阵，随机初始化后交给优化器。

训练循环：

```python
# 前向：只用自己的局部数据
for layer in range(num_layers):
    x = ...  # forward through layer

# 反向：梯度只反映局部数据
loss.backward()

# 关键的一行：跨 rank 同步梯度
for param in params:
    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
    param.grad /= world_size

# 更新参数
optimizer.step()
```

每个 rank 用自己的数据算出不同梯度，all-reduce 求平均后，各 rank 持有完全相同的梯度，于是参数更新后也保持一致。**这就是标准训练加上反向传播后的一次 all-reduce**，改动非常小、非常优雅。每个 rank 表现得像拥有全部数据一样更新参数，实际只处理了一部分数据。

注意事项：

- batch size 至少要不小于 world size，通常应明显更大；最好整除 world size，否则需要 padding。
- 对 Transformer 也一样：DDP 只负责平均梯度，不关心前向结构长什么样。
- 局限：每个 rank 都要保存完整参数、梯度和优化器状态。如果模型放不下，就需要 FSDP/ZeRO——这是下节课的主题，也是把 all-reduce 拆成 reduce-scatter + all-gather 的动机。

## 张量并行 [63:00 - 69:38]

张量并行**不切数据，而是切每一层**：每个 rank 负责每层参数的一部分。这里讲的是按列切分（column tensor parallel）：参数矩阵形状为 `num_dim × local_num_dim`，每个 rank 持有若干列。按行切分也存在，但本讲不展开。

前向传播：

1. 每个 rank 用自己那部分参数计算局部激活，形状为 `batch_size × local_num_dim`；
2. 非线性是逐元素的，可以在局部激活上直接做；
3. 但下一层需要完整的输入，因此要 **all-gather 激活**：每个 rank 分配一个长度为 world size 的列表，all-gather 后把各块拼成 `batch_size × num_dim` 的完整 `X`。

![All-gather 操作的完整代码示例及使用场景说明](/coursenotes/cs336-language-modeling-sp26/l07-parallelism-i/kf_0012_00-13-30.jpg)

反向传播是对偶的：前向做 all-gather，反向就做 **reduce-scatter**，把梯度归约并分散回各 rank。all-gather 与 reduce-scatter 在这里呈现出一一对应的关系，与 all-reduce 的分解是同一套逻辑。

与数据并行的关键区别：

- 数据并行把模型当成一个整体模块，切的是数据，模型代码几乎不用改；
- 张量并行必须深入模型内部改参数布局和通信；
- 这里显式管理通信，`autograd` 不会自动帮你插入 reduce-scatter；需要自己调用。PyTorch 生态中有自动化的实现，但本课是“from scratch”，所以手动做。

张量并行的通信量很大：每一层都要传激活。这决定了它通常只能部署在 NVLink/NVSwitch 这种高带宽域内。

## 流水线并行 [69:38 - 74:20]

流水线并行**按层切分**：每个 rank 拿到一部分层，层内保留完整维度；数据最终会以某种形式流过所有 rank。

每个 rank 只有 `local_num_layers` 层参数，每层仍是 `num_dim × num_dim`。为了减少流水线气泡，还要把 batch 切成多个 **micro-batch**。

执行流程：

```python
for micro_batch in micro_batches:
    if rank == 0:
        x = micro_batch
    else:
        x = recv(from rank - 1)      # 点对点接收
    x = forward(x, local_layers)     # 只跑本 rank 负责的层
    if rank != world_size - 1:
        send(x, to rank + 1)         # 点对点发送
```

这里用了点对点 `send` / `recv`，而不是集合操作。

**流水线气泡**：rank 0 处理完第一段后把数据发给 rank 1，此时 rank 1 才能开始；在等待期间，后面的 GPU 可能无事可做。把 batch 拆成 micro-batch 后，数据可以更细粒度地流动，让各 stage 更快地被填满，从而减少气泡。

朴素版本还有两个不足：

1. **没有重叠通信与计算**。如果给 send/recv 加上异步（`isend`/`irecv`），就可以在计算当前 micro-batch 的同时接收或发送另一个 micro-batch，进一步减少等待。
2. 数据并行的 all-reduce 也可以重叠：反向传播中某层梯度一旦就绪就可以开始发送，不必等整个反向传播结束。这是作业二会探索的内容。

## 其他并行方式与选型考量 [74:24 - 78:02]

本讲没有覆盖的并行方式：

- **序列并行**：把序列切成多段，从而并行化 attention 计算。
- **专家并行**：切分 MoE 的专家，正是 all-to-all 的用武之地。
- 以及上述方式的组合。

**选择哪种并行，强烈依赖硬件互连**：

- **张量并行**通信量大（每层都要传激活），一般放在节点内 NVLink/NVSwitch 域。
- **流水线并行**对带宽容忍度高得多，可以跨慢速互连，甚至用于节点分布在不同地理位置的去中心化训练；但你不能在那种场景下做张量并行。
- 常见组合是：节点内张量并行 + 跨节点数据并行/FSDP + 必要时流水线并行。
- 还有一个效应是 **critical batch size**：数据并行把 batch 增大到一定程度后收益递减，继续加大只是浪费算力，此时不如转向张量并行。

**PyTorch 手工原语 vs. JAX/TPU 编译器**：本课刻意用 PyTorch 的集合操作原语，让你机械地看到每一步在做什么。JAX/TPU 路线则是定义模型和分片策略，由编译器推导需要哪些通信、自动插入。后者很吸引人，但会掩盖“从零构建”的细节。

## 总结 [78:02 - 81:02]

- 并行有很多切法：按数据切、按张量切、按专家切、按流水线切、按序列切。
- 本讲只实现了 DDP；FSDP 和 ZeRO 留到下节课。
- 张量并行需要非常快的互连；流水线并行对互连要求较低，但必须努力消除气泡。
- 一个反复出现的模式：**重算、存在内存里，或者存到另一张 GPU 上**。激活 checkpointing 是前者的例子；数据并行中每个 rank 冗余保存全部参数和优化器状态，则是用冗余计算换取“不必搬运优化器状态”。
- 硬件会越来越快，但人们总会想要更大的模型，因此这种分层结构会长期存在。

## 要点回顾

- 多 GPU 的核心矛盾与单卡相同：算力离数据远，目标是避免数据传输瓶颈；用很多 GPU 容易，用得好很难。
- 用多 GPU 的两个理由：单卡显存放不下参数/激活/梯度/优化器状态；想利用更多 FLOPs 加速训练。
- 集合通信原语：broadcast、scatter、gather、reduce 是基础；all-gather、reduce-scatter、all-reduce 是训练主力；all-to-all 用于 MoE 的动态路由。
- 关键恒等式：**all-reduce = reduce-scatter + all-gather**；拆开它是 FSDP/ZeRO 管理存储的前提。
- 互连层级：L1/shared memory → HBM → NVLink/NVSwitch → InfiniBand → Ethernet，越远越慢；NVLink 5.0 约 1.8 TB/s，B200 HBM 约 8 TB/s。
- RDMA 让 GPU 直接读写远端显存、绕过 CPU；InfiniBand 支持，标准 Ethernet 不支持，RoCE 是 Ethernet 侧的补足方案。
- NCCL 负责探测拓扑、选择路径、启动通信 kernel；PyTorch `torch.distributed` 提供 `nccl`/`gloo` 后端和集合操作接口。
- 有效带宽公式：all-reduce 为 $\frac{2\cdot\text{size}\cdot(W-1)}{W\cdot T}$，随 $W$ 增大趋近 $\frac{2\cdot\text{size}}{T}$，与 world size 和拓扑无关。
- 数据并行（DDP）：切 batch，每个 rank 持有完整参数；反向传播后 all-reduce 并平均梯度，其余与标准训练相同。
- 张量并行：切每层宽度；前向 all-gather 激活，反向 reduce-scatter 梯度；通信量大，适合节点内高带宽域。
- 流水线并行：切层，配合 micro-batch 减少气泡；需要重叠通信与计算，对互连带宽要求较低。
- 并行策略的选择取决于互连带宽、模型能否放下、critical batch size 等因素，实际系统通常是多种策略的组合。
