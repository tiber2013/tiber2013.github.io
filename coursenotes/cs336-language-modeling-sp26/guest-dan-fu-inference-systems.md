---
title: "客座讲座 · 推理系统：把电力变成智能（Dan Fu）"
description: "这场客座讲座从推理系统视角讨论语言模型：训练完成后，真正把 GPU 算力转化为可用智能的是推理引擎。一次请求会经历调度、KV cache 复用、prefill 与 decode 等阶段；两者计算特征不同，因而常被解耦到不…"
---

# 客座讲座 · 推理系统：把电力变成智能（Dan Fu）

> **Stanford CS336 语言模型从零构建 · 客座** ｜ 讲者 Stanford Online ｜ 72 分钟 ｜ [原视频](https://www.youtube.com/watch?v=9EEm4iMAF5s)

## 核心结论

这场客座讲座从推理系统视角讨论语言模型：训练完成后，真正把 GPU 算力转化为可用智能的是推理引擎。一次请求会经历调度、KV cache 复用、prefill 与 decode 等阶段；两者计算特征不同，因而常被解耦到不同机器。大规模服务中的连续批处理、KV cache 分层存储、缓存感知路由等系统优化能显著提升吞吐。后半部分用两个研究案例说明全栈创新的空间：Megakernel 通过融合解码算子逼近 GPU 带宽上限，Parcae 通过受约束的循环 Transformer 在固定参数下增加计算并稳定训练。

![讲座封面幻灯片：Inference: Turning Electricity into Intelligence](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0001_00-00-44.jpg)

## 从训练到推理：推理引擎是把电力变成智能的引擎 [00:05 - 08:10]

课程此前主要关注如何训练语言模型；本次讲座转向“模型训练好之后”的另一侧：如何 serving、如何做 inference，以及从推理系统出发能看到哪些研究问题。

规模是能力提升的重要驱动力。演讲者回顾：2018 年最大的模型约 1 亿参数；2019 年 GPT-2 被认为“太危险不宜发布”；今天开源模型已达万亿参数，前沿规模可能在 5 到 10 万亿参数。随之而来的是聊天、写代码、分析复杂文本等新能力。

演讲者用 20 世纪初曼哈顿的马匹做类比：1902 年有 13 万匹工作马，马粪成为城市问题，甚至有学术会议专门讨论；但到 1912 年，汽车数量已超过马匹。演讲者认为，对语言模型而言，这个“1912 时刻”可能就是去年——他本人和团队的大部分代码已经开始由语言模型辅助编写。

![历史对比幻灯片，引用1902年曼哈顿马匹数量数据论证变革速度](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0003_00-03-35.jpg)

在算力层面，演讲者称“GPU 是新的石油”，并提到数百亿美元级的数据中心投资。更重要的是，**推理是把电力变成智能的引擎**：模型本身只是数学上的 DAG（有向无环图），推理引擎和 GPU kernel 才把这些 ML 操作映射到硬件上。理解推理和底层 kernel，才能实现机器学习算法的全栈创新。

![GPU算力投资幻灯片，展示OpenAI Stargate项目及5000亿美元数据中心投资](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0004_00-04-47.jpg)

![Transformer模型DAG结构与GPU流式多处理器映射关系图](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0005_00-05-48.jpg)

本次讲座分为三部分：一次请求/token 的生命周期、用 Megakernel 加速 LLM 解码、以及新架构 Parcae。演讲者同时代表 UCSD 实验室和 Together AI。

## 一次请求的生命周期：从文本到 token [08:10 - 16:20]

当一个请求进入推理系统，首先会被调度到不同的 GPU 上；系统可能把 prefill 和 decode 解耦到不同机器。接着会用 KV cache 检查：这些 token 或类似前缀是否见过，能否复用已有计算。然后执行核心的机器学习计算，并根据模型大小和并行策略，把计算切分到多机、多节点或多 GPU。最后输出 token。

不同应用的 workload 形状差异很大：

- **编码类工作流**：输入可能长达数万 token（整个代码库），输出可能是思考 token 或较短回答。
- **摘要/长文档对话**：用户可能粘贴整本书，然后来回讨论。
- **普通聊天**：输入输出都较短。
- **Agentic 工作流**：多轮、工具调用、模型自己迭代；不同应用对首 token 延迟和整体响应时间有不同目标。

![LLM推理引擎从请求到Token生成的完整流程图](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0008_00-14-20.jpg)

请求处理的核心是 **prefill** 和 **decode**，二者特征非常不同：

- **Prefill**：例如一次输入 1 万个未见过的 token，输出 1 个 token。这是 compute-bound 操作，和训练时的前向传播很像，只是不做反向传播，可以充分利用 GPU 算力。
- **Decode**：逐 token 生成。每生成一个 token，都要把模型权重重新加载一遍；计算量不大，但受内存带宽限制。若使用 speculative decoding，可能一次生成 3 到 4 个 token。

生成 token 后，还要做停止符检查、安全检查、detokenize，并流式返回。推理引擎本质上在循环执行：调度 → 执行 → 采样 → 重复。

## 连续批处理与 KV cache 前缀共享 [16:20 - 19:55]

生产系统要同时处理大量请求，因此使用 **continuous batching**。时间向下流动，新请求可以在运行中途加入；短请求完成后释放槽位，长请求可能因为 GPU 内存不足而排队。系统需要检查 KV block 容量、microbatch 是否超过最大 token 数，并把长 prompt 的 prefill 切成 chunk，与 decode 交错执行。

![连续批处理(Continuous Batching)的时间线示意图，展示多个请求在不同步骤中的调度与槽位释放过程](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0010_00-16-43.jpg)

**KV cache** 是另一个关键机制。很多用户都会说“hi ChatGPT”之类相同前缀；对于长文档，也可以先对整段做一次 prefill，后续对话轮次不必重新计算全部内容。系统可以用树结构查找哪些 token 见过、哪些是新的，并直接查找已有激活值。这种 prefix sharing 能显著减少重复计算。

## 模型切分与 prefill/decode 解耦 [19:55 - 21:50]

万亿参数模型无法完整放进单张 GPU，因此需要切分：

- **Tensor parallelism**：把每个 tensor 切分到多张 GPU 上。
- **Expert parallelism**：对 MoE 模型，把不同专家放到不同 GPU 上，由 router 把 token 路由到对应专家。

切分方式会决定瓶颈、所需 GPU 数量以及可同时服务的会话数。

![张量并行与专家并行的对比架构图，展示权重分片与路由分发机制](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0011_00-18-59.jpg)

Prefill 和 decode 的计算特征不同，因此常被放到不同机器上：prefill 是 compute-heavy，decode 是 memory-bandwidth-heavy；prefill 每个 prompt 只跑一次，decode 每生成一个 token 跑一次。将二者解耦后，可以针对不同阶段选择不同硬件。演讲者举例提到 Nvidia 收购 Groq、OpenAI 与 Cerebras 的合作，以及 SambaNova 等公司，都是在推理硬件上做差异化下注。

![Prefill与Decode解耦架构流程图，展示不同阶段的并行优化策略](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0012_00-20-18.jpg)

## 大规模生产中的 bug：NaN、工具调用死循环与“随机中文” [21:50 - 25:30]

当系统每天服务数万亿 token 时，小规模测试正常的东西可能在极低概率事件上出错。演讲者分享了几个生产 bug：

- **NaN bug**：某个 kernel 有极细微错误，触发条件很罕见；计算中途 logits 变成 NaN，模型开始重复输出同一个 token，例如“hi hi hi”或大量感叹号。
- **工具调用合并 bug**：模型本应发出“去搜索”的工具调用然后返回；但工具调用没有被正确处理，completion 长度暴涨，模型反复说“去搜索”，形成数万 token 的 doom loop。
- **随机输出中文**：某次故障被归咎于量化，但实际是 kernel 的 off-by-one 错误，读入了未初始化内存；经过 attention 后产生一个随机中文字符，模型随后“合理化”这个字符，继续用中文回答。

这些案例说明，生产环境的可观测性能够捕捉单元测试覆盖不到的问题。

![生产环境调试案例：展示NaN错误、工具调用合并错误和KV长度错误的监控图表](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0013_00-22-42.jpg)

## KV cache 的分层存储、预取与共享 [25:30 - 29:55]

生产系统希望 KV cache 尽可能大，以缓存更多用户、更多会话。存储层次通常是：

1. **GPU HBM**：热数据；
2. **CPU DRAM**：温数据；
3. **NVMe/SSD**：冷数据。

当 GPU 显存放不下时，系统会按 LRU 等策略驱逐、预取，并在 prefill worker 之间通过 RDMA 共享缓存。理想情况下，请求真正开始运行时，所需 KV cache 已经被预取到 GPU 中。

![KV缓存优化架构图：展示GPU HBM、CPU内存和NVMe磁盘之间的数据流动及预取机制](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0014_00-26-17.jpg)

![KV Cache 的 Offload、Prefetch 和 Share 机制架构图](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0015_00-29-00.jpg)

演讲者提到，CPU 性能因此重新变得重要：如果价值 5 亿美元的机器被上面几千美元的 CPU 卡住，会很不划算。类似地，SSD 和 DRAM 的采购也与 KV cache 存储需求有关。这里的问题很像操作系统中的经典调度：无法预测未来时用启发式；如果能预测——例如用户打开一个月前的旧对话，往往意味着马上要提问——就可以提前预取。总体上，目标是在满足 SLA 的前提下，把尽可能多的流量压到 GPU 上。

## NVL72、宽专家并行与故障容错 [29:55 - 31:30]

Blackwell 世代的 NVL72 把 72 张 GPU 用高速互连连接起来。自然的问题是：如何把万亿参数模型切分到 72 张 GPU 上？这样做能带来什么？如何处理故障？

大规模系统中 GPU 可能频繁故障，例如连接器是塑料而非金属，插拔过多会导致 NVLink 不稳定。若模型被切分到 64 张 GPU 上服务数百万用户，单张 GPU 挂掉时必须有容错方案。对宽专家并行（wide EP），节点死亡会丢失专家和正在传输的 KV cache；系统需要把孤儿专家重新分配到剩余 GPU，并保留热备。百万级上下文也带来新问题：是否要把上下文切分到多张 GPU 上处理。

![NVLink72上宽专家并行(Wide EP)的前向传播流程及节点故障容错机制示意图](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0016_00-29-55.jpg)

## Cache-aware prefill-decode disaggregation [31:30 - 34:05]

一个简单但有效的系统优化是 **cache-aware prefill-decode disaggregation（CPD）**。大多数请求是多轮对话中的后续请求，缓存命中率高；假设平均对话 10 轮，则约 10% 是全新的冷请求。冷请求可能包含数千 token，prefill 成本很高；如果把这种长 prefill 和“1+1 为什么等于 2”这类短对话放在同一组 GPU 上，会互相干扰。

做法是在路由层加两行逻辑：低缓存命中的冷请求发到一组 GPU；高缓存命中的温请求发到另一组 prefill 节点。这样可以把相似工作负载聚在一起。演讲者称，这个简单优化最多可带来 40% 的 serving 加速。

![缓存感知的预填充-解码分离(CPD)系统架构及分布式KV缓存读写流程](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0017_00-31-43.jpg)

![CPD方案在不同节点配置下相比基线方案的QPS性能对比柱状图](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0018_00-33-27.jpg)

演讲者认为，这类系统研究现在还非常早期；10 到 20 年后，人们可能会觉得这些做法理所当然。

## Megakernel：把解码从“内存加载器”中救出来 [34:05 - 41:50]

Decode 的根本挑战是：为了生成一个 token，必须跑完整个模型。GPU 本来有大量并行度，但在 batch size 为 1 的解码中，系统更像一个“昂贵的显存加载器”，而不是大规模并行计算机。

传统写法是每个操作写一个 kernel，例如 norm kernel、matmul kernel、attention kernel。这样编程简单，但会引入大量 downtime：kernel launch/teardown、tail effect（短输入等待长输入）、kernel 之间的空隙不断累加。

**Megakernel** 的思路是：不要每个操作单独写 kernel，而是用一个 kernel 覆盖多个操作，进行更激进的 fusion。它把 GPU 看成一个分布式系统：有一堆带依赖关系的工作，需要调度这些工作、把任务分配到不同的 streaming multiprocessor 上，以最大化利用率。

![演讲大纲页：包含token生命周期、Megakernels加速LLM推理及Parcae新方案](/coursenotes/cs336-language-modeling-sp26/guest-dan-fu-inference-systems/kf_0006_00-06-28.jpg)

在 attention 推理中，这种做法带来 30% 到 70% 的加速；进一步可以把整个 Llama 1B 的一层放进一个 kernel。细粒度控制允许重叠操作，例如：

- 在 QKV + RoPE 还没算完时，就开始加载 KV cache；
- 在 attention 结束前，就开始加载 O projection 的权重；
- 把下一层的权重加载与当前 attention 重叠；
- 在 attention 结束前开始部分 reduction。

实现上，他们使用 **ThunderKittens**：一个低层 kernel 编写库，类似 Triton 但控制更细，提供基于指令的抽象和虚拟化共享内存。最终在 H100 上达到 72% 的带宽利用率，接近该操作的“光速”上限。

代价是 Megakernel 非常 labor-intensive；这一点在问答部分进一步讨论。

## Parcae：用受约束的循环 Transformer 稳定训练 [41:50 - 59:30]

Parcae 来自 UCSD 实验室，合作者包括 Hayden、Zachary 和 Taylor。它属于 **looped transformer** 路线：让 token 多次经过同一组 Transformer block，而不是每层只走一次。

这样做有几个动机：

- 参数数量不变，但循环次数提供一个调节 FLOPs 的旋钮；
- 循环模型可能具有更高表达性；
- 关注“每参数质量”或“每参数+数据智能”，而不只是堆参数。

此前已有 loop transformer 工作，也有关于某模型是 looped LM 的 Twitter 传言；演讲者认为传言不实，但 loop 模型本身值得研究。

问题是，之前的 loop 模型很难训练：学习率稍微一变，9/10 的情况会崩，出现 NaN 或巨大 loss spike。以前的 hack 是每层加 norm，或者只选 2e-4 这种特定学习率。演讲者认为 loss spike 说明背后有更深层原因。

他们的分析从 **残差流** 入手：经验观察发现，每个 residual block 对激活值的改变并不大。于是把整个非线性 Transformer block 放进一个黑箱 \(r\)，把循环过程写成残差上的动态系统，只留下两个矩阵：

- \(B\)：对初始向量的变换；
- \(A\)：每次循环对残差的变换。

之前的 loop 模型要么把 \(A\) 当作 identity（直接相加），要么用完全可学习的矩阵。演讲者发现，\(A\) 和 \(B\) 的幅度支配了整个系统；如果暂时丢掉非线性项 \(r\)，可以得到闭式解，其中 \(A^t\) 起主导作用。若 \(A\) 的谱半径 \(\rho(A)\) 大于 1，循环多次后激活会爆炸，这解释了 loss spike。

Parcae 的做法是约束 \(A\) 和 \(B\)：

- 把 \(A\) 约束为负对角矩阵，使其幂次最终衰减到 0；
- 对 \(B\) 施加简单的线性 norm，因为它只应用一次，不易爆炸；
- 从而让谱半径小于 1，使系统稳定。

训练结果上，Parcae 在 6e-4 学习率下也能稳定收敛；不约束的基线会爆炸到 \(10^{19}\)。只加 norm 的模型会出现两种相反压力：模型想扩大激活以获得表达空间，norm 又把它压回去，导致 loss spike。Parcae 同时约束状态 norm，训练更稳定。

在质量上，Parcae 优于之前的 recurrent depth 模型，也优于强 Transformer 基线（例如 nanochat 一类追求快速训练的模型）。在相同基本架构上引入循环并稳定后，困惑度和端到端质量都更好。

Scaling law 部分，演讲者先回顾经典的 iso-param/iso-FLOP 曲线：如果曲线向右下倾斜，说明应同时扩大数据和参数。Parcae 的实验显示，循环次数的曲线也向右下倾斜：**当数据增加时，也应增加循环次数**。他们发现循环与 token 数联合遵循可预测的幂律。当前主流模型几乎没有循环，相当于处在曲线最左端、同时使用大量数据；这暗示可能存在更优的训练方式。

另一个实验固定模型大小和 FLOP 预算：橙色曲线是固定深度 Transformer，蓝色曲线是循环模型；在相同 FLOPs 下，循环模型用不同数据量训练，能得到更低的验证损失。这进一步支持“在大规模预训练中引入循环”的可能。

## 问答：设计权衡与开放问题 [59:30 - 71:40]

**能否在预训练模型上做循环？**  
演讲者提到，有人把 Qwen 模型的 2 到 3 层循环起来，在某些数学任务上质量提升；团队也在跟进。对预训练模型做少量循环可能有效，但原因尚不清楚，值得研究激活和权重。

**循环对推理有什么好处？**  
参数更少意味着可以放更多 KV cache、减少通信、用更少 GPU。演讲者还设想：如果 recurrent block 足够小，可以写一个 Megakernel 在循环中快速执行。Groq LPU 之类芯片内存很小（约 250MB），也许可以设计刚好放进去的模型，让权重常驻，只流式处理激活。

**Megakernel 的代价是什么？**  
非常 labor-intensive。演讲者估计，一个有才华的 kernel 工程师一年可能只能为一种硬件、两三个模型、batch size 1 到 16 写 Megakernel；batch size 变成 17 可能就要重写。Together 正在尝试用编译器自动化部分工作。Megakernel 在 GPU 编程史上经历过高峰和低谷；如果能写出来，速度会非常快，但需要大量精力。

**如果知道目标硬件，架构应如何设计？**  
首先受内存约束：先看目标芯片（如 Cerebras wafer、Groq LPU）有多少内存，再决定模型大小，确保放得下模型和足够 KV cache。中国模型的一些量化选择可能反映了对华为芯片的考虑。Nvidia 生态有 NVFP4，AMD 等生态使用 MXFP4，各有取舍。

**compute-optimal 下，应该循环还是增大参数？**  
compute-optimal 本质是在给定 FLOP 预算下选点。如果想要更高质量，就增加 FLOP 预算；如果模型大小固定，就训练更久或循环更多；如果数据耗尽，就选择合适模型大小。部署成本、开源后用户能否在笔记本上运行等，也会影响模型大小选择。一般来说，更大模型加更多数据通常更好。

**不同用例对架构有什么影响？**  
Agentic 工作流希望 KV cache 保持“热”；批处理场景每个文档只出现一次，KV cache 没那么重要。DeepSeek MLA 是对 KV cache 的激进压缩；FP8/FP4 KV cache 也会显著改变缓存大小。另一个大区别是 causal 与 non-causal attention：Google 搜索长期使用 BERT 类双向模型做批处理，因为不需要生成大量 token；T5 则是双向处理加生成的中间方案。

**Megakernel 如何处理多 GPU 通信？**  
可以把 NCCL 调用也融合进 Megakernel，但尚未找到特别杀手级的用例，因为有时会受 NCCL 调用延迟本身限制。DeepSeek 曾为 MoE 推理层发布融合通信的 Megakernel。未来更可能是“部分计算的 Megakernel”，而不是整个模型一个 Megakernel；除非愿意付出巨大的工程代价。

## 要点回顾

- 推理引擎是循环：调度 → 执行 → 采样；prefill 是 compute-bound，decode 是 memory-bandwidth-bound。
- 连续批处理让请求中途加入；KV cache 通过前缀共享复用激活，减少重复 prefill。
- 大模型需要 tensor/expert parallelism；prefill 与 decode 常解耦到不同机器，以匹配不同硬件特征。
- 大规模生产会出现罕见但严重的 bug：NaN 导致重复 token、工具调用死循环、off-by-one 导致随机中文；可观测性至关重要。
- KV cache 可分层放在 GPU HBM、CPU DRAM、NVMe，并通过 LRU、预取和 RDMA 共享提升命中率。
- Cache-aware prefill-decode disaggregation 通过路由冷/热请求，最多带来约 40% 的 serving 加速。
- Megakernel 融合多个解码算子，重叠权重加载与计算，在 H100 上达到 72% 带宽利用率；代价是极高的工程成本。
- Parcae 用残差动态系统和谱半径约束稳定 loop transformer；scaling law 显示数据增加时应增加循环次数。
