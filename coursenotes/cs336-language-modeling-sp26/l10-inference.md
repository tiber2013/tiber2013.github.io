---
title: "第 10 讲 · 推理"
description: "推理是模型训练完成后反复发生的成本，在智能体（agent）时代，生成 token 的数量几乎等同于投入的计算量。与训练可并行处理整个序列不同，自回归推理必须逐 token 生成，因此 generation 阶段是内存受限…"
---

# 第 10 讲 · 推理

> **Stanford CS336 语言模型从零构建 · L10** ｜ 讲者 Stanford Online ｜ 85 分钟 ｜ [原视频](https://www.youtube.com/watch?v=EfM546A79aM)

> **v2 详写版** · 按 20 分钟分段详写，配图覆盖全片；文末附「对照参考笔记补遗（Spring 2025 版）」，记录与 2025 版讲义笔记的核对结果。

## 核心结论

本讲从缩放定律转入推理，核心判断是效率不再是工程细节，而是决定产品能否成立的变量：训练是一次性成本，推理却每天重复付出，OpenAI 每天约处理 8.6 万亿 token，不到 4 天生成量就反超一次 32T token 训练；进入智能体范式后，内部思考轨迹产生的 token 不再供人阅读而直接是计算本身，「生成的 token 等于花掉的算力」，优化因此没有自然上限。但「快」不是一个数：TTFT、latency 与 throughput 分属交互与批处理且彼此权衡，而自回归逐 token 生成无法在序列维度并行，使 generation 阶段算术强度塌缩到不足 1，注定被 HBM 带宽卡住，这正是「推理是内存受限」的由来。讲师先用 scaling book 的记号与算术强度，把 prefill 与 generation 两阶段的 FLOPs 和字节搬运量算成可与 H100 约 295 的加速器强度比较的不等式，再代入 Llama 2 13B 的配置，量化出延迟随 batch 线性变差、吞吐渐近饱和的取舍。后半讲沿「减小 KV cache 但不伤精度」展开技术版图：GQA、MLA 与跨层共享等有损架构改动，量化、剪枝与蒸馏，数学上可证无损的投机解码，以及连续批处理与 PagedAttention 这类系统层调度；讲师提醒经验性精度结论须按自己的任务重测，并把最大空间留给天生适合推理的新架构。

## 推理问题的定义、出现场景与「一次性成本 / 重复性成本」 [00:05 - 02:40]

本讲从一次刻意的「插曲」开始：上一讲由 Tatsu 讲的是 scaling laws（缩放定律），而这一次要暂时放下它，转入 inference（推理）。讲师给推理下的定义极简：模型已经训练完成，给定一个 prompt（提示词），目标是尽可能准确、尽可能快地产生 response（响应），用转写里的原话说就是 "as accurately and as quickly as you can"。准确与速度这两个词在整门课里被反复权衡——本课程的主线是 efficiency（效率），而推理之所以值得单独用一讲来谈，恰恰是因为在这里效率不再只是训练阶段的工程细节，而是决定产品能否成立的核心变量。讲师特别强调，推理只占一讲，但它的「重要性在不断增长」（of growing importance），并且在许多地方都会出现，随后用一句 "So, inference shows up all over the place" 收束了开场。

画面上用一段带行号的伪代码 `def Landscape():` 铺开「推理出现在哪些地方」，这比转写里的口头列举更完整。代码第 64–67 行写道 "Inference shows up in many places:"，随后三条 bullet 分别是 `Actual use (chatbots, code completion, agents, batch data processing)`（实际使用：聊天机器人、代码补全、智能体、批量数据处理）、`Model evaluation (e.g., on instruction following)`（模型评估，例如指令跟随）、`Reinforcement learning (sample many generations, then apply score)`（强化学习：采样大量生成结果再打分）。转写里对此有对应补充：评估中特别指「需要生成」的那类评估（evaluation that requires generation）；强化学习里需要模型生成 rollouts（轨迹采样），对其打分，再据此更新权重。把这三条连起来看，推理并不是「训练完才开始」的附属步骤，它同时嵌在评估回路与训练回路内部；这一点是理解后面 RL 推理负载的伏笔，也说明为什么讲师在开头就说效率这门课的主题「比以往任何时候都更重要」（matters more than ever here）。

真正把效率推到台面上的，是成本结构。讲师指出 training 是 one-time cost（一次性成本）：它可能极其昂贵，但做完就结束了；而 inference 是 repeated cost（重复性成本），"you kind of incur them every single day"——每天都要重新付出。幻灯片第 69 行把这句结论高亮成 "Why **efficiency** matters: training is one-time cost, inference is repeated many times"。紧接着第 70–71 行给出两个量级：`OpenAI processes ~8.6T tokens per day [article]`（OpenAI 每天约处理 8.6 万亿 token），以及 `For reference, DeepSeek v4 was trained on 32T tokens`，并附有 HuggingFace 仓库里 `DeepSeek_V4.pdf` 的链接。这里有一个只出现在画面上的细节值得记下：幻灯片写的参照对象是 DeepSeek v4 的 32T tokens，而讲师口头上说的是「GPT-4 当年早些时候发布，训练用了 32 trillion tokens」；两者数值一致（都是 32T），只是参照模型不同，笔记按各自原文分别保留，不擅自合并。另外，画面顶部还残留着上一讲的 Summary 残影，写着 "Inference is important (actual use, evaluation, reinforcement learning)"、"Different characteristics compared to training (memory-bound, dynamic)"、"Techniques: new architectures, quantization, pruning/distillation, speculative sampling"、"Ideas from systems (speculative execution, paging)" 以及 "New architectures have huge potential for improvement"，相当于把整讲的技术版图提前摆了出来。

有了这两个数字，就能做一个很直观的除法：

$$\frac{32\times10^{12}}{8.6\times10^{12}} \approx 3.72\ \text{天},$$

也就是说不到 4 天，OpenAI 需要产出的 token 数量就超过了一次 32T token 规模训练的量级；转写里说 "in less than 4 days ... therefore the compute is at least the GPT-4"，并把结论收敛为「推理侧生成的 token 数会更高」。即便考虑到前沿模型可能用了更多训练 token，每日累计的推理生成量仍然是更大的那个数。需要留意的易错点是：8.6T 是带 `[article]` 来源标注的估算值而非精确计量；而且「token 数 ≈ 计算量」只是统领全局的一阶近似——不同 token 因所处上下文长度不同，其 attention 代价并不相同，本讲后面会用 arithmetic intensity（算术强度）把这个近似拆得更细。此外，把「每天」与「一次性」直接相除时，隐含假设是单位 token 的推理成本与训练成本可比，讲师本人也用 "you know, at least" 做了保留。这一章建立的成本视角，会在下一章被 agent 工作流进一步放大，并在第 6 章落到 H100 的具体数值阈值上。

![Landscape 代码页：推理出现的场景、8.6T / 32T token 量级与一次性成本对比](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0002_00-02-00.jpg)

## 从聊天机器人到智能体：生成的 token 就是投入的算力 [02:40 - 05:02]

推理的重要性在过去一年明显上升，原因不是模型变大了，而是使用范式变了。讲师回忆，曾经我们把语言模型主要想成 chatbots（聊天机器人）或 assistants（助手）：输入一个 prompt，拿回一个 response，并且默认「人类的目的是读这个回答」。幻灯片把这种旧范式写成一句带判断的 bullet：`Chatbots: most tokens are meant for human consumption (humans are bottleneck)`——大部分 token 是给人消费的，而 humans are bottleneck（人类才是瓶颈）。这句判断看似朴素，却是整讲的分水岭：如果输出最终要被人眼读完，那么推理速度一旦快过人类阅读速度，再往上挤速度就几乎没有额外收益。转写里把这个意思说得更直白：在 chatbot 世界里，推理到了某个点之后「够快了」，因为 humans can only read so fast（人只能读这么快）。

进入 agentic world（智能体世界）之后，链路被拉长了。一个 query 进来，agent 会 think（思考）、reason（推理）、call some tools（调用工具）、introspect（自省），最后才产出少量给人类读的结果。幻灯片用一条箭头把这个过程原样画出来，并高亮了第 75 行：

$$\text{query} \longrightarrow \text{internal trace (think / reason / call tools / introspect)} \longrightarrow \text{output for human},$$

后面还跟着半句被截掉的 "number of tokens generated can grow unbo..."（生成的 token 数可以无界增长）。紧接着第 76 行是整讲最凝练的等式：`Tokens generated = compute spent`（生成的 token = 花掉的计算量）。转写对此作了展开：agent 产生的大部分 token 其实不是给人读的，"you should really think about the number of tokens generated as really the compute spent"；而且 "there's no sort of limit"——只要问题足够有野心（ambitious enough problem），就需要非常多的算力和非常多的 token。于是可以写成一个便于后面推导的比例关系：

$$N_{\text{total}} = N_{\text{trace}} + N_{\text{out}},\quad N_{\text{trace}} \gg N_{\text{out}},\qquad \text{compute} \propto N_{\text{total}} \approx N_{\text{tokens}}\times(\text{FLOPs/token}).$$

这里的直觉是：内部轨迹的 token 不再是「输出」，而是「计算过程的载体」，因此把它们压掉就等于直接减少算力开销，而不是简单地改善体验。这与前一章「推理是每天都要付的重复成本」叠加后，结论就变成了：推理优化没有自然上限，"there's no kind of limit to how much value you can get out of squeezing more out of inference"。

商业侧的全景同样只在画面上完整给出。幻灯片列了 `Companies doing inference (a big deal for anyone who has a product or platform):`，下分两类：`Providers serving closed models (OpenAI, Anthropic, Google, etc.)`（服务闭源模型）与 `Providers serving open-weight models (Together, Fireworks, Baseten, DeepInfra, Groq, Cerebras, ...)`（服务开放权重模型）。开源软件包则逐条带出处：`vLLM: from Berkeley, pioneered PagedAttention, popular and good default [GitHub]`、`SGLang: from Berkeley, pioneered RadixAttention, good for agentic workloads [project]`、`TensorRT-LLM: from NVIDIA, highly optimized for GPUs [article]`（该行被黄色高亮）、`llama.cpp: C++ only, supports CPU inference, runs locally [GitHub]`。转写里的口头版本略有出入但可互相印证：vLLM 是 "kind of a go-to"（首选）、SGLang "particularly good for agentic workloads, but maybe not as popular yet"、TensorRT 来自 NVIDIA "really fast, but it's more narrow"、llama.cpp 则是在 CPU 上跑推理的热门选择。把这些名字与 PagedAttention、RadixAttention 两项技术对应起来，正好预告了后面要讲的系统层优化。

讲师用一句带数字的判断收尾：如果能把推理做到 twice as fast（两倍快），"or even 10% faster, that is a big deal"。这里的 10% 之所以值得单独提，是因为它作用在「每天重复发生」的成本上，会被 8.6T tokens/day 这样的分母放大成可观的绝对量。易错点有两处：其一，「tokens = compute」是刻意粗粒度的一阶代理量，不同位置、不同上下文长度的 token 实际 FLOPs 并不相同，不能拿它做精确计费；其二，「SGLang 还没那么流行」「TensorRT 更窄」都是带时间状语的判断（本讲为 Spring 2026），不宜当成永恒结论。另一个容易被忽略的局限是：这一章只论证了「为什么值得优化」，并没有回答「快到底指什么」——正是这个问题把讲述推向下一章的三个度量指标。

![推理产业全景：智能体轨迹、闭源/开源提供商与 vLLM、SGLang、TensorRT-LLM、llama.cpp](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0004_00-04-40.jpg)

## 「快」的三种度量与自回归无法并行序列维度的根本困难 [05:02 - 08:15]

既然要谈「快」，第一件事是承认「快」不是一个数。讲师明确说 "there's a few metrics that capture the notion of fast, and they're going to be applicable in different settings and have different trade-offs"——不同场景用不同指标，而且指标之间存在权衡。幻灯片把三者连同单位一并列出，这是转写里没有完整保留的信息：`Time-to-first-token (TTFT): how long user waits before any generation happens (for interactive applications)`；`Latency (seconds/token): how fast tokens appear for *one* query (for interactive applications)`（该行被高亮）；`Throughput (tokens/second): how fast tokens appear for *many* queries (for batch processing)`。注意单位：latency 是 seconds/token，throughput 是 tokens/second，二者互为倒数关系；转写里也说 "latency or one over latency and throughput are clearly very related"。于是可以写出

$$\text{per-stream rate} = \frac{1}{\text{seconds/token}},\qquad \text{throughput} \sim \frac{1}{\text{latency}}\ (\text{仅为一阶相关，非严格等式}).$$

TTFT（首 token 延迟）刻画的是用户提交 query 到第一个 token 出现之前的等待：转写里描述为向 ChatGPT 输入问题后「若干毫秒过去」，第一个 token 才来。它对交互式应用最要紧，因为那段延迟里用户 "just waiting and doing nothing"（干等），越长体验越差；而一旦 token 开始流出，后续速度就不必极快，反正人也读不了那么快。Latency 则从单个用户视角衡量 token 流出的快慢，同样服务于交互场景。Throughput 换到系统视角，衡量「很多 query 加起来」每秒吐出多少 token，服务于批量处理：讲师举的例子是要用语言模型处理 a petabyte of data（一 PB 数据），只关心「这个作业什么时候干完」，不关心其中某一条 query 是早一点还是晚一点返回。

三者并非总能一起变好。转写里先说 "many interventions will make latency and throughput better"，紧接着又埋了一句 "as we'll see later, there is actually a trade-off here"。这个伏笔很关键：后面通过 batch size（批大小）调节时，小 batch 有利于 latency 与 TTFT，大 batch 有利于 throughput，二者会明确冲突。因此把「快」说成单一数字是本章最典型的易错点——正确做法是先声明工作负载（交互式还是批处理）再选指标。另一个易错点是 TTFT 与 latency 容易被混为一谈：前者只覆盖「第一个 token 之前」的空窗，后者覆盖「token 与 token 之间」的节奏。

那么，是什么决定了上述任何一个指标下的效率？幻灯片在 "What governs efficiency?" 下给出两条对照，其中第一条被高亮：`Training (supervised): you see all tokens, can parallelize over sequence (matmul in Transformer)`，以及 `Inference: you have to generate sequentially, can't parallelize over generation, so harder to fully utilize compute`。转写把这条「本讲最值得记住的高层结论」讲得更细：训练（有监督微调）时所有 token 一次可见，可以沿序列维度并行；在 Transformer 里，attention 与 MLP 的计算中 the sequence is just a dimension（序列只是一个维度），整体就是一个大张量被做矩阵乘法，于是 "you essentially process all the tokens at once"。推理不行：因为 autoregressive（自回归）性质，必须一个 token 一个 token 顺序生成，无法在序列维度上并行。用第 4 章将要正式引入的 einsum 记号，可以把这个对照写成

```python
# 训练：序列维 T 可一次性并行，T 是自由维
Y = einsum("btd,df->btf", X, W)     # X: [B, T, D] -> Y: [B, T, F]

# 推理 generation：每步只新增一个 token，等价于 T = 1 的瘦矩阵乘
y_t = einsum("bd,df->bf", x_t, W)   # x_t: [B, D] -> y_t: [B, F]
```

正因如此，讲师给出因果链：不能并行化序列维度 → "it's going to be harder to have high arithmetic intensity or fully utilize the compute"（更难获得高算术强度、更难把算力吃满）。这既解释了为什么推理是与训练截然不同的 workload，也直接预告了下一章的路线图与第 6 章的算术强度计算。需要留意的局限是：这里的「不能并行」特指**序列维度**；跨请求的 batching 仍然可以恢复并行度，这也是后面 throughput 能被救回来的原因。把「推理无法并行」误读成「推理完全无法并行」会与后文矛盾。此外，讲师在本节末尾用 "what determines the efficiency by either any of these metrics of inference?" 自问，却没有立刻给公式，而是先花两章补记号与算术强度的工具——这种「先立工具再算账」的顺序本身也是理解本讲结构的线索。

![三个「快」的指标定义与训练/推理效率根源对照](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0006_00-07-20.jpg)

## 本讲路线图、scaling book 记号体系与类 einsum 的维度图 [08:15 - 11:00]

在动笔算账之前，讲师先给了一张路线图，并声明「这基本是一次复习，主要是把记号换进来、确保大家对 notation 达成一致」。转写里的口头版本是：先做些数学，理解一个 transformer 的 arithmetic intensity、throughput 与 latency 该怎么想；再谈各类降本手段——把 KV cache 变小、quantization（量化）、model pruning（模型剪枝）；最后是 speculative decoding（投机解码）与一些 practical concerns（工程实务）。画面上的版本则用一串函数名把这份大纲写得更精确，出现在 `kf_0007` 里：`review_of_arithmetic_intensity()`、`arithmetic_intensity_of_inference()`、`throughput_and_latency()`；随后是 "Taking shortcuts (lossy)"（走捷径，有损）小标题下的 `reduce_kv_cache_size()`、`quantization()`、`model_pruning()`，并附一句 "Summary: reduce inference complexity without hurting accuracy"（降低推理复杂度但不伤精度）。再往下是两条配方："From scratch recipe: 1. Define faster model architecture / 2. Train faster model" 与 "Distillation recipe:"（蒸馏配方，未展开）。"lossy"（有损）与 "without hurting accuracy" 并置，已经把本讲后半段的核心张力点明了：绝大多数加速手段都要拿精度去换。

素材来源也被明确交代：本讲大部分内容基于 Google 那本关于 transformers 与 inference 的 scaling book，讲师推荐大家去读，并坦承 "some of the figures are generously lifted from that book"（部分图直接取自该书）。这解释了为什么后面的电路图与代码块带着 `def ...():` 的可执行外形——它们本质上是讲义里可逐行推演的伪代码，而不是装饰性插图。理解这一点，对下一章逐行读 Transformer block 电路图很重要：图上的每一个字母都是可代入的维度，不是示意图标签。

记号页的标题是 `def review_transformer():`，副标题 `[Scaling book chapter on Transformers]`，其下写着 "Notation (similar to einops): Symbols denote dimensions (and their length): B (batch), T (sequence), D (model dim), H (head dim)"。括号里的 "and their length" 是个容易漏掉的约定：这些符号既表示**轴**，也表示该轴的**长度**，所以才能直接出现在后面的 FLOPs 与字节数算式里。转写把四个字母逐个落实：B 是 batch 维度、也是序列条数；T 是 sequence 维度、也是 token 数；D 是 model dimension（模型维度）；H 是 head dimension（头维度）。紧接着给出被高亮的示例：

$$BTD \times DH \rightarrow BTH.$$

讲师解释了颜色语义，这部分是读懂后面电路图的钥匙：标红的维度是 contracting（收缩）维，"appear in both operands and disappear from the result"（出现在两个操作数里、在结果中消失）；黑色的 regular（常规）维 "appear in one operand and stay in result"（只出现在一个操作数里并保留）；蓝色的 batching（批处理）维 "appear in both but stay in the result"，即出现在两个操作数中却保留、不参与收缩或归约。幻灯片还给了第二个例子 `BD x BD → B`，用来对照说明「两个操作数都有 B、结果仍有 B」属于 batching 而非 contracting。用带语言标注的代码块写出来就是：

```python
# 红色 D 为收缩维，蓝色 B 为批处理维
Y = einsum("btd,dh->bth", X, W)      # X:[B,T,D], W:[D,H] -> Y:[B,T,H]
Z = einsum("bd,bd->b", A, Bmat)      # 两个操作数都含 B，结果保留 B：batching
```

讲师补了一句直觉解释：这个乘法的含义是「对每一个序列、每一个 token 位置都有一个向量，把它乘上一个矩阵」，本质上是 "taking a bunch of dot products between two tensors"。易错点恰恰在颜色上：蓝色与红色都「出现在两个操作数中」，唯一判据是**是否出现在结果里**，只看「出现两次」会误把 batching 当成收缩。而讲师在此处特意留了话头——"we'll come back to why this B being both makes inference hard, why attention is kind of a bottleneck there"（后面会回到：为什么 B 同时出现在两边会让推理变难、为什么 attention 是瓶颈）。把这个「蓝色 B」记住，第 5 章的 attention 电路图和第 6 章「B 很小就 memory-bound」的结论就串成了一条线。此外，本章的时间边界也值得注意：路线图与记号都属于「工具铺垫」，本身不产出任何性能数字，真正的数值要到 14:22 之后才出现。

![review_transformer 记号页：B/T/D/H 定义、BTD x DH → BTH 示例与红黑蓝维度约定](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0008_00-10-01.jpg)

## Transformer block 电路图：Attention / MLP 的张量形状与全部符号约定 [11:00 - 14:22]

讲师把接下来这张图称为 "in all its full glory is the description of a transformer block"，并评价说它像一张 circuit diagram（电路图），是「我见过最清晰的 transformer 定义」：很多文字描述让人看不清到底发生了什么，而这张图直接给出张量的确切形状，从而可以据此推理依赖关系。结构上很直白：取某一层的激活 $X$，先过 attention，再过 MLP。转写里补了投影细节——$X$ 分别乘 query matrix、key matrix、value matrix；$N$ 是 number of heads（query 头数），$H$ 是每个头的维度，于是 query 矩阵形状是 batch × sequence × number of heads × head dimension，即 $B\times T\times N\times H$；而 K、V 不用 $N$，用的是一个「可能更小」的 $K$，也就是 key/value heads 数，形状为 $B\times S\times K\times H$。GQA（grouped-query attention，分组查询注意力）的全部动机就藏在这个 $K<N$ 里。

电路图把每一步的形状都标了出来，以下按图上原样逐行转述（红字为收缩维、蓝字为批处理维）：

```text
X : [B,T,D]
  |-- x W_Q [D,N,H] --> Q [B,T,N,H]        # BTD x DNH -> BTNH
  |-- x W_K [D,K,H] --> K [B,S,K,H]        # BSD x DKH -> BSKH
  |-- x W_V [D,K,H] --> V [B,S,K,H]        # BSD x DKH -> BSKH
Q : reshape BTNH -> BTKGH
Q.K : BTKGH x BSKH -> BTSKG                # 叠加 masks, 再做 softmax_s
attn: BTSKG x BSKH -> BTKGH
    : reshape BTKGH -> BTNH
    : BTNH x NHD -> BTD                    # x W_O
    : (+) 残差相加 -> 输出 [B,T,D]
```

图右侧的 symbol / dimension 表格是本段信息密度最高、转写里完全没有逐字念到的部分，必须完整记下：$B$ = batch；$L$ = number of layers（层数）；$T$ = sequence length (query)（query 侧序列长度）；$S$ = sequence length (key value)（key/value 侧序列长度）；$V$ = vocab（词表）；$D$ = d_model, embedding dimension；$F$ = MLP hidden dimension；$H$ = attention head dimension；$N$ = number of query heads；$K$ = number of key/value heads；$G$ = q heads per kv head = N // K。表里用 `N // K` 这种整除写法，说明 $G$ 被约定为整数。特别注意 $S$ 与 $T$ 被拆成两个符号：一个管 key/value 的长度，一个管 query 的长度，这正是后面 prefill / generation 两阶段能被统一描述的前提。

MLP 那一半同样有完整电路图：`norm D` 之后，$BTD \times DF \rightarrow BTF$ 做两次（其中一路过 `gelu`），逐元素相乘 $\ast$ 得 $BTF$，再 $BTF \times FD \rightarrow BTD$，与输入做残差 $\oplus$，最后 `norm D`；三个权重标为 $W_{In1}$、$W_{In2}$、$W_{Out}$。图下方 "Conventions:" 给出四条必须记住的约定，其中第四条被高亮：

$$F = 4D\ \text{(MLP up-projects into 4x the model dimension)},\qquad D = N H,\qquad N = K\cdot G,\qquad S = T.$$

括号里的原文分别是 "MLP up-projects into 4x the model dimension"、"model dimension split across N heads"、"for GQA, number of heads split across G groups"、"during training, condition on S input tokens to predict T output tokens"。转写对此逐一印证：见到 $F$ 就当作 $4D$；模型维度被拆到 $N$ 个头上所以 $D = N\cdot H$；GQA 情形下头数再拆成「组数 × 每组头数」；$S$ 与 $T$ 都表示序列维，但 $S$ 是输入 token 数、$T$ 是输出 token 数，训练时 $S=T$（用同样的输入预测同样的输出），而推理时 $T=1$、$S$ 是已有输入。于是

$$\text{training: } S = T;\qquad \text{inference generation: } T = 1,\ S = \text{上下文长度};\qquad G = N\,/\!/\,K.$$

这里的数字与参数很具体：上投影倍数 4、整除 `//`、推理时 $T=1$、K/V 各一路投影、共 11 行符号表。易错点有三处。第一，$F=4D$ 被明确标为 convention（本讲约定），不是普适定律，把它当成所有模型的硬约束会出错。第二，$T$ 的含义在不同位置有张力：表格写 "sequence length (query)"，口语里又说 $T$ 是「输出 token 数」、$S$ 是「输入 token 数」；二者在因果自回归下是一致的，但阅读时必须结合 $S$/$T$ 成对判断。第三，$K$ 的口径在 18:50 之后的问答里被现场修正（见第 7 章），不能只按表格字面把 $K$ 数成 KV head 数就完事。此外，$W_O$ 在图上写作 `NHD` 而非 `DNH`，输出回到 `BTD`，转置顺序是电路图里最容易看反的一处。这一章产出的 $B$、$K$、$H$、$L$、$S$、$T=1$ 会在后续 KV cache 与算术强度分析中反复登场，而 $T=1$ 正是下一章「很瘦的矩阵乘」的直接来源。

![Transformer block 电路图与完整 symbol / dimension 符号对照表](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0009_00-11-21.jpg)

![MLP 电路图与四条 Conventions：F=4D、D=NH、N=K*G、S=T](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0011_00-14-01.jpg)

## 算术强度回顾：矩阵乘的 FLOPs / 字节记账、化简为 B 与 H100 的 295 阈值 [14:22 - 18:45]

14:22 起讲师说 "another review for arithmetic intensity"，并提醒这是第二讲讲过的内容，这里当作 warm-up（热身）。设定是乘两个矩阵：$X$ 为 $B\times D$，$W$ 为 $D\times F$；直觉上 $B$ 是 batch 维度、$D$ 是 hidden / model dimension、$F$ 是 MLP 的 up-projection dimension。算术强度的定义是「每搬运一字节做多少计算」，所以要同时数 FLOPs 与 memory moved（内存搬运量）。幻灯片 `def review_of_arithmetic_intensity():` 用一段可执行的记账代码把这件事写死，注释里明确 "2 bytes for bf16"（推理时全部按 BF16 存储，每元素 2 字节）：

```python
def review_of_arithmetic_intensity():
    # Setup: multiply X (B x D) and W (D x F) matrix
    # Intuition: B is batch size, D is hidden dimension,
    #            F is up-projection dimension in MLP
    flops = 0
    bytes_transferred = 0

    # Perform the operation
    # 1. Read X (B x D) from HBM
    bytes_transferred += 2*B*D          # 2 bytes for bf16
    # 2. Read W (D x F) from HBM
    bytes_transferred += 2*D*F
    # 3. Compute Y = X (B x D) @ W (D x F)
    flops += 2*B*D*F
    # 4. Write Y (B x F) to HBM
    bytes_transferred += 2*B*F

    assert flops == 2*B*D*F
    assert bytes_transferred == 2*B*D + 2*D*F + 2*B*F
```

讲师点出这里的关键量级差：每次读写基本是「条目数」，是 quadratic-like term（二次量级），而 matmul 是 cubic（三次量级），"that's how you're going to get high arithmetic intensity"。他还特意说明全部用符号而非具体数字表示，是因为符号形式 "easier to see and work with"。于是算术强度为

$$\text{intensity} = \frac{\text{flops}}{\text{bytes transferred}} = \frac{2BDF}{2BD + 2DF + 2BF},$$

并强调 "we want this to be high"。接着做化简：讲义里写的是 `intensity = (flops / bytes_transferred).simplify()`，然后 "Assuming B is much less than D and F, then we can simplify:"，被高亮的那一行为 `intensity = intensity.subs(D, c*B).subs(F, c*B).limit(c, oo).simplify()`，末尾 `assert intensity == B`。转写把这段符号操作翻成直觉：假设 batch 维远小于 $D$ 与 $F$，令 $D = C\cdot B$、$F = C\cdot B$，再让 $C\to\infty$（$D$、$F$ 都很大而 $B$ 较小），整个式子就约简为

$$\text{intensity} \approx B.$$

讲师还把它与第二讲方阵矩阵乘的结论对照：那时得到的是 "something like N over three"（约 $N/3$），这里则是非方阵、多变量版本下的类比。这个 $B$ 非常要紧——它意味着矩阵乘的算术强度基本由 batch 大小决定，而不是由模型有多大决定。

下一步是把「计算的强度」与「硬件的强度」放到一起比。转写解释 accelerator intensity（加速器强度）的算法：查 spec sheet 里的 flops per second，查 HBM memory bandwidth（内存带宽），两者相除。幻灯片给出 H100 的具体代码与数值：`flops_per_second = 989e12`、`memory_bandwidth = 3.35e12`、`accelerator_intensity = flops_per_second / memory_bandwidth`、`assert round(accelerator_intensity) == 295`，右侧方框里同时显示 `intensity = B` 与 `accelerator_intensity = 295.2239`。也就是

$$\frac{989\times10^{12}}{3.35\times10^{12}} \approx 295.2239 \approx 295.$$

判据是两行："If computation intensity > accelerator intensity, **compute-bound** (good)" 与 "If computation intensity < accelerator intensity, **memory-bound** (bad)"，被高亮的结论句是 "Conclusion: compute-bound iff B > 295"。换句话说，对这个矩阵乘而言，只有 batch size 大于 295 才算把 H100 的算力吃满。紧接着是极端情形 "Extreme case (B = 1, corresponding to matrix-vector product):"，三条 bullet 为 "Arithmetic intensity: 1"、"Memory-bound (read D x F matrix, perform only 2 D F FLOPs)"、"This is basically what happens with inference..."。转写把这层意思说透：$B=1$ 时你仍要读整个 $D\times F$ 矩阵，却只做 $2DF$ 次 FLOPs，严重 memory-bound；"you don't get these like full matrices, you get these like very thin matrices or tensors"——这正是推理里常见的形态。

易错点集中在三处。其一，`989e12` 与 `3.35e12` 都是标称峰值（BF16 算力与 HBM 带宽），实际运行达不到，所以 295 是理想阈值而非实测常数；而且它绑定「这个矩阵乘」与 BF16 精度，换个算子或精度就要重算。其二，$\text{intensity}\approx B$ 依赖 $B\ll D,F$ 且 $C\to\infty$ 的极限假设，当 $B$ 与 $D$、$F$ 同量级时，分母里的 $2BD$ 与 $2BF$ 不可忽略，真实强度会明显低于 $B$。其三，`flops += 2*B*D*F` 里的系数 2 是把一次乘加（MAC）计为 2 FLOPs 的结果，与字节数里的系数 2（BF16 每元素 2 字节）含义完全不同，混用会算错量纲。这一章的结论直接解释了第 3 章埋下的问题：generation 阶段 $T=1$，等价于 $B$ 极小，因而注定 memory-bound；而幻灯片底部已经露出了下一段的函数名 `def arithmetic_intensity_of_inference():`，说明这组记账马上要被推广到整个 Transformer。18:45 讲师以 "let me stop there in case there's any questions" 暂停，进入现场问答。

![H100 加速器强度 295.2239、compute-bound iff B > 295 与 B=1 极端情形](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0014_00-18-02.jpg)

## 现场问答对 GQA 记号的修正，与最朴素的推理（Naive Sampling） [18:45 - 21:22]

18:50 之后是一段现场问答，画面切到讲师在讲台前的实拍（白板、笔记本、Stanford 台标），没有新的幻灯片文字，但内容对读图很关键。学生先就 "transpose slide" 提问，问转置是不是应该 across tables 而不是 across heads，期间还夹杂了麦克风与连线的插曲；随后问题落到 GQA 的记号上。转写里的对话大意是：学生确认 "K is the number of key value heads"，接着自我更正 "Sorry, G. Sorry, G per KV head"，并说明 "each key head will have a few queries"、"when you compute attention ... G is the number of query heads per key. That means we have two tables. That's my point"。讲师接受了这个意见并明确表示要改片子："I think you're right. So, K should be the number of uh groups and G should be the number of uh the you know, heads per one of those groups ... I'll fix that later." 因此本章给第 5 章的符号表打一个补丁：$K$ 应按「组数」理解，$G$ 应按「每组内的 query heads 数」理解，二者仍满足

$$N = K\cdot G,\qquad G = N\,/\!/\,K\ (\text{幻灯片写作 } N // K).$$

回顾第 5 章表格原文是 "K = number of key/value heads"、"G = q heads per kv head = N // K"；当「每组恰有一个 KV head」时两种口径一致，但讲师既然承认要修片，读图时就应以「K 为组数、G 为每组 query 头数」为准，而不是死抠字面把 $K$ 单纯数成 KV 头数。这是本段一个很典型的易错点：讲义的静态表格与口头修正并存，若不加说明，后面推导 KV cache 大小时会把 $K$ 与 $G$ 放错位置。

20:29 讲师转入 "let's talk a little bit more about the arithmetic intensity of inference"，并说先「naively 地做一遍」。流程是：给一个 prompt（"Never going to give you"），送进 Transformer，模型生成 keys、values 和 activations，最终产出覆盖输出词表 $V$ 的 logits，然后 sample a token（采样一个 token）；这个 token 被 concatenate（拼接）回 prompt，再整套重来。转写最后停在 "If you have a black box that takes a sequence and outputs a distribution of a..." 处被截断，本段到此结束，因此不能替它补上没念完的下文。

画面上的 `Naive Sampling` 手绘流程图把这个过程画成三帧，弧线箭头上写着 `concatenate`，比转写更直观：第一帧输入绿色手写 "never gonna give you"，经 `Transformer` 得到一排绿蓝相间的 `logits` 柱状图，`Sample token` 得到蓝色的 "up,"；第二帧输入变成 "never gonna give you up,"，同样过 `Transformer` 与 logits，采样出 "never"；第三帧输入 "never gonna give you up, never"，采样出 "gonna"，弧线继续向右延伸。图下方两行说明是本章最重要的增量信息，转写里完全没有念到：

```text
Naive inference: to generate each token, feed history into Transformer
Complexity: generating T tokens requires O(T^3) FLOPs (one feedforward pass is O(T^2))
```

页首还残留着 `def arithmetic_intensity_of_inference():` 与 `[Scaling book chapter on inference]` 两个标题，正好接上第 6 章末尾露出的函数名。复杂度可以这样理解：生成第 $t$ 个 token 时要对长度约 $t$ 的前缀做一次前向，attention 的代价是 $O(t^2)$，于是生成 $T$ 个 token 的总代价是 $\sum_{t=1}^{T}O(t^2)=O(T^3)$——这正是幻灯片给出的结论。用带语言标注的代码把黑盒写法补齐就是：

```python
def naive_sample(model, prompt, T_out):
    context = prompt                 # 例如 "never gonna give you"
    for _ in range(T_out):
        logits = model(context)      # 把整段历史重新喂进去
        token = sample(logits[-1])   # 在词表 V 上的分布里采样
        context = concat(context, token)
    return context
```

易错点与局限要说清楚。第一，朴素做法在语义上是正确的，问题纯在效率：每一步都把已有历史整段重算，前缀的 keys / values 被反复生成，因此 $O(T^3)$ 这个量级是「重复计算」的直接后果，而不是 attention 本身的固有下界。第二，$O(T^2)$ / $O(T^3)$ 的估计只数了随序列长度增长的 attention 二次项，把 MLP、词表投影等当作常数因子略去，属于量级分析而非精确 FLOPs 计数，不能拿去和第 6 章的 `2*B*D*F` 直接相加。第三，幻灯片中采样出的 "up,"、"never"、"gonna" 是顺着 "never gonna give you up" 这句歌词往下续写的具体例子，属于演示用的字符串，不是模型必然输出；转写里口头说的采样词与画面顺序也略有出入，按「每步采样一个 token 再拼回 prompt」这一机制理解即可。第四，本段在 21:18 处被截断，讲师尚未给出任何优化方案，因此这里只把朴素采样作为 baseline（基线）立住；它正是 `arithmetic_intensity_of_inference()` 接下来要拆解的对象，也与第 6 章 $B=1$ 的 memory-bound 结论合流——每步只新增一个 token，恰恰就是那种「很瘦的矩阵乘」。

![Naive Sampling 手绘流程：三步采样、concatenate 回拼与 O(T^3) FLOPs 复杂度](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0016_00-20-42.jpg)

## 朴素自回归推理的 O(T³) 代价与前缀共享观察 [21:22 - 22:45]

本段起点承接上一节关于算术强度的讨论，把视角正式转到「一次完整的前向传播究竟要被重复多少次」。最朴素的做法被讲者称为 Naive sampling（朴素采样）：把 prompt 喂进 Transformer，得到 logits 上的概率分布，采样出一个 token，再把这个 token 拼回历史，重新把整段序列喂进 Transformer，如此循环。幻灯片上用 "never gonna give you" 这句歌词演示：第一次生成 "up,"，第二次把 "never gonna give you up," 作为输入生成 "never"，第三次再生成 "gonna"，每一步顶上都标注了 concatenate（拼接），Transformer 方框下方是 logits 直方图与 "Sample token" 的箭头。图下那句定义写得很直白：`Naive inference: to generate each token, feed history into Transformer`。直觉上这没有任何错误——Transformer 本来就是对整个序列做一次前向传播就能给出各个位置的 logits——问题在于自回归要求「每一步都重算整段历史」，于是历史越滚越长，成本迅速膨胀，而且这种膨胀与硬件快慢无关。

讲者给出的复杂度结论是黄色高亮的一行：`Complexity: generating T tokens requires O(T^3) FLOPs (one feedforward pass is O(T^2))`。拆开看：单次前向传播里 attention 对长度 T 的序列要算 $T\times T$ 的注意力矩阵，代价是 $O(T^2)$；而为了生成 T 个 token，这个 $O(T^2)$ 的前向传播要重复 T 次，累计就是 $O(T^3)$。转写里说得更口语化："each time you generate one token you actually take order T squared time ... So generating T tokens is actually T cubed"，因为 "the attention is already T squared and you have to do that once for every token"。用记号写就是

$$\text{FLOPs}_{\text{naive}}=\sum_{t=1}^{T}O(t^2)=O(T^3).$$

这个立方增长是纯算法层面的浪费：即便把单次矩阵乘优化到极限，重复 T 次前向传播这件事本身仍会把算力吃光，因此优化方向不能只盯着算子，必须改掉「重算历史」这个结构。

转折出现在 22:04 之后："the observation is that you don't actually have to do this. A lot of the work can actually be shared across prefixes." 讲者用两条候选前缀做对比——生成 "never" 与生成 "up" 时，"never gonna give" 这些较早 token 对应的 key/value 其实被算了两遍，而 "these tokens, they shouldn't change"。根本原因在于这是 causal（因果）Transformer：若是 bidirectional（双向）模型，追加一个 token 会让所有位置的表示都变，"if you attach a token, then everything changes"；但因果掩码保证「前面的激活不依赖后面追加的任何 token」，即 "the activations here don't change based on any tokens you append"。于是同一前缀在不同采样分支之间、以及同一序列的相邻生成步之间，都天然可以共享计算，这也意味着 $O(T^3)$ 里绝大部分是重复劳动。

幻灯片底部顺势写下两条：`Observation: a lot of the work can be shared across prefixes` 与 `Solution: store KV cache in HBM`，并手写补了一个小标题 "Sampling with KV cache"。这里容易犯的错是把这份「不变性」当成普遍性质：它严格依赖因果注意力与「只追加、不回改」的自回归解码，一旦换成双向编码、或对已有 token 做编辑与重排，前缀激活就不再可复用；另外「共享」目前还只是观察，要真正省下计算，必须把不变的 key/value 落成显存里的常驻对象，这正是下一章 KV cache 要解决的事。本段末尾由此自然从「为什么慢」过渡到「怎么把重复计算存下来」。

![朴素采样流程图：每生成一个 token 都把整段历史重新喂入 Transformer，并标注 O(T³) 复杂度](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0017_00-22-02.jpg)

## KV Cache：把前缀计算存进 HBM 与推理两阶段 [22:45 - 25:22]

方案在 22:45 落地："the first obvious thing is to store a KV cache in HBM so that between successive token generations, you can just reuse the KV cache." 幻灯片给出的形式化定义是 `KV cache: for every sequence (B), token (S), layer (L), head (K), store an H-dimensional vector`；转写在 24:19 又念了一遍："for every sequence, so there's B of them, for every token there's S of them and for every layer and every head, you store H dimensional vector"。也就是对每个序列、每个已处理 token、每一层、每个 KV head，各存一个 H 维的 key 向量和一个 H 维的 value 向量。这套记号（B 序列数、S 输入/条件 token 数、L 层数、K KV head 数、H head 维度）与第 1 段的维度约定完全一致，区别在于 KV cache 是推理期新增的「有状态」对象：训练时一次看全序列、不需要它，推理时它反而成了决定显存与带宽的主角。写成尺寸公式（后面的代码会原样用到）：

```python
# KV cache: for every sequence (B), token (S), layer (L), head (K), store an H-dimensional vector
kv_cache_size_per_seq = S * (K*H) * L * 2 * 2   # 2 for key + value, 2 for bf16
```

有了缓存，推理被明确拆成两个阶段，图上写 `Two stages of inference: 1. Prefill: given a prompt, encode into vectors (parallelizable like in training); 2. Generation: generate new response tokens (sequential)`。流程图用红框圈出 Prefill：把 "never gonna give you" 一次性送进 Transformer，同时产出两样东西——「KV cache initialised」的绿白条与首个 logit 的蓝色直方图；红框下方手写标注 "Compute bound"。旁边两个蓝框都是 Generate：绿色条是已填充的 KV cache，蓝色直方图是上一步采样出的 token（"up,"、"never"），送入 Transformer 后一方面给出下一个 logits，另一方面产出 "new KVs" 追加到缓存末尾；蓝框下方手写 "memory bandwidth bound"。这两处手写标注其实提前剧透了 25:22 之后整段要花十分钟量化证明的结论：两个阶段落在完全不同的瓶颈侧。

转写从 23:13 起把 generation 的逐步流程念得很细：prefill 先 "populate the KV cache, which is the set of key and value pairs that the transformer computes"，然后 "you generate the logits ... sample at a token ... feed that through the transformer, which uses this cache ... produces both the distribution of the next tokens, but a new KV, which is basically activations corresponding to the token up"；这个新 KV "gets fed in"，于是 "you have the KV cache, you have these logits you sample and then that feeds through the transformer, you output the distribution of the next tokens as well as the new token, which you add to the KV cache and so on and so forth"。关键收益是：生成第 t 步只需为 1 个新 token 计算 K/V，并复用前 $t-1$ 个 token 的缓存，不必再重算整段历史，把朴素方法的 $O(T^3)$ 降到每步只与缓存长度 S 成正比的总量级。

讲者特意重申 prefill "is parallelizable just like in training because you see the entire prompt, you can compute the KV cache"，而 generation "you generate the new response tokens sequentially. But, at least you don't have to pay for generating the KV cache of the tokens that you already looked at"。易错点有两处：其一，别把 KV cache 当成「免费」——它常驻 HBM、随 B 与 S 线性增长，第 36 分钟会看到它本身变成内存瓶颈；其二，别误以为 prefill 与 generation 的瓶颈性质相同，红蓝框与手写标注已经提示二者分别落在 compute-bound（计算受限）与 memory-bound（内存受限）两侧。讲者在 25:08 以 "is everyone comfortable with the KV cache?" 收住这一节，紧接着就要用 FLOPs 与 bytes 把这两个词算成具体数字。

从记号上看还要分清两个容易混淆的 S：第 1 段里 T 曾被用来泛指序列长度，而本段之后 S 固定指「已经 conditioning 的、缓存在里的 token 数」，T 指「本次前向要产出 logits 的 token 数」，prefill 时二者相等、generation 时 $T=1$ 而 S 随解码步单调增长。KV cache 里存的是每一层、每个 KV head 的 key 与 value，而不是原始 token 或隐藏状态；因此缓存条目数与层数 L、KV head 数 K 成正比，而不是与 query head 数 N 或模型维度 D 直接挂钩。讲者反复强调「reuse the KV cache」，其准确含义是：新一步的 query 只与缓存中全部 S 个 key 做点积、再对 S 个 value 加权求和，缓存本身只读不改、只在末尾追加一格，这既解释了为什么 causal 性质是前提，也预告了 attention 的计算量会随 S 线性累积。

![KV cache 两阶段流程图：红框 Prefill 标注 Compute bound、蓝框 Generate 标注 memory bandwidth bound](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0018_00-23-22.jpg)

## MLP 层的 FLOPs、字节与算术强度 B·T [25:22 - 29:18]

25:22 讲者宣布 "let's try to figure out the flops and memory IO for both the MLP and attention layers"，并先约定只统计 matmul："I'm just going to look at the matmuls because the matmuls are the thing that actually require a lot of work. Everything else ... is not that many flops, and can be potentially fused into the matmul, too." 幻灯片黄色高亮写着 `MLP layers (only looking at the matrix multiplications)`，代码从 `flops = 0`、`bytes_transferred = 0`、`# Perform the operation` 起手。变量约定也在这页重申：`S is the number of tokens we're conditioning on, T is the number of tokens we're generating`，并且 `Later, we'll specialize to prefill (T = S) and generation (T = 1)`；参与的维度是 B（batch）、T（本次要生成 logits 的 token 数）、D（模型维度）、F（feed-forward / MLP 维度）。这一步的方法论比结论更重要：它把「快慢」化成两个可数的量——浮点运算次数与 HBM 与 SRAM 之间的字节搬运量——二者相除就是上一节定义的 arithmetic intensity（算术强度），从而让「compute-bound 还是 memory-bound」变成可计算的不等式。

幻灯片把 MLP 的八步拆开列出（可见的第 3–8 步原文如下），每一步都同时登记 flops 与 bytes_transferred：

```python
# 3. Compute U = X (B x T x D) @ Wup (D x F)
flops += 2*B*T*D*F
# 4. Write U (B x T x F) to HBM
bytes_transferred += 2*B*T*F
# 5. Compute G = X (B x T x D) @ Wgate (D x F)
flops += 2*B*T*D*F
# 6. Write G (B x T x F) to HBM
bytes_transferred += 2*B*T*F
# 7. Compute Y = GeLU(G)*U (B x T x F) @ Wdown (F x D)
flops += 2*B*T*D*F
# 8. Write Y (B x T x D) to HBM
bytes_transferred += 2*B*T*D
```

加上前面从 HBM 读 X 与读三个权重矩阵的开销，两条断言把总量钉死：`assert flops == 6*B*T*D*F` 与 `assert bytes_transferred == 4*B*T*D + 4*B*T*F + 6*D*F`。其中 $6BTDF$ 来自 up、gate、down 三个投影各 $2BTDF$；$6DF$ 是 $W_{up},W_{gate},W_{down}$ 三份 $D\times F$ 权重按 BF16（每元素 2 字节）读取。于是算术强度为

$$\text{intensity}_{\text{MLP}}=\frac{6BTDF}{4BTD+4BTF+6DF}=\frac{3BDFT}{2BDT+2BFT+3DF},$$

幻灯片蓝色框里正是 `intensity = 3*B*D*F*T/(2*B*D*T + 2*B*F*T + 3*D*F)`。转写里讲者没有逐步念每个矩阵，只说「the form is you read X from high bandwidth memory, you read all the parameters, you compute the upward projection ... the gate ... the down projection」，但这些断言与代码是画面独有、必须补进正文的硬信息。

接着用 SymPy 做极限化简，把「$BT$ 远小于 $D,F$」这个假设写进代码：

```python
# Compute the arithmetic intensity
intensity = (flops / bytes_transferred).simplify()
# Assume that B*T is much smaller than D and F.
intensity = intensity.subs(D, c*B*T).subs(F, c*B*T).limit(c, oo).simplify()
assert intensity == B*T
```

也就是 $\text{intensity}_{\text{MLP}}\approx BT$，与第 1 段纯矩阵乘得到的 $B$ 完全同构。讲者的解释是 "An MLP is basically a big matmul"，因为 "the batch dimension and the sequence length dimension are sort of everything is independent"，两者可以合并成一个大的「有效 batch」；蓝色浮层里也直接给出 `intensity = B*T`。落到两阶段，幻灯片写 `1. Prefill: easy to make compute-bound (good) by making B*T large enough (large batches, long sequences)`：prefill 时 $T=S$，强度约 $BS$，只要把 batch 或序列拉大，就能越过 H100 约 295 的门槛。转写补充 "if you make B*T large enough, large batches, long sequences, you'll be fine"。

但 `2. Generation: two problems` 随即列出两个麻烦：其一 `Generating one token at a time (T = 1)`，强度塌缩为 $B$；其二 `B is number of concurrent requests, unpredictable for interactive applications`（黄色高亮）。转写解释，聊天服务里的 B 就是并发用户数，"can be high, can be low, so it's a bit unpredictable, it can be changing over time"，因此要靠后面才讲的 continuous batching（连续批处理）来稳住。这里的易错点是把 MLP 的 $B$ 与训练里的 batch size 混为一谈：generation 阶段它不再是「一次塞进多少条样本」，而是「同一时刻有多少个独立请求」，是一个由线上流量决定的外生量，运维无法随意调大。幻灯片末尾还留了下一章的引子：`Attention layers (focusing on the matrix multiplications with FlashAttention)`，提示 attention 要按 FlashAttention 的口径重算，不能照搬 MLP 的乐观结论。

![MLP 层 FLOPs 与字节传输逐步代码，含 6*B*T*D*F 与 4*B*T*D+4*B*T*F+6*D*F 两条断言](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0021_00-27-23.jpg)

## Attention 的算术强度 ST/(S+T)：prefill 尚可、generation 根本受限 [29:18 - 34:04]

转写在 29:22 重新对齐记号：S 是 "previous tokens already generated"，T 是 "the number of tokens you want to generate logits for"。计算步骤在图上写为：从 HBM 读 `Q (B x T x D), K (B x S x D), V (B x S x D)`，算 $A=QK^\top$ 记 `flops += 2*B*S*T*D`，再 `Y = softmax(A) @ V` 又一次 `flops += 2*B*S*T*D`（讲者顺口说 "compute the softmax, which doesn't really matter"），最后把 `Y (B x T x D)` 写回 HBM。两条断言是 `assert flops == 4*B*S*T*D` 与 `assert bytes_transferred == 4*B*S*D + 4*B*T*D`。注意字节项里没有权重矩阵——attention 除了 QKV 投影外没有可共享的大权重，读进来的 K、V 就是每个序列自己的 KV cache。于是

$$\text{intensity}_{\text{attn}}=\frac{4BSTD}{4BSD+4BTD}=\frac{ST}{S+T},$$

代码里对应 `assert intensity == S*T / (S + T)`，讲者口述为 "that factor for attention is S * T over S + T"。他还点明一个通用规律：既然全是 matmul，FLOPs 总比 bytes "one degree higher of a polynomial"，唯一要问的就是这个因子长什么样；对 MLP 那个因子是 $BT$，对 attention 则是 $\frac{ST}{S+T}$，两者的差别正是全段的核心。

把两阶段代入，幻灯片上的代码是：

```python
# For the two stages:
# 1. Prefill: T = S
prefill_intensity = intensity.subs(T, S).simplify()
assert prefill_intensity == S/2    # Good!
# 2. Generation: T = 1
generate_intensity = intensity.subs(T, 1).simplify()
assert generate_intensity < 1      # Bad!
```

蓝色浮层同时列出三个式子：`intensity = S*T/(S + T)`、`prefill_intensity = S/2`、`generate_intensity = S/(S + 1)`。prefill 时 $T=S$ 得 $S/2$，讲者评价 "as long as you have long sequence lengths, for attention, you're going to maintain high arithmetic intensity"，并特意提醒 "notice that the batching dimension doesn't happen here"——$S/2$ 里没有 B。generation 时 $T=1$ 得 $\frac{S}{S+1}<1$，"Or even let's just call it one. And arithmetic intensity, remember one is bad. We want it to be something like 295 ... to saturate the compute." 也就是说生成阶段的 attention 强度连 1 都不到，离 H100 的约 295 差了两个数量级以上，"this is really the bottleneck"；讲者在 31:29 明确把前面 MLP/attention、prefill/generation 的四格分析收束到这一格上。

幻灯片随后给出直白的一句 `Unlike MLPs, no dependence on B, so batching doesn't help!`，并自问 `Why?`：`In MLP layers, every sequence hits the same MLP weights (Wup, Wgate, Wdown don't depend on B)`；`In attention layers, every sequence has its own KV cache vectors (Q, K, V all depend on B)`。转写把直觉讲得更透：MLP 里 "you get to load these MLP weights once ... and you can use it for all your sequences"，所以 B 越大越能摊薄权重加载；而 attention 里 "all these depend on B. So increasing B doesn't help. It's like for every sequence you're basically doing a matmul. So they're all independent, so doing more matmuls isn't helpful"。讲者还回扣课程最开头的例子：这种按坐标 batching 的运算 "is essentially basically the same as doing a dot product, which has horrible arithmetic intensity"，并指出 "this B blue B is like the cause of why the attention arithmetic intensity doesn't scale with B"——这里的 blue B 指 einsum 图里的 batching 维（出现在两个操作数里但保留、不参与收缩），它不会像 contracting 维那样带来高算术强度。

易错点有三层。第一，不要把「MLP generation 只要 B 大就没事」的结论平移到 attention，两者的 batch 维性质根本不同：前者权重跨序列共享，后者 KV cache 逐序列独立。第二，不要以为 FlashAttention 能改变这个上界，它优化的是常数与显存访问模式，$\frac{ST}{S+T}$ 在 $T=1$ 时回落成 $\frac{S}{S+1}$ 的结构比不会变。第三，讲者在 34:33 把话说死："If you're sticking with a transformer, you can't really improve this"——这不是工程实现不够好，而是自回归 attention 的结构性代价，因此后面所有「从架构上缩小 KV cache」或改变注意力形式的优化线索，都是冲着这一格 $<1$ 的强度来的；只要还保持标准 Transformer 的逐 token 注意力，这个上界就绕不过去。

![Attention 两阶段强度代码：assert prefill_intensity == S/2 与 assert generate_intensity < 1](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0024_00-31-23.jpg)

## 结论：Prefill 计算受限、Generation 内存受限 [34:04 - 35:44]

34:04 讲者把四格结论一次摆出，幻灯片 `Summary:` 下的 bullet 是：`Prefill is compute-bound, generation is memory-bound`；`Prefill MLP intensity: B*S`；`Prefill attention intensity: S/2`；`Generation MLP intensity: B (requires concurrent requests)`；`Generation attention intensity: <1 (impossible to improve)`。转写的口述版本略有取舍：prefill MLP "B * S, great"，prefill attention "S over 2, not as good, but workable"，generation MLP "also workable, requires long concurrent requests"，而 "it is really the generation intense attention intensity, which is a fundamental bottleneck"。把四格与 H100 的 295 门槛并排看，只有 prefill MLP 的 $BS$ 与 prefill attention 的 $S/2$ 有条件超过门槛，两个 generation 项都远低于它，其中 attention 的 $<1$ 更是被判为「只要还用 Transformer 就无解」。这也解释了讲者 34:57 那句带点得意的收尾："whenever you hear people say, 'Oh, inference is memory bound,' you know why"——这句流行说法不是经验之谈，而是 $\frac{ST}{S+T}$ 在 $T=1$ 时必然塌缩的直接后果。同一页还残留着蓝色浮层的三个式子 $\frac{S*T}{S+T}$、$\frac{S}{2}$、$\frac{S}{S+1}$，与上一章的代码一一对应，方便对照记忆。

接着 35:07 讲者转向三个指标——"inference metrics, throughput, and latency, and also TTFT"——并指出内存受限反而在建模上让事情变简单了："when we think about how long things take, you just look at how much memory needs to be transferred"，因为 "assuming you overlap communication and computation, the bottleneck is going to be just the amount of memory that you have to deal with"。写成式子就是一步延迟约等于

$$\text{latency}\approx\frac{\text{bytes to transfer}}{\text{memory bandwidth}},$$

不再需要逐层累加 FLOPs 与算力峰值。代价则是讲者带着无奈点破的另一面："in other ways it's frustrating that your accelerators are sitting there not doing anything"——昂贵的 Tensor Core 空转，时间全花在 HBM 与 SRAM 之间搬参数与 KV cache 上。这里与第 1 段的算术强度框架形成闭环：前面用 FLOPs/bytes 判断瓶颈归属，现在既然判定为 memory-bound，就可以把整个性能模型降维成「搬多少字节」，这正是下一节敢用一个 config 直接算秒数的底气。

这一结论也划清了后续优化的着力点：既然生成阶段被内存流量卡住，那么改善延迟与吞吐的杠杆就只有两类——减少需要搬运的字节（缩小权重、缩小 KV cache、量化），或提高有效带宽利用率与并发（continuous batching 等调度手段）。幻灯片底部此时已经露出下一节的代码骨架 `@dataclass(frozen=True)` 与 `class TransformerPerformanceStats:`，说明课程马上要把这些符号化的强度结论代入一个真实模型，算出具体的秒数与 tokens/s，而不是停留在 $B$、$S/2$ 这类抽象式子上。

需要留意的局限是：整套推导建立在「通信与计算可完美重叠、忽略开销」的理想假设上，且只统计 matmul；真实系统里 kernel launch、softmax、采样、跨卡通信都会额外吃时间，所以下一节算出的应被理解为理论上界而非实测值。这正是 35:52 那页黄色高亮 `Assumption: can overlap compute and communication perfectly and ignore overhead.` 要提前声明的原因，也是把「inference is memory bound」从定性判断转成定量公式时必须自带的免责条件。

把四格结论与更早的算术强度定义对照，还能看出一条隐含的判据：判断某阶段是否 compute-bound，本质是拿它的 $\text{intensity}$ 与加速器强度 295 比大小，$BS$ 与 $S/2$ 只在 B 或 S 足够大时才越线，$B$ 与 $<1$ 则几乎永远越不过去。这也解释了为什么课程把「prefill 好办、generation 难办」反复强调：prefill 的两个杠杆（拉长序列、加大 batch）都是工程师可以主动调的，而 generation 的 MLP 杠杆 B 由并发用户数决定、attention 杠杆干脆不存在。换用更精确的措辞，所谓「inference is memory bound」应当理解为「generation 阶段的 attention 是 memory bound」，prefill 阶段恰恰是 compute-bound，把整句话笼统套到推理全流程上是不准确的。

![Summary 总结幻灯片：Prefill compute-bound、Generation memory-bound 及四格强度公式](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0026_00-34-21.jpg)

## 实例：Llama 2 13B 在 H100 上的内存、延迟与吞吐公式 [35:44 - 39:26]

36:03 讲者抛出具体问题："for Llama 2 13B, on a H100. So, what is the latency and the throughput?" 幻灯片右侧的 `config` 表逐项给出：`S: 1024`、`D: 5120`、`F: 13824`、`N: 40`、`K: 40`、`H: 128`、`L: 40`、`V: 32000`、`memory_bandwidth: 3.350e+12`。转写逐项念过："this is a sequence length, the model dimension, feedforward dimension, number of query, number of key value heads. There's no GQA here, so n equals k, head dimension, number of layers, vocab size, and the memory bandwidth for H100"，并特意强调 $K=N=40$ 意味着这个模型没有用 GQA（分组查询注意力），KV head 与 query head 一样多，这会让后面的 KV cache 偏大。代码里还带注释，例如 `K: 40, # Number of key/value heads`、`H: 128, # Head dimension`、`L: 40, # Number of layers`、`V: 32000 # Vocabulary size`、`memory_bandwidth: 3.35e12, # Memory bandwidth (bytes/second)`。同页写明前提：`So we have shown that inference is memory-bound.`、`Let us now compute the theoretical maximum latency and throughput of a single request.`，以及黄色高亮的 `Assumption: can overlap compute and communication perfectly and ignore overhead.`。入口函数是 `config = llama2_13b_config()` 与 `stats = compute_transformer_performance_stats(config)`，要返回的统计量是 number of parameters、memory usage、latency、throughput 四项。

`compute_transformer_performance_stats` 里先把「什么东西占内存」逐项拆开（转写："the embeddings, the MLP layers, the projections of the KQV"）：

```python
# Number of parameters in the Transformer
num_params = 2*V*D + D*F*3*L + (2*D*N*H + 2*D*K*H)*L
# How much memory the parameters take
parameter_size = 2*num_params   # 2 for bf16 (training requires a larger multiple)
# How much the KV cache takes per sequence (S tokens, K heads, H head dim, L layers)
kv_cache_size_per_seq = S * (K*H) * L * 2 * 2   # 2 for key + value, 2 for bf16
```

$2VD$ 是输入/输出 embedding（这里按 2 份计），$3DFL$ 是每层 up/gate/down 三个 $D\times F$ 矩阵，$(2DNH+2DKH)L$ 是每层 Q 投影与 K/V 投影。讲者说明 "assuming that we're doing a BF16, which always the case for inference, then parameters take up this many bytes"，故乘 2；注释还顺带提醒训练需要更大的倍数（优化器状态等）。KV cache 那条连乘里 $S\cdot(KH)\cdot L$ 之后的第一个 2 是 key 与 value 各一份、第二个 2 是 BF16 的 2 字节，与第 2 章「每序列每 token 每层每头存 H 维向量」的定义完全对齐；转写也逐项念过："the number of tokens in your sequence times your heads, KV heads times the head dimension times the number of layers, you have one for the key and one for the value, and then you have two multiple two for BF16"。

把两者合起来（转写："for each sequence you have B sequences so that's B times that plus the parameter size"）：

```python
# Total memory usage
memory = B * kv_cache_size_per_seq + parameter_size
# *Latency* is determined by memory IO (read all parameters and KV cache for each step)
latency = memory / memory_bandwidth
# *Throughput* is the inverse of latency, but we're generating B tokens in parallel
throughput = B / latency
# Substitute config
num_params = num_params.subs(config).simplify()
return TransformerPerformanceStats(num_params, memory, latency, throughput)
```

即

$$\text{memory}=B\cdot\underbrace{S(KH)L\cdot2\cdot2}_{\text{KV cache / seq}}+2\cdot\text{num\_params},\qquad \text{latency}=\frac{\text{memory}}{\text{BW}},\qquad \text{throughput}=\frac{B}{\text{latency}}.$$

讲者特别澄清单位：latency 是 "seconds per token"，throughput 是 "tokens per second as well as having an additional B because you're processing a batch of B"，并解释延迟的本质是 "shuffling parameters back and forth between HBM and SRAM"。结构上 memory 是 B 的线性函数（KV cache 随 B 长、参数是常数项），latency 因此也随 B 线性增长，而 throughput 形如 $B/(aB+b)$，会渐近饱和——这三个式子把前一章「fast 有两种含义」的定性说法变成了可求导的代数。

易错点集中在单位与口径：BW 必须用 `3.350e+12` 字节/秒，不能拿 3.35 TB/s 的十进制直觉去估；参数项写成 `2*num_params` 只含权重、不含激活与临时缓冲，是刻意的理想化，真实部署还要留显存给这些开销；另外 `num_params.subs(config).simplify()` 只是符号代入，把 $S,D,F,N,K,H,L,V$ 全部换成整数后得到的才是可与 13B 标称值对照的具体数。下一章就要把 config 真正代进 `memory`、`latency`、`throughput` 三行，看这些符号式落成具体数字与 batch size 的权衡。

这套 config 的数值彼此是自洽的，值得逐项对一遍，避免把符号当成抽象字母放过：$D=5120$、$N=40$、$H=128$ 满足 $D=N\cdot H$（$40\times128=5120$），说明 head 维度与 query head 数正好拼回模型维度；$F=13824$ 是 MLP 的 up-projection 维度，$L=40$ 是层数，$V=32000$ 是词表大小，$S=1024$ 是本例假定的序列长度；`memory_bandwidth: 3.350e+12` 的单位在代码注释里写明是 bytes/second，即每秒可搬约 3.35 万亿字节。讲者特别点出 $K=40$ 与 $N=40$ 相等、"There's no GQA here, so n equals k"，意味着每个 query head 都配一份独立的 K/V，KV cache 因此按较大的 K 计算，这也是后面 838,860,800 这个每序列字节数的来源之一——若换成 GQA 减少 KV head 数，这一项会成比例缩小，正是后续要展开的优化方向。

![Llama 2 13B 的 config 参数表与 num_params、parameter_size、kv_cache_size_per_seq 计算代码](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0028_00-37-24.jpg)

![总内存、延迟与吞吐量公式代码：memory、latency = memory / memory_bandwidth、throughput = B / latency](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0029_00-38-55.jpg)

## 代入数值与 batch size 的延迟/吞吐权衡 [39:26 - 42:44]

`throughput.subs(config).simplify()` 之后，幻灯片给出四个化简结果：

```python
num_params = 13,015,449,600
memory     = 838860800*B + 26030899200
latency    = 0.000250406208955224*B + 0.00777041767164179
throughput = 127792.35838438*B/(32*B + 993)
```

讲者对第一项做了 sanity check："num parameters is 13 billion so good sanity check that is advertised as a 13 billion parameter model"；对第二项说 "about 838 I guess million times B so it's basically a linear function of B plus some other term ... this is a KV cache which grows as B, this is the parameters which is double the number of params"（$2\times13{,}015{,}449{,}600=26{,}030{,}899{,}200$ 字节即 26.03 GB）。latency 因 "just this scaled by the memory bandwidth so it has the same form as the memory"，throughput 则是 "B over that"。注意 latency 的常数项 $0.00777$ 秒来自固定要读一遍的 26 GB 参数，斜率 $0.00025$ 秒来自每多一个并发请求就要多搬 838,860,800 字节的 KV cache；throughput 的分母 $32B+993$ 与分子 $127792.36B$ 则把这两个来源压成了一个有理函数。

代码用 `b1 = stats.substitute(B, 1)`、`b64 = stats.substitute(B, 64)`、`b256 = stats.substitute(B, 256)` 实例化，并写下 `Result: worse latency, better throughput`。b1 的结果框是 `num_params: 13,015,449,600`、`memory: 26,869,760,000`、`latency: 0.0080`、`throughput: 124.6755`。转写念到 "if you have batch size one ... you get a latency of 0.008 seconds per token ... the throughput is 124 tokens per second"。$26{,}869{,}760{,}000$ 字节约 26.9 GB，恰好是参数 26.03 GB 加单序列 KV cache 0.839 GB，除以 $3.35\times10^{12}$ 得约 0.00802 秒，与幻灯片一致——这也是对整条推导链最实在的一次校验。

讲者随即点出反直觉之处："you increase the batch size, you'll see that the latency goes up but the throughput also goes up. So, latency gets worse, but the throughput improves." 机制他说得很清楚：latency 变差是因为 "the KV cache grows and in order to process stuff, you have to copy the KV cache back and forth"；throughput 变好是因为 "you're amortizing the cost over a larger batch"，但 "throughput can't possibly go to infinity"，它会 "asymptote"。于是 41:14 把结论压成一句可记的形式："latency is a linear function in B. This is where the constant is the size of KV cache and then this is the number of parameters and then throughput is some B proportional to B over B plus something"，即

$$\text{latency}=aB+b,\qquad \text{throughput}\propto\frac{B}{B+c}.$$

42:12 他把这个数学事实翻译成工程判断："we think about oh, we just want to make it go fast, but fast actually has two meanings here, which actually depending on which one you care about is completely opposite"——交互式场景要低延迟就压小 batch，批量处理文档要高吞吐就放大 batch，"if you want to tune your batch size, that's going to really determine if you want latency or throughput"。本段在 42:44 以 "let's increase the batch size even more ... the latency gets even worse" 收尾，把矛盾留在台面上，也顺势引出后续用架构手段缩小 KV cache 的动机。

必须记住的局限有两个。其一，这些数值建立在「通信与计算完美重叠、忽略开销」的理论上界上，实测 tokens/s 会更低；`def throughput_and_latency():` 里 "Let us now compute the theoretical maximum latency and throughput of a single request" 的 theoretical maximum 一词已经把话说在前面。其二，增大 B 并非无代价：KV cache 随 B 线性膨胀，很快会撞上 H100 的显存容量墙，因此 throughput 的渐近线在现实中往往还没到就被显存截断，`b256 = stats.substitute(B, 256)` 这一行正是用来暴露这个边界的。这也再次呼应第 4 章那个「根本瓶颈」——真正能同时改善延迟与吞吐的，不是调 batch，而是减小 KV cache 本身；调 batch 只是在两个互相冲突的「快」之间做取舍。

还可以把 `b1`、`b64`、`b256` 三行与前面的化简式对照着读：代码只写了 `stats.substitute(B, 1)`、`stats.substitute(B, 64)`、`stats.substitute(B, 256)`，画面里真正展开数值的是 b1 那个框，而 `Result: worse latency, better throughput` 是对增大 batch 之后方向的定性概括，并没有把三档的全部数字都列出来，因此这里不宜凭空补出 b64、b256 的具体延迟与吞吐。能确定的是方向：把 $B$ 从 1 提到 64 再提到 256，$\text{latency}=0.00025B+0.00777$ 的线性项不断累加，而 $\text{throughput}=\frac{127792.36B}{32B+993}$ 的分子分母同阶、比值向 $\frac{127792.36}{32}\approx3993$ tokens/s 渐近，永远不会发散。讲者用 "throughput does improve but up to a limit"、"this sort of asymptotes" 描述的正是这条水平渐近线；把「增大 batch 能提高吞吐」误解成「吞吐能无限增长」，是读这组公式时最常见的偏差。

![batch size 1 的实例化结果：num_params 13,015,449,600、memory 26,869,760,000、latency 0.0080、throughput 124.6755](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0031_00-41-59.jpg)

## 延迟与吞吐的张力、并行维度与 TTFT [42:44 - 45:47]

本段开头紧接上一段对 Llama 2 13B 在 H100 上的解析演算，把「不断增大 batch size」这条路走到尽头。屏幕上是一段完整的演算脚本，先用 `config = llama2_13b_config()` 生成配置，再用 `stats.substitute(B, ...)` 把三档 batch 代入符号表达式求值，最后用一条断言检查显存是否装得下：

```python
config = llama2_13b_config()
stats = compute_transformer_performance_stats(...)

# Batch size 1
b1 = stats.substitute(B, 1)
# Batch size 64
b64 = stats.substitute(B, 64)
# Result: worse latency, better throughput
# Batch size 256
b256 = stats.substitute(B, 256)
# Result: even worse latency, even better throughput

h100_memory = 80e9  # H100 memory in bytes
assert b256.memory > h100_memory  # Doesn't fit in memory!
# Result: doesn't fit into memory and throughput gains are diminishing too...
```

右侧青色结果框给出了精确到个位的数值：三档的 `num_params` 都固定为 $13{,}015{,}449{,}600$（约 13B 个参数，BF16 下占 $26{,}869{,}760{,}000$ 字节）；`memory` 随 batch 从 $26{,}869{,}760{,}000$ 涨到 $79{,}717{,}990{,}400$ 再到 $240{,}779{,}264{,}000$ 字节；`latency` 从 $0.0080$ 秒升到 $0.0238$ 秒再到 $0.0719$ 秒每 token；`throughput` 则从 $124.6755$ 升到 $2689.4807$ 再到 $3561.7685$ tokens/s。这三行数把「延迟变差、吞吐变好」的斜率量化了出来，也暴露出吞吐增益是递减的：从 $B=1$ 到 $B=64$ 吞吐涨了约 21 倍，而从 $B=64$ 到 $B=256$ 只涨了约 1.3 倍。

真正让这条路走不通的是显存容量。代码里紧接着写 `h100_memory = 80e9  # H100 memory in bytes`，下一行是 `assert b256.memory > h100_memory  # Doesn't fit in memory!`——$B=256$ 时约 240 GB 的 KV cache 加参数已远超 H100 的 80 GB HBM。幻灯片用黄色高亮把结论钉死：`Result: doesn't fit into memory and throughput gains are diminishing too...`。讲者口头补充说，换用 B200 这类更大显存的卡还能把 batch 再往上推一点，但「eventually you hit some sort of limit」，所以永远到不了吞吐的渐近线（asymptote），因为会先撞上内存墙（memory wall）；与此同时吞吐收益本身也在 diminishing。这里其实叠着两个彼此独立的限制：一个是硬件容量的硬上限，一个是参数共享带来的收益递减，二者共同决定了单靠调 batch 的天花板。这一点与前面 arithmetic intensity 的结论是连贯的——generation 阶段是 memory-bound，KV cache 随 $B$ 线性增长，内存流量迟早会压垮收益。

![增大 batch 到 256 时 KV cache 超出 H100 容量的代码与三档延迟吞吐数值表](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0032_00-43-24.jpg)

为什么增大 batch 会同时拉高延迟却改善吞吐？幻灯片上给出了两条精炼的 bullet：`Worsens latency because larger KV cache (O(B) size) to read/write` 与 `Improves throughput because amortizes the cost of reading parameters`。前者说的是每步解码都要把规模为 $O(B)$ 的 KV cache 读进来再写回去，batch 越大搬运的字节越多，单个 token 的等待就越久；后者说的是模型参数（weights）在所有序列之间共享，一次从 HBM 加载进 SRAM 之后可以服务整个 batch 的矩阵乘，把「读权重」这笔固定开销摊薄到更多序列上。讲者用等巴士（bus）作类比：等巴士时你得先等所有人到齐才发车，单个人的 latency 很高，但巴士一次能把所有人运走，throughput 很好；反过来小 batch 像打车，随到随走、延迟低，却运不了几个人。由这两条机制直接得到屏幕上 `Tradeoff between latency and throughput` 下的两行小结：`Smaller batch sizes yield better latency but worse throughput` 与 `Larger batch sizes yield better throughput but worse latency`。要特别注意，这个张力（tension）的根源是 batch 维度，而不是所有优化天生都对立——后面讲 GQA 时会明确把这一点翻转过来。

除了 batch 这条轴，讲者还快速点了一下并行（parallelism）这个「another dimension of inference」，并明说不打算深入，细节留给 scaling book 的 inference 章节。屏幕上区分了两种：黄色高亮上方一行 `Easy parallelism: if you launch M copies of the model, latency is the same, throughput increases by M!`，即启动 $M$ 份完整模型副本（数据并行式的多副本服务），每份各自处理自己的请求，单个请求的 latency 不变，而系统总吞吐线性乘以 $M$；高亮那行则是 `Harder parallelism: shard the model and the KV cache [Scaling book chapter on inference]`，即把模型权重和 KV cache 切分（shard）到多个 device 上，这涉及跨设备通信与切分策略，复杂度高得多。这里容易混淆的是：多副本只是把同样的内存瓶颈复制 $M$ 份，并没有缓解单请求的内存受限本质；而 sharding 是在改变单请求的内存布局，二者不是一回事。

![增大 batch size 的利弊、延迟吞吐权衡、M 副本并行与 TTFT 三条建议的幻灯片](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0033_00-44-45.jpg)

本段最后补上了此前一直没正式谈的第三个指标 TTFT（time-to-first-token，首 token 时间）。淡色注释里写着 `Note: time-to-first-token (TTFT) is essentially a function of prefill time`：因为 prefill 结束后才能开始解码、吐出第一个 token，所以 TTFT 本质上就是 prefill 阶段的耗时。由此得到两条直接可操作的建议：`Use smaller batch sizes during prefill for faster TTFT`（prefill 阶段用小 batch 以降低首 token 等待）与 `Use larger batch sizes during generation to improve throughput`（generation 阶段用大 batch 以提升总吞吐）。这与前面的权衡并不矛盾，反而说明同一个服务里 prefill 和 generation 两个阶段可以、也应该采用不同的 batch 策略。一个常见易错点是把 TTFT 与 latency 混为一谈：TTFT 衡量的是「从提交到第一个 token」的一次性等待，主要受 prompt 长度与 prefill 并行度影响；latency 衡量的是 token 之间流出的速度，主要受每步 KV cache 内存流量影响。讲者用一句 `Any questions about throughput and latency and how they're in tension with each other?` 收束这一节，随后正式转入「如何让推理更快」。

## 加速推理的技术全景：把 KV cache 当作主战场 [45:47 - 46:56]

到这里，课程已经建立起一整套分析推理效率的概念框架：arithmetic intensity（算术强度）判断一个操作是 compute-bound 还是 memory-bound，再用 throughput 与 latency 两个指标描述服务的快慢。讲者随即抛出新的问题：`How do we make inference faster?`，并强调可用的技术手段「quite varied」，跨度从改动模型架构（model architecture）一直到系统优化（systems optimization），中间还有各种折中，所以 inference 在某种意义上是「a fairly rich cross-cutting topic」——横切了模型、算子、运行时和硬件多个层面。这种「先看瓶颈、再选手段」的叙述顺序本身就是本段的组织线索：前面所有推演都指向同一个结论，generation 阶段的瓶颈是内存，而内存里增长最快、最可控的一块就是 KV cache。

屏幕上先出现一页 `Understanding the inference workload` 的回顾，用一组刻意写成函数调用形式的伪代码列出前几节走过的路径：`landscape()`、`review_transformer()`、`review_of_arithmetic_intensity()`、`arithmetic_intensity_of_inference()`、`throughput_and_latency()`。这种写法不只是排版趣味，它暗示后续内容会像调用函数一样，逐个「执行」新的优化模块。紧接着黄色高亮的大标题是 `Taking shortcuts (lossy)`——「走捷径（有损）」，下面淡色列出的三项正是随后要展开的函数名：`reduce_kv_cache_size()`、`quantization()`、`model_pruning()`；再往下是一句提纲式的 `Summary: reduce inference complexity without hurting accuracy`。这里的关键词是 lossy（有损）：这些方法都会以某种方式改变模型或它的数值表示，因此「不伤精度」是贯穿始终的约束，而不是可以事后忽略的附加条件。本段（42:44–64:07）几乎全部篇幅都花在第一项 `reduce_kv_cache_size()` 上，量化和剪枝留到后面的段落。

![Understanding the inference workload 回顾清单与 Taking shortcuts (lossy) 分类的幻灯片](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0034_00-46-05.jpg)

为什么第一刀就砍向 KV cache？讲者把逻辑讲得很直白：`Recall that memory is the bottleneck for inference. So let's try to reduce the size of the KV cache.`（回忆一下，内存是推理的瓶颈，那就试着把 KV cache 变小），屏幕上对应一个尚未展开的函数定义 `def reduce_kv_cache_size():`。他还特别指出一个容易被低估的事实：在足够大的 batch 下，KV cache 占用的内存「could even be larger than the number of parameters」——这正好由上一章的数值坐实，$B=256$ 时总内存约 240 GB，其中随 batch 增长的 KV cache 占了绝大部分，而模型参数只有约 26 GB。也就是说，优化权重本身（量化、剪枝）能省的只是一个较小的固定项，而缩小 KV cache 削掉的是那个随并发量线性膨胀的主项。既然 generation 是 memory-bound，减少要搬运的字节数就会直接换算成速度，这条因果链是本段所有技术的共同依据。

但「缩小」不能直接乱来。讲者紧接着提醒：`you want to make sure you don't lose too much accuracy in the process`——KV cache 里存的是每个 token 的 key/value 表示，少存或多份共享都等价于削弱 attention 的表达能力，属于 lossy change。于是本节形成了一个反复出现的两步节奏：先提出一种压缩/共享 KV 的架构改动，算出延迟与吞吐的收益；再单独去核对精度有没有掉（check that accuracy doesn't drop）。后文 GQA 章节会看到 Ainslie 2023 的评测表与 DeepSeek 论文的相反结论，MLA 章节会看到 DeepSeek 自己的 Table 8，滑动窗口与线性注意力章节则会反复出现「no free lunch」「hurts accuracy」的警告——这些都不是跑题，而是「有损改动必须校验精度」这条约束的具体兑现。一个实践上的易错点是把论文里的精度结论当成数学定理：讲者明确说，凡不是纯数学推导的结论都要「take it with a grain of salt」，因为它们依赖具体的模型规模、训练配方和评测集，换个设定就可能反转。

## 分组查询注意力 GQA：跨 head 共享 K/V [46:56 - 50:52]

第一种、也是「hindsight obvious」的手法，是前面记号系统里已经埋好的 grouped-query attention（GQA，分组查询注意力）。讲者先复习三种注意力的定义：在 multi-head attention（MHA，多头注意力）里，每个 token 的每个 head 都各自有一份 query、key、value；GQA 保留同样数量的 query heads，但只计算更少的 $K$ 组 key/value，让每组 K/V 服务多个 query heads；而 multi-query attention（MQA，多查询注意力）是 $K=1$ 的极端情形，讲者直言「no one uses because it's really bad」，因为把所有 query 都压到同一份 K/V 上，精度损失太大。论文里的 Figure 2 把三者的结构并排画了出来：Multi-head 一列是 8 个 Values、8 个 Keys、8 个 Queries 一一对应；Grouped-query 一列是 4 个 Values、4 个 Keys 对应 8 个 Queries，即每个 K/V 组服务 2 个 query head；Multi-query 一列则只有 1 个 Value、1 个 Key，虚线扇出到全部 8 个 Queries。图注原文精准概括了这种插值关系：`Multi-head attention has H query, key, and value heads. Multi-query attention shares single key and value heads across all query heads. Grouped-query attention instead shares single key and value heads for each group of query heads, interpolating between multi-head and multi-query attention.`

![GQA 论文 Figure 2：MHA、Grouped-query、Multi-query 三种注意力头共享结构对比](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0035_00-47-25.jpg)

用本课程的记号写出来，屏幕下方给出的定义是 `Idea: N query heads, but only K key and value heads, each interacting with N/K query heads`，并标注 `Multi-headed attention (MHA): K=N`、`Multi-query attention (MQA): K=1`。也就是说 $N$ 个 query heads 被分成 $K$ 组，每组 $N/K$ 个 query 共享一份 K/V。由此得到最核心的定量结论（黄色高亮）：`GQA reduces the KV cache by a factor of N/K.`。回顾 KV cache 的字节公式，其中每个因子都是后续优化的抓手：

$$\text{KV cache bytes} = B \cdot S \cdot L \cdot K \cdot H \cdot 2 \cdot 2, \qquad \frac{\text{KV cache}_{\text{GQA}}}{\text{KV cache}_{\text{MHA}}} = \frac{K}{N} = \frac{1}{N/K}.$$

这里 $K$ 正是 KV head 数，所以把 $K$ 从 $N$ 降到 $K$ 组，内存按 $N/K$ 成比例缩小（MHA 时 $K=N$，比值为 $1$；$N=40$、$K=8$ 时只剩 $1/5$）；再结合「generation 是 memory-bound」，屏幕上的提醒句 `Reminder: reducing memory usage leads to speedup (since we're memory-bound)` 就把内存下降直接翻译成了延迟与吞吐的改善。配套的折线图来自 Ainslie 2023，横轴是 `GQA groups`，刻度为 1、4、8、16、32、64，纵轴是 `Time per sample`，图例有蓝色实线的 GQA 和橙色虚线的 MQA：MQA 恒定处于最低位，GQA 曲线在 $K=1$ 到 $K=8$ 附近几乎贴着 MQA 的低位，从 $K=16$ 开始抬头，$K=32$ 明显上升，到 $K=64$ 则陡峭冲高。这正说明在精度与速度之间存在一个「somewhere in between」的甜点区——$K$ 取 8 左右仍然很快，一味增大组数就退回 MHA 的开销。

接下来讲者把这个抽象收益代回上一章那个「friendly llama model」上做数值对比。代码先在原始配置下求值：`config = llama2_13b_config({K: 40, B: 64})`，得到 `k40_b64 = compute_transformer_performance_stats(config)`；然后按 `# GQA with 1:5 ratio (K:N)` 改成 `config = llama2_13b_config({K: 8, B: 64})  # Use GQA with 1:5 ratio`，得到 `k8_b64`。青色结果框里的两组数值非常直观：`k40_b64` 的 `num_params` 为 $13{,}015{,}449{,}600$、`memory` 为 $79{,}717{,}990{,}400$、`latency` 为 $0.0238$、`throughput` 为 $2689.4807$；改成 `k8_b64` 后 `num_params` 降到 $11{,}337{,}728{,}000$、`memory` 降到 $33{,}412{,}874{,}240$、`latency` 降到 $0.0100$、`throughput` 升到 $6416.6883$。也就是说，仅仅把 KV 组数从 40 降到 8（$N/K = 5$ 倍压缩），内存约减少到原来的 $42\%$，延迟约减半，吞吐涨到约 2.4 倍，而且连参数量都因为 K/V 投影矩阵变小而少了约 1.7B。

![Llama 2 13B 上 MHA 与 1:5 GQA 在参数量、内存、延迟、吞吐上的数值对比](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0037_00-50-05.jpg)

这里有一处极容易读错、也最值得澄清的地方。幻灯片黄色高亮的注释仍写着 `Result: Worse latency, but better throughput (and it fits in memory now!)`，看上去像是沿用前面「增大 batch」情境的模板文案；但对照上面两组数值，latency 明明是从 $0.0238$ 改善到 $0.0100$。讲者在口头上专门纠正了这一点：`So, it's not that latency and throughput are always at odds. If you reduce the amount of memory, then it improves both. It's mainly the batch dimension that allows that is the point of tension.`——延迟与吞吐并非天生对立：凡是靠「减少内存」得到的收益，两者会同时改善；真正的张力只来自 batch 维度（增大 $B$ 用延迟换吞吐）。这正是本章与第 1 章的呼应，也是判断一个优化「属于哪一类」的分水岭。随后讲者继续演示联合调参：`# Now we can increase the batch size`、`config = llama2_13b_config({K: 8, B: 256})  # Increase batch size`、`k8_b256 = compute_transformer_performance_stats(config)`，结论是 `Result: Worse latency, but better throughput (and still fits in memory!)`。关键在于，之前 $B=256$ 会触发 OOM，现在因为 KV cache 被压到 $N/K$ 分之一，居然装得下了；讲者总结说「sometimes you kind of play with these parameters jointly」——先压缩 KV cache 腾出显存，再用更大的 batch 把腾出的空间换成吞吐，这是一组需要联调的参数，而不是互不相关的旋钮。易错点也在这里：如果只盯着 latency 数字，会误以为 GQA 让延迟变差；实际上变差的是「在 GQA 基础上又增大 batch」这一步，两步要分开归因。

## 有损改动必须校验精度：GQA 的评测争议与 MLA [50:52 - 54:47]

GQA 是典型的 lossy change，所以讲者把「收尾动作」单独拎出来：`The final thing you have to do whenever you do some lossy change is that you make sure the accuracy doesn't drop.`，屏幕上对应一行 `Check that accuracy doesn't drop: [Ainslie+ 2023]`。幻灯片贴出了 GQA 原论文的评测表，列头是 `Model | T_infer (s) | Average | CNN (R_1) | arXiv (R_1) | PubMed (R_1) | MediaSum (R_1) | MultiNews (R_1) | WMT (BLEU) | TriviaQA (F1)`，覆盖摘要、翻译、问答等多类任务。四行数据分别是：`MHA-Large` 推理时间 $0.37$ s、Average $46.0$、CNN $42.9$、arXiv $44.6$、PubMed $46.2$、MediaSum $35.5$、MultiNews $46.6$、WMT $27.7$、TriviaQA $78.2$；`MHA-XXL` 为 $1.51$ s、$47.2$、$43.8$、$45.6$、$47.5$、$36.4$、$46.9$、$28.4$、$81.9$；`MQA-XXL` 为 $0.24$ s、$46.6$、$43.0$、$45.0$、$46.9$、$36.1$、$46.5$、$28.5$、$81.3$；`GQA-8-XXL` 为 $0.28$ s、$47.1$、$43.5$、$45.4$、$47.7$、$36.3$、$47.2$、$28.4$、$81.6$。把关键两行并排看：GQA-8-XXL 的 Average $47.1$ 几乎追平 MHA-XXL 的 $47.2$，而 $T_{infer}$ 从 $1.51$ s 降到 $0.28$ s，约 5.4 倍加速——这正是 GQA 论文想证明的「速度大幅改善、精度几乎不掉」。

![Ainslie 2023 GQA 论文表格：MHA-Large、MHA-XXL、MQA-XXL、GQA-8-XXL 的推理时间与多项评测分数](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0038_00-51-25.jpg)

但讲者立刻给这张表泼了一盆冷水，态度值得原样记住：`I think you always have to take it with a grain of salt ... a later the deep seek paper would just show that it actually does hurt. So, you know, I guess take everything that's not just kind of math with a grain of salt here.` 换句话说，这类评测结论依赖具体模型规模、训练配方与 benchmark，后来 DeepSeek 的论文就显示 GQA 其实会伤害精度。屏幕上紧接着的高亮句把这个张力写得很直接：`First, MHA is better than GQA (though more expensive) [Table 8] [DeepSeek-AI+ 2024]`。DeepSeek 的 Table 8 在 7B 稠密模型上对比了三种注意力，列头是 `Benchmark (Metric) | # Shots | Dense 7B w/ MQA | Dense 7B w/ GQA (8 Groups) | Dense 7B w/ MHA`：`# Params` 分别为 $7.1$B、$6.9$B、$6.9$B；`BBH (EM)` 3-shot 为 $33.2$、$35.6$、$37.0$；`MMLU (Acc.)` 5-shot 为 $37.9$、$41.2$、$45.2$；`C-Eval (Acc.)` 5-shot 为 $30.0$、$37.7$、$42.9$；表格底部还列有 CMMLU 一行。可以看到在 MMLU、C-Eval 这类知识型评测上，MHA 比 GQA (8 Groups) 高出约 4 到 5 分，MHA 明显更好，只是更贵。于是本段出现了一个诚实的方法论提醒：GQA 论文与 MLA 论文的精度结论「in tension」，谁对谁错不能靠引用某一篇解决，只能针对自己的模型和任务重新做评测；凡是纯数学推导之外的经验结论，都要保留怀疑。

顺着「speaking of deep seek」的过渡，讲者引入本段第二种、压缩得更激进的手法：multi-head latent attention（MLA，多头潜在注意力），出自 `[DeepSeek-AI+ 2024]`（DeepSeek V2）。它和 GQA 的思路有本质区别：GQA 是减少 K/V 的「头数」，MLA 则「leave the number of keys and values the same, one for every token essentially, but I'm going to compress these」——头数不变，转而去压缩每份 K/V 的表示维度。幻灯片结构图用斜线填充标注 `Cached During Inference`（推理时被缓存的部分）：MHA 栏里 8 个 Values 和 8 个 Keys 全是斜线（都要缓存），GQA 栏缩到 4 个，MQA 栏只剩 1 个；而 MLA 栏里 8 个 Values、8 个 Keys、8 个 Queries 全都不带斜线，只在最右侧单独缓存一根斜线条 `Compressed Latent KV`，并通过一个向左的 `projection` 箭头在需要时把 K/V 还原出来。换句话说，MLA 缓存的不是 K 和 V 本身，而是一个低维潜在向量 $c$，用到时再 materialize（实例化）出 $K$、$V$。

![DeepSeek V2 的 MLA 结构图：MHA/GQA/MQA/MLA 缓存对比与压缩潜在 KV 的公式](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0039_00-52-45.jpg)

图下方的公式把两种做法写得很清楚。普通 attention 是 `Normal attention: KV cache consists of K = W_K h, V = W_V h (N*H dimensions)`，即激活 $h$ 分别乘 $W_K$、$W_V$ 得到 $K$、$V$，维度是 $N \cdot H$（等于模型维度，很大），缓存的正是这两个大张量。MLA 则是（黄色高亮）`MLA: store compressed vector c = W_c h (C dimensions), project up to K = W_K c, V = W_V c when needed`：先用 $W_c$ 把 $h$ 下投影到 $C$ 维的压缩向量 $c = W_c h$，只缓存 $c$，需要计算 attention 时再分别用 $W_K$、$W_V$ 上投影回 $K = W_K c$、$V = W_V c$。用 LaTeX 写即 $c = W_c h \in \mathbb{R}^{C}$，$K = W_K c$，$V = W_V c$，缓存量从 $N \cdot H$ 降到 $C$。屏幕给出的具体数字是 `DeepSeek v2: reduce N*H = 16384 to C = 512`——从 16384 维压到 512 维，约 32 倍的激进压缩（讲者口头上把 16384 说成「16,000」，幻灯片上的精确值是 16384）。

这里还有一个必须处理的技术细节（wrinkle）：`Wrinkle: MLA is not compatible with RoPE, so need to add additional 64 dimensions for RoPE, so 512 + 64 = 576 total dimensions`。RoPE（rotary position embedding，旋转位置编码）是直接作用在 keys 和 queries 上的位置变换，而 MLA 缓存的是压缩后的 $c$，无法在压缩表示上直接施加旋转；DeepSeek 的解决办法是额外留出 64 个维度专门承载 RoPE 相关信息，于是总的缓存维度变成 $512 + 64 = 576$，即便如此相对 16384 仍是很大的缩减。屏幕上随后一句 `Latency/throughput improvements follow similarly from the KV cache reduction as argued earlier` 说明收益机制与 GQA 完全一致：KV cache 变小 → 内存流量下降 → 在 memory-bound 的 generation 阶段换算成延迟与吞吐的改善，讲者形容为「almost kind of a linear scaling up until some point」（在某个点之前近似线性）。精度方面，DeepSeek 报告 MLA 与 MHA 相当、甚至略好，这与前面 GQA 论文的乐观结论和 DeepSeek 自己 Table 8 里「MHA 优于 GQA」的判断又构成一层张力。一个易错点是把 MLA 与「直接降低模型维度」混为一谈，这正是下一段学生提问的切入点；另一个是忘了 $576$ 这个数——若只记 512 会低估实际缓存开销，也会漏掉 RoPE 兼容性这个设计约束。

## 降低模型维度的问答与跨层注意力 CLA [54:47 - 56:53]

MLA 讲完后，一位学生提出了一个很自然的追问：`How does this compare, uh, to reducing the number of dimensions of the model?`（这跟直接把模型维度降下来相比如何？）。讲者先把问题复述一遍 `So, the question is, how does this compare with reducing the dimension of the model?`，然后给了一个相当诚实的回答：`These ablations don't show that.`——手头这些消融实验并没有做「整体降维」这一组对照。他随后给出的判断是猜测而非结论，措辞里全是保留：`My guess is that reducing the model dimension just makes things, you know, worse if you should because you're sort of like indiscriminately just like reducing everything.`。理由在于，模型维度 $D$ 是一个贯穿全网的全局旋钮，把 $D$ 调小会同时削弱 MLP 的 $F=4D$、attention 的 $N \cdot H$ 投影、embedding 与所有残差流，属于「不加区分地把一切都压小」；而 MLA 只针对 KV 表示这一处做下投影，模型的其余容量保持不变。讲者把这类方法的诀窍总结为 `the trick in all of this kind of is to find places in the model where you can squeeze`（找到模型里可以挤压的特定位置），并强调 `this a priori, I don't think you can necessarily know for sure. You just have to do a bunch of experimentation to see what works.`——哪些地方能压、压多少，无法先验确定，只能靠大量实验试出来。这里必须保留讲者的语气：这是猜测（my guess）、是方法论建议，不是论文里已验证的定量结论；把它写成「降维一定更差」就过度解读了。

紧接着讲者用 `Here's another idea to reduce your, uh, KV cache. This is called cross-layer attention.` 引出第三种共享思路：cross-layer attention（CLA，跨层注意力）。正常 Transformer 里每一层都有自己的 K 和 V，回顾 KV cache 的字节公式 $\text{KV cache bytes} = B \cdot S \cdot L \cdot K \cdot H \cdot 2 \cdot 2$，其中 $L$ 是层数，每一层都贡献一份 $K \cdot H$ 的 K/V。CLA 的做法是「we're just going to compute KVs for a subset of the layers and then just for this layer I'm just going to use the previous layers KV cache」——只在一个层的子集上真正计算 K/V，其余层直接复用前面某一层已经算好的 KV cache。讲者用一句类比把它和 GQA 对齐：`just like GQA shares KV's across heads. Now I'm sharing KVs across layers.`（GQA 是在 head 之间共享 K/V，CLA 是在 layer 之间共享）。于是本段至此已经有了一条清晰的分类轴：GQA 沿 head 维度共享、CLA 沿 layer 维度共享、MLA 沿表示维度（$N\cdot H \to C$）压缩，三者都落在 $\text{KV cache bytes}$ 公式的不同因子上——分别动的是 $K$、$L$、以及隐含在 $K\cdot H$ 里的每头维度。

关于效果，讲者的表述依旧克制：`empirically this paper shows that um you know doing this improves the Pareto you know frontier`。Pareto frontier（帕累托前沿）在这里指的是「精度—KV cache 大小」的权衡曲线：在给定 KV cache 预算下能拿到更高精度，或在给定精度下用更小 KV cache，就算把前沿往外推。他进一步解释比较的口径：`each of these these models is better than ... in given a method you can always uh kind of sweep this the size of the KV uh cache by changing the K and um the head dimension`——对任何一种方法，都可以把 KV cache 大小当作可扫描的横轴，具体靠改 $K$（KV 组数）和 head dimension（每头维度）这两个旋钮来扫；而这恰好 `kind of relates to your point about changing model dimension`，回应了上一位学生关于改模型维度的提问。结论是 `if you do this uh CLA cross-layer attention, it's uh it's better`，即在同一条扫描曲线上，CLA 能把前沿推得更好。这一小段需要留意两处局限：一是讲者没有给出任何具体数字或表格（画面上是他在白板前口头讲解，白板基本空白），所有结论都是定性的「improves the Pareto frontier」；二是「换 head dimension 来扫 KV cache」只是 sweeping 的实验手段，不等于 CLA 本身改变了 head dimension，两者的因果关系不要倒置。

## 滑动窗口、混合注意力与线性注意力 [56:53 - 62:17]

`So let me give you uh moving on um world wind tour of different techniques for reducing KV cache.`——讲者宣布要快速巡览一批缩小 KV cache 的手法，第一个是 local / sliding window attention（局部 / 滑动窗口注意力），并说这是「fairly kind of old idea and a very kind of natural idea」。出发点是最朴素的成本观察：full attention（全注意力）的注意力矩阵是 $n^2$ 规模（`if you look at the full attention matrix is n squared`），序列越长越贵。滑动窗口的做法是：生成某个 token 时「you just look at the last K tokens」，即每个 token 只依赖最近的 $K$ 个 token，`you essentially have a sliding window for every token you generate, you just depend on the last K`。用代价写就是，原本每步要读 $S$ 个历史 token 的 K/V，现在只读 $K$ 个（$K \ll S$）。

这一改动带来一个结构上非常漂亮的性质：`the KV cache is now independent of the sequence length. It's just, you know, um the number of uh um the batch times the the other variables.`——KV cache 的大小不再随序列长度 $S$ 增长，只剩下 $B$ 乘上其它与长度无关的因子（可写成 $\text{KV cache} \propto B \cdot K \cdot L \cdot H$，其中不再含 $S$）。讲者特别点出 `this is especially great for long context`：对长上下文场景，这一项从「随对话越滚越大」变成「恒定上限」，意义很大。他还补充了一个反直觉的细节：`because of number of layers, the effective context length has actually larger than the number of the stated context length because information can propagate um you know, farther and if you go down the layers`——由于多层堆叠，信息可以逐层向前传播，模型的有效感受野（effective context length）其实比单层宣称的窗口 $K$ 更大。此外还可以做更花样的变体：不必对每层都密集地开窗口，可以 `space it out`（把层间隔开）；也可以做 `global plus sliding window`，即「attention to a fixed um you know, grid of different, you know, uh token points plus a local uh sliding window」，对一组固定的全局 token 点做注意力，再叠加局部滑动窗口。

代价同样直白：`Now, the problem with this is that it actually still hurts accuracy. So this reduces expressivity. There's no, you know, free lunch here. Um or at least this was an expensive lunch.`（它仍然伤精度、降低表达力，没有免费午餐，或者至少这顿午餐很贵）。这与前面 GQA、MLA 章节「有损改动必须校验精度」的约束一脉相承，只是这里连精确的评测表都没有给，只给了定性警告。业界的应对是把两类层交错起来：`they interleave local attention with uh global attention. So these hybrid uh models have uh you know, full attention for some of the layers and some local layers for, you know, some of the other layers`，即 hybrid model（混合模型）里一部分层用 full attention、另一部分层用 local attention，从而在 `reduce the the KV cache a little bit` 的同时，`you're always trying to balance uh you know, accuracy with speed`。注意这里的收益被讲者限定为「a little bit」——交错方案只是折中，不像 GQA/MLA 那样给出成倍的定量压缩。

随后两位学生把讨论引向 linear attention（线性注意力）与 sliding window 的取舍。讲者先声明 `I'm not going to talk about linear attention, but maybe very quickly`，因此下面只是速览而非严格推导。核心区别在于：线性注意力「instead of storing a KV cache, this you basically um, compute some sort of like compressed representation of all the history」——不再缓存完整的 K/V 历史，而是把历史压缩成一个固定规模的状态。最朴素的线性注意力 `linear the most naive linear attention is uh, you just sum the KV uh, values up into a single vector`，即把所有 K/V 求和压成一个向量，`that's definitely independent of the sequence length`；更复杂的变体有 GateNet、DeltaNet、Mamba，它们 `allows you to um, try to compress, but not forget, you know, as much`（压缩的同时尽量少遗忘）。这些方法也被用来替代滑动窗口注意力，且「people have gotten good results with them」。更进一步，三者可以组合使用：`You can also use a combination of full attention, sliding window attention, and the linear attention, because they sort of capture different, you know, aspects`，讲者给出的分工是 `If you care about kind of local kind of high-resolution stuff, then sliding window attention's better. If you just kind of want like broad summaries of the past, then um, the other, you know, the linear attention might be better.`——局部高分辨率信息靠滑动窗口，对过去的粗粒度摘要靠线性注意力/状态空间模型。

关于长上下文，学生的追问是「for a long context, would linear attention be better?」，讲者的回答回到那句纲领：`so there's no free lunch, right?`。他用一个 needle-in-a-haystack（大海捞针）式的设定说明：`If you have a very long context, and you're sort of solving a needle in a haystack problem ... If you have to compress your entire history into like a small context, you're just going to lose information, and um, you might just you know, not be able to retrieve it.`——只要把整段历史压进一个小状态，信息必然丢失，那个「针」可能就取不回来了。不过在表达能力上他仍认为 `the Mamba and DeltaNet are more powerful um than the sliding window attention`：因为 Mamba 在做递推时可以 `just look at the last you know state`，从而至少能表示滑动窗口注意力的某些行为（`you can just as you're doing the recurrence it can just look at the last state`）；而 `once you do sliding window attention like you're done there's nothing else you can`——滑动窗口一旦定死窗口大小，就没有更多可调节的表示空间了，线性注意力及其扩展「at least they have more room」。这一整章都要记住它的证据等级：讲者明确说线性注意力只是「very quickly」带过，没有给公式、没有给数字、没有给评测表，全部是定性的机制解释与经验判断；把它当成已被严格证明的结论，或把 Mamba/DeltaNet 与滑动窗口的优劣当成普适定论，都超出了素材本身。

## DeepSeek 的压缩稀疏注意力与本节收束 [62:17 - 64:07]

最后一段讲者用 `DeepSeek continues to you know innovate different types of attention mechanisms` 把镜头拉回 DeepSeek：既然它们已经用 MLA 压缩了 key/value，后续又继续推出新的注意力机制。他一口气报出三个缩写，并自嘲 `I never kind of remember all these acronyms and what they they mean`：`compressed sparse attention`、`DeepSeek sparse attention`、`heavily compressed attention`。随即 `let's look at this diagram` 切到架构图，图的顶部横条标题是 `Shared Key-Value Multi-Query Attention`。整个数据流从底部两路输入开始：左侧一排黄色框是 `Hidden States of KV Tokens`，右侧单独一个绿色框是 `Hidden State of Query Token`；KV 一路经过 `Token-Level Compressor` 变成 `Compressed KV Entries`，再往上由 `Top k Selector` 挑出 `Selected Compressed KV Entries`，同时左侧还有一支 `Sliding Window KV Entries`；两支在 `Concatenation` 汇合后送入顶部的共享注意力模块。图的右半被一个虚线框圈成 `Lightning Indexer`（闪电索引器），里面包含由另一路 `Token-Level Compressor` 产出的 `Compressed Indexer Keys`、`Multi-Query Attention`、`Indexer Queries`、`Queries`，以及输出的 `Index Scores`——正是这组分数回过头去驱动 `Top k Selector` 做筛选。图底部三条 bullet 把缩写与机制一一对应（最后一条黄色高亮）：`Compressed Sparse Attention (CSA): compresses every m tokens into 1`、`DeepSeek Sparse Attention (DSA): selects the top k`、`Heavily Compressed Attention (HCA): compresses even more`。

![DeepSeek Shared Key-Value Multi-Query Attention 架构图：Lightning Indexer、Top k Selector 与 CSA/DSA/HCA](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0047_01-03-27.jpg)

把图和讲者的口头解释对上，逻辑是这样的。第一步是压缩：`compressed attention is going to basically compress every M tokens into one token`，对应 CSA，即每 $m$ 个 token 压成 1 个，等价于把 KV 条目数砍掉约 $m$ 倍。第二步是稀疏选择：`there's this thing called DeepSeek sparse attention which basically selects a subset of those to kind of keep`，对应 DSA 的 `selects the top k`。关键在于「怎么快速知道该保留谁」：`the way you select a subset is that you actually compute some light lighter weight queries and keys and then you do it like a, you know, a small attention to get these index scores so you know what to keep. So like a kind of a lightning fast way to figure out what tokens you need to keep and then you use those.`——用一组更轻量的 query/key 做一次小规模 attention，算出 index scores（索引分数），据此挑出要保留的 token，这就是图里 `Lightning Indexer` 那条支路的作用；它是一个「闪电般快速」的预筛，避免对全量 KV 做重计算。第三步 `and then there's a more compression that, you know, happens` 对应 HCA 的进一步压缩。图里同时出现的 `Sliding Window KV Entries` 也值得留意：它与筛选后的 `Selected Compressed KV Entries` 拼接，说明这套方案把「局部窗口」与「全局稀疏精选」组合在同一层里，恰好呼应上一章 full/local/linear 可组合的思路。讲者因时间关系 `in the interest of time I'll move on` 没有展开更多细节，因此这里不宜替他补出具体压缩率或 top-$k$ 的取值——画面和转写都没给。

本段最后用一小段把整节的因果链收拢，值得逐句记住。`The goal of this section is to reduce the KV cache um because KV cache is related to memory and we re saw that uh inference is memory bound. So um that directly translates to improvements in throughput and uh latency.`——目标始终是缩小 KV cache，因为 KV cache 直接决定内存占用，而前面已经证明 inference（尤其 generation）是 memory-bound，所以内存下降会直接换算成吞吐与延迟的改善。这与第 1 章的 $B=256$ OOM、第 3 章的 $N/K$ 压缩、第 4 章的 $16384 \to 512$ 是同一条逻辑在不同层面上的反复运用。紧接着 `And the key is to do this without hurting accuracy.` 再次把「不伤精度」立为硬约束，与 Ainslie 表格、DeepSeek Table 8、滑动窗口的「no free lunch」形成闭环。最后一句给出了本段所有手法的统一分类：`So you can do lower dimensional KV uh caches across layers um by across uh heads um and across like, you know, head dimension.`——降低 KV 维度的三个轴分别是跨层（across layers，对应 CLA）、跨头（across heads，对应 GQA）、以及跨 head dimension（对应 MLA 把 $N\cdot H$ 压到 $C$）。一个容易忽略的点是，讲者说的「lower dimensional KV caches」不只是物理上少存几个字节，而是把 $\text{KV cache bytes} = B \cdot S \cdot L \cdot K \cdot H \cdot 2 \cdot 2$ 中的 $L$、$K$、$H$ 三个因子分别作为可压缩的自由度；再加上滑动窗口/线性注意力干脆让 $S$ 这一项消失或变成常量，本段的技术地图就完整了。至于这些 lossy 方法究竟能压到什么程度而不掉精度，讲者始终没有给出普适答案——这仍要回到「靠实验、按任务评测」那句提醒上。

## 量化的定位与数值机制 [64:07 - 65:15]

本段承接上一部分对「更快架构」的讨论。画面 `kf_0048` 顶部还残留着上一章幻灯片的结尾：`DeepSeek Sparse Attention (DSA): selects the top k`、`Heavily Compressed Attention (HCA): compresses even more`，以及一段 `Summary`，其第一条即 `Goal: reduce the KV cache size (since inference is memory-bound) without hurting accuracy`，随后列出 `Lower-dimensional KV cache (GQA, MLA, CLA)`、`Local attention (truncates the KV cache) on some of the layers`、`Other ideas: linear attention / state-space-models (Mamba 2, GatedDeltaNet), diffusion models`。转写在 64:13–64:22 也以 `You can do local attention. You can do linear attention… And there's much more. And also there's also diffusion models, which is a non-autoregressive way to generate, which can be much faster` 收束了架构类方法。紧接着讲师用 `let's talk about a few other ideas` 转向量化（quantization），并特意把它与前面所有架构方法划清界限：`quantization is more of a I would say less of a architecture and much more of a systems perspective on how to make things, you know, smaller`（64:29–64:40）。也就是说，GQA、MLA、局部注意力改的是 attention 的结构与 KV cache 的元素个数，而量化改的是「数字本身占多少字节」，属于系统视角。这条分界线是理解后续 20 分钟内容的关键：从这里开始，优化对象从「模型怎么算」逐渐转向「字节怎么存、怎么搬」。

讲义以 Python 函数风格给出量化的定义，画面 `kf_0048` 中可逐字读到：

```python
def quantization():
    Key idea: reduce the precision of numbers
    Less memory means higher latency/throughput (since inference is memory-bound).
    Of course we have to worry about accuracy...   # 黄色高亮行
```

讲义里的 `Key idea: reduce the precision of numbers`（降低数值精度）是全部内容的起点。之所以有效，括号里的理由 `since inference is memory-bound` 直接呼应全课的分析框架：generation 阶段每生成一个 token 都要把权重和 KV cache 从 HBM 搬进计算单元，搬运量而非 FLOPs 决定耗时，因此每个数从 2 字节降到 1 字节，内存流量几乎减半，延迟与吞吐随之改善。这里有一个必须点破的措辞陷阱：讲义与转写 64:47–64:52 都写成 `Less memory means higher latency and throughput`，字面看像是「延迟变高」，但这与整门课结论（减小内存 → 减少字节 → 更低延迟、更高吞吐）矛盾；合理的读法是把 `higher latency/throughput` 理解成「延迟与吞吐这两项性能指标都变得更好」，是讲义压缩写法造成的歧义，不能照字面理解成延迟恶化。最后一行被黄色高亮的 `Of course we have to worry about accuracy...` 则把矛盾摆到台面上：精度换来的内存收益，必须以精度损失为代价，后面所有方法都在围着这个 `accuracy` 打转。

画面 `kf_0048` 下半部的淡灰代码把「降精度」落到了可计算的机制上，这是转写里完全没有念出、只存在于画面上的内容：

```python
# Mechanics
x = 5.2342
scale = 0.1
zero_point = 4
x_quant   = round(x / scale) + zero_point      # Quantize
x_dequant = (x_quant - zero_point) * scale     # Dequantize
```

这是标准的仿射量化（affine quantization）：用 `scale`（缩放因子）把浮点 `x` 缩到整数格点，加上 `zero_point`（零点偏移）后取整得到低精度整数 `x_quant`；反量化时减去零点、乘回 `scale` 还原。代入画面给的具体数值算一遍：$x/\text{scale} = 5.2342/0.1 = 52.342$，`round` 后为 $52$，加 `zero_point = 4` 得 $x_{quant} = 56$；反量化 $(56 - 4)\times 0.1 = 5.2$。原始值 $5.2342$ 与还原值 $5.2$ 相差 $0.0342$，这 $0.0342$ 就是量化误差，也正是高亮行所担心的 `accuracy` 损失的微观来源。`scale` 与 `zero_point` 不是全局唯一的：转写 65:58–66:10 与 `kf_0049` 都说明，朴素做法是「for every layer or tensor, you basically determine the scale and the zero point for each」，即按层或按张量各自标定。把这段 Mechanics 记住，第 2 章的 QAT「量化再反量化」和 PTQ「定 scale/zero point」都只是这两行代码在不同阶段的复用。

从与前后内容的关系看，量化与前面的架构方法正交互补：GQA/MLA 减少 KV cache 里向量的个数，量化减少每个元素的字节数，两者可以直接叠加，收益相乘。但它也带来两个局限，需要在阅读后续内容时始终记着。其一，量化是「有损」的（lossy）——`kf_0051` 会明确把 `quantization()` 归到加粗标题 `Taking shortcuts (lossy)` 之下，与 `reduce_kv_cache_size()`、`model_pruning()` 并列；它不像后面第 4 章的投机解码那样能保证输出分布精确。其二，精度并非降得越狠越好：从 BF16 一路到 INT4，字节数确实降到四分之一，但量化误差迅速放大，必须靠训练时适配（QAT）、二阶误差补偿（GPTQ）、按激活重要性分配精度（AWQ）等手段把损失拉回来——这正好是下一章的主题。另一个易错点是：内存收益直接成立的前提是「generation 是 memory-bound」；对于本就 compute-bound 的 prefill，或将来算力/带宽比大幅变化的硬件，降精度带来的加速幅度不会和这里一样。

![量化函数定义、reduce precision 与 accuracy 高亮行，以及 scale/zero_point 的量化反量化 Mechanics 代码](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0048_01-04-53.jpg)

## 精度阶梯与量化路线：QAT、PTQ、GPTQ、AWQ [65:15 - 67:40]

转写 65:01–65:06 只笼统地说量化「there's many options here all the way from BF16 all the way down to int four」，而画面 `kf_0049` 给出了完整的精度阶梯，包含每种格式的字节数、取值范围和文献引用，这是本段最重要的图上增量之一：`bf16 (2 bytes): default for inference`；`fp8 (1 byte) [-240, 240] for e4m3 on H100s: can train if you dare [Peng+ 2023]`；`int8 (1 byte) [-128, 127]: less accurate but cheaper than fp8, but for inference only [Baalen+ 2023]`；`int4 (0.5 bytes) [-8, 7]: cheaper, even less accurate [Baalen+ 2023]`。把数字摆在一起看规律很清楚：BF16 每数 2 字节是推理默认；INT4 每数 0.5 字节（一个 nibble），相对 BF16 是 $4\times$ 压缩；$[-8, 7]$ 与 $[-128, 127]$ 分别是 4-bit 与 8-bit 有符号整数的取值区间，后者恰为 $2^7$ 量级，前者为 $2^3$ 量级。FP8 的 e4m3 格式在 H100 上给出 $[-240, 240]$ 这种非均匀浮点范围，与整型范围不可直接类比。还有一个容易被忽略的标注差异：`fp8` 条目写 `can train if you dare`（敢的话也能拿来训练），而 `int8`、`int4` 都标 `for inference only`（仅推理），说明低精度整型基本只服务推理阶段。文献 `[Peng+ 2023]`、`[Baalen+ 2023]` 是转写没有念到的引用，画面把它们与具体格式绑定，便于按图索骥。

画面 `kf_0049` 的 `[Overview of approaches]` 小节把量化实现分成两条路线，逐字内容为：`Quantization-aware training (QAT)` 下三条 bullet——`During training, quantize-and-dequantize during the forward pass to simulate quantization errors`、`Pro: weights are trained to work with quantization`、`Con: requires expensive large-scale training`；`Post-training quantization (PTQ)` 下三条——`Done after training, so much cheaper`、`Run on sample data to determine scale and zero point for each layer or tensor`、以及黄色高亮的 `GPTQ: use Hessian information to update non-quantized weights to account for quantization error [Frantar+ 2022]`。转写 65:15–65:56 与之对应：如果担心量化「mess you up」，可以 `train a model with quantization in mind`，即 QAT，在前向传播里 `quantize and dequantize`，本质上就是把上一章 Mechanics 的两行代码反复执行、把量化误差当成训练噪声模拟出来，于是 `the weights are adapted towards the quantization`；代价是 `requires expensive large-scale training`。正因如此，`what typically people do is they train models and then they quantize after the fact`，也就是 PTQ，`It's much cheaper often`。注意 PTQ 的 `Run on sample data to determine scale and zero point` 正是上一章 `scale = 0.1`、`zero_point = 4` 的来历：这些参数不是拍脑袋定的，而是用少量样本数据（转写 66:04–66:10 说按每个 layer 或 tensor）标定出来的；转写 66:10–66:15 也点明这种「各张量单独量化」的朴素做法 `generally doesn't work as well`。

第三条高亮 bullet 把 GPTQ 单独拎出来，值得展开。转写 66:20–66:37 的解释是：GPTQ `uses some Hessian information to quantize layer by layer`，并且 `you keep track of the errors which get propagated into the non-quantized weights, and so it kind of allows you to correct for the errors`。用式子概括这种逐层补偿：设某层权重 $\mathbf{W}$，量化算子为 $Q(\cdot)$，则量化误差 $\mathbf{E} = \mathbf{W} - Q(\mathbf{W})$ 不被丢弃，而是借助 Hessian（二阶曲率信息）按某种比例补偿到尚未量化的权重上，使整体输出尽量不变；这就是画面上 `update non-quantized weights to account for quantization error` 的含义，也是 `[Frantar+ 2022]` 这篇工作的核心。相对地，激活感知量化（activation-aware quantization，画面 `kf_0049` 底部淡灰已预告 `Activation-aware quantization (AWQ)`）走的是另一条路：转写 66:37–67:36 说，观察到 `some activation channels are large, and those the weights that interact with those matter more`，于是 `Let's allocate more precision to these weights`。具体做法是：正常把一个 FP16 权重矩阵整体量化到 INT3，但对少数「重要」的激活通道保留 FP16，其余通道仍是 INT3（转写 67:02–67:36）。用混合精度写成分块形式：

$$
\mathbf{W}_{:,j} = \begin{cases} \text{FP16}, & j \in S \ \text{（大幅值激活通道）} \\[2pt] \text{INT3}, & j \notin S \end{cases}
$$

其中 $S$ 是由激活幅值筛出的少数通道集合。`kf_0050` 顶部残留的层结构示意图（含 `RMSNorm`、`Group Query Attention`、`MLP`、`Layer 1`、`Layer L`）提示这种「按通道」的粒度是落在具体层内部的。这里有一处必须校正的听写噪声：转写把 FP16 记成 `FE16`、把 INT3 记成 `N3`（67:02、67:07、67:30），按画面和上下文都应是 `FP16` 与 `INT3`；若照抄成 `FE16`/`N3` 会凭空造出并不存在的数值格式，是典型的易错点。

把三种方法并起来看，它们其实是在用不同信息源压制同一份量化误差：QAT 靠训练让权重主动适应，GPTQ 靠二阶信息做误差补偿，AWQ 靠激活统计找出「哪些权重更值得给高精度」。三者仍都属于 `kf_0051` 所说 `Taking shortcuts (lossy)` 的有损范畴，谁也不能保证零精度损失。更根本的局限在于 AWQ 依赖一个经验事实——`some activation channels are much higher than others`。转写 70:45–71:01 讲师被追问时明确承认：这是一种 `empirical observation`，「If this weren't true, then these techniques wouldn't necessarily work」，模型恰好被训练成这样，于是可以 `exploit that`。换言之，AWQ 与后面第 3 章剪枝的重要性判据共享同一个未经理论保证的前提；一旦某个模型各通道激活幅度齐平，或换到分布迥异的数据上，分配精度的依据就会失效。这也解释了为什么紧接着在 69:55–71:43 会出现一大段关于「如何判断哪些层/神经元重要」的问答——它并不是插曲，而是对量化与剪枝共同假设的追问。

![BF16/FP8/INT8/INT4 精度字节数与取值范围，及 QAT、PTQ、GPTQ 与 AWQ 的方法 bullet 与论文引用](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0049_01-06-24.jpg)

## 模型剪枝、有损捷径的收束与重要性判定 [67:40 - 71:43]

67:40 讲师用 `another idea here is to do model pruning` 引入剪枝（model pruning），并给出一个很直白的定义：`you just take a large model, and you rip out pieces of it, and you fix it up. It's kind of a very crude way, but it turns out to work`（67:45–67:54）。画面 `kf_0050` 上方是 NVIDIA 那篇工作的 `Algorithm` 三步，逐字为：`1. Identify important {layer, head, hidden dimension} on a small calibration dataset (1024 samples)`；`2. Remove unimportant layers to get a smaller model`；`3. Distill the original model into pruned model`。转写 68:03–68:37 与之逐句对应：先 `estimate the importance of the different parts of the model`、选 `the most important parts`，再 `remove different hidden units, and different even layers`，此时 `now we have a model. It's not going to be very good`，于是 `what you do is you post-train it. You train it some more on the data or the task that you care about to kind of heal it`（继续训练把模型「治愈」）。这里最关键的具体数字是校准集只有 `1024 samples`——用极小的校准数据就能决定哪些 `{layer, head, hidden dimension}` 该保留，说明重要性估计本身很廉价；而第 3 步用的是 distill（蒸馏）而非简单微调，意味着原模型被当作教师、剪后的小模型当学生。`kf_0050` 顶部那幅层结构图里画着带 `RMSNorm`、`Group Query Attention (GQA)`、`MLP` 的重复块（标注 `Layer 1` 到 `Layer L`），正是被「撕掉」若干层或若干 head/隐藏维度之后要重新拼起来的对象。转写 68:37–68:50 还补了一句定位：这「in some sense a training purse weighted to reduce the KV cache, but where you sort of initialize it with parts of a good model」——即用好模型的部件作初始化，来得到一个 KV cache 更小的模型，而不是从零训练。

再看 `kf_0050` 下半部的散点图是转写里没有细讲、但信息量很大的部分。纵轴标注为质量得分（画面写 `IMLU Score (%)`，刻度 55、60、65），横轴是模型规模/成本，散点标出的模型包括 `Gemma 7B`、`Nemotron-4 15B`、`LLaMa-3 8B`、`Mistral 7B`、`Minitron 8B`、`Minitron 4B`、`Phi-2`、`Nemotron-3 8B`，并有一条红色箭头配文 `40x cheaper 9% better`，指向由剪枝加蒸馏得到的 Minitron 系列。转写 68:52–69:11 给出的结论与之呼应：`they were able to take a 15B model, and reduce it to a 8B model`，`and it doesn't really hurt accuracy by too much`，而且 `the amount that you use to kind of train the model through this process is much less`。把两组数字对齐：从 Nemotron-4 15B 出发得到 Minitron 8B/4B，训练成本约 $40\times$ 更低，质量还略好（画面写 `9% better`）；这解释了为什么剪枝值得做——它不是简单地「砍小变差」，而是借原模型权重这个高质量初始化，用远少于从头训练的算力换回精度。易错点在于不要把它误解成「15B 模型无损变成 8B」：转写用词是 `doesn't really hurt accuracy by too much`，是有损但可控；画面 `40x cheaper 9% better` 也是相对特定基线与特定评测而言的经验结果，不是普适保证。

69:16–69:53 讲师对前面所有有损方法做了一次收束，而 `kf_0051` 把这次收束做成了讲义大纲，画面信息比转写更结构化。页面上方列出已讲过的函数：`review_transformer()`、`review_of_arithmetic_intensity()`、`arithmetic_intensity_of_inference()`、`throughput_and_latency()`；随后是加粗标题 `Taking shortcuts (lossy)`，其下三项正是 `reduce_kv_cache_size()`、`quantization()`、`model_pruning()`；黄色高亮的总结行是 `Summary: reduce inference complexity without hurting accuracy`。下方淡灰还给出了两条配方，转写 69:31–69:49 用更口语的方式复述了它们：`From scratch recipe: 1. Define faster model architecture 2. Train faster model`，以及 `Distillation recipe: 1. Define faster model architecture 2. Initialize weight using original model (which has a different architecture) …`，再 `repair the faster model with distillation`。讲师把后者形容成 `you kind of just make this Frankenstein thing`——把原模型的权重塞进一个架构可能都不同的快模型里，再用蒸馏修补。用一句话概括这次收束：整场游戏就是 `reduce the inference complexity without hurting accuracy`，途径无非减少参数量或减少 KV cache，而实现路径只有「定义更快架构从头训」或「从原模型初始化 + 蒸馏修复」两条。这也把第 1、2 章的量化、本章的剪枝，与上一部分的 GQA/MLA/局部注意力统一到同一个框架下。

紧接着 69:55–71:43 是一段问答（`kf_0052` 是讲师站在白板前、双手摊开作答的课堂画面，右下角有 Stanford 水印），它直接追问了剪枝与 AWQ 共同依赖的前提。学生问 `how do you distinguish the important layers from the unimportant layers?`，讲师答：准备一个 `calibration set`，把输入过一遍模型，`looking at the magnitude of the activations`，接近零的 `dead units` 可剪、幅度大的要保留（70:04–70:28）。学生继续质疑「如果激活一直很高呢？会不会只是训练 artifact」，以及「假设某个神经元对所有 prompt 恒为 100，它一定有意义吗」（70:30–71:16）。讲师的回答有两层：其一，这确实是 `empirical observation`，`some of the channels will be much higher than others`，`If this weren't true, then these techniques wouldn't necessarily work`，是训练结果的副产品，能被利用而已；其二，对「恒为 100」的神经元，`you can't just like remove it, because then everything's going to be broken`，但可以看 `variance`——若 `high mean and low variance`，`maybe there's another way to just incorporate the bias, essentially`，即把它并入 bias 而不是删掉。这里的局限非常具体：重要性判据是启发式的、依赖 calibration set 的分布，且「高激活 = 重要」只是经验规律而非理论保证；一个恒定的高幅值单元可能携带的是可被 bias 吸收的常量信息，盲目按幅值排序既可能误删、也可能误留。这也预告了本章之后必须换思路：既然有损方法总有这类说不清的精度风险，下一章的投机解码就干脆绕开「改模型」，转而追求数学上可证明的无损加速。

![NVIDIA 剪枝三步 Algorithm（1024 samples 校准集）与含 Minitron、Nemotron、Gemma、LLaMa-3 的结果散点图及 40x cheaper 9% better 标注](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0050_01-07-56.jpg)

## 投机解码：用「验证快于生成」打破逐 token 瓶颈 [71:43 - 74:00]

71:43 出现了一个明确转折。讲师先回望：`So far we've looked at lossy methods, which basically kind of really crunch down the KV cache, but it could hurt accuracy`，紧接着给出定位 `There's a very elegant way of doing this in a lossless way. This is called speculative sampling or speculative decoding`（71:47–71:57）。也就是说，前面三章的量化、剪枝、乃至更早的 GQA/MLA 都在「有损地压缩 KV cache」，而投机采样（speculative sampling，也叫 speculative decoding）是全课第一个承诺「无损（lossless）」的加速方法。这个转折的重要性在于：它不再试图缩小模型或 KV cache，而是利用推理两个阶段在硬件行为上的不对称性，把串行生成改造成并行验证，从而在不改动输出分布的前提下提速。理解它需要先回到全课反复出现的 prefill/generation 二分。

画面 `kf_0053` 上 `def speculative_sampling():` 的第一段是 `Recall the two stages of inference`，两条 bullet 为：`Prefill: given a sequence, encode tokens in parallel (compute-bound) [note: also gives you probabilities]` 与 `Generation: generate one token at a time (memory-bound)`，随后一句结论 `In other words, checking is faster than generation`。转写 72:02–72:26 把它讲得更口语：`if you're doing prefill, you can code all the tokens in parallel, and this also gives you probabilities. This is fast, this is compute-bound`，而 generation `it's one at a time`；所以 `checking is faster than generation. If I give you a sequence, it's fast to tell me how good it is, much faster than it is to generate one at a time`。那条 `[note: also gives you probabilities]` 是全课前面没有特意强调、但投机解码赖以成立的细节：一次前向不仅能编码整段序列，还顺带给出每个位置的概率分布，正好可以用来「给别人的猜测打分」。黄色高亮行给出方法与两篇几乎同时出现的文献：`Speculative sampling [Leviathan+ 2022][Chen+ 2023]`，转写 73:55–74:00 也说明 `There's a few papers that came out with speculative decoding around the same time. This is, you know, one of them`。

具体分工在 `kf_0053` 的淡灰 bullet 与转写 72:30–73:11 中给出：`Use a cheaper draft model p to guess a few tokens (e.g., 4)`，`Evaluate with target model q (process tokens in parallel), and accept if it looks good`。注意画面用小写 $p$ 表示 draft model、小写 $q$ 表示 target model，而转写 72:41–72:44 记作大写 $P$（draft）与 $Q$（target）——两者指同一对模型，只是记号大小写不一致，阅读论文截图与听课时不要混淆。讲师强调 `things are kind of choosing to be balanced`（72:50）：draft model `is smaller and cheaper`，虽然它也得 `generate one at a time`、受 memory-bound 限制，但代价不大；target model `is big and expensive`，但被要求做的是 `process a batch of tokens in parallel`（相当于一次 prefill），`so it won't also be too bad`。转写 73:11–73:44 描述课件动画：`if you use a big model token by token, that's going to be pretty slow`，而投机解码下 `the small model is generating a bunch of tokens`、`the large model basically critiquing them`，于是输出呈现 `a burst of tokens, and then maybe another burst of tokens, and so on and so forth`——一阵一阵的突发式吐字。`kf_0053` 上那两个蓝色链接 `[Speculative sampling video]`、`[article]` 正是这个演示动画与论文的入口。用代价式写这个平衡：设 draft 单步代价为 $c_p$、target 并行验证 $K$ 个 token 的代价为 $c_q(K)$（近似与单步同量级），则一轮生成 $K$ 个 token 的开销约为 $K c_p + c_q(K)$，而不是朴素自回归的 $K c_q(1)$；只要 $c_p \ll c_q$ 且接受率够高，就有净收益。

同一张 `kf_0053` 下半部已经露出论文 `Algorithm 2 Speculative Sampling (SpS) with Auto-Regressive Target and Draft Models` 的前几行，值得先记住符号，下一章会逐行展开：`Given lookahead K and minimum target sequence length T.`；`Given auto-regressive target model q(·|·), and auto-regressive draft model p(·|·), initial prompt sequence x₀,...,xₜ.`；`Initialise n ← t.`；`while n < T do`；`for t = 1 : K do`；`Sample draft auto-regressively x̃ₜ ~ p(xₜ | x₁,...,xₙ, x̃₁,...,x̃ₜ₋₁)`；`In parallel, compute K + 1 sets of logits from drafts x̃₁,...,x̃_K`。其中 $K$ 是 lookahead（一次猜几个 token），$T$ 是 minimum target sequence length（本轮至少生成到多长），$K+1$ 组 logits 是 target 一次并行算出的打分。这里有三个常见误解需要提前澄清。第一，投机解码不是「用小模型替代大模型」：target model 全程参与验证，最终分布严格等于 target，draft 只负责提议。第二，它不是近似采样或 beam search 式的启发式，转写 75:14–75:20 会强调它 `always guaranteed to get an exact sample from the target model`。第三，它的加速不是无条件的：收益取决于 draft 与 target 的对齐程度（决定接受率）以及 $K$ 的取值，而且 draft 自身仍要自回归串行跑 $K$ 步，这部分延迟省不掉。正因如此，转写 76:04–76:23 才会回到「希望 draft 尽量接近 target，最好蒸馏出来」这一与前面剪枝/蒸馏相连的话题——无损的是输出分布，不是工程代价。

## 投机采样算法、无损性证明与加速甜点 [74:00 - 76:55]

画面 `kf_0054` 把上一章只露出开头的 `Algorithm 2 Speculative Sampling (SpS) with Auto-Regressive Target and Draft Models` 完整呈现出来，这是整段信息密度最高的一张图，转写 74:04–75:04 只是它的口语版。逐行转述为（数学符号与下标按画面原样保留）：

```text
Algorithm 2  Speculative Sampling (SpS) with Auto-Regressive Target and Draft Models
Given lookahead K and minimum target sequence length T.
Given auto-regressive target model q(·|·), and auto-regressive draft model p(·|·), initial prompt sequence x₀,...,xₜ.
Initialise n ← t.
while n < T do
    for t = 1 : K do
        Sample draft auto-regressively  x̃ₜ ~ p(xₜ | x₁,...,xₙ, x̃₁,...,x̃ₜ₋₁)
    end for
    In parallel, compute K + 1 sets of logits from drafts x̃₁,...,x̃_K :
        q(x₁|x₁,...,xₙ), q(x₁|x₁,...,xₙ,x̃₁), ..., q(x₁|x₁,...,xₙ,x̃₁,...,x̃_K)
    for t = 1 : K do
        Sample r ~ U[0, 1] from a uniform distribution.
        if r < min(1, q(x|x₁,...,x_{n+t-1}) / p(x|x₁,...,x_{n+t-1}))  then
            Set x_{n+t} ← x̃ₜ and n ← n + 1.
        else
            sample x_{n+t} ~ (q(x|x₁,...,x_{n+t-1}) - p(x|x₁,...,x_{n+t-1}))₊ and exit for loop.
        end if
    end for
    If all tokens x_{n+1},...,x_{n+K} are accepted, sample extra token x_{n+K+1} ~ q(x|x₁,...,xₙ,x_{n+K}) and set n ← n + 1.
```

流程分三段：先用 draft 自回归采出 $K$ 个候选 $\tilde{x}_1,\ldots,\tilde{x}_K$；再由 target **一次并行**算出 $K+1$ 组 logits（对应前缀逐步加长的各个位置，这正是「checking is faster than generation」的落地）；最后逐个决定接受或拒绝。转写 74:09–74:45 的说法是 `we're going to generate K tokens, and we're just going to sample from this draft model. So, sorry, P is a draft model. We're going to sample K tokens, and then in parallel, we're going to compute the logits of these draft tokens using Q`，然后 `you basically are going to accept with probability min one over this ratio of Q over P`。注意最后一行：若 $K$ 个候选全被接受，还会从 $q$ 额外采一个 token，因此**一轮最多前进 $K+1$ 个 token**，这是很多人算加速比时漏掉的一项。

接受概率写成

$$
\alpha = \min\!\left(1,\ \frac{q(x\mid x_1,\ldots,x_{n+t-1})}{p(x\mid x_1,\ldots,x_{n+t-1})}\right),\qquad r\sim U[0,1],\ \ r<\alpha \Rightarrow \text{accept}.
$$

直觉由转写 74:45–74:54 给出：`if Q is much larger than P, the larger the Q is, the more likely we want to accept it`——target 认为这个 token 概率不低于 draft（$q\ge p$）时必然接受；$q<p$ 时按 $q/p$ 的比例概率接受。关键在于拒绝时并不「白跑一趟」：伪代码里 `sample x_{n+t} ~ (q - p)₊ and exit for loop`，即从残差分布 $(q-p)_+ \equiv \max(0,\,q-p)$（归一化后）采出一个 token 并结束本轮。`kf_0055` 顶部把这一步记为 `Residual probabilities max(q-p, 0): [0, 1]`。转写 75:04–75:20 点明它与普通拒绝采样的区别：`this is basically rejection sampling except for rejection sampling sometimes you just when you reject you get nothing, but if here we always are guaranteed to get an exact sample from the target model`。这是「无损」二字的机制来源——被拒的位置由残差分布补上一个精确样本，而不是丢弃重来。

此外 `kf_0055` 上半部还给出了一个两 token 情形的无损性代数验证，转写 75:20–75:35 说 `I'm going to skip this kind of simple proof … basically the same arguments as rejection sampling`，但画面把推导写全了：`Compute the probabilities of speculatively sampling a token:` 下两条 bullet 为

$$
\text{P[sampling A]} = p(A) * \frac{q(A)}{p(A)} + p(B) * 1 * 0 = q(A),
$$

$$
\text{P[sampling B]} = p(B) * 1 + p(A) * \left(1 - \frac{q(A)}{p(A)}\right) * 1 = q(B).
$$

第一式把「draft 抽到 A 且被接受」与「draft 抽到 B 却经残差判给 A」两项相加，第二式同理，结果恰好都等于 target 的概率 $q(\cdot)$。这个恒等式说明：无论 draft 分布 $p$ 多差，只要严格按 $\min(1,q/p)$ 接受、按 $(q-p)_+$ 补样，最终样本就精确服从 $q$。这也立刻给出一个重要局限：无损性是有条件的数学结论，一旦为了提速改动接受规则、截断残差分布或让 draft 打分不准，输出就退化为近似采样，`exact sample` 的保证立即失效。

加速效果由 `kf_0055` 的 Table 1 量化，表题为 `Chinchilla performance and speed on XSum and HumanEval with naive and speculative sampling at batch size 1 and K = 4. XSum was executed with nucleus parameter p = 0.8, and HumanEval with p = 0.95 and temperature 0.8.` 表头为 `Sampling Method / Benchmark / Result / Mean Token Time / Speed Up`，六行数值原样如下：

| Sampling Method | Benchmark | Result | Mean Token Time | Speed Up |
|---|---|---|---|---|
| ArS (Nucleus) | XSum (ROUGE-2) | 0.112 | 14.1ms/Token | 1× |
| SpS (Nucleus) | XSum (ROUGE-2) | 0.114 | 7.52ms/Token | 1.92× |
| ArS (Greedy) | XSum (ROUGE-2) | 0.157 | 14.1ms/Token | 1× |
| SpS (Greedy) | XSum (ROUGE-2) | 0.156 | 7.00ms/Token | 2.01× |
| ArS (Nucleus) | HumanEval (100 Shot) | 45.1% | 14.1ms/Token | 1× |
| SpS (Nucleus) | HumanEval (100 Shot) | 47.0% | 5.73ms/Token | 2.46× |

其中 ArS 是朴素自回归采样（autoregressive sampling），SpS 是投机采样。要读出两层信息：一是 `Result` 列几乎不动（$0.112\to0.114$、$0.157\to0.156$、$45.1\%\to47.0\%$），差异来自采样随机性而非系统性退化，实证印证无损；二是 `Mean Token Time` 从统一的 $14.1\text{ms/Token}$ 降到 $7.52$、$7.00$、$5.73\text{ms/Token}$，对应 `Speed Up` 为 $1.92\times$、$2.01\times$、$2.46\times$。实验条件也要一并记住：模型是 Chinchilla、`batch size 1`、`K = 4`，解码参数 `nucleus p = 0.8`（XSum）与 `p = 0.95, temperature 0.8`（HumanEval）。表格下方淡灰还有三张小图标题 `Mean Sampling Time (128 tokens)`、`Acceptance rate`、`Total loop time`（含 Human Eval / XSum 曲线），说明原始论文还分析了接受率与总循环时间随 $K$ 的变化。

最后是 $K$ 的甜点与 draft 的设计，转写 75:39–76:55 给了明确结论：`if you have too few draft tokens, you're not really leveraging the batching on the target model side. And if you have too many, then you're going to reject more often. So, there's a sweet spot around you know in this case three or four`——太少摊薄不了 target 端并行验证，太多则拒绝率上升，经验甜点约 $K=3\sim4$，与 Table 1 用的 $K=4$ 一致。`in general, the draft model is much smaller than the target model. And ideally, you want the draft model to be as close to a target, which means that you want to kind of distill it`（76:04–76:17），于是 `a lot of the same ideas that we just talked about are applicable to speculative decoding as well`：把前面通过减少 KV cache 得到的各种小模型，若精度满意就 `just serve that`，若不满意 `then it at least can be a draft model, and you can use your main model to fix things up`（76:24–76:43）。`kf_0056` 顶部残留的 Medusa 示意图（标注 `Medusa Head2`、`t₂ t₃ t₄`、`f₁ f₂ f₃`、`Embedding layer & Auto-regression Head`、`f₄`、`t₅`）提示 draft 不一定要独立小模型，也可以用多预测头；同页还给出投机解码的 `Summary` 三条：`Exact sampling from target model (thanks to math)!`、`Exploits asymmetry between checking and generation`、`Lots of room for innovation on the draft model (involves training)`。局限也在这里：转写 76:43–76:53 明确说 `there's a bunch of whole literature on speculative decoding right now that improve over the original which I'll kind of skip`，即课上只讲了原始版本，Medusa、多头预测、树状验证等后续改进都没展开，别把这一章当成该领域的全貌。

![Algorithm 2 Speculative Sampling 完整伪代码：K 个 draft token、K+1 组 logits、min(1,q/p) 接受与 (q−p)₊ 残差采样](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0054_01-14-02.jpg)

![投机采样无损性推导 P[sampling A]=q(A)、P[sampling B]=q(B) 与 Chinchilla 在 XSum/HumanEval 上 K=4 的加速比表格](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0055_01-15-41.jpg)

## 动态工作负载：连续批处理与选择性批处理 [76:55 - 79:34]

76:55 讲师用 `dynamic workloads` 把视角从「单条序列怎么更快」整体切换到「一堆不规则请求怎么一起服务」：`this is the use case is you're serving a live website and come users come and chat with your model. The requests arrive at different times. They have different shared prefixes and they have different lengths. So it's kind of pretty messy. It's far from this kind of very simple training where you have these blocks of the same number of tokens all at once`（77:00–77:29）。`kf_0056` 上被黄色高亮的函数头正是 `def continuous_batching():`，其 `Problem` 两条写得更精确：`Training: get a dense block of tokens (batch size x sequence length)`；`Inference: requests arrive and finish at different times, so you have a ragged array`。画面上两幅网格图用 $S_1\ldots S_4$ 标序列、$T_1\ldots T_8$ 标时间步，黄色/蓝色格子表示各序列在不同时刻的 token，红色 `EOD` 标记序列结束；右图可见某些序列提前 `EOD` 后，空出的位置被新到达的请求填上，形状参差不齐（jagged）。`kf_0056` 还给出 Orca 的完整引用 `[Orca: A Distributed Serving System for Transformer-Based Generative Models][talk]`，而转写 77:32–77:41 只说 `there's this idea this is system called Orca that was built. This is actually very early on which introduces idea of continuous batching`。此外这一页顶部还留着投机解码的 `Summary` 与 Medusa 头部示意，说明讲义是连续滚动的长页面——读到系统章节时，上一章的结论仍在画面上方，正是「先有单序列优化、再谈多请求调度」这一顺序的视觉印证。

连续批处理（continuous batching）的做法由转写 77:45–78:29 与 `kf_0057` 的 `Solution: iteration-level scheduling` 共同给出。转写描述：`here's a prefix of the first request and you're generating this token. Here's the second one … It's jagged because every prefix has a different length. And what we're going to do is we're going to decode step by step. So every step you decode one token for all the sequences. Next step you decode another token for all the sequences. And then if you end you just eject that sequence. And then as new requests arrive to the batch then you basically put it in the batch`。`kf_0057` 把机制凝成两条 bullet：`Decode step by step`；`Add new requests to the batch as they arrive (so don't have to wait until generation completes)`。之所以叫 `continuous batching`，转写 78:22–78:29 解释得很直白：`your batch is dynamically being updated with either old finished sequences being evicted and new ones coming in`。这里有个易错点：它与静态 batching 的差别不在「是否批处理」，而在**调度粒度**——静态批处理要等整批里最慢的序列生成完才能释放整块资源，短序列提前结束后占着的槽位一直空转；iteration-level scheduling 则在**每个 decode step** 都重新决定谁在 batch 里，短序列一结束立即驱逐、新请求立即补位，GPU 几乎不空转。

一旦 batch 里序列长度不同，就撞上第二个问题，`kf_0057` 的 `Problem` 两条 bullet 为 `Batching only works when all sequences have the same dimensionality (right?)` 与 `But each request might have a different length`，对应转写 78:34–78:47：`everything we've seen batching works when all the sequences have the same dimensionality, right? You have tensors, every slice has the same dimensionality. But if each request here has a different length, so what do you do about that?` 解法是 `Solution: selective batching`，画面四条 bullet 逐字为：`Training: when all sequences of the same length, operate on a B x S x H tensor`；`But we might have different lengths: [3, H], [9, H], [5, H], etc.`；黄色高亮的 `Attention computation: process each sequence separately`；以及淡灰的 `Non-attention computation: concatenate all the sequences together to [3 + 9 + 5, H]`。转写 79:01–79:31 举的例子正是这几个数：`let's say you have a three length three, length nine, and length five`，并说明原因：`in the attention computation, you can't really do anything about this because attention sort of depends on the length of your sequence. So if you have a 3x3 computation and 9x9 computation, you can't really share the tensor effectively. But for the non-attention, the MLP layers, which is, you know, takes up a lot of flops, you can actually just concatenate all the sequences together to form a mega sequence and you process that`。用式子把拼接写清楚：

$$
\text{MLP 输入} = \text{concat}\big([3,H],\ [9,H],\ [5,H]\big) = [3+9+5,\ H] = [17,\ H].
$$

也就是说：attention 部分因为每个序列有自己独立的 KV cache、计算形状随长度变化（$3\times3$ 与 $9\times9$），只能逐序列单独算；而 MLP 等非 attention 层对每个 token 的操作与序列来源无关，可以把所有序列的 token 摊平拼成一条 `[17, H]` 的 mega-sequence，一次大矩阵乘吃掉，从而把这部分 FLOPs 重新变成高算术强度。`kf_0060` 顶部残留的 `3. Sequences have different lengths (padding is inefficient)` 正是这一章在总结页里的浓缩句。

本章与前后内容的关系是：第 5 章的投机解码、第 1–3 章的量化与剪枝，优化的都是「一条序列内部」的成本，而连续批处理与选择性批处理优化的是「请求之间」的资源复用，属于系统层，与模型精度完全无关——这也是它们能和有损方法自由叠加的原因。局限有两条很具体。其一，selective batching 只对 non-attention 部分有效，attention 仍须逐序列单独处理，因此长度越参差、attention 占比越高，收益越有限；它并没有解决「每个序列一份 KV cache」这个内存管理问题，这正是下一章 PagedAttention 要接手的。其二，ragged array 若用 padding 强行对齐会浪费算力与显存（总结页的 `padding is inefficient`），而拼接 mega-sequence 又要求 kernel 能处理不规则边界；转写 83:27–83:32 那句 `there's a bunch of other optimizations like kernels that I'm not going to have time to go over` 暗示这些工程细节课上未展开，别以为有了调度策略就万事大吉。

![连续批处理与选择性批处理幻灯片：ragged 网格图、iteration-level scheduling、B x S x H 张量与 [3+9+5, H] mega-sequence 拼接](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0057_01-19-15.jpg)

## PagedAttention：分页、碎片与前缀共享 [79:34 - 83:46]

79:34 引出本段最后一个技术点：`final idea is page attention. This was introduced in the VLM paper. Of course, VLM has many other bells and whistles now, but this is kind of the core idea at that time`（79:34–79:46）。这里的 `VLM` 是转写听写误差，应为 **vLLM**——`kf_0059` 底部淡灰写着 `Other vLLM optimizations:`，可以互相印证。核心追问是 `the question is how is the KV cache stored? Right? So, if you think about requests coming in, you have to put them in memory somewhere`（79:49–80:04）。`kf_0058` 顶部 bullet 给出朴素做法 `Allocate section of KV cache for prompt and response (up to a max length)`，并把一条连续内存画成格子：请求 A 的 prompt 是 `Four score and seven years ago our fathers brought forth <eos>`，请求 B 的 prompt 是 `You only live once <eos>`，格子上标注 `1 slot for generated token`、`2 slots future used (reserved)`、`7 KV cache states for request A's prompt`、`3 KV cache states for request B's prompt`、`Request A/B current iteration`，以及红色的 `2038 slots never used (internal fragmentation)`、`507 slots never used (internal fragmentation)` 和 `External fragmentation`。这些具体数字是转写里没有的：转写 80:19–80:50 只举例说 `you might have a max token limit of like 1024. So you have to allocate all this memory`，画面则直接标出「预留了却用不上的 2038 / 507 个槽」。页面的 `Problem: fragmentation (what happens to your hard drive)` 下两条 bullet 为 `But this is wasteful since we might generate much fewer tokens (internal fragmentation)!` 与 `Might be extra unused space between sections (external fragmentation)!`，转写把两类碎片解释得很清楚：internal fragmentation 来自「不知道序列何时停，只能按最大 token 数预留整段」，external fragmentation 来自「不同请求之间的空隙太小、无法有效利用」，并类比 `what happens to your hard drive … you have to defrag your hard drive back in the day`。

解法被黄色高亮为 `Solution: PagedAttention (remember operating systems)`，紧接着淡灰一行 `Divide the KV cache of a sequence into non-contiguous blocks`。转写 80:52–81:04 点破思路来源：`these are systems people so they know their operating systems. They said, "Okay, well, we solved this problem once before, so let's just use the same idea here."` 具体做法：把一个序列的 KV cache 切成固定大小、物理上不必连续的块，用索引表记录每块位置即可。转写 81:04–81:25 用同一句林肯演说举例：`if you have this sequence, "Four score and seven years ago our fathers brought forth." We're just going to chunk it up into these blocks of size four. Doesn't matter where the blocks go, but they're going to be kind of aligned according to the blocks`，与 `kf_0058` 示意图里 `Query vector: forth` 通过指针指向 `Block 1 (years, ago, our, fathers)`、`Block 2 (brought, forth)`（表头 `Key and value vectors`）完全对应——这正是操作系统「逻辑页 → 物理页框」的页表映射。转写 81:44–81:54 强调 `they're in kind of interspersed, but as long as you have the indices and keep track of where everything is, that's fine`：`block of size four` 带来统一的块粒度，内部碎片被限制在最后一个块内，块间空隙也能被别的请求复用。

块可寻址之后，最有价值的副作用是前缀共享。`kf_0059` 顶部两条 bullet 为 `Sharing the system prompt` 与 `Sampling multiple responses per prompt (e.g., for program synthesis)`，结论行是 `Solution: share prefixes, copy-on-write at the block level`。图示里 Sample A1 与 Sample A2 的 `Logical KV blocks` 都从 `Block 0 = Four score and seven`、`Block 1 = years ago our …` 开始，但 A1 的第 8 个 token 是 `fathers`、A2 是 `mothers`；中间的 `Physical KV blocks` 表画出 `Block 0` 到 `Block 8`，共享块上标着红色 `Ref count: 2 → 1`，分叉处写着红色 `Copy-on-write`。转写 81:27–82:39 分两层讲收益：其一，`if you have system prompts then you can actually just cache these system prompts the KV cache in the system prompts once. Right? And that can be useful for all the queries`——`if a lot of people are using the same system prompt, then you don't have to compute the KV cache for every request`；其二，`there are many applications where you have the same prompt and you actually want to generate multiple responses. So in that case, you can also just share the KV cache for all the for the prompt and just have unique responses coming out`。转写 82:42–83:26 用 `four score and seven years ago are blank` 的例子解释 copy on write semantics：`you would start by having years ago or our. And then … we have these two samples and if they had happened to sample the same token, you just continue with that. But if they sample different tokens, you split the block. And then you can kind of continue there. So you're basically sharing as much of the prefix cache as you can`。用式子概括：`Ref count` 从 $2$ 降到 $1$ 的那一刻，才把被写的块复制一份给分叉的样本，其余共享块维持一份物理存储。

画面 `kf_0059` 底部淡灰还补上了转写 83:27–83:32 一句话带过的内容：`Other vLLM optimizations: Kernel to fuse block read and attention (reduce kernel launch overhead)`、`Use latest kernels (FlashAttention, FlashDecoding)`。这些名字是画面独有的增量。转写 83:36–83:46 用一句话收束整个系统章节：`but the general idea is that you're using kind of these operating systems and metaphors to manage your inference`——paging（分页）、copy-on-write（写时复制）、speculative execution（投机执行）这些 OS 概念被整体搬进推理服务器。局限也要点明：分页并非零成本，它引入块表维护与间接寻址，且块大小是超参——块太大则最后一个块的内部碎片回升，块太小则块表膨胀、`kernel launch overhead` 上升（这正是需要 kernel fusion 的原因）；此外 copy-on-write 的收益完全建立在「存在共享前缀」之上，若每个请求的 prompt 都不同、也没有公共 system prompt，前缀共享几乎不产生节省，PagedAttention 剩下的就只是碎片治理这一项收益。另一个易错点是把 `vLLM` 与 `VLM`（视觉语言模型）混为一谈：本段讲的是推理服务系统 vLLM 引入的 PagedAttention，与多模态模型无关。

![连续分配 KV cache 造成的 2038/507 slots 内部碎片与外部碎片，及 PagedAttention 按块映射示意图](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0058_01-21-02.jpg)

## 全课总结与新架构展望 [83:46 - 85:29]

画面 `kf_0060` 是本段的总结页，其 `Summary` 四条 bullet 逐字为：`Inference is important (actual use, evaluation, reinforcement learning)`；`Different characteristics compared to training (memory-bound, dynamic)`；`Techniques: new architectures, quantization, pruning/distillation, speculative sampling`；`Ideas from systems (speculative execution, paging)`；黄色高亮行是 `New architectures have huge potential for improvement`。转写 83:46–84:06 的口语版是：`Inference is really really important. It's very different from training even though it's the same model, but you're asking the model to do something very different. Ends up being very memory bound and it's also dynamic if you're kind of in a live chatbot use case`。这两句把全课最根本的判断又强调了一遍：同一个模型，训练时序列维可并行、是 compute-bound 的稠密张量运算；推理时自回归逐 token 生成、`generation` 阶段算术强度低于 1，被 HBM 带宽卡住；到了线上服务还要叠加请求到达时间不定、长度不一的 `dynamic` 特性。`kf_0060` 下方 `def landscape():` 还重现了全课开头的推理场景清单：`Inference shows up in many places: Actual use (chatbots, code completion, agents, batch data processing)`、`Model evaluation (e.g., on instruction following)`、`Reinforcement learning (sample many generations, then apply score)`；页面顶部残留的 `3. Sequences have different lengths (padding is inefficient)`、`continuous_batching()`、`paged_attention()` 则正好标出第 6、7 章两个函数在讲义大纲里的位置，可当作本段章节顺序的核对表。

84:06–84:37 讲师把本段（乃至全课）所有技术串到同一条原则下：`We saw a variety of different techniques to improve inference. You can quantize, you can come up with new architectures, you can prune and distill. You can also use speculative um you know sampling. But all of these are driven by the same you know principle here, which is reduce your KV cache, but don't hurt accuracy too much`，随后补上系统侧 `there's ideas from systems like paging and speculative execution that can be brought to bear for actually live kind of inference servers`。把这十来分钟的内容对号入座：量化减的是每个数的字节数（BF16 2 字节 → INT4 0.5 字节），剪枝与蒸馏减的是参数量（15B → 8B），GQA/MLA/局部注意力减的是 KV cache 的元素个数，投机解码用 target 的并行验证换掉串行生成（$K=4$ 时加速 $1.92\times\sim2.46\times$），连续批处理与 PagedAttention 则在不碰模型的前提下把不规则请求的内存与算力管起来。用一个带约束的式子写这条统一原则：

$$
\min_{\text{方法}} \ \text{KV cache 大小（或内存流量）} \qquad \text{s.t.} \qquad \text{accuracy 损失} \le \epsilon .
$$

分子是「几乎所有方法共同优化的量」，分母的 $\epsilon$ 则解释了为什么方法分成两派：量化、剪枝、新架构都是在 $\epsilon>0$ 下换取效率，只有投机解码靠 $\min(1,q/p)$ 与 $(q-p)_+$ 把 $\epsilon$ 严格压到 $0$。这是回顾本段时最有用的一根线索。

84:40–85:15 是本段真正的展望，也是黄色高亮 `New architectures have huge potential for improvement` 的落点。讲师先坦白 `One thing we didn't really get a chance to talk about, which was discussed briefly, is that I think that new architectures have actually a huge potential for improvement`，点名 `state space models or linear attention or diffusion`——这正好呼应本段开头 64:13–64:22 被一笔带过的 `local attention`、`linear attention` 与 `a non-autoregressive way to generate, which can be much faster` 的 diffusion models，首尾闭环。随后给出理由，措辞相当根本：`at some level, the KV cache and the way that attention is is built fundamentally makes it an inference-unfriendly archi- uh kind of architecture. So, if you can come up with a new architecture that is sort of designed for inference in the way that the transformer was not, this can maybe unlock a lot`。也就是说，attention 需要一份随序列长度线性增长的 KV cache，而 generation 又恰恰是 memory-bound 的，这两件事天生冲突；前面所有技巧都是在给这个「对推理不友好」的架构打补丁，真正的设计空间在于换一种天生适合推理的架构（把历史压成固定大小状态的 SSM/线性注意力，或非自回归的 diffusion）。转写 85:15–85:21 以下节课预告结束：`I will stop there. And next class uh Tats will return and talk about uh scaling laws part two`。

最后补两条阅读时的易错点与局限。第一，本段所有加速结论都建立在「generation 是 memory-bound」这一硬件前提上；一旦加速器的算力/带宽比变化，或换用非自回归架构，量化、批处理、投机的收益排序都会重写，不能把 $1.92\times\sim2.46\times$ 这类数字当成普适常数。第二，课上给出的精度与速度结论都绑定了具体模型与评测：Chinchilla 配 XSum (ROUGE-2) / HumanEval (100 Shot)、Llama 2 13B 配延迟吞吐实例、Nemotron-4 15B / Minitron 8B/4B 配 MMLU 类得分；像 GQA 精度那样，讲师在前半段已提醒过不同论文在不同 benchmark 上结论可能相反。因此判断这些经验数字能否迁移到自己场景，依据仍应回到全课建立的分析框架——算术强度、prefill/generation 两阶段、以及 $\text{KV cache bytes}=B\cdot S\cdot L\cdot K\cdot H\cdot 2\cdot 2$ 这类可重算的公式，而不是记住某一个 `Speed Up` 数值。

![全课 Summary 幻灯片：memory-bound 与 dynamic 特性、各技术清单、系统思想，以及高亮的 New architectures have huge potential for improvement](/coursenotes/cs336-language-modeling-sp26/l10-inference/kf_0060_01-25-04.jpg)

## 要点回顾

### 推理为何重要与「快」的多种度量
- 训练是 one-time cost、推理是 repeated cost：`OpenAI processes ~8.6T tokens per day` 对照 `DeepSeek v4 was trained on 32T tokens`，相除约 3.72 天，不到 4 天生成量就反超一次训练；推理嵌在 actual use、model evaluation、reinforcement learning 三处，只占一讲却「重要性在不断增长」，效率正是本课程主线。
- 从 chatbot 的 `humans are bottleneck` 转到 agent 的 internal trace 后，`Tokens generated = compute spent`（N_trace ≫ N_out），所以 twice as fast 乃至 10% faster 都是 big deal；产业栈有 vLLM（PagedAttention）、SGLang（RadixAttention）、TensorRT-LLM、llama.cpp。
- 「快」分三个指标：TTFT 是首 token 前的空窗，Latency 是 seconds/token，Throughput 是 tokens/second（如处理 a petabyte of data），两者仅一阶互为倒数；小 batch 利 latency/TTFT、大 batch 利 throughput，彼此存在 trade-off。
- 训练可用 `Y = einsum("btd,df->btf", X, W)` 沿序列维并行，自回归推理只能 `y_t = einsum("bd,df->bf", x_t, W)` 的 T=1 瘦矩阵乘，因而更难获得高算术强度。

### 记号体系与算术强度门槛
- scaling book 记号 B/T/D/H 既表轴也表长度，示例 `BTD × DH → BTH`；红色 contracting 维在两个操作数出现且从结果消失，蓝色 batching 维出现两次却保留，唯一判据是是否出现在结果里。
- 符号表含 B/L/T/S/V/D/F/H/N/K/G 共 11 行，电路图的 Conventions 为 `F = 4D`、`D = N H`、`N = K·G`、`S = T`；训练 S=T，推理 generation 时 T=1、S 是上下文长度；现场问答把 K 修正为组数、G 为每组 query 头数。
- 按 BF16（2 bytes for bf16）记账：`flops = 2*B*D*F`、`bytes_transferred = 2*B*D + 2*D*F + 2*B*F`，`intensity = 2BDF/(2BD+2DF+2BF)` 在 B≪D,F 极限下约简为 `intensity ≈ B`。
- H100 的 `989e12 / 3.35e12 ≈ 295.2239 ≈ 295`，判据 `compute-bound iff B > 295`；B=1 时读整个 D×F 矩阵只做 2DF FLOPs，正是推理的 memory-bound 形态；Naive sampling 每步重喂历史，`generating T tokens requires O(T^3) FLOPs (one feedforward pass is O(T^2))`。

### KV cache 与 prefill / generation 两阶段
- 因果掩码保证前缀激活不随追加 token 改变，`Solution: store KV cache in HBM`；每 sequence(B)、token(S)、layer(L)、head(K) 存 H 维向量，`kv_cache_size_per_seq = S*(K*H)*L*2*2`；推理拆成 Prefill（T=S，Compute bound）与 Generation（T=1，memory bandwidth bound），prefill 因一次看到整个 prompt 而可 parallelizable just like in training。
- MLP 层 `flops == 6*B*T*D*F`、`bytes == 4*B*T*D + 4*B*T*F + 6*D*F`，`intensity = 3*B*D*F*T/(2*B*D*T + 2*B*F*T + 3*D*F) ≈ B*T`；prefill 靠拉大 B*T 越过 295，generation 塌缩为 B（并发请求数，外生不可控）。
- Attention 层 `flops == 4*B*S*T*D`、`bytes == 4*B*S*D + 4*B*T*D`，`intensity = S*T/(S+T)`，prefill 得 `S/2`、generation 得 `S/(S+1) < 1`；`Unlike MLPs, no dependence on B, so batching doesn't help`，蓝色 B 是根因，标准 Transformer 下这格 `<1` 无解。
- Summary：`Prefill is compute-bound, generation is memory-bound`；既然内存受限，`latency ≈ bytes to transfer / memory bandwidth`，前提是 `can overlap compute and communication perfectly and ignore overhead`。

### Llama 2 13B 实例与 batch 的取舍
- config 为 S=1024、D=5120、F=13824、N=40、K=40（无 GQA）、H=128、L=40、V=32000、`memory_bandwidth = 3.350e12` bytes/s，其中 D = N·H = 40×128 = 5120 自洽。
- `num_params = 2*V*D + D*F*3*L + (2*D*N*H + 2*D*K*H)*L = 13,015,449,600`，`memory = B*kv_cache_size_per_seq + 2*num_params`，`latency = memory/memory_bandwidth`、`throughput = B/latency`。
- 化简得 `memory = 838860800*B + 26030899200`、`latency ≈ 0.0002504*B + 0.0077704`、`throughput = 127792.35838438*B/(32*B + 993)`；B=1 时 latency 0.0080 s/token、throughput 124.6755 tokens/s。
- b1/b64/b256 的 latency 0.0080→0.0238→0.0719、throughput 124.6755→2689.4807→3561.7685，`h100_memory = 80e9` 下 `assert b256.memory > h100_memory` 直接 OOM、增益从约 21 倍跌到约 1.3 倍；机制是 `latency = aB+b`、`throughput ∝ B/(B+c)`，TTFT 本质是 prefill time，故 prefill 用小 batch、generation 用大 batch。

### 缩小 KV cache 的有损架构
- `Taking shortcuts (lossy)` 下并列 `reduce_kv_cache_size()`、`quantization()`、`model_pruning()`，纲领 `reduce inference complexity without hurting accuracy`；`KV cache bytes = B·S·L·K·H·2·2`，大 batch 下 KV cache 甚至可比参数还大。
- GQA 用 N 个 query heads 配 K 个 KV heads，`GQA reduces the KV cache by a factor of N/K`（MHA K=N、MQA K=1）；Llama 2 13B 从 K=40,B=64 改 K=8,B=64，memory 79,717,990,400→33,412,874,240、latency 0.0238→0.0100、throughput 2689.4807→6416.6883，说明减内存能同时改善两者，张力只来自 batch 维度。
- 精度须单独校验：Ainslie 2023 中 GQA-8-XXL 的 0.28s / 47.1 几乎追平 MHA-XXL 的 1.51s / 47.2（约 5.4 倍加速），DeepSeek Table 8 却是 MHA 在 MMLU 45.2 对 GQA 41.2、C-Eval 42.9 对 37.7，故结论要 `take it with a grain of salt`。
- MLA 压表示而非头数：`store compressed vector c = W_c h (C dimensions)`，用时 `K = W_K c`、`V = W_V c`；DeepSeek v2 把 `N*H = 16384` 压到 `C = 512`（约 32 倍），因不兼容 RoPE 额外加 64 维成 576。
- CLA 跨层共享 KV，滑动窗口只看 last K tokens 使 `KV cache ∝ B·K·L·H` 与 S 脱钩，线性注意力用 Mamba、DeltaNet 把历史压成定长状态，但 `no free lunch`、hurts accuracy；三条轴是 across layers（CLA）、across heads（GQA）、across head dimension（MLA），DeepSeek 再以 CSA（every m tokens into 1）、DSA（top k）、HCA 配 Lightning Indexer 组合。

### 量化、剪枝与无损的投机解码
- 量化属 systems 视角的 `reduce the precision of numbers`：`x=5.2342, scale=0.1, zero_point=4` 得 `x_quant = round(x/scale)+zero_point = 56`、`x_dequant = (56-4)*0.1 = 5.2`，误差 0.0342 即精度损失来源；阶梯为 bf16(2 bytes)、fp8 e4m3(1 byte, [-240,240])、int8(1 byte, [-128,127])、int4(0.5 bytes, [-8,7])。
- QAT 在前向里 quantize-and-dequantize，PTQ 用 sample data 定 scale/zero point，GPTQ 借 Hessian 把 `E = W − Q(W)` 补偿给未量化权重 [Frantar+ 2022]，AWQ 给大幅值激活通道留 FP16、其余 INT3；剪枝按 NVIDIA 三步用 1024 samples 校准集估重要性、移除、蒸馏，15B→Minitron 8B/4B 得 `40x cheaper 9% better`，重要性看 activation magnitude，high mean、low variance 者宜并入 bias。
- 投机解码是全课首个 lossless 方法：`checking is faster than generation`，draft p 自回归猜 K 个（e.g. 4），target q 一次并行算 K+1 组 logits，按 `α = min(1, q/p)` 接受、拒绝时从 `(q−p)_+` 残差补样，一轮最多前进 K+1 个 token，无损性由 `P[sampling A] = q(A)`、`P[sampling B] = q(B)` 保证。
- Chinchilla 在 K=4、batch size 1 下 Mean Token Time 从 14.1ms 降到 7.52/7.00/5.73ms，Speed Up 1.92×/2.01×/2.46×，甜点约 K=3~4；draft 越接近 target 越好，最好蒸馏出来，Medusa 等多预测头是后续方向。

### 系统层调度与全课收束
- `continuous_batching()`（Orca）用 iteration-level scheduling 治理 ragged array：每步给所有序列 decode one token，序列 EOD 即驱逐、新请求随到随补；`M copies` 的 easy parallelism 只把吞吐乘 M。
- selective batching 因长度不一，attention `process each sequence separately`（3×3 与 9×9 无法共享），non-attention 则拼成 `[3 + 9 + 5, H] = [17, H]` 的 mega-sequence 一次算掉。
- PagedAttention 出自 vLLM（非 VLM），把 KV cache 切成 `block of size four` 的非连续块，治理 2038 / 507 slots 的 internal 与 external fragmentation；`share prefixes, copy-on-write at the block level` 在 Ref count 2→1 处才复制，另有 kernel fusion、FlashAttention/FlashDecoding。
- Summary 收束为 Inference is important（actual use, evaluation, reinforcement learning）、memory-bound 且 dynamic、Techniques 与 Ideas from systems (speculative execution, paging)，并高亮 `New architectures have huge potential for improvement`；统一式是 min KV cache s.t. accuracy 损失 ≤ ε，只有投机解码把 ε 压到 0；因为 attention 需要随序列线性增长的 KV cache，与 memory-bound 的 generation 天生冲突，展望落在 state space models、linear attention、diffusion。

## 视频来源

- 原始链接：https://www.youtube.com/watch?v=EfM546A79aM
- 课程：Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 10

---

## 对照参考笔记补遗（Spring 2025 版）

> 编者注：本节依据 2025 版参考笔记整理，其中的「更正」小节记录的是核对时发现的问题，对应正文已在本版订正；补遗内容不代表 2026 课程视频原文。


本节整理自 Spring 2025 版参考笔记，仅用于与本讲 2026 视频详写笔记对照补充，不代表 2026 课堂原文；凡表述不同处，应优先按课程版本差异理解。

### 从离群值到激活感知：两条被略过的量化结论

正文已经用 `scale` 与 `zero_point` 把仿射量化写成可代入的两行代码，但参考还给出一条正文没有展开的方法线：standard quantization，即按绝对最大值定标的 LLM.int8()（Dettmers+ 2022）。做法是对一条 FP16 向量先取动态范围，用 $\max_i|x_i|$ 定出缩放因子 $s$，把每个分量缩放、四舍五入到 INT8，反量化时再乘回；把这段口头描述写成式子就是 $\tilde{x}=\operatorname{round}(x/s)$、$\hat{x}=s\,\tilde{x}$，INT8 的取值范围是 $[-128,127]$。问题出在 larger networks 里会浮现 outliers（离群特征）：极少数幅值巨大的维度会把 $s$ 撑得很大，于是绝大多数普通值被压到 $\operatorname{round}(x/s)\approx 0$ 附近的少数几个整数格点上，量化分辨率几乎全被浪费。这正是正文里「按每个 layer 或 tensor 各自朴素标定 generally doesn't work as well」的根因，也解释了为什么后面必须引入更聪明的定标方式。

参考给出的解法是 mixed-precision decomposition：把 outliers 单独抽出来留在 FP16 处理，其余维度照常走 INT8。它换来了精度，却不是免费的——参考明确记下这套做法比纯 FP16 慢 $15\text{–}23\%$（15-23% slower than FP16）。把它放回正文的框架就清楚了：量化属于 systems 视角的 reduce the precision of numbers，收益来自 memory-bound 阶段搬运字节数下降，而离群值分支引入的额外 FP16 计算与判断把这部分收益吃掉了一截，所以「量化必然加速」并不成立。还要与正文的 GPTQ 分清作用对象：GPTQ 借 Hessian 把权重量化误差 $\mathbf{E}=\mathbf{W}-Q(\mathbf{W})$ 补偿给尚未量化的权重，处理的是权重误差；LLM.int8() 处理的是激活里的离群维度，一个在权重侧、一个在激活侧，不能互相替代，更不能把「15–23% 更慢」这个数字安到 GPTQ 头上。

与之并列的另一条是 activation-aware quantization（Lin+ 2023），正文讲了「按激活幅值给少数通道保留 FP16、其余压到 INT3」的机制，却没有给出收益量级。参考补齐了定量结论：依据激活筛出仅 $0.1\text{–}1\%$ 的 salient weights（显著权重）保留高精度，把 FP16 权重整体压到 INT3，可得到约 $4\times$ 的内存下降与约 $3.2\times$ 的加速。易错点有两处：$0.1\text{–}1\%$ 是「极少数权重」，不是 $10\%$ 量级，读错一个量级就会误以为绝大多数权重仍占着高精度；$4\times$ 与 $3.2\times$ 绑定该论文的模型、硬件与评测，与正文 Llama 2 13B 的 $124.6755$ tokens/s 不是同一套基准，不能横向相除或相加。把四条线合起来看，QAT 靠训练适应、GPTQ 靠二阶补偿、AWQ 靠激活统计、LLM.int8() 靠离群值隔离，压制的是同一份量化误差，却谁都无法把精度损失严格压到零——这正是正文把量化归入 `Taking shortcuts (lossy)` 的原因。

### 局部注意力的文献坐标与工程配比

正文把 sliding window attention 讲成「每个 token 只看最近 $K$ 个、KV cache 与序列长度脱钩」，参考则补上了它的出处：Beltagy+ 2020、Child+ 2019、Jiang+ 2023 是这一思路的三篇经典工作。参考还把正文那句口语化的「信息可以逐层向前传播」写成可记的定性结论——effective context scales linearly with the number of layers，有效上下文随层数线性增长。换句话说，堆叠多层之后模型真正能利用的范围不再是常数 $K$，而是一个随层数 $L$ 线性延展的量；这既解释了滑窗在长上下文里为何仍有一定表达力，也提醒我们正文「KV cache 不再含 $S$」说的是显存占用，而非模型能看到的信息范围，两者不要混为一谈。

精度代价在参考里写得更直白：long-range dependencies 丢失会 hurts accuracy，因此业界用 hybrid layers 把 local attention 与 global attention 交错起来。参考给出一个正文完全没有的具体配比：Character.ai 采用每 6 层插 1 个全局层（1 global layer every 6 layers），即全局层与局部层约成 $1:6$。这与正文「交错方案只是 reduce the KV cache a little bit」的判断一致——它是折中，不是 GQA/MLA 那种成倍压缩。同一个小节里，cross-layer attention（CLA）被明确标为 Brandon+ 2024，机制正是正文写的「像 GQA 跨 head 共享那样跨 layer 共享 KV」，参考给的结论是它 improves the Pareto frontier of accuracy and KV cache size，与正文的定性描述吻合。

读这一组材料时要守住三条压缩轴的区分：GQA 动 $K$（KV head 数）、MLA 动每头维度（$N\cdot H\to C$）、CLA 动 $L$（层数），而滑窗干脆让 $S$ 这一项消失或变成常量。易错点在于别把 Brandon+ 2024 的功劳安到 MLA 或 DeepSeek 头上，也别把 $1:6$ 当成通用最优配比——它只是 Character.ai 在自家负载上的工程选择；正文反复提醒的「经验结论须按自己任务重测」，在这里同样适用，参考自己也用 `no free lunch`、`hurts accuracy` 给这些方法留了余地。

### 超越 Transformer：状态空间模型谱系

正文结尾把 state space models、linear attention、diffusion 列为「没时间展开」的展望，参考 2025 版却有一整节 Alternatives to the Transformer（并注明取材自 CS229S 的 presentation）。动机与正文的分析框架同源：Transformer 并非为推理效率设计，autoregressive 加 full attention 造成根本的 memory-boundedness，因此要问能否走 beyond the Transformer。SSM 来自信号处理，目标是以 sub-quadratic（次二次）时间建模长上下文，正好对准正文 $\frac{ST}{S+T}$ 在 $T=1$ 时塌缩到 $\frac{S}{S+1}<1$ 的那一格——那一格被判为「只要还用标准 Transformer 就无解」，换架构正是想绕开它。

谱系与数字值得逐项记下。S4（Gu+ 2021）基于经典状态空间/线性动力系统，擅长合成的长上下文任务；软肋是 associative recall（联想回忆），参考给的典型例子是 `A B3 C6 F1 E2 -> A? C? F? E? B?` 这类按内容检索的题目，而这恰是 Transformer 擅长的。Mamba（Gu+ 2023）让 SSM 参数变成 input-dependent，从而在 $1$B 规模上追平 Transformer。Jamba（Lieber+ 2024）按 $1:7$ 交错 Transformer 层与 Mamba 层，配 $52$B MoE；BASED（Arora+ 2024）用 linear attention 加 local attention；Minimax-01 用 linear attention 加 full attention 的 $456$B 参数 MoE。linear attention 的机制是把 attention kernel 用 Taylor expansion 近似成可分解形式，例如 $\exp(q^\top k)\approx 1+q^\top k+\tfrac12(q^\top k)^2+\cdots$，从而对序列长度线性而非二次。

把这条线与正文的 KV cache 公式接起来，结论最有用：linear + local attention 仍需保留一部分 full attention 才撑得起 serious SOTA 模型，而一旦把随 $T$ 增长的 KV cache 换成固定大小的 state，推理侧的存储与搬运就从 $O(T)$ 降到 $O(1)$（参考原句在解析里把两个复杂度都留了下来，意思是用状态替代缓存对推理更高效）。易错点有三：Mamba 追平 Transformer 是限定在 $1$B scale 的结论，不是全尺度普适；Jamba 的 $1:7$ 是层类型交错比，Character.ai 的 $1:6$ 是全局层间隔，两个比例含义不同不可混记；参考这一整节在 2026 视频里被讲师以「没时间」略过，因此上述数字只能作为 2025 版的对照证据，不能当成 2026 课堂原文。

### 非自回归的另一条路：文本扩散模型

与 SSM 并列的另一类 radical architectural change 是 diffusion models。正文只把它作为一个名词留在展望里，参考却给出了机制与可对照的速度结论：扩散在图像生成里成熟，用于文本更难，思路是 Generate all tokens in parallel（不再逐 token 自回归），从整段随机噪声起步，用多个时间步 iteratively refine（迭代细化）。这与正文反复强调的「自回归无法在序列维并行」正好相反，因而理论上绕开了 generation 阶段算术强度塌缩那一格；若记 $n_{\text{steps}}$ 为固定的细化步数、$t_{\text{step}}$ 为单步耗时，则 $T_{\text{gen}}\approx n_{\text{steps}}\cdot t_{\text{step}}$ 与序列长度 $T$ 脱钩，这和正文 $\text{latency}\approx\text{bytes}/\text{memory bandwidth}$ 随 KV cache 线性变差的形态是两种截然不同的伸缩性。

参考点名 Inception Labs 的实测方向：diffusion 在 coding benchmarks 上明显更快，Mercury Coder Small / Mini 比 Gemini、Claude、GPT-4o、Jamba 等 Transformer 系模型 significantly faster。放进正文的分类里看，它属于「用更激进的架构改动换速度」，与量化、剪枝、GQA/MLA 这类给 Transformer 打补丁的手段不在同一层级，也呼应正文 `New architectures have huge potential for improvement` 那句高亮，以及「attention 需要随序列增长的 KV cache、与 memory-bound 的 generation 天生冲突」这一根本判断。

两处易错点必须留住。其一，参考同时写明 diffusion is harder to get working for text generation，且这些速度结论绑定特定 benchmark 与实现，正文「经验结论按自己任务重测、take it with a grain of salt」的提醒在此同样有效，不能把某个倍数当普适常数。其二，「并行生成所有 token」与正文投机解码的「并行验证 $K$ 个 token」是两回事：前者改变生成范式、是非自回归的，后者仍是自回归但把串行生成改造成并行验证，并且靠 $\min(1,q/p)$ 与 $(q-p)_+$ 保证 exact sample；把两者的加速来源混为一谈，就会误以为扩散模型也具备那份数学上的无损保证。

### 投机解码的 draft 侧延伸与文献归属补注

正文给了 Algorithm 2 完整伪代码、两 token 无损性恒等式与 Chinchilla 实测表，参考则补了 draft 一侧的两条延伸：Medusa 让 draft 并行生成多个 token，EAGLE 让 draft 取用 target model 的 high-level features（高层特征）。正文只出现 Medusa，EAGLE 是被漏掉的系统名。参考还给出一对直观的实践规模：target model 约 $70$B、draft model 约 $8$B（约 $9:1$），并强调希望 draft 尽量接近 target、distillation helps——这与正文「draft 越接近 target 越好，最好蒸馏出来」一致，只是补上了具体量级，也把前面剪枝/蒸馏得到的小模型与投机解码接成了一条线：精度满意就 `just serve that`，不满意就至少拿来当 draft。

把它们放回正文的代价式 $K c_p + c_q(K)$ 就好理解：$70$B/$8$B 的悬殊正是 $c_p\ll c_q$ 的来源，而 EAGLE 取 target 高层特征，本质是抬高 draft 与 target 的对齐度、进而抬高接受率 $\alpha=\min(1,\,q/p)$，让一轮里更多候选被接受、更接近「一轮最多前进 $K+1$ 个 token」的理想，也自然落在正文给的 $K=3\text{–}4$ 甜点附近。易错点很关键：EAGLE 只改 draft 的构造，不改接受规则，因此正文强调的 exact sample from the target model 依旧成立；一旦为提速改动 $\min(1,q/p)$ 或截断残差分布 $(q-p)_+$，无损性立刻退化为近似采样，正文那条「无损性是有条件的数学结论」的警告在这里再次生效。

最后补齐几处参考给出、正文略去的文献归属，便于按图索骥：PagedAttention 出自 Kwon+ 2023（正文写作「introduced in the vLLM paper」，并已纠正转写里的 VLM 听写误差）、NVIDIA 剪枝工作是 Muralidharan+ 2024、Speculative Sampling 由 Leviathan+ 2022 与 Chen+ 2023 几乎同时提出、GQA 出自 Ainslie+ 2023 且参考明确记有 Llama 3 adopted GQA。这些都是归属补注，不改变正文任何公式或数字；把它们与正文已有的 Frantar+ 2022（GPTQ）、DeepSeek-AI+ 2024（MLA）并列，便得到一张完整的引用地图。仍要记住，参考里这些人物与年份属于 2025 版讲义的标注口径，与 2026 视频里新增的 DeepSeek CSA / DSA / HCA 等后续工作并不冲突，只是编排先后不同；判断某条经验结论能否迁移，依据仍应回到正文的算术强度与 prefill/generation 两阶段框架，而不是记住某一个作者或年份。
