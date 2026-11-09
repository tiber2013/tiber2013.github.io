---
title: "第 7 讲 · 循环神经网络：从隐藏状态到 LSTM"
description: "Stanford CS231n Spring 2025 L07 中文视频笔记（2026-11 重写版）"
---

# CS231n Spring 2025 · Lecture 7 循环神经网络与序列建模

**核心结论**：本讲把视野从固定尺寸输入/输出的前馈网络转向变长序列，先按 one-to-one、one-to-many、many-to-one、many-to-many 四种拓扑形式化序列任务，再给出 RNN 的核心——一个随序列更新、由同一组权重反复递推的内部（隐）状态 $h_t=f_W(h_{t-1},x_t)$。Vanilla RNN 用 $\tanh$ 递推、$y_t=W_{hy}h_t$ 输出，讲者用「检测连续两个 1」的玩具例子手工把权重矩阵逐项拆开，说明隐状态究竟存了什么。训练上把序列展开成计算图，用随时间反向传播（BPTT）对复用的 $W$ 逐步求梯度再求和，长序列则用截断 BPTT 分块以省显存；字符级语言模型展示了这种逐步 softmax 分类如何一路演化到今天的代码生成工具。RNN 能处理任意长度、模型不随长度增大，但循环计算慢且固定隐状态难以保留远距离信息；梯度沿时间连乘 $W_{hh}^{T-1}$ 会因奇异值大于/小于 1 而爆炸或消失，于是有了 LSTM 的门控高速通路，以及近年 RWKV、Mamba 这类线性时间状态空间模型带来的「RNN 复兴」。

## 1. 开场澄清、工具与课程衔接 [00:05 - 04:49]

讲者先补两个上一讲的澄清。其一是 Dropout 测试时缩放：$p$ 在多数库里表示「被丢弃」的比例，但课件代码把 $p$ 用作「保留」概率，二者存在错位；核心不变——**测试时期望输出要与训练时一致**，所以训练丢弃 25% 时，测试要按 0.75 缩放。其二是归一化与初始化的关系：在一个 2D 输入、两层 ReLU 网络输出象限函数（点落在哪个象限就输出 1~4）的玩具问题上，对比了良好初始化（Kaiming）与坏初始化（标准差过大）各自的训练/测试损失——蓝线是坏初始化、绿线是坏初始化叠加 LayerNorm，可见归一化确实解决了一大半问题，但要拿到最好性能仍离不开良好的权重初始化。而且 LayerNorm 是否有帮助取决于问题：象限任务只需大致区域、不需要精确的 2D 坐标，归一化帮了忙；而另一些必须知道精确坐标才能输出正确的函数，减均值除标准差反而丢掉空间位置信息、会损害性能。

![Dropout正则化原理幻灯片，对比了完整神经网络与随机丢弃神经元后的网络结构](/coursenotes/cs231n-deep-learning-for-cv-sp25/l07-rnns/kf_0001_00-00-35.jpg)

随后复习前馈网络的整套固定流程：激活函数、按固定均值/标准差做图像预处理、权重初始化、归一化、迁移学习，以及训练动态（babysit 学习过程、按验证表现调超参、测试时增强）。讲者强烈推荐用 Weights & Biases 之类的工具管理多组 run——把不同超参（如 dropout 取值）列成列并颜色编码，就能直观看出较低 dropout 往往对应更高验证准确率；同类还有 TensorBoard。本讲余下部分转向序列建模：变长序列、Transformer 之前的 RNN 及其变体，以及仍在沿用 RNN 关键思想、并具备若干优于 Transformer 特性的现代状态空间模型（如 Mamba）。

## 2. 序列建模的四种拓扑 [04:49 - 07:28]

![循环神经网络处理序列的五种架构模式对比图（one-to-one至many-to-many）](/coursenotes/cs231n-deep-learning-for-cv-sp25/l07-rnns/kf_0002_00-07-11.jpg)

相对「一个固定输入 → 一个固定输出」的 vanilla 网络，序列任务有四种典型形态：**one-to-many**（固定尺寸输入如图像 → 变长输出，例子是 image captioning，输出一串词/字符）；**many-to-one**（变长输入如视频多帧 → 单个类别标签，类似图像分类但输入是多帧）；**many-to-many**（输入输出都是序列，长度不必相等，例如视频帧 → 变长 caption；也可以每个输入对应一个输出）。课件把视频帧级分类（Video classification on frame level）标为 many-to-many。讲 RNN 时主要聚焦「每个时间步都有一个输入和一个输出」的对齐 many-to-many，因为最直观，稍作改动即可套用到其他三种。

## 3. RNN 的核心：内部状态与递推公式 [07:28 - 13:13]

RNN 之所以「循环」，常用一个回指自身的箭头表示，其真正含义是网络有一个随序列处理不断更新的**内部状态 / 隐状态**：每来一个新输入，就结合它与上一个隐状态算出新隐状态。为看清梯度与运算顺序，通常把 RNN 沿时间**展开（unrolled）**成计算图，显式写出每一步对当前输入与上一状态的依赖。

![循环神经网络（RNN）基本结构示意图及内部状态更新机制说明](/coursenotes/cs231n-deep-learning-for-cv-sp25/l07-rnns/kf_0003_00-08-18.jpg)

用数学写就是对每个时间步套用同一个递推式：

$$h_t=f_W(h_{t-1},x_t)$$

![RNN隐藏状态更新公式 h_t = f_W(h_{t-1}, x_t) 及其各部分含义标注](/coursenotes/cs231n-deep-learning-for-cv-sp25/l07-rnns/kf_0004_00-11-08.jpg)

它和最初学的网络层一样是「权重矩阵乘 + 激活函数」，唯一变化是**递归**：每个时间步复用同一套 $W$ 和同一个激活函数。把隐状态变成输出是另一个函数、另一套参数 $W_{hy}$，它同时做两件事——把隐状态维度映射到输出维度（二者可不同），并施加一次可学习变换。隐状态初始化为 $h_0$，原则上可以是一个学出来的输入向量。至于上一时刻的输出 $y$ 能否影响新隐状态：在语言模型 / 自回归建模里，人们直接把上一时刻的值当作下一步输入，这就是显式让 $y$ 影响状态的常见做法。

## 4. Vanilla RNN 与手工构造实例 [13:13 - 24:06]

「Vanilla RNN」几乎特指用双曲正切 $\tanh$ 作激活的形式（也称以 Jeffrey Elman 命名的 Elman RNN）：

$$h_t=\tanh(W_{hh}h_{t-1}+W_{xh}x_t),\qquad y_t=W_{hy}h_t$$

![Vanilla RNN结构图、tanh激活函数曲线及核心递推公式](/coursenotes/cs231n-deep-learning-for-cv-sp25/l07-rnns/kf_0005_00-13-15.jpg)

$\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$ 有界于 $[-1,1]$、零中心、能表示正负，反复递推也不会让数值跑飞；最简情形输出就是隐状态的一次矩阵乘。

![Vanilla RNN具体案例：检测重复1s的Many-to-many序列建模任务](/coursenotes/cs231n-deep-learning-for-cv-sp25/l07-rnns/kf_0006_00-15-31.jpg)

为把前向传播讲透，讲者不靠梯度下降，而是**手工搭一个 RNN**：输入是 0/1 序列，任务是 many-to-many 地检测「连续两个 1」，出现重复 1 输出 1、否则 0。隐状态需要记住「上一个输入」和「当前输入」，因此设成 3 维 $[current, previous, 1]$（第三维恒为 1 是给输出阶段当偏置用），初始化 $h_0=[0,0,1]$（相当于假设此前见过两个 0）；为简化数学用 ReLU 而非 tanh。

![使用NumPy手动实现Vanilla RNN的代码及权重矩阵配置](/coursenotes/cs231n-deep-learning-for-cv-sp25/l07-rnns/kf_0007_00-19-09.jpg)

```python
w_xh = np.array([[1], [0], [0]])
w_hh = np.array([[0, 0, 0],
                 [1, 0, 0],
                 [0, 0, 1]])
h_t = np.maximum(0, np.dot(w_hh, h_prev) + np.dot(w_xh, x_t))
```

$W_{xh}$ 把当前输入放进隐状态第一维（$x=0$ 得 0、$x=1$ 得 1）；$W_{hh}$ 第一行全 0 让第一维只由当前输入决定、第二行 $[1,0,0]$ 把上一时刻的 current 搬成这一时刻的 previous、第三行 $[0,0,1]$ 保持那个恒为 1 的分量。输出用 $1\times3$ 的权重做点积，相当于 $current+previous-1$ 再过 ReLU：只有 current 与 previous 同时为 1 时结果为 1，否则为 0 或 $-1$ 被截断成 0。一般 RNN 里这两个矩阵分别负责「上一隐状态的传递」与「当前输入的注入」，本例只是把它们手工设成可实现记忆的值；真实权重仍靠梯度下降学出（见下一节）。

## 5. 计算图、BPTT 与截断 BPTT [24:06 - 36:00]

![RNN Many to Many 计算图，展示输入序列 x 与输出序列 y 的逐步对应及损失函数 L 的连接](/coursenotes/cs231n-deep-learning-for-cv-sp25/l07-rnns/kf_0008_00-25-08.jpg)

many-to-many 计算图里每个时间步都有输出，通常每步算一个 loss 再求和成总损失；由于每个时间步用的是同一个 $W$，概念上可把它视作每步一份「副本」分别求梯度，最后把梯度**全部加起来**（$\frac{\partial L}{\partial W}=\sum_t\frac{\partial L_t}{\partial W}$）。

![RNN Many to One 计算图，展示长序列输入 x 最终汇聚为单一输出 y 的结构](/coursenotes/cs231n-deep-learning-for-cv-sp25/l07-rnns/kf_0009_00-26-24.jpg)

many-to-one 只在末尾算一个 loss：可以只用最后隐状态，也可以对每步隐状态做 average / max pooling 再得 $y$（视频分类里整段视频都有信息时更合理）。one-to-many（image captioning）仍需给 $f_W$ 一个输入向量，可以填 0，也可以填上一时刻的输出。

![BPTT（随时间反向传播）结构示意图，展示序列展开及梯度回传路径](/coursenotes/cs231n-deep-learning-for-cv-sp25/l07-rnns/kf_0010_00-27-48.jpg)

**随时间反向传播（Backpropagation Through Time, BPTT）**：前向跑完整条序列算 loss，再反向跑完整条序列算梯度。问题很实际——序列很长时，必须把每一步的激活和梯度都留在显存里再求和，极易爆显存（和 NaN loss 一样是训练中最常见的麻烦）。

![截断BPTT（Truncated BPTT）单输出模式示意图，展示分段计算梯度的方法](/coursenotes/cs231n-deep-learning-for-cv-sp25/l07-rnns/kf_0011_00-32-24.jpg)

**截断 BPTT（Truncated BPTT）** 的解法是固定一个时间窗口分块：每块从 $h_0$ 出发（用上一块最后的隐状态初始化，但**梯度不跨块传递**），只在窗口邻域内算 loss、把计算图切块批处理。单输出模式下仍每步算梯度，但不再有「每步一个 loss」，而是依赖上游梯度回传；块边界只需保留一份「该隐状态对损失的梯度」，用它回推块内各步并更新权重，然后清空显存进入下一块。这不是精确计算（相当于把一次大更新拆成多次独立更新），但每步梯度仍在算，实践中非常常用；这与分布式训练里各 GPU 分别算梯度再施加到同一组权重是类似情形。讲者还强调，若每个时间步用不同的 $W$，就不再是递推建模、退化成「每个位置一层」的普通网络，既要求预先知道序列长度，也难以泛化到变长序列。

## 6. 字符级语言模型：采样、嵌入与可解释性 [36:37 - 48:19]

字符级语言模型把「预测下一个字符」当成逐步 softmax 分类：字符用 one-hot 编码（词表如 $[h,e,l,o]$），输入经隐层、输出层得到 logits 再 softmax。

![字符级语言模型采样示例及神经网络前向传播数值演示图](/coursenotes/cs231n-deep-learning-for-cv-sp25/l07-rnns/kf_0012_00-38-17.jpg)

课件给了一串具体数值：输入字符 "h"（one-hot $[1,0,0,0]$）、隐层 $[0.3,-0.1,0.9]$、输出层 $[1.0,2.2,-3.0,4.1]$、softmax 后 $[.03,.84,.00,.13]$，据此采样出 "e"；讲者当场指出这张老幻灯片其实标错了概率、模型这步预测是错的，训练时会被重罚。测试时**逐字符采样并反馈给自己**（autoregressive），直到生成完整文本。

![字符级语言模型采样示例，展示权重矩阵与独热向量相乘提取列向量的原理及嵌入层结构](/coursenotes/cs231n-deep-learning-for-cv-sp25/l07-rnns/kf_0013_00-39-36.jpg)

实际实现并不直接把 one-hot 喂进模型，而是在输入与隐层之间放一个 **embedding layer**：一个 $d\times d$ 的大矩阵，用矩阵乘按输入索引取出对应列/行（如 $[w_{11}\,w_{12}\,w_{13}\,w_{14}]\cdot[1,0,0,0]^T=w_{11}$）得到稠密向量；它可用 Kaiming 初始化成小随机值再学，优化上比稀疏 one-hot 更好。这类模型约 112 行 Python 就能写：Andrej Karpathy 2015 年的博客《The Unreasonable Effectiveness of Recurrent Neural Networks》展示了在莎士比亚十四行诗上训练，输出从 gibberish 逐步变得像英文、学会人名与文风，甚至在 Linux 源码上生成看起来合理的 C 代码。

![列举了OpenAI Codex、GitHub Copilot和Cursor IDE等AI编程工具及其Logo](/coursenotes/cs231n-deep-learning-for-cv-sp25/l07-rnns/kf_0014_00-45-41.jpg)

今天 OpenAI Codex、GitHub Copilot、Cursor IDE 等编程工具本质上沿用同一任务，只是把「下一个字符」换成「下一个 token」（一组字符）自回归预测；纯语言模型只需大量文本、无需人工标注，因为标签就是下一个 token，这正是它们能靠爬取全网文本变强的原因。采样若总取最大概率（greedy decoding）会永远生成同一结果，所以按 softmax 分布采样，另有 beam search 等一整套解码研究。此外，RNN 隐状态激活（tanh 下 $-1$ 显红、$+1$ 显蓝）常被发现高度可解释：有专门跟踪引号起止的 quote detector、跟踪行长的 line-length cell、if 语句 cell、注释 cell，以及随代码嵌套加深而增强的 code-depth cell——模型在内部自发学会了类似手工设定的状态。

## 7. RNN 的优缺点 [48:19 - 50:49]

![RNN 的优缺点对比列表，包括处理任意长度输入、权重共享对称性以及计算慢等特性](/coursenotes/cs231n-deep-learning-for-cv-sp25/l07-rnns/kf_0015_00-48-28.jpg)

优点：**能处理任意长度输入**（不像 Transformer 有 context length / 最大上下文窗口限制）；时间步 $t$ 的计算理论上可利用很多步之前的信息（只要被隐状态捕获）；**模型大小不随输入变长而增大**；每个时间步施加同一套权重，处理方式具有对称性、概念与实现都更简洁。缺点：**循环计算慢**——每个隐状态都依赖前一个，训练时难以把各步批量并行（推理时 Transformer 也要逐个出 token，差别主要在训练）；**实践中很难访问很多步之前的信息**，因为固定大小的隐状态要塞进整段历史，序列越长丢得越多。

## 8. 视觉中的 RNN 应用 [50:49 - 58:22]

### 8.1 图像描述、VQA 与导航 [50:49 - 56:26]

Image captioning 用 start token 起头、采到 end token 结束（课件例子是词级 token）。基本结构是 CNN 视觉编码器 + RNN 解码器：取 ImageNet 预训练 CNN 的**倒数第二层**（不用分类层）作为图像向量 $v$ 注入隐状态，状态更新从

$$h=\tanh(W_{xh}\,x+W_{hh}\,h)\quad\text{变为}\quad h=\tanh(W_{xh}\,x+W_{hh}\,h+W_{ih}\,v)$$

![图像描述生成模型架构图，展示了 CNN 特征提取与 RNN 解码器的连接，以及引入视觉向量 v 后的状态更新公式变化](/coursenotes/cs231n-deep-learning-for-cv-sp25/l07-rnns/kf_0016_00-52-39.jpg)

![图像描述任务的失败案例展示，包含五个错误生成的图注示例](/coursenotes/cs231n-deep-learning-for-cv-sp25/l07-rnns/kf_0017_00-54-37.jpg)

失败案例很有代表性，几乎都源于训练数据里的共现偏差：手捧扁平物体被说成「拿着鼠标」（实为手机）、毛皮衣服被说成「抱着猫」、看到沙滩就臆造冲浪板、鸟栖树枝、把接球说成投球。模型只学「这类场景里通常有什么物体/动作」，却没有解耦当前画面里到底有没有，这类幻觉在今天的视觉语言模型里依旧常见——因为训练只要求输出 caption，并不要求解释。此外 RNN 还用于 **Visual Question Answering**（两种形式：用 captioning 模型对各候选答案序列算概率；或把问题与多个候选答案一起输入，输出每个答案的概率，成为四选一分类器）、**Visual Dialogue**（如今多由一个模型统揽），以及 **Visual Language Navigation**：Agent 把语言指令编码，用 RNN 在每步视觉输入变化后生成一系列移动动作，在房屋平面图中走向目标（Wang et al., CVPR 2018）。

![视觉语言导航任务示意图，展示Agent根据指令在房屋地图中规划路径的过程](/coursenotes/cs231n-deep-learning-for-cv-sp25/l07-rnns/kf_0018_00-56-26.jpg)

### 8.2 多层 RNN [56:26 - 58:22]

![多层RNN（Multilayer RNNs）的堆叠结构示意图，展示深度与时间维度的展开](/coursenotes/cs231n-deep-learning-for-cv-sp25/l07-rnns/kf_0019_00-58-09.jpg)

RNN 也能像 CNN / 全连接网络那样堆叠成多层，实际展示的模型大多是多层 RNN。它形成 depth × time 的网格：每一层只关注**本层**上一时间步的隐状态（层内沿时间共享同一套权重，层与层之间各用独立权重）；第一层输入是 $x$，更高层把上一层的输出 $y$ 当作输入。要算右上角那个值，得先把整张计算图里所有隐状态都算出来，足见其训练开销。

## 9. 梯度问题、LSTM 与 RNN 的复兴 [58:22 - 71:38]

### 9.1 梯度消失与爆炸 [58:22 - 64:50]

把 $W_{hh}$ 与 $W_{xh}$ 拼成一个大 $W$（含大量零块，且 $W_{hh}$ 并不与 $x_t$ 交互）是常见简写；反向传播时每跨一个时间步就要乘一次 $W_{hh}$ 并乘一次 $\tanh$ 的导数。假设没有非线性，最终损失对权重的梯度形如

$$\frac{\partial L}{\partial W}=\sum_{t=1}^{T}\frac{\partial L_t}{\partial W},\qquad \frac{\partial L_T}{\partial W}=\frac{\partial L_T}{\partial h_T}\,W_{hh}^{T-1}\,\frac{\partial h_1}{\partial W}$$

![Vanilla RNN的梯度流动图及梯度消失/爆炸问题的数学推导](/coursenotes/cs231n-deep-learning-for-cv-sp25/l07-rnns/kf_0020_01-04-37.jpg)

于是 $W_{hh}$ 的**最大奇异值 $>1$ 导致梯度爆炸、$<1$ 导致梯度消失**；而 $\tanh$ 导数最大只有 1、几乎处处小于 1，更放大了消失（奇异值衡量一个单位向量能被矩阵拉伸多少）。爆炸可用梯度裁剪 / 缩放（gradient clipping）解决，消失才是人们不用超长 RNN 的主因。

### 9.2 LSTM：给信息一条高速路 [64:50 - 69:12]

LSTM 早在 1990 年代提出，在 Transformer 之前长期成功。它不再只有单一隐状态，而是用四个 gate 共同决定如何更新：input gate（是否写入）、forget gate（从过去忘掉多少）、output gate（输出多少隐状态），以及候选写入值；关键是开辟了一条**不含激活函数（没有 tanh）、只乘 forget gate 的高速通路**，让长期信息可以几乎无损地跨很多时间步流过。课程不要求实现细节，只需理解它是为弥补 RNN 的梯度消失与「固定隐状态塞不下长期依赖」而设计；它显著缓解但**不保证**彻底解决梯度消失，经验效果很好，至今仍是一些论文的 baseline，不过整体已淡出主流。讲者还把它与上一讲的 ResNet 类比：跳过若干层直接相加的 skip connection，在 ResNet 里跨的是「深度」，在 LSTM 里跨的是「时间步」。

### 9.3 状态空间模型与 RNN 复兴 [69:12 - 71:38]

RNN 近一两年有回潮：Transformer 的有限 context length 是硬限制（RoPE 等只是缓解手段），而 RNN 推理 / 训练计算随序列长度**线性**增长，且没有对整条序列做全局交互的操作（Transformer 是二次）。RWKV、Mamba 等状态空间模型主打 linear time sequence modeling，试图兼取 Transformer 的性能与 RNN 的扩展性，是当下「如何兼顾两者」的热点。下一讲进入 attention 与 Transformer。

## 要点回顾

- 序列任务分 one-to-one / one-to-many（图像描述）/ many-to-one（视频分类）/ many-to-many（帧级标注、视频转 caption）；RNN 主要讲每步对齐的 many-to-many。
- RNN 的核心是随序列更新的隐状态 $h_t=f_W(h_{t-1},x_t)$，每步复用同一套 $W$；输出由另一套参数 $W_{hy}$ 映射，$h_0$ 可学习，自回归任务把上一输出当下一输入。
- Vanilla RNN（Elman RNN）用 $h_t=\tanh(W_{hh}h_{t-1}+W_{xh}x_t)$、$y_t=W_{hy}h_t$；tanh 有界 $[-1,1]$、零中心。
- 手工实例：用 3 维隐状态 $[current,previous,1]$ 与 ReLU，靠 $W_{xh}=[[1],[0],[0]]$、$W_{hh}$ 的移位设计实现「检测连续两个 1」，输出为 $current+previous-1$ 过 ReLU。
- BPTT 前向整序列算 loss、反向整序列算梯度，同一 $W$ 的逐步梯度求和；长序列用截断 BPTT 按固定窗口分块、只保留块边界隐状态梯度以省显存，属近似更新。
- 字符级语言模型逐步 softmax 分类、测试时采样并反馈；用 embedding layer 取矩阵对应列代替 one-hot；Karpathy 2015 博客、112 行 Python、莎翁与 Linux 源码生成，延伸到 Codex / Copilot / Cursor 的下一 token 预测。
- 解码按 softmax 分布采样（greedy 会重复），有 beam search；隐状态激活可解释出引号、行长、if、注释、代码深度等 cell。
- RNN 优点：任意长度、无 context 限制、模型不随长度增大、权重共享对称；缺点：循环计算慢、训练难并行、固定隐状态难保留长程信息。
- 视觉应用：CNN 倒数第二层向量 $v$ 注入 RNN 做图像描述（$h=\tanh(W_{xh}x+W_{hh}h+W_{ih}v)$），失败源于数据共现 / 幻觉；另有 VQA、视觉对话、视觉语言导航与多层 RNN。
- 梯度沿时间连乘 $W_{hh}^{T-1}$：奇异值 >1 爆炸（可裁剪）、<1 消失，tanh 导数 ≤1 加剧消失；LSTM 用 forget / input / output gate 与无激活高速通路缓解，类似 ResNet 跨时间步的 skip；RWKV、Mamba 等状态空间模型带来线性时间的 RNN 复兴。

## 视频来源

https://www.youtube.com/watch?v=kG2lAPBF7zA
