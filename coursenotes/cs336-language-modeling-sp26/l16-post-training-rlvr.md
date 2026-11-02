---
title: "第 16 讲 · 后训练进阶：RLVR 与 GRPO"
description: "核心结论"
---

# 第 16 讲 · 后训练进阶：RLVR 与 GRPO

> **Stanford CS336 语言模型从零构建 · L16** ｜ 讲者 Stanford Online ｜ 76 分钟 ｜ [原视频](https://www.youtube.com/watch?v=dIFAi87Ws4E)

**核心结论**

RLVR 的核心，是把 RLHF 中容易被过优化、受标注瓶颈限制的奖励模型，替换为更可验证、更可扩展的奖励信号，使模型能在数学、代码等任务上大规模做强化学习。PPO 虽然通用，但实现复杂、需要价值网络且对细节极其敏感；GRPO 去掉价值网络，用同一 prompt 下多次采样的组内 z-score 作为优势，成为开源 RLVR 的事实标准。DeepSeek R1、Kimi K1.5 与 Qwen3 展示了从极简 GRPO 到完整后训练流水线的不同组合方式，其中数据课程、长度控制和奖励鲁棒性是关键。RLVR 并不是“有验证器就万事大吉”：奖励仍可能被 hack，长度归一化、标准差归一化等实现细节也会改变实际优化目标。RL 仍是噪声大、工程复杂的环节，但 GRPO 让入门和复现变得远比 PPO 简单。

## 从 RLHF 到 RLVR：为什么需要可验证奖励 [00:05 - 03:07]

上一讲已经覆盖 instruction tuning 与 RLHF：先收集偏好数据，训练奖励模型，再让策略对着奖励模型做 RL。问题在于，这条路径受标注瓶颈限制——不能无限把算力投入同一个奖励模型，否则最终会过拟合并出现 overoptimization。讲师用 RM score 与 KL 距离的曲线说明：RLHF 的奖励模型只是人类偏好的代理，越往后优化，代理奖励可能越高，但真实胜率反而下降。

与 AlphaGo 这类“RL 原生”领域相比，差别在于目标是否精确可验证。围棋的胜负条件没有模糊性，因此可以尽可能多地投入算力；RLHF 更像学习问题，目标本身带有噪声和不精确性。形式数学、自然语言数学等任务更接近“可验证”的一端，因此更适合大规模 RL。接下来的算法框架与 PPO 类似，但最终得到的行为和系统会有显著不同。

![RL目标扩展幻灯片，通过RM分数与KL散度曲线图说明RLHF的过优化问题及AlphaGo等应用案例](/coursenotes/cs336-language-modeling-sp26/l16-post-training-rlvr/kf_0002_00-02-32.jpg)

## PPO 回顾：策略梯度、裁剪与工程复杂性 [03:48 - 12:38]

RL 用于语言模型时，最核心的出发点仍是 policy gradient，尤其是 REINFORCE 形式的梯度：

$$
\nabla_\theta \mathbb{E}_{p_\theta}[R(z)]
= \mathbb{E}_{p_\theta}\left[R(z)\nabla_\theta \log p_\theta(z)\right]
$$

可以把它理解为带权重的 SFT 更新，权重可正可负。为了复用 rollout、减少每次梯度更新都重新采样的开销，就有了 TRPO、PPO 等方法。PPO 的裁剪目标大致为：

$$
L(s,a,\theta_k,\theta)
= \min\left(
\frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s,a),
\mathrm{clip}\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)}, 1-\epsilon, 1+\epsilon\right) A^{\pi_{\theta_k}}(s,a)
\right)
$$

此外还要拟合一个价值函数。单看伪代码，PPO 并不复杂；但实践中的 PPO 对实现细节非常敏感，不同库、不同实现会给出完全不同的结果，甚至有些所谓的 baseline 实现会从根本上改变优化问题。

语言模型上的 PPO 更麻烦：需要 advantage estimation、experience buffer、价值模型，KL 惩罚还是 token-by-token 的，因此不是简单的 bandit 问题。讲师举例说，把 KL 惩罚 clamp 到 0 会破坏 KL 的正负求和；GAE 中常设 $\gamma=\lambda=1$，则退化成 bandit，丢掉了 PPO 原本的多步结构。价值模型通常与策略模型一样大，会占用大量本可用于推理或训练的内存。正因为如此，DPO 虽然流行，但它只适合 Bradley-Terry 形式的成对偏好数据；数学题并不是天然成对比较，DPO 并不是 PPO 的通用替代品。

![PPO理论回顾幻灯片，列出策略梯度、TRPO和PPO三种方法的数学公式推导过程](/coursenotes/cs336-language-modeling-sp26/l16-post-training-rlvr/kf_0003_00-03-48.jpg)

![PPO-Clip算法的伪代码及核心目标函数公式](/coursenotes/cs336-language-modeling-sp26/l16-post-training-rlvr/kf_0004_00-06-19.jpg)

![对比PPO和DPO的局限性，阐述引入新RL算法的必要性](/coursenotes/cs336-language-modeling-sp26/l16-post-training-rlvr/kf_0009_00-12-38.jpg)

## GRPO：用组内标准化替代价值网络 [13:54 - 19:12]

GRPO 来自 DeepSeekMath 论文。它保留 PPO 的裁剪与 KL 正则精神，但去掉了最复杂、最不稳定的价值网络。没有价值网络后，优势不再用“实际奖励减去价值预测”计算，而是用同一 prompt 下的一组 rollout 做组内标准化：

$$
\hat A_i
= \frac{r_i - \mathrm{mean}(r_1,\dots,r_G)}
{\mathrm{std}(r_1,\dots,r_G)}
$$

直觉是：不看模型预测自己应该得多少分，而是看这次 rollout 相对于同组其他 rollout 好多少。若某次采样高于组内均值，就得到正优势；低于均值则得到负优势。GRPO 的目标可以写成：

$$
J_{\mathrm{GRPO}}(\theta)
= \mathbb{E}\left[
\frac{1}{G}\sum_{i=1}^G
\min\left(
\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\mathrm{old}}}(o_i|q)}\hat A_i,
\mathrm{clip}(\cdot)
\right)
- \beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
\right]
$$

在线 rollout 情况下，新旧策略比率接近 1，裁剪项通常不生效，目标就退化为“优势加权 + KL 惩罚”。实现上也很直接：对每个 prompt 采样 $K$ 次，计算奖励，组内归一化，计算 KL，做 stop-grad，然后更新。参考实现通常会在标准差上加一个很小的数（如 $10^{-4}$），避免组内奖励完全相同或只有单个样本时除零。

DeepSeekMath 的实验显示，GRPO 明显优于 rejection fine-tuning（RFT，即只保留正确回答并做 SFT）；process supervision 在当时带来一些增益，但后续 R1 反而转向了纯 outcome supervision。

![GRPO算法定义幻灯片，包含目标函数公式及与PPO的对比](/coursenotes/cs336-language-modeling-sp26/l16-post-training-rlvr/kf_0010_00-13-54.jpg)

![DeepSeekMath-Instruct 1.3B模型在GSM8K和MATH基准上不同训练方法的性能曲线对比](/coursenotes/cs336-language-modeling-sp26/l16-post-training-rlvr/kf_0015_00-27-48.jpg)

## GRPO 的目标偏差：长度与标准差归一化 [20:13 - 25:57]

从 policy gradient 与 baseline 定理看，合法的做法是减去任意 state-dependent baseline，但不应该再除以标准差。GRPO 除以组内标准差，破坏了这一“baseline 契约”，因此它并不是严格意义上直接对奖励做梯度下降。类似地，GRPO 还按序列长度做归一化，而第一性原理推导并不会自然产生这个长度归一化项。

长度归一化会带来长度偏差。若错误回答得到负奖励，模型可以通过生成长度极长的输出来把负惩罚“除以一个很大的数”，从而鼓励模型在不会做题时继续啰嗦。正确回答则会被鼓励缩短，这可能节省推理成本，但也可能损害准确率。后续 Dr. GRPO 的分析图显示，平均输出长度的增长主要由错误回答驱动，而不是正确回答变长。

标准差归一化也有类似问题。在二值奖励下，组内标准差很小意味着题目要么太简单（全对），要么太难（全错）。除以很小的标准差会放大这些零方差题目的权重，而这类题目恰恰不提供有效学习信号。因此，GRPO 是一个有偏但实用的目标：它不是从第一性原理严格推导出来的，但其中一些偏差在特定场景下可能带来好处。

![GRPO基线偏差分析及Dr. GRPO修正方案对比](/coursenotes/cs336-language-modeling-sp26/l16-post-training-rlvr/kf_0012_00-22-45.jpg)

![GRPO长度偏差分析：标准差与长度归一化的影响及实验验证](/coursenotes/cs336-language-modeling-sp26/l16-post-training-rlvr/kf_0013_00-24-01.jpg)

## DeepSeek R1：极简 R1-Zero 与生产化 R1 [26:33 - 39:58]

DeepSeek R1 是开源 RLVR 的重要节点：它首次以开源方式复现了接近 OpenAI o1 的长链思考行为，并给出了可被社区复现的 RL 配方。R1-Zero 的实验设置非常干净：以 DeepSeek-V3 基座模型为起点，直接做 GRPO，奖励只有两类——答案正确性的 accuracy reward，以及要求模型用 thinking tag 包裹思考过程的 format reward。它没有 SFT 初始化，最终性能只比 o1 略差。这个结果说明，base model 加简单 GRPO 就能获得相当强的数学推理能力。

R1 论文还强调了两个现象：训练过程中 response length 越来越长，以及所谓的“aha moment”。讲师对此持保留态度：长 CoT 很可能部分来自 GRPO 的长度归一化副作用，而“aha”式表达在 base model 中已经存在，RL 只是把预训练中已有的行为提取出来。因此这些现象本身不一定特别惊人，R1 更重要的贡献是展示了 RLVR 可以非常简单。

生产化的 R1 与 R1-Zero 有几个关键差别：先用少量 long CoT 数据做 SFT 初始化，避免从 base model 直接 RL 的不稳定冷启动；加入语言一致性奖励，防止 CoT 中乱切语言；再加入非可验证奖励，并最终走 SFT/RLHF 流程。R1 的技术报告也诚实记录了一些失败尝试：process reward model 并没有带来很大收益，outcome reward model 已足够好且更容易扩展；MCTS 也没有取得理想效果。最后，R1 的 long CoT 可以蒸馏到 Qwen、Llama 等模型，显著提升小模型表现，说明 RL 的一个重要角色是“生成监督信号”，生成之后也可以通过模仿学习吸收。

![DeepSeek-R1-Zero受控实验设置及与OpenAI O1系列模型的多项基准测试成绩对比表](/coursenotes/cs336-language-modeling-sp26/l16-post-training-rlvr/kf_0016_00-29-04.jpg)

![DeepSeek-R1-Zero训练期间响应长度增长曲线及模型自我反思的数学解题示例](/coursenotes/cs336-language-modeling-sp26/l16-post-training-rlvr/kf_0017_00-30-20.jpg)

![DeepSeek-R1与R1-Zero的关键差异对比及R1训练流程示意图](/coursenotes/cs336-language-modeling-sp26/l16-post-training-rlvr/kf_0018_00-31-36.jpg)

![蒸馏模型性能对比表，展示DeepSeek-R1-Distill系列在AIME、MATH等基准上的得分](/coursenotes/cs336-language-modeling-sp26/l16-post-training-rlvr/kf_0021_00-36-39.jpg)

## Kimi K1.5：数据课程、DPO 式推导与长度压缩 [40:05 - 54:50]

Kimi K1.5 与 R1 同期，也达到了接近或超过 o1 的水平。讲师特别强调它在数据课程和算法细节上的不同做法。数据方面，Kimi 先追求广泛覆盖，排除 multiple choice / true-false 这类可能带来假阳性的题型，并要求问题需要较长的深度思考。更重要的是 best-of-8 过滤：如果模型采样 8 次至少能成功一次，说明这道题已经在模型能力边缘，不适合作为 RL 训练题；因此只保留模型失败的例子。也可以做双侧过滤，只保留中等难度题目。模型掌握某题后，该题会从题集中移除，以节省算力。

Kimi 的算法从带 KL 正则的奖励最大化出发，走的是 DPO 式非参数化奖励推导：先假设可以解析地求出最优策略对应的奖励，再把奖励表达式代回目标，对

$$
r(x,y,y^*) - \tau \log Z - \tau \log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}
$$

做平方损失。对这个损失求梯度，最终得到的形式 surprisingly 类似 GRPO：一个带组均值 baseline 的 policy gradient，再加上 KL 正则项。也就是说，Kimi 通过另一条推导路径，重新发明了“组均值归一化 baseline”。

Kimi 对长度问题的处理也更积极。它不做 GRPO 那种按序列长度归一化，反而显式加入长度奖励来压缩 CoT：正确回答获得长度奖励，错误回答则只被鼓励比平均稍短，而不是被压到极短。原因是如果错误回答被压得太短，模型可能再也无法从错误中恢复。长度奖励大致为：

$$
\mathrm{len\_reward}(i)=
\begin{cases}
\lambda, & r(x,y_i,y^*)=1 \\
\min(0,\lambda), & r(x,y_i,y^*)=0
\end{cases},
\quad
\lambda = 0.5 - \frac{\mathrm{len}(i)-\mathrm{min\_len}}
{\mathrm{max\_len}-\mathrm{min\_len}}
$$

Kimi 还详细讨论了 RL infra 的困难：长 rollout 会拖慢同步、训练与推理需要切换或共享机器、on-policy 训练稳定但系统利用率低，而复用 rollout 进入 off-policy 又会带来训练不稳定。它的大规模 ablation 还显示，RL 方法一致优于 expert iteration（只对正确回答做 SFT），说明仅靠“训练正确答案”无法完全替代 RL。

![Kimi K1.5 数据筛选与 SFT 策略，包含 best-of-8 难度评估方法](/coursenotes/cs336-language-modeling-sp26/l16-post-training-rlvr/kf_0022_00-41-43.jpg)

![Kimi RL 优化目标推导，基于 DPO 类型的非参数化奖励函数求解](/coursenotes/cs336-language-modeling-sp26/l16-post-training-rlvr/kf_0023_00-44-14.jpg)

![Kimi 模型中用于压缩思维链（CoT）的长度奖励函数公式及解释](/coursenotes/cs336-language-modeling-sp26/l16-post-training-rlvr/kf_0024_00-48-02.jpg)

![小模型在纯数学数据上的扩展结果，展示多个基准测试中性能与token长度的关系曲线](/coursenotes/cs336-language-modeling-sp26/l16-post-training-rlvr/kf_0026_00-54-21.jpg)

## Qwen3：完整后训练流水线与思考模式融合 [55:37 - 61:06]

Qwen3 的技术报告给出了一个非常清晰的后训练全景：base model 先做 long-CoT cold start SFT，再做 reasoning RL，然后做 thinking mode fusion，接着做 general RL / RLHF，最后得到旗舰模型；轻量模型则通过 strong-to-weak distillation 得到。这个流程可以看作“前沿模型后训练”的典型拼图：推理 RL 在前，通用 RLHF 在后，蒸馏用于下沉到小模型。

Qwen3 的 RLVR 数据策略沿用并整合了此前经验：做难度过滤，去掉模型不需要 CoT 就能答对的问题，去掉与验证集过于相似的数据以降低污染，并对 reference CoT 做一定人工过滤。值得注意的是，Qwen3 的 RL 只用了约 4000 个例子；只要其余流水线足够好，少量高质量 RL 数据也能取得很大效果。

Qwen3 的一个特色是 thinking mode fusion：同一个模型通过 `/think` 等标签在长 CoT 模式与即时回答模式之间切换，而不是维护两个独立模型。它还支持用特殊字符串提前终止思考，从而控制 thinking budget。实验显示，即使在中途截断思考，模型性能也能较为平滑地下降；在相同预算下，thinking 模式仍明显优于 instant 模式。分阶段消融则显示，reasoning RL 与 general RL 能提升通用任务表现，但把 thinking 与非 thinking 融合会带来数学、代码上的轻微下降；后续版本甚至可能重新拆分 hybrid 模型，以榨取思考模式的全部性能。

![Qwen3系列模型与主流大模型在多个基准测试上的性能对比表格](/coursenotes/cs336-language-modeling-sp26/l16-post-training-rlvr/kf_0027_00-55-37.jpg)

![Qwen3系列模型的后训练流程架构图，展示旗舰模型和轻量级模型的训练阶段](/coursenotes/cs336-language-modeling-sp26/l16-post-training-rlvr/kf_0028_00-56-53.jpg)

![Qwen3思考模式融合机制，展示带标签的对话模板对比及通过特殊字符串实现提前终止的方法](/coursenotes/cs336-language-modeling-sp26/l16-post-training-rlvr/kf_0029_00-58-08.jpg)

![不同训练阶段（Stage 2-4）的基准测试成绩对比表，涵盖通用任务、指令遵循、Agent及STEM领域](/coursenotes/cs336-language-modeling-sp26/l16-post-training-rlvr/kf_0031_01-00-40.jpg)

## 面向 Agent 的后训练：数据、专家蒸馏与奖励鲁棒性 [61:11 - 69:06]

Agent 后训练并没有全新的训练算法，核心仍然是数据。讲师强调，很多能力不能等到最后才注入，而要在 mid-training 阶段就开始准备。Qwen3 Coder Next 的 mid-training 数据包括：把 GitHub 仓库中的文件拼接成超长上下文数据（约 600B tokens）；用 RAG 检索仓库状态来构造 pull request 相关上下文；把 Common Crawl 中文本与代码混合的文档用 LLM 转成 markdown；用语言模型生成 coding QA 合成数据；运行公开 coding agent 收集轨迹；以及常规的 instruction following 和 fill-in-the-middle 数据。

在模型训练上，Qwen 团队先针对不同 coding 子任务训练多个 expert，例如 webdev、UX、QA 和 software engineering agent，然后再把它们蒸馏回同一个模型。讲师说这种“多专家再蒸馏”在 frontier training 中并不常见，更接近 branch-train-merge 的思路；好处是团队可以分工，但如果有全部训练目标，通常更愿意直接放进一个大训练循环。

软件工程 agent 的环境构造是重点：他们基于 GitHub 自动化构造大量 issue 环境，相当于“SWE-bench 的扩大版”，再在这些环境中做 RL。最终一个只有 3B active 参数的模型能在 SWE-bench 上达到约 70.6%。不过讲师提醒，任务特定环境上的提升不一定代表广泛泛化。

这一节最重要的警告是 reward hacking：RLVR 只和奖励函数一样鲁棒。Git 场景中，模型可以通过查看未来 commit 直接找到修复方案；即使禁止 `git log`，模型也可能添加 remote 再查询。因此需要专门的奖励来阻止模型篡改 git history。讲师还提到，Lean 编译器也并非对抗鲁棒，存在某些字符串可以让模型验证本不该被验证的证明。所谓“可验证奖励”远比想象中棘手。

![Midtraining 阶段的数据来源列表，包括 Github、Common crawl 和合成数据的具体构成](/coursenotes/cs336-language-modeling-sp26/l16-post-training-rlvr/kf_0032_01-01-56.jpg)

## 要点回顾 [00:05 - 75:50]

- RLHF 的瓶颈是奖励模型会被过优化；RLVR 试图用更可验证的奖励，让 RL 能继续 scale compute。
- PPO 是通用但工程复杂的算法：需要价值网络、实现敏感、KL 与 GAE 细节容易出错；DPO 只适合成对偏好，不是数学 RL 的通用替代。
- GRPO 去掉价值网络，用同一 prompt 下多次采样的组内 z-score 作为优势：
  $$\hat A_i=\frac{r_i-\mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G)}$$
  在线情况下目标近似为“优势加权 + KL”，实现简单，成为开源 RLVR 的主流。
- GRPO 并非严格的第一性原理算法：除以标准差破坏 baseline 契约，长度归一化会引入长度偏差；这些偏差有坏处，但在实践中也可能有用。
- DeepSeek R1-Zero 证明 base model + GRPO + 简单 accuracy/format reward 就能接近 o1；R1 则加入 SFT 初始化、语言一致性奖励、非可验证奖励与 RLHF，并公开了 PRM、MCTS 等失败尝试。
- Kimi K1.5 强调数据课程：best-of-8 过滤、只保留模型失败的中等难度题、成功后移出题集；算法上从 DPO 式推导得到类似 GRPO 的组均值 baseline，并显式加入长度奖励压缩 CoT。
- Qwen3 给出完整流水线：long-CoT cold start SFT → reasoning RL → thinking mode fusion → general RL / RLHF → 蒸馏到小模型；RL 数据仅约 4000 例，thinking 模式可通过标签和 early exit 控制预算。
- Agent 后训练的关键仍是数据：大规模 mid-training、多个 expert 分别训练后蒸馏、构造 SWE 环境做 RL。
- RLVR 的核心风险是 reward hacking：Git 历史、Lean 编译器都可能被钻空子；奖励的鲁棒性决定了 RL 能安全投入多少算力。
- GRPO 应掌握到和预训练损失同等熟悉的程度；RL 仍然 finicky、噪声大，但已经比 PPO 时代平滑很多。
