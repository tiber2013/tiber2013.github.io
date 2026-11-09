---
title: "CS231n 2026 新增课节资料汇编：Lecture 17 World Modeling"
description: "Spring 2026 相对 2025 新增的课节、官方资料缺口、同主题公开讲座、第三方笔记与论文清单。"
---

# CS231n Spring 2026 新增课节资料汇编

> 资料核查日期：2026-11-09 ｜ 本文只做资料索引与事实核对，不猜测未公开课程讲了什么

**核心结论**：两版课表都是 18 讲，讲次数量没有变化；真正「多出来」的是一个全新主题——2026 年的 Lecture 17「World Modeling」由 Gordon Wetzstein 客座，替换了 2025 年同号的「Robot Learning」（Yunzhu Li）。这一讲和 L18 都**没有公开 slides**（官方 PDF 实测返回 404），2026 年的录影也只在 Stanford Canvas 校内开放。校外读者能拿到的最接近材料，是同一位讲者 2026 年秋季在 Stanford EE292R 的公开讲座《From World Models to World Agents》（54 分钟），以及第三方按 L17 主题整理的世界模型论文清单；此外 2026 的 L1–L16 课件全部公开，相对 2025 每讲都有内容改动，本文一并给出逐讲增量核对。

## 1. 「多出来的课节」到底是什么

先纠正一个容易误解的地方：2026 并没有比 2025 **多出讲次**，而是把第 17 讲整个换了主题和讲者，同时 L1–L16 的课件内容大面积更新。

| 讲次 | Spring 2025 | Spring 2026 | 变化 |
| --- | --- | --- | --- |
| L01 | Introduction | Introduction | 课件拆成 part 1 / part 2 两个 PDF |
| L02 | Image Classification with Linear Classifiers | 同 | 子题一致，课件内容更新 |
| L03 | Regularization and Optimization | 同 | 子题一致，课件内容更新 |
| L04 | Neural Networks and Backpropagation | 同 | 子题一致，课件内容更新 |
| L05 | Image Classification with CNNs | 同 | 子题一致，课件内容更新 |
| L06 | CNN Architectures | 同 | 子题一致，课件内容更新 |
| L07 | Recurrent Neural Networks | 同 | 子题一致，课件内容更新 |
| L08 | Attention and Transformers | 同 | 子题一致，课件内容更新 |
| L09 | Object Detection, Image Segmentation, Visualizing and Understanding | 同 | 子题一致，课件内容更新 |
| L10 | Video Understanding | 同 | 2026 课表明确列出 Multimodal video understanding |
| L11 | Large Scale Distributed Training | 同 | 2026 子题改为 Utilization, Parallelism, and Activation Checkpointing |
| L12 | Self-supervised Learning | 同 | 2026 子题含 Multisensory supervision |
| L13 | Generative Models 1 | 同 | 2026 明确列出 GAN 与 Autoregressive Models |
| L14 | Generative Models 2 | 同 | 2026 子题收敛为 Diffusion models |
| L15 | 3D Vision | 同 | 2026 子题含 Neural implicit representations |
| L16 | Vision and Language | Vision and Language | 课件标题改为 Vision + Language (and Foundation Models)，改由课程组主讲 |
| **L17** | **Robot Learning**（Yunzhu Li 客座） | **World Modeling**（Gordon Wetzstein 客座） | **整讲替换，是唯一的新增主题** |
| L18 | Human-Centered AI | Human-Centered AI | 标题相同，两版都没有公开 slides |

结论：按「课节」算，新增内容集中在这一讲的 **World Modeling**；按「资料」算，2026 公开的 L1–L16 课件本身也是相对 2025 的增量，见第 7 节。

## 2. 官方可确认的信息：L17 只有课表一行

[Spring 2026 官方课表](https://cs231n.stanford.edu/schedule.html) 在 5 月 28 日这一行只有两条文字：

- `Lecture 17: World Modeling`
- `Guest Lecturer: Prof. Gordon Wetzstein`

同一天 Assignment 3 截止。**没有子题列表、没有 slides 链接、没有 suggested readings**。这是校外读者能确认的全部内容。

本次实测的官方资料状态：

| 资源 | 状态 | 说明 |
| --- | --- | --- |
| `slides/2026/lecture_17.pdf` | 404 | 官方未发布 L17 课件 |
| `slides/2026/lecture_18.pdf` | 404 | L18 同样没有课件 |
| `slides/2026/lecture_16.pdf` | 200 | 2026 课件止于 L16 |
| `slides/2025/lecture_17.pdf` | 200 | 2025 旧主题 Robot Learning 的 103 页课件 |
| `slides/2026/section_3_project.pdf` | 200 | 期末专题说明，非 L17 内容 |
| 2026 课程录影 | 仅 Canvas | 校内选课学生可见；公开播放列表只有 Spring 2025 |

第三方核对也得到同样结论：quidproquo.cc 在 2026-09-30 分别试过 L17、L18 的 PDF 地址，均返回 404，并特意注明「不猜这堂讲了什么」。

所以下面所有材料，都标注了它与 L17 的距离，请勿把替代材料当成课程内容。

## 3. 最接近的公开替代：Wetzstein《From World Models to World Agents》

> ⚠️ 这是**同一位讲者在另一门课的公开讲座**，不是 CS231n L17 的录影，不能当作 2026 L17 的课堂内容引用。

- 场合：Stanford [AI + Data for Science（EE292R）Fall 2026](https://aida4sci.stanford.edu/schedule) 研讨会，讲者 Gordon Wetzstein（Stanford）。官方课表列 9 月 30 日，讲座索引页标注视频日期 2026-10-22，时长 54 分钟。
- 官方摘要要点：生成式视频模型要成为真正的 world model（稳定、可控、长时程的模拟器）仍是 Spatial AI 的重大挑战；讲座聚焦**长上下文的时序一致性与推理、缓解自回归漂移、多智能体动态、以人为中心的条件控制**，并说明如何用 scaling laws 借世界模型训练鲁棒策略。
- 视频：[YouTube · From World Models to World Agents](https://www.youtube.com/watch?v=IfkeGU4bEJI)（链接来自第三方讲座索引页；本次核查环境未能直连 YouTube 验证，按索引信息记录）
- 讲座索引与时间轴：[PostSingularity Institute 讲座页](https://postsingularityinstitute.com/en/syntheses/UCBa5G_ESCn8Yd4vw5U-gIcg_IfkeGU4bEJI)

索引页根据字幕整理的内容时间轴：

| 时间 | 内容 |
| --- | --- |
| [02:00](https://www.youtube.com/watch?v=IfkeGU4bEJI&t=120s) | 世界模型导论；Genie3 作为实时动作条件视频生成器 |
| [05:00](https://www.youtube.com/watch?v=IfkeGU4bEJI&t=300s) | 应用：自动驾驶数据增广（Tesla、Waymo）、机器人与人的训练环境 |
| [10:00](https://www.youtube.com/watch?v=IfkeGU4bEJI&t=600s) | 传统模拟器 vs 学习式世界模型：状态可见性、可编辑性、真实感 |
| [16:00](https://www.youtube.com/watch?v=IfkeGU4bEJI&t=960s) | 扩散模型基础：加噪、去噪、潜在空间与条件机制 |
| [22:00](https://www.youtube.com/watch?v=IfkeGU4bEJI&t=1320s) | 视频世界模型定义：实时、动作条件的自回归生成器；开环与闭环 |
| [28:00](https://www.youtube.com/watch?v=IfkeGU4bEJI&t=1680s) | 长上下文难题：token 上限、注意力二次复杂度、Oasis 镜头转动后失效 |
| [34:00](https://www.youtube.com/watch?v=IfkeGU4bEJI&t=2040s) | 人类记忆类比：不同时间尺度与混合专家 |
| [38:00](https://www.youtube.com/watch?v=IfkeGU4bEJI&t=2280s) | FramePack：对历史帧做时空下采样以压缩上下文 |
| [42:00](https://www.youtube.com/watch?v=IfkeGU4bEJI&t=2520s) | FramePack 效果：时序一致性与身份保持 |

涉及的技术与项目：Genie3、Oasis、FramePack、扩散模型、world action models、长上下文一致性、自回归漂移、physical AI。

## 4. 2025 同讲次资料：主题不同，只作对照

2026 把 L17 换掉之前，这个位置是 Yunzhu Li（Columbia）客座的 **Robot Learning**，资料齐全但主题不同：

- [官方 slides（103 页 PDF）](https://cs231n.stanford.edu/slides/2025/lecture_17.pdf)
- [录影（约 1 小时 18 分）](https://www.youtube.com/watch?v=XSfmOH_xVSU)
- 本站笔记：[L17 · 机器人学习：从感知-动作循环到机器人基础模型](/coursenotes/cs231n-deep-learning-for-cv-sp25/l17-robot-learning)

一条值得记的衔接线索：2025 课件最后一页标题是「Foundation Policy → Foundation World Models」，把 world model 定义为 **action-conditioned future prediction**，并列举 1X World Models、DayDreamer、NVIDIA Cosmos。第三方核对过录影字幕，确认字幕里没有这个定义和这三个例子，属于 slides 独有的内容。这条线索说明课程从机器人学习转向世界建模并非毫无铺垫，但**不能据此推断 2026 L17 讲了什么**。

## 5. 第三方整理与导读

| 资料 | 日期 | 内容 | 性质 |
| --- | --- | --- | --- |
| [quidproquo.cc · 中文](http://quidproquo.cc/posts/ai/2026-09-30-cs231n-world-models-hcai-final-project) | 2026-09-30 | 明确区分 2026 L17/L18 与 2025 替代材料，附 2025 L17 课件逐段梳理、期末专题规则 | 第三方导读，明确「不猜 2026 内容」 |
| [quidproquo.cc · English](https://quidproquo.cc/posts/ai/2026-09-30-cs231n-world-models-hcai-final-project-en) | 2026-09-30 | 同上英文版 | 第三方导读 |
| [Clawdemy · World modeling references](https://clawdemy.org/lessons/computer-vision/world-modeling/references) | 2026-11-07 更新 | 按 CS231n L17 主题组织的世界模型论文清单，声明不复制课件 | 第三方主题汇编 |
| [本站 CS231n 18 讲笔记](/coursenotes/cs231n-deep-learning-for-cv-sp25/) | 2026-11 | 基于 Spring 2025 公开录影重写的中文图文笔记 | 本站内容，2025 视频版 |

## 6. 世界模型主题参考文献

> 以下清单来自 Clawdemy 按 CS231n L17 主题汇编的参考文献页，**不是课程官方指定的 reading list**；官方课表在 L17 下没有列任何建议阅读。

**早期世界模型**

- Ha & Schmidhuber, [World Models](https://arxiv.org/abs/1803.10122)（NeurIPS 2018）：VAE + RNN + controller，确立「想象未来再规划」的框架。
- Hafner et al., [Learning Latent Dynamics for Planning from Pixels (PlaNet)](https://arxiv.org/abs/1811.04551)（ICML 2019）：潜在空间动态与规划。

**Dreamer 系列**

- [Dream to Control (Dreamer)](https://arxiv.org/abs/1912.01603)（ICLR 2020）
- [Mastering Atari with Discrete World Models (DreamerV2)](https://arxiv.org/abs/2010.02193)（ICLR 2021）
- [Mastering Diverse Domains through World Models (DreamerV3)](https://arxiv.org/abs/2301.04104)（Nature 2023）

**规划与表征**

- Schrittwieser et al., [MuZero](https://arxiv.org/abs/1911.08265)（Nature 2020）：用学习到的动态模型做树搜索。
- Assran et al., [I-JEPA](https://arxiv.org/abs/2301.08243)（CVPR 2023）；Bardes et al., [V-JEPA](https://arxiv.org/abs/2404.08471)（TMLR 2024）：联合嵌入预测式表征。

**视频世界模型**

- Brooks et al., [Video generation models as world simulators (Sora)](https://openai.com/index/video-generation-models-as-world-simulators/)（OpenAI 技术报告 2024）
- Bruce et al., [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391)（ICML 2024）：动作可控的可交互环境生成。

**前驱工作**

- Lotter, Kreiman & Cox, [Deep Predictive Coding Networks for Video Prediction (PredNet)](https://arxiv.org/abs/1605.08104)（ICLR 2017）

## 7. 2026 L1–L16 官方课件与逐讲增量

2026 课件地址形如 `https://cs231n.stanford.edu/slides/2026/lecture_N.pdf`，[官方课表](https://cs231n.stanford.edu/schedule.html) 逐讲给出链接。本次实测 L2–L16 全部可访问（200）；L1 拆成 [part 1](https://cs231n.stanford.edu/slides/2026/lecture_1_part_1.pdf) 与 [part 2](https://cs231n.stanford.edu/slides/2026/lecture_1_part_2.pdf)。本地批次已存档 16 讲 PDF 与解析文本。

为量化两版差异，把 2025、2026 两版课件解析成文本行后做集合差分，口径如下：

- **新增**：只出现在 2026 版课件里的文本行数；
- **移除**：只出现在 2025 版课件里的文本行数。

这是文本行的数量，不是幻灯片页数，也不等于知识点个数；排版改动也会计入。

| 讲次 | 新增 | 移除 | 讲次 | 新增 | 移除 |
| --- | --- | --- | --- | --- | --- |
| L01 | 474 | 475 | L09 | 353 | 607 |
| L02 | 144 | 362 | L10 | 486 | 755 |
| L03 | 218 | 177 | L11 | 526 | 388 |
| L04 | 253 | 284 | L12 | 195 | 226 |
| L05 | 203 | 445 | L13 | 227 | 205 |
| L06 | 237 | 219 | L14 | 375 | 327 |
| L07 | 338 | 368 | L15 | 495 | 537 |
| L08 | 425 | 317 | L16 | 403 | 538 |

改动量最大的是 L10、L11、L15、L16。从 2026 新增文本里可确认的关键词（仅表示课件文本出现，不代表课堂讲解重点）：

- L16：Foundation Models、Qwen、Gemini、SigLIP、World Model、Diffusion
- L14：Foundation Model、Autoregressive、Diffusion
- L15：3D Gaussian、GAN、Autoregressive
- L11：Utilization、Activation Checkpointing
- L13：Autoregressive、Diffusion
- L01：Gemini、Multimodal、GAN、Diffusion

## 8. 资料缺口与获取路径

| 想要的资料 | 现状 | 可行路径 |
| --- | --- | --- |
| 2026 L17 课件 | 官方 404 | 校内 Canvas / Ed；讲者本人或其研究组；暂无公开副本 |
| 2026 L17 录影 | 仅 Canvas | Stanford 校内账号；公开播放列表只有 2025 版 |
| 2026 L17 官方阅读清单 | 课表未列 | 无 |
| 同主题公开讲座 | 有 | 第 3 节的 EE292R 讲座（54 分钟，同一讲者） |
| 2026 L1–L16 课件 | 全部公开 | 官方课表逐讲链接 |
| 2025 L17/L18 | 齐全 | 官方 slides + [2025 播放列表](https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16) |

课件与录影版权归课程方与讲者，本文只列链接、不转载 PDF 与视频文件。

## 资料清单（速查）

- 官方：[Spring 2026 课表](https://cs231n.stanford.edu/schedule.html) ｜ [Spring 2025 课表](https://cs231n.stanford.edu/2025/schedule.html) ｜ [2026 L1–L16 课件入口](https://cs231n.stanford.edu/schedule.html) ｜ [期末专题页](https://cs231n.stanford.edu/project.html) ｜ [Section 3 课件](https://cs231n.stanford.edu/slides/2026/section_3_project.pdf) ｜ [课程笔记站](https://cs231n.github.io/)
- 2025 同讲次：[L17 slides](https://cs231n.stanford.edu/slides/2025/lecture_17.pdf) ｜ [L17 录影](https://www.youtube.com/watch?v=XSfmOH_xVSU) ｜ [L18 录影](https://www.youtube.com/watch?v=g8UaBfj6Sh8) ｜ [播放列表](https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16)
- 讲者公开讲座：[EE292R 课表](https://aida4sci.stanford.edu/schedule) ｜ [视频](https://www.youtube.com/watch?v=IfkeGU4bEJI) ｜ [索引与时间轴](https://postsingularityinstitute.com/en/syntheses/UCBa5G_ESCn8Yd4vw5U-gIcg_IfkeGU4bEJI)
- 第三方整理：[quidproquo 中文](http://quidproquo.cc/posts/ai/2026-09-30-cs231n-world-models-hcai-final-project) ｜ [quidproquo 英文](https://quidproquo.cc/posts/ai/2026-09-30-cs231n-world-models-hcai-final-project-en) ｜ [Clawdemy 参考文献](https://clawdemy.org/lessons/computer-vision/world-modeling/references)

## 要点回顾

- 2026 与 2025 都是 18 讲；唯一新增主题是 L17「World Modeling」，由 Gordon Wetzstein 客座，替换 2025 的 Robot Learning。
- 官方对 L17 只给出标题与讲者，无子题、无 slides、无阅读清单；L17/L18 的 PDF 地址实测 404，2026 录影仅校内 Canvas。
- 最接近的公开替代是同一讲者 2026 秋季 EE292R 讲座《From World Models to World Agents》，54 分钟，讲视频世界模型、长上下文一致性、FramePack 与 world action models——但**不是** CS231n L17 录影。
- 2025 L17 的 Robot Learning 课件与录影齐全，其中「Foundation Policy → Foundation World Models」一页是两版主题衔接的线索。
- 世界模型论文清单（World Models、PlaNet、Dreamer 系列、MuZero、JEPA、Sora、Genie、PredNet）来自第三方主题汇编，非官方指定阅读。
- 2026 L1–L16 课件全部公开，逐讲文本差分显示 L10、L11、L15、L16 改动最大；L16 新增 Foundation Models、Qwen、Gemini、SigLIP 等关键词。
- 站内 18 讲中文笔记对应的是 Spring 2025 录影，不要与 2026 课表条目混用。
- 所有课件与录影只列链接，不转载文件。
