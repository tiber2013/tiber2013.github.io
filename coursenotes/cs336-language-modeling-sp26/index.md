---
title: "Stanford CS336 语言模型从零构建"
description: "Language Modeling from Scratch · 2026 春 · Stanford Online —— 课堂视频笔记，逐讲整理要点与板书关键帧。"
---

# Stanford CS336 语言模型从零构建

> Language Modeling from Scratch · 2026 春 · Stanford Online ｜ 共 18 篇 ｜ 约 24 小时课程 ｜ 414 张关键帧

一门把语言模型拆到硬件层的课：分词与架构自己写，GPU/TPU 与 kernel 自己调，多卡并行与缩放定律自己拟合，再把数据、评估、后训练与对齐逐段补齐。期末交付一个从零训练的完整语言模型。

## 讲次目录

| 讲次 | 笔记 | 时长 | 配图 | 原视频标题 |
| --- | --- | --- | --- | --- |
| L01 | [L01 · 课程概览与分词](./l01-overview-and-tokenization) | 79 分钟 | 20 | Lecture 1: Overview, Tokenization |
| L02 | [L02 · PyTorch 与 einops](./l02-pytorch-and-einops) | 77 分钟 | 18 | Lecture 2: PyTorch (einops) |
| L03 | [L03 · 从零设计 Transformer 架构](./l03-architectures) | 89 分钟 | 23 | Lecture 3: Architectures |
| L04 | [L04 · 注意力替代方案与混合专家](./l04-attention-alternatives) | 86 分钟 | 18 | Lecture 4: Attention Alternatives |
| L05 | [L05 · GPU 与 TPU：硬件与性能模型](./l05-gpus-and-tpus) | 79 分钟 | 15 | Lecture 5: GPUs, TPUs |
| L06 | [L06 · Kernel、Triton 与 XLA](./l06-kernels-triton-xla) | 87 分钟 | 22 | Lecture 6: Kernels, Triton, XLA |
| L07 | [L07 · 并行化 I：多卡互连与集合通信](./l07-parallelism-i) | 81 分钟 | 25 | Lecture 7: Parallelism |
| L08 | [L08 · 并行化 II：数据、流水线、张量与专家并行](./l08-parallelism-ii) | 80 分钟 | 30 | Lecture 8: Parallelism |
| L09 | [L09 · 缩放定律](./l09-scaling-laws) | 78 分钟 | 19 | Lecture 9: Scaling Laws |
| L10 | [L10 · 推理](./l10-inference) | 85 分钟 | 30 | Lecture 10: Inference |
| L11 | [L11 · 缩放定律进阶：μP、WSD 与外推](./l11-scaling-laws-in-practice) | 77 分钟 | 30 | Lecture 11:  Scaling Laws |
| L12 | [L12 · 评估](./l12-evaluation) | 79 分钟 | 20 | Lecture 12: Evaluation |
| L13 | [L13 · 数据 I：来源、抓取与合规](./l13-data-sources) | 82 分钟 | 31 | Lecture 13: Data (Sources, Datasets) |
| L14 | [L14 · 数据 II：转换、去重与训练管道](./l14-data-pipelines) | 85 分钟 | 31 | Lecture 14: Data |
| L15 | [L15 · 中期训练与后训练](./l15-mid-and-post-training) | 80 分钟 | 23 | Lecture 15: Mid/Post-Training |
| L16 | [L16 · 后训练进阶：RLVR 与 GRPO](./l16-post-training-rlvr) | 76 分钟 | 21 | Lecture 16: Post-Training - RLVR |
| L17 | [L17 · 对齐与多模态](./l17-alignment-and-multimodality) | 78 分钟 | 23 | Lecture 17: Alignment - Multimodality |
| 客座 | [客座 · 推理系统：把电力变成智能（Dan Fu）](./guest-dan-fu-inference-systems) | 72 分钟 | 15 | Guest Lecture: Dan Fu |

含 17 讲主课与 1 场客座讲座（Dan Fu，推理系统）。
