---
title: "Stanford CS231A 计算机视觉：从三维感知到三维重建"
description: "Stanford CS231A Spring 2025 课程笔记：10 篇 Course Notes 中文全译，另补 SfM、NeRF、3D Gaussian Splatting 与客座讲座，含 18 讲课件与官方阅读材料的对照表。"
---

# Stanford CS231A 计算机视觉：从三维感知到三维重建

> **CS231A: Computer Vision — From 3D Perception to 3D Reconstruction and Beyond** ｜ Stanford 工程 1 学分制课程，2025 春季 ｜ 讲者：Silvio Savarese、Jeannette Bohg ｜ 上课时间：周一/周三 13:30–14:50，Gates B1（2025-03-31 开课）

## 这门课和 CS231n 有什么区别

两门课名字只差一个字母，方向却几乎正交。**第 1 讲课件第 24 页自己给出了官方说法**：

> CS231a = Focus on 3D w/ little Semantics
> CS231n = Focus on 2D w/ a lot of Semantics

展开讲：

- **CS231n**（Convolutional Neural Networks for Visual Recognition）是**用深度学习做视觉**：图像分类、损失与优化、反向传播、CNN 架构、检测与分割、视频、自监督、生成模型，再到视觉语言。它的主线是「怎么把网络训好、怎么让模型认出东西」。本站已收录它的 2025 春季 18 讲视频笔记。
- **CS231A** 是**视觉的几何与数学**：相机怎么把三维世界压成二维、一台/两台/许多台相机如何把像素反推回三维、如何鲁棒地拟合模型与匹配观测、如何做递推的状态估计，最后落到辐射场与高斯溅射这两种现代场景表示。它的两段式课程地图写在同一页课件上：**Space/Geometry**（用几何方法估计空间属性）与 **Time/Dynamics + Learning**（用学习方法估计语义与动态属性）。

一个直观的区别：CS231n 里出现 $W x + b$ 是为了学特征；CS231A 里出现 $K[R\ t]$ 是为了算清一个点会落在哪个像素上。

## 这门课的公开材料只有课件和 Course Notes

**CS231A 没有公开的课程录像**（YouTube 上搜不到官方播放列表）。官方公开的原始材料是两类：

- **18 讲 slides**（PDF，其中第 11、12 讲共用一份 161 页的 deck）与 **7 份 TA section 讲义**（第 7、8 次习题课的链接在 syllabus 上已被注释掉，公开版本不存在）；
- **10 篇 Course Notes**（PDF）。前 5 篇署名 Kenji Hata & Silvio Savarese，第 6 篇 Bryan Chiang & Silvio Savarese，第 7–10 篇 Bryan Chiang & Jeannette Bohg，落款都在 2022 年——也就是说官方阅读材料的正文早于 2025 年的课件编排。

所以这批笔记的做法和 CS231n 那批不同：**不是视频笔记，没有时间戳**。10 篇 Course Notes 做**逐节中文全译**（公式按原文保留 LaTeX，编号一一对应）；Course Notes 没覆盖的主题（运动恢复结构的课件处理、NeRF、3D Gaussian Splatting、客座讲座）由我们依据课件页面与原始论文另写。

## 18 讲对照表：课件 · 官方阅读 · 本站笔记

| 日期 | 讲次 | 主题 | Course Notes | 本站笔记 |
| --- | --- | --- | --- | --- |
| 03/31 | 1 | Introduction | — | 并入本页 |
| 04/02 | 2 | Camera Models | Notes 1 | [相机模型](/coursenotes/cs231a-computer-vision-3d-sp25/cn01-camera-models) |
| 04/07 | 3 | Camera Models II and Camera Calibration | Notes 1 | 同上（标定与畸变在第 5–6 节） |
| 04/09 | 4 | Single View Metrology | Notes 2 | [单视图计量](/coursenotes/cs231a-computer-vision-3d-sp25/cn02-single-view-metrology) |
| 04/14 | 5 | Epipolar Geometry | Notes 3 | [对极几何](/coursenotes/cs231a-computer-vision-3d-sp25/cn03-epipolar-geometry) |
| 04/16 | 6 | Stereo Systems | Notes 4 | [立体系统与运动恢复结构](/coursenotes/cs231a-computer-vision-3d-sp25/cn04-stereo-systems) |
| 04/21 | 7 | Structure from Motion | Notes 4 | 同上 + [第 7 讲课件笔记](/coursenotes/cs231a-computer-vision-3d-sp25/l07-structure-from-motion) |
| 04/23 | 8 | Active Stereo & Volumetric Stereo | Notes 5 | [主动立体与体积立体](/coursenotes/cs231a-computer-vision-3d-sp25/cn05-active-volumetric-stereo) |
| 04/28 | 9 | Fitting and Matching | Notes 6 | [拟合与匹配](/coursenotes/cs231a-computer-vision-3d-sp25/cn06-fitting-matching) |
| 04/30 | 10 | Representations & Representation Learning | Notes 7 | [表示与表示学习](/coursenotes/cs231a-computer-vision-3d-sp25/cn07-representations-learning) |
| 05/07 | 11 | Monocular Depth Estimation & Feature Tracking | Notes 8 | [单目深度估计与特征跟踪](/coursenotes/cs231a-computer-vision-3d-sp25/cn08-monocular-depth-tracking) |
| 05/12 | 12 | Learning-based Stereo & Monocular Depth Cont. | Notes 8 | 同上（11、12 讲共用一份 deck） |
| 05/14 | 13 | Optical and Scene Flow | Notes 9 | [光流与场景流](/coursenotes/cs231a-computer-vision-3d-sp25/cn09-optical-scene-flow) |
| 05/19 | 14 | Optimal Estimation | Notes 10 | [最优估计](/coursenotes/cs231a-computer-vision-3d-sp25/cn10-optimal-estimation) |
| 05/21 | 15 | Optimal Estimation Cont.（Flipped Format） | Notes 10 | 同上 |
| 05/28 | 16 | Neural Radiance Fields | — | [NeRF](/coursenotes/cs231a-computer-vision-3d-sp25/l16-neural-radiance-fields) |
| 06/02 | 17 | Gaussian Splatting | — | [3D Gaussian Splatting](/coursenotes/cs231a-computer-vision-3d-sp25/l17-gaussian-splatting) |
| 06/04 | 18 | Guest Lecture（Francis Engelmann） | — | [基础模型遇上 3D 视觉](/coursenotes/cs231a-computer-vision-3d-sp25/l18-guest-lecture-foundation-models-3d) |

## 翻译时发现的几处不一致

这些都不是笔误级别的小事，读课件时会直接影响对应关系，记录在此：

- **课件文件名比讲次滞后一号。** 从第 13 讲开始：第 13 讲的课件文件叫 `Lecture12_Flow.pdf`，第 14 讲叫 `Lecture13_OptimalEstimation.pdf`，第 15 讲叫 `Lecture14_OptimalEstimationCont'.pdf`（文件名里真的带一个撇号）。第 11、12 两讲则共用 `Lecture11_UsingRepresentationLearning.pdf`。判断归属以**标题页署期**为准，我们已逐份核对。
- **syllabus 把第 13 讲标题印成「Optimal and Scene Flow」**，而课件与该讲对应的 Course Notes 9 都是 *Optical* and Scene Flow（光流与场景流），不是最优估计。按内容归到光流这条线。
- **Course Notes 4 的官方标题是 "Stereo Systems"，PDF 页眉却写 "Stereo Systems and Structure from Motion"**，正文实际是三角化 + 仿射/透视 SfM + bundle adjustment，**没有**讲传统双目系统的硬件与视差搜索。第 6 讲课件里的立体系统内容，官方 Course Notes 是缺的。
- **Course Notes 9 标题含 Scene Flow，正文没有场景流章节**：三维运动只出现在运动场的定义里，没有展开估计方法。
- **Course Notes 2 不含 Criminisi 式单视图计量的「高度反推 / 参照物尺度」**，正文止于用三个两两垂直平面的消失点做单幅图像自标定。

## 怎么读这批笔记

想建立几何直觉主线，按 **Notes 1 → 3 → 4 → 10** 读：相机模型、双视图约束、多视图重建与优化、递推状态估计，这四篇撑起这门课的骨架。

Course Notes 的翻译是**全译而非摘要**，公式编号与原文一致，可以直接对着 Stanford 官方 PDF 逐式核对；每篇末尾有要点回顾。译者在核对原页时发现官方 PDF 自身有若干笔误（漏写转置、下标不一致、公式编号错位），都按原文照录并在译注里标明，没有悄悄改成「正确版本」。

本页上方对照表里的 4 篇「本站笔记」中，SfM、NeRF、3D Gaussian Splatting、客座讲座四篇没有 Course Notes 可译，是依据课件页面 + 原始论文写的，引用文献都经过核实。
