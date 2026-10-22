---
title: 第 43 章 · 单图深度估计
chapter: 43
original: https://visionbook.mit.edu/3d_learning.html
---

# 第 43 章 · 单图深度估计

<div class="src-note">
本页为《Foundations of Computer Vision》第 43 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/3d_learning.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

前面章节研究图像形成，以及如何用透视投影加一些假设从图像估计三维结构——可以用多幅图像，也可以用单幅图像加单视图度量规则。本章换一条路：不建立精确模型去整合线索，而是用学习的方法绕过建模，从数据中学如何从单幅图像恢复三维结构——把深度估计表述为回归问题。核心难点在于训练数据从哪来，因此本章依次讨论单目深度线索、三维表示、监督与无监督方法。

## 本篇要点

- 单目深度线索：明暗（shape from shading）、纹理梯度、阴影/相互反射、空气透视、已知大小物体、线性透视——多数只能给相对深度。
- 深度图 → 点云：$\mathbf{P}=z(\mathbf{p})\mathbf{K}^{-1}\mathbf{p}$；深度图惯用 turbo 色板可视化。
- 监督方法 = 像素级回归；损失要从 L2 出发处理尺度歧义（尺度不变损失）与远点误差（log 深度/视差）。
- 无监督方法把「深度 + 相机运动估计」做成图像预测问题：用估计的深度与运动把参考帧扭曲到目标帧，以光度一致性为唯一监督。
- 纯旋转相机只能诱导出单应，学不到深度——无监督训练需要复杂相机运动与丰富三维结构。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 3 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/learning_3d/from_depthmap_2_pointcloud.webp" alt="图 43.3 深度图到点云" loading="lazy" />
<figcaption>图 43.3：把深度图转成点云。这需要标定相机（即知道内参）。草图显示像素 (n,m) 如何用该位置的深度 z[n,m] 映射为三维点 P；彩色线分别平行于相机坐标系各轴方向。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这张图是「2.5D 输出如何变回 3D」的标准动作，式 (43.1) 在第 40 章（立体）、本章（深度回归）与后续章节反复出现；深度估计质量的最终检验，就是把结果投成点云看。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/learning_3d/unsupervised_depth.webp" alt="图 43.7 无监督学习系统" loading="lazy" />
<figcaption>图 43.7：无监督学习相机运动与深度的系统 [20]。目标是通过最小化重建误差学习函数 f_1、f_2 的参数。函数 f_3 是确定性的。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这张框图是自监督深度估计（Monodepth 一族）的模板——深度网络 + 位姿网络 + 可微扭曲，靠光度损失闭环；看懂它就看懂了整个无监督路线。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/learning_3d/geometry_reconstruction_12.webp" alt="图 43.8 可微扭曲的几何" loading="lazy" />
<figcaption>图 43.8：运动相机捕获的两幅图像之间的几何。两帧间对应点的坐标由点的三维世界坐标与相对相机运动约束。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：无监督深度的全部监督信号就压在这张图的几何里——深度错一点，扭曲位置就偏，光度误差立刻变大；这条链也是端到端反传能流进深度网络的原因。</p>

## 本章小结

- 单目深度线索：明暗、纹理梯度、阴影/相互反射、空气透视、已知大小物体、线性透视；除已知大小物体外多给相对深度。
- 深度图与点云通过 $\mathbf{P}=z(\mathbf{p})\mathbf{K}^{-1}\mathbf{p}$ 互转；点云无连接关系，成网格需先解决知觉组织问题。
- 监督 = 像素级回归：数据来自 RGB-D/LiDAR（NYUv2、KITTI）、立体对或多视 SfM（MegaDepth）；损失需处理尺度歧义与大深度误差。
- 架构上编码器-解码器 + 跳跃连接已足，性能主要靠数据规模；质量评估要把深度图转成点云换视点看。
- 无监督用「深度 + 位姿 → 可微扭曲 → 光度一致」闭环，监督信号来自视频本身；纯旋转运动学不到深度。

## 复习自检清单

- ☐ 能列举本章六种单目深度线索，并指出哪些只能给相对深度、谁能给绝对深度
- ☐ 能写出深度图到点云的变换（式 43.1）及其非齐次形式，说明 $a$、$c_x$、$c_y$ 的作用
- ☐ 能解释为何点云转网格需要先解决知觉组织问题
- ☐ 能写出监督设定的回归形式与参数优化目标（$h_\theta$ 与 argmin）
- ☐ 能写出尺度不变损失，说明它消去了什么、$\alpha$ 怎么算
- ☐ 能说明为何要优化 log 深度或视差而不是深度本身
- ☐ 能复述无监督方法的五个扭曲步骤，并写出合成公式 $\ell_{1\rightarrow2}(\mathbf{p})=\ell_1(\mathbf{K}\hat{\mathbf{M}}\hat{z}(\mathbf{p})\mathbf{K}^{-1}\mathbf{p})$
- ☐ 能解释损失中掩码 $v(n,m)$ 的两种作用，以及为什么纯旋转相机训练不出深度

---

原文：[Learning to Estimate Depth from a Single Image](https://visionbook.mit.edu/3d_learning.html)
