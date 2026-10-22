---
title: 符号约定
original: https://visionbook.mit.edu/notations.html
---

# 符号约定

<div class="src-note">
本页为《Foundations of Computer Vision》前言部分的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/notations.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本表定位

全书数学符号的总字典。后续每一章的公式都遵从这里的约定；复习到某章发现符号对不上、或想不起某个记号的含义时，回到本表核对即可。三处最容易踩的坑：图像 $\ell[n,m]$ 与矩阵 $\boldsymbol\ell$ 的索引转置关系、矩阵微积分的梯度布局（梯度与参数形状转置）、概率中随机变量与取值的大小写区分。

## 查阅要点

- 标量小写；向量/矩阵/张量一律粗体；张量多用小写粗体（任意维数）。
- 图像专用 $\ell[n,m]$：$n$ 为列、$m$ 为行，原点在左下角；图像矩阵 $\boldsymbol\ell$ 为 $M\times N$——两者索引转置。
- 离散信号用方括号 $x[n]$，连续信号用圆括号 $x(t)$；卷积 $\circ$、互相关 $\star$、DFT $X[u]$。
- 机器学习：$J$ 总损失、$\mathcal{L}$ 单样本损失、$\theta$ 可学习参数。
- 概率：非粗体大写作随机变量、小写作取值；$p(\mathbf{x}\mid\ldots)$ 是 $p(X=\mathbf{x}\mid\ldots)$ 的简写。
- 矩阵微积分：向量是列向量 $[N\times1]$；$\partial y/\partial\mathbf{x}$ 为 $[1\times N]$；$\partial\mathbf{y}/\partial\mathbf{x}$ 为 $[M\times N]$；$\partial\mathcal{L}/\partial\mathbf{W}$ 为 $[M\times N]$，与 $\mathbf{W}$ 的 $[N\times M]$ 转置。

## 符号表

> 以下为全书通用符号定义，便于对照阅读时快速查阅。

### 总则（General Notation）

#### 基本对象

| 记法 | 含义 |
|---|---|
| $x$、$y$、$z$ | 标量（scalar） |
| $\mathbf{x}$、$\mathbf{y}$、$\mathbf{z}$ | 向量（vector）；粗体字母用于表示向量、矩阵和张量 |
| $x_i$、$x_j$、$y_i$，或 $x[i]$、$x[j]$、$y[i]$ | 向量的第 $i$（或 $j$）个元素 |
| $\mathbf{X}$、$\mathbf{Y}$、$\mathbf{Z}$ | 矩阵（matrix）；同样用粗体字母表示 |
| $X_{ij}$、$Y_{jk}$、$Z_{ii}$，或 $X[i,j]$、$Y[j,k]$、$Z[i,i]$ | 矩阵元素 |
| $\mathbf{X}_i$ 或 $\mathbf{X}[i,:]$；$\mathbf{X}[:,j]$ | 矩阵切片：第 $i$ 行 / 第 $j$ 列 |

#### 数据集合与乘法运算

| 记法 | 含义 |
|---|---|
| $\{x^{(i)}\}_{i=1}^N$、$\{\mathbf{x}^{(i)}\}_{i=1}^N$、$\{\mathbf{X}^{(i)}\}_{i=1}^N$ | $N$ 个数据点的集合；上标 $(i)$ 为样本编号 |
| $\mathbf{x}^\mathsf{T}\mathbf{y}$ | 点积（dot product） |
| $\mathbf{A}\mathbf{B}$ | 矩阵乘积（matrix product） |
| $\mathbf{x} \odot \mathbf{y}$、$\mathbf{A} \odot \mathbf{B}$ | Hadamard 积（逐元素积，element-wise product） |
| $ab$ 或 $a*b$ | 两个标量相乘 |

### 信号处理（Signal Processing）

| 记法 | 含义 |
|---|---|
| $\mathbf{x}$、$\mathbf{y}$、$\mathbf{z}$ | 离散信号（含图像）视为向量 |
| $x[n]$ | 离散信号索引：$n$ 为离散下标，索引用方括号 |
| $x(t)$ | 连续信号：$t$ 为连续变量 |
| $\circ$ | 卷积算子（convolution operator） |
| $\star$ | 互相关算子（cross-correlation operator） |
| $X[u]$ | 离散傅里叶变换（DFT）：$u$ 为离散频率下标 |

### 图像（Images）

| 记法 | 含义 |
|---|---|
| $\ell[n,m]$ | 图像作为离散 2D 信号：$n$ 索引列、$m$ 索引行；原点 $n=m=0$ 位于图像左下角（除非另作约定）。图像大小为 $N \times M$：$N$ 列、$M$ 行 |
| $\boldsymbol\ell$ | 图像作为矩阵：大小 $M \times N$（$M$ 行、$N$ 列），可索引为 $\ell_{ij}$ |
| $\mathscr{L}[u,v]$ | 图像的傅里叶变换：$u$、$v$ 为空间频率（spatial frequency） |

### 机器学习（Machine Learning）

| 记法 | 含义 |
|---|---|
| $J$ | 总损失函数 / 代价函数 / 目标函数（total loss / cost / objective function） |
| $\mathcal{L}$ | 单数据点损失函数（per-datapoint loss function） |
| $\theta$ | 通用的可学习参数（generic learnable parameters） |

## 复习自检清单

- ☐ 看到 $\boldsymbol\ell[n,m]$ 能立即说出 $n$、$m$ 谁是列谁是行、原点在图像哪个角、图像尺寸怎么写
- ☐ 能解释为什么图像矩阵 $\boldsymbol\ell$（$M\times N$）与离散图像 $\ell[n,m]$（$N\times M$）的索引是转置的
- ☐ 能说出 $x[n]$ 与 $x(t)$、$\circ$ 与 $\star$、$X[u]$ 各自的含义与括号约定
- ☐ 能区分 $J$、$\mathcal{L}$、$\theta$，并说明 $\theta$ 包含哪些量
- ☐ 能写出 $\partial y/\partial\mathbf{x}$（$[1\times N]$）、$\partial\mathbf{y}/\partial\mathbf{x}$（$[M\times N]$）、$\partial\mathcal{L}/\partial\mathbf{W}$（$[M\times N]$，与 $\mathbf{W}$ 转置）的形状
- ☐ 能解释 $p(\mathbf{x}\mid\ldots)$ 为什么是 $p(X=\mathbf{x}\mid\ldots)$ 的简写，以及随机变量/取值的大小写约定
- ☐ 能说明卷积层特征图张量 $\mathbf{x}\in\mathbb{R}^{C\times N\times M\times\ldots}$ 的第一维为什么是通道，以及 Transformer 的 token 矩阵记法有何不同

---

原文：[Notation](https://visionbook.mit.edu/notations.html)
