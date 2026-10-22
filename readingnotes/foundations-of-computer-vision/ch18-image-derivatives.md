---
title: 第 18 章 · 图像导数
chapter: 18
original: https://visionbook.mit.edu/derivatives.html
---

# 第 18 章 · 图像导数

<div class="src-note">
本页为《Foundations of Computer Vision》第 18 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/derivatives.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

本章进入全书的"图像处理与表示"部分，紧接第 17 章空间滤波。图像导数是线性平移不变系统最基础的构造方式：它回答"图像里哪里在变化"，而变化的位置往往对应物体之间的过渡。导数是边缘、角点、纹理描述乃至卷积神经网络第一层的共同根基；本章还把导数当作一种图像表示，展示梯度域编辑、锐化滤波、Retinex 光照/反射分离等应用。

## 本篇要点

- 导数算子线性且平移不变，因此可实现为卷积；最基本的两个离散核是 $d_0=[1,-1]$（输出右移半个采样）与 $d_1=[1,0,-1]/2$（输出对齐、带通）。
- 高斯导数利用"卷积与求导可交换"，把求导转移到核上：既抗噪，又能处理连续域中不可微的点；$n$ 阶高斯导数 = Hermite 多项式 × 高斯。
- Sobel-Feldman 算子 = 居中导数核 $[1,0,-1]$ 与二项平滑 $[1,2,1]$ 的张量积，抗噪最好、最接近旋转不变；Roberts cross 用旋转坐标系消除两方向输出之间的空间错位。
- 拉普拉斯（Laplacian）= 二阶偏导之和：旋转不变、度量曲率、零交叉构成闭合轮廓；高斯拉普拉斯（LoG）形如"倒墨西哥帽"。
- 梯度域表示可用伪逆（pseudoinverse）积分重建（丢失直流分量，只能恢复到相差一个常数）；把梯度置零即可做图像修复（inpainting）；阈值化 log 梯度可分离光照与反射（Retinex）。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 9 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/derivatives/discretederivative.webp" alt="图 18.3 两种离散导数核及其输出" loading="lazy" />
<figcaption>图 18.3：(a) 输入信号 ℓ[n]；(b) 卷积核 d_0[n]，定义为 d_0[0]=1、d_0[1]=-1、其余为 0；(c) ℓ[n] 与 d_0[n] 卷积的输出；(d) 卷积核 d_1[n]，定义为 d_1[-1]=1、d_1[1]=-1、其余为 0；(e) ℓ[n] 与 d_1[n] 卷积的输出。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这两个核和它们的输出是本章的地基——"核是否居中"直接决定输出有没有半采样错位，后面所有导数核的选择都从这个观察出发。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/derivatives/d0andd1_dft.webp" alt="图 18.4 D₀ 与 D₁ 的 DFT 幅度对比" loading="lazy" />
<figcaption>图 18.4：(a) D_0[u] 与 (b) D_1[u] 的幅度，并与 |2π u/N|（细黑线）比较。两个 DFT 均在 20 个采样上计算。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：频域视角一眼看清两种离散导数的取舍——D_0 幅度更接近理想微分但引入相移；D_1 无相移但有效频带更窄（相当于先叠了一层二项平滑）。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/derivatives/stop_noise.webp" alt="图 18.6 噪声与高斯导数" loading="lazy" />
<figcaption>图 18.6 (a)：被高斯噪声污染的输入图像。</figcaption>
</figure>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/spatial_filters/gausian_der_c_2.webp" alt="图 18.8 多尺度高斯导数" loading="lazy" />
<figcaption>图 18.8 (a)：细尺度（σ 较小）的 x 导数，突出斑马条纹等细节。</figcaption>
</figure>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/spatial_filters/gauss_derivatives_triangle.webp" alt="图 18.10 直至 6 阶的 2D 高斯导数核" loading="lazy" />
<figcaption>图 18.10：直至 6 阶的高斯导数核。所有核都是可分离的；它们看起来像傅里叶基乘以高斯窗。图 18.11 表明它们与正弦、余弦波不同，更像余弦波与正弦波的乘积。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：把"高斯导数族 = Hermite 多项式 × 高斯"直接可视化；不同阶数响应不同特征（条纹、角点等），但这种表示不随图像旋转而不变。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/spatial_filters/DFTderivativeoperators.webp" alt="图 18.14 四种导数离散化的 DFT" loading="lazy" />
<figcaption>图 18.14：四种导数离散化方式 DFT 的幅度：(a) d_0；(b) d_1；(c) Roberts cross 算子；(d) Sobel-Feldman 算子。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：四种常用导数核的频域对比就是选型依据——Sobel 因带通特性对噪声容忍度最好，d_0 输出分辨率最高，Roberts cross 是旋转版的 d_1。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/spatial_filters/circle.webp" alt="图 18.15 各导数核对圆形的方向响应" loading="lazy" />
<figcaption>图 18.15：圆形沿 n、m 与 45° 方向的导数（只在幅度 >0 处显示角度）。45° 方向的导数输出由沿 n、m 的导数输出线性组合得到。注意不同核之间的差异：Sobel 算子给出最接近旋转不变的梯度幅度，但更模糊。</figcaption>
</figure>
<p class="fig-note">💡 实战提示：工程里最常用的是 Sobel（如 OpenCV `cv2.Sobel(ksize=3)`）；Prewitt 算子可看作把 Sobel 的二项平滑 [1,2,1] 换成盒式平滑 [1,1,1] 的变体，抗噪更弱。若两个方向的导数用了偶数长核（如 [1,-1]），输出会相对输入错位半像素；计算梯度幅度/方向或与其他特征对齐时，这种半像素错位会引入可见偏差，应优先使用居中核或显式做对齐。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/derivatives/hat.webp" alt="图 18.16 高斯拉普拉斯（倒墨西哥帽）" loading="lazy" />
<figcaption>图 18.16：高斯拉普拉斯（σ=1），又称倒置的墨西哥帽小波。(a) 二维图；(b) y=0 处的一维截面。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：中心为正、四周为负的结构决定了它对"与周围不同的斑块"响应最强，这是角点检测与 DoG/SIFT 的直觉源头。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/derivatives/discretelaplacian1d.webp" alt="图 18.19 一维导数与拉普拉斯" loading="lazy" />
<figcaption>图 18.19：一维信号的一阶导数与拉普拉斯输出比较。(a) 输入信号；(b) 核 d_1；(c) 导数输出，即 (a) 与 (b) 的卷积；(d) 拉普拉斯的离散近似；(e) 信号 (a) 与拉普拉斯核卷积的输出。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：一维原型讲清两种边缘定位方式——导数的局部极大与拉普拉斯的零交叉都精确落在信号跳变处，这是理解所有边缘检测器的起点。</p>

## 本章小结

- 导数是线性平移不变算子，可用卷积实现；两个基本离散核：$d_0=[1,-1]$（半采样错位、幅度最接近理想微分）与 $d_1=[1,0,-1]/2$（输出对齐、带通更窄）。
- 高斯导数利用"卷积与求导可交换"把求导移到核上：抗噪、可应对不连续点；高阶导数是 Hermite 多项式 × 高斯，尺度 $\sigma$ 控制细节与噪声的取舍；两个高斯导数卷积仍是高斯导数（阶数相加、方差相加）。
- Sobel-Feldman = 居中导数核 $[1,0,-1]$ 与二项平滑 $[1,2,1]$ 的张量积，抗噪最好且最接近旋转不变；Roberts cross 用旋转坐标系消除空间错位。
- 拉普拉斯 = 二阶偏导之和：旋转不变、度量曲率、零交叉形成闭合轮廓；LoG 是"倒墨西哥帽"，带通特性也是早期视觉系统模型 $h=-\nabla^2 g + \lambda g$ 的基础。
- 梯度域表示可用伪逆积分重建（丢失直流分量，只能恢复到相差一个常数）；把梯度置零即可做 inpainting；阈值化 log 梯度可分离光照与反射（Retinex）；锐化 = 原图减模糊，只增强已有细节。

## 复习自检清单

- ☐ 能写出 $d_0$ 与 $d_1$ 的定义和输出公式，并解释为什么 $d_0$ 的输出会右移半个采样
- ☐ 能推导 $D_0[u]$ 与 $D_1[u]$ 的 DFT，说明二者在幅度、相位上与理想微分 $jw$ 的差距
- ☐ 能解释"卷积与求导可交换"如何让高斯导数同时解决噪声与不可微两个问题，并写出一阶、二阶高斯导数公式
- ☐ 能写出 Hermite 多项式的递推式和高阶高斯导数的一般形式，说明两个高斯导数卷积后阶数、方差如何变化
- ☐ 能写出 $Sobel_x$、$Sobel_y$ 核及其二项式分解与 DFT 表达式，说出四种导数离散化各自的优劣
- ☐ 能写出五点拉普拉斯核和输出公式，列举拉普拉斯相对梯度的四个优点
- ☐ 能推导方向导数 $d_{\theta} = \cos\theta\, d_n + \sin\theta\, d_m$，说明为什么任意方向的导数不需要重新卷积
- ☐ 能解释梯度域表示为何只能重建到相差一个直流分量，并描述 Retinex 的完整流程（log 变换 → 阈值化梯度 → 积分 → 取指数）

---

原文：[Image Derivatives](https://visionbook.mit.edu/derivatives.html)
