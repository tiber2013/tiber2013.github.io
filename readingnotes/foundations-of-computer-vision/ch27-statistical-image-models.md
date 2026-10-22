---
title: 第 27 章 · 统计图像模型
chapter: 27
original: https://visionbook.mit.edu/stat_image_models_revised.html
---

# 第 27 章 · 统计图像模型

<div class="src-note">
本页为《Foundations of Computer Vision》第 27 章的<b>学习笔记摘要</b>，由 AI 辅助整理，<b>非原文翻译</b>。原书 © 2024 Antonio Torralba、Phillip Isola、William T. Freeman，MIT Press 出版，<a href="https://visionbook.mit.edu/stat_image_models_revised.html" target="_blank" rel="noopener">官网免费公开</a>，采用 <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener">CC BY-NC-ND 4.0</a> 授权。插图版权归原作者所有，此处仅作非商业学习引用。
</div>

## 本章定位

本章开启 Part VIII「图像统计模型与图模型」。前面各章教我们滤波、傅里叶分析与神经网络；本章换个角度——不追求理解图像内容，而是回答「自然图像的概率分布长什么样」。从独立像素直方图出发，经枯叶模型与 $1/f$ 功率谱，建立高斯图像模型（并用于维纳去噪），再用小波边缘模型刻画稀疏性，最后以马尔可夫随机场（MRF）的直觉导出非局部均值去噪。这些低层统计先验是第 28 章纹理、第 29 章概率图模型、第 32 章生成模型的直接先导。

## 本篇要点

- 自然图像在全部可能图像构成的高维空间中只占极小一角；统计模型的作用就是用低层规律（像素相关性、$1/f$ 功率谱、稀疏性）把这个空间约束住，服务于图像合成与去噪。
- 独立像素模型（颜色直方图）是最粗糙的基线：采样结果只有颜色分布正确、结构完全丢失——像素显然不独立。
- 自然图像的傅里叶幅度按幂律 $1/w^\alpha$（$\alpha\simeq1$，实测约 1.5）衰减且近似各向同性；据此构造的循环协方差高斯模型可被傅里叶基对角化，解析地给出维纳去噪，但样本像「云」、丢边缘。
- 带通滤波响应的直方图呈尖峰非高斯分布（广义拉普拉斯，$r\in[0.4,0.8]$）——这就是稀疏性；$r<1$ 偏好锐利边缘，其 MAP 去噪等价于一张「核化（coring）」查找表。
- MRF 直觉：已知某像素周围足够大的邻域，它的取值与图像其余部分条件独立；非局部均值（NLM）按邻域相似度对全图像素加权平均，可显著去除高斯噪声。
- 这一串模型只是逐级缩小自然图像空间，还不足以做强生成模型；更强的生成图像模型见第 32 章。

## 关键插图

<p class="fig-hint">原书标注为最高优先级（三星）的核心插图，共 6 幅。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/statistical_image_models/model_hist_example2.webp" alt="图 27.5 直方图模型的训练与采样流程" loading="lazy" />
<figcaption>图 27.5：用一幅图像估计图像模型参数，然后按训练图像的直方图独立采样各像素，生成一幅新图像。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：「先拟合参数、再采样」是本章所有模型的通用工作流；把这张图的流程看清楚，后面每个模型都只是替换了「学什么、怎么采」。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/statistical_image_models/FT_power_law_angular2.webp" alt="图 27.10 自然图像的 1/f 功率谱" loading="lazy" />
<figcaption>图 27.10：(上排) 三幅 512×512 自然图像与一幅随机图像；(中排) 它们的傅里叶变换幅度（彩色图先对三个通道取平均转为灰度）；(底排) FT 径向截面的角度平均，与 1/w、1/w^1.5、1/w^2 曲线比较（w 为径向频率）。三幅自然图像的 FT 大致按 1/w^α 衰减；噪声图像则完全不同（幅度平坦）。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是全书引用最多的自然图像统计事实之一——1/f 谱既是「图像低频主导」的定量表达，也是高斯先验、维纳去噪乃至后续生成模型讨论的出发点，面试与考试都应能默画这条对数斜率约 −1.5 的曲线。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/statistical_image_models/figure_samples_1_over_f.webp" alt="图 27.11 1/f 高斯模型的采样结果" loading="lazy" />
<figcaption>图 27.11：(上排) 随机相位、功率谱幅度按 1/(1+w^α)（α=1.5）生成的图像；(底排) 对每个颜色通道独立应用同一生成过程的彩色结果。生成的图像看起来像云。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：「随机相位 + 1/f 幅度」是全书最经典的图像合成实验之一：一眼看出高斯模型抓住了尺度与对比度统计，却完全丢失了相位携带的结构——这正是它去噪效果有限的根因。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/statistical_image_models/examplesderivativedistributions.webp" alt="图 27.16 稀疏性：不同世界的带通响应分布趋同" loading="lazy" />
<figcaption>图 27.16：不同视觉世界图像及其带通滤波版本的直方图比较。(上排) 高斯噪声——带通滤波后仍是高斯分布。(中、底排) 两幅非常不同的图像，带通滤波后具有相似的尖峰非高斯直方图。黑线为最佳高斯拟合，对尖峰直方图拟合不佳。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这张图是「自然图像稀疏性」最经典的视觉证据——不同内容的图像在带通域坍缩到同一个尖峰分布（大量系数精确为零、少量大值），是小波阈值去噪、压缩感知与 L1 正则共同的经验根基。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/statistical_image_models/wavelet_denoising.webp" alt="图 27.20 小波去噪的似然/先验/后验" loading="lazy" />
<figcaption>图 27.20：由若干噪声观测估计子带系数时的似然、先验与后验项。(a) 观测到零子带系数；(b) 观测值 0.26 使似然项平移，但后验仍在零处取峰；(c) 观测系数值 1.22 给出最大后验估计 0.9。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：一图讲透「拉普拉斯先验 + 高斯似然」的 MAP 行为——小观测被吸向零（去噪），大观测被收缩而非保留（保边缘）；这是所有小波收缩去噪的理论原型。</p>

<figure class="book-fig">
<img src="/readingnotes/foundations-of-computer-vision/figures/statistical_image_models/nlm1.webp" alt="图 27.22 NLM 的权重支撑区域" loading="lazy" />
<figcaption>图 27.22：非局部均值去噪算法的支撑区域。图引自 [14]。</figcaption>
</figure>
<p class="fig-note">💡 为什么重要：这是 MRF「邻域定值」直觉最直观的可视化——权重图清楚显示「相似邻域被跨空间地拉到一起平均」，NLM 及其后继（BM3D 的块匹配）是工业去噪的主力思路。</p>

## 本章小结

- 统计图像模型用低层规律约束巨大的图像空间：直方图模型（独立像素）只能对上颜色；像素对散点图集中在恒等线附近，相关性随距离缓慢衰减。
- 枯叶模型解释了自然图像反照率分布的成因；其最重要的推论是自然图像功率谱按 $1/w^\alpha$（$\alpha\simeq1$，实测约 1.5）衰减、近似各向同性。
- 高斯模型 = 最大熵 + 相关函数约束：平稳时为循环协方差、可被傅里叶基对角化（$\mathbf{C}=\mathbf{F}\mathbf{D}\mathbf{F}^\mathsf{T}$，$\mathbf{D}$ 对角元 $1/w^\alpha$）；采样像云但保留细节尺度与主导朝向；用于去噪即维纳滤波器 $\boldsymbol\ell=\mathbf{C}(\mathbf{C}+\sigma_g^2\mathbf{I})^{-1}\boldsymbol\ell_g$，效果有限。
- 自然图像真正的结构信息在带通域：滤波响应服从广义拉普拉斯分布（$r\in[0.4,0.8]$），尖峰、稀疏、跨视觉世界趋同；$r<1$ 偏好锐利边缘。
- 小波边缘模型的 MAP 去噪等价于核化（coring）查找表：小系数置零、大系数收缩；玩具修复实验定量展示了 $r$ 如何控制「平滑 vs 锐阶跃」的偏好。
- MRF 直觉（邻域定值）导出非局部均值：按邻域相似度加权全图平均，$\sigma=15$ 的高斯噪声也能显著去除；它不依赖傅里叶/小波基，是非参数路线的代表。

## 复习自检清单

- ☐ 能说出区分噪声与纹理的四个判据（与内容无关、不受边界影响、不随表面朝向变形、不随距离变频），并解释「世界里的平稳噪声需要与视点共谋」的含义
- ☐ 能写出独立像素模型 (27.1) 与多项分布采样流程，解释直方图模型作为生成模型失败的原因、为何它作为工程操作仍然常用
- ☐ 能写出相关系数 $\rho$ 与式 (27.2)，解释自然图像像素对散点图为何集中在恒等线、其相关衰减与白噪声有何不同
- ☐ 能写出功率谱幂律 (27.3) 并复现图 27.10 的结论（角度平均、$1/w^{1.5}$、噪声平坦谱），说出 $w$、$\alpha$ 的含义与典型取值
- ☐ 能写出高斯图像模型 (27.4)，解释「只约束相关函数」为何导致高斯解（最大熵），以及平稳协方差为何可被傅里叶基对角化（$\mathbf{C}=\mathbf{F}\mathbf{D}\mathbf{F}^\mathsf{T}$）
- ☐ 能推导维纳去噪解 $\boldsymbol\ell=\mathbf{C}(\mathbf{C}+\sigma_g^2\mathbf{I})^{-1}\boldsymbol\ell_g$ 及其频域形式，并解释它为何去噪不理想
- ☐ 能写出广义拉普拉斯分布 (27.5)，说明 $r=0.5/1/2$ 与 $r\to0/\infty$ 各对应什么分布、自然图像 $r$ 的典型范围，以及它与稀疏性的关系
- ☐ 能重推玩具修复问题的式 (27.7)，说明 $r>1$、$r=1$、$r<1$ 三种情形的最优 $a$ 及「平滑/尖锐」偏好；能写出 NLM 权重 (27.9)，解释 $Z_{n,m}$ 的作用、$h$ 的典型取法及其与 MRF 条件独立直觉的对应

---

原文：[Statistical Image Models](https://visionbook.mit.edu/stat_image_models_revised.html)
