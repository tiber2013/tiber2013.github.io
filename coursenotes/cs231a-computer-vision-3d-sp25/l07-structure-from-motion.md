---
title: "CS231A · 第 7 讲 · Structure from Motion (SfM)"
description: "Stanford CS231A 2025 春季第 7 讲：从多张图像同时恢复相机运动与三维结构，涵盖 Affine 因式分解、八点法与基础矩阵、自校准、Bundle Adjustment 与增量式/全局式流程。"
---

# CS231A · 第 7 讲 · Structure from Motion (SfM)

> **Stanford CS231A 计算机视觉：从三维感知到三维重建 · 第 7 讲（2025 春季）** ｜ 讲者：Silvio Savarese、Jeannette Bohg ｜ 本课无公开视频，笔记依据官方课件与原始论文
>
> 课件共 87 页，是同一主题三次课时的合订（标题页分别署期 21-April-25 与 19-Apr-25），正文顺序即按课件的「SfM 问题 → Affine SfM → Perspective SfM → 自校准 → 应用 → 附录」展开；下文所引页码均指该课件。

## 核心结论

这一讲把前几讲的零件装成一台完整的机器：对极几何给出「两帧之间」的约束，三角化给出「已知相机时」求点，而 SfM 要同时求出**未知的相机运动**（$m$ 个投影矩阵 $\mathbf M_i$）和**未知的三维结构**（$n$ 个点 $\mathbf X_j$），只依赖 $m\times n$ 个二维观测 $\mathbf x_{ij}$。课件的立场很明确：这件事没有一次性闭式解，只能分层做——先用线性/代数手段拿到一个带歧义的解，再用非线性优化把它拉回真实几何。Affine SfM 是刻意简化的教学入口：投影去掉分母后变成 $\mathbf x_{ij}=\mathbf A_i\mathbf X_j+\mathbf b_i$，于是全部约束塌缩成一个漂亮的**秩为 3** 的测量矩阵，一次 SVD 就同时给出运动和结构。透视情形则回到八点法与基础矩阵 $\hat{\mathbf x}'^{\top}F\hat{\mathbf x}=0$，得到的是只到 4×4 射影变换为止的重建；把它升级成度量重建就是自校准。最后所有路线都要过 Bundle Adjustment 这一关，而它正是本课程的通用语言——非线性最小二乘。

## SfM 问题：为什么它是双线性且非凸的

![三台相机与若干世界点的投影连线图，红点是光心与三维点，蓝点是像面观测；下方红框把待求量分成 motion（m 个投影矩阵）与 structure（n 个三维点）](/coursenotes/cs231a-computer-vision-3d-sp25/l07-structure-from-motion/slide_p003.jpg)

给定 $m$ 张图像里的 $n$ 个固定三维点，观测模型就是逐相机逐点的投影

$$
\mathbf x_{ij} = \mathbf M_i\,\mathbf X_j,\qquad i=1,\dots,m,\quad j=1,\dots,n,
$$

其中 $\mathbf M_i = \mathbf K_i[\mathbf R_i \mid \mathbf T_i]$ 是 $3\times4$ 投影矩阵。注意 $\mathbf x_{ij}$ 是齐次量，等式右边还带一个未知的逐观测尺度，所以严格说这是**每个观测一条比例约束**，二维点只贡献 2 个独立方程。

课件据此给出两组自由度计数：仿射情形「$2m\times n$ 个方程，$8m+3n-8$ 个未知量」，透视情形「$2m\times n$ 个方程，$11m+3n-15$ 个未知量」。这里的减项是**规范自由度（gauge freedom）**：每台未标定相机 11 个自由度（$3\times4$ 去掉整体尺度）、每个三维点 3 个，而世界坐标系可以随便挑——透视情形把第一台相机钉成典范形式 $[\mathbf I\mid\mathbf 0]$、第二台钉成 $[\mathbf A\mid\mathbf b]$ 的典范对，正好用掉 $11+4=15$ 个自由度，等价于那个 15 自由度的 4×4 射影变换群；仿射情形把第一台相机钉成典范仿射相机（$\mathbf A_1$ 的 6 个加 $\mathbf b_1$ 的 2 个）就用掉 8 个。把 $m=2$ 代入透视计数会得到「至少 8 个点对」，这正是八点法的出处。

困难在于方程对未知量是**双线性**的：$\mathbf A_i\mathbf X_j$ 里两个因子都未知，固定一边才变成另一边的线性方程。因此可行的观测集合构成的不是线性子空间而是代数簇（秩约束、行列式约束），整体是非凸的。这也解释了课件反复出现的结构：先用某种松弛（线性化、秩截断、成对求解）拿到初值，再用非线性优化收敛——第 59 页明确写了「需要好的初始条件」「Factorization 或 algebraic 方法提供优化的初值」。

## Affine SfM：把透视压成线性

课件先做减法（第 4–6 页）：仿射/正交相机下，投影矩阵第三行不再依赖 $X$，透视除法消失。把 $\mathbf M$ 拆成 $\left[\begin{smallmatrix}\mathbf A & \mathbf b\\ \mathbf 0\,\mathbf 0\,\mathbf 0 & 1\end{smallmatrix}\right]$ 后，欧氏空间里的观测就是

$$
\mathbf x_{ij} = \mathbf A_i\,\mathbf X_j + \mathbf b_i
$$

维度与课件第 6 页一致：$\mathbf x_{ij}$ 是 $2\times1$、$\mathbf A_i$ 是 $2\times3$、$\mathbf X_j$ 是 $3\times1$、$\mathbf b_i$ 是 $2\times1$。也就是把三维点先取前两个坐标 $\mathbf X^E=(m_1\mathbf X,\,m_2\mathbf X)^{\top}$、丢掉深度，只留一个放大率。这一步把「同时求运动和结构」变成了矩阵分解问题。

**第一步：中心化。** 对每张图求像点质心 $\bar{\mathbf x}_i=\frac{1}{n}\sum_{k=1}^{n}\mathbf x_{ik}$，再对质心做减法：

$$
\hat{\mathbf x}_{ij} = \mathbf x_{ij}-\bar{\mathbf x}_i
= \mathbf A_i\mathbf X_j+\mathbf b_i-\frac{1}{n}\sum_{k=1}^{n}\mathbf A_i\mathbf X_k-\frac{1}{n}\sum_{k=1}^{n}\mathbf b_i
= \mathbf A_i\big(\mathbf X_j-\bar{\mathbf X}\big) = \mathbf A_i\hat{\mathbf X}_j
$$

平移项 $\mathbf b_i$ 被干净地消掉了。若再把世界原点放在三维点集的质心上（$\bar{\mathbf X}=\mathbf 0$），就有 $\hat{\mathbf x}_{ij}=\mathbf A_i\mathbf X_j$。

**第二步：因式分解。** 把所有中心化观测排成测量矩阵（measurement matrix）：

![中心化后的观测排成 2m×n 的测量矩阵 D，右侧写成相机块 M（2m×3）与点块 S（3×n）的乘积，页面底部标明 D = MS 的秩为 3](/coursenotes/cs231a-computer-vision-3d-sp25/l07-structure-from-motion/slide_p017.jpg)

$$
\mathbf D =
\begin{bmatrix}
\hat{\mathbf x}_{11} & \hat{\mathbf x}_{12} & \cdots & \hat{\mathbf x}_{1n}\\
\hat{\mathbf x}_{21} & \hat{\mathbf x}_{22} & \cdots & \hat{\mathbf x}_{2n}\\
\vdots & \vdots & \ddots & \vdots\\
\hat{\mathbf x}_{m1} & \hat{\mathbf x}_{m2} & \cdots & \hat{\mathbf x}_{mn}
\end{bmatrix}
= \mathbf M\mathbf S,\qquad
\mathbf M=\begin{bmatrix}\mathbf A_1\\ \vdots\\ \mathbf A_m\end{bmatrix}_{2m\times3},\quad
\mathbf S=\big[\hat{\mathbf X}_1\ \cdots\ \hat{\mathbf X}_n\big]_{3\times n}
$$

每个 $\hat{\mathbf x}_{ij}$ 本身是 $2\times1$ 向量，所以 $\mathbf D$ 是 $2m\times n$。秩约束来自矩阵乘积的基本性质：

$$
\operatorname{rank}(\mathbf D)=\operatorname{rank}(\mathbf M\mathbf S)\le \min\big(\operatorname{rank}\mathbf M,\operatorname{rank}\mathbf S\big)\le 3
$$

直观地说：所有观测都由 3 个自由度（三维点的坐标）生成，因此 $2m\times n$ 的数据里只有 3 个独立方向。反过来，**秩为 3 就是仿射投影的充要条件**——这也解释了为什么退化：若所有点落在同一平面上，$\mathbf S$ 的秩降到 2，第三个维度根本不存在，因式分解无法恢复出体结构。

对 $\mathbf D=\mathbf U\mathbf W\mathbf V^{\top}$ 做 SVD，只保留 3 个非零奇异值：

$$
\mathbf D = \mathbf U_3\mathbf W_3\mathbf V_3^{\top}
= \underbrace{\mathbf U_3}_{\mathbf M}\;\underbrace{\big(\mathbf W_3\mathbf V_3^{\top}\big)}_{\mathbf S}
$$

课件第 23 页点出真实数据的问题：噪声和仿射近似都会让 $\operatorname{rank}(\mathbf D) \gt 3$。此时用前三个奇异分量是**最优的**：按 Eckart–Young–Mirsky 定理，$\mathbf U_3\mathbf W_3\mathbf V_3^{\top}$ 是 $\mathbf D$ 在 Frobenius 范数下最好的秩 3 逼近，而课件正是用这个范数写出误差度量：

$$
\|\mathbf A\|_F=\sqrt{\sum_{i}\sum_{j}|a_{ij}|^2}=\sqrt{\sum_{i}\sigma_i^2}
$$

这就是 Tomasi & Kanade（IJCV 9(2):137–154, 1992）的方法，也是课件第 10 页引用的原文；第 24 页放了他们论文里那栋山形墙房屋的重建结果图，是早期少见的「实用 SfM 算法」之一。

## 歧义的层级：从仿射到相似到射影

因式分解的解不唯一。对任意可逆 $3\times3$ 矩阵 $\mathbf H$，

$$
\mathbf M^{*}=\mathbf M\mathbf H,\qquad \mathbf S^{*}=\mathbf H^{-1}\mathbf S
\quad\Longrightarrow\quad \mathbf M^{*}\mathbf S^{*}=\mathbf M\mathbf S=\mathbf D
$$

这就是**仿射歧义**：相机和点可以同时被一个三维仿射变换搬动，图像完全看不出来（第 25–27 页）。必须再注入约束（正交性、内参）才能落到度量世界。

课件第 29–30 页把最有说服力的一条结论单独列出：即使相机已经标定，场景也只能恢复到**相似变换**（旋转、平移、缩放）为止，这叫**度量重建**，而「仅凭图像不可能估计场景的绝对尺度」。

![玩具屋与真实小孩同框配在 Similarity Ambiguity 页上：同一组图像对应无穷多个尺度不同的场景，单目无法判定绝对大小](/coursenotes/cs231a-computer-vision-3d-sp25/l07-structure-from-motion/slide_p030.jpg)

尺度不可恢复不只是哲学问题，它有具体的数值后果：度量重建的规范群有 7 个自由度（3 旋转 + 3 平移 + 1 尺度），所以 Bundle Adjustment 的信息矩阵/Hessian 必然有 7 个零特征值、不可直接求逆。Szeliski 第 7.4.3 节给的处理办法是丢掉最小的 7 个特征值，或者干脆加上 Levenberg–Marquardt 的对角阻尼。

到了未标定的透视情形，歧义升级为一个任意的 4×4 射影变换 $\mathbf H$：$\mathbf x_{ij}=\mathbf M_i\mathbf X_j=(\mathbf M_i\mathbf H^{-1})(\mathbf H\mathbf X_j)$，相机与点各自被搬动而观测不变（第 33–36 页）。于是 SfM 被拆成两个任务：先恢复到射影歧义，再把射影解升级为度量解。

## Perspective SfM：代数路线与八点法

两视图的代数路线是课件第 39–52 页的主体，三步走：从至少 8 个对应点算 $F$；用 $F$ 估计射影相机；再三角化求点。

利用射影歧义把两台相机规范化：

$$
\tilde{\mathbf M}_1=\mathbf M_1\mathbf H^{-1}=[\mathbf I\mid\mathbf 0],\qquad
\tilde{\mathbf M}_2=\mathbf M_2\mathbf H^{-1}=[\mathbf A\mid\mathbf b],\qquad \tilde{\mathbf X}=\mathbf H\mathbf X
$$

于是 $\mathbf x=[\mathbf I\mid\mathbf 0]\tilde{\mathbf X}$，而 $\mathbf x'=[\mathbf A\mid\mathbf b]\tilde{\mathbf X}=\mathbf A\mathbf x+\mathbf b$。两条向量差 $\mathbf x'-\mathbf b=\mathbf A\mathbf x$ 与 $\mathbf x'$ 应当共线，叉积为零；把叉积写成反对称矩阵乘法 $[\mathbf a]_\times$，再点乘 $\mathbf x'^{\top}$，就得到

$$
\mathbf x'^{\top}\big([\mathbf b]_\times\mathbf A\big)\mathbf x=0
\quad\Longrightarrow\quad
\boxed{\;\mathbf x'^{\top}\mathbf F\mathbf x=0\;},\qquad \mathbf F=[\mathbf b]_\times\mathbf A
$$

![代数路线的核心一页：把两台相机规范成 I|0 与 A|b，经叉积把对极约束写成 x'T F x = 0，并标注 F 等于 b 的叉积矩阵左乘 A](/coursenotes/cs231a-computer-vision-3d-sp25/l07-structure-from-motion/slide_p045.jpg)

$\mathbf F$ 是 $3\times3$、秩 2、7 个自由度；$F\mathbf x_2$ 给出对应 $\mathbf x_2$ 的第一条极线，$\mathbf F^{\top}\mathbf x_1$ 给出第二条，$\mathbf F\mathbf e_2=\mathbf 0$、$\mathbf F^{\top}\mathbf e_1=\mathbf 0$（第 51 页回顾第 5 讲的极线约束）。课件与 [HZ] 习惯把归一化（去质心、缩放到平均距离 $\sqrt2$）后的坐标写成带帽的 $\hat{\mathbf x}$，于是这条约束就写成 $\hat{\mathbf x}'^{\top}\mathbf F\hat{\mathbf x}=0$。每个点对提供一条关于 $\mathbf F$ 九个元素的线性齐次方程，8 个点对配 7 个自由度刚好超定——这就是 Longuet-Higgins 八点法（Nature 293:133–135, 1981）；实际求解必须先做上面说的归一化，否则数值病态，见 Hartley, "In Defense of the Eight-Point Algorithm", IEEE TPAMI 19(6):580–593, 1997。

反过来从 $F$ 恢复相机是课件推导最密的一段。因为 $\mathbf F=[\mathbf b]_\times\mathbf A$，有 $\mathbf F\mathbf b=[\mathbf b]_\times\mathbf A\mathbf b=\mathbf 0$ 且 $\mathbf b^{\top}\mathbf F=\mathbf 0$，所以**$\mathbf b$ 就是第二极点**，可用 SVD 在 $\|\mathbf b\|=1$ 约束下最小二乘求出。再令 $\mathbf A'=-[\mathbf b]_\times\mathbf F$，用恒等式 $[\mathbf b]_\times[\mathbf b]_\times=\mathbf b\mathbf b^{\top}-\|\mathbf b\|^2\mathbf I$ 验证

$$
[\mathbf b]_\times\mathbf A' = -\big(\mathbf b\mathbf b^{\top}-\|\mathbf b\|^2\mathbf I\big)\mathbf F = -\mathbf b(\mathbf b^{\top}\mathbf F)+\|\mathbf b\|^2\mathbf F=\mathbf F
$$

于是典范相机对为 $\tilde{\mathbf M}_1=[\mathbf I\mid\mathbf 0]$、$\tilde{\mathbf M}_2=[-[\mathbf e]_\times\mathbf F\mid \mathbf e]$（第 46–47、52 页，对应 HZ 第 254 页）。最后逐点三角化：对每个 $\tilde{\mathbf X}_j$ 汇集 $m$ 条形如 $\mathbf x_{ij}\times(\mathbf M_i\tilde{\mathbf X}_j)=\mathbf 0$ 的线性方程，SVD 求最小奇异向量（第 52 页，HZ 第 312 页）。多视图时课件的做法是每对图像各自求一组 $\tilde{\mathbf M}_k,\tilde{\mathbf M}_h,\tilde{\mathbf X}_{[k,h]}$，再靠 Bundle Adjustment 把它们拼起来（第 53 页）。

## 自校准：从射影升级到度量

![自校准页的三阶段示意：歪斜的射影线框 → 校正后的规则线框 → 贴上纹理的度量房屋，直观呈现从 perspective 到 metric 的升级](/coursenotes/cs231a-computer-vision-3d-sp25/l07-structure-from-motion/slide_p061.jpg)

自校准定义为「从射影（或仿射）重建中恢复度量重建」，做法是对相机内参做假设（第 61、63 页）。课件列了四条路：单视图计量约束（第 4 讲内容）、两视图的直接法（Kruppa 方程）、代数法、分层法（stratified）。

附录里给出直接法的链条：圆锥曲线跨视图映射 $[\mathbf e']_\times C'^{-1}[\mathbf e']_\times^{\top}=\mathbf FC^{-1}\mathbf F^{\top}$；把 $C$ 换成绝对二次曲线（imaginary circular points）$\omega=(\mathbf K\mathbf K^{\top})^{-1}$、$\omega'=(\mathbf K'\mathbf K'^{\top})^{-1}$，就得到 Kruppa 方程——用 $\mathbf F$ 的 SVD 列向量 $\mathbf u_i,\mathbf v_i$ 与奇异值 $\sigma_i$ 写出，对 $\mathbf K,\mathbf K'$ 给出两个独立约束；若假设 $\mathbf K'=\mathbf K=\operatorname{diag}(f,f,1)$，两个约束化成一个关于 $f^2$ 的二次方程 $\alpha f^2+\beta f+\gamma=0$，直接解出焦距。它的致命弱点写在第 81 页：**只对相机对有效，且当没有旋转（纯平移）时退化**——此时 $\mathbf F=[\mathbf e']_\times$，等式两边恒等，什么也约束不了。Szeliski 第 7.2.2 节也指出光轴相交一类的退化构型同样失效，并建议改用三帧以上的关系。

多视图代数法（第 83–86 页）更工程化：设已有一个射影重建 $\{\tilde{\mathbf M}_i,\tilde{\mathbf X}_j\}$，$\tilde{\mathbf M}_1=[\mathbf I\mid\mathbf 0]$，$\tilde{\mathbf M}_i=[\mathbf A_i\mid\mathbf b_i]$，寻找把射影解拉到度量解的同调

$$
\mathbf H=\begin{bmatrix}\mathbf K_1 & \mathbf 0\\ -\mathbf p^{\top}\mathbf K_1 & 1\end{bmatrix}
\quad\text{使得}\quad
\big(\mathbf A_i-\mathbf b_i\mathbf p^{\top}\big)\mathbf K_1\mathbf K_1^{\top}\big(\mathbf A_i-\mathbf b_i\mathbf p^{\top}\big)^{\top}=\mathbf K_i\mathbf K_i^{\top}
$$

未知量是 $\mathbf p$ 的 3 个加每幅图 $\mathbf K_i$ 的 5 个，每幅图提供 5 个独立方程。若进一步假设所有相机内参相同（$\mathbf K_1=\dots=\mathbf K_m$），未知量只剩 $3+5=8$，于是**至少 3 幅图**就能自校准。课件第 86 页的表把「假设越强、所需视图越少」讲得很清楚：内参恒定要 3 幅；已知长宽比与斜度、焦距和主点变化要 4 幅；只知斜度为 0、其余全变要 8 幅。同时它也点出代价——视图越多，对应关系越难建立，「Bundle adjustment helps!」。

## Bundle Adjustment：所有路线的最后一步

![Bundle adjustment 页：目标函数 E(M,X) 写成 m×n 项重投影偏差的平方和，下方示意图区分重建点、真值点与各相机的投影观测](/coursenotes/cs231a-computer-vision-3d-sp25/l07-structure-from-motion/slide_p057.jpg)

课件把 BA 定位为非线性精化方法，最小化重投影误差：

$$
E(\mathbf M,\mathbf X)=\sum_{i=1}^{m}\sum_{j=1}^{n} D\big(\mathbf x_{ij},\ \mathbf M_i\mathbf X_j\big)^2
$$

$D$ 是「测量与再投影之间的非线性映射」；标定已知时它就是像素残差 $\big\|\mathbf x_{ij}-\pi(\mathbf K_i[\mathbf R_i\mid\mathbf T_i]\mathbf X_j)\big\|^2$，未标定时把 $\mathbf K_i$ 也并入待估参数，这正是第 63 页「在 BA 优化过程中注入相机信息」的含义，也是第 87 页总结里那条并列路线（先自校准再 BA，或直接做带自校准约束的 BA）。

求解手段就是本课通用的非线性最小二乘：Newton 法需要雅可比 $\mathbf J$ 与 Hessian $\mathbf H$，Levenberg–Marquardt 不需要显式 $\mathbf H$；两者都是迭代的、从初值出发，初值离真值太远会慢，而且**结果可能依赖于初值**（第 58 页这四条要点，就是非凸性的直接后果）。BA 的优点是能处理大量视图、能处理缺失数据；缺点是参数随视图数增长、需要好的初值，所以它总是被放在最后一步（第 59 页）。

值得补两点课件没展开但决定工程成败的细节（Szeliski 第 7.4.1 节）：一是**稀疏性**——$\mathbf J^{\top}\mathbf J$ 具有相机块与点块交界的箭头形结构，用 Schur 补先消点数百万级的三维点、再解只含相机位姿的缩减 Hessian，是大规模 BA 可行的前提；二是**规范自由度**让缩减 Hessian 秩亏 7，实践中靠 LM 的对角阻尼 $\lambda\operatorname{diag}(\mathbf A)$ 就能稳住。综述见 Triggs, McLauchlan, Hartley & Fitzgibbon, "Bundle Adjustment — A Modern Synthesis"（International Workshop on Vision Algorithms, Kerkyra, Greece, 1999；后收入 LNCS 2000）。

## 可见性、遮挡与流程选择：增量式 vs 全局式

课件第 55 页把前面两种方法的短板挑明：**因式分解假设所有点在所有视图中都可见**，一旦遮挡或对应失败，$\mathbf D$ 里就出现空洞，秩 3 结构不再成立；**代数方法只在两视图上工作**。这两条缺口正是 BA 存在的理由。对因式分解的缺失数据，经典做法是交替/迭代式的：Tomasi & Kanade 原文先在更稠密的小子集上做分解，再用已知的相机或点去「补全」缺失项，从而逐步纳入更多特征与相机（Szeliski 第 7.3 节；透视相机下的迭代因子化见 Christy & Horaud, 1996，通常三到五次迭代收敛，且因深度反转歧义要把两种解都试一遍）。

把课件第 65 页那张编年史幻灯片摊开，能看到两条流程线：

- **增量式（incremental）**：先挑一对好图像做两视图初始化并三角化，然后逐批注册新相机（PnP/姿态估计），每注册一批就重跑一次局部 BA。Photo Tourism（Snavely, Seitz & Szeliski, ACM TOG / SIGGRAPH 2006, 25(3):835–846）是代表作，课件第 70 页放了它的结果图；Bundler 是随之放开的开源实现；COLMAP（Schönberger & Frahm, "Structure-from-Motion Revisited", CVPR 2016, pp. 4104–4113）把三角化、外点过滤、BA 调度各模块重新做了一遍，成为今天的事实标准。
- **全局式（global）**：一次性从视图图（view graph）估计所有相机位姿，再只做一轮 BA。历史上 Fitzgibbon & Zisserman（1998，Szeliski 第 7.4.4 节引述）先重建三元组再层次合并；Agarwal 等人 "Building Rome in a Day"（ICCV 2009, pp. 72–79）用并行 BA 重建罗马；理论工具是运动平均——旋转平均见 Hartley, Trumpf, Dai & Li, "Rotation Averaging", IJCV 103:267–305（2013），$L_\infty$ 型全局最优重建见 Kahl & Hartley, IEEE TPAMI 30(11):1603–1617（2008）；近年的全局式系统见 Pan, Baráth, Pollefeys & Schönberger, "Global Structure-from-Motion Revisited", ECCV 2024, pp. 58–77。全局式的痛点在平移平均：两视图相对平移的尺度本身未知，近似共线的运动会让尺度估计极不稳——这也是课件第 30 页那条尺度歧义在系统层面的回响。

## 课件给出的应用与数据

这一讲没有列标准评测集或指标表，给出的是一组应用证据。第 65 页（Oxford VGG 提供的城市点云图）串起从 Lucas & Kanade (81) 到 Snavely、Agarwal、Frahm 的整条谱系；第 66 页是 Pollefeys 等人 98 年起做的「重建 + 纹理映射」，把浮雕照片贴回三维表面；第 67–69 页是 Golparvar-Fard、Peña-Mora 与 Savarese 的施工场地增量重建（D4AR 系统），课件原文标注「初始对 2168 个点，完整集合 62,323 个点、160 张图像」，并把注册好的相机视锥叠回现场照片，用于施工进度核对。第 70 页是 Photo Tourism：无序网络照片集合直接变成可漫游的三维场景。下一讲预告是主动立体与体立体（第 72 页），即从稀疏点云走向稠密几何。

## 要点回顾

- SfM 同时求 $m$ 个投影矩阵（motion）与 $n$ 个三维点（structure），只依赖 $m\times n$ 个二维观测；方程对未知量双线性，可行集由秩/行列式约束刻画，因此非凸。
- 自由度计数：仿射是 $2mn$ 个方程对 $8m+3n-8$ 个未知量，透视是 $2mn$ 个方程对 $11m+3n-15$ 个未知量；减掉的 8 与 15 是把世界坐标系钉在典范相机上所用的规范自由度。
- Affine SfM 先中心化消去 $\mathbf b_i$，再构造 $2m\times n$ 测量矩阵 $\mathbf D=\mathbf M\mathbf S$，其秩必 $\le 3$；SVD 取前三个奇异分量给出 Frobenius 范数下的最优秩 3 逼近（Eckart–Young–Mirsky），即 Tomasi–Kanade 算法。
- 秩论证顺带给出退化条件：所有点共面时 $\mathbf S$ 秩降为 2，仿射因式分解无法恢复第三维。
- 歧义分三层：仿射歧义 $\mathbf M^{*}=\mathbf M\mathbf H,\ \mathbf S^{*}=\mathbf H^{-1}\mathbf S$；标定后仍剩 7 自由度的相似歧义（绝对尺度不可恢复）；未标定透视情形剩 15 自由度的射影歧义。
- 两视图代数路线：把相机规范成 $[\mathbf I\mid\mathbf 0]$ 与 $[\mathbf A\mid\mathbf b]$，推出 $\mathbf x'^{\top}\mathbf F\mathbf x=0$、$\mathbf F=[\mathbf b]_\times\mathbf A$；$\mathbf b$ 由 $\mathbf F\mathbf b=\mathbf 0$ 求得且就是极点，$\mathbf A=-[\mathbf b]_\times\mathbf F$；8 个点对定 $\mathbf F$（Longuet-Higgins 1981），实际须用归一化八点法（Hartley 1997）。
- 典范相机对 $\tilde{\mathbf M}_2=[-[\mathbf e]_\times\mathbf F\mid\mathbf e]$ 确定后用线性三角化（SVD）逐点求 $\tilde{\mathbf X}_j$；多视图成对求解再靠 BA 合并。
- 自校准即把射影/仿射重建升级为度量重建：Kruppa 方程用绝对二次曲线 $\omega=(\mathbf K\mathbf K^{\top})^{-1}$ 的跨视图传递得到内参约束，但只对相机对有效且纯旋转/纯平移退化；多视图代数法在同调 $\mathbf H=[[\mathbf K_1,\mathbf 0],[-\mathbf p^{\top}\mathbf K_1,1]]$ 下每视图给 5 个方程，内参恒定时 3 幅图即可。
- 假设越强所需视图越少：内参恒定 3 幅、已知长宽比与斜度 4 幅、只知斜度为 0 则 8 幅；视图越多对应问题越难，要靠 BA。
- Bundle Adjustment 最小化 $\sum_i\sum_j D(\mathbf x_{ij},\mathbf M_i\mathbf X_j)^2$，用 Newton 或 LM 迭代；优点是可处理多视图与缺失数据，代价是规模大、必须有好初值，故总放在最后一步。
- 度量重建的 7 维规范自由度使 BA 的 Hessian 秩亏，靠 Schur 补消点保持稀疏性、靠 LM 对角阻尼稳住求解。
- 因式分解要求所有点在所有视图可见，遮挡与匹配失败会破坏秩结构；经典补救是小子集分解后交替补全缺失项再逐步扩张。
- 增量式（Photo Tourism → Bundler → COLMAP）逐批注册相机并局部 BA；全局式（Fitzgibbon & Zisserman 1998、Building Rome in a Day、旋转/平移平均、GLOMAP）一次解出所有位姿，瓶颈在带尺度歧义的平移平均与近共线运动退化。

## 参考资料

- 课件：CS231A Lecture 7「Multi-view geometry / Structure from Motion」，Stanford 官方 syllabus：<https://cs231a.stanford.edu/>。本讲阅读清单原文：[HZ] 第 10 章「3D reconstruction of cameras and structure」、第 18 章「N-view computational methods」、第 19 章「Auto-calibration」；[FP] 第 13 章「projective structure from motion」；[Szeliski] 第 7 章「Structure from motion」。
- R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge University Press, 2003（课件第 36 页直接引用；典范相机与三角化分别见第 254、312 页）。
- R. Szeliski, *Computer Vision: Algorithms and Applications*，免费在线版第 7 章「Structure from motion」（三角化、两视图 SfM、因式分解、Bundle adjustment、带约束的结构与运动）：<https://szeliski.org/Book/>。第二版（Springer, 2022）中该主题并入「Structure from Motion and SLAM」一章，编号有变动。
- C. Tomasi and T. Kanade, "Shape and motion from image streams under orthography: A factorization method", *IJCV* 9(2):137–154, 1992。
- H. C. Longuet-Higgins, "A computer algorithm for reconstructing a scene from two projections", *Nature* 293:133–135, 1981。
- R. I. Hartley, "In Defense of the Eight-Point Algorithm", *IEEE TPAMI* 19(6):580–593, 1997。
- P. Sturm and B. Triggs, "A factorization based algorithm for multi-image projective structure and motion", *ECCV* 1996, pp. 709–720；B. Triggs, "Factorization methods for projective structure and motion", *CVPR* 1996, pp. 845–851。
- B. Triggs, P. F. McLauchlan, R. I. Hartley and A. W. Fitzgibbon, "Bundle Adjustment — A Modern Synthesis", *International Workshop on Vision Algorithms*, Kerkyra, Greece, 1999（后收入 LNCS, 2000）。
- M. Pollefeys and L. Van Gool, "From images to 3D models", *Communications of the ACM* 45(7):50–55, 2002；R. Koch, M. Pollefeys and L. Van Gool, "Realistic surface reconstruction of 3D scenes from uncalibrated image sequences", *Journal of Visualization and Computer Animation* 11:115–127, 2000（课件第 66 页的重建与纹理映射工作）。
- N. Snavely, S. M. Seitz and R. Szeliski, "Photo Tourism: Exploring Photo Collections in 3D", *ACM Transactions on Graphics* (Proc. SIGGRAPH) 25(3):835–846, 2006；同组 "Modeling the World from Internet Photo Collections", *IJCV* 80(2):189–210, 2008。Bundler 开源实现：<http://www.cs.cornell.edu/~snavely/bundler/>。
- S. Agarwal, N. Snavely, I. Simon, S. M. Seitz and R. Szeliski, "Building Rome in a Day", *ICCV* 2009, pp. 72–79。
- J. L. Schönberger and J.-M. Frahm, "Structure-from-Motion Revisited", *CVPR* 2016, pp. 4104–4113；COLMAP：<https://colmap.github.io/>。
- L. Pan, D. Baráth, M. Pollefeys and J. L. Schönberger, "Global Structure-from-Motion Revisited", *ECCV* 2024, pp. 58–77。
- R. Hartley, J. Trumpf, Y. Dai and H. Li, "Rotation Averaging", *IJCV* 103:267–305, 2013；F. Kahl and R. Hartley, "Multiple-View Geometry under the $L_\infty$ Norm", *IEEE TPAMI* 30(11):1603–1617, 2008。
- M. Golparvar-Fard, F. Peña-Mora and S. Savarese, "D4AR — a 4-dimensional augmented reality model for automating construction progress monitoring data collection, processing and communicating", *Journal of Information Technology in Construction (ITcon)*, 2009（课件第 67–69 页的施工场地增量重建）。
- R. Szeliski and S. B. Kang, "Shape Ambiguities in Structure from Motion", *IEEE TPAMI* 19(5):506–512, 1997（浮雕/bas-relief 歧义）。
