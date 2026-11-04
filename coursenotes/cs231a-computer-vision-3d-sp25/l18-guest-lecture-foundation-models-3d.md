---
title: "CS231A · 第 18 讲 · 客座讲座：基础模型遇上 3D 视觉"
description: "Francis Engelmann 客座讲座：把 CLIP/SAM 等 2D 基础模型蒸馏进 3D 表示，实现开放世界 3D 场景理解与可控 3D 生成。"
---

# CS231A · 第 18 讲 · 客座讲座：基础模型遇上 3D 视觉

> **Stanford CS231A 计算机视觉：从三维感知到三维重建 · 第 18 讲（2025 春季，2025-06-04）** ｜ 客座讲者：Francis Engelmann（课件署名 PostDoc Stanford）｜ 课件标题 *Foundation Models Meet 3D Vision: Toward Open-World 3D Scene Understanding and Controllable 3D Generation* ｜ 本课无公开视频，笔记依据官方课件与原始论文

## 核心结论

这一讲把前面 17 讲的「几何」和「表示学习」两条线接到一起，回答一个实用问题：**从手机或 LiDAR 扫出的 3D 场景里，我们能不能用任意一句自然语言把想要的东西找出来、分割出来、甚至反过来生成和编辑 3D 内容**。课件给出的答案分三步：第一步承认传统 3D 分割模型被人工标注锁死在封闭类别集（closed set）上；第二步把 CLIP、SigLIP、SAM 这类 2D 基础模型的特征**蒸馏（distillation）** 到 3D 表示里——可以蒸馏到 NeRF 的额外特征通道（OpenNeRF），也可以蒸馏到「实例掩码 + 掩码级特征」这种更结构化的表示（OpenMask3D、Segment3D）；第三步是把理解推到「功能」层面（SceneFun3D、OpenFunGraph 的功能 3D 场景图），并反过来用紧凑的超二次曲面（superquadrics）基元做可控生成与编辑。它与第 10 讲的表示学习、第 16 讲 NeRF、第 17 讲 3DGS 直接衔接：**表示决定了一个 3D 场景能被问什么问题、能被怎么改**。

## 讲座要回答的问题：3D 场景理解是什么

开场先把任务定义清楚：输入是一段 3D 扫描 / 重建结果（课件用 iPhone 的 LiDAR 模组、手持扫描仪、机器人平台和自动驾驶车顶的 LiDAR 作为采集手段的例子），输出是场景里的语义结构。课件把这一支的任务归纳成三类，分别对应三种「知识粒度」：

![3D 场景理解三类任务：左为带实例着色的室内点云分割，中为矢量化的房间平面图，右为多人场景的身体部件分割](/coursenotes/cs231a-computer-vision-3d-sp25/l18-guest-lecture-foundation-models-3d/slide_p007.jpg)

- **3D 场景分割**：输出「物体实例」，代表性方法是 Mask3D；
- **矢量化平面图**：输出「结构元素」，代表性方法是 RoomFormer；
- **人体部件分割**：输出「人—场景交互」所需的部件级标签，代表性方法是 Human3D。

动机落在具身智能上：课件用一段 Boston Dynamics Spot 机械臂在真实客厅里拉抽屉、浇花的视频说明，家庭机器人要做的不是「这是椅子」，而是「哪个抽屉可以拉开、哪个把手可以拧」。

三个代表性方法都值得一句交代，因为它们是后面所有讨论的地基：

- **Mask3D**（Schult, Engelmann, Hermans, Litany, Tang, Leibe，ICRA 2023，arXiv:2210.03105）是第一个纯 Transformer 的 3D 实例分割方法：稀疏卷积骨干（基于 MinkowskiEngine 的 U-Net）输出多尺度点特征 $F\in\mathbb{R}^{N\times D}$，$K$ 个可学习实例查询经解码器精炼后与点特征做归一化余弦相似度，直接并行生成 $K$ 个覆盖全部点的二值掩码，训练用匈牙利匹配做掩码级监督。它甩掉了「预测中心点 + 几何聚类」这套需要手工调半径的流水线，在 ScanNet 测试集上把当时最好成绩提高了 6.2 mAP。
- **RoomFormer**（Yue, Kontogianni, Schindler, Engelmann，CVPR 2023）把平面图重建写成**单层结构化预测**：平面图是一组变长多边形，多边形又是变长顶点序列，于是用「多边形查询 + 角点查询」两级查询端到端预测，边由顶点顺序隐式给出，在 Structured3D 与 SceneCAD 上取得当时最好结果，还能顺带预测房间类型、门窗。
- **Human3D**（Takmaz, Schult, Kaftan, Akçay, Leibe, Sumner, Engelmann, Tang，ICCV 2023）走的是另一条路：真实场景里带部件标签的人太少，就用合成数据渲染出人体点云及其部件真值，再迁移到真实扫描。课件用一页「How well does it really work?」把合成训练的模型放到真实室内扫描上，效果确实能跑，但遮挡与噪声下的边界仍然粗糙。

## 为什么封闭集表示不够用

第 27–30 页是全讲的转折点。课件先展示一批「看起来很好」的封闭集分割结果，然后连续给出两个失败案例：一尊北极熊雕塑被拆成 `lamp`、`pilllar`、`coffee table`；海边两只渔船被整体标成 `stair rail`，并在旁边写了一句 **stair rail ≠ fishing boat**。

![封闭集模型的失败：输入是渔船点云，输出却把整只船标成 stair rail，并注明 stair rail ≠ fishing boat](/coursenotes/cs231a-computer-vision-3d-sp25/l18-guest-lecture-foundation-models-3d/slide_p030.jpg)

原因不复杂：**训练集的人工标注本身就限定了类别表**。ScanNet 的 20 类、ScanNet200 的 200 类都是标注者事先约定的分类体系（taxonomy），模型输出层是一个 $C$ 维 softmax，类别之外的词根本无法表达。于是课件把目标改写成开放词汇（open-vocabulary）版本：**给定任意用户查询，分割出场景中对应的元素**——查询可以是「Footrest」「A comfy seat」「Armchair with floral print」这种带属性、带关系的短语，而不是词表里的一个名词。

这里必须承接第 10 讲：表示学习告诉我们，只要能把图像和文本嵌进同一个特征空间，「分类」就可以退化成「相似度检索」，类别表也就不再是模型的结构性约束。CLIP / SigLIP 正是这样一个已经训好的对齐空间——课件第 33 页画的就是这个机制：文本编码器给出 $f_t$、图像编码器给出 $f_i$，配对样本满足 $f_t \approx f_i$，非配对样本不满足，检索时按归一化内积 $\langle f_t, f_i\rangle / (\lVert f_t\rVert \lVert f_i\rVert)$ 打分。问题只剩一个：**3D 场景里没有像素，怎么把点或掩码放进这个空间。**

## 把 2D 基础模型特征提升到 3D：三条蒸馏路线

### 路线一：给 NeRF 加一个特征通道（OpenNeRF）

第 16 讲的 NeRF 把场景写成连续体，渲染时沿光线积分颜色。OpenNeRF（Engelmann 等，ICLR 2024，arXiv:2404.03650）的做法是：**在辐射场旁边多训一个特征场**，用额外的 CLIP 特征通道去拟合 2D 基础模型给出的像素级特征，于是任意新视角都能渲染出一张「CLIP 特征图」，与文本查询做点积即可零样本分割。

课件第 34 页给的是 2D 侧机制：先用 OpenSeg 之类的开放词汇分割模型算出逐像素 CLIP 特征，再与文本编码做归一化点积，得到一张「相似—不相似」的热力图（steak / mussels / crab 三个查询在同一张海鲜盘上高亮不同区域）。第 35–36 页是 OpenNeRF 的交互 demo：一句 `screens` 就能把三块显示器高亮出来——但注意，它高亮的是**一片语义区域**，无法区分这是三台不同的显示器。课件用一句 *What about different instances?* 把问题推向路线二。

补充一个参照：OpenScene（Peng, Genova, Jiang 等，CVPR 2023）属于同一类「逐点蒸馏」思路，把多视角 CLIP 像素特征反投影平均到 3D 点上；OpenNeRF 的摘要里明确把自己与 LERF、OpenScene 对比，并报告至少 **+4.9 mIoU** 的提升。

### 路线二：掩码级蒸馏（OpenMask3D）

OpenMask3D（Takmaz, Fedele, Sumner, Pollefeys, Tombari, Engelmann，NeurIPS 2023，arXiv:2306.13631）定义了「开放词汇 3D 实例分割」这个任务，思路是**把「分割成实例」和「认出是什么」解耦**：

1. 用 Mask3D 从 3D 重建里预测一批**与类别无关（class-agnostic）** 的实例掩码提案；
2. 对每个 3D 掩码，把它投影回带位姿的 RGB-D 序列，得到若干 2D 视图，并按可见点比例算一个**可见性分数**（课件给出 100% / 90% / 94% / 30% / 0% 五个候选视图）；
3. 在每个高分视图里先用 SAM 把投影掩码收紧成贴合的 2D mask，再算**多尺度** CLIP 特征（整图、裁剪框、掩码内三种尺度），最后在 top-$k$ 个视图和多个尺度上平均，得到该实例的掩码级特征 $f_{m}\in\mathbb{R}^{D}$（下标 $m$ 表示这是一个 mask 级的向量）。

![OpenMask3D 掩码级 CLIP 特征蒸馏：3D 掩码投影到多视角、按可见性分数选视图、SAM 收紧边界框后取多尺度 CLIP 特征再平均](/coursenotes/cs231a-computer-vision-3d-sp25/l18-guest-lecture-foundation-models-3d/slide_p039.jpg)

于是场景表示变成「实例掩码 + 每个掩码一个 CLIP 向量」。查询时把文本编码 $f_t$ 与所有掩码特征 $f_{m}$ 做相似度，一次返回若干独立实例——「Footrest」能同时框出两个脚凳，「Side table with a flower vase」能挑出带花瓶的那张小桌。这一表示还带来一个工程红利：它是**可持久化的场景记忆**，扫一次房就能反复用自然语言查询。

课件第 41–42 页紧接着抛出一个问题：*Who sees the limitation?*，答案写在图上——掩码提案来自 Mask3D，而 Mask3D 是在「closed-world dataset」上训出来的。**提案生成器仍然带着封闭集的偏见**：类别之外的、特别小的、非典型的物体可能根本不会被提案出来。

### 路线三：把 SAM 蒸成 3D 提案器（Segment3D）

Segment3D（Huang, Peng, Takmaz, Tombari, Pollefeys, Song, Huang, Engelmann，ECCV 2024，arXiv:2312.17232）正是冲着这个瓶颈去的。课件第 44 页把问题写成三行：Problem 是人工标注数据集天然封闭（举例 ScanNet）；Question 是能不能直接用分割基础模型 SAM 做开放集 3D 分割；Challenge 是 **2D 图像空间与 3D 几何空间之间的域差**。

![Segment3D 第一阶段：RGB-D 图像经冻结的 SAM 得到细粒度 2D 掩码，投影成 3D 伪真值，用于监督部分点云上的 3D 分割模型](/coursenotes/cs231a-computer-vision-3d-sp25/l18-guest-lecture-foundation-models-3d/slide_p044.jpg)

它的两阶段方案很值得记住：**第一阶段**只在「部分点云」（单帧 RGB-D 反投影出的残缺点云）上训练，标签来自冻结 SAM 生成的细粒度 2D 掩码投影到 3D 的伪真值；**第二阶段**用第一阶段模型在高置信度下预测出的掩码作为自监督信号，在完整 3D 点云上微调，从而跨过「部分观测 → 完整场景」的域差。效果是掩码粒度明显比 Mask3D 更细（书架侧板、水壶提手、复印机操作面板这种部件级目标都能单独成块），而且**不需要任何人工 3D 标注**；把这些 3D 掩码再配上 CLIP 特征，就直接得到开放词汇 3D 分割。

三条路线放在一起看，其实就是第 10 讲「表示学习」在 3D 里的三种落地形态：蒸馏到**连续体**（NeRF 特征通道）、蒸馏到**点**（OpenScene 式逐点特征）、蒸馏到**掩码**（OpenMask3D / Segment3D）。粒度越结构化，越容易做实例级问答和下游操作；代价是必须先有一个好的提案器——而这恰恰是基础模型能补上的地方。

## 3D 场景图：从「有什么」到「能干什么」

第 49 页一句 *Is this all we need?* 引出讲者的第二个主题。分割回答的是 where / what，但机器人还需要知道 **how**：这个抽屉往哪个方向拉开、这个旋钮怎么转。

**SceneFun3D**（Delitzas, Takmaz, Tombari, Sumner, Pollefeys, Engelmann，CVPR 2024 Oral）为此建了一个数据集：在 ARKitScenes 的高分辨率 LiDAR 扫描上，为 **710 个真实室内场景**标注了 **14.8k 以上**的功能交互元素掩码，配备 8 类功能标签（`hook_turn`、`hook_pull`、`key_press`、`plug_in`、`pinch_pull`、`tip_push`、`foot_push`、`unplug`）、自然语言任务描述（如「Open the oven door」）以及描述「怎么动」的运动参数（交互点 + 方向向量）。数据集同时定义三个基准任务：功能分割、任务驱动的可供性定位、3D 运动估计。

**OpenFunGraph**（Zhang, Delitzas, Wang, Zhang, Ji, Pollefeys, Engelmann，CVPR 2025 Highlight，arXiv:2503.19199）把这件事推到开放词汇：输入 RGB-D + 3D 重建，输出**功能 3D 场景图** $\mathcal{G}=(\mathcal{O},\mathcal{I},\mathcal{R})$，节点是物体与交互元素，边是「controls / supplies water / open or close / provide power」这类带置信度的功能关系。

![OpenFunGraph 流水线：节点检测（物体与交互元素）→ 3D 融合 → 多视角描述（VLM/LLM）→ 局部与远程功能边推理](/coursenotes/cs231a-computer-vision-3d-sp25/l18-guest-lecture-foundation-models-3d/slide_p057.jpg)

它的核心思想只有一句话，写在第 57 页上：**Key idea: Leverage Knowledge from Foundation Models to infer Functionalities**。具体是三段式：先用开放词汇检测器在 RGB-D 帧上分别提出「物体」和「交互元素」两类节点并做 3D 融合；再用多视角 caption 让 VLM/LLM 给每个节点写描述；最后把局部与远程候选关系连同描述一起交给 LLM，推断哪些功能边成立。课件第 56 页的标注图（水龙头—水槽「control the water flow」、电热插座—壶/灶「provide power」、柜门把手—柜体「open or close」）就是它要预测的目标。这一步的意义在于：**功能知识本质上是常识，常识不该靠 3D 数据集从头标注，而应该从语言基础模型里借。**

## 表示的选择：为什么是超二次曲面

第 58–60 页把话题拉回「表示」这一课的核心问题：点云、多边形网格、NeRF、Gaussian Splats、包围盒、场景图各有短板——点云没有实例与部件结构，网格提取拓扑脆弱，NeRF 编辑困难，3DGS 的椭球虽然渲染快（第 17 讲），但形状自由度只有三个轴长，表达不了方块、圆柱、锥台这类真实家具。

**SuperDec**（Fedele, Sun, Guibas, Pollefeys, Engelmann，arXiv:2504.00992；课件标注为 arxiv'25，后收录于 ICCV 2025 Oral）给出的答案是超二次曲面。椭球/高斯的隐式方程只有 3 个形状参数：

$$
f(x,y,z)=\left(\frac{\lvert x\rvert}{a_x}\right)^{2}+\left(\frac{\lvert y\rvert}{a_y}\right)^{2}+\left(\frac{\lvert z\rvert}{a_z}\right)^{2}=1
$$

超二次曲面只多两个指数 $\epsilon_1,\epsilon_2$（共 5 个形状参数），却能在同一族里连续插值出立方体、圆柱、锥台、圆角块：

$$
f(x,y,z)=\left[\left(\frac{\lvert x\rvert}{a_x}\right)^{2/\epsilon_2}+\left(\frac{\lvert y\rvert}{a_y}\right)^{2/\epsilon_2}\right]^{\epsilon_2/\epsilon_1}+\left(\frac{\lvert z\rvert}{a_z}\right)^{2/\epsilon_1}=1
$$

方法本身仍是第 10 讲那套「查询 + 交叉注意力」：$N\times 3$ 点云经 PVCNN 得到 $N\times H$ 的点特征 $\mathcal{F}_{PC}$，$P$ 个可学习基元查询 $\mathcal{F}_{SQ}$ 反复做 cross-attention 与 self-attention，每个查询接 5 个回归头输出 scale $\mathbf{s}\in\mathbb{R}^{3}$、shape $\epsilon\in\mathbb{R}^{2}$、rotation $\mathbf{r}\in\mathbb{R}^{3}$、translation $\mathbf{t}\in\mathbb{R}^{3}$、存在性 $\alpha\in\mathbb{R}$，拼成 $P\times 12$ 的参数矩阵 $\Theta$；同时用查询与点特征的内积过 softmax 得到 $N\times P$ 的分割矩阵 $M$。

![SuperDec 网络结构：PVCNN 点特征与可学习超二次曲面查询交叉注意力，输出 N×P 分割矩阵与 P×12 基元参数](/coursenotes/cs231a-computer-vision-3d-sp25/l18-guest-lecture-foundation-models-3d/slide_p063.jpg)

课件第 61 页给了一个非常直观的压缩比：**1'000'000 个点的室内点云 → 300 个超二次曲面基元**。第 64–65 页展示它的下游价值：基元是带位姿的凸性近似，可以直接喂给机器人做路径规划与抓取姿态估计，也能把台灯、椅子、圆桌分解成少数几个部件级基元。换句话说，**这是把「理解」的输出格式换成「可操作」的输入格式**。

## 可控 3D 生成与编辑

最后三页（第 66–68 页）把方向反过来：不再从真实场景提取知识，而是生成可控的 3D 资产。课件列出三种控制接口：

- **Text control**：一句风格描述（cartoonish 火箭、红色圆顶、青色圆柱中段）生成完整带纹理资产；
- **Image control**：一张海盗船插画生成对应的 3D 模型；
- **Spatial control**：用一组几何基元（形状与位置都可见的粗结构）作为空间引导，生成 Airplane / Cat / Red-white Rocket / Duck / Snowman / House with a tree 等资产。

第 68 页把空间控制的价值讲得最清楚：`add slats`、`add armrests`、`2x width`、`add pillow` 这几条编辑指令，都是**先在基元层改结构、再让生成器补外观**，得到的椅背竖条、扶手、双人宽度、抱枕都能精确落在指定位置。课件标题写的是「Spatial guidance enables fine-grained control over the object geometry」——文本太含糊、图像难编辑，而几何基元既精确又可改。

需要说明的是：**第 67–68 页没有标注论文出处**，因此本笔记不给出方法名与定量结论。讲者个人主页把这条线后续整理成了 *SpaceControl: Introducing Test-Time Spatial Control to 3D Generative Modeling*（arXiv:2512.05343，ICLR 2026），其关键词正是「geometric primitives」与免训练的 test-time 空间控制，与课件这两页的思路一致，可作延伸阅读。

## 评测与数据集

这讲课件没有一页定量结果表，全部结论以 demo、可视化与失败案例呈现，因此下面只列「哪个工作在哪份数据上被评测」这一事实性信息，数字均来自论文本身：

| 工作 | 主要数据 / 基准 | 课件里给出的证据形式 |
|---|---|---|
| Mask3D | ScanNet、ScanNet200、S3DIS、STPLS3D | 掩码与点特征矩阵的机制图（第 8–14 页） |
| RoomFormer | Structured3D、SceneCAD | 3D 扫描 → 矢量化平面图 + 房间/门窗语义（第 16–22 页） |
| Human3D | 合成人体点云训练，真实扫描测试 | 真实场景多人部件分割（第 23–26 页） |
| OpenNeRF | 真实扫描场景（与 LERF、OpenScene 比较） | 交互式检索 demo（第 35–36 页） |
| OpenMask3D | ScanNet200、Replica | 长短语查询的实例掩码可视化（第 37–42 页） |
| Segment3D | 室内 RGB-D 场景（训练标签全部自动生成） | Mask3D 与 Segment3D 的掩码粒度对比（第 46–48 页） |
| SceneFun3D | 自建：710 场景、14.8k+ 交互标注 | 功能标签、任务描述、运动参数三栏（第 51–53 页） |
| OpenFunGraph | 扩展 SceneFun3D + 自建 FunGraph3D | 功能场景图标注与流水线图（第 54–57 页） |
| SuperDec | 室内扫描与物体级点云 | 1M 点 → 300 基元、路径规划与抓取（第 61–65 页） |

## 局限与展望

课件自己点出的局限有三条，加上笔记归纳的一条：

1. **提案器仍是瓶颈**。OpenMask3D 的掩码来自封闭集训练的 Mask3D（第 42 页红字标注）；Segment3D 用 SAM 伪标签绕开人工标注，但 2D→3D 的域差要靠两阶段自监督弥补，投影与合并过程会引入噪声。
2. **蒸馏有信息损失**。CLIP 特征是整图/裁剪级的，掩码越小、遮挡越重，特征越不可靠；OpenMask3D 用可见性分数与多尺度平均来缓解，而不是解决。
3. **功能推理依赖语言模型的常识**。OpenFunGraph 把关系判断交给 LLM，好处是零标注获得开放词汇功能边，代价是幻觉与置信度不可控——课件第 55 页的边上专门写了 `most confident` / `less confident`。
4. **理解与生成开始共用同一套几何基元**。SuperDec 的场景分解与空间控制生成用的是同一族紧凑基元，而课件第 60 页把点云、网格、NeRF、Gaussian Splats、包围盒、场景图并列摆出，正说明讲者关心的是「渲染友好的表示」与「可查询、可操作的表示」之间的落差：**面向开放世界的 3D 表示要既紧凑、又可查询、还可编辑**。

## 要点回顾

- 本讲是客座讲座，讲者 Francis Engelmann（时任 Stanford PostDoc），主题是「基础模型 + 3D 视觉」，分开放世界 3D 场景理解与可控 3D 生成两条线。
- 3D 场景理解的三类任务：实例分割（Mask3D）、矢量化平面图（RoomFormer）、人体部件分割（Human3D），动机是家庭机器人与以人为中心的 AI。
- 封闭集失败的本质：类别表写死在标注与输出层里，`stair rail ≠ fishing boat` 这类错误无法通过加数据修复。
- 开放词汇的钥匙是 VLM 的对齐空间（CLIP / SigLIP），把分类变成归一化特征的内积检索，这正是第 10 讲表示学习的直接延伸。
- 蒸馏到连续体：OpenNeRF 给 NeRF 加 CLIP 特征通道，靠渲染新视角得到像素级特征，比反投影更稳；相对 LERF、OpenScene 提升至少 +4.9 mIoU。
- 蒸馏到掩码：OpenMask3D = class-agnostic 3D 掩码提案 + 投影到 top-k 可见视图 + SAM 收紧 + 多尺度 CLIP 特征平均，实现短语级实例检索。
- 蒸馏到提案器本身：Segment3D 用冻结 SAM 生成 2D 掩码投影成 3D 伪标签，两阶段（部分点云预训练 → 完整场景自监督微调）实现无人工标签的细粒度 3D 分割。
- 从语义到功能：SceneFun3D（710 场景、14.8k+ 交互标注、8 类功能标签 + 任务描述 + 运动参数，CVPR 2024 Oral）定义功能分割、可供性定位、运动估计三个任务。
- OpenFunGraph（CVPR 2025 Highlight）用 VLM/LLM 推断功能边，输出 $\mathcal{G}=(\mathcal{O},\mathcal{I},\mathcal{R})$ 形式的开放词汇功能 3D 场景图。
- SuperDec 用超二次曲面基元替代点云/网格/高斯：仅多 2 个形状指数就能表达立方体到锥台，1M 点可压到约 300 个基元，输出 $P\times 12$ 参数与 $N\times P$ 分割矩阵，直接服务路径规划与抓取。
- 可控 3D 生成三种接口：文本、图像、空间基元；空间控制的优势是「先改结构、再生成外观」，可做 add slats / add armrests / 2x width / add pillow 这类部件级编辑（课件未标注论文出处）。
- 全讲课件没有定量结果表，证据以 demo 与可视化为主；具体数字需回到原论文核对。

## 参考资料

- 课件：CS231A Lecture 18（Guest Lecture），Francis Engelmann，2025-06-04，Stanford CS231A 2025 春季 [syllabus](https://web.stanford.edu/class/cs231a/syllabus.html)；slides PDF：[2025.06.04 - GuestLecture_stanfordcs231a_.pdf](https://web.stanford.edu/class/cs231a/lectures_2025/2025.06.04%20-%20GuestLecture_stanfordcs231a_.pdf)
- 讲者主页：[Francis Engelmann](https://francisengelmann.github.io)
- Mask3D: Mask Transformer for 3D Semantic Instance Segmentation — Schult, Engelmann, Hermans, Litany, Tang, Leibe, ICRA 2023，[arXiv:2210.03105](https://arxiv.org/abs/2210.03105)
- Connecting the Dots: Floorplan Reconstruction Using Two-Level Queries（RoomFormer）— Yue, Kontogianni, Schindler, Engelmann, CVPR 2023，[arXiv:2211.15658](https://arxiv.org/abs/2211.15658)，[代码](https://github.com/ywyue/RoomFormer)
- 3D Segmentation of Humans in Point Clouds with Synthetic Data（Human3D）— Takmaz, Schult, Kaftan, Akçay, Leibe, Sumner, Engelmann, Tang, ICCV 2023
- Learning Transferable Visual Models From Natural Language Supervision（CLIP）— Radford et al., ICML 2021
- Sigmoid Loss for Language Image Pre-Training（SigLIP）— Zhai et al., ICCV 2023
- Scaling Open-Vocabulary Image Segmentation with Image-Level Labels（OpenSeg）— Ghiasi, Gu, Cui, Lin, ECCV 2022
- Segment Anything（SAM）— Kirillov et al., ICCV 2023
- OpenNeRF: Open Set 3D Neural Scene Segmentation with Pixel-Wise Features and Rendered Novel Views — Engelmann et al., ICLR 2024，[arXiv:2404.03650](https://arxiv.org/abs/2404.03650)
- OpenScene: 3D Scene Understanding with Open Vocabularies — Peng, Genova, Jiang et al., CVPR 2023（OpenNeRF 的对比基线）
- OpenMask3D: Open-Vocabulary 3D Instance Segmentation — Takmaz, Fedele, Sumner, Pollefeys, Tombari, Engelmann, NeurIPS 2023，[arXiv:2306.13631](https://arxiv.org/abs/2306.13631)
- Segment3D: Learning Fine-Grained Class-Agnostic 3D Segmentation without Manual Labels — Huang, Peng, Takmaz, Tombari, Pollefeys, Song, Huang, Engelmann, ECCV 2024，[arXiv:2312.17232](https://arxiv.org/abs/2312.17232)
- SceneFun3D: Fine-grained Functionality and Affordance Understanding in 3D Scenes — Delitzas, Takmaz, Tombari, Sumner, Pollefeys, Engelmann, CVPR 2024 (Oral)
- OpenFunGraph: Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces — Zhang, Delitzas, Wang, Zhang, Ji, Pollefeys, Engelmann, CVPR 2025 (Highlight)，[arXiv:2503.19199](https://arxiv.org/abs/2503.19199)
- SuperDec: 3D Scene Decomposition with Superquadric Primitives — Fedele, Sun, Guibas, Pollefeys, Engelmann，[arXiv:2504.00992](https://arxiv.org/abs/2504.00992)，ICCV 2025 (Oral)，[代码](https://github.com/elisabettafedele/superdec)
- 延伸阅读（课件未标注出处）：SpaceControl: Introducing Test-Time Spatial Control to 3D Generative Modeling — [arXiv:2512.05343](https://arxiv.org/abs/2512.05343)，ICLR 2026
