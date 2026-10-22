// 由脚本生成：读书笔记侧边栏。新增书籍时在 readingnotes/ 下建目录并追加一项。

export const readingNotesSidebar = [
  {
    text: '书架',
    items: [{ text: '全部书籍', link: '/readingnotes/' }],
  },
  {
    text: 'Foundations of Computer Vision',
    collapsed: false,
    items: [
      { text: '书籍首页', link: '/readingnotes/foundations-of-computer-vision/' },
    ],
  },
  {
    text: '前言',
    collapsed: true,
    items: [
      { text: '前言', link: '/readingnotes/foundations-of-computer-vision/preface' },
      { text: '符号约定', link: '/readingnotes/foundations-of-computer-vision/notation' },
    ],
  },
  {
    text: 'Part I · 基础',
    collapsed: true,
    items: [
      { text: '第 1 章 · 视觉的挑战', link: '/readingnotes/foundations-of-computer-vision/ch01-the-challenge-of-vision' },
      { text: '第 2 章 · 简单视觉系统', link: '/readingnotes/foundations-of-computer-vision/ch02-a-simple-vision-system' },
      { text: '第 3 章 · 观看图像', link: '/readingnotes/foundations-of-computer-vision/ch03-looking-at-images' },
      { text: '第 4 章 · 计算机视觉与社会', link: '/readingnotes/foundations-of-computer-vision/ch04-computer-vision-and-society' },
    ],
  },
  {
    text: 'Part II · 图像形成',
    collapsed: true,
    items: [
      { text: '第 5 章 · 成像', link: '/readingnotes/foundations-of-computer-vision/ch05-imaging' },
      { text: '第 6 章 · 镜头', link: '/readingnotes/foundations-of-computer-vision/ch06-lenses' },
      { text: '第 7 章 · 作为线性系统的相机', link: '/readingnotes/foundations-of-computer-vision/ch07-cameras-as-linear-systems' },
      { text: '第 8 章 · 颜色', link: '/readingnotes/foundations-of-computer-vision/ch08-color' },
    ],
  },
  {
    text: 'Part III · 学习基础',
    collapsed: true,
    items: [
      { text: '第 9 章 · 学习导论', link: '/readingnotes/foundations-of-computer-vision/ch09-introduction-to-learning' },
      { text: '第 10 章 · 基于梯度的学习算法', link: '/readingnotes/foundations-of-computer-vision/ch10-gradient-based-learning-algorithms' },
      { text: '第 11 章 · 泛化问题', link: '/readingnotes/foundations-of-computer-vision/ch11-the-problem-of-generalization' },
      { text: '第 12 章 · 神经网络', link: '/readingnotes/foundations-of-computer-vision/ch12-neural-networks' },
      { text: '第 13 章 · 作为分布变换器的神经网络', link: '/readingnotes/foundations-of-computer-vision/ch13-neural-networks-as-distribution-transformers' },
      { text: '第 14 章 · 反向传播', link: '/readingnotes/foundations-of-computer-vision/ch14-backpropagation' },
    ],
  },
  {
    text: 'Part IV · 图像处理基础',
    collapsed: true,
    items: [
      { text: '第 15 章 · 线性图像滤波', link: '/readingnotes/foundations-of-computer-vision/ch15-linear-image-filtering' },
      { text: '第 16 章 · 傅里叶分析', link: '/readingnotes/foundations-of-computer-vision/ch16-fourier-analysis' },
    ],
  },
  {
    text: 'Part V · 线性滤波器',
    collapsed: true,
    items: [
      { text: '第 17 章 · 模糊滤波器', link: '/readingnotes/foundations-of-computer-vision/ch17-blur-filters' },
      { text: '第 18 章 · 图像导数', link: '/readingnotes/foundations-of-computer-vision/ch18-image-derivatives' },
      { text: '第 19 章 · 时域滤波器', link: '/readingnotes/foundations-of-computer-vision/ch19-temporal-filters' },
    ],
  },
  {
    text: 'Part VI · 采样与多尺度',
    collapsed: true,
    items: [
      { text: '第 20 章 · 图像采样与混叠', link: '/readingnotes/foundations-of-computer-vision/ch20-image-sampling-and-aliasing' },
      { text: '第 21 章 · 下采样与上采样', link: '/readingnotes/foundations-of-computer-vision/ch21-downsampling-and-upsampling' },
      { text: '第 22 章 · 滤波器组', link: '/readingnotes/foundations-of-computer-vision/ch22-filter-banks' },
      { text: '第 23 章 · 图像金字塔', link: '/readingnotes/foundations-of-computer-vision/ch23-image-pyramids' },
    ],
  },
  {
    text: 'Part VII · 视觉神经网络架构',
    collapsed: true,
    items: [
      { text: '第 24 章 · 卷积神经网络', link: '/readingnotes/foundations-of-computer-vision/ch24-convolutional-neural-nets' },
      { text: '第 25 章 · 循环神经网络', link: '/readingnotes/foundations-of-computer-vision/ch25-recurrent-neural-nets' },
      { text: '第 26 章 · Transformer', link: '/readingnotes/foundations-of-computer-vision/ch26-transformers' },
    ],
  },
  {
    text: 'Part VIII · 图像概率模型',
    collapsed: true,
    items: [
      { text: '第 27 章 · 统计图像模型', link: '/readingnotes/foundations-of-computer-vision/ch27-statistical-image-models' },
      { text: '第 28 章 · 纹理', link: '/readingnotes/foundations-of-computer-vision/ch28-textures' },
      { text: '第 29 章 · 概率图模型', link: '/readingnotes/foundations-of-computer-vision/ch29-probabilistic-graphical-models' },
    ],
  },
  {
    text: 'Part IX · 生成模型与表示学习',
    collapsed: true,
    items: [
      { text: '第 30 章 · 表示学习', link: '/readingnotes/foundations-of-computer-vision/ch30-representation-learning' },
      { text: '第 31 章 · 知觉组织', link: '/readingnotes/foundations-of-computer-vision/ch31-perceptual-grouping' },
      { text: '第 32 章 · 生成模型', link: '/readingnotes/foundations-of-computer-vision/ch32-generative-models' },
      { text: '第 33 章 · 生成建模与表示学习的交汇', link: '/readingnotes/foundations-of-computer-vision/ch33-generative-modeling-meets-representation-learning' },
      { text: '第 34 章 · 条件生成模型', link: '/readingnotes/foundations-of-computer-vision/ch34-conditional-generative-models' },
    ],
  },
  {
    text: 'Part X · 学习型视觉的挑战',
    collapsed: true,
    items: [
      { text: '第 35 章 · 数据偏差与偏移', link: '/readingnotes/foundations-of-computer-vision/ch35-data-bias-and-shift' },
      { text: '第 36 章 · 训练鲁棒性与泛化性', link: '/readingnotes/foundations-of-computer-vision/ch36-training-for-robustness' },
      { text: '第 37 章 · 迁移学习与适配', link: '/readingnotes/foundations-of-computer-vision/ch37-transfer-learning-and-adaptation' },
    ],
  },
  {
    text: 'Part XI · 理解几何',
    collapsed: true,
    items: [
      { text: '第 38 章 · 图像与几何的表示', link: '/readingnotes/foundations-of-computer-vision/ch38-representing-images-and-geometry' },
      { text: '第 39 章 · 相机建模与标定', link: '/readingnotes/foundations-of-computer-vision/ch39-camera-modeling-and-calibration' },
      { text: '第 40 章 · 立体视觉', link: '/readingnotes/foundations-of-computer-vision/ch40-stereo-vision' },
      { text: '第 41 章 · 单应矩阵', link: '/readingnotes/foundations-of-computer-vision/ch41-homographies' },
      { text: '第 42 章 · 单视图度量学', link: '/readingnotes/foundations-of-computer-vision/ch42-single-view-metrology' },
      { text: '第 43 章 · 单图深度估计', link: '/readingnotes/foundations-of-computer-vision/ch43-depth-from-a-single-image' },
      { text: '第 44 章 · 多视图几何与运动恢复结构', link: '/readingnotes/foundations-of-computer-vision/ch44-multiview-geometry-and-sfm' },
      { text: '第 45 章 · 辐射场', link: '/readingnotes/foundations-of-computer-vision/ch45-radiance-fields' },
    ],
  },
  {
    text: 'Part XII · 理解运动',
    collapsed: true,
    items: [
      { text: '第 46 章 · 运动估计', link: '/readingnotes/foundations-of-computer-vision/ch46-motion-estimation' },
      { text: '第 47 章 · 三维运动及其二维投影', link: '/readingnotes/foundations-of-computer-vision/ch47-3d-motion-and-its-2d-projection' },
      { text: '第 48 章 · 光流估计', link: '/readingnotes/foundations-of-computer-vision/ch48-optical-flow-estimation' },
      { text: '第 49 章 · 学习估计运动', link: '/readingnotes/foundations-of-computer-vision/ch49-learning-to-estimate-motion' },
    ],
  },
  {
    text: 'Part XIII · 场景理解与语言',
    collapsed: true,
    items: [
      { text: '第 50 章 · 目标识别', link: '/readingnotes/foundations-of-computer-vision/ch50-object-recognition' },
      { text: '第 51 章 · 视觉与语言', link: '/readingnotes/foundations-of-computer-vision/ch51-vision-and-language' },
    ],
  },
  {
    text: 'Part XIV · 研究、写作与演讲',
    collapsed: true,
    items: [
      { text: '第 52 章 · 如何做研究', link: '/readingnotes/foundations-of-computer-vision/ch52-how-to-do-research' },
      { text: '第 53 章 · 如何写论文', link: '/readingnotes/foundations-of-computer-vision/ch53-how-to-write-papers' },
      { text: '第 54 章 · 如何做报告', link: '/readingnotes/foundations-of-computer-vision/ch54-how-to-give-talks' },
    ],
  },
  {
    text: 'Part XV · 收官',
    collapsed: true,
    items: [
      { text: '第 55 章 · 简单视觉系统回顾', link: '/readingnotes/foundations-of-computer-vision/ch55-a-simple-vision-system-revisited' },
    ],
  },
]
