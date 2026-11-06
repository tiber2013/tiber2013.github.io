---
title: "第 14 讲 · 数据 II：转换、去重与训练管道"
description: "本讲把语言模型的数据工作拆成一条管道：转换、过滤、去重、混合。原始网页和 PDF 并不是干净文本，转换本质上是有损的工程过程；过滤则是用少量“好数据”定义目标，再训练轻量分类器去海量原始数据中筛选。去重的关键挑战是把两两…"
---

# 第 14 讲 · 数据 II：转换、去重与训练管道

> **Stanford CS336 语言模型从零构建 · L14** ｜ 讲者 Stanford Online ｜ 85 分钟 ｜ [原视频](https://www.youtube.com/watch?v=5sxHosTLPF8)

> **v2 详写版** · 按 20 分钟分段详写，配图覆盖全片；文末附「对照参考笔记补遗（Spring 2025 版）」，记录与 2025 版讲义笔记的核对结果。

## 核心结论

本讲是 CS336 数据主题的第二讲，承接上节课「数据从哪里来」，转而讲「拿到原始数据后怎么处理」，把预训练管道拆成转换、过滤、去重、混合四步。转换的对象并非现成文本，而是网页、PDF 与代码仓库，线性化必然有损，PDF 还须渲染与文字识别补救；过滤以少量高质量样本为目标、用便宜分类器给海量语料打分，但阈值不通用，取决于训练多少 token；去重借哈希把两两比较压成线性时间，再让碰撞概率对齐相似度并锐化成阈值判定；混合为多来源分配权重，最需提防小而精的来源在长训练里被反复重复而过拟合，于是有封顶、小规模代理回归与模拟重复等对策。讲者随后转入后训练，数据从任务无关的网页抓取变成教师模型对构造任务生成的合成回答，并以推理、代码智能体数据集说明环境成本与「有能力的模型未必是好教师」这一反直觉结论。贯穿全讲的提醒是，这些框架只是地形图，真正决定质量的领域相关工作，仍得靠翻看具体样例打磨出来。

## 课程定位与数据管道总览 [00:05 - 01:17]

本讲是 CS336 数据主题的第二天，讲者开场用一句 "Okay, let's get started" 直接进入正题，并明确 "today is the second day on data"。要理解本段的定位，必须先接住上节课的结论：数据并不会凭空从天上掉下来（data doesn't really just fall from the sky），我们必须认真追问它从哪里来。一般互联网由一个个 live services（在线服务）构成，这些服务上的数据要么被 dump（整体导出），要么被 crawl（抓取），之后还要经过一道 processing（加工处理）才谈得上可用；此外还有一堆社会性、法律性的考量，例如 terms of service（服务条款）、copyright（版权），要么取得 license（授权许可），要么诉诸 fair use（合理使用）。正因为获取环节已经如此复杂，本讲才把镜头推进到加工环节本身。

幻灯片把这一结构写得比口语更规整，标题是 `Lecture 14: Data II`。它以一段可执行的伪代码开篇，`def main():` 下面先是一组 import，随后把整堂课拆成四个函数调用，这正是本讲要逐一展开的骨架：

```python
from edtrace import text, image, link
from lecture_13 import the_pile
from lecture_util import article_link, post_link
from references import dolma_2024, the_pile_2020, dclm_2024

def main():
    transformation()
    filtering()
    deduplication()
    data_mixing()
```

这里 `transformation()`、`filtering()`、`deduplication()`、`data_mixing()` 分别对应 transformation（转换）、filtering（过滤）、deduplication（去重）、mixing（混合）四个环节，也就是讲者口中的 data pipeline（数据管道）。幻灯片 "Last lecture" 部分把上节课的因果链条压缩成一行：`Live service (e.g., GitHub) → dump/crawl (e.g., GitHub Archive) → processed data (e.g., The Stack)`，并单列 `Considerations: terms of service, copyright (licenses or fair use)`；"This lecture" 部分则列出两块，一是 `Data pipeline: transformation, filtering, deduplication, mixing`，二是 `Mid-training + SFT: synthetic data`。讲者特别提醒，前半段（也就是本段所覆盖的部分）主要讲的是 pre-training（预训练）数据，往后才会用 post-training data（后训练数据）、尤其是当下各家怎么使用 synthetic data（合成数据）来收尾。

把这些内容与前后衔接起来看：上节课解决的是「数据从哪里来」，本讲解决「拿到原始数据之后怎么处理」，而本段（00:00–21:11）只覆盖了管道中最靠前的转换与过滤，去重与混合留到后面。一个容易踩的坑是，把这四个函数名误当成严格串行、各自独立、一次就干净完成的工序；实际上幻灯片里 `def main():` 只是教学用的抽象骨架，真实的数据工作（real data work）被讲者在 Summary 里形容为高度 domain-specific（领域相关）、需要不断翻看具体样例（looking at examples, etc.），远没有这四个调用看起来那么整齐。另一个细节是，幻灯片标题带 "II"，说明这是承接 Lecture 13 的第二讲，若把它当成独立入门材料，很容易漏掉前置的抓取与合规背景。

![Lecture 14: Data II 课程大纲幻灯片，含 def main() 数据管道四函数与上节课/本节课内容对比](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0001_00-00-39.jpg)

## 数据转换（一）：HTML 到文本的有损线性化 [01:17 - 04:19]

讲者从一句反直觉的提醒切入：raw data does not come as text（原始数据并不是以文本形态到来的）。即便你已经抓到了东西，打开 Common Crawl（公共网页抓取存档）内部看，里面也不是纯文本，而是 HTML，有时是 PDF，遇到 GitHub 这类站点还可能是 directories（目录树，即 code repositories 代码仓库）。幻灯片用 `def transformation():` 把这句话写得更精确：`Raw data does not come as text. It is HTML, PDF (arxiv), or directories (code repositories).` 这里 PDF 特意标注了 arxiv，暗示学术论文是常见的 PDF 来源。转换环节的绝大部分张力来自 HTML，因为 the most of the web is in HTML（网页绝大多数是 HTML），所以 "HTML to text (main one)" 被单列为最主要的工作项。

HTML 转文本的核心动作可以抽象成一个映射：把具有层级结构、且渲染后带有视觉布局的 HTML 文档，压成一个一维的 token 序列。若记转换函数为 $f_{\text{rule}}$，则

$$
f_{\text{rule}}:\ \text{HTML (hierarchical/visual)} \ \xrightarrow{\text{linearize}}\ (t_1, t_2, \ldots, t_n).
$$

这一步之所以 inherently lossy（本质有损），是因为 HTML 原本既有 DOM 的层级（hierarchical），又有渲染后的二维视觉（visual），而语言模型只吃一维 token 序列，线性化必然丢弃结构信息。具体做法是启发式的：remove boilerplate（去掉模板化噪声，例如 navigation 导航、ads 广告），再 extract content（抽取页面主体内容）。讲者强调「什么算 content」并不总清楚——通常要去掉 footers、headers 与 menus，但导航元素有时反而有助于模型学习「网页长什么样」，贸然删掉未必正确；网页里的 images 和 tables 更是两难。简单表格还能用 Markdown 渲染出来，一旦遇到 nested tables（嵌套表格）就相当棘手，工程上往往只能 approximate（近似）或干脆 give up（放弃）某部分。这也解释了为什么转换后紧接着还「远远没完」（you're not done, you're far from done），必须进入下一章的过滤。

工程实现上，HTML 到文本通常是 rule-based（规则驱动）的，幻灯片把工具名写得很全：`Tools (rule-based): trafilatura, resiliparse, jusText, lynx, etc.`，并标注 `Accuracy matters: [Li+ 2024]`。讲者给出的理由有二：规则处理器 very fast（极快），而且这一步不需要太多 intelligence（智能），一般规则就够用；他也留了口子，认为 model-based interventions（基于模型的干预）在这里也许有用，但前提是必须非常快、且能做出更聪明的事。工具之间是有准确率差异的：转写里提到 Brazil parse / Traflera（即 resiliparse 与 trafilatura 的口语转写混淆），并指出在 extended DCLM eval 上 resiliparse 表现优于其他工具；幻灯片底部还给出一张 Text Extraction 对比表，列头是 `CORE` 与 `EXTENDED`，其中 `WET files` 一行对应数字 `20.7` 与 `12.2`——这些具体数值是画面独有、口语没念到的增量。讲者同时提醒，只要用规则处理就一定有 failure rate（失败率），翻看真实数据会看到各种 imperfections（瑕疵），所以「选哪个抽取工具」本身就会影响下游数据质量。

这一章与前后内容的关系是：它承接上节课 dump/crawl 得到的原始 HTML/PDF，输出「text」，而这份 text 正是过滤阶段要打分的 raw data $R$。易错点有三层：其一是把转换当成无损的格式转换，忽略了线性化对层级与视觉信息的永久丢弃；其二是把「正文」当成客观存在，忽略了导航、菜单、表格、图片在不同用途下可能既有噪声也有信号；其三是误以为规则方法「够快」就等于「够准」，而 [Li+ 2024] 的评测和 20.7 / 12.2 的对比数字恰恰说明抽取器的选择会实打实地改变结果。

![def transformation() 幻灯片：HTML 到文本的去 boilerplate、有损线性化、规则工具列表与准确率表格](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0003_00-03-18.jpg)

## 数据转换（二）：PDF、FinePDFs 与 OCR 的代价 [04:19 - 06:43]

转换的第二类难点是 PDF。讲者提到 Hugging Face 的一项工作发布了名为 FinePDFs 的数据集（幻灯片标题写作 `FinePDFs [post]`，口语里被念成 "find PDFs"）。PDF 文件若不是用 PDF reader 打开、而是直接看源码，呈现的是一串绘制指令而非文字，因此需要被 render（渲染）成可训练文本。PDF 和网页一样散布在 internet 上，Common Crawl 里也确实收了一些 PDF；但讲者提醒一个实际陷阱：Common Crawl 总体偏重 text，而且当你只拿到一个 URL、且它不带扩展名时，在 fetch 之前往往根本不知道它是不是 PDF。

幻灯片用左右两栏把困难讲得最清楚：左栏 `PDF SOURCE CODE (COMPACT & ANNOTATED)`，右栏 `RENDERED PDF OUTPUT (VISUAL LAYOUT)`，中间用彩色箭头把源码片段映射到渲染后的视觉元素。左栏按六个层次组织：`1. Header & File Type` 里是 `%PDF-1.4` 版本声明；`2. Objects & Fonts (Indirect References)` 里是 `1 0 obj << /F1 2 0 R /F2 3 0 R >>` 这类字体资源映射，以及 `/BaseFont /Helvetica-Bold`；`3. Page & Content Stream (The "Paint" Instructions)` 里有页面字典 `/Type /Page /MediaBox [0 0 612 792] /Contents 10 0 R`，其中 `[0 0 612 792]` 是页面尺寸、`/Contents 10 0 R` 指向绘制指令流、字体与图像资源（`/XObject << /Im1 6 0 R >>`）都以间接引用挂上去。真正画出文字的是内容流里的一组操作符：

```text
BT /F1 24 Tf 50 738 Td (HEADER TEXT (BOLD)) Tj ET
BT /F1 12 Tf 50 708 Td (Body paragraph 1 text...) Tj ET
q 100 400 200 150 re W n /Im1 Do Q
BT /F3 10 Tf 100 390 Td (Caption for image...) Tj ET
```

可以看到文字被拆成一条条 `BT ... Tj ET`，坐标如 `50 738 Td`、`50 708 Td` 是绝对定位；图像则是一段 `6 0 obj << /Type /XObject /Subtype /Image ... /Length 9990 >>` 的二进制外部资源。左栏底部红框一句话点破本质，即 `WHY PARSING IS HARD`：`No explicit semantic structure (e.g. <p>, <h1>), only absolute positioning & "paint" commands. Reading order is not guaranteed. Text is fragmented. Layout must be heuristically inferred from geometry.` 也就是说，PDF 没有 `<p>`、`<h1>` 这类语义标签，只有绝对坐标与绘制命令；阅读顺序不被保证，文本是碎片化的，版式只能靠几何关系启发式推断。右栏于是把每个可视元素都标成「推断出来的」：`Visual "Header" (inferred by typography)`、`Visual "Separator" (Drawn line)`、`Visual "Column 1"` 与 `Visual "Column Z"`、`Visual "Image" (Parsed XObject)`，以及 `Visual "Caption" (Inferred by proximity)`——连图片说明文字都只能靠「空间邻近」来猜归属。这正好呼应讲者的对比：HTML 至少还有 H1、P 等 tags 提供语义信息，而 PDF by design all about layout（天生就是为版式设计的），语义结构反而丢了，因此 PDF 需要比网页更多的 cleanup and filtering（清洗与过滤），`A lot of layout information is missing`（大量版式信息缺失）。

工程上的处理清单写在幻灯片底部 bullet 里，属于口语没逐条念到的独有信息：`Source: Common Crawl`；`Recrawl truncated PDFs (since they are big)`；`OCR (RolmOCR) using a VLM or Docling (make these run fast)`；`Lots of cleanup and filtering`；`A lot of layout information is missing`。讲者解释了其中两条的来由：Common Crawl 里不少 PDF 因为体积大而被 truncated（截断），所以 re-crawling（重新抓取）是必需的；还有一部分 PDF 本质上是扫描件（scans），也就是图像，得上 OCR（光学字符识别），通常要动用 VLM（视觉语言模型），成本明显高于前面纯文本的处理。高亮行提到的 RolmOCR、VLM、Docling 正是这类工具，括号里 "make these run fast" 再次强调速度约束。价值判断上，讲者说 PDF 在整个互联网里占比 very small fraction（很小），却非常 valuable（有价值）：愿意费事去做一个 PDF，通常意味着发布者「确实有值得说的内容」，所以 average PDF 的质量一般高于普通 HTML 文件。

这一章与前一章并列构成 transformation 阶段：HTML 靠规则去 boilerplate，PDF 靠渲染 + OCR + 大量清洗，二者产出的 text 汇合后才进入过滤。易错点尤其密集：不能假设 PDF 有阅读顺序或语义标签；扫描件必须走昂贵的 VLM/OCR；Common Crawl 中的截断件需要先重爬；把「版式」误当「语义」会系统性丢信息；此外，占比小但质量高这一反差，也提醒不要因为 PDF 量少就在混合阶段直接忽略它。

![FinePDFs 幻灯片：PDF 源码操作符与渲染后视觉布局的彩色箭头映射，及 WHY PARSING IS HARD 红框说明](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0004_00-04-38.jpg)

## 数据过滤的通用框架：目标数据、评分函数与两类分类器 [06:43 - 11:27]

转换完成后得到的 text 只是「原料」，下一步 filtering（过滤）才决定哪些 token 真正进入训练。讲者先把问题抽象成一个几乎能装下所有过滤方法的 building block（算法构件），幻灯片用 `def filtering():` 与一句话给出定义：`Given some target data T and lots of raw data R, find subset T' of R similar to T.` 配套的集合示意图里，粉色大椭圆是 `Raw data R`，旁边一个黄色小椭圆是 `Target data T`，而落在 $R$ 内部的小黄椭圆标作 `More data T'`——直观含义是：$T$ 是少量、高质量的「样板」，$R$ 是上一步转换出来的 fresh shipment of tokens（刚到货的一大批 token），目标是在 $R$ 里找出与 $T$ 相似、但规模更大的子集 $T'$。形式化地写，就是求 $T' \subseteq R$，使 $T'$ 在某种相似度意义下逼近 $T$。

过滤的动机被幻灯片和讲者共同归为三类应用：`Language identification (English versus rest)`（语言识别，例如训练英语或德语模型时剔除非目标语言）、`Quality filtering (high quality versus low quality)`（质量过滤，要 encyclopedia 式内容、不要 spam 垃圾）、`Toxicity filtering (non-toxic versus toxic)`（毒性过滤，避免用 internet 上的 nasty content 训练）。对过滤算法的 desiderata（设计要求）有两条，第二条被高亮：`Generalize from the target data (want T and T' to be different)`——要能从目标数据泛化，不能只把 $T$ 原样背回来，否则 $T'$ 就等于 $T$，没有扩容意义；以及 `Extremely fast (have to run it on R, which is huge)`——必须极快，因为它要跑在海量 $R$ 上。讲者把「海量」量化到约 `100 trillion tokens`（100 万亿 token）规模，并说最终保留下来的往往只是 single digits fraction（个位数百分比）的极小一部分。幻灯片还给出一篇综述出处 `Survey paper on data selection [Albalak+ 2024]`。

在这个骨架下，`General framework` 被拆成两步：`1. Estimate some model based on R and T and derive a scoring function`（基于 $R$ 与 $T$ 估计某种模型，导出评分函数），`2. Keep examples in R based on their score`（按分数保留 $R$ 中的样本）。`Types of classifiers` 则列出两类，都是画面独有、需要原样保留的表达式：

$$
\text{Generative model of } T\ \text{(KenLM)}:\quad \operatorname{score}(x)=p_T(x),
$$

$$
\text{Simple classifier (FastText)}:\quad \operatorname{score}(x)=p(T\mid x).
$$

使用方式写成 `To use: keep examples x with score(x) >= threshold (stochastically)`，即设一个由 quality bar（质量门槛）决定的阈值，有时随机地（stochastically）、有时确定地保留。讲者展开说：生成式一路通常是便宜模型，不会去训一个大语言模型，典型就是 KenLM 的 5-gram（五元语法）模型，在目标数据上估计 $p_T(x)$；判别式一路则把 $T$ 当正例（positive examples）、从 $R$ 里随机采一个子集当负例（negative examples），做个 balance（类别平衡），再训练分类器。大家常用 fastText，因为它 fast，本质是一个线性 bag-of-words（词袋）分类器，输出 $p(T\mid x)$；有了模型，每来一篇新文档就打分，按阈值决定去留。

幻灯片里 `Model-based filtering?` 一节给出一份很重要的现状对照，讲者也用更长的话呼应了它：`Some deliberately do not use model-based filtering (C4, Gopher, RefinedWeb, FineWeb, Dolma)` 是几年前一批数据集的刻意选择，目的是不把偏置（bias）引入太多；而 `Some use model-based filtering (GPT-3, LLaMA, DCLM) [becoming the norm]` 被高亮，说明基于模型的过滤正在成为常态。讲者的判断是：除非 compute plentiful（算力极其充裕）、可以直接 train on everything（在全部数据上训练），否则多数团队其实是 compute poor（算力紧张），必须非常聪明地做过滤，不然就是把 FLOPs 白白浪费在 low-quality content 上。

这一章承上启下：它把转换阶段产出的 text 定义为 $R$，把「少量高质量样板」定义为 $T$，并给出可复用的两步评分框架，后面几章全是这个框架的具体实例化。易错点在于三条约束彼此拉扯：既要从 $T$ 泛化（$T'\neq T$），又要极快（$100$ 万亿 token 上跑），还要不引入太强偏置——这直接解释了为什么主力工具是 KenLM 与 fastText 这类便宜模型，而不是大模型；同时阈值不是一个可以拍脑袋写死的常数（后面第七章会专门回来处理这一点），把它当成固定超参是常见误解。

![def filtering() 的韦恩集合图：Raw data R、Target data T 与 More data T' 的关系](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0006_00-07-17.jpg)

![数据选择通用框架幻灯片：两类分类器 score(x)=p_T(x) 与 score(x)=p(T|x) 及 model-based filtering 现状](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0008_00-09-56.jpg)

## 过滤实例（一）：语言识别与 OpenMathText [11:27 - 14:12]

有了通用框架，第一个落地实例是 language identification（语言识别）。幻灯片给出的细节比口语更完整：`Goal: find text of a specific language (e.g., English)`；`fastText language identification [article]` 可直接 off-the-shelf（开箱即用）；`Supports 176 languages`（支持 176 种语言）；训练语料来自多语言站点 `Wikipedia, Tatoeba (translation site) and SETimes (Southeast European news)`，即维基百科、Tatoeba 翻译站与 SETimes 东南欧新闻；被高亮的一行给出具体阈值 `Dolma keeps pages with p(English) >= 0.5 [Soldaini+ 2024]`，即

$$
p(\text{English}) \ge 0.5.
$$

讲者补充了 Meta 训练这批 fastText 语言识别模型的背景，并强调语言识别相对而言是个 fairly easy problem（比较简单的任务）——看几个词就能分辨 Spanish 或 Japanese，因此它不是训练好语言模型的瓶颈（bottleneck）；但他也留了余地，指出 code switching（语码混用）和 dialects（方言）会带来细节问题，所以不能说这是「彻底解决了」的问题，阈值本身也 generally fairly heuristic（相当启发式）。这里与上一章的呼应很直接：Dolma 的 $p(\text{English})\ge 0.5$ 正是 `score(x) >= threshold` 的一个具体取值，而 176 种语言、0.5 这个数都是画面独有、转写没念到的参数。

第二个实例把「质量」的定义权交还给使用者。讲者说 quality filtering 应当 think about it as a tool（当作一件工具）：there's no universal notion of quality（不存在放之四海皆准的质量标准），你若把 quality 定义成 math，就能专门去找数学数据、并在数学上变强。OpenMathText 正是这样一篇论文（幻灯片标注 `OpenMathText [Paster+ 2023]`，口语说成 "open math text is this paper from 2023"），目标是 `curate large corpus of mathematical text from CommonCrawl`。它的 pipeline 不是训练单个分类器，而是把上一章的两类信号叠起来用，幻灯片逐条列出：`Use rules to filter (e.g., contains latex commands)`（先用规则检查是否包含 LaTeX 命令）；`KenLM trained on ProofPile, keep if perplexity < 15000`（在已知数学数据集 ProofPile 上训 KenLM，保留困惑度低于 15000 的文档）；`Trained fastText classifier to predict mathematical writing, threshold is 0.17 if math, 0.8 if no math`（再训 fastText 判断是否数学写作，并按是否含数学用两个不同阈值）。把阈值条件写清楚就是：

$$
\operatorname{perplexity}_{\text{KenLM}}(x) < 15000,\qquad \tau = \begin{cases}0.17,& x\ \text{含 LaTeX/math}\\ 0.8,& x\ \text{不含 math}\end{cases}.
$$

讲者解释了「两个阈值」的直觉：含 LaTeX 的样本门槛低（lower bar），不含 LaTeX 的样本必须更严格（higher bar）才肯收，否则会把大量普通网页误判成数学文本。结果是高亮行 `Result: produced 14.7B tokens, used to train 1.4B models that do better than models trained on 20x data`；转写里讲者口语约称 "15 billion tokens"，并说这些 token 用来训练模型，论文证明这种 targeted data collection（定向采集）训出的模型，在数学上优于用 20 倍（20 times）未经过滤数据训练的模型。14.7B / 15B 这两个略有出入的数字、1.4B 模型规模、20x 对比倍数、15000 的困惑度上限、0.17 与 0.8 的双阈值，都是从画面里才能拿到的精确参数。

这一章把抽象的 $T \to T'$ 框架第一次变成可执行的 recipe（配方）：语言识别里 $T$ 是「英语文本」，打分用现成 fastText；OpenMathText 里 $T$ 是 ProofPile 定义的「数学写作」，打分是规则 + 生成式 KenLM + 判别式 fastText 的组合。与后面的关系在于，它示范了「正例从哪来」可以很灵活，为第六章 GPT-3、LLaMA、phi-1 的更激进做法铺路。易错点也很具体：不要把 176 种语言的现成分类器当成万能（语码混用、方言、0.5 阈值的启发式都留了缝）；不要把单一 fastText 分数当成整条 pipeline（OpenMathText 的关键恰是多信号叠加）；更要留意「含 LaTeX 与否用同一阈值」这种简化会把双阈值设计抹平，以及「20 倍数据反而更差」提醒我们 token 数量不等于有效质量。

![语言识别与 OpenMathText 幻灯片：176 种语言、p(English)>=0.5、ProofPile 困惑度与 14.7B tokens 结果](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0011_00-13-54.jpg)

## 过滤实例（二）：质量过滤、随机保留与毒性过滤 [14:12 - 17:26]

本章继续把框架套到更多真实系统上，重点转向「正例怎么构造」。GPT-3（幻灯片标注 `GPT-3 [Brown+ 2020]`）的做法讲者一句话带过：正例是 Wikipedia、web text（从 high star Reddit posts 外链出去的页面）以及 some books，负例一般从 web 里采样，训练一个 linear classifier（线性分类器），保留打分足够高的文档。这几乎是第四章 `score(x) >= threshold` 的最朴素实例。LLaMA/RedPajama（幻灯片标注 `LLaMA/RedPajama [Touvron+ 2023]`）则列出了容易记混的一条：`Positives: samples from pages referenced by Wikipedia`（正例是「被 Wikipedia 引用的页面」），`Negatives: samples from CommonCrawl`，`Keep documents that are classified positive`。讲者在口语里特意纠正：正例是 pages referenced by Wikipedia，而不是 Wikipedia articles themselves（维基百科文章本身）——这个差别若忽略，会误以为整套方法在「抄维基百科」，其实它是在用维基百科的外链当作高质量代理信号。

关于「怎么按分数保留」，幻灯片给出一段值得原样抄下的 Python 代码，标题是 `Keep documents stochastically based on score`：

```python
def keep_document(score: float) -> bool:
    return np.random.pareto(9) > 1 - score
```

这把第四章括号里的 `stochastically`（随机地）落到了实处：不是硬阈值一刀切，而是借一个 Pareto 分布采样，让分数越高的文档被留下的概率越大，同时给低分文档留一点被采到的机会，从而兼顾质量与多样性。`np.random.pareto(9)` 里的 `9` 是形状参数，这是画面独有、口语完全没念到的实现细节。

phi-1（幻灯片标注 `phi-1 [Gunasekar+ 2023]`，口语里被含糊念成 "51 from Microsoft"）把「贵分类器造正例、便宜分类器外推」这条路走到了极致。它的哲学被写成 `Philosophy: really high quality data (textbooks) to train a small model (1.5B)`，并高亮 `Includes synthetic data from GPT 3.5 (later: GPT-4) and filtered data`（含 GPT-3.5、后来是 GPT-4 的合成数据以及过滤数据）。幻灯片里的伪代码把第四章的 $R$、$T$ 一一对应上：

```python
R = "Python subset of the Stack"  # Raw data
prompt = "determine its educational value for a student whose goal is to learn basic coding concepts"
T = "Use GPT-4 with this prompt to classify 100K subset of R to get positive examples"
# Train random forest classifier on T using output embedding from pretrained codegen model
# Select data from R that is classified positive by the classifier
```

也就是说，原始数据 $R$ 是 The Stack 的 Python 子集（全代码语料）；定义一个 prompt 让模型判断「对想学基础编程概念的学生是否有 educational value（教育价值）」；用 GPT-4 只标注 $R$ 的 `100K` 子集，把判为正例的结果当作目标数据 $T$——注意这里 $T$ 是一个昂贵分类器（expensive classifier）的输出，而非人工整理的高质量集；随后在 $T$ 上、以预训练 codegen 模型的 output embedding（输出嵌入）为特征训练一个 random forest（随机森林）分类器（讲者说其实用 fastText 大概也行），再用它从整个 $R$ 里挑出 classified positive 的数据。HumanEval 上的结果被逐条列出：`Train 1.3B LM on Python subset of The Stack (performance: 12.19% after 96K steps)` 对比 `Train 1.3B LM on new filtered subset (performance: 17.68% after 36K steps) - better!`，讲者口语概括为 performance goes up in fewer steps to a higher value（用更少步数升到更高值）。1.5B / 1.3B、100K、96K steps、36K steps、12.19% / 17.68% 都是画面独有的精确数字。

毒性过滤沿用同一框架。高亮小节 `Toxicity filtering in Dolma [Soldaini+ 2024]` 列出：`Dataset: Jigsaw Toxic Comments dataset (2018) [dataset]`；`Project goal: help people have better discussions online [article]`；`Data: comments on Wikipedia talk page annotated with {toxic, severe_toxic, obscene, threat, insult, identity_hate}`。讲者补充，这个项目初衷是帮助人们在线上更好地讨论，数据取自 Wikipedia talk pages（维基百科讨论页），争议话题下讨论可能 quite heated（相当激烈），评论被标注了是否含毒性内容，于是照样能定义正例负例、训练分类器。六个标签 toxic、severe_toxic、obscene、threat、insult、identity_hate 是画面独有的完整枚举。讲者随后收束：现在你已经能 identify a particular type of data（识别想要的某类数据）、build a classifier（为它建分类器）、再 go filter Common Crawl for that（去 Common Crawl 里把它筛出来）。

这一章与前后关系清楚：它把第四章框架里的「$T$ 从哪来」穷举了四种答案——人工定义的高质量集（Wikipedia/web text/books）、外链代理（pages referenced by Wikipedia）、昂贵 LLM 标注的子集（GPT-4 标 100K）、人工标注的毒性标签（Jigsaw）。易错点包括：把 LLaMA 的正例误记成维基百科文章本身；把 `np.random.pareto(9) > 1 - score` 的随机保留当成确定性阈值；忽略 phi-1 里「贵模型只标 100K 子集、再蒸馏到便宜分类器」这一成本折中而误以为直接用 GPT-4 扫全量；以及毒性六类标签若只取单一 toxic 二分，会丢掉 severe_toxic、identity_hate 等细粒度信号。

![LLaMA/RedPajama 与 phi-1 幻灯片：keep_document 随机保留代码、GPT-4 标注 100K 子集及 HumanEval 12.19%/17.68% 结果](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0013_00-16-33.jpg)

## 过滤阈值的尺度依赖与 157M 小规模实验 [17:26 - 21:11]

最后回到第四章埋下的伏笔：阈值到底该设多少。讲者明确指出，你「想要什么样的数据」其实 depends on what model you want to train（取决于要训什么模型），尤其取决于 the number of tokens you're training on（训练 token 数），因此 `No single optimal threshold for filtering`（不存在唯一最优阈值）——不能拍一个 `0.9` 就说它最好。直觉被幻灯片写成两条：`If training for longer, want more (lower quality) data`（训练越久，越能容忍更多、质量更低的数据）、`If training for shorter, want less (higher quality) data`（训练越短，越该只留更少、质量更高的数据）。讲者补了一句理想与现实的落差：若真能挥魔杖，当然希望「训练更久的同时还用更多高质量数据」，但 the data pool is what it is（数据池就那么大），这不是可选项。这把 $p(\text{English})\ge0.5$、$\tau=0.17/0.8$ 之类的具体数字重新定位成「在特定训练规模下才成立」的取值，而非普适常数。

为验证这个判断，讲者展示了一项由 Michael Ryan 做的 preliminary experiment（初步实验）：一个 `157M`（157 million parameters，1.57 亿参数）模型，数据池非常小，只有 `N=100 WARC`（100 个 WARC 文件，是 Common Crawl 的极小一角），然后不断训练、拉长时间看 loss。幻灯片给出图题 `Method comparison at d512 (157M), N=100 WARC vs tokens (all-epoch markers) -- expWARC_natural`，其中 d512、157M、N=100 WARC 都是画面独有的实验设定。折线图纵轴是 `eval/lima/loss`（约 3 到 5.5），横轴是 `Tokens trained (log)`（对数刻度，标有 5、100M、1B、10B 等），图例列出各条曲线及其一次 epoch 的数据量，这些 tokens 数全部来自图上：`dclm 1 epoch (tokens=97.6M)`、`high_quality 1 epoch (tokens=413.7M)`、`lim_curated 1 epoch (tokens=1.86B)`、`lim_curated_dclm_filtered 1 epoch (tokens=139.6M)`、`low_quality 1 epoch (tokens=1.67B)`、`med_quality 1 epoch (tokens=994.6M)`、`nemotron_full 1 epoch (tokens=316.6M)`、`nemotron_qhigh 1 epoch (tokens=93.5M)`、`resiliparse 1 epoch (tokens=4.42B)`；图中彩色竖虚线就是 all-epoch markers（各配置的 epoch 分界）。

讲者带读蓝色 DCLM 曲线：loss 从高处起步下降，每条竖线代表一次 epoch over the data；因为数据太少，最终不得不 repeat（重复）数据，第二个蓝点是 second epoch，loss 仍继续下降（第二次看同一批数据还能学到东西），但到某点开始 overfit（过拟合）——蓝线在末端明显上翘回约 4.9。相比之下，resiliparse 基本是 no filtering（几乎不过滤）：开头 much worse（差得多），但随着不断训练、进入多次 epoch 后也开始下降，只是 slower（更慢）。由此得到一条趋势：high quality data 在 not epoching（尚未重复数据）的区间更好；可一旦进入 lots and lots of tokens 的长训练区间，high quality data is no longer that great（高质量数据不再那么占优）。讲者还强调一个反直觉结论：用高质量数据你本来就不会想走到这个区间（因为会 overfit），大概会提前 early stop；但即便在那个提前停止点，结果也可能 worse than if you had trained for longer using low quality data（还不如用低质量数据训练更久）。

段末是一段现场问答（画面切到讲者在讲台前的照片）。学生问：图上每个点是不是对应一次独立的 training run（训练运行），做这类预训练实验要不要多次重复、算 confidence interval（置信区间）。讲者确认 each of these points is a single training run，并答：ideally 多次运行取置信区间是 good practice（好习惯），但论文里这类结果往往很 scarce（稀少），因为 each training run is fairly expensive（每次训练都很贵）；不过 in reality 他们做这些实验时，结果 generally tends to be stable（相当稳定），至少对 pre-training 是如此。这段问答正好点出本章最大的方法学局限：单条曲线、单次运行的对比，在没有误差棒的情况下不宜被当成精确结论。

这一章与全书的关系是给第四章的「阈值」加上一个尺度维度：阈值不是固定超参，而是训练 token 预算的函数；它与前面所有实例连起来，才构成完整判断——语言识别的 0.5、OpenMathText 的 0.17/0.8、phi-1 的 GPT-4 标注，都是特定算力与训练时长下的局部最优。易错点因而集中在三处：把某个阈值当成普适最优；看到 loss 在第二次 epoch 仍下降就误以为重复高质量数据无害（实则很快过拟合）；以及忽略「每个点只有一次训练、没有置信区间」这一实验精度边界，把曲线间的微小差距过度解读。

![方法对比折线图：d512 (157M) 模型在 N=100 WARC 上各配置 loss 随 Tokens trained (log) 变化，含每条曲线的 1 epoch tokens 标记](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0015_00-19-12.jpg)

## 过滤收尾：高质量数据的边际收益与去重登场 [21:11 - 22:55]

本段从一段课堂问答切入，讨论的是数据质量与训练时长之间的关系。学生问：如果我们能拿到更高质量的数据、并且在上面训练更久，是否还会出现边际收益递减（diminishing returns）？讲者的回答没有绕弯——任何一个数据集最终都会出现边际收益递减，因为数据规模本身是有限的（"every data set is going to have the diminishing returns eventually, that's finite"）；高质量数据带来的改变不是消除递减，而是把整条曲线往下压、并让它继续向下延伸（"it would just be down here and just keep on going down"）。换句话说，提升质量改变的是曲线的位置，而不是曲线的形状规律。这与前面过滤实验里「训练 token 越多越能容忍低质量数据」的结论是同一枚硬币的两面：当训练预算里的 token 数很大时，应该偏向更少但更干净的数据，否则反复 epoch 会把质量优势重新吃掉。

把这张图读成数学关系会更清楚。设横轴是已训练 token 数 $N$（图上用对数刻度标注为 `Tokens trained (log)`），纵轴是评估集上的损失 `eval/lima/loss`，那么边际收益递减可以写成

$$
\frac{\partial L}{\partial N} < 0,\qquad \left|\frac{\partial L}{\partial N}\right| \to 0 \quad (N \to \infty),
$$

而高质量数据的作用是让整条曲线整体下移，即 $L_{\text{high\_quality}}(N) < L_{\text{low\_quality}}(N)$，且这个差距在 $N$ 较小时最为显著。图上的标题原文是 `Method comparison at d512 (157M), N=100 WARCs vs tokens (all-epoch markers) -- expWARC_natural`，说明实验是在 $d=512$、157M 参数模型、$N=100$ 个 WARC 文件的小数据池上做的；图中每一条竖直虚线，是对应配置跑满一个 epoch 的标记（all-epoch markers）。

图例里的九条曲线及其各自的 1 epoch token 数，是画面上独有、转写完全没有念到的信息，值得逐条记下来：蓝色的 `dclm` 对应 `dclm 1 epoch (tokens=97.6M)`；绿色的 `high_quality` 对应 `tokens=413.7M`；`lim_curated` 对应 `tokens=1.86B`；`lim_curated_dclm_filtered` 对应 `tokens=139.6M`；红色的 `low_quality` 对应 `tokens=1.67B`；橙色的 `med_quality` 对应 `tokens=994.6M`；`nemotron_full` 对应 `tokens=316.6M`；`nemotron_hqhigh` 对应 `tokens=93.5M`；紫色的 `resiliparse` 对应 `tokens=4.42B`。纵轴刻度大约从 3 到 5.5，横轴跨 100M、1B、10B 几个量级。最值得注意的是蓝色 `dclm`：它早期 loss 最低，但越过自己的 1 epoch 标记之后曲线明显上翘，正是「小而高质量的源被反复训练导致过拟合」的可视化证据；相反，数据量最大的 `resiliparse`（4.42B token）下降平缓，却可以一直延伸到更大的 $N$ 而仍在降。

讲者在 [21:53] 用一句 "Okay, great. Let's move on" 结束问答，随即给出过滤部分的 Summary：过滤对构建好模型「pretty critical」，尤其是在算力受限的现实条件下（"when you're computed like most of us"）；严格地说，若拥有 infinite compute（无限算力），根本不需要过滤，直接把所有数据都拿去训练一个巨型模型即可，但现实中每个人都必须过滤。紧接着的 recipe 分两步：先弄清楚「好数据长什么样」（figure out what good data looks like），再训练一个分类器，把这个判断 extrapolate（外推）到其余的原始数据上。画面随后把 Summary 压成两条要点：`Filtering is critical for building a good model` 与 `Recipe: define target data (what good looks like), extrapolate to raw data`，并紧跟着一个黄色高亮的函数头 `def deduplication():`，宣告主题从 filtering 切换到 deduplication。

定义「好数据」本身有两条路径，讲者在 [22:28]–[22:50] 说得很具体：其一是发现某个现成的、自己很喜欢的数据集，只想要更多同类数据（"there's this data set out there that I really like and I just want more of it"）；其二是给语言模型写一段 prompt，先对一个大规模池子做初步过滤（preliminary filter of a large pool），再用筛出的高质量数据去训练一个更小的分类器，最后外推到全量数据。这里最容易踩的坑，是把「高质量」当成无条件的绝对优点：高质量源往往规模很小，在长训练预算下会被迫重复很多个 epoch，反而可能过拟合，把早期的 loss 优势还回去；另一个常见误解是以为换更好的数据就能消除 diminishing returns，实际上递减来自数据的有限性，只能把曲线整体下移，无法取消。此外，用小分类器做外推这一步隐含了「筛出的样本能代表目标分布」的假设，如果 prompt 或种子集本身有偏，外推结果会系统性地偏向某一类文本，这一点在画面上没有展开，但在读这张多配置对比图时应当心里有数。

![九种数据配置在157M模型上的loss曲线与各自的1-epoch标记](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0017_00-21-51.jpg)

## 两类重复：精确重复、近似重复与去重动机 [22:55 - 27:14]

讲者在 [22:55] 用 "Okay, next I'm going to talk about deduplication" 正式开启去重。此时的处境是：数据集已经过滤过，只保留了我们认定的高质量内容（"we only have what we deem to be high quality data"），但里面往往仍然有重复（duplicates）。画面上把这个新章节写成一个函数头 `def deduplication():`，并在其下给出 `Two types of duplicates:`：一类是 `Exact duplicates (mirror sites, GitHub forks) [Gutenberg mirrors]`，另一类是 `Near duplicates: same text differing by a few tokens`。精确重复的典型来源是镜像站——镜像的全部意义就是复制，而网络爬虫（web crawler）未必聪明到能识别这个镜像就是那个镜像，于是把多个站点各抓一遍，得到完全相同的内容；fork 一个 repo 也是重复，即便做了改动，通常也只改了少数文件，所以 "99% of that repo might be the same"。近似重复则不是镜像或衍生关系，只是两段文本恰好只差几个 token，可能来自复制粘贴（copying），也可能来自同一个公共来源（common source）。

画面继续给出 `Examples of near duplicates:` 两条，都是转写里点到、但没有这样归类的：`Terms of service and licenses [MIT license]` 与 `Formulaic writing (copy/pasted or generated from a template)`。讲者解释，MIT license 这类文本会出现在很多地方，页面里 license 那一段可能是逐字相同的精确拷贝（除非有人打错字），但包含它的页面其余部分可能完全不同；很多网站还共用同样的 headers 和 footers，这些也是重复。画面配的一张真实网页是 open source initiative 的 `Approved License` 页面：绿色 logo 写着 `open source initiative`，右上角有搜索图标与 MENU 菜单，中间是橙色 `Get involved` 按钮和四个社交图标，底部导航为 `About`、`Licenses`、`Open Source AI`、`Community`——正是「每个页面都长着同一套模板外壳」的直观例子。另一张表格以 `Dataset | Example | Near Duplicate Example` 为表头，列出 Wiki-40B 中带 `START_ARTICLE`、`START_PARAGRAPH` 标记的文本与其近重复版本，说明这类重复连结构化语料里都有。

真正的重头戏是画面里被黄色高亮标注的那一行：`Product description repeated 61,036 times in C4`。讲者说这是一次对 C4 数据集的 audit（审计）发现的，某段产品描述在整个数据集里重复出现了 61,036 次，并感叹 "the web is weird"、强调 "this is why it's good to look at your data"。转写里讲者口头把它描述成某种防毒面具（gas mask）的描述，而画面高亮标注下方展示的示例正文其实是一段婚礼设计文案，开头是 "by combining fantastic ideas, interesting arrangements, and follow the current trends in the field of that make you more inspired and give artistic touches"，后面提到 "apply some or all of these design in your wedding"，并附了一个 `[example page]` 链接；两处描述不完全一致，这里按各自原文如实转述，不做猜测。同一张画面还给出 LM1B（the one billion word benchmark）与 C4 的近重复对照：LM1B 里同一句话 "I left for California in 1979 and tracked Cleveland's changes on trips back to visit my sisters." 有带逗号与不带逗号、词间多空格两个版本，被归为 `Minor formatting differences in copy/pasting`；C4 里那段机票广告文案则把出发国 `'Canada'` 换成 `"USA."`、把 `From May 2019` 换成 `From April 2019`、把 `roughly 6 a week` 换成 `roughly 7 a week`、把目的地 `"Switzerland"` 换成 `"Croatia"`，其余的 "Affordable and convenient holiday flights take off from..."、"Book your Mai Khao (GOI), Dhawad (DBV) flight now" 完全保留。讲者提醒，如果在这种数据上训练，等于只是在不同实体变体上反复训练同一段模板，纯粹在浪费 GPU（"this is going to be wasting your GPUs"）。

为什么必须去重，画面给出带引用的标题 `Deduplication training data makes language models better [Lee+ 2021]` 与两条理由：`Train more efficiently (because have fewer tokens)` 和 `Avoid memorization (can mitigate copyright, privacy concerns)`。讲者在 [26:17]–[27:11] 展开：第一，去重在不真正丢失信息的前提下缩小数据集，训练更高效；第二，避免记忆（memorization）——如果某段受版权保护的内容被大量复制，模型训练后就会把它背下来，同时带来隐私（privacy）顾虑；第三，还有 decontamination（去污染），讲者说这 arguably even more important（可以说甚至更重要），本质同样是确保 test set 不出现在 training set 里。把「去重」写成定义式，精确重复是 $x_i = x_j$，近似重复则是 $x_i \approx x_j$（两者只差几个 token，即 differ by a few tokens）；而去除动作本身可以写成对一个等价类只保留一个代表元：

```python
def deduplication():
    # keep one representative per group of (near-)duplicates
```

这一章最需要警惕的易错点有三个。第一，别把「近似重复」窄化成排版差异：模板化生成、整站共用页眉页脚、许可证文本，都是只有在真正打开数据去看时才会暴露的重复形态，61,036 次这个数字正是「不看数据就发现不了」的证明。第二，去重与 decontamination 目标不同但手段相通：前者优化训练效率、避免记忆，后者保证评测干净，不能因为做了前者就默认后者也完成了。第三，讲者坦承 "I don't know why"（比如那段逗号差异、那段产品描述的来源），面对这种来源不明的异常，正确做法是先审计、再决定删除策略，而不是替它编一个成因。

![C4数据集中重复61036次的产品描述及LM1B与C4的近重复文本对照](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0020_00-25-49.jpg)

## 去重的设计空间与线性时间挑战 [27:14 - 28:29]

有了动机，接下来要先把「去重」这件事拆成可配置的维度。讲者在 [27:14] 提出 "here's the design space to think about"，画面把它整理成 `Design space:` 下的三个问题，全部是原文：`1. What is an item (sentence, paragraph, document)?`（去重的基本单位取句子、段落还是文档）；`2. How to match (exact match, existence of common subitem, fraction of common subitems)?`（判定匹配是精确相等、是否存在公共子项，还是公共子项所占的比例，后者对应 near deduplication）；`3. What action to take (remove all, remove all but one)?`（找到两个页面的重复后，是删掉全部实例，还是只保留一个）。这三点决定了后面所有算法的选择：粒度定得越细，越容易发现局部重复，但也越容易把文档切碎；匹配判据从 exact 放宽到 fraction，就正式从「精确去重」跨进「近似去重」。

紧接着画面用黄色高亮给出 `Key challenge:` 两条：`Deduplication is fundamentally about comparing items to other items`（去重本质上是在 item 与 item 之间做比较），以及 `Need linear time algorithms to scale`（要能扩展到 web scale 就必须线性时间算法）。讲者在 [27:53]–[28:27] 把对比讲得很透：过滤（filtering）针对的是单个 item——「这个 item 好不好」，可以逐条判断，天然可并行（paralyzed / parallelized），整体是线性时间；即便如此，为了快，还要用 rule-based 或非常小的模型。去重不一样，它要求比较 item 之间的关系，朴素做法是把所有东西两两比一遍，也就是 $O(n^2)$，讲者明确说 "you can't do the n square thing where you compare everything to everything"，在网页规模下必须换成 $O(n)$ 的算法。用符号区分这两类计算：

$$
\text{filtering}: \; f(x_i) \in \{0,1\},\quad \text{deduplication}: \; g(x_i, x_j) \in \{0,1\}, \qquad |\{(i,j)\}| = O(n^2) \to O(n).
$$

画面在设计空间下方还淡色排出了四个即将实现的函数，构成本段余下内容的路线图，这也是转写里没有逐字念到的：`hash_functions()`、`exact_dedupication()`、`jaccard_minhash()`、`locality_sensitive_hashing()`（画面上 `exact_dedupication` 就是这个拼写）。它们的顺序不是随意的：先用哈希把 item 压成可比较的短签名，再解决精确去重，再用 Jaccard 加 MinHash 定义并估计相似度，最后用 LSH 把随机碰撞锐化成可用的阈值判定。换句话说，后面所有看似玄妙的随机算法，都是在回答这一章提出的同一个约束——如何在不做 $n^2$ 次比较的前提下，找出相似的 item 对。

这里的易错点相当集中。最常见的是把过滤的并行套路直接搬到去重上：既然每条数据可以独立打分，就以为两两相似性也能独立算，结果在亿级文档上直接被 $O(n^2)$ 拖垮。其次是忽略粒度选择的连锁影响——item 取句子、段落还是文档，会同时改变匹配判据的含义和「remove all but one」到底删掉多少内容（后面 C4 的三句跨度就是活生生的反例）。第三，`fraction of common subitems` 里的「比例」必须先定义阈值才算完整，而阈值定得松紧直接决定误删与漏删的权衡；这一章只把问题摆出来，真正的阈值要到 Jaccard 与 LSH 才被定量地给出。还要注意，画面把「比较 item 之间的关系」列为根本挑战，意味着任何基于单条内容的启发式（比如只看长度、只看语言）都无法替代成对比较，只能作为预筛。

![去重的三个设计维度与需要线性时间算法的核心挑战](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0022_00-28-28.jpg)

## 哈希函数与精确去重：从 MurmurHash 到 C4 三句跨度 [28:29 - 31:21]

要绕开 $n^2$，讲者指出去重文献（deduplication literature）通用的一招是用 hash functions（哈希函数），并感谢 algorithms community 提供的这些 "quite nice ideas"。哈希函数的定义很朴素：把一个值（比如字符串）映射成一个短得多的值（字符串或整数），即 $h: \mathcal{S} \to \mathbb{Z}$，且 $|h(x)| \ll |x|$；hash value 远小于原始 item。hash collision（哈希碰撞）指两个不同的 item 映射到同一个值，即 $x_i \ne x_j$ 但 $h(x_i) = h(x_j)$。画面在 [28:29] 附近还给出一组转写没细念的分类：`tradeoff between efficiency and collision resistance [article]`——`Cryptographic hash functions (SHA-256): collision resistant, slow (used in bitcoin)` 属于密码学级别、抗碰撞但慢，用于 cryptography 和 bitcoin；`DJB2, MurmurHash, CityHash: not collision resistant, fast (used for hash tables)` 则是快的、不抗碰撞的，用于哈希表，那里碰撞并非世界末日。本讲选用后者，画面明确写着 `We will use MurmurHash:` 并给出示例 `h = mmh3.hash("hello")`。这个选择本身就是一个设计决策：我们不需要密码学强度，只需要快且分布够均匀。

精确去重（exact deduplication）在概念上极简单：对字符串做 exact match，然后 remove all but one（删到只剩一个）。画面把这一节的示例整理成 `def exact_deduplication():` 下的 `Simple example` 三条：`1. Item: string`、`2. How to match: exact match`、`3. Action: remove all but one`（第三条被黄色高亮）。原始输入是 `items = ["Hello!", "hello", "hello there", "hello", "hi", "bye"]`，实现写成一种 MapReduce 风格、便于并行和扩展的代码：

```python
# Compute hash -> list of items with that hash
hash_items = itertools.groupby(sorted(items, key=mmh3.hash), key=mmh3.hash)

# Keep one item from each group
deduped_items = [next(group) for h, group in hash_items]
```

先按 `mmh3.hash` 排序、再用 `itertools.groupby` 按哈希分组，每组用 `next(group)` 取第一个元素。画面上的对照表格把结果列得很清楚：`items` 一行是 `"Hello!"`、`"hello"`、`"hello there"`、`"hello"`、`"hi"`、`"bye"`，`deduped_items` 一行则是 `"hi"`、`"bye"`、`"hello"`、`"hello there"`、`"Hello!"`——重复的那个 `"hello"` 被去掉了，而 `"Hello!"`（首字母大写加感叹号）与 `"hello"` 因为是不同字符串而都被保留，说明这一步严格区分大小写和标点。画面还给出三条评价：`Pro: simple, clear semantics, high precision`（语义清晰、精度高）、`Con: does not deduplicate near duplicates`（致命缺点：处理不了近似重复）、`This code is written in a MapReduce way, can easily parallelize and scale`（容易并行、能扩展）。这三条把精确去重的定位钉死了：它是 $O(n)$ 的、可靠的，但只是近似去重的铺垫。

随后讲者把上面那套设计空间套到真实系统上：`C4 [Raffel+ 2019]`（即 T5 那篇论文处理 Common Crawl 的做法）——`1. Item: 3-sentence spans`、`2. How to match: use exact match`、`3. Action: remove all but one`。也就是以「三句跨度」为 item 做精确匹配，命中就只留一份。讲者在 [30:53] 特意提醒听众注意这里 "something kind of a bit strange"：既然 item 是三句跨度，一旦发现两份文档含有相同的三句跨度并删到只剩一份，就相当于从某篇文档中间硬生生挖掉三句话，破坏文档连贯性。画面把这一点写成黄色高亮的警告原文：`Warning: when a 3-sentence span is removed from the middle of a document, the resulting document might not be coherent`。讲者的评价是 "a little bit strange... but you know, this is what they did"，即承认这是个副作用但仍照做。画面底部随即淡色引出下一节的函数头 `def jaccard_minhash():` 与两句话 `Let's now look at approximate set membership.`、`First we need a similarity measure.`，说明精确去重到此为止，接下来必须换一套能度量「近似」的工具。

本章的易错点值得单独记。第一，精确去重的 high precision 是以牺牲召回为代价的：`"Hello!"` 与 `"hello"` 只差大小写和标点，人眼显然是近重复，哈希却认为无关，这正是 `Con` 那一条的实际含义。第二，`sorted(items, key=mmh3.hash)` 这一步不能省——`itertools.groupby` 只对相邻元素分组，不先按哈希排序就会把相同哈希的 item 散落在不同组里，导致去重失效。第三，C4 的三句跨度说明「粒度」不是纯技术细节：把 item 切得太碎，删除动作会切进文档内部，产生不连贯的残片；这也是为什么后面要转向基于集合相似度的 whole-document 式判定。第四，选用 MurmurHash 这类非密码学哈希时，要接受碰撞可能带来极少量误合并，只是对去重这种场景而言代价可忽略，不必换成 SHA-256 拖慢整个流水线。

![精确去重的MapReduce风格代码、去重前后对照表与C4三句跨度的连贯性警告](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0024_00-31-07.jpg)

## Jaccard 相似度：给「近似重复」一个可计算的定义 [31:21 - 33:44]

精确去重解决不了 `Con: does not deduplicate near duplicates`，于是讲者在 [31:21] 转向「how do we do near deduplication」，并指出第一步必须先定义什么叫 approximate match（近似匹配），也就是画面底部那句 `First we need a similarity measure.`。为此引入的是 Jaccard similarity（Jaccard 相似度），一个集合论里相当标准的度量：两个集合的 Jaccard 就是交集大小除以并集大小。写成公式：

$$
\operatorname{Jaccard}(A,B) = \frac{|A \cap B|}{|A \cup B|}.
$$

画面给出的定义原文是 `Definition: Jaccard(A, B) = |A intersect B| / |A union B|`，配套的例子是 `A = {"1", "2", "3", "4"}`、`B = {"1", "2", "3", "5"}`。交集是 $\{1,2,3\}$，大小为 3；并集是 $\{1,2,3,4,5\}$，大小为 5；所以 $\operatorname{Jaccard}(A,B) = 3/5 = 0.6$，画面右侧蓝色结果框里写的正是 `jaccard = 0.6`。注意这里集合元素是字符串 `"1"` 到 `"5"`，而不是整数，这是画面独有、转写没念到的细节。

实现非常直接，画面给出的完整代码是：

```python
def compute_jaccard(A, B):
    intersection = len(A & B)
    union = len(A | B)
    return intersection / union

jaccard = compute_jaccard(A, B)
```

即用 Python 的集合运算符 `&` 取交集、`|` 取并集，各自取 `len` 后相除。讲者在 [32:08] 强调 Jaccard 是 0 到 1 之间的数：0 表示两个集合完全不相交（disjoint），1 表示两个集合完全相同（identical）。有了这个度量，就可以把「近似重复」写成一句可判定的定义，画面原文为 `Definition: two documents are near duplicates if their Jaccard similarity >= threshold`；讲者随口给的阈值例子是 0.99，也就是说只有几乎逐字重合的两份文档才算近重复。这一句把前面设计空间里含糊的 `fraction of common subitems` 落成了明确的数值判据。

但定义完立刻撞回老问题，画面用黄色高亮写着 `Algorithmic challenge: find near duplicates in linear time`——我们仍然要在线性时间里把满足阈值的文档对找出来，不能两两算 Jaccard。讲者在 [32:28] 说这幸好是一个已解决的算法问题（"a solved algorithms question"），答案是 MinHash。画面在页面下半部已经提前淡色写出了 MinHash 的一句话定义和它的反直觉之处：`MinHash: a random hash function h so that Pr[h(A) = h(B)] = Jaccard(A, B)`；`Normally, you want different items to hash to different hashes`（通常设计哈希函数，是希望不同 item 映射到不同值、尽量避免碰撞）；`...but here, you want collision probability to depend on similarity`（但这里恰恰相反，我们希望碰撞概率依赖于相似度）。讲者在 [33:18]–[33:40] 把这一点讲得更透：一般要避免 collisions，而这里不但要碰撞，还要「控制」碰撞——让相似的东西比不相似的东西更容易碰撞（"similar things you want to collide more than the disparate things"）。这就在 hashing 的线性时间优势与 Jaccard 这个目标度量之间架起了桥：哈希让比较变便宜，Jaccard 是我们真正想算的量，MinHash 让两者的期望对齐。

本章的易错点首先在于「Jaccard 是集合度量」这一点：它要求先把文档表示成集合，而本段只用了抽象的 $\{1,2,3,4\}$ 这类玩具集合做演示，并没有在本段里展开如何从文本切出集合元素，因此不应凭空补出具体的分词或切片方案。其次，阈值 0.99 定得很高，含义是「几乎完全一致」；把它调低会把大量只是主题相近的文档误判为重复，调高则会漏掉只差几个 token 的版本。第三，别把 0 和 1 的含义记反：0 是 disjoint（毫无共同元素），1 是 identical（完全相同）。第四，`len(A & B) / len(A | B)` 在 $A$、$B$ 都为空时会出现除零，画面与转写都未讨论这一边界，属于读代码时应自己留意的细节。最后要看到承上启下的位置：Jaccard 只是「尺子」，它本身不解决线性时间问题，真正的算法加速全押在下一章的 MinHash 上。

![Jaccard相似度的定义公式、compute_jaccard实现代码与近重复阈值判定](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0025_00-32-32.jpg)

## MinHash：让碰撞概率等于 Jaccard [33:44 - 37:29]

MinHash 的定义在画面上写成一条概率等式：找一个随机哈希函数 $h$，使得

$$
\Pr_{h}\big[\min h(A) = \min h(B)\big] = \operatorname{Jaccard}(A,B) = \frac{|A\cap B|}{|A\cup B|}.
$$

也就是「两个集合的最小哈希值相等的概率，恰好等于它们的 Jaccard 相似度」。具体实现极简，画面在 [34:03] 给出的函数原文是：

```python
def minhash(S: set[str], seed: int):
    return min(mmh3.hash(x, seed) for x in S)
```

对集合 $S$ 里每个元素用带 `seed` 的 `mmh3.hash` 散列，取所有哈希值里的最小者作为整个集合的签名。讲者在 [34:08] 特意安抚第一次见到这个构造的人：取 min 看起来奇怪，其实取 max 也行（"you can take a max too doesn't really matter"），它只是一种打破平局（break ties）的方式。真正的关键是 `seed`：不同 seed 对应不同的随机哈希函数，从而得到不同的随机排列。

为什么这条等式成立，画面用 characteristic matrix representation（特征矩阵表示）给出了完整证明，这部分是转写讲得比较含糊、而图上列得最清楚的内容。把 $A=\{1,2,3,4\}$、$B=\{1,2,3,5\}$ 写成矩阵（1 表示元素属于该集合）：

| item | A | B |
|---|---|---|
| 1 | 1 | 1 |
| 2 | 1 | 1 |
| 3 | 1 | 1 |
| 4 | 1 | 0 |
| 5 | 0 | 1 |

画面原文接着写 `Random hash function induces a permutation over items`（随机哈希函数在 item 上诱导出一个随机排列）、`Look at which item is first in A and which item is first in B.`、`Each item has the same probability as being first (min)`，并给出两条 bullet：`If 1, 2, 3 is first, then first in A = first in B.` 与黄色高亮的 `If 4, 5 is first, then first in A ≠ first in B.`。讲者在 [34:47] 举了一个具体排列 "4 3 1 5 2 or something else"：随机排列决定哪一行排在最前，min 就是告诉我们在该排列下 A 与 B 各自第一个出现的元素。若排在最前的是交集元素 1、2、3 中的某一个，两个集合的「第一个」相同，于是 $\min h(A) = \min h(B)$；若排在最前的是只属于 A 的 4 或只属于 B 的 5，两者就不同。因为每一行被排到最前的概率相同，所以「碰撞」的概率正好等于交集行数占并集行数的比例，也就是 $3/5 = 0.6$。讲者在 [36:08] 把它总结为：在随机哈希函数的期望（in expectation over your random choice of hash functions）下，碰撞概率就等于我们想要的相似度。

光有理论不够，讲者在 [36:30] 说 "let's just check this in code"，画面 [37:05] 给出的验证代码是本段最完整的独有信息：

```python
# Verify MinHash approximates Jaccard as advertised
n = 100  # Generate this many random hash functions
matches = [minhash(A, seed) == minhash(B, seed) for seed in range(n)]
estimated_jaccard = len([m for m in matches if m]) / len(matches)
assert abs(estimated_jaccard - jaccard) < 0.01
```

即生成 $n = 100$ 个由不同 `seed` 给出的随机哈希函数，逐个检查 `minhash(A, seed) == minhash(B, seed)`，用匹配比例 `estimated_jaccard` 去估计真实 `jaccard`，最后断言二者之差小于 0.01。画面蓝色结果框同时显示 `jaccard = 0.6` 与 `estimated_jaccard = 0.6`，与理论值吻合；讲者在 [36:52] 也口头确认「you get 0.6」。关键在于，MinHash 的价值不在于估得准，而在于 [37:04] 讲者强调的那一点：不必做 $n^2$ 次比较，可以分别算出 $\min h(A)$、$\min h(B)$ 以及任意多个其它集合的 minhash，然后只在这些短签名里找碰撞（"I just look for collisions"），这正是第三章要求的线性时间算法。讲者在 [37:16] 还停顿问了一句 "any questions so far about Minhash"，可见这是全段的一个理解门槛。

本章的易错点不少。第一，碰撞在这里是刻意设计、必须被「控制」的，不是哈希函数的缺陷；如果沿用第四章「避免碰撞」的直觉，会完全误解 MinHash。第二，min 与 max 只是打破平局的约定，但所有集合必须用同一个排列（同一个 `seed`）比较，混用不同 seed 的签名会让等式失效。第三，单个 MinHash 只是一次伯努利试验，方差很大，`estimated_jaccard` 要靠 $n=100$ 这样的多次独立采样才稳定；`< 0.01` 的断言是经验验证，不是理论保证。第四，$\Pr[\min h(A)=\min h(B)]=\operatorname{Jaccard}(A,B)$ 是「在随机哈希函数上的概率」，对固定一个 seed 的签名而言只有相等或不等两种结果，不能把它当成某对文档的确定相似度。第五，讲者在 [37:29] 马上补了一句 "we're not done yet"，预告 MinHash 只解决了「碰撞概率 = 相似度」，还没解决「相似度是否超过阈值」，这正是下一章 LSH 要补的缺口。

![用100个随机seed验证MinHash估计Jaccard的代码与jaccard=0.6的结果](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0028_00-37-05.jpg)

## 局部敏感哈希：用分带把随机碰撞锐化成阈值 [37:29 - 40:49]

讲者在 [37:29] 开门见山地指出 MinHash 还不够："now we can hash our items, but we're not done yet, because a collision doesn't tell us that Jaccard is above some threshold"。我们真正想要的是找到满足 $\operatorname{Jaccard}(A,B) > 0.99$ 的文档对，而目前只得到「两物碰撞的概率等于 Jaccard」这一随机陈述；单次碰撞非常 stochastic（随机）、方差很大（"the variance is, you know, quite large here"），不足以可靠判定。画面把这一困境和目标写成几行原文：`P[A and B collide] = Jaccard(A, B)`、`On average, more similar items will collide, but very stochastic...`，以及黄色高亮的 `Goal: have A and B collide if Jaccard(A, B) > threshold`，随后是 `We have to somehow sharpen the probabilities...`（必须设法把概率「锐化」，不能让碰撞概率字面等于 Jaccard）。讲者介绍，解决方案是理论计算机科学里非常经典的 locality sensitive hashing（LSH，局部敏感哈希），并注明 `[book chapter]`；核心思路是 `Solution: use n hash functions`，而且这些哈希函数要相互独立。

具体构造是把 $n$ 个哈希函数切成 $b$ 个 band（带），每个 band 含 $r$ 个哈希函数，画面原文为 `Break up into b bands of r hash functions each (n = b * r)`，即

$$
n = b \cdot r.
$$

[40:06] 的画面给出一组具体参数，全部是转写里念过、但以代码形式列出的原文：

```python
n = 12   # Number of hash functions
b = 3    # Number of bands
r = 4    # Number of hash functions per band
```

对应的 `Hash functions:` 排布是 `h1 h2 h3 h4 | h5 h6 h7 h8 | h9 h10 h11 h12`，即三个 band 分别装 H1–H4、H5–H8、H9–H12。每个哈希函数对 $A$、$B$ 各自给出「碰撞 / 不碰撞」两种结果，等价于判断 $h_i(a) = h_i(b)$。判定规则由画面黄色高亮的那句给出：`Key: A and B collide if for some band, all its hash functions return same value`——只要存在某一个 band，其内部所有 $r$ 个哈希函数都返回相同值，就判定 $A$、$B$ 碰撞。讲者在 [40:06]–[40:43] 用 H5–H8 举例：要 $h_5(a)=h_5(b)$ 且 $h_6(a)=h_6(b)$、$h_7(a)=h_7(b)$、$h_8(a)=h_8(b)$，这个 band 才算被触发；三个 band 之间是「或」的关系，任一被触发即可，并不要求 12 个哈希函数全部相等，也不要求所有 band 都相等。

这里的结构值得单独点明：band 内部是 AND（$r$ 个函数必须全相等），band 之间是 OR（$b$ 个 band 任一成立即可）。画面把这称为 `the and-or structure of the bands sharpens the threshold`。直觉上，AND 让「高相似度」的必要条件变苛刻——只要相似度为 $s$，某个固定 band 全匹配的概率是 $s^r$，随 $r$ 指数衰减；OR 又用 $b$ 个独立机会把概率抬回来。两者叠加，就把原本线性的、方差很大的 $P[\text{collide}] = s$ 变成一条在低相似度处贴近 0、在高相似度处迅速跳到接近 1 的陡峭曲线，从而实现「相似度超过阈值才碰撞」的目标。讲者在 [40:49] 正要说 "there's sort of this and or structure here that is doing the lifting"，本段后面一章就用代码把这条曲线算出来。

本章的易错点集中在「读错触发条件」。第一，最容易把「某个 band 内所有哈希函数相等」误记成「所有哈希函数都相等」或「所有 band 都相等」；前者会把条件推到极端苛刻，后者则忽略了 band 间的 OR，两者都会让概率公式完全变样。第二，$n = b \cdot r$ 里 $b$ 与 $r$ 的作用方向相反：$r$ 变大让固定 band 更难匹配（AND 更严），$b$ 变多则给更多次机会（OR 更宽），调参时不能混为一谈。第三，`n = 12, b = 3, r = 4` 只是讲者用来讲清分带结构的示意参数，与后面概率计算示例采用的 `b = 5, r = 10` 是两组不同的数，别把它们当成同一套配置。第四，LSH 依赖 $n$ 个哈希函数相互独立；若 seed 选取相关，`s^r` 与后面的概率公式都不再成立。第五，要记得 LSH 是在 MinHash 之上的「锐化层」：它不改变 MinHash 签名本身，只改变何时把一对候选判定为碰撞，因此仍属于线性时间的候选生成，而不是精确的两两验证。

![将12个哈希函数分为3个带每带4个的分带方案与band内全相等才碰撞的判定规则](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0030_00-40-06.jpg)

## 碰撞概率的计算：从 s^r 到 S 型曲线 [40:49 - 42:22]

最后一步是把分带的 and-or 结构写成可计算的概率。讲者在 [41:02] 设问：给定某个 $\operatorname{Jaccard}(A,B)$，按上面的定义，$A$ 与 $B$ 碰撞的概率是多少？画面 [41:37] 给出的示例参数是 $s = 0.8$（蓝色框里写作 `sim = 0.8`）、`b = 5`（5 个 band）、`r = 10`（每带 10 个哈希函数），蓝色框里还预先算出了 `prob_match = 0.1074`。推导分两层：先看某一个固定 band，它要求 $r$ 个独立哈希函数全部相等，每个相等的概率都是 $s$，所以

$$
P_{\text{match}}(s) = s^{r}, \qquad P_{\text{match}}(0.8) = 0.8^{10} = 0.1073741824 \approx 0.1074,
$$

这与蓝色框的 `prob_match = 0.1074` 完全对上，也印证了讲者 [41:59] 那句「固定 band 匹配的概率一般很低，是 exponential in r」。再用 band 之间的 OR：碰撞等于「至少一个 band 匹配」，其补事件是「$b$ 个 band 全都不匹配」，单带不匹配概率为 $1 - s^r$，$b$ 个都不匹配为 $(1-s^r)^b$，因此

$$
P_{\text{collision}}(s) = 1 - \left(1 - s^{r}\right)^{b}.
$$

画面把这段计算写成了完整函数，逐行原文如下（黄色高亮落在 `prob_collision` 那一行）：

```python
def get_prob_collision(sim, b, r):
    prob_match = sim ** r          # Probability that a fixed band matches
    prob_collision = 1 - (1 - prob_match) ** b   # Probability that some band matches
    return prob_collision
```

并在 `Example` 下给出调用 `prob_collision = get_prob_collision(sim=0.8, b=5, r=10)`。画面左下角还画了一条粉红色的 S 型曲线，纵轴标了 1.0 与 0.5，横轴是相似度：曲线在低相似度处贴近 0，越过某个相变区后快速抬升并趋近 1，正是「锐化阈值」的可视化结果。讲者在 [42:04]–[42:22] 收尾时说，碰撞概率就是「某个 band 匹配」的概率，也就是 1 减去「所有 band 都不匹配」的概率，把 $b$ 个 band 用 `raise it to the B` 连乘起来；转写在这里结束。画面左侧同时保留了上一章的 `n = 12 / b = 3 / r = 4` 与 `h1 ... h12` 分带示意，说明这两组参数（结构示意用 12/3/4，概率示例用 `sim=0.8, b=5, r=10`）在同一张幻灯片上并置，功能不同。

把公式里的两个参数方向再压实一遍：$P_{\text{match}} = s^r$ 里的 $r$ 出现在指数上，所以 $r$ 稍增一点，固定 band 匹配概率就急剧下降；而 $b$ 出现在外层的 $1-(1-\cdot)^b$ 里，$b$ 越大，相当于给了越多「任一带命中」的机会，整体曲线被抬高、向左移。二者的乘积 $n = b\cdot r$ 决定要算多少个 MinHash 签名，也就是计算成本。讲者想要的是「$\operatorname{Jaccard} > 0.99$ 才碰撞」，而 $P_{\text{collision}}(s)$ 这条 S 型曲线在阈值附近越陡，误判（低相似度却碰撞）与漏判（高相似度却不碰撞）就越少；调 $b$、$r$ 就是在调这条曲线的陡峭程度与位置。

本章的易错点很具体。第一，别把 $s^r$ 当成最终碰撞概率：它只是「某一个固定 band 全匹配」的 AND 概率，必须再套一层 $1-(1-s^r)^b$ 才是 OR 之后的碰撞概率。第二，`sim ** r` 与 `** b` 两层幂不能合并或写反，代码里的注释 `# Probability that a fixed band matches` 与 `# Probability that some band matches` 正好标明了两层的含义。第三，`0.1074` 是 $0.8^{10}$ 的四位近似，不是独立测出来的常数；把 $s$、$r$ 换掉，这个值必须重算。第四，这条公式仍然建立在「$n$ 个哈希函数相互独立」的假设上，是期望意义下的概率，单次运行依旧有随机性，所以 LSH 给出的是候选对，之后仍需验证。第五，本段只推到 $1-(1-s^r)^b$ 这个表达式与一次 `get_prob_collision(sim=0.8, b=5, r=10)` 调用，转写在 [42:22] 处截断，画面上的 S 曲线也只是示意；因此不应在本段里擅自补出具体的相变阈值取值或调参结论。至此，从「碰撞概率等于 Jaccard」的 MinHash，到「用分带锐化成阈值」的 LSH，本段把第三章提出的线性时间近似去重问题完整地落成了一套可推导、可编码的方案。

![get_prob_collision函数、sim=0.8与b=5与r=10的参数代入及碰撞概率的S型曲线](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0031_00-41-37.jpg)

## 碰撞概率的 S 型曲线：把随机碰撞锐化成阈值 [42:22 - 46:37]

本段开头承接上一段末尾对 `get_prob_collision(sim=0.8, b=5, r=10)` 的调用。讲者在 [42:27] 用一句 "one minus that is the probability that some match" 点明：碰撞概率等于「一减去没有任何 band 匹配的概率」。写成式子就是 $P_{\text{collision}}(s)=1-(1-s^r)^b$，其中 $s$ 是两文档的相似度（similarity，具体由 Jaccard 给出），$b$ 是 band（分带）的数量，$r$ 是每个 band 内的哈希函数个数。单个 band 里 $r$ 个哈希值全部相等的概率是 $s^r$，$b$ 个 band 全都不匹配的概率是 $(1-s^r)^b$，取补集便得到「至少有一个 band 匹配」的碰撞概率。这一步的意义在于：它把 MinHash 原本「碰撞概率恰好等于 Jaccard」的线性关系，改造成一个接近阶跃的判定器；讲者在 [42:32] 还顺口说明碰撞概率会偏高，因为 "you're given [more] tries to get a match"（给了多次尝试去命中）。

[42:47]–[43:45] 讲者把这条曲线画出来：横轴是 similarity ($s$)，纵轴是碰撞概率 $P$。画面（`kf_0032`）上横轴刻度标着 0.25、0.5、0.75、1.0，右侧纵轴标着 0.0、0.5、1.0，左侧还用蓝色散点标了 `candidates?` 的 0/1 两行。他强调两个端点必须成立：相似度为 0 时概率应为 0，相似度为 1 时概率应为 1，中间则呈现 "interesting S-shaped"（有趣的 S 形）。这正是前面 wanted to sharpen（想要锐化）的目标：相似度低于某个阈值时希望匹配概率尽可能低，高于阈值时希望尽可能高，让它像一次 phase transition（相变）。画面顶部的 `Example` 行写着 `prob_collision = get_prob_collision(sim=0.8, b=5, r=10)`，说明这一屏是在具体参数下求碰撞概率；转写里也明确提到 "we were looking at 0.8" 这个相似度取值。

![碰撞概率随相似度变化的 S 型曲线，横轴 similarity (s)，右侧纵轴 probability (P)](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0032_00-43-02.jpg)

[44:01]–[44:29] 讲者取一组相似度 `sims = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]`，用 `b=10, r=10` 逐点求值，代码写成字典推导式 `probs = {sim: get_prob_collision(sim=sim, b=10, r=10) for sim in sims}`。`kf_0033` 的变量面板把结果精确列了出来：0.7 对应 0.2491、0.75 对应 0.4399、0.8 对应 0.6789、0.85 对应 0.8884、0.9 对应 0.9863、0.95 对应 0.9999、0.98 对应 1.0000，正好覆盖转写里说的 "0.25 to 1" 这个区间。讲者据此在 [44:34]–[44:59] 讨论阈值：若把 threshold（阈值）设在 0.9，低于它的仍可能碰撞、产生 false positives（假阳性），需要的话可以再过滤掉；而高于阈值的基本都会保留。这也解释了为什么单看 MinHash 碰撞概率不够、必须靠分带把 S 形压陡：0.7 和 0.9 之间只差 0.2 的相似度，碰撞概率却从 0.2491 跳到 0.9863。

![b=10、r=10 时各相似度对应的碰撞概率字典，以及增大 r 的黄色高亮结论](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0033_00-44-22.jpg)

[45:04]–[45:51] 讨论增大 $r$（记住 $r$ 是 number of hash functions within a bucket，即一个 band 内的哈希函数个数）。因为 $s^r$ 里的指数变大，曲线被 sharpen（锐化）并整体右移，everything becomes harder to match（一切更难匹配）。画面把参数改成 `probs = {sim: get_prob_collision(sim=sim, b=10, r=20) for sim in sims}`，黄色高亮句写着 "Increasing r sharpens the threshold and moves the curve to the right (harder to match)"。`kf_0034` 的字典显示 0.7 从 0.2491 掉到 0.0080、0.75 变 0.0377、0.8 变 0.1059、0.85 变 0.3265、0.9 变 0.7264、0.95 变 0.9882、0.98 仍是 1.0000；讲者口头概括为 "before it was like 0.25 and now it's 0.08"（数字被口语压缩成 0.08）。画面下方 `b values` 的小图还画出 50、25、20、5 四条不同 $b$ 的曲线，为后面的 $b$ 讨论做铺垫。

[45:51]–[46:23] 反过来增大 $b$（number of bands，分带数）。因为 "some band has to match"（只要某个 band 匹配即可），band 越多匹配机会越多，曲线左移、easier to match（更容易匹配）。淡色代码写成 `probs = {sim: get_prob_collision(sim=sim, b=20, r=20) for sim in sims}`，并配一句 "Increasing b moves the curve to the left (easier to match)"。讲者对比前后：0.9 这一档 "before this 0.9 was only .72 and now it's 0.92"，同时低相似度一侧的概率也上升但 "not by that much"（幅度没那么大）。[46:23]–[46:37] 的结论是：同时增大 $b$ 和 $r$ 可以把这条 phase transition 做得 "as sharp as you want"（要多陡有多陡），但 "if you can increase B and R too much, then it can be more expensive"——哈希函数总数 $n=b\cdot r$ 变大，计算与存储成本随之上升。这里的易错点是把「锐化」当成免费午餐：$r$ 增大虽压低假阳性，却也可能把真正相似的候选推向漏检；$b$ 增大虽提高召回，却把假阳性一起抬起来。两者必须成对调节，而「阈值该定在哪、参数取多大」正好引出下一章真实论文的具体设置。

## 真实去重参数、相变阈值与碰撞常数 [46:37 - 48:55]

[46:37]–[46:58] 讲者从「可以任意锐化」转向 "a more real world setting"（更贴近现实的设置），举了一篇 deduplication（去重）论文：用了 `b = 20` bands、`r = 450`。画面（`kf_0035`、`kf_0036`）把这段代码完整写出来：`Example setting [Lee+ 2021]: n = 9000, b = 20, r = 450`，另起两行 `b = 20`、`r = 450`。这里 $n=9000$ 是哈希函数总数，恰好等于 $b\cdot r=20\times 450=9000$；转写只念了 "B 20 bands and R is 450" 和 "the magnitude of these numbers"（这些数的量级），而画面补出了 $n=9000$ 与文献引用 `[Lee+ 2021]`。相比上一章教学用的 $b=10, r=10$（$n=100$），真实去重用的哈希数量高两个数量级，锐化程度与计算代价都大得多。`kf_0035` 上方还画着黑、蓝、红、青四条随 similarity (s) 上升的曲线，横轴刻度 0.25、0.5、0.75、1.0，纵轴可见 0.0、0.25、0.5，直观对应不同参数下相变位置的移动。

[47:04]–[47:23] 给出相变发生的位置：threshold（阈值）近似为 $(1/b)^{1/r}$，转写念作 "one over b raised to the power of one over r"。画面写成可直接执行的一行 `threshold = (1 / b) ** (1 / r)`，上一行还写着问题 "What is the threshold (where the phase transition happens)?"。讲者进一步说明用途：如果想按 "jakard greater than 0.9"（Jaccard 大于 0.9）来过滤，就得把 $b$、$r$ 设成让这个阈值等于 0.9，再通过改变 $b$、$r$ 把相变做得更陡。换句话说，阈值的位置由 $(1/b)^{1/r}$ 决定，相变的陡峭程度由 $b$、$r$ 的绝对大小决定，二者可以分开调节：先定门槛，再定锐度。

[47:29]–[48:10] 讲者回到相变中心。他提醒 "probability that a fixed band matches is one over b"，即阈值处单个 band 匹配概率是 $1/b$；把它代入碰撞公式，A 和 B 碰撞的概率就变成一个常数。画面（`kf_0036`）把这两步写成 `Probability that a fixed band matches: prob_match = (1 / b)` 与 `Probability that A and B collide is a constant (≈ 1-1/e): prob_collision = 1 - (1 - 1 / b) ** b`，并把后一行黄色高亮。转写里讲者把这个常数念成 "64"（约 0.64），画面则点明它约等于 $1-1/e$。这个 0.64 正是 S 形曲线的中心：低于阈值的部分随 $b$、$r$ 增大会趋向 0，高于阈值的部分趋向 1，而中间这个交叉点大致固定在 0.64 不动。这也解释了上一章「增大 $b$ 或 $r$ 是在把曲线绕着中心压陡」的直觉：无论怎么锐化，相变中心的概率高度都不变。

![Lee+ 2021 真实去重参数、相变阈值公式与碰撞常数 1-1/e 的完整代码](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0036_00-48-20.jpg)

画面下方还留着几行转写完全没念到的 Python 辅助函数：`def billion(x): return x * 10**9`、`def trillion(x): return x * 10**12`，以及 `def data_mixing():` 开头、带注释 `# LMs are trained on multiple data sources` 的函数定义。`billion`/`trillion` 是把 "10B"、"10T" 这类口语数字变成可计算常量的小工具，后面几章算 epoch 时会反复用到；`data_mixing()` 则预告了下一主题。这行注释也点明了从去重转到混合的原因：单个数据源处理完之后，语言模型实际是在 multiple data sources（多个数据源）上训练的，于是「怎么分配这些来源」才成为问题。

[48:10]–[48:39] 有学生提问后，讲者补了一个术语澄清：这套方法叫 "minash LSH"（即 MinHash LSH），因为 LSH（locality-sensitive hashing，局部敏感哈希）本身 "really works for any hash functions"（对任意哈希函数都成立），而在语言模型处理的去重里用的是 "the minash which approximates your card"（近似 Jaccard 的 MinHash），所以这里是 MinHash 与 LSH 的组合，并且 "that basically is the one you should use here"（这里该用的就是它）。这里的易错点有两个：一是把常数 0.64 误当成可调的 threshold，其实 threshold 是相似度轴上的 $(1/b)^{1/r}$，0.64 只是概率轴上的相变中心高度；二是把 LSH 当成 MinHash 的一部分，实际上 LSH 是分带锐化的通用框架、MinHash 只是被代入的那个哈希，换别的哈希也能套 LSH，但只有 MinHash 的碰撞概率等于 Jaccard，才能对应我们想要的相似度语义。[48:39] 讲者以 "questions about DDUP" 收尾本主题，紧接着在下一章补一条关于去重范围的实践提醒。

## 从去重到数据混合：跨数据集冗余与 The Pile 的权重表 [48:55 - 51:02]

[48:55]–[49:19] 讲者先补 "one note about DDUP"：数据集常常是陆续进来的，去重有时只在单个数据集内部做，但 "you actually have to do duplication across your entire data set"（应当跨整个训练集去重），因为 "often data sets can be redundant"（不同数据集之间往往互相冗余）。他加了一句 "sometimes that's not done but it should be"（有时没做，但应该做）。这是对上一章 MinHash LSH 适用范围的修正：算法能在线性时间内找出近似重复，但比对范围必须覆盖全部数据源，否则跨源的重复会漏网；去重因此不是「每个来源各跑一遍」，而是全局的一次操作。

[49:19]–[49:39] 讲者宣布 "let's go on to the next topic which is data mixing"，并回顾整条管道：把 raw HTML or PDFs（原始 HTML/PDF）转成文本，filter for high quality（过滤出高质量内容），dduped（去重），于是得到一个 "smaller set of high quality documents"（更小的高质量文档集）。`kf_0037` 的大纲页正好停在这里：它把 `This lecture` 拆成 `Data pipeline: transformation, filtering, deduplication, mixing` 与 `Mid-training + SFT: synthetic data` 两行，并用一行行函数调用把管道写成 `transformation()`、`filtering()`、`dedupplication()`、`data_mixing()`，其中 `data_mixing()` 被黄色高亮，下面还有 `post_training_data()`；页面上半还保留着 `Last lecture` 的 `Live service (e.g., GitHub) → dump/crawl (e.g., GitHub Archive) → processed data (e.g., The Stack)` 与 `Considerations: terms of service, copyright (licenses or fair use)`。底部 `Summary` 里有一句概括本主题动机的话："Mixing: try mixtures at small scale, extrapolate to optimal mixture and large scale"（在小规模上试混合，外推到最优混合与大规模），这正是后面回归式方法的提纲。

[49:39]–[50:35] 讲者指出前面的管道 "generally happens for a given data source"（一般针对单个数据源），但语言模型是在 multiple data sources（多个数据源）上训练的。他提到 Marin 的一个网站会追踪 "the different data sources that the next model will be trained on"（下一代模型将训练于的各类数据源），画面里能辨认出的有 fine PDFs（前面讲过的）、institutional books（机构书籍）、code（代码）等。于是问题变成 "how do you combine these?"（怎么把这些组合起来）。这一步把「单来源的高质量文档集」升级成「多来源的集合」，混合（mixing）就是给这个集合定一个采样分布。

[50:35]–[50:56] 讲者翻出老论文 The Pile，说它给每个组件 "assign a particular weight"（分配一个特定权重），于是 "you have a distribution of resources"（得到资源上的一个分布）。`kf_0038` 把这张 `The Pile [Gao+ 2020]` 表完整列出，列标题是 Component、Raw Size、Weight、Epochs、Effective Size、Mean Document Size，共 22 个组件加一行合计。数字很具体：Pile-CC 的 Raw Size 是 227.12 GiB、Weight 18.11%、Epochs 1.0、Effective Size 227.12 GiB、Mean Document Size 4.33 KiB；PubMed Central 是 90.27 GiB、14.40%、2.0 个 epoch、180.55 GiB、30.55 KiB；Books3 是 100.96 GiB、12.07%、1.5、151.44 GiB、538.36 KiB；Wikipedia (en) 只有 6.38 GiB，却给了 3.0 个 epoch、Weight 1.53%、Effective Size 19.13 GiB；合计行是 Raw Size 825.18 GiB、Effective Size 1254.20 GiB、Mean Document Size 5.91 KiB。从这些数能看出 Effective Size 大致是 Raw Size 乘 Epochs（如 $90.27\times2\approx180.55$），而 Weight 又按 Effective Size 归一化（$227.12/1254.20\approx18.11\%$）：也就是说 The Pile 不是简单按原始大小采样，而是先人为给小的高质量源多跑几个 epoch 放大它，再据此定权重。表格里多数来源的 Epochs 是 2.0，Wikipedia (en) 和 Gutenberg (PG-19) 分别到 3.0 与 2.5，正说明「上采样小来源」是被显式写进配比的。

![The Pile（Gao+ 2020）各组件的 Raw Size、Weight、Epochs、Effective Size 统计表](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0038_00-50-59.jpg)

[50:56]–[51:02] 讲者抛出 "where does that weight come from?"（权重从哪来），画面底部那句 "Key question: what distribution over the data sources should we use?"（关键问题：我们该在数据源上用什么样的分布）把问题钉死。从管道位置看，去重解决的是「文档是否重复」，混合解决的是「重复清理完之后，每个来源占多少比例」，二者是先后相接的两步；而 The Pile 表是「手工给每个组件填一个权重」的一个历史答案。这里的易错点是把 The Pile 的 Weight 列当成「数据源的原始占比」直接照搬——它其实是 Epochs 人为上采样之后的结果，若只抄权重、不意识到背后已经隐含了多轮 epoch，就会在下一章的语料有限性问题上踩坑。下一章先把这个问题形式化成一个分布 $p(s)$，再逐一检视几种定权重的基线。

## 数据混合的三种基线与多样性、有限性两个约束 [51:02 - 53:41]

[51:02]–[51:15] 讲者把问题收敛成 "let's say you have three sources you want to find a data mixture"（假设有三个来源，要找一个数据混合），并给出定义："a data mixture is just a distribution over your sources"（数据混合就是各来源上的一个分布）。`kf_0040` 用一屏代码把它写死：`Example: sources = {"Wikipedia", "CC", "GitHub"}`，以及 `p = {"Wikipedia": 0.3, "CC": 0.5, "GitHub": 0.2}  # One possible data mixture`。也就是说混合就是给每个来源 $s$ 一个概率 $p(s)$，权重之和为 1，训练时按这个分布抽来源。转写里没有念出 0.3、0.5、0.2 这三个具体数字，也没有把三个来源的名字一并报全，都是画面独有的增量。这个定义把上一章 The Pile 表里那一列 Weight 抽象成了可操作的概率向量。

![数据混合示例与 Vibes、Uniform、Proportional 三条基线及多样性/有限性约束](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0040_00-53-38.jpg)

[51:15]–[52:32] 讲者从 "vibes"（凭感觉，他自嘲这是 "opposite of first principles"、与第一性原理相反）开始列基线。画面把它们写成 `Baselines:` 下的三条：其一是 `Vibes: set p(s) manually based on intuition (quite common)`，即凭直觉手动设 $p(s)$，讲者强调 "more often than you might think"（比想象的常见），并说 "even more recent papers" 也常 "use some method and then you just like tweak things"（用个方法再手调）；其二是 `Uniform sampling: sample uniformly (p(s) ∝ 1)`，对每个来源等概率，每取一个 chunk（数据块）就抽一个来源；其三是 `Proportional mixing: sample proportional to the number of tokens in a source (p(s) ∝ num_tokens(s))`，按来源的 token 数占比采样，token 多的来源权重大，"this is also generally a rational thing to do"（一般也算合理）。这三条把「分布」从抽象定义落到了可执行的采样规则上，其中 $\propto 1$ 与 $\propto \text{num\_tokens}(s)$ 是画面写出的正比关系，转写只做了口头描述。

讲者立刻指出比例混合的毛病：如果有一个 "a huge lowquality data set"（巨大的低质量数据集），它会 "eat up a lot of your tokens"（吃掉大量 token 预算），"this doesn't seem quite optimal either"（也不算最优）。于是 [52:35]–[52:41] 给出直觉 `Intuition: should upweight higher quality sources`（应该给高质量来源更高权重），画面把它单独成行。但紧接着的 `However...` 后面跟着两个必须同时记住的约束，它们决定了「加权重」不能无限做下去。

第一个是多样性："We want to ensure diversity (e.g., across incomparable sources: literature, code, papers)"——文学、代码、论文这些来源 "incomparable"（不可比），"you can't really say that this paper is higher quality than this code"（没法说这篇论文比这段代码质量高），所以不能 "put all your mass on just papers"（把全部概率质量压在论文上），否则语言模型的能力面会变窄。第二个是有限性："Each source is finite, so if put too much weight on a small source, then need to epoch over it"（每个来源都是有限的，给小来源过大权重就得反复 epoch），也就是 "keep on training on the same literally the same tokens"（反复在字面上完全相同的 token 上训练），讲者预告 "this is going to be bad for reasons we'll see"（后面会看到这很糟）。`kf_0040` 把第二条黄色高亮，说明它是本讲的转折点；画面下方还预先铺了 `source_token_counts = {"low": trillion(10), # 10T tokens (abundant), "high": billion(10), # 10B tokens (scarce)}` 与 `p = {"low": 0.5, "high": 0.5}  # Naive data mixture`，为下一章的定量计算备好了数据。

从结构上看，这一章把上一章 The Pile 表里那些 Weight 数字抽象成了 $p(s)$，并第一次把「权重要不要按质量调」与「来源有多大」这两件事分开。易错点是把 "upweight higher quality"（给高质量加权重）无条件当成正确方向：它同时被多样性与有限性两个约束夹住，只顾质量会把小语料压榨到几十个 epoch。另一个易错点是以为 uniform sampling 最安全——它同样会给 10B 的小来源和 10T 的大来源各 0.5 的概率，有限性问题一点没缓解。下一章就用一个两来源、1T token 的具体算例把这个陷阱量化出来。

## epoch 陷阱：10T 与 10B 两来源的 50 倍重复 [53:41 - 57:10]

[53:41]–[54:06] 讲者把上一章的「有限性」展开成具体算例：一个 low quality（低质量）来源有 10 trillion tokens（10T），一个 high quality（高质量）来源只有 10 billion tokens（10B），并说明 "Generally high quality sources are smaller"（高质量来源通常更小）。若采用 naive（天真）的均匀混合 `p = {"low": 0.5, "high": 0.5}  # Naive data mixture`，并 "train for one trillion tokens"（训练 1T token）。`kf_0041`、`kf_0042` 把设定写成可执行代码：`source_token_counts = {"low": trillion(10), # 10T tokens (abundant), "high": billion(10), # 10B tokens (scarce)}` 与 `train_tokens = trillion(1)  # Train for 1T tokens`，其中 `trillion()`、`billion()` 正是第 2 章画面里那两个 `x * 10**12`、`x * 10**9` 的辅助函数。讲者特别在 [54:06]–[54:23] 澄清："train for one trillion tokens" 指的并不是 1T 个 unique tokens（唯一 token），而是训练步数乘 batch size 得到的总消耗量，这个区分是后面所有 epoch 计算成立的前提。

[54:23]–[54:59] 定义 "the number of times a particular element in the low quality data set was trained on"（低质量集中某个元素被训练的次数），也就是该数据点的 epoch 数。计算式是 $p(\text{low})\cdot\text{train\_tokens}\ /\ |\text{low}|$，即向该来源请求的总 token 数除以该来源拥有的 token 总数。画面写成两行：`low_num_epochs = (p["low"] * train_tokens) / source_token_counts["low"]` 与 `high_num_epochs = (p["high"] * train_tokens) / source_token_counts["high"]`，第一行在 `kf_0041` 里被黄色高亮。代入 $p=0.5$、$\text{train\_tokens}=10^{12}$、$|\text{low}|=10^{13}$ 得 $0.05$，也就是 "less than one"（小于 1）：讲者说这意味着低质量集里只有 "5% of the tokens in this low quality data set I'm even going to touch and train on once"（只有 5% 的 token 会被碰到、且只训练一次）。

[55:07]–[55:36] 看高质量侧就完全相反：$0.5\times10^{12}=500$ billion tokens（5000 亿），但来源只有 10 billion，于是 $500/10=50$，"the number of epochs is 50"。讲者把因果说得很直白："I need to train up 500 billion tokens of high quality data. But I don't have 500 billion tokens of data. So I have to repeat each data point 50 times."（我需要 5000 亿高质量 token，但我没有，只能把每条数据重复 50 次）。`kf_0042` 右侧变量面板把两个结果并排显示：`low_num_epochs = 0.05`、`high_num_epochs = 50`，下方黄色高亮句写着 `50x epochs on high quality data...can lead to overfitting!`（高质量数据上 50 倍 epoch 会导致过拟合）。同一个 0.5 的权重，在 10T 上只碰 5% 的数据，在 10B 上却要把每条重复 50 遍，落差达三个数量级。

![低质量 10T 与高质量 10B 两来源下 low_num_epochs=0.05、high_num_epochs=50 的计算代码](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0042_00-56-17.jpg)

[55:36]–[56:03] 讲者强调 "this is actually really important"，并说 "some big model training runs have kind of messed this up"（一些大模型训练在这上面翻过车）。核心教训是不能 "naively look at the data sets and define a distribution and then you sample"（天真地看着数据集定个分布就开始采样），因为 "if you just look at the quality ... you're missing how many data points are there"（只盯着质量，就漏掉了数据点数量这个维度）。这把上一章两个约束里的「有限性」从定性警告变成了可算的 $\text{epoch}(s)=p(s)\cdot T/|s|$：权重 $p(s)$ 与语料规模 $|s|$ 必须一起看。讲者在 [56:00] 特意停下来 "pause for questions"，说明这是本讲最想让学生记住的点。

[56:25]–[57:10] 有学生问 "why do we need 50 epochs in the first place"（一开始为什么需要 50 个 epoch）、少训练些不是也能达到相近效果吗。讲者回答得很干脆："you don't need 50 epochs"——"best case is wasting compute and worst case you're overfitting"（最好情况是浪费算力，最坏情况是过拟合）。50 这个数字不是设计目标，而是 "if you just go in and you define this data mixture then you're going to be end up doing 50 epochs without realizing unless you're paying close attention"（只要按那个混合比例去定义、又不格外留意，就会不知不觉跑出 50 个 epoch）。因此他给出的 lesson 是 "look at how many epochs you're actually doing on your data"（去看你实际在数据上做了多少 epoch）。这里的局限也很清楚：这个算例只用了两个来源、一个静态分布，没有涉及训练过程中动态调权；而「混合比例在训练时究竟怎么落地」正是下一章第一个问答要解决的。

## 训练时混合如何落地，以及 UniMax 的 epoch 硬上限 [57:10 - 59:59]

[57:10]–[57:25] 学生问 "how do the mixtures get expressed during training"（混合在训练时怎么表达）：是每一步切换来源，还是 "steps of 10 saying oh I want to do 10 code 10 to see"（每 10 步说要做 10 步代码、10 步别的）？[57:25]–[58:02] 讲者回答：一般是要 "fill a batch"（填满一个 batch），于是 "for each element" 采样它 "which mixture component it comes from"（来自哪个混合组件），再用来自该组件的序列把 batch 填满。他强调 "Usually every sequence comes from one mixture component. You're not sampling like per token."（通常每条序列来自单个组件，不是逐 token 采样）。这把上一章的抽象分布 $p(s)$ 落到了 batch 构造上：$p(s)$ 是逐序列（element）的抽样概率，而非逐 token 的概率，因此一个 chunk 内部不会在来源之间跳来跳去。

[58:02]–[58:25] 学生接着确认 "each batch should be mixed"（每个 batch 应该是混合的），讲者同意并补充 "in general you want to reduce variance you want to train from multiple ... a batch should have some mixture in it"（为降低方差，一个 batch 里应当有多个来源的混合）。也就是说混合是在 batch 层面生效的假设：不是「这一整步全是代码、下一步全是论文」，而是同一个 batch 内就掺着多个组件。这一段是纯口头问答，画面是讲者在讲台前的照片，没有新公式，但它澄清了 $p(s)$ 的作用粒度，是理解前面所有混合计算的必要前提；若误以为混合是逐 step 切换，前面按 token 数算 epoch 的整个框架就对不上了。

[58:25]–[58:56] 讲者说 "this problem has been noticed quite a long time ago"（这个问题很早就被注意到），并引出一篇论文 "introduced a unimax"（提出 UniMax），当时的设定是 "training in multilingual models"（训练多语言模型），"some languages were very low resource"（有些语言是极低资源），所以必须处理小来源被过度 epoch 的问题。画面（`kf_0044`）给出确切引用 `UniMax [Chung+ 2023]` 与 `Setting: balancing different languages for multilingual models`。[58:56]–[59:09] 讲者回顾更早的做法：把 proportional mixing（比例混合）"raise it to some power"（取某个幂）"to kind of flatten it out the distribution"（把分布压平）；画面把它写成 `Previous work: between uniform and proportional mixing (p(s) ∝ num_tokens(s)^α for α in [0, 1])` 并黄色高亮——这里 $\alpha\in[0,1]$ 是在「按 token 数比例」（$\alpha=1$）与「均匀」（$\alpha=0$）之间插值，转写只说「取某个幂」，指数范围与这条公式是画面独有的增量。

[59:09]–[59:47] 讲者说这篇论文要 "be a bit more explicit"（更明确一点）："sample the sources uniformly but with a hard cap on the number of epochs"（对来源均匀采样，但对 epoch 数设硬上限）。画面写成 `Idea: sample sources uniformly but with a hard cap C on number of epochs for any source` 与 `Specifically, p(s) * num_training_tokens ≤ C for all sources s`。讲者举例 "I'm only going to take 20 epochs over a particular data source"（某个来源最多只取 20 个 epoch），一旦达到 "too bad ... you don't get any more tokens. You move on to the next thing"（就没更多 token 了，换下一个），他把这称为 "a safety net"（安全网）。写成约束就是画面里的 $p(s)\cdot\text{num\_training\_tokens}\le C\ \text{for all sources }s$；讲者口头把它表述为 "the probability assigned to a source times the number of training tokens that should be less or equal to the cap"。这正好能把上一章算出的 50 强行截到 20 这个量级。[59:47]–[59:59] 讲者说 "there's a simple procedure for actually just determine the mixture subject to this constraint"（有个简单过程能在该约束下解出混合比例），但没有展开细节。

![UniMax（Chung+ 2023）的幂次法、hard cap C 约束与回归式混合预告](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0044_00-58-56.jpg)

从结构上看，UniMax 是对上一章 epoch 陷阱最直接的补丁：不改变「按分布采样」的框架，只加一条硬约束，把小来源的重复次数钉死。画面下方已经露出下一主题的两行 `Regression-based mixing [Liu+ 2024][Chen+ 2026]`，以及 `① Train small-scale proxy models`、`② Fit a regression model using data mixture as source`，预告讲者要 "switch gears"（换个角度）去问「混合本身该怎么定」。这里的易错点是把 hard cap 当成调权重：它并不重新分配 $p(s)$，而是给每个来源独立设一道天花板；此外 $\alpha\in[0,1]$ 的幂次法只是把分布在均匀与比例之间平滑插值，并不能保证某个具体来源不超过 epoch 上限，两者作用层次不同。另一个局限是 UniMax 的动机来自多语言低资源场景，把它直接搬到「按质量分层」的通用语料上时，cap 该取多少仍需另行判断。

## 回归式数据混合 RegMix：用小规模代理实验外推最优配比 [59:59 - 63:34]

[59:59]–[60:26] 讲者 "switch gears a little bit"（换个角度），问 "how do we define the mixture in the first place"（混合最初该怎么定）："if you have 50 sources, there's 50 numbers you have to fill"（有 50 个来源就要填 50 个数），所以靠直觉手调并不现实。他先否掉两条路：proportional mixing（比例混合）"isn't really enough"（不够）；即便估计某种 quality metric（质量指标）再按之采样，"that is also heristic"（也是启发式）。[60:26]–[60:40] 他给出评价：最 principled（最有原则）、也最容易看懂的是 "these regression based mixing methods"（这类基于回归的混合方法），点名 "ragmix ... mix and other papers like this"（RegMix 等）。画面（`kf_0045`、`kf_0046`）给出确切标题与引用 `Regression-based mixing [Liu+ 2024][Chen+ 2026]`，把转写里含糊的论文名落实为两条文献。

[60:40]–[61:57] 讲者描述流程，`kf_0046` 把它画成编号流程图。第 ① 步 `Train small-scale proxy models`：不去直接训大模型，先在小规模（讲者举例 "300 million parameters"，即 3 亿参数）上试不同混合，"you train a swarm of small models"（训一群小模型），每个模型给出 "a loss on some target metric"（某个目标指标上的 loss），"it could be some downstream evals, it could be perplexity, whatever you want"（可以是下游评测，也可以是困惑度）。画面用 Hacker News、Github、Philpapers 三个来源做示例，列出三组配比及各自得到的 `Target ↓` 值：`9.5% / 35.9% / 54.6% → 5.46`、`87.7% / 12.0% / 0.3% → 5.57`、`24.4% / 1.4% / 74.2% → 6.07`，每组旁边还画了一条随 step 下降的小 loss 曲线。第 ② 步 `Fit a regression model using data mixture as source`：用这些数据点拟合一个回归模型，输入是混合权重、输出是 loss，画面框里给出两种候选 `Linear Model` 与 `Tree Model`。讲者说这样就有了 "a very cheap model"（一个很便宜的模型），能回答 "if I were to train on this mixture what loss would I get"（若用这个混合训练会得到什么 loss）。

第 ③ 步 `Simulate new data mixtures and predict Target`：在这个便宜模型上 "optimize that function for the optimal data mixture"（优化出最优混合）。画面画了一个 3D 曲面，横轴 `Hacker News (%)`、纵轴 `Github (%)`、竖轴 `Prediction`，色标从 5.4 到 6.0，红蓝两色对应预测值的高低。第 ④ 步 `Train a large-scale model on the best mixture`：把 `Simulated Best Data Mixture` 定为 `Hacker News 22.8% / Github 67.0% / Philpapers 10.2%`，其 `Predicted Best Target` 是 `Prediction (Lowest) 5.34`，再用这个混合去训练大模型。这些百分比、5.46 / 5.57 / 6.07 / 5.34 以及 5.4–6.0 的色标都是画面独有、转写完全没念到的细节；转写只讲到 "train a swarm of small models"、"each model gives you a loss"、"fit a regression"、"optimize that function" 这一层。[62:03]–[62:14] 讲者还点出它 "similarities with scaling laws"（与缩放定律相似）：都是用廉价计算先找答案，再 scale up（放大规模）。

![RegMix 回归式数据混合四步流程图，含三组代理配比、loss 值与最优混合 22.8%/67.0%/10.2%](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0046_01-01-35.jpg)

[62:19]–[63:19] 讲者列 "a few design decisions"。`kf_0047` 把它们写成三条 bullet 并高亮第三条：其一是 `Define distribution over mixtures p (e.g., Dirichlet)`——因为 "you need a distribution over distributions here"（需要一个分布之上的分布）来采样要试的混合，讲者口头念成 "durlay distribution"，画面明确是 Dirichlet（狄利克雷）分布；其二是 `Define regression method (e.g., linear, gradient boosted trees)`，对应转写里的 "linear models or boosted decision trees"；其三是 `Define target based on downstream evals (careful not to overfit)`，黄色高亮。讲者对第三条的警告最长：预训练本应产出 "general purpose model"（通用模型），"we're not trying to fit some downstream evals"（并不是要拟合某些下游评测），若 "you have a bunch of code evals, then well, guess what? You're going to upweight all the code data"（若一堆代码评测，优化器自然会给代码数据加权重），等你想 "generate some poetry"（生成诗歌）时 "you might realize that you've over over fit"（才发现过拟合了）。他补一句 "proportional mixing and uniform mixing don't have this problem because there's no downstream evals"（比例混合与均匀混合没这毛病，因为根本不看下游评测）。

`kf_0047` 还有一条淡色子 bullet `Discrepancy between small and large scale (tradeoff cost and accuracy)`，对应 [63:27] 起讲者最后提的 "how much of a difference between small and large scale is there and this is a cost and accuracy trade-off"（小规模与大规模差多少，是成本与精度的权衡）；转写在 63:34 处被截断，未展开。把这一章与前面连起来看：第 4 章的 vibes / uniform / proportional 是零成本的启发式、没有下游评测因而不会过拟合；RegMix 用小规模代理把「混合 → loss」拟合成可优化的函数，代价是引入了三个新的信念跳跃——代理实验的混合分布（Dirichlet）覆盖不到的外推区域预测未必可靠、目标选下游 eval 会把模型带偏、小模型上的最优配比未必迁移到大模型。这与第 3 章画面 `Summary` 里那句 "try mixtures at small scale, extrapolate to optimal mixture and large scale" 首尾呼应，也回答了 The Pile 表底下那句 "what distribution over the data sources should we use?"：与其手填 50 个数，不如用一群小模型把这张表学出来。

## 回归式数据混合的方法谱系与两次信念跳跃 [63:34 - 66:24]

本段一开头，讲者先为前面建立的「回归式混合」（regression-based mixing）框架补上一个尺度前提。这个框架的骨架是：训练一批小规模代理模型（proxy models），让每台代理跑一种不同的数据混合（mixture），再用这些实验结果拟合一个从混合权重到 loss 的回归函数，在这个回归函数上做优化得到最优混合，最后把这个混合迁移到大模型上。讲者提醒，如果代理模型训练得「really really small」，得到的结论未必有代表性（might not be representative）；可如果反过来真的在最大规模上把各种混合都跑一遍，那就等于在最大尺度上做超参数调优（hyperparameter tuning at the largest scale），成本过于昂贵，等于把这套框架的意义抵消掉了。换句话说，整套方法的价值恰恰卡在一个中间地带：代理要足够小才付得起，又不能小到失去外推能力。讲者顺带把这类方法的 proxy model 规模概括为数千万参数（tens of millions of parameters）量级，但紧接着展示的表格说明真实取值跨度要大得多。

![对比七种回归式数据混合方法设计选择的完整表格](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0048_01-04-19.jpg)

讲者口中称作 Omix（也写作 OMIX）的论文给出一张横向对照表，把落进这一框架的多种方法并排列出。转写里只含糊说了「a nice table which shows you for a number of different methods」，并没有念出表里的具体条目，因此表格内容几乎全是画面独有的增量信息。表格第一列是 Design Choice（设计选择），纵向分成三大块：Swarm Construction（代理实验群的构造）、Regression Model（回归模型）、Mixture Optimization（混合优化）。横向的七种方法依次是 RegMix (Liu et al., 2025a)、DML (Ye et al., 2025)、AutoScale (Kang et al., 2025)、BiMix (Ge et al., 2025b)、ADMIRE-BayesOpt (Chen et al., 2025b)、CLIMB (Diao et al., 2025) 以及表格末列的 OlmixBase (Algorithm 1)。表格上方还留着三条设计 bullet：Define regression method (e.g., linear, gradient boosted trees)、Define target based on downstream evals (careful not to overfit!)、Discrepancy between small and large scale (tradeoff cost and accuracy)——也就是说，选回归族、选优化目标、权衡大小尺度的成本与精度，是这一框架必须先定下的三个旋钮。

逐行读表可以看到设计空间有多分散。Proxy model size 一行从 1M、70/160/305/410M、Target、280M、1M 与 60M、350M 一直排到 30M；Swarm size (vs $m$ domains) 一行给出 512 ($m=17$)、20 ($m=7$)、$2m+1$、4、101 ($m=17$)、112 ($m=21$)、$3(m+1)$。这里的 $m$ 讲者在转写里当场纠正过一次：他先说成 parameters in your mixture，随即改口为 the number of domains rather，即「域的数量」而非混合参数的个数，也就是要采多少个不同的混合点。Swarm distribution（代理混合的采样分布）一行出现 Dirichlet with natural prior（带自然先验的狄利克雷分布）、Exponential grid（指数网格）、Entropy-weighted、Dynamic，以及 OlmixBase 写得更细的 Dirichlet with natural prior (sparse for topics, dense for sources)——对 topic 用稀疏先验、对 source 用稠密先验。讲者口述只提到「you can use Dirichlet, you can use exponential」，表格则把这些分布与具体方法一一对应了起来。

Regression model family 一行列出 LightGBM、Log-Linear、Power Law、Power Law、Gaussian Process、LightGBM、Log-Linear；讲者特别点出「you can fit a log linear model which tends to work pretty well」，正对应表中反复出现的 Log-Linear（对数线性模型）。Regression granularity（回归粒度）只有两类：Aggregated（把所有任务聚合成一个指标）与 Per-Task（逐任务各拟合一个）。Mixture Optimization 下面两行更关键：Data repetition constraints（数据重复约束）一列前六种方法全是 No，只有 OlmixBase 是 Yes；Optimization solver（优化求解器）一列则有 Search、Gradient Descent、Exact Solver，以及 OlmixBase 的 Exact Solver with KL reg.（带 KL 正则的精确求解器）。这几行把后面要讲的 epoch 问题提前埋进了表格：多数方法根本没约束数据重复，只有一家显式加了约束。

表格底部把这套框架的两处 leap of faith（信念跳跃）写成带祈祷手势的两行：Hope 1: regression model is accurate at minimizer（回归模型在最优点处仍然准确）；Hope 2: optimal data mixtures transfer from small to large scale（小规模最优混合能迁移到大规模）。讲者展开第一条：回归模型是在一批小规模代理 runs 上训练的，随后却要在它上面做优化；如果只是随机采一个混合来预测，经典泛化（generalization）理论保证 in-distribution 预测大体可靠，但优化（optimizing）的本质是往 extremes（极端区域）走，而极端区域恰恰是训练分布覆盖可能不足的地方，回归预测未必还准。第二条是小到大迁移：讲者说在开放社区能做的尺度上，这件事「tends to be true or not blatantly false」，但明确存在 scale-dependent effects（尺度依赖效应）。他回指本讲前面过滤部分的结论——训练 token 越多，低质量数据越能接受——说明最优混合显然不会与小规模相同，实践上只能「hope for the best」。

从前后关系看，这一章既是上一节「回归式混合」的收束，也是下一章的引子：讲者紧接着就说「Hold on. There's at least one scale-dependent effect」，要把 epoch 这个具体效应单独拎出来处理。易错点也很清楚：这张表说明框架并没有唯一标准实现，光 proxy 规模、swarm 分布、回归族、回归粒度、求解器就有七种以上的组合，任何「回归式混合有效/无效」的结论都绑定在具体设计选择上；而把 downstream evals 当作优化目标时，表上那句 careful not to overfit! 是硬提醒——目标里代码类评测多，优化器自然偏向代码数据，未必带来通用能力。此外，OlmixBase 是表中唯一同时设了数据重复约束并采用带 KL 正则精确求解的方法，这个例外恰好指向下一章要解决的过拟合风险。

## 尺度依赖效应：高质量小源上的 epoch 陷阱 [66:24 - 67:40]

讲者用一个人为构造的例子（他特地说明 this is a made up example）把尺度依赖效应具体化。设定是：低质量数据有 10 trillion tokens（$10\text{T}$，充裕 abundant），高质量数据只有 10 billion tokens（$10\text{B}$，稀缺 scarce）。幻灯片把它写成了可直接读的代码，这段原文在转写里没有被逐字念出，属于画面独有：

```python
source_token_counts = {
    "low": trillion(10),   # 10T tokens (abundant)
    "high": billion(10),   # 10B tokens (scarce)
}
```

如果在低 token 预算下训练小模型，因为总量小、根本不会触发重复采样（not epoching），优化器会「发现」Wikipedia 这类高质量源特别好用，于是把大量质量（mass）压到它上面，得到 $p = \{\text{"low"}: 0.1,\ \text{"high"}: 0.9\}$ 这样的混合——幻灯片在字典后加了注释 `# More mass on high quality data`。讲者的口语版本是「oh wow, Wikipedia is so great. Let's just train on Wikipedia」。问题在于，把这个在小规模上看起来最优的混合直接拿去训练大模型、跑大 token 预算时，高质量源只有 $10\text{B}$，却要被请求远多于它容量的 token，结果就是对这一小块数据反复 epoch（epoing a lot），最终过拟合（overfit），幻灯片把结论高亮成：But if train large model on this mixture, we will epoch a ton on high quality data and overfit!

![scale-dependent effect 的 source_token_counts 代码与低质量/高质量混合比例](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0050_01-07-21.jpg)

这里的概念是 epoch（训练轮数）：某个源被采样到的 token 数除以它自身拥有的 token 数，就是该源被「过了几遍」。设总训练量为 $T$、混合权重为 $p(s)$、源 $s$ 的规模为 $N_s$，则 $\text{epochs}(s) = p(s)\,T / N_s$。小规模时 $T$ 小，即使 $p(\text{high})=0.9$，$\text{epochs}(\text{high})$ 也可能还接近 1；但把 $T$ 放大到万亿量级，$p(\text{high})\,T / 10\text{B}$ 会变成几十倍甚至更多，性质就从「学一遍」变成「反复背诵」，是 qualitatively different operations（性质不同的操作）。这与前一章第二条信念跳跃直接呼应：最优混合之所以不能从小迁移到大，epoch 结构在两端的巨大差异就是最具体的机制。

讲者给出两类修补方向。第一类被他归到 OMIX 论文名下：直接给 epoch 数设上限，即 capped epoching（封顶 epoch），对应上一章表格里 OlmixBase 那一列 Data repetition constraints = Yes。第二类是下一章要展开的 simulate epoching（模拟 epoch），其一般原则是 make your small scale look like your large scale——让小规模实验「看起来像」大规模训练。讲者强调这是整门课的 general theme（通用主题），并回指 Tatsu 讲 MuP（Maximal Update Parametrization）时的思路：通过重新参数化模型，让超参数能够跨尺度迁移；这里做的不是参数化，而是让数据侧的实验条件跨尺度对齐。

易错点与局限在于：这个例子里的 0.1/0.9 是人为设定，用来暴露机制而非报告实测结果；真实语料里「高质量」与「低质量」也没有 $10\text{T}$ 比 $10\text{B}$ 这样整齐的界限。更根本的是，讲者后面会总结成一句警告——if you're trying to optimize anything you have to be very careful because you are in danger of optimizing the wrong thing：优化器在代理尺度上最小化的 loss，未必是你在目标尺度上真正关心的量。封顶 epoch 虽然直接，但只是把症状压住；模拟 epoch 则试图从实验设计上消除尺度错位，代价是要重新构造代理数据集。这两条路线的差别，正是下一章的内容。

## 模拟 epoching 的实现、参数与混合小节总结 [67:40 - 70:36]

simulate epoching（讲者说它 goes by probably multiple names，其中一个名字就是 simulate epoching，幻灯片标注引用 [Held+ 2025]）的目标是把上一章的尺度错位消掉。幻灯片把一般原则写成 General idea: make small scale look like large scale (general theme of this course)，把具体做法写成 Instantiation: downsample all sources proportionally（按比例下采样所有来源）。之所以要「按比例」，是因为朴素放大会造成小尺度上不 epoch、大尺度上疯狂 epoch 的定性差异；只有让代理实验也按同样比例缩减每个源的可用量，两边的数据稀缺结构才一致。完整代码原文如下，转写只口述了 10 billion、one trillion 与 one in 100 downsampling 这几个数，代码本身是画面独有：

```python
small_run_tokens = billion(10)
large_run_tokens = trillion(1)
ratio = small_run_tokens / large_run_tokens
downsampled_source_token_counts = {s: count * ratio for s, count in source_token_counts.items()}
```

其中 $\text{ratio} = \dfrac{\text{small\_run\_tokens}}{\text{large\_run\_tokens}} = \dfrac{10^{10}}{10^{12}} = 0.01$，正对应讲者口头说的「one in 100 kind of downsampling」（百分之一的下采样）。画面右侧的蓝色方框把代入后的数字写得很具体：`low_num_epochs = 0.05`、`high_num_epochs = 50`、`ratio = 0.01`，以及 `downsampled_source_token_counts` 里 low 变成 `100,000,000,000`、high 变成 `100,000,000`。也就是说，低质量源从 $10\text{T}$ 缩到 $100\text{B}$，高质量源从 $10\text{B}$ 缩到 $100\text{M}$；在代理实验里，若仍按 0.9 的质量压在 high 上，它会被 epoch 约 50 次，而 low 只有 0.05 次。这个 50 与 0.05 的悬殊对比，就是让优化器「自己感觉到」过拟合代价的关键。

![模拟 epoching 的 downsampling 代码、ratio 代入与 num_epochs 数值方框](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0051_01-08-52.jpg)

于是幻灯片给出两条推论：In this downsampled mixture, models that epoch too much won't look good.（在这个下采样混合里，epoch 太多的混合不会显得好）；So the optimum will be more balanced.（因此最优解会更均衡）。对应的新最优写作 $p = \{\text{"low"}: 0.7,\ \text{"high"}: 0.3\}$，注释仍是 `# More mass on high quality data`，但 high 的权重已从上一章的 0.9 回落到 0.3。讲者的口语解释是：你不再被允许「只训 Wikipedia」，只能训它的一个 minuscule fraction（极小比例），于是会意识到自己要在上面反复 epoch，loss 会很差，优化自然转向更平衡的配比。他把这概括成 you're kind of simulating the data scarcity that you would get in your high regime at a low scale——在低尺度上把高尺度才会出现的数据稀缺提前模拟出来。

紧接着幻灯片用三条 Summary 收束整个数据混合小节，这三条同样是画面原文：Problem: how to weight different data sources (e.g., Wikipedia, general, code)（问题是如何给 Wikipedia、通用网页、代码等不同来源分配权重）；Regression-based mixing: estimate mixture → loss at small scale, optimize (analogous to scaling laws)（回归式混合：在小尺度估计「混合 → loss」的映射再优化，并明确类比 scaling laws 缩放定律）；Important consideration: epoching and overfitting (solution: cap or simulated)（关键考量是 epoch 与过拟合，解法是封顶或模拟）。转写里讲者还把来源具体列为 Wikipedia、common crawl、code、以及从某处抓来的 math data。这里与前后内容的关系很紧密：它把本讲前半段（过滤、去重）与后半段（后训练数据）之间的预训练数据管道收了尾，随后 70:36 起进入听众提问，72:57 起转入 post-training。

易错点有两层。其一是「优化错误目标」的一般教训，讲者在 70:22 把它说成 one lesson here：只要你在优化某个量，就有 optimizing the wrong thing 的危险；回归式混合优化的是代理尺度上的 loss，而真正想要的是目标尺度上的模型质量。其二是外推可靠性：模拟 epoch 只能对齐 epoch 结构，无法对齐所有尺度相关因素（比如长训练下低质量数据的可接受度变化），所以它是缓解而非根治；封顶 epoch 与模拟 epoch 也常被同时使用。下一章的问答正好会追问模拟 epoch 自身带来的新风险——下采样是否会把数据压得太小。

## 两处追问：下采样过小与域内混合网格 [70:36 - 72:57]

第一位听众的问题围绕 dev sample：如果下采样把某个源压得「too small that you can't really generalize as well」，会不会反而损害外推。讲者先把问题复述成 if you're downsampling is there's a risk maybe you get too small of a data set，然后肯定回答「yeah absolutely」。他的推理是：确实可能出现某个源只剩极少的 token（so few tokens）；这时优化器给出的最优解会把非常小的 mass 放在该源上（put very small mass on that），而当规模放大回去时，反而可能得到正确的 token 集合，所以 in principle it should work out。这个回答与上一章的 $\text{ratio}=0.01$ 直接相关：下采样比例是全局施加的，小源被乘上 ratio 后可能小到失去统计意义，但优化器会通过压低权重来「自我修正」。

真正的隐患是 rounding errors（舍入误差）：讲者举例，本来应该在某个数据集上训练一次（training once），舍入后可能意外变成训练零次（train like zero times by accident）。用上一章的记号，就是 $\text{epochs}(s) = p(s)\,T_{\text{small}} / N_s^{\text{down}}$ 在 $N_s^{\text{down}} = 0.01 \cdot N_s$ 很小、$p(s)$ 又被压得很低时，数值上落到 0 附近。他给出的 workaround 很朴素：just like you always train once on this data，即强制每个源至少被训练一次，用下限兜住舍入。这与前面 OlmixBase 在表里设 Data repetition constraints = Yes 是同一类「加约束」的思路，只不过一个约束上限、一个保证下限。易错点在于：这种兜底是工程上的补丁，改变了优化问题的可行域，因此代理尺度上的最优未必还能干净地映射回目标尺度；讲者用 in principle、maybe 这类措辞，也说明他没有把它当成有保证的结论。

第二位听众问的是粒度：这些来源恰好对应不同 topic，但会不会在一个 diverse data set 内部也做 data mixing，而不只是在来源之间做。讲者明显欢迎这个问题（That's actually I forgot to mention that. So I'm glad you brought it up），并以 Neotron 论文以及 AI2 的相关工作为例展开。做法是：即便只给你一份 Common Crawl，也可以先把它拆开——用 AI2 的 web organizer 按 domain（域）分组、按 topic 归类，同时再叠加 quality filter（质量过滤）；于是得到一张 two-dimensional grid（二维网格），一维是 domains，一维是 quality，而 each of these cells is something that you would data mix，即每个格子 $(d, q)$ 都作为一个独立的混合单元参与配比。这相当于把「来源」这个粗粒度概念细化成 $(\text{domain},\ \text{quality})$ 的笛卡尔积，让回归式混合的自变量从寥寥几个来源扩展成网格里的许多单元。讲者把这一步称为 one automatic way to determine the domains（自动确定域的一种方式），随后再 on top of that add extra sources that people hand you，把人工交付的额外来源叠加到自动网格之上。

从前后关系看，这一问把第 1 章表格里 $m$（number of domains）的含义落到了实处：$m$ 不必是手工指定的七八个来源，也可以由 web organizer 自动切出几十上百个 $(\text{domain},\ \text{quality})$ 单元，而表格里 512 ($m=17$)、112 ($m=21$)、$3(m+1)$ 这类 swarm size 设计正是为不同 $m$ 准备的。同时也埋下局限：网格切得越细，每个 cell 的 token 数越少，就越容易撞上本段反复讨论的小源 epoch 问题与刚刚提到的下采样过小问题；而自动分组与质量过滤本身依赖分类器质量，等于把过滤环节的误差传导进了混合环节。讲者在 72:57 用「I'm going to quickly go through post training data here」结束这一小节，数据混合部分到此收束，下面转入任务相关的后训练数据。

## 后训练数据的通用配方与 OpenThoughts 推理数据集 [72:57 - 76:49]

讲者先划清与前半讲的边界：到目前为止讨论的基本是 pre-training 或 mid-training（预训练/中期训练）数据，它们 generally fairly task agnostic（大体任务无关），目标是 develop basic skills（培养基础能力）；唯一的例外是前面的 reg mix，虽然它在优化 loss，但数据本身仍是任务无关的。而到了 post-training（后训练），数据变得 very task dependent（高度任务相关）。他明确说这一节不做 comprehensive review，只想点出几个近期发布的、有意思的后训练数据集，尤其是 coding（代码）方向，因为「that's of great interest these days」。这一定位很重要：它解释了为什么后文所有例子都围绕 math、science、code 这些可评测、可构造任务展开，而不再谈网页抓取。

画面把这套通用配方写成了一个函数签名与三条编号步骤，代码原文是转写没有逐字念出的部分：

```python
def post_training_data():
    # Recipe:
    # 1. Define a set of environments
    # 2. Define a set of tasks / prompts
    # 3. Collect responses from a strong model (teacher)
```

讲者把第一步的 environments 具体化为「in the case think about code which might be GitHub repos」——对代码而言环境就是 GitHub 仓库；第二步是定义 tasks 或 prompts；第三步是从一个强模型或教师模型（strong model / teacher）收集 responses（回答）。他随后补充了数据来源的现实约束：在开放社区里，almost all the post-training data 里绝大部分是 synthetically generated（合成生成）的；也可以把强模型换成 human（人类标注），但那样 slow and costs a lot of money，几年前前沿实验室不得不 pay a lot of people a lot of money 来拿回答；而如今即便在前沿，也可以做 hybrid human AI things（人类与 AI 混合）。但无论如何，核心不变：there's some teacher out there that's giving you responses。这里的 teacher（教师模型）指提供示范回答的模型，与 student（被训练的模型）相对，是 distillation（蒸馏）语境下的术语。

![OpenThoughts 论文卡片：Data Recipes for Reasoning Models 的作者、日期与摘要](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0055_01-15-03.jpg)

第一个具体例子是 OpenThoughts（幻灯片标注 [Guha+ 2025]）。讲者说明它的动机来自 o1 发布后业界对 reasoning（推理）的关注，主要集中在 math and science；项目最终产出 1.2 million examples（约 120 万条示例），using this teacher model。画面上被高亮的论文卡片给出了更完整的信息：标题 OpenThoughts: Data Recipes for Reasoning Models，作者列有 Etash Guha、Ryan Marten、Sedrick Keh、Negin Raoof、Georgios Smyrnis 等（标注 ... 40 more ...），以及 Jenia Jitsev、Reinhard Heckel、Maheswaran Sathiamoorthy、Alexandros G. Dimakis、Ludwig Schmidt，日期 2025-06-04，链接 https://arxiv.org/abs/2506.04178。摘要里写着：Reasoning models have made rapid progress on many benchmarks involving math, code, and science，但 best training recipes 仍有很多开放问题，因为 state-of-the-art models often rely on proprietary datasets（依赖不公开的专有数据集）；OpenThoughts 项目的目标正是 create open-source datasets for training reasoning models，并提到 OpenThoughts2-1M dataset 催生了 OpenThinker2-32B。这些论文元数据是转写完全没念到的画面独有内容。

关于题目来源，讲义写成 1.2M examples using QwQ-32B as a teacher，以及 Questions come from 27 human and synthetic sources (e.g., StackExchange, NuminaMath, Chemistry)——27 个人类与合成来源，示例包括 StackExchange、NuminaMath、Chemistry；左侧列表还逐项标注了各来源的 Number of Questions（题目数），量级在十万级，例如 StackExchange CodeGolf、OpenRuleReasoning、CodeFeedback 等。讲者的口述与之对应：来源里既有 human sources like stack exchange or num math，也有 synthetic sources；他强调这些只是 the coding ones 的一个子集，另外还有更多数学与化学来源，并提醒 this is a big project that involved a lot of different contributions from many people。他还对来源的真实性做了点评：有些是 coding exercises（编程练习），有些更贴近真实场景；code golf 他判断不算 realistic（realistic 指贴近实际软件开发），而 code review 大概更 realistic。

从前后关系看，这一章把前面「训练数据来自互联网」的图景整个换成了「训练数据来自教师模型对构造任务的回答」，是整讲从 data pipeline 走向 Mid-training + SFT: synthetic data 的转折点。易错点也在这里：既然回答来自 teacher，数据质量就同时受任务设计与教师能力两头制约，讲者后面会明确说 capable models 不等于 good teachers；而 1.2M 这个数字是 examples（示例）而非独立题目，下一章会说明它要除以采样次数才是真实题目数，直接把它当作数据规模会高估多样性。此外，合成数据的代价被转写直白点出——人类标注慢且贵，所以开放社区几乎全用合成，这本身是一种可得性偏差，不是质量结论。

## OpenThoughts 的实验结论与代码 agent 数据：SWE-smith、SWE-zero [76:49 - 81:43]

讲者接着报告 OpenThoughts 对「给定这个数据集，该怎么生成回答」做的 fairly comprehensive analysis，结论在转写里念到、并在屏幕上以 bullet 形式残留（可见 Sampling multiple (16) generations、Better models aren't necessarily ...）。第一，having a few sources was actually good：来源不必贪多，与其把所有来源都用上，不如只用少数几个。第二，sampling multiple generations is helpful like 16：对同一题目多次采样生成回答有帮助，典型次数是 16。第三，having better models aren't necessarily better teachers——更强的模型未必是更好的教师；具体例子是 QwQ 32B（讲者称它 now a very old and small model）作为教师反而优于 DeepSeek R1，而后者在当时 probably one of the strongest open models。第四，basic answer filtering wasn't helpful：简单的答案过滤没有帮助。整条 pipeline 因此被描述成：一批来源进入，随机下采样/复制题目，对每题生成多个回答，汇总成最终数据集。于是 1.2 million 指的是 examples，而 divided by 16 gives you the number of actual questions——真实题目数约为 $1.2\times10^6 / 16 = 75{,}000$ 量级。这个换算是对上一章 1.2M 的重要修正，也说明「多采样」是在用回答冗余换覆盖，而非增加题目多样性。

![SWE-smith 从真实仓库生成合成任务的四阶段流程图](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0057_01-18-35.jpg)

话题随即转向 agentic coding（智能体编程）：讲者说近期更受关注的是 not just a model that can just generate some code but actually a model that can do software development——不止能生成代码片段，而是能做软件开发的模型。SWE-smith（幻灯片标注 [Yang+ 2025]）是第一个例子：给定 repository（仓库），用语言模型自动生成任务。流程图把过程拆成四段，画面文字比转写细得多。Real Repositories 一列列出 Source Code（.env/、README.md、setup/、setup.py）与 Unit Tests（tests/test_api.py、tests/test_utils.py、tests/test_diff... 等）；Environment Creation 一列是 SWE-agent（Attempts to install the repo and execute tests，尝试安装仓库并执行测试）与 Developer（Writes Dockerfile based on SWE-agent work，据其工作写 Dockerfile）；Task Gen. Strategies（任务生成策略）一列给出 Procedural Modification、LM Generated、Combine Bugs、PR Mirroring 四种；New Task Instances 一列则是 Environment Image、Synthetic Tasks、Generated Issue、Bugged Patch、Verified Tests。高亮结论写成 Given a repository, use LM to generate tasks (introduce bugs with LM)，即用 LM 修改代码、引入 bug 来造任务，再做验证（verified）。淡色 bullet 补充规模：128 GitHub repositories yields 50K tasks（128 个仓库产出约 5 万个任务）；讲者口述 50k 在当时（last year）算 quite large，并强调这些是 synthetic tasks。讲义上方的柱状图还把 Code、Math、SWE-smith 等不同来源的题量与最终 1.2M 并列，直观呈现数据规模的层级。

![执行反馈开关下四个模型在 SWE-bench (V)/(M) 上的得分表](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0058_01-20-21.jpg)

下一个观察来自 Nvidia 的 SWE-zero（转写口述作 sweet zero）。讲者指出，与 math 任务不同，SWE 任务有 heavy dependencies（很重的依赖）：most GitHub repos don't even run，得装一堆依赖，而且回滚到某个 PR 时依赖往往 out of date，是 a mess、a nightmare；为成千上万个仓库各配一个 repo specific docker image 是基础设施灾难。关键观察是模型已经足够强，without execution feedback（没有执行反馈）也能解决不少任务。屏幕表格把这一点量化成四行，列头为 Model、Execution、SWE-bench (V)、SWE-bench (M)，每行用 ×（不允许执行）与 ✓（允许执行）对照：MiniMax-M2.5 为 × 69.5 / 57.2 与 ✓ 80.2 / 74.1；Qwen3-Coder-Next 为 × 56.9 / 50.7 与 ✓ 71.3 / 64.3；Qwen3-Coder-480B-A35B-Instruct 为 × 59.4 / 44.3 与 ✓ 69.6 / 54.7；SWE-HERO-32B (Ours) 为 × 57.7 / 42.2 与 ✓ 62.2 / 44.1。这正对应讲者口语的「allowing execution you get like 80, if you don't allow execution you get almost 70」，表格则给出了精确到小数的逐模型数值。高亮句把解释写成 Key: strong models have internal "world model" of code semantics（强模型内部具备代码语义的「世界模型」），比转写里的 internal semantics of code 更进一步。

基于这一观察，他们生成了 300,000 agent trajectories（约 30 万条智能体轨迹），all of these are real GitHub PR，因此 unlike the SWE-Smith examples 更具真实性。实现细节同样密集：使用 OpenHands scaffold（智能体脚手架），并有大量如何 prevent agent hacking（防止智能体作弊）的设计——标准做法是给 agent 一条 explore / test / implement 的指令，而他们限制它不能运行 Python 代码，只能做 set、grep 一类基本操作；讲义淡色部分还写着 remove future git commits to prevent "git hacking" by agent（移除未来的 git commit 防止作弊），以及 Standard OpenHands Setup (Execution-Based) 的 System prompt 与 Problem Solving Workflow（Exploration、Analysis、Testing、Implementation、Verification）。随后他们 distilled from a big coding model（从大代码模型蒸馏）并做 filtered，因为有时编码模型会忽略指令、仍试图执行代码；另有 13k agent trajectories that do require execution feedback（约 1.3 万条需要执行反馈的轨迹）。训练顺序是先在 SWE-zero 这类无执行反馈数据上微调，再在 SWE-HERO 这类需要执行反馈的数据上继续微调；讲者承认 the frontier is still pretty high up，但确实 make some progress。

这一章与前半讲的关系是「环境成本」的极端化：前面过滤、去重、混合都在处理文本，这里第一次出现需要安装依赖、执行测试、防作弊的真实软件环境。易错点集中在三处：一是「更强模型 ≠ 更好教师」（QwQ 32B 优于 DeepSeek R1），说明选 teacher 不能只看榜单；二是 1.2M / 16 与 300K / 13K 这类数字分别是示例数、轨迹数而非独立任务数，混用会严重误判规模；三是无执行反馈路线的可行性建立在「模型已有代码语义世界模型」这一经验观察上（× 与 ✓ 只差约 10 分），它随模型能力变化，并非恒定前提。下一章会把这一路线继续放大到千万级，并收束全讲。

## SWE-rebench、千万级轨迹与全讲总结 [81:43 - 84:45]

讲者用「I'll quickly go through rebench」引出另一条抓 PR 的路线，并概括这类工作的共性：get a bunch of GitHub repos, try to install the repo，most of them probably fail，try harder，然后用语言模型给出 responses。画面主体是 SWE-rebench（标注 [Badertinov+ 2025]），给出了转写没念到的具体规模：21K interactive Python SWE tasks from 3.4K GitHub repositories（从 3400 个仓库得到 2.1 万个交互式 Python SWE 任务）、450K PRs from GitHub and GitHub Archive（45 万个 PR）、Used Qwen 2.5-72B-Instruct to install dependencies and assess PR quality（用 Qwen 2.5-72B-Instruct 安装依赖并评估 PR 质量）。流程图分三列：Preliminary filtering（GitHub Download repositories；GHArcive Retrieve metadata about issues and pull requests；Collection Merge and filter data to extract preliminary tasks）、Environment setup（Validation Try to install the repository and run tests；LLM Hypothesize dependency installation script）、LLM Labeling（LLM Label instances based on specific criteria；Final Dataset 21,000+ samples collected）。这与上一章 SWE-zero 的区别在于：这里仍在努力把环境真正跑起来，而 SWE-zero 直接绕开执行。

![SWE-rebench 的初步过滤、环境搭建与 LLM 标注流程图](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0059_01-22-06.jpg)

回到 SWE-zero 思路的放大：讲者提到一份「just came out today」的工作，把 SWE-zero 扩展到 12 million agent trajectories（1200 万条智能体轨迹），并强调 the nice benefit of SWE-zero is that it's very lightweight（非常轻量）。它使用 SWE-rebench-v2 的任务；讲义淡色后续页把构成写成 32K executable tasks + 120K non-executable tasks（3.2 万个可执行任务与 12 万个不可执行任务）。讲者的口语版本是「they were trying to get things to execute, they only got 32,000 of them to execute」，随即点出 but zero doesn't care, you can use all of them——SWE-zero 不依赖执行反馈，因此可执行与不可执行的任务全都能纳入，这正是轻量的来源。他还补了一句 this is a very small model because now this data set is quite large，指模型很小而数据集很大，与前面动辄数百 B 的教师模型形成对照。规模演进因此有了清晰的数字链：SWE-smith 的 50K 合成任务 → SWE-zero 的 300K 真实 PR 轨迹 → 12M 轨迹，跨度接近三个数量级。

![全讲总结页：数据管道代码与 Summary 六条 bullet](/coursenotes/cs336-language-modeling-sp26/l14-data-pipelines/kf_0060_01-24-20.jpg)

讲者随后对后训练数据做收束：data sets are getting more and more sophisticated，从 environment free things like math 走到 coding，且 coding 数据集还在 grow quite a bit。他把设计空间归纳成 prompt 来源上的三类 trade-offs：fully synthetic（全合成）、semi-synthetic（真实环境 + 合成任务）、real（真实任务）；而 responses generally are coming from just capable models but they need also need to be good teachers——回答来自有能力的模型，但「有能力」不等于「好教师」，这正是第 6 章 QwQ 32B 例子的回响。他最后坦承 Code environments are a pain，且有 a lot of filtering and other details which we don't have time for。全讲总结页则用代码把四个环节并排写出，原文是：

```python
# Data pipeline
transformation()
filtering()
deduplication()
data_mixing()

# Post-training data
post_training_data()
```

页面上方的 This lecture 两条 bullet 是 Data pipeline: transformation, filtering, deduplication, mixing 与 Mid-training + SFT: synthetic data；下方 Summary 六条依次是 Filtering: train classifier (language id, quality, toxicity) for what good looks like（定义「好」的样子、训练分类器）；Deduplication: hashing scales to large datasets for fuzzy matching（哈希使模糊匹配可扩展到大数据集）；Mixing: try mixtures at small scale, extrapolate to optimal mixture and large scale（小规模试混合再外推）；Applications: language identification, quality filtering, toxicity filtering；Post-training data: looks like evaluations, use of synthetic data；以及被高亮的 A lot of data work is domain-specific, looking at examples, etc.。转写里的总结与之对应：define what good looks like 再 train a lightweight classifier 去筛 web crawl；deduplication 用于 avoid overfitting and saving flops；mixing 是 try mixture at small scales, extrapolate to large scale；最后看了一些 post-training data。

讲者收尾时特意给这堂课加了一条边界：a lot of the data work can be very grungy（很琐碎脏乱）、very domain specific，且 requires looking at concrete examples to make these high quality data sets，所以 this lecture is not really representative what data work is like，只希望给出 the data landscape（数据领域的地形图）。这句自评与本段开头「优化时可能 optimizing the wrong thing」的警告首尾呼应：本讲给出的是可迁移的框架与算法（过滤、MinHash/LSH、回归式混合、合成后训练），而真正决定数据质量的往往是具体样例上的领域判断。易错点也因此明确：把框架性结论当作现成配方直接套用、忽视 epoch 与尺度依赖、把 capable model 当作 good teacher、把示例数当作任务数，都是这堂课反复提醒的坑。讲者以「Okay, that will be it for today」结束，时间落在 84:35 左右。

## 要点回顾

### 数据管道与转换

- `Lecture 14: Data II` 的 `def main():` 把全课拆成 `transformation()`、`filtering()`、`deduplication()`、`data_mixing()`，承接上节课 `Live service → dump/crawl → processed data` 及 terms of service、copyright 考量。
- `Raw data does not come as text`：原始数据是 HTML、PDF (arxiv) 或 directories；HTML 经 $f_{\text{rule}}$ 线性化压成一维 token 序列，inherently lossy，先 remove boilerplate 再 extract content，导航、表格、图片去留两难。
- 规则工具 `trafilatura, resiliparse, jusText, lynx` 很快而 `[Li+ 2024]` 强调准确率，WET files 的 CORE/EXTENDED 为 20.7/12.2，resiliparse 在 extended DCLM eval 更优。
- PDF 的 `FinePDFs` 一栏点明 `WHY PARSING IS HARD`：源码只有 `BT ... Tj ET` 绘制指令与 `50 738 Td` 绝对坐标，无 `<p>`/`<h1>`、阅读顺序不保证，需 `OCR (RolmOCR)` 借 VLM 或 Docling，截断件要 recrawl。

### 过滤框架、分类器与阈值尺度

- 通用构件 `Given target data T and lots of raw data R, find subset T' of R similar to T`，即求 $T'\subseteq R$；要求从 T 泛化（$T'\neq T$）且 extremely fast，要在约 100 trillion tokens 上跑，最终只留个位数百分比。
- 两步框架先估模型导出评分函数、再按分保留：`score(x)=p_T(x)` 对应 KenLM 5-gram 生成式，`score(x)=p(T|x)` 对应 fastText 线性词袋，用法为 `keep examples x with score(x) >= threshold (stochastically)`。
- 现状对照：C4、Gopher、RefinedWeb、FineWeb、Dolma 刻意不用 model-based filtering，GPT-3、LLaMA、DCLM 使用并 becoming the norm，综述见 [Albalak+ 2024]。
- `No single optimal threshold`：阈值取决于训练 token 数；157M (d512)、N=100 WARC 实验里 dclm 过一个 epoch 后上翘过拟合，resiliparse（4.42B tokens）几乎不过滤却降得更久，且每个点仅单次训练、无置信区间。

### 过滤实例：语言、质量与毒性

- 语言识别用现成 fastText，支持 176 种语言（Wikipedia、Tatoeba、SETimes），`Dolma keeps pages with p(English) >= 0.5 [Soldaini+ 2024]`，但 0.5 只是启发式，语码混用与方言仍留缝。
- OpenMathText [Paster+ 2023] 叠加规则查 LaTeX、`KenLM on ProofPile, perplexity < 15000`、fastText 双阈值 0.17（含 math）/0.8（不含），产出 14.7B tokens，1.4B 模型优于 20x 数据。
- 正例构造：GPT-3 用 Wikipedia、web text、books 训线性分类器；LLaMA/RedPajama 的正例是 pages referenced by Wikipedia 而非维基条目本身；`np.random.pareto(9) > 1 - score` 把随机保留落地。
- phi-1 用 GPT-4 只标 R（The Stack 的 Python 子集）的 100K 子集当 T，再以 random forest 外推，HumanEval 从 12.19%（96K steps）升到 17.68%（36K steps）；Dolma 毒性过滤用 Jigsaw 的 toxic、severe_toxic、obscene、threat、insult、identity_hate 六标签。

### 去重：动机、设计空间与精确去重

- 两类重复是 Exact duplicates（mirror sites、GitHub forks）与 Near duplicates（differ by a few tokens）；C4 audit 出某产品描述重复 61,036 次，MIT license、模板化写作皆典型。
- 动机见 `Deduplication training data makes language models better [Lee+ 2021]`：训练更高效、避免 memorization（版权、隐私），以及 arguably even more important 的 decontamination。
- 设计空间三问 what is an item、how to match、what action；核心挑战是去重需比较 item 之间，朴素 $O(n^2)$ 须降到 $O(n)$，而 filtering 只判单条、天然线性。
- 选 `mmh3.hash("hello")` 的 MurmurHash（快、不抗碰撞，区别 SHA-256）；`sorted(items, key=mmh3.hash)` 配 `itertools.groupby` 与 `next(group)` 做精确去重，high precision 但 `does not deduplicate near duplicates`；C4 [Raffel+ 2019] 用 3-sentence spans 会从文档中间挖掉三句。

### Jaccard、MinHash 与 LSH 分带

- `Jaccard(A,B)=|A∩B|/|A∪B|`，A={1,2,3,4}、B={1,2,3,5} 得 0.6；`two documents are near duplicates if their Jaccard similarity >= threshold`，讲者举例 0.99。
- `def minhash(S, seed): return min(mmh3.hash(x, seed) for x in S)` 让 $\Pr[\min h(A)=\min h(B)]=\operatorname{Jaccard}(A,B)$，特征矩阵中碰撞概率为 3/5；代码用 n=100 个 seed 验证 `abs(estimated_jaccard - jaccard) < 0.01`。
- LSH 把 $n=b\cdot r$ 个哈希分 b 个 band、每带 r 个，规则是 `A and B collide if for some band, all its hash functions return same value`（band 内 AND、band 间 OR），把随机碰撞锐化成阈值。
- 概率为 $P_{\text{collision}}(s)=1-(1-s^{r})^{b}$，`get_prob_collision(sim=0.8, b=5, r=10)` 中 $0.8^{10}\approx0.1074$；真实设置 [Lee+ 2021] 是 `n = 9000, b = 20, r = 450`，相变阈值 `threshold = (1 / b) ** (1 / r)`、中心常数 $1-1/e\approx0.64$。

### 数据混合：分布、epoch 陷阱与回归式配比

- 混合是各来源上的分布：`sources = {"Wikipedia", "CC", "GitHub"}`、`p = {"Wikipedia": 0.3, "CC": 0.5, "GitHub": 0.2}`；基线为 Vibes、Uniform ($p(s)\propto1$)、Proportional ($p(s)\propto\text{num\_tokens}(s)$)，并受多样性（literature、code、papers 不可比）与有限性两个约束夹住。
- The Pile [Gao+ 2020] 给 22 个组件填权重：Pile-CC 227.12 GiB / 18.11% / 1.0 epoch，Wikipedia (en) 仅 6.38 GiB 却给 3.0 epoch，合计 Raw 825.18 GiB、Effective 1254.20 GiB，Effective ≈ Raw × Epochs。
- epoch 陷阱：low 10T、high 10B、训练 1T tokens、$p=\{0.5,0.5\}$ 算出 `low_num_epochs = 0.05`、`high_num_epochs = 50`，50x 重复会过拟合；UniMax [Chung+ 2023] 用 `p(s)*num_training_tokens <= C` 设硬上限，旧做法是 $p(s)\propto\text{num\_tokens}(s)^\alpha,\alpha\in[0,1]$。
- RegMix [Liu+ 2024][Chen+ 2026] 四步：训小规模代理模型（如 300M 的 swarm）、拟合 `Linear Model`/`Tree Model`、模拟新配比预测 target、再训大模型，最优例为 Hacker News 22.8% / Github 67.0% / Philpapers 10.2%（5.34）；两处信念跳跃是回归在最优点仍准确、小规模最优能迁移到大规模。
- 为对齐尺度有 simulate epoching [Held+ 2025]：`ratio = 0.01` 按比例下采样所有来源，把 high 的 0.9 权重回压到 0.3；回归式混合还须小心 downstream evals 过拟合与「优化错目标」。

### 后训练合成数据与全讲收束

- `def post_training_data():` 的配方是定义 environments、定义 tasks/prompts、从 strong model（teacher）收集 responses，开放社区几乎全用 synthetically generated 数据，人类标注 slow and costs a lot of money。
- OpenThoughts [Guha+ 2025] 产出 1.2 million examples，用 QwQ-32B 作教师、题目来自 27 个人类与合成来源；结论是少数来源更好、每题采样 16 次有益、`Better models aren't necessarily better teachers`（QwQ 32B 优于 DeepSeek R1），故 1.2M/16 才约 75,000 道真实题。
- 代码 agent 数据沿 SWE-smith [Yang+ 2025]（128 仓库产 50K 合成任务、用 LM 引入 bug）、SWE-zero（300,000 条真实 PR 轨迹，无执行反馈也可，因强模型有 internal "world model" of code semantics）、SWE-rebench（21K 任务 / 450K PRs）扩到 12M 条轨迹，尺度链跨越三个数量级。
- 全讲 Summary 六条把四个环节收束，并高亮 `A lot of data work is domain-specific, looking at examples, etc.`；讲者自评数据工作 grungy 且领域相关，框架只是 data landscape，别把 capable model 当 good teacher、把示例数当任务数。

## 视频来源

- 原始链接：https://www.youtube.com/watch?v=5sxHosTLPF8
- 课程：Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 14

---

## 对照参考笔记补遗（Spring 2025 版）

> 编者注：本节依据 2025 版参考笔记整理，其中的「更正」小节记录的是核对时发现的问题，对应正文已在本版订正；补遗内容不代表 2026 课程视频原文。


本节内容来自 Qihong Ruan 整理的 Spring 2025 版《CS336 Lecture 14: Data 2》参考笔记，仅用于对照补充 2026 版笔记未展开的过滤算法原理，不代表 2026 课程视频原文，也不据此判定 2026 版笔记有误。

### 从 n-gram 语言模型到 KenLM：极大似然、稀疏计数与平滑

前面讲过滤框架时把「生成式」一路概括成在目标数据上估计一个便宜的语言模型打分。2025 版把这一路拆到了最底层：所谓 n-gram language model，是用极大似然估计（Maximum Likelihood Estimation, MLE）条件概率 $P(w_n \mid w_{n-1}\cdots w_1)$。以 $n=3$ 为例，估计「the cat」之后接「in」的概率就是数一数语料里两个串各出现多少次，即 $P(\text{in}\mid\text{the cat})=\dfrac{\text{count(the cat in)}}{\text{count(the cat)}}$。这个式子极其朴素，实现上「just count and normalize」，正是它足够快、能跑在海量原始语料上的原因。

但 MLE 有一个致命的稀疏计数问题（sparse counts）：当 $n$ 取大一点，绝大多数合理的 n-gram 在有限语料里计数都是 0，于是概率被估成 0，这就是统计语言模型历史上所谓「维数灾难」（curse of dimensionality）。补救办法是 Kneser-Ney smoothing（Kneser-Ney 平滑）：当高阶 n-gram 计数不足时「回退」（back-off）到低阶，让 $P(\text{in}\mid\text{the cat})$ 同时依赖 $P(\text{in}\mid\text{cat})$。KenLM 便是这套 Kneser-Ney n-gram 模型的快速开源实现，最初为机器翻译而做。2025 版给它的总评是「fast but crude」——快而粗糙，恰好呼应本讲为什么要用便宜模型过滤、而不是动用大语言模型。

### 困惑度 perplexity：归一化定义、数值与 n-gram 的语义盲区

KenLM 真正用作质量分的不是概率本身，而是困惑度（perplexity）。设模型对加了首尾标记 `<s> ... </s>` 的文本给出对数概率得分 `score = model.score(content)`，token 数为 $N$，则 $\text{perplexity}=\exp(-\text{score}/N)$。这里按 token 数归一化是关键一步：它避免打分偏向短文档（avoid favoring short documents），否则短文本天然占便宜。困惑度越低，表示这段文本在目标分布下越「可能」、越像训练目标语料。

参考给出三个自洽的示例值：维基百科摘录「Stanford University was founded in 1885 by Leland and Jane Stanford」得分 $-151.73$、困惑度 $187.19$；一段课程网站文本 $-180.81$ / $204.01$；乱码「asdf asdf asdf asdf asdf」$-44.88$ / $272.08$。规律清楚：越规范的百科文本困惑度越低，乱码最高。但这里藏着一个必须记住的易错点：把「the the the...」这类机械重复喂进去，反而会得到很低的困惑度（约 62），因为 n-gram 只刻画局部词序列的搭配是否常见，完全不理解全局语义是否连贯。另需留意，参考材料在该示例上把同一串乱码标成了 62.61 又标成 272.08，代码标签与正文互相矛盾，属解析错位；阅读时只取上面三个自洽数值即可，不要把「重复即合理」当成困惑度的结论。

### CCNet：把困惑度用于大规模语料过滤的经典系统

把困惑度从演示变成工业管线的，是 CCNet（Wenzek 等，2019，*Extracting High Quality Monolingual Datasets from Web Crawl Data*）。做法非常简洁：以段落（paragraph）为处理单位，对网页爬取语料的每个段落算 KenLM 困惑度，按困惑度升序排序，只保留前三分之一（keep the top 1/3）。这与前面 OpenMathText 用绝对阈值 `perplexity < 15000` 属于同一族思路，差别在于一个取相对分位、一个卡固定阈值；前者不依赖阈值标定，后者便于和其它信号叠加。

读这段时要把两条线分清：CCNet 是「段落级困惑度排序」，而 fastText 质量过滤是「p(T|x) 二分类打分」，二者虽同属高质量数据筛选，却是不同的打分机制，不可混为一谈。还需谨慎的是，参考随 CCNet 附了一句「该法用于构建首个 LLaMA 数据集」，但同一份参考在 fastText 章节又把 LLaMA 的质量过滤归到线性分类器，两处归属含糊、且与 2026 版「正例为被维基百科引用的页面」的讲法并不一致；因此这里只采信 CCNet 本身的做法，不把这句因果归属当作确定的事实。

### FastText 的内部架构：从 V×K 基线到 H×(V+K)

2026 版把判别式一路概括为「线性 bag-of-words 分类器」，2025 版则展开了它为什么是线性的。先看一个「并未采用」的基线（bag of words, not what they did）：输入长度 $L=32$、词表 $V=8192$、类别数 $K=64$，直接令 $W=\text{nn.Embedding}(V,K)$、$y=\text{softmax}(\text{mean}(W(x)))$，参数量是 $V\times K$，既庞大又稀疏。fastText 的改动是插入一个远小于 $K$ 的隐藏维 $H=16$：$U=\text{nn.Embedding}(V,H)$、$W=\text{nn.Linear}(H,K)$、$y=\text{softmax}(\text{mean}(W(U(x))))$。

关键是嵌入层与线性层之间**没有非线性**（no non-linearity），所以整个模型本质仍是一个线性分类器，也可视作矩阵分解（matrix factorization），参数量从 $V\times K$ 降到 $H\times(V+K)$。质量过滤只有好坏两类（$K=2$）时，fastText 就是一个线性分类器。实现上采用并行异步 SGD（parallelized, asynchronous SGD），学习率从某个初值线性插值衰减到 0；其经验结论是效果「as good as much slower neural network classifiers」——与慢得多的神经网络分类器相当。这解释了本讲为何反复强调过滤必须用便宜模型：把算力省下来训练真正的语言模型，比用一个强分类器去扫全量网页更划算。

### Hashing trick、实现代价与工程 trade-off

fastText 还把词袋扩展到 n-gram（bag of n-grams），随即撞上另一个问题：bigram 之类组合的数量可以无限增长。解法是 hashing trick（哈希技巧）：固定桶数 `num_bins = 8`（实践中用 10M bins），把每个 n-gram 映射进去，即 $\text{hashed\_x}=[\,\text{hash}(\text{bigram}) \bmod num\_bins\,]$。碰撞（collisions）不可避免，但参考特别点明「the model learns to account for them during optimization」——模型会在训练中自行吸收碰撞带来的噪声，从而在固定内存里容纳近乎无限的 n-gram 词表。

这里有一个最易张冠李戴的易错点：这个 hashing trick 是把特征压进固定桶的特征哈希，而本讲去重章节用的 `mmh3.hash`（MurmurHash）是把整条 item 压成短签名、再靠碰撞找相似文档。两者都叫 hash，目的却完全不同——一个为特征降维、一个为生成去重签名，切勿因同名而互相套用。算力上的 trade-off 也值得并记：过滤阶段当然可以换 BERT、LLaMA 之类更强分类器，但它们更慢，而过滤要扫的是海量原始数据 $R$，单条推理成本哪怕略增也会被样本数放大成巨大开销，所以「用强模型过滤」往往不如「把这份算力用于训练」。另外，参考此处把 `H = K - 2` 与 `H = K = 2` 写成了两个互相冲突的版本，前者显然是解析乱码，以 $H=K=2$ 的自洽表述为准。

### DSIR：重要性重采样作为数据选择的第三条路

2025 版把过滤实现明确列为三种（KenLM / FastText / DSIR），第三种是 DSIR（Data Selection for Language Models via Importance Resampling，Xie 等，2023），2026 版已不讲它，属版本差异。它既不训生成式语言模型、也不训判别式分类器，而是用重要性重采样（importance resampling）选数据：先用 raw 与 target 数据估重要性权重（一个简单 bag-of-n-grams estimator），再按权重从原始语料重采样出目标子集。这里的目标分布 $p$ 与提议分布 $q$ 正对应过滤框架里的 $T$ 与 $R$：设 $p=[0.1,0.2,0.3,0.4]$、$q=[0.4,0.3,0.2,0.1]$、词表 $[0,1,2,3]$，先按 $q$ 抽 $N=100$ 个样本，再算权重 $w\propto p/q$（即 $w=[p[x]/q[x]]$）、$z=\sum w$，按归一化后的权重重采样，便把 $q$ 的样本「掰」成近似 $p$ 的分布。

DSIR 的设定是目标集 $D_p$ 很小、提议（原始）集 $D_q$ 很大。Take 1 想分别把 $p$ 拟合到 $D_p$、$q$ 拟合到 $D_q$ 再做重采样，立刻遇到「target data $D_p$ is too small to estimate a good model」——目标数据太少，估不出像样的分布；Take 2 于是改用 hashed n-grams，把「the cat in the hat」这类文本的 n-gram 哈希成稳健的特征估计，绕开数据稀疏。把它与前面两类分类器并排看更有意思：三者都落在同一个 $T/R/T'$ 构件和「估模型 → 打分 → 按分保留」两步框架里，只是 DSIR 打分靠密度比 $p/q$ 而非 $p_T(x)$ 或 $p(T\mid x)$。最后补一个术语：参考把 Bloom Filter（布隆过滤器）列为精确去重（exact deduplication）的经典工具，与近似去重的 MinHash / LSH 并列，共同说明「hashing 是去重的基础」；我们精确去重用的是排序加哈希分组，未点这个经典结构名，可作为一个术语补丁记下——不过参考正文在 DSIR 处即被截断、该词仅存于提纲，故此处也只点名，不臆造其实现细节。
