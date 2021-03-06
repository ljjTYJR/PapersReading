2022-04-11~2022-04-17

- LSG-CPD 可以看作是一种 point-to-plane的方式，能否将其扩展成为Generalized-ICP的方式呢？从这个角度来看，有点像是E-M对于GMMReg的变种。

  这其中有两个问题：1. 是否还要用E-M的框架；2. 如何通过协方差来包含局部信息？

  类似于 symmetric ICP, 能否添加两个的输入的 *normal* 的信息？

- Fuzzy clusters 得到的那么多clusters肯定不是全部都是有用的，是否可以过滤掉一些小的，或者赋予他们更低的权重？

  权重这个我感觉没有必要，因为 协方差矩阵对于距离的贡献已经存在了。但是需要研究一下协方差矩阵对距离到底有什么影响？

  能否通过衡量两个协方差矩阵的相似程度，或者除了距离之外，能否用修改一下协方差矩阵，从中添加一个关于两个cluster的相似度的度量。

  > 1. 在公式中两个协方差矩阵相加，其对距离有什么影响？—— 这一部分可以参考G-ICP；
  > 2. 能否在协方差矩阵添加一些项，使其增强对相似的clusters的权重？（包括cluster的大小，形状等）；

- 类似于混合的高斯分布，包含了语义信息和color信息，能否将其融合到聚类中呢？

  也就是说不仅需要包含positional 信息，还需要包含high-level的信息（例如语义）。 如何能够将语义信息添加进去？

- 对初始的位置进行预估: 通过自监督学习的方式对初始的alignment进行预估，从而达到一个错配准的目的？

- 一套的certification的配准估计，达到非常鲁棒的目的：包括：

  - 配准前检测是否能够配准；
  - 进行粗配准的确认；
  - 进行精配准；
  - 配准结果的确认；

  一个框架类型的工作！

- GMM可以套用semantic的方式，那么NDT呢？

  可以参考的文献是：《LCR-SMM: Large Convergence Region Semantic Map Matching Through Expectation Maximization》

  NDT匹配的是最近的grid，可否将其改成匹配最近的多个grid？(采取D2D的方式)。

  如何将semantic的方式加入呢？因为semantic的信息是pixel-wise的，也就是逐点的。可以划分每个grid属于某一类的概率；或者先进行分割，然后选取概率最大的进行配准；

  同时配合SGD进行使用？
  
- 配准过程不使用source 和 target的方式，而是假定是一个模型，其他两个都是由这个模型生成的。即JRMPS的方式，效果是否会更好？

- 似乎还没有找到将color/semantic等信息用到fuzzy的好方法，但是也许可以借助于概率的方式？

---

04-24

- 借鉴于 *Fast Global Registration* 中的 *robust function*， 如何将其放到ICP中，然后进行一个全局优化；或者说如何能够先进行prune?

  prune的方式可以改进么？outlier的prune的方式可以改进么？如何从一个图理论上进行更新呢？