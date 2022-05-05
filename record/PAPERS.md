- **TEASER**

  总结来说，本文主要解决的是点云配准的问题。基于correspondence, 有点类似于RANSAC。但是RANSAC只适用于没有outlier的情况，本文主要是针对该问题进行求解。

  本文借鉴了一些其他文章的方法，比如一些旋转/平移不变的量作为输入，将transformation进行解耦，分成 translation, rotation 和 scale。

  总的来说，对于每一个解耦的在求解时，都是解一个TLS(截断最小二乘)的问题。不同点在于：对于 *scale* 和 *translation* 的求解比较方便，但是对于 *rotation* 来说，其是一个非凸的问题，是难以进行求解的，如何将凸问题转化为非凸是问题的关键。

  > 本文作者并没有回避一个比较难的问题，而是沿着一个路一直走下去。个人感觉这需要强大的知识储备去解决一个问题。
  >
  > 包括，出于什么motivation去解决这样一个问题；
  >
  > 现有问题的方法，从什么角度考虑去接下来的工作；
  >
  > 难点在于什么地方，以及如何考虑去解决这些难题；
  >
  > 接下来如何进一步开展工作；

  **对于scale的估计**

  估计 *scale* 时，目的是解一个 在outliers 存在的时候对参数进行估计。对应的 *objective function* 最小化；因为objective function 是非凸的(因为truncated的缘故，因而需要进行分片进行求解)

  基本的思想是找到一个最大的 *consensus set*，不同点在于 *maximum consensus set* 地目的是追求包含最大的 *consensus set*, 但是 *adaptive voting* 的目的在于最小化 *objective function*. 

  另外一个问题就是 $s_{ij}$ 是如何构造的？
  $$
  \begin{align*}
  b_{ij} &= sa_{ij} + o_{ij}
  \\
  \Vert b_{ij} \Vert &= \Vert sRa_{ij} + o_{ij}\Vert
  \\ &= \Vert sRa_{ij} \Vert+ \hat{o_{ij}}
  \end{align*}
  $$
  当两边同时除以 $\Vert Ra_{ij}\Vert$ 的时候，有：
  $$
  \Vert R a_{ij} \Vert = \sqrt{\Vert R a_{ij}\Vert_{2}^{2}} = \sqrt{(Ra_{ij})^{T}(Ra_{ij})} = \sqrt{a_{ij}^TR^TRa_{ij}} = \sqrt{a_{ij}^Ta_{ij}} = \Vert a_{ij} \Vert
  $$
  具体到 *adaptive voting* 的算法来说，其首先将所有的 $s_{ij}$ 在实轴上划分为有限个区间。可以肯定的是，我们要估计的 $s$ 的值一定在这些区间内的某一点，我们暂且假设为各个区间的 *中点*。（总体的思想是通过先 *猜测* ，再进行验证的方式来进行估计），然后对中点进行验证。获取中点后然后进行验证其在哪一个区间内，如果在区间内，就将其加入到 *consensus set*。

  接下来有两种方式：

  1. 选取 *consensus set* 最大的团作为 $s$ 的估计；

  2. 因为 *outliers* 的存在，不能直接选取 *consensus* 最大的团，而是先在 *consensus set* 中估计 $\hat{s}$， 然后将其代入函数进行验证。计算的方式可以看成是：
     $$
     \frac{\sum_{1}^{K}m_{k}s_{k}}{\sum_{1}^{K}m_{k}}
     $$
     这样不同于 *consensus set* 的地方在于，可以有效地去除 *outliers*