# TEASER-Fast and Certifiable Point Cloud

## Pre-Knowledge

- **Semidefinite programming** (半定规划)

  > [[wiki]](https://en.wikipedia.org/wiki/Semidefinite_programming)

- **quaternion**(四元数)

- **SVD**(奇异值分解)

## Related work

- **Correspondence functions**

  - detect and match key points using feature descriptors.

  - sensitive to the noise and outliers.

  - **Without outliers**

    - Closed-form solutions for single transformation, rotation and scale registration without noise.

      > read related papers :flags:

  - Robust Registration

    - RANSAC: efficiency without noise but sensitive to noise.

- **Simultaneous Pose and Correspondence Methods**

  - Local Methods
    - ICP
  - Global Methods
  - Deep Learning Methods

## Reference

- **Closed-form solution of absolute orientation using unit quaternions** [[URL]](http://people.csail.mit.edu/bkph/papers/Absolute_Orthonormal.pdf)
- **Least-Squares Fitting of Two 3-D Point Sets**[[URL]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4767965)

> correspondence:  Given two point set $p$ and $p'$, the correspondence means :
> $$
> p_i' = Rp_i + T + N_i
> $$
> 点集分别为$p'$ 和 $p$， 上式表示的就是correspodence；其中$N_i$ 表示的是噪声；对于已知correspodence的两个点集，其配准的就诊R和T是可以求解析解的。（在没有$N_i$ 的情况下）
>
> 总结而言，对于两个点集进行配准，如果知道**准确的correspodence**（上式的表示）是可以知道配准的解析解的，（即所谓的 **ground truth**)

其实还可以进一步的抽象化的是：对于点云配准问题，就是求R（旋转矩阵）和T（平移）来使得两个点集的overlapping部分尽可能地重合。而如何求R&T是配准问题的关键。不同方法的核心在于 $objective function$ 的不同。



但是在实际的工程应用中，Lidar采集到的数据有很大的噪声。这个时候就很难知道 *ground truth* ，想要进行配准需要使用的算法：

- **A method for registration of 3-D shapes** (ICP)

  > key idea  If the correct correspondences are known, the correct relative rotation/translation can be calculated in closed form.
  >
  > [[Wolfram Burgard, et al. Introduction to Mobile Robotics: Iterative Closest Point Algorithm]](https://link.zhihu.com/?target=http%3A//ais.informatik.uni-freiburg.de/teaching/ss11/robotics/slides/17-icp.pdf) :flags:

  ICP算法的 *motivation* 其实也来自于上面的 *Least-Squares Fitting of Two 3-D Point Sets*， 对于correspondence已知的情况下，是可以求出解析解的，但是对于correspodence未知的情况下呢（这种情况下是无法求出解析解的）

  —— 一个朴素的想法就是找到这个correspodence，那么假定的correspodence是什么样的呢？

  1. correspondence 是怎么获取的—— 对于mobile robot来说，扫描同一个场景的时候，摄像机知道自己的lidar的位姿的变化，根据这个变化其实是将扫描得到的scan的坐标进行了变化，使得两个场景有一个点的correspodence.

     而closet的含义其实就是赋予这个correspodence，也就是说选取两个点云之间的最近点作为correspodence；

     iterative的含义是，上述选取的correspodence显然是不正确的，因此需要进行反复的迭代，直到收敛；

  2. 上述的过程中其实可以看到ICP算法的局限性：

     - effiency比较低，需要的计算量很大；
     - Roubtness比较低，对noise抵抗性较弱；
     - Accuracy需要比较好的initialization;

- **A Globally Optimal Solution to 3D ICP Point-Set Registration**（GO-ICP)

  

- **Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography ** (RANSAC)

  > [[URL]](http://www.cs.ait.ac.th/~mdailey/cvreadings/Fischler-RANSAC.pdf)
  > [[Wiki]](https://zh.wikipedia.org/wiki/%E9%9A%A8%E6%A9%9F%E6%8A%BD%E6%A8%A3%E4%B8%80%E8%87%B4)

  简而言之，RANSAC算法是这样的算法：其作用是为了从带有噪声(outliers)的数据中，估算出一个比较好的模型。基本的想法是，会搜索符合输入数据的一系列模型，如果某一个模型含有的点最多，则将其认为是描述数据的模型。

  该方法如果用在点云配准上面，主要的应用是这个样子：首先估计一系列的correspodence，这些correspodence有inliers，也有outliers，但是会寻求找到一个 exact correspodence来描述两个点云的配准关系，只要找了了这两个点云的correspodence（精确）那么就仍然可以采用SVD的方法来求解两个点云配准的解析解。

  

- https://zhuanlan.zhihu.com/p/431903717

  可能有用，有空看看