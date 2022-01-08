# Efficient variants of the ICP algorithm

这篇文章主要对基于ICP方法的变种进行了实验和分析，分析的重点在于如何提升 ICP-based 的算法的收敛速度。

- ICP算法可以基本总结为两个步骤：

  1. 选取具有correspondence的点；

  2. 构造objective function描述这种correspondence并进行优化。

- standard ICP的方法主要有两个缺点：
  1. 对初值敏感；
  2. 对noise/outliers的鲁棒性差；

- 对于ICP算法的两个步骤可以进行改进的地方有：

  - 如何寻找correspondence:

    - closest
    - projection
    - 使用color信息

    未使用color：与形状有比较密切的关系

  - 如何剔除无效点：

    - 对distance设定阈值
    - 剔除最大的10%
    - 用距离的标准差
    - 用数据构造（例如所有点集中最大值的0.5?)
    - 去除边界点

    —— 可能影响鲁棒性，但是无法进行加速。其中去除边界点可以很好处理 partial overlapping 的情况。

  - 如何构造objective function

    - 2范数，即点距的平方和 —— 可以通过SVD求解析解
    - 将其转化为一般的非凸优化问题，通过数值解法求解
    - point-to-plane

    —— point-to-plane的效果更好

  - 对点对进行reweighting

    - 使用常数大小
    - 用点对之间的法线 $n_1 \cdot n_2$

    —— 影响较小

  - 如何对点进行采样

    - 使用全部点
    - 随机采样
    - 均匀采样
    - **在法向量空间/角度空间进行采样(choosing points such that the distribution of normals among selected points is as large as possible)** (本文创新)

    —— 1. 根据法向量进行采样的鲁棒性更好； 2. 从两个mesh分别采样效果更好