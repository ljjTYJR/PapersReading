# New Knowledge & Math Basis

- **Rodrigues' rotation formula**

  > [[wiki]](https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula)

  在空间中，一个向量 $k$ 绕旋转轴 v 旋转一定的角度 $\theta$，得到的新向量为：

  - 向量形式：
    
    ![](Rodrigues' rotation formula-vector.svg)
  
  - 矩阵形式：
  
    ![](D:\papers\basis\Rodrigues' rotation formula-matrix.svg)

​			—— 其中，$I$ 为单位矩阵;

​			![](D:\papers\basis\Rodrigues-K.svg)

​		—— 最终：
$$
v_{rot} = R v
$$

- **Trace of a Matrix**

  > [[wiki]](https://zh.wikipedia.org/wiki/%E8%B7%A1)

  矩阵的“迹”，其含义是矩阵主对角线的和。

  - 和特征多项式的关系：

    假设矩阵的特征多项式为：
    $$
    P_{A}(\lambda) = (-1)^{n}(\lambda - r_{1})^{\alpha_1}(\lambda-r_2)^{\alpha_2}...(\lambda-r_{k})^{\alpha_k}
    $$
    —— 其中，$r_k$ 表示特征根；$\alpha_k$表示代数重数；

    “迹”与特征根的关系可以表示为：
    $$
    trace(A) = \alpha_1r_1 + \alpha_2r_2+...+\alpha_kr_k
    $$

- **Axis–angle representation**

  > [[wiki]](https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation)

  对于欧几里得空间中的旋转矩阵$R^3$，根据*Rodrigues' rotation formula*, 可以将其转化为一个$(axis, angle)$ 的格式。其中$axis$ 表示的是一个单位轴(长度为1)，$angle$表示的是绕这个轴旋转的角度。

  继续往下看：假设轴的表示为：$axis = [x, y, z]$, 即一个三维的向量（实际上只需要两个变量即可表示，因为：$\sqrt{x^2+y^2+z^2} = 1$；继续表示的话：
  $$
  <axis,angle> = <[x,y,z]^T, \theta>=[x\theta, y\theta, z\theta]^T
  $$
  其在三维空间中的含义为：半径为$|\theta|$ 的球。因为$|axis|$ 的长度为1，再加上角度（假设角度的范围为$-\pi~\pi$)，也就是说构成了一个半径为$\pi$ 的球体。

- **Octree**(八叉树)

  > [[wiki]](https://en.wikipedia.org/wiki/Octree)  [[chinese]](https://zh.wikipedia.org/wiki/%E5%85%AB%E5%8F%89%E6%A0%91)

  八叉树作为一种数据结构，主要用于分割三维立体空间。其中每个分割的节点表示一个子空间。每个子空间可以采用递归的方法，再按照root space的方式来对leaf space进行同样的操作，直到达到要求。

- **Angular Distance**

  > [[wiki]](https://en.wikipedia.org/wiki/Angular_distance)

- **Errors and residuals** (误差和残差)

  误差是观测值与实际值之间的偏差；

  残差是观测值与模型估计值之间的偏差；

- **Best-first search**

  > [[wiki]](https://en.wikipedia.org/wiki/Best-first_search)

- **Branch and Bound** (BnB)

  > 对于非凸函数的全局优化方法

- **Heuristic**(启发式算法)

  > A **heuristic function**, also simply called a **heuristic**, is a [function](https://en.wikipedia.org/wiki/Function_(mathematics)) that ranks alternatives in [search algorithms](https://en.wikipedia.org/wiki/Search_algorithm) at each branching step based on available information to decide which branch to follow. For example, it may approximate the exact solution
  
- **两点表示一个线段**

  给定两个点$p_1$ 和 $p_2$ ，对于线段$p_1 p_2$ 的表示为：$p = p_1 + t(p_2 - p_1)$， 其中 $t \sub [0,1]$, 更为一般的形式为：
  $$
  p = tp_2 + (1-t)p_1, t \sub [0,1]
  $$

- **Semidefinite programming(SDP) 半正定规划**

- **Newton's Algorithm**

  - 用于求函数的零点

  - 通过迭代公式：
    $$
    x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}
    $$
    迭代直到收敛

- **Gradient Descent** **(梯度上升/梯度下降)**

  - 用来求函数的极值

  - 通过找函数在某一点处的梯度，按照一定的步长进行迭代直到收敛:

    