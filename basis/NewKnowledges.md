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

- **Angular Distance**

  > [[wiki]](https://en.wikipedia.org/wiki/Angular_distance)

  