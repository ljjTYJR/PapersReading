# A Method for Registration of 3-D Shapes

- **ICP 方法的小结**
  - 通过closet点集的方法寻找correspondence;
  - 通过SVD的方法在当前correspondence寻找点集$X$ (参考点集)，$P$（sensed data)，之间的变化 $\hat{q}$，以及此时两个点集closet距离 $\hat{d}$.
  - 对当前点集$P_k$, 进行$\hat{q}P_k$ 得到新的点集$P_{k+1}$，并计算$\hat{d_{k+1}}$
  - 收敛条件：$|\hat{d_{k+1}} - \hat{d_k}| < \epsilon$

- **ICP 优点**
  - 不需要correspondence，通过closet即可；
  - 只要是点云数据就可以处理，简单易懂；
- **ICP 缺点**
  - 收敛到局部最优；
  - 对初值敏感，需要一个好的initialization；
  - 对outlier非常敏感，不能有什么noise;