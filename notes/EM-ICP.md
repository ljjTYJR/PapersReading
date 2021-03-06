# EM-ICP

- **方法的简介**
  - 使用最大似然估计来对两个点云之间的配准进行优化；（简化为混合高斯模型）
  - 使用E-M算法进行优化；
  - 使用模拟退火的算法机制来进行收敛；
  - 使用抽取(decimation)机制来加速算法运行；
- **方法的优点**
  - 提升了鲁棒性和收敛速度
- **方法的缺点**
  - 依然需要一个比较好的Initialization

---

## Comments

- Granger and Pennec [33] proposed an algorithm named Multi-scale EM-ICP where an annealing scheme on GMM variance was also used.

  > Go-ICP: A Globally Optimal Solution to 3D ICP Point-Set Registration

- Also, a naive implementation of ICP is know to be prone to outliers which prompted several more robust variations [1, 2]. 

  > A robust algorithm for point set registration using mixture of Gaussians

- To address this problem, we would like to extend our formulation in order to relax the one-to-one correspondence assumption to one-to-many allowing fuzzy correspondences

  > Sparse Iterative Closest Point

- Other works explored the utility of relaxed assignments [14

  > Fast Global Registration

  解释一下什么是松弛问题，松弛对应的名词是”约束“。松弛问题的解就是通过将原问题的比较苛刻的约束条件去掉，然后在一个相对简单的约束条件下进行求解的过程就叫做”松弛“(relaxation)。

  一般来说松弛问题的求解是通过解约束后，对每一个解进行设置一个上确界/下确界，再寻求符合约束条件的最优解。

  
