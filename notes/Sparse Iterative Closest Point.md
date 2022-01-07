# Sparse Iterative Closest Point

将ICP的配准问题转化为一个稀疏$l_p$ 优化问题。

主要提升：

- heuristic-free
- robust
- only one free parameter

## Motivation

- 对于standard ICP方法：

  算法的最终的performance决定于correspondence的选取，而correspondence的选取则取决于：

  - initialization
  - outliers
  - partially overlapping or fully overlapping

- 一些改进方法

  *prune* low-quality correspondence: (参见张正友的文章) 主要是在ICP的过程中对距离非常大的进行剔除

  缺点：1. 难以实现； 2. 参数难以选择

- 本文

  we propose a solution that implicitly models outliers using sparsity.