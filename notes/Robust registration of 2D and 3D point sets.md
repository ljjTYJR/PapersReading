# Robust registration of 2D and 3D point sets

## Intro

- The LM algorithm combines the gradient descent and Gauss–Newton approaches to function minimization.
- This paper abandons one of the basic characteristics of ICP—its closed-form inner loop—and employs instead a standard iterative non-linear optimizer, the Levenberg–Marquardt (LM) algorithm.

本文在ICP的基础上进行修改：

1. ICP其实可以分为两步骤：1)选取correspondence的对应点(closet point, 其实选取点也可以有一些优化)；2)根据SVD的方法直接求解析解；
2. 本文将上述步骤的第二步，不直接通过SVD求解析解，而是通过LM优化的方式进行更新；

- **好处**
  - 目前来看，与传统的ICP方式不同，LM-ICP 采用数值优化的方式对loss-function进行优化，而非SVD的方式。将correspondence和优化的过程同时进行而不是分开进行。大大提高了算法的运行效率。
- **额外补充**
  - robust kernel， 对于outlier的处理；
  - 加速的方法；

---

既然ICP已经可以通过SVD求解析解，那么为什么要使用LM来求一个数值解呢？



## Citations

> Recently, there has been interest in the use of generic non-linear optimization techniques instead of the more specific closed form approaches [9]. These techniques are advantageous in that they allow for more generic minimization functions rather then just the sum of Euclidean distances. [9] uses non-linear optimization with robust statistics to show a wider basin of
> convergence