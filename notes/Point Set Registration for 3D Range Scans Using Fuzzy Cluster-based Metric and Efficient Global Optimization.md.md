# Point Set Registration for 3D Range Scans Using Fuzzy Cluster-based Metric and Efficient Global Optimization

## Pre-knowledge

- Fuzzy-clustering:

  > [[wiki]](https://en.wikipedia.org/wiki/Fuzzy_clustering)

  > **Fuzzy clustering** (also referred to as **soft clustering** or **soft \*k\*-means**) is a form of clustering in which each [data point](https://en.wikipedia.org/wiki/Data_point) can belong to more than one cluster.

  - Fuzzy C-means clustering(FCM)

- Expectation-maximization Algorithm(E-M)

  > [[wiki]](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)

- Branch and Bound(BnB)

  > [[wiki]](https://en.wikipedia.org/wiki/Branch_and_bound)

  An algorithm for solving discrete and combinational optimization problem base on search and trimming.

## Methods

1. Applying fuzzy clustering to the two point set. 
2. Minimizing the corresponding fuzzy c-means objective function.

- **Optimization Methodology**
  - Combination of BnBs and gradient-based.
  - **For BnBs**:
    - The search space is just the initial rotation cube $C_r$ and the initial translation cube $C_t$ ; —— The optimal solution is included in the initial search space.
    - Though the initial space for $C_r$ is a ball, to facilitate the BnBs process. Assuming that the ball is enclosed by a cube. —— [:question:] how if the variable is out of the bound of the ball? 