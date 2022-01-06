# Iterative Point Matching for Registration of Free-Form Curves and Surfaces

- **Methods**
  - we exploit a robust technique to discard several of them by analyzing the statistics of the distances
- **优点**
  - 并不是point-to-point的进行配准，而是通过数据的操作去除一些异常值；
  - 对于outlier更加鲁棒；
  - 点之间的配准的correspondence更好，收敛速度更快；

---

## Some citations

- > The first class uses robust estimators [**19**,11] to deal with outliers (i.e. erroneous or occulted points). However, these techniques are not designed to be robust w.r.t. the initial transformation, which is our main point
  > of interest in this paper.
  >
  > —— Multi-scale EM-ICP: A Fast and Robust Approach for Surface Registration
  
- > Zhang[5] almost simultaneously describes ICP, but adds a robust method of outlier rejection in the correspondence selection phase of the algorithm.