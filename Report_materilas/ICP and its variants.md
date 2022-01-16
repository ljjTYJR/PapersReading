# ICP and its variants

## Standard ICP

- pros

  

- cons

  - it nevertheless suffers from two key limitations: It requires pre-alignment of the range surfaces to a reasonable starting point and it is not robust to outliers arising either from noise or low surface overlap.

    > Silva, L., Bellon, O. R. P., & Boyer, K. L. (2005). Precision range image registration using a robust surface interpenetration measure and enhanced genetic algorithms. *IEEE transactions on pattern analysis and machine intelligence*, *27*(5), 762-776.

  - as the original algorithm assumes outlier-free data and P being a subset of M; in the sense that each point
    of P has a valid correspondence in M

    > Chetverikov, D., Stepanov, D., & Krsek, P. (2005). Robust Euclidean alignment of 3D point sets: the trimmed iterative closest point algorithm. *Image and vision computing*, *23*(3), 299-309.



---

- 点云配准的应用
  - This problem has been considered in 3D model acquisition (reverse engineering, scene reconstruction) and motion
    analysis, including model-based tracking.