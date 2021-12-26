# Registration of large-scale terrestrial laser scanner point clouds A review and benchmark

## Obstacles for Point Set Registration

- **Outliers** and **noise** of the data; missed points; variable point density;
- Overlapping point clouds collected from **different viewpoints can result in variable point distributions at the same position**, and thus the registration method must consider view-independent criteria to match features.
- **Limited overlapping** between different scans will lead to insufficient common points between successive scans, so most of the predicted correspondences will be outliers.
- **repetitive structures** increases ambiguity of possible matches.
- **Huge volumes of data** containing billions of points will incur high computational costs that require highly efficient approaches.

## Evaluation of The Techniques

- Effectiveness(Efficiency)
- Robustness
- Reliability(Accuracy)

## Registration Techniques

- ### [:question:]Two groups:

  - Pairwise;
  - Multiview;

  Depending on the amount of the input point clouds.

  ——[[Hierarchical registration of unordered TLS point clouds based on binary shape context descriptor]](Hierarchical registration of unordered TLS point clouds based on binary shape context descriptor)

- ### The Process:

  - Coarse Registration

    > In particular, coarse registration algorithms are first applied to calculate the approximate six degrees of freedom orientation and the translation parameters between adjacent point clouds.

  - Fine Registration

    - NDT and its variants;
    - ICP and its variants;

    > refine the approximate orientation and translation.

### Coarse Registration

#### - Hand-crafted feature-based methods

- **steps**:
  - Extracting geometric characteristics (e.g. points, curves, planes, and surfaces) 
  - Identifying correspondences
- [:question:] What does the 'hand-crafted' mean?
- Limitations:
  - vulnerable to noise and outliers;
  - impractical in unistructural scenes.
- [:question:] Example

#### - [:question:]Four-points congruent set (4PCS)-based registration method 

#### - Probabilistic registration methods

- employ relaxation matching to identify correspondences
- probabilistic registration methods model the distribution of the point clouds as a density function, before optimizing the density function with a correlation-based approach or expectation maximization-based (EM) optimization framework
- **GMMs:**
- **EM-based Methods**:
  - CPD(coherent point drift)[:question:] [pdf]()
- Limitations:
  - large-scale point set;
  - The estimation of the outlier ratio affects the accuracy and efficiency.

#### - Deep learning-based registration methods

- Motivation

  - traditional methods use handcrafted features to distinguish correspondences, and thus they are affected by the experience of their designers and the capacity for parameter adjustment.
  - Deep learning-based methods can **directly learn high-level feature representations from a large volume of data** to achieve good performance in terms of both the descriptive capacity and robustness to variations in the point density and viewpoints

- [:question:]Approaches

  [:question:] Find some papers to read relevant cases.

- Advantages:

  Learn the features automatically, more robust.

- Limitation:

  Impractical in the large-scale scenes.

### Fine Registration

#### - ICP and its variants

- ICP 方法的含义
- ICP方法的局限性
- ICP 的变种，以及他们要解决什么问题；采用的是什么方法；
- Limitation:
  - A good initialization for not trapped in local optimum.
  - Incorrect closest point correspondence can result in local optimum easily.

#### - NDT and its variants

- NDT 方法的步骤
- NDT 存在的问题及其变种的方法；解决的问题；
- Limitation:
  - Dependent on an accurate initial value;
  - Second, a transformed point may cross PDF cells during nonlinear optimization when the estimated transformation is updated, and the transformed point may be located in the wrong corresponding PDF cell during this process[:question:]

> 其实从上面的阅读来看，Registration 的过程都可以分为 Coarse 和 Fine的两个过程，而Fine的两个过程都需要比较好的Initial pose，因此可以在他们之前再加一步Coarse的过程，来自动地获取一个比较好的initial pose.



TODO:

这些方法的基本流程；

典型应用；

兴起的原因

需要补充概率方面的内容（可以一边看论文一边顺便补充）

