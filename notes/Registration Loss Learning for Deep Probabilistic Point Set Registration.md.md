# Registration Loss Learning for Deep Probabilistic Point Set Registration

## Related work

> 整理一下该论文中对PSR问题的分类

- Classical: ICP and its variants. (correspondence 和 optimization 交替进行)；

- Feature Matching: 先通过一些方法找两个点云之间的correspondence，然后用诸如RANSAC的方法进行匹配；

- Probabilistic: 基于概率模型的方法

  > 基于概率模型其实也是因为无论是ICP还是feature matching，都是为了找寻两个点云之间的特征关系。而采用概率模型的方式去模拟两个点云，可以有更好的鲁棒性和数学工具去处理它。

## Methods

![](./DEEP-GMM.png)

上图可以看作这篇论文的基本框架，我认为可以将处理过程看作是前端和后端。

1. 基于[[A Generative Model for the Joint Registration of Multiple Point Sets]](https://link.springer.com/content/pdf/10.1007%2F978-3-319-10584-0_8.pdf)(大致的含义是对于各个模型的GMM参数和RT变化的参数同时进行优化，将GMM模型的mean值作为cluster的centroid) 。

2. > Specifically, we model the joint distribution of points and features as a Gaussian Mixture Model (GMM), where each component represents the density of the spatial coordinates and features in a local region.

   将点云中的坐标点和深度学习学到的特征一起作为GMM的参数 ( **如何能够将特征作为GMM的参数呢？**)；

3. 使用 **EM** 算法优化GMM的参数和transformation的参数；

   **EM算法具体的实施是什么呢？**