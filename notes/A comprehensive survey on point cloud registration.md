# **A comprehensive survey on point cloud registration**

## Keywords

- cross-source
- potential directions

## Notes

- The motivation of the survey mainly survey the PSR problems, and divide these problems to 2 classed: optimized-based and deep-learning based.
- Besides the above two classed, the article also investigate the cross-source point set registration. So what does the cross-source mean? For example, the kinect and lidar, these two kinds of point set can combine together to generate a whole 3D scene. The 3D CAD model can also combines with the lidar point set scan for assessment.
- 我会去如何定义一个点云配准问题？
- 深度学习的方法主要有两种：
  - 通过深度学习对点云的features进行提取，然后获取correspondence，并将获取的correspondence传递给传统的方法。关于correspondence对于ICP来说是point-point的，对于GMM来说则是GMM之间的correspondence;
  - 单纯的深度学习，也就是所谓的end-to-end(端到端)的方式，输入两个点云，经过网络之后，直接输出结果（transformation)；
- 