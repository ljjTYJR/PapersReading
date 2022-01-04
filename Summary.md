# Temp Summary

- 语义(Semantic)在地图中的作用：

1. 用于点云配准， NDT/ICP都需要一个比较好的初始条件，通过语义进行配准，可以获得一个较好的初值条件；
2. 对于SLAM中的回环检测更好(loop closure)，与传统的通过特征进行检测不同，通过语义可以检测得更加精准，获得更好的准确性(没有语义它是怎么做的？)
3. For higher-level task，例如路径规划，更高级的任务；
4. Dynamics 对于动态环境，能够进行物体识别，将点云中动态物体识别出来，不进行地图的构建；

有什么可以创新的地方：

1. 和点云配准算法进行结合（点云配准是一个非常基础的工作，也是一个 narrow direction，这一部分可以顺着FUZZY的工作看一下能够继续挖）
   1. 能否添加semantic的相关信息？
   2. 稀疏/稠密的点云地图，Lidar和RGB-D？能否进行加速等？
2. 语义的提取也可以，是从RGB CNN 网络中进行提取，还是直接从三维点云中进行label?
3. 融入动态的进去?

---

- point cloud 中 semantic 获取的主要方式：

  - PointNet / PointNet++ : 直接从3D点云中进行语义标注/分割，Pixel级别；

  - SemanticFusion: RGB-D输入，利用RGB的2D image 作为输入，同样是pixel-wise的标注；

    上述的两个网络都需要对网络的object进行训练；2D输入的缺点：根据[1]，是计算量比较大，无法处理dense point cloud的输入；

---

无语义的时候，SLAM进行 loop closure的方式：

