# SegMap Segment-based mapping and localization using data-driven descriptors

## Abstract

1. This article mainly presents a kind of Map called SegMap which performs well in localization and map reconstruction.
2. The main useful scene is a large-scale and unconstructed environment.
3. The key idea of kind of approach is that the surroundings of the robot can be decomposed into a set of segments, and each segment can be represented by distinctive, low-dimensional **learning-based descriptors**.
4. In localization, it can improve the accuracy and correct the drift of the open-loop odometry estimate.
5. In mapping, it helps to reconstruct the map and extract the semantic information.

## Introduction

- motivation : 当前的方法(SLAM)大都使用的是**局部的里程计估计**。
- 当前方法的缺陷：
  - **local feature** ：局部的feature准确度不够，计算量比较大；
  - **global scan descriptor**: 动态环境性能差；基于视角，只对旋转有效；
- SegMap is formed **on the basis of partitioning point clouds into sets of descriptive segments** (Dube´
  et al., 2017a)
- Previous work on segment-based localization considered hand-crafted features and provided only a sparse representa-
  tion (Dube´ et al., 2017a; Tinchev et al., 2018). 
-  In this work, we overcome these shortcomings by introducing a novel data-driven segment descriptor that offers
  high retrieval performance.

**contributions**:

- Data-driven 3D segment descriptors;
- A novel technique for reconstructing the environment based on the same compact features used for localization;
- A dataset

##  Related work

1. convolutional neural networks (CNNs) have become the state-of-the-art method for generating learning-based descriptors, owing to their ability to find complex patterns in data.

   > CNN的主要作用是从各种图形中进行特征提取(这些特征可能是人分辨不出来的？)

2. **3DMatch**： extracting features from the point set.

3. semi-handcrafted global descriptor;

4. **autoencoders**: an encoder network compresses the input to a small dimensional representation, and a decoder network attempts to decompress the representation back into the original input.

   > 自编码器：将数据进行压缩，并通过解码器将压缩的数据还原。

   The compressed representation can be used as a descriptor for performing 3D object classification.

5. **Improvement**:本文对自编码器进行了一定的改进，必须认识到，自编码器中的数据压缩和特征提取是一个对数据处理相反的任务。（因为数据压缩的程度越深，那么特征提取就越难）

   > 解决的方案：
   >
   > we combine the advantages of the encoding–decoding architecture of autoencoders with a technique proposed by Parkhi et al. (2015)

## Methods

我主要关注的点目前在于mapping reconstruction上面。

- **The modules of the whole system**

  整个系统分为几个模块：

  - Segmentation
    - Lidar 的输入是3D点云流，范围绕机器人半径为R的一个范围；
    - 基于两种增量式的算法，3D分割被”增量式“地叠加到已有的点云中；（这难道不是配准的过程么？）:question:
  - Description
  - Localization
  - Reconstruction
  - Semantics

- **The introduction of the descriptors**

基本上来讲，这篇文章是首先进行建图。

然后在建图的基础上进行特征提取，将诸如vehicles，buildings等物体进行提取，然后在reconstruction的时候将这些物体进行提出，再进行地图重建。