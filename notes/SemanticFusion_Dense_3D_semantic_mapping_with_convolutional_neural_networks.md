# Semantic  Fusion Dense 3D semantic mapping with convolutional neural networks

## Abstract

- The semantic information can add high-level knowledge in the mapping, which is beneficial to the localization and motion planning.

## Intro

- In this work, we combine the geometric information from a state-of-the-art SLAM system ElasticFusion [26] with recent advances in semantic segmentation using Convolutional Neural Networks (CNNs).
- Our approach is to use the SLAM system to provide correspondences from the 2D frame into a globally consistent
  3D map.

<img src="./pipeline of SegMap.png" alt="The pipeline" style="zoom:80%;" />

## Related work

Compared to the related work, this article mainly addresses these problems:

- Real-time and the computational efficiency is high.
- Using incremental mapping reconstruction can save a global semantic map.
- Using loop closure, be quicker(里程计的含义是根据此来进行图像的校正，理解新的事务要从概念入手，并且越具体越好)；
- Not hand-crafted and by learning from the data-set, the CNNs network can recognize more things.
- incrementally fusing the predictions online and using loop closure can maintain a full map.

## Method

- SLAM的作用：提供correspondence，通过loop closure进行全局地图的建立；
- 输入CNN的图像：RGB或RGB-D的图像