# PointNet Deep Learning on Point Sets for 3D Classification and Segmentation

## Intro

本文的主要工作：

- 从输入的三维点云中至今进行物体的分割，识别； <u>传统的方法</u>是将点云进行三维网格化/转化为2D的multi-view的图像；
- 本文也需要对输入的图像做一定的处理（对称化等）

## Related work

- Point Cloud Features: 认为手动根据人眼敏感的信息进行特征的设计，当camera提取到这些特征的时候就进行自动的识别；

- Deep learning on 3D Date: 

  - Volumetric CNNs: 将点云划分为3D的网格，计算量大，由于网格稀疏，表示性能差；
  - FPNN and Vote3D: 难以处理large-scale的输入；
  - Multiview-CNNs: 将3D场景投射到2D，性能较好，但是很难进行在3D场景中进行high-level的任务（分割，提取，识别）
  - Spectral CNNs: 
  - Feature-based DNNs:

  上面的总结而言，都不是在原始的raw data上进行处理，对于3D的数据输入的表示是有一定的限制的。

## Method

<img src="./PointNet Architecture.png" style="zoom:60%;" />

The architecture of the PointNet.

整个网络的结构中，输入的是3D点云数据，输出的是一个score，标记的是每个点云对于每一个class的score；（是pointwise的）