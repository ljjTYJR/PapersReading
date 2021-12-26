# Integrating Deep Semantic Segmentation into 3D point cloud registration

## Introduction

- The PSR problem can be formulated as a optimization problem of minimizing a distance metric between two scans with respect to the 6-DOF transform.
- The NDT/ICP can be used in indoors scenes successfully, but has poor performance in outdoor scenes.
- The article use the *PointNet* as the source of sematic labels, combine the sematic labels with the NDT/ICP approaches to extend these approaches.

## Related work

- Comparison : ICP / DNT
- Fast Point Feature Histograms (FPFH) is also a kind of approach that can provide initial alignment and possible correspondence.(The idea is similar to Semantic)

## Semantic assisted NDT

- The point set is partitioned according to per-point semantic labels.
- The NDT of each label is established and registered independently.
- **Steps**
  1. he point clouds are ==segmented== into disjoint sets according to their labels, $M_n$ being the set
     of points with label n that belong to M.
  2. 对每一个label的point set进行NDT，然后分别进行各块的NDT配准。

## Semantic extraction - PointNet



# TODO

1. pipeline；
2. 实际引用一下；