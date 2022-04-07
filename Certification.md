# Certification

- **CorAl – Are the point clouds Correctly Aligned?**

  > Adolfsson, Daniel, et al. "CorAl–Are the point clouds Correctly Aligned?." *2021 European Conference on Mobile Robots (ECMR)*. IEEE, 2021.
  >
  > **Citations:** 1
  >
  > [[pdf]](./papers/CorAl__Are_the_point_clouds_Correctly_Aligned.pdf)

  - Summary
    - Differential entropy for separate point cloud and joint cloud(Which in fact the sum of covariance of the points). 
    - Only overlapping points are used in the computation(The definition of overlapping points).
  
- **Efficient Continuous-Time SLAM for 3D Lidar-Based Online Mapping**

  > Droeschel, David, and Sven Behnke. "Efficient continuous-time SLAM for 3D lidar-based online mapping." *2018 IEEE International Conference on Robotics and Automation (ICRA)*. IEEE, 2018.
  >
  > **Citations:** 108
  >
  > [[pdf]](./papers/Efficient_Continuous-Time_SLAM_for_3D_Lidar-Based_Online_Mapping.pdf)

  - Summary

- **Entropy Minimization SLAM Using Stereo Vision**

  > Sáez, Juan Manuel, and Francisco Escolano. "Entropy minimization SLAM using stereo vision." *Proceedings of the 2005 IEEE International Conference on Robotics and Automation*. IEEE, 2005.
  >
  > **Citations:**44

  - It also provides a kind of way based on information theory to measure the quality of point set alignment. It only gives a value, not a certification.
  
- **Learning-based Localizability Estimation for Robust LiDAR** :heavy_check_mark:

  > Nubert, Julian, et al. "Learning-based Localizability Estimation for Robust LiDAR Localization." *arXiv preprint arXiv:2203.05698* (2022).
  >
  > [[pdf]](./papers/2203.05698.pdf)

  - Summary
    - The work is based on deep learning. The proposed method is used to estimate whether a point cloud can be registered well.
      - The training data is generated manually, specifically, the input data first sampled and use ICP to register to get a result. So the inputs are point cloud and an estimated registration error.
      - Then after training, the network can predict whether a point cloud can be registered well or not.
