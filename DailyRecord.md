# 01/24-01/30

本周主要阅读：GMM相关的变种；

每天1.5篇journal/3篇会议

## 01-27

- **SLAM for Dummpies**

- **Exploiting Prior Information in GraphSLAM** 

  > Parsley, Martin P., and Simon J. Julier. "Exploiting prior information in GraphSLAM." *2011 IEEE International Conference on Robotics and Automation*. IEEE, 2011.
  >
  > **Citations:**15
  >
  > [[pdf]](./papers/Exploiting Prior Information in GraphSLAM.pdf)

  - In this paper we presented a method of integrating a prior map into GraphSLAM by forming constraints directly
    between features in the state and prior map.

- **Exploiting building information from publicly available maps in graph-based SLAM**

  > Vysotska, Olga, and Cyrill Stachniss. "Exploiting building information from publicly available maps in graph-based SLAM." *2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*. IEEE, 2016.
  >
  > **Citations:**38
  >
  > [[pdf]](./papers/Exploiting Building Information from Publicly Available Maps in Graph-Based SLAM.pdf)

**Registration:**

- **Registration of Multi-View Point Sets Under the Perspective of Expectation-Maximization** :heavy_check_mark:

  > Zhu, Jihua, et al. "Registration of multi-view point sets under the perspective of expectation-maximization." *IEEE Transactions on Image Processing* 29 (2020): 9176-9189.
  >
  > **Citations:**6
  >
  > [[pdf]](./papers/Registration of Multi-View Point Sets Under the Perspective of Expectation-Maximization.pdf)

  - Multiview point set registration.
  - Based on CPD, for each point in a point set, it is regarded as the point from a GMM whose centroids are the nearest neighbour point in other point sets.
  - :a: Some references are more valuable.
  - :taco: Can be transferred to other algos.

- **Deep closest point: Learning representations for point cloud registration**

  > Wang, Y., & Solomon, J. M. (2019). Deep closest point: Learning representations for point cloud registration. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 3523-3532).
  >
  > **Citations:**285
  >
  > [[url]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Closest_Point_Learning_Representations_for_Point_Cloud_Registration_ICCV_2019_paper.pdf) [[pdf]](./papers/Deep Closest Point Learning Representations for Point Cloud Registration.pdf)

## 01-28

- **Generalized Coherent Point Drift With Multi-Variate Gaussian Distribution and Watson Distribution**

  > Min, Zhe, et al. "Generalized coherent point drift with multi-variate gaussian distribution and watson distribution." *IEEE Robotics and Automation Letters* 6.4 (2021): 6749-6756.
  >
  > **Citations:**1
  >
  > [[pdf]](./papers/Generalized Coherent Point Drift With Multi-Variate Gaussian Distribution and Watson Distribution.pdf)

- **Joint Rigid Registration of Multiple Generalized Point Sets With Hybrid Mixture Models**

  > Min, Zhe, Jiaole Wang, and Max Q-H. Meng. "Joint rigid registration of multiple generalized point sets with hybrid mixture models." *IEEE Transactions on Automation Science and Engineering* 17.1 (2019): 334-347.
  >
  > **Citations:**28
  >
  > [[pdf]](./papers/Joint Rigid Registration of Multiple Generalized Point Sets With Hybrid Mixture Models.pdf)

- **Robust and Accurate Point Set Registration with Generalized Bayesian Coherent Point Drift**

  > Zhang, Ang, et al. "Robust and Accurate Point Set Registration with Generalized Bayesian Coherent Point Drift." *2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*. IEEE.
  >
  > **Citations:**0
  >
  > [[pdf]](./papers/Robust and Accurate Point Set Registration with Generalized Bayesian Coherent Point Drift.pdf)

  - The above three articles look the same, just some parameters are adjusted. For example, the covariance of the Gaussian distribution, the components of the mixed models.... :heavy_check_mark:

- **Density Adaptive Point Set Registration**

  > Lawin, Felix Järemo, et al. "Density adaptive point set registration." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2018.
  >
  > **Citations:**35
  >
  > [[pdf]](./papers/Density Adaptive Point Set Registration.pdf)

- **Quasi-Newton Solver for Robust Non-Rigid Registration** :heavy_check_mark: :no_entry:

  > Yao, Yuxin, et al. "Quasi-newton solver for robust non-rigid registration." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2020.
  >
  > **Citations:**12
  >
  > [[pdf]](./papers/Quasi-Newton Solver for Robust Non-Rigid Registration.pdf)

  - For non-rigid registration
  - Formulate the function by Welsch's function to induce sparsity.

- **A Robust Point Matching Algorithm for Autoradiograph Alignment**

  > Rangarajan, Anand, et al. "A robust point-matching algorithm for autoradiograph alignment." *Medical image analysis* 1.4 (1997): 379-398.
  >
  > **Citations:**232
  >
  > [[pdf]](./papers/A Robust Point Matching Algorithm for Autoradiograph Alignment.pdf)

  - Deterministic annealing

- **4-points congruent sets for robust surface registration**

  > Aiger, Dror, Niloy J. Mitra, and Daniel Cohen-Or. "4-points congruent sets for robust pairwise surface registration." *ACM SIGGRAPH 2008 papers*. 2008. 1-10.
  >
  > **Citations:**690
  >
  > [[pdf]](./papers/4-Points Congruent Sets for Robust Pairwise Surface Registration.pdf)

- **An adaptive data representation for robust point-set registration and merging**

  > Campbell, Dylan, and Lars Petersson. "An adaptive data representation for robust point-set registration and merging." *Proceedings of the IEEE international conference on computer vision*. 2015.
  >
  > **Citations:**60
  >
  > [[pdf]](./papers/An Adaptive Data Representation for Robust Point-Set Registration and Merging.pdf)

- **Acceleration of non-rigid point set registration with downsampling and Gaussian process regression**

  > Hirose, Osamu. "Acceleration of non-rigid point set registration with downsampling and Gaussian process regression." *IEEE Transactions on Pattern Analysis and Machine Intelligence* (2020).
  >
  > **Citations:**3
  >
  > [[pdf]](./papers/Acceleration of Non-Rigid Point Set Registration With Downsampling and Gaussian Process Regression.pdf)

- **Non-rigid point set registration: Coherent point drift** :heavy_check_mark:

  > Myronenko, Andriy, Xubo Song, and Miguel A. Carreira-Perpinán. "Non-rigid point set registration: Coherent point drift." *Advances in neural information processing systems* 19 (2007): 1009.
  >
  > **Citations:**385
  >
  > [[pdf]](./papers/Non-rigid point set registration Coherent Point Drift.pdf)

  - **Non-rigid**, **deterministic annealing**
  - Treat one point set as the GMM centroids, and the other the data set sampled by the GMM. By MLE(maximizing Likelihood function), the solution is solved.
  - Compared to TPS/RPM, the core idea is **CPD**, which can be compared with TPS/RPM when investigating non-rigid registration.
  - To get the wider basin, deterministic annealing is used but will lead to slower converge.

## 01-30

- **New Algorithms for 2D and 3D Point Matching: Pose Estimation and Correspondence**

  > Gold, Steven, et al. "New algorithms for 2D and 3D point matching: pose estimation and correspondence." *Pattern recognition* 31.8 (1998): 1019-1031.
  >
  > **Citations:**656
  >
  > [[pdf]](./papers/)

- **Shape Matching and Object Recognition Using Shape Contexts**

  > Belongie, S., Malik, J., & Puzicha, J. (2002). Shape matching and object recognition using shape contexts. *IEEE transactions on pattern analysis and machine intelligence*, *24*(4), 509-522.
  >
  > **Citations:**8311
  >
  > [[url]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=993558) [[pdf]](./papers/Shape Matching and Object Recognition Using Shape Contexts.pdf)

- **A Bayesian Formulation of Coherent Point Drift** :heavy_check_mark:

  > Hirose, O. (2020). A Bayesian formulation of coherent point drift. *IEEE transactions on pattern analysis and machine intelligence*.
  >
  > **Citations:**26
  >
  > [[url]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8985307) [[pdf]](./papers/A Bayesian Formulation of Coherent Point Drift.pdf)

- **GOGMA: Globally-Optimal Gaussian Mixture Alignment**

  > Campbell, Dylan, and Lars Petersson. "Gogma: Globally-optimal gaussian mixture alignment." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.
  >
  > **Citations:**71
  >
  > [[pdf]](./papers/GOGMA Globally-Optimal Gaussian Mixture Alignment.pdf)