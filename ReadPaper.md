> Record for the papers I have read and want to read

# PSR(Point Set Registration)

## ICP & Variants

- **A Method for Registration of 3-D Shapes**

  > <TPAMI>
  >
  > [[url]](https://www.cvl.iis.u-tokyo.ac.jp/class2004/wedenesday/report/besl.pdf) [[pdf]](./papers/A Method for Registration of 3-D Shapes.pdf)

  The classical approach for PSR: ICP.

- **Go-ICP_A_Globally_Optimal_Solution_to_3D_ICP_Point-Set_Registration**

  > <TPAMI-2016>
  > [[pdf]](./papers/Go-ICP_A_Globally_Optimal_Solution_to_3D_ICP_Point-Set_Registration.pdf) [[url]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7368945) 
  
  Use BnBs approach to get the global optimization(Deal with the R and T separately).
  
  - *global optimal*
  
  - <u>slow & large memory consuming</u>
  
  :question: The details of BnB searching process.

## Probabilistic Based

- **GMM**

  - **Robust Point Set Registration Using Gaussian Mixture Models**

    > <TPAMI2011>
    >
    > [[URL]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5674050)

  - **Point Set Registration: Coherent Point Drift** (CPD)

    > <TPAMI2010>
    >
    > [[URL]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5432191) [[PDF]](./papers/Point_Set_Registration_Coherent_Point_Drift.pdf)
    >
    > [[notes]](./notes/Point Set Registration - Coherent Point Drift.md)

- **NDT & Variants**

  - **The Normal Distributions Transform: A New Approach to Laser Scan Matching**

    > <RSJ2003>
    >
    > [[pdf]](./papers/The_normal_distributions_transform_a_new_approach_to_laser_scan_matching.pdf) [[url]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1249285)


  - **Scan Registration for Autonomous Mining Vehicles Using 3D-NDT**

    > <JFR2007>
    > [[pdf]](./papers/Scan registration for autonomous mining vehicles using 3D‐NDT.pdf) [[url]](https://onlinelibrary.wiley.com/doi/epdf/10.1002/rob.20204)


## Fuzzy

- **Point Set Registration for 3D Range Scans Using Fuzzy Cluster-based Metric and Efficient Global Optimization**

  > <TPAMI>
  >
  > [[pdf]](./papers/Point_Set_Registration_for_3D_Range_Scans_Using_Fuzzy_Cluster-Based_Metric_and_Efficient_Global_Optimization.pdf)[[url]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9026868)
  >
  > [[notes]](./notes/Point Set Registration for 3D Range Scans Using Fuzzy Cluster-based Metric and Efficient Global Optimization.md)

  Fuzzy-clustering for point set registration.

  The approach can be divided into two steps: 1. coarse registration: fuzzy clustering and use BnB to search globally. 2. fine registration: use an algorithm based on the gradient to get a local convergence. 
  
- **FuzzyPSReg_Strategies_of_Fuzzy_Cluster-Based_Point_Set_Registration**

  > <TRO>
  >
  > [[URL]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9623345) [[PDF]](./papers/FuzzyPSReg_Strategies_of_Fuzzy_Cluster-Based_Point_Set_Registration.pdf)
  >
  > [[notes]](./notes/FuzzyPSReg_Strategies_of_Fuzzy_Cluster-Based_Point_Set_Registration.md)

## Semantic

- **SegMap Segment-based mapping and localization using data-driven descriptors**

  > <IJRR>
  > [[pdf]](./papers/SegMap Segment-based mapping and localization using data-driven descriptors.pdf)[[url]](https://journals.sagepub.com/doi/pdf/10.1177/0278364919863090)

- **Recurrent-OctoMap: Learning State-Based Map Refinement for Long-Term Semantic Mapping With 3-D-Lidar Data**

  > <RA-L>

- **Sattler_Understanding_the_Limitations_of_CNN-Based_Absolute_Camera_Pose_Regression_CVPR_2019_paper**

  > <CVPR>

- **Self-Supervised_Learning_of_Lidar_Segmentation_for_Autonomous_Indoor_Navigation**

  > <ICRA>

- **Semantic Fusion_Dense_3D_semantic_mapping_with_convolutional_neural_networks**

  > <ICRA 2017>

- **SuMa++: Efficient LiDAR-based Semantic SLAM**

  > <IROS 2019>

- **Integrating Deep Semantic Segmentation into 3D point cloud registration**

  > <RA-L 2018>
  >
  > [[pdf]](./papers/Integrating Deep Semantic Segmentation into 3D point cloud registration.pdf)
  >
  > [[notes]](./notes/Integrating Deep Semantic Segmentation into 3D point cloud registration.md)

  - Use the *PointNet* to learn and pre-predict per-point semantic labels, and then use the output of the Net as the input for the NDT algorithm.

  - Do not need a good initialization.

  - > NDT, PointNet

- **PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation**

  > <CVPR2017>
  > [[pdf]](PointNet_Deep_Learning_CVPR_2017_paper.pdf) [[url]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)
  >
  > [[notes]]()

  PointNet, DL for 3D segmentation.
  
- **Global Optimality for Point Set Registration Using Semidefinite Programming**

  > <CVPR2020>
  > [[url]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Iglesias_Global_Optimality_for_Point_Set_Registration_Using_Semidefinite_Programming_CVPR_2020_paper.pdf) [[pdf]](./papers/Global_Optimality_for_Point_Set_Registration_Using_Semidefinite_Programming_CVPR_2020_paper.pdf)

- **Deep Global Registration**

  > <CVPR2020>
  > [[url]](https://arxiv.org/pdf/2004.11540.pdf) 

## TEASER

- **TEASER: Fast and Certifiable Point Cloud Registration**

> <T-RO>
>
> [[pdf]](./papers/TEASER-Fast and Certifiable Point Cloud.pdf) [[url]](https://sci-hub.ru/https://ieeexplore.ieee.org/abstract/document/9286491/)
>
> [[notes]](./notes/TEASER-Fast and Certifiable Point Cloud.md)

Teaser: Truncated least-squares Estimation And semidefinite Relaxation

## Learning-based Approach

- **PRNet: Self-Supervised Learning for Partial-to-Partial Registration**

  > [[URL]](https://arxiv.org/pdf/1910.12240.pdf) [[pdf]](./papers/PRNet Self-Supervised Learning for Partial-to-Partial Registration.pdf)
  
- **The Perfect Match: 3D Point Cloud Matching with Smoothed Densities**

  > <CVPR2019>
  >
  > [[URL]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Gojcic_The_Perfect_Match_3D_Point_Cloud_Matching_With_Smoothed_Densities_CVPR_2019_paper.pdf)

- **Registration Loss Learning for Deep Probabilistic Point Set Registration**

  > <International Conference on 3D Vision>
  >
  > [[pdf]](./papers/Registration Loss Learning for Deep Probabilistic Point Set Registration.pdf)
  >
  > [[NOTES]](./notes/Registration Loss Learning for Deep Probabilistic Point Set Registration.md)

## Cross-source Combination

## Different kinds of Improvements

- **Efficient Global Point Cloud Registration by Matching Rotation Invariant Features Through Translation Search**

  > [[URL]](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Yinlong_Liu_Efficient_Global_Point_ECCV_2018_paper.pdf) [[pdf]](./papers/Efficient Global Point Cloud Registration by Matching Rotation Invariant Features Through Translation Search)
  >
  > [[notes]](./notes/Efficient Global Point Cloud Registration by Matching Rotation Invariant Features Through Translation Search.md)
  
- **SDRSAC: Semidefinite-Based Randomized Approach for Robust Point Cloud Registration without Correspondences**

  > <CVPR2019>
  >
  > [[URL]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Le_SDRSAC_Semidefinite-Based_Randomized_Approach_for_Robust_Point_Cloud_Registration_Without_CVPR_2019_paper.pdf) [[PDF]](./papers/SDRSAC_Semidefinite-Based_Randomized_Approach_for_Robust_Point_Cloud_Registration_Without_CVPR_2019_paper.pdf)

# Survey

- **A comprehensive survey on point cloud registration**

  > <arXiv>
  > [[url]](https://arxiv.org/pdf/2103.02690.pdf) [[pdf]](./papers/A comprehensive survey on point cloud registration.pdf)
  >
  > [[detailed notes]](./notes/A comprehensive survey on point cloud registration.md)

- **Registration of large-scale terrestrial laser scanner point clouds A review and benchmark**

  > <ISPRS Journal of Photogrammetry and Remote Sensing>
  > [[url]](https://reader.elsevier.com/reader/sd/pii/S0924271620300836?token=601731D7F5A970C99DA0F576524F984B32A54C306453727528A63B21BE9B8B9B81E18ED8BE62C0ECA2F16B6CDC4CE878&originRegion=eu-west-1&originCreation=20211224092123)
  > [[pdf]](./papers/Registration of large-scale terrestrial laser scanner point clouds A reviewand benchmark.pdf)
  > [[detailed notes]](./notes/Registration of large-scale terrestrial laser scanner point clouds A reviewand benchmark.pdf.md)
  

# Math Basis

- **Global optimization through rotation space search**

  > [[URL]](https://link.springer.com/content/pdf/10.1007/s11263-008-0186-9.pdf)