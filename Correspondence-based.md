## Correspondence

### Hand-crafted features

### learning-based features

- **RPM-Net: Robust Point Matching using Learned Features**

  > Yew, Zi Jian, and Gim Hee Lee. "Rpm-net: Robust point matching using learned features." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2020.
  >
  > **Citations:** 123
  >
  > [[pdf]](./papers/Yew_RPM-Net_Robust_Point_Matching_Using_Learned_Features_CVPR_2020_paper.pdf)

  - Summary
    - A deep learning-based **RPM** method.

## Optimization

- **Robust L2E Estimation of Transformation for Non-Rigid Registration**

  > Ma, Jiayi, et al. "Robust $ L_ {2} E $ estimation of transformation for non-rigid registration." *IEEE Transactions on Signal Processing* 63.5 (2015): 1115-1129.
  >
  > **Citations:** 267
  >
  > [[pdf]](./papers/Robust_L_2E__Estimation_of_Transformation_for_Non-Rigid_Registration.pdf)

- **A fast automatic method for registration of partially-overlapping range images**

  > Chen, Chu-Song, Yi-Ping Hung, and Jen-Bo Cheng. "A fast automatic method for registration of partially-overlapping range images." *Sixth International Conference on Computer Vision (IEEE Cat. No. 98CH36271)*. IEEE, 1998.
  >
  > **Citations:** 123
  >
  > [[pdf]](./papers/A_fast_automatic_method_for_registration_of_partially-overlapping_range_images.pdf) 
  
- **CLIPPER: A Graph-Theoretic Framework for Robust Data Association**

  > [[pdf]](./papers/CLIPPER_A_Graph-Theoretic_Framework_for_Robust_Data_Association.pdf)
  
- **Fast global registration** :heavy_check_mark:

  > Zhou, Q. Y., Park, J., & Koltun, V. (2016, October). Fast global registration. In *European conference on computer vision* (pp. 766-782). Springer, Cham.
  >
  > **Citations:** 421
  >
  > [[url]](http://vladlen.info/papers/fast-global-registration.pdf)  [[pdf]](./papers/Fast Global Registration.pdf)
  
  - Not the ICP variant
  - Need the correspondence, (FPFH or other)
  - Use an estimator called *scaled Geman-McClure estimator* to reject the outliers and noise
  - Induce the *Black-Rangarajan duality* to optimize the objective function
  - Faster and more accurate than ICP, no need to find the correspondence and closet in the inner loop.

# Other Methods

- **Sampling Network Guided Cross-Entropy Method for Unsupervised Point Cloud Registration** :heavy_check_mark:

  > Jiang, Haobo, et al. "Sampling network guided cross-entropy method for unsupervised point cloud registration." *Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2021.
  >
  > **Citations:** 3
  >
  > [[pdf]](./papers/Jiang_Sampling_Network_Guided_Cross-Entropy_Method_for_Unsupervised_Point_Cloud_Registration_ICCV_2021_paper.pdf)

  - Summary
    - Reinforcement Learning method: The method consists of two modules: 1) Sampling module and 2) Cross-entropy module. The sampling module generates a parameter distribution, and CEM module is used to assess the output from the first module.