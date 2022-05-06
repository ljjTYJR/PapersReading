[toc]

# 2022-05-04

- 阅读 survey 文章:  

  > Advances in Inference and Representation for Simultaneous Localization and Mapping

  - Learning to navigate

    Some methods try to learn to navigate end-to-end such as the Ph.D. thesis:  [Building Intelligent Autonomous Navigation Agents](./papers/2106.13415.pdf). It mainly focuses on navigation by learning.

  - Open Questions

    - Sensor Fusion, especially long-term sensor fusion, or online calibration.
    - Hierarchical models, representation of maps.
      - What to learn? Like humans, or has its own learning elements?
      - What to remember? —— Commonly, a lot of memory is needed to be used for navigation, but not all is needed.

    - Where to get semantics?
      - From a single image, a single frame of point cloud, it may be difficult to recover the semantic information.
      - Second, for tasks, it may need different kinds of semantics of properties. But how to organize this kind of semantic information?

  - Future work

    - long-term autonomy: The capacity of a robotic system to operate reliably, for extended periods of time, without human supervision or intervention

      Reliable and failure-aware, and active SLAM which can detect the failure. **Certificate!**

    - Lifelong map learning: The map constructed by the robot may change as time goes by, it is important for a robot to update its map.

      **Estimate the uncertainty of the perception module and choose to active SLAM, or choose to take actions to reduce this kind of uncertainty.**

      **Or spread this kind of uncertainty to the back-end, to facilitate the the optimization**

    - SLAM with deep learning: Combine the traditional estimation algorithms and new deep learning perception algorithms?

- 一个小想法：光看，没有一个prior knowledge的话，似乎印象不深刻，因为读的不是很了解，必须自己掌握了，达到能够解释给别人的程度才可以。否则是比较低效的。
- 具体的这些challange，其实需要和具体有SLAM经验的人进行沟通交流；（Danile & Martin?)

# 2022-05-05

1. GNC: graduated non-convex ; it can be s