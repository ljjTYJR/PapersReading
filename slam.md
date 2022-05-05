[toc]

# SLAM

## Basis

- **A Tutorial on Graph-Based SLAM**

  > Grisetti, Giorgio, et al. "A tutorial on graph-based SLAM." *IEEE Intelligent Transportation Systems Magazine* 2.4 (2010): 31-43.

  - 整个状态用 *pose-graph* 进行表示，每一个 *node* 表示的是一个 *pose*。 时间T时刻的位姿记为 $x_{T}$. 两个 *pose* 之间的“约束”表示的是两个 *pose* 之间相应的 *transformation*。*transformation* 既可以从机器人本身的里程计进行获取，也可以通过 *lidar* 或者 *visual* 进行获取。任何的节点 *i* 和 *j* 之间都可以用激光雷达/视觉的方式获取 *transformation*。

  - Information Matrix: 衡量不确定性；

  - 对于一个位姿图而言，从一个node $x_i$出发，到另外一个node $x_j$ ，但是实际的观测可能是 $x_{j}'$ , 观测和实际之间的差距 error就是要优化的对象；

  - 后端优化的 cost function 就是要：
    $$
    x^* = argmin_{x} \sum_{ij} e_{ij}^T \Omega_{ij} e_{ij}
    $$

  - 线性化 cost function : 使用泰勒展开将其展开；
  - Relative uncertainty 用于评估 loop closure，检测是否存在 loop closure.

## Robust SLAM

### perception

- **Semantic Localization Considering Uncertainty of Object Recognition**

  > Akai, Naoki, Takatsugu Hirayama, and Hiroshi Murase. "Semantic localization considering uncertainty of object recognition." *IEEE Robotics and Automation Letters* 5.3 (2020): 4384-4391.

### Active SLAM

- **Active SLAM using Connectivity Graphs as Priors**

  > Soragna, Alberto, et al. "Active SLAM using connectivity graphs as priors." *2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*. IEEE, 2019.
  >
  > **Citations:** 5

## Survey

- Advances in Inference and Representation for Simultaneous Localization and Mapping

  > Rosen, David M., et al. "Advances in inference and representation for simultaneous localization and mapping." Annual Review of Control, Robotics, and Autonomous Systems 4 (2021): 215-242.
  >
  > **Citations:** 15
  >
  > [[pdf]](./papers/2103.05041.pdf)

