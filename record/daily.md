2022-05-05

今日做的事情：

1. 阅读了关于关于**GNC**的内容，GNC也可以看作是TEASER中部分内容的的抽象，关于“渐进优化”的内容，自己其实也看过类似的，但并没有如此地进行重视。我觉得该文作者的眼光很高，总是能够把握问题的本质，把表面的问题进行抽象成一个简洁的数学问题，这一点是我尤其需要学习的。

   我认为要达成这个目标要养成这样的习惯：1. 加深自己的理论功底（这也和自己最初来读博的初衷一致）2. 对平时遇到的文章要有这样的意识；3. 经常性的回顾和反思之前读过的文章；

   多读一些好文章还是能给自己一些很好的启发的，尤其要读一些深入理论的文章，找一找比较简短的数学类的内容。

2. 下午听了Yuxuan Yang的中期报告，结论只有：论文是硬通货，没有比这更重要的事情；

3. 下午看了Martin演示的代码，个人感觉demo比较简单，明天需要结合阅读的survey明确一下将来的方向。（仍然是向理论方向偏）。

4. 晚上粗略地看了一些论文，但是没有收获，粗略地看这样的事情，应该放在空闲时间做，下次不要这样了，力求每天都有理论收获。

**2022-05-06**

1. 上午：和Bosch的工程师交流了一下，单纯的几何SLAM确实没什么好做的东西。不过也不必专门专注于某一个东西，我的任务还是发paper，这是最为主要的。

   语义分割是在RGB图像中实现的，从RGB中分割，然后在深度图像中进行投影，最后在点云中表现出来。

2. 一点小感想：从根本上扭转自己的想法，不能只是在工作中做自己想成为的人。“重新注册登记”

3. 中午和dasun聊天，聊到什么应该作为一个PHD的课题，我们都一致认为应该博士的课题应该是一个research question，也就是说一定是学术性的。工程项目可以不着急，以后有的是机会做。但是这几年最好专注于一个学术问题，专注于一个点，或者说是一种方法。举例就是Heng Yang，将SDP松弛问题用到各个地方，就是很成功。

   找准自己的研究方向，需要和别人多多交谈。

   另外一个就是看到了浙大的FeiGao在science robotics发的文章，这也是课题组有积累，知道前沿，有价值的研究问题是什么才可以。但是我们的实验室似乎不具备这样的条件，都是一些非常具体的问题，过于具体，那就是工程了，我们应该要做的是一种数学抽象，是一种方法，是一种可以迁移，用于解决一类问题的方法。

4. 从别人的经验来看，作为PHD，多读paper也是必须要完成的事情，不仅要多读，对于重要的paper还要精读，只有精读，多掌握理论，才能有好的idea，因此，暂时制定一个计划，每天至少有两个小时进行理论的学习(新的理论，推公式等)。

   目前一天的计划：

   8:00~10:00 写paper；

   10:00~12:00 读新论文；

   pm:

   8:00~10:00 读博士论文/经典论文/课本；

**2022-05-07**

> 从一而终，不在于学到什么，而在于成为一个什么样的人。

- 想要了解一下OT在registration中的应用，单纯地看课本的话，还是有点迷茫（当然，投入的时间也不够），因此打算从实际的应用的论文出发，了解一下其在registration中的实际应用，然后再去补充一下论文的相关理论部分。

  从NIPS的论文 < *<u>Accurate Point Cloud Registration with Robust Optimal Transport</u>* > 出发，但是这篇论文的Comments其实并不好，因此我要找一些相近的论文去看。 因此从一篇 [*<u>Optimal Transport for Diffeomorphic Registration</u>*](../papers/MICAI2017.pdf)出发了解一下。

  > 完全没看懂啊...似乎是一个完全非常小众的理论。

- 个人认为，我现在需要认真打一下基础，理论基础。要制定一个计划，看相关的课本和书。

  > 制定一个计划，学习主要的课本的内容，主要是打好基础（一定要牢固，也就是说，论文和课本要结合地看）

- 阅读CDC这篇文章，目前觉得点云配准好像没什么要特别重要的，需要解决的问题。如果想不到问题，我认为是不应该做的，或者说在解决问题的过程中遇到什么问题，然后进行解决。（比如说进行建图？）

  在CDC中遇到了一些概率相关的问题，下载了 <u>*概率论和数理统计*</u> 这本书，然后重新阅读一下相关的基本概念。从知识的密度来看，教材和博士论文的知识密度是最高的。

总结经验：

1. 对于读到的每一篇论文，都要进行总结学到了哪些东西，不要拘泥于具体的应用场景，一定要跳出来，总结出一个抽象的/通用的方法出来，要么是框架，要么是一种通用的数学方法。

**2022-05-08**

- 关于昨晚和dasun的聊天，对于当前博一的看法就是，先不要去寻找什么research question，如果没有的话，发publication是最为重要的事情。有publication有这么几个好处：1.心态平和，有了平和的心态才能去从容地做一些其他的事情；2. supervisor的宽松度会更大一些，会放手让自己做一些其他的事情；3. 有了毕业的保障，可以放手去做一些其他的事情，理论的内容

  总之，要有一个很高的目标，但是要有一个非常务实的行动。做人做事要脚踏实地一些，不要基于求成，先把最基础的要求达到。

- 到5月底，完成survey的初稿（个人认为应该是可行的）

---

- 协方差矩阵可以看作是局部几何形状的描述；—— 在边缘的时候更多的是椭圆，在加角落附近更多的是圆。

  :question: 协方差矩阵如何计算呢？

- :question: 如何衡量两个矩阵的偏差？—— 在fuzzy的方法中，用作objective的权重

- 如何获得全局的correspondence? 直接一步估计出大概的粗位姿？

  —— 是否可以使用分层的clusters配准的方法？比如说首先只有几个cluster将3D场景大致分割，然后进行配准；有了大cluster之后，然会在每个大cluster之后进行小cluster的配准；

---

- 阅读论文 <u>Simultaneous Covariance Driven Correspondence (CDC) and Transformation Estimation in the Expectation Maximization Framework</u>

  文章总结：文章将error的uncertainty，将其表示为covariance matrix，并将其带入到objective function中；

  使用robust function对distance进行估计；并且带有权重 $w_{ij}$

  > 尚未整理

- 阅读论文 <u>Point Set Registration With Similarity and Affine Transformations Based on Bidirectional KMPE Loss</u> (1/2)

  文章总结：

  1. Bi-directional 的函数构造（基于ICP，寻找correspondence的点）；原文的解释是防止correspondence的“病态”，防止一些点全都对应了同一个单独的点；
  2. 用Robust函数替换L2E;
  3. Optimization 并没有太值得说的，借鉴最小二乘的问题，通过SVD进行求解。

  

-  最小二乘估计的基础是，假定残差满足高斯噪声分布。

- 克罗克内积：通常记为 $A \otimes B$:
  $$
  A \otimes B = [a_{11}B ... a_{1n} B]
  $$
  即：得到一个新的矩阵，每一个元素都是A中的元素与B进行乘积。

**2022-05-09**

- 安排会议，每周的Supervision会议；

- 阅读论文；

-  Cholesky factorization(克劳斯基分解)：将正定的 *埃尔米特矩阵*  分解成一个下三角矩阵与其 *共轭转置* 的乘积；

  1. 什么是 *埃尔米特* 矩阵：对于矩阵A，其元素 $a_{ij} = \overline{a_{ji}}$。 对于实矩阵而言，对称实矩阵就是特殊的埃尔米特矩阵；
  2. Cholesky 分解的表示：$A = L L^{*}$ ，其中 $L^*$ 表示的矩阵 $L$ 的共轭转置，如果是实矩阵，那就是其转置；
  3. 主要作用：求解线性方程组

- 正交矩阵：首先，正交矩阵 $Q$ 是一个方阵，其行向量和列向量彼此正交。有如下的性质：

  - $det(Q) = +/- 1$
  - $Q^{T} = Q^{-1}$ 
  - 对于旋转矩阵而言：$det(Q)=+1$

- QR分解

  - 正规矩阵：$A^{H}A = AA^{H}$ : 其中$H$ 表示的是矩阵的共轭转置；
  - QR分解：将矩阵$X$分解为两个矩阵相乘的形式，$X=QR$, 其中Q是正规正交矩阵；R是上三角矩阵；
  - 应用：求矩阵的特征值；

- Permutation Matrix: 每一行/列中只有一个元素是1，其他都是0；（由概念可知，其也应该是一个方阵）

- 阅读论文 <u>Point Set Registration With Similarity and Affine Transformations Based on Bidirectional KMPE Loss</u> (2/2)

  - 本文主要是针对RPM的方法，在原来的objective function的基础上进行改进，将原始的函数改为 *a separable concave quadratic function with a few nonlinear terms*. 并且改进了BnB的的搜索方法以提升效率；

  - 标题中的 *非对称* 的含义是 并不是每一个点都一个对应点 --> 主要的目的是用于处理部分重叠/干扰等部分；

  - 一些评论：

    which has the same dimensionality as that of the spatial transformation, by assuming a point-to-point correspondence. This assumption makes it difficult for APM to register partially overlapping point clouds or point clouds with gross outliers. （来源：Efficient Global Point Cloud Registration by Matching Rotation Invariant Features Through Translation Search）

- 我记得在ROBIN/TEASER中，其在图阶段的滤波是考虑一个全连接的图？也就是每个点都和周围的点进行连接？但是这样是不是扩大的搜索的范围？是否可以不必要全连接的图，而是只考虑一个点周围的几个点进行过滤？（目的是为了提高处理的速度）

- 阅读论文 <u>Robust Point Set Registration Using EM-ICP with Information-Theoretically Optimal Outlier Handling</u>

  - 本文是将两个point sets都看作是GMM，然后利用KL散度对两个概率分布进行逼近；
  - 文章题目中的 *outlier handling* 的含义是：将GMM的系数 $w_i$ 看作是一种“权重”用来衡量 inlier 和 outlier；不仅仅对 transformation parameter进行优化，还需要对 *coefficient* 进行优化；因为对于KL散度，其用了最大化 *scene* 的香农熵和最小化 *scene&model* 的香农交叉熵；这样从<u>定性分析</u>，是进行了 *outlier rejection*；

- 阅读论文 <u>Provably Approximated Point Cloud Registration</u>

  - SVD分解求解registration是如何做到的，我想要看一下；
  - 引申论文：<u>A polynomialtime bound for matching and registration with outliers</u> (从计算几何来 *provable guarantees*)
  - 介绍了一个 *core-set* 的观点（reference中）我觉得挺有意思的，就是数据压缩，用很少的点去近似原来的点集。（这个和我刚开始的思路差不多，读的论文多了，反而思路变窄了）

**2022-05-10**

- 将各种关系用函数进行抽象表示，函数就是一种映射关系；

- 阅读论文 <u>Provably Approximated Point Cloud Registration</u>

  - Theorm1 证明了存在一个 *witness set* 能够在给定的 *correspondence* 下 通过算法1对最优的 *cost function* 进行估计；——> 用当前的数据进行bound的估计，那么：如何找到这个 *witness set* 呢？--> 使用RANSAC-style的方式进行寻找这个set；

  - 用类似于RANSAC的方式进行采样，迭代各个 *witness set* 然后计算相应的 *(R,t)* 最后取最小的一个；

  - 感觉这个方法还有很大的提升空间：当前的采样方式是一种完全随机的采样（可以说就是随机地进行采样，然后进行恢复Rt以及相应的对应关系，利用恢复后的对应关系寻找最小的一个）

    —— 问题的关键在于其所谓的 *witness set* 这个理论是否足够扎实？——> 后续需要好好看一看这个理论。

  - 总结而言，当前的 *provable* 或是 *certifiably* 都是先求出一个解，然后对该解进行验证；

  - TODO：需要对该问题的基本理论进行一个稍微深入的研究，即：其 *provable* 是否非常solid，如果是，感觉这个问题可以进行一个比较大的提升。

- 阅读论文 <u>DICP: Doppler Iterative Closest Point Algorithm</u>

  - 一个初步的想法是，这篇文章更加关注的是新的硬件：FMCW Lidar: 能够给出像素级的速度；
  - 本文主要的思路是，FMCW lidar可以测量每一个点相对于lidar的速度。可以求点的期望速度，然后和测量速度进行对比，得到一个误差项，将该误差项添加到 *cost function* 中。
  - 另外，可以用测量速度和计算速度进行比较，计算速度是在假设所有的点都是“静态”的时候进行的估计，如果二者之间有一个差值，那么可以作为一个阈值来进行动态点的去除；

- 阅读论文 <u>Convex Hull Aided Registration Method (CHARM)</u>

**2022-05-11**

- 阅读论文 <u>Convex Hull Aided Registration Method (CHARM)</u>

  - > performing first rigid registration and then nonrigid registration are usually helpful for achieving more accurate results

    进行非刚性配准，首先进行刚性配准，获得一个较好的初值之后，然后进行非刚性配准；

  - 本文首先通过原点云构造 *convex hull*， 然后将这些 *convex* 上的 *triangular facet* 投影到平面上，通过平面找 *correspondence* （2D找correspondence比较成熟，简单）；然后再映射会3D找三维的点的映射；

- 阅读论文 <u>Robust low-overlap 3-D point cloud registration for outlier rejection</u>

  - **马尔可夫过程** ： 下一个状态只与当前在状态有关，而与过去的状态无关（其描述了一类随机过程）

  - **流形 manifold?**

  - 马尔可夫随机场：具有马尔可夫性质的随机场-- 即在随机场中的变量其值只与其周围的几个变量有关，与其他的节点无关。

  - 随机场：就是一个随机过程，但是其变量不再是“时间”，而是一个随机变量的子集/或者说是一个小的 *manifold*；

  - 吉比斯分布(也被称为 **Boltzmann distribution** (玻尔兹曼分布))：
    $$
    p_{i} \propto \exp{-\epsilon_{i}/(kT)}
    $$

  - <u>没太读懂</u>这篇论文，大致的含义是用 *马尔可夫随机场* 对点进行估计，估计基于两个前提：

    - inlier 通常比 outlier 更靠近最近点；
    - inlier 的周围通常也是 inlier；

  - 主要表达的含义就是通过 *hidden Markov random fields* 来进行 *outlier rejection*，可以处理overlap很小的情况

  - 能够探测 *Overlap* 区域我认为是一个很重要的内容，可以用来对 *CorAI* 进行改进，如果能够探测到哪个地方是 *Overlap* 的，再对最后的结果进行判断；

- 阅读论文 <u>The Richer Representation the Better Registration</u>

  - Levenberg–Marquardt algorithm / 高斯牛顿算法 / 梯度下降算法 读论文的时候遇到了这几个方法的名称，可以考虑一下空间的时候去研究一下；

    LM 算法的主要应用：优化 非线性 最小二乘的函数。

  - B样条是什么东西？什么是PCA（主成分分析？）

  - 线性函数：满足条件 1) 维持向量加法；2) 维持标量乘法；

  - 该文章主要提出的问题是用一个多面体/高维曲面去模拟一个点云（这背后的思想其实是，点云过于稀疏，是无法很好地描绘原来点云表示的图形的）。然后用 *source* 和这个 *target* 曲面的距离作为objective function进行使用。

    那么其实这个地方可能有一个可以扩展的点：

    1. 是否有其他表示点云的方式？—— 是对点云数据进行压缩；还是对点云进行补全，有更好的表达方式呢？
    2. 这是点对B样条函数，能否有函数对函数的方式呢？

  - 总结：文章将 *target* 点集表示成为 *implicit B-spline* 的格式，然后基于此构造目标函数；通过LM对目标函数进行优化，实现了精度的提升。（不需要寻找 *correspondence*， 只需要构造距离函数即可）

- 如何对文章进行分析：

  - 看这篇文章使用的数据集；
  - 看这篇文章和什么方法进行对比，对比的效果如何；
  - 这篇文章的引文进行引用时的评价；

**2022-05-12**

- 阅读论文 <u>Gravitational Approach for Point Set Registration</u>

  - N体问题；
  - 文章总结
    - 这篇文章主要contribution是什么：本文主要提出了一个新的算法，用天体动力学中的N体问题来模拟点云配准，取得了在noise的场景中比较好的情况；
    - 其novelty在哪里：从天体物理中引入了新的解释，理论比较新颖；
    - 其和之前的方法比好在哪：并无很大的提升，对比的方法是单纯的ICP和CPD；
    - 不足之处：
    - 总结一下：modeled point set as particles with gravity as attractive force, and registration is accomplished by solving the differential equations of Newtonian mechanics.
  - 奇异点：没有定义的点，通常来说，如果参数在分母上，需要注意分母不能为0；
  - 如果求两个点集的旋转；如何通过SVD来进行求解？—— 最初的那篇SVD的文章需要看一下。
  - 什么是二阶动力学系统？

- 阅读论文 <u>Efficient Registration of High-Resolution Feature Enhanced Point Clouds</u>

  - 什么是 *Monte-Carlo Simulation*(蒙特卡洛仿真)？

  - 小总结：

    > Interpret point sets as rigid swarms of particles with masses moving in gravitational or electromagnetic force fields. The optimal alignment corresponds to the state of the minimum potential energy of the system

  - 有点像 <u>Gravitational Approach for Point Set Registration</u>。

    > Our approach is based on the principles of mechanics and thermodynamics. We solve the registration problem by assuming point clouds as rigid bodies consisting of particles. Forces can be applied between both particle systems so that they attract or repel each other.

- 阅读论文 <u>Fast Rotation Search with Stereographic Projections for 3D Registration</u>

  - 本文是利用BnB寻找最优的全局 *rotation*，前提是 *translation* 已经给出；（这种解耦的方法可以了解一下，也可以总结一下如何进行 *rotation* 的估计);
  - 本文的主要的 *motivation* 是对 *BnB* 在6D的搜索空间进行优化；
  - 什么是 *M-Estimator*?

**2022-05-13**

- 今天上午主要是上课，制作PPT然后进行讨论；

- 下午进行上课，下午安装了 *ubuntu*， 一开始安装了 *ubuntu18.04* 一堆的问题（没有wifi; 分辨率不对；连不上网等等）；安装 *ubuntu20.04* 才算成功，但是还有一个问题， 那就是 *nvidia* 显卡驱动安装出现问题；

  我的安装步骤：

  - *Software&update* 中安装驱动；
  - 安装CUDA;
  - 安装 *nvidia-setting*
  - 通过 *prime-select* 选择对应显卡

  错误是运行 *nvidia-setting* 失败；解决方法：在 *BIOS* 中将 *secure-boot* 禁掉（猜测是ubuntu不支持nvidia的驱动运行）；

- 阅读论文 <u>Fast Rotation Search with Stereographic Projections for 3D Registration</u>

  - 该文的前提是：给定了translation之后，单独求两个点云之间的旋转；基于的思想是使得 *correspondence* 的数目最多；

  - 至少需要给定三个正确的对应点才能使找到的 *rotation* 是正确的。

  - *Indicator Function*: 指示函数：值域是一个离散的几何；例如：
    $$
    F(x) = 0 \\if x \in A;\\
    F(x) = 1 \\ if x \notin A
    $$

**2022-05-15**

- 阅读论文 <u>Fast Rotation Search with Stereographic Projections for 3D Registration</u>

  - 基本的思想：将 *translation* 和 *rotation* 进行解耦；本文的主要的 *contribution* 是 *global, optimal rotation search*; 

  - 单词：hasten（加速）

  - BnB的基本思想：分割有限的定义域，不断逼近最终的全局最优点；我认为使用 *BnB* 的最大前提是，有一个 **有限的定义域**，可能是一个几何体，总之，是能够进行分割的定义域。在这里 *BnB* 能够使用的前提就是使用 *axis-angle* 对旋转进行参数化；

    BnB的算法的主要瓶颈在于寻找 *Bound*，在分支定界中，每一次都需要 *prune* 一下新的范围，

  - 本文的主要内容就是改进了 *BnB*，提高了算法的效率，加快了寻找 *Bound* 的过程。

  - :question: BnB 不是太了解，这个地方需要有空静下心来进行研究；

- 阅读论文 <u>Context-Aware Gaussian Fields for Non-rigid Point Set Registration</u>

  - 希尔伯特空间是什么含义？以及 *Reproducing kernel Hilbert space*(再生希尔伯特空间)
  - 从摘要和结论中看到的东西：1. 高斯场；2. 保留 *invariant features*；
  - **Hadamard product**：两个相同维度的矩阵 **逐元素** 进行相乘；
  - 论文 <u>Shape Classification Using the Inner-Distance</u> 好像看起来挺有意思的，提出了一种新的距离描述，更好地用于形状结构的描述子；
  - 论文 <u>Shape matching and object recognition using shape contexts</u> (Citations:8371) 描述局部信息，值得一读(SC)
  - 本片论文的主要的 *contribution* 是使用 *inner distance* 作为 *SC* 的描述子进行生成。<u>看引用文献进行总结的时候，还是需要看相关作者/导师的评语比较准确。</u>

- 阅读论文 <u>Fine-To-Coarse Global Registration of RGB-D Scans</u>

  - 之前的研究中，都是研究 *pairwise* 的配准，是否可以研究一下连续好几帧的配准。（主要的考虑是是否和 *CorAI* 进行搭配使用）

  - 虽然是多帧匹配，但是有一定的借鉴意义：

    > incorporating additional constraints between planes, such as parallelism or orthogonality, into registration

  - *agglomerative hierarchical clustering* —— 用来对点云进行平面提取？

  - 主要的观念就是：在 optimization 中，进行和**几何特征**的提取，并用以进行构造 objective function；

- 阅读论文 <u>Discriminative Optimization: Theory and Applications to Point Cloud Registration</u>

  - 将点云配准以一个数值优化的角度来看：1 构造 *Objective Function*; 2. 寻找 *Global Optimum*

  - 问题：当前数值优化方法（高维度）：1. 很多局部最优解；2. 计算开销大；各种 *cost function* 在退化环境下会导致局部最优点的增加，也就是说 *cost function* 构造的前提是一个 *ideal model*.

  - 判别优化：不需要构造 cost function，直接从 *data* 进行学习?

  - *Stationary Point* 静止点的概念是？

  - 从效果上来看，DO的方法似乎能够处理 *overlap* 非常小的情况。

  - > A method that learns search directions from data without the need for a cost function. 
    >
    > The drawback of this approach is that the features and maps are specific to each object and do not generalize.

  - 没太读懂这篇文章。

## 05-16

- 图神经网络听起来挺有意思，是不是有空研究一下？	