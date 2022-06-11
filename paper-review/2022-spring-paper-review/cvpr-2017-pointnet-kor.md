---
description: Qi et al. / PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation / CVPR 2020
---

# PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation \[Kor\]

## 1. Problem definition
Deep learning has shown amazing performance in various computer vision tasks.
However, most studies have proposed a methodology dealing with 2D images and video.
In this study, point cloud, one of the 3D data, is to be dealt with using deep learning.
Point cloud is one of the forms of data representing three-dimensional information, which is a set of multiple points forming a cloud shape.
For example, LiDAR data for autonomous driving is composed of point cloud.
Each point in the point cloud generally has a three-dimensional coordinate value of (x, y, z), and it is sometimes composed of six-dimensional data including colors such as RGB.
One of the main reasons that point cloud is difficult to handle is the order between points does not exist (unordered set).
In other words, unlike the case of a 2D image, the point cloud only has (x, y, z) coordinates for each point without any order between them.
The previous works transformed the point cloud into 3D voxel (voxeliazation) or several 2D images by projection to deal with this point cloud.
However, this process requires high computational cost or causes information loss during the projection process.
To overcome these problems, this paper proposes PointNet, which uses raw point cloud as input to extract meaningful representations and utilize them for tasks such as classification and segmentation.

## 2. Motivation
Several deep learning approaches have been proposed to handle point cloud, but most have followed the approach of applying a typical CNN model after transforming into 3D voxel or 2D image.
These methods require additional computation cost and can cause information loss because they doesn't treat raw point cloud itself and require additional transformation and post-processing.
Therefore, the study aim to proposes a method to process raw point cloud as it is.

### 2.1 Related work
**Deep Learning on 3D Data**
- Volumetric CNN: Voxelization to 3D voxel is one of the most common ways to deal with 3D data. Voxelization is mapping (x, y, z) coordinates to a regularized 3D voxel coordinate, and the voxelized result has an order of values as in general data. 3D voxelization has the advantage to apply the powerful 3D volumetric CNNs. However, if the values of the original point cloud are widely distributed, there is a limitation that it is difficult to achieve high performance with a typical 3D CNN due to the large sparsity of the voxelized data. In addition, a lot of costly operations are required in the voxelization process.
- Multiview CNN: It is a method of rendering 3D data into multiple 2D images and applying traditional 2D convolution. It follows the mechanisom of how a person recognizes 3D data; uses a 2D view from a specific perspective. The biggest advantage of this approach is that it can utilize high-performance 2D CNNs such as ResNet and use pre-trained weights learned on large datasets like ImageNet as initial training points. However, there is a limitation in that inevitable information loss occurs in the process of projection to 2D image, and the spatial characteristics of 3D data cannot be fully utilized.

### 2.2 Idea
The key idea of PointNet to handle raw point cloud itself is the continuous MLP operations.
MLP has the advantage of being able to process an unordered set of points without any pre-processing and maintain the characteristics of each point.
In addition, this work proposes to utilize max-pooling as a symmetric function to aggregate the features of each point obtained as a result of MLP operations.
Using Max-pooling, the network can select significant vectors from among the total of features and the result can be used for vision tasks such as classification.


## 3. Method

### PointNet Architecture

![PointNet Architecture](/.gitbook/assets/2022spring/19/architecture.png)

The overall structure of PointNet is shown in the figure above. The network consists of the following three key elements.
- Max poling layer: symmetric function to aggregate the information from all points
- Local and global information combination
- Two joint alignment networks: align the input point and point features.

The specific descriptions for each component is below.

### 3.1 Symmetry Function for Unordered Input
One of the characteristics to process an unordered input is that the result must be consistent even if the order of given input is different (input permutation).
In order to ensure that the model is not affected by such input permutation, the paper utilizes an aggregate symmetric function that combines the information obtained at each point.
Here, the symmetric function is received as an input with n vectors and one new vector is output, and the result is independent of the order of the inputs.
For $$N$$ points, the symmetric function can be expressed as follows.

$$
f({x_1, ..., x_n}) \approx g( h(x_1), ..., h(x_n))
$$

여기서 $$f : 2^{\mathbb{R}^N} \rightarrow \mathbb{R}, h : \mathbb{R}^N \rightarrow \mathbb{R}^K, g : \mathbb{R}^K \times \cdots \times \mathbb{R}^K \rightarrow \mathbb{R}$$이며 $$g$$가 symmetric function이다. 

해당 논문에서는 실험적으로 $$h$$를 일반적인 MLP network, $$g$$는 간단한 variable function과 max pooling function의 조합으로 추정하였다.
즉, max pooling을 하나의 symmetric function으로 사용하는 것이 PointNet의 핵심 요소이다.

### 3.2 Local and Global Information Aggregation
Symmetric function을 통해 얻은 정보는 point cloud에 대한 global information을 반영하게 되며 classification과 같은 task에는 충분한 정보를 제공할 수 있다.
하지만 높은 segmentation 성능의 네트워크를 구성하기 위해서는 local한 정보와 global 정보를 모두 고려하는 것이 필요하다.
해당 논문에서는 local and global 정보를 모두 반영하기 위하여 위에서 언급한 방법으로 얻은 global features를 이전 단계의 local features에 연결(concatenation)하는 간단한 방법을 채택하였다.
위 figure의 Segmentation Network의 입력 부분에 해당 과정이 나타나있다.
이러한 과정을 통해 각 point는 고유의 local 정보 뿐만 아니라 cloud 전반에 걸친 global 정보도 가질 수 있게 된다.
해당 방법은 실험적으로 shape part segmentation과 scene segmentation 모두에서 좋은 성능을 보여주었다.

### 3.3 Joint Alignment Network
Point cloud는 입력에 대해 rigid transformation과 같은 geometric transformation이 가해져도 semantic labeing은 변하지 않아야 한다는 특징이 있다.
T-net이라는 mini-network를 통해 affine transformation matrix를 예측하도록 하고 해당 transformation을 입력 points에 적용함으로써 해당 문제를 해결하였다.
해당 mini-network은 전체 network와 유사하며 point independent feature extraction, max pooling 그리고 fully connected layer와 같은 일반적은 모듈로 구성되었다.
해당 네트워크 학습을 위하여 regularization term을 loss에 추가하였으며 아래의 수식과 같이 feature transformation matrix가 orthogonal matrix와 유사하다는 제약을 걸어주었다.

$$
L_{reg} = || I - AA^T ||_{F}^2 
$$

여기서 $$A$$는 mini-network가 예측한 feature alignment network이다.

## 4. Experiment & Result
### 4.1 Experimental setup
**Tasks and Dataset**
- 3D Object Classification: 주어진 point cloud를 분류하는 문제이다. ModelNet40 shape classifiation benchmark에 대해 실험하였으며 해당 데이터셋은 12,311개의 CAD 모델로 구성되어 있으며 총 40개의 category가 존재한다.

- 3D Object Part Segmentation: Part segmentation은 난이도가 높은 fine-grained 3D 문제이다. 3D scan 혹은 mesh가 주어졌을 때, 각 point 혹은 face의 part category label (예시. 의자 다리, 컵 손잡이)을 예측하는 것이 목표이다. ShapeNet part dataset에 실험을 진행하였으며 해당 데이터셋은 16,881개의 shape으로 구성되어 있으며 총 16개의 category가 존재한다.

- Semantic Segmentation in Scenes: Part segmentation과 유사하며 해당 task에서 예측하고자 한 point label이 object part label에서 semantic object class로 바뀐 셋팅이다. Standford 3D semantic parsing dataset에 대해 실험을 진행하였으며 해당 데이터셋은 6개의 구역(area)에 대한 총 271개의 방(room)으로 구성되어 있으며 총 13개의 category가 존재한다.

### 4.2 Result
- 3D Object Classification

![3D Object Classification Result](/.gitbook/assets/2022spring/19/cls_table.png)

위 table은 ModelNet40 dataset에 대한 object classification 결과이다.
해당 실험에서는 point cloud에서 feature를 추출하는 전통적인 방법들 (point density, D2, shape contour 등)과 PointNet을 비교하였다.
PointNet이 deep learning을 활용한 방법들 중에서 가장 높은 성능을 달성하였다.

- 3D Object Part Segmentation

![3D Object Part Segmentation Result](/.gitbook/assets/2022spring/19/shapenet_table.png)

위 table은 ShapeNet에 대한 object part segmentation 결과이다.
해당 실험에서는 category 별 IoU와 mean IoU 결과를 측정하였다.
PointNet이 baseline 모델에 비해 2.3%의 mean IoU 향상이 있는 것을 확인할 수 있으며 대부분의 category에서 높은 성능을 기록하였다.

![3D Object Part Segmentation Visualization](/.gitbook/assets/2022spring/19/part_seg_vis.png)

모델의 강인함(robustness)를 측정하기 위해 Blensor Kinect Simulator 데이터셋애 대해 추가 실험을 진행한 결과를 나타낸 것이다.
PointNet을 이용하면 구분하기 어려운 part도 정확하게 예측하고 있는 것을 확인할 수 있다.

- Semantic Segmentation in Scenes

![Semantic Segmentation in Scenes Result](/.gitbook/assets/2022spring/19/semantic_seg_table.png)

위 table은 Standford 3D semantic parsing dataset에 대한 결과이다.
논문에서 제안하는 PointNet이 baseline에 비해 압도적으로 높은 성능을 보여준 것을 확인할 수 있다.


![Semantic Segmentation in Scenes Visualization](/.gitbook/assets/2022spring/19/semantic_seg_vis.png)

PointNet의 semantic segmentation 결과를 나타낸 정성적 결과이다.
해당 네트워크가 매끄러운(smooth) 결과를 예측할 수 있고 누락(missing)되거나 가려진(occlusion) 점들에 대해 robust하게 예측하고 있음을 확인할 수 있다.


### 4.3. Architecture Design Analysis
#### Comparison with Alternative Order-invariant Methods

![Comparison with Alternative Order-invariant Methods](/.gitbook/assets/2022spring/19/order_invariance.png)

해당 논문에서 point cloud의 unordered 특성을 반영하기 위해 symmetric function으로 max pooling을 활용하였다.
Max pooling의 성능을 확인하기 위하여 다양한 order-invariant methods와 비교하였다.
비교한 방법들로는 MLP와 RNN의 기반의 LSTM이 있으며 symmetric function으로는 attention sum과 average pooling을 비교하였다.
실험 결과 모든 방법들에 비해서 max pooling이 가장 높은 성능을 보인 것을 확인할 수 있다.


#### Effectiveness of Input and Feature Transformations

![Effectiveness of Input and Feature Transformations](/.gitbook/assets/2022spring/19/feature_transforms.png)

논문에서 제안하는 input and feature transformation의 효과에 대해 실험한 결과이다.
이때 transformation을 적용하지 않은 baseline이 이미 충분히 높은 성능을 달성한 것이 주목할만 하다.
Input transformation과 regualization loss 모두 성능을 향상시킨 것을 확인할 수 있다.


### 4.4. Time and Space Complexity Analysis

![Time and Space Complexity Analysis](/.gitbook/assets/2022spring/19/time_comparison.png)

기존 모델인 Subvolume와 MVCNN과 비교하여 PointNet의 parameter 수와 FLOPs을 계산하였다.
이전 실험 결과들을 통해 Subvolume과 MVCNN이 PointNet에 비해 더 높은 성능을 달성하였지만, PointNet이 computational cost 측면에서 모두 효율적인 것을 확인할 수 있다.
또한 PointNet은 입력 point의 수에 따라 linear하게 complexitiy가 변하기 때문에 훨씬 scalable하다는 장점이 있다.
PointNet은 1080X GPU에 대해 실험하였을 때, point cloud classification을 1초에 약 1,000개의 objects 그리고 semantic segmentation은 1초에 약 2개의 방(room)을 처리할 수 있을 정도로 빠른 속도를 보여주었다.


## 5. Conclusion
해당 논문에서는 이전 연구들과 달리 별도의 전처리 없이 raw point cloud를 입력으로 하는 PointNet이라는 deep neural network를 제언하였다.
Point cloud는 데이터가 순서가 없다는 특징이 있기 떄문에 이를 효과적으로 처리할 수 있는 방법으로 MLP를 활용한 point별 feature extraction과 max pooling 기반의 feature aggreation을 제안하였다.
PointNet을 이용하여 object classification, part segmentation 그리고 semantic segmentation과 같은 여러 종류의 3D recognition을 효과적으로 수행하였다.

### Take home message \(오늘의 교훈\)
> Point cloud는 이미지나 3D voxel과 달리 sparse하고 unordered하다는 특징이 있다.
>
> Point cloud 데이터에 변형을 가하지 않고 있는 그대로 입력으로 받는 PointNet을 제안하였다.

## Author / Reviewer information


### Author

**최동민 \(Dongmin Choi\)** 

* KAIST AI
* dmchoi@kaist.ac.kr

### Reviewer
TBD

## Reference & Additional materials

1. Qi, Charles R., et al. "Pointnet: Deep learning on point sets for 3d classification and segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
2. Codes: https://github.com/charlesq34/pointnet
3. Youtube: https://www.youtube.com/watch?v=Cge-hot0Oc0&ab_channel=ComputerVisionFoundationVideos
