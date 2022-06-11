---
description: Qi et al. / PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation / CVPR 2020
---

# PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation \[Kor\]

## 1. Problem definition
딥러닝은 다양한 컴퓨터 비전 task에서 굉장히 좋은 성능을 보여왔다.
하지만 대부분의 연구가 2D 이미지와 영상을 다루는 방법론을 제안하였고 3D 데이터에 대한 접근은 비교적 부족한 상황이었다.
해당 연구에서는 3D 데이터 중 하나인 point cloud를 딥러닝을 이용하여 다루고자 한다.
Point cloud는 3차원 정보를 나타내는 데이터 형태 중 하나이며 여러개의 point들이 모여서 구름(cloud) 형태를 띄고 있다.
예를 들어, LiDAR 센서에서 얻은 자율주행을 위한 데이터가 있다.
일반적으로 point cloud의 각 point는 (x, y, z)의 3차원 좌표값을 가지고 있으며, 추가로 RGB와 같은 색상을 포함한 6차원 데이터로 구성된 경우도 있다.
Point cloud가 다루기 어려운 가장 큰 이유 중 하나는 point 간의 순서가 존재하지 않는 unordered set이라는 것이다.
다시 말해, 2D 이미지 같은 경우에는 특정 pixel의 오른쪽에는 또 다른 pixel이 존재하며 해당 pixel은 1개의 pixel만큼 떨어져 있다는 순서가 존재하지만, point cloud는 여러 개의 점 각각에 대한 (x, y, z) 좌표만 존재하기 때문에 이들 간의 순서가 존재하지 않는다.
이러한 point cloud를 다루기 위해서 기존의 연구에서는 point cloud를 3D voxel(voxeliazation)이나 projection을 거친 여러 2D 이미지의 조합으로 변형하여 다루었다.
하지만 이와 같은 과정을 많은 연산을 필요로 하거나 혹은 projection 과정에서 유의미한 정보를 손실하는 문제점을 유발한다.
이와 같은 문제점을 극복하기 위하여 해당 논문에서는 주어진 point cloud를 있는 그대로 사용하여 유의미한 표현을 추출하고 classification과 segmentation과 같은 task에 활용하는 PointNet을 제안한다.

## 2. Motivation
Point cloud를 다루기 위한 여러가지 딥러닝 접근법이 제안되었지만 대부분 3D voxel이나 2D image로 변형(transformation)한 후 일반적인 CNN 모델을 적용하는 방식을 따랐다.
이러한 방법들은 주어진 point cloud를 있는 그대로 다루지 못하고 추가 변형 및 후처리를 필요로 하기 때문에 많은 연산을 필요로 하고 정보 손실을 유발할 수 있다.
따라서 해당 연구에서는 주어진 point cloud를 있는 그대로 다루기 위한 방법을 제안한다.

### 2.1 Related work
**Deep Learning on 3D Data**
- Volumetric CNN: 3D 데이터를 다루기 위한 가장 흔한 방법 중 하나로 voxelization을 마친 3D 데이터를 다룬다. 여기서 voxelization이란, point cloud를 좌표를 정규화된 3D 좌표계로 mapping하는 것을 뜻하며 voxelzation으로 얻은 결과는 일반적인 3D 데이터와 같이 값들의 순서가 존재한다. 3D voxelization을 거치게 되면 일반적으로 3D 데이터를 다루기 위해 제안된 여러가지 3D Volumnetric CNN을 적용할 수 있다는 장점이 있다. 하지만 원본 point cloud의 값들이 많이 분산되어 있다면 voxelization으로 얻은 데이터의 sparsity가 크기 때문에 일반적인 3D CNN으로 높은 성능을 달성하기 어렵다는 한계가 있다. 또한 point cloud를 다루기 위해 voxelization을 거쳐야 하기 때문에 많은 연산이 추가로 필요하다.
- Multiview CNN: 3D 데이터를 2D 이미지들로 렌더링하여 2D convolution을 거치는 방법이다. 사람이 3D 데이터를 인식하는 방법과 유사한 원리를 따르며 주어진 point cloud를 특정 시각에서 바라보았을 때 얻을 수 있는 2D view를 이용하는 방법이다. 해당 접근법의 가장 큰 장점은 ResNet과 같은 높은 성능의 2D CNN을 활용할 수 있으며 ImageNet과 같은 대규모 데이터셋에 학습된 pre-trained weights를 initial point로 사용할 수 있다는 점이다. 하지만 2D image로 rendering하는 과정에서 필연적인 정보 손실이 발생하며 3D 데이터가 가지고 있는 공간적인 특성을 제대로 활용하지 못한다는 한계가 존재한다.

### 2.2 Idea
주어진 point cloud를 있는 그대로 다루기 위한 PointNet의 핵심 아이디어는 연속된 MLP 연산을 이용하는 것이다.
MLP를 활용하면 별도의 전처리 과정 없이 순서가 없는 point 집합을 처리할 수 있으며 point 간의 상관성을 고려하면서도 각 point의 특징을 유지할 수 있다는 장점이 있다.
또한 MLP 연산 결과로 얻은 각 점들에 대한 feature를 aggregate하기 위하여 max-pooling을 symmetric function으로 활용하는 것을 제안하였다.
Max-pooling을 활용하여 여러 point에 대한 feature 중에서 유의미한 값을 network가 선택(selection)할 수 있으며 해당 값들이 classification과 같은 task에 활용될 수 있다.



## 3. Method

### PointNet Architecture

![PointNet Architecture](/.gitbook/assets/2022spring/19/architecture.png)

PointNet의 전체 구조는 위의 그림과 같다. 해당 네트워크는 다음과 같은 3가지 핵심 요소로 구성되어 있다.
- Max poling layer: 모든 points로부터의 정보를 합쳐주는(aggregate) symmetric function
- Local and global information combination
- Two joint alignment networks: 입력 points와 point features를 align한다.

아래에서 각 요소에 대한 자세한 설명을 아래와 같다.

### 3.1 Symmetry Function for Unordered Input
순서가 없는 입력을 처리하기 위해 가져야 하는 특성 중 하나는 해당 입력값이 주어지는 순서가 달라져도(input permutation) 결과가 일관되어야 한다는 것이다.
이러한 input permutation에 모델이 영향을 받지 않도록 하기 위해 해당 논문에서는 각 point에서 얻은 정보를 합쳐주는(aggreagte) symmetric function을 활용하는 것이다.
여기서 symmetric function을 n개의 벡터를 입력으로 받아 하나의 새로운 벡터를 출력하며 이때 입력의 순서에 무관한 결과를 내뱉어야 한다.
$$N$$개의 points에 대하여 symmetric function은 아래와 같이 나타낼 수 있다.

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
<img src="/.gitbook/assets/2022spring/19/order_invariance.png" width="400" align="center">
Point cloud의 unordered 특성을 반영하기 위한 3가지 order-invariant methods인 attention sum, average pooling, 그리고 max pooling을 비교하였다.

#### Effectiveness of Input and Feature Transformations
<img src="/.gitbook/assets/2022spring/19/feature_transforms.png" width="400" align="center">
Input과 feature를 해당 논문에서 제안하는 방법들을 이용하여 transformation하였을 때의 결과를 비교하였다.

#### Robustness Test
<img src="/.gitbook/assets/2022spring/19/robustness_test.png" width="400" align="center">
입력에 조작이 가해졌을 때(input corruption) 해당 모델이 얼마나 민감하게 반응하는지를 확인하기 위한 robustness test를 진행하였다.

### 4.3. Time and Space Complexity Analysis
<img src="/.gitbook/assets/2022spring/19/time_comparison.png" width="400" align="center">
기존 모델인 Subvolume와 MVCNN과 비교하여 PointNet의 parameter 수와 FLOPs을 계산하였다.

## 5. Conclusion
직접적으로 raw point cloud를 입력으로 하는 PointNet이라는 deep neural network를 제언하였다.
해당 네트워크를 이용하여 object classification, part segmentation 그리고 semantic segmentation과 같은 여러 종류의 3D recognition을 효과적으로 수행하였다.

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
