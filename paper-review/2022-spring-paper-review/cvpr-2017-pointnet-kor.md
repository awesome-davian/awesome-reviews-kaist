---
description: Qi et al. / PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation / CVPR 2020
---

# PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation \[Kor\]

## 1. Introduction
해당 연구에서는 3D 데이터 중 하나인 point cloud를 딥러닝을 이용하여 다루고자 한다.
기존의 대부분의 연구에서는 3D voxel이나 projection을 거친 여러 2D 이미지의 조합으로 변형하여 point cloud를 다루었다.
이와 달리 해당 논문에서는 주어진 point cloud를 있는 그대로 사용하여 유의미한 표현을 추출하고 classification과 segmentation과 같은 task에 활용하는 PointNet을 제안한다.

## 2. Related Work
### Deep Learning on 3D Data
- Volumetric CNN: 3D 데이터를 다루기 위한 가장 흔한 방법 중 하나로 voxelization을 마친 3D 데이터를 다룬다. 하지만 데이터의 sparsity와 3D convolution의 computation cost가 높다는 한계가 있다.
- Multiview CNN: 3D 데이터를 2D 이미지들로 렌더링하여 2D convolution을 거치는 방법이다. 높은 성능의 2D CNN을 활용할 수 있지만 렌더링 과정에서 정보 손실이 있고 3D 정보를 제대로 활용하지 못한다는 한계가 있다. 

### Deep Learning on Unordered Sets
Point cloud의 가장 큰 특징 중 하나는 2D 이미지나 3D voxel과 달리 값들의 순서가 없는 unordered sets라는 것이다.

## 3. Problem Statement
해당 연구에서는 unordered point sets를 직접적으로 입력값으로 사용하는 딥러닝 프레임워크를 제안한다.
Object classification task에서는 해당 네트워크가 모든 k개의 클래스 각각에 대한 스코어를 출력한다.
Semantic segmentation task를 위해서는 각 point에 대해 m개의 semantic class에 대한 스코어를 출력한다.

## 4. Deep Learning on Points Sets
### 4.1 Properties of Point Sets
- Unordered
- Interaction among points
- Invariance under transformations

### 4.2. PointNet Architecture
<img src="/.gitbook/assets/2022spring/19/architecture.png" width="1000" align="center">
PointNet의 전체 구조는 위의 그림과 같다. 해당 네트워크는 다음과 같은 3가지 핵심 요소로 구성되어 있다.
- Max poling layer: 모든 points로부터의 정보를 합쳐주는(aggregate) symmetric function
- Local and global information combination
- Two joint alignment networks: 입력 points와 point features를 align한다.

#### Symmetry Function for Unordered Input
$$f({x_1, ..., x_n}) \approx g( h(x_1), ..., h(x_n)),$$
Max pooling을 하나의 symmetric function으로 사용하는 것이 PointNet의 핵심 요소이다.

#### Local and Global Information Aggregation
Global point cloud feature vector를 계산한 다음, 해당 값들을 point features에 concat해줌으로써 local한 정보와 global한 정보를 모두 갖도록 하였다.

#### Joint Alignment Network
Point cloud는 입력에 대해 geometric transformation이 가해져도 semantic labeing은 변하지 않아야 한다는 특징이 있다.
T-net이라는 mini-network를 통해 affine transformation matrix를 예측하도록 하고 해당 transformation을 입력 points에 적용함으로써 해당 문제를 해결하였다.

## 4. Experiment
### 4.1 Applications
#### 3D Object Classification
<img src="/.gitbook/assets/2022spring/19/cls_table.png" width="400" align="center">

#### 3D Object Part Segmentation
<img src="/.gitbook/assets/2022spring/19/shapenet_table.png" width="400" align="center">
<img src="/.gitbook/assets/2022spring/19/part_seg_vis.png" width="400" align="center">

#### Semantic Segmentation in Scenes
<img src="/.gitbook/assets/2022spring/19/semantic_seg_table.png" width="400" align="center">
<img src="/.gitbook/assets/2022spring/19/semantic_seg_vis.png" width="400" align="center">

### 4.2. Architecture Design Analysis
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