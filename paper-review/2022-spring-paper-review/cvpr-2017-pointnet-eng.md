---
description: Qi et al. / PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation / CVPR 2020
---

# PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation \[Kor\]

## 1. Introduction
This paper deal with point cloud, which is one of the 3D data.
Most of previous works voxelized point cloud or projected to series of 2D images. 
However, this paper proposes PointNet that directly use raw point cloud as input to extract meaningful features.

## 2. Related Work
### Deep Learning on 3D Data
- Volumetric CNN: the most widely used approach which use point cloud after voxelizing. However, it has limitations that the method is sensitive to the sparsity of point cloud and has high computation cost.
- Multiview CNN: this method renders point cloud to 2d images and utilize high performance 2D CNNs. However, it has information loss during the projection.

### Deep Learning on Unordered Sets
One of a key feature of point cloud is unordered sets unlike 2D image or 3D voxel.

## 3. Problem Statement
This paper propose a deep learning framework which directly use unordered point sets as input.
The network outputs $$k$$ scores for $$k$$ number of classes for object classification task.
The network outputs $$m$$ semantic scores for each point for semantic segmentation task.

## 4. Deep Learning on Points Sets
### 4.1 Properties of Point Sets
- Unordered
- Interaction among points
- Invariance under transformations

### 4.2. PointNet Architecture
<img src="/.gitbook/assets/2022spring/19/architecture.png" width="1000" align="center">
The figure is an entire architecture of PointNet. The network is composed of three key components as below.
- Max poling layer: a symmetric function to aggregate the information from all points.
- Local and global information combination
- Two joint alignment networks: align bost input points and point features

#### Symmetry Function for Unordered Input
$$f({x_1, ..., x_n}) \approx g( h(x_1), ..., h(x_n)),$$
The key component of PointNet is to use max pooling as a symmetric function.

#### Local and Global Information Aggregation
After computing the global point cloud feature vector, PointNet concats the vector to the original point features to obtain both local and global information.

#### Joint Alignment Network
Point cloud has a characteristic that the semantic labeling is invariant when a geometric transformation is applied.
This paper use a mini-network called T-net which predicts an affine transformation matrix and apply this transformation to input points.


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
They compare three order-invariant methods to relect the unordered feature os point cloud: attention sum, average pooling and max pooling.

#### Effectiveness of Input and Feature Transformations
<img src="/.gitbook/assets/2022spring/19/feature_transforms.png" width="400" align="center">
They compare the results when applying the transformations proposed in this work.

#### Robustness Test
<img src="/.gitbook/assets/2022spring/19/robustness_test.png" width="400" align="center">
The robustness test is conduected to check how model reacts to the input corruption.

### 4.3. Time and Space Complexity Analysis
<img src="/.gitbook/assets/2022spring/19/time_comparison.png" width="400" align="center">
This analysis compare PoineNet to the two previous method, Subvolume and MVCNN, in terms of the number of parameters and FLOPs.

## 5. Conclusion
This paper propose a deep neural network called PointNet which directly use raw point cloud as input.
PointNet successfully tackes number of 3D recognition tasks such as object classification, part segmentation and semantic segmentation.

### Take home message \(오늘의 교훈\)
> Point cloud has a sparse and unordered characteristic unlike 2D image and 3D voxel.
> 
> This work propose a PointNet which directly use raw point cloud as input.

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