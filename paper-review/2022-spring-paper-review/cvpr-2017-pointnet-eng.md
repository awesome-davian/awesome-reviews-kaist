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
A key point to process an unordered input is that the result must be consistent even if the order of given input is changed (input permutation).
In order to ensure that the model is not affected by such input permutation, the paper select to use an aggregate symmetric function that combines the information obtained at each point.
Here, the symmetric function uses $$n$$ vectors as an input and output a new vector and the output is independent of the order of the inputs.
For $$N$$ points, the symmetric function can be expressed as follows.

$$
f({x_1, ..., x_n}) \approx g( h(x_1), ..., h(x_n))
$$

where $$f : 2^{\mathbb{R}^N} \rightarrow \mathbb{R}, h : \mathbb{R}^N \rightarrow \mathbb{R}^K, g : \mathbb{R}^K \times \cdots \times \mathbb{R}^K \rightarrow \mathbb{R}$$ and $$g$$ is the symmetric function. 

This work composes $$h$$ as common MLP layers and $$g$$ as combination of variable function and max pooling function.
That is, a key point of PointNet is leverage max pooling as symmetric function.

### 3.2 Local and Global Information Aggregation
The output of symmetric function contains global information about the point cloud and can provide sufficient information for tasks such as classification.
However, in order to achieve high segmentation performance, it is necessary to consider both local and global information.
In this paper, a simple method is adopted to reflect both local and global information, which is concatenating global features obtained by the above-mentioned method to local features of the previous stage.
The above Figure describes this process at the right bottom of Segmentation Network.
Through this aggregation process, each point can have global information throughout the cloud as well as unique local information.
The method experimentally showed high performance in both shape part segmentation and scene segmentation.

### 3.3 Joint Alignment Network
Point cloud has the characteristic that semantic labeling should not change even when geometrical transformation such as rigid transformation is applied to the input.
The work solves this problem by predicting the affine transformation matrix through a mini-network called T-net and applying the corresponding transformation to the input points.
The corresponding mini-network is similar to the overall network and consists of widely-used modules such as point independent feature extraction, max pooling, and FC  layers.
A regulation term is added to the loss, and as shown in the following equation, the feature transformation matrix is constrained to be similar to the orthogonal matrix.

$$
L_{reg} = || I - AA^T ||_{F}^2 
$$

where $$A$$ is the feature alginment network predicted by mini-network.

## 4. Experiment & Result
### 4.1 Experimental setup
**Tasks and Dataset**
- 3D Object Classification: this task aim to classify the category of given point cloud. Experiments are conducted on ModelNet40 shape classifiation benchmark, which is composed of 12,311 CAD models with a total of 40 categories.

- 3D Object Part Segmentation: this task aim to assign the category of each point of face in given 3D scan or mesh. Experiments are conducted on ShapeNet part dataset, which is composed of 16,881 shape with a total of 16 categories.

- Semantic Segmentation in Scenes: this task is similar to part segmentation and the point label to be predicted in the task has been changed from object part label to semantic object class. Experiments are conducted on Standford 3D semantic parsing dataset, which is composed of 6 areas including 271 rooms with a total of 13 categories.


### 4.2 Result
- 3D Object Classification

![3D Object Classification Result](/.gitbook/assets/2022spring/19/cls_table.png)

The table above shows the result of object classification for ModelNet40 dataset.
In this experiment, this paper compared PointNet with traditional methods (point density, D2, shape control, etc.) to extract features from the point cloud.
PointNet achieved the highest performance among the methods using deep learning.

- 3D Object Part Segmentation

![3D Object Part Segmentation Result](/.gitbook/assets/2022spring/19/shapenet_table.png)

The table above shows the result of object part segmentation for ShapeNet.
In this experiment, IoU and mean IoU results by category were measured.
It can be seen that PointNet has a 2.3% mean IoU improvement compared to the baseline model and high performance is recorded in most categories.

![3D Object Part Segmentation Visualization](/.gitbook/assets/2022spring/19/part_seg_vis.png)

It shows the results of additional experiments on the Blensor Kinect Simulator dataset to measure the robustness of the model.
Using PointNet, it can be seen that parts that are difficult to distinguish are also accurately predicted.

- Semantic Segmentation in Scenes

![Semantic Segmentation in Scenes Result](/.gitbook/assets/2022spring/19/semantic_seg_table.png)

The table above shows the results for the Standford 3D semantic parsing dataset.
PointNet showed overwhelmingly high performance compared to baseline.


![Semantic Segmentation in Scenes Visualization](/.gitbook/assets/2022spring/19/semantic_seg_vis.png)

This figure is a qualitative result of the semantic segmentation on Standford 3D dataset.
PointNet is capable of predicting smooth results and robustly predicting missing or occluded points.


### 4.3. Architecture Design Analysis
#### Comparison with Alternative Order-invariant Methods

![Comparison with Alternative Order-invariant Methods](/.gitbook/assets/2022spring/19/order_invariance.png)

This paper proposed to use max pooling as a symmetric function to reflect the unordered characteristics of the point cloud.
In order to confirm the performance of max pooling, the setting was compared with various order-invariant methods.
The comparative methods include MLP and RNN-based LSTM, and the attraction sum and average pooling are compared as symmetric functions.
As a result of the experiment, it was confirmed that max pooling showed the highest performance compared to all methods.


#### Effectiveness of Input and Feature Transformations

![Effectiveness of Input and Feature Transformations](/.gitbook/assets/2022spring/19/feature_transforms.png)

This is the result of an experiment on the effect of input and feature transformation proposed in this paper.
It is noteworthy that baseline without transformation has already achieved sufficiently high performance.
Both input transformation and localization loss have improved performance.


### 4.4. Time and Space Complexity Analysis

![Time and Space Complexity Analysis](/.gitbook/assets/2022spring/19/time_comparison.png)

The number of parameters and FLOPs of PointNet were calculated compared to the existing models, Subvolume and MVCNN.
Previous experimental results show that Subvolume and MVCNN achieve higher performance than PointNet, but PointNet is efficient in terms of both computational cost.
PointNet also has the advantage of being much more scalable because its complexity varies linearly with the number of input points.
When PointNet tested on a 1080X GPU, it showed that the approach was efficient to proess point cloud classification about 1,000 objects per second and semantic segmentation about two rooms per second.


## 5. Conclusion
This paper suggests a deep neural network called PointNet, which inputs raw point clouds without pre-processing such as voxelization or projection.
Since the point cloud has the characteristic that the data are out of order, this work proposes feature extraction for each point using MLP and feature aggregation based on max pooling as a way to effectively process it.
PointNet effectively solved 3D recognition tasks such as object classification, part segmentation, and semantic segmentation.

### Take home message \(오늘의 교훈\)
> Point cloud has a characteristic that the distribution of data is sparse and unordered unlike 2D image and 3D voxel.
>
> PointNet uses raw point cloud as input without applying a transformation to the original data.

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
