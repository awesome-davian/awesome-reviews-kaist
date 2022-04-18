---
description: Yin et al. / Center-based 3D Object Detection and Tracking / CVPR 2021
---
# CenterPoint \[Kor\]


## 1. Problem definition
자율주행을 위한 3D object detection을 수행할 때 가장 흔하게 사용하는 방법은 LiDAR를 이용해 얻은 point cloud에서 인식할 물체의 3D bounding box를 찾는 것이다. 이 때 2D object detection에 비해 더 높은 난이도를 제공하는 부분으로 이 논문에서 제시하는 것은 다음과 같다.

1. Point cloud가 sparse하게 분포함.
2. 실제 3D bounding box가 global 좌표와 평행하지 않음.
3. 3D object 의 크기, 모양 등이 다양함.

기존의 2D detetion에서는 axis-aligned 2D bounding box를 이용해 이미지 위 여기저기에서 원하는 물체를 찾아 인식하는 방법을 많이 사용하였다. 하지만 3D point cloud의 경우 sparse하기 때문에 bounding box를 이용해 모든 구역을 조사하기에는 너무 낭비가 심하다.



# 2. Motivation
### Related Work

2D object detection의 경우 대개 이미지에서 수직, 수평한 직사각형 모양의 bounding box를 찾는 방식으로 진행된다. 이 때 먼저 이미지 상에서 물체가 있을 것으로 예상되는 부분의 bounding box를 찾은 뒤, 해당 위치의 물체를 분류하는 방식도 있고, 처음부터 특정 카테고리의 bounding box를 찾는 방식도 있다. Center-based 방식은 anchor box를 만들어 모든 구역의 사각형을 조사하는 것이 아니라, 전체 이미지에서 bounding box의 중심점으로 추정되는 지점을 찾는 것으로 대표적으로 CenterNet이 있다.

3D object detector는 많은 경우에 이 2D object detector의 개념을 확장하여 만들게 된다. 2D 공간의 pixel과 같이 3D 공간을 일정한 간격으로 나눈 것을 voxel이라고 하는데, 이 voxel을 이용하는 방법으로 VoxelNet 등이 있다. PointPillars는 voxel 대신 pillar를 이용하는 것으로 수직상으로 같은 열에 속하는 voxel을 하나로 묶어 이용한다.

 #### CenterNet

2D object detector 중 하나인 CenterNet의 경우 bounding box의 중심점만을 찾고, 나머지 특징은 이 regression을 통해 계산한다. 이 CenterNet에 $I\in R^{W \times H \times 3}$인 이미지를 입력하면 keypoint heatmap $\hat{Y} \in [0, 1]^{\frac{W}{R} \times \frac{H}{R} \times C}$를 생성하는데, 이 때 $R$은 output stride, $C$는 keypoint 종류, 즉 class의 개수이다. 여기서 heatmap $\hat{Y}$의 각 좌표가 keypoint인 경우엔 1, 배경인 경우엔 0이 되도록 하는 것이 목표이다. 이를 위해 training에 사용할 target heatmap $Y$를 생성하는데, 이 때 각 ground truth keypoint를 중심으로 Gaussian kernal을 적용하여 heatmap을 생성한다. 이 Gaussian kernal의 $\sigma_p$는 해당 keypoint $p \in R^2$를 중심으로 하는 물체의 크기에 따라 달라진다. 따라서 이를 기준으로 CenterNet을 학습시키면, 학습 후 estimation 결과에 keypoint 위치 정보 뿐 아니라 크기 정보도 함께 포함될 수 있다. CenterNet은 heatmap $\hat{Y}$에서 keypoint에 해당하는 local maxima들을 찾고, 해당 keypoint 위치 $\left(x_i, y_i\right)$의 값인 $\hat{Y}_{x_iy_ic}$ 을 이용하여 위치, 방향 등 다른 공간 정보를 구한다.

![][figure1]



### Idea

이 논문에서는 CenterNet의 방식을 3D object detection에 이용하여 anchor-based가 아니라 center-based로 물체의 bounding box를 찾는 방식을 제안한다. 중심점을 찾으면 방향이 정해져있지 않기 때문에 방향에 상관 없이 일정한 정보를 학습하는데 유용하고, 모델이 heatmap 정보만 학습하고 나머지 정보는 regress하기 때문에 속도가 더 빠르다는 장점이 있다.



# 3. Method

### Overall Framework

![][figure2]

먼저 point cloud 데이터에 3D backbone을 적용하여 Map-view features $\mathrm{M}\in\mathbb{R}^{W\times H \times F}$을 얻는다. 다음으로 여기에 center-head를 적용하여 class 개수만큼의 channel을 갖는 heatmap $\hat{Y}$를 생성한다. 이 $\hat{Y}$의 target은 3D의 형태를 상공에서 보고 2D로 나타냈을 때, 각 물체의 중심에 gaussian filter를 적용한 것으로, 이는 2D CenterNet과 같은 방식이다. 이 heatmap의 local maxima가 찾고자하는 물체들의 중심이 된다. 속도, 방향, 크기 등의 다른 속성은 이렇게 찾은 중심점의 feature에 regression을 적용하여 찾는다.

#### Two-Stage CenterPoint

이렇게 heatmap을 생성하여 물체의 중심점을 찾고, 이를 바탕으로 갖가지 다른 속성을 구하는 과정이 첫번째 단계이다. 하지만 이렇게 중심점에서만 많은 특성을 뽑아내기엔 여기에 담겨있는 정보가 충분하지 않을 수 있다. 따라서 이 논문에서는 두번째 단계를 제안한다. 첫번째 단계에서 찾은 크기, 방향 등의 정보를 이용해 bounding box의 외곽선을 찾고, Map-view feature인 $\mathrm{M}$에서 bilinear interpolation을 이용해 해당 위치의 feature를 가져온다. 이 feature를 중심점의 feature와 합쳐 MLP를 통과시켜 얻은 결과를 이용해 confidence score를 구하고, 첫번째 단계에서 찾은 bounding box의 정보를 더 세밀하게 조절하도록 학습시킨다.



# 4. Experiment & Result

### Experimental setup

이 논문에선 실험을 위해 Waymo Open Dataset과 nuScens Dataset을 사용했다. 이 dataset은 모두 자율주행을 위한 학습을 위한 것으로 도로 위 사진, LiDAR 등의 데이터와 각 시점에서 차량과 보행자 등 도로 위 물체들의 공간적 정보를 제공한다. 또한 3D encoder로는 VoxelNet과 PointPillars를 사용하였다.



### Result

![][table1]

![][table2]

위의 표는 각각 Waymo와 nuScenes dataset을 이용하여 기존의 3D detection 모델들과 성능을 비교한 결과이다. 결과를 보면 CenterPoint가 기존의 다른 모델들에 비해 더 상향된 성능을 보인다는 것을 알 수 있다.



![][table5]

![][table6]

위의 표는 각각 Waymo와 nuScense dataset을 이용하여 동일한 3D encoder를 사용하였을 때 anchor-based 방식과 center-based 방식의 성능을 비교한 것으로, center-based 방식을 이용하였을 때 더 나은 성능을 보이는 것을 알 수 있다.



# 5. Conclusion

이 논문에서 저자는 CenterNet의 개념을 바탕으로 하여 3D object detector를 기존의 anchor-based가 아닌 center-based 방식으로 설계하였다. CenterPoint 모델에서는 3D point cloud를 3D encoder를 이용해 위에서  수직으로 내려 보았을 때의 2D feature map으로 변환한 뒤, 이를 이용해 heatmap을 생성하였고, 이 heatmap의 local maxima를 찾아 각 물체의 bounding box의 중심점을 구했다. 또한, 이 중심점의 feature를 이용하여 bounding box의 대략적인 공간 정보를 찾고, 이렇게 찾은 위치에 해당하는 feature를 다시 이용하여 더 세밀한 정보를 찾아내었다. 이러한 방식으로 CenterPoint는 anchor-based 방식과 비교하였을 때 더 간단하면서 더 좋은 성능을 보였다.



## Author / Reviewer information

### Author

**이남진 (Namjin Lee)**

- KAIST EE
- namjin@kaist.ac.kr

### Reviewer

1. Korean name (English name): Affiliation / Contact information
2. Korean name (English name): Affiliation / Contact information
3. …



## Reference & Additional materials

1. Tianwei Yin, Xingyi Zhou, Philipp Krähenbühl; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 11784-11793
2. [Official GitHub repository](https://github.com/tianweiy/CenterPoint)
3. Xingyi Zhou, Dequan Wang, and Philipp Krähenbühl. Objects as points. arXiv:1904.07850, 2019.
4. Peiyun Hu, Jason Ziglar, David Held, and Deva Ramanan. What you see is what you get: Exploiting visibility for 3d object detection. CVPR, 2020.
5. Ross Girshick. Fast r-cnn. ICCV, 2015.
6. Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. CVPR, 2014.
7. Yin Zhou and Oncel Tuzel. Voxelnet: End-to-end learning for point cloud based 3d object detection. CVPR, 2018.
8. Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, and Oscar Beijbom. Pointpillars: Fast encoders for object detection from point clouds. CVPR, 2019.



[figure1]:imgs\figure1.png
[figure2]:https://d3i71xaburhd42.cloudfront.net/22d40963e633e1b4af4a9fefda68e1b8dc96ba63/4-Figure2-1.png
[table1]: imgs\table1.png
[table2]:table2.png
[table5]:imgs\table5.png
[table6]:imgs\table6.png
