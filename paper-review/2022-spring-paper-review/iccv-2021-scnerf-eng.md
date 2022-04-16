---
description: Jeong et al. / Self-Calibrating Neural Radiance Fields / ICCV 2021
---

# Self-Calibrating Neural Radiance Fields \[Eng]

## Self-Calibrating Neural Radiance Fields \[Eng]

한국어로 쓰인 리뷰를 읽으려면 [**여기**](broken-reference/)를 누르세요.

## 1. Problem definition

Given a set of images of a scene, the proposed method, dubbed SCNeRF, jointly learns the geometry of the scene and the accurate camera parameters without any calibration objects. This task can be expressed as the following equation.

> Find $$K, R, t, k, r_{o}, r_{d}, \theta$$, when&#x20;
>
> $$\mathbf{r}=(\mathbf{r_o}, \mathbf{r_d})=f_{cam}(K, R, t, k, r_o, r_d)$$&#x20;
>
> $$\hat{\mathbf{C}}(\mathbf{r})=f_{nerf}(\mathbf{r};\theta)$$

where $$\mathbf{r}$$ is a ray, $$\mathbf{r_o}$$ and $$\mathbf{r_d}$$ is ray origin and ray direction, $$f_{cam}$$ is a function that generates ray from camera parameters, $$(K,R,t,k,r_o,r_d)$$ are camera parameters, $$\hat{\mathbf{C}}(\mathbf{r})$$ is an estimated color of ray $$\mathbf{r}$$, $$\theta$$ is a parameter set of NeRF model, $$f_{nerf}$$ is a function that estimate color of a ray given using NeRF parameters.

Generally, scene geometry is learned given known camera parameters, or camera parameters are estimated without improving or learning scene geometry.

Unlike the previous approach, the purpose of this paper is to learn camera parameters $$(K,R,t,k,r_o,r_d)$$ and NeRF model parameters $$\theta$$ jointly.

## 2. Motivation&#x20;

### Related work

#### Camera Model

Because of its simplicity and generality, traditional 3D vision tasks often assume that the camera model is a simple pinhole model. However, with the development of camera models, various camera models have been introduced, including fish eye models, and per-pixel generic models. A basic pinhole camera model is not enough to represent these kinds of complex camera models.

#### Camera Self-Calibration

Self-Calibration is a research topic that calibrates camera parameters without an external calibration object (e.g., a checkerboard pattern) in the scene.

In many cases, calibration objects are not readily available. Thus, calibrating camera parameters without any external objects has been an important research topic.

However, conventional self-calibration methods solely rely on the geometric loss or constraints based on the epipolar geometry that only uses a set of sparse correspondences extracted from a non-differentiable process. This could lead to diverging results with extreme sensitivity to noise when a scene does not have enough interest points. Lastly, conventional self-calibration methods use an off-the-shelf non-differentiable feature matching algorithm and do not improve or learn the geometry. It is well known that the better we know the geometry of the scene, the more accurate the camera model gets.

#### Neural Radiance Fields(NeRF) for Novel View Synthesis&#x20;

NeRF is a work that synthesizes a novel view of the scene by optimizing a separate neural continuous volume representation network for each scene.&#x20;

At the time when the NeRF was published, this work achieves state-of-the-art results for synthesizing novel views of complex scenes by optimizing an underlying continuous volumetric scene function using a sparse set of input views.

However, this requires not only a dataset of captured RGB images of the scene but also the corresponding camera poses and intrinsic parameters, which are not always available.

### Idea

* Pinhole camera model parameters, fourth-order radial distortion parameters, and generic noise model parameters that can learn arbitrary non-linear camera distortions are included to overcome the limitation of the pinhole camera model.
* To overcome the limitation of geometric loss used in the previous self-calibration methods, additional photometric consistency is used.&#x20;
* To get a more accurate camera model using improved geometry of the scene, the geometry represented using Neural Radiance Fields is learned jointly.

## 3. Method

### Differentiable Camera Model

#### Pinhole Camera Model

The first component of our differentiable camera unprojection is based on the pinhole camera model, which maps a 4-vector homogeneous coordinate in 3D space to a 3-vector in the image plane.&#x20;

$$P'_{3\times1} = M_{3\times4}P_W=K_{3\times3}\left[R\; T\right]_{3\times 4} P_{W4\times 1}$$

First, we decompose the camera intrinsics into the initialization K0 and the residual parameter matrix ΔK. This is due to the highly non-convex nature of the intrinsics matrix that has a lot of local minima.

Similarly, we use the extrinsics initial values R0 and t0 and residual parameters to represent the camera rotation R and translation t. However, directly learning the rotation offset for each element of a rotation matrix would break the orthogonality of the rotation matrix. Thus, we adopt the 6-vector representation \[26] which uses unnormalized first two columns of a rotation matrix to represent a 3D rotation:

#### Fourth Order Radial Distortion

#### Generic Non-Linear Camera Distortion

#### Computational Graph of Ray Direction & origin



### Loss



### Curriculum Learning



## 4. Experiment & Result



