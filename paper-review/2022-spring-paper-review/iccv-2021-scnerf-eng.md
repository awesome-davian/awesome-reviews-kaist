---
description: Jeong et al. / Self-Calibrating Neural Radiance Fields / ICCV 2021
---

# Self-Calibrating Neural Radiance Fields \[Eng]

## Self-Calibrating Neural Radiance Fields \[Eng]

한국어로 쓰인 리뷰를 읽으려면 [**여기**](broken-reference/)를 누르세요.

![](../../.gitbook/assets/2022spring/35/figure1.png)

## 1. Problem definition

Given a set of images of a scene, the proposed method, dubbed SCNeRF, jointly learns the geometry of the scene and the accurate camera parameters without any calibration objects. This task can be expressed as the following equation.

> Find $$K, R, t, k, r_{o}, r_{d}, \theta$$, when
>
> $$\mathbf{r}=(\mathbf{r_o}, \mathbf{r_d})=f_{cam}(K, R, t, k, r_o, r_d)$$
>
> $$\hat{\mathbf{C}}(\mathbf{r})=f_{nerf}(\mathbf{r};\theta)$$

where $$\mathbf{r}$$ is a ray, $$\mathbf{r_o}$$ and $$\mathbf{r_d}$$ is ray origin and ray direction, $$f_{cam}$$ is a function that generates ray from camera parameters, $$(K,R,t,k,r_o,r_d)$$ are camera parameters, $$\hat{\mathbf{C}}(\mathbf{r})$$ is an estimated color of ray $$\mathbf{r}$$, $$\theta$$ is a parameter set of NeRF model, $$f_{nerf}$$ is a function that estimate color of a ray given using NeRF parameters.

Generally, scene geometry is learned given known camera parameters, or camera parameters are estimated without improving or learning scene geometry.

Unlike the previous approach, the purpose of this paper is to learn camera parameters $$(K,R,t,k,r_o,r_d)$$ and NeRF model parameters $$\theta$$ jointly.

## 2. Motivation

### Related work

#### Camera Model

Because of its simplicity and generality, traditional 3D vision tasks often assume that the camera model is a simple pinhole model. However, with the development of camera models, various camera models have been introduced, including fish eye models, and per-pixel generic models. A basic pinhole camera model is not enough to represent these kinds of complex camera models.

#### Camera Self-Calibration

Self-Calibration is a research topic that calibrates camera parameters without an external calibration object (e.g., a checkerboard pattern) in the scene.

In many cases, calibration objects are not readily available. Thus, calibrating camera parameters without any external objects has been an important research topic.

However, conventional self-calibration methods solely rely on the geometric loss or constraints based on the epipolar geometry that only uses a set of sparse correspondences extracted from a non-differentiable process. This could lead to diverging results with extreme sensitivity to noise when a scene does not have enough interest points. Lastly, conventional self-calibration methods use an off-the-shelf non-differentiable feature matching algorithm and do not improve or learn the geometry. It is well known that the better we know the geometry of the scene, the more accurate the camera model gets.

#### Neural Radiance Fields(NeRF) for Novel View Synthesis

NeRF is a work that synthesizes a novel view of the scene by optimizing a separate neural continuous volume representation network for each scene.

At the time when the NeRF was published, this work achieves state-of-the-art results for synthesizing novel views of complex scenes by optimizing an underlying continuous volumetric scene function using a sparse set of input views.

However, this requires not only a dataset of captured RGB images of the scene but also the corresponding camera poses and intrinsic parameters, which are not always available.

### Idea

* Pinhole camera model parameters, fourth-order radial distortion parameters, and generic noise model parameters that can learn arbitrary non-linear camera distortions are included to overcome the limitation of the pinhole camera model.
* To overcome the limitation of geometric loss used in the previous self-calibration methods, additional photometric consistency is used.
* To get a more accurate camera model using improved geometry of the scene, the geometry represented using Neural Radiance Fields is learned jointly.

## 3. Method

### Differentiable Camera Model

#### Pinhole Camera Model

The first component of differentiable camera unprojection is based on the pinhole camera model, which maps a 4-vector homogeneous coordinate in 3D space $$P_{4 \times 1}$$ to a 3-vector in the image plane $$P'_{3 \times 1}$$.

$$
P'_{3\times1} = M_{3\times4}P=K_{3\times3}\left[R\; T\right]_{3\times 4} P_{4\times 1}
$$

Where $$K$$ is the intrinsics matrix, $$R$$ is the rotation matrix, $$T$$ is the translation matrix

First, the camera intrinsic parameters are decomposed into the initialization $$K_0$$ and the residual parameter matrix $$\Delta K$$(=$$z_K$$). This is due to the highly non-convex nature of the intrinsics matrix that has a lot of local minima.

$$
K=\begin{bmatrix} f_x+\Delta f_x & 0 & c_x + \Delta c_x \\ 0 & f_y + \Delta f_y & c_y + \Delta c_y \\ 0 & 0 & 1 \end{bmatrix} = K_0 + \Delta K \in \mathbb{R}^{3\times 3}
$$

Similarly, the extrinsics initial values $$R_0$$ and $$t_0$$ and residual parameters to represent the camera rotation R and translation t. However, directly learning the rotation offset for each element of a rotation matrix would break the orthogonality of the rotation matrix. Thus, the 6-vector representation which uses unnormalized first two columns of a rotation matrix is utilized to represent a 3D rotation:

$$
\mathbf{t} = \mathbf{t_0} + \Delta \mathbf{t}\\R=f(\mathbf{a_0}+\Delta \mathbf{a})\\f\begin{pmatrix}\begin{bmatrix} | & | \\ a_1& a_2\\ | & | \end{bmatrix}\end{pmatrix} = \begin{bmatrix}|&|&|\\\mathbf{b_1} & \mathbf{b_2} & \mathbf{b_3}\\| & | & |\end{bmatrix}_{3 \times 3}
$$

What $$f$$ does is quite similar to Gram-Schmidt process. To make it clear, I draw conceptual image as follows. Here, $$N(\cdot)$$ is $$L2$$ normalization.

![](../../.gitbook/assets/2022spring/35/gram\_schmidt\_like\_process.png)

As we can see in the figure, from unnormalized two vectors $$\mathbf{a_1}$$ and $$\mathbf{a_2}$$, orthonormal vectors $$\mathbf{b_1}, \mathbf{b_2}, \mathbf{b_3}$$ can be obtained.

####

#### Fourth Order Radial Distortion

Since commercial lenses deviates from ideal lens with single lens focal length, this creates a number of aberrations. The most common one is referred to as “radial distortion”.

![](../../.gitbook/assets/2022spring/35/radial\_distortion\_types.png)

Camera model of SCNeRF is extended to incorporate such radial distortions.

![](../../.gitbook/assets/2022spring/35/H360\_barrel\_distortion.png)

Undistorted normalized pixel coordinate $$(n'_x, n'_y)$$ converted from pixel coordinate $$(p_x, p_y)$$ can be expresses as the following.

$$
(n_x, n_y) = (\frac{p_x-c_x}{f_x},\frac{p_y-c_y}{f_y})\\r=\sqrt{n^2_x+n^2_y}\\\left[n'_x, n'_y, 1 \right]^T = K^{-1} \left[p_x(1+(k_1+z_{k_1}) r^2 + (k_2+z_{k_2}) r^4), p_y(1+(k_1+z_{k_1}) r^2 + (k_2+z_{k_2}) r^4),1 \right]
$$

where $$(k_1, k_2)$$ is initial radial distortion parameter denoted as $$k_0$$ and $$(z_{k_1}, z_{k_2})$$ are residuals denoted as $$z_k$$.

#### Ray Direction & Origin

Using[#pinhole-camera-model](iccv-2021-scnerf-eng.md#pinhole-camera-model "mention") and [#fourth-order-radial-distortion](iccv-2021-scnerf-eng.md#fourth-order-radial-distortion "mention"), ray direction $$\mathbf{r_d}$$ and ray origin $$\mathbf{r_o}$$ in the world coordinate can be expressed as the following.

$$
\mathbf{r_d} = N(R \cdot \left[n'_x, n'_y, 1 \right]^T)\\\mathbf{r_o}=\mathbf{t}
$$

where $$N(\cdot)$$ is vector normalization. For those who may confuse why $$\mathbf{t}$$ equals the ray origin $$\mathbf{r_o}$$ in the world coordinate, I draw conceptual image that shows the geometric meaning of vector $$\mathbf{t}$$.

![](../../.gitbook/assets/2022spring/35/H360\_ray\_origin\_t.png)

Since these ray parameters $$\mathbf{r_d}$$ and $$\mathbf{r_o}$$ are functions of intrinsics, extrinsics, and dirtortion paramameter residuals ($$\Delta f, \Delta c, \Delta a, \Delta t, \Delta k$$), we can pass gradients from the rays to the residuals to optimize the parameters. Note that $$K_0,R_0, t_0, k_0$$are initial values of each parameters and not optimized.

#### Generic Non-Linear Camera Distortion

Complex optical abberations in real lenses cannot be modeled using a parametric camera. For such noise, generic non-linear aberration model is used. Specifically, local ray parameter residuals $$\mathbf{z_d} = \Delta \mathbf{r}_d(\mathbf{p})$$, $$\mathbf{z}_o = \Delta \mathbf{r}_o(\mathbf{p})$$ is used, where $$\mathbf{p}$$ is the image coordinate.

$$
\mathbf{r}'_d = \mathbf{r}_d + \mathbf{z}_d \\\mathbf{r}'_o=\mathbf{r}_o+\mathbf{z}_o
$$

Bilinear interpolation is used to extract continuous ray distortion parameters.

$$
\mathbf{z}_d(\mathbf{p}) = \sum_{x=\lfloor\mathbf{p}_x\rfloor}^{\lfloor\mathbf{p}_x\rfloor+1}\sum_{x=\lfloor\mathbf{p}_y\rfloor}^{\lfloor\mathbf{p}_y\rfloor+1} \left(1-|x-\mathbf{p}_x|\right)\left(1-|y-\mathbf{p}_y|\right)\mathbf{z}_d\left[x,y\right]
$$

where $$\mathbf{z}_d[x, y]$$ indicates the ray direction offset at a control point in discrete 2D coordinate $$(x, y)$$. $$\mathbf{z}_d[x, y]$$ is learned at discrete locations only. Dual comes for free.

$$
\mathbf{z}_o(\mathbf{p}) = \sum_{x=\lfloor\mathbf{p}_x\rfloor}^{\lfloor\mathbf{p}_x\rfloor+1}\sum_{x=\lfloor\mathbf{p}_y\rfloor}^{\lfloor\mathbf{p}_y\rfloor+1} \left(1-|x-\mathbf{p}_x|\right)\left(1-|y-\mathbf{p}_y|\right)\mathbf{z}_o\left[x,y\right]
$$

To help your understanding, the conceptual image of a generic non-linear aberration model is attached below.

![](../../.gitbook/assets/2022spring/35/10kparam\_local\_smoothness.png) ![](../../.gitbook/assets/2022spring/35/10kparam\_control\_point.png)

#### Computational Graph of Ray Direction & origin

From [#pinhole-camera-model](iccv-2021-scnerf-eng.md#pinhole-camera-model "mention"), [#fourth-order-radial-distortion](iccv-2021-scnerf-eng.md#fourth-order-radial-distortion "mention"), [#generic-non-linear-camera-distortion](iccv-2021-scnerf-eng.md#generic-non-linear-camera-distortion "mention"), the final ray direction and ray origin can be expressed using the following graph.

![](../../.gitbook/assets/2022spring/35/figure2.png)

### Loss

To optimize calibration parameters, both geometric consistency loss and photometric consistency loss is exploited.

#### Geometric Consistency Loss

![](../../.gitbook/assets/2022spring/35/geometric\_consistency\_loss\_overall.png)

Geometric Consistency Loss is $$d_\pi$$ in the above figure. Let's break this down into pieces.&#x20;

First, let $$\left(\mathbf{p_A} \leftrightarrow \mathbf{p_B}\right)$$ be a correspondence on camera 1 and 2 respectively. The projected ray distance loss directly measures the discrepancy between rays. When all the camera parameters are calibrated, the ray $$\mathbf{r}_A$$ and $$\mathbf{r}_B$$ should intersect at the 3D point that generated point $$\mathbf{p}_A$$ and $$\mathbf{p}_B$$. However, when there’s a misalignment due to an error in camera parameters, we can measure the deviation by computing the shortest distance between corresponding rays.&#x20;

Let a point on $$\mathbf{r}_A$$ be $$\mathbf{x}_A(t_A) = \mathbf{r}_{o,A} + t_A\mathbf{r}_{d,A}$$, and a point on $$\mathbf{r}_B$$ be $$\mathbf{x}_B(t_B) = \mathbf{r}_{o,B} + t_A\mathbf{r}_{d,B}$$. A distance between the $$\mathbf{r}_A$$ and a point on the $$\mathbf{r}_B$$ is $$d$$ as we can see in the above figure.&#x20;

Solving $$\frac{\mathbf{d}d^2}{\mathbf{d}t_B}|_{\hat{t}_B}=0$$ gives us $$\hat{t}_B$$ that makes the distance between $$\mathbf{x}_B$$ and $$\mathbf{r}_A$$ minimum.&#x20;

$$
\hat{t}_B = \frac{\left(\mathbf{r}_{A,o}-\mathbf{r}_{B,o}\right) \times \mathbf{r}_{A,d}\cdot \left( \mathbf{r}_{A,d} \times \mathbf{r}_{B,d}\right)}{\left(\mathbf{r}_{A,d}\times\mathbf{r}_{B,d}\right)^2}
$$

From this, we can find a point on ray B that has the shortest distance to ray A.

$$
\mathbf{x_B} = \mathbf{x_B}(\hat{t}_B)
$$

Dual comes for free.

$$
\mathbf{x_A} = \mathbf{x_A}(\hat{t}_A)
$$

After projecting the points to image planes and computing distance on the image planes, geometric consistency loss $$d_\pi$$ in the above figure can be obtained, where $$\pi(\cdot)$$ is a projection function.

Note that correspondences for a point far from the cameras would have a large deviation, while correspondences for a point close to the cameras would have a small deviation. Thus, to remove this depth sensitivity, the distance between the two points is computed on the image plane, not 3D space.

#### Photometric Consistency Loss

Photometric consistency loss is defined as the following.

$$
\mathcal{L} = \sum_{\mathbf{p}\in\mathcal{I}}||C(\mathbf{p})-\hat{C}(\mathbf{r(p)})||^2_2
$$

where $$\mathbf{p}$$ is a pixel coordinate, and $$\mathcal{I}$$ is a set of pixel coordinates in an image, $$\hat{C}(\mathbf{r})$$ is the output of the volumetric rendering using the ray $$\mathbf{r}$$ of corresponding pixel $$\mathbf{p}$$, $$C(\mathbf{p})$$ is the ground truth color.

> **HOW TO ESTIMATE** $$\hat{C}(\mathbf{r})$$**?**
>
> The color value $$\mathbf{C}$$ of a ray can be represented as an integral of all colors weighted by the opaqueness along a ray, or can be approximated as the weighted sum of colors at N points along a ray as following.
>
> $$\mathbf{\hat{C}} \approx \sum_i^N\left( \prod_{j=1}^{i-1}\alpha (\mathbf{r}(t_j), \Delta_j) \right)\left( 1-\alpha(t_i, \Delta_i) \right) \mathbf{c}\left( \mathbf{r}(t_i), \mathbf{v} \right)$$
>
> where $$\alpha(\cdot)$$ is transparency, $$\mathbf{c(\cdot)}$$is color, $$\Delta_i = t_{i+1}-t_{i}$$
>
> For those who want to learn more about this equation, please refer the "NeRF" paper in[#reference-and-additional-materials](iccv-2021-scnerf-eng.md#reference-and-additional-materials "mention")

Note that photometric consistency loss is differentiable with respect to the learnable camera parameters. From this, we can define gradients for the camera parameters and ablel to calibrate cameras.

### Curriculum Learning

To optimize geometry and camera parameters, we learn the neural radiance field and the camera model jointly. However, it is impossible to learn accurate camera parameters when the geometry is unknown or too coarse for selfcalibration. Thus, we sequentially learn parameters: geometry and a linear camera model first and complex camera model parameters.The camera parameters determine the positions and directions of the rays for NeRF learning, and unstable values often result in divergence or sub-optimal results. Thus, we add a subset of learning parameters to the optimization process to jointly reduce the complexity of learning cameras and geometry. First, we learn the NeRF networks while initializing the camera focal lengths and focal centers to half the image width and height. Learning coarse geometry first is crucial since it initializes the networks to a more favorable local optimum for learning better camera parameters. Next, we sequentially add camera parameters for the linear camera model, radial distortion, and nonlinear noise of ray direction, ray origin to the learning. We learn simpler camera models first to reduce overfitting and faster training.

Following is the final learning algorithm. $$get\_params$$ function returns a set of parameters of the curriculum learning which returns a set of parameters for the curriculum learning which progressively adds complexity to the camera model. Next, we train the model with the projected ray distance by selecting a target image at random with sufficient correspondences. Heuristically, we found selecting images within maximum 30°from the source view gives an optimal result.

![](../../.gitbook/assets/2022spring/35/algorithm1.png)

## 4. Experiment & Result

Here, not all but some representative experimental results are shown.

### Experimental Setup

* **Dataset**
  * <mark style="color:red;">LLFF</mark>
    * 8 scenes
  * Tanks and Temples
    * 4 scenes
  * Custom data collected by the author
    * 6 scenes
    * fish-eye camera
* **Experiments**
  * <mark style="color:red;">Improve over NeRF</mark>
  * Improve over NeRF++
  * Fish-eye Lens Reconstruction
  * <mark style="color:red;">Ablation Study</mark>

In this article, only the dataset and experiment highlighted in red will be covered.

### Improvement over NeRF

![](../../.gitbook/assets/2022spring/35/table1.png) ![](../../.gitbook/assets/2022spring/35/W400\_table2.png)

We train our model from scratch to demonstrate that our model can self-calibrate the camera information. We initialize all the rotation matrices, the translation vectors, and focal lengths to an identity matrix, zero vector, and height and width of the captured images. Table 1 reports the qualities of the rendered images in the training set. Although our model does not adopt calibrated camera information, our model shows a reliable rendering performance. Moreover, for some scenes, our model outperforms NeRF, trained with COLMAP \[16] camera information. We have visualized the rendered images in Figure 7.

![](../../.gitbook/assets/2022spring/35/figure7.png)

### Ablation Study

![](../../.gitbook/assets/2022spring/35/table5.png)

To check the effects of the proposed models, we conduct an ablation study. We check the performance for each phase in curriculum learning. We train 200K iterations for each phase. From this experiment, we have observed that extending our model is more potential in rendering clearer images. However, for some scenes, adopting projected ray distance increases the overall projected ray distance.\\

![](../../.gitbook/assets/2022spring/35/figure8.png)

## 5. Conclusion

### Summary

SCNeRF proposes a self-calibration algorithm that learns geometry and camera parameters jointly end-to-end. The camera model consists of a pinhole model, radial distortion, and non-linear distortion, which capture real noises in lenses. We also propose projected ray distance to improve accuracy, which allows our model to learn fine-grained correspondences. We show that our model learns geometry and camera parameters from scratch when the poses are not given, and our model improves both NeRF to be more robust when camera poses are given.

### Personal Opinion

* In my perspective, this paper is worthy because it shows a way to calibrate camera parameters and neural radiance fields jointly.
* I wonder why the result in the paper reports training set accuracy instead of val/test set accuracy.
* Some errors are noticed in equations and corrected as I think they should be. Please feel free to comment if you find any errors in the equations used in this article.

### Take home message

> SCNeRF learns geometry and camera parameters from scratch w/o poses
>
> SCNeRF uses the camera model consists of a pinhole model, radial distortion, and non-linear distortion
>
> SCNeRF proposed projected ray distance to improve accuracy

## Reviewer information

None

## Reference & Additional materials

1. &#x20;Citation of this paper
   1. &#x20;Jeong, Yoonwoo, et al. "Self-calibrating neural radiance fields." _Proceedings of the IEEE/CVF International Conference on Computer Vision_. 2021.
   2. [https://arxiv.org/abs/2108.13826](https://arxiv.org/abs/2108.13826)
2. &#x20;Official Project Page : [https://postech-cvlab.github.io/SCNeRF/](https://postech-cvlab.github.io/SCNeRF/)
3. &#x20;Official GitHub repository : [https://github.com/POSTECH-CVLab/SCNeRF](https://github.com/POSTECH-CVLab/SCNeRF)
4. &#x20;Citation of related work
   1. Mildenhall, Ben, et al. "Nerf: Representing scenes as neural radiance fields for view synthesis." _European conference on computer vision_. Springer, Cham, 2020.
   2. Schops, Thomas, et al. "Why having 10,000 parameters in your camera model is better than twelve." _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_. 2020.
   3. Zhou, Yi, et al. "On the continuity of rotation representations in neural networks." _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_. 2019.
5. Other useful materials
   1. Lens Aberrations : chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/viewer.html?pdfurl=https%3A%2F%2Fwww.nao.org%2Fwp-content%2Fuploads%2F2020%2F04%2FLens-Aberrations.pdf\&chunk=true
   2. camera models : chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/viewer.html?pdfurl=https%3A%2F%2Fcvgl.stanford.edu%2Fteaching%2Fcs231a\_winter1415%2Flecture%2Flecture2\_camera\_models\_note.pdf\&clen=4519068\&chunk=true

