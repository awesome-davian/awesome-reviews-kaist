---
description: Yu et al. / pixelNeRF - Neural Radiance Fields from One or Few Images / CVPR 2021
---

# pixelNeRF \[Eng]

[**한국어**](cvpr-2021-pixelnerf-kor.md)로 쓰인 리뷰를 읽으려면 여기를 누르세요.

### 1. Introduction

Today I'll introduce the paper [PixelNeRF: Neural Radiance Fields from one or few images](https://arxiv.org/abs/2012.02190), kind of follow-up paper of [**NeRF**(ECCV 2020)](https://arxiv.org/abs/2003.08934) that achieves great performance on **view synthesis** area. 

#### Problem Definition: View Synthesis

* View synthesis is a problem of reconstructing photo from new angle using multiple photos taken from other angles. When we take pictures, 3D objects in the real world are recorded as two-dimensional images. In the process, some information regarding the depth or amount of light received from the object is lost. 
* So to generate a image from new angle, we should infer and restore the rest of the information about model real-world objects based on given (limited) information. This problem is very difficult to solve because it is not just possible to create an image from a new angle by interpolating a given image, but also requires consideration of a variety of external factors.

* To date, NeRF is a SOTA algorithm on view synthesis task and get a lot of attention for its great performance.

### 2 Motivation

Before we look into pixelNeRF, let me explain more about NeRF and other related studies to find out which points pixelNeRF tried to improve.

#### 2.1 Related Work

#### NeRF

NeRF is a model for the task of restoring "light and perspective" from 2D images taken using a view synthesis, that is, a camera, to create 2D images of objects from a new angle. At this time, creating a 2D image from a new angle can mean modeling the entire 3D object. In this modeling, NeRF uses a "function that computes the RGB value of each pixel given its coordinate," called natural radius field ($$\approx$$ neural implicit reprsentation). The function here is defined as a deep natural network and can be expressed as the formula below.

(3D objects are very sparse unlike 2D iamges, so neural implicit representation kind of method is more efficient than computing RGB values into discrete matrices. Not only for 3D object reconstructing, the method is widely used in various CV fields such as super-resolution.)

$$
F_\Theta: (X,d) \rightarrow (c,\sigma)
$$

* Input: position of pixel $$X \in \mathbb{R}^3$$ and the viewing direction unit vector $$d \in \mathbb{R}^2$$
* Output: color value $$c$$ and density value $$\sigma$$

Then, How can we render a new image from the color and density values from $$F_\Theta$$ ?

The color values computed above mean the RGB value at 3D coordinates. At this time, in order to create a 2D image from a different angle, it is necessary to consider whether the 3D object is covered by the front part (from that viewing direction), or the back part is reflected. That's why we need to compute the density value through the function.
Considering all of these, the equation of converting RGB values in three dimensions into RGB values in 2D images is as follows.

$$
\hat{C}_r=\int_{t_n}^{t_f} T(t)\sigma(t)c(t)dt\
$$

**Notation**

* camera ray $$r(t)=o+td$$
  * $$t$$: how far the object is (from the focus)
  * $$d$$: viewing direction unit vector
  * $$o$$: origin
*   $$T(t)=exp(-\int_{t_n}^t\sigma(s)ds)$$

    : summation of the density values of the points blocking point $$t$$. ($$\approx$$ The probability that light rays will move from $$t_n$$ to $$t$$ without hitting other particles.)
    
* $$\sigma(t)$$ : density value on point $$t$$
* $$c(t)$$: RGB value on point $$t$$

The training proceeds by calculating the loss by the difference between the estimated RGB value $$\hat{C}_r$$ and the actual RGB value $$C(r)$$.

$$
\mathcal{L}=\Sigma_r ||\hat{C}_r -C(r)||^2_2
$$

It is possible to optimize by gradient descent algorithm because every process is differentiable.

![](/.gitbook/assets/19/figure2.png)

To summarize one more time through the figure, (a) extract three-dimensional coordinates (x, y, z) and direction d from the 2D image. _(The extraction process follows the author's previous study, [_LLFF_](https://arxiv.org/pdf/1905.00889.pdf))_ (b) After that, the color and density values at each coordinate are obtained using the natural radius field function. (c) Rendering the three-dimensional volume into a two-dimensional image through the equation described above. (d) Compare the RGB value at each coordinate with ground truth to optimize the function.



> This is the explanation of NeRF to understand this paper. If you think it is not enough, please refer to the [link](https://www.youtube.com/watch?v=CRlN-cYFxTk). :)

####

#### View synthesis by learning shared priors

There have been various studies using learned priors for the few-shot or single-shot view synthesis before pixelNeRF. 

![](/.gitbook/assets/19/figure3.png)

However, most of them uses 2.5 dimension of data, not 3 dimension, or just uses traditional methods (like estimating depth using interpolation). There are also several limitations in modeling 3D objects, such as requiring information about the entire 3D object (not 2D images) or considering only the global feature of the image. Furthermore, most 3D learning methods use an object-centered coordinate system that aligns only in a certain direction, which has the disadvantage of being difficult to predict. 
Pixel NeRF improved the performance of the model by supplementing these shortcomings of existing methodologies.

#### 2.2 Idea

It is a NeRF that made a huge wave on view synthesis fields with its high performance, but there are also limitations. In order to reconstruct high-quality images, multiple angles of images are needed for one object, and it takes quite long time to optimize the model. The pixel NeRF complements these limitations of NeRF and proposes a way to create images from a new point of view with only a small number of images in a much shorter time.

In order to be able to create a plausible image with only a small number of images, the model must be able to learn the **spatial relationship** of each scene. To this end, pixelNeRF extracts the spatial features of the image and uses it as input. At this time, the feature is a fully convolutional image feature.

As shown in the figure below, you can see that pixel NeRF produces great results even for fewer input images compared to NeRF.

![](/.gitbook/assets/19/figure1.png)

### 3. Methods

Then let's move on to the PixelNeRF. The structure of the model can be largely divided into two parts.

* fully-convolutional image encoder $$E$$ : encoding input images into the pixel-aligned features
* NeRF network $$f$$ : computing the color and density values of object

The output of encoder $$E$$ goes into the input of nerf network. 

#### 3.1 Single-Image pixelNeRF

This paper introduces the method by dividing Pixelnerf into single-shot and multi-shot. 
First of all, let's take a look at Single-image pixel NeRF.

**Notation**

* $$I$$: input image
* $$W$$: extracted spatial feature $$=E(I)$$
* $$x$$: camera ray
* $$\pi(x)$$: image coordinates
* $$\gamma(\cdot)$$ : positional encoding on $$x$$
* $$d$$: unit vector about viewing direction

![](/.gitbook/assets/19/figure4.png)

1. Extract the spatial feature vector W by putting input image $$I$$ into the encoder $$E$$. 
2. After that, for the points on camera ray $$x$$, we obtain the each corresponding image feature.
   * Project the camera ray $$x$$ onto image plane and compute the corresponding image coordinate $$\pi(x)$$.
   * Compute corresponding spatial feature $$W(\pi(x))$$ by using bilinear interpolation.
3. Put the $$W(\pi(x))$$ and $$\gamma(x)$$ and $$d$$ in the NeRF network and obtain the color and density values. (-> Nerf network)

$$
f(\gamma(x),d;W(\pi(x)))=(\sigma,c)\
$$

4\. Do volume rendering in the same way as NeRF.

> That is the main difference with NeRF is that the feature of the input image is extracted through pre-processing and added to the network.
Adding (spatial) feature information allows the network to learn implicit relationships between individual information in units of pixels, which allows stable and accurate inference with less data.

#### 3.2 Multi-view pixelNeRF

In the case of the Few-shot view synthesis, multiple photos come in, so we can see the importance of a specific image feature through the query view direction. If the input and target direction are similar, the model can be inferred based on the input, otherwise you will have to utilize the existing learned prior.

The basic framework of the multi-view model structure is almost similar to the single-shot pixel NeRF, but there are some points to consider because of additional images.

1. First of all, in order to solve the multi-view task, it is assumed that we can know the relative camera location of each image.
2. Then we transform the objects coordimate (located at origin in each image $$I^{(i)}$$) to match the coordinates at the target direction we want to see.

    $$P^{(i)} = [R^{(i)} \; t^{(i)}], \ x^{(i)}= P^{(i)}x$$, $$d^{(i)}= R^{(i)}d$$
    
3. When extracting features through encoder, select them independently for each view frame, and put them in the NeRF network. In the final layer of the NeRF networ, we combine them. This is for extracting as many spatial features as possible from images from various angles.

   *   In the paper, to express this as an expression, we denote the initial layer of the NeRF network as $$f_1$$, the intermediate layer as $$V^{(i)}$$, and the final layer as $$f_2$$.

       $$
       V^{(i)}=f_1(\gamma(x^{(i)}),d^{(i)}; W^{(i)}(\pi(x^{(i)}))) \\\ (\sigma,c)= f_2 (\psi(V^{(i)},...,V^{(n)}))\
       $$

       * $$\psi$$: average pooling operator

> Single-view pixel NeRF is the simplified version of multi-view pixel NeRF.

### 4. Experiments

**Baselines & Dataset**

* Comparing pixelNeRF with SRN and DVR, which were SOTA models of the existing few-shot view synthesis, and NeRF, which uses similar networks structures.
* Experiments are conducted on ShapeNet, a benchmark dataset for 3D objects, and DTU datasets that resemble more real photos, showing the performance of pixelNeRF.

**Metrics**

For the performance indicator, widely used image qualifying metrics(PSNR, SSIM) are used. 

* PSNR: $$10 log_{10}(\frac{R^2}{MSE})$$
  - It is used for evaluating information loss on image quality as a ratio of noise to a maximum signal that may have.
  - $$R$$: maximum value of the certain image
* SSIM: $$\frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+ \mu_y^2+ C_1)(\sigma_x^2+\sigma_y^2+C_2)}$$
  - Based on the assumption that the degree of distortion of image structure information has great influence on image quality, it is a metric designed for evaluating perceptual image differences, not numerical errors.
  - Intuitively speaking, It can be computed as Luminance x contrast x correlation coefficient between two images. 

**Training setup**

In the experiment of this paper, the reset34 model pretrained on the imagenet is used as the backbone network. Feature is extracted up to the $$4^{th}$$ pooling layer, and after that, the layer goes through the process of finding a feature that fits the corresponding coordinates (as described in 3 above). To use both local and global features, we extract features in the form of feature pyramid. (feature pyramid refers to a form in which feature maps of different resolutions are stacked.)

NeRF network $$f$$ also use the ResNet structure, putting the coordinate and viewing direction ($$\gamma(x), d$$) as inputs first and then add feature vector $$W(\phi(x))$$ as a residual at the beginning of each ResNet block.


In the paper, hree major experiments are conducted and shows the performance of pixel NeRF well.

1.  Evaluating pixelNeRF on category-specific and category-agnostic view synthesis task on ShapeNet. 

    ![](/.gitbook/assets/19/figure5.png) ![](/.gitbook/assets/19/figure6.png)
    
A single pixelNerF model is trained on the largest 13 cateogries of shapenet. As can be seen from the above results, pixel NeRF shows SOTA results in terms of view synthesis. For both category-specific and category-agnostic setting, all create the most sophisticated and plausible images, while image performance measures PSNR and SSIM also show the highest figures.


2\. Through the learned prior, they have shown that view synthesis is also applicable to unseen categories or multi-object data in ShapeNet data.

![](/.gitbook/assets/19/figure7.png)

This is the result of training the pixelNeRF only for some categories (cars, airplanes, and chairs) and then conducting a view synthesis for other categories. As you can see, the performance of pixelNeRF is also good for unseen categories. The author explains that these generalization is possible because the camera's relative position (view space) was used, not the canonical space.

3\. View synthesis for real scene data such as DTU MVS dataset

The model can reconstruct the real scene data from different angle as well as limited object pictures like shapenet. Even if the experiment is conducted based on only 88 learning image scenes, compared to NeRF, images from various angles are created very well as below.

![](/.gitbook/assets/19/figure8.png)

According to these experiments, it is proven that the pixelNeRF can be applied to not only standard 3d object images but also more general cases such as multi-object image, unseen image, real scene image. In addition, it seemed that all of these processes are possible with much fewer images than the vanilla NeRF.

### 5. Conclusion

In order to solve the view synthesis task well with only a small number of images, the pixel NeRF complements the limitations of existing view synthesis models, including NeRF, by adding a process of learning spatial feature vectors to the existing NeRF. In addition, the experiments have shown that pixelNeRF works well in various generalized environments (multi-objects, unseen categories, real data etc.).

But there are still some limitations. Like NeRF, rendering takes a very long time, and it is scale-variant because parameters for ray sampling boundaries or positional encoding need to be manually adjusted. In addition, experiments on DTU showed potential applicability to real images, but since this dataset was created in a limited situation, it is not yet guaranteed that it will perform similarly on real raw datasets.

Nevertheless, I think it is a meaningful study in that it has improved the performance of NeRF, which is currently receiving a lot of attention, and expanded it to a more generalized setting. If you have any questions after reading this review of the paper, please feel free to contact me :)


#### Take home message

* Recently, studies that model real objects with only 2D images and show them from various angles have been actively conducted.
* In particular, studies using natural impression presentation are attracting much attention.
* If you extract spatial feature and use them with the pixel values in a given image, you can perform much better (both in terms of reconstruction and efficiency).


### 6. Author

**권다희 (Dahee Kwon)**

* KAIST AI
* Contact
  * email: [daheekwon@kaist.ac.kr](mailto:daheekwon@kaist.ac.kr)
  * github: [https://github.com/daheekwon](https://github.com/daheekwon)

### 7. Reference & Additional materials

* [NeRF paper](https://arxiv.org/abs/2003.08934)
* [NeRF explanation video](https://www.youtube.com/watch?v=CRlN-cYFxTk)
* [pixelNeRF official site](https://alexyu.net/pixelnerf/)
* [pixelNeRF code](https://github.com/sxyu/pixel-nerf)

