---
description: Pumarola et al. / D-NeRF; Neural Radiance Fields for Dynamic Scenes / CVPR 2021
---

#  D-NeRF \[Eng\]

********************

This article reviews in the paper titled "D-NeRF: Neural Radiance Fields for Dynamic Scenes" in detail, while trying to explain some non-obvious key points more clearly.

##  1. Problem definition

The authors of the paper are trying to extend the Neural Radiance Fields (NeRF) -technology to include temporally changing scenes. They call the method Dynamic Neural Radiance Fields (D-NeRF). On top of expanding the NeRF method to dynamic scenes, one of the key points of the paper is that this can be done using one image at time, and multiple images from one time-instance are not needed. 

NeRF is a deep neural networking method for image synthesis. The general idea train the model with images of the same object from different 3D-angles. The goal is to overfit the model to the given object. The overfitted model can subsequently be used to generate images of the object from novel 3D-angles. The training for traditional NeRF is done from a 5D-input vector of spatial coordinate **x** (x, y, z) and viewing angle **d**  ($$\theta$$, $$\phi$$) to 4D-output vector of color **c** (r, g, b) and volume density $$\sigma$$. D-NeRF adds to this by adding one dimension, time t, to input vector, making the model function (**x**, **d**, t) &#8594; (**c**, $$\sigma$$).

<img src="/.gitbook/assets/2022spring/26/exampleNERFwork.png" alt="example" class="bg-primary mb-1" width="3200px">

*General idea behind NeRF visualized. Images from different angles are used to train the NeRF model. Afterwards novel viewing angles can be produced from the model. Image from original NeRF paper Mildenhall et al. (2020)*



## 2. Motivation

The authors were trying to expand the promising technology of NeRF into moving images. While academically very interesting, the method could obviously have immediate applications in the field of any type of image rendering. While the training of the model into even one object/scene is computationally expensive, the final trained model is very small in memory. As the 3D model of the object is readily extractable from the trained model, could NeRF-models potentially be used in fields where 3D-models of objects are crucial (etc. video games).  The D-NeRF could be used to extend this idea to make it possible to produce multiple (in different time-instances) 3D-reconstructions of an object. The method could also be used as a novel way of image rendering for demonstration purposes for (for example) real life factory machinery.  

### Related work

The pioneering work of Mildenhall et al. (NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, 2020) introduces a novel way of image synthesis from different viewing angles. As a pioneering work of NeRF, it introduces the method of overfitting the object-function (**x**, **d**) &#8594; (**c**, $$\sigma$$) to allow image synthesis from angles the model was not originally trained for. The method can also be used to view objects in different lighting-environments, if **d** for viewing angle is switched to mean the angle of the lighting (i.e. training the model with varying lighting rather than varying viewing angle). While the paper succesfully demonstrated the effectiveness of NeRF, it left some room for improvements and new features, such as expansion to dynamic scenes. 

The paper "NeRF in the Wild" by Martin-Brualla et al. (CVPR, 2021) expands the idea of NeRF to be usable with pictures taken at, for example, different times of day and with varying obstructions. The model then separates the image specific dynamic elements so that the training can be done solely on the static elements (such as buildings). The model showed great results on highly photographed objects, such Sacre Ceour, and readily introduced object viewing in different lighting conditions, such as at night. The model, however, is not capable of rendering images of moving objects at different times, like D-NeRF is. 

### Idea

As stated before, the D-NeRF paper expands the NeRF-model to include dynamic scenes, by adding a new input dimension t into the training dataset. Previous work in NeRF would produce highly blurry images with dynamic scenes, because modeling a function with 6D input values (a moving 3D object with 2 degrees of freedom in viewing angle is) using only 5D model is impossible. This was demonstrated by the paper, proving that D-NeRF is superior in training for moving objects. 

## 3. Method

In the paper the neural networks are used to map a 6D input vector consisting of 3D-point **x** (x, y, z), viewing angle **d** ($$\theta$$, $$\phi$$) and time (t) to 4D-output vector of color **c** (r, g, b) and volume density $$\sigma$$. Here, volume density is a property depending solely on **x**, while the color **c** can depend also on the viewing angle **d**. This quarantees that the objects conserve their shape in different viewing angles, but can show different color in different viewing angles (for example due to lighting).

In the paper they  claim that better results are found when training was done with two different neural networks. The first network ($$\Psi_t$$) is used to deform the 3D-points **x**  to given time t. $$\Psi_t$$ deforms the object/3D-space points from canonical scene t=0 to given t by adding deformation $$\Delta$$**x**  to the canonical scene points **x** . The network itself works as a function (**x**, t) &#8594; ($$\Delta$$**x**). The second neural network ($$\Psi_x$$) is used to produce the novel scenes from the deformed points gained from $$\Psi_t$$. $$\Psi_x$$ therefore works somewhat like neural architectures in regular NeRF.  It works as a function (**x**(t), **d**) &#8594; (**c**, $$\sigma$$), where **x**(t) = **x** + $$\Delta$$**x** is the deformed set of points from $$\Psi_t$$. The model also uses encoders to achieve better performance. The encoder heighten the dimensionality of the inputs by a factor of 2. Specifically the encoder is applied to every input value as  $$\gamma(p)$$ = ($$\sin(2^l\pi p$$), $$\cos(2^l\pi p$$)), where $$\gamma$$ is the encoder, $$p$$ is the input value (such as t or y). For $$l$$, values of 10 (**x**) and 4 (**d**, t) were used.

<img src="/.gitbook/assets/2022spring/26/networks.png" alt="networks" class="bg-primary mb-1" width="3200px">

*Example of how the two different neural networks produce the final synthesised image.* 



The image rendering is done by integrating over all the points **x** which are in the straight ray path for the given pixel (in the given scene **d**).
$$
C(p, t)=\int_{h_{n}}^{h_{f}} \mathcal{T}(h, t) \sigma(\mathbf{p}(h, t)) \mathbf{c}(\mathbf{p}(h, t), \mathbf{d}) d h \\
\text { where } \mathbf{p}(h, t)=\mathbf{x}(h)+\Psi_{t}(\mathbf{x}(h), t), \\
{[\mathbf{c}(\mathbf{p}(h, t), \mathbf{d}), \sigma(\mathbf{p}(h, t))]=\Psi_{x}(\mathbf{p}(h, t), \mathbf{d})} \\
\text { and } \mathcal{T}(h, t)=\exp \left(-\int_{h_{n}}^{h} \sigma(\mathbf{p}(s, t)) d s\right)
$$


Here, $$C$$ is the color of the pixel $$p$$ in the render frame t. The integrated variable $$h$$ is the distance from the viewing point, and $$h_n$$ and $$h_f$$ are the near and far bounding edges for integrating. $$\textbf{p}$$ are the points in time t (where deformation from $$\Psi_t$$ is already applied) $$\mathcal{T}$$ is calculates the visibility of the given point. If the integral in the expression for $$\mathcal{T}$$ is large, it means that some points are in the way of the current point $$h$$ obscuring it. This point won't therefore be visible in the rendering. $$\mathcal{T}$$ also allows for the image to have transparent objects (like windows), since it can allow partial visibility for objects behind translucent points. 

## 4. Experiment & Result

### Experimental setup

Both $$\Psi_t$$ and $$\Psi_x$$ are quite simple architectures consisting of only 8-layers. ReLU-activations were used in each layer, and a final sigmoid nonlinearity to outputs from $$\Psi_x$$. As an optimizer, Adam was used with with learning rate of 5e − 4, β1 = 0.9, β2 = 0.999 and exponential decay to 5e − 5. To improve convergence, the networks were trained by gradually using training images with more and more deformation from the canonical image t = 0. A batch size of 4096 rays over 800 000 iterations was used. The weights in both networks were learned simultaneously by using the mean squared error between the rendered and the real pixels as a loss function.

The dataset consisted of 400x400 pixel images. They used only computer generated moving objects, of which they have provided 8 datasets, pretrained models, results etc. online. Training dataset consisted of 50 to200 images (depending on the dataset), each from an unique time/deformation and angle. To control training, 20 test images were used. For one object, the training of the model takes 2 days with a single Nvidia® GTX 1080.  

<img src="/.gitbook/assets/2022spring/26/r_comparison.png" alt="r_comparison" class="bg-primary mb-1" width="2400px">

*Example of a training images used in the paper. All the images were similar, computer generated of an existing model. The two images approximately show the range of the movement/deformation used in the paper, i.e. they were quite simple movements such as raising a hand, or squatting. The image on the left was the 148th of the 150 training images, and the image on right was 8th. These images were chosen here for presentative purposes due to most other pictures were from different angles, making comparison in different timeframes difficult. The images were fetched from the training dataset provided by the author.*

### Results

The results are show that the method adequately can be used to synthesise novel viewing angles at wanted timeframes of the trained object. Also, 3D representations of the object can also be extracted easily from the model.

<img src="/.gitbook/assets/2022spring/26/results1.png" alt="results" class="bg-primary mb-1" width="4000px">

*Synthesised images from trained models at different timeframes. The meshes and depths show the volume densities, which dictate the shape of the object. The column **x**+$$\Delta$$**x** shows the deformation information at different times. Color coding corresponds to same points which have been deformed. This shows that the deformation network has worked as intended, and for example, the points in a hand are deformed into hand, and not something else. Note that similar (RGB) images can be obtained from any angle ($$\theta$$, $$\phi$$)*, each which have natural looking lighting/coloring specific to that viewing angle. 

The D-NeRF was also compared with original NeRF, and the T-NeRF, which is the naive approach to the problem. In T-NeRF, the model the whole learning of the dynamic scenes was handled by a single neural network, which inputs a 6D-vector (**x**, **d**, t) and outputs the rendering parameters (**c**, $$\sigma$$). Recall, that this was handled by a two separate networks $$\Psi_t$$ and $$\Psi_x$$ in D-NeRF. The comparisons are found below. 



<img src="/.gitbook/assets/2022spring/26/comparisonNERFs.png" alt="comparisonOfNeRFs" class="bg-primary mb-1" width="4000px">

*In this figure, D-NeRF is compared to ground truth (GT),  original NeRF, and T-NeRF values. It can be clearly seen that D-NeRF provides the best results, almost indistinguishable of the real images. While T-NeRF provides adequate results, it is clear that dividing the model into two separate neural networks is a better solution. Original NeRF architecture is, unsurprisingly, unable to model moving objects clearly, and is blurry without exceptions.*

For systematic comparison between the three models (D-NeRF, T-NeRF, NeRF), four metrics were used. These are: Mean Squared Error (MSE), Peak Signal-toNoise Ratio (PSNR), Structural Similarity (SSIM) and Learned Perceptual Image Patch Similarity (LPIPS). The arrows next to the given metric in the following table show if the picture quality was better with lower ($$\downarrow$$) or higher ($$\uparrow$$) value of the metric. 

<img src="/.gitbook/assets/2022spring/26/metrics.png" alt="metrics" class="bg-primary mb-1" width="4000px">

*The performance of the different models with the used datasets. The best among the three in given category is highlighted in bold.  From the results, the conclusion can be drawn that D-NeRF was the best at almost all cases. T-NeRF, however, performed better in the "Lego" dataset. The reasons on why D-NeRF didn't perform better in this dataset is uknown.*



## 5. Conclusion

In conclusion, the paper provided a new method of training NeRF to include moderately dynamic scenes. The methodology was needed to provide these kind of results, as previous NeRF-papers only used static models, which don't take into account the possible deformations of the object. The paper, however, only demonstrated simple movements, not involving for example emergence of new, previously hidden points. 

The work on the paper could easily expanded to include such cases, I think. Also, I think that by adding new input dimensions (for each type of movement wanted), it could be possible to train a model which could render any kind of combination of the movements (for example, varying squatting with varying hand waving). This of course requires usage of bigger training datasets. It could however be useful in video games to make the possibilities of NPC movements practically infinite and more lifelike and varying. 

I think the paper was very successful in what it tried to achieve. Overall, a very successful and already famous paper expanding on the field of NeRF. One noticeable drawback the paper has is the usage of computer rendered training images. While generation of real photos would of course be much more difficult, would it provide much more interesting framework to work with. Difficulties in real photos include the smooth transition/somewhat constant deformation between timeframes, and the usage of different camera angles. 

### Take home message

> Generation of images of an object from continuous choice of angles at a continuous choice of phases of movement is possible using D-NeRF. 
>
> The model simultaneously provides the 3D shape of the object.  
>
> In theory, the FPS of a video could be infinitely increased by using D-NeRF (although there probably are better ways to do this) 

## Author / Reviewer information

### Author

**Jussi Kelavuori**

* KAIST / Tampere University, Department of Physics
* I'm an exchange student at KAIST from Finland for the spring semester 2022.
* contact: jussi.kelavuori at tuni.fi

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. Citation of the paper

   Albert Pumarola, Enric Corona, Gerard Pons-Moll, and
   Francesc Moreno-Noguer. D-nerf: Neural radiance fields for
   dynamic scenes. arXiv preprint arXiv:2011.13961, 2020

2. Official GitHub repository:  https://github.com/albertpumarola/D-NeRF

3. Images were from original paper, if not stated otherwise.
