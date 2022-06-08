---
description: Niemeyer et al. / GIRAFFE] Representing Scenes as Compositional Generative Neural Feature Fields / CVPR 2021 (oral, best paper award)
---

# GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields \[Eng\]

한국어로 쓰인 리뷰를 읽으려면 **[여기](cvpr-2021-GIRAFFE-kor.md)** 를 누르세요.

##  1. Problem definition

Through Generative Adversarial Networks(GANs), people have succeeded in generating highly-realistc images and even learning disentangled representations without explicit supervision. However, operating in 2D domain had limitations due to the 3-dimensional nature of this world. Recent investigations started to focus on incorporating 3D representations using voxels or radiance fields, but was restricted to single object scenes and showed less consistent results in high resolution and more complex images. 

So this paper suggests incorporating a **compositional** 3D scene representation into the **generative** models, leading to more controllable image synthesis.

## 2. Motivation

### Related work

**Implicit Neural Representation (INR)**

Existing Neural Networks(NN) played a role in prediction tasks(e.g. image classification) and generation (e.g. generative models). However, in INR, network parameter contains the information of the data iteself, so the network size is proportional to the complexity of the data, which is especially beneficial for representing 3D scenes. In addtion, as we learn a function, for example that maps the coordinate to RGB value for a single image, this enables us to represent the data in a continuous form. 

- **NeRF : Neural Radiance Field** $$\quad$$ $$f_{\theta}:R^{L_x}\times R^{L_d}\to R^+ \times R^3$$ $$(\gamma(x),\gamma(d)) \to (\sigma, c)$$

    A scene is represented using a fully-connected network, whose input is a single continous 5D coordinate (position + direction) that goes through positional encoding $$\gamma(x)$$ for higher dimensional information, and outputs the volume density and view-dependent RGB value (radiance). 5D coordinates for each direction(camera ray) $$r(t)$$ are sampled, and the produced color $$c(r(t),d)$$ and density  $$\sigma(r(t))$$ composites an image through volume rendering technique (explained in section 3). As a loss function, the difference between the volume redered image and the ground truth *posed* image is used.
    
 ![Figure 1: NeRF architecture](/.gitbook/assets/2022spring/47/NeRF.PNG)
 
- **GRAF : Generative Radiance Field** $$\quad$$ $$f_{\theta}:R^{L_x}\times R^{L_d}\times R^{M_s}\times R^{M_a} \to R^+\times R^3$$ $$(\gamma(x),\gamma(d),z_s,z_a) \to (\sigma, c)$$
    
    It proposes an adversarial **generative** model for **conditional** radiance fields, whose input is a sampled camera pose &epsilon; (sampled from upper hemisphere facing origin, uniformly) and a sampled K x K patch, whose center is $$(u,v)$$ and has scale $$s$$ , from the *unposed* input image. As a condition, shape $$z_s$$ and apperance $$z_a$$ code is added, and the model (fully connected network with ReLU activations) outputs a predicted patch, just like the NerF model. Next, the discriminator (convolutional neural network) is trained to distinguish between the predicted patch and real patch sampled from an image in the image distribution.
    
![Figure 2: GRAF architecture](/.gitbook/assets/2022spring/47/GRAF.PNG)

### Idea

GRAF achieves controllable image synthesis at high resolution, but is restricted to single-object scenes and the results tend to degrade on more complex imagery. So this paper propose a model that is able to disentangle individual objects and allows translation and rotation as well as changing the camera pose. 

## 3. Method
![Figure 3: GIRAFFE architecture](/.gitbook/assets/2022spring/47/GIRAFFE.PNG)


- **Neural Feature Field** : Replaces GRAF’s formulation for 3D color output c with $$M_f$$ -dimensional feature <br />
$$h_{\theta}:R^{L_x} \times R^{L_d} \times R^{M_s} \times R^{M_a} \to R^+ \times R^{M_f}$$ $$(\gamma(x),\gamma(d),z_s,z_a) \to (\sigma, f)$$
    
    **Object Representation** In NerF and GRAF, entire scene is represented by a single model, but in order to disentange different entities in the scene, GIRAFFE represents each object (in the scene) using a separate feature field in combination with affine transformation. The parameters 
        $$T=\{s,t,R\}$$    ($$s$$:scale, $$t$$: translation, $$R$$: rotation) are sampled from dataset-dependent distribution  <br/>
        $$k(x)=R\cdot\begin{bmatrix} s_1 & & \\  & s_2 &\\ & & s_3 \end{bmatrix}\cdot x + t$$  <br/>
         Therefore, it gains control over the pose, shape and appearance of individual objects. <br/>
         Then, through volume rendering, we can create 2D projection of a 3D discretly sampled dataset. <br/>
        $$(\sigma,f)=h_{\theta}(\gamma(k^{-1}(x)),\gamma(k^{-1}(d)),z_s,z_a)$$  <br />  
    **Composition Operator** A scene is described as compositions of N entities (N-1 objects and 1 background). The model uses density-weighted mean to combine all features at $$(x,d)$$ 
    
    $$C(x,d)=(\sigma,{1\over\sigma} \sum_{i=1}^{N}\sigma_if_i), \quad where \quad \sigma = \sum_{i=1}^N\sigma_i$$ <br />
    **3D volume rendering** Unlike previous models that volume render an RGB color value, GIRAFFE renders an $$M_f$$-dimensional feature vector $$f$$. Along a camer ray $$d$$, the model samples $$N_s$$ points and the operator $$\pi_{vol}$$ maps them to the final feature vector $$f$$. <br/>
        $$\pi_{vol} : (R^+ \times R^{M_f})^{N_s} \to R^{M_f}$$  <br/>
        They use the same numerical intergration method as in NeRF. <br/>
        $$f=\sum_{j=1}^{N_s}\tau_i\alpha_i f_i \quad \tau_j=\prod_{k=1}^{j-1}(1-\alpha_k) \quad \alpha_j=1-e^{-\sigma_i\delta_j}$$ <br/>
        Where, $$\delta_j=||x_{j+1} - x_j ||_2$$ is the distance between neighboring sample points and with density $$\delta_j$$, it defines the alpha value $$\alpha_j$$. By accumulating the alpha values, we can compute the transmittance $$\tau_j$$. The entire feature image is obtained by evaluation $$\pi_{vol}$$ at every pixel. <br/>
        For efficiency, they obtain a feature map of $$16^2$$ resolution, which is lower than the input resolution($$64^2$$ or $$256^2$$ pixels).  
        
- **2D neural rendering** In order to upsample the feature map to a higher-resolution image, the paper use 2D neural rendering as the figure below. <br/>
    $$\pi_\theta^{neural}:R^{H_v \times W_v \times M_f} \to R^{H \times W \times 3}$$

![Figure 4: 2d neural rendering architecture](/.gitbook/assets/2022spring/47/2d%20neural%20rendering.PNG)
  
  
- **Training**  
    - Generator   
    
    $$G_\theta(\left\{z_s^i,z_a^i,T_i\right\}_{i=1}^N,\epsilon)=\pi_\theta^{neural}(I_v),\quad where \quad I_v=\{\pi_{vol}(\{C(x_{jk},d_k)\}_{j=1}^{N_s})\}_{k=1}^{H_v \times W_v}$$
    
    - Discriminator : CNN with leaky ReLU 
    
    - Loss Function = non-saturating GAN loss + R1-regularization 
    
    $$V(\theta,\phi)=E_{z_s^i,z_a^i \sim N, \epsilon \sim p_T} \[f(D_\phi(G_\theta\(\{z_s^i,z_a^i,T_i\}_i,\epsilon)\)\] + E_{I\sim p_D}\[f(-D_\phi(I))- \lambda\vert \vert\bigtriangledown D_\phi (I)\vert\vert^2 \] $$ 
    
    $$where \quad f(t)=-log(1+exp(-t)), \lambda=10 $$
 
## 4. Experiment & Result


### Experimental setup

- DataSet
    - commonly used single object dataset: Chairs, Cats, CelebA, CelebA-HQ 
    - challenging single-object dataset: CompCars, LSUN Churches, FFHQ
    - testing on multi-object scenes: Clevr-N, Clevr-2345
- Baseline
    - voxel-based PlatonicGAN, BlockGAN, HoloGAN
    - radiance field-based GRAF
- Training setup
    - number of entities in the scene $$N \sim p_N$$, latent codes $$z_s^i,z_a^i \sim N(0,I)$$
    - camera pose $$\epsilon \sim p_{\epsilon}$$, transformations $$T_i \sim p_T$$ </br>
        ⇒ In practice, $$p_{\epsilon}$$ and $$p_T$$ is uniform distribution over data-dependent camera elevation angles and valid object tranformations each.
    - All object fields share their weights and are paramterized as MLPs with ReLU activations ( 8 layers with hidden dimension of 128, $$M_f=128$$ for objects & half the layers and hidden dimension for background features)
    - $$L_x=2,3,10$$ and $$L_d=2,3,4$$ for positional encoding
    - sample 64 points along each ray and render feature images at $$16^2$$ pixels 
- Evaluation Metric
    - Frechet Inception Distance (FID) score with 20,000 real and fake samples

### Result

- disentangled scene generation   

![Figure 5: disentanglement](/.gitbook/assets/2022spring/47/controllable.PNG)
    
- comparison to baseline methods   

![Figure 6: qualitative comparison](/.gitbook/assets/2022spring/47/qualitative%20comparison.PNG)
   
- ablation studies
    - importance of 2D neural rendering and its individual components
    
    ![Figure 7: neural rendering architecture ablation](/.gitbook/assets/2022spring/47/ablation.PNG) <br/>
        Key difference with GRAF is that GIRAFFE combines volume rendering with neural rendering. This method helps the model to be more expressive and better handle the complex real scenes. Furthermore, rendering speed is increased compared to GRAF(total rendering time is reduced from 110.1ms to 4.8ms, and from 1595.0ms to 5.9ms for $$64^2$$ and $$256^2$$ pixels, respectively.)
        
    - positional encoding
        
      $$r(t,L) = (sin(2^0t\pi), cos(2^0t\pi),...,sin(2^Lt\pi),cos(2^Lt\pi))$$ 
      
      ![Figure 8: positional encoding](/.gitbook/assets/2022spring/47/positional%20encoding.PNG)
        
- limitations
    - struggles to disentangle factors of variation if there is an inherent bias in the data. (eg. eye and hair translation)
    - disentanglement failures due to mismatches between assumed uniform distribution (camera poses and object-lebel transformations) and their real distributions
    
    ![Figure 9: limitation_disentangle failure](/.gitbook/assets/2022spring/47/disentanglement%20failure.png)
   
## 5. Conclusion

⇒ By representing scenes as compostional generative neural feature fields, they disentangle individual objects from the background as well as their shape and appearance without explicit supervision

⇒ Future work
- Investigate how the distributions over object level transformations and camera poses can be learned from data
- Incorporate supervision which is easy to obtain (eg. object mask) -> scale to more complex, multi-object scenes


### Take home message \(오늘의 교훈\)

- 3D scene representation through Implicit Neural Representation is a recent trend that shows superior result.
- Using individual feature field for each entitiy helps disentangle their movements.
- Rather than limiting the features to its original size (coodinate : 3, RGB : 3), using positional encoding or neural rendering help represent the information more abundantly.
## Author / Reviewer information

{% hint style="warning" %}
You don't need to provide the reviewer information at the draft submission stage.
{% endhint %}

### Author

**김소희(Sohee Kim)** 

* KAIST AI
* Contact: joyhee@kaist.ac.kr

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. [GIRAFFE paper](https://arxiv.org/abs/2011.12100)
2. [GIRAFFE supplementary material](http://www.cvlibs.net/publications/Niemeyer2021CVPR_supplementary.pdf)
3. [GIRAFFE - Github ](https://github.com/autonomousvision/giraffe)
4. [INR explanation](https://www.notion.so/Implicit-Representation-Using-Neural-Network-c6aac62e0bf044ebbe70abcdb9cc3dd1)
5. [NeRF paper](https://arxiv.org/abs/2003.08934)
6. [GRAF paper](https://arxiv.org/abs/2007.02442)

