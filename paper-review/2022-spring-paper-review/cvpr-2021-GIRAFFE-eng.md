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

- **NeRF : Neural Radiance Field** <img src = "https://latex.codecogs.com/svg.image?f_\theta:R^{L_x}&space;\times&space;R^{L_d}&space;&space;\to&space;R^&plus;&space;\times&space;R^3" />

    A scene is represented using a fully-connected network, whose input is a single continous 5D coordinate (position + direction) that goes through positional encoding for higher dimensional information, and outputs the volume density and view-dependent RGB value (radiance). 5D coordinates ![formula](https://render.githubusercontent.com/render/math?math=d) for each direction(camera ray) ![formula](https://render.githubusercontent.com/render/math?math=r(t)) are sampled, and the produced color <img src="https://latex.codecogs.com/svg.image?c(r(t),&space;d)" /> and density  ![formula](https://render.githubusercontent.com/render/math?math=\sigma(r(t))) composites an image through volume rendering technique. (explained in section 3) As a loss function, the difference between the volume redered image and the ground truth *posed* image is used.
 ![Figure 1: NeRF architecture](/.gitbook/assets/2022spring/47/NeRF.PNG)
 
- **GRAF : Generative Radiance Field** <img srf = "https://latex.codecogs.com/svg.image?f_\theta:R^{L_x}&space;\times&space;R^{L_d}&space;\times&space;R^{M_s}&space;\times&space;R^{M_a}&space;\to&space;R^&plus;&space;\times&space;R^3" />
    
    It proposes an adversarial **generative** model for **conditional** radiance fields, whose input is a sampled camera pose &epsilon;, (sampled from upper hemisphere facing origin, uniformly) and a sampled K x K patch, whose center is <img src="https://latex.codecogs.com/svg.image?(u,v)" /> and has scale <img src="https://latex.codecogs.com/svg.image?s" /> , from the input *unposed* image. As a condition, shape and apperance code is added, and the model (fully connected network with ReLU activations) outputs a predicted patch, just like the NerF model. Next, the discriminator (convolutional neural network) is trained to distinguish between the predicted patch and real patch sampled from an image in the image distribution.
![Figure 2: GRAF architecture](/.gitbook/assets/2022spring/47/GRAF.PNG)

### Idea

GRAF achieves controllable image synthesis at high resolution, but is restricted to single-object scenes and the results tend to degrade on more complex imagery. So this paper propose a model that is able to disentangle individual objects and allows translation and rotation as well as changing the camera pose. 

## 3. Method
![Figure 3: GIRAFFE architecture](/.gitbook/assets/2022spring/47/GIRAFFE.PNG)


- **Neural Feature Field** : Replaces GRAF’s formulation for 3D color output c with <img src ="https://latex.codecogs.com/svg.image?M_f" /> -dimensional feature

    <img src="https://latex.codecogs.com/svg.image?h_\theta:R^{L_x}&space;\times&space;R^{L_d}&space;\times&space;R^{M_s}&space;\times&space;R^{M_a}&space;\to&space;R^&plus;&space;\times&space;R^{M_f}" />
    
    
    **Object Representation** In NerF and GRAF, entire scene is represented by a single model, but in order to disentange different entities in the scene, GIRAFFE represents each object using a separate feature field in combination with affine transformation. Therefore, it gains control over the pose, shape and appearance of individual objects.  <br /> 
        <img src = "https://latex.codecogs.com/svg.image?T=\left\{s,t,R\right\}&space;" />     
        (<img src = "https://latex.codecogs.com/svg.image?s" />:scale, <img src= "https://latex.codecogs.com/svg.image?t" />: translation, <img src= "https://latex.codecogs.com/svg.image?R" />: rotation) sampled from dataset-dependent distribution  <br />  <br /> 
        <img src = "https://latex.codecogs.com/svg.image?k(x)&space;=&space;R&space;\cdot&space;\begin{bmatrix}&space;s_1&space;&&space;&&space;\\&space;&&space;s_2&space;&&space;\\&space;&&space;&&space;s_3&space;\end{bmatrix}&space;\cdot&space;x&space;&plus;&space;t" />  <br />      
        <img src = "https://latex.codecogs.com/svg.image?(\sigma,f)&space;=&space;h_{\theta}(\gamma(k^{-1}(x)),&space;\gamma(k^{-1}(d)),&space;z_s,&space;z_a)" />  <br />  
    **Composition Operator** A scene is described as compositions of N entities (N-1 objects and 1 background). The model uses density-weighted mean to combine all features at <img src="https://latex.codecogs.com/svg.image?(x,d)" /> <br />
    
    <img src = "https://latex.codecogs.com/svg.image?C(x,d)=(\sigma,&space;{1&space;\over&space;\sigma}\sum_{i=1}^{N}\sigma_if_i),&space;\,&space;where&space;\;&space;\sigma&space;=&space;\sum_{i=1}^N\sigma_i" />
        
    **3D volume rendering** Use numerical intergration as in NeRF.  <br />  <br /> 
        <img src = "https://latex.codecogs.com/svg.image?\pi_{vol}&space;:&space;(R^&plus;&space;\times&space;R^{M_f})^{N_s}&space;\to&space;R^{M_f}" />  <br />  <br />
        <img src = "https://latex.codecogs.com/svg.image?f=\sum_{j=1}^{N_s}\tau_i\alpha_if_i&space;\quad&space;\tau_j=\prod_{k=1}^{j-1}(1-\alpha_k)&space;\quad&space;\alpha_j=1-e^{-\sigma_i\delta_j}" />
        
- **2D neural rendering**
    <img src="https://latex.codecogs.com/svg.image?\pi_\theta^{neural}&space;:&space;R^{H_v&space;\times&space;W_v&space;\times&space;M_f}&space;\to&space;R^{H&space;\times&space;W&space;\times&space;3}" />

![Figure 4: 2d neural rendering architecture](/.gitbook/assets/2022spring/47/2d%20neural%20rendering.PNG)
  
  
- **Training**  <br /> 
    - Generator   <br /> 
    <img src = "https://latex.codecogs.com/svg.image?G_\theta(\left\{z_s^i,z_a^i,T_i\right\}_{i=1}^N,&space;\epsilon)=\pi_\theta^{neural}(I_v),\quad&space;where&space;\quad&space;I_v=\left\{\pi_{vol}(\left\{C(x_{jk},d_k)\right\}_{j=1}^{N_s})\right\}_{k=1}^{H_v&space;\times&space;W_v}" />
    
    - Discriminator : CNN with leaky ReLU <br /> <br />
    - Loss Funcion = non-saturating GAN loss + R1-regularization <br />
    <img src = "https://latex.codecogs.com/svg.image?V(\theta,&space;\phi)=E_{z_s^i,&space;z_a^i&space;\sim&space;N,&space;\epsilon&space;\sim&space;p_T}&space;\[f(D_\phi(G_\theta\(\{z_s^i,z_a^i,T_i\}_i,\epsilon)\)\]&space;&plus;&space;E_{I&space;\sim&space;p_D}&space;\[f(-D_\phi(I))-\lambda\vert&space;\vert\bigtriangledown&space;D_\phi&space;(I)\vert\vert^2&space;\]&space;\quad&space;,where&space;\quad&space;f(t)=-log(1&plus;exp(-t)),&space;\lambda=10&space;" />
 
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
    - number of entities in the scene <img src="https://latex.codecogs.com/svg.image?N&space;\sim&space;p_N" />, latent codes <img src="https://latex.codecogs.com/svg.image?z_s^i,z_a^i&space;\sim&space;N(0,I)" />
    - camera pose <img src="https://latex.codecogs.com/svg.image?\epsilon&space;\sim&space;p_{\epsilon}" />, transformations <img src ="https://latex.codecogs.com/svg.image?T_i&space;\sim&space;p_T" /> </br>
        ⇒ In practice, <img src ="https://latex.codecogs.com/svg.image?p_{\epsilon}" /> and <img src="https://latex.codecogs.com/svg.image?p_T" /> is uniform distribution over data-dependent camera elevation angles and valid object tranformations each.
    - All object fields share their weights and are paramterized as MLPs with ReLU activations ( 8 layers with hidden dimension of 128, <img src = "https://latex.codecogs.com/svg.image?M_f=128" /> for objects & half the layers and hidden dimension for background features)
    - <img src = "https://latex.codecogs.com/svg.image?L_x=2,3,10" /> and <img src = "https://latex.codecogs.com/svg.image?L_d=2,3,4" /> for positional encoding
    - sample 64 points along each ray and render feature images at <img src = "https://latex.codecogs.com/svg.image?16^2" /> pixels 
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
        Key difference with GRAF is that GIRAFFE combines volume rendering with neural rendering. This method helps the model to be more expressive and better handle the complex real scenes. Furthermore, rendering speed is increased compared to GRAF(total rendering time is reduced from 110.1ms to 4.8ms, and from 1595.0ms to 5.9ms for <img src = "https://latex.codecogs.com/svg.image?64^2" /> and <img src = "https://latex.codecogs.com/svg.image?256^2" /> pixels, respectively.)
        
    - positional encoding
        
      <img src = "https://latex.codecogs.com/svg.image?r(t,L)&space;=&space;(sin(2^0t\pi),&space;cos(2^0t\pi),&space;...,sin(2^Lt\pi),&space;cos(2^Lt\pi))" /> <br/>
      ![Figure 8: positional encoding](/.gitbook/assets/2022spring/47/positional%20encoding.PNG)
        
- limitations
    - struggles to disentangle factors of variation if there is an inherent bias in the data. (eg. eye and hair translation)
    - disentanglement failures due to mismatches between assumed uniform distribution (camera poses and object-lebel transformations) and their real distributions
    ![Figure 9: limitation_disentangle failure](/.gitbook/assets/2022spring/47/disentanglement%20failure.png)
   
## 5. Conclusion

⇒ By representing scenes as compostional generative neural feature fields, we disentangle individual objects form the background as well as their shape and appearance without explicit supervision

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

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. [GIRAFFE] https://arxiv.org/abs/2011.12100
2. [GIRAFFE supplementary material] http://www.cvlibs.net/publications/Niemeyer2021CVPR_supplementary.pdf
3. [GIRAFFE - Github ] https://github.com/autonomousvision/giraffe
4. [INR explanation] https://www.notion.so/Implicit-Representation-Using-Neural-Network-c6aac62e0bf044ebbe70abcdb9cc3dd1
5. [NeRF] https://arxiv.org/abs/2003.08934
6. [GRAF] https://arxiv.org/abs/2007.02442

