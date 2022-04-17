---
description: Niemeyer et al. / GIRAFFE] Representing Scenes as Compositional Generative Neural Feature Fields / CVPR 2021 (oral, best paper award)
---

# GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields \[Kor\]

**[English version](cvpr-2021-GIRAFFE-eng.md)** of this article is available.

##  1. Problem definition

GAN(Generative Adversarial Network) 을 통해 우리는 사실적인 이미지를 무작위로 생성할 수 있게 되었고, 더 나아가 각각의 표현(머리색, 이목구비 등)을 독립적으로 조절할 수 있는 경지에 이르렀다. 하지만 3차원 세계를 2D 로 나타냄으로써 한계에 부딪히게 되었고, 최근 연구들은 3D representation 을 효과적으로 나태내는 것에 주력하고 있다. 가장 대표적인 방법은 2장에서 소개될 implicit neural representation 인데, 기존 연구들은 물체가 하나이거나 복잡하지 않은 이미지에 대해서만 좋은 성능을 보였다. 본 논문은 각각의 물체를 3D representation 의 개별적인 구성 요소로 대하는 생성 모델을 구성하여 여러 물체가 있는 복잡한 이미지에서도 좋은 성능을 보인다. 

## 2. Motivation

### Related work

**Implicit Neural Representation (INR)**

기존 인공신경망(neural network) 은 추정(ex. image classification) 과 생성(ex. generative models) 의 역할을 수행하였다. 이에 반해 Implicit representation 은 표현의 기능을 수행하여, network paramter 가 이미지 정보 자체를 의미하게 된다. 그래서 네트워크의 크기는 정보 자체의 복잡도에 비례하게 된다 (단순한 원보다 벌의 사진을 나타내는 모델이 더 복잡하다). 더 나아가 NeRF 에서 처럼 좌표가 입력값으로 들어왔을 때 RGB 값을 산출하는 연속적인 함수를 학습함으로써 연속적인 표현도 가능해지게 되었다.

- **NeRF : Neural Radiance Field** <img src = "https://latex.codecogs.com/svg.image?f_\theta:R^{L_x}&space;\times&space;R^{L_d}&space;&space;\to&space;R^&plus;&space;\times&space;R^3" />

    하나의 장면은 5D 좌표 (3d 위치와 방향) 에 대한 RGB 값과 부피 intensity 을 산출하는 fully connected layer 로 표현된다. 이때 더 높은 차원의 정보를 얻기 위해 5D 입력값은 positional encoding 을 거치게 된다. <br />
    특정 방향에서 빛을 쏘았을 때 생기는 camera ray 내의 점을 n 개 샘플링하여, 각각의 color 와 density 값을 volume rendering technique (3. Methods 에 설명) 을 통해 합침으로써 이미지 pixel 의 값을 예측한다. 학습은 GT(ground truth) posed 이미지와 예측된 volume rendered 이미지 간의 차이를 줄이는 방향으로 이루어진다.
    
    <p align="center">
      <img src="https://github.com/nooppi18/awesome-reviews-kaist/blob/master/.gitbook/assets/2022spring/47/NeRF.PNG">
    </p>
- **GRAF : Generative Radiance Field** <img srf = "https://latex.codecogs.com/svg.image?f_\theta:R^{L_x}&space;\times&space;R^{L_d}&space;\times&space;R^{M_s}&space;\times&space;R^{M_a}&space;\to&space;R^&plus;&space;\times&space;R^3" />
    
    본 논문은 NeRF 와 달리 unposed image 를 활용하여 3D representation 을 학습한다. Input 으로는 sampling 된 camera pose &epsilon;, (위쪽 반구에서 중심을 바라보는 방향 중에서 uniform 하게 sample) 과 sampling 된 K x K patch (unposed image 에서 중심이 (u,v) 이고 scale 이 s 인 K x K 이미지) 를 가진다. 추가로, shape 와 appearance 코드를 condition 으로 넣어주어, patch 의 pixel 값을 예측하고, discriminator 에서 predicted patch 는 fake, 이미지 분포에서 sampling 된 image 의 실제 K x K patch 는 real 로 분류하는 학습을 진행한다. 
    <p align="center">
      <img src="https://github.com/nooppi18/awesome-reviews-kaist/blob/master/.gitbook/assets/2022spring/47/GRAF.PNG">
    </p>
### Idea

GRAF 가 제어가능한 고해상도의 image synthesis 를 해내지만, 단일 물체만 있는 비교적 간단한 imagary 에서만 좋은 성능을 보이는 한계점을 가진다. 이를 해결하기 위해서, GIRAFFE 에서는 개별 object 를 구분하여 변형하고 회전시킬 수 있는 neural representation 을 제안한다. 

## 3. Method
<p align="center">
   <img src="https://github.com/nooppi18/awesome-reviews-kaist/blob/master/.gitbook/assets/2022spring/47/GIRAFFE.PNG">
</p>

- **Neural Feature Field** : GRAF formulation 과 유사하지만, 3D color 를 output 하는 것이 아니라 <img src ="https://latex.codecogs.com/svg.image?M_f" />-dimensional feature 를 output 한다.

    <img src="https://latex.codecogs.com/svg.image?h_\theta:R^{L_x}&space;\times&space;R^{L_d}&space;\times&space;R^{M_s}&space;\times&space;R^{M_a}&space;\to&space;R^&plus;&space;\times&space;R^{M_f}" />
    
    
    **Object Representation** NeRF 와 GRAF 에서는 전체 scene 이 하나의 model 로 표현 되었는데, 각 물체를 독립적으로 다루기 위해서 개별적인 feature field 로 나타낼 것을 본 논문에서는 제안한다. 이때 affine transformation도 활용함으로써 pose, shape, appearance 를 모두 제어할 수 있게 된다.  <br /> <br />
        <img src = "https://latex.codecogs.com/svg.image?T=\left\{s,t,R\right\}&space;" />     
        (<img src = "https://latex.codecogs.com/svg.image?s" />:scale, <img src= "https://latex.codecogs.com/svg.image?t" />: translation, <img src= "https://latex.codecogs.com/svg.image?R" />: rotation) sampled from dataset-dependent distribution   
        <img src = "https://latex.codecogs.com/svg.image?k(x)&space;=&space;R&space;\cdot&space;\begin{bmatrix}&space;s_1&space;&&space;&&space;\\&space;&&space;s_2&space;&&space;\\&space;&&space;&&space;s_3&space;\end{bmatrix}&space;\cdot&space;x&space;&plus;&space;t" />  <br />      
        <img src = "https://latex.codecogs.com/svg.image?(\sigma,f)&space;=&space;h_{\theta}(\gamma(k^{-1}(x)),&space;\gamma(k^{-1}(d)),&space;z_s,&space;z_a)" />  <br />  
    **Composition Operator** 각 scene 은 N 가지의 entitiy 로 정의된다(N-1 objects, 1 background). 각 entity 의 density 와 feature 를 합치기 위해 density-weighted mean 을 사용한다. <br />
    
    <img src = "https://latex.codecogs.com/svg.image?C(x,d)=(\sigma,&space;{1&space;\over&space;\sigma}\sum_{i=1}^{N}\sigma_if_i),&space;\,&space;where&space;\;&space;\sigma&space;=&space;\sum_{i=1}^N\sigma_i" />
        
    **3D volume rendering** NeRF 와 동일하게 numerical integration 을 해준다.  <br />  <br /> 
        <img src = "https://latex.codecogs.com/svg.image?\pi_{vol}&space;:&space;(R^&plus;&space;\times&space;R^{M_f})^{N_s}&space;\to&space;R^{M_f}" />  <br />  <br />
        <img src = "https://latex.codecogs.com/svg.image?f=\sum_{j=1}^{N_s}\tau_i\alpha_if_i&space;\quad&space;\tau_j=\prod_{k=1}^{j-1}(1-\alpha_k)&space;\quad&space;\alpha_j=1-e^{-\sigma_i\delta_j}" />
        
- **2D neural rendering**
    <img src="https://latex.codecogs.com/svg.image?\pi_\theta^{neural}&space;:&space;R^{H_v&space;\times&space;W_v&space;\times&space;M_f}&space;\to&space;R^{H&space;\times&space;W&space;\times&space;3}" />
    
    <p align="center">
      <img src = "https://github.com/nooppi18/awesome-reviews-kaist/blob/master/.gitbook/assets/2022spring/47/2d%20neural%20rendering.PNG" >
    </p>
    
- **Training**  <br /> 
    - Generator   <br /> 
    <img src = "https://latex.codecogs.com/svg.image?G_\theta(\left\{z_s^i,z_a^i,T_i\right\}_{i=1}^N,&space;\epsilon)=\pi_\theta^{neural}(I_v),\quad&space;where&space;\quad&space;I_v=\left\{\pi_{vol}(\left\{C(x_{jk},d_k)\right\}_{j=1}^{N_s})\right\}_{k=1}^{H_v&space;\times&space;W_v}" />
    
    - Discriminator : CNN with leaky ReLU <br /> <br />
    - Loss Funcion = non-saturating GAN loss + R1-regularization <br />
    <img src = "https://latex.codecogs.com/svg.image?V(\theta,&space;\phi)=E_{z_s^i,&space;z_a^i&space;\sim&space;N,&space;\epsilon&space;\sim&space;p_T}&space;\[f(D_\phi(G_\theta\(\{z_s^i,z_a^i,T_i\}_i,\epsilon)\)\]&space;&plus;&space;E_{I&space;\sim&space;p_D}&space;\[f(-D_\phi(I))-\lambda\vert&space;\vert\bigtriangledown&space;D_\phi&space;(I)\vert\vert^2&space;\]&space;\quad&space;,where&space;\quad&space;f(t)=-log(1&plus;exp(-t)),&space;\lambda=10&space;" />
 
## 4. Experiment & Result


### Experimental setup

- DataSet
    - single object dataset로 자주 사용되는 Chairs, Cats, CelebA, CelebA-HQ 
    - single-object dataset 중 까다롭다고 알려진 CompCars, LSUN Churches, FFHQ
    - multi-object scenes으로는 Clevr-N, Clevr-2345
- Baseline
    - voxel-based PlatonicGAN, BlockGAN, HoloGAN
    - radiance field-based GRAF
- Training setup
    - 한 장면 내의 entity 수 <img src="https://latex.codecogs.com/svg.image?N&space;\sim&space;p_N" />, latent codes <img src="https://latex.codecogs.com/svg.image?z_s^i,z_a^i&space;\sim&space;N(0,I)" />
    - camera pose <img src="https://latex.codecogs.com/svg.image?\epsilon&space;\sim&space;p_{\epsilon}" />, transformations <img src ="https://latex.codecogs.com/svg.image?T_i&space;\sim&space;p_T" /> </br>
        ⇒ <img src ="https://latex.codecogs.com/svg.image?p_{\epsilon}" /> 과 <img src="https://latex.codecogs.com/svg.image?p_T" /> 는 uniform distribution 를 따른다고 가정하고 실험을 진행한다 (데이터 종속적인 camera elavation 과 object transformation)
    - 개별 object field 는 모두 MLP weight 를 공유하며 ReLU activation 을 사용한다.(object 들은 8 layers MLP(hidden dimension of 128), <img src = "https://latex.codecogs.com/svg.image?M_f=128" /> 를 사용하고, background 는 이의 절반을 사용한다.)
    - <img src = "https://latex.codecogs.com/svg.image?L_x=2,3,10" /> 과 <img src = "https://latex.codecogs.com/svg.image?L_d=2,3,4" /> 를 positional encoding parameter 
    - 각 ray 따라 64 points를 sample 하고 image 별로 <img src = "https://latex.codecogs.com/svg.image?16^2" /> pixels 를 사용하여 계산 효율성을 얻는다.
- Evaluation Metric
    - 20,000 real & fake samples 로 Frechet Inception Distance (FID) score 계산

### Result

- disentangled scene generation   
    <p align="center">
      <img src = "https://github.com/nooppi18/awesome-reviews-kaist/blob/master/.gitbook/assets/2022spring/47/controllable.PNG" >
    </p>
    
- comparison to baseline methods   
    <p align="center">
      <img src = "https://github.com/nooppi18/awesome-reviews-kaist/blob/master/.gitbook/assets/2022spring/47/qualitative%20comparison.PNG" >
    </p>
- ablation studies
    - importance of 2D neural rendering and its individual components
      <p align="center">
        <img src = "https://github.com/nooppi18/awesome-reviews-kaist/blob/master/.gitbook/assets/2022spring/47/ablation.PNG" >
      </p>
      GRAF 와의 가장 큰 차이는 neural rendering 을 volumne rendering 과 함께 사용했다는 점이다. 이 방법은 모델의 표현력을 향상시키고 더 복잡한 real scene 도 다룰 수 있게 한다.
        
    - positional encoding
        
      <img src = "https://latex.codecogs.com/svg.image?r(t,L)&space;=&space;(sin(2^0t\pi),&space;cos(2^0t\pi),&space;...,sin(2^Lt\pi),&space;cos(2^Lt\pi))" />
      <p align="center">
        <img src = "https://github.com/nooppi18/awesome-reviews-kaist/blob/master/.gitbook/assets/2022spring/47/positional%20encoding.PNG" >
      </p>
        
- limitations
    - 데이터 내에 inherent bias 가 있으면 같이 변화해야하는 factor 들이 고정되는 문제가 발생한다. (ex. 눈과 헤어 rotation)
    - camera pose 와 obejct 단위의 transformation 이 uniform distribution 을 따른다고 가정하는데, 실제로는 그렇지 않을 것이기에 아래와 같은 disentanglement failure 가 발생한다.
    <p align = "center">
        <img src = "https://github.com/nooppi18/awesome-reviews-kaist/blob/master/.gitbook/assets/2022spring/47/disentanglement%20failure.png">
    </p>
    
## 5. Conclusion

⇒ 한 장면을 compositional generative neural feature field 로 나타냄으로써, 개별 object 를 background 뿐만 아니라 shape 과 appenarance 로부터 disentangle하여 별다른 supervision 없이도 독립적으로 control 할 수 있다.

⇒ Future work
- 개별 object 의 tranformation 과 camera pose 의 distribution 을 데이터로부터 학습할 수는 없을까?
- object mask 와 같이 얻기 쉬운 supervision 을 활용하면 더 복잡한 multi-object scene 을 더 나타낼 수 있을 것으로 보인다.

### Take home message \(오늘의 교훈\)

- Implicit Neural Representation 을 활용한 3D scene representation 은 최근에 각광 받고 잇는 방식이다.
- 각각의 entity 를 개별 feature field 로 나타내는 것은 그들의 movement 를 disentangle 하는데 도움이 된다.
- 각 feature 를 원래 dimension 그대로로 사용하기 보다는 positional encoding 이나 neural rendering 을 통해 더 high dimensional space 로 embedding 하여 활용하면 더 풍부한 정보를 활용할 수 있게 된다.
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

