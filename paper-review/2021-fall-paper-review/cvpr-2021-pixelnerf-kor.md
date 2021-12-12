---
description: Yu et al. / pixelNeRF - Neural Radiance Fields from One or Few Images / CVPR 2021
---

# pixelNeRF \[Kor]

[**English version**](cvpr-2021-pixelnerf-eng.md) of this article is available.

### 1. Introduction

오늘 소개할 논문은 [PixelNeRF: Neural Radiance Fields from one or few images](https://arxiv.org/abs/2012.02190)로 **view synthesis** 분야에 큰 발전을 이룬 [**NeRF**(ECCV 2020)](https://arxiv.org/abs/2003.08934)의 후속 연구입니다.

#### Problem Definition: View Synthesis

* 특정 각도에서 찍은 여러 사진들을 활용해 임의의 새로운 각도에서의 사진을 생성하는 문제입니다. 사진을 찍으면 현실 세계에서의 3d물체가 2차원의 이미지로 기록이 되는데요, 그 과정에서 물체의 깊이/받는 빛의 양 등에 대한 정보가 일부 소실되게 됩니다. 이렇게 주어진 (제한된) 정보들을 바탕으로 나머지 정보를 잘 추론하고 복원하여 현실 세계의 물체를 모델링하고, 이것을 다시 다른 각도에서의 2차원 이미지로 보여주는 것이라 이해하시면 될 것 같습니다. 이 문제는 단순히 주어진 이미지들을 interpolation한다고 하여 새로운 각도에서의 이미지를 만들 수 있는 것이 아닐 뿐더러 다양한 외부적인 요소들을 고려해야 하기때문에 굉장히 풀기 어려운 문제 중 하나입니다.
* 현재까지 view synthesis분야의 SOTA 알고리즘으로는 2020년 ECCV에 발표된 NeRF로, 기존보다 훨씬 훌륭한 성능을 내 많은 관심을 받았습니다.

> NeRF에 대한 설명은 아래 related work 파트를 확인해주세요 :)

### 2 Motivation

그럼 pixelNeRF에 대해 알아보기 전에 베이스 논문이라 할 수 있는 NeRF와 그 외 관련 연구들에 대해 알아보고, pixelNeRF가 어떤 점을 발전시키려 했는지 알아봅시다.

#### 2.1 Related Work

#### NeRF

NeRF는 view synthesis, 즉 카메라를 이용해서 찍은 n개의 2D 이미지에서 빛과 원근감을 복원하여 새로운 각도에서 물체를 찍은 2D 이미지를 생성하는 task를 위한 모델입니다. 이때, 새로운 각도에서의 2D 이미지를 생성하겠다는 것은 결국 3D object를 통째로 모델링 하겠다는 의미로 볼 수 있습니다. 이 모델링에 있어서 NeRF는 neural radiance field($$\approx$$ neural implicit representation)라 불리는 "각 pixel 좌표 값을 input으로 주면 해당 위치의 RGB 값을 연산하는 함수"를 사용합니다. 여기서의 함수는 deep neural network로 정의되며, 아래의 수식처럼 표현할 수 있습니다.

_(3D object는 2D와 달리 굉장히 sparse하므로 RGB값을 discrete한 행렬로 연산하는 것 보다 이와 같은 방법이 효율적이라고 합니다. 뿐만 아니라 이미지의 super-resolution등 다양한 CV분야에서 사용되고 있습니다.)_

$$
F_\Theta: (X,d) \rightarrow (c,\sigma)
$$

* Input: pixel의 위치 $$X \in \mathbb{R}^3$$ 와 보는 방향을 나타내는 unit vector $$d \in \mathbb{R}^2$$
* Output: color 값과 density $$\sigma$$

그렇다면, 함수 $$F_\Theta$$를 통해 구한 color와 density값으로 어떻게 새로운 이미지를 랜더링할까요? (이 이미지 랜더링 과정을 논문에선 volume rendering이라 칭합니다.)

함수로 연산한 color값은 3차원 좌표에서의 RGB값을 말합니다. 이때 다른 각도에서 바라본 2D 이미지를 생성하려면, (그 방향에서 바라보았을 때) 앞에 위치한 부분에 가려지거나, 뒤에 위치한 것이 비치는 경우 등을 고려해야 합니다. 바로 이게 우리가 함수를 통해 구한 density가 필요한 이유이지요.

이러한 것들을 다 고려해 3차원에서의 RGB값들을 2D 이미지에서의 RGB값으로 변환하는 수식이 아래와 같습니다.

$$
\hat{C}_r=\int_{t_n}^{t_f} T(t)\sigma(t)c(t)dt\
$$

**Notation**

* camera ray $$r(t)=o+td$$
  * $$t$$: 실제 물체(원점)에서부터 구하고자 하는 사이의 거리
  * $$d$$: 바라보는 방향에 대한 unit vector
  * $$o$$: 원점
*   $$T(t)=exp(-\int_{t_n}^t\sigma(s)ds)$$

    : t점을 가로막고 있는 점들의 density의 합 ($$\approx$$ 광선이 다른 입자에 부딪히지 않고 $$t_n$$에서 $$t$$로 이동할 확률)
* $$\sigma(t)$$ : t 지점에서의 density값
* $$c(t)$$: t점에서의 RGB값

이렇게 구한 추정된 RGB값 $$\hat{C}_r$$과 실제 RGB값 $$C(r)$$ 의 차이로 loss를 계산하여 학습을 진행합니다.

$$
\mathcal{L}=\Sigma_r ||\hat{C}_r -C(r)||^2_2
$$

이 과정들은 모두 미분이 가능하기에 gradient descent로 최적화 가능합니다.

![](/.gitbook/assets/19/figure2.png)

그림을 통해 한번 더 정리하자면, 우선 (a) 2D이미지에서 3차원 좌표 (x,y,z) 및 direction d를 추출합니다. (\_추출 과정은 본 논문 저자의 이전 연구인 \_[_LLFF_](https://arxiv.org/pdf/1905.00889.pdf)_를 따릅니다.)_ (b) 그 후 neural radiance field를 이용해 각 좌표에서의 color와 density값을 구합니다. (c) 위에서 설명한 식을 통해 3차원의 volume을 2차원의 이미지로 랜더링 합니다. (d) 이렇게 구한 각 2D 좌표에서의 RGB값을 ground truth와 비교하며 함수를 최적화합니다.

_이 기본 구조 외에도 논문에선 positional encoding , hierarchical volume sampling등 성능 향상을 위한 다양한 기법들을 사용해 모델의 성능을 높이지만, 본 paper review의 주제를 벗어나므로 그 부분은 생략하도록 하겠습니다._

> 여기까지가 본 논문에 대한 이해를 위해 필요한 기본적인 NeRF에 대한 설명입니다. 혹시나 이 설명이 부족하다 생각하신 분은 포스팅 아래 참고자료의 [링크](https://www.youtube.com/watch?v=zkeh7Tt9tYQ)를 참고해주세요 :)

####

#### View synthesis by learning shared priors

PixelNeRF 이전에도 few-shot or single-shot view synthesis를 위해 학습된 prior를 사용하는 다양한 연구가 존재하였습니다.

![](/.gitbook/assets/19/figure3.png)

그러나, 대부분이 3차원이 아닌 2.5차원의 데이터를 사용하거나, interpolation을 활용해 depth 추정하는 고전적인 방법을 사용하였습니다. 3D 객체를 모델링함에 있어서도 (2D 이미지가 아닌) 3D object 전체에 대한 정보를 필요로 하거나 이미지의 global한 feature만 고려하는 등 여러 한계가 존재하였습니다. 또한, 대부분의 3D learning 방법들은 일정한 방향으로만 정렬되는 예측 공간 (object-centered coordinate system)을 사용했는데, 이런 경우 다양한 예측이 어렵다는 단점이 있습니다. pixelNeRF는 이러한 기존 방법론의 단점들을 보완하여 모델의 성능을 향상시켰습니다.

#### 2.2 Idea

높은 성능으로 큰 파장을 불러일으킨 NeRF이지만, 한계점도 존재합니다. 고품질 이미지를 합성하기 위해 하나의 객체에 대한 **여러 각도** 의 이미지가 필요하고, 모델을 **최적화**하는데 긴 시간이 소요된다는 것인데요. 오늘 소개할 pixel NeRF는 이런 NeRF의 한계점을 보완하면서 훨씬 더 짧은 시간에 안에 적은 수의 이미지 만으로 새로운 시점에서의 이미지를 생성하는 방법을 제안합니다.

적은 수의 이미지만으로도 그럴듯한 이미지를 생성할 수 있으려면, 각 scene들의 공간적인 관계를 모델이 학습할 수 있어야 합니다. 이를 위해 pixelNeRF는 이미지의 **spatial features**를 추출해 input으로 사용합니다. 즉, spatial features들이 view synthesis를 하기 위한 scene prior로서의 역할을 하는 것입니다. (이때 feature는 fully convolutional image feature를 사용합니다.)

아래 그림과 같이 pixelNeRF는 NeRF보다 더 적은 입력 이미지에 대해서도 훌륭한 결과를 생성한다는 것을 알 수 있습니다.

![](/.gitbook/assets/19/figure1.png)

### 3. Methodology

그럼 이제 PixelNeRF모델의 작동 메커니즘에 대해 알아봅시다. 모델의 구조는 크게 두 파트로 나눌 수 있습니다.

* fully-convolutional image encoder $$E$$ : input image를 pixel-aligned feature로 인코딩 하는 부분
* NeRF network $$f$$ : 객체의 색과 밀도를 연산하는 부분

인코더 $$E$$ 의 output값이 nerf network의 input으로 들어가게 되는 것이지요. 이제 이 과정에 대해 자세히 설명해보도록 하겠습니다.

#### 3.1 Single-Image pixelNeRF

이 논문은 pixelnerf를 single-shot과 multi-shot으로 나누어 학습방법을 소개합니다. 우선 Single-image pixelNeRF부터 살펴보도록 합시다.

**Notation**

* $$I$$: input image
* $$W$$: extracted spatial feature $$=E(I)$$
* $$x$$: camera ray
* $$\pi(x)$$: image coordinates
* $$\gamma(\cdot)$$ : positional encoding on $$x$$
* $$d$$: unit vector about viewing direction

![](/.gitbook/assets/19/figure4.png)

1. 우선 input image $$I$$ 를 encoder에 넣어 spatial feature vector W를 추출합니다.
2. 그 후 camera ray $$x$$ 위의 점들에 대해, 각각에 대응되는 image feature를 구합니다.
   * camera ray $$x$$ 를 이미지 평면에 projection시키고 이에 해당하는 좌표 $$\pi(x)$$ 구합니다.
   * 이 좌표에 해당하는 spatial feature $$W(\pi(x))$$를 bilinear interpolation을 사용해 구합니다.
3. 이렇게 구한 $$W(\pi(x))$$ 와 $$\gamma(x), d$$ 를 NeRF network에 넣고 color와 density값을 구합니다. (이 단계가 implicit network를 사용하는 단계입니다.) 

$$
f(\gamma(x),d;W(\pi(x)))=(\sigma,c)\
$$

4\. NeRF에서와 동일한 방법으로 volume rendering을 진행합니다.

> 즉, nerf와 달리 input에 대한 pre-processing을 통해 input image의 spatial feature를 추출(1,2)하고 이것을 nerf network에 추가한다는 점이 기존 nerf와 차별화된 점이라 할 수 있습니다. 이렇게 feature에 대한 정보를 같이 넣어주면 네트워크가 pixel단위의 개별적인 정보간 유기적인 관계를 implicit하게 학습할 수 있고, 이것은 보다 적은 데이터로도 안정적이고 정확한 추론을 할 수 있도록 만든다. 

#### 3.2 Multi-view pixelNeRF

Few-shot view synthesis의 경우, 여러 사진이 들어오기 때문에 query view direction을 통해 (target direction에 대한) 특정 image feature의 중요도를 볼 수 있습니다. 만약 input view와 target direction이 비슷하다면, 모델은 input으로 학습된 데이터를 바탕으로 추론하면 될 것이고, 그렇지 않다면 기존 학습된 prior를 활용해야 할 것입니다.

multi-view 모델 구조의 기본적인 틀은 single-shot pixelNeRF와 거의 유사하지만, 여러 이미지를 모두 고려하기 위해 달라지는 부분들이 있습니다.

1. 우선 multi-view task를 풀기 위해 저자는 각 이미지들의 상대적인 카메라 위치를 알 수 있다고 가정합니다.
2.  각각의 이미지 $$I^{(i)}$$ 속에서 원점에 위치한 객체들을 우리가 보고자하는 target 각도에서의 좌표에 맞게 변환합니다.

    $$P^{(i)} = [R^{(i)} \; t^{(i)}], \ x^{(i)}= P^{(i)}x$$, $$d^{(i)}= R^{(i)}d$$
3. encoder를 통해 feature를 뽑을 땐 각각의 view frame마다 독립적으로 뽑아 NeRF network에 넣고 NeRF network의 final layer에서 합칩니다. 이는 다양한 각도에서의 이미지에서 최대한 많은 spatial feature을 뽑아내기 위한 것입니다.
   *   논문에선 이를 수식으로 나타내기 위해 NeRF network의 initial layer를 $$f_1$$, intermediate layer를 $$V^{(i)}$$, final layer를 $$f_2$$ 라 표기합니다.

       $$
       V^{(i)}=f_1(\gamma(x^{(i)}),d^{(i)}; W^{(i)}(\pi(x^{(i)}))) \\\ (\sigma,c)= f_2 (\psi(V^{(i)},...,V^{(n)}))\
       $$

       * $$\psi$$: average pooling operator

multi-view pixelNeRF의 단순화 버전이 single-view pixelNeRF인 셈입니다.

### 4. Experiments

**Baselines & Dataset**

* 기존 few-shot / single-shot view synthesis의 SOTA 모델이었던 SRN과 DVR, 그리고 비슷한 구조(neural radiance field)의 네트워크를 사용한 NeRF와 비교합니다.
* 3D 물체에 대한 벤치마크 데이터셋인 ShapeNet, 그리고 보다 실제 사진과 흡사한 DTU 데이터셋에 대해 모두 실험을 진행하며 pixelNeRF의 성능을 보여줍니다.

**Metrics**

이때 성능 지표로는 많이 사용하는 image qualifying metric들을 사용하였습니다.

* PSNR: $$10 log_{10}(\frac{R^2}{MSE})$$
  - 신호가 가질 수 있는 최대 신호에 대한 잡음의 비로 화질에 대한 손상 정보를 평가하기 위한 목적으로 사용된다. 
  - $$R$$: 해당 이미지에서의 최대값
* SSIM: $$\frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+ \mu_y^2+ C_1)(\sigma_x^2+\sigma_y^2+C_2)}$$
  - 이미지 구조 정보의 왜곡 정도가 이미지 품질에 큰 영향을 미친다는 가정을 바탕으로 수치적인 에러가 아닌 인간의 시각적 화질 차이 및 유사도 평가를 위해 고안된 방법이다. 
  - 직관적으로 설명하자면, 두이미지의 휘도x대비x상관계수로 계산한다. 

**Training setup**

본 논문의 실험에선 imagenet에 pretrained된 resnet34 모델을 backbone network로 사용합니다. 4번째 pooling layer까지 feature를 추출하고, 그 이후 layer에선 (위 3에서 설명했듯이) 대응되는 좌표에 맞는 feature를 찾는 과정을 거칩니다. 이때, local한 feature와 global한 feature를 모두 사용하기위해, feature pyramid형태로 추출합니다. 여기서 feature pyramid란 서로 다른 해상도의 feature map을 쌓아올린 형태를 말합니다.

또한, NeRF network $$f$$에서도 ResNet구조를 차용하여 좌표 및 viewing direction $$\gamma(x), d$$를 먼저 입력하고 feature vector $$W(\phi(x))$$를 residual로써 각 ResNet block 앞부분에 더합니다.

***

크게 세가지의 실험을 통해 pixelNeRF의 성능을 잘 보여주었습니다.

1.  ShapeNet 벤치마크 데이터셋에서 category-specific한 경우와 category-agnostic한 경우 모두에서의 view synthesis를 시행하였습니다.

    ![](/.gitbook/assets/19/figure5.png)  
    ![](/.gitbook/assets/19/figure6.png)

    하나의 pixelNeRF모델을 shapenet 내 가장 많은 13개의 카테고리에 대해 학습한 실험입니다. 위 결과를 보면 알 수 있듯이 pixelNeRF는 view synthesis의 측면에서 SOTA 결과를 보이고 있습니다. category-specific / category-agnostic한 경우 모두에서 가장 정교하고 그럴듯한 이미지를 생성하며, 이미지 성능 측도인 PSNR, SSIM 또한 가장 높은 수치를 보입니다.

2\. 학습된 prior를 통해 ShapeNet 데이터 내 unseen category혹은 multi-object data에 대해서도 view synthesis를 적용 가능함을 보였습니다.

![](/.gitbook/assets/19/figure7.png)

모델을 자동차와 비행기 그리고 의자에 대해서만 학습을 시킨 후, 다른 카테고리에 대해 view synthesis를 진행한 결과입니다. 여기서도 pixelNeRF의 성능이 잘 나타남을 알 수 있습니다. 논문에선 이렇게 일반화 가능한 이유가 바로 canonical space가 아닌 카메라의 상대적인 위치(view space)를 사용했기 때문이라고 설명합니다.

3\. DTU MVS dataset과 같은 실제 장면에 대해서도 view synthesis를 시행하였습니다.

위 shapenet처럼 특정 물체에 대해 제한적으로 찍은 이미지가 이닌, 실제 이미지 데이터에 대해서도 scene 전체의 관점을 이동시키는 task도 비교적 잘 해냅니다. 88개의 학습 이미지 씬을 바탕으로 실험을 진행하여도 위와 같이 다양한 각도에서의 이미지를 만들어 냅니다. NeRF와 비교하면 적은 데이터로 전체 이미지 씬에 대한 모델링을 훨씬 잘 하고 있다는 것을 볼 수 있습니다.

![](/.gitbook/assets/19/figure8.png)

위 실험들을 통해 pixelNeRF가 ShapeNet과 같은 정형화된 3D dataset 뿐만 아니라, multi-object image, unseen image, real scene image등 다양한 환경에 적용할 수 있음이 증명되었습니다. 또한, 이 모든 과정이 기존 NeRF보다 훨씬 적은 이미지만으로도 가능함도 보였습니다.

### 5. Conclusion

본 논문에서 제안한 pixelNeRF는 적은 수의 이미지 만으로도 view synthesis task를 잘 해결하기 위해 기존 NeRF에 spatial feature vectors를 학습하는 과정을 추가하여 NeRF를 비롯한 기존 view synthesis 모델들의 한계점을 보완하였습니다. 또한, 다양한 실험을 통해 pixelNeRF가 일반화된 다양한 환경(multi-objects, unseen category, real dataset etc.)에서 잘 작동함을 보였습니다.

그러나 아직도 몇가지 한계점들은 존재하는데요. NeRF와 마찬가지로 rendering시간이 굉장히 오래걸리며, ray sampling bounds/positional encoding에 사용되는 parameter등을 수동으로 조정해야하기때문에 scale-variant합니다. 또한, DTU에 대한 실험으로 real image에 대한 적용 가능성을 보였지만 이 데이터셋도 어느정도 제약된 상황에서 만들어졌기 때문에 굉장히 raw한 real dataset에대해서도 비슷한 성능을 낼 수 있을지는 아직 보장되지 않았습니다.

그럼에도 현재 많은 관심을 받고 있는 NeRF의 성능을 높이고 보다 일반화된 task로 확장시켰다는 점에서 충분히 의미있는 연구라 생각이 됩니다. 논문에 대한 설명을 읽고 궁금한 사항이 생기시면 언제든 아래 주소로 연락주시면 답변해드리겠습니다 :)

#### Take home message

* 최근 2D 이미지 만으로 실제 물체를 모델링해 여러 각도에서 보여주는 연구들이 활발히 진행되며 특히 neural implicit representation을 활용한 연구가 많은 관심을 받고 있다.
* 이때 주어진 이미지 내 pixel값 뿐만 아니라 이미지가 내포하고 있는 feature들을 추출해 사용하면 (복원력/효율성 측면에서 모두) 훨씬 더 좋은 성능을 낼 수 있다.

### 6. Author

**권다희 (Dahee Kwon)**

* KAIST AI
* Contact
  * email: [daheekwon@kaist.ac.kr](mailto:daheekwon@kaist.ac.kr)
  * github: [https://github.com/daheekwon](https://github.com/daheekwon)

### 7. Reference & Additional materials

* [NeRF paper](https://arxiv.org/abs/2003.08934)
* [NeRF 설명영상](https://www.youtube.com/watch?v=zkeh7Tt9tYQ)
* [pixelNeRF official site](https://alexyu.net/pixelnerf/)
* [pixelNeRF code](https://github.com/sxyu/pixel-nerf)
