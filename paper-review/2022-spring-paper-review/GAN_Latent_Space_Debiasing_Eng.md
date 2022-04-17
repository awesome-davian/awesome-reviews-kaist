---
description: Vikram V. Ramaswamy / Fair Attribute Classification through Latent Space De-biasing / CVPR 2021 Oral
---

# GAN Latent Space De-biasing \[Eng\]

한국어로 쓰인 리뷰를 읽으려면 [여기](./GAN_Latent_Space_Debiasing_Kor.md)를 누르세요.

##  1. Problem definition

지금까지 수많은 딥러닝 모델이 개발되면서 인공지능의 성능은 크게 향상되었다. 그러나 모델들 대부분은 데이터셋의 전반적인 예측 정확도에 초점을 두고 개발되었기 때문에, 모델이 데이터셋 내의 특정 집단에 대해 불리한 판단을 내릴 여지가 존재한다. 예를 들어, 서구권 국가에서 개발된 얼굴 인식 AI의 경우 아시아인의 얼굴을 백인의 얼굴보다 더 부정확하게 판별할 가능성이 높다. 우리는 이와 같은 현상을 가리켜 '인공지능의 공정성 문제'라 부른다. 아무리 인공지능의 성능이 좋아진다고 해도, 인공지능의 공정성 문제가 해결되지 않는다면 인공지능 모델은 장애인이나 노인과 같이 사회적으로 소외받는 집단에 대해 잘못된 판단을 쉽게 내릴 수 있을 것이고, 이는 심각한 사회 문제를 초래할 것이다. 그러므로 인공지능을 더욱 공정하게 만드는 것은 매우 중요한 일인데, 최근 인공지능 학계에서는 인공지능의 성능을 크게 희생하지 않으면서도 공정성을 향상시킬 수 있는 방법에 대해 활발하게 연구가 이루어지고 있다.

딥러닝 모델의 공정성을 향상시키는 방법은 다양한데, 논문의 저자는 적대적 생성 신경망(GAN)을 통한 데이터 증강(Data Augmentation)을 시도한다. 즉 GAN을 이용해 그럴듯한 이미지들을 생성한 뒤 이들의 잠재 공간(latent space)을 수정함으로써, 특정 집단에 대한 편향성이 제거되도록 훈련 데이터셋을 늘리는 방식이다. 지금까지 이와 비슷한 연구는 이전에도 있었으나, 알고리즘이 더욱 복잡해지고 연산량이 늘어난다는 단점이 있었다. 반면에 논문 저자는 단 하나의 GAN을 사용하는, 간단하고 효과적인 데이터 증강 방법을 제시한다.

## 2. Motivation

### Related work

(1) De-biasing methods

많은 경우에 딥러닝 모델의 불공정성은 훈련 데이터에 내재된 편향성에 의해 생겨난다. 이를 해결하기 위해 훈련데이터의 편향성을 줄이는 방법을 쓰기도 하고, 모델의 학습 과정을 보완하는 방법을 쓰기도 한다. 훈련 데이터의 편향성을 줄이는 방법으로는 취약 집단을 대상으로 오버샘플링을 적용하는 방법, 적대적 학습을 이용하는 방법 등이 있다. 모델의 학습 과정을 보완하는 방법으로는 모델의 손실함수(loss function)에 공정성과 관련된 규제(regularization) 항을 추가하는 방법 등이 있다. 이 논문에서는 공정성 향상을 위해 훈련데이터의 편향성을 줄이는 방법을 이용한다.

(2) Generative Adversarial Network (GAN)

적대적 생성 신경망(GAN)은 생성자와 판별자로 이루어진 신경망인데, 여기서 생성자의 학습 방식과 판별자의 학습 방식은 적대적인 관계에 있다. 즉 생성자는 자기가 거짓으로 만들어 낸 데이터를 판별자가 가짜로 인식하지 못하도록 학습하고, 판별자는 생성자가 자기를 속이지 못하도록 학습한다. 이와 같이 적대적인 학습을 시킴으로써 진짜처럼 보이는 가짜 데이터를 만들어 내는 신경망이 바로 적대적 생성 신경망이다. 그동안 적대적 생성 신경망은 많은 개선을 거쳤고, 이제는 현실과 구분하기 매우 어려운 이미지를 생성할 수 있을 정도가 되었다.  

(3) Data augmentation through latent-space manipulation

생성된 이미지를 변형시키기 위해 GAN의 잠재 공간을 조작해 볼 수 있다. 여기서 잠재 공간이란 생성자가 랜덤하게 이미지를 생성하는 데 이용하는 특성들의 공간으로, 잠재 공간에는 이미지의 다양한 속성이 압축되어 있다. 잠재 공간을 잘 조작한다면 이미지에 특정 속성(머리 색, 안경 착용 여부 등)을 부여하거나 이를 조절하는 것이 가능하다. 또한 특정 속성에 대해서만 각기 다른 값을 가진 이미지들을 생성함으로써 딥러닝 모델이 해당 속성에 대해 얼마나 불공정성을 지니고 있는 지 측정해 볼 수 있으며, 딥러닝 모델의 불공정성과 가장 크게 연관되어 있는 속성을 찾아낼 수도 있다. 이와 같이 GAN의 잠재 공간을 적잘히 이용한다면, 속성 편향성이 해소되는 방향으로 훈련 데이터를 증강하는 것이 가능하다.

### Idea

GAN의 잠재 공간을 조작하여 훈련 데이터의 편향성을 조절하는 것은 효율적인 데이터 증강 방법이라 할 수 있다. GAN을 이용하면 이미 가진 훈련 데이터만을 이용해서 새로운 이미지를 만들어 낼 수 있고, 따라서 훈련 데이터를 추가적으로 수집하기 위해 돈과 시간을 낭비할 필요가 적기 때문이다. 그러나 이러한 데이터 증강 방식을 위해 기존에 사용되었던 훈련 방법들은 연산량이나 GAN 모델의 구조적 복잡성의 측면에서 분명히 단점을 지녔다. 편향성을 제거하고자 하는 속성이 있을 때마다 새로운 GAN 모델을 만들어 훈련시켰으므로, 고려되는 속성이 여러 개일 경우에는 연산 시간이 길어진다는 문제가 있었다. 그리고 image-to-image translation GAN과 같은 복잡한 구조의 GAN을 이용하므로, 알고리즘의 복잡도가 증가한다는 문제도 있었다. 이러한 문제점들을 해결하기 위해 논문 저자는 데이터셋 전체에서 훈련된 단 하나의 GAN을 이용해 모든 속성의 편향을 개선하는 방법을 이용한다. 

## 3. Method

### 3-1. De-correlation definition

이 논문에서는 이미지의 속성과 레이블 간에 상관관계가 있는 경우를 다룬다. 예를 들어, 미국에서는 야외에서 선글라스를 쓰고 다니는 사람이 모자도 같이 착용하고 있는 경우가 많다. 그러므로, 아래의 사진에서와 같이, 선글라스를 쓰는 것(속성)과 모자의 착용 여부(레이블) 사이에 상관관계가 존재한다고 할 수 있다. 이러한 상황에서 야외 이미지들을 데이터 증강을 거치지 않고 바로 훈련 데이터로 사용한다면, 모자의 착용 여부를 판단하는 딥러닝 모델은 선글라스를 쓴 사람들보다 선글라스를 쓰지 않은 사람들에 대해 더 부정확한 예측을 내 놓을 수 있다. 그러므로 사전에 속성과 레이블 간의 상관관계가 제거되도록 훈련 데이터에 대해 데이터 증강 작업을 거치는 것은 중요하다. 

![Figure : Analytic expression of z'](../../.gitbook/assets/how-to-contribute/correlated.png)

데이터 증강을 거쳐 편향성이 제거된 데이터셋을 X<sub>aug</sub> 이라 하고, 공정성과 관련해서 고려하는 속성을 a 라고 하자. 딥러닝 모델이 임의의 x &in; X<sub>aug</sub> 에 대하여 예측하는 레이블 값을 t(x)라 정의하고 x의 예측 속성값을 a(x)라 하자. 가능한 레이블은 -1 또는 1 뿐이라고 가정하고, 속성값에 대해서도 똑같이 가정하자. 그렇다면 t(x)=1일 확률은 a(x)의 값과 무관해야 하며, 수식으로 표현하면 아래와 같다. 

![Figure : Analytic expression of z'](../../.gitbook/assets/how-to-contribute/decorrelation_condition.png)

### 3-2. De-correlation key idea

이 논문에서는 편향성이 제거된 데이터셋을 만들기 위해 예측 레이블은 동일하면서 예측 속성값은 서로 반대인 이미지 쌍을 생성하는 방법을 이용한다. GAN 모델이 기존 데이터셋에 대해 훈련을 마쳤다고 가정하자. 잠재 공간 내에서 임의로 z라는 점을 선택하면, GAN 모델은 점 z을 특정한 이미지로 변환할 것이다. 그 이미지에 대해 분류기 모델이 예측하는 레이블을 t(z)라 하고 예측 속성값을 a(z)라고 하자. 논문에서는 이때 아래의 조건을 만족하는 잠재 공간 내의 점 z’ 생성하여 z와 쌍을 이루게 한다.

![Figure : Analytic expression of z'](../../.gitbook/assets/how-to-contribute/z_prime_def.png)

이런 식으로 모든 z에 대해 쌍을 만든다면, 에측 레이블이 주어졌을 때 그에 해당하는 이미지들이 균등한 예측 속성 분포를 가질 것이다. 그러므로 최종적으로 얻어지는 데이터셋 X<sub>aug</sub>은 속성과 레이블 간의 상관 관계가 해소되었다고 할 수 있다. 아래의 사진은 (z, z') 쌍을 생성하는 식으로 데이터 증강을 함으로써 속성(안경 착용 여부)과 레이블(모자 착용 여부) 사이의 상관 관계를 제거한 결과를 보여준다.

![Figure : Analytic expression of z'](../../.gitbook/assets/how-to-contribute/augmentation_overview.png)

### 3-3. How to calculate z’

논문 저자는 z'을 해석적으로 구하기 위하여, 잠재 공간이 속성에 대해 선형 분리가 가능하다(linearly separable)는 가정을 도입한다. 그러면 두 함수 t(z)와 a(z)를 각각 초평면 w<sub>t</sub>와 w<sub>a</sub> 라 간주하는 것이 가능하다. 여기서 a(z)의 절편을 b<sub>a</sub>이라 할 때, z'의 식은 논문에 의하면 아래와 같다.

![Figure : Analytic expression of z'](../../.gitbook/assets/how-to-contribute/z_prime.png)




We strongly recommend you to provide us a working example that describes how the proposed method works.  
Watch the professor's [lecture videos](https://www.youtube.com/playlist?list=PLODUp92zx-j8z76RaVka54d3cjTx00q2N) and see how the professor explains.

## 4. Experiment & Result

{% hint style="info" %}
If you are writing **Author's note**, please share your know-how \(e.g., implementation details\)
{% endhint %}

This section should cover experimental setup and results.  
Please focus on how the authors of paper demonstrated the superiority / effectiveness of the proposed method.

Note that you can attach tables and images, but you don't need to deliver all materials included in the original paper.

### Experimental setup

This section should contain:

* Dataset
* Baselines
* Training setup
* Evaluation metric
* ...

### Result

Please summarize and interpret the experimental result in this subsection.

## 5. Conclusion

In conclusion, please sum up this article.  
You can summarize the contribution of the paper, list-up strength and limitation, or freely tell your opinion about the paper.

### Take home message \(오늘의 교훈\)

> 훈련 데이터셋에서 속성과 레이블 간의 상관관계를 제거함으로써 딥러닝 모델의 공정성을 향상시킬 수 있다.
>
> GAN의 잠재 공간을 이용하여 데이터셋의 편향성을 줄이는 작업은 시간적으로나 경제적으로나 효율적이다.
>
> 단 하나의 GAN 모델로 데이터 증강을 함으로써 연산량을 줄인다는 점이 흥미롭다.

## Author / Reviewer information

### Author

김대혁 \(Kim Daehyeok\) 

* KAIST 전기및전자공학부, U-AIM 연구실
* 관심 분야 : 음성인식 및 공정성
* 연락 이메일 : kimshine@kaist.ac.kr

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. Ramaswamy, Vikram V., Sunnie SY Kim, and Olga Russakovsky. "Fair attribute classification through latent space de-biasing." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
2. Official \(unofficial\) GitHub repository
3. Citation of related work
4. Other useful materials
5. ...

