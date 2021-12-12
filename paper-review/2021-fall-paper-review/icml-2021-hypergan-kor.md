---
description: Ratzlaff et al. / HyperGAN - A Generative Model for Diverse, Performant Neural Networks / ICML 2019
---

#  HyperGAN \[Kor\]

[**English version**](./icml-2021-hypergan-eng.md) of this article is available.

##  1. Problem definition

HyperGAN은 신경망 매개 변수의 분포를 학습하기 위한 생성 모델이다. 특히, 컨볼루션 필터의 변수값들은 latent 층과 혼합(Mixer) 층으로 생성된다.

![alt text](../../.gitbook/assets/17/Screen%20Shot%202021-10-24%20at%207.17.22%20PM.png)

## 2. Motivation & Related work

서로 다른 무작위 초기화로부터 심층 신경망을 훈련시킬 수 있다는 것은 잘 알려져 있다. 또한, 심층 네트워크의 앙상블은 더 나은 성능과 견고성을 가지고 있다는 것이 추가로 연구되었다. 베이지안 딥 러닝에서는 네트워크 매개 변수에 대한 사후(posterior) 분포를 학습하는 것이 중요한 관심사이며, 드롭아웃은(dropout) 베이지안 근사를 위해 일반적으로 사용된다. 한 예시로서, 모델 불확실성을 추정하기 위한 MC dropout이 제안되었다. 그러나 모든 계층에 드롭아웃을 적용하면 데이터의 적합도가 낮아질 수 있으며 단일 초기화에서만 도달할 수 있는 모델 공간에 갇히게 된다.

또 다른 흥미로운 방향으로, 대상(target) 신경망에 대한 매개 변수를 출력하는 하이퍼 네트워크라는 분야가 연구되고 있다. 하이퍼네트워크와 대상 네트워크는 공동으로 훈련되는 단일 모델을 형성한다. 그러나 이전의 하이퍼네트워크는 사후분포를 만들기 위해 normalizing flow에 의존했고, 이는 모델 변수의 확장성을 제한했다.

본 연구는 고정된 노이즈 모델이나 생성 함수의 기능적 형태를 가정하지 않고 신경망의 모든 매개 변수를 한 번에 생성하는 접근법을 탐구한다. 저자는 normalizing flow 모델을 사용하는 대신 GAN을 활용한다. 이 방법은 여러 개의 무작위 초기화(앙상블) 또는 과거의 변형 방법을 사용한 훈련보다 더 다양한 모델을 제공한다.

### Idea

HyperGAN은 변수를 직접 모델링하기 위해 GAN의접근법을 활용한다. 그러나 이를 위해서는 훈련 데이터로 훈련된 많은 모델 매개 변수 세트가 필요하다. (image를 생성해내는 GAN을 위해서 real image가 필요한 것 처럼). 그래서 저자들은 다른 접근 방식을 취해서, 직접 대상 모델의 supervised 학습 목표를 최적한다. 이 방법은 normalzing flow를 사용하는 것보다 유연할 수 있으며 각 계층의 매개 변수가 병렬로 생성되기 때문에 계산적으로 효율적이다. 또한 많은 모델을 훈련시켜야 하는 앙상블 모델과 비교했을 때 계산적이고 메모리 효율적이다.

## 3. Method

Introduction 섹션의 위 그림은 HyperGAN의 구조를 보여준다. 표준 GAN과는 달리, 저자들은 s ~ *S*를 혼합 잠재 공간 Z에 매핑하는 fully connected 네트워크인 *Mixer* Q를 제안한다. 믹서는 한 계층의 출력이 다음 계층에 대한 입력이 필요하므로 네트워크 계층 간의 가중치 매개변수가 강하게 상관되어야 한다는 관찰에 의해 제안되었다. 혼합 잠재 공간 *Q*(z|s)에서 *Nd*차원 혼합 잠재 벡터를 생성하며, 이는 모두 상관관계가 있다(correlated). 잠재 벡터는 각각 *d*차원 벡터가 되는 *N* 레이어 임베딩으로 분할된다. 마지막으로 *N* 병렬 생성기는 각 N 계층에 대한 매개 변수를 생성한다. 이러한 방식은 매개변수의 극도로 높은 차원 공간이 현재 여러 잠재 벡터에 완전히 연결되어 있는 대신 별도로 연결되어 있기 때문에 메모리 효율적이다.

이제 새 모델이 학습 세트에서 평가되고 생성된 파라미터가 손실 *L*에 대해 최적화된다.

![alt text](../../.gitbook/assets/17/Screen%20Shot%202021-10-24%20at%207.17.36%20PM.png)

그러나 *Q*(z|s)에서 추출한 코드가 MLE에 따라 축소될 수도 있다(mode collapse). 이를 방지하기 위해 저자는 혼합 잠재 공간에 적대적 제약(adversarial constraint)을 추가하고 *P* 이전의 높은 엔트로피에서 너무 많이 벗어나지 않도록 한다. 이를 위한 HyperGAN objective는 다음과 같다:

![alt text](../../.gitbook/assets/17/Screen%20Shot%202021-10-24%20at%207.17.44%20PM.png)

*D*는 모든 두 분포 사이의 거리 함수일 수 있다. 여기서, 판별기 네트워크는(discriminator network) 적대적 손실과 함께 거리 함수를 근사하는 데 사용된다.


![alt text](../../.gitbook/assets/17/Screen%20Shot%202021-10-24%20at%208.39.57%20PM.png)

고차원 공간에서는 판별기를 배우기 어렵고 그러한 매개 변수에는 (이미지와 달리) 구조가 없기 때문에 잠재 공간에서는 정규화를 통해 이 문제를 해결한다. (자세한 방식은 논문에 언급되지 않습니다.)


## 4. Experiment & Result

### Experimental setup

- MNIST 와 CIFAR-10에서의 분류기 학습 및 성능평가
- 단순 1D 데이터 세트의 분산(variance) 학습
- 분포 외 예제의 이상 탐지(Anomaly detection of out-of-distribution examples)
  - MNIST에 대해 학습한 모델/notMNIST로 테스트한 모델
  - CIFAR-10 5개 클래스에 대해 학습한 모델 / 나머지 클래스에서 테스트된 모델

- baselines
  - APD(Wang et al., 2018), MNF(Louizos & Welling, 2016), MC Dropout(Gal & Ghahramani, 2016)


### Result

#### Classification 결과

![alt text](../../.gitbook/assets/17/Screen%20Shot%202021-10-24%20at%207.18.17%20PM.png)

#### Anomaly detection 결과

![alt text](../../.gitbook/assets/17/Screen%20Shot%202021-10-24%20at%207.18.38%20PM.png)

### Ablation Study

첫째, 목적에서 정규화 부분인 *D*(Q), *P*를 제거하면 네트워크의 다양성이 감소한다. 이를 확인하기 위해 저자들은 100개의 weight 샘플의 L2 norm을 측정하고 표준 편차를 평균으로 나눈다. 또한, 저자들은 시간이 지남에 따라 다양성이 감소한다는 것을 확인하고 학습의 조기 중단을 제안한다(early stopping). 다음으로 저자들은 믹서 *Q*를 제거한다. 정확성은 유지되지만 다양성은 크게 저하된다. 믹서가 없으면 유효한 최적화를 찾기 어렵다는 가설도 세웠는데, 믹서를 사용하면 다른 계층의 매개 변수들 사이에 내재된 상관관계가 최적화를 더 쉽게 만들 수 있다고 주장한다.

## 5. Conclusion

결론적으로 HyperGAN은 매우 강력하고 신뢰할 수 있는 앙상블 모델을 구축하기 위한 훌륭한 방식이다. 믹서 네트워크 및 정규화 용어를 사용하여 모드 붕괴(mode collapse) 없이 GAN 방식으로 파라미터를 생성할 수 있다는 장점이 있다. 그러나 이 작업은 MNIST 및 CIFAR10과 같은 작은 데이터 세트를 가진 소규모 대상 네트워크로 구축되어 간단한 분류 작업만을 수행한다는 단점이 있다. ResNets와 같은 대규모 네트워크에서 더 큰 데이터 세트를 사용하여 작업을 수행할 수 있다면 더 흥미로울 것이다.

### Take home message \(오늘의 교훈\)

하이퍼네트워크(Hypernetworks)를 GAN방식으로 학습시켜서 효과적인 베이지안 뉴럴 네트워크(bayesian neural networks) 만들 수 있다.

## Author / Reviewer information

### Author

**형준하 (Junha Hyung)**
* KAIST AI대학원 M.S.
* Research Area: Computer Vision
* sharpeeee@kaist.ac.kr


### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

[[1]](https://arxiv.org/abs/1609.09106)Ha, D., Dai, A. M., and Le, Q. V. Hypernetworks. CoRR

[[2]](http://bayesiandeeplearning.org/2018/papers/121.pdf)Henning, C., von Oswald, J., Sacramento, J., Surace, S. C., Pfister, J.P., and Grewe, B. F. Approximating the predic- tive distribution via adversarially-trained hypernetworks

[[3]](https://arxiv.org/abs/1710.04759)Krueger, D., Huang, C.W., Islam, R., Turner, R., Lacoste, A., and Courville, A. Bayesian Hypernetworks

[[4]](https://arxiv.org/abs/1802.09419)Lorraine, J. and Duvenaud, D. Stochastic hyperparameter optimization through hypernetworks. CoRR

[[5]](https://arxiv.org/abs/1711.01297)Pawlowski, N., Brock, A., Lee, M. C., Rajchl, M., and Glocker, B. Implicit weight uncertainty in neural networks

