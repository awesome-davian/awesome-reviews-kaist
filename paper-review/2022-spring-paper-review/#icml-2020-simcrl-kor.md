---
description: Ting Chen al. / A Simple Framework for Contrastive Learning of Visual Representation / ICML '2020
---

# SimCLR [Korean]

(Description) Ting Chen / A Simple Framework for Contrastive Learning of Visual Representation / ICML '2020



# 1. Problem definition

대규모 비표시 이미지 데이터셋에 대한 사전 훈련은 여러 논문에서 입증된 것처럼 컴퓨터 비전 작업에서 성능을 향상시킬 수 있는 잠재력을 가지고 있습니다. 이러한 방법은  비주도 학습 문제를 비표시 이미지 데이터셋에서 대용 레이블을 생성하여 주도된 문제로 전환하는 기술의 계열인 자기주도 학습의 핵심입니다. 그러나, 현재 이미지 데이터에 대한 자기주도 기술은 복잡하여 아키텍처 또는 훈련 절차에 상당한 수정이 필요하며 널리 채택되지 않았습니다.

 해당 논문은 자기 지도 학습에서 주된 요소들을 연구하며, 이미지에 대한 자기주도적 표현 학습에 대한 이전의 접근 방식을 간소화할 뿐만 아니라 개선하는 방법에 대한 SimCLR이라는 기초 프레임 워크를 제시합니다. 또한, 이러한 방법론을 이용하여 SOTA 성능을 달성합니다. 본 논문의 접근 방식의 단순성은 기존의 주도 학습관에 쉽게 통합될 수 있다는 것을 의미합니다.

# 2. Motivation

## Related work

사람 없이 효과적인 시각적 표현에 대한 지도 학습은 오랫동안 연구로서 다루어져 왔습니다. 대부분의 주류 접근 방식은 생성적 또는 차별적이라는 두 가지 클래스 중 하나로 분류할 수 있습니다. 생성적 접근은 모델의 라벨링에 대해 생성하는 방법을 배우거나 그렇지 않다면 입력 공간의 픽셀을 모델링합니다. 하지만, 이러한 픽셀 단위의 생성은 계산적으로 매우 비싼 비용이 들 뿐만 아니라, 표현 학습에 꼭 필요하지 않을 수 있습니다. 

변별적인 접근 방법은 이미지에 대한 표현을 지도 학습의 목적함수와 비슷한 목적함수를 사용하여 학습합니다. 그러나 입력과 레이블이 모두 레이블이 지정되지 않은 데이터셋에서 오며, 이를 통해 네트워크를 학습한다는 점이 지도 학습과의 차이점입니다. 이러한 방식은 연구자가 정의한 업무(pretext task)를 만드는 것이 매우 휴리스틱하게 이루어졌습니다.

## Idea

본 연구에서는 시각적 표현의 대조 학습을 위한 SimCLR이라는 간단한 프레임워크를 소개합니다.

본 연구는 먼저 비 지정된 레이블의 데이터셋에서 이미지의 일반적인 표현을 학습한 다음, 소량의 레이블이 있는 이미지로 미세 조정하여 주어진 분류 작업에 대해 우수한 성능을 달성할 수 있습니다. 

SimCLR은 원본 데이터셋에서 예제를 무작위로 추출하여 간단한 확대 (임의 자르기, 임의의 색상 왜곡 및 가우시안 블러)의 조합을 사용하여 각 예제를 두 번 변환하여 두 세트의 해당 보기를 만듭니다. 개별 이미지의 이러한 간단한 변환의 근거는

1.  변환시 동일한 이미지의 일관된 표현을 장려하는 것입니다.
2. 사전 훈련 데이터에 레이블이 없기 때문에 어떤 이미지에 어떤 객체가 포함되어 있는지를 사전에 알 수 없습니다.
3. 우리는 이러한 간단한 변환이 신경망이 좋은 표현을 배우기에 충분하다는 것을 알았지만 더 복잡한 변환 정책도 통합될 수 있습니다. 

그런 다음, SimCLR은 ResNet 아키텍처 기반 합성곱 신경망 변형을 사용하여 이미지 표현을 계산합니다. 그 후 SImCLR은 Fully-Connected Network를 사용하여 이미지 표현의 비선형 투영을 계산합니다. 이 기능은 변하지 않는 기능을 증폭시키고 동일한 이미지의 다른 변환을 식별하는 네트워크의 기능을 최대화합니다. 우리는 대비 목표의 손실 함수를 최소화하기 위해 확률적 경사하강법을 사용하여 CNN과 MLP를 모두 업데이트합니다. 레이블이 없는 이미지를 사전 학습한 후에는 CNN의 출력을 이미지 표현으로 직접 사용하거나 레이블이 있는 이미지로 미세 조정하여 다운스트림 작업에서 우수한 성능을 얻을 수 있습니다. 

* 

# 3. Method

#### The Contrastive Learning Framework

![image-20220411174743658](/.gitbook/assets/7/AI604_2.png)

먼저, 본 논문에서 사용한 SimCLR이라는 학습 프레임워크입니다. 이는 최근 대조 학습 알고리즘에서 영감을 받았으며, 동일 데이터 예시에 대하여 다르게 증강된 뷰를 사이의 일치를 최대화합니다.  이는 두 잠재 공간에서의 대조 손실 함수를 통해 처리됩니다. 

Figure 2.는 4개의 요소로 구성된 SimCLR 프레임 워크를 구성합니다. 먼저 확률적 데이터 증강 모듈이 주어진 임의의 데이터 예시를 랜덤하게 두 개의 연관된 뷰로 만듭니다. 이것을 긍정 쌍이라고 부릅니다. 본 연구에서는, 3가지의 단순한 증강법을 순차적으로 이용합니다. 이는 랜덤 잘라내기, 랜덤 색 왜곡, 그리고 랜덤 가우시안 블러입니다. 

증강된 데이터 예제에서 표현 벡터를 추출하는 신경망 기반 인코더를 이용합니다. 이러한 인코더 f는 본 연구는 단순함을 추구하기에, 본 연구의 프레임워크는 제약 조건없이 다양한 네트워크 아키텍쳐를 고를 수 있으며, 일반적으로 사용되는 ResNet를 f로서 본 연구에서는 사용했습니다. 

작은 뉴럴 네트워크 프로젝션 헤드 g는 이미지의 표시를 대조 손실 함수가 적용되는 공간으로 매핑시킵니다. 즉, 1개의 은닉층과 ReLu 함수가 있는 MLP를 이용하여 손실 함수에 쓸 비선형 함수를 얻어냅니다. 

대조 손실 함수는 대조 예측 작업을 위해 정의됩니다. x_i와 x_j의 긍정쌍을 포함한 x_k이라는 집합이 주어졌을 때, 대조 예측 작업은 x_i가 주어졌을 때, x_i가 아닌 x_j를 x_k 집합에서 찾는 것에 주력합니다.

따라서, 긍정쌍을 가진 손실함수는 ![image-20220411180010373](/.gitbook/assets/7/AI604_3.png)

로서 정의됩니다.

아래 알고리즘은 SimCLR의 메인 학습 알고리즘입니다.

![image-20220411180040809](/.gitbook/assets/7/AI604_4.png)



#### Training with Large Batch Size

단순함을 유지하기 위해, 메모리 뱅크로 훈련하지 않습니다. 대신, 본 연구에서는 훈련 배치 크기를 256에서 8129까지 다양하게 합니다. 8192의 배치 사이즈는 증강된 뷰로부터 나온 각각의 긍정쌍마다 16382개의 부정 예시를 제공합니다. 

선형 학습률 스케일링을 이용한 SGD/Momentum를 적용했을 때, 큰 사이즈의 배치로 학습하면 안정하지 않을 수 있습니다. 따라서 본 연구에서는 LARS 최적화 방식을 사용합니다. 

# 

#### Data Augmentation for Contrastive Representation Learning

#### ![image-20220411181641032](/.gitbook/assets/7/AI604_5.png)

데이터 증강을 위해서는 랜덤하게 일어나는 crop과 resize, 그리고 색 왜곡과 가우시안 블러 등을 하였습니다.

본 연구에서는 데이터 증강의 영향을 체계적으로 연구하기 위해, 랜덤하게 일어나는 crop과 resize, 색 왜곡, 가우시안 블러 외에도 데이터의 회전, 컷아웃, 대비 및 채도 변화 등과 같은 변화를 포함하여 증강하였다. 개별적으로 혹은 짝으로 데이터 증강을 적용할 때, 본 프레임 워크의 성능을 조사하였다. ImageNet의 이미지들은 각자 다른 사이즈들이므로, 본 연구에서는 항상 자르기와 resize를 하였다. 

단일 변환은 최고의 표현을 제공하는 예측 작업을 정의하는 데 충분하지 않다. 하지만, 랜덤 크랍과 랜덤 색 왜곡이라는 두 가지 변환이 가장 두드러지는 영향을 주었다. 자르기나 색 왜곡이 자체적으로 고성능을 내주진 않지만, 이 두 가지 변형을 구성하면 최신 결과를 얻을 수 있다.

#### Architectures for Encoder and Head

SimCLR에서는 대조 학습 목표에 대한 손실함수가 계산되기 전에 MLP 기반 비선형 투영이 적용되어 각 입력 이미지의 변하지 않는 특징을 식별하고 동일한 이미지의 다른 변환을 식별하는 네트워크의 능력을 최대화합니다. 본 연구가 했던 실험에서, 이러한 비선형 투영법을 사용하면 표현 품질을 향상하고 SImCLR 학습된 표현에 대해 훈련된 선형 분류기의 성능을 10% 이상 향상하는 데에 도움이 된다는 것을 알았습니다.   

![image-20220411183518877](/.gitbook/assets/7/AI604_6.png)



#### Loss Functions and Batch Size

 본 연구에서는 NT-Xent 손실 함수를 다른 일반적인 대조 손실 함수들(로지스틱 손실 함수)과 비교했습니다. 

![image-20220411183639657](/.gitbook/assets/7/AI604_7.png)

Table 2는 손실 함수의 인풋의 그래디언트와 목적함수를 보여줍니다. 

본 연구에서는 L2 정규화와 적절한 temperature가 모델 학습에 도움을 줄 수 있다는 것을 알았습니다.







# 4. Experiment & Result



If you are writing **Author's note**, please share your know-how (e.g., implementation details)

This section should cover experimental setup and results. Please focus on how the authors of paper demonstrated the superiority / effectiveness of the proposed method.

Note that you can attach tables and images, but you don't need to deliver all materials included in the original paper.

## Experimental setup

본 연구는 평가를 위한 프로토콜을 제시했습니다. 이는 다른 디자인 선택에 대해 본 연구의 프레임워크를 이해하는 데에 초점을 두었습니다.

- Dataset

  데이터 셋으로는 ImageNet ILSVRC-2012 dataset은 본 연구의 대부분에서 활용되는 비지도학습에서 사용되었습니다. 즉, 인코더 네트워크 f (Figure 2)를 라벨 없이 학습시키는 데에 확용했습니다. 몇몇 추가적인 사전 훈련을 위해서는 CIFAR-10 데이터셋을 이용하였으며, 전이 학습을 테스트해보았습니다.

- Baselines

   MoCo나 PIPL, CPC v2, Local Agg, BigBiGAN.

- Test setting

  Optimizer : LARS optimizer

  learning rate : 4.8 ( = 0.3 * BatchSize/256) 

  weight decay : 10^-6

  Batch Size : 4096 for 100 epochs.

  Using linear warmup for the first 10 epochs.

- Evaluation metric

  

- Default Setting

  데이터 증강을 위해 랜덤 크랍, 랜덤 리사이즈, 색 왜곡, 가우시안 블러를 이용했으며, 베이스 인코더 네트워크 (Figure 2의 f) ResNet-50을 이용했습니다. 그리고 2-layer의 MLP 프로젝션 헤드인 g를 이용했으며 이는 이미지 표시를 128 차원의 latent space로 전이시킵니다. 

  Loss function으로는 NT-Xent를 이용했다.

## Result

* #### Comparison with State-of-the-art

  본 논문에서 제안된 framework를 통해 SOTA 성능을 달성했다는 것을 결과로 보여주고 있다.

![image-20220411174153866](/.gitbook/assets/7/AI604_8.png)



![image-20220411174218031](/.gitbook/assets/7/AI604_9.png)



![image-20220411174232961](/.gitbook/assets/7/AI604_10.png)







# 5. Conclusion

#### 

본 논문은 대조적 시각적 표현 학습을 위한 간단한 프레임 워크와 인스턴스화를 제시했습니다.

우리 연구 결과를 결합하여 자체 지도, 대조 학습 및 전이 학습이 이전 방법보다 상당히 개선되었습니다.

본 연구는 표준 지도학습과는 다릅니다. 

본 논문은 아래와 같은 세 가지를 보였습니다.

* 구성 데이터 증강은 효과적인 예측 작업을 정의하는 것에 아주 중요한 역할을 합니다. 

* 표현과 대조 손실 사이에 학습 가능한 비선형 변환을 도입하면 학습된 표현의 품질이 크게 향상됩니다.
* 대조 교차 엔트로피 손실 함수를 사용한 표현 학습은 정규화된 임베딩과 적절히 조절된 온도 매개 변수로부터 이점을 얻습니다. 

* 대조 학습은 더 큰 배치 사이즈와 더 큰 트레이닝 스텝에 대하여 지도학습과 비교했을 때에 큰 이점을 가져갑니다.

  #### 







## Take home message (오늘의 교훈)

Please provide one-line (or 2~3 lines) message, which we can learn from this paper.

> All men are mortal.
>
> Socrates is a man.
>
> Therefore, Socrates is mortal.

# Author / Reviewer information



You don't need to provide the reviewer information at the draft submission stage.

## Author

**김하준 (Kim Hajun)**

- 

  KAIST Mechanical Engineering

- 

  I'm studying about robotics. I'm interested in control, path planning, state estimation with optimization or learning based framework.

- 

  

- 

  **...**

## Reviewer

1. 1.

   Korean name (English name): Affiliation / Contact information

2. 2.

   Korean name (English name): Affiliation / Contact information

3. 3.

   ...

# Reference & Additional materials

1. 1.

   Citation of this paper

2. 2.

   Official (unofficial) GitHub repository

3. 3.

   Citation of related work

4. 4.

   Other useful materials
