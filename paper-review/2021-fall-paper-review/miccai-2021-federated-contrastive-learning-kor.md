---
description: Yawen Wu / Federated Contrastive Learning for Volumetric Medical Image Segmentation / MICCAI 2021 Oral
---

# **Federated Contrastive Learning for Volumetric Medical Image Segmentation \[Kor\]**

[**English version**](https://github.com/2na-97/awesome-reviews-kaist/blob/master/paper-review/2021-fall-paper-review/miccai-2021-federated-contrastive-learning-eng.md) of this article is available.

## 1. Problem Definition
  해당 논문에서는 의료 영상으로 인공지능 모델을 학습할 때 겪는 대표적인 두 가지 문제를 제시했다.
  > 1) 레이블(label)이 있는 데이터로 학습을 시키는 지도 학습(Supervised Learning)은 많은 분야에서 좋은 결과를 보이고 있으나, 의료 데이터의 레이블을 구하기 위해서는 의료 전문가들이 필요하며 상당한 시간을 요구하기 때문에 레이블이 있는 방대한 양의 의료 데이터셋을 찾는 것이 쉽지 않다.
  > 2) 환자들의 개인 정보 보호가 중요시 되기 때문에 병원 혹은 의사들 간에 의료 데이터를 서로 교환해서 보는 것이 어렵다.

  일반적으로 레이블이 부족한 데이터들로 학습을 시키기 위해 자기 지도 학습(Self-Supervised Learning)이라는 방법이 활발하게 연구되고 있다.
  자기 지도 학습 방법 중 하나인 **Contrastive Learning**은 레이블이 없는 방대한 양의 데이터를 서로 비교하는 방법을 통해 pre-training을 한 후 레이블이 있는 데이터셋에 대해서 fine tuning을 하며 학습시키는 방법이다.
  의료 데이터로 Contrastive Learning을 하기 위해서는 데이터가 부족하기 때문에 하나의 공통된 모델에 개인이 가지고 있는 데이터들로 학습을 시키는 **Federative Learning**을 도입하는 방법을 제시했다.
  Federative Learning 데이터를 직접적으로 공유하지 않기 때문에, 개인 정보는 보호하면서도 각각의 개개인들이 가지고 있는 데이터를 활용하여 하나의 공통된 모델을 학습한 후 Contrastive Learning을 접목시킨 **Federative Contrastive Learning**이라는 개념을 도입했다.
  


## 2. Motivation
### 2.1. Related Work
#### 2.1.1. Federated Learning
![FL](../../.gitbook/assets/federated-learning.png)

Federated Learning(FL)이란 위의 그림에서처럼 공통의 모델에 대해서 개인(client)이 가지고 있는 데이터로 하나의 모델을 학습시키게 된다. 이런 client들이 많아지게 되면 한 개인이 가지고 있는 데이터의 양은 많지 않더라도, client들이 가지고 있는 데이터 전체에 대해서 학습한 모델을 얻을 수 있다. 데이터의 직접적인 공유를 하지 않고도 전체 데이터에 대한 학습이 가능하다는 특성때문에, 환자의 개인 정보 보호가 필요한 의료 데이터의 경우 유용하게 사용될 수 있는 방법이다. 하지만 현존하는 FL은 모든 데이터에 대해서 레이블을 필요로 하는 supervised learning을 통해서 이뤄진다. 그렇기 때문에 labeling cost가 높은 의료 데이터의 경우 FL을 실전에서 사용하는 것은 현실적인 어려움이 있다는 문제점이 있다.


#### 2.1.2. Contrastive Learning
* **Self-Supervised Learning: Generative Learning vs Contrastive Learning**
![Generative-Contrastive](../../.gitbook/assets/gen-cont.png)

Self-Supervised Learning의 대표적인 두 가지 방법에는 Generative Learning과 Contrastive Learning이 있다. Generative Learning는 위 그림에서와 같이 입력 이미지를 넣은 후 생성된 출력 이미지에 대한 loss를 비교하는 방법이다. 반면에 Contrastive Learning은 입력받은 이미지들의 비슷함 정도를 비교하는 과정을 거친다. 이는 비슷한 이미지끼리는 \'positive sample\'으로 분류하고 다른 이미지끼리는 \'negative sample\'으로 분류하여 대조하는 과정을 통해서 representation을 학습한다.

* **Contrastive Learning: SimCLR**
<div align="center">
  <img width="50%" alt="SimCLR Illustration" src="https://1.bp.blogspot.com/--vH4PKpE9Yo/Xo4a2BYervI/AAAAAAAAFpM/vaFDwPXOyAokAC8Xh852DzOgEs22NhbXwCLcBGAsYHQ/s1600/image4.gif">
</div>
이미지 간의 representation 비교를 통해 자기 지도 학습을 하는 contrastive Learning의 대표적인 논문에는 SimCLR가 있다. SimCLR는 위 그림에서 같이 같은 이미지에 대해서 서로 다른 augmentation을 적용한 후, 같은 이미지에서 augmentation된 이미지들의 similarity는 높게 하되 다른 이미지와의 similarity는 낮추는 방향으로 학습을 할 수 있도록 한다. 이미지에서 유의미한 representation을 학습하는 함수와 같은 이미지에 대해서는 augmentation으로 인한 변화를 무시하는 방향으로 학습하고 다른 이미지에 대해서는 representation간의 간격을 크게 하여 label이 없더라도 이미지의 유의미한 feature를 뽑아내는 pre-train된 인코더를 얻을 수 있게 된다.

Contrastive Learning(CL)을 통해 많은 데이터로 학습한 pre-trained 인코더 weight을 얻게 된 후에는, 실제로 우리가 학습시키고자 하는 데이터에 대해서 학습을 시키며 fine tuning을 하는 과정을 거친다. 간단하게 말하자면, 우리가 모델을 처음부터 학습시키기 보다는 ImageNet과 같은 방대한 데이터로 학습한 weight를 기반으로 학습을 하는 것과 비슷하다고 생각하면 된다. CL은 레이블이 존재하지 않는 많은 데이터로부터 유의미한 feature를 추출하는 방법을 학습시켜 다양한 데이터셋에 pre-training weight으로 활용할 수 있도록 하여 모델을 처음부터 학습시키는 것보다 나은 성능을 보일 수 있도록 하는 것을 목표로 한다. SimCLR, MoCo, 그리고 BYOL과 같이 다양한 CL 방법들이 활발하게 연구되고 있는데, 이들은 supervised learning과 비슷한 수준의 정확도를 보이고 있다.


### 2.2. Idea
![FCL](../../.gitbook/assets/FCL.png)

본 논문에서는 Federated Learning의 단점과 Contrastive Learning의 단점을 보완한 후 두 학습 방법의 장점만 활용하여 합친 **Federated Contrastive Learning(FCL)** 이라는 방법을 제안한다. 저자가 주장하는 FL과 CL의 문제점은 다음과 같다.

* 개인이 가지고 있는 적은 양의 데이터로 CL을 할 경우, positive sample과 negative sample의 양도 부족하고 데이터 간의 다양성이 떨어지기 때문에 CL을 하는 데에 효과적이지 않다.
* 기존의 FL 방법으로 단순히 개인이 학습한 모델을 합치게 되면, 개인이 가지고 있는 데이터에 치중된 feature space가 만들어져 모델을 합치는 과정에서 feature space 간의 불일치가 일어나 실질적인 성능 향상 효과를 보기 어려울 수도 있다.

FCL에서는 이러한 단점들을 보완하기 위한 아이디어를 제시했는데, 데이터의 보안을 유지하면서도 client간의 데이터 교류를 할 수 있도록 하여 많은 양의 데이터를 기반으로 자기 지도 학습을 하는 효과를 볼 수 있도록 했다. 저자는 이러한 방법이 실전에서도 차용할 수 있는 의료 영상 딥러닝 모델이 될 것이라 기대했다.



## 3. Method
![overview](../../.gitbook/assets/overview.png)

FCL을 활용한 학습은 위 그림에서와 같이 레이블이 없는 많은 양의 데이터에 대해서는 FCL로 학습한 후, 레이블이 있는 소량의 데이터에 대해서 FCL로 학습한 인코더를 fine tuning하게 된다. Fine tuning을 하는 과정은 레이블이 있는 데이터로 쉽게 이루어지기 때문에 fine tuning을 위한 좋은 인코더가 될 수 있도록 학습하는 FCL 방법에 집중해서 살펴보도록 하자.

FCL에서는 본인이 학습하는 공간을 **\'local\'**, 다른 client가 학습하는 공간을 **\'remote\'** 라고 부른다. FL에서와 같이 local에서 학습한 후 이를 remote와 공유하게 되며, local과 remote의 데이터에 전체에 대해서 CL에서와 같이 비슷한 데이터끼리는 similarity가 높고 다른 데이터끼리는 similarity가 낮아지도록 학습이 진행된다. 각각의 local에서는 먼저 볼륨을 몇 개의 구역으로 나눈 후\(위 그림에서는 주황색, 청록색, 노란색, 초록색의 4가지 영역으로 나누었다.\), 구역의 순서는 유지하면서 각 구역에서 랜덤한 2D 샘플을 뽑아낸다. 이 2D 슬라이스들을 입력 이미지로 받아 학습된 U-Net의 인코더는 볼륨의 구조적 특징들을 뽑아낼 수 있게 된다. 모든 client가 개인이 가지고 있는 볼륨 데이터에 대해서 같은 과정을 거치면 client의 수만큼의 인코더가 만들어지게 된다. 이 때, 환자의 개인 정보 보호를 위해 직접적인 데이터 교환은 하지 않으면서도 다른 client가 가지고 있는 데이터도 학습시에 반영할 수 있도록 하기 위해서 local의 인코더에서 추출한 feature를 교환하는 방법을 생각했다. 개인의 encoder로 추출한 feature를 교환하는 방식을 통해서 client들이 가지고 있는 모든 데이터에 대해서 학습하는 효과를 누릴 수 있도록 한다. 이 경우 각각의 데이터로 학습한 모델을 단순히 합치는 것보다 client간 feature space의 일관성을 높이는 효과도 보일 수 있다.

![CL](../../.gitbook/assets/CL.png)

교환을 통해서 각각의 client들은 local과 remote의 feature들을 가지게 된다. 각 client들의 feature들은 **memory bank**에 저장되는데, memory bank에 저장된 local과 remote feature들을 가지고 각각의 client들은 위 그림과 같이 CL을 하게 된다. 본 논문의 경우, 같은 구역(partition)에 있는 2D 슬라이스들은 positive sample이 되고 다른 구역에 있는 2D 슬라이스들은 negative sample이 된다. 예시에서는 주황색 partition의 슬라이스들을 positive sample이라고 두었기 때문에, 주황색 구역에서 뽑은 2D 슬라이스의 feature끼리는 가까워지고 다른 색깔의 구역에서 뽑은 feature끼리는 서로 멀어지게 손실 함수가 계산된다. 이를 통해 각 구역마다의 고유한 representation을 학습하게 되는 것이다.

feature들을 뽑아내는 인코더는 두 가지 종류의 인코더가 있는데, contrastive loss를 계산할 때 사용되는 **Main Encoder**와 memory bank에 저장할 때 사용되는 **Momentum Encoder**가 있다.
* Main Encoder: 
* Momentum Encoder: 
