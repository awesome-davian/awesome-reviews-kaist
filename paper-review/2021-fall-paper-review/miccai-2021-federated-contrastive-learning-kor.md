---
description: 'Yawen Wu / Federated Contrastive Learning for Volumetric Medical Image Segmentation / MICCAI 2021 Oral
---

# **Federated Contrastive Learning for Volumetric Medical Image Segmentation \[Kor\]**

[**English version**](https://github.com/2na-97/awesome-reviews-kaist/blob/master/paper-review/2021-fall-paper-review/miccai-2021-federated-contrastive-learning-eng.md) of this article is available.

## 1. Problem Definition
  해당 논문에서는 의료 영상으로 인공지능 모델을 학습할 때 겪는 대표적인 두 가지 문제를 제시했다. 첫 번째는 레이블(label)이 있는 데이터로 학습을 시키는 지도 학습(Supervised Learning)은 많은 분야에서 좋은 결과를 보이고 있으나, 의료 데이터의 레이블을 구하기 위해서는 의료 전문가들이 필요하며 상당한 시간을 요구하기 때문에 레이블이 있는 방대한 양의 의료 데이터셋을 찾는 것이 쉽지 않다는 것이다. 두 번째는 환자들의 개인 정보 보호가 중요시 되기 때문에 병원 혹은 의사들 간에 의료 데이터를 서로 교환해서 보는 것이 어렵다는 점이다.

  일반적으로 레이블이 부족한 데이터들로 학습을 시키기 위해 자기 지도 학습(Self-Supervised Learning)이라는 방법이 활발하게 연구되고 있다. 자기 지도 학습 방법 중 하나인 **Contrastive Learning**은 레이블이 없는 방대한 양의 데이터를 서로 비교하는 방법을 통해 pre-training을 한 후 레이블이 있는 데이터셋에 대해서 fine tuning을 하며 학습시키는 방법이다. 의료 데이터로 Contrastive Learning을 하기 위해서는 데이터가 부족하기 때문에 하나의 공통된 모델에 개인이 가지고 있는 데이터들로 학습을 시키는 **Federative Learning**을 도입하는 방법을 제시했다. 하지만 Federative Learning에서 데이터를 직접적으로 공유하지 않기 때문에, 개인 정보는 보호하면서도 각각의 개개인들이 가지고 있는 데이터를 활용하여 하나의 공통된 모델을 학습한 후 Contrastive Learning을 접목시킨 **Federative Contrastive Learning**이라는 개념을 도입했다.
  

## 2. Motivation
### 2.1. Related Work
#### 2.1.1. Federated Learning
![FL](../../../.gitbook/assets/federated learning.PNG)

Federated Learning(FL)이란 공통의 모델에 대해서 개인(client)이 가지고 있는 데이터로 학습을 시키게 된다. 이런 client들이 많아지게 되면 개인이 가지고 있는 데이터의 양은 많지 않더라도, client들이 가지고 있는 데이터 전체에 대해서 학습한 모델을 얻을 수 있다. 이러한 특성때문에, 환자의 개인 정보 보호가 필요한 의료 데이터의 경우 

#### 2.1.2. Contrastive Learning
