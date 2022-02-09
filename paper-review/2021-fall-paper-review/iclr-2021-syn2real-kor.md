---
description: Chen et al. / Contrastive Syn-To-Real Generalization / ICLR 2021
---

# Contrastive Syn-To-Real Generalization \[Kor\]



**[English version](./iclr-2021-syn2real-eng.md)** of this article is available.




##  1. Problem definition



본 논문은 mesh image로 구성된 소위 '인위적인(synthetic)' dataset으로 모델을 학습하고 바로 real image에 모델을 적용했을 때 모델의 성능을 높이고자 합니다. 여기에서 모델은 classification과 segmentation 모델을 다룹니다. 

synthetic data와 real data 각각을 하나의 domain으로 보고 두 도메인 간의 차이를 줄여줄 수 있는 일반화 알고리즘, 즉 domain generalization이라는 문제를 synthetic image와 real image의 세팅에서 풀고자 합니다. 

이를 수식으로 나타내면 아래와 같습니다. 

$$min_{h} E_{(x,y)\in S_{test}}[L(h(x), y)]$$, where $$S_{train}=synthetic\_images, S_{test}=real\_images$$​ .



이러한 문제 정의를 논문에서는 **Zero-shot domain generalization on synthetic training task** 로 정의했는데요, 이 문제는 3 파트로 나눌 수 있습니다.



**Domain generalization (DG)**: 

DG의 목표는 학습 중에 본 데이터 셋과 다른 분포를 가지는 데이터에서도 모델이 예측을 잘하게 하는 것을 목표로 합니다[1]. 예를 들어, 일반적인 DG 데이터 셋은 아래와 같이 스케치, 만화, 그림으로 학습을 하고 모델을 평가할 때는 실제 사진을 사용하곤 합니다. 



![Domain generalization의 데이터 셋 예시](/.gitbook/assets/32/DG_example.png)

*(Generalizing to Unseen Domains: A Survey on Domain Generalization / Wang et al. / IJCAI 2021)*



만약 우리가 $$M$$개의 학습(soruce) 도메인이 주어졌다면, 학습 데이터를 아래와 같이 정의할 수 있습니다. 

$$S_{train} = \{S^i | i=1, ..., M\}$$, where $$S^i = {(x^i_j, y^i_j)}^{n_i}_{j=1}$$ denotes the i-th domain and $$n_i$$ is image set size.

*$$(x^i_j,y^i_j)$$ 는 i-th domain의 j 번째 이미지 샘플이 될 것입니다.*



Domain generalization 이 목표는 어떤 데이터에도 균일한 성능을 내고 일반화 가능한 펑션 $$h$$ 를 학습하도록 하는 것입니다. 이때 $$h$$는 학습 도메인 $$X$$  (i.e., $$S_{train}$$) 과 타깃 도메인 $$Y$$ (i.e., $$S_{test}$$) 을 연결해주 는 함수로,  $$X\to Y$$ 의 관계를 학습할 때 보지 않은 Y 도메인의 데이터 셋에서의 예측 오류를 최소화하는 방식으로 학습이 됩니다. 이를 수식으로 표현하면 아래와 같이 정의할 수 있습니다.



$$P^{test}_{XY}\neq P^i_{XY}$$  for  $$i \in \{1, ... , M\} $$):

$$min_{h} E_{(x,y)\in S_{test}}[L(h(x), y)]$$, 

where $$E$$ is the expectation, $$L(\cdot, \cdot)$$ is the loss function, and $$h$$ is our main training function. 



위에 있는 데이터 셋을 예시로 들자면, 스케치, 만화, 그림 image 만으로 학습한 모델은 real image에 테스트했을 때의 loss를 최소화하는 것을 학습하게 되는 것이지요.



**Synthetic training dataset**: 

이 논문에서는 domain generalization 중에서도 synthetic-to-real 세팅, 즉 학습 데이터 셋이 synthetic image이고 테스트 데이터 셋이 real image 일 때의 문제를 다룹니다. 

$$S_{train}=synthetic\_images, S_{test}=real\_images$$

이러한 문제 세팅을 syn2real이라고 부르는데, 가장 대표적인 데이터 셋은 VisDA-17 dataset입니다.

*ICLR open review에서 한 리뷰어는 일반적인 DG setting과 달리 train, test dataset 세팅을 synthetic과 real dataset으로 제한하여서 본 논문의 중요도가 작아졌다고 말하기도 했습니다.*



![VisDA-17 dataset of classification task](/.gitbook/assets/32/vis_da.png)





**Zero-shot learning**: 

syn2real problem setting 중에서도 zero-shot learning 학습방법을 채택한 이 논문. 뭐가 다른 걸까요?

사실 기존 syn2real 문제에서는 real image로 이루어진 validation dataset으로 모델에 대한 fine-tuning을 진행하곤 했습니다. 하지만, 이 논문의 경우 그러한 fine-tuning step 없이 synthetic dataset으로 학습한 모델을 $$S_{test}$$에 바로 적용하여 모델의 성능을 측정합니다. 



![VisDA-17 dataset of classification task on zero-shot learning](/.gitbook/assets/32/vis_da_2.png)

이 경우 validation dataset을 사용하지 않습니다(빨간 X). 파란 화살표와 같이 train dataset에 대해서 학습을 진행한 후 바로 test datset에 대해서 성능을 평가합니다. 

예를 들어 소영이가 말과 아이는 본 적이 있는데 얼룩말을 본 적이 없다고 가정해 봅시다. 그럼 얼룩말을 처음 본 소영이는 얼룩말을 보고 어떻게 생각해야 옳을까요? 아마 "말인데 줄무늬가 있는 동물"이라고 생각하는 것이 알맞을 겁니다. 얼룩말을 보고 아기라고 생각한다면 소영이는 교육을 다시 받아야 할 것 같습니다. [source](https://www.quora.com/What-is-zero-shot-learning)

여기서 '소영이'가 모델, '아기, 말'이 학습 데이터 셋, '얼룩말'이 테스트 데이터 셋으로 생각할 수 있습니다. 

즉, zero-shot learning은 모델이 기존 학습 도메인에서 배운 지식을 활용하여 한 번도 학습하지 않은 도메인의 데이터에 대해서도 어느 정도의 성능을 내게 하는 것입니다. 세상에 존재하는 모든 물체 (동물, 식물 등)의 클래스를 데이터 셋으로 가지고 있기는 어렵습니다. 따라서, 모델이 보지 못한 클래스의 이미지를 인풋으로 받았을 때, 보유하고 있는 데이터 중에 가장 유사한 클래스로 이미지를 구분함으로써, 어떤 상황에서도 모델이 일정 수준 이상의 성능을 가지도록 학습시키고자 하는 것입니다. 





## 2. Motivation



### Related work

Related work 섹션은 두 갈래로 나눌 수 있습니다. 첫 번째는 태스크 로서의 domain generalization이고,  두번째는 학습 방법으로서의 contrastive learning입니다.



**1. Domain generalization**

타깃 도메인에 대한 아무런 지식 없이 모델을 일반화하는 것은 듣기만 해도 상당히 어려운 문제입니다.

이 문제를 해결하기 위해 다양한 모델들이 제안되었는데요, 오늘 소개하는 논문에서 비교 모델로 사용하며 가장 유사한 세팅을 가진 두 논문만 소개하도록 하겠습니다. 

만약 이 주제에 관심이 있으시다면, [이 깃헙 페이지](https://github.com/amber0309/Domain-generalization )를 참조하시면 좋을 듯합니다. 



**Yue et al.** [2] [paper link](https://arxiv.org/abs/1909.00889)

이 연구는 syn2real 일반화 문제를 semantic segmentation 태스크에 집중하여 해결하고자 한 논문입니다. synthetic, real 도메인 간의 격차를 줄이기 위해 synthetic image를 real image의 스타일로 랜덤하게 변환시키는 방법을 사용하였고, 이를 통해 모델로 하여금 도메인에서 변하지 않는 진실과 같은 representation을 배우도록 하였습니다. 즉 real에서 synthetic dataset으로 스타일 정보를 전이한 것(transfer)이라고 볼 수 있습니다. 

이 논문은 좋은 성능을 보여주었으나, 스타일 정보를 이전할 때 결국 real dataset의 스타일을 참조하고 이 과정에서 pyramid consistency라는 추가적인 loss를 흘려주게 되어 꽤 많은 연산을 요구하게 됩니다. 참고로, 이 모델을 학습할 때에는 8 NVIDIA Tesla P40 GPUs and 8 NVIDIA Tesla P100 GPUs 을 사용하였다는 것을 보면, 얼마나 많은 연산을 필요로 하는 지 알 수 있을 것 같습니다. 



**Automated Synthetic-to-Real Generalization (ASG)** [3] [paper link](https://arxiv.org/abs/2007.06965)

*사실 ASG 논문은 저희가 리뷰하는 논문 저자의 이전 논문입니다. 이를 통해 본 저자들이 얼마나 syn2real 태스크에 관심이 있는지를 알 수 있겠죠?*

이 논문은 syn2real generalization에 대하여 논의한 첫 논문이기도 합니다. ASG는 synthetic dataset에서 학습된 모델이 유사한 representation을 유지하도록 모델을 학습하는데 집중하였고, learning-to-optimize, 즉 레이어 별로 learning rate를 다르게 하는 학습방법을 제시하였습니다. 

즉, $$M, M_{o}$$라는 두 모델이 주어졌을 때, generalization이라는 목표를 이루기 위해 두 가지 loss를 모델로 하여금 학습하게 합니다. 이 과정을 설명하면 아래와 같이 설명할 수 있을 텐데요,

* $$M, M_{o}$$ 은 모두 ImageNet pretrained model입니다.
* 이때 $$M_0$$ 는 $$M$$의 파라미터를 유지하면서 synthetic image에 맞게 조금씩 학습됩니다.
  * 주어진 태스크 (즉 prediction이나 segmentation)에 대해서 $$M_0$$는 cross-entropy loss로 업데이트됩니다 (첫 번째 loss).
* 또한 $$M, M_0$$간의 transfer learning을 원활히 하기 위해 $$M, M_0$$ 간의 KL-divergnece loss를 최소화하도록 모델을 학습합니다 (두 번째 loss). 

이 논문이 처음으로 syn2real이라는 태스크를 제안하였지만, 여전히 일반 domain generalization 연구와 같이 세밀한 학습 하이퍼 파라미터 (hyperparameter) 조정이 필요하다는 한계가 있습니다. 



**2. Contrastive learning**

*참조: https://nuguziii.github.io/survey/S-006/*

Contrastive learning는 metric learning이라고 하여, 예를 들어 각 이미지를 고차원의 공간으로 매핑했을 때의 해당 공간이 유사한 클래스, 성격을 가진 이미지(레트리버 1, 레트리버 2)는 가까이하게 하고 서로 다른 클래스, 다른 성격의 이미지(레트리버 1, 고양이 2)는 멀게 하도록 하는 '공간'을 학습하는 방법입니다. 이 방법은 사실 self-supervised 방식과 결합하여 contrastive self-supervised learning이라는 메서드로 동작하곤 합니다. 

![Contrastive Self-supervised learning](/.gitbook/assets/32/constrastive_self_supervised_sample.png)

*ref: https://blog.naver.com/mini_shel1/222520820060*

positive, negative sample을 만드는데, 기준이 되는 레트리버 이미지 1을 anchor라고 한다면, 레트리버 이미지를 회전시키고 확대해서 자르고 하면서 원본 이미지를 변형시킨다고 해도 이 이미지가 레트리버 클래스라는 성질은 변하지 않을 테고 이는 positive sample이 됩니다. 반대로 고양이 이미지가 있다면 이걸 어떻게 변형시킨다고 해도 레트리버 1과는 다른 negative sample이 되겠죠. 

그럼 어떠한 가상 공간에서 anchor 가 positive sample에 가깝게 하고, negative sample 과는 멀게 하도록 모델을 어떻게 학습시킬 수 있을까요? Contrastive learning에서 가장 유명한 loss는 NCE loss와 InfoNCE loss이며, 다양한 방법론들이 제안되어 왔습니다. 여기서는 저희가 리뷰하는 논문에서 다루는 메서드 두 가지만 간략히 소개하고 넘어가도록 하겠습니다.



**InfoNCE loss** [4] [paper link](https://arxiv.org/abs/1807.03748) 

![InfoNCE loss](/.gitbook/assets/32/info_nce.png)

이미지의 $$L_N$$ 가 InfoNCE loss인데요, 이미지에서 보이는 것처럼 positive sample들의 representation 간의 거리는 가까이하고 서로 다른 성질의 이미지 즉 negative sample들의 representation 간의 거리는 멀게 하는 loss입니다. 예를 들어 레트리버 클래스의 이미지는 서로 유사한 임베딩 representation을 가지지만 고양이의 것과는 달라야 한다는 점을 기반으로 한 거죠. 두 임베딩 벡터, representation 간의 유사도는 보통 consine-similarity로 계산을 하게 됩니다. 



**MoCov2** [5] [paper link](https://arxiv.org/abs/2003.04297)  | [git](https://github.com/facebookresearch/moco)

![SimCLR and MoCo](/.gitbook/assets/32/moco.png)

[SimCLR](https://github.com/google-research/simclr) (a)라는 contrastive learning approach에서 문제가 된 부분은 모델을 학습할 때 최대한 많은 postiive, negative sample을 모으고 batch size를 최대한으로 키워야 한다는 점이었습니다. 이미지의 (a) 와 같이, 모든 데이터 셋을 연산장치에 올린 다음 한 번에 encoder를 학습시켜야 했거든요. 참고로 원래 논문에서는 batch-size를 10,000으로 키웠을 때 가장 좋은 성능을 냈습니다. 하지만 모두 아시다시피, 이렇게 연산을 할 경우 연산이 너무 많아지고 학습하기 어렵다는 단점이 존재합니다. 

반면, MoCo (b)는 momentum encoder와 negative sample을 queue 형태의 dictionary로 저장하는 것을 제안합니다. SimCLR와 달리 한 번에 많은 postivie, negative sample을 넣을 필요가 없게 된 거죠. MoCo의 두 인풋은 모두 postivie sample(왼쪽 encoder에 들어가는 샘플이 anchor, 오른쪽 momentum encoder에 들어가는 샘플이 positive sample이라고 이해하면 편할 겁니다)이고 negative sample은 사전에 저장해둔 queue dictionary에서 불러옵니다. 그리고 InfoNCE loss를 계산해서 샘플들의 representation을 학습합니다. 

MoCov2는 이름과 같이 MoCo라는 모델의 업그레이드 버전으로, 모델의 마지막 딴에 MLP head와 data augmentation을 추가한 버전입니다 (이미지의 affinity matrix).



**3. Hyperspherical energy**

*Learning towards Minimum Hyperspherical Energy / Liu and Lin et al. / NeurIPS 2018*

이미지 샘플을 encoder을 태워서 피처임베딩 벡터를 만들었다고 칩시다. 그러면 이 임베딩 벡터들이 매핑된 공간에서 얼마나 잘, 골고루 분포해있는지를 어떻게 계산할 수 있을까요? 저희가 리뷰하는 논문에서는 hyperspherical enery (HSE)를 기준으로 삼았습니다.

![HSE score (eq.1)](/.gitbook/assets/32/eq1.png)

HSE가 등장한 원래 논문은 사실 hyperspherical enery라는 것을 최소화하여 모델을 regluriation 하고자 했는데요, 즉 neural net의 각 레이어의 뉴런끼리의 diversity를 조절하는 기준으로 HSE를 사용한 것이죠. HSE가 어디서 나왔느냐 하면, Thomson problem에서 영감을 받았다고 하네요. Thomson problem이라는 건 어떻게 N 개의 전자를 구에 최대한 균일하게 분포하게 하여 잠재적 에너지를 최소화할 것인가?-라는 문제입니다. 



![Feature embedding with and without minizing HSE score method](/.gitbook/assets/32/energe.png)

사실, 저희가 여기서 기억해야 할 포인트는 하나입니다.

높은 에너지$$E_s$$ 는 한쪽에 편향되어 있는 경향이 강하고, (Figure 4 - a), 에너지 스코어가 낮을수록 뉴런들이 다양하고 균일하게 분포(Figure 4 - b) 했다는점이죠 .





### Idea

자, related work 섹션이 조금 길었네요. 다시 이 논문의 목적을 되새겨 봅시다.

**Zero-shot domain generalization on synthetic training task**. 

이 논문의 저자들은 세 가지 다른 데이터 셋에서 학습된 임베딩 벡터들의 분포를 시각화했는데요, 각각 ImageNet, VisDA17-real dataset, VisDA-17-synthetic dataset입니다.



![Distribution of embedding vectors](/.gitbook/assets/32/fig2.png)

Fig.2 에서 보이듯, real image에서 학습된 임베딩 representation들은 상대적으로 고르게 분포해있지만 (Fig.2- a, b), synthetic image에 학습된 representation들은 한곳으로 몰려있습니다(Fig.2 - c). 

이 관찰에 기반하여, 본 논문은 synthetic dataset의 피처 공간이 한쪽으로 쏠린 분포를 가지는 것이 syn2real generalization 태스크에서 낮은 성능을 보여주는 이유라고 가정하였습니다. 따라서, 본 논문은 synthetic 과 real 도메인의 임베딩을 유사하게 만들어야 할 뿐 아니라, synthetic 도메인의 피처 공간을 잘 퍼뜨려야 한다고 주장합니다 (위와 같은 collapse를 피하기 위해서요).



이전 연구들의 한계와 본 논문의 노벨티는 아래와 같이 정리할 수 있습니다.

**Limitation of previous works**

* 이전 연구 대부분은 real2real transfer learning에 집중하였고 이 세팅의 하위 태스크(예를 들어, classification, segmentation)의 성능을 향상시키는 것을 목적으로 하였습니다.
* ASG 모델의 경우 syn2real generalization 문제에서 synthetic과 real 도메인 간의 임베딩 피처 거리를 최소화하는데 집중하였습니다.

**Improvements of this work**

* 이 논문은 synthetic-to-real transfer learning 세팅을 classification, segmentation 두 가지 태스크에 제안합니다.
* syn2real 임베딩간의 피처 거리를 줄일 뿐만 아니라, synthetic 피처 임베딩들이 한 곳으로 집중되는 것을 피하는 *push and pull* 방법론을 제안합니다.





## 3. Method

이 섹션에서는 이 모델이 어떻게 동작하고 어떠한 방식으로 학습이 되는지를 이해해 봅시다.



자세한 과정에 들어가기 전에, 몇 가지 중요한 전제들을 되짚어볼까요?

* 우리가 학습과정에서 볼 수 있는 건 뭘까요?
  * Synthetic image $$x$$  와 그것의 정답인 $$y$$  와 (즉, 이미지의 정답 클래스나 segmented된 결과 이미지겠죠?)
  * ImageNet에서 사전학습(pretrained) 된 weight를 지니고 있는 Encoder만을 볼 수 있습니다. 

* 우리 모델은 어느 데이터 셋으로 성능을 평가받나요? 
  * Real image 와 그것의 정답 (ground-truth)입니다.



### Overview and notions

이 논문의 가장 주가 되는 방법은 push and pull입니다

* Pull: synthetic 데이터의 피처와 ImageNet-pretraiend 피처 간의 거리를 minimize 합니다.
* Push: synthetic 도메인에서 이미지들의 피처들이 서로서로를 밀어내어 어느 정도의 거리를 만들게 합니다.

ASG 모델 (Fig.3-a)와 비교하면, 본 논문의 아키텍처 (Fig.3-b)는 아래와 같습니다. 



![Model architecture](/.gitbook/assets/32/fig3.png)

Notions *(편의상 영어로 작성하겠습니다.)*

* $$f_{e,o}$$ : ImageNet-pretrained model, $$f_e$$ : synthetically trained model
* $$L_{syn}$$ : task loss ($$L_{task}$$), loss of classification or segmentation
* $$x^a$$ : input synthetic image, this becomes **anchor** in contrastive learning
  * embeddings of $$x^a \to$$  $$z^a$$ from $$f_e$$  , $$z^+$$ from $$f_{e,o}$$
* K negative images $$\{x^-_1, ... , x^-_K\}$$  and its embeddings $$\{z^-_1, ... , z^-_K\}$$  for every anchor $$x^a$$ 
* $$h/\tilde{h} : \mathbb{R}^C \to \mathbb{R}^c$$ , non linear projection head with {FC, ReLU, FC} layers.  



만약 우리가 anchor image의 임베딩 벡터를 얻는 과정은 아래와 같이 기술될 수 있습니다.

**$$z^a = f_e \circ g \circ h(\tau(x^a))$$**

그럼 각각의 $$f, g, h, \tau$$ 가 뭔지 하나씩 알아볼까요?



### $$h(\tau(x))\to$$ Augment image and model 

**Image augmentation: $$\tau$$**

![Image augmentation example](/.gitbook/assets/32/image_aug.png)

*image from  https://nuguziii.github.io/survey/S-006/*



Image augmentation은 모델의 성능을 향상시키기에 좋은 방법입니다. 모델로 하여금 다양한 상황에 있는 이미지를 보게 함으로써 다양한 인풋 조건들에 모델이 균일한 성능을 낼 수 있도록 도와주는 것이죠. 즉, 모델의 generality를 향상시킵니다.

본 논문에서는 image augmentation 기법으로 [RandAugment](https://arxiv.org/abs/1909.13719) 을 사용하였습니다. RandAugment을 간단히 말하자면 각도 비틀기, 이동, 색 변환과 같은 여러 가지 augmentation 기법들을 랜덤하게 순서를 정해서 인풋 이미지를 변환시키는 방법입니다. 



**Model augmentation: $$h$$**

![Representations for each samples](/.gitbook/assets/32/eq2.png)

인풋 이미지를 조작할 뿐만 아니라, 이 논문은 모델도 조금씩 조작을 합니다. Fig.3-b를 보면 $$f_{e,o}$$  뒤에 $$\tilde{h}$$ 라는 레이어가 하나 더 붙어있는데요, $$\tilde{h}$$ 라는 non-linear projection 레이어를 추가함으로써 모델 $$f$$ 를 조작하려고 한 겁니다.

피처 임베딩들의 다양성을 높이기 위해 저자들은 mean-teacher styled moving average model을 사용했는데요, 즉 exponential moving average를 통해 모델을 추가적으로 조작하고자 한 것이죠. 



moving average를 처음 들어보시는 분도 계실 것 같습니다.

예를 들어, $$W_0$$ 가 initial state이고 $$W_k$$ 는 $$k$$-th batch dataset에서 학습된 parameter라고 합시다.

이때 moving average function은 $$W_0$$를 아래와 같은 방식으로 업데이트합니다.

 $$W_0 = \alpha * W_0 + \beta * W_k$$ where $$k \in \{1, ..., K\}, \alpha + \beta = 1$$ .

일반적으로 $$\alpha=0.99 , \beta=0.01$$ 로 세팅을 하는데요, exponential moving average function은 특이하게도 $$k $$ 가 증가함에 따라 $$\beta$$ 의 값이 감소합니다 (e.g., 처음엔 0.01이었다가 나중엔 0.001이 되는 거죠). 이렇게 되면 모델로 하여금 현재 다루고 있는 데이터에 집중을 하고 과거의 데이터에는 신경을 덜 쓰게 합니다.

우리는  $$W_0 \to \tilde{h}$$ , $$W_k \to h$$  라고 이해할 수 있을 것 같습니다. 즉, ImageNet (과거 데이터)에 사전 학습된 피처 임베딩의 정보를 조금씩 synthetic dataset (현재 데이터)에 맞게 튜닝을 시키는 것이죠.





### *Train $$f_e\to$$* Contrastive Loss 

**Loss**

여러 contrastive learning 방식 중에서, 본 논문은 InfoNCE loss를 사용합니다 (Related work 섹션에서 이미 다뤘습니다).

![Contrastive loss](/.gitbook/assets/32/eq3.png)

, where $$\tau = 0.007$$ is a temperature hyper-parameter in this work.

$$L_{NCE}$$ 는 임베딩 공간 상에서 positive sample들의 피처 임베딩이 서로 가깝게 만들고, 그렇지 않은 경우는 멀게 합니다.



그리고 본 논문은 classification, segmentation 두 태스크를 다루기 때문에 최종적인 loss는 아래와 같이 기술됩니다. 

![Final loss](/.gitbook/assets/32/eq4.png)

where $$L_{Task}$$ is loss of classificaion or segmentation.



**Details of $$L_{NCE}$$**

피처 임베딩 벡터는 인코더의 어느 레이어에서든 뽑아낼 수 있습니다. 그렇다면, 어느 레이어들을 뽑아서 loss를 흘리는 것이 모델을 더 general 하게 만들어줄까요?

어느 레이어가 가장 좋은 임베딩 벡터를 만들어주는지는 아무도 모르기 때문에, 몇 개의 레이어 ( $$\mathcal{G}$$ ) 를 선택하고 각 레이어에서 나온 임베딩 벡터로 NCE loss를 측정한 뒤 평균을 내어 가장 좋은 점수를 내는 레이어 셋을 찾아볼 수 있을 겁니다. 

식으로 나타내면 아래와 같은데, Eq.3에서 바뀐 부분은 $$\sum_{l\in\mathcal{G}}$$ 밖에 없다는 걸 알 수 있습니다.

![Equation 5](/.gitbook/assets/32/eq5.png)

Ablation study에서는 3, 4번째 레이어를 선택하는 것이 가장 좋은 generalization 성능을 냈다고 합니다. 



또한, classification 과 달리 segmentation task에서 NCE loss는 patch-wise로 이미지들을 잘라서 계산할 수 있습니다.

Segmentation 태스크의 정답 이미지에는 픽셀 별로 해당 픽셀이 어느 클래스인지를 나타냅니다. 그럼 이 좋은 정보를 버릴 수 없겠죠? 따라서, 우리는 피처 맵의 부분들로 NCE loss를 구할 수 있습니다. 실제 실험에서는 인풋 이미지 $$x$$ 를 8x8로 잘라서 총 64개의 local patches ($$N_l = 8*8 = 64$$) 를 만들어 segmentation 태스크를 학습시켰습니다. 

![Patch-wise NCE loss](/.gitbook/assets/32/eq6.png)



예를 들어,  $$i=3, N_l = 2*2 = 4$$ 라면   $$L^{l,3}_{NCE}$$  를 계산하는 방식은 아래와 같이 그릴 수 있을 겁니다. 

![Example of patch-wise NCE loss](/.gitbook/assets/32/loss_sample.png)



### *Improve average pooling $$g\to$$* A-Pool:

지금까지 우리가 한 걸 생각해봅시다: $$f_e, f_{e,o}, h, \tilde{h}, \tau$$ 

![Representations for each samples](/.gitbook/assets/32/eq2.png)

그럼 이제 마지막으로 $$g$$ , pooling layer가 어떻게 동작하는지 알아볼까요?



$$g$$ 는 $$f_e , f_{e,o}$$ 에서 나온 피처 맵들을 풀링하여 최종 피처 임베딩 벡터를 만듭니다. 만약 $$g$$ 를 global average pooling function으로 둔다면, 모든 피처 벡터들을 같은 weight로 더하게 됩니다. 

하지만, synthetic image의 경우 배경이 없고 물체 하나만 있는 경우가 많기 때문에, 이렇게 average pooling을 하게 되면 의미 없는 배경 벡터(예를 들어 흰 배경)까지 같이 더하게 되어, 오히려 물체의 특성을 희석시키게 됩니다. 

이런 상황을 피하기 위해, 본 논문은 인코더에서 나온 피처 맵을 attention score ($$a_{i,j}$$) 를 기준으로 풀링하는 것을 제안합니다. 



![Attentional pooling layer](/.gitbook/assets/32/a_pool.jpeg)

이때 $$a_{i,j}$$ 는 여러 개의 피처 벡터($$v_{:,i,j}$$) 와 피처 벡터의 평균($$\bar{v}$$) 사이의 유사도로 계산되는데요, 각각은 아래와 같이 계산됩니다.

* global average pooled vector: $$\bar{v} = g(v) = \frac{1}{hw} [\sum_{i,j} v_{1,i,j}, ... , \sum_{i,j} v_{C,i,j}] , i \in [1, h] , j \in [1, w]$$

* 픽셀 위치 (i,j)에서의 attention score: $$a_{i,j} = \frac{<v_{:,i,j}, \bar{v}>}{\sum_{i', j'} <v_{:,i',j'}, \bar{v}>} (i' \in [1,h], j'\in[1,w])$$

이 둘을 가지고 우리는 attentional pooling layer (줄여서 A-pool) 을 이렇게 정의할 수 있습니다. 

* $$\hat{v} = g_a(v) = [\sum_{i,j} v_{1,i,j} \cdot a_{i,j} , ... , \sum_{i,j} v_{C,i,j} \cdot a_{i,j}]$$ .



Attention score로 가중치를 두어 풀링을 하게 되면, 피쳐 벡터가 이미지에서 중요한 부분의 정보를 더 많이 가지고 있으리라 기대할 수 있습니다. 

여기서 주의할 것은 attention score가 $$f_e$$ 에서만 계산된다는 점입니다. $$f_{e,o}$$ 의 경우, $$f_e$$ 에서 $$g$$ 를 계산하고 그 값을 복사합니다. 





### Review the overall process

한번 전체 과정을 요약해 볼까요?

우리가 anchor로 고양이 이미지를 가지고 있고, negative sample로 강아지와 나무가 있으며, 태스크는 classification이라고 해봅시다. 

![Example of overall pipeline](/.gitbook/assets/32/overall_process.png)

1. 인풋 이미지는 RandAugment를 통해 랜덤하게 조작됩니다.
2. $$f_{e,o}$$ 는 인풋 이미지로 개, 나무, 고양이를 받고, $$f_{e}$$ 는 고양이 이미지를 인풋으로 받습니다.
3. 각 인코더를 통해 나온 피처 맵이 attentional pooling 레이어까지 통과하게 되면, 우리는 $$z^{l,+}_{cat}, z^{l,-}_{dog}, z^{l,-}_{tree}, z^{l,a}_{cat}$$ 를 얻게 됩니다.
4. $$f_e$$ 를 두 loss로 학습시킵니다.
   1. $$L_{NCE}$$ : (1) $$z^{l,+}_{cat}\cdot z^{l,a}_{cat}$$ 의 유사도를 최대화하고, (2) $$z^{l,a}_{cat}\cdot z^{l,-}_{dog},  z^{l,a}_{cat} \cdot z^{l,-}_{tree}$$ 의 유사도를 최소화합니다. 이 loss의 gradient는 주황색 형광펜으로 표시해두었습니다.
   2. $$L_{CE} (=L_{task})$$ : classification 태스크의 cross-entropy loss를 최소화합니다. 이 loss의 gradient는 하늘색으로 표시해두었습니다.





## 4. Experiment & Result



### 4.1. Classification task

**Experimental setup** *(편의상 영어로 작성하겠습니다)*

* Dataset : VisDA-17-classification dataset (http://ai.bu.edu/visda-2017/ )
  * ![VisDA-17 classification](/.gitbook/assets/32/visda_classification.png)
* Baselines: distillation strategies
  * Weight l2 distance (Kirkpatrick et al., 2017) [6]
  * Synaptic Intelligence (Zhenke et al., 2017) [7]
  * feature $$l_2$$ regularization [8]
  * KL divergence: ASG [3]
* Training setup
  * backbone: ImageNet pretrained ResNet-101
  * SGD optimizer, learning rate $$1 * 10^{-4}$$ , weight decay $$5 * 10^{-4}$$ , momentum $$0.9$$
  * Batch size $$32$$ , the model is trained for 30 epochs, $$\lambda$$ for $$L_{NCE} = 0.1$$
* Evaluation metric
  * generalization performance as hyperspherical enery (HSE) [9] *(details are in related work section)*
    * In experiments, HSE score on the feature embeddings is extracted by different methods.
  * classification accuracy



**Result**

![Resuls of classification task](/.gitbook/assets/32/table1.png)

테이블 1은 HSE 점수 (피처 공간의 분포를 나타내는 지표)와 generalization 성능 (accuracy) 간의 상관관계가 있다는 것을 보여줍니다. $$l_2$$ distance model을 제외하고, accuracy는 HSE 점수가 줄어듦에 따라 증가합니다. 또한 이 논문이 제시한 모델인 CSG는 가장 낮은 HSE 점수와 가장 높은 accuracy 점수를 보여줍니다.

이 실험 결과로 본 논문의 첫 가설 ('피처를 골고루 분포하는 모델일수록 더 좋은 generalization 성능을 보여줄 것이다.') 을 확인할 수 있습니다. 개인적으로 실험 결과의 일관성이 본 논문의 퀄리티를 높이고 가설의 설득력을 높여주었다고 생각합니다. 





### 4.2. Segmentation task

**Experimental setup**  *(편의상 영어로 작성하겠습니다)*

* Dataset
  * synthetic dataset: GTA5 (https://download.visinf.tu-darmstadt.de/data/from_games/)
  * Real dataset: Cityscapes (https://www.cityscapes-dataset.com/) 
  * ![GTA5 and Cityscapes dataset](/.gitbook/assets/32/dataset_seg.png)
* Baselines
  * IBN-Net : improves domain generalization by carefully mix the instance and batch normalization in the backbone. [10]
  * Yue et al. [2] *(details are on related work section)*
  * ASG [3] *(details are on related work section)*
* Training setup
  * backbone: DeepLabv2 with both ResNet-50 and ResNet-101, pretrained on ImageNet.
  * SGD optimizer, learning rate $$1 * 10^{-3}$$ , weight decay $$5 * 10^{-4}$$ , momentum $0.9$
  * Batch size: 6
  * Crop the images into patches of 512x512 and train the model with multi-scale augmentation (0.75~1.25) and horizontal flipping
  * the model is trained for 50 epochs, and $ $\lambda$$ for$$L_{NCE} = 75.$$
* Evaluation metric
  * mIoU: mean IoU across semantic classes (e.g., car, tree, road, etc.)
    * ![IoU](/.gitbook/assets/32/iou.png)
    * ![Examples of IoU](/.gitbook/assets/32/iou_ex.png)
    * *images from [ref](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)* 



**Result**

*Segmentation performance comparison*

![Table 5](/.gitbook/assets/32/table5.png)

* 각 모델을 적용하지 않았을 때와 적용했을 때의 성능 차이를 보면, 본 논문이 제안한 CSG가 가장 큰 차이로 모델의 성능을 향상시킨 것을 알 수 있습니다. 
* 한편, Yue et al. 은 CSG 보다 좋은 성능을 보여주었습니다. 하지만, 이 모델은 학습 중간에 ImageNet의 이미지를 사용합니다 (related work section 참조). 이런 점을 고려해 보면, CSG가 모델을 학습할 때 어떠한 real-world 이미지를 참조하지 않고도 좋은 성능을 낸 것으로 보 수 있습니다.



*Feature diversity after CSG*

![Comparison with and without CSG](/.gitbook/assets/32/fig6.png)

* Idea 섹션에서 조작한 것과 유사하게, GTA 5 학습 데이터 셋의 일부를 샘플링 합니다. Cityscapes 학습 데이터와의 사이즈를 맞추기 위해서입니다.
* Fig.6는 Fig.2와 유사하게 real image로 학습된 모델이 더 다양한 피처 공간을 가지고, synthetic 이미지는 몰려있는 것이 보입니다. 하지만, Fig.2와 비교해 보면 synthetic 데이터로 학습한 모델의 피처 공간이 이전보다 고르게 분포해있고 낮은 $$E_s$$ 점수를 가집니다.
* Fig.6 는 segmentation 태스크에서의 성능 개선이 이전보다 고르게 분포된 피처 공간에 기반한다고 볼 수 있습니다. 비록 Fig.2가 classification 태스크에서 만들어진 그래프이지만, synthetic image로 학습된 피처 공간이 덜 편향된 걸 알 수 있죠.
* 이 또한 본 논문의 첫 가설을 증명합니다. '피처를 골고루 분포하는 모델일수록 더 좋은 generalization 성능을 보여줄 것이다.'
* 한계
  * Fig.2 와 Fig.6는 각각 서로 다른 classification, segmentation 태스크에서 학습하고 시각화를 진행하였습니다. 만약 두 피겨 모두 같은 태스크에서 학습된 모델의 결과였다면 더 설득력이 있을 것 같습니다.





## 5. Conclusion

**강점**

* ImageNet pratrained model을 사용하더라도 real image의 정보를 synthetic image에 맞춰 transfer 하게 되면, synthetic image로 학습된 피처 공간의 경우 한쪽으로 몰리며 다양성이 줄어듦을 보여주었습니다. (Fig.2에서 real image로 학습한 경우와 상당히 대조된 결과를 보여줍니다.)
* 위 결과는 학습된 representation의 다양성이 syn2real generalization 문제에서 중요한 역할을 수행함을 보여줍니다. 본 논문은 이를 inductive bias, 즉 연구의 중심이 되는 가설로 설정하였습니다.
* 실험 결과들은 위 가설이 어느 정도 맞고 피처의 분포를 고르게 하는 것이 generalization performance를 향상시키며, 또한 어떠한 디테일한 학습 세팅 없이도 이전의 state-of-the-arts 모델의 성능을 넘어서는 것을 보여주었습니다. 

**약점**

* 이 연구는 syn2real 세팅에만 제한되어 있습니다. 본 연구를 더 대중적으로 만들기 위해선 일반적인 domain generalization problem 세팅에서의 실험 결과를 포함하면 좋았을 듯합니다.. 



### Take home message \(오늘의 교훈\)

* 통계적 관찰과 이를 시각화한 것은 저자들의 가설을 증명하는 데 중요한 역할을 하였습니다. 
* hyper-parameter에 대한 세밀한 튜닝 없이도 본 논문은 syn2real 태스크에서 SOTA 성능을 보여주었습니다.
* 우리가 어떠한 문제에 접근할 때에도, 데이터의 분포와 통계적 정보를 더 자세히 보고 분석해 봅시다. 문제는 생각보다 더 간단하고 멋있게 해결될지도 모릅니다!





## Author / Reviewer information

### Author

**양소영 \(Soyoung Yang\)** 

* KAIST AI
* My research area is widely on computer vision and NLP, also HCI.  
* [Mail](sy_yang@kaist.ac.kr) | [GitHub](https://github.com/dudrrm) | [Google Scholar](https://scholar.google.co.kr/citations?user=5Mw3sVAAAAAJ&hl=ko)

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. In Search of Lost Domain Generalization / Gulrijani and Lopez-Paz / ICLR 2021
   Generalizing to Unseen Domains: A Survey on Domain Generalization / Wang et al. / IJCAI 2021
2. Domain Randomization and Pyramid Consistency: Simulation-to-Real Generalization without Accessing Target Domain Data / Yue et al. / ICCV 2019
3. Automated Synthetic-to-Real Generalization / Chen et al. / ICML 2020
4. Representation Learning with Contrastive Predictive Coding / Oord et al. / arXiv preprint 2018
5. Improved Baselines with Momentum Contrastive Learning / Chen et al. / arXiv preprint 2020
6. Overcoming catastrophic forgetting in neural networks / Kirkpatrick et al. / Proceeding of national Academy of Sciences 2017
7. Continual learning through synaptic intelligence / Zenke et al. / ICML 2017
8. ROAD: Reality Oriented Adaptation for Semantic Segmentation of Urban Scenes / Chen et al. / CVPR 2018
9. Learning towards Minimum Hyperspherical Energy / Liu and Lin et al. / NeurIPS 2018
10. Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net / Pan et al. / ECCV 2018
11. Korean blog describing contrastive learning:  https://nuguziii.github.io/survey/S-006/
