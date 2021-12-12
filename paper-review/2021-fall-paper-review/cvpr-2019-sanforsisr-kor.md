---
description: Dai et al. / Second-Order Attention Network for Single Image Super-Resolution / CVPR 2019
---

# SAN for SISR \[Kor\]

## 1. Introduction

Single Image Super-Resolution(**SISR**) 분야에 Convolutional Neural Network(**CNN**)가 도입되며 큰 성능의 향상이 이루어졌다. 여기서 기존 CNN based SISR methods 는 wider/deeper architecture design에 집중하는데, 이는 intermediate layers 간 feature correlation을 무시하여 CNN의 representational power을 방해하는 결과를 낳았다. 이를 해결하기 위해, 이 논문에서는 Second-order Attention Network(**SAN**)를 제안한다.

Second-Order Channel Attention(**SOCA**) module은 first-order보다 더 나은 feature correlation 학습을 위한 메커니즘이다. 이는 discriminative representation 향상을 위해 second-order feature statics를 사용하였다. channel-wise features를 adaptively rescale하는 방법으로 이를 활용하였는데, 이는 네트워크가 '더 중요한 정보를 갖는 feature'에 집중하게 만들어 학습 능력을 향상시켰다는 것이다.

Non-locally Enhanced Residual Group(**NLRG**) structure은 Local-Source Residual Attention Group(**LSRAG**)를 포함하는 연산으로, long-distance spatial contextual information을 수집하는 non-local 연산이다. 추상적인 feature representation 학습을 위한 LSRAG로, Low-Resolution(LR) image에서 많은 정보를 수집하고 low frequency 정보를 통과시키는 방법을 활용하였다.



## 2. Related Work

### **CNN-based SR models**

최근 CNN-based methods는 nonlinear 표현의 강점 때문에 SR에 많이 이용되었다. 이는 SR을 이미지-이미지 간 문제로 생각하여 LR-HR 간 매핑으로 직접 러닝을 실행한다. 이러한 기존 방법들은 주로 deeper/wider 네트워크 설계를 중점으로 하였다.

### **Attention mechanism**

인간은 시각 정보를 adaptive하게 처리하며, 중요한 영역에 시각을 집중하는 경향을 갖고 있다. 이러한 원리를 CNN에 적용한 것이 Attention의 시작이다. 

SENet은 channel-wise relationship 활용을 통해 이미지 분류를 진행한다. SR 성능 향상을 위해 deep-CNN에 도입했으나, SENet은 first-order statistics만 활용한다. 즉, higher order statistics를 무시하기 때문에 네트워크의 discriminative ability가 저하된다는 단점을 갖고 있다.



## 3. Method

### Second-order Attention Network (SAN)

#### Network Framework

![1_networkframework](/.gitbook/assets/56/1_networkframework.png)



##### - Shallow feature extraction

단일 convolution layer만 사용하여 shallow feature을 추출하는 단계이다.

![2_Shallowfeatureextraction](/.gitbook/assets/56/2_Shallowfeatureextraction.png)



##### - Non-locally enhanced residual group (NLRG) based deep feature extraction

2개의 Region-level Non-local module(RL-NL) 사이에 Share-source Residual Group(SSRG)으로 구성된 단계이다. 여기서 SSRG는 여러(G)개의 Local-Source Residual Attention Groups(LSRAG)로  이루어져 있다. LSRAG는 2개의 Residual block 사이에 여러 Conv. layers + 1 ReLU layer 의 구성에 SOCA module이 들어있는 형태이다.

NLRG 내부의 module 및 layers를 그림으로 표현하면 아래와 같다.

![3_NLRG](/.gitbook/assets/56/3_NLRG.png)

SSRG: Share Source Skip Connection(SSC)을 활용하는 G * LSRAG modules로 구성되어 있다.

LSRAG: SSC를 활용하는 M * residual blocks로 구성되어 있다.

SOCA: inter-dependencies를 활용한다.

전체적 구성에서 보이듯이, residual blocks를 많이 사용한다는 것을 확인할 수 있다. 이는 더 깊은 CNN의 활용을 가능하게 하나, bottle-neck이 발생할 가능성이 존재한다. 그렇기에 LSRAG의 활용을 제안되었으나, LSRAG 만으로는 성능이 부족했기에 SSC를 추가로 활용하여 학습 촉진및 low-frequency 정보를 통과시키도록 하였다.

*g-th LSRAG(H_g):* ![4_gthLSRAG](/.gitbook/assets/56/4_gthLSRAG.png)

*g-th LSRAG, m-th residual block:* ![5_gthLSRAGmthresidualblock](/.gitbook/assets/56/5_gthLSRAGmthresidualblock.png)

*Local source skip connection:* ![6_Localsourceskipconnection](/.gitbook/assets/56/6_Localsourceskipconnection.png)

RL-NL: non-local NN은 high-level task에서 전체 image의 long-range dependency를 계산하는 방법 제시한다. 그러나 global-level non-local operation은 과도한 연산량 등의 문제점이 있으므로, 이를 global-level이 아닌 region-level로 진행하는 것이 RL-NL이다.

위와 같이 구성된 NLRG는 매우 깊은 depth 와 receptive field를 가지고 있는데, 이를 다음과 같은 식으로 요약할 수 있다.

![7_NLRGeq](/.gitbook/assets/56/7_NLRGeq.png)

##### - Up-scale module

위의 과정으로부터 얻은 정보를 바탕으로 Up-scale을 진행하는 단계이다. 여러가지 선택지가 존재하므로, complexity와 performance 간의 trade-off를 고려하여 선택해야한다.

![8_Upscalemodule](/.gitbook/assets/56/8_Upscalemodule.png)



본 논문에서는 최근 CNN-based SR에서 자주 사용되는 방법인 *pixel shuffle method*를 사용하였다.



##### - Reconstruction

단일 convolution layer을 이용해 feature을 SR image로 mapping하는 단계이다.

![9_Reconstruction](/.gitbook/assets/56/9_Reconstruction.png)

이 때, Loss function(*L1 loss*)는 다음과 같다.

![10_lossfunc](/.gitbook/assets/56/10_lossfunc.png)



## 4.  Experiment & Result

### Experiment

#### Setup

- SSRG 내부 LSRAG 개수 G = 20
- LSRAG 내부 residual block 개수 M = 10 : SOCA module (reduction ratio 16인 1x1 convolution filter) + convolution filters (3x3 64 channel filter)
- Up-scale module: pixel shuffle method
- Training set: DIV2K



### Result

#### - Zoom visual from Urban 100

![11_zoomvisualurban100](/.gitbook/assets/56/11_zoomvisualurban100.png)

본 논문의 모델인 (h)가 이 중 (a) HR 이미지와 가장 유사하다는 것에서, 다른 SR 모델 (b)~(g)와 비교하여 visual quality 및 image detail이 좋다는 것이 확인 가능하다.



#### - Urban 100

![12_urban100](/.gitbook/assets/56/12_urban100.png)

위 figure은 Visual comparision for 4x SR with BI model on Urban100 dataset 이다. 각 경우의 첫번째 사진이 HR(original)이고, 10번째 사진이 본 논문의 SAN을 적용한 결과, 그리고 9번째 사진이 기존 연구 중 가장 SAN과 유사한 원리를 갖는 RCAN method 이다.

위 figure의 두 케이스를 통해, SAN은 기존 연구와 비교하여 유의미한 visual quality의 상승을 가져왔음이 확인 가능하다.



## 5. Conclusion

본 논문에서는 보다 정확한 SR을 위해 SAN을 제안하였다. 여기서 NLRG structure을 활용한 SAN은 네트워크에 long-distance dependencies & structural information를 캡처하였는데, 이 NLRG에 추가로 SSC를 활용하여 low-frequency 정보를 통과시켜 러닝 효과를 상승시켰다. 

추가로, 논문에서는 보다 discriminative representations를 위해 global covariance pooling을 통해 feature interdependencies를 학습하기 위해 SOCA module을 제안한다.

이를 BI & BD degradation models에 실험해본 결과, SAN은 SR에 대해 quantative/visual 적으로 좋은 결과를 냈음을 확인 가능했다.

## Take home message (오늘의 교훈)

Attention mechanism 에 대한 이해 및 배경 지식을 늘릴 수 있는 좋은 기회였다.
SR 쪽에 대한 연구를 진행하고 있었기에 적용할 수 있는 mechanism이 다양해진 것 같다.
연관 분야에 대한 공부가 현재 연구에 크게 도움이 될 수 있다는 사실을 느꼈다.


## Author / Reviewer information

### Author

**양승훈 \(Seunghoon Yang\)** 

* KAIST Mechanical Engineering
* https://github.com/SeunghoonYang
* shyang9512@kaist.ac.kr

### Reviewer

1. Korean name (English name): Affiliation / Contact information

2. Korean name (English name): Affiliation / Contact information

3. ...

   

## Reference & Additional materials

1. T. Dai, J. Cai, Y. Zhang, S. Xia and L. Zhang, "Second-Order Attention Network for Single Image Super-Resolution," 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 11057-11066, doi: 10.1109/CVPR.2019.01132.
2. https://github.com/daitao/SAN.git
3. Ding Liu, Bihan Wen, Yuchen Fan, Chen Change Loy, and Thomas S Huang. Non-local recurrent network for image restoration. In NIPS, 2018.
4.  Yulun Zhang, Yapeng Tian, Yu Kong, Bineng Zhong, and Yun Fu. Residual dense network for image super-resolution. In CVPR, 2018.
5. Zhang, Yulun, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong and Yun Raymond Fu. “Image Super-Resolution Using Very Deep Residual Channel Attention Networks.” *ECCV* (2018).
