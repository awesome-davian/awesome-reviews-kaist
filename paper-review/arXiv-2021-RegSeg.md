---
description: (Description) Roland Gao / Rethink Dilated Convolution for Real-time Semantic Segmentation / arXiv 2021
---

# RegSeg \[Kor\]


##  1. Problem definition
본 논문에서는 real time scene segmentation에서 사용되는 ImageNet backbone으로부터 비롯되는 문제를 해결하고자 합니다.  
기존 real time scene segmentation 논문들에서 사용한 ImageNet backbone은 끝 부분의 합성곱 레이어는 지나치게 많은 채널수를 초래합니다. 예를 들어, ResNet18은 512개, ResNet50은 2048개까지 생성됩니다. 이는 실시간 환경에서 많은 연산량을 부담시키는 문제가 있습니다.
또한 ImageNet 모델들이 입력받는 이미지의 크기는 224 x 244인 반면, semantic segmentation의 데이터셋은 1024 x 2048으로 훨씬 큽니다. 이는 ImageNet 모델들의 field-of-view가 큰 이미지를 인코딩하는데 부족함을 의미합니다.  
RegSeg는 정확도를 저해하지 않으면서 연산양을 줄이고 충분한 field-of-view를 확보할 수 있는 구조를 제한합니다.

## 2. Motivation

### Related work
Segmentation 분야에서 정확도와 연산 속도 모두 효과적으로 향상시키기 위한 기존의 연구들에 대해 간략하게 다뤄보겠습니다.
* Semantic segmentation
    * Fully Convolutional Networks  
    Classification 모델을 segmentation에 적용하기 위해 fc-layer를 모두 Conv-layer로 교체하였습니다.
    * DeepLabv3  
    다양한 dilation rates를 적용한 dilated conv를 ImageNet 모델에 추가하여 receptive field를 크게 하였습니다.
    * PSPNet  
    Pooling rate를 달리한 layer를 여러 개 병렬로 추가한 Pyramid Pooling Moudle을 통해 Global context information을 학습할 수 있게 하였습니다.
    * Deeplabv3+  
    Deeplabv3에 디코더와 1 x 1 convlution을 추가하여 학습을 안정시켰습니다.
* Real-time semantic segmentation
    * BiseNetV2  
    Spatial Path와 Context Path 두 개의 가지를 만든 후 합쳐 사전 학습된 ImageNet 모델 없이 좋은 성능을 보여주었습니다.
    * STDC  
    BiseNet의 Spatial Path를 없애고 하나의 Path만을 거치게 하여 더 빠르게 작동하게 하였습니다.
    * DDRNet-23  
    두 분기 사이에 상호 융합을 추가한 Deep Aggregation Pyramid Pooling Module(DAPPM)을 backbone 끝에 추가하여 Cityscapes 데이터셋에서 SOTA 성능을 보이고 있습니다.
* Desinging Network design Spaces  
네트워크 디자인에서 선택지가 늘어나면서 manual network design은 어려워졌습니다. 좋은 네트워크를 많이 찾을 수는 있었지만 그 원리를 찾은 것은 아니었기 때문에 수많은 실험과 시뮬레이션을 통해 블록 타입의 RegNetY를 새로운 네트워크 디자인 패러다임으로 제안하였습니다.

### Idea
기존의 Semantic segmentation 연구들이 ImageNet 모델을 대체하기 위해 real-time semantic segmentation 연구들에선 연산량이 방대하게 증가하였습니다. DDRNet-23의 경우 20.0M개의 파라미터가 사용되었습니다. 본 논문에서는 연산량을 줄이면서 동시에 receptive field를 늘리기 위해 RegNet의 블록을 참고하여 dilated conv가 적용된 블록 구조를 제안하고, 이를 반복하여 쌓았습니다.

## 3. Method

### Dilated block
저자는 RegNet의 Y 블록에서 3 x 3 conv를 하는 단계를 두 개의 갈래로 나눈 dilated conv로 대체하였습니다. 이를 Dilated Block(D Block)으로 명명하였고 dilated rate를 바꿔가면서 총 18번 반복하였습니다. Y블록과 D블록의 차이는 다음과 같이 확인할 수 있습니다. dilated rate가 모두 1일 때는 D블록이 Y블록과 같습니다.

![figure 1](/.gitbook/assets/2022spring/1.png)

Stride가 2일 때의 D블록은 다음과 같습니다.

![figure 2](/.gitbook/assets/2022spring/2.png)

각 D블록에서의 dilated rate와 stride는 다음 표에서 확인할 수 있습니다. 각 dilated rate를 달리하면서 multi-scale featrues를 추출할 수 있었습니다.

![figure 3](/.gitbook/assets/2022spring/3.png)

이와 같이 D블록을 반복하여 구성된 backbone은 RegNet의 스타일과 유사하며 각 블록의 dilated rate는 실험을 통해 정해져습니다. 또한, dilation branch를 4개로 했을 때 2개보다 좋은 결과를 보여주지 못하여 2개로만 나뉘어졌습니다.

### Decoder
위의 backbone에서 소실된 local deatils을 복구하기 위해 디코더를 추가하였습니다. Backbone으로부터 1/4, 1/8, 그리고 1/16 크기의 featrue maps을 입력받아 1 x 1 conv와 upsampling을 거쳐 합쳐집니다. 디코더의 단순한 구조는 연산량을 크게 늘리지 않습니다.

![figure 4](/.gitbook/assets/2022spring/4.png)

## 4. Experiment & Result

### Experimental setup
본 논문에서는 Cityscapes, CamVid에서 DDRNet-23을 비롯한 state-of-the-art model들과 성능을 비교하는 실험을 진행했습니다. Cityscapes에 대한 Training setup은 다음과 같습니다.

* momentum 0.9의 SGD
* initial learning rate: 0.05
* weight decay: 0.0001
* ramdon scaling [400, 1600]
* random cropping 768 x 768
* 0.5%의 class uniform sampling
* batch size = 8, 1000 epochs

Camvid에서는 Citycapes pretrained model을 사용하였고 Cityscapes 실험 환경과의 차이는 다음과 같습니다.
* random horizontal flipping
* random scaling of [288, 1152]
* batch 12, 200 epochs
* classuniform sampling 사용하지 않음

### Result

#### Cityscapes
Cityscapes에서의 결과는 다음과 같습니다.

![figure 5](/.gitbook/assets/2022spring/5.png)

모델 간의 FPS는 직접 비교할 수 없지만, RegSeg는 추가적인 데이터 없는 SOTA 모델인 HardDNet보다 1.5%p 더 높고, 피어 리뷰 결과가 가장 우수한 SFNet을 0.5%p 능가합니다.  

![figure 6](/.gitbook/assets/2022spring/6.png)

Cityscapes test set에서 가장 우수한 정확도와 파라미터 사이의 균형을 유지하고 있습니다.

#### Ablation Studies
작은 dilation rates를 앞에서 사용하고 큰 dilateion rates를 뒤에서 사용하되 무작정 filed-of-view를 늘리는 것이 정확도 향상을 이끌어내지 않는 것을 알 수 있습니다.

![figure 7](/.gitbook/assets/2022spring/7.png)

## 5. Conclusion
* DDRNet-23의 정확도를 유지하면서 파라미터를 줄이지는 못하였지만 그래도 상당히 우수한 교환비를 통해 real-time-segmentation에서 좋은 성능을 보여주었습니다.
* Field-of-view를 늘리기 위한 dilated conv은 DeepLab부터 사용되었지만, 가지를 두 개로 줄이면서 파라미터 수를 줄이는데 효과적이었습니다.
* 상당히 많은 실험을 통해 효율적인 dilated rate와 구조를 찾는 기여가 있었습니다.

### Take home message

> Dilated conv branch는 최소화하면서 깊이 쌓는게 효율적이다.
>
> Field-of-view를 무작정 늘린다고 꼭 정확도가 향상되지는 않는다.

### Author

**이명석 \(MyeongSeok Lee\)** 

* M.S Student in School of ETRI, UST (Advisor: [_Prof. ChiYoon Chung_](https://etriai.notion.site/))
* ims@etri.re.kr


## Reference & Additional materials

1. Gao, R. (2021). Rethink Dilated Convolution for Real-time Semantic Segmentation. arXiv preprint arXiv:2111.09957.
2. Radosavovic, I., Kosaraju, R. P., Girshick, R., He, K., & Dollár, P. (2020). Designing network design spaces. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10428-10436).