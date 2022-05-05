---
description: Rene et al. / Vision Transformers for Dense Prediction / ICCV 2021
---

# Vision Transformers for Dense Prediction \[Kor\]

##  1. Problem definition

* 대부분의 dense prediction에서는 encoder와 decoder로 구성된 networks 구조를 채택하고 있으며 기존 연구는 encoder는 convolutional networks를 이용한채 decodoer의 구조와 aggregation strategy에 집중되고 있다.
* 하지만, encoder(or backbone)에서 잃은 정보를 decoder에서 되찾기 힘드므로 encoder(or backbone)의 구조가 전체 model의 성능에 매우 많은 영향을 주므로 본 논문에서는 encoder(or backbone)의 구조에 집중하였다.
* 대부분의 convolutional network를 backbone으로 사용하고 있는 기존 model들은 downsampling process로 인해 dense prediction task에서 deeper stages로 갈수록 **feature resolution** 과 **feature granularity** 를 잃어버리는 단점을 가지게 된다.
* 이러한 문제점을 해결하기 위해 본 논문에서는 vision transformer를 encoder의 기본 block으로 사용하는 **dense predition transformer(DPT)** 를 제안하였다.


## 2. Motivation

### Related work
#### Fully-convolutional networks
* semantic segmentation, keypoint detection과 같은 pixel level의 desnse predeiction에 fully-convolutional networks [1, 2]를 기반으로 다양한 모델이 제안되었다.
* 하지만, 기존 제안된 모델의 경우 convolution과 subsampling block을 이용하므로써 dense prediction에서 convoulional network의 문제점인 feature resolution과 feature granularity를 해결하지 못하였다.
#### Attention-based model
* NLP분야에서 제안된 transformer mechanism[3]를 이미지 분석 분야에 적용하는 연구가 활발히 이루어지고 있다[4-5].
* 하지만, NLP분야의 transformer와 마찬가지로 vision transforemr(의 성능을 유지하기 위해서는 충분히 많은 양의 training data가 필요하다.

### Idea
* convolutional network를 backbone으로 사용하는 기존 dense prediction model과는 다르게 vision transformer를 encoder로 convolutional network를 decoder로 사용하였다.
* 구체적으로, **feature resolution** 과 **feature granularity** 문제를 해결하기 위해 아래와 같은 특징을 갖는 ViT를 적용하였다. 
* 초기 image embedding후에 downsampling을 수행하지 않은 Vit를 적용함으로써 모든 processing  stage에서 변합없는 dimensionality의 representional을 유지할 수 있으며 매 stage마다 global receptive field를 가질 수 있다.


## 3. Method
* 저자는 dense prediction task를 위해 아래그램과 같은 구조를 제안하였다.
* 전체적인 구조(또는 흐름)는 1) input image를 token으로 변환(그림에서는 주황색으로 표현), 2) image embedding을 위치 embiding과 patch-independent readout token(그림에서는 빨강색으로 표현)으로 augmention, 3) token들에 multiple transformer stage  적용, 4) 각 단계들의 transformer output인 token들을 image 표현과 같에 재조합(reassembling), 5) 세분화된 예측(fine-grained prediction) 생성을 위해  전단계 represention을 융합 및 upsampling하는 fusion modul 적용함.
* Reassemble block에서는 token들을 input image의 $1/s$ spatial resolution을 갖는 feature map으로 assembling한다.
* Fusion block에서는 residual convoution unit을 사용하여 feature들을 결함하고 feature map을 upsampling한다.

![Figure 1](/.gitbook/assets/2022spring/21/figure1.jpg)


## 4. Experiment & Result
* 본 논문에서는 제안한 DPT의 성능을 비슷한 capacity를 같는 convolutional network와 비교하기 위하여 2가지 dese prediction task에(monocular depth estimation, semantic segmentation) 대한 실험을 수행하였다.

### Experimental setup for Monocular Depth Estimation
* Dataset: MIX 5 (in MiDaS [6]) and MIX 6 (extend MIX 5 with five additional datasets, contains 1.4 million images)
* Baselines: follows the prococol of Ranftle et al. [6]
* Training setup: multi-objective optimization together with Adam, backbone에는 1e-5 decoder weight를 위해서는 1e-4의 learing rate, encoder는 ImgaeNet-pretrained weitght decoder는 random하고 초기화, output header는 3개의 convolutional layer이용
### Result for Monocular Depth Estimation
* 아래 표는 training에 사용되지 않은 6개의 dataset에 적용한 zero-shot transfer의 결과를 보여주며 모든 error metric에서 제안하는 DPT가 다른 최신의 모델들보다 좋은 결과를 얻었다.
![Figure 2](/.gitbook/assets/2022spring/21/figure2.jpg)

### Experimental setup for Semantic Segmentation
* Dataset: ADE20K semantic segmentation dataset[7]을 이용하였음
* Baselines: follows the prococol of Zhang et al. [8]
* Training setup: SGD with momentum 0.9, decay factor 0.9의 polynomial learning rate scheduler 이용, fusion laryey에 batch nomalization 이용, 0.002 learing rate이용
* Evaluation Metric: pixel accuracy(pixAcc), mean Intersection over Union(mIoU)
### Result for Semantic Segmentation
* 아래 표는 semantic segmentation task시 pixAcc와 mIoU의 결과값을 보여주며 DPT-Hybird의 경우 다른 fully-convoultional 모델들 보다 더 좋은 결과를 얻을 수 있었다.

![Figure 3](/.gitbook/assets/2022spring/21/figure3.jpg)


## 5. Conclusion
* 본 논문에서는 기존 dense prediction task에서 사용되는 모델(convolutional network를 backbone으로 사용)의 단점인 **feature resolution** 과 **feature granularity** 를 해결하기 위해 vision transformer를 encoder 기본 block으로 사용하는 DPT 제안하였다.
* 제안한 DPT 모델은 monocular depth estimation과 segmentation task에서 기존의 fully-convolutional 구조의 dense predection 모델보다 더 좋은 성능을 얻을 수 있었다.

### Take home message \(오늘의 교훈\)

> 기존 dense prediction task에서의 주된 decoder구조 연구에서 encoder 파트에 ViT를 적용 하여 기존 모델보다 더 좋은 결과를 얻은점에서 흥미로운 논문이라 생각된다.
>


## Author / Reviewer information

{% hint style="warning" %}
You don't need to provide the reviewer information at the draft submission stage.
{% endhint %}

### Author

**Korean Name \(English name\)** 

* Affiliation \(KAIST AI / NAVER\)
* \(optional\) 1~2 line self-introduction
* Contact information \(Personal webpage, GitHub, LinkedIn, ...\)
* **...**

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. Pierre Sermanet, et al. OverFeat: Integrated recognition, localization and detection using convolutional netwrks. In ICLR, 2014.
2. Jonathan Long, et al. Fully Convolutional Networks for Semantic Segmentation, In CVPR, 2015.
3. Ashish Vaswani, et al. Attention is all you need, In NeurIps, 2017.
4. Huiyu Wang, et al. Axial-DeepLab: Stand-alone axial-attention for panoptic segmentation, In ECCV, 2020.
5. Hengshuang Zhao, et al. Exploring self-attention for image recognition, In CVPR, 2020.
6. Rene Ranftl, et al. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer, in TPAMI, 2020.
7. Bolei Zhou, et al. Scene parsing through ADE20K dataset, In CVPR, 2017. 
8. Hang Zhang, et al. ResNest: Split-attention networks, In CVPR, 2020.

