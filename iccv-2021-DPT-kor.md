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
* **feature resolution** 과 **feature granularity** 문제를 해결하기 위해 
* convolutional network를 backbone으로 사용하는 기존 dense prediction model과는 다르게 vision transformer를 encoder로 convolutional network를 decoder로 사용하였다.
* 
* 대부분의 convolutional network를 backbone으로 사용하고 있는 기존 model들은 downsampling process로 인해 dense prediction task에서 deeper stages로 갈수록 **feature resolution** 과 **feature granularity** 를 잃어버리는 단점을 가지게 된다.
{% hint style="info" %}
If you are writing **Author's note**, please share your know-how \(e.g., implementation details\)
{% endhint %}

The proposed method of the paper will be depicted in this section.

Please note that you can attach image files \(see Figure 1\).  
When you upload image files, please read [How to contribute?](../../how-to-contribute.md#image-file-upload) section.

![Figure 1: You can freely upload images in the manuscript.](../../.gitbook/assets/how-to-contribute/cat-example.jpg)

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

Please provide one-line \(or 2~3 lines\) message, which we can learn from this paper.

> All men are mortal.
>
> Socrates is a man.
>
> Therefore, Socrates is mortal.

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


