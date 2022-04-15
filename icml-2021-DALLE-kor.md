---
description:  Ramech et al. / Zero-shot Text-to-Image Generation / IMCL 2021
---

# DALL-E: Zero-shot Text-to-Image Generation \[Kor\]

## Guideline

{% hint style="warning" %}
Remove this section when you submit the manuscript
{% endhint %}

Write the manuscript/draft by editing this file.

### Title & Description

Title of an article must follow this form: _Title of article \[language\]_

#### Example

* Standardized Max Logit \[Kor\]
* VITON-HD: High-Resolution Virtual Try-On \[Eng\]
* Image-to-Image Translation via GDWCT \[Kor\]
* Coloring with Words \[Eng\]
* ...

Description of an article must follow this form: _&lt;1st author&gt; / &lt;paper name&gt; / &lt;venue&gt;_

#### Example

* Jung et al. / Standardized Max Logit: A simple yet Effective Approach for Identifying Unexpected Road Obstacles in Urban-scene Segmentation / ICCV 2021 Oral
* Kim et al. / Deep Edge-Aware Interactive Colorization against Color-Bleeding Effects / ICCV 2021 Oral
* Choi et al. / RobustNet: Improving Domain Generalization in Urban-Scene Segmentation via Instance Selective Whitening / CVPR 2021 Oral
* ...

## \(Start your manuscript from here\)

##  1. Text-to-Image Generation

computer vision 분야에는 다양한 task 들이 존재한다. 널리 알려진 image classification, object detection, segmentation 뿐만 아니라 최근 활발하게 연구되고 있는 task 중 하나는 text-to-image generation 이다. image caption에 의해 설정한 조건에 맞는 image를 생성하는  task로, 이 논문에서는 단순히 해당 task를 수행하는 것이 아닌 “zero-shot”으로 고품질의 이미지를 생성했다는 것이 주목할 만한 포인트 이다.  

일반적인 text-to-image generation고정된 Dataset 에 대해 더 좋은 모델링을 할 수 있는 방법 ( 예: 복잡한 아키텍쳐, 손실함수, segmentation 마스크 등의 추가적인 정보) 을 찾는 것에 포커스를 맞춰왔습니다. 그러나, 이 논문은 전혀 다른 접근 방식을 택하고 있다.

인터넷에서 얻은 대규모의 text-image pair를 autoregressive transformer에 입력으로 넣어 모델을 학습 시킨다. 이렇게 충분히 학습된 모델은 zero-shot 방식으로 text-to-image generation task를 잘 수행한다는 것을 논문에서 보여주고 있다.

Please provide the problem definition in this section.

We recommend you to use the formal definition \(mathematical notations\).

## 2. Motivation

generative model이 발전함에 따라 text에 의해 설정된 조건에 따라 적절한 이미지를 생성하는 task에 대한 연구가 활발히 이루어졌다. 그러나 고정된 데이터에 대해 학습하는 것은 그 한계가 명확하다.  관련 데이터의 수가 많지 않을 뿐더러, 이렇게 학습된 모델의 경우 학습 과정에서 보지 못한 데이터의 경우 전혀 이해하지 못할 가능성이 높다(generalization이 어려움).

또한  최근 large-scale generative model의 성공과 text, image, audio 등 다양한 분야에서 제안된 autoregressive transformer의 성공으로 언어 모델인 GPT-3 와 같은 구조를 vision에도 적용해보려는 motivation을 기반으로 수행된 연구이다.

### Related work

####GPT-3


### Idea

DALL-E는 [openAI의 소개](https://openai.com/blog/dall-e/)에서도 언급하고 있듯이, 120억개의 파라미터와 2억 5천개의 이미지-텍스트 쌍으로 학습시킨 vision task를 위한 [GPT-3](https://arxiv.org/abs/2005.14165) 라고 할 수 있다. 

해당 논문에서 제안한 모델 DALL-E의 목표는 텍스트와 이미지 토큰을 하나의 stream을 입력으로, autoregressive transformer를 학습시키는 것이다. 즉, 텍스트와 이미지 전체에 대해 한 토큰 뒤에 다음 토큰이 올 likelihood를 최대화하는 방향으로 모델을 학습시킨 것이다. 이 때 이미지와 텍스트를 하나의 stream으로 입력함으로써 텍스트와 이미지는 동일한 latent space 상에 있는  embedding으로 학습된다. 

구체적인 method 에 대해서는 아래에서 좀 더 자세히 다루겠지만, 한가지 짚고 넘어가자면  ‘이미지 토큰’ 을 사용했다는 것을 들 수 있다.   이미지를 pixel 단위로 다루게 되면 고해상도의 이미지를 위해서는 엄청난 양의 메모리를 사용하게 된다. 뿐만 아니라, 우리가 실제로 이미지를 인식하는 구조(low-frequency)보다 이미지의 사소한 디테일(high-frequency)를 학습하게 되는 문제점이 발생한다([pixelCNN++](https://arxiv.org/abs/1701.05517)). 이 문제를 해결하기 위해 DALL-E는 ‘이미지 토큰’을 통해 이미지를 pixel 단위가 아니라  토큰 단위로 다루게 된다.

이를 통해 이미지를 해당 논문에서는 192배 압축하면서 visual quality는 유지할 수 있도록 하였다.

## 3. Method

###Training Steps
> The overall procedure can be viewed as maximizing the evidence lower bound (ELB) on the joint likelihood of the model distribution over image x, captions y, and the tokens z for the encoded RGB image. >



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

1. Citation of this paper
2. Official \(unofficial\) GitHub repository
3. Citation of related work
4. Other useful materials
5. ...

