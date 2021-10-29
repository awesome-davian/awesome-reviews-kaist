---
description: Yulun Zhang et al. / Image Super-Resolution Using Very Deep Residual Channel Attention Networks / ECCV 2018
---

# Image Super Resolution via RCAN \[Kor\]


## \(Start your manuscript from here\)
한국어로 쓰인 리뷰를 읽으려면 **여기**를 누르세요.

**English version** of this article is available.

##  1. Problem definition

기존의 CNN 기반 초해상화 (Super-Resolution, SR) 기법은 i) 층이 깊어질수록 학습이 어려우며, ii) 저해상도 (Low Resolution, LR) 이미지에 포함된 저주파(low-frequency) 정보가 모든 채널에서 동등하게 다루어짐으로써 각 feature map의 대표성이 약화된다는 한계점을 가지고 있다. i)과 ii)의 한계점을 극복하기 위해, 해당 논문에서는 Deep-RCAN (Residual Channel Attention Networks)을 제안한다. 

## 2. Motivation

### **Related work**

본 논문의 baseline인 deep-CNN과 attention 기법과 관련된 paper들은 다음과 같다.

#### **1. CNN 기반 SR**

* **[SRCNN & FSRCNN]**: CNN을 SR에 적용한 최초의 기법으로서, 3층의 CNN을 구성함으로써 기존의 Non-CNN 기반 SR 기법들에 비해 크게 성능을 향상시켰음. FSRCNN은 SRCNN의 네트워크 구조를 간소화하여 추론과 학습 속도를 증대시킴.
* **[VDSR & DRCN]**: SRCNN보다 층을 더 깊게 적층하여 (20층), 성능을 크게 향상시킴.
* **[SRResNet & SRGAN]**: SRResNet은 SR에 ResNet을 최초로 도입하였음. 또한 SRGAN에서는 SRResNet에 GAN을 도입함으로써 블러현상을 완화시킴으로써 사실에 가까운(photo-realistic) SR을 구현하였음. 하지만, 의도하지 않은 인공적인(artifact) 객체를 생성하는 경우가 발생함.
* **[EDSR & MDSR]**: 기존의 ResNet에서 불필요한 모듈을 제거하여, 속도를 크게 증가시킴. 하지만, 이미지 처리에서 관건인 깊은 층을 구현하지 못하며, 모든 channel에서 low-frequency 정보를 동일하게 다루어 불필요한 계산이 포함되고 다양한 feature를 나타내지 못한다는 한계를 지님.

### Idea

After you introduce related work, please illustrate the main idea of the paper. It would be great if you describe the idea by comparing or analyzing the drawbacks of the previous work.

## 3. Method

{% hint style="info" %}
If you are writing **Author's note**, please share your know-how \(e.g., implementation details\)
{% endhint %}

The proposed method of the paper will be depicted in this section.

Please note that you can attach image files \(see Figure 1\).  
When you upload image files, please read [How to contribute?](../../how-to-contribute.md#image-file-upload) section.

![Figure 1: You can freely upload images in the manuscript.](../../.gitbook/assets/cat-example.jpg)

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

