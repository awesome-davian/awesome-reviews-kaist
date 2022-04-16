---
description: Vikram V. Ramaswamy / Fair Attribute Classification through Latent Space De-biasing / CVPR 2021 Oral
---

# Latent Space De-biasing \[Kor\]

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

{% hint style="info" %}
If you are writing manuscripts in both Korean and English, add one of these lines.

You need to add hyperlink to the manuscript written in the other language.
{% endhint %}

{% hint style="warning" %}
Remove this part if you are writing manuscript in a single language.
{% endhint %}

\(In English article\) ---&gt; 한국어로 쓰인 리뷰를 읽으려면 **여기**를 누르세요.

\(한국어 리뷰에서\) ---&gt; **English version** of this article is available.

##  1. Problem definition

지금까지 수많은 딥러닝 모델이 개발되면서 인공지능의 성능은 크게 향상되었다. 그러나 모델들 대부분은 데이터셋의 전반적인 예측 정확도에 초점을 두고 개발되었기 때문에, 모델이 데이터셋 내의 특정 집단에 대해 불리한 판단을 내릴 여지가 존재한다. 예를 들어, 서구권 국가에서 개발된 얼굴 인식 AI의 경우 아시아인의 얼굴을 백인의 얼굴보다 더 부정확하게 판별할 가능성이 높다. 우리는 이와 같은 현상을 가리켜 '인공지능의 공정성 문제'라 부른다. 아무리 인공지능의 성능이 좋아진다고 해도, 인공지능의 공정성 문제가 해결되지 않는다면 인공지능 모델은 장애인이나 노인과 같이 사회적으로 소외받는 집단에 대해 잘못된 판단을 쉽게 내릴 수 있을 것이고, 이는 심각한 사회 문제를 초래할 것이다. 그러므로 인공지능을 더욱 공정하게 만드는 것은 매우 중요한 일인데, 최근 인공지능 학계에서는 인공지능의 성능을 크게 희생하지 않으면서도 공정성을 향상시킬 수 있는 방법에 대해 활발하게 연구가 이루어지고 있다.

딥러닝 모델의 공정성을 향상시키는 방법은 다양한데, 논문의 저자는 적대적 생성 신경망(GAN)을 통한 데이터 증강(Data Augmentation)을 시도한다. 즉 GAN을 이용해 그럴듯한 이미지들을 생성한 뒤 이들의 잠재 공간(latent space)을 수정함으로써 특정 집단에 대한 편향성이 제거된 훈련 데이터셋을 만든다는 것이다. 지금까지 이와 비슷한 연구는 이전에도 있었으나, 알고리즘이 더욱 복잡해지고 연산량이 늘어난다는 단점이 있었다. 반면에 논문 저자는 단 하나의 GAN을 사용하는, 간단하고 효과적인 데이터 증강 방법을 제시한다.

## 2. Motivation

### Related work

(1) De-biasing methods

많은 경우에 딥러닝 모델의 불공정성은 훈련 데이터에 내재된 편향성에 의해 생겨난다. 이를 해결하기 위해 훈련데이터의 편향성을 줄이는 방법을 쓰기도 하고, 모델의 학습 과정을 보완하는 방법을 쓰기도 한다. 훈련 데이터의 편향성을 줄이는 방법으로는 취약 집단을 대상으로 오버샘플링을 적용하는 방법, 적대적 학습을 이용하는 방법 등이 있다. 모델의 학습 과정을 보완하는 방법으로는 모델의 손실함수(loss function)에 공정성과 관련된 규제(regularization) 항을 추가하는 방법 등이 있다. 이 논문에서는 공정성 향상을 위해 훈련데이터의 편향성을 줄이는 방법을 이용한다.

(2) Generating and perturbing images using GANs

적대적 생성 신경망(GAN)은 생성자와 판별자로 이루어진 신경망인데, 여기서 생성자의 학습 방식과 판별자의 학습 방식은 적대적인 관계에 있다. 즉 생성자는 자기가 거짓으로 만들어 낸 데이터를 판별자가 가짜로 인식하지 못하도록 학습하고, 판별자는 생성자가 자기를 속이지 못하도록 학습한다. 이와 같이 적대적인 학습을 시킴으로써 진짜처럼 보이는 가짜 데이터를 만들어 내는 신경망이 바로 적대적 생성 신경망이다. 그동안 적대적 생성 신경망은 많은 개선을 거쳤고, 이제는 실제 사람 얼굴과 구분이 어려운 이미지를 생성할 수 있을 정도가 되었다.  

### Idea

After you introduce related work, please illustrate the main idea of the paper. It would be great if you describe the idea by comparing or analyzing the drawbacks of the previous work.

## 3. Method

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

김대혁 \(Kim Daehyeok\) 

* KAIST EE, U-AIM Lab.
* Research Interest : Speech Recognition, Fairness
* Contact email : kimshine@kaist.ac.kr

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. Ramaswamy, Vikram V., Sunnie SY Kim, and Olga Russakovsky. "Fair attribute classification through latent space de-biasing." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
2. Official \(unofficial\) GitHub repository
3. Citation of related work
4. Other useful materials
5. ...

