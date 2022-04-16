---
description: Wu, Weibin, et al. / Improving the Transferability of Adversarial Samples with Adversarial Transformations / CVPR2021
---

# Improving the Transferability of Adversarial Samples with Adversarial Transformations \[Kor\]

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

### 적대적 예제 (Adversarial Samples)
적대적 예제는 사람의 눈으로는 인식할 수 없는 미세한 잡음\(perturbation\)을 의도적으로 원래의 입력에 더해 생성한 예제이다.
이렇게 생성된 예제는 신경망을 높은 확률로 오분류하도록 한다.

구체적으로 아래 그림과 같이 원본 이미지 $x$에 미세한 잡은 $\delta$를 더해 적대적 예제 $x_{adv}$를 생성할 수 있다.  
$ x_{adv} = x + \delta $ 
    
![adv_example](https://github.com/ming1st/awesome-reviews-kaist/blob/master/.gitbook/assets/2022spring/17/adv_sample.png)

### 적대적 공격 (Adversarial Attacks)
적대적 공격은 의도적으로 생성된 적대적 예제를 이용하여 네트워크가 오작동하도록 하는 공격이다.
적대적 공격은 공격자가 가지고 있는 네트워크의 정보에 따라 크게 두가지로 나눌 수 있다.  
- white box 공격: 공격자가 타겟 모델의 구조나 파라미터를 아는 환경에서 하는 적대적 공격.  
-  black box 공격: 공격자가 타겟 모델의 내부 정보를 알 수 없는 환경에서 하는 적대적 공격.
    
### 전이성 기반 적대적 공격 (Transfer-based Attack)
소스 모델을 이용해 생성한 적대적 예제로 타겟 모데을 교란하는 공격이다.
black box 공격ㄱ에서 학습 데이터에 접근할 수 있지만 타겟 모델에는 접근 할 수 없는 경우, 전이성을 기반으로 공격할 수 있다.
전이성이 높은 적대적 예제는 전이성 기반 적대적 공격의 성공률을 증가시킨다.
그러나 적대적 예제가 소스 모델에 과적합(overfitting)된 경우, 낮은 전이성을 가지게 된다.  
- **전이성** : 어떤 모델 _A_ (소스모델)를 이용행 생성한 적대적 예제가 구조가 다른 여러 모델 _B, C, D, E, ..._(타겟 모델)에 대해서도 적대적으로 작용하는 특성.

## 2. Motivation

### Related work

#### 입력의 다변화 (Input Transformation)
적대적 예제의 전이성을 향상시키는 방법 중 하나로, 적대적 예제의 생성과정에서 소스 모델의 입력을 변환하여 소스 모델에 과적합되는 것을 방지하는 방법이다.  
- Translate-Invariant Method [(TIM)](http://openaccess.thecvf.com/content_CVPR_2019/papers/Dong_Evading_Defenses_to_Transferable_Adversarial_Examples_by_Translation-Invariant_Attacks_CVPR_2019_paper.pdf)  
 입력 이미지를  $x$축, $y$축으로 몇 픽셀씩 평행 이동 시킨 여러 이미지를 생성한 후, 그 이미지들을 이용하여 적대적 예제를 생성하는 방법이다. 
 
- Scale-Invariant Method [(SIM)](https://arxiv.org/pdf/1908.06281.pdf)  
 입력 이미지값의 픽셀 값의 크기를 조절하여 생성한 여러 장의 이미지를 생성한 후, 그 이미지 세트를 사용하여 적대적 예제를 생성하는 방법이다.
 
- Diversity Input Method [(DIM)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xie_Improving_Transferability_of_Adversarial_Examples_With_Input_Diversity_CVPR_2019_paper.pdf)  
 입력 이미지를 무작위 리사이징 (resizing) 한 후, 0 값으로 패딩 (padding) 한 이미지를 생성하여 그 이미지로 적대적 예제를 생성하는 방법이다.
 

### Idea

소개하는 논문에서는 입력 이미지가 소스 모델에 과적합되는 것을 방지하기 위한 적대적 변환 네트워크 (Adversarial Transformation Network)를 제안한다.
모든 입력 이미지에 같은 변환을 적용하거나, 변환의 정도만 바꿔서 적용하는 것은 그 변환 자체에 과적합되어, 전이성을 향상시키는 데 한계가 있을 수 있다.
각 입력 이미지에 대해 적합한 변환을 적용하여 효과적으로 소스 모델에 대한 과적합을 피하고 적대적 예제의 전이성을 높이고자 한다.
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

