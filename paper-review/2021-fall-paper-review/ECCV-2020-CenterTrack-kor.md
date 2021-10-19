---
description: Xingyi Zhou / Tracking Objects as Points / ECCV 2020
---

# Tracking Objects as Points \[Kor\]

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

## 0. Introduction

논문에서 제안하는 CenterTrack은 이미지 기반의 Multi-Object Tracking(MOT)을 위한 모델입니다. 따라서 CenterTrack을 잘 이해하기 위해서는 MOT가 어떤 task인지를 이해할 필요가 있습니다. 

MOT는 연속적인 프레임에서 객체를 검출하고, 검출된 객체의 추적하는 task입니다. 여기서 연속적인 프레임은 LiDAR의 point cloud 또는 이미지 등이 될 수 있습니다. 이렇게 검출된 객체를 추척하는 이유는 각 객체의 이동 경로를 파악하기위해서 입니다. 이렇게 추적되어 생성된 객체의 궤적 또는 경로는 action recognition, trajectory precdiction 등 다양한 분야에서 활용될 수 있습니다.


## 1. Introduction

기존의 객체 추적 연구는 tracking-by-detection의 프레임워크를 많이 따랐습니다. 이는 각각의 프레임에서 객체를 검출하고, 검출된 객체를 매칭하여 추적하는 방법으로 딥러닝의 발전에 따라 객체 검출 기술이 급속도로 발전하다 이 흐름에 따라 객체 검출 결과를 잘 활용하는 객체 추적 기술이라고 할 수 있습니다. 하지만 tracking-by-detection 방법의 경우 복잡한 association, 즉 복잡한 매칭 전략이 필요하기 때문에 네트워크가 전체적으로 느려지고 복잡해지는 경향이 있습니다. 이를 해결하기 위해 최근들어 객체 검출과 추적을 함께 진행하는 joint detection and tracking의 프레임워크에 대한 연구가 많이 진행되고 있으며 CenterTrack 또한 이 방법에 해당합니다.

CenterTrack에서 주장하는 contribution은 다음과 같이 정리할 수 있습니다.


##  1. Problem definition


$$
Please provide the problem definition in this section.

We recommend you to use the formal definition \(mathematical notations\).

## 2. Motivation

In this section, you need to cover the motivation of the paper including _related work_ and _main idea_ of the paper.

### Related work

Please introduce related work of this paper. Here, you need to list up or summarize strength and weakness of each work.

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
