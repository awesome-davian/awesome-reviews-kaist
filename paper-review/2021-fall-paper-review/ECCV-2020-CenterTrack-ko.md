---
description: Xingyi Zhou et al. / Tracking Objects as Points / ECCV 2020 Spotlight
---

# Tracking Objects as Points \[Kor]

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

MOT는 연속적인 프레임에서 객체를 검출하고, 검출된 객체의 추적하는 task입니다. 여기서 연속적인 프레임은 LiDAR의 point cloud 또는 이미지 등이 될 수 있습니다. 이렇게 검출된 객체를 추척하는 이유는 각 객체의 이동 경로를 파악하기위해서 입니다. 이렇게 추적되어 생성된 객체의 궤적 또는 경로는 action recognition, trajectory prediction 등 다양한 분야에서 활용될 수 있습니다.

##  1. Problem definition

이미지 기반의 다중 객체 추적 문제는 일반적으로 다음과 같이 정의할 수 있습니다.

시간 $$t$$ 와 이전 프레임 $$t-1$$에서 카메라를 통해 들어온 이미지를 각각 $$I^{(t)} \in R^{W \times H \times 3}$$ , $$I^{(t-1)} \in \mathbb{R}^{W \times H \times 3}$$라고 정의하고 \$$t-1\$$에서 검출되고 추적된 객체 정보를 $$T^{(t-1)}=\{b_0^{(t-1)}, b_0^{(t-1)},\ldots\}$$라고 했을 때 이미지 기반 MOT의 목적은 $$I^{(t)}, I^{(t-1)}$$ 그리고 $$T^{(t-1)}$$를 입력으로 사용하여 $$t$$에 존재하는 객체들의 정보에 해당하는 $$T^{(t)}=\{b_0^{(t)}, b_0^{(t)},\ldots\}$$를 찾고 두 시계열 이미지에서 검출된 같은 객체에 대해 같은 $$id$$를 부여하는 것 입니다. 객체 정보 $$b={\textbf{p},\textbf{s},w,id}$$에서 $$\textbf{p} \in \mathbb{R}^{2}$$ 는 객체의 중심점의 위치, $$\textbf{s}\in \mathbb{R}^{2}$$ 사이즈, $$w \in [0,1]$$ 는 confidence, 그리고 $$id \in \mathbb{L}$$은 unique identification 에 해당합니다.

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
