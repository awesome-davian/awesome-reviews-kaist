---
description: Yufan Zhou / LAFITE; Towards Language-Free Training for Text-to-Image Generation / CVPR 2022
---

# LAFITE; Towards Language-Free Training for Text-to-Image Generation \[Kor\]



##  1. Problem definition

이 논문의 주요 task는 text-to-image generation입니다. MS COCO와 같은 complex scene dataset에 대해 text caption을 input으로 현실적인 image를 출력하는 것은 매우 어려운 task입니다. 왜냐하면 text-image pair로 이루어진 dataset은 image만으로 구성된 dataset보다 훨씬 양이 적기 때문입니다.

LAFITE는 pretrained CLIP과 StyleGAN2 구조를 활용해서 text-to-image generation을 구현하였고 dataset의 부족을 해결하기 위해 CLIP을 이용해 pseudo-text-feature를 구해 활용하였습니다.

## 2. Motivation

우선 text-to-image와 관련한 multimodal task에서 가장 중요한 점은 서로 다른 형태의 두 data를 어떻게 semantically align 시킬 것인가입니다.

### Related work

### 1) CLIP

CLIP은 open-ai에서 나온 classifier model로 image와 text를 multimodal joint space에 mapping 시키는 방식으로 학습을 시켰습니다. Text를 CLIP을 활용해 embedding 시키면 corresponding image를 CLIP을 활용해 embedding 시킨 것과 유사한 곳에 mapping이 됩니다. LAFITE는 text-image pair data 대신 image data만을 활용해서 학습을 하였는데, image의 CLIP embedding과 그에 해당하는 우리가 가지고 있지 않은 text의 CLIP embedding이 유사할 것이라는 가정 하에 text data 대신 CLIP image embedding을 살짝 변형시켜서 만든 pseudo text feature를 사용합니다.

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

