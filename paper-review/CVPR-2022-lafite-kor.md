---
description: Yufan Zhou / LAFITE; Towards Language-Free Training for Text-to-Image Generation / CVPR 2022
---

# LAFITE; Towards Language-Free Training for Text-to-Image Generation \[Kor\]



##  1. Problem definition

이 논문의 주요 task는 text-to-image generation입니다. MS COCO와 같은 complex scene dataset에 대해 text caption을 input으로 현실적인 image를 출력하는 것은 매우 어려운 task입니다. 왜냐하면 text-image pair로 이루어진 dataset은 image만으로 구성된 dataset보다 훨씬 양이 적기 때문입니다.

LAFITE는 pretrained CLIP과 StyleGAN2 구조를 활용해서 text-to-image generation을 구현하였고 dataset의 부족을 해결하기 위해 CLIP을 이용해 pseudo text feature를 구해 활용하였습니다.

## 2. Motivation

우선 text-to-image와 관련한 multimodal task에서 가장 중요한 점은 서로 다른 형태의 두 data를 어떻게 semantically align 시킬 것인가입니다.

### Related work

### - CLIP

CLIP은 open-ai에서 나온 classifier model로 image와 text를 multimodal joint space에 mapping 시키는 방식으로 학습을 시켰습니다. Text를 인코딩할 때는 기존의 다른 여럿 text encoder와 같이 Transformer를 사용했습니다. Image를 인코딩할 때는 CNN이 아닌 Visual Transformer를 사용해서 image의 patch별 feature를 Transformer에 넣는 방식으로 학습을 했습니다. CLIP 역시 multimodal model이므로 학습할 때 text-image pair data가 많이 필요한데 CLIP은 이를 보완하는 새로운 방식을 도입했습니다. Image와 그에 해당하는 label이 있으면 (Image, "a photo of {label}") 이 pair를 이용하여 text caption 없이 학습을 진행하였습니다.

### Idea

Text를 CLIP을 활용해 embedding 시키면 corresponding image를 CLIP을 활용해 embedding 시킨 것과 유사한 곳에 mapping이 됩니다. LAFITE는 text-image pair data 대신 image data만을 활용해서 학습을 하였는데, image의 CLIP embedding과 그에 해당하는 우리가 가지고 있지 않은 text의 CLIP embedding이 유사할 것이라는 가정 하에 text data 대신 CLIP image embedding을 살짝 변형시켜서 만든 pseudo text feature를 사용합니다. 

## 3. Method

Figure 1
Lafite는 두 가지 세팅이 있는데 하나는 text data를 사용하지 않고 image data만 사용하는 language-free setting이고 하나는 image-text pair를 사용하는 standard setting입니다. language-free setting에서는 text feature 대신 앞서말한 pseudo text feature를 사용하는데 image embedding을 standard gaussian noise로 perturb 시킨 방식과 NN을 사용해 noise의 mean과 variance를 구해서 perturb 시키는 방식이 있습니다. 이 외에는 두 세팅은 동일한 방법으로 실험이 진행됩니다.

Figure 2
우선 구조를 살펴보면 StyleGAN2와 거의 유사합니다. 다만 기존의 StyleGAN은 random image 생성을 위한 모듈이므로 noise를 통해 style을 구하는 부분에 text feature(혹은 pseudo text feature)를 넣어서 conditional style vector를 넣게 됩니다.

Figure 3
Loss에서는 Discriminator를 통해 구한 feature를 통해 Real/Fake를 판단하는 conditional GAN loss, Discriminator를 통해 구한 feature와 text feature 사이의 contrastive loss, 새로 생성한 image의 CLIP embedding과 text embedding 사이의 contrastive loss 이렇게 크게 3가지를 사용합니다.

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

