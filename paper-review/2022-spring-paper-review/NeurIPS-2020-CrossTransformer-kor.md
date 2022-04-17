---
description: Carl Doersch et al. / CrossTransformer - spatially-aware few-shot transfer / NeurIPS 2020
---

# CrossTransformer \[Kor\]

##  1. Problem definition

본 논문은 Transformer에 PrototypeNet을 결합한 few-shot learning을 다룹니다.

현재의 vision system은 소수의 데이터로 새로운 task를 부여하면, 그 성능이 현저히 저하됩니다. 
즉, 방대한 양의 데이터가 있을 때만 새로운 task를 성공적으로 수행할 수 있습니다. 
하지만 현실 세계에서 매번 방대한 양의 데이터를 학습하기는 어렵고, 그렇게 labelling된 데이터를 구하기도 어렵습니다.

예를 들어 보겠습니다.
Home Robot이 집에 새로운 물건이 들어왔는데, 이를 전혀 인식하지 못한다면 어떻게 될까요?
공장에서 assurance system이 새로운 제품에 대해 결함을 바로 인식하지 못한다면 또 어떨까요?
Home Robot과 assurance system을 다시 학습시켜야 하는데, 새로운 데이터라 학습이 어려울 것입니다. 

Vision System의 궁극적 목표는 새로운 환경, 즉 task에 곧바로 적용하는 것입니다. 
이에 저자들은 적은 양의 데이터로 model의 빠른 학습을 통해, 곧바로 새로운 data를 처리할 수 있도록 하는 것을 목표로 Transformer를 발전시켰습니다. 
즉, Transformer가 few-shot learning이 가능하도록 한 것입니다.
<br></br>

먼저, 저자들이 생각한 기존 few-shot learning의 문제점을 살펴보겠습니다. 

지금까지 few-shot learning은 meta-learning과 함께 연구되어 왔습니다. 
그리고 Prototypical Nets는 Meta-Dataset의 SOTA model 입니다. 
하지만 이 Prototypical Nets는 training-set의 image class만 represent하고 out-of-distribution classes, 즉 새로운 image class를 classify하는데 필요할 수 있는 정보는 버려버린다는 문제를 가지고 있습니다. 

아래의 그림을 보겠습니다.

![Figure 1: Illustration of supervision collapse with nearest neighbors](../../.gitbook/assets/2022spring/20/Fig1.png)

<div align="center"><b>Figure 1: Illustration of supervision collapse with nearest neighbors</b></div>
<br></br>

## 2. Motivation

In this section, you need to cover the motivation of the paper including _related work_ and _main idea_ of the paper.

### Related work

1. Few-shot image classification

2. Attention for few-shot learning

3. correspondences for visual recognition

4. Self-supervised learning for few-shot

### Idea

After you introduce related work, please illustrate the main idea of the paper. It would be great if you describe the idea by comparing or analyzing the drawbacks of the previous work.

## 3. Method

![Figure 2: CrossTransformer](../../.gitbook/assets/2022spring/20/Fig2.png)
<div align="center"><b>Figure 2: CrossTransformers</b></div>

![Figure 3: Visualization of the attention](../../.gitbook/assets/2022spring/20/Fig3.png)
<div align="center"><b>Figure 3: Visualization of the attention</b></div>

## 4. Experiment & Result

This section should cover experimental setup and results.  
Please focus on how the authors of paper demonstrated the superiority / effectiveness of the proposed method.

Note that you can attach tables and images, but you don't need to deliver all materials included in the original paper.

### Experimental setup

* Dataset: 아래 표(Figure 4, 5)의 x축을 참고해 주세요
* Baselines: 아래 표(Figure 5)의 y축을 참고해 주세요
* Training setup: 
* Evaluation metric: Accuracy

### Result

![Figure 4: Effects of architecture and SimCLR Episodes on Prototypical Nets, for Meta-Dataset Train-on-ILSVRC](../../.gitbook/assets/2022spring/20/Fig4.png)
<div align="center"><b>Figure 4: Effects of architecture and SimCLR Episodes on Prototypical Nets, for Meta-Dataset Train-on-ILSVRC</b></div>

<br></br>
![Figure 5: CrossTransformers(CTX) comparison to state-of-the-art](../../.gitbook/assets/2022spring/20/Fig5.png)
<div align="center"><b>Figure 5: CrossTransformers(CTX) comparison to state-of-the-art</b></div>

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

### Author

**성지현 \(Jihyeon Seong\)** 

* M.S. student in KAIST AI
* [Github](https://github.com/monouns)
* E-mail: tjdwltnsfP1@gmail.com / jihyeon.seong@kaist.ac.kr

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. Carl Doersch, Ankush Gupta, Andrew Zisserman, "CrossTransformers: spatially-aware few-shot transfer", 2020 NeurIPS
2. [Official GitHub repository](https://github.com/google-research/meta-dataset)
3. [Unofficial Github repository with Pytorch](https://github.com/lucidrains/cross-transformers-pytorch)
4. Citation of related work
5. Other useful materials

