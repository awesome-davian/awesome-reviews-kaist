---
description: (Description) Saeed Anwar, Nick Barnes / Densely Residual Laplacian Super-Resolution / IEEE 2019
---

# Densely Residual Laplacian Super-Resolution \[Kor]


##  1. Problem definition

상대적으로 낮은 해상도의 이미지를 보다 높은 해상도로 복원하는 작업을 초해상화(Super-Resolution)라고 한다.
최근 수년간, 초해상화 작업은 고해상도 이미지를 요하는 작업들에 의해 연구 수요가 증가하였다.
이 논문에서는 단일 저해상도 이미지를 초해상화하는 작업인 Single Image Super-Resolution (SISR)을 목표로 한다.
이미지 초해상화는 입력되는 저해상도 이미지에 대응하는 고해상도 이미지 출력의 크기가 달라서 1개의 유일한 해가 존재하는 것이 아닌 여러 해가 존재하게 되는 불량조건문제(ill-posed problem)가 발생한다. 이러한 문제를 해결하기 위해 심층 컨볼루션 신경망(Deep Convolutional Neural Network,Deep CNN)이 적용되었고 현재까지 많은 종류의 알고리즘이 연구되어 왔다. 
이 논문에서는 현존하는 초해상화 딥러닝 알고리즘보다 더 정확하고 실행 시간이 빠른 모델을 연구하였다.

## 2. Motivation



또한 입력된 이미지의 스케일 별로 특징(feature)을 추출하여 각 스케일

### Related work
현존하는 초해상화를 위한 Deep CNN 알고리즘(SRCNN, RCAN 등)은 매우 복잡한 구조를 가지고 있으며, 복잡한(Deep) 네트워크일 수록 긴 실행시간의 비효율적인 결과를 보여준다. 
이에 따라 네트워크의 깊이를 줄여 효율성을 높인 모델들(DRCN, DRRN 등)이 연구되었다. 하지만 이런 모델들은 총 parameter수는 감소하더라도 총 연산량은 증가하게되는 문제를 갖고있다.
뒤이어 컨볼루션 계층 간의 dense한 연결을 이용한 SRDenseNet과 RDN, parameter 수와 연산속도를 모두 최적화하기 위해 group 컨볼루션을 사용한 CARN이 등장했으나, 대부분의 CNN 모델은 하나의 스케일을 사용하거나 여러 스케일을 사용하더라도 각 스케일의 가중치를 동일하게 부여하기에 다양한 해상도에 따른 적응력이 떨어진다.

### Idea

1. 초해상화의 정확도 향상을 위해 저해상도의 정보를 충분히 이용하는 방법을 적용하였다.
2. Densely connected residual block에서는 여러번 shortcut을 사용하여 원래 이미지의 정보를 포함한 특징 정보를 동시에 학습한다.
3. Laplacian attention network를 통해 여러 스케일의 특징 정보를 학습하며, 모델과 특징 사이의 의존도를 학습한다.

## 3. Method




We recommend you to use the formal definition \(mathematical notations\).


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

