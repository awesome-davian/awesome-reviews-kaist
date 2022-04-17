##  1. Problem definition
* 망막은 비침습적으로 심혈관계(cardiovascular system)를 관찰할 수 있는 유일한 조직(tissue)이다. 
* 이를 통해 심혈관 질환의 발달과 미세혈관의 형태변화와 같은 구조를 파악할 수 있다.
* 이미지 분할(Image Segmentation)을 통해 상기된 형태적 데이터를 획득 한다면 안과 진단에 중요한 지표가 될수 있다.
* 본 연구에서는, U-Net(및 Residual U-net) 모델을 활용하여 복잡한 망막 이미지(영상)으로 부터 혈관을 분할(segmentation)하고자 한다.   

<p align="left"><img src = "https://user-images.githubusercontent.com/72848264/163723910-a4437d4a-bdb5-492a-a6fc-b9bf930a2307.png">
<img src = "https://user-images.githubusercontent.com/72848264/163723999-192f183e-d400-4266-acaf-e40a1fa93a3f.png " height="50%" width="50%">


##### *U-Net : Biomedical 분야에서 이미지 분할(Image Segmentation)을 목적으로 제안된 End-to-End 방식의 Fully-Convolutional Network 기반 모델이다.*

###### Link: [U-net][googlelink]
[googlelink]: https://medium.com/@msmapark2/u-net-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-u-net-convolutional-networks-for-biomedical-image-segmentation-456d6901b28a 

## 2. Motivation

### Related work

현재 이미지 분할(Image Segmentation)은 대부분 CNN을 기반으로 구성되어 있다.
  - [Cai et al., 2016] 우리가 잘 알고있는 VGG net또한 CNN을 기반으로 하고있다.
  - 

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

