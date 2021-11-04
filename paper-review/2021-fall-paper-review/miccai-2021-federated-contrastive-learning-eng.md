---
description: 'Yawen Wu / Federated Contrastive Learning for Volumetric Medical Image Segmentation / MICCAI 2021 Oral'
---

# Federated Contrastive Learning for Volumetric Medical Image Segmentation [Eng]

한국어로 쓰인 리뷰를 읽으려면 [**여기**](https://github.com/2na-97/awesome-reviews-kaist/blob/master/paper-review/2021-fall-paper-review/miccai-2021-federated-contrastive-learning-kor.md)를 누르세요.

## 1. Problem Definition
  In this paper, we presented two major problems of deep learning models with medical images.
  1. Supervised learning, which trains with labeled data, has shown good results in many fields, but medical experts are required to obtain labels of medical data and it takes a considerable amount of time. In this reason, it is hard to find a large amount of high-quality medical datasets.
  2. Since the privacy problems are important to protect patient information, it is difficult to exchange medical data held by hospitals or doctors.

  In general, a method called self-supervised learning is being actively studied to learn from data lacking labels.  
  `Contrastive Learning`, one of self-supervised learning methods, is a method of pre-training by comparing a large amount of unlabeled data with each other, and then performing fine tuning on a labeled dataset.  
  In order to do Contrastive Learning with insufficient medical data, the author proposed a method of introducing `Federative Learning`, which allows learning with the data that individuals have in one common model.  
  Inspired by the fact that Federative Learning does not share data directly, the concept of 'Federative Contrastive Learning' that combines Contrastive Learning after learning a common model using each individual's data while protecting personal information has been introduced.  
  
## 2. Motivation
### 2.1. Related Work
#### 2.1.1. Federated Learning
<div align="center">
  <img src="../../.gitbook/assets/30/federated-learning.png" width="100%" alt="Federated Learning"></img><br/>
</div>


  `Federated Learning(FL)` is to learn a single model with data that `client` has for a common model as shown in the figure above.  
  If the number of clients increases, a model that has been learned about all the data the clients have can be obtained even if the amount of data that an individual has is not large.  
  Since it is possible to learn about the entire data without sharing the data directly, it can be usefully used in the case of medical data that requires the protection of patient's personal information.  
  However, the existing FL is achieved through "Supervised Learning" that requires labels for all data.  
  Therefore, in the case of medical data which need high labeling cost, there is a problem in that it is difficult to use FL in practice.  
  

#### 2.1.2. Contrastive Learning
* **Self-Supervised Learning: Generative Learning vs Contrastive Learning**
<div align="center">
  <img src="../../.gitbook/assets/30/gen-cont.png" width="100%" alt="Generative-Contrastive"></img><br/>
</div>


  Two representative methods of Self-Supervised Learning are `Generative Learning` and `Contrastive Learning`.  
  `Generative Learning` is a method of comparing the loss of the generated output image after inserting the input image as shown in the figure above.  
  On the other hand, `Contrastive Learning` goes through the process of comparing the similarity of the input images.  
  It learns "representations" through the process of classifying similar images as `positive samples` and different images as `negative samples`.  
  
  
  * **Contrastive Learning: SimCLR**
<div align="center">
  <img width="50%" alt="SimCLR Illustration" src="https://1.bp.blogspot.com/--vH4PKpE9Yo/Xo4a2BYervI/AAAAAAAAFpM/vaFDwPXOyAokAC8Xh852DzOgEs22NhbXwCLcBGAsYHQ/s1600/image4.gif">
</div>
  
<div align="right">
  Cite: ![SimCLR github page](https://github.com/google-research/simclr)
</div>

  
  The most famous paper of contrastive learning that performs self-supervised learning by _comparing representations_ between images is `SimCLR`.  
  First, `SimCLR` applys different augmentation to the same image as shown in the figure above.  
  Then it tries to increase the similarity of augmented images from the same image, while it tends to decrease the similarity with other images.  
  A function that learns a meaningful representation from an image trains to ignore changes caused by augmentation for the same image, and increases the distance between representations for different images.  
  In this way, it is possible to obtain a pre-trained encoder that extracts meaningful features of an image even without labels.  
  
  After obtaining the pre-trained encoder weights learned with a lot of data through `Contrastive Learning (CL)`, it be trained on the target dataset through the process of fine-tuning.  
  Briefly saying, we can think of it as similar to training a model based on pre-trained weights from massive data such as ImageNet, rather than training a model from scratch.  
  CL learns how to extract _meaningful features_ from a lot of data without labels, so that it can be used as a pre-training weight for various datasets.  
  As a result, it can show better performance than training a model from scratch.  (It is the same reason why we usally use ImageNet pre-trained weight!)  
  Various CL methods such as SimCLR, MoCo, and BYOL are being actively studied, and they show similar levels of accuracy to supervised learning.  
  

### 2.2. Idea
![FCL](../../.gitbook/assets/30/FCL.png)
<div align="center">
  <img width="100%" alt="FCL" src="../../.gitbook/assets/30/FCL.png">
</div>

  본 논문에서는 Federated Learning의 단점과 Contrastive Learning의 단점을 보완한 후 두 학습 방법의 장점만 활용하여 합친 **Federated Contrastive Learning(FCL)** 이라는 방법을 제안한다. 저자가 주장하는 FL과 CL의 문제점은 다음과 같다.

* 개인이 가지고 있는 적은 양의 데이터로 CL을 할 경우, positive sample과 negative sample의 양도 부족하고 데이터 간의 다양성이 떨어지기 때문에 CL을 하는 데에 효과적이지 않다.
* 기존의 FL 방법으로 단순히 개인이 학습한 모델을 합치게 되면, 개인이 가지고 있는 데이터에 치중된 feature space가 만들어져 모델을 합치는 과정에서 feature space 간의 불일치가 일어나 실질적인 성능 향상 효과를 보기 어려울 수도 있다.

FCL에서는 이러한 단점들을 보완하기 위한 아이디어를 제시했는데, 데이터의 보안을 유지하면서도 client간의 데이터 교류를 할 수 있도록 하여 많은 양의 데이터를 기반으로 자기 지도 학습을 하는 효과를 볼 수 있도록 했다. 저자는 이러한 방법이 실전에서도 차용할 수 있는 의료 영상 딥러닝 모델이 될 것이라 기대했다.


