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
  ![FL](../../.gitbook/assets/30/federated-learning.png)
</div>
  `Federated Learning(FL)` is to learn a single model with data that `client` has for a common model as shown in the figure above.  
  If the number of clients increases, a model that has been learned about all the data the clients have can be obtained even if the amount of data that an individual has is not large.  
  Since it is possible to learn about the entire data without sharing the data directly, it can be usefully used in the case of medical data that requires the protection of patient's personal information.  
  However, the existing FL is achieved through \"Supervised Learning\" that requires labels for all data.  
  Therefore, in the case of medical data which need high labeling cost, there is a problem in that it is difficult to use FL in practice.  
  

#### 2.1.2. Contrastive Learning
* **Self-Supervised Learning: Generative Learning vs Contrastive Learning**
<div align="center">
  ![Generative-Contrastive](../../.gitbook/assets/30/gen-cont.png)
</div>

  Two representative methods of Self-Supervised Learning are `Generative Learning` and `Contrastive Learning`.
  `Generative Learning` is a method of comparing the loss of the generated output image after inserting the input image as shown in the figure above.
  On the other hand, 'Contrastive Learning' goes through the process of comparing the similarity of the input images.
  It learns \"representations\" through the process of classifying similar images as `positive samples' and different images as `negative samples'.
  
  
