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
  'Contrastive Learning', one of self-supervised learning methods, is a method of pre-training by comparing a large amount of unlabeled data with each other, and then performing fine tuning on a labeled dataset.
  In order to do Contrastive Learning with medical data, since there is insufficient data, the author proposed a method of introducing `Federative Learning`, which allows learning with the data that individuals have in one common model.
  Since Federative Learning does not share data directly, they introduced the concept of 'Federative Contrastive Learning', which combines Contrastive Learning after learning a common model using each individual's data while protecting personal information.
  
## 2. Motivation
### Federated Learning
### Contrastive Learning
