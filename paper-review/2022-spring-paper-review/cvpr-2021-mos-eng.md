---
description: Huang et al. / MOS: Towards Scaling Out-of-distribution Detection for Large Semantic Space / CVPR 2021
---

# MOS: Towards Scaling Out-of-distribution Detection for Large Semantic Space \[Eng\]

한국어로 쓰인 리뷰를 읽으려면 [**여기**](cvpr-2021-mos-kor.md)를 누르세요.


##  1. Problem definition

Out-of-distribution detection

- Central challenge for safely depolying machin learning model

- Eliminating the impact of distribution shift

- Crucial for building performance-promising deep models

- Highly over-confident predictions under domain shift

  

![](../../.gitbook/assets/2022spring/8/ood_detection.png)



## 2. Motivation



### Related work

- MSP, A baseline for detecting misclasified an out-of-distribution examples in neural networks, ICLR 2017

- Characterizing adversarial subspaces using local instrinsic dimensionality, ICLR 2018
  DNN feature + Euclidean distance

- ODIN, Enhancing the reliability of out-of-distribution image detection in neural networks, ICLR 2018
  pre-trained model, temperature scaling, input preprocessing, ood detector

- Mahalanobis, A simple unified framework for detecting out-of-distribution samples and adversarial attacks, NeurIPS 2018
  pre-trained model (feature ensemble) -> modeling with parameters of Gaussian distribution
  pre-trained model (feature ensemble) -> find closest class -> perturbation -> Mahalanobis distance (generative classifier)

- PaDiM, A patch distribution modeling framework for anomaly detection and localization, ICPR 2020
  Modeling normality for each patch -> multivariate Gaussian (low + high hidden features) -> Mahalanobis distance

  

  **"Only tested with small and low-resolution datasets"**

### Idea



![baseline performance](../../.gitbook/assets/2022spring/8/baseline_ood_detection_performance.png)

Baseline Performance
* Baseline approach
  * performance FPR95 degrades rapidly from 17.34% to 76.94%
  * FPR95 = FPR@TPR95%
  * positive is in-distribution


![](../../.gitbook/assets/2022spring/8/toy_example_in_2d.png)



* Toy Example in 2D
  * in-dist data consist of class-conditional Gaussians
  * Without grouping, ood data is determined by all classes and becomes increasingly complex as the number of classes grows



## 3. Method

Decompose the large semantic sapce into smaller group
* Feature extraction with pre-trained BiT-S
  * Big transfer (bit): General visual representation learning, ECCV 2020 (Google)
* Add "Others" Class
* Minimum Other Score

![](../../.gitbook/assets/2022spring/8/overview.png)



## 4. Experiment & Result



### Experimental setup

* Training and inference
  * training
    * group-based learning, group wise softmax for group $$k$$
    $$ \hat{p}^k=\underset{c \in \mathcal{g'_k}}\max p^k_c({x}) $$
    * Objective is a sum of cross-entropy losses in each group
    $$L_{GS} = -\frac{1}{N}\sum_{n=1}^{N}\sum_{k=1}^K\sum_{c \in \mathcal{g_k}}y_c^k\log(p_c^k(x))$$
    * Use of category "others" creates "virtual" group-level outlier data without any external data
    * in group where c is not included, class "others" will be defined as the ground-truth class
  * inference
    * group-wise class prediction for each group
    $$ \hat{p}^k=\underset{c \in \mathcal{g'_k}}\max p^k_c({x}) , \hat{c}^k=\underset{c \in \mathcal{g'_k}}{\arg \max} p^k_c({x})$$
    * use the maximum group-wise softmax score
    $$k_*=\underset{1 \leq k \leq K}{\arg \max} \hat{p}^k$$
    * Final prediction is category $$\hat{c}^k$$ from groupd $$g_{k_*}$$

* OOD detection with MOS
  * MOS (Minimum Others Score)
  $$S_{MOS}(x) = - \underset{1 \leq k \leq K}{\min} p^k_{others} (x) $$  
  * Category "others" carries useful information for how likely an image is OOD with respect to each group
  * OOD input will be mapped to "others" with high confidence in all groups
  * In-dist input will have a low score on category "others" in the group it belongs to



![](../../.gitbook/assets/2022spring/8/average_of_others_scores.png)



* Datasets
  * ImageNet-1K for in-dist
    * 10 times more labels compared to CIFAR + higher resolution than CIFAR and MNIST
  * iNaturalist
    * Manually select 110 plant classes (not present in ImageNet-1K)
    * Randomly sample 10,000 for 110 classes
  * SUN
    * SUN and ImageNet-1K have overlapping categories
    * Carefully select 50 nature-related concepts (unique in SUN)
    * Randomly sample 10,000 samples for 50 classes
  * Places
    * Not present in ImageNet-1K
    * Randomly sample 10,000 images for 50 categories
  * Textures
    * 5,640 images -> use entire dataset

* Grouping Strategies
  * Taxonomy
    * ImageNet is organized according to the WordNet hierachy (Adopt 8 super-classes)
  * Feature Clustering
    * Feature embedding with pre-trained model -> K-means clustering
  * Random grouping
    * To estimate the lower bound
  * Baseline: MSP
 
![](../../.gitbook/assets/2022spring/8/grouping_strategy.png)

* Ablation Study
  * Size of feature extractor
  * Number of residual block for fine-tuning

### Result

![](../../.gitbook/assets/2022spring/8/table_performance_comparison.png)



![](../../.gitbook/assets/2022spring/8/plot_performance_comparison.png)



## 5. Conclusion

* Group-based OOD detection framework
* MOS, Novel scoring funciton
* Scales OOD detection to large-scale(real-world) setting
* Significantly improve the performance

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

* KAIST AI
* \(optional\) 1~2 line self-introduction
* Contact information \(Personal webpage, GitHub, LinkedIn, ...\)
* **...**

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. Citation of this paper
2. [official github](https://github.com/deeplearning-wisc/large_scale_ood)
3. Citation of related work
4. Other useful materials
5. ...

