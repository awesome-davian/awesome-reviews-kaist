---
description: Huang et al. / MOS; Towards Scaling Out-of-distribution Detection for Large Semantic Space / CVPR 2021
---

# MOS: Towards Scaling Out-of-distribution Detection for Large Semantic Space \[Eng\]

한국어로 쓰인 리뷰를 읽으려면 [**여기**](cvpr-2021-mos-kor.md)를 누르세요.


##  1. Problem definition

>Out-of-distribution detection is..
>- Central challenge for safely depolying machin learning model
>- Eliminating the impact of distribution shift
>- Crucial for building performance-promising deep models
>- Highly over-confident predictions under domain shift

Out-of-distribution detection (hereinafter referred to as OOD detection) is a central challenge for safely deploying machine learning models. It is a technology that detects and informs situations that cannot guarantee the performance of learning-based methods such as class and outlier samples that are unseen when learning models. It's also essential performance indicators with accuracy and speed when applying solutions to the real world. Only when distribution shifts can be detected and notified can it be a reliable solution. Until now, deep neural networks have a problem presenting wrong answers with high confidence even in the domain shift situation (over-confident predictions)

Generally, we define OOD detection as a binary classification and the below figure shows the basic in-distribution and out-of-distribution dataset configurations.
  

![](../../.gitbook/assets/2022spring/8/ood_detection.png)



## 2. Motivation
The previous works related to OOD detection have been evaluated only with a limited number of classes and low-resolution images such as MNIST and CIFAR-10. On the other hand, this paper attempts to check and improve the performance change of OOD detection in more diverse class situations with various datasets including high-resolution images by scaling up this limited situation like the real world.


### Related work

The related works that are introduced and compared in this paper are as follows.

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
If the total number of classes increased, the change in performance of the MSP baseline algorithm, which is an existing study, is shown in the graph below.

The AUROC and FPR95 performance of this OOD detection algorithm rapidly decrease as the total number of classes increases. It shows that the FPR95 decreased from 17.34% to 76.94%. The lower the FPR, the better the performance indicator. FPR95 means the performance of the False Positive Rate at a True Positive Rate of 95%, where positive means in-distribution.


![baseline performance](../../.gitbook/assets/2022spring/8/baseline_ood_detection_performance.png)

The author explains this phenomenon with toy examples in 2D. In the below figure, as the class increases, the decision boundary becomes complicated between in-distribution data and out-of-distribution, increasing the difficulty of classifying tasks. Therefore, to solve this problem, the author suggests grouping in-distribution classes with each other.


![](../../.gitbook/assets/2022spring/8/toy_example_in_2d.png)



* Toy Example in 2D
  * in-dist data consist of class-conditional Gaussians
  * Without grouping, ood data is determined by all classes and becomes increasingly complex as the number of classes grows


## 3. Method

In this paper, large datasets such as ImageNet-1k, iNaturalist, SUN, Places, and Textures were used to extend the OOD detection task to the real world scale, decompose this broadened semantic space into small groups, add a new virtual class called "Others" and propose Minimum Other Score (MOS) to measure OOD. Pre-trained BiT-S was used to extract the feature vector for the input image. (Big transfer (bit): General visual representation learning, ECCV 2020)

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


### Result

The result of OOD detection experiment. All methods were evaluated using the pre-trained BiT-S-R101x1 backbone, which pre-learned ImageNet-1k as in-distribution dataset.

The higher the AUROC and the lower the FPR95, the better the OOD detection performance, and the better the AUROC and the lower the FPR95, all achieved excellent performance and short test time except for the Textures dataset.

![](../../.gitbook/assets/2022spring/8/table_performance_comparison.png)



![](../../.gitbook/assets/2022spring/8/plot_performance_comparison.png)

* Ablation Study
  * Size of feature extractor
  * Number of residual block for fine-tuning
  * 
![](../../.gitbook/assets/2022spring/8/ablation_effective.png)

![](../../.gitbook/assets/2022spring/8/ablation_finetune.png)

## 5. Conclusion

In this paper, beyond evaluating OOD detection limited to limited datasets in previous works, they redefine the OOD detection problem in a real-world setting, identify the issue, and propose a way to overcome it.

In particular, they propose a group-based OOD detection framework to overcome the rapidly deteriorating OOD detection performance as classes increase in existing algorithms. They offer a virtual class concept called others and a concrete method for learning with existing datasets. They propose a novel method of measuring OOD degree, called minimum others score, and show a significant improvement in performance through a comparative experiment.


### Take home message 

> The actual problems of out-of-distribution detection in the wild are more complicated than benchmarks using public datasets that have been used in previous works. In this case, the divide and conquer strategy can be helpful through semantic grouping or clustering algorithms.
>
> It is an excellent approach to define some virtual classes and design how to train them. 


## Author / Reviewer information

### Author

**신호근 \(Ho-Kuen Shin\)** 

* KAIST Graduate School of AI
* vision@kaist.ac.kr
* SAIT


### Reviewer

1. 이지현 \(jyunlee\)
2. 손민지 \(ming1st\)
3. 윤여동 \(YeodongYoun95\)


## Reference & Additional materials

1. Huang et al, MOS: Towards Scaling Out-of-distribution Detection for Large Semantic Space, CVPR 2021
2. Hendrycks et al, A baseline for detecting misclasified an out-of-distribution examples in neural networks, ICLR 2017
3. Liang et al, Enhancing the reliability of out-of-distribution image detection in neural networks, ICLR 2018
4. Lee et al, A simple unified framework for detecting out-of-distribution samples and adversarial attacks, NeurIPS 2018
5. [official github](https://github.com/deeplearning-wisc/large_scale_ood)
6. [Blog article for out-of-distribution detection](https://hoya012.github.io/blog/anomaly-detection-overview-2/)
