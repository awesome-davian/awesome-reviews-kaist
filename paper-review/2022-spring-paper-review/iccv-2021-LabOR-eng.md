---
description: Inkyu Shin / LabOR- Labeling Only if Required for Domain Adaptive Semantic Segmentation / ICCV 2021
---


# LabOR \[Eng\]

한국어로 쓰인 리뷰를 읽으려면 [여기](https://awesome-davian.gitbook.io/awesome-reviews/paper-review/2022-spring-paper-review/iccv-2021-labor-kor)를 누르세요.



##  1. Problem definition

- Domain Adaptation (DA)

  - Domain adaptation is a field of computer vision.

  - The main goal of DA is to train a neural network on a **source dataset** and secure a good accuracy on the **target dataset** which is significantly different from the source dataset. 

  - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2022-06/img1.png?raw=true" alt="drawing" width="900"/>

    

- Unsupervised Domain Adaptation (UDA) 

  - UDA has been actively studied, which is to transfer the knowledge from the labeled source dataset to the unlabeled target domain. 
  - However, UDA still has a long way to go to reach the fully supervised performance

- Domain Adaptation with Few Labels. 

  - Due to the weakness of UDA, some researchers propose to use small parts of ground truth labels of the target dataset.

- Semantic segmentation

  - The task of clustering parts of an image together which belong to the same object class, so-called a form of pixel-level prediction.

  - <img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2022-06/img2.png?raw=true" alt="drawing" width="900"/>

    

  

## 2. Motivation

- In order to reduce the efforts of the human annotator, this work studies domain adaptation with **the least labels** of target dataset. 
- **What points** should be labeled to maximize the performance of the segmentation model?
- This work aims to find **these points**, i.e, an efficient pixel-level sampling approach.

### Related work

1. Unsupervised Domain Adaptation
   - Adversarial learning approaches have aimed to minimize the discrepancy between source and target feature distribution
   - this approach has been studied mainly on output-level alignment \[[AdaptSeg](https://arxiv.org/abs/1802.10349), [ADVENT](https://arxiv.org/abs/1811.12833)\].
   - But, despite much research on UDA, the performance of UDA is **much lower than that of supervised learning.**
2. Domain Adaptation with Few Labels
   - In order to mitigate the aforementioned limitation, some researchers attempt to use a few target labels.\[[Alleviating semantic-level shift](https://arxiv.org/abs/2004.00794), [Active Adversarial Domain Adaptation](https://arxiv.org/abs/1904.07848), [Playing for Data](https://arxiv.org/abs/1608.02192), [DA_weak_labels](https://arxiv.org/abs/2007.15176)\]
   - These works aim to find data (**full image label**) that would increase the performance of the model the most. 
   - In contrast, this work focuses on the **pixel-level label** that would have the best potential performance increase.

### Idea

- This work utilizes "the novel predictor" to find **uncertain regions** that require human annotations and train these regions with ground truth labels in a supervised manner. 
- **The uncertain regions** would have the best potential performance increase instead of randomly picking labels.





## 3. Method

### 3.1 Method Summary

- Please refer to \[the below note-taken figure\] and \[the ordered method summary\] together. 
- The below specification is corresponding to the green number in the figure.

1. "The novel predictor"(**pixel selector model (model B)**) consists of a shared backbone model (Ex, Resnet50) and two pixel-level classifiers.
2. A mini-batch of target data is passed through the backbone and both classifiers. Then we can extract two segmentation results.
3. With two segmentation results, **Inconsistent Mask**(= Masks with different prediction results) is calculated.
4. Among inconsistent mask, the human annotator give the real label, which is supervision signal for **semantic segmentation model (model A)**
5. Both above-labeled target data and originally labeled source data are used for optimize **semantic segmentation model** (model A) while output-level adversarial learning\[[AdaptSeg](https://arxiv.org/abs/1802.10349)\] is also utilized. 
6. For updating classifiers in pixel selector model (model B), parameters in each classifier are applied to the loss to push away from each other, i.e, maximization of the discrepancy between the two classifiers \[[Maximum classifier discrepancy](https://arxiv.org/abs/1712.02560)\].

<img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2022-06/img3.png?raw=true" alt="drawing" width="900"/>



### 3.2 Details of methods

- Loss1,2: With the labeled source data and a few labeled target data, Train a model by minimizing cross-entropy loss
- Loss3: Adversarial learning (Details are in [AdaptSeg](https://arxiv.org/abs/1802.10349), [IAST](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710409.pdf) paper)
- Equ 4:  **Inconsistent Mask**(= Masks with different prediction results)
- Loss5: Pseudo label loss for pixel selector model
- Loss6: The classifier discrepancy maximization (Details are in [MCDDA](https://openaccess.thecvf.com/content_cvpr_2018/papers/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.pdf) paper)

<img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2022-06/img4.png?raw=true" alt="drawing" width="900"/>



### 3.3 Segment-based and Pinted-based

- This work proposes **two different labeling strategies**, namely “Segment based Pixel-Labeling (SPL)” and “Point based Pixel-Labeling (PPL).”
- **SPL** labels every pixel on the inconsistency mask in a segment-like manner.
- **PPL** places its focus more on the labeling effort efficiency by finding the representative points. The process of finding this point is described below.
  1. Define **the set of uncertain pixels** D^(k)
  2. Compute **the class prototype vector** µ_(k) for each class k as the mean vectors of D^(k)
  3. Select the points that have **the most similar probability pixels** for each prototype vector. 

<img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2022-06/img5.png?raw=true" alt="drawing" width="900"/>





## 4. Experiment & Result

### Experimental setup

* Dataset: The source dataset is [GTA5](https://arxiv.org/pdf/1608.02192v1.pdf) (synthetic dataset) and The target dataset is [Cityscape](https://www.cityscapes-dataset.com/)(real-world data).
* Implementation detail: (1) ResNet101 (2) Deeplab-V2

### Result

<img src="https://github.com/junha1125/Imgaes_For_GitBlog/blob/master/2022-06/img6.png?raw=true" alt="drawing" width="900"/>

- **Figure 1**
  1. LabOR (PPL and SPL) significantly outperforms previous UDA models ([IAST](https://github.com/Raykoooo/IAST)).
  2. SPL shows the performance comparable with fully supervised learning
  3. PPL achieves compatible performance improvements compared to [WDA](https://arxiv.org/abs/2007.15176).
- **Table 1**
  1. They show the quantitative results of both of our methods PPL and SPL compared to other state-of-the-art UDA methods.
  2. Even when compared to the fully supervised method, **SPL** is only down by 0.1 mIoU in comparison.
  3.  **PPL** also shows significant performance gains over previous state-of the-art UDA or WDA methods.
- **Figure 2**
  1. Qualitative result of SPL
  2. The proposed method, SPL, shows the correct segmentation result similar to the fully supervised approach.



## 5. Conclusion

- This work proposes a new framework for domain adaptive semantic segmentation in a human-in-the-loop manner.
- Two pixel-selection methods that we call “Segment based Pixel-Labeling” and “Point based Pixel-Labeling.” are introduced.
- limitation (my thought)
  1. In **SPL** and **PPL**, human annotators need to label \[2.2% area\] and \[40 labeled points\] per image. It sounds like it needs a few labeled annotations and efforts. But if I were a human annotator, I may think the effort for labeling \[2.2% area\] and \[40 labeled points\] is equal to one for labeling a full image using [Interactive segmentation tool](https://github.com/saic-vul/ritm_interactive_segmentation). Specifically, labeling \[2.2% area\] is likely to be harder to label (see the image in Sec. 3.3).



### Take home message \(오늘의 교훈\)

> It may be **more** efficient to obtain a supervision signal at a low cost **than** using complex unsupervised methods to achieve very small performance gains.
>





## Author / Reviewer information

### Author

1. **신인규 \(**Inkyu Shin**\)** 
   * KAIST / RCV Lab
   * https://dlsrbgg33.github.io/
2. **김동진 \(**DongJin Kim**\)** 
   * KAIST / RCV Lab
   * https://sites.google.com/site/djkimcv/
3. **조재원 \(**JaeWon Cho**\)** 
   * KAIST / RCV Lab
   * https://chojw.github.io/





## Reference & Additional materials

1. Citation of this paper
   1. [Towards Fewer Annotations: Active Learning via Region Impurity and Prediction Uncertainty for Domain Adaptive Semantic Segmentation](https://www.semanticscholar.org/paper/Towards-Fewer-Annotations%3A-Active-Learning-via-and-Xie-Yuan/34bc77414f517268e890c8dd31d91d1c65b480cd)
   2. [D2ADA: Dynamic Density-aware Active Domain Adaptation for Semantic Segmentation](https://www.semanticscholar.org/paper/D2ADA%3A-Dynamic-Density-aware-Active-Domain-for-Wu-Liou/6935ed45c7218f236fc6adba7066a395e6c6107f)
   3. [Unsupervised Domain Adaptation for Semantic Image Segmentation: a Comprehensive Survey](https://www.semanticscholar.org/paper/Unsupervised-Domain-Adaptation-for-Semantic-Image-a-Csurka-Volpi/abb79bf15896e0922427ca9d35b0e36ec6718e6e)
   4. [ADeADA: Adaptive Density-aware Active Domain Adaptation for Semantic Segmentation](https://www.semanticscholar.org/paper/ADeADA%3A-Adaptive-Density-aware-Active-Domain-for-Wu-Liou/9371f28a456121815431373fc083072456a1b611)
   5. [MCDAL: Maximum Classifier Discrepancy for Active Learning](https://www.semanticscholar.org/paper/MCDAL%3A-Maximum-Classifier-Discrepancy-for-Active-Cho-Kim/86b13e61d93b1f5c72b834c37ad6c129d6364fa5)

2. Reference for this post
   1. [AdaptSeg](https://arxiv.org/abs/1802.10349) 
   2. [ADVENT](https://arxiv.org/abs/1811.12833)
   2. [IAST](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710409.pdf) 
   3. [Alleviating semantic-level shift](https://arxiv.org/abs/2004.00794), [Active Adversarial Domain Adaptation](https://arxiv.org/abs/1904.07848) 
   4. [Playing for Data](https://arxiv.org/abs/1608.02192)
   5. [DA_weak_labels](https://arxiv.org/abs/2007.15176)
   4. [Maximum classifier discrepancy](https://arxiv.org/abs/1712.02560)
   5.  [WDA](https://arxiv.org/abs/2007.15176)

