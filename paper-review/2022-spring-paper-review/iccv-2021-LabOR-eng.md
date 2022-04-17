---
description: Inkyu Shin / LabOR- Labeling Only if Required for Domain Adaptive Semantic Segmentation / ICCV 2021
---


# LabOR \[Eng\]

한국어로 쓰인 리뷰를 읽으려면 **여기**를 누르세요.



##  1. Problem definition

- Domain Adaptation (DA)

  - Domain adaptation is a field of computer vision.

  - The main goal of DA is to train a neural network on a **source dataset** and secure a good accuracy on the **target dataset** which is significantly different from the source dataset. 

  - <img src="https://user-images.githubusercontent.com/46951365/163710796-3129e996-1121-48cd-b32d-bf071d42b116.png" alt="drawing" width="600"/>

    

- Unsupervised Domain Adaptation (UDA) 

  - UDA has been actively studied, which is to transfer the knowledge from the labeled source dataset to the unlabeled target domain. 
  - However, UDA still has a long way to go to reach the fully supervised performance

- Domain Adaptation with Few Labels. 

  - Due to the weakness of UDA, some researchers propose to use small parts of ground truth labels of the target dataset.

- Semantic segmentation

  - The task of clustering parts of an image together which belong to the same object class, so-called a form of pixel-level prediction.

  - <img src="https://user-images.githubusercontent.com/46951365/163709097-825280b6-845a-4779-8e5c-c2498ab5d80a.png" alt="drawing" width="600"/>

    

  

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

![image](https://user-images.githubusercontent.com/46951365/163712822-4d8c5e41-4975-44e4-9d97-e11ba20b1b15.png)



### 3.2 Details of methods

- Loss1,2: With the labeled source data and a few labeled target data, Train a model by minimizing cross-entropy loss
- Loss3: Adversarial learning (Details are in [AdaptSeg](https://arxiv.org/abs/1802.10349), [IAST](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710409.pdf) paper)
- Equ 4:  **Inconsistent Mask**(= Masks with different prediction results)
- Loss5: Pseudo label loss for pixel selector model
- Loss6: The classifier discrepancy maximization (Details are in [MCDDA](https://openaccess.thecvf.com/content_cvpr_2018/papers/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.pdf) paper)

![image](https://user-images.githubusercontent.com/46951365/163713620-5e21f50c-7511-43ae-953d-434edc1f97aa.png)



### 3.3 Segment-based and Pinted-based

- This work proposes **two different labeling strategies**, namely “Segment based Pixel-Labeling (SPL)” and “Point based Pixel-Labeling (PPL).”
- **SPL** labels every pixel on the inconsistency mask in a segment-like manner.
- **PPL** places its focus more on the labeling effort efficiency by finding the representative points. The process of finding this point is described below.
  1. Define **the set of uncertain pixels** D^(k)
  2. Compute **the class prototype vector** µ_(k) for each class k as the mean vectors of D^(k)
  3. Select the points that have **the most similar probability pixels** for each prototype vector. 


![image](https://user-images.githubusercontent.com/46951365/163714119-80d18142-1b63-4810-b94e-65c657ea5805.png)





## 4. Experiment & Result

### Experimental setup

* Dataset: The source dataset is [GTA5](https://arxiv.org/pdf/1608.02192v1.pdf) (synthetic dataset) and The target dataset is [Cityscape](https://www.cityscapes-dataset.com/)(real-world data).
* Implementation detail: (1) ResNet101 (2) Deeplab-V2

### Result

![results](https://user-images.githubusercontent.com/46951365/163714468-66da980e-6191-4cff-8690-593fca606f3b.png)

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

{% hint style="warning" %}
You don't need to provide the reviewer information at the draft submission stage.
{% endhint %}

### Author

**Korean Name \(English name\)** 

* Affiliation \(KAIST AI / NAVER\)
* \(optional\) 1~2 line self-introduction
* Contact information \(Personal webpage, GitHub, LinkedIn, ...\)
* **...**



## Reference & Additional materials

1. Citation of this paper
2. Official \(unofficial\) GitHub repository
3. Citation of related work
4. Other useful materials
5. ...

