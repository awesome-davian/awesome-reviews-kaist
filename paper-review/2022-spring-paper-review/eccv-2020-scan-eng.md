---
description: Gansbeke et al. / SCAN: Learning to Classify Images without Labels / ECCV 2020
---

# SCAN: Learning to Classify Images without Labels \[Eng\]

##  1. Problem definition

The goal of *unsupervised image classification* is to group images into clusters such that images within the same cluster belong to the same or similar semantic classes, while images in different clusters are semantically dissimilar. 
This happens when there is no access to ground-truth semantic labels at training time, or the semantic classes or even their total number are not priori known.
This paper proposes SCAN (Semantic Clustering by Adopting Nearest neighbors) which is a two-step approach for unsupervised image classification. 

## 2. Motivation

In this section, we cover the related works and main idea of the proposed method, SCAN.

### Related work

The task of unsupervised image classification have recently attracted considerable attention in two dominant paradigms.

#### Representation Learning
- Step 1: Self-supervised learning (e.g., SimCLR, MoCo)
- Step 2: Clustering (e.g., K-Means)
- **Problems:** Imbalanced clusters or mismatch with semantic labels

#### End-to-end Learning
- Iteratively refine the clusters based on the supervision from confident samples.
- Maximize mutual information between an image and its augmentations.
- **Problems:** Initialization-sensitive or heuristic mechanisms

### Idea

To address the limitations of the existing methods, SCAN is designed as a two-step algorithm for unsupervised image classification.
- Step 1: Learn feature representations and mine K-nearest neighbors.
- Step 2: Train a clustering model to integrate nearest neighbors.

In step 1, instead applying K-means directly to the image features, SCAN mines the nearest neighbros of each image.
In step 2, SCAN encourages invariance with respect to the nearest neighbors and not soley with respect to augmentations.

## 3. Method

### Step 1: Learn feature representations and mine K-nearest neighbors.

- Certain pretext tasks may yield undesired features for semantic clustering. 
  - Thus, SCAN selects a pretext task that minimizes the distance between an image and its augmentations.
  - **Instance discrimination** satisfies this condition.
  
<img src="/.gitbook/assets/2022spring/3/pretext_loss.PNG" width="400" align="center">

- For each image, mine K nearest neighbors.
  - The nearest neighbors tend to belong to the same semantic labels.  

<img src="/.gitbook/assets/2022spring/3/observation.png" width="300" align="center">

### Step 2: Train a clustering model to integrate nearest neighbors.

- Adopt the nearest neighbors as the prior for semantic clustering.
  - The first term imposes neighbors to have similar labels.
  - The second term maximizes the entropy to avoid assigning all samples to a single cluster.

<img src="/.gitbook/assets/2022spring/3/clustering_loss.PNG" width="500" align="center">

- Fine-tune the clustering model.
  - Some of the nearest neighbors may not belong to the same cluster.
  - But highly confident predictions tend to be classified to the proper cluster.
  - Filter the confident images whose soft assignment is above the threshold. 
  - For the confident images, fine-tune the clustering model by minimizing the cross entropy loss.

## 4. Experiment & Result

In this section, we summarize the experimental results of this paper.

### Experimental setup

- **Dataset**: CIFAR10, CIFAR100-20, STL10, ImageNet
- **Backbone**: RestNet-18
- **Pretext task**: SimCLR and MoCo
- **Baselines**: DeepCluster, IIC, GAN, DAC, etc. 
- **Evaluation metric**: Accuracy, NMI, and ARI

### Result

Here are the results of SCAN.

#### Comparison with SOTA
SCAN outperforms the prior work by large margins on ACC, NMI, and ARI.

<img src="/.gitbook/assets/2022spring/3/sota.PNG" width="800" align="center">

#### Qualitative results
The obtained clusters are semantically meaningful.

<img src="/.gitbook/assets/2022spring/3/qualitative.png" width="800" align="center">

#### Ablation study: Pretext tasks
SCAN selects a pretext task that minimizes the distance between an image and its augmentations.
- RotNet does not minimize the distances.
- Instance discrimination tasks satisfy the invaraince criterion.

<img src="/.gitbook/assets/2022spring/3/pretext.PNG" width="400" align="center">

#### Ablation study: Self-labeling
Fine-tuning the network through self-labeling enhances the quality of clusters.

<img src="/.gitbook/assets/2022spring/3/self_labeling.PNG" width="400" align="center">

## 5. Conclusion
- SCAN is a two-step algorithm for unsupervised image classification.
- SCAN adopts nearest neighbors to be semantically similar. 
- SCAN outperforms the SOTA methods in unsupervised iamge classification.

### Take home message 
> Nearest neighbors are likely to be semantically similar.
>
> Filtering confident images and using them for supervision enhances the performance.



## Author / Reviewer information

### Author

**이건 (Geon Lee)** 

* KAIST AI
* geonlee0325@kaist.ac.kr

### Reviewer
TBD

## Reference & Additional materials
- Van Gansbeke, Wouter, et al. "Scan: Learning to classify images without labels." European Conference on Computer Vision. Springer, Cham, 2020.
- Slides: https://wvangansbeke.github.io/pdfs/unsupervised_classification.pdf
- Codes: https://github.com/wvangansbeke/Unsupervised-Classification

