---
description: Inkyu Shin / MM-TTA / CVPR 2022
---

# MM-TTA: Multi-Modal Test-Time Adaptation for 3D Semantic Segmentation \[Kor\]


### Title & Description

_MM-TTA: Multi-Modal Test-Time Adaptation for 3D Semantic Segmentation \[Eng\]_

_&lt;Shin et al.&gt; / &lt;MM-TTA: Multi-Modal Test-Time Adaptation for 3D Semantic Segmentation&gt; / &lt;CVPR 2022&gt;_


\(한국어 리뷰에서\) ---&gt; [**English version**](cvpr-2022-mmtta-eng.md) of this article is available.

##  1. Problem definition

Domain adaptation는 source data에서 train된 모델이 target data에 적합하도록 모델을 적응 시키는 task입니다.

Source data가 항상 접근 가능한 것이 아니기에 Test-time adaptation이 시도되고 있습니다.

Uni-modal semantic segmentation에서 사용하는 방법들은 multi-modal에 그대로 적용할 수는 없습니다.

이 논문에서는 multi-modality task의 장점을 최대한 이용할 수 있는 방법을 제안합니다.

## 2. Motivation

### Related work

Test-time adaptation은 source data 없이 domain adaptation을 수행하는 기법입니다. Test-time training은 proxy task를 통해 model parameter를 업데이트 합니다. 그러나 training sample을 필요로 하고, 최적의 proxy task를 찾는 것은 어렵습니다. TENT는 proxy task없이 batch norm parameter를 업데이트 하는 첫 번째 방법으로, 간단하고 효과적인 방법입니다. 그러나 TENT는 entropy를 최소화하는 방식으로 이루어지기에 잘못된 prediction에 대한 confidence를 높이는 경향이 있습니다. S4T는 pseudo label을 regularize하는 방법으로 이루어지는데, spatial augmentation이 가능한 task에 한해서만 적용할 수 있다는 한계가 있습니다.

3D semantic segmentation은 3D scene에 대한 이해를 통해 각 LiDAR point를 분류하는 방법을 연구하는 중요한 task로 알려져 있습니다. 3D point들을 2D image plane에 정사영하거나 point cloud를 voxelize, 혹은 SparseConvNet을 활용하는 방법들은 2D 문맥 정보를 사용하지 않는데, 복잡한 의미론적 정보를 이해하는데에는 이 2D 문맥 정보가 매우 중요합니다.

### Idea

이런 단점들을 해결하기 위해서 multi-modal 3D segmentation에 대한 연구가 이루어지고 있습니다. Multi-modal semantic segmentation에서는 RGB와 point cloud의 두가지 정보를 잘 융합하는 기법이 중요한데, RGB는 문맥적 정보를, point cloud는 기하학적 정보를 가지고 있습니다. 2D data에는 style distribution, 3D data에는 point distribution의 dataset bias가 존재하는데, 이 때문에 multi-modality model의 domain adaptation이 더 까다롭습니다. 이 논문에서는 test-time adaptation 환경에서 multi-modal 3D semantic segmentation의 두 modality model이 jointly learn하는 방법에 대해 연구하였습니다.


## 3. Method

Intra-modal pseudo label generation
이 논문에서는 Intra-PG라는 모듈을 제안하였는데, 각각의 modality에서 신뢰할 수 있는 online pseudo label을 만드는 역할을 합니다. 다른 속도로 업데이트 되는 두개의 다른 모델을 활용하는 방법으로, Fast model은 batch normalization 통계들을 바로 업데이트 하고 Slow model은 fast model로부터 momentum update됩니다(식 6). 두 모델은 공격적으로, 점진적으로 stable하고 상보적인 supervisory signal을 줍니다. Inference time에는 Slow model만 사용됩니다. 두 모델은 logit의 평균을 통해 fusion됩니다.

![](../../.gitbook/assets/2022spring/5/update.png)

Inter-modal pseudo label refinement
이 논문에서는 Inter-PR이라는 모듈을 제안하였는데, Cross-modal fusion을 통해 pseudo label을 발전시키는 방식입니다. 다른 속도로 업데이트 되는 두 모델의 consistency를 이용해 어떤 modality의 output을 pseudo label로 취할 것인지 정합니다. Modality를 고르는 방법에는 hard와 soft selection 방법이 있는데 harder selection은 두 모델 사이 consistency가 높은 modality를 그대로 취하는 것이고 soft selection은 두 모델의 output의 weighted sum을 통해 pseudo label을 구합니다. Consistency는 KL Divergence의 역수를 통해 측정합니다. 두 modality의 consistency가 일정 threshold보다 낮은 경우 해당 pseudo label은 무시합니다. Loss 함수는 아래와 같습니다.

![](../../.gitbook/assets/2022spring/5/loss.png)

![](../../.gitbook/assets/2022spring/5/main.png)

## 4. Experiment & Result

### Experimental setup

#### Dataset
For the A2D2-to-SemanticKITTI setting, A2D2 consists of a 2.3 MegaPixels camera and 16-channel LiDAR. SemanticKITTI uses a 0.7 MegaPixels camera and 64-channel LiDAR. The nuScenes Day-to-Night is used for the real-world case. The images captured during the day and night are obviously different while the LiDAR is almost invariant to lighting conditions. Synthia-to-SemanticKITTI is conducted to evaluate test-time adaptation between synthetic and real data.

#### Baselines
Self-learning with Entropy is originally proposed by TENT. They optimize the model by minimizing the entropy of model predictions. Only the fast model is used in this setting. This objective only encourages sharp output distributions, which may reinforce wrong predictions, and may not lead to cross-modal consistency. 

![](../../.gitbook/assets/2022spring/5/entropy.png){: width="10%" height="10%"}

Self-learning with Consistency aims to achieve multi-modal test-time adaptation via a consistency loss between predictions of 2D and 3D modalities with KL divergence. Different from the scenarios where the source data is accessible as xMUDA, MM-TTA is not regularized by the source data. Therefore, it may fail to capture the correct consistency when one of the branches provides a wrong prediction.

![](../../.gitbook/assets/2022spring/5/consistency.png)

Self-learning with pseudo-labels optimizes the segmentation loss and the pseudo-labels provide supervisory signals. The pseudo-labels are obtained by thresholding the prediction as eq4. Since only the batch norm statistics are updated and the model still lacks information exchange between the modality to refine the pseudo-labels, it is sub-optimal.


![](../../.gitbook/assets/2022spring/5/pseudo_label.png)

#### Training Setup

They follow xMUDA the two-stream multi-modal framework. U-Net with a ResNet34 encoder is adopted for the 2D branch and the U-Net that utilizes sparse convolution on the voxelized point cloud input is used for the 3D branch. They use either sparseconvnet or minkowskinet.

They directly borrow the xMUDA official pre-trained model for fair comparison for the sparseconvnet. Mincowskinet was trained from scratch on source data.

TTA only optimizes for batch norm affine parameters during training and then reports performance after 1 epoch of adaptation. 

#### Evaluation metric
They use the metric ‘mIoU’ to evaluate their approach. mIoU is the common evaluation metric for the semantic segmentation task. To calculate the mIoU, the confusion matrix is required. The confusion matrix is obtained by counting how many the category pairs are. Category pair here means the pair of categories of ground truth and the prediction. There are #class*#class combinations of the category pairs. The diagonal elements on the matrix are considered intersections and the numbers on the cross on the diagonal elements are counted as a union. By taking an average of the IoU of every combination, we can get mIoU.

### Result
Please summarize and interpret the experimental result in this subsection.

For UDA, they compare with the xMUDA framework that utilizes consistency loss and self-training using offline pseudo-labels. For TTA baselines, they evaluate TENT, xMUDA, xMUDApl.  They extend TENT to multiple modalities with entropy minimization on the ensemble of 2d and 3d logits. They evaluated the combination of them as well.

MM-TTA method performs favorably against all the TTA baselines in the three benchmark settings. Entropy and pseudo-labeling-based methods perform better than the consistency loss, due to the difficulty of capturing the correct consistency across modalities on TTA baselines on A2D2-to-SemantiKITTI and Synthia-to-SemanticKITTI. Even though some TTA baselines improve the performance of individual 2D and 3D predictions, the ensemble results are all worse than the source-only model. This is because the method is not designed to jointly learn between the multimodal outputs.

The domain gap is larger for RGB than LiDAR in the nuScences Day-to-Night there by the challenge mainly lies in how to improve the 2D branch and obtain effective ensemble results. Inter-PR contributes to this point. It demonstrates their effectiveness.

![](../../.gitbook/assets/2022spring/5/qualitative.png)
![](../../.gitbook/assets/2022spring/5/quantitative.png)

## 5. Conclusion

In this paper, they proposed a new problem setting of test-time adaptation on the multi-modal 3D semantic segmentation. Instead of adopting the method that has limitations, they suggested a novel method to refine the pseudo label intra, and inter the modality. Since the method didn’t analyze the domain-specific characteristics deeply, there is still room to improve and the method can be adapted for other tasks that deal with the multi-modal supervisory signals.

### Take home message \(오늘의 교훈\)

 > Test-time adaptation is the emerging task that is practical for the real-world scenario. 
 
 > Starting from this work, the community can improve the performance by adding the module that is aiming to refine task or modality-specific features.
 
 > Others can bring this framework to their field as well.
 
 > As an early work for the test-time adaptation, this work is suggesting a large insight into all the machine learning communities.
 
## Author / Reviewer information

### Author

** 류형곤 \(Hyeonggon Ryu\)** 

* Affiliation \(KAIST\)
* Contact information \(gonhy.ryu@kaist.ac.kr)

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. Citation of this paper
2. Citation of related work
