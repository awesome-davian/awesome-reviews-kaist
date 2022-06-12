---
description: Shin et al. / MM-TTA / CVPR 2022
---

# MM-TTA: Multi-Modal Test-Time Adaptation for 3D Semantic Segmentation \[Eng\]


### Title & Description

_MM-TTA: Multi-Modal Test-Time Adaptation for 3D Semantic Segmentation \[Eng\]_

_&lt;Shin et al.&gt; / &lt;MM-TTA: Multi-Modal Test-Time Adaptation for 3D Semantic Segmentation&gt; / &lt;CVPR 2022&gt;_


한국어로 쓰인 리뷰를 읽으려면 [**여기**](cvpr-2022-mmtta-kor.md)를 누르세요.

##  1. Problem definition

Domain adaptation is the task that adapts the model trained with the source data to be suitable for the target data.

Since the source data is not always available the test-time adaptation approach has recently emerged as a solution.

The previous methods that target unimodal semantic segmentation cannot be directly adapted for the multi-modal task.

This paper proposes a method to take full advantage of multi-modality.

## 2. Motivation

### Related work

Test-time adaptation aims to enable adaptation without a source dataset that is used to train the pre-trained model. 
Test-time training is the approach that updates the model parameters with the proxy task that requires training samples, and the optimal proxy task that is hard to find.
TENT the first test-time adaptation method that updates batch norm parameters without a proxy task is a simple yet effective technique. However, TENT the entropy-minimization-based method tends to encourage the model to increase the confidence with the false prediction.
S4t selective self-training that regularizes pseudo labels is only considered for the specific task that spatial augmentation can be performed.

3D semantic segmentation has been recognized as an important 3D scene understanding task aimed at classifying each LiDAR point into semantic categories.
The unimodal approaches such as range-based methods that project a 3D point onto the 2D image plane, voxelizing the point cloud and utilizing SparseConvNet lack 2D contextual information that is essential to understanding the complex semantics of the scene.

### Idea

To address the weakness, the approaches that use multi-modal inputs for 3D segmentation are explored with the fusion techniques between the two modality features, RGB and point cloud that contains contextual and geometric information respectively. However, since each modality has different dataset biases(style distribution in 2D and the point distribution in 3D), multi-modality models are harder to adapt to new data. This paper tackles multi-modal 3D semantic segmentation in the test-time adaptation setting with the method that helps the two modality models jointly learn.

## 3. Method

Intra-modal pseudo label generation
They proposed Intra-PG to generate reliable online pseudo labels within each modality by having two models updated at a different pace. The fast model directly updates the batch normalization statistics and the slow model is slowly updated with the momentum update scheme from the fast model(equation 6). The models are updated aggressively and gradually provide a stable and complementary supervisory signal. The only slow updated model is used during the inference time.  They take an average of the logits from the two models to fuse their prediction.

![](../../.gitbook/assets/2022spring/5/update.png)

Inter-modal pseudo label refinement
They proposed the Inter-PR module to improve pseudo labels via cross-modal fusion. The consistency between the two different pace models is introduced to select the output of which modality to be the pseudo label. There are two variants, hard select and soft select. Hard select use the modality that has the higher consistency between the slow and fast model, and the soft select use the weighted sum of the output of the two modalities. The consistency is measured by the inverse of KL divergence. The pseudo labels of which the maximum consistency measure over the two modalities is below a threshold are ignored. The objective to use the generated pseudo label for updating batch norm statistics is 
as below.

![](../../.gitbook/assets/2022spring/5/loss.png)

![](../../.gitbook/assets/2022spring/5/main.png)

###### *Q. inter-PR 에서 hard selection 시, 두 모델 사이 consistency 가 높은 modality 를 취한다고 하는데, 두 모델은 fast model 와 slow model 인가요? 그래서 각 modality 별로 fast 와 slow model 의 consistency 를 구하고, 더 consistent 한 modality 를 선택하는건가요? 그렇다면, 두 모델의 consistency 를 기준으로 잡은 이유가 궁금합니다. 두 모델이 consistent 하지 않다면, 불안정한 모델로 여기는 건가요?*

###### *A. 본 논문에서 fast, slow model의 consistency를 고려하는 것은 source data 접근이 불가능한 TTA setting을 보완하기 위함이라 생각합니다. UDA의 경우 Source data에 대해서도 꾸준히 학습하여 모델이 task의 전체적인 구조 보다 test set에 대한 loss만 줄이는 방향으로 빠지는 것을 막을 수 있습니다. 그러나 TTA에서는 그것이 불가능하기 때문에 Source data의 학습된 모델의 prediction 으로부터 크게 벗어나지 않는 선에서 test data에 adapt 하는 방식으로 학습한 것으로 보입니다.*


## 4. Experiment & Result

### Experimental setup

#### Dataset
For the A2D2-to-SemanticKITTI setting, A2D2 consists of a 2.3 MegaPixels camera and 16-channel LiDAR. SemanticKITTI uses a 0.7 MegaPixels camera and 64-channel LiDAR. The nuScenes Day-to-Night is used for the real-world case. The images captured during the day and night are obviously different while the LiDAR is almost invariant to lighting conditions. Synthia-to-SemanticKITTI is conducted to evaluate test-time adaptation between synthetic and real data.

#### Baselines
Self-learning with Entropy is originally proposed by TENT. They optimize the model by minimizing the entropy of model predictions. Only the fast model is used in this setting. This objective only encourages sharp output distributions, which may reinforce wrong predictions, and may not lead to cross-modal consistency. 

###### *Q. 두 modality 간의 consistency 를 계산하지 못한다고 나와 있는데, 따로 penalty 를 주지 않아도 cross-modal consistency 가 보존되는 건가요? 아니면, 본 연구는 두 modality 중 더 consistent 한 modality 를 선택하기 때문에, cross-modal consistency 는 중요하지 않은건가요?*

###### *A. 두 modality 간의 consistency를 제대로 측정하지 못하는 이유 역시 source data에 접근을 할 수 없기 때문입니다. 특히 두 modality의 prediction이 동일한 오답일 경우가 좋은 예라고 할 수 있습니다. Consistent 함에도 불구하고 그 prediction에 대해서는 penalize 하지 않는 것이지요. 따라서 본 논문에서는 두 modality의 prediction에서 consensus를 고려하기 보다는 1번 질문의 답변과 같이 더 consistent한 model의 output을 pseudo-label로 하여 두 modality가 같은 prediction을 하도록 합니다.*

###### *Q. Baseline 모델에서 entropy, consistency, pseudo label 을 이용한 self-learning 모델이 각각 TENT, xMUDA, MM-TTA 라고 이해하면 될까요? 각 category 에 해당하는 baseline 모델이 무엇인지 헷갈립니다*.

###### *A. TENT의 경우 entropy를 고려한 방법이고, xMUDA가 consistency를 고려한 방법인 것은 맞습니다만 xMUDA에서도 pseudo-label을 사용한 setting이 있습니다. Cross-modal consistency에 추가적으로 각 modality 내에서 pseudo-label로 self-training을 하는 것이지요. 본 논문에서 제안하는 방법인 MM-TTA의 핵심은 두 modality간의 interaction을 통한 pseudo-label generation이라고 할 수 있습니다.*



![](../../.gitbook/assets/2022spring/5/entropy.png)

Self-learning with Consistency aims to achieve multi-modal test-time adaptation via a consistency loss between predictions of 2D and 3D modalities with KL divergence. Different from the scenarios where the source data is accessible as xMUDA, MM-TTA is not regularized by the source data. Therefore, it may fail to capture the correct consistency when one of the branches provides a wrong prediction.

![](../../.gitbook/assets/2022spring/5/consistency.png)

Self-learning with pseudo-labels optimizes the segmentation loss and the pseudo-labels provide supervisory signals. The pseudo-labels are obtained by thresholding the prediction as eq4. Since only the batch norm statistics are updated and the model still lacks information exchange between the modality to refine the pseudo-labels, it is sub-optimal.


![](../../.gitbook/assets/2022spring/5/pseudo_label.png)

#### Training Setup

They follow xMUDA the two-stream multi-modal framework. U-Net with a ResNet34 encoder is adopted for the 2D branch and the U-Net that utilizes sparse convolution on the voxelized point cloud input is used for the 3D branch. They use either SparseConvNet or MinkowskiNet.

They directly borrow the xMUDA official pre-trained model for fair comparison for the SparseConvNet. Mincowskinet was trained from scratch on source data.

TTA only optimizes for batch norm affine parameters during training and then reports performance after 1 epoch of adaptation. 

#### Evaluation metric
They use the metric ‘mIoU’ to evaluate their approach. mIoU is the common evaluation metric for the semantic segmentation task. To calculate the mIoU, the confusion matrix is required. The confusion matrix is obtained by counting how many the category pairs are. Category pair here means the pair of categories of ground truth and the prediction. There are #class*#class combinations of the category pairs. The diagonal elements on the matrix are considered intersections and the numbers on the cross on the diagonal elements are counted as a union. By taking an average of the IoU of every combination, we can get mIoU.

### Result

For UDA, they compare with the xMUDA framework that utilizes consistency loss and self-training using offline pseudo-labels. For TTA baselines, they evaluate TENT, xMUDA, xMUDA_pl.  They extend TENT to multiple modalities with entropy minimization on the ensemble of 2d and 3d logits. They evaluated the combination of them as well.

MM-TTA method performs favorably against all the TTA baselines in the three benchmark settings. Entropy and pseudo-labeling-based methods perform better than the consistency loss, due to the difficulty of capturing the correct consistency across modalities on TTA baselines on A2D2-to-SemantiKITTI and Synthia-to-SemanticKITTI. Even though some TTA baselines improve the performance of individual 2D and 3D predictions, the ensemble results are all worse than the source-only model. This is because the method is not designed to jointly learn between the multimodal outputs.

The domain gap is larger for RGB than LiDAR in the nuScences Day-to-Night there by the challenge mainly lies in how to improve the 2D branch and obtain effective ensemble results. Inter-PR contributes to this point. It demonstrates their effectiveness.

![](../../.gitbook/assets/2022spring/5/qualitative.png)
![](../../.gitbook/assets/2022spring/5/quantitative.png)

$xMUDA$ : Consistency between the two modalities

$xMUDA_{PL}$ : Consistency between the two modalities + intra pseudo-label

$TENT$ : Self-training with entropy.

$TENT_{ENS}$ : Self-training with entropy. Entropy minimization on the ensemble of the logits from the two modalities.

$MM-TTA$ : The self-training with the pseudo-label generated with the interaction between the two modalities


## 5. Conclusion

In this paper, they proposed a new problem setting of test-time adaptation on the multi-modal 3D semantic segmentation. Instead of adopting the method that has limitations, they suggested a novel method to refine the pseudo label intra, and inter the modality. Since the method didn’t analyze the task-specific characteristics deeply, there is still room to improve and the method can be adapted for other tasks that deal with the multi-modal supervisory signals.

###### *Q. 실제 시나리오에 필요한 unseen data에 대한 practical한 방법으로 test time adaptation관련 연구가 진행되고 있고, 특히 multi modal 상황에서 다양한 센서 입력을 활용하는 방법으로 보입니다. 다양한 입력 센서 데이터의 fusion시, 입력 주기가 다르고 sync가 맞지 않는 부분들은 구체적으로 어떻게 처리되었는지 궁금합니다.*

###### *A. 본 논문은 두 modality에서의 representation을 fuse하지 않고 각각이 prediction을 하고 높은 confident를 갖는 modality의 prediction을 취하는 방식이며 real time에서 실제 sync를 맞추는 것보다는 두 modality간의 interaction에 더 무게를 둔 연구입니다. 다양한 센서의 입력의 sync를 맞추는 방법에 대한 연구는 향후 연구 방향으로 아주 좋은 주제인 것 같습니다.*


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

1. Inkyu Shin, Yi-Hsuan Tsai, Bingbing Zhuang, Samuel Schulter, Buyu Liu, Sparsh Garg, In So Kweon, Kuk-Jin Yoon. MM-TTA: Multi-Modal Test-Time Adaptation for 3D Semantic Segmentation. In CVPR, 2022.
2. Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Ol- shausen, and Trevor Darrell. Tent: Fully test-time adaptation by entropy minimization. In ICLR, 2021.
3. MaximilianJaritz,Tuan-HungVu,RaouldeCharette,E ́milie Wirbel, and Patrick Pe ́rez. xmuda: Cross-modal unsupervised domain adaptation for 3d semantic segmentation. In CVPR, 2020.
4. Christopher Choy, JunYoung Gwak, and Silvio Savarese. 4d spatio-temporal convnets: Minkowski convolutional neural networks. In CVPR, 2019.
5. Benjamin Graham, Martin Engelcke, and Laurens Van Der Maaten. 3d semantic segmentation with submanifold sparse convolutional networks. In CVPR, 2018.