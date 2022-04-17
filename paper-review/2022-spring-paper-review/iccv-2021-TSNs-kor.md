---
description: Sun, Guolei, et al. / Task Switching Network for Multi-Task Learning / ICCV 2021

---

# TSNs [Kor]

[**English version**](iccv-2021-SML-eng.md) of this article is available.

Sun, Guolei, et al. "Task Switching Network for Multi-Task Learning." *Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2021. ([paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Sun_Task_Switching_Network_for_Multi-Task_Learning_ICCV_2021_paper.pdf))

<br/>

"Task Switching Network for Multi-Task Learning" 논문의 리뷰에 앞서서, multi-task learning 이 무엇인지 간략히 소개하고,  기존 연구들이 취한 접근법을 알아보겠습니다.

<br/>

## 1. Problem definition

Multi-task learning은 머신러닝의 하위 분야 중 하나로, 여러 task가 공유하는 모델에서 각 task를 동시에 학습시키는 학습 패러다임을 말합니다.  전통적으로, MTL (Multi-Task Learning)은 사전에 학습시킬 task set을 정하고 이를 동시에 학습시키며, 이 때 각 task들은 동등하게 취급됩니다. 즉, multi-task learning 상황에서는 하나의 "main task"와 다른 보조 task들을 정의하거나, 학습을 진행하는 task set이 시간에 따라 변화하는 상황을 가정하지 않습니다. 이 점은 transfer learning과 multi-task learning을 구별하는 중요한 차이점이기도 합니다.

우리가 Multi-task learning을 수행함으로써 얻을 수 있는 이점은 다음과 같이 정리할 수 있습니다.

* Data efficiency (memory usage, computation burden)
* Reduced over-fitting through shared representations
* Fast learning by leveraging auxiliary information

Multi-task learning을 수행할 때는 각 task 사이의 정보 교환을 이용할 수 있기 때문에, 각 task를 분리된 single task network에서 학습시킬 때보다 작은 크기의 network를 이용할 수 있습니다. 이는 memory usage, computation burden 차원에서 유리하다고 볼 수 있습니다. 또, 서로 다른 task 사이의 효과적인 정보 교환을 통하여 shared representation을 여러 task가 공유하는 architecture가 배울 수 있도록 하고, 이는 network의 over-fitting을 줄일 수 있습니다. 마지막으로 task 사이의 이런 유용한 정보 교환은 auxiliary information으로 이용되어 network의 빠른 학습에 도움을 줄 수 있습니다.

Multi-task learning에서 우리가 풀어야할 문제들을 요약하면 다음과 같습니다.

* 서로 다른 task들이 서로 충돌하는 needs를 가지고 있을 수 있습니다. 즉 특정 task의 성능을 향상시키는 것이 다른 task의 성능 저하를 발생시킬 수 있다는 것입니다. (이를 negative transfer 또는 destructive interference라고 부릅니다.)
* 따라서, negative transfer을 최소화하는 것이 multi-task learning의 중요 목표라고 할 수 있습니다.

Multi-task learning은 각 task가 각자의 needs가 있는 상황에서, 이러한 서로 다른 task를 동시에 학습시키는 패러다임입니다. 만약 optimization 과정에서 서로 다른 task가 같은 방향으로 나아간다면 (positive transfer) 학습에 긍정적인 영향을 미칠 것입니다. 하지만 정반대의 상황에서는, task 사이의 negative transfer을 최소화하기 위해서 task-specific feature를 담는 space를 network 안에 따로 구성하거나, attention mechanism이 이용되기도 합니다. 하지만 task 사이의 information을 분리하는 것은 여전히 어려운 문제로 남아 있습니다.

Multi-task learning의 문제들을 풀기위한 필요조건은 다음과 같습니다.

* 새로운 architecture design, optimization method
* 함께 학습시킬 task 집합 구성

Shared architecture에서 서로 다른 task 사이의 정보 교환을 촉진하기 위해서는, 적절한 architecture 설계와 optimization method가 필요합니다. 또, 함께 학습시켰을 때, 높은 성능을 보여줄 수 있는, 서로 관련되어 있는 task들을 찾아내야 합니다.

<br/>

## 3. Motivation

### Related work

Multi-task learning을 위한 기존 접근법을 아래에 간략히 정리하였습니다.

* Encoder-based methods - 여러 task가 공유하는 정보와 task-specific 정보를 효과적으로 담아내기 위하여 encoder의  encoding 능력 향상에 집중 [1, 2, 3]
* Decoder-based methods - Encoder가 뽑아낸 여러 feature를 각 task에 맞게 refine하는데 집중 [4, 5]
* Optimization-based methods - Task 사이의 간섭이나 negative transfer을 다루기 위하여 각 task의 loss의 가중치를 조절하거나 task 학습 순서를 재배치 [6, 7]

Multi-task learning을 위한 architecture에도 여러 종류가 있습니다. 간단히 정리하면

* Shared trunk - 모든 task가 공유하는 convolutional layer로 이루어진 global feature extractor와 각 task 별로 분리된 output branch가 존재 [1, 8, 9]
* Cross talk - 각 task 별로 분리된 network가 존재하며 같은 깊이의 layer 사이의 수평적인 정보교환이 일어남 [2]
* Prediction distillation - 각 task 별로 미리 prediction을 수행한 후, 이를 조합하여 개선된 최종 결과를 만듦 [4, 10]
* Conditional architecture - Architecture의 일부가 network의 input과 task에 따라 조건부로 선택됨 [11. 12]

<br/>

 좀 더 자세한 정보를 알고 싶으신 분은, 글의 끝에 있는 reference를 참고하시기 바랍니다.

<br/>

### Idea

Multi-task learning에 대한 기존의 접근법들은 어떤 식으로든 각 task에 대한 task-specific 모듈을 갖추어야할 필요가 있었습니다. 반면 해당 논문은 모든 task에 대하여 network가 최대한 많은 부분을 공유하는 것을 목표로 합니다. 구체적으로 모든 network의 parameter가 task에 상관없이 공유되고 task의 수 T에 관계없이 network가 일정한 크기를 유지하는 것을 목표로 합니다.

<br/>

## 4. Method

##### Task Switching networks

![Task_Switching_Network_Overview](.gitbook/assets/41/Task_Switching_Network_Overview.png)

위의 그림에서 task switching network의 전체적인 모습을 볼 수 있습니다. 제안된 네트워크는 task에 따라 조건부로 바뀌는 decoder를 이용함으로써 multi-tasking을 수행합니다. U-Net 구조를 base로 이용하고 있고, encoder는 이미지 $I_n$ 을 input으로 받아 여러 layer에서 feature $F_i$를 추출합니다. 제안된 network는 두번째 input으로, 수행하려는 task $\tau$에 따라 만든 task encoding vector $v_\tau$를 input으로 받습니다. 그러면 task embedding network $C$가 각 task에 대하여 latent embedding $l_\tau$를 생성합니다.  이 latent embedding vector는 그림의 파란 path를 따라서 모듈 $A$ 에 input으로 이용되어 decoder에 영향을 미칩니다. Decoder의 각 layer의 output은 encoder에서 뽑은 feature  $F_i$와 decoder의 feature $O_i$를 bottom-up 방식으로 합쳐 계산됩니다.

Key points

* 모든 task가 decoder 전체를 공유
* 각 task 별 task-specific 모듈로 인한 추가적인 parameter가 없음 
* Fully connected layers로 이루어진 task embedding network
* StyleGAN에서 이용된 것과 비슷한 방식의 모듈을 이용하여 decoder에 영향

<br/>

모듈 A는 각 task 별 embedding vector $l_\tau$를 이용하여, input으로 들어온 feature를 각 새로운 feature로 변환시킵니다. Layer j에서 decoder가 내놓는 output을 $O_j$라고 하면, 다음과 같이 표현할 수 있습니다.

$$O_j = \begin{cases} A([u(O_{j+1}),A(F_j,l_\tau)],l_\tau),\;\;\;\;\;for\;\;j\leq4, \\ A(F_j,l_\tau),\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;for\;\;j=5\end{cases}$$

여기서 $[\cdot,\cdot]$는 두 feature tensor의 channel dimension으로의 concatenation을 의미하고 $u(\cdot)$는 upsampling 연산을 의미합니다.

<br/>

##### Conditional Convolution Module

제안된 모듈(block $A$)의 역할은, 모든 task가 공유하는 encoder에서 뽑아내는 feature를 각 task에 맞는 새로운 feature로 전환하는 것이라고 할 수 있습니다. Module $A$ 가 수행하는 내용을 정리하면 다음과 같습니다.

1. Module로의 input feature $x\in R^{1\times c_1\times h\times w}$가  $W$의 filter weight를 가지는 convolution layer를 통과 $\hat{x}=x*W$  하여  $\hat{x}\in R^{1\times c_2\times h\times w}$를 생성합니다.

2. 동시에, $l_\tau$가 weight matrices $W_\gamma \in R^{d \times c_2}$ 과 $W_\beta \in R^{d \times c_2}$를 가지는 두 fully connected layers를 통하여 normalization coefficients $\gamma \in R^{1 \times c_2}$ 와 $\beta \in R^{1 \times c_2}$를 생성하고, 이는 이어지는 AdaIN에 이용됩니다.

3. Feature $\hat{x}$에 대해서, AdaIN은 다음과 같은 normalization을 수행합니다. $AdaIN(\hat{x},\beta,\gamma) = \gamma\frac{(\hat{x}-\mu)}{\sqrt{\sigma^2}} + \beta$ 

   이때,   $\beta$ 와 $\sigma^2$ 는  $\hat{x}$의 mean과 variance를 의미합니다.

   요약하면, module $A$는 다음과 같은 연산을 수행합니다. $A(x,l_\tau) = l_\tau W_\gamma\frac{(x*W-\mu)}{\sqrt{\sigma^2}} + l_\tau W_\beta$ 

<br/>

##### Task embedding network

각 task는 task 별로 유일한 task-condition vector $v_\tau$를 가지고,  TSNs은 이 $v_\tau$를 task embedding network $C$에 넣어 task 간의 변환에 이용합니다. Embedding network $C:R^d\rarr R^d$는 task $\tau$의 latent space $l_\tau=C(v_\tau)$로의 matching을 학습하고, 이는 각 모듈 $A$로 부터 AdaIN의 coefficients 생성에 이용됩니다.

Task-condition vector의 initialization을 위해서, orthogonal $v_\tau$ (binary vector)와 Gaussian random vector가 각각 실험되었습니다.

<br/>

## 5. Experiment

실험에서는

PASCAL-context 데이터셋 (Edge detection, semantic segmentation, human parts segmentation, surface normals, and saliency detection)과 NYUD 데이터셋 (Edge detection, semantic segmentation, surface normals, and depth estimation) 이 이용되었습니다.

![Task_Switching_Performance](.gitbook/assets/41/Task_Switching_Performance.png)

위의 table이 보여주듯, TSNs은 task 별로 single task architecture를 구성하거나, 다른 multi-tasking 방법을 수행하는 것보다 상당히 작은 model 크기를 사용한다는 것을 알 수 있습니다. 또 각 task를 따로 학습하는 것보다 task embedding을 이용하여 normalization coefficient를 함께 학습하는 것이 더 높은 performance를 보이는 것을 확인할 수 있습니다. 그리고 제안된 방법은 task-specific INs 와 BNs 이용할 때보다 더 빠른 수렴 속도를 보였습니다.

<br/>

![Impact_of_task_embedding_strategy](.gitbook/assets/41/Impact_of_task_embedding_strategy.png)

위의 table은 2가지 종류의 task-condition vector $v_\tau$의 선택에 따른 network의 성능이 정리되어 있습니다. Orthogonal encoding의 경우에는 embedding dimensionality d에 크게 상관없이 좋은 성능을 보이는 것을 확인할 수 있었으나 그중에서 d=100일 때 가장 좋은 performance를 보였습니다. Gaussian encoding의 경우에는 100 아래에서 orthogonal encoding과 비슷한 성능을 보였습니다.

<br/>

![Model_Parameter_Scaling](.gitbook/assets/41/Model_Parameter_Scaling.png)

위의 그림은 task의 수에 따라 model의 parameter가 어떻게 변화하는지 보여줍니다. 제안된 방법(TSNs)은 이용한 task의 수에 관계없이 일정한 수의 parameter를 가지는 가집니다. 이에 반해서, 기존 방법(RCM, Multi-decoder 등)의 경우에는 task의 수에 따라, model의 parameter 수가 비례하여 증가하는 것을 확인할 수 있습니다.

<br/>

![Qualitative_results](.gitbook/assets/41/Qualitative_results.png)

위 그림에 baseline(Task-specific INs)와 논문에서 제안된 model의 정성적 결과를 확인할 수 있습니다. 제안된 방법은 baseline에 비해서 high-level task인 semantic segmentation, parts, 그리고 saliency detection에서 좋은 성능을 보이는 것을 확인할 수 있습니다.

<br/>

## 6. Conclusion

* Multi-task learning에 단일 encoder와 decoder를 적용한 첫번째 시도
* Task switching Network를 사용하여 simplicity와 parameter efficiency 차원에서 큰 이점
* 훨씬 적은 parameter를 사용하여 기존 state-of-the-art multi-decoder 접근법보다 좋은 성능을 보임

<br/>

## 7. Author / Reviewer information

##### 정우성 (WooSeong Jeong)

* KAIST ME M.S.

* Research Area: Computer Vision

* stk14570@kaist.ac.kr

  <br/>

## 8. Reference

- Sun, Guolei, et al. "Task Switching Network for Multi-Task Learning." *Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2021. ([paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Sun_Task_Switching_Network_for_Multi-Task_Learning_ICCV_2021_paper.pdf))

- Ruder, Sebastian. "An overview of multi-task learning in deep neural networks." *arXiv preprint arXiv:1706.05098* (2017). ([paper](https://arxiv.org/abs/2009.09796))
- Karras, Tero, Samuli Laine, and Timo Aila. "A style-based generator architecture for generative adversarial networks." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2019. ([paper](https://arxiv.org/abs/1812.04948))

1. Iasonas Kokkinos. Ubernet: Training a universal convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory. In CVPR, 2017.
2. I. Misra, Abhinav Shrivastava, A. Gupta, and M. Hebert. Cross-stitch networks for multi-task learning. CVPR, 2016.
3. Y. Lu, Abhishek Kumar, Shuangfei Zhai, Yu Cheng, T. Javidi, and R. Feris. Fully-adaptive feature sharing in multitask networks with applications in person attribute classification. CVPR, 2017.
4. D. Xu, Wanli Ouyang, X. Wang, and N. Sebe. Pad-net: Multi-tasks guided prediction-and-distillation network for simultaneous depth estimation and scene parsing. CVPR, 2018.
5. Simon Vandenhende, S. Georgoulis, and L. Gool. Mti-net: Multi-scale task interaction networks for multi-task learning. In ECCV, 2020.
6. Z. Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient ormalization for adaptive loss balancing in deep multitask networks. arXiv, 2018.
7. Alex Kendall, Yarin Gal, and R. Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. CVPR, 2018.
8. Felix J. S. Bragman, Ryutaro Tanno, S´ebastien Ourselin, D. Alexander, and M. Cardoso. Stochastic filter groups for multi-task cnns: Learning specialist and generalist convolution kernels. ICCV, 2019.
9. Y. Lu, Abhishek Kumar, Shuangfei Zhai, Yu Cheng, T. Javidi, and R. Feris. Fully-adaptive feature sharing in multitask networks with applications in person attribute classification. CVPR, 2017.
10. Z. Zhang, Zhen Cui, Chunyan Xu, Zequn Jie, Xiang Li, and Jian Yang. Joint task-recursive learning for semantic segmentation and depth estimation. In ECCV, 2018.
11. Menelaos Kanakis, David Bruggemann, Suman Saha, Stamatios Georgoulis, Anton Obukhov, and Luc Van Gool. Reparameterizing convolutions for incremental multi-task learning without task interference. ECCV, 2020.
12. Kevis-Kokitsi Maninis, Ilija Radosavovic, and Iasonas Kokkinos. Attentive single-tasking of multiple tasks. In CVPR, 2019

