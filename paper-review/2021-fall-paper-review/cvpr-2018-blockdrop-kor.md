---
description: Wu et al. / BlockDrop - Dynamic Inference Paths in Residual Networks / CVPR 2018
---

# BlockDrop: Dynamic Inference Paths in Residual Networks [Korean]


## 1. Problem definition

ResNet은 두 개 이상의 컨볼루션 레이어로 구성된 Residual Block과 두 Residual Block 사이의 직접 경로를 가능하게 하는 Skip-connection으로 구성되어 있습니다. 이러한 Skip-connection은 ResNet을 상대적으로 얕은 네트워크의 앙상블처럼 작동하도록 하여, ResNet의 특정 Residual Block이 제거되더라도 일반적으로 전체 성능에 약간의 부정적인 영향만 미치게 됩니다.


## 2. Motivation

Residual Network의 layer를 drop하는 것은 일반적으로 Dropout과 DropConnect와 같이 모델을 training하는 과정에서 이루어집니다. 이러한 방법들은 모두 test 과정에서는 layer를 drop하지 않고 고정시킨 채로 실험을 진행합니다. 만약 test 과정에서 layer를 효율적으로 drop한다면 성능은 거의 유지한 채로 inference 과정에서 speed up을 기대할 수 있습니다.

### Related work

#### Residual networks behave like ensembles of relatively shallow networks

해당 논문에서는 ResNet이 test time에서 layer dropping에 resilient하다는 것을 보였습니다. 그러나 성능 저하는 최소화하면서 layer를 drop할 수 있는 dynamic한 방법은 제시하지 않았습니다.

#### Data-driven sparse structure selection for deep neural networks

해당 논문에서는 sparsity constraint를 활용하여 어떤 residual blocks를 drop시킬 것인지 결정하는 방법을 제안하였습니다. 그러나 input image에 dependent하게, 다시 말해 instance-specific하게 어떤 block을 drop 시킬 것인지 결정하는 방법을 제안하지는 못하였습니다.


### Idea

이 논문은 최적의 block dropping 구조를 찾기 위해 reinforcement learning을 활용하여 성능 저하가 거의 없는 상태로 speed up을 inference time에서 이뤄냅니다.

## 3. Method

입력 이미지가 주어졌을 때 최적의 block dropping 전략을 찾기 위해서 binary policy vector를 출력하는 policy network를 구성합니다. Training 과정에서 reward는 block usage와 prediction accuracy를 모두 고려하여 결정됩니다.


## 4. Experiment & Result

### Experimental Setup

CIFAR-10, CIFAR-100의 경우 pretrained resnet은 resnet-32와 resnet-110으로 실험이 진행되었으며, ImageNet의 경우 pretrained resnet은 resnet-101으로 실험이 진행되었습니다. Policy Network의 경우 CIFAR에 대해서는 resnet-8을 사용하였고 ImageNet에 대해서는 resnet-10을 사용하였는데 ImageNet에서는 input image를 112x112로 downsampling하여 policy network에 전달하였습니다.

### Result

해당 논문은 임의로 residual block을 drop시킨 random 방법과 순서상 앞에 있는 residual block을 drop 시킨 first 방법 등을 baseline으로 하고 본 논문에서 제안하는 BlockDrop 방법의 성능을 비교하였습니다. CIFAR-10에서 ResNet-32를 pretrained backbone으로 하는 경우 Full ResNet의 성능(accuracy)이 92.3이었다면 FirstK는 16.6의 성능을 보였고 RandomK는 20.5의 성능을 보였으며 BlockDrop은 88.6의 성능을 보였습니다.

## 5. Conclusion

본 논문은 ResNet을 활용할 때 더 빠른 속도로 inference할 수 있도록 Residual Block을 instance specific하게 drop하는 BlockDrop을 제안하였고 CIFAR 및 ImageNet에 대한 광범위한 실험을 수행하여 efficiency-accuracy trade-off에서 상당한 이점이 있음을 관찰하였습니다.


### Take home message (오늘의 교훈)

> 이 논문은 inference 속도 향상을 위해 instance specific하게 residual block을 drop 시키며, 성능 하락을 방지하기 위해 reinforcement learning을 활용하였습니다.

## Author / Reviewer information

### Author

**이현수 (Hyunsu Rhee)**

- KAIST
- ryanrhee@kaist.ac.kr

### Reviewer

1. Korean name (English name): Affiliation / Contact information
2. Korean name (English name): Affiliation / Contact information
3. …

## Reference & Additional materials

1. Z. Wu, T. Nagarajan, A. Kumar, S. Rennie, L. S. Davis, K. Grauman, and R. Feris. Blockdrop: Dynamic inference paths in residual networks. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

