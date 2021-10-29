---
description: Yulun Zhang et al. / Image Super-Resolution Using Very Deep Residual Channel Attention Networks / ECCV 2018
---

# Image Super Resolution via RCAN \[Eng\]

한국어로 쓰인 리뷰를 읽으려면 [**여기**](/paper-review/2021-fall-paper-review/eccv-2018-rcan-kor.md)를 누르세요.

##  1. Problem definition

<p align="center"><img src = "/.gitbook/assets/63/0srex.PNG" height = "300"></center>

The Single Image Super-Resolution (SISR) technique aims to restore a low resolution (LR) image to a high resolution (HR) while removing blur and various noises in the image. SR is expressed as an equation as follows, where x and y are LR and HR images, respectively.

<p align="center"><img src = "/.gitbook/assets/63/eqn1.PNG" height = "27"></center>

Recently, CNN-based SR has been actively studied, since CNN works effectively on SR, However, CNN-based SR has the following two limitations.

* Gradient Vanishing [Note i] occurs as the layer deepens, making learning more difficult

* The representativeness of each feature map is weakened as low-frequency information included in the LR image is treated equally in all channels.

To overcome the aforementioned goals of SR and the above two limitations, this paper proposes Deep-RCAN (Residual Channel Attention Networks).

> [Note i] **Gradient Vanishing**: As the input value goes through the activation function, it is squeezed into a small range of output values, so it means the state that the initial input value has little effect on the output value as it goes through the activation functions of several layers. Accordingly, the rate of change of the parameter values of the initial layers with respect to the output becomes small, making learning impossible.

## 2. Motivation

### **2.1. Related work**

The papers related to deep-CNN and attention technique, which are the baselines of this paper, are as follows.

#### **1. CNN 기반 SR**

* **[SRCNN & FSRCNN]**: SRCNN, the first technique applying CNN to SR, significantly improved performance compared to existing Non-CNN based SR techniques by constructing a 3-layer CNN. FSRCNN simplifies the network structure of SRCNN to increase inference and learning speed.
* **[VDSR & DRCN]**: By stacking layers deeper than SRCNN (20 layers), the performance is greatly improved.
* **[SRResNet & SRGAN]**: SRResNet was the first to introduce ResNet to SR. In SRGAN, photo-realistic SR was implemented by mitigating blur by introducing GAN to SRResNet. However, there are cases where an unintentional artifact object is created.
* **[EDSR & MDSR]**: By removing unnecessary modules from the existing ResNet, the speed is greatly increased. However, it cannot implement the deep layer, which is the key in image processing, and has limitations in that it includes unnecessary calculations and does not represent various features by treating low-frequency information equally in all channels.

#### **2. Attention 기법**

Attention은 인풋 데이터에서 관심 있는 특정 부분에 처리 리소스를 편향시키는 기법으로서, 해당 부분에 대한 처리 성능을 증가시킨다. 현재까지 attention은 객체인식이나 이미지 분류 등 high-level vision task에 일반적으로 사용되었고, 이미지 SR 등의 low-level vision task에서는 거의 다루어지지 않았다. 본 논문에서는 고해상도(High-Resolution, HR) 이미지를 구성하는 고주파(High-Frequency)를 강화하기 위해, LR 이미지에서 고주파 영역에 attention을 적용한다.

### **2.2. Idea**
해당 논문의 idea와 이에 따른 contribution은 아래 세가지로 요약할 수 있다.

#### **1. Residual Channel Attention Network (RCAN)**

Residual Channel Attention Network (RCAN) 을 통해 기존의 CNN 기반 SR보다 더욱 층을 깊게 쌓음으로써, 더 정확한 SR 이미지를 획득한다.

#### **2. Residual in Residual (RIR)**

Residual in Residual (RIR)을 통해 i) 학습가능한(trainable) 더욱 깊은 층을 쌓으며, ii) RIR 블록 내부의 long and short skip connection으로 저해상도 이미지의 low-frequency 정보를 우회시킴으로써 더 효율적인 신경망을 설계할 수 있다.

#### **3. Channel Attention (CA)**

Channel Attention (CA)을 통해 Feature 채널 간 상호종속성을 고려함으로써, 적응식 feature rescaling을 가능케 한다.


## 3. Residual Channel Attention Network (RCAN)
### **3.1. Network Architecture**

<p align="center"><img src = "/.gitbook/assets/63/1Modelarchitecture.PNG" height = "280"></center>

RCAN의 네트워크 구조는 크게 4 부분으로 구성되어 있다: i) Shallow feature extraction, ii) RIR deep feature extraction, iii) Upscale module, iv) Reconstruction part. 본 논문에서는 i), iii), iv)에 대해서는 기존 기법인 EDSR과 유사하게 각각 one convolutional layer, deconvolutional layer, L1 loss가 사용되었다. ii) RIR deep feature extraction을 포함하여, CA와 RCAB에 대한 contribution은 다음 절에서 소개한다.

### **3.2. Residual in Residual (RIR)**
RIR에서는 residual group (RG)과 long skip connection (LSC)으로 구성된 G개의 블록으로 이루어져 있다. 특히, 1개의 RG는 residual channel attention block(RCAB)와 short skip connection (SSC)을 단위로 하는 B개의 연산으로 구성되어 있다. 이러한 구조로 400개 이상의 CNN 층을 형성하는 것이 가능하다. RG만을 깊게 쌓는 것은 성능 측면에서 한계가 있기 때문에 LSC를 RIR 마지막 부에 도입하여 신경망을 안정화시킨다. 또한 LSC와 SSC를 함께 도입함으로써 LR이미지의 불필요한 저주파 정보를 더욱 효율적으로 우회시킬 수 있다.

### **3.3. Residual Channel Attention Block (RCAB) [Work in progress]**

<p align="center"><img src = "/.gitbook/assets/63/2channelattention.PNG" height = "150"</center>

본 논문에서는 Channel Attention (CA)를 Residual Block (RB)에 병합시킴으로써, Residual Channel Attention Block (RCAB)를 제안하였다. 특히, CNN이 local receptive field만 고려함으로써 local region 이외의 전체적인 정보를 이용하지 못한다는 점을 극복하기 위해 CA에서는 global average pooling으로 공간적 정보를 표현하였다.

<p align="center"><img src = "/.gitbook/assets/63/4RCAB.PNG" height = "150"></center>
  
한편, 채널간 연관성을 나타내기 위해, gating 매커니즘을 추가로 도입하였다. gating 매커니즘은 일반적으로 채널간 비선형성을 나타내야 하며, one-hot 활성화에 비해 다수 채널의 feature가 강조되면서 상호 배타적인 관계를 학습해야 한다. 이러한 기준을 충족하기 위해, sigmoid gating과 ReLU가 선정되었다.

## 4. Experiment & Result
### **4.1. Experimental setup**
#### **1. Datasets and degradation models**

<p align="center"><img src = "/.gitbook/assets/63/7dataset.PNG" height = "350"></center>

학습용 이미지는 DIV2K 데이터셋의 일부 800개 이미지를 이용하였으며, 테스트 이미지로는 Set5, B100, Urban 100과 Manga109를 사용하였다. Degradation 모델로는 bicubic (BI)와 blur-downscale (BD)가 사용되었다.

#### **2. Evaluation metrics**
PSNR과 SSIM으로 처리된 이미지의 YCbCr color space의 Y 채널을 평가하였음. 또한 recognition error에서 1~5위의 타 SR 기법과 비교하여, 성능 우위를 확인하였음.

#### **3. Training settings**
앞서 언급한 DIV2K 데이터셋에 있는 800개의 이미지에 회전, 상하반전 등 data augmentation을 적용하고, 각 training batch에서는 48x48 사이즈의 16개의 LR 패치가 인풋으로 추출되었다. 또한 최적화 기법으로는 ADAM이 사용되었다.

### **4.2. Result [Work in progress]**
### **1. Effects of RIR and CA**

<p align="center"><img src = "/.gitbook/assets/63/5result.PNG" height = "150"></center>

기존기법이 37.45dB의 성능을 보여준데 반해, LSC과 SSC가 포함된 RIR과 CA를 이용함으로써, 37.90dB까지 성능을 높였다.

### **2. Model Size Analyses**

<p align="center"><img src = "/.gitbook/assets/63/6result2.PNG" height = "220"></center>

RCAN은 타 기법들 (DRCN, FSRCNN, PSyCo, ENet-E)과 비교하여 가장 깊은 신경망을 이루면서도, 전체 파라미터 수는 가장 적지만, 가장 높은 성능을 보여주었다.

## 5. Conclusion
본 논문에서는 높은 정확도의 SR 이미지를 획득하기 위해 RCAN이 적용되었다. 특히, RIR 구조와 LSC 및 SSC를 함께 활용함으로써, 깊은 층을 형성할 수 있었다. 또한 RIR은 LR 이미지의 불필요한 정보인 저주파 정보를 우회시킴으로써, 신경망이 고주파 정보를 학습할 수 있도록 하였다. 더 나아가, CA를 도입하여 채널간의 상호종속성을 고려함으로써 channel-wise feature를 적응식으로 rescaling하였다. 제안한 기법은 BI, DB degradation 모델을 이용하여 SR 성능을 검증하였으며, 추가로 객체 인식에서도 우수한 성능을 나타내는 것을 확인하였다.

## Take home message \(오늘의 교훈\)
> 이미지 내에서 관심 있는 영역의 정보를 분할해내고, 해당 정보에 attention을 적용함으로써 학습과정에서 비중을 더 높일 수 있다.

> 전체 파마리터 개수를 늘리는 것보다 신경망을 더 깊게 쌓는 것이 성능을 높이는데 더 효과적이다.

## Author / Reviewer information
### 1. Author

**한승호 \(Seungho Han\)** 
* KAIST ME
* Research Topics: Formation Control, Vehicle Autonomous Driving, Image Super Resolution
* https://www.linkedin.com/in/seung-ho-han-8a54a4205/

### 2. Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. **[Original Paper]** Zhang, Yulun, et al. "Image super-resolution using very deep residual channel attention networks." Proceedings of the European conference on computer vision (ECCV). 2018.
2. **[Github]** https://github.com/yulunzhang/RCAN
3. **[Github]** https://github.com/dongheehand/RCAN-tf
4. **[Github]** https://github.com/yjn870/RCAN-pytorch
5. **[Attention]** https://wikidocs.net/22893
6. **[Dataset]** Xu, Qianxiong, and Yu Zheng. "A Survey of Image Super Resolution Based on CNN." Cloud Computing, Smart Grid and Innovative Frontiers in Telecommunications. Springer, Cham, 2019. 184-199.
7. **[BSRGAN]** Zhang, Kai, et al. "Designing a practical degradation model for deep blind image super-resolution." arXiv preprint arXiv:2103.14006 (2021).
8. **[Google's SR3]** https://80.lv/articles/google-s-new-approach-to-image-super-resolution/

