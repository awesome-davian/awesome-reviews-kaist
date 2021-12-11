---
description: Chen et al. / Pre-Trained Image Processing Transformer / CVPR 2021
---

# IPT \[Kor\]

## 1. Problem definition
이미지 처리(Image processing)는 보다 글로벌한 이미지 분석 또는 컴퓨터 비전 시스템의 low-level 부분 중 하나입니다.  
이미지 처리의 결과는 이미지 데이터의 인식 및 이해를 수행하는 후속 상위 레벨 부분에 크게 영향을 미칠 수 있습니다.   
최근 딥러닝은 GPU를 활용한 하드웨어 컴퓨팅 성능이 강력하게 증가했고 Pre-Trained Deep Learning Model과 대규모 데이터셋을 통하여 기존 방법보다 좋은 결과를 보여주었습니다.   
또한, 이러한 딥러닝 기술은 이미지 초고해상도(super-resolution), 인페인팅(inpainting), 디레인(deraining), 채색(colorization)과 같은 낮은 수준의 비전 작업을 해결하기 위해 널리 적용되고 있습니다.  
하지만 Pre-Training을 통해 여러 이미지 처리 Task들을 일반화한 연구는 거의 없습니다.  

본 논문에서는 Pre-Trained Deep Learning Model인 IPT(image processing transformer)를 통해 노이즈 제거, 초고해상도 및 디레이닝와 같은 low-level 컴퓨터 비전 Task에 대해 일반화하고 현 state-of-the-art 이상의 결과(성능)를 보여줍니다.   
또한 많은 양의 손상된 이미지 쌍을 실험에 사용하기 위해 잘 알려진 ImageNet 벤치마크를 활용합니다.  

## 2. Motivation
### A. Related work
#### 1. Image processing
이미지 처리는 super-resolution(해상도를 높이는 작업), denoising(노이즈 제거), dehazing(연무, 안개 등 대기 중의 미세입자 노이즈 제거) , deraining(비내리는듯한 노이즈 제거), debluring(블러 제거 작업) 등을 포함한 이미지 조작으로 구성됩니다.  

- (Dong et al.) 은 초고해상도를 위해 SRCNN을 제안하였습니다. Low Resolution(저해상도) 이미지에서 High Resolution(고해상도) 이미지를 재구성하는 end-to-end 모델을 도입한 선구적인 연구입니다.
- (Kim et al.) 은 위의 연구에서 더 깊은 컨볼루션 네트워크를 사용하여 심층 신경망의 크기을 키웠습니다.
- (Ahn et al. & Lim et al.) 은 SR(super-resolution) Task에 Residual block 개념을 추가하였습니다.
- (Zhang et al. & Anwar & Barnes) 는 attention의 강력한 성능을 SR Task에 활용하였습니다.

이외에도 다른 Task들에 대한 연구도 많이 있습니다.
- (Tian et al. 이하 5개 논문)에서는 노이즈 제거와 관련된 Denoising에 대해 연구했습니다.
- (Cai et al. 이하 4개 논문)에서는 dehazing에 대해 연구했습니다.
- (Hu et al. 이하 6개 논문)에서는 deraining에 대해 연구했습니다.
- (Tao et al. 이하 4개 논문)에서는 debluring에 대해 연구했습니다.

##### Idea 1. 위의 연구들에서는 개별적인 다른 방법을 사용하여 연구했지만, 이 논문에서는 하나의 큰 모델(pre-trained)과 대용량의 데이터를 사용하여 여러 이미지 처리 Task들에 대해서 실험하고 좋은 결과를 보여주었습니다.  

#### 2. Transformer 
- (Vaswani et al.) Transfomer는 다양한 자연어 처리 작업에서 강력한 unsupervised 또는 self-supervised pretraining framework로 성공을 입증했습니다.
- (Radford et al.) GPTs는 거대한 텍스트 데이터 세트에서 다음 단어를 예측하는 자기회귀 방식으로 사전 훈련됩니다.
- (Devlin et al.) BERT는 명시적인 감독 없이 데이터에서 학습하고 컨텍스트를 기반으로 마스킹 단어를 예측합니다.
- (Colin et al.)는 여러 Downstream Task에 대한 보편적인 Pre-training Framework를 제안합니다.

NLP 분야에서 Transformer 기반 모델의 성공으로 인해 컴퓨터 비전 분야에서도 Transformer 기반 모델을 활용하려는 연구가 있습니다.  
- (Yuan et al.)에서는 이미지 분할을 위한 spatial attention을 소개합니다.
- (Fu et al.)는 spatial attention과 channel attention을 결합하여 context 정보를 활용한 DANET을 제안했습니다.
- (Kolesnikov et al.)은 Transformer 블록으로 이미지 분류를 수행합니다.(convolutional neural network를 self‑attention block으로 대체)
- (Wu et al. &  Zhao et al.)은 이미지 인식 작업을 위한 Transformer 기반 모델에 대한 사전 학습 방법을 제안합니다.
- (Jiang et al.)은 Transformer를 사용하여 이미지를 생성하기 위해 TransGAN을 제안합니다.

##### Idea 2. 이미지 처리에 대한 연구와 Transformer를 컴퓨터 비전 분야에 활용하는 연구들을 많이 있었지만, Transformer와 같은 Pre-Training모델을 활용하여 이미지 처리와 같이 low-level vision tasks에 초점을 맞춘 관련 연구는 거의 없습니다. 따라서 이 논문에서는 이미지 처리 작업에 대한 보편적인 Pre-Training 접근 방식을 탐구합니다.  


## 3. Method
### A. Image Processing Transformer (IPT)
IPT의 전체 아키텍처는 4가지 구성 요소로 구성됩니다. (**_Heads - Incoder - Decoder - Tails_**)  
손상된 Input Image(노이즈가 있는 이미지 및 저해상도 이미지)에서 Feature을 추출하기 위한 **_Head_**  
Input Data에서 소실된 정보를 복구하기 위한 **_인코더 - 디코더 Transformer_**  
디코더에서 나온 representation들을 적절하게 이미지로 복원하는 **_Tails_**  
![image](/.gitbook/assets/42/1.png)  

#### 1. Heads
다른 이미지 처리 Task을 조정하기 위해 다중 헤드 아키텍처를 사용하여 각 Task를 개별적으로 처리합니다.   
각 Head는 3개의 컨볼루션 레이어로 구성됩니다.  입력 이미지를 다음과 같이 표시합니다.  
$$ x ∈ R^{3×H×W} $$ (3 means R, G, and B) , 헤드는 C(보통 64)개의 채널을 가진 feature map $$ f_{H} ∈ R^{C×H×W} $$ 을 생성합니다.  
공식화하자면 $$ f_{H} = H^{i}(x) $$ 이며, 여기서 $$ H^{i} (i = {1, ... , N_{t}}) $$ 는 i번째 Task의 헤드, $$ N_{i} $$ 는 task의 수로 나타냅니다.  

#### 2. Transformer encoder
Input  features를 Transformer body에 적용시키기 전에 features를 "word"처럼 간주 될 수 있도록 **패치(Patch)**로 분할됩니다.  
구체적으로 feature map $$ f_{H} ∈ R^{C×H×W} $$ 에서 아래의 식과 같이 패치들의 sequence로 재구성됩니다.   
$$ f_{p^{i}} ∈ R^{P^{2}×C} , i = {1, . . . , N} $$ 여기서 $$ N = HW/P^{2} $$ 는 패치의 갯수(sequence의 길이)이며 P는 패치 사이즈입니다.  
각 패치의 위치 정보를 유지하기 위해 Feature $$ f_{p^{i}} $$ 의 각 패치에 대한 $$ E_{p^{i}} ∈ R^{P^{2}×C} $$ 로 학습 가능한 위치 인코딩을 추가합니다. 이후,  $$ E_{p^{i}} + f_{p^{i}} $$ 는 Transformer encoder의 입력 값이 됩니다.  
Encoder layer에는 original Transformer 구조와 같이 multihead self-attention module 과 a feed forward network로 구성되어있습니다. 역시 Encoder의 Input과 Output은 같은 사이즈이며 다음과 같이 공식을 계산할 수 있습니다.  
![image](/.gitbook/assets/42/2.png)  
여기서, l 은 인코더의 레이어 갯수이며, MSA는 Multi-head Self-Attention module, LN은 Layer Normalization, FFN은 두개의 Fully Connected Layers를 포함한 Feed Forward Network를 나타냅니다.  

#### 3. Transformer decoder
디코더 또한 기존 Transformer와 동일한 아키텍처를 따르며, 2개의 MSA 레이어와 1개의 FFN 레이어로 구성됩니다. 한가지 차이점이 있다면, Task별 임베딩을 디코더의 Input으로 추가 활용한다는 것입니다. Task별 임베딩의 경우 $$ E^{i}_{t} ∈ R^{P^{2}×C} , i = {1, ... , N_{t}} $$ 으로 나타내며, 각각 다른 Task 별로 feature를 decode 합니다.   
디코더의 경우 다음과 같이 공식을 계산할 수 있습니다.  
![image](/.gitbook/assets/42/3.png)  
여기서, $$ F_{D_{i}} ∈R^{P^{2}×C} $$ 는 디코더의 outputs이고, decode된 $$ P^{2}×C $$ size의 N개의 패치 feature의 경우 $$ C × H × W $$ size를 갖는 $$ f_{D} $$ feature로 재구성 됩니다.  

#### 4. Tails
Tails의 경우 Heads의 속성과 동일하며 multi tails를 사용하여 각각 다른 Task별로 처리합니다. 다음과 같이 공식화 할 수 있습니다.  
$$ f_{T} = T^{i}(f_{D}) $$ 여기서 $$ T^{i} (i = {1, ... , N_{t}}) $$ 는 i번째 Task의 Head를 나타내며, $$ N_{t} $$ 는 task의 갯수입니다.  
output $$ f_{t} $$ 는 특정 task에 의해 결정된 $$ 3 × H' × W' $$ 이미지 사이즈가 됩니다.   
예를 들어, $$ H' = 2H, W' = 2H $$ 라면 2배 확대한 super-resolution task(고해상도 작업)이 될 수 있습니다.  

### B. Pre-training on ImageNet
Transformer 자체의 아키텍처 외에도 성공적인 학습의 핵심 요소 중 하나는 대규모 데이터 세트를 잘 활용해야 합니다.  
또한, 학습을 위해서는 정상 이미지와 손상된 이미지가 사용되므로 이에 맞는 데이터 세트가 필요합니다.  
ImageNet 벤치마크의 이미지는 질감 및 색상이 풍부한 100만 개 이상의 nature 이미지가 포함되어있고 1000개 이상의 다양한 카테고리를 가지고 있습니다. 따라서 레이블을 제거하고 다양한 Task에 맞게 사용될 수 있도록 이미지를 저하 모델을 사용하여 수동으로 다음 공식과 같이 손상시켜 데이터 세트를 준비할 수 있습니다.  $$ I_{corrupted} = f(I_{clean}) $$ 여기서, f 는 저하(손상) 변환이라 할 수 있으며 Task에 따라 달라집니다.  
지도 방식으로 IPT를 학습하기 위한 손실 함수는 다음과 같이 공식화할 수 있습니다.  
$$ L_{supervised} = sum _{i=1} ^{N_{t}} L1(IPT(I_{corrupted}^{i}), I_{clean}) $$
여기서 L1은 기존 L1 손실을 나타내고 프레임워크가 여러 이미지 처리 작업으로 동시에 훈련되었음을 의미합니다.  
 IPT 모델을 pre-training한 후에는 다양한 이미지 처리 task에 대한 고유한 feature과 변환을 캡처(weight를 저장)하므로 새로 제공된 데이터 세트를 사용하여 원하는 작업에 적용하도록 더욱 Fine-tuning할 수 있습니다. 이때, 계산 비용을 절약하기 위해 다른 Heads와 Tails는 삭제되고 남은 Heads와 Tails 및 Transformer body의 매개변수는 역전파에 따라 업데이트 됩니다.  

다양한 데이터 품질 저하 모델이 있고 모든 이미지 처리 task에 적용시킬 수 없기에 IPT의 일반화 성능이 더욱 좋아야 합니다.  
NLP에서의 Word 처럼 Patch끼리의 관계도 중요하기에 동일한 feature map에서 잘린 patch는 유사한 위치에 포함되어야합니다.  
대조학습(contrastive learning)을 통해 보편적인 features를 학습하여 unseen tasks에 대해서도 IPT모델이 활용될 수 있도록 했습니다.   
같은 이미지의 패치 feature 사이의 거리를 최소화하며 다른 이미지의 패치 feature 사이의 거리는 최대화하도록 하였습니다.  
대조학습의 Loss Function은 다음과 같습니다.  
![image](/.gitbook/assets/42/4.png)  
또한, supervised 및 self-supervised 정보를 완전히 활용하기 위해 IPT의 최종 목적 함수를 다음과 같이 공식화 할 수 있습니다.   
![image](/.gitbook/assets/42/5.png)  


## 4. Experiment & Result
### A. Experimental Setup
#### 1. DataSet
1백만 개 이상의 컬러 이미지 ImageNet 데이터 세트를 사용하며 3채널 48X48 패치들로 crop됩니다. (1천만 개 이상의 패치)  
손상된 데이터는 6가지(2배, 3배, 4배 bicubic interpolation, 30, 50 level 가우시안 노이즈, rain streaks(비 내리는 노이즈))로 생성합니다.  
공정한 비교를 위해 CNN 기반 모델에도 동일한 테스트 전략이 적용되었으며 CNN 모델의 결과 PSNR 값은 기준선의 값과 동일합니다.  

#### 2. Training & Fine-tuning.
NVIDIA V100 32장을 사용하여 Adam optimizer β1 = 0.9, β2 = 0.999로 300에폭 수정된 ImageNet dataset을 학습합니다.  
Learning rate는 $$ 5e^{-5} $$ 부터 $$ 2e^{-5} $$ 까지 256 배치 크기로 200 에폭 동안 줄어듭니다.  
훈련 세트는 서로 다른 작업으로 구성되어 있어 단일 배치에 메모리 한계로 모든 input을 태울 수 없습니다.  
따라서 각 반복에서 무작위로 선택된 작업의 이미지 배치를 쌓습니다.   
IPT Model을 pre-training 한 이후 원하는 task(e.g., 3배 super-resolution)를 $$ 2e^{-5} $$ learning rate로 30 에폭 동안 학습합니다.   
SRCNN 방식 또한 ImageNet 학습방식을 사용하면 super-resolution task의 성능이 개선됨을 보여줬습니다.  

### B. Result

초해상도와 영상 잡음 제거를 포함한 다양한 image processing tasks 에서 pre-trained된 IPT의 성능은 state-of-the-art를 능가했습니다.   
#### 1. Super-resolution
IPT Model을 몇몇의 state-of-the-art CNN-based SR 방식과 비교했고 Table 1에서와 같이 모든 데이터셋에서 ×2, ×3, ×4 scale 성능이 가장 좋았고 ×2 scale Urban100 dataset에서 33.76dB PSNR을 달성함을 강조했습니다. 이전 모델들이 이전 SOTA보다 <0.2dB 씩 개선되었었지만 이번 모델은 ~0.4dB이나 개선되어 대규모 pre-trained Model의 우수성을 나타냈습니다.  

#### 2. Denoising
학습 및 테스트 데이터는 깨끗한 이미지에서 σ = 30, 50 level의 가우스 잡음을 추가하여 생성되었고 SOTA Model과 비교했습니다.  
Table 2는 BSD68 및 Urban100 데이터 세트에 대한 컬러 이미지 노이즈 제거 결과이며, IPT 모델이 다양한 가우스 노이즈 레벨에서 최상의 성능을 보여줍니다. Urban100 데이터셋에서는 ∼2dB 성능 향상을 보여주고, Pre-training 방식, Transformer 기반 모델의 우수성을 나타내었습니다.  
![image](/.gitbook/assets/42/6.png)

기존 방식으로는 노이즈 이미지에서 깨끗한 이미지로의 복구가 어려웠고 충분한 디테일을 재구성하지 못해 비정상적인 픽셀을 생성했습니다. IPT의 경우 머리카락의 몇 가지 디테일까지 잘 복구하며 시각적인 품질이 이전 모델을 능가했습니다.  
![image](/.gitbook/assets/42/7.png)
![image](/.gitbook/assets/42/8.png)

#### 3. Generalization Ability
다양한 손상된 이미지 생성은 가능해도, 자연적인 이미지는 복잡도가 높고 transformer의 pre-training을 위해 모든 이미지 데이터셋을 합성(생성)할 수 없는 한계가 있습니다. 따라서 IPT 모델이 Vision task를 넘어 NLP분야에서까지 여러 task를 잘 처리할 수 있는 능력이 있어야 합니다. 이러한 일반화 능력을 검증하고자 ImageNet 이외에 손상된 이미지(노이즈 10 & 70 level)의 노이즈 제거 테스트를 진행했습니다.  
IPT 모델은 CNN 및 다른 모델보다 좋은 성능을 보여주었습니다.  
![image](/.gitbook/assets/42/9.png)

#### 4. Impact of data percentage
데이터 백분율이 Transformer 및 CNN 모델의 pre-training 성능에 어떠한 영향을 주는지 실험합니다.  
ImageNet 데이터 세트의 20%, 40%, 60%, 80% 및 100%을 사용하여 Figure 6과 같이 결과를 확인하였습니다.  
모델이 pre-training하지 않거나 소량 학습되는 경우 CNN 모델이 더 좋은 성능을 보여주지만, 대규모 데이터에선 transformer 기반 pre-trained 모델(IPT)이 성능을 압도합니다.  

#### 5. Impact of contrastive learning
Pre-trained model의 성능을 개선시키고자 ×2 scale super-resolution task에서 Set4 데이터셋을 활용해 λ 매개변수를 실험합니다.  
λ=0 에서보다 λ = 0.1 에서 0.1dB 높은 38.37dB PSNR 값이 나왔고 최적의 λ 매개변수 값을 찾았습니다.  
![image](/.gitbook/assets/42/10.png)

## 5. Conclusion
이 논문에서는 NLP 분야에서 그리고 컴퓨터 비전 분야까지 발전되고 있는 Transformer 기반 Pre-training 기법을 사용하여 IPT모델을 개발하고 다양한 이미지 처리 문제에서 최신 SOTA 이상의 성능을 보여주었습니다. 원본 이미지와 손상된 이미지 데이터 쌍을 통해 IPT 모델을 사전 학습하여 각 이미지 처리 task에 따라 신속하게 미세 조정할 수 있도록 합니다. 따라서 하나의 모델로도 다양한 Task에 적용할 수 있고 일반화 될 수 있는 능력을 입증했습니다. 특히 대규모 데이터셋에서 압도적인 성능을 보여주었고 데이터의 비례하여 성능이 높아질 것이라고 판단됩니다.  

### A. Take home message (오늘의 교훈)
1. 이미지 처리 Task에서도 대규모 데이터셋을 활용한 Transformer 기반 모델의 Pre-training & Fine-tuning 기법은 성능이 아주 효과적이였습니다. 또한 데이터의 양이 많으면 많을수록 비례하여 성능은 좋아집니다.  
2. NLP의 Word와 같이 이미지 input 데이터를 Patch로 변환하여 Transformer 기반의 모델을 사용할 수 있습니다.  
3. IPT 모델을 사전 학습한 후 각 Task에 맞는 고유 Feature들과 변환을 캡쳐하여 Fine-tuning 시 원하는 Task에 맞게 필요없는 매개변수는 삭제하여 비용적인 측면에서도 유리해보였습니다.  

## Author / Reviewer information

### Author

**박준형 (Junhyung Park)**

- Affiliation (KAIST AI / NAVER)
- Machine Learning Engineer @ NAVER Shopping AI Team

### Reviewer

1. Korean name (English name): Affiliation / Contact information
2. Korean name (English name): Affiliation / Contact information
3. …


## Reference & Additional materials

1. [Chen, H., Wang, Y., Guo, T., Xu, C., Deng, Y., Liu, Z., ... & Gao, W. (2021). Pre-trained image processing transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 12299-12310).](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Pre-Trained_Image_Processing_Transformer_CVPR_2021_paper.html)
2. [Official GitHub repository](https://github.com/huawei-noah/Pretrained-IPT)
3. [Official Gitee repository](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/IPT)
