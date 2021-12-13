---
description: Jianrui Cai / Toward Real-World Single Image Super-Resolution; A New Benchmark and A New Model / ICCV 2019 Oral
---

# Toward Real-World Single Image Super-Resolution: A New Benchmark and A New Model \[Kor\]

##  1. Problem definition

 Single image super-resolution (SISR)의 정의는 한 장의 저해상도 이미지로부터 고해상도 이미지를 복원하는 task이다. 여기서 해상도란 이미지가 나타낼 수 있는 섬세함 정도를 의미하고, 따라서 해상도가 낮을 수록 섬세한, 즉, 작거나 edge와 같은 부분을 잘 표현하지 못한다는 의미가 된다.

 우리가 흔히 핸드폰이나 카메라로 찍어서 볼 수 있는 이미지는 이미징 하고자 하는 대상으로부터 반사된 빛들이 렌즈를 통과하여 이미지 센서에 의해 기록된 정보이다. 이미지가 기록되는 과정에서 많은 빛의 정보를 잃게 되는데, 특히 렌즈가 빛을 잘 모으지 못하거나 이미지 센서의 픽셀 사이즈가 충분히 작지 못할 경우 문제가 발생한다. 예를 들어, 아래 그림을 보자. 
 
 <p align="center">
 <img src="/.gitbook/assets/iccv-2019-lp-kpn-kor/image-resolution.png"/>
 </p>
같은 원에 대한 이미지여도, 왼쪽 이미지는 원의 형태가 뭉개져 마름모처럼 보이지만 우측으로 갈 수록 제대로 된 원의 형태를 볼 수 있다. 섬세하게 원을 표현하지 못한 첫 번째 이미지가 저해상도 이미지가 되고, 섬세하게 원을 잘 표현한 오른쪽 이미지가 고해상도 이미지가 된다. 즉, SISR은 왼쪽 이미지 한 장에서 오른쪽 이미지로 변환해주는 task이다. 

 이를 좀 더 분석해보면, $$3 \times N \times N$$ 저해상도 이미지를 $$3 \times M \times M$$ 고해상도 이미지로 복원하려면 $$\frac{3 \times M \times M}{3 \times N \times N} = \frac{M^2}{N^2}$$ 만큼의 정보를 정보를 저해상도 이미지로부터 유추해내야 한다. 따라서, SISR은 ill-posed problem에 속하며 이를 해결하기 위한 많은 연구들이 있었다.

## 2. Motivation

 SISR에 딥러닝을 적용하는 연구는 2014년 [“Image Super-Resolution Using Deep Convolutional Networks, 2014 ECCV”](https://arxiv.org/pdf/1501.00092.pdf)[1] 이라는 연구에서 처음으로 Convolution neural network를 적용하면서 시작되었다. 이후에 다른 컴퓨전 비전 분야와 마찬가지로 GAN[2], Residual Dense Network[3] 등 새로운 framework를 적용해가며 고해상도 이미지 복에 대한 성능이 꾸준히 향상되었다. SISR에 대한 state-of-the-art를 기록한 논문들은 다음 사이트를 참고하기 바란다([SISR SOTA](https://paperswithcode.com/task/image-super-resolution)).
 
 하지만 이전 연구들에서는 단순한 이미지 degradation을 가정한 simulation 데이터셋만을 활용하여 모델을 train하고 evaluate했다는 공통적인 한계가 존재한다. 여기서 이미지 degradation이란 고해상도 이미지가 저해상도 이미지로 변환되는 과정이다. 앞서 말했던, 이미지가 기록되면서 빛의 정보를 잃게되는 과정이라고 생각을 하면 편하다. 예시로, bicubic degradation은 인접해 있는 14개의 픽셀 값들에 가중치를 곱하는 형태의 계산을 통해 저해상도 이미지가 되는 과정이다 (고해상도 -> 저해상도는 샘플링이 많이 된 부분을 더해서 하나의 픽셀로 만드는 과정). 이런 simulation을 통해 앞선 연구들은 진행되어 왔으며, 아래 그림은 현재 SISR에서 state-of-the-art를 기록하고 있는 RCAN이라는 모델을 활용하여 (1) bicubic degradation(BD), (2) multiple degradataion(MD) (3)  real-world super-resolution
(RealSR) dataset (저자들이 모은 데이터)에 대해 train 및 test를 한 결과를 보여준다.

 <p align="center">
 <img src="/.gitbook/assets/iccv-2019-lp-kpn-kor/motivation.png"/>
 </p>
 (a)는 실제 카메라로 찍은 이미지를 나타내며, (a)에 빨간 박스 쳐진 부분에 BD가 적용된 이미지 (b) 그리고 (1-3)번의 모델을 통해 복원된 이미지(c-e)이다. 확실히 (b, c)는 simulation에 의해 만들어진 데이터를 기반으로 학습되었기 때문에, real-world data를 적용하여 복원했을 때 이미지에 왜곡도 많고 edge부분이 깔끔하지 않다. 반면에 RealSR로 학습된 (e)는 훨씬 매끄럽고 섬세한 이미지가 복원된 것을 확인할 수 있다. 

하지만, RealSR 데이터 셋을 모으더라도 다가 아니다. RealSR data는 simulation으로 degrade된 이미지와 다르게 훨씬 복잡하다. 특히, 실제 이미지에서는 하나의 장면안에 얼마나 깊은 정보(카메라 렌즈로부터 대상들 가지의 거리)가 담겨있냐에 따라 이미지가 degrade되는 방식이 달라진다. 이는 한 장면 안에서도 나타날 수 있기 때문에, spatially variant한 blur kernel이 존재하다고 말한다. 본 논문에서는 크게 이 두가지 문제점을 해결하기 위해 RealSR dataset 구축 및 kernel prediction network기반의 super-resolution이미지를 복원하는 네트워크를 제안했다.
 
### Related work

 #### 1. RealSR dataset 구축

 여태까지의 SISR에 주로 사용된 데이터 셋으로는 Set5, Set14, BSD300 등이 있었지만, 해당 데이터 셋에 대응하는 저해상도 이미지는 bicubic downsampling 이나 gaussian blurring (2D 가우시한 커널과 이미지사이에 convolution 연산을 통해 이미지를 blur하게 만드는 과정)과 같은 단순한 방식으로 얻어졌다. 이후에 Generalization capacity를 늘리기 위한 연구로 좀 더 복잡한 image degradation을 적용하였으나 simulation보다 훨씬 복잡하게 표현되는 실제 image degradation에 적용되기에는 여전히 거리가 멀었다. 
 
 또 다른 시도로는 고해상도-저해상도 이미지 pair를 얻으려는 노력도 있었다. 논문에서는 두개의 선행 연구를 소개했는데, (1) 이미징 시스템을 구축한 뒤 beam splitter(빛을 두 방향으로 갈라주는 optical component)와 두개의 카메라를 활용해 face image에 대한 고해상도-저해상도 이미지 pair를 얻는 방법 (2) 하드웨어(카메라)를 통해 저해상도 이미지를 얻은 다음 이미지 후 처리를 통한 여러 버전의 저해상도 이미지를 얻는 방법들이 있었다. 하지만 두 경우 모두 laboratory(실험실) 안에서만 진행되어 실제 세상에 적용되기에는 데이터 셋의 다양성이 매우 부족했다.
 
 
#### 2. kernel prediction network(KPN)

RealSR dataset은 일반적으로 하나의 이미지 안에서도 locally 다른 degradation(spatially variant)이 존재하기 때문에, 이를 해결하기 위한 노력 또한 필요하다. KPN은 Monte Carlo noise를 제거하기 위한 연구에서 처음 적용이 되었으며, 훨씬 안정적이고 빠른 convergence 가져다 주며 denoising 부문에서 state-of-the-art를 기록했다. 또한, KPN은 dynamic blurring이나 video interpolation의 convolution kernel에서의 blur kernel을 estimation하는 연구에 적용되기도 하였다.


### Idea

앞서 말했듯이, 본 논문에서는 simulation 기반의 SISR 연구에 대한 실용성에 문제점을 제기하고, 해당 문제를 해결하기 위해 RealSR dataset을 구축하게 된다. 또한, RealSR dataset의 spatially variant한 image degradation을 해결하기 위해 Lapalce pyramid가 결합된 KPN을 도입하여 LP-KPN모델을 제시한다.



## 3. Method

### 1. Real-world SISR Dataset
 
본 논문에서는 dataset을 모으는 것 또한 method에 해당한다. 카메라 센서, 렌즈를 어떤 것을 사용하느냐에 따라 scaling factor, 해상도 등을 고려해야하기 때문에, 해당 과정에 대해서 자세한 방법을 기술하였다.

#### lens에 의한 이미지 형성 과정

일반적으로 lens의 초점 거리 f, 렌즈부터 물체까지의 거리 u, 렌즈부터 이미지 센서까지의 거리 v가 있을 때, 그 관계는 다음과 같이 기술되고 이를 thin lens equation이라고 한다.

$$\frac{1}{f} = \frac{1}{u}+\frac{1}{v}$$

이때, u>>f 이고, h1의 크기를 같는 물체는 이미지 센서에 h2의 크기로 맺힌다고 할때, h1과 h2의 관계는 닮음비로부터 쉽게 구할 수 있다(아래 그림 참고 + thin lens equation). 

$$h2 = \frac{v}{u}h1 = \frac{f}{u}h1$$

 <p align="center">
 <img src="/.gitbook/assets/iccv-2019-lp-kpn-kor/thin_lens.png"/>
 </p>

#### Data collection

데이터를 얻는 과정에서 카메라 활용은 다음과 같다.
- DSLR cameras (Canon 5D3 and Nikon D810) 사용
- Canon 5D3와 Nikon D810의 픽셀 사이즈는 각각 $$5760 \times 3840$$, $$7360 \times 4912$$
- 양한 해상도의 이미지를 얻기 위해서 각각의 카메라에 105mm, 50mm, 35mm, and 28mm의 초점 거리를 갖는 렌즈를 사용하였고, 105mm는 고해상도 이미지, 나머지 세개의 렌즈는 저해상도 이미지를 얻는데 사용
- Generality에 대한 보장을 얻기위해 실내, 실외 환경에서 이미지 촬영
- 두개의 카메라로 총 234장의 scene을 촬영하였고, Canon 5D3와 Nikon D810이 동일한 scene에 대해 촬영을 하지 않았다.

#### Image pair registration

lens의 이미지 형성 과정과 data collection을 보면 고해상도 이미지와 저해상도 이미지 간의 pair 데이터를 만들기 위해서는 post processing이 필요함을 유추할 수 있다. 특히 해상도가 변함에 따라 물체의 scaling factor가 다르기 때문에(focal length에 따라 h1과 h2의 관계가 변한다, 위 식 참고), 이를 보정하는 과정이 필요하다. 본 논문에서는 다음 그림과 같은 과정을 통해서 image registration을 하였다.

 <p align="center">
 <img src="/.gitbook/assets/iccv-2019-lp-kpn-kor/image_registration.png"/>
 </p>

우선, Photoshop을 활용하여 lens distortion correction을 한 뒤 가운데 부분을 crop 한다(distortion correction이 가운데 부분을 제외하고는 완벽히 correct해주지 않기 때문에). 105mm 초점거리를 갖는 렌즈로부터 촬영된 이미지에서 center region crop된 부분이 고해상도 이미지 데이터로 사용되고, 나머지 세 초점거리 (50mm, 35mm, 28mm)를 갖는 렌즈로부터 촬영된 이미지에서 center region crop된 부분이 저해상도 이미지 데이터로 사용된다.

여기서 추가적으로, 본 논문에서 개발한 image regsitration 과정은 다음과 같다.
서로 다른 초점거리를 갖는 렌즈로 촬영된 이미지들은 luminance(얼마나 밝은지의 정도) 다르기 때문에, 이를 보정하며 pair데이터를 만들어주는 과정이 필요하고 위 그림에서 iterative registration과정에 해당한다. 본 논문에서 제시한 저해상도-고해상도 이미지의 luminance 보정 및 image registration은 다음 식을 minimize하는 것으로 해결할 수 있다.

$$ \min_{\tau} ||\alpha C(\tau\circ  I_L) + \beta - I_H||^p_p $$

$$\tau$$는 affine transformation matrix, C는 $$I_L$$을 $$I_H$$와 동일한 크기로 crop 해주는 operation, $$\alpha, \beta$$는 luminance보정 파라미터에 해당한다. 

위 식은 locally linear approximation을 적용한 뒤 iteratively reweighted least square problem (IRLS) 기법을 적용하면 다음과 같이 식이 정리된다.

$$ \min_{\triangle \tau} ||w\odot (A \triangle \tau - b||^2_2 $$  

where, $$ \triangle \tau = (A^{'} diag(w)^2 A)^{-1} A^{'}  diag(w)^2 b $$
에서 최종적으로 iterative하게  $$ \tau = \tau + \triangle \tau $$ 업데이트한다.


### 2. Laplacian Pyramid based Kernel Prediction Network(LP-KPN)

앞서 설명했듯이, 본 논문에서는 kernel prediction network(KPN)를 사용했고, 좀 더 구체적으로는 Laplacian pyramid(LP)기반으로 다음과 같이 설계되었다.

 <p align="center">
 <img src="/.gitbook/assets/iccv-2019-lp-kpn-kor/network.png"/>
 </p>

KPN을 사용한 가장 큰 이유는 각 픽셀에 대해 개별 커널을 학습 할 수 있는 구조이기 때문이다. KPN의 구조에 대해 좀 더 자세히 살펴보면, KPN은 저해상도 이미지를 input으로 받아 $$ T \in R^{(k \times k)\times h \times w} $$의 사이즈를 갖는 output tensor를 생성해낸다. 이 텐서에서의 $$ T(i,j) \in R^{(k \times k)} $$ 벡터는 $$ k \times k $$ 사이즈의 커널 $$ K(i,j) $$로 reshape 된다. 이렇게 reshape된 per-pixel 커널 $$K(i,j)$$은 input 저해상도 이미지 $$ I_L^A (i,j) $$ 의 $$ k \times k $$ 픽셀 만큼에 커널 연산이 적용된다. (per-pixel 커널이라고 부르는 이유가 각각의 픽셀에 하나의 커널 값이 대응되기 때문이라고 생각이 든다.) 커널 연산이 적용됨으로써 최종적으로는 고해상도 이미지 $$ I_H^P (i,j) $$가 다음과 같은 연산을 통해 생성된다.

$$ I_H^P (i,j) = <K(i,j), V(I_L^A (i,j))> $$

$$ V(I_L^A (i,j))>$$은 $$ I_L^A (i,j) $$ 의 $$ k \times k $$픽셀 그리고 $$ < \bullet >$$ 은 내적 연산이다.

다음으로 LP를 설명하기 앞서, Image pyramid에 대해서 간략하게 설명해보면, Image pyramid란 이미지를 해상도와 스케일에 따라 decomposition한 뒤 나눠진 이미지 세트를 의미한다. 일반적으로 원본 이미지가 있고, 단계가 높아질수록 이미지 해상도가 줄고 스케일이 커지므로(pixel 하나의 스케일) 스택을 쌓으면 마치 피라미드처럼 보이게 된다. 위 그림에서는 Laplacian pyramid decomposition을 통해 뒤집힌 형태의 Image pyramid를 확인할 수 있다. 여기서 decomposition하는 방식에서 laplacian decompostion을 사용한 것이 Laplacian pyramid를 만들게 된다. 본 논문에서는 세단계의 이미지 피라미드를 활용했고($$ S_1 , S_2 , S_3 $$), 각각의 이미지 피라미드는 이미지 사이즈의 1배, 1/2배, 1/4배에 해당한다. 

최종적으로 LP-KPN은, KPN을 통해 생성된 kernel tensor $$ T_1 , T_2 , T_3 $$과 Laplacian pyramid의 각각의 stage와의 per-pixel 커널 연산(위 식) 이후에 Laplacian pyramid reconstruction을 통해 고해상도 이미지를 최종적으로 복원하게 된다. 

LP-KPN의 장점으로는, Laplacian pyramid를 사용함으로써 세개의 $$ k \times k $$ 커널 (위 그림에서 각 stage)이 $$ 4k \times 4k $$ 커널과 동일한 receptive field를 갖는 효과를 얻게 된다고 한다. 이는 $$ 4k \times 4k $$ 커널 하나를 학습하는 것보다 세개의 $$ k \times k $$를 학습하는 것이 동일한 효과를 보면서도 훨씬 적은 computational cost를 필요로 하기 때문에, 훨씬 학습이 잘된다고 말한다.

## 4. Experiment & Result

### Experimental setup

LP-KPN의 backbone(위 그림에서 다섯개의 convolution block에 해당하는 부분)은 17개의 residual block을 기본으로 하고, 각각의 residual block은 2개의 convolution layer, ReLU로 구성되어 있다. loss function으로는 L2 loss를 활용하였으며 식은 다음과 같다.

$$ L(I_H, I_H^P) = ||I_H - I_H^P ||_2^2 $$

$$ I_H $$는 ground truth 고해상도 이미지, I_H^P 는 LP-KPN을 통해 생성된 고해상도 이미지이다.

또한, 본 논문에서 사용한 네트워크 structure는 다음과 같다.

 <p align="center">
 <img src="/.gitbook/assets/iccv-2019-lp-kpn-kor/network_architecture.png"/>
 </p>
 
 학습에는 Adam optimizer($$ (\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 1e-8) $$ )를 사용했으며, learning rate는 1e-4로 고정하였다. 각 네트워크들은 1,000K만큼의 iteration을 돌렸다.

### Result

우선, 논문에서는 simulation data와 본 논문에서 모은 RealSR dataset을 활용해 모델에 대한 테스트를 진행했고, 그 결과는 다음과 같다.
 <p align="center">
 <img src="/.gitbook/assets/iccv-2019-lp-kpn-kor/result1.png"/>
 </p>
 
 위 표에서 BD는 bicubic degradation, MD는 multiple degradation을 적용한 경우를 의미하고, Our이 본 논문에서 제작한 데이터를 적용한 경우이다. 표에서 확인할 수 있듯이, 네트워크에 상관없이 본 논문에서 제시한 데이터를 활용한 경우가 가장 좋은 성능을 보여줌을 확인할 수 있다. 이 테스트에 대한 예시는 아래 그림을 통해 확인할 수 있다.
 
  <p align="center">
 <img src="/.gitbook/assets/iccv-2019-lp-kpn-kor/result2.png"/>
 </p>
 
 그림에서 볼 수 있듯이, RealSR이 적용된 경우 가장 깔끔한 고해상도 이미지가 복원이 됐음을 확인할 수 있다.
 
 다음으로, 본 논문에서 제시한 LP-KPN의 성능에 대한 검증과정도 거쳤으며, 그 결과는 다음 표를 통해 확인할 수 있다. 
 
  <p align="center">
 <img src="/.gitbook/assets/iccv-2019-lp-kpn-kor/result3.png"/>
 </p>

여기서 k는 위의 네트워크 구조에서의 커널 사이즈에 해당하며, LP-KPN에서 k=5일때, 가장 우수한 성능을 보여준 것을 확인할 수 있다. 이 테스트에 대한 예시는 아래 그림을 통해 확인할 수 있다.

 <p align="center">
 <img src="/.gitbook/assets/iccv-2019-lp-kpn-kor/result4.png"/>
 </p>

## 5. Conclusion

 본 논문에서는 SISR task에 simulation data만으로 학습 및 테스트 하는 것에 대한 문제를 제기하고 이를 해결하기 위해 RealSR 데이터 셋을 구축하였다. 총 두개의 카메라를 활용하여 300장 가량의 이미지를 다양한 환경에서 직접 촬영했다. 이후 고해상도-저해상도 이미지 pair를 만들기 위해서 scaling factor, luminance의 차이를 고려하였고 image registration 과정을 거쳐 최종 데이터 셋을 확보했다. 이를 Laplacian pyramid 기반의 kernel prediction network에 적용하였고, 최종적으로 실제 세상에 적용가능한 SISR framework를 제시했다. 본 논문이 특별한 이유는 네트워크 제안 뿐만이 아닌, 특정 task에 대한 dataset을 구축하는 framework를 제안했다는 점이다. 

### Take home message \(오늘의 교훈\)

- simulation data만으로는 real world dataset에 적용하는데에 한계가 있고, 이를 해결하기 위해 실제 데이터 셋을 구축하는 과정은 반드시 필요하다.
- 하지만 데이터 셋을 구축하는 과정은 쉽지 않기 때문에, 연구 방향에 있어서 장단점을 고려해야한다.
- conventional한 방법과 deep learning의 결합은 그냥 deep learning을 활용하는 것보다 더 좋은 성능을 낼 가능성이 있다.

## Author / Reviewer information

**이찬석 \(Chanseok Lee\)** 

* Affiliation \(KAIST Bio and Brain Engineering)
* mail: cslee@kaist.ac.kr 

### Reviewer


## Reference & Additional materials
### Reference

[1] Dong, Chao, et al. "Image super-resolution using deep convolutional networks." IEEE transactions on pattern analysis and machine intelligence 38.2 (2015): 295-307.

[2] Ledig, Christian, et al. "Photo-realistic single image super-resolution using a generative adversarial network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

[3] Lim, Bee, et al. "Enhanced deep residual networks for single image super-resolution." Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2017.

[4] Cai, Jianrui, et al. "Toward real-world single image super-resolution: A new benchmark and a new model." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.
