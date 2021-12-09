---
description: Niklaus et al. / Video Frame Interpolation via Adaptive Convolution / CVPR 2017
---

# Video Frame Interpolation via Adaptive Convolution [Kor]

## 1. Problem definition

---

 Video frame interpolation은 기존의 프레임들을 이용하여 연속되는 프레임 사이 중간 프레임을 새로 생성함으로써 비디오 프레임율을 높이는 task입니다. 1초에 몇개의 프레임이 재생이 되는지를 나타내는 프레임율이 작으면 영상이 연속적이지 않아 낮은 퀄리티를 보이게 됩니다. 이때 video frame interpolation을 이용하여 중간 프레임들을 새롭게 생성해냄으로써 영상을 더욱 연속적이게 보이게하여 높은 퀄리티를 가지도록 만들 수 있습니다.

 하나의 비디오에 5개의 연속된 프레임이 있다고 가정하였을 때, video frame interpolation을 통해 연속되는 프레임 사이에 하나의 프레임을 새롭게 만들어냄으로써 총 9개의 프레임을 가진 비디오를 만들어 낼 수 있습니다. 이로써 물체의 움직임이 더욱 연속적으로, 자연스럽게 보일 수 있도록 만드는 것입니다.


![FRUC.png](/.gitbook/assets/46/FRUC.png)

Figure 1: Convert low frame rate to high frame rate


## 2. Motivation

---

 보통의 video frame interpolation 기법은 프레임들 간의 움직임을 추정하고 이를 바탕으로 기존의 프레임들의 픽셀 값을 합성하게 됩니다. 이때 interpolation 결과는 프레임 사이의 움직임이 얼마나 정확하게 추정이 되는지에 따라 달라지게 됩니다. 해당 논문은 움직임 추정과 픽셀 합성의 두 단계 과정을 한 단계로 합침으로써 강인한 video frame interpolation 기법을 제안하였습니다.

### Related work

- **기존의 frame interpolation 기법**
    - Werlberger *et al*. Yu *et al*. Baker *et al*. 에서 제안된 기존의 많은 frame interpolation 기법들은 optical flow 또는 stereo matching을 이용하여 두 연속된 프레임들 사이의 모션을 예측하고, 이를 바탕으로 두 프레임 사이에 하나 또는 여러 개의 프레임을 interpolate 하였습니다.
    - Meyer *et al*.은 기존의 motion estimation 방법과 달리 input 프레임들 사이의 phase 차이를 구하고 이 phase 정보를 multi-scale pyramid level에서 propagating 시킴로써 더 좋은 video frame interpolation 결과를 얻고자 하였습니다.
    
- **딥러닝 기반의 frame interpolation 기법**
    - Zhou *et al*.의 논문에서는 동일한 물체를 여러 다른 시각으로 바라본 것들은 서로 연관성이 높다는 점을 이용하여 새로운 frame interpolation을 제안하였습니다. 여러 input view들을 흐름에 따라 warping 시키고, 그것들을 합침으로써 새로운 view 합성을 위한 적당한 픽셀을 고르는 방법을 제안하였습니다.
    - Flynn *et al*. 은 input 이미지를 여러개의 depth plane으로 projection을 시키고 각각의 depth plane에 있는 색들을 합침으로써 새로운 이미지를 합성하고자 하였습니다.

### Idea

 해당 video frame interpolation 기법은 기존에 분리되어 진행되던 모션 추정과 픽셀 합성을 하나의 과정으로 합쳤습니다. 프레임 사이의 움직임에 대한 정보를 이용하여 어떤 픽셀들이 합성에 이용될 것인지, 그리고 이들 중 어떤 픽셀에 더 많은 weight를 줄 것인지를 나타내주는 interpolation coefficient가 표현되어 있는 convolution kernel을 예측하고자 한 것 입니다. 이렇게 예측된 kernel을 input 이미지와 결합시킴으로써 최종 중간 프레임을 얻을 수 있게 됩니다.

 이때, 제안한 기법은 별도로 optical flow나 multiple depth plane을 이용하여 input 이미지를 warping 시키는 과정을 거치지 않아도 되기 때문에 연산량이 감소하고, occlusion과 같이 합성이 어려운 경우에도 좋은 결과를 내보낼 수 있다는 장점이 있습니다.

## 3. Method

---

 제안하는 video frame interpolation 기법은 두 개의 input frame  I1, I2가 있을 때 두 프레임의 중간에 있는, 새로운 프레임 ![equ4.png](/.gitbook/assets/46/equ4.png) 을 interpolate 하는 것을 목표로 합니다.

**Overall method**


![Approach.PNG](/.gitbook/assets/46/Approach.PNG)

Figure 2: Interpolation by convolution (a): previous work (b): proposed method


 Figure 2 (a)에서 볼 수 있듯이, 기존의 video frame interpolation 기법은 모션 추정을 통해 ![equ4.png](/.gitbook/assets/46/equ4.png) 의 픽셀 (x, y)에 상응하는 I1, I2에서의 픽셀들을 구하고 이들을 weighted sum을 하여 최종 interpolate frame를 구하였습니다. 반면 Figure 2 (b)의 제안하는 방법은 모션 추정과 픽셀 합성을 하나의 과정으로 합치기위해 interpolation에 대한 정보가 들어있는 kernel을 예측하고,입력 프레임들의 patch인 P1,P2와 kernel의 local convolution을 수행함으로 interpolation을 진행하였습니다.

![Architecture.PNG](/.gitbook/assets/46/Architecture.PNG)

Figure 3: Overall process of proposed method

  Figure 3는 제안하는 방법의 전반적인 과정을 보여주고 있습니다. ![equ4.png](/.gitbook/assets/46/equ4.png) 에서 얻고자하는 픽셀의 위치를 (x, y) 라고 했을 때, 각각  I1, I2에서 (x, y)를 중심으로 하는 receptive field patch R1, R2가 fully convolutional neural network(Convnet)의 input으로 들어가게 됩니다. 이때 Convnet은 input 프레임의 정보들을 이용하여 프레임들 사이의 모션을 추정함으로써 input의 어떤 픽셀들을 interpolation에 이용할지, 그 중 어느 픽셀에 비중을 두어 합성할 지에 대한 정보가 담긴 kernel을 output으로 내보내게 됩니다.

 이렇게 얻은 kernel은 input frame patch P1, P2 와 convolve 됩니다. 이때 P1, P2는 앞서 Convnet의 input  R1, R2 보다는 작은 사이즈이지만, (x, y)를 center로 하는 input patch를 의미합니다. 즉, kernel K를 이용하여 P1, P2와의 convolution을 진행함으로써 최종 interpolated frame의 (x, y)에 해당하는 위치의 pixel 값을 얻을 수 있는 것이다.

 
![equ1.png](/.gitbook/assets/46/equ1.png)


 이 과정을 ![equ4.png](/.gitbook/assets/46/equ4.png) 의 모든 픽셀에 대해 반복함으로써, ![equ4.png](/.gitbook/assets/46/equ4.png)의 모든 픽셀값을 얻어 최종 interpolated된 frame을 얻을 수 있습니다.  
 

**Convolution kernel estimation**

![Convnet.PNG](/.gitbook/assets/46/Convnet.PNG)

Table 1: Architecture of Convnet

  Table 1은 receptive field patch  R1, R2를 input으로 하여 kernel K를 output으로 내보내는 Convnet의 구조를 나타내고 있습니다. Input으로는 79 * 79의 spatial size와 RGB 3개의 채널을 가지는 R1, R2가 concat되어 들어가고, 이 input은 여러개의 convolutional layer들을 거치게 됩니다. 마지막 feature map은 spatial softmax를 거쳐 모든 weight의 합이 1이 되도록 해주고, reshape 함수를 이용한 이미지 size 조정을 통해 output으로 내보내게 됩니다. 이때 output의 크기는 41 * (41+41)의 형태로, 41 * 41의 크기를 가지는 input patch P1, P2 와 local convolution이 수행됩니다.


이때, 두 Convnet의 input인 R1, R2는 channel 축으로, output인 P1, P2는 width 축으로 concatenate가 됩니다. R1, R2을 width 축으로 concatenate를 하여 convnet의 input으로 만들어버리면 concat된 이미지가 하나의 이미지로 인식이 되어 convolution 연산이 같이 진행되기 때문에 두 이미지가 spatial dimension에서 섞인채로 feature map이 만들어지게 됩니다. 즉, 두 receptive field가 spatial information을 잃어버리게 되기 때문에 receptive field는 channel 축으로 concatenate가 이루어지게 되는것입니다. 또한 kernel과 input patch와의 곱셈에서는 P1, P2가 channel축으로 concatenate된 형태로 나오게 되더라도 kernel도 각각의 patch에 맞게 곱해질 수 있는 형태로 나오게 된다면, 문제가 없을것이라고 예상이 됩니다.



 **Loss function**

 먼저, 제안하는 방법은 두가지 loss 함수를 사용합니다. 첫번째로, Color loss는 L1 loss를 사용하여 interpolated pixel color와 ground-truth color 사이의 차를 구하게 됩니다. 이때 단순히 color loss만 사용했을 때 발생하는 블러 문제는 gradient loss를 사용하여 완화시켜주게 됩니다. Gradient loss는 input patch의 gradient를 convnet의 입력으로 했을 때의 output과 ground-truth gradient 사이의 L1 loss를 통해 구할 수 있습니다. 이때 gradient는 중심 픽셀을 기준으로 8개의 neighboring pixel과 중심 픽셀의 차이를 의미합니다.

![equa2.PNG](/.gitbook/assets/46/equa2.PNG)

## 4. Experiment & Result

---

### Experimental setup

 **4.1. Training dataset**

 해당 논문의 dataset은 optical flow와 같은 별도의 ground-truth가 필요 없기 때문에 인터넷의 모든 비디오를 사용가능합니다. 따라서 해당 논문에서는 Flickr with a Creative Commons license로부터 "driving", "dancing", "surfing", "riding", 그리고 "skiing"의 키워드가 담긴 3000개의 비디오를 얻었습니다. 이 중에서 저화질의 비디오는 제거하고 1280 * 720의 해상도로 scaling을 한 후, 연속적인 세개의 프레임씩 묶어 triple-frame group을 형성하였습니다. 이들 중 모션이 작은것들은 최대한 피하기 위해 프레임들 사이의 optical flow와 엔트로피가 높은 250,000개의 triple-patch 그룹을 선별함으로써 비교적 높은 모션을 가진 frame으로 이루어진 dataset을 구성하였습니다.

 

 **4.2. Hyper-parameter selection**

 Deep neural network를 위해 설정해야할 두가지 중요한 hyper-parameter는 convolution kernel size와 receptive field path size입니다. 모션 예측을 잘하기 위해서 kernel의 size는 training data에서 프레임간의 최대 motion 크기였던 38 pixel 보다 큰 41 pixel, 즉 41 * 41로 정하였습니다. 또한 receptive field patch의 size는 convolution kernel size보다 크지만 너무 많은 연산량을 차지하지 않도록 79 * 79로 정하였습니다. 

 **4.3. Training setup**

-Parameter initialization: Xaiver initialization

-Optimizer: AdaMax with ![equ3.png](/.gitbook/assets/46/equ3.png)

-Learning rate: 0.001

-Batch size: 128

-Inference time: 9.1 second for 1280*720

### Result

**Quantitative result**

![quan_result.PNG](/.gitbook/assets/46/quan_result.PNG)

Table 2: Evaluation on the Middlebury testing set (average interpolation error)

Table 2에서 real-world scene의 네가지 예시(Backy, Baske, Dumpt, Everg)에 대해서는 가장 낮은 interpolation error, 즉 가장 높은 성능을 보였습니다. 하지만 synthetic한 frame 이거나 lab scene의 네가지 예시(Mequ., Schef., Urban, Teddy)에 대해서는 좋은 성능을 보이지 않는것을 알 수 있습니다. 그 이유 중 하나로, training dataset의 차이를 들 수 있습니다. 앞서 언급한 것처럼 제안하는 네트워크는 유튜브와 같이 인터넷에서 구할 수 있는 실제 영상, real-world scene의 frame들을 dataset으로 사용하였습니다. 합성이 된 frame들과 real-world의 frame의 특성이 다르기 때문에 합성으로 만들어진 frame에 대해서는 성능이 비교적 좋지 않게 되는 것 입니다.

**Qualitative result**

**-Blur**

![qual_blur.PNG](/.gitbook/assets/46/qual_blur.PNG)

Figure 4: Qualitative evaluation on blurry videos 

 Figure 4에서는 카메라의 움직임, 피사체의 움직임 등으로 인하여 블러가 있는 비디오에 대한 video frame interpolation 결과입니다. 제안한 방법과 Meyer et al에서의 방법이 다른 방법들에 비해 artifact가 거의 없고 sharp한 이미지를 낸다는 것을 확인할 수 있습니다.

**-Abrupt brightness change**

![qual_brightness.PNG](/.gitbook/assets/46/qual_brightness.PNG)

Figure 5: Qualitative evaluation in video with abrupt brightness change

 Figure 5에서는 input frame들 사이의 갑작스러운 밝기 변화로 인해 brightness consistency에 대한 가정이 침해된 경우에 대한 video frame interpolation 결과를 보여주고 있습니다. 이 경우에도 제안하는 방법과 Meyer et al에서 제안한 방법이 artifact가 거의 없는 결과가 나왔습니다. 그 중에서도 특히, 이 논문에서 제안하는 방법이 흐릿함 없이 가장 좋은 결과가 나왔다는 것을 확인할 수 있습니다.

**-Occlusion**

![qual_occl.PNG](/.gitbook/assets/46/qual_occl.PNG)

Figure 6: Qualitative evaluation with respect to occlusion

 Figure 6에서는 새로운 피사체의 유입 등으로 occlusion이 발생할 때의 video frame interpolation 결과를 확인 할 수 있습니다. Artifact가 생기는 다른 방법들에 비해서 제안하는 방법에서는 선명하게, 잘 합성된 결과가 나오는 것을 확인함으로써 제안하는 방법이 occlusion과 같은 어려운 경우에도 frame interpolation을 잘 하는 것을 확인할 수 있습니다.

즉, 이러한 결과를 통해 제안하는 방법이 기존의 video frame interpolation으로 해결하기 어려운 blur, abrupt brightness change, occlusion 과 같은 상황에서도 좋은 결과를 보인다는 것을 확인 할 수 있습니다.

## 5. Conclusion

---

 저자는 모션 추정과 픽셀 합성의 두가지 과정을 하나의 과정으로 합침으로써 더욱 더 강인한 video frame interpolation 기법을 제안하였습니다. 각 픽셀마다 모션과 합성에 대한 정보가 담긴 새로운 kernel을 만들어 interpolation을 수행함으로써 occlusion과 같이 video frame interpolation을 하기 어려운 상황에서도 좋은 성능을 이끌어 냈습니다. 

 하지만 각 pixel마다 큰 크기의 kernel을 생성해내야 하기 때문에 너무 많은 메모리가 사용되고 연산량이 많다는 단점이 있습니다.

### Take home message (오늘의 교훈)

> 꼭 optical flow과 같은 motion estimation을 위한 추가적인 방법을 사용하지 않더라도 좋은 성능의 video frame interpolation을 수행 할 수 있다
> 
> 
> 각 픽셀을 위한 kernel을 예측해 냄으로써 각 픽셀의 상황에 맞게 픽셀 합성을 할 수 있고, 이것이 더욱 결과를 좋게 할 수 있다. 
> 

## Author / Reviewer information

### Author

**이유진 (Yujin Lee)**

- KAIST
- dldbwls0505@kaist.ac.kr


### Reviewer

1. Korean name (English name): Affiliation / Contact information
2. Korean name (English name): Affiliation / Contact information
3. …

## Reference & Additional materials

- S. Baker, D. Scharstein, J. P. Lewis, S. Roth, M. J. Black, and R. Szeliski. A database and evaluation methodology for optical flow. International Journal of Computer Vision, 92(1):1–31, 2011.
- M. Werlberger, T. Pock, M. Unger, and H. Bischof. Optical flow guided TV-L 1 video interpolation and restoration. In Energy Minimization Methods in Computer Vision and Pattern Recognition, volume 6819, pages 273–286, 2011
- Z. Yu, H. Li, Z. Wang, Z. Hu, and C. W. Chen. Multi-level video frame interpolation: Exploiting the interaction among different levels. IEEE Trans. Circuits Syst. Video Techn., 23(7):1235–1248, 2013
- S. Meyer, O. Wang, H. Zimmer, M. Grosse, and A. SorkineHornung. Phase-based frame interpolation for video. In IEEE Conference on Computer Vision and Pattern Recognition, pages 1410–1418, 2015
- J. Flynn, I. Neulander, J. Philbin, and N. Snavely. DeepStereo: Learning to predict new views from the world’s imagery. In IEEE Conference on Computer Vision and Pattern Recognition, pages 5515–5524, 2016
- T. Zhou, S. Tulsiani, W. Sun, J. Malik, and A. A. Efros. View synthesis by appearance flow. In ECCV, volume 9908, pages 286–301, 2016
