---
description: Chen et al. / Scene Text Telescope - Text-focused Scene Image Super-Resolution / CVPR 2021
---

#  Scene Text Telescope: Text-focused Scene Image Super-Resolution \[Kor]



**[English version](https://awesome-davian.gitbook.io/awesome-reviews/paper-review/2021-fall-paper-review/cvpr-2021-scenetext-eng)** of this article is available.



##  1. Problem definition

> __*Scene Text Recognition (STR)란, 일상적인 풍경 이미지에서 글자를 인식하는 task입니다.*__
>
> (활용 예시:  운전면허증에 있는 문자 읽기, ID card에서의 글자 인식, etc)

![Figure1](/.gitbook/assets/25/Figure1.PNG) 

- 최근 STR분야에 대한 연구가 활발하지만, 저해상도(Low-Resolution, 이하 LR) 이미지에서는 아직까지도 많은 성능 개선이 필요합니다.

- 하지만 실생활에서 LR 텍스트 이미지를 사용하는 경우는 꽤나 많습니다. 예를 들어, 초점이 잘 맞지 않는 카메라로 찍은 이미지나 용량을 줄이기 위해 불가피하게 압축된 텍스트 이미지들이 있습니다.

   → 이러한 문제점을 해결하기 위해, 본 논문에서는 텍스트에 초점을 맞춘 초해상화 (Super-Resolution, 이하 SR) 프레임워크를 제안합니다.
   
   

## 2. Motivation

### Related work

- __Scene Text Recognition__

  - _Shi, Baoguang, Xiang Bai, and Cong Yao. "An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition." *IEEE transactions on pattern analysis and machine intelligence* 39.11 (2016): 2298-2304._

    : 이 논문에서는 CNN과 RNN을 결합하여 텍스트 이미지에서 sequential한 특징을 구했으며, CTC decoder [1]를 사용하여 ground truth에 가장 가깝게 접근할 수 있는 path를 선택할 확률을 최대화했다고 합니다.

  - _Shi, Baoguang, et al. "Aster: An attentional scene text recognizer with flexible rectification." *IEEE transactions on pattern analysis and machine intelligence* 41.9 (2018): 2035-2048._

    : 이 논문에서는 Spatial Transformer Network를 사용하여 텍스트 이미지를 어느정도 rectify하고 attention mechanism을 활용하여 각 타임스텝마다 특정 문자에 초점을 두었다고 합니다.

     → 하지만 위의 논문들 경우 이미지에서 휘어있는(curved) 텍스트들을 처리하기에는 적합하지 않다고 합니다. 

    

- __Text Image Super-Resolution__

  - _Mou, Yongqiang, et al. "Plugnet: Degradation aware scene text recognition supervised by a pluggable super-resolution unit." *Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XV 16*. Springer International Publishing, 2020._

    : 이 논문에서는 multi-task 프레임워크를 고안하여 text-specific한 특징들을 고려하였다고 합니다.

  - _Wang, Wenjia, et al. "Scene text image super-resolution in the wild." *European Conference on Computer Vision*. Springer, Cham, 2020._

    : 이 논문의 경우에는 text SR 데이터셋인 _TextZoom_을 제안하고, _TSRN_이라는 SR네트워크를 제안했습니다.

     → 하지만, 이 두가지 논문의 경우 이미지의 모든 픽셀을 고려하기 때문에 배경으로 인한 disturbance 문제가 생길 수 있으며, 이는 텍스트를 upsampling했을 때 성능 문제를 야기할 수 있다고 합니다.
    
    

### Idea

> 기본적으로, 본 논문에서는 _Scene Text Telescope_ (텍스트에 초점을 맞춘 SR 프레임워크)를 제안합니다.

- 먼저, 임의의 방향으로 회전되어있는 텍스트를 처리하기 위해, _TBSRN (Transformer-Based Super-Resolution Network)_ 을 고안하여 텍스트의 sequential한 information을 고려했습니다
- 또한, 위에서 언급했던 이미지 배경으로 인한 disturbance문제를 해결하기 위해, SR을 이미지 전체에 집중하여 하기보다는 텍스트에 초점을 두었습니다. 따라서, 텍스트 각 문자의 position과 content를 고려하는 _Position-Aware Module_ 과  _Content-Aware Module_ 을 두었습니다.
- 나아가, LR 이미지에서 헷갈릴 수 있는 문자들을 고려하여 _Content-Aware Module_ 에서 _weighted cross-entropy loss_ 를 사용했습니다.



- __추가적으로, 아래의 논문들은 본 논문의 Model 과 Evaluation에서 참고된 논문들입니다.__

  - _Luo, Canjie, Lianwen Jin, and Zenghui Sun. "Moran: A multi-object rectified attention network for scene text recognition." *Pattern Recognition* 90 (2019): 109-118._

  - _Shi, Baoguang, Xiang Bai, and Cong Yao. "An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition." *IEEE transactions on pattern analysis and machine intelligence* 39.11 (2016): 2298-2304._

  - _Shi, Baoguang, et al. "Aster: An attentional scene text recognizer with flexible rectification." *IEEE transactions on pattern analysis and machine intelligence* 41.9 (2018): 2035-2048._

  - _Wang, Wenjia, et al. "Scene text image super-resolution in the wild." *European Conference on Computer Vision*. Springer, Cham, 2020._

    

## 3. Method

> _Scene Text Telescope_ 는 크게 아래의 세가지 모듈로 구성되어 있습니다.
>
> → _**Pixel-Wise Supervision Module + Position-Aware Module + Content-Aware Module**_


![Figure2](/.gitbook/assets/25/Figure2.PNG) 



- **Pixel-Wise Supervision Module**

  1. 먼저, LR 이미지는 [2]에서 언급되었던 misalignment 문제 해결을 위해 _STN (Spatial Transformer Network)_ 을 통과합니다.

  2. 그 후, rectified된 이미지는 _TBSRN_ 을 통과합니다. _TBSRN_ 의 구성은 아래 그림과 같습니다. 

     > **TBSRN (Transformer-based Super-Resolution Networks)**
     >
     > ![Figure3](/.gitbook/assets/25/Figure3.PNG) 
     >
     > - _CNN × 2_ : feature map을 추출하기 위한 부분
     >
     > - _Self-Attention Module_ : sequential한 정보를 고려하기 위한 부분
     >
     > - _2-D Positional Encoding_ : spatial / positional한 정보를 고려해주는 부분

  3. 마지막으로, 이미지는 _pixel-shuffling_ 을 통해 SR로 upsampling됩니다.

     

     +) 해당 모듈에서, loss는 ![Eq11](/.gitbook/assets/25/Eq11.gif) 으로 표현되며, 이때 ![Eq12](/.gitbook/assets/25/Eq12.gif)은 각각 HR이미지와 SR이미지입니다.

      

- **Position-Aware Module**

  1.  Position-Aware 모듈에서는 먼저 synthetic 텍스트 데이터셋 (_Syn90k_ [3], _SynthText_ [4], etc) 을 이용하여 트랜스포머를 기반으로 한 recognition 모델을 pre-train시킵니다.

  2. 이때, 각 time-step의 attending region을 positional clue로 사용합니다.

     - HR 이미지가 주어졌을 때, 트랜스포머의 output은 attention map들의 리스트 형태입니다. 다시 말해, output은 ![Eq2](/.gitbook/assets/25/Eq2.gif) 로 표현될 수 있는데, 이 때 ![Eq3](/.gitbook/assets/25/Eq3.gif) 는 i번째 time-step에서의 attention map이며, ![Eq4](/.gitbook/assets/25/Eq4.gif) 은 text label의 길이입니다.

     - SR이미지 또한 트랜스포머를 통과시켜 ![Eq5](/.gitbook/assets/25/Eq5.gif)를 구합니다.

  3.  위의 과정에서 구한 attention map들로 _L1 loss_ 를 계산합니다.

     ![Eq6](/.gitbook/assets/25/Eq6.gif) 

     

- **Content-Aware Module**

  1. 해당 모듈에서는 먼저, _EMNIST_ [5]를 이용하여 _VAE (Variational Autoencoder)_ 를 학습시켜 텍스트 각 문자의 2차원 latent representaion을 구합니다.
  
     ![Figure4](/.gitbook/assets/25/Figure4.PNG) 
  
  2. 각 time-step마다 pre-train된 트랜스포머의 결과값 (![Eq7](/.gitbook/assets/25/Eq7.gif))과 ground-truth label을 비교합니다. 
  
     즉, ![Eq8](/.gitbook/assets/25/Eq8.gif)(content loss)는 아래와 같이 계산할 수 있습니다.
  
      → ![Eq9](/.gitbook/assets/25/Eq9.gif) (![Eq10](/.gitbook/assets/25/Eq10.gif)= t번째 step에서의 ground-truth)
  
  
  
- __Overall Loss Function__

  ![Eq1](/.gitbook/assets/25/Eq1.PNG) 

     (위의 식에서 _lambda_ 들은 loss term들 사이의 균형을 조절하기 위한 hyperparameter입니다.)

  

  ****

## 4. Experiment & Result

### Experimental setup

* __Dataset__

  > __TextZoom__ [2] : 학습을 위한 LR-HR 이미지 17,367쌍 + testing을 위한 이미지 4,373쌍 (easy subset 1,619쌍 / medium  1,411쌍 / hard 1,343쌍)
  >
  > +) LR 이미지 해상도 : 16 × 64 / HR 이미지 해상도 : 32 × 128
  >
  > ![Figure6](/.gitbook/assets/25/Figure6.PNG) 

* __Evaluation metric__

  > __SR 이미지__ 의 경우, 아래의 두가지 metric을 사용합니다.
  >
  > - PSNR (Peak Signal-to-Noist Ratio)
  > - SSIM (Structural Similarity Index Measure)

  > 나아가, 텍스트에 초점을 맞춘 metric을 두가지 더 사용합니다. 참고로, 아래의 두가지 metric은 논문에서 제안된 metric들입니다. 이 두가지 metric의 경우, _SynthText_ [4] 와 _U-Net_ [6] 에서의 방법을 사용하여 이미지의 텍스트 부분만 고려합니다.
  >
  > - TR-PSNR (Text Region PSNR)
  > - TR-SSIM (Text Region SSIM)
  >
  
* __Implementation Details__

  > __HyperParameters__
  >
  > - Optimizer : Adam
  >
  > - Batch 크기 : 80
  >
  > - Learning Rate : 0.0001

  > 사용한 GPU : NVIDIA TITAN Xp GPUs (12GB × 4)



### Result

- __Ablation Study__

  > - 나아가, 본 논문에서는 각 모듈 및 요소 (backbone, Position-Aware Module, Content-Aware Module, etc.) 들의 효과를 검증하기 위해 ablation study를 진행했습니다. 
  >
  > - 데이터셋 : _TextZoom_ [2]
  >
  >   +) 아래의 표들에서 Recognition 정확도는 pre-train된 _CRNN_ [7]을 기반으로 계산되었습니다.
  >
  >   ![Table](/.gitbook/assets/25/Table.PNG) 
  >

- __Results on _TextZoom_ [2]__

  > - 각각 다른 backbone을 기반으로 세 가지 모델 (_CRNN_ [7], _ASTER_ [8], _MORAN_ [9]) 에서의 정확도를 비교했으며, 결과는 아래 표와 같습니다. 
  > - 본 논문의 _TBSRN_ 를 backbone으로 사용했을 때의 정확도가 상대적으로 높음을 확인할 수 있습니다. 
  >
  > ![Table5](/.gitbook/assets/25/Table5.PNG) 
  >
  > - _Visualized Examples_
  >
  >   ![Figure8](/.gitbook/assets/25/Figure8.PNG) 

- __Failure Cases__

  > ![Figure10](/.gitbook/assets/25/Figure10.PNG) 
  >
  > 또한, 논문에서는 SR에서 제대로 인식을 하지 못한 경우도 조사를 했는데, 해당 경우들은 아래와 같습니다.
  >
  > - 길거나 작은 텍스트
  >
  > - 배경이 복잡하거나 occlusion이 있는 경우
  > - Artistic한 폰트 또는 손글씨
  > - 학습 데이터셋에 label이 없는 이미지들



## 5. Conclusion

- 요약하자면, 본 논문은

  - 불규칙한 텍스트 이미지들을 처리하기 위해 self-attention mechanism을 사용한 _TBSRN_ 을 backbone으로 사용했고,
  - 헷갈릴만한, 즉, 인식이 까다로운 문자들을 고려해 weighted cross-entropy loss를 사용하고,
  - 텍스트에 초점을 둔 여러가지 module로 구성된,
  - Super-Resolution 모델 (_Scene Text Telescope_) 을 제안한 논문입니다.

### Take home message

> - SR technique을 텍스트에 초점을 두어 사용하면 성능이 향상될 수 있다.
> - Ablation study나 Failure case 설명 등이 잘 되어 있는 논문은 fancy하다!



### Author

**박나현 \(Park Na Hyeon\)** 

- _NSS Lab, KAIST EE_
- _julia19@kaist.ac.kr_



### Reviewer

1. Korean name (English name): Affiliation / Contact information
2. Korean name (English name): Affiliation / Contact information
3. ...



## Reference & Additional materials

1. _Graves, Alex, et al. "Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks." *Proceedings of the 23rd international conference on Machine learning*. 2006._
2. _Wang, Wenjia, et al. "Scene text image super-resolution in the wild." *European Conference on Computer Vision*. Springer, Cham, 2020._
3. _Jaderberg, Max, et al. "Reading text in the wild with convolutional neural networks." *International journal of computer vision* 116.1 (2016): 1-20._
4. _Gupta, Ankush, Andrea Vedaldi, and Andrew Zisserman. "Synthetic data for text localisation in natural images." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016._
5. _Cohen, Gregory, et al. "EMNIST: Extending MNIST to handwritten letters." *2017 International Joint Conference on Neural Networks (IJCNN)*. IEEE, 2017._
6. _Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." *International Conference on Medical image computing and computer-assisted intervention*. Springer, Cham, 2015._
7. _Shi, Baoguang, Xiang Bai, and Cong Yao. "An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition." *IEEE transactions on pattern analysis and machine intelligence* 39.11 (2016): 2298-2304._
8. _Shi, Baoguang, et al. "Aster: An attentional scene text recognizer with flexible rectification." *IEEE transactions on pattern analysis and machine intelligence* 41.9 (2018): 2035-2048._
9. _Luo, Canjie, Lianwen Jin, and Zenghui Sun. "Moran: A multi-object rectified attention network for scene text recognition." *Pattern Recognition* 90 (2019): 109-118._

