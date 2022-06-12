---
description: Keisuke Tateno et al. / Distortion-Aware Convolutional Filters for Dense Prediction in Panoramic Images / ECCV 2018
---

# Distortion-Aware Convolution \[Kor\]

##  1. Problem definition

본 논문에서는 360도 panoramic 이미지에서 depth map을 추정하는 task를 수행합니다. 우리가 익히 알고 있는 perspective한 이미지에서와 달리, 360도 depth 추정의 경우 큰 데이터셋이 존재하지 않아 기존의 딥러닝 방법을 이용하여 높은 성능을 내기 어려웠습니다. 이에, 본 논문에서는 convolution의 커널 모양을 바꾸어 perspective image를 사용해서 학습을 진행하고, 이를 360도 이미지에 적용할 수 있는 방법을 제시합니다. 

## 2. Motivation

위에서 언급한 바와 같이, 360도 depth 데이터셋 구축을 위해서는 여러대의 depth 카메라가 필요하며, 가격도 비쌀 뿐더러 setup time도 오래걸리는 한계점을 가지고 있습니다. 이에, 본 논문의 저자는 기존 perspective 이미지에서 학습을 진행하고, 해당 모델을 360도 이미지에도 적용하기 위해 컨볼루션 필터의 형태를 바꾸는 방법을 제시합니다. 

### Related work
먼저, 우리가 가장 많이 사용하는 컨볼루션 필터는 아래의 그림과 같이 정사각형의 형태를 띄고 있는 3x3 필터입니다. 몇몇 연구자들은 이런 정사각형 형태의 고정된 필터를 사용하는 것은 기하학적으로 일정한 패턴만을 볼 수 있기 때문에 복잡한 이미지 데이터를 유연하게 이해하기 어렵다고 주장하였고, 이에 convolution filter의 shape을 변경하기 시작했습니다.

![image](/.gitbook/assets/2022spring/52/1.jpg)

이러한 배경에서 2017년도에 Deformable Convolution Network라는 논문이 등장했습니다. 기본적으로 convolution에서 사용하는 sampling grid에 2d offset을 더해서 다양한 패턴으로 변형시켜 사용하는 것입니다. (a) 에서는 인접한 부분들만 이용해서 연산을 했다면, offset을 추가한 (b), (c), (d)에서는 좀더 wide한 값들을 유동적으로 사용할 수 있는 것이죠. 

기존 convolution에서 output feature map을 연산하는 것을 생각해보면, regular grid R에 있는 포인트 p_n에 대해서 그 위치에 해당하는 weight값 w와 같은 위치에 있는 input을 곱한것들의 합으로 계산을 했습니다. 

<p align="center">
<img src = "/.gitbook/assets/2022spring/52/1-1.png" width=30%>
 </p>
 <p align="center">
<img src = "/.gitbook/assets/2022spring/52/1-2.png" width=40%>
</p>
근데 이제는 여기에 Δp_n을 추가해서 input의 어떤 위치를 sampling을 할지를 추가적으로 넣어줄 수 있게 됩니다.  

<p align="center">
<img src = "/.gitbook/assets/2022spring/52/1-3.png" width=35%>
</p>

네트워크 구조를 통해 한번 더 이해해보면, 일단 전체에 convolution layer을 통과시켜줘서 offset field를 구합니다. 그리고 deformable convolution을 하고자 하는 포인트를 offset field에서 추출한 후 이 offset 값들을 사용해 deformable convolution을 수행해주게 됩니다.
<p align="center">
<img src = "/.gitbook/assets/2022spring/52/2.jpg" width=60%>
</p>

이러한 방식을 통해 detection task에서 큰 물체에는 큰 receptive field를, 반대로 작은 물체에는 작게 filter을 스스로 학습하여 적용함으로써 성능 향상을 보였습니다. 

Deformable Convolution Network 논문의 경우 offset을 학습을 하여 적용하는 방법을 띄지만, 본 논문에서는 360도 이미지에 대한 offset들을 각 위치마다 고정할 수 있기 때문에 offset 학습을 진행하지는 않습니다. 하지만 기본적인 아이디어는 filter의 shape을 바꾸는 것에서 출발합니다. 


### Idea
<p align="center">
<img src = "/.gitbook/assets/2022spring/52/3.png">
</p>
본 논문에서 하고자 하는 것은 perspective image(기존 이미지)를 이용해서 학습을 하고, 해당 모델을 이용해서 360도 이미지에서 depth estimation을 진행하도록 하는 것입니다. Train과 test에서 생각해보면 다른 domain, 즉 다른 포맷의 이미지를 사용한다는 것은, 네트워크에서 train에서 학습한 weight를 실제 test에서는 해당 의도에 맞지 않게 사용이 된다는 것입니다. 여기서 test에서 사용하고자 하는 equirectangular image의 경우 360도 구의 형태를 지구본을 세계지도로 펼치는 것처럼 나타내는 방식인데, 위의 그림과 같이 양쪽 극단에 심한 왜곡현상이 일어나고 이러한 왜곡은 depth prediction에 상당한 오류를 야기합니다. 이런 문제를 가장 간단히 해결하는 방법은 cube map projection을 사용하는 것인데, cube map은 이미지 경계에 불연속적인 부분들이 존재하고, depth-estimation에서도 해당 부분에서 불연속적으로 추정되는 오류를 야기합니다.

따라서 본 논문에서는 equirectangular 에 distortion aware convolution 방식을 도입해서 이미지 왜곡에 대응하고자 합니다. 


## 3. Method
<p align="center">
<img src = "/.gitbook/assets/2022spring/52/4.png">
 </p>
기존에 우리가 많이 쓰는 필터의 sampling grid R는 다음과 같이 정사각형의 모양입니다. Feature map 에서의 한 pixel의 위치를 p = (x(p), y(p))라고 하면, convolution 연산을 통해 얻은 output feature map에서의 해당 포인트는 다음과 같은 식으로 나타낼 수 있습니다. 
<p align="center">
<img src = "/.gitbook/assets/2022spring/52/5.png" >
  </p>
Distortion-aware convolution에서는 변형된 sampling grid를 사용하고, 이를 수식으로 나타내면 다음과 같습니다. 
<p align="center">
<img src = "/.gitbook/assets/2022spring/52/6.png" width=35%>
</p>

Sampling grid δ를 이용함으로써 receptive field를 rectified 할 수 있게 되고, 여기서 델타는 실수이기 때문에 위의 식을 bilinear interpolation을 이용해서 feature map의 RGB 값을 구합니다. 
<p align="center">
<img src = "/.gitbook/assets/2022spring/52/7.png">
</p>
이렇게 distortion aware convolution을 정의를 해보았는데, 이제 여기서 equirectangular format에 맞게 sampling grid 델타를 정의해보자. 먼저, equirectangular 이미지에서의 한점 p는 다음과 같이 세타와 파이로 나타낼 수 있다. 이 세타와 파이를 이용해서 ps를 3차원 공간상의 unit sphere 좌표 pu로 바꿀 수 있습니다. 우리는 rectified 된 kerne을 만들고자 하고, 이를 위해서 해당 점 pu에서 tangent plane의 coordinate을 다음과 같이 정의합니다. 
<p align="center">
<img src = "/.gitbook/assets/2022spring/52/8.png">
</p>
이제, 해당 tangent plane에 투영된 이미지가 픽셀 포인트 p에서의 rectified image라고 할 수 있습니다. 그러므로, 우리가 원하는 distorted pixel location은 이 tangent plane에 있는 regular grid를 다시 equirectangular coordinate으로 projection 시켜서 구할 수 있고, 이 새로만든 sampling grid를 rsphere이라고 할겁니다. 
<p align="center">
<img src = "/.gitbook/assets/2022spring/52/9.png">
</p>
이제 tangent plane에서 sampling grid에 해당하는 location은 포인트 pu에서 rsphere만큼 더한 위치들이 됩니다. 
<p align="center">
<img src = "/.gitbook/assets/2022spring/52/10.png">
</p>


이제 다했는데, 이 각각의 pu,r들을 equirectangular image domain으로 옮겨놓으면 됩니다. 다시 back projection 시켜줍니다.
<p align="center">
<img src = "/.gitbook/assets/2022spring/52/11.png">
</p>

이렇게 하면 결과적으로 x, y 값을 구할 수 있습니다. 
<p align="center">
<img src = "/.gitbook/assets/2022spring/52/12.png">
</p>

구한 x, y값과 기존의 convolution 연산을 하고자 하는 포인트 p에 대해서 상대적인 coordinate을 구하면 우리가 원하는 sampling grid가 완성됩니다. 

<p align="center">
<img src = "/.gitbook/assets/2022spring/52/13.png">
</p>

이 sampling grid의 경우 같은 horizontal line의 포인트들에 대해서는 다 같기 때문에, vertical line offset들에 대해서만 저장합니다. 결과적으로 위의 그림에서 보시는 바와 같이 equirectangular 이미지에 대해서 rectified receptive field를 얻어낼 수 있습니다.

이제 dense predictio을 위해서 간단한 cnn architectur을 쓰는데, fully convolutional residual network를  변형해서 사용하고자 합니다. FCRN에서의 spatial convolution을 distortion aware convolution을 변경하고, max unpooling을 avaerage로 바꿔주었다고 합니다. 그리고 나머지 loss function과 optimization은 다음과 같습니다. 

FCRN (fully convolutional residual network)
Spatial convolution unit -> distortion aware convolution
Max unpooling -> average unpooling

Loss function: reverse Huber function
Optimizer: SGD (Stochastic Gradient Descent)
Train with perspective images
Test using equirectangular panoramic images
<p align="center">
<img src = "/.gitbook/assets/2022spring/52/14.png">
</p>

이제 이런 간단한 모델 구조를 가지고 할 수 있는 것은 기존의 perspective RGB-D 이미지로 학습을 하고, test에서는  동일한 네트워크 구조, 동일한 weight를 가지고 standard convolution을 distortion aware convolution으로만 변경하면 equirectangular image로 inference가 가능합니다. Training에 사용할 360도 이미지 annotation을 만드는게 굉장히 시간이 많이 드는 작업이었는데, 해결할 수 있게 된 것이죠. 




## 4. Experiment & Result
본 논문에서는 distortion aware convolution 방식을 사용하여 depth prediction, semantic segmentation, style transfer에 대해서 기존 방식 대비 우수한 성능을 보여주었습니다. 본 리뷰에서는 depth prediction에 비중을 두어 설명합니다.


### Experimental setup

먼저 데이터셋으로는 360도 파노라믹 이미지의 depth랑 semantic label을 제공하는 Stanford 2D-3D-S dataset을 사용하였습니다. 실험을 위해 conventional convolution에 이용할 perspective image를 Stanford dataset으로 만들었고, 이를 리스케일링 해서 사용하였습니다.

성능 비교를 위해 Standard Convolution을 사용했을때, Cube Map에 대해 Standard Convolution을 사용했을때 그리고 Distortion Convolution을 사용했을때의 rel, rms, log10을 비교합니다.

학습은 위에 언급했던 바와 같이 perspective RGB-D image와 standard CNN으로 학습을 진행하고, 해당 모델의 컨볼루션 필터들만 distortion aware filter로 변경하여 Equirectangular image에 대해서 test를 진행합니다.

### Result
<p align="center">
<img src = "/.gitbook/assets/2022spring/52/15.png">
<img src = "/.gitbook/assets/2022spring/52/16.png">
  </p>

Depth prediction 에 대해서 표에서 보이시는 바와 같이 기존의 방법들 대비 c distconv가 에러가 가장 낮은 것을 확인하실 수 있습니다. 표의 (1)은 train은 stanford 데이터셋에서 만든 perspective 이미지로 진행했다면, 2번은 perspective dataset인 NYU dataset으로 학습을 진행한 결과입니다. 
<p align="center">
<img src = "/.gitbook/assets/2022spring/52/17.png">
  </p>
여기서도 standard convolution은 distortion으로 인한 artifact를 만들어내고, cubemap의 경우 불연속적인 것들을 볼 수 있는데, distortion convolution으로 많이 개선된 것을 볼 수 있습니다. 
<p align="center">
<img src = "/.gitbook/assets/2022spring/52/18.png">
  </p>
Semantic segmentation task에 대해서도 stdconv보다 결과가 좋았고, 특히나 왜곡이 심한 바닥 부분에서 miou가 차이 많이나게 높아진 것을 볼 수 있습니다.  
<p align="center"
<img src = "/.gitbook/assets/2022spring/52/19.png">
    </p>

Style transfer에서는 FCRN 대신에 VGG를 쓰고 encode 부분의 convolution을 distortion aware로 바꾸어서 실험을 진행했고, 왼쪽 오른쪽 경계 부분이나, cube map border에서의 불연속적인 것도 해결할 수 있었다고 합니다. 





## 5. Conclusion

Contribution으로는 equirectangular 이미지에 맞는 kernel sampling을 제안하여 distortion을 해결하였고, perspective image을 이용해서 학습을 진행함으로써 360도 이미지 데이터셋이 부족한 것에 대한 해결책을 제시하였다. Future work로는 다양한 프로젝션에 적용해보고, 여러 task들에 적용해보겠다 하였습니다. 


### Take home message \(오늘의 교훈\)
도메인이 다른 데이터를 커널을 변경해서 해결해주는 신박한 논문!


## Author / Reviewer information

### Author

**박하늘 \(Haneul Park\)** 

* Affiliation \(KAIST CT \)
* https://github.com/sky0701

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. Tateno, Keisuke, Nassir Navab, and Federico Tombari. "Distortion-aware convolutional filters for dense prediction in panoramic images." Proceedings of the European Conference on Computer Vision (ECCV). 2018.
2. Dai, Jifeng, et al. "Deformable convolutional networks." Proceedings of the IEEE international conference on computer vision. 2017.

