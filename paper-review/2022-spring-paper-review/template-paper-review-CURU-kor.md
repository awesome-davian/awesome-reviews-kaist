##  1. Problem definition
* 망막은 비침습적으로 심혈관계(cardiovascular system)를 관찰할 수 있는 유일한 조직(tissue)이다. 
* 이를 통해 심혈관 질환의 발달과 미세혈관의 형태변화와 같은 구조를 파악할 수 있다.
* 이미지 분할(Image Segmentation)을 통해 상기된 형태적 데이터를 획득 한다면 안과 진단에 중요한 지표가 될수 있다.
* 본 연구에서는, U-Net(및 Residual U-net) 모델을 활용하여 복잡한 망막 이미지(영상)으로 부터 혈관을 분할(segmentation)하고자 한다.   

<p align="left"><img src = "https://user-images.githubusercontent.com/72848264/163723910-a4437d4a-bdb5-492a-a6fc-b9bf930a2307.png">
<img src = "https://user-images.githubusercontent.com/72848264/163723999-192f183e-d400-4266-acaf-e40a1fa93a3f.png " height="50%" width="50%">


##### *U-Net : Biomedical 분야에서 이미지 분할(Image Segmentation)을 목적으로 제안된 End-to-End 방식의 Fully-Convolutional Network 기반 모델이다.*

###### Link: [U-net][googlelink]
[googlelink]: https://medium.com/@msmapark2/u-net-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-u-net-convolutional-networks-for-biomedical-image-segmentation-456d6901b28a 

## 2. Motivation

### Related work

현재 이미지 분할(Image Segmentation)은 대부분 CNN을 기반으로 구성되어 있다.   
  - [Cai et al., 2016] 우리가 잘 알고있는 VGG net또한 CNN을 기반으로 하고있다.
  - [Dasgupta et al., 2017] CNN과 구조화된 예측(structured prediction)을 결합하여 다중 레이블 추론 작업(multi-label inference task)을 수행함.
  - [Alom et al., 2018] 잔류 블록(residual blocks)을 도입하고 recurrent residual Convolution Layer로 보완하였다.
  - [Zhuang et al., 2019] 두 개의 U-net을 쌓아 잔류 블록(residual blocks)의 경로를 증가시켰다.
  - [Khanal et al., 2019] 모호한 픽셀에 대해 한번 더 축소된 네트워크를 사용하여 배경 픽셀과 혈관 사이의 균형을 잘 잡기 위해 확률적 가중치(stochastic weights)를 사용했다.
  


### Idea

본 연구는 *U-Net* 과 *U-Net with residual blocks*를 서로 연결시킨 구조를 제안한다.    
  - 첫 번째 부분(U-Net)은 특징 추출을 수행하고
  - 두 번째 부분(U-Net with residual blocks)은 잔류 블록(residual block)으로 부터 새로운 특징을 감지하고 모호한 픽셀을 감지한다.
  <p align="center"><img src = "https://user-images.githubusercontent.com/72848264/163726690-f24a5c57-7263-4d4d-a502-5a2d45229172.png" " height="70%" width="70%">  
    
  <p align="center"><img src = "https://user-images.githubusercontent.com/72848264/163726699-142a3135-26cb-464e-8aff-5dba13b19274.png" " height="70%" width="70%">




## 3. Method
본 연구의 워크플로우(work flow)는 아래와 같다.

1. 이미지획득
   - 망막 이미지를 수집
    
2. 전처리(pre-processing)
   - 특징 추출(feature extraction), 특정 패턴 highliting, 정규화 등을 진행
   - 이 중에서 CNN architecture에 적용할 특성(characteristics)들을 선택
    
3. 성능 평가 및 가중치 조정
   - 최상의 결과를 위해 해당 과정은 지속적으로 진행
    
4. 결과해석   
    
### 1. Pre-Processing
전처리를 통해 이미지의 품질을 향상시킬 수 있는데, 이는 CNN이 특정 특성 탐지에 매우 중요한 단계이다.

#### Step1
RGB이미지를 흑백이미지로 변환해준다. 이는 혈관과 배경(background)의 대비를 높여 구분시켜준다.

관련식은 아래와 같다

![image](https://user-images.githubusercontent.com/72848264/163793575-f3a78125-06c7-4d7c-a4cd-b4037a8ebf22.png)   
여기서, R G B는 각각 이미지의 채널이다. 위 식에서는, G(Green)을 가장 강조시켰다. 녹색이 가장 노이즈가 적고 이미지의 디테일한 부분 까지 포함한다고 한다.

#### Step2
데이터 정규화(normalization) 단계이다. 이 단계는 분류 알고리즘과 특히 역전파(backpropagation) 신경망 구조에 매우 유용하다. 각 훈련 데이터(trainig data)로 부터 추출된 값들을 정규화한다면 훈련속도 향상을 기대할수 있다.
    

본 연구에서는 2가지의 정규화 방법이 사용되었다. 하기 될 2가지 방법이 가장 일반적으로 사용되는 방법이라고 한다.
1. 최소-최대 정규화(Min-Max normalization)
- 데이터를 정규화하는 가장 일반적인 방법이다. 모든 feature에 대해 각각의 최소값 0, 최대값 1로, 그리고 다른 값들은 0과 1 사이의 값으로 변환하는 거다. 예를 들어 어떤 특성의 최소값이 20이고 최대값이 40인 경우, 30은 딱 중간이므로 0.5로 변환된다. 이는 입력 데이터를 선형변환하고 원래 값을 보존할 수 있다.
  

만약 v라는 값에 대해 최소-최대 정규화를 한다면 아래와 같은 수식을 사용할 수 있다.   
    
![image](https://user-images.githubusercontent.com/72848264/163801296-f8fe968f-fbb2-41c0-af74-f45941359719.png)   
    
  - v′: 는 정규화된 값
  - v: 원래 값
  - A: 속성 값 (여기서는 각 채널의 밝기이다. 0이면 가장 어둡고, 255는 가장밝다)
  - MAX<sub>A</sub>: 입력 데이터(이미지)내에서 가장 작은 밝기 값
  - MIN<sub>A</sub>: 입력 데이터(이미지)내에서 가장 큰 밝기 값   
   
 

    
    
    
2. Z-점수 정규화(Z-Score Normalization)
- Z-점수 정규화는 이상치(outlier) 문제를 피하는 데이터 정규화 전략이다. 만약 feature의 값이 평균과 일치하면 0으로 정규화되겠지만, 평균보다 작으면 음수, 평균보다 크면 양수로 나타난다. 이 때 계산되는 음수와 양수의 크기는 그 feature의 표준편차에 의해 결정되는 것이다. 그래서 만약 데이터의 표준편차가 크면(값이 넓게 퍼져있으면) 정규화되는 값이 0에 가까워진다. 최대-최소 정규화에 비해 이상치(outlier)을 효과적으로 처리할 수 있다.   
    
![image](https://user-images.githubusercontent.com/72848264/163801342-240454d4-695e-48af-af36-ff6fdef67197.png)

  - σ<sub>A</sub>: 표준편차
  - A′: A의 평균값
    
#### Step3
세 번째 단계는 흑백 망막 이미지의 세부 사항을 균일하게 개선하는 효과적인 방법인 "대비 제한 적응 히스토그램 균등화(Contrast Limited Adaptive Histogram Equalization, CLAHE)"를 적용하는 것이다.
    
- 이미지의 히스토그램이 특정영역에 너무 집중되어 있으면 contrast가 낮아 좋은 이미지라고 할 수 없음
- 전체 영역에 골고루 분포가 되어 있을 때 좋은 이미지라고 할 수 있는데, 특정 영역에 집중되어 있는 분포를 골고루 분포하도록 하는 작업을 Histogram Equalization 이라고 함   
- 기존 히스토그램 균일화 작업은 전체 픽셀에 대해 진행해 원하는 결과를 얻기 힘든 반면, CLAHE는 이미지를 일정한 크기를 작은 블록으로 구분하여 균일화를 진행하기 때문에 좋은 품질의 이미지를 얻을 수 있다.
###### Link: [CLAHE][1]
[1]: https://m.blog.naver.com/samsjang/220543360864
 
#### Step4
마지막 단계는 감마 값을 통해 밝기를 조절하는 것이다. 이는 밝기가 한곳에 집중되어 특징 추출에 장애가 되는 것을 방지해준다.
    
    
전 처리를 거쳐 획득한 이미지는 아래와 같다   
<img src = "https://user-images.githubusercontent.com/72848264/163806930-194ff7d3-92a3-43c2-a5b3-f2961aea24c1.png " height="50%" width="50%">
    
전 처리한 이미지로 부터 패치(patches)를 추출하여 더 큰 규모의 데이터 세트를 획득하고 구성된 신경망 훈련에 이용한다. 또 이 패치(patches)에 여러가지 변형(flipping)을 주어 가용 데이터를 추가 확보한다. 
    

### 2. Architecture
본 연구에서는 이중 연결된 U-Net을 사용되었고, 두 번째 부분은 잔류 네트워크(residual network)가 사용되었다. 
    
    
#### [U-Net][googlelink]은 이미지의 전반적인 컨텍스트 정보를 얻기 위한 네트워크와 정확한 지역화(Localization)를 위한 네트워크가 대칭 형태로 구성되어 있다.
Expanding Path의 경우 Contracting Path의 최종 특징 맵으로부터 보다 높은 해상도의 Segmentation 결과를 얻기 위해 몇 차례의 Up-sampling을 진행한다.
다시 말해, Coarse Map에서 Dense Prediction을 얻기 위한 구조이다.
Coarse Map to Dense Map 개념 뿐만 아니라 U-Net은 FCN의 Skip Architecture 개념도 활용하여 얕은 층의 특징맵을 깊은 층의 특징맵과 결합하는 방식을 제안하였다.
이러한 CNN 네트워크의 Feature hierarchy의 결합을 통해 Segmentation이 내제하는 Localization과 Context(Semantic Information) 사이의 트레이드오프를 해결할 수 있다.
    
    
#### **U-Net:**   
The Contracting Path
  - 3x3 convolutions을 두 차례씩 반복 (패딩 없음)
  - 활성화 함수는 ReLU
  - 2x2 max-pooling (stride: 2)
  - Down-sampling 마다 채널의 수를 2배로 늘림

Expanding Path는 Contracting Path와 반대의 연산으로 특징맵을 확장한다.   

   
    
The Expanding Path
  - 2x2 convolution (“up-convolution”)
  - 3x3 convolutions을 두 차례씩 반복 (패딩 없음)
  - Up-Conv를 통한 Up-sampling 마다 채널의 수를 반으로 줄임
  - 활성화 함수는 ReLU
  - Up-Conv 된 특징맵은 Contracting path의 테두리가 Cropped된 특징맵과 concatenation 함
  - 마지막 레이어에 1x1 convolution 연산   
위와 같은 구성으로 총 23-Layers Fully Convolutional Networks 구조이다.
주목해야 하는 점은 최종 출력인 Segmentation map의 크기는 Input Image 크기보다 작다는 것이다. Convolution 연산에서 패딩을 사용하지 않았기 때문이다.
    
    
#### **잔류 블록(Residual block):**   
열화(Degradation) 문제를 해결하기 위해 잔류블록도 제안되었다.   
![image](https://user-images.githubusercontent.com/72848264/163810751-5967a425-3242-47b7-b9ab-4abbce4b4321.png)   
여기서 FM(x)은 F(x)로 표현되는 입력 형상에 두 개의 컨볼루션 레이어를 적용하는 것에서 예상되는 형상 맵이며, 이 변환에 원래 입력 x가 추가되었다. 원래 형상 맵을 추가하면 모델에 나타나는 열화 문제가 완화된다. 아래는 본 작업에 사용된 프로세스이다.   
    
![image](https://user-images.githubusercontent.com/72848264/163811036-56dbcf73-cc23-48ae-81c3-5e9b93d787e7.png)
   
       
    
- U-Net2 with Residual blocks: 
U-Net 네트워크의 출력과 두 번째 네트워크의 입력을 구성한다. 각 수준의 채널 수와 이미지 크기는 앞 절반의 디코딩 부분과 동일하게 유지되었다. 하지만 Contracting과 Expanding 모두 새로운 수준에서 잔류 블럭이 추가되었다. 그리고 마지막 Expanding에서 이진 분류 작업이 수행되므로, 1x1 컨볼루션을 적용하였다.   
    
![image](https://user-images.githubusercontent.com/72848264/163812584-eee949df-59da-4dfa-9ca9-9159d757a715.png)   
    
해당 이미지의 픽셀은 대부분 배경이고 소수만이 혈관 구조를 나타낸다(클래스 불균형). 이 때문에 손실함수가 사용되고 방정식은 아래와 같다.
![image](https://user-images.githubusercontent.com/72848264/163812902-df5d3c9b-2a79-4423-b78f-209870a1e918.png)   
    
이 함수는 분류가 잘못되었거나 불분명할 때 높은 손실 값을 주고 예측이 모형의 예상과 일치할 때 낮은 손실 값을 부여하여 데이터의 전체 확률을 최대화 한다. 로그는 패널티를 수행하고, 확률이 낮을수록 로그값은 증가한다. 확률들은 0과 1 사이의 값을 가진다. 그리고 각 클래스에 가중치 부여한다.   
![image](https://user-images.githubusercontent.com/72848264/163813687-7da187c5-ecf1-47c2-bb45-797f1ab1d8e0.png)   
    
여기서 무게 w는 1과 α 값 사이에서 무작위로 변화하며, s는 스텝이다. 이러한 동적 가중치 변화는 네트워크가 지역 최소값으로 떨어지는 것을 방지한다. 로그 확률을 얻기 위해 LogSoftmax 함수가 신경망 마지막 레이어에 적용된다.

    
## 4. Experiment & Result

### Dataset   
1. DRIVE
- Each image resolution is 584*565 pixels with eight bits per color channel (3 channels). 
- 20 images for training set
- 20 images for testing set
    
    
2. CHASEDB
- Each image resolution is 999*960 pixels with eight bits per color channel (3 channels).

### Evaluation metric   
망막 이미지는 클래스의 불균형을 보여주므로 적절한 metric을 선택해야 한다. 본 논문에서는 **Recall, precision, F1-score, accurarcy**를 채택하였다.   
    
- **Recall:** tells us how many relevant samples are selected.   
![image](https://user-images.githubusercontent.com/72848264/163916511-27ca1a9f-3d94-4418-9d34-e8547acdc2dc.png)

- **Precision:** tells us how many predicted samples are relevant.   
![image](https://user-images.githubusercontent.com/72848264/163916539-3dc46abc-f260-4813-90db-d7c351d4b783.png)

- **F1-Score:** is the harmonic mean between recall and precision.   
![image](https://user-images.githubusercontent.com/72848264/163916575-d1705aeb-bc8f-4a98-a9ce-a8a3f74665de.png)

- **Accuracy:** measures how many observations, both positive and negative, were correctly classified.   
![image](https://user-images.githubusercontent.com/72848264/163916588-9fddcf76-b3d1-44cc-bcef-27645342dd3f.png)

    
### Results
    
1. 전반적 성능  
<img src = "https://user-images.githubusercontent.com/72848264/163916942-7be141aa-fb61-4fe7-96d6-e33c91690fdf.png" height="40%" width="40%"> <img src = "https://user-images.githubusercontent.com/72848264/163982322-05b37196-d9c4-400c-a69e-6145eec775b2.png" height="43%" width="43%">
    
- 상기된 측정지표들을 바탕으로, 선행 연구들과 성능을 비교함
- F1-Score의 높은 수치덕에 Precision 과 Recall 모두 골고루 높은 값을 가짐
    -혈관 분류에 적합함
- Accuracy에서는 가장 높은 수치를 보여주었고, F1-Score에 대해서 2번째로 높은 결과를 보여줌
- 본 연구는 대부분의 경우 ground truth와 일치하였고, FP, FN 또한 적다고 볼수 있다.   
    
    
    
    
2. 소요시간   
![image](https://user-images.githubusercontent.com/72848264/163981962-222e788e-453b-4d2e-a951-502732c9ba81.png)

- 본 아키텍쳐는 Khanal et al. 에 비해 많은 시간을 단축시켰다
    - DRIVE 데이터 셋에 대해서는 약 1시간
    - CHASEDB 데이터 셋에 대해서는 약 10시간
   
   
   
3. 분할(segmentation)과 구조 유사도 지수(The structural similarity index, SSIM)   
    
<img src = "https://user-images.githubusercontent.com/72848264/163982446-49a353bd-012a-49e4-aa9a-91a1ee21ce07.png " height="40%" width="40%"> <img src = "https://user-images.githubusercontent.com/72848264/163982518-aa9a2d81-bc2c-4362-81f9-a94f4e6c9e6d.png " height="42%" width="42%">   
Drive 데이터셋과 CHASEDB 데이터셋의 분할(segmentation)결과   
   
   
   
   
**구조 유사도 지수(The structural similarity index, SSIM)** 은 분할(segmentation) 프로세스를 평가하기위해 도입함, U-Net1 만 있는 첫 번째 단계와 잔류 블록이 추가된 두 번째 단계(U-Net2 with residual block)를 비교하기 위함.   
<img src = "https://user-images.githubusercontent.com/72848264/163997016-f6de07d7-f347-4470-ad73-9309b3a2d523.png" height="40%" width="40%"> <img src = "https://user-images.githubusercontent.com/72848264/163982741-27d1bdb4-ff6d-4775-96b8-9561d3e60b0c.png " height="42%" width="42%">   
   
구조 유사도 지수는 gtound truth와 테스트 이미지들 간의 viewing distance와 edge information를 분석한다. 이는 이미지 품질 저하를 수치화하여 측정한다.(이미지 압축 같은 곳에서 사용) 이는 0 ~ 1 의 값을 가지고, 높을수록 좋다. 그림 6은 U-Net1과 ground truth를 비교한 것이고, 그림 7은 전체 아키텍쳐(U-Net1 + U-Net2 with residual block)과 ground truth와 비교한것이다. 후자가 더 높은 수치를 가진다.   


4. 분할(segmentation) 성능에 영향을 주는 요소   

- Chunk(덩어리진 혈관)
<img src = "https://user-images.githubusercontent.com/72848264/164000556-a2949650-41b7-4873-a3f9-bb6a6e9a6376.png" height="40%" width="40%">   


파란색 동그라미친 부분을 보면, 혈관들이 비교적 뭉쳐있는 것을 볼수 있다.
이미지 분할(segmentation)에서 중요한 문제인데, 위는 잘 구분한 것을 볼 수 있다.


- 병변 부위를 잘 피해갔는지   
<img src = "https://user-images.githubusercontent.com/72848264/163983163-371e45b7-045f-45b2-a992-22bc0403be7e.png " height="42%" width="42%">
DRIVE 데이터셋에는 7개의 병변이 포함된 이미지가 있는데, 이를 혈관으로 착각하고 분할(segmentation)을 할 수 있다.
위 사진을 보면, 병변부위(c)를 피해 잘 수행 된것으로 보인다.
    
**--> 수치화된 지표가 있었으면 좋겠다.**
    
    
    





## 5. Conclusion

1. 본 연구의 노벨티는 크게 2가지로 볼 수 있다.
  - 첫 번째, 기존 U-Net 네트워크에 잔류 블럭을 추가한 것이다. 이는 이미지의 열화(degradation)을 완화하는데 큰 기여를 했다. 
  - 두 번째, 앞의 U-Net에서 얻은 정보를 뒤의 U-Net(U-Net with residual blocks)의 잔류 블럭과 연결시켜 정보손실을 최소화 하였다.   

2. 본 연구는 성능과 훈련시간 둘다 잡았다.
  - 선행 연구와 비슷한 수준의 성능을 보여주지만
  - 훈련시간을 크게 단축 시켰다는 것에 의의를 둘 수 있다.
    
3. 이미지 전처리 과정
  - 그레이 스케일로 변환, 정규화, CLAHE, 감마값 조절 작업으로 품질 좋은 입력 이미지로 만들었고
  - 원본 이미지를 패치(patch)작업하여 부족했던 데이터들을 증강하여 확보함   

    
### Take home message \(오늘의 교훈\)



### Author

**Korean Name \(English name\)** 

* Affiliation \(KAIST AI / NAVER\)
* \(optional\) 1~2 line self-introduction
* Contact information \(Personal webpage, GitHub, LinkedIn, ...\)
* **...**

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. **[Original Paper]** G. Alfonso Francia, C. Pedraza, M. Aceves and S. Tovar-Arriaga, "Chaining a U-Net With a Residual U-Net for Retinal Blood Vessels Segmentation," in IEEE Access, vol. 8, pp. 38493-38500, 2020
2. **[Blog]** https://medium.com/@msmapark2/u-net-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-u-net-convolutional-networks-for-biomedical-image-segmentation-456d6901b28a

