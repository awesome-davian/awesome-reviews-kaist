---
description: Hatami et al. / Classification of Time-Series Images Using Deep Convolutional Neural Networks / ICMV 2017
---

# Time-Series viewed as images \[Kor\]



##  1. Problem definition

이 논문에서 다루는 주된 문제는 시계열 분류 (Time-series Classification) 입니다. 단변량 시계열의 경우, 모델은 $$ x  = (x_1,\; ...,\; x_l) $$ 를 입력받아 해당 시계열을 $$N$$개의 카테고리 $$ y \in (c_1, ... , c_N) $$  중 하나로 분류하게 되며 $$ x \rightarrow y $$ 의 함수를 학습하게됩니다. 학습된 모델은 학습하지 않은 시계열에 대해 카테고리 분류가 잘되는 일반화된 모델링을 목표로 합니다. 



## 2. Motivation
Convolutional Neural Networks (CNN)은 게층화된 특징표현을 통하여 이미지 분류 및 인식 문제에서 엄청난 성능향상을 보였습니다. 이러한 성공은 다른 여러 문제들에도 적용되어 많은 발전을 이루었지만, 시계열의 경우 일반적으로 1차원의 신호로 간주하기에 적용이 쉽지 않았습니다. 이를 위해 1차원 합성곱 연산를 사용하기도 하지만 이 논문에서는 1차원 신호를 2차원의 이미지 Recurrence Plot (RP)로 변환하고 2차원 합성곱연산을 온전히 활용하고자 합니다.  

### Related work
리뷰를 하는 2021년 시점에서 CNN의 장단점, 동작과 다양한 구조에 대해서는 많은 연구가 있었고 잘 알려져 있습니다. 때문에 이 논문을 이해하기 위해선 RP에 대한 이해가 주가 됩니다.  RP는 시각화 도구로서 M차원의 변화양상 탐색을 목표로 하며 2차원의 행렬로 표현됩니다. 어떤 시계열이 주어졌을때 이 행렬은 다음과 같이 정의됩니다. 
$$
x = (x_1,\; ...,\; x_l),\;\; x_i \in \mathrm{R}
$$
$$
s = (\vec{s_1},\;\vec{s_2},\;...,\;\vec{s_{l-1}} ) = (s_{12},\;s_{23} ...,\; s_{l-1, l}) \;\;\; \text{where}\; s_{i-1,i} = (x_{i-1}, x_{i})\\
$$
$$
R_{\mathrm{i}, \mathrm{j}}=\theta\left(\epsilon-\left\|\vec{s}_{i}-\vec{s}_{j}\right\|\right), \quad \vec{s}(.) \in \Re^{m}, \quad i, j=1, \ldots, K
$$
$$l$$길이의 시계열 $$x$$에서 공간궤도 $$s_i$$는 각 $$x_{i}$$와 $$x_{i+1}$$로 이루어진 2차원 벡터로 정의되고 RP $$R$$행렬의 각 요소는 이 공간궤도 $$s$$간의 차가입력으로 주어졌을때 단위 계단 함수(Heaviside function)의 출력으로 정의됩니다. 이 때 역치값 이하인 경우 1로 활성화됩니다. 여기서 단위 계단 함수는 연속적인 값변화를 이산화하고 이때 엡실론은 시계열의 잡음을 무시하는 정도를 결정하게 됩니다.

![image](https://user-images.githubusercontent.com/26558158/138040464-491eeb0d-7820-4614-9b58-90084e56d64f.png)

이렇게 변환된 RP는 시계열 데이터가 가진 주기성등을 시각적 패턴으로 보여줍니다.  

![image](https://user-images.githubusercontent.com/26558158/138072688-61acccb7-6192-47b4-a00b-ff633fcc38bc.png)

위 그림에서 적색일수록 궤도간의 차이가 크며 어두운 청색일수록 그 차이가 작게 표현되어 있습니다.  대표적인 시각적 패턴을 살펴보면  
 *  RP 행렬의 대각성분은 궤도 자신과의 차이기에 항상 0으로 어두운청색으로 표현됩니다.
 *  대각 성분을 기준으로 우상과 좌하단은 대칭적입니다.
 *  특정 위치에서 수평 혹은 수직으로 길게 나타난 적색선은 해당 위치에서 다른 모든 위치와 수준이 다른 변화 (상승과 하락)가 표현됩니다.
 *  우상단의 적색은 가까운 시간내의 궤도간의 차이가 심하다는것을 의미하며 우하단은 시간적으로 먼 궤도간의 차이가 심함을 나타내게 됩니다.
 *  두번째 그럼처럼 좌상단에서 우하단으로 적색 대각선은 시간의 흐름에 따라 궤도가 반대일 경우입니다.
 *  같은 모양이 시간축사이에 반복되면 해당 패턴의 주기성을 의미합니다.

이처럼 RP는 시계열내에 보이는 여러 동적인 변화들을 시각적인 패턴으로 표현가능합니다. 

### Idea

이러한 RP와 CNN이 결합은 다양한 패턴을 모델링 가능합니다. 예를 들어, CNN은 어떤 이미지안에서 객체의 공간적 이동변화에 대해 무관하게 객체의 존재여부를 알수 있습니다. 이러한 특징은 시계열에서 특정패턴이 서로 다른 시간에 존재할때 시간 변화에 무관하게 탐지가능합니다.  이는 시계열 분류에서 성능을 높일 수 있는 요소가 될수 있습니다.

## 3. Method

RP를 이해하고 나면 그외 나머지는 단순한 셋팅적인 부분에 불과합니다.  

### 모델 구조

먼저 논문에서 제시한 모델 구조에 대해서 보면 2층의 합성곱 블록을 사용합니다. 한 층은 합성곱 연산, 통합층, 비선형 활성함수가 순차적으로 구성되어있습니다.  각 시계열데이터가 RP로 변환되어 2계층의 합성곱을 통과하고 나면 마지막 층의 채널사이즈 크기의 잠재변수 공간에 맵핑되게 되고 이후에는  완전연결 2계층을 통해 분류를 위한 결정경계를 배우게 됩니다. 

### 학습
학습은 역전파 알고리즘을 통해 모델의 파라미터를 갱신합니다. 이때 손실함수로서 categorical-crossentropy를 사용하게 되고 최적화 알고리즘은 Adam을 사용합니다. 모델의 일반화성능 평가를 위해 학습셋, 유효셋 그리고 평가셋으로 데이터를 분할하고 유효셋에서 최대성능이 나올때의 모델 파라미터와 하이퍼 파라미터로 고정하고 평가셋에 대해 성능을 측정하여 비교합니다. 



## 4. Experiment & Result

실험 단락에서는 논문에서 모델의 평가한 데이터셋, 분류 성능,  그리고 학습된 모델의 필터의 시각화 RP의 연관성에 대해서 다룹니다.

### Experimental setup

논문에서는 UCR 시계열 분류 데이터셋을 사용했습니다. UCR 데이터셋은 서로다른 85개의 시계열 데이터셋으로 이루어져 있으며 각각은 시계열 데이터의 길이, 분류되는 수, 그리고  시계열의 도메인이 상이합니다. 이 중 20개의 시계열 데이터셋에 대해 평가하였습니다.

 성능 비교를 위한 방법론과 알고리즘으로 1-NN DTW, Shapelet, Bop, SAX-VSM, TFRP, MCNN, 그리고 GAF-MTF 까지 7개와 제안된 모델을 비교하였습니다.

비교 방법으로 먼저 각 데이터 셋에 대해여 모델들의 오차율을 구하고 오차율에 따라서 순위를 나열하고 이때 20개 데이터셋에서 평균적인 순위와 1등인 경우에 대해서 보고하였습니다.

### Result

![image](https://user-images.githubusercontent.com/26558158/138048433-7b5b23e9-3b44-41ec-bed9-47bba87eeea0.png)

위 표는 20개 데이터셋에서 7개의 비교 모델과의 실험 결과이며 제안한 모델이 평균적인 순위와 승리 횟수상으로 가장 우수함을 주장합니다.

![image](https://user-images.githubusercontent.com/26558158/138050163-76d0d969-6c91-4ba1-8f91-51b630d164e9.png)

그림 4에서 두 합성곱 블럭의 학습된 필터들을 시각화하여 제시하고 있으며 3 x 3 필터를 사용함으로 최대 길이 5의 패턴을 활성가능합니다.  이 시각화를 통해서 실제로 RP에서 어떤 패턴을 찾는 필터인지 연결가능하며 1차원 신호를 학습할때에 비해 설명가능한 장점이 있습니다.

## 5. Conclusion

본 논문에서는 성공적인 결과를 보이는 합성곱 신경망의 효과를 시계열 분류에서 이용하기 위해 1차원 신호인 시계열 데이터를 위상정보를 반영하는 2차원 RP 행렬로 변환하였습니다.  1차원 신호에서 유사한 패턴의 정렬을 하는것은 어려운 문제이나 RP와 합성곱신경망의 시간 불변의 특성을 결합하여 유사패턴의 정렬문제에 대해 어느정도 해결가능한것으로 보입니다.  실험적증명으로서 기존의 1차원 신호로서 시계열을 표현할때 보다 시계열 분류 문제에서 높은 성능 향상을 보여주었습니다. 



### Take home message

2021년 시점에 2017의 논문이 다소 간단하고 나이브 할수 있지만 제출된 학회의 인지도에 비해 현재 인용수 208회로 높은것을 볼수 있습니다. 이 부분은 1차원 신호로 익순한 시계열 데이터를 2차원 표현으로 변환하는 아이디어 자체에 새로움 떄문이 아닐까 생각합니다.  

## Author / Reviewer information

### Author

**박준우 \(Junwoo Park\)** 

* KAIST AI 
* [Github](github.com/junwoopark92)

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1.  Hatami, Nima, Yann Gavet, and Johan Debayle. "Classification of time-series images using deep convolutional neural networks." *Tenth international conference on machine vision (ICMV 2017)*. Vol. 10696. International Society for Optics and Photonics, 2018.
2. [Recurrence plot with python:  파이썬에서 RP 활용하기](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=rkdwnsdud555&logNo=221381428891)
