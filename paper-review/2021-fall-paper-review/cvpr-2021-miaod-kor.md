---
description: (Description) Yuan _et al_. / Multiple instance active learning for object detection / CVPR 2021
---

# Multiple Instance Active Learning for Object Detection \[Kor\]

**English version** of this article is available [**here**](cvpr-2021-miaod-eng.md).

## **1. 서론**
이 논문의 제목을 보시면 저자들이 ***Object Detection***을 위해서 ***Multiple Instance Active Learning*** 하고 싶은 것을 알 수 있습니다. 근데 **Active Learning**과 **Instance**라는 것은 뭘까요? 개념들을 제대로 한번 정해봅시다.

### **Active Learning**
**Active Learning**에 대해 인터넷에서 한번 검색해보시면 결과가 많이 나왔죠? 그들은 두 카테고리 (사람의 Active Learning과 AI Active Learning)로 나뉠 수 있습니다. 먼저 사람의 Active Learning라는 방법을 간단히 설명드리자면 학습자가 자기의 학습 과정을 적극적으로 스스로 이끄는 것입니다. 한국어에는 자기주도학습이랍니다 [\[humanactivelearn\]][humanactive1]. 예를들어 수업을 들면 그냥 듣기만 하는 게 아니고 그룹 프로젝트, 토론, 발표 등 같은 활동도 포함됩디다.

인공지능 환경에서도 마찬가지입니다. 모델들에게 데이터를 많이 주고 오랫동안 학습시키는 것은 좋지 않은 경우도 있습니다. 예를 들면 데이터 품질은 좋지 않으면 그냥 시간 낭비뿐만 아니라 모델 정확도가 줄어질 수도 있습니다. 반면에 모델이 학습 데이터를 스스로 고르는 게 어떨까요? 만약에 그거 진짜 가능하냐고 궁금하신 불들이 계시면 '네. 가능죠. 가능할 뿐만 아니라 모델 성능도 향상시킬 수도 있습니다.'라는 답변 드리죠. 자세한 학습 방법에 대해 바로 알려드리고 싶지만 지금은 개념 설명이 좀 더 있습니다.

### **Multiple Instance Learning**
Object Detection에서 input는 보통 이미지나 동영상 프레임이며 그 이미지나 프레임 속에는 인간, 동물, 자동차, 오토바이크 등등 같은 다양한 물체가 나타나는 겁니다. Object Detection model은 그 물체들 위치를 찾은 후 bounding box 그리고 물체류를 추즉해야 합니다. 그려야 할 bounding boxes를 예측하기 위해, 다수의 방식들은 수많은 anchor boxes를 먼저 생성합니다. 그 후에 이 anchor boxes로 모델은 bounding boxes 정확하게 그리도록 학습할 겁니다.

![그림 1: Object Detection 결과의 에시 [소스: https://pjreddie.com/darknet/yolo/]](/.gitbook/assets/11/object-detection.png)

RetinaNet [\[lin2017\]][lin2017] 작동 과정에서 첫 번째 머쥴은 이미지 속에 물체가 있는 가능성이 높은 수많은 구역들을 추천하고 이 구역들은 anchor boxes이나 (이 논문에서는) instance라고 합니다. background만 포함하는 instance는 ***negative instance***라고 반면에 실제로 물제가 있는 instance는 positive instance라고 저의됩니다. 또한 이미지는 속에 많은 instance가 생길 수도 있기 때문에 instance bag이라고요.

![그림 2: Instance bags의 예시 [소스: MI-AOD의 그림 2]](/.gitbook/assets/11/instance-bag.png)

사진을 보시면 많은 instances 중에 빨간섹인 플러스과 마이너스 기호가 몇 개가 있죠. 그 것들이 학습 과정에 유익한 instance이기 때문에 이 논문의 최고의 목적은 그 것들을 찾는 겁니다.

### **Formal Definition**

우리는 여기까지 배웠던 가장 중요한 단어를 이제 수학적으로 한번 다시 정의해봅시다. 

기계학습과 인공지능에서 잘 레이블링된 데이터는 최고입니다. 하지만 데이터 레이블링 작업은 시간과 비용을 너무 많이 걸립니다. 따라서 이 논문의 제안된 방식은 레이블링되지 않은 데이터 중에 유익한 샘플들을 고르고 그들로 모델 훈련하려고 합니다. 또한 제안된 방식은 효과를 명확히 보여주기 위해 레이블링된 데이터세트 ![][x-y-0-l]는 레이블링되지 않은 세트 ![][y-0-l]에 비하면 너무 적다는 가정을 합니다. 

이미지 ![][x-in-x-0-l]는, 아니면  ![][x-in-x-0-u], bag of instances ![][x-set]로 보일 수 있습니다. 레이블링된 세트 ![][y-0-l]는 bounding box의 좌표 세트 ![][y-loc-x]과 물체류 ![][y-cls-x] 세트로 구성됩니다. 이 논문에서, 저자들은 먼저 detection 모델 (예: RetinaNet [\[lin2017\]][lin2017])을 양이 적은 레이블링된 데이터세트 ![][x-y-0-l]로 훈련한 후, 그 모델로 레이블링되지 않은 세트에서 ![][k] 가지의 가장 좋은 샘플를 샌택해서 선택된 샘플들를 레이블링된 이지미 세트 ![][x-0-l]에 추가합니다. 

## **2. Motivation**

### **Uncertainty**

![그림 3: 강아지-고양이 classifier의 예시 [소스: https://www.youtube.com/watch?v=toTcf7tZK8c&t=2061s]](/.gitbook/assets/11/catdog.png)

1. Uncertainty의 두 가지의 종류.

   
   * 인공지능 모델은 Active Learning하려면 자기가 얼마나 모르는지 항상 궁금해야 됩니다. 하지만 그 것을 그냥 softmax 함수로 측정하면 안 됩니다. 외냐하면 softmax 함수 결과들의 합계는 항상 1이죠.

   * 예를 들어 강아지 이미지와 고양이 이미지만으로 훈련 된 위에 있는 모델에 강아지와 고양이 둘 다 있는 이미지가 들어가면 나오는 확률은 고양이 0.5, 강아지 0.5겠죠. 그 다음에 모델은 고양이와 강아지 중에 하나를 선택합니다. 하지만 그 것은 맞습니까? 아니죠?
  
   * 또 다른 예시를 봅시다. 우리가 객관식 유형의 시험에 볼 때 각 문제마다 A, B, C, D 선택이 있어서 다 A를 선택하면 나오는 점수는 25/100점 이겠습니다. 항상 그렇고 바꿀 수는 없습니다. 왜냐하면 데이터 배포는 원래 그렇기 때문입니다. 이 현상는 ***Aleatoric Uncertainty***나 ***uncertainty of data***라고 합니다. [\[ulkumen-uncertainty\]][ulkumen-uncertainty]. 
  
   * 하지만 종은 학생들은 25점을 받는 것이 안 돼서 더 열심히 공부하고 싶겠죠. 그래서 각 문제마다 조심히 살펴보고 문제에 대해 얼마나 아는지 모르는지 알게 됩니다. 이 것은 **Epistemic Uncertainty*** 또 ***uncertainty of prediction***이라고 합니다 [\[ulkumen-uncertainty\]][ulkumen-uncertainty].

2. Epistemic Uncertainty를 측정하는 법

![그림 4: Uncertainty 측정을 위한 Dropout  [소스: https://www.youtube.com/watch?v=toTcf7tZK8c&t=2061s]](/.gitbook/assets/11/dropout.png)

   * Test 때의 dropout

     * 보통 Dropout라는 기법은 regularization을 위해서 훈련 때만 쓰이지만 test 때 stochastic sampling 목적으로 쓰여도 됩니다.

   * Model Ensemble
    
     * 독같은 모델와 독같은 데이터이라도 훈련 후 나오는 결과들의 사이에 차이가 생길 수도 있습니다. 따라서 독립하게 훈련 된 모델들을 모아서 같은 test sample에 쓰는 것은 결과의 배포을 측정하기 위한 sampling으로 보일  수 있습니다.
  
![그림 5: Uncertainty 측정을 위한 Model Ensembling [소스: https://www.youtube.com/watch?v=toTcf7tZK8c&t=2061s]](/.gitbook/assets/11/model-ensemble.png)

   * 마지막에 수많은 수집된 sample 결과들로 모델 output의 기대값 (Expected Value)와 분산값 (Variance)을 계산할 수 있습니다. Variance 값이 클수록 Uncertainty 도가 더 큽니다.

![Figure 6: 모델의 웨이트 분포 [source: https://www.youtube.com/watch?v=toTcf7tZK8c&t=2061s]](/.gitbook/assets/11/variance.png)

![그림 7: 모델의 예측 결과의 기대값과 분산값 측정 [소스: https://www.youtube.com/watch?v=toTcf7tZK8c&t=2061s]](/.gitbook/assets/11/variance2.png)

### **관련연구**
1. Uncertainty 기반 Active Learning
    
    이 글을 쓰는 지금까지 Active Learning을 주제로 제안된 방법의 수가 적지 않습니다. 
    * Lewis와 Catlett [\[lewis1994a\]][lewis1994a]는 데이터 incremental하게 선택하기 위해 heterogeneous uncertainty 샘플링 기법을 제안했습니다. 이 기법은 소요 시간 및 착오율 면에서는 인간 레이블링보다 성능이 더 좋았습니다. 
    * Lewis [\[lewis1994b\]][lewis1994b]는 다른 uncertainty 샘플링 기법으로 text classifiers를 훈련 했습니다. 
    * [\[roth2006\]][roth2006][\[joshi2010\]][joshi2010]은 성능을 향상시키기 위해서 classifier들의 decision boundary의 margin를 기법을 제안했습니다. [\[lou2013\]][lou2013][\[settles2008\]][settles2008]은 entropy를 측정함으로써 uncertainty를 계산했습니다.
    * [\[carbonneau2017\]][carbonneau2017][\[wang2017\]][wang2017][\[hwang2017\]][hwang2017]은 본 논문같이 multiple instnace learning 기술을 사용했지만 사진 분류에만 가능합니다.


## **제안된 기법**

이 논문에서 Object Detection을 위한 Multiple Instance Active Learning 기법 (MI-AOD)가 제안됐습니다. 위에 말씀드렸던 대로 최고의 목표는 가장 유익한 레이블링되지 않은 이미지를 선택하고 레이블링된 세트에 추가하는 겁니다. 또한 uncertainty도를 측정하기 위해 dropout나 model ensemble를 써야 하죠. 하지만 그런 방법을 사용하면 시간을 많이 걸리고 자원이 너무 많이 필요합니다. 따라서 본 논문에서 저자들은 Instance Uncertainty Learning (IUL)을 위해 classifier head 두 가지가 있는 모델을 head들의 예측 불일치을 최대화 하도록 훈련을 합니다. 

![그림 8: Multiple Instance Uncertainty Learning [소스: MI-AOD의 그림 2]](/.gitbook/assets/11/iul.png)

최대화 후 다시 최적화라니? 사진을 보신 후 좀 당황스럽죠. 저도 그랬습니다. 실은 이 둘은 차이가 좀 있습니다. 첫 번째의 것은 아까 전에 말씀드렸습니다. Classifier head들 (![][f1]과 ![][f2])의 예측 불일치 최대화를 한다는 거죠. 그 다음에 레이블링된 세트과 레이블링되지 않은 세트의 bias 차이를 최적화해야 합니다.

그럼 이제 모델은 유익한 이미지를 선택할 수 있는건가요? 아니오. 훈련은 아직 좀 더 해야 합니다.

![그림 9: Multiple Instance Uncertainty Reweighting [소스: MI-AOD의 그림 2]](/.gitbook/assets/11/iur.png)

예를 들어 우리가 강아지를 더 잘 인식하기 위해 훈련하고 있고, 두 개의 사진이 있다고 상상해 보세요. 하나는 강아지로 가득하고 다른 하나는 다른 많은 대표적인 물건들 중 오직 한 개의 강아지만 가지고 있으면, 두 중에 어떤 것이 더 좋을까요? 두 사진에 모두 강아지가 있기 때문에 '강아지'라는 레이블이 붙을 수 있지만 당연히 강아지들로 가득 찬 것이은우리 모델에 더 유용할 것이라는 것은 명백합니다. 여기서 Instance Uncertainty과 Image Uncertainty을 구별해야 합니다. 따라서 MI-AOD는 Multiple Instance Learning (MIL) 모듈을 사용하여 Instance Uncertainty Learning (IUL)를 수행하여 이미지 간에 외관 일관성을 강제합니다. 그래야 레이블링되지 않은 데이터 세트에서 유용한 사진를 찾을 수 있습니다.

IUL과 IUR의 학습 절차는 거의 동일합니다. 유일한 차이점은 IUR은 Instance 레이블과 이미지 레이블 간의 일관성을 확인하려고 한다는 것입니다. 자세한 내용을 살펴보기 전에 훈련 절차를 간략히 살펴봅시다. 각 훈련 주기는 IUL과 IUR의 두 단계로 구성됩니다. 각각은 다르지만 주요 3단계로 이루어져 있다고 할 수 있습니다.

* 레이블링된 세트로 훈련
* 두 classifier 사이의 Instance Uncertainty을 최대화하기 위한 훈련
* 레이블링된 세트와 레이블링되지 않은 세트로 사이의 Instance Uncertainty을 최소화하기 위한 훈련.

### ***Instace Uncertainty Learning (IUL)***

![그림 10: IUL 훈련 process [소스: MI-AOD의 그림 3]](/.gitbook/assets/11/iul-training.png)

1. 레이블링된 세트로 훈련

    그림의 (a) 부분을 보시면, 모델는 4가지 성분을 볼 수 있습니다. ![][g]는 기본 네트워크, RetinaNet [\[lin2017\]][lin2017], feature extraction을 담당하며 ![][theta-g]는 ![][g]의 바라미터 세터입니다. 앞에서 언급했듯이, 모델은 ![][f1] 및 ![][f2]의 두 classifiers가 base 네트워크 ![][g] 위에 쌓여 있습니다. 또한 regressor ![][fr]는 bounding box 학습을 담당합니다. ![][theta-f1] 및 ![][theta-f2]가 독립적으로 초기화됩니다. 각 이미지의 detection loss:

    ![][equation1] (1),

      * FL(.)은 RetinaNet [\[lin2017\]][lin2017]에서 제안된 focal loss
      * ![][i]는 instance 번호
      * ![][yhat-f1], ![][yhat-f2] 및 ![][yhat-fr]는 ![][i] 번째의 instance의 예측

    이 단계에서는 레이블링된 세트에서만 훈련 수행됩니다. 목표는 모델이 레이블링된 데이터에 익숙해지도록 해서 나중에 레이블링되지 않은 세트에서 regularization할 수 있도록 하는 겁니다. ![][f1]과 ![][f2]는 독립적으로 초기화되었기 때문에 그들의 예측에서 약간의 불일치를 볼 수 있습니다. 그러나, 이것은 이 단계의 주요된 목적이 아닙니다.

2. 두 classifier 사이의 Instance Uncertainty을 최대화하기 위한 훈련
   
    그림의 (b) 부분에서는 레이블링되지 않은 데이터가 사용된 것이 보일 수 있습니다. 그러나 한 가지 이상한 점은 기본 네트워크의 바라미터 ![][theta-g]가 frozen된다는 것입니다. 이는 전 단계 동안 기본 네트워크가 Instance들 feature를 인식하는 것을 배웠기 때문입니다. 이제는 그것을 freeze해야 두 classifier 사이의 예측 불일치를 최대화하는 데 훈련에 집중할 수 있습니다. loss 함수는 다음과 같다.

    ![][equation2] (2)

    적대적 훈련를 위해 불일치는 다음과 같이 정의됩니다.

    ![][equation3] (3)

    그림 2를 살펴보면 두 classifier의 decision boundary가 서로 멀어지는 것을 알 수 있습니다. 

3. 레이블링된 세트와 레이블링되지 않은 세트로 사이의 Instance Uncertainty을 최소화하기 위한 훈련

    Classfier의 Instance Uncertainty을 극대화하는 훈련하기 했지만 또 다른 문제가 생겼습니다. 라벨링된 세트의 데이터 분포는 라벨링되지 않은 데이터 분포와 어느 정도 확실히 다릅니다. 항상 그럽니다. 이 문제를 해결하기 위해 classifier와 regressor를 freeze하여 기본 네트워크에만 집중합니다. Loss 함수는 다음과 같다.

    ![][equation4] (4)

### **Instance Uncertainty Re-weighting (IUR)**

![그림 11: IUR 훈련 절차 [소스: MI-AOD의 그림 4]](/.gitbook/assets/11/iur-training.png)

1. Multiple Instance Learning (MIL)

    Instance Uncertainty과 이미지 Image Uncertainty 사이에 일관성을 강요하려면 먼저 이미지를 분류할 수 있어야 합니다. 분류 점수는 다음과 같이 계산됩니다.

    ![][equation5](5)

    * ![][yhat-ic]은 ![][i] 번째의 Instance가 *c* 클래스라는 분류 점수.
    * ![][f-mil]은 multiple instance classifier.

    너무 익숙해진 softmax 함수가 나타나죠. 첫 번째 항은 ![][f-mil] 예측에 기초하여 이 이미지가 *c*에 속하는 확률입니다. 하지만 더 중요한 것은 2항입니다. ![][f1] 및 ![][f2]는 이미지에서 클래스 *c*에 속하는 많은 Instance를 찾을 수 없다면 전체 점수는 0에 가까질 겁니다. Loss 함수는 다음과 같습니다.

2. Uncertainty Re-weighting

    여기서는 전체 데이터 세트에서 Instance Uncertainty과 Image Uncertainty가 일치하도록 훈련 더 합니다. 조정된 후의 차이는 다음과 같이 같다.

    ![][equation7] (7)

    Loss 함수는:

    ![][equation8] (8)

    여기서는 ![][yhat-i-cls]를 (7)에 통합함으로써 ![][tilde-l-dis]가 각 Instance의 클래스 점수를 고려하도록 합니다. 또한 ![][theta-g]를 고정함으로써 classifier들은 분류 점수가 높은 Instance만 고려하도록 훈련하면서 ![][f1]과 ![][f2]는 여전히 Instance Uncertainty을 최대화하려고 노력합니다. 

    마지막 단계에서는 classifier와 regressor가 frozen된 상태에서 기본 네트워크를 재훈련하여 레이블링된 세트와 레이블링되지 않은 세트 사이의 분포 편향을 최소화하는 겁니다.

    ![][equation9] (9)

## **4. 실험 및 결과분석**

### **실험환경**

1. 데이터 세트

    이 모델을 훈련시키기 위해서 두 가지의 표준 Object Detection 데이터 세트를 사용했습니다.

    * PASCAL VOC 2007: Active 훈련을 위한 *trainval* 및 mAP 평가를 위한 *test* 세트 [\[voc2007\]][voc2007]
    * MS COCO: Active 훈련을 위한 *train* 세트 및 mAP 평가를 위한 *val* 세트 [\[coco2015\]][coco2015]

2. Active Learning
   
    MI-AOD의 성능을 평가하기 위해 두 가지 Object Detection 모델이 사용됩니다.

    * ResNet-50 [\[he2015\]][he2015]을 base로 사용하는 RetinaNet [\[lin2017\]][lin2017]
      * 각 사이클마다 모델은 26 Epoch 동안 0.001 학습 속도와 2의 minibatch로 훈련됩니다.
      * 20 epoch 이후 학습률은 10배 감소됩니다.
      * Momentum와 weight decay rate는 가각 0.9, 0.0001입니다.

    * VGG-16 [\[simonyan2015\]][simonyan2015]을 base로 사용하는 SSD [\[liu2016\]][liu2016]
      * 각 사이클마다 모델은 0.001 학습 속도로 240 Epoch 동안 훈련되며, 이후 나머지 60 Epoch 동안 0.0001로 훈련됩니다.
      * Minibatch 크기는 32입니다.

### **결과분석**

![그림 12: MI-AOD와 다른 방법의 결과 비교 [소스: MI-AOD의 그림 5]](/.gitbook/assets/11/performance.png)

전반적으로, 사진을 보시면 MI-AOD가 모든 레이블 설정에서 Object Detection 작업에 다른 모든 Active Learning 기법를 능가한다는 것을 알 수 있습니다. 이는 Instance Uncertainty 학습하는 것이 모델이 유용한 feature와 유익한 훈련 샘플에 집중하는 데 도움이 된다는 것을 증명합니다. 단 다른 논문에 나타나는 Object Detection 모델에 비해 AP가 매우 낮은 줄 알 수도 있는데 그 것은 MI-AOD가 VOC2007과 COCO 데이터 세트의 각각 5%와 2% 레이블링된 데이터로 훈련되었기 때문입니다.

### **Ablation Study**

Ablation Study 결과를에 제미있는 점 몇 개가 있습니다. 저는 Ablation Study를 보고 저자들의 주장이 잘 맞는지 안 맞는지를 파악하는 것이 좋다고 생각합니다.

* IUL 및 IUR

![](/.gitbook/assets/11/table1.png)

![](/.gitbook/assets/11/table2.png)

  * 표 1을 보시면 IUL과 IUR는 랜덤 샘플을 사용해도 성능이 크게 향상됩니다. 머신 런링에서는 원래 안 좋은 데이터로 학습시키면 모델 성능이 나빠질 경수도 많습니다. 그리고 랜덤 샘플링이라면 유용한 것도 안 좋은 것도 나올 수 있겠죠. 따라서 MI-AOD는 유용하지 않은 이미지을 잘 제거해서 training을 효과적으로 잘 했다는 것을 알 수 있습니다. 
  * Mean Uncertainty 샘플링의 결과가 Max Uncertainty보다 높은 것은 Instance Uncertainty를 평균화하는 것이 이미지를 더 잘 표현한다는 점을 보여줍니다.
  * 그 것은 표3에서도 보일 수 있습니다. ![][yhat-i-cls]로 저자들이 다른 유용하지 않은 클래스를 압축할 수 있었습니다.

![](/.gitbook/assets/11/table4.png)
  
* Hyper-parameters

    * 표 4는 ![][lambda]와 ![][k]의 여러의 값을 지정할 때 모델의 성능을 보여줍니다.
    * (2), (4), (8), (9)에 따르면 ![][lambda]의 값이 너무 낮으면 레이블이 없는 집합에 대한 Uncertainty 학습은 거의 영향을 미치지 않습니다.
    * 그러나 ![][lambda]를 증가시키면, 우리는 어떤 단계에 따라 Uncertainty 도을 높이거나 억제합니다. 그래서 중립값 0.5가 가장 잘 하는 이유일 수도 있다고 생각합니다. 가 단계마다 다른 ![][lambda] 값을 지정하면 재미있는 것을 볼 수 있다는 생각도 합니다.

![](/.gitbook/assets/11/table5.png)

* 표 5는 다른 두 가지 방법과 비교하여 MI-AOD의 훈련 시간을 보여줍니다.


### **모델 분석**

![그림 13: MI-AOD의 각 단계의 결과의 비주얼 분석 [소스: MI-AOD의 그림 6]](/.gitbook/assets/11/visual-analysis.png)

1. 비주얼 분석

   이 그림은 각 단계의 모델 output heatmap를 보여줍니다. 각 Heatmap는 합계된 이미지의 모든 Instance의 Uncertainty 점수입니다. Object of Interest의 주위에 유용한 feature가 많아서 Object에 가까울수록 점수가 더 높습니다.

2. 통계적 분석

   그림 14는 각 방법의 정확한 Instance의 수를 보여줍니다.

![그림 14: MI-AOD와 다른 방법의 결과의 통계학적 분석 [소스: MI-AOD의 그림 7]](/.gitbook/assets/11/stat-analysis.png)

## **5. 결론**

솔직히 말하면, 저는 Instance Uncertainty에 기반한 Object Detection를 위한 학습 방법을 제안한 이 논문을 정말 재미있게 잘 정말 읽었습니다. IUL와 IUR의 Uncertainty를 학습기키기 위해 네트워크의 각 부분을 freeze하는 훈련 방번을 잘 이해하면 너무 좋다고 생각합니다. 

하나, 이 논문의 단점도 몇 개가 있는 것 같습니다. 예를 들어 학습 절차에서 단계가 많죠. 그 것 때문에 hyper-parameters의 수도 많아질 겁니다. 또한 저자들이 수집된 결과에 따라 자기의 생각이 얼마나 잘 맞는지에 대해 더 설명했으면 좋겠습니다.

### **오늘의 교훈**

        Softmax 확률은 예측의 확신과 다른 것입니다..

        머신에게도 인간처럼 자기 잘 모르는 것을 더 깊게 학습하는 것은 좋을 것 같습니다.

## Author/Reviewer Information

### Author
* 소속: KAIST 전산학부
* 연락처:
  * 이메일: tungnt at kaist.ac.kr
  * Github: tungnt101294

### Reviewer
1. a
2. b
3. c

## References and Additional Materials

### References

[\[humanactivelearn\]][humanactive1] “자기주도학습의 정의” https://slc.gangdong.go.kr:9443/front/intropage/intropageShow.do?page_id=f637a1b9b0ec443bbcd15ec58ca3bb97. (accessed Oct. 23, 2021).

[humanactive1]: https://www.queensu.ca/teachingandlearning/modules/active

[\[mitlecture\]][mitlecture] 
A. Amini, “MIT 6.S191: Evidential Deep Learning and Uncertainty.” https://www.youtube.com/watch?v=toTcf7tZK8c&t=2061s (accessed Oct. 23, 2021).

[mitlecture]: https://www.youtube.com/watch?v=toTcf7tZK8c&t=2061s

[\[ulkumen-uncertainty\]][ulkumen-uncertainty] C. R. Fox and G. Ulkumen, “Distinguishing Two Dimensions of Uncertainty,” SSRN Electron. J., 2021, doi: 10.2139/ssrn.3695311.

[ulkumen-uncertainty]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3695311

[\[lewis1994a\]][lewis1994a] D. D. Lewis and J. Catlett, “Heterogeneous Uncertainty Sampling for Supervised Learning,” Mach. Learn. Proc. 1994, pp. 148–156, 1994, doi: 10.1016/b978-1-55860-335-6.50026-x.

[lewis1994a]: http://www.cs.cornell.edu/courses/cs6740/2010fa/papers/lewis-catlett-uncertainty-sampling.pdf

[\[lewis1994b\]][lewis1994b] D. D. Lewis and W. A. Gale, “A sequential algorithm for training text classifiers,” Proc. 17th Annu. Int. ACM SIGIR Conf. Res. Dev. Inf. Retrieval, SIGIR 1994, pp. 3–12, 1994, doi: 10.1007/978-1-4471-2099-5_1.

[lewis1994b]: https://arxiv.org/pdf/cmp-lg/9407020.pdf

[\[roth2006\]][roth2006] D. Roth and K. Small, “Margin-based active learning for structured output spaces,” Lect. Notes Comput. Sci. (including Subser. Lect. Notes Artif. Intell. Lect. Notes Bioinformatics), vol. 4212 LNAI, pp. 413–424, 2006, doi: 10.1007/11871842_40.

[roth2006]: https://doi.org/10.1007/11871842_40

[\[joshi2010\]][joshi2010] A. J. Joshi, F. Porikli, and N. Papanikolopoulos, “Multi-class active learning for image classification,” 2009 IEEE Conf. Comput. Vis. Pattern Recognit., pp. 2372–2379, 2010, doi: 10.1109/cvpr.2009.5206627.

[joshi2010]: https://doi.org/10.1109/cvpr.2009.5206627

[\[lou2013\]][lou2013] W. Luo, A. G. Schwing, and R. Urtasun, “Latent structured active learning,” Adv. Neural Inf. Process. Syst., pp. 1–9, 2013.

[lou2013]: https://papers.nips.cc/paper/2013/hash/b6f0479ae87d244975439c6124592772-Abstract.html

[\[settles2008\]][settles2008] B. Settles and M. Craven, “An analysis of active learning strategies for sequence labeling tasks,” EMNLP 2008 - 2008 Conf. Empir. Methods Nat. Lang. Process. Proc. Conf. A Meet. SIGDAT, a Spec. Interes. Gr. ACL, pp. 1070–1079, 2008, doi: 10.3115/1613715.1613855.

[settles2008]: https://www.biostat.wisc.edu/~craven/papers/settles.emnlp08.pdf

[\[settles2007\]][settles2007] B. Settles, M. Craven, and S. Ray, “Multiple-instance Active Learning,” in NIPS, 2007, pp. 1289–1296.

[settles2007]: https://dl.acm.org/doi/10.5555/2981562.2981724%0A%0A.

[\[roy2007\]][roy2007] N. Roy, A. Mccallum, and M. W. Com, “Toward optimal active learning through monte carlo estimation of error reduction.,” Proc. Int. Conf. Mach. Learn., pp. 441–448, 2001.

[roy2007]: https://dl.acm.org/doi/10.5555/645530.655646

[\[carbonneau2017\]][carbonneau2017] M. Carbonneau, E. Granger, and G. Gagnon, “Bag-Level Aggregation for Multiple Instance Active Learning in Instance Classification Problems,” arXiv, 2017.

[carbonneau2017]: https://arxiv.org/abs/1710.02584

[\[wang2017\]][wang2017] R. Wang, X. Z. Wang, S. Kwong, and C. Xu, “Incorporating Diversity and Informativeness in Multiple-Instance Active Learning,” IEEE Trans. Fuzzy Syst., vol. 25, no. 6, pp. 1460–1475, 2017, doi: 10.1109/TFUZZ.2017.2717803.

[wang2017]: https://ieeexplore.ieee.org/document/7953641

[\[hwang2017\]][hwang2017] S. Huang, N. Gao, and S. Chen, “Multi-Instance Multi-Label Active Learning,” in International Joint Conference on Artificial Intelligence, 2017, pp. 1886–1892.

[hwang2017]: https://www.ijcai.org/proceedings/2017/0262.pdf

[\[sener2018\]][sener2018] O. Sener and S. Savarese, “Active Learning for Convolutional Neural Networks: A Core-Set Approach,” in ICLR, 2018, pp. 1–13.

[sener2018]: https://arxiv.org/abs/1708.00489

[\[lin2017\]][lin2017] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, “Focal Loss for Dense Object Detection,” arXiv, Aug. 2017, [Online]. Available: http://arxiv.org/abs/1708.02002.

[lin2017]: https://arxiv.org/abs/1708.02002

[\[voc2007\]][voc2007] E. M., V.-G. L., W. C. K. I., W. J., and Z. A., “The Pascal Visual Object Classes (VOC) Challenge,” Int. J. Comput. Vis., vol. 88, no. 2, pp. 303–338, 2010.

[voc2007]: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/

[\[coco2015\]][coco2015] T. Lin et al., “Microsoft COCO: Common Objects in Context,” arXiv, pp. 1–15, May 2015.

[coco2015]: http://arxiv.org/abs/1405.0312

[\[he2015\]][he2015] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in CVPR, 2016, pp. 770–778, doi: 10.1002/chin.200650130.

[he2015]: https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html

[\[simonyan2015\]][simonyan2015] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in 3rd International Conference on Learning Representations, ICLR 2015 - Conference Track Proceedings, 2015, pp. 1–14.

[simonyan2015]: https://arxiv.org/pdf/1409.1556.pdf

[\[liu2016\]][liu2016] W. Liu et al., “SSD: Single Shot MultiBox Detector,” in ECCV, 2016, vol. 9905, pp. 21–37, doi: 10.1007/978-3-319-46448-0.

[liu2016]: https://link.springer.com/chapter/10.1007/978-3-319-46448-0_2


[iur]: /.gitbook/assets/11/iur.png


[g]: /.gitbook/assets/11/equations/g.png
[i]: /.gitbook/assets/11/equations/i.png
[yhat-f1]: /.gitbook/assets/11/equations/yhat-f1.png
[yhat-f2]: /.gitbook/assets/11/equations/yhat-f2.png
[yhat-fr]: /.gitbook/assets/11/equations/yhat-fr.png
[equation1]:/.gitbook/assets/11/equations/equation1.png
[equation2]:/.gitbook/assets/11/equations/equation2.png
[equation3]:/.gitbook/assets/11/equations/equation3.png
[equation4]: /.gitbook/assets/11/equations/equation4.png
[equation5]: /.gitbook/assets/11/equations/equation5.png
[equation6]: /.gitbook/assets/11/equations/equation6.png
[equation7]: /.gitbook/assets/11/equations/equation7.png
[equation8]: /.gitbook/assets/11/equations/equation8.png
[equation9]: /.gitbook/assets/11/equations/equation9.png
[equation10]: /.gitbook/assets/11/equations/equation10.png
[theta-g]: /.gitbook/assets/11/equations/theta-g.png
[f1]: /.gitbook/assets/11/equations/f1.png
[f2]: /.gitbook/assets/11/equations/f2.png
[fr]: /.gitbook/assets/11/equations/fr.png
[theta-set]: /.gitbook/assets/11/equations/theta-set.png
[theta-f1]: /.gitbook/assets/11/equations/theta-f1.png
[theta-f2]: /.gitbook/assets/11/equations/theta-f2.png
[yhat-ic]: /.gitbook/assets/11/equations/yhat-ic.png
[f-mil]: /.gitbook/assets/11/equations/f-mil.png
[x-0-u]: /.gitbook/assets/11/equations/x-u-0.png
[x-0-l]: /.gitbook/assets/11/equations/x-0-l.png
[x-y-0-l]: /.gitbook/assets/11/equations/x-y-0-l.png
[x-in-x-0-l]: /.gitbook/assets/11/equations/x-in-x-0-l.png
[x-in-x-0-u]: /.gitbook/assets/11/equations/x-in-x-0-u.png
[x-set]: /.gitbook/assets/11/equations/x-set.png
[y-loc-x]: /.gitbook/assets/11/equations/y-loc-x.png
[y-cls-x]: /.gitbook/assets/11/equations/y-cls-x.png
[y-0-l]: /.gitbook/assets/11/equations/y-0-l.png
[k]: /.gitbook/assets/11/equations/k.png
[lambda]: /.gitbook/assets/11/equations/lambda.png

[yhat-i-cls]: /.gitbook/assets/11/equations/yhat-i-cls.png
[tilde-l-dis]: /.gitbook/assets/11/equations/tilde-l-dis.png
[table1]: /.gitbook/assets/11/table1.png
[table2]: /.gitbook/assets/11/table2.png
[table3]: /.gitbook/assets/11/table3.png
[table4]: /.gitbook/assets/11/table4.png
[table5]: /.gitbook/assets/11/table5.png