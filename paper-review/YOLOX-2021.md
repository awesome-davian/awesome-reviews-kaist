---
description: Ge et al / YOLOX: Exceeding YOLO Series in 2021 / ArXiv 2021
---

# YOLOv4 [Kor]

##  1. Problem definition

![Figure 1: Semantic segmentation.](https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/assets/demo.png)
 <center>Figure 1. YOLOX 활용 예시 </center>

Real-Time Object Detection(실시간 객체 감지)는 기본 수준의 정확도를 유지하면서 실시간으로 객체 감지를 빠르게 수행하는 작업이며, 기존 Object Detection의 방법보다 월등히 빠른 처리 속도가 요구됩니다. Real-Time Object Detection 관련 모델은 이미지 classification과 localization 의 multi-task로 정의되었던 기존 Object Detection 을 하나의 regression 문제로 재해석하여 단일 신경망 구조로 개선한 YOLO(You Only Look Once, CVPR 2016) 모델이 가장 대표적입니다. 이후 YOLO 모델은 여러 시리즈로 이어지면서 실시간 이미지 처리를 위해서 최적의 Speed / Accuracy Trade-off를 가지게끔 설계되곤 했습니다. YOLOv5 모델의 경우 13.7ms 만에 48.2% AP를 가지는 최적의 Trade Off를 가지고 있습니다.

한편, 최근 학계에서는 anchor-free detectors, advanced label assignment strategies, end-to-end (NMS-free) detectors 등 다양한 object detection 기법이 새로 제시되었지만, 기존 YOLO 시리즈에 적용되지는 않았습니다. 본 논문은 이러한 기법들을 기존 YOLO 모델을 개선시키는 데에 적용하고 성능을 개선한 모델인 'YOLOX'을 제안하고 있습니다.

## 2. Motivation

### Related work

<!-- Please introduce related work of this paper. Here, you need to list up or summarize strength and weakness of each work. -->

YOLO (You Only Look Once) model은 Josept Redmon이 2015년 공개한 version 1 을 시작으로 version 5까지 진행 중에 있습니다. YOLO model의 핵심 아이디어는 classification 과 localization 을 별도의 task로 분리하지 않고, 하나의 regression problem 으로 CNN을 실시간으로 적용한 것입니다. 이름에서 알 수 있듯이, 이 알고리즘은 객체를 감지하기 위해 신경망의 단일 순방향 전파만 요구됩니다. YOLO 알고리즘의 원리는 세 가지로 구성됩니다.
1. **Residual blocks**: 이미지를 동일한 차원의 그리드 셀로 나누고, 모든 그리드 셀은 그 안에 나타나는 개체를 감지합니다. 예를 들어, 객체 중심이 특정 그리드 셀 내에 나타나면 해당 셀이 이를 감지합니다.
2. **Bounding box regression**: Bounding box는 이미지 내 객체를 강조하여 표시하는 윤곽선으로, width ($bw$) / height ($bh$) / class ($c$) / bounding box center($bx, by$)로 구성됩니다. YOLO는 Bounding box regression을 사용하여 object 의 width, height, class 및 center 를 예측하여 이미지 내 object가 나타날 확률을 나타냅니다.
3. **Intersection over union (IOU)**: Intersection Over Union는 bounding box가 겹치는 방식을 표현하는 object detection 의 현상입니다. YOLO는 IOU를 사용하여 개체를 완벽하게 둘러싸는 출력 상자를 제공합니다.

주요 YOLO 시리즈의 계보 및 핵심은 아래와 같습니다.

<p align="center">
    <img src="https://pjreddie.com/media/image/map50blue.png" alt="drawing" width="500"/>
</p>

- YOLOv3
  - 2018년 4월 발표. Joseph Redmon 이 마지막으로 발표한 YOLO 모델이며, Darknet 53을 기반으로 개발되었습니다. 
- YOLOv4
  - 2020년 4월 발표. Alexey Bochkousky 로 연구자가 바뀌었으며, 다양한 딥러닝 기법(WRC, CSP ...) 등을 사용해 v3에 비해 AP, FPS가 각각 10%, 12%가 증가하였습니다. CSPNet 기반의 backbone(CSPDarkNet53)을 설계하여 사용했습니다.
- YOLOv5
  - 2020년 6월 발표. Glenn Jocher가 발표했으며, v4와 같은 CSPNet 기반의 backbone을 설계하여 사용했고 성능은 비슷하나 경량화된 모델 크기와 속도 면에서 우수합니다. 다만 공식적인 논문으로 발표되지 않고 pytorch 코드 공개만으로 그쳐 공식적인 v5로 명칭을 붙이기에는 논란이 있습니다.
- PP-YOLO
  - 2020년 7월 발표. Shing Long이 발표했으며, v4보다 정확도와 속도가 더 높습니다. v3 모델을 기반으로 하나, Darknet3 backbone을 ResNet 으로 교체했으며 오픈소스 machine learning framework인 PaddlePaddle 기반으로 개발되었습니다. 

### Idea

<!-- After you introduce related work, please illustrate the main idea of the paper. It would be great if you describe the idea by comparing or analyzing the drawbacks of the previous work. -->

최근 학계에서는 anchor-free detectors, advanced label assignment strategies, end-to-end (NMS-free) detectors 등 다양한 object detection 기법이 새로 제시되었지만, 기존 YOLO 시리즈에 적용되지는 않았습니다. 본 논문은 이러한 기법들을 기존 YOLO 모델을 개선시키는 데에 적용하고 성능을 개선한 모델인 'YOLOX'을 제안하고 있습니다.
YOLOv4와 YOLOv5의 파이프라인은 Anchor Based 위주로 최적화가 진행되어있기 때문에, General 한 성능이 오히려 떨어질 수 있다고 생각하여 본 논문의 저자들은 YOLOv3-SPP를 베이스 모델로 삼았습니다.

## 3. Method

<p align="center">
    <img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2021-08-26_at_2.55.44_PM_JVfxCw7.png" alt="drawing" width="500"/>
</p>

- YOLOX는 기본적으로 1 Stage Detector로 Input - Backbone - Neck - Dense Prediction의 구조를 가진다.
- YOLOX는 Darknet53의 Backbone을 통해 Feature Map을 추출하며, SPP Layer를 통해 성능을 개선한다.
- FPN을 통해 Multi-Scale Feature Map을 얻고 이를 통해 작은 해상도의 Feature Map에서는 큰 Object를 추출하고 큰 해상도의 Feature Map에서는 작은 Object를 추출하게끔 한 Neck 구조를 차용하였다.
- 최종적으로 Head 부분에서는 기존 YOLOv3와 달리 Decoupled Head를 사용하였다.

The proposed method of the paper will be depicted in this section.

Please note that you can attach image files \(see Figure 1\).  
When you upload image files, please read [How to contribute?](../../how-to-contribute.md#image-file-upload) section.

![Figure 1: You can freely upload images in the manuscript.](../../.gitbook/assets/how-to-contribute/cat-example.jpg)

## 4. Experiment & Result

{% hint style="info" %}
If you are writing **Author's note**, please share your know-how \(e.g., implementation details\)
{% endhint %}

This section should cover experimental setup and results.  
Please focus on how the authors of paper demonstrated the superiority / effectiveness of the proposed method.

Note that you can attach tables and images, but you don't need to deliver all materials included in the original paper.

### Experimental setup

This section should contain:

* Dataset
  * 
* Baselines
* Training setup
* Evaluation metric
* ...

### Result

Please summarize and interpret the experimental result in this subsection.

## 5. Conclusion

In conclusion, please sum up this article.  
You can summarize the contribution of the paper, list-up strength and limitation, or freely tell your opinion about the paper.

### Take home message \(오늘의 교훈\)

Please provide one-line \(or 2~3 lines\) message, which we can learn from this paper.

> ㅁㅁㅁㅁㅁ

## Author / Reviewer information

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

1. Citation of this paper
2. [Official GitHub repository](https://github.com/Megvii-BaseDetection/YOLOX)
3. Citation of related work
4. Other useful materials
5. ...

