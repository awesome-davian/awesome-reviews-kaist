---
description: Ge et al / YOLOX: Exceeding YOLO Series in 2021 / ArXiv 2021
---

# YOLOv4 [Kor]

##  1. Problem definition

![Figure 1: Semantic segmentation.](https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/assets/demo.png)
 <center>Figure 1. YOLOX 활용 예시 </center>
 

## 2. Motivation

### Related work

<!-- Please introduce related work of this paper. Here, you need to list up or summarize strength and weakness of each work. -->

Nevertheless, over the past two years, the major ad- vances in object detection academia have focused on anchor-free detectors [29, 40, 14], advanced label assign- ment strategies [37, 36, 12, 41, 22, 4], and end-to-end (NMS-free) detectors [2, 32, 39]. These have not been inte- grated into YOLO families yet, as YOLOv4 and YOLOv5 are still anchor-based detectors with hand-crafted assigning rules for training.

- YOLO 시리즈들은 실시간 이미지 처리를 위해서 최적의 Speed / Accuracy Trade-off를 가지게끔 설계되곤 했었다.
- 최근에 YOLOV5 모델의 경우 13.7ms 만에 48.2% AP를 가지는 최적의 Trade Off를 가지고 있다.
- 뿐만 아니라, 2년간 Object Detection 기술이 발전하면서 학계에서는 Anchor Free Detector, Advanced Label Assignment Strategies, End-to-end Detector등 다양한 기법을 연구해나가고 있다.
- YOLO 시리즈는 아직까지 이러한 최신 기법들이 적용된 사례가 존재하지 않으며, 따라서 본 논문에서 저자들은 YOLO 에 이러한 최신 기법들을 적용하는 시도를 진행하였다.
- YOLOv4와 YOLOv5의 파이프라인은 Anchor Based 위주로 최적화가 진행되어있기 때문에, General 한 성능이 오히려 떨어질 수 있다고 생각하여 본 논문의 저자들은 YOLOv3-SPP를 기본 베이스 모델로 삼았다.
- 다양한 기법을 적용한 결과, Baseline에 비해서 AP가 크게 개선되었으며, 기존에 공개됬던 ultralytics의 YOLOv3 보다도 높은 성능을 지닌다.

### Idea

After you introduce related work, please illustrate the main idea of the paper. It would be great if you describe the idea by comparing or analyzing the drawbacks of the previous work.

## 3. Method

- YOLOX는 기본적으로 1 Stage Detector로 Input - Backbone - Neck - Dense Prediction의 구조를 가진다.
- YOLOX는 Darknet53의 Backbone을 통해 Feature Map을 추출하며, SPP Layer를 통해 성능을 개선한다.
- FPN을 통해 Multi-Scale Feature Map을 얻고 이를 통해 작은 해상도의 Feature Map에서는 큰 Object를 추출하고 큰 해상도의 Feature Map에서는 작은 Object를 추출하게끔 한 Neck 구조를 차용하였다.
- 최종적으로 Head 부분에서는 기존 YOLOv3와 달리 Decoupled Head를 사용하였다.

The proposed method of the paper will be depicted in this section.

Please note that you can attach image files \(see Figure 1\).  
When you upload image files, please read [How to contribute?](../../how-to-contribute.md#image-file-upload) section.

![Figure 1: You can freely upload images in the manuscript.](../../.gitbook/assets/how-to-contribute/cat-example.jpg)

We strongly recommend you to provide us a working example that describes how the proposed method works.  
Watch the professor's [lecture videos](https://www.youtube.com/playlist?list=PLODUp92zx-j8z76RaVka54d3cjTx00q2N) and see how the professor explains.

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

> 
>

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
2. Official \(unofficial\) GitHub repository https://github.com/Megvii-BaseDetection/YOLOX
3. Citation of related work
4. Other useful materials
5. ...

