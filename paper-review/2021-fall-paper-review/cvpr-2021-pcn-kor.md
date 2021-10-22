---
Yang et al. / Progressively Complementary Network for Fisheye Image Rectification Using Appearance Flow / 2021 CVPR Oral
---

# Progressively Complementary Network for Fisheye Image Rectification Using Appearance Flow [Kor]

##  1. Problem definition

시야각이 넓은 광각렌즈 또는 초광각 렌즈를 사용하여 이미지를 촬영하면 넓은 범위를 볼 수 있지만, 아래 그림처럼 상대적으로 이미지 가장자리의 왜곡이 심해지는 문제가 발생합니다.

<img src="../../.gitbook/assets/view-fisheye.jpg" alt="그림 1: 초광각 렌즈로 촬영한 이미지" style="zoom: 33%;" />

이러한 왜곡을 어안 왜곡(fisheye distortion)이라고 부르는데요. 이러한 영상 왜곡은 심미적인 문제 외에도, 영상을 분석하여 좌표 추정등의 정확한 수치계산이 필요할 때 큰 문제가 될 수 있습니다. 따라서 이러한 왜곡을 보정하는 문제는 여러 컴퓨터 비전 기술들을 적용하기 위해서 해결해야 할 부분이라고 볼 수 있습니다.

기존의 영상처리 분야에서는 이러한 어안 렌즈 왜곡 문제를 camera calibration을 활용하여 보정하였습니다. 하지만 이러한 방식은 이미지에서 3D 좌표를 추정할 수 있도록 이미지내에 chessboard가 함께 촬영되어야 하는 등의 추가적인 노력이 필요한 경우가 많습니다. 따라서 주어진 이미지만을 가지고 보정을 수행할 수 있는 automatic correction 방법 또한 여러 개가 개발되었지만, 카메라 내 왜곡을 측정하기 위해 여러 특징들을 추출하는 과정이 불완전하여 왜곡 보정이 잘 안되는 경우가 종종 발생하고는 합니다.

딥러닝의 발전에 힘입어, 최근에는 이러한 이미지의 어안 왜곡 문제를 딥러닝으로 해결하려는 시도가 늘고 있으며 여러 논문들이 발표가 되고 있습니다. 이번 글에서는 그 중 최근 CVPR'21에서 발표된 [Progressively Complementary Network for Fisheye Image Rectification Using Appearance Flow](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Progressively_Complementary_Network_for_Fisheye_Image_Rectification_Using_Appearance_Flow_CVPR_2021_paper.pdf)라는 논문을 소개해볼까 합니다.

## 2. Motivation

### Related work

전통적인 영상처리 분야에서 제안된 왜곡 보정 방식들의 단점을 보완하고자, 딥러닝 기반의 왜곡 보정 방식이 많이 발표되었습니다. 이러한 딥러닝 기반 접근 방법들은 크게 pixel을 옮기고자 하는 목표 좌표를 추정하는 regression-based 방식과, 아예 왜곡이 보정된 이미지를 생성하는 생성 모델 기반의 generation-based 방식으로 나눌 수 있습니다.

#### Regression-Based Method

Regression-Based Method는 Non-linear한 예측을 수행할 수 있는 CNN의 능력을 활용하여 pixel단위의 image-to-image translation map을 추정하는 방법입니다.

16년에 발표된 [[1](##Reference & Additional materials)]은 CNN을 활용하여 어안 렌즈 왜곡 문제를 푼 선구자적인 논문입니다. 하지만 단순한 설계로 인해 복잡한 왜곡이미지는 제대로 보정을 하지 못하는 한계를 보였습니다.

ECCV'18에서 발표된 [[2](##Reference & Additional materials)]는 3개의 CNN 모델을 조합, 여러 semantic feature를 추출하여 보정에 사용하는 모델을 제안하였습니다. 하지만 이렇게 추출된 semantic feature들을 활용하려다보니 입력되는 이미지의 차원수가 늘어났고, 이는 이미지를 보정하는데 제약이 될 수밖에 없었습니다. 

CVPR'19에서 발표된 [[3](##Reference & Additional materials)]은 딥러닝 모델에 geometry constraints을 부과하는 보조 module을 설계하여, 곡선 형태의 왜곡이 좀 더 직선형태로 보정 될 수 있도록 하는 모델을 제안하였습니다. 이 모델은 기존의 두 모델보다 더 나은 성능을 보였지만, 학습을 위해서는 기 제안된 방법들에 비해 상당한 양의 정답 데이터(e.g. edge labels, distortion parameter labels, normal images)를 필요로 하는 단점이 존재하며, 보조 module은 미리 학습(pre-trained)되어야 할 필요가 있어 학습 또한 기존 모델에 비해 복잡한 단점 또한 존재합니다.

#### Generation-Based Method

Generation-Based Method는 생성모델, 그 중에도 GAN을 활용하여 왜곡이 보정된 이미지를 직접 생성하는 방법입니다.

19년에 발표된 [[4](##Reference & Additional materials)]는 DR-GAN이라는 이름이 모델을 제안했는데, 이는 adversarial framework를 사용하여 왜곡을 보정하는 첫 번째 논문이였습니다. 따라서 모델이 왜곡된 이미지와 정상 이미지의 이미지 분포 패턴을 직접 학습하게 됩니다. 또한 정상 이미지 외에는 추가 label이 필요가 없는 점 또한 장점으로 볼 수 있습니다. 하지만, 이 모델은 이미지의 구조(structure)와 이미지의 내용(content)를 동시에 재생성해내야 하기 때문에, 모델에 부담이 커지는 구조로 설게되어 있습니다. 그로 인해, 재생성된 이미지가 뿌옇게 보이는(blurred) 경우가 존재하고, 왜곡이 보정되는 정도 또한 만족스럽지 못한 경우가 있습니다.

DR-GAN을 발표한 연구팀은 20년에 발표된 [[5](##Reference & Additional materials)]를 발표하는데, 해당 논문은 어안 왜곡뿐만 아니라 다양한 렌즈왜곡을 처리할 수 있는 왜곡 보정 프레임워크(a model-free distortion rectification framework)를 제안하는 논문입니다. 이는 기존의 방식에 비해 좀더 구조적인 왜곡을 잘 보정하였지만, 이미지의 디테일한 부분을 잘 재현하지 못하였고, 논문에서 도입한 skip-connection으로 인해 왜곡이 퍼지는 현상(distortion diffusion)이 나타났습니다.

### Idea

앞서 살펴본 Generation 기반 방법들은 크게 2가지 문제가 관찰되었는데, **1. skip-connection으로 인해 왜곡 정보를 가진 이미지의 feature들이 곧바로 전파되는 문제**, **2. 이미지를 재현하는 decoder가 구조적인 왜곡에 대한 보정과 컨이미지 내용에 대한 재현을 동시에 수행함으로써 성능이 떨어지는 문제**가 그것입니다.  본 논문에서 **PCN(Progressively Complementary Network)**이라는 모델을 제안하여 기존의 GAN 기반의 방식에서 나타난 이 2가지 단점들을 보완하고자 하였습니다.

본 논문에서 제안하는 문제의 해결책은 다음과 같습니다.

1. Feature correction layer를 skip-connection에 삽입함으로써, decoder가 좀 더 그럴듯한 결과를 만들어낼 수 있도록 합니다.
2. 이미지의 왜곡된 구조 측정을 전담하는 Flow estimation module을 추가하였습니다. 이를 통해 decoder의 부담을 줄여 최종 결과의 성능을 높였습니다.
3. 저자들은 또한 이미지의 왜곡이 Encoder의 레이어를 통과할때마다, 왜곡이 조금씩 보정되는 현상을 발견했다고 합니다. 본 논문에서는 Gradual generation characteristic이라고 명명된 이 특성을 적극 활용할 수 있도록 모델 구조를 설계하였습니다.

## 3. Method

해당 모델을 학습하기 위한 합성 데이터를 생성하는 방법을 먼저 소개하고, 이후 모델 설계 구조와 학습 방법에 대해 설명하도록 하겠습니다.

### Fisheye Models for Synthetic Data

```
해당 파트는 카메라 투영 모델에 대한 지식이 없다면 이해가 힘들 수 있습니다.
```

어안 왜곡 이미지 데이터를 실제로 수집하기에는 많은 cost가 소요됨으로 현실적이지 않습니다. 다행히 어안렌즈 왜곡을 인위적으로 생성할 수 있는 수학적 모델이 이미 개발되어 있습니다. 정상이미지를 통해 어안 렌즈 왜곡이미지를 생성하는 방식은 일반적으로 division model과 polynomial model, 이 2가지 방법이 가장 많이 사용됩니다.

이미지 좌표계에서, 임의의 점 $P_u(x,y)$와 이미지의 중심점 $P_0(x_0,y_0)$ 의 유클리디언 거리를 $r_u$라고 정의합니다. 또한 $P_u(x,y)$는 왜곡된 이미지에서 $P_d(x_d,y_d)$의 위치에 존재한다고 가정합니다. 그리고 왜곡된 이미지에서, 점 $P_d$와 왜곡된 이미지의 중심점과의 유클리디언 거리를 $r_d$라고 정의합시다.

Division model은 $r_u$와 $r_d$사이의 관계를 다음과 같이 정의합니다.
$$
r_u = \frac{r_d}{1+\sum_{i=1}^{n}{k_i r_d^{2i-1}}}
$$

이 때, $k_i$는 distortion parameter입니다. 이 값을 조절함으로써 어안 왜곡의 정도를 조절할 수 있습니다. 또한 $n$은 parameter의 수를 나타냅니다. $n$이 커질수록, 복잡한 왜곡 상태를 표현할 수 있게 됩니다.

Polynomial model은 입사광선의 각도를 수식에 포함시킨 모델입니다. 다음과 같이 표현됩니다.
$$
\theta_u = \sum_{i=1}^{n}{k_i \theta_d^{2i-1}}
$$
$\theta_u$는 입사광선의 입사각을 의미하며, $\theta_d$는 렌즈를 통과한 빛의 각도입니다. 일반적으로 $r_d$와 $\theta_d$는 $r_d=f\theta_d$(이 때, $f$는 어안 렌즈 카메라의 focal length)가 성립하는데, 이를 등거리 투영관계(the equidistant projection relation)를 만족한다고 표현합니다. 가장 기본적인 카메라 모델(Pinhole 카메라 모델)에서는, $r_u=f\tan\theta_u$의 투영모델을 사용합니다.

간략화를 위해 $\theta_u=\arctan(\frac{r_u}{f})\approx\frac{r_u}{f}$ 라고 가정합니다. 그러면, polynomial model에서 $r_u$와 $r_d$사이의 관계를 다음과 같이 계산할 수 있습니다.
$$
r_u = f\sum_{i=1}^{n}{k_i r_d^{2i-1}}
$$
$k_i$와 $f$ 둘 다 사용자가 설정해야 하는 값이므로, 이 둘을 합치면 최종적으로 다음의 수식을 얻을 수 있습니다.
$$
r_u = \sum_{i=1}^{n}{k_i r_d^{2i-1}}
$$
본 논문에서는 polynomial model을 활용하여 fisheye iamge를 합성합니다.

### Network Architecture

앞서 기존의 generation 기반 방식들의 모델이 하나의 decoder에서 이미지의 구조적인 왜곡 보정과 이미지의 컨텐츠 재생성을 동시에 수행하려고 하다 보니 성능이 떨어진다는 점을 지적하였는데요. 본 논문에서는 이를 해결하기 위해, 왜곡된 구조 보정(structure correction)과 이미지의 컨텐츠 재생성(content reconstruction)을 담당하는 모듈을 나누어 설계하여 해당 역할을 나누고자 하였습니다.

<img src="../../.gitbook/assets/architecture.jpg" alt="그림 2: PCN 모델 구조"  />

입력으로는 256 x 256 크기의 이미지를 사용하는데, 해당 이미지를 동시에 2개의 모듈에 입력합니다. 그림의 위에 위치하는 Flow Estimation 모듈은 왜곡된 정도를 측정하고, 왜곡을 보정하기 위해서 pixel들이 어디로 이동해야 하는지를 나타내는 appearance map을 출력합니다. 밑에 위치한 Distortion Correction Module은 실질적으로 이미지를 재생성해내는 모듈입니다. 위에서 출력되는 appearance map을 활용하여 구조적인 왜곡 보정의 도움을 받아 보정된 이미지를 재생성하게 됩니다.

#### Appearance Flow Estimation Module

#### Feature Correction Layer

#### Progressively Complementary Mechanism

#### Distortion Correction Module

### Training strategy

#### Reconstruction Loss

#### Adversarial Loss

#### Enhanced Loss


## 4. Experiment & Result

This section should cover experimental setup and results.  
Please focus on how the authors of paper demonstrated the superiority / effectiveness of the proposed method.

Note that you can attach tables and images, but you don't need to deliver all materials included in the original paper.

### Experimental setup

This section should contain:

* Dataset
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

> All men are mortal.
>
> Socrates is a man.
>
> Therefore, Socrates is mortal.

## Author / Reviewer information

### Author

**진영화 \(Jin YeongHwa\)** 

* Affiliation \(KAIST AI / NAVER\)
* Machine Learning Engineer @ NAVER Papago team
* https://www.linkedin.com/in/yeonghwa-jin-66b241106/

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. [J. Rong, S. Huang, Z. Shang, and X. Ying. Radial lens distortion correction using convolutional neural networks trained with synthesized images. In ACCV, 2016](https://link.springer.com/chapter/10.1007/978-3-319-54187-7_3)
2. [X.Yin, X. Wang, J. Yu, M. Zhang, P. Fua, and D. Tao. Fisheyerecnet: A multi-context collaborative deep network for fisheye image rectification. In ECCV, pages 475–490, 2018.](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xiaoqing_Yin_FishEyeRecNet_A_Multi-Context_ECCV_2018_paper.pdf)
3. [Z. Xue, N., G. Xia, and W. Shen. Learning to calibrate straight lines for fisheye image rectification. CVPR, pages 1643–1651, 2019.](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xue_Learning_to_Calibrate_Straight_Lines_for_Fisheye_Image_Rectification_CVPR_2019_paper.pdf)
4. [K. Liao, C. Lin, Y. Zhao, and M. Gabbouj. DR-GAN: Automatic radial distortion rectification using conditional GAN in real-time. IEEE Transactions on Circuits and Systems for Video Technology, 2019](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8636975)
5. [K. Liao, C. Lin, Y. Zhao, and M. Xu. Model-free distortion rectification framework bridged by distortion distribution map. IEEE Transactions on Image Processing, 29:3707– 3718, 2020. 1, 2, 3, 6, 7](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8962122)
6. Official \(unofficial\) GitHub repository
7. Citation of related work
8. Other useful materials
9. https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=sinachoco&logNo=221103182738
10. https://blog.daum.net/kim1951a/1155
11. https://darkpgmr.tistory.com/31
