---
description: Huang et al. / Densely Connected Convolutional Networks / CVPR 2017
---

# DenseNet \[Kor\]

##  1. Problem definition

본 논문에서 해결하고자 하는 문제는 visual object recognition 입니다.

이는 기존에 많은 논문에서 다뤘고 현재까지도 꾸준히 다뤄지는 기초적이지만 중요한 task 중에 하나로, 주어진 image가 어떤 object를 담고 있는지 classify하는 문제라고 할 수 있습니다. 수식으로 예를 들면, model $$F$$ 가 $$x_{in} \in \mathbb{R}^{C \times W \times H}$$ 를 input으로 받아 output으로 $$y=F(x_{in}) \in [0, n_{classes}-1]$$ 를 생성하여 해당 input이 어떤 object에 해당하는지 classify하는 문제입니다.

이러한 문제를 해결하기 위해 VGGNet과 같이 깊은 신경망 구조를 갖는 model들이 제안되었고 뛰어난 성능을 보이는 것이 입증되었지만, 신경망이 깊어질 수록 처음 layer의 feature map이 손실되거나 역전파 과정에서 처음 gradient가 손실되는 문제가 새롭게 생겨났습니다. DenseNet은 깊은 신경망의 학습에서 나타나는 vanishing-gradient 문제를 해결하면서 feature map의 reuse를 강화할 수 있는 구조를 제안합니다.



## 2. Motivation

### Related works

깊은 신경망을 효과적으로 학습하고자 하는 기존의 연구들에 대해 간략히 짚어보도록 하겠습니다.

* **Highway Networks**

  Highway Networks는 100개가 넘는 깊은 신경망을 학습하기 위해 gating unit을 활용한 bypassing path라는 개념을 도입하였습니다. 이는 어떤 layer의 output에 trasform gate와 carry gate를 추가하여, 이 gate의 값에 따라 해당 layer가 plain layer로서의 동작과 단순히 input을 통과시키는 동작 사이에서 조절이 됩니다. 이를 수식으로 표현하면, $$\mathbf{x}$$가 layer $$H(\cdot)$$에 input으로 주어졌을 때, output $$y$$가 $$y=H(\mathbf{x}) \cdot T(\mathbf{x}) + \mathbf{x} \cdot C(\mathbf{x})$$로 계산되는 것입니다. 여기서 $$T$$와 $$C$$가 각각 transform gate와 carry gate로, 이 둘의 값에 따라 output이 $$H(\mathbf{x})$$나 $$\mathbf{x}$$에 가까워지게 됩니다.

* **ResNets**

  Highway Netwroks의 bypassing path 개념을 응용하여 pure identity mapping을 bypassing path로 사용하는 ResNet이 제안되었습니다.

* **Stochastic depth**

  Stochastic depth는 training 중에 무작위로 layer들을 drop하는 방식을 사용하여 1202개의 layer를 갖는 ResNet을 학습하는 데 성공하였습니다. 이는 단순히 깊은 신경망을 잘 학습하는 방법만을 제시한 게 아니라, 아주 깊은 신경망에 residual connection을 추가하더라도 많은 layer가 redundant하다는 것을 보여주었습니다.

### Idea

앞선 연구들이 깊은 신경망을 효과적으로 학습하기 위해 여러가지 구조와 학습 방법을 제시했지만, 공통적으로 **앞의 layer와 뒤의 layer 사이에 short path를 추가한다**는 idea를 바탕으로 하고 있습니다. 본 논문에서는 short path가 information flow를 강화하는 이점을 극대화하기 위해 모든 layer 간에 short path를 추가한 DenseNet 구조를 제시합니다.



## 3. Method

$$L$$개의 layer를 갖는 convolution network에 single image input $$\mathbf{x}_{0}$$가 주어집니다. 이 network의 $$l$$번째 layer는 non-linear transformation $$H_l({\cdot})$$을 갖는데, 이 $$H_l(\cdot)$$은 Batch Normalization이나 ReLU, Pooling, Convolution 등의 oepration이 합쳐진 함수를 나타냅니다. 각 $$l$$번째 layer는 $$\mathbf{x}_l$$을 output으로 만들어냅니다.



### ResNets

우선 short path 개념을 도입한 기존 연구 중에 가장 대표적이고 좋은 성능을 보이는 ResNet에 대해서 살펴보겠습니다. RenNet의 각 layer의 동작은 다음과 같이 나타낼 수 있습니다.
$$
\mathbf{x}_{l} = H_l(\mathbf{x}_{l-1})+\mathbf{x}_{l-1}
$$
$$l$$번째 layer의 output과 $$(l-1)$$번째 layer의 output (i.e. $$l$$번째 layer의 input)을 합하여 최종 output을 만들어냅니다. 이 identity function을 통해 뒤쪽 layer의 gradient가 앞쪽 layer로 잘 전달될 수 있는 장점이 있지만, $$H_l(\mathbf{x}_{l-1})$$와 identity function이 덧셈을 통해 합쳐지는 과정에서 information이 random하게 손실될 수 있다는 단점이 있습니다. 본 논문에서는 이를 "summation이 information flow를 방해한다"고 설명합니다.



### Dense connectivity

ResNet이 성능을 비약적으로 끌어올릴 수 있었던 가장 중요한 idea는 몇몇개의 layer들 사이에 skip connection을 둠으로써 information flow를 향상시킨 것입니다. DenseNet은 이를 극대화하기 위해 각 layer를 다음으로 나오는 모든 layer와 연결하였습니다. 즉, $$l$$번째 layer는 앞선 $$l-1$$개의 layer의 output을 input으로 받습니다. 이는 다음과 같이 나타낼 수 있습니다.
$$
\mathbf{x}_{l} = H_{l}([\mathbf{x}_{0}, \mathbf{x}_{1}, ... , \mathbf{x}_{l-1}])
$$
여기서 $$[\mathbf{x}_{0}, \mathbf{x}_{1}, ... , \mathbf{x}_{l-1}]$$는 $$l-1$$개의 layer의 output을 concatenate한 것으로, ResNet에서 summation을 했던 것과 구별되는 차이점입니다.

![Dense connectivity](/.gitbook/assets/2/denseconnect.png)



### Composite function

앞서 $$H_l(\cdot)$$은 여러가지 operation이 합쳐진 함수라고 말씀드렸는데, DenseNet에서는 $$H_l(\cdot)$$을 batch normalization, ReLU, 3x3 convolution이 순서대로 합쳐진 함수로 정의하였습니다. 이는 ResNet의 pre-activation 구조를 참고한 것입니다.



### Pooling layers

Dense connectivity에서 사용된 concatenation은 $$\mathbf{x}_{0}, \mathbf{x}_{1}, ... , \mathbf{x}_{l-1}$$들이 모두 같은 크기를 가질 때 가능합니다. 하지만 convolution network의 핵심 중 하나인 pooling operation은 feature map의 크기를 바꾸기 때문에 dense connectivity의 적용에 문제가 됩니다. 본 논문에서는 이를 해결하기 위해 아래 그림과 같이 전체 network를 dense connectivity가 적용된 여러 개의 dense block으로 나누고, 각 dense block 사이에서 pooling operation을 수행하도록 했습니다. 이때 각 dense block 사이의 layer들을 transition layers라고 지칭하고, 본 논문에서는 transition layers가 batch normalization layer와 1x1 convolutional layer, 2x2 pooling layer가 차례로 적용되도록 설계되었습니다.

![Dense blocks](/.gitbook/assets/2/denseblock.png)



### Growth rate

DenseNet의 각 $$H_l(\cdot)$$이 $$k$$개의 feature map (i.e. $$k$$ channels)을 만들어내고 input $$\mathbf{x}_0$$가 $$k_0$$개의 feature map을 갖고 있다고 하면, 이전의 $$l-1$$개의 layer의 output이 concatenate되어 $$k_0 + k \times (l-1)$$개의 feature map이 $$l$$번째 layer의 input으로 주어지게 됩니다. DenseNet에서는 이 $$k$$를 growth rate이라는 hyperparameter로 두고 이를 조절할 수 있도록 설계했습니다.

VGGNet 등의 network에서 각 layer들이 많게는 256개의 feature map을 output으로 만드는 반면에 DenseNet에서는 $$k=12$$만으로도 충분히 좋은 성능을 보여준다고 합니다. 또한 적은 수의 feature map을 만들어낸다는 것은 각 layer의 weight parameter의 크기가 작다는 것이기 때문에 상대적으로 적은 개수의 parameter로도 좋은 성능을 이끌어 낼 수 있다는 장점으로 연결됩니다.



### Bottleneck layers & compression

추가적으로, bottleneck layer가 추가된 DenseNet-B와 여기에 compression까지 적용된 DensNet-BC를 DensNet과 비교하는 실험을 진행했습니다.

Bottleneck layer는 ResNet이나 Inception 등에서 사용된 개념으로, 1x1 convolution을 통해 feature map의 개수 (i.e. channel의 개수)를 줄임으로써 이후의 3x3 convolution에 필요한 weight parameter의 크기를 줄여 computational efficiency를 높여주는 구조입니다. DensNet-B에서는 각 $$H_l(\cdot)$$을 BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3)로 구성하 bottleneck layer를 적용하였습니다. 이때 1x1 convolution이 4k개의 feature map을 만들어내도록 설정했다고 합니다.

Compression은 각 dense block 사이의 transition layer에 있는 1x1 convolution에서 feature map의 개수를 얼마나 줄일지 조절하는 compression factor $$\theta$$를 도입하여 model의 compactness를 향상시킬 수 있도록 한 것입니다. Compression factor $$\theta$$는 $$0 \le \theta \le 1$$의 값을 가지며, dense block의 output이 $$m$$개의 feature map을 가질 때 transition layer가 $$\lfloor \theta m \rfloor$$개의 feature map을 output으로 내도록 조절합니다. 즉, $$\theta=1$$인 경우 transition layer가 feature map의 크기를 줄이지 않습니다. 본 논문에서는 $$\theta = 0.5$$로 설정하여 compression을 적용한 DenseNet을 DenseNet-C라고 하고, bottleneck layer와 함께 적용된 DenseNet을 DenseNet-BC라고 명명했습니다.



### Overall architecture

ImageNet 학습에 사용된 DenseNet의 구조를 예를 들면 다음과 같습니다. $$k=32$$를 사용하였고, 각 "conv" layer는 BN-ReLU-Conv가 composite된 것을 의미합니다.

![DenseNet architecture](/.gitbook/assets/2/densenetarchi.png)



## 4. Experiment & Result

본 논문에서는 CIFAR-10, CIFAR-100, Street View House Numbers (SVHN), ImageNet dataset에서 DenseNet과 당시의 state-of-the-art model인 ResNet 및 ResNet의 변형 network들의 성능을 비교하는 실험을 진행했습니다. CIFAR dataset에 대해서는 data augmentation을 적용한 dataset도 추가로 사용했습니다 (각각 C10+, C100+로 표기). Training setup은 다음과 같습니다.

* 모든 network들은 SGD를 이용하여 학습
* CIFAR와 SVHN dataset에서는 batch size를 64로하여 각각 300 epoch, 40 epoch으로 학습
  * Learning rate은 0.1에서 시작하여 전체 training epoch의 50%와 75% 지점마다 10으로 나눔
* ImageNet dataset에서는 batch size를 256으로 하여 90 epoch으로 학습
  * Learning rate은 0.1에서 시작하고 30번째와 60번째 epoch에서 10으로 나눔
* Weight decay는 $$10^{-4}$$로 설정하였고, Nesterov momentum을 dampening 없이 0.9로 설정
* Weight initialization은 [여기](https://arxiv.org/abs/1502.01852)서 소개된 방법을 사용
* Data augmentation이 없는 CIFAR-10, CIFAR-100, SVHN dataset에서는 첫 convolutional layer를 제외한 모든 convolutional layer 다음에 dropout layer를 추가하고 dropout rate은 0.2로 설정

또한, DenseNet training의 memory consumption을 줄이기 위해 [이 논문](https://arxiv.org/abs/1707.06990)에서 설명된 방법으로 memory-efficient하게 구현한 DenseNet을 사용하였다고 합니다.



### Result

#### CIFAR and SVHN

CIFAR와 SVHN에서의 결과는 다음과 같습니다.

![Evaluation results (CIFAR and SVHN)](/.gitbook/assets/2/evalresult.png)

표에서 각 값은 error rate (%)를 나타내며 **Bold**로 표시된 값은 기존 network들보다 좋은 성능을 보여주는 값이고, **Blue**로 표시된 값은 가장 좋은 성능을 보이는 값입니다. 표에서 알 수 있듯이, CIFAR와 SVHN 모두에서 DenseNet이 가장 좋은 성능을 보여주었으며 대부분의 configuration에서 state-of-the-art network들보다 좋은 성능을 보였습니다.

마지막행에 표시된 190-layer DensNet-BC의 결과를 보면, data augmentation이 적용된 C10+과 C100+에 대해서 가장 좋은 성능을 보이고 있습니다. 이 당시에 많이 쓰이던 ResNet 계열의 모델들 (e.g., Wide ResNet)의 error rate이 C10+과 C100+에서 각각 4%, 20%대였던 것에 비해, 190-layer DenseNet-BC는 각각 3.46%와 17.18%의 error rate을 보였습니다. 또한 data augmentation이 적용되지 않은 C10과 C100에 대해서는 더 많은 성능 개선이 이루어졌는데, 이는 250-layer DenseNet-BC의 결과에서 확인할 수 있습니다. 기존 ResNet 계열 모델들이 10%, 35% 정도의 error rate을 보였던 반면, 250-layer DenseNet-BC는 약 2배 가량 개선된 5.19%, 19.64%의 error rate을 보였습니다. 하지만 이 configuration의 DenseNet-BC가 SVHN에서는 더 작은 모델인 100-layer DenseNet보다 높은 error rate을 보였는데, 저자들은 이에 대한 이유로 SVHN은 비교적 간단한 task이기 때문에 250-layer DenseNet-BC처럼 너무 깊은 모델은 오히려 overfit이 될 수 있기 때문이라고 설명했습니다.

한가지 더 주목할 점은 parameter의 개수입니다. 기존에 좋은 성능을 보였던 FractalNet이나 Wide Resnet의 경우에는 가장 좋은 성능을 보이는 모델의 parameter 수가 각각 38.6M, 36.5M 개였는데, 앞서 설명드렸던 250-layer DenseNet-BC의 parameter는 15.3M 개로, 훨씬 적은 parameter로 기존 state-of-the-art 모델들의 성능을 뛰어넘었다는 것을 알 수 있습니다.



#### ImageNet

ImageNet에서의 결과는 다음과 같습니다.

![Evaluation results (ImageNet)](/.gitbook/assets/2/imagenetresult.png)

오른쪽 그래프로부터 DenseNet은 ResNet보다 더 적은 parameter를 가지고 동일한 수준의 성능을 보일 수 있다는 것을 알 수 있습니다. 예를 들어, 약 20M 개의 parameter를 갖는 DenseNet-201이 약 40M 개의 parameter를 갖는 ResNet-101과 비슷한 성능을 보입니다.





## 5. Conclusion

* State-of-the-art인 ResNet의 주요한 idea인 short path의 장점을 극대화하기 위해 모든 layer 간에 short path를 만들고 concatenation으로 연결하는 dense connectivity를 적용한 DenseNet을 제안하였다.
* DenseNet의 각 layer는 적은 수의 feature map을 만들기 때문에 parameter의 수가 상대적으로 적지만 ResNet과 ResNet의 변형 network들보다 좋은 성능을 보였다.

### Take home message \(오늘의 교훈\)

> 단순히 sequential한 연결만으로 신경망을 깊게 만드는 것은 한계를 보이기 때문에 ResNet이나 Inception과 같이 다양한 변형이 필요하다.
>
> DenseNet은 이러한 점에서 또 다른 layer 간의 연결 방식을 제안했다는 점에서 흥미로운 논문이라고 생각한다.



## Author / Reviewer information

### Author

**방제현 (Jehyeon Bang)** 

* M.S. student in School of Electrical Engineering, KAIST (Advisor: [*Prof. Minsoo Rhu*](https://www.google.com/url?q=https%3A%2F%2Fsites.google.com%2Fview%2Fkaist-via%2Fpeople%2Fprofessor%3Fauthuser%3D0&sa=D&sntz=1&usg=AFQjCNF0B9afUSYs9L3XqwYQGNQ7aYSHmA))
* [Personal webpage](https://sites.google.com/view/jehyeonbang)

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. G. Huang, Z. Liu, L. Van Der Maaten and K. Q. Weinberger, "Densely Connected Convolutional Networks," 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 2261-2269, doi: 10.1109/CVPR.2017.243.
2. "GitHub - liuzhuang13/DenseNet: Densely Connected Convolutional Networks, In CVPR 2017 (Best Paper Award).", *GitHub*, 2021. [Online]. Available: https://github.com/liuzhuang13/DenseNet. [Accessed: 24- Oct- 2021].
3. K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778, doi: 10.1109/CVPR.2016.90.
