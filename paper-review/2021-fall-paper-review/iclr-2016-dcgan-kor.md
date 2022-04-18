---
description: Radford et al. / Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, 2016
---
## 1. Problem definition
당시 2016년도에는 지도학습 방식의 CNN이 이미지 태스크에 활발하게 응용되고 있었으나 비지도학습 방식의 CNN은 많은 주목을 받지 못했었다. 또한 Generative Adversarial Networks (GANs)은 unlabeled data로부터 재사용 가능한 image representation을 학습할 수 있는 구조로 최대 가능도 방법 (maximum likelihood method)의 좋은 대체재로 떠오르고 있지만 터무니없거나 비슷한 이미지를 생성하고 학습과정도 불안정 했다. 이에 DCGANs 저자는 Convolutional Networks를 활용한 새로운 GANs 구조를 제안한다.
- - - -
## 2. Motivation
### 2.1. Related work
#### 2.1.1. Representation Learning From Unlabeled Data
레이블이 없는 데이터로부터 표현 정보를 학습하는 방법은 크게 K-means와 같이 데이터 클러스터를 통해 학습하는 방법과 auto-encoders([Vincent et al., 2010](https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf), [Zhao et al., 2015](https://arxiv.org/abs/1506.02351), [Rasmus et al., 2015](https://arxiv.org/abs/1507.02672))를 학습하는 방법이 있다. 또한 [Deep belief networks](http://robotics.stanford.edu/~ang/papers/icml09-ConvolutionalDeepBeliefNetworks.pdf) 또한 계층적인 정보 표현을 학습하는데 유의미한 결과를 내는 것을 확인할 수 있다.

#### 2.1.2. Generating Natural Images
이미지 모델을 생성하는 방법은 크게 모수적 방법 (parametric)과 비모수적 방법(non-parametric)이 있다.
비모수적 방법은 기존에 있는 이미지를 맞추는 방식인데 텍스쳐 생성, 슈퍼 레솔루션, 인페이팅에서 활용한다.
모수적 방법은 variational sampling 방법([Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114)), iterative forward diffusion process([Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585)), GAN ([Goodfellow et al., 2014](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)) 등이 있었으나 생성된 이미지가 흐릿하게 보이거나 울어보이는 문제가 있었다. 최근에는 RNN([Gregor et al., 2015](https://arxiv.org/pdf/1502.04623.pdf))이나 deconvolution network 방법([Dosovitskiy et al., 2014](https://arxiv.org/abs/1411.5928))도 제안되었으나 지도학습 태스크에는 적용되지 않았다.

#### 2.1.3. Visualizaing the Internals of CNNs
신경망에 대한 오랜 비판은 내부에서 수행하는 작업에 대해 거의 이해가 불가능한 블랙박스 방식이라는 것이다. 이에 deconvolutions을 활용하여 최대 활성화를 필터링하여 각 convolution filter의 대략적인 목적을 찾을 수 있는 연구가 제안되었다. ([Zeiler & Fergus, 2014](https://arxiv.org/abs/1311.2901)) 유사하게 입력값에 대해 경사하강법을 사용하여 필터의 일부분을 활성화할 수 잇는 이상적인 이미지를 찾을 수 있는 방법이 제안되었다. ([Mordvintsev et al](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html))

### 2.2. Idea
* 거의 대부분의 환경에서 안정적으로 학습할 수 있는 Convolutional GANs 구조를 제시하며 이를 Deep Convolutional GANs (DCGAN)이라고 부른다.
* Discriminators를 이미지 분류 태스크에 적용하여 다른 비지도학습 알고리즘과 비교하여 경쟁력 있는 성능을 볼 수 있었다.
* GANs으로 학습된 필터를 시각화하고 특정 필터가 특정 객체를 그리는 방법을 경험적으로 보여줄 수 있다.
* Generators가 생성된 이미지 샘플에 대해 쉬운 조작을 할 수 있는 흥미로운 수학적인 벡터 속성을 가지고 있는 것을 볼 수 있었다.

- - - -
## 3. Method
대량의 unlabeled data로부터 안정적으로 훈련하면서 고해상도의 image representation을 표현할 수 있는 DCGANs 구조를 아래와 같이 제안한다. 아래 이미지는 (a)는 DCGAN의 Generator, (b)는 DCGAN의 Discriminator를 도식화하였다. 중요한 내용인만큼 원문과 번역문을 동시에 올린다.

![DCGAN Architecture](/.gitbook/assets/52/dcgan.png)

[Img Reference](https://medium.com/swlh/dcgan-under-100-lines-of-code-fc7fe22c391)

* Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator)
→ [All Convolutional Net](https://arxiv.org/abs/1412.6806)에서 제안한 방법으로 차원축소를 위해 maxpooling과 같은 pooling layer이 아닌 strided convolutions을 활용한 방법이다. Generator에서는 strided convolutions으로 spatial downsampling을 하고 discriminator에서는 [fractional-strided convolutions](https://datascience.stackexchange.com/questions/49299/what-is-fractionally-strided-convolution-layer)으로 spatial upsampling(spatial resolution 확대)을 한다.
* Use batch norm in both the generator and the discriminator
→ Generator와 Discriminator 모두 batch norm 사용한다. 이는 generator가 특정 포인트에만 수렴하는 것을 방지한다. 다만 모든 레이어 적용하게 되면 출력된 이미지가 흔들리거나 불안정하기 때문에 generator의 입력 레이어와 discriminator의 출력 레이어에는 batch norm을 제외했다.
* Remove fully connected hidden layers for deeper architectures
→ 실험 결과 global average pooling은 모델의 안정성을 높이지만 convergence speed가 급속도로 느려졌다. 또한 highest convolutional features를 generator의 입력값, discriminator의 출력값으로 바로 연결하는 것이 더 효과가 좋았다. 마지막으로 discriminator의 마지막 convolution layer은 flatten하고 single sigmoid output을 통해 출력했다.
* Use ReLU activation in generator for all layers except for the output, which uses Tanh
→ 실험 결과 모델이 더 빠르게 학습했고 학습 데이터의 색 범위를 잘 파악할 수 있었다.
* Use LeakyReLU activation in the discriminator for all layers
→ Maxout activation을 활용한 기존 GAN 논문과 다르게 더 높은 해상도 이미지 모델링을 위해 활용했다.

- - - -
## 4. Experiment

![Figure 1](/.gitbook/assets/52/figure1.png)

### 4.1. Dataset
* LSUN bedrooms dataset
	* 3백만장 정도의 이미지 학습
	* Generator가 학습 이미지를 기억하거나 over-fitting 문제를 해결하고자 중복된 이미지를 제거하는 작업을 추가적으로 진행
	* 32*32 downsampled center-cropped 한 학습 데이터에서 대해 3072-128-3072 de-noising dropout regularized ReLU autoencoder를 적용하여 [semantic hashing](https://arxiv.org/abs/1410.1165)을 생성하고 이를 활용하여 유사한 이미지를 제거한다. 이 과정을 통해 275,000개의 유사 이미지를 제거했다.
  
![Figure 2](/.gitbook/assets/52/figure2.png)

![Figure 3](/.gitbook/assets/52/figure3.png)

* Faces
	* 웹에서 3백만장의 사람 이미지를 크롤링하고 OpenCV face detector로 고해상도의 이미지만 유지하여 약 350,000개의 face box 이미지 활용
* ImageNet 1-k
	* 32*32 resized center crop하여 사용
  
### 4.2.  Training Setup
* Generator의 출력층에 사용된 tanh 활성화값 범위인 -1~1로 이미지 픽셀값 스케일링 (이외 별도의 augmentation 적용 X)
* Adam Optimizer 사용
* Learning Rate : 0.0002
* 가중치 초기값 : ~N(0, 0.02)
* LeakyReLU parameter : 0.2
* Batch Size : 128
* Momentum term β1 : 0.5 (0.9보다 안정적으로 학습함)

- - - -
## 5. Result

### 5.1. Empricial Validation of DCGANs Capabilities
지금은 GANs 성능을 평가할 때 Frechet Inception Distance (FID)를 주로 사용하고 있지만 이 논문에서는 비지도 데이터셋으로 학습한 모델의 최상위 층에 선형 레이어를 덧대 지도학습 데이터셋에 대해 평가하는 방법으로 DCGAN 모델의 성능을 평가했다.

> FID Score는 생성된 이미지의 품질을 측정하는 지표로, 생성된 이미지의 representation distribution과 학습 이미지의 representation distribution을 비교한다. [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://papers.nips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html)에서 2017년에 제안된 metric으로 DCGAN에서는 해당 지표로 성능을 측정하지 않았다.

#### 5.1.1. Classifying CIFAR-10 using GANs as a Feature Extractor
DCGAN의 성능을 측정하기 위해 비지도학습 방법으로 CIFAR-10에 대해 학습한 모델과 Imagenet-1k에 대해 학습한 DCGANs 모델을 CIFAR-10 데이터에 대해 비교 실험을 진행하였다. 비지도학습 방법에 대한 baseline 모델은 4800 feature maps의 K-means 모델을 활용했다. 또한 DCGANs을 지도학습 데이터에 대해 테스트하기 위해 discriminator의 모든 convolutional feature에 대해 maxpooling하고 이를 4*4 spatial grid로 만든 후 flattened하여 28672 차원의 벡터로 생성되도록 하였다. 마지막으로 regularized linear L2-SVM classifier로 class를 분리하였다. 아래 표와 같이 DCGANs의 경우 baseline 모델보다 성능이 높았으나 비지도학습을 위한 Excemplar CNNs보다는 성능이 낮은 것을 볼 수 있었다. 하지만 기존 CNNs보다 학습하는 feature maps 수는 줄었으며 Imagenet-1k에 대해 학습했음에도 CIFAR-10을 분리할 수 있는 것은 DCGANs이 학습 데이터와 상관없이 전반적인 image representation을 잘 학습했다는 것을 볼 수 있다.

![Table 1](/.gitbook/assets/52/table1.png)

#### 5.1.2. Classifying SVHN Digits using GANs as a Feature Extractor
위의 방식대로 DCGANs 모델을 On the StreetView House Numbers dataset (SVHN)에 대해 태스크를 수행했다. 그 결과 다른 CNNs 응용 모델과 비교하여 1000 labels에 대해 22.48%의 test error를 달성했다. 또한 논문에서는 DCGANs의 동일한 CNNs 구조에 대해 따로 학습하면서 DCGANs의 CNN 구조가 상대적으로 낮은 test error 달성해주는데 주요 요인은 아닌 것으로 파악했다.

![Table 2](/.gitbook/assets/52/table2.png)

### 5.2. Investigating and Visualizing the Internals of the Networks
다음으로 훈련된 Generator와 Discriminator를 다양한 방식으로 관찰했다. 이 때 nearest neighbor search와 log-likelihood metrics는 사용하지 않았다.

#### 5.2.1.  Walking in the Latent Space
전체 latent space 탐색하면서 memorization (ex. sharp transition) 혹은 hierarchically collapsed 된 부분이 있는지 탐색한다. 만약 표면적인 변화가 생긴다면 해당 모델은 의미있는 image representation을 학습했다고 볼 수 있다.

![Figure 4](/.gitbook/assets/52/figure4.png)

#### 5.2.3. Visualizing the Discriminator Features
지도학습 방식의 CNNs 구조 뿐만 아니라 비지도학습 방식의 DCGANs 또한 대량의 이미지 데이터로부터 의미있는 image representation을 잘 학습한다는 것을 볼 수 있었다. 아래 그림과 같이 (reference)에 제시된 guided backpropagation을 활용하여 discriminator가 침실의 침대 혹은 창문에 반응하는 것을 볼 수 있었다.

![Figure 5](/.gitbook/assets/52/figure5.png)

### 5.3. Manipulating the Generator Representation
#### 5.3.1. Forgetting to Draw Certain Objects
 Generator가 잘 학습했는지 확인하기 위해 창문을 학습했다고 생각되는 레이어를 제거하여 창문이 생성되지 않는지 확인하였다. 150개의 샘플 중 52개의 창문 부분을 수동으로 체크한 후, 두번째로 높은 convolution layers features에서 logistic refression으로 창문에 대해 positive한 경우 weight를 0으로 만들어 해당 부분을 학습하지 못하도록 하였다. 아래 그림의 결과처럼 Generator에서 창문을 더 이상 그리지 못하는 것을 확인할 수 있었다.

![Figure 6](/.gitbook/assets/52/figure6.png)

#### 5.3.2. Vector Arithmetic on Face Samples
(Reference)에 영감을 받아 얼굴 이미지에 대해 학습한 generator의 Z latent vector가 벡터 연산이 가능한지 확인하였다. 아래 그림처럼 단일 벡터에 대해서는 불안정하였지만 3개의 샘플에 대해 평균을 취하여 연산을 하니 비교적 안정한 벡터 연산이 가능한 것을 볼 수 있었다. 이를 통해 generator 학습에 필요한 이미지량을 획기적으로 줄일 수 있을 것이라고 기대한다.

![Figure 7](/.gitbook/assets/52/figure7.png)

![Figure 8](/.gitbook/assets/52/figure8.png)

- - - -
## 6. Conclusion
이 논문이 기여한 부분을 다음과 같이 정리할 수 있다.
* CNN과 GAN 구조를 활용하여 더 안정적인 DCGANs 구조를 제안
* 비지도방법으로 대량의 이미지 데이터에 대해 의미있는 image representation을 학습 가능
* Generator와 discriminator의 latent space가 어떤 식으로 잘 학습하고 있는지 경험적으로 보여줌
물론 GANs에 대한 성능을 지금의 FID 방식으로 비교하지 못했고, 논문에서 언급했듯이 학습이 오래될 수록 일부 필터가 single oscillating mode로 되면서 잘 학습이 안되기도 하지만 비지도학습으로 이미지를 학습할 수 있다는 것을 보여준 논문이다.

### Take home message \(오늘의 교훈\)
> GANs에 convolution을 적용한 것도 흥미롭지만 pooling layer가 아닌 strided convolution을 적용하여 GANs의 성능을 높인 것이 흥미로웠다. 또한 지금까지 블랙박스라고 여겼던 generator와 discriminator의 학습된 latent space를 살펴보기 위해 의도적으로 학습된 layer를 지우거나 arithmetic vector를 적용하는 것 또한 추후 다른 연구에 좋은 아이디어를 제공할 것이라고 생각된다.

- - - -
## Author / Reviewer information

{% hint style=“warning” %}
You don’t need to provide the reviewer information at the draft submission stage.
{% endhint %}

### Author

선한결 \(Sun Hangyeol\) 

* KAIST AI
* Currently working as an AI Researcher in Finance Industry. Interested in NLP and vision.
* Contact information \([Github](https://github.com/hannarbos), [LinkedIn](https://www.linkedin.com/in/hangyeol-sun/)\)


### Reviewer

1. 김기범 : [Github](https://github.com/LimGyeongrok)
2. 임경록 : [Github](https://github.com/rlqja1107)

## Reference & Additional materials

1. [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
