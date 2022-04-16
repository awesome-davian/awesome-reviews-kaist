---
description:  Ramech et al. / Zero-shot Text-to-Image Generation / IMCL 2021
---

# DALL-E\[Kor\]

##  1. Text-to-Image Generation

computer vision 분야에는 다양한 task 들이 존재한다. 널리 알려진 image classification, object detection, segmentation 뿐만 아니라 최근 활발하게 연구되고 있는 task 중 하나는 text-to-image generation 이다. image caption에 의해 설정한 조건에 맞는 image를 생성하는  task로, 이 논문에서는 단순히 해당 task를 수행하는 것이 아닌 “zero-shot”으로 고품질의 이미지를 생성했다는 것이 주목할 만한 포인트 이다.  

일반적인 text-to-image generation고정된 Dataset 에 대해 더 좋은 모델링을 할 수 있는 방법 ( 예: 복잡한 아키텍쳐, 손실함수, segmentation 마스크 등의 추가적인 정보) 을 찾는 것에 포커스를 맞춰왔다. 그러나, 이 논문은 전혀 다른 접근 방식을 택하고 있다.

인터넷에서 얻은 대규모의 text-image pair를 autoregressive transformer에 입력으로 넣어 모델을 학습 시킨다. 이렇게 충분히 학습된 모델은 zero-shot 방식으로 text-to-image generation task를 잘 수행한다는 것을 논문에서 보여주고 있다.


We recommend you to use the formal definition \(mathematical notations\).

## 2. Motivation

generative model이 발전함에 따라 text에 의해 설정된 조건에 따라 적절한 이미지를 생성하는 task에 대한 연구가 활발히 이루어졌다. 그러나 고정된 데이터에 대해 학습하는 것은 그 한계가 명확하다.  관련 데이터의 수가 많지 않을 뿐더러, 이렇게 학습된 모델의 경우 학습 과정에서 보지 못한 데이터의 경우 전혀 이해하지 못할 가능성이 높다(generalization이 어려움).

또한  최근 large-scale generative model의 성공과 text, image, audio 등 다양한 분야에서 제안된 autoregressive transformer의 성공으로 언어 모델인 GPT-3 와 같은 구조를 vision에도 적용해보려는 motivation을 기반으로 수행된 연구이다.

### 2.1 Related work

### 2.1.1GPT-3


### 2.2 Idea

DALL-E는 [openAI의 소개](https://openai.com/blog/dall-e/)에서도 언급하고 있듯이, 120억개의 파라미터와 2억 5천개의 이미지-텍스트 쌍으로 학습시킨 vision task를 위한 [GPT-3](https://arxiv.org/abs/2005.14165) 라고 할 수 있다. 

해당 논문에서 제안한 모델 DALL-E의 목표는 텍스트와 이미지 토큰을 하나의 stream을 입력으로, autoregressive transformer를 학습시키는 것이다. 즉, 텍스트와 이미지 전체에 대해 한 토큰 뒤에 다음 토큰이 올 likelihood를 최대화하는 방향으로 모델을 학습시킨 것이다. 이 때 이미지와 텍스트를 하나의 stream으로 입력함으로써 텍스트와 이미지는 동일한 latent space 상에 있는  embedding으로 학습된다. 

구체적인 method 에 대해서는 아래에서 좀 더 자세히 다루겠지만, 한가지 짚고 넘어가자면  ‘이미지 토큰’ 을 사용했다는 것을 들 수 있다.   이미지를 pixel 단위로 다루게 되면 고해상도의 이미지를 위해서는 엄청난 양의 메모리를 사용하게 된다. 뿐만 아니라, 우리가 실제로 이미지를 인식하는 구조(low-frequency)보다 이미지의 사소한 디테일(high-frequency)를 학습하게 되는 문제점이 발생한다([pixelCNN++](https://arxiv.org/abs/1701.05517)). 이 문제를 해결하기 위해 DALL-E는 ‘이미지 토큰’을 통해 이미지를 pixel 단위가 아니라  토큰 단위로 다루게 된다.

이를 통해 이미지를 해당 논문에서는 192배 압축하면서 visual quality는 유지할 수 있도록 하였다.

## 3. Method

<!-- ### Training Step -->
DALL-E의 학습과정은 간단하게 다음으로 요약하고 있다. 

> The overall procedure can be viewed as maximizing the evidence lower bound (ELB) on the joint likelihood of the model distribution over image x, captions y, and the tokens z for the encoded RGB image. 

즉, 본 논문에서 제안하고 있는 DALL-E의 학습은 이미지, 텍스트(caption), encoding된 이미지 토큰 z에 대한 joint likelihood를 최대화(maximize) 하는 것이다. 이 때, 확률 분포를 다루고 있는 모델에서 일반적으로 활용하는 Evidence Lower Bound(ELB)를 통해 모델을 학습시킨다. 

구체적으로 나타내면, 학습할 모델의 distribution을 factorization하면, 
```math
$$p_{\theta, \psi}(x,y) = p_{\theta}(x|y,z)p_{\psi}(y,z)$$
```
이고, 이 때 lower bound는 
```math
$$\ln p_{\theta, \psi}(x,y) \ge \mathbb E_{z~q_\phi(z|x)}(\ln p_\theta(x|y,z) - \beta D_{KL}(q_\phi(y,z|x), p_\psi(y,z)))$$
```
이며, 각 분포는\ 
* $$q_\phi$$ : RGB 이미지 에 대해 dVAE encoder에 의해 생성된 32x32 이미지 토큰의 distribution\
* $$p_\theta$$ : 이미지 토큰에 대해 dVAE decoder에 의해 생성된 RGB 이미지의 distribution\
* $$p_\psi$$ : transformer에 의해 모델링된 이미지와 텍스트의 joint distribution\

를 나타낸다. 

확률분포에 대한 정의를 통해 알 수 있듯이, 해당 모델은 학습과정을 두 단계로 나누고 있다. 첫번째 단계에서는 dVAE를 통해 이미지에 대한 visual codebook을 학습하고, 두번째 단계에서는 tranformer를 통해 텍스트와 이미지의 '토큰'에 대한 joint distribution을 학습한다. 

### 3.1 Stage 1: Learing the Visual Codebook 

위에서 잠깐 언급했지만, 이미지를 학습함에 있어, raw image를 그대로 사용하지 않고, 256x256 RGB 이미지를 32x32 image token으로 압축하여 사용한다. 이를 통해 이미지 품질의 저하 없이 trasnformer 학습에 필요한 context를 192배 가량 줄이는 효과를 얻는다. 이를 위해 본 논문에서는 dVAE(discrete Variational AutoEncoder)를 제안한다. 

일반적으로 VAE는 continuous 한 distribution을 출력으로 갖는다. 그러나 2017년 발표된 논문에서 제안된 모델인 [VQ-VAE](https://arxiv.org/abs/1711.00937) 에서 motivation을 얻어 이산적인 표현을 다루는 VAE를 사용한다. 그렇다면 왜 discrete한 표현을 사용하는 VAE를 사용하였을까?

그 이유는 VQ-VAE의 논문에서 주장하는 바에 따르면, 실제로 우리가 사용하는 언어가 본질적으로 이산적(discrete)이며, 이미지나 텍스트는 이렇게 이산적인 언어로 표현된다는 것이다.
(추가적으로 설명하자면, 우리가 사용하는 단어의 수는 한정적이며 단어와 단어의 사이가 명확하지 않다는 점에서 언어는 이산적이라고 할 수 있다.) 뿐만 아니라 VQ-VAE 논문이 제시하고 있는 결과에 의하면 이렇게 discrete한 latent representation은 latent space를 효율적으로 활용하여, 고차원의 데이터 (픽셀 수가 많은 이미지, 음성의 음소 등)에서 중요한 특징을 성공적으로 모델링하여 노이즈나 실제로 인간이 인지하기 어려운 디테일한 부분들에 모델의 capacity를 불필요하게 소비하지 않도록 한다.

이런 이유에서 본 논문에서는 VQ-VAE와 전체적인 틀은유사하지만, 그 방법에서는 조금 차이가 있는 dVAE를 통해 discrete latent representation을 얻고자 한다.
먼저 VQ-VAE에 대한 학습과정을 살펴본 뒤 dVAE는 어떤 부분이 다른 지를 통해 구체적인 학습 과정을 살펴보자. 

![VQ-VAE](/.gitbook/assets/2022spring/37/VQ-VAE.png)

위 그림은 VQ-VAE 논문에서 발췌하였다. 일단 latent embedding space $$e \in R^{KxD}$$ 를 정의한다. 이 때, $$K$$는 discrete latent space 의 크기(카테고리 수) 이며, $$D$$는 각 latent embedding vector $$e_i$$ 의 dimension 이다. 따라서, latent embedding space 에는 $$K$$ 개의 embedding vectors $$e_i \in R^D, i \in 1,2,...,K$$ 가 있는 것이다. 이런 embedding vector들이 모여있는 set을  그 다음 이미지 $$x$$를 encoder에 입력하여 encoder output $$z_e(x)$$ 를 얻는다. 이렇게 얻는 $$z_e(x)$$와 위에서 이미 정의된 embedding vectors 간의 거리를 계산하여 가장 가까운 embedding vector가 discrete latent representation이 되며, 해당 과정을 수식으로 표현하면 다음과 같다.
```math
$$
q(z=k|x) = \begin{cases}
1 &\text{for} k=\argmin_j ||z_e(x)-e)j||_2 \\
0 &\text{otherwise}
\end{cases}
$$

$$
z_q(x) = e_k, \textrm{where} k = \argmin_i ||z_e(x)-e)j||_2
$$
```
dVAE 역시 전반적인 과정은 위와 유사하다, 그러나 VQ-VAE에서는 가장 가까운 벡터를 deterministic 하게 선택한다면, dVAE에서는 uncertainty를 부여한다.

![dVAE](/.gitbook/assets/2022spring/37/dVAE.png)

이미지의 원본은 [해당 블로그](https://ml.berkeley.edu/blog/posts/dalle2/)에서 확인할 수 있다. 마찬가지의 방식으로 이미지를 encoding하여 codebook에서 벡터를 선택하고자 하려는 목표는 동일하지만, 거리가 가장 가까운 벡터를 단순히 선택(argmax)하는 대신 ["gumbel-softmax relaxation"](https://arxiv.org/abs/1611.01144) 을 통해 각 카테고리에 대한 weights를 얻어, 결과적으로 얻게 되는 sampled latent vector는 codebook vector 들의 weighted sum 이 된다.

이렇게 복잡한 방식으로 latent vector를 sampling하는데에는 이유가 있다. dVAE는 연속적인 latent space 대신 이산적인 categorical distribution을 가지므로 backpropagation을 할 수 없다. 이를 해결하기 위해 VQ-VAE는 vector quantization 을 사용하였고, dVAE의 경우 위에서 제시한 relaxation을 사용한 것이다. 

![dVAE2](/.gitbook/assets/2022spring/37/dVAE2.png)

위 과정은 첫번째 그림 이후의 학습과정이다. 이렇게 얻은 sampled latent vector를 다시 decoder에 입력으로 넣어 이미지를 reconstruction 하고, 위에서 언급한 우리가 일반적으로 사용하는 VAE의 학습방식에 따라 dVAE 역시 학습된다. 

마지막으로 prior 이라고 불리는 $$p(z)$$ 는 전체 codebook vectors에 대해 uniform distribution 으로 initialize 되어 있고, 다음 stage에서 언급하겠지만, transformer model을 학습하면서 이 prior를 업데이트하여 prior 역시 학습을 통해 얻음으로써 loss fucntion을 보다 더 최소화하게 된다. 

### 3.2 Stage2: Learning the Prior
이 stage에서는 텍스트와 이미지 쌍을 입력으로 받는 transformer를 학습시킨다. 

입력으로 받는 text-image 쌍은 모두 토큰 형태로, text의 경우 BPE(Byte Pair Encoding) 방식으로 최대 256 tokens (vocab size = 16,384)을 encoding 하여 사용하고, image 의 경우 위에서 언급했듯이 32x32 = 1024 tokens (vocab size = 8192)를 사용하여 이미지를 encoding한다. transformer에는 text token과 image token이 concatenate되어 하나의 stream으로 입력되며, 여기에서 사용되는 transformer는 autoregressive model로 이전의 입력을 통해 그 다음에 올 token을 예측하는 model이라고 할 수 있다. 

위 과정을 그림으로 나타내면 다음과 같다. 

![transformer](/.gitbook/assets/2022spring/37/transformer.png)

위와 같은 방식으로 text와 이전의 image token에 대해 다음 image token의 출력 결과를 다시 dVAE의 codebook vector로 변환하고 그 set of vectors를 dVAE의 decoder에 넣어 이미지를 출력으로 얻게 된다. 

![image_generation](/.gitbook/assets/2022spring/37/image_generation.png)



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

### 4.1 Experimental setup

### 4.1.1 Training Dataset
최초 실험은 12억개의 parameter를 가진 모델에 대해 MS-COCO의 확장형 버전이라고 볼 수 있는 330만개의 text-image pair로 구성된[Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/) 로 진행되었다. 그리고 이를 120억개의 parameter를 사용하는 모델로 키우기 위해, [JFT-300M](https://arxiv.org/abs/1707.02968v2) 와 비슷한 크기의 2억 5천여에 달하는 text-image pair를 인터넷에서 수집하여 데이터셋으로 사용한다. 이 데이터셋보다 MS-COCO dataset이 더 늦게 만들어졌기 때문에 MS-COCO를 포함하고 있지는 않지만, Conceptual Captions와 YFCC100M의 일부를 포함하고 있고, MS-COCO는 YFCC100M으로 부터 만들어졌기 때문에 학습데이터는 MS-COCO의 validation image 중 일부가 training data에 포함되어 있다.(해당 이미지에 상응되는 text는 다름) 

### 4.1.2 Evaluation
DALL-E의 경우 일반적인 text-to-image 생성모델과는 달리 해당 task를 통해 모델을 학습한 것이 아니기 때문에 MS-COCO

#### 4.1.3 Baseline 
Image를 생성하는 모델이기 때문에 GAN과의 성능을 비교할 수 있다. 해당 논문에서는 [AttnGAN](https://arxiv.org/abs/1711.10485), [DM-GAN](https://arxiv.org/abs/1904.01310), [DF-GAN](https://arxiv.org/abs/2008.05865)(당시 SOTA 모델) 과의 비교를 통해 제안한 모델의 성능을 평가하고 있다. 

#### 4.1.4 Score
* IS (Inception Score) : 생성된 이미지의 질을 평가하는 척도. 일반적으로 GAN 모델의 평가에 사용된다. 사전학습된 딥러닝 모델(i.e. inception-V3)을 사용하여 생성된 이미지를 분류한다. 생성된 이미지의 quality(어떤 물체인가)와 diversitiy(다양한 물체가 생성되었는가)의 기준으로 평가되며, 최저 1점 ~ 최고 1000점까지 점수를 메긴다(사전학습된 모델의 class 수)
* FID (Fréchet inception distance) : 실제 이미지와 생성된 이미지 사이의 feature vector간의 거리를 계산한 점수이다. IS와 마찬가지로 사전학습된 딥러닝 모델(i.e. inception V-3)을 사용하여 마지막 pooling layer에서 나온 벡터 간의 거리를 평가한다. GAN 모델 평가의 표준 척도로 사용되고 있으며, FID 가 낮을 수록 좋은 모델이라고 평가할 수 있다.


### 4.2 Result

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

{% hint style="warning" %}
You don't need to provide the reviewer information at the draft submission stage.
{% endhint %}

### Author

**Korean Name \(English name\)** 

* 윤은섭 \(KAIST EE\)

* EMAIL_esyoon97@kaist.ac.kr

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. Citation of this paper
2. Official \(unofficial\) GitHub repository
3. Citation of related work
4. Other useful materials
5. ...

