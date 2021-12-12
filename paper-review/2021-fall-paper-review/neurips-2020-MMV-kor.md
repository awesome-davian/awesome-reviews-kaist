---
description: Alayrac et al. / Self-Supervised MultiModal Versatile Networks / NeurIPS 2020
---

# Self-Supervised MultiModal Versatile Networks [KOR]


영어로 쓰인 리뷰를 읽으려면 [**여기**](https://awesome-davian.gitbook.io/awesome-reviews/paper-review/2021-fall-paper-review/neurips-2020-multimodal-versatile-eng)를 누르세요.

## 1. Problem definition

이 논문은 시각, 청각, 언어 정보를 모두 가지고 있는 Multimodal 데이터인 비디오를 기반으로 세 가지 모달리티(Modality) 모두에 적합한 네트워크를 만들고자 한다. 

좀 더 구체적으로 표현하면, 비디오 $$x$$의 Modality를 $$m$$으로 표현할 때, $$m \in \{v,a,t\}$$이다. $$x_v, x_a, x_t$$는 각각 비디오의 RGB 이미지, 오디오 샘플, 텍스트에 해당한다. $$n$$개의 비디오 학습셋이 주어질 때 ($$\{x^i\}_{i=1}^n$$), 논문에서는 먼저 각 모달리티에 맞는 임베딩 $$f_m: x_m \rightarrow \mathbb{R}^{d_m}$$ 을 찾고자 한다. $$f_m$$은 $$x_m$$을 입력값으로 받아 $$d_m$$ 차원에 임베딩(vector representation)한다.

각 모달리티 임베딩이 되면, 공유(Shared/Joint) 공간으로 확장한다. 공유 임베딩 공간은 $$\mathcal{S}_s \subset \mathbb{R}^{d_s}$$로 표현되며, 이 때  $$s \in \{va, vt, at, vat\}$$이다. 싱글 모달리티 표현(Representation) $$f_m(x_m)$$을 조인트 스페이스 $$\mathcal{S}_s$$로 투사하는 프로젝션 헤드(Projection Head) $$g_{m\rightarrow s}: \mathbb{R}^{d_m} \rightarrow \mathbb{R}^{d_s}$$ 를 찾는 것이 두 번째 문제이다. 결과적으로 만들어지는 조인트 임베딩 $$z_{m,s} = g_{m \rightarrow s} (f_m(x_m))$$는 학습된 맵핑 $$g_{m \rightarrow s}$$를 통해 계산할 수 있다.

결국 조인트 임베딩은 두 개 이상의 모달리티를 함께 임베딩하는 공간이며, 이를 활용하면 모달리티간의 Search를 간편하게 할 수 있다는 장점이 있다.

## 2. Motivation

인간의 인식은 멀티모달로 이루어진다. 따라서 어떤 데이터에서 동시 다발적으로 일어나는 여러 모달리티들 간의 유용한 관계를 도출하면 이를 활용하여 물리 세계를 더 잘 표현(Representation)할 수 있을 것이다. 이러한 점에 기인하여, 이 논문은 특히 비디오 데이터에 주목하였다. 비디오에는 시각, 오디오, 텍스트의 세 가지 다른 모달들이 자연스럽게 존재하며 이것을 통해 심층 신경망(Deep Neural Nets)의 표현 학습(Representation Learning)을 자기 지도(Self-supervised) 학습 방식으로 훈련이 가능하다.  이렇게 학습된 멀티모달 표현은 여러 모달리티의 데이터를 포함하는 다운스트림(Downstream) 태스크에 사용하여 성능 향상을 가져올 수 있다.

### Related work

서로 다른 모달리티 간의 좋은 Representation을 얻으려면 여러 연구 분야의 테크닉을 결합하는 것이 필요하다. 이 논문은 Single Modality를 위한 자기 지도 학습, 시각-언어 / 시각-음성 / 시각-음성-텍스트 표현 학습, 비디오와 이미지를 함께 처리하는 기법과 같은 기존 연구를 참고한다.

**single modality에서의 자기 지도 학습**

Chen et al.은 시각 표현의 대조 학습(Contrastive Learning)을 위한 간단한 프레임워크(SimCLR)를 제안한다. SimCLR는 augment한 이미지 간의 contrastive loss를 이용한 자기 지도 학습으로(같은 이미지에서 augment한 이미지는 positive pair로, 다른 이미지에서 augment한 이미지는 negative pair로 사용) ImageNet 벤치마크에서 뛰어난 결과를 보여준다. 저자는 동일한 자기 지도 학습 방법에서 영감을 얻었으며 멀티모달 네트워크에 contrastive loss와 비선형 projection head를 차용했다.

**이미지-텍스트 표현 학습**

이미지와 텍스트를 하나의 공간에 임베딩하려는 연구는 계속되어 왔으며, 이는 두 모달리티간의 large-scale search가 가능하게 만들었다. 하나의 공간에 이미지, 텍스트가 모두 임베딩이 가능하면 이 임베딩된 벡터들간의 dot product 계산 만으로 유사도를 측정할 수 있기 때문이다. 최근에는 비디오에서ASR(Automatic Speech Recognition)을 통해 나레이션을 텍스트로 만들어 자기 지도 학습으로 많이 사용하고 있다. 이 논문의 저자들도 이런 방법들에서 아이디어를 얻어 학습에 적용하였다.

**이미지-오디오 표현 학습**

Alwassel et al.은 한 모달리티(예: 오디오)의 표현을 다른 모달리티(예: 비디오)에서 배우는 자기 지도 학습 방법인 XDC(Cross-Modal Deep Clustering)를 제안했다. 이 방법은 지도 학습 방법을 능가한 성능을 보였으나, 텍스트 모달리티에 대한 고려는 하지 않았다.

**이미지-오디오-텍스트 표현 학습**

Aytar et al.은 시각, 소리, 언어에 대한 cross-modal CNN 네트워크를 제안하였다. 이미지-텍스트 그리고 이미지-소리 쌍으로 네트워크를 훈련한다. 저자에 따르면 텍스트-소리 간의 표현은 직접적인 훈련 없이 학습 효과가 일어난다고 한다. 한 가지 단점은 이미지-텍스트 쌍을 훈련하기 위해 COCO 및 Visual Genome이라는 주석이 달린 데이터 세트를 사용하는데, 완전한 자기 지도 학습 방식을 사용할 수 없기 때문이다.

**이미지와 비디오를 함께 처리하기**

이전의 연구에서 이미지와 비디오를 모두 처리하는 작업은 일반적으로 이미지 네트워크에서 비디오 네트워크로 이동한다. Girdhar et al.은 이미지 데이터셋에 사전학습된 SOTA모델을 사용한 distillation 프레임워크를 통해 비디오 representation을 배우는 방법을 제안하였다. 그러나 본 논문의 저자는 세상에 대한 우리의 인식이 정지 이미지보다 비디오와 더 비슷하기 때문에 비디오에서 배우는 것이 더 자연스럽다고 말한다. 따라서 비디오에 훈련된 네트워크를 이미지에 바로 적용할 수 있는 디플레이션(deflation)을 제안한다.

### Idea

이 논문의 핵심 아이디어는 레이블되지 않은 비디오 데이터를 사용하여 자기 지도 학습을 기반으로 어디에나 적용 가능한 MMV(Multi Modal Versatile) 네트워크와 그 네트워크의 학습 방법이다. MMV 네트워크는 다음 네 가지 원칙에 따라 설계되었다. 1) 세 가지(시각, 오디오, 텍스트) 모달리티 중 어느 것이든 입력으로 받을 수 있어야 한다. 2) 각 모달리티의 특성을 반영하여 데이터를 처리해야 한다(예: 오디오나 시각 모달리티가 텍스트보다 훨씬 정밀도가 높음) (Method에서 FAC 네크워크에 해당). 3) 훈련 중 보지 못한 데이터에 대해서도 다른 모달리티 간의 비교가 가능해야 한다. 4) 동적 비디오와 정적 이미지 형태의 인풋 모두에 효율적으로 적용할 수 있어야 한다 (Method에서 Deflation 기능에 해당).
MMV 접근 방식은 비디오 데이터에 레이블을 다는 작업이 필요하지 않다. 이는 답이 달린 데이터에 의존하던 이전의 작업과 차별화 된다.

## 3. Method

논문에서 제안하는 방법은 아래 그림과 같다.

![Untitled](/.gitbook/assets/59/Figure1.png)

Figure 1: 논문에서 제안하는 멀티모달 네트워크 디자인들

MMV 네트워크의 목표는 비디오 데이터(즉, 시각, 오디오 및 텍스트)의 세 가지 다른 데이터 모달리티들의 의미적 관계(semantic relation)를 간단하게 내적을 계산 함으로서 알아낼수 있는—즉, 다른 모달리티간의 유사성을 내적으로 검사할 수 있는—공통의 의미 벡터 공간(jointly-shared vector space)인 $$\mathcal{S}$$에 embedding 하는 것이다. 목표를 달성하기 위해, 이 논문에서는 MMV 네트워크에 대한 세 가지 아키텍처를 고려한다. 

"(a) Shared" 공간에서는 세 가지 모달리티가 공동으로 공유된 단일 벡터 공간  $$\mathcal{S}_{vat}$$에 embedding되며 다른 모달리티와의 직접 비교가 가능하다. 그러나 단점은 모든 모달리티가 동일한 데이터 정밀도(granularity)를 가지고 있다고 가정하며, 모달리티 별 특성을 고려하지 않는다는 것이다.

"(b) Disjoint" 공간에서는 두개의 모달리티를 공유하는 시각-오디오 그리고 시각-텍스트 공간인 $$\mathcal{S}_{va}$$와 $$\mathcal{S}_{vt}$$를 각각 학습한다. 이 옵션은 서로 다른 모달리티의 특수성, 특히 각기 다른 정밀도를 고려하여 두개의 다른 공유 공간을 학습하지만, 단점은 오디오와 텍스트 양식을 더 이상 직접 비교할 수 없다는 것이다.

"(c) FAC"(fine- and coarse-grained, 즉 고정밀과 저정밀) 공간에서도 두 개의 임베딩 공간에 대한 학습을 제안한다. 고정밀 공유 공간 $$\mathcal{S}_{va}$$에서 고정밀 모달리티 쌍인 시각과 오디오를 임베딩 하고, 저정밀 공유 공간인 $$\mathcal{S}_{vat}$$에서는 저정밀 모달리티인 텍스트를 포함한다. 시각과 오디오는 $$\mathcal{S}_{va}$$에서 직접 비교할수 있고, 텍스트는 $$\mathcal{S}_{vat}$$에서 시각과 오디오와 내적을 계산하여 직접 비교가 가능하다. 여기서, $$\mathcal{S}_{va}$$에서 $$\mathcal{S}_{vat}$$(혹은 반대)로 가는 선형 매핑이 필요하므로 학습을 통해 알아내야 하기 때문에 FAC 옵션은 세 종류의 심층 신경망이 학습이 필요한 대신 "Shared"와 "Disjoint" 옵션의 단점을 제거하였다. 

FAC는 MMV 네트워크의 목표를 달성하기 위한 최적의 아키텍쳐이며, 이 논문은 Self-supervised 방식으로 FAC를 학습하는 방법에 대해 설명한다. 제안하는 네트워크는 인터넷에서 흔히 찾을 수있는 많은 양의 비디오 데이터를 활용하여 학습할 수 있기 때문에 레이블링된 데이터가 전혀 필요하지 않다. 저자들은 이 비디오를 가지고 멀티모달용 자기지도 학습을 위한 Pretext Task를 셋업하는 방법을 알려준다.

제안된 Self-supervised 학습에 필요한 멀티모달 Contrastive Loss의 수식은 다음과 같다.

$$\mathcal{L}(x) = \lambda_{va} \textrm{NCE}(x_v,x_a) + \lambda_{vt} \textrm{MIL-NCE}(x_v,x_t)$$

여기서 $$\lambda_{va}$$와 $$\lambda_{vt}$$는 Regularization 파라미터이며, NCE Loss와 MIL-NCE Loss의 비중을 결정한다. NCE는 noise contrastive estimation으로 Contrastive Loss를 활용한다. FAC는 negative sampling을 사용한다. NCE와 MIL-NCE Loss Function은 다음과 같이 수식으로 표현된다. (MIL은 Positive 샘플과 Negative 샘플 갯수를 매칭시켜 Loss를 계산하는 Multiple Instance Learning을 말한다.) 

$$\textrm{NCE}(x_v,x_a) = - \log \left ( \frac{\exp(\frac{z^\top_{v,va} z_{a,va}}{\tau})}{\exp(\frac{z^\top_{v,va} z_{a,va}}{\tau}) + \sum_{z'\sim \mathcal{N}(x)} \exp(\frac{z'^\top_{v,va} z'_{a,va}}{\tau})} \right )$$

$$\textrm{MIL-NCE}(x_v,x_t) = - \log \left ( \frac{\sum_{z \in \mathcal{P}(x)} \exp(\frac{z^\top_{v,vat} z_{t,vat}}{\tau})}{\sum_{z \in \mathcal{P}(x)} \exp(\frac{z^\top_{v,vat} z_{t,vat}}{\tau}) + \sum_{z'\sim \mathcal{N}(x)} \exp(\frac{z'^\top_{v,vat} z'_{t,vat}}{\tau})} \right )$$

마지막으로 MMV 네트워크에는 디플레이션(Deflation)이라는 기능이 있는데, 이는 비디오 네트워크를 단일 이미지를 인풋으로도 돌릴수 있는 네트워크로 변환하는 것이다. Deflated된 네트워크는 비디오 데이터에 훈련된 오리지널 네트워크를 사용하여, 비디오 인풋이 아닌 이미지 인풋의 다운스트림 태스크에 바로 적용이 가능하다.

이 논문에서는 두 가지 유형의 비디오 네트워크 디플레이션을 고려한다. 첫번째는 3D 시공간 필터를 시간 차원에 대해 합산하여 2D 필터를 구동하는 방식이며, 두번째는 TSM (Temporal Shift Module) 네트워크라는 Channel Shifting을 없애 이미지를 인풋으로 하는 Residual 아키텍쳐 방식이다.

## 4. Experiment & Result

실험은 세 가지로 구성된다. 먼저 멀티모달 네트워크에 대한 다양한 아키텍쳐를 실험한다. 두 번째는 아키텍처 비교 결과 가장 성능이 뛰어난 것을 선택하여 모델을 Scale Up하여 SOTA 결과와 비교한다. 마지막으로 비디오 기반으로 학습된 네트워크를 정지된 이미지에 적용하여 deflation 접근 방법의 효과를 보여준다.

### Experimental setup, datasets and downstream tasks

- Network architectures
    - Video
        - Backbone: S3D-G, TSM with a ResNet50, TSM with a ResNet50x2
        - 벡터 $$f_v(x_v)$$를 얻기 위해 Backbone의 마지막 계층에서 시/공간 Average Pooling
        - 32개(2번째 실험의 경우 16개) 프레임을 10fps로 샘플링하고 200 × 200 크롭
        - 표준 Augmentation: 무작위 자르기, 수평 뒤집기, 시간 샘플링, 크기 변경, 색상 변경
    - Audio
        - 80개의 bin이 있는 log MEL 스펙트로그램으로 표현
        - ResNet50으로 처리
        - 프레임과 싱크하여 샘플링
        - $$d_a$$ = 2048 차원의 $$f_a(x_a)$$ 벡터를 얻기 위해 공간 Pooling
    - Text
        - 불용어를 제거하고 입력값을 16단어로 제한한 후, word2vec을 이용하여 300차원으로 추출한 후 linear layer를 적용하여 2048 차원으로 맵핑함
        - 공유되는 subspace의 차원은 512임. 예외적으로 FAC(Fine And Coarse) 디자인의 경우 $$\mathcal{S}_{va}$$(fine)은 512차원, $$\mathcal{S}_{vat}$$(coarse)는 256차원.
- Hyperparameters & Optimization
    - NCE와 MIL-NCE loss에서 내적을 계산하기 전에 벡터를 정규화
    - softmax에서 temperature τ = 0.07
    - HowTo100M은 10:1, HotTo100M+AudioSet은 1:1로 loss weight에 가중치
    - Initial learning rate = 0.002, Adam optimizer, warm up step 5K, helf-period cosine schedule
- Datasets (자기지도 사전학습에 사용)
    - HowTo100M: ASR을 이용하여 오디오를 텍스트로 변환한 1억개의 나레이션된 비디오 클립
    - AudioSet의 학습데이터: 2백만 개의 다른 인터넷 비디오에서 가져온 10초 클립으로 구성(텍스트 데이터 없음)
- Downstream tasks
    
    시각, 오디오, 텍스트 Representation을 평가하기 위해 다양한 다운스트림 태스크를 사용하였다. 자세한 내용은 아래 표에 정리하였다.
    
| Task                              | Evaluation                                         | Benchmark (Evaluation Metric)                                                  |
|-----------------------------------|----------------------------------------------------|--------------------------------------------------------------------------------|
| Action Classification             | Visual Representation                              | UCF101 (top-1 accuracy), HMDB51 (top-1 accuracy), Kinetics600 (top-1 accuracy) |
| Audio Classification              | Audio Representation                               | ESC-50 (top-1 accuracy), AudioSet (mAP)                                        |
| Zero-shot text-to-video retrieval | Text-Video Representation                          | MSRVTT (recall at 10), YouCook2 (recall at 10)                                 |
| Image Classification              | Transfer from video representations to image tasks | PASCAL VOC 2007 (mAP), ImageNet (top-1 and top-5 accuracies)                   |

### Results

**Design explorations**

첫 번째 실험은 멀티모달 네트워크 설계 별로 실험하여 평가하여 가장 뛰어난 디자인을 선별한다. 주요 포인트는 세 가지 모달을 모두 함께 학습하는 것이 두 가지 모달로 훈련된 모델보다 성능이 우수하다는 것이다. 제시된 디자인 중 FAC(fine-and-coarse) 방법이 가장 뛰어나다.

![Untitled](/.gitbook/assets/59/Result1.PNG)

**Large-scale experiments and comparison to the state-of-the-art**

SOTA 모델과 비교하기 위해서 앞서 Design explorations 실험에서 찾은 가장 우수한 아키텍처를 선택하여 모델의 사이즈를 키운다. 결과를 보면 제안된 FAC 어프로치가  UCF101, HMDB51, Kinetics600, AudioSet, ESC-50 benchmarks를 포함한 모든 다운스트림 태스크에서 SOTA를 능가한다. 

![Untitled](/.gitbook/assets/59/Result2.PNG)

**Transfer to image tasks via network deflation**

Deflation의 효과를 확인하기 위해 앞에서 훈련된 MMV 네트워크를 정적 이미지 태스크에 적용한다. 결과적으로 deflation 모델은 inflated 입력(즉, 정지 이미지 대신 전체 비디오)에서 비디오 모델과 거의 유사한 성능을 보인다. 제안된 deflation 방법은 naive deflation보다는 성능이 우수하지만 이미지에 대해 자기 지도 학습으로 훈련된 SOTA모델들이 이미지 태스크에서 MMV 네트워크보다 성능이 여전히 뛰어나다.

![Untitled](/.gitbook/assets/59/Result3.PNG)

## 5. Conclusion

이 논문은 비디오 데이터에 존재하는 시각, 오디오, 텍스트 모달리티를 함께 처리할 수 있는 MMV 네트워크를 제시한다. MMV 네트워크는 모달리티를 결합하여 joint representation함으로써 Downstream Task에서 성능을 향상시킬 수 있다. 제안된 FAC 접근 방식을 사용하면 시각 및 오디오 모달리티의 고정밀한(fine-grained) 표현을 유지하면서 비교적 저정밀의(coarse-grained) 텍스트 모달리티를 함께 임베딩할 수 있다. 또한 이 논문은 동적 비디오와 정적 이미지 형태의 시각 데이터 모두를 처리할 수 있는 MMV 네트워크에 대한 새로운 디플레이션 프로세스도 제안하였다. MMV 네트워크는 온라인에서 쉽게 찾을 수 있는 레이블링되지 않은 다량의 비디오 데이터를 통해 contrastive loss를 이용하여 자기 지도 학습 방법으로 훈련할 수 있다. 이렇게 학습된 MMV 네트워크는 UCF101, HMDB51, Kinetics600, AudioSet, ESC-50 벤치마크에서 SOTA를 달성하였다. 

주요 기술적 컨트리뷰션은 다음과 같다. 1) 자기 지도 학습 방법을 기반으로 다른 모달리티간의 임베딩 방법(shared, disjoint, FAC)에 대한 실험 연구, 2) 비디오 또는 정적 이미지를 효율적으로 처리할 수 있는 deflation 접근 방법, 3) 다운스트림 태스크에서의 우수한 성능.

### Take home message (오늘의 교훈)

데이터에 여러 모달리티가 있는 경우 하나만 선택하여 single-modality learning에만 집중하지 말고 모든 모달리티를 활용하여 모달리티 간의 관계를 찾고 활용하는 것이 이득이다!

## Author / Reviewer information

### Author

최현진 **(Hyunjin Choi)**

- KAIST Software Graduate Program
- Email: anneshj@kaist.ac.kr

### Reviewer

1. Korean name (English name): Affiliation / Contact information
2. Korean name (English name): Affiliation / Contact information
3. …

## Reference & Additional materials

1. Other useful materials
- NeurIPS Video
    
    [https://crossminds.ai/video/self-supervised-multimodal-versatile-networks-606fec66f43a7f2f827c1107/](https://crossminds.ai/video/self-supervised-multimodal-versatile-networks-606fec66f43a7f2f827c1107/)
    
- Original Paper
    
    [Self-Supervised MultiModal Versatile Networks](https://arxiv.org/abs/2006.16228)
