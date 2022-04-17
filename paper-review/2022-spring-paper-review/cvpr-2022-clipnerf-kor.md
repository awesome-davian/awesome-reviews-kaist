---
description: Can Wang et al. / CLIP-NeRF; Text-and-Image Driven Manipulation of Neural Radiance Fields / CVPR 2022
---

# CLIP-NeRF \[Kor\]
**English version** of this article is available.

##  1. Introduction
이 글에서 제가 소개드릴 논문은 [CLIP-NeRF; Text-and-Image Driven Manipulation of Neural Radiance Fields(CVPR'22)](https://arxiv.org/abs/2112.05139)로, view synthesis 분야에서 뛰어난 성과를 보여 최근 큰 주목을 받은 [NeRF(ECCV'20)](https://arxiv.org/abs/2003.08934)와 대용량의 (텍스트, 이미지) 쌍을 수집해 텍스트와 이미지 사이의 상관관계를 학습시킨 [CLIP(ICML'21)](https://arxiv.org/abs/2103.00020)의 방법론을 합쳐 prompt text 혹은 exemplar image만을 가지고 NeRF가 생성해낸 view를 변형할 수 있는 방법을 제안하고 있습니다.

### Problem Definition
이 논문에서 풀고자 하는 문제는 **text prompt나 single reference image를 가지고 NeRF를 조작하는 방법을 알아내는 것**입니다. 물체의 모양을 변형시킬 수 있는 shape code와 물체의 색을 바꿀 수 있는 appearance code로 latent space를 분리한 disentangled conditional NeRF 구조를 baseline으로 하여 이루어졌습니다.

We recommend you to use the formal definition \(mathematical notations\).

## 2. Motivation
CLIP-NeRF는 NeRF와 CLIP의 방법론을 합쳐 NeRF를 조작하는 방법을 소개하고 있기 때문에 먼저 NeRF와 CLIP에 대해 자세히 소개하겠습니다.

### Related work
#### NeRF

![Figure 1: NeRF overview](../../.gitbook/assets/2022spring/49/nerf_overview.png)

view synthesis는 3D 물체나 장면을 여러 각도에서 찍은 사진들을 학습해 임의의 새로운 각도에서 물체를 찍은 사진을 생성하는 방법입니다. volume rendering을 이용해 neural implicit representation을 진행하는 NeRF는 고품질의 view synthesis를 가능하게 합니다. 구체적으로 NeRF는 3D scene의 특정 위치 $$(x, y, z)$$와 3D scene을 보는 view point $$(\theta, \phi)$$가 주어졌을 때 특정 위치 $$(x, y, z)$$에서의 방출되는 색 $$c = (r, g, b)$$과 빛이 투과하지 못하고 반사되는 정도인 불투명도 나타내는 volume density $$\sigma$$를 반환하는 deep neural network입니다. 이때 volume density는 특정 위치에 존재하는 물질의 종류에 의해서 결정되는 고유한 특성이라 view point에 무관한 값을 가져야 하지만, 방출되는 색은 아래의 그림첨 어느 각도에서 보는지에 따라 달라질 수 있습니다. 물체의 색이 바라보는 각도에 따라 바뀌는 현상을 non-Lambertian effect라고 합니다.

![Figure 1: NeRF overview](../../.gitbook/assets/2022spring/49/non_lambertian_effect.png)

NeRF를 잘 학습시키면 특정 viewpoint에서 3D scene의 모든 지점에 대한 색과 volume density를 얻을 수 있습니다. 이를 얻은 뒤에는 classical volume rendering 방법을 이용해 viewpoint를 생성할 수 있습니다.

##### classical volume rendering
특정 지점 $$\mathbf{o}$$에서 $$\mathbf{d}$$ 방향으로 빛을 쏜다고 하면 camera ray의 궤적은 직선의 방정식 $$\mathbf{r}(t) = \mathbf{o} + t \mathbf{d}$$ 로 나타낼 수 있고, 이 camera ray의 궤적이 3D scene과 만나는 $$t$$의 범위가 $$[t_n, t_f]$$까지라고 한다면 $\mathbf{o}$에서 관측된 3D scene의 색 $C(\mathbf{r})$은 아래와 같이 표현됩니다.

$$
C(\mathbf{r}) = \int_{t_n}^{t_f}{T(t)\sigma(\mathbf{r}(t))\mathbf{c}(\mathbf{r}(t),\mathbf{d})}dt,

\\~\text{where}~ T(t) = \exp\Big(-\int_{t_n}^{t}\sigma(\mathbf{r}(s))ds\Big).
$$

이를 직관적으로 해석하면, $$\mathbf{d}$$ viewpoint에서 본 3D scene의 $$\mathbf{r}(t_n)$$에서 $$\mathbf{r}(t_f)$$까지의 색들을 NeRF를 통해 얻고, 이를 적분하여 최종적인 색을 얻을 수 있다는 것입니다. 이때, $$\mathbf{c}(\mathbf{r}(t),\mathbf{d})$$ 앞에 곱해지는 $$T(t)\sigma(\mathbf{r}(t))$$는 weight 역할을 합니다. 만약 현재 위치의 물체 앞에 불투명한 물체가 많다면 현재 위치의 물체가 최종적인 색에 기여하는 양이 줄어들게 될 것입니다. $$T(t)$$는 이를 반영한 값으로, 현재까지 누적된 volume density를 나타냅니다. 만약 현재까지 누적된 volume density가 크다면 $$\int_{t_n}^{t}\sigma(\mathbf{r}(s))ds$$의 값이 커져 $$T(t)$$는 작아지게 되고, 결국 현재 위치가 최종적인 색에 기여하는 양이 줄어들게 되는 것이죠. 또한, 최종적인 색에 기여하는 양은 특정 지점에서의 불투명도 $$\sigma(\mathbf{r}(t))$$에도 비례하게 될 것입니다. 이 두 요소를 곱한 $$T(t)\sigma(\mathbf{r}(t))$$가 특정 지점에서의 weight가 됩니다. classical volume rendering을 통해서 camera ray를 한 번 쏠 때마다 2D view image의 특정 pixel의 RGB값을 계산할 수 있게 되고, camera ray를 반복해서 쏘아 최종적인 2D image를 생성하는 것이 NeRF view synthesis의 원리입니다.

##### hierarchical volume sampling
이때 NeRF에서는 위의 적분을 샘플링을 통한 수치해석적인 방법으로 계산하게 됩니다. 구체적으로 $$[t_n, t_f]$$를 $N$개의 균일한 구간으로 나누고 각각의 구간에서의 uniform distribution에서 sampling을 진행하여 색과 volume density를 추정하는 coarse network $$\hat{C}_c(\mathbf{r})$$와 coarse network로부터 계산된 각각의 구간의 volume density에 비례하게 inverse transform sampling을 진행하여 색과 volume density를 추정하는 fine network $$\hat{C}_f(\mathbf{r})$$를 학습합니다. 이러한 hierarchical volume sampling을 통해 결과적으로 최종적인 색 계산에 많이 관여하는 부분에 많은 샘플이 존재하게 되는 importance sampling을 구현할 수 있게 됩니다.

##### architecture
NeRF의 구체적인 architecture는 다음과 같습니다. NeRF $$F_{\Theta}$$는 MLP 기반의 deep neural network로 이루어져 있습니다. 먼저 3D coordinate $$\mathbf{x}$$를 8 fully-connected layer(ReLU activation, 256 channels per layer)에 통과시켜 volume density $$\sigma$$와 256-dimensional feature vector를 얻습니다. 반환된 feature vector와 view point를 concat하여 최종적인 RGB를 얻습니다. volume density를 view point에 무관하게 만들기 위해 neural network에서 volume density $$\sigma$$ 값을 얻은 후에 view point $\mathbf{d}$를 넣어준 것을 확인할 수 있습니다.

![Figure 1: NeRF overview](../../.gitbook/assets/2022spring/49/nerf_architecture.png)

##### positional encoding
NeRF의 저자들은 위치 정보 $$(x, y, z)$$와 view point $$(\theta, \phi)$$를 NeRF $$\mathcal{F}_{\Theta}$$에 직접 넣는 것이 3D scene에서 물체의 모양, 색 등이 빠르게 변하는 부분을 표현할 때 적합하지 않음을 확인했습니다. 이를 해결하기 위해 high frequency function을 이용해 위치 정보와 view point를 higher dimensional space로 mapping한 뒤 NeRF 넣는 방법을 도입했습니다. 저자들은 transformer에서와 유사한 positional encoding 방법을 이용했습니다. 즉, $$F_{\Theta}$$를 $$F_{\theta}' \circ \gamma$$로 나타내고 $$\gamma(p) = (\sin(2^0 \pi p), \cos(2^0 \pi p), \cdots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p))$$를 normalized position $$\mathbf{x} = (x, y, z)$$와 normalized viewing director unit vector $$\mathbf{d} = (\sin\theta \cos\phi, \sin\theta\sin\phi, \cos\theta)$$의 각각의 element에 독립적으로 적용합니다. transformer에서는 positional encoding을 위치 정보를 제공하기 위해 이용했지만, NeRF에서는 high frequency function을 생성하기 위해 이용했다는 차이가 있습니다.

##### loss function
이때 loss function은 다음과 같습니다.

$$
\mathcal{L} = \sum\limits_{\mathbf{r} \in \mathcal{R}} \Big[ ||\hat{C}_c(\mathbf{r}) - C(\mathbf{r}) ||_2^2 + ||\hat{C}_f(\mathbf{r}) - C(\mathbf{r}) ||_2^2 \Big]
$$


#### CLIP



하지만 NeRF를 통해 생성된 view 변형하는 것은 극도로 어려운 일인데, 그 이유는 1) NeRF가 특정 각도에서 본 image를 결과물로 바로 내는 것이 아니라, 특정 각도에서 본 implicit function이기 때문에 explicit representation에 대한 방법들을 이용하기 어렵고, 2) multi-view dependency가 

Please introduce related work of this paper. Here, you need to list up or summarize strength and weakness of each work.

### Idea

After you introduce related work, please illustrate the main idea of the paper. It would be great if you describe the idea by comparing or analyzing the drawbacks of the previous work.

## 3. Method

{% hint style="info" %}
If you are writing **Author's note**, please share your know-how \(e.g., implementation details\)
{% endhint %}

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

{% hint style="warning" %}
You don't need to provide the reviewer information at the draft submission stage.
{% endhint %}

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
2. Official \(unofficial\) GitHub repository
3. Citation of related work
4. Other useful materials
5. ...

Transformer