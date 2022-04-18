---
description: Mathilde Caron et al. / Emerging Properties in Self-Supervised Vision Transformers / ICCV 2021
---

# DINO: Emerging Properties in Self-Supervised Vision Transformers\[ENG\]

## \(Start your manuscript from here\)

{% hint style="info" %}
If you are writing manuscripts in both Korean and English, add one of these lines.

You need to add hyperlink to the manuscript written in the other language.
{% endhint %}

{% hint style="warning" %}
Remove this part if you are writing manuscript in a single language.
{% endhint %}

\(In English article\) ---&gt; 한국어로 쓰인 리뷰를 읽으려면 **여기**를 누르세요.

\(한국어 리뷰에서\) ---&gt; **English version** of this article is available.

##  1. Problem definition

이 논문은 Self-Supervised Learning 방법을 ViT(Vision Trnasformer)에 적용하였다. DINO(knowledge DIstillation with NO labels) 방식을 이용한 ViT는 semantic segmentation에 대한 정보를 직접적으로 드러냈으며, 굉장히 훌륭한 k-NN classifier로써의 성능을 보여준다.

![Figure 1: Self-Attention from ViT with DINO](../../.gitbook/assets/50/self_attention.png)

Transformer는 attention에 기초한 DNN 구조로써, 자연어처리에서 굉장한 성능을 보여주었다. 자연스럽게 해당 Transformer 구조를 Computer vision 분야에서 차용하여 사용하기 시작하였고, ViT라는 이미지 패치를 기반으로 한 Vision Transformer가 기존 Convolution Network들과 비등한 성능을 보여주었다.

그러나, ViT의 경우 기존의 ConvNet과 비슷한 성능을 내기 위해서는 pretraining을 위한 굉장히 큰 labeled training dataset을 필요로 한다. 이러한 데이터셋은 굉장히 얻기 어려울 뿐더러 computation의 측면에서도 비효율적이다. 하지만 기존의 ViT는 위와같은 단점들을 수용할만한 장점을 보여주지 못했다. 따라서, 이 논문은 ViT가 이와같은 성능을 내기 위해서 supervision dataset이 필요하다는 것을 해결하기 위해 self-supervised learning을 이용한다.


## 2. Motivation

NLP 분야에서 Transformer 구조가 성공을 거둘 수 있었던 가장 큰 이유중 하나는 self-supervised 방법을 이용한 것이다. 예를 들어, NLP에서 굉장히 좋은 성능을 보이고 있는 BERT라는 모델과 같은 경우 bi-directional word masking & prediction 방식의 SSL 을 이용하여 state-of-the-art(SOTA) 성능을 보여준다. 이는 supervised object보다 training에 있어서 더욱 효과적이다. 이와 같은 word masking은 text-specific 하나, contrastive learning과 같이 이미지를 위한 SSL 방식이 여럿 존재하고, 따라서 저자는 Vision 분야에서도 NLP처럼 이미지를 위한 SSL을 사용한다면 더욱 좋은 성능을 보일 수 있을것이라는 motivation에서 시작한다.

### Related work

**BYOL: Bootstrap your own latent**

BYOL은 positive augemented image pair로부터 visual representation을 학습하는 SSL 방법이다. 해당 방법은 2개의 DNN을 사용하는데, target network는 target output을 생성하고 online network는 해당 target output을 따라가는 방향으로 학습한다. 전체 프로세스를 살펴보면, 먼저 하나의 이미지에서 랜덤한 수정(random crop, gaussian blur, solarize 등)을 가하여 2개의 서로 다른 view를 생성하고 이를 positive pair라고 한다. 그 후 각각의 이미지를 각각 target network와 online network에 보낸 이후 두 ouput latent vector의 Mean Square Error를 loss로 사용한다. 매 iteration마다 online network는 해당 loss로부터 back propagation을 하여 update 되고, target network의 경우에는 online network의 weight를 Exponential Moving Average(EMA) 형태로 적용하여 업데이트 한다. 또한 SSL에서 자주 발견되는 문제인 collapsing을 피하기 위하여 BYOL은 predictor 를 도입하여, online network의 마지막에 추가한다.
기존 방식들은 collapsing을 피하기 위해 negative sample을 사용했는데, 이는 전체 computation cost를 늘린다. 허나 BYOL은 오직 positive pair만 이용하여 SSL을 진행함에도 기존 방식들과 비등한 성능을 보여준다. 하지만, 해당 방식은 predictor를 사용하여 target과 online network간의 구조 차이가 존재하며 오로지 ConvNet에만 해당 방식을 적용했다는 한계점이 있다.

### Idea

이 논문의 main idea는 NLP 분야에서 Transformer의 성능을 높은 수준으로 끌어올린 SSL을 Vision Transformer에 도입한 것이다. 논문에서는 SSL을 적용한 ViT가 Sementic Segmentation과 k-NN classification에서 놀라운 결과를 보였으며, 이는 기존 SSL with ConvNet에서는 보이지 않았던 결과라고 한다. 또한 BYOL과는 다르게, DINO는 collapsing을 막기 위하여 predictor가 아닌 단순한 centering과 sharpening 방식을 사용한다.

<!-- After you introduce related work, please illustrate the main idea of the paper. It would be great if you describe the idea by comparing or analyzing the drawbacks of the previous work. -->

## 3. Method
<!-- 
{% hint style="info" %}
If you are writing **Author's note**, please share your know-how \(e.g., implementation details\)
{% endhint %} -->

![Figure 3: DINO](../../.gitbook/assets/50/dino.png)




DINO(knowledge DIstillation with NO labels) 는 완전히 동일한 구조의 두 DNN으로 이루어져 있다. 각각 **teacher network** $g_{\theta_t}$ 와 **student network** $g_{\theta_s}$ 으로 표기하도록 하겠다. Backbone network로는 다양한 모델을(ex. ResNet50, ViT 등) 사용가능하다. 전체 프로세스는 아래와 같다. 먼저 BYOL과 비슷하게 하나의 이미지로부터 augmented image들을 생성하나, 이 경우 multi-crop을 사용하여 여러 image들을 생성한다. 그 중 두개의 이미지를 각각 teacher network와 student network에 보낸 후, 그 output을 softmax를 통해 K-dimension의 probability 형태로 만든 이후 Cross-Entropy loss를 적용한다. 

$$min_{\theta_s}\ H(P_t(x), P_s(x)),\ \ \ \   H(a,b) = -a log b$$

$$where,\ P_s(x)^{(i)} = \frac{exp(g_{\theta_s}(x)^{(i)}/\tau_s)}{sum^{K}_{k=1}exp(g_{\theta_s}(x)^{(k)}/\tau_s)}$$

이 CEL는 student network $\theta_s$ 를 업데이트 하는데 사용되며, teacher network의 output을 따라가는 방향이므로 knowledge distillation이라고 생각할 수 있다.

Teacher network같은 경우에도 업데이트가 필요한데, 이 경우 Loss를 사용하는 것이 아니라 student network의 weight를 EMA를 이용하여 적용하는 방식으로 사용한다. 이는 BYOL의 방식과 유사하며, 다른 논문들에서는 momentum update라고도 얘기한다. 실제 업데이트는 $\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$ 를 따라서 이뤄지며, momentum의 경우 파라미터 $\lambda$ 를 어떻게 바꾸냐에 따라서 조정될 수 있고 이 논문에서는 0.996부터 1까지 cosine schedule을 이용한다.

위에서 기술한 data augmentation을 위해서는 multi-crop 방식을 이용한다. DINO는 하나의 이미지로부터 view들의 set $V$를 생성한다. 해당 $V$는 두개의 global view와 더 낮은 resolution을 가지고 있는 몇개의 local view로 이루어진다. 따라서 SSL이 진행 될 때, student network는 모든 view들을 이용하여 inference를 하나, teacher의 경우 오로지 global view를 이용하여 output을 도출하고 위에서 언급한 CLE는 view가 어떻게 주어지는지를 생각한다면 아래와 같이 쓸 수 있다.

$$ min_{\theta_s}  \sum_{x \in {x_1^g, x_2^g}} \sum _{x' \in V, x' \neq x} H(P_t(x), P_s(x)), \ \ x^g \ is \ global$$

마지막으로 DINO는 collapsing을 피하기 위하여 centering과 sharpening을 이용한다. Collapsing은 단순히 trivial한 정답만을 내놓는 방향으로 모델이 수렴하는 것을 얘기하는데, 이는 많은 SSL에서 피하고자 하는 문제이다. Centering 같은 경우는 한 dimension이 dominate하는 것을 방지하나 uniform distribution으로 collapsing하게 조장하는 경향이 있고 sharpening은 그 반대이다. Centering의 경우 teacher의 결과에 bias $c$를 더하는 것이라고 생각할 수 있으며, 해당 $c$ 같은 경우 EMA를 이용하여 매 batch마다 업데이트 된다. 업데이트 방식은 아래와 같다 $c \leftarrow mc + (1-m)\frac{1}{B} \sum^{B}_{i=1}g_{\theta_t}(x_i)$.

<!-- 
Please note that you can attach image files \(see Figure 1\).  
When you upload image files, please read [How to contribute?](../../how-to-contribute.md#image-file-upload) section.

![Figure 1: You can freely upload images in the manuscript.](../../.gitbook/assets/how-to-contribute/cat-example.jpg)

We strongly recommend you to provide us a working example that describes how the proposed method works.  
Watch the professor's [lecture videos](https://www.youtube.com/playlist?list=PLODUp92zx-j8z76RaVka54d3cjTx00q2N) and see how the professor explains. -->

## 4. Experiment & Result


<!-- This section should cover experimental setup and results.  
Please focus on how the authors of paper demonstrated the superiority / effectiveness of the proposed method.

Note that you can attach tables and images, but you don't need to deliver all materials included in the original paper. -->

이 논문은 다양한 실험을 진행하였으나, 이 리뷰에서는 중요한 몇가지 결과만 언급하도록 한다.

* Compare with other SSL frameworks
* ViT trained with DINO - kNN classifier
* ViT trained with DINO - discovering sementic layout


### Experimental setup

기본적인 구현은 Facebook의 DeiT를 따랐다고 한다. Pre-training을 위해서 label이 없는 ImageNet 데이터셋을 이용하였으며 adamw optimizer를 사용하였고 image patch size로는 8과 16을 사용하였다고 한다. 기본적인 data augmentation 방식은 BYOL의 color jittering, Gaussian blur와 solarization을 차용하였다.

두가지 방식의 evaluation 방법이 사용되었는데, 첫번째는 기존의 방식인 linear evaluation & finetuning evaluation이다. 각각 pre-trained 모델을 고정하고 linear classifier를 도입하거나, pre-trained model을 initial weight으로 이용한 후 classification accuracy를 측정하였다. 두번째 방식의 경우 단순히 pre-trained network의 output을 k-NN에 적용하여 evaluation을 진행하였다.

더 디테일한 부분에 있어서는 실험에 따라 다름으로, 실제 실험에 작성하도록 한다.


### Result

#### Comapare with other SSL frameworks
가장 먼저 기존의 다른 SSL 방식들과의 결과 비교이다. 해당 실험을 위하여 이 논문은 Imagenet dataset을 사용하였고 각각 linear eval과 k-NN eval을 진행하였다. 먼저 backbone 모델이 ResNet50이었을 경우, 기존 다른 SOTA SSL 방식들보다 조금 더 나은 evaluation 결과를 보여주었다. 반면 ViT를 backbone 모델로 사용하였을 경우 굉장히 큰 폭의 성능 개선을 보여주었으며 k-NN eval의 경우 심지어 약 10%의 차이로 다른 방식들의 결과를 압도하였다.


![Figure 4: Compare with other SSL](../../.gitbook/assets/50/result1.png)


#### ViT trained with DINO - kNN classifier
지금부터는 DINO 방식을 이용하여 ViT를 training 했을 때의 결과를 보여준다. retrieval을 위하여, DINO 방식으로 pre-trained 된 ViT에서 나온 output 결과에 k-NN을 적용하여 바로 retrieval에 사용하였다. 이 경우 metric은 Mean Average Precision(mAP)이고, 데이터셋은 revisited Oxford와 Paris dataset 이다. Pretraining을 위해서는 Imagenet과 GLDv2 dataset을 이용하였다. 결과를 확인하면, DINO 방식을 이용한 ViT가 심지어 Supervised 방식으로 train 된 ViT보다 더 높은 mAP를 달성하는 것을 볼 수 있다.

![Figure 5: Image retrieval](../../.gitbook/assets/50/result2.png)


#### ViT trained with DINO - discovering sementic layout
DINO를 이용한 ViT의 self-attention map은 이미지의 segmentation에 대한 정보를 담고 있는것을 확인 할 수 있다. 따라서 실험 metric으로는 mean region similarity $\mathcal{J}_m$ 와 mean countour-based accuracy $\mathcal{F}_m$ 를 사용하여 다른 SSL 방식과 supervised 방식으로 train 된 ViT와 ImageNet에 대하여 비교를 진행하였다. 아래 표에서 DINO를 이용해 학습한 ViT가 두 metric 모두에 대하여 가장 높은 결과를 가지는 것을 볼 수 있다. 또한 아래 figure에서 실제 Supervised 방식으로 train 된 ViT의 self-attention map에서는 attention이 object 바깥의 부분에도 많이 존재하는 것을 볼 수 있으나 DINO의 경우 object 자체에 훨씬 집중되어 있는 것을 볼 수 있다.

![Figure 6: Sementic Segmentaion table](../../.gitbook/assets/50/result3.png)

![Figure 7: Sementic Segmentation Image](../../.gitbook/assets/50/result4.png)


## 5. Conclusion

이 논문은 새로운 SSL 방식으로써 DINO를 제시했다. DINO는 label이 없는 knowledge distillation방식을 사용하여, training 과정에서 생성되는 teacher network를 직접적으로 이용한다. 그 결과로써, 해당 방식은 ResNet backbone에서 다른 SSL과 비교할만한 성능을 보이며 ViT와 같이 사용되었을 경우 훨씬 더 좋은 성능을 보인다. 특히 Sementic segmentation과 k-NN classifier로써 훌륭한 결과를 보여줬다. DINO 방식은 기존 SOTA SSL 방식과 비교하여 computation cost 또한 더욱 적기 때문에, 굉장히 효율적인 SSL 방식이라고 생각된다.

### Take home message 

> 효과적인 SSL 방식인 DINO는 ConvNet이나 ViT 모두에 대하여 SOTA performance를 보여준다.
>
> DINO를 이용하여 학습한 ViT는 직접적으로 semantic segmentation 정보를 추출할 수 있으며 훌륭한 k-NN classifier로 동작한다.
>
> DINO를 이용한 ViT는 심지어 Supervised 방식으로 학습한 ViT보다도 좋은 성능을 보이는 경우가 있다.

## Author / Reviewer information

{% hint style="warning" %}
You don't need to provide the reviewer information at the draft submission stage.
{% endhint %}

### Author

**이윤헌 \(Yunheon Lee\)** 

* Affiliation: \(KAIST EE\)



### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. Caron, Mathilde, et al. "Emerging properties in self-supervised vision transformers." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
2. https://github.com/facebookresearch/dino.git
3. Grill, Jean-Bastien, et al. "Bootstrap your own latent-a new approach to self-supervised learning." Advances in Neural Information Processing Systems 33 (2020): 21271-21284.
4. Other useful materials
5. ...

