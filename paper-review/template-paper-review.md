---
description: (Description) 1st auhor / Paper name / Venue
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


This paper applies the Self-Supervised Learning method to the ViT(Vision Transformer). ViT with DINO (kwoledge DIstillation with NO labels) shows explicit information about semantic segmentation[fig1] and works as a great k-NN classifier.

![Figure 1: Self-Attention from ViT with DINO](../../.gitbook/assets/50/self_attention.png)

Transformer, which based on the attention achieves astonishing result in NLP domain. Thus it is natural to apply such Transformer architecture to the vision domain, and the ViT comes up that apply transformer architecture directly to the vision domain with image patches. 

However, even though ViT shows competitive result with convnets, it requires enormous annotated dataset for pre-training with no clear benefit compare to convnets. Thus this paper try to tackle the argue that large supervision training set is required for ViT’s success with the new self-supervised learning framework called DINO. 


## 2. Motivation

In the NLP domain, one of the main ingredient for the success of Transfomer was self-supervised learning. For instance, BERT use the self-supervised learning method that apply bi-directional mask word prediction as a pretext task. This pretext task provide a richer learning signal than the supervised objective, to make the model can predict the masked word very well.
While this method is text-specific, there also exists lots of self-supervised learning methods for images with convolutional networks. 


### Related work

**BYOL: Bootstrap your own latent**

BYOL is self-supervised learning methods that learn the visual representation from the positively augmented image pair. They use two similar networks, __target network__ that generate the target output, and __online network__ that learns from the target network. From single image, BYOL generate 2 different augmented views with random modifications (random crop, gaussian blur, solarize, etc.) Then pass each image to each network and calculate the MSE of both latent vectors. Every iteration, online network updated by backpropagation with MSE loss while the target network updated with an exponential moving average of the online network's parameter. To avoid the collapsing, BYOL introduce the __predictor__ to the online network.
Its impressive that this SSL method does not require the negative samples which make the training more expensive while achieving comparable result with other method. However, this work uses the predictor to avoid collapsing and also only appled to the ConvNets in the paper.


**ViT: An Image is Worth 16x16 Words Transformers for Image Recognition at Scale**

ViT uses the transformer architecture to 

Please introduce related work of this paper. Here, you need to list up or summarize strength and weakness of each work.

![Figure 1: ViT architecture](../../.gitbook/assets/50/vit.png)

### Idea

The main idea of this paper is that applying the SSL method, which make the Transfomer dominating in the NLP domain, to the Vision Transfomer. They observe that the self-supervised Vision Transformer shows surprising sementic segmentation and k-NN classification results, which were not found in other self-supervised convnets. Unlike to the BYOL, DINO uses the centering and the sharpening the output of teacher network rather than the predictor.


<!-- After you introduce related work, please illustrate the main idea of the paper. It would be great if you describe the idea by comparing or analyzing the drawbacks of the previous work. -->

## 3. Method
<!-- 
{% hint style="info" %}
If you are writing **Author's note**, please share your know-how \(e.g., implementation details\)
{% endhint %} -->

![Figure 3: DINO](../../.gitbook/assets/50/dino.png)

DINO(knowledge DIstillation with NO labels) have two networks, **teacher network** $g_{\theta_t}$ parameterized with $\theta_t$ and **student network** $g_{\theta_s}$ parameterized with $\theta_s$. Both teacher network and student network have exactly same structure, allowing any type of backbone networks (ResNet50, ViT, etc.) to be used. Basically, DINO create few augmented images form single input image. Than feed image to both teacher and student network, and calculate the **Cross-Entropy loss** between outputs after softmax from teacher and studetn[eq 1]. 

$$min_{\theta_s}\ H(P_t(x), P_s(x)),\ \ \ \   H(a,b) = -a log b$$

$$where,\ P_s(x)^{(i)} = \frac{exp(g_{\theta_s}(x)^{(i)}/\tau_s)}{sum^{K}_{k=1}exp(g_{\theta_s}(x)^{(k)}/\tau_s)}$$

This Cross Entropy Loss is exploited to update the student network $\theta_s$, to learn the output of the teacher network which can be explained as the kwnoledge distillaion.

However, the teacher network itself needs to be udpated. Similar with the BYOL method, DINO uses the expoenetial moving average of $\theta_s$ to update the teacher network parameter $\theta_t$. This method is called Momentum Encoder in other works such as BYOL, or MOCO. The update $\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$ can be controlled with the momentum parameter $\lambda$, and default setting of DINO is cosine schedule form 0.996 to 1.

For the data augmentation, they uses multi crop strategy that generate a set of views $V$ with different distortions and crops. This set $V$ contains two __global view__ and few __local views__ of smaller resolution. All crops are passed through the student network while only global views passed throgh the teachser. Thus the loss in [eq 1] will be changed as below.

$$ min_{\theta_s}  \sum_{x \in {x_1^g, x_2^g}} \sum _{x' \in V, x' \neq x} H(P_t(x), P_s(x)), \ \ x^g \ is \ global$$


Some SSL methods are suffered by collapsing, which means the output converge to the trivial solution. Since DINO was not free from the collapsing, they use the **centering** and **sharpening** to avoid such collapse. Centering adds the bias term $c$ to the teacher; $g_t(x) \leftarrow g_t(x) + c$, where $c$ is based on the first-order batch statistics and updated with EMA for every batches. $c \leftarrow mc + (1-m)\frac{1}{B} \sum^{B}_{i=1}g_{\theta_t}(x_i)$. 

<!-- 
Please note that you can attach image files \(see Figure 1\).  
When you upload image files, please read [How to contribute?](../../how-to-contribute.md#image-file-upload) section.

![Figure 1: You can freely upload images in the manuscript.](../../.gitbook/assets/how-to-contribute/cat-example.jpg)

We strongly recommend you to provide us a working example that describes how the proposed method works.  
Watch the professor's [lecture videos](https://www.youtube.com/playlist?list=PLODUp92zx-j8z76RaVka54d3cjTx00q2N) and see how the professor explains. -->

## 4. Experiment & Result


This section should cover experimental setup and results.  
Please focus on how the authors of paper demonstrated the superiority / effectiveness of the proposed method.

Note that you can attach tables and images, but you don't need to deliver all materials included in the original paper.

This paper includes various experiments for the DINO method. But, I will show only important result in this review. You can see more details in the DINO paper.

* Compare with other SSL frameworks
* ViT trained with DINO - kNN classifier
* ViT trained with DINO - discovering sementic layout


### Experimental setup

They follow the implementation used in DeiT. For the pre-training, ImageNet dataset without labels was used. The optimizer is adamw, image patch size is 16 & 8. Data augemntation follows the BYOL (color jittering, Gaussian blur and solarization).

Two evaluation methods are used. First, Linear evaluation, they learn a linear classifier on frozen features and report the accuracy on a central crop. Second, Finetuning evaluations, they initialize the networks with the pretrained weights ad adapt them with training. Futhermore, since both methods are sensitive to the hyperparmeters so that they also evaluate the quality of features with a simple k-NN(k=20 was best) with freezed pretrained model. 

For more detail, since they use different setup for each experiment, I'll mention them at result part.


### Result

#### Comapare with other SSL frameworks
They show the result of DINO compare with other existing SSL frameworks.
For the experiment, they use ImageNet as a dataset and did linear evaluation and k-NN evaluation. For the backbone model, they use both ResNet 50 and ViT small. As a result, with the ResNet, DINO shows the best result among all SSL methods in both linear and k-NN evaluation. Furthermore, with the ViT, DINO shows best performance and beat other methods with ~10% gap in k-NN evaluation.

![Figure 4: Compare with other SSL](../../.gitbook/assets/50/result1.png)


#### ViT trained with DINO - kNN classifier
They show the performnace of ViT trained with DINO for the image retrieval task. For the retrieval, they freeze the features nad directly apply k-NN for the retrieval. The report the Mean Average Precision(mAP) for the revisited Oxford and Paris dataset. For pre-training, they use the Imagenett and GLDv2 dataset. As a result, ViT with the DINO shows even better mAP compare with the model that trained with the supervision.

![Figure 5: Image retrieval](../../.gitbook/assets/50/result2.png)


#### ViT trained with DINO - discovering sementic layout
The self-attention maps for ViT trained with DINO contain information about the segmentation of an image. Thus they report the mean region similarity $\mathcal{J}_m$ and mean countour-based accuracy $\mathcal{F}_m$ and compare with the other SSL methods and supervised ViT trained on ImageNet. We can see the ViT trained with DINO shows the best result among all of them in both metric. In addition, the below figures shows that the attention map of ViT with DINO works much better than the attention map of ViT in supervision manner.

![Figure 6: Sementic Segmentaion table](../../.gitbook/assets/50/result3.png)

![Figure 7: Sementic Segmentation Image](../../.gitbook/assets/50/result4.png)


## 5. Conclusion


This paper suggest new self-supervised learning method called DINO. DINO uses the knowledge distillation with no labels by learn from the teacher network that dynamically built during training. As a result, this method shows SOTA performance when the backbone is ResNet and dominating performance when the backbone is ViT, especailly when using simple k-NN method for evaluation. This method looks very effective since the computation cost is lower than other SOTA SSL methods while achieve competitive/exceeding performance in some tasks.

### Take home message 

> DINO, effective SSL method, can avhieve SOTA result with both ConvNets and ViT.
>
> ViT traind with DINO can directly extract the segmentation information , and work as a excellent k-NN classifier
>
> DINO with ViT even works better than supervised ViT.

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

