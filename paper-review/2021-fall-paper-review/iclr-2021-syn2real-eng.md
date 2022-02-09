---
description: Chen et al. / Contrastive Syn-To-Real Generalization / ICLR 2021
---

# Contrastive Syn-To-Real Generalization \[Eng\]



한국어로 쓰인 리뷰를 읽으려면 **[여기](./iclr-2021-syn2real-kor.md)** 를 누르세요.




##  1. Problem definition



This paper aims to solve a problem setting: **Zero-shot domain generalization on synthetic training task**. 

In short, the authors want to find a model that can work well on real-world image dataset even when trained with synthetic image dataset. 

$$min_{h} E_{(x,y)\in S_{test}}[L(h(x), y)]$$, where $$S_{train}=synthetic\_images, S_{test}=real\_images$$​ .



In detail, this problem definition can be divided into three parts as follow. 



**Domain generalization (DG)**: 

The goal of domain generalization algorithms is to predict distributions different from those seen during training [1]. 

![Dataset example of Domain Generaliation](/.gitbook/assets/32/DG_example.png)

*Example dataset of domain generalization*
*(Generalizing to Unseen Domains: A Survey on Domain Generalization / Wang et al. / IJCAI 2021)*

We are given $M$ training (source) domains,

$$S_{train} = \{S^i | i=1, ..., M\}$$, where $$S^i = {(x^i_j, y^i_j)}^{n_i}_{j=1}$$ denotes the i-th domain and $$n_i$$ is image set size.

*For example, $$(x^i_j,y^i_j)$$ is j-th sample of i-th domain*



The goal of domain generalization is to learn a robust and generalizable predictive function $$h: X \rightarrow Y$$ from the $$M$$ training domains $$S_{train}$$  to achieve a minimum prediction error on an unseen test domain $$S_{test}$$ (i.e., $$S_{test}$$ cannot be accessed in training and $$P^{test}_{XY}\neq P^i_{XY}$$  for  $$i \in \{1, ... , M\} $$):



$$P^{test}_{XY}\neq P^i_{XY}$$  for  $$i \in \{1, ... , M\} $$):

$$min_{h} E_{(x,y)\in S_{test}}[L(h(x), y)]$$, 

where $$E$$ is the expectation, $$L(\cdot, \cdot)$$ is the loss function, and $$h$$ is our main training function. 



For example, let's consider the example dataset of the upper figure. Our model has to minimize the loss on the photo image dataset ($$S_{test}$$), only learning with the other datasets (i.e., sketch, cartoon, art painting) ($$S_{train}$$).





**Synthetic training dataset**: 

Especially in this paper, they define the domain generalization task on the synthetic-to-real setting, i.e.,  $$S_{train}=synthetic\_images, S_{test}=real\_images$$.

*One of the ICLR reviewers points out that this syn2real task would limit this paper's impact.*

![VisDA-17 dataset of classification task](/.gitbook/assets/32/vis_da.png)





**Zero-shot learning**: 

While the synthetic-to-real dataset can utilize a validation dataset of real images to fine-tune the trained model, this paper use the model directly on the $S_{test}$ without fine-tuning process.

![VisDA-17 dataset of classification task on zero-shot learning](/.gitbook/assets/32/vis_da_2.png)

*In this case, we don't use the validation set (red X) and directly use the trained model to the test dataset (blue arrow)*





## 2. Motivation



### Related work

Note that this related work is divided into two categories: **domain generalization task and learning framework**.



**1. Domain generalization**

Generalizing a model to the unseen target domain without any supervision of it is a challenging problem. To alleviate this problem, various studies have been done. 

For readability, I'll summarize the most correlated two recent papers that deal with syn2real setting. If you're interested in other domain generalization tasks, let's refer to this[ summary GitHub site](https://github.com/amber0309/Domain-generalization).



**Yue et al.** [2] [paper link](https://arxiv.org/abs/1909.00889)

Yue et al. aim to alleviate the syn2real generalization problems, especially on semantic segmentation tasks. They try to generalize the model by randomly augmenting the synthetic image with the style of real images to learn domain-invariant representations. In short, they utilize the transfer-learning approach, i.e., transfer style information from real to synthetic dataset.

Their model shows excellent performance. However, the domain randomization process has to infer the styles of ImageNet classes, and their pyramid consistency model requires expensive computation. For example, their machines are equipped with 8 NVIDIA Tesla P40 GPUs and 8 NVIDIA Tesla P100 GPUs.



**Automated Synthetic-to-Real Generalization (ASG)** [3] [paper link](https://arxiv.org/abs/2007.06965)

*This is a prior work of the author of our mainly reviewing paper. It implies how they are interested in feature embedding methods on syn2real task.*

It is the first paper that discusses the syn2real generalization. The authors aim to encourage the synthetically trained model to maintain a similar representation and propose a learning-to-optimize strategy to automate the selection of layer-wise learning rates. When given two models (M, M_{o}), two losses are used for generalization.

* Given ImageNet pre-trained model $$M_o$$, we update $$M_{o}$$ with synthetic images while maintaining frozen ImageNet pre-trained model $$M$$.
* For a given task (i.e., classification or segmentation), $$M_o$$ is updated with cross-entropy loss.
* For transfer learning, minimize the KL divergence loss between the output of $$M$$ and $$M_o$$. 

While they propose the syn2real generalization on both classification and segmentation tasks, they still require heuristic training details, such as the size of the learning rate and set of layers to apply the learning rates.



**2. Contrastive learning**

*reference: https://nuguziii.github.io/survey/S-006/*

![Contrastive Self-supervised learning](/.gitbook/assets/32/constrastive_self_supervised_sample.png)

*ref: https://blog.naver.com/mini_shel1/222520820060*

Contrastive learning aims to build representations by learning to encode what makes two things similar or different. This usually includes employing large numbers of negative samples and designing semantically meaningful augmentations to generate diverse images. The most famous methods of contrastive learning are NCE loss and InfoNEC loss. Among such superior studies, I'll briefly introduce two ways that appear in our main reviewing paper.



**InfoNCE loss** [4] [paper link](https://arxiv.org/abs/1807.03748) 

![InfoNCE loss](/.gitbook/assets/32/info_nce.png)

Usually, we utilize InfoNCE loss ($$L_N$$ in the image) to make representations between positive samples close while ones between negative samples are far. For example, images in the retriever class should have similar feature embedding but different from cat images. Cosine-similarity is usually used to estimate the similarity between the embeddings, and this loss leads to two effects.

* Make embeddings close between positive samples: increase the similarity between two feature vectors, e.g., retriever_1 and retriever_2. 
* Make embeddings different between negative samples: decrease the similarity between two feature vectors, e.g., retriever_2 and cat_1.



**MoCov2** [5] [paper link](https://arxiv.org/abs/2003.04297)  | [git](https://github.com/facebookresearch/moco)

![SimCLR and MoCo](/.gitbook/assets/32/moco.png)

This is an improved version of MoCo by adding MLP head and data augmentation.

In [SimCLR](https://github.com/google-research/simclr), we should add positive and negative samples by increasing the batch size as much as possible(e.g., batch size of 10000) for best performance. However, since SimCLR requires lots of computation resources and the same amount of pos- and negative samples, MoCo proposes using a momentum encoder and dictionary of negative samples as a queue structure.

In MoCo, both inputs of two encoders are positive samples and load the negative samples saved in the queue. InfoNCE is calculated similarity between positive pair from inputs and another similarity between the negative pair.



**3. Hyperspherical energy**

*Learning towards Minimum Hyperspherical Energy / Liu and Lin et al. / NeurIPS 2018*

To measure how well distributed the feature embeddings are, the authors of our reviewing paper choose hyperspherical energy (HSE) as a criterion.

Original paper suggests minimum hyperspherical energy (MHE) regularization framework, where the diversity of neurons of layer is promted by minimizing the hyperspherical energy in each layer. It is inspired by Thomson problem, where one seeks to find a state that distributes N electrons on a unit sphere as evenly as possible with minimum potential energy. 

![HSE score (eq.1)](/.gitbook/assets/32/eq1.png)



Higher energy implies higher redundancy (Figure 4 - a), while lower energy indicates that these neurons are more diverse and uniformly spaced (Figure 4 - b).

![Feature embedding with and without minizing HSE score method](/.gitbook/assets/32/energe.png)



The only fact we have to remember during our reviewing paper is that lower energy represents diverse and uniformly distributed space. 





### Idea

REMIND that this paper's goal: **Zero-shot domain generalization on synthetic training task**. 

The authors analyze the distribution of embedding vectors trained on ImageNet, VisDA17-real dataset, VisDA-17-synthetic dataset.



![Distribution of embedding vectors](/.gitbook/assets/32/fig2.png)

We can observe that the embeddings of real images (a,b) are distributed widely, but synthetic images (c) collapse to a specific point.

Based on this observation, this paper assumes that the collapsed distribution of the synthetic dataset is a reason for poor performance on the sny2real generalization task.

Therefore, this paper aims to make similar embeddings between synthetic and real domains and distribute the synthetic domain features avoiding collapse.

The limitation of previous works and the main novelty of this paper can be summarized as follows.



**Limitation of previous works**

* Most of them concentrate on the representation learning for real-to-real transfer learning settings and improving the performance on the downstream tasks (e.g., classification or segmentation). 
* Significantly, the ASG model focuses on minimizing feature distance between synthetic and real on the domain generalization approach.

**Improvements of this work**

* This model suggests synthetic-to-real transfer learning setting both on classification and segmentation tasks.
* Minimizing feature distance between synthetic and real embeddings and avoiding concentration of synthetic feature embeddings.





## 3. Method

In this section, we'll understand how this model works and which way it is trained.

Before dive into the detailed process, let's keep in mind some notations.

* What can we see during the training phase?
  * Synthetic image $$x$$ and its ground-truth $$y$$  (i.e., class or segmented result)
  * Encoder which is pre-trained with ImageNet dataset
* Which dataset is our model evaluated?
  * Real image and its ground-truth



### Overview and notions

The main strategy of this paper is *push and pull*.

* Pull: minimize the distance of synthetic feature and ImageNet-pretrained feature
* Push: pushing the feature embeddings away from each other across different images on the synthetic domain. 

Compared with the ASG model, this framework can be visualized as follow.

![Model architecture](/.gitbook/assets/32/fig3.png)

Notions

* $$f_{e,o}$$ : ImageNet-pretrained model, $$f_e$$ : synthetically trained model
* $$L_{syn}$$ : task loss ($$L_{task}$$), loss of classification or segmentation
* $$x^a$$ : input synthetic image, this becomes **anchor** in contrastive learning
  * embeddings of $$x^a \to$$  $$z^a$$ from $$f_e$$  , $$z^+$$ from $$f_{e,o}$$
* K negative images $$\{x^-_1, ... , x^-_K\}$$  and its embeddings $$\{z^-_1, ... , z^-_K\}$$  for every anchor $$x^a$$ 
* $$h/\tilde{h} : \mathbb{R}^C \to \mathbb{R}^c$$ , non linear projection head with {FC, ReLU, FC} layers.  



If we get embedding of anchor image, the process is described as ...

**$$z^a = f_e \circ g \circ h(\tau(x^a))$$**

Let's figure out each function step by step!



### $$h(\tau(x))\to$$ Augment image and model 

**Image augmentation: $$\tau$$**

![Image augmentation example](/.gitbook/assets/32/image_aug.png)

*image from  https://nuguziii.github.io/survey/S-006/*

Image augmentation has been shown to improve the performance of the model. Guiding the model to observe images in diverse situations helps the model robust on diverse input conditions, i.e., improve its generality.

This paper utilizes[RandAugment](https://arxiv.org/abs/1909.13719) for image augmentation. There are diverse image augment functions, including translation, rotation, color normalization. RandAugment makes the diverse sequence of augmentation functions, so the training model would perform well which type of input image comes in.



**Model augmentation: $$h$$**

![Representations for each samples](/.gitbook/assets/32/eq2.png)

Not only augment the input image, but this model also augments the model by augmenting the non-linear projection head of frozen ImageNet-pretrained model, i.e., $$\tilde{h}$$. 

To create different views of feature embeddings, they use the mean-teacher-styled moving average of a model, i.e., exponential moving average.

Let  $$W_0$$ be the initial state, and  $$W_k$$ is the learned parameter from $$k$$-th batch dataset.

Moving average function updates $$W_0 = \alpha * W_0 + \beta * W_k$$ where $$k \in \{1, ..., K\}, \alpha + \beta = 1$$ .

In general, $$\alpha=0.99 , \beta=0.01$$. However, especially on exponential moving average function, $$\beta$$ decays when $$k $$ becomes larger (e.g., 0.01 at first, 0.001 at second). This leads the model to concentrate on the current dataset and forget the information of the old one. 

We can understand $$W_0 \to \tilde{h}$$ and $$W_k \to h$$ , leading to slight augmentation of ImageNet embedding information tuned to the synthetic one.  





### *Train $$f_e\to$$* Contrastive Loss 

**Loss**

Among diverse contrastive learning approaches, this model utilizes InfoNCE loss (detailed description is in 2.1. related work section).

![Contrastive loss](/.gitbook/assets/32/eq3.png)

, where $$\tau = 0.007$$ is a temperature hyper-parameter in this work.

$$L_{NCE}$$ guides the embedding vectors of positive samples located close to embedding space and vice versa. 



Sine this model aims to improve the classification and segmentation task, to final loss can be represented as follow.

![Final loss](/.gitbook/assets/32/eq4.png)

where $$L_{Task}$$ is loss of classification or segmentation.



**Details of $$L_{NCE}$$**

If we can collect a set of layers, which set of layers ($$\mathcal{G}$$) can generalize the model better?

Since we don't know which layer generates the best embeddings, we can calculate NCE loss across embedding outputs of each selected layer and sum up the loss. Note that the only change from Eq. 3 is $$\sum_{l\in\mathcal{G}}$$ .

![Equation 5](/.gitbook/assets/32/eq5.png)

In the ablation study, layer group {3, 4} performs best at generalization. 



In the segmentation task, we can compute NCE loss in a patch-wise manner.

Since the images of the segmentation task have more dense representations than those of the classification task, we utilize NCE loss on cropped feature map patches. In practice, the users crop $$x$$  into $$N_l = 8*8 = 64$$ local patches during segmentation training phase. 

![Patch-wise NCE loss](/.gitbook/assets/32/eq6.png)



For example, when $$i=3, N_l = 2*2 = 4$$ , the $$L^{l,3}_{NCE}$$ computation process can be drawn as follow.

![Example of patch-wise NCE loss](/.gitbook/assets/32/loss_sample.png)



### *Improve average pooling $$g\to$$* A-Pool:

Remember what have been considered: $$f_e, f_{e,o}, h, \tilde{h}, \tau$$ in

![Representations for each samples](/.gitbook/assets/32/eq2.png)

The only remaining part is $$g$$ , a pooling layer.

$$g$$ is a pooling layer that pools feature map from $$f_e , f_{e,o}$$ . If we leave $$g$$ as the global average pooling function, this will summarize all feature vectors with the same weight.

However, since the synthetic images can appear single object (e.g., human, plant, train, etc.), the average pooling function would merge all non-meaningful vectors (e.g., white background) into output embedding. 

To avoide such situation, this paper suggest to pool the feature map based on attention score ($$a_{i,j}$$) between the feature vecters ($$v_{:,i,j}$$) and average pooled feature vector ($$\bar{v}$$). 



![Attentional pooling layer](/.gitbook/assets/32/a_pool.jpeg)

* global average pooled vector $$\bar{v} = g(v) = \frac{1}{hw} [\sum_{i,j} v_{1,i,j}, ... , \sum_{i,j} v_{C,i,j}] , i \in [1, h] , j \in [1, w]$$

* attention score per each pixel at (i,j) as $$a_{i,j} = \frac{<v_{:,i,j}, \bar{v}>}{\sum_{i', j'} <v_{:,i',j'}, \bar{v}>} (i' \in [1,h], j'\in[1,w])$$

 We can defin this attentional pooling as A-pool, 

* $$\hat{v} = g_a(v) = [\sum_{i,j} v_{1,i,j} \cdot a_{i,j} , ... , \sum_{i,j} v_{C,i,j} \cdot a_{i,j}]$$ .



Via this attention-weighted pooling function, we can expect to augment the feature vector focusing on the spatially meaningful aspects.

Note that this attention score is only calculated on $$f_e$$ . We just copy the attention value of $$g$$  from $$f_e$$ to $$f_{e,o}$$ . 





### Review the overall process

We can summarize the overall process if we have an anchor image of the cat and negative samples of the dog and tree and the task is the image classification.

![Example of overall pipeline](/.gitbook/assets/32/overall_process.png)

1. RandAugment randomly augments the input images.
2. $$f_{e,o}$$ takes inputs of images of dog, tree, and cat, and $$f_e$$ takes the input of image of the cat. 
3. After attentional pooling the feature map from each encoder, we get $$z^{l,+}, z^{l,-}_{dog}, z^{l,-}_{tree}, z^{l,a}$$ . 
4. We train $$f_e$$ via two losses:
   1. $$L_{NCE}$$ : maximize cosine similarity of $$z^{l,+}\cdot z^{l,a}$$, minimizing sum of cosine similarity of  $$z^{l,a}\cdot z^{l,-}_{dog},  z^{l,a} \cdot z^{l,-}_{tree}$$ . Its gradient is highlighted as orange.
   2. $$L_{CE}$$ : minimize cross-entropy loss on classification task. Its gradient is highlighted as blue.



## 4. Experiment & Result



### 4.1. Classification task

**Experimental setup**

* Dataset : VisDA-17-classification dataset (http://ai.bu.edu/visda-2017/ )
  * ![VisDA-17 classification](/.gitbook/assets/32/visda_classification.png)
* Baselines: distillation strategies
  * Weight l2 distance (Kirkpatrick et al., 2017) [6]
  * Synaptic Intelligence (Zhenke et al., 2017) [7]
  * feature $$l_2$$ regularization [8]
  * KL divergence: ASG [3]
* Training setup
  * backbone: ImageNet pretrained ResNet-101
  * SGD optimizer, learning rate $$1 * 10^{-4}$$ , weight decay $$5 * 10^{-4}$$ , momentum $$0.9$$
  * Batch size $$32$$ , the model is trained for 30 epochs, $$\lambda$$ for $$L_{NCE} = 0.1$$
* Evaluation metric
  * generalization performance as hyperspherical enery (HSE) [9] *(details are in related work section)*
    * In experiments, HSE score on the feature embeddings is extracted by different methods.
  * classification accuracy



**Result**

![Resuls of classification task](/.gitbook/assets/32/table1.png)

Table 1 shows the relationship between HSE score (feature distribution) and generalization performance (accuracy). Except for the feature $$l_2$$ distance model, the accuracy increases as the HSE score decreases. Also on this paper's method (CSG) show the lowest HSE score and highest accuracy.

This confirms this paper's initial hypothesis: a model with diversely scattered features will achieve better generalization performance. 

The consistency of the experimental results and their inductive bias (hypothesis) improves the paper’s quality and makes the paper more persuasive.



### 4.2. Segmentation task

**Experimental setup**

* Dataset
  * synthetic dataset: GTA5 (https://download.visinf.tu-darmstadt.de/data/from_games/)
  * Real dataset: Cityscapes (https://www.cityscapes-dataset.com/) 
  * ![GTA5 and Cityscapes dataset](/.gitbook/assets/32/dataset_seg.png)
* Baselines
  * IBN-Net : improves domain generalization by carefully mixing the instance and batch normalization in the backbone. [10]
  * Yue et al. [2] *(details are on related work section)*
  * ASG [3] *(details are on related work section)*
* Training setup
  * backbone: DeepLabv2 with both ResNet-50 and ResNet-101, pretrained on ImageNet.
  * SGD optimizer, learning rate $$1 * 10^{-3}$$ , weight decay $$5 * 10^{-4}$$ , momentum $0.9$
  * Batch size: 6
  * Crop the images into patches of 512x512 and train the model with multi-scale augmentation (0.75~1.25) and horizontal flipping
  * the model is trained for 50 epochs, and $ $\lambda$$ for$$L_{NCE} = 75.$$
* Evaluation metric
  * mIoU: mean IoU across semantic classes (e.g., car, tree, road, etc.)
    * ![IoU](/.gitbook/assets/32/iou.png)
    * ![Examples of IoU](/.gitbook/assets/32/iou_ex.png)
    * *images from [ref](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)* 



**Result**

*Segmentation performance comparison*

![Table 5](/.gitbook/assets/32/table5.png)

* CSG (proposed model) gets the best performance gain between when with and without application in both backbones.
* Secondly, Yue et al. perform well following the CSG. However, this model utilizes ImageNet images during the training phase, unlike CSG, implicitly leveraging ImageNet styles. Considering this fact, CSG performs successfully without any real-world domain knowledge.



*Feature diversity after CSG*

![Comparison with and without CSG](/.gitbook/assets/32/fig6.png)

* Similar to 2. Idea step, this randomly samples a subset of the GTA5 training set to match the size of the Cityscapes training set.

* Models trained on real images have relatively diverse features, and synthetic training leads to collapsed features. However, compared to the previous one, the synthetic training set records lower $$E_s$$ than classification due to Eq.6 . 

* Fig.6 shows that improvement on the segmentation task is based on better-distributed feature embedding space than before. When comparing with Fig.2 in the idea section (while each visualization is done on a different task), we can observe that the collapse of synthetic images is alleviated better than before.

* This also demonstrates the initial hypothesis: a model with diversely scattered features will achieve better generalization performance.

* Limitation

  * Figures 2 and 6 come from the different tasks, i.e., classification and segmentation. It would be better if the diversity of the segmentation dataset is provided for a fair comparison.

  



## 5. Conclusion

**strength**

* Although assisted with ImageNet initialization, transferring the pretrained knowledge on synthetic images gives collapsed features with poor diversity in sharp contrast to training with real images. 
* The results imply that the diversity of learned representation could play an essential role in synthetic-to-real generalization.
* Experiments showed that the proposed framework could improve generalization by leveraging this inductive bias and can outperform previous state-of-the-arts without bells and whistles.

**weakness**

* The task is limited to the syn2real task. This paper should also present results when using the proposed method to address common domain generalization problems for general interest.





### Take home message \(오늘의 교훈\)

* Statistical observation and its visualization take significant rules to demonstrate the author’s hypothesis.

* Without any bells and whistles, this approach shows SOTA score at the syn2real task.

* When we analyze some problem, let’s consider the dataset’s distribution more carefully and utilize the statistical information to resolve the problem set. 



## Author / Reviewer information

### Author

**양소영 \(Soyoung Yang\)** 

* KAIST AI
* My research area is widely on computer vision and NLP, also HCI.  
* [Mail](sy_yang@kaist.ac.kr) | [GitHub](https://github.com/dudrrm) | [Google Scholar](https://scholar.google.co.kr/citations?user=5Mw3sVAAAAAJ&hl=ko)

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. In Search of Lost Domain Generalization / Gulrijani and Lopez-Paz / ICLR 2021

   Generalizing to Unseen Domains: A Survey on Domain Generalization / Wang et al. / IJCAI 2021
2. Domain Randomization and Pyramid Consistency: Simulation-to-Real Generalization without Accessing Target Domain Data / Yue et al. / ICCV 2019
3. Automated Synthetic-to-Real Generalization / Chen et al. / ICML 2020
4. Representation Learning with Contrastive Predictive Coding / Oord et al. / arXiv preprint 2018
5. Improved Baselines with Momentum Contrastive Learning / Chen et al. / arXiv preprint 2020
6. Overcoming catastrophic forgetting in neural networks / Kirkpatrick et al. / Proceeding of national Academy of Sciences 2017
7. Continual learning through synaptic intelligence / Zenke et al. / ICML 2017
8. ROAD: Reality Oriented Adaptation for Semantic Segmentation of Urban Scenes / Chen et al. / CVPR 2018
9. Learning towards Minimum Hyperspherical Energy / Liu and Lin et al. / NeurIPS 2018
10. Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net / Pan et al. / ECCV 2018
11. Korean blog describing contrastive learning:  https://nuguziii.github.io/survey/S-006/
