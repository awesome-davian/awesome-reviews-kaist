---
description: Chen et al. / Contrastive Syn-To-Real Generalization / ICLR 2021
---

# Contrastive Syn-To-Real Generalization \[Eng\]



\(In English article\) ---&gt; 한국어로 쓰인 리뷰를 읽으려면 **[여기]()**를 누르세요.
* 한국어 버전 추후 업데이트 예정


##  1. Problem definition



This paper aims to solve a problem setting: **Zero-shot domain generalization on synthetic training task**. 

In short, the authors want to find a model that can work well on real-world image dataset even when trained with synthetic image dataset. 

$$min_{h} E_{(x,y)\in S_{test}}[L(h(x), y)]$$, where $$S_{train}=synthetic\_images, S_{test}=real\_images$$​ .



In detail, this problem definiction can be divided into three parts as follow. 



**Domain generalization (DG)**: 

The goal of domain generalization algorithms is to predict well on distributions different from those seen during training [1]. 

![img](/.gitbook/assets/32/DG_example.png)

*Example dataset of domain generalization*
*(Generalizing to Unseen Domains: A Survey on Domain Generalization / Wang et al. / IJCAI 2021)*

We are given $M$ training (source) domains,

$$S_{train} = \{S^i | i=1, ..., M\}$$, where $$S^i = {(x^i_j, y^i_j)}^{n_i}_{j=1}$$ denotes the i-th domain.

The goal of domain generalization is to learn a robust and generalizable predictive function $h: X \rightarrow Y$ from the $M$ training domains to achieve a minimum prediction error on an unseen test domain $S_{test}$ (i.e., $S_{test}$ cannot be accessed in training and $P^{test}_{XY}\neq P^i_{XY}$  for  $i \in \{1, ... , M\} $):

$min_{h} E_{(x,y)\in S_{test}}[L(h(x), y)]$,

where E is the expectation and L($\cdot$, $\cdot$) is the loss function.

For example, if we consider the exmple dataset of upper figure, our model have to minimize the loss on photo image dataset ($S_{test}$) only learning with the other datasets (i.e., sketch, cartoon, art painting) ($S_{train}$).





**Synthetic training dataset**: 

Especially, in this paper, they define the domain generalization task on synthetic-to-real setting, i.e.,  $S_{train}=synthetic\_images, S_{test}=real\_images$ .

*Actually, one of ICLR reviewer points out that this syn2real task would limit this paper's impact.*

![img](/.gitbook/assets/32/vis_da.png)

*VisDA-17 dataset of classification task*





**Zero-shot learning**: 

While the synthetic-to-real dataset can utilize a validation dataset of real images to fine-tuning the trained model, this paper use directly test the model on the $S_{test}$ without fine-tuning process.

![Screen Shot 2021-10-24 at 12.41.23 AM](/.gitbook/assets/32/vis_da_2.png)

*VisDA-17 dataset of classification task on zero-shot learning: In this case, we don't use the validation set (red X) and directly use the trained model to the test dataset (blue arrow)*







## 2. Motivation



### Related work

Note this related work is seperated in two categories: **first for task, second for learning framework**.



**1. Domain generalization**

Generalizing a model to the unseen target domain without any supervision of it is a challenging problem. To alleviate this problem, diverse studies have been done. 

For readibility, I'll summarize the most correlated two recent papers that deals with syn2real setting. If you're interesting with other domain generalization task, let's refere this [summary github site](https://github.com/amber0309/Domain-generalization ).



**Yue et al.** [2] [paper link](https://arxiv.org/abs/1909.00889)

Yue et al. aim to alleviate syn2real generalization problem especially on semantic segmentation task. They try to genelize the model by randomly augment the synthetic image with the style of real images to learn domain-invariant representations. In short, they utilize the transfer-learning approach, i.e., transfer style information from real to synthetic dataset. 

Their model show great performance, however, the domain randomization process has to infer the styles of ImageNet classes, and their pyramid consistency model require expensive computation. For example, their machines are equipped with 8 NVIDIA Tesla P40 GPUs and 8 NVIDIA Tesla P100 GPUs.



**Automated Synthetic-to-Real Generalization (ASG)** [3] [paper link](https://arxiv.org/abs/2007.06965)

*Actually, this is a prior work of the author of our mainly reviewing paper. This implies how they are interested in feature embedding methods on syn2real task.*

This is a first paper that discuss the syn2real generalization. They aim to encourage the synthetically trained model to maintain similar representation, and propose a learning-to-optimize strategy to automate the selection of layer-wise learning rates. When given two models ($M, M_{o}$), two losses are used for generalization.

* Given ImageNet pretrained model $M_o$, we update $M_{o}$ with synthetic images while maintaining frozen ImageNet pretrained model $M$.
* For given task (i.e., classification or segmentation), $M_o$ is updated with cross-entropy loss.
* For transfer learning, minimize the KL divergence loss between the output of $M$ and $M_o$. 

While they propose the syn2real generalization on both classification and segmentation task, they still require heuristic training details, such as size of learning rate and set of layers to apply the learning rates. 





**2. Contrastive learning**

*reference : https://nuguziii.github.io/survey/S-006/*

Contrastive learning aims to build representations by learning to encode what makes two things similar or different. This is usually includes emplying large numbers of negative samples and designing semantically meaningful autmentations to generate diverse views of images. The most famous methods of constrastive learing is NCE loss and InfoNEC loss. Among such superior studies, I'll breifly introduce two methods which appear in our main reviewing paper. 



**InfoNCE loss** [4] [paper link](https://arxiv.org/abs/1807.03748) 

![FDrdxiy](/.gitbook/assets/32/info_nce.png)

Usually, we utilize InfoNCE loss ($L_N$ in the image) to make representations between positive samples to be close while ones between negative samples to be far. For example, images in class of retriever should have similar feature embedding but different with images of cat.  Consine-similarity is usually used to estimate similarity between the embeddings. This loss leads two effects.

* Make embeddings close between positive samples: increase the similarity between two feature vectors, e.g., retriever_1 and retriever_2. 
* Make embeddings different between negative samples: decrease the similarity between two feature vectors, e.g., retriever_2 and cat_1.



**MoCov2** [5] [paper link](https://arxiv.org/abs/2003.04297)  | [git](https://github.com/facebookresearch/moco)

![스크린샷 2020-12-11 오후 6.36.37](/.gitbook/assets/32/moco.png)

This is an improved version of MoCo by adding MLP head and data augmentation. 

In [SimCLR](https://github.com/google-research/simclr) , we should add positive samples and negtive samples by increasing the batch-size as much as possible(e.g., batch-size of 10000) for best performance. However, since SimCLR require lots of computation resources and same amount of pos- and negative samples, MoCo proposes to use momentum encoder and dictionary of negtaive samples as structure of queue. 

In MoCo, both inputs of two encders are positive samples and load the negative samples saved in queue. InfoNCE is calculated similarity between positive pair from inputs and another similarity between the negative pair. 



**3. Hyperspherical energy**

*Learning towards Minimum Hyperspherical Energy / Liu and Lin et al. / NeurIPS 2018*

To measure how well distributed the feature embeddings, the authors of our reviewing paper choose hyperspherical energy (HSE) as a criterion. 

Original paper suggests minimum hyperspherical energy (MHE) regularization framework, where the diversity of neurons of layer is promted by minimizing the hyperspherical energy in each layer. It is inspired by Thomson problem, where one seeks to find a state that distributes N electrons on a unit sphere as evenly as possible with minimum potential energy. 

![Screen Shot 2021-10-24 at 1.50.10 PM](/.gitbook/assets/32/eq1.png)



Higer energy implies higher redundancy (Figure 4 - a), while lower energy indicates that these neurons are more diverse and more uniformly spaced (Figure 4 - b). 

![Screen Shot 2021-10-24 at 2.39.37 PM](/.gitbook/assets/32/energe.png)



The only fact we have to remember during our reviewing paper is that lower energy represents diverse and uniformly distributed space. 





### Idea

REMIND that this paper's goal: **Zero-shot domain generalization on synthetic training task**. 

The authors analyze the distribution of embedding vectors, which are trained on ImageNet, VisDA17--real dataset, VisDA-17-synthetic dataset.



![img](/.gitbook/assets/32/fig2.png)



We can observe that the embeddings of real images (a,b) are distributed widely , but the ones of synthetic images (c) collapse to a specific point. 

Based on this observation, this paper assume that the collapsed distribution of synthetic dataset is a reason of poor performance on sny2real generalization task. 

Therefore, this paper aim to not only make similar embeddings between synthetic and real domains, but also distribute the synthetic domain features avoiding the collapse. 



Limitation of previous works and main novelty of this paper can be summarized as follows. 

**Limitation of previous works**

* Most of them concentrate on the representation learning for real-to-real transfer learning setting and improving the performance on the downstream tasks (e.g., classification or segmentation).  

* Especially, ASG model focuses on to minimize feature distance between synthetic and real ones on domain generalization approach.

**Improvements of this work**

* This model suggests synthetic-to-real transfer learning setting both on classification and segmentation tasks.
* Not only minimizing feature distance between synthetic and real embeddings, but also avoiding concentration of synthetic feature embeddings.





## 3. Method

In this section, we'll understand how this model works and which way it is trained.



Before dive in the detailed process, let's keep in mind some notations.

* What can we see during the training phase?
  * Synthetic image $x$ and its ground-truth $y$  (i.e., class or segmented result)
  * Encoder which is pretrained with ImageNet datset
* Which dataset our model is evaluated?
  * Real image and its ground-truth



### Overview and notions

Main strategy of this paper is *push and pull* .

* Pull: minimize the distance of synthetic feature and ImageNet-pretrained feature
* Push: pushing the feature embeddings away from each other across different images on synthetic domain. 

Compared with ASG model, this framework can be visualized as follow.

![Screen Shot 2021-10-24 at 3.17.01 AM](/.gitbook/assets/32/fig3.png)

Notions

* $f_{e,o}$ : ImageNet-pretrained model, $f_e$ : synthetically trained model
* $L_{syn}$ : task loss ($L_{task}$), loss of classification or segmentation
* $x^a$ : input synthetic image, this becomes **anchor** in contrastive learning
  * embeddings of $x^a \to$  $z^a$ from $f_e$  , $z^+$ from $f_{e,o}$
* K negative images $\{x^-_1, ... , x^-_K\}$  and its embeddings $\{z^-_1, ... , z^-_K\}$  for every anchor $x^a$ 
* $h/\tilde{h} : \mathbb{R}^C \to \mathbb{R}^c$ , non linear projection head with {FC, ReLU, FC} layers.  



If we get embedding of anchot image, the process is described as ...

**$z^a = f_e \circ g \circ h(\tau(x^a))$ **

Let's figure out each function step by step!



### $h(\tau(x))\to$ Augment image and model 

**Image augmentation: $\tau$**

![img](/.gitbook/assets/32/image_aug.png)

*image from  https://nuguziii.github.io/survey/S-006/*



Image augmentation has shown to improve the performance of model. Guiding the model to observe images in diverse situations help the model to be robust on diverse input conditions, i.e., improve its generality. 

This paper utilize [RandAugment](https://arxiv.org/abs/1909.13719) for image augmentation. There are diverse image augment functions including translation, rotation, color normalization. RandAugment makes diverse sequence of augmentation functions, so the training model would perform well which type of input image comes in. 



**Model augmentation: $h$**

![Screen Shot 2021-10-24 at 3.51.41 AM](/.gitbook/assets/32/eq2.png)

Not only augment the input image, this model also augment model by augmenting non-linear projection head of frozen ImageNet-pretrained model, i.e., $\tilde{h}$. 

To create different views of feature embeddings, they use mean-teacher styled moving average of a model, i.e., exponential moving average. 

Let $W_0$ is initial state and $W_k$ is learned parameter from $k$-th batch dataset.

Moving average function updates $W_0 = \alpha * W_0 + \beta * W_k$ where $k \in \{1, ..., K\}, \alpha + \beta = 1$ .

In general, $\alpha=0.99 , \beta=0.01$. However, especially on exponential moving average function, $\beta$ decays when $k $ becomes lager (e.g., 0.01 at first, 0.001 at second). This leads the model to concentrate on current dataset and forget the information of old one.  

We can understand $W_0 \to \tilde{h}$ and $W_k \to h$  , leading slightly augment ImageNet embedding information tuned to the synthetic one.  





### *Train $f_e\to$* Contrastive Loss 

**Loss**

Among diverse contrastive learning approaches, this model utilize InfoNCE loss (detailed description is in 2.1. related work section).

![Screen Shot 2021-10-24 at 3.51.58 AM](/.gitbook/assets/32/eq3.png)

$L_{NCE}$ guides the embedding vectors of positive samples locates close on embedding space and vice versa. 



Sine this model aim to improve the classification and segmentation task, to final loss can be represented as follow.

![Screen Shot 2021-10-24 at 3.52.19 AM](/.gitbook/assets/32/eq4.png)

where $L_{Task}$ is loss of classificaion or segmentation.



**Details of $L_{NCE}$ **

If we can collect a set of layers, which set of layers ($\mathcal{G}$) can generalize the model better?

Since we don't know which layer generates best embeddings, we can calculate NCE loss across embedding outputs of each selected layer and sum up the loss. Note that the only change from Eq. 3 is $\sum_{l\in\mathcal{G}}$ .

![Screen Shot 2021-10-24 at 3.53.25 AM](/.gitbook/assets/32/eq5.png)

In ablation study, layer group of {3, 4} performs best at generalization. 



In segmentation task, we can compute NCE loss in patch-wise manner.

Since the images of segmentation task has more dense representations than ones of classification task, we utilize NCE loss on cropped feature map patches. In practice, the users crop $x$  into $N_l = 8*8 = 64$ local patches during segmentation training phase. 

![Screen Shot 2021-10-24 at 3.54.30 AM](/.gitbook/assets/32/eq6.png)



For example, when $i=3, N_l = 2*2 = 4$ , the $L^{l,3}_{NCE}$ computation process can be drawn as follow.

![IMG_1123A879B1AB-1](/.gitbook/assets/32/loss_sample.png)



### *Improve average pooling $g\to$* A-Pool:

Remember what have been considered: $f_e, f_{e,o}, h, \tilde{h}, \tau$ 	in

![Screen Shot 2021-10-24 at 3.51.41 AM](/.gitbook/assets/32/eq2.png)

Only remaining part is $g$ , a pooling layer.

$g$ is a poolinag layer that pools feature map from $f_e , f_{e,o}$ . If we leave $g$ as global average pooling function, this would summarize all feature vectors with same weight. 

However, since the synthetic images can appear single object (e.g., human, plant, train, etc.), the average pooling function would merge all non meaningful vectors (e.g., white background) into output embedding. 

To avoide such situation, this paper suggest to pool the feature map based on attention score ($a_{i,j}$) between the feature vecters ($v_{:,i,j}$) and average pooled feature vector ($\bar{v}$). 

* global average pooled vector $\bar{v} = g(v) = \frac{1}{hw} [\sum_{i,j} v_{1,i,j}, ... , \sum_{i,j} v_{C,i,j}] , i \in [1, h] , j \in [1, w]$

* attention score per each pixel at (i,j) as $a_{i,j} = \frac{<v_{:,i,j}, \bar{v}>}{\sum_{i', j'} <v_{:,i',j'}, \bar{v}>} (i' \in [1,h], j'\in[1,w])$

 We can defin this attentional pooling as A-pool, 

* $\hat{v} = g_a(v) = [\sum_{i,j} v_{1,i,j} \cdot a_{i,j} , ... , \sum_{i,j} v_{C,i,j} \cdot a_{i,j}]$  .



Via this attention weighted pooling function, we can expect to augment the feature vector focusing on the spatially meaningful aspects.

Note that this attention score is only calculated on $f_e$ . We just copy the attention value of $g$  from $f_e$ to $f_{e,o}$ . 

![Screen Shot 2021-10-24 at 3.55.09 AM](/.gitbook/assets/32/a_pool.jpeg)





### Review the overall process

If we have an acnhor image of cat and negative samples of dog and tree and task is classification, we can summarize the overall process as this. 

![IMG_67FC8774824B-1](/.gitbook/assets/32/overall_process.png)

1. The input images are randomly augmented by RandAugment.
2. $f_{e,o}$ takes inputs of images of dog, tree, and cat, and $f_e$ takes input of image of cat. 
3. After attentional pooling the feature map from each encoder, we get $z^{l,+}, z^{l,-}_{dog}, z^{l,-}_{tree}, z^{l,a}$ . 
4. We train $f_e$ via two losses ...
   1. $L_{NCE}$ : maximize cosine similarity of $z^{l,+}\cdot z^{l,a}$, minimizing sum of cosine similarity of  $z^{l,a}\cdot z^{l,-}_{dog},  z^{l,a} \cdot z^{l,-}_{tree}$ . Its gradient is highlighted as orange.
   2. $L_{CE}$ : minimize cross-entropy loss on classification task. Its gradient is highlighted as blue.



## 4. Experiment & Result



### 4.1. Classification task

**Experimental setup**

* Dataset : VisDA-17-classification dataset (http://ai.bu.edu/visda-2017/ )
  * ![img](/.gitbook/assets/32/visda_classification.png)
* Baselines: distillation strategies
  * Weight l2 distance (Kirkpatrick et al., 2017) [6]
  * Synaptic Intelligence (Zhenke et al., 2017) [7]
  * feature $l_2$ regularization [8]
  * KL divergence: ASG [3]
* Training setup
  * backbone: ImageNet pretrained ResNet-101
  * SGD optimizer, learning rate $1 * 10^{-4}$ , weight decay $5 * 10^{-4}$ , momentum $0.9$
  * Batch size $32$ , the model is trained for 30 epochs, $\lambda$ for $L_{NCE} = 0.1$
* Evaluation metric
  * generalization performance as hyperspherical enery (HSE) [9] *(details are in related work section)*
    * In experiments, HSE score on the feature embeddings is extracted by different methods.
  * classification accuracy



**Result**

![Screen Shot 2021-10-24 at 3.56.31 AM](/.gitbook/assets/32/table1.png)

Table 1 show that there is relationship between HSE score (feature distribution) and generalization performance (accuracy). Except feature $l_2$ distance model, the accuracy increases as the HSE socre decreases. Also on this paper's method (CSG) show lowest HSE score and highest accuracy. 

This confirms this paper's initial hypothesis: a model with diversely scattered features will achieve better generalization performance. 

It seems that the consistency of the experimental results and their inductive bias (hypothesis) improves the quality of paper, and make the paper more persuasive. 





### 4.2. Segmentation task

**Experimental setup**

* Dataset
  * synthetic dataset: GTA5 (https://download.visinf.tu-darmstadt.de/data/from_games/)
  * Real dataset: Cityscapes (https://www.cityscapes-dataset.com/) 
  * ![img](/.gitbook/assets/32/dataset_seg.png)
* Baselines
  * IBN-Net : improves domain generalization by carefully mix the instance and batch normalization in the backbone. [10]
  * Yue et al. [2] *(details are on related work section)*
  * ASG [3] *(details are on related work section)*
* Training setup
  * backbone: DeepLabv2 with both ResNet-50 and ResNet-101, pretrained on ImageNet.
  * SGD optimizer, learning rate $1 * 10^{-3}$ , weight decay $5 * 10^{-4}$ , momentum $0.9$
  * Batch size: 6
  * Crop the images into patches of 512x512 and train the model with multi-scale augmentation (0.75~1.25) and horizontal flipping
  * the model is trained for 50 epochs, and  $\lambda$ for $L_{NCE} = 75.$
* Evaluation metric
  * mIoU: mean IoU across semantic classes (e.g., car, tree, road, etc.)
    * ![img](/.gitbook/assets/32/iou.png)
    * ![img](/.gitbook/assets/32/iou_ex.png)
    * *images from [ref](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)* 



**Result**

*Segmentation performance comparison*

![Screen Shot 2021-10-24 at 3.57.31 AM](/.gitbook/assets/32/table5.png)

* CSG (proposed model) get best performance gain between when with and without application of the model in both backbones. 

* Secondly, Yue et al. performs well following the CSG. However, this model utilize ImageNet images during training phase unlike CSG, implicitly leveraging ImageNet styles. Considering this fact, CSG performs successfully without any real-world doamin knowledge.



*Feature diversity after CSG*

![Screen Shot 2021-10-24 at 3.26.11 PM](/.gitbook/assets/32/fig6.png)

* Similar to 2.Idea step, randomly sample a subset of the GTA5 training set to match the size of the Cityscapes training set.
* Models trained on real images have relatively diverse features, and synthetic training leads to collapsed features. However, compared to the previous one, the synthetic training set records lower $E_s$ than classification due to the Eq.6 . 
* Fig.6 shows that improvement on segmentation task is based on better-distributed feature embedding space than before. When comparing with Fig.2 in idea section (while each visualization is done on different task), we can observe that the collapse of synthetic images is alleviated better than before. 
* This also demonstrates the initial hyphotesis: a model with diversely scattered features will achieve better generalization performance. 
* Limitation
  * Figure 2 and 6 comes from different task, i.e., classification and segmentation. It would be better that the diversity of segmentation datset is provided for fair comparison.





## 5. Conclusion

**strength**

* Although assisted with ImageNet initialization, trasfering the pretrained knowledge on synthetic images tends to give collapsed features with poor diversity in sharp contrast to training with real images. 

* This indicates that the diversity of learned representation could play an important role in synthetic-to-real generalization. 

* Experiments showed that the proposed framework can improve generalization by leveraging this inductive bias and can outperform previous state-of-the-arts without bells and whistles.

**weakness**

* The task is limited on syn2real task. For general interest, this paper should also present results when using the proposed method to address ordinary domain generalization problems. 





### Take home message \(오늘의 교훈\)

* Statistical observation and its visualization takes significant rules to demonstrate the author’s hypothesis.

* Without any bells and whistles, this approach shows SOTA score at syn2real task.

* When we analyze some problem, let’s consider the dataset’s distribution more carefully and utilize the statistical information to resolve the problem setting. 



## Author / Reviewer information

### Author

**양소영 \(Soyoung Yang\)** 

* KAIST AI
* My research area is widely on computer vision and NLP, also HCI.  
* [mail](sy_yang@kaist.ac.kr) | [github](https://github.com/dudrrm) | [google scholar](https://scholar.google.co.kr/citations?user=5Mw3sVAAAAAJ&hl=ko)

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. In Search of Lost Domain Generalization / Gulrijani and Lopez-Paz / ICLR 2021
   1. Generalizing to Unseen Domains: A Survey on Domain Generalization / Wang et al. / IJCAI 2021
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
