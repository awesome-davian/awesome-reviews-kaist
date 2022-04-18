---
description: Alayrac et al. / Self-Supervised MultiModal Versatile Networks / NeurIPS 2020
---

# Self-Supervised MultiModal Versatile Networks [Eng]


한국어로 쓰인 리뷰를 읽으려면 [**여기**](https://awesome-davian.gitbook.io/awesome-reviews/paper-review/2021-fall-paper-review/neurips-2020-mmv-kor)를 누르세요.

## 1. Problem definition

Given a training set of multimodal data examples such as video (which contains visual, audio, and language/text modalities), the paper seeks to learn 1) modality-specific representation (i.e., single-modal embedding/vector representation and 2) joint embedding/vector representation for multiple modalities. 

Formally, consider a modality $$m \in \{v,a,t\}$$ for a video $$x$$ such that $$x_v, x_a, x_t$$ correspond to frames of RGB images, audio samples, and discrete word tokens, respectively. Given a training set containing $$n$$ videos $$\{x^i\}_{i=1}^n$$, the paper first seeks to learn modality-specific representation $$f_m: x_m \rightarrow \mathbb{R}^{d_m}$$, where $$f_m$$ parameterizes modality-specific neural network taking $$x_m$$, modality $$m$$ from video $$x$$ as an input to produce a vector representation (embedding) of dimension $$d_m$$. 

For a joint (or shared) embedding space $$\mathcal{S}_s \subset \mathbb{R}^{d_s}$$ with $$s \in \{va, vt, at, vat\}$$, the paper secondly seeks to learn a projection head $$g_{m\rightarrow s}: \mathbb{R}^{d_m} \rightarrow \mathbb{R}^{d_s}$$  that can embed a single-modality representation $$f_m(x_m)$$ into the joint space $$\mathcal{S}_s$$. The resulting joint embedding (vector representation) $$z_{m,s} = g_{m \rightarrow s} (f_m(x_m))$$ can be easily computed using the learned mapping $$g_{m \rightarrow s}$$.

oint embedding is a space for embedding two or more modalities together, and using this has the advantage of making it easy to search between modalities.

## 2. Motivation

Driven by the multimodal nature of human perception, it is beneficial to draw useful relationships among different modalities of data (that occur synchronously) and use those relationships for facilitating good representations of the physical world. The authors are in particular motivated by video data, where three different modalities (namely, visual, audio, and text) are present naturally to enable multimodal (self) supervision sufficient to train a deep neural net for representation learning. Such learned multimodal representations can be used to enhance the performance of downstream tasks involving multiple data modalities. 

### Related work

Obtaining a good representation of different modalities requires combining several techniques from different research areas. The paper is related to the areas such as self-supervised learning for single modality, vision and language representation learning, vision and audio representation learning, vision, audio and language representation learning and training techniques for video and image.

Chen et al. propose a simple framework for contrastive learning of visual representations (SimCLR). Using contrastive loss across augmented images, SimCLR show remarkable results on the ImageNet benchmark. The authors are inspired by the same self-supervised learning technique, and they have adopted a contrastive loss and nonlinear projection heads for their multimodal networks.

There have been attempts to embed image and text jointly onto a shared space. This allows  large-scale search across modalities by measuring semantic similarity from simple dot product of feature vectors in the shared space. By using Automatic Speech Recognition (ASR) to generate text from narration, self-supervised approaches have become popular. The authors are also motivated by the idea of self-supervised learning and embedding vectors on the same space.

Alwassel et al. propose Cross-Modal Deep Clustering (XDC), a self-supervised technique to learn representation of one modality (e.g., audio) from another modality (e.g., video). The method outperforms the supervised one, but they do not consider text modality.

Aytar et al. present a cross-modal convolutional network with vision, sound, and language modalities. They train the network with image-text and image-sound pairs. They claim that text-sound representation can be learned without training. To train image-text pair, however, they use annotated dataset named COCO and Visual Genome.

Previous work on processing both image and video typically goes from an image network to a video network. Girdhar et al. propose a method to learn video representation via distillation framework using state-of-the-art models pre-trained on image datasets. The authors indicate that learning from video is more natural because our perception of the world is more like video than still images. Therefore, they propose deflation that enables video-trained network to be directly applicable to image.

### Idea

The key idea of the paper is multimodal *versatile* (MMV) network and its training methods based on self-supervised learning using unlabeled video. MMV network is designed on the following four principles: 1) MMV network should be able to take any of the three (visual, audio, text) modalities as input; 2) MMV network should process data based on the specificity of modalities contained (e.g., respecting that audio and visual modalities are much more fine-grained than language); 3) different modalities should be easily compared even for those unseen together during training; and 4) MMV network should be efficiently applicable to visual data coming in the form of dynamic videos or static images. 

The MMV approach requires no annotation or manual efforts to label video data, which is differentiated from previous work that relies on curated annotated datasets.

## 3. Method

The proposed method of the paper is illustrated in Figure 1.

![Untitled](/.gitbook/assets/59/Figure1.png)

Figure 1: Proposed methods of the paper

The goal of MMV network is to embed three different modalities in video data (i.e., visual, audio, and text) into a *joint* semantic vector space where cross-modal similarities can be examined by a simple dot product. To achieve the goal, the paper considers three possible architectural choices for MMV network. In "shared" space option, three modalities are forced to embed onto a single jointly-shared vector space $$\mathcal{S}_{vat}$$, and direct comparisons among different modalities are possible. The drawback, however, is the strong assumption on all modalities having equal granularity, which does not respect their specificities.

In "disjoint" space option, different visual-audio and visual-text spaces, $$\mathcal{S}_{va}$$ and $$\mathcal{S}_{vt}$$, are learned, respectively. This option respects the specificity of different modality pairs, but the obvious drawback is that direct comparison between audio and text modalities is no longer possible. 

In "FAC" (fine and coarse) spaces option, the authors propose learning of two embedding spaces: 1) fine-grained $$\mathcal{S}_{va}$$ by learning jointly visual and audio modalities and 2) lower-dimensional coarse-grained $$\mathcal{S}_{vat}$$ where text is compared with vision and audio. The FAC option requires three different deep neural networks for multimodal representation learning, but it does not suffer from any shortcomings by "shared" and "disjoint" options. 

FAC gives the best option to achieve the goal of MMV network. The paper details on self-supervised training of FAC, which can take place without any form of manual efforts in labeling or annotating training examples. This is because the paper leverages large amounts of videos available on the Internet. The authors have constructed successfully pretext tasks for setting up self-supervised learning that align bi-modal pairs (i.e., vision-audio or vision-text) from extracting the appropriate modality streams from the same location of a video example.

The proposed self-supervised learning is possible by minimizing the multimodal contrastive loss:

$$\mathcal{L}(x) = \lambda_{va} \textrm{NCE}(x_v,x_a) + \lambda_{vt} \textrm{MIL-NCE}(x_v,x_t),$$

where $$\lambda_{va}$$ and $$\lambda_{vt}$$ are regularization parameters controlling loss contributions by the bi-modal pairs $$va$$ and $$vt$$ to the overall loss. NCE is noise contrastive estimation that transforms an unsupervised problem into a (self-)supervised one by computing a loss against a noise distribution. For FAC, negative sampling is used

$$\textrm{NCE}(x_v,x_a) = - \log \left ( \frac{\exp(\frac{z^\top_{v,va} z_{a,va}}{\tau})}{\exp(\frac{z^\top_{v,va} z_{a,va}}{\tau}) + \sum_{z'\sim \mathcal{N}(x)} \exp(\frac{z'^\top_{v,va} z'_{a,va}}{\tau})} \right ).$$

MIL-NCE, multiple instance learning noise contrastive estimation, is given by 

$$\textrm{MIL-NCE}(x_v,x_t) = - \log \left ( \frac{\sum_{z \in \mathcal{P}(x)} \exp(\frac{z^\top_{v,vat} z_{t,vat}}{\tau})}{\sum_{z \in \mathcal{P}(x)} \exp(\frac{z^\top_{v,vat} z_{t,vat}}{\tau}) + \sum_{z'\sim \mathcal{N}(x)} \exp(\frac{z'^\top_{v,vat} z'_{t,vat}}{\tau})} \right ).$$

Finally, MMV network is equipped with deflation operation. Deflation transforms a video network into a network that can ingest a single image. The deflated network is applied to popular image downstream tasks despite using the original (un-deflated) network trained on videos. 

In the paper, two types of video network deflation are considered: 1) 3D convolutional neural network (3D-CNN) summing the 3D spatiotemporal filters over the temporal dimension to obtain 2D filters and 2) temporal shift module (TSM) network turning off the channel shifting that can result in residual architecture (e.g., ResNet) for images.

## 4. Experiment & Result

The paper presents the three main experiments. First, the paper explores various design choices for a multimodal network. After evaluating the choices, the paper elects the best design to scale up multimodal networks and compare them to current state-of-the-art. Finally, by applying the trained video networks on still image tasks, the authors demonstrate the effect of their deflation **approach.

### Experimental setup, datasets and downstream tasks

- Network architectures
    - Video
        - Backbone: S3D-G, TSM with a ResNet50, TSM with a ResNet50x2
        - Temporal and spatial average pooling at the last layer of the backbone to obtain a single vector $$f_v(x_v)$$
        - 32 (16 for the exploration design) frames are sampled at 10 fps and 200 × 200 crops are used
        - Standard augmentation: random crop, horizontal flipping, temporal sampling and scale jittering, and color augmentation
    - Audio
        - Represented as log MEL spectrogram with 80 bins and processed with ResNet50 and is sampled in sync with the frames
        - Spatial pooling is applied to obtain $$f_a(x_a)$$ of dimension $$d_a$$ = 2048
    - Text
        - Processed by removing stop words, retaining a maximum or padding to 16 words, then extracting 300-dimensional
        Google News pre-trained word2vec and finally applying a linear layer to independently map the word inputs to 2048 dimension followed by a max pooling layer over the 16 words ($$d_t$$ = 2048)
        - The dimension of the shared subspaces is 512, except for the Fine And Coarse (FAC) design where we use 512 dimensions for $$\mathcal{S}_{va}$$ (fine) and 256 for $$\mathcal{S}_{vat}$$ (coarse)
- Hyperparameters & Optimization
    - Normalize vectors prior to computing their dot products in the NCE and MIL-NCE losses and use a temperature of τ = 0.07 in the softmax
    - 10:1 loss weight ratio for training on HowTo100M and 1:1 fro HotTo100M+AudioSet
    - Adam with an initial learning rate of 0.002, 5K steps of warm up and a half-period cosine schedule
- Datasets (used for self-supervised pretraining)
    - HowTo100M: 100 millions narrated video clips coming from 1 million unique videos where the audio narration is transcribed into text
    using ASR
    - Train split of AudioSet: consists of 10 seconds clips coming from 2 million different internet videos (text is missing)
- Downstream tasks
    
    To evaluate visual, audio and text representation, they use wide range of downstream tasks. Details are below.

| Task                              | Evaluation                                         | Benchmark (Evaluation Metric)                                                  |
|-----------------------------------|----------------------------------------------------|--------------------------------------------------------------------------------|
| Action Classification             | Visual Representation                              | UCF101 (top-1 accuracy), HMDB51 (top-1 accuracy), Kinetics600 (top-1 accuracy) |
| Audio Classification              | Audio Representation                               | ESC-50 (top-1 accuracy), AudioSet (mAP)                                        |
| Zero-shot text-to-video retrieval | Text-Video Representation                          | MSRVTT (recall at 10), YouCook2 (recall at 10)                                 |
| Image Classification              | Transfer from video representations to image tasks | PASCAL VOC 2007 (mAP), ImageNet (top-1 and top-5 accuracies)                   |

### Results

**Design explorations**

Empirical evaluations on various design choices for multiple modalities are conducted to yield the best design. The key finding is that learning jointly with all three modalities outperforms the models trained on a pair of modalities (bi-modal). Among strategies, fine-and-coarse (FAC) method perform the best.

![Untitled](/.gitbook/assets/59/Result1.PNG)

**Large-scale experiments and comparison to the state-of-the-art**

To compare with state-of-the-art model, they scale up their model with the best architecture determined from design explorations. The result shows the proposed FAC approach outperforms the current state-of-the-art on all downstream tasks including UCF101, HMDB51, Kinetics600, AudioSet, and ESC-50 benchmarks.

![Untitled](/.gitbook/assets/59/Result2.PNG)

**Transfer to image tasks via network deflation**

The best MMV networks trained above are applied on static image tasks to verify the effect of deflation. As a result, the deflated model performs almost similar to the original video model on inflated input (i.e., the entire video instead of a still image). The proposed deflation method outperforms naive deflation, but state-of-the-art self-supervised models trained on images outperform the best MMV networks.

![Untitled](/.gitbook/assets/59/Result3.PNG)

## 5. Conclusion

The paper presents multimodal versatile (MMV) network that ingests vision, audio, and text modalities present in video data.  MMV network can combine the modalities for joint representations that can enhance the performance of downstream tasks. In doing so, the proposed FAC approach enables that fine-grained representations of the visual and audio modalities can be maintained while also integrating coarse-grained text modality into a common embedding. Also, the paper describes a novel process of deflation for an MMV network such that visual data either in the form of video or a static image can be taken in. MMV networks can be trained on a large volume of unlabeled video data found online by the self-supervised learning technique via multimodal contrastive loss. Using the learned multimodal representations, the authors have demonstrated state-of-the-art performance on the UCF101, HMDB51, Kinetics600, AudioSet, and ESC-50 benchmarks. 

Key technical contributions of the paper are: 1) investigation on different modality embedding strategies, namely, shared, disjoint, and FAC with effective self-supervised training  method for multimodal representation of visual, audio, and language streams, 2) deflation approach for an MMV network that can efficiently ingest video or a static image, and 3) demonstration of superior multimodal downstream task performances.

The strength of the paper is readability. Main technical portions of the paper are easy to follow and understand. Simplicity and soundness of the proposed method are also the strength of the paper. The weakness of the paper is that the experimental section is a bit complicated and somewhat difficult to figure out important takeaways. Due to the NeurIPS page limit, some important details should be left out and enclosed in the appendix, which adversely affected the flow of the paper.  

### Take home message (오늘의 교훈)

When multiple modalities are present in given data, do not pick only one modality and focus just on single-modality learning, but leverage all modalities, and more importantly, reveal their relationships and exploit them.

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
