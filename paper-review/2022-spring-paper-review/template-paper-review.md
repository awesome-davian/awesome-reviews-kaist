---
description: Zhang et al. / DatasetGAN - Efficient Labeled Data Factory with Minimal Human Effort
 / CVPR 2021
---

# DatasetGAN \[Eng\]

##  1. Problem definition

Current deep networks are extremely data hungry, and often require tens of thousands of labeled data to train (sometimes even more). This process of labeling is often a very expensive and time consuming one. It is especially complex in segmentation tasks, where for certain types of images, it takes a human annotator a couple hours to annotate a few images. The authors propose DatasetGAN, a network able to generate synthetic images and its corresponding segmentation labels at a fairly high precision. All you need is a few labeled examples (10~20) for the network to learn from this, and generate by itself an indefinite amount of synthetic data.

## 2. Motivation

### Related work

Recently, GANs have shown a great capacity for synthesizing high-quality images. The authors take inspiration especially from the StyleGAN’s network architecture, which has the particularity of having an additional mapping layer in the generator: this provides a disentanglement between the object identity and its underlying visual features (viewpoint, texture, color …).

Similar works of semi-supervised learning have been employed to amortize the need for large annotated datasets by employing a large set of unlabeled images and a small set of annotated images. [1] and [2] use this kind of learning to perform image segmentation by treating the segmentation network as a generator and training it adversarially with the small set of real annotations. Pseudo-labels and consistency regularization have also been studied in [3] and [4] to train semantic segmentation networks. 

### Idea

The key insight of DatasetGAN is that, in order to produce highly realistic images, generative models such as StyleGAN must learn very accurate and rich semantic knowledge in their high dimensional latent space. The authors try to exploit and interpolate the disentangled dimensions in StyleGAN in order to teach the network proper labeling, the intuition being that if the network sees a human annotation for one latent code, it will be able to propagate this knowledge across its latent space.

To recall, StyleGAN brings a novelty to the traditional GAN architecture, which is that it disentangles the latent space allowing us to control the attributes at a different level. The disentanglement is done by introducing an additional mapping network that maps the input z (noise/random vector sampled from a normal distribution) to separate vector w and feed it into different levels of the layers. Hence, each part of the z input controls a different level of features.

## 3. Method

Using a StyleGAN backbone for the generator, the architecture of DatasetGAN adds a Style Interpreter block which will take as input the feature maps coming from the AdaIN (adaptive instance normalization) layers of the StyleGAN and output the segmentations and labels corresponding to the image-annotation pair.

### Style Interpreter

The style interpreter is the main component in the DatasetGAN architecture, and can be seen as the label-generating branch. As the feature maps {S^0, S^1,...S^k} are extracted from the AdaIN layers of StyleGAN, they are first upsampled to the highest resolution (resolution of S^k) and then concatenated into a 3D feature tensor. A three-layer MLP classifier is then trained on top of each feature vector to predict labels. $equation$

The way the training process works is that we first obtain a small number of synthesized images from StyleGAN. Afterwards, a human annotator annotates these images with a desired set of labels. A simple ensemble of MLP classifiers is trained on top of the preprocessed pixel-wise feature vectors to match the target human-provided labeling. A few annotated examples is sufficient for achieving good performance.

### Loss functions

![Figure 1: You can freely upload images in the manuscript.](../../.gitbook/assets/2022spring/51/mIOU.png)

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

