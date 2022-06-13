---
description: (Description) Roland Gao / Rethink Dilated Convolution for Real-time Semantic Segmentation / arXiv 2021
---

# RegSeg \[Eng\]


##  1. Problem definition
In this paper, we want to solve the problem caused by the imagenet backbone used in real time scene segmentation.  
The imagenet backbone used in the existing real time scene segmentation paper, the convolutional layer at the end causes too many channels. For example, 512 resnet18 and 2048 resnet50 are generated. This is a problem that puts a lot of computation on the shoulders of real-time environments.
In addition, the size of the image being input by the imaging models is 224 x 244, whereas the dataset of semantic segmentation is much larger, 1024 x 2048. This means that the field-of-view of the imagenet models is insufficient to encode large images.  
Regseg limits the structure that can reduce the amount of computation and obtain sufficient field-of-view without compromising accuracy.

## 2. Motivation

### Related work
Let's take a brief look at some of the previous studies that have been done to improve both accuracy and computational speed in Segmentation.
* Semantic segmentation
    * Fully Convolutional Networks  
    In order to apply the Classification model to segmentation, all fc-layers have been replaced with Conv-layers.
    * DeepLabv3  
    A variety of dilation to the rates dilated conv imagenet the model, in addition to receptive field the years.
    * PSPNet  
    The Pyramid Pooling Moudle, which adds several layers with different pooling rates in parallel, enables us to learn the Global context information.
    * Deeplabv3+  
    We stabilized learning by adding a decoder and 1 x 1 convolution to Deeplabv3.
* Real-time semantic segmentation
    * BiseNetV2  
    After creating two branches, Spatial Path and Context Path, we combined them to show good performance without a pre-trained ImageNet model.
    * STDC  
    We eliminated BiseNet's Spatial Path and allowed it to go through only one path, making it work faster.
    * DDRNet-23  
    The Deep Aggregation Pyramid Pooling Module (DAPPM), which has added mutual fusion between the two branches, is added to the end of the backbone to show SOTA performance in the Citiescapes dataset.
* Desinging Network design Spaces  
Manual network design has become more difficult as more choices are made in network design. We were able to find many good networks, but we didn't find the principles, so we proposed block-type RegNetY as a new network design paradigm through numerous experiments and simulations.

### Idea
In order for existing Semantic segmentation studies to replace the ImageNet model, real-time semantic segmentation studies have significantly increased the computation volume. For DDRNet-23, 20.0M parameters were used. In this paper, we propose a block structure with dilated conv applied by referring to the blocks in RegNet to reduce computation volume and increase the receive field, and we repeatedly stack them.

## 3. Method

### Dilated block
The author replaced the step of doing 3 x 3 conv in the Y block of RegNet with a dialed conv divided into two parts. It was named Dilated Block and repeated 18 times by changing the dilated rate. The difference between the Y block and the D block can be seen as follows. When all the dilated rates are 1, the D block is the same as the Y block.

![figure 1](/.gitbook/assets/regseg1.png)

The D block with Stride 2 is as follows.

![figure 2](/.gitbook/assets/regseg2.png)

The dilated rate and stride in each D block can be found in the following table. The multi-scale features could be extracted while varying each dilated rate.

![figure 3](/.gitbook/assets/regseg3.png)

The dilated rate and stride in each D block can be found in the following table. The multi-scale features could be extracted while varying each dilated rate.

### Decoder
Decoders have been added to recover lost local datils from the above backbone. It receives 1/4, 1/8, and 1/16 size feature maps from Backbone and merges through 1x1 conv and upsampling. The simple structure of the decoder does not significantly increase the computation.

![figure 4](/.gitbook/assets/regseg4.png)

## 4. Experiment & Result

### Experimental setup
In this paper, we conducted an experiment comparing performance with state-of-the-art models including DDRNet-23 in Citiescapes and CamVid. Training setup for Citiescapes is as follows.

* momentum 0.9 SGD
* initial learning rate: 0.05
* weight decay: 0.0001
* ramdon scaling [400, 1600]
* random cropping 768 x 768
* 0.5% class uniform sampling
* batch size = 8, 1000 epochs

Camvid uses the Citycapes pretrained model, and the differences from the Cityscapes experimental environment are as follows.
* random horizontal flipping
* random scaling of [288, 1152]
* batch 12, 200 epochs
* Do note use classuniform sampling

### Result

#### Cityscapes
The results from Citiescapes are as follows:

![figure 5](/.gitbook/assets/regseg5.png)

Although FPS between models cannot be directly compared, RegSeg is 1.5%p higher than HardDNet, an additional data-free SOTA model, and outperforms SFNet by 0.5%p with the best peer review results.

![figure 6](/.gitbook/assets/regseg6.png)

Cityscapes test set balances the best accuracy and parameters.

#### Ablation Studies
It can be seen that small dilation rates are used in the front and large dilation rates are used in the back, but increasing the filed-of-view recklessly does not lead to an improvement in accuracy.

![figure 7](/.gitbook/assets/regseg7.png)

## 5. Conclusion
* Although we did not reduce the parameters while maintaining the accuracy of the DDRNet-23, we showed good performance in real-time segmentation with significant replacement costs.
* Dilated conv for increasing field-of-view was used from DeepLab, but it was effective in reducing the number of parameters while reducing the number of branches to two.
* Through considerable experiments, there has been a contribution to finding an efficient dilated rate and structure.

### Take home message

> Dilated conv branch is efficient to build depth while minimizing it.
>
> Randomly increasing the field-of-view does not necessarily improve accuracy.

### Author

**이명석 \(MyeongSeok Lee\)** 

* M.S Student in School of ETRI, UST (Advisor: [_Prof. ChiYoon Chung_](https://etriai.notion.site/))
* ims@etri.re.kr


## Reference & Additional materials

1. Gao, R. (2021). Rethink Dilated Convolution for Real-time Semantic Segmentation. arXiv preprint arXiv:2111.09957.
2. Radosavovic, I., Kosaraju, R. P., Girshick, R., He, K., & Dollár, P. (2020). Designing network design spaces. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10428-10436).
