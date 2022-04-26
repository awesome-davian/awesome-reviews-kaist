---
description: Vaksman et al. / Patch Craft; Video Denoising by Deep Modeling and Patch Matching / ICCV 2021
---

# Patch Craft [Eng]


## Problem definition

The problem under study is video denoising. For many years, it has been an active reaserch area. Deep learning models have recently brought improvements to this field. The authors of this paper propose a model combining features from different existing models. 

## Motivation


### Related work

Existing work are divided in two categories: the "classically-oriented" algorithms and the convolutional neural networks (CNN). The former models utilize a property of video sequences called "self-similarity". It is an observation that the images take from nature have similar local elements which occur several times in the image. The algorithm looks for these similarities by splitting the image into overlapping patches. The most famous of them would be Non-Local-Means algorithm which averages the patches but other more elaborated methods build more complex models such as graphs or binary trees. 
Recently, CNN have shown massive improvements and can now compete and even surpass classical algorithms which have progressively been abandoned. The main difference is that CNN consider the whole image and does not split into smaller pieces anymore. Thus, they don't harness the self similarity, nor the time redundancy, which are powerful assets of video processing.  

### Idea

The idea is to create an algorithm which combines the two previous ones. This model would harness non-local self-similarity within a CNN architecture. Using the "patch-craft frame" concept, the images are decomposed in overlapping patches. Then, the algorithm looks for the nearest neighbours in a space-time window (space and time redundancy) which are then augmented (i.e artificially modified to provide more data) and used in a denoising CNN. To avoid heavy computations, they use a CNN with "separable convolutional layers) (SepConv).This algorithm works in two phases: the space filtering and the time filtering. Space filtering aims to feed a CNN based on self-similatiry and time filtering is used to ensure time consistency between the frames of the video. 

## Method

The spatial denoising network gets as input a group of patches. Since a conventional CNN could not handle this amount of data, the algorithm use a StepConv layer composed of three filters. The input data in the network is a five dimensional vector of input size $$n_{in}*f_{in}*c*v*h$$ and output size $$n_{out}*f_{out}*c*v*h$$ with v*h the grame size, c the number of colours, f the patch size and n the number of neighbours.
The first layer is a filter on dimensions v and h, the second one gives an intermediate output of dimension $$n_{in}*f_{out}*c*v*h$$ and the last one gives the final output. The network is composed of many SepConv layers. The first block is a SepConv followed by a ReLU activation function, the last one is just a SepConv layer and all the blocks in between are composed of SepConv followed by a Batch Normalization then followed by a ReLU function. 

Then, for the temporal filtering, the algorithm uses another CNN using a window of frames, the number of frame is a hyperparameter.

## Experiment and result

### Experimental setup

The train dataset is a set of 90 video sequences at 480p resolutionextracted from the DAVIS dataset.
The test phase is a comparison of the performances of this algorithm, called PaCNet, and other classically oriented algorithms and convolutional networks. 
The training has two phases: first the spatial part is trained alone and when the parameters are fixed the temporal part is trained. Here are some parameters used for the test: patch size 7x7, 14 neighbours, five blocks in the spatial CNN and seven frames for the time window. The total number of trainable parameters is 2.87E6.
The evaluation metric is the PSNR (Peak Signal to Noise Ratio), defined by the relation $$PSNR = 10log_{10}(\frac{MAX^{2}_{I}}{MSE})$$.

### Result

According to the table, the PaCNet performs better in all cases and especially when the noise is high. Compared to other CNN models, PaCNet improves the performances in a 0.8 to 1.4dB range. Another interesting test is with the same spatial CNN but without the temporal CNN: the improvement reaches 0.8dB.

## Conclusion

This work provides a new model for video denoising by harnessing both advantages from classical algorithms and convolutional networks, respectively via image properties such as self-similarity or time redundancy and image augmenting.



## Reference and additional material

1. Antoni Buades, Bartomeu Coll, and J-M Morel. A non-local algorithm for image denoising. In 2005 IEEE Computer So- ciety Conference on Computer Vision and Pattern Recogni- tion (CVPR’05), volume 2, pages 60– 65. IEEE, 2005.

2. Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbela é z, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017.

3. Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and Lei Zhang. Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising. IEEE transactions on image processing, 26(7):3142–3155, 2017.
