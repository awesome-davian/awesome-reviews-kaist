---
description: Yulun Zhang et al. / Image Super-Resolution Using Very Deep Residual Channel Attention Networks / ECCV 2018
---

# Image Super Resolution via RCAN \[Eng\]

한국어로 쓰인 리뷰를 읽으려면 [**여기**](/paper-review/2021-fall-paper-review/eccv-2018-rcan-kor.md)를 누르세요.

##  1. Problem definition

<p align="center"><img src = "/.gitbook/assets/63/0srex.PNG" height = "300"></center>

The Single Image Super-Resolution (SISR) technique aims to restore a low resolution (LR) image to a high resolution (HR) while removing blur and various noises in the image. SR is expressed as an equation as follows, where x and y are LR and HR images, respectively.
$$
\textbf{y}=(\textbf{x} \otimes \textbf{k} )\downarrow_s + \textbf{n}
$$
where y and x denote high and low resolution image, respectively, and k and n mean blur and noise matrix, respectively. Recently, CNN-based SR has been actively studied, since CNN works effectively on SR, However, CNN-based SR has the following two limitations.

* Gradient Vanishing [Note i] occurs as the layer deepens, making learning more difficult

* The representativeness of each feature map is weakened as low-frequency information included in the LR image is treated equally in all channels.

To overcome the aforementioned goals of SR and the above two limitations, this paper proposes Deep-RCAN (Residual Channel Attention Networks).

> [Note i] **Gradient Vanishing**: As the input value goes through the activation function, it is squeezed into a small range of output values, so it means the state that the initial input value has little effect on the output value as it goes through the activation functions of several layers. Accordingly, the rate of change of the parameter values of the initial layers with respect to the output becomes small, making learning impossible.

## 2. Motivation

### **2.1. Related work**

The papers related to deep-CNN and attention technique, which are the baselines of this paper, are as follows.

#### **1. CNN based SR**

* **[SRCNN & FSRCNN]**: SRCNN, the first technique applying CNN to SR, significantly improved performance compared to existing Non-CNN based SR techniques by constructing a 3-layer CNN. FSRCNN simplifies the network structure of SRCNN to increase inference and learning speed.
* **[VDSR & DRCN]**: By stacking layers deeper than SRCNN (20 layers), the performance is greatly improved.
* **[SRResNet & SRGAN]**: SRResNet was the first to introduce ResNet to SR. In SRGAN, photo-realistic SR was implemented by mitigating blur by introducing GAN to SRResNet. However, there are cases where an unintentional artifact object is created.
* **[EDSR & MDSR]**: By removing unnecessary modules from the existing ResNet, the speed is greatly increased. However, it cannot implement the deep layer, which is the key in image processing, and has limitations in that it includes unnecessary calculations and does not represent various features by treating low-frequency information equally in all channels.

#### **2. Attention Method**

Attention is a technique for biasing processing resources on a specific part of interest in input data, and increases the processing performance for that part. Until now, attention has been generally used for high-level vision tasks such as object recognition and image classification, and has hardly been dealt with in low-level vision tasks such as image SR. In this paper, attention is applied to the high-frequency region in the LR image to enhance the high-frequency constituting the high-resolution (HR) image.

### **2.2. Idea**
The idea of the paper and its contribution can be summarized in the following three categories.

#### **1. Residual Channel Attention Network (RCAN)**
Through Residual Channel Attention Network (RCAN), a more accurate SR image is obtained by layering more deeply than the existing CNN-based SR.

#### **2. Residual in Residual (RIR)**

By building deeper layers that are trainable through Residual in Residual (RIR), and bypassing low-frequency information of low-resolution images with long and short skip connections inside RIR blocks, more efficient neural networks can be designed.

#### **3. Channel Attention (CA)**

By considering interdependencies between feature channels through Channel Attention (CA), adaptive feature rescaling is possible.


## 3. Residual Channel Attention Network (RCAN)
### **3.1. Network Architecture**

<p align="center"><img src = "/.gitbook/assets/63/1Modelarchitecture.PNG" height = "280"></center>

The network structure of RCAN is mainly composed of 4 parts: i) Shallow feature extraction, ii) RIR deep feature extraction, iii) Upscale module, and iv) Reconstruction part. In this paper, one convolutional layer, deconvolutional layer, and L1 loss are used for i), iii), and iv), respectively, similar to the existing EDSR technique. ii) Contributions to CA and RCAB, including RIR deep feature extraction, are introduced in the next section.
$$
L(\Theta  )=\frac{1}{N}\sum_{N}^{i=1}\left \| H_{RCAN}(I_{LR}^i)-I_{HR}^i   \right \|_1
$$

### **3.2. Residual in Residual (RIR)**
RIR consists of G blocks consisting of a residual group (RG) and a long skip connection (LSC). In particular, one RG consists of B operations in units of residual channel attention block (RCAB) and short skip connection (SSC). With this structure, it is possible to form more than 400 CNN layers. Since piling only RG deeply has limitations in terms of performance, LSC is introduced at the end of the RIR to stabilize the neural network. In addition, by introducing LSC and SSC together, unnecessary low-frequency information in the LR image can be bypassed more efficiently.

### **3.3. Residual Channel Attention Block (RCAB)**

<p align="center"><img src = "/.gitbook/assets/63/2channelattention.PNG" height = "150"</center>

In this paper, the Residual Channel Attention Block (RCAB) is proposed by merging Channel Attention (CA) with the Residual Block (RB). In particular, to overcome the fact that CNN cannot use overall information other than the local region by considering only the local receptive field, CA expressed spatial information using global average pooling.

<p align="center"><img src = "/.gitbook/assets/63/4RCAB.PNG" height = "150"></center>
  
On the other hand, in order to show the correlation between channels, a gating mechanism [Note ii] was additionally introduced. In general, the gating mechanism should exhibit nonlinearity between channels, and the mutually exclusive relationship should be learned while the features of multiple channels are emphasized compared to one-hot activation. To meet these criteria, sigmoid gating and ReLU were selected.

> [Note ii] **Gating Mechanisms**: Gating mechanisms were introduced to address the vanishing gradient problem and have proven to be crucial to the success of RNNs. This mechanism essentially smooths out the update. [Gu, Albert, et al. "Improving the gating mechanism of recurrent neural networks." International Conference on Machine Learning. PMLR, 2020.]


## 4. Experiment & Result
### **4.1. Experimental setup**
#### **1. Datasets and degradation models**

<p align="center"><img src = "/.gitbook/assets/63/7dataset.PNG" height = "350"></center>

Some 800 images of the DIV2K dataset were used for training images, and Set5, B100, Urban 100, and Manga109 were used as test images. As the degradation models, bicubic (BI) and blur-downscale (BD) were used.

#### **2. Evaluation metrics**
The Y channel of the YCbCr color space [Note iii] of the PSNR and SSIM-processed images was evaluated. Also, compared with other SR techniques ranked 1st to 5th in recognition error, the performance advantage was confirmed.

> [Note iii] **YcbCr**: YCbCr, Y′CbCr, or Y Pb/Cb Pr/Cr, also written as YCBCR or Y′CBCR, is a family of color spaces used as a part of the color image pipeline in video and digital photography systems. Y′ is the luma component and CB and CR are the blue-difference and red-difference chroma components. Y′ (with prime) is distinguished from Y, which is luminance, meaning that light intensity is nonlinearly encoded based on gamma corrected RGB primaries. [Wikipedia]

#### **3. Training settings**
Data augmentation such as rotation and vertical inversion was applied to 800 images in the aforementioned DIV2K dataset, and 16 LR patches with a size of 48x48 were extracted as inputs from each training batch. ADAM was used as an optimization technique.

### **4.2. Result**
### **1. Effects of RIR and CA**

<p align="center"><img src = "/.gitbook/assets/63/5result.PNG" height = "150"></center>

While the existing technique showed performance of 37.45dB, by using RIR and CA including long skip connection (LSC) and short skip connection (SSC), the performance was increased to 37.90dB.

### **2. Model Size Analyses**

<p align="center"><img src = "/.gitbook/assets/63/6result2.PNG" height = "220"></center>

Compared to other techniques (DRCN, FSRCNN, PSyCo, ENet-E), RCAN achieves the deepest neural network, has the smallest number of parameters, but shows the highest performance.

## 5. Conclusion
In this paper, RCAN is applied to obtain high-accuracy SR images. In particular, by utilizing the RIR structure together with LSC and SSC, it was possible to form a deep layer. In addition, RIR allows the neural network to learn high-frequency information by bypassing the low-frequency information, which is unnecessary information in the LR image. Furthermore, channel-wise features were adaptively rescaled by introducing CA and considering interdependencies between channels. The proposed technique verified the SR performance using the BI and DB degradation models, and it was confirmed that it also showed excellent performance in object recognition.

## Take home message \(오늘의 교훈\)
> By segmenting the information in the area of interest in the image and applying attention to the information, the weight can be increased in the learning process for interest part.

> It is more effective to increase the performance by building the neural network deeper than increasing the total number of parameters.

## Author / Reviewer information
### 1. Author

**한승호 \(Seungho Han\)** 
* KAIST ME
* Research Topics: Formation Control, Vehicle Autonomous Driving, Image Super Resolution
* https://www.linkedin.com/in/seung-ho-han-8a54a4205/

### 2. Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. **[Original Paper]** Zhang, Yulun, et al. "Image super-resolution using very deep residual channel attention networks." Proceedings of the European conference on computer vision (ECCV). 2018.
2. **[Github]** https://github.com/yulunzhang/RCAN
3. **[Github]** https://github.com/dongheehand/RCAN-tf
4. **[Github]** https://github.com/yjn870/RCAN-pytorch
5. **[Attention]** https://wikidocs.net/22893
6. **[Dataset]** Xu, Qianxiong, and Yu Zheng. "A Survey of Image Super Resolution Based on CNN." Cloud Computing, Smart Grid and Innovative Frontiers in Telecommunications. Springer, Cham, 2019. 184-199.
7. **[BSRGAN]** Zhang, Kai, et al. "Designing a practical degradation model for deep blind image super-resolution." arXiv preprint arXiv:2103.14006 (2021).
8. **[Google's SR3]** https://80.lv/articles/google-s-new-approach-to-image-super-resolution/
9. **[SRCNN]** Dai, Yongpeng, et al. "SRCNN-based enhanced imaging for low frequency radar." 2018 Progress in Electromagnetics Research Symposium (PIERS-Toyama). IEEE, 2018.
10. **[FSRCNN]** Zhang, Jian, and Detian Huang. "Image Super-Resolution Reconstruction Algorithm Based on FSRCNN and Residual Network." 2019 IEEE 4th International Conference on Image, Vision and Computing (ICIVC). IEEE, 2019.
11. **[VDSR]** Hitawala, Saifuddin, et al. "Image super-resolution using VDSR-ResNeXt and SRCGAN." arXiv preprint arXiv:1810.05731 (2018).
12. **[SRResNet ]** Ledig, Christian, et al. "Photo-realistic single image super-resolution using a generative adversarial network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
13. **[SRGAN]** Nagano, Yudai, and Yohei Kikuta. "SRGAN for super-resolving low-resolution food images." Proceedings of the Joint Workshop on Multimedia for Cooking and Eating Activities and Multimedia Assisted Dietary Management. 2018.
