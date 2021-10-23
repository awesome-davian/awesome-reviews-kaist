---
Description: Ledig et al. / Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network / CVPR 2017 Oral
---

# SRResNet and SRGAN \[Eng\]

##  1. Problem Definition

Recent convolutional neural networks (CNN) being employed to address super-resolution (SR) problems have become both faster and deeper. These networks usually minimize the pixel-wise mean squared error (MSE) between the SR image and the ground truth, producing high peak signal-to-noise ratio (PSNR). Despite this, the models usually face difficulty recovering finer texture details and the resulting SR images are not perceptually appealing. To address this problem, the authors of this paper proposed a new architecture that can super-resolve low resolution (LR) images with $${4\times}$$ upscaling factor and, at the same time, recover high-frequency details.

## 2. Motivation

### Related Work

Early approaches in tackling the SR problem involved filtering methods like linear, bicubic, and Lanczos filtering [2]. However, despite being fast, they produce SR images with overly smooth textures.

CNN-based single image super resolution (SISR) recently gained attention because it showed superior performance over other methods of upscaling. One prominent network is described in [3] where images were upscaled using bicubic interpolation and a CNN was trained to produce state-of-the-start SR outputs. Better performance was also observed in [4] when networks are allowed to learn their own upscaling filters. And in order to produce visually convincing results, this paper relied heavily on the loss functions of [5] and [6].

### Idea

This paper proposes a deep network architecture for $${4\times}$$ upscaling of LR images—the SRResNet which used residual blocks and skip-connections. The model managed to learn its own upscaling filters and was trained using the MSE loss.

Further, the perceptual loss (combination of the content loss and adversarial loss) was used in this paper in order to train the SRGAN. The MSE-trained SRResNet served as the generator network of the GAN.

## 3. Method

### SRResNet / Generator

The proposed SRResNet contains 16 residual blocks. Each residual block has two convolutional layers that used 64 $${3\times3}$$ filters. Batch normalization and the Parametric ReLU followed the convolutional layer. Each residual block has skip-connection. And in order to achieve $${4\times}$$ upscaling, two sub-pixel layers were added at the end of the network. The figure below shows the SRResNet architecture.

![Figure 1. Generator G.](../../.gitbook/assets/12/generator.PNG)

As mentioned above, the SRResNet was trained using the MSE loss and was used as the generator for the GAN.

### Discriminator

A discriminator was also trained to solve the maximization problem. Instead of the Parametric ReLU used in the Generator, the Leaky ReLU (with $${\alpha=0.2}$$) was chosen as non-linearity. Its convolutional layers used $${3\times3}$$ filters, doubling the number of kernels for every layer of convolution starting from 64 up to 512. Strided convolutions were used to reduce the image size instead of pooling layers. The network was then terminated by two dense layers and a sigmoid function to determine whether the image is the original HR or generated SR. The figure below shows the architecture of the Discriminator network used in the paper.

![Figure 2. Discriminator D.](../../.gitbook/assets/12/discriminator.PNG)

### Loss Functions

#### MSE Loss

For the SRResNet architecture, the authors used the MSE loss. However, although it achieved higher PSNR and SSIM values, this loss generated blurry SR images, i.e. the images lack finer details. The MSE loss is shown below.

![Figure 3. MSE Loss.](../../.gitbook/assets/12/mse_loss.PNG)

#### Perceptual Loss

The SRGAN, on the other hand, utilized a different loss function which the authors call the perceptual loss. It is the combination of the content loss (VGG loss) and the adversarial loss. The equation below shows the perceptual loss.

![Figure 4. Perceptual Loss.](../../.gitbook/assets/12/perceptual_loss.PNG)

##### *VGG Loss*

The VGG loss (content loss) puts more importance on the perceptual instead of the pixel space similarity. The VGG loss is shown below where $${\phi_{i,j}}$$ is the *j*-th convolution (after activation) before the *i*-th maxpooling layer of the VGG19 network.

![Figure 5. VGG Loss.](../../.gitbook/assets/12/vgg_loss.PNG)

##### *Adversarial Loss*

The discriminator network was trained to tell the difference between the generated SR and the HR images using the adversarial loss shown below.

![Figure 6. Adversarial Loss.](../../.gitbook/assets/12/adversarial_loss.PNG)

## 4. Experiment & Result

### Experimental setup

The authors randomly selected 350 thousand images from the ImageNet dataset and downscaled them with a scale of 4 using bicubic interpolation to obtain LR images. Adam optimizer was used with a learning rate of 10<sup>-4</sup> and each mini-batch of training consisted of 16 random 96 x 96 patches generated from the training images. The authors employed an NVIDIA Tesla M40 GPU to train their model.

After training the MSE-based SRResNet, it was used as a generator for the GAN training. The learning rate was 10<sup>-4</sup> for the first 10<sup>5</sup> iterations and 10<sup>-5</sup> for next 10<sup>5</sup> iterations.

The performance of the proposed architecture was measured by comparing the PSNR and SSIM values of the SRResNet, SRGAN, nearest neighbor, bicubic, SRCNN, SelfExSR, DRCN, and ESPCN on the Set5, Set14, and BSD100 datasets. Further, by asking 26 people to rate the SR images, mean opinion score (MOS) test was performed to quantify how convincing the outputs are.

### Result

The SRResNet with MSE loss recorded the highest PSNR and SSIM values as shown in the figure below.

![Figure 7. SRResNet and SRGAN Results.](../../.gitbook/assets/12/srresnet_srgan_result.PNG)

However, the SRGAN obtained higher MOS amongst all the given SR methods. The details are shown in the table below.

![Figure 8. MOS Results.](../../.gitbook/assets/12/mos_result.PNG)

The next table shows the comparison in the PSNR, SSIM, and MOS of the SRResNet and SRGAN with other SR methods. Note that the SRResNet outperformed the other methods in terms of PSNR, SSIM, and MOS that are existing at the time.

![Figure 9. PSNR, SSIM, and MOS of Various SR Networks.](../../.gitbook/assets/12/benchmark.PNG)

Reconstruction results are provided in the figure below. Notice that the SRGAN-VGG54 managed to retrieve better texture details.

![Figure 10. Reconstruction Results.](../../.gitbook/assets/12/reconstruction_results.PNG)

## 5. Conclusion

In conclusion, the authors successfully designed, built, and trained a network that can super-resolve LR images with $${4\times}$$ upscaling factor. The SRResNet obtained state-of-the-art PSNR and SSIM values for the SR images it produced while the SRGAN, although it recorded relatively lower PSNR and SSIM values compared to the former, obtained higher MOS. The SRResNet produced blurry SR images due to its MSE loss while the SRGAN, with its perceptual loss, managed to super-resolve more photo-realistic images.

### Take Home Message

> Despite producing smaller PSNR and SSIM values than the SRResNet, the SRGAN can generate more realistic and perceptually acceptable SR images.

## Author / Reviewer Information

### Author

**Samuel Teodoro** 

* KAIST
* Email: sateodoro@kaist.ac.kr

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. Korean name \(English name\): Affiliation / Contact information

## Reference & Additional Materials

1. C. Ledig, L. Theis, F. Huszar, J. Caballero, A. Cunningham, A. Acosta, A. Aitken, A. Tejani, J. Totz, Z. Wang, et al. Photorealistic single image super-resolution using a generative adversarial network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4681–4690, 2017. ([link](https://arxiv.org/pdf/1609.04802.pdf))
2. C. E. Duchon. Lanczos Filtering in One and Two Dimensions. In Journal of Applied Meteorology, volume 18, pages 1016–1022. 1979. ([link](https://journals.ametsoc.org/view/journals/apme/18/8/1520-0450_1979_018_1016_lfioat_2_0_co_2.xml))
3. C. Dong, C. C. Loy, K. He, and X. Tang. Learning a deep convolutional network for image super-resolution. In European Conference on Computer Vision (ECCV), pages 184–199. Springer, 2014. ([link](https://arxiv.org/pdf/1501.00092.pdf))
4. W. Shi, J. Caballero, F. Huszar, J. Totz, A. P. Aitken, R. Bishop, D. Rueckert, and Z. Wang. Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1874–1883, 2016. ([link](https://arxiv.org/pdf/1609.05158.pdf))
5. J. Johnson, A. Alahi, and F. Li. Perceptual losses for real-time style transfer and super-resolution. In European Conference on Computer Vision (ECCV), pages 694–711. Springer, 2016. ([link](https://arxiv.org/pdf/1603.08155.pdf))
6. J. Bruna, P. Sprechmann, and Y. LeCun. Super-resolution with deep convolutional sufficient statistics. In International Conference on Learning Representations (ICLR), 2016. ([link](https://arxiv.org/pdf/1511.05666.pdf))
