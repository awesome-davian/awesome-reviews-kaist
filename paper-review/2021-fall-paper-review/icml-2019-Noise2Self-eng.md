---
description: Batson et al. / Noise2Self Blind Denoising by Self-Supervision / ICML 2019
---

# Noise2Self: Blind Denoising by Self-Supervision \[Eng\]
[Batson Joshua, and Loic Royer, "Noise2self: Blind denoising by self-supervision.", International Conference on Machine Learning. PMLR, 2019.
](https://arxiv.org/abs/1901.11365)

---&gt; 한국어로 쓰인 리뷰를 읽으려면 [**여기**](icml-2019-Noise2Self-kor.md)를 누르세요.

In this paper, they propsed a self-supervision denoising method without a clean image.


##  1. Problem definition
### - Traditional denoising methods & Supervised-learning method :          
In order to denoise using the traditional method, it was necessary to pre-train the noise property of the input image. In this case, it is difficult to fit properly when new noises that I have not trained on go into the input. 
                      
$$||f_{Θ}(x)-y||^2$$             
In the other case, it can be tranied by pairing (x, y) of a noisy image and a clean image. As you can see the above function, it aims to minimize the difference between the output of the denoise function f<sub>Θ</sub>(x) and the ground truth y. Altough it can be used for the convolution neural network and showed good performance in various areas, it requires a long time to train.


In this paper, they propsed a self-supervision denoising method which performed better than the traditional denoising methods and can train denoise without a clean image.
### - $J$-invariant, independet of the input image  :          
<img src="../../.gitbook/assets/18/J_invariant.png" width="20%" height="20%" alt="J_invariant"></img>       
In order to use the self-supervision method, we need to automatically generate label values from unlabeled datasets through special processing. This paper propsed the `J-invariant`, which is a space independent of the input image, as the processing method.
In the image above, x is the dimension of the input image. J is a partition of the dimensions {1, ..., m} and J ∈ $J$. f is a $J$-invariant function that the value of f(x) restricted to dimensions in J. The value of $J$-invariant fuction, f(x)<sub>J</sub>, is independent of x<sub>J</sub>.
    
### - self-supervised loss :     
$$L(f) = E||f(x)-x||^2$$       
In self-supervised case, the result of f(x) is similar with a label in the supervised learning. If J ∈ $J$ is independent from the noise, the self-supervised loss is the sum of the supervised loss and the variance of the noise like the formula below.           
$$E||f(x)-x||^2 = E||f(x)-y||^2 + E||x-y||^2$$     

Therefore, we can find the optimal $J$-invariant function f(x) by minimizing the self-supervised loss like the supervised loss.

## 2. Motivation
### Related work
Here are various ways to remove noise.

#### 1) Traditional Methods
- Smoothness : This is a method of removing noise by calculating the average value of the surrounding pixels to make the center pixel similar to that of the surrounding pixels.
- Self-Similarity : If there are similar patches in the image, replacing the central pixel value with a weighted average value between similar patches. However, the hyperparameters have a large impact on performance, and new datasets unknown noise distribution are unlikely to see the same performance.

#### 2) Use the Convolutional Neural Nets
- Generative : Differentiable generative models can denoise the data using generative adversarial loss.
- Gaussianity : If the noise follows an indepentent identically distributied (i.i.d) Gaussian distribution, use Stein's unbiased risk estimator to train the neural network.
- Sparsity : If the image is sparse, it can be used a compression algorithm to denoise. Howerver, in this case, artifacts remain in the image and it needs a long time to seek sparse features.
- Compressibility : Noise is removed by compressing and decompressing noisy data..
- Statistical Independence : UNet, which is trained to predict true noise by measuring independent noise from the same input data, can predict the real signals(Noise2Noise).

### Idea
There are many methods on how to denoise images like traditional methods such as smoothness or using convolutional neural nets such as UNets recently. However, these methods were possible only when we know the noise property in advance or there was a clean image. So in this paper, they propsed the denoising method based on `self-supervision` rather than the supervised learning method

## 3. Method
### - classic denoiser vs donut denoiser              
<img src="../../.gitbook/assets/18/denoiser.png" width="50%" height="50%"   alt="denoiser"></img> 
> - classic denoiser : Using a median filer that replaces each pixel with the median of a disk of radius r → g<sub>r</sub>
> - donut denoiser : Same as classic denoiser except that the center part is removed → f<sub>r</sub>             

In the graph above, you can see the difference for each denoiser. r is the radius of each filter.       
For the donut denoiser (blue), the self-supervised minimum (red arrow) is same (r=3) with the ground truth minimum. The vertical difference between self-supervised and ground truth means the variance of the noise.         
On the other hand, in the case of classic denoiser (orange), self-supervised MSE continues to increase and there is no correlation with ground truth results.      
In other words, the donut denoiser can adjust the loss value with self-supervised, but the classic denoiser can adjust the loss value only when there is a ground truth.         


### - $J$-invariant function : f<sub>Θ</sub>            
$$f_{Θ}(x)_{J} := g_{Θ}(1_{J}ㆍs(x) + 1_{J^c}ㆍx)_{J}$$   
$J$-invariant f<sub>Θ</sub> function can be defined as above. g<sub>Θ</sub> is any classical denoiser, and J(J ∈ $J$) is any partition of the pixels to distinguish it from adjacent pixels like a mask. s(x) is the function replacing each pixel with the average of its neighbors (interpolation). That is, f<sub>Θ</sub> function interpolates with s(x) only in the area corresponding to J, and applies the original image x to other areas($J^c$), then applies the classical denoiser.           
f<sub>Θ</sub>(x)<sub>J</sub> gets independent results with x<sub>J</sub> because g<sub>Θ</sub> was applied after interpolation of x in $J$ space. As a result, image x performed better when g<sub>Θ</sub> was applied after interpolation than when applied directly to the classical denoiser g<sub>Θ</sub>. 


## 4. Experiment & Result
### Experimental setup
|   Dataset  | Hanzi | CellNet |   ImageNet   |
|:----------:|:-----:|:-------:|:------------:|
| Image size | 64x64 | 128x128 | 128x128(RGB) |
| batch size |   64  |    64   |      32      |
|    epoch   |   30  |    50   |       1      |                           

They compared the denoise performance when self-supervised by applying the $J$-invariant function. There are three data sets: Hanzi, a Chinese character data set, CellNet, a microscope data set and an ImageNet data set. Unet and DnCNN were used to compare the performance of each. They use a random partition of 25 subsets for $J$-invariant and Peak-Signal-to-Noise Raio (PSNR) was used as an evaluation metric. A larger value of PSNR means less loss of image quality.

### Result
<img src="../../.gitbook/assets/18/result1.png" width="40%" height="40%"   alt="result1"></img>   
The table above shows the PSNR results according to each data and denoise architecture. Noise2Self(N2S) performed better than NLM and BM3D, which are traditional denoiser methods, and shows similar performance to Noise2Truth(N2T) trained with clean target and Noise2Noise(N2N) trained together with independent noise.     

<img src="../../.gitbook/assets/18/result2.png" width="50%" height="40%"   alt="result2"></img>     
When looking at the result of denoising as an image, N2S performed better at removing noise than NLM and BM3D and showed similar results to N2N and N2T.


## 5. Conclusion
Noise2Self removes noise in a self-supervision method, unlike other denoising methods. The advantage of this model is that it can remove noise without prior learning about the noise and can be trained without a clean image. However, there is a trade-off between bias and variance depending on how the size of J is set.
     

### Take home message
> Self-supervised learning can be used to learn without target data.
>
> The noise data and the result of $J$-invariant function f(x) are independent of each other.
>
> With self-supervised learning, it can denoise only with the noise data and the result of $J$-invariant function, without clean data.

## Author / Reviewer information
### Author

**황현민** 
* KAIST AI
* [GitHub Link](https://github.com/HYUNMIN-HWANG)
* hyunmin_hwang@kaist.ac.kr

### Reviewer
...

## Reference & Additional materials
1. Batson, J.D., & Royer, L.A. (2019). Noise2Self: Blind Denoising by Self-Supervision. ArXiv, abs/1901.11365. ([link](https://arxiv.org/abs/1901.11365))
2. Lehtinen, J., Munkberg, J., Hasselgren, J., Laine, S., Karras, T., Aittala, M., & Aila, T. (2018). Noise2noise: Learning image restoration without clean data. arXiv preprint arXiv:1803.04189. ([link](https://arxiv.org/abs/1803.04189))
3. Local averaging ([link](https://swprog.tistory.com/entry/OpenCV-%EC%9E%A1%EC%9D%8Cnoise-%EC%A0%9C%EA%B1%B0%ED%95%98%EA%B8%B0-Local-Averaging-Gaussian-smoothing)) 
4. Noise2Self github ([link](https://github.com/czbiohub/noise2self)) 
5. PSNR ([link](https://ko.wikipedia.org/wiki/%EC%B5%9C%EB%8C%80_%EC%8B%A0%ED%98%B8_%EB%8C%80_%EC%9E%A1%EC%9D%8C%EB%B9%84))  
