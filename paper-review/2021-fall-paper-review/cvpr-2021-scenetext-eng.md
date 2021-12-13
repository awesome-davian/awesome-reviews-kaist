---
description: Chen et al. / Scene Text Telescope - Text-focused Scene Image Super-Resolution / CVPR 2021
---

#  Scene Text Telescope: Text-focused Scene Image Super-Resolution \[Eng]



한국어로 쓰인 리뷰를 읽으려면 **[여기](https://awesome-davian.gitbook.io/awesome-reviews/paper-review/2021-fall-paper-review/cvpr-2021-scenetext-kor)**를 누르세요.



##  1. Problem definition

> __*Scene Text Recognition (STR) : a task to recognize text in scene images.*__
>
> _(Example Applications : extraction of car license plate, reading ID card, etc)_

![Figure1](/.gitbook/assets/25/Figure1.PNG) 

- Despite the ongoing active researches in STR tasks, recognition on low-resolution (LR) images still has subpar performance. 
- This needs to be solved since LR text images exist in many situations, for example, when a photo is taken with low-focal camera or under circumstances where a document image is compressed to reduces disk usages.

   → To address this problem, this paper proposes a text-focused super-resolution framework, called _Scene Text Telescope_.



## 2. Motivation

### Related work

- __Works on Scene Text Recognition__

  - _Shi, Baoguang, Xiang Bai, and Cong Yao. "An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition." *IEEE transactions on pattern analysis and machine intelligence* 39.11 (2016): 2298-2304._

    : combines CNN and RNN to obtain sequential features of text images and utilizes CTC decoder [1] to maximize the probability of paths that can reach the ground truth

  - _Shi, Baoguang, et al. "Aster: An attentional scene text recognizer with flexible rectification." *IEEE transactions on pattern analysis and machine intelligence* 41.9 (2018): 2035-2048._

    : employs a Spatial Transformer Network to rectify text images and utilizes attention mechanism to focus on specific character at each time step

     → Not suitable for tackling curved texts!

    

- __Works on Text Image Super-Resolution__

  - _Mou, Yongqiang, et al. "Plugnet: Degradation aware scene text recognition supervised by a pluggable super-resolution unit." *Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XV 16*. Springer International Publishing, 2020._

    : considers text-specific properties by designing multi-task framework to recognize and upsample text images

  - _Wang, Wenjia, et al. "Scene text image super-resolution in the wild." *European Conference on Computer Vision*. Springer, Cham, 2020._

    : captures sequential information of text images 

     → Can suffer from disturbances from backgrounds which can degrade the performance of upsampling on text results!

(* Note that these related works and their limitations are mentioned by the paper)



### Idea

> _This paper proposes text-focused super-resolution framework, called **Scene Text Telescope**_

1) To deal with texts in arbitrary orientations,

   → _Utilize a novel backbone, named **TBSRN (Transformer-Based Super-Resolution Network)** to capture sequential information_

2) To solve background disturbance problem,

   → _Put **Position-Aware Module** and **Content-Aware Module** to focus on the position and content of each character_

3) To deal with confusing characters in Low-Resolution, 

   → _Employ a **weighted cross-entropy loss** in Content-Aware Module_  



- __Works that are utilized on model and evaluation__

  - _Luo, Canjie, Lianwen Jin, and Zenghui Sun. "Moran: A multi-object rectified attention network for scene text recognition." *Pattern Recognition* 90 (2019): 109-118._

  - _Shi, Baoguang, Xiang Bai, and Cong Yao. "An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition." *IEEE transactions on pattern analysis and machine intelligence* 39.11 (2016): 2298-2304._

  - _Shi, Baoguang, et al. "Aster: An attentional scene text recognizer with flexible rectification." *IEEE transactions on pattern analysis and machine intelligence* 41.9 (2018): 2035-2048._

  - _Wang, Wenjia, et al. "Scene text image super-resolution in the wild." *European Conference on Computer Vision*. Springer, Cham, 2020._

    



## 3. Method

> _The overall architecture is composed of_...
>
> _**Pixel-Wise Supervision Module + Position-Aware Module + Content-Aware Module**_


![Figure2](/.gitbook/assets/25/Figure2.PNG) 



- **Pixel-Wise Supervision Module**

  1. LR (Low-Resolution) image is rectified by a _STN (Spatial Transformer Network)_ to solve misalignment problem [2].

  2. Then, rectified image goes through _TBSRN (Transformer-based Super-Resolution Networks)_.

     > **TBSRN (Transformer-based Super-Resolution Networks)**
     >
     > ![Figure3](/.gitbook/assets/25/Figure3.PNG) 
     >
     > - _Two CNNs_ : to extract feature map  
     >
     > - _Self-Attention Module_ : to capture sequential information
     >
     > - _2-D Positional Encoding_ : to consider spatial positional information

  3. Finally, the image gets upsampled to SR (Super-Resolution) image through _pixel-shuffling_.

     
     
     +) In this module, ![Eq11](/.gitbook/assets/25/Eq11.gif) where ![Eq12](/.gitbook/assets/25/Eq12.gif)are the image of each resolution
     
      

- **Position-Aware Module**

  1. Pretrain a _Transformer-based recognition model_ using synthetic text datasets (including _Syn90k_ [3] and _SynthText_ [4])

  2. Leverage its attending regions at each time-step as positional clues

     - Given an HR image, Transformer outputs a list of attention maps (![Eq2](/.gitbook/assets/25/Eq2.gif) where ![Eq3](/.gitbook/assets/25/Eq3.gif) = attention map at the i-th time-step & ![Eq4](/.gitbook/assets/25/Eq4.gif) = length of its text label)

     - Generated SR image is also fed into Transformer to obtain ![Eq5](/.gitbook/assets/25/Eq5.gif).

  3.  Employ _L1 loss_ to supervise two attention maps 

     ![Eq6](/.gitbook/assets/25/Eq6.gif) 

     

- **Content-Aware Module**

  1. Train a _VAE (Variational Autoencoder)_ using _EMNIST_ [5] to obtain each character's 2D latent representation
  
     ![Figure4](/.gitbook/assets/25/Figure4.PNG) 
  
     → Positions of similar characters are usually close in the latent space
  
  2. Assume that each at each time-step t, the pre-trained Transformer generates an output vector ![Eq7](/.gitbook/assets/25/Eq7.gif). The content loss ![Eq8](/.gitbook/assets/25/Eq8.gif) for all time-steps is computed as ![Eq9](/.gitbook/assets/25/Eq9.gif) (![Eq10](/.gitbook/assets/25/Eq10.gif)= ground-truth at t-th step)
  
  
  
- __Overall Loss Function__

  ![Eq1](/.gitbook/assets/25/Eq1.PNG) 

     _(Here, lambdas are hyperparameters to balance three terms)_

  

  ****

## 4. Experiment & Result

### Experimental setup

* __Dataset__

  > __TextZoom__ [2] : 17,367 LR-HR pairs for training + 4,373 pairs for testing (1,619 for easy subset / 1,411 for medium / 1,343 for hard)
  >
  > LR images : 16 × 64 / HR images : 32 × 128
  >
  > ![Figure6](/.gitbook/assets/25/Figure6.PNG) 

* __Evaluation metric__

  > __For SR images,__
  >
  > - PSNR (Peak Signal-to-Noist Ratio)
  > - SSIM (Structural Similarity Index Measure)

  > Proposes __metrics that focuses on text regions__
  >
  > - TR-PSNR (Text Region PSNR)
  > - TR-SSIM (Text Region SSIM)
  >
  > → Only take the pixels in the text region (This is done by utilizing _SynthText_ [4], _U-Net_ [6] )

* __Implementation Details__

  > __HyperParameters__
  >
  > - Optimizer : Adam
  >
  > - Batch size : 80
  >
  > - Learning Rate : 0.0001

  > GPU details : NVIDIA TITAN Xp GPUs (12GB × 4)



### Result

- __Ablation Study__

  > - This paper evaluated the effectiveness of each component on backbone, Position-Aware Module, Content-Aware Module, etc. 
  >
  > - Dataset : _TextZoom_ [2]
  >
  >   +) Recognition Accuracy is computed by the pre-trained _CRNN_ [7].
  >
  >   ![Table](/.gitbook/assets/25/Table.PNG) 

- __Results on _TextZoom_ [2]__

  > - Compared the model with other SR models on three recognition models (_CRNN_ [7], _ASTER_ [8], and _MORAN_ [9])
  > - As we can see from the tables below, the recognition accuracy when utilizing _TBSRN_ is relatively   higher than the others.
  >
  > ![Table5](/.gitbook/assets/25/Table5.PNG) 
  >
  > - _Visualized Examples_
  >
  >   ![Figure8](/.gitbook/assets/25/Figure8.PNG) 

- __Failure Cases__

  > ![Figure10](/.gitbook/assets/25/Figure10.PNG) 
  >
  > - Long & Small texts
  > - Complicated background / Occlusion
  > - Artistic fonts / Hand-writting texts
  > - Images whose labels have not appeared in the training set



## 5. Conclusion

- To summarize, this paper

  - Proposed a Text-focused Super Resolution Model (_Scene Text Telescope_)

    - Used _TBSRN_ as a backbone which utilizes self-attention mechanism to handle irregular text images

    - Used weighted cross-entropy loss to handle confusable characters
  

### Take home message

> - Text-focused SR technique can be very effective in handling LR text images than generic SR techniques.
> - Ablation study and Explanation of failure cases can make paper look fancy!



### Author

**박나현 \(Park Na Hyeon\)** 

- _NSS Lab, KAIST EE_
- _julia19@kaist.ac.kr_



### Reviewer

1. Korean name (English name): Affiliation / Contact information
2. Korean name (English name): Affiliation / Contact information
3. ...



## Reference & Additional materials

1. _Graves, Alex, et al. "Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks." *Proceedings of the 23rd international conference on Machine learning*. 2006._
2. _Wang, Wenjia, et al. "Scene text image super-resolution in the wild." *European Conference on Computer Vision*. Springer, Cham, 2020._
3. _Jaderberg, Max, et al. "Reading text in the wild with convolutional neural networks." *International journal of computer vision* 116.1 (2016): 1-20._
4. _Gupta, Ankush, Andrea Vedaldi, and Andrew Zisserman. "Synthetic data for text localisation in natural images." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016._
5. _Cohen, Gregory, et al. "EMNIST: Extending MNIST to handwritten letters." *2017 International Joint Conference on Neural Networks (IJCNN)*. IEEE, 2017._
6. _Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." *International Conference on Medical image computing and computer-assisted intervention*. Springer, Cham, 2015._
7. _Shi, Baoguang, Xiang Bai, and Cong Yao. "An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition." *IEEE transactions on pattern analysis and machine intelligence* 39.11 (2016): 2298-2304._
8. _Shi, Baoguang, et al. "Aster: An attentional scene text recognizer with flexible rectification." *IEEE transactions on pattern analysis and machine intelligence* 41.9 (2018): 2035-2048._
9. _Luo, Canjie, Lianwen Jin, and Zenghui Sun. "Moran: A multi-object rectified attention network for scene text recognition." *Pattern Recognition* 90 (2019): 109-118._

