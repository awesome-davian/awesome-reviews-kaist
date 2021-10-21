# paper-review-eng

---

Description: Zhao et al. / Pyramid Scene Parsing Network / CVPR 2017 

---

# Pyramid Scene Parsing Network [English]

한국어로 쓰인 리뷰를 읽으려면 **여기**를 누르세요.

**English version** of this article is available.

## 1. Problem definition

Semantic segmentation is to know the category label of each pixels for known objects only. Scene parsing, which is based on semantic segmentation, is to know the category label of ALL pixels within the image. This is how these to tasks differ. Scene parsing provides complete understanding of the scene, where semantic segmentation only provides the category label of *known* objects. From scene parsing, one could further predict location as well as shape of each element. 

Mathematically explained, for input RGB image $I^{\{W\times H\times 3\}}$, the model predicts probability map $P^{\{W\times H\times C\}}$ where $C$ denotes the number of classes to predict. Each pixel values are the probability for each classes, and $I'^{\{W\times H\}}=\argmax(P^{\{W\times H\times C\}})$ can be used to predict the final class for each pixel.

## 2. Motivation

Prior to PSPNet, state-of-the-art scene parsing frameworks are mostly based on the fully convolutional network (FCN). However, FCN still faces challenges considering diverse scenes. 

Let's have a look at Fig. 1.

![Untitled](paper-review-eng%20247bb6c8837b4f8aa0cf7a301134a6cc/Untitled.png)

{Figure 1. Scene parsing issues observed.}

**Mismatched Relationship**  As shown in the first row of Fig. 1, FCN predicts the boat in the yellow box as a car based on its appearance only. This is because of its shape and appearance. But we all know that the car cannot float on a river. Lack of contextual information increases the chance of misclassification. If the network could get information about the context, say water around the object *boat,* it will correctly classify.

**Confusion Categories**  The second row shows confusion case where class building is easily confused as skyscraper. They are with similar appearances. This should be excluded so that the whole object is either skyscraper or building. 

**Inconspicuous Classes**  Scene contains objects/stuff of arbitrary size. Small objects are hard to find while they might be critical to detect, such as traffic light and signboard. On the other hand, big objects are tend to exceed the receptive field of FCN and cause discontinuous prediction. As shown in the third row, the pillow has similar appearance with the sheet. Overlooking the global scene category may fail to parse the pillow.

To summarize above observations, many errors are related to contextual relationship of the scene and global information for different receptive fields. 

### Related work

Please introduce related work of this paper. Here, you need to list up or summarize strength and weakness of each work.

Thanks to **Fully Convolutional Network for Semantic Segmentation[3]**, scene parsing and semantic segmentation achieve great progress inspired by replacing the fully-connected layer in classification. However, the major issue for FCN based models is lack of suitable strategy to utilize global scene category clues as shown in the first row of Fig. 1.

To enlarge the receptive field of neural networks, **Multi-Scale Context Aggregation by Dilated Convolutions[4]** used dilated convolution which helps in increasing the receptive field. This dilated convolution layers are placed in the last two blocks of the backbone of proposed network. Figure 2. show how dilated convolution works differently from convolutions. We can see that the receptive field for dilated convolution is larger as compared to the standard convolution, hence much more context information.

![dilated.gif](paper-review-eng%20247bb6c8837b4f8aa0cf7a301134a6cc/dilated.gif)

![normal_convolution.gif](paper-review-eng%20247bb6c8837b4f8aa0cf7a301134a6cc/normal_convolution.gif)

{Figure 2. Dilated convolution (left), normal convolution (right).}

**Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs[5]** used conditional random field (CRF) as post processing to refine the segmentation result. This improves the localization ability of scene parsing where predicted semantic boundary fits objects. But there is still much room to exploit necessary information in complex scenes.

**ParseNet[6]** proved that global average pooling with FCN improve semantic segmentation results. The idea is to generate one feature map for each corresponding category of the classification task in the last layer, as shown in Fig. 3. However, the experiments in this paper show that these global descriptors are not representative enough for the challenging ADE20K data.

![Untitled](paper-review-eng%20247bb6c8837b4f8aa0cf7a301134a6cc/Untitled%201.png)

{Figure 3. Illustration of global average pooling.}

Spatial pyramid pooling was widely used where spatial statistics provide a good descriptor for overall scene interpretation. **Spatial Pyramid Pooling network[7]** further enhances the ability. 

### Idea

After you introduce related work, please illustrate the main idea of the paper. It would be great if you describe the idea by comparing or analyzing the drawbacks of the previous work.

uses FCN and dilated conv, global pyramid pooling

This paper uses FCN with dilated convolution so that the network has larger receptive field for more context information. This dilated convolution layers are placed in the last two blocks of the backbone. Hence the feature received at the end of the backbone contains richer features. 

Global average pooling is a good baseline model as the global contextual prior. But this strategy is not enough to cover necessary information in complex-scene images, where pixels are annotated regarding many stuff and objects. Directly fusing them to form a single vector may lose the spatial relationship and cause ambiguity. Global context information along with sub-region context is helpful in this regard to distinguish among various categories. To this end, different fro global pooling, this paper exploit the capability of global context information by different-region-based context aggregation via our pyramid scene parsing network.

In **Spatial Pyramid Pooling network[7]**, feature maps in different levels generated by pyramid pooling were finally flattened and concatenated to be fed into a fully connected layer for classification. To further enhance context information between different sub-regions, this paper proposes a hierarchical global prior, containing information with different scales and varying among different sub-regions.

## 3. Method

![Untitled](paper-review-eng%20247bb6c8837b4f8aa0cf7a301134a6cc/Untitled%202.png)

{Figure. 4. Overview of PSPNet.}

Overview of proposed PSPNet is shown in Fig. 4. First, given the input image (a), the network uses CNN to get the feature map of the last convolutional layer (b). Here PSPNet uses a pretrained ResNet model with the dilated network strategy to extract the feature map. The final feature map size is 1/8 of the input image. Then a pyramid parsing module is applied to get different sub-region representations, followed by upsampling and concatenation layers to form the final feature representation, which carries both local and global context information in (c). Finally, the convolution is applied to this feature representation to get the final pixel-wise prediction (d).

### 3.1. Pyramid Pooling Module

The pyramid pooling module fuses features extracted by CNN under four different pyramid scales. The coarsest level highlighted in red is global pooling to generate a single bin output, just like global average pooling. The following level divides the feature map into 4 sub-region and forms 2x2 pooled representation, as highlighted in yellow. The following levels form 3$\times$3, 6$\times$6 pooling respectively. In this paper, pyramid pooling module is one with bin sized of 1$\times$1, 2$\times$2, 3$\times$3 and 6$\times$6. For the type of pooling operation between max and average, experiments in Section 4. show the difference. To maintain the weight of global feature, 1$\times$1 convolution is applied to each pyramid level to reduce the dimension by $1/N$, where $N$ is the level size of pyramid. Then upsample with bilinear interpolation is applied to the low-dimension feature maps to get the same size feature map as the original feature map. Finally, different levels of features are concatenated as the final pyramid pooling global feature.

### 3.2. Network Architecture

Given an input image in Fig. 4(a), pretrained ResNet model with dilated network strategy is used to extract the feature map. The size of extracted feature map is $1/8$ of the input image. Then the pyramid pooling module is added after as described in Section 3.1. to gather global context information. Using 4-level pyramid, the pooling kernels can cover the whole, half of, and some parts of the scene image. Then these global prior kernels are concatenated with the original feature map to form the whole feature with global context information. Finally a convolution layer is applied to generate the final prediction map.

### 3.3. Deep Supervision for ResNet-Based FCN

![Untitled](paper-review-eng%20247bb6c8837b4f8aa0cf7a301134a6cc/Untitled%203.png)

{Figure. 5. Illustration of auxiliary loss in ResNet101.}

Apart from the main branch using softmax loss to train the final classifier, another classifier is applied after the fourth stage. An example of this deeply supervised ResNet101 model is illustrated in Fig. 5. This auxiliary loss helps optimize the learning process, while the master branch loss takes the most responsibility.

## 4. Experiment & Result

### Experimental Setup

### 4.1. Implementation Details

Inspired by **DeepLab[8]** PSPNet uses the "poly" learning rate policy, described as $lr=base\_lr(1-{iter\over maxiter})^{power}$. Here, base learning rate is set to $0.01$ and $power$ to $0.9$. Momentum and weight decay are set to $0.9$ and $0.0001$ respectively. For data augmentation, random mirror and random resize between 0.5 and 2, random rotation between $-10$ and $10$ degrees, and random Gaussian blur for ImageNet and PASCAL VOC are applied. PSPNet contains dilated convolution following **DeepLab[8]**. Batchsize of $16$ is used and for the auxiliary loss, the weight is set to $0.4$ in experiments.

### 4.2. ImageNet Scene Parsing Challenge 2016

The ADE20K dataset is used in ImageNet scene parsing challenge 2016. ADE20K is challenging for the up to 150 classes and diverse scenes with a total of 1,038 image-level labels. For evaluation, both pixel-wise accuracy (Pixel Acc.) and mean of class-wise intersection over union (Mean IoU) are used.

{Table 1. Baseline is ResNet50-based FCN with dilated network. 'B1' and 'B1236' denote pooled feature maps of bin sizes $\{1\times 1\}$ and $\{1\times 1,2\times 2,3\times 3,6\times6\}$ respectively. MAX represent max pooling, AVE average pooling, and DR dimension reduction with $1\times 1$ convolution after pooling. The results are tested on the validation set with the single-scale input. }

### Result

**Ablation Study for Pooling**  To evaluate PSPNet, the author conduct experiments with several settings, including pooling types of max and average, pooling with just one global feature or four-level features, with and without dimension reduction after the polling operation and before concatenation. As listed in Table 1, in terms of pooling, average works better. Pooling with four-level features outperforms that with global feature. The best setting is four-level pyramid of average pooling, followed by dimension reduction with $1\times1$ convolution. 

{Table 2. Performance with various auxiliary loss weight.}

**Ablation Study for Auxiliary Loss**  The auxiliary loss helps optimize the learning process while not influencing learning in the master branch. Table 2. shows experiment result with different settings of auxiliary loss weight $\alpha$ and $\alpha=0.4$ yields the best performance. 

![Untitled](paper-review-eng%20247bb6c8837b4f8aa0cf7a301134a6cc/Untitled%204.png)

{Figure. 6. Performance grows with deeper networks.}

{Table 3. Deeper pre-trained model gets higher performance.}

**Ablation Study for Pre-trained Model**  To further analyze PSPNet, experiments for different depths of pre-trained ResNet have been conducted. As shown in Fig. 6, deeper pre-trained model get higher performance. The multi-scale testing helps to improve the results as well, listed in Table 3. 

{Table 4. 'DA' refers to data augmentation, 'AL' denotes the auxiliary loss, }

More Detailed Performance Analysis  Table 4. shows more detailed analysis on the validation set of ADE20K. The baseline is adapted from ResNet50 with dilated network. "ResNet269+DA+AL+PSP+MS" achieves highest performance among them.

{Table 5. Per-class results on PASCAL VOC 2012 testing set. Methods pre-trained on MS-COCO are marked with '$\dag$' }

Table 5. shows results compared to previous best-performing methods on the PASCAL VOC 2012 dataset, with or without pre-training on MS-COCO dataset. PSPNet outperforms prior methods on both settings, especially getting the highest accuracy on all 20 classes without pre-trained. Several examples are shown in Fig. 7. The baseline model treats "cows" in first row as "horse" and "dog" while PSPNet corrects these errors. For "aeroplane" and "table" in the second and third row, PSPNet finds missing parts. For "person", "bottle" and "plant" in following rows, PSPNet performs well on these small-size object classes in the images compared to the baseline model.

![Untitled](paper-review-eng%20247bb6c8837b4f8aa0cf7a301134a6cc/Untitled%205.png)

{Figure. 7. Visual improvements on PASCAL VOC 2012 data.}

{Table 6. Results on Cityscapes testing sets.}

Cityscapes is dataset for semantic urban scene understanding with 19 categories. Also, 20,000 coarsely annotated images are provided for two settings in comparison, that is training with only fine data or with both the fine and coarse data. Methods trained using both fine and coarse data are marked with '$\ddag$'. Here the base model is ResNet101 as in DeepLab[4] for fair comparison. Table 6. show that PSPNet outperforms other methods with significant advantage. Several examples are shown in Fig. 8.

![Untitled](paper-review-eng%20247bb6c8837b4f8aa0cf7a301134a6cc/Untitled%206.png)

{Figure 8. Examples of PSPNet results on Cityscapes dataset.}

## 5. Conclusion

This paper's main contributions are threefold:

- Proposes a pyramid scene parsing network to embed difficult scenery context features in an FCN based pixel prediction framework.
- Develops an effective optimization strategy for deep ResNet based on deeply supervised loss (auxiliary loss).
- Builds a practival system for state-of-the-art scene parsing and semantic segmentation with all crucial implementation details included.

Several experiments on different datasets show that PSPNet achieves the state-of-the-art performance compared to contemporary methods. However, PSPNet itself is just an encoder, which means it is just half of what is required for image segmentation. Future work could include working on decoder that is suitable for pyramid pooling module.

### Take home message (오늘의 교훈)

> This paper emphasizes the importance of global scene context for image segmentation.
> 

## Author / Reviewer information

### Author

**신주엽 (Juyeb Shin)**

- KAIST
- Research interest in computer vision, especially semantic visual SLAM
- [juyebshin.notion.site](http://juyebshin.notion.site)

### Reviewer

1. Korean name (English name): Affiliation / Contact information
2. Korean name (English name): Affiliation / Contact information
3. …

## Reference & Additional materials

1. Citation of this paper
2. Official (unofficial) GitHub repository
3. J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015.
4. F. Yu and V. Koltun. Multi-scale context aggregation by dilated convolutions. arXiv:1511.07122, 2015.
5. L. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille. Semantic image segmentation with deep convolutional nets and fully connected crfs. arXiv:1412.7062, 2014.
6. W. Liu, A. Rabinovich, and A. C. Berg. Parsenet: Looking wider to see better. arXiv:1506.04579, 2015.
7. J. Dai, K. He, and J. Sun. Boxsup: Exploiting bounding boxes to supervise convolutional networks for semantic segmentation. In ICCV, 2015.
8. L. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. arXiv:1606.00915, 2016