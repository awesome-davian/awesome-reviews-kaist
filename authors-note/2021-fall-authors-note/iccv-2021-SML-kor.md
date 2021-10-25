---
Description: Jung et al. / Standardized Max Logits - A Simple yet Effective Approach for Identifying Unexpected Road Obstacles in Urban-Scene Segmentation / ICCV 2021 Oral
---

# Standardized Max Logits \[Kor\]

[**English version**](iccv-2021-SML-eng.md) of this article is available.



안녕하세요, 이 포스팅에서 소개드리고자 하는 논문은 이번 ICCV 2021에 Oral presentation으로 등재된 논문인 Standardized Max Logits (SML)에 대해 설명드리고자 합니다. 해당 논문에는 저와 이정수 석사과정생이 공동 1저자로 참여하였으며 도로 주행 semantic segmentation에서의 Out-of-Distribution 탐지 문제를 해결하고자 한 논문입니다. 저희의 방법론은 Fishyscapes라는 public leaderboard에서 state-of-the-art 성능을 보였습니다 ([Fishyscapes](https://fishyscapes.com/results)).

##  1. Problem definition

![Teaser Image](../../.gitbook/assets/50/Introduction.png)

최근 도로 주행 semantic segmentation의 발전은 다양한 benchmarking dataset에서 큰 성과를 이루었습니다. 하지만 이런 노력에도 불구하고 여전히 이러한 모델들은 실제 주행 환경에 적용되기 힘듭니다. 그 이유는 모델의 학습 시에 저희가 가정한 몇 개의 pre-define된 class만을 이용해서 학습하게 되고, 이렇게 학습한 모델은 input image의 모든 픽셀을 pre-define된 class중 하나로 예측하게 됩니다. 따라서, 실제 주행 시에 pre-define된 class가 아닌 unexpected obstacle이 등장하면 위 그림에서 보이다시피 제대로 대처할 수 없게 됩니다. 따라서, anomalous한 영역은 찾아내는 것이 안전이 중요한 application인 도로 주행에서 큰 문제이며 저희의 방법론은 이러한 영역을 따로 다룰 수 있게 도와주는 시발점 역할을 해줍니다.

자세한 설명에 들어가기 앞서, semantic segmentation task의 정의에 대해 설명해보도록 하겠습니다. 주어진 input image $x\in{\mathbb{X}_{train}}^{H\times{W}}$와 그 픽셀별로의 정답을 가지고 있는  $y\in{\mathbb{Y}_{train}}^{H\times{W}}$ 에 대하여 우리는 $x$에 대한 예측 값 $\hat{y}$를 내뱉는 segmentation model $G$를 cross-entropy loss를 사용하여 학습하게 됩니다.
$$
CrossEntropy = -\sum\limits_{x\in\mathbb{X}}{y\log{\hat{y}}},
$$
여기에서도 알 수 있다시피, 모델 $G$는 anomalous한 영역에 대해서도 pre-defined class로 예측하게 됩니다. 이러한 점을 해결하고자 저희의 논문에서는 각 픽셀에 대해 anomaly score를 예측하는 간단하고 효과적인 방법론을 제시하며 다른 방법론들과 달리 추가적인 training이나 다른 network module을 사용하지 않습니다.

## 2. Preliminary

OoD 탐지를 위해 다양한 이전 연구들이 있어왔습니다. 그 중, 저희가 주목한 방법론들은 Maximum Softmax Probability (MSP) [1]과 Max Logit [2] 입니다. 먼저 MSP [1]는 network prediction에 softmax를 취한 값을 anomaly score로 사용하는 것을 제안한 seminal 연구입니다. 하지만 MSP score의 경우,  exponential함수가 빠르게 증가하는 성질을 가지고 있기 때문에 anomaly image들이 높은 MSP score를 갖는 (낮은 anomaly score) 문제가 있었습니다. 이러한 문제를 해결하기 위해 제안된 방법론이 Max Logit [2] 입니다. Softmax에 들어가기 전의 logit 값을 anomaly score로 사용하는 방법을 제안하였으며 exponential function을 사용하지 않기 때문에 MSP에서의 over-confident 문제를 해결할 수 있었습니다. 저희의 연구에서는 이 Max Logit이 semantic segmentation에서 가질 수 있는 문제를 이야기하고 이를 해결하기 위한 방법을 제시합니다.

### Other related work

Semantic segmentation의 OoD 탐지 문제를 해결하기 위해, 다양한 연구들 [3, 4, 5, 6, 7, 8]이 제안되었습니다. 몇몇의 연구 [3, 4]들은 PASCAL VOC에서 pre-defined class에 해당하지 않는 object들을 찾아서 training dataset인 Cityscapes에 합성하여 segmentation model을 학습시켰고 다른 종류의 연구 [5, 6, 7, 8]들은 image resynthesis 방법을 사용하였습니다. 이 방법론들은 image resynthesis 모델이 unseen object는 맞게 생성해내지 못한다는 직관에서 시작되었습니다. 하지만 이 두 방법론 모두 추가적인 OoD dataset을 필요로 하거나 또는 추가적인 학습이 필요하였습니다. 

## 3. Motivation

![Motivation](../../.gitbook/assets/50/Motivation.png)

### Findings from previous work

Above figure explains the motivation of our work. We plotted the prediction of a pre-trained segmentation network on Fishyscapes Lost&Found OoD dataset. Each bar denotes the distribution of pixels, and red denotes in-distribution (pre-defined classes), and blue indicates unexpected pixels. Gray area stands for the overlapped region between  minimum score of in-distribution pixels and maximum score of unexpected pixels (i.e., amount of false positives and false negatives). As aforementioned in the Preliminary section, MSP suffers from over-confident problem. In the Max Logit case,  the distribution varies a lot for each class. This is problematic since we utilize a single threshold for all the classes to identify anomalous pixels. 

### Idea

From this finding, we propose Standardized Max Logit (SML) score, which is class-wisely standardized score of Max Logit. As shown in the figure, one can notice  that the overlapped region significantly decreased compared to the previous methods. In addition to this SML, we further propose subsequent techniques to improve the anomaly detection performance by suppressing boundary regions and removing small irregulars.

## 3. Method

![Method](../../.gitbook/assets/50/Method.png)

Our proposed method is illustrated in the above figure. We first infer an image and obtain the Max Logit scores. Afterwards, we standardizes the Max Logit values with the statistics obtained from training samples and iteratively replace the values in the boundary with those of surrounding non-boundary pixels. Lastly, we smooth out the whole image so that small irregular values can be removed. 

The following describes how we obtain the Max Logit and prediction from a pre-trained segmentation network. Let $X \in \mathbb{R}^{3\times{H}\times{W}}$ and $C$ denote an input image and the number of pre-defined classes. The logit output $F\in\mathbb{R}^{C\times{H}\times{W}}$ can be obtained from the network output before the softmax layer. Then, the max logit $L\in\mathbb{R}^{H\times{W}}$ and prediction $\hat{Y}\in\mathbb{R}^{H\times{W}}$ at each location $h,w$ are defined as 
$$
\boldsymbol{L}_{h,w} = \max_\limits{c}\boldsymbol{F}_{c,h,w}\\
\boldsymbol{\hat{Y}}_{h,w} = \arg{\max}_\limits{c}\boldsymbol{F}_{c,h,w},
$$
where $c\in\{1, ..., C\}$.

### 3-1. Standardized Max Logits (SML)

For the standardization, we should first obtain the statistics of training samples for each class. To be specific, we calculate the mean and variance of each class in training samples. This process can be defined as following 
$$
\mu_c = \frac{\sum_i\sum_{h,w}\mathbb{1}(\boldsymbol{\hat{Y}}^{(i)}_{h,w} = c)\cdot{\boldsymbol{L}^{(i)}_{h,w}}}{\sum_i\sum_{h,w}\mathbb{1}(\boldsymbol{\hat{Y}}^{(i)}_{h,w} = c)}\\

\sigma_c = \frac{\sum_i\sum_{h,w}\mathbb{1}(\boldsymbol{\hat{Y}}^{(i)}_{h,w} = c)\cdot{(\boldsymbol{L}^{(i)}_{h,w} - \mu_c)^2}}{\sum_i\sum_{h,w}\mathbb{1}(\boldsymbol{\hat{Y}}^{(i)}_{h,w}=c)}
$$
where $i$ indicates the $i$-th training sample and $\mathbb{1}(\cdot)$ represents the indicator function.

With these obtained training statistics, we calculate SML $\boldsymbol{S}\in\mathbb{R}^{H\times{W}}$ by standardizing the Max logit values of test images as 
$$
\boldsymbol{S}_{h,w}=\frac{\boldsymbol{L_{h,w}}-\mu_{\boldsymbol{\hat{Y}_{h,w}}}}{\sigma_{\hat{Y}_{h,w}}}.
$$
Utilizing SML converts the previous Max Logit values to have the same meaning, which are relative scores in their class. This mapping to the same semantic space enables us to apply subsequent techniques such as Iterative Boundary Suppression and Dilated Smoothing.

### 3-2. Iterative Boundary Suppression

![Boundary Suppression](../../.gitbook/assets/50/BoundarySuppression.png)

Boundary regions tend to be more uncertain than inner regions of classes because those regions where the transition from one class to another occurs. Therefore, we come up with Iterative Boundary Suppression that propagates more certain values of surrounding pixels into the boundary regions. This process is illustrated in the above figure. We first obtain the boundary mask from the initial prediction. Then, apply Boundary Aware Pooling on those regions so that the surrounding values can replace the boundary values. We iteratively apply this process by reducing the boundary width. 

To be more specific, we set the initial boundary width to $r_0$ and reduce it for each iteration by $\Delta{r}$. With a given boundary width $r_i$ at the $i$-th iteration and the prediction $\hat{Y}$, we obtain the non-boundary mask $\boldsymbol{M}^{(i)}\in\mathbb{R}^{H\times{W}}$ at each pixel $h, w$ as 
$$
\boldsymbol{M}^{(i)} = \begin{cases}
0, & \text{if $^\exists{h^\prime, w^\prime}\ \  \text{\textit{s.t.,}}\  \boldsymbol{\hat{Y}}_{h, w} \neq \boldsymbol{\hat{Y}}_{h^\prime, w^\prime}$} \\
1, & \text{otherwise}
\end{cases}\quad,
$$
for $^\forall{h^\prime, w^\prime}$ that satisfies $|h - h^\prime| + |w - w^\prime| \leq r_i$. 

Then, we apply BAP with this mask $\boldsymbol{M}^{(i)}$, which is defined as 
$$
BAP(\boldsymbol{S}^{(i)}_\mathcal{R}, \boldsymbol{M}^{(i)}_{\mathcal{R}}) = \frac{\sum_{h,w}{\boldsymbol{S}^{(i)}_{h,w} \times \boldsymbol{M}^{(i)}_{h,w}}}{\sum_{h,w}{\boldsymbol{M}^{(i)}_{h,w}}},
$$
where $\boldsymbol{S}^{(i)}_\mathcal{R}$ and $\boldsymbol{M}^{(i)}_\mathcal{R}$ denote the patch of receptive field $\mathcal{R}$ on $\boldsymbol{S}^{(i)}$ and $\boldsymbol{M}^{(i)}$, and $(h, w) \in \mathcal{R}$ enumerates the pixels in $\mathcal{R}$. We iteratively apply this process for $n$ times, so that we can completely replace the boundary regions with confident values. We set the initial boundary width $r_0$ as 8, reduce rate $\Delta{r}$ as 2, the number of iteration $n$ as 4, and the size of receptive field $\mathcal{R}$ as $3\times3$.  By employing this process, we are able to successfully remove the false positives and false negatives in boundary regions.

### 3-3. Dilated Smoothing

Iterative Boundary Suppression alone, however, cannot remove small false positives and false negatives not reside in the boundary regions. Gaussian smoothing is well-known for its effectiveness in removing small noises.  Therefore, we smooth out the boundary suppressed values with Gaussian smoothing. In addition, to reflect wider regions while smoothing, we broaden the receptive field with dilation. 

## 4. Experiment & Result

### Experimental setup

We measured area under receiver operating characteristics (AUROC) and average precision (AP). In addition, we measured false positive rate at true positive rate of 95% (FPR$_{95}$). For the qualitative analysis, we used the threshold at true positive rate at 95% (TPR$_{95}$). 

We validated our approach on the following datasets

* Fishyscapes Lost & Found [9] - Real driving scene dataset containing 37 types of small unexpected obstacles such as boxes.
* Fishyscapes Static [9] - Composited images that unexpected obstacles are overlaid on Cityscapes validation images.
* Road Anomaly [5] - Web collected anomaly dataset containing unusual dangers which vehicles confront on roads.

### Implementation Details

We adopted DeepLabv3+ [10] as our segmentation architecture and utilized ResNet101[11] as our backbone. We set the output stride as 8, batch size as 8, initial running rate as 1e-2, and momentum of 0.9. We pre-trained our model on Cityscapes dataset for 60K iterations using polynomial learning rate scheduling with the power of 0.9 and using auxiliary loss proposed in PSPNet [12] with loss weight $\lambda$ of 0.4. For data augmentation, we applied color and positional augmentations such as color jittering, Gaussian blur, random horizontal flip, and random cropping. We also applied class-uniform sampling [13, 14] with a rate of 0.5. 

For boundary suppression, we calculated the boundary mask by subtracting the eroded prediction map from the dilated prediction map. For the dilation and erosion, we utilized L1 filters. We set the number of boundary iterations $n$, the initial boundary width $r_0$, and the dilation rate $d$ as 4, 8, and 6, respectively. Additionally, we set the sizes of the boundary-aware average pooling kernel and the smoothing kernel size as $3\times{3}$ and $7\times7$, respectively.

We used the negative of the final SML values as our anomaly scores. Official implementation can be found at https://github.com/shjung13/Standardized-max-logits

### Qualitative Result

![LostandFound](../../.gitbook/assets/50/Qualitative2.png)

![](../../.gitbook/assets/50/Qualitative3.png)

Above figures illustrates qualitative results of MSP, Max Logit, and our method on Fishyscapes Lost&Found and Static, respectively. White pixels indicate the pixels predicted as unexpected. As shown, ours successfully removed lots of false positives and false negatives compared to MSP and Max Logits.

![Analysis](../../.gitbook/assets/50/Qualitative.png)

Above figure illustrates the qualitative results of applying SML, iterative boundary suppression, and dilated smoothing, respectively on Fishyscapes Lost & Found images. On the yellow boxes, we can see iterative boundary suppression successfully remove the false positive pixels in boundary regions. Moreover, in the green boxes, one can check the small false positives disappear after applying dilated smoothing.



### Quantitative Results

We first show the results on the leaderboard and then show the validation results on various datasets.

![Leaderboard](../../.gitbook/assets/50/Leaderboard.png)

Above table is the leaderboard results in Fishyscapes Lost&Found test set and Static test set. As can be shown, our method achieved state-of-the-art performance on Fishyscapes Lost&Found among the ones did not require additional training and OoD data.

![Validation](../../.gitbook/assets/50/validation.png)

This is the validation results on Fishyscapes Lost&Found and Static validation set and Road Anomaly. Our method outperforms other baselines with a large gap. Note that our method does not require additional training or OoD dataset for training.

## 5. Conclusion

Our work proposed a simple yet effective approach to identify unexpected obstacles on urban scenes. Ours require minimal overhead on inference time and memory usage. Moreover, our method can combined with other techniques that requires additional training or OoD dataset, since we proposes post-processing techniques. However, there still remains room for improvement. First, our method depends on how the segmentation network trained since our approach depend on Max Logit distributions. In addition, after dilated smoothing, small OoD obstacles can be remove as well along with false positives. Those are the remaining further work to do in the future.

### Take home message 

> Aligning class-wise distributions of Max Logit is helpful for OoD detection
>
> Post-processing method can be an effective solution since we can use it for any segmentation networks, off-the-shelf
>
> In semantic segmentation or OoD detection in semantic segmentation, boundary regions can be uncertain, and addressing such regions properly can be critical for some cases.

## Author / Reviewer information

### Author

**정상헌 \(Sanghun Jung\)** 

* Sanghun Jung \(KAIST AI)
* Personal page: https://shjung13.github.io
* Github: https://github.com/shjung13
* LinkedIn: https://www.linkedin.com/in/sanghun-jung-b17a4b1b8/

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. Sanghun Jung, Jungsoo Lee, Daehoon Gwak, Sungha Choi, and Jaegul Choo. Standardized Max Logits: A Simple yet Effective Approach for Identifying Unexpected Road Obstacles in Urban-Scene Segmentation. In Proc. of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 15425-15434, 2021.
2. Github: https://github.com/shjung13/Standardized-max-logits
3. Citation of related work
   1. [1] Dan Hendrycks and Kevin Gimpel. A baseline for detecting misclassified and out-of-distribution examples in neural networks. In Proc. of the International Conference on Learning Representations (ICLR), 2017.
   2. [2] Dan Hendrycks, Steven Basart, Mantas Mazeika, Mohammadreza Mostajabi, Jacob Steinhardt, and Dawn Song. Scaling out-of-distribution detection for real-world settings. arXiv preprint arXiv:1911.11132, 2020.
   3. [3] Petra Bevandic, Ivan Kre ´ so, Marin Or ˇ siˇ c, and Sini ´ saˇ Segvi ˇ c.´ Dense outlier detection and open-set recognition based on training with noisy negative images. arXiv preprint arXiv:2101.09193, 2021.
   4. [4] Robin Chan, Matthias Rottmann, and Hanno Gottschalk. Entropy maximization and meta classification for out-ofdistribution detection in semantic segmentation. arXiv preprint arXiv:2012.06575, 2020.
   5. [5] Krzysztof Lis, Krishna Nakka, Pascal Fua, and Mathieu Salzmann. Detecting the unexpected via image resynthesis. In Proc. of IEEE international conference on computer vision (ICCV), pages 2151–2161, 2019.
   6. [6] Krzysztof Lis, Sina Honari, Pascal Fua, and Mathieu Salzmann. Detecting road obstacles by erasing them. arXiv preprint arXiv:2012.13633, 2020.
   7. [7] Yingda Xia, Yi Zhang, Fengze Liu, Wei Shen, and Alan L. Yuille. Synthesize then compare: Detecting failures and anomalies for semantic segmentation. In Proc. of the European Conference on Computer Vision (ECCV), pages 145– 161, 2020.
   8. [8] Toshiaki Ohgushi, Kenji Horiguchi, and Masao Yamanaka. Road obstacle detection method based on an autoencoder with semantic segmentation. In Proc. of the Asian Conference on Computer Vision (ACCV), pages 223–238, 2020.
   9. [9] Hermann Blum, Paul-Edouard Sarlin, Juan Nieto, Roland Siegwart, and Cesar Cadena. The fishyscapes benchmark: Measuring blind spots in semantic segmentation. arXiv preprint arXiv:1904.03215, 2019.
   10. [10] Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. Encoder-decoder with atrous separable convolution for semantic image segmentation. In Proc. of the European Conference on Computer Vision (ECCV), pages 801–818, 2018.
   11. [11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proc. of IEEE conference on computer vision and pattern recognition (CVPR), pages 770–778, 2016.
   12. [12] Hanchao Li, Pengfei Xiong, Jie An, and Lingxue Wang. Pyramid attention network for semantic segmentation. In Proc. of the British Machine Vision Conference (BMVC), page 285, 2018.
   13. [13] Samuel Rota Bulo, Lorenzo Porzi, and Peter Kontschieder. In-place activated batchnorm for memory-optimized training of dnns. In Proc. of IEEE conference on computer vision and pattern recognition (CVPR), pages 5639–5647, 2018.
   14. [14] Yi Zhu, Karan Sapra, Fitsum A Reda, Kevin J Shih, Shawn Newsam, Andrew Tao, and Bryan Catanzaro. Improving semantic segmentation via video propagation and label relaxation. In Proc. of IEEE conference on computer vision and pattern recognition (CVPR), pages 8856–8865, 2019.
4. Other useful materials
5. ...

