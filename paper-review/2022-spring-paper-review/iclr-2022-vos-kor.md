---
description: Du et al. / VOS: Learning What You Don’t Know by Virtual Outlier Synthesis / ICLR 2022 Poster
---

# VOS: OOD detection by Virtual Outlier Synthesis \[Kor\]

##  1. Problem definition
최근 딥러닝이 발전하여 많은 Computer vision task에서 뛰어난 성능을 달성하고 있으나, Out-of-Distribution(OOD) 데이터에 대해서는 아직 높은 Confidence prediction을 내리는 등의 문제점이 존재한다. OOD detection을 위한 방법 중 하나는 충분한 Unknown 데이터를 모델에게 학습시켜 In-distribution(ID)과 Out-of-distribution(OOD)을 구분하는 타이트한 decision boundary를 생성하도록 하는 것이다. 논문의 저자는 이를 위해 가상의 Outlier 데이터를 합성하는 새로운 방법을 제시한다.

또한, 기존의 OOD detection 방법들은 주로 image 전체에 대해서 OOD를 판별했으나, 실제로는 이미지에 여러개의 object가 혼재되어 있으며(e.g. Object Detection) 그 중에서 어떤 region이 anomalous한지를 판단하는 것은 매우 중요하다. 따라서 논문의 저자는 image가 아닌 object level에서 OOD를 식별하는 것을 목표로 한다.

**Problem Setup**:  input과 label space는 각각 다음과 같다. $\mathcal{X} = \mathbb{R}^d, \mathcal{Y}={1,2,...,K}$. 이때, $x\in\mathcal{X}$는 input image, $b\in\mathbb{R}^4$는 object instance의 bounding box, $y\in\mathcal{Y}$는 K-way classification에서 object의 semantic label이다. 우리의 Object detection 모델은 unknown joint distribution인 $\mathcal{P}$에서 뽑힌 in-distribution data $D={(x_i, b_i, y_i)}_{i=1}^{N}$로 부터 학습된다. 모델은 bounding box regression $p_\theta(b|x,y)$과 classification $p_\theta(y|x)$를 수행하기 위한 모델 파라미터 $\theta$를 학습한다.

OOD detection은 ID와 OOD object를 구분하는 binary classification problem으로 볼 수 있다. $P_{\mathcal{X}}$를 $\mathcal{X}$에 대한 marginal probability distribution이라고 하자. test input $x^*\sim P_{\mathcal{X}}$과 object detector가 예측한 $b^*$가 주어졌을 때, OOD를 위한 목표는 $p_\theta(g|x^*, b^*)$를 예측하는 것이다. 이때, $g=1$는 object가 ID임을 의미하고, $g=0$는 OOD를 의미한다.


## 2. Motivation

### Related work
1. **OOD detection for classification**
	크게 다음의 2가지의 방법으로 구분할 수 있다: Post hoc & regularization-based method. Post hoc 방법으로는 OOD input에 대해서 높은 softmax confidence score를 예측하도록 하는 방법이 주로 baseline(Hendrycks & Gimpel, 2017.;  Hein et al., 2019)으로 사용된다. 이를 발전시켜 ODIN (Liang et al., 2018), Mahalanobis distance (Lee et al., 2018b), energy score (Liu et al., 2020a), Gram matrices based score (Sastry & Oore, 2020), and GradNorm score (Huang et al., 2021)의 방법들이 제시되었다. 또 다른 방법인 regularization-based method에서는 natural outlier image(Hendrycks et al., 2019; Mohseni et al., 2020; Zhang et al., 2021)나 GAN 등을 통한 합성 이미지를 활용(Lee et al., 2018)하여 모델을 regularization한다. 하지만 real outlier data를 얻는 것은 매우 힘들다는 한계점이 있다. 
	이러한 방법들은 image 단위의 OOD detection에서는 좋은 성능을 달성하였으나, object detection과 같이 하나의 image 내에 여러개의 object instane가 존재하는 object level OOD detection에서는 성능이 검증되지 않았다.
	
2. **OOD detection for object detection**
	아직까지 object detection을 위한 OOD detection 연구는 활발히 진행되지 않았다. Joseph et al. (2021)은 energy score를 활용하여 OOD object를 분별하고 이를 활용하여 incremental object detection을 수행하였다. 특히, negative proposals를 unknown sample로 활용하여 model을 regularization 하였는데, 이는 ID와 OOD data를 동일한 data distribution에서 뽑게 하여 optimal한 decision boundary를 학습하지 못한다. 몇가지 다른 논문에서는 OOD를 판별하는 것이 아닌 uncertainty를 estimation하는 데에 집중하였고(Harakeh & Waslander, 2021; Riedlinger et al., 2021), 또 다른 연구로는 Bayesian method를 활용하여 OOD detection을 수행하기도 하였으나 multiple inference pass가 불가피하다는 한계점이 있다.(Dhamija et al., 2020; Miller et al., 2019; 2018; Hall et al., 2020; Deepshikha et al., 2021)
	

### Idea
1. image level에서 OOD detection을 수행하던 기존의 방법들과는 달리, image 내의 object level에서 OOD detection을 수행할 수 있는 새로운 framework를 제시한다.(OOD detection for Object detection task)
2. high-dimentional pixel space에서 outlier 데이터를 합성하던 기존의 방법(ex) GAN)과는 달리, feature space에서 outlier를 합성하는 것이 더 좋은 성능을 달성한다는 것을 보인다.
3. 새로운 unknwon-aware training objective를 제시함으로써 ID와 합성된 outlier(OOD) 간의 uncertainty surface를 contrastively shape함

## 3. Method
다음의 3가지 research question에 대한 답을 도출한다.
1. 어떻게 virtual outlier를 합성할 것인가?
2. 어떻게 합성된 outlier를 활용하여 효과적인 model regularization을 수행할 것인가?
3. Inference 때 어떻게 OOD detection을 수행할 것인가?

논문에서 제시하는 framework인 VOS의 전체적인 그림은 아래와 같다.
![Figure ?: The framwork of VOS](.gitbook/assets/2022spring/40/framework.png)

### 3.1. VOS: Virtual Outlier Synthesis
Key idea는 high-dimentional pixel space에서 image를 합성하는 것은 optimize하는 것이 어렵기 때문에, feature space에서 virtual ourlier를 합성하자는 것이다. 우선 object instance의 feature representation을 다음과 같이 class-conditional multivariate Gaussian distritution으로 가정한다.
$$p_\theta(h(x,b)|y=k) = \mathcal{N}(\mu_k, \Sigma)$$
그리고 위의 gaussian 분포의 parameter들을 estimate하기 위해 training sample의 empirical class mean과 covariance를 다음과 같이 계산한다. 이때, 계산 효율을 위해 각 class마다 일정 개수의 instance를 queue에 저장해두면서 계산한다.
$$\hat{\mu}_k = \frac{1}{N_k}  \sum_{i:y_i=k}h(x_i, b_i) \\
\hat{\Sigma}=\frac{1}{N}\sum_{k}\sum_{i:y_i=k}(h(x_i, b_i)-\hat{\mu}_k)(h(x_i, b_i)-\hat{\mu}_k)^\top$$
생성된 virtual outlier들은 ID와 OOD 사이의 compact decision boundary를 estimate할 수 있어야 한다. 따라서 추정된 class-conditional distribution의 $\epsilon$-likelihood region에서 샘플링을 진행한다.
$$\mathcal{V}_k = \{  \mathrm{v}_k|\frac{1}{(2\pi)^{m/2} |\hat\Sigma^{1/2}|}  \exp  \left( -\frac{1}{2}(\mathrm{v}_k - \hat\mu_k)^\top  \hat\Sigma^{-1} (\mathrm{v}_k - \hat\mu_k) \right) < \epsilon  \}$$
위 식에서, $\mathrm{v}_k \sim  \mathcal{N}(\hat\mu_k, \hat\Sigma)$는 class k에 대해 sampling된 virtual outliers를 의미한다.
마지막으로 위에서 샘플링된 virtual outlier로 부터 classification output은 다음과 같이 정해진다.
$$ f(\mathrm{v};\theta) = W_{cls}^\top  \mathrm{v}$$
위 식에서 weight는 classification을 수행하기 직전의 last fully connected layer의 weight를 의미한다.

### 3.2. Unknown-Aware Training Objective
여기서 Key idea는 ID data에 대해서는 낮은 OOD score를 예측하고, 합성된 outlier에 대해서는 높은 OOD score를 예측하도록 model을 regularize하는 것이다.
쉬운 이해를 위해 우선 multi-class classification setting에서 uncertainty regularization의 작동 방식은 다음과 같이 설명할 수 있다.

먼저, input data $x$에 대해 $\log p(x)$를 direct하게 추정하는 것은 intractable하므로, log partition function $E(x;\theta) := -\log\Sigma_{k=1}^{K}e^{f_k(x;\theta)}$이 $\log p(x)$와 비례(with some unknown factor)하다는 것을 이용한다. 아래의 식으로부터 비례 관계를 보일 수 있다.

$$p(y|x) = \frac{p(x,y)}{p(x)} = \frac{e^{f_y(x;\theta)}}{\Sigma_{k=1}^{K}e^{f_k(x;\theta)}}$$
이때 negative log partition function은 free energy라고도 불리는데, 이것은 OOD detection을 위한 uncertainty measurement에 매우 효과적임이 증명되었다.(Liu et al., 2020)

따라서, 위에서 도출된 Energy function을 binary sigmoid loss와 합하여 uncertainty loss를 다음과 같이 나타낼 수 있다.

$$ \mathcal{L}_{uncertainty} = \mathbb{E}_{\mathrm{v}\sim\mathcal{V}}\left[ -\log  \frac{1}{1+\exp^{-\theta_u \cdot E(\mathrm{v};\theta) }}  \right] + \mathbb{E}_{\mathrm{x}\sim\mathcal{D}}\left[ -\log  \frac{\exp^{-\theta_u \cdot E(\mathrm{x};\theta) }}{1+\exp^{-\theta_u \cdot E(\mathrm{x};\theta) }}  \right] $$
이때, $\theta_u$는 slope of sigmoid를 modulate하는 learnable parameter이다. 위의 loss를 통해 ID data에 대해서는 high probability를 예측하고, OOD data에 대해서는 low probability를 예측한다.

이제 classification이 아닌 Object detection setting에서는 다음과 같이 Energy를 정의할 수 있다.
$$ E(x,b;\theta) = -\log\Sigma_{k=1}^{K} w_k \cdot e^{f_k((x,b);\theta)} $$
이때,$f_k((x,b);\theta)$는 classification branch로부터 나온 class k에 대한 logit output이며 $w_k$는 object detection dataset의 class imbalane 를 해결하기 위한 learning parameter 이다.

따라서, OOD detection for object detection의 최종 training objective는 다음과 같다.

$$\min_{\theta} \mathbb{E}_{(x,b,y)\sim \mathcal{D}}   [  \mathcal{L}_{cls} + \mathcal{L}_{loc} ] + \beta\cdot \mathcal{L}_{uncertainty}  $$

이때, $\beta$는 uncertainty regularization weight, $\mathcal{L}_{cls}$, $\mathcal{L}_{loc}$는 각각 classification과 bounding box regression loss이다. 




### 3.3. Inference-Time OOD Detection
Inference시에는 OOD detection을 위해 logistic regression uncertainty branch의 ouput을 사용한다. test input $x^*$, predicted bounding box $b^*$가 주어졌을 때, object $(x^*, b^*)$에 대한 OOD uncertainty score는 다음과 같다.

$$ p_\theta(g|x^*, b^*) = \frac{\exp^{-\theta_u \cdot E(x^*,b^*) }}{1+\exp^{-\theta_u \cdot E(x^*,b^*) }} $$

그리고 위의 score로부터 ID와 OOD를 구분하기 위해 다음과 같이 threshold $\gamma$를 활용한다.

$$G(x^*, b^*) = 
\left\{ 
  \begin{array}{ }
    1(ID)               & \quad \textrm{if } p_\theta(g|x^*,b^*) \geq \gamma  \\
    0(OOD)              & \quad \textrm{if } p_\theta(g|x^*,b^*) < \gamma
  \end{array}
\right.$$

threshold $\gamma$는 ID data의 95%가 올바르게 구분될 수 있도록 하는 수치로 정해진다.

## 4. Experiment & Result
### Experimental setup
* Dataset
	* ID training data: 
		* **PASCAL VOC**(Everingham et al., 2010)
		* **Berkeley DeepDrive(BDD-100k2)** (Yu et al., 2020)
	* OOD training data: subset images from the following datasets which do not contain ID category
		* **MS-COCO** (Lin et al., 2014)
		* **OpenImages(validation set)** (Kuznetsova et al., 2020)
* Baselines
	* **Maximum Softmax Probability** (Hendrycks & Gimpel, 2017)
	* **ODIN** (Liang et al., 2018)
	* **Mahalanobis distance** (Lee et al., 2018)
	* **Generalized ODIN** (Hsu et al., 2020)
	* **energy score** (Liu et al., 2020)
	* **CSI** (Tack et al., 2020)
	* **Gram matrices** (Sastry & Oore, 2020)

* Training setup
	* 2가지 backbone architectures: ResNet-50 and RegNetX-4.0GF
	* class-conditional Gaussian 추정을 위해 1,000개의 sample 사용
	* training iterations
		* For PASCAL VOC: 18,000 iters
		* For BDD-100k: 90,000 iters
	* uncertainty regularizer는 training의 2/3 시점부터 적용($\beta = 0.1$)
	* Python 3.8.5 and PyTorch 1.7.0, using NVIDIA GeForce RTX 2080Ti GPUs

* Evaluation metric
	1. **FPR95**: the false positive rate of OOD samples when the true positive rate of ID samples is at 95%
	2. **AUROC**: the area under the receiver operating characteristic curve
	3. **mAP**: mean Average Precision for the object detection performance on the ID task


### Result
![Figure ?: Main results table](.gitbook/assets/2022spring/40/main_results_table.png)

위의 Main results table에서 확인할 수 있듯이, 논문에서 제시하는 VOS는 다른 baseline들보다 OOD detection과 object detection 성능 모두 뛰어나다. 먼저 OOD detection 성능을 나타내는 FPR95와 AUROC metric을 보면 VOS가 다른 방법들보다 월등하게 좋다. 이와 동시에, VOS는 ID에 대한 object detection 성능을 나타내는 mAP를 헤치지 않으면서 좋은 OOD detection 성능을 달성하는 것을 확인할 수 있다.

![Figure ?: Visualization result](.gitbook/assets/2022spring/40/visualization_result.png)

다음으로 visualization result는 위의 그림과 같다. 그림은 OOD image에 대한 object detection 결과이며, top row는 vanilla Faster-RCNN model이고 bottom row는 VOS이다. Blue bounding box는 ID class로 분류된 object이고, green bounding box는 VOS에 의해 OOD로 분류된 object이다. 
그림에서 확인할 수 있듯이, VOS는 OOD object를 잘 detection하여 false positive를 감소시킨다.또한, 3rd column을 보면 false positive object에 대한 confidence score를 낮게 예측함으로써 더 robust한 detection 결과를 생성한다는 것을 알 수 있다.

## 5. Conclusion
논문에서는 OOD detection을 위해 새로운 unknwon-aware training framework(=VOS)를 제시한다. 실제 outlier data를 필요로 했던 기존의 방법과는 다르게 VOS는 training 때 feature space로부터 virtual outlier를 합성한다. 그리고 이것을 활용하여 model이 ID와 OOD object를 구분하는 decision boundary를 잘 학습할 수 있도록 한다. 그 결과 VOS는 ID task 성능은 유지한 채, state-of-the-art OOD detection 성능을 달성하였으며 object detection에서도 잘 작동함이 증명되었다. 

### Take home message \(오늘의 교훈\)
> 기존의 이미지 합성을 위한 GAN, noise injection 등의 방법들과 다르게, 논문에서 제시한 feature space의 low-likelihood region에서 virtual outlier를 샘플링하는 방법이 흥미로웠다. 또한, 아직까지는 object detection에서 OOD detection을 위한 방법이 별로 없었는데 논문에서 powerful한 baseline을 제공한 것 같고, 이를 계기로 OOD detection for Object detection 연구가 활발히 진행될 것으로 예상된다.

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

1. [VOS: Learning What You Don't Know by Virtual Outlier Synthesis](https://openreview.net/forum?id=TW7d65uYu5M) 
2. [Official VOS GitHub repository](https://github.com/deeplearning-wisc/vos)
3. Hendrycks, D., & Gimpel, K. A baseline for detecting misclassified and out-of-distribution examples in neural networks. arXiv preprint arXiv:1610.02136, 2016.
4. Hein, M., Andriushchenko, M., & Bitterwolf, J. Why relu networks yield high-confidence predictions far away from the training data and how to mitigate the problem. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 41-50, 2019.
5. Shiyu Liang, Yixuan Li, and Rayadurgam Srikant. Enhancing the reliability of out-of-distribution image detection in neural networks. In International Conference on Learning Representations, ICLR 2018.
6. Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A simple unified framework for detecting out-of-distribution samples and adversarial attacks. In Advances in Neural Information Processing Systems, pp. 7167–7177, 2018.
7. Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan Li. Energy-based out-of-distribution detection. Advances in Neural Information Processing Systems, 2020.
8. Chandramouli Shama Sastry and Sageev Oore. Detecting out-of-distribution examples with gram matrices. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, volume 119, pp. 8491–8501, 2020.
9. Rui Huang, Andrew Geng, and Yixuan Li. On the importance of gradients for detecting distributional shifts in the wild. In Advances in Neural Information Processing Systems, 2021.
10. Dan Hendrycks, Mantas Mazeika, and Thomas Dietterich. Deep anomaly detection with outlier exposure. In International Conference on Learning Representations, 2019.
11. Sina Mohseni, Mandar Pitale, JBS Yadawa, and Zhangyang Wang. Self-supervised learning for generalizable out-of-distribution detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 5216–5223, 2020.
12. Jingyang Zhang, Nathan Inkawhich, Yiran Chen, and Hai Li. Fine-grained out-of-distribution detection with mixup outlier exposure. CoRR, abs/2106.03917, 2021.
13. Kimin Lee, Honglak Lee, Kibok Lee, and Jinwoo Shin. Training confidence-calibrated classifiers for detecting out-of-distribution samples. In International Conference on Learning Representations, 2018.
14. K. J. Joseph, Salman Khan, Fahad Shahbaz Khan, and Vineeth N. Balasubramanian. Towards open world object detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2021.
15. Ali Harakeh and Steven L. Waslander. Estimating and evaluating regression predictive uncertainty in deep object detectors. In International Conference on Learning Representations, 2021.
16. Tobias Riedlinger, Matthias Rottmann, Marius Schubert, and Hanno Gottschalk. Gradient-based quantification of epistemic uncertainty for deep object detectors. CoRR, abs/2107.04517, 2021.
17. Akshay Raj Dhamija, Manuel Gunther, Jonathan Ventura, and Terrance E. Boult. The overlooked ¨ elephant of object detection: Open set. In IEEE Winter Conference on Applications of Computer Vision, WACV 2020, pp. 1010–1019, 2020.
18. Dimity Miller, Feras Dayoub, Michael Milford, and Niko Sunderhauf. Evaluating merging strategies ¨ for sampling-based uncertainty techniques in object detection. In International Conference on Robotics and Automation, ICRA 2019, pp. 2348–2354, 2019.
19. Dimity Miller, Lachlan Nicholson, Feras Dayoub, and Niko Sunderhauf. Dropout sampling for ¨ robust object detection in open-set conditions. In IEEE International Conference on Robotics and Automation, ICRA 2018, pp. 1–7, 2018. doi: 10.1109/ICRA.2018.8460700.
20. David Hall, Feras Dayoub, John Skinner, Haoyang Zhang, Dimity Miller, Peter Corke, Gustavo Carneiro, Anelia Angelova, and Niko Sunderhauf. Probabilistic object detection: Definition and ¨ evaluation. In IEEE Winter Conference on Applications of Computer Vision, WACV 2020, pp. 1020–1029, 2020.
21. Kumari Deepshikha, Sai Harsha Yelleni, P. K. Srijith, and C. Krishna Mohan. Monte carlo dropblock for modelling uncertainty in object detection. CoRR, abs/2108.03614, 2021.
22. Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan Li. Energy-based out-of-distribution detection. Advances in Neural Information Processing Systems, 2020.
23. Mark Everingham, Luc Van Gool, Christopher K. I. Williams, John M. Winn, and Andrew Zisserman. The pascal visual object classes (VOC) challenge. International Journal of Computer Vision, 88(2):303–338, 2010.
24. Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, and Trevor Darrell. BDD100K: A diverse driving dataset for heterogeneous multitask learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, pp. 2633–2642, 2020.
25. Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ´ European conference on computer vision, pp. 740–755, 2014.
26. Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4. International Journal of Computer Vision, pp. 1–26, 2020.

