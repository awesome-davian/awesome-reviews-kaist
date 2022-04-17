---
description: Vikram V. Ramaswamy / Fair Attribute Classification through Latent Space De-biasing / CVPR 2021 Oral
---

# GAN Latent Space De-biasing \[Eng\]

한국어로 쓰인 리뷰를 읽으려면 [여기](./GAN_Latent_Space_Debiasing_Kor.md)를 누르세요.

##  1. Problem definition

Until now, the performance of AI has significantly improved with the invention of various deep learning models. However, deep learning models have potential to give wrong judgements to some groups in dataset because the models are developed focused mainly on overall prediction accuracy. For example, human face recognition models made in Western countries are likely to show poor performance on Asian people. We call this phenomenon the problem of “Fairness in AI.” Even if the performance of AI is improved, the AI models can work adversely to socially/historically vulnerable people (e.g. old/disabled people) when the problem of fairness is not solved, which may cause serious social issues. Therefore, it is crucial to improve the fairness of AIs. Nowadays, many people in AI industry are trying to find methods to raise the fairness of AIs while not sacrificing the performance significantly.

Among many ways to improve fairness, the author of this paper tries Data Augmentation using Generative Adversarial Network (GAN). The augmentation is implemented so that bias toward a specific group is removed by manipulating GAN’s latent space. Even though some similar researches were done in the past, they had disadvantages in algorithmic/computational complexity. In contrast, the author suggests an effective method that uses only a single GAN, which can be used to overcome the previous disadvantages.

## 2. Motivation

### Related work

(1) De-biasing methods

In many cases, the unfairness of deep learning model is derived from the bias in training dataset. To address this, developers either de-bias the training data or modify the training process. In the former case, some methods such as oversampling the vulnerable groups or applying adversarial learning are introduced. In the latter case, methods such as adding a fairness-related regularization term to the model’s loss function are possible. Note that the method used in this paper corresponds to the former case.

(2) Generative Adversarial Network (GAN)

GAN is a network comprised of generator and discriminator, in which they are in a negative relationship. In other words, the generator is trained to deceive the discriminator by generating fake data that resemble the real data, while the discriminator is trained to judge the data from the generator as fake data. After training GAN in this way, it is possible to generate natural-looking fake data. GANs have undergone several improvements until now, and GANs nowadays are able to generate images that are extremely hard to distinguish from the real ones. 

(3) Data augmentation through latent-space manipulation

생성된 이미지를 변형시키기 위해 GAN의 잠재 공간을 조작해 볼 수 있다. 여기서 잠재 공간이란 생성자가 랜덤하게 이미지를 생성하는 데 이용하는 특성들의 공간으로, 잠재 공간에는 이미지의 다양한 속성이 압축되어 있다. 잠재 공간을 잘 조작한다면 이미지에 특정 속성(머리 색, 안경 착용 여부 등)을 부여하거나 이를 조절하는 것이 가능하다. 또한 특정 속성에 대해서만 각기 다른 속성값을 가진 이미지들을 생성함으로써 딥러닝 모델이 해당 속성에 대해 얼마나 불공정성을 지니고 있는 지 측정해 볼 수 있으며, 딥러닝 모델의 불공정성과 가장 크게 연관되어 있는 속성을 찾아낼 수도 있다. 이와 같이 GAN의 잠재 공간을 적잘히 이용한다면, 속성 편향성이 해소되는 방향으로 훈련 데이터를 증강하는 것이 가능하다.

### Idea

Alleviating bias in training data via GAN latent space manipulation is an efficient data augmentation method. With GAN, it is possible to generate new images using only the original dataset, which reduces the need for consuming a lot of time and money to collect more training data. However, the training methods used for this kind of data augmentation were disadvantageous in the aspect of computational/architectural complexity of GAN models. Because new GAN model was created and trained whenever an attribute in need of de-biasing appeared, the computation time was long when there are many attributes in consideration. Also, some complex GANs such as image-to-image translation GAN were introduced, which made the implementation and interpretation of data augmentation more difficult. To address these problems, the author utilizes only a single GAN trained over the entire training dataset to alleviate the bias of all attributes under consideration.

## 3. Method

### 3-1. De-correlation definition

이 논문에서는 이미지의 속성과 레이블 간에 상관관계가 있는 경우를 다룬다. 예를 들어, 미국에서는 야외에서 선글라스를 쓰고 다니는 사람이 모자도 같이 착용하고 있는 경우가 많다. 그러므로, 아래의 사진에서와 같이, 선글라스를 쓰는 것(속성)과 모자의 착용 여부(레이블) 사이에 상관관계가 존재한다고 할 수 있다. 이러한 상황에서 야외 이미지들을 데이터 증강을 거치지 않고 바로 훈련 데이터로 사용한다면, 모자의 착용 여부를 판단하는 딥러닝 모델은 선글라스를 쓴 사람들보다 선글라스를 쓰지 않은 사람들에 대해 더 부정확한 예측을 내 놓을 수 있다. 그러므로 사전에 속성과 레이블 간의 상관관계가 제거되도록 훈련 데이터에 대해 데이터 증강 작업을 거치는 것은 중요하다. 

![Figure : Analytic expression of z'](../../.gitbook/assets/how-to-contribute/correlated.png)

데이터 증강을 거쳐 편향성이 제거된 데이터셋을 X<sub>aug</sub> 이라 하고, 공정성과 관련해서 고려하는 속성을 a 라고 하자. 딥러닝 모델이 임의의 x &in; X<sub>aug</sub> 에 대하여 예측하는 레이블 값을 t(x)라 정의하고 x의 예측 속성값을 a(x)라 하자. 가능한 레이블은 -1 또는 1 뿐이라고 가정하고, 속성값에 대해서도 똑같이 가정하자. 그렇다면 t(x)=1일 확률은 a(x)의 값과 무관해야 하며, 수식으로 표현하면 아래와 같다. 

![Figure : Analytic expression of z'](../../.gitbook/assets/how-to-contribute/decorrelation_condition.png)

### 3-2. De-correlation key idea

이 논문에서는 편향성이 제거된 데이터셋을 만들기 위해 예측 레이블은 동일하면서 예측 속성값은 서로 반대인 이미지 쌍을 생성하는 방법을 이용한다. GAN 모델이 기존 데이터셋에 대해 훈련을 마쳤다고 가정하자. 잠재 공간 내에서 임의로 z라는 점을 선택하면, GAN 모델은 점 z을 특정한 이미지로 변환할 것이다. 그 이미지에 대해 분류기 모델이 예측하는 레이블을 t(z)라 하고 예측 속성값을 a(z)라고 하자. 논문에서는 이때 아래의 조건을 만족하는 잠재 공간 내의 점 z’ 생성하여 z와 쌍을 이루게 한다.

![Figure : Analytic expression of z'](../../.gitbook/assets/how-to-contribute/z_prime_def.png)

If the pair (z, z’) is formed for each z in this way, the images corresponding to a given estimated label will have a uniform attribute distribution. Therefore the generated dataset X<sub>aug</sub> is a dataset in which the correlation between attribute and label is removed. The figure below shows how wearing glasses (attribute) and wearing a hat (label) are de-correlated by performing data augmentation based on the pairing (z, z’) in GAN latent space.

![Figure : Analytic expression of z'](../../.gitbook/assets/how-to-contribute/augmentation_overview.png)

### 3-3. How to calculate z’

The author introduces the linear-separability assumption of latent space with respect to attributes to find an analytic expression of z’. Then it is possible to regard the functions t(z) and a(z) as hyperplanes w<sub>t</sub> and w<sub>a</sub>, respectively. When the intercept of the hyperplane a(z) is denoted by b<sub>a</sub>, the equation of z’ is as shown below, according to the paper.

![Figure : Analytic expression of z'](../../.gitbook/assets/how-to-contribute/z_prime.png)


## 4. Experiment & Result

### Experimental setup

#### Dataset
해당 논문에서는 딥러닝 모델의 '성별'에 따른 공정성을 측정하는 실험을 한다. 즉 성별을 제외한 속성들의 값을 예측할 때, 예측 결과가 성별에 따라 얼마나 차이를 보이는 지 측정한다. 저자는 실험을 위해, 유명인의 얼굴 사진으로 이루어진 데이터셋인 CelebA를 이용한다. 여기에는 약 200만 개의 이미지가 들어 있고. 각 이미지에는 40개의 이진 속성(binary attributes)에 대한 정보가 담겨 있다. 저자는 40개의 속성 중 Male 속성을 '성별'로 간주하고 모델의 훈련에 이용하며, Male을 제외한 나머지 39개의 속성은 공정성 측정 단계에서 레이블로 이용한다. 논문에서는 39개의 속성을 데이터의 일관성 및 성별과의 연관성에 따라 아래의 세 가지 범주로 분류한다.

(1) Inconsistently Labeled : 속성값과 실제 이미지를 비교했을 때 일관성이 부족한 경우

(2) Gender-dependent : 속성값과 실제 이미지 간의 관계가 Male 여부에 영향을 받는 경우

(3) Geneder-independent : 그 외의 경우


#### Baseline model
실험에서 사용되는 기준 모델(baseline model)로서 사전에 ImageNet에서 훈련된 ResNet-50 모델을 이용한다. 해당 모델에서 완전연결 계층(fully-connected layer)은 크기 2,048의 은닉층을 사이에 둔 이중 선형 레이어로 교체되며, 드롭아웃 및 ReLU가 도입된다. 그런 다음 CelebA 훈련 데이터셋을 이용하여 이 모델을 20 에포크(epoch)동안 학습시킨다. 학습률은 1e-4이고, 배치 사이즈는 32이다. 손실함수로 이진 크로스 엔트로피(binary cross entropy)가 사용되며, 최적화 알고리즘으로는 Adam을 이용한다.

#### Data Augmentation
Progressive GAN is used during the de-biasing data augmentation. The latent space is set 512 dimensional, and the hyperplanes t(z) and a(z) are derived using linear SVM.

CelebA training dataset is used to train the progressive GAN. Then data augmentation is done using the trained GAN, in which 10k image are produced.

#### Evaluated model & Training setup
The model under evaluation is basically the same as the baseline model. However, it is trained using both the datasets X and X<sub>aug</sub>, while the baseline model is trained using only the biased dataset X. The training conditions are the same as the baseline model.

#### Evaluation Metrics 
The author uses four evaluation metrics, which are described below. The metrics except AP are used to evaluate fairness, and each of them is assumed to be better as it moves closer to zero.

(1) AP (Average Precision) : The overall precision accuracy.

(2) DEO (Difference in Equality of Opportunity) : The difference in false negative rates for different attribute values.

(3) BA (Bias Amplification) : A measure of how more frequently the model estimates a label compared to the actual label frequency. 

(4) KL : The KL divergence between the classifier output score distributions for different attribute values. To overcome the dissimilarity of KL divergence, it is added to the KL divergence obtained by switching the two distributions.

### Result

The table below shows the evaluation results of the baseline model and the new model, based on the four evaluation metrics (AP, DEO, BA, KL). Each metric is derived for each attribute group (Inconsistently Labeled, Gender-dependent, Gender-independent); each figure indicates the average of metrics calculated for the attributes in the group.

![Figure : Analytic expression of z'](../../.gitbook/assets/how-to-contribute/result.png)

표를 보면 데이터 증강 후 세 공정성 지표(DEO, BA, KL)가 모두 이전보다 개선된 것을 알 수 있다. 성별 의존적(Gender-dependent) 속성 집단의 경우 다른 집단에 비해 공정성의 향상이 약하게 이루어진 것을 볼 수 있는데, 저자에 따르면 논문의 Section 5에 설명된 것처럼 데이터 증강 방법을 확장함으로써 이 문제를 개선하는 것이 가능하다. 한편 전반적인 예측 정확도(AP)는 감소한 것을 볼 수 있는데, 이는 공정성을 향상시키기 위해 정확도를 약간 희생한 것으로 생각할 수 있다. 정확도의 감소 폭이 크지 않기 때문에, 모델의 공정성이 중요한 경우 이 논문의 데이터 증강 방법을 이용하는 것은 괜찮은 시도라고 할 수 있다. 

## 5. Conclusion

이 논문에서는 딥러닝 모델의 공정성 문제를 해결하기 위해 GAN 모델의 잠재 공간을 이용하여 편향성이 제거된 데이터셋을 생성하고 이를 이용해 원래의 훈련 데이터셋을 증강하는 방법을 이용하였다. 그리고 실험을 통해, 이 방법이 모델의 정확도를 크게 희생하지 않으면서 공정성을 높일 수 있는 방법이라는 것을 확인할 수 있었다. 개인적으로, 데이터 증강을 위해 GAN을 이용하는 것은 매력적인 방법이라 생각한다. 새로운 훈련 데이터를 GAN을 통해 자동으로 생성할 수 있으므로, 수작업으로 하는 것에 비하면 데이터 증강에 드는 시간 및 비용이 매우 적다. 또한 GAN에서 생성되는 이미지가 실제 이미지와 매우 비슷하므로, 고전적인 영상 처리 방법을 이용하는 것에 비해 더욱 자연스러운 이미지를 만들어낼 수 있을 것이다. 또한, 이 논문의 데이터 증강 방법에서 오직 한 개의 GAN 모델이 이용되므로, 논문에서 제시한 방법은 실제 구현 난이도 측면에서 이점이 있다고 생각한다.

### Take home message \(오늘의 교훈\)

> Un-biased dataset can be generated by the manipulation of GAN latent space, thus improving the model fairness.
>
> Data augmentation using GAN is advantageous in terms of efficiency and data quality.
>
> Using only a single GAN is attractive in the aspect of actual implementation.

## Author / Reviewer information

### Author

김대혁 \(Kim Daehyeok\) 

* KAIST EE, U-AIM Lab.
* Research Interest : Speech Recognition, Fairness
* Contact Email : kimshine@kaist.ac.kr

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. Ramaswamy, Vikram V., Sunnie SY Kim, and Olga Russakovsky. "Fair attribute classification through latent space de-biasing." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
2. https://github.com/princetonvisualai/gan-debiasing
3. Rameen Abdal, Yipeng Qin, and Peter Wonka. Im- age2StyleGAN: How to embed images into the StyleGAN latent space? In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019.
4. Mohsan Alvi, Andrew Zisserman, and Christoffer Nella ̊ker. Turning a blind eye: Explicit removal of biases and variation from deep neural network embeddings. In Proceedings of the European Conference on Computer Vision (ECCV), 2018.
5. Sina Baharlouei, Maher Nouiehed, Ahmad Beirami, and Meisam Razaviyayn. Re ́nyi fair inference. In Proceedings of the International Conference on Learning Representations (ICLR), 2020.
6. GuhaBalakrishnan,YuanjunXiong,WeiXia,andPietroPer- ona. Towards causal benchmarking of bias in face analysis algorithms. In Proceedings of European Conference on Com- puter Vision (ECCV), 2020.
7. DavidBau,Jun-YanZhu,JonasWulff,WilliamPeebles,Hen- drik Strobelt, Bolei Zhou, and Antonio Torralba. Seeing what a GAN cannot generate. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019.
8. RachelK.E.Bellamy,KuntalDey,MichaelHind,SamuelC. Hoffman, Stephanie Houde, Kalapriya Kannan, Pranay Lo- hia, Jacquelyn Martino, Sameep Mehta, Aleksandra Mo- jsilovic, Seema Nagar, Karthikeyan Natesan Ramamurthy, John Richards, Diptikalyan Saha, Prasanna Sattigeri, Monin- der Singh, Kush R. Varshney, and Yunfeng Zhang. AI Fairness 360: An extensible toolkit for detecting, understanding, and mitigating unwanted algorithmic bias, Oct. 2018.
9. Steffen Bickel, Michael Bru ̈ckner, and Tobias Scheffer. Dis- criminative learning under covariate shift. Journal of Machine Learning Research, 10(Sep):2137–2155, 2009.
10. Tolga Bolukbasi, Kai-Wei Chang, James Y Zou, Venkatesh Saligrama, and Adam T Kalai. Man is to computer program- mer as woman is to homemaker? debiasing word embeddings. In Advances in Neural Information Processing Systems, pages 4349–4357, 2016.
11. Joy Buolamwini and Timnit Gebru. Gender shades: Intersec- tional accuracy disparities in commercial gender classification. In Proceedings of the Conference on Fairness, Accountability, and Transparency, pages 77–91, 2018.
12. Toon Calders, Faisal Kamiran, and Mykola Pechenizkiy. Building classifiers with independency constraints. In 2009 IEEE International Conference on Data Mining Workshops, pages 13–18. IEEE, 2009.
13. Mingliang Chen and Min Wu. Towards threshold invariant fair classification. In Proceedings of the Conference on Un- certainty in Artificial Intelligence (UAI), 2020. 
14. Kristy Choi, Aditya Grover, Rui Shu, and Stefano Ermon. Fair generative modeling via weak supervision. In Proceedings of the International Conference on Machine Learning (ICML), 2020. 
15. Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, and Jaegul Choo. StarGAN: Unified genera- tive adversarial networks for multi-domain image-to-image translation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018.
16. Emily Denton, Ben Hutchinson, Margaret Mitchell, and Timnit Gebru. Image counterfactual sensitivity analysis for detecting unintended bias. In CVPR 2019 Workshop on Fair- ness Accountability Transparency and Ethics in Computer Vision, 2019.
17. Charles Elkan. The foundations of cost-sensitive learning. In Proceedings of the International Joint Conferences on Artificial Intelligence (IJCAI), volume 17, pages 973–978. Lawrence Erlbaum Associates Ltd, 2001.
18. FAIR HDGAN. Pytorch GAN Zoo.
19. Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, pages 2672–2680, 2014.
20. Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron C Courville. Improved training of Wasserstein GANs. In Advances in Neural Information Pro- cessing Systems, pages 5767–5777, 2017.
21. Moritz Hardt, Eric Price, and Nati Srebro. Equality of oppor- tunity in supervised learning. In Advances in Neural Informa- tion Processing Systems, pages 3315–3323, 2016.
22. Bharath Hariharan and Ross Girshick. Low-shot visual recog- nition by shrinking and hallucinating features. In Proceedings of the IEEE/CVF International Conference on Computer Vi- sion (ICCV), pages 3018–3027, 2017.
23. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016.
24. Lisa Anne Hendricks, Kaylee Burns, Kate Saenko, Trevor Darrell, and Anna Rohrbach. Women also snowboard: Over- coming bias in captioning models. In Proceedings of Euro- pean Conference on Computer Vision (ECCV), pages 793– 811. Springer, 2018.
25. Khari Johnson. Google Cloud AI removes gender labels from Cloud Vision API to avoid bias, 02 2020.
26. Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of GANs for improved quality, stability, and variation. In Proceedings of the International Conference on Learning Representations (ICLR), 2018.
27. Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4401–4410, 2019.
28. Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings of the International Conference on Learning Representations (ICLR), 2015.
29. Steven Liu, Tongzhou Wang, David Bau, Jun-Yan Zhu, and Antonio Torralba. Diverse image generation via self- conditioned GANs. In Proceedings of the IEEE/CVF Confer- ence on Computer Vision and Pattern Recognition (CVPR), 2020.
30. Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3730–3738, 2015.
31. Vishnu Suresh Lokhande, Aditya Kumar Akash, Sathya N. Ravi, and Vikas Singh. FairALM: Augmented lagrangian method for training fair models with little regret. In Proceed- ings of European Conference on Computer Vision (ECCV), 2020.
32. Junhyun Nam, Hyuntak Cha, Sungsoo Ahn, Jaeho Lee, and Jinwoo Shin. Learning from failure: Training debiased classi- fier from biased classifier. In Advances in Neural Information Processing Systems, 2020.
33. F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830, 2011.
34. Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, San- jeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International Journal of Com- puter Vision, 115(3):211–252, 2015.
35. Hee Jung Ryu, Hartwig Adam, and Margaret Mitchell. Inclu- siveFaceNet: Improving face attribute detection with race and gender diversity. In International Conference on Machine Learning (ICML) FATML Workshop, 2018.
36. Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training GANs. In Advances in Neural Information Pro- cessing Systems, pages 2234–2242, 2016.
37. PrasannaSattigeri,SamuelCHoffman,VijilChenthamarak- shan, and Kush R Varshney. Fairness GAN: Generating datasets with fairness properties using a generative adver- sarial network. IBM Journal of Research and Development, 63(4/5):3–1, 2019.
38. ViktoriiaSharmanska,LisaAnneHendricks,TrevorDarrell, and Novi Quadrianto. Contrastive examples for addressing the tyranny of the majority, 2020.
39. Yujun Shen, Jinjin Gu, Xiaoou Tang, and Bolei Zhou. Inter- preting the latent space of GANs for semantic face editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9243–9252, 2020.
40. AngelinaWang,ArvindNarayanan,andOlgaRussakovsky. REVISE: A tool for measuring and mitigating bias in visual datasets. In Proceedings of the European Conference on Computer Vision (ECCV), 2020.
41. Angelina Wang and Olga Russakovsky. Directional bias amplification. arXiv preprint arXiv:2102.12594, 2021.
42. TianluWang,JieyuZhao,MarkYatskar,Kai-WeiChang,and Vicente Ordonez. Balanced datasets are not enough: Estimat- ing and mitigating gender bias in deep image representations. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 5310–5319, 2019.
43. ZeyuWang,KlintQinami,IoannisKarakozis,KyleGenova, Prem Nair, Kenji Hata, and Olga Russakovsky. Towards fairness in visual recognition: Effective strategies for bias mitigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.
44. Depeng Xu, Shuhan Yuan, Lu Zhang, and Xintao Wu. Fair- GAN: Fairness-aware generative adversarial networks. In 2018 IEEE International Conference on Big Data (Big Data), pages 570–575. IEEE, 2018.
45. Kaiyu Yang, Klint Qinami, Li Fei-Fei, Jia Deng, and Olga Russakovsky. Towards fairer datasets: Filtering and balancing the distribution of the people subtree in the imagenet hierarchy. In Proceedings of the Conference on Fairness, Accountability, and Transparency, pages 547–558, 2020.
46. Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Ro- driguez, and Krishna P Gummadi. Fairness beyond disparate treatment & disparate impact: Learning classification without disparate mistreatment. In Proceedings of the International Conference on World Wide Web (WWW), pages 1171–1180, 2017.
47. Brian Hu Zhang, Blake Lemoine, and Margaret Mitchell. Mitigating unwanted biases with adversarial learning. In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society (AIES), pages 335–340, 2018.
48. Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. Men also like shopping: Reducing gen- der bias amplification using corpus-level constraints. In Pro- ceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2017.
49. Jiapeng Zhu, Yujun Shen, Deli Zhao, and Bolei Zhou. In- domain GAN inversion for real image editing. In Proceedings of European Conference on Computer Vision (ECCV), 2020.
50. XinyueZhu,YifanLiu,JiahongLi,TaoWan,andZengchang Qin. Emotion classification with data augmentation using generative adversarial networks. In Dinh Phung, Vincent S. Tseng, Geoffrey I. Webb, Bao Ho, Mohadeseh Ganji, and Lida Rashidi, editors, Advances in Knowledge Discovery and Data Mining (KDD), pages 349–360, Cham, 2018. Springer International Publishing.



