---
description: (Description) Yuan _et al_. / Multiple instance active learning for object detection / CVPR 2021
---

# Multiple Instance Active Learning for Object Detection \[Eng\]

한국어로 쓰인 리뷰를 읽으려면 [**여기**](cvpr-2021-miaod-kor.md)를 누르세요.
## **1. Problem Definition**
As given away by the title of the paper, the authors here are trying to use **Active Learning** for **Object Detection** on the **instance** level. So, let's clear out the definitions one-by-one.

### **Active Learning**

If you look up the term Active Learning on the internet, you would find several definitions, which are divided into two main categories: Human Active Learning and AI Active Learning. The former is basically a learning method where students are given more responsibility for their own learning journeys. They are encouraged to discuss with other students and dig deeper to find the solutions for the problems they stumble upon along the way [\[humanactivelearn\]][humanactive1]. For example, here at KAIST, for each course, instead of having a sylabus filled with only lectures, we are given time to work on projects, individual case studies that would actually help us fill our own knowledge gaps. So, the point of Active Learning is that the learners have to be actively seeking what to learn.  

In the context of Artifical Intelligent, the above-mentioned definition still holds. Here, instead of making the models learn all the available data samples, some of which may be not very useful, we can let the models themselves decide what to learn. This would actually save a lot of resources \(e.g., time and computation units\), while better improve the performances of our models. 

But how is that possible for the machine to know the knowledge it lacks, before actually having acquired that knowledge? It turns out that there is a way. We need to measure how uncertain the machine is about its predictions. And we will cover that in the next section.

### **Multiple Instance Learning**

In Object Detection, our input is usually a picture or a video frame, in which there could appear multiple objects of various categories including human, animal and vehicle. The job for our object detection model is to locate and classify the objects by drawing bounding boxes around them and give them the correct label. To predict the bounding boxes that we need to draw, a lot of modern approaches rely on first generating a huge number of anchor boxes. Based on those boxes, the models would start making modifications to them, so at the end we would have accurately drawn bounding boxes. 

![Figure 1: An example of Object Detection [source: https://pjreddie.com/darknet/yolo/]](/.gitbook/assets/11/object-detection.png)



For RetinaNet [\[lin2017\]][lin2017], the first step would be generating, or proposing, anchor boxes, or in the context of this paper, ***instances***. Since the task is to locate objects, we label the ones that contain only the background ***negative instances*** and the other that contain a part or the whole object ***positive instances***, where we could learn something useful about the objects. And a group of instances is called a ***bag***. In this paper, the authors refer to each image as a instance bag.

![Figure 2: Example of instance bags [source: MI-AOD's Figure 2]](/.gitbook/assets/11/instance-bag.png)

Among the instances, there are informative ones \(colored red\) that would benefit our model the most. Just as a human learner would learn the most from the subjects they do not know, these informative instances are the ones our model is most uncertain about. And the goal of this whole paper is ***to find the most informative bags of instances***.

### **Formal Definition**

By now, we have familiarized ourselves with the main concepts that would appear in the paper, it's time we gave the problem some formal definition.

In Machine Learning, there is simply nothing greater than well-labeled data. However, labeling data is no easy task and could take a mountain of human efforts. Therefore, being able to train models effectively on unlabelled data is the next best thing. In this paper, we have a small set of labeled data, denoted ![][x-y-0-l],
and a much larger set of unlabeled image data, denoted ![][x-0-u]. Each set ![][x-0-l] and ![][x-0-u] contains a number of images. Each image ![][x-in-x-0-l] or ![][x-in-x-0-u] is represented as a bag of instances ![][x-set], where *N* is the number of instances. The image label of the label set ![][y-0-l] consists of the coordinates of the bounding boxes ![][y-loc-x] and the categories ![][y-cls-x].

In this paper, the model *M* is first trained on the labeled data ![][x-y-0-l], and then retrained on the unlabeled set. The goal here is to reap the benefits of the unlabeled set, without having to label its data manually. So through this Active Learning frame work, the model would be able to select ![][k] new images from the unlabeled set and give them some generated labels, so we can incorperate them into ![][x-0-l] to form the new labeled set.

## **2. Motivation**

### **Uncertainty**

![Figure 3: A cat-dog classifier [source: https://www.youtube.com/watch?v=toTcf7tZK8c&t=2061s]](/.gitbook/assets/11/catdog.png)

1. Two kinds of uncertainty
    
    Before going further, it is imperative that we clear out another concept. Earlier, we talked about how the images, or bag of instances, that are informative are actually the ones that the model is uncertain about. But how exactly do we do that? It cannot be simply done by measuring the output probabilities of, say, the logistic function, because those probilities will always sum to 1.

    For example, if we input a picture of a cat and a dog [\[mitlecture\]][mitlecture] into a model that has been trained with cat and dog pictures, we will probably get 0.51 and 0.49, as the output possibilities. Using that result, the model will still decide and be confident about its decision. But is that correct if we categorize this image into either cat or dog?

    To make it even simpler, let's consider a midterm exam consisting of 10 questions, each of which has 4 choices (A, B, C, and D). If you decide to choose only A, you may not choose the right answers for some questions, but at the end of the test you always get 25% of the points. This is refered to as ***Aleatoric Uncertainty*** or the ***uncertainty of data*** [\[ulkumen-uncertainty\]][ulkumen-uncertainty]. 
    
    However, as you study for the exam, you want to measure your knowledge gap to be filled. One way is to count how many right answers after you have finished 10 questions. Another way, more difficult but also more effective, is to measure how much you are uncertain about each question. This is refered to as ***Epistemic Uncertainty*** or the ***uncertainty of prediction*** [\[ulkumen-uncertainty\]][ulkumen-uncertainty], which is also what we would want to measure so that our model can get better from the questions it is uncertain about.

2. A way to measure Epistemic Uncertainty?
   
* Dropout at test time
  * Usually, we would only use dropout for the train phase, but here we can use it as a form of stochastic sampling.
  * For each dropout case, we would likely have a different output.

![Figure 4: Dropout Sampling to measure uncertainty [source: https://www.youtube.com/watch?v=toTcf7tZK8c&t=2061s]](/.gitbook/assets/11/dropout.png)

* Model Emsemble
  * In this case, we use model independently trained for sampling.
    
![Figure 5: Model Ensembling to measure uncertainty [source: https://www.youtube.com/watch?v=toTcf7tZK8c&t=2061s]](/.gitbook/assets/11/model-ensemble.png)

* At the end, by looking at many sample outputs for the same input, we can calculate the expectation and variance of the model's prediction. The larger the variance is, the more uncertain the model is.
    
![Figure 6: Model weights' distribution [source: https://www.youtube.com/watch?v=toTcf7tZK8c&t=2061s]](/.gitbook/assets/11/variance.png)

![Figure 7: Calculating Expectation and Variance for a model's output [source: https://www.youtube.com/watch?v=toTcf7tZK8c&t=2061s]](/.gitbook/assets/11/variance2.png)

### **Related Work**
1. Uncertainty-based Active Learning
    
    Over the years, there have been quite a number of attempts at Active Learning. Lewis and Catlett [\[lewis1994a\]][lewis1994a] proposed a heterogeneous uncertainty sampling method to incrementally select data, and achieved significant improvement compared to human labeling method in terms of both time and error rate. Another uncertainty sampling method was proposed in [\[lewis1994b\]][lewis1994b] to train text classifiers using only a small amount of labeled data. This method can generalize to other classification problems. [\[roth2006\]][roth2006][\[joshi2010\]][joshi2010] proposed to look at margins between individual classifiers in order to improve the global performance. [\[lou2013\]][lou2013][\[settles2008\]][settles2008] also considered margins of local classifiers but instead used entropy to measure uncertainty. [\[roy2007\]][roy2007] proposed an Entropy-reducing algorithm for a Naive Bayes text classifier.

    Multiple instance learning methods including [\[carbonneau2017\]][carbonneau2017][\[wang2017\]][wang2017][\[hwang2017\]][hwang2017] follow a similar approach compared to this paper to find the most representative instaces among training data. However, they can only be applied to image classification.


### **The proposed idea**

In this paper, the authors proposed a Multiple Instance Active Object Detection (MI-AOD) approach. As explained earlier, the goal of this apporach is to find the most informative unlabeled images to be incorperated into the labeled training set. Therefore, it needs the abilities to:
* Measure the uncertainty of an image.
* Give that image a label.
  
Let's discuss them one by one.

In Section 1, we saw that a simple way to measure the model's uncertainty is that for each input image, we sample a lot of not only outputs but also the network weights through either dropout or model ensemble. However, it is prohibitively expensive to do so. For example, a regular medium-sized model nowadays can take up to a few GB in the GPU. If we are to sample enough samples, say a few thousands, can you imagine the number of GPUs we need!

So, instead of doing the unthinkable, the authors employ another method. To perform instance uncertainty learning (IUL), they use a two-head network with each head being a classifier (f1 and f2 in Figure 8) and train the network in a way that would maximize the prediction discrepancy. The intuition here is that instead of traing many identical networks and see how each performance is different from the others, we can train two classifiers to perform as differently as possible. 

Looking at the figure, you may be quite confused now, rightfully as I was when I first saw it. Maximizing Instance Uncertainty and then Minimizing it? Actually, this is two slightly different training processes. The first one, as we have discussed earlier, focuses on maximizing the discrepancy between two classifiers. But once f1 and f2 have learned to make wildly different predictions, we can use them to reduce the bias discrepancy between the labeled and unlabeled sets.

![Figure 8: Multiple Instance Uncertainty Learning [source: MI-AOD's Figure 2]](/.gitbook/assets/11/iul.png)

Now, we have been able to measure the instance uncertainty of the model, we should be able to pick informative images? No, at least not yet. Imagine we are training to better recognize dogs and we have two pictures, one full of dogs and the other has only one dog among many other more representative objects. Assuming, the model show the same level of uncertainty for both pictures, both pictures can be labeled 'dog', since there are dogs in both of them. However, it is glaringly obvious that the one filled with dogs would be more useful to our model. Here, we have to distinguish between instance uncertainty and image label uncertainty. MI-AOD uses a Multiple instance learning (MIL) module to perform instance uncertainty reweighting (IUR), forcing appearance consistency across images. Only then can we find the informative images within the unlabeled dataset.

![Figure 9: Multiple Instance Uncertainty Reweighting [source: MI-AOD's Figure 2]](/.gitbook/assets/11/iur.png)

The training procedures of IUL and IUR are nearly identical. The only difference is that IUR tries to achieve the consistency between instance label and image label. We would see how both are designed in detail in the next section.

## **3. Method**

Before we dive into the details, let's take a quick overview look at the training procedure. Each training cycle consists of two phases, IUL and IUR, which both are made up from 3 stages:

* Label training
* Training to Maximize the instance uncertainty between two classifiers
* Training to Minimize instance uncertainty between the labeled and unlabeled training sets.

### ***Instace Uncertainty Learning (IUL)***

![Figure 10: IUL training process [source: MI-AOD's Figure 3]](/.gitbook/assets/11/iul-training.png)

1. Label Set Training

    Looking at part (a) of the figure, we can see 4 components. ![][g] is the base network, RetinaNet [\[lin2017\]][lin2017], in charge of extracting the features, and ![][theta-g] is the set of parameters of ![][g]. As mentioned earlier, we have two classifier heads, ![][f1] and ![][f2], stacked on top of the network ![][g]. Moreover, regressor ![][fr] is in charge of learning the bounding boxes. ![][theta-set] denotes the set of all parameters, in which ![][theta-f1] and ![][theta-f2] are independently initialized as they are supposed to be trained adversarially. We have the detection loss for each image:

    ![][equation1] (1),

    Where:
        * FL(.) is the focal loss proposed in RetinaNet [\[lin2017\]][lin2017].
        * ![][i] is the instance number.
        * ![][yhat-f1],![][yhat-f2], and ![][yhat-fr] are the prediction of each classifier for instance number ![][i]

    At this stage, the training is only done on the labeled set. The objective is to get the model familiarzed with the labeled training data so it can later generalize on the unlabeled set. Since ![][f1] and ![][f2] were initialized independently, we could see some discrepancy in their predictions. However, this is not the objective a`t this stage.

2.  Maximizing Instance Uncertainty
   
    In part (b) of the figure, it can be seen that the unlabeled data is now put to use. But one particularly strange thing is the weights of the base network ![][theta-g] are frozen. This is because during the last stage, the base network has learnt to recognize the features of the instances. We now can freeze it to focus the training on maximizing prediction discrepancy between two classifiers. The loss function becomes:

    ![][equation2] (2)

    To force the adversarial training, the discrepancy is defined as:

    ![][equation3] (3)

    Here, if we take a look at Figure 9, we will see that the decision boundaries of two classifiers will be distant from each other. especially on the instances we consider informative.

3.  Minimizing Instance Uncertainty

    Now that we have gotten two classifiers that maximize the instance uncertainty, we have another concern. The data distributions of the labeled set are most certainly different from those of the unlabeled to some extent. To remedy this issue, we freeze the classifiers and regressor to focus on training only the base network. The loss function becomes:

    ![][equation4] (4)

### **Instance Uncertainty Re-weighting (IUR)**

RetinaNet generates roughly 100k instances per image, some of which simply contain background noise. Therefore, in this phase, to improve the efficiency of the model, we need to make sure that instance uncertainty is aligned with image uncertainty.

![Figure 11: IUR training process [source: MI-AOD's Figure 4]](/.gitbook/assets/11/iur-training.png)

1. Multiple Instance Learning (MIL)
    
    To force a consistency between instance uncertainty and image uncertainty, we first must be able to classify the image. The classification score is calculated as below:

    ![][equation5](5)

    Where:
    * ![][yhat-ic] is the classification score indicating that instance ![][i] belongs to the class *c*.
    * ![][f-mil] is the multiple instance classifier.

    Here we see something very familiar, the softmax function. The first term is given by the classification score, indicating the probability of image belonging to class *c* , based on the predictions of ![][f-mil]. But what's more important is in the second term. If ![][f1] and ![][f2] cannot find a large number of instances that belong to class *c* in the image, the overall score will be close to 0. The loss function is as below:

    ![][equation6] (6)

    This is a standard classification entropy loss. It makes sure that when an image is to be trained with class label *c* it must have enough instances that belong to class *c*, and that instances with low score will be considered background or noise.

2. Uncertainty Re-weighting
    Here, we go a step further to make sure that instance uncertainty is consistent with image uncertainty across the whole dataset. The re-weighted discrepancy is modeled as:

    ![][equation7] (7)

    The loss function becomes:

    ![][equation8] (8)

    By incorporating ![][yhat-i-cls] into (7), we force ![][tilde-l-dis] to consider the class score for each instance. By keeping ![][theta-g] fixed, the classifiers will be trained to consider only the instance with high classification scores, while ![][f1] and ![][f2] still try to maximize the instance uncertainty.

    For the last stage, with the weights of the classifiers and the regressor frozen, we retrain the base network to mitigate distribution biases between the labeled and unlabeled set.

    ![][equation9] (9)

## **4. Experiment and Result**

### **Experimental Setup**
1. Datasets
   
   The authors use two standard Object Detection dataset for training this model.

   * PASCAL VOC 2007 *trainval* for Active Training  and *test* for evaluating mAP [\[voc2007\]][voc2007]
   * MS COCO *train* for Active Training and and *val* for evaluating mAP [\[coco2015\]][coco2015]

2. Active Learning settings
   
   Two object detection models are employed for evaluating the performance of MI-AOD.
   * RetinaNet [\[lin2017\]][lin2017] with ResNet-50 [\[he2015\]][he2015]
     * Each cycle, the model is trained at 0.001 learning rate and mini batches of 2 for 26 epochs. 
     * After 20 epochs, the learning rate is decreased 10 times.
     * Momentum and weight decay rate is 0.9 and 0.0001 respectively.
   * SSD [\[liu2016\]][liu2016] with VGG-16 [\[simonyan2015\]][simonyan2015]
     * Each cycle, the model is trained for 240 epochs at 0.001 learning rate and later at 0.0001 for the remaining 60 epochs.
     * The minibatch size is 32.

### **Performance**

![Figure 12: Performance of MI-AOD compared to other methods [source: MI-AOD's Figure 5]](/.gitbook/assets/11/performance.png)

Overall, we can see that MI-AOD outperforms all other Active Learning instances for the task of Object Detection, for all the label portion settings. This proves that learning from instance uncertainty helps the model focus on the useful feature instances and informative training samples. You may see the APs are very low compared to other Object Detection models these days. But keep in mind that in this paper, the model was initially trained with only 5% and 2% labeled data for VOC2007 and COCO respectively. The rest was unlabeled and the model itself had to label it and integrate the informative samples into the training data. So, this is a very decent result.

### **Ablation Study**

![](/.gitbook/assets/11/table1.png)

There are some interesting things we can point out in the ablation study. I think it is better to look at the data to see how much it supports the authors' arguments in the previous sections.

![](/.gitbook/assets/11/table2.png)

* IUL and IUR

    * Looking at Table 1, for both IUL and IUR, even using random sample selections still improve the performance significantly. If we assume the random unlabeled data added into the labeled set is not very useful and sometimes can be actually harmful to the model, then we can say the model has done a good job filtering out uninformative instances and images.
    * It is quite interesting to see the Mean Uncertainty sample selection outperforms Max Uncertainty as I think it confirms one of the arguments earlier that Instance Uncertainty can be inconsistent with Image Uncertainty. Thus, averaging the uncertainty out help represent the image better.
    * This can be further illustrated in Table 3. Using ![][yhat-i-cls] means we are trying to surpress the classes of objects that are not going to be very useful.

![](/.gitbook/assets/11/table4.png)

* Hyper-parameters

    * Table 4 shows the performances of the model when using different values of ![][lambda] and ![][k].
    * From equations (2), (4), (8), and (9), if ![][lambda] is too low, the uncertainty learning on unlabeled set has little impact.
    * If we increase ![][lambda], we either encourage or discourage instance uncertainty, depending on which stage. That could be the reason a neutral value, 0.5, works best. It would be interesting to see the performance if we use two ![][lambda] values for two stages.

![](/.gitbook/assets/11/table5.png)

*  Table 5 shows the training time of MI-AOD compared to two other methods.


   [k]: /.gitbook/assets/11/equations/k.png
   [lambda]: /.gitbook/assets/11/equations/lambda.png

### **Model Analysis**

![Figure 13: Visual Analysis of MI-AOD's performance at different stages [source: MI-AOD's Figure 6]](/.gitbook/assets/11/visual-analysis.png)

1. Visual Analysis

    Figure 13 shows the heat map of model's output after each stage. It is calculated by summarizing the uncertainty score of all instances. The high uncertainty score should be focused around the objects of interest, because the closer all the uncertain instances are to the objects, the more useful features we could learn. We can see that by applying different stages, the uncertain region slowly closed down on the objects.

2. Statistical Analysis

    Figure 14 shows the number of true positive instances hit by each methods.

![Figure 14: Statistical Analysis of MI-AOD's performance compared to other methods [source: MI-AOD's Figure 7]](/.gitbook/assets/11/stat-analysis.png)

## 5. Conclusion

To be honest, I really enjoyed reading this paper. It proposed a well-designed active learning method for Object Detection based on Instance Uncertainty. The two uncertainty learning modules, IUL and IUR, where different parts of the model are frozen at different stages to maximize the uncertainty gain are very interesting.

Where I think the paper still has room for improvement is that it has too many stages, which means there could be a lot of hyper-parameters to be tuned. Also, I would really love it more if the authors spent more time analysising the results and comparing them to their initial arguments.

### Take home message
        Prediction output probablity and Prediction confidence (or Uncertainty) are two different things.

        Like humans, it is probably good for the machine to dig deeper where it is uncertain.

## Author/Reviewer Information

### Author
* Affiliation: KAIST CS
* I'm a PhD student at KAIST CS, mainly working on Computer Vision for Edge Computing. If you want to discuss some ideas, feel free to hit me up.
* Contact:
  * Email: tungnt at kaist.ac.kr
  * Github: tungnt101294

### Reviewer
1. a
2. b
3. c

## References and Additional Materials

### References

[\[humanactivelearn\]][humanactive1] “What is Active Learning?” https://www.queensu.ca/teachingandlearning/modules/active/. (accessed Oct. 23, 2021).

[humanactive1]: https://www.queensu.ca/teachingandlearning/modules/active

[\[mitlecture\]][mitlecture] 
A. Amini, “MIT 6.S191: Evidential Deep Learning and Uncertainty.” https://www.youtube.com/watch?v=toTcf7tZK8c&t=2061s (accessed Oct. 23, 2021).

[mitlecture]: https://www.youtube.com/watch?v=toTcf7tZK8c&t=2061s

[\[ulkumen-uncertainty\]][ulkumen-uncertainty] C. R. Fox and G. Ulkumen, “Distinguishing Two Dimensions of Uncertainty,” SSRN Electron. J., 2021, doi: 10.2139/ssrn.3695311.

[ulkumen-uncertainty]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3695311

[\[lewis1994a\]][lewis1994a] D. D. Lewis and J. Catlett, “Heterogeneous Uncertainty Sampling for Supervised Learning,” Mach. Learn. Proc. 1994, pp. 148–156, 1994, doi: 10.1016/b978-1-55860-335-6.50026-x.

[lewis1994a]: http://www.cs.cornell.edu/courses/cs6740/2010fa/papers/lewis-catlett-uncertainty-sampling.pdf

[\[lewis1994b\]][lewis1994b] D. D. Lewis and W. A. Gale, “A sequential algorithm for training text classifiers,” Proc. 17th Annu. Int. ACM SIGIR Conf. Res. Dev. Inf. Retrieval, SIGIR 1994, pp. 3–12, 1994, doi: 10.1007/978-1-4471-2099-5_1.

[lewis1994b]: https://arxiv.org/pdf/cmp-lg/9407020.pdf

[\[roth2006\]][roth2006] D. Roth and K. Small, “Margin-based active learning for structured output spaces,” Lect. Notes Comput. Sci. (including Subser. Lect. Notes Artif. Intell. Lect. Notes Bioinformatics), vol. 4212 LNAI, pp. 413–424, 2006, doi: 10.1007/11871842_40.

[roth2006]: https://doi.org/10.1007/11871842_40

[\[joshi2010\]][joshi2010] A. J. Joshi, F. Porikli, and N. Papanikolopoulos, “Multi-class active learning for image classification,” 2009 IEEE Conf. Comput. Vis. Pattern Recognit., pp. 2372–2379, 2010, doi: 10.1109/cvpr.2009.5206627.

[joshi2010]: https://doi.org/10.1109/cvpr.2009.5206627

[\[lou2013\]][lou2013] W. Luo, A. G. Schwing, and R. Urtasun, “Latent structured active learning,” Adv. Neural Inf. Process. Syst., pp. 1–9, 2013.

[lou2013]: https://papers.nips.cc/paper/2013/hash/b6f0479ae87d244975439c6124592772-Abstract.html

[\[settles2008\]][settles2008] B. Settles and M. Craven, “An analysis of active learning strategies for sequence labeling tasks,” EMNLP 2008 - 2008 Conf. Empir. Methods Nat. Lang. Process. Proc. Conf. A Meet. SIGDAT, a Spec. Interes. Gr. ACL, pp. 1070–1079, 2008, doi: 10.3115/1613715.1613855.

[settles2008]: https://www.biostat.wisc.edu/~craven/papers/settles.emnlp08.pdf

[\[settles2007\]][settles2007] B. Settles, M. Craven, and S. Ray, “Multiple-instance Active Learning,” in NIPS, 2007, pp. 1289–1296.

[settles2007]: https://dl.acm.org/doi/10.5555/2981562.2981724%0A%0A.

[\[roy2007\]][roy2007] N. Roy, A. Mccallum, and M. W. Com, “Toward optimal active learning through monte carlo estimation of error reduction.,” Proc. Int. Conf. Mach. Learn., pp. 441–448, 2001.

[roy2007]: https://dl.acm.org/doi/10.5555/645530.655646

[\[carbonneau2017\]][carbonneau2017] M. Carbonneau, E. Granger, and G. Gagnon, “Bag-Level Aggregation for Multiple Instance Active Learning in Instance Classification Problems,” arXiv, 2017.

[carbonneau2017]: https://arxiv.org/abs/1710.02584

[\[wang2017\]][wang2017] R. Wang, X. Z. Wang, S. Kwong, and C. Xu, “Incorporating Diversity and Informativeness in Multiple-Instance Active Learning,” IEEE Trans. Fuzzy Syst., vol. 25, no. 6, pp. 1460–1475, 2017, doi: 10.1109/TFUZZ.2017.2717803.

[wang2017]: https://ieeexplore.ieee.org/document/7953641

[\[hwang2017\]][hwang2017] S. Huang, N. Gao, and S. Chen, “Multi-Instance Multi-Label Active Learning,” in International Joint Conference on Artificial Intelligence, 2017, pp. 1886–1892.

[hwang2017]: https://www.ijcai.org/proceedings/2017/0262.pdf

[\[sener2018\]][sener2018] O. Sener and S. Savarese, “Active Learning for Convolutional Neural Networks: A Core-Set Approach,” in ICLR, 2018, pp. 1–13.

[sener2018]: https://arxiv.org/abs/1708.00489

[\[lin2017\]][lin2017] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, “Focal Loss for Dense Object Detection,” arXiv, Aug. 2017, [Online]. Available: http://arxiv.org/abs/1708.02002.

[lin2017]: https://arxiv.org/abs/1708.02002

[\[voc2007\]][voc2007] E. M., V.-G. L., W. C. K. I., W. J., and Z. A., “The Pascal Visual Object Classes (VOC) Challenge,” Int. J. Comput. Vis., vol. 88, no. 2, pp. 303–338, 2010.

[voc2007]: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/

[\[coco2015\]][coco2015] T. Lin et al., “Microsoft COCO: Common Objects in Context,” arXiv, pp. 1–15, May 2015.

[coco2015]: http://arxiv.org/abs/1405.0312

[\[he2015\]][he2015] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in CVPR, 2016, pp. 770–778, doi: 10.1002/chin.200650130.

[he2015]: https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html

[\[simonyan2015\]][simonyan2015] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in 3rd International Conference on Learning Representations, ICLR 2015 - Conference Track Proceedings, 2015, pp. 1–14.

[simonyan2015]: https://arxiv.org/pdf/1409.1556.pdf

[\[liu2016\]][liu2016] W. Liu et al., “SSD: Single Shot MultiBox Detector,” in ECCV, 2016, vol. 9905, pp. 21–37, doi: 10.1007/978-3-319-46448-0.

[liu2016]: https://link.springer.com/chapter/10.1007/978-3-319-46448-0_2


[iur]: /.gitbook/assets/11/iur.png

[g]: /.gitbook/assets/11/equations/g.png
[i]: /.gitbook/assets/11/equations/i.png
[yhat-f1]: /.gitbook/assets/11/equations/yhat-f1.png
[yhat-f2]: /.gitbook/assets/11/equations/yhat-f2.png
[yhat-fr]: /.gitbook/assets/11/equations/yhat-fr.png
[equation1]:/.gitbook/assets/11/equations/equation1.png
[equation2]:/.gitbook/assets/11/equations/equation2.png
[equation3]:/.gitbook/assets/11/equations/equation3.png
[equation4]: /.gitbook/assets/11/equations/equation4.png
[equation5]: /.gitbook/assets/11/equations/equation5.png
[equation6]: /.gitbook/assets/11/equations/equation6.png
[equation7]: /.gitbook/assets/11/equations/equation7.png
[equation8]: /.gitbook/assets/11/equations/equation8.png
[equation9]: /.gitbook/assets/11/equations/equation9.png
[equation10]: /.gitbook/assets/11/equations/equation10.png
[theta-g]: /.gitbook/assets/11/equations/theta-g.png
[f1]: /.gitbook/assets/11/equations/f1.png
[f2]: /.gitbook/assets/11/equations/f2.png
[fr]: /.gitbook/assets/11/equations/fr.png
[theta-set]: /.gitbook/assets/11/equations/theta-set.png
[theta-f1]: /.gitbook/assets/11/equations/theta-f1.png
[theta-f2]: /.gitbook/assets/11/equations/theta-f2.png
[yhat-ic]: /.gitbook/assets/11/equations/yhat-ic.png
[f-mil]: /.gitbook/assets/11/equations/f-mil.png
[x-0-u]: /.gitbook/assets/11/equations/x-u-0.png
[x-0-l]: /.gitbook/assets/11/equations/x-0-l.png
[x-y-0-l]: /.gitbook/assets/11/equations/x-in-x-0-l.png
[x-in-x-0-l]: /.gitbook/assets/11/equations/x-in-x-0-l.png
[x-in-x-0-u]: /.gitbook/assets/11/equations/x-in-x-0-u.png
[x-set]: /.gitbook/assets/11/equations/x-set.png
[y-loc-x]: /.gitbook/assets/11/equations/y-loc-x.png
[y-cls-x]: /.gitbook/assets/11/equations/y-cls-x.png
[y-0-l]: /.gitbook/assets/11/equations/y-0-l.png
[yhat-i-cls]: /.gitbook/assets/11/equations/yhat-i-cls.png
[tilde-l-dis]: /.gitbook/assets/11/equations/tilde-l-dis.png
[table1]: /.gitbook/assets/11/table1.png
[table2]: /.gitbook/assets/11/table2.png
[table3]: /.gitbook/assets/11/table3.png
[table4]: /.gitbook/assets/11/table4.png
[table5]: /.gitbook/assets/11/table5.png

