---
description: Rajendran et al. / Meta-Learning Requires Meta-Augmentation / NeurIPS 2020
---

# Meta-Augment \[Eng\]

##  1. Problem definition

A standard supervised machine learning problem considers a set of training data ($$x^i, y^i$$) indexed by i and sampled from a task $$\mathcal{T}$$, where the goal is to learn a function $$x \mapsto \widehat{y}$$. Following [1], we rely mostly on meta-learning centric nomenclature but borrow the terms "support", "query", "episode" from the few-shot learning literature. In meta-learning, we have a set of tasks {$$\mathcal{T}^i$$}, where each task $$\mathcal{T}^i$$ is made of a support set $$\mathcal{D}^s_i$$ contains $$(x_s, y_s)$$, and a query set $$\mathcal{D}^q_i$$ contains ($$x_q, y_q$$) samples. The grouped support and query sets are referred to as an episode. The training and test sets of examples are replaced by _meta-training_ and _meta-test_ sets of tasks, each of which consist of episodes. The goal is to learn a _base learner_ that first observes support data  ($$x_s, y_s$$) for a new task, then outputs a model which yields correct prediction $$\widehat{y}_q$$ for $$x_q$$. The learned model for a i-th task is parameterized by $$\phi_i$$ and the well-generalized model which is adapted to each i-th task, i.e., the based learner, is $$\theta_0$$. The generalization of the adapted model is measured on the query set $$\mathcal{D}^q_i$$, and in turn used to optimize the based learner $$\theta_0$$ during meta-training.  In the scope of the project, to make it simplifier, we only consider classification tasks, this is commonly described as k-shot, N-way classification, indicating k examples in the support set, with class labels $$y_s, y_q \in [1, N]$$. Let $$\mathcal{L}$$ and $$\mu$$ denote the loss function and the inner-loop learning rate, respectively. The above process is formulated as the following optimization problem
$$
\theta^*_0 := \underset{\theta_0}{min} \mathbb{E}_{\mathcal{T}^i \sim  p(\mathcal{T})}[\mathcal{L}(f_{\phi_i}(\textbf{X}^q_i), \textbf{Y}^q_i)], \\
        s.t. \phi_i = \theta_0 - \mu\nabla_{\theta_0}\mathcal{L}(f_{\theta_0}(\textbf{X}^s_i), \textbf{Y}^s_i),
$$
where $$(\textbf{X}^{s(q)}_i, \textbf{Y}^{s(q)}_i)$$ represent the collection of samples and their corresponding labels for the support and query set respectively. In the meta-testing phase, to solve the new task $$\mathcal{T}^*$$, the optimal $$\theta^*_0$$ is fine-tuned on it support set $$\mathcal{D}^{s*}$$ to the resulting task-specific parameters $$\theta_*$$

## 2. Motivation

There are two forms of overfitting: (1) memorization overfitting, in which the model is able to overfit to the training set without relying on the learner and (2) learner overfitting, in which the learner overfits to the training set and does not generalize to the test set. Both types of overfitting hurt the generalization from meta-training to meta-testing tasks.

<img src="/.gitbook/assets/20/fig1.JPG" width="800" height="400" />

This paper introduces an information-theoretic framework of meta-augmentation, whereby adding randomness discourages the base learner and model from learning trivial solutions that do not generalize to new tasks. Specifically they propose to augment new tasks based on existing tasks. 

### Related work

Data augmentation has been applied to several domains with strong results, including image classification, speech recognition, reinforcement learning, and language learning. Within meta-learning, augmentation has been applied in several ways. Mehrotra and Dukkipati [3] train a generator to generate new examples for one-shot learning problems. Santoro et al. [4] augmented Omniglot using random translations and rotations to generate more examples within a task. Liu et al. [5] applied similar transforms, but treated it as task augmentation by defining each rotation as a new task. These augmentations add more data and tasks, but do not turn non-mutually-exclusive problems into mutually-exclusive ones, since the pairing between $$x_s, y_s$$ is still consistent across meta-learning episodes, leaving open the possibility of memorization overfitting. Antoniou and Storkey [6] and Khodadadeh et al. [7] generate tasks by randomly selecting xs from an unsupervised dataset, using data augmentation on xs to generate more examples for the random task.  The authors instead create a mutually-exclusive task setting by modifying ys to create more tasks with shared xs. The large interest in the field has spurred the creation of meta-learning benchmarks, investigations into tuning few-shot models, and analysis of what these models learn. For overfitting in MAML in particular, regularization has been done by encouraging the model’s output to be uniform before the base learner updates the model [8], limiting the updateable parameters [9], or regularizing the gradient update based on cosine similarity or dropout [10, 11]. Yin et al. [2] propose using a Variational Information Bottleneck to regularize the model by restricting the information flow between $$x_q$$ and $$y_q$$.  

### Idea

Correctly balancing regularization to prevent overfitting or underfitting can be challenging, as the relationship between constraints on model capacity and generalization are hard to predict.  For example, overparameterized networks have been empirically shown to have lower generalization error[12]. Rather than crippling the model by limiting its access to $$x_q$$, the authors instead use data augmentation to encourage the model to pay more attention to $$(x_s, y_s)$$.

We define an augmentation to be CE-preserving (conditional entropy preserving) if conditional entropy $$H(y^{'}|x^{'})=H(y|x)$$ is conserved; for instance, the rotation augmentation is CE-preserving because rotations in $$x^{'}$$ do not affect the predictiveness of the original or rotated image to the class label. CE-preserving augmentations are commonly used in image-based problems. Conversely, an augmentation is CE-increasing if it increases conditional entropy, $$H(y^{'}|x^{'})>H(y|x)$$. For example, if Y is continuous and $$\epsilon∼ U[−1, 1]$$, then $$f(x, y, \epsilon)=(x, y+\epsilon)$$ is CE-increasing, since $$(x^{'}, y^{'})$$ will have two examples $$(x, y_1),(x, y_2)$$ with shared x and different y, increasing $$H(y^{'}|x^{'})$$.

The authors propose to augment in the same way as classical machine learning methods, by applying a CE-preserving augmentation to each task. However, the overfitting problem in meta-learning requires different augmentation. They wish to couple $$(x_s, y_s),(x_q, y_q)$$ together such that the model cannot minimize training loss using $$x_q$$ alone. This can be done through CE-increasing augmentation. Labels $$y_s, y_q$$ are encrypted to $$y^{'}_s, y^{'}_q$$ with the same random key $$\epsilon$$, in a way such that the base learner can only recover $$\epsilon$$ by associating $$x_s  \rightarrow y^{'}_s$$ and doing so is necessary to associate $$x_q  \rightarrow y^{'}_q$$.

## 3. Method

<img src="/.gitbook/assets/20/fig2.JPG" width="800" height="400" />

**Theorem 1.** Let $$\epsilon$$ be a noise variable independent from X, Y, and $$g:\epsilon, Y \rightarrow Y$$ be the augmentation function. Let $$y^{'}=g(\epsilon, y)$$, and assume that $$(\epsilon, x, y)\mapsto (x,y^{'})$$ is a one-to-one function. Then $$H(Y^{'}|X)=H(Y|X)+H(\epsilon)$$.

In order to lower $$H(Y^{'}_q|X_q)$$ to a level where the task can be solved, the learner must extract at least $$H(\epsilon)$$ bits from $$(x_s, y^{'}_s)$$. This reduces memorization overfitting, since it guarantees $$(x_s, y^{'}_s)$$ has some information required to predict $$y^{'}_q$$, even given the model and $$x_q$$. 

By adding new and different varieties of tasks to the meta-train set, CE-increasing augmentations also help avoid learner overfitting and help the base learner generalize to test set tasks. This effect is similar to the effect that data augmentation has in classical machine learning to avoid overfitting.

Few shot classification benchmarks such as Mini-ImageNet [13] have meta-augmentation in them by default. New tasks created by shuffling the class index y of previous tasks are added to the training set. Here $$y^{'}=g(\epsilon, y)$$, where $$\epsilon$$ is a permutation sampled uniformly from all permutations. The $$\epsilon$$ can be viewd as an encryption key, which g applies to y to get $$y^{'}$$. This augmentation is CE-increasing, since given an initial label distribution Y, augmenting with $$Y^{'}=g(\epsilon, Y)$$ gives a uniform $$Y^{'}|X$$. Therefore, this makes the task setting mutually-exclusive, thereby reducing memorization overfitting. This is accompanied by creation of new tasks through combining classes from different tasks, adding more variation to the meta-train set. These added tasks help avoid learner overfitting.

For multivariate regression tasks where the support set contains a regression target, the dimensions of $$y_s, y_q$$ can be treated as class logits to be permuted. This reduces to an identical setup to the classification case. For scalar meta-learning regression tasks and situations where output dimensions cannot be permuted, we show the CE-increasing augmentation of adding uniform noise to the regression targets  $$y^{'}_s=y_s+\epsilon, y^{'}_q=y_q+\epsilon$$ generates enough new tasks to help reduce overfitting.

## 4. Experiment & Result

### 4.1. Few-shot image classification (Omniglot, Mini-ImageNet, D’Claw)

#### Experimental setup

* General settings
  * 1-shot, 5-way
  * Turn default mutually-exclusive benchmarks into non-mutually-exclusive versions of themselves by partitioning the classes into groups of N classes without overlap. These groups form the meta-train tasks, and over all of training, class order is never changed.
* Dataset
  * Omniglot: 1623 different handwritten characters from different alphabets
  * Mini-ImageNet: is a more complex few-shot dataset, based on the ILSVRC object classification dataset. There are 100 classes in total, with 600 samples each.
  * D'Claw: a small robotics-inspired image classification dataset that authors collected.
* Classification models: MAML, Prototypical, Matching
* Baselines: non-mutually-exclusive settings
* Evaluation metric: classification accuracy

<img src="/.gitbook/assets/20/fig3.JPG" width="750" height="230" />

#### Result

Common few-shot image classification benchmarks, like Omniglot and Mini-ImageNet, are already mutuallyexclusive by default through meta-augmentation. In order to study the effect of meta-augmentation using task shuffling on various datasets, we turn these mutually-exclusive benchmarks into nonmutually-exclusive versions of themselves by partitioning the classes into groups of N classes without overlap. These groups form the meta-train tasks, and over all of training, class order is never changed.

<img src="/.gitbook/assets/20/fig4.JPG" width="800" height="400" />

Table 1: Few-shot image classification test set results. Results use MAML unless otherwise stated. All results are in 1-shot 5-way classification, except for D’Claw which is 1-shot 2-way. (unit: %)

| Problem setting              | Non-mutually-exclusive accuracy | Intrashuffle accuracy | Intershuffle accuracy |
| ---------------------------- | ------------------------------- | --------------------- | --------------------- |
| Omniglot                     | 98.1                            | 98.5                  | **98.7**              |
| Mini-ImageNet (MAML)         | 30.2                            | 42.7                  | **46**                |
| Mini-ImageNet (Prototypical) | 32.5                            | 32.5                  | **37.2**              |
| Mini-ImageNet (matching)     | 33.8                            | 33.8                  | **39.8**              |
| D'Claw                       | 72.5                            | 79.8                  | **83.1**              |

### 4.2. Regression tasks (Sinusoid, Pascal3D Pose Regression)

#### Experimental setup

* General settings
  * Scalar output
* Dataset
  * Sinusoid: Synthesized 1D sine wave regression problem
  * Pascal3D pose regression . Each task is to take a 128x128 grayscale image of an object from the Pascal 3D dataset and predict its angular
    orientation $$y_q$$ (normalized between 0 and 10) about the Z-axis, with respect to some unobserved canonical pose specific to each object.
* Baselines: MAML, MR-MAML, CNP, MR-CNP
* Evaluation metric: Prediction mean square error and standard deviations

#### Result

<img src="/.gitbook/assets/20/fig5.JPG" width="800" height="400" />

Table 2: Pascal3D pose prediction error (MSE) means and standard deviations. Removing weight decay (WD) improves the MAML baseline and augmentation improves the MAML, MR-MAML, CNP, MR-CNP results. Bracketed numbers copied from Yin et al. [2].

| Method | MAML<br/> (WD=1e-3) | MAML<br/> (WD=0) | MR-MAML<br/>(β=0.001) | MR-MAML<br/>(β=0) | CNP           | MR-CNP       |
| ------ | ------------------- | ---------------- | --------------------- | ----------------- | ------------- | ------------ |
| No Aug | [5.39±1.31]         | 3.74 ± .64       | 2.41 ± .04            | 2.8 ± .73         | [8.48±.12]    | [2.89±.18]   |
| Aug    | **4.99±1.22**       | ** 2.34 ± .66 ** | **1.71 ± .16 **       | **1.61 ± .06 **   | **2.51±.17 ** | **2.51±.20** |



## 5. Conclusion

Memorization overfitting is just a classical machine learning problem in disguise: a function approximator pays too much attention to one input $$x_q$$, and not enough to the other input $$(x_s, y_s)$$, when the former is sufficient to solve the task at training time. The two inputs could take many forms, such as different subsets of pixels within the same image. By a similar analogy, learner overfitting corresponds to correct function approximation on input
examples $$((x_s, y_s), x_q)$$ from the training set, and a systematic failure to generalize from those to the test set. Although meta-augmentation is helpful, it still has its limitations. Distribution mismatch between train-time tasks and test-time tasks can be lessened through augmentation, but augmentation may not entirely remove it. Creating new tasks by shuffling the class index of previous tasks is a simple procedure. It may be more explored by considering a mathematical framework to augment novel tasks that systematically abide by a specific rule.

### Take home message \(오늘의 교훈\)

> 2 forms of meta-over fitting: memorization overfitting, learner overfitting.
>
> Meta-augmentation avoid both by shuffle classes in classification and add a random variable to y in regression.
>

## Author / Reviewer information

### Author

**Nguyen Ngoc Quang** 

* KAIST AI
* https://github.com/quangbk2010

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. Eleni, T., e. a. Meta-dataset: A dataset of datasets for learning to learn from few examples. In ICLR, 2020
2. Yin, M., T. G. Z. M. L. S. and Finn, C. Meta-learning without memorization. In ICLR, 2020
3. Akshay Mehrotra and Ambedkar Dukkipati. Generative adversarial residual pairwise networks for one shot learning. arXiv preprint arXiv:1703.08033, 2017
4. Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, and Timothy Lillicrap. Meta-learning with memory-augmented neural networks. In International conference on machine learning, pages 1842–1850, 2016 
5. Jialin Liu, Fei Chao, and Chih-Min Lin. Task augmentation by rotating for meta-learning. arXiv preprint arXiv:2003.00804, 2020
6. Antreas Antoniou and Amos Storkey. Assume, augment and learn: Unsupervised few-shot meta-learning via random labels and data augmentation. arXiv preprint arXiv:1902.09884, 2019
7. Siavash Khodadadeh, Ladislau Boloni, and Mubarak Shah. Unsupervised meta-learning for few-shot image classification. In Advances in Neural Information Processing Systems, pages 10132–10142, 2019
8. Muhammad Abdullah Jamal and Guo-Jun Qi. Task agnostic meta-learning for few-shot learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 11719–11727, 2019
9. Luisa Zintgraf, Kyriacos Shiarli, Vitaly Kurin, Katja Hofmann, and Shimon Whiteson. Fast context adaptation via meta-learning. In International Conference on Machine Learning, pages 7693–7702, 2019
10. Simon Guiroy, Vikas Verma, and Christopher Pal. Towards understanding generalization in gradient-based meta-learning. arXiv preprint arXiv:1907.07287, 2019
11. Hung-Yu Tseng, Yi-Wen Chen, Yi-Hsuan Tsai, Sifei Liu, Yen-Yu Lin, and Ming-Hsuan Yang. Regularizing meta-learning via gradient dropout, 2020
12. Roman Novak, Yasaman Bahri, Daniel A Abolafia, Jeffrey Pennington, and Jascha SohlDickstein. Sensitivity and generalization in neural networks: an empirical study. arXiv preprint arXiv:1802.08760, 2018
13. Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al. Matching networks for one shot learning. In Advances in neural information processing systems, pages 3630–3638, 2016
