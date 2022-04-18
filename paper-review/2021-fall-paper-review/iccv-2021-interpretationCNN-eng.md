---
description: Lam et al. / Finding Representative Interpretations on Convolutional Neural Networks / ICCV 2021
---

# Finding Representative Interpretations on Convolutional Neural Networks \[Eng]

한국어로 쓰인 리뷰를 읽으려면 [여기](iccv-2021-interpretationCNN-kor.md)를 누르세요.

## 1. Problem definition

Despite the success of deep learning models on various tasks, there is **a lack of interpretability to understand the decision logic behind deep learning models**. In fields where decision-making is critical to results, such as process systems and healthcare, it is hard to use models that lack interpretability due to reliability issues. It requires sufficient interpretability to make deep learning models widely applicable.

In this paper, the authors propose a new framework to interpret **the decision-making process of Deep convolutional neural networks(CNNs)** which are the basic architecture of many deep learning models. The goal is to develop **representative interpretations of a trained CNN to reveal the common semantics data** that contribute to many closely related predictions.

How can we find such representative interpretations of a trained CNN? Before reviewing the details, I introduce the summary for the paper.

1. Consider a function that maps the feature map produced by the last convolutional layer to the logits that denote the final decision.
2. Since this function is a piecewise linear function, it applies different decision logics for regions separated by linear boundaries.
3. For each image, the authors propose to solve the optimization problem to construct a subset of linear boundaries that provides good representative interpretations.

{% hint style="info" %}
\[Opinion]

It is reasonable to determine a CNN model with ReLU activation functions as a target to interpret, since this architecture has been sufficiently validated to perform well. Furthermore, the proposed method use optimization rather than heuristics so that it can give trustworthy solutions.
{% endhint %}



## 2. Motivation

### Related Work

There are various types of existing interpretation methods for CNNs.

1. Conceptual interpretation methods
   
   - e.g. [Automated Concept-based Explanation (ACE)](https://arxiv.org/abs/1902.03129)
   
   * identify a set of concepts that contribute to the predictions on a pre-defined group of conceptually similar images.
   * These methods require sophisticated customization on deep neural networks.
   
2. Example-based methods
   
   - e.g. [MMD-critic](https://dl.acm.org/doi/10.5555/3157096.3157352), [Prototypes of Temporally Activated Patterns (PTAP)](https://dl.acm.org/doi/abs/10.1145/3447548.3467346)
   
   * Find exemplar images to interpret the decision of a deep neural network.
   * Prototype-based methods summarize the entire model using a small number of instances as prototypes.
   * The selection of prototypes considers little about the decision process of the model.

### Idea

In this paper, the goal is to provide representative interpretations in a general CNN model by considering decision boundaries.

* Find the linear decision boundaries of the convex polytopes that encode the decision logic of a trained CNN.
* This problem can be formulated as a co-clustering problem. The co-clustering problem means that it finds one cluster for the set of similar images and the other cluster for the set of linear boundaries that cover the similar images.
* Convert the co-clustering problem into a submodular cost submodular cover (SCSC) problem to make the problem feasible.

## 3. Method

### Setting

Consider image classification using a CNN with ReLU activation functions.

* $$\cal{X}$$: the space of images
* $$C$$: the number of classes
* $$F:\mathcal{X}\rightarrow\mathbb{R}^C$$: a trained CNN, and $$Class(x)=\argmax_i F_i(x)$$
* a set of reference images $$R\subseteq\mathcal{X}$$
* $$\psi(x)$$: the feature map produced by the last convolutional layer of $$F$$
* $$\Omega=\{\psi(x)\;|\;x\in\mathcal{X} \}$$ the space of feature maps
* $$G:\Omega\rightarrow\mathbb{R}^C$$, the mapping from the feature map $$\psi(x)$$ to $$Class(x)$$
* $$\mathcal{P}$$: the set of the linear boundaries (hyperplanes) of $$G$$

Reference images denote unlabeled images that we want to interpret by this method.

### Representative Interpretations

Before formulating our problem, we have to specify a goal to find representative interpretations.

\[Representative interpretation]

- For an input image $$x\in\mathcal{X}$$, a representative interpretation on $$x$$ is an interpretation that reveals the common decision logic of $$F$$.

* It is a general approach to explain a decision logic by using $$G$$, which is the function from the feature map of the last convolutional layer to the class of $$x$$, when analyzing predictions of a trained DNN.
* Since $$G$$ is a piecewise linear function, it applies **different decision logics for regions separated by linear boundaries**. I recommend reading [the paper](https://dl.acm.org/doi/abs/10.1145/3219819.3220063?casa\_token=MojIMpYRbLcAAAAA:19vihLVFk09s\_3zS1mtVpaxYvX7Cor5Fbkvso6UlSJYhW\_qPkO2oM7MCKIqJrTZ\_GgQsPeNgC8RK) to understand the details.

![Decision logic of a CNN](../../.gitbook/assets/23/cnn\_decision\_logic.png)

\[Linear boundaries]

- The decision logic of $$G$$ can be characterized by a piecewise linear decision boundary that consists of connected pieces of decision hyperplanes. Denote the set of linear boundaries of $$G$$ by $$\cal{P}$$.

* The linear boundaries in $$\cal P$$ partition the space of feature maps $$\Omega$$ into a large number of convex polytopes. Each convex polytope defines a decision region that predicts all the images contained in the region to be the same class.

* However, not all convex polytopes play an important role in distinguishing labels. Therefore, finding a good decision region, which is a subset of $$\cal P$$ and includes $$x$$, provides a representative interpretation. That is, the goal is to find a good representative interpretation $$P(x)\subseteq\mathcal{P}$$.

{% hint style="info" %}
\[Goal]

For an input image $$x$$, find a representative interpretation that provides a good decision region $$P(x)\subseteq\mathcal{P}$$.
{% endhint %}

### Finding Representative Interpretations

What is a 'good' representative interpretation? It requires two conditions:

1. Maximize the representativeness of $$P(x)$$.

   → A decision region $$P(x)$$ has to cover a large number of reference images.

   → maximize $$|P(x)\cap R|$$

2. Avoid covering images in different classes.

   → $$|P(x)\cap D(x)|=0$$ where $$D(x)=\{x'\in R\;|\;Class(x')\neq Class(x)\}$$

It can be formulated as the following optimization problem. The authors call this problem as the co-clustering problem, since it finds one cluster for the set of similar images and the other cluster for the set of linear boundaries that cover the similar images simultaneously.

\[Co-clustering Problem]
$$
\max_{P(x)\subseteq\mathcal{P}}|P(x)\cap R|\\ \mathsf{s.t.}\quad|P(x)\cap D(x)|=0
$$

![Finding the optimal subset of linear boundaries](../../.gitbook/assets/23/RI\_cnn\_prob\_def.png)



However, a set optimization problem such as the co-clustering problem is computationally complex to optimize. Therefore, in this paper, the authors:

1. sample $$\cal Q$$ from $$\cal P$$ to reduce the size;
2. define submodular optimization problem to make the problem feasible. 

{% hint style="info" %}
What is Submodular Optimization?

* A set optimization problem that finds the optimal subset from candidates is computationally complex, since the computational cost increases exponentially as the number of candidates increases.
* When the objective function satisfies submodularity, the greedy algorithm achieves at least a constant fraction of the objective value obtained by the optimal solution.
* Therefore, submodular optimization makes a set optimization problem feasible with guaranteeing a sufficiently good performance.
* Submodularity requires diminishing return property. You can check the details [here](https://en.wikipedia.org/wiki/Submodular\_set\_function).
  {% endhint %}

{% hint style="info" %}
\[Opinion]

Even though the authors randomly sampled linear boundaries for $$\cal Q$$ to reduce complexity, we should verify whether important linear boundaries are omitted or not.
{% endhint %}

### Submodular Cost Submodular Cover problem

\[SCSC Problem]
$$
\max_{P(x)\subseteq\mathcal{Q}}|P(x)\cap R|\\ \mathsf{s.t.}\quad|P(x)\cap D(x)|\leq\delta
$$

* We can construct a set of linear boundaries $$\cal P$$ from function $$G$$ by [the method introduced in this paper](https://dl.acm.org/doi/abs/10.1145/3219819.3220063?casa\_token=MojIMpYRbLcAAAAA:19vihLVFk09s\_3zS1mtVpaxYvX7Cor5Fbkvso6UlSJYhW\_qPkO2oM7MCKIqJrTZ\_GgQsPeNgC8RK). Then, sample a subset of linear boundaries $$\cal Q$$ from $$\cal P$$.

* Due to sampling, the images covered in the same convex polytope may not be predicted by $$F$$ as the same class 
  → Relax the constraint $$|P(x)\cap D(x)|=0$$ into $$|P(x)\cap D(x)|\leq\delta$$.

* This formulation satisfies conditions for submodular cost and submodular cover. You can check it in Appendix A of [the paper](https://openaccess.thecvf.com/content/ICCV2021/html/Lam\_Finding\_Representative\_Interpretations\_on\_Convolutional\_Neural\_Networks\_ICCV\_2021\_paper.html).

* Finally, the SCSC problem can be solved by iteratively selecting a linear boundary through the following greedy algorithm.

![The greedy algorithm to find representative interpretations.](../../.gitbook/assets/23/greedy\_alg.png)

### Ranking Similar Images

Define a new semantic distance to evaluate images $$x'\in P(x)$$.

\[Semantic Distance]

$$
Dist(x.x')=\sum_{\mathbf{h}\in P(x)}\Big\vert \langle \overrightarrow{W}_\mathbf{h},\psi(x)\rangle -\langle \overrightarrow{W}_\mathbf{h},\psi(x')\rangle \Big\vert
$$

* $$\overrightarrow{W}_\mathbf{h}$$ is the normal vector of the hyperplane of a linear boundary $$\mathbf{h}\in P(x)$$.

* That is, it measures how far $$x'$$ is from $$x$$ in terms of hyperplanes in $$P(x)$$. Unlike the Euclidean distance, it can quantify the distance between $$x'$$ and $$x$$ in terms of the decision region.

* Rank the images covered by $$P(x)$$ according to their semantic distance to $$x$$ in ascending order.

![Semantic distances are different in terms of the decision region.](../../.gitbook/assets/23/semantic\_dist.png)

The figure describes the difference between the semantic distance and the Euclidean distance. Even though the Euclidean distances are the same, the semantic distance between  $$x_2$$ and $$x$$ is larger than the case of $$x_1$$ in terms of the decision region.

## 4. Experiment & Result

### Experimental setup

The authors compare representative interpretation (RI) method with Automatic Concept-based Explanation(ACE), CAM-based methods(Grad-CAM, Grad-CAM++, Score-CAM).

* Apply sampling with $$|\mathcal{Q}|=50$$.
* Such methodologies use channel weights to provide interpretability. Reuse the channel weights computed from the input image $$x\in\mathcal{X}$$, and follow the same heat map to generate the interpretation for $$x_{new}$$. Compare the results from the methodologies.
  * In the case of RI, use the semantic distance to find a set of similar images $$x_{new}$$.
  * In the other cases, use the Euclidean distance in the space of $$\Omega$$ to find a set of similar images $$x_{new}$$.
* Dataset: Gender Classification (GC), ASIRRA, Retinal OCT Images (RO), FOOD datasets
* Target model: VGG-19

### Result

#### Case Study

* This experiment evaluates if each method provides a proper interpretation for similar images.
* The first row shows the result retrieved by RI method. Unlike the other methods, the heat maps in images indicate consistent semantics in the images.
* RI method can successfully find the interpretation for the input image, as well as a set of images sharing the same interpretation.

![A case study on the GC dataset.](../../.gitbook/assets/23/case\_study.png)

#### Quantitative Experiment

In this experiment, the authors quantitatively evaluate how computed interpretations can be used to classify unseen dataset. The following two measures:

\[Average Drop (AD)]

$$
\frac{1}{|S|}\sum_{e\in S}\frac{\max(0,Y_c(e)-Y_c(e'))}{Y_c(e)}
$$
\[Average Increase (AI)]

$$
\frac{1}{|S|}\sum_{e\in S}\mathbb{1}_{Y_c(e)<Y_c(e')}
$$

* $$S\subseteq \mathcal{X}$$: a set of unseen images

* $$Y_c(e)$$: the prediction score for class $$c$$ in an image $$e\in S$$

* $$e'$$: a masked image produced by keeping 20% of the most important pixels in $$e$$

When keeping only important pixels in the image, AD indicates a decrease in accuracy and AI indicates the percentage of samples that increases in accuracy. A small mean AD(mAD) and a large mean AI(mAI) say that the interpretation can be validly reused to accurately identify important regions on the unseen images in $$S$$. In the figure, we can see that RI method achieves the best performances in most cases.

![Quantative results.](../../.gitbook/assets/23/quant\_exp.png)

## 5. Conclusion

- In this paper, a co-clustering problem is formulated to interpret the decision-making process of CNN by considering decision boundaries.
- To solve the co-clustering problem, the greedy algorithm can be applied by converting it into the SCSC problem.

* It has been experimentally shown that proposed representative interpretations reflect common semantics in the unseen images.

### Take home message

> As deep neural networks have been widely used in various fields, it is more important to interpret a decision logic of DNNs. In this spirit, it is impressive to suggest representative interpretations by considering decision boundaries and I hope to extend such studies further.

## Author / Reviewer information

### Author

**장원준 (Wonjoon Chang)**

* KAIST AI, Statistical Artificial Intelligence Lab.
* one\_jj@kaist.ac.kr
* Research Topics: Explainable AI, Time series analysis.
* https://github.com/onejoon

### Reviewer

- 

## Reference & Additional materials

1. Lam, P. C. H., Chu, L., Torgonskiy, M., Pei, J., Zhang, Y., & Wang, L. (2021). Finding representative interpretations on convolutional neural networks. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*.
2. Ghorbani, A., Wexler, J., Zou, J., & Kim, B. (2019). Towards automatic concept-based explanations.
3. Kim, B., Khanna, R., & Koyejo, O. O. (2016). Examples are not enough, learn to criticize! criticism for interpretability. *Advances in neural information processing systems*, *29*.
4. Cho, S., Chang, W., Lee, G., & Choi, J. (2021, August). Interpreting Internal Activation Patterns in Deep Temporal Neural Networks by Finding Prototypes. In *Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining*.
5. [Wikipedia: Submodular set function](https://en.wikipedia.org/wiki/Submodular\_set\_function)
6. Chu, L., Hu, X., Hu, J., Wang, L., & Pei, J. (2018, July). Exact and consistent interpretation for piecewise linear neural networks: A closed form solution. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*.
