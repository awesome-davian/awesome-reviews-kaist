---
description: (Description) Juan _et al_. / Multiple instance active learning for object detection / CVPR 2021
---

# Multiple Instance Active Learning for Object Detection \[Eng\]
Juan _et al_. / Multiple instance active learning for object detection / CVPR 2021

한국어로 쓰인 리뷰를 읽으려면 **여기**를 누르세요.
## 1. **Problem Definition**
---
As given away by the title of the paper, the authors here are trying to use **Active Learning** for **Object Detection** on the **instance** level. So, let's clear out the definitions one-by-one.

### **Active Learning**

If you look up the term Active Learning on the internet, you would find several definitions, which are divided into two main categories: Human Active Learning and AI Active Learning. The former is basically a learning method where students are given more responsibility for their own learning journeys. They are encouraged to discuss with other students and dig deeper to find the solutions for the problems they stumble upon along the way [\[1\]][humanactive1]. For example, here at KAIST, for each courses, instead of having a sylabus filled with only lectures, we are given time to work on projects, individual case studies that would actually help us fill our own knowledge gaps. So, the point of Active Learning is that the learners have to be actively seeking what to learn.  

In the context of Artifical Intelligent, the above-mentioned definition still holds. Here, instead of making the models learn all the available data samples, some of which may be not very useful, we can let the models themselves decide what to learn. This would actually save a lot of resources \(e.g., time and computation units\), while better improve the performances of our models. 

But how is that possible for the machine to know the knowledge it lacks, before actually having acquired that knowledge? It turns out that there is a way. We need to measure how uncertain the machine is about its predictions. And we will cover that in the next section.

### **Multiple Instance Learning**

In Object Detection, our input is usually a picture or a video frame, in which there could appear multiple objects of various categories including human, animal and vehicle. The job for our object detection model is to locate and classify the objects by drawing bounding boxes around them and give them the correct label. There have been a huge number of attempts to solve this problem, but they usually fall into either one of the two categories: one-stage detector and two-stage detector. Both of their strengths and weaknesses, but in the context of the paper, and thus this article, we will only discuss the latter.

<figure>
    <center>
        <img
            src=".gitbook/assets/object-detection.png"
        </img>
        </center>
  <center>
    <figcaption>Figure 1: A cat-dog classifier</figcaption>
</center>
</figure>

For two-stage detectors, such as RetinaNet [retinanet], the first stage would be generating, or proposing, a number of prospective candidate locations where the bounding boxes are located. These candidates are called anchor boxes, or in the context this paper, ***instances***. Since the task is to locate objects, we label the ones that contain only the background ***negative instances*** and the other that contain a part or the whole object ***positive instances***, where we could learn something useful about the objects. And a group of instances is called a ***bag***. In this paper, the authors refer to each image as a instance bag.

<figure>
    <center>
        <img
            src=".gitbook/assets/instance-bag.png"
        </img>
        </center>
  <center>
    <figcaption>Figure 1: A cat-dog classifier</figcaption>
</center>
</figure>

Among the instances, there are informative ones \(colored red\) that would benefit our model the most. Just as a human learner would learn the most from the subjects they do not know, these informative instaces are the ones our model must uncertain about. And the goal of this whole paper is ***to find the most informative bags of instances***.

### **Formal Definition**

By now, we have familiarized ourselves with the main concepts that would appear in the paper, it's time we gave the problem some formal definition.

In Machine Learning, there is simply nothing greater than well-labeled data. However, labeling data is no easy task and could take a mountain of human efforts. Therefore, being able to train models effectively on unlabelled data is the next best thing. In this paper, we have a small set of labeled data, denoted $(\mathcal{X}^0_L, \mathcal{Y}^0_L)$
and much larger set of unlabeled image data, denoted $(\mathcal{X}^0_U)$. Each set $\mathcal{X}^0_L$ and $\mathcal{X}^0_U$ contains a number of images. Each image $X \in \mathcal{X}^0_L$ or $X \in \mathcal{X}^0_U$ is represented as a bag of instances $X = \{x_i, i = 1,...,N\}$, where $N$ is the number of instances. The image label of the label set $\mathcal{Y}^0_L$ consists of the coordinates of the bounding boxes $y^{loc}_x$ and the categories $y^{cls}_x$.

In this paper, the model $M$ is first trained on the labeled data $(\mathcal{X}^0_L, \mathcal{Y}^0_L)$, and then retrained on the unlabeled set to:
* Label each image $X_U \in \mathcal{X}^0_U$ with:
  $$y^{pseudo}_c \in \{0, 1\}$$
  Where:
  * $c$ is one of the categories.
  * Label 1 means this image is informative.
  * Label 0 signal otherwise.
*  Select $k$ images from the unlabeled set to incorperate into $\mathcal{X}^0_L$ to form the new labeled set.

## **2. Motivation**
---

### **Uncertainty**

Earlier, we have come to the minor conclusion that we need to measure how uncertain a prediction is. 
But this cannot be simply done by measuring the output probabilities of, say, a logistic function, because those probilities will always sum to 1.

<figure>
    <center>
        <img
            src=".gitbook/assets/catdog.png"
        </img>
        </center>
  <center>
    <figcaption>Figure 1: A cat-dog classifier</figcaption>
</center>
</figure>

For example, if we input a picture of a cat and a dog [\[2\]][mitlecture] into a model that has been trained with cat and dog pictures, we will get the 0.5 and 0.5. It will be hard for the model to decide either that is a cat or a dog but it will decide and be confidient about it.

<figure>
    <center>
        <img
            src=".gitbook/assets/boat.png"
        </img>
        </center>
  <center>
    <figcaption>Figure 1: A cat-dog classifier</figcaption>
</center>
</figure>

Let's use another 

## 3. Method

## 4. Experiment and Result

### Experimental Setup

### Result

## 5. Conclusion

### Take home message

## Reviewer/Reviewer Information

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

[\[1\]][humanactive1] What is Active Learning? (n.d.). Retrieved October 23, 2021, from https://www.queensu.ca/teachingandlearning/modules/active/04_what_is_active_learning.html

[\[2\]][mitlecture] 
Amini, A. (n.d.). MIT 6.S191: Evidential Deep Learning and Uncertainty. Retrieved October 23, 2021, from https://www.youtube.com/watch?v=toTcf7tZK8c&t=2061s

[humanactive1]: https://www.queensu.ca/teachingandlearning/modules/active/04_what_is_active_learning.html#:~:text=Active%20learning%20is%20an%20approach,role%20plays%20and%20other%20methods.
[mitlecture]: https://www.youtube.com/watch?v=toTcf7tZK8c&t=2061s
