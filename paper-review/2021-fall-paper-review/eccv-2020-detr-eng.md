---
description: Carion et al. / End-to-end Object Detection with Transformers / ECCV 2020
---

# End-to-End Object Detection with Transformers [Eng]

This article will walk you through a detailed explanation of the paper titled "End-to-End Object Detection with Transformers" with some illustrative examples for some of the concepts used in this paper but not explained by the authors of the paper explicitly.

# I- Problem Definition:

To begin with, let's first understand the main goal of this paper. As the title suggests, this paper presents a model that was mainly designed to solve two problems:

- Perform **object detection** tasks on a given set of images, as shown in Fig. 1. And the model is said to be successful in detecting objects, if it can provide two main pieces of information:
    - Bound each object of interest by a bounding box.
    - Predict the Label/class of each bounded object.
- **Remove** **post-processing** **steps that were manually added by the designers of the previous object detection models**. And replace them with neural network blocks. So that the model and its training process do not heavily rely on prior knowledge of the task in hand but rather learn it by itself!

![Figure 1: Performing object detection on an image that includes three objects of interest, namely, a dog, a bicycle and a car.](../../.gitbook/assets/36/Problem_definition.jpeg)

Object detection itself was, and still is, a very challenging task in computer vision. There are dozens of methods that try to approach this problem in different ways. But each method has its own limitations. And what the authors of DETR (**DE**tection with **TR**ansforms) tried to achieve with their model, is to reduce the limitations of the previously published methods as it will be explained in details in the following sections.

# II- Motivation & Idea:

As mentioned before, the work that was published before this paper relied a lot on some manually designed post-processing steps. And one of this paper's goals, is to eliminate all of these post-processing steps; hence, the name "End-to-end object detection".

Let's now discuss what were these post-processing steps and why they were important in the previous models.

### 2.1 Non-maximum Suppression (NMS):

**Brief goal of NMS: To remove duplicate bounding boxes that refer to the same object.**

Researchers, have create many models that find objects in an image, and then find the bounding box that encapsulates these objects. And so these models output a tuple  $$y_i=(c_i, b_i)$$, where  $$c_i$$ is the predicted class for object $$(i)$$ in the image, and $$b_i$$ is the box that bounds that object. 

However, due to the fact that many objects can be found in one image, and that there is no limit to the number of objects, the models were sometimes confused by bounding the same object more than one time! That is because the models are not perfect, and that they are still learning. This can be further illustrated as shown in Fig. 2. In this figure, we can find that the same dog was predict around 5 times with different bounding boxes. This is because the model thinks that each time its finding new object (dog) in the scene. In order to solve this problem, researchers used the famous NMS method. This method is used to "**suppress**" certain features that do not provide maximum score at a given function.

![Figure 2: How NMS is used with object detection tasks](../../.gitbook/assets/36/07_img_0034.png)

So, in order to use the NMS, we need to define a function that takes these bounding boxes as input, and spits out the scores of each bounding box. Without going into deep details with this function, it basically searches for maximum predicted probability in the list of predictions $$y=[y_1, y_2, ..., y_n]$$. The algorithm then assumes that the bounding box that corresponds to this prediction is indeed a true bounding box for an object in the image. So by now it starts investigating the other bounding boxes to find if these bounding boxes are actually covering the same object, which in this case means that they are duplicate for the same object. So what it does, is that it computes a score that is going to be close to 1, if the bounding boxes almost cover the same area, and that will be 0 if the bounding box do not coincide with each other. This score is given by a function that is called **Intersection-over-Union** or **IoU** for short. So, this function will be fed two bounding boxes, and it will compute the area of intersection between the two bounding boxes over the area of the Union of the two bounding boxes, as seen in Fig. 3.

![Figure 3: Visualization of the IoU function](../../.gitbook/assets/36/Untitled.png)

By giving two bounding boxes (bb1, bb2) to IoU function, we can see that if they are identically the same, then their intersection will be the same as their union, which will result in $$IoU(bb_1, bb_2)=1$$. On the other hand, if they didn't intersect at all, then $$IoU(bb_1, bb_2)$$ will be zero. So these two cases are the extreme cases, and for the values inbetween 0 and 1, as the IoU gets closer to 1, it will indicate that the boxes may refer to the same object, and as the IoU's value get's closer to 0, this will give the intuition that these two bounding boxes refers to different objects in the image.

So what the researchers did in models prior to DETR, is to predict a set of bounding boxes along with a set of labels for each bounding box, and then do the following steps:

- Find maximum class prediction score: As an example, consider an image that includes 5 different objects. And now consider that all of them were detected and that the model had the following output:
    
    $$[(dog 92\%, bb_1), (cat 80\%, bb_2), (dog 99\%, bb_3), (dog 97\%, bb_4), (cat 84\%, bb_5)]$$
    
    In this case, this step will choose $$bb_3$$, because it has the highest prediction probability/confidence.
    
- Compute IoU for between the bounding box found in the previous step (in this case $$bb_3$$) and all the other boxes.
- Remove the bounding boxes that will result in IoU lower than a certain threshold, most probably the threshold is set to be 0.7.

By following these steps, researchers managed to remove duplicate predictions in their object detection pipelines. However, as it can be seen, this method is not optimal as it depends on many assumptions that are not usually valid assumptions. Not to mention that it makes the prediction process reliant on the designers prior knowledge of the task in hand, which is also bad, since it is better to learn everything by the model and not encode our knowledge in such a way.

### 2.2 Region Proposals:

Let's go over this without digging deep into the details that do not matter in this context. In object detection tasks, the models usually will have an input image that has a high resolution compared to images used in normal classification tasks. So a problem always arises with this, and this issue is that the image contains gazillions of different possible bounding boxes, so which bounding boxes should the model consider?! Well, this was solved (prior to DETR) by using what is known as region proposals. Region proposals are a set of fixed number (fixed by the designer, and will stay fixed during the training and inference phases. but can change from one model to the other while designing the model) of bounding boxes that are believed to contain objects within them. Most probably these region proposals are found based on heuristics (e.g. finding regions that contains blobs) and they are based on hand made algorithms like selective search for example. 

So the same disadvantages that were discussed in NMS, are also applied here as well. Which is that these region proposals rely heavily on the designer, and that they are hand-designed and not learned by the model.

![Figure 4: Example of region proposals](../../.gitbook/assets/36/region_proposals.jpg)


### 2.3 Related Work:

DETR used the output of previously published papers in areas that include, bipartite matching loss in the set predictions, transformers and of course the object detection. However, this paper worked in a clever way to combine these ideas that in usual cases were not considered to be combined in one model.

- **Bipartite matching loss:** in short, this is a scenario when you have **N** objects on one side and you want to match them with **N,** on the other side. While keeping a cost function minimized. So each possible matching has its own computed loss, and we are searching for the matching that has the minimum loss. And this loss or cost function is to be defined by the user as $$\mathcal{L}_{match}(\hat{y},y)$$. See Fig. 5 as an example. As shown in the example, each object in the left, get to be connected with one and only one object on the right and vice versa. This can be achieved using the Hungarian algorithm.

![Figure 5: An example of bipartite matching while having a constraint of connecting one object with the one and only one object on the other side.](../../.gitbook/assets/36/Capture.png)


- **Transformers:** Transformers are unique and powerful types of layers that can be added to neural network models. They were initially used with language processing tasks as speech recognition, language translation, and some tasks that require the model to predict the next word in a series of words, etc. They are mainly powerful because they can be given a sequence of input of any size and they can also output another sequence with arbitrary size as well. They also have a powerful feature that inspired which mainly what inspired the authors of DETR to use them. This feature is the transformers have what is called attention blocks. These attentions block are capable of finding relationships between different elements of a given set no matter how long it is. Thus they are capable of having long memory like capabilities. Figure 6. shows the initial structure of a transformer block.

![Figure 6: Basic transformer layer, which consists of an encoder and a decoder](../../.gitbook/assets/36/transformer-model-architecture.png)

- **Object Detection:** The object detection is obviously the core of the DETR model, thus this paper was compared to previous object detection architecture. As well as the fact that it addresses other object detection models' weakness points and how it proved that it can handle most of these weakness in a smart way. So, prior to DETR, the state-of-the-art models were mainly R-CNN (region based CNNs), Fast R-CNN and Faster R-CNN, while Faster R-CNN being the top at that time in terms of speed and accuracy. But as mentioned before all of these variants of R-CNN relied on some post-processing steps such as NMS or they relied in intermediate steps where they generate region proposals. Fig. 7, shows the architecture of Faster R-CNN, which was directly compared with DETR in this paper.

![Figure 7: Fast R-CNN architecture.](../../.gitbook/assets/36/Untitled_3.png)

### 2.4 Idea:

The main idea and intuition that the authors had, was that the transformers that were usually used to work for language processing tasks, can actually be of high benefits here, since they have these attention blocks that can relate each element of the sequence with the other elements in that sequence. Because in the case of an image we can replace the sequence of words with a sequence of features or pixels, and by doing so, the transformer will then be able to match pixels of the same object together because they share a common relation, which is that they all belong to the same object. 

On the other hand, the authors had an amazing idea that could solve the duplication problem. Which is to match each of the predicted bounding boxes with the true bounding boxes using the bipartite matching loss. And by doing so the model will now learn how to chose the proposed bounding boxes to not be duplicate of the same object.

# III. Method:

### 3.1 DETR Overall Architecture:

As it can be seen from Fig. 8, the architecture of DETR is quite simple compared to other state-of-the-art models.

![Figure 8: DETR Architecture](../../.gitbook/assets/36/Untitled_4.png)

### 3.2 Backbone:

The backbone in this architecture is simply a CNN that is used mainly to extract features from the image. Any type of CNN will work, but the authors used ResNet, and ResNet-101 as their backbone models as they are really good at extracting features from images. Because this model uses a transform, this transformer need to be fed with a sequence of data; thus, the image features that were output from the CNN model, will then be flattened and concatenated with positional encoding. This positional encoding is used so that the position information of each pixel is not lost and so that it can be taken into consideration while training the transformer. In an intuitive manner, the position is important, because pixels that belong to the same will have a relationship between each other in terms of their position.

### 3.3 Transformer Encoder-Decoder:

The transformer takes the input flattened image and tries to encode the information in it using the encoder block. Which mainly outputs **Keys** and **Values** that are then used with the **Query** in the decoder block. So basically what all of this means, is that the encoders, gives every flattened feature a value and a key to that value. And then we can think of the query as a certain random region in the image that asks the decoder if there is an object in it or not. And through the training process these regions are then optimized so that each query gives a unique and informative information about a region in the image, while ensuring that other queries covers the rest of the regions in the image. This is ensured, because the queries can communicate together through the attention blocks that are inside the transformer. 

After this communication between the the Values, Keys and Queries, the decoder then spits **N** learned positional embeddings. This number of embeddings is fixed as said earlier and is large enough to ensure that all the objects will be detected.

### 3.4 Feed-forward Networks (FFNs):

Each output embedding from the decoder block is then fed to a feed-forward network that consists of 3-layer perceptron with $$d$$ neurons and a ReLU activation function and a linear prejection layer. The output of each feed-forward network will then be the prediction that contains the predicted tuple, which consists of the class and the normalized bounding box. And of course since there will always be **N,** then most of these predictions have to be garbage. Thus, the classes to be predicted are $$C+1$$, where $$C$$ is the number of classes of interest, and the added class represents the "no-object class ($$\emptyset$$)".

### 3.5 Training Loss:

The loss that is used during training is a combination of a negative-likelihood with a bounding box. And the formula for that loss is as follow:

$$\mathcal{L}_{Hungarian}(y,\hat{y})=\sum_{i=1}^{N}{[-log\hat{p}_{\hat{\sigma}(i)}(c_i)+\mathbb{1}_{c_i\neq\emptyset}\mathcal{L}_{box}(b_i, \hat{b}_{\sigma}(i))]}$$

The authors also reduced the weight of the log-likelihood probability of the "no-object" class by a factor of 10 to count for the class imbalance. And unlike previous models, DETR can global reason about the final set of predicted bounding boxes.

# IV. Experiment & Results:

The authors have tested different types of the DETR model on the COCO dataset and compared it with different types of Faster R-CNN models. And the results were as follow:

![ ](../../.gitbook/assets/36/Untitled_5.png)

Which shows that such a simple model, can compete with complex models, but also outperform them in some metrics.

Also the authors tried to give intuition on how exactly the model works, by adding examples that illustrates the power of the transformers use in DETR. And fig. 9, shows that the encoder is capable of separate instances in the image, when it was given a reference point of that instance using the self-attention that is embedded in the the transformer.

![Figure 9: separating instances based on a reference point using the self-attention in the encoder.](../../.gitbook/assets/36/Untitled_6.png)

And the following image shows that the model can detect many objects in one image without having any trouble to do so.

![Figure 10: detecting many objects in the same image](../../.gitbook/assets/36/Untitled_7.png)

And the most astonishing result of them all, is the one that shows how the decoder learns to give importance to the edges of the same object, because these edges are the important pixels that will contribute to creating the bounding box.

![Figure 11: Visualization of the decoder's attention for every predicted object.](../../.gitbook/assets/36/Untitled_8.png)


It is important to notice that the decoder successfully detected the foot of the elephant behind although it's occluded by the small elephant. And the same happens with zebras.  

![Figure 12: box prediction on images for COCO validation set](../../.gitbook/assets/36/Untitled_9.png)


The figure above shows that each of the N inputs in the decoder learn to focus on one region in the image and to be specialized in it. It is similar to asking many people about what do they see in the image. But giving each one of them one region to focus on and specialized with.

At the end, the authors showed that the DETR model can be extended easily to cover other tasks and that this is just a start for that model. They showed this by tweaking the DETR model so that it can be adapted on the panoptic segmentation, as shown below.

![Figure 13: Panoptic segmentation using DETR model](../../.gitbook/assets/36/Untitled_10.png)

And they compared the results with state-of-the-art methods to proof, that it can compete with them just by a simple adjustment.

![](../../.gitbook/assets/36/Untitled_11.png)

# V. Conclusion:

The paper showed a good solution to problems that were encountered by the previous state-of-the-art methods such as the use of post-processing steps that were hand-crafted by the designer and replaced them with powerful tools that can learn autonomously. And it has opened the way for transformers to be explored further more in tasks that different from language processing. Which inspires many researchers around the world to come up with new and creative ideas for using transformers abilities in their favor.

# Author / Reviewer information

## Author

**Ahmed Magd**

- contact: a.magd@kaist.ac.kr
- MS student, Robotics, KAIST
- Research Interest: computer vision with robotics

## reviewer

1. Korean name (English name): Affiliation / Contact information
2. Korean name (English name): Affiliation / Contact information
3. ...

# Reference & Additional materials

1. Carion, Nicolas, et al. "End-to-end object detection with transformers." *European Conference on Computer Vision*. Springer, Cham, 2020.
2. Vaswani, Ashish, et al. "Attention is all you need." *Advances in neural information processing systems*. 2017.
3. Ren, Shaoqing, et al. "Faster r-cnn: Towards real-time object detection with region proposal networks." *Advances in neural information processing systems* 28 (2015): 91-99.
4. Girshick, Ross. "Fast r-cnn." *Proceedings of the IEEE international conference on computer vision*. 2015.
5. Cai, Z., Vasconcelos, N.: Cascade R-CNN: High quality object detection and instance
segmentation. PAMI (2019)
6. Professor Justin Johnson's lectures-12-13-15: [https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r](https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)
7. Analysis on DETR by Yannic Kilcher: [https://www.youtube.com/watch?v=T35ba_VXkMY](https://www.youtube.com/watch?v=T35ba_VXkMY)
