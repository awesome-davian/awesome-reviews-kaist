---
Haoming Lu et al./ Deep Learning for 3D Point Cloud Understanding: A Survey 
---

#  Deep Learning for 3D Point Cloud Understanding: A Survey \[Eng\]


##  1. Problem definition

While deep learning is considered as a powerful AI technique and has shown outstanding performance while solving 2D vision tasks, some practical areas such as autonomous driving have not only relied on the processing of 2D images but also on the aquisition of 3D data that provides an opportunity to better understand the environment and thus make decisions that are more accurate and reliable. However, the analysis and processing of 3D point clouds is a challenging task due to irregularities, high dimensionality and disordered data structure. 
This paper presents the recent progress of deep learning techniques for understanding 3D points clouds and covers the latest developments and contributions in different major tasks such as 3D shape classification and segmentation, 3D shape object detection and tracking and 3D point cloud matching and registration. It also provides an introduction to well-established datasets in this area, different metrics and state-of-the-art results.
**INSERT POINT CLOUD PICTURE**

## 2. Motivation

The acquisition and processing of high-resolution 3D data, usually produced by 3D sensors such as laser scanners, LIDARs or RGB-D cameras, has become an important research topic in different fiels such as autonomous driving and mobile robotics. Those devices scan the surfaces of the visible environment in a grid of individual points. The result is a point cloud consisting of many millions of individual coordinates. Many individual scans are then stacked in the correct position to form the overall point cloud which gives a precise 3D representation of the surrounding environment or the scanned objects. 
Due to increasing attention from academia and industry and the rapid development of 3D scanning devices, the processing of point cloud using deep learning has become a popular topic in 3D understanding and despite the complexity of the task, promising results have been obtained on many different point cloud tasks. This motivated the authors to gather all these findings, to provide comprehensive comparisons of different state-of-the-art models and datasets, and to inspire and support future research in 3D understanding.

### Related work

Please introduce related work of this paper. Here, you need to list up or summarize strength and weakness of each work.

Since it is a survey that discusses several categories in the point cloud processing field, the related works as well as the experiments of each category will be included in the corresponding section. 

On the other hand, another similar survey has been conducted in 2020 by Yulan Guo et al. and according to the authors it was the first survey paper to cover deep learning methods for different point cloud understanding tasks. If we consider the difference between the two surveys, they normally follow the same approach except that the one that I am currently reviewing provides more recent findings and focuses particularly on the widely used models and techniques.

**ADD CONCRETE DIFFERENCES BETWEEN THE TWO PAPERS**

### Idea

To approach the field of point cloud processing with deep learning and cover each of its components and to structure their findings, the authors start by introducing the existing datasets and corresponding metrics. Then, they divided the recent works on point clouds in five diffent categories and provided detailed descriptions on each of them:

1. Classification
2. Segmentation
3. Detection, tracking and flow estimation
4. Registration and matching
5. Augmentation and Completion

As stated below, the authors mention the experiments and results of the different methods in each of these categories and provide in some cases different comparisons to other techniques.


## 3. Method

## Datasets

To allow training models more accurately and provide a convictive comparison among different algorithms, availability of different labeled datasets is very important. 

The following table shows some of the most commonly used 3D point cloud datasets. Most datasets contains not only information from LIDAR but also provide corresponding images which make them suitable for different tasks such as classification, segmentation and detection.

![Table 1: Communly used 3D point cloud datasets in recent works.](../../.gitbook/assets/2022spring/9/datasets.png)

To reduce the gap between research and industrys, several other large-scale datatasets have been developed especially with the emergence of the autonomous driving field. 

## Metrics

To test the different available methods for various point cloud understanding tasks and provide a valid evaluation, different metrics are required depending on the task.
## 4. Experiment & Result



This section should cover experimental setup and results.  
Please focus on how the authors of paper demonstrated the superiority / effectiveness of the proposed method.

Note that you can attach tables and images, but you don't need to deliver all materials included in the original paper.

### Experimental setup

This section should contain:

* Dataset
* Baselines
* Training setup
* Evaluation metric
* ...

### Result

Please summarize and interpret the experimental result in this subsection.

## 5. Conclusion

In conclusion, please sum up this article.  
You can summarize the contribution of the paper, list-up strength and limitation, or freely tell your opinion about the paper.

### Take home message \(오늘의 교훈\)

Please provide one-line \(or 2~3 lines\) message, which we can learn from this paper.

> All men are mortal.
>
> Socrates is a man.
>
> Therefore, Socrates is mortal.

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

1. Citation of this paper
2. Official \(unofficial\) GitHub repository
3. Citation of related work
4. Other useful materials
5. ...

