---
description: Jeong et al. / Self-Calibrating Neural Radiance Fields / ICCV 2021
---

# Self-Calibrating Neural Radiance Fields \[Kor]

## Self-Calibrating Neural Radiance Fields \[Kor]

\*\*\*\*[**English version**](iccv-2021-scnerf-eng.md) of this article is available.

## 1. Problem definition

&#x20;해당 논문에서는 하나의 scene을 촬영한 여러장의 이미지가 입력으로주어졌을 때, 이미지를 촬영할 때 사용된 카메라의 intrinsic/extrinsic parameter와 Neural Radiance Field 형태로 표현된 해당 scene의 geometry를 동시에 학습합니다. 일반적으로 카메라의 intrinsic/extrinsic을 추정할 때는 checker board와 같은 calibration pattern을 촬영한 이미지가 필요하지만 해당 논문에서는 calibration pattern없이도calibration이 가능합니다.

&#x20;수식으로는 아래와 같이 표현할 수있습니다.

$$
\mathbf{r}=f_{cam}(K,R,t,k,r_o,r_d)
$$

$$
\mathbf{C} = f_{nerf}(\mathbf{r};\theta)
$$

여기서 $$\mathbf{r}$$은 ray, $$\mathbf{C}$$는 rendered image, $$f_{cam}$$은카메라   &#x20;

Please provide the problem definition in this section.

We recommend you to use the formal definition (mathematical notations).

## 2. Motivation

In this section, you need to cover the motivation of the paper including _related work_ and _main idea_ of the paper.

### Related work

Please introduce related work of this paper. Here, you need to list up or summarize strength and weakness of each work.

### Idea

After you introduce related work, please illustrate the main idea of the paper. It would be great if you describe the idea by comparing or analyzing the drawbacks of the previous work.

## 3. Method

{% hint style="info" %}
If you are writing **Author's note**, please share your know-how (e.g., implementation details)
{% endhint %}

The proposed method of the paper will be depicted in this section.

Please note that you can attach image files (see Figure 1).\
When you upload image files, please read [How to contribute?](broken-reference/) section.

![Figure 1: You can freely upload images in the manuscript.](broken-reference)

We strongly recommend you to provide us a working example that describes how the proposed method works.\
Watch the professor's [lecture videos](https://www.youtube.com/playlist?list=PLODUp92zx-j8z76RaVka54d3cjTx00q2N) and see how the professor explains.

## 4. Experiment & Result

{% hint style="info" %}
If you are writing **Author's note**, please share your know-how (e.g., implementation details)
{% endhint %}

This section should cover experimental setup and results.\
Please focus on how the authors of paper demonstrated the superiority / effectiveness of the proposed method.

Note that you can attach tables and images, but you don't need to deliver all materials included in the original paper.

### Experimental setup

This section should contain:

* Dataset
* Baselines
* Training setup
* Evaluation metric
* ...
