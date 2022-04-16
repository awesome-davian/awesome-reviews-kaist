---
description: Jeong et al. / Self-Calibrating Neural Radiance Fields / ICCV 2021
---

# Self-Calibrating Neural Radiance Fields \[Kor]

## Self-Calibrating Neural Radiance Fields \[Kor]

\*\*\*\*[**English version**](iccv-2021-scnerf-eng.md) of this article is available.

## 1. Problem definition

해당 논문에서는 하나의 scene을 촬영한 여러장의 이미지가 입력으로주어졌을 때, 이미지를 촬영할 때 사용된 카메라의 intrinsic/extrinsic parameter와 해당 scene의 geometry를 표현하는 Neural Radiance Field 파라미터를 동시에 학습합니다. 일반적으로 카메라의 intrinsic/extrinsic을 추정할 때는 checker board와 같은 calibration pattern을 촬영한 이미지가 필요하지만 해당 논문에서는 calibration pattern을 촬영한 이미지 없이 calibration을 수행이 가능합니다. &#x20;

&#x20;수식으로는 아래와 같이 표현할 수있습니다.

> Find $$K, R, t, k, r_{o}, r_{d}, \theta$$ when $$(\mathbf{r_o}, \mathbf{r_d})=f_{cam}(K, R, t, k, r_o, r_d)$$, $$\hat{\mathbf{C}}(\mathbf{r})=f_{nerf}(\mathbf{r_o},\mathbf{r_d};\theta)$$

여기서 $$\mathbf{r_o}$$와 $$\mathbf{r_d}$$는 ray의 origin과 direction, $$f_{cam}$$은 카메라 파라미터로부터 ray를 생성해내는 함수, $$(K,R,t,k,r_o,r_d)$$는 카메라 calibration 파라미터, $$\hat{\mathbf{C}}(\mathbf{r})$$는 ray $$\mathbf{r}$$에 대한 color , $$\theta$$는 Neural Radiance Field 파라미터, $$f_{nerf}$$는 ray가 주어졌을 때 $$\theta$$를 이용하여 이미지를 rendering하는 함수를 의미합니다.&#x20;

기존의 방법들은 카메라 파라미터를 알고있다는 가정 하에 scene의 geometry만 학습하거나, scene geometry에 대한 학습 없이 카메라 파라미터만을 학습했다면, 본 논문의 목적은 $$(K,R,t,k,r_o,r_d)$$와 $$\theta$$를 동시에 학습하는 것입니다.    &#x20;

## 2. Motivation

In this section, you need to cover the motivation of the paper including _related work_ and _main idea_ of the paper.

### Related work

#### Camera Self/Auto-Calibration

Camera Self-Calibration은 별도의 calibration object없이 카메라의 파라미터를 추정하는 분야입니다. 일반적인 self-calibration 방법론들은 sparse한 대응점들만을 사용하는 geometric loss만을 사용하거나 epipolar geometry 가정에 의존하기 때문에 scene이 충분히 많은 feauture를 갖지 않는 경우 결과값이 발산합니다.&#x20;

#### Novel View Synthesis



Please introduce related work of this paper. Here, you need to list up or summarize strength and weakness of each work.

### Idea

기존의 방법들과 달리, 해당&#x20;

After you introduce related work, please illustrate the main idea of the paper. It would be great if you describe the idea by comparing or analyzing the drawbacks of the previous work.

## 3. Method



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
