---
description: Jeong et al. / Self-Calibrating Neural Radiance Fields / ICCV 2021
---

# Self-Calibrating Neural Radiance Fields \[Eng]

## Self-Calibrating Neural Radiance Fields \[Eng]

한국어로 쓰인 리뷰를 읽으려면 [**여기**](broken-reference/)를 누르세요.

test image

![Figure1](../../.gitbook/assets/2022spring/35/figure1.png)

## 1. Problem definition

Please provide the problem definition in this section.

We recommend you to use the formal definition (mathematical notations).

## 2. Motivation

In this section, you need to cover the motivation of the paper including _related work_ and _main idea_ of the paper.

### Related work

#### Camera Self/Auto-Calibration

Camera Self-Calibration은 별도의 calibration object없이 카메라의 파라미터를 추정하는 분야입니다. 별도의 calibration object가 없어도 카메라 파라미터를 추정할 수 있다는 장점이 있으나, 일반적인 Camera Self-Calibration 방법론들은 sparse한 대응점들만을 사용하는 geometric loss만을 사용하거나 epipolar geometry 가정에 의존하기 때문에 scene이 충분히 많은 feauture를 갖지 않는 경우 결과값이 발산합니다. 또한 더 정확한 scene의 geometry를 알 수록 더 정확한 카메라 모델을 얻을 수 있음에도 불구하고, 일반적인 self-calibration 방법론들은  geometry를 개선하거나 학습하는 과정을 포함하지 않습니다.

#### Novel View Synthesis

This research field synthesizes a novel view of the scene by optimizing a separate neural continuous volume representation network for each scene. This requires only a dataset of captured RGB images of the scene, the corresponding camera poses and intrinsic parameters, and scene bounds

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
