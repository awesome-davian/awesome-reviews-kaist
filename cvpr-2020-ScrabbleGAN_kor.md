---
Sharon Fogel / ScrabbleGAN: Semi-Supervised Varying Length Handwritten Text Generation / CVPR2020
---

# ScrabbleGAN: Semi-Supervised Varying Length Handwritten Text Generation\[Kor\]

ScrabbleGAN 논문은 CVPR 2020 논문 중 하나이다. Handwritten Text Generation을 주제로 하고있다. Fully Convolutional Neural Network GAN 구조와 Handwritten Text Recognition(HTR) 모델을 전체 구조로 다양한 스타일로 realistic한 Handwritten Text Generation이 가능한 생성 모델을 제안하였고, 그 결과물들을 활용하여  기존 HTR 모델들의 성능을 향상시켰다. 

##  1. Problem definition

그렇다면, 이 논문에서는 기존에 있던 어떤 문제들을 해결하려고 했을까?

1.  **RNN 구조에서 CNN구조로의 탈피**

​	첫번째는 기존의 Handwritten Text Generation 모델들은 RNN기반의 모델들인데, 본 논문에서는 CNN기반의 모델 구조를 제안하였다. 기존 논문들이 RNN(정확하게는 CRNN, LSTM구조를 쓰는거 같다.)기반의 모델일 수 밖에 없는 이유는 Handwritten Text Generation 모델의 데이터를 보면 이해할 수 있는데, Handwritten Text Generation에서 데이터는 같은 사이즈나 비슷한 사이즈로 묶여있는 이미지 데이터셋과 다르게 글자에 따라 그 다양성이 크다. 따라서 input을 일정하게 resize시키는 방법은 적절치 않고,  output의 길이제약이 없는 many(input) to many(output) 구조를 가질 수 있는 RNN 구조를 사용하는 것이다. 

​	하지만 맨 첫글자는 마지막 글자에 영향을 끼치는냐고 생각하면 아닐 가능성이 크다. 이를 논문에서는 non-trivial하다고 지목한다. 따라서 본 논문은 RNN구조를 사용하는 대신, CNN구조를 제안한다.

![data_sample](/.gitbook/assets/24/data_sample.png)

< Figure 3에 있는 다양한  ScrabbleGAN의 결과, 데이터셋도 이와 비슷하게 다양한 길이와 단어로 이루어져있다. 오른쪽부터,  retrouvailles, ecriture, les, e'toile, feuilles, s'oleil, pe'ripate'ticien and chaussettes >

​	2. **GAN 구조를 이용한 semi-supervised learning**

두 번째는 정확히 레이블된 데이터셋으로만 기존 Handwritten Text Generation task가 이루어 졌다는 것이다. 이렇게 되면 데이터셋에 크게 의존할 수 밖에 없다. 하지만 논문에서는 Generator와 Discrimminator 간 레이블이 필요 없는 GAN구조를 사용함으로써 semi-supervised learning이 가능하게 하여  Handwritten Text Generation분야의 performance를 끌어올리는 방식을 제안한다. 

​	3. **기존 데이터셋의 한계 극복**

마지막으로, 앞서 말한 데이터셋의 한계를  Handwritten Text Generation으로 추가 데이터를 확보하여 문제를 극복하려고 하였다.



이 논문의 주요 Contribution으로는 손 글씨 특성 상 출력의 크기가 일정하지않아 기존  Handwritten Text Generation에 쓰이는 RNN-based모델이 아닌, Fully Convolutional Neural Network를 제안했다는 점,  unlabeled data에 대해 Semi-supervised learning을 시도했다는 점, 그리고 해당 모델을 기존 데이터셋과 추가적으로 구성함으로써 데이터셋의 다양성을 확보해 기존 Handwritten Text Recognition(HTR) 모델의 성능을 올렸다는 점이다. 

## 2. Motivation

**Online과 Offline 방식의 차이**

관련 연구를 살펴보기 전에, Handwritten Text는 Online과 Offline 방식에 차이가 있다는 것을 알아야 소개할 논문의 컨셉들이 이해가 된다. 온라인 방식은 그 과정을 샘플링한 Stroke라는 개념을 통해 손 글씨가 써지는 과정에 대한 정보가 있다. 하지만 오프라인 방식은 그 과정에 대한 정보가 아니라, 최종 결과물만 볼 수 있다. 따라서, Online이냐 Offline이냐는 그 논문의 컨셉에 중요한 영향을 미친다.  예시로, Handwritten Text Generation에서 Online은 sequantial한 순서를 생성하는 것이 될 수도 있지만, Offline에서는 한 장의 이미지를 생성하는 것이 된다.

논문에서는 Stroke를 기록해야하는 도구가 있어야하는 Online 데이터가 수집하기도 힘들고, 오프라인에는 아예 적용할 수 없지만, 반대로 Offline의 방법론은 Online에도 적용 가능한 범용성이 있기 때문에 오프라인 방법론에 대해 초점을 맞췄다고 한다.

### Related work

Please introduce related work of this paper. Here, you need to list up or summarize strength and weakness of each work.

[Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.]

### Idea

After you introduce related work, please illustrate the main idea of the paper. It would be great if you describe the idea by comparing or analyzing the drawbacks of the previous work.

## 3. Method

ㅎㅎ





![arch_superkali](/.gitbook/assets/24/arch_superkali.gif)



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

