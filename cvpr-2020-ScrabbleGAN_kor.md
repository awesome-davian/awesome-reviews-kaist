---
Sharon Fogel / ScrabbleGAN: Semi-Supervised Varying Length Handwritten Text Generation / CVPR2020
---

# ScrabbleGAN: Semi-Supervised Varying Length Handwritten Text Generation\[Kor\]

논문에 들어가기 전, 지금까지 일반 글씨체는 내용에 대한 설명이고, <u>*이런 기울여지고 밑줄친 글씨체는 작성자의 생각이 담긴 것으로 구분해서 보면 될거같다.*</u> 	

ScrabbleGAN 논문은 CVPR 2020에 나온 논문이다. Handwritten Text Generation을 주제로 하고있다. Fully Convolutional Neural Network GAN 구조와 Handwritten Text Recognition(HTR) 모델을 전체 구조로 다양한 스타일로 realistic한 Handwritten Text Generation이 가능한 생성 모델을 제안하였고, 그 결과물들을 활용하여  기존 HTR 모델들의 성능을 향상시켰다. 리뷰에 들어가기 전 전체적인 동작과 결과를 보여주는 사진을 먼저 보자. 그럼 전체적인 이해에 도움이 될거 같다.

![arch_superkali](/.gitbook/assets/24/arch_superkali.gif)

<ScrabbleGAN 논문의 Official Github에 가보면 단어 "meet" 를 생성하는 과정과 가장 긴 단어라고 알려진 “Supercalifragilisticexpialidocious”의 다양한 스타일을 보여준다. https://github.com/amzn/convolutional-handwriting-gan>



##  1. Problem definition

그렇다면, 이 논문에서는 기존에 있던 어떤 문제들을 해결하려고 했을까?

1.  **RNN 구조에서 CNN구조로의 탈피**

​	첫번째는 기존의 Handwritten Text Generation 모델들은 RNN기반의 모델들인데, 본 논문에서는 CNN기반의 모델 구조를 제안하였다. 기존 논문들이 RNN(정확하게는 CRNN, LSTM구조를 쓰는거 같다.)기반의 모델일 수 밖에 없는 이유는 Handwritten Text Generation 모델의 데이터를 보면 이해할 수 있는데, Handwritten Text Generation에서 데이터는 같은 사이즈나 비슷한 사이즈로 묶여있는 이미지 데이터셋과 다르게 글자에 따라 그 다양성이 크다. 따라서 input을 일정하게 resize시키는 방법은 적절치 않고,  output의 길이제약이 없는 many(input) to many(output) 구조를 가질 수 있는 RNN 구조를 사용하는 것이다. 하지만 맨 첫글자는 마지막 글자에 영향을 끼치는냐고 생각하면 아닐 가능성이 크다. 이를 논문에서는 non-trivial하다고 지목한다. 따라서 본 논문은 RNN구조를 사용하는 대신, CNN구조를 제안한다.

![data_sample](/.gitbook/assets/24/data_sample.png)

< ScrabbleGAN 논문에 Figure 3에 있는 다양한 결과, 데이터셋도 이와 비슷하게 다양한 길이와 단어로 이루어져있다. 오른쪽부터,  retrouvailles, ecriture, les, e'toile, feuilles, s'oleil, pe'ripate'ticien and chaussettes >



​	2. **GAN 구조를 이용한 semi-supervised learning**

​	두 번째는 정확히 레이블된 데이터셋으로만 기존 Handwritten Text Generation task가 이루어 졌다는 것이다. 이렇게 되면 데이터셋에 크게 의존할 수 밖에 없다. 하지만 논문에서는 Generator와 Discrimminator 간 레이블이 필요 없는 GAN구조를 사용함으로써 semi-supervised learning이 가능하게 하여  Handwritten Text Generation분야의 performance를 끌어올리는 방식을 제안한다. 



​	3. **기존 데이터셋의 한계 극복**

​	마지막으로, 앞서 말한 데이터셋의 한계를  Handwritten Text Generation으로 추가 데이터를 확보하여 문제를 극복하려고 하였다. 이 논문의 주요 Contribution으로는 손 글씨 특성 상 출력의 크기가 일정하지않아 기존  Handwritten Text Generation에 쓰이는 RNN-based모델이 아닌, Fully Convolutional Neural Network를 제안했다는 점,  unlabeled data에 대해 Semi-supervised learning을 시도했다는 점, 그리고 해당 모델을 기존 데이터셋과 추가적으로 구성함으로써 데이터셋의 다양성을 확보해 기존 Handwritten Text Recognition(HTR) 모델의 성능을 올렸다는 점이다. 





## 2. Motivation

**Online과 Offline 방식의 차이**

​	관련 연구를 살펴보기 전에, Handwritten Text는 Online과 Offline 방식에 차이가 있다는 것을 알아야 소개할 논문의 컨셉들이 이해가 된다. 온라인 방식은 그 과정을 샘플링한 Stroke라는 개념을 통해 손 글씨가 써지는 과정에 대한 정보가 있다. 하지만 오프라인 방식은 그 과정에 대한 정보가 아니라, 최종 결과물만 볼 수 있다. 따라서, Online이냐 Offline이냐는 그 논문의 컨셉에 중요한 영향을 미친다.  예시로, Handwritten Text Generation에서 Online은 sequantial한 순서를 생성하는 것이 될 수도 있지만, Offline에서는 한 장의 이미지를 생성하는 것이 된다. 논문에서는 Stroke를 기록해야하는 도구가 있어야하는 Online 데이터가 수집하기도 힘들고, 오프라인에는 아예 적용할 수 없지만, 반대로 Offline의 방법론은 Online에도 적용 가능한 범용성이 있기 때문에 오프라인 방법론에 대해 초점을 맞췄다고 한다.

![online_data](/.gitbook/assets/24/online_data.jpg)

<Deepwriting 논문에서 설명한 Online data: 시간에 따라 sampling된 순서가 정해진 stroke라는 개념이 있다.>



### Related work

<u>*이 챕터에서는 관련된 논문으로 소개한 논문 중 중요하다 생각하는 것들을 짧게 요약 및 정리를 해보았다. 사실 이 related work를 다 follow up 했으면, 이번 논문의 컨셉을 단번에 이해할 수 있다.*</u>



**[Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.]**

먼저는 토론토 대학의 Alex Graves가 발표한 Generating sequences with recurrent neural networks이란 논문인데 citation 수가 무려 3500여 회로 굉장히 영향력 있는 논문이다. 이 논문에서는 제목 그대로 RNN을 이용한 sequential한 생성에 대해 이야기한다. 여기서 사용한 데이터는 위에서 언급한 stroke가 포함된 IAM online 손글씨 데이터셋을 사용하여, 글씨를 쓰는 과정에 있어서 다음 지점이 어딘지 LSTM을 통해 예측하고 계속해서 글씨를 만들어 낸다.

![alex_paper_prediction visualization](/.gitbook/assets/24/alex_paper_prediction visualization.jpg)

<Generating sequences with recurrent neural networks 논문에서는 다음과 같이 prediction 과정을 시각화해서 보여줬다. *<u>글씨를 생성하는거로 볼 수도 있지만, 다음 순서를 예측하는 거로밖에 안보이기도 한다.</u>*> 



**[Bo Ji and Tianyi Chen. Generative adversarial network for handwritten text. arXiv preprint arXiv:1907.11845, 2019]**

이 논문은 GAN 구조를 이용한 손글씨 생성을 제안한 논문이다. *<u>글자마다 길이가 다른 손 글씨가 가지고 있는 특징 때문인거 같은데,</u>* 이 논문에서는 sequential한 데이터를 CNN-LSTM방식의 discriminator를 제안하여 LSTM모델을 generator로 CNN-LSTM 구조를 discriminator로 하여 GAN 구조로 손글씨 학습을 시도했다. 이 논문 또한 IAM online 손 글씨 데이터셋을 사용한 것으로 보인다. *<u>GAN구조가 realistic한 이미지를 만드는 것에 어느정도 정평이 나있다고 생각했는데, 손 글씨 생성하는 논문이 2019년에야 제안되었다니 생각보다 늦다고 할 수 있다.</u>*



**[Eloi Alonso, Bastien Moysset, and Ronaldo Messina. Adversarial generation of handwritten text images conditioned on sequences. arXiv preprint arXiv:1903.00277, 2019.]**

​	이 논문은 ScrabbleGAN의 Result 파트에서 중점적으로 비교하는 모델이다. 그 이유는 ScrabbleGAN과 전체적으로 매우 유사한 구조를 가지고있기 때문이다. 바로 위에 언급했던 단순한 GAN구조(generator와 discriminator의 적대적 학습 방식)를 사용하는 것이 아니라,  text recognition을 위한  auxiliary network을 적용시켰다. 또한 online 데이터셋이 아닌 이미지를 생성하는 task로 바라보았다. 

​	하지만 이 논문에서는 명확한 한계점들이 있다. 첫 번째로 일정 길이 이상의 단어를 생성해내지 못한다는 것이다. *<u>이는 밑에 사진 rho부분에서는 글자를 순서대로 입력받는  bidirectional LSTM recurrent layers로 구성하여 단어에 대한 embedding vector를 출력으로 반환한다. 따라서 당연하게도 긴 단어일 수록 정보의 손실이 있을 뿐더러 최종 출력의 크기가 고정된 상태에서 더더욱 그런 문제점이 발생할 여지가 있어서 라고 본다.</u>*

두 번째는, writing style을 잘 표현해 내지 못한점, 이 논문에서는 style을 조절하지 못한점을 언급하기도 한다.

![Eloi_paper_model_structure](/.gitbook/assets/24/Eloi_paper_model_structure.jpg)

<Adversarial generation of handwritten text images conditioned on sequences에서 제안한 Network 구조>



### Idea

After you introduce related work, please illustrate the main idea of the paper. It would be great if you describe the idea by comparing or analyzing the drawbacks of the previous work.

이에 본 논문은 현재까지의 한계점을 극복하고자 다음과 같은 아이디어를 제시한다. 특히, Adversarial generation of handwritten text images conditioned on sequences에서의 한계점을 극복하기 위한 시도가 ScrabbleGAN의 주요 아이디어라고도 볼 수 있다. 

1. bidirectional LSTM 으로 구성되었던 Embedding network를 없애고 filter bank라는 개념의 각 charactor의 embedding vector을 사용하여 각 글자를 독립적으로 생성한다.  
2. 또한 각 charactor간의 interaction을 위해 overlapped receptive field를 적용하여 인접한 글자간 자연스러운 손글씨를 생성하도록 하였고, Discriminator와 Recognizer는 overlapped receptive field를 포함하여 각 글자를 real/fake인지, 인식할 수 있는지 여부를 판단한다.

이 두 가지가 ScrabbleGAN이 글자를 생성하는 방법론이다. Method 파트에서 더 자세히 알아보자.



## 3. Method



![filterbank_overlapped](/.gitbook/assets/24/filterbank_overlapped.jpg)

**====모델 구조====**

​	먼저 generator를 보자, 저자는 RNN이 아닌 CNN 구조를 사용한 이유에 대해 설명한다. RNN구조는 시작부터 현재까지의 state를 모두 사용한다는 점이 글자를 생성하는데 not trivial하다고 하며 좋지 않다고 지적한다. 하지만 CNN구조를 사용함으로써, 오직 양 옆에있는 글자만 연관되어 글자를 생성함으로 이런 문제를 해결했다고 한다. 논문에서 제안한 overlapped receptive field는 글자간 상호작용하고 부드러운 변화를 만든다. 

​	논문에서는 Meet라는 글자를 만들 때를 예시로 든다. 위의 사진에서와 같이 filter bank에 각 해당하는 글자를 넣는다. 그럼 m,e,e 그리고 t 각 4개의 filter bank가 나오는 건데. 여기에 스타일을 나타내는 noise z를 곱해주어 글자를 생성하기 위한 입력을 만든다. 그리고 위에 말한던 것 같이 각 필터뱅크를 입력으로 생성하는 네트워크에서는 양 옆 과 overlapped receptive field를 공유하면서 생성하게된다, 이런 방식은 길이의 제약이 없으며, 전체 글자의 스타일도 일관된다고 말한다. 또한 저자는 한 filterbank는 overlapped receptive field가 있다 하더라도 작은 부분이기 때문에 생성한 글자는 타겟으로하는 글자가 명확히 생성된다. 하지만, overlapped receptive field로써 양 옆 글자가 달라짐에따라 다양성을 확보할수 있다고 말한다.

​	다음으로는 Discriminator를 보자. Discriminator의 역할은 앞서 말했듯 진짜 같은(realistic) 이미지를 만드는 것과 여기서는 스타일을 분간하는 역할도 있다고 한다. 한 필터뱅크에서 나온 (오버랩포함)글자마다 하나씩 넣고 평균을 내는 식으로 작동하기 때문에 최종 출력의 길이 변화에 따른 영향이 없이 학습이 가능하다. 마지막으로 Recognizer는 읽을 수 있는 텍스트를 만드는데 기여한다. Discrimminator를 손글씨 같은 정도를 만든다 치면 다른 일임에 이해하기 쉽다. Recognizer는 오직 라벨이 있는 real sample에서만 학습이 가능하다.

Recognizer도 CNN구조를 사용했는데, 그 이유로는 많은 모델들이 앞뒤 문맥을 볼 수 있는 bidirectional LSTM을 선택했지만, 이 모델은 글씨 자체가 제대로 쓰여있지 않아도 문맥상으로 때려 맞추는 문제가 있다고 지목한다. *<u>자주 쓰는 단어는 세 글자중 가운데가 이상해도 알아보듯이 말이다.</u>* 논문에서는 이 문제를 지목하며 한 글자 글자가 제대로 인식해야하는 Recognizer구조로 convolutional backbone을 사용했다고 한다.



**====학습 디테일====**

다음로 학습에서의 디테일을 살펴보자. 

![total_loss](/.gitbook/assets/24/total_loss.jpg)

<Total loss: *위의 식의 lambda와 밑의 식의 alpha는 같은 기호로 봐야한다.>

학습은 전체적인 구조에서도 알 수 있듯, Recognizer에서 나오는 Loss R과 Discriminator에서 나오는 Loss D로 이루어진다. 논문에서는 두 로스의 밸런스를 맞추기 위해 Gradient of Loss R의 stadard deviation을 Gradient of Loss D에 맞춰준다. 밑에 수식을 보면 이해할 수 있는데, 여기서 위에서도 언급한 Adversarial generation of handwritten text images conditioned on sequences 논문에서와 다르게 평균은 Gradient of Loss D에 맞게 옮겨주지 않는다.  논문에서는 그 이유를 **평균을 이동하면서 gradient 부호가 바뀌는 문제를 방지하고자 했다고 한다. *<u>하지만 이동을 안해서 두 로스간 scale의 평균이 안맞는 생기는 문제도 있을거 같은데, 표준편차만 맞춰줘서 생기는 장점과 단점에 대해서 논문에서 별다른 언급이 없다.</u>* 

![balance_two_losses](/.gitbook/assets/24/balance_two_losses.jpg)

<Gradient R의 표준편차 scaling: *위의 식의 lambda와 밑의 식의 alpha는 같은 기호로 봐야한다.>

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

