---
description: Fogel et al. / ScrabbleGAN; Semi-Supervised Varying Length Handwritten Text Generation / CVPR 2020
---

# ScrabbleGAN: Semi-Supervised Varying Length Handwritten Text Generation\[Kor\]

논문에 들어가기 전, 지금까지 일반 글씨체는 내용에 대한 설명이고, <u>*이런 기울여지고 밑줄친 글씨체는 작성자의 생각이 담긴 것으로 구분해서 보면 될거같다.*</u> 

ScrabbleGAN 논문은 CVPR 2020에 나온 논문이다. Handwritten Text Generation을 주제로 하고있다. Fully Convolutional Neural Network GAN 구조와 Handwritten Text Recognition(HTR) 모델을 전체 구조로 다양한 스타일로 realistic한 Handwritten Text Generation이 가능한 생성 모델을 제안하였고, 그 결과물들을 활용하여  기존 HTR 모델들의 성능을 향상시켰다. 리뷰에 들어가기 전 전체적인 동작과 결과를 보여주는 사진을 먼저 보자. 그럼 전체적인 이해에 도움이 될거 같다.



![ScrabbleGAN 논문의 Official Github에 가보면 단어 "meet" 를 생성하는 과정과 가장 긴 단어라고 알려진 “Supercalifragilisticexpialidocious”의 다양한 스타일을 보여준다.](/.gitbook/assets/24/arch_superkali.gif)




##  1. Problem definition

그렇다면, 이 논문에서는 기존에 있던 어떤 문제들을 해결하려고 했을까?

​	**1.RNN 구조에서 CNN구조로의 탈피**

​	첫번째는 기존의 Handwritten Text Generation 모델들은 RNN기반의 모델들인데, 본 논문에서는 CNN기반의 모델 구조를 제안하였다. 기존 논문들이 RNN(정확하게는 CRNN, LSTM구조를 쓰는거 같다.)기반의 모델일 수 밖에 없는 이유는 Handwritten Text Generation 모델의 데이터를 보면 이해할 수 있는데, Handwritten Text Generation에서 데이터는 같은 사이즈나 비슷한 사이즈로 묶여있는 이미지 데이터셋과 다르게 글자에 따라 그 다양성이 크다. 따라서 input을 일정하게 resize시키는 방법은 적절치 않다.  

따라서, output의 길이제약이 없는 many(input) to many(output) 구조를 가질 수 있는 RNN 구조를 사용하는 것이다. 하지만 맨 첫글자는 마지막 글자에 영향을 끼치는냐고 생각하면 아닐 가능성이 크다. 이를 논문에서는 non-trivial하다고 지목한다. 따라서 본 논문은 RNN구조를 사용하는 대신, CNN구조를 제안한다.

또한 각 글자간의 연속성과 자연스러움을 표현하기 위해서 overlapped receptive field를 사용한다. 자신의 양 옆의 글자와  receptive field를 공유함으로써, 자신의 앞뒤의 sequential한 information을 RNN이 아닌 CNN에서도 local하게 사용할 수 있도록 디자인 하였다. 

![ ScrabbleGAN 논문에 Figure 3에 있는 다양한 결과, 데이터셋도 이와 비슷하게 다양한 길이와 단어로 이루어져있다. 오른쪽부터,  retrouvailles, ecriture, les, e'toile, feuilles, s'oleil, pe'ripate'ticien and chaussettes ](/.gitbook/assets/24/data_sample.png)



​	**2. GAN 구조를 이용한 semi-supervised learning**

​	두 번째는 정확히 레이블된 데이터셋으로만 기존 Handwritten Text Generation task가 이루어 졌다는 것이다. 이렇게 되면 데이터셋에 크게 의존할 수 밖에 없다. 하지만 논문에서는 Generator와 Discrimminator 간 레이블이 필요 없는 GAN구조를 사용함으로써 semi-supervised learning이 가능하게 하여  Handwritten Text Generation분야의 performance를 끌어올리는 방식을 제안한다. 



​	**3. 기존 데이터셋의 한계 극복**

​	마지막으로, 앞서 말한 데이터셋의 한계를  Handwritten Text Generation으로 추가 데이터를 확보하여 문제를 극복하려고 하였다. 이 논문의 주요 Contribution으로는 손 글씨 특성 상 출력의 크기가 일정하지않아 기존  Handwritten Text Generation에 쓰이는 RNN-based모델이 아닌, Fully Convolutional Neural Network를 제안했다는 점,  unlabeled data에 대해 Semi-supervised learning을 시도했다는 점, 그리고 해당 모델을 기존 데이터셋과 추가적으로 구성함으로써 데이터셋의 다양성을 확보해 기존 Handwritten Text Recognition(HTR) 모델의 성능을 올렸다는 점이다. 





## 2. Motivation

**Online과 Offline 방식의 차이**

​	관련 연구를 살펴보기 전에, Handwritten Text는 Online과 Offline 방식에 차이가 있다는 것을 알아야 소개할 논문의 컨셉들이 이해가 된다. 온라인 방식은 그 과정을 샘플링한 Stroke라는 개념을 통해 손 글씨가 써지는 과정에 대한 정보가 있다. 하지만 오프라인 방식은 그 과정에 대한 정보가 아니라, 최종 결과물만 볼 수 있다. 따라서, Online이냐 Offline이냐는 그 논문의 컨셉에 중요한 영향을 미친다.  예시로, Handwritten Text Generation에서 Online은 sequantial한 순서를 생성하는 것이 될 수도 있지만, Offline에서는 한 장의 이미지를 생성하는 것이 된다. 논문에서는 Stroke를 기록해야하는 도구가 있어야하는 Online 데이터가 수집하기도 힘들고, 오프라인에는 아예 적용할 수 없지만, 반대로 Offline의 방법론은 Online에도 적용 가능한 범용성이 있기 때문에 오프라인 방법론에 대해 초점을 맞췄다고 한다.

![Deepwriting 논문에서 설명한 Online data: 시간에 따라 sampling된 순서가 정해진 stroke라는 개념이 있다.](/.gitbook/assets/24/online_data.jpg)



### Related work

<u>*이 챕터에서는 관련된 논문으로 소개한 논문 중 중요하다 생각하는 것들을 짧게 요약 및 정리를 해보았다. 사실 이 related work를 다 follow up 했으면, 이번 논문의 컨셉을 단번에 이해할 수 있다.*</u>



**[Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.]**

먼저는 토론토 대학의 Alex Graves가 발표한 Generating sequences with recurrent neural networks이란 논문인데 citation 수가 무려 3500여 회로 굉장히 영향력 있는 논문이다. 이 논문에서는 제목 그대로 RNN을 이용한 sequential한 생성에 대해 이야기한다. 여기서 사용한 데이터는 위에서 언급한 stroke가 포함된 IAM online 손글씨 데이터셋을 사용하여, 글씨를 쓰는 과정에 있어서 다음 지점이 어딘지 LSTM을 통해 예측하고 계속해서 글씨를 만들어 낸다.





![Generating sequences with recurrent neural networks 논문에서는 다음과 같이 prediction 과정을 시각화해서 보여줬다. *<u>글씨를 생성하는거로 볼 수도 있지만, 다음 순서를 예측하는 거로밖에 안보이기도 한다.</u>*](/.gitbook/assets/24/alex_paper_prediction_visualization.jpg)



**[Bo Ji and Tianyi Chen. Generative adversarial network for handwritten text. arXiv preprint arXiv:1907.11845, 2019]**

이 논문은 GAN 구조를 이용한 손글씨 생성을 제안한 논문이다. *<u>글자마다 길이가 다른 손 글씨가 가지고 있는 특징 때문인거 같은데,</u>* 이 논문에서는 sequential한 데이터를 CNN-LSTM방식의 discriminator를 제안하여 LSTM모델을 generator로 CNN-LSTM 구조를 discriminator로 하여 GAN 구조로 손글씨 학습을 시도했다. 이 논문 또한 IAM online 손 글씨 데이터셋을 사용한 것으로 보인다. *<u>GAN구조가 realistic한 이미지를 만드는 것에 어느정도 정평이 나있다고 생각했는데, 손 글씨 생성하는 논문이 2019년에야 제안되었다니 생각보다 늦다고 할 수 있다.</u>*



**[Eloi Alonso, Bastien Moysset, and Ronaldo Messina. Adversarial generation of handwritten text images conditioned on sequences. arXiv preprint arXiv:1903.00277, 2019.]**

​	이 논문은 ScrabbleGAN의 Result 파트에서 중점적으로 비교하는 모델이다. 그 이유는 ScrabbleGAN과 전체적으로 매우 유사한 구조를 가지고있기 때문이다. 바로 위에 언급했던 단순한 GAN구조(generator와 discriminator의 적대적 학습 방식)를 사용하는 것이 아니라,  text recognition을 위한  auxiliary network을 적용시켰다. 또한 online 데이터셋이 아닌 이미지를 생성하는 task로 바라보았다. 

​	하지만 이 논문에서는 명확한 한계점들이 있다. 첫 번째로 일정 길이 이상의 단어를 생성해내지 못한다는 것이다. *<u>이는 밑에 사진 rho부분에서는 글자를 순서대로 입력받는  bidirectional LSTM recurrent layers로 구성하여 단어에 대한 embedding vector를 출력으로 반환한다. 따라서 당연하게도 긴 단어일 수록 정보의 손실이 있을 뿐더러 최종 출력의 크기가 고정된 상태에서 더더욱 그런 문제점이 발생할 여지가 있어서 라고 본다.</u>*

두 번째는, writing style을 잘 표현해 내지 못한점, 이 논문에서는 style을 조절하지 못한점을 언급하기도 한다.

![Adversarial generation of handwritten text images conditioned on sequences에서 제안한 Network 구조](/.gitbook/assets/24/Eloi_paper_model_structure.jpg)



### Idea

After you introduce related work, please illustrate the main idea of the paper. It would be great if you describe the idea by comparing or analyzing the drawbacks of the previous work.

이에 본 논문은 현재까지의 한계점을 극복하고자 다음과 같은 아이디어를 제시한다. 특히, Adversarial generation of handwritten text images conditioned on sequences 논문에서 한계점을 극복하기 위한 시도가 ScrabbleGAN의 주요 아이디어라고도 볼 수 있다. 

1. bidirectional LSTM 으로 구성되었던 Embedding network를 없애고 filter bank라는 개념의 각 charactor의 embedding vector을 사용하여 각 글자를 독립적으로 생성한다.  
2. 또한 각 charactor간의 interaction을 위해 overlapped receptive field를 적용하여 인접한 글자간 자연스러운 손글씨를 생성하도록 하였고, Discriminator와 Recognizer는 overlapped receptive field를 포함하여 각 글자를 real/fake인지, 인식할 수 있는지 여부를 판단한다.

이 두 가지가 ScrabbleGAN이 글자를 생성하는 방법론이다. Method 파트에서 더 자세히 알아보자.



## 3. Method



![Generator Part](/.gitbook/assets/24/filterbank_overlapped.jpg)

* 모델 구조

​	먼저 generator를 보자, 저자는 RNN이 아닌 CNN 구조를 사용한 이유에 대해 설명한다. RNN구조는 시작부터 현재까지의 state를 모두 사용한다는 점이 글자를 생성하는데 non-trivial 하다고 하며 좋지 않다고 지적한다. 하지만 CNN구조를 사용함으로써, 오직 양 옆에있는 글자만 연관되어 글자를 생성함으로 이런 문제를 해결했다고 한다. 논문에서 제안한 overlapped receptive field는 글자간 상호작용하고 부드러운 변화를 만든다. 

​	논문에서는 Meet라는 글자를 만들 때를 예시로 든다. 위의 사진에서와 같이 filter bank에 각 해당하는 글자를 넣는다. 그럼 m,e,e 그리고 t 각 4개의 filter bank가 나오는 건데. 여기에 스타일을 나타내는 noise z를 곱해주어 글자를 생성하기 위한 입력을 만든다. 그리고 위에 말한던 것 같이 각 필터뱅크를 입력으로 생성하는 네트워크에서는 양 옆 과 overlapped receptive field를 공유하면서 생성하게된다, 이런 방식은 길이의 제약이 없으며, 전체 글자의 스타일도 일관된다고 말한다. 또한 저자는 한 filterbank는 overlapped receptive field가 있다 하더라도 작은 부분이기 때문에 생성한 글자는 타겟으로하는 글자가 명확히 생성된다. 하지만, overlapped receptive field로써 양 옆 글자가 달라짐에따라 다양성을 확보할수 있다고 말한다.

![Discriminator와 Recognizer Part 두 network에서의 Loss를 통해 전체 네트워크가 학습한다.](/.gitbook/assets/24/DandR.jpg)

​	다음으로는 Discriminator를 보자. Discriminator의 역할은 앞서 말했듯 진짜 같은(realistic) 이미지를 만드는 것과 여기서는 스타일을 분간하는 역할도 있다고 한다. 한 필터뱅크에서 나온 (오버랩포함)글자마다 하나씩 넣고 평균을 내는 식으로 작동하기 때문에 최종 출력의 길이 변화에 따른 영향이 없이 학습이 가능하다. 마지막으로 Recognizer는 읽을 수 있는 텍스트를 만드는데 기여한다. Discrimminator를 손글씨 같은 정도를 만든다 치면 다른 일임에 이해하기 쉽다. Recognizer는 오직 라벨이 있는 real sample에서만 학습이 가능하다.

Handwritten Text Recognition(HTR)network인 Recognizer도 CNN구조를 사용했다. 그 이유로는 많은 모델들이 앞뒤 문맥을 볼 수 있는 bidirectional LSTM을 선택했지만, 이 모델은 글씨 자체가 제대로 쓰여있지 않아도 문맥상으로 때려 맞추는 문제가 있다고 지목한다. *<u>자주 쓰는 단어는 세 글자중 가운데가 이상해도 알아보듯이 말이다.</u>* 논문에서는 이 문제를 지목하며 한 글자 글자가 제대로 인식해야하는 Recognizer구조로 convolutional backbone을 사용했다고 한다.

*여기서 Handwritten Text Recognition 분야는 말그대로 손글씨를 인식하는 분야이다. Discriminator와 역할이 혼동이 될 수도 있는데, Discriminator는 해당 이미지가 글씨같이 생겼냐 안생겼냐를 판단하는 것이지 이게 무슨 글자, 알파벳인가를 구분하지 않는다. 정확한 예시는 아니지만, 굳이 예시를 들자면 Discrimminator는 사람이 손으로 쓴거 같냐(realistic)하냐 이고, Recognizer는 쓰인 글씨가 label과 일치하냐 만약 "meet"라고 쓴거면 "m", "e", "e" 그리고 "t"라고 읽히냐를 판단한다. 

* Loss Function

다음으로 학습에서의 디테일을 살펴보자. 

![Total loss: *lambda와 밑의 식의 alpha는 같은 기호로 봐야한다.](/.gitbook/assets/24/total_loss.jpg)

학습은 전체적인 구조에서도 알 수 있듯, Recognizer에서 나오는 Loss R과 Discriminator에서 나오는 Loss D로 이루어진다. 논문에서는 두 로스의 밸런스를 맞추기 위해 Gradient of Loss R의 stadard deviation을 Gradient of Loss D에 맞춰준다.  lambda의 역할이 loss_D와 loss_R간의 스케일을 조절하는 역할이라고 볼 수 있다. 밑에 수식에서는 alpha로 표현되었다.

밑에 수식을 보면 좀더 자세히 기술이 되어있다. Recognizer에서 나오는 gradient R은 gradient D의 표준편차와 맞춰주고, 그다음 상수 alpha를 곱해 위의 lambda와 같이 스케일을 조절하여 두 loss_D 와 Loss_R가 적절히 학습되게 한다. 

여기서 위에서도 언급한 Adversarial generation of handwritten text images conditioned on sequences 논문에서와 다르게 평균은 Gradient of Loss D에 맞게 옮겨주지 않는다.  논문에서는 그 이유를 **평균을 이동하면서 gradient 부호가 바뀌는 문제를 방지하고자 했다고 한다. *<u>하지만 이동을 안해서 두 로스간 scale의 평균이 안맞는 생기는 문제도 있을거 같은데, 표준편차만 맞춰줘서 생기는 장점과 단점에 대해서 논문에서 별다른 언급이 없다.</u>* 

![Gradient R의 표준편차 scaling: *위의 식의 lambda와  alpha는 같은 기호로 봐야한다.](/.gitbook/assets/24/balance_two_losses.jpg)



## 4. Experiment & Result

### Experimental setup

* Dataset and Evaluation metric

  ​	데이터셋으로는 RIMES, IAM 그리고 CVL이라는 데이터셋을 사용했다. Evaluation Metirc은 두 가지를 사용했다. 첫번째로는 word error rate(WER)이다. 말그대로 전체 단어중에 몇 개의 단어가 잘못 읽혔냐를 평가한다. 두번째는 normalized edit-distance(NED)인데, true와 prediction사이에 edit-distance를 측정한다고 한다. 

  ![word error rate(WER)의 수식, 예시로, A 단어가 B단어가 되기위해 수행해야하는 치환, 삭제 등 여러가지 요소를 수치화하여 계산한다.](/.gitbook/assets/24/WER.jpg)

  ![normalized edit-distance(NED)의 수식. 이때 A_i 와 B_i는 각 글자의 position 이다.예를들어 abc와 acb면 a-a, b-c, c-d 순으로 비교한다.](/.gitbook/assets/24/NED.png)
  

* Training setup

  ​	먼저 논문에서는 한 글자의 생성하는 이미지를 높이 32로 고정하였고 넓이는 16 픽셀로 고정했다. 입력으로 들어가는 Filter bank의 크기는 32x8192인데 여기에 32dim-noise z 를 곱한다. 그럼 n개의 글자를 생성할 때 n x 8192가 된다고 하는데,  n 개의 Filterbank*z((1x32) * (32x8192))을 n개 concat한거라고 이해하면 된다.

  ​	그 다음, reshape을 통해 512x4x4n (8192 = 512x4x4)가 되고, 이때 각 글자는 4x4 spatial size를 가지고 있다고 한다. 그 다음 3개의 residual blocks을 통과한 후에 Up-Sampling 후, 겹쳐진 영역을 만들어서 최종 32x16n사이즈의 이미지를 만든다. 

  ​	Discriminator 구조는 BigGAN 모델에서 차용했는데 4개의 residual blocks로 구성되고 마지막에 fc레이어가 하나 있는 구조이다. 앞서 이야기 한대로 Fully Conv Layers로 구성되어있고, 각 패치(글자)의 평균이 최종 prediction이 된다. *<u></u>*


### Result

* **Comparison to Alonso el al.**

  ​	Adversarial generation of handwritten text images conditioned on sequences에서 제안한 Network와 비교한다 밑에 표와 사진에서는 "Alonso et al. [2]"라교 표기된 논문이다. 먼저 밑의 사진을 먼저 보면, ScrabbleGAN에서 이전 모델이 잘 만들어내지 못한 글씨들도 잘 만들고 있음을 정성적으로 확인 할 수 있다. 또한 그 아래 표를 보면, Fre'chet Inception Distance (FID)와 geometric-score (GS) 스코어로 두 모델의 성능을 비교하였다. 이 표를 통해 ScrabbleGAN 정량적으로도 좋은 이미지를 만들고 있음을 보인다.

![Comparison_with_[2]](/.gitbook/assets/24/Comparison_with_[2].png)



![Comparison_with_[2]2](/.gitbook/assets/24/Comparison_with_[2]2.png)



* **Generating different styles**

  ​	다음으로는 다양한 스타일에 대한 생성이 가능함을 보여준다. 아래 이미지 같이 같은 단어를 다양한 스타일로 생성함을 보여줌으로 다른 각 다른 스타일의 글자가 잘 생성됨을 보였다. 또한, 각 글자마다 같은 스타일 vector z가 따로 곱해졌고, overlapped receptive field로 인해 인접한 글자마다의 interaction도 잘 되어 스타일이 유지되면서 자연스러운 글씨가 생성되었다고 말한다.

![enerating different styles](/.gitbook/assets/24/style_generation.png)



* **Boosting HTR performance**

  ​	다음으로는 제안한 네트워크로 생성한 dataset을 추가로 적용하여 기존의 HTR performance를 늘린 부분에 대해서 말한다. 예상할 수 있듯, 본 논문에서 제안한 방식으로 데이터셋을 추가로 구축한 결과가 더 나음을 설명한다. 표에 따르면 기존데이터를 augmentation한 데이터 셋보다 ScrabbleGAN에서 생성한 이미지를 이용해 학습한 것이 더 좋은 성능을 보인다. ScrabbleGAN의 결과들이 데이터의 다양성을 확보하는데 도움을 준다는 것이다.

![Boosting HTR performance](/.gitbook/assets/24/BoostHTR.png)



## 5. Conclusion

​	본 논문에서는 RNN구조로 전체의 글자생성을 통으로 하나를 보는 것이 아니라 잘라서 local problem으로 만들었다고 한다. 이런 점으로 길이와 스타일에 제약받지 않은 이미지를 잘 생성할 수 있고 오버랩된 receptive field로 인접한 글자간 자연스러움을 더했다고 말한다. 

​	향후 연구 방향으로는 few shot learning으로의 방향성, style과 글씨체(굵기, 날림정도) controllable, 그리고 마지막으로는 각 글자마다 다른 receptive field를 적용시키는 방법을 제안한다. *<u>나도 읽으면서 생각한 한계점인데 같은 글자에 스타일은 달라도 글씨의 한 글자에 해당한 길이가 일정해서 그런 측면에서 다양성이 없다는 것인데 저자도 이점을 지목했다.</u>*

*<u>My opinion</u>*: 본 논문은 기존 RNN구조를 CNN구조로 바꿨다는 것에 큰 contribution이 있다. 전체 글씨를 생성하는 process를 한 글자 기준 양 옆의 글자를 생성하는 문제로 divide and conquer한 것이다. 그 성능이 기존 RNN을 사용 한 것보다 좋은 것을 보이며, 양 옆만 참고해서 글자를 만들어 내는 것이 근거 있는 가정이라는 것을 보였다.  

하지만 논문에서도 말했듯 같은 n개의 글자가 들어간 단어는 i가 100개든 m이 100개든 같은 길이를 가진다는 명확한 한계점이 있다.  또한 다양한 스타일의 결과는 보여줬지만 controllable한 모습은 보여주지 못했다.

### Take home message \(오늘의 교훈\)

> RNN으로 풀어온 문제도 문제 정의만 잘 하면 CNN으로 풀 수 있는 문제도 있다. 
>
> Text Generation 분야는 Recognizable과 Realistic이라는 target을 가진  이미지 Generation과는 또 다른 느낌의 흥미로운 분야인거 같다.

## Author / Reviewer information

### Author

**김기훈(GiHoon Kim)** 

* KAIST GSCT, Visual Media Lab
* gihoon@kaist.ac.kr

### Reviewer
1. 권다희 \(Kwon Dahee\): KAIST / -
2. 백정엽 \(Baek Jeongyeop\): KAIST/ -
3. 한정민 (Han Jungmin): KAIST/-

## Reference & Additional materials

1. Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.
3. Bo Ji and Tianyi Chen. Generative adversarial network for handwritten text. arXiv preprint arXiv:1907.11845, 2019
4. Eloi Alonso, Bastien Moysset, and Ronaldo Messina. Adversarial generation of handwritten text images conditioned on sequences. arXiv preprint arXiv:1903.00277, 2019.
5. Emre Aksan, Fabrizio Pece, and Otmar Hilliges. Deepwriting: Making digital ink editable via deep generative modeling. In Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems, pages 1–14, 2018. 
5. Official GitHub repository:  https://github.com/amzn/convolutional-handwriting-gan
6. Author's Video: https://www.youtube.com/watch?v=jGG5Q8S1Rus
