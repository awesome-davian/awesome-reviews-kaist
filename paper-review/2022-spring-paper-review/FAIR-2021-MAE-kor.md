---
description: Kaiming He, Xinlei Chen / Masked Autoencoders Are Scalable Vision Learners / Facebook AI Research(FAIR) 2021
---

# Masked AutoEncoder(MAE) \[Kor\]


Kaiming He, Xinlei Chen / Masked Autoencoders Are Scalable Vision Learners / Facebook AI Research(FAIR) 2021

##  1. Problem definition

컴퓨터 비전 분야에서 label이 있는 수만장의 데이터를 얻기란 어려운 일이다. 그런데 하드웨어 등의 발전으로 큰 모델을 학습할 수 있게 되면서 **self-supervised learning**을 통해 이미지를 라벨 없이 판별하는 것에 대한 관심이 높아졌다. 이런 self-supervised learning이 NLP 분야에서는 활발히 연구되고 있다. 그 중에서 가장 유명한 GPT[1]와 BERT[2]는 데이터의 일부를 지우고 그것을 예측하는 방식으로 이루어졌다. 이 논문에서는 NLP에만 적용되던 이런 masked modeling 방식을 컴퓨터비전에도 적용하고자 하였다.

## 2. Motivation

### Related work

#### Masked language modeling

NLP 분야에 있어서 BERT[2]와 GPT[1]는 masked lanugage modeling을 사용하는 대표적인 model이다. 이들은 input sequence의 일부를 제거하고 그 없어진 부분을 예측하는 방식으로 pre-training이 이루어진다. 이 pretraining된 것을 downstream task에 적용하여 좋은 결과가 나오는 것을 확인할 수 있었다.

#### Autoencoding

Autoencoding[3]은 learning representations의 대표적인 방법이다. 이것은 input을 latent representation에 mapping하는 encoder와 input을 복원하는 decoder로 이루어져 있다. Denosing autodencoders(DAE)[4]는 input signal을 붕괴시키고 original signal로 복원하는 autodencoder이다. 이 논문의 MAE는 DAE와는 다른 방식을 취하고 있다.

#### Masked image encoding

이 방식은 image가 masking에 의해 붕괴되었을 때 이에 대한 representation을 배우는 것이다. DAE에서는 masking을 noise type으로 보여졌다. Context Encoder에서는 CNN을 통해 사라진 부분을 찾고자 하였다. 최근에 NLP 분야에서는 Transformers[5]를 기반으로 encoding한 것에서 착안하여, iGPT[6]는 unknown pixel을 transformer로 예측하고자 하였다. 더 최근에는 BEiT는 discrete tokens을 예측하는 것을 제시하였다.

#### Self-supervised learning

이 방식은 최근에 컴퓨터비전에서 많이 연구되고 있으며 pre-training에 대해 다른 pretext tasks들을 연구하고 있다. 그 중에는 image의 유사성과 비유사성을 학습하는 contrastive learning[7],[8],[9]가 있다. 이는 data augmentation에 의존하고 있다.

### Idea

이 논문의 MAE는 masked된 input image를 encoder를 통해 latent representation에 mapping하고 decoder를 통해 원래의 신호로 복원하고자 한다. NLP에만 사용된 masked autoencoding을 vision에도 사용하기 위해 language와 vision의 질문들을 해결하였다.

1. vision 분야에서는 CNN이 널리 사용되고 있다. 그러나 이것은 masked tokens과 positional embedding을 사용하는 데 한계가 있었다. 그러나 이것은 Vision Transformer(ViT)[10]를 통해 해결할 수 있었다.

2. information density가 vision과 language는 다르다. language의 경우에는 인간이 만든 것이기 때문에 highly semantic하고 information-dense하다. 그래서 사라진 몇개의 단어를 예측하기 위해서는 language 전체에 대한 이해가 필요하다. 그러나 image의 경우에는 nature-made이기 때문에 몇 개의 information이 사라지더라도 전체에 대한 이해 없이도 이웃한 patch로부터 예측이 가능하다. 이런 low-level image semantic을 해결하기 위해서 image의 많은 비율의 random patch를 뽑아내고자 하였다. 이는 self-supervised task를 더 어렵게 만든다.
   
3. text와 image의 decoder의 목적이 다르다. text의 경우에는 decoder가 missing words를 예측해야 하고 이는 rich semantic information을 포함하고 있다. 그러나 image decoder의 경우에는 pixel를 복원하는 것이기 때문에 text decoder의 recognition task 보다는 lower semantic level을 지닌다. 그래서 decoder가 semantic level를 결정하는데 중요한 역할을 한다.

이 세가지 질문을 해결하기 위해 간단하고 효율적인 masked autoencoder(MAE)에 대해 연구하였다. 이 모델은 input image에 random patches 들을 masking하고 missing된 부분을 decoder를 통해 복원하고자 하였다. MAE의 encoder와 decoder는 비대칭적인 디자인을 가지고 있다.(Figure1)

![Figure1](/.gitbook/assets/24/2022spring/Figure1.png)
*Figure1. Masked Autoencoder architecture*

75%가 masked된 image에 대해서 visible patch만 encoder에 넣고, latent representation을 도출한다. 그 후 mask tokens과 함께 latent representation을 small decoder에 넣어 사라진 부분을 복원하고자 한다. 이 때, encoder에서 small portion만 진행되기 때문에 pre-training time과 memory consumption을 줄일 수 있었다. 

## 3. Method

MAE는 비대칭적인 design을 가진 encoder와 ecoder를 사용하였다.(Figure 1) encoder는 masked된 input에서 visible patches만 보고 lightweight deocder를 이용하여 사라진 부분을 예측하였다.

**Masking**

input image를 masking하기 위해 겹치지 않게 patch를 나누고, uniform distribution에 따라 random patches를 샘플링하였다. 이 때, 이웃한 patch로 부터 예측(extrapolation)하는 것을 방지하기 위해서 uniform distributin으로 샘플링하였다. 또한, 이는 center만 치중하여 masking되는 것을 방지하고, 효율적인 encoder를 만들도록 하였다.

**MAE encoder**

standard ViT의 경우에는 patch들을 linear projection하지만 이 논문에서는 masked patches들은 제거하고 visible patches에만 작동하도록 하였다. 이를 통해 일부에만 encoder가 적용되도록 하여 time과 memory의 사용을 줄였다.

**MAE decoder**

Figure 1에서 볼 수 있듯이 decoder에 들어가는 input은 encoded visible patches와 mask tokens로 이루어져 있다. 그 후 각 token에 positional embedding을 더하여 image의 위치 정보를 더하도록 하였다. MAE decoder의 경우에는 recognition task를 하는 text decoder와 달리 reconstruction task에만 사용되기 때문에 encoder design과 독립적으로 design 될 수 있다. 이는 encoder에 비해 더 작고 narrower한 decoder의 사용을 가능하게 했다. 

**Reconstruction target**

decoder는 each masked patch의 pixel 값들을 예측한다. 그래서 decoder output의 channel수는 각 patch의 pixel 갯수와 같다. 그 후 reshape를 통해 복원된 image로 만든다. 이 때, reconstucted image와 original image를 비교하기 위해 the mean squared error (MSE)를 사용한다. 이 논문에서는 여기에서 더 나아가 각 patch의 값을 normalization 하여서 학습이 이루어지도록 하였다. 이 normalization이 결과를 향상시킨다는 사실을 실험을 통해 밝히고 있다.

**imple implementation**

MAE pre-training 시 다음의 방식을 사용하였다. 먼저 모든 input patch에 token을 형성하였다. 그 후, tokens들을 랜덤하게 섞고 masking ratio에 따라 일부 patch들을 제거하였다. 이 일부의 tokens들을 encoder에 넣어 과정이 진행되도록 한 것은 input을 masking한 것과 같다. encoding 후에는 mask token을 더하여 섞지 않은 채로 positional embedding을 더하여 decoder에 넣어지도록 계산되었다. 이런 방식은 sparse operation 없이 빠르게 작동되도록 하였다.

## 4. Experiment & Result

### Experimental setup

**Dataset**

이 논문에서는 self-supervised pre-training을 위해 ImageNet-1K(IN1K) training set을 사용하였다. 

**Evaluation**

Pre-trained된 model에 대해 supervised training을 하여 (1)end-to-end fine tuning (2)linear probing에 대해 evaluation을 하였다. 이 때, 224*224 crop된 image에 대해 top-1 validation accuracy를 도출하였다.

**baseline**

이 논문에서는 ViT-Large (ViT-L/16)을 backbone으로 사용하였다. 이 논문에서는 scratch부터 ViT-Large를 사용했을 때에 비해 baseline MAE로부터 fine-tuned 하였을 때 더 높은 accuracy를 얻을 수 있음을 밝혀냈다. 


### Result

#### Main Properties

![Table1](/.gitbook/assets/24/2022spring/Figure2.png)
*Table1. Experiment Result*

**Masking ratio**

![Figure2](/.gitbook/assets/24/2022spring/Figure3.png)
*Figure2. Masking ratio에 따른 Accuracy 변화*

Figure2는 masking ratio의 영향을 보여주고 있다. masking ratio을 15%로 설정하는 BERT와 달리 MAE는 75%의 masking ratio가 좋은 결과를 도출함을 알 수 있었다. 또한, Figure2에서는 fine-tuning과 linear probing에서 다른 경향성을 나타내는 것을 보여준다. fine-tuning의 경우에는 40-80%의 masking ratio에서 비슷한 결과를 내지만, line-probing의 경우에는 75%의 masking ratio의 경우에 10%의 masking ratio의 경우에 비해 약 20%의 더 높은 accuracy를 도출하였다. 이 후 실험에서는 75%의 masking을 통해 pre-training을 진행하였다.

**Decoder design**

이전에도 말했듯이 decoder는 reconstuction task에만 사용되기 때문에 자유롭게 design 될 수 있다. Table1-(a)에서는 decoder depth에 따른 정확도의 변화를 보여주고 있다. 이때 deep decoder는 linear probing에 더 많은 영향을 끼치는 것을 볼 수 있다. 그 이유는 autoencoder에서의 마지막 부분에 있는 layer들은  recognition보다는 reconstuction에 더 특화되어 있기 때문이다. 이는 deep decoder를 사용할수록 reconstuction에 더 특화된 다는 것을 의미한다. 그래서 deep decoder를 사용할수록 중간의 layer들이 recognition에 더 특화되기 때문에 linear probing의 경우 더 좋은 accuracy를 얻을 수 있다. 이는 Table1의 (a)에서 8%의 정확도 향상이 도출되는 것을 통해 확인할 수 있다. 그러나 fine-tuning의 경우에는 마지막 layer까지 모두 사용하기 때문에 마지막 layer를 recognition에 맞게 fine-tuning 할 수 있다. 그래서 fine-tuning의 경우 decoder depth에 관계없이 accuracy가 같게 도출되는 것을 볼 수 있다. 이때, fine-tuning을 사용한다면 1개의 decoder block으로도 84.8%의 정확도를 얻을 수 있기 때문에 1개의 decoder block을 speed-up pre-training에 사용할 수 있다.

Table1-(b)에서는 decoder의 width에 따른 accuracy를 보여주고 있다. 그래서 fine-tuning과 linear probing에서 모두 좋은 accuracy를 도출하는 512 width를 사용하기로 했다. 이후 실험에서는 512 width를 가진 8개의 block의 decoder를 사용하기로 한다. 

**Mask token**

Method에서는 mask token을 encoder에 넣지 않고 decoder에만 넣기로 하였다. Table1-(c)에서는 그것에 관한 실험을 해보기로 하였다. mask token을 encoder에 넣어서 실험해보면 linear probing의 경우 14%의 accuracy가 떨어지는 것으 볼 수 있다. 이런 accuracy의 감소는 pre-training과 deployment사이의 차이로 인해 발생한다. pre-training 시에는 mask token이 더해지지만 deployment는 input image에서 붕괴된 부분이 없기 때문에 그 차이로 인해 accuracy가 감소한다. 그래서 붕괴된 부분에 대한 mask token은 decoder에서만 사용하기로 한다.

![Table2](/.gitbook/assets/24/2022spring/Figure4.png)
*Table2. baseline model에 따른 MAE training 시간*

또한, Table2에서는 encoder에서 visible한 input patch만 사용함으로써 training 시간을 줄이는 것을 확인할 수 있다. large encoder(ViT-H)를 사용할수록, decoder depth를 줄일수록 pre-training 시간이 줄어드는 것을 확인할 수 있다. 그 시간의 줄어듦은 self-attention complexity의 증가로 인해 이차함수적으로 감소하는 것을 볼 수 있다.


**Reconstruction target**

Table1-(d)에서는 input에 따른 accuracy의 차이를 보여주고 있다. 각 patch에 normalization을 적용했을 때 적용하지 않을 때보다 더 높은 accuracy를 얻는 것을 볼 수 있다. 또한, PCA를 적용했을 때에는 accuracy가 감소하는 것을 볼 수 있다. 이것은 high frequency component가 patch안에 유지될 때 더 좋은 결과를 얻을 수 있음을 알 수 있다.

또한, normalization 대신 tokenization을 사용할 때의 accuracy 차이도 측정하였다. DALLE pre-trained dVAE[11]를 tokenizer로 사용하여 decoder에서 token을 예측하고자 하였다. 그러나 unnormalized에 비해 조금의 accuracy가 증가 하거나 오히려 감소하기도 하였다. 또한, tokenization을 사용하면 dVAE의 pre-training이 필요하여 시간이 더 걸린다. 이는 patch단위의 normalization이 더 효율적임을 알 수 있다.


**Data augmentation**

Table1-(e)에서는 data augmentation에 따른 accuracy의 변화를 보여주고 있다. cropping은 더 좋은 accuracy를 보여주고 있지만, color jittering은 오히려 accuracy를 감소시키고 있다. 여기에서 주목할 점은 data augmentation을 적용하지 않아도 좋은 accuracy를 도출할 수 있다는 점이다. 이는 다양한 data augmentaion을 사용하는 contrastive learning이나 BYOL[12], SimCLR[13]과 같은 방식과 큰 차이를 보이고 있다. 또한, MAE는 data augmentation 대신의 randomness를 random masking을 통해 더하고 있다. 

**Mask sampling strategy**

![Figure3](/.gitbook/assets/24/2022spring/Figure5.png)
*Figure3. Masking sampling strategy*

Figure3에서는 다른 mask sampling 전략을 보여주고, 이에 대한 accuracy 차이를 Table1-(f)에서 보여주고 있다. Figure5의 중간 그림처럼 block-wise로 masking을 했을 때 50%만 degrading 했음에도 random sampling에 비해 더 높은 training loss와 blurring한 결과를 얻었다. 또한, Figure5의 오른쪽 그림처럼 grid masking을 했을 때에는 더 낮은 training loss와 sharper한 reconstuction 그림을 얻었지만, 중간중간 grid 형태가 보이는 좋지 못한 그림을 도출함을 볼 수 있다. 이를 통해 higher masking ratio를 가진 random sampling이 가장 좋은 reconstruction과 accuracy를 얻을 수 있음을 알 수 있다.

**Training schedule**

![Figure4](/.gitbook/assets/24/2022spring/Figure6.png)
*Figure4. Epoch에 따른 accuracy 변화*

Figure4에서는 Epcoh에 따른 accuracy의 변화를 볼 수 있다. 두 경우의 모두 epoch에 따라 accuracy가 steadily하게 증가하는 것을 볼 수 있다. 이는 300epoch 이후에는 더이상 accuracy가 증가하지 않는 contrastive learning과는 다르다. 이는 한 epoch당 보는 patch의 수가 MAE에 비해 contrasitve learning의 경우 훨씬 많기 때문이다. 또한, MAE의 경우 적은 수의 patch가 random하게 들어오기 때문에 accuracy가 계속 증가할 수 있다.


#### Comparisons with Previous Results

![Table3](/.gitbook/assets/24/2022spring/Figure9.png)
*Table3. ImageNet-1K에 대한 method에 따른 results*

Table3에서는 ImageNet-1K에 대한 self-supervised method와 MAE를 비교한 결과에 대해 제시하고 있다. Figure6에서 알 수 있듯이 다른 self-supervised learning에 비해 MSE가 더 높은 accuracy를 도출함을 알 수 있다. 그리고 더 큰 모델인 ViT-H를 사용할수록 더 높은 accuracy를 도출한다. 또한, BEiT[2]와 비교해봤을 때에도 MAE가 더 높은 accuracy를 도출한다. 여기에서 중요한 점은 MAE가 더 빠르고 간단하게 pre-training 된다는 점이다. 마지막으로 MAE는 빠르게 pre-trained되기 때문에 1600 epoch으로 학습할 때의 시간이 MoCo v3를 300 epcoh으로 학습했을 때 시간보다 더 적다.

#### Partial Fine-tuning

![Figure5](/.gitbook/assets/24/2022spring/Figure7.png)
*Figure5. Partial Fine-tuning*

Figure5에서는 Fine-tuning하는 block의 갯수에 따른 accuracy의 변화를 보여주고 있다. 이 때, 0 block fine-tuning은 linear probing, 24 block fine-tuning은 full fine-tuning을 의미한다. linear probing의 경우 feature layer를 사용하기 때문에 다른 feature들을 사용할 기회를 잃게 된다. 그래서 partial fine-tuning을 적용하고자 하였고, 1개의 partial fine-tuning을 적용하였을 때 73.5%에서 81%로 accuracy가 크게 증가하는 것을 볼 수 있다. 또한, 약간의 fine-tuning만 적용해도 full fine-tuning만큼 좋은 accuracy를 얻을 수 있는 것으로 보아 partial fine-tuning이 MAE에 효율적임을 알 수 있다.

또한, Figure5에서 contrastive learning을 사용한 MoCo v3[14]와의 결과를 비교하고 있는데, partial fine-tuning을 적용한 MAE의 경우 훨씬 높은 accuracy를 도출함을 알 수 있다.

#### Transfer Learning Experiments

![Table4](/.gitbook/assets/24/2022spring/Figure8.png)
*Table4. COCO object detection and segmentation*

Table4는 pre-trained model을 이용하여 downstream task를 평가 한 것이다. COCO datset을 이용하여 object detection과 segmentation을 하였을 때 label이 있는 supervised learning에 비해 더 높은 point를 도출하는 것을 볼 수 있다.(50.3 vs 47.9 / 53.3 vs 49.3) 비슷하게, 다른 task인 semantic segmentation과 classification tasks도 MSE로 pre-trained한 모델이 supervised learning보다 더 높은 accuracy를 도출한다.

## 5. Conclusion

이 논문에서는 self-supervised learning을 computer vision에 적용하는 방법에 대해서 설명하고 있다. Masked Autoencoder 방식을 활용하여 label을 이용한 supervised learning이 아닌, input의 사라진 부분을 복원하면서 self-supervised learning을 하고 있다. 이 때, object를 제거하는 등의 semantic 하게 지우는 것이 아니라 pixel을 random하게 제거하여 이를 복원하도록 하고 있다. 이를 통해 supervised learning보다 더 높은 accuracy를 도출할 수 있음을 보여주고 있다.

이렇게 computer vision에서 self-supervised learning 방식을 적용하였다는 것이 흥미로웠다. 그리고 Masked Autoencoder을 활용하여 pre-training의 시간과 memory 사용을 줄인 것도 큰 contribution이라고 생각된다. 이 방식을 데이터가 많이 없는 task나 데이터가 일부 소실된 image에 대해서 적용해 볼 수 있을 것 같다.

아쉬운 점은 이 논문에서 주로 fine-tuning, linear probing을 이용한 accuracy에 대한 결과만 제시할 뿐 복원된 이미지 자체에 대한 이야기는 적은 것 같다. 논문에 따르면 복원된 이미지가 original 이미지와 비교해 보았을 때 blurring 한 것을 제외하면 대부분 잘 복원 하는 것으로 보여진다. 그러나 제대로 복원되지 못한 이미지에 대해서는 제시하고 있지 않다. 그런 경우를 제시하여 모델이 무엇과 혼동을 했고 왜 그런 결과가 나왔는지에 대한 분석이 조금 더 있었으면 좋았을 것 같다. 또한, original image와 reconstuction image를 비교할 때 MSE Loss를 사용했다고 말하고 있다. MSE Loss 이외에도 다른 Loss를 사용하여 reconstuction image의 resolution을 높히는 것에 대한 연구가 더 있었다면 좋았을 것 같다.



### Take home message \(오늘의 교훈\)



> Self-supervised learning을 MAE라는 방식으로 이미지에 적용하여 그 정확도를 높히고 있다
>
> 이를 통해 언어와 이미지의 경계는 없는 것을 알 수 있다.
>
> 앞으로 그런 독창적인 방식에 대한 연구가 더 필요할 것으로 생각된다.

## Author / Reviewer information

### Author

**김세희 (Sehui Kim)** 

* Affiliation \(KAIST AI)
* Contact information \(sae0919@kaist.ac.kr)

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

[1] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Ben- jamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language mod- els are few-shot learners. In NeurIPS, 2020.

[2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In NAACL, 2019.

[3] Geoffrey E Hinton and Richard S Zemel. Autoencoders, minimum description length, and helmholtz free energy. In NeurIPS, 1994.

[4] Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre- Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In ICML, 2008.

[5] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017.

[6] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pix- els. In ICML, 2020.

[7] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual rep- resentations. In ICML, 2020.

[8] Xinlei Chen and Kaiming He. Exploring simple Siamese represen- tation learning. In CVPR, 2021.

[9] Jean-Bastien Grill, Florian Strub, Florent Altche ́, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Remi Munos, and Michal Valko. Boot- strap your own latent - a new approach to self-supervised learning. In NeurIPS, 2020.

[10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa De- hghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.

[11] AdityaRamesh,MikhailPavlov,GabrielGoh,ScottGray,Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, 2021.

[12] Jean-Bastien Grill, Florian Strub, Florent Altche ́, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Remi Munos, and Michal Valko. Boot- strap your own latent - a new approach to self-supervised learning. In NeurIPS, 2020.

[13] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual rep- resentations. In ICML, 2020.

[14] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised Vision Transformers. In ICCV, 2021.
