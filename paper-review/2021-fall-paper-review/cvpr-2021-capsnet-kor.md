---
description: Gu et al. / Capsule Network is Not More Robust than Convolutional Network / CVPR 2021
---

# 1. Viewpoint Equivariance

딥러닝은 수 년간 성능을 높여 왔고 많은 영역에서 인간을 추월했지만, 불행히도 근본적인 부분에서의 발전은 더디다. 현재의 딥러닝이 hard AI가 되기 위해서는 극복해야 할 많은 과제 중, Capsule network가 주목하는 것은 viewpoint equivariance에 대한 것이다. 

<img src = "https://github.com/humandream1/awesome-reviews-kaist/blob/master/.gitbook/assets/sofa.PNG" width="50%" height="50%">

위 표는 잘 학습된 Object detection 모델에서 소파를 여러 각도에서 찍은 사진을 넣었을 때 Average Precision(AP)가 어떻게 변했는지를 나타낸 것이다. 사람은 소파를 어떤 각도에서 보아도 소파라고 인지할 수 있지만, 딥러닝 모델의 성능은 0.1에서 1.0까지 다양하게 분포한다.

그 이유를 딥러닝의 언어로 설명하자면 training data(이 경우 PASCAL VOC)에 bias가 있었기 때문이다. 사람은 소파를 무작위 elevation과 azimuth에서 균일하게 촬영하지 않기 때문이다. 그러면 모든 방향에서 촬영한 소파를 데이터에 포함시키면 문제가 해결될까? 이론상 그렇다. 하지만 모든 object를 모든 각도에서 촬영한 데이터를 만드는 것은 현실적으로 불가능하고, 바람직하지도 않다. 

> "The set of real world images is infinitely large and so it is hard for any dataset, no matter how big, to be representative of the complexity of the real world." [2]

이는 분명 어려운 문제이지만 근본적으로 해결이 불가능한 문제로 취급하는 것은 망설여진다. 왜냐하면 인간은 '소파'라는 물체를 인식하기 위해서 수백 개의 소파를 모든 방향에서 관찰한 수만 장의 데이터를 필요로 하지 않기 때문이다.  Geoffrey Hinton은 이것이 인간은 물체의 part-whole hierarchy를 파악할 수 있기 때문이라고 생각했고, 오래 전부터 이를 딥 러닝에서 실현하기 위한 연구를 했다.



### Part-Whole Hierarchy

이 개념은 CapsNet[3] 논문을 통해 널리 알려졌지만, 그 고민은 훨씬 이전의 논문들에서부터[4] [5] 찾을 수 있다. 요약하자면, 사람은 소파를 어떤 특정한 각도에서의 이미지로써 기억하는 대신 '긴 깔개 뒤에 등받이가 있고 양 옆에 팔걸이가 있는 것'으로 인지하기 때문에 viewpoint equivariance를 자연스럽게 달성한다는 것이다. 

> "There is strong psychological evidence that people parse visual scenes into part-whole hierarchies and model the viewpoint-invariant spatial relationship between a part and a whole as the coordinate transformation between intrinsic coordinate frames that they assign to the part and the whole."[6]

사람은 실제로 뇌 속에서 3d 좌표계를 만들어 그 속에서 object의 형태를 인식한다. 이는 psychological evidence로 뒷받침되고, 우리의 상식에도 부합한다. Part-whole hierarchy를 인지할 수 있다면 viewpoint equivariance는 자연스럽게 달성될 것이고, 반대로 part-whole hierarchy 없이 viewpoint equivariance를 달성하는 것 역시 상상하긴 힘들다. 

물론 다른 방식으로 이를 실현하기 위한 연구들이  있었지만, 특정한 transform에 invariant한 kernel을 사용하거나 data augmentation에 의존하는 등의 방식으로 viewpoint equivariance를 자연스럽게 달성하는 방향과는 거리가 있었다. [7,8,9] 그리고 현실에서 가능한 transformation matrix는 무한히 많기에 이를 invariance를 통해 달성하려는 방식은 명확한 한계가 있다. 

그렇다면 part-whole hierarchy를 딥러닝에 어떻게 구현할 것인가 라는 문제만 남는다. Bottom-up 방식의 딥 러닝에서 우리가 무엇이 '팔걸이'이고 무엇이 '등받이'인지 직접 알려줄 수는 없지만, 최소한 네트워크가 이런 개념들을 담고 학습할 수 있도록 구조를 top-down으로 구축해 줄 수는 있을 것이다. 그렇게 등장한 기념비적인 첫 작품이 <Dynamic routing between capsules>[3], 'Capsule Network'였다. 이 논문은 딥러닝 커뮤니티에서 큰 이슈가 되었고, 수백 편의 후속 연구가 나왔다.



# 2. Capsule Network

<img src = "https://github.com/humandream1/awesome-reviews-kaist/blob/master/.gitbook/assets/capsnet.PNG" width="100%" height="100%">

Capsule network를 어렵게 설명할 방법은 많지만 개념적으로는 단순하게 이해할 수 있다. 우선, 팔걸이의 존재 유무를 단 하나의 scalar value로 표현하는 것은 무리가 있을 것이다. 팔걸이의 색상이나 질감 등은 물론이고, part-whole hierarchy를 위해서는 팔걸이가 어떤 각도로 붙어 있는지에 대한 정보도 유지해야만 한다. 좌석에 등받이가 수평으로 연결되어 있다면 이는 더이상 소파가 아니기 때문이다. 이런 정보를 담기 위해 간단한 CNN을 통해 feature vector를 뽑고, 8~16개의 feature를 묶어 하나의 capsule로 만든다.  

Deep learning의 연산은 기본적으로 linear하기 때문에 단순히 몇 개의 feature를 묶어 놓는 것 만으로는 아무 일도 발생하지 않는다. 각각의 capsule의 object를 나타내는 단위로써 기능하게 하기 위해 선택한 방법은 'routing algorithm'이라는 것인데, 이는 capsule 단위로 작동하는 Hebbian learning[10]으로 이해할 수 있다. 초기에는 모든 캡슐이 동일한 강도로 연결된다. 그렇게 high-level capsule의 값이 결정되면, 그 activation과 잘 align되는 low-level capsule과의 연결이 강화된다. 그렇게 high-level capsule의 값을 업데이트하고, 이를 반복한다. 

여기서 갑자기 Hebbian learning이 왜 등장했는지, 어떻게 이런 과정이 (3d object의) part-whole hierarchy를 구축하고 viewpoint equivariance를 달성할 수 있는지 이해하지 못했다면, 당신의 이해력이 부족한 것이 아니라 지극히 정상적인 사고를 한 것이니 안심해도 좋다. Routing algorithm이 어떤 메커니즘으로 viewpoint equivariance를 달성하는지에 대한 설명은 논문 어디에도 찾아볼 수 없으며, 잠시 후에 보겠지만 실제로 그런 메커니즘은 존재하지 않기 때문에 설명할 방법도 없다. 

간단한 사고 실험을 해 보자. 3D viewpoint equivariance보다 훨씬 간단한 것으로 우리는 2D에서 rotational equivariance를 달성하는 것을 하위 목표로 생각할 수 있을 것이다. 우리가 capsule network를 잘 학습 시켜서 'T' 라는 글자를 인식하게 만들었다고 하자. 첫 번째 capsule은 중앙에 있는 '긴 vertical line'을 학습했고, 두 번째 capsule은 '짧은 horizontal line'을 인지하도록 학습되었다. 그리고 상위 캡슐이 '긴 vertical line 위에 짧은 horizontal line이 있다면 이것은 letter 'T'이다' 라는 것을 어떻게든 학습했다고 가정해 보자. 

그런데 갑자기, 학습 데이터에는 없던 기울어진 'T'가 input으로 들어온다. 약간만 기울어졌다고 생각해도 좋고 완전히 수평으로 누웠다고 생각해도 좋다. 이제 capsule은 '긴 horizontal line 우측에 짧은 vertical line이 있다면 이것인 letter 'T'이다' 라는 것을 인지해야만 한다. 첫 번째 capsule은 본래 vertical line을 학습했지만 돌연 horizontal line에 activate 되어야 하고, 두 번째 캡슐은 방향을 바꾸는 것뿐만이 아니라 아예 위치가 변화하며, 상위 캡슐은 첫 번째 캡슐이 horizontal line에 activate되었다는 것으로부터 T의 윗면을 우측에서 찾아야 한다는 것을 사고해야만 한다. 어떻게 이게 가능할까? Capsule 사이에 적용되는 Hebbian learning이 어떻게 이를 달성할 수 있을까? 애초에 약간이라도 도움이 되기는 할까? 

Original Capsule paper[3,11]에 의하면, ''그렇다''. 그들은 routing algorithm이 완벽한 viewpoint equivariance는 아닐지라도, 최소한 기존의 CNN보다 더 robust함을 실험적으로 보였다. 그것이 많은 사람들이 capsule network에 충격을 받은 이유이다. 하지만 이는 어떤 수학적 증명은 물론이고 충분한 납득 가능한 설명 없이 오직 실험으로써 증명되었다. 만약에 그 실험이 부정된다면, 우리는 Capsule Network가 현재 어떻게 작동하고 있는지에 대해 다시 한 번 생각해 보아야 할 것이다. 



# 3. Capsule Network is Not More Robust than Convolutional Network[12]

이 논문은 CapsNet의 성능이 기대 이하라는 결과를 담은 첫 번째 연구가 아니다. 단순한 성능 비교에서 이점을 찾지 못한 연구도 있었고[13,14], SmallNorb나 Rotational MNIST 등의 데이터를 통해 CapsNet이 일반 CNN보다 딱히 viewpoint change에 robust하지 않다는 것을 보인 연구들도 있었다[15,16,17]. 

그렇다면 문제가 생긴다. 같은 dataset에서 같은 network로 실험을 했는데 다른 결론이 나왔다. 그렇다면 둘 중 한 쪽은 거짓말을 하고 있다는 것인가? 제프리 힌튼이 실험 결과를 조작해서 논문을 쓰고 NIPS와 ICLR에 실었다는 의미인가? 아니, 꼭 그렇지는 않다. 이 논문은 그런 '오해'가 발생한 과정을 상세히 밝힌다. 

### The Baseline Problem

CapsNet 논문에서, 저자는 Capsule network가 CNN보다 general performance가 더 높은 것은 물론이고 viewpoint change에 대해 더 robust하다고 실험을 통해 밝혔다. 하지만 'CNN'은 단일한 모델을 지칭하지 않는다. AlexNet이나 VGG도 있고, ResNet, SENet, MobileNet, EfficientNet 등 수많은 architecture가 존재한다. 그러면 이들 중 가장 좋은 SOTA 모델과 비교하면 될까? 이는 공정한 비교이기는 하지만, 단 4개의 layer를 가진 CapsNet이 수백 개의 layer와 수천만 개의 parameter를 가진 모델을 이겨야만 가치를 인정받을 수 있다면 이는 지나치게 가혹한 처사가 될 것이다. 

그래서 저자들이 채택한 방식은 CapsNet보다 더 큰, layer 수는 비슷하되 더 많은 parameter를 가진 모델을 가져와 비교한 것이다. 언뜻 이는 공정해 보인다. 무려 parameter 개수가 두 배나 더 많은 CNN을 상대로 승리했기 때문이다. 엄밀하게 말하자면 CapsNet은 '비슷한 크기를 가진 모든 가능한 CNN들의 성능의 upper bound'를 넘어야 한다. 하지만 이것이 불가능한 것을 알기에 자신보다 더 큰 CNN을 적당히 하나 만들고, 이것이 적절한 upper bound가 되리라 믿었다. 설마 그런 사소한 세부사항들로 인해 parameter 2배의 차이가 뒤집히진 않을 것이다. 그러므로 CapsNet은 routing algorithm이 잘 작동해서 좋은 성능을 내는 것이 틀림없다. 그런가?

크게 보자면, Capsule network에서 routing algorithm을 제거하면 일반 CNN이 된다. 더 구체적으로는, shared transform matrix와 생소한 activation function(squash), 그리고 reconstruction으로 auxiliary loss를 주고 MarginLoss로 학습하는 CNN이 된다. 그렇다면 이런 차이들을 하나씩 on/off해 가면서 실험을 해 본다면 CapsNet의 어떤 요소가 실제로 성능에 영향을 미쳤는지를 알 수 있을 것이다.

### Experiment

<img src = "https://github.com/humandream1/awesome-reviews-kaist/blob/master/.gitbook/assets/AffNEST.PNG" width="50%" height="50%">
<img src = "https://github.com/humandream1/awesome-reviews-kaist/blob/master/.gitbook/assets/table2.PNG" width="70%" height="70%">

Model의 (viewpoint) transformation에 대한 robustness를 직접적으로 측정하기 위해 AffNIST dataset[3,17]이 주로 사용된다. Training 시에는 정상적인 MNIST 데이터만 보여주고, 여기에 각종 affine transform을 가한 이미지로 evaluate하여 generalization power를 측정한다.  

만약에 CapsNet의 robustness가 capsule 구조에 의한 것이라면 routing algorithm을 사용했을 때 가장 큰 성능의 향상이 있을 것이다. 하지만 routing algorithm은 robustness에 도움이 되지 않고 오히려 성능이 소폭 감소했으며, 이 결과는 다른 연구에서 보고한 것[16,17]과 같다. 오히려 squash function 등 부가적인 요소가 성능을 끌어올린 요인인 것으로 보이며, 이 외에 저자들은 kernel size가 AffNIST에서의 성능에 결정적인 영향을 미친다는 사실을 알아냈다. 

<img src = "https://github.com/humandream1/awesome-reviews-kaist/blob/master/.gitbook/assets/kernel_size.PNG" width="70%" height="70%">

네트워크 구조에 관계없이 kernel size가 커질수록 robustness가 커짐을 알 수 있다. CapsNet은 (9,9) kernel을 사용했고 원 논문에서 baseline CNN은 (5,5)를 사용했다. 이는 의도했건 의도치 않았건 CapsNet에게 유리한 실험 설계였던 것으로 보이며, 저자들은 위와 같은 실험 결과를 토대로 (9,9) kernel과 average pooling을 사용한 간단한 3-layer network(5.3M parameter)로 AffNIST에서 CapsNet을 크게 상회하는 성능을 얻을 수 있었다. 

<img src = "https://github.com/humandream1/awesome-reviews-kaist/blob/master/.gitbook/assets/final.PNG" width="40%" height="40%">

원 논문에서 CapsNet이 35M개의 parameter를 가진 CNN보다 성능이 좋았다고 보고한 것을 생각하면 네트워크의 구조에 따라 transform에 대한 robustness에 큰 차이가 발생함을 알 수 있다. 그리고 CapsNet의 저자들은 이를 예상치 못하고 너무 나쁜 baseline을 설정하여 잘못된 결론을 도출한 것으로 보인다. 이는 우연일 수도 있고, 원하는 결과가 나올 때까지 실험을 반복했기 때문일 수도 있다.



# 4. Discussion

<img src = "https://github.com/humandream1/awesome-reviews-kaist/blob/master/.gitbook/assets/meme.png" width="30%" height="30%">

딥러닝 연구는 noise에 특히 취약하다. 적절한 Baseline을 잡는 것은 언제나 어려우며, 똑같은 실험을 수행해도 매번 다른 결과가 나오고, 약간의 차이로 완전히 잘못된 결과를 얻을 수도 있기에 실험 결과의 재연도 어렵다. 때문에 잘못된 논문이 나왔을 때 이를 검증하는 것도 쉽지 않다. 결과가 재연되지 않는 대부분의 논문들은 대부분 조용히 묻히지만 Capsule network의 경우에는 아직까지도 잘못된 가정에 기초한 후속 연구들이 꾸준히 나오고 있다.

이 사건은 논문을 쓸 때 적절한 baseline을 가지고 가설을 직접 검증하는 것의 중요성을 remind해 준다. 그리고 그런 원칙을 제대로 지키지 않은 논문에 대해서는 한 번 더 의심하고 검증해야 할 것이다.

CapsNet 자체는 성공적이지 않았지만 그 과정에 이르는 논리는 여전히 주목할 만 하며, part-whole hierarchy를 위해 capsule 구조가 필요하다는 논리는 여전히 유효할 수 있다. CapsNet은 성공적이지 못했지만 capsule이라는 개념의 존재 가치가 부정되었다기보다는 각 capsule에 의미를 부여하는 routing algorithm이 제대로 작동하지 않는다고 해석하는 것이 더 정확할 것이다. 특히 처음 제안된 두 routing algorithm은 수학적으로 stable하지 못한 것으로 보인다.[15]

Geoffrey Hinton은 이후의 논문에서 각 patch에 하나의 capsule을 할당하고 이들이 계층 간에 상호작용하는 새로운 구조를 제안하면서 CapsNet이 성공하지 못한 이유를 다음과 같이 분석했다. 

> "The fundamental weakness of capsules is that they use a mixture to model the set of possible parts. This forces a hard decision about whether a car headlight and an eye are really different parts. If they are modeled by the same capsule, the capsule cannot predict the identity of the whole. If they are modeled by different capsules the similarity in their relationship to their whole cannot be captured."[18]

이 예시가 적절한지 아닌지의 여부와 무관하게, 우리는 capsule이라는 구조가 필요하다는 사실에 대체로 공감하지만 capsule network가 우리가 원하는 대로 동작하도록 학습하게 만드는 법을 아직 찾지 못했다. 어떤 사람들은 데이터의 부족을 이유로 꼽는다. 물체의 구조를 파악하고 part-whole hierarchy를 구축하는 것은 image classification하는 것보다 훨씬 어렵다. Linear transform의 반복으로도 잘 할 수 있는 task를 굳이 더 어려운 방법으로 수행할 이유도 없고, 그렇게 할 만한 정보도 부족하다는 것이다. 때문에 viewpoint equivariance는 training 방법의 혁신(unsupervised learning 등)이 선행되어야 달성될 수도 있다. 어쨌거나 그 끝에는 capsule이 있을 것이라 믿어 의심치 않는 사람들이 있고, 나도 거기에 일부 공감한다.

<br/><br/><br/><br/><br/>
  
  



[1] Qiu, Weichao, and Alan Yuille. "Unrealcv: Connecting computer vision to unreal engine." *European Conference on Computer Vision*. Springer, Cham, 2016.

[2] Yuille, Alan L., and Chenxi Liu. "Deep nets: What have they ever done for vision?." *International Journal of Computer Vision* 129.3 (2021): 781-802.

[3] Sabour, Sara, Nicholas Frosst, and Geoffrey E. Hinton. "Dynamic routing between capsules." *arXiv preprint arXiv:1710.09829* (2017).

[4] Hinton, Geoffrey E., Alex Krizhevsky, and Sida D. Wang. "Transforming auto-encoders." *International conference on artificial neural networks*. Springer, Berlin, Heidelberg, 2011.

[5] Hinton, Geoffrey E. "Mapping part-whole hierarchies into connectionist networks." *Artificial Intelligence* 46.1-2 (1990): 47-75.

[6] Hinton, Geoffrey. "Some demonstrations of the effects of structural descriptions in mental imagery." *Cognitive Science* 3.3 (1979): 231-250.

[7] Esteves, Carlos, et al. "Equivariant multi-view networks." *Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2019.

[8] Kim, Jinpyo, et al. "CyCNN: a rotation invariant CNN using polar mapping and cylindrical convolution layers." *arXiv preprint arXiv:2007.10588* (2020).

[9] Marcos, Diego, Michele Volpi, and Devis Tuia. "Learning rotation invariant convolutional filters for texture classification." *2016 23rd International Conference on Pattern Recognition (ICPR)*. IEEE, 2016.

[10] “Hebbian theory” *Wikipedia*, Wikimedia Foundation, https://en.wikipedia.org/wiki/Hebbian_theory

[11] Hinton, Geoffrey E., Sara Sabour, and Nicholas Frosst. "Matrix capsules with EM routing." *International conference on learning representations*. 2018.

[12] Gu, Jindong, Volker Tresp, and Han Hu. "Capsule Network is Not More Robust than Convolutional Network." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2021.

[13] Andersen, Per-Arne. "Deep reinforcement learning using capsules in advanced game environments." *arXiv preprint arXiv:1801.09597* (2018).

[14] Xi, Edgar, Selina Bing, and Yang Jin. "Capsule network performance on complex data." *arXiv preprint arXiv:1712.03480* (2017).

[15] Paik, Inyoung, Taeyeong Kwak, and Injung Kim. "Capsule networks need an improved routing algorithm." *Asian Conference on Machine Learning*. PMLR, 2019.

[16] Mukhometzianov, Rinat, and Juan Carrillo. "CapsNet comparative performance evaluation for image classification." *arXiv preprint arXiv:1805.11195* (2018).

[17] Gu, Jindong, and Volker Tresp. "Improving the robustness of capsule networks to image affine transformations." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020.

[18] Hinton, Geoffrey. "How to represent part-whole hierarchies in a neural network." *arXiv preprint arXiv:2102.12627* (2021).
