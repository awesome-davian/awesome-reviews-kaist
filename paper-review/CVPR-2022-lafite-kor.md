---
description: Yufan Zhou / LAFITE; Towards Language-Free Training for Text-to-Image Generation / CVPR 2022
---

# LAFITE; Towards Language-Free Training for Text-to-Image Generation \[Kor\]



##  1. Problem definition

이 논문의 주요 task는 text-to-image generation입니다. MS COCO와 같은 complex scene dataset에 대해 text caption을 input으로 현실적인 image를 출력하는 것은 매우 어려운 task입니다. 왜냐하면 text-image pair로 이루어진 dataset은 image만으로 구성된 dataset보다 훨씬 양이 적기 때문입니다.

LAFITE는 pretrained CLIP과 StyleGAN2 구조를 활용해서 text-to-image generation을 구현하였고 dataset의 부족을 해결하기 위해 CLIP을 이용해 pseudo text feature를 구해 활용하였습니다.

## 2. Motivation

우선 text-to-image와 관련한 multimodal task에서 가장 중요한 점은 서로 다른 형태의 두 data를 어떻게 semantically align 시킬 것인가입니다.

### Related work

#### - CLIP

CLIP은 open-ai에서 나온 classifier model로 image와 text를 multimodal joint space에 mapping 시키는 방식으로 학습을 시켰습니다. Text를 인코딩할 때는 기존의 다른 여럿 text encoder와 같이 Transformer를 사용했습니다. Image를 인코딩할 때는 CNN이 아닌 Visual Transformer를 사용해서 image의 patch별 feature를 Transformer에 넣는 방식으로 학습을 했습니다. CLIP 역시 multimodal model이므로 학습할 때 text-image pair data가 많이 필요한데 CLIP은 이를 보완하는 새로운 방식을 도입했습니다. Image와 그에 해당하는 label이 있으면 (Image, "a photo of {label}") 이 pair를 이용하여 text caption 없이 학습을 진행하였습니다.

### Idea

Text를 CLIP을 활용해 embedding 시키면 corresponding image를 CLIP을 활용해 embedding 시킨 것과 유사한 곳에 mapping이 됩니다. LAFITE는 text-image pair data 대신 image data만을 활용해서 학습을 하였는데, image의 CLIP embedding과 그에 해당하는 우리가 가지고 있지 않은 text의 CLIP embedding이 유사할 것이라는 가정 하에 text data 대신 CLIP image embedding을 살짝 변형시켜서 만든 pseudo text feature를 사용합니다. 

## 3. Method

![image](https://user-images.githubusercontent.com/45480548/163987906-2c073490-ca37-4668-8fb6-907eca1f8103.PNG)
Lafite는 두 가지 세팅이 있는데 하나는 text data를 사용하지 않고 image data만 사용하는 language-free setting이고 하나는 image-text pair를 사용하는 standard setting입니다. language-free setting에서는 text feature 대신 앞서말한 pseudo text feature를 사용하는데 image embedding을 standard gaussian noise로 perturb 시킨 방식과 NN을 사용해 noise의 mean과 variance를 구해서 perturb 시키는 방식이 있습니다. 이 외에는 두 세팅은 동일한 방법으로 실험이 진행됩니다.

![image](https://user-images.githubusercontent.com/45480548/163987907-594b45a1-afed-4ad8-9382-bf2d2e1829aa.PNG)
우선 구조를 살펴보면 StyleGAN2와 거의 유사합니다. 다만 기존의 StyleGAN은 random image 생성을 위한 모듈이므로 noise를 통해 style을 구하는 부분에 text feature(혹은 pseudo text feature)를 넣어서 conditional style vector를 넣게 됩니다.

![image](https://user-images.githubusercontent.com/45480548/163987896-dcdbc105-dd57-4156-a05a-18e3c4f25be3.PNG)
Loss에서는 Discriminator를 통해 구한 feature를 통해 Real/Fake를 판단하는 conditional GAN loss, Discriminator를 통해 구한 feature와 text feature 사이의 contrastive loss, 새로 생성한 image의 CLIP embedding과 text embedding 사이의 contrastive loss 이렇게 크게 3가지를 사용합니다.

## 4. Experiment & Result

CLIP을 사용한 language-free setting은 기존 text-to-image model들이 standard setting으로 학습한 대부분의 모델보다 더 좋은 성능을 보였습니다. 이는 CLIP의 multimodal joint space의 특성을 잘 활용했기 때문입니다.

### Experimental setup

* Dataset: MS-COCO, CUB, LN-COCO, MM CelebA-HQ

* Training setup: 4 Nvidia Tesla V100 GPUs

* Evaluation metric: FID, IS

### Result

아래 그림은 LAFITE로 부터 language-free 세팅에서 생성된 이미지들입니다.
![image](https://user-images.githubusercontent.com/45480548/163986156-be81e8ee-ef59-45b4-989b-6aef3cfa88e2.PNG)
Text data 없이 image data로만 학습시키기는데도 불구하고 text와 image가 매우 align된 것을 볼 수 있고 이는 text data 대신 CLIP을 통해 구한 pseudo text feature를 사용하는 것이 효과적인 방법임을 보여줍니다.
![image](https://user-images.githubusercontent.com/45480548/163987347-cb944ffe-9340-4060-a963-4fca7c50087e.PNG)
이 표는 MS-COCO 데이터셋에서 zero-shot 성능을 보여주는 표입니다. DALL-E나 CogView와 같은 1000억개가 넘는 parameter를 2억5천개의 text-image pair data로 학습한 초대형 모델보다 LAFITE는 7500만개의 parameter를 훨씬 적은 data인 CC3M에서만 학습하였는데도 FID에서 더 좋은 성능을 보임을 확인할 수 있습니다.
![image](https://user-images.githubusercontent.com/45480548/163987352-4bb631b7-9860-4824-bc4b-3bdb89be72a9.PNG)
LAFITE의 제일 큰 장점인 image data로만 학습이 가능하다는 것 이외에 standard-setting에서도 뒤지지 않습니다. 대부분의 dataset에서 LAFITE는 제일 좋은 성능을 보여주고 있습니다.

## 5. Conclusion

이 논문은 text-to-image generation의 성능에 큰 향상을 가져온 LAFITE라는 model에 관한 연구입니다. LAFITE는 1) multimodal learning에서 제일 큰 어려움으로 여겨지는 data 부족 문제를 CLIP이라는 multimodal encoder를 활용해서 해결하였으며 2) StyleGAN2라는 검증된 구조의 network를 활용해 language-free setting 뿐 아니라 zero-shot 성능이나 standard setting 성능도 최고 수준입니다. 물론 엄청난 dataset에서 학습된 pretrained CLIP과 image generation에서 제일 많이 쓰이는 구조인 StyleGAN의 구조를 가져오긴 했지만 language-free setting에서의 contribution은 확실한 논문인 것 같습니다.

### Take home message \(오늘의 교훈\)

논문의 contribution을 위해서는 처음부터 새로운 구조를 만들고 새로운 방법을 만들어서 연구하는 것도 좋지만 이미 나온 여러 성능 좋은 module들을 적절히 조합해서 새로운 contribution을 만들어 내는 것도 하나의 뜻 깊은 연구라는 생각이 들었습니다.

## Author / Reviewer information

### Author

**이재웅 \(Jaewoong Lee\)** 

* KAIST AI
* GitHub: https://github.com/hello3196

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. Alec Radford, et al. "Learning transferable visual models from natural language supervision" 
2. [github](https://github.com/drboog/Lafite)

