---
description: Peihao et al. / Barbershop: GAN-based Image Compositing using Segmentation Masks / SIGGRAPH Asia 2021
---

# Barbershop \[Korean\]

## 1. Problem definition

Image compositing은 원하는 이미지를 합성하여 진짜 같은 이미지를 만들어내는 것 입니다. 이 중 한가지 방법으로 여러 이미지의 특징을 모아서 혼합하여 새로운 이미지를 만들어내는 방법이 있습니다. GAN의 최근 발전 덕분에 이러한 이미지 합성에 대한 연구의 결과들이 나오고 있으나, 여러 이미지들 간의 조명, 기하학, partial occlusion 등의 차이가 혼합에 어려움을 유발합니다. 특히 얼굴 이미지를 혼합하거나 편집하는 부분은 머리카락이나, 눈, 이빨과 같은 서로 다른 특징의 이미지 패치가 많이 존재하기 때문에 특히 어렵습니다.

최근 GAN을 활용하는 얼굴 이미지 편집은 크게 두가지 방법으로 나뉩니다. 하나는 학습된 네트워크(보통은 StyleGAN\[3,4\])의 latent space를 조작하는 방법입니다. 이는 이미지의 전체적인 특징인 성별, 표정, 피부색, 포즈 등과 같은 특징을 바꾸는 데에 효과적입니다. 또 다른 방법은 conditional GAN구조를 활용하여 원하는 속성 변화 정보를 입력으로 넣어 주는 방법입니다. 이러한 방법들을 활용하여 헤어 스타일을 변환시키는 방법 중 conditional GAN구조를 활용하여 합성할 영역에 대한 정보를 입력하여 해당 영역을 생성하는 방법이 뛰어난 결과를 보여줍니다. 기존의 방법들은 긴머리에서 짧은 머리로 바뀌어서 배경이 노출되는 경우 처럼 영역이 없어지는 경우를 해결하기하여 pre-trained inpatining network를 사용합니다. 하지만 이러한 경우, 이미지 합성 네트워크와 inpainting nerwork의 결과들의 퀄리티 차이가 발생하여, 접합 부분이 어색해지거나 원하지 않는 아티팩트가 생기는 경우가 많습니다.

따라서 이 논문에서는 이러한 문제점을 해결하기 위해 GAN-inversion 방법을 사용하여 하나의 네트워크만 활용함으로써 더 좋은 품질의 이미지를 합성해냅니다.

![그림 1. Barbershop 이미지 합성 결과](/.gitbook/assets/2022spring/13/bbs_fig1.png)

## 2. Motivation

최근 StyleGAN \[3,4\] 과 같은 생성모델은 높은 품질의 다양한 얼굴 이미지들을 생성할 수 있게 되었고 이를 활용하여 여러 이미지의 특징을 합성하는 연구들을 진행하였습니다. 특히, 주로 GAN inversion 기법을 사용하였는데 우선 참조 얼굴 이미지을 미리 학습된 생성모델을 사용하여 latent code로 맵핑 시키고 원하는 합성 결과 이미지를 나타내는 새로운 latent code를 optimization을 이용하여 찾아내는 방법으로 고품질의 얼굴 이미지를 생성하였습니다. 다만, 이러한 latent code를 찾아내는 방법에 따라 결과물이 천차만별로 다르고, 서로 다른 특성을 지닌 이미지의 latent code를 합성하는 방법은 특징들 간의 spatial correlation 으로 인하여 쉽지 않았습니다.

이 논문에서는 미리 학습된 생성모델에서 더 좋은 합성 결과 이미지를 나타내는 latent code를 segmentation map을 참고하여 찾아내는 것을 목표합니다.

### Related Work

#### GAN-inversion

StyleGAN\[3,4\]은 생성 결과 이미지는 놀라운 퀄리티를 보여줍니다. 또한, StyleGAN의 latent space를 조작함으로써 생성 이미지의 포즈나, 성별과 같은 속성들을 자연스럽게 변화시킬 수 있습니다. 하지만, StyleGAN이 생성한 이미지는 모두 실제 이미지가 아닌 가짜 이미지이기 때문에 실제 이미지를 StyleGAN이 표현 할 수 있도록 해야 이러한 조작을 실제 이미지에 대해서도 사용할 수 있게 됩니다. 따라서 pre-trained StyleGAN이 실제 이미지를 표현 할 수 있는 latent space를 찾아 내고자 하는 GAN-inversion 연구들이 있습니다. 이러한 방법에는 크게 두가지 방법이 있습니다. 실제 타겟 이미지에 대하여 gradient를 계산하여 latent code나 mapping network의 입력 $$z$$ 를 optimization 하는 **embed** 방식과, 실제 이미지를 입력으로 넣으면 StyleGAN의 latent space의 code로 변환시켜주는 encoder를 학습하는 **project** 방식입니다. I2S\[5]나, StyleGAN저자가 사용한 것과 같은 embed방식은 타겟 실제 이미지를 높은 퀄리티로 생성해 내지만, optimization방식을 활용하기 때문에 한장의 이미지를 표현하기 위해서 학습하는 것과 같이 오랜시간이 걸린다는 단점이 있습니다. 실제 이미지를 입력으로 넣으면 이제 해당하는 latent code로 변환시키는 encoder를 학습하는 방식들은 (psp\[6]\, e4e\[7\] 한번의 network feed forward만 이루어지기 때문에 빠르게 이미지를 표현할 수 있다는 장점이 있지만 embed방식 보다는 낮은 퀄리티의 이미지를 생성해 냅니다.

#### How to control the output result images of GAN?

최근 연구들은 StyleGAN처럼 mapping network를 통해서 만들어진 latent space가 많은 정보를 지니고 있고, 이를 조작함으로써 생성 이미지의 특징을 변화시킬 수 있음을 보여주었습니다. 이를 조작하여 이미지를 컨트롤 하는 방법중 pre-trained StyleGAN을 활용하는 방법과, 비슷한 구조의 Image2Image tralanslation구조를 활용하여 입력에 원하는 정보를 주는 방법이 있습니다. pre-trained StyleGAN의 latent space를 조절하는 방법으로는 속성을 latent code에 반영하는 Styleflow\[8\], text input을 활용하여 latent space를 조작하는 StyleCLIP\[9\]등이 있습니다. Image2Image tranlation구조를 활용하여 입력을 조절하는 방법으로는 segmentation map을 입력으로 받아 이미지로 변형하는 SPADE\[10\], SEAN\[11\]이나, 변형하고자 하는 속성을 참고 이미지로 받아서 전체 이미지의 스타일을 변형하는 StarGAN-v2\[12\]등이 있습니다. 

### Idea

본 논문의 Barbershop은 퀄리티를 위하여 embed방식을 활용하였습니다. 또한 segmentation map을 활용하여 각 영역별로 다른 이미지를 타겟으로 한 loss function을 활용하여 실제 이미지를 pre-trained StyleGAN에 embed 하는 방식을 활용하였습니다. 이를 통해 영역별로 다른 스타일을 지닌 이미지를 합성해 냅니다.

## 3. Method

본 논문에서는 segmentation map을 활용하여 각 영역별로 다른 이미지를 타겟으로 한 loss function을 활용하여 실제 이미지를 pre-trained StyleGAN에 embed 하는 방식을 활용하였습니다.

#### Overview
본 논문에서는 이미지 합성을 위하여 segmentation map을 활용합니다. 따라서 합성된 이미지의 형태는 활용된 segmentation map의 형태에 따라 정해지게 됩니다. 이미지 합성에 메인으로 활용되는 방법은 실제 이미지를 StyleGAN latent space에 project시키는 GAN-inversion방법을 활용합니다. 본 논문에서 사용하는 StyleGAN은 StyleGANv2\[4\]를, embed방법으로는 II2S\[13\]을 활용하였습니다. II2S를 활용함에 있어서 결과 이미지의 디테일을 위하여 $$C = (F, S)$$라는 새로운 latent code 를 제안합니다. StyleGANv2는 18개의 latent code를 사용하는데 $$F$$는 StyleGANv2의 8번째 style block의 output feature map을 의미하며 나머지 10개의 latent code를 $$S$$, appearance code라고 명명합니다. 따라서 II2S방법으로 실제 이미지를 표현하는 $$C$$ 값을 찾게 됩니다.
위에 언급된 방법들도 이미지를 합성하는 순서는 다음과 같습니다.

1. 스타일 변화에 사용될 참조 이미지들의 segmentation map을 구합니다.
2. 구한 segmentation map을 정렬하여 target segmentation map을 만듭니다.
3.  target segmentation map에 맞도록 참조 이미지들을 align합니다.
4.  align된 이미지들을 embedding 하여 각 이미지 별로 해당하는 $$C = (F, S)$$ 값을 찾아냅니다.
5. targe tsegmentation map의 영역별로 타겟 이미지가 다르게 학습하는 masked-appearance loss function을 활용하여 여러 이미지의 appearance와 structure 가 혼합된 C 값을 찾아냅니다.

#### Target Semgmentation
Target segmentation map을 생성하는 방법은 간단합니다. 그림1과 같이 이미지 별로 원하는 영역을 추출하여 적절하게 섞음으로써 target segmentation map을 만들게 됩니다. 만약 영역이 많이 빗나가서 빈 영역이 많이 생기는 경우에는 그림 2와 같이 이미지의 가운데 부분을 기준으로 영역을 채우는 단순한 방법으로 영역을 채우게 됩니다.

![그림 2. Target segmentation map 만들기 및 Inpainting 과정](/.gitbook/assets/2022spring/13/inpainting.png)

#### Image align & embedding

이미지를 합성하기 위해서 먼저 참조 이미지들을 만들어진 target segmentation $$M$$에 맞도록 정렬합니다. 정렬 방법은 두가지 스텝으로 이루어 집니다. StyleGAN에 각 참조 이미지 $$Z_{k}$$ 마다 embedding 방법을 활용하여 각 이미지를 복원하는 latent code $$C_{k}^{rec}$$ 를 아래와 같은 loss를 활용하여 찾아냅니다.

![](/.gitbook/assets/2022spring/13/loss1.png)

<!--
$$
\begin{aligned}
C_{k}^{rec} = argmin_{C} L_{LPIPS}(C) + L_{F} \\
L_{F} = ||F - F_{k}^{init}|| \\
F_{k}^{init} = G_{8}(w_{k}^{rec})
\end{aligned}
$$
-->

이때 embed하는 latent space를 $$FS$$ space 라고 하며, SytleGAN의 8번째 블록의 feature map $$F$$와 나머지 블록에 들어가는 10개의 latent code $$S$$로 이루어져 있습니다.
이렇게 찾아낸 $$C_{k}^{rec}$$를 활용하여 $$F_{k}^{align}$$ 을 찾아내는데 그 이유는 처음부터 target segmentation $$M$$ 에 맞는 latent code를 찾는 것보다 spatial 정보가 많은 $$F $$ code의 디테일을 살리기 위해서라고 합니다.
Align된 참조 이미지를 표현하는 $$w^{align}$$을 찾기 위하여 masked style-loss와 segmentation output에 대한 cross-entropy loss 두가지를 활용합니다.

![](/.gitbook/assets/2022spring/13/loss2.png)
<!--
$$
\begin{aligned}
L_{s} = \sum_{l} \vert\vertK_{l}(I_{k}(G(w)) \bullet (G(w)) – K_{l}( I_{k}(Z) \bullet Z_{k}))\vert\vert^{2} \\
L_{seg} = XEnt(M, Segment(G(W))) \\
L_{align}(W) = L_{seg} + \lambda_{s}L_{s} \\
\end{aligned}
$$
-->

$$I_{k}(Z) = 1\{Segment(Z) = k\}$$ 는 각 참조 이미지 $$Z_{k}$$ 에서 $$M$$에 사용된 영역에 해당 되는 부분만을 추출한 마스크이며 따라서 $$I_{k}(Z)\bulletZ_{k}$$는 해당 영역만 이미지에서 추출한 것을  표현 합니다. $$K$$는 style loss 에 사용되는 gram matrix를 의미합니다. $$XEnt$$는 생성된 이미지의 segmentation map과 target segmentation $$M$$을 비교하기 위한 creoss-entropy loss 입니다.

이렇게 찾아낸 $$w_{k}^{align}$$ 를 기반으로 $$F_{k}^{align}$$을 새로 찾아냅니다. Embed된 이미지의 영역과 타겟 마스크가 겹치는 영역($$H$$)의 $$F$$ code는 $$F_{k}^{rec}$$에서 가져오고 나머지 영역은 $$w_{k}^{align}$$으로 embed한 이미지에서 가져오는 것으로 $$F_{k}^{align}$$을 만듭니다.

![](/.gitbook/assets/2022spring/13/f_align.png)
<!--
$$F_{k}^{align} = U\bulletF_{k}^{rec} + (1-U)\bulletG_{8}(w_{k}^{align})$$
-->

#### Image Blending

이미지 별로 찾아낸 $$C_{k}^{align} = (F_{k}^{align}, S_{k}^{align})$$을 이용하여 이미지를 합성하게 됩니다. 최종 합성 이미지를 표현하는 $$C^{blend}$$를 찾기 위해서 이미지의 structure를 표현하는 코드 $$F$$는 target mask $$M$$에서 해당하는 영역 $$\alpha$$의 이미지 코드 $$F_{k}^{align}$$ 을 합치는 것으로 $$F_{k}^{align}$$를 찾아냅니다. 나머지 특징을 표현하는 $$S_{blend}$$는 마스크 $$M$$에 해당하는 영역별로 LPIPS loss를 계산하여 해당하는 이미지 마다 적용되는 weight $$u$$를 찾아냅니다.

![](/.gitbook/assets/2022spring/13/fs_blend.png)
<!--
$$
\begin{aligned}
F^{blend} = \sum_{k}^{K} \alpha_k \bullet F_{k}^{align} \\
S^{blend} = \sum_{k} u_{k} \bullet S_{k} \\
\sum_{k} u_{k} = 1
\end{aligned}
$$
-->


## 4. Experiment & Result

### 4.1. Experimental Setup 

이 논문에서는 II2S를 통해서 embedding 된 이미지 120개의 1024x1024 사이즈의 이미지를 활용합니다. 이를 통해서 만들어진 198개의 이미지 pair들로 이미지 합성 실험을 진행합니다.
$$C_{k}^{rec}$$을 찾기 위해서 400 iterations을, $$C_{k}^{align}$$을 찾기 위해서 100 iterations을, $$S$$ 코드들의 합성 weight $$u$$를 찾기 위해서는 600 iterations의 embedding 과정을 진행하였습니다.

#### 4.2. Result

![](/.gitbook/assets/2022spring/13/result_table.png)

정량적 평가를 위해서 기존의 방법들과 RMSE, PSNR, SSIM, perceptual similarity, LPIPS, FID를 비교하였습니다. 표의 baseline을 제안한 방법 중 align 방법을 적용 하지 않고, $$FS$$ space가 아닌 기존의 $$W+$$ space에 대하여 embed했을 때의 결과물 입니다.
또한 396명의 참가자들을 통해 이미지 품질 평가를 진행한 결과, LOHO\[15\]의 결과와 비교해서는 378:18의 선택을, MichiGAN\[16\]의 결과와 비교해서는 381:14의 선택을 받았습니다.

![](/.gitbook/assets/2022spring/13/comp_fig.png)
![](/.gitbook/assets/2022spring/13/face_swap.png)

위 결과에서 보면 기존의 다른 방법들 보다 Barbershop의 이미지 합성 능력이 훨씬 뛰어난 것을 확인 할 수 있습니다. 또한 헤어스타일 뿐만 아니라 face swapping에서도 뛰어난 결과를 생성하는 것을 볼 수 있습니다.


## 5. Conclusion

이 논문에서는 미리 학습된 생성모델과 segmentation mask를 활용하여서 이미지 합성을 하였습니다. 특히, embed방법에 사용할 새로운 latent space $$FS$$ space를 제안하고 이에 대하여 먼저 target mask를 지정하고 이에 맞춰서 참조 이미지들을 모두 정렬하는 aligned code를 찾는 방법과, 참조 이미지 영역별로 style을 반영하는 masked-style loss function을 이용한 embed 방법을 활용하여 높은 품질의 이미지 합성 결과를 보여주었습니다.

### Take home message \(오늘의 교훈\)

> Using latent space of GAN is useful

## Author / Reviewer information

### Author

**조영주 \(Youngjoo Jo\)** 

* KAIST AI
* \[github\](https://github.com/run-youngjoo)

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. Zhu, Peihao, et al. "Barbershop: GAN-based image compositing using segmentation masks." ACM Transactions on Graphics (TOG) 40.6 (2021): 1-13.
2. Official GitHub repository : https://github.com/ZPdesu/Barbershop
3. Karras, Tero, Samuli Laine, and Timo Aila. "A style-based generator architecture for generative adversarial networks." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.
4. Karras, Tero, et al. "Analyzing and improving the image quality of stylegan." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
5. Abdal, Rameen, Yipeng Qin, and Peter Wonka. "Image2stylegan: How to embed images into the stylegan latent space?." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.
6. Richardson, Elad, et al. "Encoding in style: a stylegan encoder for image-to-image translation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
7. Tov, Omer, et al. "Designing an encoder for stylegan image manipulation." ACM Transactions on Graphics (TOG) 40.4 (2021): 1-14.
8. Abdal, Rameen, et al. "Styleflow: Attribute-conditioned exploration of stylegan-generated images using conditional continuous normalizing flows." ACM Transactions on Graphics (TOG) 40.3 (2021): 1-21.
9. Patashnik, Or, et al. "Styleclip: Text-driven manipulation of stylegan imagery." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
10. Park, Taesung, et al. "Semantic image synthesis with spatially-adaptive normalization." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.
11. Zhu, Peihao, et al. "Sean: Image synthesis with semantic region-adaptive normalization." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
12. Choi, Yunjey, et al. "Stargan v2: Diverse image synthesis for multiple domains." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
13. Zhu, Peihao, et al. "Improved stylegan embedding: Where are the good latents?." arXiv preprint arXiv:2012.09036 (2020).
