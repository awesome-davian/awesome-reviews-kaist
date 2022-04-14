---
description: >-
  Lee et al. / Pop-Out Motion - 3D-Aware Image Deformation via Learning the
  Shape Laplacian / CVPR 2022
---

# Pop-Out Motion \[Kor]

안녕하세요. 본 포스팅에서는 올해 CVPR에 발표될 Pop-Out Motion이라는 논문을 소개드리고자 합니다. 자연스러운 3D-Aware Image Deformation을 위한 학습 기반의 파이프라인을 제안한 논문이며, `3D Vision`, `Shape Deformation`, `2D-to-3D Reconstruction` 등의 키워드에 관심이 있으신 분들이라면 [논문 본문](https://arxiv.org/pdf/2203.15235.pdf) 및 [프로젝트 페이지](https://jyunlee.github.io/projects/pop-out-motion/)를 구경해주시면 감사하겠습니다. 해당 논문은 제가 1저자로 참여하였으며, KAIST 전산학부의 성민혁교수님과 김태균교수님께서 지도해주셨습니다. (좋은 연구 지도를 해주신 두 교수님께 감사드립니다.)

## 1. Problem Definition

본 논문은 `3D-Aware Image Deformation` 이라는 문제를 해결하고자 합니다. 사용자가 이미지 내의 객체 모양을 자연스럽게 변형하는 것을 가능하게 하되, 2D 영상의 피사체가 마치 3D 공간에 존재하는 것과 같이 변형할 수 있도록 하는 것이 목표입니다. 이 때 직관적인 이미지 수정을 위하여 사용자가 키포인트 등의 `Deformation Handle` 을 매개체로서 사용할 수 있도록 합니다. 아래의 그림 예시를 보시면, 사용자가 이미지에 키포인트들 (파란색 원 표시) 을 지정하고 그 중 하나를 선택하여 움직일 경우 (빨간색 화살표 표시), 그림 내의 사람 객체 모양이 그에 맞추어 자연스럽게 변형되는 것을 볼 수 있습니다. 이 때 팔이 몸통 부분 앞에 위치하게 되거나, 한 발이 다른 발 뒤로 가려지는 등의 3D 공간에 대한 이해를 기반으로 한 영상 변형이 일어나게 됩니다. 이러한 `3D-Aware Image Deformation` 기능은 인터렉티브 영상 편집 어플리케이션 등에 유용하게 쓰일 수 있습니다.

![](../../.gitbook/assets/2022spring/4/fig\_1.png)

## 2. Motivation

### Related work

기존에도 3D 공간에 대한 이해를 기반으로 영상 편집을 가능하게 한 기법들이 많이 연구되어 왔지만, 글로벌한 Scene 정보 (예. 뷰포인트, 카메라 파라미터, 조명) 나 깊이 정보를 수정하는 것에 제한되어 있었습니다. Human Pose Transfer 쪽의 연구들은 영상 속 사람의 자세를 변형하는 것을 가능하게 했지만, 사람이 아닌 다른 종류 (예. 만화 캐릭터) 의 영상 속 객체에 대해서는 동작하지 않는다는 한계점이 있었습니다. 3D 모델 기반 변형 기법들은 영상 속 객체 종류에 국한되지 않고 동작한다는 장점이 있지만, 입력 영상에 대응되는 정확한 3D 모델을 필요로 한다는 단점이 존재합니다. 이러한 한계점들을 개선하기 위하여 저희 연구에서는 객체 종류에 국한되지 않고 최대한 자유롭게 영상 변형이 가능한 프레임워크를 고안하는 것을 목표로 하였습니다.

### Idea

객체 종류에 국한되지 않고 최대한 자유롭게 영상 변형이 가능하게 하기 위하여 입력 영상으로부터 복원된 3D Shape에 대해 Handle-Based Deformation Weight \[1] 을 기반으로 영상 변형을 모델링합니다. (1) Tetrahedral Mesh 형태의 3D Shape $$\mathcal{M} = \{\mathcal{V}, \mathcal{F}\}$$ 및 (2) 사용자가 지정한 Deformation Handle $$\{ \mathcal{H}_k \}_{k=1 \cdots m}$$ 이 주어졌을 때, Handle-Based Deformation은 다음과 같이 모델링됩니다:

$$
\mathbf{v}_i' = \sum_{k=1}^{m} w_{k,i} \mathbf{T}_k \mathbf{v}_i.
$$

위 수식에서 $$\mathbf{v}_i$$ 와 $$\mathbf{v}_i'$$ 는 입력 Mesh의 $$i$$번째 Vertex에 대한 변형 전 및 변형 후 위치, $$w_{k,i}$$는 Vertex $$\mathbf{v_i}$$와 Handle $$\mathcal{H_k}$$에 대응되는 Deformation Weight, $$\mathbf{T_k}$$는 사용자가 Handle $$\mathcal{H_k}$$ 에 가하는 Affine Transformation 행렬을 의미합니다.

이 때 사용하는 Handle-Based Deformation Weight \[1] 은 다음과 같은 수식을 통해 계산됩니다:

$$
\underset{\{ \mathbf{w}_k \}_{k=1 \cdots m}}{\mathop{\mathrm{argmin}}} \sum_{k=1}^{m} \frac{1}{2}\; \mathbf{w}_k^T\, A\, \mathbf{w}_k\\ \text{subject to: }\; w_{k,i} = 1 \quad \forall i \quad \text{s.t.} \quad \mathbf{v}_i \in \mathcal{H}_k \\ \qquad \qquad \qquad w_{k,i} = 0 \quad \forall i \quad \text{s.t.} \quad \mathbf{v}_i\in \mathcal{H}_{l, l \neq k} \\ \qquad \qquad \quad \textstyle \sum_{k=1}^{m} w_{k,i}=1, \enspace i=1,\cdots,n, \\ \qquad \qquad \qquad \qquad \qquad \quad 0 \leq w_{k,i} \leq 1, \enspace k=1,\cdots,m, \enspace i=1,\cdots,n.
$$

위 수식에서 각 Deformation Handle에 대한 Deformation Weights $$\mathbf{w}_k = \{w_{k,1}, \cdots, w_{k,n}\}^T$$ 는 Deformation Energy $$A$$에 대한 Constrained Optimization 문제의 해로서 정의됩니다.

해당 Deformation Energy $$A$$는 입력 Mesh의 Shape Laplacian을 이용하여 정의되는데, **2D-to-3D Reconstruction을 통해 복원된 Mesh로부터는 부정확한 Shape Laplacian이 계산된다는 문제가 있습니다**. Shape Laplacian은 Mesh Topology (즉, Mesh Vertex 간의 Edge로서 표현된 연결 관계) 를 기반으로 하여 정의되는데, 2D 영상으로부터 정확한 Mesh Topology 정보를 복원할 수 있는 Topology-Aware Mesh Reconstruction은 여러 어려움들 때문에 아직 풀리지 않은 문제로 남아있습니다. **따라서 저희의 핵심 아이디어는 2D로부터 복원된 3D Shape에 대한 Shape Laplacian 정보를 학습 기반의 기법을 통해 정확하게 예측한 후, 이를 Handle-Based Deformation Weight 계산에 이용하는 것입니다.**

## 3. Method

앞서 언급드렸듯이, 저희는 3D-Aware Image Deformation을 모델링하기 위한 학습 기반의 기법을 제안합니다. 우선 입력 영상에 대하여 3D Reconstruction Method (PIFu \[2]) 를 적용함으로써 영상 속 객체에 대응하는 3D Point Cloud를 예측합니다. (저희는 Mesh Edge 정보가 사용되는 Shape Laplacian 계산을 학습 기반의 기법으로 대체할 것이기 때문에, Mesh가 아닌 Point Cloud 형태의 Shape을 사용합니다.) 다음은 복원된 3D Point Cloud에 대한 Shape Laplacian을 세심하게 설계된 뉴럴넷을 이용하여 예측합니다. 이렇게 예측된 Shape Laplacian을 이용하여 사용자가 임의로 지정한 Deformation Handle에 대한 Handle-Based Deformation Weight \[1]을 계산하고, 이를 통해 모델링 된 3D Deformation을 다시 2D Image Plane에 투사함으로써 3D-Aware Image Deformation을 가능하게 합니다.

지금부터는 저희의 핵심 아이디어인 Point Cloud로부터 Shape Laplacian을 예측하는 네트워크에 대하여 자세하게 소개드리겠습니다. Shape Laplacian의 구성 요소인 Cotangent Laplacian Matrix $$L \in \mathbb{R}^{n \times n}$$ 와 Inverse Mass Matrix $$M^{-1} \in \mathbb{R}^{n \times n}$$ 를 따로 예측하도록 네트워크를 구성한 후, 각 정보에 대한 직접적인 Superivsion을 통하여 네트워크를 학습시킵니다. 아래의 그림을 보시면 알 수 있듯이, 제안 프레임워크는 크게 세 가지의 모듈 - (1) Feature Extraction Module, (2) Cotangent Laplacian Prediction Module, (3) Inverse Mass Prediction Module - 로 구성되어있습니다.

![](../../.gitbook/assets/2022spring/4/fig\_2.png)

**Feature Extraction Module**은 입력 2D 이미지로부터 복원된 3D Point Cloud $$\mathcal{P} = \{ \mathbf{p}_i \}_{i = 1 \cdots n}$$ 를 입력으로 받아 Point Cloud Feature $$\mathcal{F} = \{ \mathbf{f}_i \}_{i = 1 \cdots n}$$ 를 생성합니다. 이 때 $$\mathbf{f}_i \in \mathbb{R} ^ d$$ 은 $$\mathbf{p}_i$$ 에 대응되는 Per-Point Feature를 의미합니다. 모듈의 구조로는 Point Transformer \[3] 를 활용하였습니다.

**Cotangent Laplacian Prediction Module**은 3D Point Cloud $$\mathcal{P} = \{ \mathbf{p}_i \}_{i = 1 \cdots n}$$ 와 Point Cloud Feature $$\mathcal{F} = \{ \mathbf{f}_i \}_{i = 1 \cdots n}$$ 를 입력으로 받아 $$\mathcal{P}$$에 대한 Cotangent Laplacian Matrix $$L \in \mathbb{R}^{n \times n}$$ 를 예측합니다. Cotangent Laplacian의 정의에 따라 $$L$$은 Symmetric하고 매우 Sparse한 특성을 가지고 있는데, $$\mathbf{p}_i$$ 와 $$\mathbf{p}_j$$ 사이의 Edge 연결 관계가 있어야 $$L_{ij}$$이 0이 아닌 값으로 정의되기 때문입니다. 저희는 Point Cloud 내의 각 Point Pair ($$\mathbf{p}_i$$, $$\mathbf{p}_j$$) 를 입력으로 받아 이에 대응되는 Laplacian Matrix의 Element ($$L_{ij}$$) 를 병렬적으로 예측하는 구조를 취하는데, Euclidean Distance가 먼 Point Pair 끼리는 연결 관계가 있을 확률이 적기 때문에 이들을 1차적으로 걸러주는 역할을 합니다. 논문에서 `KNN-Based Point Pair Sampling (KPS)` 으로 지칭하는 부분인데, 각 포인트들에 대하여 $$k$$ 개의 가까운 점들에 대해서만 Point Pair를 구성하는 기법입니다. 이러한 Sampling 기법을 쓰지 않을 경우 Imbalanced Regression Problem이 일어나 네트워크 학습이 잘 되지 않는 현상이 있었습니다.

다음은 `KNN-Based Point Pair Sampling (KPS)` 을 통해 선택된 각 Point Pair Candidate ($$\mathbf{p}_i$$, $$\mathbf{p}_j$$) 에 대하여 `Symmetric Feature Aggregation` 을 수행해줍니다:

$$
\mathbf{g}_{m} = ( \gamma_1(\mathbf{p}_i,\, \mathbf{p}_j), \gamma_2(\mathbf{f}_i, \mathbf{f}_j) ).
$$

위 수식에서 $$\gamma_1(\cdot)$$ 및 $$\gamma_2(\cdot)$$로는 Symmetric Function을 사용하는데, 이는 나중에 예측될 Cotangent Laplacian Matrix의 Symmetry를 보장하기 위함입니다. 해당 함수는 각각 Absolute Difference와 Element-Wise Multiplication으로 구현되었습니다. 이렇게 생성된 Point Pair Feature $$\mathbf{g}_{m}$$ 에 대응되는 Cotangent Laplacian Element $$L_{ij}$$는 다음과 같이 예측됩니다:

$$
L_{ij} = \alpha(\mathbf{g}_{m}) \odot \phi(\mathbf{g}_{m}).
$$

$$\phi(\cdot)$$ 은 Real-Valued Scalar를 출력하는 함수이며 $$\alpha(\cdot)$$ 는 $$L_{ij}$$이 Non-Zero 값일지에 대한 확률을 모델링하는 Weight $$W_{ij} \in [0, 1]$$ 출력 함수입니다. 두 함수는 MLP로 구현되었으며, 최종 $$L_{ij}$$ 값은 두 출력 값의 곱으로서 표현됩니다.

**Inverse Mass Prediction Module**은 3D Point Cloud $$\mathcal{P} = \{ \mathbf{p}_i \}_{i = 1 \cdots n}$$ 와 Point Cloud Feature $$\mathcal{F} = \{ \mathbf{f}_i \}_{i = 1 \cdots n}$$ 를 입력으로 받아 $$\mathcal{P}$$에 대한 Inverse Mass Matrix $$M^{-1} \in \mathbb{R}^{n \times n}$$ 를 예측합니다. Inverse Mass의 정의에 따라 $$M^{-1}$$ 은 Diagonal 하며, $$i$$번째 Digonal Element는 $$\mathbf{p}_i$$의 Volume과 관계된 정보를 담고 있습니다. 따라서 $$\mathcal{P}$$ 내의 각 포인트 $$\mathbf{p}_i$$ 와 대응되는 Per-Point Feature $$\mathbf{f}_i$$ 를 Concatenate 시켜준 후 MLP에 통과시키는 방식을 통해 Inverse Mass Matrix 내의 $$M^{-1}_{ii}$$ Element를 예측합니다.

본 Shape Laplacian 예측 네트워크는 $$L$$, $$W$$, $$M^{-1}$$ 예측 값에 대한 L1-Loss 기반의 Ground Truth Supervision을 통해 학습됩니다. 자세한 Loss 계산 정보는 논문 본문을 참조해주시면 감사하겠습니다.

## 4. Experiment & Result

제안한 3D-Aware Image Deformation 기법의 효과성을 검증하기 위하여 크게 두 종류의 실험을 진행하였습니다. 첫 번째로는 저희가 모델링한 Deformation의 퀄리티를 **정량적**으로 평가하기 위해 3D Point Cloud Deformation 실험을 진행하였습니다. 두 번째로는 저희의 목표 기능인 3D-Aware Image Deformation 결과를 확인하기 위한 **정성적** 평가를 진행하였습니다. 더욱 다양한 실험 결과 (예. Partial Point Cloud Deformation, Ablation Study) 는 논문 본문에서 확인해주시면 감사하겠습니다.

### Experimental setup

* **Dataset**
  * **DFAUST \[4]:** 정량적 평가에 사용된 3D Human Point Cloud 데이터셋입니다.
  * **RenderPeople \[5], Mixamo \[6]:** 정성적 평가에 사용된 3D Human \[5] 및 3D Character \[6] Dataset입니다. 저희의 목적은 Image Deformation의 결과를 확인하는 것이므로, 해당 3D Model들을 렌더링하여 생성한 영상들을 실험에 사용하였습니다.
* **Baselines**
  * 저희의 핵심 아이디어는 Mesh Reconstruction 결과로부터 부정확한 Shape Laplacian이 계산되므로 해당 정보를 학습 기반의 기법을 통해 보다 정확하게 예측하자는 것이었습니다. 따라서, Mesh Reconstruction 기법을 사용하여 Shape Laplacian을 얻은 후 Deformation Weight을 계산하는 상황을 베이스라인으로 설정하였습니다. 저희 실험에서 고려된 Mesh Reconstruction 기법들은 다음과 같습니다:
    * **Screened Poisson Surface Reconstruction (PSR) \[7]**,
    * **Algebraic Point Set Surfaces (APSS) \[8]**,
    * **Ball-Pivoting Algorithm (BPA) \[9]**,
    * **DeepSDF \[10]**,
    * **Deep Geometric Prior (DGP) \[11]**,
    * **Meshing Point Clouds with IntrinsicExtrinsic Ratio (MIER) \[12]**.
  * 또한, 기존의 Point Cloud Laplacian 기법을 이용하여 입력 Point Cloud로부터 Shape Laplacian의 근사 값을 바로 계산하는 기법들도 고려하였습니다:
    * **PCD Laplace (PCDLap) \[13]**,
    * **Nonmanifold Laplacians (NMLap) \[14]**.
     
* **Training Setup**
  * 각 데이터별로 실험에 사용한 세팅이 다르므로, 자세한 사항은 논문 본문 및 Supplementary를 참고해주시면 감사하겠습니다.
* **Evaluation Metric**
  * 저희의 정량적 평가에는 다음과 같은 메트릭이 사용되었습니다:
    * **예측 및 정답 Deformation Weights 간의 L1 Distance (Weight L1)**,
    * **예측 및 정답 Deformed Shape 간의 Chamfer Distance (Shape CD)**,
    * **예측 및 정답 Deformed Shape 간의 Hausdorff Distance (Shape HD)**.

### Result

#### 3D Point Cloud Deformation

아래의 표는 DFAUST \[4] 데이터셋에 대한 정량적 비교 평가 결과를 나타낸 것입니다. 저희가 제안한 기법이 다른 Mesh Reconstruction 베이스라인 기법들을 사용했을 때 보다 더 나은 Shape Deformation 결과를 보이는 것을 알 수 있습니다.

![](../../.gitbook/assets/2022spring/4/fig\_3.png)

위의 결과에 대한 정성적 결과 (아래 그림) 또한 저희 기법이 더욱 자연스러운 Shape Deformation을 모델링할 수 있음을 보여줍니다.

![](../../.gitbook/assets/2022spring/4/fig\_4.png)

#### 3D-Aware Image Deformation

[본 동영상](https://www.youtube.com/watch?v=gHxwHxIZiuM\&feature=youtu.be)은 저희의 3D-Aware Image Deformation 기법을 이용해서 생성한 모션 동영상입니다. Mesh Reconstruction 베이스라인 기법들보다 더욱 자연스러운 Image Deformation을 생성할 수 있음을 보여줍니다.

![](../../.gitbook/assets/2022spring/4/fig\_5.png)

[Interactive Demo](https://jyunlee.github.io/projects/pop-out-motion/demo.html)도 체험해보시기를 바랍니다. 사용자가 직관적인 Deformation Handle (Keypoint)를 이용하여 영상을 변형할 수 있습니다.

## 5. Conclusion

본 연구에서는 Shape Laplacian을 학습함으로써 보다 자연스러운 3D-Aware Deformation을 가능하게하는 프레임워크를 제안하였습니다. 저희가 알기로는 이가 뉴럴넷 기반 기법이 Shape Lapacian 예측에 효과적일 수 있음을 처음으로 보인 연구라고 알고 있습니다. 본 프레임워크를 발전시키기 위한 더욱 다양한 아이디어가 많은데, 기회가 된다면 해당 방향으로 더욱 연구해보고 싶습니다.

### Take-Home Message (오늘의 교훈)

> 제가 개인적으로 이 프로젝트를 통해 배운 교훈은 "끝까지 포기하지 않고 집념을 가지며 연구 문제를 풀자"는 것입니다. 본 프레임워크 개발 단계에서 자잘한 Challenge들이 많았었고, 그 과정 중 원래 진행하려던 연구 방향으로부터 크게 바뀌어 마무리된 부분도 있습니다. 그래도 동작하는 솔루션을 찾아내고 뜻 깊게 프로젝트를 마무리할 수 있어서 개인적으로는 매우 기억에 남는 연구 경험이 되었습니다. 그 과정 중 큰 도움과 조언을 주신 두 지도 교수님께 깊은 감사를 드립니다.

## Author / Reviewer information

### Author

**이지현 (Jihyun Lee)**

* KAIST CS
* I am a first-year Ph.D. student in Computer Vision and Learning Lab at KAIST advised by Prof. Tae-Kyun Kim. I am also currently co-advised by Prof. Minhyuk Sung. My research interests lie in machine learning for 3D computer vision and graphics - especially on humans.
* [\[Google Scholar\]](https://scholar.google.com/citations?user=UaMiOq8AAAAJ\&hl=en) [\[Github\]](https://github.com/jyunlee)

### Reviewer

1. Korean name (English name): Affiliation / Contact information
2. Korean name (English name): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. [Citation of this paper (Bibtex)](https://jyunlee.github.io/projects/pop-out-motion/data/bibtex.txt)
2. [Project website](https://jyunlee.github.io/projects/pop-out-motion/)
3. [Official GitHub repository](https://github.com/jyunlee/Pop-Out-Motion)
4. Citation of related work
   1. _Alec Jacobson, Ilya Baran, Jovan Popovic, and Olga Sorkine. Bounded biharmonic weights for real-time deformation. In SIGGRAPH, 2011._
   2. _Shunsuke Saito, Zeng Huang, Ryota Natsume, Shigeo Morishima, Angjoo Kanazawa, and Hao Li. PIFu: Pixel-aligned implicit function for high-resolution clothed human digitization. In ICCV, 2019._
   3. _Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip HS Torr, and Vladlen Koltun. Point transformer. In ICCV, 2021._
   4. _Federica Bogo, Javier Romero, Gerard Pons-Moll, and Michael J. Black. Dynamic FAUST: Registering human bodies in motion. In CVPR, 2017._
   5. _Renderpeople GmbH. RenderPeople. https://renderpeople.com/._
   6. _Adobe Systems Inc. Mixamo. https://www.mixamo.com._
   7. _Michael Kazhdan and Hugues Hoppe. Screened poisson surface reconstruction. ACM TOG, 2013._
   8. _Gael Guennebaud and Markus Gross. Algebraic point set surfaces. In SIGGRAPH, 2007._
   9. _Fausto Bernardini, Joshua Mittleman, Holly Rushmeier, Claudio Silva, and Gabriel Taubin. The ball-pivoting algorithm for surface reconstruction. IEEE TVCG, 1999._
   10. _Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. DeepSDF: Learning continuous signed distance functions for shape representation. In CVPR, 2019._
   11. _Francis Williams, Teseo Schneider, Claudio Silva, Denis Zorin, Joan Bruna, and Daniele Panozzo. Deep geometric prior for surface reconstruction. In CVPR, 2019._
   12. _Minghua Liu, Xiaoshuai Zhang, and Hao Su. Meshing point clouds with predicted intrinsic-extrinsic ratio guidance. In ECCV, 2020._
   13. _Mikhail Belkin, Jian Sun, and Yusu Wang. Constructing laplace operator from point clouds in rd. In Proc. Annu. ACM-SIAM Symp. Discrete Algorithms, pages 1031–1040. SIAM, 2009._
   14. _Nicholas Sharp and Keenan Crane. A laplacian for nonmanifold triangle meshes. In SGP, 2020._
