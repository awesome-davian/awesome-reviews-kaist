---
description: Lee et al. / Pop-Out Motion - 3D-Aware Image Deformation via Learning the Shape Laplacian / CVPR 2022
---

# Pop-Out Motion \[Korean\]

안녕하세요. 본 포스팅에서는 올해 CVPR에 발표될 Pop-Out Motion이라는 논문을 소개드리고자 합니다. 자연스러운 3D-Aware Image Deformation을 위한 학습 기반의 파이프라인을 제안한 논문이며, `3D Vision`, `Shape Deformation`, `2D-to-3D Reconstruction` 등의 키워드에 관심이 있으신 분들이라면 [논문 본문](https://arxiv.org/pdf/2203.15235.pdf) 및 [프로젝트 페이지](https://jyunlee.github.io/projects/pop-out-motion/)를 구경해주시면 감사하겠습니다. 해당 논문은 제가 1저자로 참여하였으며, KAIST 전산학부의 성민혁교수님과 김태균교수님께서 지도해주셨습니다. (좋은 연구 지도를 해주신 두 교수님께 감사드립니다.)

##  1. Problem Definition

본 논문은 `3D-Aware Image Deformation` 이라는 문제를 해결하고자 합니다. 사용자가 이미지 내의 객체 모양을 자연스럽게 변형하는 것을 가능하게 하되, 2D 영상의 피사체가 마치 3D 공간에 존재하는 것과 같이 변형할 수 있도록 하는 것이 목표입니다. 이 때 직관적인 이미지 수정을 위하여 사용자가 키포인트 등의 `Deformation Handle` 을 매개체로서 사용할 수 있도록 합니다. 아래의 그림 예시를 보시면, 사용자가 이미지에 키포인트들 (파란색 원 표시) 을 지정하고 그 중 하나를 선택하여 움직일 경우 (빨간색 화살표 표시), 그림 내의 사람 객체 모양이 그에 맞추어 자연스럽게 변형되는 것을 볼 수 있습니다. 이 때 팔이 몸통 부분 앞에 위치하게 되거나, 한 발이 다른 발 뒤로 가려지는 등의 3D 공간에 대한 이해를 기반으로 한 영상 변형이 일어나게 됩니다. 이러한 `3D-Aware Image Deformation` 기능은 인터렉티브 영상 편집 어플리케이션 등에 유용하게 쓰일 수 있습니다.

* Insert Figure 5 of the paper

## 2. Motivation

### Related work

기존에도 3D 공간에 대한 이해를 기반으로 영상 편집을 가능하게 한 기법들이 많이 연구되어 왔지만, 글로벌한 Scene 정보 (예. 뷰포인트, 카메라 파라미터, 조명) 나 깊이 정보를 수정하는 것에 제한되어 있었습니다. Human Pose Transfer 쪽의 연구들은 영상 속 사람의 자세를 변형하는 것을 가능하게 했지만, 사람이 아닌 다른 종류 (예. 만화 캐릭터) 의 영상 속 객체에 대해서는 동작하지 않는다는 한계점이 있었습니다. 3D 모델 기반 변형 기법들은 영상 속 객체 종류에 국한되지 않고 동작한다는 장점이 있지만, 입력 영상에 대응되는 정확한 3D 모델을 필요로 한다는 단점이 존재합니다. 이러한 한계점들을 개선하기 위하여 저희 연구에서는 객체 종류에 국한되지 않고 최대한 자유롭게 영상 변형이 가능한 프레임워크를 고안하는 것을 목표로 하였습니다.

### Idea

객체 종류에 국한되지 않고 최대한 자유롭게 영상 변형이 가능하게 하기 위하여 입력 영상으로부터 복원된 3D Shape에 대해 Handle-Based Deformation Weight [1] 을 기반으로 영상 변형을 모델링합니다. (1) Tetrahedral Mesh 형태의 3D Shape $$\mathcal{M} = \{\mathcal{V}, \mathcal{F}\}$$ 및 (2) 사용자가 지정한 Deformation Handle $$\{ \mathcal{H}_k \}_{k=1 \cdots m}$$ 이 주어졌을 때, Handle-Based Deformation은 다음과 같이 모델링됩니다:

$$
  \mathbf{v}_i' = \sum_{k=1}^{m} w_{k,i} \mathbf{T}_k \mathbf{v}_i.
$$

위 수식에서 $$\mathbf{v}_i$$ 와 $$\mathbf{v}_i'$$ 는 입력 Mesh의 $$i$$번째 Vertex에 대한 변형 전 및 변형 후 위치, $$w_{k,i}$$는 Vertex $$\mathbf{v_i}$$와 Handle $$\mathcal{H_k}$$에 대응되는 Deformation Weight, $$\mathbf{T_k}$$는 사용자가 Handle $$\mathcal{H_k}$$ 에 가하는 Affine Transformation 행렬을 의미합니다.

이 때 사용하는 Handle-Based Deformation Weight [1] 은 다음과 같은 수식을 통해 계산됩니다:

$$
  \underset{\{ \mathbf{w}_k \}_{k=1 \cdots m}}{\mathop{\mathrm{argmin}}} \sum_{k=1}^{m} \frac{1}{2}\; \mathbf{w}_k^T\, A\, \mathbf{w}_k
$$

$$
  \text{subject to: }\;  w_{k,i} = 1 \quad \forall i \quad \text{s.t.} \quad \mathbf{v}_i \in \mathcal{H}_k  \\
     \qquad \qquad \qquad w_{k,i} = 0 \quad \forall i \quad \text{s.t.} \quad \mathbf{v}_i\in \mathcal{H}_{l, l \neq k} \\
     \qquad \qquad \quad \textstyle \sum_{k=1}^{m} w_{k,i}=1, \enspace i=1,\cdots,n, \\
     \qquad \qquad \qquad \qquad \qquad \quad 0 \leq w_{k,i} \leq 1, \enspace k=1,\cdots,m, \enspace i=1,\cdots,n.
$$

위 수식에서 각 Deformation Handle에 대한 Deformation Weights $\mathbf{w}_k = \{w_{k,1}, \cdots, w_{k,n}\}^T$ 는 Deformation Energy $$A$$에 대한 Constrained Optimization 문제의 해로서 정의됩니다. 

해당 Deformation Energy $A$는 입력 Mesh의 Shape Laplacian을 이용하여 정의되는데, **2D-to-3D Reconstruction을 통해 복원된 Mesh로부터는 부정확한 Shape Laplacian이 계산된다는 문제가 있습니다**. Shape Laplacian은 Mesh Topology (즉, Mesh Vertex 간의 Edge로서 표현된 연결 관계) 를 기반으로 하여 정의되는데, 2D 영상으로부터 정확한 Mesh Topology 정보를 복원할 수 있는 Topology-Aware Mesh Reconstruction은 여러 어려움들 때문에 아직 풀리지 않은 문제로 남아있습니다. **따라서 저희의 핵심 아이디어는 2D로부터 복원된 3D Shape에 대한 Shape Laplacian 정보를 학습 기반의 기법을 통해 정확하게 예측한 후, 이를 Handle-Based Deformation Weight 계산에 이용하는 것입니다.**

## 3. Method

앞서 언급드렸듯이, 저희는 3D-Aware Image Deformation을 모델링하기 위한 학습 기반의 기법을 제안합니다. 우선 입력 영상에 대하여 3D Reconstruction Method (PIFu [2]) 를 적용함으로써 영상 속 객체에 대응하는 3D Point Cloud를 예측합니다. (저희는 Mesh Edge 정보가 사용되는 Shape Laplacian 계산을 학습 기반의 기법으로 대체할 것이기 때문에, Mesh가 아닌 Point Cloud 형태의 Shape을 사용합니다.) 다음은 복원된 3D Point Cloud에 대한 Shape Laplacian을 세심하게 설계된 뉴럴넷을 이용하여 예측합니다. 이렇게 예측된 Shape Laplacian을 이용하여 사용자가 임의로 지정한 Deformation Handle에 대한 Handle-Based Deformation Weight [1]을 계산하고, 이를 통해 모델링 된 3D Deformation을 다시 2D Image Plane에 투사함으로써 3D-Aware Image Deformation을 가능하게 합니다.

지금부터는 저희의 핵심 아이디어인 Point Cloud로부터 Shape Laplacian을 예측하는 네트워크에 대하여 자세하게 소개드리겠습니다. 저희는 Shape Laplacian의 구성 요소인 Cotangent Laplacian Matrix $L \in \mathbb{R}^{n \times n}$ 와 Inverse Mass Matrix $M^{-1} \in \mathbb{R}^{n \times n}$ 를 따로 예측하도록 네트워크를 구성한 후, 각 정보에 대한 직접적인 Superivsion을 통하여 네트워크를 학습시킵니다. 아래의 그림을 보시면 알 수 있듯이, 저희의 프레임워크는 크게 세 가지의 모듈 -- (1) Feature Extraction Module, (2) Cotangent Laplacian Prediction Module, (3) Inverse Mass Prediction Module -- 로 구성되어있습니다. 

* Insert Figure 2

**Feature Extraction Module**은 입력 2D 이미지로부터 복원된 3D Point Cloud $$\mathcal{P} = \{ \mathbf{p}_i \}_{i = 1 \cdots n}$$ 를 입력으로 받아 Point Cloud Feature $$\mathcal{F} = \{ \mathbf{f}_i \}_{i = 1 \cdots n}$$ 를 생성합니다. 이 때 $$\mathbf{f}_i  \in \mathbb{R} ^ d$$ 은 $$\mathbf{p}_i$$ 에 대응되는 Per-Point Feature를 의미합니다. 모듈의 구조로는 Point Transformer [3] 를 활용하였습니다.

**Cotangent Laplacian Prediction Module**은 3D Point Cloud $\mathcal{P} = \{ \mathbf{p}_i \}_{i = 1 \cdots n}$ 와 Point Cloud Feature $\mathcal{F} = \{ \mathbf{f}_i \}_{i = 1 \cdots n}$ 를 입력으로 받아 $\mathcal{P}$에 대한 Cotangent Laplacian Matrix $L \in \mathbb{R}^{n \times n}$ 를 예측합니다. Cotangent Laplacian의 정의에 따라 $L$은 Symmetric하고 매우 Sparse한 특성을 가지고 있는데, $\mathbf{p}_i$ 와 $\mathbf{p}_j$ 사이의 Edge 연결 관계가 있어야 $L_{ij}$이 0이 아닌 값으로 정의되기 때문입니다. 저희는 Point Cloud 내의 각 Point Pair ($\mathbf{p}_i$, $\mathbf{p}_j$) 를 입력으로 받아 이에 대응되는 Laplacian Matrix의 Element ($L_{ij}$) 를 병렬적으로 예측하는 구조를 취하는데, Euclidean Distance가 먼 Point Pair 끼리는 연결 관계가 있을 확률이 적기 때문에 이들을 1차적으로 걸러주는 역할을 합니다. 논문에서 `KNN-Based Point Pair Sampling (KPS)` 으로 지칭하는 부분인데, 각 포인트들에 대하여 $k$ 개의 가까운 점들에 대해서만 Point Pair를 구성하는 기법입니다. 이러한 Sampling 기법을 쓰지 않을 경우 Imbalanced Regression Problem이 일어나 네트워크 학습이 잘 되지 않는 현상이 있었습니다.

다음은 `KNN-Based Point Pair Sampling (KPS)` 을 통해 선택된 각 Point Pair Candidate ($\mathbf{p}_i$, $\mathbf{p}_j$) 에 대하여 `Symmetric Feature Aggregation` 을 수행해줍니다: 
$$ \mathbf{g}_{m} = ( \gamma_1(\mathbf{p}_i,\, \mathbf{p}_j), \gamma_2(\mathbf{f}_i, \mathbf{f}_j) ).$$ 
위 수식에서 $\gamma_1(\cdot)$ 및 $\gamma_2(\cdot)$로는 Symmetric Function을 사용하는데, 이는 나중에 예측될 Cotangent Laplacian Matrix의 Symmetry를 보장하기 위함입니다. 해당 함수는 각각 Absolute Difference와 Element-Wise Multiplication으로 구현되었습니다. 이렇게 생성된 Point Pair Feature $\mathbf{g}_{m}$ 에 대응되는 Cotangent Laplacian Element $L_{ij}$는 다음과 같이 예측됩니다: 
$$ L_{ij} = \alpha(\mathbf{g}_{m}) \odot \phi(\mathbf{g}_{m}). $$
$\phi(\cdot)$ 은 Real-Valued Scalar를 출력하는 함수이며 $\alpha(\cdot)$ 는 $L_{ij}$이 0일지에 대한 확률을 모델링하는 Weight $W_{ij} \in [0, 1]$ 출력 함수입니다. 두 함수는 MLP로 구현되었으며, 최종 $L_{ij}$ 값은 두 출력 값의 곱으로서 표현됩니다.

**Inverse Mass Prediction Module**은 3D Point Cloud $\mathcal{P} = \{ \mathbf{p}_i \}_{i = 1 \cdots n}$ 와 Point Cloud Feature $\mathcal{F} = \{ \mathbf{f}_i \}_{i = 1 \cdots n}$ 를 입력으로 받아 $\mathcal{P}$에 대한 Inverse Mass Matrix $M^{-1} \in \mathbb{R}^{n \times n}$ 를 예측합니다. Inverse Mass의 정의에 따라 $M^{-1}$ 은 Diagonal 하며, $i$번째 Digonal Element는 $\mathbf{p}_i$의 Volume과 관계된 정보를 담고 있습니다. 따라서 $\mathcal{P}$ 내의 각 포인트 $\mathbf{p}_i$ 와 대응되는 Per-Point Feature $\mathbf{f}_i$ 를 Concatenate 시켜준 후 MLP에 통과시키는 방식을 통해 Inverse Mass Matrix 내의 $M^{-1}_{ii}$ Element를 예측합니다.

본 Shape Laplacian 예측 네트워크는 $L$, $W$, $M^{-1}$ 예측 값에 대한 L1-Loss 기반의 Ground Truth Supervision을 통해 학습됩니다. 자세한 Loss 계산 정보는 논문 본문을 참조해주시면 감사하겠습니다.

## 4. Experiment & Result

{% hint style="info" %}
If you are writing **Author's note**, please share your know-how \(e.g., implementation details\)
{% endhint %}

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

### Author

**이지현 \(Jihyun Lee\)** 

* KAIST CS
* I am a first-year Ph.D. student in Computer Vision and Learning Lab at KAIST advised by Prof. Tae-Kyun Kim. I am also currently co-advised by Prof. Minhyuk Sung. My research interests lie in machine learning for 3D computer vision and graphics - especially on humans.
* [\[Google Scholar\]](https://scholar.google.com/citations?user=UaMiOq8AAAAJ&hl=en) [\[Github\]](https://github.com/jyunlee)

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

