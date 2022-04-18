## Description

* Xin Lin et al. / GPS-Net: Graph Property Sensing Network for Scene Graph Generation / CVPR 2020

---



##  1. Problem definition

Scene Graph Generation (SGG) 는, 이미지를 입력으로 받았을 때 이를 그래프로 바꾸어주는 Task 입니다.


![1](/.gitbook/assets/4/figure1.png) 


그림1은 SGG 의 일련의 과정을 나타내고 있습니다. 사람과 말이 있는 이미지를 입력으로 받아 모델이 그래프를 생성합니다.

이 때 우리가 생성하고 싶은 그래프 G는 **_V, E, R, O_**   총 4가지 컴포넌트를 가지고 있습니다.  

**_V_** 는 노드, object detector의 proposal 로 구성되며  **_E_** 는 edge로, 연관이 있는 object 끼리 연결이 됩니다. 

또한 SGG 에서는 각 노드와 엣지의 label 의 class 가 무엇인지 구분하는 classification Task도 수행합니다.

**_R_** 은 Edge의 Relation class를 뜻하며, **_O_** 은 Object의 class를 뜻합니다.

따라서 최종 얻은 Graph 는  

<object, predicate, subject> (사람, 먹이주다, 말) 와 같은 triplet 의 조합으로 이루어지게 됩니다.



그러면 위의 식으로 부터    

**P(_V | I_ ) - object detector**

**P(_E | V, I_ ) - relation proposal netowrk**

**P(_R, O | V, E, I_ ) - Classification models for entity and predicate.**  

이 3가지를 모델링 하면 저희는 Scene Graph 를 생성할 수 있는 문제를 정의할 수 있게됩니다.





## 2. Motivation

그렇다면 Scene Graph Generation 할 때 기존에 사용했던 모델은 무엇들이 있으며, 기존 모델들의 문제는 무엇이었을지 짚어보아야 합니다.

여기서는 Previous Works 에 대해 간단한 요약과, 저자의 Idea를 살펴보겠습니다.

### Related work

*Knowledge Graph Embedding*
VTransE, DTransE [2], [3] 와 같은 모델들은 Knowledge Graph Embedding Method를 사용하여, object, predicate, subject 를
동일한, 또는 각각의 Latent Space 에 Mapping 합니다. 그 hidden representation의 유사성을 측정하여 Scene Graph Generation 에 적용한
framework 입니다. 하지만, 이 모델들은 주변 context는 고려하지 않고, 오직 각각의 object의 embedding 만을 보고 graph를 생성하기 때문에
이미지 상에 존재하는 정보를 충분히 이용하지는 못합니다.

*Scene Graph Generation*
Neural-Motif [4] 은 주변 컨택스트, 또는 entity A(subject), entity B (subject) 사이의 관계를 예측하기 위해 주변 entity 의 feature를
사용합니다. 이를 위해 bi-directional RNN 과 같은 sequnce 모델을 사용합니다. Graph R-CNN [5] 은 Neural Motif 를 좀 더 효율적으로 그래프
자체에서 모델링하기 위에 제안되었습니다. GNN을 사용하여 주변 context를 보다 효율적으로 결합하고, 이용하여 Scene Graph Generation을 하게 됩니다.
하지만, Graph R-CNN 또한 SGG 를 위한 최적의 framework 라고 할 수 없습니다. 그 이유는 다음 세션에서 GPS-Net의 Idea와 함께 설명하겠습니다.

### Idea


![2](/.gitbook/assets/4/figure2.png) 

그림 2는 GPS-Net 저자의 Motivation을 명확히 보여주는 그림입니다. 여기서 저자는 3가지 중요한 사실을 지목합니다.

**첫째, 모델은 방향성을 인식해야한다. (b)**

기존 Graph Neural Network (GNN) 을 일괄 적용할 경우에는 triplet 의 방향을 인식하지 못합니다. 따라서 방향성을 고려하는 

_Direct aware Message Passing Neural Network_ (DMP)을 제안합니다.


**둘째, degree가 높은 node가 중요하다 (c)**

SGG는 Image 를 여러개의 모듈을 거쳐 Graph 를 생성하게 됩니다. 이 과정에서 hub node (degree가 높은 node)가 잘못 clasfficiation

되어 있다면, GNN을 통해 주변노드를 업데이트할 때, 잘못된 정보를 많이 퍼뜨리게 될 것 입니다. 따라서, degree가 더 높은 노드를 집중적으로

학습하는 _Node Priority Sensitive Loss_를 제안합니다.

**셋째, SGG는 Imblanced Classification 문제이다**

subject, object 사이의 Predicate 를 예측할 때 Predicate class 는 long-tail distribution 을 따릅니다.

쉽게 설명하자면, 'on', 'has' 와 같은 predicate는 정말 빈번히 등장합니다. 반면, standing in, feeding 와 같은 디테일한 행동들은

상대적으로 적게 등장하는 label class 입니다. 따라서 on, has 위주로 모델이 예측하게 된다면, 높은 performance 를 기록할 수 있습니다.

*하지만 on, has 가 많이 등장하는 Scene Graph가 아닌, 높은 퀄리티의 정보를 담을 Scene Graph를 생성하는 것이 저희의 궁극적인 목표입니다 !!*



## 3. Method

GPS-Net은 Object Detector를 Faster R-CNN 의 구조를 그대로 가져오게 됩니다. Pretrained detector를 통해 Object proposal이 생성해내고, 각각의 box로 부터 visual feature, class logits, box 의 위치를 추출합니다. 
box i에서 이와 같은 feature 들을 묶어 x_i 라고 칭하겠습니다.

또, 기존 Graph R-CNN 과 달리 추가적으로 2개의 box를 union한, union feature u_ij 도 추출합니다.

제안된 feature를 x_1,.., x_n 과 u_12, ..., u_ij, ... 를 얻었다면 앞서 언급한 GPS-Net의 architecture에 통과시킵니다.

### 1. Direction-aware Message Passing


![3](/.gitbook/assets/4/figure3.png) 


그림3은 기존에 사용하는 Message Passing Network 들의 구조 (a), (b)와 제안된 DMP 구조 (c) 를 가져온것 입니다.
여기서 x_i 는 업데이트하고자 하는 Target, x_j는 Target을 업데이트 하기 위한 Neighbor의 Feature vector이며, u_ij는 두 bounding box i, j
의 union box의 feature를 나타냅니다.
Message Passing Network의 핵심은 Message를 어떻게 만드느냐 입니다.

먼저, (a)의 경우 Target과 Neighbor의 Feature를 단순히 concat 하여 Weight를 곱해준 것이 메세지입니다. 
이 메세지를 Transforemer에 통과시키고, 마지막으로 자신의 Feature와 다시 업데이트 하게 됩니다.

(b)의 경우  Message를 오직 Neighbor의 Feature만 가지고 업데이트를 합니다. 그리고 마찬가지로 Transforemr
Layer에 통과시킨후 자기자신의 Feature와 업데이트를 하게 됩니다.

하지만, SGG의 Framework에서 이것은 문제가 됩니다. SGG를 수행하기 위해, GNN을 사용하여 주변 Object들의 Feature를
모으게 되는데, 이 때 중요한 사실은 GNN에 사용할 Graph가 Clean하지 않고 Noise하다는 것입니다. 다시 말하면, 이 GNN에 사용할 그래프는
Object Detector Proposal Boxe들의 연결관계를 임의로 정해둔 것 입니다. 따라서 이 그래프는 방향성 조차 애매한 상황입니다.

(c)는 이러한 상황을 다루기 위해 양쪽 방향성을 다 고려하는 Message를 만들고자 했고, 양쪽 방향성을 다 고려하기 위해 다음 두가지 차이를 두었습니다.

차이를 보자면

####1. MPNN Layer 에 u_ij 라는 edge feature 가 같이 도입되었다.
	
	u_ij 는 앞서 말했듯 union box 로부터 뽑은 visual feature 입니다. 기존 Graph-RCNN 사용하지 않았던 추가적인 feature를 사용한 것인데,
	이는 relation 을 예측할 때 보다 넓은 receptive field를 활용하게 됩니다. 또한 GNN 의 구조적 특성상 layer를 많이 쌓을 수록 주변으로 정보를 propagation 을 하기 때문에	
	image를 예측할 때의 context 도 더 잘 반영할 수 있게 됩니다. 
	예를 들면, 사람(Object)와 말(Object) 사이의 Relation을 예측 할 때, 사람 손과 말이 겹치는 부분의 visual feature가 도움이 될 것 입니다. (Union Box의 역할)

####2. MPNN Layer의 Element wise product를 Kronecker Prdouct로 대체하었다.

	구조를 보았을 때 (a) 는 x_i, x_j를 단순 concat 하였고 (b) 는 destination node(x_j) 의 정보만을 추출하여 Message passing을 수행하게 됩니다.
	반면, 저자가 제안한 DMP는 (x_i, x_j, u_ij) 를 통해 attention weight 를 추출하고, destination node 에 곱을 하여 Message passing을 수행합니다.
	즉, (c) 는 feature가 들어오는 방향에 따라 각각의 attention weight가 달라지며, 방향이 달라지면 destination node의 업데이트할 양이 조절되도록 합니다. 
	이를 Kronecker Product로 구현 하였는데, 이는 MPNN 구조가 Direction-aware를 가능하게 했다고 합니다.

### 2. Node Prioirty Sensitive Loss

저자는 Node 의 priority 에 따라서 다른 update를 해줘야 한다고 언급하고 있습니다. 
SGG Task 자체가 Faster R-CNN, Graph Generation, Object classification, Edge Classification 과 같이 많은 Task들을 순차적으로 진행하는데, 수행하는 Task가 많고 또 모델의 크기가 크다보니 중간에 잘못된 예측을 할 수 있습니다.

가령, Faster R-CNN 에서 개를 고양이라고 잘못 Detect 했다고 가정해보겠습니다. 그러면 그 뒤에 있는 모든 MPNN Layer 는 잘못된 node feature를 propagate 할 것 입니다. 그러한 노드가 degree가 높은 hub node라면 ?
잘못된 정보가 더 많이 퍼질 것 입니다. 이러한 상황을 컨트롤하기 위해 Node sensitive loss를 제안한 것으로 보입니다.



![4](/.gitbook/assets/4/figure4.png) 

그림 4는 제안된 로스의 수식입니다.

세타는 priority 를 나타내는데, 전체 triplet 의 수 중에서 해당 node를 거치는 triplet의 수를 나타냅니다. 즉 자신을 거치는 triplet이 많다면 priority가 높다고 볼 수 있겠습니다. 이를 degree 가 높은 node로 이해해보겠습니다. 

그 다음, priority를 기반으로 focusing factor를 계산하게 됩니다. 세타가 0과 1사이의 수이므로,  세타가 클수록 focusing factor가 작아지게 됩니다.

마지막은 Focal Loss 입니다. gamma 값은 node 에 따라 바뀌게 되는데요. 우선 gamma값이 1이라면, binary cross entropy 의 loss 형태를 떠올릴 수 있을 것 입니다.
만약 gamma 값이 크다면, Loss가 작을 것입니다. 그렇다면 상대적으로 해당 node에 대해서 gradient update를 적게 할 것입니다. 반대로 gamma 작다면 Loss가 상대적으로 클 것 이고, 해당 node에 대해 더 많은 update를 할 것 입니다.

즉,
	**degree가 높다 -> focusing factor(gamma)가 작다 -> Loss가 크다 -> update 더 많이 수행.**
	**degree가 낮다 -> focusing factor(gamma)가 크다 -> Loss가 작다 -> update 더 적게 수행.**

Degree가 높은 node 에 대해 더욱 중점적으로 학습할 수 있게 됩니다.


### Adaptive Reasoning Module

마지막으로, Loss를 SGG 의 상황에 맞춰 Adapation 할 수 있는 장치들을 더해주게 됩니다.

바로 **Frequency Softening** 과 **Bias Adaptation** 인데요.


![5](/.gitbook/assets/4/figure5.png) 


그림 5를 통해 수식을 보시면, 바로 이해하실 수 있습니다. 

Bias Adaptation 은 training data에 등장하는 label distribution 의 패턴을 bias로서 넣어주자는 것 입니다.

이 아이디어는 Neural-motifs [3] 에서 등장한 개념인데요. 특정 triplet 패턴이 많이 등장하면, 그것을 예측하도록 유도하는 bias를 더해준다고 보시면 될 것 같습니다.

Bias Adaptation의 앞쪽의 fusion term은 DMP를 통해 얻은 feature 들을 통해 class를 예측하는 것입니다.
그 뒤에 더해진 d*p term 이 frequency softening 부분이라고 볼 수 있겠습니다.
주어진 union feature u_ij 를 통해, 많이 등장했던 triplet인지 판단하여 d를 계산하고, training data의 distribution이 반영된 p를 곱해주게 됩니다.
이렇게 함으로서, 많이 등장한 패턴에 대해 적합한 bias를 더해준 것으로 볼 수 있습니다.

다만 여기서는 Frequency Softening 의 구조를 조금 변형하여 사용하는데요. SGG를 수행하는 visual genome dataset이  long-tail shaped class distribution 을 가지고 있기 때문입니다. GPS-Net에서는 이러한 long-tail distribution을 고려하여 Frequency softening 하기 위해서 log-softmax function을 사용하여
적은 label 에 대해서도 등장할 가능성을 조금 열어두게 합니다.


## 4. Experiment & Result


### Experimental setup

SGG Framework 에서 Data 는 Visual genome 을 사용하는 것이 정형화 되어있습니다. 이때 예측하는 metric 은 Recall@K 이며 *SGDET*, *SGCLS*, *PREDCLS* 3가지 Task를 비교합니다.


*SGDET* -   Image -> Object detect / object classification / predicate classification  수행.

	전형적으로 이미지가 주어졌을 때, Graph를 생성하는 태스크 입니다. 세가지 중에 가장 어려운 태스크라고 볼 수 있으며,
	말 그대로 이미지가 그래프 자체로 변환하는 맵핑을 배우는 것 입니다. 따라서, Object Detector, Graph Edge Prediction, Object, relation classifier의
	모든 성능을 다 체크하는 것이라고 할 수 있겠습니다.

*SGCLS* - Ground Truth Box -> object classification / Predicate classification 수행

	이미지가 주어지고, 실제 Bounding Box가 주어졌을 때 Scene Graph를 만드는 태스크 입니다. Object Detector에 Dependent하지 않기 때문에
	위의 SGDET Task보다는 살짝 쉬워진 Task 입니다. 오직 Object, Predicate Classifer의 성능을 측정하는 기준 입니다.

	
*PREDCLS* - Ground Truth Box, object category -> Predciate Classification 수행

	마지막으로, 이미지가 주어지고, 실제 Bounding Box와 Object의 Classs까지 무엇인지 주어졌을 때 Scene Graph를 만드는 태스크 입니다. 
	Object Detector에 Dependent하지 않고, Object의 Class도 이미 알기 때문에 가장 쉬운 태스크입니다. 오직, Predicate Classifer의 성능을 측정하는 기준 입니다.





### Result

![6](/.gitbook/assets/4/table1.png) 

표1 은 Recall@K 를 K=20, 50, 100 에 따라 각각의 Task에 비교한 것을 볼 수 있습니다. 모델 옆의 도형은 동일한 object detector 를 사용한 것 끼리
묶은 것 입니다. 보시는 바와 같이 GPS-Net은 어떤 object detector를 사용했던간에 기존의 모델들을 모든 Task에서 압도하고 있네요.


![7](/.gitbook/assets/4/table2.png) 

표2는 각각의 class별 Recall@K를 따로 구하고, 모든 class의 평균을 취한 mR@K 를 비교하였습니다. 또한 각각의 class 별로 performance gain이 얼만큼
일어 났는지 비교하였습니다. 확실히 mR@K가 증가하고, 우측 그림을 보았을 때 long-tail class 에 대해서 성능이 향상되었음을 확인할 수 있습니다.


![8](/.gitbook/assets/4/table3.png) 


표3 (a), (b)는 각각 모델 component 들에 대해 ablation study를 한 결과 입니다. 표(a) 를 살펴보면 SGDET와 SGCLS의 Task에서는  기존의 모델에 **DMP를 추가하였 때** 가장 큰 performance gain이 있었음을 확인할 수 있습니다. SGG에서 방향이 얼마나 중요한지 살펴볼 수 있는 대목입니다.
또한 NPS, ARM 또한 조금씩의 performance gain에 도움을 주었습니다. DMP 만큼은 아니지만, 종합적으로 보았을 때 기존에 비해 성능 개선에 도움을 주었습니다. 반면, PREDCLS Task에서는 ARM이 가장 큰 개선을 주었습니다. ARM이 타겟하는 파트가 PREDCLS와 연관이 가장 큰 만큼,
이 TASK 에서는 DMP보다 더 많은 성능 개선을 보였습니다.

표(b) 에서는 DMP의 성능을 stack을 했을때, 기존 MP 와의 비교를 실험하였고, 또한 NPS에서 node focusing을 얼마나 할것인지를 조절하는 hyperparmaeter, mu에 따른 성능 비교를 진행하였습니다. 

전반적으로 Baseline과의 실험 비교와, 제안된 모델에 대한 ablation study가 착실하게 잘 이루어진 논문으로서 이해하기 쉬우면서도 실험을 통한 가설의 검증이 잘 진행되었다고 보여집니다.


## 5. Conclusion

본 GPS-Net 에서는 Scene Graph Generation 에서 다루어야할 새로운 문제들을 제시하였습니다. 모델이 object 간의 방향을 인지하여야하고, 각 node별 중요도가 다르다는 점을 인식하게 할 수 있는 모델을 제안하였습니다.
적절한 실험을 통해 이것들을 해소하는 것을 보였습니다. 다만, 실험 구성 면에서 appendix에 u_ij 라는 feature 에 대한 역할 규명을 보여주었으면 더 좋을 것 같습니다. 모델 구조 때문인지 저 feature를 추가함으로서 얻은 성능향상인지 구분할 수 없기 때문입니다.
Recall을 살펴보았을 때, image to graph 를 하는 task들이, 실제 생활에서 사용하기에는 아직 너무도 낮은 수치라고 생각됩니다. 

### Take home message 

> SGG 문제 상에서 존재할 수 있는 간단한 가설을 입증하기 위해, 무수히 많은 노력, 실험을 한 논문이라고 보여집니다.
> 쉬운 가설 하나를 세우는 것은 찰나이지만, 입증하기 위해서는 정말 많은 노력과 시간이 필요함을 느끼고, 본받습니다.


### Author

**윤강훈 \(Kanghoon Yoon\)** 

* Affiliation \(KAIST Industrial Engineering Department\)
* \(optional\) ph.D students in DSAIL

## Reference & Additional materials

1. Visual translation embedding network for visual relation detection
2. Representation learning for scene graph completion via jointly structural and visual embedding
3. Neural Motifs: Scene Graph Parsing with Global Context
4. Graph R-CNN for Scene Graph Generation.
5. GPS-net: Graph property sensing network for scene graph generation
