---
description: Lam et al. / Finding Representative Interpretations on Convolutional Neural Networks / ICCV 2021
---

# Finding Representative Interpretations on Convolutional Neural Networks \[Kor]

[English version](iccv-2021-interpretationCNN-eng.md) of this article is available.

## 1. Problem definition

최근 다양한 영역에서 딥러닝 기반의 인공지능 모델들이 성공적인 성능을 보이고 있지만, **딥러닝 모델의 의사결정 과정에 대한 해석은 아직 부족**하다. 공정 시스템이나 헬스케어처럼 각각의 판단이 매우 중요한 영역에서는 해석성이 부족한 모델은 신뢰할 수 없어 사용하는데 제약이 있다. 때문에 의사결정 과정에 대한 충분한 해석성이 제공되어야 딥러닝 모델들을 신뢰가능하게 만들 수 있을 것이다.

이 논문에서는 많은 딥러닝 모델의 기본 구조가 되는 **Deep convolutional neural networks(CNNs)의 의사결정 과정**에 대한 해석 framework를 제시한다. 목표는 학습된 CNN으로부터 비슷한 예측을 갖는 데이터들을 대표하는 **common semantics를 알아내기 위해 representative interpretations를 찾는 것**이다. 즉, representative interpretations는 CNN의 의사결정 과정에서 구분되는 대표적인 특징 그룹을 보여준다.

어떻게 학습된 CNN으로부터 이러한 representative interpretations를 찾을 수 있을까? 본 내용 리뷰에 앞서, 전체흐름을 다음과 같이 요약할 수 있다.

1. Feature map에 해당하는 the last convolutional layer로부터 최종 의사결정에 해당하는 logit까지의 연산을 함수로서 인식한다.
2. 이 함수는 piecewise linear function이기 때문에, linear boundaries로 구분된 영역마다 다른 결정 로직을 적용한다.
3. 각 image에 대하여, 좋은 interpretation을 제공하는 linear boundaries의 subset을 고르도록 최적화 문제를 푼다.

{% hint style="info" %}
\[Opinion]

ReLU 기반의 CNN 모델은 충분히 성능이 검증된 모델이기 때문에 이를 분석 대상으로 정한 것이 합당하다. 또한 제안된 방법은 heuristics이 아닌 최적화로 문제를 풀어냈기 때문에 신뢰할만한 방법을 제시했다.
{% endhint %}

## 2. Motivation

### Related Work

CNN의 로직을 설명하기 위한 다양한 해석기법들이 연구되어 왔다.

1. Conceptual interpretation methods
   
   - e.g. [Automated Concept-based Explanation (ACE)](https://arxiv.org/abs/1902.03129)
   
   * 컨셉적으로 비슷한 이미지들로 사전에 정의된 그룹에서 예측에 기여하는 컨셉들의 집합을 찾는 방법이다.
   * 그러나 이 방법은 DNN에 복잡한 customization을 요구하기 때문에 일반적인 CNN에 범용적으로 적용하기 어렵다.
   
2. Example-based methods
   
   - e.g. [MMD-critic](https://dl.acm.org/doi/10.5555/3157096.3157352), [Prototypes of Temporally Activated Patterns (PTAP)](https://dl.acm.org/doi/abs/10.1145/3447548.3467346)
   
   * DNN의 의사결정을 해석하기 위해 모범 이미지(exemplar images)를 찾는다.
   * Prototype-based methods는 prototypes라 불리는 적은 수의 instances를 사용하여 전체 모델을 요약한다.
   * Prototype selection 방법은 모델의 의사결정 과정에 대한 고려가 부족할 수 있다.

### Idea

이 논문은 일반적은 CNN 모델에서 decision boundaries를 고려하여 의사결정에 대한 대표적인 해석성을 제공하는 것을 목표로 한다.

* 학습된 CNN의 decision logic을 encode하여 interpretation을 제공하는 decision region을 찾자.

* 이 문제를 co-clustering problem으로 formulation하였다. 여기서 Co-clustering은 비슷한 이미지들을 묶고 동시에 이들을 묶어주는 linear boundaries를 찾는다는 의미다.

* Co-clustering problem을 submodular cost submodular cover(SCSC) problem으로 치환하여 최적화 문제를 효과적으로 풀 수 있는(feasible) 방법을 제안한다.

  

## 3. Method

### Setting

이미지 분류 문제에서 학습된 ReLU activation function을 사용하는 CNN 모델을 생각해보자.

* $$\cal{X}$$: 이미지 공간
* $$C$$: 이미지 클래스의 수
* $$F:\mathcal{X}\rightarrow\mathbb{R}^C$$: 학습된 CNN, $$Class(x)=\argmax_i F_i(x)$$
* Reference images의 집합 $$R\subseteq\mathcal{X}$$
* $$\psi(x)$$: $$F$$의 마지막 convolutional layer로부터 생성된 feature map
* $$\Omega=\{\psi(x)\;|\;x\in\mathcal{X} \}$$ feature map 공간
* $$G:\Omega\rightarrow\mathbb{R}^C$$, feature map $$\psi(x)$$를 $$Class(x)$$로 매핑하는 함수
* $$\mathcal{P}$$: $$G$$의 linear boundaries(hyperplanes)의 집합

Reference images는 이 방법을 통해 해석하고 싶은 unlabeled images를 가리킨다.

### Representative Interpretations

문제를 formulation하기 앞서 representative interpretation을 찾는다는 목표를 구체화할 필요가 있다.

[Representative interpretation]

- 이미지 $$x\in\mathcal{X}$$에 대한 representative interpretation은 $$x$$에 대한 모델 $$F$$의 일반적인 의사결정을 드러내는 해석을 의미한다.

*   학습된 DNN 모델의 예측을 feature map을 통해 분석할 때, 많은 현존하는 연구에서 마지막 convolutional layer로부터 최종 class로의 매핑인 $$G$$를 이용하여 의사결정 로직을 설명한다.
*   $$G$$는 piecewise linear function이기 때문에, **linear boundaries로 구분된 구역에 따라 의사결정을 해석**할 수 있다. 자세한 내용은 [다음 논문](https://dl.acm.org/doi/abs/10.1145/3219819.3220063?casa\_token=MojIMpYRbLcAAAAA:19vihLVFk09s\_3zS1mtVpaxYvX7Cor5Fbkvso6UlSJYhW\_qPkO2oM7MCKIqJrTZ\_GgQsPeNgC8RK)을 참조하길 권한다.

![Decision logic of a CNN](../../.gitbook/assets/23/cnn\_decision\_logic.png)

\[Linear boundaries]

- $$G$$로 인한 의사결정 과정은 hyperplanes의 조각들로 구성된 piecewise linear decision boundary로 구분지어질 수 있다. $$G$$의 linear boundaries의 집합을 $$\cal{P}$$라 하자.

*   $$\cal P$$의 linear boundaries는 feature map space $$\Omega$$를 convex polytopes로 나눈다. 각각의 convex polytope는 해당 지역 안에 있는 이미지들을 동일한 class로 분류하는 decision region을 정의한다.
*   하지만 모든 convex polytopes가 label을 구분하는데 효과적인 역할을 하는 것은 아니다. 따라서 $$\cal P$$의 부분집합으로부터 $$x$$를 포함한 decision region을 잘 정의하는 것이 representative interpretation을 제공한다. 즉, 좋은 representative interpretation에 대응되는 $$P(x)\subseteq\mathcal{P}$$를 찾는 것이 목표이다.

{% hint style="info" %}
\[Goal]

각 image $$x$$에 대하여 좋은 representative interpretation이 될 수 있는 decision region $$P(x)\subseteq\mathcal{P}$$를 찾자.
{% endhint %}

### Finding Representative Interpretations

'좋은' representative interpretation이란 무엇일까? 이는 다음과 같은 두가지 조건을 만족해야한다.

1. $$P(x)$$의 representativeness를 최대화해야 한다.

   → Decision region $$P(x)$$가 최대한 많은 reference images를 커버해야한다.

   → maximize $$|P(x)\cap R|$$

2. $$x$$와 다른 class에 속하는 이미지들을 포함하지 않아야 한다.

   → $$|P(x)\cap D(x)|=0$$ where $$D(x)=\{x'\in R\;|\;Class(x')\neq Class(x)\}$$

이는 다음과 같은 최적화 문제로 표현할 수 있다. 이 문제는 비슷한 이미지의 집합과 중요한 linear boundaries의 집합을 동시에 찾기 때문에 저자는 이를 co-clustering problem이라 부른다. 

\[Co-clustering Problem]
$$
\max_{P(x)\subseteq\mathcal{P}}|P(x)\cap R|\\ \mathsf{s.t.}\quad|P(x)\cap D(x)|=0
$$

![Finding the optimal subset of linear boundaries](../../.gitbook/assets/23/RI\_cnn\_prob\_def.png)



그러나 co-clustering problem과 같은 set optimization problem은 실제로 풀기에 매우 복잡한 문제이다. 때문에 이 논문에서는,

1. Sampling을 통해 $$\cal P$$를 $$\cal Q$$로 사이즈를 줄인 부분집합을 사용하고;
2. $$\cal Q$$에 대해 submodular optimization 문제를 정의하여 문제를 풀 수 있도록 치환한다.

{% hint style="info" %}
Submodular Optimization이란?

* 최적의 set을 찾아야하는 set optimization 문제는 후보군들의 수가 많아질수록 경우의 수가 기하급수적으로 증가하기 때문에 계산적으로 복잡한 문제가 된다.
* 목적함수가 submodularity 성질을 만족하면, greedy algorithm을 통해 얻은 해가 적어도 실제 optimal solution의 성능의 일부분을 보장한게 된다. (The greedy algorithm achieves at least a constant fraction of the objective value obtained by the optimal solution.)
* 따라서, submodular optimization은 어느정도 성능을 보장하면서 동시에 거대하고 복잡한 set optimization 문제를 feasible하게 다룰 수 있게 해준다.
* Submodularity는 diminishing return property를 요구하는데, 자세한 내용은 [여기](https://en.wikipedia.org/wiki/Submodular\_set\_function)에서 확인할 수 있다.
{% endhint %}

{% hint style="info" %}
\[Opinion]

저자는 문제의 복잡도를 줄이기 위하여 random sampling으로 $$\cal Q$$에 해당하는 linear boundaries만 사용하였지만, 이 과정에서 빠진 linear boundaries가 의사결정에 중요한 역할을 하진 않을지 검증할 필요가 있다.
{% endhint %}

### Submodular Cost Submodular Cover problem

\[SCSC Problem]
$$
\max_{P(x)\subseteq\mathcal{Q}}|P(x)\cap R|\\ \mathsf{s.t.}\quad|P(x)\cap D(x)|\leq\delta
$$

* 함수 $$G$$로부터 linear boundaries의 집합 $$\cal P$$를 찾아내는 방법은 [다음 논문](https://dl.acm.org/doi/abs/10.1145/3219819.3220063?casa\_token=MojIMpYRbLcAAAAA:19vihLVFk09s\_3zS1mtVpaxYvX7Cor5Fbkvso6UlSJYhW\_qPkO2oM7MCKIqJrTZ\_GgQsPeNgC8RK)을 참조한다. 그 후, $$\cal P$$로부터 $$\cal Q$$를 sampling한다.

*   모든 linear boundaries를 사용하는 대신에 부분집합인 $$\cal Q$$만을 후보군으로 두어 최적화하기 때문에, 같은 convex polytope(decision region)에 속한 이미지 중에서 다른 class로 예측되는 경우가 발생할 수 있다.

    → $$|P(x)\cap D(x)|=0$$을 relax하여 제약조건을 $$|P(x)\cap D(x)|\leq\delta$$로 바꾸어준다.
    
* 이렇게 formulation한 문제의 목적함수와 제약조건은 submodular cost와 submodular cover 조건을 만족한다. 이에 대한 확인은 [본 논문](https://openaccess.thecvf.com/content/ICCV2021/html/Lam\_Finding\_Representative\_Interpretations\_on\_Convolutional\_Neural\_Networks\_ICCV\_2021\_paper.html)의 Appendix A를 참조하길 바란다.

* 결론적으로, 이 SCSC problem은 다음과 같은 greedy algorithm에 의해 순차적으로 linear boundary를 선택함으로써 해를 얻을 수 있다.

![The greedy algorithm to find representative interpretations.](../../.gitbook/assets/23/greedy\_alg.png)

### Ranking Similar Images

Decision region $$P(x)$$에 의해 포함되는 이미지($$x'$$)들의 유사성을 평가하기 위해서, 새로운 semantic distance를 다음과 같이 정의한다.

\[Semantic Distance]

$$
Dist(x.x')=\sum_{\mathbf{h}\in P(x)}\Big\vert \langle \overrightarrow{W}_\mathbf{h},\psi(x)\rangle -\langle \overrightarrow{W}_\mathbf{h},\psi(x')\rangle \Big\vert
$$

*   $$\overrightarrow{W}_\mathbf{h}$$ 는 linear boundary $$\mathbf{h}\in P(x)$$에 대응하는 hyperplane의 normal vector이다.
*   즉, 이미지 $$x'$$가 $$P(x)$$에 포함된 각각의 hyperplane을 $$x$$와 비교하여 얼마나 멀어지려하는지 측정하는 척도이다. 이는 Euclidean distance와 달리, decision region을 벗어나는 척도로 생각할 수 있다.
*   이 semantic distance를 이용하여 오름차순으로 $$P(x)$$에 의해 커버되는 이미지들을 랭킹한다.

![Semantic distances are different in terms of the decision region.](../../.gitbook/assets/23/semantic\_dist.png)

Semantic distance에 대한 이해를 돕기 위한 예시이다. $$x_1,x_2$$는 $$x$$와 같은 Euclidean distance만큼 떨어져있지만, decision region 관점에서 semantic distance를 계산하면 $$x_2$$와 $$x$$의 거리가 더 먼 것으로 측정된다.

## 4. Experiment & Result

### Experimental setup

저자는 representative interpretation (RI) method와 Automatic Concept-based Explanation(ACE), CAM-based methods(Grad-CAM, Grad-CAM++, Score-CAM)을 비교하였다.

* $$|\mathcal{Q}|=50$$으로 샘플링되었다.
* 위의 방법들은 channel weights를 이용하여 해석성을 제공한다. $$x\in\mathcal{X}$$에 대하여, 동일한 channel weights를 쓰면서 비슷한 이미지 $$x_{new}$$에 대해 heat map이 어떻게 보여지는지 비교한다.
  * RI의 경우, 비슷한 이미지 $$x_{new}$$를 정의하기 위헤 semantic distance를 사용한다.
  * 다른 방법의 경우, 비슷한 이미지 $$x_{new}$$를 정의하기 위해 feature map space $$\Omega$$ 상에서의 Euclidean distance를 사용한다.
* 데이터셋은 Gender Classification (GC), ASIRRA, Retinal OCT Images (RO), FOOD datasets 사용하였다.
* 해석할 타겟 모델로 VGG-19를 학습하였다.

### Result

#### Case Study

* 각 방법론이 비슷한 이미지에 대한 해석성을 적절히 제공하는지 비교한 실험이다.
* 첫번째 행은 RI method의 결과를 보여준다. 다른 방법들과 달리, 주어진 이미지에 대해서 표시된 heat map이 비슷한 이미지들에 대해서도 동일하게 표시된다.
* RI method는 co-clustering problem을 풀어 같은 interpretation을 공유하는 이미지들을 성공적으로 찾아내고, 모델이 그 이미지들을 어떻게 해석하는지 잘 보여준다.

![A case study on the GC dataset.](../../.gitbook/assets/23/case\_study.png)

#### Quantitative Experiment

Reference dataset으로 계산된 interpretations가 unseen dataset에 대한 예측을 해석하는데 얼마나 잘 쓰일 수 있는지 정량적으로 비교하였다. 이를 위해 두가지 지표를 설정하였다.

\[Average Drop (AD)]

$$
\frac{1}{|S|}\sum_{e\in S}\frac{\max(0,Y_c(e)-Y_c(e'))}{Y_c(e)}
$$
\[Average Increase (AI)]

$$
\frac{1}{|S|}\sum_{e\in S}\mathbb{1}_{Y_c(e)<Y_c(e')}
$$

*   $$S\subseteq \mathcal{X}$$: unseen images 집합
*   $$Y_c(e)$$: 이미지 $$e\in S$$에 대한 class $$c$$ 예측 점수
*   $$e'$$: 가장 중요한 20%의 pixels만을 남긴 masked image

따라서 AD는 방법론이 가리키는 중요한 부분만을 남겼을 때 저하되는 예측률을, AI는 중요한 부분만을 남겼을 때 예측률이 상승한 샘플의 비율을 의미한다. mean AD(mAD)가 작고 mean AI(mAI)가 클수록, interpretations가 보지못했던 데이터에 대해서도 유효하게 쓰일 수 있음을 나타낸다. 아래의 표를 보면, 대부분의 상황에서 RI method가 가장 좋은 성능을 갖는다는 것이 보여졌다.

![Quantative results.](../../.gitbook/assets/23/quant\_exp.png)

## 5. Conclusion

* 이 논문에서는 CNN의 의사결정 과정을 해석하기 위해 decision boundaries를 고려한 co-clustering problem을 제시하였다.
* Co-cluster problem을 풀기 위해 SCSC problem으로 치환하여 greedy algorithm을 적용할 수 있게 만들었다.
* 계산한 representative interpretations common semantics를 잘 반영한다는 것을 실험적으로 보여주었다.

### Take home message

> Deep neural networks가 다양한 분야에서 쓰임에 따라 의사결정 로직을 해석하는 것 매우 중요해졌다. 때문에 decision boundary를 고려하여 해석성을 제시하려는 접근이 인상적이었고, 이러한 연구가 더 확장되길 바란다.&#x20;

## Author / Reviewer information

### Author

**장원준 (Wonjoon Chang)**

* KAIST AI, Statistical Artificial Intelligence Lab.
* one\_jj@kaist.ac.kr
* Research Topics: Explainable AI, Time series analysis.
* https://github.com/onejoon

### Reviewer

*

## Reference & Additional materials

1. Lam, P. C. H., Chu, L., Torgonskiy, M., Pei, J., Zhang, Y., & Wang, L. (2021). Finding representative interpretations on convolutional neural networks. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*.
1. Ghorbani, A., Wexler, J., Zou, J., & Kim, B. (2019). Towards automatic concept-based explanations.
1. Kim, B., Khanna, R., & Koyejo, O. O. (2016). Examples are not enough, learn to criticize! criticism for interpretability. *Advances in neural information processing systems*, *29*.
1. Cho, S., Chang, W., Lee, G., & Choi, J. (2021, August). Interpreting Internal Activation Patterns in Deep Temporal Neural Networks by Finding Prototypes. In *Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining*.
2. [Wikipedia: Submodular set function](https://en.wikipedia.org/wiki/Submodular\_set\_function)
3. Chu, L., Hu, X., Hu, J., Wang, L., & Pei, J. (2018, July). Exact and consistent interpretation for piecewise linear neural networks: A closed form solution. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*.

***
