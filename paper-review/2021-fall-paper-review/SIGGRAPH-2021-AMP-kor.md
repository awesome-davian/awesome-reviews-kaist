---
description: >-
  Xue Bin Peng et al. / AMP: Adversarial Motion Priors for Stylized
  Physics-Based Character Control / Transactions on Graphics (Proc. ACM SIGGRAPH
  2021)
---

# AMP \[KOR]

[**English version** of this article is **NOT YET** available.](https://www.google.com)

## 1. Problem definition

실제 생명체처럼 자연스럽게 움직이는 물리적 모델은 영화 및 게임 등에서 필수적인 요소이다. 이러한 실감나는 움직임에 대한 요구는 VR의 등장으로 더욱 커졌다. 또한, 이러한 자연스러운 움직임은 안전과 에너지 효율성을 내재하고 있기에 로봇과 연관된 주요 관심사이다. 이러한 자연스러운 움직임의 예시는 풍부한 반면, 그 특성을 이해하고 밝혀내는 것은 난해하며 이를 컨트롤러에 복제하는 것은 더욱 어렵다.

실제로 PPO 등 모방학습이 없이 생성된 걸음을 보면, 무릎을 굽히고 걷거나 팔을 부자연스러운 형태로 하는 등 "주어진 목표"만 잘 수행하는, 안정성과 기능성 등을 고려하면 매우 부적합한 행동을 하는 것을 볼 수 있다. 이러한 문제를 해결하기 위하여서는 아마 매우 복잡한 리워드의 설계가 필요할 것이나, 이미 이러한 사항들이 고려되어있는 실제 생명체의 행동과 비슷한 행동을 장려함으로써 해결 가능하다. 이것이 로보틱스에서 모방학습이 각광받기 시작한 이유이다.

그러나, 단순히 동작을 따라하도록 하는 것은 결국 에이전트가 학습된 한 가지 동작 이외에는 배울 수 없도록 만든다. 본 연구는 사용자가 high-level task objective를 설정할 수 있으며, 그에 따른 움직임의 low-level style은 정돈되지 않은 형태로 제공되는 모션 캡쳐 예시들로부터 생성되는 시스템의 개발을 목표로 한다.

###


## 2. Motivation

### Related work

동물의 자연스러운 움직임은 안정적이고 효율적이며, 보기에 자연스럽다. 이를 물리적 환경에서 구현하는 것은 로보틱스 및 게임 등 다양한 분야에서 연구되어왔다. 본 챕터에서는 대표적인 방법론 네 가지를 소개하고자 한다.

**_Kinematic Methods:_**  
Kinematic method에 기반한 연구들은 모션 캡쳐 등의 motion clip을 사용하여 캐릭터의 움직임을 생성한다. 모션 데이터셋을 기반으로 상황에 따른 적절한 모션 클립을 실행하는 컨트롤러를 생성하는 것이 대표적이며, 선행 연구들에서 이를 위하여 Gaussian process나 neural network 등의 generator들이 사용된다. 충분한 양의 질 좋은 데이터가 제공될 때, kinematic method는 다양한 복잡한 움직임을 실제처럼 구현할 수 있음이 많은 연구에서 보여졌다. 그러나, 오로지 실제 dataset에만 의존하는 것이 이 방법론의 한계이다. 새로운 상황이 주어졌을 때 kinematic method는 사용이 어려우며, 복잡한 task와 환경에 대해 충분한 데이터를 모으는 것은 쉽지 않다.

**_Physics-Based Methods:_**  
Physics-based mothod는 일반적으로 물리 시뮬레이션을 활용하거나 운동방적실을 활용하여 캐릭터의 움직임을 생성한다. 이 방법론에서 경로 최적화 및 강화학습과 같은 최적화 이론들이 주로 objective function 최적화를 통해 캐릭터의 움직임을 생성하는 데에 사용된다. 그러나, 자연스러운 움직임을 유도하는 objective function을 디자인하는 것은 매우 어려운 일이다. 대칭성, 안정성 혹은 에너지 소모 최적화와 같은 요소를 최적화하고 생명체의 구조와 비슷한 actuator 모델을 사용하는 등의 연구들이 있어왔으나, 자연스러운 움직임을 완벽히 생성하는 것은 성공하지 못했다.

**_Imitation Learning:_**  
앞서 언급된 objective function 설계의 어려움으로 인하여 자연스러운 움직임의 데이터를 활용하는 imitation learning이 활발하게 연구되고 있다. 동작 생성에서 imitation objective는 주로 생성된 동작과 실제 동작 데이터의 차이를 최소화하는 것을 목표로 한다.  이 과정에서 생성된 모션과 실제 모션 데이터의 동기화를 위하여 페이즈 정보를 추가 input data로 사용하기도 한다. 그러나 제시된 방법들은 여러가지 동작 데이터를 학습하는데 어려움이 있으며, 특히 페이즈 정보가 있을 때에는 여러 동작간의 동기화가 거의 불가능할 수 있다. 또한, 이러한 알고리즘은 동작 추적 최적화에서 pose error metric를 사용한다. 이는 주로 사람이 직접 디자인하며, 캐릭터에게 여러 동작을 학습시킬 때 모든 동작에 적용 가능한 metric의 설계는 쉽지 않다. Adversarial imitation learning은 다양한 대안을 제시하는데, 사람이 직접 설계하지 않고도 adversarial learning 과정을 통하여 주어진 동작의 특성을 학습시킬 수 있다. 그러나, adversarial learning 알고리즘은 굉장히 unstable한 결과를 가져올 수 있다. 저자의 지난 연구는 information bottleneck을 통하여 discriminator의 빠른 학습을 제한함으로써 자연스러운 데이터 생성에 성공하였다. 그러나, 이러한 방법은 여전히 동기화를 위해 페이즈 정보를 요구하였으며, 따라서 정책에 여러 실제 모션 정보를 학습시킬  수 없었다.

**_Latent Space Models:_**  
Latent space model 또한 motion prior의 형태로 작동할 수 있으며, 이러한 모델들은 latent representation 정보에서 특정한 control을 생성하는 방법을 학습한다. latent representation이 참조되는 모션 데이터의 행동을 encode하도록 학습시킴으로써 자연스러운 동작의 생성이 가능하다. 또한, latent representation에서는 latent space model을 low-level controller로 사용하고 high-level controller는 latent space를 통하여 따로 학습시키는 방법을 통하여 컨트롤의 우선순위 설정이 가능하다. 그러나 생성되는 모션은 latent representation을 통해 함축적으로 실제 움직임을 참고하기에, high-level control의 영향으로 자연스럽지 않은 움직임이 생성될 수 있다.



### Idea

선행된 연구들에서 설명된 것과 같이, 이전의 연구들은 자연스러운 움직임의 생성에 어려움이 있거나 한 가지 동작만 학습에 참고가 가능하다는 문제점이 있었다.  
본 연구의 원리는 학습된 에이전트가 하는 행동이 "실제 생명체의 행동"의 범주에 포함되도록 하는 것이다. 즉, 정책에서 생성된 행동의 확률분포가 실제 생명체의 확률분포와 유사하도록 만드는 것이 이 논문의 핵심이라고 할 수 있다. 이는 GAN의 목표와 매우 유사하며, 실제로 알고리즘을 이해할 때에도 "action" domain에서의 GAN으로 이해하면 편할 것이다. 이러한 목표는 style reward 설계를 통하여 성취하게 되며, style reward의 판단 근거는 distribution의 유사성 판단, 즉 discriminator를 통하여 이루어진다.  
이 연구에서 저자는 Generative Adversarial Learning을 기반으로 주어진 task에 따라 실제 동작을 참고할 수 있는 에이전트의 생성을 목표로 한다. 이러한 목표를 위하여 알고리즘은 task reward와 함께 시뮬레이션 모션과 실제 모션의 유사성에 대한 style reward를 포함하게 된다.

다음 장에서 style reward에 대한 상세한 설명 및 전체 알고리즘에 대해 설명할 것이다.





## 3. Method

### Backgroud

**로보틱스 시뮬레이션에 관하여**  

로봇 분야의 시뮬레이션은 기본적으로 Agent가 주어진 환경에서 Goal(ex. 걷기)을 잘 수행하는 것을 목표로 한다. 기본적으로 환경은 물리 엔진 기반으로 구성되어 있으며, 여기에서 Agent가 수행하게 되는 action은 "모터에 들어가는 입력(전류, 간혹 토크로 표현)"이다.

즉, 로봇의 시뮬레이션이란 로봇(주어진 모터 전류에 대한 관절 움직임 수행)과 환경(물리적 충돌과 중력 등)은 이미 정해져 있는 상태에서, "관측된 현재 상태(observed state)"를 기반으로 로봇의 컨트롤러가 각 모터에 "어떤 출력(action)"을 내보내야 로봇이 환경에서 목표를 수행할 수 있을지를 설계하는 과정이다.  


**목표 기반 강화학습**

목표 기반 강화학습은 설계된 reward function을 기반으로, reward를 최대로 만드는 agent를 생성하는 것이 그 목표이다.  
(기본적인 강화학습의 용어들은 설명을 생략한다.)

![eq1](/.gitbook/assets/57/eq1.png)

결과적으로, agent는 위 수식으로 정의된 optimization objective를 최대치로 하는 policy를 학습하게 된다.  
본 논문에서는 [PPO 알고리즘](https://arxiv.org/abs/1707.06347)을 기반으로 agent를 학습시킨다.



**Generative Adversarial Imitation Learining**

이 연구의 핵심은 [GAIL 알고리즘](https://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning.pdf)을 사용한 motion prior의 생성이다.

GAIL 알고리즘의 objective는 다음과 같다.

![eq2](/.gitbook/assets/57/eq2.png)

또한, reward는 아래 수식으로 정의된다.

![eq3](/.gitbook/assets/57/eq3.png)
  
(바탕이 되는 알고리즘은 [GAN](https://papers.nips.cc/paper/5423-ge...al-nets.pdf)과 같으며, data가 아닌 state-action을 대상으로 한다)  
위와 같은 optimization을 통하여 agent는 실제 모션 캡쳐 데이터의 distribution과 최대한 구분이 불가능한 action을 생성하게 된다.


### Notations

**기본 Notations**  
$$g$$: 목표  
$$s$$: 상태(state)  
$$a$$: 행동(action)  
$$\pi$$: 정책(policy)

**논문의 Notatations**  
$$M$$: 실제 사람의 모션클립 데이터 도메인  
$$d^{M}$$: 실제 사람 행동의 probability distribution  
$$d^{\pi}$$: 정책을 통해 생성된 probability distribution  

### System

![fig2](/.gitbook/assets/57/fig2.png)

위의 그림은 본 논문의 전체 시스템 구조도이다. 이를 기반으로 제안된 알고리즘을 설명할 것이다.  
앞서 말한 것과 같이, 본 논문의 전체 구조는 PPO agent를 학습시키는 것이 주요 내용이다.  
위 agent는 다음과 같은 reward function를 최대화 할 수 있도록 학습된다.

![eq4](/.gitbook/assets/57/eq4.png)

위 수식에서 $$r^G$$는 high-level의 목표(ex. 특정 지점 향하기, 공 드리블 등)에 대한 reward이며, 이는 직접 디자인된 간단한 수식이 될 것이다.  
반면에, $$r^S$$는 agent가 생성하는 움직임에 대한 *style-reward*이다.  
Style reward를 통하여 agent는 최대한 주어진 motion data와 유사한 동작을 생성하도록 학습된다.  
이 style reward의 결정이 본 연구의 핵심 내용이 될 것이다.  
$$w^G$$와 $$w^S$$는 각 reward에 대한 가중치이다. 본 연구에서 모든 내용은 두 가중치 모두 0.5로 설정하여 진행되었다.  


### Style reward

앞서 밝혔듯, style reward는 GAIL 알고리즘에서 판단된다.  
그러나 모션 클립들은 action이 아닌 state의 형태로 제공된다.  
따라서 action이 아닌 state transitions에 기반하여 알고리즘이 최적화되며, 이는 GAIL objective를 다음과 같이 변경하게 된다.

![eq5](/.gitbook/assets/57/eq5.png)

이에 더해서, 본 논문에서는 [선행 연구](https://doi.org/10.1109/ICCV.2017.304)에 기반하여 vanishing gradient의 방지를 위하여 cross-entropy 가 아닌 least-squares loss에 기반하여 discriminator를 최적화한다.  

![eq6](/.gitbook/assets/57/eq6.png)

GAN으로 생성된 dyanmics의 instability의 주요 원인 중 하나는 discriminator에서의 function approximation error에 기인한다.
이러한 현상의 완화를 위하여 nonzero gradient에 페널티를 주는 방식을 활용할 수 있으며, gradient penalty를 적용한 최종적인 objective는 다음과 같다.

![eq7](/.gitbook/assets/57/eq8.png)

그리고, style reward는 앞서 형성된 objective를 기반으로 다음과 같이 정해진다.

![eq8](/.gitbook/assets/57/eq7.png)

위 reward가 앞서 정의된 style-reward로 사용된다.



### Discriminator observations

앞서 discriminator가 state transtion에 기반함을 설명하였다.  
그렇다면, discriminator의 관찰 대상이 될 feature들이 필요하다.  
본 연구는 다음과 같은 feature들의 집합을 input(observed states)으로 사용하였다.

  * Global coordinate에서 캐릭터의 원점(pelvis)의 선속도 및 회전속도
  * 각 joint의 local rotation / velocity
  * 각 end-effector의 local coordinate


### Training

본 연구의 actor(generator), critic, 그리고 discriminator는 모두 2-layer 1024 and 512 ReLU 네트워크 구조에 기반한다.

전체 학습 알고리즘은 다음과 같다.

![algorithm](/.gitbook/assets/57/algorithm.png)  



## 4. Experiment & Result

### Experimental setup

* Dataset
  
  Descriminator가 비교하게 될 실제 motion data를 위하여 여러 사람의 motion capture data가 사용되었다.  
  복잡한 task의 경우, 하나의 task에 여러 motion data가 함께 사용되기도 하였다.
  
* Baselines
  
  비교에는 저자의 이전 연구인 [Deepmimic 알고리즘](https://doi.org/10.1145/3197517.3201311)으로 생성된 데이터가 사용되었다.  
  (해당 연구가 state-of-the-art 이었기 때문)

* Training setup
  
  실험된 high-level task들은 다음과 같다.
  
  * Target heading: 캐릭터가 정해진 heading direction을 향하여 target speed에 맞춰 움직인다.
  * Target location: 캐릭터가 특정 target location을 향해 움직인다.
  * Dribbling: 복잡한 task에 대한 평가를 위하여, 캐릭터는 축구공을 target location으로 옮기는 task를 수행한다.
  * Strike: 다양한 모션 정보를 혼합할 수 있는지 평가하기 위하여, 캐릭터가 target object를 정해진 end-effector로 타격하는 task를 수행한다.
  * Obstacles: 복잡한 환경에서 시각적 인식 정보와 interaction이 가능한지 평가하기 위하여, 캐릭터가 장애물로 채워진 지형을 가로지르는 task를 수행한다.
  
* Evaluation metric
  
  Task에 대한 평가로는 task return 값을 사용하였으며, 주어진 동작과의 유사성 비교에는 average pose error가 계산되었다. 특정 time step에서의 pose error의 계산식은 다음과 같다.
  
  ![eq10](/.gitbook/assets/57/eq10.png)



### Result

![fig3](/.gitbook/assets/57/fig3.png)

[저자가 공개한 동영상](https://youtu.be/wySUxZN_KbM)에서 확인할 수 있듯, 제시된 방법들로 훈련된 agent는 복잡한 환경과 다양한 task들에 대하여 굉장히 뛰어난 성능을 보였으며 생성된 움직임 또한 사람처럼 자연스러움을 확인할 수 있다.

제시된 task들에 대한 return값은 다음과 같으며, 실제 실행에서 문제 없이 여러 움직임을 조합하여 task를 달성하는 모습을 보여준다.

![table1](/.gitbook/assets/57/table1.png)

기존의 state-of-the-art와 비교하였을 때, 다음의 표에서 볼 수 있듯 동작의 재현에서는 정량적으로 조금 낮은 수치를 보여준다. 그러나 절대적인 수치로 보았을 때 부족함이 없는 수준이며, 하나의 motion data만을 사용하는 기존의 방법과 비교하여 이 연구에서는 에이전트가 task에 따라 여러 motion data 중에 필요한 동작을 수행한다.

![table3](/.gitbook/assets/57/table3.png)



## 5. Conclusion

본 논문은 현재 locomotion simulation의 state-of-the-art이다.

이 논문의 가장 큰 기여는 agent가 여러 동작 데이터들을 한번에 학습하며, 주어진 상황에 맞춰 필요한 motion을 생성한다는 것이다.

동영상에서 볼 수 있듯, strike task에서 agent는 자연스럽게 object로 걸어가 주먹을 뻗어 object를 타격한다.  
이 동작의 학습에 사용된 것은 오직 실제 사람의 걷는 동작과 주먹을 뻗는 동작 데이터 뿐이다.

마찬가지로, 장애물 지형에서의 달리기 등 놀라울 정도로 복잡한 과정을 수행하는 것에 비하여 학습에 필요로 하는 data는 매우 단순하며 쉽게 얻을 수 있다. 딥러닝의 가장 큰 어려움 중 하나가 학습을 위한 충분한 데이터의 획득이라는 사실을 감안할 때, 이러한 특성은 굉장한 장점으로 생각할 수 있다.

이 연구의 결과는 로봇, 게임, 에니메이션 등 다양한 분야에 큰 진보를 가져다 줄 것으로 기대된다.



### Take home message (오늘의 교훈)

> 본 연구는 기존의 알고리즘들만을 적절히 조합하여 최고의 성과를 얻었다.
>
> 새로운 연구들에 대한 끊임없는 공부와 시도가 중요하다.

## Author / Reviewer information

{% hint style="warning" %}
You don't need to provide the reviewer information at the draft submission stage.
{% endhint %}

### Author

**안성빈 (Seongbin An)**

* KAIST Robotics Program
* I am really new to this field. Thanks in advance for any advice.
* sbin@kaist.ac.kr

### Reviewer

1. Korean name (English name): Affiliation / Contact information
2. Korean name (English name): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. Peng, Xue Bin, et al. "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control." arXiv preprint arXiv:2104.02180 (2021).
2. [Official GitHub repository](https://github.com/xbpeng/DeepMimic)
3. Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
4. Ho, Jonathan, and Stefano Ermon. "Generative adversarial imitation learning." Advances in neural information processing systems 29 (2016): 4565-4573.
5. Peng, Xue Bin, et al. "Deepmimic: Example-guided deep reinforcement learning of physics-based character skills." ACM Transactions on Graphics (TOG) 37.4 (2018): 1-14.
6. Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems 27 (2014).
