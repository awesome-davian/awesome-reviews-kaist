---
description: >-
  Xue Bin Peng et al. / AMP: Adversarial Motion Priors for Stylized
  Physics-Based Character Control / Transactions on Graphics (Proc. ACM SIGGRAPH
  2021)
---

# AMP \[KOR]

한국어로 쓰인 리뷰를 읽으려면 [**여기**](/paper-review/2021-fall-paper-review/SIGGRAPH-2021-AMP-kor.md)를 누르세요

## 1. Problem definition

The physical model which moves naturally like 
Physical models that move naturally like real life are essential in movies and games. In addition, this natural movement is a major concern associated with robots as safety and energy efficiency are inherent. While examples of these natural movements are plentiful, understanding and elucidating their characteristics is very difficult, and replicating them in a controller is even more difficult.

The generated locomotions without imitation learning perform very inappropriate behaviors in consideration of stability and functionality, such as walking with knees bent or walking with arms in an unnatural shape, etc. In order to solve this problem, it is probably necessary to design a very complex reward, but it can be solved by encouraging behavior similar to the locomotion of real life. This is the reason why imitation learning has been in the spotlight in robotics.

However, simply having them imitate the behavior eventually makes the agent unable to learn anything but one learned behavior. The research in this paper aims to develop a system in which a user can set a high-level task objective, and the low-level style of movement is generated from motion data of the real life provided in an unordered form.

###


## 2. Motivation

### Related work

The natural movements of real life are stable, efficient, and natural to look at. Implementing this in a physical environment has been studied in various fields such as robotics and games. In this chapter, we will introduce four representative methodologies.

**_Kinematic Methods:_**  
Studies based on the kinematic method generate the movement of a character using motion clips such as motion capture. It is typical to create a controller that executes an appropriate motion clip according to the situation based on a motion data set, and generators such as a Gaussian process or a neural network are used for this in previous studies. It has been shown in many studies that kinematic methods can realistically implement various complex movements when a sufficient amount of high-quality data is provided. However, relying solely on the actual dataset is a limitation of this methodology. Given a new situation, kinematic methods are difficult to use, and collecting sufficient data for complex tasks and environments is not easy.

**_Physics-Based Methods:_**  
Physics-based mothod typically utilizes physics simulation or a kinetic spinning room to generate the character's movements. In this methodology, optimization theories such as path optimization and reinforcement learning are mainly used to generate the movement of a character through objective function optimization. However, it is very difficult to design an objective function that induces a natural movement. There have been studies to optimize factors such as symmetry, stability, or energy consumption optimization, and to use actuator models that are similar to the structure of living things, but they have not succeeded in completely generating natural movements.

**_Imitation Learning:_**  
Due to the aforementioned difficulties in objective function design, imitation learning using natural movement data is being actively studied. In motion generation, the imitation objective mainly aims to minimize the difference between the generated motion and the actual motion data. In order to synchronize the motion generated in this process with the actual motion data, phase information is also used as additional input data. However, the proposed methods have difficulty in learning various motion data, and in particular, when there is phase information, synchronization between multiple motions may be almost impossible. In addition, these algorithms use pose error metric in motion tracking optimization. This is mainly designed by humans, and it is not easy to design a metric that can be applied to all movements when a character is taught several movements. Adversarial imitation learning suggests various alternatives, and it is possible to learn the characteristics of a given motion through the adversarial learning process without human design. However, adversarial learning algorithms can lead to very unstable results. The author's last study succeeded in generating natural data by limiting the rapid learning of the discriminator through the information bottleneck. However, this method still required phase information for synchronization, so it was not possible to learn various actual motion information in the policy.

**_Latent Space Models:_**  
The latent space model can also operate in the form of motion prior, and these models learn how to generate specific controls from latent representation information. By learning the latent representation to encode the behavior of the referenced motion data, the creation of natural behavior is possible. Also, in the latent representation, the latent space model is used as a low-level controller, and the priority of the control can be set by separately learning the high-level controller through the latent space. However, since the generated motion implicitly refers to the actual motion through the latent representation, unnatural motion may be generated under the influence of high-level control.


### Idea

As explained in previous studies, previous studies had a problem in that it was difficult to generate a natural motion or that only one motion could be referenced for learning.
The principle of this study is to ensure that the behavior of the learned agent is included in the distribution of “actual life behavior”. In other words, it can be said that the core of this thesis is to make the probability distribution of the behavior generated by the policy similar to the probability distribution of real life. This is very similar to the goal of GAN, and it will be convenient to understand it as a GAN in the "action" domain even when actually understanding the algorithm. This goal is achieved through style reward design, and the basis for judging style reward is through the similarity judgment of distribution from a discriminator.
In this study, the author aims to create an agent that can refer to actual behavior according to a given task based on Generative Adversarial Learning. For this goal, the algorithm includes a style reward for the similarity between the simulated motion and the real motion along with the task reward.

In the next chapter, the detailed description of style reward and the overall algorithm will be explained.





## 3. Method

### Backgroud

**About RL in Robotics**  

Simulation in the field of robotics basically aims for the agent to perform a goal (eg, walking) well in a given environment. Basically, the environment is configured based on a physics engine, and the action that Agent performs here is “inputs to the actuators (current, sometimes as torque)”.

In other words, the robot simulation means that the robot (performing movements for given actuator inputs) and environment (physical collision and gravity, etc.) are already set, and the robot's controller is It is the process of designing which “action” to each motor should be sent so that the robot can perform its goal in the environment.

**Goal-Conditioned Reinforcement Learning**

The goal of goal-based reinforcement learning is to create an agent that maximizes the reward based on the designed reward function.
(The basic reinforcement learning terms will not be explained.)

![eq1](/.gitbook/assets/57/eq1.png)

As a result, the agent learns a policy that maximizes the optimization objective defined by the above equation.  
This research trains the agents based on [PPO](https://arxiv.org/abs/1707.06347).



**Generative Adversarial Imitation Learining**

The basis of this reserch is the generation of the motion prior with [GAIL](https://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning.pdf).

The original objective of the GAIL is as below.

![eq2](/.gitbook/assets/57/eq2.png)

The original reward of the GAIL is as below.

![eq3](/.gitbook/assets/57/eq3.png)
  
(The GAIL shares its principle with [GAN](https://papers.nips.cc/paper/5423-ge...al-nets.pdf), but it has state and action domain instead of data such as image)  
Through the above optimization, the agent creates an action that is indistinguishable from the distribution of the actual motion capture data as much as possible.


### Notations

**Baisic Notations**  
$$g$$: Goal  
$$s$$: State  
$$a$$: Action  
$$\pi$$: Policy

**Notatations for AMP**  
$$M$$: The real-life motion data domain  
$$d^{M}$$: Probability distribution of the real-life  
$$d^{\pi}$$: Probability distribution generated from the learned policy  

### System

![fig2](/.gitbook/assets/57/fig2.png)

The figure above is the overall system structure of this paper. Based on this, the proposed algorithm will be described.  
As mentioned above, the main content of the overall structure of this paper is to train the PPO agent.  
The above agent is trained to maximize the following reward function.

![eq4](/.gitbook/assets/57/eq4.png)

In above equation, $$r^G$$ is a reward for a high-level goal(ex. heading, walking), and it will be designed specifically dependent on a task to be achieved.  
On the other hand, $$r^S$$ is a *style-reward* for the motion generated by the agent.  
Through style reward, the agent is trained to generate a motion similar to the given motion data as much as possible.  
The determination of this style reward will be the core content of this study.  
$$w^G$$ and $$w^S$$ are the simple weights for each, and simply set as 0.5 and 0.5 in the experiments.  


### Style reward

As stated earlier, the style reward is judged by the GAIL algorithm.  
However, motion clips are provided in the form of state, not action.  
Therefore, the algorithm is optimized based on state transitions rather than actions, which changes the GAIL objective as follows.

![eq5](/.gitbook/assets/57/eq5.png)

Additionally, this research optimizes the disriminator based on the least-squares loss to prevent the vanishing gradient problem based on [a previous research](https://doi.org/10.1109/ICCV.2017.304).

![eq6](/.gitbook/assets/57/eq6.png)

One of the main causes of the instability of GAN-generated dyanmics is the function approximation error in the discriminator.  
To alleviate this phenomenon, a method of penalizing the nonzero gradient can be used, and the final objective to which the gradient penalty is applied is as follows.

![eq7](/.gitbook/assets/57/eq8.png)

And, the style reward is determined as follows based on the previously formed objective.

![eq8](/.gitbook/assets/57/eq7.png)


### Discriminator observations

Previously, it was explained that the discriminator is based on state transition.  
If so, features to be observed by the discriminator are needed to be set.  
In this study, the following sets of features were used as inputs (observed states).

  * Linear speed and rotation speed of the character's origin (pelvis) in global coordinates
  * Local rotation / velocity of each joint
  * Local coordinate of each end-effector


### Training

The actors (generator), critic, and discriminator in this study are all based on the 2-layer 1024 and 512 ReLU network architecture.

The full learning algorithm is as follows.

![algorithm](/.gitbook/assets/57/algorithm.png)  



## 4. Experiment & Result

### Experimental setup

* Dataset
  
  Motion capture data of several people was used for the actual motion data to be compared by the Descriminator.  
  In the case of a complex task, several motion data were used together in one task.
  
* Baselines
  
  The results were compared with [Deepmimic](https://doi.org/10.1145/3197517.3201311), which was the state-of-the-art in the field.

* Training setup
  
  The high-level tasks tested are as follows.
  
  * Target heading: The character moves in accordance with the target speed in the specified heading direction.
  * Target location: The character moves towards a specific target location.
  * Dribbling: For evaluation of a complex task, the character performs the task of moving a soccer ball to a target location.
  * Strike: In order to evaluate whether various motion information can be mixed, the character performs the task of striking the target object with a specified end-effector.
  * Obstacles: A character is tasked with traversing terrain filled with obstacles to evaluate the ability to interact with visual perception information in a complex environment.
  
* Evaluation metric
  
  The task return value was used to evaluate the task, and the average pose error was calculated for similarity comparison with the given action. The formula for calculating the pose error at a specific time step is as follows.
  
  ![eq10](/.gitbook/assets/57/eq10.png)



### Result

![fig3](/.gitbook/assets/57/fig3.png)

As can be seen in [the video published by the author](https://youtu.be/wySUxZN_KbM), the agent trained with the presented methods showed excellent performance in complex environments and various tasks, and the generated movements are also natural like real-life.

The return values for the suggested tasks are as follows, and it shows how the task is achieved by combining several movements without any problem in actual execution.

![table1](/.gitbook/assets/57/table1.png)

Compared with the existing state-of-the-art, as can be seen in the following table, it shows a slightly lower numerical value in the reproduction of the motion. However, when viewed as an absolute number, there is no shortage, and compared to the existing method using only one motion data, in this study, the agent performs the necessary motion among several motion data according to the task.

![table3](/.gitbook/assets/57/table3.png)



## 5. Conclusion

This paper is a state-of-the-art of locomotion simulation.

The biggest contribution of this paper is that the agent learns several motion data at once and generates the necessary motion according to the given situation.

As you can see in the video, in the strike task, the agent naturally walks to the object and strikes the object by extending his fist.
Only real human walking motion and fist-stretching motion data were used to learn this motion.

Similarly, compared to performing a surprisingly complex process such as running on obstacle terrain, the data required for learning is very simple and easy to obtain. Considering the fact that one of the biggest difficulties of deep learning is the acquisition of sufficient data for learning, this characteristic can be considered a great advantage.

The results of this study are expected to bring great progress in various fields such as robots, games, and animation.


### Take home message (오늘의 교훈)

> This study obtained the best performance by properly combining only the existing algorithms.
>
> It is important to constantly study and try new research.

## Author / Reviewer information

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
