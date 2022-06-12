---
Ben et al. / NeRF; Representing Scenes as Neural Radiance Fields for View Synthesis / ECCV 2020
---

# NeRF [Kor]
##  1. Problem definition

NeRF가 수행하고자 하는 View Synthesis 라는 문제는, 다양한 카메라 각도에서 찍은 물체의 이미지들을 input으로 받아, 새로운 각도에서 바라보는 물체의 이미지를 만들어내는(예측하는) 것입니다.
$$
F_{\theta} : (X,d) \rightarrow (c,\sigma)
$$
공간 좌표 $$x = (x,y,z)$$와 보는 각도 $$d = (\theta, \phi)$$를 input으로 받아(총 5D 좌표가 된다), 해당 물체의 volume density와 emitter color을 산출한다.



![Figure 1. View Synthesis 문제](../../.gitbook/assets/2022spring/15/fig1.PNG)

공간 좌표

## 2. Motivation

### 2.1. Related work

최근에 computer vision 분야에서, 3D 위치 정보로부터 물체 형태의 

signed distance 같은 방법들이 제안되었습니다. 하지만, 이러한 방법들은 조금 더 현실적인 scene들을 복잡한 구조를 가진 

##### - Neural 3D shape representations

그러나 이러한 방법은 지금까지 삼각형 메시 또는 복셀 그리드와 같은 이산 표현을 사용하여 장면을 나타내는 기술과 동일한 정밀도로 복잡한 지오메트리를 가진 사실적인 장면을 재현할 수 없었다.

이 섹션에서는 이러한 두 가지 baseline을 검토하고 신경 장면 표현 기능을 향상시켜 복잡한 사실적 장면을 렌더링하기 위한 최첨단 결과를 생성하는 접근 방식과 대조한다.

NeRF는 neural scene representation 성능을 향상시키고, 복잡한 현실적인 scene을 렌더링하는 state-of-the-art 방법입니다.

MLP을 사용하여 

복잡한 scene을 렌더링하는 neural scene representation

첫번째 방법은 signed distance입니다. $$x, y, z$$ 좌표를

하지만 이러한 방법들은 3D 형태에 대한 ground truth 정보를 필요로 합니다.





##### - View synthesis and image-based rendering

여러 방향에서의 이미지들을 샘플링하여 포토리얼리스틱한 

더 적은 개수의 view를 샘플링하여 

그래디언트 기반의 mesh 최적화 방법을 사용하여 이미지를 reprojection하는 것은 매우 어렵습니다. local minima에 빠지기 쉽고, 

하지만 장점이 있지만, 단점이 있습니다.



Please introduce related work of this paper. Here, you need to list up or summarize strength and weakness of each work.

### 2.2. Main Idea

After you introduce related work, please illustrate the main idea of the paper. It would be great if you describe the idea by comparing or analyzing the drawbacks of the previous work.





NeRF는 위의 방법들과 달리 복잡하고 고해상도의 형체를 

저자는 오직 MLP만을 이용하여 

또한, positional encoding

구체적인 Method는 아래에서 소개한다.

## 3. Method

### 3.1. Neural Radiance Field Scene Representation

먼저, NeRF는 3차원의 위치 정보 $$X = (x, y, z)$$와 2차원의 보는 방향 $$d = (\theta, \phi)$$ 을 input으로 받아 

즉, 해당 시점(5차원 벡터)에서 바라보는 물체의 모습을 

NeRF는 이를 MLP을 



$$F_{\Theta} : (X,d) \rightarrow (c,\sigma)$$


![Figure 2. Overview of NeRF](../../.gitbook/assets/2022spring/15/fig2.PNG)

We strongly recommend you to provide us a working example that describes how the proposed method works.  

MLP의 구체적인 구조는 아래의 Figure 3. 과 같다. 초록색이 Input 벡터이고, 중간의 hidden layer가 파란색, output 벡터가 빨간색으로 표시되어 있다. 모든 layer는 fully-connected이고, 검은색 화살표는 ReLU activation, 주황색 화살표는 activation function이 없는 것, 검은색 점선 화살표는 sigmoid activation을 의미한다ㅏ. 

<img src="../../.gitbook/assets/2022spring/15/fig7.PNG" alt="Figure 3. Neural Network 구조" style="zoom:75%;" />



### 3.2. Volume Rendering with Radiance Field

NeRF는 전통적인 volume rendering 기법들을 사용하여 렌더링을 진행한다.



### 3.3. Positional Encoding

We strongly recommend you to provide us a working example that describes how the proposed method works.  

비록 Neural network가 모든 function을 approximate 할 수 있다 해도, $$F_{\Theta}$$ 를  

 <img src="../../.gitbook/assets/2022spring/15/eq2.PNG" alt="Table 1. NeRF 성능" style="zoom:50%;" />

### 3.4. Hierarchical volume sampling

We strongly recommend you to provide us a working example that describes how the proposed method works.  




## 4. Experiment & Result

This section should cover experimental setup and results.  
Please focus on how the authors of paper demonstrated the superiority / effectiveness of the proposed method.

Note that you can attach tables and images, but you don't need to deliver all materials included in the original paper.



### 4.1. Experimental setup

##### - Dataset

Synthetic rendering of objects 데이터셋을 사용한다. Diffuse Synthetic 360º와 Realistic Synthetic 360º



##### - Baselines

* Neural Volumes (NV)
* Scene Representation Networks (SRN)
* Local Light Field Fusion (LLFF) 



##### - Training setup

* batch size = 4096, Adam optimizer with lr = 5e-4 and exponentially-decaying to 5e-5, 하나의 scene에 대해 100-300k 정도의 iteration이 걸렸고, single NVIDIA V100 GPU를 사용했습니다. (하루에서 이틀 정도 걸림)



##### - Evaluation metric





### 4.2. Result

Please summarize and interpret the experimental result in this subsection.

![Table 1. NeRF 성능](../../.gitbook/assets/2022spring/15/table1.PNG)

asdasdasd

![Figure 1. NeRF 성능](../../.gitbook/assets/2022spring/15/table1.PNG)



### 4.3. Ablation Study

Please summarize and

![Table 2. Ablation Study](../../.gitbook/assets/2022spring/15/table2.PNG)

asdsad

## 5. Conclusion

In conclusion, please sum up this article.  

You can summarize the contribution of the paper, list-up strength and limitation, or freely tell your opinion about the paper.

NeRF는 







### Take home message \(오늘의 교훈\)

Please provide one-line \(or 2~3 lines\) message, which we can learn from this paper.

> All men are mortal.
>
> 

## Author / Reviewer information

### Author

**유태형 \(Taehyung Yu\)** 

* KAIST AI
* KAIST Data Mining Lab.
* taehyung.yu@kaist.ac.kr

### Reviewer

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. Mildenhall, Ben, et al. "Nerf: Representing scenes as neural radiance fields for view synthesis." *European conference on computer vision*. Springer, Cham, 2020.
2. Official \(unofficial\) GitHub repository
3. Citation of related work
4. Other useful materials
5. ...

