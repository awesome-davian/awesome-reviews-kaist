---
description: (Description) Xie et al./Improving transferability of adversarial examples with input diversity/CVPR 2019
---

# Diversity Input Method \[Kor\]

\(In English article\) ---&gt; 한국어로 쓰인 리뷰를 읽고 싶으시면 **여기**를 누르세요.

\(한국어 리뷰에서\) ---&gt; **English version** of this article is available.



## 1. Introduction

#### 적대적 공격(Adversarial Attack)

**적대적 공격**이란, 그림과 같이 이미지에 미세한 _잡음 (noise)_을 의도적으로 추가하여 모델의 잘못된 예측을 유도하는 기법입니다. 적대적 공격은 공격자가 타겟 모델의 예측을 특정한 클래스로 유도하는 공격인 표적 공격 (targeted attack)과, 유도하지 않고 단순히 예측을 틀리게 하는 무표적 공격 (non-targeted attack)으로 분류됩니다.

![](/31/duck.png)

공격하고자 하는 모델에 접근이 가능한 화이트 박스 (white box) 공격은 모델의 가중치(weight)에 접근할 수 있으므로, 입력 이미지에 대한 손실 함수 (loss function)의 경사도(gradient)를 구할 수 있습니다. 이렇게 구한 경사도는 적대적 이미지를 생성할 때 이용됩니다. 



#### 전이 기반 적대적 공격(Transfer-Based Adversarial-Attack)

공격하고자 하는 모델에 **_접근이 불가능한 경우_**라면, 적대적 이미지의 **전이성**을 이용하여 **전이 기반 적대적 공격**을 시도해야 합니다. 이는 소스 모델에 화이트 박스 공격을 가해 생성한 적대적 이미지를 통해 타겟 모델도 공격하는 것입니다. 따라서 전이 기반 적대적 공격 성공률을 향상시키기 위해서는 적대적 이미지 형성 시, 적대적인 이미지가 소스 모델에 의존하여 소스 모델에서만 높은 성능을 보이게 되는 _**과적합(overfitting)**_ 현상을 방지하는 것이 매우 중요합니다.

**Diversity Input Method (DI 기법)** 은 **랜덤 크키 변환**과 **랜덤 패딩**을 거친 이미지를 모델의 입력으로 사용하여 적대적 이미지를 생성합니다. 이는 적대적인 이미지는 크키와 위치가 변화하더라도 적대적으로 작용해야 한다는 가정에서 착안합니다. 이를 통해 적대적 이미지가 소스 모델에 _과적합_ 되는 현상을 방지하여, 여러 모델에서 적대성을 유지합니다. 



## 2. Method

DI 기법의 핵심 아이디어는 **랜덤 크키 변환(randomly resizing)**과 **랜덤 패딩(random padding)** 된 이미지의 경사도를 사용함으로써 적대적 이미지가 소스 모델에 의존하는 현상을 방지한 것입니다. 이 변환 과정을 DI 변환 이라고 하겠습니다. 아래 이미지는 원본 이미지와 DI 변환 후의 이미지를 비교한 것 입니다.

![image-20211024163337664](C:\Users\hskim\AppData\Roaming\Typora\typora-user-images\image-20211024163337664.png)



본 논문에서 DI 변환을 구현한 방법은 다음과 같습니다 :

* **랜덤 크키 변환** : 이미지를 rnd × rnd × 3 로 크기 변환 (rnd ∈ [299, 330))

* **랜덤 패딩** : 이미지를 330 × 330 × 3 이 되도록 상하좌우에 랜덤하게 패딩



본 논문에서는 TensorFlow를 사용하였으며, DI 변환 이후 이미지 사이즈를 330 × 330 × 3으로 고정시켜 구현했습니다. (이후, 모델 입력 사이즈에 맞춰 다시 이미지 크기변환을 진행합니다.) 저는 Python을 이용해 논문의 _랜덤 크키 변환_과 _랜덤 패딩_의 과정을 유지하되, DI 변환 이후의 이미지 사이즈를 원본 이미지 사이즈와 동일하도록 코드를 구현하여 후처리 과정을 거치지 않아도 되도록 구현했습니다.

## 3. Implementation

This section covers the actual implementation.

When you write the manuscript, please follow the rules below:

* Use`code block`when you write codes.
* Use **Python** language, especially version 3 \(3.8 &gt;= recommended\).
* Use **PyTorch**, **TensorFlow**, and **JAX** \(**Numpy** is okay\) for the deep learning library.
* Use _manual seed_.
* A module should be implemented in a _function_ or _class_.
* Do not use **GPU,** but use **CPU** instead.
* Use _4 spaces_ \(_= 1 tab_\) for indentation.
* _Type hint_ is optional.
* Naming convention
  * _class_ name: `CamelCaseNaming`
  * _function_ and _variable_ name: `snake_case_naming`

### 

### Environment

DI 기법 구현과정에서 필요한 환경(env_di-fgsm)을 yml 파일로 만들었습니다. 아나콘다 가상환경을 이용하며, 아래의 명령어를 입력해 환경설정을 할 수 있습니다.

```bash
# Environment setup using conda
conda env create -f env_di-fgsm.yml
```



### DIFGSM.py

```python
## DI-FGSM

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torchgeometry as tgm
from attack import Attack


class DIFGSM(Attack):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=20, decay=0.0,
                 di_pad_amount=31, di_prob=0.5, random_start=False):
        super().__init__("HSFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.di_pad_amount = di_pad_amount
        self.di_prob = di_prob
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']

    def diverse_input(self, x_adv):
        x_di = x_adv  # .clone().detach()
        ori_size = x_di.shape[-1]
        rnd = int(torch.rand(1) * self.di_pad_amount) + ori_size
        # x_di = transforms.Resize((rnd, rnd), interpolation=InterpolationMode.NEAREST)(x_di)

        h, w = x_di.shape[-1], x_di.shape[-1]

        pad_max = ori_size + self.di_pad_amount - rnd
        pad_left = int(torch.rand(1) * pad_max)
        pad_right = pad_max - pad_left
        pad_top = int(torch.rand(1) * pad_max)
        pad_bottom = pad_max - pad_top
        max_size = x_di.shape[-1]
        
        points_src = torch.FloatTensor([[
            [0, 0], [w - 1, 0], [w - 1 + 0, h - 1 + 0], [0, h - 1 + 0],
        ]])
        points_dst = torch.FloatTensor([[
            [pad_left, pad_top], [ori_size - pad_right - 1, pad_top],
            [ori_size - pad_right - 1, ori_size - pad_bottom - 1], [pad_left, ori_size - pad_bottom - 1],
        ]])

        M = tgm.get_perspective_transform(points_src, points_dst)
        x_di = tgm.warp_perspective(x_di, torch.cat(x_di.shape[0] * [M]).cuda(), dsize=(max_size, max_size)).cuda()
        x_di = transforms.Resize((ori_size, ori_size), interpolation=InterpolationMode.NEAREST)(x_di)
        cond = torch.rand(x_adv.shape[0]) < self.di_prob
        cond = cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x_di = torch.where(cond.cuda(), x_di, x_adv)

        return x_di

    def forward(self, images, labels):
        """
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(self.diverse_input(adv_images))

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
```
{% endcode %}

### Module 1

{% hint style="info" %}
You can freely change name of the subsection \(Module 1\) and add subsections.
{% endhint %}

Please provide the implementation of each module or algorithm with detailed \(line-by-line\) comments.

**Note that you must specify the shape of input, intermediate, and output tensors.**

{% tabs %}
{% tab title="Implementation 1" %}
You can add code blocks with multiple tabs.

{% code title="example2.py" %}



### Example code

```python
## Example - Transfer Attack

from torch.utils.data import DataLoader, TensorDataset
import torchvision.utils
import torchvision.datasets as dsets
import warnings
warnings.filterwarnings('ignore')
from models import Source, Target
from DIFGSM import *


## check version
print("PyTorch", torch.__version__)
print("Torchvision", torchvision.__version__)


## Load Data
batch_size = 24

cifar10_train = dsets.CIFAR10(root='./data', train=True,
                              download=True, transform=transforms.ToTensor())
cifar10_test  = dsets.CIFAR10(root='./data', train=False,
                              download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(cifar10_train,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(cifar10_test,
                                          batch_size=batch_size,
                                          shuffle=False)


## Attack Source Model & Save Adversarial Images
model = Source()
model.load_state_dict(torch.load("./data/source.pth"))
model = model.eval().cuda()

atk = DIFGSM(model, eps=16 / 255, alpha=2 / 255, steps=10, di_pad_amount=5)
atk.set_return_type('int') # Save as integer.
print('\n#################Source Model#################')
atk.save(data_loader=test_loader, save_path="./data/cifar10_DIFGSM.pt", verbose=True)


## Load Adversarial Images & Attack Target Model
adv_images, adv_labels = torch.load("./data/cifar10_DIFGSM.pt")
adv_data = TensorDataset(adv_images.float()/255, adv_labels)
adv_loader = DataLoader(adv_data, batch_size=128, shuffle=False)

model = Target().cuda()
model.load_state_dict(torch.load("./data/target.pth"))


## Target Model : Clean Accuracy
print('#################Target Model#################')
model.eval()
correct = 0
total = 0

for images, labels in test_loader:
    images = images.cuda()
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Standard accuracy: %.2f %%' % (100 * float(correct) / total))


## Target Model : Robust Accuracy
model.eval()
correct = 0
total = 0

for images, labels in adv_loader:
    images = images.cuda()
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Robust accuracy: %.2f %%' % (100 * float(correct) / total))

```


### Results

```bash
PyTorch 1.9.0
Torchvision 0.10.0
Files already downloaded and verified
Files already downloaded and verified

#################Source Model#################
- Save progress: 100.00 % / Accuracy: 0.03 % / L2: 2.26462 (0.039 it/s) 	
- Save complete! 

#################Target Model#################
Standard accuracy: 87.26 %
Robust accuracy: 38.84 %

Process finished with exit code 0
```


## Author / Reviewer information

### Author

**김희선 \(Hee-Seon Kim\)**

* KAIST EE
* hskim98@kaist.ac.kr



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
