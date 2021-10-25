---
description: Xie et al. / Improving transferability of adversarial examples with input diversity/ CVPR 2019
---



# Diversity Input Method \[Eng]

한국어로 쓰인 리뷰를 읽고 싶으시면 **[여기](cvpr-2019-inputdiversity-kor.md)** 를 누르세요.



## 1. Introduction

### ✔Adversarial Attack

​	**Adversarial Attack** is a technique that induces **incorrect prediction** of the model by intentionally adding _noise_ to the image as shown in the figure. Adversarial attacks are classified into targeted attacks and non-targeted attacks. Targeted attack is an attack that induces the prediction of the target model into a specific class. And non-targeted is and attack that does not induce, but simply mispredicts.

![](C:\Users\.gitbook\assets\31\duck.png)

​	A _white box attack_ that can access the model to be attacked, also can access the weight of the model, so it is possible to obtain the **gradient** of the loss function for the input image. This gradient is used to create an adversarial image.



### ✔Transfer-Based Adversarial-Attack

​	If the model you want to attack is **_inaccessible_**, you should try **transfer-based adversarial attack** using **transferability** of your adversarial image. This is an adversarial image created by a white box attack on the _source model_, also attacks the _target model_. Therefore, in order to improve the transfer-based adversarial attack success rate, it is very important to prevent _**overfitting**_ phenomenon in which the adversarial image depends on the source model and shows high performance only in the source model.

​	**Diversity Input Method(DIM)** generates a adversarial image using an image that has undergone **random resizing** and **random padding** as input to the model. This is based on the assumption that a adversarial image should act adversarially even if its size and location change. This prevents adversarial images from _overfitting_ the source model, maintaining adversity across multiple models.





## 2. Method

### Diversity Input Method(DIM)✨

​	The core idea of the Diversity Input Method(DIM) is to avoid the dependence of the adversarial image on the source model by using the slope of the transformed image with **randomly resizing** and **random padding**. This tranform process will be called DI transform. The image below compares the original image with the image after DI transform.

<p align="center"> 	<img src="../../.gitbook/assets/31/di.png"> </p>

The implementation of the DI transformation in this paper is as follows:

* **randomly resizing** : Resize image to rnd × rnd × 3 (rnd ∈ [299, 330))
* **random padding** : Randomly pad the image to the top, bottom, left, and right so that it is 330 × 330 × 3

​	In this paper, TensorFlow is used, and the image size is fixed to 330 × 330 × 3 after DI transform. (After that, the image size is converted again according to the model input size.)  I use **PyTorch** to maintain the process of _random resizing_ and _random padding_ of the paper, but change the image size after DI transform as original image. In this way, it does not have to go through the post-processing process.

​	DI transform has the advantage that it can be used with known transfer-based adversarial attacks (I-FGSM, MI-FGSM). In the case of attacking using the I-FGSM attack technique with DI transform, it will be referred to as **DI-FGSM**. In the related work below, I will also introduce each attack method.



### Related work✨

#### 1) Iterative Fast Gradient Sign Method (I-FGSM)

​	The fast gradient sign method (FGSM) changes each pixel of X by ε in the direction of increasing loss function L(X,y(true)) for the input image X and the real class y(true), to create a hostile image X^{ adv}.
$$
X^{adv}=X+ε·sign(∇_X L(X,y^{true})).
$$
​	iterative Fast Gradient Sign Method (I-FGSM) is that repeatedly executes an FGSM attack that changes each pixel by α.
$$
X_0^{adv}=X,
$$

$$
X_{n+1}^{adv}=Clip_X^ε(X_n^{adv}+α·sign(∇_X L(X_n^{adv},y^{true})).
$$



#### 2)  momentum iterative FGSM (MI-FGSM)

​	As a method of preventing overfitting to the source model, there is a method using momentum (MI-FGSM). MI-FGSM is iteratively performed like I-FGSM, and it accumulates gradient (gt) information from the beginning to the present and uses it for adversarial image update. The difference is that the sign of gt is used for update, not the sign of the loss function.
$$
g_{n+1}= μg_n + {∇_X L(X_n^{adv},y^{true} )\over ||∇_X L(X_n^{adv},y^{true})||_1 },
$$

$$
X_{n+1}^{adv}=X_{n}^{adv} +α·sign(g_{t+1}).
$$

​	Accumulating gradients helps not to fall into a poor local maxima, and it is stable because the direction of the repeatedly updated adversarial change is similar to that of I-FGSM. Therefore, MI-FGSM shows better transferability than I-FGSM.



## 3. Implementation

* Use **Python** language, version >= 3.6

* Using **PyTorch** in the code implementation process

* Use _manual seed_ : used to fix randomness (included in the example code below)

  

### 🔨 Environment

​	The environment **(env_di-fgsm.yml)** required in the process of implementing the DI transform was created as a yml file. You can use the Anaconda virtual environment and set the environment by entering the following command.

```bash
# Environment setup using conda
conda env create -f env_di-fgsm.yml
```



### 📋DI-FGSM

​	In this file, DI-FGSM is implemented. I used  _**comments **_to explain the overall code. The size of tensors is shown as an example based on the CIFAR-10 image (size: 32, 32) which used in the example file (Transfer Attack.py) to be introduced below.

​	The **_diverse_input_** function part in class DIFGSM is the core part of DI-FGSM. **Random resizing** and **Random padding** parts are implemented. After calling the _diverse_input_ function in the _forward_ function, backpropagation occurs.

```python
## DI-FGSM : DIFGSM.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torchgeometry as tgm
from attack import Attack


class DIFGSM(Attack):

    def __init__(self, model, eps=8/255, alpha=2/255, steps=20, di_pad_amount=31, di_prob=0.5):
        super().__init__("DIFGSM", model)
        self.eps = eps # Maximum change in one pixel for total step (range 0-255)
        self.steps = steps # number of di-fgsm steps
        self.alpha = alpha # Maximum change in one pixel for one step (range 0-255)
        self.di_pad_amount = di_pad_amount # Maximum value that can be padded
        self.di_prob = di_prob # Probability of deciding whether to apply DI transform or not
        self._supported_mode = ['default', 'targeted'] # deciding targeted attack or not

    def diverse_input(self, x_adv):
        x_di = x_adv # size : [24,3,32,32]
        h, w = x_di.shape[2], x_di.shape[3] # original image size, h: 32, w: 32
        # random value that be padded
        pad_max = self.di_pad_amount - int(torch.rand(1) * self.di_pad_amount) # pad_max : 2
        # random value that be padded left
        pad_left = int(torch.rand(1) * pad_max) # pad_left : 1
        # random value that be padded right
        pad_right = pad_max - pad_left # pad_right : 1
        # random value that be padded top
        pad_top = int(torch.rand(1) * pad_max) # pad_top : 1
        # random value that be padded bottom
        pad_bottom = pad_max - pad_top  # pad_bottom : 1

        # four vertices of the original image
        # tensor([[[ 0.,  0.], [31.,  0.], [31., 31.], [ 0., 31.]]])
        points_src = torch.FloatTensor([[
            [0, 0], [w - 1, 0], [w - 1 + 0, h - 1 + 0], [0, h - 1 + 0],
        ]]) 

        # four vertices of the image after DI transform
        # tensor([[[ 1.,  1.], [30.,  1.], [30., 30.], [ 1., 30.]]])
        points_dst = torch.FloatTensor([[
            [pad_left, pad_top], [w - pad_right - 1, pad_top],
            [w - pad_right - 1, h - pad_bottom - 1], [pad_left, h - pad_bottom - 1],
        ]]) 

        # Matrix used in the transformation process
        # tensor([[[0.9355, 0.0000, 1.0000], [0.0000, 0.9355, 1.0000], [0.0000, 0.0000, 1.0000]]])
        M = tgm.get_perspective_transform(points_src, points_dst) 
        
        # The image is resized and padded so that the vertices of the original image go to the new vertices.
        x_di = tgm.warp_perspective(x_di, torch.cat(x_di.shape[0] * [M]).cuda(), dsize=(w, h)).cuda()
        x_di = transforms.Resize((w, h), interpolation=InterpolationMode.NEAREST)(x_di)
        
        # If the random value is less than or equal to di_prob, di conversion does not occur.
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

        if self._targeted: # targeted attack case, get target label
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss() # use Cross-Entropy loss for classification
        adv_images = images.clone().detach()


        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(self.diverse_input(adv_images)) # after DI transform image

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels) # targeted attack case, use -loss function
            else:
                cost = loss(outputs, labels) # else, (untargeted attack case), use +loss function

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)

            adv_images = adv_images.detach() + self.alpha*grad.sign() # I-fgsm step
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps) # limiting changes beyond epsilon
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
```



### 📋Example code

In _**Transfer Attack.py**_, I tested the performance of Transfer Attack using DI-FGSM.

##3 : This part indicate the attack process and result of the source model. You can specify an attack as _atk = DIFGSM(model, eps=16 / 255, alpha=2 / 255, steps=10, di_pad_amount=5)_. 

##5, ##6: Shows the **clean accuracy** tested on the target model with the validation set, and the **robust accuracy** tested with the adversarial image created in ##3.

#### example🚀 

```python
##0 Example - Transfer Attack.py

from torch.utils.data import DataLoader, TensorDataset
import torchvision.utils
import torchvision.datasets as dsets
import random
import warnings
warnings.filterwarnings('ignore')
from models import Source, Target
from DIFGSM import *


##1 check version
print("PyTorch", torch.__version__)
print("Torchvision", torchvision.__version__)

my_seed = 7777
random.seed(my_seed)
torch.manual_seed(my_seed)
torch.cuda.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

##2 Load Data
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


##3 Attack Source Model & Save Adversarial Images
model = Source()
model.load_state_dict(torch.load("./data/source.pth"))
model = model.eval().cuda()

atk = DIFGSM(model, eps=16 / 255, alpha=2 / 255, steps=10, di_pad_amount=5)
atk.set_return_type('int') # Save as integer.
print('\n#################Source Model#################')
atk.save(data_loader=test_loader, save_path="./data/cifar10_DIFGSM.pt", verbose=True)


##4 Load Adversarial Images & Attack Target Model
adv_images, adv_labels = torch.load("./data/cifar10_DIFGSM.pt")
adv_data = TensorDataset(adv_images.float()/255, adv_labels)
adv_loader = DataLoader(adv_data, batch_size=128, shuffle=False)

model = Target().cuda()
model.load_state_dict(torch.load("./data/target.pth"))


##5 Target Model : Clean Accuracy
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


##6 Target Model : Robust Accuracy
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

#### results🚀 

​	The **clean accuracy** tested with the _validation set_ on the **target model** is 87.26 %, which shows relatively high comparative performance.

​	On the other hand, the **robust accuracy** of testing the **target model** performance with an _adversarial image_ made with DI-FGSM through the _source model_ showed low performance at 38.87%, indicating that it is a _**successful transfer-based adversarial attack**_.



```bash
System :  3.6.13 |Anaconda, Inc.| (default, Mar 16 2021, 11:37:27) [MSC v.1916 64 bit (AMD64)]
Nunpy :  1.19.2
PyTorch :  1.9.0
Files already downloaded and verified
Files already downloaded and verified

#################Source Model#################
- Save progress: 100.00 % / Accuracy: 0.03 % / L2: 2.26292 (0.047 it/s) 	
- Save complete! 

#################Target Model#################
Standard accuracy: 87.26 %
Robust accuracy: 38.87 %

Process finished with exit code 0

```



## Author / Reviewer information

### Author😍

**Hee-Seon Kim**

* KAIST EE 
* https://github.com/khslily98
* hskim98@kaist.ac.kr



### Reviewer😍

1. Korean name \(English name\): Affiliation / Contact information
2. Korean name \(English name\): Affiliation / Contact information
3. ...

## Reference & Additional materials

1. Citation of this paper
2. Official \(unofficial\) GitHub repository
3. Citation of related work
4. Other useful materials
5. ...

