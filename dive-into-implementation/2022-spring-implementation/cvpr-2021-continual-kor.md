---
description: Mai, Zheda / Supervised contrastive replay- Revisiting the nearest class mean classifier in online class-incremental continual learning / CVPR 2021
---

Supervised Contrastive Replay: Revisiting the Nearest Class Mean Classifier in Online Class-Incremental Continual Learning[Kor]



## 1. Introduction

### Continaul Learning (CL)
CL이란, 연속적으로 주어지는 Data Stream을 Input으로 받아, 연속적으로 학습하는 모델을 만들어내는 것을 목표로 하는 문제 세팅입니다. 현재 딥 러닝 기반의 모델들은, 새로운 데이터셋을 학습할 경우 이전 데이터셋에서의 성능은 매우 떨어집니다. 이러한 현상을 Catastrophic Forgetting(CF)라고 부릅니다. 예를 들어 설명하자면, Cifar10을 학습한 모델이 MNIST를 학습할 경우, MNIST에서의 성능은 높지만, Cifar10의 성능은 낮아집니다.(단순히 MNIST를 트레이닝 한 경우, 거의 0%에 가까운 성능을 보입니다.) 이저에 Cifar10에서의 성능이 어땟던 간에, 극적인 성능 하락이 나타나게 됩니다. 이때 Cifar10과 MNIST 같이 연속적으로 들어오는 Dataset들을 Task라고 부릅니다.

CF는 딥 러닝이 여기저기에 쓰이고 있는 과정에서 꼭 해결해야 할 문제입니다. 한번 모델을 훈련시키고 난 후, 그 모델을 실제 서비스에 서빙할 경우 데이터는 더 쌓이게 됩니다. 하지만 이 데이터를 추가로 학습시키게 되면, 모델은 오히려 성능이 떨어질 수 있습니다. 이전에 모델을 트레이닝 할 때 사용했던 데이터를 전부 다 다시 사용하고, 추가로 추가 데이터를 넣어주어서 트레이닝을 시켜야 하는 것입니다. 이는 극적인 계산 비효율성을 부릅니다. 자동으로 데이터를 찾아서 점점 똑똑해지는, 영화와 같은 AI는 지금 나타나지 않는 이유입니다.

이러한 CF를 해결하고자 하는 문제 세팅이 CL입니다. 이 논문의 저자 Zheda Mai는 CL 분야에서 최근 좋은 논문을 많이 내며 SOTA에 가까운 방법론들을 매번 제시하고 있습니다. Mai의 논문 중에서도 이 논문은, 비록 트릭을 사용하기는 했지만 CL로서는 상상도 하지 못했던 높은 성능을 보여주는 논문이기 때문에 상당히 매력적입니다.

### Experience Replay(ER)
CL 문제 세팅에서 현재 지배적이라고 할 수 있는 방법론은 Experience Replay입니다. 단순한 방법에도 불구하고 좋은 성능을 보이고, 개선할 여지가 모듈적으로 많이 남아있기 때문에 많이 연구되고 있습니다. ER의 방법론은 간단합니다. 이전 태스크에서 몇가지 데이터를 뽑아 External Memory에 저장해둡니다. 새로운 태스크가 들어오면 External Memory에 있는 데이터와 함께 훈련시킵니다.

당연히 External Memory가 많으면 많을 수록 이전 태스크의 성능 저하를 잘 막을 수 있습니다. ER의 최종 목표는 최소한의 External Memory를 이용해서 최대한 CF를 줄이는 것 입니다.

ER의 현재 최신 세팅을 간략하게 정리하자면 다음과 같은 점이 중요하다고 할 수 있습니다.

- 현재 태스크의 batch 1개 + External Memory에서의 batch 1개를 함께 트레이닝 한다.
- External Memory의 경우 크기가 보통 작기 때문에 둘을 그대로 함께 트레이닝 해버리면 둘의 Class Imbalance가 일어나서 성능이 떨어지게 됩니다. 둘의 비율을 맞춰서 트레이닝 해 주는 것이 ER의 성능을 높이는 팁입니다.

## 2. Method

### SoftMax Classifier의 CL에서의 문제점

이 논문의 핵심 Contribution이자 저자가 주장하는 것은 Softmax Classifier의 문제점입니다. Softmax Classifier는 많은 부분에서 최고의 성능을 내고 있지만, CL에서 만큼은 좋지 않다는 것이 저자의 생각입니다. 그 이유는 다음과 같습니다.

- 새로운 클래스가 들어오는 것에 유연하지 않다
    - Softmax의 특성상 처음부터 클래스의 갯수를 정해줘야 합니다. 이 때문에 태스크가 얼마나 들어올지 모르는 CL 세팅의 특성에 맞지 않습니다. (하지만 현재 CL 연구는 대부분 태스크가 얼마나 들어올지 알고 있습니다. 이것은 후의 실험을 보시면 더 잘 이해됩니다.)
- representation과 classification이 연결되어 있지 않다
    - Encoder가 바뀔 경우 Softmax layer는 새로 훈련되어야 합니다.
- Task-recency bias
    - 이전의 여러 연구에서, Softmax classifier가 최근 태스크에 치중되는 경향이 있다는 것이 관찰되었습니다. 이는 데이터의 분포가 현재 태스크에 치중되어있는 CL의 특성상 성능에 치명적일 수 있습니다.

### Nearest Class Mean(NCM) Classifier

저자는 이를 해결하기 위해서, Few-shot learning에서 주로 사용되는 NCM Classifier를 사용하자고 주장합니다. NCM Classifier의 경우 Prototype Classifier라고도 불립니다. 이 Classifier는 트레이닝이 끝난 후, 트레이닝에 사용되었던 모든 클래스 데이터의 평균을 내어 저장합니다. 이렇게 저장된 평균값은 Prototype처럼 작동합니다. Test시, 가장 가까운 Prototype을 가지는 클래스로 클래스를 추측하게 됩니다.

NCM Classifier는 SoftMax의 문제를 해결하면서, few-shot learning처럼 data 부족 현상에 시달리는 CL과 굉장히 궁합이 잘 맞습니다.  실제로 NCM Classfier를 적용하는 것만으로도 대부분의 CL 방법론의 성능이 크게 상승합니다. 

$$u_c = \frac{1}{n_c}\sum_i f(x_i) \cdot 1\{y_i = c \}$$

$$y^* = argmin_{c=1,...,t} ||f(x) - u_c ||$$

NCM classifier를 위해 사용되는 수식은 위와 같다. 여기서 c는 클래스를 뜻하고, 1{y=c} 는 y가 c일 때문 1이라는 것을 의미한다. 클래스 별 메모리에 들어있는 데이터의 평균을 구하고, 그 평균에 가장 가까운 클래스로 Inference를 진행한다.

### Supervisied Contrastive Replay

NCM Classifier의 포텐셜을 더 높일 수 있는 방법이 SCR입니다. NCM Classifier는 Representation 간 거리를 중심으로 inference를 진행합니다. 이런 상황에서 다른 클래스는 더 멀리, 같은 클래스는 더 가까이 붙여두는 Contrastive Learning은 NCM에 큰 도움이 될 수 있습니다. 저자는 트레이닝 데이터에 단순한 Augmented View를 추가하고, 이 데이터들을 이용하여 Contrastive Learning을 진행합니다.  메모리 데이터와 현재 데이터를 함께 사용합니다.

$$L_{SCL}(Z_I) = \sum_{i\in I} \frac{1}{|P(i)|} \sum{p\in P(i)} log \frac{exp(z_i\cdot z_p / \tau)}{\sum{j \in A(i)}exp(z_i \cdot z_j / \tau) }$$

Loss 식은 위 식과 같습니다. $B = \{x_k,y_k\}_{k=1,...,b}$의 Mini Batch라고 할 때, $\tilde{B}$ $= \{  \tilde{x_k} = Aug(x_k), y_k \}_{k=1,...,b}$ 입니다. 그리고 $B_I = B \cap \tilde{B}$ 입니다.  $I$는 $B_I$의 지수들의 집합이고, $A(i)=I \setminus \{i\}$ 입니다.  $P(i) = \{p \in A(i) : y_p = y_i\}$ 입니다. 복잡해 보이지만 찬찬히 뜯어보면 어렵지 않습니다. 결국 $P(i)$는 샘플 i를 제외한 것 중에서 label이 같은 것, 그러니까 Positive sample을 의미합니다. $Z_I = \{z_i\}_{i \in I} = Model(x_i)$ 이고, $\tau$는 조정을 위한 temperature parameter 입니다.


Implementation에서는 Continual Learning의 벤치마크라고 할 수도 있는 Split Cifar-10에서 실험을 진행합니다. 일반적인 BaseLine으로 많이 사용되는 Experience Replay에 대한 구현과, 이 논문에서 제안한 NCN Classifier를 사용한 Experience Replay에 대한 구현을 준비했습니다. 

### Environment

Colab 환경에서 실험하기를 추천드립니다.

{% code title=“example1.py” %}

```python
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as D
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
```

{% endcode %}

### Setting of Continual Learning

이 챕터에서는 Continual Learning evaluation을 위한 기본적인 세팅을 준비합니다.  데이터셋은 Cifar-10을 5개의 태스크로 나눈 Split Cifar-10을 사용했습니다. 논문에서는 Reduced_ResNet18을 베이스 모델로 사용합니다. 하지만 이 Implementation에서는 구현의 간단함을 위해 작은 CNN모델을 사용합니다. 이 코드에서는 Split Cifar-10을 만들고, Reduced_ResNet18을 정의합니다.

```python
# Made Split-Cifar10

def setting_data():
	transform_train = transforms.Compose([
	        transforms.RandomCrop(32, padding=4),
	        transforms.RandomHorizontalFlip(),
	        transforms.ToTensor(),
	    ]) #settign transform
	
	    transform_test = transforms.Compose([
	        transforms.ToTensor(),
	    ])
	#원본 Cifar-10 dataset을 다운로드 받아 줍니다
	    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) 
	    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	    train_loader  = torch.utils.data.DataLoader(dataset=train_dataset,
	                                               batch_size=1,
	                                               shuffle=False)
	
	    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
	                                             batch_size=1,
	                                             shuffle=False)
#아래의 코드는 Cifar10을 임의의 순서(5로 나눈 나머지)로 5개의 태스크로 분리합니다.
	    set_x = [[] for i in range(5)]
	    set_y = [[] for i in range(5)]
	    set_x_ = [[] for i in range(5)]
	    set_y_ = [[] for i in range(5)]
	    if shuffle==False:
	        for batch_images, batch_labels in train_loader:
	          if batch_labels >= 5:
	            y = batch_labels-5
	          else :
	            y = batch_labels
	          set_x_[y].append(batch_images)
	          set_y_[y].append(batch_labels)
	        for i in range(5):
	          set_x[i] = torch.stack(set_x_[i])
	          set_y[i] = torch.stack(set_y_[i])
	        set_x_t = [[] for i in range(5)]
	        set_y_t = [[] for i in range(5)]
	        set_x_t_ = [[] for i in range(5)]
	        set_y_t_ = [[] for i in range(5)]
	        for batch_images, batch_labels in test_loader:
	          if batch_labels >= 5:
	            y = batch_labels-5
	          else :
	            y = batch_labels
	          set_x_t_[y].append(batch_images)
	          set_y_t_[y].append(batch_labels)
	        for i in range(5):
	          set_x_t[i] = torch.stack(set_x_t_[i])
	          set_y_t[i] = torch.stack(set_y_t_[i])
	return set_x,set_y,set_x_t,set_y_t
```

아래 코드는 사용될 Base CNN 모델인 Reduced ResNet18을 정의합니다. 마지막 FC 레이어를 사용하지 않는 features라는 함수가 존재하는 점이 특이할만한 점입니다. 이 features는 후에 NCM classifier를 구현할때 사용됩니다,

```python
class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    def all_forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out1))
        out += self.shortcut(x)
        out = F.relu(out)
        return out1,out
    

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Reduced_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,dropout=0.3):
        super(Reduced_ResNet, self).__init__()
        self.in_planes = 20

        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(20)
        self.layer1 = self._make_layer(block, 20, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 40, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 80, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 160, num_blocks[3], stride=2)
        self.d1 = nn.Dropout(p=dropout)
        self.linear = nn.Linear(160*block.expansion, num_classes)
        self.linear3 = nn.Linear(400, num_classes)
        self.linear2 = nn.Linear(640,400)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    def features(self, x):
        '''Features before FC layers'''
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

def reduced_ResNet18(num_classes=10):
    return Reduced_ResNet(BasicBlock,[2,2,2,2],num_classes=num_classes)

```

## Experience Replay

아래 코드는 Continual Learning에서 가장 많이 쓰이는 베이스라인 중 하나인 Experience Replay를 구현합니다. Memory size, training epoch, learning rate 등 다양한 옵션들을 바뀌며 성능이 어떻게 변하는지 알아보면 재미있을 것입니다.

먼저 아래 코드에서는 External Memory를 구현합니다. 메모리는 어떤 식으로 구현해도 상관은 없지만, 랜덤으로 메모리에 들어갈/메모리에서 뽑힐 데이터를 쉽게 구현하기 위해 클래스를 하나 만들었습니다.

```python
class Memory():
    def __init__(self,mem_size,size=32): #mem_size는 메모리의 크기를 결정합니다.
        self.mem_size = mem_size
        self.image = []
        self.label = []
        self.num_tasks = 0
        self.image_size=size
    
    def add(self,image,label): #메모리에 들어갈 image와 label을 input으로 받습니다. 선언할때 정해준 메모리 사이즈에 맞추어 자동으로 사이즈가 조정됩니다.
        self.num_tasks +=1
        image_new= []
        label_new = []
        task_size = int(self.mem_size/self.num_tasks)
        if self.num_tasks != 1 :
            for task_number in range(len(self.label)):
                numbers = np.array([i for i in range(len(self.label[task_number]))])
                choosed = np.random.choice(numbers,task_size)
                image_new.append(self.image[task_number][choosed])
                label_new.append(self.label[task_number][choosed])
        numbers = np.array([i for i in range(len(label))])
        choosed = np.random.choice(numbers,task_size)
        image_new.append(image[choosed])
        label_new.append(label[choosed])
        self.image = image_new
        self.label = label_new
        
    def pull(self,size):
#메모리에서 size만큼의 image-label 쌍을 꺼냅니다. 역시 랜덤으로 조정해줍니다.
        image = torch.stack(self.image).view(-1,3,self.image_size,self.image_size)
        label = torch.stack(self.label).view(-1)
        numbers = np.array([i for i in range(len(label))])
        choosed = np.random.choice(numbers,size)
        return image[choosed],label[choosed]
```

메모리에 들어갈 샘플과, 꺼내지는 샘플을 정하는 것은 ER method에서 중요한 부분입니다. 기본적인 ER method는 모든 것을 랜덤으로 조정하지만, MIR, GSS, ASER 등의 추가적인 메소드는 이 부분으로 주요하게 조정합니다.

메모리를 만들었으니 다음으로 진행할 것은 트레이닝, 테스트, 그리고 Continual Leaerning setting입니다. 진행하기 편하게 트레이닝과 테스트를 따로 함수화 하고, Continual Learning process는 ER 함수에서 따로 정의해줍니다.

```python
from typing_extensions import TypeAlias
def training(model,training_data,memory,opt,epoch,mem=False,mem_iter=1,mem_batch=10):
    model.train()
    dl = D.DataLoader(training_data,batch_size=10,shuffle=True)
    criterion = nn.CrossEntropyLoss()
    for ep in range(epoch):
        for i, batch_data in enumerate(dl):
            batch_x,batch_y = batch_data
            batch_x = batch_x.view(-1,3,32,32)
            batch_y = batch_y.view(-1)
            if mem==True:
                  for j in range(mem_iter) :
                        logits = model.forward(batch_x)
                        loss = criterion(logits,batch_y)
                        opt.zero_grad()
                        loss.backward()
                        mem_x, mem_y = memory.pull(mem_batch)
                        mem_x = mem_x.view(-1,3,32,32)
                        mem_y = mem_y.view(-1)
                        mem_logits = model.forward(mem_x)
                        mem_loss = criterion(mem_logits,mem_y)
                        mem_loss.backward()
            else :
                    logits = model.forward(batch_x)
                    loss = criterion(logits,batch_y)
                    opt.zero_grad()
                    loss.backward()
            opt.step()

def test(model,tls):
    accs = []
    model.eval()
    for tl in tls:
        correct = 0
        total = 0
        for x,y in tl:
            x = x
            y = y
            total += y.size(0)
            output = model(x)
            _,predicted = output.max(1)
            correct += predicted.eq(y).sum().item()
        accs.append(100*correct/total)
    return accs

def make_test_loaders(set_x_t,set_y_t):
  tls = []
  for i in range(len(set_x_t)):
    ds = D.TensorDataset(set_x_t[i].view(-1,3,32,32),set_y_t[i].view(-1))
    dl = D.DataLoader(ds,batch_size=100,shuffle=True)
    tls.append(dl)
  return tls

def ER(mem_size):
      set_x,set_y,set_x_t,set_y_t = setting_data()
      test_loaders = make_test_loaders(set_x_t,set_y_t)
      model = reduced_ResNet18()
      optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
      mem_x = []
      mem_y = []
      accs = []
      Mem = Memory(mem_size)
      for i in range(0,len(set_x)):
          training_data = D.TensorDataset(set_x[i].view(-1,3,32,32),set_y[i].view(-1))
          if i !=0:
              training(model,training_data,Mem,optimizer,1,mem=True)
          else:
              training(model,training_data,[],optimizer,1,mem=False)
          Mem.add(set_x[i].view(-1,3,32,32),set_y[i].view(-1))
          acc = test(model,test_loaders)
          accs.append(acc)
          print(acc)
          
      print('final accracy : ', np.array(acc).mean())
```

colab cpu를 사용할 경우 약 20분 정도가 소요됩니다. Memory size 1000, epoch 1의 상황에서 최종 성능의 평균은 약 34-36정도로 나온다면 훌륭합니다. 저자의 논문에 나온 평균값은 대략 37 정도입니다. learning rate을 0.05-0.08 정도로 낮춘다면 저자의 성능에 근접한 값을 얻을 수 있습니다.

### Use NCM Classifier

여기서 Contrastive Learning까지 구현하는 것은 CPU만 사용하는 특성상 어렵기 때문에, NCM Classifier를 구현하고, 성능 상승을 확인할 수 있도록 Implementation 할 것입니다.

```python
def ncm_test(model,mem_x,mem_y,tls):
    labels = np.unique(np.array(mem_y))
    classes= labels
    exemplar_means = {}
    cls_sample = {label : [] for label in labels}
    ds = D.TensorDataset(mem_x.view(-1,3,32,32),mem_y.view(-1))
    dl = D.DataLoader(ds,batch_size=1,shuffle=False)
    accs = []
    for image,label in dl:
        cls_sample[label.item()].append(image)
    for cl, exemplar in cls_sample.items():
        features = []
        for ex in exemplar:
            feature = model.features(ex.view(-1,3,32,32)).detach().clone()
            feature.data= feature.data/feature.data.norm()
            features.append(feature)
        if len(features)==0:
            mu_y = torch.normal(0,1,size=tuple(model.features(x.view(-1,3,32,32)).detach().size()))
        else :
            features = torch.stack(features)
            mu_y = features.mean(0)
        mu_y.data = mu_y.data/mu_y.data.norm()
        exemplar_means[cl] = mu_y
    with torch.no_grad():
        model = model
        for task, test_loader in enumerate(tls):
            acc = []
            correct = 0
            size =0
            for  batch_x,batch_y in test_loader:
                batch_x = batch_x
                batch_y = batch_y
                feature = model.features(batch_x)
                for j in range(feature.size(0)):
                    feature.data[j] = feature.data[j] / feature.data[j].norm()
                feature = feature.view(-1,160,1)
                means = torch.stack([exemplar_means[cls] for cls in classes]).view(-1,160)
                means = torch.stack([means] * batch_x.size(0))
                means =  means.transpose(1,2)
                feature = feature.expand_as(means)
                dists = (feature-means).pow(2).sum(1).squeeze()
                _,pred_label = dists.min(1)
                correct_cnt = (np.array(classes)[pred_label.tolist()]==batch_y.cpu().numpy()).sum().item()/batch_y.size(0)
                correct += correct_cnt * batch_y.size(0)
                size += batch_y.size(0)
            accs.append(correct/size)
        return accs

def NCM_ER(mem_size):
      set_x,set_y,set_x_t,set_y_t = setting_data()
      test_loaders = make_test_loaders(set_x_t,set_y_t)
      model = reduced_ResNet18()
      optimizer = torch.optim.SGD(model.parameters(),lr=0.05)
      mem_x = []
      mem_y = []
      accs = []
      Mem = Memory(mem_size)
      for i in range(0,len(set_x)):
          training_data = D.TensorDataset(set_x[i].view(-1,3,32,32),set_y[i].view(-1))
          if i !=0:
              training(model,training_data,Mem,optimizer,1,mem=True)
          else:
              training(model,training_data,[],optimizer,1,mem=False)
          Mem.add(set_x[i].view(-1,3,32,32),set_y[i].view(-1))
          acc = ncm_test(model,Mem,test_loaders)
          print(acc)
          
      print('final accracy : ', np.array(acc).mean())
```

{% endcode %}

NCM_ER을 이용할 경우, Colab CPU에서 약 21분이 소요됩니다. 성능은 memory size 1000 기준으로 약 38-41 정도로, 저자의 reference 값보다 낮게 나오더라도 괜찮습니다. hyperparemeter tuning을 잘 수행한다면 저자의 성능에 근접하게 성능을 올릴 수 있습니다.

### Author

권민찬 **(MINCHAN KWON)**

- KAIST AI
- https://kmc0207.github.io/CV/
- kmc0207@kaist.ac.kr

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

