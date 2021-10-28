---
description: (Description) 1st auhor / Paper name / Venue
---

# \(Template\) Title \[Language\]

## Guideline

{% hint style="warning" %}
Remove this section when you submit the manuscript
{% endhint %}

Write the manuscript/draft by editing this file.

### Title & Description

Title of an article must follow this form: _Title of article \[language\]_

#### Example

* Standardized Max Logit \[Kor\]
* VITON-HD: High-Resolution Virtual Try-On \[Eng\]
* Image-to-Image Translation via GDWCT \[Kor\]
* Coloring with Words \[Eng\]
* ...

Description of an article must follow this form: _&lt;1st author&gt; / &lt;paper name&gt; / &lt;venue&gt;_

#### Example

* Jung et al. / Standardized Max Logit: A simple yet Effective Approach for Identifying Unexpected Road Obstacles in Urban-scene Segmentation / ICCV 2021 Oral
* Kim et al. / Deep Edge-Aware Interactive Colorization against Color-Bleeding Effects / ICCV 2021 Oral
* Choi et al. / RobustNet: Improving Domain Generalization in Urban-Scene Segmentation via Instance Selective Whitening / CVPR 2021 Oral
* ...

## \(Start your manuscript from here\)

{% hint style="info" %}
If you are writing manuscripts in both Korean and English, add one of these lines.

You need to add hyperlink to the manuscript written in the other language.
{% endhint %}

{% hint style="warning" %}
Remove this part if you are writing manuscript in a single language.
{% endhint %}

\(In English article\) ---&gt; 한국어로 쓰인 리뷰를 읽고 싶으시면 **여기**를 누르세요.

\(한국어 리뷰에서\) ---&gt; **English version** of this article is available.

## 1. Introduction

Please provide the general information of the selected paper / method.  
This can be a shortened version of **Paper review**.

## 2. Method

In this section, you need to describe the method or algorithm in theory.

Also, please provide us a working example that describe how the proposed method works.  
Watch the professor's [lecture videos](https://www.youtube.com/playlist?list=PLODUp92zx-j8z76RaVka54d3cjTx00q2N) and see how the professor explains.

Note that you can attach images and tables in this manuscript.  
When you upload those files, please read [How to contribute?](../../how-to-contribute.md#image-file-upload) section.

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

### Environment

{% hint style="info" %}
You can use _**Hint block**_ in this section.
{% endhint %}

Please provide the dependency information and manual seed for reproducibility.

```bash
# Environment setup using conda
conda create -n tutorial python=3.8
conda activate tutorial
conda install ...
# or
pip install ...
```

{% code title="example1.py" %}
```python
import os
import sys
import random
from typing import List, Dict, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# please provide version information
print(sys.version)
print(np.__version__)
print(torch.__version__)

# you should set manual seed
my_seed = 7777
random.seed(my_seed)
torch.manual_seed(my_seed)
torch.cuda.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
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
```python
class MyModule(nn.Module):

    def __init__(self, ...):
        
        self.temp = nn.Linear(...)
        
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
    
        # input
        # x (batch, dim1, dim2, ...)
        # y (batch, dim1, dim2, ...)
        # return
        # out (batch, ...)
    
        out = self.temp(x) # fc-layer (batch, ...)
        ...
        
        return out
        
if __name__ == '__main__':
    
    test_x = torch.randn(...)
    test_model = MyModule(...)
    test_out = test_model(x)
    
    print(test_x)
    print(test_out)
    print(test_out.size())
```
{% endcode %}
{% endtab %}

{% tab title="Implementation 2" %}
You can add code blocks with multiple tabs.

{% code title="example3.py" %}
```python
class MyModule(nn.Module):

    def __init__(self, ...):
        
        self.temp = ...
        
    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    
        # input
        # x (batch, dim1, dim2, ...)
        # y (batch, dim1, dim2, ...)
        # output
        # out1 (batch, ...)
        # out2 (batch, ...)
        
        out = x + y # add two tensors (batch, ...)
        ...
        
        return out1, out2
        
if __name__ == '__main__':
    
    test_x = torch.randn(...)
    test_y = torch.randn(...)
    test_model = MyModule(...)
    test_out = test_model(x, y)
    
    print(test_x)
    print(test_y)
    print(test_out)
    print(test_out.size())
```
{% endcode %}
{% endtab %}
{% endtabs %}

### \(Module 2 ...\)

{% code title="hi.py" %}
```bash
# you can add subsections if you need
```
{% endcode %}

## Author / Reviewer information

{% hint style="warning" %}
You don't need to provide the reviewer information at the draft submission stage.
{% endhint %}

### Author

**Korean Name \(English name\)** 

* Affiliation \(KAIST AI / NAVER\)
* \(optional\) 1~2 line self-introduction
* Contact information \(Personal webpage, GitHub, LinkedIn, ...\)
* **...**

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

