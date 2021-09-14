---
description: 'CNN으로 다른 예제 구현 이전에, CNN의 Layer들부터 한 번 제작하는 페이지다.'
---

# Creating CNN Layer

드 전체적인 개요:

1. **1st Convolutional Layer:**  합성곱\(in\_channel = 1, out\_channel = 32, kernel\_size=3, stride=1, padding=1\) + 활성화 함수 ReLU MaxPooling\(kernel\_size=2, stride=2\)\) 
2. **2nd Convolutional Layer:** 합성곱\(in\_channel = 32, out\_channel = 64, kernel\_size=3, stride=1, padding=1\) + 활성화 함수 ReLU MaxPooling\(kernel\_size=2, stride=2\)\) 
3. **Full-Contacted Layer:**  Open Feature Maps Full-Connected Layer\(뉴런 10개\) + 활성화 함수 Softmax

```python
import torch
import torch.nn as nn

inputs = torch.Tensor(1,1,28,28)
print('텐서의 크기: {}'.format(inputs.shape))
```

`텐서의 크기: torch.Size([1, 1, 28, 28]`  
위에서부터 차례로 batch\_size, 채널, 높이\(height\), 너비\(width\)이다.

```python
conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
print(conv1)
conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
print(conv2)pool = nn.MaxPool2d(kernel_size=2, stride =2)
print(pool)
```

`Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1 ,1))`  
첫번째 Convolutional Layer는 1채널짜리를 입력받아서, 32채널을 뽑아낸다. 커널 사이즈는 3, padding은 1이다.

`Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))`

`MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)`  
pooling은 MaxPooling으로 구현한다. `Maxpool2d(2)`로만 입력해도, 커널 사이즈와 padding 둘 다 '2'로 지정된다.

```python
out = conv1(inputs)
print(out.shape)
out = pool(out)
print(out.shape)

out = conv2(out)
print(out.shape)
out = pool(out)
print(out.shape)
```

`torch.Size([1, 32, 28, 28])  
torch.Size([1, 32, 14, 14])  
torch.Size([1, 64, 14, 14])  
torch.Size([1, 64, 7, 7])`

```python
out = out.view(out.size(0), -1)
print(out.shape)
```

`torch.Size([1, 3136])`  
배치 차원을 제외하고, 모두 1차원으로 통합되었다. 이것이 앞서 말한 Flatten 과정이라고 보면 무방하다. 그리고, 이를 출력층인 Fully-Contacted Layer로 통과시킬 것이다.\]

```python
fc = nn.Linear(3136, 10)
#input_dimension: 3136, ouput_dimension: 10
out= fc(out)
print(out.shape)
```

출력층은 10개의 뉴런으로 배치되어 있으며, 총 10개의 차원의 텐서로 변환한다.`torch.Size([1, 10])`

