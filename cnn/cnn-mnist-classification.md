---
description: CNN을 이용하여 유명한 MNIST 데이터셋을 분류하는 페이지다.
---

# CNN MNIST Classification

{% hint style="warning" %}
```python
nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

```

CNN MNIST Classification 같은 경우에, 밑 GitBook 설명보다는 필자가 작성 Google Colab에서의 예제가 코드 생략없이 더 자세한 내용을 담고 있다.  
[https://colab.research.google.com/drive/1\_iaYMJ21BiDfrURCTsOCtxDOrUiHHbvX?usp=sharing\#scrollTo=YHe2argxK1Fd](https://colab.research.google.com/drive/1_iaYMJ21BiDfrURCTsOCtxDOrUiHHbvX?usp=sharing#scrollTo=YHe2argxK1Fd)
{% endhint %}

```python
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.init
```

```python
if torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
```

torch와 딥러닝 관련 라이브러리를 모두 import 해준다. torch 같은 경우에는 학습을 진행하는 주체가 CPU일 수도 있고, GPU일 수도 있다. 이런 경우를 대비하여, `toch.cuda.is_available()`를 이용하여 자신의 작업환경이 GPU를 지원하는지 체크한다.   
딥러닝 과정을 거치면서도, torch에서는 랜덤성을 가지는 요인들이 존재한다. 이를 배제하기 위해서 Seed값을 고정함으로써, 이후에 코드를 재연할 수 있도록 한다.

```python
learning_rate = 0.001
training_epochs = 15
batch_size = 100
```

Learning rate\(학습률\)은 Linear Regression\(선형 회귀\) 부분에서 기본적인 설명이 되어있다. 그리고, epoch = 15, batch\_size = 100으로 설정해주었다. 

```python
mnist_train = datasets.MNIST(root='MNIST_data/',
    train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root='MNIST_data/',
    train=False, transform=transforms.ToTensor(), download=True)
```

MNIST 데이터셋을 받아줍니다. `root = 'MNIST_data/'`를 이용하여, 같은 폴더내에 MNIST\_data라는 폴더를 생성하고, 여기에 `download = True`로 파일을 다운로드한. 만약, 다운로드 경로를 지정해주지 않는다면, 해당 폴더에 데이터를 다운로드한. 훈련 데이터는 `train = True`, 테스트 데이터는 `train = False`로 한다.  이를 실행시키면, 밑과 같이 알아서 다운로드를 진행하는 것을 볼 수 있다.

![](../.gitbook/assets/image%20%2826%29.png)

```python
data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size,
    shuffle=True, drop_last=True)
```

파이토치에서는 데이터를 쉽게 다룰 수 있도록 Dataset과 DataLoader를 제공한다. 이를 사용하여, 미니 배치 학습, 데이터 셔플, 병렬 처리까지 간단하게 수행할 수 있다. 일단, 기본적으로 DataSet을 정의하고, 이를 DataLoader에 전달함으로써 데이터를 로드하는 것이다.  

```python
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # Flatten for Full-Contacted Layer 
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
```

`torch.nn.Sequential()`은 편하게 함수를 순차적으로 실행할수 있도록 하는 매크로라고 생각하면 편하다. Sequential 함수가 없다면, out = self.layer1\(x\)가 아닌, Sequential 함수 안에 있는 모든 작업을 매번 입력해주어야 할 것이다. \(파이토치에서는 선형회귀 모델을 torch.nn.Linear로 제공하고 있다. parameter로는 input과 ouput의 dimension으로 받고 있다\)  
\(`torch.nn.init.xavier_uniform_(self.fc.weight)`은 fc layer 한정으로 가중치를 초기화함을 말한다.  그리고, `out.view`를 이용하여 Flatten한다.\)

```python
model = CNN().to(device)

costFunction = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```

이미 위 코드에서 Sequential 작업을 마쳤기 때문에, `model = CNN().to(device)`만으로 모델 생성이 이루어졌다. 그리고, 이제 Cost Function과 optimizer를 정의해야 한다. SoftMax 함수가 포함되어 있는 Cost Function과 WING AI 분과 '대충 알아보기 세미나'에서 언급되었던 Adam을  optimizer를 이용한다 \(SGD를 이용해도 무방하다\)

```python
total_batch = len(data_loader)
print('총 배치의 수 : {}'.format(total_batch))
```

`총 배치의 수 : 600`  
위에서 배치의 크기는 100으로 설정했고, 배치의 수는 600개이므로 총 훈련 데이터는 60,000개임을 알 수 있다. 이제 모델을 훈련시킬 일이 남았다.

```python
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = costFunction(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:.9}'.format(epoch+1, avg_cost))
```

앞서 정의했던 data\_loader를 이용하여, for문과 함께 미니 배치 단위로 훈련을 진행한다. X는 미니 배치, Y는 레이블을 뜻한다. 그리고, `optimizer.zero_grad()`를 이용하여 역전파 단계 이전에 gradient를 0으로 만든다. 그리고 Cost 변수에다가 비용 함수의 결괏값을 매번 저장해다.

`cost.backward()  
optimizer.step()`

위 두 줄이 CNN의 핵심 중 하나이다. 첫번째 줄은 역전파 단계로서, 모델의 매개변수에 따라 손실\(loss\)의 변화도를 계산한다. 그리고, 두번째 줄에서는 optimizer의 step함수를 호출하여 매개변수를 갱신하는 것이다.   
그리고, loss의 평균을 계산해서, 직접 확인할 수 있도록 출력한다. 위 코드는 epochs=15를 모두 달성하기 위 훈련과정의 시간이 꽤 걸린다.

![](../.gitbook/assets/image%20%2813%29.png)

```python
with torch.no_grad():
    X_test = mnist_test.data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.targets.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
```

`torch.no_grad()`를 이용해서 파이토치의 autograd engine을 꺼버린다. 이미 역전파도 끝낸 상태이며, 더 이상 gradient를 계산할 필요가 없으므로, 메모리 사용량도 줄이는 동시에 연산속도를 높이기 위해서 이런 과정을 추가했다. 밑과 같이 약 98%의 정확도를 보였다.

![](../.gitbook/assets/figure_1.png)

