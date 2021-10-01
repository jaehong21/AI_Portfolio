---
description: 저번 페이지에 제시되어 있는 기본적인 개념을 Python을 이용하여 간단히 구현해 보는 페이지입니다.
---

# Simple Example using Python

Numpy\(Numerical Python\)은 파이썬 라이브러리로서 벡터 및 행렬 연산에 있어서 편리한 기능을 제공합니다. 이후에 pandas, matplotlib라는 라이브러리와 함께 계속하여 사용할 것입니다. 

```python
import numpy as np
x_train = np.array ( [1., 2., 3., 4., 5., 6.] )
y_train = np.array ( [11., 15., 19., 23., 27., 31.] )
```

sample 값과 target을 위와 같이 정의합니다. 이 데이터를 가장 잘 나타내는 선은   
H\(x\) = 4x + 7입니다. 즉, 학습을 완료한 이후, W와 b는 각각 4, 7의 값을 얻어야 할 것입니다.

```python
W = 0.0
b = 0.0

n_data = len(x_train) # 6
epoch = 5000
learning_rate = 0.01 
```

초기의 가중치는  대부분 random으로 설정하지만, 이 예제에서는 0으로 초기화 했습니다. epoch는 한 번 학습을 하는 사이클을 의미하여, 이 예제에서는 학습 횟수를 5000으로 설정하였습니다. 

```python
for i in range(epoch):
    predict_function = x_train * W + b
    cost = np.sum((predict_function - y_train) ** 2 / n_data
```

`predict_function`은 우리가 구하고자 하는 예측함수입니다. 그리고, 이전 페이지에서 언급했던 Cost Function을 위와 같이 예측 함수에서 실제 데이터와의 차이를 제곱한 합의 형태를 데이터의 수로 나눈 방식으로 python에서 구현했습니다. 

```python
# 아래 있는 code는  coding block 'for'에 포함되어 있는 코드입니다

gradient_w = np.sum((W*x_train+b - y_train) * 2 * x_train) / n_data   
gradient_b = np.sum((W*x_train+b - y_train) * 2 ) / n_data

W = W - learning_rate * gradient_w
b = b - learning_rate * gradient_b

if i % 200 == 0: 
     print('Epoch({:4.0f}/{:.0f}) cost:{:15.10f}, W:{:.10f}, b:{:.10f}'
     .format(i, epoch, cost, W, b) )
```

`gradient_w`는 Cost Function 혹은 MSE를 W에 대해 편미분한 값입니다. 그리고, 학습률과 곱하여 경사하강법을 적용하여 W를 계속하여 새로운 값으로 초기화 해줍니다.   
`gradient_b`는 MSE를 b에 대해 편미분한 값입니다. 

이후 200번째 사이클마다 학습 횟수/cost/W 값/b 값을 출력하도록 합니다.  
format\( \) 함수를 이용하여 프린트문을 출력하고자 합니다.   
`{ : 15.10f }`라고 했을 때,  실수가 15칸의 자리공간을 차지하며, 소수점 10자리까지 표시하겠다는 뜻입니다.

![](../.gitbook/assets/image%20%2817%29.png)

이를 출력하면, 위와 같은 결과를 얻을 수 있습니다. 학습을 반복할수록 cost가 줄어듦과 동시에 W와 b의 값이 우리가 목표한 값\(W=4, b=7\)에 점점 가까워짐을 확인할 수 있습니다. 그리고, 이전 페이지 '수렴'에서 언급했듯이 사이클을 반복할수록 변동폭이 줄어듦을 알 수 있습니다. 

![Final result after 5000 cycles](../.gitbook/assets/image%20%2839%29.png)

