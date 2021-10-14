---
description: RNN은 간단하게, 파이썬 혹은 파이토치를 이용하여 구현해 볼 수 있다.
---

# Creating Simple RNN

## Using Python

```python
import numpy as np

timesteps = 10
# 시점의 수. NLP에서는 보통 문장의 길이가 된다.
input_size = 4
# 입력의 차원. NLP에서는 보통 단어 벡터의 차원이 된다
hidden_size = 8
# 은닉 상태의 크기 (= 메모리 셀의 용량)

inputs = np.random.random((timesteps, input_size))
# 2D 텐서

hidden_state_t = np.zeros((hidden_size))
# 초기 상태는 0벡터로 초기화
# 은닉 상태의 크기는 hidden_size로 생성
```

```python
print(hidden_state_t)
# [0. 0. 0. 0. 0. 0. 0. 0.]
```

```python
Wx = np.random.random((hidden_size, input_size))
# (8,4)크기의 2D 텐서로 입력에 대한 가중치
Wh = np.random.random((hidden_size, hidden_size))
# (8,8)크기의 2D 텐서로 은닉 상태에 대한 가중치
b = np.random.random((hidden_size)) 
# (8,) 크기의 1D 텐서 생성. 이 값은 bias

print(np.shape(Wx))
print(np.shape(Wh))
print(np.shape(b))
# (8, 4)
# (8, 8)
# (8,)
```

```python
total_hidden_states = []

# 메모리 셀 동작
for input_t in inputs:
    output_t = np.tanh(np.dot(Wx, input_t) + np.dot(Wh, hidden_state_t) + b)
    total_hidden_states.append(list(output_t))
    
    print(np.shape(total_hidden_states))
    # 각 시점인 t별 메모리 셀의 출력 크기는 (timestep, output_dim)
    hidden_state_t = output_t

total_hidden_states = np.stack(total_hidden_states, axis=0)
print(total_hidden_states)
# total_hidden_states는 (10, 8)의 모양을 가진
```

## Using Pytorch

```python
import torch
import torch.nn as nn
```

```python
input_size = 4
hidden_size = 8

inputs = torch.Tensor(1, 10, 4)
# (batch_size, time_steps, input_size)
```

```python
cell = nn.RNN(input_size, hidden_size, batch_first=True)
# batch_First=True로 입력 텐서의 첫번째 차원이 배치 크기임을 알려준다
# RNN은 num_layers, bidirectional와 같은 parameter 또한 존재한다
```

```python
outputs, _status = cell(inputs)
# RNN 셀은 두 개의 입력값을 리턴한다
# 첫번째는 모든 시점의 hidden state, 둘째는 마지막 timestep의 은닉상태이다.
```

```python
print(outputs.shape)
# 10번의 t동안 8차원의 hidden state가 출력되었다는 의미
# ([1, 10, 8])
# 여기에서 1은 이 RNN의 층이 1이였음을 나타낸
```

```python
print(_status.shape)
# 마지막 시점의 hidden state는 (1, 1, 8)의 크기를 가짐
```

다음 페이지에서는 앞서 언급 내용을 토대로, 데이터 전처리 과정부터 시작하여 '네이버 영화 데이터 리뷰 데이터셋'을 이용하여 감정 분석(Sentimental Classification) task를 다뤄보고자 한다.

사진 및 내용 출처: \
[https://wikidocs.net/24996](https://wikidocs.net/60690)
