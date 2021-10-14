---
description: 순환 신경망은 인공 신경망의 한 종류로, 유닛간의 연결이 순환적 구조를 갖는 특징을 갖고 있다.
---

# Basic Information

RNN은 Recurrent Neural Network의 줄임말로, 시퀀스 데이터를 모델링 하기 위해 등장했다. 이때, **시퀀스**란 무엇인가? 

### 시퀀스 (Sequence)

시퀀스는 파이썬 내에서 가장 기본적인 데이터 구조로서, 데이터를 순서대로 하나씩 나열하여 나타낸 형태이다. 시퀀스의 각 요소에는 인덱스가 지정되어있다. 파이썬은 여러가지 시퀀스 컬렉션을 제공한다. 대표적으로 **List, Tuple, range, string**이 있다.

## RNN (Recurrent Neural Network)

번역기를 살펴보아도, 문장 혹은 단어들 또한 위에서 설명한 시퀀스에 해당한다. 그중에서도 RNN은 이러한 시퀀스를 다루는 가장 기본적인 딥러닝 모델이다. 

![](<../.gitbook/assets/image (22).png>)

지금껏 살펴본 모든 인공신경망은 활성화 함수를 거쳐서 입력층, 은닉층(hidden layer), 출력층 방향으로 값들이 향했다. 이러한 신경망을 Feed Forward Neural Network라고 부른다. \
이에 반해, RNN은 은닉층의 Node(=Cell)에서 활성화 함수를 거친 이후 결과값을 출력층에 보냄과 동시에 다시 은닉층의 Node로도 계산의 입력으로 보내는 특징을 가지고 있다.

### Memory/RNN Cell

RNN의 은닉층의 Node는 그저 결과값을 내보내는 것이 아니라, 이전의 값을 기억하는 메모리 역할을 한다. 그리하여, 이것을 **메모리 셀** 혹은 **RNN **셀이라고도 부른다.

## RNN 작동방식

앞으로는 현재 시점을 변수 t로 표현해자. 그리고, 현재 시점 t에서의 메모리 셀이 갖고있는 값은 이미 과거의 메모리 셀들의 영향을 받은 것을 의미한다. 그리고 다음 시점에 출력층과 다음 시점의 t+1의 자신에게 보내는 값을 **은닉 상태(hidden state)**라고 부른다.

![](<../.gitbook/assets/image (23).png>)

### RNN의 다양한 형태

RNN은 Autoencoder와 다르게 입력과 출력의 길이를 다르게 설정할 수 있다. 그리고, 이 길이에 따라 자연어 처리에서 다양한 용도로 이용될 수 있다. (RNN의 입출력 단위는 여러가지이지만, 일반적으로는 '단어 벡터'가 쓰인다. 

![](<../.gitbook/assets/image (24).png>)

하나의 출력(many-to-one)을 하는 모델은 단어 시퀀스에 대해서 입력 문서가 긍정적인지 부정적인지를 판별하는 감성 분류(sentiment classification), 또는 메일이 정상 메일인지 스팸 메일인지 판별하는 스팸 메일 분류(spam detection)에 사용할 수 있다.

![many-to-one 예시](<../.gitbook/assets/image (25).png>)

many-to-many에서는 입력 문장으로부터 대답 문장을 출력하는 챗봇 혹은 번역기 등에 \
쓰일 수 있다. 밑 그림은 개체명 인식에서 쓰이는 RNN 아키텍쳐를 보여준다.

![many-to-many 예시](<../.gitbook/assets/image (26).png>)

마지막으로 one-to-many는 하나의 이미지를 입력받아 사진의 제목을 출력하는 형식의 캡셔닝(Image Captioning) 작업에 이용될 수 있다. 사진의 제목은 단어의 나열로서 one-to-many에서 many에 속한다. 

### RNN의 수학적 원리

![](<../.gitbook/assets/image (27).png>)

$$
은닉 상태값: h_t \\ 
입력층의 가중치: W_x \\
은닉 상태값의 가중치: W_h \\
$$

$$
은닉층: h_t = f(W_xx_t + W_hh_{t-1} + b) \\
출력층: y_t = f(W_yh_t + b) \\
비선형 활성화 함수: f
$$

RNN의 은닉층 연산은 벡터와 행렬 연산으로 이루어진다. 자연어 처리에서 입력은 대부분 단어 벡터로 간주할 수 있다. 이때, 단어 벡터의 크기는 d, 은닉 상태의 크기는 D라고 하자.

![](<../.gitbook/assets/image (28).png>)

이때, 배치의 크기가 1이고, 단어 벡터와 은닉 상태의 크기 모두 4라고 가정한다면, RNN의 은닉층 연산은 밑과 같은 그림으로 표현된다. (밑 그림에서 활성화 함수로는 _tanh_을 사용했다. ReLU 함수를 사용해도 무관하다)

![](<../.gitbook/assets/image (29).png>)

사진 및 내용 출처: [https://wikidocs.net/22886](https://wikidocs.net/22886)

## +) Deep RNN, Bidirectional RNN
