---
description: Autoencoder는 레이블이 없이 특징을 추출하는 신경망이다.
---

# Basic Information

## Autoencoder 기초 

밑과 같이 이미지에 정답이 Label 되어 있다면, 머신러닝 모델이 매우 효율적으로 학습할 수 있다. 하지만, 데이터셋에 특정 정답이 없다면, 문제가 복잡해진다. 지도학습(Supervised Learning)과 달리 비지도 학습(Unsupervised Learning)은 정답 Label이 따로 없는 데이터를 이용하여 새로운 데이에 대한 결과를 예측하는 기계학습을 일컫는다. 

![](<../.gitbook/assets/image (19).png>)

비지도 학습은 정해진 정답이 없어서 Loss를 구하는 과정이 쉽지 않다. 종속변수 Y 없이 독립변수 X만으로 어떤 의미를 추출할 수 있는 것일까? Autoencoder는 **입력 데이터를 압축시킨 후, 다시 확장하여 결과 데이터를 입력 데이터와 동일하게 만들도록 하는 인공신경망 모델**이다. 즉, X 자체가 일종의 레이블이 되도록 하는 것이다.\
그렇다면, 오토인코더에서 오찻값은 어떻게 구하는 것일까? 입력 X와 정답 Y가 뚜렷하게 정해져있다면, 그 둘 사이의 오찻값(Loss)를 구하면 그만이다. 하지만, Autoencoder는 그렇지 않다.  Autoencoder에서는 Reconstruction Loss(정보손실값)이라는 용어를 사용한다. 말 그대로, Autoencoder가 입력값 X를 얼마나 잘 복원했는가에 대한 지표이다. 

![](<../.gitbook/assets/스크린샷 2021-06-30 오후 8.27.58.png>)

Autoencoder의 중요한 특징은 위 사진과 같이 입력과 출력의 크기는 같으나 중간으로 갈수록 신경망의 차원이 줄어드는 것이다. 위 그림은 숫자 1을 Autoencoder 모델에 입력하고 결과값으로 입력 데이터와 유사한 비슷한 손글씨 1을 출력하는 모습이다. 입력데이터 784개의 뉴런을 500개, 300개, 이후 결과적으로 2개로 압축시킨다. 앞서 CNN에서 언급한 비슷한 원리로 Autoencoder는 이렇게 입력데이터를 압축시킴으로써, 이미지의 대표적인 특징이 압축되도록 한다. 이러한 작은 차원에 압축된 데이터를 Latent variable(잠재 변수)라고 부르며, 짧게는 z라고도 한다. 앞서 설명한 압축 과정을 **Encoder**라고 한다. 그 이후에 압축된 표현을 풀어내어 입력을 다시 복원하는 과정인 뒷부분을 **Decoder**라고 한다. 즉, Autoencoder는 Input Data를 Encoder Network에 통과시켜, 압축된 z를 얻은 이후에, 압축된 z vector에서 다시 Input Data와 크기가 동일한 출력값을 생성한다. 그리고, 입력값과 Decode를 거친 값의 차이를 Loss값으로 설정함으로써, 얼마나 성공적으로 복원이 되었는지 확인한다.  

![](<../.gitbook/assets/image (20).png>)

위와 같은 구조에서 주목해야 할 부분은 바로 중간에 있는 hidden layer의 노드이다. hidden layer의 개수는 input과 ouput의 개수와 다르다. 이는 hidden layer에서는 차원이 축소되었음을 말한다. 그리고, 이 축소된 hidden layer가 input을 잘 복원해낸 ouput을 구현해낸다면, 그만큼 이 Encoding 과정이 의미있음을 말하는 것이다. 그리하여, 사실 Autoencoder는  input을 output으로 그냥 그대로 출력하기 위해서 존재하는 것이 아니라, input을 output으로 변환하기 위해서 표현하는 중간 상태를 잘 만들어내는 것이 목표이다. 
