---
description: 번역기에서 대표적으로 사용하는 Sequence-to-Sequence에 대한 기본적인 구조와 원리를 살펴보는 페이지다.
---

# Basic Information

Sequence-to-Sequence는 입력 시퀀스로부터 출력 시퀀스를 출력하며, 이를 각각 질문과 대답으로 설정한다면, Chatbot\(챗봇\)이나 기계번역 등서 다양하게 이용된다. 시퀸스-투-시퀸스\(Sequence-to-Sequence\)는 흔히 줄여서 seq2seq라고 쓴다. 이 앞에서 설명할 것은 대부분 RNN에서 언급한 내용이지만, 어떻게 이를 조립하느냐에 따라  seq2seq가 만들어진다.

## seq2seq의 기본적인 구조

![](../.gitbook/assets/image%20%2861%29.png)

위 그림은 seq2seq의 기본적인 구조로, 영어 문장을 받아 프랑스어 문장을 출력하는 모습이다. seq2seq는 크게 **'인코더'**와 **'디코더'** 로 이루어진다. 인코더는 입력한 모든 단어를 순차적으로 압축해서  하나의 벡터로 만든다. 이 압축된 벡터를 **Context vector**라고 한다. 그리고 디코더는 Context vector를 받아서 번역된 단어를 한 개씩 순차적으로 출력한다.

![](../.gitbook/assets/image%20%2867%29.png)

조금 더 깊게 살펴보자. 사실, 인코더와 디코더는 또 그 안에서 여러개의 RNN으로 이루어져 있다. 실제로는 성능 때문에 RNN 셀이 아닌 LSTM 혹은 GRU 셀로 구성된다. 

### seq2seq가 이러한 구조를 갖게 된 배경 

#### `English: I am not the black cat. French: Je ne suis pas chat noir.`

**위 문장에서 눈 여겨 봐야할 점은 2가지이다.  
1\) black cat이 프랑스어로는 순서가 chat\(=cat\) noir\(=black\)으로 뒤바뀌었다.  
2\) 영어에서 not을 프랑스어에서는 ne/pas로 2글자를 이용한다.**

이러한 문제들 word by word로 직역한다고 했을 때, 번역기가 절대로 해결해 낼 수 없는 문제이다. 그리하여, 단순히 한 단어가 가지는 의미를 학습하여 직역하는 것 아니라, 한 문장이 전체적으로 내포하는 의미를 담아낸 이후에 이를 풀어내는 과정이 필요하다고 판단한 것이다.

그리하여, seq2seq는 문장의 의미를 함축하여 context vector로 표현하는 Encoder와 context vector의 의미를 다시 풀어내는 Decoder를 가진 구조를 갖게 되었다. 

### Encoder/Decoder & Context Vector \(훈련 과정\) 

입력 문장은 단어 토큰화에 의해 단어 단위로 쪼개지고, 각각의 토큰은 RNN 셀의 각 시점이 된다. 그리고, 인코더 RNN 셀의 마지막 시점의 hidden state\(=Context vector\)를 디코더 RNN 셀로 넘겨준다. 즉, Context vector는 디코더 RNN 셀의 첫번째 hidden state로 사용되는 것이다.

디코더는 시작할 때 &lt;sos&gt;라는 심볼, 끝날 때는 문장의 끝을 나타내는 &lt;eos&gt;라는 심볼을 갖는다. 디코더가 &lt;sos&gt;를 받으면 처음에 올 높은 확률의 단어를 예측한다. 그리고, 첫번째로 올 것이라 예측한 단어\(**je**\)는 다음 디코더 RNN 셀에 들어간다. 그리고 다음에 두번째로 올 단어\(**suis**\)는 그 다음 디코더 RNN 셀에 들어가고, 이러한 행위가 &lt;eos&gt;가 다음 단어로 나올 때까지 반복된다. 

훈련 과정에서는 디코더가 인코더에게 **'컨텍스트 벡터'**및 **'&lt;sos&gt; I am a student'**를 받았을 때, **'je suis étudiant &lt;eos&gt;'**가 나와야 한다는 정답을 알려주는 방식으로 훈련을 진행한다. 하지만 테스트 과정은 컨텍스트 벡터와 &lt;sos&gt;의 입력만으로 다음 시점의 단어를 예측해야 한다. 

#### **이후로 설명하는 seq2seq는 테스트 과정에서의 이야기다.**

### Word Embedding

컴퓨터가 단어보다는 숫자 및 벡터를 다루는 것에 더 능하다는 것은 모두가 아는 사실이다. 그리고 이를 위해 자연어 처리에서 텍스트를 벡터로 바꾸는 것을 **'워드 임베딩'**이라고 한다. 그리고 워드 임베딩을 통해 얻은 벡터를 **'임베딩 벡터'**라고 부른다.

![](../.gitbook/assets/image%20%2846%29.png)

결국, seq2seq에서도 LSTM 셀에 들어가기 전에 워드 임베딩을 거친다. 모든 단어들은 벡터로 표현된 임베딩 벡터가 된다. 보통 임베딩 벡터는 수백개의 차원까지 가질 수 있다.

### RNN 기본 구조 복습

![&#xAE30;&#xBCF8;&#xC801;&#xC778; RNN &#xAD6C;&#xC870;](../.gitbook/assets/image%20%2857%29.png)

위 그림같은 RNN 구조에서 time step이 t라고 했을 때, RNN 셀은 **1\) t-1의 hidden state**와 **2\) t에서의 입력 벡터**를 받는다. 그리고, 또 다른 입력층 혹은 출력층으로 t에서의 hidden state를 보낼 수 있습니다. 그 이후에 t+1의 RNN 셀의 입력으로 hidden state를 보낸다.

seq2seq에서 Context vector는 결국 인코더의 마지막 RNN 셀의 hidden stat라고 봐도 무방하다. 그리고 Context vector에는 입력 문장의 모든 단어 토큰들의 정보를 담고 있는 것이다.

### Decoder

![seq2seq&#xC758; Decoder](../.gitbook/assets/image%20%2848%29.png)

디코더는 첫번째 hidden state로 Context vector와 입력값인 &lt;sos&gt;를 받는다. 이 값들을 토대로, 다음에 올 단어를 예측한다. 예측한 단어는 다시 t+1 RNN 셀의 입력값이 된다. 

RNN 셀로부터 출력된 값은 dense layer과 함수를 거쳐서, 하나의 단어를 골라 예측해야 한다. 이때, **softmax 함수**를 사용할 수 있다. softmax 함수는 RNN 셀의 출력 벡터를 받아 각 단어별 확률값을 반환한다. 그리고, 디코더는 출력할 단어를 결정하게 된다. 



사진 및 내용 출처:   
[https://wikidocs.net/24996](https://wikidocs.net/24996)  
[https://pytorch.org/tutorials/intermediate/seq2seq\_translation\_tutorial.html](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

