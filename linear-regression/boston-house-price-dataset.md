---
description: 'sklearn에서 제공하는 데이터셋을 활용하여, 보스턴 지역의 주택 가격을 예측하는 회귀 모델을 제작하는 페이지입니다.'
---

# Boston house price Dataset

#### 데이터 로딩 및 탐색 

pandas는 데이터 분석을 위한 파이썬에서 제공하는 라이브러리입니다. 그 이외에도, 함수를 만들고 그래프를 그릴 수 있게 해주는 것처럼 데이터 시각화를 맡고 있는 matplotlib와 seaborn 라이브러리가 존재합니다. 

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn import datasets
house = datasets.load_boston()
house.keys()
```

sklearn에서 우리가 사용하고자 하는 dataset을 불러들이고, house라는 변수에 이를 저장합니다. 이 dataset은 파이썬에서 딕셔너리의 형태로 제공됩니다.   
`house.keys()`를 통해서, 파이썬 딕셔너리의 키만 추출합니다. 

`dict_keys(['data', 'target', 'feature _names', 'DESCR', 'filename'])`

```python
data = pd.DataFrame(house['data'], columns=house['feature_names'])
target = pd.DataFrame(house['data'], columns=['Target'])
```

그리고 데이터의 가시성을 높이고,  처리하기 용이하기 위해 데이터셋을 pandas 라이브러리를 이용하여 데이터프레임의 형태로 변환합니다.   
data는 13개의 속성에 대하여 506개 주택의 정보를 담고 있다. 그래서, 2차원 구조로 `(506, 13)`를 이루고 있습다.   
\(밑 Table에서 13개의 속성에 관한 자세한 설명을 다루고 있습니다\) 

| 속 | 설명 |
| :---: | :--- |
| CRIM | 해당 지역의 1인당 범죄 발생 |
| ZN | 면적 25,000평방피트를 넘는 주택용 토지의 비율 |
| INDUS | 해당 지역의 비소매 상업 지역 비율 |
| CHAS | 해당 부지의 찰스강 인접 여부 \(인접한 경우 1, 그렇지 않은 경우 0\) |
| NOX | 일산화질소 농도 |
| RM | \(거주 목적의\) 방의 개 |
| AGE | 1940년 이전에 건축된 자가 주택의 비율 |
| DIS | 보스턴의 5대 고용 지역까지의 거리 |
| RAD | 고속도로 접근성 |
| TAX | 재산세 |
| PTRATIO | 교사-학생 비율 |
| B | 흑인 거주 비율 |
| LSTAT | 저소득층 비율 |
| MEDV | 소유주 거주 주택의 주택 가격 \(중간값\) |



```python
df = pd.concat([data, target], axis=1)
```

pd.concat 함수를 활용하여, data와 target 데이터프레임을 결합하여 줍니다. \(데이터의 속성, 형태가 동일한 데이터를 위아래로 합칠 때에는 axis=0, 좌우로 합칠 때에는 axis=1을 이용합니다\)

![df.head\( \)](../.gitbook/assets/image%20%2827%29.png)

![df.info\( \)](../.gitbook/assets/image%20%2828%29.png)

`df.info()`를 이용하여, 데이터프레임의 전체적인 정보를 확인해 봅시다. Non-Null Count를 통하여, 이 데이터프레임의 결측값\(missing value\)이 하나도 없음을 알 수 있습니다. 또한, 자료형은 64bit float임을 알 수 있습니다. 

```python
df_corr = df.corr()

plt.figure(figsize = (10,10))
sns.heatmap(df_corr, annot = True)
plt.show()
```

corr 함수를 이용하여, 숫자 데이터를 갖는 변수 간의 상관계수를 계산합니다. `annot=True`를 통하여, heatmap 위에서도 상관 계수 값을 표시해주도록 합니다. 

![](../.gitbook/assets/image%20%2849%29.png)

```python
corr_order = df.corr().loc['CRIM':'LSTAT', 'Target'].abs().sort_values(ascending=False)
corr_order
```

Target을 목표로 하여 상관계수를 CRIM부터 LSTAT까지 계산합니다. 그리고, 이 상관계수에 절댓값을 적용한 이후에 내림차으로 나열한 것을 corr\_order에 저장합니다. 이를 나열하면, 밑과 같은 결과를 얻을 수 있습니다.

![](../.gitbook/assets/image%20%2856%29.png)

그렇다면, Target과 상관계수가 높은 4가지, LSTAT, RM, PTRATIO, INDUS만 따로 추출하여 새로운 데이터프레임에 저장하자고 합니다.  \(위 4가지 속성은 차례로, 저소득층 비율, 방의 개수, 교사-학생 비율, 해당 지역의 비소매 상업 지역 비율입니다\) 

```python
plot_cols = ['Target', 'LSTAT', 'RM', 'PTRATIO', 'INDUS']
plot_df = df.loc[ : plot_cols]
plot_df.head()
```

![plot\_df.head\( \)](../.gitbook/assets/image%20%2851%29.png)

Target과 함께 4가지 속성에 관한 데이터를 원래 존재했던 데이터프레임 df에서 가져와 plot\_df에 새롭게 저장해줍니다. 

```python
plt.figure(figsize=(10,10))
for idx, col in enumerate(plot_cols[1:]):
    ax1 = plt.subplot(2, 2, idx+1)
    sns.regplot(x=col, y=plot_cols[0], data=plot_df, ax=ax1)
plt.show()
```

`corr( )` 함수를 이용하여, 가장 상관계수가 높은 속성 4가지를 추렸지만, seaborn의 regplot 함수 이용하여, 시각적으로 선형관계를 파악해봅시다. x축은 plot\_cols의 column들인 4가지 속성들로 설정해주고, y축은 plot\_cols\[0\]에 위치한 Target으로 설정합니다. \(위 코드블럭에서 나타내는 `plt.subplot(2, 2, idx+1)` 한 번에 여러개의 그래프를 그릴 수 있게 합니다. 저 함수 없이는 단 하나의 좌표평면위에 여러 개의 그래프가 겹쳐서 보이게 됩니다\)

![](../.gitbook/assets/image%20%2866%29.png)

육안으로 보았을 때, LSTAT와 RM은 Target과의 상관관계가 뚜렷해보입니다. 데이터 전처리 이전에 Target\(주택 가격 데이터\)의 분포를 displot 함수를 이용하여 그려봅시다. 밑 결과 사진은 Target 데이터의 분포를 오직 히스토그램으로 출력한 사진만 첨부했습니다. `kind='kde'`로 한다면, KDE 밀도함수를 볼 수 있습니다.

```python
sns.displot(x='Target', kind='hist', data=df)
plt.show()
sns.displot(x='Target', kind='kde', data=df)
plt.show()
```

![Target data &#xBD84;&#xD3EC; \(&#xD788;&#xC2A4;&#xD1A0;&#xADF8;&#xB7A8;\)](../.gitbook/assets/image%20%2869%29.png)

모델을 완성하기에 앞서, 우리의 현재 Feature들은 다른 단위들을 가지고 있습니다. 이런 데이터의 크기에 따라 우리가 제작하고자 하는 모델에 상대적 영향력의 차이가 있을 수 있습니다. 이러한 차이를 제거해주기 위해서는 Feature\(열\)의 크기를 비슷하게 맞춰주는 작업이 필요합니다\(= Featrue Scaling이 필요한 이유에 관련된 사이트: [https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-8-Feature-Scaling-Feature-Selection](https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-8-Feature-Scaling-Feature-Selection)\). 

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df_scaled = df.iloc[:, :-1]
scaler.fit(df_scaled)
df_scaled = scaler.transform(df_scaled)

df.iloc[:, :-1] = df_scaled[:, :]
df.head()
```

사이킷런 MinMaxScaler를 이용하여 정규화\(Normalization\)을 거칠 것입니다. Target 열을 제외할 수  도록, 마지막 열을 나타내는 -1을  포함하지 않은채로 iloc 인덱서로 목표 데이터를 `df_scaled`로 추출합니다. \(iloc와 loc은 다릅니다. iloc는 인덱스 번호로 데이터를 지정할 때 쓰이며, loc은 데이터 행이나 열의 이름을 통하여 데이터를 지정합니다\)   
이 데이터를 MinMaxScaler 객체에 fit 메서드를 이용하여 각 Feature의 데이터를 0과 1 사이로 변환합니다. transform 함수를 이용하면, 변환식을 실제로 이용하여 데이터를 정규화합니다. 

![Feature scaling](../.gitbook/assets/image%20%2835%29.png)

이제 컴퓨터에게 학습할 데이터와 테스트할 데이터를 나누어봅시다. 학습 데이터 X\_data로는 선형관계가 뚜렷한 LSTAT와 RM을 택합니다. 506개의 샘플 중에서 20%를 모델 평가에 사용합시다. train의 test\_size를 0.2로 입력하면 됩니다. 

```python
from sklearn.model_selection import train_test_split
X_data = df.loc[ :, ['LSTAT', 'RM'] ]
y_data = df.loc[ :, 'Target']
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
                                                      test_size = 0.2)
```

이로써, 506개의 주택 데이터를 404개의 학습 데이터와 102개의 테스트 데이터로 분할합니다.   일반적으로, 테스트 데이터는 10~30% 수준으로 설정합니다. 검증 데이터의 비중이 낮으면 과적합\(overfitting\)이 일어날 수 있다. 반대로 검증 데이터가 너무 많으면 모델 학습이 잘 이루어지지 않을 수 있다. 

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

print("회귀계수, gradient:" np.round(lr.coef_, 2))
print("상수항, intercept:" np.round(lr.intercept_, 2))
```

이제 앞에서 계속 언급해왔던 선형회귀가 등장할 차례입니다. LinearRegression 클래스  객체를 생성한 이후에, fit 메서드에 학습 데이터를 입력하면, 선형 회귀식을 찾아줍니다.  선형 회귀 모델의 coef\_ 속성으로 각 열에 대한 기울기 값\(W\)을 얻고, intercept\_ 속성에서 상수항\(절편, b\) 값을 얻습니다. 그러면 이와 같은 결과를 얻을 수 있습니다. \(np.round를 이용하여 기울기와 절편 소수점 둘째 자리까지 표시하도록 하였습니다\)  
`회귀계수, gradient: [-23.34 25.33]  
상수항, intercept: 16.41`

```python
y_test_pred = lr.predict(X_test)

plt.figure(figsize = (10, 5))
plt.scatter(X_test['LSTAT'], y_test, label='y_test')
plt.scatter(X_test['LSTAT'], y_test_pred, c='r', label='y_pred')
plt.legend(loc = 'best')
plt.show()
```

predict 함수에 X\_test를 입력하면, 우리가 제작한 선형회귀 모델에 대한 예측값을 얻을 수 있습니다. 예측값을 y\_pred, 실제값을 y\_test에 저장하고, 선점도를 그려 비교합니다. matplotlib의 scatter 함수를 이용합니다. c='r'을 이용하여 예측값은 빨간색으로 표시했습니다. \(loc = 'best'는 범례를 표시하는 곳을 표시하는 옵션입니다\)

![y\_test&#xC640; y\_pred&#xC758; &#xBD84;&#xD3EC;](../.gitbook/assets/image%20%2814%29.png)

하지만, 예측한 값이 실제값과 완전히 동일하다고는 보기 힘듭니다. 실체 예측값과 정답의 오차\(loss\)는 상당한 편으로 보입다. 이러한 loss를 확인해보는 방법으로는 앞선 페이지에서 언급한 MSE, MAE등이 있습다. \(또한, MSE의 제곱근인 RMSE도 있습다\)

