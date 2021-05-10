---
layout: post
title: "머신러닝 과제4"
---

## 1. 로지스틱 회귀 구현

코드 작성에 있어서 기본 설정이다.


```python
# 파이썬 ≥3.5 필수
import sys
assert sys.version_info >= (3, 5)

# 사이킷런 ≥0.20 필수
import sklearn
assert sklearn.__version__ >= "0.20"

# 공통 모듈 임포트
import numpy as np
import pandas as pd
import os

# 노트북 실행 결과를 동일하게 유지하기 위해
np.random.seed(42)

# 깔끔한 그래프 출력을 위해
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
```

### 데이터 불러오기

사이킷런이 기본으로 제공하는 붓꽃 데이터를 가져온다


```python
from sklearn import datasets
iris = datasets.load_iris()
```


```python
X = iris["data"][:, (2, 3)]  # 꽃잎 길이, 꽃잎 넓이
y = (iris["target"] == 2).astype(np.int)  # 버지니카(Virginica) 품종일 때 1(양성)
```


```python
X_with_bias = np.c_[np.ones([len(X), 1]), X]   #특성에 편향 추가
```


```python
test_ratio = 0.2                                         # 테스트 세트 비율 = 20%
validation_ratio = 0.2                                   # 검증 세트 비율 = 20%
total_size = len(X_with_bias)                            # 전체 데이터셋 크기

test_size = int(total_size * test_ratio)                 # 테스트 세트 크기: 전체의 20%
validation_size = int(total_size * validation_ratio)     # 검증 세트 크기: 전체의 20%
train_size = total_size - test_size - validation_size    # 훈련 세트 크기: 전체의 60%

rnd_indices = np.random.permutation(total_size)
```

예측 타겟과 연산이 잘 이뤄지도록 타겟의 어레이 형태를 미리 바꿔준다.


```python
X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]
y_train = y_train.reshape(len(y_train), 1)

X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]
y_valid = y_valid.reshape(len(y_valid), 1)


X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]
y_test = y_test.reshape(len(y_test), 1)
```

### 로지스틱 회귀 함수 구현

시그모이드 함수를 정의한다.


```python
def sigmoid(x):
    return 1/(1+np.exp(-1 * x))
```

조기종료와 경사 하강법을 이용한 로지스틱 회귀 모델이다. 인자로 훈련세트, 검증세트, 훈련타겟, 검증타겟, 학습률, 에포크 수, 알파(규제정도)를 넣는다. 최종적으로 학습된 세타값이 나오게 된다.


```python
def custom_logistic(x_t, x_v, y_t, y_v, e, iter, alpha):
  epsilon = 1e-7
  best_loss = np.infty
  best_theta = np.array([])    #학습된 세타
  Theta = np.random.randn(x_t.shape[1], 1) * 0.01  #특성 개수 만큼 랜덤한 값의 세타 생성
  count = 0

  for iteration in range(iter):     
      logits = x_t.dot(Theta)   #훈련세트로 모델 훈련
      Y_proba = sigmoid(logits)
      Y_pro_train = np.array([]) 

      for i in Y_proba:     #sigmoid 함수로 예측한 값을 0과 1로 변환
        if i < 0.5:
          Y_pro_train = np.append(Y_pro_train, np.array([0]))
        else:
          Y_pro_train = np.append(Y_pro_train, np.array([1]))
      Y_pro_train = Y_pro_train.reshape(len(Y_pro_train), 1)

      error = Y_pro_train - y_t     #그레디언트 계산
      gradients = 1/len(x_t) * x_t.T.dot(error) + np.r_[np.zeros([1, 1]), alpha * Theta[1:]]  #편향에는 규제를 적용하지 않음
      Theta = Theta - e * gradients   #세타값 수정

      #검증세트로 비용함수 계산
      logits = x_v.dot(Theta)
      Y_proba = sigmoid(logits)
      Y_pro_valid = np.array([]) 
      for i in Y_proba:
        if i < 0.5:
          Y_pro_valid = np.append(Y_pro_valid, np.array([0]))
        else:
          Y_pro_valid = np.append(Y_pro_valid, np.array([1]))
      Y_pro_valid = Y_pro_valid.reshape(len(Y_pro_valid), 1)

      logistic_loss = -1/len(x_v) * (np.sum(y_v * np.log(Y_proba + epsilon) + (1 - y_v) * np.log(1 - Y_proba + epsilon)))  #로지스틱 회귀의 비용함수
      l2_loss = 1/2 * np.sum(np.square(Theta[1:]))   #릿지 규제
      loss = logistic_loss + alpha * l2_loss  #규제를 적용한 손실비용

      if iteration % 500 == 0:    #500에포크마다 손실비용 출력
          print(iteration, loss)

      if loss < best_loss:   #현재 손실비용이 전 에포크의 손실비용보다 좋으면 세타와 손실비용, 몇 에포크인지 저장
          best_loss = loss
          update_theta = Theta
          best_iteration = iteration
      else:
          count = count + 1     #훈련이 너무 빨리 종료되지 않도록 비용함수가 증가하는 시점부터 500에포크 만큼 여유를 더 줌
          if count == 500:
            print(best_iteration, best_loss)
            print(iteration, loss, "조기 종료!")
            best_theta = Theta
            break
  return best_theta
```


```python
best_theta = custom_logistic(X_train, X_valid, y_train, y_valid, 0.008, 5001, 0.1)
```

__output__

    0 0.6937924415670413
    500 0.6781515868268262
    277 0.6761331341641763
    586 0.6784351840733734 조기 종료!
    

정확도를 출력하는 함수이다.


```python
def score(x, y, theta):
  logits = x.dot(theta)              
  Y_proba = sigmoid(logits)
  y_predict = np.array([])
  for i in Y_proba:
    if i < 0.5:
      y_predict = np.append(y_predict, np.array([0]))
    else:
      y_predict = np.append(y_predict, np.array([1]))
  y_predict = y_predict.reshape(len(Y_proba), 1)

  accuracy_score = np.mean(y_predict == y)

  return accuracy_score
```

붓꽃의 길이와 넓이로 버지니카인지 이진분류하는 정확도이다.


```python
score(X_test, y_test, best_theta)
```

__output__

    0.9666666666666667



## 2. 일대다 방식을 적용한 로지스틱 회귀 다중 클래스 분류

### 데이터 불러오기

1번과 같이 붓꽃 데이터를 다시 불러온다.


```python
X = iris["data"][:, (2, 3)]  # 꽃잎 길이, 꽃잎 넓이
y = iris["target"]   #모든 꽃잎 종류
```


```python
X_with_bias = np.c_[np.ones([len(X), 1]), X]    #특성에 편향 추가
```


```python
test_ratio = 0.2                                         # 테스트 세트 비율 = 20%
validation_ratio = 0.2                                   # 검증 세트 비율 = 20%
total_size = len(X_with_bias)                            # 전체 데이터셋 크기

test_size = int(total_size * test_ratio)                 # 테스트 세트 크기: 전체의 20%
validation_size = int(total_size * validation_ratio)     # 검증 세트 크기: 전체의 20%
train_size = total_size - test_size - validation_size    # 훈련 세트 크기: 전체의 60%

rnd_indices = np.random.permutation(total_size)
```


```python
X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]

X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]


X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]
```


```python
y_train[:5]
```

__output__

    array([0, 0, 2, 0, 0])



타겟이 범주형으로 표현 되어 있으므로 원핫인코딩을 수행한다.


```python
def to_one_hot(y):
    n_classes = y.max() + 1                 # 클래스 수
    m = len(y)                              # 샘플 수
    Y_one_hot = np.zeros((m, n_classes))    # (샘플 수, 클래스 수) 0-벡터 생성
    Y_one_hot[np.arange(m), y] = 1          # 샘플 별로 해당 클래스의 값만 1로 변경. (넘파이 인덱싱 활용)
    return Y_one_hot
```

원핫인코딩이 잘 이뤄졌는지 확인한다.


```python
to_one_hot(y_train[:5])
```

__output__

    array([[1., 0., 0.],
           [1., 0., 0.],
           [0., 0., 1.],
           [1., 0., 0.],
           [1., 0., 0.]])
<br/>



```python
Y_train_one_hot = to_one_hot(y_train)
Y_valid_one_hot = to_one_hot(y_valid)
Y_test_one_hot = to_one_hot(y_test)
```


```python
n_inputs = X_train.shape[1]    #세타 생성시 필요한 변수
```

### 다중 클래스 분류

1번에서 구현한 로지스틱 회귀를 클래스 수에 맞게 3번 반복 실행한다. 최종적으로는 각 클래스별로 학습된 세타 값이 나온다.


```python
eta = 0.008
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.1            
best_loss = np.infty   
best_theta = np.array([])
count = 0

for i in range(Y_train_one_hot.shape[1]):   #클래스 수 만큼 반복
    Theta = np.random.randn(n_inputs, 1)
    train_labels = np.array([])    #클래스별 타겟을 순서대로 저장
    valid_labels = np.array([])
    for j in range(len(Y_train_one_hot)):
        train_labels = np.append(train_labels, np.array([Y_train_one_hot[j][i]]))
    for k in range(len(Y_valid_one_hot)):
        valid_labels = np.append(valid_labels, np.array([Y_valid_one_hot[k][i]]))

    train_labels = train_labels.reshape(len(train_labels), 1)
    valid_labels = valid_labels.reshape(len(valid_labels), 1)

    for iteration in range(n_iterations):     
        logits = X_train.dot(Theta)
        Y_proba = sigmoid(logits)
        Y_pro_train = np.array([]) 

        for t in Y_proba:
          if t < 0.5:
            Y_pro_train = np.append(Y_pro_train, np.array([0]))
          else:
            Y_pro_train = np.append(Y_pro_train, np.array([1]))
        Y_pro_train = Y_pro_train.reshape(len(Y_pro_train), 1)

        error = Y_pro_train - train_labels     
        gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, 1]), alpha * Theta[1:]]
        Theta = Theta - eta * gradients

        logits = X_valid.dot(Theta)
        Y_proba = sigmoid(logits)
        Y_pro_valid = np.array([]) 
        for t in Y_proba:
          if t < 0.5:
            Y_pro_valid = np.append(Y_pro_valid, np.array([0]))
          else:
            Y_pro_valid = np.append(Y_pro_valid, np.array([1]))
        Y_pro_valid = Y_pro_valid.reshape(len(Y_pro_valid), 1)

        logistic_loss = -1/len(X_valid) * (np.sum(valid_labels * np.log(Y_proba + epsilon) + (1 - valid_labels) * np.log(1 - Y_proba + epsilon)))
        l2_loss = 1/2 * np.sum(np.square(Theta[1:]))
        loss = logistic_loss + alpha * l2_loss

        if iteration % 500 == 0:
            print(iteration, loss)

        # 에포크마다 최소 손실값 업데이트
        if loss < best_loss:
            best_loss = loss
            update_theta = Theta
            best_iteration = iteration
        else:
            count = count + 1
            if count == 500:                                   
              print(best_iteration, best_loss)        
              print(iteration, loss, "조기 종료!\n")
              if len(best_theta) == 0:      #클래스별 학습된 세타 값을 저장
                best_theta = np.append(best_theta, np.array(update_theta))
              else:
                best_theta = np.c_[best_theta, update_theta]
              best_loss = np.infty
              count = 0
              break
```

__output__

    0 4.0225913870387
    500 0.5296157580910943
    97 0.5214856310184777
    595 0.5324772187115792 조기 종료!
    
    0 2.3441280791734154
    500 0.7087807437143531
    1000 0.6918664544815131
    1081 0.6879379492433929
    1094 0.689597892068635 조기 종료!
    
    0 1.1625869805728124
    500 0.763400022007789
    1000 0.6795820408458436
    1313 0.6782773409511903
    1496 0.6786808310648912 조기 종료!
    
    

이진 분류의 정확도를 측정하는 함수와는 예측 값의 형태만 다르다. 클래스별 예측값 중 가장 큰 값을 1로 바꾸고 나머지는 0으로 바꾼다.


```python
def score_multie(theta, x, y):
    result = np.array([])
    for i in theta:
      logits = x.dot(i)              
      Y_proba = sigmoid(logits)

      if len(result) == 0:
        result = np.append(result, np.array([Y_proba]))
      else:
        result = np.c_[result, Y_proba]
    
    for i in range(len(result)):
      if np.max(result[i]) < 0.5:
        result[i] = np.where(result[i] <= np.max(result[i]), 0, result[i])
      else:
        result[i] = np.where(result[i] < np.max(result[i]), 0, result[i])
        result[i] = np.where(result[i] == np.max(result[i]), 1, result[i])
      
    accuracy_score = np.mean(result == y)  # 정확도 계산
    return accuracy_score
```

이진분류 보다는 정확도가 떨어졌다.


```python
score_multie(best_theta.T, X_test, Y_test_one_hot)
```

__output__

    0.8333333333333334



# 3. 사진 분류하기

### 데이터 가져오기

깃허브에 미리 100*100사이즈로 리사이징한 사진 100장을 저장해 놓았다.

낮(실내)25장, 낮(실외)25장, 밤(실내)25장, 밤(실외)25장으로 구성되어있다.

컬러 이미지는 R,G,B별로 각각 100*100사이즈의 특성을 가지는 3차원 어레이로 구성 되어있다.

각 특성은 0~255사이의 값을 가지기 때문에 특성 값을 줄이기 위해 255로 나누었다.

그리고 특성의 순서는 상관없기 때문에 다루기 쉽게 1차원으로 바꿔준다.


```python
import urllib.request
import cv2

img_set = np.zeros((1, 30000))

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/min020/dataset/main/day_and_night/"
datapath = os.path.join("dataset", "day_indoor", "")   #사진이 저장될 폴더명
os.makedirs(datapath, exist_ok=True)
for i in range(1, 26):     
    filename = str(i) + ".jpg"
    url = DOWNLOAD_ROOT + "day/day_indoor/" + filename
    urllib.request.urlretrieve(url, datapath + filename)
    img = np.ravel(cv2.imread(datapath + str(i) + ".jpg", cv2.IMREAD_COLOR)) / 255.0   #사진을 불러온후 ravel을 이용해 1차원으로 바꾼다.
    img_set = np.concatenate((img_set, img.reshape(1, len(img))))    #1차원으로 바꾼 특성을 순서대로 저장한다.


datapath = os.path.join("dataset", "day_outdoor", "")
os.makedirs(datapath, exist_ok=True)
for i in range(26, 51):
    filename = str(i) + ".jpg"
    url = DOWNLOAD_ROOT + "day/day_outdoor/" + filename
    urllib.request.urlretrieve(url, datapath + filename)
    img = np.ravel(cv2.imread(datapath + str(i) + ".jpg", cv2.IMREAD_COLOR)) / 255.0
    img_set = np.concatenate((img_set, img.reshape(1, len(img))))

datapath = os.path.join("dataset", "night_indoor", "")
os.makedirs(datapath, exist_ok=True)
for i in range(51, 76):
    filename = str(i) + ".jpg"
    url = DOWNLOAD_ROOT + "night/night_indoor/" + filename
    urllib.request.urlretrieve(url, datapath + filename)
    img = np.ravel(cv2.imread(datapath + str(i) + ".jpg", cv2.IMREAD_COLOR)) / 255.0
    img_set = np.concatenate((img_set, img.reshape(1, len(img))))

datapath = os.path.join("dataset", "night_outdoor", "")
os.makedirs(datapath, exist_ok=True)
for i in range(76, 101):
    filename = str(i) + ".jpg"
    url = DOWNLOAD_ROOT + "night/night_outdoor/" + filename
    urllib.request.urlretrieve(url, datapath + filename)
    img = np.ravel(cv2.imread(datapath + str(i) + ".jpg", cv2.IMREAD_COLOR)) / 255.0
    img_set = np.concatenate((img_set, img.reshape(1, len(img))))
img_set = np.delete(img_set, 0, 0)
```

낮과 밤, 실내와 실외로 나누어 라벨링을 한다. 낮과 실내를 1로 설정했다.


```python
label_time = np.concatenate((np.tile(np.array([1]), (25)), np.tile(np.array([1]), (25)), np.tile(np.array([0]), (25)), np.tile(np.array([0]), (25))))
label_place = np.concatenate((np.tile(np.array([1]), (25)), np.tile(np.array([0]), (25)), np.tile(np.array([1]), (25)), np.tile(np.array([0]), (25))))
```


```python
test_ratio = 0.2                                         # 테스트 세트 비율 = 20%
validation_ratio = 0.2                                   # 검증 세트 비율 = 20%
total_size = len(img_set)                            # 전체 데이터셋 크기

test_size = int(total_size * test_ratio)                 # 테스트 세트 크기: 전체의 20%
validation_size = int(total_size * validation_ratio)     # 검증 세트 크기: 전체의 20%
train_size = total_size - test_size - validation_size    # 훈련 세트 크기: 전체의 60%

rnd_indices = np.random.permutation(total_size)

img_train = img_set[rnd_indices[:train_size]]
time_train = label_time[rnd_indices[:train_size]]
place_train = label_place[rnd_indices[:train_size]]
time_train = time_train.reshape(len(time_train), 1)
place_train = place_train.reshape(len(place_train), 1)

img_valid = img_set[rnd_indices[train_size:-test_size]]
time_valid = label_time[rnd_indices[train_size:-test_size]]
place_valid = label_place[rnd_indices[train_size:-test_size]]
time_valid = time_valid.reshape(len(time_valid), 1)
place_valid = place_valid.reshape(len(place_valid), 1)

img_test = img_set[rnd_indices[-test_size:]]
time_test = label_time[rnd_indices[-test_size:]]
place_test = label_place[rnd_indices[-test_size:]]
time_test = time_test.reshape(len(time_test), 1)
place_test = place_test.reshape(len(place_test), 1)
```

### A. 낮과 밤으로 분류하는 로지스틱 회귀 모델

1번에서 구현한 모델과 같지만 사진특성에는 편향을 추가해 주지 않았기 때문에 그레디언트 부분을 수정해서 다시 선언해준다.


```python
def custom_logistic(x_t, x_v, y_t, y_v, e, iter, alpha):
  epsilon = 1e-7
  best_loss = np.infty
  best_theta = np.array([])
  Theta = np.random.randn(x_t.shape[1], 1) * 0.01
  count = 0

  for iteration in range(iter):     
      logits = x_t.dot(Theta)
      Y_proba = sigmoid(logits)
      Y_pro_train = np.array([]) 

      for i in Y_proba:
        if i < 0.5:
          Y_pro_train = np.append(Y_pro_train, np.array([0]))
        else:
          Y_pro_train = np.append(Y_pro_train, np.array([1]))
      Y_pro_train = Y_pro_train.reshape(len(Y_pro_train), 1)

      error = Y_pro_train - y_t    
      gradients = 1/len(x_t) * x_t.T.dot(error) + alpha * Theta
      Theta = Theta - e * gradients  

      logits = x_v.dot(Theta)
      Y_proba = sigmoid(logits)
      Y_pro_valid = np.array([]) 
      for i in Y_proba:
        if i < 0.5:
          Y_pro_valid = np.append(Y_pro_valid, np.array([0]))
        else:
          Y_pro_valid = np.append(Y_pro_valid, np.array([1]))
      Y_pro_valid = Y_pro_valid.reshape(len(Y_pro_valid), 1)

      logistic_loss = -1/len(x_v) * (np.sum(y_v * np.log(Y_proba + epsilon) + (1 - y_v) * np.log(1 - Y_proba + epsilon)))
      l2_loss = 1/2 * np.sum(np.square(Theta))  
      loss = logistic_loss + alpha * l2_loss

      if iteration % 500 == 0:    
          print(iteration, loss)

      if loss < best_loss:   
          best_loss = loss
          update_theta = Theta
          best_iteration = iteration
      else:
          count = count + 1     
          if count == 500:
            print(best_iteration, best_loss)
            print(iteration, loss, "조기 종료!")
            best_theta = update_theta
            break
  return best_theta
```


```python
best_theta_time = custom_logistic(img_train, img_valid, time_train, time_valid, 0.005, 5001, 0.1)
```

__output__

    0 3.516324506958805
    500 0.741685141452624
    46 0.7384023240072761
    506 0.740791760665329 조기 종료!
<br/>    


```python
score(img_test, time_test, best_theta_time)
```

__output__

    0.8



### B. 낮과 밤, 실내와 실외로 분류하는 다중 레이블 분류

낮과 밤을 분류하도록 훈련된 세타값은 A번에서 가져온다.

실내와 실외를 분류하는 정확도이다.


```python
best_theta_place = custom_logistic(img_train, img_valid, place_train, place_valid, 0.005, 5001, 0.1)
```

__output__

    0 1.77084082891388
    500 1.1598873879153397
    7 0.9209910467744262
    502 1.1590067416079748 조기 종료!
<br/>


```python
score(img_test, place_test, best_theta_place)
```

__output__

    0.7



낮과 밤, 실내와 실외를 모두 분류한 다중 레이블 분류의 정확도이다.


```python
y = np.c_[time_test, place_test]

logits = img_test.dot(best_theta_time)              
Y_proba = sigmoid(logits)
y_predict = np.array([])
result = np.array([])
for i in Y_proba:
  if i < 0.5:
    y_predict = np.append(y_predict, np.array([0]))
  else:
    y_predict = np.append(y_predict, np.array([1]))

result = np.append(result, np.array([y_predict]))

logits = img_test.dot(best_theta_place)              
Y_proba = sigmoid(logits)
y_predict = np.array([])
for i in Y_proba:
  if i < 0.5:
    y_predict = np.append(y_predict, np.array([0]))
  else:
    y_predict = np.append(y_predict, np.array([1]))

result = np.c_[result, y_predict]
    

accuracy_score = np.mean(result == y)
accuracy_score
```

__output__

    0.75



### C. 사이킷런에서 제공하는 LogisticRegression 모델과 성능 비교

직접 구현한 로지스틱 회귀와 사이킷런에서 제공하는 로지스틱 회귀와 성능 비교를 해본다.

낮과 밤을 분류한 정확도이다.


```python
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver="lbfgs", random_state=42)
log_reg.fit(img_train, time_train)
```

__output__

    /usr/local/lib/python3.7/dist-packages/sklearn/utils/validation.py:760: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=42, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)
<br/>



```python
prediction = log_reg.predict(img_test)
```


```python
np.mean(prediction.reshape(len(prediction), 1) == time_test)
```

__output__

    0.9



실내와 실외를 분류한 정확도이다.


```python
from sklearn.linear_model import LogisticRegression
log_reg_place = LogisticRegression(solver="lbfgs", random_state=42)
log_reg_place.fit(img_train, place_train)
```

__output__

    /usr/local/lib/python3.7/dist-packages/sklearn/utils/validation.py:760: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=42, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)
<br/>



```python
prediction = log_reg_place.predict(img_test)
```


```python
np.mean(prediction.reshape(len(prediction), 1) == place_test)
```

__output__

    0.75



사이킷런이 제공하는 로지스틱 회귀 모델이 더 좋은 성능을 보이긴 하지만 얼추 비슷하긴 하다. 직접 구현한 로지스틱 회귀 모델에서 조기종료 부분을 좀 더 손보면 성능이 더 높아질 것으로 예상된다.
