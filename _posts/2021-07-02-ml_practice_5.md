---
layout: post
title: "랜덤포레스트&스태킹"
excerpt: "랜덤포레스트와 스태킹 직접 구현해보기"
category: ml_practice
date: 2021-07-02
last_modified_at: 2021-07-02
---

```python
# 파이썬 ≥3.5 필수
import sys
assert sys.version_info >= (3, 5)

# 사이킷런 ≥0.20 필수
import sklearn
assert sklearn.__version__ >= "0.20"

# 공통 모듈 임포트
import numpy as np
import os

# 노트북 실행 결과를 동일하게 유지하기 위해
np.random.seed(42)
```

## 1. 랜덤 포레스트 직접 구현하기

### 결정트리 모델 훈련

데이터는 moons데이터셋을 사용한다. 샘플수는 10000개에 잡음을 추가해주고 `train_test_split`를 사용해서 훈련세트와 테스트세트를 8:2비율로 나눠준다.


```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

랜덤 포레스트는 배깅을 사용하여 결정트리 모델로 앙상블 학습을 하는 모델이다. 그렇기 때문에 먼저 랜덤 포레스트에서 사용할 결정트리 모델의 최적의 파라미터를 찾기 위해 그리드 탐색을 실시한다.
+ `max_leaf_nodes` : 리프 노드의 최대 개수
+ `min_samples_split` : 노드 분할에 필요한 최소 샘플 개수

두 가지 파라미터의 최적 값을 찾는다.

`verbose`는 그리드 탐색의 결과 메시지 출력에 관련되어 있다. 0, 1, 2 각각 출력안함, 간단하게, 파라미터별로 출력이다.

다음 그리드 탐색은 98*3 = 294번씩 수행하고 3번의 교차검증을 실시하기 때문에 총 882번의 학습을 실시한다.


```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)

grid_search_cv.fit(X_train, y_train)
```

__output__

    Fitting 3 folds for each of 294 candidates, totalling 882 fits
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 882 out of 882 | elapsed:    8.9s finished
    




    GridSearchCV(cv=3, error_score=nan,
                 estimator=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None,
                                                  criterion='gini', max_depth=None,
                                                  max_features=None,
                                                  max_leaf_nodes=None,
                                                  min_impurity_decrease=0.0,
                                                  min_impurity_split=None,
                                                  min_samples_leaf=1,
                                                  min_samples_split=2,
                                                  min_weight_fraction_leaf=0.0,
                                                  presort='deprecated',
                                                  random_state=42,
                                                  splitter='best'),
                 iid='deprecated', n_jobs=None,
                 param_grid={'max_leaf_nodes': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                13, 14, 15, 16, 17, 18, 19, 20, 21,
                                                22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                31, ...],
                             'min_samples_split': [2, 3, 4]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=1)

<br/>

그리드 탐색으로 찾은 최적의 파라미터를 확인해 보면 `max_leaf_nodes` = 17, `min_samples_split` = 2 이 조합이 최적의 조합이란 것을 알 수 있다.


```python
grid_search_cv.best_estimator_
```


__output__

    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=None, max_features=None, max_leaf_nodes=17,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=42, splitter='best')

<br/>

### 랜덤 포레스트 구현하기

몇 개의 샘플로 몇 개의 모델을 학습 시킬지 정해야 한다. 우선 한 모델당 100개의 샘플을 가지고 1000개의 모델을 학습 시킬 것이다. 그러기 위해서 `ShuffleSplit`을 이용해 훈련세트에서 100개를 무작위로 선택해 1000개의 작은 훈련세트로 만들어 준다.
+ `n_splits` : 나눠지는 세트 수
+ `train_size` : 훈련세트(`X_train`)에서 선택할 샘플 수


```python
from sklearn.model_selection import ShuffleSplit

n_trees = 1000
n_instances = 100

mini_sets = []

rs = ShuffleSplit(n_splits=n_trees, train_size=n_instances, random_state=42)
for mini_train_index, _ in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))
```
<br/>

이제 나눠진 작은 훈련세트들로 학습할 차례이다.

`clone`은 해당 모델을 동일한 파라미터로 복사하는 역할을 한다. 즉 `forest`에 1000개의 모델이 들어가 있는 형태가 된다. 각 모델들을 학습시킨 후 `X_test`세트에 대해 예측을 한다.


```python
from sklearn.base import clone
from sklearn.metrics import accuracy_score

forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)]

Y_pred = np.empty([n_trees, len(X_test)])

for (tree_index, tree), (X_mini_train, y_mini_train) in zip(enumerate(forest), mini_sets):
    tree.fit(X_mini_train, y_mini_train)
    Y_pred[tree_index] = tree.predict(X_test)
```
<br/>

`Y_pred`에 `X_test`세트의 샘플 개수인 2000개를 예측한 것이 1000개 들어가 있다. 이제 각 샘플마다 1000개의 모델들이 예측한 값 중 최빈값을 찾아야 하는데 `mode`를 사용하면 찾을 수 있다.
+ `y_pred_majority_votes` : 각 샘플에 대한 최빈값
+ `n_votes` : 최빈값의 빈도 수


```python
from scipy.stats import mode

y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)
```

<br/>

5개의 샘플만 살펴보면 1, 2번 샘플은 1로 많이 예측했고 3, 4, 5번 샘플은 0으로 많이 예측한 것을 알 수 있다.


```python
y_pred_majority_votes[0][:5]
```


__output__

    array([1., 1., 0., 0., 0.])

<br/>


빈도수를 살펴보면 1000개의 모델중 몇개의 모델이 해당 값으로 예측했는지 알 수 있다.


```python
n_votes[0][:5]
```


__output__

    array([951, 912, 963, 951, 738])

<br/>

### 정확도 비교

사이킷런에서 제공하는 랜덤 포레스트 모델과 직접 구현한 모델을 비교해 볼 것이다. 파라미터는 앞서 그리드 탐색에서 찾았던 파라미터를 같이 사용해 준다.


```python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=1000, max_leaf_nodes=17, min_samples_split=2, max_samples=100, random_state=42)
rnd_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)
accuracy_score(y_test, y_pred_rf)
```


__output__

    0.8725

<br/>

사이킷런의 랜덤 포레스트 모델의 정확도는 약 87퍼센트가 나왔다. 이제 직접 구현한 랜덤 포레스트 모델의 정확도를 측정해보자.


```python
accuracy_score(y_test, y_pred_majority_votes.T)
```


__output__

    0.872


<br/>

## 2. 스태킹 모델 직접 구현하기

### 데이터 준비

스태킹 모델 구현에는 0~9까지 숫자데이터가 들어가 있는 MNIST세트를 사용한다. 타겟이 문자형으로 저장되어 있기 때문에 정수형으로 변환해 준다.


```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(int)
```

<br/>

훈련, 검증, 테스트 3개의 세트로 나눠준다. 각각 50000, 10000, 10000개의 샘플을 가진다.


```python
X_train_val, X_test, y_train_val, y_test = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=10000, random_state=42)
```

<br/>

### 예측기와 블렌더 학습

예측기 모델로 랜덤 포레스트, 엑스트라 트리, 서포트벡터머신을 사용할 것이다. 각 예측기 별로 훈련세트에 대해 학습을 시킨다.


```python
from sklearn.ensemble import  ExtraTreesClassifier
from sklearn.svm import LinearSVC

random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
svm_clf = LinearSVC(max_iter=100, tol=20, random_state=42)

estimators = [random_forest_clf, extra_trees_clf, svm_clf]
for estimator in estimators:
    estimator.fit(X_train, y_train)
```

<br/>

훈련세트에 대해 훈련이 완료된 예측기들로 검증세트에 대해 예측을 실시한다. 그리고 나온 예측값으로 블렌더를 학습시킨다.


```python
from sklearn.linear_model import LogisticRegression

X_val_predictions = np.empty((len(X_val), len(estimators)))

for index, estimator in enumerate(estimators):
    X_val_predictions[:, index] = estimator.predict(X_val)

logistic_blender = LogisticRegression(max_iter=1500)
logistic_blender.fit(X_val_predictions, y_val)
```


__output__

    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=1500,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)


<br/>

### 정확도 비교

학습된 블렌더로 테스트 세트에 대해 정확도를 살펴볼 차례이다. 블렌더로 테스트 세트를 예측하기 위해 각 예측기 별로 테스트 세트에 대한 예측값을 생성한다.


```python
X_test_predictions = np.empty((len(X_test), len(estimators)))

for index, estimator in enumerate(estimators):
    X_test_predictions[:, index] = estimator.predict(X_test)
```

<br/>

블렌더로 테스트 세트에 대한 정확도를 살펴본다.


```python
y_pred = logistic_blender.predict(X_test_predictions)
accuracy_score(y_test, y_pred)
```


__output__

    0.9557


<br/>

약 95퍼센트의 정확도가 나왔다. 하지만 개별 예측기 보다 정확도가 높아야 의미가 있기 때문에 개별 예측기들의 테스트 세트에 대한 정확도를 살펴본다.


```python
[estimator.score(X_test, y_test) for estimator in estimators]
```


__output__

    [0.9645, 0.9691, 0.866]


<br/>

랜덤 포레트스와 엑스트라 트리 모델이 오히려 더 정확도가 높다. 3가지의 문제 상황이 있을 것 같다.
+ 구현을 잘못한 경우
+ 서포트 벡터 머신이 성능을 저하
+ 블렌더 모델을 잘못 선택


먼저 구현을 잘못한 것인지 확인해 보기 위해 사이킷런에서 제공하는 스태킹 모델과 비교해 본다. 예측기 모델은 위에서 훈련 세트에 대해 학습한 모델을 그대로 복사해서 사용한다.


```python
from sklearn.ensemble import StackingClassifier

estimators = [('rf', clone(random_forest_clf)), ('extree', clone(extra_trees_clf)), ('svm', clone(svm_clf))]
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1500))
clf.fit(X_val, y_val)
y_pred_stacking = clf.predict(X_test)
accuracy_score(y_test, y_pred_stacking)
```


__output__

    0.9514


<br/>

직접 구현한 것보다 정확도가 조금 높긴 하지만 그래도 똑같이 예측기의 정확도보다 높지 않은 것을 보아 구현을 잘못한 것은 아닌거 같다. 이번에는 서포트 벡터 머신이 성능을 저하 시킨건지 확인해 보기 위해 예측기에서 제외해 본다.


```python
estimators = [random_forest_clf, extra_trees_clf]
X_val_predictions = np.empty((len(X_val), len(estimators)))

for index, estimator in enumerate(estimators):
    X_val_predictions[:, index] = estimator.predict(X_val)

logistic_blender.fit(X_val_predictions, y_val)
```


__output__

    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=1500,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)


<br/>


```python
X_test_predictions = np.empty((len(X_test), len(estimators)))

for index, estimator in enumerate(estimators):
    X_test_predictions[:, index] = estimator.predict(X_test)

y_pred = logistic_blender.predict(X_test_predictions)
accuracy_score(y_test, y_pred)
```


__output__

    0.9612


<br/>

직접 구현한 스태킹 모델은 정확도가 올라갔다. 하지만 아직까진 예측기보다 성능이 조금이긴 하지만 떨어진다. 사이킷런의 스태킹 모델 정확도를 살펴보자.


```python
from sklearn.ensemble import StackingClassifier

estimators = [('rf', clone(random_forest_clf)), ('extree', clone(extra_trees_clf))]
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1500))
clf.fit(X_val, y_val)
y_pred_stacking = clf.predict(X_test)
accuracy_score(y_test, y_pred_stacking)
```


__output__

    0.9552


<br/>

마찬가지로 정확도가 올라가긴 했지만 아까와 같이 구현한 모델보다 정확도가 떨어진다.

이번에는 정확도가 더 올라갈 것이라 기대해 보고 블렌더를 바꿔보자. 로지스틱 모델을 예측기에 넣고 랜덤 포레스트 모델을 블렌더로 바꾼다.


```python
logistic_reg = LogisticRegression(max_iter=1500)
logistic_reg.fit(X_train, y_train)
random_forest_blender = RandomForestClassifier(n_estimators=100, random_state=42)

estimators = [extra_trees_clf, logistic_reg]
X_val_predictions = np.empty((len(X_val), len(estimators)))

for index, estimator in enumerate(estimators):
    X_val_predictions[:, index] = estimator.predict(X_val)

random_forest_blender.fit(X_val_predictions, y_val)
```

__output__

    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=42, verbose=0,
                           warm_start=False)


<br/>

```python
X_test_predictions = np.empty((len(X_test), len(estimators)))

for index, estimator in enumerate(estimators):
    X_test_predictions[:, index] = estimator.predict(X_test)

y_pred = random_forest_blender.predict(X_test_predictions)
accuracy_score(y_test, y_pred)
```


__output__

    0.9686


<br/>

조금씩 오르긴 하지만 기대한 만큼 나오지는 않았다. warning메세지가 나오는건 로지스틱 모델이 최소비용에 수렴하기 전에 학습이 끝났기 때문인데 `max_iter`값을 올리다보니 실행 시간이 너무 길어져서 1500정도까지밖에 올리지 못했다. 수렴할 때 까지 학습을 시키면 정확도가 더 올라갈 것 같다.


```python
from sklearn.ensemble import StackingClassifier

estimators = [('extree', clone(extra_trees_clf)), ('logistic', clone(logistic_reg))]
clf = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier(n_estimators=100, random_state=42))
clf.fit(X_val, y_val)
y_pred_stacking = clf.predict(X_test)
accuracy_score(y_test, y_pred_stacking)
```

__output__

    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)


    0.9568


<br/>

마찬가지로 사이킷런의 로지스틱 모델도 정확도가 오르긴 했지만 기대한 만큼은 아니다. 나중에 `max_iter`값을 더 늘리거나 MNIST데이터를 전처리해서 크기를 좀 더 줄여야 정확한 성능을 알 수 있을 것 같다.

어쨌든 직접 구현한 모델과 사이킷런의 모델을 비교해 봤을 때 오히려 직접 구현한 모델의 성능이 근소하지만 조금 더 좋게 나왔다.
