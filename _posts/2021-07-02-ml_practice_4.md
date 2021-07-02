---
layout: post
title: "엑스트라트리"
excerpt: "선형 모델 학습"
category: ml_practice
date: 2021-07-02
last_modified_at: 2021-07-02
use_math: true
---

```python
# 파이썬 ≥3.5 필수
import sys
assert sys.version_info >= (3, 5)

# 사이킷런 ≥0.20 필수
import sklearn
assert sklearn.__version__ >= "0.22"

# 공통 모듈 임포트
import numpy as np
import os

# 노트북 실행 결과를 동일하게 유지하기 위해
np.random.seed(42)
```

### 결정트리 모델 훈련

데이터는 moons데이터셋을 사용한다. 샘플수는 10000개에 잡음을 추가해주고 `train_test_split`를 사용해서 훈련세트와 테스트세트를 8:2비율로 나눠준다.


```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

<br/>

엑스트라 트리는 랜덤 포레스트 모델 보다 무작위성이 더 추가된 모델이다. 그렇기 때문에 결정트리 모델의 `splitter`와 `max_features`를 바꿔준다. 그리고 다른 파라미터는 그리드 탐색으로 최적의 값을 찾아낸다.
+ `max_leaf_nodes` : 리프 노드의 최대 개수
+ `min_samples_split` : 노드 분할에 필요한 최소 샘플 개수
+ `splitter` : 기본 값은 `best`이다. `best`일때는 노드를 분할 할때 특성의 최적 값을 찾아서 분할 하지만 `random`을 사용하면 특성 값을 무작위로 정한 후 최적의 분할을 하는 값을 선택하는 형식이다.
+ `max_features` : 사용할 특성의 개수를 정한다. `auto`로 설정하면 $\sqrt{특성 수}$ 만큼 무작위로 선택한다.

사실 사이킷런에서 제공하는 앙상블 학습 모델인 `ExtraTreesClassifier`의 결정트리 모델이 `ExtraTreeClassifier`인데 `DecisionTreeClassifier`에서 `splitter`와 `max_features`를 `random`과 `auto`로 바꾼 것과 똑같다. 때문에 `DecisionTreeClassifier`대신 `ExtraTreeClassifier`를 써도 같은 결과가 나온다.


```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(splitter='random', max_features='auto'), params, verbose=1, cv=3)

grid_search_cv.fit(X_train, y_train)
```

__output__

    Fitting 3 folds for each of 294 candidates, totalling 882 fits    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 882 out of 882 | elapsed:    2.2s finished
    
    GridSearchCV(cv=3, error_score=nan,
                 estimator=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None,
                                                  criterion='gini', max_depth=None,
                                                  max_features='auto',
                                                  max_leaf_nodes=None,
                                                  min_impurity_decrease=0.0,
                                                  min_impurity_split=None,
                                                  min_samples_leaf=1,
                                                  min_samples_split=2,
                                                  min_weight_fraction_leaf=0.0,
                                                  presort='deprecated',
                                                  random_state=None,
                                                  splitter='random'),
                 iid='deprecated', n_jobs=None,
                 param_grid={'max_leaf_nodes': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                13, 14, 15, 16, 17, 18, 19, 20, 21,
                                                22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                31, ...],
                             'min_samples_split': [2, 3, 4]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=1)

<br/>

그리드 탐색으로 찾은 최적의 `max_leaf_nodes`와 `min_samples_split`의 값이다.


```python
grid_search_cv.best_estimator_
```


__output__

    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=92,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=4,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=None, splitter='random')

<br/>

### 엑스트라 트리 구현하기

엑스트라 트리가 랜덤 포레스트와 또 다른 점은 bootstrap을 사용하지 않는 것이다. bootstrap을 사용하지 않으면 훈련 세트 전체를 사용하기 때문에 따로 샘플을 나누지 않고 앙상블 학습에 사용할 모델의 개수만 정한다.


```python
from sklearn.base import clone
from sklearn.metrics import accuracy_score

n_trees = 1000
extra_forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)]

Y_pred = np.empty([n_trees, len(X_test)])

for tree_index, tree in enumerate(extra_forest):
    tree.fit(X_train, y_train)
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

    array([957, 996, 988, 997, 880])

<br/>

### 모델 비교

사이킷런에서 제공하는 엑스트라 트리 모델, 배깅 모델과 직접 구현한 모델을 비교해 볼 것이다. 하이퍼 파라미터는 앞서 그리드 탐색에서 찾았던 하이퍼 파라미터와 똑같이 사용한다.


```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier

ext_clf = ExtraTreesClassifier(n_estimators=1000, max_leaf_nodes=(grid_search_cv.best_estimator_.max_leaf_nodes),
                               min_samples_split=(grid_search_cv.best_estimator_.min_samples_split), random_state=42)
ext_clf.fit(X_train, y_train)

bag_clf = BaggingClassifier(base_estimator=clone(grid_search_cv.best_estimator_), n_estimators=1000, bootstrap=False)
bag_clf.fit(X_train, y_train)

bag_pred = bag_clf.predict(X_test)
y_pred_ext = ext_clf.predict(X_test)
```

<br/>

사이킷런에서 제공하는 엑스트라 트리 모델의 정확도이다.


```python
accuracy_score(y_test, y_pred_ext)
```


__output__

    0.872

<br/>

배깅 모델의 정확도이다.


```python
accuracy_score(y_test, bag_pred)
```


__output__

    0.873

<br/>

사이킷런의 엑스트라 트리 모델과 배깅 모델의 정확도는 약 87퍼센트가 나왔다. 이제 직접 구현한 랜덤 포레스트 모델의 정확도를 측정해본다.


```python
accuracy_score(y_test, y_pred_majority_votes.T)
```


__output__

    0.8725

<br/>

약 87퍼센트로 거의 유사하게 나왔다. 그러면 얼마나 비슷하게 예측했는지 예측값끼리 비교해본다.


```python
accuracy_score(y_pred_ext, y_pred_majority_votes.T)
```


__output__

    0.9955


<br/>

```python
accuracy_score(bag_pred, y_pred_majority_votes.T)
```


__output__

    0.9955

<br/>

거의 똑같이 예측한 것으로 보인다. 그럼 이번에는 다르게 예측한 샘플의 빈도수를 확인해본다.


```python
votes_list = []
for index, (i, j) in enumerate(zip(y_pred_ext, y_pred_majority_votes.T)):
  if i != j:
    votes_list.append(n_votes[0][index])

votes_list
```


__output__

    [550, 510, 532, 525, 508, 522, 507, 513, 521]

<br/>

빈도수를 살펴보니 500근처의 값이 나왔다. 1000개의 모델도 확실하게 결정하기 어려운 샘플인 것 같다. 3개의 모델의 정확도가 조금씩 다른것도 아마 저 샘플들 때문이 아닐까 싶다.
