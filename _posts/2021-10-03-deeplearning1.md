---
layout: post
title: "텐서플로우"
excerpt: "텐서 정리"
category: deep_learning
date: 2021-10-03
last_modified_at: 2021-10-03
use_math: true
---

## 텐서

```python
import tensorflow as tf
import numpy as np
```

텐서는 다차원 배열이다. 그리고 np.arrays와 비슷하지만 Python의 숫자 및 문자열과 같이 변경할 수 없다.

새로운 텐서를 만들 수만 있고 내용을 업데이트 할 수는 없다.

<br/>
<br/>

## 기본 텐서 조작

### 스칼라(0차원)

스칼라 텐서는 하나의 값을 가지며 축이 없다. 기본 dtype은 int32이다.


```python
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)
```

__output__

    tf.Tensor(4, shape=(), dtype=int32)
    
<br/>

### 벡터(1차원)

벡터 텐서는 값들의 list이고 하나의 축을 가지고 있다. 텐서 생성시 dtype을 따로 지정하지 않으면 값에 따라서 자동으로 변한다.


```python
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)
```

__output__

    tf.Tensor([2. 3. 4.], shape=(3,), dtype=float32)
    
<br/>

```python
rank_1_tensor = tf.constant([2, 3, 4])
print(rank_1_tensor)
```

__output__

    tf.Tensor([2 3 4], shape=(3,), dtype=int32)

<br/>

### 행렬(2차원)

행렬 텐서는 2개의 축을 가지고 있다. dtype을 아래와 같이 따로 지정해 줄 수 있다.


```python
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor)
```

__output__

    tf.Tensor(
    [[1. 2.]
     [3. 4.]
     [5. 6.]], shape=(3, 2), dtype=float16)
    
<br/>

### 3차원

텐서는 2개의 축 뿐만 아니라 더 많은 축을 가질 수 있다. 다음은 3개의 축을 가지고 있는 텐서이다.


```python
rank_3_tensor = tf.constant([
                            [[0, 1, 2, 3, 4],
                            [5, 6, 7, 8, 9]],
                            [[10, 11, 12, 13, 14],
                            [15, 16, 17, 18, 19]],
                            [[20, 21, 22, 23, 24],
                            [25, 26, 27, 28, 29]],])
print(rank_3_tensor)
```

__output__

    tf.Tensor(
    [[[ 0  1  2  3  4]
      [ 5  6  7  8  9]]
    
     [[10 11 12 13 14]
      [15 16 17 18 19]]
    
     [[20 21 22 23 24]
      [25 26 27 28 29]]], shape=(3, 2, 5), dtype=int32)

<br/>

### NumPy 배열로 변환

`np.array` 또는 `tensor.numpy` 메소드를 사용하여 텐서를 NumPy 배열로 변환할 수 있다.


```python
np.array(rank_2_tensor)
```

__output__


    array([[1., 2.],
           [3., 4.],
           [5., 6.]], dtype=float16)

<br/>


```python
rank_2_tensor.numpy()
```


__output__

    array([[1., 2.],
           [3., 4.],
           [5., 6.]], dtype=float16)

<br/>

### 산술 연산

텐서끼리 기본적인 산술 연산을 수행할 수 있다.


```python
a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]])
print(tf.add(a, b), "\n")        #덧셈
print(tf.multiply(a, b), "\n")   #값끼리 곱셈
print(tf.matmul(a, b), "\n")     #행렬의 곱셈
print(tf.subtract(a, b), "\n")   #뺄셈
print(tf.divide(a, b))           #나눗셈
```

__output__

    tf.Tensor(
    [[2 3]
     [4 5]], shape=(2, 2), dtype=int32) 
    
    tf.Tensor(
    [[1 2]
     [3 4]], shape=(2, 2), dtype=int32) 
    
    tf.Tensor(
    [[3 3]
     [7 7]], shape=(2, 2), dtype=int32) 
    
    tf.Tensor(
    [[0 1]
     [2 3]], shape=(2, 2), dtype=int32) 
    
    tf.Tensor(
    [[1. 2.]
     [3. 4.]], shape=(2, 2), dtype=float64)
    
<br/>

메소드 사용 없이 직접 연산도 가능하다.


```python
print(a + b, "\n")  #덧셈
print(a * b, "\n")  #값끼리 곱셈(multiply)
print(a @ b)        #행렬의 곱셈(matmul)
```

__output__

    tf.Tensor(
    [[2 3]
     [4 5]], shape=(2, 2), dtype=int32) 
    
    tf.Tensor(
    [[1 2]
     [3 4]], shape=(2, 2), dtype=int32) 
    
    tf.Tensor(
    [[3 3]
     [7 7]], shape=(2, 2), dtype=int32)

<br/>

모든 종류의 연산에도 사용가능하다.


```python
c = tf.constant([[4.0, 5.0], [10.0, 1.0]])

print(tf.reduce_max(c))   #가장 큰 값
print(tf.argmax(c))       #가장 큰 값의 인덱스
print(tf.nn.softmax(c))   #softmax 연산
```

__output__

    tf.Tensor(10.0, shape=(), dtype=float32)
    tf.Tensor([1 0], shape=(2,), dtype=int64)
    tf.Tensor(
    [[2.6894143e-01 7.3105854e-01]
     [9.9987662e-01 1.2339458e-04]], shape=(2, 2), dtype=float32)

<br/>
<br/>

## Shape

+ Shpae : 텐서에서 각 차원(축)의 길이(요소의 수)
+ Rank : 텐서 축(차원)의 수 ex)스칼라 = 0, 벡터 = 1
+ Axis 또는 Dimension : 텐서에서 어느 한 특정 차원
+ Size : 텐서의 총 항목 수

`tf.zeros`메소드를 이용해 4차원 텐서를 생성 후 위의 속성들을 알아 볼 수 있다.


```python
rank_4_tensor = tf.zeros([3, 2, 4, 5])

print("모든 요소의 타입 :", rank_4_tensor.dtype)
print("축(차원)의 수 :", rank_4_tensor.ndim)
print("텐서의 모양 :", rank_4_tensor.shape)
print("텐서의 첫번째 축 요소 수 :", rank_4_tensor.shape[0])
print("텐서의 마지막 축 요소 수 :", rank_4_tensor.shape[-1])
print("모든 요소의 수(3*2*4*5) :", tf.size(rank_4_tensor).numpy())
```

![png](https://raw.githubusercontent.com/min020/deep_learning/main/picture/4-axis_block.png)

__output__

    모든 요소의 타입 : <dtype: 'float32'>
    축(차원)의 수 : 4
    텐서의 모양 : (3, 2, 4, 5)
    텐서의 첫번째 축 요소 수 : 3
    텐서의 마지막 축 요소 수 : 5
    모든 요소의 수(3*2*4*5) : 120

<br/>

### Shape 조작하기

`shape`는 각 차원의 크기를 보여주는 TensorShape 객체를 리턴한다.


```python
var_x = tf.Variable(tf.constant([[1], [2], [3]]))
print(var_x.shape)
```

__output__

    (3, 1)
    
<br/>

해당 객체를 파이썬의 리스트로 변환할 수 있다.


```python
print(var_x.shape.as_list())
print(var_x.shape.as_list()[0])
```

__output__

    [3, 1]
    3
    
<br/>    

텐서를 새로운 shape으로 바꿀 수 있다. 주의해야 할 점은 기존 텐서와 새로운 shape의 텐서의 요소 수가 같아야 한다.


```python
reshaped = tf.reshape(var_x, [1, 3])

print(var_x.numpy())
print(var_x.shape, "\n")
print(reshaped.numpy())
print(reshaped.shape)
```

__output__

    [[1]
     [2]
     [3]]
    (3, 1) 
    
    [[1 2 3]]
    (1, 3)
    
<br/>

데이터는 메모리안에서 레이아웃을 유지하고 있다가 새로운 shape의 요청이 들어오면 같은 데이터를 가르키는 새로운 텐서가 생성된다.

텐서를 평평하게 만들면 어떤 순서로 메모리에 배치되어 있는지 확인할 수 있다.


```python
print(rank_3_tensor.numpy(), "\n")
print(tf.reshape(rank_3_tensor, [-1]).numpy())
```

__output__

    [[[ 0  1  2  3  4]
      [ 5  6  7  8  9]]
    
     [[10 11 12 13 14]
      [15 16 17 18 19]]
    
     [[20 21 22 23 24]
      [25 26 27 28 29]]] 
    
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
     24 25 26 27 28 29]
    
<br/>

일반적인 `tf.reshape`의 용도는 인접한 축을 결합하거나 분할하는 것이다.

이 $3\times2\times5$ 텐서의 경우 $(3\times2)\times5$ 또는 $3\times(2\times5)$로 재구성하는 것이 합리적이다.


```python
print(tf.reshape(rank_3_tensor, [3*2, 5]), "\n")
print(tf.reshape(rank_3_tensor, [3, -1]))
```

__output__

    tf.Tensor(
    [[ 0  1  2  3  4]
     [ 5  6  7  8  9]
     [10 11 12 13 14]
     [15 16 17 18 19]
     [20 21 22 23 24]
     [25 26 27 28 29]], shape=(6, 5), dtype=int32) 
    
    tf.Tensor(
    [[ 0  1  2  3  4  5  6  7  8  9]
     [10 11 12 13 14 15 16 17 18 19]
     [20 21 22 23 24 25 26 27 28 29]], shape=(3, 10), dtype=int32)

![png](https://raw.githubusercontent.com/min020/deep_learning/main/picture/reshape-good.png)

<br/>

요소 수가 같으면 `rf.reshape`이 작동하지만, 축의 순서를 고려하지 않으면 별로 쓸모가 없다.


```python
print(tf.reshape(rank_3_tensor, [2, 3, 5]), "\n")
print(tf.reshape(rank_3_tensor, [5, 6]))
```

__output__

    tf.Tensor(
    [[[ 0  1  2  3  4]
      [ 5  6  7  8  9]
      [10 11 12 13 14]]
    
     [[15 16 17 18 19]
      [20 21 22 23 24]
      [25 26 27 28 29]]], shape=(2, 3, 5), dtype=int32) 
    
    tf.Tensor(
    [[ 0  1  2  3  4  5]
     [ 6  7  8  9 10 11]
     [12 13 14 15 16 17]
     [18 19 20 21 22 23]
     [24 25 26 27 28 29]], shape=(5, 6), dtype=int32)
     
![png](https://raw.githubusercontent.com/min020/deep_learning/main/picture/reshape-bad.png)     
     
<br/>
<br/>

## 인덱싱

TensorFlow는 표준 파이썬 인덱싱 규칙과 numpy 인덱싱의 기본 규칙을 따른다.
+ 인덱스는 0부터 시작
+ 음수 인덱스는 끝에서부터 계산
+ 콜론`:`은 슬라이스에 사용


```python
rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print(rank_1_tensor.numpy())
```

__output__

    [ 0  1  1  2  3  5  8 13 21 34]
    
<br/>

스칼라를 사용하여 인덱싱하면 축이 제거된다.(차원이 낮아진다)


```python
print("첫번째 :", rank_1_tensor[0].numpy())
print("두번째 :", rank_1_tensor[1].numpy())
print("마지막 :", rank_1_tensor[-1].numpy())
```

__output__

    첫번째 : 0
    두번째 : 1
    마지막 : 34
    
<br/>

`:`슬라이스를 사용하여 인덱싱하면 축이 유지된다.


```python
print("전체 :", rank_1_tensor[:].numpy())
print("4번 인덱스 이전 :", rank_1_tensor[:4].numpy())
print("4번 인덱스부터 끝까지 :", rank_1_tensor[4:].numpy())
print("2번 인덱스부터 7번인덱스 전까지:", rank_1_tensor[2:7].numpy())
print("2개씩 건너뛰어서 인덱싱 :", rank_1_tensor[::2].numpy())
print("거꾸로 :", rank_1_tensor[::-1].numpy())
```

__output__

    전체 : [ 0  1  1  2  3  5  8 13 21 34]
    4번 인덱스 이전 : [0 1 1 2]
    4번 인덱스부터 끝까지 : [ 3  5  8 13 21 34]
    2번 인덱스부터 7번인덱스 전까지: [1 2 3 5 8]
    2개씩 건너뛰어서 인덱싱 : [ 0  1  3  8 21]
    거꾸로 : [34 21 13  8  5  3  2  1  1  0]
    
<br/>

더 높은 차원의 텐서는 여러 인덱스를 전달하여 인덱싱된다.

벡터에서와 같은 규칙이 각 축에 독립적으로 적용된다.


```python
print(rank_2_tensor.numpy())
```

__output__

    [[1. 2.]
     [3. 4.]
     [5. 6.]]
    
<br/>

각 인덱스에 정수를 전달하면 스칼라가 결과로 나온다.


```python
print(rank_2_tensor[1, 1].numpy())
```

__output__

    4.0
    
<br/>

정수와 슬라이스를 조합하여 인덱싱 할 수 있다.


```python
print("두번째 행 :", rank_2_tensor[1, :].numpy())
print("두번째 열 :", rank_2_tensor[:, 1].numpy())
print("마지막 행 :", rank_2_tensor[-1, :].numpy())
print("첫번째 행 제외 :")
print(rank_2_tensor[1:, :].numpy())
```

__output__

    두번째 행 : [3. 4.]
    두번째 열 : [2. 4. 6.]
    마지막 행 : [5. 6.]
    첫번째 행 제외 :
    [[3. 4.]
     [5. 6.]]
    
<br/>

3차원 텐서의 예시이다.


```python
print(rank_3_tensor.numpy())
```

__output__

    [[[ 0  1  2  3  4]
      [ 5  6  7  8  9]]
    
     [[10 11 12 13 14]
      [15 16 17 18 19]]
    
     [[20 21 22 23 24]
      [25 26 27 28 29]]]
    
<br/>

```python
print(rank_3_tensor[:, :, 4])
```

__output__

    tf.Tensor(
    [[ 4  9]
     [14 19]
     [24 29]], shape=(3, 2), dtype=int32)
        
![png](https://raw.githubusercontent.com/min020/deep_learning/main/picture/index1.png)        
        
<br/>
<br/>

## DTypes

`tf.Tensor`의 데이터 유형을 확인하기 위해서는 `Tensor.dtype` 속성을 사용한다.

`tf.Tensor`를 만들 때 데이터 유형을 선택해서 지정할 수 있다.

그렇지 않으면 TensorFlow는 데이터를 나타낼 수 있는 유형을 자동으로 선택한다. 

정수는 `tf.int32`로, 부동소수점은 `tf.float32`로 변환한다.


```python
the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)
the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)
print(the_f16_tensor, "\n")
the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)    #부동소수점을 정수형으로 변환하면 수소점은 제외된다.
print(the_u8_tensor)
```

__output__

    tf.Tensor([2.2 3.3 4.4], shape=(3,), dtype=float16) 
    
    tf.Tensor([2 3 4], shape=(3,), dtype=uint8)
    
<br/>
<br/>

## 브로드캐스팅

작은 크기의 텐서가 결합 연산을 실행하면 더 큰 크기의 텐서에 맞게 자동으로 확장되는 개념이다.

가장 간단하고 일반적인 경우는 스칼라 텐서와 다른 크기의 텐서를 곱할 때 스칼라 텐서가 다른 인수와 같은 형상으로 브로드캐스트된다.


```python
x = tf.constant([1, 2, 3])
y = tf.constant(2)
z = tf.constant([2, 2, 2])

print(tf.multiply(x, 2))
print(x * y)
print(x * z)
```

__output__

    tf.Tensor([2 4 6], shape=(3,), dtype=int32)
    tf.Tensor([2 4 6], shape=(3,), dtype=int32)
    tf.Tensor([2 4 6], shape=(3,), dtype=int32)
    
<br/>

축의 크기가 1인 인수는 다른 인수와 일치하도록 확장할 수 있다.

이 경우 $3\times1$ 행렬에 요소별로 $1\times4$ 행렬을 곱하여 $3\times4$ 행렬을 만든다.


```python
x = tf.reshape(x,[3,1])
y = tf.range(1, 5)

print(x, "\n")
print(y, "\n")
print(tf.multiply(x, y))
```

__output__

    tf.Tensor(
    [[1]
     [2]
     [3]], shape=(3, 1), dtype=int32) 
    
    tf.Tensor([1 2 3 4], shape=(4,), dtype=int32) 
    
    tf.Tensor(
    [[ 1  2  3  4]
     [ 2  4  6  8]
     [ 3  6  9 12]], shape=(3, 4), dtype=int32)
 
<br/> 

만약 브로드캐스팅 없이 같은 연산을 하려면 다음과 같이 실행해야 한다.


```python
x_stretch = tf.constant([[1, 1, 1, 1],
                         [2, 2, 2, 2],
                         [3, 3, 3, 3]])

y_stretch = tf.constant([[1, 2, 3, 4],
                         [1, 2, 3, 4],
                         [1, 2, 3, 4]])

print(x_stretch * y_stretch)
```

__output__

    tf.Tensor(
    [[ 1  2  3  4]
     [ 2  4  6  8]
     [ 3  6  9 12]], shape=(3, 4), dtype=int32)
    
<br/>

브로드캐스팅은 메모리에서 브로드캐스트 연산으로 확장된 텐서를 구체화하지 않기 때문에 시간과 공간적으로 효율적이다.

`tf.broadcast_to`를 사용하여 브로드캐스팅이 어떤 모습인지 알 수 있다.


```python
print(tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3]))
```

__output__

    tf.Tensor(
    [[1 2 3]
     [1 2 3]
     [1 2 3]], shape=(3, 3), dtype=int32)
    
<br/>
<br/>

## tf.convert_to_tensor

`tf.matmul` 및 `tf.reshape`와 같은 대부분의 연산은 클래스 `tf.Tensor`의 인수를 사용한다.

전부는 아니지만 대부분의 연산은 텐서가 아닌 인수에 대해 `convert_to_tensor`를 호출한다.

변환 레지스트리가 있어서 NumPy의 `ndarray`, `TensorShape`, Python의 list, `tf.Variable`과 같은 대부분의 객체 클래스는 자동으로 변환된다.

<br/>
<br/>

## 비정형 텐서

축마다 요소의 수가 다양한 텐서를 비정형(ragged)텐서라고 한다.

비정형 텐서는 정규 텐서로 표현할 수 없고 `tf.ragged.RaggedTensor`를 사용한다.


```python
ragged_list = [
              [0, 1, 2, 3],
              [4, 5],
              [6, 7, 8],
              [9]]
try:
  tensor = tf.constant(ragged_list)
except Exception as e:
  print(f"{type(e).__name__}: {e}")
```

__output__

    ValueError: Can't convert non-rectangular Python sequence to Tensor.
    
<br/>

```python
ragged_tensor = tf.ragged.constant(ragged_list)
print(ragged_tensor)
```

__output__

    <tf.RaggedTensor [[0, 1, 2, 3], [4, 5], [6, 7, 8], [9]]>
    
<br/>

그리고 `tf.RaggedTensor`의 shape에는 축마다 요소의 수가 다르기 때문에 길이를 알 수 없는 축이 포함되어 있다.


```python
print(ragged_tensor.shape)
```

__output__

    (4, None)
    
![png](https://raw.githubusercontent.com/min020/deep_learning/main/picture/ragged.png)    
    
<br/>
<br/>

## 문자열 텐서

`tf.string`은 텐서에서 문자열과 같은 데이터를 나타낼 수 있다.

문자열은 쪼갤수 없으므로 Python 문자열과 같은 방식으로 인덱싱할 수 없다.

문자열의 길이는 텐서 축 개수와 상관이 없다.


```python
scalar_string_tensor = tf.constant("Gray wolf")
print(scalar_string_tensor)
```

__output__

    tf.Tensor(b'Gray wolf', shape=(), dtype=string)
    
<br/>

```python
tensor_of_strings = tf.constant(["Gray wolf",
                                 "Quick brown fox",
                                 "Lazy dog"])

print(tensor_of_strings)
```

__output__

    tf.Tensor([b'Gray wolf' b'Quick brown fox' b'Lazy dog'], shape=(3,), dtype=string)

<br/>

출력에서 b 접두사는 `tf.string` dtype이 유니코드 문자열이 아니라 바이트 문자열임을 나타내는 것이다.

유니코드 문자를 전달하면 UTF-8로 인코딩된다.


```python
tf.constant("🥳👍")
```

__output__


    <tf.Tensor: shape=(), dtype=string, numpy=b'\xf0\x9f\xa5\xb3\xf0\x9f\x91\x8d'>

<br/>

`tf.strings.split`을 포함한 문자열의 기본 함수는 `tf.strings`에서 찾을 수 있다.

축의 개수가 0인 문자열 텐서를 분리하는 것은 상관 없지만 축의 개수가 1이상인 문자열 텐서를 분리하면 비정형 텐서로 리턴된다.

왜냐하면 문자열들이 각각 요소의 수가 다르게 분리될 수 있기 때문이다.


```python
print(tf.strings.split(scalar_string_tensor, sep=" "))
print(tf.strings.split(tensor_of_strings))
```

__output__

    tf.Tensor([b'Gray' b'wolf'], shape=(2,), dtype=string)
    <tf.RaggedTensor [[b'Gray', b'wolf'], [b'Quick', b'brown', b'fox'], [b'Lazy', b'dog']]>

![png](https://raw.githubusercontent.com/min020/deep_learning/main/picture/string-split.png)

<br/>

`tf.string.to_number` 메소드를 이용하면 숫자로 이루어진 문자열 텐서를 숫자로 변환할 수 있다.


```python
text = tf.constant("1 10 100")
print(tf.strings.to_number(tf.strings.split(text, " ")))
```

__output__

    tf.Tensor([  1.  10. 100.], shape=(3,), dtype=float32)
    
<br/>

문자열을 숫자로 변환할 수는 없지만 바이트로 변환 후 아스키 코드 값으로 변환할 수는 있다.


```python
byte_ints = tf.io.decode_raw(tf.constant("Duck"), tf.uint8)
print("Bytes:", byte_ints)
```

__output__

    Bytes: tf.Tensor([ 68 117  99 107], shape=(4,), dtype=uint8)
    
<br/>

유니코드 문자인 경우는 `tf.strings.unicode_decode`를 사용하면 변환 가능하다.


```python
unicode_bytes = tf.constant("가나다")
unicode_values = tf.strings.unicode_decode(unicode_bytes, "UTF-8")

print("Unicode bytes:", unicode_bytes)
print("\nUnicode values:", unicode_values)
```

__output__

    Unicode bytes: tf.Tensor(b'\xea\xb0\x80\xeb\x82\x98\xeb\x8b\xa4', shape=(), dtype=string)
    
    Unicode values: tf.Tensor([44032 45208 45796], shape=(3,), dtype=int32)
    
<br/>
<br/>

## 희소 텐서

텐서에 값이 모든 위치에 있는 것이 아니라 드문드문 있는 경우 사용할 수 있는 방식이다.

값이 있는 곳의 인덱스에 따라 비어 있지 않는 값들만 저장하여 효율적으로 데이터를 저장한다.


```python
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],  #값이 존재하는 곳의 인덱스
                                       values=[1, 2],             #위의 인덱스에 존재하는 값
                                       dense_shape=[3, 4])        #실제 희소 텐서의 모양

print(tf.sparse.to_dense(sparse_tensor))
```

__output__

    tf.Tensor(
    [[1 0 0 0]
     [0 0 2 0]
     [0 0 0 0]], shape=(3, 4), dtype=int32)

![png](https://raw.githubusercontent.com/min020/deep_learning/main/picture/sparse.png)
