# 📘 1주차 - 파이썬 기초 (2번째 수업)

---

## 📦 import와 패키지 사용

### 🔹 기본 import 사용법

```python
import math
print(math.sqrt(16))  # 4.0
```

---

### 🔹 alias 사용

```python
import numpy as np
arr = np.array([1, 2, 3])
print(arr)
```

---

### 🔹 from-import 구문

```python
from math import pi
print(pi)
```

---

## 🔢 NumPy 기본 사용법

### 배열 생성

```python
a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])
print(a.shape)
print(b.shape)
```

---

### 배열 생성 함수

```python
np.zeros((2, 3))
np.ones((3, 3))
np.eye(4)
np.arange(0, 10, 2)
np.linspace(0, 1, 5)
```

---

### 브로드캐스팅

```python
a = np.array([1, 2, 3])
b = 5
print(a + b)
```

---

### 인덱싱과 슬라이싱

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr[0, 1])
print(arr[:, 1])
print(arr[arr > 3])
```

---

### 통계 함수

```python
data = np.array([1, 2, 3, 4, 5])
print(np.mean(data))
print(np.std(data))
print(np.max(data))
```

---

### 배열 변형

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
a_reshaped = a.reshape(3, 2)
a_flattened = a.flatten()
```

---

### 난수 생성

```python
np.random.seed(0)
np.random.rand(3, 3)
np.random.randn(3, 3)
```

---

### 선형대수 연산

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[2, 0], [1, 2]])
print(np.dot(a, b))
print(a @ b)
print(a.T)
```

---

## 📝 실습 과제

### 과제 1️⃣: 통계 계산

```python
data = input("숫자 여러 개 입력 (예: 1 2 3): ")
nums = np.array([int(i) for i in data.split()])
print("최댓값:", np.max(nums))
print("최솟값:", np.min(nums))
print("평균:", np.mean(nums))
print("표준편차:", np.std(nums))
```

---

### 과제 2️⃣: 각 요소에 5 더하기

```python
plus5 = nums + 5
print("각 요소에 5를 더한 결과:", plus5)
```
