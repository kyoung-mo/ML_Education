
# 📘 1주차 - 파이썬 기초 (2번째 수업)

---

## 📦 import와 패키지 사용

파이썬은 다양한 표준 라이브러리와 외부 패키지를 제공하며, `import` 문을 통해 필요한 기능을 불러와 사용할 수 있습니다.

- `import 모듈명`: 전체 모듈을 불러옵니다.
- `import 모듈명 as 별칭`: 모듈에 별칭을 붙여 간편하게 사용할 수 있습니다.
- `from 모듈 import 함수`: 특정 함수나 클래스만 불러올 수 있습니다.

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

**NumPy(Numerical Python)**는 다차원 배열 객체와 벡터화 연산을 지원하는 파이썬의 핵심 패키지입니다.  
머신러닝과 딥러닝에서 사용되는 수치 계산의 대부분은 NumPy 기반으로 구성됩니다.

---

### 배열 생성

NumPy의 `array()` 함수를 사용하여 1차원, 2차원, 다차원 배열을 생성할 수 있습니다.

```python
a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])
print(a.shape)
print(b.shape)
```

---

### 배열 생성 함수

배열을 손쉽게 초기화하는 함수들이 제공됩니다.  
`zeros`, `ones`, `eye`, `arange`, `linspace` 등이 대표적입니다.

```python
np.zeros((2, 3))
np.ones((3, 3))
np.eye(4)
np.arange(0, 10, 2)
np.linspace(0, 1, 5)
```

---

### 브로드캐스팅

크기가 다른 배열 간 연산을 자동으로 확장해서 처리하는 기능입니다.  
딥러닝에서 편향값을 더할 때와 같이 많이 사용됩니다.

```python
a = np.array([1, 2, 3])
b = 5
print(a + b)
```

---

### 인덱싱과 슬라이싱

배열의 특정 위치에 접근하거나, 특정 범위의 값을 추출할 수 있습니다.  
2차원 배열에서는 행과 열을 기준으로 접근합니다.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr[0, 1])
print(arr[:, 1])
print(arr[arr > 3])
```

---

### 통계 함수

`mean`, `std`, `max`, `min` 등 통계적 연산을 위한 다양한 함수들이 제공됩니다.  
데이터 분석과 전처리 과정에서 유용하게 사용됩니다.

```python
data = np.array([1, 2, 3, 4, 5])
print(np.mean(data))
print(np.std(data))
print(np.max(data))
```

---

### 배열 변형

`reshape`, `flatten` 등을 사용하여 배열의 형태를 자유롭게 바꿀 수 있습니다.  
머신러닝에서는 입력 데이터를 네트워크 구조에 맞게 조정할 때 사용됩니다.

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
a_reshaped = a.reshape(3, 2)
a_flattened = a.flatten()
```

---

### 난수 생성

난수 배열은 데이터 샘플링, 초기 가중치 설정 등에 사용됩니다.  
정규분포, 균등분포 기반 난수 생성이 가능합니다.

```python
np.random.seed(0)
np.random.rand(3, 3)
np.random.randn(3, 3)
```

---

### 선형대수 연산

`dot`, `@`, `transpose` 등을 통해 벡터와 행렬 연산을 수행할 수 있습니다.  
신경망의 연산과 직결되는 매우 중요한 기능입니다.

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

사용자로부터 여러 숫자를 입력받아 배열로 변환한 후,  
최댓값, 최솟값, 평균, 표준편차를 계산해 출력합니다.

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

입력받은 숫자 배열의 각 요소에 5를 더한 새 배열을 출력합니다.  
브로드캐스팅을 활용한 연산 예시입니다.

```python
plus5 = nums + 5
print("각 요소에 5를 더한 결과:", plus5)
```
