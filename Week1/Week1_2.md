
# 📘 1주차 - 파이썬 기초 (2번째 수업)

---

## 🧱 객체지향 언어

### 객체(Object)
객체는 자체의 속성과 행동을 함께 가지는 독립적인 단위로, 현실 세계의 사물이나 개념을 소프트웨어적으로 모델링한 것입니다.  
간단히 말하면 **자체적인 변수와 함수를 갖는 새로운 속성의 코드**입니다.

### 등장 배경
소프트웨어가 커지면서 절차지향 언어로는 유지보수가 어려워졌고, 객체지향언어는 **다수의 객체**를 통해 현실 세계와 유사하게 시스템을 구성합니다.  
예: 지도를 그릴 때 건물을 하나의 객체로 보고 위치, 기능 등을 각각 속성으로 저장하면 수정이 쉬움.

---

## 🏗️ 클래스(Class)

클래스는 객체를 만들기 위한 설계도입니다. 모든 파이썬의 자료형은 클래스입니다.

```python
class A:
    B = 1
    def C(self):
        print('C')
```

클래스를 인스턴스로 선언하여 사용합니다.

```python
test = A()
print(test.B)  # 클래스 내부 변수 접근
test.C()       # 클래스 내부 메소드 호출
```

---

### 📌 self

```python
class A:
    B = 1

    def C(self):
        print(self.B)

    def D(self):
        self.B += 1

test = A()
test.C()
test.D()
test.C()
```

`self`는 클래스 자기 자신을 가리킵니다. 내부 변수와 메소드에 접근할 때 사용합니다.

---

## ✨ 매직 메소드

### 생성자 `__init__`

```python
class A:
    def __init__(self):
        print('인스턴스 생성됨')

test = A()
```

### 매개변수 사용 예

```python
class A:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def B(self):
        print(self.a, self.b)

test = A(1, 2)
test.B()
```

---

## 🔁 상속

```python
class person:
    def __init__(self, a, b):
        self.age = a
        self.country = b

    def old(self):
        self.age += 1

class programmer(person):
    def __init__(self, a, b):
        super().__init__(a, b)

    def prin(self):
        super().old()
        print(self.age, self.country)

test = programmer(26, "korea")
test.old()
test.prin()
```

- `super()`는 부모 클래스의 메소드를 호출할 때 사용
- 자식 클래스는 다수의 부모 클래스를 가질 수 있음

---

## 📦 import와 패키지 사용

```python
import math
print(math.sqrt(16))  # 4.0

import numpy as np
print(np.array([1, 2, 3]))

from math import pi
print(pi)
```

---

## 🔢 NumPy 기본 사용법

NumPy는 고성능 수치 계산 라이브러리입니다.

### 배열 생성

```python
a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])
print(a.shape)
print(b.shape)
```

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
print(a + 5)
```

---

### 인덱싱 / 슬라이싱

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr[0, 1])     # 2
print(arr[:, 1])     # 두 번째 열
print(arr[arr > 3])  # 4, 5, 6
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
print(a.reshape(3, 2))
print(a.flatten())
```

---

### 난수 생성

```python
np.random.seed(0)
print(np.random.rand(3, 3))
print(np.random.randn(3, 3))
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
