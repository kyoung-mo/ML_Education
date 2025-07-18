## 1️⃣ 단일 퍼셉트론 복습: OR 문제와 XOR 문제

###  🔹퍼셉트론(perceptron)이란?

퍼셉트론은 가장 단순한 형태의 인공신경망으로, 입력값 $x_1, x_2, \dots, x_n$에  
가중치 $w_1, w_2, \dots, w_n$를 곱하고 바이어스 $b$를 더한 값을  
활성화 함수에 통과시켜 출력값을 생성하는 구조이다.

$$
z = w \cdot x + b \\
$$
$$
y = \sigma(z)
$$

단일 퍼셉트론에서는 흔히 계단 함수(step function)나 시그모이드 함수 등을 사용하며, 출력은 주로 0 또는 1로 표현된다.

---

###  🔹OR 문제

**문제 정의**  
OR 게이트는 입력값 중 하나라도 1이면 출력이 1이 되는 논리 연산이다.

| $x_1$ | $x_2$ | $y = x_1  \text{ or }  x_2$ |
| ----- | ----- | ----------------------------- |
| 0     | 0     | 0                             |
| 0     | 1     | 1                             |
| 1     | 0     | 1                             |
| 1     | 1     | 1                             |

**파이썬 예제**

```python
import numpy as np
import matplotlib.pyplot as plt

# 퍼셉트론 함수 정의 (OR 문제)
def perceptron(x1, x2):
    x = np.array([x1, x2])
    w = np.array([1.0, 1.0])
    b = -0.5
    z = np.dot(w, x) + b
    return int(z > 0)

# 입력 데이터와 OR 결과 라벨
points = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
labels = np.array([perceptron(x1, x2) for x1, x2 in points])

# 시각화
plt.figure(figsize=(6, 6))
for point, label in zip(points, labels):
    # 출력 결과가 1이면 파란 점, 0이면 빨간 점으로 표시
    color = 'blue' if label == 1 else 'red'
    plt.scatter(point[0], point[1], color=color, s=100)
    plt.text(point[0] + 0.02, point[1], f"{label}", fontsize=12)

# 결정 경계: x1 + x2 = 0.5
x_vals = np.linspace(-0.1, 1.1, 100)
y_vals = 0.5 - x_vals
plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary: x1 + x2 = 0.5')

# 그래프 설정
plt.title("Linear Separability of OR Gate")
plt.xlabel("x1")
plt.ylabel("x2")
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.grid(True)
plt.legend()
plt.gca().set_aspect('equal')
plt.show()

```

이 문제는 직선 하나로 클래스(0과 1)를 구분할 수 있기 때문에 **단일 퍼셉트론으로 해결 가능**하다.

---
###  🔹XOR 문제

**문제 정의**  
XOR 게이트는 두 입력이 서로 다를 때만 출력이 1이 되는 논리 연산이다.

| $x_1$​ | $x_2$​ | $y=x_1 \; ⊕ \; x_2$ |
| ------ | ------ | ------------------- |
| 0      | 0      | 0                   |
| 0      | 1      | 1                   |
| 1      | 0      | 1                   |
| 1      | 1      | 0                   |

**한계**  
이 문제는 하나의 직선으로는 분리가 불가능하다. 즉, **선형 분리가 되지 않기 때문에 단일 퍼셉트론으로는 해결 불가능**하다.

**시각적 설명**  
XOR 문제의 포인트를 평면상에 그려보면 다음과 같이 배치된다:

- (0,0) → 0
- (0,1) → 1
- (1,0) → 1
- (1,1) → 0

이 4개의 점은 어떤 직선으로도 두 클래스를 정확히 분리할 수 없다.

---
## 2️⃣ 다층 퍼셉트론(MLP)의 구조

### 🔹 왜 은닉층(Hidden Layer)이 필요한가?

단일 퍼셉트론은 직선 하나로 데이터를 구분하는 **선형 분류기**이므로, XOR 같은 **비선형 문제**를 해결할 수 없다. 이 한계를 극복하기 위해, 입력과 출력 사이에 **은닉층(Hidden Layer)** 을 추가한다. 은닉층은 입력을 변환하여 **더 복잡한 함수 형태**를 모델링할 수 있도록 도와준다.

---
### 🔹 MLP의 기본 구조

다층 퍼셉트론(Multi-Layer Perceptron)은 다음과 같은 구조를 갖는다.

- **입력층(Input Layer)** : 외부에서 데이터를 받아들이는 층
- **은닉층(Hidden Layer)** : 비선형성을 학습하는 중간 층 (1개 이상 가능)
- **출력층(Output Layer)** : 최종 예측 결과를 내는 층

구조적으로는 다음과 같다:
입력층 → 은닉층(1) → 은닉층(2) → ... → 출력층

<img width="846" height="653" alt="image" src="https://github.com/user-attachments/assets/8d598bbf-fe96-460f-813a-db95d47cc428" />


---
### 🔹 노드(Node) 또는 유닛(Unit)

각 층은 여러 개의 **노드(node)** 또는 **유닛(unit)** 으로 구성된다.  

- $n_1$ 과 $n_2$의 값은 각각 단일 퍼셉트론의 값과 같다.

$$
n_1 = \sigma(x_1 \cdot w_{11} + x_2 \cdot w_{21} + b_1)
$$
$$
n_2 = \sigma(x_1 \cdot w_{12} + x_2 \cdot w_{22} + b_2)
$$
- 위 두 식의 결과값이 출력층으로 보내진다.
- 출력층에서는 시그모이드 함수를 통해 y값이 정해진다.
$$
y_{out} = \sigma(n_1 \cdot w_{31} \text{ }+\text{ }n_2 \cdot w_{32} + b_3 )
$$
---
### 🔹 가중치와 바이어스

- 위에서 가중치($W$)와 바이어스($b$) 값은 배열로 표시 가능
- 은닉층을 포함해 가중치 6개와 바이어스 3개가 필요함

$$
W_{(1)} = 
\begin{bmatrix}
w_{11} & w_{12} \\
w_{21} & w_{22}
\end{bmatrix}
\quad
W_{(2)} = 
\begin{bmatrix}
w_{31} \\
w_{32}
\end{bmatrix}
$$
$$
B_{(1)} = 
\begin{bmatrix}
b_{1} \\
b_{2}
\end{bmatrix}
\quad
B_{(2)} = 
\begin{bmatrix}
b_{3}
\end{bmatrix}
$$

---
## 🔹XOR 문제 풀이

**사전 지식 : XOR 게이트는 NAND 게이트와 OR 게이트의 AND 연산으로 구현 가능하다.**
- NAND 게이트의 진리표

| $x_1$​ | $x_2$​ | $y = (x_1 \cdot x_2)'$ |
| ------ | ------ | ---------------------- |
| 0      | 0      | 1                      |
| 0      | 1      | 1                      |
| 1      | 0      | 1                      |
| 1      | 1      | 0                      |
- OR 게이트의 진리표

| $x_1$​ | $x_2$​ | $y=x_1 \; + \; x_2$ |
| ------ | ------ | ------------------- |
| 0      | 0      | 0                   |
| 0      | 1      | 1                   |
| 1      | 0      | 1                   |
| 1      | 1      | 1                   |
- NAND 게이트와 OR 게이트의 AND 연산 결과

| $x_1$​ | $x_2$​ | $y=(x_1 \cdot x_2)' \cdot (x_1 \; + \; x_2) = x_1 \; ⊕ \; x_2$ |
| ------ | ------ | -------------------------------------------------------------- |
| 0      | 0      | 0                                                              |
| 0      | 1      | 1                                                              |
| 1      | 0      | 1                                                              |
| 1      | 1      | 0                                                              |

$W_1, W_2, B_1, B_2$ 값을 아래와 같이 설정하고, $n_1, n_2, y_{out}$ 을 각각 계산해보자

$$
W_{(1)} = 
\begin{bmatrix}
-2 & 2 \\
-2 & 2
\end{bmatrix}
\quad
W_{(2)} = 
\begin{bmatrix}
1 \\
1
\end{bmatrix}
$$
$$
B_{(1)} = 
\begin{bmatrix}
3 \\
-1
\end{bmatrix}
\quad
B_{(2)} = 
\begin{bmatrix}
-1
\end{bmatrix}
$$

그림으로 표현하면

<img width="840" height="652" alt="image" src="https://github.com/user-attachments/assets/81f626e3-0649-407e-89ef-4bdddd84959b" />


이제 각각 $x_1$의 값과 $x_2$의 값을 입력해 $y$값이 우리가 원하는 값으로 나오는지 점검해보자. 

| $x_1$ | $x_2$ | $n_1$ 계산식                                       | $n_2$ 계산식                                     | $y_{\text{out}}$ 계산식                          |  값  |
| :---: | :---: | :---------------------------------------------- | :-------------------------------------------- | :-------------------------------------------- | :-: |
|   0   |   0   | $\sigma(0 \cdot -2 + 0 \cdot -2 + 3) \approx 1$ | $\sigma(0 \cdot 2 + 0 \cdot 2 - 1) \approx 0$ | $\sigma(1 \cdot 1 + 0 \cdot 1 - 1) \approx 0$ |  0  |
|   0   |   1   | $\sigma(0 \cdot -2 + 1 \cdot -2 + 3) \approx 1$ | $\sigma(0 \cdot 2 + 1 \cdot 2 - 1) \approx 1$ | $\sigma(1 \cdot 1 + 1 \cdot 1 - 1) \approx 1$ |  1  |
|   1   |   0   | $\sigma(1 \cdot -2 + 0 \cdot -2 + 3) \approx 1$ | $\sigma(1 \cdot 2 + 0 \cdot 2 - 1) \approx 1$ | $\sigma(1 \cdot 1 + 1 \cdot 1 - 1) \approx 1$ |  1  |
|   1   |   1   | $\sigma(1 \cdot -2 + 1 \cdot -2 + 3) \approx 0$ | $\sigma(1 \cdot 2 + 1 \cdot 2 - 1) \approx 1$ | $\sigma(0 \cdot 1 + 1 \cdot 1 - 1) \approx 0$ |  0  |

아래 python 예제를 통해 결과를 확인해보자.

```python
import numpy as np

# 가중치 및 바이어스 설정
w11 = np.array([-2, -2])
w12 = np.array([2, 2])
w2 = np.array([1, 1])

b1 = 3
b2 = -1
b3 = -1

# 단일 퍼셉트론 함수
def MLP(x, w, b):
    y = np.sum(w * x) + b
    if y <= 0:
        return 0
    else:
        return 1

# 게이트 구현
def NAND(x1, x2):
    return MLP(np.array([x1, x2]), w11, b1)

def OR(x1, x2):
    return MLP(np.array([x1, x2]), w12, b2)

def AND(x1, x2):
    return MLP(np.array([x1, x2]), w2, b3)

def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))

# 테스트
if __name__ == '__main__':
    for x in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = XOR(x[0], x[1])
        print("입력 값:", x, "출력 값:", y)

```


---
### 🔹 Fully Connected Layer (Dense Layer)

MLP의 각 층은 바로 이전 층의 모든 노드와 연결되어 있다.  
이런 구조를 **완전 연결층(Fully Connected Layer)** 또는 **Dense Layer** 라고 한다.

즉, **각 노드는 이전 층의 모든 노드로부터 입력을 받는다.**  
이는 다음 층이 충분한 정보를 받아 **복잡한 패턴**을 학습할 수 있도록 한다.

---
## 3️⃣ 활성화 함수 (Activation Function)

### 🔹 왜 활성화 함수가 필요한가?

다층 퍼셉트론에서 각 노드는 선형 계산  
($z = w \cdot x + b$)을 수행한 뒤, **비선형 함수**를 적용( $\sigma(z)$ )하여 출력을 만든다.

선형 함수의 정의
- $f(x+y) = f(x) + f(y)$
- $f(ax) = af(x)$

선형 함수만 사용하면, 몇 층을 쌓아도 전체 모델은 결국 하나의 선형 함수로 축약된다.  
즉, 다층 구조의 의미가 없어지며 **비선형 문제를 해결할 수 없다**.

**왜 비선형 함수인가?**

- 선형 함수로 층을 쌓는 경우
	- $f(x) = Cx \quad (C \in \mathbb{R})$
	- $f(f(x)) = f(Cx) = C^2 x$
	- $C \in \mathbb{R} \Rightarrow C^2 \in \mathbb{R}$
	- $f(f(x)) = f(x)$ 
	- $f(x) = C^2 x$

- 비선형 함수로 층을 쌓는 경우

$$
\sigma(x) = \frac{e^x}{e^x + 1}
$$
$$
\sigma(\sigma(x)) = \sigma\left( \frac{e^x}{e^x + 1} \right)
$$
$$
= \frac{e^{\frac{e^x}{e^x + 1}}}{e^{\frac{e^x}{e^x + 1}} + 1}
$$
$$
\sigma(x) \ne \sigma(\sigma(x))
$$


이를 해결하기 위해 **활성화 함수(activation function)** 를 도입한다.  
활성화 함수는 모델에 **비선형성**을 추가하여 **복잡한 함수 형태**를 학습할 수 있도록 해준다.

---
### 🔹 대표적인 활성화 함수
---
#### 1. 시그모이드(Sigmoid) 함수

<img width="633" height="475" alt="image" src="https://github.com/user-attachments/assets/cd8b06f5-9d95-42a6-9b74-e2feff0f2247" />

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$
- 출력 범위가 $0$ ~ $1$사이의 값
- 이진 분류 문제의 출력층에서 자주 사용
- 확률처럼 해석 가능

**장점**:
- 부드러운 곡선, 출력값을 확률로 해석 가능

**단점**:
- 입력값이 커지면 기울기(미분값)가 0에 수렴 → **기울기 소멸(vanishing gradient) 문제가 발생**
- 출력이 0 중심이 아님 → 학습이 느려짐

---
#### 2. Tanh (Hyperbolic Tangent) 함수

<img width="650" height="471" alt="image" src="https://github.com/user-attachments/assets/5aaeeba9-a28c-4187-afe4-57df1a1bccad" />

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$
- 시그모이드와 비슷하게 생겼지만 출력 범위가 $-1$ ~ $1$ 사이의 값
- 시그모이드를 두 배 해주고 -1만큼 평행이동한 값과 같음

$$
tanh(x)=2\sigma(2x)-1
$$

**장점**:
- Sigmoid보다 학습에 유리함 (0 중심)
- 자연스러운 스케일링

**단점**:
- 여전히 큰 입력값에서 **기울기 소실(vanishing gradient)** 발생 가능

---
#### 3. ReLU (Rectified Linear Unit)

<img width="843" height="465" alt="image" src="https://github.com/user-attachments/assets/47f5c4f8-9f1e-4338-8a86-dd96ab83a183" />

$$
\text{ReLU}(z) = \max(0, z)
$$
- 출력 범위가 $0$ ~ $\infty$ 의 값
- 입력이 0을 넘으면 그 입력을 그대로 출력하고, 0 이하면 0을 출력하는 함수
- 은닉층에서 가장 널리 사용되는 함수

**장점**:
- 계산이 단순하여 매우 빠름
- 양의 영역에서는 기울기(gradient) 유지 → 빠른 수렴

**단점**:
- 음의 입력에서는 기울기(gradient)가 0이 되어 **죽은 뉴런 문제(dead neuron)** 발생 가능
---
## 4️⃣ 순전파 (Forward Propagation)

### 🔹 순전파란?

순전파는 신경망에서 **입력 → 은닉층 → 출력층**으로 정보가 흐르며  
각 노드가 계산을 수행해 나가는 과정을 말한다.

각 층의 노드는  
1. 이전 층의 출력을 받아  
2. 가중치 곱과 바이어스를 더한 뒤  
3. 활성화 함수를 통과시켜  
다음 층으로 전달한다.

---
### 🔹 연산 방식

각 노드는 아래 수식을 따르며 계산된다:

$$
z = w \cdot x + b
$$

$$
a = \sigma(z)
$$

여기서  
- $x$ : 입력 벡터 (또는 이전 층의 출력)  
- $w$ : 가중치 벡터  
- $b$ : 바이어스  
- $z$ : 선형 결합 결과 (가중합)  
- $\sigma$ : 활성화 함수 (ReLU, Sigmoid 등)  
- $a$ : 다음 층으로 전달되는 **활성화 값 (activation output)**  
	- 이 값이 다시 다음 층의 입력이 된다.

---
### 🔹 층 간 계산 흐름

예를 들어, 2층 신경망(입력 → 은닉층 → 출력층)에서의 순전파 계산은 다음과 같다:

- 은닉층 계산
  
$$
z^{(1)} = W^{(1)} x + b^{(1)}
$$

$$
a^{(1)} = \sigma(z^{(1)})
$$

- 출력층 계산
  
$$
z^{(2)} = W^{(2)} a^{(1)} + b^{(2)}
$$

$$
a^{(2)} = \sigma(z^{(2)})
$$

여기서  
- $W^{(1)}, b^{(1)}$ : 입력 → 은닉층 가중치와 바이어스  
- $W^{(2)}, b^{(2)}$ : 은닉층 → 출력층 가중치와 바이어스  
- $a^{(2)}$ : 최종 출력값 (예측값)
---
## 5️⃣ 손실 함수 (Loss Function)

### 🔹 손실 함수란?

손실 함수는 **모델의 예측값과 실제값의 차이(오차)** 를 수치로 표현한 함수이다.

모델이 얼마나 잘못 예측했는지를 수치화하여 학습에 사용하며,  
이 값이 작을수록 모델의 예측 성능이 높다는 것을 의미한다.

즉, 손실 함수는 **모델의 학습 방향을 결정짓는 기준**이다.

---
### 🔹 주요 손실 함수 예시
---
#### 1. 평균 제곱 오차 (MSE, Mean Squared Error)

회귀 문제에서 자주 사용되는 손실 함수이다.

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

- $y_i$ : 실제값  
- $\hat{y}_i$ : 예측값  
- $N$ : 전체 데이터 개수

**특징**:
- 오차가 클수록 더 큰 패널티를 줌 (제곱)
- 이상치에 민감함

---
#### 2. 이진 크로스 엔트로피 (Binary Cross Entropy)

이진 분류 문제에서 사용되는 손실 함수로, 확률 예측에 적합하다.

$$
\text{BCE} = -\frac{1}{n}\sum_{i=1}^{n}\left[ y_i \log(\hat{y_i}) + (1 - y_i) \log(1 - \hat{y_i}) \right]
$$

- $y_i$ : 실제값 (0 또는 1)  
- $\hat{y}_i$ : 예측 확률 (0~1 사이 값)

**특징**:
- 예측이 실제값과 가까울수록 손실이 작아짐
- 확률 예측이 중요할 때 적합

---
#### 3. 범주형 크로스 엔트로피 (Categorical Cross Entropy)

다중 클래스 분류에서 사용되는 손실 함수이다.

$$
\text{CCE} = -\sum_{i=1}^{n} y_i \log(\hat{p}_i)
$$

- $K$ : 클래스 개수  
- $y_k$ : 정답인 클래스에 1, 나머지는 0 (One-hot encoding)  
- $\hat{p}_k$ : 예측 확률 (Softmax 결과)

**특징**:
- 소프트맥스와 함께 사용됨
- 정답 클래스의 예측 확률이 높을수록 손실 감소

---
## 6️⃣ 역전파 (Backpropagation)

### 🔹 손실을 바탕으로 가중치를 어떻게 조정하는가?

신경망 학습의 핵심은 예측값과 실제값의 오차(손실)를 줄이는 것이다.  
이를 위해, **손실 함수의 값이 작아지는 방향으로 가중치와 바이어스를 조정**해야 한다.

이 조정은 **기울기(gradient)** 를 계산하여 수행되며,  
이때 사용되는 알고리즘이 바로 **역전파 알고리즘**이다.

---
### 🔹 체인 룰(Chain Rule)을 통한 미분

신경망은 여러 층으로 구성되며, 각 층은 함수의 합성 구조를 가진다.

예를 들어, 순전파에서 다음과 같은 계산이 있다고 하자:

$$
z = w \cdot x + b
$$

$$
a = \sigma(z)
$$

$$
L = \text{Loss}(a, y)
$$

이때, 손실 $L$을 가중치 $w$에 대해 미분하기 위해선  
**체인 룰**을 사용하여 다음과 같이 계산한다:

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

각 항은 다음 의미를 갖는다:

- $\frac{\partial L}{\partial a}$: 예측값이 손실에 얼마나 영향을 주는지  
- $\frac{\partial a}{\partial z}$: 활성화 함수의 기울기  
- $\frac{\partial z}{\partial w}$: 가중치가 선형합 $z$에 미치는 영향

---
### 🔹 오차가 각 층을 따라 역방향으로 전파

출력층에서 계산된 오차는 **이전 층으로 역방향 전파**되며,  
각 층의 가중치에 대한 손실 기울기를 계산한다.

예를 들어, 은닉층과 출력층이 하나씩 있는 2층 구조에서는 다음과 같은 흐름이 된다:

1. 출력층 오차 계산:

$$
\delta^{(2)} = \frac{\partial L}{\partial a^{(2)}} \cdot \sigma'(z^{(2)})
$$

2. 은닉층 오차 전파:

$$
\delta^{(1)} = (W^{(2)})^T \delta^{(2)} \cdot \sigma'(z^{(1)})
$$

3. 각 층의 가중치에 대한 기울기 계산:

$$
\frac{\partial L}{\partial W^{(2)}} = \delta^{(2)} (a^{(1)})^T
$$
$$
\frac{\partial L}{\partial W^{(1)}} = \delta^{(1)} (x)^T
$$

---
### 🔹 순전파, 역전파 예제 코드

```python
import numpy as np

# 시그모이드 함수 및 그 도함수
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# 입력과 실제 정답
x = np.array([[0.5], [0.8]])   # 입력 (2x1)
y = np.array([[1]])            # 실제값 (스칼라 -> 1x1 벡터)

# 초기 가중치와 바이어스 설정 (2-2-1 구조)
W1 = np.array([[0.1, 0.4],
               [0.2, 0.3]])     # 은닉층 가중치 (2x2)
b1 = np.array([[0.1], [0.1]])   # 은닉층 바이어스 (2x1)

W2 = np.array([[0.3, 0.6]])     # 출력층 가중치 (1x2)
b2 = np.array([[0.05]])         # 출력층 바이어스 (1x1)

# === 순전파 ===
z1 = np.dot(W1, x) + b1
a1 = sigmoid(z1)

z2 = np.dot(W2, a1) + b2
a2 = sigmoid(z2)  # 예측값

print("예측값 (a2):", a2)

# === 역전파 ===
# 출력층 오차 (BCE + sigmoid 가정)
delta2 = a2 - y
dW2 = np.dot(delta2, a1.T)
db2 = delta2

# 은닉층 오차
delta1 = np.dot(W2.T, delta2) * sigmoid_derivative(z1)
dW1 = np.dot(delta1, x.T)
db1 = delta1

print("\n[출력층]")
print("오차 delta2:", delta2)
print("가중치 변화량 dW2:", dW2)
print("바이어스 변화량 db2:", db2)

print("\n[은닉층]")
print("오차 delta1:", delta1)
print("가중치 변화량 dW1:", dW1)
print("바이어스 변화량 db1:", db1)
```

역전파를 진행하여 파라미터를 업데이트 한 이후, 다시 순전파를 진행시켜보고 결과를 비교하라

```python
# === 파라미터 업데이트 (경사 하강법 적용) ===
lr = 0.1  # 학습률

W2_new = W2 - lr * dW2
b2_new = b2 - lr * db2

W1_new = W1 - lr * dW1
b1_new = b1 - lr * db1

# === 업데이트된 파라미터로 순전파 다시 수행 ===
z1_new = np.dot(W1_new, x) + b1_new
a1_new = sigmoid(z1_new)

z2_new = np.dot(W2_new, a1_new) + b2_new
a2_new = sigmoid(z2_new)

print("\n[업데이트 후 순전파 결과]")
print("최종 출력값 (a2):", a2_new)
```


## 순전파 및 역전파 결과 정리

### 1. 순전파 결과

- 입력값: `x = [0.5, 0.8]`
- 예측값 (`a2`) ≈ `0.6456`

- 모델은 "정답이 1일 확률이 약 64.6%"라고 예측하였다.
- 실제값은 1이므로, 예측이 부족한 상태이다.
- 이 예측 오차를 바탕으로 모델의 가중치를 조정할 필요가 있다.

---
### 2. 출력층 역전파 결과

- 출력층 오차 (`delta2`) ≈ `-0.3544`  
- 출력층 가중치 변화량 (`dW2`) ≈ `[-0.2181, -0.2156]`  
- 출력층 바이어스 변화량 (`db2`) ≈ `-0.3544`

- 예측값이 정답보다 낮았기 때문에, 가중치를 **증가시키는 방향**으로 조정해야 한다.
- 가중치 변화량은 은닉층의 출력값에 따라 다르게 계산된다.
- 바이어스도 오차만큼 함께 업데이트된다.

---
### 3. 은닉층 역전파 결과

- 은닉층 오차 (`delta1`) ≈ `[-0.0252, -0.0507]`
- 은닉층 가중치 변화량 (`dW1`):  `[[-0.0126, -0.0201], [-0.0253, -0.0405]]`
- 은닉층 바이어스 변화량 (`db1`):  `[[-0.0252], [-0.0507]]`

- 출력층의 오차가 역전파되어 은닉층 각 뉴런에 얼마나 책임이 있는지 계산된다.
- 이 오차는 은닉층 뉴런 각각의 활성화 함수의 기울기(도함수)에 영향을 받는다.
- 은닉층의 각 가중치는 입력값 x에 따라 얼마나 조정해야 하는지를 계산한 것이다.

---
### 4. 전체 요약

- 순전파는 입력을 통해 예측값을 구하는 과정이며, 역전파는 오차를 바탕으로 각 가중치가 얼마나 영향을 미쳤는지 계산하여 업데이트 방향을 정하는 과정이다.
- 출력층부터 은닉층 방향으로 오차가 역전파되며, 각 층의 가중치와 바이어스를 **기울기(gradient)** 를 통해 조정한다.
- 이 과정을 반복하면서 모델은 예측 정확도를 점점 향상시킨다.

---
## 7️⃣ 경사 하강법 (Gradient Descent)

### 🔹 손실을 줄이기 위한 가중치 업데이트

<img width="609" height="464" alt="image" src="https://github.com/user-attachments/assets/2ea59ccd-1542-46df-994b-b66e92946bdf" />


신경망 학습의 목표는 손실 함수 $L$을 최소화하는 가중치 $w$를 찾는 것이다.  
이를 위해 사용하는 최적화 알고리즘이 **경사 하강법(Gradient Descent)** 이다.

<img width="838" height="267" alt="image" src="https://github.com/user-attachments/assets/6bbbc96a-cdf2-414d-a9ee-2df1e4e5b54c" />


경사 하강법은 **손실 함수의 기울기(gradient)** 를 계산하여 
손실이 작아지는 방향으로 가중치를 조금씩 이동시키는 방식이다.

---
### 🔹 가중치 업데이트 공식

$$
w = w - \eta \cdot \frac{\partial L}{\partial w}
$$

여기서  
- $w$ : 현재 가중치  
- $\eta$ : 학습률(learning rate)  
- $\frac{\partial L}{\partial w}$ : 손실 함수 $L$에 대한 가중치 $w$의 기울기

---
### 🔹 학습률 (Learning Rate)

**학습률 $\eta$** 는 한 번의 업데이트에서 **얼마나 이동할 것인지**를 결정하는 하이퍼파라미터다.

- 너무 작으면: 학습 속도가 느림  
- 너무 크면: 최솟값을 지나쳐 발산할 수 있음

따라서 $\eta$는 모델 성능에 큰 영향을 주므로 **적절한 설정이 중요**하다.

---
### 🔹 경사 하강법의 세 가지 변형

경사 하강법은 **기울기를 계산하는 데이터 범위**에 따라 다음 세 가지 방식으로 나뉜다:

---
#### 1. 배치 경사 하강법 (Batch Gradient Descent)

- 전체 데이터셋을 한 번에 사용해 기울기 계산
  
$$
\nabla L = \frac{1}{N} \sum_{i=1}^{N} \nabla L^{(i)}
$$

- **장점**: 안정적인 수렴  
- **단점**: 데이터가 많을 경우 느리고 메모리 사용이 큼

---
#### 2. 확률적 경사 하강법 (SGD, Stochastic Gradient Descent)

- 데이터 한 개를 기준으로 기울기 계산 및 즉시 업데이트

$$
w := w - \eta \cdot \nabla L^{(i)}
$$

- **장점**: 계산 빠름, 자주 업데이트됨  
- **단점**: 손실 함수가 불안정하게 출렁일 수 있음

---
#### 3. 미니배치 경사 하강법 (Mini-Batch Gradient Descent)

- 여러 개의 샘플을 묶은 **미니배치** 단위로 학습

$$
\nabla L = \frac{1}{B} \sum_{i=1}^{B} \nabla L^{(i)}
$$

- **장점**: 속도와 안정성의 균형  
- **현대 딥러닝의 표준 방식**

---
## 8️⃣ 과적합과 정규화

### 🔹 과적합(Overfitting)과 과소적합(Underfitting)

모델이 학습 데이터에 대해 너무 잘 맞춰져서 **일반화 성능이 떨어지는 현상**을  
**과적합(Overfitting)** 이라고 한다.

반대로 모델이 학습 데이터조차 제대로 학습하지 못해  
**훈련 성능도 낮은 경우**는 **과소적합(Underfitting)** 이라고 한다.

| 현상         | 특징                                                  |
|--------------|--------------------------------------------------------|
| 과적합       | 훈련 정확도 ↑, 검증 정확도 ↓ → **너무 외운 상태**      |
| 과소적합     | 훈련 정확도 ↓, 검증 정확도 ↓ → **모델이 단순한 상태**  |

---
### 🔹 정규화 (Regularization)

정규화는 **모델이 과도하게 복잡해지는 것을 막고**,  
**일반화 성능을 높이기 위한 기법**이다.

---
#### 1. L2 정규화 (L2 Regularization, 가중치 감쇠)

손실 함수에 가중치 제곱합을 추가하여, **너무 큰 가중치가 발생하지 않도록 제한**한다.

정규화된 손실 함수:

$$
L_{\text{reg}} = L_{\text{original}} + \lambda \sum_i w_i^2
$$

- $\lambda$: 정규화 강도 (0이면 정규화 없음)
- $w_i$: 각 가중치 파라미터

**효과**: 큰 가중치를 억제하여 **모델 복잡도 감소**, 과적합 완화

---
#### 2. 드롭아웃 (Dropout)

학습 중 무작위로 일부 노드를 제거하여, **특정 노드에 의존하지 않도록 강제**하는 기법

- 예: 학습 시 은닉층 노드 중 30%를 무작위로 꺼버림  
- 매 스텝마다 꺼지는 노드가 달라짐 → **앙상블 효과** 유사

**효과**: 신경망이 **더 강건해지고**, 과적합 위험이 줄어듦

---
## 9️⃣ MLP의 한계와 CNN으로의 전환

### 🔹 다층 퍼셉트론(MLP)의 한계

MLP는 이론적으로 어떤 함수든 근사할 수 있지만,  
현실적으로는 다음과 같은 **구조적, 계산적 한계**를 가진다:

---
#### 1. 지역 최적값(Local Minima)에 빠짐

- 손실 함수가 **비선형**이고 고차원이기 때문에  
  **최적이 아닌 지점에 수렴**할 수 있음
- 특히 역전파 기반 학습에서 잘못된 초기값 또는 학습률로 인해  
  **전역 최적값을 찾지 못함**

---
#### 2. 계산량 급증

- 모든 노드가 이전 층의 모든 노드와 연결되는  
  **Fully Connected 구조**는  
  **매우 많은 파라미터와 연산량**을 요구함

예를 들어, 입력 이미지가 $28 \times 28 = 784$차원이고  
은닉층에 100개의 노드가 있다면,  
가중치만 $784 \times 100 = 78,\!400$개 발생

---
#### 3. 입력의 순서나 공간 구조를 반영하지 못함

MLP는 **입력 벡터를 단순히 1차원 배열로 처리**하므로,  
다음과 같은 정보를 **무시**하게 된다:

- 이미지의 **공간적 위치 관계 (locality)**  
- 시계열의 **순서 정보 (temporal dependency)**

즉, **픽셀 간의 관계**나 **인접한 특성 간의 의미**를 반영할 수 없다.

---
### 🔹 이미지 처리에는 부적합

이미지, 음성, 자연어 등 **구조적 특성이 중요한 데이터**에 대해  
MLP는 입력 차원이 커지면 **비효율적이고 일반화가 어려움**

---
### 🔹 CNN(합성곱 신경망)으로의 발전

이러한 한계를 이유로 **합성곱 신경망(CNN, Convolutional Neural Network)** 이 생겨났다.

CNN의 특징
- **지역 receptive field**: 입력의 국소 영역만 처리  
- **가중치 공유**: 동일한 필터를 전체 입력에 적용  
- **풀링 계층**: 공간 크기를 줄이며 요약

이를 통해 CNN은  
- 파라미터 수를 줄이고  
- 공간 구조를 보존하며  
- 이미지나 시계열과 같은 데이터에 적합한 구조를 제공함

---
# 🔹과제 1) 활성 함수의 종류를 설명한 것 (시그모이드, tanh, ReLU) 외에 아래 보기 중 선택하여 두 가지를 조사해오기

조사 내용으로는 식, 함수 모양, 그 활성함수의 장단점 각각 2가지 이상

- GELU(Gaussian Error Linear Unit)
- Softmax
- SoftPlus
- Leakt rectified linear unit(Leaky ReLU)
- Gaussian


# 🔹과제 2) 배치 경사하강법과 미니배치 경사하강법의 알고리즘에 대해 조사

간단하게 알고리즘의 단계와 해당 단계에서의 설명을 간단하게 작성
