# 📘 Week3‑2 – 다층 퍼셉트론 (MLP)

---

## 🔍 기본 개념 *(Multi-Layer Perceptron)*

### ▶️ 퍼셉트론(Perceptron)
- 단일 입력값과 가중치의 곱이 임계값을 넘으면 1, 아니면 0 출력 (이진 분류)
- 단일 층 신경망 구조

### ▶️ 다층 퍼셉트론 (MLP)
- 입력층(Input layer) → 은닉층(Hidden Layer) → 출력층(Output layer)
- 한 개 이상의 은닉층이 존재하는 신경망 (비선형 패턴 학습 가능)

### ▶️ PyTorch 구조 예시
- 한 Layer는 Linear(선형) 연산 + 비선형 활성화 함수 조합

```python
from torch import nn
model = nn.Sequential(
    nn.Linear(2, 4),   # input: 2 features → hidden: 4 units
    nn.ReLU(),
    nn.Linear(4, 1),   # hidden: 4 units → output: 1
)
```

---

## 🔢 신경망 Forward 연산

- 각 Layer에서 입력값과 가중치의 선형 결합 결과가 비선형 함수(ReLU, Sigmoid 등)를 거쳐 다음 Layer로 전달됨
- 여러 층을 쌓으면 더 복잡한 함수/공간 구조도 근사 가능

---

## 🔬 [실습] 넘파이 vs 파이토치로 2단 MLP 순전파(Forward) 비교

### 1️⃣ 넘파이로 구현

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 임의 파라미터/입력
W1 = np.array([[0.1, 0.2], [0.3, 0.4]])
b1 = np.array([0.1, 0.2])
W2 = np.array([[0.5], [0.6]])
b2 = np.array([0.3])

x = np.array([1.0, 0.5])

# Forward 연산
z1 = np.dot(x, W1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, W2) + b2
a2 = sigmoid(z2)

print("NumPy 최종 출력:", a2)
```

---

### 2️⃣ 파이토치로 구현

```python
import torch
import torch.nn as nn

# 동일 파라미터/입력
W1 = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
b1 = torch.tensor([0.1, 0.2], dtype=torch.float32)
W2 = torch.tensor([[0.5], [0.6]], dtype=torch.float32)
b2 = torch.tensor([0.3], dtype=torch.float32)
x = torch.tensor([1.0, 0.5], dtype=torch.float32)

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# Forward 연산
z1 = torch.matmul(x, W1) + b1
a1 = sigmoid(z1)
z2 = torch.matmul(a1, W2) + b2
a2 = sigmoid(z2)

print("PyTorch 최종 출력:", a2.item())
```

---

### ✅ [실험 결과]
- 같은 파라미터, 같은 연산식이면 **NumPy와 PyTorch 모두 순전파(Forward) 결과가 동일**함을 직접 확인  
- 신경망 구조가 같으면 프레임워크에 상관없이 연산 결과도 같다

---

## 프로젝트: XOR 문제 학습

```python
import torch
import torch.nn as nn
import torch.optim as optim

x = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([[0.],[1.],[1.],[0.]])

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(10000):
    y_pred = model(x)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d}: loss = {loss.item():.4f}")

print(model(x).detach())
```

---

## 🔹 결과 보고
- XOR 문제는 단일 퍼셉트론으로는 구현 불가능
- 다층 구조를 적용하면 구현 가능

---

## 🔧 과제

1. 위 실습 코드를 실행하고 파라미터(W, b)와 입력값을 바꿔가며 결과 비교  
2. 시그모이드 대신 다른 활성화 함수(ReLU 등)로 실험  
3. 3단 이상 MLP 구조로 확장  
4. (심화) 4주차에서 역전파(gradient)까지 파이토치/넘파이로 비교해보기  
5. XOR 문제 프로젝트 작성 후, loss가 0.01 이하가 되도록 학습, 출력값과 파라미터 기록

---

✅ **필요 환경**: Python 3.x, Google Colab, numpy, torch >= 2.0  
설치: `!pip install numpy torch torchvision -q`
