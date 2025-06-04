# 📘 Week3-2 - 손실함수, 역전파

---

## 📉 손실 함수 (Loss Function)

### 🔹 개념
- 모델의 예측값과 실제값의 차이를 수치로 표현한 것
- 손실(loss)을 최소화하는 방향으로 모델이 학습됨

### 🔹 주요 손실 함수
| 함수명 | 설명 | 사용 예 |
|--------|------|--------|
| `MSELoss` | 평균 제곱 오차 | 회귀 문제 |
| `CrossEntropyLoss` | 다중 클래스 분류 | 분류 문제 |

### 예시
```python
import torch
import torch.nn as nn

# 예측값과 실제값
pred = torch.tensor([0.8, 0.1, 0.1])
target = torch.tensor([0])

# 손실 함수 정의
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(pred.unsqueeze(0), target)
print(loss.item())
```

---

## 🔁 역전파 (Backpropagation)

### 🔹 개념
- 손실 함수의 값이 줄어들도록 파라미터(가중치, 편향)를 업데이트하는 알고리즘
- 체인 룰(Chain Rule)을 통해 각 파라미터에 대한 미분값 계산

### 🔹 PyTorch에서의 역전파 흐름
```python
loss.backward()      # 손실에 대한 역전파 수행
optimizer.step()     # 파라미터 업데이트
optimizer.zero_grad()  # 기울기 초기화
```

---

## 🧪 실습 예시: 간단한 학습
```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])

for epoch in range(100):
    pred = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model.weight.item(), model.bias.item())
```

---

## 🧪 과제
1. `nn.Linear(1,1)`을 사용하여 y=3x에 근사하는 모델을 학습하시오.
2. 손실 함수로 `nn.MSELoss()`를 사용하고, 에폭마다 손실 값을 출력하시오.
3. 학습이 끝난 후 weight와 bias 값을 출력하시오.

---

✅ 사용 환경: Python 3.x, Google Colab, PyTorch 설치 필요
