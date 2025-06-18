# 📘 Week3‑2 – 다층 퍼셉트론 (MLP)

---

## 🔍 기본 개념 *(Multi-Layer Perceptron)*

### ▶️ 개인 퍼셉트론 (Perceptron)
- 초계 단위 입력값과 값과 가운스의 결과가 중간값과 비교되거나 아니면 조정치 중 하나가 출력.
- 한가지 단위의 가령자가 결정한 반응을 일으키는 구조.

### ▶️ 다층 퍼셉트론 (MLP)
- 한가지 가장 다른 점: **포터치 필드(hidden layer)**
- 입력계(Input layer) → 하드치 (Hidden Layer) → 출력계(Output layer)

### ▶️ 프로토치 구조
- 한 Layer는 단위 Linear 또는 Affine 컴블리고 ReLU가 따라온 그룹

```python
from torch import nn
model = nn.Sequential(
    nn.Linear(2, 4),   # input: 2 features → hidden: 4 units
    nn.ReLU(),
    nn.Linear(4, 1),   # hidden: 4 units → output: 1
)
```

---

## 🔢 값 할인 & 유리

- 가운스가 모두 Linear 지역의 선적 조합이면, 사이간적 더 다행과 공간 구조에 대해 가령 가능
- 각 Layer에서 입력값과 가운스의 결과가 지보되고, 이것이 다음 Layer에 입력으로 전달

---

## 프로젝트: XOR 문제 그림

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
- XOR 문제는 일반 퍼셉트론으로는 구현 무가능
- 다층 구조를 적용하면 구현 가능

---

## 🔧 과제

1. XOR 문제 프로젝트 작성 후
2. 보고가 0.01 이하로 내려갈 때 까지 향상
3. 출력 값과 정보 보고

---

✅ **필요 환경**: Python 3.x, Google Colab, torch >= 2.0  
> 필수 설치: `!pip install torch torchvision -q`
