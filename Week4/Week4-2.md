# 📘 Week4-2 – 손실함수 & 역전파

---

## 🔍 “값의 차”를 말하는 것: Loss Function

### ▶️ 기호 정보
| 기호 | 읽는 방법 | 의미 |
|--------|-------------|------|
| `y` | 실제값 (Ground Truth) | 정보, 결과 |
| `ŷ` (y-hat) | 예측값 (Prediction) | 모델의 출력 |

> “값과 값의 차”와 같은 단어가 매우 자세한 것과 같지만, 여기서고 바로 **Loss**로 계산됩니다.

---

## 🔢 손실함수(Loss Function) 분류

| 범주 | 함수 | 식 | 특징 / 사용처 |
|------|------|----------|----------------|
| 회귀 | MSE | $\dfrac{1}{N} \sum (y - \hat{y})^2$ | 복수가 큰 경우 무가능 |
|        | MAE | $\dfrac{1}{N} \sum \lvert y - \hat{y} 
vert$ | 이상치 강인 |
| 이지르 분류 | BCE | $- [y \log \hat{y} + (1 - y) \log (1 - \hat{y})]$ | 시그몬드 후 사용 |
| 다중 분류 | Cross-Entropy | $-\sum y_k \log \hat{p}_k$ | 소프트맥스 후 사용 |
| 불규포 | Focal Loss | $-(1 - \hat{p})^{\gamma} \log \hat{p}$ | 어느 사람이 더 어린지 조정 |
| 분포 거리 | KL Divergence | $\sum p \log \dfrac{p}{q}$ | 지시 증류, VAE |

---

## 💡 PyTorch 예제: CrossEntropyLoss

```python
import torch
import torch.nn as nn

logits  = torch.tensor([[2.0, 0.5, -1.0]])  # 모델의 추리
labels = torch.tensor([0])                # 정보

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
print("Cross-Entropy Loss:", loss.item())
```
> `nn.CrossEntropyLoss()`는 내부적으로 `softmax(logits)`를 계산해 결과를 보여주는 패턴입니다.

---

## 🔄 역전파 *(Backpropagation)*

### ▶️ 기본 개념
- Loss가 최소화되도록 **파라미터** (weights, bias)를 변경
- 그래프의 결과를 통해 역전적으로 반영
- 반복되는 양수적 계산을 **Chain Rule**로 차가하게 계산

### ▶️ PyTorch 경우

| 단계 | 코드 | 역할 |
|------|--------|------|
| 순전파 | `y_pred = model(x)` | 그래프 생성 |
| Loss 계산 | `loss = criterion(y_pred, y)` | 노드 추가 |
| 역전파 | `loss.backward()` | \( \partial L/\partial 	heta \) 계산 |
| 갱신 | `optimizer.step()` | \( 	heta \leftarrow 	heta - \eta \cdot \partial L/\partial 	heta \) |
| 그래디언트 리셋 | `optimizer.zero_grad()` | 계산 처음화 |

---

## 🔧 실습: y = 3x 그리기

```python
import torch, torch.nn as nn, torch.optim as optim

model = nn.Linear(1, 1)                 # y = wx + b
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[3.0], [6.0], [9.0]])  # y=3x

for epoch in range(200):
    pred = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}: Loss={loss.item():.4f}")

print("
학습된 w, b:", model.weight.item(), model.bias.item())
```

---

## 🎯 과제

1. `nn.Linear(1,1)`로 y=3x 그리기
2. `nn.MSELoss()` 사용, 경과에 조명 표시
3. 파라미터 (weight 그룹) 값 검색

---

✅ 필요 환경: Python 3.x, Google Colab, PyTorch >= 2.0  
> 설치: `!pip install torch torchvision -q`
