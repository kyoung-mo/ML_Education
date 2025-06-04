
# 📘 Week3‑2 – 손실 함수 & 역전파
---

## 📉 손실 함수 *(Loss Function)*

### 1️⃣ 정의
- 모델 **예측(\(\hat{y}\))** 과 **실제 값(\(y\))** 사이의 차이를 수치화한 지표  
- **목표**: 학습 동안 손실을 최소화 ⇒ 일반화 성능 ↑

### 2️⃣ 손실 함수 모음

| 범주 | 함수 | 식(요약) | 특징 / 사용처 |
|------|------|----------|---------------|
| **회귀** | **MSE**<br>(Mean Squared Error) | \(\frac1N\sum (y-\hat{y})^2\) | 이상치 민감, 가장 보편적 |
| | MAE | \(\frac1N\sum |y-\hat{y}|\) | 이상치 강인, 미분이 0/1 |
| **이진 분류** | **BCE**<br>(Binary Cross‑Entropy) | \(-[y\log \hat{y} + (1-y)\log(1-\hat{y})]\) | 시그모이드 후 사용 |
| **다중 분류** | **Cross‑Entropy** | \(-\sum y_k\log\hat{p}_k\) | 소프트맥스 후 사용 |
| **불균형 데이터** | Focal Loss | \(-(1-\hat{p})^{\gamma} \log\hat{p}\) | 어려운 샘플 집중 학습 |
| **예측 분포** | KL Divergence | \(\sum p\log(p/q)\) | 지식 증류, VAE |

> **PyTorch**: `nn.MSELoss`, `nn.L1Loss`, `nn.BCELoss`, `nn.BCEWithLogitsLoss`, `nn.CrossEntropyLoss`, `nn.KLDivLoss` …

---

### 3️⃣ Cross‑Entropy 예제
```python
import torch, torch.nn as nn

pred = torch.tensor([[2.0, 0.5, -1.0]])   # 로짓(logit)
target = torch.tensor([0])                # 정답 레이블

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(pred, target)
print("CE Loss:", loss.item())
```
> `nn.CrossEntropyLoss` = **`LogSoftmax` + `NLLLoss`** 를 한 번에 수행합니다.

---

## 🔁 역전파 *(Backpropagation)*

### 1️⃣ 아이디어
- 다층 함수 \(\mathcal{L}(\theta)=f^{(n)}\circ\dots\circ f^{(1)}(x)\) 의 **기울기**를  
  체인 룰(Chain Rule)로 효율적으로 계산  
- 그래프를 **뒤에서 앞으로** 따라가며 \(\frac{\partial \mathcal{L}}{\partial \theta}\) 누적

<div align="center"><img src="https://raw.githubusercontent.com/ml-assets/backprop-flow/main/backprop_graph.svg" width="480"/></div>

### 2️⃣ PyTorch 자동 미분 흐름

| 단계 | 코드 | 역할 |
|------|------|------|
| **순전파** | `y_pred = model(x)` | Computational Graph 생성 |
| **손실 계산** | `loss = criterion(y_pred, y)` | 그래프에 노드 추가 |
| **역전파** | `loss.backward()` | 그래프를 따라 ∂L/∂θ 계산 & 저장 |
| **옵티마이저** | `optimizer.step()` | θ ← θ − η·∂L/∂θ |
| **기울기 초기화** | `optimizer.zero_grad()` | 그래디언트 누적 방지 |

---

## 🛠️ 간단 학습 예제: 선형 회귀

```python
import torch, torch.nn as nn, torch.optim as optim

# 모델: y = wx + b
model = nn.Linear(1, 1)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[3.0], [6.0], [9.0]])  # y = 3x

for epoch in range(200):
    pred = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()  # ⇤ 필수
    loss.backward()        # ∂L/∂θ
    optimizer.step()       # θ 업데이트

    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}: Loss={loss.item():.4f}")

print("학습된 w, b:", model.weight.item(), model.bias.item())
```

---

## ⚙️ 옵티마이저 비교

| Optimizer | 특징 | 하이퍼파라미터 |
|-----------|------|----------------|
| SGD | 단순, 메모리 적음 | lr, momentum |
| Adam | 적응형 학습률, 대부분 ‘기본값’ OK | lr, β1, β2, eps |
| RMSprop | RNN에 강함 | lr, α |
| Adagrad | 드물게 업데이트되는 파라미터에 유리 | lr |

> 옵티마이저 변경 = 코드 한 줄  
> `optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)`

---

## 🎯 과제

**목표**: 선형 모델로 \(y=3x\) 근사  
1. `nn.Linear(1,1)` 으로 모델 정의 후  **200 에폭** 학습  
2. `nn.MSELoss()` 사용, **에폭마다 손실** 출력  
3. 학습 종료 후 **weight와 bias** 를 출력 (이론값: 3, 0)

> **Tip**: 학습률(lr)을 0.05~0.1로 키우면 빠르게 수렴하지만 너무 크면 발산할 수도 있습니다.

---

✅ **환경**: Python 3.x, Google Colab, **PyTorch ≥ 2.0**  
설치: `!pip install torch torchvision -q`
