# 📘 Week4‑1 – 손실 함수 & 역전파

---

## 📉 손실 함수 *(Loss Function)*

### 🔍 `y`·`ŷ`(y‑hat) 기호 이해
| 기호 | 읽는 법 | 의미 |
|------|--------|------|
| **`y`** | 와이 | **실제값**·정답(Ground Truth) |
| **`ŷ`** (y‑hat) | 와이‑햇 | **모델이 예측한 값**·추정치(Prediction) |

> 모자(^) 모양 **“hat”** 은 통계·머신러닝에서 “추정(estimation)”을 나타내는 관례적 표기입니다.  
> 즉, 손실 함수는 **실제값 `y`** 와 **예측값 `ŷ`** 의 차이를 수치화합니다.

---

### 1️⃣ 손실 함수 모음

| 범주 | 함수 | 식 (요약) | 특징 / 사용처 |
|------|------|-----------|----------------|
| 회귀 | **MSE** | $\dfrac{1}{N} \sum (y - \hat{y})^2$ | 이상치 민감, 보편적 |
|      | MAE | $\dfrac{1}{N} \sum \lvert y - \hat{y} \rvert$ | 이상치 강인 |
| 이진 분류 | **BCE** | $- \big[ y \log \hat{y} + (1 - y) \log (1 - \hat{y}) \big]$ | 시그모이드 이후 사용 |
| 다중 분류 | **Cross-Entropy** | $- \sum y_k \log \hat{p}_k$ | 소프트맥스 이후 |
| 불균형 | Focal Loss | $- (1 - \hat{p})^{\gamma} \log \hat{p}$ | 어려운 샘플 집중 |
| 분포 거리 | KL Divergence | $\sum p \log \frac{p}{q}$ | 지식 증류, VAE |

> 여기서 $\hat{p}$ 역시 **예측 확률**을 뜻합니다.

---

### 2️⃣ PyTorch 예제

```python
import torch
import torch.nn as nn

logits  = torch.tensor([[2.0, 0.5, -1.0]])  # 모델이 낸 점수(logit)
targets = torch.tensor([0])                 # 정답 레이블

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, targets)
print("Cross‑Entropy Loss:", loss.item())
```

---

## 🔁 역전파 *(Backpropagation)*

### 1️⃣ 개념
- 손실이 줄어들도록 **파라미터(가중치·편향)** 를 업데이트하는 알고리즘  
- **체인 룰**로 다층 미분을 효율적으로 계산

### 2️⃣ PyTorch 학습 절차

| 단계 | 코드 | 역할 |
|------|------|------|
| 순전파 | `y_pred = model(x)` | 그래프 생성 |
| 손실 계산 | `loss = criterion(y_pred, y)` | 노드 추가 |
| 역전파 | `loss.backward()` | ∂L/∂θ 계산 |
| 업데이트 | `optimizer.step()` | θ ← θ − η·∂L/∂θ |
| 초기화 | `optimizer.zero_grad()` | 그래디언트 리셋 |

---

## 🔬 [실습] 넘파이 vs 파이토치로 2단 MLP “역전파(gradient)” 비교

... (중략, 이전 응답 내용 전체 포함) ...
