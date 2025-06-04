
# 📘 Week3‑1 – 인공신경망 구조 & 파라미터
---

## 🧠 인공신경망(Artificial Neural Network, **ANN**)

### 1️⃣ 개념
- **인공뉴런(Perceptron)** 수백~수만 개를 **계층(layer)** 으로 연결한 함수 근사 모델  
- 입력 → 가중합 → 비선형 변환 → 출력 과정을 **전파(Propagation)** 하며,  
  **오차(손실)** 를 역으로 전파(Back‑propagation)해 파라미터를 학습  
- 이론적으로 은닉층이 충분하면 **모든 연속함수**를 근사할 수 있음(Universal Approximation).

<img src="https://raw.githubusercontent.com/ml-assets/ann-drawing/main/ann_layers.svg" width="480"/>

---

### 2️⃣ 뉴런(Neuron)의 수식
\[
y = f(\underbrace{\sum_{i=1}^{n}w_i x_i}_{\text{가중합}} + \; b)
\]

| 기호 | 의미 | 학습 대상? |
|------|------|-----------|
| \(x_i\) | 입력 Feature | ❌ |
| \(w_i\) | 가중치(Weight) | ✅ |
| \(b\) | 편향(Bias) | ✅ |
| \(f(\cdot)\) | **활성화 함수** | ❌ (선택 사항) |

> **벡터화**  
> \(\boldsymbol{y}=f(\mathbf{W}\mathbf{x}+\mathbf{b})\) 로 표현하며, 라이브러리에서는 행렬곱(\*) 한 줄로 구현합니다.

---

## ⚙️ 활성화 함수 한눈에

| 함수 | 식 | 특징 |
|------|----|------|
| **Sigmoid** | \( \sigma(z)=\frac1{1+e^{-z}} \) | 확률 해석 용이, **vanishing gradient** |
| **Tanh** | \( \tanh(z) \) | 0 중심이지만 여전히 vanishing |
| **ReLU** | \( \max(0,z) \) | 계산 단순, 빠른 수렴, dying ReLU 주의 |
| **Leaky ReLU** | \( \max(\alpha z, z) \) | ReLU의 음수 영역 기울기 보완 |
| **Softmax** | \( \frac{e^{z_k}}{\sum_j e^{z_j}} \) | 다중 클래스 확률 분포 |

---

## 🔩 파라미터(Parameter)

### 1️⃣ 종류
| 파라미터 | 계층당 개수 | 역할 |
|----------|------------|------|
| **Weights** | 입력노드 × 출력노드 | 데이터 간 중요도 학습 |
| **Biases** | 출력노드 | 결정경계 이동·출력 보정 |

### 2️⃣ 파라미터 수 계산 예
> **구조**: Input 4 → Hidden 8 → Output 3  
> \[
\#W = 4 \times 8 + 8 \times 3 = 44, \quad
\#b = 8 + 3 = 11, \quad
\textbf{총}\;55\;\text{개}
\]

---

## 🔧 PyTorch 실습 예제

```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),   # 입력 4, 은닉 8
            nn.ReLU(),
            nn.Linear(8, 3)    # 은닉 8, 출력 3
        )

    def forward(self, x):
        return self.net(x)
```

### 파라미터 확인
```python
model = SimpleNN()
print(model)

total = sum(p.numel() for p in model.parameters())
print("총 파라미터:", total)  # 55개
```

### 미니 학습 루프 예시
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    optimizer.zero_grad()
    logits = model(x_train)          # 순전파
    loss = criterion(logits, y_train)
    loss.backward()                  # 역전파
    optimizer.step()                 # 파라미터 갱신
```

---

## 🧑‍🏫 ANN 학습 프로세스 요약
1. **순전파(Forward Pass)**: 입력을 통해 출력 계산 → 손실 \(\mathcal{L}\) 산출  
2. **역전파(Backward Pass)**: \(\frac{\partial \mathcal{L}}{\partial w},\frac{\partial \mathcal{L}}{\partial b}\) 계산  
3. **옵티마이저(Optimizer)**: SG​D, Adam 등으로 파라미터 업데이트  
4. **반복(Epoch)**: 데이터셋을 한 바퀴 순회 → 손실·정확도 모니터링  
5. **일반화(Generalization)**: 검증/테스트 데이터로 과적합 여부 확인

---

## 🎯 과제

> **조건**: PyTorch 사용, Colab 추천

1. **모델 설계** – 입력 2, 은닉층 4(1개 층), 출력 1인 신경망을 구축하세요.  
2. **파라미터 수 산출** – 직접 계산한 값과 `sum(p.numel() for p in model.parameters())` 결과를 비교해 보세요.  
3. **`nn.Sequential` 구현** – 과제 1과 동일한 구조를 `nn.Sequential` 로 재구현해 보세요.

### 힌트
```python
seq_model = nn.Sequential(
    nn.Linear(2, 4),
    nn.Tanh(),
    nn.Linear(4, 1)
)
print(seq_model)
print("파라미터:", sum(p.numel() for p in seq_model.parameters()))
```

> **추가 도전 🌟**   
> • `torchinfo.summary(seq_model, input_size=(1,2))`로 상세 구조를 출력해 보세요(패키지 설치 필요).  
> • 같은 문제를 Keras/TensorFlow로도 구현해 비교해 보세요.

---

✅ **사용 환경**: Python 3.x, Google Colab, *PyTorch ≥2.0*  
`!pip install torch torchvision torchinfo -q` 로 설치 후 실행하면 됩니다.
