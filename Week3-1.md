# 📘 Week3-1 - 인공신경망 구조, parameter

---

## 🧠 인공신경망(Artificial Neural Network, ANN)

### 🔹 개념
- 인간의 뇌 구조에서 영감을 받아 만들어진 모델
- 입력층(Input layer), 은닉층(Hidden layer), 출력층(Output layer)으로 구성됨

### 🔹 뉴런 구조
- 하나의 뉴런은 여러 입력 값을 받아 가중치(weight)를 곱하고, 편향(bias)을 더한 뒤 활성화 함수(activation function)를 통해 출력 생성

### 뉴런 수식
\[ y = f(w_1 x_1 + w_2 x_2 + ... + w_n x_n + b) \]

---

## 🔩 파라미터(Parameter)

### 🔹 주요 파라미터
- **가중치(Weights)**: 각 입력값의 중요도를 나타냄
- **편향(Bias)**: 출력값을 조절하는 추가 값
- 학습은 이 두 파라미터를 조절해 오차를 줄여가는 과정

---

## 🔧 PyTorch를 활용한 신경망 예시

### 🔹 기본 신경망 모델 구성
```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 8)  # 입력 4, 출력 8
        self.fc2 = nn.Linear(8, 3)  # 입력 8, 출력 3

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### 🔹 모델 생성 및 파라미터 확인
```python
model = SimpleNN()
print(model)

# 파라미터 개수 확인
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

---

## 🧪 과제
1. 입력 2, 출력 1인 신경망을 설계하시오 (은닉층 1개, 4개 뉴런).
2. 모델의 전체 파라미터 수를 계산하고 출력하시오.
3. `torch.nn.Sequential`을 이용하여 같은 모델을 다시 만들어보시오.

---

✅ 사용 환경: Python 3.x, Google Colab, PyTorch 설치 필요
