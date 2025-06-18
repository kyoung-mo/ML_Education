# 📘 Week5-1 – PyTorch 기초

---

## 🔍 PyTorch란?

- **PyTorch**는 Facebook AI Research가 개발한 딥러닝 프레임워크입니다.
- Numpy와 유사한 **Tensor(텐서)** 연산 지원 + **GPU 가속** 지원.
- 연구/실험, 모델 개발, 실제 배포까지 널리 사용됨.

---

## 🔢 주요 개념

| 용어 | 설명 |
|------|---------------------------|
| Tensor | N차원 배열, NumPy의 ndarray와 유사 |
| Autograd | 자동 미분 기능, 역전파 자동화 |
| nn.Module | 신경망 레이어 및 전체 모델의 베이스 클래스 |
| Optimizer | 파라미터 갱신 알고리즘(SGD, Adam 등) |

---

## 🖥️ 기본 실습 예제

### 1️⃣ Tensor 만들기

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.ones(3, 2)
c = torch.randn(2, 3)  # 평균 0, 표준편차 1

print("a:", a)
print("b:", b)
print("c:", c)
```

---

### 2️⃣ Tensor 연산

```python
x = torch.tensor([[1., 2.], [3., 4.]])
y = torch.tensor([[5., 6.], [7., 8.]])

print("덧셈:", x + y)
print("곱셈:", x * y)
print("행렬곱:", x @ y)
```

---

### 3️⃣ GPU 활용

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
t = torch.arange(10).to(device)
print("장치:", t.device)
```

---

### 4️⃣ 간단한 신경망 정의

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)
    def forward(self, x):
        return self.fc(x)

model = SimpleNet()
print(model)
```

---

## 🛠️ 실습 과제

1. 임의의 2×3 Tensor를 만들어 연산(덧셈, 곱셈, 평균) 수행  
2. CPU/GPU 상에서 Tensor를 생성하고, 장치 이동해보기  
3. nn.Module을 상속한 나만의 신경망 정의 및 임의 입력값 전달해 출력값 확인  

---

✅ **환경**: Python 3.x, Google Colab, PyTorch ≥ 2.0  
설치: `!pip install torch torchvision -q`
