# 📘 Week6-1 – 순환신경망 (RNN) 이론과 구조

---

## 🔍 RNN이란?

- **RNN(Recurrent Neural Network)**은 시퀀스(연속적 데이터, 시계열 등)를 처리할 수 있는 신경망 구조입니다.
- 입력뿐만 아니라 **이전 상태의 정보를 반복적으로 활용**(메모리 효과).
- 대표적 활용: 자연어 처리, 음성 인식, 시계열 예측 등

---

## ⏳ 왜 RNN이 필요한가?

- **MLP, CNN**은 고정 길이 입력만 처리 (과거 맥락 활용 X)
- RNN은 입력 길이가 달라도, **과거의 상태/문맥을 고려**  
  - (예: 문장 생성, 음악 생성, 주가 예측, 기상 데이터 분석 등)

---

## 🧩 RNN의 기본 구조

- **step t** 시점  
  - 입력: $x_t$
  - 이전 hidden state: $h_{t-1}$
  - 현재 hidden state: $h_t = f(W_x x_t + W_h h_{t-1} + b)$
- 출력: $y_t = W_{hy} h_t + b_y$

![RNN 기본구조](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Recurrent_neural_network_unfold.svg/700px-Recurrent_neural_network_unfold.svg.png)

> RNN은 타임스텝마다 **같은 가중치**를 반복 사용(Weight Sharing).

---

## 🧮 수식/연산 과정

1. $h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$
2. $y_t = W_{hy} h_t + b_y$

- $\tanh$, $\text{ReLU}$, $\text{sigmoid}$ 등 다양한 활성화 함수 사용 가능
- 초기 hidden state $h_0$는 0 또는 학습 가능한 값으로 설정

---

## 🔢 파이토치 RNN 구현 (기초)

```python
import torch
import torch.nn as nn

# (batch, seq_len, input_size)
x = torch.tensor([[[1.0], [2.0], [3.0]]])  # batch=1, seq=3, input=1

rnn = nn.RNN(input_size=1, hidden_size=2, batch_first=True)
output, h_n = rnn(x)

print("output shape:", output.shape)  # (batch, seq_len, hidden_size)
print("output:", output)
print("h_n:", h_n)
```

- **input_size**: 입력 벡터 크기 (예: 단어 임베딩 차원)
- **hidden_size**: hidden state의 차원

---

## 🗂️ RNN 시퀀스 처리 흐름

| 시점 | 입력 $x_t$ | 이전 상태 $h_{t-1}$ | 새로운 상태 $h_t$ | 출력 $y_t$ |
|------|---------|--------------------|-----------------|----------|
| 1    | $x_1$    | $h_0$               | $h_1$            | $y_1$     |
| 2    | $x_2$    | $h_1$               | $h_2$            | $y_2$     |
| ...  | ...      | ...                 | ...              | ...       |
| $T$  | $x_T$    | $h_{T-1}$           | $h_T$            | $y_T$     |

---

## 🛠️ 실습 과제

1. 입력 시퀀스 길이와 hidden_size를 바꿔가며 RNN 결과 관찰
2. 여러 개의 시퀀스(batch) 입력 후, output/h_n shape 분석
3. RNN 내부 가중치/파라미터 구조 출력 및 해석

---

✅ **환경**: Python 3.x, Google Colab, PyTorch ≥ 2.0  
설치: `!pip install torch torchvision -q`
