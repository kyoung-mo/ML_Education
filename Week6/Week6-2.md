# 📘 Week6-2 – RNN 심화: LSTM & GRU, 한계와 극복

---

## ⚠️ RNN의 한계

- **기울기 소실(Gradient Vanishing)**: 긴 시퀀스 학습 시, 역전파 과정에서 gradient가 0에 수렴 → 과거 정보 “망각”
- **기울기 폭주(Gradient Explosion)**: gradient가 무한대로 커져서 학습 불안정
- **장기 의존성 학습 어려움**: 예를 들어, “긴 문장”에서 앞 단어와 뒷 단어 관계를 기억 못함

---

## 🧩 LSTM (Long Short-Term Memory) 구조

- **장기 의존성**을 기억하도록 설계된 RNN의 변형
- **게이트 구조** (입력, 망각, 출력 게이트)로 정보 흐름 조절

### LSTM 구조(개념 그림)

![LSTM cell](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

### 주요 연산

- **입력게이트** $i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + b_i)$
- **망각게이트** $f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + b_f)$
- **출력게이트** $o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + b_o)$
- **셀상태** $c_t = f_t * c_{t-1} + i_t * \tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c)$
- **히든스테이트** $h_t = o_t * \tanh(c_t)$

> $\sigma$는 시그모이드, $*$는 원소곱(element-wise multiplication)

---

## 🧩 GRU (Gated Recurrent Unit)

- LSTM보다 단순, 유사한 성능
- **업데이트 게이트**와 **리셋 게이트**만 사용 (셀 상태 X)

---

## 🗂️ 파이토치 LSTM·GRU 예제

```python
import torch
import torch.nn as nn

x = torch.rand(2, 5, 3)  # batch=2, seq_len=5, input_size=3

lstm = nn.LSTM(input_size=3, hidden_size=4, batch_first=True)
output, (h_n, c_n) = lstm(x)
print("LSTM output:", output.shape, "| h_n:", h_n.shape, "| c_n:", c_n.shape)

gru = nn.GRU(input_size=3, hidden_size=4, batch_first=True)
output, h_n = gru(x)
print("GRU output:", output.shape, "| h_n:", h_n.shape)
```

- **LSTM, GRU 사용법은 거의 동일**
- (output, h_n), (output, (h_n, c_n)) 반환

---

## 🛠️ 실습 과제

1. 입력 시퀀스, hidden_size, layer 개수 바꿔가며 LSTM/GRU 출력값 관찰
2. 시계열 데이터를 LSTM/GRU로 예측하는 간단한 예제 구성
3. LSTM/GRU의 파라미터 개수와 RNN 비교

---

## 📚 참고: 실제 적용 분야

- 번역, 챗봇, 감정분석, 주가예측 등
- LSTM/GRU는 여전히 다양한 시퀀스 문제에서 강력한 baseline

---

✅ **환경**: Python 3.x, Google Colab, PyTorch ≥ 2.0  
설치: `!pip install torch torchvision -q`
