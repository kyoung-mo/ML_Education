# 📘 Week7-1 – Transformer: 이론과 구조

---

## 🔍 Transformer란?

- 2017년 구글이 발표한 “Attention is All You Need” 논문에서 처음 제안된 **딥러닝 시퀀스 모델**입니다.
- RNN/GRU/LSTM 없이도 **모든 시퀀스 정보를 한 번에 처리**할 수 있도록 설계된 구조.
- 대표적 특징: **Self-Attention**(자기어텐션), **병렬 처리**, **장기 의존성 문제 해결**
- 자연어처리(NLP), 비전, 멀티모달 등 광범위하게 사용

---

## 🤔 왜 Transformer가 필요할까?

| RNN/LSTM/GRU | Transformer |
|--------------|-------------|
| 시퀀스 데이터 앞/뒤로 순차처리 | 한 번에 전체 입력 처리 (병렬화) |
| 긴 의존관계 학습 어려움(기울기 소실) | 멀리 떨어진 토큰도 쉽게 연결 |
| 계산속도 느림 (시퀀스 길이에 비례) | 빠른 연산(병렬)과 확장성 |

- **장점:** 긴 문장, 문맥, 복잡한 의존성, 대규모 데이터에서 우수한 성능  
- **혁신:** GPT, BERT, ViT 등 거의 모든 최신 딥러닝 모델의 기반

---

## 🧩 Transformer의 기본 구조

- **입력 임베딩(Input Embedding)**
- **포지셔널 인코딩(Positional Encoding)**
- **인코더(Encoder) 블록**
- **디코더(Decoder) 블록**  
- **출력(예: 단어, 토큰 등)**

> 구조의 핵심: **Self-Attention**과 **Feedforward Network**  
> 각 블록은 Layer Normalization, 잔차연결(Residual Connection), Dropout 등으로 구성

---

### 전체 아키텍처 그림

![Transformer 구조](https://jalammar.github.io/images/t/transformer_architecture.png)

출처: [jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)

---

### **Encoder/Decoder 요약**

| Encoder                   | Decoder                         |
|---------------------------|---------------------------------|
| 입력 임베딩 + 포지셔널 인코딩 | 입력 임베딩 + 포지셔널 인코딩    |
| Self-Attention            | Self-Attention (Masked)         |
| Feedforward Layer         | Cross-Attention (인코더와 연결)  |
| (N회 반복)                | Feedforward Layer (N회 반복)    |

- **Self-Attention**: 각 토큰이 문맥 내 다른 토큰과 관계를 동적으로 파악  
- **Cross-Attention**: 디코더가 인코더의 출력과도 관계 파악

---

## 🔢 포지셔널 인코딩 (Positional Encoding)

- Transformer는 입력 순서를 직접 인식하지 못함 → **위치 정보를 임베딩에 더해줌**
- 대표적 방식(논문 기준): 사인/코사인 함수를 각 차원마다 다르게 사용

$$
PE_{(pos, 2i)} = \sin \left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$
$$
PE_{(pos, 2i+1)} = \cos \left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

- $pos$: 위치, $i$: 임베딩 내 차원 인덱스, $d_{model}$: 임베딩 차원수

---

### 파이토치 예시 코드

```python
import torch
import math

def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i]   = math.sin(pos / (10000 ** ((2 * i)/d_model)))
            if i+1 < d_model:
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * i)/d_model)))
    return pe

pe = positional_encoding(8, 16)
print(pe)
```

---

## 📝 Week7-1 정리

- Transformer는 순환 없이 전체 시퀀스를 동시에 처리
- Self-Attention, Position Encoding 등으로 문맥과 위치 정보 모두 활용
- Encoder-Decoder 구조는 번역, 요약, 생성 등 광범위하게 적용

---

## 🛠️ 실습 과제

1. 위 파이토치 Positional Encoding 코드 실행, 다양한 seq_len/d_model로 시각화  
2. Encoder/Decoder 블록을 그림으로 그려보고 각 레이어 역할 정리  
3. 본인의 전공/관심 분야에서 Transformer 응용 가능성을 조사  
4. (심화) 논문 요약 “Attention is All You Need” 2~3문장으로 요약

---

✅ **환경**: Python 3.x, Google Colab, PyTorch ≥ 2.0  
설치: `!pip install torch torchvision -q`
