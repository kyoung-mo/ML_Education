# 📘 Week7-2 – Transformer 심화: Self-Attention, Multi-Head, 구현

---

## ✨ Self-Attention이란?

- 각 입력(토큰)이 전체 시퀀스의 모든 입력과 **상호작용**하여 “문맥의 중요도”를 동적으로 학습
- RNN과 달리 “한 번에 모든 위치의 관계”를 계산

### 수식 (Scaled Dot-Product Attention)
- 입력: Query(Q), Key(K), Value(V) 행렬  
- 계산:
  $$
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
  $$
  - $d_k$: K의 차원수 (스케일링, 안정성↑)

---

## 🔢 Self-Attention 연산 흐름

1. **Query/Key/Value** 행렬 계산 (선형 변환)
2. $QK^T$ (유사도 점수) → softmax로 “중요도” 결정
3. 각 위치의 Value를 중요도만큼 합산 (가중합)

|  | 입력 문장 | Q/K/V 생성 | 유사도 | 가중합 | 출력 |
|--|----------|-----------|--------|-------|------|
| ex | I like pizza | Q, K, V | I↔pizza | pizza 강조 | |

---

## 🧩 Multi-Head Attention

- 여러 개의 Self-Attention을 병렬로 수행, 서로 다른 “관점(서브스페이스)”에서 문맥 파악  
- 다양한 관계를 동시에 학습 가능

### 그림

![multi-head attention](https://jalammar.github.io/images/t/multi_head_attention.png)

---

## 🔧 Feed Forward, LayerNorm, Residual

- **Feed Forward Network (FFN)**: 각 토큰별로 독립적으로 처리되는 MLP(2개 Linear + 활성화)
- **Layer Normalization**: 각 레이어의 분산/평균 정규화 (학습 안정화)
- **Residual Connection**: 입력+출력 더해 gradient 흐름 개선, 성능 향상

---

## 🗂️ 파이토치 Transformer 구현 예시

```python
import torch
import torch.nn as nn

model = nn.Transformer(
    d_model=16, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
    dim_feedforward=32, batch_first=True
)
src = torch.rand(2, 5, 16)  # (batch, seq, feature)
tgt = torch.rand(2, 5, 16)
out = model(src, tgt)
print(out.shape)  # (batch, seq, d_model)
```

---

## 🛠️ 실습 과제

1. nn.Transformer를 사용해, 입력 차원/레이어/head 개수를 바꿔가며 실험  
2. 직접 Multi-Head Self-Attention 레이어 구현 (파이토치 공식문서 참고)
3. FeedForward/LayerNorm/Residual 역할 실험 (추가/삭제 후 성능 비교)
4. 실제 논문/실제 모델(GPT, BERT 등)에서 Attention 결과 시각화 자료 조사

---

## 📚 참고: Transformer 응용/파생

- **BERT**: Encoder만, 문장 이해/분류/질문응답
- **GPT**: Decoder만, 텍스트 생성
- **ViT**: 이미지 패치 입력, 비전 영역
- **LLM/멀티모달**: 대규모 파라미터, 언어+이미지 등 다양한 입력

---

✅ **환경**: Python 3.x, Google Colab, PyTorch ≥ 2.0  
설치: `!pip install torch torchvision -q`
