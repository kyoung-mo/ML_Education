# 📘 Week5-2 – CNN (합성곱 신경망, Convolutional Neural Network)

---

## 🔍 CNN이란?

- **CNN(Convolutional Neural Network)**은 이미지, 음성 등 공간/시계열 데이터 분석에 특화된 신경망 구조입니다.
- 합성곱(Convolution) 연산을 이용해 **공간적 특징(패턴, 엣지 등)**을 자동으로 추출.
- 주요 분야: 이미지 분류, 객체 탐지, 음성 인식 등

---

## 🧩 기본 구조

1. **합성곱층(Conv Layer)**  
   - 이미지 특징(엣지, 패턴 등) 추출  
   - 필터(커널) 사용
2. **활성화 함수(ReLU 등)**
3. **풀링(Pooling)**
   - 특징의 공간 크기 축소 (예: MaxPooling)
4. **완전연결층(FC, Linear)**
   - 최종 분류

---

## 🗂️ 구조 예시

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # 1채널 입력, 8채널 출력, 3x3 커널
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8*14*14, 10)           # (입력이미지 28x28 가정)
    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))     # 합성곱 → ReLU → 풀링
        x = x.view(-1, 8*14*14)                     # 평탄화
        x = self.fc1(x)
        return x

model = SimpleCNN()
print(model)
```

---

## 🛠️ CNN 특징

| 용어 | 설명 |
|------|--------------------------|
| 파라미터 공유 | 같은 필터(커널)가 전체 영역을 훑음 |
| 지역 연결성 | 입력의 일부분(로컬 패치)만 연결 |
| 풀링 | 공간 정보 요약(축소), 과적합 감소 |

---

## 🖥️ 실습 과제

1. 위 코드 참고하여, 임의의 이미지(1×28×28) 입력 후 출력 shape 확인  
2. Conv2d, MaxPool2d 등 다양한 파라미터(커널, 스트라이드) 바꿔가며 출력 shape 관찰  
3. 파라미터 개수 세고, FC 신경망과 비교

---

✅ **환경**: Python 3.x, Google Colab, PyTorch ≥ 2.0  
설치: `!pip install torch torchvision -q`
