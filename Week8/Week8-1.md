# 📘 Week8-1 – 딥러닝 학습 실전 & 성능 개선/최적화

---

## 🏁 딥러닝 학습 전체 과정

1. **데이터 준비**: 전처리, 증강, 분할(train/val/test)
2. **모델 설계**: 구조(레이어, 활성화 등) 정의
3. **손실함수/Optimizer 설정**: 예) CrossEntropy, Adam
4. **학습/검증 루프**: Forward → Loss → Backward → Optimizer
5. **성능 평가 및 저장**: 모델/파라미터 저장, 테스트

---

## 🔧 주요 하이퍼파라미터

| 항목          | 역할/효과                           |
|---------------|-------------------------------------|
| 학습률        | 너무 크면 발산, 너무 작으면 느림      |
| 배치크기      | 계산 효율성, 일반화 성능 영향         |
| 에폭 수       | 반복 학습 횟수 (많으면 과적합 위험)   |
| 옵티마이저    | SGD, Adam, RMSprop 등                 |
| 초기화        | 파라미터 분포 설정 (Xavier 등)        |

---

## 🎛️ 정규화 & 드롭아웃

- **정규화(Normalization)**: BatchNorm, LayerNorm 등 → 학습 안정화, 속도 향상
- **드롭아웃(Dropout)**: 일부 뉴런 무작위 제거(학습 시) → 과적합 방지

---

## 🚨 Overfitting vs Underfitting

| 구분         | 특징/징후            | 해결법 |
|--------------|----------------------|--------|
| Overfitting  | 학습/검증 성능 차이, val loss 증가 | 정규화, 드롭아웃, 데이터증강, EarlyStopping |
| Underfitting | train/val 모두 성능 낮음           | 모델 복잡도 ↑, 학습 더 오래, 학습률 ↑ 등  |

---

## 🔢 데이터 증강 (예시: 이미지)

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])
```

- 증강 기법 조합 → 일반화 성능 ↑

---

## 🛠️ 실습 예시: CIFAR-10 이미지 분류 학습/튜닝

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 데이터셋 및 증강
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 간단한 CNN 모델
class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16*16*16, 10)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16*16*16)
        x = self.fc(x)
        return x

model = MyCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프(요약)
for epoch in range(5):
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} 완료")
```

---

## 🛠️ 실습/튜닝 과제

1. 학습률, 배치크기, 드롭아웃 등 변경하며 성능 비교  
2. 데이터 증강 적용 전/후 성능 차이 측정  
3. Overfitting/Underfitting 시나리오 직접 만들어보기  
4. (심화) EarlyStopping, 학습 곡선 시각화 코드 추가

---

✅ **환경**: Python 3.x, Google Colab, PyTorch ≥ 2.0  
설치: `!pip install torch torchvision -q`
