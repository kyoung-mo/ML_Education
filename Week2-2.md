# 📘 Week2-2 - scikit-learn, 머신러닝 기본 개념

---

## 🤖 머신러닝 기본 개념

### 🔹 머신러닝이란?
- 데이터로부터 패턴을 학습하여 예측하거나 분류하는 기술
- 지도학습(Supervised), 비지도학습(Unsupervised), 강화학습(Reinforcement Learning)으로 분류됨

### 🔹 용어 정리
- **특징(feature)**: 입력 데이터의 속성 값
- **레이블(label)**: 예측하고자 하는 값 (정답)
- **모델(model)**: 학습을 통해 만들어진 예측 시스템
- **훈련(training)**: 데이터를 이용해 모델을 학습시키는 과정

---

## 🔧 scikit-learn 소개

### 🔹 scikit-learn이란?
- 파이썬 기반의 머신러닝 라이브러리
- 다양한 모델과 전처리 기능을 제공

### 설치
Google Colab에서는 기본 설치되어 있음
```python
import sklearn
print(sklearn.__version__)
```

---

## 🛠️ 간단한 모델 실습: 붓꽃(Iris) 분류

### 🔹 데이터 불러오기
```python
from sklearn.datasets import load_iris

iris = load_iris()
print(iris.data[:5])
print(iris.target[:5])
```

### 🔹 훈련/테스트 데이터 분리
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
```

### 🔹 모델 학습 및 평가
```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"정확도: {accuracy:.2f}")
```

---

## 🧪 과제
1. `load_wine()` 데이터를 불러오고, KNN 분류기로 정확도를 측정하시오.
2. `test_size`를 0.3으로 변경하여 성능 차이를 비교하시오.
3. `n_neighbors`를 1, 3, 5로 바꾸어 보고 정확도를 비교하시오.

---

✅ 사용 환경: Python 3.x, Google Colab (scikit-learn 기본 설치됨)
