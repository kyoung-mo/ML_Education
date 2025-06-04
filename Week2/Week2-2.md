
# 📘 Week2‑2 – scikit‑learn, 머신러닝 기본 개념
---

## 🤖 머신러닝 기본 개념

### 🔹 머신러닝이란?
- 명시적 규칙(rule)을 코딩하는 대신, **데이터로부터 패턴을 학습**하여 새로운 입력을 예측·분류하는 기술입니다.  
- 크게 **지도학습(Supervised)**, **비지도학습(Unsupervised)**, **강화학습(Reinforcement)** 으로 나뉩니다.

| 학습 유형 | 입력 | 목표 | 대표 알고리즘 |
|-----------|------|------|--------------|
| 지도학습 | 특징 + 정답 | 보이지 않는 데이터의 **정답 예측** | Linear/Logistic Regression, SVM, Random Forest |
| 비지도학습 | 특징만 | **구조 발견·군집·차원축소** | K‑Means, PCA, AutoEncoder |
| 강화학습 | 상태 + 보상 | **행동 정책 최적화** | Q‑Learning, Policy Gradient |

### 📊 전체 흐름도

```
[데이터 수집]
      ↓
[전처리 및 특징/레이블 분리]
      ↓
[학습 데이터 / 평가 데이터 나누기]
      ↓
[모델 선택 및 학습]
      ↓
[모델 평가]
      ↓
[새로운 입력 데이터 예측]
```

### 📘 주요 용어

| 용어 | 의미 | 예시 |
|------|------|------|
| **특징 (feature)** | 입력 데이터의 속성 값 | 키, 몸무게, 성별, 나이 |
| **레이블 (label)** | 예측하려는 정답 값 | 비만 여부 (예: 0: 정상, 1: 비만) |
| **모델 (model)** | 학습된 예측 함수 | "이 키와 몸무게면 비만일 확률은 0.87" |
| **훈련 (training)** | 모델이 데이터를 보고 패턴을 학습하는 과정 | 여러 사람의 키·몸무게와 비만 여부를 이용 |
| **평가 (evaluation)** | 모델이 얼마나 잘 맞추는지 측정 | 새로운 사람의 비만 여부를 테스트 |
| **과적합 (overfitting)** | 훈련 데이터에만 너무 맞춤 | 새 데이터에선 오답이 많아짐 |
| **일반화 (generalization)** | 새 데이터에서도 잘 작동하는 능력 | 새로운 경우에도 높은 정확도 유지 |


> **Tip**   
> 과적합을 방지하려면 **데이터 분할, 교차검증, 규제(regularization)** 를 적극 활용하세요!

---

## 🔧 scikit‑learn 소개

### 🔹 scikit‑learn이란?
- NumPy/SciPy 기반 **범용 머신러닝 라이브러리**  
- **일관된 API**(`fit()`, `predict()`, `score()`)와 **풍부한 전처리·모델·평가** 도구 제공  
- 대규모 딥러닝보다는 **클래식 ML**에 최적화 (CPU, 중소규모 데이터셋)

```python
import sklearn
print("scikit-learn version:", sklearn.__version__)
```

### 🔹 scikit‑learn 파이프라인 구조
```
데이터 ➜ 전처리(Scaler/Encoder) ➜ 모델 학습 ➜ 예측 ➜ 평가
```
- **`Pipeline`** 클래스로 절차를 **체인**화하면, 데이터 누설(data leakage)을 방지하고 하이퍼파라미터 탐색을 자동화할 수 있습니다.

---

## 🔧 전처리(Preprocessing)란?

전처리는 **머신러닝 모델에 데이터를 넣기 전에, 데이터를 정리하고 가공하는 과정**입니다.  
모델이 데이터를 잘 이해하고 학습할 수 있도록 돕는 매우 중요한 단계입니다.

---

## ✅ 전처리가 필요한 이유

- 현실의 데이터는 종종 **결측치, 이상치, 문자열, 크기 차이** 등이 포함되어 있어 그대로는 학습이 어렵습니다.
- 전처리를 통해 **데이터를 숫자화하고, 정리하고, 스케일을 맞춤**으로써 모델의 성능을 높일 수 있습니다.

---

## 💡 주요 전처리 작업

- **결측값 처리**: 빠진 값을 평균 등으로 채우거나 제거
- **이상치 제거**: 너무 큰/작은 값 제거
- **인코딩**: 문자 데이터를 숫자로 변환
- **정규화/표준화**: 값의 범위를 일정하게 맞춤
- **중복 제거 및 타입 변환** 등
---

## 🛠️ 실습: 붓꽃(Iris) 분류

### 🔹 1) 데이터 불러오기
```python
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)  # DataFrame 형태로 받기
X, y = iris.data, iris.target
```

### 🔹 2) 훈련 / 테스트 분리
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### 🔹 3) 전처리 + 모델 파이프라인
```python
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn",    KNeighborsClassifier(n_neighbors=3))
])

pipe.fit(X_train, y_train)
```

### 🔹 4) 성능 평가
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = pipe.predict(X_test)

print("정확도:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

---

## 🔍 하이퍼파라미터 튜닝

`GridSearchCV`를 활용하면 `n_neighbors`, `metric` 등 파라미터를 **교차검증**으로 최적화할 수 있습니다.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {"knn__n_neighbors": [1, 3, 5, 7]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print("최적 파라미터:", grid.best_params_)
print("검증 정확도:", grid.best_score_)
```

---

## 🧪 과제

> **데이터셋**: `load_wine()` (와인 품질 데이터)

1. 와인 데이터를 불러오고, KNN 분류기로 **기본 정확도**를 측정하세요.  
2. `test_size`를 **0.3**으로 변경해 정확도 변화를 비교하세요.  
3. `n_neighbors`를 **1, 3, 5**로 바꾸어 각각 정확도를 기록하세요.

### 🔑 힌트
```python
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score

wine = load_wine(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target,
    test_size=0.3, stratify=wine.target, random_state=42
)

for k in [1, 3, 5]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"k={k}, 정확도={acc:.3f}")
```
> **추가 도전 🌟**   
> • `StandardScaler`를 파이프라인에 넣어 성능이 어떻게 변하는지 살펴보세요.  
> • `cross_val_score`로 여러 폴드 평균 정확도를 구해보세요.

---

## 🛠️ scikit‑learn 퀵 레퍼런스

| 범주 | 대표 클래스 | 설명 |
|------|-------------|------|
| 전처리 | `StandardScaler`, `OneHotEncoder`, `PolynomialFeatures` | 정규화·인코딩·특성 생성 |
| 분류 | `KNeighborsClassifier`, `SVC`, `RandomForestClassifier`, `LogisticRegression` | |
| 회귀 | `LinearRegression`, `SVR`, `RandomForestRegressor` | |
| 클러스터링 | `KMeans`, `DBSCAN`, `AgglomerativeClustering` | 비지도 군집 |
| 차원축소 | `PCA`, `TSNE`, `TruncatedSVD` | |
| 모델 선택 | `train_test_split`, `GridSearchCV`, `cross_val_score` | 데이터 분할·CV·튜닝 |
| 평가 지표 | `accuracy_score`, `mean_squared_error`, `roc_auc_score` | |

---

✅ **사용 환경**: Python 3.x, Google Colab (scikit‑learn 기본 탑재)  
필요 시 `!pip install scikit-learn -U` 로 최신 버전을 설치하세요.
