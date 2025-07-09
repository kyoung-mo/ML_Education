# 머신러닝 분류 및 군집화 알고리즘 정리

기계학습에서는 데이터의 특성과 목적에 따라 다양한 알고리즘이 사용된다.  
그중 `k-최근접 이웃 (kNN)`, `서포트 벡터 머신 (SVM)`, `결정 트리 (Decision Tree)`은 분류 그리고 `k-평균 클러스터링 (k-Means)`은 군집화 문제에서 자주 사용되는 대표적인 알고리즘이다.

---
## 1️⃣ kNN (k-Nearest Neighbors)

- **설명**: 주변의 가장 가까운 k개의 이웃 데이터의 라벨을 참조해 분류하는 비모수적 지도학습 알고리즘이다.

- **작동 원리**
  1. 새로운 입력과 훈련 데이터 간의 거리를 계산 (보통 유클리디안 거리).
  2. 가장 가까운 k개의 이웃을 선택.
  3. 이웃 중 가장 많이 나타나는 클래스(다수결)로 분류.

- **장점**
  - 구현이 간단하고 직관적이다.
  - 훈련 과정이 따로 없고, 저장만 해두면 된다.
  - 다양한 클래스와 복잡한 결정 경계도 잘 표현할 수 있다.

- **단점**
  - 모든 훈련 데이터와 거리를 계산하므로 예측 시 계산량이 크다.
  - 고차원 데이터에서는 성능이 저하됨(차원의 저주).
  - 이상치에 민감하고, 데이터가 불균형할 경우 성능이 낮아질 수 있다.

예제)

```python
# 라이브러리 불러오기
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1. Create synthetic data
X, y = make_moons(n_samples=300, noise=0.3, random_state=0)

# 2. Train/Test split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)

# 3. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_all_scaled = scaler.transform(X)

# 4. Meshgrid for plotting
def plot_decision_boundary(knn, X, y, ax, k):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
  
    cmap_light = ListedColormap(['#FFBBBB', '#BBFFBB'])
    cmap_bold = ListedColormap(['#FF0000', '#00AA00'])
  
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k')
    ax.set_title(f"k = {k}")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
   
# 5. Try different k values
k_values = [1, 3, 5, 15]
fig, axes = plt.subplots(1, len(k_values), figsize=(18, 4))
  
for ax, k in zip(axes, k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    plot_decision_boundary(knn, X_all_scaled, y, ax, k)
  
plt.tight_layout()
plt.show()
```

---
## 2️⃣ SVM (Support Vector Machine)

- **설명**: 클래스 간의 경계를 최대한 넓게 확보하는 결정 경계를 찾아 분류하는 지도학습 알고리즘이다.

- **작동 원리**
  1. 두 클래스 사이에서 마진(여백)이 최대가 되는 선형 결정 경계(초평면)를 탐색.
  2. 결정 경계에 가장 가까운 점들을 `서포트 벡터`라고 하며, 이들이 경계를 결정함.
  3. 선형 분리가 어려운 경우, 커널 트릭을 사용해 고차원으로 변형하여 분류 수행.

- **장점**
  - 마진을 최대화함으로써 일반화 성능이 우수.
  - 다양한 커널 함수로 비선형 문제도 처리 가능.
  - 과적합 위험이 적다.

- **단점**
  - 훈련 시간이 오래 걸릴 수 있으며, 대규모 데이터셋에 비효율적.
  - 커널과 하이퍼파라미터 선택에 민감.
  - 예측 확률 값을 직접 제공하지 않음.

```python
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 1. Create a simple 2-class dataset
X, y = make_blobs(n_samples=100, centers=2, random_state=6, cluster_std=1.2)

# 2. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train SVM with linear kernel
svm_clf = SVC(kernel='linear', C=1.0)
svm_clf.fit(X_scaled, y)

# 4. Visualization
def plot_svm_decision_boundary(clf, X, y):
    plt.figure(figsize=(8,6))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='bwr', edgecolors='k')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # Plot decision boundary and margins
    plt.contour(XX, YY, Z, colors='k',
                levels=[-1, 0, 1], alpha=0.7,
                linestyles=['--', '-', '--'])

    # Plot support vectors
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=100, linewidth=1, facecolors='none', edgecolors='k', label = 'Support Vectors')
    
    plt.title("SVM Decision Boundary with Margins")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_svm_decision_boundary(svm_clf, X_scaled, y)
```

---
## 3️⃣ Decision Tree (결정 트리)

- **설명**: 트리 구조의 노드를 따라 데이터의 특징을 기반으로 분기하며 분류를 수행하는 알고리즘.

- **작동 원리**
  1. 정보 이득 등을 기준으로 데이터 분할 기준(feature)을 선택하여 트리의 노드를 구성.
  2. 각 노드는 조건에 따라 데이터를 분기.
  3. 분기를 재귀적으로 반복하여 리프 노드(결정 결과)에 도달.

- **장점**
  - 결과 해석이 용이하며, 시각화에 적합.
  - 정규화나 스케일링 불필요.
  - 범주형 데이터도 잘 처리 가능.

- **단점**
  - 과적합(overfitting)이 자주 발생.
  - 데이터에 민감하여 예측이 불안정할 수 있음.
  - 클래스 불균형에 취약.

```python
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터 준비
X, y = make_moons(n_samples=200, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 2. 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 3. 모델 학습
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train_scaled, y_train)

# 4. 트리 구조 시각화
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=["Feature 1", "Feature 2"], class_names=["Class 0", "Class 1"])
plt.title("Decision Tree Structure")
plt.show()

```

---
# 4️⃣ k-Means 알고리즘

- 설명
  - **k-means**는 비지도학습에 속하는 클러스터링 알고리즘으로, 주어진 데이터를 k개의 그룹(클러스터)로 자동 분류합니다.
  - 각 클러스터는 중심점(centroid)을 기준으로 정의되며, 중심점에 가장 가까운 데이터들을 묶어 그룹화합니다.

- 작동 원리
   1. 초기화 단계 : 데이터 포인트 중 무작위로 k개의 중심점(centroid)을 선택하거나 알고리즘적으로 초기화합니다.
   2.  할당 단계 (Assignment step) : 각 데이터 포인트를 가장 가까운 중심점에 할당하여 클러스터를 형성합니다. (보통 유클리디안 거리 기준)  
   3.  업데이트 단계 (Update step) : 각 클러스터에 속한 데이터의 평균을 계산하여 새로운 중심점으로 갱신합니다.
   4.  반복 : 할당 ↔ 업데이트 단계를 반복하며, 중심점의 위치가 더 이상 바뀌지 않거나 최대 반복 횟수에 도달하면 종료합니다.

- 장점
  - 작동 방식이 **직관적**이고 **구현이 간단**함
  - 대용량 데이터에 대해 **빠르고 효율적**으로 동작
  - 클러스터링 결과가 **명확한 경계와 중심**으로 표현되어 해석 용이
  - 데이터 분포를 **자동 학습**해 군집을 형성하므로 라벨이 없는 경우에도 사용 가능

- 단점
  - 사용자가 클러스터 개수 **k를 사전에 지정**해야 함 (최적 k 선택이 어려움)
  - **초기 중심점** 선택에 따라 결과가 달라질 수 있음 (지역 최솟값에 수렴)
  - 클러스터의 형태가 **구형(원형)에 가까울 때 성능이 좋음**  
    → 비구형·비선형 구조에서는 부정확
  - **이상치(outliers)**의 영향이 큼  
    → 중심 계산 시 왜곡 가능성 있음

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 생성 (4개의 군집 중심으로)
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)    
# 2. k-Means 모델 학습 (k=4)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# 3. 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1], 
            s=200, c='black', marker='X', label='Centroids')
plt.title("k-Means Clustering (k=4)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
```

---
## 요약 비교
| 항목     | kNN               | SVM                | Decision Tree | k-Means             |
| ------ | ----------------- | ------------------ | ------------- | ------------------- |
| 학습 방식  | 메모리 기반 (비훈련)      | 마진 최적화             | 재귀적 분할        | 중심 기반 군집화 (비지도)     |
| 예측 속도  | 느림 (거리 계산 필요)     | 빠름                 | 빠름            | 빠름                  |
| 데이터 적합 | 소규모 데이터, 실시간 업데이트 | 고차원 데이터, 마진 문제에 적합 | 비선형/구조적 데이터   | 클러스터 구조가 뚜렷한 데이터    |
| 과적합 위험 | 중간                | 낮음                 | 높음            | 초기값/클러스터 수 의존       |
| 장점     | 간단, 유연            | 일반화 우수, 마진 최적화     | 직관적, 시각화 가능   | 단순, 빠른 클러스터링 수행     |
| 단점     | 느린 예측, 이상치 민감     | 커널 선택 민감, 느린 학습    | 과적합, 불안정한 구조  | 군집 수 지정 필요, 이상치에 민감 |

---
## 인공지능, 머신러닝, 딥러닝 개념 다시 정리


![image](https://github.com/user-attachments/assets/89225a57-f1ea-4f43-aca7-e12630696991)


- 깊은 학습(deep learning) : 입력된 데이터를 기반으로 특징과 매핑을 신경망을 활용한 학습 방법

- 기계 학습(mechine learning) : 인공신경세포/신경망(neuron, neural network)이 발표된 후 **인간이 설계한 특징**들과 신경망을 활용하여 특징을 결과로 연결(매핑)하여 만드는 학습 방법으로 deeplearning 방법이 포함됨

- 인공 지능(Artificial Intelligence) : 순수 인간의 아이디어에 의해 구성된 **프로그램**만을 사용하여 구현한 내용을 사용하는 방법으로 기계학습 방법과 deep learning 방법이 포함됨

---
# 생물학적 뉴런이란?

![image](https://github.com/user-attachments/assets/a7996f33-3ace-41bb-a1d1-83f3446996b5)


생물학적 뉴런은 **사람이나 동물의 뇌를 구성하는 기본적인 정보 처리 단위**이다.  
뉴런은 외부 또는 다른 뉴런으로부터 신호를 받아들이고, 이를 처리한 후 다음 뉴런으로 전달하는 역할을 수행한다.

---
## 1️⃣ 기본 구조

생물학적 뉴런은 다음과 같은 주요 구성 요소를 가진다:

| 구성 요소 | 기능 |
|-----------|------|
| 수상돌기 (Dendrite) | 다른 뉴런으로부터 **신호를 수신** |
| 세포체 (Cell Body, Soma) | 수상돌기에서 받은 **신호를 통합 및 처리** |
| 축삭 (Axon) | 처리된 신호를 **다른 뉴런으로 전달** |
| 시냅스 (Synapse) | 축삭 말단과 다음 뉴런 사이의 **신호 전달 경로** |

---
## 2️⃣ 작동 원리

1. **신호 수신**:  
   다른 뉴런에서 시냅스를 통해 전달된 신호가 수상돌기로 유입됨.

2. **신호 통합**:  
   세포체에서 여러 입력 신호가 전기적 형태로 합산됨.

3. **임계값 도달**:  
   입력의 총합이 일정 임계값(threshold)을 넘으면, **액션 포텐셜(action potential)**이라 불리는 전기 신호가 생성됨.

4. **신호 전파**:  
   생성된 액션 포텐셜은 축삭을 따라 이동하여 다음 뉴런의 시냅스에 도달함.

---
## 3️⃣ 신호의 강도

- 신호는 **이진적**(발화하거나 발화하지 않음)인 특성을 가지며,
- **시냅스의 강도**는 학습이나 반복 경험에 의해 달라질 수 있음.
  예를 들어, 자주 사용되는 시냅스는 강화되며 이는 **장기 기억 형성**에 기여함.

---
## 4️⃣ 정보 처리 네트워크

- 뇌에는 수십억 개의 뉴런이 존재하며, 이들은 서로 복잡하게 연결되어 있다.
- 이러한 연결망은 감각 처리, 기억, 사고, 감정 등의 **복잡한 인지 활동**을 가능하게 만든다.
---
# 인공신경망과 퍼셉트론 개념 정리

## 인공신경망이란?

인공신경망(Artificial Neural Network, ANN)은 생물학적 뉴런의 구조와 기능에서 영감을 받아 구성된 기계학습 모델이다. **인공지능은 인간의 학습, 추론, 지각 등의 지능적 능력을 컴퓨터가 수행할 수 있도록 만든 기술이다.**  이러한 기능을 구현하기 위해, 사람의 뇌 구조에서 영감을 받은 **인공신경망(Artificial Neural Network)** 이 도입되었으며,  그 기본 단위가 바로 **퍼셉트론(Perceptron)** 이다.  
퍼셉트론은 생물학적 뉴런의 기능을 수학적으로 모방한 모델로, 입력을 받아 가중치를 적용하고 출력으로 전달하는 연산을 수행한다. 퍼셉트론은 딥러닝의 기초이자, 머신러닝에서 분류 문제를 해결하는 지도학습 알고리즘으로 분류된다.

### 🔹 인공신경망 구조 분류 및 퍼셉트론의 역할

| 신경망 유형 | 구성 방식 | 퍼셉트론의 역할 |
|------------|------------|-----------------|
| 단층 퍼셉트론 | 입력층 → 출력층 (1층) | 단일 퍼셉트론으로 구성되어 선형 분류 수행 |
| 다층 퍼셉트론 (MLP) | 입력층 → 은닉층 → 출력층 | 여러 퍼셉트론이 층을 이루며 복잡한 문제 처리 |
| 합성곱 신경망 (CNN) | 입력 → 합성곱층 → 풀링층 → 출력 | 이미지의 지역적 특징을 감지하는 연산 단위로 활용 |
| 순환 신경망 (RNN) | 입력 → 은닉 상태 순환 → 출력 | 시간 순서 정보가 있는 데이터 처리에 사용됨 |

---

# 퍼셉트론 (Perceptron) 정의

퍼셉트론은 인공 신경망의 가장 기본적인 단위로, 생물학적 뉴런의 동작을 모방한 이진 분류 모델이다.  
1958년 프랭크 로젠블렛(Frank Rosenblatt)이 제안한 고전적인 단층 신경망 모델로, 입력값의 선형 조합을 통해 결과를 이진 분류한다. 단층 퍼셉트론은 오직 이진 분류 문제만 해결할 수 있으며, 다중 클래스 분류에는 직접 적용하기 어렵다.

---

## 단층 퍼셉트론의 구조 및 수식

| 구성 요소                          | 설명                            |
| ------------------------------ | ----------------------------- |
| 입력값 \( x_1, x_2, ..., x_n \)   | 각 입력 변수 (예: 픽셀 값, 특징값 등)      |
| 가중치 \( w_1, w_2, ..., w_n \)   | 입력의 중요도를 조절하는 계수              |
| 바이어스 \( b \)                   | 결정 경계를 이동시키는 값                |
| 총합 (z = w₁x₁ + w₂x₂ + ... + b) | 가중합                           |
| 활성화 함수 \( f(z) \)              | 출력값을 이진으로 변환 (보통 단위 계단 함수 사용) |
- 가중치와 바이어스란?
	- **가중치(weight)**는 입력값의 중요도를 조절하는 계수로, 학습을 통해 조정된다.
	- **바이어스(bias)**는 모든 입력이 0일 때도 출력이 특정 기준을 넘도록 도와주는 보정값이다.
	- 퍼셉트론의 결정 경계를 유연하게 이동시켜주는 중요한 파라미터이다.

- 활성화 함수란?
	- **활성화 함수**는 뉴런이 출력 신호를 생성할지 결정하는 함수이다.
	- 입력의 총합 \( z \)가 임계값 이상이면 1, 아니면 0을 출력하는 **계단 함수**가 단층 퍼셉트론에서 사용된다.
	- 다층 퍼셉트론에서는 **비선형 활성화 함수**(예: ReLU, sigmoid, tanh 등)가 사용된다.

단층 퍼셉트론의 출력은 다음과 같이 표현된다.

$$
\text{output} = f\left(\sum_{i=1}^{n} w_i x_i + b \right)
$$

보통 사용되는 활성화 함수는 **계단 함수**(step function)이다. 이 함수는 출력값을 이진으로 나누어, 선형 결정 경계를 형성하는 데 사용된다.

$$
f(z) = 
\begin{cases}
1, & \text{if } z \ge 0 \\
0, & \text{otherwise}
\end{cases}
$$

---

## 단층 퍼셉트론의 학습 방법 (Perceptron Learning Rule)

퍼셉트론은 **오차 기반 학습법**을 통해 가중치를 수정한다.

가중치 업데이트 식:

$$
w_i \leftarrow w_i + \eta \cdot (y - \hat{y}) \cdot x_i
$$

바이어스 업데이트 식:

$$
b \leftarrow b + \eta \cdot (y - \hat{y})
$$

여기서

- η (eta): 학습률 (learning rate)  
- y: 실제 정답 (target label)  
- ŷ (y-hat): 모델의 예측값 (predicted output)을 의미한다.


---

## 간단한 예제 (AND 문제)

```python
import numpy as np
import matplotlib.pyplot as plt

# AND 데이터
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 0, 0, 1])

# 초기값 설정
w = np.zeros(2)
b = 0
lr = 0.1

# 정확도 기록용 리스트
acc_list = []

# 학습
for epoch in range(10):
    correct = 0
    for i in range(len(X)):
        z = np.dot(X[i], w) + b
        y_hat = 1 if z >= 0 else 0
        error = y[i] - y_hat

        if y[i] == y_hat:
            correct += 1

        # 업데이트
        w += lr * error * X[i]
        b += lr * error

    acc = correct / len(X)
    acc_list.append(acc)
    print(f"Epoch {epoch+1}: Accuracy = {acc*100:.0f}%, w = {w}, b = {b}")

# 최종 결과
print("\n최종 학습된 가중치:", w)
print("최종 학습된 바이어스:", b)

# 시각화
def plot_decision_boundary(X, y, w, b):
    plt.figure(figsize=(6, 5))
    for i in range(len(X)):
        if y[i] == 0:
            plt.plot(X[i][0], X[i][1], "ro")
        else:
            plt.plot(X[i][0], X[i][1], "go")

    # 결정 경계 직선 (w1*x1 + w2*x2 + b = 0)
    x1 = np.linspace(-0.2, 1.2, 100)
    x2 = -(w[0]*x1 + b) / w[1]
    plt.plot(x1, x2, "b--", label="Decision Boundary")

    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("AND Perceptron Classification")
    plt.grid(True)
    plt.legend()
    plt.show()

plot_decision_boundary(X, y, w, b)
```

---

## XOR 문제와 단층 퍼셉트론의 한계

XOR(배타적 논리합) 문제는 단층 퍼셉트론이 해결할 수 없는 대표적인 **비선형 분류 문제**이다.
XOR 논리는 다음과 같이 정의된다:

| 입력 \( x_1 \) | 입력 \( x_2 \) | 출력 \( y \) |
|----------------|----------------|---------------|
| 0              | 0              | 0             |
| 0              | 1              | 1             |
| 1              | 0              | 1             |
| 1              | 1              | 0             |

XOR은 두 입력이 서로 다를 때만 출력이 1이 되므로, 두 클래스를 하나의 직선으로 나눌 수 없다.  
즉, **선형 결정 경계**로 분리할 수 없는 구조이기 때문에 단층 퍼셉트론으로는 학습이 불가능하다.

---
### 단층 퍼셉트론으로 XOR 학습 (실패 예시)

```python
import numpy as np

# XOR 데이터셋
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 1, 1, 0])

# 초기 설정
w = np.zeros(2)
b = 0
lr = 0.1

# 학습
for epoch in range(10):
    correct = 0
    for i in range(len(X)):
        z = np.dot(X[i], w) + b
        y_hat = 1 if z >= 0 else 0
        error = y[i] - y_hat

        if y[i] == y_hat:
            correct += 1

        # 가중치, 바이어스 업데이트
        w += lr * error * X[i]
        b += lr * error

    acc = correct / len(X)
    print(f"Epoch {epoch+1}: Accuracy = {acc*100:.0f}%, w = {w}, b = {b}")

```

---
##  단층 퍼셉트론의 한계

- 단층 퍼셉트론은 **선형 분리 가능한 문제만 해결 가능** (예: AND, OR)
- **XOR 문제는 해결할 수 없음**
- 이 한계를 극복하기 위해 **다층 퍼셉트론 (MLP)**, 즉 **은닉층이 있는 신경망**이 등장

---

## 과제 1) OR 연산을 단층 퍼셉트론으로 구현하고 결정 경계를 시각화하시오

### 설명

다음 조건에 따라 OR 연산을 수행하는 퍼셉트론을 구현하고, 학습 결과를 시각화하시오.
``
- 입력값 ( `X = [[0,0], [0,1], [1,0], [1,1]]` )
- 출력값 ( `y = [0, 1, 1, 1]` )
- 초기 가중치 ( `w = [0.0, 0.0]` ), 바이어스 ( `b = 0.0` )
- 학습률 ( `η (eta) = 0.1` )
- 학습 반복 횟수: 10회

### 요구사항

1. 퍼셉트론 학습 알고리즘을 구현하여 OR 문제를 해결하시오.
2. 학습 도중 정확도와 가중치, 바이어스를 출력하시오.
3. 학습이 완료된 후, 결정 경계를 시각화하시오.
