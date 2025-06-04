# 📘 Week2-1 - pandas, 파일 시스템, 데이터셋

---

## 📂 파일 시스템 이해

### 🔹 디렉토리와 경로
- 상대경로: 현재 작업 디렉토리를 기준으로 한 경로
- 절대경로: 루트(`/`)부터 시작하는 전체 경로

### 🔹 파일 입출력 기본
```python
# 파일 쓰기
with open("example.txt", "w") as f:
    f.write("Hello, file!")

# 파일 읽기
with open("example.txt", "r") as f:
    content = f.read()
    print(content)
```

---

## 🐼 pandas 기초

### 🔹 pandas 소개
- 표 형태의 데이터 처리를 위한 파이썬 라이브러리
- 주요 객체: `Series`, `DataFrame`

### 🔹 Series 생성
```python
import pandas as pd
s = pd.Series([10, 20, 30])
print(s)
```

### 🔹 DataFrame 생성
```python
data = {
    '이름': ['철수', '영희', '민수'],
    '점수': [90, 85, 78]
}
df = pd.DataFrame(data)
print(df)
```

---

## 🛠️ pandas 주요 기능

### 🔹 데이터 탐색
```python
print(df.head())      # 상위 5개 행
print(df.tail())      # 하위 5개 행
print(df.info())      # 요약 정보
print(df.describe())  # 기초 통계량
```

### 🔹 열 선택 및 조건 필터링
```python
print(df['이름'])                  # 특정 열 선택
print(df[df['점수'] > 80])         # 조건 필터링
```

---

## 📁 실습: CSV 파일 불러오기
```python
# CSV 파일 불러오기
csv_df = pd.read_csv("./sample_data.csv")
print(csv_df.head())
```

### 🔹 특정 열만 보기
```python
print(csv_df['컬럼명'])
```

### 🔹 조건으로 행 필터링
```python
print(csv_df[csv_df['컬럼명'] > 값])
```

---

## 🧪 과제
1. `student_scores.csv` 파일을 pandas로 불러오고 상위 3개 데이터를 출력하시오.
2. 평균 점수가 80점 이상인 학생만 출력하시오.
3. '수학' 점수 열만 출력하시오.


---

✅ 사용 환경: Python 3.x, Google Colab 권장
