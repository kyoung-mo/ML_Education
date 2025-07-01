# 📘 Week2-1 – pandas, 파일 시스템, 데이터셋 

---

## 패키지 관리 도구: pip & conda

# 📦 pip vs conda 정리

Python 개발 환경에서 패키지를 설치하고 관리하기 위한 대표적인 도구는 `pip`과 `conda`입니다. 각각의 특징과 사용법을 아래와 같이 정리합니다.


## 🐍 pip (Python Package Installer)

### ✅ 개요
- Python 표준 패키지 관리자
- `PyPI`(Python Package Index)에 등록된 수많은 패키지 설치 가능
- 가볍고 범용적인 설치 도구

### ✅ 주요 명령어

| 명령어 | 설명 |
|--------|------|
| `pip install 패키지명` | 패키지 설치 |
| `pip install --upgrade 패키지명` | 패키지 업그레이드 |
| `pip uninstall 패키지명` | 패키지 제거 |
| `pip list` | 설치된 패키지 목록 확인 |
| `pip show 패키지명` | 특정 패키지 정보 확인 |

### ✅ 장점
- 설치 속도가 빠르고 가볍다
- PyPI의 다양한 패키지에 접근 가능
- Colab, Jupyter 등 대부분의 환경에서 기본 제공

### ⚠️ 단점
- 패키지 간 의존성 충돌 가능성이 있음
- 가상환경을 별도로 구성해 사용하는 것이 안전

---

## 🐾 conda (Anaconda Package Manager)

### ✅ 개요
- Anaconda 배포판에서 사용하는 패키지 & 환경 관리 도구
- 패키지 설치뿐 아니라 **Python 버전 및 가상환경 전체를 통합 관리** 가능

### ✅ 주요 명령어

| 명령어 | 설명 |
|--------|------|
| `conda create -n 환경명 python=버전` | 새로운 가상환경 생성 |
| `conda activate 환경명` | 환경 활성화 |
| `conda deactivate` | 환경 비활성화 |
| `conda install 패키지명` | 패키지 설치 |
| `conda list` | 설치된 패키지 목록 확인 |

### ✅ 장점
- Python 버전까지 통합 관리 가능
- 과학·수치계산 라이브러리 설치 시 안정적
- 의존성 자동 해결 기능이 탁월

### ⚠️ 단점
- 초기 설치 용량이 크고 무겁다
- 실행 속도가 pip보다 다소 느림

---

## 📊 비교 요약

| 항목 | pip | conda |
|------|-----|--------|
| 사용 환경 | Python 기본 | Anaconda 전용 |
| 패키지 소스 | PyPI | Anaconda Repository, conda-forge |
| 가상환경 지원 | 별도 도구 필요 (venv) | 기본 내장 |
| 의존성 해결 | 낮음 | 높음 |
| 설치 속도 | 빠름 | 다소 느림 |
| 과학 계산 패키지 | 제한적 | 강력 (numpy, scipy 등 포함) |

---

## ✅ 정리
- **간단하고 빠른 설치**가 필요하면 `pip` 사용
- **환경 전체를 안정적으로 구성**하려면 `conda` 사용

> 실무에서는 둘을 **혼용**하기도 하며, `conda` 환경에서 `pip` 명령도 사용할 수 있습니다.

---

# 📂 파일 시스템 이해 – 절대경로 vs 상대경로

파일을 다루는 대부분의 프로그램은 **경로(path)** 를 사용하여 특정 파일이나 디렉토리를 참조합니다. Python에서도 `open()`, `pd.read_csv()`, `os` 모듈 등에서 경로를 지정하는 것이 매우 중요합니다.

---

## 📌 경로(Path)란?

경로는 컴퓨터 내의 파일이나 폴더의 **위치 정보를 나타내는 문자열**입니다. 경로를 통해 운영체제나 프로그램은 원하는 파일에 접근할 수 있습니다.

경로에는 다음 두 가지 방식이 존재합니다:

---

## ✅ 절대 경로 (Absolute Path)

- 루트 디렉토리(`/`) 또는 드라이브 문자(`C:/` 등)부터 시작하는 **전체 경로**
- **항상 동일한 파일 위치**를 참조하며, 어디서 실행하든 같은 파일을 가리킴

### 예시 (Linux/Mac 기준)
```text
/home/user/project/data/sample.csv
```

### 예시 (Windows 기준)
```text
C:\Users\user\project\data\sample.csv
```

---

## ✅ 상대 경로 (Relative Path)

- 현재 작업 디렉토리(Working Directory)를 기준으로 파일의 위치를 지정
- **프로그램 실행 위치에 따라 달라질 수 있음**
- 이동성과 유연성이 좋아 프로젝트 내부 파일 참조 시 자주 사용됨

### 상대경로 표현 기호

| 기호 | 의미 |
|------|------|
| `.` | 현재 디렉토리 |
| `..` | 상위 디렉토리 |
| `./data/file.txt` | 현재 폴더 아래 `data` 폴더 |
| `../data/file.txt` | 상위 폴더의 `data` 폴더 |

---

## 🧪 예제: 절대경로 vs 상대경로

```python
import os
import pandas as pd

# 현재 작업 디렉토리 확인
print("현재 위치:", os.getcwd())

# 절대경로로 파일 읽기
abs_path = "/content/sample_data/california_housing_train.csv"
df_abs = pd.read_csv(abs_path)
print("절대경로 읽기 성공")

# 상대경로로 파일 읽기
rel_path = "./sample_data/california_housing_train.csv"
df_rel = pd.read_csv(rel_path)
print("상대경로 읽기 성공")
```

> ⚠️ 실제 경로는 여러분의 실행 환경에 맞게 수정해야 합니다. Colab에서는 `/content/`가 루트 경로입니다.

---

## 🧠 정리

| 항목 | 절대경로 | 상대경로 |
|------|-----------|-----------|
| 기준 위치 | 루트(`/`) 또는 드라이브 | 현재 작업 디렉토리 |
| 이동성 | 낮음 (시스템 고정) | 높음 (프로젝트 내부에 적합) |
| 예시 | `/home/user/data.csv` | `../data/data.csv` |
| 사용 용도 | 외부 파일, 시스템 경로 | 프로젝트 내부 파일 처리 |

---

> 📝 실무 팁:  
> - 팀 프로젝트나 공유 코드는 상대경로를 사용하는 것이 좋습니다.  
> - 경로가 꼬이지 않도록 항상 `os.getcwd()`로 현재 위치를 확인하고 디버깅하세요!


---


## 🐼 pandas 기초

### 🔹 pandas 소개
- **pandas**는 **관계형·라벨형 데이터**(표, 시계열 등)를 쉽고 빠르게 다루기 위한 라이브러리입니다.  
- 메모리에 로드된 데이터를 **엑셀처럼** 편집하면서도, **SQL처럼** 질의·집계할 수 있다는 장점이 있습니다.  
- 핵심 객체는 **`Series`(1차원)와 `DataFrame`(2차원)** 두 가지입니다.

### 🔹 왜 pandas인가?
| 기능 | 엑셀 | SQL | pandas |
|------|------|-----|--------|
| 대용량 처리 | ❌ 수십만 행에서 느려짐 | ✅ | ✅ (메모리 제한 내) |
| 복잡한 계산 | ❌ | ✅ | ✅ |
| 인터랙티브 탐색 | ✅ | ❌ | ✅ (Jupyter/Colab) |

---

### 🔹 Series(1차원)
```python
import pandas as pd

s = pd.Series([10, 20, 30], index=['A', 'B', 'C'])
print(s)
print(s.index)   # Index(['A', 'B', 'C'], dtype='object')
print(s.values)  # array([10, 20, 30])
```
*Series는 값(value)과 인덱스(index)가 한 쌍으로 묶인 1차원 배열*입니다.  
넘파이 배열과 달리 **인덱스를 활용한 라벨 기반 접근**이 가능합니다.

---

### 🔹 DataFrame(2차원)
```python
data = {
    "이름": ["철수", "영희", "민수"],
    "국어": [90, 85, 78],
    "수학": [95, 88, 82]
}
df = pd.DataFrame(data)
df
```
`DataFrame`은 **여러 개의 Series가 같은 인덱스를 공유**하며 모인 **표 형태**의 자료구조입니다.  
열(column)마다 자료형이 달라도 되므로 SQL 테이블과 유사한 개념입니다.

### 🔹 DataFrame은 numpy처럼 작동

- `DataFrame`은 내부적으로 **numpy 배열 기반**이기 때문에 **벡터 연산, 조건 연산, 브로드캐스팅**이 가능하다.

```python
import pandas as pd

df = pd.DataFrame({
    "국어": [90, 85, 78],
    "수학": [95, 88, 82]
}, index=["철수", "영희", "민수"])

# 열 단위 연산: 평균 점수 계산 (각 행에 대해)
df["평균"] = (df["국어"] + df["수학"]) / 2

# 브로드캐스팅: 수학 점수에 5점 가산
df["수학+보너스"] = df["수학"] + 5

print(df)
```

**출력 예시**:

```
      국어  수학   평균  수학+보너스
철수   90  95  92.5      100
영희   85  88  86.5       93
민수   78  82  80.0       87
```

- `numpy`처럼 연산자를 그대로 사용해 계산이 가능하며, 열 단위로 자동 정렬됨.
- `df["평균"] = np.mean(df, axis=1)`처럼 `numpy` 함수도 그대로 적용 가능.

> **Note**: `DataFrame`은 `numpy`의 2차원 배열 + 라벨 기능을 추가한 구조로 이해하면 됩니다.


---

## 🛠️ pandas 사용법 한눈에 보기

| 기능 | 메서드(예시) | 설명 |
|------|-------------|------|
| 행·열 미리 보기 | `df.head(3)`, `df.tail()` | 상·하단 일부 출력 |
| 행·열 선택 | `df['열']`, `df.loc[행라벨, '열']`, `df.iloc[행번호, 열번호]` | 라벨/정수 기반 인덱싱 |
| 조건 필터링 | `df[df['점수'] > 80]` | 불린 마스크 |
| 정렬 | `df.sort_values('점수', ascending=False)` | 값 기준 정렬 |
| 통계 요약 | `df.describe()`, `df.mean()` | 기초 통계량 |
| 결측치 처리 | `df.isna()`, `df.dropna()`, `df.fillna(값)` | NA 탐지/삭제/대체 |
| 그룹 분석 | `df.groupby('반')['점수'].mean()` | 그룹별 집계 |
| 데이터 병합 | `pd.concat([df1, df2])`, `pd.merge(df1, df2, on='키')` | 행·열 이어붙이기 / 조인 |
| 파일 입출력 | `pd.read_csv()`, `df.to_excel()` | CSV, Excel, JSON 등 |

---

### 🔹 인덱싱 & 슬라이싱 심화

```python
# 라벨 기반
df.loc[0, '국어']      # 첫 번째 행의 국어 점수
df.loc[:, '수학']       # 모든 행의 수학 열

# 정수 위치 기반
df.iloc[0, 1]          # [행 0, 열 1] 값

# 다중 조건
subset = df[(df['국어'] > 80) & (df['수학'] > 85)]
```

> **Tip**   
> `loc`은 라벨, `iloc`은 정수 위치를 사용합니다. 헷갈릴 때는 “L = label, I = integer”라고 기억하세요!

---

### 🔹 결측치(NA) 다루기

```python
df['영어'] = [88, None, 91]  # None은 NA로 인식
df.isna().sum()              # 열별 NA 개수 확인
df_filled = df.fillna(df.mean(numeric_only=True))  # 평균으로 대체
```

---

### 🔹 그룹별 집계와 피벗 테이블

```python
group_mean = df.groupby('이름')['국어'].mean()
pivot = df.pivot_table(index='이름', values=['국어', '수학'], aggfunc='mean')
```
- **`groupby`**: SQL의 `GROUP BY`와 동일.  
- **`pivot_table`**: 엑셀 피벗 테이블처럼 다차원 요약.

---

### 🔹 외부 파일 입출력

| 형식 | 읽기 | 쓰기 |
|------|------|------|
| CSV | `pd.read_csv('data.csv')` | `df.to_csv('out.csv', index=False)` |
| Excel | `pd.read_excel('data.xlsx')` | `df.to_excel('out.xlsx', index=False)` |
| JSON | `pd.read_json('data.json')` | `df.to_json('out.json')` |

---

## 📁 실습: CSV 불러와서 분석하기

# 📄 CSV란?

**CSV**(Comma-Separated Values)는 데이터를 **쉼표(,)**로 구분하여 저장하는 **텍스트 파일 형식**입니다.  
표 형식 데이터를 간단하고 효율적으로 저장하거나 다양한 프로그램 간 데이터 교환에 사용됩니다.

예를 들어:



```python
import pandas as pd

# 1) CSV 로드
csv_df = pd.read_csv("./student_scores.csv")

# 2) 상위 3개 확인
print(csv_df.head(3))

# 3) 평균 >= 80 필터
print(csv_df[csv_df['평균'] >= 80])

# 4) 수학 점수만
print(csv_df['수학'])
```

---

## 🧪 과제
1. `student_scores.csv` 파일을 pandas로 불러오고 상위 3개 데이터를 출력하시오.  
2. 평균 점수가 80점 이상인 학생만 출력하시오.  
3. **'수학'** 점수 열만 출력하시오.  

> **힌트**   
> • `pd.read_csv()` → DataFrame 로드  
> • `head()`  
> • 조건 필터링 `df[df['평균'] >= 80]`  
> • 열 선택 `df['수학']`

---

✅ **사용 환경**: Python 3.x, Google Colab 권장  
`!pip install pandas` 로 설치 가능 (Colab은 기본 내장)
