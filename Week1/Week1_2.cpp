# ========================================
# 📘 [수업용 코드] 패키지와 import 설명
# ========================================

# 🔹 Step 1: 기본 import 문법 설명
# 파이썬 표준 라이브러리 모듈 예시 (math)
import math

print("원주율:", math.pi)
print("제곱근(16):", math.sqrt(16))

# 🔹 Step 2: as 키워드로 별칭 사용하기
import math as m

print("로그(10):", m.log(10))

# 🔹 Step 3: 특정 함수만 import 하기 (from-import)
from math import pow, factorial

print("2의 3제곱:", pow(2, 3))
print("5!: ", factorial(5))

# ========================================
# 📘 [수업용 코드] 외부 패키지 사용과 numpy 기초
# ========================================

# 🔹 Step 4: 외부 패키지 설치 필요 (ex. numpy)
# (코랩에서는 이미 설치되어 있어 따로 설치할 필요는 없음)
import numpy as np

# numpy 배열 생성
a = np.array([1, 2, 3, 4])
print("numpy 배열 a:", a)

# 🔹 Step 5: numpy 배열 연산
b = np.array([10, 20, 30, 40])
print("a + b:", a + b)
print("a * 2:", a * 2)

# 🔹 Step 6: numpy 배열의 통계적 함수
print("a의 평균:", np.mean(a))
print("a의 합:", np.sum(a))
print("a의 표준편차:", np.std(a))

# ========================================
# 📝 [과제용 코드] numpy 배열 실습
# ========================================

"""
[문제 설명]
1. 사용자로부터 정수를 쉼표(,)로 구분된 형태로 입력받아 numpy 배열로 변환합니다.
2. 이 배열의 다음 값을 출력하세요:
   - 최대값
   - 최소값
   - 평균
   - 표준편차
   - 배열의 각 요소에 5를 더한 새 배열
"""

  
# 답

import numpy as np

user_input = input("숫자들을 쉼표로 구분해서 입력하세요 (예: 10,20,30): ")
numbers = np.array([int(x) for x in user_input.split(',')])

print("입력한 배열:", numbers)

# 여기서부터 학생이 직접 작성
print("최댓값:", np.max(numbers))
print("최솟값:", np.min(numbers))
print("평균:", np.mean(numbers))
print("표준편차:", np.std(numbers))

new_array = numbers + 5
print("각 요소에 5를 더한 새 배열:", new_array)
