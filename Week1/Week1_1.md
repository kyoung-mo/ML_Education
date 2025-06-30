# Python
## 프로그래밍 언어
컴퓨터는 실제 사람이 사용하는 언어와 다른 형태의 언어인 **기계어**를 사용한다.

**0010 0110 1111 0001** (기계어) <-> **안녕하세요** (사람의 언어)

초기의 개발자들은 기계어를 사용하여 프로그램을 만들었지만, 현대의 개발자들은 **프로그래밍 언어**를 사용하여 프로그램을 만든다. 프로그래밍 언어는 컴퓨터 프로그램을 작성할 때 사용하는 **형식적인 언어**로, 특정 문법과 규칙을 갖고 있으며 컴퓨터가 이해하고 실행할 수 있는 명령어의 조합이다. 프로그래밍 언어는 다음과 같이 분류할 수 있다.

- **고급 언어 vs 저급 언어**
    
    - **고급 언어**: 사람이 이해하기 쉬운 언어 (예: Python, Java, C++)
        
    - **저급 언어**: 기계 가까운 언어 (예: 어셈블리어, 기계어)
        
- **컴파일 언어 vs 인터프리터 언어**
    
    - **컴파일 언어**: 전체 코드를 기계어로 번역한 후 실행 (예: C, C++)
        
    - **인터프리터 언어**: 코드를 한 줄씩 해석하며 실행 (예: Python, JavaScript)
        
- **절차지향 vs 객체지향**
    
    - **절차지향**: 명령어를 순서대로 실행 (예: C)
        
    - **객체지향**: 데이터를 객체로 묶어 구성 (예: Java, Python)
## Python
Python은 쉽고 직관적인 문법을 갖춘 **고급 객체지향 인터프리터 언어**로, 웹 개발부터 인공지능, 회로 설계까지 다양한 분야에서 사용된다. 
- 다른 언어에 비해 코드가 간결하고 읽기 쉬움
- 인터프리터 언어 : 코드를 한 줄씩 실행하므로 빠른 테스트와 디버깅이 가능. 
- pip 패키지 관리 시스템을 사용하여 풍부한 라이브러리 사용 가능
- Windows, Mac, Linux 등 어떤 운영체제에서도 실행 가능
# 개발환경
## Colab
본 동아리에서는 개발을 위해 Google에서 제공하는 무료 온라인 Python 실행 환경인 **Google Colaboratory**을 사용한다. 별도 설치 없이 웹브라우저만 있으면 누구나 바로 Python 코드를 실행할 수 있다. 

https://colab.research.google.com/ 해당 웹페이지에 접속하여 Colab을 사용할 수 있다. Colab에서 작성한 코드는 모두 Google Drive에 저장 가능하다. 좌측 상단에 파일 - Drive의 새 노트북 - 로그인을 통해 Google Drive에 새 파일을 생성하라.
![[Pasted image 20250523193531.png]]
## Jupyter Notebook
이전에 프로그래밍 경험이 있는 개발자라면 해당 개발 환경이 컴파일 언어와는 다르다는 것을 느꼈을 것이다. 보통의 프로그래밍 언어의 경우 IDE Tool을 사용하여 스크립트를 작성하고 해당 스크립트를 통째로 실행할 것이다. 그러나 Python은 인터프리터 언어고, 한 줄씩 코드를 실행할 수 있기 때문에 다른 언어와는 다른 추가적인 기능을 사용할 수 있다.

**Jupyer Notebook**은 이러한 기능을 활용한 개발 환경으로 코드를 Cell 단위로 실행할 수 있고 일반적인 Python 스크립트 확장자 `.py` 가 아닌 `.ipynb` 확장자를 사용한다. Colab에서는 기본적으로 Notebook 파일을 작성 및 실행할 수 있다.

![[Pasted image 20250523230857.png]]
Colab 상단 좌측의 파일명을 수정하여 Notebook 파일명을 바꿀 수 있다. 

이제 직접 코드를 작성하여 Cell을 사용하자. 첫 번째 셀에 `a = 1` 코드를 작성하여 실행시켜라. Cell 좌측의 화살표 버튼을 클릭하여 실행할 수 있다. (또는,  `Shift + Enter`) 이후, 새로운 Cell에서 `print(a)` 코드를 실행시켜 a변수의 값을 확인하라. 

두 번째 Cell의 출력은 10이 나온다. 즉, 하나의 Notebook 파일 내에선 각 Cell들 간에 실행 내역을 공유한다. 이전 Cell에서 선언한 변수, 함수 등을 다른 셀에서 접근할 수 있으며, Cell의 배치 순서가 아닌 **실행 순서**대로 코드가 실행된다.
# 자료구조
Python은 다양한 자료형과 자료구조를 제공하여 데이터를 효율적으로 저장하고 처리할 수 있다. 이 섹션에서는 파이썬에서 자주 사용되는 기본 자료구조에 대해 설명한다.
## 변수 (Variable)
변수는 메모리에서 데이터를 저장하는 공간을 지칭한다. 변수에 데이터를 넣는 것을 **선언**한다고 한다. 
- C 나 Java 등 고전적인 프로그래밍 언어에서는 변수 선언 시 변수의 **데이터 타입**을 명시해야 했다. 
- Python은 데이터 타입을 명시하지 않고 선언이 가능하다. 이를 **동적 타이핑(Dynamic Typing)** 이라고 한다.

변수에 들어간 데이터를 읽어서 다양한 연산을 수행할 수 있다. 이를 변수에 **접근**한다고 한다.
## 원시 자료형 (Primitive Data Type)
원시 자료형은 가장 기본적이고 단순한 형태의 데이터 타입을 말한다. 프로그래밍 언어에서 **더 이상 분할할 수 없는 단일 값**을 표현할 때 사용한다.

- `int` : 정수형. 가장 기본적인 자료형으로 1, -0, 4 같은 **정수**를 표현하는 데 사용한다.
- `float` : 실수형. 부동소수점 숫자를 사용하여 3.14, -0.13 같은 **실수**를 표현하는 데 사용한다.
- `bool` : Boolean. 참/거짓을 표현하기 위한 자료형으로 오직 2개의 상태를 갖는다.
- `str` : 문자열. `"Hello"`,`'Hi'` 같은 단어나 문장을 표현하기 위한 자료형이다. C언어 처럼 **인덱스로 접근이 가능하다.**
- `NoneType` : 값이 없음을 표현. 변수의 초기값이나 함수에서 반환할 값이 없는 경우 사용되는 **특별한 상수**로 취급된다.

``` python
int_Type = 10
float_Type = 3.14
bool_Type = True
str_Type = '안녕하세요.'
None_Type = None
```

변수를 선언할 때는 해당 코드처럼 괄호를 사용하여 선언할 수 있다. 선언된 변수를 확인하는 가장 빠른 방법은 `print` 함수를 사용하는 것이다. `print` 함수는 인자로 들어간 값을 화면(Console)에 출력하는 함수다.

``` python
print(int_Type)
print(float_Type)
print(bool_Type)
print(str_Type)
print(None_Type)
```

`print` 함수를 사용하여 변수를 확인하라. 

고전적인 프로그래밍 언어에서는 연산할 변수끼리 같은 자료형이어야 했지만, Python에서는 자료형을 **자동 변환(coercion)** 하거나 **직접 변환(casting)** 할 수 있다.

- 자동 변환의 예시
``` python
a = int_Type + float_Type
print(a)
b = int_Type + bool_Type
print(b)
```

- 직접 변환의 예시
``` python
c = str_Type + str(float_Type)
print(c)
d = str_Type + str(bool_Type)
print(c)
```

이처럼 문자열의 경우 직접 변환을 사용하여 다른 자료형과의 연산이 가능하다. **문자열을 다른 자료형으로 변환하는 것도 가능하다.**
## 컬렉션 자료형 (Collection Data Type)
C의 구조체, 배열 또는 Java의 ArrayList, HashMap처럼 Python에서도 **여러 자료형을 묶어서 저장**할 수 있는 자료형이 존재한다. 

컬렉션 자료형에는 `list, tuple, dict, set` 등이 있다. 
### 리스트 (List)
리스트는 여러 자료형을 **순차적으로 저장할 수 있는 자료형**이다. **리스트 내 요소들은 변경이 가능**하며, `[]`를 사용하여 선언이 가능하다.

``` python
a = 10
list_type = [1, 3.14, "머신러닝", True, a]
print(list_type)
```

리스트는 다양한 메소드를 활용하여 요소를 추가, 제거, 정렬할 수 있다.

``` python
list_type.append(99)
print(list_type)
list_type.remove("머신러닝",True)
print(list_type)
list_type.sort()
print(list_type)
```
- `.append(A)` : 리스트의 마지막에 `A` 를 추가한다.
- `.remove(A)` : 리스트의 가장 첫번째로 나오는 `A` 를 제거한다.
- `.sort()` : 리스트를 크기 순서대로 정렬한다. 문자열과 NoneType이 없을 때 사용 가능하다.
### 튜플 (Tuple)
튜플은 **여러 자료형을 순차적으로 저장하지만, 요소 변경이 불가능한 자료형**이다. `()`를 사용하여 선언이 가능하다.

``` python
a = 10
tuple_type = (1, 3.14, "머신러닝", True, a)
print(tuple_type)
```

리스트와 유사하지만, 튜플은 불변 자료형이므로 사용할 수 있는 메소드가 많지 않다.

``` python
print(tuple_type.count(1))
print(tuple_type.index(3.14))
```
- `.count(A)` : 튜플에 포함된 `A` 의 갯수를 반환한다.
- `.index(A)` : 튜플에 포함된 첫번째 `A` 의 인덱스를 반환한다.
### 딕셔너리 (Dict)
딕셔너리는 **키-값(key-value) 쌍**을 저장하는 자료형으로 Java의 `HashMap` 과 같은 개념이다. `{}`와 `':',','`를 사용하여 선언이 가능하다.

키에 해당하는 값을 빠르게 찾을 수 있으므로 데이터 베이스, 사용자 설정 등 다양한 용도로 활용된다. 기본적으로 하나의 키는 하나의 값과 1대1 매칭된다.

``` python
person = {
    "name": "지민구",
    "age": 26,
    "height": 180.1
    479 : "inform"
}
```

키는 반드시 불변해야 하므로 **원시자료형이나 튜플**을 사용해야 하고, 중복된 키를 허용하지 않는다. 

``` python
print(person.keys())  
print(person.values())
print(person.items())
print(person.get("age"))        
print(person.get("job"))
print(person.get("job", "unknown"))
person["job"] = "Programmer"
print(person)
```

- `.keys()` : 딕셔너리의 모든 키를 리스트로 묶어 반환한다.
- `.values()` : 딕셔너리의 모든 값을 리스트로 묶어 반환한다.
- `.items()` : 딕셔너리의 키와 값을 각각 튜플로 묶은 후 리스트로 묶어 반환한다.
- `.get(A, B)` : 딕셔너리에 `A` 키에 해당하는 값을 반환한다. `A` 가 없을 경우 `B` 를 반환하고, `B` 를 인가하지 않으면 `None` 을 반환한다.
- 딕셔너리에 `[]` 를 사용하여 새로운 키-값을 추가할 수 있다.
### 집합 (Set)
집합은 중복을 허용하지 않고 순서가 없는 자료형이다. 수학의 집합과 유사하게 작동하며, 교집합/합집합/차집합 같은 연산에 유용하게 사용된다. 

서로 다른 자료형도 포함될 수 있지만 불변 자료형만 집합에 포함될 수 있다. 

``` python
list_for_set = [5,2,6,2,7,1,1]
str_for_set = "Hello"
tuple_for_set = (6,2,3,3,2,1)
a = set(list_for_set)
b = set(str_for_set)
c = set(tuple_for_set)
print(a)
print(b)
print(c)

ex_set = {4, "Love", 5.68, False}
print(ex_set)
```

 `set()` 함수를 사용하여 리스트, 문자열, 튜플 등의 자료형을 기반으로 집합을 생성하거나 `{}` 를 사용하여 집합을 생성할 수 있다.

``` python
X = {1, 2, 3}
Y = {1, 2, 3, 4}

Y.add(5)
print(Y)
Y.remove(5)
print(Y)
Z = X & Y
print(Z)
print(X.issubset(Y))
```

- `.add(A)` : 집합에 `A` 요소를 추가한다.
- `.remove(A)` : 집합에서 `A` 요소를 제거한다.
- `X & Y or X.union(Y)`: 집합 `X` 와 집합 `Y` 의 교집합을 연산하여 반환한다.
- `X.issubset(Y)`: 집합 `X` 가 집합 `Y` 의 부분집합이라면 `True`를 반환하고 아니라면 `False`를 반환한다.
### 인덱스 (Index)
문자열, 리스트, 튜플처럼 값들이 순서대로 나열된 자료형을 시퀀스 자료형이라고 한다. 인덱스는 이러한 시퀀스 자료형에서 중요한 개념으로 **요소의 위치**를 나타내는 값이다.

``` python
X = [1, 2, 3]
Y = (4, 5, 6)
Z = "Kangwon"
print(X[0])
print(Y[1])
print(Z[2])

Matrix = ([1,2], [3,4])
print(Matrix[0][1])
```

파이썬에서 인덱스는 `0`부터 시작한다. 개발자는 시퀀스 자료형에 인덱스를 활용하여 원하는 순서의 요소를 참조할 수 있다.

인덱스는 음수로도 사용할 수 있다. 이러한 경우 **시퀀스 자료형의 뒤에서부터 접근**가능하다.

``` python
print(X[-1])
print(Y[-2])
print(Z[-3])
```

인덱스를 활용하여 시퀀스 자료형의 특정 범위 요소로 새로운 부분 시퀀스를 추출하는 것을 **슬라이싱**이라고 한다. `:` 를 사용해서 슬라이싱을 구현할 수 있다.

``` python
print(X[:2])
print(Y[1:])
print(Z[1:4])
```
# 제어문 (Control Statements)
단순히 자료구조와 기본 연산자로는 복잡한 프로그램을 구현할 수 없다. 개발자들은 여러 연산의 **반복, 조건 처리, 예외 처리** 등을 사용하여 복잡한 프로그램을 구현한다.

제어문은 변수 선언, 연산 같은 일반 실행문과 달리 **코드의 실행 순서**를 바꿀 수 있다. 파이썬에서 제어문은 `Block` 으로 구성되고 `:`와 `Tab` 키를 사용한 들여쓰기로 하나의 `Block`을 묶을 수 있다.
``` python
if ... :
	Block 1의 코드1
	Block 1의 코드2
else :
	Block 2의 코드1
	Block 2의 코드2
	if ... :
		Block 2에 포함된 또다른 Block 2-1의 코드1
		Block 2에 포함된 또다른 Block 2-1의 코드2
```
## 조건문 (Conditional Statement)
조건문은  **주어진 조건에 따라 실행할 코드를 선택**하는 데 사용된다. 파이썬에서는 `if`, `elif`, `else` 문을 사용하여 조건문을 구성한다.
### if 기본 구조
`if` 만 사용하는 조건문의 기본 구조는 조건의 참(`True`), 거짓(`False`)를 판단하여 아래의 코드 블록을 실행할지 결정한다. **따라서, 조건으로 구성되는 변수, 연산자 또는 함수의 출력은 Bool 자료형으로 변환되어 조건을 판단한다.**

``` python
x = 10

if x > 5: #x가 5보다 크다 <- 조건
    print("x는 5보다 큽니다")

if 3:
	print("3은 Bool 자료형으로 변환되었을 때 True입니다.")
```
### if ~ else 구조
조건이 참일 때와 거짓일 때를 나누어 실행할 때 사용한다. 코드는 반드시 `if` 블록과 `else` 블록 중 하나를 실행한다. 즉, 두 블록의 조건은 **여집합**으로 생각할 수 있다.

``` python
x = 3

if x >= 5:
    print("5 이상")
else:
    print("5 미만")
```
### if ~ elif ~ else 구조
여러 조건을 순차적으로 검사하여 판단할 때 사용한다. **판단 순서는 가장 윗 조건부터 판단하기 때문에 조건 순서를 잘 정해야 한다.**

``` python
score = 85

if score >= 90:
    print("A")
elif score >= 80:
    print("B")
elif score >= 70:
    print("C")
else:
    print("F")

if score >= 70:
    print("C")
elif score >= 80:
    print("B")
elif score >= 90:
    print("A")
else:
    print("F")
```
## 반복문 (Loop)
같은 코드를 여러 번 반복해서 실행해야 할 때 사용하는 제어문이다. `for` 문과 `while` 문이 대표적인 반복문이다.
### for 문
`for`문은 **반복 가능한 객체(iterable)** 안의 요소를 순회하면서 블록 내부의 코드를 반복 실행하는 문법이다. 리스트, 문자열, 튜플, 딕셔너리, 집합 등 다양한 자료형이 이에 포함된다.

``` python
fruits = ["apple", "banana", "cherry"]

for fruit in fruits:
    print(fruit)
```

`for a in b` 형태로 사용하며, `a`는 `b`를 순회하며 각 요소의 값을 갖는다.

`for` 문에서 자주 사용되는 자료형 중에는 `range` 가 있다. 이는 **연속된 숫자 시퀀스**를 갖는 자료형이다. `range(A, B, C)` 는 **A부터 B까지 C간격으로 건너뛰며 요소를 생성**한다. 

``` python
r = range(3, 10, 2)

for v in r:
    print(v)
```

`enumerate()` 함수도 `for` 문과 자주 사용되는 함수다. 해당 함수를 사용하면 **인덱스와 요소를 동시에 사용할 수 있다.**

``` python
r = range(3, 10, 2)

for i, v in enumerate(r):
    print(i, v)
```
### while 문
`while`문은 조건이 `True`라면 계속해서 블록을 실행하는 반복문이다. `for`문과 달리 **반복 횟수를 결정하기 어렵거나 지속적으로 실행해야 할 프로그램의 경우** 주로 이 `while`문을 사용한다.

``` python
i = 0

while i < 5:
    print(i)
    i += 1
```
### 반복 제어 키워드
프로그래밍 도중 필요 시 반복을 중단하는 구현을 해야할 때가 있다. 이 경우 반복 제어 키워드를 사용하여 구현이 가능하다.
- `break` : 현재 실행 중인 **반복문 전체를 중단**한다.
- `continue` : 현재 반복 실행 중인 **블록을 중단하고 다음 반복 차수로 넘어간다.**

``` python
i = 0

while True:
    print(i)
    i += 1
    if i > 10 and i <= 20:
	    print("반복 횟수가 10번이 넘었습니다.")
	    continue
	if i > 20:
		break
```
# 함수
파이썬에서 함수는 같은 작업을 반복하지 않기 위해 **여러 줄의 코드를 묶어 하나의 이름으로 정의한 블록**이다. 

``` python
def 함수이름(매개변수1, 매개변수2, ...):
    실행할_코드
    return 반환값
```

기본적으로 함수는 `def` 를 사용하여 선언된다. 어떤 함수는 반환값이 없을 수도 있고, 매개변수가 없을 수도 있고, 둘 다 존재하지 않는 경우도 있다.

``` python
def add(a, b):
    return a + b

print(add(1, 2))
print(add(a = 3, b =4))
```

함수를 정의할 때 작성한 매개변수 순서에 맞게 인자를 전달하거나, 직접 매개변수에 인자를 매칭하여 함수를 사용할 수 있다.

``` python
def add(a=0, b=0):
    return a + b

print(add())
```

함수는 정의할 때 등호를 사용하여 매개변수의 **기본값**을 설정할 수 있다. 해당 매개변수에 매칭되는 인자가 들어오지 않으면, 함수는 기본값을 사용하여 실행된다.
## 가변 위치 인자와 가변 키워드 인자
가끔 개발자는 **전달되는 인자의 개수를 미리 정할 수 없을 때**가 있다. 이러한 경우 **가변 위치 인자**와 **가변 키워드 인자**를 사용하여 해결할 수 있다.
### 가변 위치 인자
가변 위치 인자는 매개변수 앞에 `*`를 넣어 사용할 수 있다. 함수의 인자들을 모두 **튜플**로 묶여져서 함수에 입력된다.

``` python
def print_all(*args):
    return args

print(add_all(1, 2, 3))      
print(add_all(10, 20, 30, 40))  
```
### 가변 키워드 인자
가변 키워드 인자는 매개변수 앞에 `**`를 넣어 사용할 수 있고, 함수의 인자들을 모두 **디렉토리**로 묶여져서 함수에 입력된다. 인자의 키워드는 **문자열**로 고정된다.

``` python
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key} = {value}")

print_info(name="영희", age=22, city="Seoul")
```
## 가변과 불변 
함수의 인자로 들어가는 **자료형에 따라 함수의 동작이 변할 수 있다.** 
- 가변 : 리스트, 딕셔너리, 집합
- 불변 : 튜플, 원시자료형

``` python
def modify(x):
    x += 1
    print("내부 x:", x)

a = 10
modify(a)
print("외부 a:", a)

def modify(lst):
    lst.append(100)
    print("내부 lst:", lst)

a = [1, 2, 3]
modify(a)
print("외부 a:", a)
```

해당 코드를 실행시켰을 때 결과를 확인하라. **가변 자료형은 함수 내부에서 요소가 변할 수 있다.**
# 예제

``` python
def process_scores(data):
    passed = []
    failed = []

    for name, score in data.items():
        if score >= 60:
            passed.append((name, score))
        else:
            failed.append((name, score))

    return passed, failed

input_data = [
    ("지민", 85),
    ("수지", 58),
    ("태현", 90),
    ("하늘", 45),
    ("현우", 75)
]

score_dict = {}
for name, score in input_data:
    score_dict[name] = score  

passed_list, failed_list = process_scores(score_dict)

print("합격자:")
for name, score in passed_list:
    print(f"{name}: {score}점")

print("불합격자:")
for name, score in failed_list:
    print(f"{name}: {score}점")
```

``` python
def summarize_weather(**city_temps):
    summary = {}

    for city, temps in city_temps.items():
        highest = max(temps)
        lowest = min(temps)
        average = round(sum(temps) / len(temps), 1)
        summary[city] = (highest, lowest, average)

    return summary

weather_data = {
    "서울": [22, 24, 19, 21, 23],
    "부산": [25, 27, 26, 28, 24],
    "대구": [30, 29, 28, 31, 32],
    "제주": [20, 22, 21, 23, 22]
}

result = summarize_weather(**weather_data)

for city, (high, low, avg) in result.items():
    print(f"{city} 최고: {high}℃, 최저: {low}℃, 평균: {avg}℃")
```
# 과제 1-1
## 1번 과제
사칙 연산을 수행하는 계산기 함수를 작성하라. 인자로 A, B, C를 받을 수 있다. C는 `+, -, *, /` 중 하나의 문자열을 인자로 받아 A와 B의 연산을 수행하고 이를 출력해야 한다.

## 2번 과제
버블 정렬이란 서로 인접한 두 원소를 비교하여 오른쪽 원소가 더 작다면 두 원소의 위치를 교환하는 것으로, 이를 좌측으로 순회하며 반복 실행하여 크기 순서대로 데이터를 정렬하는 알고리즘이다.
![[Pasted image 20250625235320.png]]
리스트를 인자로 받아 버블 정렬을 수행하는 함수를 작성하라.
