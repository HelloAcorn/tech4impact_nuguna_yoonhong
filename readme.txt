민윤홍의 테크포임팩트 - 누구나데이터 LAB 개발내용 정리

혼자 개발하는거라 그냥 편하게 테스트 한거 깃에 올리는 코드.

## requirements.txt 만들기
열심히 필요한 라이브러리 설치하고 이후 터미널에 아래 명령어 입력.
conda list | grep -v "conda-forge" | awk '{print $1"=="$2}' > requirements.txt


## 가상환경 생성 및 설정

이 프로젝트는 Python 3.10.15 버전의 가상환경을 사용합니다. 아래 단계를 따라 가상환경을 설정하세요.

### 1. Python 3.10.15 버전 가상환경 생성
python3.10 -m venv [가상환경 이름]

### 2. 가상환경 활성화
source [가상환경 이름]/bin/activate


### 3. 패키지 설치
pip install -r requirements.txt
