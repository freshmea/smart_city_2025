# 🚗 주차장 탑뷰 이미지 분석 시스템 - UV 환경 가이드

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# UV 의존성 설치
uv sync

# 또는 개발 환경 설정
./dev.sh setup
```

### 2. 분석 실행
```bash
# 간편한 실행 (대화형)
./run_analysis.sh

# 직접 실행
uv run main.py              # OpenCV 고급 분석
uv run simple_parking.py    # 간단한 분석
```

## 📋 주요 명령어

### UV 환경 관리
```bash
uv sync                     # 의존성 동기화
uv add <패키지명>            # 패키지 추가
uv remove <패키지명>         # 패키지 제거
uv pip list                 # 설치된 패키지 목록
```

### 개발 도구 (dev.sh)
```bash
./dev.sh setup             # 개발 환경 설정
./dev.sh run               # 분석 실행
./dev.sh info              # 환경 정보 확인
./dev.sh clean             # 임시 파일 정리
./dev.sh help              # 도움말
```

### 분석 실행
```bash
./run_analysis.sh          # 대화형 분석 실행
uv run main.py             # OpenCV 고급 분석
uv run simple_parking.py   # 간단한 분석
```

## 📊 분석 결과

### OpenCV 고급 분석
- **감지된 주차 영역**: 18개
- **수평선**: 39개, **수직선**: 14개
- **출력 파일**:
  - `parking_detection_result.jpg`: 결과 이미지
  - `parking_detection_info.txt`: 상세 정보
  - `parking_comparison.png`: 원본-결과 비교

### 간단한 분석
- **자동 감지**: 44개 구역
- **수동 격자**: 15개 구역
- **출력 파일**:
  - `output_auto/simple_parking_result.jpg`: 자동 감지 결과
  - `output_manual/simple_parking_result.jpg`: 수동 격자 결과
  - `parking_analysis_comparison.png`: 전체 비교

## 🛠️ 개발 팁

### 코드 수정 후 실행
```bash
# 코드 수정 후 즉시 실행
uv run main.py

# 여러 번 테스트할 때
./dev.sh clean && uv run simple_parking.py
```

### 새 패키지 추가
```bash
# 런타임 의존성 추가
uv add opencv-contrib-python

# 개발 의존성 추가
uv add --dev pytest black flake8
```

### 환경 초기화
```bash
# 깨끗한 환경으로 재설정
rm -rf .venv uv.lock
uv sync
```

## 🔧 커스터마이징

### 주차 영역 크기 조정 (main.py)
```python
# 라인 155-156 근처
if 30 < width < 200 and 50 < height < 300:  # 크기 조정
    parking_spots.append((x1, y1, x2, y2))
```

### 감지 민감도 조정 (simple_parking.py)
```python
# 라인 85-86 근처
h_threshold = np.percentile(h_projection, 70)  # 70 -> 다른 값
v_threshold = np.percentile(v_projection, 70)  # 70 -> 다른 값
```

## 📁 프로젝트 구조
```
parkingArea/
├── main.py                 # OpenCV 고급 분석
├── simple_parking.py       # 간단한 분석
├── run_analysis.sh         # 실행 스크립트
├── dev.sh                  # 개발 도구
├── pyproject.toml         # UV 프로젝트 설정
├── requirements.txt       # pip 호환성
├── README.md              # 상세 문서
└── parkinglot1.jpg        # 샘플 이미지
```

## 🐛 문제 해결

### UV 명령어가 작동하지 않을 때
```bash
# UV 설치 확인
which uv

# UV 재설치 (필요시)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Python 환경 문제
```bash
# 현재 Python 버전 확인
uv run python --version

# 특정 Python 버전 사용 (필요시)
uv python pin 3.10
```

### 의존성 충돌
```bash
# 의존성 재해결
rm uv.lock
uv sync

# 캐시 클리어
uv cache clean
```