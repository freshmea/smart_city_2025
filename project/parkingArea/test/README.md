# 🔍 YOLO-OBB 테스트 도구 모음

YOLO-OBB 모델의 실행 결과를 다양한 방식으로 시각화하고 모니터링할 수 있는 테스트 도구들입니다.

## 📁 포함된 도구들

### 1. 📊 상세 OBB 모니터링 (`obb_monitor.py`)

가장 포괄적인 분석 도구로, YOLO-OBB 결과를 상세하게 분석하고 시각화합니다.

**주요 기능:**
- 🎯 다중 신뢰도 임계값 테스트 (0.1, 0.25, 0.5)
- 📐 정확한 차량 크기 측정 (폭, 높이, 면적, 회전각)
- 📈 차량 크기 분포 시각화
- 🆔 각 차량별 상세 정보 표시
- 📊 크기 균일성 통계
- 💾 결과 자동 저장

**실행:**
```bash
uv run obb_monitor.py
```

**출력 예시:**
```
📊 YOLO-OBB 감지 결과 요약
============================================================
🔍 총 감지 객체: 5개
🚗 차량: 3대

📋 차량 상세 정보:
ID | 클래스     | 신뢰도 | 크기(W×H)    | 면적     | 각도   | 중심점
----------------------------------------------------------------------
 1 | car        |   0.85 |   180×95   |    17100 |   -2.3° | (245,156)
 2 | car        |   0.78 |   175×98   |    17150 |    1.8° | (456,187)
 3 | truck      |   0.72 |   220×110  |    24200 |   -0.5° | (678,203)

📏 크기 통계:
  폭: 평균 191.7px, 표준편차 23.6px
  높이: 평균 101.0px, 표준편차 7.9px
  면적: 평균 19483px², 표준편차 4050px²
  크기 균일성: 89.2%
```

### 2. ⚡ 빠른 OBB 뷰어 (`quick_obb_viewer.py`)

간단하고 빠른 시각화를 위한 경량 도구입니다.

**주요 기능:**
- 🚀 빠른 실행
- 📸 원본 vs 감지결과 비교
- 🏷️ 차량 정보 라벨링
- 📊 기본 감지 통계

**실행:**
```bash
uv run quick_obb_viewer.py
```

### 3. 📹 실시간 모니터링 (`realtime_obb_monitor.py`)

웹캠이나 이미지 파일을 대화형으로 모니터링할 수 있는 도구입니다.

**주요 기능:**
- 📹 웹캠 실시간 감지
- 🖼️ 이미지 파일 대화형 모니터링
- ⚙️ 실시간 신뢰도 임계값 조정
- 📊 실시간 FPS 및 통계 표시
- 💾 프레임 저장 기능
- 🎮 키보드 조작

**조작법:**
- `q`: 종료
- `c`: 신뢰도 임계값 변경
- `s`: 현재 프레임 저장

**실행:**
```bash
uv run realtime_obb_monitor.py
```

## 🚀 빠른 시작

### 1. 통합 실행 스크립트 사용

```bash
# 실행 권한 부여
chmod +x run_obb_tests.sh

# 실행
./run_obb_tests.sh
```

### 2. 개별 도구 실행

```bash
cd test

# 상세 분석
uv run obb_monitor.py

# 빠른 확인
uv run quick_obb_viewer.py

# 실시간 모니터링
uv run realtime_obb_monitor.py
```

## 📋 시스템 요구사항

- **Python 3.10+**
- **CUDA 지원 PyTorch** (GPU 가속, 선택사항)
- **OpenCV** (`cv2`)
- **matplotlib** (시각화)
- **ultralytics** (YOLO 모델)
- **numpy**

## 📐 테스트 이미지

도구들은 다음 경로에서 테스트 이미지를 자동으로 찾습니다:

```
../parkinglot1.jpg
../../parkinglot1.jpg
../data/parkinglot1.jpg
parkinglot1.jpg
```

## 🎯 사용 사례

### 1. 모델 성능 검증
```bash
# 다양한 신뢰도에서 감지 성능 확인
uv run obb_monitor.py
```

### 2. 빠른 결과 확인
```bash
# 간단한 시각적 확인
uv run quick_obb_viewer.py
```

### 3. 실시간 데모
```bash
# 웹캠으로 실시간 데모
uv run realtime_obb_monitor.py
# 선택: 1 (웹캠)
```

### 4. 이미지 분석
```bash
# 특정 이미지 상세 분석
uv run realtime_obb_monitor.py
# 선택: 2 (이미지)
```

## 📊 출력 파일

### 생성되는 파일들

- `obb_monitor_result_*.png`: 상세 분석 결과
- `realtime_capture_*.jpg`: 실시간 캡처 프레임
- `image_monitor_result.jpg`: 이미지 모니터링 결과

### 파일 구조 예시

```
test/
├── obb_monitor.py                    # 상세 분석 도구
├── quick_obb_viewer.py              # 빠른 뷰어
├── realtime_obb_monitor.py          # 실시간 모니터링
├── run_obb_tests.sh                 # 통합 실행 스크립트
├── README.md                        # 이 파일
├── obb_monitor_result_parkinglot1.png    # 생성된 결과
└── realtime_capture_000001.jpg     # 캡처된 프레임
```

## 🔧 고급 설정

### 1. 신뢰도 임계값 조정

```python
# obb_monitor.py에서
confidence_levels = [0.1, 0.2, 0.3, 0.5, 0.7]  # 사용자 정의

# quick_obb_viewer.py에서
quick_obb_test(image_path, conf_threshold=0.2)  # 기본값 변경
```

### 2. 모델 경로 변경

```python
# 다른 YOLO 모델 사용
monitor = OBBMonitor("path/to/your/model.pt")
```

### 3. 차량 클래스 필터링

```python
# 특정 차량 타입만 감지
vehicle_classes = [2]  # 자동차만
vehicle_classes = [2, 7]  # 자동차 + 트럭
```

## 🚨 문제 해결

### 1. 모델 로드 실패
```bash
# YOLO 모델 경로 확인
ls -la ../../yolov8n-obb.pt
```

### 2. 이미지 로드 실패
```bash
# 테스트 이미지 확인
ls -la ../parkinglot1.jpg
```

### 3. GPU 메모리 부족
```python
# CPU 모드로 강제 실행
device = torch.device('cpu')
```

### 4. 웹캠 연결 실패
```python
# 다른 카메라 ID 시도
camera_id = 1  # 또는 2, 3...
```

## 📈 성능 최적화

### GPU 가속 확인
```python
# 현재 디바이스 확인
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

### 메모리 사용량 최적화
```python
# 이미지 크기 조정
image = cv2.resize(image, (640, 480))
```

## 🎉 추가 기능

### 배치 처리
```python
import glob

for image_path in glob.glob("../test_images/*.jpg"):
    monitor.monitor_image(image_path)
```

### 비디오 파일 처리
```python
# realtime_obb_monitor.py 수정하여 비디오 파일 지원
cap = cv2.VideoCapture("video.mp4")
```

---

**💡 팁**: 처음 사용하시는 경우 `quick_obb_viewer.py`로 시작하여 기본 동작을 확인한 후, `obb_monitor.py`로 상세 분석을 진행하세요!