# ------------------------------------------------------------
# YuNet 얼굴검출 최소 실행 스크립트 (OpenCV Zoo / ONNXRuntime-ORT 버전)
# - 불필요한 argparse 제거
# - 설정 상수만 바꿔서 이미지/카메라 처리
# - 핵심 경로: load -> infer -> visualize
# ------------------------------------------------------------

import os
import sys

import cv2 as cv
import numpy as np


# ===== [환경 체크] OpenCV 버전 확인 =====
def _parse_ver(s): return tuple(map(int, s.split(".")))
assert _parse_ver(cv.__version__) >= _parse_ver("4.10.0"), \
    "opencv-python 4.10.0 이상이 필요합니다.  e.g.  pip install -U opencv-python"

# OpenCV Zoo YuNet ORT 바인딩
from yunet_ort import YuNet

# (필요 시 OpenCV DNN YuNet 사용)
# from yunet import YuNet

# ===== [설정] 여기만 바꿔서 사용합니다 =====
# 1) 입력: 이미지 경로가 있으면 파일 처리, None이면 카메라 처리
INPUT_IMAGE = "/home/aa/smart_city_2025/project/data/14_Traffic_Traffic_14_105.jpg"                 # 카메라 인덱스
INPUT_IMAGE_FOLDER = "/home/aa/smart_city_2025/project/data/"    # 이미지 폴더 경로 (배치 처리용)
# 2) YuNet ONNX 모델 경로
MODEL_PATH = "/home/aa/smart_city_2025/data/face_detection_yunet_2023mar.onnx"

# 3) 백엔드/타깃 (OpenCV가 지원하는 조합 중 선택)
#    0: OpenCV DNN + CPU (기본)
#    1: CUDA + GPU
#    2: CUDA + GPU(FP16)
#    3: TIM-VX + NPU
#    4: CANN + NPU
BACKEND_TARGET_INDEX = 0

# 4) 검출 하이퍼파라미터
CONF_THRESHOLD = 0.9            # 낮출수록 더 많이 잡지만 오검출 증가
NMS_THRESHOLD  = 0.3
TOP_K          = 5000

# 5) 출력 옵션
SAVE_RESULT = True              # 이미지 입력일 때만 저장
SHOW_WINDOW = True              # 결과 창 표시


# ===== [백엔드/타깃 매핑] =====
_backend_target_pairs = [
    (cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU),
    (cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA),
    (cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16),
    (cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU),
    (cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU),
]
BACKEND_ID, TARGET_ID = _backend_target_pairs[BACKEND_TARGET_INDEX]


# ===== [시각화 유틸] 바운딩박스/랜드마크/ FPS 오버레이 =====
def visualize(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
    """
    results: shape (N, 15) = [x, y, w, h, l0x, l0y, ..., l4x, l4y, score]
    """
    out = image.copy()
    landmark_color = [(255,0,0), (0,0,255), (0,255,0), (255,0,255), (0,255,255)]

    if fps is not None:
        cv.putText(out, f'FPS: {fps:.2f}', (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv.LINE_AA)

    if results is None or len(results) == 0:
        return out

    for det in results:
        x, y, w, h = det[0:4].astype(np.int32)
        # cv.rectangle(out, (x, y), (x+w, y+h), box_color, 2)
        roi = out[y:y+h, x:x+w]
        # 원 형태의 mask 생성
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (w // 2, h // 2)
        cv.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        # 마스크를 이용해 원형 영역 추출
        roi = cv.bitwise_or(roi, roi, mask=mask)
        cv.GaussianBlur(roi, (99, 99), 5, dst=roi)
        # 원형 영역을 원래 이미지에 복사를 타원안의 이미지만 덮어쓰기
        out[y:y+h, x:x+w] = cv.bitwise_and(out[y:y+h, x:x+w], out[y:y+h, x:x+w], mask=~mask) + cv.bitwise_and(roi, roi, mask=mask)

    return out


# ===== [모델 생성] =====
def create_model():
    """
    YuNet ORT 백엔드로 모델 초기화.
    inputSize는 실제 프레임 크기에 맞춰 매 호출 전 setInputSize로 갱신합니다.
    """
    model = YuNet(
        modelPath=MODEL_PATH,
        inputSize=[320, 320],          # dummy 초기값 (실사용 전에 setInputSize로 업데이트)
        confThreshold=CONF_THRESHOLD,
        nmsThreshold=NMS_THRESHOLD,
        topK=TOP_K,
        backendId=BACKEND_ID,
        targetId=TARGET_ID
    )
    return model

def img_scale(img, max_size=640):
    """
    이미지 크기를 max_size 이하로 비율 유지 축소.
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv.resize(img, (new_w, new_h))
    return resized


# ===== [이미지 한 장 처리] =====
def run_on_image(img_path):
    # 폴더에서 파일 리스트 만들기
    model = create_model()
    img_files = [f for f in os.listdir(INPUT_IMAGE_FOLDER) if f.endswith((".jpg", ".png"))]
    for img_file in img_files:
        img = cv.imread(os.path.join(INPUT_IMAGE_FOLDER, img_file))
        if img is None:
            raise FileNotFoundError(f"이미지를 열 수 없습니다: {img_path}")
        img = img_scale(img)
        h, w = img.shape[:2]
        model.setInputSize([w, h])             # YuNet은 입력 프레임 크기를 맞춰야 정확
        results = model.infer(img)             # 검출
        print(f"{len(results)} faces detected.")
        for idx, det in enumerate(results):
            # 디버그용: 좌표/점/스코어 출력
            print(f"{idx}: " + " ".join([f"{v:.0f}" if i<14 else f"{v:.4f}"
                                        for i, v in enumerate(det)]))
        vis = visualize(img, results)

        if SAVE_RESULT:
            out_path = "/home/aa/smart_city_2025/project/output/"
            cv.imwrite(out_path + img_file, vis)
            print(f"결과 저장: {out_path}")

    if SHOW_WINDOW:
        cv.imshow("YuNet Result", vis)
        cv.waitKey(0)
        cv.destroyAllWindows()

# ===== [엔트리 포인트] =====
if __name__ == "__main__":
    run_on_image(INPUT_IMAGE)
