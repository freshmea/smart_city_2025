# ------------------------------------------------------------
# InsightFace 얼굴검출 및 비식별화 스크립트
# - InsightFace 모델을 사용한 정확한 얼굴 검출
# - 얼굴 영역 블러 처리로 비식별화
# - 이미지/폴더 배치 처리 지원
# ------------------------------------------------------------

import json
import os
import sys
from datetime import datetime

import cv2 as cv
import numpy as np
from insightface.app import FaceAnalysis

# InsightFace 및 관련 패키지 필요:
# uv add insightface onnxruntime-gpu

# ===== [설정] 여기만 바꿔서 사용합니다 =====
# 1) 입력: 이미지 경로가 있으면 파일 처리, None이면 카메라 처리
INPUT_IMAGE = "/home/aa/smart_city_2025/project/data/14_Traffic_Traffic_14_105.jpg"
INPUT_IMAGE_FOLDER = "/home/aa/smart_city_2025/project/data/"    # 이미지 폴더 경로 (배치 처리용)

# 2) 검출 및 분석 설정
DETECTION_SIZE = (640, 640)     # 검출용 이미지 크기
CONFIDENCE_THRESHOLD = 0.7      # 얼굴 검출 신뢰도

# 3) 비식별화 설정
BLUR_KERNEL_SIZE = 99           # 블러 커널 크기 (홀수, 클수록 더 흐림)
BLUR_SIGMA = 30                 # 가우시안 블러 시그마 값

# 4) 출력 옵션
SAVE_RESULT = True              # 이미지 입력일 때만 저장
SHOW_WINDOW = True              # 결과 창 표시
OUTPUT_FOLDER = "/home/aa/smart_city_2025/project/output/"
LOG_ANALYSIS = True             # 분석 로그 저장 여부
LOG_FILE_PATH = '/home/aa/smart_city_2025/project/output/face_analysis_log.jsonl'

# 5) 기준 인물 설정 (특정 인물 보호 시 사용)
REFERENCE_IMAGE_PATH = '/home/aa/smart_city_2025/data/face/choi.jpg'  # 기준 인물 이미지
SIMILARITY_THRESHOLD = 0.3      # 기준 인물과의 유사도 임계값
PROTECT_REFERENCE_ONLY = False  # True: 기준 인물만 보호, False: 모든 얼굴 보호


# ===== [InsightFace 모델 초기화] =====
def initialize_face_analysis():
    """InsightFace FaceAnalysis 모델을 초기화합니다."""
    print("InsightFace 모델을 로드합니다...")
    try:
        # GPU 사용을 명시하여 모델 로드 (onnxruntime-gpu 필요)
        app = FaceAnalysis(providers=['CUDAExecutionProvider'])
        app.prepare(ctx_id=0, det_size=DETECTION_SIZE)
        print("GPU 모델 로드 완료.")
        return app
    except Exception as e:
        print(f"GPU 모델 로드 실패: {e}")
        print("CPU 모델로 다시 시도합니다...")
        try:
            app = FaceAnalysis(providers=['CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=DETECTION_SIZE)
            print("CPU 모델 로드 완료.")
            return app
        except Exception as e_cpu:
            print(f"CPU 모델 로드도 실패했습니다: {e_cpu}")
            return None

def get_reference_embedding(app, image_path):
    """기준 이미지에서 얼굴 특징 벡터(embedding)를 추출합니다."""
    if not PROTECT_REFERENCE_ONLY or not os.path.exists(image_path):
        return None

    print(f"기준 이미지({image_path})에서 특징을 추출합니다...")
    try:
        img_ref = cv.imread(image_path)
        if img_ref is None:
            print(f"오류: 기준 이미지를 찾을 수 없습니다: {image_path}")
            return None

        faces_ref = app.get(img_ref)
        if not faces_ref:
            print(f"오류: 기준 이미지에서 얼굴을 찾지 못했습니다.")
            return None

        # 가장 큰 얼굴을 기준으로 삼음
        main_face = sorted(faces_ref, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[0]
        print("기준 이미지 특징 추출 완료.")
        return main_face.normed_embedding
    except Exception as e:
        print(f"기준 특징 추출 중 오류 발생: {e}")
        return None

def log_data(data):
    """분석된 데이터를 JSONL 형식으로 파일에 기록합니다."""
    if not LOG_ANALYSIS:
        return
    try:
        os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
        with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"로그 파일 작성 중 오류 발생: {e}")


# ===== [시각화 및 비식별화 유틸] =====
def apply_face_deidentification(image, faces, feat_ref=None):
    """
    얼굴 영역에 비식별화 처리를 적용합니다.
    faces: InsightFace에서 반환된 얼굴 검출 결과
    feat_ref: 기준 인물 특징 벡터 (None이면 모든 얼굴 처리)
    """
    out = image.copy()
    all_faces_data = []

    if not faces:
        return out, all_faces_data

    for i, face in enumerate(faces):
        # 바운딩 박스 추출
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        # 이미지 경계 확인
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        should_blur = True
        similarity_score = None

        # 기준 인물과의 유사도 검사 (PROTECT_REFERENCE_ONLY가 True일 때)
        if PROTECT_REFERENCE_ONLY and feat_ref is not None:
            feat_current = face.normed_embedding
            similarity_score = float(np.dot(feat_ref, feat_current))
            should_blur = similarity_score > SIMILARITY_THRESHOLD

        # 얼굴 분석 데이터 로깅
        face_data = {
            'timestamp': datetime.now().isoformat(),
            'face_id': i,
            'bounding_box': [x1, y1, x2, y2],
            'detection_score': float(face.det_score),
            'gender': 'Male' if face.gender == 1 else 'Female',
            'age': int(face.age),
            'should_blur': should_blur,
            'similarity_score': similarity_score,
        }
        all_faces_data.append(face_data)

        # 비식별화 처리
        if should_blur:
            roi = out[y1:y2, x1:x2]
            if roi.size > 0:
                # 타원형 마스크 생성
                mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                center = ((x2-x1)//2, (y2-y1)//2)
                axes = ((x2-x1)//2, (y2-y1)//2)
                cv.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

                # 가우시안 블러 적용
                blurred_roi = cv.GaussianBlur(roi, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), BLUR_SIGMA)

                # 마스크를 이용해 타원형 영역만 블러 처리
                out[y1:y2, x1:x2] = cv.bitwise_and(out[y1:y2, x1:x2], out[y1:y2, x1:x2], mask=~mask) + \
                                   cv.bitwise_and(blurred_roi, blurred_roi, mask=mask)

                # 처리된 영역 표시 (디버깅용)
                if SHOW_WINDOW:
                    cv.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv.putText(out, f"ID:{i}", (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return out, all_faces_data

def visualize(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
    """
    결과 시각화 (InsightFace 버전)
    """
    out = image.copy()

    if fps is not None:
        cv.putText(out, f'FPS: {fps:.2f}', (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv.LINE_AA)

    return out


# ===== [이미지 처리 유틸] =====
def img_scale(img, max_size=800):
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
    """이미지 파일 또는 폴더의 모든 이미지를 처리합니다."""
    # InsightFace 모델 초기화
    app = initialize_face_analysis()
    if app is None:
        print("모델 초기화에 실패했습니다.")
        return

    # 기준 인물 특징 추출 (필요한 경우)
    feat_ref = get_reference_embedding(app, REFERENCE_IMAGE_PATH)

    # 폴더에서 파일 리스트 만들기
    input_folder = INPUT_IMAGE_FOLDER
    if os.path.isdir(input_folder):
        img_files = [f for f in os.listdir(input_folder)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        print(f"{len(img_files)}개의 이미지 파일을 찾았습니다.")
    else:
        img_files = [os.path.basename(img_path)]
        input_folder = os.path.dirname(img_path)

    # 출력 폴더 생성
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    total_faces = 0

    for idx, img_file in enumerate(img_files):
        print(f"\n[{idx+1}/{len(img_files)}] 처리 중: {img_file}")

        img_path = os.path.join(input_folder, img_file)
        img = cv.imread(img_path)

        if img is None:
            print(f"이미지를 열 수 없습니다: {img_path}")
            continue

        # 이미지 크기 조정 (처리 속도 향상)
        img = img_scale(img)

        # 얼굴 검출
        faces = app.get(img)

        print(f"  - {len(faces)}개의 얼굴을 검출했습니다.")
        total_faces += len(faces)

        # 비식별화 처리
        processed_img, faces_data = apply_face_deidentification(img, faces, feat_ref)

        # 분석 데이터 로깅
        if faces_data:
            log_data({
                'image_file': img_file,
                'total_faces': len(faces),
                'faces': faces_data
            })

        # 결과 저장
        if SAVE_RESULT:
            name, ext = os.path.splitext(img_file)
            out_filename = f"{name}_deidentified{ext}"
            out_path = os.path.join(OUTPUT_FOLDER, out_filename)
            cv.imwrite(out_path, processed_img)
            print(f"  - 결과 저장: {out_path}")

        # 결과 표시 (마지막 이미지만)
        if SHOW_WINDOW and idx == len(img_files) - 1:
            cv.imshow("Original", img)
            cv.imshow("Deidentified", processed_img)
            print("Press any key to continue...")
            cv.waitKey(0)
            cv.destroyAllWindows()

    print(f"\n처리 완료: {len(img_files)}개 이미지, 총 {total_faces}개 얼굴 처리")

# ===== [엔트리 포인트] =====
if __name__ == "__main__":
    run_on_image(INPUT_IMAGE)
