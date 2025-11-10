"""
간단한 YOLO-OBB 실시간 시각화 도구
빠른 테스트용 간소화된 버전
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO


def quick_obb_test(image_path: str, model_path: str = "../../yolov8n-obb.pt"):
    """빠른 OBB 테스트"""
    print(f"🚗 빠른 OBB 테스트: {os.path.basename(image_path)}")

    # 디바이스 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 디바이스: {device}")

    # 모델 로드
    try:
        model = YOLO(model_path)
        print(f"✅ 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지 로드 실패")
        return

    print(f"📐 이미지 크기: {image.shape}")

    # YOLO 추론
    results = model(image, verbose=False, conf=0.25)

    # 결과 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 원본 이미지
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(image_rgb)
    axes[0].set_title('원본 이미지')
    axes[0].axis('off')

    # 감지 결과
    axes[1].imshow(image_rgb)
    axes[1].set_title('OBB 감지 결과')

    detection_count = 0
    vehicle_count = 0

    for result in results:
        if hasattr(result, 'obb') and result.obb is not None:
            for obb, conf, cls in zip(result.obb.xyxyxyxy, result.obb.conf, result.obb.cls):
                detection_count += 1
                class_id = int(cls)
                confidence = float(conf)

                # OBB 점들
                obb_points = obb.cpu().numpy().reshape(-1, 2)

                # 차량 클래스 확인
                is_vehicle = class_id in [2, 3, 5, 7]  # car, motorcycle, bus, truck
                if is_vehicle:
                    vehicle_count += 1

                # 색상 선택
                color = 'red' if is_vehicle else 'blue'

                # OBB 그리기
                polygon = plt.Polygon(obb_points, fill=False, edgecolor=color, linewidth=2)
                axes[1].add_patch(polygon)

                # 중심점 계산
                center_x = np.mean(obb_points[:, 0])
                center_y = np.mean(obb_points[:, 1])

                # 라벨 추가
                class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
                class_name = class_names.get(class_id, f'class_{class_id}')

                if is_vehicle:
                    # 크기 계산
                    width = np.linalg.norm(obb_points[1] - obb_points[0])
                    height = np.linalg.norm(obb_points[2] - obb_points[1])

                    label = f"{class_name}\n{width:.0f}x{height:.0f}\n{confidence:.2f}"
                    axes[1].text(center_x, center_y, label,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                               fontsize=8, ha='center', va='center')

        # 일반 박스 처리 (OBB가 없는 경우)
        elif hasattr(result, 'boxes') and result.boxes is not None:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                detection_count += 1
                class_id = int(cls)
                confidence = float(conf)

                is_vehicle = class_id in [2, 3, 5, 7]
                if is_vehicle:
                    vehicle_count += 1

                x1, y1, x2, y2 = [int(x) for x in box.cpu().numpy()]

                color = 'red' if is_vehicle else 'blue'
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                   fill=False, edgecolor=color, linewidth=2)
                axes[1].add_patch(rect)

                if is_vehicle:
                    class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
                    class_name = class_names.get(class_id, f'class_{class_id}')
                    label = f"{class_name}\n{x2-x1}x{y2-y1}\n{confidence:.2f}"
                    axes[1].text((x1+x2)/2, (y1+y2)/2, label,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                               fontsize=8, ha='center', va='center')

    axes[1].axis('off')

    # 제목 업데이트
    axes[1].set_title(f'감지 결과: {detection_count}개 객체, {vehicle_count}대 차량')

    plt.tight_layout()
    plt.show()

    # 요약 출력
    print(f"📊 감지 결과:")
    print(f"   총 객체: {detection_count}개")
    print(f"   차량: {vehicle_count}대")

    return detection_count, vehicle_count

def main():
    """메인 함수"""
    print("🔍 빠른 YOLO-OBB 시각화 도구")
    print("="*40)

    # 이미지 경로 찾기
    possible_paths = [
        "../parkinglot1.jpg",
        "../../parkinglot1.jpg",
        "../data/parkinglot1.jpg",
        "parkinglot1.jpg"
    ]

    image_path = None
    for path in possible_paths:
        if os.path.exists(path):
            image_path = path
            break

    if image_path is None:
        print("❌ 테스트 이미지를 찾을 수 없습니다.")
        print("💡 다음 경로에 parkinglot1.jpg 파일을 배치해주세요:")
        for path in possible_paths:
            print(f"   {path}")
        return

    print(f"✅ 이미지 발견: {image_path}")

    # OBB 테스트 실행
    try:
        detection_count, vehicle_count = quick_obb_test(image_path)
        print(f"\n🎉 테스트 완료!")

        if vehicle_count > 0:
            print(f"✅ {vehicle_count}대의 차량이 감지되었습니다.")
        else:
            print("⚠️ 차량이 감지되지 않았습니다. 신뢰도 임계값을 낮춰보세요.")

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()