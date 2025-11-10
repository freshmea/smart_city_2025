"""
YOLO-OBB 실행 결과 실시간 모니터링 시스템
OBB (Oriented Bounding Box) 결과를 시각적으로 확인할 수 있는 테스트 도구
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

# 상위 디렉토리의 모듈 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class OBBMonitor:
    """YOLO-OBB 결과 실시간 모니터링 클래스"""

    def __init__(self, model_path: str = "../../yolov8n-obb.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 디바이스: {self.device}")

        # YOLO-OBB 모델 로드
        try:
            self.model = YOLO(model_path)
            print(f"✅ YOLO-OBB 모델 로드 완료: {model_path}")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            self.model = None

        # COCO 클래스 이름 매핑
        self.class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow'
        }

        # 차량 관련 클래스
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

        # 색상 매핑
        self.colors = {
            2: (0, 255, 0),    # car - 녹색
            3: (255, 0, 0),    # motorcycle - 파란색
            5: (0, 0, 255),    # bus - 빨간색
            7: (255, 255, 0),  # truck - 청록색
        }

        # Perspective Transform 관련 변수들
        self.perspective_matrix = None
        self.detected_lines = []

    def detect_with_obb(self, image_path: str, conf_threshold: float = 0.25):
        """YOLO-OBB로 객체 감지"""
        if self.model is None:
            print("❌ 모델이 로드되지 않았습니다.")
            return None, None

        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 이미지 로드 실패: {image_path}")
            return None, None

        print(f"📐 이미지 크기: {image.shape}")

        # YOLO-OBB 추론
        try:
            results = self.model(image, verbose=False, conf=conf_threshold)
            print(f"🔍 추론 완료 (신뢰도 임계값: {conf_threshold})")
            return image, results
        except Exception as e:
            print(f"❌ 추론 실패: {e}")
            return image, None

    def extract_detections(self, results):
        """감지 결과에서 정보 추출"""
        detections = []

        for result in results:
            # OBB 결과 처리
            if hasattr(result, 'obb') and result.obb is not None:
                print(f"📦 OBB 감지: {len(result.obb)} 개 객체")

                for i, (obb, conf, cls) in enumerate(zip(result.obb.xyxyxyxy, result.obb.conf, result.obb.cls)):
                    class_id = int(cls)
                    confidence = float(conf)
                    class_name = self.class_names.get(class_id, f"class_{class_id}")

                    # OBB 점들을 numpy 배열로 변환
                    obb_points = obb.cpu().numpy().reshape(-1, 2)

                    # 바운딩 박스 계산
                    x_coords = obb_points[:, 0]
                    y_coords = obb_points[:, 1]
                    bbox = [int(np.min(x_coords)), int(np.min(y_coords)),
                           int(np.max(x_coords)), int(np.max(y_coords))]

                    # 크기 계산
                    width = np.linalg.norm(obb_points[1] - obb_points[0])
                    height = np.linalg.norm(obb_points[2] - obb_points[1])
                    area = width * height

                    # 중심점
                    center = [int(np.mean(x_coords)), int(np.mean(y_coords))]

                    # 회전 각도 계산
                    angle = np.arctan2(obb_points[1][1] - obb_points[0][1],
                                     obb_points[1][0] - obb_points[0][0]) * 180 / np.pi

                    detection = {
                        'id': i,
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'obb_points': obb_points,
                        'bbox': bbox,
                        'center': center,
                        'size': (width, height),
                        'area': area,
                        'angle': angle,
                        'is_vehicle': class_id in self.vehicle_classes
                    }
                    detections.append(detection)

            # 일반 박스 결과도 처리 (OBB가 없는 경우)
            elif hasattr(result, 'boxes') and result.boxes is not None:
                print(f"📦 일반 박스 감지: {len(result.boxes)} 개 객체")

                for i, (box, conf, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)):
                    class_id = int(cls)
                    confidence = float(conf)
                    class_name = self.class_names.get(class_id, f"class_{class_id}")

                    # 박스 좌표
                    x1, y1, x2, y2 = [int(x) for x in box.cpu().numpy()]
                    bbox = [x1, y1, x2, y2]

                    # 사각형 모서리 점들 (OBB 형태로 변환)
                    obb_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

                    # 크기 계산
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height

                    # 중심점
                    center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]

                    detection = {
                        'id': i,
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'obb_points': obb_points,
                        'bbox': bbox,
                        'center': center,
                        'size': (width, height),
                        'area': area,
                        'angle': 0.0,  # 일반 박스는 회전 없음
                        'is_vehicle': class_id in self.vehicle_classes
                    }
                    detections.append(detection)

        return detections

    def detect_lines_from_obb(self, detections: List[Dict]) -> List[Tuple[float, float]]:
        """OBB polygon과 겹치는 수직/수평 선분들 생성"""
        lines = []

        # 차량만 필터링
        vehicles = [d for d in detections if d['is_vehicle']]

        if len(vehicles) < 1:
            print("⚠️ 직선 검출을 위한 차량이 없습니다")
            return lines

        print(f"🚗 {len(vehicles)}대 차량에서 수직/수평 선분 생성")

        # 각 차량에 대해 수직/수평 선분들 생성
        for vehicle_idx, vehicle in enumerate(vehicles):
            obb_points = vehicle['obb_points']
            if len(obb_points) >= 4:
                points = np.array(obb_points)

                # 차량 경계 상자 계산
                min_x = np.min(points[:, 0])
                max_x = np.max(points[:, 0])
                min_y = np.min(points[:, 1])
                max_y = np.max(points[:, 1])

                center_x, center_y = vehicle['center']

                # 확장 픽셀 (polygon 주변으로 선분 확장)
                extend_pixels = 30

                # 1. 수직 선분들 생성 (차량 좌우)
                # 좌측 수직선
                left_x = min_x - extend_pixels
                vertical_slope_left = float('inf')  # 수직선 표현을 위한 특수값
                lines.append(('vertical', left_x, min_y - extend_pixels, max_y + extend_pixels))

                # 우측 수직선
                right_x = max_x + extend_pixels
                lines.append(('vertical', right_x, min_y - extend_pixels, max_y + extend_pixels))

                # 중앙 수직선 (차량 중심)
                lines.append(('vertical', center_x, min_y - extend_pixels, max_y + extend_pixels))

                # 2. 수평 선분들 생성 (차량 상하)
                # 상단 수평선
                top_y = min_y - extend_pixels
                lines.append(('horizontal', top_y, min_x - extend_pixels, max_x + extend_pixels))

                # 하단 수평선
                bottom_y = max_y + extend_pixels
                lines.append(('horizontal', bottom_y, min_x - extend_pixels, max_x + extend_pixels))

                # 중앙 수평선 (차량 중심)
                lines.append(('horizontal', center_y, min_x - extend_pixels, max_x + extend_pixels))

                print(f"   차량 {vehicle_idx+1}: 수직선 3개, 수평선 3개 생성")

        print(f"🔍 총 생성된 선분: {len(lines)}개")

        # 기존 형식으로 변환 (시각화를 위해)
        converted_lines = []
        for line_data in lines:
            if line_data[0] == 'vertical':
                # 수직선: x=상수 형태를 기울기가 매우 큰 직선으로 변환
                x = line_data[1]
                y1, y2 = line_data[2], line_data[3]
                slope = 1000  # 매우 큰 기울기로 수직선 근사
                intercept = y1 - slope * x
                converted_lines.append((slope, intercept))
            else:  # horizontal
                # 수평선: y=상수 형태
                y = line_data[1]
                x1, x2 = line_data[2], line_data[3]
                slope = 0  # 수평선
                intercept = y
                converted_lines.append((slope, intercept))

        self.detected_lines = converted_lines
        self.line_segments = lines  # 원본 선분 정보 저장
        return converted_lines

    def apply_perspective_transform(self, image, lines: List[Tuple[float, float]]) -> Optional[np.ndarray]:
        """검출된 수직/수평 선분들을 이용한 perspective transform 적용"""
        if not hasattr(self, 'line_segments') or not self.line_segments:
            print("⚠️ 선분 정보가 없습니다")
            return None

        h, w = image.shape[:2]

        # 수직선과 수평선 분리
        vertical_lines = []
        horizontal_lines = []

        for line_data in self.line_segments:
            if line_data[0] == 'vertical':
                vertical_lines.append(line_data)
            elif line_data[0] == 'horizontal':
                horizontal_lines.append(line_data)

        print(f"� Transform용 선분: 수직 {len(vertical_lines)}개, 수평 {len(horizontal_lines)}개")

        if len(vertical_lines) >= 2 and len(horizontal_lines) >= 2:
            # 수직선 2개와 수평선 2개를 선택하여 직사각형 격자 생성
            print("🔲 직사각형 격자 기반 변환")

            # 가장 바깥쪽 수직선들 선택
            v_lines_sorted = sorted(vertical_lines, key=lambda x: x[1])  # x 좌표로 정렬
            left_vertical = v_lines_sorted[0]
            right_vertical = v_lines_sorted[-1]

            # 가장 바깥쪽 수평선들 선택
            h_lines_sorted = sorted(horizontal_lines, key=lambda x: x[1])  # y 좌표로 정렬
            top_horizontal = h_lines_sorted[0]
            bottom_horizontal = h_lines_sorted[-1]

            # 교점들 계산
            left_x = left_vertical[1]
            right_x = right_vertical[1]
            top_y = top_horizontal[1]
            bottom_y = bottom_horizontal[1]

            # 원본 포인트 (현재 기울어진 격자의 교점들)
            src_points = np.array([
                [left_x, top_y],      # 좌상
                [right_x, top_y],     # 우상
                [left_x, bottom_y],   # 좌하
                [right_x, bottom_y]   # 우하
            ], dtype=np.float32)

            # 목표 포인트 (완전한 직사각형)
            margin = 50
            dst_points = np.array([
                [margin, margin],                    # 좌상
                [w - margin, margin],                # 우상
                [margin, h - margin],                # 좌하
                [w - margin, h - margin]             # 우하
            ], dtype=np.float32)

        elif len(vertical_lines) >= 2:
            # 수직선만으로 변환
            print("📏 수직선 기반 변환")

            v_lines_sorted = sorted(vertical_lines, key=lambda x: x[1])
            left_vertical = v_lines_sorted[0]
            right_vertical = v_lines_sorted[-1]

            left_x = left_vertical[1]
            right_x = right_vertical[1]

            # 중앙 y 좌표들
            top_y = h * 0.2
            bottom_y = h * 0.8

            src_points = np.array([
                [left_x, top_y],
                [right_x, top_y],
                [left_x, bottom_y],
                [right_x, bottom_y]
            ], dtype=np.float32)

            # 수직으로 만들기
            center_x = w * 0.5
            width_half = abs(right_x - left_x) * 0.5

            dst_points = np.array([
                [center_x - width_half, top_y],
                [center_x + width_half, top_y],
                [center_x - width_half, bottom_y],
                [center_x + width_half, bottom_y]
            ], dtype=np.float32)

        elif len(horizontal_lines) >= 2:
            # 수평선만으로 변환
            print("📏 수평선 기반 변환")

            h_lines_sorted = sorted(horizontal_lines, key=lambda x: x[1])
            top_horizontal = h_lines_sorted[0]
            bottom_horizontal = h_lines_sorted[-1]

            top_y = top_horizontal[1]
            bottom_y = bottom_horizontal[1]

            # 중앙 x 좌표들
            left_x = w * 0.2
            right_x = w * 0.8

            src_points = np.array([
                [left_x, top_y],
                [right_x, top_y],
                [left_x, bottom_y],
                [right_x, bottom_y]
            ], dtype=np.float32)

            # 수평으로 만들기
            center_y = h * 0.5
            height_half = abs(bottom_y - top_y) * 0.5

            dst_points = np.array([
                [left_x, center_y - height_half],
                [right_x, center_y - height_half],
                [left_x, center_y + height_half],
                [right_x, center_y + height_half]
            ], dtype=np.float32)
        else:
            print("⚠️ Transform을 위한 충분한 선분이 없습니다")
            return None

        try:
            # Perspective 변환 행렬 계산
            self.perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

            # 변환 적용
            transformed = cv2.warpPerspective(image, self.perspective_matrix, (w, h))

            print("✅ 수직/수평 선분 기반 변환 완료")
            return transformed

        except Exception as e:
            print(f"⚠️ Transform 계산 오류: {e}")
            return None

    def visualize_detections(self, image, detections, save_path: Optional[str] = None,
                           transformed_image: Optional[np.ndarray] = None,
                           lines: Optional[List[Tuple[float, float]]] = None):
        """감지 결과 시각화"""
        # OpenCV 이미지를 matplotlib용으로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 차량만 필터링
        vehicles = [d for d in detections if d['is_vehicle']]
        all_objects = detections

        # Figure 설정 - 2x3 레이아웃으로 확장
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle('YOLO-OBB 감지 결과 및 Perspective Transform 모니터링', fontsize=16, fontweight='bold')

        # 1. 원본 이미지
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title(f'원본 이미지 ({image.shape[1]}x{image.shape[0]})')
        axes[0, 0].axis('off')

        # 2. 모든 객체 감지 결과
        axes[0, 1].imshow(image_rgb)
        axes[0, 1].set_title(f'모든 객체 감지 ({len(all_objects)}개)')

        for detection in all_objects:
            # OBB 다각형 그리기
            polygon = Polygon(detection['obb_points'],
                            fill=False,
                            edgecolor='red' if detection['is_vehicle'] else 'blue',
                            linewidth=2)
            axes[0, 1].add_patch(polygon)

            # # 라벨 추가
            # x, y = detection['center']
            # label = f"{detection['class_name']}\n{detection['confidence']:.2f}"
            # axes[0, 1].text(x, y, label,
            #                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
            #                fontsize=8, ha='center')

        axes[0, 1].axis('off')

        # 3. 차량만 상세 표시
        axes[1, 0].imshow(image_rgb)
        axes[1, 0].set_title(f'차량 감지 상세 ({len(vehicles)}대)')

        for i, vehicle in enumerate(vehicles):
            # OBB 다각형 그리기
            color = self.colors.get(vehicle['class_id'], (128, 128, 128))
            color_normalized = tuple(c/255.0 for c in color)  # matplotlib용 색상 정규화

            polygon = Polygon(vehicle['obb_points'],
                            fill=False,
                            edgecolor=color_normalized,
                            linewidth=3)
            axes[1, 0].add_patch(polygon)

            # 상세 정보 표시
            x, y = vehicle['center']
            w, h = vehicle['size']
            info = f"V{i+1}: {vehicle['class_name']}\n"
            info += f"크기: {w:.0f}x{h:.0f}\n"
            info += f"각도: {vehicle['angle']:.1f}°\n"
            info += f"신뢰도: {vehicle['confidence']:.2f}"

            axes[1, 0].text(x, y, info,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                           fontsize=7, ha='center', va='center')

            # 중심점 표시
            axes[1, 0].plot(x, y, 'ro', markersize=5)

            # ID 표시
            axes[1, 0].text(x, y-30, f"ID: {i+1}",
                           bbox=dict(boxstyle="round,pad=0.2", facecolor=color_normalized, alpha=0.7),
                           fontsize=8, ha='center', fontweight='bold')

        axes[1, 0].axis('off')

        # 4. 통계 차트
        if vehicles:
            # 차량 크기 분포
            widths = [v['size'][0] for v in vehicles]
            heights = [v['size'][1] for v in vehicles]
            areas = [v['area'] for v in vehicles]

            axes[1, 1].scatter(widths, heights, c=areas, cmap='viridis', s=100, alpha=0.7)
            axes[1, 1].set_xlabel('폭 (픽셀)')
            axes[1, 1].set_ylabel('높이 (픽셀)')
            axes[1, 1].set_title('차량 크기 분포')
            axes[1, 1].grid(True, alpha=0.3)

            # 차량별 라벨
            for i, vehicle in enumerate(vehicles):
                w, h = vehicle['size']
                axes[1, 1].annotate(f'V{i+1}', (w, h),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=8, fontweight='bold')

            # 색상바 추가
            cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
            cbar.set_label('면적 (픽셀²)')
        else:
            axes[1, 1].text(0.5, 0.5, '감지된 차량 없음',
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=14, fontweight='bold')
            axes[1, 1].set_title('차량 크기 분포')

        # 5. 검출된 직선 표시
        axes[0, 2].imshow(image_rgb)
        axes[0, 2].set_title(f'검출된 선분 ({len(lines) if lines else 0}개)')

        # 원본 선분 정보가 있으면 그것을 사용, 없으면 변환된 lines 사용
        if hasattr(self, 'line_segments') and self.line_segments:
            # 새로운 선분 형식으로 그리기
            h, w = image.shape[:2]

            vertical_count = 0
            horizontal_count = 0

            for i, line_data in enumerate(self.line_segments):
                if line_data[0] == 'vertical':
                    # 수직선: x = 상수
                    x = line_data[1]
                    y1, y2 = line_data[2], line_data[3]

                    # 이미지 경계 내로 클리핑
                    y1 = max(0, min(h-1, y1))
                    y2 = max(0, min(h-1, y2))

                    color = 'red' if vertical_count % 2 == 0 else 'darkred'
                    axes[0, 2].plot([x, x], [y1, y2], color=color, linewidth=2,
                                   label=f'수직선 {vertical_count+1}')

                    # 라벨 표시
                    mid_y = (y1 + y2) / 2
                    axes[0, 2].text(x + 5, mid_y, f'V{vertical_count+1}',
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
                                   fontsize=8, fontweight='bold', color='white')
                    vertical_count += 1

                elif line_data[0] == 'horizontal':
                    # 수평선: y = 상수
                    y = line_data[1]
                    x1, x2 = line_data[2], line_data[3]

                    # 이미지 경계 내로 클리핑
                    x1 = max(0, min(w-1, x1))
                    x2 = max(0, min(w-1, x2))

                    color = 'blue' if horizontal_count % 2 == 0 else 'darkblue'
                    axes[0, 2].plot([x1, x2], [y, y], color=color, linewidth=2,
                                   label=f'수평선 {horizontal_count+1}')

                    # 라벨 표시
                    mid_x = (x1 + x2) / 2
                    axes[0, 2].text(mid_x, y - 10, f'H{horizontal_count+1}',
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
                                   fontsize=8, fontweight='bold', color='white')
                    horizontal_count += 1

            # 범례를 간단하게 표시
            if vertical_count > 0 or horizontal_count > 0:
                legend_text = f"수직: {vertical_count}개, 수평: {horizontal_count}개"
                axes[0, 2].text(0.02, 0.98, legend_text,
                               transform=axes[0, 2].transAxes,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                               fontsize=10, fontweight='bold', va='top')

        elif lines:
            # 기존 방식으로 표시 (호환성을 위해)
            h, w = image.shape[:2]
            for i, (slope, intercept) in enumerate(lines):
                # 직선을 이미지 경계까지 그리기
                if abs(slope) > 100:  # 수직선에 가까운 경우
                    x = int(-intercept / slope) if slope != 0 else w//2
                    axes[0, 2].plot([x, x], [0, h-1], color='red', linewidth=2,
                                   label=f'수직선 {i+1}')
                else:  # 일반 직선
                    x1, x2 = 0, w
                    y1 = int(slope * x1 + intercept)
                    y2 = int(slope * x2 + intercept)

                    # 이미지 경계 내에서 클리핑
                    if y1 < 0:
                        y1 = 0
                        x1 = int((y1 - intercept) / slope) if abs(slope) > 1e-6 else 0
                    elif y1 >= h:
                        y1 = h - 1
                        x1 = int((y1 - intercept) / slope) if abs(slope) > 1e-6 else 0

                    if y2 < 0:
                        y2 = 0
                        x2 = int((y2 - intercept) / slope) if abs(slope) > 1e-6 else w
                    elif y2 >= h:
                        y2 = h - 1
                        x2 = int((y2 - intercept) / slope) if abs(slope) > 1e-6 else w

                    color = 'blue' if abs(slope) < 0.1 else 'green'
                    axes[0, 2].plot([x1, x2], [y1, y2], color=color, linewidth=2,
                                   label=f'Line {i+1}')
        else:
            axes[0, 2].text(0.5, 0.5, '검출된 직선 없음',
                           ha='center', va='center', transform=axes[0, 2].transAxes,
                           fontsize=14, fontweight='bold')

        axes[0, 2].axis('off')        # 6. Perspective Transform 결과
        if transformed_image is not None:
            transformed_rgb = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
            axes[1, 2].imshow(transformed_rgb)
            axes[1, 2].set_title('Perspective Transform 결과')

            # 변환된 이미지에 격자 오버레이
            h, w = transformed_image.shape[:2]
            grid_size = 50

            # 수직선 그리기
            for x in range(0, w, grid_size):
                axes[1, 2].axvline(x=x, color='cyan', alpha=0.3, linewidth=1)

            # 수평선 그리기
            for y in range(0, h, grid_size):
                axes[1, 2].axhline(y=y, color='cyan', alpha=0.3, linewidth=1)

        else:
            axes[1, 2].text(0.5, 0.5, 'Transform 실패\n(직선 부족)',
                           ha='center', va='center', transform=axes[1, 2].transAxes,
                           fontsize=14, fontweight='bold', color='red')
            axes[1, 2].set_title('Perspective Transform 결과')

        axes[1, 2].axis('off')

        plt.tight_layout()

        # 저장
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 결과 저장: {save_path}")

        # 화면에 표시
        plt.show()

        return fig

    def print_detection_summary(self, detections):
        """감지 결과 요약 출력"""
        vehicles = [d for d in detections if d['is_vehicle']]

        print("\n" + "="*60)
        print("📊 YOLO-OBB 감지 결과 요약")
        print("="*60)

        print(f"🔍 총 감지 객체: {len(detections)}개")
        print(f"🚗 차량: {len(vehicles)}대")

        if vehicles:
            print("\n📋 차량 상세 정보:")
            print("ID | 클래스     | 신뢰도 | 크기(W×H)    | 면적     | 각도   | 중심점")
            print("-" * 70)

            for i, vehicle in enumerate(vehicles):
                w, h = vehicle['size']
                x, y = vehicle['center']
                print(f"{i+1:2d} | {vehicle['class_name']:10s} | {vehicle['confidence']:6.2f} | "
                      f"{w:5.0f}×{h:5.0f} | {vehicle['area']:8.0f} | {vehicle['angle']:6.1f}° | "
                      f"({x:3d},{y:3d})")

            # 크기 통계
            widths = [v['size'][0] for v in vehicles]
            heights = [v['size'][1] for v in vehicles]
            areas = [v['area'] for v in vehicles]

            print(f"\n📏 크기 통계:")
            print(f"  폭: 평균 {np.mean(widths):.1f}px, 표준편차 {np.std(widths):.1f}px")
            print(f"  높이: 평균 {np.mean(heights):.1f}px, 표준편차 {np.std(heights):.1f}px")
            print(f"  면적: 평균 {np.mean(areas):.0f}px², 표준편차 {np.std(areas):.0f}px²")

            # 크기 균일성 계산
            size_uniformity = 100 - (np.std(widths) + np.std(heights)) / (np.mean(widths) + np.mean(heights)) * 100
            print(f"  크기 균일성: {size_uniformity:.1f}%")

        print("="*60)

    def monitor_image(self, image_path: str, conf_threshold: float = 0.25, save_result: bool = True):
        """이미지 모니터링 전체 프로세스"""
        print(f"🚗 YOLO-OBB 모니터링 시작: {image_path}")

        # 1. 감지 실행
        image, results = self.detect_with_obb(image_path, conf_threshold)
        if results is None:
            return None, None

        # 2. 결과 추출
        detections = self.extract_detections(results)

        # 3. 요약 출력
        self.print_detection_summary(detections)

        # 4. 직선 검출 및 Perspective Transform
        lines = self.detect_lines_from_obb(detections)
        transformed_image = None
        if image is not None:
            transformed_image = self.apply_perspective_transform(image, lines)

        # 5. 시각화
        save_path = None
        if save_result:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = f"obb_monitor_result_{base_name}.png"

        fig = self.visualize_detections(image, detections, save_path, transformed_image, lines)

        return detections, fig


def main():
    """메인 실행 함수"""
    print("🔍 YOLO-OBB 실시간 모니터링 시스템")
    print("="*50)

    # 모니터 초기화
    monitor = OBBMonitor()

    # 테스트 이미지 경로
    image_path = "../parkinglot1.jpg"

    if not os.path.exists(image_path):
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
        print("💡 다음 경로들을 확인해보세요:")
        possible_paths = [
            "../parkinglot1.jpg",
            "../../parkinglot1.jpg",
            "../data/parkinglot1.jpg",
            "parkinglot1.jpg"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ 발견: {path}")
                image_path = path
                break
            else:
                print(f"❌ 없음: {path}")

        if not os.path.exists(image_path):
            return

    # 다양한 신뢰도 임계값으로 테스트
    confidence_levels = [0.1, 0.25, 0.5]

    for conf in confidence_levels:
        print(f"\n🎯 신뢰도 임계값: {conf}")
        print("-" * 30)

        detections, fig = monitor.monitor_image(image_path, conf_threshold=conf, save_result=True)

        if detections:
            vehicles = [d for d in detections if d['is_vehicle']]
            if vehicles:
                print(f"✅ {len(vehicles)}대 차량 감지됨")
            else:
                print("⚠️ 차량이 감지되지 않았습니다")
        else:
            print("❌ 감지 결과 없음")

        # 사용자 입력 대기 (다음 테스트로 넘어가기)
        if conf != confidence_levels[-1]:  # 마지막이 아니면
            input("\n⏸️ 다음 테스트로 넘어가려면 Enter를 누르세요...")

    print("\n🎉 모든 테스트 완료!")
    print("📁 결과 파일들:")
    for file in os.listdir("."):
        if file.startswith("obb_monitor_result_"):
            print(f"   💾 {file}")


if __name__ == "__main__":
    main()