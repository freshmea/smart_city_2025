"""
Perspective Transform 기반 정밀 주차장 분석 시스템
YOLO-OBB 결과를 활용한 perspective correction과 균일한 주차 구역 인식
"""

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from ultralytics import YOLO


class ParkingSpotStatus(Enum):
    EMPTY = "empty"
    OCCUPIED = "occupied"
    UNKNOWN = "unknown"

@dataclass
class Vehicle:
    """차량 정보"""
    bbox: Tuple[int, int, int, int]
    obb: Optional[np.ndarray]  # Oriented Bounding Box
    confidence: float
    size: Tuple[float, float]  # (width, height)
    center: Tuple[int, int]

@dataclass
class ParkingSpot:
    """주차 구역 정보"""
    id: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    area: float
    status: ParkingSpotStatus
    confidence: float
    grid_position: Optional[Tuple[int, int]] = None  # (row, col)
    corners: Optional[List[Tuple[int, int]]] = None

class PerspectiveParkingDetector:
    """Perspective Transform 기반 주차장 감지 시스템"""

    def __init__(self, yolo_obb_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 디바이스: {self.device}")

        # YOLO-OBB 모델 로드
        if yolo_obb_path and os.path.exists(yolo_obb_path):
            try:
                self.yolo_model = YOLO(yolo_obb_path)
                print(f"✅ YOLO-OBB 모델 로드: {yolo_obb_path}")
            except Exception as e:
                print(f"⚠️ YOLO-OBB 로드 실패: {e}")
                self.yolo_model = None
        else:
            self.yolo_model = None

        # 표준 주차 구역 크기 (미터 단위)
        self.standard_parking_width = 2.5  # 2.5m
        self.standard_parking_length = 5.0  # 5.0m

        # 크기 허용 오차 (%)
        self.size_tolerance = 0.3  # 30%

    def detect_vehicles_with_obb(self, image: np.ndarray) -> List[Vehicle]:
        """YOLO-OBB로 차량 감지 및 크기 정규화"""
        if self.yolo_model is None:
            return []

        try:
            # YOLO-OBB 추론
            results = self.yolo_model(image, verbose=False, conf=0.3)
            vehicles = []

            for result in results:
                # OBB 결과 처리
                if hasattr(result, 'obb') and result.obb is not None:
                    for i, (obb, conf, cls) in enumerate(zip(result.obb.xyxyxyxy, result.obb.conf, result.obb.cls)):
                        # 차량 클래스만 (자동차, 트럭, 버스 등)
                        if int(cls) in [2, 3, 5, 7]:  # COCO 클래스
                            # OBB 점들을 numpy 배열로 변환
                            obb_points = obb.cpu().numpy().reshape(-1, 2)

                            # 바운딩 박스 계산
                            x_coords = obb_points[:, 0]
                            y_coords = obb_points[:, 1]
                            x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
                            x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))

                            # 차량 크기 계산 (OBB의 실제 크기)
                            width = np.linalg.norm(obb_points[1] - obb_points[0])
                            height = np.linalg.norm(obb_points[2] - obb_points[1])

                            # 중심점
                            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                            vehicle = Vehicle(
                                bbox=(x1, y1, x2, y2),
                                obb=obb_points,
                                confidence=float(conf),
                                size=(float(width), float(height)),
                                center=center
                            )
                            vehicles.append(vehicle)

                # 일반 박스 결과도 처리 (OBB가 없는 경우)
                elif hasattr(result, 'boxes') and result.boxes is not None:
                    for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                        if int(cls) in [2, 3, 5, 7]:
                            x1, y1, x2, y2 = [int(x) for x in box.cpu().numpy()]
                            width = x2 - x1
                            height = y2 - y1
                            center = ((x1 + x2) // 2, (y1 + y2) // 2)

                            vehicle = Vehicle(
                                bbox=(x1, y1, x2, y2),
                                obb=None,
                                confidence=float(conf),
                                size=(float(width), float(height)),
                                center=center
                            )
                            vehicles.append(vehicle)

            print(f"🚗 YOLO-OBB로 감지된 차량: {len(vehicles)}대")

            # 차량 크기 정규화 및 필터링
            vehicles = self.normalize_vehicle_sizes(vehicles)

            return vehicles

        except Exception as e:
            print(f"⚠️ YOLO-OBB 감지 오류: {e}")
            return []

    def normalize_vehicle_sizes(self, vehicles: List[Vehicle]) -> List[Vehicle]:
        """차량 크기 정규화 및 이상값 제거"""
        if len(vehicles) < 2:
            return vehicles

        # 차량 크기들 수집
        sizes = [(v.size[0], v.size[1]) for v in vehicles]
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]

        # 중간값 계산
        median_width = np.median(widths)
        median_height = np.median(heights)

        print(f"📏 차량 크기 중간값: {median_width:.1f} x {median_height:.1f}")

        # 표준 편차 계산
        std_width = np.std(widths)
        std_height = np.std(heights)

        # 정규화된 차량 목록
        normalized_vehicles = []

        for vehicle in vehicles:
            w, h = vehicle.size

            # 크기가 중간값의 허용 범위 내에 있는지 확인
            width_ratio = abs(w - median_width) / median_width
            height_ratio = abs(h - median_height) / median_height

            if width_ratio <= self.size_tolerance and height_ratio <= self.size_tolerance:
                normalized_vehicles.append(vehicle)
            else:
                print(f"⚠️ 크기 이상값 제거: {w:.1f}x{h:.1f} (기준: {median_width:.1f}x{median_height:.1f})")

        print(f"✅ 정규화 후 차량: {len(normalized_vehicles)}대")
        return normalized_vehicles

    def estimate_perspective_transform(self, vehicles: List[Vehicle], image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """차량 위치를 기반으로 perspective transform 매트릭스 계산"""
        if len(vehicles) < 4:
            print("⚠️ Perspective transform을 위한 차량이 부족합니다 (최소 4대 필요)")
            return None

        height, width = image_shape[:2]

        # 차량 중심점들을 사용하여 주차장 평면 추정
        vehicle_centers = np.array([v.center for v in vehicles], dtype=np.float32)

        try:
            # 차량들이 격자 패턴을 형성한다고 가정
            # 가장 바깥쪽 4개 점을 찾아서 사각형 형성

            # 극값 찾기
            min_x_idx = np.argmin(vehicle_centers[:, 0])
            max_x_idx = np.argmax(vehicle_centers[:, 0])
            min_y_idx = np.argmin(vehicle_centers[:, 1])
            max_y_idx = np.argmax(vehicle_centers[:, 1])

            # 모서리 점들 추정
            corners = []

            # 좌상단: 최소 x, 최소 y 근처
            top_left = vehicle_centers[np.argmin(vehicle_centers[:, 0] + vehicle_centers[:, 1])]

            # 우상단: 최대 x, 최소 y 근처
            top_right = vehicle_centers[np.argmin(-vehicle_centers[:, 0] + vehicle_centers[:, 1])]

            # 좌하단: 최소 x, 최대 y 근처
            bottom_left = vehicle_centers[np.argmin(vehicle_centers[:, 0] - vehicle_centers[:, 1])]

            # 우하단: 최대 x, 최대 y 근처
            bottom_right = vehicle_centers[np.argmax(vehicle_centers[:, 0] + vehicle_centers[:, 1])]

            src_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

            # 목표 사각형 (정면에서 본 모습)
            margin = 50
            dst_points = np.array([
                [margin, margin],
                [width - margin, margin],
                [width - margin, height - margin],
                [margin, height - margin]
            ], dtype=np.float32)

            # Perspective transform 매트릭스 계산
            transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

            print("✅ Perspective transform 매트릭스 계산 완료")
            return transform_matrix

        except Exception as e:
            print(f"⚠️ Perspective transform 계산 실패: {e}")
            return None

    def apply_perspective_correction(self, image: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """Perspective correction 적용"""
        height, width = image.shape[:2]
        corrected = cv2.warpPerspective(image, transform_matrix, (width, height))
        return corrected

    def detect_parking_grid_from_corrected_image(self, corrected_image: np.ndarray,
                                               vehicles: List[Vehicle]) -> List[ParkingSpot]:
        """보정된 이미지에서 균일한 주차 격자 생성"""
        height, width = corrected_image.shape[:2]

        if not vehicles:
            # 차량이 없으면 기본 격자 생성
            return self.generate_default_uniform_grid(corrected_image.shape)

        # 차량 위치를 기반으로 격자 패턴 추정
        vehicle_centers = [v.center for v in vehicles]

        # X, Y 좌표별 클러스터링
        x_coords = [c[0] for c in vehicle_centers]
        y_coords = [c[1] for c in vehicle_centers]

        # DBSCAN을 사용하여 격자 라인 추정
        def cluster_coordinates(coords, min_samples=1):
            coords_array = np.array(coords).reshape(-1, 1)
            clustering = DBSCAN(eps=50, min_samples=min_samples).fit(coords_array)

            clusters = {}
            for i, label in enumerate(clustering.labels_):
                if label != -1:  # 노이즈가 아닌 경우
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(coords[i])

            # 각 클러스터의 중심값 계산
            cluster_centers = []
            for cluster_coords in clusters.values():
                cluster_centers.append(int(np.mean(cluster_coords)))

            return sorted(cluster_centers)

        # 격자 라인 좌표 계산
        grid_x_lines = cluster_coordinates(x_coords)
        grid_y_lines = cluster_coordinates(y_coords)

        print(f"📐 감지된 격자: X축 {len(grid_x_lines)}개, Y축 {len(grid_y_lines)}개 라인")

        # 격자가 부족하면 보완
        if len(grid_x_lines) < 3:
            grid_x_lines = list(range(width//8, width, width//4))
        if len(grid_y_lines) < 3:
            grid_y_lines = list(range(height//6, height, height//3))

        # 주차 구역 생성
        parking_spots = []
        spot_id = 1

        for i in range(len(grid_y_lines) - 1):
            for j in range(len(grid_x_lines) - 1):
                x1 = grid_x_lines[j]
                x2 = grid_x_lines[j + 1]
                y1 = grid_y_lines[i]
                y2 = grid_y_lines[i + 1]

                # 크기 검증 (균일한 크기)
                w, h = x2 - x1, y2 - y1
                if 80 < w < 200 and 120 < h < 300:  # 적절한 주차 공간 크기
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    area = w * h

                    spot = ParkingSpot(
                        id=spot_id,
                        bbox=(x1, y1, x2, y2),
                        center=center,
                        area=area,
                        status=ParkingSpotStatus.UNKNOWN,
                        confidence=0.8,
                        grid_position=(i, j),
                        corners=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    )
                    parking_spots.append(spot)
                    spot_id += 1

        print(f"🅿️ 생성된 균일 주차 구역: {len(parking_spots)}개")
        return parking_spots

    def generate_default_uniform_grid(self, image_shape: Tuple[int, int],
                                    rows: int = 6, cols: int = 10) -> List[ParkingSpot]:
        """기본 균일 격자 생성"""
        height, width = image_shape[:2]

        # 여백 설정
        margin_x = width // 10
        margin_y = height // 8

        effective_width = width - 2 * margin_x
        effective_height = height - 2 * margin_y

        # 균일한 주차 구역 크기 계산
        spot_width = effective_width // cols
        spot_height = effective_height // rows

        parking_spots = []
        spot_id = 1

        for row in range(rows):
            for col in range(cols):
                x1 = margin_x + col * spot_width
                y1 = margin_y + row * spot_height
                x2 = x1 + spot_width - 5  # 약간의 간격
                y2 = y1 + spot_height - 5

                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                area = (x2 - x1) * (y2 - y1)

                spot = ParkingSpot(
                    id=spot_id,
                    bbox=(x1, y1, x2, y2),
                    center=center,
                    area=area,
                    status=ParkingSpotStatus.UNKNOWN,
                    confidence=0.7,
                    grid_position=(row, col),
                    corners=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                )
                parking_spots.append(spot)
                spot_id += 1

        return parking_spots

    def analyze_occupancy_with_id(self, vehicles: List[Vehicle],
                                parking_spots: List[ParkingSpot]) -> List[ParkingSpot]:
        """ID 기반 정확한 주차 점유 분석"""

        # 각 차량에 대해 가장 적합한 주차 구역 찾기
        for vehicle in vehicles:
            v_x1, v_y1, v_x2, v_y2 = vehicle.bbox
            v_center = vehicle.center

            best_spot = None
            best_overlap_ratio = 0

            for spot in parking_spots:
                s_x1, s_y1, s_x2, s_y2 = spot.bbox

                # 겹치는 영역 계산
                overlap_x1 = max(v_x1, s_x1)
                overlap_y1 = max(v_y1, s_y1)
                overlap_x2 = min(v_x2, s_x2)
                overlap_y2 = min(v_y2, s_y2)

                if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    vehicle_area = (v_x2 - v_x1) * (v_y2 - v_y1)
                    spot_area = (s_x2 - s_x1) * (s_y2 - s_y1)

                    # 겹침 비율 계산 (차량 기준 + 주차구역 기준)
                    vehicle_overlap_ratio = overlap_area / vehicle_area
                    spot_overlap_ratio = overlap_area / spot_area

                    # 양방향 겹침 비율의 평균
                    combined_ratio = (vehicle_overlap_ratio + spot_overlap_ratio) / 2

                    if combined_ratio > best_overlap_ratio and combined_ratio > 0.3:
                        best_overlap_ratio = combined_ratio
                        best_spot = spot

            # 가장 적합한 주차 구역에 점유 표시
            if best_spot:
                best_spot.status = ParkingSpotStatus.OCCUPIED
                best_spot.confidence = min(1.0, best_spot.confidence + vehicle.confidence * 0.4)
                print(f"🚗 차량이 주차구역 P{best_spot.id}에 주차됨 (겹침: {best_overlap_ratio:.2f})")

        # 나머지는 빈 자리로 설정
        for spot in parking_spots:
            if spot.status == ParkingSpotStatus.UNKNOWN:
                spot.status = ParkingSpotStatus.EMPTY

        return parking_spots

    def process_parking_lot_with_perspective(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, List[ParkingSpot], Dict]:
        """Perspective transform을 적용한 전체 처리 파이프라인"""
        print(f"🚗 Perspective 기반 주차장 분석 시작: {image_path}")

        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지 로드 실패: {image_path}")

        print(f"📐 이미지 크기: {image.shape}")

        # 1. 차량 감지 (YOLO-OBB)
        vehicles = self.detect_vehicles_with_obb(image)

        # 2. Perspective transform 계산
        transform_matrix = self.estimate_perspective_transform(vehicles, image.shape)

        corrected_image = image.copy()
        if transform_matrix is not None:
            # 3. Perspective correction 적용
            corrected_image = self.apply_perspective_correction(image, transform_matrix)

            # 차량 좌표도 변환
            transformed_vehicles = []
            for vehicle in vehicles:
                # 차량 중심점 변환
                center_point = np.array([[vehicle.center]], dtype=np.float32)
                transformed_center = cv2.perspectiveTransform(center_point, transform_matrix)
                new_center = (int(transformed_center[0][0][0]), int(transformed_center[0][0][1]))

                # 바운딩 박스 변환
                bbox_points = np.array([
                    [[vehicle.bbox[0], vehicle.bbox[1]]],
                    [[vehicle.bbox[2], vehicle.bbox[3]]]
                ], dtype=np.float32)
                transformed_bbox = cv2.perspectiveTransform(bbox_points, transform_matrix)

                new_bbox = (
                    int(transformed_bbox[0][0][0]), int(transformed_bbox[0][0][1]),
                    int(transformed_bbox[1][0][0]), int(transformed_bbox[1][0][1])
                )

                # 변환된 차량 정보 생성
                transformed_vehicle = Vehicle(
                    bbox=new_bbox,
                    obb=vehicle.obb,
                    confidence=vehicle.confidence,
                    size=vehicle.size,
                    center=new_center
                )
                transformed_vehicles.append(transformed_vehicle)

            vehicles = transformed_vehicles

        # 4. 균일한 주차 격자 생성
        parking_spots = self.detect_parking_grid_from_corrected_image(corrected_image, vehicles)

        # 5. 점유 상태 분석
        parking_spots = self.analyze_occupancy_with_id(vehicles, parking_spots)

        # 6. 통계 계산
        empty_count = sum(1 for spot in parking_spots if spot.status == ParkingSpotStatus.EMPTY)
        occupied_count = sum(1 for spot in parking_spots if spot.status == ParkingSpotStatus.OCCUPIED)

        stats = {
            'total_spots': len(parking_spots),
            'empty_spots': empty_count,
            'occupied_spots': occupied_count,
            'vehicles_detected': len(vehicles),
            'occupancy_rate': occupied_count / len(parking_spots) * 100 if parking_spots else 0,
            'perspective_corrected': transform_matrix is not None,
            'uniform_grid': True,
            'average_spot_area': np.mean([spot.area for spot in parking_spots]) if parking_spots else 0
        }

        print(f"✅ 분석 완료: {stats}")

        return image, corrected_image, parking_spots, stats

    def draw_results_with_perspective(self, original_image: np.ndarray, corrected_image: np.ndarray,
                                    parking_spots: List[ParkingSpot], vehicles: List[Vehicle] = None) -> Tuple[np.ndarray, np.ndarray]:
        """원본과 보정된 이미지 모두에 결과 시각화"""

        # 원본 이미지 결과
        original_result = original_image.copy()

        # 보정된 이미지 결과
        corrected_result = corrected_image.copy()

        # 주차 구역 그리기 (보정된 이미지에)
        for spot in parking_spots:
            x1, y1, x2, y2 = spot.bbox

            # 상태별 색상
            if spot.status == ParkingSpotStatus.EMPTY:
                color = (0, 255, 0)  # 녹색
                text = "EMPTY"
            elif spot.status == ParkingSpotStatus.OCCUPIED:
                color = (0, 0, 255)  # 빨간색
                text = "OCCUPIED"
            else:
                color = (0, 255, 255)  # 노란색
                text = "UNKNOWN"

            # 박스 그리기
            cv2.rectangle(corrected_result, (x1, y1), (x2, y2), color, 2)

            # ID와 상태 텍스트
            cv2.putText(corrected_result, f"P{spot.id:02d}", (x1+5, y1+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(corrected_result, text, (x1+5, y1+40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # 격자 위치 표시 (있는 경우)
            if spot.grid_position:
                row, col = spot.grid_position
                cv2.putText(corrected_result, f"({row},{col})", (x1+5, y1+55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # 차량 박스 그리기
        if vehicles:
            for i, vehicle in enumerate(vehicles):
                x1, y1, x2, y2 = vehicle.bbox
                cv2.rectangle(corrected_result, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(corrected_result, f"V{i+1} ({vehicle.confidence:.2f})",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                # 차량 크기 정보
                w, h = vehicle.size
                cv2.putText(corrected_result, f"{w:.0f}x{h:.0f}",
                           (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        return original_result, corrected_result

    def save_results_with_perspective(self, original_image: np.ndarray, corrected_image: np.ndarray,
                                    parking_spots: List[ParkingSpot], stats: Dict, vehicles: List[Vehicle] = None,
                                    output_dir: str = "perspective_results"):
        """결과 저장"""
        os.makedirs(output_dir, exist_ok=True)

        # 결과 이미지
        original_result, corrected_result = self.draw_results_with_perspective(
            original_image, corrected_image, parking_spots, vehicles)

        cv2.imwrite(f"{output_dir}/perspective_original.jpg", original_result)
        cv2.imwrite(f"{output_dir}/perspective_corrected.jpg", corrected_result)

        # JSON 데이터
        json_data = {
            'statistics': stats,
            'parking_spots': [
                {
                    'id': spot.id,
                    'bbox': [int(x) for x in spot.bbox],
                    'center': [int(x) for x in spot.center],
                    'area': float(spot.area),
                    'status': spot.status.value,
                    'confidence': float(spot.confidence),
                    'grid_position': spot.grid_position,
                    'corners': spot.corners
                }
                for spot in parking_spots
            ],
            'vehicles': [
                {
                    'bbox': [int(x) for x in vehicle.bbox],
                    'center': [int(x) for x in vehicle.center],
                    'size': [float(x) for x in vehicle.size],
                    'confidence': float(vehicle.confidence)
                }
                for vehicle in vehicles or []
            ]
        }

        with open(f"{output_dir}/perspective_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"💾 결과 저장 완료: {output_dir}/")


def main():
    """메인 실행"""
    print("🚗 Perspective Transform 기반 주차장 분석 시스템")
    print("=" * 50)

    # 모델 경로
    yolo_path = "../../yolov8n-obb.pt"

    # 감지기 초기화
    detector = PerspectiveParkingDetector(yolo_path)

    # 이미지 처리
    image_path = "parkinglot1.jpg"

    if not os.path.exists(image_path):
        print(f"❌ 이미지 없음: {image_path}")
        return

    try:
        # 분석 실행
        original, corrected, spots, stats = detector.process_parking_lot_with_perspective(image_path)

        # 차량 정보도 다시 가져오기
        vehicles = detector.detect_vehicles_with_obb(corrected)

        # 결과 저장
        detector.save_results_with_perspective(original, corrected, spots, stats, vehicles)

        # 결과 출력
        print("\n" + "=" * 50)
        print("📊 최종 결과")
        print("=" * 50)
        print(f"🅿️  총 주차구역: {stats['total_spots']}개")
        print(f"🟢 빈 자리: {stats['empty_spots']}개")
        print(f"🔴 점유된 자리: {stats['occupied_spots']}개")
        print(f"🚗 감지된 차량: {stats['vehicles_detected']}대")
        print(f"📈 점유율: {stats['occupancy_rate']:.1f}%")
        print(f"🔧 Perspective 보정: {'✅' if stats['perspective_corrected'] else '❌'}")
        print(f"📐 균일 격자: {'✅' if stats['uniform_grid'] else '❌'}")
        print(f"📏 평균 구역 크기: {stats['average_spot_area']:.0f} 픽셀²")

        # 시각화
        plt.figure(figsize=(20, 12))

        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("원본 이미지")
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
        plt.title("Perspective 보정된 이미지")
        plt.axis('off')

        # 결과 이미지들
        original_result, corrected_result = detector.draw_results_with_perspective(original, corrected, spots, vehicles)

        plt.subplot(2, 3, 3)
        plt.imshow(cv2.cvtColor(corrected_result, cv2.COLOR_BGR2RGB))
        plt.title(f"분석 결과 ({stats['total_spots']}개 구역)")
        plt.axis('off')

        plt.subplot(2, 3, 4)
        labels = ['빈 자리', '점유']
        sizes = [stats['empty_spots'], stats['occupied_spots']]
        colors = ['lightgreen', 'lightcoral']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        plt.title("주차 현황")

        plt.subplot(2, 3, 5)
        categories = ['총 구역', '빈 자리', '점유', '차량']
        values = [stats['total_spots'], stats['empty_spots'],
                 stats['occupied_spots'], stats['vehicles_detected']]
        bars = plt.bar(categories, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        plt.title("통계 요약")
        plt.ylabel("개수")

        # 막대 위에 값 표시
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(value), ha='center', va='bottom')

        plt.subplot(2, 3, 6)
        # 주차 구역별 상태 히트맵
        if spots:
            max_row = max(spot.grid_position[0] for spot in spots if spot.grid_position) + 1
            max_col = max(spot.grid_position[1] for spot in spots if spot.grid_position) + 1

            heatmap = np.zeros((max_row, max_col))
            for spot in spots:
                if spot.grid_position:
                    row, col = spot.grid_position
                    if spot.status == ParkingSpotStatus.OCCUPIED:
                        heatmap[row, col] = 1
                    elif spot.status == ParkingSpotStatus.EMPTY:
                        heatmap[row, col] = 0.5

            plt.imshow(heatmap, cmap='RdYlGn_r', aspect='auto')
            plt.title("주차 구역 히트맵")
            plt.xlabel("열")
            plt.ylabel("행")
            plt.colorbar(label="점유 상태")

        plt.tight_layout()
        plt.savefig("perspective_parking_analysis.png", dpi=150, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()