"""
실제 주차장 이미지에 특화된 고정밀 감지 시스템
컬러 정보, 그림자, 실제 주차선 패턴을 고려한 알고리즘
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
class ParkingSpot:
    id: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    area: float
    status: ParkingSpotStatus
    confidence: float
    color_features: Optional[Dict] = None

class RealWorldParkingDetector:
    """실제 주차장에 특화된 고정밀 감지 시스템"""

    def __init__(self, yolo_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 디바이스: {self.device}")

        # YOLO 모델 로드
        if yolo_path and os.path.exists(yolo_path):
            try:
                self.yolo_model = YOLO(yolo_path)
                print(f"✅ YOLO 모델 로드: {yolo_path}")
            except:
                self.yolo_model = None
        else:
            self.yolo_model = None

    def advanced_preprocessing(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """고급 전처리 with 컬러 정보 활용"""
        results = {}

        # 원본
        results['original'] = image.copy()

        # 색상 공간 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        results['hsv'] = hsv
        results['lab'] = lab
        results['gray'] = gray

        # 밝기 정규화
        normalized = cv2.equalizeHist(gray)
        results['normalized'] = normalized

        # CLAHE with 더 강한 설정
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        results['enhanced'] = enhanced

        # 그림자 제거 (LAB 색상 공간 활용)
        l_channel = lab[:,:,0]
        shadow_removed = cv2.bilateralFilter(l_channel, 9, 75, 75)
        results['shadow_removed'] = shadow_removed

        # 노면 색상 감지 (HSV에서 회색/흰색 범위)
        lower_road = np.array([0, 0, 100])
        upper_road = np.array([180, 30, 255])
        road_mask = cv2.inRange(hsv, lower_road, upper_road)
        results['road_mask'] = road_mask

        # 주차선 색상 감지 (흰색/노란색)
        # 흰색 범위
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # 노란색 범위
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        line_mask = cv2.bitwise_or(white_mask, yellow_mask)
        results['line_mask'] = line_mask

        # 엣지 검출 (다중 방법)
        # Canny with 자동 임계값
        v = np.median(enhanced)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        auto_canny = cv2.Canny(enhanced, lower, upper)
        results['auto_canny'] = auto_canny

        # Sobel 필터
        sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobelx**2 + sobely**2)
        sobel_combined = np.uint8(sobel_combined / sobel_combined.max() * 255)
        results['sobel'] = sobel_combined

        # 형태학적 연산
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))  # 수평선 강조
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))  # 수직선 강조

        horizontal_lines = cv2.morphologyEx(auto_canny, cv2.MORPH_OPEN, kernel_h)
        vertical_lines = cv2.morphologyEx(auto_canny, cv2.MORPH_OPEN, kernel_v)

        results['horizontal_morph'] = horizontal_lines
        results['vertical_morph'] = vertical_lines

        return results

    def detect_parking_lines_advanced(self, processed: Dict[str, np.ndarray]) -> Tuple[List, List]:
        """고급 주차선 감지"""

        # 주차선 마스크와 형태학적 결과 결합
        line_enhanced = cv2.bitwise_or(
            processed['line_mask'],
            cv2.bitwise_or(processed['horizontal_morph'], processed['vertical_morph'])
        )

        # 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        line_enhanced = cv2.morphologyEx(line_enhanced, cv2.MORPH_CLOSE, kernel)

        # 다단계 Hough 변환
        all_lines = []

        # 1단계: 강한 선 감지
        lines1 = cv2.HoughLinesP(
            line_enhanced, 1, np.pi/180, threshold=100,
            minLineLength=80, maxLineGap=20
        )
        if lines1 is not None:
            all_lines.extend(lines1)

        # 2단계: 약한 선 감지 (더 낮은 임계값)
        lines2 = cv2.HoughLinesP(
            line_enhanced, 1, np.pi/180, threshold=50,
            minLineLength=40, maxLineGap=30
        )
        if lines2 is not None:
            all_lines.extend(lines2)

        # 3단계: 수직선 특화 감지 (수직 방향 강조)
        vertical_enhanced = processed['vertical_morph']
        lines3 = cv2.HoughLinesP(
            vertical_enhanced, 1, np.pi/180, threshold=30,
            minLineLength=30, maxLineGap=40
        )
        if lines3 is not None:
            all_lines.extend(lines3)

        if not all_lines:
            return [], []

        # 선 분류 및 필터링
        horizontal_lines = []
        vertical_lines = []

        for line in all_lines:
            x1, y1, x2, y2 = line[0]

            # 선의 각도와 길이 계산
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # 최소 길이 필터
            if length < 20:
                continue

            # 각도별 분류 (더 관대한 기준)
            if abs(angle) <= 20 or abs(angle) >= 160:  # 수평선
                horizontal_lines.append((x1, y1, x2, y2, length, angle))
            elif 70 <= abs(angle) <= 110:  # 수직선
                vertical_lines.append((x1, y1, x2, y2, length, angle))

        # 길이 기준 정렬 및 필터링
        horizontal_lines.sort(key=lambda x: x[4], reverse=True)  # 길이순
        vertical_lines.sort(key=lambda x: x[4], reverse=True)

        # 중복 제거 및 병합
        h_merged = self.merge_lines_advanced(horizontal_lines, is_horizontal=True)
        v_merged = self.merge_lines_advanced(vertical_lines, is_horizontal=False)

        print(f"📏 고급 선 감지: 수평 {len(h_merged)}개, 수직 {len(v_merged)}개")

        return h_merged, v_merged

    def merge_lines_advanced(self, lines: List, is_horizontal: bool,
                           distance_threshold: int = 25, angle_threshold: float = 10) -> List:
        """고급 선 병합 알고리즘"""
        if not lines:
            return []

        # DBSCAN 클러스터링을 사용한 선 그룹화
        if is_horizontal:
            # 수평선은 y 좌표와 각도로 그룹화
            features = np.array([[line[1], line[3], line[5]] for line in lines])  # y1, y2, angle
        else:
            # 수직선은 x 좌표와 각도로 그룹화
            features = np.array([[line[0], line[2], line[5]] for line in lines])  # x1, x2, angle

        if len(features) == 0:
            return []

        # 정규화
        features_normalized = features.copy()
        features_normalized[:, :2] /= distance_threshold  # 거리 정규화
        features_normalized[:, 2] /= angle_threshold      # 각도 정규화

        # DBSCAN 클러스터링
        clustering = DBSCAN(eps=1.0, min_samples=1).fit(features_normalized)
        labels = clustering.labels_

        # 클러스터별로 선 병합
        merged_lines = []
        for cluster_id in set(labels):
            if cluster_id == -1:  # 노이즈
                continue

            cluster_lines = [lines[i] for i in range(len(lines)) if labels[i] == cluster_id]

            # 클러스터 내 선들의 평균으로 대표선 생성
            avg_x1 = int(np.mean([line[0] for line in cluster_lines]))
            avg_y1 = int(np.mean([line[1] for line in cluster_lines]))
            avg_x2 = int(np.mean([line[2] for line in cluster_lines]))
            avg_y2 = int(np.mean([line[3] for line in cluster_lines]))

            # 길이 재계산
            length = np.sqrt((avg_x2 - avg_x1)**2 + (avg_y2 - avg_y1)**2)

            if length > 15:  # 최소 길이 유지
                merged_lines.append((avg_x1, avg_y1, avg_x2, avg_y2))

        return merged_lines

    def detect_parking_regions_by_color(self, image: np.ndarray, processed: Dict) -> List[ParkingSpot]:
        """색상 정보를 활용한 주차 구역 감지"""

        # 아스팔트 색상 마스크 (어두운 회색)
        hsv = processed['hsv']
        lower_asphalt = np.array([0, 0, 20])
        upper_asphalt = np.array([180, 50, 120])
        asphalt_mask = cv2.inRange(hsv, lower_asphalt, upper_asphalt)

        # 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        asphalt_mask = cv2.morphologyEx(asphalt_mask, cv2.MORPH_OPEN, kernel)
        asphalt_mask = cv2.morphologyEx(asphalt_mask, cv2.MORPH_CLOSE, kernel)

        # 윤곽선 찾기
        contours, _ = cv2.findContours(asphalt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        parking_spots = []
        spot_id = 1

        for contour in contours:
            area = cv2.contourArea(contour)

            # 주차 공간 크기 필터링
            if 2000 < area < 50000:

                # 윤곽선 근사
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # 바운딩 박스
                x, y, w, h = cv2.boundingRect(contour)

                # 종횡비 검사
                aspect_ratio = w / h if h > 0 else 0
                if 0.5 < aspect_ratio < 4.0:

                    # 컨벡스성 검사
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0

                    if solidity > 0.6:  # 충분히 단순한 형태
                        center = (x + w // 2, y + h // 2)

                        # 색상 특징 추출
                        roi = image[y:y+h, x:x+w]
                        if roi.size > 0:
                            color_features = self.extract_color_features(roi)

                            spot = ParkingSpot(
                                id=spot_id,
                                bbox=(x, y, x + w, y + h),
                                center=center,
                                area=area,
                                status=ParkingSpotStatus.UNKNOWN,
                                confidence=0.7,
                                color_features=color_features
                            )
                            parking_spots.append(spot)
                            spot_id += 1

        return parking_spots

    def extract_color_features(self, roi: np.ndarray) -> Dict:
        """주차 구역의 색상 특징 추출"""
        if roi.size == 0:
            return {}

        # HSV 변환
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 색상 통계
        mean_color = np.mean(roi, axis=(0, 1))
        std_color = np.std(roi, axis=(0, 1))

        # HSV 통계
        mean_hsv = np.mean(hsv_roi, axis=(0, 1))

        # 밝기 분포
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        brightness_hist, _ = np.histogram(gray_roi, bins=32, range=(0, 256))

        # 엣지 밀도
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        return {
            'mean_bgr': mean_color.tolist(),
            'std_bgr': std_color.tolist(),
            'mean_hsv': mean_hsv.tolist(),
            'brightness_hist': brightness_hist.tolist(),
            'edge_density': float(edge_density),
            'dominant_brightness': float(np.argmax(brightness_hist) * 8)  # 0-255 범위로 변환
        }

    def smart_grid_generation(self, image_shape: Tuple[int, int],
                            h_lines: List, v_lines: List,
                            color_regions: List[ParkingSpot]) -> List[ParkingSpot]:
        """지능형 격자 생성 (선과 색상 정보 결합)"""

        height, width = image_shape[:2]

        # 1. 선 기반 격자가 가능한 경우
        if len(h_lines) >= 2 and len(v_lines) >= 2:
            return self.generate_line_based_grid(h_lines, v_lines)

        # 2. 색상 기반 구역이 충분한 경우
        elif len(color_regions) > 5:
            return self.refine_color_regions(color_regions, image_shape)

        # 3. 하이브리드 접근
        elif len(h_lines) >= 2 or len(v_lines) >= 1:
            return self.generate_hybrid_grid(image_shape, h_lines, v_lines, color_regions)

        # 4. 기본 적응형 격자
        else:
            return self.generate_adaptive_grid(image_shape, color_regions)

    def generate_line_based_grid(self, h_lines: List, v_lines: List) -> List[ParkingSpot]:
        """선 기반 격자 생성"""
        parking_spots = []
        spot_id = 1

        # 선들을 정렬
        h_sorted = sorted(h_lines, key=lambda x: (x[1] + x[3]) // 2)  # y 평균으로 정렬
        v_sorted = sorted(v_lines, key=lambda x: (x[0] + x[2]) // 2)  # x 평균으로 정렬

        for i in range(len(h_sorted) - 1):
            for j in range(len(v_sorted) - 1):
                # 교차점으로 사각형 생성
                h1, h2 = h_sorted[i], h_sorted[i + 1]
                v1, v2 = v_sorted[j], v_sorted[j + 1]

                x1 = min(v1[0], v1[2])
                x2 = max(v2[0], v2[2])
                y1 = min(h1[1], h1[3])
                y2 = max(h2[1], h2[3])

                # 크기 검증
                w, h = x2 - x1, y2 - y1
                if 30 < w < 400 and 40 < h < 500:
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    area = w * h

                    spot = ParkingSpot(
                        id=spot_id,
                        bbox=(x1, y1, x2, y2),
                        center=center,
                        area=area,
                        status=ParkingSpotStatus.UNKNOWN,
                        confidence=0.9
                    )
                    parking_spots.append(spot)
                    spot_id += 1

        return parking_spots

    def generate_adaptive_grid(self, image_shape: Tuple[int, int],
                             color_regions: List[ParkingSpot]) -> List[ParkingSpot]:
        """적응형 격자 생성"""
        height, width = image_shape[:2]

        # 색상 구역이 있으면 그것을 기반으로 격자 크기 추정
        if color_regions:
            avg_width = np.mean([r.bbox[2] - r.bbox[0] for r in color_regions])
            avg_height = np.mean([r.bbox[3] - r.bbox[1] for r in color_regions])

            cols = max(3, int(width * 0.8 / avg_width))
            rows = max(2, int(height * 0.8 / avg_height))
        else:
            # 기본값
            cols = 6
            rows = 3

        # 여백 설정
        margin_x = width // 10
        margin_y = height // 10

        effective_width = width - 2 * margin_x
        effective_height = height - 2 * margin_y

        spot_width = effective_width // cols
        spot_height = effective_height // rows

        parking_spots = []
        spot_id = 1

        for row in range(rows):
            for col in range(cols):
                x1 = margin_x + col * spot_width
                y1 = margin_y + row * spot_height
                x2 = x1 + spot_width - 15
                y2 = y1 + spot_height - 15

                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                area = (x2 - x1) * (y2 - y1)

                spot = ParkingSpot(
                    id=spot_id,
                    bbox=(x1, y1, x2, y2),
                    center=center,
                    area=area,
                    status=ParkingSpotStatus.UNKNOWN,
                    confidence=0.6
                )
                parking_spots.append(spot)
                spot_id += 1

        return parking_spots

    def analyze_occupancy_advanced(self, image: np.ndarray, parking_spots: List[ParkingSpot],
                                 vehicles: List) -> List[ParkingSpot]:
        """고급 점유 상태 분석"""

        # 1. 차량 기반 점유 판단
        for vehicle in vehicles:
            v_bbox = vehicle['bbox']
            v_center = ((v_bbox[0] + v_bbox[2]) // 2, (v_bbox[1] + v_bbox[3]) // 2)

            # 겹치는 주차 구역 찾기
            for spot in parking_spots:
                if self.check_overlap(v_bbox, spot.bbox):
                    spot.status = ParkingSpotStatus.OCCUPIED
                    spot.confidence = min(1.0, spot.confidence + vehicle['confidence'] * 0.4)

        # 2. 색상 기반 점유 판단
        for spot in parking_spots:
            if spot.status != ParkingSpotStatus.OCCUPIED:
                occupancy_score = self.analyze_spot_occupancy_by_color(image, spot)

                if occupancy_score > 0.7:
                    spot.status = ParkingSpotStatus.OCCUPIED
                    spot.confidence = min(1.0, spot.confidence + occupancy_score * 0.3)
                elif occupancy_score < 0.3:
                    spot.status = ParkingSpotStatus.EMPTY
                else:
                    spot.status = ParkingSpotStatus.UNKNOWN

        return parking_spots

    def check_overlap(self, bbox1: Tuple, bbox2: Tuple, threshold: float = 0.3) -> bool:
        """두 박스의 겹침 정도 확인"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # 교집합 계산
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x1_inter >= x2_inter or y1_inter >= y2_inter:
            return False

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # 더 작은 박스 대비 겹침 비율
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        smaller_area = min(area1, area2)

        overlap_ratio = inter_area / smaller_area if smaller_area > 0 else 0

        return overlap_ratio > threshold

    def analyze_spot_occupancy_by_color(self, image: np.ndarray, spot: ParkingSpot) -> float:
        """색상 분석을 통한 점유 점수 계산"""
        x1, y1, x2, y2 = spot.bbox
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return 0.5

        # 현재 색상 특징 추출
        current_features = self.extract_color_features(roi)

        # 점유 가능성 점수 계산
        score = 0.0

        # 1. 밝기 분산 (차량이 있으면 더 다양한 밝기)
        if 'brightness_hist' in current_features:
            hist = np.array(current_features['brightness_hist'])
            brightness_variance = np.var(hist)
            if brightness_variance > 1000:  # 임계값은 실험적으로 조정
                score += 0.3

        # 2. 엣지 밀도 (차량이 있으면 더 많은 엣지)
        if current_features.get('edge_density', 0) > 0.1:
            score += 0.4

        # 3. 색상 복잡성 (빈 아스팔트는 단순한 색상)
        if 'std_bgr' in current_features:
            color_complexity = np.mean(current_features['std_bgr'])
            if color_complexity > 20:
                score += 0.3

        return min(1.0, score)

    def process_parking_lot(self, image_path: str) -> Tuple[np.ndarray, List[ParkingSpot], Dict]:
        """메인 처리 파이프라인"""
        print(f"🚗 실제 주차장 분석 시작: {image_path}")

        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지 로드 실패: {image_path}")

        print(f"📐 이미지 크기: {image.shape}")

        # 1. 고급 전처리
        processed = self.advanced_preprocessing(image)

        # 2. 차량 감지
        vehicles = []
        if self.yolo_model:
            try:
                results = self.yolo_model(image, verbose=False, conf=0.2)
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                            if int(cls) in [2, 3, 5, 7]:  # 차량 클래스
                                x1, y1, x2, y2 = [int(x) for x in box.cpu().numpy()]
                                vehicles.append({
                                    'bbox': (x1, y1, x2, y2),
                                    'confidence': float(conf),
                                    'class': int(cls)
                                })
            except Exception as e:
                print(f"⚠️ YOLO 감지 오류: {e}")

        print(f"🚗 감지된 차량: {len(vehicles)}대")

        # 3. 고급 주차선 감지
        h_lines, v_lines = self.detect_parking_lines_advanced(processed)

        # 4. 색상 기반 구역 감지
        color_regions = self.detect_parking_regions_by_color(image, processed)
        print(f"🎨 색상 기반 구역: {len(color_regions)}개")

        # 5. 지능형 격자 생성
        parking_spots = self.smart_grid_generation(image.shape, h_lines, v_lines, color_regions)
        print(f"🅿️ 최종 주차 구역: {len(parking_spots)}개")

        # 6. 점유 상태 분석
        parking_spots = self.analyze_occupancy_advanced(image, parking_spots, vehicles)

        # 통계 계산
        empty_count = sum(1 for spot in parking_spots if spot.status == ParkingSpotStatus.EMPTY)
        occupied_count = sum(1 for spot in parking_spots if spot.status == ParkingSpotStatus.OCCUPIED)
        unknown_count = sum(1 for spot in parking_spots if spot.status == ParkingSpotStatus.UNKNOWN)

        stats = {
            'total_spots': len(parking_spots),
            'empty_spots': empty_count,
            'occupied_spots': occupied_count,
            'unknown_spots': unknown_count,
            'vehicles_detected': len(vehicles),
            'horizontal_lines': len(h_lines),
            'vertical_lines': len(v_lines),
            'color_regions': len(color_regions),
            'occupancy_rate': occupied_count / len(parking_spots) * 100 if parking_spots else 0,
            'confidence_avg': np.mean([spot.confidence for spot in parking_spots]) if parking_spots else 0
        }

        print(f"✅ 고급 분석 완료")
        print(f"   📊 점유: {occupied_count}개, 빈자리: {empty_count}개, 불명: {unknown_count}개")
        print(f"   📈 점유율: {stats['occupancy_rate']:.1f}%, 평균 신뢰도: {stats['confidence_avg']:.1%}")

        return image, parking_spots, stats

    def draw_results(self, image: np.ndarray, parking_spots: List[ParkingSpot],
                    vehicles: List = None) -> np.ndarray:
        """결과 시각화"""
        result = image.copy()

        # 주차 구역 그리기
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

            # 박스 그리기 (신뢰도에 따른 두께)
            thickness = max(1, int(spot.confidence * 3))
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

            # 텍스트
            font_scale = 0.4
            cv2.putText(result, f"P{spot.id}", (x1+3, y1+15),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
            cv2.putText(result, text, (x1+3, y1+30),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
            cv2.putText(result, f"{spot.confidence:.2f}", (x1+3, y1+45),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

        # 차량 그리기
        if vehicles:
            for vehicle in vehicles:
                x1, y1, x2, y2 = vehicle['bbox']
                cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(result, f"Vehicle {vehicle['confidence']:.2f}",
                           (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        return result

    def save_results(self, image: np.ndarray, parking_spots: List[ParkingSpot],
                    stats: Dict, output_dir: str = "realworld_results"):
        """결과 저장"""
        os.makedirs(output_dir, exist_ok=True)

        # 결과 이미지
        result_image = self.draw_results(image, parking_spots)
        cv2.imwrite(f"{output_dir}/realworld_parking_result.jpg", result_image)

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
                    'color_features': spot.color_features
                }
                for spot in parking_spots
            ]
        }

        with open(f"{output_dir}/realworld_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"💾 실제 주차장 분석 결과 저장: {output_dir}/")


def main():
    """메인 실행"""
    print("🚗 실제 주차장 특화 고정밀 분석 시스템")
    print("=" * 50)

    # 모델 경로
    yolo_path = "../../yolov8n-obb.pt"

    # 감지기 초기화
    detector = RealWorldParkingDetector(yolo_path)

    # 이미지 처리
    image_path = "parkinglot1.jpg"

    if not os.path.exists(image_path):
        print(f"❌ 이미지 없음: {image_path}")
        return

    try:
        # 분석 실행
        image, spots, stats = detector.process_parking_lot(image_path)

        # 차량 재감지 (시각화용)
        vehicles = []
        if detector.yolo_model:
            try:
                results = detector.yolo_model(image, verbose=False, conf=0.2)
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                            if int(cls) in [2, 3, 5, 7]:
                                x1, y1, x2, y2 = [int(x) for x in box.cpu().numpy()]
                                vehicles.append({
                                    'bbox': (x1, y1, x2, y2),
                                    'confidence': float(conf),
                                    'class': int(cls)
                                })
            except:
                pass

        # 결과 이미지 생성
        result_image = detector.draw_results(image, spots, vehicles)

        # 결과 저장
        detector.save_results(image, spots, stats)

        # 결과 출력
        print("\n" + "=" * 50)
        print("📊 실제 주차장 분석 최종 결과")
        print("=" * 50)
        print(f"🅿️  총 주차구역: {stats['total_spots']}개")
        print(f"🟢 빈 자리: {stats['empty_spots']}개")
        print(f"🔴 점유된 자리: {stats['occupied_spots']}개")
        print(f"🟡 불명확: {stats['unknown_spots']}개")
        print(f"🚗 감지된 차량: {stats['vehicles_detected']}대")
        print(f"📈 점유율: {stats['occupancy_rate']:.1f}%")
        print(f"🎯 평균 신뢰도: {stats['confidence_avg']:.1%}")

        # 고급 시각화
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 원본 이미지
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("원본 이미지")
        axes[0, 0].axis('off')

        # 결과 이미지
        axes[0, 1].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f"고정밀 분석 결과 ({stats['total_spots']}개)")
        axes[0, 1].axis('off')

        # 전처리 결과 (예시)
        processed = detector.advanced_preprocessing(image)
        axes[0, 2].imshow(processed['line_mask'], cmap='gray')
        axes[0, 2].set_title("주차선 감지")
        axes[0, 2].axis('off')

        # 통계 차트들
        # 주차 상태 파이 차트
        labels = ['빈 자리', '점유', '불명확']
        sizes = [stats['empty_spots'], stats['occupied_spots'], stats['unknown_spots']]
        colors = ['lightgreen', 'lightcoral', 'lightyellow']

        axes[1, 0].pie([s for s in sizes if s > 0],
                      labels=[l for l, s in zip(labels, sizes) if s > 0],
                      colors=[c for c, s in zip(colors, sizes) if s > 0],
                      autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title("주차 상태 분포")

        # 감지 방법별 결과
        detection_methods = ['수평선', '수직선', '색상구역', '차량']
        detection_counts = [stats['horizontal_lines'], stats['vertical_lines'],
                          stats['color_regions'], stats['vehicles_detected']]

        axes[1, 1].bar(detection_methods, detection_counts,
                      color=['skyblue', 'lightpink', 'lightsteelblue', 'gold'])
        axes[1, 1].set_title("감지 요소별 결과")
        axes[1, 1].set_ylabel("개수")

        # 성능 지표
        performance_labels = ['총 구역', '감지율', '신뢰도', '점유율']
        performance_values = [stats['total_spots'],
                            (stats['empty_spots'] + stats['occupied_spots']) / stats['total_spots'] * 100,
                            stats['confidence_avg'] * 100,
                            stats['occupancy_rate']]

        axes[1, 2].bar(performance_labels, performance_values,
                      color=['purple', 'orange', 'green', 'red'])
        axes[1, 2].set_title("성능 지표")
        axes[1, 2].set_ylabel("비율 (%)")

        plt.tight_layout()
        plt.savefig("realworld_parking_analysis.png", dpi=150, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()