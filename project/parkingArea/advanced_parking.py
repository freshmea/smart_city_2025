"""
YOLO-OBB 기반 정밀 주차장 분석 시스템
더 정확한 주차 영역 감지를 위한 고급 컴퓨터 비전 기법 적용
"""

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO


class ParkingSpotStatus(Enum):
    """주차 상태 열거형"""
    EMPTY = "empty"
    OCCUPIED = "occupied"
    UNKNOWN = "unknown"

@dataclass
class ParkingSpot:
    """주차 구역 정보"""
    id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]
    area: float
    status: ParkingSpotStatus
    confidence: float
    corners: Optional[List[Tuple[int, int]]] = None  # OBB corners

class AdvancedParkingDetector:
    """YOLO-OBB 기반 고급 주차장 감지 시스템"""

    def __init__(self, yolo_obb_path: str = None):
        """
        초기화
        Args:
            yolo_obb_path: YOLO-OBB 모델 경로
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 디바이스: {self.device}")

        # YOLO-OBB 모델 로드
        if yolo_obb_path and os.path.exists(yolo_obb_path):
            self.yolo_model = YOLO(yolo_obb_path)
            print(f"✅ YOLO-OBB 모델 로드: {yolo_obb_path}")
        else:
            # 기본 모델 사용
            try:
                self.yolo_model = YOLO('yolov8n-obb.pt')
                print("✅ 기본 YOLO-OBB 모델 로드")
            except:
                print("⚠️ YOLO-OBB 모델 로드 실패, 전통적 방법 사용")
                self.yolo_model = None

        self.parking_spots = []
        self.parking_template = None

    def preprocess_image_advanced(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """고급 이미지 전처리"""
        results = {}

        # 원본
        results['original'] = image.copy()

        # 그레이스케일
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results['grayscale'] = gray

        # 적응형 히스토그램 평활화
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        results['enhanced'] = enhanced

        # 가우시안 블러
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        results['blurred'] = blurred

        # 모폴로지 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
        results['morphology'] = morph

        # 적응형 임계값
        adaptive_thresh = cv2.adaptiveThreshold(
            morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        results['adaptive_threshold'] = adaptive_thresh

        # Canny 엣지 (여러 매개변수)
        canny1 = cv2.Canny(morph, 50, 150)
        canny2 = cv2.Canny(morph, 100, 200)
        canny_combined = cv2.bitwise_or(canny1, canny2)
        results['canny'] = canny_combined

        return results

    def detect_with_yolo_obb(self, image: np.ndarray) -> List[ParkingSpot]:
        """YOLO-OBB를 사용한 주차 영역 감지"""
        if self.yolo_model is None:
            return []

        try:
            # YOLO 추론 실행
            results = self.yolo_model(image, verbose=False)
            parking_spots = []

            for i, result in enumerate(results):
                if hasattr(result, 'obb') and result.obb is not None:
                    # OBB (Oriented Bounding Box) 결과 처리
                    for j, (obb, conf, cls) in enumerate(zip(result.obb.xyxyxyxy, result.obb.conf, result.obb.cls)):
                        if conf > 0.3:  # 신뢰도 임계값
                            # OBB 좌표를 일반 bbox로 변환
                            corners = obb.cpu().numpy().reshape(-1, 2)
                            x_coords = corners[:, 0]
                            y_coords = corners[:, 1]

                            x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
                            x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))

                            center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            area = (x2 - x1) * (y2 - y1)

                            spot = ParkingSpot(
                                id=len(parking_spots) + 1,
                                bbox=(x1, y1, x2, y2),
                                center=center,
                                area=area,
                                status=ParkingSpotStatus.UNKNOWN,
                                confidence=float(conf),
                                corners=[(int(x), int(y)) for x, y in corners]
                            )
                            parking_spots.append(spot)

                elif hasattr(result, 'boxes') and result.boxes is not None:
                    # 일반 bounding box 처리
                    for j, (box, conf, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)):
                        if conf > 0.3:
                            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                            center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            area = (x2 - x1) * (y2 - y1)

                            spot = ParkingSpot(
                                id=len(parking_spots) + 1,
                                bbox=(x1, y1, x2, y2),
                                center=center,
                                area=area,
                                status=ParkingSpotStatus.UNKNOWN,
                                confidence=float(conf)
                            )
                            parking_spots.append(spot)

            return parking_spots

        except Exception as e:
            print(f"⚠️ YOLO-OBB 감지 오류: {e}")
            return []

    def detect_with_contour_analysis(self, processed_images: Dict[str, np.ndarray]) -> List[ParkingSpot]:
        """윤곽선 분석을 통한 주차 영역 감지"""
        # 여러 전처리 결과를 조합
        combined = cv2.bitwise_or(processed_images['canny'], processed_images['adaptive_threshold'])

        # 형태학적 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        # 윤곽선 찾기
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        parking_spots = []
        min_area = 2000   # 최소 면적 증가
        max_area = 40000  # 최대 면적 조정

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            if min_area < area < max_area:
                # 윤곽선을 사각형으로 근사
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # 바운딩 박스
                x, y, w, h = cv2.boundingRect(contour)

                # 종횡비 검사 (주차 공간의 일반적인 비율)
                aspect_ratio = w / h if h > 0 else 0
                if 0.7 < aspect_ratio < 2.5:  # 더 엄격한 비율 조건
                    # 컨벡스 헐 검사 (너무 복잡한 형태 제외)
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0

                    if solidity > 0.7:  # 충분히 단순한 형태만
                        center = (x + w // 2, y + h // 2)

                        spot = ParkingSpot(
                            id=len(parking_spots) + 1,
                            bbox=(x, y, x + w, y + h),
                            center=center,
                            area=area,
                            status=ParkingSpotStatus.UNKNOWN,
                            confidence=0.8,
                            corners=[(int(pt[0][0]), int(pt[0][1])) for pt in approx] if len(approx) <= 8 else None
                        )
                        parking_spots.append(spot)

        return parking_spots

    def detect_with_template_matching(self, image: np.ndarray, processed_images: Dict[str, np.ndarray]) -> List[ParkingSpot]:
        """템플릿 매칭을 통한 주차 영역 감지 (개선된 버전)"""
        gray = processed_images['enhanced']

        parking_spots = []

        # 더 현실적인 주차 공간 크기들 (실제 주차장 비율 고려)
        template_sizes = [(90, 180), (100, 200), (80, 160)]

        for template_w, template_h in template_sizes:
            # 더 정교한 템플릿 생성
            template = np.ones((template_h, template_w), dtype=np.uint8) * 200

            # 주차선 패턴 추가
            cv2.rectangle(template, (0, 0), (template_w, 5), 100, -1)  # 상단 선
            cv2.rectangle(template, (0, template_h-5), (template_w, template_h), 100, -1)  # 하단 선
            cv2.rectangle(template, (0, 0), (5, template_h), 100, -1)  # 좌측 선
            cv2.rectangle(template, (template_w-5, 0), (template_w, template_h), 100, -1)  # 우측 선

            template = cv2.GaussianBlur(template, (3, 3), 0)

            # 템플릿 매칭 (더 엄격한 임계값)
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.75  # 더 높은 임계값

            locations = np.where(result >= threshold)

            for pt in zip(*locations[::-1]):
                x, y = pt

                # 중복 제거를 위한 거리 검사
                is_duplicate = False
                for existing_spot in parking_spots:
                    ex, ey = existing_spot.center
                    distance = np.sqrt((x + template_w//2 - ex)**2 + (y + template_h//2 - ey)**2)
                    if distance < min(template_w, template_h) * 0.6:  # 더 엄격한 중복 검사
                        is_duplicate = True
                        break

                if not is_duplicate:
                    center = (x + template_w // 2, y + template_h // 2)
                    confidence = float(result[y, x])

                    spot = ParkingSpot(
                        id=len(parking_spots) + 1,
                        bbox=(x, y, x + template_w, y + template_h),
                        center=center,
                        area=template_w * template_h,
                        status=ParkingSpotStatus.UNKNOWN,
                        confidence=confidence
                    )
                    parking_spots.append(spot)

                    # 너무 많은 결과 방지
                    if len(parking_spots) > 100:
                        return parking_spots

        return parking_spots    def merge_detection_results(self, *detection_results: List[List[ParkingSpot]]) -> List[ParkingSpot]:
        """여러 감지 결과를 병합"""
        all_spots = []
        for spots in detection_results:
            all_spots.extend(spots)

        if not all_spots:
            return []

        # 거리 기반 중복 제거 및 병합
        merged_spots = []
        merge_threshold = 50  # 병합 거리 임계값

        for spot in all_spots:
            merged = False

            for existing_spot in merged_spots:
                # 중심점 간 거리 계산
                distance = np.sqrt(
                    (spot.center[0] - existing_spot.center[0])**2 +
                    (spot.center[1] - existing_spot.center[1])**2
                )

                if distance < merge_threshold:
                    # 더 높은 신뢰도의 결과로 업데이트
                    if spot.confidence > existing_spot.confidence:
                        existing_spot.bbox = spot.bbox
                        existing_spot.center = spot.center
                        existing_spot.area = spot.area
                        existing_spot.confidence = spot.confidence
                        if spot.corners:
                            existing_spot.corners = spot.corners
                    merged = True
                    break

            if not merged:
                spot.id = len(merged_spots) + 1
                merged_spots.append(spot)

        return merged_spots

    def analyze_parking_occupancy(self, image: np.ndarray, parking_spots: List[ParkingSpot]) -> List[ParkingSpot]:
        """주차 상태 분석 (점유/빈자리)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for spot in parking_spots:
            x1, y1, x2, y2 = spot.bbox
            roi = gray[y1:y2, x1:x2]

            if roi.size > 0:
                # 통계적 분석
                mean_intensity = np.mean(roi)
                std_intensity = np.std(roi)

                # 엣지 밀도 분석
                edges = cv2.Canny(roi, 50, 150)
                edge_density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])

                # 색상 히스토그램 분석 (컬러 이미지에서)
                roi_color = image[y1:y2, x1:x2]
                hist = cv2.calcHist([roi_color], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist_complexity = np.std(hist)

                # 점유 상태 판단 (휴리스틱)
                if edge_density > 0.1 and hist_complexity > 100:
                    spot.status = ParkingSpotStatus.OCCUPIED
                    spot.confidence = min(spot.confidence + 0.2, 1.0)
                elif mean_intensity > 120 and std_intensity < 30:
                    spot.status = ParkingSpotStatus.EMPTY
                    spot.confidence = min(spot.confidence + 0.1, 1.0)
                else:
                    spot.status = ParkingSpotStatus.UNKNOWN

        return parking_spots

    def process_parking_lot(self, image_path: str) -> Tuple[np.ndarray, List[ParkingSpot], Dict]:
        """전체 주차장 분석 파이프라인"""
        print(f"🚗 주차장 이미지 분석 시작: {image_path}")

        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

        print(f"📐 이미지 크기: {image.shape}")

        # 고급 전처리
        processed_images = self.preprocess_image_advanced(image)

        # 다중 감지 방법 적용
        detection_results = []

        # 1. YOLO-OBB 감지
        if self.yolo_model:
            print("🔍 YOLO-OBB 감지 실행...")
            yolo_spots = self.detect_with_yolo_obb(image)
            detection_results.append(yolo_spots)
            print(f"   YOLO-OBB 결과: {len(yolo_spots)}개")

        # 2. 윤곽선 분석
        print("🔍 윤곽선 분석 실행...")
        contour_spots = self.detect_with_contour_analysis(processed_images)
        detection_results.append(contour_spots)
        print(f"   윤곽선 분석 결과: {len(contour_spots)}개")

        # 3. 템플릿 매칭
        print("🔍 템플릿 매칭 실행...")
        template_spots = self.detect_with_template_matching(image, processed_images)
        detection_results.append(template_spots)
        print(f"   템플릿 매칭 결과: {len(template_spots)}개")

        # 결과 병합
        print("🔄 감지 결과 병합...")
        merged_spots = self.merge_detection_results(*detection_results)
        print(f"   병합 후 결과: {len(merged_spots)}개")

        # 주차 상태 분석
        print("📊 주차 상태 분석...")
        final_spots = self.analyze_parking_occupancy(image, merged_spots)

        # 통계 정보
        stats = {
            'total_spots': len(final_spots),
            'empty_spots': sum(1 for spot in final_spots if spot.status == ParkingSpotStatus.EMPTY),
            'occupied_spots': sum(1 for spot in final_spots if spot.status == ParkingSpotStatus.OCCUPIED),
            'unknown_spots': sum(1 for spot in final_spots if spot.status == ParkingSpotStatus.UNKNOWN),
            'avg_confidence': np.mean([spot.confidence for spot in final_spots]) if final_spots else 0,
            'detection_methods': {
                'yolo_obb': len(yolo_spots) if 'yolo_spots' in locals() else 0,
                'contour': len(contour_spots),
                'template': len(template_spots)
            }
        }

        print(f"✅ 분석 완료: {stats['total_spots']}개 주차구역 감지")

        return image, final_spots, stats

    def draw_results(self, image: np.ndarray, parking_spots: List[ParkingSpot]) -> np.ndarray:
        """결과를 이미지에 그리기"""
        result = image.copy()

        for spot in parking_spots:
            # 상태에 따른 색상 설정
            if spot.status == ParkingSpotStatus.EMPTY:
                color = (0, 255, 0)  # 초록색
                status_text = "EMPTY"
            elif spot.status == ParkingSpotStatus.OCCUPIED:
                color = (0, 0, 255)  # 빨간색
                status_text = "OCCUPIED"
            else:
                color = (0, 255, 255)  # 노란색
                status_text = "UNKNOWN"

            # 바운딩 박스 그리기
            x1, y1, x2, y2 = spot.bbox
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # OBB 코너가 있으면 다각형 그리기
            if spot.corners and len(spot.corners) >= 3:
                pts = np.array(spot.corners, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(result, [pts], True, color, 2)

            # 중심점 표시
            cv2.circle(result, spot.center, 5, color, -1)

            # 텍스트 정보
            text = f"P{spot.id}: {status_text}"
            conf_text = f"Conf: {spot.confidence:.2f}"

            cv2.putText(result, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(result, conf_text, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return result

    def save_results(self, image: np.ndarray, parking_spots: List[ParkingSpot],
                    stats: Dict, output_dir: str = "advanced_results"):
        """결과 저장"""
        os.makedirs(output_dir, exist_ok=True)

        # 결과 이미지 저장
        result_image = self.draw_results(image, parking_spots)
        result_path = os.path.join(output_dir, "advanced_parking_result.jpg")
        cv2.imwrite(result_path, result_image)
        print(f"💾 결과 이미지 저장: {result_path}")

        # JSON 결과 저장 (NumPy 타입을 Python 네이티브 타입으로 변환)
        spots_data = []
        for spot in parking_spots:
            spot_dict = {
                'id': int(spot.id),
                'bbox': [int(x) for x in spot.bbox],
                'center': [int(x) for x in spot.center],
                'area': float(spot.area),
                'status': spot.status.value,
                'confidence': float(spot.confidence),
                'corners': [[int(x), int(y)] for x, y in spot.corners] if spot.corners else None
            }
            spots_data.append(spot_dict)

        json_data = {
            'statistics': stats,
            'parking_spots': spots_data,
            'analysis_info': {
                'timestamp': str(pd.Timestamp.now()),
                'total_spots': len(parking_spots),
                'device_used': str(self.device)
            }
        }

        json_path = os.path.join(output_dir, "advanced_parking_analysis.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"💾 분석 데이터 저장: {json_path}")

        # 텍스트 리포트 저장
        report_path = os.path.join(output_dir, "parking_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 고급 주차장 분석 리포트 ===\n")
            f.write(f"총 주차구역: {stats['total_spots']}개\n")
            f.write(f"빈 자리: {stats['empty_spots']}개\n")
            f.write(f"점유된 자리: {stats['occupied_spots']}개\n")
            f.write(f"불명확한 자리: {stats['unknown_spots']}개\n")
            f.write(f"평균 신뢰도: {stats['avg_confidence']:.3f}\n\n")

            f.write("감지 방법별 결과:\n")
            for method, count in stats['detection_methods'].items():
                f.write(f"  {method}: {count}개\n")

            f.write("\n=== 상세 주차구역 정보 ===\n")
            for spot in parking_spots:
                f.write(f"구역 {spot.id}: {spot.status.value.upper()} "
                       f"(신뢰도: {spot.confidence:.3f}, 면적: {spot.area:.0f})\n")

        print(f"💾 분석 리포트 저장: {report_path}")


def main():
    """메인 실행 함수"""
    print("🚗 고급 주차장 분석 시스템 시작")
    print("=" * 50)

    # YOLO-OBB 모델 경로 설정
    yolo_obb_path = "../../yolov8n-obb.pt"  # 프로젝트 루트의 모델 파일

    # 감지기 초기화
    detector = AdvancedParkingDetector(yolo_obb_path)

    # 이미지 경로
    image_path = "parkinglot1.jpg"

    if not os.path.exists(image_path):
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
        return

    try:
        # 주차장 분석 실행
        original_image, parking_spots, stats = detector.process_parking_lot(image_path)

        # 결과 저장
        detector.save_results(original_image, parking_spots, stats)

        # 시각화
        result_image = detector.draw_results(original_image, parking_spots)

        # 결과 출력
        print("\n" + "=" * 50)
        print("📊 분석 결과 요약")
        print("=" * 50)
        print(f"🅿️  총 주차구역: {stats['total_spots']}개")
        print(f"🟢 빈 자리: {stats['empty_spots']}개")
        print(f"🔴 점유된 자리: {stats['occupied_spots']}개")
        print(f"🟡 불명확: {stats['unknown_spots']}개")
        print(f"📈 평균 신뢰도: {stats['avg_confidence']:.1%}")

        # matplotlib으로 시각화
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 원본 이미지
        axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("원본 이미지")
        axes[0, 0].axis('off')

        # 결과 이미지
        axes[0, 1].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f"감지 결과 ({stats['total_spots']}개 구역)")
        axes[0, 1].axis('off')

        # 통계 차트
        labels = ['빈 자리', '점유', '불명확']
        sizes = [stats['empty_spots'], stats['occupied_spots'], stats['unknown_spots']]
        colors = ['lightgreen', 'lightcoral', 'lightyellow']

        axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title("주차 상태 분포")

        # 감지 방법별 결과
        methods = list(stats['detection_methods'].keys())
        counts = list(stats['detection_methods'].values())

        axes[1, 1].bar(methods, counts, color=['skyblue', 'lightpink', 'lightsteelblue'])
        axes[1, 1].set_title("감지 방법별 결과")
        axes[1, 1].set_ylabel("감지된 구역 수")

        plt.tight_layout()
        plt.savefig("advanced_parking_analysis.png", dpi=150, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # pandas import 추가
    try:
        import pandas as pd
    except ImportError:
        # pandas가 없으면 timestamp를 다른 방식으로 처리
        from datetime import datetime
        class pd:
            class Timestamp:
                @staticmethod
                def now():
                    return datetime.now().isoformat()

    main()