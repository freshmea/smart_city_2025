import os
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


class ParkingLotDetector:
    """주차장 탑뷰 이미지에서 주차 라인과 주차 영역을 감지하는 클래스"""

    def __init__(self):
        self.parking_spots = []
        self.parking_lines = []

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 가우시안 블러 적용
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 대비 향상
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        return enhanced

    def detect_lines(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Hough 변환을 사용하여 직선 감지"""
        # Canny 엣지 검출
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # 형태학적 연산으로 선 강화
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Hough 직선 변환
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )

        if lines is not None:
            return lines.reshape(-1, 4)
        return []

    def filter_lines(self, lines: List[Tuple[int, int, int, int]],
                    image_shape: Tuple[int, int]) -> Tuple[List, List]:
        """수직선과 수평선 분리 및 필터링"""
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line

            # 선의 각도 계산
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # 최소 길이 필터링
            if length < 20:
                continue

            # 각도에 따른 분류 (허용 오차 ±15도)
            if abs(angle) <= 15 or abs(angle) >= 165:  # 수평선
                horizontal_lines.append(line)
            elif 75 <= abs(angle) <= 105:  # 수직선
                vertical_lines.append(line)

        return horizontal_lines, vertical_lines

    def merge_similar_lines(self, lines: List, threshold: int = 20) -> List:
        """유사한 위치의 선들을 병합"""
        if not lines:
            return []

        merged_lines = []
        used = [False] * len(lines)

        for i, line1 in enumerate(lines):
            if used[i]:
                continue

            x1, y1, x2, y2 = line1
            similar_lines = [line1]
            used[i] = True

            for j, line2 in enumerate(lines[i+1:], i+1):
                if used[j]:
                    continue

                x3, y3, x4, y4 = line2

                # 수평선의 경우 y 좌표로 비교
                if abs(y1 - y2) < abs(x1 - x2):  # 수평선
                    if abs(y1 - y3) < threshold and abs(y2 - y4) < threshold:
                        similar_lines.append(line2)
                        used[j] = True
                # 수직선의 경우 x 좌표로 비교
                else:
                    if abs(x1 - x3) < threshold and abs(x2 - x4) < threshold:
                        similar_lines.append(line2)
                        used[j] = True

            # 유사한 선들의 평균으로 병합
            if similar_lines:
                avg_line = np.mean(similar_lines, axis=0).astype(int)
                merged_lines.append(tuple(avg_line))

        return merged_lines

    def detect_parking_spots(self, horizontal_lines: List, vertical_lines: List,
                           image_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """주차 영역 감지"""
        parking_spots = []

        # 선들을 정렬
        horizontal_lines = sorted(horizontal_lines, key=lambda x: x[1])
        vertical_lines = sorted(vertical_lines, key=lambda x: x[0])

        # 인접한 수평선과 수직선으로 직사각형 영역 생성
        for i in range(len(horizontal_lines) - 1):
            for j in range(len(vertical_lines) - 1):
                h_line1 = horizontal_lines[i]
                h_line2 = horizontal_lines[i + 1]
                v_line1 = vertical_lines[j]
                v_line2 = vertical_lines[j + 1]

                # 직사각형의 좌표 계산
                x1 = min(v_line1[0], v_line1[2])
                x2 = max(v_line2[0], v_line2[2])
                y1 = min(h_line1[1], h_line1[3])
                y2 = max(h_line2[1], h_line2[3])

                # 유효한 크기인지 확인
                width = x2 - x1
                height = y2 - y1

                if 30 < width < 200 and 50 < height < 300:
                    parking_spots.append((x1, y1, x2, y2))

        return parking_spots

    def draw_results(self, image: np.ndarray, horizontal_lines: List,
                    vertical_lines: List, parking_spots: List) -> np.ndarray:
        """결과를 이미지에 그리기"""
        result = image.copy()

        # 수평선 그리기 (파란색)
        for line in horizontal_lines:
            x1, y1, x2, y2 = line
            cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 수직선 그리기 (초록색)
        for line in vertical_lines:
            x1, y1, x2, y2 = line
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 주차 영역 그리기 (빨간색)
        for i, (x1, y1, x2, y2) in enumerate(parking_spots):
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # 주차 영역 번호 표시
            cv2.putText(result, f'P{i+1}', (x1+5, y1+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return result

    def process_image(self, image_path: str) -> Tuple[np.ndarray, dict]:
        """전체 이미지 처리 파이프라인"""
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

        print(f"이미지 크기: {image.shape}")

        # 전처리
        processed = self.preprocess_image(image)

        # 직선 감지
        lines = self.detect_lines(processed)
        print(f"감지된 직선 수: {len(lines)}")

        # 수평/수직선 분리
        horizontal_lines, vertical_lines = self.filter_lines(lines, image.shape[:2])
        print(f"수평선: {len(horizontal_lines)}, 수직선: {len(vertical_lines)}")

        # 유사한 선들 병합
        horizontal_lines = self.merge_similar_lines(horizontal_lines)
        vertical_lines = self.merge_similar_lines(vertical_lines)
        print(f"병합 후 - 수평선: {len(horizontal_lines)}, 수직선: {len(vertical_lines)}")

        # 주차 영역 감지
        parking_spots = self.detect_parking_spots(horizontal_lines, vertical_lines, image.shape[:2])
        print(f"감지된 주차 영역 수: {len(parking_spots)}")

        # 결과 이미지 생성
        result_image = self.draw_results(image, horizontal_lines, vertical_lines, parking_spots)

        # 결과 정보
        results = {
            'horizontal_lines': horizontal_lines,
            'vertical_lines': vertical_lines,
            'parking_spots': parking_spots,
            'total_spots': len(parking_spots)
        }

        return result_image, results

    def save_results(self, result_image: np.ndarray, results: dict, output_dir: str = '.'):
        """결과 저장"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 결과 이미지 저장
        output_path = os.path.join(output_dir, 'parking_detection_result.jpg')
        cv2.imwrite(output_path, result_image)
        print(f"결과 이미지 저장: {output_path}")

        # 텍스트 결과 저장
        result_text_path = os.path.join(output_dir, 'parking_detection_info.txt')
        with open(result_text_path, 'w', encoding='utf-8') as f:
            f.write("=== 주차장 감지 결과 ===\n")
            f.write(f"총 주차 영역 수: {results['total_spots']}\n")
            f.write(f"수평선 수: {len(results['horizontal_lines'])}\n")
            f.write(f"수직선 수: {len(results['vertical_lines'])}\n\n")

            f.write("주차 영역 좌표:\n")
            for i, (x1, y1, x2, y2) in enumerate(results['parking_spots']):
                f.write(f"주차구역 {i+1}: ({x1}, {y1}) - ({x2}, {y2})\n")

        print(f"분석 결과 저장: {result_text_path}")


def main():
    """메인 함수"""
    # 주차장 감지기 초기화
    detector = ParkingLotDetector()

    # 이미지 경로 설정
    image_path = 'parkinglot1.jpg'

    if not os.path.exists(image_path):
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        return

    try:
        # 이미지 처리
        print("주차장 이미지 분석 중...")
        result_image, results = detector.process_image(image_path)

        # 결과 저장
        detector.save_results(result_image, results)

        # 결과 출력
        print("\n=== 분석 완료 ===")
        print(f"총 {results['total_spots']}개의 주차 영역이 감지되었습니다.")

        # 결과 이미지 표시 (matplotlib 사용)
        plt.figure(figsize=(15, 10))

        # 원본 이미지
        plt.subplot(1, 2, 1)
        original = cv2.imread(image_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        plt.imshow(original_rgb)
        plt.title('원본 이미지')
        plt.axis('off')

        # 결과 이미지
        plt.subplot(1, 2, 2)
        result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        plt.imshow(result_rgb)
        plt.title(f'주차 영역 감지 결과 ({results["total_spots"]}개)')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('parking_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"오류 발생: {e}")


if __name__ == "__main__":
    main()
