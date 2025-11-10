"""
주차장 탑뷰 이미지 분석을 위한 간단한 대안 코드
OpenCV가 설치되지 않은 경우에도 사용 가능
"""

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class SimpleParkingDetector:
    """PIL과 numpy만을 사용한 간단한 주차장 감지기"""

    def __init__(self):
        self.parking_areas = []

    def load_image(self, image_path: str) -> np.ndarray:
        """이미지 로드"""
        img = Image.open(image_path)
        return np.array(img)

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """그레이스케일 변환"""
        if len(image.shape) == 3:
            # RGB to Grayscale
            return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        return image

    def simple_edge_detection(self, image: np.ndarray) -> np.ndarray:
        """간단한 엣지 검출 (Sobel 필터)"""
        # Sobel 커널
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # 패딩 추가
        padded = np.pad(image, 1, mode='edge')

        # 엣지 검출
        edges_x = np.zeros_like(image)
        edges_y = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+3, j:j+3]
                edges_x[i, j] = np.sum(region * sobel_x)
                edges_y[i, j] = np.sum(region * sobel_y)

        # 엣지 강도 계산
        edges = np.sqrt(edges_x**2 + edges_y**2)
        return (edges > np.percentile(edges, 85)).astype(np.uint8) * 255

    def detect_grid_pattern(self, image: np.ndarray,
                          min_spot_width: int = 30,
                          min_spot_height: int = 50) -> List[Tuple[int, int, int, int]]:
        """격자 패턴 기반 주차 구역 감지"""
        gray = self.to_grayscale(image)
        edges = self.simple_edge_detection(gray)

        # 수직 및 수평 투영
        h_projection = np.sum(edges, axis=1)
        v_projection = np.sum(edges, axis=0)

        # 피크 검출을 위한 임계값
        h_threshold = np.percentile(h_projection, 70)
        v_threshold = np.percentile(v_projection, 70)

        # 수평선 위치 찾기
        h_lines = []
        for i in range(1, len(h_projection) - 1):
            if h_projection[i] > h_threshold and \
               h_projection[i] > h_projection[i-1] and \
               h_projection[i] > h_projection[i+1]:
                h_lines.append(i)

        # 수직선 위치 찾기
        v_lines = []
        for i in range(1, len(v_projection) - 1):
            if v_projection[i] > v_threshold and \
               v_projection[i] > v_projection[i-1] and \
               v_projection[i] > v_projection[i+1]:
                v_lines.append(i)

        # 주차 구역 생성
        parking_spots = []
        for i in range(len(h_lines) - 1):
            for j in range(len(v_lines) - 1):
                x1, y1 = v_lines[j], h_lines[i]
                x2, y2 = v_lines[j+1], h_lines[i+1]

                width = x2 - x1
                height = y2 - y1

                # 크기 검증
                if width >= min_spot_width and height >= min_spot_height:
                    parking_spots.append((x1, y1, x2, y2))

        return parking_spots

    def manual_grid_detection(self, image_shape: Tuple[int, int, int],
                            rows: int = 3, cols: int = 5) -> List[Tuple[int, int, int, int]]:
        """수동 격자 생성 (참고용)"""
        height, width = image_shape[:2]

        # 여백 제외
        margin_x = width // 10
        margin_y = height // 10

        effective_width = width - 2 * margin_x
        effective_height = height - 2 * margin_y

        spot_width = effective_width // cols
        spot_height = effective_height // rows

        parking_spots = []
        for row in range(rows):
            for col in range(cols):
                x1 = margin_x + col * spot_width
                y1 = margin_y + row * spot_height
                x2 = x1 + spot_width - 5  # 여백
                y2 = y1 + spot_height - 5  # 여백

                parking_spots.append((x1, y1, x2, y2))

        return parking_spots

    def draw_results(self, image: np.ndarray,
                    parking_spots: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """결과를 이미지에 그리기"""
        # PIL 이미지로 변환
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        # 폰트 설정 (기본 폰트 사용)
        try:
            font = ImageFont.load_default()
        except:
            font = None

        # 주차 구역 그리기
        for i, (x1, y1, x2, y2) in enumerate(parking_spots):
            # 직사각형 그리기 (빨간색)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

            # 번호 표시
            text = f"P{i+1}"
            if font:
                draw.text((x1+5, y1+5), text, fill=(255, 0, 0), font=font)
            else:
                draw.text((x1+5, y1+5), text, fill=(255, 0, 0))

        return np.array(pil_image)

    def analyze_parking_lot(self, image_path: str,
                          method: str = "auto") -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """주차장 분석 실행"""
        # 이미지 로드
        image = self.load_image(image_path)
        print(f"이미지 크기: {image.shape}")

        # 주차 구역 감지
        if method == "auto":
            parking_spots = self.detect_grid_pattern(image)
        elif method == "manual":
            parking_spots = self.manual_grid_detection(image.shape)
        else:
            raise ValueError("method는 'auto' 또는 'manual'이어야 합니다.")

        print(f"감지된 주차 구역 수: {len(parking_spots)}")

        # 결과 이미지 생성
        result_image = self.draw_results(image, parking_spots)

        return result_image, parking_spots

    def save_results(self, result_image: np.ndarray,
                    parking_spots: List[Tuple[int, int, int, int]],
                    output_dir: str = "."):
        """결과 저장"""
        os.makedirs(output_dir, exist_ok=True)

        # 결과 이미지 저장
        result_pil = Image.fromarray(result_image)
        output_path = os.path.join(output_dir, "simple_parking_result.jpg")
        result_pil.save(output_path)
        print(f"결과 이미지 저장: {output_path}")

        # 분석 결과 저장
        info_path = os.path.join(output_dir, "simple_parking_info.txt")
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("=== 간단한 주차장 분석 결과 ===\n")
            f.write(f"총 주차 구역 수: {len(parking_spots)}\n\n")

            for i, (x1, y1, x2, y2) in enumerate(parking_spots):
                width = x2 - x1
                height = y2 - y1
                f.write(f"주차구역 {i+1}: ({x1}, {y1}) - ({x2}, {y2}) "
                       f"크기: {width}x{height}\n")

        print(f"분석 정보 저장: {info_path}")


def demo_simple_parking():
    """간단한 주차장 감지 데모"""
    detector = SimpleParkingDetector()
    image_path = "parkinglot1.jpg"

    if not os.path.exists(image_path):
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        print("샘플 이미지를 생성합니다...")

        # 샘플 이미지 생성
        sample_image = create_sample_parking_image()
        sample_path = "sample_parking.jpg"
        Image.fromarray(sample_image).save(sample_path)
        image_path = sample_path
        print(f"샘플 이미지 생성: {sample_path}")

    try:
        # 자동 감지
        print("\n=== 자동 감지 방식 ===")
        result_auto, spots_auto = detector.analyze_parking_lot(image_path, method="auto")

        # 수동 격자 방식
        print("\n=== 수동 격자 방식 ===")
        result_manual, spots_manual = detector.analyze_parking_lot(image_path, method="manual")

        # 결과 저장
        detector.save_results(result_auto, spots_auto, "output_auto")
        detector.save_results(result_manual, spots_manual, "output_manual")

        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 원본 이미지
        original = detector.load_image(image_path)
        axes[0, 0].imshow(original)
        axes[0, 0].set_title("원본 이미지")
        axes[0, 0].axis('off')

        # 그레이스케일 + 엣지
        gray = detector.to_grayscale(original)
        edges = detector.simple_edge_detection(gray)
        axes[0, 1].imshow(edges, cmap='gray')
        axes[0, 1].set_title("엣지 검출 결과")
        axes[0, 1].axis('off')

        # 자동 감지 결과
        axes[1, 0].imshow(result_auto)
        axes[1, 0].set_title(f"자동 감지 ({len(spots_auto)}개 구역)")
        axes[1, 0].axis('off')

        # 수동 격자 결과
        axes[1, 1].imshow(result_manual)
        axes[1, 1].set_title(f"수동 격자 ({len(spots_manual)}개 구역)")
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig("parking_analysis_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\n=== 분석 완료 ===")
        print(f"자동 감지: {len(spots_auto)}개 구역")
        print(f"수동 격자: {len(spots_manual)}개 구역")

    except Exception as e:
        print(f"오류 발생: {e}")


def create_sample_parking_image() -> np.ndarray:
    """샘플 주차장 이미지 생성"""
    width, height = 600, 400
    image = np.ones((height, width, 3), dtype=np.uint8) * 200  # 밝은 회색 배경

    # 격자 그리기
    rows, cols = 3, 5
    cell_width = width // cols
    cell_height = height // rows

    for i in range(rows + 1):
        y = i * cell_height
        image[y:y+2, :] = [100, 100, 100]  # 수평선

    for j in range(cols + 1):
        x = j * cell_width
        image[:, x:x+2] = [100, 100, 100]  # 수직선

    return image


if __name__ == "__main__":
    demo_simple_parking()