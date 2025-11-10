"""
실시간 웹캠 YOLO-OBB 모니터링
웹캠으로부터 실시간 영상을 받아 OBB 결과를 실시간으로 표시
"""

import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class RealTimeOBBMonitor:
    """실시간 OBB 모니터링 클래스"""

    def __init__(self, model_path: str = "../../yolov8n-obb.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 디바이스: {self.device}")

        # YOLO-OBB 모델 로드
        try:
            self.model = YOLO(model_path)
            print(f"✅ YOLO-OBB 모델 로드 완료")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            self.model = None

        # 클래스 이름과 색상
        self.class_names = {
            2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'
        }

        self.colors = {
            2: (0, 255, 0),    # car - 녹색
            3: (255, 0, 0),    # motorcycle - 파란색
            5: (0, 0, 255),    # bus - 빨간색
            7: (255, 255, 0),  # truck - 청록색
        }

        # 통계
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

    def draw_obb(self, image, obb_points, class_id, confidence, vehicle_id):
        """OBB 그리기"""
        color = self.colors.get(class_id, (128, 128, 128))
        class_name = self.class_names.get(class_id, f"class_{class_id}")

        # OBB 다각형 그리기
        points = obb_points.astype(np.int32)
        cv2.polylines(image, [points], True, color, 2)

        # 중심점 계산
        center_x = int(np.mean(obb_points[:, 0]))
        center_y = int(np.mean(obb_points[:, 1]))

        # 크기 계산
        width = np.linalg.norm(obb_points[1] - obb_points[0])
        height = np.linalg.norm(obb_points[2] - obb_points[1])

        # 중심점 표시
        cv2.circle(image, (center_x, center_y), 5, color, -1)

        # 라벨 배경
        label = f"V{vehicle_id}: {class_name}"
        size_text = f"{width:.0f}x{height:.0f}"
        conf_text = f"{confidence:.2f}"

        # 텍스트 크기 계산
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        (size_w, size_h), _ = cv2.getTextSize(size_text, font, font_scale-0.1, thickness-1)
        (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, font_scale-0.1, thickness-1)

        # 라벨 위치 계산
        label_x = center_x - label_w // 2
        label_y = center_y - 30

        # 배경 사각형
        cv2.rectangle(image,
                     (label_x - 5, label_y - label_h - 5),
                     (label_x + max(label_w, size_w, conf_w) + 5, label_y + size_h + conf_h + 10),
                     (0, 0, 0), -1)

        # 텍스트 그리기
        cv2.putText(image, label, (label_x, label_y), font, font_scale, color, thickness)
        cv2.putText(image, size_text, (label_x, label_y + 15), font, font_scale-0.1, (255, 255, 255), thickness-1)
        cv2.putText(image, conf_text, (label_x, label_y + 30), font, font_scale-0.1, (255, 255, 255), thickness-1)

        return center_x, center_y, width, height

    def process_frame(self, frame, conf_threshold=0.3):
        """프레임 처리"""
        if self.model is None:
            return frame, 0

        # YOLO 추론
        results = self.model(frame, verbose=False, conf=conf_threshold)

        vehicle_count = 0
        vehicle_info = []

        for result in results:
            # OBB 결과 처리
            if hasattr(result, 'obb') and result.obb is not None:
                for obb, conf, cls in zip(result.obb.xyxyxyxy, result.obb.conf, result.obb.cls):
                    class_id = int(cls)
                    confidence = float(conf)

                    # 차량 클래스만 처리
                    if class_id in [2, 3, 5, 7]:
                        vehicle_count += 1
                        obb_points = obb.cpu().numpy().reshape(-1, 2)

                        center_x, center_y, width, height = self.draw_obb(
                            frame, obb_points, class_id, confidence, vehicle_count
                        )

                        vehicle_info.append({
                            'id': vehicle_count,
                            'class': self.class_names.get(class_id, f"class_{class_id}"),
                            'center': (center_x, center_y),
                            'size': (width, height),
                            'confidence': confidence
                        })

            # 일반 박스 결과 처리
            elif hasattr(result, 'boxes') and result.boxes is not None:
                for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    class_id = int(cls)
                    confidence = float(conf)

                    if class_id in [2, 3, 5, 7]:
                        vehicle_count += 1
                        x1, y1, x2, y2 = [int(x) for x in box.cpu().numpy()]

                        # 박스를 OBB 형태로 변환
                        obb_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

                        center_x, center_y, width, height = self.draw_obb(
                            frame, obb_points, class_id, confidence, vehicle_count
                        )

                        vehicle_info.append({
                            'id': vehicle_count,
                            'class': self.class_names.get(class_id, f"class_{class_id}"),
                            'center': (center_x, center_y),
                            'size': (width, height),
                            'confidence': confidence
                        })

        return frame, vehicle_count, vehicle_info

    def draw_info_panel(self, frame, vehicle_count, vehicle_info):
        """정보 패널 그리기"""
        height, width = frame.shape[:2]

        # FPS 계산
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time

        # 상단 정보 패널
        panel_height = 80
        cv2.rectangle(frame, (0, 0), (width, panel_height), (0, 0, 0), -1)

        # 제목
        cv2.putText(frame, "YOLO-OBB Real-time Monitor", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 통계
        stats_text = f"FPS: {self.current_fps} | Vehicles: {vehicle_count} | Device: {self.device.type.upper()}"
        cv2.putText(frame, stats_text, (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # 차량 정보 (우측 패널)
        if vehicle_info:
            panel_width = 250
            panel_x = width - panel_width
            panel_y = panel_height

            info_height = min(len(vehicle_info) * 60 + 40, height - panel_height)
            cv2.rectangle(frame, (panel_x, panel_y), (width, panel_y + info_height), (0, 0, 0), -1)

            cv2.putText(frame, "Vehicle Details:", (panel_x + 10, panel_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            for i, vehicle in enumerate(vehicle_info[:8]):  # 최대 8개까지만 표시
                y_pos = panel_y + 50 + i * 60

                # 차량 정보
                info_text = f"V{vehicle['id']}: {vehicle['class']}"
                size_text = f"Size: {vehicle['size'][0]:.0f}x{vehicle['size'][1]:.0f}"
                conf_text = f"Conf: {vehicle['confidence']:.2f}"

                cv2.putText(frame, info_text, (panel_x + 10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, size_text, (panel_x + 10, y_pos + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, conf_text, (panel_x + 10, y_pos + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # 조작 가이드
        guide_y = height - 60
        cv2.rectangle(frame, (0, guide_y), (width, height), (0, 0, 0), -1)
        cv2.putText(frame, "Controls: 'q' = Quit | 'c' = Change confidence | 's' = Save frame",
                   (10, guide_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run_webcam_monitor(self, camera_id=0):
        """웹캠 모니터링 실행"""
        print(f"📹 웹캠 모니터링 시작 (카메라 ID: {camera_id})")

        # 웹캠 초기화
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ 카메라를 열 수 없습니다 (ID: {camera_id})")
            return

        # 카메라 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("✅ 웹캠 연결 성공")
        print("🎮 조작법:")
        print("   - 'q': 종료")
        print("   - 'c': 신뢰도 임계값 변경")
        print("   - 's': 현재 프레임 저장")

        conf_threshold = 0.3
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 프레임을 읽을 수 없습니다")
                break

            frame_count += 1

            # 프레임 처리
            processed_frame, vehicle_count, vehicle_info = self.process_frame(frame, conf_threshold)

            # 정보 패널 그리기
            self.draw_info_panel(processed_frame, vehicle_count, vehicle_info)

            # 화면에 표시
            cv2.imshow('YOLO-OBB Real-time Monitor', processed_frame)

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("🛑 사용자가 종료를 요청했습니다")
                break
            elif key == ord('c'):
                print(f"현재 신뢰도 임계값: {conf_threshold}")
                new_conf = input("새 신뢰도 임계값 (0.1-0.9): ")
                try:
                    conf_threshold = float(new_conf)
                    conf_threshold = max(0.1, min(0.9, conf_threshold))
                    print(f"✅ 신뢰도 임계값 변경: {conf_threshold}")
                except:
                    print("❌ 잘못된 입력")
            elif key == ord('s'):
                filename = f"realtime_capture_{frame_count:06d}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"💾 프레임 저장: {filename}")

        # 정리
        cap.release()
        cv2.destroyAllWindows()
        print("🎉 모니터링 종료")

    def run_image_monitor(self, image_path):
        """단일 이미지 모니터링"""
        print(f"🖼️ 이미지 모니터링: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 이미지를 읽을 수 없습니다: {image_path}")
            return

        conf_threshold = 0.3

        while True:
            # 원본 이미지 복사
            frame = image.copy()

            # 프레임 처리
            processed_frame, vehicle_count, vehicle_info = self.process_frame(frame, conf_threshold)

            # 정보 패널 그리기
            self.draw_info_panel(processed_frame, vehicle_count, vehicle_info)

            # 화면에 표시
            cv2.imshow('YOLO-OBB Image Monitor', processed_frame)

            # 키 입력 처리
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c'):
                print(f"현재 신뢰도 임계값: {conf_threshold}")
                new_conf = input("새 신뢰도 임계값 (0.1-0.9): ")
                try:
                    conf_threshold = float(new_conf)
                    conf_threshold = max(0.1, min(0.9, conf_threshold))
                    print(f"✅ 신뢰도 임계값 변경: {conf_threshold}")
                except:
                    print("❌ 잘못된 입력")
            elif key == ord('s'):
                filename = f"image_monitor_result.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"💾 결과 저장: {filename}")

        cv2.destroyAllWindows()


def main():
    """메인 함수"""
    print("🔍 실시간 YOLO-OBB 모니터링 시스템")
    print("="*50)

    monitor = RealTimeOBBMonitor()

    print("\n실행 모드를 선택하세요:")
    print("1. 웹캠 실시간 모니터링")
    print("2. 이미지 파일 모니터링")

    choice = input("선택 (1 또는 2): ")

    if choice == "1":
        camera_id = input("카메라 ID (기본값: 0): ")
        camera_id = int(camera_id) if camera_id.isdigit() else 0
        monitor.run_webcam_monitor(camera_id)

    elif choice == "2":
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
            image_path = input("이미지 파일 경로: ")

        if os.path.exists(image_path):
            monitor.run_image_monitor(image_path)
        else:
            print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")

    else:
        print("❌ 잘못된 선택입니다")

if __name__ == "__main__":
    import os
    main()