import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


class YOLOVisualizer:
    def __init__(self, xml_path, image_dir=None):
        """
        YOLO XML 파일을 사용한 바운딩 박스 시각화 클래스

        Args:
            xml_path (str): XML 파일 경로
            image_dir (str): 이미지 파일들이 있는 디렉토리 경로
        """
        self.xml_path = xml_path
        self.image_dir = image_dir or os.path.dirname(xml_path)
        self.annotations = {}
        self.class_colors = {}

        self._parse_xml()
        self._generate_colors()

    def _parse_xml(self):
        """XML 파일을 파싱하여 어노테이션 정보 추출"""
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        # 각 이미지별 어노테이션 정보 저장
        for image in root.findall('image'):
            image_id = image.get('id')
            image_name = image.get('name')
            width = int(image.get('width'))
            height = int(image.get('height'))

            boxes = []
            for box in image.findall('box'):
                label = box.get('label')
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))

                boxes.append({
                    'label': label,
                    'xtl': xtl,
                    'ytl': ytl,
                    'xbr': xbr,
                    'ybr': ybr
                })

            self.annotations[image_id] = {
                'name': image_name,
                'width': width,
                'height': height,
                'boxes': boxes
            }

    def _generate_colors(self):
        """클래스별 색상 생성"""
        # XML에서 모든 클래스 추출
        classes = set()
        for annotation in self.annotations.values():
            for box in annotation['boxes']:
                classes.add(box['label'])

        # 각 클래스별로 랜덤 색상 할당
        for class_name in classes:
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            self.class_colors[class_name] = color

        print(f"클래스별 색상: {self.class_colors}")

    def draw_boxes_on_image(self, image_id, save_path=None, show=True):
        """
        특정 이미지에 바운딩 박스 그리기

        Args:
            image_id (str): 이미지 ID
            save_path (str): 저장할 경로 (None이면 저장하지 않음)
            show (bool): 화면에 표시할지 여부
        """
        if image_id not in self.annotations:
            print(f"이미지 ID {image_id}를 찾을 수 없습니다.")
            return None

        annotation = self.annotations[image_id]
        image_name = annotation['name']

        # 이미지 파일 경로 찾기
        image_path = os.path.join(self.image_dir, image_name)
        if not os.path.exists(image_path):
            print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            return None

        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 로드할 수 없습니다: {image_path}")
            return None

        # 바운딩 박스 그리기
        for box in annotation['boxes']:
            label = box['label']
            x1, y1 = int(box['xtl']), int(box['ytl'])
            x2, y2 = int(box['xbr']), int(box['ybr'])

            # 색상 가져오기
            color = self.class_colors.get(label, (255, 255, 255))

            # 바운딩 박스 그리기
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # 라벨 텍스트 배경
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)

            # 라벨 텍스트
            cv2.putText(image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 이미지 정보 텍스트
        info_text = f"Image: {image_name} | Boxes: {len(annotation['boxes'])}"
        cv2.putText(image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 저장
        if save_path:
            cv2.imwrite(save_path, image)
            print(f"이미지 저장됨: {save_path}")

        # 화면 표시
        if show:
            cv2.imshow(f'YOLO Visualization - {image_name}', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return image

    def draw_all_images(self, output_dir=None, show=False):
        """
        모든 이미지에 바운딩 박스 그리기

        Args:
            output_dir (str): 출력 디렉토리
            show (bool): 각 이미지를 화면에 표시할지 여부
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        for image_id, annotation in self.annotations.items():
            image_name = annotation['name']
            save_path = None

            if output_dir:
                save_path = os.path.join(output_dir, f"annotated_{image_name}")

            print(f"처리 중: {image_name} (ID: {image_id})")
            self.draw_boxes_on_image(image_id, save_path, show)

    def get_statistics(self):
        """데이터셋 통계 정보 출력"""
        total_images = len(self.annotations)
        total_boxes = sum(len(ann['boxes']) for ann in self.annotations.values())

        # 클래스별 박스 개수
        class_counts = {}
        for annotation in self.annotations.values():
            for box in annotation['boxes']:
                label = box['label']
                class_counts[label] = class_counts.get(label, 0) + 1

        print("=" * 50)
        print("데이터셋 통계")
        print("=" * 50)
        print(f"총 이미지 수: {total_images}")
        print(f"총 바운딩 박스 수: {total_boxes}")
        print(f"평균 박스 수/이미지: {total_boxes/total_images:.2f}")
        print("\n클래스별 바운딩 박스 수:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")
        print("=" * 50)

        return {
            'total_images': total_images,
            'total_boxes': total_boxes,
            'class_counts': class_counts
        }

def main():
    """메인 함수 - 사용 예제"""
    xml_path = "/home/aa/smart_city_2025/yolo_uv/yolo_data_vis/Suwon_CH01_20200721_2030_TUE_9m_NH_highway_TW5_sunny_FHD.xml"

    # 이미지 디렉토리 (XML과 같은 디렉토리라고 가정)
    image_dir = "/home/aa/smart_city_2025/yolo_uv/yolo_data_vis/images"

    # 시각화 객체 생성
    visualizer = YOLOVisualizer(xml_path, image_dir)

    # 통계 정보 출력
    visualizer.get_statistics()

    # 첫 번째 이미지만 시각화 (테스트용)
    first_image_id = list(visualizer.annotations.keys())[0]
    visualizer.draw_boxes_on_image(first_image_id,
                                  save_path="test_output.jpg",
                                  show=True)

    # 모든 이미지 시각화 (주석 해제하여 사용)
    # output_dir = "/home/aa/smart_city_2025/yolo_uv/yolo_data_vis/output"
    # visualizer.draw_all_images(output_dir=output_dir, show=False)

if __name__ == "__main__":
    main()
