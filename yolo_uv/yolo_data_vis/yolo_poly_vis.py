# /home/aa/smart_city_2025/suwon_image_annotator.py
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


class SuwonImageAnnotator:
    def __init__(self, xml_path, image_dir):
        self.xml_path = xml_path
        self.image_dir = image_dir
        self.annotations = self.parse_xml()

    def parse_xml(self):
        """Parse Suwon 20200720 XML file and extract annotation data"""
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        annotations = {}

        for image_elem in root.findall('.//image'):
            filename_full = image_elem.get('name', '')
            # Extract just the filename from full path
            filename = os.path.basename(filename_full) if filename_full else ''
            width = int(image_elem.get('width', 0))
            height = int(image_elem.get('height', 0))

            objects = []

            # Parse polygon annotations (not boxes)
            for polygon in image_elem.findall('.//polygon'):
                points_str = polygon.get('points', '')
                if points_str:
                    # Parse polygon points and convert to bounding box
                    points = []
                    for point_pair in points_str.split(';'):
                        if ',' in point_pair:
                            x, y = point_pair.split(',')
                            points.append((float(x), float(y)))

                    if points:
                        # Convert polygon to bounding box
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]

                        obj = {
                            'label': polygon.get('label', 'unknown'),
                            'xtl': min(x_coords),
                            'ytl': min(y_coords),
                            'xbr': max(x_coords),
                            'ybr': max(y_coords),
                            'polygon_points': points  # Keep original polygon points
                        }
                        objects.append(obj)

            annotations[filename] = {
                'width': width,
                'height': height,
                'objects': objects
            }

        return annotations

    def draw_annotations(self, image, objects):
        """Draw polygons, bounding boxes and labels on image"""
        colors = {
            'person': (0, 255, 0),
            'car': (255, 0, 0),
            'bicycle': (0, 0, 255),
            'motorcycle': (255, 255, 0),
            'bus': (255, 0, 255),
            'truck': (0, 255, 255),
            'unknown': (128, 128, 128)
        }

        for obj in objects:
            label = obj['label']
            color = colors.get(label, colors['unknown'])

            # Draw polygon if available
            if 'polygon_points' in obj and obj['polygon_points']:
                points = np.array([(int(p[0]), int(p[1])) for p in obj['polygon_points']], np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.polylines(image, [points], True, color, 2)

                # Fill polygon with semi-transparent color
                overlay = image.copy()
                cv2.fillPoly(overlay, [points], color)
                cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

            # Draw bounding box
            x1, y1 = int(obj['xtl']), int(obj['ytl'])
            x2, y2 = int(obj['xbr']), int(obj['ybr'])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return image

    def process_single_image(self, image_filename, output_dir=None):
        """Process and display a single image with annotations"""
        if image_filename not in self.annotations:
            print(f"No annotations found for {image_filename}")
            return None

        image_path = os.path.join(self.image_dir, image_filename)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None

        objects = self.annotations[image_filename]['objects']
        annotated_image = self.draw_annotations(image.copy(), objects)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"annotated_{image_filename}")
            cv2.imwrite(output_path, annotated_image)
            print(f"Saved annotated image: {output_path}")

        return annotated_image

    def process_all_images(self, output_dir=None, display=True):
        """Process all images with annotations"""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        for image_filename in self.annotations.keys():
            print(f"Processing: {image_filename}")
            annotated_image = self.process_single_image(image_filename, output_dir)

            if annotated_image is not None and display:
                # Resize for display if image is too large
                height, width = annotated_image.shape[:2]
                if width > 1200 or height > 800:
                    scale = min(1200/width, 800/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    annotated_image = cv2.resize(annotated_image, (new_width, new_height))

                cv2.imshow(f'Annotated - {image_filename}', annotated_image)
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()

                if key == ord('q'):
                    break

    def show_statistics(self):
        """Show annotation statistics"""
        total_images = len(self.annotations)
        total_objects = sum(len(ann['objects']) for ann in self.annotations.values())

        label_counts = {}
        for ann in self.annotations.values():
            for obj in ann['objects']:
                label = obj['label']
                label_counts[label] = label_counts.get(label, 0) + 1

        print(f"\n=== Suwon 20200720 Dataset Statistics ===")
        print(f"Total images: {total_images}")
        print(f"Total objects: {total_objects}")
        print(f"Average objects per image: {total_objects/total_images:.2f}")
        print(f"\nObject distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count}")

def main():
    # Configuration
    xml_file = "/home/aa/smart_city_2025/yolo_uv/yolo_data_vis/Suwon_CH02_20200720_1700_MON_9m_NH_highway_TW5_sunny_FHD.xml"
    image_directory = "/home/aa/smart_city_2025/yolo_uv/yolo_data_vis/Suwon_CH02_20200720_1700_MON_9m_NH_highway_TW5_sunny_FHD"
    output_directory = "/home/aa/smart_city_2025/yolo_uv/yolo_data_vis/"

    # Create annotator instance
    annotator = SuwonImageAnnotator(xml_file, image_directory)

    # Show dataset statistics
    annotator.show_statistics()

    # Process all images
    print("\nProcessing images...")
    annotator.process_all_images(output_directory, display=True)

    print("Processing completed!")

if __name__ == "__main__":
    main()