# pip install uv
# uv init
# uv add ultralytics
# uv run main.py

import cv2
import torch
from ultralytics import YOLO


def main():
    folder_path = "/home/aa/smart_city_2025/data/Cat/"
    print(torch.cuda.is_available())
    model = YOLO("yolov8n.pt")
    path = folder_path + "cat.4.jpg"
    img = cv2.imread(path)
    results = model.predict(img, verbose=True)
    res = results[0]
    annotated_frame = res.plot()
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    cv2.waitKey(0)  # ms

if __name__ == "__main__":
    main()
