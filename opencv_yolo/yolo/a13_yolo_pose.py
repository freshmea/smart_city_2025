import time

import cv2
import torch
from ultralytics import YOLO


def main():
    print("cuda on" if torch.cuda.is_available() else "cuda off")

    model = YOLO("yolov8n-pose.pt")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    fps = cap.get(cv2.CAP_PROP_FPS) # 30
    print("fps:", fps)

    tm = cv2.TickMeter()
    while True:
        tm.start()
        ret , img = cap.read()
        if not ret:
            break
        cv2.imshow("video", img)
        if tm.getTimeTicks() < 1000 // int(fps):
            time.sleep((1000 // int(fps) - tm.getTimeTicks()) / 1000)
        cv2.putText(img, f"FPS: {tm.getFPS():.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("video", img)
        tm.stop()
    cap.release()
    results = model.predict(img)  # type: ignore
    print(results[0].keypoints)
    print(results[0].names)