import cv2
import numpy as np

# pip install opencv-python

def main():
    # folder_path = "/home/aa/smart_city_2025/data/"
    # cap = cv2.VideoCapture(folder_path + "vtest.avi")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    fps = 30
    delay = int(1000 / fps)
    a = 0
    while True:
        ret, black_clone = cap.read()
        if not ret:
            break
        cv2.rectangle(black_clone, (50+a, 50+a), (350-a, 350-a), (255, 0, 0), thickness=3)
        a = (a + 5) % 150
        a += 1
        black_clone = cv2.Canny(black_clone, 100, 200)
        if cv2.waitKey(delay) == 27:
            break
        cv2.imshow("black window", black_clone)

if __name__ == "__main__":
    main()
