import cv2
import numpy as np

# pip install opencv-python

def main():
    folder_path = "/home/aa/smart_city_2025/data/"
    faker = cv2.imread(folder_path + "faker.jpg")
    print("OpenCV version:", cv2.__version__)
    fps = 30
    delay = int(1000 / fps)
    a = 0
    while True:
        black_clone = faker.copy() # type: ignore
        cv2.rectangle(black_clone, (50+a, 50+a), (350-a, 350-a), (255, 0, 0), thickness=3)
        a = (a + 5) % 150
        a += 1
        if cv2.waitKey(delay) == 27:
            break
        cv2.imshow("black window", black_clone)

if __name__ == "__main__":
    main()
