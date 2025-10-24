# pip install opencv-python

import cv2
import numpy as np


def main():
    black = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.imshow("black screen", black)
    car = cv2.imread("data/car.bmp")
    cv2.imshow("car", car)
    cv2.waitKey(0) # ms


if __name__ == "__main__":
    main()
