# pip install opencv-python

import cv2
import numpy as np


def main():
    car = cv2.imread("data/car.bmp")
    cv2.dnn.readNet()
    cv2.imshow("car", car)
    cv2.waitKey(0) # ms


if __name__ == "__main__":
    main()
