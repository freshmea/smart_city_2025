# pip install opencv-python

from typing import Text

import cv2
import numpy as np
from textSprite import TextSprite


class MainClass:
    def __init__(self):
        self.text = TextSprite(50, 50, "안녕하세요!", font_size=24, color=(255, 255, 255, 255), font_path='data/NanumGothic-Regular.ttf')
        self.fps = 30

    def run(self):
        while True:
            img = np.zeros((400, 600, 3), dtype=np.uint8)
            self.text.draw(img)
            self.text.color = (np.random.randint(256), np.random.randint(256), np.random.randint(256), 255)
            self.text.x += 10
            if self.text.x > img.shape[1]:
                self.text.x = 0
            self.text.update()
            cv2.imshow("mainclass", img)
            if cv2.waitKey(1000//int(self.fps)) == 27: # ms fps
                break

def main():
    app = MainClass()
    app.run()


if __name__ == "__main__":
    main()
