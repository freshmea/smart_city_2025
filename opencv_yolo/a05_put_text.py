# pip install pillow freetype-py

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    # black = np.zeros((512, 512, 3), dtype=np.uint8)
    # cv2.putText(black, "Smart City Fighting!!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    korean_text = "스마트시티 파이팅!!"
    font = ImageFont.truetype("data/NanumGothic-Regular.ttf", 30)
    pt1, pt2 = (50, 230), (50, 310)

    image= np.zeros((350, 500, 3), dtype=np.uint8)
    image.fill(0)

    pil_img = Image.new('RGBA', (500, 350), (255,255,255,0))
    draw = ImageDraw.Draw(pil_img)

    bbox = draw.textbbox((0, 0), korean_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x, y = pt2[0], pt2[1] - text_height
    draw.text((x,y), korean_text, font=font, fill=(0, 0, 0, 255))

    pil_rgb = pil_img.convert('RGB') # RGBA-> RGB
    open_cv_img = cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR) # RGB -> BGR
    cv2.imshow("text image", open_cv_img)

    # cv2.imshow("black screen", black)
    cv2.waitKey(0) # ms


if __name__ == "__main__":
    main()
