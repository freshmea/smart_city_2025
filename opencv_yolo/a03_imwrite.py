import cv2


def main():
    vtest = "data/vtest.avi"
    cap = cv2.VideoCapture(vtest)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps:", fps)

    # jpg compression quality 0 - 100
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]

    while True:
        ret , img = cap.read()
        if not ret:
            break
        cv2.imshow("video", img)
        if cv2.waitKey(1000//int(fps)) == 27: # ms fps
            cv2.imwrite("data/frame.png", img)
            cv2.imwrite("data/frame.jpg", img, encode_param)
            break
    cap.release()

if __name__ == "__main__":
    main()
