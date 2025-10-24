import cv2


def main():
    vtest = 0


    cap = cv2.VideoCapture(vtest)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps:", fps)

    while True:
        ret , img = cap.read()
        if not ret:
            break
        cv2.imshow("video", img)
        if cv2.waitKey(1000//int(fps)) == 27: # ms fps
            break
    cap.release()

if __name__ == "__main__":
    main()
