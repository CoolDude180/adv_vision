import cv2

cam1 = cv2.VideoCapture(0)  #камера с индексом 0

try:
    if not cam1.isOpened():
        raise RuntimeError("Не удалось открыть камеру")

    cv2.namedWindow("Camera 1", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Camera 1", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

    while True:
        success, frame = cam1.read()

        if not success:
            print("Не удалось получить кадр")
            break

        cv2.imshow("Camera 1", frame)

        if cv2.waitKey(1) & 0xFF == 27:     #"0xFF" отсекает лишние биты в возвращ. значении (т.е. это даблчек что нажат именно esc)
            break   #выход по esc

        if cv2.getWindowProperty("Camera 1", cv2.WND_PROP_VISIBLE) < 1:
            break   # выход по закрытию окна


finally:
    cam1.release()
    cv2.destroyAllWindows()