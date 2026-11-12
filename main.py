import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo26n.pt")   #название нейронной модели
camera1 = cv2.VideoCapture(0)   #камера с индексом 0
success, frame = camera1.read()   #считываем 1 кадр

results = model.predict(   #задание параметров модели (YOLO)
    frame,   #входной параметр
    conf=0.5,   #порог уверенности
    classes=[0, 2, 16],   #классы: 0-человек, 2-автомобиль, 16-собака
    max_det=30,   #макс число объектов в кадре
    quantize=16, #точность результата (16 бит на число)
    agnostic_nms = False,
    verbose=False   #логировать результаты распознавания
)

try:
    if not camera1.isOpened():
        raise RuntimeError("Не удалось открыть камеру")

    cv2.namedWindow("camera 1", cv2.WINDOW_AUTOSIZE)

    while True:
        success, frame = camera1.read()   #считывает кадр в frame

        if not success:
            print("не удалось получить кадр")
            break

        results = model.predict(  # задание параметров модели (yolo), передача кадра нейронке
            frame,  # входной параметр
            conf=0.5,  # порог уверенности
            classes=[0, 2, 16],  # классы: 0-человек, 2-автомобиль, 16-собака
            max_det=30,  # макс число объектов в кадре
            quantize=16,  # точность результата (16 бит на число)
            vid_stride = 1,   # обрабатывает 1 из n кадров
            verbose=False,  # логировать результаты распознавания
            agnostic_nms=False
        )

        res = results[0]

        frame_marked = res.plot()   #разметка кадра

        cv2.imshow("camera 1", frame_marked)   # вывод кадра

        if cv2.waitKey(1) & 0xFF == 27:   # "0xFF" отсекает лишние биты в возвращ. значении (т.е. даблчек того, что нажат именно esc)
            break   # выход по esc

        if cv2.getWindowProperty("camera 1", cv2.WND_PROP_VISIBLE) < 1:   # проверяет, закрыто ли окно (через крестик)
            break   # выход по закрытию окна


finally:
    camera1.release()
    cv2.destroyAllWindows()