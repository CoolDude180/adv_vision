from gc import is_tracked
import cv2
import numpy as np
from ultralytics import YOLO
model26n = YOLO("yolo26n.pt")   #название нейронной модели
camera1 = cv2.VideoCapture(0)   #камера с индексом 0

def track_26n (picture):   #прогоняет кадр через нейросеть
    results = model26n.track(  #задание параметров модели (yolo), передача кадра нейронке
        picture,  #входной параметр
        conf=0.5,  #порог уверенности
        classes=[0, 2, 16],  #классы: 0-человек, 2-автомобиль, 16-собака
        max_det=30,  #макс число объектов в кадре
        quantize=16,  #точность результата (16 бит на число)
        vid_stride=1,  #обрабатывает 1 из n кадров
        verbose=False,  #логировать результаты распознавания
        agnostic_nms=False
    )
    return results

def frame_center(boxes):   #центр bb объекта [x, y]
    coords = [boxes.xywh[0], boxes.xywh[1]]
    return coords

def ind_general(boxes, id_tracked):   #из индекса (id_tracked) по трекеру находит общий индекс в тензоре(массиве) boxes
    for box in boxes:
        if int(box.id[0]) == id_tracked:
            return box.id[0]
    return None


try:
    if not camera1.isOpened(): raise RuntimeError("Не удалось открыть камеру")   #проверка открылась ли камера
    cv2.namedWindow("camera 1", cv2.WINDOW_AUTOSIZE)   #создание окна
    while True:
        success, frame = camera1.read()   #считывает кадр в frame
        if not success:
            print("не удалось получить кадр")
            break
        ###начало###алгоритма###

        res = track_26n(frame)[0]   #res = (1й) запредикченный кадр (формат->??)
        frame_marked = res.plot()  # разметка кадра, frame_marked - изображение (массив numpy)
        boxes = res.boxes   #доступ к атрибуту объекта, содержащему информацию об обнаруженных объектах
        idS_tracked = boxes.id.int().tolist()   #список id_tracked
        # for id_tracked in idS_tracked:
        #     ind = ind_general(boxes, id_tracked)
        #     cv2.circle(frame_marked, center=(frame_center(boxes)[0], frame_center(boxes)[1]), radius=5, color=(0, 0, 255), thickness=-1)




        cv2.imshow("camera 1", frame_marked)   #вывод кадра
        if cv2.waitKey(1) & 0xFF == 27:   #"0xFF" отсекает лишние биты в возвращ. значении (т.е. даблчек того, что нажат именно esc)
            break   #выход по esc
        if cv2.getWindowProperty("camera 1", cv2.WND_PROP_VISIBLE) < 1:   #проверка закрыто ли окно (через крестик)
            break   #выход по закрытию окна

finally:
    camera1.release()
    cv2.destroyAllWindows()