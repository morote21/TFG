from ultralytics import YOLO
import numpy as np
import copy
import cv2


def draw_bb_player(frame, box, identity, segmented_court, association):

    floor_point = ((box[0] + box[2]) // 2, box[3])
    if segmented_court[floor_point[1]][floor_point[0]]:  # al ser coordenadas (x, y) van al reves de (row, col)
        if association:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.putText(frame, str(identity), (box[0], box[1] - 10), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 2)
        cv2.circle(frame, floor_point, 3, (0, 255, 0), 2)

    return frame


class Tracker:

    def __init__(self):
        self.model = YOLO("yolov8x.pt")
        #self.model = YOLO("./runs/detect/train2/weights/best.pt")

    def track_players(self, frame):
        results = self.model.track(source=frame, show=False, save=False, persist=True, tracker="bytetrack.yaml")
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        # Aqui falta filtrar con el segmented court

        return boxes, ids
