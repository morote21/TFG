from ultralytics import YOLO
import numpy as np
import copy
import cv2


class Tracker:

    def __init__(self):
        self.model = YOLO("yolov8x.pt")

    def track_players(self, frame):
        results = self.model.track(source=frame, show=False, save=False, persist=True, tracker="bytetrack.yaml")
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        # Aqui falta filtrar con el segmented court

        return boxes, ids

    def draw_bb_player(self, frame, boxes, ids, segmented_court):
        for box, id in zip(boxes, ids):
            floor_point = ((box[0] + box[2]) // 2, box[3])
            if segmented_court[floor_point[1]][floor_point[0]]:  # al ser coordenadas (x, y) van al reves de (row, col)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, str(id), (box[0], box[1] - 10), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 2)
                cv2.circle(frame, floor_point, 3, (0, 255, 0), 2)

        return frame
