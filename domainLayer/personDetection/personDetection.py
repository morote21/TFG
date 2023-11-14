from ultralytics import YOLO
import cv2


def draw_bb_player(frame, box, identity, segmented_court, association):

    floor_point = ((box[0] + box[2]) // 2, box[3])
    """
    if segmented_court[floor_point[1]][floor_point[0]]:  # al ser coordenadas (x, y) van al reves de (row, col)
        if association:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.putText(frame, str(identity), (box[0], box[1] - 10), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 2)
        cv2.circle(frame, floor_point, 3, (0, 255, 0), 2)
    """
    if association:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    else:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    cv2.putText(frame, str(identity), (box[0], box[1] - 10), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 2)
    cv2.circle(frame, floor_point, 3, (0, 255, 0), 2)


    return frame


class Tracker:

    def __init__(self):
        self.model = YOLO("./models/yolov8m.pt")

    def trackPlayers(self, frame):
        results = self.model.track(source=frame, show=False, save=False, persist=True, tracker="bytetrack.yaml", conf=0.3)
        if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls

        else:
            boxes, ids, classes = [], [], []
        # Aqui falta filtrar con el segmented court

        return boxes, ids, classes
