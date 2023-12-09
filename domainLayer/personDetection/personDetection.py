from ultralytics import YOLO
import cv2


def drawBoundingBoxPlayer(frame, box, identity, segmentedCourt, association, action):

    floorPoint = ((box[0] + box[2]) // 2, box[3])
    # get value of floorpoint in segmentedCourt
    # value = segmentedCourt[floorPoint[1]][floorPoint[0]]
    value = 1
    if value:
        if association == 0:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
        elif association == 1:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
        else:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 165, 0), 1)

        cv2.putText(frame, str(identity), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, str(identity), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.circle(frame, floorPoint, 3, (255, 0, 0), 2)

        #print action after identity
        cv2.putText(frame, str(action), (box[0] + 25, box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, str(action), (box[0] + 25, box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame

class Tracker:

    def __init__(self):
        self.model = YOLO("./domainLayer/models/yolov8m.pt")

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

    def getClassName(self, id):
        id = int(id.item())             # Convert tensor to int
        return self.model.names[id]