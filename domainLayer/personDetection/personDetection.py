from ultralytics import YOLO
import cv2


def drawBoundingBoxPlayer(frame, box, identity, segmentedCourt, association, action, playerWithBall):
    """
    Draws a bounding box around the player and writes its identity
    :param frame: frame to draw (np.array)
    :param box: bounding box of the player (list)
    :param identity: identity of the player (int)
    :param segmentedCourt: segmented court (np.array)
    :param association: association of the player (int)
    :param action: action of the player (string)
    :param playerWithBall: player with the ball (int)
    :return: frame with the bounding box and the identity (np.array)
    """

    floorPoint = ((box[0] + box[2]) // 2, box[3])
    # get value of floorpoint in segmentedCourt
    value = segmentedCourt[floorPoint[1]][floorPoint[0]]
    if value:
        if playerWithBall is not None and playerWithBall == identity:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 2)
            
        else:
            if association == 0:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            elif association == 1:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 165, 0), 2)

        cv2.putText(frame, str(identity), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, str(identity), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.circle(frame, floorPoint, 3, (255, 0, 0), 2)

        #print action after identity
        cv2.putText(frame, str(action), (box[0] + 25, box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, str(action), (box[0] + 25, box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame

class Tracker:

    def __init__(self):
        """
        Initializes the tracker
        """
        self.model = YOLO("./domainLayer/models/yolov8m.pt")

    def trackPlayers(self, frame):
        """
        Returns a list of bounding boxes, ids and classes of the players in the frame
        :param frame: frame to track (np.array)
        :return: boxes, ids and classes of the players (list, list, list)     
        """
        results = self.model.track(source=frame, show=False, save=False, persist=True, tracker="botsort.yaml", conf=0.3, device=0)
        if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls

            indexes = [i for i, cls in enumerate(classes) if self.getClassName(cls) == "person"]    # cambiar por el cls directo de person para no buscar en self.model.names

            boxes = boxes[indexes]
            ids = ids[indexes]
            classes = classes[indexes]

        else:
            boxes, ids, classes = [], [], []

        return boxes, ids, classes

    def getClassName(self, cls):
        """
        Returns the name of the class
        :param cls: class (int)
        :return: name of the class (string)
        """
        return self.model.names[int(cls)]    # Convert tensor to int and get name from model