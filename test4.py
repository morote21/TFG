from ultralytics import YOLO
import numpy as np
import cv2
import copy
from domainLayer import utils
from domainLayer.topviewTransform.topview import allIntersections

def getClassName(cls):
        return model.names[int(cls)]


def courtSegmentation(pts, imgShape):
    # p1 = pts[0]
    # p2 = pts[2]
    # p3 = pts[1]
    # p4 = pts[3]

    # center = tv.intersection_point(p1, p2, p3, p4)

    # marker = np.zeros(shape=(img_shape[0], img_shape[1]), dtype=np.uint8)
    # mask = np.ones(shape=(img_shape[0], img_shape[1]), dtype=np.uint8)

    # marker[int(center[0])][int(center[1])] = 1

    # pts = pts.reshape((-1, 1, 2))
    # cv2.polylines(mask, [pts], True, (0, 0, 0), 5)

    # segmented_court = imreconstruct(marker, mask)

    segmentedCourt = np.zeros(imgShape, dtype=np.uint8)
    segmentedCourt = cv2.fillConvexPoly(segmentedCourt, pts, 255)
    segmentedCourt = cv2.cvtColor(segmentedCourt, cv2.COLOR_BGR2GRAY)
    segmentedCourt = cv2.threshold(segmentedCourt, 0, 255, cv2.THRESH_BINARY)[1].astype("bool")

    return segmentedCourt

model = YOLO("/home/morote/Desktop/TFG/domainLayer/models/yolov8m.pt")
IMAGE_PATH = "/home/morote/Pictures/person_outbounds.png"

image = cv2.imread(IMAGE_PATH)
image = utils.resizeFrame(image, height=1080)

scenePoints = utils.getBorders(image)
pts = np.array(allIntersections(scenePoints), np.int32)
print(pts)
segmentedCourt = courtSegmentation(pts, image.shape) 
cv2.imshow("court", segmentedCourt.astype(np.uint8) * 255)
cv2.waitKey(0)

results = model.predict(image, show=False, device=0)
boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
# ids = results[0].boxes.id.cpu().numpy().astype(int)
classes = results[0].boxes.cls
indexes = [i for i, cls in enumerate(classes) if getClassName(cls) == "person"] 

boxes = boxes[indexes]
classes = classes[indexes]

for box in boxes:
    floorPoint = ((box[0] + box[2]) // 2, box[3])
    insideCourt = segmentedCourt[floorPoint[1]][floorPoint[0]]
    if True: 
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)


cv2.imshow("image", image)
cv2.waitKey(0)