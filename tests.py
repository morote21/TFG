import numpy as np
import cv2
import copy
from domainLayer import utils
from ultralytics import YOLO

# dataset definitions
PERSON = 2
BALL = 0
RIM = 3


def getRimCoords(image):
    def clickEvent(event, x, y, flags, param):        
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_LBUTTONUP:
            param.append((x, y))
    
    coords = []
    cv2.namedWindow("rim coords")
    cv2.setMouseCallback("rim coords", clickEvent, param=coords)

    while 1:
        cv2.imshow("rim coords", image)
        if len(coords) == 2:
            cv2.destroyAllWindows()
            break
        
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

    return np.array(coords)



OFFICIAL_RIM_DIAMETER = 46
OFFICIAL_BALL_DIAMETER = 24

VIDEO_PATH = "/home/morote/Desktop/input_tfg/shoot_test_2.mp4"
video = cv2.VideoCapture(VIDEO_PATH)
fps = video.get(cv2.CAP_PROP_FPS)
frameDelay = int(1000 / fps)

ret, frame = video.read()
frame = utils.resizeFrame(frame, height=1080) 

sceneCopy = copy.deepcopy(frame)

rimPoints = getRimCoords(sceneCopy)
print(rimPoints)
rimCenter = np.mean(rimPoints, axis=0)
rimDiameter = np.linalg.norm(rimPoints[0] - rimPoints[1])

centerTop = int(rimCenter[1] - rimDiameter)
centerBottom = int(rimCenter[1] - 2)
centerLeft = int(rimCenter[0] - rimDiameter / 2)
centerRight = int(rimCenter[0] + rimDiameter / 2)

belowTop = int(rimCenter[1] + 2)
belowBottom = int(rimCenter[1] + rimDiameter)
belowLeft = int(rimCenter[0] - rimDiameter / 2)
belowRight = int(rimCenter[0] + rimDiameter / 2)

# above center square coordinates
aboveTop = int(rimCenter[1] - rimDiameter * 2)
aboveBottom = int(rimCenter[1] - rimDiameter - 2)
aboveLeft = int(rimCenter[0] - rimDiameter / 2)
aboveRight = int(rimCenter[0] + rimDiameter / 2)

# right center square coordinates
rightTop = int(rimCenter[1] - rimDiameter)
rightBottom = int(rimCenter[1] - 2)
rightLeft = int(rimCenter[0] + rimDiameter / 2 + 2)
rightRight = int(rimCenter[0] + rimDiameter * 1.5)

# left center square coordinates
leftTop = int(rimCenter[1] - rimDiameter)
leftBottom = int(rimCenter[1] - 2)
leftLeft = int(rimCenter[0] - rimDiameter * 1.5)
leftRight = int(rimCenter[0] - rimDiameter / 2 - 2)



aboveSquare = sceneCopy[centerTop:centerBottom, centerLeft:centerRight]
belowSquare = sceneCopy[belowTop:belowBottom, belowLeft:belowRight]


model = YOLO('domainLayer/models/all_detections_model.pt')

shots_made = 0
last_square  = None

# draw square on sceneCopy
while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = utils.resizeFrame(frame, height=1080)
    sceneCopy = copy.deepcopy(frame)

    # make predictions
    results = model.predict(frame, device=0, conf=0.3, show=False, save=False)

    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    
    # draw boxes
    for i, box in enumerate(boxes):
        if classes[i] == PERSON:
            cv2.rectangle(sceneCopy, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
        elif classes[i] == BALL:
            cv2.rectangle(sceneCopy, (box[0], box[1]), (box[2], box[3]), (255, 165, 0), 1)
        elif classes[i] == RIM:
            cv2.rectangle(sceneCopy, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)

    
    # calculate center of ball
    ballCenter = np.array([0, 0]).astype('float64')
    ballCount = 0
    for i, box in enumerate(boxes):
        if classes[i] == BALL:
            ballCenter += np.array([box[0] + box[2], box[1] + box[3]]) / 2.0
            ballCount += 1
    

    if centerLeft < ballCenter[0] < centerRight and centerTop < ballCenter[1] < centerBottom:
        last_square = "center"
    
    elif aboveLeft < ballCenter[0] < aboveRight and aboveTop < ballCenter[1] < aboveBottom:
        last_square = "above"
    
    elif leftLeft < ballCenter[0] < leftRight and leftTop < ballCenter[1] < leftBottom:
        last_square = "left"
    
    elif rightLeft < ballCenter[0] < rightRight and rightTop < ballCenter[1] < rightBottom:
        last_square = "right"

    
    if ballCenter[1] > belowBottom:
        if last_square == "center":
            shots_made += 1

        last_square = None

    
    
    
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)

    if last_square == "center":
        cv2.rectangle(sceneCopy, (centerLeft, centerTop), (centerRight, centerBottom), GREEN, 2)
    else:
        cv2.rectangle(sceneCopy, (centerLeft, centerTop), (centerRight, centerBottom), RED, 2)
    if last_square == "above":
        cv2.rectangle(sceneCopy, (aboveLeft, aboveTop), (aboveRight, aboveBottom), GREEN, 2)
    else:
        cv2.rectangle(sceneCopy, (aboveLeft, aboveTop), (aboveRight, aboveBottom), RED, 2)
    if last_square == "left":
        cv2.rectangle(sceneCopy, (leftLeft, leftTop), (leftRight, leftBottom), GREEN, 2)
    else:
        cv2.rectangle(sceneCopy, (leftLeft, leftTop), (leftRight, leftBottom), RED, 2)
    if last_square == "right":
        cv2.rectangle(sceneCopy, (rightLeft, rightTop), (rightRight, rightBottom), GREEN, 2)
    else:
        cv2.rectangle(sceneCopy, (rightLeft, rightTop), (rightRight, rightBottom), RED, 2)

    if last_square is None:
        cv2.rectangle(sceneCopy, (centerLeft, centerTop), (centerRight, centerBottom), RED, 2)
        cv2.rectangle(sceneCopy, (aboveLeft, aboveTop), (aboveRight, aboveBottom), RED, 2)
        cv2.rectangle(sceneCopy, (leftLeft, leftTop), (leftRight, leftBottom), RED, 2)
        cv2.rectangle(sceneCopy, (rightLeft, rightTop), (rightRight, rightBottom), RED, 2)

    # show top left of window number of shots made
    cv2.putText(sceneCopy, "shots made: ", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, BLUE, 2)
    cv2.putText(sceneCopy, str(shots_made), (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, BLUE, 2)

    


    cv2.imshow("scene", sceneCopy)
    if cv2.waitKey(frameDelay) == 27:
        break

video.release()
cv2.destroyAllWindows()
