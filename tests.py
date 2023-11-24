import numpy as np
import cv2
import copy
from domainLayer import utils


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

VIDEO_PATH = "/home/morote/Desktop/input_tfg/nba2k_test.mp4"
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

aboveTop = int(rimCenter[1] - rimDiameter)
aboveBottom = int(rimCenter[1] - 2)
aboveLeft = int(rimCenter[0] - rimDiameter / 2)
aboveRight = int(rimCenter[0] + rimDiameter / 2)

belowTop = int(rimCenter[1] + 2)
belowBottom = int(rimCenter[1] + rimDiameter)
belowLeft = int(rimCenter[0] - rimDiameter / 2)
belowRight = int(rimCenter[0] + rimDiameter / 2)

aboveSquare = sceneCopy[aboveTop:aboveBottom, aboveLeft:aboveRight]
belowSquare = sceneCopy[belowTop:belowBottom, belowLeft:belowRight]

# draw square on sceneCopy
while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = utils.resizeFrame(frame, height=1080)
    sceneCopy = copy.deepcopy(frame)

    cv2.rectangle(sceneCopy, (aboveLeft, aboveTop), (aboveRight, aboveBottom), (0, 255, 0), 2)
    cv2.rectangle(sceneCopy, (belowLeft, belowTop), (belowRight, belowBottom), (0, 255, 0), 2)

    cv2.imshow("scene", sceneCopy)
    cv2.waitKey(frameDelay)

video.release()
cv2.destroyAllWindows()
