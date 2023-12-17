import numpy as np
import cv2
from ultralytics import YOLO
import copy
import domainLayer.utils as utils

PERSON = 2
BALL = 0
RIM = 3

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

class ShotMadeDetector:
    def __init__(self, rimCoords):
        self.model = YOLO("./domainLayer/models/all_detections_model.pt")
        self.rimCoords = rimCoords

        self.rimCenter = np.mean(rimCoords, axis=0)
        self.rimDiameter = np.linalg.norm(rimCoords[0] - rimCoords[1])

        self.centerTop = int(self.rimCenter[1] - self.rimDiameter)
        self.centerBottom = int(self.rimCenter[1] - 2)
        self.centerLeft = int(self.rimCenter[0] - self.rimDiameter / 2)
        self.centerRight = int(self.rimCenter[0] + self.rimDiameter / 2)

        self.belowTop = int(self.rimCenter[1] + 2)
        self.belowBottom = int(self.rimCenter[1] + self.rimDiameter)
        self.belowLeft = int(self.rimCenter[0] - self.rimDiameter / 2)
        self.belowRight = int(self.rimCenter[0] + self.rimDiameter / 2)

        # above center square coordinates
        self.aboveTop = int(self.rimCenter[1] - self.rimDiameter * 2)
        self.aboveBottom = int(self.rimCenter[1] - self.rimDiameter - 2)
        self.aboveLeft = int(self.rimCenter[0] - self.rimDiameter / 2)
        self.aboveRight = int(self.rimCenter[0] + self.rimDiameter / 2)

        # right center square coordinates
        self.rightTop = int(self.rimCenter[1] - self.rimDiameter)
        self.rightBottom = int(self.rimCenter[1] - 2)
        self.rightLeft = int(self.rimCenter[0] + self.rimDiameter / 2 + 2)
        self.rightRight = int(self.rimCenter[0] + self.rimDiameter * 1.5)

        # left center square coordinates
        self.leftTop = int(self.rimCenter[1] - self.rimDiameter)
        self.leftBottom = int(self.rimCenter[1] - 2)
        self.leftLeft = int(self.rimCenter[0] - self.rimDiameter * 1.5)
        self.leftRight = int(self.rimCenter[0] - self.rimDiameter / 2 - 2)

        self.lastSquare = None


    def checkBallPresence(self, backboardCrop, model, dictBackboard, lastSquare):
        backboardCropToDraw = copy.deepcopy(backboardCrop)
        resultsCrop = model.predict(backboardCrop, device=0, conf=0.3, show=False, save=False)
        boxesCrop = resultsCrop[0].boxes.xyxy.cpu().numpy().astype(int)
        classesCrop = resultsCrop[0].boxes.cls.cpu().numpy().astype(int)

        ballCenter = np.array([0, 0]).astype('float64')
        ballCount = 0

        for i, box in enumerate(boxesCrop):
            if classesCrop[i] == BALL:
                cv2.rectangle(backboardCropToDraw, (box[0], box[1]), (box[2], box[3]), (255, 165, 0), 2)
                ballCenter = np.array([box[0] + box[2], box[1] + box[3]]) / 2.0
                ballCount += 1


                if dictBackboard["center"][0] < ballCenter[0] < dictBackboard["center"][2] and dictBackboard["center"][1] < ballCenter[1] < dictBackboard["center"][3]:
                    cv2.rectangle(backboardCropToDraw, (dictBackboard["center"][0], dictBackboard["center"][1]), (dictBackboard["center"][2], dictBackboard["center"][3]), GREEN, 2)
                    lastSquare = "center"

                elif dictBackboard["above"][0] < ballCenter[0] < dictBackboard["above"][2] and dictBackboard["above"][1] < ballCenter[1] < dictBackboard["above"][3]:
                    cv2.rectangle(backboardCropToDraw, (dictBackboard["above"][0], dictBackboard["above"][1]), (dictBackboard["above"][2], dictBackboard["above"][3]), GREEN, 2)
                    lastSquare = "above"
                
                elif dictBackboard["left"][0] < ballCenter[0] < dictBackboard["left"][2] and dictBackboard["left"][1] < ballCenter[1] < dictBackboard["left"][3]:
                    cv2.rectangle(backboardCropToDraw, (dictBackboard["left"][0], dictBackboard["left"][1]), (dictBackboard["left"][2], dictBackboard["left"][3]), GREEN, 2)
                    lastSquare = "left"
                
                elif dictBackboard["right"][0] < ballCenter[0] < dictBackboard["right"][2] and dictBackboard["right"][1] < ballCenter[1] < dictBackboard["right"][3]:
                    cv2.rectangle(backboardCropToDraw, (dictBackboard["right"][0], dictBackboard["right"][1]), (dictBackboard["right"][2], dictBackboard["right"][3]), GREEN, 2)
                    lastSquare = "right"
        
        if lastSquare == "center":
            cv2.rectangle(backboardCropToDraw, (dictBackboard["center"][0], dictBackboard["center"][1]), (dictBackboard["center"][2], dictBackboard["center"][3]), GREEN, 2)
        else:
            cv2.rectangle(backboardCropToDraw, (dictBackboard["center"][0], dictBackboard["center"][1]), (dictBackboard["center"][2], dictBackboard["center"][3]), RED, 2)
        
        if lastSquare == "above":
            cv2.rectangle(backboardCropToDraw, (dictBackboard["above"][0], dictBackboard["above"][1]), (dictBackboard["above"][2], dictBackboard["above"][3]), GREEN, 2)
        else:
            cv2.rectangle(backboardCropToDraw, (dictBackboard["above"][0], dictBackboard["above"][1]), (dictBackboard["above"][2], dictBackboard["above"][3]), RED, 2)
        
        if lastSquare == "left":
            cv2.rectangle(backboardCropToDraw, (dictBackboard["left"][0], dictBackboard["left"][1]), (dictBackboard["left"][2], dictBackboard["left"][3]), GREEN, 2)
        else:
            cv2.rectangle(backboardCropToDraw, (dictBackboard["left"][0], dictBackboard["left"][1]), (dictBackboard["left"][2], dictBackboard["left"][3]), RED, 2)
        
        if lastSquare == "right":
            cv2.rectangle(backboardCropToDraw, (dictBackboard["right"][0], dictBackboard["right"][1]), (dictBackboard["right"][2], dictBackboard["right"][3]), GREEN, 2)
        else:
            cv2.rectangle(backboardCropToDraw, (dictBackboard["right"][0], dictBackboard["right"][1]), (dictBackboard["right"][2], dictBackboard["right"][3]), RED, 2)


        if lastSquare == None:
            cv2.rectangle(backboardCropToDraw, (dictBackboard["center"][0], dictBackboard["center"][1]), (dictBackboard["center"][2], dictBackboard["center"][3]), RED, 2)
            cv2.rectangle(backboardCropToDraw, (dictBackboard["above"][0], dictBackboard["above"][1]), (dictBackboard["above"][2], dictBackboard["above"][3]), RED, 2)
            cv2.rectangle(backboardCropToDraw, (dictBackboard["left"][0], dictBackboard["left"][1]), (dictBackboard["left"][2], dictBackboard["left"][3]), RED, 2)
            cv2.rectangle(backboardCropToDraw, (dictBackboard["right"][0], dictBackboard["right"][1]), (dictBackboard["right"][2], dictBackboard["right"][3]), RED, 2)
        

        return lastSquare, backboardCropToDraw


    def inference(self, frame, frameToDraw):
        # make predictions
        results = self.model.predict(frame, device=0, conf=0.3, show=False, save=False)

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        ballCenter = np.array([0, 0]).astype('float64')
        ballCount = 0
        for i, box in enumerate(boxes):
            if classes[i] == BALL:
                cv2.rectangle(frameToDraw, (box[0], box[1]), (box[2], box[3]), (255, 165, 0), 2)
                ballCenter += np.array([box[0] + box[2], box[1] + box[3]]) / 2.0
                ballCount += 1
        
        backboardCrop = frame[self.aboveTop - int(self.rimDiameter/4):self.centerBottom + int(self.rimDiameter/4), 
                                  self.leftLeft - int(self.rimDiameter/4):self.rightRight + int(self.rimDiameter/4)]
        
        dictBackboard = {"center": (self.centerLeft-self.leftLeft + int(self.rimDiameter/4), self.centerTop - self.aboveTop + int(self.rimDiameter/4), self.centerRight-self.leftLeft + int(self.rimDiameter/4), self.centerBottom - self.aboveTop + int(self.rimDiameter/4)),
                         "above":  (self.aboveLeft-self.leftLeft + int(self.rimDiameter/4),  self.aboveTop - self.aboveTop + int(self.rimDiameter/4),  self.aboveRight-self.leftLeft + int(self.rimDiameter/4),  self.aboveBottom - self.aboveTop + int(self.rimDiameter/4)),
                         "left":   (self.leftLeft-self.leftLeft + int(self.rimDiameter/4),   self.leftTop-self.aboveTop + int(self.rimDiameter/4),     self.leftRight-self.leftLeft + int(self.rimDiameter/4),   self.leftBottom-self.aboveTop + int(self.rimDiameter/4)),
                         "right":  (self.rightLeft-self.leftLeft + int(self.rimDiameter/4),  self.rightTop-self.aboveTop + int(self.rimDiameter/4),    self.rightRight-self.leftLeft + int(self.rimDiameter/4),  self.rightBottom-self.aboveTop + int(self.rimDiameter/4))}
        
        self.lastSquare, backboardCropToDraw = self.checkBallPresence(backboardCrop, self.model, dictBackboard, self.lastSquare)

        cv2.imshow("backboardCrop", backboardCropToDraw)

        if ballCenter[1] > self.belowBottom:
            if self.lastSquare == "center":
                self.lastSquare = None
                return True
            else:
                self.lastSquare = None
                return False

        return False

    def getCroppedFrame(self, frame):
        return frame[self.aboveTop - int(self.rimDiameter/4):self.centerBottom + int(self.rimDiameter/4), 
                     self.leftLeft - int(self.rimDiameter/4):self.rightRight + int(self.rimDiameter/4)]