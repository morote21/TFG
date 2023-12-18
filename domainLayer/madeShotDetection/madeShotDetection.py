import numpy as np
import cv2
from ultralytics import YOLO
import copy
import domainLayer.utils as utils

BALL = 0
MADE = 1
PERSON = 2
RIM = 3
SHOOT = 4

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

        self.lastBallPos = None



    def checkBallPresence(self, backboardCrop, dictBackboard):
        backboardCropToDraw = copy.deepcopy(backboardCrop)
        resultsCrop = self.model.predict(backboardCrop, device=0, conf=0.3, show=False, save=False)
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
                    self.lastSquare = "center"

                elif dictBackboard["above"][0] < ballCenter[0] < dictBackboard["above"][2] and dictBackboard["above"][1] < ballCenter[1] < dictBackboard["above"][3]:
                    cv2.rectangle(backboardCropToDraw, (dictBackboard["above"][0], dictBackboard["above"][1]), (dictBackboard["above"][2], dictBackboard["above"][3]), GREEN, 2)
                    self.lastSquare = "above"
                
                elif dictBackboard["left"][0] < ballCenter[0] < dictBackboard["left"][2] and dictBackboard["left"][1] < ballCenter[1] < dictBackboard["left"][3]:
                    cv2.rectangle(backboardCropToDraw, (dictBackboard["left"][0], dictBackboard["left"][1]), (dictBackboard["left"][2], dictBackboard["left"][3]), GREEN, 2)
                    self.lastSquare = "left"
                
                elif dictBackboard["right"][0] < ballCenter[0] < dictBackboard["right"][2] and dictBackboard["right"][1] < ballCenter[1] < dictBackboard["right"][3]:
                    cv2.rectangle(backboardCropToDraw, (dictBackboard["right"][0], dictBackboard["right"][1]), (dictBackboard["right"][2], dictBackboard["right"][3]), GREEN, 2)
                    self.lastSquare = "right"
        
        if self.lastSquare == "center":
            cv2.rectangle(backboardCropToDraw, (dictBackboard["center"][0], dictBackboard["center"][1]), (dictBackboard["center"][2], dictBackboard["center"][3]), GREEN, 2)
        else:
            cv2.rectangle(backboardCropToDraw, (dictBackboard["center"][0], dictBackboard["center"][1]), (dictBackboard["center"][2], dictBackboard["center"][3]), RED, 2)
        
        if self.lastSquare == "above":
            cv2.rectangle(backboardCropToDraw, (dictBackboard["above"][0], dictBackboard["above"][1]), (dictBackboard["above"][2], dictBackboard["above"][3]), GREEN, 2)
        else:
            cv2.rectangle(backboardCropToDraw, (dictBackboard["above"][0], dictBackboard["above"][1]), (dictBackboard["above"][2], dictBackboard["above"][3]), RED, 2)
        
        if self.lastSquare == "left":
            cv2.rectangle(backboardCropToDraw, (dictBackboard["left"][0], dictBackboard["left"][1]), (dictBackboard["left"][2], dictBackboard["left"][3]), GREEN, 2)
        else:
            cv2.rectangle(backboardCropToDraw, (dictBackboard["left"][0], dictBackboard["left"][1]), (dictBackboard["left"][2], dictBackboard["left"][3]), RED, 2)
        
        if self.lastSquare == "right":
            cv2.rectangle(backboardCropToDraw, (dictBackboard["right"][0], dictBackboard["right"][1]), (dictBackboard["right"][2], dictBackboard["right"][3]), GREEN, 2)
        else:
            cv2.rectangle(backboardCropToDraw, (dictBackboard["right"][0], dictBackboard["right"][1]), (dictBackboard["right"][2], dictBackboard["right"][3]), RED, 2)


        if self.lastSquare == None:
            cv2.rectangle(backboardCropToDraw, (dictBackboard["center"][0], dictBackboard["center"][1]), (dictBackboard["center"][2], dictBackboard["center"][3]), RED, 2)
            cv2.rectangle(backboardCropToDraw, (dictBackboard["above"][0], dictBackboard["above"][1]), (dictBackboard["above"][2], dictBackboard["above"][3]), RED, 2)
            cv2.rectangle(backboardCropToDraw, (dictBackboard["left"][0], dictBackboard["left"][1]), (dictBackboard["left"][2], dictBackboard["left"][3]), RED, 2)
            cv2.rectangle(backboardCropToDraw, (dictBackboard["right"][0], dictBackboard["right"][1]), (dictBackboard["right"][2], dictBackboard["right"][3]), RED, 2)
        

        return backboardCropToDraw


    def getShotResult(self, frame, frameToDraw, ballCenter, ballSize):
        """
        Returns whether a shot was made or not, the size of the ball and whether the shot ended or not
        :param frame: frame to make inference on
        :param frameToDraw: frame where the inference will be drawn
        :param ballCenter: center of the ball
        :param ballSize: size of the ball
        :return: whether a shot was made or not, the size of the ball and whether the shot ended or not
        """

        if ballSize is not None:
            cv2.rectangle(frameToDraw, (ballSize[0], ballSize[1]), (ballSize[2], ballSize[3]), (255, 165, 0), 2)
            cv2.circle(frameToDraw, (int(ballCenter[0]), int(ballCenter[1])), 3, (255, 0, 0), 2)
            
        
        backboardCrop = frame[self.aboveTop - int(self.rimDiameter/4):self.centerBottom + int(self.rimDiameter/4), 
                                  self.leftLeft - int(self.rimDiameter/4):self.rightRight + int(self.rimDiameter/4)]
        
        dictBackboard = {"center": (self.centerLeft-self.leftLeft + int(self.rimDiameter/4), self.centerTop - self.aboveTop + int(self.rimDiameter/4), self.centerRight-self.leftLeft + int(self.rimDiameter/4), self.centerBottom - self.aboveTop + int(self.rimDiameter/4)),
                         "above":  (self.aboveLeft-self.leftLeft + int(self.rimDiameter/4),  self.aboveTop - self.aboveTop + int(self.rimDiameter/4),  self.aboveRight-self.leftLeft + int(self.rimDiameter/4),  self.aboveBottom - self.aboveTop + int(self.rimDiameter/4)),
                         "left":   (self.leftLeft-self.leftLeft + int(self.rimDiameter/4),   self.leftTop-self.aboveTop + int(self.rimDiameter/4),     self.leftRight-self.leftLeft + int(self.rimDiameter/4),   self.leftBottom-self.aboveTop + int(self.rimDiameter/4)),
                         "right":  (self.rightLeft-self.leftLeft + int(self.rimDiameter/4),  self.rightTop-self.aboveTop + int(self.rimDiameter/4),    self.rightRight-self.leftLeft + int(self.rimDiameter/4),  self.rightBottom-self.aboveTop + int(self.rimDiameter/4))}
        
        backboardCropToDraw = self.checkBallPresence(backboardCrop, dictBackboard)

        cv2.imshow("backboardCrop", backboardCropToDraw)
        
        # TODO: mirar que en el frame anterior el centro haya estado por encima de belowBottom
        if ballSize is not None:
            if self.isBallBelowRim(ballCenter, self.centerBottom):
                if self.lastSquare == "center":
                    self.lastSquare = None
                    return True, ballSize, True
                else:
                    self.lastSquare = None
                    return False, ballSize, True
        
        
        return False, ballSize, False

    def getCroppedFrame(self, frame):
        return frame[self.aboveTop - int(self.rimDiameter/4):self.centerBottom + int(self.rimDiameter/4), 
                     self.leftLeft - int(self.rimDiameter/4):self.rightRight + int(self.rimDiameter/4)]
    

    def isBallOverRim(self, ballCenter):
        """
        Returns whether the ball is over the rim or not
        :param ballCenter: center of the ball
        :param rimHeight: height of the rim
        :return: whether the ball is over the rim or not
        """

        return ballCenter[1] < self.centerBottom


    def isBallBelowRim(self, ballCenter, rimHeight):
        """
        Returns whether the ball is below the rim or not
        :param ballCenter: center of the ball
        :param rimHeight: height of the rim
        :return: whether the ball is below the rim or not
        """
        return ballCenter[1] > rimHeight
    

    def whereBall(self, frame, frameToDraw):
        results = self.model.predict(frame, device=0, conf=0.3, show=False, save=False)

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        bestBallConf = 0
        bestBallBox = None
        for i, box in enumerate(boxes):
            if classes[i] == BALL:
                if results[0].boxes.conf[i] > bestBallConf:
                    bestBallConf = results[0].boxes.conf[i]
                    bestBallBox = box
        
        if bestBallBox is not None:
            ballCenter = np.array([bestBallBox[0] + bestBallBox[2], bestBallBox[1] + bestBallBox[3]]) / 2.0
            cv2.rectangle(frameToDraw, (bestBallBox[0], bestBallBox[1]), (bestBallBox[2], bestBallBox[3]), (255, 165, 0), 2)
            return ballCenter, bestBallBox

        return None, None
    

    def isBallInBackboard(self, frame):
        backboardCrop = frame[self.aboveTop - int(self.rimDiameter/4):self.centerBottom + int(self.rimDiameter/4), 
                                  self.leftLeft - int(self.rimDiameter/4):self.rightRight + int(self.rimDiameter/4)]
        backboardCropToDraw = copy.deepcopy(backboardCrop)
        
        ballCenter, ballBox = self.whereBall(backboardCrop, backboardCropToDraw)

        if ballCenter is not None:
            return True
        
        return False

