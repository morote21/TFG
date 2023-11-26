import cv2
import sys
import copy
import numpy as np
import torch
from domainLayer.topviewTransform.topview import Topview
from ultralytics import YOLO
from domainLayer.courtSegmentation import courtSegmentation as cs
from domainLayer.personDetection.personDetection import Tracker, drawBoundingBoxPlayer
from domainLayer.playerRecognition.teamAssociation import Teams
from domainLayer.actionAnalysis.actionAnalysis import ActionAnalysis
import domainLayer.actionRecognition.actionRecognition as ar
import domainLayer.utils as utils
import matplotlib.pyplot as plt
from domainLayer.statisticsGeneration.statisticsGeneration import StatisticsGenerator
from persistenceLayer.database import storeStatistics

# VIDEO_PATH = "/home/morote/Desktop/input_tfg/2000_0226_194537_003.MP4"
VIDEO_PATH = "/home/morote/Desktop/input_tfg/IMG_0500.mp4"
TOPVIEW_PATH = "/home/morote/Desktop/input_tfg/synthetic_court2.jpg"
TEAM_1_PLAYER = "/home/morote/Desktop/input_tfg/team1_black.png"
TEAM_2_PLAYER = "/home/morote/Desktop/input_tfg/team2_white.png"

SIZE_OF_ACTION_QUEUE = 7


def extractFrames(videoPath, team1path, team2path):
    videoFrames = []
    playerBoxes = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    video = cv2.VideoCapture(videoPath)

    success, frame = video.read()

    if not success:
        print('Failed to read video')
        sys.exit(1)

    team1Img = cv2.imread(team1path)
    team2Img = cv2.imread(team2path)

    teams = Teams(team1Img, team2Img, 6)                                # CREATE TEAMS OBJECT, WHICH CONTAINS DATA ABOUT BOTH TEAMS 

    playerTracker = Tracker()

    frameCount = 0
    while video.isOpened():
        print("Frame:", frameCount)
        success, frame = video.read()
        if not success:
            print("Video Ended")
            break

        frame = utils.resizeFrame(frame, height=1080)
        Width = frame.shape[1]
        Height = frame.shape[0]

        boxes, ids, classes = playerTracker.trackPlayers(frame=frame)

        frameCount += 1














def executeStatisticsGeneration(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    topviewImg = cv2.imread(TOPVIEW_PATH)

    actionAnalyzer = ActionAnalysis(topviewImage=topviewImg)
    statisticsGenerator = StatisticsGenerator(topviewImage=topviewImg)

    team1Img = cv2.imread(args.get("team1Path"))
    team2Img = cv2.imread(args.get("team2Path"))

    teams = Teams(team1Img, team2Img, 6)                                # CREATE TEAMS OBJECT, WHICH CONTAINS DATA ABOUT BOTH TEAMS 

    video = cv2.VideoCapture(args.get("videoPath"))                     # READ VIDEO

    ret, firstFrame = video.read()                                           # READ FIRST FRAME FOR TOPVIEW TRANSFORM COMPUTATION
    firstFrame = utils.resizeFrame(firstFrame, height=1080)                       # RESIZE FRAME TO 1080p

    if not ret:
        print("Error reading video frame.\n")

    sceneCpy = copy.deepcopy(firstFrame)
    topviewCpy = copy.deepcopy(topviewImg)

    # GET BORDERS OF THE COURT AND THE TOPVIEW IMAGE
    scenePoints = utils.getBorders(sceneCpy)
    topviewPoints = utils.getBorders(topviewCpy)

    twTransform = Topview()
    twTransform.computeTopview(scenePoints, topviewPoints)              # COMPUTE TOPVIEW MATRIX

    pts = np.array(twTransform.getSceneIntersections(), np.int32)

    segmentedCourt = cs.courtSegmentation(pts, firstFrame.shape)            # MASK TO FILTER PEOPLE INSIDE THE COURT                              

    playerTracker = Tracker()                                           # INITIALIZE TRACKER OBJECT            
    actionRecognizer = ar.ActionRecognition()                           # INITIALIZE ACTION RECOGNITION OBJECT

    prevActionsClassifications = {}                                     # DICTIONARY TO STORE THE PREVIOUS CLASSIFICATIONS OF EACH PLAYER

    while True:
        ret, frame = video.read()

        if not ret:
            break

        frame = utils.resizeFrame(frame, height=1080)                   # RESIZE FRAME TO 1080p
        boxes, ids, classes = playerTracker.trackPlayers(frame=frame)   # TRACK PLAYERS IN FRAME

        actions = actionRecognizer.inference(frame, boxes, ids, classes)          # PERFORM ACTION RECOGNITION
        
        for box, identity, cls in zip(boxes, ids, classes):         # DRAW BOUNDING BOXES WITH ID AND ACTION
            if playerTracker.getClassName(cls) == "person":
                crop = frame[box[1]:box[3], box[0]:box[2]]              # CROP PLAYER FROM FRAME FOR TEAM ASSOCIATION
                association = teams.associate(crop)                     # ASSOCIATE PLAYER WITH A TEAM
                frame = drawBoundingBoxPlayer(frame, box, identity, segmentedCourt, association, actions[identity]) 

                floorPoint = ((box[0] + box[2]) / 2, box[3])
                floorPointTransformed = twTransform.transformPoint(floorPoint)
                statisticsGenerator.storeStep((int(floorPointTransformed[0]), int(floorPointTransformed[1])), association)

                if actions[identity] == "shoot" and prevActionsClassifications[identity] == "ball in hand":
                    pos = (int(floorPointTransformed[0]), int(floorPointTransformed[1]))
                    shotValue = actionAnalyzer.shotDetected(pos)
                    statisticsGenerator.storeShot(pos, association, shotValue)

        prevActionsClassifications = copy.deepcopy(actions)
        
        topviewImageCpy = topviewImg.copy()

        # DRAW PLAYERS IN TOPVIEW
        for box in boxes:
            floorPoint = ((box[0] + box[2]) / 2, box[3])
            floorPointTransformed = twTransform.transformPoint(floorPoint)
            cv2.circle(topviewImageCpy, (int(floorPointTransformed[0]), int(floorPointTransformed[1])), 3,
                       (0, 255, 0), 2)


        cv2.imshow("frame", frame)
        cv2.imshow("topview", topviewImageCpy)
        key = cv2.waitKey(1)
        if key == 27:
            break

    video.release()
    cv2.destroyAllWindows()

    statisticsDict = statisticsGenerator.getStatistics()
    
    statisticsDict["firstFrame"] = firstFrame
    statisticsDict["scenePoints"] = scenePoints.tolist()
    storeStatistics(statisticsDict)

if __name__ == '__main__':
    argsDict = {
        "videoPath": VIDEO_PATH,
        "team1Path": TEAM_1_PLAYER,
        "team2Path": TEAM_2_PLAYER,
    }
    
    executeStatisticsGeneration(args=argsDict)