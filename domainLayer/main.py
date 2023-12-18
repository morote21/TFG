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
from persistenceLayer.database import storeStatistics, readTopviewPoints, readScenePoints
import json
from pathlib import Path
from domainLayer.madeShotDetection.madeShotDetection import ShotMadeDetector

# VIDEO_PATH = "/home/morote/Desktop/input_tfg/2000_0226_194537_003.MP4"
VIDEO_PATH = "/home/morote/Desktop/input_tfg/IMG_0500.mp4"
TOPVIEW_PATH = "database/topview/topview_image.jpg"
TEAM_1_PLAYER = "/home/morote/Desktop/input_tfg/team1_black.png"
TEAM_2_PLAYER = "/home/morote/Desktop/input_tfg/team2_white.png"
TOPVIEW_POINTS = "database/topview/topview_coords.json"

SIZE_OF_ACTION_QUEUE = 10

BLUE = (255, 0, 0)


def executeStatisticsGeneration_noImshow(videoPath, team1path, team2path):
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








def preprocessFrame(frame):
    frame = utils.resizeFrame(frame, height=1080)                   # RESIZE FRAME TO 1080p
    return frame





def executeStatisticsGeneration(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    topviewImg = cv2.imread(TOPVIEW_PATH)

    actionAnalyzer = ActionAnalysis(topviewImage=topviewImg)
    statisticsGenerator = StatisticsGenerator(topviewImage=topviewImg)

    if args.get("team1Path") is not None and args.get("team2Path") is not None:
        team1Img = cv2.imread(args.get("team1Path"))
        team2Img = cv2.imread(args.get("team2Path"))

        teams = Teams(team1Img, team2Img, 6)                                # CREATE TEAMS OBJECT, WHICH CONTAINS DATA ABOUT BOTH TEAMS
        statisticsGenerator.setNTeams(2)

    else:
        teams = None
        statisticsGenerator.setNTeams(0)
    

    video = cv2.VideoCapture(args.get("videoPath"))                     # READ VIDEO

    ret, firstFrame = video.read()                                           # READ FIRST FRAME FOR TOPVIEW TRANSFORM COMPUTATION
    firstFrame = utils.resizeFrame(firstFrame, height=1080)                       # RESIZE FRAME TO 1080p

    if not ret:
        print("Error reading video frame.\n")

    sceneCpy = copy.deepcopy(firstFrame)
    topviewCpy = copy.deepcopy(topviewImg)

    scenePoints = None
    topviewPoints = None
    # GET BORDERS OF THE COURT AND THE TOPVIEW IMAGE

    # if the scene is new, get the borders of the court and the topview image side from the user input
    if args.get("scenePointsPath") is None:
        scenePoints = utils.getBorders(sceneCpy)
        rimPoints = utils.getRim(sceneCpy)
        topviewPoints = readTopviewPoints(args.get("courtSide"))
    # if the scene is already used, get the borders of the court from the scenePoints.json file
    else:
        scenePoints, rimPoints, courtSide = readScenePoints(args.get("scenePointsPath"))
        topviewPoints = readTopviewPoints(courtSide)


    twTransform = Topview()
    twTransform.computeTopview(scenePoints, topviewPoints)              # COMPUTE TOPVIEW MATRIX

    pts = np.array(twTransform.getSceneIntersections(), np.int32)

    segmentedCourt = cs.courtSegmentation(pts, firstFrame.shape)            # MASK TO FILTER PEOPLE INSIDE THE COURT                              

    playerTracker = Tracker()                                           # INITIALIZE TRACKER OBJECT            
    actionRecognizer = ar.ActionRecognition()                           # INITIALIZE ACTION RECOGNITION OBJECT
    madeShotDetector = ShotMadeDetector(rimPoints)

    prevActionsClassifications = {}                                     # DICTIONARY TO STORE THE PREVIOUS CLASSIFICATIONS OF EACH PLAYER

    shotsMade = 0

    lastPlayerWithBall = None
    playerWhoShot = None
    posOfShot = None
    shotEnded = False

    processingShot = False

    lastBallCenter = None


    while True:
        ret, frame = video.read()

        if not ret:
            break

        frame = utils.resizeFrame(frame, height=1080)                   # RESIZE FRAME TO 1080p
        #frame = preprocessFrame(frame)                                  # PREPROCESS FRAME 

        frameToDraw = copy.deepcopy(frame)

        boxes, ids, classes = playerTracker.trackPlayers(frame=frame)   # TRACK PLAYERS IN FRAME

        actions = actionRecognizer.inference(frame, boxes, ids, classes)          # PERFORM ACTION RECOGNITION

        ballCenter, ballSize = madeShotDetector.whereBall(frame, frameToDraw)

        playerWithBall = None
        if ballSize is not None:
            playerWithBall = utils.whoHasPossession(zip(ids, boxes), ballSize)
            if (playerWithBall != playerWhoShot) and not processingShot:
                playerWhoShot = None
                posOfShot = None
                shotEnded = False

        # put text processingball
        cv2.putText(frameToDraw, "processing ball: ", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, BLUE, 2)
        cv2.putText(frameToDraw, str(processingShot), (450, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, BLUE, 2)

        if ballCenter is not None:
            lastBallCenter = ballCenter


        # correct some parameters
        if processingShot and playerWhoShot is None:
            processingShot = False
            shotEnded = False
            posOfShot = None 
        

        if not processingShot and lastBallCenter is not None:
            processingShot = madeShotDetector.isBallOverRim(lastBallCenter) or madeShotDetector.isBallInBackboard(frame)

        if playerWhoShot is not None and processingShot:
            made, ballSize, shotEnded = madeShotDetector.getShotResult(frame, frameToDraw, ballCenter, ballSize)

        
        if playerWithBall is not None:
            lastPlayerWithBall = playerWithBall

        if processingShot and shotEnded and playerWhoShot is not None:
            shotValue = actionAnalyzer.shotDetected(posOfShot)
            statisticsGenerator.storeShot(posOfShot, association, shotValue, made)
            cv2.circle(topviewImg, (int(posOfShot[0]), int(posOfShot[1])), 3, (255, 0, 0), 2)
            playerWhoShot = None
            posOfShot = None
            processingShot = False
            shotEnded = False

            cv2.destroyWindow("backboardCrop")

            if made:
                shotsMade += 1
        
        for box, identity, cls in zip(boxes, ids, classes):         # DRAW BOUNDING BOXES WITH ID AND ACTION
            #if playerTracker.getClassName(cls) == "person":
                
            crop = frame[box[1]:box[3], box[0]:box[2]]              # CROP PLAYER FROM FRAME FOR TEAM ASSOCIATION
            association = -1
            if teams is not None:
                association = teams.associate(crop)                     # ASSOCIATE PLAYER WITH A TEAM
            
            frameToDraw = drawBoundingBoxPlayer(frameToDraw, box, identity, segmentedCourt, association, actions[identity], playerWithBall) 

            floorPoint = ((box[0] + box[2]) / 2, box[3])
            floorPointTransformed = twTransform.transformPoint(floorPoint)
            statisticsGenerator.storeStep((int(floorPointTransformed[0]), int(floorPointTransformed[1])), association)

            # if actions[identity] == "shoot" and prevActionsClassifications[identity] == "ball in hand":
            #     pos = (int(floorPointTransformed[0]), int(floorPointTransformed[1]))
            #     shotValue = actionAnalyzer.shotDetected(pos)
            #     statisticsGenerator.storeShot(pos, association, shotValue)

            if actions[identity] == "shoot" and lastPlayerWithBall == identity and playerWhoShot is None:
                playerWhoShot = identity
                posOfShot = (int(floorPointTransformed[0]), int(floorPointTransformed[1]))
                #add point to topview

                shotEnded = False
                #shotValue = actionAnalyzer.shotDetected(posOfShot)
                #statisticsGenerator.storeShot(posOfShot, association, shotValue)

        prevActionsClassifications = copy.deepcopy(actions)
        
        topviewImageCpy = topviewImg.copy()

        # DRAW PLAYERS IN TOPVIEW
        for box in boxes:
            floorPoint = ((box[0] + box[2]) / 2, box[3])
            floorPointTransformed = twTransform.transformPoint(floorPoint)
            cv2.circle(topviewImageCpy, (int(floorPointTransformed[0]), int(floorPointTransformed[1])), 3,
                       (0, 255, 0), 2)

        cv2.putText(frameToDraw, "shots made: ", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, BLUE, 2)
        cv2.putText(frameToDraw, str(shotsMade), (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, BLUE, 2)

        # put text who made the shot
        if playerWhoShot is not None:
            cv2.putText(frameToDraw, "player who shot: ", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, BLUE, 2)
            cv2.putText(frameToDraw, str(playerWhoShot), (450, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, BLUE, 2)
        
        
            

        cv2.imshow("frame", frameToDraw)
        cv2.imshow("topview", topviewImageCpy)
        key = cv2.waitKey(1)
        if key == 27:
            break

    video.release()
    cv2.destroyAllWindows()

    statisticsDict = statisticsGenerator.getStatistics()
    
    statisticsDict["firstFrame"] = None
    statisticsDict["scenePoints"] = None        # None if scene already exists, else create new scene with scenePoints
    statisticsDict["courtSide"] = None
    statisticsDict["rimPoints"] = None

    if args.get("scenePointsPath") is None:
        statisticsDict["firstFrame"] = firstFrame
        statisticsDict["scenePoints"] = twTransform.getSceneIntersections().tolist()
        statisticsDict["courtSide"] = args.get("courtSide")
        statisticsDict["rimPoints"] = rimPoints.tolist()

    storeStatistics(statisticsDict)

if __name__ == '__main__':
    argsDict = {
        "videoPath": VIDEO_PATH,
        "team1Path": TEAM_1_PLAYER,
        "team2Path": TEAM_2_PLAYER,
        "courtSide": "right",
        "scenePointsPath": None     # Path of folder containing scenePoints.json
    }
    
    executeStatisticsGeneration(args=argsDict)