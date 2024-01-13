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

TOPVIEW_PATH = "database/topview/topview_image.jpg"
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

    topviewCleanCpy = copy.deepcopy(topviewImg)
    actionAnalyzer = ActionAnalysis(topviewImage=topviewImg)
    statisticsGenerator = StatisticsGenerator(topviewImage=topviewCleanCpy)

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
    madeShotDetector = ShotMadeDetector(rimPoints)                      # INITIALIZE MADE SHOT DETECTOR OBJECT

    prevActionsClassifications = {}                                     # DICTIONARY TO STORE THE PREVIOUS CLASSIFICATIONS OF EACH PLAYER

    shotsMade = 0

    lastPlayerWithBall = None                                           # ID OF THE LAST PLAYER WHO HAS HAD THE BALL
    playerWhoShot = None                                                # ID OF THE PLAYER WHO HAS SHOT THE BALL
    posOfShot = None                                                    # POSITION OF THE SHOT IN THE TOPVIEW                
    shotEnded = False                                                   # BOOLEAN TO KNOW IF THE SHOT HAS ENDED (TRUE ONLY IN ONE FRAME WHEN THE MADE SHOT DETECTION HAS ENDED)

    processingShot = False                                              # BOOLEAN TO KNOW IF THE SHOT IS BEING PROCESSED, THAT IS, IF THE BALL IS OVER THE RIM OR IN THE BACKBOARD

    lastBallCenter = None                                               # LAST POSITION WHERE THE BALL HAS BEEN DETECTED


    while True:
        ret, frame = video.read()

        if not ret:
            break

        frame = utils.resizeFrame(frame, height=1080)                   # RESIZE FRAME TO 1080p
        #frame = preprocessFrame(frame)                                  # PREPROCESS FRAME 

        frameToDraw = copy.deepcopy(frame)

        boxes, ids, classes = playerTracker.trackPlayers(frame=frame)               # TRACK PLAYERS IN FRAME

        #teams.fitPlayers(boxes, frame)                                                # FIT PLAYERS TO CREATE TEAMS

        actions = actionRecognizer.inference(frame, boxes, ids, classes)            # PERFORM ACTION RECOGNITION

        ballCenter, ballSize = madeShotDetector.whereBall(frame, frameToDraw)       # DETECT BALL IN FRAME

        playerWithBall = None                                                       # GET PLAYER WHO HAS THE BALL, IF ANY
        if ballSize is not None:
            playerWithBall = utils.whoHasPossession(zip(ids, boxes), ballSize)
            if (playerWithBall != playerWhoShot) and not processingShot:
                playerWhoShot = None
                posOfShot = None
                shotEnded = False

        if ballCenter is not None:                                                  # LAST PLACE WHERE BALL IS DETECTED
            lastBallCenter = ballCenter


        
        # if the ball is over the rim or in the backboard, start processing the shot
        if not processingShot and lastBallCenter is not None:
            processingShot = madeShotDetector.isBallOverRim(lastBallCenter) or madeShotDetector.isBallInBackboard(frame)

        # if the ball has a minimum of height, but no one has shot, then stop processing the shot
        if processingShot and playerWhoShot is None:                                
            processingShot = False
            shotEnded = False
            posOfShot = None 
        
        # if a player has shot the ball, and the shot has a minimum of height, then start detecting the result
        if playerWhoShot is not None and processingShot:
            made, ballSize, shotEnded = madeShotDetector.getShotResult(frame, frameToDraw, ballCenter, ballSize)

        # store last player with possession detected 
        if playerWithBall is not None:
            lastPlayerWithBall = playerWithBall

        # if the shot has been processed, and the shot has ended, and someone has shot, then analyze and store the shot
        if processingShot and shotEnded and playerWhoShot is not None:
            shotValue = actionAnalyzer.shotDetected(posOfShot)
            statisticsGenerator.storeShot(posOfShot, association, shotValue, made)

            if made:
                cv2.circle(topviewImg, (int(posOfShot[0]), int(posOfShot[1])), 3, (0, 121, 4), 2)
                shotsMade += 1
            else:
                cv2.line(topviewImg, (int(posOfShot[0])-5, int(posOfShot[1])-5), (int(posOfShot[0])+5, int(posOfShot[1])+5), (0, 0, 255), 2)
                cv2.line(topviewImg, (int(posOfShot[0])+5, int(posOfShot[1])-5), (int(posOfShot[0])-5, int(posOfShot[1])+5), (0, 0, 255), 2)

            playerWhoShot = None
            posOfShot = None
            processingShot = False
            shotEnded = False

            cv2.destroyWindow("backboardCrop")


        # for each player detected
        for box, identity, cls in zip(boxes, ids, classes):             # DRAW BOUNDING BOXES WITH ID AND ACTION
                
            crop = frame[box[1]:box[3], box[0]:box[2]]                  # CROP PLAYER FROM FRAME FOR TEAM ASSOCIATION
            association = -1
            if teams is not None:
                association = teams.associate(crop)                     # ASSOCIATE PLAYER WITH A TEAM
            
            frameToDraw = drawBoundingBoxPlayer(frameToDraw, box, identity, segmentedCourt, association, actions[identity], playerWithBall) 

            floorPoint = ((box[0] + box[2]) / 2, box[3])
            floorPointTransformed = twTransform.transformPoint(floorPoint)
            statisticsGenerator.storeStep((int(floorPointTransformed[0]), int(floorPointTransformed[1])), association)  # STORE STEP IN HEATMAP

            # if actions[identity] == "shoot" and prevActionsClassifications[identity] == "ball in hand":
            #     pos = (int(floorPointTransformed[0]), int(floorPointTransformed[1]))
            #     shotValue = actionAnalyzer.shotDetected(pos)
            #     statisticsGenerator.storeShot(pos, association, shotValue)

            if actions[identity] == "shoot" and lastPlayerWithBall == identity and playerWhoShot is None:   # DETECT IF PLAYER HAS SHOT
                playerWhoShot = identity    
                posOfShot = (int(floorPointTransformed[0]), int(floorPointTransformed[1]))
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

