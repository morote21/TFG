import cv2
import sys
import copy
import numpy as np
import torch
from topviewTransform.topview import Topview
from ultralytics import YOLO
from courtSegmentation import courtSegmentation as cs
from personDetection.personDetection import Tracker, drawBoundingBoxPlayer
from playerRecognition.teamAssociation import Teams
import actionRecognition.actionRecognition as ar
import utils

# VIDEO_PATH = "/home/morote/Desktop/input_tfg/2000_0226_194537_003.MP4"
VIDEO_PATH = "/home/morote/Desktop/input_tfg/nba2k_test.mp4"
TOPVIEW_PATH = "/home/morote/Desktop/input_tfg/synthetic_court2.jpg"
TEAM_1_PLAYER = "/home/morote/Pictures/team1_black.png"
TEAM_2_PLAYER = "/home/morote/Pictures/team2_white.png"




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














def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    topviewImg = cv2.imread(TOPVIEW_PATH)

    team1Img = cv2.imread(TEAM_1_PLAYER)
    team2Img = cv2.imread(TEAM_2_PLAYER)

    teams = Teams(team1Img, team2Img, 6)                                # CREATE TEAMS OBJECT, WHICH CONTAINS DATA ABOUT BOTH TEAMS 

    video = cv2.VideoCapture(VIDEO_PATH)

    ret, frame = video.read()                                           # READ FIRST FRAME FOR TOPVIEW TRANSFORM COMPUTATION
    frame = utils.resizeFrame(frame, height=1080)

    if not ret:
        print("Error reading video frame.\n")

    sceneCpy = copy.deepcopy(frame)
    topviewCpy = copy.deepcopy(topviewImg)

    # GET BORDERS OF THE COURT AND THE TOPVIEW IMAGE
    scenePoints = utils.getBorders(sceneCpy)
    topviewPoints = utils.getBorders(topviewCpy)

    twTransform = Topview()
    twTransform.computeTopview(scenePoints, topviewPoints)              # COMPUTE TOPVIEW MATRIX

    pts = np.array(twTransform.getSceneIntersections(), np.int32)

    segmentedCourt = cs.courtSegmentation(pts, frame.shape)
    segmentedCourt = segmentedCourt > 0                                 # MASK TO FILTER PEOPLE INSIDE THE COURT

    playerTracker = Tracker()
    actionRecognizer = ar.ActionRecognition()

    playersFrames = {}

    while True:
        ret, frame = video.read()

        if not ret:
            break

        frame = utils.resizeFrame(frame, height=1080)
        boxes, ids, classes = playerTracker.trackPlayers(frame=frame)

        if len(ids) != len(playersFrames.keys()):
            if len(playersFrames.keys()) < len(ids):
                for iden in ids:
                    if iden not in list(playersFrames.keys()):
                        playersFrames[iden] = []
            
            else:
                for iden in list(playersFrames.keys()):
                    if iden not in ids:
                        playersFrames.pop(iden)

        for box, identity, cls in zip(boxes, ids, classes):
            # print(playerTracker.getClassName(cls))
            hasAction = False
            if playerTracker.getClassName(cls) == "person":

                crop = frame[box[1]-20:box[3]+20, box[0]-20:box[2]+20]
                cropAndResize = ar.cropPlayer(frame, box)
                association = teams.associate(crop)
                
                for ide, frames in playersFrames.items():
                    print(f"Player {ide} has {len(frames)} frames")

                # # QUEUE OF 16 FRAMES
                playersFrames[identity].append(cropAndResize)
                if len(playersFrames[identity]) > 16:
                     playersFrames[identity].pop(0)

                # INFERENCE
                if len(playersFrames[identity]) == 16:
                    inputFrames = ar.inferenceShape(torch.Tensor(playersFrames[identity]))
                    inputFrames = inputFrames.to(device=device)

                    with torch.no_grad():
                        output = actionRecognizer.inference(inputFrames)
                        action = actionRecognizer.getLabel(output[0])
                        print(action)
                        hasAction = True
                        #_, pred = torch.max(output, 1)
                        #print(pred)

                if hasAction:
                    frame = drawBoundingBoxPlayer(frame, box, identity, segmentedCourt, association, action)
                else:
                    frame = drawBoundingBoxPlayer(frame, box, identity, segmentedCourt, association, "undefined")

        topviewImageCpy = topviewImg.copy()

        for box in boxes:
            floorPoint = ((box[0] + box[2]) / 2, box[3])
            floorPointTransformed = twTransform.transformPoint(floorPoint)
            cv2.circle(topviewImageCpy, (int(floorPointTransformed[0]), int(floorPointTransformed[1])), 3,
                       (0, 255, 0), 2)

        cv2.imshow("frame", frame)
        cv2.imshow("topview", topviewImageCpy)
        key = cv2.waitKey(1)

    video.release()
    cv2.destroyAllWindows()
        

if __name__ == '__main__':
    main()