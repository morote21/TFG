import cv2
import sys
import copy
import numpy as np
from topviewTransform.topview import Topview
from ultralytics import YOLO
from courtSegmentation import courtSegmentation as cs
from personDetection.personDetection import Tracker, draw_bb_player
from playerRecognition.teamAssociation import Teams
import utils

# VIDEO_PATH = "/home/morote/Desktop/input_tfg/2000_0226_194537_003.MP4"
VIDEO_PATH = "/home/morote/Desktop/input_tfg/nba2k_test.mp4"
TOPVIEW_PATH = "/home/morote/Desktop/input_tfg/synthetic_court2.jpg"
TEAM_1_PLAYER = "/home/morote/Pictures/team1_black.png"
TEAM_2_PLAYER = "/home/morote/Pictures/team2_white.png"



def main():

    topviewImg = cv2.imread(TOPVIEW_PATH)

    team1Img = cv2.imread(TEAM_1_PLAYER)
    team2Img = cv2.imread(TEAM_2_PLAYER)

    teams = Teams(team1Img, team2Img, 6)                                # CREATE TEAMS OBJECT, WHICH CONTAINS DATA ABOUT BOTH TEAMS 

    print('Select 6 points in the same order on both images:')
    video = cv2.VideoCapture(VIDEO_PATH)

    ret, frame = video.read()                                           # READ FIRST FRAME FOR TOPVIEW TRANSFORM COMPUTATION

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

    while True:
        ret, frame = video.read()
        if not ret:
            break

        boxes, ids, classes = playerTracker.trackPlayers(frame=frame)
        for box, identity, cls in zip(boxes, ids, classes):
            crop = frame[box[1]:box[3], box[0]:box[2]]
            association = teams.associate(crop)
            frame = draw_bb_player(frame, box, identity, segmentedCourt, association)
            print(cls)

        topview_image_copy = topviewImg.copy()

        for box in boxes:
            floor_point = ((box[0] + box[2]) / 2, box[3])
            floor_point_transformed = twTransform.transformPoint(floor_point)
            cv2.circle(topview_image_copy, (int(floor_point_transformed[0]), int(floor_point_transformed[1])), 3,
                       (0, 255, 0), 2)
        # Descomentar para hacer bien el topview
        # floor_point_transformed = tw_transform.transform_point(floor_point)
        # cv2.circle(topview_image_copy, (int(floor_point_transformed[0]), int(floor_point_transformed[1])), 3, (0, 255, 0), 2)

        cv2.imshow("frame", frame)
        cv2.imshow("topview", topview_image_copy)
        key = cv2.waitKey(1)

    video.release()
    cv2.destroyAllWindows()
        

if __name__ == '__main__':
    main()