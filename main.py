import cv2
import sys
import copy
import numpy as np
from input_data.coordinate_store import CoordinateStore
from topview_transform.topview import Topview
from ultralytics import YOLO
from court_segmentation import court_segmentation as cs
from person_detection.person_detection import Tracker, draw_bb_player
from player_recognition.team_association import Teams

VIDEO_PATH = "/home/morote/Desktop/input_tfg/nba2k_test.mp4"
TOPVIEW_PATH = "/home/morote/Desktop/input_tfg/synthetic_court2.jpg"
TEAM_1_PLAYER = "/home/morote/Pictures/team1_black.png"
TEAM_2_PLAYER = "/home/morote/Pictures/team2_white.png"



def main():

    video_path = VIDEO_PATH

    topview_path = TOPVIEW_PATH
    topview_image = cv2.imread(topview_path)

    team1_image = cv2.imread(TEAM_1_PLAYER)
    team2_image = cv2.imread(TEAM_2_PLAYER)

    # CREATE TEAMS OBJECT, WHICH CONTAINS DATA ABOUT BOTH TEAMS
    teams = Teams(team1_image, team2_image, 6)

    print('Select 6 points in the same order on both images:')
    video = cv2.VideoCapture(video_path)

    # READ FIRST FRAME FOR TOPVIEW TRANSFORM COMPUTATION
    ret, frame = video.read()

    if not ret:
        print("Error reading video frame.\n")

    # GET POINT CORRELATIONS BETWEEN SCENE AND TOPVIEW
    scene_copy = copy.deepcopy(frame)
    topview_copy = copy.deepcopy(topview_image)

    cc = CoordinateStore()
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    param = [0, scene_copy, topview_copy]
    cv2.setMouseCallback('image', cc.select_point, param)

    # SELECT 4 VECTORS IN EACH IMAGE IN THE SAME ORDER
    while (1):

        if (param[0] == 0):
            cv2.imshow('image', scene_copy)
        elif (param[0] == 1):
            cv2.imshow('image', topview_copy)
        else:
            cv2.destroyAllWindows()
            break

        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

    scene_points = cc.get_points_scene()
    topview_points = cc.get_points_topview()

    tw_transform = Topview()
    tw_transform.compute_topview(scene_points, topview_points)

    tw_transform.print_homography()

    pts = np.array(tw_transform.get_scene_intersections(), np.int32)

    segmented_court = cs.court_segmentation(pts, frame.shape)
    # Definitive mask for knowing if someone is inside the court
    segmented_court = segmented_court > 0

    #model = YOLO("./runs/detect/train/weights/best.pt")

    player_tracker = Tracker()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        boxes, ids = player_tracker.track_players(frame=frame)
        for box, identity in zip(boxes, ids):
            crop = frame[box[1]:box[3], box[0]:box[2]]
            association = teams.associate(crop)
            frame = draw_bb_player(frame, box, identity, segmented_court, association)

        topview_image_copy = topview_image.copy()

        for box in boxes:
            floor_point = ((box[0] + box[2]) / 2, box[3])
            floor_point_transformed = tw_transform.transform_point(floor_point)
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