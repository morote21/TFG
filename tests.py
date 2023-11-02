# from ultralytics import YOLO

# model = YOLO("./runs/detect/train/weights/best.pt")

# results = model.track(source="/home/morote/Desktop/input_tfg/nba2k_test.mp4", show=True, conf=0.5)

import pickle
import numpy as np

# file = open("/home/morote/Desktop/dataset/examples/0000000.npy", "rb")

# l = np.load("/home/morote/Desktop/dataset/examples/0000000.npy", allow_pickle=True)
# print(l)

from domain_layer.input_data.coordinate_store import RimCoordinates
import cv2
import copy

OFFICIAL_RIM_DIAMETER = 46
OFFICIAL_BALL_DIAMETER = 24

VIDEO_PATH = "/home/morote/Desktop/input_tfg/nba2k_test.mp4"
video = cv2.VideoCapture(VIDEO_PATH)

ret, frame = video.read()
scene_copy = copy.deepcopy(frame)

rc = RimCoordinates()
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
param = [0, scene_copy]
cv2.setMouseCallback('image', rc.select_rim_diameter, param)

while 1:

    if not rc.array_filled:
        cv2.imshow('image', scene_copy)

    else:
        cv2.destroyAllWindows()
        break

    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

rim_points = rc.get_rim_coordinates()
rim_left_point = rim_points[0]
rim_right_point = rim_points[1]

