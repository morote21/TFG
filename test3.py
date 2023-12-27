from ultralytics import YOLO
import numpy as np
import cv2
import copy
from domainLayer import utils


model = YOLO("/home/morote/Desktop/TFG/domainLayer/models/yolov8m-pose.pt")
VIDEO_PATH = "/home/morote/Desktop/input_tfg/20231215_131239_Trim.mp4"
IMAGE_PATH = "/home/morote/Pictures/equipo_test_8.png"
# video = cv2.VideoCapture(VIDEO_PATH)
# fps = video.get(cv2.CAP_PROP_FPS)


# while video.isOpened():
#     ret, frame = video.read()
#     frame = utils.resizeFrame(frame, height=1080)

#     results = model.predict(frame, show=True)

#     cv2.waitKey(0)


image = cv2.imread(IMAGE_PATH)
image = utils.resizeFrame(image, height=600)
result = model.predict(image, show=True)

keypoints = result[0].keypoints.xy.cpu().numpy()[0]
ls = keypoints[5]   # left shoulder
rs = keypoints[6]   # right shoulder
lh = keypoints[11]  # left hip
rh = keypoints[12]  # right hip

# crop image by left-most shoulder and right-most hip
ls_x = int(ls[0])
rs_x = int(rs[0])
lh_x = int(lh[0])
rh_x = int(rh[0])

ls_y = int(ls[1])
rh_y = int(rh[1])

if ls_x < rs_x:
    left_torax = ls_x
else:
    left_torax = rs_x

if lh_x > rh_x:
    right_hip = lh_x
    right_torax = lh_x
    left_hip = rh_x
else:
    right_torax = rh_x
    right_hip = rh_x
    left_hip = lh_x

# crop image by 25% above and 40% below
height = image.shape[0]
width = image.shape[1]

roiTorax = image[ls_y:rh_y, left_torax:right_torax]
cv2.imshow("torax", roiTorax)

roiHip = image[int(rh_y - height*0.10):int(rh_y + height*0.10), int(left_hip - width*0.10):int(right_hip + width*0.10)]
cv2.imshow("hip", roiHip)

cv2.waitKey(0)