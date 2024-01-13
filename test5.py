from ultralytics import YOLO
import numpy as np
import cv2
import copy
from domainLayer import utils

PATH_IMAGE1 = "/home/morote/Desktop/TFG/database/games/game24/Team1ShotTrack.png"
PATH_IMAGE2 = "/home/morote/Desktop/TFG/database/games/game25/Team1ShotTrack.png"
PATH_IMAGE3 = "/home/morote/Desktop/TFG/database/games/game26/Team1ShotTrack.png"
PATH_IMAGE4 = "/home/morote/Desktop/TFG/database/games/game27/Team1ShotTrack.png"

PATH_GT_IMAGE1 = "/home/morote/Downloads/gt_scn1_4k_30fps_noteams_1_clip1.png"
PATH_GT_IMAGE2 = "/home/morote/Downloads/gt_scn1_4k_30fps_noteams_1_clip2.png"
PATH_GT_IMAGE3 = "/home/morote/Downloads/gt_scn1_4k_30fps_noteams_1_clip3.png"
PATH_GT_IMAGE4 = "/home/morote/Downloads/gt_scn1_4k_30fps_noteams_1_clip4.png"


R = [0, 0, 255]
G = [0, 121, 4]
W = [255, 255, 255]

R_GT = [36, 28, 237]
G_GT = [76, 177, 34]

def superpose_images(image1, image2):
    superposed_image = copy.deepcopy(image1)
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            if (image2[i][j][:] != W).any():
                if (image2[i][j][:] == R).all():
                    superposed_image[i][j][:] = R
                elif (image2[i][j][:] == G).all():
                    superposed_image[i][j][:] = G
    
    return superposed_image

def superpose_gt_images(image1, image2):
    superposed_image = copy.deepcopy(image1)
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            if (image2[i][j][:] != W).any():
                if (image2[i][j][:] == R_GT).all():
                    superposed_image[i][j][:] = R_GT
                elif (image2[i][j][:] == G_GT).all():
                    superposed_image[i][j][:] = G_GT
    
    return superposed_image


image1 = cv2.imread(PATH_IMAGE1)
image2 = cv2.imread(PATH_IMAGE2)
image3 = cv2.imread(PATH_IMAGE3)
image4 = cv2.imread(PATH_IMAGE4)

image_superposed_1 = superpose_images(image1, image2)
image_superposed_2 = superpose_images(image3, image4)

image_superposed = superpose_images(image_superposed_1, image_superposed_2)

image_gt_1 = cv2.imread(PATH_GT_IMAGE1)
image_gt_2 = cv2.imread(PATH_GT_IMAGE2)
image_gt_3 = cv2.imread(PATH_GT_IMAGE3)
image_gt_4 = cv2.imread(PATH_GT_IMAGE4)

image_gt_superposed_1 = superpose_gt_images(image_gt_1, image_gt_2)
image_gt_superposed_2 = superpose_gt_images(image_gt_3, image_gt_4)

image_gt_superposed = superpose_gt_images(image_gt_superposed_1, image_gt_superposed_2)

cv2.imshow("image_superposed", image_superposed)
cv2.imshow("image_gt_superposed", image_gt_superposed)
cv2.waitKey(0)

