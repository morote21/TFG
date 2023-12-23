import cv2
import sys
import copy
import numpy as np

IMAGE_PATH = "/home/morote/Pictures/prueba24to8bit.png"

def colordepth24bit_to_8bit(img):
    B_2msb = np.right_shift(img, 16)  # 2 bit size
    G_3msb = np.right_shift(img, 8)

    B_2msb = np.bitwise_and(B_2msb, 0xff)  # 2 bit size
    G_3msb = np.bitwise_and(G_3msb, 0xff)
    R_3msb = np.bitwise_and(img, 0xff)

    B_2msb = np.right_shift(B_2msb, 6)  # 2 bit size
    G_3msb = np.right_shift(G_3msb, 5)
    R_3msb = np.right_shift(R_3msb, 5)

    B_2msb = np.left_shift(B_2msb, 6)  # 8 bit size
    G_3msb = np.left_shift(G_3msb, 3)

    BGR_8bit = np.bitwise_or(B_2msb, G_3msb)
    BGR_8bit = np.bitwise_or(BGR_8bit, R_3msb)

    BGR_8bit = BGR_8bit.astype(np.uint8)
    BGR_8bit = BGR_8bit * 32

    return BGR_8bit



# read image
img = cv2.imread(IMAGE_PATH)

cv2.imshow("BGR_8bit", colordepth24bit_to_8bit(img))

cv2.waitKey(0)
