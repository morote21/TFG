import cv2
import numpy as np

topview = cv2.imread("database/topview/topview_image.jpg")
cv2.imshow("topview", topview)
cv2.waitKey(0)