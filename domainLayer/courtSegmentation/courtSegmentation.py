import numpy as np
import cv2
import matplotlib.pyplot as plt
from domainLayer.topviewTransform import topview as tv


def imreconstruct(marker, mask):
    # Kernel has to be odd and square in order to reconstruct to all directions
    kernel = np.ones((3, 3), dtype=np.uint8)
    while True:
        expanded = cv2.dilate(src=marker, kernel=kernel)
        cv2.bitwise_and(src1=expanded, src2=mask, dst=expanded)

        if (marker == expanded).all():
            return expanded

        marker = expanded


def courtSegmentation(pts, imgShape):
    """p1 = pts[0]
    p2 = pts[2]
    p3 = pts[1]
    p4 = pts[3]

    center = tv.intersection_point(p1, p2, p3, p4)

    marker = np.zeros(shape=(img_shape[0], img_shape[1]), dtype=np.uint8)
    mask = np.ones(shape=(img_shape[0], img_shape[1]), dtype=np.uint8)

    marker[int(center[0])][int(center[1])] = 1

    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(mask, [pts], True, (0, 0, 0), 5)

    segmented_court = imreconstruct(marker, mask)"""

    segmentedCourt = np.zeros(imgShape, dtype=np.uint8)
    segmentedCourt = cv2.fillConvexPoly(segmentedCourt, pts, 255)
    segmentedCourt = cv2.cvtColor(segmentedCourt, cv2.COLOR_BGR2GRAY)
    segmentedCourt = cv2.threshold(segmentedCourt, 0, 255, cv2.THRESH_BINARY)[1].astype("bool")

    return segmentedCourt
