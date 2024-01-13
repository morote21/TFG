import numpy as np
import cv2

def courtSegmentation(pts, imgShape):
    """
    Segments the court from the image
    :param pts: points of the court (np.array)
    :param imgShape: shape of the image (tuple)
    :return: segmented court (np.array)
    """

    segmentedCourt = np.zeros(imgShape, dtype=np.uint8)
    segmentedCourt = cv2.fillConvexPoly(segmentedCourt, pts, 255)
    segmentedCourt = cv2.cvtColor(segmentedCourt, cv2.COLOR_BGR2GRAY)
    segmentedCourt = cv2.threshold(segmentedCourt, 0, 255, cv2.THRESH_BINARY)[1].astype("bool")

    return segmentedCourt
