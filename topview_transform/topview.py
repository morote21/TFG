import cv2
import numpy as np


class Topview:
    def __init__(self):
        self.h_matrix = None

    def compute_topview(self, scene_points, topview_points):
        # The topview transformation is done using homography matrix
        self.h_matrix, status = cv2.findHomography(scene_points, topview_points, cv2.RANSAC, 3.0)

    def transform_point(self, point):
        x = point[0]
        y = point[1]
        p = np.array([x, y, 1])
        p = np.matmul(self.h_matrix, p)
        q = np.array([p[0] / p[2], p[1] / p[2]])
        return tuple((q[0], q[1]))

    def print_homography(self):
        print(self.h_matrix)
