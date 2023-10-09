import cv2
import numpy as np


# https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection Given two points on each line intersection
def intersection_point(p1, p2, p3, p4):
    # Normalize input
    points = np.array([p1, p2, p3, p4])
    min_val = np.min(points)
    max_val = np.max(points)
    points = (points - min_val) / (max_val - min_val)

    p1, p2, p3, p4 = points

    denom = ((p1[0] - p2[0]) * (p3[1] - p4[1])) - ((p1[1] - p2[1]) * (p3[0] - p4[0]))

    # If denom is zero, lines are parallel: no intersection. Return None.
    if denom == 0:
        return None

    p_x = ((p1[0] * p2[1] - p1[1] * p2[0]) * (p3[0] - p4[0]) - (p1[0] - p2[0]) * (
            p3[0] * p4[1] - p3[1] * p4[0])) / denom

    p_y = ((p1[0] * p2[1] - p1[1] * p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (
            p3[0] * p4[1] - p3[1] * p4[0])) / denom

    # Revert normalization
    p_x = p_x * (max_val - min_val) + min_val
    p_y = p_y * (max_val - min_val) + min_val

    return np.array([p_x, p_y])


def all_intersections(points):
    intersections = np.zeros(shape=(4, 2))

    iter_intersects = 0
    for i in range(0, len(points) - 2, 2):
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[i + 2]
        p4 = points[i + 3]

        # Compute intersection
        intersect_point = intersection_point(p1, p2, p3, p4)
        if intersect_point is not None:
            intersections[iter_intersects] = intersect_point
            iter_intersects += 1

    # Last line and the first one again
    p1 = points[0]
    p2 = points[1]

    n_points = len(points)
    p3 = points[n_points - 2]
    p4 = points[n_points - 1]

    intersect_point = intersection_point(p1, p2, p3, p4)
    if intersect_point is not None:
        intersections[iter_intersects] = intersect_point

    return intersections


class Topview:
    def __init__(self):
        self.h_matrix = None
        self.scene_intersections = np.zeros(shape=(4, 2))
        self.topview_intersections = np.zeros(shape=(4, 2))

        self.players_positions = None

    def compute_topview(self, scene_points, topview_points):
        # The topview transformation is done using homography matrix

        scene_intersections = np.zeros(shape=(4, 2))
        tw_intersections = np.zeros(shape=(4, 2))

        self.scene_intersections = all_intersections(scene_points)
        self.topview_intersections = all_intersections(topview_points)

        self.h_matrix, st = cv2.findHomography(self.scene_intersections, self.topview_intersections, cv2.RANSAC, 3.0)

    def transform_point(self, point):
        x = point[0]
        y = point[1]
        p = np.array([x, y, 1])
        p = np.matmul(self.h_matrix, p)
        q = np.array([p[0] / p[2], p[1] / p[2]])
        return tuple((q[0], q[1]))

    def print_homography(self):
        print(self.h_matrix)

    def get_scene_intersections(self):
        return self.scene_intersections
