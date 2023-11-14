import cv2
import numpy as np


# https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection Given two points on each line intersection
def intersectionPoint(p1, p2, p3, p4):
    # Normalize input
    points = np.array([p1, p2, p3, p4])
    minVal = np.min(points)
    maxVal = np.max(points)
    points = (points - minVal) / (maxVal - minVal)

    p1, p2, p3, p4 = points

    denom = ((p1[0] - p2[0]) * (p3[1] - p4[1])) - ((p1[1] - p2[1]) * (p3[0] - p4[0]))

    # If denom is zero, lines are parallel: no intersection. Return None.
    if denom == 0:
        return None

    pX = ((p1[0] * p2[1] - p1[1] * p2[0]) * (p3[0] - p4[0]) - (p1[0] - p2[0]) * (
            p3[0] * p4[1] - p3[1] * p4[0])) / denom

    pY = ((p1[0] * p2[1] - p1[1] * p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (
            p3[0] * p4[1] - p3[1] * p4[0])) / denom

    # Revert normalization
    pX = pX * (maxVal - minVal) + minVal
    pY = pY * (maxVal - minVal) + minVal

    return np.array([pX, pY])


def allIntersections(points):
    intersections = np.zeros(shape=(4, 2))

    intersectsIdx = 0
    for i in range(0, len(points) - 2, 2):
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[i + 2]
        p4 = points[i + 3]

        # Compute intersection
        intersectPoint = intersectionPoint(p1, p2, p3, p4)
        if intersectPoint is not None:
            intersections[intersectsIdx] = intersectPoint
            intersectsIdx += 1

    # Last line and the first one again
    p1 = points[0]
    p2 = points[1]

    nPoints = len(points)
    p3 = points[nPoints - 2]
    p4 = points[nPoints - 1]

    intersectPoint = intersectionPoint(p1, p2, p3, p4)
    if intersectPoint is not None:
        intersections[intersectsIdx] = intersectPoint

    return intersections


class Topview:
    def __init__(self):
        self.h = None
        self.sceneIntersections = np.zeros(shape=(4, 2))
        self.topviewIntersections = np.zeros(shape=(4, 2))

        self.players_positions = None

    def computeTopview(self, scenePoints, topviewPoints):
        # The topview transformation is done using homography matrix
        self.sceneIntersections = allIntersections(scenePoints)
        self.topviewIntersections = allIntersections(topviewPoints)

        self.h, st = cv2.findHomography(self.sceneIntersections, self.topviewIntersections, cv2.RANSAC, 3.0)

    def transformPoint(self, point):
        x = point[0]
        y = point[1]
        p = np.array([x, y, 1])
        p = np.matmul(self.h, p)
        q = np.array([p[0] / p[2], p[1] / p[2]])
        return tuple((q[0], q[1]))

    def printHomography(self):
        print(self.h)

    def getSceneIntersections(self):
        return self.sceneIntersections
