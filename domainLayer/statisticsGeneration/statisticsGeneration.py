import cv2
import numpy as np
import matplotlib.pyplot as plt

class StatisticsGenerator:
    def __init__(self):
        self.movementHeatmapTeam1 = np.zeros(self.topviewImage.shape)
        self.movementHeatmapTeam2 = np.zeros(self.topviewImage.shape)
        self.shotTrackTeam1 = np.zeros(self.topviewImage.shape)
        self.shotTrackTeam2 = np.zeros(self.topviewImage.shape)