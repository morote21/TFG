import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

# d^2 = (x2 - x1)^2 + (y2 - y1)^2
def pixelInsideCircle(shape, x, y, c, r):
    """
    Returns true if the pixel is inside the circle
    :param shape: shape of the image (tuple)
    :param x: x coordinate of the pixel (int)
    :param y: y coordinate of the pixel (int)
    :param c: center of the circle (tuple)
    :param r: radius of the circle (int)
    :return: true if the pixel is inside the circle (bool)
    """
    if x < 0 or x >= shape[1] or y < 0 or y >= shape[0]:
        return False
    
    return (x - c[0])**2 + (y - c[1])**2 <= r**2

def generateHeatMap(tracks, topview):
    """
    Generates a heatmap from the tracks
    :param tracks: tracks of the players (np.array)
    :param topview: topview image (np.array)
    :return: heatmap (np.array)
    """
    blurred = cv2.GaussianBlur(tracks, (5, 5), 0)
    matrixNormalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    heatmap = cv2.applyColorMap(matrixNormalized, cv2.COLORMAP_JET)
    heatmap = cv2.addWeighted(heatmap, 0.5, topview, 0.5, 0)
    return heatmap

def generateShotTrack(tracks, topview):
    """
    Generates a shot track map from the tracks
    :param tracks: tracks of the players (np.array)
    :param topview: topview image (np.array)
    :return: shot track (np.array)
    """
    topviewCopy = copy.deepcopy(topview)
    # draw an x for each shot
    for y in range(tracks.shape[0]):
        for x in range(tracks.shape[1]):
            if tracks[y][x] < 0:
                cv2.line(topviewCopy, (x-5, y-5), (x+5, y+5), (0, 0, 255), 2)
                cv2.line(topviewCopy, (x+5, y-5), (x-5, y+5), (0, 0, 255), 2)
            elif tracks[y][x] > 0:
                cv2.circle(topviewCopy, (x, y), 5, (0, 121, 4), 2)
    
    return topviewCopy


class StatisticsGenerator:
    def __init__(self, topviewImage):
        self.nTeams = -1
        self.topview = topviewImage
        self.movementHeatmapTeam1 = np.zeros(self.topview.shape)
        self.movementHeatmapTeam2 = np.zeros(self.topview.shape)
        self.shotTrackTeam1 = np.zeros((self.topview.shape[0], self.topview.shape[1]))
        self.shotTrackTeam2 = np.zeros((self.topview.shape[0], self.topview.shape[1]))

        self.FGAteam1 = 0        # FIELD GOAL ATTEMPTS
        self.FGAteam2 = 0        # FIELD GOAL ATTEMPTS
        self.threePAteam1 = 0    # THREE POINT ATTEMPTS
        self.threePAteam2 = 0    # THREE POINT ATTEMPTS

        self.FGMteam1 = 0        # FIELD GOAL MADE
        self.FGMteam2 = 0        # FIELD GOAL MADE
        self.threePMteam1 = 0    # THREE POINT MADE
        self.threePMteam2 = 0    # THREE POINT MADE
    

    def storeStep(self, pos, team):
        """
        Stores a step in the heatmap
        :param pos: position of the player (tuple)
        :param team: team of the player (int)
        :return: None
        """
        radius = 8
        for y in range(pos[1]-radius, pos[1]+radius):
            for x in range(pos[0]-radius, pos[0]+radius):
                if pixelInsideCircle(self.topview.shape, x, y, pos, radius):
                    if team == 0 or team == -1:
                        self.movementHeatmapTeam1[y][x] += 5
                    else:
                        self.movementHeatmapTeam2[y][x] += 5


    def storeShot(self, pos, team, value, made):
        """
        Stores a shot in the heatmap
        :param pos: position of the player (tuple)
        :param team: team of the player (int)
        :param value: value of the shot (int)
        :param made: true if the shot has been made (bool)
        :return: None
        """
        if team == 0 or team == -1:
            if made:
                self.shotTrackTeam1[pos[1]][pos[0]] = 1
            else:
                self.shotTrackTeam1[pos[1]][pos[0]] = -1

            if value == 2:
                self.FGAteam1 += 1
                if made:
                    self.FGMteam1 += 1
                
            else:
                self.threePAteam1 += 1
                self.FGAteam1 += 1
                if made:
                    self.threePMteam1 += 1
                    self.FGMteam1 += 1
                
                
        else:
            if made:
                self.shotTrackTeam2[pos[1]][pos[0]] = 1
            else:
                self.shotTrackTeam2[pos[1]][pos[0]] = -1

            if value == 2:
                self.FGAteam2 += 1
                if made:
                    self.FGMteam2 += 1
                
            else:
                self.threePAteam2 += 1
                self.FGAteam2 += 1
                if made:
                    self.threePMteam2 += 1
                    self.FGMteam2 += 1
                

    
    def getStatistics(self):
        """
        Returns the statistics
        :return: statistics (dictionary)
        """
        if self.nTeams == -1:
            print("ERROR: not number of teams setted")
            return None
        
        elif self.nTeams == 0:
            statisticsDict = {
                "Team1" : {
                    "FGA" : self.FGAteam1,
                    "FGM" : self.FGMteam1,
                    "3PA" : self.threePAteam1,
                    "3PM" : self.threePMteam1,
                    "MotionHeatmap" : generateHeatMap(self.movementHeatmapTeam1, self.topview),
                    "ShotHeatmap" : generateHeatMap(self.shotTrackTeam1, self.topview),
                    "ShotTrack" : generateShotTrack(self.shotTrackTeam1, self.topview)
                }
            }

        elif self.nTeams == 2:      
            statisticsDict = {
                "Team1" : {
                    "FGA" : self.FGAteam1,
                    "FGM" : self.FGMteam1,
                    "3PA" : self.threePAteam1,
                    "3PM" : self.threePMteam1,
                    "MotionHeatmap" : generateHeatMap(self.movementHeatmapTeam1, self.topview),
                    "ShotHeatmap" : generateHeatMap(self.shotTrackTeam1, self.topview),
                    "ShotTrack" : generateShotTrack(self.shotTrackTeam1, self.topview)
                },
                "Team2" : {
                    "FGA" : self.FGAteam2,
                    "FGM" : self.FGMteam2,
                    "3PA" : self.threePAteam2,
                    "3PM" : self.threePMteam2,
                    "MotionHeatmap" : generateHeatMap(self.movementHeatmapTeam2, self.topview),
                    "ShotHeatmap" : generateHeatMap(self.shotTrackTeam2, self.topview),
                    "ShotTrack" : generateShotTrack(self.shotTrackTeam2, self.topview)
                }
            }

        return statisticsDict


    def setNTeams(self, nTeams):
        """
        Sets the number of teams
        :param nTeams: number of teams (int)
        :return: None
        """
        self.nTeams = nTeams