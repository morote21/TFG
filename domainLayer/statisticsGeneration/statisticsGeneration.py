import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

# d^2 = (x2 - x1)^2 + (y2 - y1)^2
def pixelInsideCircle(shape, x, y, c, r):
    if x < 0 or x >= shape[1] or y < 0 or y >= shape[0]:
        return False
    
    return (x - c[0])**2 + (y - c[1])**2 <= r**2

def generateHeatMap(tracks, topview):
    blurred = cv2.GaussianBlur(tracks, (5, 5), 0)
    matrixNormalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    heatmap = cv2.applyColorMap(matrixNormalized, cv2.COLORMAP_JET)
    heatmap = cv2.addWeighted(heatmap, 0.5, topview, 0.5, 0)
    return heatmap

def generateShotTrack(tracks, topview):
    
    topviewCopy = copy.deepcopy(topview)
    # draw an x for each shot
    for y in range(tracks.shape[0]):
        for x in range(tracks.shape[1]):
            if tracks[y][x].any() > 0:
                cv2.line(topviewCopy, (x-5, y-5), (x+5, y+5), (0, 0, 255), 2)
                cv2.line(topviewCopy, (x+5, y-5), (x-5, y+5), (0, 0, 255), 2)
    
    return topviewCopy


class StatisticsGenerator:
    def __init__(self, topviewImage):
        self.nTeams = -1
        self.topview = topviewImage
        self.movementHeatmapTeam1 = np.zeros(self.topview.shape)
        self.movementHeatmapTeam2 = np.zeros(self.topview.shape)
        self.shotTrackTeam1 = np.zeros(self.topview.shape)
        self.shotTrackTeam2 = np.zeros(self.topview.shape)

        self.FGAteam1 = 0        # FIELD GOAL ATTEMPTS
        self.FGAteam2 = 0        # FIELD GOAL ATTEMPTS
        self.threePAteam1 = 0    # THREE POINT ATTEMPTS
        self.threePAteam2 = 0    # THREE POINT ATTEMPTS
    

    def storeStep(self, pos, team):
        radius = 8
        for y in range(pos[1]-radius, pos[1]+radius):
            for x in range(pos[0]-radius, pos[0]+radius):
                if pixelInsideCircle(self.topview.shape, x, y, pos, radius):
                    if team == 0 or team == -1:
                        self.movementHeatmapTeam1[y][x] += 5
                    else:
                        self.movementHeatmapTeam2[y][x] += 5


    def storeShot(self, pos, team, value):
        if team == 0 or team == -1:
            self.shotTrackTeam1[pos[1]][pos[0]] += 1
            if value == 2:
                self.FGAteam1 += 1
            else:
                self.threePAteam1 += 1
                
        else:
            self.shotTrackTeam2[pos[1]][pos[0]] += 1
            if value == 2:
                self.FGAteam2 += 1
            else:
                self.threePAteam2 += 1

    
    def printHeatmaps(self):
        motionHeatmap1 = generateHeatMap(self.movementHeatmapTeam1, self.topview)
        motionHeatmap2 = generateHeatMap(self.movementHeatmapTeam2, self.topview)
        shotHeatmap1 = generateHeatMap(self.shotTrackTeam1, self.topview)
        shotHeatmap2 = generateHeatMap(self.shotTrackTeam2, self.topview)

        cv2.imshow("motion heatmap team 1", motionHeatmap1)
        cv2.imshow("motion heatmap team 2", motionHeatmap2)
        cv2.imshow("shot heatmap team 1", shotHeatmap1)
        cv2.imshow("shot heatmap team 2", shotHeatmap2)
        cv2.waitKey(0)
    
    def getStatistics(self):
        if self.nTeams == -1:
            print("ERROR: not number of teams setted")
            return None
        
        elif self.nTeams == 0:
            statisticsDict = {
                "Team1" : {
                    "FGA" : self.FGAteam1,
                    "3PA" : self.threePAteam1,
                    "MotionHeatmap" : generateHeatMap(self.movementHeatmapTeam1, self.topview),
                    "ShotHeatmap" : generateHeatMap(self.shotTrackTeam1, self.topview),
                    "ShotTrack" : generateShotTrack(self.shotTrackTeam1, self.topview)
                }
            }

        elif self.nTeams == 2:      
            statisticsDict = {
                "Team1" : {
                    "FGA" : self.FGAteam1,
                    "3PA" : self.threePAteam1,
                    "MotionHeatmap" : generateHeatMap(self.movementHeatmapTeam1, self.topview),
                    "ShotHeatmap" : generateHeatMap(self.shotTrackTeam1, self.topview),
                    "ShotTrack" : generateShotTrack(self.shotTrackTeam1, self.topview)
                },
                "Team2" : {
                    "FGA" : self.FGAteam2,
                    "3PA" : self.threePAteam2,
                    "MotionHeatmap" : generateHeatMap(self.movementHeatmapTeam2, self.topview),
                    "ShotHeatmap" : generateHeatMap(self.shotTrackTeam2, self.topview),
                    "ShotTrack" : generateShotTrack(self.shotTrackTeam2, self.topview)
                }
            }

        return statisticsDict


    def setNTeams(self, nTeams):
        self.nTeams = nTeams