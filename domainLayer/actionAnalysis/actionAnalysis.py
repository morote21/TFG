import cv2
import numpy as np

# d^2 = (x2 - x1)^2 + (y2 - y1)^2
def pixelInsideCircle(x, y, c, r):
    return (x - c[0])**2 + (y - c[1])**2 <= r**2



class ActionAnalysis:
    def __init__(self, topviewImage):
        self.topviewImage = topviewImage
        self.topviewImageBin = cv2.cvtColor(topviewImage, cv2.COLOR_BGR2GRAY)
        self.topviewImageBin = cv2.GaussianBlur(self.topviewImageBin, (5, 5), 0)
        self.topviewImageBin = cv2.threshold(self.topviewImageBin, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        connectedComponents = cv2.connectedComponentsWithStats(self.topviewImageBin, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = connectedComponents
        
        # sort labels by area
        labelAreas = [stats[i, cv2.CC_STAT_AREA] for i in range(0, numLabels)]
        sortedLabels = np.argsort(labelAreas)
        sortedLabels = sortedLabels[::-1]   # DESCENDING ORDER
        
        # LABELS ARE SORTED BY AREA, SO THE 1 AND 2 ARE THE THREE ZONES (1 IS THE BORDERS EXTERIOR TO THE COURT)        
        # MAKE OR OPERATION WITH SORTEDLABEL[1] AND SORTEDLABEL[2] TO GET THE TWO ZONES IN A NEW MASK
        self.zone1 = (labels == sortedLabels[1]).astype("uint8") * 255
        self.zone2 = (labels == sortedLabels[2]).astype("uint8") * 255
        self.threeZone = cv2.bitwise_or(self.zone1, self.zone2)
        self.threeZone = self.threeZone.astype("bool")

        self.movementHeatmapTeam1 = np.zeros(self.topviewImage.shape)
        self.movementHeatmapTeam2 = np.zeros(self.topviewImage.shape)
        self.shotTrackTeam1 = np.zeros(self.topviewImage.shape)
        self.shotTrackTeam2 = np.zeros(self.topviewImage.shape)

        self.FGAteam1 = 0        # FIELD GOAL ATTEMPTS
        self.FGAteam2 = 0        # FIELD GOAL ATTEMPTS
        self.threePAteam1 = 0    # THREE POINT ATTEMPTS
        self.threePAteam2 = 0    # THREE POINT ATTEMPTS

    
    def shotDetected(self, pos, team):
        if team == 0:
            self.shotTrackTeam1[pos[1]][pos[0]] += 1
            if self.threeZone[pos[1]][pos[0]]:
                self.threePAteam1 += 1
            else:
                self.FGAteam1 += 1
        else:
            self.shotTrackTeam2[pos[1]][pos[0]] += 1
            if self.threeZone[pos[1]][pos[0]]:
                self.threePAteam2 += 1
            else:
                self.FGAteam2 += 1
        
    def setStep(self, pos, team):
        radius = 8
        for y in range(pos[1]-radius, pos[1]+radius):
            for x in range(pos[0]-radius, pos[0]+radius):
                if pixelInsideCircle(x, y, pos, radius):
                    if team == 0:
                        self.movementHeatmapTeam1[y][x] += 5
                    else:
                        self.movementHeatmapTeam2[y][x] += 5


    def getShotDetectedMap(self):
        return self.shotTrackTeam1, self.shotTrackTeam2
    
    def printHeatmapStepsTeam1(self):
        # TODO: port this function to statisticsGeneration.py, but saving the image instead of showing it
        blurred = cv2.GaussianBlur(self.movementHeatmapTeam1, (5, 5), 0)
        matrixNormalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        heatmap = cv2.applyColorMap(matrixNormalized, cv2.COLORMAP_JET)
        heatmap = cv2.addWeighted(heatmap, 0.5, self.topviewImage, 0.5, 0)
        cv2.imshow("movementHeatmapTeam1", heatmap)
        cv2.waitKey(0)
    
    def printHeatmapStepsTeam2(self):
        # TODO: port this function to statisticsGeneration.py, but saving the image instead of showing it
        blurred = cv2.GaussianBlur(self.movementHeatmapTeam2, (5, 5), 0)
        matrixNormalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        heatmap = cv2.applyColorMap(matrixNormalized, cv2.COLORMAP_JET)
        heatmap = cv2.addWeighted(heatmap, 0.5, self.topviewImage, 0.5, 0)
        cv2.imshow("movementHeatmapTeam2", heatmap)
        cv2.waitKey(0)