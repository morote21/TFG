import cv2
import numpy as np

class ActionAnalysis:
    def __init__(self, topviewImage):
        self.topviewImage = cv2.cvtColor(topviewImage, cv2.COLOR_BGR2GRAY)
        self.topviewImage = cv2.GaussianBlur(self.topviewImage, (5, 5), 0)
        self.topviewImage = cv2.threshold(self.topviewImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        connectedComponents = cv2.connectedComponentsWithStats(self.topviewImage, 4, cv2.CV_32S)
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
        # TODO: hay que aumentar area en la que se incrementa los valores (poner dentro de un area a la redonda)
        if team == 0:
            self.movementHeatmapTeam1[pos[1]][pos[0]] += 20
            self.movementHeatmapTeam1[pos[1]+1][pos[0]] += 20
            self.movementHeatmapTeam1[pos[1]-1][pos[0]] += 20
            self.movementHeatmapTeam1[pos[1]][pos[0]+1] += 20
            self.movementHeatmapTeam1[pos[1]][pos[0]-1] += 20
            self.movementHeatmapTeam1[pos[1]+1][pos[0]+1] += 20
            self.movementHeatmapTeam1[pos[1]-1][pos[0]-1] += 20
            self.movementHeatmapTeam1[pos[1]+1][pos[0]-1] += 20
            self.movementHeatmapTeam1[pos[1]-1][pos[0]+1] += 20

        else:
            self.movementHeatmapTeam2[pos[1]][pos[0]] += 20
            self.movementHeatmapTeam2[pos[1]+1][pos[0]] += 20
            self.movementHeatmapTeam2[pos[1]-1][pos[0]] += 20
            self.movementHeatmapTeam2[pos[1]][pos[0]+1] += 20
            self.movementHeatmapTeam2[pos[1]][pos[0]-1] += 20
            self.movementHeatmapTeam2[pos[1]+1][pos[0]+1] += 20
            self.movementHeatmapTeam2[pos[1]-1][pos[0]-1] += 20
            self.movementHeatmapTeam2[pos[1]+1][pos[0]-1] += 20
            self.movementHeatmapTeam2[pos[1]-1][pos[0]+1] += 20


    def getShotDetectedMap(self):
        return self.shotTrackTeam1, self.shotTrackTeam2
    
    def printHeatmapStepsTeam1(self):
        # print the sum of all the values of self.movementHeatmapTeam1 matrix
        print(np.sum(self.movementHeatmapTeam1))
        print("----------------------------------")
        blurred = cv2.GaussianBlur(self.movementHeatmapTeam1, (5, 5), 0)
        matrixNormalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        heatmap = cv2.applyColorMap(matrixNormalized, cv2.COLORMAP_JET)
        cv2.imshow("movementHeatmapTeam1", heatmap)
        cv2.waitKey(0)
    
    def printHeatmapStepsTeam2(self):
        print(np.sum(self.movementHeatmapTeam2))
        blurred = cv2.GaussianBlur(self.movementHeatmapTeam2, (5, 5), 0)
        matrixNormalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        heatmap = cv2.applyColorMap(matrixNormalized, cv2.COLORMAP_JET)
        cv2.imshow("movementHeatmapTeam2", heatmap)
        cv2.waitKey(0)