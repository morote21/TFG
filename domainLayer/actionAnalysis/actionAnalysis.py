import cv2
import numpy as np


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
        # TODO: FALTA CERRAR LAS ZONAS CON IMOPEN O IMCLOSE PARA QUE NO HAYA HUECOS EN LAS ZONAS

    
    def shotDetected(self, pos):
        if self.threeZone[pos[1]][pos[0]]:
            return 3
        else:
            return 2

        

    