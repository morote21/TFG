import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import copy
from ultralytics import YOLO

def colordepth24bitTo8bit(img):
    B_2msb = np.right_shift(img, 16)  # 2 bit size
    G_3msb = np.right_shift(img, 8)

    B_2msb = np.bitwise_and(B_2msb, 0xff)  # 2 bit size
    G_3msb = np.bitwise_and(G_3msb, 0xff)
    R_3msb = np.bitwise_and(img, 0xff)

    B_2msb = np.right_shift(B_2msb, 6)  # 2 bit size
    G_3msb = np.right_shift(G_3msb, 5)
    R_3msb = np.right_shift(R_3msb, 5)

    B_2msb = np.left_shift(B_2msb, 6)  # 8 bit size
    G_3msb = np.left_shift(G_3msb, 3)

    BGR_8bit = np.bitwise_or(B_2msb, G_3msb)
    BGR_8bit = np.bitwise_or(BGR_8bit, R_3msb)

    BGR_8bit = BGR_8bit.astype(np.uint8)
    BGR_8bit = BGR_8bit * 32

    return BGR_8bit


def getDistance(mean1, mean2):
    return (mean1[0]-mean2[0])**2 + (mean1[1]-mean2[1])**2 + (mean1[2]-mean2[2])**2


def getMeanOfEachColorChannel(image):

    # BGR
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

    meanB = np.mean(b)
    meanG = np.mean(g)
    meanR = np.mean(r)

    return meanB, meanG, meanR


def get_color_histogram(image):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Poner una mascara mas adelante por la parte del torax
    width = im.shape[1]  # Use the width of the image
    height = im.shape[0]  # Use the height of the image

    # Mascara donde se situa la ROI (tren superior, que es donde se encuentra la camiseta)
    quarter_wide = width // 4
    quarter_height = height // 4
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[quarter_height:(height - quarter_height), quarter_wide:(width - quarter_wide)] = 255

    hist = cv2.calcHist([im], [0, 1, 2], mask, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    return hist

def getRoiOfPlayer(image):
    # crop image by 25% above and 40% below
    height = image.shape[0]
    width = image.shape[1]
    cropAbove = int(height * 0.25)
    cropBelow = int(height * 0.4)
    image = image[cropAbove:height - cropBelow, :]

    # crop image by 15% left and 15% right
    cropLeft = int(width * 0.15)
    cropRight = int(width * 0.15)
    image = image[:, cropLeft:width - cropRight]

    return image


    

class Teams:
    def __init__(self, image_player_1, image_player_2, n_players_team):
        self.players_per_team = n_players_team
        # self.team1_histogram = get_color_histogram(image_player_1)
        # self.team2_histogram = get_color_histogram(image_player_2)
        self.model = YOLO("/home/morote/Desktop/TFG/domainLayer/models/yolov8m-pose.pt")
        self.clusters = None

        # image_player_1 = getRoiOfPlayer(image_player_1)
        # image_player_2 = getRoiOfPlayer(image_player_2)
        # self.team1mean = getMeanOfEachColorChannel(image_player_1)
        # self.team2mean = getMeanOfEachColorChannel(image_player_2)

        self.team1Torax = getMeanOfEachColorChannel(self.getRoiOfPlayer2(image_player_1)[0])
        self.team2Torax = getMeanOfEachColorChannel(self.getRoiOfPlayer2(image_player_2)[0])
        self.team1Hip = getMeanOfEachColorChannel(self.getRoiOfPlayer2(image_player_1)[1])
        self.team2Hip = getMeanOfEachColorChannel(self.getRoiOfPlayer2(image_player_2)[1])

    def associate2(self, image):
        # im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # hist = get_color_histogram(im)
        # cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        # dist_team1 = cv2.compareHist(hist, self.team1_histogram, cv2.HISTCMP_BHATTACHARYYA)
        # dist_team2 = cv2.compareHist(hist, self.team2_histogram, cv2.HISTCMP_BHATTACHARYYA)
        imageCpy = copy.deepcopy(image)
        imageCpy = getRoiOfPlayer(imageCpy)
        #imageCpy = colordepth24bitTo8bit(imageCpy)
        mean = getMeanOfEachColorChannel(imageCpy)
        dist_team1 = (mean[0]-self.team1mean[0])**2 + (mean[1] - self.team1mean[1])**2 + (mean[2]-self.team1mean[2])**2
        dist_team2 = (mean[0]-self.team2mean[0])**2 + (mean[1] - self.team2mean[1])**2 + (mean[2]-self.team2mean[2])**2

        if dist_team1 < dist_team2:
            return 0
        else:
            return 1


    def getRoiOfPlayer2(self, image):
        player = copy.deepcopy(image)
        results = self.model.predict(player)
        keypoints = results[0].keypoints.xy.cpu().numpy()[0]
        ls = keypoints[5]   # left shoulder
        rs = keypoints[6]   # right shoulder
        lh = keypoints[11]  # left hip
        rh = keypoints[12]  # right hip

        # crop image by left-most shoulder and right-most hip
        ls_x = int(ls[0])
        rs_x = int(rs[0])
        lh_x = int(lh[0])
        rh_x = int(rh[0])

        ls_y = int(ls[1])
        rh_y = int(rh[1])

        if ls_x < rs_x:
            left_torax = ls_x
        else:
            left_torax = rs_x

        if lh_x > rh_x:
            right_hip = lh_x
            right_torax = lh_x
            left_hip = rh_x
        else:
            right_torax = rh_x
            right_hip = rh_x
            left_hip = lh_x
        
        # crop image by 25% above and 40% below
        height = player.shape[0]
        width = player.shape[1]


        roiTorax = image[ls_y:rh_y, left_torax:right_torax]

        roiHip = image[int(rh_y - height*0.10):int(rh_y + height*0.10), int(left_hip - width*0.10):int(right_hip + width*0.10)]

        return roiTorax, roiHip
    
        

    def get2PlayersWithHighestBGRDistance(self, all_test_images):
        """
        Gets the players with the highest RGB distance
        :param all_test_images: list of images of the players
        :return: the 2 players with the highest RGB distance between them
        """
        
        maxDistance = 0
        player1 = None
        player2 = None

        for i in range(len(all_test_images)):
            for j in range(i + 1, len(all_test_images)):
                player1Means = getMeanOfEachColorChannel(self.getRoiOfPlayer2(all_test_images[i]))
                player2Means = getMeanOfEachColorChannel(self.getRoiOfPlayer2(all_test_images[j]))
                distance = getDistance(player1Means, player2Means)
                if distance > maxDistance:
                    maxDistance = distance
                    player1 = player1Means
                    player2 = player2Means

        return player1, player2
    

    def fitPlayers(self, boxes, image):
        bboxes = []
        for box in boxes:
            bboxes.append(image[box[1]:box[3], box[0]:box[2]])
        # image = colordepth24bitTo8bit(image)
        self.clusters = self.get2PlayersWithHighestBGRDistance(bboxes)
        self.clusters = np.array(self.clusters)
        
        if self.team1 is None:
            self.team1 = self.clusters[0]
            self.team2 = self.clusters[1]

        else:
            # update teams so the clusters represent the same team as the previous frame
            distc1t1 = getDistance(self.clusters[0], self.team1)
            distc2t1 = getDistance(self.clusters[1], self.team1)
            distc1t2 = getDistance(self.clusters[0], self.team2)
            distc2t2 = getDistance(self.clusters[1], self.team2)
            # get min distance, so the cluster with closest distance to the team is the same team
            minDist = min(distc1t1, distc2t1, distc1t2, distc2t2)
            if minDist == distc1t1 or minDist == distc1t2:
                self.team1 = self.clusters[0]
                self.team2 = self.clusters[1]
            elif minDist == distc2t1 or minDist == distc2t2:
                self.team1 = self.clusters[1]
                self.team2 = self.clusters[0]


    def associate(self, player):
        distancePlayerToC1Torax = getDistance(getMeanOfEachColorChannel(self.getRoiOfPlayer2(player)[0]), self.team1Torax)
        distancePlayerToC2Torax = getDistance(getMeanOfEachColorChannel(self.getRoiOfPlayer2(player)[0]), self.team2Torax)

        distancePlayerToC1Hip = getDistance(getMeanOfEachColorChannel(self.getRoiOfPlayer2(player)[1]), self.team1Hip)
        distancePlayerToC2Hip = getDistance(getMeanOfEachColorChannel(self.getRoiOfPlayer2(player)[1]), self.team2Hip)

        distToC1 = min(distancePlayerToC1Torax, distancePlayerToC1Hip)
        distToC2 = min(distancePlayerToC2Torax, distancePlayerToC2Hip)

        if distToC1 < distToC2:
            return 0
        else:
            return 1

