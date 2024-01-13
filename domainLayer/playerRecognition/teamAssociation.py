import numpy as np
import matplotlib.pyplot as plt
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
    """
    Gets the distance between two means
    :param mean1: mean of the first player (tuple)
    :param mean2: mean of the second player (tuple)
    :return: distance between the two means (float)
    """
    return (mean1[0]-mean2[0])**2 + (mean1[1]-mean2[1])**2 + (mean1[2]-mean2[2])**2


def getMeanOfEachColorChannel(image):
    """
    Gets the mean of each color channel of the image
    :param image: image (np.array)
    :return: mean of each color channel (tuple)
    """
    # BGR
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

    meanB = np.mean(b)
    meanG = np.mean(g)
    meanR = np.mean(r)

    return meanB, meanG, meanR

    

class Teams:
    def __init__(self, image_player_1, image_player_2, n_players_team):
        self.players_per_team = n_players_team

        self.model = YOLO("/home/morote/Desktop/TFG/domainLayer/models/yolov8m-pose.pt")
        self.clusters = None

        self.team1Torax = getMeanOfEachColorChannel(self.getRoiOfPlayer(image_player_1)[0])
        self.team2Torax = getMeanOfEachColorChannel(self.getRoiOfPlayer(image_player_2)[0])
        self.team1Hip = getMeanOfEachColorChannel(self.getRoiOfPlayer(image_player_1)[1])
        self.team2Hip = getMeanOfEachColorChannel(self.getRoiOfPlayer(image_player_2)[1])

    def getRoiOfPlayer(self, image):
        """
        Gets the region of interest of the player
        :param image: image of the player (np.array)
        :return: region of interest of the player (np.array)
        """
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


    def associate(self, player):
        """
        Associates the player to a team
        :param player: player to associate (np.array)
        :return: team of the player (int)
        """
        distancePlayerToC1Torax = getDistance(getMeanOfEachColorChannel(self.getRoiOfPlayer(player)[0]), self.team1Torax)
        distancePlayerToC2Torax = getDistance(getMeanOfEachColorChannel(self.getRoiOfPlayer(player)[0]), self.team2Torax)

        distancePlayerToC1Hip = getDistance(getMeanOfEachColorChannel(self.getRoiOfPlayer(player)[1]), self.team1Hip)
        distancePlayerToC2Hip = getDistance(getMeanOfEachColorChannel(self.getRoiOfPlayer(player)[1]), self.team2Hip)

        distToC1 = min(distancePlayerToC1Torax, distancePlayerToC1Hip)
        distToC2 = min(distancePlayerToC2Torax, distancePlayerToC2Hip)

        if distToC1 < distToC2:
            return 0
        else:
            return 1

