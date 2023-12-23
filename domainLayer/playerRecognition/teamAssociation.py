import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import copy

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
        image_player_1 = getRoiOfPlayer(image_player_1)
        image_player_2 = getRoiOfPlayer(image_player_2)
        self.team1mean = getMeanOfEachColorChannel(image_player_1)
        self.team2mean = getMeanOfEachColorChannel(image_player_2)

    def associate(self, image):
        # im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # hist = get_color_histogram(im)
        # cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        # dist_team1 = cv2.compareHist(hist, self.team1_histogram, cv2.HISTCMP_BHATTACHARYYA)
        # dist_team2 = cv2.compareHist(hist, self.team2_histogram, cv2.HISTCMP_BHATTACHARYYA)
        imageCpy = copy.deepcopy(image)
        imageCpy = getRoiOfPlayer(imageCpy)
        imageCpy = colordepth24bitTo8bit(imageCpy)
        mean = getMeanOfEachColorChannel(imageCpy)
        dist_team1 = (mean[0]-self.team1mean[0])**2 + (mean[1] - self.team1mean[1])**2 + (mean[2]-self.team1mean[2])**2
        dist_team2 = (mean[0]-self.team2mean[0])**2 + (mean[1] - self.team2mean[1])**2 + (mean[2]-self.team2mean[2])**2

        if dist_team1 < dist_team2:
            return 0
        else:
            return 1
