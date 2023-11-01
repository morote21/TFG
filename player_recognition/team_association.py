import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib


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


class Teams:
    def __init__(self, image_player_1, image_player_2, n_players_team):
        self.players_per_team = n_players_team
        self.team1_histogram = get_color_histogram(image_player_1)
        self.team2_histogram = get_color_histogram(image_player_2)

    def associate(self, image):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hist = get_color_histogram(im)
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        dist_team1 = cv2.compareHist(hist, self.team1_histogram, cv2.HISTCMP_BHATTACHARYYA)
        dist_team2 = cv2.compareHist(hist, self.team2_histogram, cv2.HISTCMP_BHATTACHARYYA)

        if dist_team1 < dist_team2:
            return 0
        else:
            return 1
