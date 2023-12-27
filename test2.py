import cv2
import sys
import copy
import numpy as np
from pathlib import Path
from ultralytics import YOLO

IMAGE_PATH = "/home/morote/Pictures/prueba24to8bit.png"

IMAGE_PATH_EQUIPMENT_1 = "/home/morote/Desktop/input_tfg/camiseta_salle_verde.jpg"
IMAGE_PATH_EQUIPMENT_2 = "/home/morote/Desktop/input_tfg/camiseta_salle_negra.jpg"

IMAGE_TEST_PATH = "/home/morote/Pictures"

def colordepth24bit_to_8bit(img):
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


def getRoiOfPlayer2(image):
    player = copy.deepcopy(image)
    results = model.predict(player)
    keypoints = results[0].keypoints.xy.cpu().numpy()[0]
    ls = keypoints[5]
    rs = keypoints[6]
    lh = keypoints[11]
    rh = keypoints[12]

    # crop image by left-most shoulder and right-most hip
    ls_x = int(ls[0])
    rs_x = int(rs[0])
    lh_x = int(lh[0])
    rh_x = int(rh[0])

    if ls_x < rs_x:
        cropLeft = ls_x
    else:
        cropLeft = rs_x

    if lh_x > rh_x:
        cropRight = lh_x
    else:
        cropRight = rh_x
    
    player = player[:, cropLeft:cropRight]

    # crop image by 25% above and 40% below
    height = player.shape[0]
    width = player.shape[1]
    cropAbove = int(height * 0.25)
    cropBelow = int(height * 0.4)
    player = player[cropAbove:height - cropBelow, :]

    return player


def getMeanOfEachColorChannel(image):

    # BGR
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

    meanB = np.mean(b)
    meanG = np.mean(g)
    meanR = np.mean(r)

    return np.array([meanB, meanG, meanR])


def getAssociationScore(image, equip1, equip2):
    """
    Gets the association score between the player and the two teams
    :param image: image of the player
    :param equip1: image of the first team equipment
    :param equip2: image of the second team equipment
    :return: 1 if the player is in the first team, 2 if the player is in the second team
    """

    meanPlayer = getMeanOfEachColorChannel(getRoiOfPlayer(image))
    meanEquip1 = getMeanOfEachColorChannel(getRoiOfPlayer(equip1))
    meanEquip2 = getMeanOfEachColorChannel(getRoiOfPlayer(equip2))

    distTeam1 = (meanPlayer[0]-meanEquip1[0])**2 + (meanPlayer[1] - meanEquip1[1])**2 + (meanPlayer[2]-meanEquip1[2])**2 
    distTeam2 = (meanPlayer[0]-meanEquip2[0])**2 + (meanPlayer[1] - meanEquip2[1])**2 + (meanPlayer[2]-meanEquip2[2])**2

    if distTeam1 < distTeam2:
        return 1
    else:
        return 2


def readImageTests(testsPath):
    images = []
    path = Path(testsPath)
    # get all images named equipo_test_*.png from path and store them in images list
    for file in path.glob("equipo_test_*.png"):
        images.append(cv2.imread(str(file)))

    return images


def resizeFrame(frame, height=1080):
    aspect_ratio = frame.shape[1] / frame.shape[0]  # width / height
    width = int(height * aspect_ratio)
    resized_frame = cv2.resize(frame, (width, height))
    return resized_frame


def removeIllumination(image):
    imageCpy = copy.deepcopy(image)
    imageCpy = cv2.cvtColor(imageCpy, cv2.COLOR_BGR2YUV)

    channelY = imageCpy[:, :, 0]

    channelY = cv2.equalizeHist(channelY)

    imageCpy[:, :, 0] = channelY

    imageCpy = cv2.cvtColor(imageCpy, cv2.COLOR_YUV2BGR)

    return imageCpy


def getDistance(mean1, mean2):
    return (mean1[0]-mean2[0])**2 + (mean1[1]-mean2[1])**2 + (mean1[2]-mean2[2])**2

def get2PlayersWithHighestBGRDistance(all_test_images):
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
            player1Means = getMeanOfEachColorChannel(getRoiOfPlayer2(all_test_images[i]))
            player2Means = getMeanOfEachColorChannel(getRoiOfPlayer2(all_test_images[j]))
            distance = getDistance(player1Means, player2Means)
            if distance > maxDistance:
                maxDistance = distance
                player1 = player1Means
                player2 = player2Means

    return player1, player2





model = YOLO("/home/morote/Desktop/TFG/domainLayer/models/yolov8m-pose.pt")

# read image
equip_1 = cv2.imread(IMAGE_PATH_EQUIPMENT_1)
equip_1 = resizeFrame(equip_1, height=200)
equip_2 = cv2.imread(IMAGE_PATH_EQUIPMENT_2)
equip_2 = resizeFrame(equip_2, height=200)

all_test_images = readImageTests(IMAGE_TEST_PATH)

# cv2.imshow("equip_1", removeIllumination(equip_1))
# cv2.imshow("equip_2", removeIllumination(equip_2))

# cv2.imshow("test_image_1", getRoiOfPlayer(all_test_images[0]))

print("Mean equipment 1:", getMeanOfEachColorChannel(getRoiOfPlayer(equip_1)))
print("Mean equipment 2:", getMeanOfEachColorChannel(getRoiOfPlayer(equip_2)))

print("Mean test image 1:", getMeanOfEachColorChannel(getRoiOfPlayer(all_test_images[0])))

teamAssociated = getAssociationScore(all_test_images[0], equip_1, equip_2)
print(f"First associated to team {teamAssociated}")


furthestPlayers = get2PlayersWithHighestBGRDistance(all_test_images)
print(furthestPlayers)

cluster1 = []
cluster2 = []

for player in all_test_images:

    distancePlayerToC1 = getDistance(getMeanOfEachColorChannel(getRoiOfPlayer2(player)), furthestPlayers[0])
    distancePlayerToC2 = getDistance(getMeanOfEachColorChannel(getRoiOfPlayer2(player)), furthestPlayers[1])

    if distancePlayerToC1 < distancePlayerToC2:
        cluster1.append(player)
    else:
        cluster2.append(player)

print("Cluster 1:", len(cluster1))
print("Cluster 2:", len(cluster2))

for player in cluster1:
    cv2.imshow("cluster1", player)
    cv2.waitKey(0)

for player in cluster2:
    cv2.imshow("cluster2", player)
    cv2.waitKey(0)