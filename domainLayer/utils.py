import cv2
import numpy as np
import torch

def resizeFrame(frame, height=1080):
    """
    Resizes the frame to a specific height
    :param frame: frame to resize
    :param height: height to resize to
    :return: resized frame
    """
    aspect_ratio = frame.shape[1] / frame.shape[0]  # width / height
    width = int(height * aspect_ratio)
    resized_frame = cv2.resize(frame, (width, height))
    return resized_frame


def getBorders(image):
    """
    Gets the borders of the court and the topview image
    :param image: image to get borders from
    """

    def clickEvent(event, x, y, flags, param):        
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_LBUTTONUP:
            param.append((x, y))
    
    borders = []
    cv2.namedWindow("borders")
    cv2.setMouseCallback("borders", clickEvent, param=borders)

    while 1:
        cv2.imshow("borders", image)
        if len(borders) == 8:
            cv2.destroyAllWindows()
            break
        
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

    return np.array(borders)


def getRim(image):
    """
    Gets the position of the rim in the image
    :param image: image to get rim position from
    :return: position of the rim in the image
    """

    def clickEvent(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_LBUTTONUP:
            param.append((x, y))

    coords = []
    cv2.namedWindow("rim coords")
    cv2.setMouseCallback("rim coords", clickEvent, param=coords)

    while 1:
        cv2.imshow("rim coords", image)
        if len(coords) == 2:
            cv2.destroyAllWindows()
            break

        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

    return np.array(coords)


def load_weights(model, modelPath, baseModelName, startEpoch, lr):
    """
    Loads previously trained weights into a model given an epoch and the model itself
    :param base_model_name: name of the base model in training session
    :param model: model to load weights into
    :param epoch: what epoch of training to load
    :param model_path: path of where model is loaded from
    :return: the model with weights loaded in
    """

    pretrainedDict = torch.load('{}/{}_{}_{}.pt'.format(modelPath, baseModelName, startEpoch, lr))['state_dict']
    modelDict = model.state_dict()
    pretrainedDict = {k: v for k, v in pretrainedDict.items() if k in modelDict}
    modelDict.update(pretrainedDict)
    model.load_state_dict(modelDict)

    return model


def getMostCommonElement(array):
    """
    Gets the most common element in an array
    :param array: array to get most common element from
    :return: most common element
    """
    return max(set(array), key=array.count)


def getIntersection(box1, box2):
    """
    Gets the intersection between two boxes
    :param box1: first box
    :param box2: second box
    :return: intersection between the two boxes
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    return max(0, x2 - x1) * max(0, y2 - y1)

def getUnion(box1, box2):
    """
    Gets the union between two boxes
    :param box1: first box
    :param box2: second box
    :return: union between the two boxes
    """
    intersection = getIntersection(box1, box2)
    areaBox1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    areaBox2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = areaBox1 + areaBox2 - intersection

    return union


def whoHasPossession(playersIdsAndBoxes, ballSize):
    """
    Gets the player who has possession of the ball, which is the one who is the closest to the ball
    :param playersIdsAndBoxes: list of tuples containing the player id and the player box
    :param ball: ball box
    :return: player who has possession of the ball
    """
    # MIN_DIST = 5
    # ballCenter = np.array([ballSize[0] + ballSize[2], ballSize[1] + ballSize[3]]) / 2.0
    # player = None
    # bestDistToBall = 100000
    # for playerId, playerBox in playersIdsAndBoxes:
    #     centerPlayer = np.array([playerBox[0] + playerBox[2], playerBox[1] + playerBox[3]]) / 2.0
    #     distToPlayer = np.linalg.norm(centerPlayer - ballCenter)
    #     minDistToPlayer = (ballSize[2] - ballSize[0]) * MIN_DIST
    #     if distToPlayer < bestDistToBall and distToPlayer < minDistToPlayer:
    #         bestDistToBall = distToPlayer
    #         player = playerId


    # check possession with intersection of at least 0.8
    player = None
    bestIntersection = 0
    for playerId, playerBox in playersIdsAndBoxes:
        intersection = getIntersection(ballSize, playerBox)

        if intersection > bestIntersection:
            bestIntersection = intersection
            player = playerId
    
    return player

            