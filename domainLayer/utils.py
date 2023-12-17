import cv2
import numpy as np
import torch

def resizeFrame(frame, height=1080):
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