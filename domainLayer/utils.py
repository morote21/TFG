import cv2
import numpy as np
import torch

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
