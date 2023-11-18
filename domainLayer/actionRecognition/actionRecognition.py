import cv2
import numpy as np
import sys
sys.path.insert(0, '..')

import copy
from easydict import EasyDict
from random import randint
from imutils.video import FPS

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.video import R2Plus1D_18_Weights
from domainLayer.utils import load_weights

MODEL_PATH = "/home/morote/Desktop/TFG/domainLayer/models/model_checkpoints/r2plus1d_augmented-3/r2plus1d_multiclass_12_0.0001.pt"

# SEGURAMENTE ESTE MAL, HACERLO A MI MANERA PERO RESPETANDO COMO SERIA LA ENTRADA DE LA RED
def cropAction(clip, cropWindow, player=0):

    video = []
    for i, frame in enumerate(clip):
        x = int(cropWindow[i][player][0])
        y = int(cropWindow[i][player][1])
        w = int(cropWindow[i][player][2])
        h = int(cropWindow[i][player][3])

        croppedFrame = frame[y:y+h, x:x+w]

        try:
            resizedFrame = cv2.resize(croppedFrame, dsize=(int(128), int(176)), interpolation=cv2.INTER_NEAREST)
        
        except:
            if len(video) == 0:
                resizedFrame = np.zeros((int(176), int(128), int(3)), dtype=np.uint8) 
            else:
                resizedFrame = video[i-1]

        assert resizedFrame.shape == (176, 128, 3)  # Check if size is correct
        video.append(resizedFrame) 
    
    return video

def cropPlayer(frame, boundingbox):
    x = int(boundingbox[0])
    y = int(boundingbox[1])
    w = int(boundingbox[2]) - int(boundingbox[0])
    h = int(boundingbox[3]) - int(boundingbox[1])

    croppedFrame = frame[y:y+h, x:x+w]

    try:
        resizedFrame = cv2.resize(croppedFrame, dsize=(int(128), int(176)), interpolation=cv2.INTER_NEAREST)
    
    except:
        resizedFrame = np.zeros((int(176), int(128), int(3)), dtype=np.uint8) 

    assert resizedFrame.shape == (176, 128, 3)  # Check if size is correct
    
    return resizedFrame


def inferenceShape(batch):
    batch = batch.permute(3, 0, 1, 2)    # (time, height, width, channels) -> (channels, time, height, width)
    # add 1 dimension to the beginning of the tensor of size 1
    batch = batch.unsqueeze(0)
    print(batch.shape)
    return batch


class ActionRecognition:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.video.r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT, progress=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10, bias=True)
        self.model = load_weights(self.model, "/home/morote/Desktop/TFG/domainLayer/models/model_checkpoints/r2plus1d_augmented-3",
                                  "r2plus1d_multiclass", 12, 0.0001)
        if torch.cuda.is_available():
            self.model = self.model.to(device)
        
        self.model.eval()
        
        self.labels = {0 : "block", 1 : "pass", 2 : "run", 3 : "dribble", 4 : "shoot", 5 : "ball in hand", 6 : "defense", 7 : "pick" , 8 : "no_action" , 9 : "walk" , 10 : "discard"}
    

    def inference(self, input):
        outputs = self.model(input)
        _, pred = torch.max(outputs, 1)
        return pred.cpu().numpy()
    
    def getLabel(self, label):
        return self.labels[label]