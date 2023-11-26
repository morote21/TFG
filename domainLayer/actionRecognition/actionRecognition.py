import cv2
import numpy as np

import copy
from easydict import EasyDict
from random import randint
from imutils.video import FPS

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.video import R2Plus1D_18_Weights
from domainLayer.utils import load_weights, getMostCommonElement

MODEL_PATH = "/home/morote/Desktop/TFG/domainLayer/models/model_checkpoints/r2plus1d_augmented-3/r2plus1d_multiclass_12_0.0001.pt"
SIZE_OF_ACTION_QUEUE = 7


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
    x = max(int(boundingbox[0]) - 10, 0)
    y = max(int(boundingbox[1]) - 10, 0)
    w = int(boundingbox[2]) - int(boundingbox[0]) + 20
    h = int(boundingbox[3]) - int(boundingbox[1]) + 20

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
    return batch


class ActionRecognition:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.video.r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT, progress=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10, bias=True)
        self.model = load_weights(self.model, "/home/morote/Desktop/TFG/domainLayer/models/model_checkpoints/r2plus1d_augmented-3",
                                  "r2plus1d_multiclass", 12, 0.0001)
        if torch.cuda.is_available():
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        self.labels = {0 : "block", 1 : "pass", 2 : "run", 3 : "dribble", 4 : "shoot", 5 : "ball in hand", 6 : "defense", 7 : "pick" , 8 : "no_action" , 9 : "walk" , 10 : "discard"}

        self.playersFrames = {}                                                  # DICTIONARY TO STORE THE LAST 16 FRAMES OF EACH PLAYER
        self.playersFrameFlag = {}                                               # DICTIONARY TO STORE WHEN WE ARE ABLE TO STORE A FRAME FOR A CERTAIN PLAYER
        self.playersPartialClassifications = {}                                  # DICTIONARY TO STORE THE PARTIAL CLASSIFICATIONS OF EACH PLAYER
        self.playersFinalClassifications = {}                                    # DICTIONARY TO STORE THE FINAL CLASSIFICATION OF EACH PLAYER


    def inference(self, frame, boxes, ids, classes):
        # CHECK IF THERE ARE NEW PLAYERS OR PLAYERS THAT HAVE LEFT THE COURT OR NOT TRACKED DUE TO OCCLUSION
        if len(ids) != len(self.playersFrames.keys()):
            if len(self.playersFrames.keys()) < len(ids):
                for iden in ids:
                    if iden not in list(self.playersFrames.keys()):
                        self.playersFrames[iden] = []
                        self.playersPartialClassifications[iden] = []
                        self.playersFrameFlag[iden] = True
                        self.playersFinalClassifications[iden] = "undefined"
            
            else:
                for iden in list(self.playersFrames.keys()):
                    if iden not in ids:
                        self.playersFrames.pop(iden)
                        self.playersFrameFlag.pop(iden)
                        self.playersPartialClassifications.pop(iden)
                        self.playersFinalClassifications.pop(iden)
        
        # DRAW BOUNDING BOXES, ASSOCIATE PLAYERS WITH TEAMS AND PERFORM ACTION RECOGNITION FOR EACH PLAYER
        for box, identity, cls in zip(boxes, ids, classes):
            
            hasAction = False

            cropAndResize = cropPlayer(frame, box)               # CROP PLAYER FROM FRAME FOR ACTION RECOGNITION

            # QUEUE OF 16 FRAMES
            if self.playersFrameFlag[identity]:
                self.playersFrames[identity].append(cropAndResize)
                self.playersFrameFlag[identity] = False
            else:
                self.playersFrameFlag[identity] = True

            if len(self.playersFrames[identity]) > 16:
                    self.playersFrames[identity].pop(0)

            # INFERENCE
            if len(self.playersFrames[identity]) == 16:
                inputFrames = inferenceShape(torch.Tensor(self.playersFrames[identity]))  # ADJUST SHAPE FOR INFERENCE
                inputFrames = inputFrames.to(device=self.device)                             # SEND TO GPU

                with torch.no_grad():
                    outputs = self.model(inputFrames)                    # INFERENCE
                    _, pred = torch.max(outputs, 1)
                    pred = pred.cpu().numpy()
                    action = self.labels[pred[0]]                       # GET LABEL
                    self.playersPartialClassifications[identity].append(action)                     # APPEND TO QUEUE OF CLASSIFICATIONS

                    if len(self.playersPartialClassifications[identity]) > SIZE_OF_ACTION_QUEUE: 
                        self.playersPartialClassifications[identity].pop(0)

                    if len(self.playersPartialClassifications[identity]) == SIZE_OF_ACTION_QUEUE:   # QUEUE OF 5 CLASSIFICATIONS
                        self.playersFinalClassifications[identity] = getMostCommonElement(self.playersPartialClassifications[identity]) # GET MOST COMMON CLASSIFICATION
                        #hasAction = True

        return self.playersFinalClassifications
    
    def getLabel(self, label):
        return self.labels[label]