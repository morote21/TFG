import cv2
import numpy as np

from imutils.video import FPS

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.video import R2Plus1D_18_Weights
from domainLayer.utils import load_weights, getMostCommonElement

MODEL_PATH = "/home/morote/Desktop/TFG/domainLayer/models/model_checkpoints/r2plus1d_augmented-3/r2plus1d_multiclass_12_0.0001.pt"
SIZE_OF_ACTION_QUEUE = 6

VIDEO_FPS = 30

def updateIds(dictFrames, dictPlayersFrameFlag, dictPartialClassifications, dictFinalClassifications, ids):
    """
    Updates the keys of the dictionaries with the ids of the players
    :param dictFrames: dictionary with the frames of each player (dictionary)
    :param dictPlayersFrameFlag: dictionary with the flag of each player (dictionary)
    :param dictPartialClassifications: dictionary with the partial classifications of each player (dictionary)
    :param dictFinalClassifications: dictionary with the final classifications of each player (dictionary)
    :param ids: ids of the players (list)
    :return: None
    """
    setOld = set(dictFrames.keys())
    setIds = set(ids)

    if setOld != setIds:
        for identity in setOld:
            if identity not in setIds:
                dictFrames.pop(identity)
                dictPartialClassifications.pop(identity)
                dictFinalClassifications.pop(identity)
                if VIDEO_FPS == 60:
                    dictPlayersFrameFlag.pop(identity)
        
        for identity in setIds:
            if identity not in setOld:
                dictFrames[identity] = []
                dictPartialClassifications[identity] = []
                dictFinalClassifications[identity] = "undefined"
                if VIDEO_FPS == 60:
                    dictPlayersFrameFlag[identity] = False


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
    """
    Crops the player from the frame
    :param frame: frame to crop (np.array)
    :param boundingbox: bounding box of the player (list)
    :return: cropped frame (np.array)
    """
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
    """
    Adjusts the shape of the batch for inference
    :param batch: batch of frames (torch.Tensor)
    :return: batch of frames (torch.Tensor)
    """
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
        """
        Performs inference on the frame
        :param frame: frame to perform inference (np.array)
        :param boxes: bounding boxes of the players (list)
        :param ids: ids of the players (list)
        :param classes: classes of the players (list)
        :return: final classifications of the action of each player detected (dictionary)
        """
        # CHECK IF THERE ARE NEW PLAYERS OR PLAYERS THAT HAVE LEFT THE COURT OR NOT TRACKED DUE TO OCCLUSION
        updateIds(self.playersFrames, self.playersFrameFlag, self.playersPartialClassifications, self.playersFinalClassifications, ids)
        
        # DRAW BOUNDING BOXES, ASSOCIATE PLAYERS WITH TEAMS AND PERFORM ACTION RECOGNITION FOR EACH PLAYER
        for box, identity, cls in zip(boxes, ids, classes):
            
            cropAndResize = cropPlayer(frame, box)               # CROP PLAYER FROM FRAME FOR ACTION RECOGNITION

            # QUEUE OF 16 FRAMES
            if VIDEO_FPS == 60:
                if self.playersFrameFlag[identity]:
                    self.playersFrames[identity].append(cropAndResize)
                    self.playersFrameFlag[identity] = False
                else:
                    self.playersFrameFlag[identity] = True
            
            else:
                self.playersFrames[identity].append(cropAndResize)

            if len(self.playersFrames[identity]) > 16:
                    self.playersFrames[identity].pop(0)

            # INFERENCE
            if len(self.playersFrames[identity]) == 16:
                inputFrames = inferenceShape(torch.Tensor(np.array(self.playersFrames[identity])))  # ADJUST SHAPE FOR INFERENCE
                inputFrames = inputFrames.to(device=self.device)                                    # SEND TO GPU

                with torch.no_grad():
                    outputs = self.model(inputFrames)                                               # INFERENCE
                    _, pred = torch.max(outputs, 1)
                    pred = pred.cpu().numpy()
                    action = self.labels[pred[0]]                                                   # GET LABEL
                    
                    self.playersPartialClassifications[identity].append(action)                     # APPEND TO QUEUE OF CLASSIFICATIONS

                    if len(self.playersPartialClassifications[identity]) > SIZE_OF_ACTION_QUEUE: 
                        self.playersPartialClassifications[identity].pop(0)

                    if len(self.playersPartialClassifications[identity]) == SIZE_OF_ACTION_QUEUE:   # QUEUE OF 5 CLASSIFICATIONS
                        if len(set(self.playersPartialClassifications[identity])) == 1:
                            self.playersFinalClassifications[identity] = self.playersPartialClassifications[identity][0]
                        
                    else:
                        action = "undefined"

                    #self.playersFinalClassifications[identity] = action

        return self.playersFinalClassifications
   
   