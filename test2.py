from ultralytics import YOLO
import torch
import cv2
import sys
import copy
import numpy as np

def resizeFrame(frame, height=1080):
    aspect_ratio = frame.shape[1] / frame.shape[0]  # width / height
    width = int(height * aspect_ratio)
    resized_frame = cv2.resize(frame, (width, height))
    return resized_frame



VIDEO_PATH = "/home/morote/Desktop/input_tfg/nba2k_test.mp4"

model = YOLO('domainLayer/models/all_detections_model.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
model.predict(source=VIDEO_PATH, save=True)