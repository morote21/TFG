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



VIDEO_PATH = "/home/morote/Desktop/input_tfg/20231215_131239_Trim.mp4"

model = YOLO('domainLayer/models/yolov8m-pose.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
model.predict(source=VIDEO_PATH, show=True, device=0)