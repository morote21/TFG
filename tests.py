# from ultralytics import YOLO

# model = YOLO("./runs/detect/train/weights/best.pt")

# results = model.track(source="/home/morote/Desktop/input_tfg/nba2k_test.mp4", show=True, conf=0.5)

import pickle
import numpy as np

#file = open("/home/morote/Desktop/dataset/examples/0000000.npy", "rb")

l = np.load("/home/morote/Desktop/dataset/examples/0000000.npy", allow_pickle=True)
print(l)