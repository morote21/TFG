from ultralytics import YOLO

model = YOLO('yolov8x.yaml')

results = model.train(data='/home/morote/Desktop/datasets/everything_yolov8/data.yaml', epochs=300, batch=-1, device=0)