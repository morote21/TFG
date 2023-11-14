from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')

results = model.train(data='/home/morote/Desktop/spacejam-cls-dataset', epochs=100, batch=-1, device=0, imgsz=224)