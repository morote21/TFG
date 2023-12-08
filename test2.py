from ultralytics import YOLO

model = YOLO('domainLayer/models/yolov8m.pt')
model.export(format='onnx')