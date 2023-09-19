from ultralytics import YOLO

# Load pretrained YOLOv8n model
model = YOLO('yolov8m-pose.pt')

# Run inference on the source
results = model.track(source='IMG_0500.mp4', show=True, save=True, tracker="bytetrack.yaml")

print(len(results)) # canitad de frames del video


first_frame = results[0]

boxes = first_frame.boxes

for box in boxes:
    print(box.xywh)
