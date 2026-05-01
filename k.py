from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8 model
model = YOLO('yolov8n.pt')

# Test image from Ultralytics
img = 'https://ultralytics.com/images/bus.jpg'

# Run detection
results = model(img, show=True)

# Print detected object names
for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        name = model.names[cls]
        print(name)

