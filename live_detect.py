from ultralytics import YOLO
import cv2

# Load YOLOv8 pretrained model
model = YOLO('yolov8n.pt')

# Start webcam (0 = default front camera)
cap = cv2.VideoCapture(0)

# Allowed objects (COCO class names)
allowed_objects = ['person', 'chair', 'laptop', 'cell phone', 'book', 'handbag', 'backpack']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection with confidence threshold 0.35
    results = model(frame, conf=0.35)

    # Annotate frame (shows boxes for all objects detected)
    annotated_frame = results[0].plot()
    cv2.imshow("Live Detection", annotated_frame)

    # Print only allowed objects
    detected_this_frame = set()  # To avoid duplicate prints in one frame
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            if name in allowed_objects:
                detected_this_frame.add(name)

    # Print detected objects once per frame
    if detected_this_frame:
        print(", ".join(detected_this_frame))

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

