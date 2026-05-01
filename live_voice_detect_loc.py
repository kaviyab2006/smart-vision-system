from ultralytics import YOLO
import cv2
import pyttsx3
import threading
import time

# ----------------------------
# Configuration
# ----------------------------
CONF_THRESHOLD = 0.35
SPEAK_GAP = 2.0
ALLOWED_OBJECTS = [
    'person', 'chair', 'laptop', 'cell phone', 'book',
    'handbag', 'backpack', 'bottle'
]
# ----------------------------

# Load YOLOv8 model
model = YOLO('yolov8n.pt')
print("✅ YOLOv8 model loaded successfully!")

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera!")
    exit()

# Initialize voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 170)

def speak(text):
    def run():
        try:
            engine.say(text)
            engine.runAndWait()
        except:
            pass
    threading.Thread(target=run, daemon=True).start()

# Track last spoken time per object
last_spoken_time = {}

print("🎥 Starting detection with direction guidance. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔁 Flip the frame horizontally for correct left/right
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape
    results = model(frame, conf=CONF_THRESHOLD)
    annotated = results[0].plot()
    cv2.imshow("Smart Object Detection + Direction", annotated)

    current_time = time.time()
    detected_objects = []

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_index = int(box.cls.item())
            name = model.names[cls_index]
            if name in ALLOWED_OBJECTS:
                x1, y1, x2, y2 = box.xyxy[0]
                center_x = (x1 + x2) / 2

                # Determine direction based on position (after flip)
                if center_x < width / 3:
                    position = "on your left"
                elif center_x > 2 * width / 3:
                    position = "on your right"
                else:
                    position = "in front of you"

                detected_objects.append((name, position))

    # Speak detected objects with direction
    for obj, position in detected_objects:
        last_time = last_spoken_time.get((obj, position), 0)
        if current_time - last_time > SPEAK_GAP:
            message = f"I see {obj} {position}"
            print(f"🔊 {message}")
            speak(message)
            last_spoken_time[(obj, position)] = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Detection stopped.")
