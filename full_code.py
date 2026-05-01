from ultralytics import YOLO
import cv2
import pyttsx3
import threading
import time
import speech_recognition as sr

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

# Global variable for specific object command
specific_object = None
recognizer = sr.Recognizer()

# ----------------------------
# Voice command listener
# ----------------------------
def listen_commands():
    global specific_object
    print(sr.Microphone.list_microphone_names())   # this prints all mic devices once
    mic = sr.Microphone(device_index=2)            # use 0 or the index of your actual mic

    while True:
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, phrase_time_limit=3)
                command = recognizer.recognize_google(audio).lower()
                if "find" in command:
                    words = command.split()
                    for obj in ALLOWED_OBJECTS:
                        if obj in words:
                            specific_object = obj
                            speak(f"Tracking {obj} now")
                            print(f"🎤 Command received: Track {obj}")
                            break
                elif "stop tracking" in command:
                    specific_object = None
                    speak("Tracking all objects now")
                    print("🎤 Command received: Stop tracking")
        except:
            pass  # Ignore unrecognized speech

# Start voice listener in background
threading.Thread(target=listen_commands, daemon=True).start()

print("🎥 Starting detection with voice command mode. Say 'Find [object]' or 'Stop tracking'. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for correct left/right
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # YOLO detection
    results = model(frame, conf=CONF_THRESHOLD)
    annotated = results[0].plot()
    cv2.imshow("Smart Object Detection + Voice Command", annotated)

    current_time = time.time()
    detected_objects = []

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_index = int(box.cls.item())
            name = model.names[cls_index]

            # Filter by allowed objects and specific object
            if name not in ALLOWED_OBJECTS:
                continue
            if specific_object and name != specific_object:
                continue

            x1, y1, x2, y2 = box.xyxy[0]
            center_x = (x1 + x2) / 2
            box_height = y2 - y1

            # Determine direction
            if center_x < width / 3:
                position = "on your left"
            elif center_x > 2 * width / 3:
                position = "on your right"
            else:
                position = "in front of you"

            # Estimate distance
            if box_height > height * 0.5:
                distance = "very close to you"
            elif box_height > height * 0.25:
                distance = "close to you"
            elif box_height > height * 0.1:
                distance = "a bit far"
            else:
                distance = "far away"

            detected_objects.append((name, position, distance))

    # Speak detected objects with direction and distance
    for obj, position, distance in detected_objects:
        last_time = last_spoken_time.get((obj, position, distance), 0)
        if current_time - last_time > SPEAK_GAP:
            message = f"I see {obj} {position}, {distance}"
            print(f"🔊 {message}")
            speak(message)
            last_spoken_time[(obj, position, distance)] = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Detection stopped.")
