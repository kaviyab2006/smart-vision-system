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
SPEAK_GAP = 2.0  # seconds before repeating same object
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

# Track last spoken time per object
last_spoken_time = {}

# Global variable for specific object command
specific_object = None

# ----------------------------
# Function to speak without blocking
# ----------------------------
def speak(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

# ----------------------------
# Voice command listener
# ----------------------------
def listen_commands():
    global specific_object
    recognizer = sr.Recognizer()
    mic = sr.Microphone(device_index=0)  # change if needed

    while True:
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                print("🎙️ Listening for voice command...")
                audio = recognizer.listen(source, phrase_time_limit=3)
                command = recognizer.recognize_google(audio).lower()
                print("🗣️ You said:", command)

                if "find" in command:
                    for obj in ALLOWED_OBJECTS:
                        if obj in command:
                            specific_object = obj
                            speak(f"Tracking {obj} now")
                            print(f"🎤 Command received: Track {obj}")
                            break
                elif "stop tracking" in command:
                    specific_object = None
                    speak("Tracking all objects now")
                    print("🎤 Command received: Stop tracking")
        except sr.UnknownValueError:
            pass  # ignore unrecognized speech
        except Exception as e:
            print("❌ Voice listener error:", e)

# Start listener in a background thread
threading.Thread(target=listen_commands, daemon=True).start()

# ----------------------------
# Detection loop
# ----------------------------
print("🎥 Starting detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not captured. Exiting.")
        break

    # Flip frame for selfie camera
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # YOLO detection
    results = model(frame, conf=CONF_THRESHOLD)
    annotated = results[0].plot()
    cv2.imshow("Smart Object Detection + Voice Command", annotated)

    current_time = time.time()
    detected_objects = []

    # Process detected boxes
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_index = int(box.cls.item())
            name = model.names[cls_index]

            # Filter by allowed objects
            if name not in ALLOWED_OBJECTS:
                continue
            if specific_object and name != specific_object:
                continue

            x1, y1, x2, y2 = box.xyxy[0]
            center_x = (x1 + x2) / 2
            box_height = y2 - y1

            # Determine direction
            if center_x < width / 3:
                position = "on your right"   # flipped frame
            elif center_x > 2 * width / 3:
                position = "on your left"    # flipped frame
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

    # Speak detected objects
    for obj, position, distance in detected_objects:
        last_time = last_spoken_time.get((obj, position, distance), 0)
        if current_time - last_time > SPEAK_GAP:
            message = f"I see {obj} {position}, {distance}"
            print(f"🔊 {message}")
            speak(message)
            last_spoken_time[(obj, position, distance)] = current_time

    # Quit safely
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera properly
cap.release()
cv2.destroyAllWindows()
print("👋 Detection stopped.")
    