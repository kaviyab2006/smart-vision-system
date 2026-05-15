import streamlit as st
import cv2
from ultralytics import YOLO
import win32com.client
import numpy as np
import threading
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize voice engine
speaker = win32com.client.Dispatch("SAPI.SpVoice")

# Store last spoken object
last_spoken = ""

# Prevent multiple voice calls
is_speaking = False

# Camera handle for webcam mode
cap = None

# Voice function
def speak(text):

    global is_speaking

    if not is_speaking:

        is_speaking = True

        speaker.Speak(text)

        is_speaking = False

# Streamlit title
st.title("SMART VISION SYSTEM FOR OBJECT DETECTION")

# Sidebar options
option = st.sidebar.selectbox(
    "Choose Detection Type",
    ["Image Detection", "Webcam Detection"]
)

if "running" not in st.session_state:
    st.session_state.running = False

# ---------------- IMAGE DETECTION ---------------- #

if option == "Image Detection":

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.read()

        np_arr = np.frombuffer(file_bytes, np.uint8)

        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # YOLO Detection
        results = model(image)

        # Draw detections
        annotated_image = results[0].plot()

        # Display image
        st.image(
            annotated_image,
            channels="BGR"
        )

        # Voice detection
        boxes = results[0].boxes

        spoken_objects = []

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            if class_name not in spoken_objects:
                threading.Thread(
                    target=speak,
                    args=(f"{class_name} is detected",),
                    daemon=True
                ).start()
                spoken_objects.append(class_name)

        # Show detected objects
        st.subheader("Detected Objects")
        for obj in spoken_objects:
            st.write("✅", obj)

        if not spoken_objects:
            st.write("No objects detected.")


# ---------------- WEBCAM DETECTION ---------------- #

elif option == "Webcam Detection":

    col1, col2 = st.columns(2)
    with col1:
        start = st.button("START")
    with col2:
        stop = st.button("STOP")

    frame_window = st.image([])

    if start:
        st.session_state.running = True
        if cap is None:
            cap = cv2.VideoCapture(0)

    if stop:
        st.session_state.running = False
        if cap is not None:
            cap.release()
            cap = None

    if st.session_state.get('running', False):

        if cap is None:
            cap = cv2.VideoCapture(0)

        ret, frame = cap.read()

        if not ret:
            st.error("Unable to access webcam")
            if cap is not None:
                cap.release()
                cap = None
            st.session_state.running = False
        else:
            # YOLO prediction
            results = model(frame)

            # Draw detections
            annotated_frame = results[0].plot()

            # Get detected objects
            boxes = results[0].boxes
            spoken_objects = []

            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                if class_name not in spoken_objects:
                    threading.Thread(
                        target=speak,
                        args=(f"{class_name} is detected",),
                        daemon=True
                    ).start()
                    spoken_objects.append(class_name)

            frame_window.image(
                annotated_frame,
                channels="BGR"
            )

            if spoken_objects:
                st.subheader("Detected Objects")
                for obj in spoken_objects:
                    st.write("✅", obj)
            else:
                st.write("No objects detected.")

            time.sleep(0.05)
            st.experimental_rerun()