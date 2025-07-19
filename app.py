import streamlit as st
import cv2
import tempfile
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

st.title("Real-Time Object Detection (Webcam)")

run = st.checkbox('Start Webcam')

FRAME_WINDOW = st.image([])

# Capture webcam if checkbox is checked
if run:
    cap = cv2.VideoCapture(0)

    st.text("Press 'Stop' to end webcam stream.")

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame.")
            break

        results = model(frame)[0]

        # Draw detections
        annotated_frame = results.plot()

        # Convert to RGB for Streamlit
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(annotated_frame)

    cap.release()
else:
    st.write('Webcam is stopped.')
