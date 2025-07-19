# app.py
import cv2
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO

# Load YOLO model once
model = YOLO("yolov8n.pt")  # Replace with your custom model path if needed

st.title("🎥 Real-Time Object Detection with YOLOv8")

class ObjectDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run detection
        results = model(img)
        annotated_frame = results[0].plot()

        return annotated_frame

# Start webcam stream
webrtc_streamer(
    key="object-detection",
    video_transformer_factory=ObjectDetectionTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
