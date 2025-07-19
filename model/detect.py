from ultralytics import YOLO
import cv2
import torch

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
model = YOLO("yolov8s.pt")  # Or yolov8n.pt for speed

# Open webcam (0 is default camera)
cap = cv2.VideoCapture(0)

# Set resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    results = model(frame, device=device)

    # Plot results on the frame
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("Webcam Detection", annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
