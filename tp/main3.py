import streamlit as st
import cv2
from ultralytics import YOLO
import requests
import os
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Function to load the YOLO model
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

# Function to predict objects in the image
def predict_image(model, image, conf_threshold, iou_threshold):
    # Predict objects using the model
    res = model.predict(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        device='cpu',
    )

    class_name = model.model.names
    classes = res[0].boxes.cls
    class_counts = {}

    # Count the number of occurrences for each class
    for c in classes:
        c = int(c)
        class_counts[class_name[c]] = class_counts.get(class_name[c], 0) + 1

    # Generate prediction text
    prediction_text = 'Predicted '
    for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
        prediction_text += f'{v} {k}'

        if v > 1:
            prediction_text += 's'

        prediction_text += ', '

    prediction_text = prediction_text[:-2]
    if len(class_counts) == 0:
        prediction_text = "No objects detected"

    # Convert the result image to RGB
    res_image = res[0].plot()
    res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)

    return res_image, prediction_text

# Function to capture and write videos from multiple cameras
def capture_and_write_videos(urls, model, conf_threshold, iou_threshold):
    # Create VideoCapture objects for each camera
    caps = [cv2.VideoCapture(url) for url in urls]

    # Check if all video captures are successfully opened
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            st.error(f"Error: Unable to open video stream for camera {i + 1}.")
            return

    # Create an empty space for streaming
    live_stream = st.empty()

    while True:
        frames = [cap.read()[1] for cap in caps]  # Read frames from all cameras

        # Combine frames horizontally
        combined_frame = np.hstack(frames)

        # Predict objects in the combined frame
        _, text = predict_image(model, combined_frame, conf_threshold, iou_threshold)

        # Display the prediction text
        st.success(text)

        # Update the live stream space
        live_stream.image(combined_frame, caption="Live Stream", use_column_width=True)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    # Release the VideoCaptures
    for cap in caps:
        cap.release()


        
def main():
    # Model selection
    model_type = st.radio("Select Model Type", ("Fire Detection", "General"), index=0)

    models_dir = "general-models" if model_type == "General" else "fire-models"
    model_files = [f.replace(".pt", "") for f in os.listdir(models_dir) if f.endswith(".pt")]

    selected_model = st.selectbox("Select Model Size", sorted(model_files), index=2)

    # Load the selected model
    model_path = os.path.join(models_dir, selected_model + ".pt")
    model = load_model(model_path)

    # Set confidence and IOU thresholds
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.20, 0.05)
    iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.5, 0.05)

    # Camera streaming
    num_cameras = st.number_input("Number of Cameras", min_value=1, max_value=10, value=2, step=1)
    camera_urls = []
    for i in range(num_cameras):
        camera_url = st.text_input(f"Camera {i + 1} URL:")
        camera_urls.append(camera_url)

    start_button = st.button("Start Camera Streaming")
    if start_button:
        with st.spinner("Starting Camera Streaming..."):
            capture_and_write_videos(camera_urls, model, conf_threshold, iou_threshold)

if __name__ == "__main__":
    main()
