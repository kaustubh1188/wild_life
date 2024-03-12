import streamlit as st
import cv2
from ultralytics import YOLO
import requests
from PIL import Image
import os
from numpy import random
import io
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

    # Calculate inference latency
    latency = sum(res[0].speed.values())  # in ms, need to convert to seconds
    latency = round(latency / 1000, 2)
    prediction_text += f' in {latency} seconds.'

    # Convert the result image to RGB
    res_image = res[0].plot()
    res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)

    return res_image, prediction_text

# Function to capture and write videos from multiple cameras
def capture_and_write_videos(urls, output_paths, model, conf_threshold, iou_threshold):
    # Create VideoCapture objects for each camera
    caps = [cv2.VideoCapture(url) for url in urls]

    # Check if all video captures are successfully opened
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            st.error(f"Error: Unable to open video stream for camera {i + 1}.")
            return

    # Get the video frame width and height from the first camera
    frame_width = int(caps[0].get(3))
    frame_height = int(caps[0].get(4))

    # Define the codec and create a VideoWriter object for each camera
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs like MJPG, XVID, etc.
    outs = [cv2.VideoWriter(output_paths[i], fourcc, 20.0, (frame_width, frame_height)) for i in range(len(caps))]

    # Create a named window with normal size
    cv2.namedWindow('Live Stream', cv2.WINDOW_NORMAL)

    while True:
        frames = [cap.read()[1] for cap in caps]  # Read frames from all cameras

        # Combine frames horizontally
        combined_frame = np.hstack(frames)

        # Predict objects in the combined frame
        _, text = predict_image(model, combined_frame, conf_threshold, iou_threshold)

        # Display the combined frame with prediction text
        st.image(combined_frame, caption="Live Stream", use_column_width=True)
        st.success(text)

        # Show the live stream in a separate OpenCV window
        cv2.imshow('Live Stream', combined_frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    # Release the VideoWriters and VideoCaptures
    for out in outs:
        out.release()

    for cap in caps:
        cap.release()

    # Close the named window
    cv2.destroyAllWindows()

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

    # Image selection
    image = None
    image_source = st.radio("Select image source:", ("Enter URL", "Upload from Computer"))
    if image_source == "Upload from Computer":
        # File uploader for image
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        else:
            image = None

    else:
        # Input box for image URL
        url = st.text_input("Enter the image URL:")
        if url:
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    image = Image.open(response.raw)
                else:
                    st.error("Error loading image from URL.")
                    image = None
            except requests.exceptions.RequestException as e:
                st.error(f"Error loading image from URL: {e}")
                image = None

    if image:
        # Display the uploaded image
        with st.spinner("Detecting"):
            prediction, text = predict_image(model, image, conf_threshold, iou_threshold)
            st.image(prediction, caption="Prediction", use_column_width=True)
            st.success(text)

        prediction = Image.fromarray(prediction)

        # Create a BytesIO object to temporarily store the image data
        image_buffer = io.BytesIO()

        # Save the image to the BytesIO object in PNG format
        prediction.save(image_buffer, format='PNG')

        # Create a download button for the image
        st.download_button(
            label='Download Prediction',
            data=image_buffer.getvalue(),
            file_name='prediction.png',
            mime='image/png'
        )

    # Add a section divider
    st.markdown("---")

    # Camera streaming
    num_cameras = st.number_input("Number of Cameras", min_value=1, max_value=10, value=2, step=1)
    camera_urls = []
    for i in range(num_cameras):
        camera_url = st.text_input(f"Camera {i + 1} URL:")
        camera_urls.append(camera_url)

    output_paths = [f"output_camera_{i + 1}.avi" for i in range(num_cameras)]
    start_button = st.button("Start Camera Streaming")
    if start_button:
        with st.spinner("Starting Camera Streaming..."):
            capture_and_write_videos(camera_urls, output_paths, model, conf_threshold, iou_threshold)

if __name__ == "__main__":
    main()
