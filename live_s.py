import cv2
import numpy as np

# Define the URLs for the cameras
urls = ["http://10.10.13.36:8080/video", "http://10.10.13.24:8080/video"]

# Create VideoCapture objects for each camera
caps = [cv2.VideoCapture(url) for url in urls]

# Check if all video captures are successfully opened
for i, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"Error: Unable to open video stream for camera {i + 1}.")
        exit()

# Get the video frame width and height from the first camera
frame_width = int(caps[0].get(3))
frame_height = int(caps[0].get(4))

# Define the codec and create a VideoWriter object for each camera
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs like MJPG, XVID, etc.
outs = [cv2.VideoWriter(f'output_camera_{i + 1}.avi', fourcc, 20.0, (frame_width, frame_height)) for i in range(len(caps))]

# Create a named window with normal size
cv2.namedWindow('Grid', cv2.WINDOW_NORMAL)

while True:
    frames = [cap.read()[1] for cap in caps]  # Read frames from all cameras

    # Arrange frames in a 2x2 grid
    top_row = np.hstack((frames[0], frames[1]))
    combined_frame = np.vstack((top_row, top_row))

    cv2.imshow('Grid', combined_frame)

    # Write the combined frame to the output video file for each camera
    for i, out in enumerate(outs):
        out.write(frames[i])

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