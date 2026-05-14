from ultralytics import YOLO
import cv2
from scipy.spatial import distance as dist
import numpy as np
import threading
import pyttsx3
import depthai as dai
import os
import time

# Initialize pyttsx3 engine
engine = pyttsx3.init()
engine.setProperty('rate', 110)  # Speed of speech
engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)

# Global flag to track speaking state
speaking = False

# Define callback functions
def on_start(name):
    global speaking
    speaking = True
    print(f'Starting: {name}')

def on_end(name, completed):
    global speaking
    speaking = False
    print(f'Finished: {name}, Completed: {completed}')

# Connect callbacks to events
engine.connect('started-utterance', on_start)
engine.connect('finished-utterance', on_end)

# Function to speak text in a separate thread
def speak(text):
    global speaking
    if not speaking:
        threading.Thread(target=lambda: engine.say(text) or engine.runAndWait()).start()

# Define paths
model_path = "best.pt"

# Initialize OAK-D pipeline
pipeline = dai.Pipeline()
cam_rgb = pipeline.createColorCamera()
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
xout_video = pipeline.createXLinkOut()
xout_video.setStreamName("video")
cam_rgb.video.link(xout_video.input)

# Load DriPE pose estimation model
model = YOLO(model_path)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue(name="video", maxSize=30, blocking=False)

    # Initialize variables for FPS calculation
    start_time = time.time()
    frame_count = 0

    while True:
        video_frame = video_queue.get()
        frame = video_frame.getCvFrame()

        # Pose estimation
        results = model(source=frame, show=False, conf=0.7, save=False)
        x1, y1, x2, y2 = 251, 180, 368, 380

        for result in results:
            keypoints = result.keypoints.xy
            boxes = result.boxes
            img = result.orig_img
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for n, i in enumerate(keypoints[0]):
                x, y = int(i[0]), int(i[1])
                left_wrist = keypoints[0][9]
                right_wrist = keypoints[0][10]
                lwx, lwy = int(left_wrist[0]), int(left_wrist[1])
                rwx, rwy = int(right_wrist[0]), int(right_wrist[1])
                cv2.circle(img, (x, y), 1, (0, 255, 0), 2)

                # Check if wrists are outside the rectangle
                if n == 9 or n == 10:
                    if not (x1 < lwx < x2 and y1 < lwy < y2):
                        print("Danger")
                        cv2.circle(img, (lwx, lwy), 1, (0, 0, 255), 2)
                        cv2.putText(img, "Left Wrist is not on Steering", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        speak("Left wrist is not on steering")
                    if not (x1 < rwx < x2 and y1 < rwy < y2):
                        print("Danger")
                        cv2.circle(img, (rwx, rwy), 1, (0, 0, 255), 2)
                        cv2.putText(img, "Right Wrist is not on Steering", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        speak("Right wrist is not on steering")

        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        # Display FPS on frame
        cv2.putText(frame, f'FPS: {fps:.2f}', (frame_width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Resize frame for display
        frame_resized = cv2.resize(frame, (640, 360))

        # Display the frame
        cv2.imshow("Frame", frame_resized)
        cv2.moveWindow("Frame", 450, 0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

