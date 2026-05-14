# Driver Fatigue Detection using Human Pose / Facial Landmark Estimation and attention mechanism

This project implements a real-time driver monitoring prototype for detecting drowsiness, yawning, and eye blinking using computer vision. The system uses OpenCV, Dlib facial landmarks, and geometric facial measurements to monitor driver fatigue-related behavior and generate a voice alert when unsafe signs are detected.

The main purpose of this project is to support driver safety by identifying early signs of fatigue and distraction, such as prolonged eye closure, frequent blinking, and yawning. Simple code in python to detect Drowsiness and Yawn and alert the user using Dlib.

## Project Overview

Driver fatigue and drowsiness are major causes of road accidents. This project uses a camera-based approach to monitor the driver’s face in real time. The system detects facial landmarks using Dlib’s 68-point landmark model and calculates important fatigue indicators such as Eye Aspect Ratio (EAR) and mouth opening distance.

When the driver’s eyes remain closed for a specific number of frames, the system classifies it as drowsiness and gives a voice alert. Similarly, when the mouth opening crosses a defined threshold, the system detects yawning.

## Main Features

- Real-time face detection using OpenCV
- Facial landmark detection using Dlib 68-point predictor
- Eye closure detection using Eye Aspect Ratio (EAR)
- Yawn detection using mouth landmark distance
- Blink counting
- Voice alert using `pyttsx3`
- Webcam-based real-time testing
- light-weight attention-mechanism implementation
- deep feature fusion
- Adjustable threshold values for different users and camera distances

## Project Workflow

The system follows this basic workflow:

1. Capture video from webcam or input video.
2. Detect the driver’s face in each frame.
3. Extract facial landmarks using Dlib.
4. Calculate eye and mouth measurements.
5. Detect drowsiness, yawning, and blinking.
6. Generate an alert if unsafe behavior is detected.

## Files in This Repository

```text
drowsiness_yawn.py                        Main drowsiness and yawn detection script
main.py                                   Additional project implementation file
new.py                                    Testing/experimental script
haarcascade_frontalface_default.xml       Face detection cascade file
shape_predictor_68_face_landmarks.dat     Dlib facial landmark model
README.md                                 Project documentation
Hardware Implementation                   Hardware imeplemntation for OAK-D & Jetson Nano           

## Dependencies

1. Python 3.7 (must)
2. pyttsx3
3. scipy
4. numpy
5. imutils
6. argparse
7. cmake
8. opencv
9. dlib
```
https://youtu.be/9S_c67AipYE?si=lres_z0Cf9gJmKf2		//Dlib Installation Guide
```

## Run 

```
Python3.7 drowsiness_yawn.py -- webcam 0		//For external webcam, use the webcam number accordingly
```