# Drowsiness and Yawn detection with voice alert using Dlib

Simple code in python to detect Drowsiness and Yawn and alert the user using Dlib.

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

## Setups

Change the threshold values according to your need
```
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 40
YAWN_THRESH = 20	        //change this according to the distance from the camera
BLINK_THRESH = 20
```



