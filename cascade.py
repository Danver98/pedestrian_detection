import cv2, sys, imutils
import numpy as np
from videopaths import video_paths
#second variant
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Initiate video capture for video file
number = 0
if (len(sys.argv) > 1):
    number = int(sys.argv[1])
cap = cv2.VideoCapture(video_paths[number-1])

# Loop once video is successfully loaded
while cap.isOpened():
    
    # Read first frame
    ret, frame = cap.read()
    #frame = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    # here we are resizing the frame, to half of its size, we are doing to speed up the classification
    # as larger images have lot more windows to slide over, so in overall we reducing the resolution
    #of video by half that’s what 0.5 indicate, and we are also using quicker interpolation method that is #interlinear
    frame = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    frame = imutils.resize(frame,  
                               width=min(400, frame.shape[1]))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.2, 5, maxSize=(50,100))
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Pedestrians', frame)

    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()