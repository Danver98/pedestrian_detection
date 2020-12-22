# import the necessary packages
import numpy as np
import cv2
import imutils
from nms import non_max_suppression_fast
# initialize the HOG descriptor/person detector
video_paths = ["D:\Programming\ImageProcessing\АОМТ-2\people1.mp4","D:\Programming\ImageProcessing\АОМТ-2\street0.mp4",
"D:\Programming\ImageProcessing\АОМТ-2\street1.mp4","D:\Programming\ImageProcessing\АОМТ-2\street2.mp4"]
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cv2.startWindowThread()
# open webcam video stream/file
cap = cv2.VideoCapture(video_paths[1])
# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    original_shape = frame.shape
    factor = frame.shape[1]/min(400, frame.shape[1])
    frame = imutils.resize(frame,  
                               width=min(400, frame.shape[1])) 
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(gray, winStride=(4,4),padding=(8, 8), scale = 1.01 )
    """
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    picks = non_max_suppression_fast(boxes, overlapThresh=0.65)
    for (xA, yA, xB, yB) in picks:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
    """
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Write the output video 
    out.write(frame.astype('uint8'))
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
