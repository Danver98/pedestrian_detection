import numpy as np
import cv2
import pandas as pd

cap = cv2.VideoCapture(
    "cars_day.mp4")
frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

width = int(width)
height = int(height)
print(frames_count, fps, width, height)


sub = cv2.createBackgroundSubtractorMOG2()  # create background subtractor # !!!! CHECK THIS FUNC
# information to start saving a video file
ret, frame = cap.read()  # import image
ratio = 1.0  # resize ratio
image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
width2, height2, channels = image.shape
#video = cv2.VideoWriter('traffic_counter.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (height2, width2), 1)

while True:
    ret, frame = cap.read()  # import image
    if not ret:  # if vid finish repeat
        frame = cv2.VideoCapture(# !!!! CHECK THIS FUNC
            "cars_day.mp4")
        continue
    if ret:  # if there is a frame continue with code
        image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
        cv2.imshow("image", image)  # @
        # converts image to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # !!!! CHECK THIS FUNC
        cv2.imshow("gray", gray)  # @
        fgmask = sub.apply(gray)  # uses the background subtraction # !!!! CHECK THIS FUNC
        cv2.imshow("fgmask", fgmask)  # @
        # applies different thresholds to fgmask to try and isolate cars
        # just have to keep playing around with settings until cars are easily identifiable
        # kernel to apply to the morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) # !!!! CHECK THIS FUNC
        closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("closing", closing)  # @
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        cv2.imshow("opening", opening)  # @
        dilation = cv2.dilate(opening, kernel) # !!!! CHECK THIS FUNC
        cv2.imshow("dilation", dilation)  # @
        retvalbin, bins = cv2.threshold( # !!!! CHECK THIS FUNC
            dilation, 220, 255, cv2.THRESH_BINARY)  # removes the shadows
        cv2.imshow("retvalbin", retvalbin)  # @
        # creates contours
        # cv2.imshow('bins',bins)
        contours, hierarchy = cv2.findContours( # !!!! CHECK THIS FUNC
            bins, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        minarea = 400
        # max area for contours, can be quite large for buses
        maxarea = 50000
        # vectors for the x and y locations of contour centroids in current frame
        cxx = np.zeros(len(contours))
        cyy = np.zeros(len(contours))

        for i in range(len(contours)):  # cycles through all contours in current frame
            # using hierarchy to only count parent contours (contours not within others)
            if hierarchy[0, i, 3] == -1:
                area = cv2.contourArea(contours[i])  # area of contour
                if minarea < area < maxarea:  # area threshold for contour
                    # calculating centroids of contours
                    cnt = contours[i]
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    # gets bounding points of contour to create rectangle
                    # x,y is top left corner and w,h is width and height
                    x, y, w, h = cv2.boundingRect(cnt)
                    # creates a rectangle around contour
                    cv2.rectangle(image, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)
                    # Prints centroid text in order to double check later on
                    # cv2.putText(image, str(cx) + "," + str(cy), (cx + 10,
                    #                                              cy + 10), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255), 1)
                    cv2.drawMarker(image, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS,
                                   markerSize=8, thickness=3, line_type=cv2.LINE_8)
    cv2.imshow("countours", image)
    key = cv2.waitKey(20) # !!!! CHECK THIS FUNC
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
