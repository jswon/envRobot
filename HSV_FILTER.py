import Kinect_Snap
import numpy as np
import cv2

def nothing(x):
    pass

global_cam = Kinect_Snap.global_cam()  # Load Camera

img = global_cam.snap()

hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

cv2.imshow("hsv", hsv)

cv2.namedWindow('result')

cv2.createTrackbar('h', 'result',0,179,nothing)
cv2.createTrackbar('s', 'result',0,255,nothing)
cv2.createTrackbar('v', 'result',0,255,nothing)

while(1):
    # get info from track bar and appy to result
    h = cv2.getTrackbarPos('h','result')
    s = cv2.getTrackbarPos('s','result')
    v = cv2.getTrackbarPos('v','result')

    # Normal masking algorithm
    lower_blue = np.array([h,s,v])
    upper_blue = np.array([180,255,255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    result = cv2.bitwise_and(img, img, mask = mask)

    cv2.imshow('result',result)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break