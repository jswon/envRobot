import PyKinectV2
import PyKinectRuntime
import sys
import cv2
import numpy as np

class global_cam(object):
    def __init__(self):
        # Kinect runtime object, we want only color
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)

    def snap(self):
        while(True):
            if self._kinect.has_new_color_frame():
                raw_array = self._kinect.get_last_color_frame()

                raw_img = raw_array.reshape((1080, 1920, 4))  # to Mat
                flipped_img = cv2.flip(raw_img, 1)            # Flipped Image
                cropped_img = flipped_img[153:910, 580:1330]   # cropped ROI, Global View
                result_img = cv2.resize(cropped_img, (256, 256))  # resized image (256,256) RGBA
                result_img = cv2.cvtColor(result_img, cv2.COLOR_RGBA2RGB) # Foramt : RGB
                return result_img