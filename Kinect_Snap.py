# import os
# import sys
# file_directory = os.getcwd()+ "\\PyKinect2-master\\pykinect2"
# sys.path.append(file_directory)
import PyKinectV2
import PyKinectRuntime
import sys

from PIL import Image as Im

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

class take_snap(object):
    def __init__(self):
        # Kinect runtime object, we want only color
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)

    def snap(self):
        while(True):
            if self._kinect.has_new_color_frame():
                frame = self._kinect.get_last_color_frame()
                result_image = Im.frombuffer("RGBA", (self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), frame, "raw", "RGBA", 0, 1)
                b, g, r, a = result_image.split()
                result_image = Im.merge("RGBA", (r, g, b, a))
                result_image = result_image.transpose(Im.FLIP_LEFT_RIGHT)

                return result_image

    def endthekinect(self):
        self._kinect.close()

