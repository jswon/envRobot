import PyKinectV2
import PyKinectRuntime
import cv2
import numpy as np
import numpy.linalg as lin


class global_cam(object):
    def __init__(self):
        # Kinect runtime object
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                                                       PyKinectV2.FrameSourceTypes_Depth)

        self.RotationMat_RGB_to_IR = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.Transitionvec_RGB_to_IR = np.array([-15.0 * 0.001, +7.0 * 0.001, -26.0 * 0.001]).T
        self.focalLengthX_Color = 1123.872
        self.focalLengthY_Color = 1135.098
        self.focalLengthX_IR = 383.094
        self.focalLengthY_IR = 385.131
        self.centerX_Color = 976.473
        self.centerY_Color = 548.871
        self.centerX_IR = 257.407
        self.centerY_IR = 212.489

        self.Color_Intrinsic_MAT = np.array(
            [[self.focalLengthX_Color, 0, self.centerX_Color], [0, self.focalLengthY_Color, self.centerY_Color],
             [0, 0, 1]])

        self.IR_Intrinsic_MAT = np.array(
            [[self.focalLengthX_IR, 0, self.centerX_IR], [0, self.focalLengthY_IR, self.centerY_IR], [0, 0, 1]])

        self.Camera_to_RobotArm_Matrix = np.array(
            [[-0.9994, 0.0220, 0.0251, 0.0013], [0.0273, 0.9720, 0.2336, -0.1258], [-0.0193, 0.2341, -0.9720, 1.0000],
             [0, 0, 0, 1]])

        self.depthframe = None
        self.cloudpointArray = None

    def snap(self):  # Cropped Tray Input IMG
        while(True):
            if self._kinect.has_new_color_frame():
                raw_array = self._kinect.get_last_color_frame()

                raw_img = raw_array.reshape((1080, 1920, 4))  # to Mat
                flipped_img = cv2.flip(raw_img, 1)            # Flipped Image
                # cropped_img = flipped_img[105:905, 550:1330]   # cropped ROI, Global View
                cropped_img = flipped_img[55:1000, 400:1400]   # cropped ROI, Global View
                result_img = cv2.resize(cropped_img, (256, 256))  # resized image (256,256) RGBA
                result_img = cv2.cvtColor(result_img, cv2.COLOR_RGBA2RGB) # Foramt : RGB
                cv2.imshow("result_image", flipped_img)

                return result_img

    def pixel2robot(self, RGB_pixel):
        # it must conduct 'self.snapdepth2' previously.
        # And when you get a pixel position from rgb image, you must get pixel position from rgb image that didn't use flipflop.
        # input : RGB_pixel = [pixel_x ,pixel_y]
        # output : result = [X,Y,Z,1]

        depthframe = self.snapdepth2()

        RGB_pixel = np.array([int(RGB_pixel[0]), int(RGB_pixel[1]), 1])
        RGB_3d = np.dot(lin.inv(self.Color_Intrinsic_MAT), RGB_pixel.T)

        IR_3d = np.dot(self.RotationMat_RGB_to_IR, RGB_3d) + self.Transitionvec_RGB_to_IR

        IR_pixel = np.dot(self.IR_Intrinsic_MAT, IR_3d)

        IR_pixel = IR_pixel.T
        IR_pixel[0] = IR_pixel[0] / (IR_pixel[2])
        IR_pixel[1] = IR_pixel[1] / (IR_pixel[2])

        Z = depthframe[int(IR_pixel[1]), int(IR_pixel[0])] * 0.001
        X = (RGB_pixel[0] - self.centerX_Color) * Z / self.focalLengthX_Color
        Y = (RGB_pixel[1] - self.centerY_Color) * Z / self.focalLengthX_Color

        X = -X + 0.03
        Y = Y
        Z = Z * 0.81

        result = np.dot(lin.inv(self.Camera_to_RobotArm_Matrix), np.array([X, Y, Z, 1]).T).tolist()

        return result

    def snapdepth2(self):
        while (True):
            if self._kinect.has_new_depth_frame():
                # import depthframe
                depth = self._kinect.get_last_depth_frame()
                # Change depthframe to array(row: 424,column : 512)
                depthframe = np.asarray(depth)
                depthframe = self.depthframe.reshape((self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width))

                return depthframe
            #
            #     break
            # else:
            #     pass

    def endthekinect(self):
        self._kinect.close()