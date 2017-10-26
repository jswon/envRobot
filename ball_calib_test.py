# import segmentation_graph
import tensorflow as tf
import cv2
import numpy as np
from Kinect_Snap import Kinect
import urx
import time
from PIL import Image as Im

cam = Kinect()

color_map = [
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [0, 0, 0],
  ]

PI = np.pi
HOME = (90 * PI / 180, -90 * PI / 180, 0, -90 * PI / 180, 0, 0)
INITIAL_POSE = (0, -90 * PI / 180, 0, -90 * PI / 180, 0, 0)
starting_pose = (1.2345331907272339, -1.776348892842428, 1.9926695823669434, -1.7940548102008265, -1.5028379599200647, -0.41095143953432256)

global_cam = Kinect()
rob = urx.Robot("192.168.0.31")
sess = tf.InteractiveSession()

# segmentation_model = segmentation_graph.SegmentationGraph('segmentation_model\\')
idx = 1

def getPoints(label_image_array, object_index):
    return np.argwhere(label_image_array == object_index)

while True:
    with tf.Session() as sess:
        rob.movej(HOME, acc =1, vel=1)  # Move to center
        # time.sleep(5)

        img, depth_f = global_cam.snap()  # snapshot

        small_img = np.copy(img)

        cv2.imwrite("origin_img.png", img)

        # Filter
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_blue = np.array([130, 88, 165])  ##
        upper_blue = np.array([180, 255, 255])  # FIX
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        result = cv2.bitwise_and(img, img, mask=mask)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        ret, result = cv2.threshold(result, 100, 255, cv2.THRESH_BINARY)

        padding = np.zeros((28, 256), dtype=np.uint8)
        input_img = np.vstack((padding, result))

        cv2.imwrite("seg_img.png", input_img)

        global_view = cv2.imread("view.bmp")

        pxl_list = getPoints(input_img, 255)

        for y, x in pxl_list:
            small_img[y-28, x] = np.array([255, 255, 255])

        mean_small = (np.sum(pxl_list, axis=0)/pxl_list.shape[0]).astype(np.uint8)

        small_img[mean_small[0] - 28, mean_small[1]] = np.array([255, 0, 0])

        cv2.imwrite("small_img.png", small_img)

        for y, x in pxl_list:
            g_x = int((255 - x) * 3.035 + 573)   # Width revision, rate : 3.03515625, offset : 573
            g_y = int((y + 127) * 3.1953125 + 143)       # Height revision, rate : 3.1953125, offset : 143
            global_view[g_y, g_x, :] = np.array([255, 255, 255])

        mean_pxl = np.mean(pxl_list, axis=0).astype(np.uint64)

        pxl_patch = []
        start_pxl = mean_pxl - np.array([2, 2])

        for i in range(5):
            for j in range(5):
                pxl_patch.append(start_pxl+np.array([i, j]))

        for y, x in pxl_patch:
            g_x = int((255 - x) * 3.035 + 573)   # Width revision, rate : 3.03515625, offset : 573
            g_y = int((y + 127) * 3.1953125 + 143)       # Height revision, rate : 3.1953125, offset : 143
            global_view[g_y, g_x, :] = np.array([0, 0, 255])

        cv2.imwrite("global_view.png", global_view)

        xyz_list = []
        [xyz_list.append(global_cam.color2xyz(depth_f, i)) for i in pxl_patch]
        xyz_list = np.array(xyz_list)

        nan_idx = np.sort(np.transpose(np.argwhere(np.isnan(xyz_list))[0::3])[0])

        for x in reversed(nan_idx):
            xyz_list = np.delete(xyz_list, x, 0)

        mean_xyz = np.mean(xyz_list, axis=0) + np.array([0, 0, + 0.05])

        rob.movej(INITIAL_POSE, acc =1, vel=1)  # Move to center
        rob.movej(starting_pose,acc =1, vel=1)  # Move to starting position,
        goal = np.append(mean_xyz, [0, -3.14, 0])  # Initial point
        rob.movel(goal, acc =1, vel=1)
