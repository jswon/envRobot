"""
For scenario, Demo

Shuffle, get obj , movel , get obj_point
"""
# Env Composition
import socket
import urx
import Kinect_Snap
import serial
from ur_safety import *
# utils
# from util import *
import random
import cv2
import copy
import math
import datetime
import numpy as np


# ----------------- Define -----------------------------------------------------
PI = np.pi
HOME = (90 * PI / 180, -90 * PI / 180, 0, -90 * PI / 180, 0, 0)
INITIAL_POSE = (0, -90 * PI / 180, 0, -90 * PI / 180, 0, 0)
shf_way_pt = np.array([[-0.82222461061452856, -1.5587535549561358, -2.0142844897266556, -1.0569713950077662, 1.5327481491014201, 0.23544491609403506],
                       [-1.5591026208065346, -0.87423542232395968, 0.88383473320992845, -1.5660839378145119, 4.6582837735728653, 1.5947073375472192],
                       [-1.524196035766648, -0.79656827061021196, 1.0779153460316979, -1.8202038769048863, 4.6553167138444751, 1.5924384095196262],
                       [-1.535017077129013, -1.1815879036001613, 1.0709340290237206, -1.557706357404939, 4.6504297919388904, 1.5929620082952245],
                       [-1.4950490372583425, -1.5502014416213632, 1.0733774899765127, -1.3962634015954636, 4.6293113079897603, 1.6215854080279315],
                       [-1.5042992822939125, -1.2472122834751478, 1.073901088752111, -1.5341444125030159, 4.6266933141117681, 1.6214108751027323],
                       [-1.5100588688254937, -0.79656827061021196, 1.0751228192285072, -1.8051940453377351, 4.6223299909817817, 1.6214108751027323],
                       [-1.5334462808022178, -1.3646729421343662, 1.011069235680315, -1.7208946424664087, 4.6116834825446169, 1.5678292670665062]])
# -------------------------------------------------------------------------------


def add_opts(parser):
    parser.add_argument('--event-log-out', type=str, default=None, help="path to record event log.")
    parser.add_argument('--num-cameras', type=int, default=2, help="how many camera points to render; 1 or 2")
    parser.add_argument('--action-repeats', type=int, default=1, help="number of action repeats")
    parser.add_argument('--max-episode-len', type=int, default=200, help="maximum episode len for cartpole")
    parser.add_argument('--use-raw-pixels', default=True, action='store_true', help="use raw pixels as state instead of cart/pole poses")
    parser.add_argument('--render-width', type=int, default=256, help="if --use-raw-pixels render with this width")
    parser.add_argument('--render-height', type=int, default=256, help="if --use-raw-pixels render with this height")
    parser.add_argument('--reward-calc', type=str, default='fixed',
                        help="'fixed': 1 per step. 'angle': 2*max_angle - ox - oy. 'action': 1.5 - |action|. 'angle_action': both angle and action")


class envRobot:
    def __init__(self, socket_ip):
        # Connect to Environment
        # self.gripper = pyGrip.gripper(host=SOCKET_IP)             # Gripper
        self.rob = urx.Robot(socket_ip)                             # Robot
        # self.safety = safety_chk(host=socket_ip)                    # Robot - Dashboard ( for collision check)
        # self.bluetooth = serial.Serial("COM5", 9600, timeout=1)     # Tray
        self.robo = socket.create_connection((socket_ip, 30002), timeout=0.5)


        # Camera interface
        self.global_cam = Kinect_Snap.Kinect()                  # Kinect Camera

        # Robot
        self.acc = 1.5
        self.vel = 1.5
        self.r2s = np.deg2rad([-90, 90, 0, 90, 0, 0])
        self.default_tcp = [0, 0, 0.1485, 0, 0, 0]  # (x, y, z,, rx, ry, rz)


        # object position
        self.obj_pos = np.zeros([3])

        self.set_tcp(self.default_tcp)
        self.movej(HOME)

        print("Robot Environment Ready.", file=sys.stderr)

    def reset(self):
        # Robot Reset
        self.movej(HOME)
        self.obj_pos = self.get_obj_pos()
        self.movej(INITIAL_POSE)    # Move to center

    def step(self, action, target_obj, num_data, exploration_noise, is_training):
        action = np.append(action, [0, 0])

        current_angle = self.getj()              # radian
        target_angle = current_angle + action

        if is_training:
            noise = exploration_noise.noise()
            noise[0] /= 2.0
        else:
            noise = np.zeros(4)

        noise[3] *= math.radians(45)
        noise[2] *= math.radians(45)
        noise[1] *= math.radians(90)
        noise[0] *= math.radians(90)

        max_ori = np.radians(np.array([-10.0, -75.0, 0.0, 0]))
        min_ori = np.radians(np.array([-135.0, -135.0, -150.0, -90]))

        for idx, (maxx, minn) in enumerate(zip(max_ori, min_ori)):
            if target_angle[idx] < minn:
                action[idx] = abs(target_angle[idx] - minn) + action[idx] + abs(noise[idx])
                target_angle[idx] = current_angle[idx] + action[idx]
            elif target_angle[idx] > maxx:
                action[idx] = action[idx] - abs(target_angle[idx] - maxx) - abs(noise[idx])
                target_angle[idx] = current_angle[idx] + action[idx]

        target_angle[4] = math.radians(90)
        target_angle[5] = 0

    def getj(self):
        return np.around(np.array(self.rob.getj()), decimals=4)

    def get_ef(self):
        return np.around(np.array(self.rob.getl()[0:3]), decimals=4)

    def movejo(self, goal_pose, r=0):
        p_1 = "def myProg():\nmovej([{},{},{},{},{},{}], a=1.7, v=1.7, t=0, r={})\nend\nmyProg()\n".format(*goal_pose, r)
        self.robo.send(p_1.encode())

    def movej(self, goal_pose, acc=1.7, vel=1.7, wait = True, relative = False, threshold = None):
        self.rob.movej(goal_pose, acc, vel, wait = wait, relative = relative, threshold = threshold)

    def movel(self, goal_pose, acc=1.7, vel=1.7):
        self.rob.movel(goal_pose, acc, vel)

    def shuffle_obj(self):
        self.movej(HOME)

        pt = [[0.1507703842858799, -0.3141178727128849, -0.030569762928828032, 2.2260958131016335, -2.1985942225522668, 0.05813081679518341],
              [-0.1491136865872555, -0.31021647495551335, -0.03052286866235667, -2.1570671726727104, 2.2799584355377167, -0.029214990891798114],
              [0.18414105144817525, -0.25296564835782714, -0.021194048759978847, -2.238562873918888, 2.1370170873680085, 0.10380902269948922],
              [-0.1417132578497845, -0.26528795089888607, -0.03157633192772823, -2.1045306382633795, 2.243489988356314, 0.0360411778983116],
              [0.20295675954768064, -0.16877330661979667, -0.038960103050602234, 0.018988621965437776, -3.007839525712047, -0.875295581190713],
              [-0.13352761097358432, -0.16247247277870452, -0.025328902795204018, 0.15436650469465374, -2.6904628274570306, -1.555752726987423]]

        # TRAY
        self.bluetooth.write("1".encode())

        region = self._obj_region_chk(*self.get_obj_pos()[:-1])  # Center : 1, 2, 3
        random.seed(datetime.datetime.now())
        random_dir = random.randrange(0, 3)

        if region in [0, 1, 2]:
            if random_dir == 0:
                self.movej([1.2345331907272339, -1.776348892842428, 1.9926695823669434, -1.7940548102008265, -1.5028379599200647, -0.41095143953432256])
                self.movel(pt[region*2])
                self.movel(pt[region*2+1])
                self.movej([1.27391582, -1.683021, 2.22669106, -2.14867484, -1.56398954, -0.33615041])
            elif random_dir == 1:
                self.movej([1.2345331907272339, -1.776348892842428, 1.9926695823669434, -1.7940548102008265, -1.5028379599200647, -0.41095143953432256])
                self.movel(pt[region * 2 + 1])
                self.movel(pt[region * 2])
                self.movej([1.27391582, -1.683021, 2.22669106, -2.14867484, -1.56398954, -0.33615041])
            else:
                pass

        else:
            pass

        self.movej(HOME)

        time.sleep(3)  # waiting

    def get_camera(self, camera_num):
        if camera_num == 1:
            return self.global_cam.snap()

    @staticmethod
    def _obj_region_chk(x, y):
        y = abs(y)
        if - 0.08 <= x <= 0.08:
            if y >= 0.287:
                return 0
            elif 0.2 <= y < 0.287:
                return 1
            elif y <= 0.2:
                return 2
        elif x > 0.08:
            return 3
        elif x < -0.08:
            return 4

    def get_obj_pos(self):
        # TODO : Reserved, Segmentation
        img, depth = self.global_cam.snap()
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_blue = np.array([114, 75, 85])    # Threshold
        upper_blue = np.array([180, 255, 255])  # Threshold
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        result = cv2.bitwise_and(img, img, mask=mask)
        ret, thresh = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
        blurred = cv2.medianBlur(thresh, 5)
        blurred = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        th3 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        _, contours, hierarchy = cv2.findContours(th3, 1, 2)
        max_radius = 0

        # Center of Object on Image
        cx = 0
        cy = 0

        for cnt in contours:
            if cv2.contourArea(cnt) < 60000:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)

                if radius > max_radius:
                    max_radius = radius

        # pixel to robot coordinate
        obj_y = (2.9579207920792 * cy - 405.33415841584) / 1000
        obj_x = (401.5000081602 - 3.1829680283322 * cx) / 1000
        obj_z = 50 / 1000

        return np.array([cx, cy])

        return np.array([obj_x, obj_y, obj_z], dtype=np.float32)

    def set_tcp(self, tcp):
        self.rob.set_tcp(tcp)
