import sys
import urx
import pyGrip
import serial
import numpy as np
import ur_safety
import pyueye
import Kinect_Snap
from util import *
import random
import cv2
import copy
import math
import argparse
from datetime import datetime

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

objects_name = ['O_00_Big_USB', 'O_01_Black_Tape', 'O_02_Blue_Glue', 'O_03_Brown_Box', 'O_04_Green_Glue',
                    'O_05_Pink_Box', 'O_06_Red_Cup', 'O_07_Small_USB', 'O_08_White_Tape', 'O_09_Yellow_Cup' ]


def add_opts(parser):
  parser.add_argument('--event-log-out', type=str, default=None, help="path to record event log.")
  parser.add_argument('--max-episode-len', type=int, default=200, help="maximum episode len for cartpole")
  parser.add_argument('--use-raw-pixels', default=True, action='store_true', help="use raw pixels as state instead of cart/pole poses")
  parser.add_argument('--render-width', type=int, default=256, help="if --use-raw-pixels render with this width")
  parser.add_argument('--render-height', type=int, default=256, help="if --use-raw-pixels render with this height")
  parser.add_argument('--reward-calc', type=str, default='fixed',
                      help="'fixed': 1 per step. 'angle': 2*max_angle - ox - oy. 'action': 1.5 - |action|. 'angle_action': both angle and action")

class envRobot :
    def __init__(self, SOCKET_IP, opts):
        # Connect to Environment
        # self.gripper = pyGrip.gripper(host=SOCKET_IP)             # Gripper
        self.rob = urx.Robot(SOCKET_IP)                             # Robot
        self.safety = ur_safety.safety_chk(host= SOCKET_IP)         # Robot - Dashboard ( for collision check)
        self.bluetooth = serial.Serial("COM5", 9600, timeout=1)     # Tray
        # self.local_cam = pyueye.uEyeCAM()                         # Kinect Camera
        self.global_cam = Kinect_Snap.global_cam()                  # Wrist(loca) Camera

        # Robot
        self.acc = 1.5; self.vel = 1.5

        # Variables
        self.opts = opts
        self.render_width = opts.render_width
        self.render_height = opts.render_height
        self.num_cameras = opts.num_cameras
        self.repeats = opts.action_repeats

        # decide observation space
        # in high dimensional case each observation is an RGB images (H, W, 3)
        # we have R repeats and C cameras resulting in (H, W, 3, R, C)
        # final state fed to network is concatenated in depth => (H, W, 3*R*C)
        self.state_shape = (self.render_height, self.render_width, 3,
                            self.num_cameras, self.repeats)

        self.state = np.empty(self.state_shape, dtype=np.float32)
        self.init_img = np.empty((self.render_height, self.render_width, 3))
        self.initial_state = np.empty(self.state_shape, dtype=np.float32)
        self.action_dim = opts.action_dim

        # object position
        self.obj_pos = [0, 0, 0]

        # Make directory for save Data Path
        dir_path = "E:\\save_data\\"
        self.path_element = [dir_path + "initial_image\\", dir_path + "current_image\\", dir_path + "obj_position\\", dir_path + "target_obj\\"]

        if not os.path.exists(dir_path):
            [os.makedirs(x) for x in self.path_element]

        print("Robot Environment Ready.", file = sys.stderr)


    def reset(self, target_obj, num_data, save_path):
        self.rob.movej(HOME,acc=1.5, vel=1.5)
        # degree = self.check_move_finish(HOME) # Maybe we don't need.

        self.init_img = self.get_camera(1)
        self.state[:, :, :, 1, 0] = self.init_img
        self.initial_state = self.state[:, :, :, 1, 0]

        self.obj_pos = self.get_obj_pos(target_obj)

        #TODO Save initial state.
        if num_data > 0:
            cv2.imwrite(self.path_element[0] + str(num_data) + '.bmp', self.img_proc(self.init_img))  # Save the initial state ??????????????

        self.rob.movej(INITIAL_POSE, 1.5, 1.5)

        self.state[:, :, :, 0, 0] = self.get_camera(1)

        if num_data > 0:
            cv2.imwrite(self.path_element[1] + str(num_data) + '.bmp', self.set_state_element(1))   # Save the current image

            file = open(self.path_element[2] + str(num_data) + '.txt', "w")
            file.write("%f %f %f %f" % (self.obj_pos[0], self.obj_pos[1], self.obj_pos[2], self.rob.getj()[0]))  # Save the obj_pos, [x, y, z, base_angle]
            file.close()

            file = open(self.path_element[3] + str(num_data) + '.txt', "w")
            pos = "%d %d %d %d %d %d %d %d %d %d" % (target_obj[0], target_obj[1], target_obj[2], target_obj[3],
                                                     target_obj[4], target_obj[5], target_obj[6], target_obj[7],
                                                     target_obj[8], target_obj[9])

            file.write(pos)
            file.close()

        obj_angle_rad = math.atan2(self.obj_pos[1], self.obj_pos[0])

        endEffector_pos = [round(x, 4) for x in self.rob.getl()[0:3]]  # unit : mili

        # TODO: Reserved Data
        #r_xyz = [round(x, 4) for x in raw[3:]] # Rotation

        epsilon = 0.0001

        ef_angle_rad = math.atan2(endEffector_pos[1] - 0, endEffector_pos[0] - 0) # Base joint angle = end effector pose to angle

        diff = np.array(ef_angle_rad) - np.array(obj_angle_rad)
        reward = 3.15 - abs(diff)

        self.cur_end_effector_pos = np.asarray(endEffector_pos)
        self.cur_joint_angles = self.rob.getj()

        # TODO END_EFFECTOR
        self.internal_state = self.get_internal_state()

        # return this state
        return np.copy(self.state), reward

    def step(self, action, target_object_index, target_obj, num_data, previous_reward, save_path):
        self.done = False
        action = list(action.reshape(-1, ))

        previous_joint_angle = self.rob.getj()

        target_joint_angle = previous_joint_angle[0] + action[0]

        epsilon = 0.00001

        cost = 0.0

        if target_joint_angle > math.radians(90) :
            target_joint_angle = math.radians(90)

        elif target_joint_angle < math.radians(-90) :
            target_joint_angle = math.radians(-90)

        action[0] = target_joint_angle

        action.append(-90 * PI / 180)
        action.append(0)
        action.append( -90 * PI / 180)
        action.append(0)
        action.append(0)

        self.rob.movej(action, 1.5, 1.5)
        current_angles = self.rob.getj()  # To degree
        # actual_action = current_angles - previous_joint_angle

        actual_action = [a_i - b_i for a_i, b_i in zip(current_angles, previous_joint_angle)]

        image = self.set_state_element(1)

        obj_pos_copy = copy.deepcopy(self.obj_pos)

        obj_angle_rad = math.atan2(obj_pos_copy[1], obj_pos_copy[0])
        obj_angle_deg = math.degrees(obj_angle_rad)

        endEffector_pos = [round(x, 4) for x in self.rob.getl()[0:3]]  # unit : mili

        ef_angle_rad = math.atan2(endEffector_pos[1] - 0, endEffector_pos[0] - 0)
        ef_angle_deg = math.degrees(ef_angle_rad)

        diff = np.array(ef_angle_rad) - np.array(obj_angle_rad)

        dist = abs(diff)

        actural_reward = 3.14 - dist
        reward_advantage = actural_reward - previous_reward
        reward = reward_advantage

        # self.cur_end_effector_pos = np.asarray(self.get_endEffector_pos()) * 10
        self.cur_end_effector_pos = np.asarray(endEffector_pos)

        self.cur_joint_angles = self.rob.getj()

        if num_data > 0:
            cv2.imwrite(self.path_element[1] + str(num_data) + '.bmp', self.set_state_element(1))  # Save the current image

            file = open(self.path_element[2] + str(num_data) + '.txt', "w")
            file.write("%f %f %f %f" % (self.obj_pos[0], self.obj_pos[1], self.obj_pos[2], self.rob.getj()[0]))  # Save the obj_pos, [x, y, z, base_angle(rad)]
            file.close()

            file = open(self.path_element[3] + str(num_data) + '.txt', "w")
            file.write(str(target_obj)[1:-1])  # Save the target_obj  ( one of hot encoding )
            file.close()

        return np.copy(self.state), reward, actual_action, self.done, actural_reward

    def get_internal_state(self):
        base_angle = math.degrees(self.rob.getj()[0])
        base_angle = base_angle / 90.0

        internal_state = np.float32(np.zeros(5))
        internal_state[0] = base_angle
        internal_state[1] = base_angle + 0.5
        internal_state[2] = base_angle + 1.0
        internal_state[3] = base_angle + 1.5
        internal_state[4] = base_angle + 2.0

        return internal_state

    def collision_chk(self):
        status = self.safety.status_chk()
        if status == "COLLISION":
            return True
        elif status == "NORMAL":
            return False

    def set_gripper(self, speed, force):
        self.gripper.set_gripper(speed, force)

    def gripper_close(self):
        # TODO Maybe value < THRESHOLD -> Grasp
        self.gripper.close()
        if self.gripper.DETECT_OBJ:
            #TODO
            k = 'DETECT_OBJ'

    def gripper_open(self):
        self.gripper.open()

    def shuffle_obj(self):
        acc = 1.5
        vel = 1.5

        self.rob.movej(HOME,acc=1.5, vel=1.5)

        # chk = self.collision_chk()    # Why made this ?

        # #self.rob.movej(shf_way_pt[0], acc=acc, vel=vel)  # random
        # self.rob.movej(shf_way_pt[1], acc=acc, vel=vel)
        # self.gripper.move(104)                           # near handle open
        # self.rob.movej(shf_way_pt[2], acc=acc, vel=vel)
        # self.gripper.move(229)                           # handle grip
        # self.rob.movej(shf_way_pt[3], acc=acc, vel=vel)
        # self.rob.movej(shf_way_pt[4], acc=acc, vel=vel)
        # time.sleep(2)                                    # stop delay 2~3 sec. # tray shuffle
        # self.rob.movej(shf_way_pt[5], acc=acc, vel=vel)
        # self.rob.movej(shf_way_pt[6], acc=acc, vel=vel)
        # self.gripper.move(104)                           # near handle open
        # self.rob.movej(shf_way_pt[7], acc=acc, vel=vel)

        # MIX TRAY
        # self.gripper_close()

        pt = [[0.1507703842858799, -0.3141178727128849, -0.030569762928828032, 2.2260958131016335, -2.1985942225522668, 0.05813081679518341],
              [-0.1491136865872555, -0.31021647495551335, -0.03052286866235667, -2.1570671726727104, 2.2799584355377167, -0.029214990891798114],
              [0.18414105144817525, -0.25296564835782714, -0.021194048759978847, -2.238562873918888, 2.1370170873680085, 0.10380902269948922],
              [-0.1417132578497845, -0.26528795089888607, -0.03157633192772823, -2.1045306382633795, 2.243489988356314, 0.0360411778983116],
              [0.20295675954768064, -0.16877330661979667, -0.038960103050602234, 0.018988621965437776, -3.007839525712047, -0.875295581190713],
              [-0.13352761097358432, -0.16247247277870452, -0.025328902795204018, 0.15436650469465374, -2.6904628274570306, -1.555752726987423]]

        # TRAY
        self.bluetooth.write("1".encode())

        self.x, self.y, _ = self.get_obj_pos(0)
        region = self._obj_region_chk(self.x, self.y)  # Center : 1, 2, 3

        random.seed(datetime.now())
        random_dir = random.randrange(0, 3)

        if region in [0, 1, 2]:
            if random_dir == 0 :
                self.rob.movej([1.2345331907272339, -1.776348892842428, 1.9926695823669434, -1.7940548102008265, -1.5028379599200647, -0.41095143953432256], acc, vel)
                self.rob.movel(pt[region*2], acc, vel)
                self.rob.movel(pt[region*2+1], acc, vel)
                self.rob.movej([1.27391582, -1.683021, 2.22669106, -2.14867484, -1.56398954, -0.33615041], acc, vel)
            elif random_dir == 1:
                self.rob.movej([1.2345331907272339, -1.776348892842428, 1.9926695823669434, -1.7940548102008265, -1.5028379599200647, -0.41095143953432256], acc, vel)
                self.rob.movel(pt[region * 2 + 1], acc, vel)
                self.rob.movel(pt[region * 2], acc, vel)
                self.rob.movej([1.27391582, -1.683021, 2.22669106, -2.14867484, -1.56398954, -0.33615041], acc, vel)
            else:
                pass

        else:
            pass

        self.rob.movej(HOME, 1.5, 1.5)

        time.sleep(3)  # waiting

    def img_proc(self, image):
        hsv = cv2.cvtColor(self.init_img, cv2.COLOR_RGB2HSV)
        lower_blue = np.array([105, 153, 0])
        upper_blue = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        result = cv2.bitwise_and(self.init_img, self.init_img, mask=mask)
        _, thresh = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
        blurred = cv2.medianBlur(thresh, 5)
        blurred = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        th = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        _, contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        idx = 0
        roi = np.empty((0, 4))
        pre = 0
        size_rank = 0

        for cnt in contours:
            if cv2.contourArea(cnt) < 60000:
                idx += 1
                cur = cv2.contourArea(cnt)

                if pre < cur:
                    size_rank = idx
                    pre = cur

                x, y, w, h = cv2.boundingRect(cnt)
                roi = np.vstack([roi, [x, y, w, h]])

        if roi.size != 0:
            roi = roi.astype(int)
            x = roi[size_rank - 1][0]
            y = roi[size_rank - 1][1]
            w = roi[size_rank - 1][2]
            h = roi[size_rank - 1][3]

            patch = image[y:y + h, x:x + w]
            blank_image = np.zeros((256, 256, 3), np.uint8)
            blank_image[y:y + h, x:x + w] = patch
            return blank_image

        else:
            return np.zeros((256, 256, 3), np.uint8)

    def get_camera(self, camera_num):
        if camera_num  == 1 :
            return self.global_cam.snap()

        if camera_num  == 2 :
            return self.local_cam.snap()

    def _obj_region_chk(self, x, y):
        y = abs(y)
        if x >= - 0.08 and x <= 0.08:
            if y >= 0.287:
                return 0
            elif y <= 0.287 and y > 0.2:
                return 1
            elif y <= 0.2:
                return 2
        elif x > 0.08:
            return 3
        elif x < -0.08:
            return 4

    def teaching_mode(self):
        # TODO : Reserved
        pass

    def get_obj_name(self, object_index):
        return objects_name[object_index]

    def get_obj_pos(self, obect_idx):
        # TODO : Reserved, RCNN
        img = self.global_cam.snap()
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

        # Center of Obejct on Image
        cx = 0
        cy = 0

        for cnt in contours:
            if cv2.contourArea(cnt) < 60000:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)

                if radius > max_radius:
                    max_radius = radius

        # pixel to robot coordinate
        obj_y = ( 2.9579207920792 * cy - 405.33415841584) / 1000
        obj_x = (401.5000081602 - 3.1829680283322 * cx) / 1000
        obj_z = 50 / 1000

        return obj_x, obj_y , obj_z

    def set_state_element(self, repeat):
        temp_img = self.get_camera(1)
        blank_image = self.img_proc(temp_img)

        self.state[:, :, :, 0, repeat-1] = temp_img
        self.state[:, :, :, 1, 0] = self.initial_state

        return blank_image
