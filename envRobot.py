import random
import sys
import urx
import pyGrip
import serial
import numpy as np
import ur_safety
import pyueye
import Kinect_Snap
from util import *
from PIL import Image as Im
import cv2
import copy
import math
import argparse

# ----------------- Define -------------------------
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
# --------------------------------------------------

objects_name = ['O_00_Big_USB', 'O_01_Black_Tape', 'O_02_Blue_Glue', 'O_03_Brown_Box', 'O_04_Green_Glue',
                    'O_05_Pink_Box', 'O_06_Red_Cup', 'O_07_Small_USB', 'O_08_White_Tape', 'O_09_Yellow_Cup' ]


def add_opts(parser):
  parser.add_argument('--event-log-out', type=str, default=None,
                      help="path to record event log.")
  parser.add_argument('--max-episode-len', type=int, default=200,
                      help="maximum episode len for cartpole")
  parser.add_argument('--use-raw-pixels', default=True, action='store_true',
                      help="use raw pixels as state instead of cart/pole poses")
  parser.add_argument('--render-width', type=int, default=256,
                      help="if --use-raw-pixels render with this width")
  parser.add_argument('--render-height', type=int, default=256,
                      help="if --use-raw-pixels render with this height")
  parser.add_argument('--reward-calc', type=str, default='fixed',
                      help="'fixed': 1 per step. 'angle': 2*max_angle - ox - oy. 'action': 1.5 - |action|. 'angle_action': both angle and action")

class envRobot :
    # def __init__(self, SOCKET_IP, opts):
    def __init__(self, SOCKET_IP, opts):
        # Connect
        self.gripper = pyGrip.gripper(host=SOCKET_IP)               # Gripper
        self.rob = urx.Robot(SOCKET_IP)                             # Robot
        self.safety = ur_safety.safety_chk(host= SOCKET_IP)         # Robot - Dashboard ( for collision check)
        chk = self.safety.status_chk()
        self.bluetooth = serial.Serial("COM9", 9600, timeout=1)     # Tray
        # self.local_cam = pyueye.uEyeCAM()
        self.global_cam = Kinect_Snap.global_cam()

        # Robot
        self.acc = 1.5; self.vel = 1.5
        self.isCollision = False

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
        self.initial_state = np.empty(self.state_shape, dtype=np.float32)
        self.action_dim = opts.action_dim

        print("Robot Environment Ready.", file = sys.stderr)


    def reset(self, obj_pos, target_obj, num_data, save_path):
        self.rob.movej(HOME)
        # degree = self.check_move_finish(HOME) # Maybe we don't need.

        self.state[:, :, :, 1, 0] = self.get_camera(1)
        # image.transpose(Im.FLIP_TOP_BOTTOM).save('C:\\tensorflow_code\\Vrep\\approaching_code\\DDPG\imgs\\temp.bmp','bmp')
        self.initial_state = self.state[:, :, :, 1, 0]


        # if num_data > 0:
        #     initial_image_name = save_path + 'save_data\\initial_image\\'
        #
        #     initial_image_name = initial_image_name + str(num_data) + '.bmp'
        #     image = image.transpose(Im.FLIP_TOP_BOTTOM)
        #     image.save(initial_image_name, format='bmp')

        degree = (-90, 0, 0, 0, 0, 0)
        self.rob.movej(INITIAL_POSE)
        #degree = self.check_move_finish(INITIAL_POSE)
        self.state[:, :, :, 0, 0] = self.get_camera(1)

        # if num_data > 0:
        #     current_image_name = save_path + 'save_data\\current_image\\'
        #     current_image_name = current_image_name + str(num_data) + '.bmp'
        #     image = image.transpose(Im.FLIP_TOP_BOTTOM)
        #     image.save(current_image_name, format='bmp')
        #
        #     angle_radian = math.radians(degree[0])
        #
        #     obj_position_file_name = save_path + 'save_data\\obj_position\\'
        #     obj_position_file_name = obj_position_file_name + str(num_data) + '.txt'
        #     file = open(obj_position_file_name, "w")
        #     pos = "%f %f %f %f" % (obj_pos[0], obj_pos[1], obj_pos[2], angle_radian)
        #     file.write(pos)
        #     file.close()
        #
        #     target_obj_file_name = save_path + 'save_data\\target_obj\\'
        #     target_obj_file_name = target_obj_file_name + str(num_data) + '.txt'
        #     file = open(target_obj_file_name, "w")
        #     pos = "%d %d %d %d %d %d %d %d %d %d" % (target_obj[0], target_obj[1], target_obj[2], target_obj[3],
        #                                              target_obj[4], target_obj[5], target_obj[6], target_obj[7],
        #                                              target_obj[8], target_obj[9])
        #     file.write(pos)
        #     file.close()

        obj_pos_copy = copy.deepcopy(obj_pos) # deep copy?

        # rob_y = 2.9579207920792*c_y - 405.33415841584
        # rob y=401.5000081602-3.1829680283322x

        # obj_angle_rad = math.atan2(obj_pos[1], obj_pos[0])
        obj_angle_rad = math.atan2(obj_pos[0], obj_pos[1])    #  angle(rad) =  atan(x, y) ?

        raw = self.rob.getl()
        endEffector_pos = [round(x * 1000, 4) for x in raw[0:3]]  # unit : mili
        #r_xyz = [round(x, 4) for x in raw[3:]] # Rotation

        epsilon = 0.00001

        ef_angle_rad = math.atan2(endEffector_pos[1] - 0, endEffector_pos[0] - 0)
        # ef_angle_rad = math.atan2(endEffector_pos[0] - 0, endEffector_pos[1] - 0)  #  angle(rad) =  atan(x, y) ?
        diff = np.array(ef_angle_rad) - np.array(obj_angle_rad)
        reward = 3.15 - abs(diff)

        self.cur_end_effector_pos = np.asarray(endEffector_pos) #TODO : END EFFECTOR
        self.cur_joint_angles = self.rob.getj()

        # TODO END_EFFECTOR
        self.internal_state = np.concatenate((self.cur_joint_angles, self.cur_end_effector_pos))

        # return this state
        return np.copy(self.state), reward

    def step(self, action, target_object_index, initial_obj_pos, target_obj, num_data, previous_reward, save_path):

        self.done = False
        object_moved = False
        reward = 0
        action = list(action.reshape(-1, ))

        previous_joint_angle = self.rob.getj()
        target_joint_angle = previous_joint_angle[0] + action

        epsilon = 0.00001

        cost = 0.0
        if target_joint_angle < -180:
            diff_angle = abs(target_joint_angle + 180.0) + action[0]
            action[0] = diff_angle
            target_joint_angle = -180

        elif target_joint_angle > 0:
            diff_angle = action[0] - target_joint_angle
            action[0] = diff_angle
            target_joint_angle = 0

        action[0] = target_joint_angle
        action.append(0)
        action.append(0)
        action.append(0)
        action.append(0)
        action.append(0)

        self.rob.movej(action)
        actual_action = current_angles - previous_joint_angle

        image = self.set_state_element(1)

        obj_pos = self.get_obj_pos()
        obj_pos_copy = copy.deepcopy(obj_pos)

        diff = np.array(initial_obj_pos) - np.array(obj_pos)
        dist = np.linalg.norm(diff, 2)

        if dist > 0.02:
            object_moved = True
            return np.copy(self.state), reward, actual_action, self.done, object_moved, reward

        obj_angle_rad = math.atan2(obj_pos_copy[1], obj_pos_copy[0])
        obj_angle_deg = math.degrees(obj_angle_rad)

        endEffector_pos = [round(x * 1000, 4) for x in self.rob.getl()[0:3]]  # unit : mili

        ef_angle_rad = math.atan2(endEffector_pos[1] - 0, endEffector_pos[0] - 0)
        ef_angle_deg = math.degrees(ef_angle_rad)

        diff = np.array(ef_angle_rad) - np.array(obj_angle_rad)

        dist = abs(diff)

        actural_reward = 3.14 - dist
        reward_advantage = previous_reward - actural_reward
        reward = reward_advantage

        # self.cur_end_effector_pos = np.asarray(self.get_endEffector_pos()) * 10
        self.cur_end_effector_pos = np.asarray(endEffector_pos) * 10

        self.cur_joint_angles = self.rob.getj()


        if num_data > 0:
            current_image_name = save_path + 'save_data\\current_image\\'
            current_image_name = current_image_name + str(num_data) + '.bmp'
            image = image.transpose(Im.FLIP_TOP_BOTTOM)
            image.save(current_image_name, format='bmp')

            angle_radian = math.radians(self.cur_joint_angles[0])

            obj_position_file_name = save_path + 'save_data\\obj_position\\'
            obj_position_file_name = obj_position_file_name + str(num_data) + '.txt'
            file = open(obj_position_file_name, "w")
            pos = "%f %f %f %f" % (obj_pos_copy[0], obj_pos_copy[1], obj_pos_copy[2], angle_radian)
            file.write(pos)
            file.close()

            target_obj_file_name = save_path + 'save_data\\target_obj\\'
            target_obj_file_name = target_obj_file_name + str(num_data) + '.txt'
            file = open(target_obj_file_name, "w")
            pos = "%d %d %d %d %d %d %d %d %d %d" % (
            target_obj[0], target_obj[1], target_obj[2], target_obj[3], target_obj[4],
            target_obj[5], target_obj[6], target_obj[7], target_obj[8], target_obj[9])
            file.write(pos)
            file.close()

        self.internal_state = np.concatenate((self.cur_joint_angles, self.cur_end_effector_pos))

        return np.copy(self.state), reward, actual_action, self.done, object_moved, actural_reward

    def get_internal_state(self):
        return self.internal_state



#    def reset(self):
#        # Reset state
#        self.rob.movej(HOME)
#        self.gripper.open()
#        self.done = False
#        self.isCollision = False

 #       # TODO : Data?

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
        acc = 1.0; vel = 1.0

        chk = self.collision_chk()
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

        # Vibrate tray, shuffle

        # MIX TRAY
        #self.gripper_close()

        pt = [[0.1507703842858799, -0.3141178727128849, -0.030569762928828032, 2.2260958131016335, -2.1985942225522668,
               0.05813081679518341],
              [-0.1491136865872555, -0.31021647495551335, -0.03052286866235667, -2.1570671726727104, 2.2799584355377167,
               -0.029214990891798114],
              [0.20295675954768064, -0.16877330661979667, -0.038960103050602234, 0.018988621965437776,
               -3.007839525712047, -0.875295581190713],
              [-0.13352761097358432, -0.16247247277870452, -0.025328902795204018, 0.15436650469465374,
               -2.6904628274570306, -1.555752726987423]]

        isShuffle = ['YES', 'NO']
        starting_pt = [0, 1, 2, 3]

        self.bluetooth.write("1".encode())

        if random.choice(isShuffle) == 'YES':
            self.rob.movej(
                [1.7627151012420654, -1.2260602156268519, 2.2933664321899414, -2.6315630117999476, -1.542201344166891,
                 -4.7985707418263246e-05], 1.5, 1.5)
            chk_pt = random.choice(starting_pt)
            if chk_pt == 1 or chk_pt == 3:
                self.rob.movel(pt[chk_pt], acc, vel)
                self.rob.movel(pt[chk_pt-1], acc, vel)
            elif chk_pt == 0 or chk_pt == 2:
                self.rob.movel(pt[chk_pt], acc, vel)
                self.rob.movel(pt[chk_pt + 1], acc, vel)

            # random.shuffle(starting_pt)
            # # for i in starting_pt:
            # #     self.rob.movel(pt[i], acc=1.5, vel=1.5)
            self.rob.movej(HOME, 1.5, 1.5)

        else:
            pass

        time.sleep(3)  # waiting

    def get_camera(self, camera_num):
        if camera_num  == 1 :
            return self.global_cam.snap()
        if camera_num  == 2 :
            return self.local_cam.snap()



    def get_state(self):
        global_img = self.global_cam.snap()
        local_img = self.local_cam.snap()

        # TODO 1 : Camera Image resizing

        # TODO 2 : Data
        rob_joint = self.rob.getj()
        tcp_pose = self.rob.getl()

        return [global_img, local_img, rob_joint, tcp_pose]

    def teaching_mode(self):
        # TODO : Reserved
        pass


    def get_obj_name(self, object_index):
        return objects_name[object_index]

    def get_obj_pos(self):
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
        cx =  0
        cy = 0

        for cnt in contours:
            if cv2.contourArea(cnt) < 60000:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                print(cx, cy)

                if radius > max_radius:
                    max_radius = radius

        # pixel to robot coordinate
        obj_y = 2.9579207920792 * cy - 405.33415841584
        obj_x = 401.5000081602 - 3.1829680283322 * cx
        obj_z = 50

        return obj_x, obj_y , obj_z