"""
For grasping using IK v2, data collect for pre-training
with calibration, segmentation

latest Ver.171120
"""

# Robot
from Kinect_Snap import *
from pyueye import *
import socket
import pyGrip

# utils
import copy
import cv2
import serial
import random

from util import *

# ----------------- Define -----------------------------------------------------
PI = np.pi
HOME = (90 * PI / 180, -90 * PI / 180, 0, -90 * PI / 180, 0, 0)
INITIAL_POSE = (0, -90 * PI / 180, 0, -90 * PI / 180, 0, 0)
starting_pose = [ 1.2985, -1.7579,  1.6851, -1.5005, -1.5715, -0.2758]

shf_pt = [[-1.5937, -1.3493, 1.6653, -1.8875, -1.5697, -1.5954],
          [-1.5839, -1.2321, 1.8032, - 2.1425, - 1.5727, -1.5859],
          [-1.567 , -1.4621, 1.8072 ,- 2.0248, - 1.6009, -1.5858],
          [-1.5522, -2.055,  1.8372, - 1.8303, - 1.6125, -1.5792]]

j_pt = [[ 3.2842, -1.389,   1.7775, -1.9584, -1.5712,  0.1392],
        [-0.6439, -1.6851,  2.0874, -1.9749, -1.5725, -0.6631],
        [ 2.4898, -1.6037,  2.0139, -1.9841, -1.5702,  0.9178],
        [-0.0679, -2.0635,  2.3301, -1.8392, -1.5719, -0.7164],
        [-1.1394, -1.6986,  2.0996, -1.9742, -1.5719, -0.2761],
        [ 2.9123, -1.8265,  2.1973, -1.9434, -1.5732,  0.4029],
        [-0.2053, -1.6117,  2.0214, -1.9833, -1.5722,  0.1113],
        [-0.6512, -2.1626,  2.3689, -1.7789, -1.5735,  0.1435],
        [ 2.1833, -1.9322,  2.263 , -1.9026, -1.5712, -1.3421]]

OBJ_LIST = ['O_00_Black_Tape', 'O_01_Glue', 'O_02_Big_USB', 'O_03_Glue_Stick', 'O_04_Big_Box',
            'O_05_Red_Cup', 'O_06_Small_Box', 'O_07_White_Tape', 'O_08_Small_USB',  'O_09_Yellow_Cup']

bkg_padding_img = cv2.imread('new_background\\1.bmp')[:31, :, :]
# -------------------------------------------------------------------------------


def add_opts(parser):
    parser.add_argument('--event-log-out', type=str, default=None, help="path to record event log.")
    parser.add_argument('--max-episode-len', type=int, default=200, help="maximum episode len for cartpole")
    parser.add_argument('--render-width', type=int, default=256, help="if --use-raw-pixels render with this width")
    parser.add_argument('--render-height', type=int, default=256, help="if --use-raw-pixels render with this height")
    parser.add_argument('--reward-calc', type=str, default='fixed',
                        help="'fixed': 1 per step. 'angle': 2*max_angle - ox - oy. 'action': 1.5 - |action|. 'angle_action': both angle and action")


class Env:
    def __init__(self, socket_ip, opts):
        # Connect to Environment
        self.rob = urx.Robot(socket_ip)                             # Robot
        self.gripper = pyGrip.Gripper(host=socket_ip)             # Gripper

        # Dashboard Control
        self.Dashboard_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.Dashboard_socket.connect((socket_ip, 29999))
        self._program_send("")

        # Tray Control
        self.bluetooth = serial.Serial("COM5", 9600, timeout=1)     # Tray

        # Camera interface
        self.global_cam = Kinect()                         # Kinect Camera
        self.local_cam = UEyeCam()                         # Local Camera

        # Segmentation Model
        self.seg_model = None
        self.obj_angle = 0

        # Robot
        self.acc = 3
        self.vel = 3
        self.available = True
        self.unreachable = False

        # Variables
        self.opts = opts
        self.render_width = opts.render_width
        self.render_height = opts.render_height
        self.default_tcp = [0, 0, 0.150, 0, 0, 0]  # (x, y, z, rx, ry, rz)
        self.done = False

        # States - Local cam img. w : 256, h : 256, c :3
        self.state_shape = (self.render_height, self.render_width, 3)
        self.state = np.empty(self.state_shape, dtype=np.float32)
        self.internal_state = np.empty([6], dtype=np.float32)

        # Action Dim
        self.action_dim = opts.action_dim    # 1 dim.

        # object position
        self.target_cls = 0
        self.obj_pos = np.zeros([3])         # (x, y, z)
        self.depth_f = np.zeros([1])
        self.eigen_value = np.zeros([2])

        # maximum and minimum angles of the 6-th joint for orienting task
        self.max_ori = -100
        self.min_ori = 100
        self.target_object_orientation = 0

        # Make directory for save Data Path
        # DATA SAVER FOR PRE-TRAINING
        if self.opts.with_data_collecting:
            self.path_element = []
            self.max_num_list = []
            #self.max_num_list = np.zeros([10], dtype= np.uint8)
            dir_path = "E:\\save_data_ik_1201\\"

            [self.path_element.append(dir_path + str(x) + "\\") for x in np.arange(10)]
            if not os.path.exists(dir_path):
                [os.makedirs(x) for x in self.path_element]
                self.update_max_list()
            else:         # Make file indexing
                self.update_max_list()

        # Reset Environment
        self.set_tcp(self.default_tcp)
        self.movej(HOME, self.acc, self.vel)

        print("Robot Environment Ready.", file=sys.stderr)

    def update_max_list(self):
        for path in self.path_element:
            dummy = []
            if len(os.listdir(path)) != 0:
                for i, x in enumerate(os.listdir(path)):
                    dummy.append(int(x[:-4].split('_')[1::2][0]))  # file num in
                self.max_num_list.append(max(dummy)+1)

            else:
                self.max_num_list.append(1)

    def state_update(self):   # State : Local Camera View
        img = self.local_cam.snap()
        self.state = np.asarray(img)

        cv2.imshow("image", self.state)

    def set_segmentation_model(self, segmentation_model):
        self.seg_model = segmentation_model

    def reset(self, target_cls):
        self.movej(HOME, self.acc, self.vel)
        self.gripper_open()
        self.approaching(target_cls)             # robot move

        if self.obj_pos is None or self.obj_pos[2] > 0.275:  # 1.5는 로봇 기준 에러 난 z축 위치 스킵 하는거 추가하자. # 수치가 부정확하단거. 이거 확인해봐야함.
            return None

        self.state_update()                      # local view
        self.internal_state_update()             # Internal State Update. Last joint angles

        if self.opts.with_data_collecting:
            self.store_data(self.target_cls)

        return np.copy(self.state)

    def store_data(self, class_idx):
        if self.available is True:
            save_path = self.path_element[class_idx]
            num = self.max_num_list[class_idx]

            # save_image
            self.state_update()
            cv2.imwrite(save_path+"{}_{}.bmp".format(class_idx, num), self.state)

            data = np.round(np.round(self.getl()[0:3], 4) - self.obj_pos, 5)
            j_6 = np.round(self.getj()[-1], 4)

            with open(save_path + "{}_{}.txt".format(class_idx, num), "w") as f:
                f.write("{} {} {} {} {} {}".format(*data, *self.eigen_value, j_6))

            self.max_num_list[class_idx] += 1

    def approaching(self, class_idx):
        a, v = [self.acc, self.vel]
        seg_img, color_seg_img = self.get_seg()
        cv2.imshow("color_seg_img", color_seg_img)
        cv2.waitKey(10)

        # if the target class not exist, pass
        if class_idx not in np.unique(seg_img):
            print("Failed to find %s" % OBJ_LIST[class_idx], file=sys.stderr)
            self.obj_pos = None
            return

        else:
            self.obj_pos, self.eigen_value = self.get_obj_pos(class_idx)

            if self.obj_pos is None:
                self.unreachable = True
            elif (self.obj_pos[0] < -0.306) and (self.obj_pos[1] < - 0.420):
                self.unreachable = True
            elif (self.obj_pos[0] > 0.292) and (self.obj_pos[1] < - 0.420):
                self.unreachable = True

            self.movej(starting_pose, a, v)      # Move to starting position,

            if self.obj_pos is None:
                return

            if class_idx == 5 and self.obj_pos[2] < -0.07:
                goal = np.append(self.obj_pos + np.array([0, 0, 0.15]), [0, -3.14, 0])  # Initial point  Added z-dummy 0.05
            else:
                goal = np.append(self.obj_pos + np.array([0, 0, 0.1]), [0, -3.14, 0])      # Initial point  Added z-dummy 0.05

            self.movel(goal, self.acc, self.vel)

            if self.unreachable:
                self.obj_pos = None
                self.unreachable = False

    def grasp(self, target_cls):
        self.obj_pos[2] = -0.056

        goal = np.append(self.obj_pos, self.getl()[3:])  # Initial point

        self.movel(goal, 0.5, 0.5)
        self.gripper_close()  # 닫고

        if not self.gripper.DETECT_OBJ:
            return

        # Move to starting_pose
        self.movej(starting_pose, self.acc, self.vel)

        if self.gripper.DETECT_OBJ:
            print("Success grasp")
        else:
            print("Failed grasp")

        # Move to pose
        self.movej(j_pt[target_cls], self.acc, self.vel)

        l = self.getl()
        l[2] -= 0.057
        self.movel(l, 1, 1)

        self.gripper_open()

        l = self.getl()
        l[2] += 0.057
        self.movel(l, 1, 1)

    def get_plus_or_minus_angle(self, step):
        target_object_orientation, symm = self.seg_model.get_angle(self.target_cls)
        target_boxed_angle = self.seg_model.get_boxed_angle(self.target_cls)

        plus_or_minus_angle_code = np.zeros([2], dtype=np.float)
        move_to_plus_angle = True

        if symm is not None:
            self.available = True
            joint1_orientation = np.rad2deg(self.getj()[0])
            joint1_orientation = joint1_orientation - 90
            joint6_orientation = -1 * np.rad2deg(self.getj()[-1])
            current_joint6_orientation = joint1_orientation + joint6_orientation

            if step == 1:
                # target_object_orientation = np.rad2deg(target_object_orientation)
                if symm == 1 and self.target_cls not in [0, 7]:

                    target_object_orientation = target_boxed_angle
                    target_object_orientations = np.zeros(shape=[4])
                    target_object_orientations[0] = target_object_orientation

                    if (target_object_orientation >= -180) and (target_object_orientation < -90):
                        target_object_orientations[1] = target_object_orientation + 90
                        target_object_orientations[2] = target_object_orientation + 180
                        target_object_orientations[3] = target_object_orientation + 270
                    if (target_object_orientation >= -90) and (target_object_orientation < 0):
                        target_object_orientations[1] = target_object_orientation + 90
                        target_object_orientations[2] = target_object_orientation + 180
                        target_object_orientations[3] = target_object_orientation - 89
                    if (target_object_orientation >= 0) and (target_object_orientation < 90):
                        target_object_orientations[1] = target_object_orientation + 90
                        target_object_orientations[2] = target_object_orientation - 90
                        target_object_orientations[3] = target_object_orientation - 179
                    if (target_object_orientation >= 90) and (target_object_orientation < 180):
                        target_object_orientations[1] = target_object_orientation - 90
                        target_object_orientations[2] = target_object_orientation - 180
                        target_object_orientations[3] = target_object_orientation - 269

                    diffs = target_object_orientations - current_joint6_orientation
                    abs_diffs = abs(diffs)
                    index = np.argmin(abs_diffs)
                    self.target_object_orientation = target_object_orientations[index]

                elif self.target_cls in [0, 7] and symm == 1:
                    self.target_object_orientation = 0
                else:
                    target_object_orientation = target_boxed_angle
                    target_object_orientations = np.zeros([2])
                    target_object_orientations[0] = target_object_orientation

                    if target_object_orientation <= 0:
                        target_object_orientations[1] = target_object_orientation + 179
                    else:
                        target_object_orientations[1] = target_object_orientation - 179

                    diffs = target_object_orientations - current_joint6_orientation
                    abs_diffs = abs(diffs)
                    index = np.argmin(abs_diffs)
                    self.target_object_orientation = target_object_orientations[index]

                if current_joint6_orientation > self.target_object_orientation:
                    self.max_ori = np.deg2rad(-joint6_orientation + (-self.target_object_orientation + 35))
                    self.min_ori = np.deg2rad(-joint6_orientation - 10)
                    plus_or_minus_angle_code[0] = 0  # move minus angle
                    plus_or_minus_angle_code[1] = 1
                    move_to_plus_angle = True
                else:
                    self.max_ori = np.deg2rad(-joint6_orientation + 10)
                    self.min_ori = np.deg2rad(-joint6_orientation - (self.target_object_orientation + 35))
                    plus_or_minus_angle_code[0] = 1  # move plus angle
                    plus_or_minus_angle_code[1] = 0
                    move_to_plus_angle = False
            else:
                if current_joint6_orientation > self.target_object_orientation:
                    plus_or_minus_angle_code[0] = 0  # move minus angle
                    plus_or_minus_angle_code[1] = 1
                    move_to_plus_angle = True
                else:
                    plus_or_minus_angle_code[0] = 1  # move plus angle
                    plus_or_minus_angle_code[1] = 0
                    move_to_plus_angle = False
        else:
            self.available = False
            print("Error : calc_max_min_orientation_for_orienting_task, {}".format(symm), file=sys.stderr)

        return plus_or_minus_angle_code, move_to_plus_angle

    def calc_reward_for_orienting_task(self, cost):

        joint1_orientation = np.rad2deg(self.getj()[0])
        joint1_orientation = joint1_orientation - 90
        joint6_orientation = -1 * np.rad2deg(self.getj()[-1])
        current_joint6_orientation = joint1_orientation + joint6_orientation

        abs_diff = abs(self.target_object_orientation - current_joint6_orientation)
        dist = abs_diff / 180

        if abs_diff < 0.14:
            dist = 0.0

        print("Target orientation : {}, Joint6_orientation : {}".format(self.target_object_orientation, current_joint6_orientation))

        return dist * -0.8 - cost

    def plate_obj(self):
        pass

    def step(self, action, exploration_noise, num_step, move_to_plus_angle, is_training):
        target_angle = self.getj()[5] + action
        is_terminal = True
        reward = 0
        actual_action = 0

        if is_training:
            noise = exploration_noise.noise()
        else:
            noise = 0

        cost = 0.0

        if self.available:
            if self.min_ori > target_angle:
                action = action + np.abs(self.min_ori - target_angle) + abs(noise)

            if target_angle > self.max_ori:
                action = action - np.abs(target_angle - self.max_ori) - abs(noise)

            if move_to_plus_angle == True:
                if action < 0.0:
                    cost = 0.05
            if move_to_plus_angle == False:
                if action >= 0.0:
                    cost = 0.05

            target_angle = self.getj()[5] + action

            target_angle = np.append(self.getj()[:-1], target_angle)    # 6dim.

            actual_action = action
            self.movej(target_angle, self.acc, self.vel)                                   # Real robot move to goal
            self.done = False

            reward = self.calc_reward_for_orienting_task(cost)

            self.state_update()
            self.internal_state_update()

        return np.copy(self.state), reward, actual_action, self.done, is_terminal, self.available

    def internal_state_update(self):    # last joint angle
        internal_state = []

        for idx in range(6):            # Current angles 5 * 6 = 30
            internal_state.extend(np.array([-1.0, -0.5, 0, 0.5, 1]) - self.getj()[idx])

        internal_state.extend(self.getl()[0:3])
        self.internal_state = np.array(internal_state)

    def getl(self):
        return np.array(self.rob.getl())

    def getj(self):
        return np.around(np.array(self.rob.getj()), decimals=4)

    def get_ef(self):
        return np.around(np.array(self.rob.getl()[0:3]), decimals=4)

    def movej(self, goal_pose, acc=1.2, vel=1.2):
        try:
            self.rob.movej(goal_pose, acc, vel)
        except urx.RobotException:
            self.status_chk()
            self.movej(HOME, 5, 5)
            self.available = False

    def movel(self, goal_pose, acc=1.2, vel=1.2):
        try:
            self.rob.movel(goal_pose, acc, vel)
        except urx.RobotException:
            self.status_chk()
            self.unreachable = True

    def _program_send(self, cmd):
        self.Dashboard_socket.send(cmd.encode())
        return self.Dashboard_socket.recv(1024).decode("utf-8")  # received byte data to string

    def status_chk(self):
        # ~status chk reward, cur_angle, next_angle use ?
        robotmode = self._program_send("robotmode\n")[0:-1].split(' ')[1]

        if robotmode == 'POWER_OFF':
            self._program_send("power on\n")
            self._program_send("brake release\n")
            time.sleep(5)

        safetymode = self._program_send("safetymode\n")[0:-1].split(' ')[1]

        if safetymode == "NORMAL":
            pass
        elif safetymode == "PROTECTIVE_STOP":
            print("Protective stopped !", file=sys.stderr)
            self._program_send("unlock protective stop\n")
        elif safetymode == "SAFEGUARD_STOP":
            print("Safeguard stopped !", file=sys.stderr)
            self._program_send("close safety popup\n")
        else:
            print("Unreachable position self.obj_pos")
            print(safetymode)

    def set_gripper(self, speed, force):
        self.gripper.set_gripper(speed, force)

    def gripper_close(self):
        self.gripper.close()

    def gripper_open(self):
        self.gripper.open()

    def shuffle_obj(self):
        self.movej(HOME, self.acc, self.vel)

        # Tray Control
        if self.opts.num_eval > 0:
            self.movej(starting_pose, self.acc, self.vel)
            self.movej(shf_pt[0], self.acc, self.vel)
            self.gripper.move(104)
            self.movej(shf_pt[1], self.acc, self.vel)
            self.gripper.close()
            self.movej(shf_pt[2], self.acc, self.vel)
            self.movej(shf_pt[3], self.acc, self.vel)
            time.sleep(1)
            self.movej(shf_pt[2], self.acc, self.vel)
            self.movej(shf_pt[1], self.acc, self.vel)
            self.gripper.move(104)
            self.movej(shf_pt[0], self.acc, self.vel)
            self.gripper.open()
            # self.movej(starting_pose, self.acc, self.vel)

        # MIX TRAY

        if not self.opts.num_eval > 0:
            self.gripper_close()

            pt = [[-0.20, -0.45, -0.0484, -2.18848, -2.22941, 0.05679],
                  [0.20, -0.45, -0.0484, -2.18848, -2.22941, 0.05679],
                  [0.20, -0.340179, -0.0484, -2.18848, -2.22941, 0.05679],
                  [-0.20, -0.340179, -0.0484, -2.18848, -2.22941, 0.05679],
                  [-0.20, -0.217842, -0.0484, -0.01386,  3.08795, 0.39780],
                  [0.20, -0.217842, -0.0484, -0.01386, 3.08795, 0.39780]]

            dir_idx = random.sample([0, 1, 2, 3], 1)  # 리스트에서 6개 추출

            dir = [[0, 1, 2, 3, 4, 5],
                   [1, 0, 3, 2, 5, 4],
                   [4, 5, 2, 3, 0, 1],
                   [5, 4, 3, 2, 1, 0]]

            for idx in dir_idx:
                self.bluetooth.write("1".encode())
                self.bluetooth.write("1".encode())

                for i, point in enumerate(dir[idx]):
                    if i == 0:
                        pt_copy = copy.deepcopy(pt)
                        pt_copy[point][0] = pt_copy[point][0] + (0.05 * pow(-1, point + 1))
                        pt_copy[point][2] = pt_copy[point][2] + 0.05
                        self.movej(starting_pose)
                        self.movel(pt_copy[point])
                        pt_copy[point][2] = pt_copy[point][2] - 0.05
                        self.movel(pt_copy[point])
                    else:
                        self.movel(pt[point])

                self.movej(starting_pose, self.acc, self.vel)
                self.gripper_open()

        # self.movej(starting_pose, self.acc, self.vel)

        self.movej(HOME, self.acc, self.vel)

    def get_seg(self):
        time.sleep(0.2)
        img = self.global_cam.snap()  # Segmentation input image  w : 256, h : 128
        padded_img = cv2.cvtColor(np.vstack((bkg_padding_img, img)), cv2.COLOR_RGB2BGR)  # Color fmt    RGB -> BGR
        # cv2.imshow("Input_image", padded_img)

        return self.seg_model.run(padded_img)

    def get_obj_pos(self, class_idx):  # TODO : class 0~8 repeat version
        self.target_cls = class_idx
        print(">> Target Object : ", OBJ_LIST[class_idx], file=sys.stderr)

        pxl_list, eigen_value = self.seg_model.getData(class_idx)  # Get pixel list

        if pxl_list is not None:
            xyz = self.global_cam.color2xyz(pxl_list)   # patched image's averaging pose [x, y, z]

            return [xyz, eigen_value]
        else:
            return [None, None]

    def set_tcp(self, tcp):
        self.rob.set_tcp(tcp)

    def shutdown(self):
        self._program_send("Shutting down")

    # def calc_max_min_orientation_for_orienting_task(self):
    #     target_object_orientation, symm = self.seg_model.get_angle(self.target_cls)
    #     target_boxed_angle = self.seg_model.get_boxed_angle(self.target_cls)
    #
    #     if symm is not None:
    #         # target_object_orientation = np.rad2deg(target_object_orientation)
    #         joint1_orientation = np.rad2deg(self.getj()[0])
    #         joint1_orientation = joint1_orientation - 90
    #         joint6_orientation = -1 * np.rad2deg(self.getj()[-1])
    #         current_joint6_orientation = joint1_orientation + joint6_orientation
    #
    #         if symm == 1 and self.target_cls not in [0, 7]:
    #             shape = (4)
    #             target_object_orientation = target_boxed_angle
    #             target_object_orientations = np.zeros(shape=shape)
    #             target_object_orientations[0] = target_object_orientation
    #
    #             if target_object_orientation >= -180 and target_object_orientation < -90:
    #                 target_object_orientations[1] = target_object_orientation + 90
    #                 target_object_orientations[2] = target_object_orientation + 180
    #                 target_object_orientations[3] = target_object_orientation + 270
    #             if target_object_orientation >= -90 and target_object_orientation < 0:
    #                 target_object_orientations[1] = target_object_orientation + 90
    #                 target_object_orientations[2] = target_object_orientation + 180
    #                 target_object_orientations[3] = target_object_orientation - 89
    #             if target_object_orientation >= 0 and target_object_orientation < 90:
    #                 target_object_orientations[1] = target_object_orientation + 90
    #                 target_object_orientations[2] = target_object_orientation - 90
    #                 target_object_orientations[3] = target_object_orientation - 179
    #             if target_object_orientation >= 90 and target_object_orientation < 180:
    #                 target_object_orientations[1] = target_object_orientation - 90
    #                 target_object_orientations[2] = target_object_orientation - 180
    #                 target_object_orientations[3] = target_object_orientation - 269
    #
    #             diffs = target_object_orientations - current_joint6_orientation
    #             abs_diffs = abs(diffs)
    #             index = np.argmin(abs_diffs)
    #             target_object_orientation = target_object_orientations[index]
    #
    #         elif self.target_cls in [0, 7] and symm == 1:
    #             target_object_orientation = 0
    #         else:
    #             shape = (2)
    #             target_object_orientation = target_boxed_angle
    #             target_object_orientations = np.zeros(shape=shape)
    #             target_object_orientations[0] = target_object_orientation
    #
    #             if target_object_orientation <= 0:
    #                 target_object_orientations[1] = target_object_orientation + 179
    #             else:
    #                 target_object_orientations[1] = target_object_orientation - 179
    #
    #             diffs = target_object_orientations - current_joint6_orientation
    #             abs_diffs = abs(diffs)
    #             index = np.argmin(abs_diffs)
    #             target_object_orientation = target_object_orientations[index]
    #
    #         if current_joint6_orientation > target_object_orientation:
    #             self.max_ori = np.deg2rad(-joint6_orientation -1 * (target_object_orientation - 35) )
    #             self.min_ori = np.deg2rad(-joint6_orientation) - 10
    #         else:
    #             self.max_ori = np.deg2rad(-joint6_orientation) + 10
    #             self.min_ori = np.deg2rad(-joint6_orientation -1 * (target_object_orientation + 35) )
    #
    #         return True
    #     else:
    #         print("Error : calc_max_min_orientation_for_orienting_task, {}".format(symm), file=sys.stderr)
    #         return False