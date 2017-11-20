"""
For grasping using IK v2, data collect for pre-training
with calibration, segmentation

latest Ver.171117
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
starting_pose = (1.2345331907272339, -1.776348892842428, 1.9926695823669434, -1.7940548102008265, -1.5028379599200647, -0.41095143953432256)
shf_way_pt = np.array([[-0.82222461061452856, -1.5587535549561358, -2.0142844897266556, -1.0569713950077662, 1.5327481491014201, 0.23544491609403506],
                       [-1.5591026208065346, -0.87423542232395968, 0.88383473320992845, -1.5660839378145119, 4.6582837735728653, 1.5947073375472192],
                       [-1.524196035766648, -0.79656827061021196, 1.0779153460316979, -1.8202038769048863, 4.6553167138444751, 1.5924384095196262],
                       [-1.535017077129013, -1.1815879036001613, 1.0709340290237206, -1.557706357404939, 4.6504297919388904, 1.5929620082952245],
                       [-1.4950490372583425, -1.5502014416213632, 1.0733774899765127, -1.3962634015954636, 4.6293113079897603, 1.6215854080279315],
                       [-1.5042992822939125, -1.2472122834751478, 1.073901088752111, -1.5341444125030159, 4.6266933141117681, 1.6214108751027323],
                       [-1.5100588688254937, -0.79656827061021196, 1.0751228192285072, -1.8051940453377351, 4.6223299909817817, 1.6214108751027323],
                       [-1.5334462808022178, -1.3646729421343662, 1.011069235680315, -1.7208946424664087, 4.6116834825446169, 1.5678292670665062]])

OBJ_LIST = ['O_00_Black_Tape', 'O_01_Glue', 'O_02_Big_USB', 'O_03_Glue_Stick', 'O_04_Big_Box',
            'O_05_Red_Cup', 'O_06_Small_Box', 'O_07_White_Tape', 'O_08_Small_USB',  'O_09_Yellow_Cup']

bkg_padding_img = cv2.imread('new_background\\1.bmp')[:28, :, :]
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


class Env:
    def __init__(self, socket_ip, opts):
        # Connect to Environment
        self.gripper = pyGrip.Gripper(host=socket_ip)             # Gripper
        self.rob = urx.Robot(socket_ip)                             # Robot

        # Dashboard Control
        self.Dashboard_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.Dashboard_socket.connect((socket_ip, 29999))
        self._program_send("")

        # Tray Control
        self.bluetooth = serial.Serial("COM5", 9600, timeout=1)     # Tray

        # Camera interface
        self.global_cam = Kinect()                         # Kinect Camera
        self.local_cam = UEyeCam()                         # Local Camera

        # Segmentetation Model
        self.segmentation_model = None
        self.obj_angle = 0

        # Robot
        self.acc = 1.5
        self.vel = 1.5
        self.available = True

        # Variables
        self.opts = opts
        self.render_width = opts.render_width
        self.render_height = opts.render_height
        self.num_cameras = opts.num_cameras
        self.repeats = opts.action_repeats
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
        self.target_angle = 0
        # maximum and minimum angles of the 6-th joint for orienting task
        self.max_ori = -100
        self.min_ori = 100


        # Make directory for save Data Path
        # DATA SAVER FOR PRE-TRAINING
        if self.opts.with_data_collecting:
            self.path_element = []
            self.max_num_list = []
            #self.max_num_list = np.zeros([10], dtype= np.uint8)
            dir_path = "E:\\save_data_ik_v2\\"

            [self.path_element.append(dir_path + str(x) + "\\") for x in np.arange(10)]
            if not os.path.exists(dir_path):
                [os.makedirs(x) for x in self.path_element]
                self.update_max_list()
            else:         # Make file indexing
                self.update_max_list()

        # Reset Environment
        self.set_tcp(self.default_tcp)
        self.movej(HOME)

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
        self.state = np.asarray(self.local_cam.snap())

    def set_segmentation_model(self, segmentation_model):
        self.segmentation_model = segmentation_model

    def reset(self, target_cls):
        self.movej(HOME)

        # Approaching
        self.approaching(target_cls)             # robot move

        if self.obj_pos is None or self.obj_pos[2] > 0.275 :  # 1.5는 로봇 기준 에러 난 z축 위치 스킵 하는거 추가하자. # 수치가 부정확하단거. 이거 확인해봐야함.
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

            with open(save_path + "{}_{}.txt".format(class_idx, num), "w") as f:   # (x, y, z, angle, symm)
                f.write("{} {} {} {} {}".format(*data, *self.eigen_value))

            self.max_num_list[class_idx] += 1

    def approaching(self, class_idx):
        seg_img, color_seg_img = self.get_seg()

        cv2.imshow("color_seg_img", color_seg_img)
        cv2.waitKey(10)

        # if the target class not exist, pass
        if class_idx not in np.unique(seg_img):
            print("Failed to find %s" % OBJ_LIST[class_idx], file=sys.stderr)
            self.obj_pos = None
            return

        else:
            self.obj_pos, self.eigen_value = self.get_obj_pos(seg_img, class_idx)

            if self.obj_pos is None:
                return
            elif self.obj_pos[0] < -0.306 and  self.obj_pos[1] < - 0.425:
                return
            elif self.obj_pos[0] > 0.292 and self.obj_pos[1] < - 0.425:
                return

            self.movej(starting_pose, 1, 1)      # Move to starting position,

            if class_idx == 5 and (self.obj_pos[2] + 0.1) < 0:
                goal = np.append(self.obj_pos + np.array([0, 0, 0.30]), [0, -3.14, 0])  # Initial point  Added z-dummy 0.05
            else:
                goal = np.append(self.obj_pos + np.array([0, 0, 0.1]), [0, -3.14, 0])      # Initial point  Added z-dummy 0.05

            self.movel(goal, 1, 1)

            obj_ang = self.segmentation_model.get_boxed_angle(class_idx)
            self.target_angle = np.rad2deg(self.getj()[-1]) - obj_ang

    def grasp(self):
        # Down move
        # 수정해야함
        a = self.obj_pos  # 확인 하기

        goal = np.append(self.obj_pos, self.getl()[3:])  # Initial point
        # 내리고
        self.movel(goal, 0.5, 0.5)
        self.gripper_close()  # 닫고

        # Move to position
        self.movej(starting_pose)
        self.movej(np.append(np.array([1.57]), self.getj()[1:]))

        # 물건 남아 있는지 확인
        seg, _ = self.get_seg()

        if self.target_cls in np.unique(seg):
            print("Failed Grasping..")

        # TODO : Reward 를 준다면?
        # if self.gripper.DETECT_OBJ and remain_obj.len() < 10:
        #     reward = 1
        # else:
        #     reward = -1

#        return reward

    def get_plus_or_minus_angle(self):
        target_object_orientation, symm = self.segmentation_model.get_angle(self.target_cls)
        target_boxed_angle = self.segmentation_model.get_boxed_angle(self.target_cls)

        plus_or_minus_angle_code = np.zeros((2), dtype=np.float)

        if symm is not None:
            # target_object_orientation = np.rad2deg(target_object_orientation)
            joint1_orientation = np.rad2deg(self.getj()[0])
            joint1_orientation = joint1_orientation - 90
            joint6_orientation = -1 * np.rad2deg(self.getj()[-1])
            current_joint6_orientation = joint1_orientation + joint6_orientation

            if symm == 1 and self.target_cls not in [0, 7]:
                shape = (4)
                target_object_orientation = target_boxed_angle
                target_object_orientations = np.zeros(shape=shape)
                target_object_orientations[0] = target_object_orientation

                if target_object_orientation >= -180 and target_object_orientation < -90:
                    target_object_orientations[1] = target_object_orientation + 90
                    target_object_orientations[2] = target_object_orientation + 180
                    target_object_orientations[3] = target_object_orientation + 270
                if target_object_orientation >= -90 and target_object_orientation < 0:
                    target_object_orientations[1] = target_object_orientation + 90
                    target_object_orientations[2] = target_object_orientation + 180
                    target_object_orientations[3] = target_object_orientation - 89
                if target_object_orientation >= 0 and target_object_orientation < 90:
                    target_object_orientations[1] = target_object_orientation + 90
                    target_object_orientations[2] = target_object_orientation - 90
                    target_object_orientations[3] = target_object_orientation - 179
                if target_object_orientation >= 90 and target_object_orientation < 180:
                    target_object_orientations[1] = target_object_orientation - 90
                    target_object_orientations[2] = target_object_orientation - 180
                    target_object_orientations[3] = target_object_orientation - 269

                diffs = target_object_orientations - current_joint6_orientation
                abs_diffs = abs(diffs)
                index = np.argmin(abs_diffs)
                target_object_orientation = target_object_orientations[index]

            elif self.target_cls in [0, 7] and symm == 1:
                target_object_orientation = 0
            else:
                shape = (2)
                target_object_orientation = target_boxed_angle
                target_object_orientations = np.zeros(shape=shape)
                target_object_orientations[0] = target_object_orientation

                if target_object_orientation <= 0:
                    target_object_orientations[1] = target_object_orientation + 179
                else:
                    target_object_orientations[1] = target_object_orientation - 179

                diffs = target_object_orientations - current_joint6_orientation
                abs_diffs = abs(diffs)
                index = np.argmin(abs_diffs)
                target_object_orientation = target_object_orientations[index]

            if current_joint6_orientation > target_object_orientation:
                self.max_ori = np.deg2rad(-joint6_orientation -1 * (target_object_orientation - 25) )
                self.min_ori = np.deg2rad(-joint6_orientation)
                plus_or_minus_angle_code[0] = 0   # move minus angle
                plus_or_minus_angle_code[1] = 1
            else:
                self.max_ori = np.deg2rad(-joint6_orientation)
                self.min_ori = np.deg2rad(-joint6_orientation -1 * (target_object_orientation + 25) )
                plus_or_minus_angle_code[0] = 1 # move plus angle
                plus_or_minus_angle_code[1] = 0

            return plus_or_minus_angle_code
        else:
            print("Error : calc_max_min_orientation_for_orienting_task, {}".format(symm), file=sys.stderr)
            return plus_or_minus_angle_code


    def calc_reward_for_orienting_task(self):
        target_object_orientation, symm = self.segmentation_model.get_angle(self.target_cls)
        target_object_orientation = np.rad2deg(target_object_orientation)
        target_boxed_angle = self.segmentation_model.get_boxed_angle(self.target_cls)

        joint1_orientation = np.rad2deg(self.getj()[0])
        joint1_orientation = joint1_orientation - 90
        joint6_orientation = -1 * np.rad2deg(self.getj()[-1])
        current_joint6_orientation = joint1_orientation + joint6_orientation

        if symm == 1 and self.target_cls not in [0, 7]:
            shape = (4)
            target_object_orientation = target_boxed_angle
            target_object_orientations = np.zeros(shape=shape)
            target_object_orientations[0] = target_object_orientation

            if target_object_orientation >= -180 and target_object_orientation < -90:
                target_object_orientations[1] = target_object_orientation + 90
                target_object_orientations[2] = target_object_orientation + 180
                target_object_orientations[3] = target_object_orientation + 270
            if target_object_orientation >= -90 and target_object_orientation < 0:
                target_object_orientations[1] = target_object_orientation + 90
                target_object_orientations[2] = target_object_orientation + 180
                target_object_orientations[3] = target_object_orientation - 89
            if target_object_orientation >= 0 and target_object_orientation < 90:
                target_object_orientations[1] = target_object_orientation + 90
                target_object_orientations[2] = target_object_orientation - 90
                target_object_orientations[3] = target_object_orientation - 179
            if target_object_orientation >= 90 and target_object_orientation < 180:
                target_object_orientations[1] = target_object_orientation - 90
                target_object_orientations[2] = target_object_orientation - 180
                target_object_orientations[3] = target_object_orientation - 269

            abs_diffs = abs(target_object_orientations - current_joint6_orientation)
            index = np.argmin(abs_diffs)
            target_object_orientation = target_object_orientations[index]

            dist = abs_diffs[index] / 180
            print("Target orientation : {}, Joint6_orientation : {}".format(target_object_orientation, current_joint6_orientation))

        elif self.target_cls in [0, 7] and symm == 1:
            # Tape : target angle 0'
            target_object_orientation = 0
            diff = target_object_orientation - current_joint6_orientation
            dist = abs(diff) / 180
            print("Target orientation : {}, Joint6_orientation : {}".format(target_object_orientation, current_joint6_orientation))
        else:
            # ver.1
            # diff = np.abs(self.target_angle - np.rad2deg(self.getj()[-1]))

            ## Ver. 2
            diff = np.zeros([2])
            diff[0] = np.abs(self.target_angle - np.rad2deg(self.getj()[-1]))
            diff[1] = 180 - diff[0]   # ? absolute??
            min_diff = np.min(diff)

            print("Target orientation : {}, Joint6_orientation : {}, absolute difference : {}".format(target_object_orientation, current_joint6_orientation, min_diff))
            dist = min_diff / 180

        return dist * -0.5

    def calc_max_min_orientation_for_orienting_task(self):
        target_object_orientation, symm = self.segmentation_model.get_angle(self.target_cls)
        target_boxed_angle = self.segmentation_model.get_boxed_angle(self.target_cls)

        if symm is not None:
            # target_object_orientation = np.rad2deg(target_object_orientation)
            joint1_orientation = np.rad2deg(self.getj()[0])
            joint1_orientation = joint1_orientation - 90
            joint6_orientation = -1 * np.rad2deg(self.getj()[-1])
            current_joint6_orientation = joint1_orientation + joint6_orientation

            if symm == 1 and self.target_cls not in [0, 7]:
                shape = (4)
                target_object_orientation = target_boxed_angle
                target_object_orientations = np.zeros(shape=shape)
                target_object_orientations[0] = target_object_orientation

                if target_object_orientation >= -180 and target_object_orientation < -90:
                    target_object_orientations[1] = target_object_orientation + 90
                    target_object_orientations[2] = target_object_orientation + 180
                    target_object_orientations[3] = target_object_orientation + 270
                if target_object_orientation >= -90 and target_object_orientation < 0:
                    target_object_orientations[1] = target_object_orientation + 90
                    target_object_orientations[2] = target_object_orientation + 180
                    target_object_orientations[3] = target_object_orientation - 89
                if target_object_orientation >= 0 and target_object_orientation < 90:
                    target_object_orientations[1] = target_object_orientation + 90
                    target_object_orientations[2] = target_object_orientation - 90
                    target_object_orientations[3] = target_object_orientation - 179
                if target_object_orientation >= 90 and target_object_orientation < 180:
                    target_object_orientations[1] = target_object_orientation - 90
                    target_object_orientations[2] = target_object_orientation - 180
                    target_object_orientations[3] = target_object_orientation - 269

                diffs = target_object_orientations - current_joint6_orientation
                abs_diffs = abs(diffs)
                index = np.argmin(abs_diffs)
                target_object_orientation = target_object_orientations[index]

            elif self.target_cls in [0, 7] and symm == 1:
                target_object_orientation = 0
            else:
                shape = (2)
                target_object_orientation = target_boxed_angle
                target_object_orientations = np.zeros(shape=shape)
                target_object_orientations[0] = target_object_orientation

                if target_object_orientation <= 0:
                    target_object_orientations[1] = target_object_orientation + 179
                else:
                    target_object_orientations[1] = target_object_orientation - 179

                diffs = target_object_orientations - current_joint6_orientation
                abs_diffs = abs(diffs)
                index = np.argmin(abs_diffs)
                target_object_orientation = target_object_orientations[index]

            if current_joint6_orientation > target_object_orientation:
                self.max_ori = np.deg2rad(-joint6_orientation -1 * (target_object_orientation - 25) )
                self.min_ori = np.deg2rad(-joint6_orientation)
            else:
                self.max_ori = np.deg2rad(-joint6_orientation)
                self.min_ori = np.deg2rad(-joint6_orientation -1 * (target_object_orientation + 25) )

            return True
        else:
            print("Error : calc_max_min_orientation_for_orienting_task, {}".format(symm), file=sys.stderr)
            return False

    def step(self, action, exploration_noise, num_step, is_training):
        target_angle = self.getj()[5] + action
        is_terminal = True
        self.available = True
        reward = 0
        actual_action = 0

        if is_training:
            noise = exploration_noise.noise() / 2
        else:
            noise = 0

        if num_step == 1:
            self.available = self.calc_max_min_orientation_for_orienting_task()

        if self.available:
            if num_step == 1:
                if self.min_ori > target_angle:
                    if self.min_ori - target_angle > 0.4:   # 23 degree
                        old_max = self.max_ori
                        self.max_ori = self.min_ori
                        self.min_ori = self.min_ori - (old_max - self.min_ori)

                if self.max_ori < target_angle:
                    if target_angle - self.max_ori > 0.4:   # 23 degree
                        old_min = self.min_ori
                        self.min_ori = self.max_ori
                        self.max_ori = self.max_ori + (self.max_ori - old_min)

            if self.min_ori > target_angle:
                action = np.abs(target_angle - self.min_ori) + action + abs(noise)

            if target_angle > self.max_ori:
                action = action - np.abs(target_angle - self.max_ori) - abs(noise)

            target_angle = self.getj()[5] + action

            target_angle = np.append(self.getj()[:-1], target_angle)    # 6dim.

            actual_action = action
            self.movej(target_angle)                                   # Real robot move to goal
            self.done = False

            reward = self.calc_reward_for_orienting_task()

            self.state_update()
            self.internal_state_update()

        # TODO : Reward
        # reward = self.grasp()

        return np.copy(self.state), reward, actual_action, self.done, is_terminal, self.available

    def internal_state_update(self):    # last joint angle
        internal_state = []

        # Current angles 5 * 6 = 30
        for idx in range(6):
            internal_state.extend(np.array([-1.0, -0.5, 0, 0.5, 1]) - self.getj()[idx])

        # 30 + 3 e.e pose.
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
            self.movej(HOME, 1, 1)
            self.available = False

    def movel(self, goal_pose, acc=1.2, vel=1.2):
        try:
            self.rob.movel(goal_pose, acc, vel)
        except urx.RobotException:
            self.status_chk()

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
        self.movej(HOME)

        self.gripper_close()

        # Tray Control
        # #self.movej(shf_way_pt[0])  # random
        # self.movej(shf_way_pt[1])
        # self.gripper.move(104)                           # near handle open
        # self.movej(shf_way_pt[2])
        # self.gripper.move(229)                           # handle grip
        # self.movej(shf_way_pt[3])
        # self.movej(shf_way_pt[4])
        # time.sleep(2)                                    # stop delay 2~3 sec. # tray shuffle
        # self.movej(shf_way_pt[5])
        # self.movej(shf_way_pt[6])
        # self.gripper.move(104)                           # near handle open
        # self.movej(shf_way_pt[7])

        # MIX TRAY

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

            time.sleep(1)
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

            self.movej(starting_pose)

        self.movej(starting_pose)

        self.movej(HOME)
        self.gripper_open()

    def teaching_mode(self):
        pass

    @staticmethod
    def get_obj_name(object_index):
        return OBJ_LIST[object_index]

    def get_seg(self):
        img = self.global_cam.snap()  # Segmentation input image  w : 256, h : 128
        padded_img = cv2.cvtColor(np.vstack((bkg_padding_img, img)), cv2.COLOR_RGB2BGR)  # Color fmt    RGB -> BGR
        cv2.imshow("Input_image", padded_img)

        # Run Network.
        return self.segmentation_model.run(padded_img)

    def get_obj_pos(self, segmented_image, class_idx):  # TODO : class 0~8 repeat version
        self.target_cls = class_idx
        print(">> Target Object : ", self.get_obj_name(class_idx), file=sys.stderr)

        pxl_list, eigen_value = self.segmentation_model.getData(segmented_image, class_idx)  # Get pixel list
        xyz = self.global_cam.color2xyz(pxl_list)   # patched image's averaging pose [x, y, z]

        return [xyz, eigen_value]

    def set_tcp(self, tcp):
        self.rob.set_tcp(tcp)

    def shutdown(self):
        self._program_send("Shutting down")
