"""
For grasping using IK v2, Data collect for pre-training
latest Ver.171031
"""

# Robot
import urx
from Kinect_Snap import *
from pyueye import *
import socket
import pyGrip

# utils
import copy
import cv2
import serial
from ou_noise import OUNoise

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
        chk = self._program_send("")

        # Tray Control
        self.bluetooth = serial.Serial("COM5", 9600, timeout=1)     # Tray

        # Camera interface
        self.global_cam = Kinect()                         # Kinect Camera
        self.local_cam = UEyeCam()                         # Local Camera

        # Segmentetation Model
        self.segmentation_model = None

        # Robot
        self.acc = 1.5
        self.vel = 1.5

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
        self.action_noise = OUNoise(6, 0, theta=0.2, sigma=0.04)


        # Action Dim
        self.action_dim = opts.action_dim    # 1dim?

        # object position
        self.target_cls = 0
        self.obj_pos = np.zeros([3])         # (x, y, z)
        self.depth_f = np.zeros([1])

        # Make directory for save Data Path
        # DATA SAVER FOR PRE-TRAINING
        if self.opts.with_data_collecting:
            self.path_element = []
            self.max_num_list = []
            dir_path = "E:\\local_pre-set\\"
            [self.path_element.append(dir_path + str(x) + "\\") for x in np.arange(10)]
            if not os.path.exists(dir_path):
                [os.makedirs(x) for x in self.path_element]

            else:         # Make file indexing
                self.update_max_list()

        # Reset Environment
        self.set_tcp(self.default_tcp)
        self.movej(HOME)

        print("Robot Environment Ready.", file=sys.stderr)

    def update_max_list(self):
        self.max_num_list = []
        for path in self.path_element:
            dummy = []
            if len(os.listdir(path)) != 0:
                for i, x in enumerate(os.listdir(path)):
                    dummy.append(int(x[:-4].split('_')[1::2][0]))  # file num in
                self.max_num_list.append(max(dummy)+1)

            else:
                self.max_num_list.append(1)

    def state_update(self):
        self.state = np.asarray(self.get_camera(2))

    def set_segmentation_model(self, segmentation_model):
        self.segmentation_model = segmentation_model

    def reset(self):
        """
        Robot Reset
        :return: state
        """
        self.movej(HOME)
        self.gripper.open()

        # Approaching
        for target_obj_idx in range(9):
            self.movej(HOME)
            time.sleep(2)
            self.approaching(target_obj_idx)             # robot move


            # if self.opts.with_data_collecting:
            #   self.store_data(self.target_cls)

        return np.copy(self.state)

    def store_data(self, class_idx):
        save_path = self.path_element[class_idx]
        num = self.max_num_list[class_idx]

        # save_image
        cv2.imwrite(save_path+"{}_{}.bmp".format(class_idx, num), self.state)

        data = self.getl()[0:3] - self.obj_pos
        with open(save_path + "{}_{}.txt".format(class_idx, num), "w") as f:   # (x, y, z, base angle)
            f.write("{} {} {}".format(*data))
        self.max_num_list[class_idx] += 1

    def approaching(self, class_idx):
        seg_img = self.get_seg()

        # if the target class not exist, pass
        if class_idx not in np.unique(seg_img):
            return

        else:
            self.obj_pos = self.get_obj_pos(seg_img, class_idx)

            self.movej(INITIAL_POSE)    # Move to center
            self.movej(starting_pose, 1, 1)      # Move to starting position,

            goal = np.append(self.obj_pos + np.array([0, 0, 0.1]), [0, -3.14, 0])      # Initial point  Added z-dummy 0.05

            for _ in range(5):  # repeat
                time.sleep(1)
                self.movel(goal, 1, 1)
                a_n = self.action_noise.noise()
                a_n[4:] = 0
                noisy_goal = self.rob.getj() + a_n
                self.movej(noisy_goal, 1, 1)
                self.state_update()            # local view

                if self.opts.with_data_collecting:
                    self.store_data(self.target_cls)

            time.sleep(1)
            self.movel(goal, 1, 1)

            self.state_update()  # local view
            self.internal_state_update()  # Internal State Update. Last joint angles

            if self.opts.with_data_collecting:
                self.store_data(self.target_cls)

    def grasp(self):
        # Down move
        self.obj_pos -= [0, 0, 0.05]  # subtract z-dummy 0.05
        goal = np.append(self.obj_pos, self.getl()[3:])  # Initial point
        self.movel(goal)

        self.gripper_close()

        # Move to position
        self.movej(starting_pose)
        self.movej(np.append(np.array([1.57]), self.getj()[1:]))

        seg = self.get_seg()

        remain_obj = self.segmentation_model.getPoints(seg, self.target_cls)

        if self.gripper.DETECT_OBJ and remain_obj < 10:
            reward = 1
        else:
            reward = -1

        return reward

    def step(self, action, exploration_noise, is_training):
        target_angle = self.getj()[5] + action
        is_terminal = True

        if is_training:
            noise = exploration_noise.noise() / 2
        else:
            noise = 0

        min_ori = -1.570796326
        max_ori = 1.570796326

        if min_ori > target_angle:
            action = np.abs(target_angle - min_ori) + action + abs(noise)
        elif target_angle > max_ori:
            action = action - np.abs(target_angle - max_ori) - abs(noise)

        target_angle = self.getj()[5] + action

        target_angle = np.append(self.getj()[:-1], target_angle)    # 6dim.

        actual_action = action
        self.movej(target_angle)                                   # Real robot move to goal
        self.done = False

        # reward = self.grasp()
        reward = 1  # temp reward

        self.done = True

        return np.copy(self.state), reward, actual_action, self.done, is_terminal

    def internal_state_update(self):    # last joint angle
        self.internal_state = np.array([-1.0, -0.5, 0, 0.5, 1]) * self.getj()[-1]

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
        if safetymode == "PROTECTIVE_STOP":
            self._program_send("unlock protective stop\n")
        if safetymode == "SAFEGUARD_STOP":
            self._program_send("close safety popup\n")

    def set_gripper(self, speed, force):
        self.gripper.set_gripper(speed, force)

    def gripper_close(self):
        self.gripper.close()

    def gripper_open(self):
        self.gripper.open()

    def shuffle_obj(self):
        self.movej(HOME)

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

        pt = [[-0.25099, -0.45, -0.04704, -2.18848, -2.22941, 0.05679],
              [0.25099, -0.45, -0.04704, -2.18848, -2.22941, 0.05679],
              [0.26807, -0.340179, -0.04704, -2.18848, -2.22941, 0.05679],
              [-0.26807, -0.340179, -0.04704, -2.18848, -2.22941, 0.05679],
              [-0.26807, -0.217842, -0.05096, -0.01386,  3.08795, 0.39780],
              [0.26807, -0.217842, -0.05096, -0.01386, 3.08795, 0.39780]]

        dir_idx = np.random.randint(0, 4, 1)[0]

        dir = [[0, 1, 2, 3, 4, 5],
               [1, 0, 3, 2, 5, 4],
               [4, 5, 2, 3, 0, 1],
               [5, 4, 3, 2, 1, 0]]

        for i, point in enumerate(dir[dir_idx]):
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

        self.bluetooth.write("1".encode())
        self.movej(starting_pose)
        self.movej(HOME)
        time.sleep(4)  # waiting

    def get_camera(self, camera_num):
        if camera_num == 1:
            return self.global_cam.snap()

        if camera_num == 2:
            return self.local_cam.snap()

    def teaching_mode(self):
        # TODO : Reserved
        pass

    @staticmethod
    def get_obj_name(object_index):
        return OBJ_LIST[object_index]

    def get_seg(self):
        time.sleep(4)
        img = self.global_cam.snap()  # Segmentation input image  w : 256, h : 128
        padded_img = cv2.cvtColor(np.vstack((bkg_padding_img, img)), cv2.COLOR_RGB2BGR)  # Color fmt    RGB -> BGR

        # Run Network.
        return self.segmentation_model.run(padded_img)

    def get_obj_pos(self, segmented_image, class_idx):  # TODO : class 0~8 repeat version
        self.target_cls = class_idx
        print(self.get_obj_name(class_idx))

        pxl_list = self.segmentation_model.getPoints(segmented_image, class_idx)  # Get pixel list
        xyz = self.global_cam.color2xyz(pxl_list)   # patched image's averaging pose [x, y, z]

        return xyz

    def set_tcp(self, tcp):
        self.rob.set_tcp(tcp)

    def shutdown(self):
        pass
