"""
For grasping using IK v2,
with calibration,
latest Ver.171025
"""

# system
import os

# Robot
import urx
from Kinect_Snap import *
from pyueye import *
import socket
import pyGrip

# utils
import random
import cv2
import serial
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

grasp_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

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
        # self.gripper = pyGrip.Gripper(host=socket_ip)             # Gripper
        self.rob = urx.Robot(socket_ip)                             # Robot

        # Dashboard Control
        self.Dashboard_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.Dashboard_socket.connect((socket_ip, 29999))

        # Tray Control
        # self.bluetooth = serial.Serial("COM5", 9600, timeout=1)     #  Tray

        # Camera interface
        self.global_cam = Kinect()                         # Kinect Camera
        self.local_cam = UEyeCam()                         # Local Camera

        # Robot
        self.acc = 1.5
        self.vel = 1.5

        # Variables
        self.opts = opts
        self.render_width = opts.render_width
        self.render_height = opts.render_height
        self.num_cameras = opts.num_cameras
        self.repeats = opts.action_repeats
        self.default_tcp = [0, 0, 0.150, 0, 0, 0]                                     # (x, y, z, rx, ry, rz)     # TODO: TCP z position
        self.done = False

        # States - Local cam img. w : 256, h : 256, c :3
        self.state_shape = (self.render_height, self.render_width, 3)
        self.state = np.empty(self.state_shape, dtype=np.float32)
        self.internal_state = np.empty([6], dtype=np.float32)

        # Action Dim
        self.action_dim = opts.action_dim    # 1dim?

        # object position
        self.obj_pos = np.zeros([3])         # (x, y, z)
        self.depth_f = np.zeros([1])

        # Make directory for save Data Path
        # TODO DATA SAVER FOR PRE-TRAINING
        if self.opts.with_data_collecting:
            self.path_element = []
            self.max_num_list = []
            dir_path = "E:\\local_pre-set\\"
            [self.path_element.append(dir_path + str(x) + "\\") for x in grasp_list]
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
        self.state = self.get_camera(2)

    def set_segmentation_model(self, segmentation_model):
        self.segmentation_model = segmentation_model

    def reset(self, target_obj, num_data):
        # Robot Reset
        self.movej(HOME)
        # self.gripper.open()

        # Approaching
        target_obj_idx = int(np.argmax(target_obj))

        self.approaching(target_obj_idx)             # robot move
        self.state_update()            # local view
        self.internal_state_update()   # Internal State Update. Last joint angles

        if self.opts.with_data_collecting and num_data > 0:
            self.store_data(target_obj_idx)

        return np.copy(self.state)

    def store_data(self, class_idx):
        save_path = self.path_element[class_idx]
        num = self.max_num_list[class_idx]

        # save_image
        cv2.imwrite(save_path + "{}_{}.bmp".format(class_idx, num), self.state)  # Save the current image

        # TODO : SAVE END - OBJ_POSE, TARGET_OBJ_
        # data = self.getl[0:3] - self.obj_pos
        data = [-1, -2, 3]
        with open(save_path + "{}_{}.txt".format(class_idx, num), "w") as f:   # (x, y, z, base angle)
            f.write("{} {} {}".format(*data))
        self.max_num_list[class_idx] += 1

    def approaching(self, class_idx):
        self.obj_pos = self.get_obj_pos(class_idx) + np.array([0, 0, 0.05])    # z-dummy 0.05
        self.movej(INITIAL_POSE)    # Move to center
        self.movej(starting_pose)      # Move to starting position,
        goal = np.append(self.obj_pos, [0, -3.14, 0])      # Initial point
        try :
            self.movel(goal)
        except urx.RobotException:
            pass

    def grasp(self):
        # Down move
        self.obj_pos -= [0, 0, 0.05]  # subtract z-dummy 0.05
        goal = np.append(self.obj_pos, self.getl()[3:])  # Initial point
        self.movel(goal)

        self.gripper_close()

        # Move to position
        self.movej(starting_pose)

        if self.gripper.DETECT_OBJ:
            reward = 1
        else:
            reward = -1

        return reward

    def step(self, action, target_obj, num_data, exploration_noise, is_training):
        # action = np.array(action)                # action 1 dim.
        target_angle = self.getj()[5] + action
        is_terminal = True

        # TODO : step
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
        reward = 1

        if self.opts.with_data_collecting and num_data > 0:
            self.store_data(target_obj)

        else:   # Collide, robot not move, hold state but update reward
            pass

        self.done = True

        return np.copy(self.state), reward, actual_action, self.done, is_terminal

    def internal_state_update(self):    # last joint angle
        self.internal_state = np.array([-1.0, -0.5, 0, 0.5, 1]) * self.getj()[-1]

    def getl(self):
        return self.rob.getl()

    def getj(self):
        return np.around(np.array(self.rob.getj()), decimals=4)

    def get_ef(self):
        return np.around(np.array(self.rob.getl()[0:3]), decimals=4)

    def movej(self, goal_pose, acc=1.5, vel=1.5):
        self.rob.movej(goal_pose, acc, vel)

    def movel(self, goal_pose, acc=1.5, vel=1.5):
        self.rob.movel(goal_pose, acc, vel)

    def set_gripper(self, speed, force):
        self.gripper.set_gripper(speed, force)

    def gripper_close(self):
        # TODO Maybe value < THRESHOLD -> Grasp
        self.gripper.close()
        if self.gripper.DETECT_OBJ:
            pass
            # TODO
            # k = 'DETECT_OBJ'

    def gripper_open(self):
        self.gripper.open()

    def shuffle_obj(self):
        self.movej(HOME)

        # chk = self.collision_chk()    # Why made this ?

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
        # self.gripper_close()

        pt = [[0.1507703842858799, -0.3141178727128849, -0.030569762928828032, 2.2260958131016335, -2.1985942225522668, 0.05813081679518341],
              [-0.1491136865872555, -0.31021647495551335, -0.03052286866235667, -2.1570671726727104, 2.2799584355377167, -0.029214990891798114],
              [0.18414105144817525, -0.25296564835782714, -0.021194048759978847, -2.238562873918888, 2.1370170873680085, 0.10380902269948922],
              [-0.1417132578497845, -0.26528795089888607, -0.03157633192772823, -2.1045306382633795, 2.243489988356314, 0.0360411778983116],
              [0.20295675954768064, -0.16877330661979667, -0.038960103050602234, 0.018988621965437776, -3.007839525712047, -0.875295581190713],
              [-0.13352761097358432, -0.16247247277870452, -0.025328902795204018, 0.15436650469465374, -2.6904628274570306, -1.555752726987423]]

        # TRAY
        # self.bluetooth.write("1".encode())

        # TODO Segmentation & Shuffle
        #region = self._obj_region_chk(*self.get_obj_pos()[:-1])  # Center : 1, 2, 3
        region = 0
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

        if camera_num == 2:
            # TODO : PIL
            # return Image.fromarray(self.local_cam.snap()).resize((256, 256), Image.NEAREST)
            return cv2.resize(self.local_cam.snap(), (256, 256))

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

    def teaching_mode(self):
        # TODO : Reserved
        pass

    @staticmethod
    def get_obj_name(object_index):
        return OBJ_LIST[object_index]

    def get_obj_pos(self, class_idx):
        img, self.depth_f = self.global_cam.snap()   # < #delay
        time.sleep(1)               # Segmentation input image  w : 256, h : 128
        padded_img = cv2.cvtColor(np.vstack((bkg_padding_img, img)), cv2.COLOR_RGB2BGR)    # Color fmt    RGB -> BGR
        cv2.imwrite("input.bmp", padded_img)
        # Run Network.
        segmented_image = self.segmentation_model.run(padded_img)
        color_img = self.segmentation_model.convert_grey_label_to_color_label(segmented_image)
        cv2.imwrite("result.bmp", color_img)

        # TODO : object index is not seg_index
        pxl_list = self.segmentation_model.getPoints(segmented_image, class_idx)

        test_pxl = np.copy(pxl_list)

        global_view = cv2.imread("view.bmp")

        for y, x in test_pxl:
            g_x = int((255 - x) * 3.035 + 573)   # Width revision, rate : 3.03515625, offset : 573
            g_y = int((y + 127) * 3.1953125 + 143)       # Height revision, rate : 3.1953125, offset : 143
            global_view[g_y, g_x, :] = np.array([0, 0, 255])

        cv2.imwrite("global_error.bmp", global_view)

        for [x, y] in pxl_list:
            padded_img[x, y] = np.array([0, 0, 255])

        cv2.imwrite("Error.bmp", padded_img)

        print("Start time : {}".format(datetime.datetime.now()))
        # TODO : Crop center test

        mean_pxl = np.mean(pxl_list, axis=0).astype(np.uint64)

        pxl_patch = []
        start_pxl = mean_pxl - np.array([2, 2])

        for i in range(5):
            for j in range(5):
                pxl_patch.append(start_pxl+np.array([i, j]))

        for y, x in pxl_patch:
            g_x = int((255 - x) * 3.035 + 573)   # Width revision, rate : 3.03515625, offset : 573
            g_y = int((y + 127) * 3.1953125 + 143)       # Height revision, rate : 3.1953125, offset : 143
            global_view[g_y, g_x, :] = np.array([255, 255, 0])

        cv2.imwrite("global_patch.bmp", global_view)

        xyz_list = []
        [xyz_list.append(self.global_cam.color2xyz(self.depth_f, i)) for i in pxl_patch]
        xyz_list = np.array(xyz_list)

        nan_idx = np.sort(np.transpose(np.argwhere(np.isnan(xyz_list))[0::3])[0])

        for x in reversed(nan_idx):
            xyz_list = np.delete(xyz_list, x, 0)

        # # # seg -> 256,256 -->>>> coordinate 1920 x 1024
        # seg_result_pixel = [cx, cy]
        # xyz = self.global_cam.color2xyz(self.depth_f, seg_result_pixel)

        print("Calculated num. : %s" % xyz_list.shape[0])

        # TODO : Average
        # 실제로는 모든 pixel 에 대해 average 예정
        mean_xyz = np.mean(xyz_list, axis=0)

        print("End time : {}".format(datetime.datetime.now()))

        with open("1.txt", "w") as f:   # (x, y, z, base angle)
            for data in xyz_list:
                f.write("{} {} {}\n".format(*data))

        return mean_xyz

    def set_tcp(self, tcp):
        self.rob.set_tcp(tcp)

    def shutdown(self):
        # TODO : Reserved Shutdown
        pass
