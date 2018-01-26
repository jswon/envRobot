"""
Vision based object grasping
latest Ver.180124
"""

# Robot
from Kinect_Snap import *
import socket
import pyGrip

# utils
import cv2
import serial
from sklearn import svm
from matplotlib import pyplot as plt

from util import *

# ----------------- Define -----------------------------------------------------
HOME = (90 * np.pi / 180, -90 * np.pi / 180, 0, -90 * np.pi / 180, 0, 0)
INITIAL_POSE = (0, -90 * np.pi / 180, 0, -90 * np.pi / 180, 0, 0)
starting_pose = [ 1.2985, -1.7579,  1.6851, -1.5005, -1.5715, -0.2758]

shf_pt = [[-1.5937, -1.3493, 1.6653, -1.8875, -1.5697, -1.5954],
          [-1.5839, -1.2321, 1.8032, - 2.1425, - 1.5727, -1.5859],
          [-1.567 , -1.4621, 1.8072, - 2.0248, - 1.6009, -1.5858],
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

bkg_padding_img = cv2.imread('new_background\\bkg_img.png')[:36, :, :]
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
        self.gripper = pyGrip.Gripper(host=socket_ip)             # Gripper
        self.rob = urx.Robot(socket_ip)                             # Robot

        # Dashboard Control
        self.Dashboard_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.Dashboard_socket.connect((socket_ip, 29999))
        self._program_send("")

        # Tray Control
        # self.bluetooth = serial.Serial("COM5", 9600, timeout=1)     # Tray

        # Camera interface
        self.global_cam = Kinect()                         # Kinect Camera

        # Segmentation Model
        self.seg_model = None
        self.obj_angle = 0
        self.steps = 0

        # Robot
        self.acc = 5
        self.vel = 5

        # Variables
        self.opts = opts
        self.render_width = opts.render_width
        self.render_height = opts.render_height
        self.default_tcp = [0, 0, 0.150, 0, 0, 0]  # (x, y, z, rx, ry, rz)
        self.done = False

        # States - Local cam img. w : 256, h : 256, c :3
        self.state_shape = (self.render_height, self.render_width, 3)
        self.state = np.empty(self.state_shape, dtype=np.float32)

        # object position
        self.target_cls = 0
        self.obj_pos = np.zeros([3])         # (x, y, z)
        self.eigen_value = np.zeros([2])
        self.target_angle = 0
        # maximum and minimum angles of the 6-th joint for orienting task

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
        self.movej(HOME, self.vel, self.acc)

        # msg = input("Use detahcer? (T/F)")
        msg = "F"
        if msg == "T":
            self.use_detacher = True
        else:
            self.use_detacher = False

        self.x_boundary = [-0.297, 0.3034]
        self.y_boundary = [-0.447, -0.226]  # 427 432s 447

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

    def set_segmentation_model(self, segmentation_model):
        self.seg_model = segmentation_model

    def reset(self, target_cls, n):
        self.steps = n
        self.movej(HOME, self.acc, self.vel)
        self.approaching(target_cls)             # robot move

    def approaching(self, target_cls):
        seg_img, color_seg_img = self.get_seg()

        cv2.imshow("Current", color_seg_img)
        cv2.waitKey(1)

        print(">> Target Object : ", OBJ_LIST[target_cls], file=sys.stderr)

        if target_cls in (np.unique(seg_img)):
            self.obj_pos, _ = self.get_obj_pos(target_cls)
        else:
            self.obj_pos = None
            return

        if self.use_detacher and (self.x_boundary[0]< self.obj_pos[0] < self.x_boundary[1]) and (self.y_boundary[0] < self.obj_pos[1] < self.y_boundary[1]):
            for i in range(2):
                _, color_seg_img = self.get_seg()
                cv2.imshow("Current", color_seg_img)
                cv2.imshow("Dir", color_seg_img)
                cv2.waitKey(1)
                self.detacher(target_cls, i)

        seg_img, color_seg_img = self.get_seg()
        cv2.imshow("Current", color_seg_img)
        cv2.imshow("Dir", color_seg_img)
        cv2.waitKey(1)

        # if the target class not exist, pass
        if target_cls not in np.unique(seg_img):
            print("Failed to find %s" % OBJ_LIST[target_cls], file=sys.stderr)
            self.obj_pos = None
            return

        else:
            self.obj_pos, _ = self.get_obj_pos(target_cls)

            # y축 보정
            # self.obj_pos[1] -= 0.014

            if self.obj_pos is None:
                return

            # Safe zone        self.y_boundary = [-0.427, -0.226]
            if (self.x_boundary[0]< self.obj_pos[0] < self.x_boundary[1]) and (self.y_boundary[0] < self.obj_pos[1] < self.y_boundary[1]):
                self.movej(starting_pose, self.acc, self.vel)      # Move to starting position,

                if target_cls == 5 and self.obj_pos[2] < -0.1:
                    goal = np.append(self.obj_pos + np.array([0, 0, 0.30]), [0, -3.14, 0])  # Initial point  Added z-dummy 0.05
                else:
                    goal = np.append(self.obj_pos + np.array([0, 0, 0.1]), [0, -3.14, 0])      # Initial point  Added z-dummy 0.05

                self.movel(goal, self.acc, self.vel)

                obj_ang = self.seg_model.get_boxed_angle(target_cls)
                self.target_angle = np.rad2deg(self.getj()[-1]) - obj_ang
            else:
                self.obj_pos = None

    def grasp(self, target_cls):
        # Target angle orienting
        if target_cls in [0, 7]:
            obj_angle = np.array([0])
        else:
            obj_angle = self.seg_model.get_boxed_angle(target_cls)
            obj_angle = np.deg2rad(obj_angle)

        goal = np.append(self.rob.getj()[:-1], self.rob.getj()[-1] - obj_angle)
        self.rob.movej(goal, self.acc, self.vel)

        # 내리고
        self.obj_pos[2] = -0.058
        goal = np.append(self.obj_pos, self.getl()[3:])  # Initial point

        self.movel(goal, 0.5, 0.5)
        self.gripper_close()  # 닫고

        # Move to starting_pose
        self.movej(starting_pose, self.acc, self.vel)

        # Move to obj fixed point
        self.movej(j_pt[target_cls], self.acc, self.vel)

        l = self.getl()
        l[2] -= 0.057
        self.movel(l, 1, 1)
        self.gripper_open()
        l = self.getl()
        l[2] += 0.057
        self.movel(l, 1, 1)

    def detacher(self, target_cls, i):
        seg_img, color_img = self.get_seg()
        seg_img = np.array(seg_img)

        end_pt = [0, 0]
        start_pt = [0, 0]

        sorted_neighboring_obj = self.seg_model.find_neighboring_obj(seg_img, target_cls)

        if len(sorted_neighboring_obj) == 0:
            return

        center_x = 0.
        center_y = 0.

        for n_obj in [sorted_neighboring_obj[0]]:
            # Linear classification
            seg_img[0] = 127 - seg_img[0]
            target_pt = np.argwhere(seg_img == target_cls).astype("float64")
            compare_pt = np.argwhere(seg_img == n_obj).astype("float64")
            center_x = np.mean(np.concatenate((target_pt, compare_pt)), axis=0)[1]
            center_y = np.mean(np.concatenate((target_pt, compare_pt)), axis=0)[0]

            target_label = [0] * target_pt.shape[0]
            compare_label = [1] * compare_pt.shape[0]

            data = np.concatenate((target_pt, compare_pt))
            temp = np.array([data[:, 1], data[:, 0]]).transpose()
            label = np.concatenate((target_label, compare_label))

            clf = svm.SVC(kernel='linear', gamma=0.7, C=1)
            clf.fit(temp, label)

            w = clf.coef_[0]
            w_temp = np.copy(w[1])

            orthogonal = 0

            if -10e-3 < w_temp < 10e-3:                                      # 수직으로 그래프가 나오면 계산이 안됨.
                xxx = [center_x] * 128
                yyy = list(range(0, 128))
                orthogonal = 1
            else:                                                             # 수직 아닐때. 계산은 됨.
                g = -w[0] / w_temp                                           # Decision Boundary 의 기울기.

                xx = np.arange(0, 256)
                yy = g * xx - (clf.intercept_[0]) / w[1]
                true_range = np.where(np.logical_and(yy < 128, yy > 0))

                if g > 0:
                    min_idx = true_range[0][0]
                    max_idx = true_range[0][-1]
                else:
                    min_idx = true_range[0][-1]
                    max_idx = true_range[0][0]

                xxx = np.linspace(xx[min_idx], xx[max_idx])
                yyy = g * xxx - (clf.intercept_[0]) / w[1]

            distance = np.inf
            new_center_x = 0
            new_center_y = 0

            for x_1, y_1 in zip(xxx, yyy):
                temp_dist = np.linalg.norm([x_1 - center_x, y_1 - center_y])

                if temp_dist < distance:
                    distance = temp_dist
                    new_center_x = x_1
                    new_center_y = y_1

            # 수직일 때.
            across_upper = 0
            across_under = 0

            if orthogonal == 1:
                x_st = x_et = int(round(new_center_x))

                y_under_range = list(reversed(np.linspace(0, new_center_y)))
                y_upper_range = np.linspace(new_center_y, 127)

                for y_st in y_under_range:
                    m = 0

                    for i in range(-4, 5):
                        for j in range(-4, 5):
                            y1 = int(round(j + y_st))

                            try:
                                abc = seg_img[y1, x_st]
                                if abc != 10:
                                    m += 1
                            except IndexError:
                                pass

                    if m == 0:
                        y_st = int(round(y_st))
                        # plt.scatter(x_st, y_st, marker='x', c='red')
                        start_pt = [y_st, x_st]
                        break

                for y_et in y_upper_range:
                    m = 0
                    for i in range(-4, 5):
                        for j in range(-4, 5):
                            y_et_1 = int(round(j + y_et))
                            try:
                                abc = seg_img[y_et_1, x_et]
                                if abc != 10:
                                    m += 1
                            except IndexError:
                                pass

                    if m == 0:
                        y_et = int(round(y_et))
                        # plt.scatter(x_et, y_et, marker='x', c='blue')
                        end_pt = [y_et, x_et]
                        break

            # 수직이 아닐 떄
            else:
                if g > 0:
                    x_under_range = list(reversed(np.linspace(xx[min_idx], new_center_x)))
                    x_upper_range = np.linspace(new_center_x, xx[max_idx])
                else:
                    x_under_range = np.linspace(new_center_x, xx[min_idx])
                    x_upper_range = list(reversed(np.linspace(xx[max_idx], new_center_x)))

                for x_st in x_under_range:
                    y_st = g * x_st - (clf.intercept_[0]) / w[1]
                    m = 0

                    for i in range(-4, 5):
                        for j in range(-4, 5):
                            x1 = int(round(i + x_st))
                            y1 = int(round(j + y_st))

                            try:
                                abc = seg_img[y1, x1]
                                if abc != 10:
                                    m += 1
                                    across_under = 1

                            except IndexError:
                                pass

                    if m == 0:
                        x_st = int(round(x_st))
                        y_st = int(round(y_st))
                        start_pt = [y_st, x_st]
                        break

                for x_et in x_upper_range:
                    y_et = g * x_et - (clf.intercept_[0]) / w[1]
                    m = 0

                    for i in range(-4, 5):
                        for j in range(-4, 5):
                            x_et_1 = int(round(i + x_et))
                            y_et_1 = int(round(j + y_et))
                            try:
                                abc = seg_img[y_et_1, x_et_1]
                                if abc != 10:
                                    m += 1
                                    across_upper = 1
                            except IndexError:
                                pass

                    if m == 0:
                        x_et = int(round(x_et))
                        y_et = int(round(y_et))
                        end_pt = [y_et, x_et]
                        break

                if across_under == 1:
                    print("across_under")
                if across_upper == 1:
                    print("across_upper")

        if (start_pt[0] + end_pt[0])/2 > 64:
            end_pt, start_pt = start_pt, end_pt

        if start_pt[0] < 40:
            start_pt[0] = 40
        elif start_pt[0] > 107:
            start_pt[0] = 106
        if end_pt[0] < 40:
            end_pt[0] = 40
        elif end_pt[0] > 107:
            end_pt[0] = 106

        if start_pt[1] < 12:
            start_pt[1] = 12
        elif start_pt[1] > 243:
            start_pt[1] = 243
        if end_pt[1] < 12:
            end_pt[1] = 12
        elif end_pt[1] > 243:
            end_pt[1] = 243

        if seg_img[start_pt[0], start_pt[1]] != 10:
            return

        # 2/3 push
        mid_pt = [0, 0]
        mid_pt[0] = int(round((start_pt[0] + 2 * end_pt[0]) / 3))
        mid_pt[1] = int(round((start_pt[1] + 2 * end_pt[1]) / 3))
        end_pt = mid_pt

        cv2.line(color_img, (end_pt[1], end_pt[0]), (start_pt[1], start_pt[0]), (0, 255, 0), 1)
        cv2.circle(color_img, (end_pt[1], end_pt[0]), 2, (255, 255, 255), -1)
        cv2.circle(color_img, (start_pt[1], start_pt[0]), 2, (0, 0, 255), -1)

        cv2.imshow("Dir", color_img)
        cv2.waitKey(10)

        # starting to end point
        start_xyz = self.global_cam.color2xyz([start_pt])
        goal_xyz = self.global_cam.color2xyz([end_pt])  # patched image's averaging pose [x, y, z]

        # y축 보정
        # goal_xyz[1] -= 0.014
        # start_xyz[1] -= 0.014

        # z-Axis
        goal_xyz[2] = -0.0555
        start_xyz[2] = -0.0555

        self.gripper_close(255)
        self.movej(starting_pose, self.acc, self.vel)

        move_start_pt = np.append(start_xyz, [0, -3.14, 0])  # Initial point  Added z-dummy 0.05
        move_end_pt = np.append(goal_xyz, [0, -3.14, 0])  # Initial point  Added z-dummy 0.05

        self.movel(move_start_pt + np.array([0, 0, 0.15, 0, 0, 0]), 2, 2)
        self.movel(move_start_pt, 1, 1)

        self.movel(move_end_pt, 0.6, 0.6)
        self.movel(move_end_pt + np.array([0, 0, 0.15, 0, 0, 0]), 2, 2)
        self.movej(HOME, self.acc, self.vel)
        self.gripper_open(255)

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
            self.movej(HOME, self.acc, self.vel)

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

    def gripper_close(self, spd=50, force=80):
        self.gripper.close(spd, force)

    def gripper_open(self, spd=50, force=80):
        self.gripper.open(spd, force)

    def shuffle_obj(self):
        a, v = [self.acc, self.vel]
        self.movej(HOME, a, v)

        # Tray Control
        self.movej(shf_pt[0], a, v)
        self.gripper.move(104)
        self.movej(shf_pt[1], a, v)
        self.gripper.close(255)
        self.movej(shf_pt[2], a, v)
        self.movej(shf_pt[3], a, v)
        time.sleep(1)
        self.movej(shf_pt[2], a, v)
        self.movej(shf_pt[1], a, v)
        self.gripper.move(104)
        self.movej(shf_pt[0], a, v)
        self.gripper.open(255)

        # MIX TRAY

        # self.gripper_close()
        #
        # pt = [[-0.20, -0.45, -0.0484, -2.18848, -2.22941, 0.05679],
        #       [0.20, -0.45, -0.0484, -2.18848, -2.22941, 0.05679],
        #       [0.20, -0.340179, -0.0484, -2.18848, -2.22941, 0.05679],
        #       [-0.20, -0.340179, -0.0484, -2.18848, -2.22941, 0.05679],
        #       [-0.20, -0.217842, -0.0484, -0.01386,  3.08795, 0.39780],
        #       [0.20, -0.217842, -0.0484, -0.01386, 3.08795, 0.39780]]
        #
        # dir_idx = random.sample([0, 1, 2, 3], 1)  # 리스트에서 6개 추출
        #
        # dir = [[0, 1, 2, 3, 4, 5],
        #        [1, 0, 3, 2, 5, 4],
        #        [4, 5, 2, 3, 0, 1],
        #        [5, 4, 3, 2, 1, 0]]
        #
        # for idx in dir_idx:
        #     self.bluetooth.write("1".encode())
        #     self.bluetooth.write("1".encode())
        #
        #     time.sleep(1)
        #     for i, point in enumerate(dir[idx]):
        #         if i == 0:
        #             pt_copy = copy.deepcopy(pt)
        #             pt_copy[point][0] = pt_copy[point][0] + (0.05 * pow(-1, point + 1))
        #             pt_copy[point][2] = pt_copy[point][2] + 0.05
        #             self.movej(starting_pose)
        #             self.movel(pt_copy[point])
        #             pt_copy[point][2] = pt_copy[point][2] - 0.05
        #             self.movel(pt_copy[point])
        #         else:
        #             self.movel(pt[point])
        #
        #     self.movej(starting_pose)
        #
        # self.movej(starting_pose)

        self.movej(HOME, a, v)
        self.gripper_open()

        # msg = input("Use detacher? (T/F)")
        msg = 'F'
        if msg == 'T':
            self.use_detacher = True
        else:
            self.use_detacher = False

    def get_seg(self):
        img = self.global_cam.snap()  # Segmentation input image  w : 256, h : 128
        padded_img = cv2.cvtColor(np.vstack((bkg_padding_img, img)), cv2.COLOR_RGB2BGR)  # Color fmt    RGB -> BGR

        show_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        cv2.imshow("Input_image", show_img)

        return self.seg_model.run(padded_img)

    def get_obj_pos(self, target_cls):  # TODO : class 0~8 repeat version
        _, _ = self.get_seg()
        pxl_list, eigen_value = self.seg_model.getData(target_cls)  # Get pixel list

        xyz = None
        if pxl_list is not None:
            xyz = self.global_cam.color2xyz(pxl_list)   # patched image's averaging pose [x, y, z]

        return [xyz, eigen_value]

    def set_tcp(self, tcp):
        self.rob.set_tcp(tcp)

    def shutdown(self):
        self._program_send("Shutdown")
