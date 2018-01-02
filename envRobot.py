"""
Vision based object grasping
latest Ver.180102
- Add a method that using dilation to find neighboring object
- Add a solution for objects out of bounds

"""

# Robot
from Kinect_Snap import *
import socket
import pyGrip

# utils
import cv2
import serial
import itertools

from util import *

# ----------------- Define -----------------------------------------------------
HOME = (90 * np.pi / 180, -90 * np.pi / 180, 0, -90 * np.pi / 180, 0, 0)
INITIAL_POSE = (0, -90 * np.pi / 180, 0, -90 * np.pi / 180, 0, 0)
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
        msg = True
        self.use_detacher = False

        if msg == "T":
            self.use_detacher = True
        else:
            self.use_detacher = False

        self.x_boundary = [-0.297, 0.3034]
        self.y_boundary = [-0.427, -0.226]

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

    def reset(self, target_cls):
        self.movej(HOME, self.acc, self.vel)
        self.approaching(target_cls)             # robot move

    def approaching(self, target_cls):
        seg_img, color_seg_img = self.get_seg()

        cv2.imshow("color_seg_img", color_seg_img)
        cv2.waitKey(10)

        print(">> Target Object : ", OBJ_LIST[target_cls], file=sys.stderr)

        if self.use_detacher:
            [self.detacher(seg_img, target_cls) for _ in range(3)]

        seg_img, color_seg_img = self.get_seg()
        # if the target class not exist, pass
        if target_cls not in np.unique(seg_img):
            print("Failed to find %s" % OBJ_LIST[target_cls], file=sys.stderr)
            self.obj_pos = None
            return

        else:
            self.obj_pos, _ = self.get_obj_pos(target_cls)

            # y축 보정
            self.obj_pos[1] -= 0.015

            if self.obj_pos is None:
                return

            # Safe zone
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

    def path_gen(self, seg, new_x, new_y, theta):
        starting_pt = [0, 0]
        end_pt = [0, 0]

        opposite_theta = np.radians([180]) + theta

        r = 20

        for w in range(128):
            next1 = 0

            start_x = new_x + (r + w) * np.cos(opposite_theta)
            start_y = new_y - (r + w) * np.sin(opposite_theta)

            start_xx = np.round(start_x).astype(np.int)
            start_yy = np.round(start_y).astype(np.int)

            for i in [-6, 0, 6]:
                for k in [-6, 0, 6]:
                    if seg[start_yy + k, start_xx + i] != 10:
                        next1 = 1

            if next1 == 0:
                starting_pt = [int(start_yy), int(start_xx)]
#                 cv2.line(color_img, (ccy, ccx), (start_xx, start_yy), (0, 255, 0))
                break

        return starting_pt, end_pt

    def detacher(self, seg, target_cls):
        pxl_boundary_x = [14, 241]
        pxl_boundary_y = [14, 113]

        color_img = self.seg_model.convert_grey_label_to_color_label(seg)

        attached_list = {"obj_list": [], "points": []}

        obj_list = np.unique(seg)[:-1]
        obj_list = np.delete(obj_list, np.argwhere(obj_list == target_cls))

        center_xy = np.mean(np.argwhere(seg == target_cls), axis=0)
        end_pt = [0., 0.]
        starting_pt = [0., 0.]

        shape = (128, 256)
        binary_image_array = np.zeros(shape=shape, dtype=np.uint8)
        binary_image_array.fill(0)
        target_cls_pointList = self.seg_model.getPoints(seg, target_cls)
        binary_image_array = self.seg_model.make_binary_label_array(target_cls_pointList, binary_image_array)

        kernel = np.ones((5, 5), np.uint8)
        img_dilation = cv2.dilate(binary_image_array, kernel, iterations=2)
        target_cls_dilat_pointList = self.seg_model.getPoints(img_dilation, 255)
        num_points = len(target_cls_dilat_pointList)
        close_obj_index_list = []

        for i in range(0, num_points):
            pixel = target_cls_dilat_pointList[i]
            y = pixel[0]
            x = pixel[1]

            label = seg[y, x]
            if label != target_cls and label != 10:
                close_obj_index_list.append(label)

        close_obj_index_list = np.unique(close_obj_index_list)

        for comp_idx in close_obj_index_list:
            comp_xy = np.mean(np.argwhere(seg == comp_idx), axis=0)

            attached_list['obj_list'].append(comp_idx)
            attached_list['points'].append(comp_xy)

        attached_list['num'] = attached_list['obj_list'].__len__()

        if attached_list['num'] == 0:
            return
        elif attached_list['num'] == 1:
            ccx, ccy = np.round(center_xy).astype(np.int)

            pt = attached_list['points'][0]

            center_temp = np.copy(center_xy)

            pt[0] = 127 - pt[0]
            center_temp[0] = 127 - center_temp[0]

            angle = np.degrees(np.arctan2(pt[0] - center_temp[0], pt[1] - center_temp[1]))

            angle = angle + 90
            if angle > 180:
                angle = angle + -360

            theta = np.radians(angle)

            new_x = center_xy[1] + 7 * np.cos(theta)
            new_y = 127 - (center_temp[0] + 10 * np.sin(theta))

            new_x = np.round(new_x).astype(np.int)  # goal_position
            new_y = np.round(new_y).astype(np.int)

            opposite_theta = np.radians([180]) + theta

            for w in range(128):
                next1 = 0
                start_x = new_x + (20 + w) * np.cos(opposite_theta)
                start_y = new_y - (20 + w) * np.sin(opposite_theta)

                start_xx = np.round(start_x).astype(np.int)
                start_yy = np.round(start_y).astype(np.int)

                for i in [-6, 0, 6]:
                    for k in [-6, 0, 6]:
                        if seg[start_yy + k, start_xx + i] != 10:
                            next1 = 1

                if next1 == 0:
                    starting_pt = [int(start_yy), int(start_xx)]
                    cv2.line(color_img, (ccy, ccx), (start_xx, start_yy), (0, 255, 0))
                    break

            new_xx = np.round(new_x).astype(np.int)
            new_yy = np.round(new_y).astype(np.int)

            end_pt = [new_yy, new_xx]  # int

        elif attached_list['num'] > 1:
            ccx, ccy = np.round(center_xy).astype(np.int)

            pt_list = np.arange(0, attached_list['points'].__len__())
            pt_points = attached_list['points']

            subset = list(itertools.combinations(pt_list, 2))
            init_array = []
            subset_angle = np.array(init_array)

            for pt_1, pt_2 in subset:
                temp_img = np.copy(color_img)

                xy1 = pt_points[pt_1]
                xy2 = pt_points[pt_2]

                xy11 = np.round(xy1).astype(np.int)
                xy22 = np.round(xy2).astype(np.int)

                cv2.line(temp_img, (ccy, ccx), (xy11[1], xy11[0]), (255, 255, 255))
                cv2.line(temp_img, (ccy, ccx), (xy22[1], xy22[0]), (255, 255, 255))

                th1 = np.arctan2((xy2[0] - center_xy[0]), (xy2[1] - center_xy[1]))
                th2 = np.arctan2((xy1[0] - center_xy[0]), (xy1[1] - center_xy[1]))

                dtheta = np.abs(th1 - th2)
                theta = np.min([dtheta, 6.28 - dtheta])
                subset_angle = np.append(subset_angle, np.degrees(theta))

            where_pt = 'none'  # 'inner' or 'outer'

            for t_ang in subset_angle:
                if attached_list['num'] == 2:
                    where_pt = 'outer'
                    target_angle = t_ang
                    break

                else:
                    temp = np.copy(subset_angle)
                    temp = np.delete(subset_angle, np.argwhere(temp == t_ang))

                    for case in np.array(list(itertools.combinations(temp, attached_list['num'] - 1))):
                        sum_others = np.round(np.sum(case))
                        sum_all = t_ang + sum_others

                        if 359 <= np.round(sum_all) < 361:
                            where_pt = 'inner'
                            target_angle = np.max(subset_angle)
                            break

                        else:
                            if sum_others - 1 <= t_ang < sum_others + 1:
                                where_pt = 'outer'
                                target_angle = t_ang
                                break

                            elif sum_others - 1 <= 360 - t_ang < sum_others + 1:
                                where_pt = 'inner'
                                target_angle = t_ang
                                break

                if where_pt is not 'none':
                    break

            comb_idx = np.where(subset_angle == target_angle)[0]
            pair = subset[comb_idx[0]]

            a_pt = pt_points[pair[0]]
            b_pt = pt_points[pair[1]]

            a_pt[0] = 127 - a_pt[0]
            b_pt[0] = 127 - b_pt[0]

            center_temp = np.copy(center_xy)
            center_temp[0] = 127 - center_temp[0]

            a_angle = np.degrees(np.arctan2(a_pt[0] - center_temp[0], a_pt[1] - center_temp[1]))
            b_angle = np.degrees(np.arctan2(b_pt[0] - center_temp[0], b_pt[1] - center_temp[1]))
            base_angle = 0

            # 작은놈이 기준.
            if (a_angle > 0) and (b_angle > 0):
                if a_angle > b_angle:
                    base_angle = b_angle
                else:
                    base_angle = a_angle

            # 둘 중 하나가 -1 일 때
            if a_angle * b_angle < 0:
                temp_angle_a = a_angle + target_angle / 2

                if temp_angle_a > 180:
                    temp_angle_a = -360 + temp_angle_a

                temp_angle_b = b_angle - target_angle / 2

                if temp_angle_b > 180:
                    temp_angle_b = -360 + temp_angle_b

                if -1 < temp_angle_a - temp_angle_b < 1:
                    base_angle = a_angle
                else:
                    base_angle = b_angle

            # 둘다 - 일때 더 작은놈이 기준.
            if (a_angle < 0) and (b_angle < 0):
                if a_angle > b_angle:
                    base_angle = b_angle
                else:
                    base_angle = a_angle

            if where_pt is "outer":
                target_angle = (360 - target_angle) / 2

                theta = base_angle - target_angle

                if theta < -180:
                    theta = 360 + theta

                theta = np.radians(theta)

                new_x = center_xy[1] + 7 * np.cos(theta)
                new_y = 127 - (center_temp[0] + 7 * np.sin(theta))

                if not (pxl_boundary_x[0] <= new_x < pxl_boundary_x[1]) and (pxl_boundary_y[0] <= new_y < pxl_boundary_y[1]):
                    theta += np.radians([180])
                    new_x = center_xy[1] + 7 * np.cos(theta)
                    new_y = 127 - (center_temp[0] + 7 * np.sin(theta))

                new_xx = np.round(new_x).astype(np.int)
                new_yy = np.round(new_y).astype(np.int)
                end_pt = [new_yy, new_xx]  # int

                cv2.line(color_img, (ccy, ccx), (new_xx, new_yy), (255, 255, 255))

                # Find starting point
                opposite_theta = np.radians([180]) + theta

                for w in range(128):
                    next1 = 0

                    start_x = new_x + (20 + w) * np.cos(opposite_theta)
                    start_y = new_y - (20 + w) * np.sin(opposite_theta)

                    start_xx = np.round(start_x).astype(np.int)
                    start_yy = np.round(start_y).astype(np.int)

                    for i in [-6, 0, 6]:
                        for k in [-6, 0, 6]:
                            if seg[start_yy + k, start_xx + i] != 10:
                                next1 = 1

                    if next1 == 0:
                        starting_pt = [int(start_yy), int(start_xx)]
                        cv2.line(color_img, (ccy, ccx), (start_xx, start_yy), (0, 255, 0))
                        break

            elif where_pt is "inner":
                target_angle = target_angle / 2
                theta = base_angle + target_angle

                if theta > 90:
                    theta = -360 + theta

                theta = np.radians(theta)

                new_x = center_xy[1] + 10 * np.cos(theta)
                new_y = 127 - (center_temp[0] + 10 * np.sin(theta))

                if not (pxl_boundary_x[0] <= new_x < pxl_boundary_x[1]) and (pxl_boundary_y[0] <= new_y < pxl_boundary_y[1]):
                    theta += np.radians([180])
                    new_x = center_xy[1] + 7 * np.cos(theta)
                    new_y = 127 - (center_temp[0] + 7 * np.sin(theta))

                # Find starting point
                opposite_theta = np.radians([180]) + theta

                for w in range(128):  # minimum r = 20
                    next1 = 0

                    start_x = new_x + (20 + w) * np.cos(opposite_theta)
                    start_y = new_y - (20 + w) * np.sin(opposite_theta)

                    start_xx = np.round(start_x).astype(np.int)
                    start_yy = np.round(start_y).astype(np.int)

                    for i in [-6, 0, 6]:
                        for k in [-6, 0, 6]:
                            if seg[start_yy + k, start_xx + i] != 10:
                                next1 = 1

                    if next1 == 0:
                        starting_pt = [int(start_yy), int(start_xx)]
                        cv2.line(color_img, (ccy, ccx), (start_xx, start_yy), (0, 255, 0))
                        break

                new_xx = np.round(new_x).astype(np.int)
                new_yy = np.round(new_y).astype(np.int)

                end_pt = [new_yy, new_xx]
                cv2.line(color_img, (ccy, ccx), (new_xx, new_yy), (255, 255, 255))

        cv2.imshow("Dir complete", color_img)
        cv2.waitKey(1)

        # starting to end point
        start_xyz = self.global_cam.color2xyz([starting_pt])
        goal_xyz = self.global_cam.color2xyz([end_pt])  # patched image's averaging pose [x, y, z]

        # y축 보정
        goal_xyz[1] -= 0.015
        start_xyz[1] -= 0.015

        # z-Axis
        goal_xyz[2] = -0.053
        start_xyz[2] = -0.053

        self.gripper_close(255)
        self.movej(starting_pose, self.acc, self.vel)

        move_start_pt = np.append(start_xyz, [0, -3.14, 0])  # Initial point  Added z-dummy 0.05
        move_end_pt = np.append(goal_xyz, [0, -3.14, 0])  # Initial point  Added z-dummy 0.05

        self.movel(move_start_pt + np.array([0, 0, 0.15, 0, 0, 0]), 2, 2)
        self.movel(move_start_pt, 2, 2)

        self.movel(move_end_pt, 1, 1)
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

        msg = input("Use detacher? (T/F)")
        msg = 'T'
        if msg == 'T':
            self.use_detacher = True
        else:
            self.use_detacher = False



    def get_seg(self):
        img = self.global_cam.snap()  # Segmentation input image  w : 256, h : 128
        padded_img = cv2.cvtColor(np.vstack((bkg_padding_img, img)), cv2.COLOR_RGB2BGR)  # Color fmt    RGB -> BGR
        cv2.imshow("Input_image", padded_img)

        return self.seg_model.run(padded_img)

    def get_obj_pos(self, target_cls):  # TODO : class 0~8 repeat version
        pxl_list, eigen_value = self.seg_model.getData(target_cls)  # Get pixel list
        xyz = self.global_cam.color2xyz(pxl_list)   # patched image's averaging pose [x, y, z]

        return [xyz, eigen_value]

    def set_tcp(self, tcp):
        self.rob.set_tcp(tcp)

    def shutdown(self):
        self._program_send("Shutdown")
