"""
Vision based object grasping

latest Ver.180411

데모용, 알고리즘 수정 버전. 버그 수정 전 백업.
"""

# Robot
from Kinect_Snap import *
import socket
import pyGrip
import urx

# utils
import cv2

import queue
import serial
from numpy.ma.core import true_divide
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

OBJ_LIST = ['Black_Tape', 'Glue', 'Big_USB', 'Glue_Stick', 'Big_Box',
            'Red_Cup', 'Small_Box', 'White_Tape', 'Small_USB',  'Yellow_Cup']

bkg_padding_img = cv2.imread('background/bkg_img.png')[:36, :, :]
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
        self.gripper_init_pose = 0
        self.all_clear = True

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
        self.acc = 4
        self.vel = 6

        self.start_pt = [0, 0]
        self.end_pt = [0, 0]

        # Variables
        self.opts = opts
        self.render_width = opts.render_width
        self.render_height = opts.render_height

        self.default_tcp = [0, 0, 0.150, 0, 0, 0]  # (x, y, z, rx, ry, rz)
        self.done = False

        # States - Local cam img. w : 256, h : 256, c :3
        # self.state_shape = (self.render_height, self.render_width, 3)
        # self.state = np.empty(self.state_shape, dtype=np.float32)

        # object position
        self.target_cls = 0
        self.obj_pos = np.zeros([3])         # (x, y, z)
        self.eigen_value = np.zeros([2])
        self.target_angle = 0

        # Make directory for save Data Path
        # DATA SAVER FOR PRE-TRAINING
        if self.opts.with_data_collecting:
            self.path_element = []
            self.max_num_list = []
            # self.max_num_list = np.zeros([10], dtype= np.uint8)
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

        # msg = input("Use scatter? (T/F)")
        msg = 'T'

        if msg == "T":
            self.use_scatter = True
        else:
            self.use_scatter = False

        self.x_boundary = [-0.297, 0.3034]
        self.y_boundary = [-0.447, -0.226]  # 427 432s 447

        self.input_img = np.empty((128, 256))
        self.vis_dir = False

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

    def env_img_update(self):
        seg_img, color_seg_img = self.get_seg()
        env_img = self.global_cam.scene()
        pxl_list = np.argwhere(np.array(seg_img) == self.target_cls)
        temp_img = np.zeros(shape=(128, 256), dtype=np.uint8)
        [temp_img.itemset((y, x), 255) for [y, x] in pxl_list]

        _, contours, _ = cv2.findContours(temp_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

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

            cv2.rectangle(env_img, (x, y + 128), (x + w, y + h + 128), (0, 0, 255), 1)
            cv2.rectangle(self.input_img, (x, y), (x + w, y + h), (0, 0, 255), 1)

        vert_img = np.vstack((self.input_img, color_seg_img))  # Color fmt    RGB -> BGR
        cur_scene = np.hstack((env_img, vert_img))

        if self.vis_dir is True :
            cv2.rectangle(cur_scene, (256, 128), (511, 255), (0, 255, 0), 1)

            self.end_pt[1] += 255
            self.end_pt[0] += 127
            self.start_pt[1] += 255
            self.start_pt[0] += 127

            cv2.line(cur_scene, (self.end_pt[1], self.end_pt[0]), (self.start_pt[1], self.start_pt[0]), (0, 255, 0), 1)
            cv2.circle(cur_scene, (self.end_pt[1], self.end_pt[0]), 2, (255, 255, 255), -1)
            cv2.circle(cur_scene, (self.start_pt[1], self.start_pt[0]), 2, (0, 0, 255), -1)

        # image show
        cur_scene = cv2.resize(cur_scene, (1024, 512))
        text_size = cv2.getTextSize("Target : {}".format(OBJ_LIST[self.target_cls]), 1, 2, 0)[0][0]

        cur_scene[:60, 1023 - text_size-20:] = (255, 255, 255)
        cv2.putText(cur_scene, "Target : {}".format(OBJ_LIST[self.target_cls]), (1023-text_size, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # cur_scene = cv2.resize(cur_scene, (2048, 1024)) For Demo
        cv2.imshow("Current Scene", cur_scene)
        cv2.waitKey(1)

    def approaching(self, target_cls):
        self.target_cls = target_cls
        seg_img, color_seg_img = self.get_seg()
        self.env_img_update()

        print(">> Target Object : ", OBJ_LIST[target_cls], file=sys.stderr)

        if target_cls in (np.unique(seg_img)):
            self.obj_pos, _ = self.get_obj_pos(target_cls)
        else:
            self.obj_pos = None
            return

        if self.use_scatter and (self.x_boundary[0] < self.obj_pos[0] < self.x_boundary[1]) and (self.y_boundary[0] < self.obj_pos[1] < self.y_boundary[1]):
            for i in range(3):
                _, color_seg_img = self.get_seg()
                self.env_img_update()
                self.vis_dir = True
                self.scatter(target_cls)
                self.vis_dir = False

        seg_img, color_seg_img = self.get_seg()
        self.env_img_update()

        # if the target class not exist, pass
        if target_cls not in np.unique(seg_img):
            print("Failed to find %s" % OBJ_LIST[target_cls], file=sys.stderr)
            self.obj_pos = None
            return

        else:
            self.obj_pos, _ = self.get_obj_pos(target_cls)

            # y축 보정 # TODO :
            self.obj_pos[1] -= 0.014

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

                obj_ang, _ = self.seg_model.get_boxed_angle(target_cls)
                self.target_angle = np.rad2deg(self.getj()[-1]) - obj_ang
            else:
                self.obj_pos = None

    def grasp(self, target_cls):
        # Target angle orienting
        if target_cls in [0, 7]:
            obj_angle = np.array([0])
            self.gripper_init_pose = 35
        else:
            obj_angle, w = self.seg_model.get_boxed_angle(target_cls)
            obj_angle = np.deg2rad(obj_angle)

            self.gripper_init_pose = 226 - int(round(17 * w))
            if self.gripper_init_pose < 13:
                self.gripper_init_pose = 13

        self.gripperMoveToInit()  # gripper init_pose
        time.sleep(0.3)

        goal = np.append(self.rob.getj()[:-1], self.rob.getj()[-1] - obj_angle)

        self.rob.movej(goal, self.acc, self.vel)

        # 내리
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

    def hide_target_adj(self, seg_img, target_cls, n_obj):
        hide_img = np.array(seg_img)
        target_pt = np.argwhere(hide_img == target_cls)
        n_obj_pt = np.argwhere(hide_img == n_obj)

        total_pt = np.concatenate((target_pt, n_obj_pt), axis=0)

        for pt_y, pt_x in total_pt:
            hide_img[pt_y][pt_x] = 10

        return hide_img

    def scatter(self, target_cls):
        seg_img, color_img = self.get_seg()
        seg_img = np.array(seg_img)

        end_pt = [0, 0]
        start_pt = [0, 0]
        slope = 0

        sorted_neighboring_obj = self.seg_model.find_neighboring_obj(seg_img, target_cls)

        if len(sorted_neighboring_obj) == 0:
            return

        mean_target = 0
        mean_neighbor = 0

        for n_obj in [sorted_neighboring_obj[0]]:
            # Linear classification
            target_pt = np.argwhere(seg_img == target_cls).astype("float64")
            compare_pt = np.argwhere(seg_img == n_obj).astype("float64")

            mean_target = np.mean(target_pt, axis=0)
            mean_neighbor = np.mean(compare_pt, axis =0)

            # Target, Neighbor total Mean
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
            w_1 = np.copy(w[1])   # 기울기

            othogonal = 0

            if -10e-3 < w_1 < 10e-3:                                          # 수직인 경우의.
                xxx = [center_x] * 128
                yyy = list(range(0, 128))
                othogonal = 1
            else:                                                             # 수직 아닐때. 계산은 됨.
                slope = -w[0] / w_1                                               # Decision Boundary 의 기울기.

                xx = np.arange(0, 256)
                yy = slope * xx - (clf.intercept_[0]) / w[1]

                true_range = np.sort(np.where(np.logical_and(yy < 128, yy > 0))[0])

                xxx = np.arange(true_range[0], true_range[-1])
                yyy = slope * xxx - (clf.intercept_[0]) / w[1]

            # 직선상 중심점 찾기.
            idx = np.argmin(np.linalg.norm(np.array([xxx, yyy]).transpose() - np.array([center_x, center_y]), axis=1))
            new_center_x = np.round(xxx[idx]).astype(np.int)
            new_center_y = np.round(yyy[idx]).astype(np.int)

            ## 수직일 때.
            if othogonal == 1:
                y_under_range = np.arange(new_center_y, 128)
                y_upper_range = np.arange(0, new_center_y)

                list_size = y_under_range.size if y_under_range.size < y_upper_range.size else y_upper_range.size

                hide_img = self.hide_target_adj(np.copy(seg_img), target_cls, n_obj)

                overlap_under = []
                overlap_upper = []

                y_upper_range = y_upper_range[::-1]

                for y_i in range(list_size):
                    if hide_img[y_under_range[y_i]][new_center_x] != 10:
                        overlap_under.append(hide_img[y_under_range[y_i]][new_center_x])

                    if hide_img[y_upper_range[y_i]][new_center_x] != 10:
                        overlap_upper.append(hide_img[y_upper_range[y_i]][new_center_x])

                    try:
                        if y_i == 25:
                            break
                    except UnboundLocalError:
                        pass

                overlap_under = np.unique(overlap_under)
                overlap_upper = np.unique(overlap_upper)

                overlap_upper_area = self.overlapped_area(seg_img, overlap_upper)
                overlap_under_area = self.overlapped_area(seg_img, overlap_under)

                m = 2

                if (overlap_under.size == 0) and (overlap_upper.size == 0):
                    if new_center_y / 2 > 46:
                        start_pt = [y_upper_range[-18], new_center_x]
                        end_pt = [y_under_range[10], new_center_x]
                    else:
                        start_pt = [y_under_range[-18], new_center_x]
                        end_pt = [y_upper_range[10], new_center_x]

                if m == -1:
                    end_pt[0] = np.round((3 * new_center_y - 1 * start_pt[0]) / 2).astype(np.int)
                    end_pt[1] = np.round((3 * new_center_x - 1 * start_pt[1]) / 2).astype(np.int)
                    break

            else:
                if slope < 0:  #  ax+b 에서 a가 0보다 클 때
                    upper_x_pt = xxx[-1]
                    upper_y_pt = np.round(yy[xxx[-1]]).astype(np.int)

                    under_x_pt = xxx[0]
                    under_y_pt = np.round(yy[xxx[0]]).astype(np.int)

                else:
                    upper_x_pt = xxx[0]
                    upper_y_pt = np.round(yy[xxx[0]]).astype(np.int)

                    under_x_pt = xxx[-1]
                    under_y_pt = np.round(yy[xxx[-1]]).astype(np.int)

                # list component : [y, x]
                upper_list = self.make_path([upper_x_pt, upper_y_pt], [new_center_x, new_center_y], 'upper')
                under_list = self.make_path([new_center_x, new_center_y], [under_x_pt, under_y_pt], 'under')

                list_size = under_list.shape[0] if under_list.shape[0] < upper_list.shape[0] else upper_list.shape[0]

                overlap_under = []
                overlap_upper = []
                hide_img = self.hide_target_adj(np.copy(seg_img), target_cls, n_obj)

                for i in range(list_size):
                    if hide_img[under_list[i][0]][under_list[i][1]] != 10:
                        overlap_under.append(hide_img[under_list[i][0]][under_list[i][1]])

                    if hide_img[upper_list[i][0]][upper_list[i][1]] != 10:
                        overlap_upper.append(hide_img[upper_list[i][0]][upper_list[i][1]])

                    if i == 30:
                        break

                overlap_under = np.unique(overlap_under)
                overlap_upper = np.unique(overlap_upper)

                overlap_upper_area = self.overlapped_area(seg_img, overlap_upper)
                overlap_under_area = self.overlapped_area(seg_img, overlap_under)

                m = 2

                if (overlap_under.size == 0) and (overlap_upper.size ==0):
                    for k, [y_1, x_1] in enumerate(under_list):
                        m = 0
                        for i in range(-4, 5):
                            for j in range(-4, 5):
                                try:
                                    abc = seg_img[i + y_1][j + x_1]
                                    if abc != 10:
                                        m += 1
                                except IndexError:
                                    pass

                        if m == 0:
                            if new_center_y / 2 > 46:
                                start_pt = under_list[k]  # [y_1, x_1]
                                end_pt = upper_list[k]
                            else:
                                start_pt = upper_list[k]
                                end_pt = under_list[k]
                            break

                elif (overlap_upper_area < overlap_under_area ) or ((overlap_under.size != 0) and (overlap_upper.size == 0)):
                    print("Exist object where under Starting Point.")
                    for y_1, x_1 in upper_list:
                        m = 0
                        for i in range(-4, 5):
                            for j in range(-4, 5):
                                try:
                                    abc = seg_img[i + y_1][j + x_1]
                                    if abc != 10:
                                        m += 1
                                except IndexError:
                                    pass

                        if m == 0:
                            start_pt = [y_1, x_1]
                            m = -1
                            break

                elif (overlap_upper_area > overlap_under_area ) or ((overlap_under.size == 0) and (overlap_upper.size != 0)):
                    print("Exist object where upper Starting Point.")
                    for y_1, x_1 in under_list:
                        m = 0
                        for i in range(-4, 5):
                            for j in range(-4, 5):
                                try:
                                    abc = seg_img[i + y_1][j + x_1]
                                    if abc != 10:
                                        m += 1
                                except IndexError:
                                    pass

                        if m == 0:
                            start_pt = [y_1, x_1]
                            m = -1
                            break

                if m == -1:
                    end_pt[0] = np.round((3 * new_center_y - 1 * start_pt[0]) / 2).astype(np.int)
                    end_pt[1] = np.round((3 * new_center_x - 1 * start_pt[1]) / 2).astype(np.int)
                    break


        # Clip Working Space.
        # start_pt = clip(start_pt, [40, 106], [12, 243])
        # end_pt = clip(end_pt, [40, 106], [12, 243])

        # if not (40 < start_pt[0] < 106) or not (12 < start_pt[1] < 243):
        #     return
        #
        # if seg_img[start_pt[0], start_pt[1]] != 10:
        #     return

        # 2/3 push
        # mid_pt = [0, 0]
        # mid_pt[0] = int(round((start_pt[0] + 2 * end_pt[0]) / 3))
        # mid_pt[1] = int(round((start_pt[1] + 2 * end_pt[1]) / 3))
        # end_pt = mid_pt

        # shift_x = 1
        # shift_y = 1
        #
        # if mean_neighbor[0] < mean_target[0]:
        #     shift_y = -1
        #
        # if mean_neighbor[1] < mean_target[1]:
        #     shift_x = -1
        #
        # # Shift Points
        # end_pt[0] = end_pt[0] + shift_y * 2
        # end_pt[1] = end_pt[1] + shift_x * 2
        #
        # start_pt[0] = start_pt[0] + shift_y * 2
        # start_pt[1] = start_pt[1] + shift_x * 2

        cv2.line(color_img, (end_pt[1], end_pt[0]), (start_pt[1], start_pt[0]), (0, 255, 0), 1)
        cv2.circle(color_img, (end_pt[1], end_pt[0]), 2, (255, 255, 255), -1)
        cv2.circle(color_img, (start_pt[1], start_pt[0]), 2, (0, 0, 255), -1)

        # Current
        self.end_pt = np.copy(end_pt)
        self.start_pt = np.copy(start_pt)

        self.env_img_update()

        # starting to end point
        start_xyz = self.global_cam.color2xyz([start_pt])
        goal_xyz = self.global_cam.color2xyz([end_pt])  # patched image's averaging pose [x, y, z]

        # y축 보정 필요 한가?
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

    @staticmethod
    def overlapped_area(seg_img, overlap_obj):
        area = 0

        for cls in overlap_obj:
            area += np.argwhere(np.array(seg_img) == cls).shape[0]

        return area

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

    def make_path(self, start, goal, dir):
        start = np.array(start).astype(np.int)
        goal = np.array(goal).astype(np.int)
        zero = np.zeros((128, 256)).astype(np.uint8)
        cv2.line(zero, (start[0], start[1]), (goal[0], goal[1]), (255, 255, 255), 1)

        path = np.argwhere(zero == 255)

        if dir is 'upper':
            path = path[::-1]

        cv2.imshow(dir, zero)
        cv2.waitKey(1)

        return path

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

    def gripperMoveToInit(self, spd=255, force=80):
        self.set_gripper(spd, force)
        self.gripper.move(self.gripper_init_pose)

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

        #msg = input("Use scatter? (T/F)")
        msg = 'T'
        if msg == 'T':
            self.use_scatter = True
        else:
            self.use_scatter = False

    def get_seg(self):
        img = self.global_cam.snap()  # Segmentation input image  w : 256, h : 128
        padded_img = cv2.cvtColor(np.vstack((bkg_padding_img, img)), cv2.COLOR_RGB2BGR)  # Color fmt    RGB -> BGR

        self.input_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        return self.seg_model.run(padded_img)

    def get_obj_pos(self, target_cls):  # TODO : class 0~8 repeat version
        _, _ = self.get_seg()
        pxl_list, mean_xy = self.seg_model.getData(target_cls)  # Get pixel list

        xyz = None
        if pxl_list is not None:
            xyz = self.global_cam.color2xyz(pxl_list)   # patched image's averaging pose [x, y, z]

        return [xyz, mean_xy]

    def set_tcp(self, tcp):
        self.rob.set_tcp(tcp)

    def shutdown(self):
        self._program_send("Shutdown")
