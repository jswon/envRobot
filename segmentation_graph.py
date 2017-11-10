import tensorflow as tf
import numpy as np
import datetime, os, yaml
import cv2
from PIL import Image as Im

color_map = [
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0,128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0,],
    [192, 0, 0],
    [64, 128, 0],
    [0, 0, 0],
]


class SegmentationGraph():
    """  Importing and running isolated TF graph """
    def __init__(self, loc):
        # Create local graph and use it in the session

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph

            ckpt_info_file = loc + "checkpoint"

            info = yaml.load(open(ckpt_info_file, "r"))
            assert 'model_checkpoint_path' in info
            #      most_recent_ckpt = "%s\\%s" % (self.ckpt_dir, info['model_checkpoint_path'])
            most_recent_ckpt = "%s" % (info['model_checkpoint_path']) + '.meta'
            saver = tf.train.import_meta_graph(most_recent_ckpt)
            most_recent_ckpt = "%s" % (info['model_checkpoint_path'])
            saver.restore(self.sess, most_recent_ckpt)
            # Get activation function from saved collection
            # You may need to change this in case you name it differently
            #self.activation = tf.get_collection('activation')[0]
            self.segmentation = tf.get_collection('argmax_output')[0]

    def run(self, image):
        # The 'x' corresponds to name of input placeholder
        result = self.sess.run(self.segmentation, feed_dict={'input:0': [image]})[0]
        new_segmented_array, new_segmented_color_array = self.new_segmented_image_by_connected_component(result)

        self.new_segmented_array = new_segmented_array
        # Im.fromarray(new_segmented_color_array).show()  # show
        return new_segmented_array

    OBJ_LIST = ['O_00_Black_Tape', 'O_01_Glue', 'O_02_Big_USB', 'O_03_Glue_Stick', 'O_04_Big_Box',
                'O_05_Red_Cup', 'O_06_Small_Box', 'O_07_White_Tape', 'O_08_Small_USB', 'O_09_Yellow_Cup']

    def get_angle(self, new_segmented_array, cls_idx):
        symm = 0

        pointsList = np.argwhere(new_segmented_array == cls_idx)
        pointsList = pointsList - np.mean(pointsList, 0)

        cov = np.cov(pointsList.transpose())
        evals, evecs = np.linalg.eig(cov)

        OBJ_LIST = ['O_00_Black_Tape', 'O_01_Glue', 'O_02_Big_USB', 'O_03_Glue_Stick', 'O_04_Big_Box',
                    'O_05_Red_Cup', 'O_06_Small_Box', 'O_07_White_Tape', 'O_08_Small_USB', 'O_09_Yellow_Cup']
        # 비슷하면 1, 아니면 0
        if np.abs(np.diff(evals)) <= 10 and cls_idx in [2, 8]:  # if | eval_0 - eval_1 | < threshold
            symm = 1
        elif np.abs(np.diff(evals)) <= 20 and cls_idx not in [2, 8]:
            symm = 1

        if symm == 1 and cls_idx in [1, 2, 3, 8]:  # Glue and USB data ignore
            return None, None

        sort_indices = np.argsort(evals)[::-1]
        evec = evecs[sort_indices[0]]

        x_v1, y_v1 = evec  # Eigen vector with largest eigenvalue

        theta = np.round(np.arctan(x_v1 / y_v1), 5)  # Radian, Largest Eigen value
        # TODO : deg/180 ? rad/pi?

        if symm == 1 and cls_idx in [0, 7]:  # sym. object : [Black and White Tape],
            return np.array([0, 0])
        else:
            return np.array([theta, symm])

    def getPoints(self, label_image_array, object_index):
        return np.argwhere(label_image_array == object_index)

    def convert_grey_label_to_color_label(self, grey_label_array):
        height = grey_label_array.shape[0]
        width = grey_label_array.shape[1]
        channel = 3

        shape = (height, width, channel)
        color_label = np.zeros(shape=shape, dtype=np.uint8)

        for i in range(0, height):
            for j in range(0, width):
                value = grey_label_array[i, j]
                color = color_map[value]

                color_label[i, j, 0] = color[0]
                color_label[i, j, 1] = color[1]
                color_label[i, j, 2] = color[2]

        return np.copy(color_label)

    def make_binary_label_array(self, target_object_pixels_list, binary_array):
        num_points = len(target_object_pixels_list)

        for i in range(0, num_points):
            pixel = target_object_pixels_list[i]
            y = pixel[0]
            x = pixel[1]
            binary_array[y, x] = 255

        return binary_array

    def make_new_label_array(self, dest_array, target_object_pixels_list, object_index):
        num_points = len(target_object_pixels_list)

        for i in range(0, num_points):
            pixel = target_object_pixels_list[i]
            y = pixel[0]
            x = pixel[1]
            dest_array[y, x] = object_index

        return dest_array

    def new_segmented_image_by_connected_component(self, org_label_array):

        shape = (128, 256)
        binary_image_array = np.zeros(shape=shape, dtype=np.uint8)
        new_label_image_array = np.zeros(shape=shape, dtype=np.uint8)
        new_label_image_array.fill(10)

        for j in range(0, 10):
            pointList = self.getPoints(org_label_array, j)
            binary_image_array.fill(0)
            binary_image_array = self.make_binary_label_array(pointList, binary_image_array)

            connectivity = 8
            output = cv2.connectedComponentsWithStats(binary_image_array, connectivity, cv2.CV_32S)

            # The first cell is the number of labels
            num_labels = output[0]
            # The second cell is the label matrix
            labels = output[1]
            stats = output[2]

            if num_labels > 1:
                second_max_index = 1
                second_max = 0
                for m in range(0, num_labels):
                    pixels_num = stats[m, cv2.CC_STAT_AREA]
                    if pixels_num > second_max and pixels_num < 10000:
                        second_max = pixels_num
                        second_max_index = m

                pointList2 = self.getPoints(labels, second_max_index)
                new_label_image_array = self.make_new_label_array(new_label_image_array, pointList2, j)

        new_label_color_image_array = self.convert_grey_label_to_color_label(new_label_image_array)

        return new_label_image_array, new_label_color_image_array

