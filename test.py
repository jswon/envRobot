import envRobot
import argparse
import util
import Kinect_Snap
import cv2
import urx
import numpy as np

rob = urx.Robot("192.168.0.31")
global_cam = Kinect_Snap.global_cam()  # Load Camera

img = global_cam.snap()   # snapshot

# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#
# parser.add_argument('--num-eval', type=int, default=0,
#                     help="if >0 just run this many episodes with no training")
# parser.add_argument('--max-num-actions', type=int, default=0,
#                     help="train for (at least) this number of actions (always finish current episode)"
#                          " ignore if <=0")
# parser.add_argument('--max-run-time', type=int, default=0,
#                     help="train for (at least) this number of seconds (always finish current episode)"
#                          " ignore if <=0")
# parser.add_argument('--root-dir', type=str, default='C:\\tensorflow_code\\UR3_simple_task\\Supervised_learning\\SP_pretrain_data_collect\\', help="")
# parser.add_argument('--ckpt-dir', type=str, default='C:\\tensorflow_code\\UR3_simple_task\\Supervised_learning\\SP_pretrain_data_collect\\save', help="if set save ckpts to this dir")
# #parser.add_argument('--ckpt-dir', type=str, default=None, help="if set save ckpts to this dir")
# parser.add_argument('--ckpt-freq', type=int, default=1800, help="freq (sec) to save ckpts")
# parser.add_argument('--batch-size', type=int, default=32, help="training batch size")
# parser.add_argument('--batches-per-step', type=int, default=5,
#                     help="number of batches to train per step")
# parser.add_argument('--dont-do-rollouts', action="store_true",
#                     help="by dft we do rollouts to generate data then train after each rollout. if this flag is set we"
#                          " dont do any rollouts. this only makes sense to do if --event-log-in set.")
# parser.add_argument('--target-update-rate', type=float, default=0.0001,
#                     help="affine combo for updating target networks each time we run a training batch")
# parser.add_argument('--use-batch-norm', default=True, action="store_true",
#                     help="whether to use batch norm on conv layers")
# parser.add_argument('--actor-hidden-layers', type=str, default="200,200,50", help="actor hidden layer sizes")
# parser.add_argument('--critic-hidden-layers', type=str, default="200,200,50", help="critic hidden layer sizes")
# parser.add_argument('--actor-learning-rate', type=float, default=0.0001, help="learning rate for actor")
# parser.add_argument('--critic-learning-rate', type=float, default=0.001, help="learning rate for critic")
# parser.add_argument('--discount', type=float, default=0.99, help="discount for RHS of critic bellman equation update")
# parser.add_argument('--event-log-in', type=str, default=None,
#                     help="prepopulate replay memory with entries from this event log")
# parser.add_argument('--replay-memory-size', type=int, default=10000, help="max size of replay memory")
# parser.add_argument('--replay-memory-burn-in', type=int, default=50, help="dont train from replay memory until it reaches this size")
# parser.add_argument('--eval-action-noise', action='store_true', help="whether to use noise during eval")
# parser.add_argument('--action-noise-theta', type=float, default=0.1,
#                     help="OrnsteinUhlenbeckNoise theta (rate of change) param for action exploration")
# parser.add_argument('--action-noise-sigma', type=float, default=0.2,
#                     help="OrnsteinUhlenbeckNoise sigma (magnitude) param for action exploration")
# parser.add_argument('--joint-angle-low-limit', type=float, default=-110,
#                     help="joint angle low limit for action")
# parser.add_argument('--joint-angle-high-limit', type=float, default=110,
#                     help="joint angle high limit for action")
# parser.add_argument('--action_dim', type=float, default=1,
#                     help="number of joint angle for robot action")
# parser.add_argument('--internal_state_dim', type=float, default=18,
#                     help="internal_state_dim")
# parser.add_argument('--action_repeat_per_scene', type=float, default=10,
#                     help="number of actions per a scene")
# parser.add_argument('--number_of_scenes_per_shuffle', type=float, default=10,
#                     help="number of scenes per a shuffle")
# parser.add_argument('--use-full-internal-state', default=False, action="store_true",
#                     help="whether to use full internal state")
# parser.add_argument('--action-repeats', type=int, default=1,
#                     help="number of action repeats")
# parser.add_argument('--num-cameras', type=int, default=2,
#                     help="how many camera points to render; 1 or 2")
#
# util.add_opts(parser)
#
# envRobot.add_opts(parser)
# opts = parser.parse_args()
#
# env = envRobot.envRobot("192.168.0.31", opts)
# for _ in range(100) :
#     env.rob.movej(envRobot.HOME, 2.5, 2.5)
#     env.shuffle_obj()
# k = 1

render_height = 256
render_width = 256
num_cameras = 2
repeats = 1
state_shape = (render_height, render_width, 3, num_cameras, repeats)
state = np.empty(state_shape, dtype=np.float32)
test_np = state[:,:,:,1,0]

test_img = test_np[:, :, :]

hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
lower_blue = np.array([114, 75, 85])
upper_blue = np.array([180,255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

result = cv2.bitwise_and(img,img,mask = mask)


cv2.imshow("test",result)
ret,thresh = cv2.threshold(result,127,255,cv2.THRESH_BINARY)
blurred = cv2.medianBlur(thresh,5)
blurred = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
th3 = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

cv2.imshow("thresh",img)

_, contours, hierarchy = cv2.findContours(th3, 1, 2)
max_radius = 0
for cnt in contours:
    if cv2.contourArea(cnt) < 60000 :
        bx, by, bw, bh = cv2.boundingRect(cnt)
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        print(cx, cy)

        if radius > max_radius :
            max_radius = radius

cv2.circle(img, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)  # draw circle in red color
cv2.imshow('input', img)
#
# rob_y = 2.9579207920792*cy - 405.33415841584
# rob_x =401.5000081602-3.1829680283322*cx
k = 1
#
#
#
# # import sys
# # import tensorflow as tf
# # import numpy as np
# # import random
# # import argparse
# # import replay_memoryd
# # import util
# # import envRobot
# #
# # np.set_printoptions(precision=5, threshold=10000, suppress=True, linewidth=10000)
# #
# # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # parser.add_argument('--num-eval', type=int, default=0,
# #                     help="if >0 just run this many episodes with no training")
# # parser.add_argument('--max-num-actions', type=int, default=0,
# #                     help="train for (at least) this number of actions (always finish current episode)"
# #                          " ignore if <=0")
# # parser.add_argument('--max-run-time', type=int, default=0,
# #                     help="train for (at least) this number of seconds (always finish current episode)"
# #                          " ignore if <=0")
# # parser.add_argument('--ckpt-dir', type=str, default='ckpt\\ddpg2', help="if set save ckpts to this dir")
# # parser.add_argument('--ckpt-freq', type=int, default=3600, help="freq (sec) to save ckpts")
# # parser.add_argument('--batch-size', type=int, default=32, help="training batch size")
# # parser.add_argument('--batches-per-step', type=int, default=5,
# #                     help="number of batches to train per step")
# # parser.add_argument('--dont-do-rollouts', action="store_true",
# #                     help="by dft we do rollouts to generate data then train after each rollout. if this flag is set we"
# #                          " dont do any rollouts. this only makes sense to do if --event-log-in set.")
# # parser.add_argument('--target-update-rate', type=float, default=0.0001,
# #                     help="affine combo for updating target networks each time we run a training batch")
# # parser.add_argument('--use-batch-norm', default=False, action="store_true",
# #                     help="whether to use batch norm on conv layers")
# # parser.add_argument('--actor-hidden-layers', type=str, default="200,200,50", help="actor hidden layer sizes")
# # parser.add_argument('--critic-hidden-layers', type=str, default="200,200,50", help="critic hidden layer sizes")
# # parser.add_argument('--actor-learning-rate', type=float, default=0.0001, help="learning rate for actor")
# # parser.add_argument('--critic-learning-rate', type=float, default=0.001, help="learning rate for critic")
# # parser.add_argument('--discount', type=float, default=0.99, help="discount for RHS of critic bellman equation update")
# # parser.add_argument('--event-log-in', type=str, default=None,
# #                     help="prepopulate replay memory with entries from this event log")
# # parser.add_argument('--replay-memory-size', type=int, default=10000, help="max size of replay memory")
# # parser.add_argument('--replay-memory-burn-in', type=int, default=100, help="dont train from replay memory until it reaches this size")
# # parser.add_argument('--eval-action-noise', action='store_true', help="whether to use noise during eval")
# # parser.add_argument('--action-noise-theta', type=float, default=0.1,
# #                     help="OrnsteinUhlenbeckNoise theta (rate of change) param for action exploration")
# # parser.add_argument('--action-noise-sigma', type=float, default=0.01,
# #                     help="OrnsteinUhlenbeckNoise sigma (magnitude) param for action exploration")
# # parser.add_argument('--joint-angle-low-limit', type=float, default=-180,
# #                     help="joint angle low limit for action")
# # parser.add_argument('--joint-angle-high-limit', type=float, default=180,
# #                     help="joint angle high limit for action")
# # parser.add_argument('--action_dim', type=float, default=4,
# #                     help="number of joint angle for robot action")
# # parser.add_argument('--internal_state_dim', type=float, default=18,
# #                     help="internal_state_dim")
# # parser.add_argument('--action_repeat_per_scene', type=float, default=20,
# #                     help="number of actions per a scene")
# # parser.add_argument('--number_of_scenes_per_shuffle', type=float, default=10,
# #                     help="number of scenes per a shuffle")
# # parser.add_argument('--use-full-internal-state', default=False, action="store_true",
# #                     help="whether to use full internal state")
# #
# # opts = parser.parse_args()
# #
# # util.add_opts(parser)
# #
# # env = envRobot.envRobot()
# #
# # class TestAgent(object):
# #     def __init__(self):
# #         self.env = env
# #         # state_shape = self.env.state_shape
# #         # action_dim = self.env.action_space.shape[1]
# #         #
# #         # self.replay_memory = replay_memory.ReplayMemory(opts.replay_memory_size, state_shape, action_dim, opts)
# #
# #     def run_training(self, max_num_actions, max_run_time, batch_size, hatches_per_step, saver_util):
# #         while True:
# #             rewards =[]
# #             losses = []
# #
# #             env.shuffle_obj()
# #             action = self.actor
# #
# #
# #
# #     def run_eval(num_eval, eval_action_noise):
# #         pass
# #
# #
# # def main():
# #     config = tf.ConfigProto()
# #
# #     with tf.Session(config=config) as sess:
# #         agent = TestAgent(env=env)
# #
# #         saver_util = None
# #         if opts.ckpt_dir is not None :
# #             saver_util = util.SaverUtil(sess, opts.ckpt_dir, opts.ckpt_freq)
# #         else:
# #             sess.run(tf.global_variables_initializer())
# #
# #         for v in tf.global_variables():
# #             print(v.name, util.shape_and_product_of(v), file=sys.stderr)
# #
# #         agent.post_var_init_setup()
# #
# #         if opts.num_eval>0:
# #             agent.run_eval(opts.num_eval, opts.eval_action_noise)
# #         else:
# #             agent.run_training(opts.max_num_actions, opts.max_run_time,
# #                                opts.batch_size, opts.batches_per_step,
# #                                saver_util)
# #             if saver_util is not None:
# #                 saver_util.force_save()
# #
# #         env.reset()
# #
# # if __name__ == "__main__":
# #     main()