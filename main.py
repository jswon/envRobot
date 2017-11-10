#!/usr/bin/env python
import argparse
import collections
import numpy as np
import replay_memory
import sys
import tensorflow as tf
import time
import util
import math
import os, yaml
import envRobot
import copy
import actor_network
import critic_network
from ou_noise import OUNoise
import segmentation_graph

np.set_printoptions(precision=5, threshold=10000, suppress=True, linewidth=10000)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-eval', type=int, default=0, help="if >0 just run this many episodes with no training")
parser.add_argument('--max-num-actions', type=int, default=0, help="train for (at least) this number of actions (always finish current episode)" " ignore if <=0")
parser.add_argument('--max-run-time', type=int, default=0, help="train for (at least) this number of seconds (always finish current episode)" " ignore if <=0")
parser.add_argument('--root-dir', type=str, default='', help="")
parser.add_argument('--ckpt-dir', type=str, default='save\\', help="if set save ckpts to this dir")
parser.add_argument('--ckpt-freq', type=int, default=1800, help="freq (sec) to save ckpts")
parser.add_argument('--batch-size', type=int, default=32, help="training batch size")
parser.add_argument('--batches-per-step', type=int, default=10, help="number of batches to train per step")
parser.add_argument('--dont-do-rollouts', action="store_true", help="by dft we do rollouts to generate data then train after each rollout. if this flag is set we"
                                                                    " dont do any rollouts. this only makes sense to do if --event-log-in set.")
parser.add_argument('--target-update-rate', type=float, default=0.001, help="affine combo for updating target networks each time we run a training batch")
parser.add_argument('--use-batch-norm', default=False, action="store_true", help="whether to use batch norm on conv layers")
parser.add_argument('--actor-hidden-layers', type=str, default="200,200,50", help="actor hidden layer sizes")
parser.add_argument('--critic-hidden-layers', type=str, default="200,200,50", help="critic hidden layer sizes")
parser.add_argument('--actor-learning-rate', type=float, default=0.00002, help="learning rate for actor")
parser.add_argument('--critic-learning-rate', type=float, default=0.001, help="learning rate for critic")
parser.add_argument('--discount', type=float, default=0.99, help="discount for RHS of critic bellman equation update")
parser.add_argument('--event-log-in', type=str, default=None, help="pre-populate replay memory with entries from this event log")
parser.add_argument('--replay-memory-size', type=int, default=20000, help="max size of replay memory")
parser.add_argument('--replay-memory-burn-in', type=int, default=100, help="don't train from replay memory until it reaches this size")
parser.add_argument('--eval-action-noise', action='store_true', help="whether to use noise during eval")
parser.add_argument('--action-noise-theta', type=float, default=0.2, help="OrnsteinUhlenbeckNoise theta (rate of change) param for action exploration")
parser.add_argument('--action-noise-sigma', type=float, default=0.15, help="OrnsteinUhlenbeckNoise sigma (magnitude) param for action exploration")
parser.add_argument('--joint-angle-low-limit', type=float, default=-90, help="joint angle low limit for action")
parser.add_argument('--joint-angle-high-limit', type=float, default=90, help="joint angle high limit for action")
parser.add_argument('--action_dim', type=float, default=1, help="number of joint angle for robot action")
parser.add_argument('--internal_state_dim', type=float, default=18, help="internal_state_dim")
parser.add_argument('--action_repeat_per_scene', type=float, default=1, help="number of actions per a scene")
parser.add_argument('--number_of_scenes_per_shuffle', type=float, default=10, help="number of scenes per a shuffle")
parser.add_argument('--use-full-internal-state', default=False, action="store_true", help="whether to use full internal state")
parser.add_argument('--with-data-collecting', default=True, help="with data collecting for pre-training")

util.add_opts(parser)
envRobot.add_opts(parser)

opts = parser.parse_args()
sys.stderr.write("%s\n" % opts)

robot = envRobot.Env('192.168.0.31', opts)


class DeepDeterministicPolicyGradientAgent(object):
    def __init__(self, sess, pretrained_model):
        self.trainable_model_vars()
        self.pretrained_model = pretrained_model

        self.exploration_noise = OUNoise(robot.action_dim, 0, opts.action_noise_theta, opts.action_noise_sigma)
        # for now, with single machine synchronous training, use a replay memory for training.
        # this replay memory stores states in a Variable (ie potentially in gpu memory)
        # TODO: switch back to async training with multiple replicas (as in drivebot project)
        self.replay_memory = replay_memory.ReplayMemory(opts.replay_memory_size, robot.state_shape, robot.action_dim, opts)

        # s1 and s2 placeholders
        batched_state_shape = [None] + list(robot.state_shape)
        s_a = tf.placeholder(shape=batched_state_shape, dtype=tf.float32)
        s_ta = tf.placeholder(shape=batched_state_shape, dtype=tf.float32)
        s_c = tf.placeholder(shape=batched_state_shape, dtype=tf.float32)
        s_tc = tf.placeholder(shape=batched_state_shape, dtype=tf.float32)

        action_shape = [None, opts.action_dim]
        input_action = tf.placeholder(shape=action_shape, dtype=tf.float32)
        target_input_action = tf.placeholder(shape=action_shape, dtype=tf.float32)

        if opts.use_full_internal_state:
            temp = [1, 18]
        else:
            temp = [5]

        batched_internal_state_shape = [None] + temp
        internal_state_a = tf.placeholder(shape=batched_internal_state_shape, dtype=tf.float32)
        internal_state_ta = tf.placeholder(shape=batched_internal_state_shape, dtype=tf.float32)
        internal_state_c = tf.placeholder(shape=batched_internal_state_shape, dtype=tf.float32)
        internal_state_tc = tf.placeholder(shape=batched_internal_state_shape, dtype=tf.float32)
        temp = [10]
        batched_taget_obj_shape = [None] + temp
        taget_obj_a = tf.placeholder(shape=batched_taget_obj_shape, dtype=tf.float32)
        taget_obj_ta = tf.placeholder(shape=batched_taget_obj_shape, dtype=tf.float32)
        taget_obj_c = tf.placeholder(shape=batched_taget_obj_shape, dtype=tf.float32)
        taget_obj_tc = tf.placeholder(shape=batched_taget_obj_shape, dtype=tf.float32)

        is_training_for_actor = tf.placeholder(tf.bool)
        is_target_training_for_actor = tf.placeholder(tf.bool)

        is_training_for_critic = tf.placeholder(tf.bool)
        is_target_training_for_critic = tf.placeholder(tf.bool)

        # initialise base models for actor / critic and their corresponding target networks
        # target_actor is never used for online sampling so doesn't need explore noise.
        self.actor = actor_network.ActorNetwork("actor", sess, s_a, s_ta, internal_state_a, internal_state_ta, taget_obj_a, taget_obj_ta,
                                                robot.action_dim, opts, self.pretrained_model, is_training_for_actor, is_target_training_for_actor)
        self.critic = critic_network.CriticNetwork("critic", sess, s_c, s_ta, internal_state_c, internal_state_tc, taget_obj_c, taget_obj_tc, input_action, target_input_action,
                                                   robot.action_dim, opts, self.pretrained_model, is_training_for_critic, is_target_training_for_critic)

    @staticmethod
    def trainable_model_vars():
        v = []
        for var in tf.global_variables():
            v.append(var)
            print(var.name)
        return v

    def load_dataset(self, dir, num_training_data, num_test_data):
        initial_state_temp_index = []
        initial_state_index = []
        state_index = []

        class_idx = np.arange(10)

        full_path = dir + '\\'

        initial_state_dir = full_path + 'initial_image\\'
        state_dir = full_path + 'current_image\\'
        #      position_dir = full_path + 'obj_position\\'
        position_dir = full_path + 'obj_position\\'
        target_obj_dir = full_path + 'target_obj\\'

        num_data = 0
        for root, dirs, files in os.walk(initial_state_dir):
            prev_index = 0
            for file in files:
                num_data = num_data + 1
                index = file.replace(".bmp", "")
                index = int(index)

                #              if index == 47104:
                #                  continue

                initial_state_temp_index.append(index)

        initial_state_temp_index.sort()

        prev_file_idx = 0
        index = 0
        for file_idx in initial_state_temp_index:

            #          if file_idx == 47093:
            #              a = 0

            if prev_file_idx > 0:
                diff = file_idx - prev_file_idx

                if diff > 11:
                    prev_file_idx = file_idx
                    continue

                idx = 0
                for idx2 in range(prev_file_idx, file_idx):
                    initial_state_index.append(prev_file_idx)
                    state_index.append(prev_file_idx + idx)
                    idx = idx + 1

            prev_file_idx = file_idx

            #    num_data = num_data * 10
        num_data = len(state_index)

        idxs = np.random.randint(0, num_training_data + num_test_data, num_training_data + num_test_data)
        training_idx = idxs[:num_training_data]
        test_idx = idxs[num_training_data:]  # num_training_data+num_test_data

        n = 0
        num_data = 0
        image_position_sequence = []
        for index in training_idx:
            idx = state_index[index]
            file = state_dir + str(idx) + '.bmp'

            image = Im.open(file)
            state = np.array(image)
            robot.state[:, :, :, 0, 0] = state

            tidx = initial_state_index[index]

            file = initial_state_dir + str(tidx) + '.bmp'

            image = Im.open(file)
            state = np.array(image)
            robot.state[:, :, :, 1, 0] = state

            file = position_dir + str(idx) + '.txt'
            f = open(file, 'r')

            x = []
            count = 0
            for y in f.read().split(' '):
                x.append(float(y))
                count = count + 1
                if count == 3:
                    break

            target = np.asarray(x)
            f.close()

            file = target_obj_dir + str(idx) + '.txt'
            f = open(file, 'r')

            x = []
            for y in f.read().split(' '):
                x.append(float(y))

            target_obj = np.asarray(x)
            f.close()

            image_position_sequence.append((np.copy(robot.state), target, target_obj))
            num_data = num_data + 1
            n = n + 1
            if num_data % 10 == 0:
                self.replay_memory.add_episode(image_position_sequence)
                num_data = 0
                image_position_sequence = []
                print(n)

        n = 0
        num_data = 0
        image_position_sequence = []
        for index in test_idx:

            idx = state_index[index]
            file = state_dir + str(idx) + '.bmp'

            image = Im.open(file)
            state = np.array(image)
            robot.state[:, :, :, 0, 0] = state

            tidx = initial_state_index[index]

            file = initial_state_dir + str(tidx) + '.bmp'

            image = Im.open(file)
            state = np.array(image)
            robot.state[:, :, :, 1, 0] = state

            file = position_dir + str(idx) + '.txt'
            f = open(file, 'r')

            x = []
            count = 0
            for y in f.read().split(' '):
                x.append(float(y))
                count = count + 1
                if count == 3:
                    break

            target = np.asarray(x)
            f.close()

            file = target_obj_dir + str(idx) + '.txt'
            f = open(file, 'r')

            x = []
            for y in f.read().split(' '):
                x.append(float(y))

            target_obj = np.asarray(x)
            f.close()

            image_position_sequence.append((np.copy(robot.state), target, target_obj))
            num_data = num_data + 1
            n = n + 1
            if num_data % 10 == 0:
                self.replay_memory_test.add_episode(image_position_sequence)
                num_data = 0
                image_position_sequence = []
                print(n)

    @staticmethod
    def one_hot_encode(x):
        return np.eye(len(x))[x]

    def run_training(self, sess, batch_size, batches_per_step, saver_util):
        # run for some max number of actions
        n = 0

        obj_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        one_hot_list = self.one_hot_encode(obj_list)
        num_data = 1

        while True:
            rewards = []
            robot.shuffle_obj()

            # # TODO : Target Object Random Pick
            # target_object_index = int(input("Pick the object : "))
            # objects_name = robot.get_obj_name(target_object_index)
            # print('target object:', objects_name)

            # start a new episode, reset & approaching
            state_1 = robot.reset()  # state_1 : current State

            num_data = num_data + 1
            obj_pos = robot.obj_pos

            print('\nx = ', obj_pos[0], ", y = ", obj_pos[1], "z = ", obj_pos[2])

            # prepare data for updating replay memory at end of episode
            initial_state = np.copy(state_1)
            action_reward_state_sequence = []

            done = False
            step = 1
            target_qs = []
            target_obj = one_hot_list[robot.target_cls]

            while not done:
                # choose action
                internal_state = robot.internal_state
                action = self.actor.action(state_1, internal_state, target_obj)[0]  # 1 dim action
                action += self.exploration_noise.noise()
                action = np.clip(action, -1, 1)  # action output is _always_ (-1, 1) clipping
                action = action * math.radians(90)

                # take action step in env
                # returned action is 6 dim.
                state_2, reward, action, done, is_terminal = robot.step(action, self.exploration_noise, True)

                # action = [x / math.radians(90) for x in action]

                num_data = num_data + 1

                rewards.append(reward)
                # cache for adding to replay memory

                # roll state for next step.
                action_reward_state_sequence.append((np.copy(action), np.copy(reward), np.copy(state_1), np.copy(internal_state), target_obj, is_terminal))

                state_1 = state_2
                step = step + 1

                if step == opts.action_repeat_per_scene:
                    done = True

            # at end of episode update replay memory
            if self.replay_memory.size() > self.replay_memory.buffer_size - (opts.action_repeat_per_scene * 2):
                self.replay_memory._reset()

            if len(action_reward_state_sequence) > 0:
                self.replay_memory.add_episode(initial_state, action_reward_state_sequence)

            # do a training step (after waiting for buffer to fill a bit...)
            if self.replay_memory.size() > opts.replay_memory_burn_in:
                # run a set of batches
                batches_per_steps = batches_per_step

                if n % 5 == 0 and n > 0:
                    batches_per_steps = batches_per_step * 4

                for idx in range(batches_per_steps):
                    batch = self.replay_memory.batch(batch_size)

                    state_batch = batch.state_1
                    action_batch = batch.action
                    reward_batch = batch.reward
                    terminal_mask = batch.terminal_mask
                    next_state_batch = batch.state_2
                    internal_state_batch = batch.internal_state
                    next_internal_state_batch = batch.internal_state_2
                    target_obj_batch = batch.target_obj

                    y_batch = []
                    for i in range(opts.batch_size):
                        y_batch.append(reward_batch[i])

                    y_batch = np.resize(y_batch, [opts.batch_size, 1])

                    # Update critic by minimizing the loss L
                    if not n % 5 == 0:
                        self.critic.train(y_batch, state_batch, action_batch, internal_state_batch, target_obj_batch)

                    if n % 5 == 0:
                        # Update the actor policy using the sampled gradient:
                        action_batch_for_gradients = self.actor.actions(state_batch, internal_state_batch, target_obj_batch)
                        q_gradient_batch = self.critic.gradients(state_batch, action_batch_for_gradients, internal_state_batch, target_obj_batch)
                        self.actor.train(q_gradient_batch, state_batch, internal_state_batch, target_obj_batch)

                        sys.stdout.write("\r>> training actor : {}/{}".format(str(idx + 1), str(batches_per_steps)))
                        sys.stdout.flush()

                # dump some stats and progress info
                stats = collections.OrderedDict()
                stats["time"] = time.time()
                stats["n"] = n
                stats["total_reward"] = np.sum(rewards)
                stats["episode_len"] = len(rewards)
                sys.stdout.flush()
                print('\n>> n:', n + 1, " total_reward:", np.sum(rewards), " mean_target_q:", np.mean(target_qs), " episode_len:", len(rewards))
                n += 1

                # save if required
                if saver_util is not None:
                    Is_saved = saver_util.save_if_required()
                    if Is_saved:
                        pass
                        # exit when finished
                        # if n % 10 == 0:
                        #   self.run_eval(1)

    def run_eval(self, num_episodes, add_noise=False):
        """ run num_episodes of eval and output episode length and rewards """
        obj_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        one_hot_list = self.one_hot_encode(obj_list)
        obj_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        target_object_index = 6
        target_obj = one_hot_list[target_object_index, :]

        for i in range(num_episodes):
            robot.shuffle_obj()
            state, previous_reward = robot.reset(target_obj, 0, opts.root_dir)  # state_1 : current State

            target_object_index = 6
            # objects_name = robot.get_obj_name(obj_index[target_object_index])
            # print('target object:', objects_name)

            total_reward = 0
            steps = 0
            done = False

            while not done:
                internal_state = robot.get_internal_state()
                target_obj = one_hot_list[target_object_index, :]
                action = self.actor.action(state, internal_state, target_obj)
                #        actions = np.clip(1, -1, action)  # action output is _always_ (-1, 1)
                action = action * math.radians(90)
                #        action[0, 0] = action[0, 0] - 90
                # take action step in env
                # state, reward, action, done, problem, actural_reward = robot.step(action, target_object_index, pos, target_obj, 0, previous_reward, opts.root_dir, self.exploration_noise, False)

                state, reward, action, done, actural_reward = robot.step(action, target_object_index, target_obj, 0, previous_reward, opts.root_dir, self.exploration_noise, False)

                if actural_reward == 0:
                    break

                previous_reward = actural_reward

                print("EVALSTEP r%s %s %s %s %s" % (i, steps, np.squeeze(action), np.linalg.norm(action), actural_reward))
                total_reward += reward
                steps += 1

                done = False

                if steps == 10:
                    done = True

            print("EVAL", i, steps, total_reward)
        # self.env.dummy_action()

        sys.stdout.flush()


def trainable_model_vars():
    v = []
    for var in tf.global_variables():
        v.append(var)
        print(var.name)
    return v


def load_pretrained_ckpt(sess):
    """loads latests ckpt from dir. if there are non run init variables."""
    # if no latest checkpoint init vars and return
    pretrained_model = []

    path = opts.root_dir + 'save_pretrain'
    ckpt_info_file = "%s\\checkpoint" % path
    if os.path.isfile(ckpt_info_file):
        # load latest ckpt
        info = yaml.load(open(ckpt_info_file, "r"))
        assert 'model_checkpoint_path' in info

        most_recent_ckpt = "%s" % (info['model_checkpoint_path']) + '.meta'
        sys.stderr.write(">> Loading ckpt %s\n" % most_recent_ckpt)

        pretrain_saver = tf.train.import_meta_graph(most_recent_ckpt)
        most_recent_ckpt = "%s" % (info['model_checkpoint_path'])
        pretrain_saver.restore(sess, most_recent_ckpt)
        pretrain_graph = tf.get_default_graph()

        # trainable_model_vars()

        actor_input_beta = sess.run(pretrain_graph.get_tensor_by_name('actor/input/bn/beta:0'))
        actor_input_gamma = sess.run(pretrain_graph.get_tensor_by_name('actor/input/bn/gamma:0'))
        actor_input_moving_mean = sess.run(pretrain_graph.get_tensor_by_name('actor/input/bn/moving_mean:0'))
        actor_input_moving_variance = sess.run(pretrain_graph.get_tensor_by_name('actor/input/bn/moving_variance:0'))
        actor_conv1_weight = sess.run(pretrain_graph.get_tensor_by_name('actor/conv1/weights:0'))
        actor_conv1_bias = sess.run(pretrain_graph.get_tensor_by_name('actor/conv1/biases:0'))
        actor_conv2_weight = sess.run(pretrain_graph.get_tensor_by_name('actor/conv2/weights:0'))
        actor_conv2_bias = sess.run(pretrain_graph.get_tensor_by_name('actor/conv2/biases:0'))
        actor_conv3_weight = sess.run(pretrain_graph.get_tensor_by_name('actor/conv3/weights:0'))
        actor_conv3_bias = sess.run(pretrain_graph.get_tensor_by_name('actor/conv3/biases:0'))
        actor_conv4_weight = sess.run(pretrain_graph.get_tensor_by_name('actor/conv4/weights:0'))
        actor_conv4_bias = sess.run(pretrain_graph.get_tensor_by_name('actor/conv4/biases:0'))
        actor_hidden1_weight = sess.run(pretrain_graph.get_tensor_by_name('actor/hidden1/weights:0'))
        actor_hidden1_bias = sess.run(pretrain_graph.get_tensor_by_name('actor/hidden1/biases:0'))
        temp_actor_hidden2_weight = sess.run(pretrain_graph.get_tensor_by_name('actor/hidden2/weights:0'))
        actor_hidden2_bias = sess.run(pretrain_graph.get_tensor_by_name('actor/hidden2/biases:0'))
        actor_hidden3_weight = sess.run(pretrain_graph.get_tensor_by_name('actor/hidden3/weights:0'))
        actor_hidden3_bias = sess.run(pretrain_graph.get_tensor_by_name('actor/hidden3/biases:0'))

        actor_conv1_beta = sess.run(pretrain_graph.get_tensor_by_name('actor/conv1/bn/beta:0'))
        actor_conv1_gamma = sess.run(pretrain_graph.get_tensor_by_name('actor/conv1/bn/gamma:0'))
        actor_conv1_moving_mean = sess.run(pretrain_graph.get_tensor_by_name('actor/conv1/bn/moving_mean:0'))
        actor_conv1_moving_variance = sess.run(pretrain_graph.get_tensor_by_name('actor/conv1/bn/moving_variance:0'))
        actor_conv2_beta = sess.run(pretrain_graph.get_tensor_by_name('actor/conv2/bn/beta:0'))
        actor_conv2_gamma = sess.run(pretrain_graph.get_tensor_by_name('actor/conv2/bn/gamma:0'))
        actor_conv2_moving_mean = sess.run(pretrain_graph.get_tensor_by_name('actor/conv2/bn/moving_mean:0'))
        actor_conv2_moving_variance = sess.run(pretrain_graph.get_tensor_by_name('actor/conv2/bn/moving_variance:0'))
        actor_conv3_beta = sess.run(pretrain_graph.get_tensor_by_name('actor/conv3/bn/beta:0'))
        actor_conv3_gamma = sess.run(pretrain_graph.get_tensor_by_name('actor/conv3/bn/gamma:0'))
        actor_conv3_moving_mean = sess.run(pretrain_graph.get_tensor_by_name('actor/conv3/bn/moving_mean:0'))
        actor_conv3_moving_variance = sess.run(pretrain_graph.get_tensor_by_name('actor/conv3/bn/moving_variance:0'))
        actor_conv4_beta = sess.run(pretrain_graph.get_tensor_by_name('actor/conv4/bn/beta:0'))
        actor_conv4_gamma = sess.run(pretrain_graph.get_tensor_by_name('actor/conv4/bn/gamma:0'))
        actor_conv4_moving_mean = sess.run(pretrain_graph.get_tensor_by_name('actor/conv4/bn/moving_mean:0'))
        actor_conv4_moving_variance = sess.run(pretrain_graph.get_tensor_by_name('actor/conv4/bn/moving_variance:0'))

        actor_hidden1_beta = sess.run(pretrain_graph.get_tensor_by_name('actor/hidden1/bn/beta:0'))
        actor_hidden1_gamma = sess.run(pretrain_graph.get_tensor_by_name('actor/hidden1/bn/gamma:0'))
        actor_hidden1_moving_mean = sess.run(pretrain_graph.get_tensor_by_name('actor/hidden1/bn/moving_mean:0'))
        actor_hidden1_moving_variance = sess.run(pretrain_graph.get_tensor_by_name('actor/hidden1/bn/moving_variance:0'))
        # temp_actor_concat_beta = sess.run(pretrain_graph.get_tensor_by_name('actor/concat/bn/beta:0'))
        # temp_actor_concat_gamma = sess.run(pretrain_graph.get_tensor_by_name('actor/concat/bn/gamma:0'))
        # temp_actor_concat_moving_mean = sess.run(pretrain_graph.get_tensor_by_name('actor/concat/bn/moving_mean:0'))

        # temp_actor_concat_moving_variance = sess.run(pretrain_graph.get_tensor_by_name('actor/concat/bn/moving_variance:0'))
        # actor_hidden2_beta = sess.run(pretrain_graph.get_tensor_by_name('actor/hidden2/bn/beta:0'))
        # actor_hidden2_gamma = sess.run(pretrain_graph.get_tensor_by_name('actor/hidden2/bn/gamma:0'))
        # actor_hidden2_moving_mean = sess.run(pretrain_graph.get_tensor_by_name('actor/hidden2/bn/moving_mean:0'))
        # actor_hidden2_moving_variance = sess.run(pretrain_graph.get_tensor_by_name('actor/hidden2/bn/moving_variance:0'))
        # actor_hidden3_beta = sess.run(pretrain_graph.get_tensor_by_name('actor/hidden3/bn/beta:0'))
        # actor_hidden3_gamma = sess.run(pretrain_graph.get_tensor_by_name('actor/hidden3/bn/gamma:0'))
        # actor_hidden3_moving_mean = sess.run(pretrain_graph.get_tensor_by_name('actor/hidden3/bn/moving_mean:0'))
        # actor_hidden3_moving_variance = sess.run(pretrain_graph.get_tensor_by_name('actor/hidden3/bn/moving_variance:0'))
        # out_weight = sess.run(pretrain_graph.get_tensor_by_name('actor/output/weights:0'))
        # out_bias = sess.run(pretrain_graph.get_tensor_by_name('actor/output/biases:0'))

        # TODO : Real 215
        f = 210  # 215
        actor_hidden2_weight = np.random.uniform(-1 / math.sqrt(f), 1 / math.sqrt(f), [f, 200])
        actor_hidden2_weight = np.float32(actor_hidden2_weight)
        actor_hidden2_weight[:210, :] = temp_actor_hidden2_weight

        # actor_concat_beta = np.zeros([f])
        # actor_concat_beta = np.float32(actor_concat_beta)
        # actor_concat_beta[:210] = temp_actor_concat_beta
        # actor_concat_gamma = np.ones([f])
        # actor_concat_gamma = np.float32(actor_concat_gamma)
        # actor_concat_gamma[:210] = temp_actor_concat_gamma
        # actor_concat_moving_mean = np.zeros([f])
        # actor_concat_moving_mean = np.float32(actor_concat_moving_mean)
        # actor_concat_moving_mean[:210] = temp_actor_concat_moving_variance
        # actor_concat_moving_variance = np.ones([f])
        # actor_concat_moving_variance = np.float32(actor_concat_moving_variance)
        # actor_concat_moving_variance[:210] = temp_actor_concat_moving_variance

        f = 215 + 5
        critic_hidden2_weight = np.random.normal(-1 / math.sqrt(f), 1 / math.sqrt(f), [f, 200])
        critic_hidden2_weight = np.float32(critic_hidden2_weight)
        # critic_hidden2_weight[:215, :] = actor_hidden2_weight
        critic_hidden2_weight[:210, :] = actor_hidden2_weight

        tf.reset_default_graph()

        pretrained_model.append(actor_input_beta)
        pretrained_model.append(actor_input_gamma)
        pretrained_model.append(actor_input_moving_mean)
        pretrained_model.append(actor_input_moving_variance)

        pretrained_model.append(actor_conv1_weight)
        pretrained_model.append(actor_conv1_bias)
        pretrained_model.append(actor_conv2_weight)
        pretrained_model.append(actor_conv2_bias)
        pretrained_model.append(actor_conv3_weight)
        pretrained_model.append(actor_conv3_bias)
        pretrained_model.append(actor_conv4_weight)
        pretrained_model.append(actor_conv4_bias)
        pretrained_model.append(actor_hidden1_weight)
        pretrained_model.append(actor_hidden1_bias)
        pretrained_model.append(actor_hidden2_weight)
        pretrained_model.append(actor_hidden2_bias)
        pretrained_model.append(actor_hidden3_weight)
        pretrained_model.append(actor_hidden3_bias)

        pretrained_model.append(actor_conv1_beta)
        pretrained_model.append(actor_conv1_gamma)
        pretrained_model.append(actor_conv1_moving_mean)
        pretrained_model.append(actor_conv1_moving_variance)
        pretrained_model.append(actor_conv2_beta)
        pretrained_model.append(actor_conv2_gamma)
        pretrained_model.append(actor_conv2_moving_mean)
        pretrained_model.append(actor_conv2_moving_variance)
        pretrained_model.append(actor_conv3_beta)
        pretrained_model.append(actor_conv3_gamma)
        pretrained_model.append(actor_conv3_moving_mean)
        pretrained_model.append(actor_conv3_moving_variance)
        pretrained_model.append(actor_conv4_beta)
        pretrained_model.append(actor_conv4_gamma)
        pretrained_model.append(actor_conv4_moving_mean)
        pretrained_model.append(actor_conv4_moving_variance)
        pretrained_model.append(actor_hidden1_beta)
        pretrained_model.append(actor_hidden1_gamma)
        pretrained_model.append(actor_hidden1_moving_mean)
        pretrained_model.append(actor_hidden1_moving_variance)
        #  pretrained_model.append(actor_concat_beta)
        #  pretrained_model.append(actor_concat_gamma)
        #  pretrained_model.append(actor_concat_moving_mean)
        #  pretrained_model.append(actor_concat_moving_variance)
        #  pretrained_model.append(actor_hidden2_beta)
        #  pretrained_model.append(actor_hidden2_gamma)
        #  pretrained_model.append(actor_hidden2_moving_mean)
        #  pretrained_model.append(actor_hidden2_moving_variance)
        #  pretrained_model.append(actor_hidden3_beta)
        #  pretrained_model.append(actor_hidden3_gamma)
        #  pretrained_model.append(actor_hidden3_moving_mean)
        #  pretrained_model.append(actor_hidden3_moving_variance)

        pretrained_model.append(critic_hidden2_weight)

    return pretrained_model


def main():
    config = tf.ConfigProto()
    #  config.gpu_options.allow_growth = True
    #  config.log_device_placement = True

    sess = tf.InteractiveSession()
    pretrained_model = load_pretrained_ckpt(sess)
    segmentation_model = segmentation_graph.SegmentationGraph('segmentation_model\\')
    robot.set_segmentation_model(segmentation_model)

    with tf.Session(config=config) as sess:
        agent = DeepDeterministicPolicyGradientAgent(sess, pretrained_model)
        # setup saver util and either load latest ckpt or init variables
        saver_util = None
        if opts.ckpt_dir is not None:
            saver_util = util.SaverUtil(sess, opts.ckpt_dir, opts.ckpt_freq)
        else:
            sess.run(tf.global_variables_initializer())

        # run either eval or training
        if opts.num_eval > 0:
            agent.run_eval(opts.num_eval, opts.eval_action_noise)
        else:
            agent.run_training(sess, opts.batch_size, opts.batches_per_step, saver_util)
            if saver_util is not None:
                saver_util.force_save()


if __name__ == "__main__":
    main()
