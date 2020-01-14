"""
 DDPG for Carla 0.9.6
"""

import tensorflow as tf
import numpy as np
import time
import os
import shutil
import prioritied_sampling as p
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


#  hyper parameters
LOG_DIR = './logs'  # tensorboard
MAX_EPISODES = 100000  # total episodes in training
LR_A = 0.001  # learning rate for actor net
LR_C = 0.002  # learning rate for critic net
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000  # memory size
BATCH_SIZE = 32
RENDER = False  # display


#  RL Method

class DDPG(object):
    def __init__(self, a_dim, s_dim, train=True,
                 tensorboard_graph=True,
                 memory_size=10000,
                 batch_size=32,
                 prioritized=True,
                 ):
        tf.reset_default_graph()
        self.memory = np.zeros(
            (MEMORY_CAPACITY,
             s_dim * 2 + a_dim + 1),
            dtype=np.float32)  # initialize memory buffer
        self.sess = tf.Session()
        self.a_dim, self.s_dim = a_dim, s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.a = self._build_a(self.S)
        self.prioritized = prioritized
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory_counter = 0

        # --- prioritized  ddpg
        if self.prioritized:
            self.memory = p.Memory(capacity=memory_size)
        # need checking
        else:
            self.memory = np.zeros((self.memory_size, s_dim*2+2))

        # evaluate Q(s,a)
        q = self._build_c(self.S, self.a, )
        # get valuable
        a_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        # soft replacement
        ema = tf.train.ExponentialMovingAverage(
            decay=1 - TAU)

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        # soft update operation
        target_update = [ema.apply(a_params), ema.apply(c_params)]
        # replaced target parameters, 'reuse' create multiple parallel or nested namespaces
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(
            LR_A).minimize(a_loss, var_list=a_params)

        # soft replacement happened at here
        # minimize td_error to train c_net
        with tf.control_dependencies(target_update):
            q_target = self.R + GAMMA * q_
            self.td_error = tf.losses.mean_squared_error(
                labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(
                LR_C).minimize(self.td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if train is False:
            self.restore_net()
        self.restore_net()
        if tensorboard_graph:
            if os.path.exists(LOG_DIR):
                shutil.rmtree(LOG_DIR)
            tf.summary.FileWriter(LOG_DIR, self.sess.graph)

    def choose_action(self, s):
        a = self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
        return a

    def learn(self):
        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

        bs = batch_memory[:, :self.s_dim]
        ba = batch_memory[:, self.s_dim: self.s_dim + self.a_dim]
        br = batch_memory[:, -self.s_dim - 1: -self.s_dim]
        bs_ = batch_memory[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(
            self.ctrain, {
                self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        if self.prioritized:
            abs_errors = self.sess.run(
            self.td_error, {
                self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
            self.memory.batch_update(tree_idx, abs_errors)     # update priority





    def store_transition(self, s, a, r, s_):
        if self.prioritized:    # prioritized replay
            transition = np.hstack((s, a, r, s_))
            self.memory.store(transition)
            self.memory_counter += 1

        else:       # random replay
            transition = np.hstack((s, a, r, s_))
            # replace the old memory with new memory
            index = self.memory_counter % MEMORY_CAPACITY
            self.memory[index, :] = transition
            self.memory_counter += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            layer1 = tf.layers.dense(
                s,
                100,
                activation=tf.nn.relu,
                name='l1',
                trainable=trainable,
                kernel_initializer=w_init
            )
            layer2 = tf.layers.dense(
                layer1,
                100,
                activation=tf.nn.relu,
                name='l2',
                trainable=trainable,
                kernel_initializer=w_init
            )
            steer = tf.layers.dense(
                layer2,
                1,
                activation=tf.nn.tanh,
                name='a_s',
                trainable=trainable,
                kernel_initializer=w_init
            )
            throttle = tf.layers.dense(
                layer2,
                1,
                activation=tf.nn.sigmoid,
                name='a_t',
                trainable=trainable,
                kernel_initializer=w_init
            )
            brake = tf.layers.dense(
                layer2,
                1,
                activation=tf.nn.relu,
                name='a_b',
                trainable=trainable,
                kernel_initializer=w_init
            )
            steer = tf.clip_by_value(steer, -1, 1)
            throttle = tf.clip_by_value(throttle, 0, 1)
            brake = tf.clip_by_value(brake, 0, 1)
            a = tf.concat([steer, throttle, brake], axis=1)
            return a

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 100
            w1_s = tf.get_variable(
                'w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable(
                'w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            layer1 = tf.nn.relu(
                tf.matmul(
                    s,
                    w1_s) +
                tf.matmul(
                    a,
                    w1_a) +
                b1,
                name='c_l1')
            layer2 = tf.layers.dense(
                layer1,
                n_l1,
                activation=tf.nn.relu,
                name='c_l2',
                trainable=trainable,
                kernel_initializer=w_init
            )

            return tf.layers.dense(layer2, 1, trainable=trainable,
                kernel_initializer=w_init
            )  # Q(s,a)

    def save_net(self):
        save_path = self.saver.save(self.sess, 'DDPG' + "/save_net.ckpt")
        print("model has saved in", save_path)

    def restore_net(self):
        self.saver.restore(self.sess, 'DDPG' + "/save_net.ckpt")
        print('model has restored')


# TRAIN OR TEST

# def main(train=True):
#     s_dim = 12
#     a_dim = 3
#
#     if train:
#         ddpg = DDPG(a_dim, s_dim, train=train)
#         s = np.ones(12)  # s = world.reset()
#         var = 3  # control exploration
#         t1 = time.time()
#         for i in range(MAX_EPISODES):
#
#             ep_reward = 0
#             while True:
#                 # Add exploration noise
#                 a = ddpg.choose_action(s)
#                 # add randomness to action selection for exploration
#                 # s_, r, done = world.step(a)
#
#                 ddpg.store_transition(s, a, r / 10, s_)
#
#                 if ddpg.memory_counter > MEMORY_CAPACITY:
#                     var *= .9995  # decay the action randomness
#                     ddpg.learn()
#
#                 s = s_
#                 ep_reward += r
#                 if done:
#                     print(
#                         'Episode:',
#                         i,
#                         ' Reward: %i' %
#                         int(ep_reward),
#                         'Explore: %.2f' %
#                         var,
#                     )
#                     # if ep_reward > -300:RENDER = True
#                     break
#         ddpg.save_net()
#         print('Running time: ', time.time() - t1)
#
#     else:
#
#         s = np.ones(12)  # s = world.reset()
#         ddpg = DDPG(a_dim, s_dim, train=train)
#         saver = tf.train.Saver()
#         var = 3  # control exploration
#         t1 = time.time()
#         for i in range(MAX_EPISODES):
#
#             ep_reward = 0
#             while True:
#                 # Add exploration noise
#                 steer, throttle, brake, a = ddpg.choose_action(s)
#                 # add randomness to action selection for exploration
#                 # s_, r, done, info = world.step(a)
#
#                 s = s_
#                 ep_reward += r
#                 if done:
#                     print(
#                         'Episode:',
#                         i,
#                         ' Reward: %i' %
#                         int(ep_reward),
#                         'Explore: %.2f' %
#                         var,
#                     )
#                     # if ep_reward > -300:RENDER = True
#                     break
#
#         print('Running time: ', time.time() - t1)
#
#
# if __name__ == '__main__':
#     main(train=True)
