"""
PPO for Carla

"""
import threading
import queue
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MAX_EPISODES = 1000
EP_LEN = 200
N_WORKER = 1  # parallel workers
GAMMA = 0.9  # reward discount factor
A_LR = 0.0001  # learning rate for actor
C_LR = 0.0002  # learning rate for critic
MIN_BATCH_SIZE = 64  # minimum batch size for updating PPO
UPDATE_STEP = 10  # loop update operation n-steps
EPSILON = 0.2  # for clipping surrogate objective


class PPO(object):
    def __init__(self, s_dim, a_dim, train=True):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, s_dim], 'state')

        # critic
        l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
        self.v = tf.layers.dense(l1, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        # operation of choosing action
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)
        self.update_oldpi_op = [
            oldp.assign(p) for p, oldp in zip(
                pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, a_dim], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv  # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(  # PPO2
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if train is False:
            self.restore_net()

    def update(self):
        while not COORD.should_stop():
            if GLOBAL_EP < MAX_EPISODES:
                UPDATE_EVENT.wait()  # wait until get batch of data
                self.sess.run(self.update_oldpi_op)  # copy pi to old pi
                # collect data from all workers
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]
                data = np.vstack(data)
                s, a, r = data[:, :s_dim], data[:,
                                                s_dim: s_dim + a_dim], data[:, -1:]
                adv = self.sess.run(
                    self.advantage, {
                        self.tfs: s, self.tfdc_r: r})
                # update actor and critic in a update loop
                [self.sess.run(self.atrain_op,
                               {self.tfs: s,
                                self.tfa: a,
                                self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(
                    UPDATE_STEP)]
                UPDATE_EVENT.clear()  # updating finished
                GLOBAL_UPDATE_COUNTER = 0  # reset counter
                ROLLING_EVENT.set()  # set roll-out available

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(
                self.tfs, 200, tf.nn.relu, trainable=trainable)
            mu_steer = tf.layers.dense(l1, 1, tf.nn.tanh, trainable=trainable)
            mu_throttle = tf.layers.dense(l1, 1, tf.nn.sigmoid, trainable=trainable)
            mu_brake = tf.layers.dense(l1, 1, tf.nn.relu, trainable=trainable)
            mu = tf.concat([mu_steer, mu_throttle, mu_brake], axis=1)
            sigma = tf.layers.dense(
                l1, a_dim, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        a_steer = np.clip(a[0], -1, 1)
        a_throttle = np.clip(a[1], 0, 1)
        a_brake = np.clip(a[2], 0, 1)
        return np.array([a_steer, a_throttle, a_brake])

    def get_v(self, s):
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def save_net(self):
        save_path = self.saver.save(self.sess, 'DDPG' + "/save_net.ckpt")
        print("model has saved in", save_path)

    def restore_net(self):
        self.saver.restore(self.sess, 'DDPG' + "/save_net.ckpt")


class Worker(object):
    def __init__(self, wid, coord):
        self.wid = wid
        self.ppo = GLOBAL_PPO
        self.COORD = coord

    def work(self):
        # global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not self.COORD.should_stop():
            # s = self.env.reset()
            s = np.ones(12)
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():  # while global PPO is updating
                    ROLLING_EVENT.wait()  # wait until PPO is updated
                    # clear history buffer, use new policy to collect data
                    buffer_s, buffer_a, buffer_r = [], [], []
                a = self.ppo.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                # normalize reward, find to be useful
                buffer_r.append((r + 8) / 8)
                s = s_
                ep_r += r

                # count to minimum batch size, no need to wait other workers
                GLOBAL_UPDATE_COUNTER += 1
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []  # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(
                        buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))  # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()  # stop collecting data
                        UPDATE_EVENT.set()  # globalPPO update

                    if GLOBAL_EP >= MAX_EPISODES:  # stop training
                        COORD.request_stop()
                        break

            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(ep_r)
            else:
                GLOBAL_RUNNING_R.append(
                    GLOBAL_RUNNING_R[-1] * 0.9 + ep_r * 0.1)
            GLOBAL_EP += 1
            print(
                '{0:.1f}%'.format(
                    GLOBAL_EP /
                    MAX_EPISODES *
                    100),
                '|W%i' %
                self.wid,
                '|Ep_r: %.2f' %
                ep_r,
            )


def main(train=True):
    s_dim = 12
    a_dim = 3
    s = np.ones(12)
    if train is True:

        UPDATE_EVENT.clear()  # not update now
        ROLLING_EVENT.set()  # start to roll out
        workers = [Worker(wid=i, coord=COORD) for i in range(N_WORKER)]

        GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
        GLOBAL_RUNNING_R = []

        QUEUE = queue.Queue()  # workers putting data in this queue
        threads = []
        for worker in workers:  # worker threads
            t = threading.Thread(target=worker.work, args=())
            t.start()  # training
            threads.append(t)
        # add a PPO updating thread
        threads.append(threading.Thread(target=GLOBAL_PPO.update, ))
        threads[-1].start()
        COORD.join(threads)
        GLOBAL_PPO.save_net()
        # plot reward change and test
        plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
        plt.xlabel('Episode')
        plt.ylabel('Moving reward')
        plt.ion()
        plt.show()
    else:
        pass


if __name__ == '__main__':
    global GLOBAL_EP , GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()  # workers putting data in this queue
    s_dim = 12
    a_dim = 3
    s = np.ones(12)
    GLOBAL_PPO = PPO(s_dim, a_dim)
    main(train=True)
