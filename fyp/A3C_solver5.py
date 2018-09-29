# OpenGym CartPole-v0 with A3C on GPU
# -----------------------------------
#
# A3C implementation with GPU optimizer threads.
#
# Made as part of blog series Let's make an A3C, available at
# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
#
# author: Jaromir Janisch, 2017

import numpy as np
import tensorflow as tf

import gym
import time
import random
import threading

from keras.models import *
from keras.layers import *
from keras import backend as K
# from collections import deque

import environment5 as ev
#-- constants
ENV = 'CartPole-v0'

RUN_TIME = 900
THREADS = 8
OPTIMIZERS = 2
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 4
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0
EPS_STOP = 0
EPS_STEPS = 7500

MIN_BATCH = 32
LEARNING_RATE = 5e-3

LOSS_V = .5         # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient

#---------


class Brain:
    train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self, i):
        self.model_path = 'trained_models/a3c5_v2.h5'
        self.weights_path = 'trained_models/weights_a3c5_v2.h5'
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        if i != 0:
            self.model.load_weights(self.weights_path)
        self.default_graph.finalize()  # avoid modifications

    def _build_model(self):

        l_input = Input(batch_shape=(None, 12, 12, 3))
        # l_input = Reshape((1, 4, 4, 3))(l_input0)

        l1 = Conv2D(32, (5, 5), activation='relu', padding='same')(l_input)
        l2 = Conv2D(64, (5, 5), activation='relu', padding='same')(l1)
        l3 = Flatten()(l2)
        l4 = Flatten()(l_input)
        l5 = Dense(NUM_ACTIONS * 2, activation='relu')(l3)
        l6 = Dense(64, activation='relu')(l4)

        out_actions = Dense(NUM_ACTIONS, activation='softmax')(l5)
        out_value = Dense(1, activation='linear')(l6)

        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model._make_predict_function()  # have to initialize before threading

        return model

    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, 12, 12, 3))
        a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))  # not immediate, but discounted n step reward

        # s_t = tf.reshape(s_t, (-1, 4, 4, 3))
        p, v = model(s_t)

        log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = r_t - v

        loss_policy = - log_prob * tf.stop_gradient(advantage)                                  # maximize policy
        loss_value = LOSS_V * tf.square(advantage)                                              # minimize value error
        entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)  # maximize entropy (regularization)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return s_t, a_t, r_t, minimize, loss_total

    def optimize(self):
        # print('here')
        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0)  # yield
            # print('1')
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < MIN_BATCH:  # more thread could have passed without lock
                # print('2')
                return                                  # we can't yield inside lock

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [[], [], [], [], []]

        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        if len(s) > 5 * MIN_BATCH:
            print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state

        s_t, a_t, r_t, minimize, loss_total = self.graph
        loss = self.session.run([loss_total, minimize], feed_dict={s_t: s, a_t: a, r_t: r})
        print(loss)

    def train_push(self, s, a, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(np.array(NONE_STATE))
                # print('1', np.array(NONE_STATE).shape)
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(np.array(s_))
                # print('2', np.array(s_).shape)
                self.train_queue[4].append(1.)

    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return v

    def save_weights(self):
        self.model.save_weights(self.weights_path)

    # def _load_model(self):
    #     self.model = load_model(self.model_path)

#---------
frames = 0


class Agent:
    def __init__(self, eps_start, eps_end, eps_steps):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps

        # self.memory = deque(100)  # used for n_step return
        self.memory = []
        self.R = 0.

    def getEpsilon(self):
        if(frames >= self.eps_steps):
            return self.eps_end
        else:
            return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate

    def act(self, s, render=False):
        eps = self.getEpsilon()
        global frames
        frames = frames + 1

        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1)

        else:
            # s = np.array([s])
            # p = brain.predict_p(s)[0]
            p = brain.predict_p(s)[0]  # #################################what
            # print(p)

            # a = np.argmax(p)
            a = np.random.choice(NUM_ACTIONS, p=p)
            if render:
                print('output prob')
                print(p)
            return a

    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]

            return s, a, self.R, s_

        a_cats = np.zeros(NUM_ACTIONS)  # turn action into one-hot representation
        a_cats[a] = 1

        self.memory.append((s, a_cats, r, s_))

        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                brain.train_push(s, a, r, s_)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)


    # possible edge case - if an episode ends in <N steps, the computation is incorrect

#---------


class Environment(threading.Thread):
    stop_signal = False

    def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
        threading.Thread.__init__(self)

        self.render = render
        # self.env = gym.make(ENV)
        self.env = ev.Environment()
        self.agent = Agent(eps_start, eps_end, eps_steps)

    def runEpisode(self):
        s = self.env.reset()

        R = 0
        while True:
            time.sleep(THREAD_DELAY)  # yield
            a = 0
            if self.render:
                self.env.render()
            # print(len(self.agent.train_queue))  # CHECK memory leak
            s = s.reshape(-1, 12, 12, 3)
            a = self.agent.act(s)
            #     a = self.agent.act(s, True)
            # else:
            #     a = self.agent.act(s, True)
            # s_, r, done, info = self.env.step(a)
            s_, r, done = self.env.step(a)
            s_ = s_.reshape(-1, 12, 12, 3)
            if done:  # terminal state
                s_ = None

            self.agent.train(s, a, r, s_)

            s = s_
            R += r

            if done or self.stop_signal:
                break

        print("Total R:", R)

    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
        self.stop_signal = True

#---------


class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            brain.optimize()

    def stop(self):
        self.stop_signal = True


#-- main
# env_test = Environment(render=True, eps_start=0., eps_end=0.)
# NUM_STATE = env_test.env.observation_space.shape[0]
# NUM_ACTIONS = env_test.env.action_space.n
NUM_STATE = 432
NUM_ACTIONS = 288
NONE_STATE = np.zeros((1, 12, 12, 3))
# brain = Brain(0)

brain = Brain(1)  # brain is global in A3C

envs = [Environment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]

for o in opts:
    o.start()

for e in envs:
    e.start()

time.sleep(RUN_TIME)

for e in envs:
    e.stop()
for e in envs:
    e.join()

for o in opts:
    o.stop()
for o in opts:
    o.join()

brain.save_weights()
print("Training check_point\n_________________________________________________________________")
# env_test.start()
# time.sleep(2)
# env_test.stop()
