#!/usr/bin/env python
from environment_two import *
import numpy as np
import gym
import random
from collections import deque
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DeepQLearningAgent:

    def __init__(self, env):
        self.NUM_ACTION = 128
        self.NUM_STATE = 384
        self.NUM_STATE_ACTION = 385
        self.env = env
        self.memory = deque(maxlen=1000)
        self.batch_size = 10
        self.gamma = 0.95
        self.epsilon = 0
        self.epsilon_min = 0
        # self.epsilon_decay = 0.99998    #for 2M ep batch
        self.epsilon_decay = 0.99998  # for 1k ep batch
        self.alpha = 0.01
        self.alpha_decay = 0.01
        self.learning_rate = 0.9

        self.model = load_model('trained_models/8x8_three.h5')

    def remember(self, state, action, reward, next_state, done, value=0):
        self.memory.append([state, action, reward, next_state, done])

    # def update_value(self, i_step):
    #     self.memory[-1][5] = self.memory[-1][2]
    #     for i in range(1, i_step):
    #         self.memory[-i - 1][5] = self.memory[-i - 1][2] + self.gamma * self.memory[-i][5]

    def preprocess_state(self, state):
        return np.reshape(state, (6, 8, 8))

    def action_filter(self, state):
        action_list = []
        for action in range(self.NUM_ACTION):
            i = 0
            flag = 0
            if action < 8 * 8:
                row = action % 8
                col = action // 8
            else:
                col = (action - 64) % 8
                row = (action - 64) // 8
            while (i < 8 and state[-1, 0, i] == 1):
                if (action < 64 and col + i > 7) or (action >= 64 and row + i > 7):
                    flag = 1
                i += 1
            if flag == 1:
                continue
            j = 0
            while (j < 8 and state[-1, j, 0] == 1):
                if (action < 64 and row + j > 7) or (action >= 64 and col + j > 7):
                    flag = 1
                j += 1
            if flag == 1:
                continue
            action_list.append(action)
        return action_list

    def best_action(self, state, actions):
        action = 0
        state_action = np.reshape(np.append(state.flatten(), action), [1, self.NUM_STATE_ACTION])
        # print(state_action.shape)
        value = self.model.predict(state_action)
        max_value = value
        max_action = action

        print(state[:3])
        # print("action:" + str(action) + "\tvalue:" + str(value))
        for action in actions:
            state_action = np.reshape(np.append(state.flatten(), action), [1, self.NUM_STATE_ACTION])
            value = self.model.predict(state_action)
            print("action:" + str(action) + "\tvalue:" + str(value))
            if value > max_value:
                max_value = value
                max_action = action
        return max_action, max_value

    def act(self, state):
        actions = self.action_filter(state)
        if np.random.random() <= self.epsilon:
            # Explore.
            # action = self.env.action_space.sample()
            max_action = np.random.choice(actions)
        else:
            # Exploit.
            max_action, max_value = self.best_action(state, actions)
            # print("len: " + str(len(lis)) + "\tlist: " + str(lis))
            # print("Action chosen:" + str(max_action))
        return max_action

    def learn(self, i_step):
        # minibatch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        state, action, reward, next_state, done = self.memory[-1]
        x_batch, y_batch = [], []
        mean = 0
        indexes = random.sample(list(range(len(self.memory))), self.batch_size if self.batch_size < len(self.memory) else len(self.memory))
        for i in indexes:
            state, action, reward, next_state, done = self.memory[-i - 1]
            state_action = np.append(state.flatten(), action)
            if done:
                value = reward
                predicted_value = self.model.predict(np.reshape(state_action, [1, self.NUM_STATE_ACTION]))
            else:
                next_actions = self.action_filter(next_state)
                _, max_value = self.best_action(state, next_actions)
                predicted_value = self.model.predict(np.reshape(state_action, [1, self.NUM_STATE_ACTION]))
                value = (1 - self.learning_rate) * predicted_value + self.learning_rate * (reward + self.gamma * max_value)
                value = value[0]
            x_batch.append(state_action)
            y_batch.append(value)
            mean += (value - predicted_value) ** 2
        # for state, action, reward, next_state, done in minibatch:
        #     y = self.model.predict(state)[0]
        #     y[action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
        #     x_batch.append(state[0])
        #     y_batch.append(y)
        mean = mean / self.batch_size
        print("Current mean square error: %.2f" % (mean))
        # print(np.array(y_batch).shape)
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        self.model.save('trained_models/8x8_three.h5')


if __name__ == '__main__':
    NUM_EPISODE = 5
    MAX_FRAME = 10

    env = Environment()
    agent = DeepQLearningAgent(env)
    reward = 0
    done = False
    running_reward = 0
    for i_episode in range(NUM_EPISODE):
        state = agent.preprocess_state(env.reset())
        sum_reward = 0
        done = False
        i_step = 0
        while not done:
            # print(state.shape)
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            print("Reward: " + str(reward))
            next_state = agent.preprocess_state(next_state)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            sum_reward += reward
            i_step += 1

            if i_step > MAX_FRAME:
                reward = -10000
                done = True
        agent.learn(i_step)
        running_reward = running_reward * 0.95 + sum_reward * 0.05
        print('episode %d i_step %d reward %d sum_reward %d running_reward %f epsilon %.2f' % (i_episode, i_step, reward, sum_reward, running_reward, agent.epsilon))
