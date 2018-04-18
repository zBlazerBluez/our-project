#!/usr/bin/env python
from environment import *
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
        self.env = env
        self.memory = deque(maxlen=10000)
        self.batch_size = 128
        self.gamma = 0.95
        self.epsilon = 0.2
        self.epsilon_min = 0.15
        self.epsilon_decay = 0.99998
        self.alpha = 0.01
        self.alpha_decay = 0.01

        # Deep QLearning model.
        self.model = load_model('trained_models/updated_rule_pretrained.h5')

    def remember(self, state, action, reward, next_state, done, value=0):
        self.memory.append([state, action, reward, next_state, done, value])

    def update_value(self,i_step):
    	self.memory[-1][5] = self.memory[-1][2]
    	for i in range(1,i_step):
    		self.memory[-i-1][5] = self.memory[-i-1][2] + self.gamma * self.memory[-i][5]


    def preprocess_state(self, state):
        return np.reshape(state, [1, 34])

    def act(self, state):
        if np.random.random() <= self.epsilon:
            # Explore.
            # action = self.env.action_space.sample()
            action = random.randint(0,14)
        else:
            # Exploit.
            action = np.argmax(self.model.predict(state))
        return action

    def learn(self,i_step):
        # minibatch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        state, action, reward, next_state, done, value = self.memory[-1]
        x_batch, y_batch = [], []
        for i in range(i_step):
        	state, action, reward, next_state, done, value = self.memory[-i-1]
        	x_batch.append(state[0])
        	y = self.model.predict(state)[0]
        	y[action] = value
        	y_batch.append(y)
        # for state, action, reward, next_state, done in minibatch:
        #     y = self.model.predict(state)[0]
        #     y[action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
        #     x_batch.append(state[0])
        #     y_batch.append(y)
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        self.model.save('trained_models/updated_rule_pretrained2.h5')

if __name__ == '__main__':
    NUM_EPISODE = 300000
    MAX_FRAME = 24

    f = open('logs/updated_rule_pretrained2.txt','w')

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
            # env.render(mode='rgb_array')

            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = agent.preprocess_state(next_state)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            sum_reward += reward
            i_step += 1

            if i_step > MAX_FRAME:
            	reward = -100
            	done = True
            	
        agent.update_value(i_step)
        agent.learn(i_step)
        running_reward = running_reward * 0.95 + sum_reward * 0.05
        if i_episode % 5000 == 0:
            print('episode %d i_step %d reward %d sum_reward %d running_reward %f' %(i_episode, i_step, reward, sum_reward, running_reward))
            f.write('%d, %d, %d \n' %(sum_reward, running_reward, agent.epsilon))
            agent.save_model()
        #print('episode %d i_step %d reward %d sum_reward %d running_reward %f' %(i_episode, i_step, reward, sum_reward, running_reward))
        # if (i_episode%100 == 0):
        #     env.render()
    f.close()

    