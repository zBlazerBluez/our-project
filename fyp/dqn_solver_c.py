#!/usr/bin/env python
from environment_c import *
import numpy as np
import gym
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DeepQLearningAgent:

    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=10000)
        self.memory2 = deque(maxlen=10000)
        self.batch_size = 128
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_min = 0.15
        self.epsilon_decay = 0.99998    #for 2M ep batch
        self.epsilon_decay = 0.9998     #for 300k ep batch
        self.alpha = 0.01
        self.alpha_decay = 0.01

        # Deep QLearning model macro, output is value of each block choices.
        self.macro_model = Sequential()
        self.macro_model.add(Dense(61, input_dim=34, activation='relu')) #ROW_SIZE*COL_SIZE*2+2 (first 2 layers and 2 remaining number)
        self.macro_model.add(Dense(61, activation='relu'))
        self.macro_model.add(Dense(3, activation='linear'))
        self.macro_model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

        # Deep QLearning for micro, output is (x,y).
        self.micro_model = Sequential()
        self.micro_model.add(Dense(61, input_dim=33, activation='relu')) #ROW_SIZE*COL_SIZE*2+1 (first 2 layers and block chosen)
        self.micro_model.add(Dense(61, activation='relu'))
        self.micro_model.add(Dense(2, activation='relu'))
        self.micro_model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

        self.critic_model = Sequential()
        self.critic_model.add(Dense(61, input_dim=33, activation='relu')) #ROW_SIZE*COL_SIZE*2+1 (first 2 layers and block chosen)
        self.critic_model.add(Dense(61, activation='relu'))
        self.critic_model.add(Dense(2, activation='relu'))
        self.critic_model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

    def remember(self, state, action, reward, next_state, done, value=0):
        self.memory.append([state, action, reward, next_state, done, value])

    def remember2(self, state, action, reward, next_state, done, value=0):
        self.memory2.append([state, action, reward, next_state, done, value])

    def update_value(self,i_step):
        self.memory[-1][5] = self.memory[-1][2]
        for i in range(1,i_step):
            self.memory[-i-1][5] = self.memory[-i-1][2] + self.gamma * self.memory[-i][5]
    def update_value2(self,i_step):
        self.memory2[-1][5] = self.memory2[-1][2]
        for i in range(1,i_step):
            self.memory2[-i-1][5] = self.memory2[-i-1][2] + self.gamma * self.memory2[-i][5]


    def preprocess_state(self, state):
        return np.reshape(state, [1, 34])

    def preprocess_state2(self, state):
        return np.reshape(state, [1, 33])

    def get_state2(self, state, macro_action):
        state2 = state.tolist()[0]
        state2.pop()
        state2.pop()
        state2.append(macro_action)
        return np.reshape(state2, [1, 33])

    def act(self, state):
        if np.random.random() <= self.epsilon:
            # Explore.
            # action = self.env.action_space.sample()
            macro_action = random.randint(0,2)
        else:
            # Exploit.
            macro_action = np.argmax(self.macro_model.predict(state))
            #micro_action = self.micro_model.predict(state.append())
            #action = (np.argmax(self.macro_model.predict(state)),self.micro_model.predict(state))
        return macro_action
    def act2(self, state):
        if np.random.random() <= self.epsilon:
            # Explore.
            # action = self.env.action_space.sample()
            micro_action = (random.randint(0,2),random.randint(0,2))
        else:
            # Exploit.
            #macro_action = np.argmax(self.macro_model.predict(state))
            micro_action = self.micro_model.predict(state)
            #action = (np.argmax(self.macro_model.predict(state)),self.micro_model.predict(state))
        return micro_action

    def learn(self,i_step):
        # minibatch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        state, action, reward, next_state, done, value = self.memory[-1]
        x_batch, y_batch = [], []
        for i in range(i_step):
            state, action, reward, next_state, done, value = self.memory[-i-1]
            x_batch.append(state[0])
            y = self.macro_model.predict(state)[0]
            y[action] = value
            y_batch.append(y)
        self.macro_model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

    def learn2(self,i_step):
        # minibatch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        state, action, reward, next_state, done, value = self.memory2[-1]
        x_batch, y_batch = [], []
        for i in range(i_step):
            state, action, reward, next_state, done, value = self.memory2[-i-1]
            x_batch.append(state[0])
            y = self.micro_model.predict(state)[0]
            y[action] = value
            y_batch.append(y)
        self.micro_model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        self.macro_model.save('trained_double_models/macro1.h5')
        self.micro_model.save('trained_double_models/micro1.h5')

if __name__ == '__main__':
    NUM_EPISODE = 300001
    MAX_FRAME = 50

    f = open('logs/double_models.txt','w')

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

            macro_action = agent.act(state)
            state2 = agent.get_state2(state, macro_action)
            #state2 = preprocess_state2(state2)
            micro_action = agent.act2(state2)

            next_state, reward, done = env.step(macro_action, micro_action)

            next_state = agent.preprocess_state(next_state)
            agent.remember(state, macro_action, reward, next_state, done)
            agent.remember2(state2, micro_action, reward, next_state, done)
            state = next_state
            sum_reward += reward
            i_step += 1

            if i_step > 24:
                reward = -100
                done = True
                
        agent.update_value(i_step)
        agent.update_value2(i_step)
        agent.learn(i_step)
        agent.learn2(i_step)
        running_reward = running_reward * 0.95 + sum_reward * 0.05
        if i_episode % 5000 == 0:
            print('episode %d i_step %d reward %d sum_reward %d running_reward %f' %(i_episode, i_step, reward, sum_reward, running_reward))
            f.write('%d, %d, %d \n' %(sum_reward, running_reward, agent.epsilon))
            agent.save_model()
            
    f.close()