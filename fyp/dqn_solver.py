from environment import *
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
        self.batch_size = 128
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.998
        self.alpha = 0.01
        self.alpha_decay = 0.01

        # Deep QLearning model.
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=34, activation='tanh')) #ROW_SIZE*COL_SIZE*2+2 (first 2 layers and 2 remaining number)
        self.model.add(Dense(48, activation='tanh'))
        self.model.add(Dense(15, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

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

    def learn(self):
        minibatch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        x_batch, y_batch = [], []
        for state, action, reward, next_state, done in minibatch:
            y = self.model.predict(state)[0]
            y[action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y)
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        self.model.save('box_stack_dqn.h5')

if __name__ == '__main__':
    NUM_EPISODE = 5000
    MAX_FRAME = 50

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
        agent.learn()
        running_reward = running_reward * 0.95 + sum_reward * 0.05
        print('episode %d reward %d running_reward %f' %(i_episode, reward, running_reward))
        # if (i_episode%100 == 0):
        #     env.render()

    agent.save_model()