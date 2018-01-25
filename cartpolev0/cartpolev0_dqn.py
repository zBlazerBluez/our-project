from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import gym
import random

class AgentKhang:
	def __init__(self, env):
		self.memory = deque(maxlen=5000)
		self.epsilon = 1.0
		self.epsilon_min = 0.0001
		self.epsilon_decay = 0.998
		self.gamma = 0.95
		self.env = env
		self.batch_size = 64

		self.alpha = 0.01;
		self.alpha_decay = 0.01;
		self.model = Sequential()
		self.model.add(Dense(24, input_dim=4, activation='tanh'))
		self.model.add(Dense(48, activation='tanh'))
		self.model.add(Dense(2, activation='linear'))
		self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if (np.random.random() <= self.epsilon):
			# Explore.
			action = self.env.action_space.sample()
		else:
			# Exploit.
			action = np.argmax(self.model.predict(state))
		return action
	def learn(self):
		x_batch = []
		y_batch = []
		minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
		for state, action, reward, next_state, done in minibatch:
			y = self.model.predict(state)
			if done:
				y[0][action] = reward
			else:
				y[0][action] = reward + self.gamma*np.max(self.model.predict(next_state))
			x_batch.append(state[0])
			y_batch.append(y[0])

		self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def preprocess_state(self, state):
		return np.reshape(state, [1,4])

if __name__ == '__main__':
    NUM_EPISODE = 500
    MAX_FRAME = 200

    env = gym.make('CartPole-v0')
    agent = AgentKhang(env)
    reward = 0
    done = False
    running_reward = 0
    for i_episode in range(NUM_EPISODE):
        state = env.reset()
        state = agent.preprocess_state(state)
        sum_reward = 0
        for i_frame in range(MAX_FRAME):
            env.render()
            action = agent.act(state)

            next_state, reward, done, info = env.step(action)
            next_state = agent.preprocess_state(next_state)
            #print(i_frame)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            sum_reward += reward
            if done:
                print('Episode %d has %d frames' % (i_episode, i_frame))
                break
        
        running_reward = running_reward * 0.95 + sum_reward * 0.05
        print('episode %d running reward %f' %(i_episode, running_reward))  