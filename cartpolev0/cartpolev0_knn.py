from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
import gym
import random


class CartPoleAgentKnn:
    """Agent to solve cartpolev0 using K-nearest-neighbors algorithm.

    Learning: The agent saves all the observations and rewards. At the end of each episode,
    it computes the values associated with every observation in that episode.

    Exploit: The agent take K nearest observations based on L2 distance, and choose the action
    that resolves in maximum average value.
    """
    
    def __init__(self, action_space):
        # Hyper-parameters.
        self.needed_mem = 1000
        self.max_memory = 500000
        self.gamma = 0.95
        self.K = 800                  # K nearest neighbor

        # Variables.
        self.frame = 0
        self.mem_pointer = 0
        self.ep_start_pointer = 0
        self.explore_prob = 1.0
        self.explore_prob_decay = 0.98
        
        # Data.
        self.action_space = action_space
        self.db_observations = []
        self.db_rewards = []
        self.db_actions = []
        self.db_values = []
        
    def act(self, observation, reward, done, info):
        best_action = None

        if self.mem_pointer >= self.needed_mem and random.random() > self.explore_prob:
            # Exploit if we have enough data.
            # Select the K nearest neighbor based on L2 distance.
            dist = [sum([observation[i] - db_observation[i] for i in range(len(observation))]) ** 2 for db_observation in self.db_observations]
            ix = np.argsort(dist)
            ix = ix[:min(len(ix), self.K)]

            # Do a vote and select the action with highest average value.
            value_dict = defaultdict(int)
            n_dict = defaultdict(int)
            for i in ix:
                value_dict[self.db_actions[i]] += self.db_values[i]
                n_dict[self.db_actions[i]] += 1
            for action, value in value_dict.items():
                value_dict[action] /= n_dict[action]
            values_array = [(value, action) for action, value in value_dict.items()]
            values_array.sort(reverse=True)
            best_action = values_array[0][1]
        else:
            # Explore.
            best_action = self.action_space.sample()
        
        # Add the current observation to the database.
        if self.mem_pointer < self.max_memory:
            self.db_observations.append(observation)
            self.db_actions.append(best_action)
            self.db_rewards.append(0)
            self.db_values.append(0)
            if self.mem_pointer > 0:
                self.db_rewards[self.mem_pointer-1] = reward
            self.mem_pointer += 1
            
        # At the end of an episode, compute the value.
        if done:
            value = 0
            for i in range(self.mem_pointer - 1, self.ep_start_pointer - 1, -1):
                value = value * self.gamma + self.db_rewards[i]
                self.db_values[i] = value
            self.ep_start_pointer = self.mem_pointer

            self.explore_prob *= self.explore_prob_decay
            
        return best_action

if __name__ == '__main__':
    NUM_EPISODE = 500
    MAX_FRAME = 200

    env = gym.make('CartPole-v0')
    agent = CartPoleAgentKnn(env.action_space)
    reward = 0
    done = False
    running_reward = 0
    for i_episode in range(NUM_EPISODE):
        ob = env.reset()
        sum_reward = 0
        for i_frame in range(MAX_FRAME):
            env.render(mode='rgb_array')
            action = agent.act(ob, reward, done, None)
            ob, reward, done, info = env.step(action)
            sum_reward += reward
            if done:
                # print('Episode %d has %d frames' % (i_episode, i_frame))
                break
        running_reward = running_reward * 0.95 + sum_reward * 0.05
        print('episode %d running reward %f' %(i_episode, running_reward))  