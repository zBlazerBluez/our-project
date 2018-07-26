#!/usr/bin/env python
from environment2 import *
from keras.models import load_model
import numpy as np

if __name__ == '__main__':
    NUM_EPISODE = 3
    MAX_FRAME = 400

    env = Environment()

    running_reward = 0
    for i_episode in range(NUM_EPISODE):
        reward = 0
        done = False
        ob = env.reset()
        ob = np.reshape(ob, [1, 290])
        sum_reward = 0
        for i_frame in range(MAX_FRAME):
            print('frame %d immidiate reward %d sum_reward %d' % (i_frame, reward, sum_reward))
            env.render()
            model = load_model('trained_models/12x12.h5')
            predicted_values = model.predict(ob)
            action = np.argmax(predicted_values)
            #action = 9;
            print(predicted_values)
            print('Chose action %d' % action)
            ob, reward, done = env.step(action)
            ob = np.reshape(ob, [1, 290])
            sum_reward += reward
            if done:
                # print('Episode %d has %d frames' % (i_episode, i_frame))
                break
        env.render()
        running_reward = running_reward * 0.95 + sum_reward * 0.05
        print('***Episode %d i_frame %d reward %d sum_reward %f' % (i_episode, i_frame, reward, sum_reward))
        print('____________________________________________________________________________________________')
