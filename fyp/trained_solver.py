from environment import *
from keras.models import load_model
import numpy as np

if __name__ == '__main__':
    NUM_EPISODE = 3
    MAX_FRAME = 50

    env = Environment()		
    reward = 0
    done = False
    running_reward = 0
    for i_episode in range(NUM_EPISODE):
        ob = env.reset()
        ob = np.reshape(ob, [1, 34])
        sum_reward = 0
        for i_frame in range(MAX_FRAME):
            print('frame %d immidiate reward %d sum_reward %d' %(i_frame, reward, sum_reward))
            env.render()
            model = load_model('box_stack_dqn.h5')
            action = np.argmax(model.predict(ob))
            ob, reward, done = env.step(action)
            ob = np.reshape(ob, [1, 34])
            sum_reward += reward
            if done:
                # print('Episode %d has %d frames' % (i_episode, i_frame))
                break
        running_reward = running_reward * 0.95 + sum_reward * 0.05
        print('episode %d sum_reward %f' %(i_episode, sum_reward))