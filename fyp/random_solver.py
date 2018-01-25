from environment import *

if __name__ == '__main__':
    NUM_EPISODE = 3
    MAX_FRAME = 50

    env = Environment()		
    reward = 0
    done = False
    running_reward = 0
    for i_episode in range(NUM_EPISODE):
        ob = env.reset()
        sum_reward = 0
        for i_frame in range(MAX_FRAME):
            print(i_frame)
            env.render()
            action = ACTION_DICT[random.randint(0,7)]
            ob, reward, done = env.step(action)
            sum_reward += reward
            if done:
                # print('Episode %d has %d frames' % (i_episode, i_frame))
                break
        running_reward = running_reward * 0.95 + sum_reward * 0.05
        print('episode %d sum_reward %f' %(i_episode, sum_reward))