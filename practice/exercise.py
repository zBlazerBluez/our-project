import gym

env = gym.make('CartPole-v0')

print(env.action_space)
env.reset()
for time in range(5):
    observation = env.reset()
    for t in range(100):
    	env.render()
    	print(observation)

    	if (observation[0]<0):
    		action = 1
    	else:
    		action = 0
    	observation, reward, done, info = env.step(action)
    	# if done:
    	# 	print("Episode finished after {} timesteps".format(t+1))
    	# 	break
