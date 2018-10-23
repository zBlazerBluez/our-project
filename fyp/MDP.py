from copy import deepcopy
from environment_dqn import *

NUM_ACTION = 128
GAMMA = 0.95
# queue = [(2, 2), (2, 2), (2, 2), (4, 2), (4, 2), (6, 2), (8, 2), (4, 4), (6, 4)]
# stack = []
# s = 0
# for item in queue[::-1]:
#     s += item[0] * item[1]
#     stack.append(s)

board = Board()
env = Environment()
env.reset()

def copy_env(env):
    new_env = Environment()
    new_env.queue = list(env.queue)
    new_env.board.data = deepcopy(env.board.data)
    new_env.board.num_layer = env.board.num_layer
    return new_env

def action_filter(state):
    action_list = []
    for action in range(NUM_ACTION):
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

def search(env, depth):
    if depth == 0:
        Q_max = -1000
        for i in range(70):
            next_env = copy_env(env)
            Q = 0
            done = False
            step = 0
            while not done:
                _, reward, done = next_env.step(np.random.choice(action_filter(next_env.get_current_state())))
                Q += reward * (GAMMA ** step)
                step += 1
            if Q > Q_max:
                Q_max = Q
        return Q_max, None
    else:
        Q_max = -1000
        action_took = None
        for action in action_filter(env.get_current_state()):
            next_env = copy_env(env)
            _, reward, done = next_env.step(action)
            if done:
                Q = reward
            else:
                next_Q, _ = search(next_env, depth - 1)
                Q = reward + GAMMA * next_Q
            if Q_max < Q:
                Q_max = Q
                action_took = action
        return Q_max, action_took


# def next_state(state, action):


# def reward(state, action):
#     num = 0
#     for i in range(5):
#         for j in range(8):
#             for k in range(8):
#                 num += state[i][j][k]
#     step = 0
#     vol = 0
#     for i in range(len(stack)):
#         if num == stack[i]:
#             step = i + 1
#             vol = stack[i]
#             break

d = False
action_set = []
while not d:
    Q_max, action = search(env, 2)
    print(Q_max, action)
    action_set.append(action)
    _, _, d = env.step(action)

# for action in [34, 2, 64, 1, 16, 20, 32, 34, 36]:
# for action in [32, 2, 112, 65, 2, 26, 30, 44, 46]:
# for action in [43, 64, 3, 19, 1, 49, 99]:
# for action in [34, 65, 3, 1, 116, 17, 106, 33]:
test_env = Environment()
test_env.reset()
for action in action_set:
    test_env.step(action)
    test_env.render()
print(action_set)
