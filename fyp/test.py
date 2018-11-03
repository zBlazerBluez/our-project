from environment_dqn import *
import numpy as np

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.models import load_model

NUM_STATE_ACTION = 385
memory = deque(maxlen=200)
batch_size = 100

def action_filter(state):
    action_list = []
    for action in [0, 1, 70, 100]:
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

# model = Sequential()

# model.add(Dense(128, input_dim=NUM_STATE_ACTION, activation='relu'))  # ROW_SIZE*COL_SIZE*2+1 (first 2 layers and action)
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='linear'))

# model.compile(loss='mse', optimizer=Adam())
model = load_model('trained_models/test.h5')

env = Environment()
f = open('logs/test.txt', 'w')
running_loss = 0

for i in range(1000000):
    state = np.reshape(env.reset(), (6, 8, 8))
    done = False
    while not done:
        action = np.random.choice(action_filter(state))
        next_state, reward, done = env.step(action)
        state_action = np.append(state.flatten(), action)
        memory.append([state_action, reward])
        # print(state_action.shape)
        # X = []
        # X.append(state_action)
        # Y = []
        # Y.append(reward)
        indexes = random.sample(list(range(len(memory))), batch_size if batch_size < len(memory) else len(memory))
        X = []
        Y = []
        for index in indexes:
            X.append(memory[index][0])
            Y.append(memory[index][1])
        if i % 500 == 0:
            predicted_r = float(model.predict(np.reshape(state_action, [1, NUM_STATE_ACTION]))[0][0])
        model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0)
        # print(hist.history)
        state = np.reshape(next_state, (6, 8, 8))
        running_loss = running_loss * 0.95 + (reward - predicted_r)**2 * 0.05
        if i % 500 == 0:
            predicted_r2 = float(model.predict(np.reshape(state_action, [1, NUM_STATE_ACTION]))[0][0])
            print((reward - predicted_r)**2, reward, predicted_r, running_loss)
            print((reward - predicted_r2)**2, reward, predicted_r2)
            f.write("%lf, %lf, %lf, %lf, %lf\n" % ((reward - predicted_r)**2, running_loss, reward, predicted_r, predicted_r2))
            model.save('trained_models/test.h5')

f.close()
