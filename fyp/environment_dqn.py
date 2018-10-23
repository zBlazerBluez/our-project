import random
import numpy as np

ROW_SIZE = 8
COL_SIZE = 8
SQUARE = 0

NUM_STATE = 8 * 8 * 6
NUM_ACTION = 8 * 8 * 2


class Board(object):
    ROW_SIZE = 8
    COL_SIZE = 8

    def __init__(self):
        self.data = []
        self.num_layer = 0
        self.add_layer()
        self.ulti_weight = 100
        self.balance_weight = 50

    def add_layer(self):
        new_layer = [[0 for _ in range(self.ROW_SIZE)] for _ in range(self.COL_SIZE)]
        self.data.append(new_layer)
        self.num_layer += 1

    def display(self):
        # for layer in range(self.num_layer + 1):
        #   for row in range(self.ROW_SIZE):
        #       print(self.data[layer][row])
        #   print('')
        # for row in range(self.ROW_SIZE):
        #     for layer in range(self.num_layer):
        #         print(self.data[layer][row], end='')
        #     print('')
        for row in range(self.ROW_SIZE):
            for layer in range(self.num_layer):
                print('[', end='')
                for col in range(self.COL_SIZE):
                    print(self.data[layer][row][col], end='')
                print(']', end='')
            print('')

    def get(self, layer, row, col):
        return self.data[layer][row][col]

    def get_layer(self, layer):
        if layer < self.num_layer:
            return self.data[layer]
        else:
            return np.zeros((self.ROW_SIZE, self.COL_SIZE))

    def set_current_layer(self, cl, k=0):
        self.data[self.num_layer - k - 1] = cl

    def observable(self):
        ob = np.zeros((6, self.ROW_SIZE, self.COL_SIZE))
        for i in range(5):
            ob[i] = np.array(self.get_layer(i))
        return ob

    def check_collide(self, block, position, layer):
        x, y = position
        for i in range(len(block)):
            for j in range(len(block[i])):
                if layer[x + i][y + j] == 1:
                    return True
        return False

    def add_block(self, block, position):
        # Check if adding command is valid
        guided_punishment = 0
        cl = self.get_layer(self.num_layer - 1)
        x, y = position
        k = 1
        if not self.check_collide(block, position, cl):
            if self.num_layer != 1:
                while (k != self.num_layer and not self.check_collide(block, position, self.get_layer(self.num_layer - k - 1))):
                    # print('Dropping 1 layer down')
                    k += 1
            cl = self.get_layer(self.num_layer - k)
        else:
            # for i in range(len(cl)):
            #     for j in range(len(cl[i])):
            #         if cl[i][j] == 0:
            #             guided_punishment += 0.5
            guided_punishment += 5
            self.add_layer()
            cl = self.get_layer(self.num_layer - 1)
        # Do add nowwwwwwwwwwwwwwwwwwwwwwwwwww
        for i in range(len(block)):
            for j in range(len(block[i])):
                cl[x + i][y + j] = 1
        self.set_current_layer(cl, k - 1)
        return guided_punishment

    def compute_reward(self):
        reward = 0
        count = 0
        ulti = 0.0
        # for layers other than latest one:
        for i in range(self.num_layer):
            for j in range(self.ROW_SIZE):
                for k in range(self.COL_SIZE):
                    if self.data[i][j][k] == 1:
                        count += 1
        ulti = count / (self.num_layer * self.ROW_SIZE * self.COL_SIZE)
        reward += self.ulti_weight * ulti
        # for latest layer, check for symetry:
        up, down, left, right = 0, 0, 0, 0
        for j in range(self.ROW_SIZE):
            for k in range(self.COL_SIZE):
                if self.data[self.num_layer - 1][j][k] == 1:
                    if (j < self.ROW_SIZE / 2):
                        up += 1
                    else:
                        down += 1
                    if (k < self.COL_SIZE / 2):
                        left += 1
                    else:
                        right += 1
        unbalance_index = (abs(up - down) + abs(left - right)) / (self.ROW_SIZE * self.COL_SIZE)
        reward -= unbalance_index * self.balance_weight
        return reward


class Environment(object):
    def __init__(self):
        self.ROW_SIZE = ROW_SIZE
        self.COL_SIZE = COL_SIZE
        self.NUM_ACTION = NUM_ACTION
        self.NUM_STATE = NUM_STATE
        # self.action_space = [x for x in range(self.NUM_ACTION)]
        # self.queue = [(2, 2), (2, 2), (2, 2), (4, 2), (4, 2), (6, 2), (8, 2), (4, 4), (6, 4)]
        # self.queue = [(3, 1), (7, 1), (4, 2), (3, 3), (5, 2), (4, 3), (5, 3)]
        # self.queue = [(4, 4), (6, 2), (4, 2), (4, 2), (6, 2), (4, 4), (4, 2), (4, 4)]
        self.queue = [(4, 4), (4, 4), (4, 4), (6, 2), (4, 2), (2, 4), (6, 2), (4, 4), (4, 2), (4, 4)]
        self.board = Board()

    def reset(self):
        self.board = Board()
        # self.queue = [(2, 2), (2, 2), (2, 2), (4, 2), (4, 2), (6, 2), (8, 2), (4, 4), (6, 4)]
        # self.queue = [(3, 1), (7, 1), (4, 2), (3, 3), (5, 2), (4, 3), (5, 3)]
        # self.queue = [(4, 4), (6, 2), (4, 2), (4, 2), (6, 2), (4, 4), (4, 2), (4, 4)]
        self.queue = [(4, 4), (4, 4), (4, 4), (6, 2), (4, 2), (2, 4), (6, 2), (4, 4), (4, 2), (4, 4)]
        #print('There are %d quares and %d rectangulars' %(self.num_square, self.num_rect))
        return self.get_current_state()

    def render(self):
        # self.board.display()
        print(self.get_current_state())

    def get_current_state(self):
        observation = self.board.observable()
        last_layer = np.zeros((self.ROW_SIZE, self.COL_SIZE))
        if self.queue:
            for i in range(self.queue[-1][0]):
                for j in range(self.queue[-1][1]):
                    last_layer[i][j] = 1
        observation[5] = last_layer
        return observation

    def step(self, action):
        reward = 0
        done = 0
        check = 0
        current_state = self.get_current_state()
        # pos_x, pos_y = ACTION_DICT[action % len(ACTION_DICT)]
        if action < 8 * 8:
            row = action % 8
            col = action // 8
        else:
            col = (action - 64) % 8
            row = (action - 64) // 8
        pos_x = row
        pos_y = col
        # print("pos_x, pos_y: " + str(pos_x) + str(pos_y))
        if action < 64:
            x, y = self.queue[-1]
        else:
            y, x = self.queue[-1]
        # print("x, y: " + str(x) + str(y))
        block = [[1 for _ in range(y)] for _ in range(x)]
        position = (pos_x, pos_y)

        if pos_x + len(block) > COL_SIZE or pos_y + len(block[0]) > ROW_SIZE:
            print("Warning, invalid move chosen. action chosen:" + str(action))
            print(block)
            print("pos_x:" + str(pos_x) + "\tlen_x:" + str(len(block)))
            print("pos_y:" + str(pos_y) + "\tlen_y:" + str(len(block[0])))
            return (current_state, -10, done)
        self.queue.pop()
        # print('popped')
        reward -= self.board.add_block(block, position)
        if self.board.num_layer > 5:
            return (current_state, -100, 1)
        if (len(self.queue) == 0):
            done = 1
            reward += self.board.compute_reward()
        next_state = self.get_current_state()
        return (next_state, reward, done)


# board = Board()
# board.add_block(SQUARE,(1,2))
# board.add_block(RECT_VER,(0,0))
# # board.add_block(RECT_VER,(1,2))
# # board.add_block(RECT_VER,(1,4))
# # board.add_block(RECT_VER,(1,2))
# # board.add_block(RECT_HOR,(1,4))
# board.display()
# print('Reward is equal to: %d' %board.compute_reward())
# raw_input()
