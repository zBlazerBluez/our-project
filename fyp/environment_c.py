import random

ROW_SIZE = 4
COL_SIZE = 4
SQUARE_CODE = 0
RECT_VER_CODE = 1
RECT_HOR_CODE = 2
HORIZONTAL = 0
VERTICAL = 1
SQUARE = [[1 for _ in range(2)] for _ in range(2)]
RECT_VER = [[1 for _ in range(2)] for _ in range(4)]
RECT_HOR = [[1 for _ in range(4)] for _ in range(2)]
MACRO_DICT = {0: SQUARE, 1: RECT_VER, 2: RECT_HOR}
ACTION_DICT = {0: (SQUARE, (0, 0)), 1: (SQUARE, (0, 2)), 2: (SQUARE, (2, 0)), 3: (SQUARE, (2, 2)),
               4: (RECT_HOR, (0, 0)), 5: (RECT_HOR, (2, 0)),
               6: (RECT_VER, (0, 0)), 7: (RECT_VER, (0, 2)),
               8: (SQUARE, (0, 1)), 9: (SQUARE, (1, 0)), 10: (SQUARE, (1, 1)), 11: (SQUARE, (1, 2)), 12: (SQUARE, (2, 1)),
               13: (RECT_HOR, (1, 0)), 14: (RECT_VER, (0, 1))}


class Board(object):
    ROW_SIZE = 4
    COL_SIZE = 4
    SQUARE = 0
    RECT = 1
    HORIZONTAL = 0
    VERTICAL = 1

    def __init__(self):
        self.data = []
        self.num_layer = 0
        self.add_layer()

    def add_layer(self):
        new_layer = [[0 for _ in range(self.ROW_SIZE)] for _ in range(self.COL_SIZE)]
        self.data.append(new_layer)
        self.num_layer += 1

    def display(self):
        # for layer in range(self.num_layer + 1):
        # 	for row in range(self.ROW_SIZE):
        # 		print(self.data[layer][row])
        # 	print('')
        for row in range(self.ROW_SIZE):
            for layer in range(self.num_layer):
                print(self.data[layer][row], end='')
            print('')

    def get(self, layer, row, col):
        return self.data[layer][row][col]

    def get_layer(self, layer):
        return self.data[layer]

    def set_current_layer(self, cl, k=0):
        self.data[self.num_layer - k - 1] = cl

    def observable(self):
        layers_to_look_at = 2
        vectorized_list = []
        if (self.num_layer == 1):
            layers_to_look_at = 1
        # looking at highest layer
        for i in range(self.ROW_SIZE):
            for j in range(self.COL_SIZE):
                vectorized_list.append(self.data[self.num_layer - 1][i][j])
        if self.num_layer == 1:
            for _ in range(self.ROW_SIZE * self.COL_SIZE):
                vectorized_list.append(0)
        else:
            for i in range(self.ROW_SIZE):
                for j in range(self.COL_SIZE):
                    vectorized_list.append(self.data[self.num_layer - 2][i][j])
        return vectorized_list

    def check_collide(self, block, position, layer):
        x, y = position
        for i in range(len(block)):
            for j in range(len(block[i])):
                if layer[x + i][y + j] == 1:
                    return True
        return False

    def add_block(self, block_code, position):
        # Check if adding command is valid
        block = MACRO_DICT[block_code]
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
            for i in range(len(cl)):
                for j in range(len(cl[i])):
                    if cl[i][j] == 0:
                        guided_punishment += 2
            self.add_layer()
            cl = self.get_layer(self.num_layer - 1)
        # Do add nowwwwwwwwwwwwwwwwwwwwwwwwwww
        for i in range(len(block)):
            for j in range(len(block[i])):
                cl[x + i][y + j] = 1
        self.set_current_layer(cl, k - 1)
        return guided_punishment

    def compute_reward(self):
        reward = 116
        # for layers other than latest one:
        for i in range(self.num_layer - 1):
            for j in range(self.ROW_SIZE):
                for k in range(self.COL_SIZE):
                    if self.data[i][j][k] == 1:
                        reward += 0
                    else:
                        reward -= 1
        # for latest layer, check for symetry:
        up, down, left, right = 0, 0, 0, 0
        for j in range(self.ROW_SIZE):
            for k in range(self.COL_SIZE):
                if self.data[self.num_layer - 1][j][k] == 1:
                    if (j < 2):
                        up += 1
                    else:
                        down += 1
                    if (k < 2):
                        left += 1
                    else:
                        right += 1
        unbalance_index = abs(up - down) + abs(left - right)
        reward -= unbalance_index * 1
        return reward


class Environment(object):
    def __init__(self):
        self.macro_action_space = [x for x in range(3)]
        self.micro_action_space = [x for x in range(2)]
        # self.board = Board()
        # self.num_square = random.randint(1,19)
        # self.num_rect = 20 - num_square
        # print('There are %d quares and %d rectangulars' %self.num_square %self.num_rect)
        # self.batch_remaining = 4

    def reset(self):
        self.board = Board()
        self.num_square = random.randint(1, 7)
        self.num_rect = 8 - self.num_square
        #print('There are %d quares and %d rectangulars' %(self.num_square, self.num_rect))
        return self.get_current_state()

    def render(self):
        self.board.display()

    def get_current_state(self):
        observation = self.board.observable()
        observation.append(self.num_square)
        observation.append(self.num_rect)
        return observation

    def step(self, macro_action, micro_action):
        reward = -2
        done = 0
        check = 0
        current_state = self.get_current_state()
        # if current_state[:32] == [0 for _ in range(32)] and action == 5: #need change
        #	check = 1

        #block, position = ACTION_DICT[action]
        if macro_action == SQUARE_CODE:
            if self.num_square == 0:
                return (current_state, reward, done)
            else:
                self.num_square -= 1
        else:
            if self.num_rect == 0 or (macro_action == RECT_HOR_CODE and micro_action[1] > 0) or (macro_action == RECT_VER_CODE and micro_action[0] > 0):
                return (current_state, reward, done)
            else:
                self.num_rect -= 1
        print(macro_action)
        print(micro_action)
        reward -= self.board.add_block(macro_action, micro_action)

        if (self.num_square == 0 and self.num_rect == 0):
            done = 1
            reward += self.board.compute_reward()
        next_state = self.get_current_state()
        if check == 1:
            reward = -50
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
