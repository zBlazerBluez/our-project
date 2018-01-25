import random


SQUARE = 0
RECT = 1
HORIZONTAL = 0
VERTICAL = 1
SQUARE = [[1 for _ in range(2)] for _ in range(2)]
RECT_VER = [[1 for _ in range(2)] for _ in range(4)]
RECT_HOR = [[1 for _ in range(4)] for _ in range(2)]
ACTION_DICT = 	{0:(SQUARE,(0,0)), 1:(SQUARE,(0,2)), 2:(SQUARE,(2,0)), 3:(SQUARE,(2,2)),
				4:(RECT_HOR,(0,0)), 5:(RECT_HOR,(2,0)), 
				6:(RECT_VER,(0,0)), 7:(RECT_VER,(0,2))}

class Board(object):
	ROW_SIZE = 4
	COL_SIZE = 4
	SQUARE = 0
	RECT = 1
	HORIZONTAL = 0
	VERTICAL = 1
	def __init__(self):
		self.data = []
		self.num_layer = -1
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
			for layer in range(self.num_layer + 1):
				print(self.data[layer][row]),
			print('')

	def get(self, layer, row, col):
		return self.data[layer][row][col]
	def get_layer(self, layer):
		return self.data[layer]
	def set_current_layer(self, cl, k=0):
		self.data[self.num_layer-k] = cl
	def check_collide(self, block, position, layer):
		x,y = position
		for i in range(len(block)):
			for j in range(len(block[i])):
				if layer[x+i][y+j] == 1:
					return True
		return False
	def add_block(self, block, position):
		# Check if adding command is valid

		cl = self.get_layer(self.num_layer)
		x,y = position
		k = 1
		if self.check_collide(block, position, cl) == False:
			if self.num_layer != 0:
				while (k!=self.num_layer+1 and self.check_collide(block, position, self.get_layer(self.num_layer-k)) == False ):
					print('Dropping 1 layer down')
					k += 1
			cl = self.get_layer(self.num_layer - k + 1)
		else:
			self.add_layer()
			cl = self.get_layer(self.num_layer)
		# Do add nowwwwwwwwwwwwwwwwwwwwwwwwwww
		for i in range(len(block)):
			for j in range(len(block[i])):
				cl[x+i][y+j] = 1
		self.set_current_layer(cl,k-1)
		return True #block added sucessfully

	def compute_reward(self):
		reward = 100
		for i in range(self.num_layer + 1):
			for j in range(4):
				for k in range(4):
					if self.data[i][j][k] == 1:
						reward += 0
					else:
						reward -= 1
		return reward

class Environment(object):
	def __init__(self):
		pass
		# self.board = Board()
		# self.num_square = random.randint(1,19)
		# self.num_rect = 20 - num_square
		# print('There are %d quares and %d rectangulars' %self.num_square %self.num_rect)
		# self.batch_remaining = 4
	def reset(self):
		self.board = Board()
		self.num_square = random.randint(1,7)
		self.num_rect = 8 - self.num_square
		print('There are %d quares and %d rectangulars' %(self.num_square, self.num_rect))

	def render(self):
		self.board.display()
	def get_current_state(self):
		return (self.board.data, self.num_square, self.num_rect)
	def step(self, action):
		reward = -0.1
		done = 0
		current_state = self.get_current_state()	

		block, position = action
		if block == SQUARE:
			if self.num_square == 0:
				return (current_state, reward, done)
			else:
				self.num_square -= 1
		else:
			if self.num_rect == 0:
				return (current_state, reward, done)
			else:
				self.num_rect -= 1
		self.board.add_block(block, position)
		
		if (self.num_square == 0 and self.num_rect == 0):
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





