
SQUARE = 0
RECT = 1
HORIZONTAL = 0
VERTICAL = 1
SQUARE = [[1 for _ in range(2)] for _ in range(2)]
RECT_VER = [[1 for _ in range(2)] for _ in range(4)]
RECT_HOR = [[1 for _ in range(4)] for _ in range(2)]

class Board(object):
	ROW_SIZE = 8
	COL_SIZE = 8
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
		for layer in range(self.num_layer + 1):
			# for row in range(self.ROW_SIZE):
				# print(self.data[layer][row])
			for row in range(4):
				print(self.data[layer][row][:4])
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
	def add_block(self, block, position, orientation = 0):
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

	def compute_reward(self):
		reward = 0
		for i in range(self.num_layer + 1):
			for j in range(4):
				for k in range(4):
					if self.data[i][j][k] == 1:
						reward += 1
					else:
						reward -= 0.5
		return reward

class Environment(object):
	def __init__(self):
		self.board = Board()
		self.num_square = random(4)
		self.num_rect = 4 - num_square
		self.remaining = 20
	def reset(self):
		self.board = Board()
	def render(self):
		print()

board = Board()
board.add_block(SQUARE,(1,2))
board.add_block(RECT_VER,(0,0))
# board.add_block(RECT_VER,(1,2))
# board.add_block(RECT_VER,(1,4))
# board.add_block(RECT_VER,(1,2))
# board.add_block(RECT_HOR,(1,4))
board.display()
print('Reward is equal to: %d' %board.compute_reward())
raw_input()

		# if block_type == self.SQUARE:
		# 	if (cl[x][y] == 1 or cl[x][y+1] == 1 or cl[x+1][y] == 1 or cl[x+1][y+1] == 1):
		# 		self.add_layer()
		# 	cl = self.get_layer(self.num_layer)
		# 	cl[x][y] = 1
		# 	cl[x+1][y] = 1
		# 	cl[x][y+1] = 1
		# 	cl[x+1][y+1] = 1
		# 	# self.set_current_layer(cl)
		# else:
		# 	if orientation == self.HORIZONTAL:
		# 		print("HERE1")
		# 		if (cl[x][y] == 1 or cl[x][y+1] == 1 or cl[x+1][y] == 1 or cl[x+1][y+1] == 1 or cl[x][y+2] == 1 or cl[x][y+3] == 1 or cl[x+1][y+2] == 1 or cl[x+1][y+3] == 1):
		# 			self.add_layer()
		# 		print("HERE2")
		# 		cl = self.get_layer(self.num_layer)
		# 		cl[x][y+0] = 1
		# 		cl[x][y+1] = 1
		# 		cl[x][y+2] = 1
		# 		cl[x][y+3] = 1
		# 		cl[x+1][y+0] = 1
		# 		cl[x+1][y+1] = 1
		# 		cl[x+1][y+2] = 1
		# 		cl[x+1][y+3] = 1
		# 		# self.set_current_layer(cl)
		# 	else:
		# 		if (cl[x][y] == 1 or cl[x][y+1] == 1 or cl[x+1][y] == 1 or cl[x+1][y+1] == 1 or cl[x+2][y] == 1 or cl[x+3][y] == 1 or cl[x+2][y+1] == 1 or cl[x+3][y+1] == 1):
		# 			self.add_layer()
		# 		cl = self.get_layer(self.num_layer)
		# 		cl[x+0][y] = 1
		# 		cl[x+1][y] = 1
		# 		cl[x+2][y] = 1
		# 		cl[x+3][y] = 1
		# 		cl[x+0][y+1] = 1
		# 		cl[x+1][y+1] = 1
		# 		cl[x+2][y+1] = 1
		# 		cl[x+3][y+1] = 1
		# self.set_current_layer(cl)


