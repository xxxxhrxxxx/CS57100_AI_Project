from mazelib import Maze
from mazelib.generate.Prims import Prims
import matplotlib.pyplot as plt
import numpy as np

def showPNG(grid):
	plt.figure(figsize=(10, 5))
	plt.imshow(grid, cmap='hot', interpolation='nearest')
	plt.xticks([]), plt.yticks([])
	plt.show()



#check all the available slots
def region_growing(grid):
	m, n = grid.shape
	print(grid.shape)
	showPNG(grid)

	new_grid = np.array(grid)
	
	cluster_id = 1
	visited = set()
	for i in range(m):
		for j in range(n):
			if grid[i][j] == 1 and (i,j) not in visited:
				start = [(i,j)]
				visited.add((i,j))
				while start:
					new_start = []
					for temp_i, temp_j in start:
						new_grid[temp_i][temp_j] = 1

						#up
						new_i = max(temp_i - 1, 0)
						if new_i != temp_i:
							if grid[new_i][temp_j] == 1 and (new_i, temp_j) not in visited:
								visited.add((new_i, temp_j))
								new_start.append((new_i, temp_j))

						#down
						new_i = min(temp_i + 1, m-1)
						if new_i != temp_i:
							if grid[new_i][temp_j] == 1 and (new_i, temp_j) not in visited:
								visited.add((new_i, temp_j))
								new_start.append((new_i, temp_j))


						#left
						new_j = max(temp_j - 1, 0)
						if new_j != temp_j:
							if grid[temp_i][new_j] == 1 and (temp_i, new_j) not in visited:
								visited.add((temp_i, new_j))
								new_start.append((temp_i, new_j))


						#right
						new_j = min(temp_j + 1, n-1)
						if new_j != temp_j:
							if grid[temp_i][new_j] == 1 and (temp_i, new_j) not in visited:
								visited.add((temp_i, new_j))
								new_start.append((temp_i, new_j))
					start = new_start
				cluster_id += 1
	print(cluster_id)
	print(len(visited))
	showPNG(new_grid)


	with open('large_maze.npy', 'wb') as f:
		np.save(f, grid)
	f.close()
			


def save_grid(size):
	m = Maze()
	m.generator = Prims(size, size)
	m.generate()
	grid = m.grid
	print(type(grid))

	region_growing(grid)





if __name__ == '__main__':
	save_grid(50)
