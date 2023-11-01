import os
import sys
import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def showPNG(grid):
	plt.figure(figsize=(10, 5))
	plt.imshow(grid, cmap='hot', interpolation='nearest')
	plt.xticks([]), plt.yticks([])
	plt.show()

#uniform cost search for shortest path
def uniform_cost_search(maze, src_loc, stop_loc, graph):
	#basically bfs in the current setting
	start = [[src_loc, []]]
	visited = set()
	visited.add(src_loc)
	graph[src_loc] = {}

	m, n = maze.shape

	while start:
		new_start = []
		for t in range(len(start)):
			temp_start = start[t][0]
			current_path = start[t][1]
			temp_i, temp_j = temp_start

			if (temp_i, temp_j) in stop_loc:
				graph[src_loc][temp_start] = current_path

			search_order = np.random.choice([0, 1, 2, 3], 4, replace = False)
			
			for k in search_order:
				if k == 0:
					new_i = max(temp_i - 1, 0)
					if new_i != temp_i:
						if maze[new_i][temp_j] >= 1 and (new_i, temp_j) not in visited:
							visited.add((new_i, temp_j))
							new_start.append([(new_i, temp_j), current_path[:] + [(new_i, temp_j)]])

				elif k == 1:
					#down
					new_i = min(temp_i + 1, m-1)
					if new_i != temp_i:
						if maze[new_i][temp_j] >= 1 and (new_i, temp_j) not in visited:
							visited.add((new_i, temp_j))
							new_start.append([(new_i, temp_j), current_path[:] + [(new_i, temp_j)]])

				elif k == 2:
					#left
					new_j = max(temp_j - 1, 0)
					if new_j != temp_j:
						if maze[temp_i][new_j] >= 1 and (temp_i, new_j) not in visited:
							visited.add((temp_i, new_j))
							new_start.append([(temp_i, new_j), current_path[:] + [(temp_i, new_j)]])

				else:
					#right
					new_j = min(temp_j + 1, n-1)
					if new_j != temp_j:
						if maze[temp_i][new_j] >= 1 and (temp_i, new_j) not in visited:
							visited.add((temp_i, new_j))
							new_start.append([(temp_i, new_j), current_path[:] + [(temp_i, new_j)]])

		start = new_start


#extract high-level graph
def graph_extraction(maze, stop_loc):
	graph = {}

	for i in range(len(stop_loc)):
		src_loc = stop_loc[i]
		remain_loc = stop_loc[:i] + stop_loc[i+1:]
		uniform_cost_search(maze, src_loc, remain_loc, graph)

	print(len(graph[stop_loc[0]]))
	

	#check path
	for target in graph[stop_loc[0]]:
		vis_maze = np.array(maze)
		path = graph[stop_loc[0]][target]
		for temp_i, temp_j in path[:-1]:
			vis_maze[temp_i][temp_j] = 3
	
		#showPNG(vis_maze)

	random_baseline_path = random_baseline_solver(graph, stop_loc)

	path_visualizer(maze, random_baseline_path, graph)
	
def random_baseline_solver(graph, stop_loc):
	plan = [stop_loc[0]]
	remain_stop = stop_loc[1:]
	plan_index = np.random.choice([_ for _ in range(len(remain_stop))], len(remain_stop), replace = False)
	for index in plan_index:
		plan.append(remain_stop[index])
	plan.append(stop_loc[0])
	return plan

def path_visualizer(maze, path, graph):
	
	#plt.figure(figsize=(10, 5))
	#plt.imshow(grid, cmap='hot', interpolation='nearest')
	#plt.xticks([]), plt.yticks([])
	#plt.show()

	detailed_path = []
	path_length_index = [0]
	path_length = [0]
	accu = 0
	for i in range(len(path)-1):
		src_loc = path[i]
		tar_loc = path[i+1]
		detailed_path += graph[src_loc][tar_loc]
		accu += len(graph[src_loc][tar_loc])
		path_length.append(accu)

	delta_color = [10/(len(path_length) + 5)]
	print(delta_color)

	fig = plt.figure(figsize=(10, 10))
	im = plt.imshow(maze, cmap='hot', interpolation='nearest')

	fps = 30
	nSeconds = 500
	
	vis_maze = np.array(maze)
	current_color = delta_color[:]

	def animate_func(i):


		if i >= len(detailed_path):
			im.set_array(vis_maze)
		else:
			temp_i, temp_j = detailed_path[i]
			if (temp_i, temp_j) not in path:
				vis_maze[temp_i][temp_j] = 3


		im.set_array(vis_maze)

	anim = animation.FuncAnimation(fig, animate_func, frames = nSeconds * fps, interval = 10/fps)
	
	plt.show()




def load_maze(maze_file_path):
	maze = None
	with open(maze_file_path, 'rb') as f:
		maze = np.load(f)

	stop_count = 0
	if 'large' in maze_file_path:
		stop_count = 40
	elif 'medium' in maze_file_path:
		stop_count = 20
	else:
		stop_count = 10

	m, n = maze.shape
	free_space = []

	for i in range(m):
		for j in range(n):
			if maze[i][j] == 1:
				free_space.append((i,j))

	#randomly generate middle stops
	stops = np.random.choice([_ for _ in range(len(free_space))], stop_count+1, replace = False)
	stop_loc = []
	for index in stops:
		temp_i, temp_j = free_space[index]
		maze[temp_i][temp_j] = 5
		stop_loc.append((temp_i, temp_j))
	
	maze[stop_loc[0][0]][stop_loc[0][1]] = 10
	showPNG(maze)

	graph_extraction(maze, stop_loc)	

if __name__ == '__main__':
	maze_file_path = 'maze_envs/large_maze.npy'

	load_maze(maze_file_path)
	
