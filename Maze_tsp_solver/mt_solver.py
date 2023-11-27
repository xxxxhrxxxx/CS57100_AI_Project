import os
import sys
import math
import bisect
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def showPNG(grid):
	plt.figure(figsize=(10, 5))
	plt.imshow(grid, cmap='hot', interpolation='nearest')
	plt.xticks([]), plt.yticks([])
	plt.show()

def randomize_maze_weight(maze):
	m, n = maze.shape
	for i in range(m):
		for j in range(n):
			if maze[i][j] >= 1:
				maze[i][j] = random.randint(1, 10)

def uniform_cost_search_weight(maze, src_loc, stop_loc, graph):
	start = [[src_loc, []]]
	cost_arr = [0]
	visited = set()
	graph[src_loc] = {}

	m, n = maze.shape

	target_size = len(stop_loc)

	while start:
		temp_start, current_path = start.pop(0)
		temp_i, temp_j = temp_start
		current_cost = cost_arr.pop(0)
		visited.add(temp_start)
			
		if temp_start in stop_loc:
			graph[src_loc][temp_start] = [current_path, current_cost]
			if len(graph[src_loc]) == target_size:
				break

		search_order = np.random.choice([0, 1, 2, 3], 4, replace = False)

		for k in search_order:
			#up
			if k == 0:
				new_i = max(temp_i - 1, 0)
				if new_i != temp_i:
					if maze[new_i][temp_j] >= 1 and (new_i, temp_j) not in visited:
						new_cost = current_cost + maze[new_i][temp_j]
						temp_index = bisect.bisect_right(cost_arr, new_cost)
						cost_arr.insert(temp_index, new_cost)
						start.insert(temp_index, [(new_i, temp_j), current_path[:] + [(new_i, temp_j)]])
			#down
			elif k == 1:
				new_i = min(temp_i + 1, m - 1)
				if new_i != temp_i:
					if maze[new_i][temp_j] >= 1 and (new_i, temp_j) not in visited:
						new_cost = current_cost + maze[new_i][temp_j]
						temp_index = bisect.bisect_right(cost_arr, new_cost)
						cost_arr.insert(temp_index, new_cost)
						start.insert(temp_index, [(new_i, temp_j), current_path[:] + [(new_i, temp_j)]])
			#left
			elif k == 2:
				new_j = max(temp_j - 1, 0)
				if new_j != temp_j:
					if maze[temp_i][new_j] >= 1 and (temp_i, new_j) not in visited:
						new_cost = current_cost + maze[temp_i][new_j]
						temp_index = bisect.bisect_right(cost_arr, new_cost)
						cost_arr.insert(temp_index, new_cost)
						start.insert(temp_index, [(temp_i, new_j), current_path[:] + [(temp_i, new_j)]])
			#right
			else:
				new_j = min(temp_j + 1, n-1)
				if new_j != temp_j:
					if maze[temp_i][new_j] >= 1 and (temp_i, new_j) not in visited:
						new_cost = current_cost + maze[temp_i][new_j]
						temp_index = bisect.bisect_right(cost_arr, new_cost)
						cost_arr.insert(temp_index, new_cost)
						start.insert(temp_index, [(temp_i, new_j), current_path[:] + [(temp_i, new_j)]])
		

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
				graph[src_loc][temp_start] = [current_path, len(current_path)]

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

def graph_extraction_test2(maze, stop_loc):
	#test2: different cost at different loc + can reuse previous loc
	graph = {}

	randomize_maze_weight(maze)

	for i in range(len(stop_loc)):
		src_loc = stop_loc[i]
		remain_loc = stop_loc[:i] + stop_loc[i+1:]
		uniform_cost_search_weight(maze, src_loc, remain_loc, graph)

	#check path
	for target in graph[stop_loc[0]]:
		vis_maze = np.array(maze)
		path = graph[stop_loc[0]][target]
		for temp_i, temp_j in path[0][:-1]:
			vis_maze[temp_i][temp_j] = 3
	
		#showPNG(vis_maze)

	#start random baseline*******************************************************
	random_baseline_path, random_cost = random_baseline_solver(graph, stop_loc)

	#path_visualizer(maze, random_baseline_path, graph)

	print('Total cost of random policy is: {0}'.format(random_cost))
	#end random baseline*********************************************************

	#start greedy baseline*******************************************************
	greedy_baseline_path, greedy_cost = greedy_baseline_solver(graph, stop_loc)

	#path_visualizer(maze, greedy_baseline_path)

	print(greedy_baseline_path)
	print(len(greedy_baseline_path))
	
	print('Total cost of greedy policy is: {0}'.format(greedy_cost))
	#end greedy baseline*********************************************************
	
	#MCTS_path, MCTS_cost = MCTS_solver(graph, stop_loc)

	#path_visualizer(maze, MCTS_path)

	#print(MCTS_path)
	#print(len(MCTS_path))
	
	#print('Total cost of MCTS policy is: {0}'.format(MCTS_cost))



#extract high-level graph
def graph_extraction_test1(maze, stop_loc):
	#test1: uniform cost + can reuse previous loc
	graph = {}

	for i in range(len(stop_loc)):
		src_loc = stop_loc[i]
		remain_loc = stop_loc[:i] + stop_loc[i+1:]
		uniform_cost_search(maze, src_loc, remain_loc, graph)

	#check path
	for target in graph[stop_loc[0]]:
		vis_maze = np.array(maze)
		path = graph[stop_loc[0]][target]
		for temp_i, temp_j in path[0][:-1]:
			vis_maze[temp_i][temp_j] = 3
	
		#showPNG(vis_maze)

	#start random baseline*******************************************************
	random_baseline_path, random_cost = random_baseline_solver(graph, stop_loc)

	#path_visualizer(maze, random_baseline_path, graph)

	print('Total cost of random policy is: {0}'.format(random_cost))
	#end random baseline*********************************************************

	#start greedy baseline*******************************************************
	greedy_baseline_path, greedy_cost = greedy_baseline_solver(graph, stop_loc)

	#path_visualizer(maze, greedy_baseline_path)

	print(greedy_baseline_path)
	print(len(greedy_baseline_path))
	
	print('Total cost of greedy policy is: {0}'.format(greedy_cost))
	#end greedy baseline*********************************************************
	
	MCTS_path, MCTS_cost = MCTS_solver(graph, stop_loc)

	#path_visualizer(maze, MCTS_path)

	print(MCTS_path)
	print(len(MCTS_path))
	
	print('Total cost of MCTS policy is: {0}'.format(MCTS_cost))


class MCTS_treenode:
	def __init__(self, path_loc, remain_loc, graph, current_cost):
		#visited and stop_loc are lists
		self.path_loc_ = path_loc
		self.remain_loc_ = remain_loc
		self.reward_ = 0.0
		self.count_ = 0.0
		self.graph_ = graph
		self.current_ = path_loc[-1]
		self.visited_ = set(path_loc)
		self.children_ = []
		self.parent_ = None
		self.current_cost_ = current_cost

	def gen_next_loc(self):
		#for all loc in remain, get the average distance
		#first try to pure strategy
		max_expansion = 1
		distance_arr = []

		#add a nearest heursitic
		tracker = {}

		alpha = 0.1

		for loc in self.remain_loc_:
			total_distance = 0.0
			total_count = 0.0
			tracker[loc] = len(self.graph_[self.current_][loc])
			for key,value in self.graph_[loc].items():
				if key not in self.visited_:
					total_distance += len(value)
					total_count += 1.0
			if total_count != 0:
				tracker[loc] += -alpha * total_distance * 1.0/total_count
				tracker[loc] += 0
			else:
				tracker[loc] += sys.maxsize
		
		distance_arr = []
		for key, value in tracker.items():
			distance_arr.append((value, key))
		#distance_arr.sort(key = lambda x : x[0], reverse = True)
		distance_arr.sort(key = lambda x : x[0], reverse = False)
		res = [x[1] for x in distance_arr[:max_expansion]]
		random.shuffle(res)
		return res

	def get_ucb_value(self):
		
		if self.count_ == 0.0:
			return sys.maxsize
		else:
			if not self.parent_:
				print('Something wrong in tree structure, cannot get parent for ucb calculation!')
				sys.exit(1)
			hyper_c = 2
			ucb = self.reward_/self.count_ + hyper_c * math.sqrt(math.log(self.parent_.count_)*1.0/self.count_)
			return ucb


	def add_child(self, child):
		self.children_.append(child)

	def set_parent(self, parent):
		self.parent_ = parent

	def finished(self):
		return len(self.remain_loc_) == 0
		

def MCTS_selection(current_node):
	start_node = current_node
	while True:
		if len(start_node.children_) == 0:
			break
		else:
			max_ucb = -sys.maxsize
			chosen_node = None
			for child in start_node.children_:
				child_ucb = child.get_ucb_value()
				if child_ucb > max_ucb:
					max_ucb = child_ucb
					chosen_node = child
			start_node = chosen_node
	return start_node
		
def MCTS_expansion(current_node, graph):

	current_path = current_node.path_loc_
	current_cost = current_node.current_cost_

	stop_loc_set = set(current_node.remain_loc_)

	next_candidates = current_node.gen_next_loc()
	for can in next_candidates:
		if can not in stop_loc_set:
			print('next candidates cannot be found in the expansion module!')
			sys.exit(1)
		stop_loc_set.remove(can)
		updated_cost = current_cost + len(graph[current_node.current_][can])
		new_treenode = MCTS_treenode(current_path + [can], list(stop_loc_set), graph, updated_cost)
		stop_loc_set.add(can)
		
		#update tree structure
		new_treenode.set_parent(current_node)
		current_node.add_child(new_treenode)

	if current_node.children_: return current_node.children_[0]
	else: return current_node

def MCTS_rollout(current_node, graph):	
	#monte carlo rollout until termination
	start_node = current_node	
	while not start_node.finished():
		current_path = start_node.path_loc_
		current_cost = start_node.current_cost_
		stop_loc_set = set(start_node.remain_loc_)		

		next_can = start_node.gen_next_loc()[0]
		stop_loc_set.remove(next_can)
		update_cost = current_cost + len(graph[start_node.current_][next_can])
		new_treenode = MCTS_treenode(current_path + [next_can], list(stop_loc_set), graph, update_cost)
	
		#update tree structure
		new_treenode.set_parent(start_node)
		start_node.add_child(new_treenode)
		start_node = new_treenode
	return start_node

def MCTS_backprop(current_node, graph, start_node):
	current_cost = current_node.current_cost_
	current_cost += len(graph[current_node.current_][start_node])
	while current_node:
		current_node.reward_ += - current_cost
		current_node.count_ += 1
		current_node = current_node.parent_

def tree_back_traversal(final_node):
	path = []
	while final_node:
		path.append(final_node.current_)
		final_node = final_node.parent_
	return path
	
def MCTS_solver(graph, stop_loc):
	root = MCTS_treenode([stop_loc[0]], stop_loc[1:], graph, 0.0)

	#stop_loc_set = set(stop_loc)

	final_path = []
	final_cost = sys.maxsize
	res_count = 0
	while True:
		selected_node = MCTS_selection(root)
		#print(selected_node.reward_, selected_node.count_)
		if selected_node.finished():
			final_node = selected_node
			
			temp_path = tree_back_traversal(final_node)
			temp_path = temp_path[::-1] + [stop_loc[0]]
			temp_cost = 0.0
			
			for t in range(len(temp_path)-1):
				src, tar = temp_path[t], temp_path[t+1]
				temp_cost += len(graph[src][tar])
			
			if temp_cost < final_cost:
				final_cost = temp_cost
				final_path = temp_path

			res_count += 1
			if res_count > 5: break
		rollout_node = selected_node
		if selected_node.count_ != 0.0:
			rollout_node = MCTS_expansion(selected_node, graph)
		finish_node = MCTS_rollout(rollout_node, graph)
		MCTS_backprop(finish_node, graph, stop_loc[0])
		rollout_node.children_ = []

	return final_path, final_cost
	
	




def greedy_baseline_solver(graph, stop_loc):
	plan = [stop_loc[0]]
	visited = set()
	visited.add(stop_loc[0])
	current = stop_loc[0]


	while len(visited) != len(stop_loc):
		nearest_cost = sys.maxsize
		nearest_node = None
		for key, value in graph[current].items():
			if key not in visited:
				if value[1] < nearest_cost:
					nearest_cost = value[1]
					nearest_node = key
		current = nearest_node
		plan.append(nearest_node)
		visited.add(nearest_node)
	
	plan.append(plan[0])
	cost = 0
	for i in range(len(plan)-1):
		src = plan[i]
		tar = plan[i+1]
		
		cost += graph[src][tar][1]
	
	return plan, cost
	

	

def random_baseline_solver(graph, stop_loc):
	plan = [stop_loc[0]]
	remain_stop = stop_loc[1:]
	plan_index = np.random.choice([_ for _ in range(len(remain_stop))], len(remain_stop), replace = False)
	for index in plan_index:
		plan.append(remain_stop[index])
	plan.append(stop_loc[0])

	cost = 0
	for i in range(len(plan)-1):
		src = plan[i]
		tar = plan[i+1]

		cost += graph[src][tar][1]

	return plan, cost

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
	#case1: uniform cost + can reuse previous loc
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
	#showPNG(maze)

	#graph_extraction_test1(maze, stop_loc)	
	graph_extraction_test2(maze, stop_loc)	

if __name__ == '__main__':
	maze_file_path = 'maze_envs/large_maze.npy'

	load_maze(maze_file_path)

	
	
