# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

q_table = None
with open("q_table_1500.pkl", "rb") as f:
    q_table = pickle.load(f)
visited = []
has_picked_up = False
target_loc = None
destination = None

def get_state(obs, target_loc=None):
	stations = [[0, 0], [0, 4], [4, 0], [4,4]]
	taxi_row, taxi_col, stations[0][0],stations[0][1] ,stations[1][0],stations[1][1],stations[2][0],stations[2][1],stations[3][0],stations[3][1],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
	stations = [tuple(i) for i in stations]	

	assert target_loc is not None
	x_dir = target_loc[0] - taxi_row
	y_dir = target_loc[1] - taxi_col
	x_dir = 0 if x_dir == 0 else x_dir // abs(x_dir)
	y_dir = 0 if y_dir == 0 else y_dir // abs(y_dir)
	return (x_dir, y_dir, obstacle_north, obstacle_south, obstacle_east, obstacle_west, has_picked_up)
  
def get_action(obs):
	"""
	# Selects the best action using the trained Q-table.
	"""
	global q_table
	global visited
	global has_picked_up
	global target_loc
	global destination
 
	stations = [[obs[2], obs[3]], [obs[4], obs[5],], [obs[6], obs[7]], [obs[8], obs[9]]]
	if target_loc is None:
		target_loc = stations[0]
		print (target_loc)
	taxi_row, taxi_col, stations[0][0],stations[0][1] ,stations[1][0],stations[1][1],stations[2][0],stations[2][1],stations[3][0],stations[3][1],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
 
   
	if is_in_station(obs) and (taxi_row, taxi_col) not in visited:
			visited.append((taxi_row, taxi_col))
			if destination_look:
				destination = (taxi_row, taxi_col)
			if has_picked_up and destination is not None:
				target_loc = destination
			else:
				# choose a station that has not been visited yet
				for station in stations:
					if station not in visited:
						target_loc = station
						break	
				# print (visited, (taxi_row, taxi_col), target_loc)
  
	# if get_state(obs, target_loc) not in q_table:
	# 		action_probs = np.ones(6) / 6
	# 		if obstacle_south:
	# 			action_probs[0] = 0
	# 		if obstacle_north:
	# 			action_probs[1] = 0
	# 		if obstacle_east:
	# 			action_probs[2] = 0
	# 		if obstacle_west:
	# 			action_probs[3] = 0	
	# 		if not passenger_look or has_picked_up or not is_in_station(obs):
	# 			action_probs[4] = 0
	# 		if not destination_look or not has_picked_up or not is_in_station(obs):
	# 			action_probs[5] = 0
	# 		action_probs = action_probs / np.sum(action_probs)
	# 		action = np.random.choice(6, p=action_probs)  # Random action
	# else:
	action = np.argmax(q_table[get_state(obs, target_loc)])  # Greedy action
	
	if not has_picked_up and passenger_look and is_in_station(obs) and action == 4:
			has_picked_up = True
	if has_picked_up and destination_look and is_in_station(obs) and action == 5:
			done = True
	
	# action_probs = np.ones(6) / 6
	# if obstacle_south:
	# 	action_probs[0] = 0
	# if obstacle_north:
	# 	action_probs[1] = 0
	# if obstacle_east:
	# 	action_probs[2] = 0
	# if obstacle_west:
	# 	action_probs[3] = 0	
	# if not passenger_look or has_picked_up or not is_in_station(obs):
	# 	action_probs[4] = 0
	# if not destination_look or not has_picked_up or not is_in_station(obs):
	# 	action_probs[5] = 0
	# action_probs = action_probs / np.sum(action_probs)
	# action = np.random.choice(6, p=action_probs)  # Random action
	return action

def is_in_station(obs):
	"""
	# Checks if the taxi is in a station.
	"""
	stations = [[0, 0], [0, 4], [4, 0], [4,4]]
	taxi_row, taxi_col,stations[0][0],stations[0][1] ,stations[1][0],stations[1][1],stations[2][0],stations[2][1],stations[3][0],stations[3][1],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
	stations = [tuple(i) for i in stations]
	return (taxi_row, taxi_col) in stations