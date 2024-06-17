from itertools import product
import time
import random
import math
from copy import deepcopy
import numpy as np
from collections import deque

ids = [""]

RESET_PENALTY = 2
DROP_IN_DESTINATION_REWARD = 4
INIT_TIME_LIMIT = 300
TURN_TIME_LIMIT = 0.1
MARINE_COLLISION_PENALTY = 1

class OptimalPirateAgent:
    def __init__(self, initial):
        self.start_time = time.time()
        self.initial = initial
        """
        each state will be built from a tuple of 3 tuples:
        1. tuple of tuples of pirate ships: ((x,y), capacity)
        2. tuple of marine ships indicies: idx
        3. tuple of treasures locations: (x,y)
        for example: (((0,0,2),(1,1,1)), (0,0), ((0,2),(1,2)))
        therefor we will hold a dict for indices of each type of object
        """
        # problem info
        self.horizontal_bound = len(initial["map"][0])
        self.vertical_bound = len(initial["map"])
        self.problem_map = initial["map"]
        self.base = next(iter(initial["pirate_ships"].values()))["location"] # Assuming all pirate ships start at the same location
        self.num_of_turns = initial["turns to go"]

        # indecies for state and action generation
        self.pirate_idx = {i: pirate for i, pirate in enumerate(initial["pirate_ships"].keys())}
        self.marine_idx = {i: {"name":marine, "path":initial["marine_ships"][marine]["path"]}
                              for i, marine in enumerate(initial["marine_ships"].keys())}
        self.treasure_idx = {i: {"name":treasure,
                                 "possible_locations":initial["treasures"][treasure]["possible_locations"],
                                 "prob_change_location":initial["treasures"][treasure]["prob_change_location"]}
                             for i, treasure in enumerate(initial["treasures"].keys())}

        # MDP params for VI
        self.start_state = create_state(initial)
        self.states = construct_state_space(self, initial)
        self.num_states = len(self.states)
        self.values = {t: {state: 0 for state in self.states} for t in range(self.num_of_turns + 1)}
        self.policy = {t: {state: None for state in self.states} for t in range(self.num_of_turns)}
        # self.values = {state: {t: 0}    for state in self.states for t in range(self.num_of_turns + 1)}
        # self.policy = {state: {t: None} for state in self.states for t in range(self.num_of_turns)}
        # self.values = {t: {} for t in range(self.num_of_turns + 1)}
        # self.policy = {t: {} for t in range(self.num_of_turns)}
        # self.states = {self.start_state}
        # self.values[self.num_of_turns][self.start_state] = 0
        
        self.value_iteration()
        # convinient for debugging
        print()
        

    def value_iteration(self):
        for t in range(self.num_of_turns - 1, -1, -1):
            if time.time() - self.start_time >= INIT_TIME_LIMIT - 1:
                return
            for s in self.states:
                if time.time() - self.start_time >= INIT_TIME_LIMIT - 1:
                    return
                best_action_value = float('-inf')
                best_action = None
                for a in possible_actions(self, s):
                    expected_value = 0
                    next_states, probabilities = transition_function(self, s, a)
                    for next_s, p in zip(next_states, probabilities):
                        reward = reward_function(self, a, s, next_s)
                        expected_value += p * (reward + self.values[t + 1][next_s])
                    if expected_value > best_action_value:
                        best_action_value = expected_value
                        best_action = a
                self.values[t][s] = best_action_value if best_action is not None else 0
                self.policy[t][s] = best_action


    def choose_action(self, state, step):
        start = time.time()
        actions = possible_actions(self, state)
        marine_movement = marine_actions(state[1], self.marine_idx)
        treasure_movement = treasure_actions(state[2], self.treasure_idx)
        
        best_value = float('-inf')
        best_state = None

        for action in actions:
            if time.time() - start >= 0.01:
                return actions[random.randint(0, len(actions) - 1)]
            pirate_move = pirate_movements(self.pirate_idx, state, action)
            states = list(product(pirate_move, marine_movement, treasure_movement))
            for next_state in states:
                if time.time() - start >= 0.01:
                    return actions[random.randint(0, len(actions) - 1)]
                value = self.values[next_state][step + 1]
                if value > best_value:
                    best_value = value
                    best_state = next_state
        
        if best_value == float('-inf'):
            return actions[random.randint(0, len(actions) - 1)]
        else:
            return best_state
        


    def act(self, state_dict):
        state = create_state(state_dict)
        step = self.num_of_turns - state_dict["turns to go"]

        if step < 0 or step > self.num_of_turns:
            print("step out of bounds - choosing random action")
            return self.choose_action(state, step)
        
        if state not in self.states:
            print("state not indexed - choosing random action")
            return self.choose_action(state, step)
        
        value = self.values[step][state]
        if value == float('-inf'):
            print("choosing random action")
            return self.choose_action(state, step)
        
        if value < 0:
            return ("terminate")
        
        action = self.policy[step][state]

        return action

 
######################################################

class InfinitePirateAgent:
    def __init__(self, initial, gamma):
        self.initial = initial
        self.gamma = gamma
        self.epsilon = 0.01
        self.time_limit = time.time() + 299
        self.problem_map = initial["map"]

        self.base = next(iter(initial["pirate_ships"].values()))["location"] # Assuming all pirate ships start at the same location
        self.horizontal_bound = len(initial["map"][0])
        self.vertical_bound = len(initial["map"])



        self.pirate_idx = {i: pirate for i, pirate in enumerate(initial["pirate_ships"].keys())}
        self.marine_idx = {i: {"name":marine, "path":initial["marine_ships"][marine]["path"]}
                                for i, marine in enumerate(initial["marine_ships"].keys())}
        self.treasure_idx = {i: {"name":treasure,
                                 "possible_locations":initial["treasures"][treasure]["possible_locations"],
                                 "prob_change_location":initial["treasures"][treasure]["prob_change_location"]}
                             for i, treasure in enumerate(initial["treasures"].keys())}
        
        self.states = construct_state_space(self, initial)
        self.num_states = len(self.states)

        self.values = {s: 0 for s in self.states}
        self.policy = {s: None for s in self.states}

        self.value_iteration()


    def value_iteration(self):
        while True:
            delta = 0
            for s in self.states:
                if time.time() >= self.time_limit:
                    return

                best_action_value = float('-inf')
                best_action = None

                for a in possible_actions(self, s):
                    expected_value = 0
                    next_states, probabilities = transition_function(self, s, a)

                    for next_s, p in zip(next_states, probabilities):
                        reward = reward_function(self, a, s, next_s, real_reward=True)
                        expected_value += p * (reward + self.gamma * self.values[next_s])

                    if expected_value > best_action_value:
                        best_action_value = expected_value
                        best_action = a

                delta = max(delta, abs(best_action_value - self.values[s]))
                self.values[s] = best_action_value
                self.policy[s] = best_action

            # Check for epsilon convergence
            if delta < self.epsilon * (1 - self.gamma) / self.gamma:
                break


    def act(self, state_dict):
        state = create_state(state_dict)
        if state not in self.states:
            return ("terminate")

        return self.policy[state]

    def value(self, state):
        return self.values[create_state(state)]
    

######################################################


def create_state(state_dict):
    pirate_state = []
    marine_state = []
    treasure_state = []
    for _, value in state_dict["pirate_ships"].items():
        pirate_state.append((value["location"], value["capacity"]))
    
    for _, value in state_dict["marine_ships"].items():
        marine_state.append(value["index"])

    for _, value in state_dict["treasures"].items():
        treasure_state.append(value["location"])

    return (tuple(pirate_state), tuple(marine_state), tuple(treasure_state))
    


def construct_state_space(agent, state_dict):
    map = state_dict["map"]
    len_pirate_ships = len(state_dict["pirate_ships"])
    len_treasures = len(state_dict["treasures"])
    len_marine_ships = len(state_dict["marine_ships"])

    state_space = []

    pirate_states = [[]] * len_pirate_ships
    for idx, _ in agent.pirate_idx.items():
        pirate_locations = []
        for x in range(len(map)):
            for y in range(len(map[0])):
                if map[x][y] != 'I':
                    for capacity in range(3):
                        pirate_locations.append(((x, y), capacity))
        pirate_states[idx] = pirate_locations
        
    
    pirate_loc_combinations = list(product(*pirate_states))
    
    marine_locations = [[]] * len_marine_ships
    for idx, marine_ship in agent.marine_idx.items():
        marine_locations[idx] = [i for i in range(len(marine_ship["path"]))]

    marine_loc_combinations = list(product(*marine_locations))


    treasure_locations = [[]] * len_treasures
    for idx, treasure in agent.treasure_idx.items():
        treasure_locations[idx] = [location for location in treasure["possible_locations"]]
    
    treasure_loc_combinations = list(product(*treasure_locations))

    for pirate_combination in pirate_loc_combinations:
        for marine_combination in marine_loc_combinations:
            for treasure_combination in treasure_loc_combinations:
                state_space.append((pirate_combination, marine_combination, treasure_combination))

    return state_space


def reward_function(agent, action, state, next_state, real_reward=False):
    if action is str and action == "reset":
        return -2

    reward = 0
    pirates = {agent.pirate_idx[idx]: tup for idx, tup in enumerate(state[0])}
    for atomic_action in action:
        if atomic_action[0] == "deposit":
            capacity = pirates[atomic_action[1]][1]
            reward += 4 * (2 - capacity)

    reward -= marine_encounters(next_state, agent, real_reward)

    return reward

# forum says that only one point goes per ship if encounters a marine ship
# meaning if multiple marines are at the same location with a pirate only 1 is deducted
def marine_encounters(state, agent, real_reward=False):
    marine_ships = {agent.marine_idx[marine_idx]["path"][path_idx]: 0 for marine_idx, path_idx in enumerate(state[1])}
    next_pirates = {agent.pirate_idx[idx]: tup for idx, tup in enumerate(state[0])}
    marine_encounters = 0
    for ship_tup in next_pirates.values():
        if ship_tup[0] in marine_ships:
            # marine_encounters += 1
            # while we in practice only lose 1 point, losing treasures is more severe
            marine_encounters += 1 + (2 - ship_tup[1]) * 4 if not real_reward else 1
            # marine_encounters += 1 
    return marine_encounters



def possible_states(agent, state):
    actions = possible_actions(agent, state)
    dic = {}
    for action in actions:
        pirate_movement = pirate_movements(agent.pirate_idx, state, action)
        dic[action] = list(product(pirate_movement, marine_movement, treasure_movement))
    marine_movement = marine_actions(state[1], agent.marine_idx)
    treasure_movement = treasure_actions(state[2], agent.treasure_idx)
    return dic

# returns 2 lists, one of the next states and one of the probabilities
def transition_function(agent, state, action):
    if action == "reset":
        # When resetting, always go back to the initial state with one fewer turns.
        reset_state = create_state(agent.initial)
        return [reset_state], [1]
    
    curr_marines = state[1]
    curr_treasures = state[2]
    next_states = []
    probabilities = []

    possible_pirate_movements = pirate_movements(agent.pirate_idx, state, action)
    possible_marine_movements, marine_prob = marine_movements(curr_marines, agent.marine_idx)
    possible_treasure_movements, treasure_probs = treasure_movements_prob(curr_treasures, agent.treasure_idx)

    for pirate_state in possible_pirate_movements:
        for marine_state in possible_marine_movements:
            for i, treasure_state in enumerate(possible_treasure_movements):
                next_state = (pirate_state, marine_state, treasure_state)
                next_states.append(next_state)
                probabilities.append(marine_prob * treasure_probs[i])

    return next_states, probabilities



def treasure_movements_prob(curr_treasures, treasure_idx_dict):
    treasure_tuples = [[]] * len(curr_treasures)

    for treasure_idx, location in enumerate(curr_treasures):
        treasure_prob = []
        possible_locations = treasure_idx_dict[treasure_idx]["possible_locations"]
        prob_change = treasure_idx_dict[treasure_idx]["prob_change_location"]
        treasure_prob.append((location, 1-prob_change + prob_change * (1/len(possible_locations))))
        for loc in possible_locations:
            if loc != location:
                treasure_prob.append((loc, prob_change * (1/len(possible_locations))))
        treasure_tuples[treasure_idx] = treasure_prob

    combinations = list(product(*treasure_tuples))
    probabilities = []
    treasure_combinations = []

    for combination in combinations:
        probability = 1
        comb = []
        for tup in combination:
            probability *= tup[1]
            comb.append(tup[0])

        probabilities.append(probability)
        treasure_combinations.append(tuple(comb))

    return treasure_combinations, probabilities


def marine_movements(curr_marines, marine_dict):
    probability = 1
    new_marine_indecies = [[]] * len(curr_marines)

    for marine_idx, path_idx in enumerate(curr_marines):
        new_marine_indecies[marine_idx] = possible_marine_movements(marine_dict[marine_idx]["path"], path_idx)
        probability *= 1/len(new_marine_indecies[marine_idx])

    combinations = list(product(*new_marine_indecies))
    return combinations, probability # the same probability for all combinations


def possible_marine_movements(path, idx):
    if len(path) == 1:
        return [idx]
    if idx < len(path) - 1 and idx > 0:
        return [idx - 1, idx, idx + 1]
    elif idx == 0:
        return [idx, idx + 1]
    else: # idx == len(path) - 1
        return [idx - 1, idx]


def pirate_movements(pirate_idx, state, action):
    pirates = {pirate_idx[idx]: (idx, tup) for idx, tup in enumerate(state[0])}
    new_state = [()] * len(action)
    for atomic_action in action:
        capacity = pirates[atomic_action[1]][1][1]
        location = pirates[atomic_action[1]][1][0]
        index = pirates[atomic_action[1]][0]
        if atomic_action[0] == "sail":
            new_state[index] = (atomic_action[2], capacity)
        elif atomic_action[0] == "collect":
            new_state[index] = (location, capacity - 1)
        elif atomic_action[0] == "deposit":
            new_state[index] = (location, 2)
        elif atomic_action[0] == "wait":
            new_state[index] = (location, capacity)
    return [tuple(new_state)]


def possible_actions(agent, state):
    actions_list = []

    pirate_ships = state[0]
    treasures = state[2]

    for idx, (location, capacity) in enumerate(pirate_ships): # check for possible moves
        actions = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (location[0] + dx, location[1] + dy)
            if is_valid_move(agent, new_pos):
                actions.append(('sail', agent.pirate_idx[idx], new_pos))

            if new_pos in treasures and capacity > 0: #check to collect treasure action
                for treasure_idx, loc in enumerate(treasures):
                    if loc == new_pos:
                        actions.append(('collect', agent.pirate_idx[idx], agent.treasure_idx[treasure_idx]["name"]))
                
        if capacity < 2:
            if location == agent.base:
                actions.append(('deposit', agent.pirate_idx[idx]))
        
        actions.append(('wait', agent.pirate_idx[idx])) #adding 'wait' action for each ship
        actions_list.append(actions)
    
    combinations_list = list(product(*actions_list)) + ["reset"]
                    
    return combinations_list


def marine_actions(marine_state, marine_dict):
    marine_actions = []
    for idx, path_idx in enumerate(marine_state):
        marine_actions.append(possible_marine_movements(marine_dict[idx]["path"], path_idx))
    return list(product(*marine_actions))

def treasure_actions(treasure_state, treasure_dict):
    treasure_actions = []
    for idx, location in enumerate(treasure_state):
        treasure_actions.append(treasure_dict[idx]["possible_locations"])
    return list(product(*treasure_actions))
    

def is_valid_move(agent, pos):
    """Check if the move position is valid (not an island)"""
    x, y = pos
    if not (0 <= x < agent.vertical_bound and 0 <= y < agent.horizontal_bound):
        return False  # Out of bounds
    if agent.problem_map[x][y] == 'I':
        return False  # Island
    return True



def find_base_location(map_layout):
        for y, row in enumerate(map_layout):
            for x, cell in enumerate(row):
                if cell == 'B':
                    return (y, x)
        return None

def print_mytest(agent):
    lst = [[[0]*agent.horizontal_bound ] * agent.vertical_bound] * 3
    for state, value in agent.value_table.items():
        capacity = state[0][0][1]
        x = state[0][0][0][0]
        y = state[0][0][0][1]
        lst[capacity][x][y] = value

    for i in range(3):
        print("______________________")
        for row in lst[i]:
            print(row)

class PirateAgent:
    def __init__(self, initial):
        self.start_time = time.time()
        self.act_start_time = time.time()
        self.act_time_limit = TURN_TIME_LIMIT - 0.01
        self.initial = initial
        self.base = next(iter(initial["pirate_ships"].values()))["location"]
        self.problem_map = initial["map"]
        self.distances = init_distances(self)

        self.relaxed_init, _ = self.relax_state(self.initial)
        self.max_distance = len(self.problem_map) + len(self.problem_map[0])
        self.horizontal_bound = len(initial["map"][0])
        self.vertical_bound = len(initial["map"])

        self.num_of_turns = initial["turns to go"]
        self.gamma = 1
        self.my_reward = 0
        self.pirate_idx = {i: pirate for i, pirate in enumerate(initial["pirate_ships"].keys())}
        self.marine_idx = {i: {"name":marine, "path":initial["marine_ships"][marine]["path"]}
                              for i, marine in enumerate(initial["marine_ships"].keys())}
        self.treasure_idx = {i: {"name":treasure,
                                 "possible_locations":initial["treasures"][treasure]["possible_locations"],
                                 "prob_change_location":initial["treasures"][treasure]["prob_change_location"]}
                             for i, treasure in enumerate(initial["treasures"].keys())}
        self.idx_trerasure = {treasure: i for i, treasure in enumerate(initial["treasures"].keys())}

        self.policy = {}
        self.values = {}
        self.fall_back_policy = {}


    def create_marine_combinations(self, state_dict):
        len_marine_ships = len(state_dict["marine_ships"])
        marine_locations = [[]] * len_marine_ships
        for idx, marine_ship in self.marine_idx.items():
            marine_locations[idx] = [i for i in range(len(marine_ship["path"]))]

        self.marine_loc_combinations = list(product(*marine_locations))

    def create_treasure_combinations(self, state_dict):
        len_treasures = len(state_dict["treasures"])
        treasure_locations = [[]] * len_treasures
        for idx, treasure in self.treasure_idx.items():
            treasure_locations[idx] = [location for location in treasure["possible_locations"]]
        
        self.treasure_loc_combinations = list(product(*treasure_locations))

    
    
    def reward_function(self, action, state, next_state, real_reward=False):
        if action is str and action == "reset":
            return -2

        reward = 0
        pirates = {self.pirate_idx[idx]: tup for idx, tup in enumerate(state[0])}
        treasure_loc = next_state[2][0]
        pirate_loc = next_state[0][0][0]
        new_capacity = next_state[0][0][1]

        for marine_idx, path_idx in enumerate(state[1]):
            if path_idx == -1:
                continue
            marine_loc = self.marine_idx[marine_idx]["path"][path_idx]
            dis = self.distances[marine_loc][pirate_loc[0]][pirate_loc[1]]
            if dis == 0 or pirate_loc in self.marine_idx[marine_idx]["path"]:
                reward -= (1 + (2 - new_capacity) * 4) / (dis + 1)
            if self.my_reward + reward < 0:
                return reward

        for atomic_action in action:
            if atomic_action[0] == "deposit":
                capacity = pirates[atomic_action[1]][1]
                reward += 4 * (2 - capacity)
        
        for atomic_action in action:
            if atomic_action[0] == "sail" or atomic_action[0] == "collect":
                if new_capacity >= 1:
                    reward += 8 / (self.distances[treasure_loc][self.base[0]][self.base[1]] + new_capacity
                                   + self.distances[treasure_loc][pirate_loc[0]][pirate_loc[1]] + 1)
                else:
                    reward += 8 / (self.distances[self.base][pirate_loc[0]][pirate_loc[1]] + 1)

        return reward
    

    def value_for_state(self, state, steps):
        if state in self.values[steps]:
            return self.values[steps][state]
        if steps <= 0:
            return 0
        if time.time() - self.act_start_time > self.act_time_limit:
            raise TimeLimitExceededException("Time limit exceeded during value iteration")

        max_val = -float('inf')
        for action in possible_actions(self, state):
            next_states, probabilities = self.relaxed_transition(state, action)
            if time.time() - self.act_start_time > self.act_time_limit:
                raise TimeLimitExceededException("Time limit exceeded during value iteration")
            expected_value = 0
            for next_state, prob in zip(next_states, probabilities):
                if time.time() - self.act_start_time > self.act_time_limit:
                    raise TimeLimitExceededException("Time limit exceeded during value iteration")
                expected_value += prob * (self.reward_function(action, state, next_state) + self.gamma * self.value_for_state(next_state, steps - 1))
            if expected_value > max_val:
                max_val = expected_value
                self.policy[steps][state] = action
            
            self.values[steps][state] = max_val

        return max_val

        
    def relax_state(self, state_dict):
        pirate = next(iter(state_dict["pirate_ships"].values()))
        pirate_state = (pirate["location"], pirate["capacity"])
        marine_state = []
        for marine_dict in state_dict["marine_ships"].values():
            if self.distances[marine_dict["path"][marine_dict["index"]]][pirate_state[0][0]][pirate_state[0][1]] <= 3:
                marine_state.append(marine_dict["index"])
            else:
                marine_state.append(-1)
        
        closet_treasure = None
        distance = float('inf')
        treasure = None
        for name, treasure_dict in state_dict["treasures"].items():
            bfs_dis = self.distances[treasure_dict["location"]][pirate_state[0][0]][pirate_state[0][1]]
            if bfs_dis is not None and bfs_dis < distance:
                distance = bfs_dis
                closet_treasure = treasure_dict["location"]
                treasure = name

        treasure_state = [closet_treasure]

        return ((pirate_state,), tuple(marine_state), tuple(treasure_state)), treasure
    
    def relaxed_transition(self, state, action):
        if action == "reset":
            return [self.relaxed_init], [1]
        
        curr_marines = state[1]
        curr_treasures = state[2]
        next_states = []
        probabilities = []
        possible_pirate_movements = pirate_movements(self.pirate_idx, state, action)
        possible_marine_movements, marine_prob = self.relaxed_marine_movements(curr_marines, self.marine_idx)
        possible_treasure_movements, treasure_probs = treasure_movements_prob(curr_treasures, self.treasure_idx)

        for pirate_state in possible_pirate_movements:
            for marine_state in possible_marine_movements:
                if time.time() - self.act_start_time > self.act_time_limit:
                    raise TimeLimitExceededException("Time limit exceeded during value iteration")
                for i, treasure_state in enumerate(possible_treasure_movements):
                    next_state = (pirate_state, marine_state, treasure_state)
                    next_states.append(next_state)
                    probabilities.append(marine_prob * treasure_probs[i])

        return next_states, probabilities
    
    def relaxed_marine_movements(self, curr_marines, marine_dict):         
        probability = 1
        new_marine_indecies = [[]] * len(curr_marines)

        for marine_idx, path_idx in enumerate(curr_marines):
            if path_idx == -1:
                new_marine_indecies[marine_idx] = [-1]
                continue

            new_marine_indecies[marine_idx] = possible_marine_movements(marine_dict[marine_idx]["path"], path_idx)
            probability *= 1/len(new_marine_indecies[marine_idx])

        combinations = list(product(*new_marine_indecies))
        return combinations, probability

    

    def act(self, state_dict):
        self.act_start_time = time.time()

        state, treasure = self.relax_state(state_dict)
        if treasure is None: # no reachable treasure
            return ("terminate")
        
        step = min(2, state_dict["turns to go"])
        # self.gamma = step / state_dict["turns to go"]
        self.gamma = 0.5
        

        self.values = {t: {} for t in range(step + 1)}
        self.policy = {t: {} for t in range(step + 1)}
        fall_back = self.value_for_state(state, 1)
        fall_back_policy = deepcopy(self.policy)


        try:
            self.values = {t: {} for t in range(step + 1)} 
            self.policy = {t: {} for t in range(step + 1)}
            _ = self.value_for_state(state, step)
            rel_action = self.policy[step][state]
            pos_states, _ = self.relaxed_transition(state, rel_action)
            for next_state in pos_states:
                if self.reward_function(rel_action, state, next_state) + self.my_reward < 0:
                    rel_action = fall_back_policy[1][state]
                    break
                    
        except TimeLimitExceededException:
            step = 1
            rel_action = self.fall_back_policy[1][state]
            
        

        if self.my_reward + fall_back <= 0:
            return ("terminate")
        
        if isinstance(rel_action, str) and rel_action == "reset":
            return ("reset")
    
        action = []

        if rel_action[0][0] == "deposit":
            self.my_reward = DROP_IN_DESTINATION_REWARD * (2 - state[0][0][1])
        

        for pirate, values in state_dict["pirate_ships"].items():
            a, p, *rest = list(rel_action[0])
            if a == "sail":
                action.append((a, pirate, rest[0]))
            elif a == "collect":
                action.append((a, pirate, treasure))
            else:
                action.append((a, pirate))

        # print(action)

        return tuple(action)
            
    def num_marine_moves(self, marine_idx, idx):
        path_len = len(self.marine_idx[marine_idx]["path"])
        if path_len == 1:
            return 1
        if idx < path_len - 1 and idx > 0:
            return 3
        else:
            return 2

    def get_immediate_reward(self, state, action):
        if action[0] == "reset":
            return -RESET_PENALTY
        
        immediate_reward = 0
        pirates, marines, _ = state
        pirate_loc, capacity = pirates[0]

        if action[0] == "deposit" and pirate_loc == self.base and capacity < 2:
            immediate_reward += DROP_IN_DESTINATION_REWARD * (2 - capacity)
        if action[0] == "sail":
            pirate_loc = action[2]
        
        for marine_idx, idx in enumerate(marines):
            if idx == -1:
                continue
            marine_loc = self.marine_idx[marine_idx]["path"][marines[idx]]
            if marine_loc == pirate_loc:
                immediate_reward -= MARINE_COLLISION_PENALTY

        return immediate_reward
    

                    

def init_distances(agent):
    distance = {}
    distance[agent.base] = get_directions_to_goal(agent.problem_map, agent.base)
    for val in agent.initial["treasures"].values():
        for location in val["possible_locations"]:
            if location not in distance:
                distance[location] = get_directions_to_goal(agent.problem_map, location)
    
    for val in agent.initial["marine_ships"].values():
        for location in val["path"]:
            if location not in distance:
                distance[location] = get_directions_to_goal(agent.problem_map, location)
    return distance

def get_directions_to_goal(map, target):
    rows, cols = len(map), len(map[0])
    directions = [[None for _ in range(cols)] for _ in range(rows)]  # To store directions
    visited = [[False for _ in range(cols)] for _ in range(rows)]  # To track visited cells

    moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    queue = deque()
    queue.append((target[0], target[1], 0))
    

    # Perform BFS from the target
    while queue:
        r, c, distance = queue.popleft()
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and map[nr][nc] != 'I':
                visited[nr][nc] = True
                directions[nr][nc] = distance + 1  # Store the reverse direction
                queue.append((nr, nc, distance + 1))

    return directions


class TimeLimitExceededException(Exception):
    pass

def manhattan(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])

    # def pos_hue(self, state):
    #     pirates, marines, treasures = state
    #     (pirate_loc, capacity) = pirates[0]

    #     heuristic_value = 0

    #     closest_treasure_distance = min(manhattan(pirate_loc, tres_loc) for tres_loc in treasures)
    #     base_distance = manhattan(pirate_loc, self.base)

    #     if capacity == 0:
    #         heuristic_value += self.max_distance / (base_distance + 1)
    #     else:
    #         heuristic_value += self.max_distance / closest_treasure_distance

    #     for marine_idx, idx in enumerate(marines):
    #         if idx != -1:
    #             continue
    #         marine_loc = self.marine_idx[marine_idx]["path"][marine_idx]
    #         if manhattan(pirate_loc, marine_loc) <= 4:
    #             heuristic_value -= self.max_distance / manhattan(pirate_loc, marine_loc)

    #     return heuristic_value

# def treasure_to_pirate(state):
#     pirates = state[0]
#     treasures = state[2]
#     total_dis = 0
#     for (location, capacity) in pirates:
#         min_distance = float('inf')
#         for treasure in treasures:
#             distance = manhattan(location, treasure)
#             if distance < min_distance:
#                 min_distance = distance
#         total_dis += min_distance * capacity
#     return total_dis

# def pirate_to_base(agent, state):
#     pirates = state[0]
#     total_dis = 0
#     for (location, capacity) in pirates:
#         total_dis += manhattan(location, agent.base) * (2 - capacity)
#     return total_dis

# def manhattan(x, y):
#     return abs(x[0] - y[0]) + abs(x[1] - y[1])