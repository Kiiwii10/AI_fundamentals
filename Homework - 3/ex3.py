IDS = [""]
from simulator import Simulator
import random
from itertools import product
import time
import math
from collections import deque
from copy import deepcopy


ACT_TIMEOUT = 4.9
INIT_TIMEOUT = 59.5
MOVEMENTS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DISTANCE = {}
SAIL_ACTIONS = {}


class Node:
    def __init__(self, player, turns_to_go, parent=None, action=None):
        self.parent = parent
        self.player = player
        self.action = action
        self.score = [0, 0]
        self.visits = 0
        self.ucb1 = float('inf')
        self.children = {}  # Using a dictionary to map actions to children
        self.turns_to_go = turns_to_go

    def add_child(self, action):
        if action not in self.children:
            child = UCTNode(3 - self.player, self.turns_to_go - 1, self, action)
            self.children[action] = child
        return self.children[action]

    def expand(self, possible_actions):
        for action in possible_actions:
            if action not in self.children:
                self.add_child(action)

    def select_child(self, state):
        possible_actions = heuristic_actions(state, self.player)
        self.expand(possible_actions)
        
        #TODO: maybe add the hue val here
        best_child = max((self.children[action] for action in possible_actions
                          if action in self.children),
                         key=lambda child: child.ucb1, default=None)

        return best_child
    
    def update(self, data):
        self.visits += 1

        self.score[0] += data[0]
        self.score[1] += data[1]

        self.ucb1 = self.uct_value()

    def uct_value(self):
        if self.visits == 0 or self.parent is None or self.parent.visits == 0:
            return float('inf')
        else:
            # because we are looking at this node from the parents percpective the score is reversed.
            score = self.score[2 - self.player] - self.score[self.player - 1] 
            return score / self.visits + math.sqrt(2 * math.log(self.parent.visits) / self.visits)
    
    
    def get_heuristic_val(self, state):
        pirate_ships = state["pirate_ships"]
        treasures = state["treasures"]
        pirate_treasures = {}
        treasure_loc = {}
        num_uncollected = 0
        for treasure, val in treasures.items():
            if not isinstance(val["location"], tuple):
                if val["location"] not in pirate_treasures:
                    pirate_treasures[val["location"]] = {}
                pirate_treasures[val["location"]][treasure] = val["reward"]
            else:
                if val["location"] not in treasure_loc:
                    treasure_loc[val["location"]] = {}
                treasure_loc[val["location"]][treasure] = val["reward"]
                num_uncollected += 1

        res = 0

        for ship, val in pirate_ships.items():
            if val["player"] == self.player:
                ship_pos = val["location"]
                capacity = val["capacity"]
                
                if capacity == 2:
                    for location, val in treasure_loc.items():
                        for treasure, value in val.items():
                            res += value / (DISTANCE[location][ship_pos] +\
                                            DISTANCE[location][state["base"]] + 2)
                    res /= num_uncollected if num_uncollected > 0 else 1
                else:
                    for value in pirate_treasures[ship].values():
                        res += value / (DISTANCE[state["base"]][ship_pos] + 1)
                    if capacity == 1:
                        for location, val in treasure_loc.items():
                            for treasure, value in val.items():
                                res += value / (DISTANCE[location][ship_pos] +\
                                                DISTANCE[location][state["base"]] + 2)
                        res /= num_uncollected if num_uncollected > 0 else 1

                for enemy_ship in pirate_ships:
                    if enemy_ship in pirate_treasures:
                        for value in pirate_treasures[enemy_ship].values():
                            res += value / (manhattan(pirate_ships[enemy_ship]["location"], ship_pos) + 1)

        return res


class Agent:
    def __init__(self, initial_state, player_number):
        self.ids = IDS
        self.player_number = player_number
        self.my_ships = []
        self.simulator = Simulator(initial_state)
        self.time_limit = 0
        init_distances(initial_state)

    def selection(self, root):
        current_node = root
        while len(current_node.children) != 0:
            player = current_node.player
            current_node = current_node.select_child(self.simulator.state)
            self.simulator.act(current_node.action, player)
        return current_node

    def expansion(self, parent_node):
        possible_actions = heuristic_actions(self.simulator.state, parent_node.player)
        for action in possible_actions:
            parent_node.add_child(action)
        

    def simulation(self, player):
        if player == 2 and self.simulator.turns_to_go > 0:
            action = random.choice(heuristic_actions(self.simulator.state, 2))
            self.simulator.act(action, 2)

            self.simulator.check_collision_with_marines()
            self.simulator.move_marines()

        while self.simulator.turns_to_go > 0:
            if time.time() >= self.time_limit:
                raise TimeoutError
            for p in range(1, 3):
                action = random.choice(heuristic_actions(self.simulator.state, p))
                self.simulator.act(action, p)

            self.simulator.check_collision_with_marines()
            self.simulator.move_marines()
            
        return [self.simulator.get_score()["player 1"], self.simulator.get_score()["player 2"]]

    def backpropagation(self, node, res):
        while node is not None:
            node.update(res)
            node = node.parent

    def act(self, state):
        root = UCTNode(self.player_number, state["turns to go"])
        self.time_limit = time.time() + ACT_TIMEOUT
        
        while time.time() < self.time_limit:
            self.simulator = Simulator(state)
            node = self.selection(root)
            if node.turns_to_go == 0:
                self.backpropagation(node, 
                                     [self.simulator.get_score()["player 1"], self.simulator.get_score()["player 2"]])
            else:
                try:
                    self.expansion(node)
                    result = self.simulation(node.player)
                    self.backpropagation(node, result)
                except TimeoutError:
                    break

        return max(root.children.values(), key=lambda child: child.ucb1).action
    

def heuristic_actions(state, player):
    treasures = state["treasures"]
    pirate_ships = state["pirate_ships"]
    enemy_ships = [ship for ship, val in pirate_ships.items() if val["player"] != player]
    marine_locations = {val["path"][val["index"]] for val in state["marine_ships"].values()}
    pirate_treasures = {}
    treasure_loc = {}
    for treasure, val in treasures.items():
        if not isinstance(val["location"], tuple):
            if val["location"] not in pirate_treasures:
                pirate_treasures[val["location"]] = []
            pirate_treasures[val["location"]].append(treasure)
        else:
            if val["location"] not in treasure_loc:
                treasure_loc[val["location"]] = []
            treasure_loc[val["location"]].append(treasure)


    actions = []
    for ship, val in pirate_ships.items():
        ship_actions = []
        if val["player"] == player:
            capacity = val["capacity"]
            ship_pos = val["location"]
            wait_flag = True

            if ship_pos == state["base"] and capacity < 2:
                for treasure in pirate_treasures[ship]:
                    ship_actions.append(('deposit', ship, treasure))
                actions.append(ship_actions)
                continue

            for dx, dy in MOVEMENTS:
                new_pos = (ship_pos[0] + dx, ship_pos[1] + dy)
                
                if new_pos in treasure_loc and capacity > 0 and ship_pos not in marine_locations:
                    for treasure in treasure_loc[new_pos]:
                        ship_actions.append(('collect', ship, treasure))
                    wait_flag = False

                if (0 <= new_pos[0] < len(state["map"]) and 0 <= new_pos[1] < len(state["map"][0]))\
                    and state["map"][new_pos[0]][new_pos[1]] != 'I'\
                    and not (new_pos in marine_locations and capacity < 2):
                    ship_actions.append(('sail', ship, new_pos))

            
            for enemy_ship in enemy_ships:
                if ship_pos == pirate_ships[enemy_ship]["location"] and enemy_ship in pirate_treasures:
                    ship_actions.append(('plunder', ship, enemy_ship))
                    wait_flag = False
            
            if wait_flag:
                ship_actions.append(('wait', ship))

            actions.append(ship_actions)


    all_combinations = product(*actions)
    return [comb for comb in all_combinations if is_valid_combination(comb)]



def init_distances(state):
    global DISTANCE
    DISTANCE[state["base"]] = get_directions_to_goal(state["map"], state["base"])
    for i in range(len(state["map"])):
        for j in range(len(state["map"])):
            if state["map"][i][j] == "I" and (i, j) not in DISTANCE:
                DISTANCE[(i,j)] = get_directions_to_goal(state["map"], (i,j))
    
    for val in state["marine_ships"].values():
        for location in val["path"]:
            if location not in DISTANCE:
                DISTANCE[location] = get_directions_to_goal(state["map"], location)


def get_directions_to_goal(map, target):
    rows, cols = len(map), len(map[0])
    directions = {(i,j): None for i in range(cols) for j in range(rows)}  # To store directions
    visited = {(i,j): False for i in range(cols) for j in range(rows)}  # To track visited cells

    moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    queue = deque()
    queue.append((target[0], target[1], 0))
    

    # Perform BFS from the target
    while queue:
        r, c, dist = queue.popleft()
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[(nr, nc)] and map[nr][nc] != 'I':
                visited[(nr, nc)] = True
                directions[(nr, nc)] = dist + 1  # Store the reverse direction
                queue.append((nr, nc, dist + 1))

    return directions



class TimeLimitExceededException(Exception):
    pass

def manhattan(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])

    

class UCTNode:
    def __init__(self, player, turns_to_go, parent=None, action=None):
        self.parent = parent
        self.player = player
        self.action = action
        self.score = [0, 0]
        self.visits = 0
        self.ucb1 = float('inf')
        self.children = {}  # Using a dictionary to map actions to children
        self.turns_to_go = turns_to_go

    def add_child(self, action):
        if action not in self.children:
            child = UCTNode(3 - self.player, self.turns_to_go - 1, self, action)
            self.children[action] = child
        return self.children[action]

    def expand(self, possible_actions):
        for action in possible_actions:
            if action not in self.children:
                self.add_child(action)

    def select_child(self, state):
        # Ensure all possible actions are expanded
        possible_actions = get_possible_actions(state, self.player)
        self.expand(possible_actions)
        
        # Select the child with the highest UCB1 value among the available actions
        best_child = max((self.children[action] for action in possible_actions
                          if action in self.children),
                         key=lambda child: child.ucb1, default=None)

        return best_child
    
    def update(self, data):
        self.visits += 1
        # if data == 3 - self.player:
        #     self.score += 1
        self.score[0] += data[0]
        self.score[1] += data[1]

        self.ucb1 = self.uct_value()

    def uct_value(self):
        if self.visits == 0 or self.parent is None or self.parent.visits == 0:
            return float('inf')
        else:
            # TODO: maybe we need to use min max after all and only backpropogate the score on the agent.
            # because we are looking at this node from the parents percpective the score is reversed.
            score = self.score[2 - self.player] - self.score[self.player - 1] 
            return score / self.visits + math.sqrt(2 * math.log(self.parent.visits) / self.visits)
    

class UCTAgent:
    def __init__(self, initial_state, player_number):
        self.ids = IDS
        self.player_number = player_number
        self.my_ships = []
        self.simulator = Simulator(initial_state)
        self.time_limit = 0
        for ship_name, ship in initial_state['pirate_ships'].items():
            if ship['player'] == player_number:
                self.my_ships.append(ship_name)

    def selection(self, root):
        current_node = root
        while len(current_node.children) != 0:
            player = current_node.player
            current_node = current_node.select_child(self.simulator.state)
            self.simulator.act(current_node.action, player)
        return current_node

    def expansion(self, parent_node):
        possible_actions = get_possible_actions(self.simulator.state, parent_node.player)
        for action in possible_actions:
            parent_node.add_child(action)
        

    def simulation(self, player):
        if player == 2 and self.simulator.turns_to_go > 0:
            action = random.choice(get_possible_actions(self.simulator.state, 2))
            self.simulator.act(action, 2)

            self.simulator.check_collision_with_marines()
            self.simulator.move_marines()

        while self.simulator.turns_to_go > 0:
            if time.time() >= self.time_limit:
                raise TimeoutError
            for p in range(1, 3):
                action = random.choice(get_possible_actions(self.simulator.state, p))
                self.simulator.act(action, p)

            self.simulator.check_collision_with_marines()
            self.simulator.move_marines()
            
        return [self.simulator.get_score()["player 1"], self.simulator.get_score()["player 2"]]

    def backpropagation(self, node, res):
        while node is not None:
            node.update(res)
            node = node.parent

    def act(self, state):
        root = UCTNode(self.player_number, state["turns to go"])
        self.time_limit = time.time() + ACT_TIMEOUT
        
        while time.time() < self.time_limit:
            self.simulator = Simulator(state)
            node = self.selection(root)
            if node.turns_to_go == 0:
                self.backpropagation(node, 
                                     [self.simulator.get_score()["player 1"], self.simulator.get_score()["player 2"]])
            else:
                try:
                    self.expansion(node)
                    result = self.simulation(node.player)
                    self.backpropagation(node, result)
                except TimeoutError:
                    break

        return max(root.children.values(), key=lambda child: child.ucb1).action
    

def get_possible_actions(state, player):
    treasures = state["treasures"]
    pirate_ships = state["pirate_ships"]
    enemy_ships = [ship for ship, val in pirate_ships.items() if val["player"] != player]
    pirate_treasures = {}
    treasure_loc = {}
    for treasure, val in treasures.items():
        if not isinstance(val["location"], tuple):
            if val["location"] not in pirate_treasures:
                pirate_treasures[val["location"]] = []
            pirate_treasures[val["location"]].append(treasure)
        else:
            if val["location"] not in treasure_loc:
                treasure_loc[val["location"]] = []
            treasure_loc[val["location"]].append(treasure)


    actions = []
    for ship, val in pirate_ships.items():
        ship_actions = []
        if val["player"] == player:
            capacity = val["capacity"]
            ship_pos = val["location"]

            if ship_pos == state["base"] and ship in pirate_treasures:
                for treasure in pirate_treasures[ship]:
                    ship_actions.append(('deposit', ship, treasure))

            for dx, dy in MOVEMENTS:
                new_pos = (ship_pos[0] + dx, ship_pos[1] + dy)
                
                if new_pos in treasure_loc and capacity > 0:
                    for treasure in treasure_loc[new_pos]:
                        ship_actions.append(('collect', ship, treasure))

                if (0 <= new_pos[0] < len(state["map"]) and 0 <= new_pos[1] < len(state["map"][0]))\
                    and state["map"][new_pos[0]][new_pos[1]] != 'I':
                    ship_actions.append(('sail', ship, new_pos))

            
            for enemy_ship in enemy_ships:
                if ship_pos == pirate_ships[enemy_ship]["location"]:
                    ship_actions.append(('plunder', ship, enemy_ship))
            
            ship_actions.append(('wait', ship))

            actions.append(ship_actions)


    all_combinations = product(*actions)
    return [comb for comb in all_combinations if is_valid_combination(comb)]


def is_valid_combination(combination):
    collects = [t for t in combination if t[0] == 'collect']
    return len({t[2] for t in collects}) == len(collects)

def init_sail_actions(map):
    global SAIL_ACTIONS
    for i in range(len(map)):
        for j in range(len(map[0])):
            if map[i][j] != 'I':
                SAIL_ACTIONS[(i,j)] = []
                for dx, dy in MOVEMENTS:
                    new_pos = (i + dx, j + dy)
                    if 0 <= new_pos[0] < len(map) and 0 <= new_pos[1] < len(map[0]) and map[new_pos[0]][new_pos[1]] != 'I':
                        SAIL_ACTIONS[(i,j)].append(new_pos)
