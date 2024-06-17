import search_
import random
import math
from itertools import product



ids = [""]


class OnePieceProblem(search_.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        
        # save the unchanging items in the problem object, basically the map.
        self.problem_map = initial["map"]
        self.horizontal_bound = len(self.problem_map)
        self.vertical_bound = len(self.problem_map[0])
        self.treasures_names = {}
        self.treasures_positions = {}
        self.base = ()
        for pos in initial["pirate_ships"].values():
            self.base = pos
            break

        for treasure, position in initial['treasures'].items():
            self.treasures_names[treasure] = position
            if position in self.treasures_positions:
                self.treasures_positions[position].append(treasure)
            else:
                self.treasures_positions[position] = [treasure]
            

        # (priate_ship, position, collected_treasures[])
        pirate_ships = frozenset((pirate_ship, (pos, ())) for pirate_ship, pos in initial['pirate_ships'].items())
        
        # (marine_ship, path, index, moving_forward_flag)
        marine_ships = frozenset((ship, (tuple(path), 0, True)) for ship, path in initial['marine_ships'].items())

        # (treasure[])
        delivered_treasures = frozenset()


        transformed_initial = (pirate_ships, marine_ships, delivered_treasures)

        search_.Problem.__init__(self, transformed_initial)

        

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        actions_list = []

        pirate_ships, _, _ = state

        for ship, (ship_pos, collected_treasures) in pirate_ships: #check for possible moves
            actions = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_pos = (ship_pos[0] + dx, ship_pos[1] + dy)
                if self.is_valid_move(new_pos):
                    actions.append(('sail', ship, new_pos))

                if new_pos in self.treasures_positions and len(collected_treasures) < 2: #check to collect treasure action
                    for treasure in self.treasures_positions[new_pos]:
                        actions.append(('collect_treasure', ship, treasure))

            if len(collected_treasures) > 0:
                if ship_pos == self.base:
                    actions.append(('deposit_treasures', ship))
            
            actions.append(('wait', ship)) #adding 'wait' action for each ship
            actions_list.append(actions)
        
        combinations_list = list(product(*actions_list))


                        
        return combinations_list


    def is_valid_move(self, pos):
        """Check if the move position is valid (not an island)"""
        x, y = pos
        if not (0 <= x < self.horizontal_bound and 0 <= y < self.vertical_bound):
            return False  # Out of bounds
        if self.problem_map[x][y] == 'I':
            return False  # Island
        return True



    def result(self, state, actions):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""

        pirate_ships, marine_ships, delivered_treasures = state

        pirate_ships_dict = {pirate_ship: (position, collected_treasures) for pirate_ship, (position, collected_treasures) in pirate_ships}
        marine_ships_arr = []
        marine_locations = {}

        # advance the marine ships
        for marine_ship, (path, index, moving_forward) in marine_ships:
            if len(path) <= 1:
                pass
            elif moving_forward:
                if index < len(path) - 1:
                    index += 1
                else:
                    moving_forward = False
                    index -= 1
            else:
                if index > 0:
                    index -= 1
                else:
                    moving_forward = True
                    index += 1
            marine_ships_arr.append((marine_ship, (path, index, moving_forward)))
            marine_locations[path[index]] = 0


        for action in actions:
            action_type, ship, *details = action

            # execute the action
            if action_type == 'sail' or action_type =='wait':
                new_pos = pirate_ships_dict[ship][0]
                if action_type == 'sail':
                    new_pos = details[0]

                # check if the ship lands on the same tile as a marine and if carries treasure, lose it
                if len(pirate_ships_dict[ship][1]) > 0 and new_pos in marine_locations:
                    pirate_ships_dict[ship] = (new_pos, ())
                else:
                    pirate_ships_dict[ship] = (new_pos, pirate_ships_dict[ship][1])

            elif action_type == 'collect_treasure' and pirate_ships_dict[ship][0] not in marine_locations:
                    pirate_ships_dict[ship] = (pirate_ships_dict[ship][0], pirate_ships_dict[ship][1] + tuple(details))
            
            elif action_type == 'deposit_treasures':
                delivered_treasures = set(delivered_treasures)  # Ensure it's a mutable set
                delivered_treasures.update(pirate_ships_dict[ship][1])  # Add elements from another set
                delivered_treasures = frozenset(delivered_treasures)
                pirate_ships_dict[ship] = (pirate_ships_dict[ship][0], ())


        new_pirate_ships = frozenset(pirate_ships_dict.items())
        marine_ships_fs = frozenset(marine_ships_arr)

        return (new_pirate_ships, marine_ships_fs, delivered_treasures)



    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        
        _, _, delivered = state
        return len(delivered) == len(self.treasures_names)



    

    # hue 3
    def h(self, node):
        pirates, marine_ships, delivered = node.state
        marine_ships_dict = {marine_ship: path[index] for marine_ship, (path, index, _) in marine_ships}

        collected_treasures= {}
        uncollected_treasures = {}
        uncollected_locations = {}
        pirates_1 = {}
        pirates_0 = {}

        sum_hue = 0

        for pirate, (pos, treasures) in pirates:
            if len(treasures) == 0:
                pirates_0[pirate] = pos
            else:
                for tres in treasures:
                    if tres in collected_treasures: # penalty for carrying the same treasure
                        sum_hue += 10

                    if pos in marine_ships_dict.values(): # marine colition with treasure penalty
                        sum_hue += 10

                    collected_treasures[tres] = pos

                    if len(treasures) == 1:
                        pirates_1[pirate] = pos

        for tres, pos in self.treasures_names.items():
            if tres not in delivered and tres not in collected_treasures:
                uncollected_treasures[tres] = pos
                if pos in uncollected_locations:
                    uncollected_locations[pos] += 1
                else:
                    uncollected_locations[pos] = 1

        multi_treasures = {pos: num for pos, num in uncollected_locations.items() if num > 1}

        # distance treasures from base
        sum_hue += self.cal_h_2(collected_treasures, uncollected_treasures)


        # treasure distance from available pirates
        for _, cur_pos in uncollected_treasures.items():
            distances_1 = [manhattan(cur_pos, pos) for _, pos in pirates_1.items()]
            distances_0 = [manhattan(cur_pos, pos) for _, pos in pirates_0.items()]
            if distances_0 and distances_1:
                sum_hue += min(min(distances_0), min(distances_1))
            elif distances_0:
                sum_hue += min(distances_0)
            elif distances_1:
                sum_hue += min(distances_1)
            else:
                base_distances = self.base_distances(cur_pos)
                if base_distances:
                    sum_hue += min(base_distances)
                else:
                    sum_hue += float('inf')


        # targeting the same treasures penalty
        h_val = 0
        treasure_target_counts = {tres: 0 for tres in uncollected_treasures.keys()}

        for pirate, (pos, treasures) in pirates:
            closest_treasure, min_distance = None, float('inf')
            for tres, tres_pos in uncollected_treasures.items():
                if cur_pos in multi_treasures and multi_treasures[cur_pos] > 2:
                    continue

                distance = manhattan(pos, tres_pos)
                if distance < min_distance:
                    closest_treasure, min_distance = tres, distance

            if closest_treasure is not None:
                treasure_target_counts[closest_treasure] += 1

        for tres, count in treasure_target_counts.items():
            if count > 1:
                h_val += count * 2
        sum_hue += h_val


        val = 0
        for tres_pos, count in multi_treasures.items():
            dis_num = [(manhattan(pos, tres_pos), len(treasures)) for _, (pos, treasures) in pirates]
            sorted_pirates = sorted(dis_num, key=lambda x: x[1])
            base_dis = manhattan(self.base, tres_pos)
            for dis, num in sorted_pirates:
                if count < 1:
                    break
                if num == 0:
                    val += dis * 2 + base_dis
                    count -= 2
                else:
                    val += dis + base_dis * 3
                    count -= 2 - num
        
        sum_hue += val
                

        return sum_hue

    """Feel free to add your own functions
    (-2, -2, None) means there was a timeout"""


    def h_1(self, node):
        """number of uncollected treasures divided by the number of pirates."""
        pirates, _, delivered = node.state
        collected_treasures = {}
        for pirate, (pos, treasures) in pirates:
            for tres in treasures:
                collected_treasures[tres] = pos

        return self.cal_h_1(pirates, delivered, collected_treasures)
    
    def cal_h_1(self, pirates, delivered, collected_treasures):
        return (len(self.treasures_names) - (len(delivered) + len(collected_treasures))) / len(pirates)
    
    
    def h_2(self, node):
        """Sum of the distances from the pirate base to the closest sea cell adjacent to a treasure
           for each treasure, divided by the number of pirates.
           If there is a treasure which all the adjacent cells are islands, return infinity."""
        pirates, _, delivered = node.state

        collected_treasures= {} # {treasure_name: (x,y) - position}
        uncolected_treasures = {} # {treasure_name: (x,y) - position}

        for _, (pos, treasures) in pirates:
            for tres in treasures:
                collected_treasures[tres] = pos

        for tres, pos in self.treasures_names.items():
            if tres not in delivered and tres not in collected_treasures:
                uncolected_treasures[tres] = pos
        

        return self.cal_h_2(collected_treasures, uncolected_treasures) / len(pirates)
    
    def cal_h_2(self, collected_treasures, uncollected_treasures):  # gets min distances with sea cell buffer
        h_value = 0
        for tres_pos in uncollected_treasures.values():
            closest_distances = self.base_distances(tres_pos)
            if closest_distances:
                h_value += min(closest_distances)
            else:
                return float('inf')
            
        for tres_pos in collected_treasures.values():
            closest_distances = self.base_distances(tres_pos)
            if closest_distances:
                h_value += min(closest_distances)
            
                
        return h_value
    

    def base_distances(self, pos): # gets the distances from the base to the closest sea cell of pos
        return [manhattan(self.base, (pos[0] + x, pos[1] + y)) # self.base - base position
                                for x, y in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                if 0 <= pos[0] + x < self.horizontal_bound # horizontal map bound
                                    and 0 <= pos[1] + y < self.vertical_bound # vertical map bound
                                    and self.problem_map[pos[0] + x][pos[1] + y] != "I"]
    
        
    
def manhattan(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])


def create_onepiece_problem(game):
    return OnePieceProblem(game)

