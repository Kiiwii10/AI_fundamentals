from copy import deepcopy
import logging
import random


TREASURE_ARRIVAL_PROBABILITY = 0.3
TREASURE_NAMES = ["treasure_1", "treasure_2","treasure_3","treasure_4","treasure_5","treasure_6","treasure_7",
                  "treasure_8","treasure_9","treasure_10","treasure_11","treasure_12","treasure_13","treasure_14"]

class Simulator:
    """
    This the simulator class. You may use it for your agent.
    The functions that may interest you are neighbors(), act()
    move_marines() and check_collision_with_marines()
    """
    def __init__(self, initial_state):
        self.state = deepcopy(initial_state)
        self.score = {'player 1': 0, 'player 2': 0}
        self.dimensions = len(self.state['map']), len(self.state['map'][0])
        self.turns_to_go = self.state['turns to go']
        self.base_location = self.state['base']
        self.MARINE_COLLISION_PENALTY = 1

    def neighbors(self, location):
        """
        return the neighbors of a location
        """
        if (type(location) == str):
            return []
        x, y = location[0], location[1]
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        for neighbor in tuple(neighbors):
            if neighbor[0] < 0 or neighbor[0] >= self.dimensions[1] or neighbor[1] < 0 or neighbor[1] >= \
                    self.dimensions[0] or self.state['map'][neighbor[0]][neighbor[1]] == 'I':
                neighbors.remove(neighbor)
        return neighbors

    def check_if_action_legal(self, action, player):
        def _is_move_action_legal(move_action, player):
            pirate_name = move_action[1]
            if pirate_name not in self.state['pirate_ships'].keys():
                logging.error(f"Pirate {pirate_name} does not exist!")
                return False
            if player != self.state['pirate_ships'][pirate_name]['player']:
                logging.error(f"Pirate {pirate_name} does not belong to player {player}!")
                return False
            l1 = self.state['pirate_ships'][pirate_name]['location']
            l2 = move_action[2]
            if l2 not in self.neighbors(l1):
                logging.error(f"Pirate {pirate_name} cannot move from {l1} to {l2}!")
                return False
            return True

        def _is_collect_action_legal(collect_action, player):
            pirate_name = collect_action[1]
            treasure_name = collect_action[2]
            if player != self.state['pirate_ships'][pirate_name]['player']:
                return False
            # check adjacent position
            l1 = self.state['treasures'][treasure_name]['location']
            if self.state['pirate_ships'][pirate_name]['location'] not in self.neighbors(l1):
                return False
            # check ship capacity
            if self.state['pirate_ships'][pirate_name]['capacity'] <= 0:
                return False
            return True

        def _is_deposit_action_legal(deposit_action, player):
            pirate_name = deposit_action[1]
            treasure_name = deposit_action[2]
            # check same position
            if player != self.state['pirate_ships'][pirate_name]['player']:
                return False
            if self.state["pirate_ships"][pirate_name]["location"] != self.base_location:
                return False
            if self.state['treasures'][treasure_name]['location'] != pirate_name:
                return False
            return True

        def _is_plunder_action_legal(plunder_action, player):
            pirate_1_name = plunder_action[1]
            pirate_2_name = plunder_action[2]
            if player != self.state["pirate_ships"][pirate_1_name]["player"]:
                return False
            if self.state["pirate_ships"][pirate_1_name]["location"] != self.state["pirate_ships"][pirate_2_name]["location"]:
                return False
            return True

        def _is_action_mutex(global_action):
            assert type(
                global_action) == tuple, "global action must be a tuple"
            # one action per ship
            if len(set([a[1] for a in global_action])) != len(global_action):
                return True
            # collect the same treasure
            collect_actions = [a for a in global_action if a[0] == 'collect']
            if len(collect_actions) > 1:
                treasures_to_collect = set([a[2] for a in collect_actions])
                if len(treasures_to_collect) != len(collect_actions):
                    return True

            return False

        players_pirates = [pirate for pirate in self.state['pirate_ships'].keys() if self.state['pirate_ships'][pirate]['player'] == player]

        if len(action) != len(players_pirates):
            logging.error(f"You had given {len(action)} atomic commands, while you control {len(players_pirates)}!")
            return False
        for atomic_action in action:
            # trying to act with a pirate that is not yours
            if atomic_action[1] not in players_pirates:
                logging.error(f"Pirate ship {atomic_action[1]} is not yours!")
                return False
            # illegal sail action
            if atomic_action[0] == 'sail':
                if not _is_move_action_legal(atomic_action, player):
                    logging.error(f"Sail action {atomic_action} is illegal!")
                    return False
            # illegal collect action
            elif atomic_action[0] == 'collect':
                if not _is_collect_action_legal(atomic_action, player):
                    logging.error(f"Collect action {atomic_action} is illegal!")
                    return False
            # illegal deposit action
            elif atomic_action[0] == 'deposit':
                if not _is_deposit_action_legal(atomic_action, player):
                    logging.error(f"Deposit action {atomic_action} is illegal!")
                    return False
            # illegal plunder action
            elif atomic_action[0] == "plunder":
                if not _is_plunder_action_legal(atomic_action, player):
                    logging.error(f"Plunder action {atomic_action} is illegal!")
                    return False
            elif atomic_action[0] != 'wait':
                return False
        # check mutex action
        if _is_action_mutex(action):
            logging.error(f"Actions {action} are mutex!")
            return False
        return True

    def apply_action(self, action, player):
        for atomic_action in action:
            self._apply_atomic_action(atomic_action, player)
        self.turns_to_go -= 1

    def check_collision_with_marines(self):
        """
        Checks collisions with marines, applies penalties. Does not move them
        """
        marine_locations = []
        treasures_to_remove = []
        for marine_stats in self.state["marine_ships"].values():
            index = marine_stats["index"]
            marine_locations.append(marine_stats["path"][index])

        for ship_name in self.state["pirate_ships"].keys():
            if self.state["pirate_ships"][ship_name]["location"] in marine_locations:
                self.state["pirate_ships"][ship_name]["capacity"] = 2
                player = self.state["pirate_ships"][ship_name]["player"]
                self.score[f"player {player}"] -= self.MARINE_COLLISION_PENALTY
                for treasure_name in self.state["treasures"].keys():
                    if self.state["treasures"][treasure_name]["location"] == ship_name:
                        treasures_to_remove.append(treasure_name)
        for treasures_r in treasures_to_remove:
            del self.state["treasures"][treasures_r]

    def move_marines(self):
        """
        Moves marines uniformly along their path.
        """
        for marine in self.state['marine_ships']:
            marine_stats = self.state["marine_ships"][marine]
            index = marine_stats["index"]
            if len(marine_stats["path"]) == 1:
                continue
            if index == 0:
                marine_stats["index"] = random.choice([0, 1])
            elif index == len(marine_stats["path"])-1:
                marine_stats["index"] = random.choice([index, index-1])
            else:
                marine_stats["index"] = random.choice(
                    [index-1, index, index+1])

    def _apply_atomic_action(self, atomic_action, player):
        """
        apply an atomic action to the state
        """
        pirate_name = atomic_action[1]
        if atomic_action[0] == 'sail':
            self.state['pirate_ships'][pirate_name]['location'] = atomic_action[2]
            return
        elif atomic_action[0] == 'collect':
            treasure_name = atomic_action[2]
            self.state["pirate_ships"][pirate_name]["capacity"] -=1
            self.state["treasures"][treasure_name]["location"] = pirate_name
            return
        elif atomic_action[0] == 'deposit':
            treasure_name = atomic_action[2]
            self.state['pirate_ships'][pirate_name]['capacity'] += 1
            self.score[f"player {player}"] += self.state['treasures'][treasure_name]['reward']
            del self.state['treasures'][treasure_name]
            return
        elif atomic_action[0] == 'plunder':
            advers_pirate_name = atomic_action[2]
            plundered_treasures = []
            self.state["pirate_ships"][advers_pirate_name]["capacity"] = 2
            for treasure_adv in self.state["treasures"].keys():
                if self.state["treasures"][treasure_adv]["location"] == advers_pirate_name:
                    plundered_treasures.append(treasure_adv)
            for p_treas in plundered_treasures:
                del self.state["treasures"][p_treas]
            return
        elif atomic_action[0] == 'wait':
            return
        else:
            raise NotImplemented

    def add_treasure(self):
        if len(self.state['treasures']) > 9:
            return
        if random.random() < TREASURE_ARRIVAL_PROBABILITY:
            while True:
                treasure_name = random.choice(TREASURE_NAMES)
                if treasure_name not in self.state['treasures'].keys():
                    break
            while True:
                treasure_location = (
                random.randint(0, self.dimensions[0] - 1), random.randint(0, self.dimensions[1] - 1))
                if self.state['map'][treasure_location[0]][treasure_location[1]] == 'I':
                    break

            reward = random.randint(1, 9)
            self.state['treasures'][treasure_name] = {'location': treasure_location,
                                                        'reward': reward}

    def act(self, action, player):
        if self.check_if_action_legal(action, player):
            self.apply_action(action, player)
            self.add_treasure()
        else:
            raise ValueError(f"Illegal action!")

    def print_scores(self):
        print(f"Scores: player 1: {self.score['player 1']}, player 2: {self.score['player 2']}")

    def print_state(self):
        for key, value in self.state.items():
            print(f'{key}:')
            try:
                for secondary_key, secondary_value in value.items():
                    print(f"{secondary_key}: {secondary_value}")
            except AttributeError:
                if key == 'map':
                    for row in value:
                        print(row)
                else:
                    print(f"{self.turns_to_go}")
            print("------------------")

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return deepcopy(self.state)

    def get_score(self):
        return self.score
