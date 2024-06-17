from simulator import Simulator
import ex3
import sample_agent
from copy import deepcopy
import time

CONSTRUCTOR_TIMEOUT = 60
ACTION_TIMEOUT = 5
DIMENSIONS = (7, 7)
PENALTY = 10000


class Game:
    """
    This class plays the game for you. You are given a sample agent to play against.
    """
    def __init__(self, an_input):
        self.initial_state = deepcopy(an_input)
        self.simulator = Simulator(self.initial_state)
        self.ids = []
        self.agents = []
        self.score = [0, 0]

    def initiate_agent(self, module, player_number, UCT_flag=False):
        """
        :param UCT_flag: Uses UCT_Agent instead of general one
        :return: agent
        """
        start = time.time()
        if UCT_flag:
            agent = module.UCTAgent(self.initial_state, player_number)
        else:
            agent = module.Agent(self.initial_state, player_number)
        if time.time() - start > CONSTRUCTOR_TIMEOUT:
            raise ValueError(f'agent timed out on constructor!')
        return agent

    def get_action(self, agent, player):
        start = time.time()
        action = agent.act(self.simulator.get_state())
        finish = time.time()
        if finish - start > ACTION_TIMEOUT:
            self.score[player] -= PENALTY
            raise ValueError(f'{self.ids[player]} timed out on action!')
        return action

    def play_episode(self, swapped=False):
        global errors
        length_of_episode = int(self.initial_state["turns to go"]/2)
        for i in range(length_of_episode):
            for number, agent in enumerate(self.agents):
                try:
                    action = self.get_action(agent, number)
                except (AssertionError, ValueError) as e:
                    print(e)
                    errors.append(f'{agent.ids} timeout on action!')
                    self.score[number] -= PENALTY
                    return
                try:
                    self.simulator.act(action, number + 1)
                except (AssertionError, ValueError):
                    print(f'{agent.ids} chose illegal action!')
                    errors.append(f'{agent.ids} chose illegal action!\naction - {action}\n state - {self.simulator.state}')

                    self.score[number] -= PENALTY
                    return
                print(f"{agent.ids} chose {action}")

            print(f"marine locations: {[val['path'][val["index"]] for val in self.simulator.state['marine_ships'].values()]}")
            self.simulator.check_collision_with_marines()
            self.simulator.move_marines()
            print(f"score: {self.simulator.get_score()}")
            print(f"-----")
        if not swapped:
            self.score[0] += self.simulator.get_score()['player 1']
            self.score[1] += self.simulator.get_score()['player 2']
        else:
            self.score[0] += self.simulator.get_score()['player 2']
            self.score[1] += self.simulator.get_score()['player 1']


            print(f'***********  end of round!  ************ \n \n')

    def play_game(self):
        """
        When initiating the agents in this function, you can set UCT_flag to True in initiate_agent(), when not using
        the general agent. You may also use an agent of your own, instead of sample agent.
        """
        print(f'***********  starting a first round!  ************ \n \n')
        self.agents = [self.initiate_agent(ex3, 1, UCT_flag=True),
                       self.initiate_agent(sample_agent, 2)]
        self.ids = ['Your agent', 'Rival agent']
        self.play_episode()
        print(self.simulator.state)

        print(f'***********  starting a second round!  ************ \n \n')
        self.simulator = Simulator(self.initial_state)

        self.agents = [self.initiate_agent(sample_agent, 1),
                       self.initiate_agent(ex3, 2, UCT_flag=True)]
        self.ids = ['Rival agent', 'Your agent']
        self.play_episode(swapped=True)
        print(f'end of game!')
        return self.score

errors = []

def main():
    global errors
    an_input = {
        "map": [
            ['S', 'S', 'I', 'S', 'S', 'S', 'S'],
            ['S', 'S', 'I', 'S', 'S', 'S', 'S'],
            ['B', 'S', 'S', 'S', 'S', 'S', 'S'],
            ['S', 'S', 'I', 'S', 'S', 'I', 'S'],
            ['S', 'S', 'I', 'S', 'S', 'I', 'S'],
            ['S', 'S', 'S', 'S', 'S', 'I', 'S'],
            ['S', 'S', 'S', 'S', 'S', 'I', 'I']
        ],
        "base": (2, 0),
        "pirate_ships": {'pirate_ship_1': {"location": (2, 0),
                                           "capacity": 2,
                                           "player": 1},
                         'pirate_ship_2': {"location": (2, 0,),
                                           "capacity": 2,
                                           "player": 1},
                         'pirate_ship_3': {"location": (2, 0,),
                                           "capacity": 2,
                                           "player": 2},
                         'pirate_ship_4': {"location": (2, 0,),
                                           "capacity": 2,
                                           "player": 2}
                         },
        "treasures": {'treasure_1': {"location": (0, 2),
                                     "reward": 4}
                      },
        "marine_ships": {'marine_1': {"index": 0,
                                      "path": [(0, 1), (1, 1), (2, 1), (2, 2), (2, 3), (2, 4)]},
                         'marine_2': {"index": 0,
                                      "path": [(2, 5), (2, 4), (3, 4), (4, 4)]}
                         },
        "turns to go": 200
    }
    result = []
    for _ in range(10):
        game = Game(an_input)
        results = game.play_game()
        result.append(results)
        print(f'Score for {ex3.IDS} is {results[0]}, score for {sample_agent.IDS} is {results[1]}')

    print(f'scores: {result}')
    print(f'errors: {errors}')
    print(f'Average score for {ex3.IDS} is {sum([row[0] for row in result])/len(result)}')
    print(f'Average score for {sample_agent.IDS} is {sum(row[1] for row in result)/len(result)}')

if __name__ == '__main__':
    main()
