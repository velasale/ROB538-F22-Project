import numpy as np
import random


class AgentBase():
    def __init__(self, rows, cols) -> None:
        self.id = None
        # Class identifier
        self.robot_class = 0
        # Class specific action
        self.action_type = 0
        # current position
        self.cur_pos = [0, 0]
        # current comms channel (ROBOT ID)
        self.comms_channel = None

        # Alejo's
        # Q_state_action table for each agent
        self.q_sa_table = np.zeros((rows, cols))
        self.learning_rate = 0.05
        self.epsilon = 1.0
        self.epsilon_updater = 0.95
        self.gamma = 0.9
        self.accumulated_reward = 0
        self.reward_evolution = []
        self.reward = 0
        self.key = ""
        self.move = []
        self.move_2 = []


    def reset_agent(self):
        self.epsilon = 1
        self.accumulated_reward = 0

    def random_move(self, valid_moves, valid_keys):
        # randomly chooses a valid move, takes in a list of valid x,y moves and the corresponding valid keys
        # returns the [x,y] of next move and the key: up, down, left, right or interact
        choice = np.random.randint(len(valid_moves))
        return valid_moves[choice], valid_keys[choice]

    def send_communication(self):
        # TODO: Should return internal map
        return None

    def recieve_communication(self, comms):
        # TODO: Should recieve an internal map from another bot and set comms channel back to None
        self.comms_channel = None
        pass

    def request_communication(self, id):
        # TODO: Sets internal comms channel to the id of the robot we want a map from. This is then checked and sent over
        # in orchardsim.
        # SHOULD BE SET INTERNALLY
        self.comms_channel = id

    def update_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_updater

    def epsilon_greedy(self, values: list, epsilon: float):
        """
        Policy
        :param Q: List with all the states being as rows, and all columns being possible actions
        :param state: State at which we would like to explore next action
        :param epsilon: Parameter to balance exploration and exploitation
        :return:
        """

        # Generate a random number from 0 to 1
        p = random.random()
        choices = len(values)
        # print(choices)
        # if choices == 0:
        #     # If there are not choices to move, remain in the same place
        #     action = self.cur_pos
        #
        # else:
        if p < epsilon:
            # Encourage the agent to Explore!
            action = random.randrange(0, choices)  # outputs random from 0,1,2 or 3
        else:
            # Otherwise explore
            action = np.argmax(values)

        return action

    def update_value(self, move: list, move_2: list, reward: int):
        """
        Updates Qtable using Q_algorithm, which is TD using the max value of the state of the next action
        :param move:    Next state's value
        :param move_2:  Value of the state given that the best action is taken
        :param reward:
        :return:
        """

        # --- Q learning algorithm ---
        # Q(s,a) <-- Q(s,a) + alpha * [(reward + gamma * max Q(s',a') - Q(s,a)]
        current_value = self.q_sa_table[move[0]][move[1]]       # Q(s,a)
        prime_value = self.q_sa_table[move_2[0]][move_2[1]]     # Q(s',a')
        TD = reward + self.gamma * prime_value - current_value
        next_value = current_value + self.learning_rate * TD

        # Update value in Q_sa_table
        self.q_sa_table[move[0]][move[1]] = next_value

    def choose_move(self, observed_points, observed_vals, valid_moves, valid_keys):
        # TODO: Fill out for classes RL stuff
        # randomly chooses a valid move
        # points is a list of the observed points, observed vals is a list of corresponding id of each x,y
        # returns the [x,y] of next move and the key: up, down, left, right or interact

        # --- Step 0: Map valid moves to values from Q_sa_table
        values = []
        for i in range(len(valid_moves)):
            see = valid_moves[i]
            see_row = see[0]
            see_col = see[1]
            values.append(self.q_sa_table[see_row][see_col])

        # --- Step 1: Implement e-greedy to select the next move
        choice = self.epsilon_greedy(values, self.epsilon)

        return valid_moves[choice], valid_keys[choice]


class AgentPick(AgentBase):
    def __init__(self, rows, cols) -> None:
        AgentBase.__init__(self, rows, cols)
        self.id = None
        # Class identifier
        self.robot_class = 100
        # Class specific action
        self.action_type = 1
        # current position
        self.cur_pos = [0, 0]
        # current comms channel (ROBOT ID)
        self.comms_channel = None


class AgentPrune(AgentBase):
    def __init__(self, rows, cols) -> None:
        AgentBase.__init__(self, rows, cols)
        self.id = None
        # Class identifier
        self.robot_class = 200
        # Class specific action
        self.action_type = 2
        # current position
        self.cur_pos = [0, 0]
        # current comms channel (ROBOT ID)
        self.comms_channel = None