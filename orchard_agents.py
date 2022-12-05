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
        self.q_sa_table = np.zeros((rows, cols, 5))
        self.q_sa_table_3d = np.zeros((rows, cols, 5))
        self.learning_rate = 0.05
        self.epsilon = 1.0
        self.gamma = 0.9
        self.accumulated_reward = 0
        self.reward_evolution = []
        self.reward = 0
        self.key = ""
        self.move = []
        self.move_2 = []
        self.points = []
        self.vals = []
        self.valid_moves = []
        self.valid_keys = []
        self.previous_pose_step = 0
        self.previous_pose = []
        self.previous_previous_pose = []
        self.interactions = 0
        self.ineffective_steps = 1

        self.agents_keys = ["down", "down"]

    def reset_agent(self):
        self.epsilon = 1
        self.accumulated_reward = 0
        self.interactions = 0
        self.ineffective_steps = 0

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

    def update_epsilon(self, updater):
        self.epsilon = self.epsilon * updater

    def epsilon_greedy(self, values: list, epsilon: float):
        """
        Epsilon greedy policy. It allows exploration depending on the value of epsilon, otherwise
        it exploits by choosing the max value
        :param values: List of Qtable values to choose from
        :param epsilon: Parameter to balance exploration and exploitation
        :return:
        """

        # Generate a random number from 0 to 1
        p = random.random()
        choices = len(values)

        if p < epsilon:
            # Encourage the agent to Explore!
            action = random.randrange(0, choices)  # outputs random from 0,1,2 or 3
        else:
            # Otherwise explore
            action = np.argmax(values)

        return action

    def qlearning_update_value(self, state: list, s_prime: list, reward: int, other_agent_action = 0):
        """
        Updates Qtable using Qlearning algorithm, which is TD using the max value of the state of the next action
        :param state:           Next state (S)
        :param s_prime:     State prime (S'), after taking the best action
        :param reward:           Reward (R) of state (S)
        :return:
        """
        # --- Q learning algorithm ---
        # Q(S,A) <-- Q(S,A) + alpha * [R + gamma * max Q(S',a) - Q(S,A)]
        current_value = self.q_sa_table[state[0]][state[1]][other_agent_action]  # current Q(S,A)
        prime_value = self.q_sa_table[s_prime[0]][s_prime[1]][other_agent_action]  # maxQ(S',a)
        next_value = current_value + self.learning_rate * (reward + self.gamma * prime_value - current_value)

        # Update value in Q_sa_table
        self.q_sa_table[state[0]][state[1]][other_agent_action] = next_value  # update Q(S,A)

    def choose_move_egreedy(self, observed_points, observed_vals, valid_moves, valid_keys, other_agent_action =0):
        """ Chooses next move from a list (valid_moves) following an epsilon greedy policy
        """

        # --- Step 0: Map valid moves to values from Q_sa_table
        values = []
        for i in range(len(valid_moves)):
            see = valid_moves[i]
            see_row = see[0]
            see_col = see[1]
            values.append(self.q_sa_table[see_row][see_col][other_agent_action])

        # --- Step 1: Implement e-greedy to select the next move
        choice = self.epsilon_greedy(values, self.epsilon)

        return valid_moves[choice], valid_keys[choice]

    def choose_move_max(self, observed_points, observed_vals, valid_moves, valid_keys, other_agent_action = 0):
        """Choose max value from a list (valid_moves).
        This is required for Qlearning"""

        # --- Step 0: Map valid moves to values from Q_sa_table
        values = []
        for i in range(len(valid_moves)):
            see = valid_moves[i]
            see_row = see[0]
            see_col = see[1]
            values.append(self.q_sa_table[see_row][see_col][other_agent_action])

        # --- Step 1: Simply choose max to select the next move
        choice = np.argmax(values)

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
