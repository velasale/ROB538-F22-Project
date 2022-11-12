import numpy as np


class AgentBase():
    def __init__(self) -> None:
        self.id = None
        # Class identifier
        self.robot_class = 0
        # Class specific action
        self.action_type = 0
        # current position
        self.cur_pos = [0, 0]
        # current comms channel (ROBOT ID)
        self.comms_channel = None

    def reset_agent(self):
        # TODO: Used to reset the agent after each episode
        pass

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

    def choose_move(self, observed_points, observed_vals, valid_moves, valid_keys):
        # TODO: Fill out for classes RL stuff
        # randomly chooses a valid move
        # points is a list of the observed points, observed vals is a list of corresponding id of each x,y
        # returns the [x,y] of next move and the key: up, down, left, right or interact
        return None, None


class AgentPick(AgentBase):
    def __init__(self) -> None:
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
    def __init__(self) -> None:
        self.id = None
        # Class identifier
        self.robot_class = 200
        # Class specific action
        self.action_type = 2
        # current position
        self.cur_pos = [0, 0]
        # current comms channel (ROBOT ID)
        self.comms_channel = None
